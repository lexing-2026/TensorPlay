"""Deciding what runs together, and in what order.

Lowering turns a program into a list of results, each written by one piece of
work, each read by others.  What this module decides is which of those pieces
share a kernel, and what order the kernels run in, so that nothing reads a
value before it is written and as little memory as possible is held at once.

Two things are being balanced.  Running pieces together means fewer launches
and more reuse of values that are already in hand, but only where what one
reads and what the other writes line up.  Choosing an order means holding less
memory, but only where nothing is waiting on something that has not run.
"""

from __future__ import annotations

import collections
import enum
from collections import defaultdict
import dataclasses
import functools
import itertools
import logging
import math
import pprint
import sys
import textwrap
import typing
from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import sympy

import tensorplay as tp
from tensorplay.utils import _pytree as pytree

from . import choices, config, dependencies, ir
from .dependencies import MemoryDep
from .runtime.hints import ReductionHint
from .codegen.common import BackendFeature, IndentedBuffer, KernelArgs
from .ir import (
    Buffer,
    CommBufferLayout,
    ComputedBuffer,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
    OrderedSet,
    Pointwise,
    Reduction,
)
from . import metrics
from .loops import LoopBody, V
from .sizevars import SimplifyIndexing
from .utils import sympy_index_symbol, sympy_subs
from ....graph.experimental.sympy_functions import (
    FloorDiv,
    Identity,
    SymT,
    is_power_of_2,
    symbol_is_type,
)
from .memory import (
    FreeableInputBuffer,
    MemoryPlanningInfoForBuffer,
    MemoryPlanningInfoForNode,
)
from .ir import is_gpu
from .utils import (
    cache_on_self,
    cache_on_self_and_args,
    get_current_backend,
    get_dtype_size,
    sympy_product,
)

log = logging.getLogger(__name__)

fusion_log = logging.getLogger(f"{__name__}.fusion")
loop_ordering_log = logging.getLogger(f"{__name__}.loop_ordering")
compute_dependencies_log = logging.getLogger(f"{__name__}.compute_dependencies")

#: Where measured runtimes are kept between compilations, and what this
#: compilation did, counted by name.
counters: dict = defaultdict(float)

_TILING_MEMORY_MISS = float("inf")

_REINDEXING_FUSION_LAUNCH_OVERHEAD_NS = 1_000


def _real_dep_names(deps) -> OrderedSet:
    """The names that are really read or written, not merely ordered against.

    A dependency that exists only to say an order does not need the memory to
    stay alive, so it is left out of the answer.
    """

    return OrderedSet(dep.name for dep in deps if not isinstance(dep, dependencies.WeakDep))


def is_multi_outputs_template(node) -> bool:
    """Whether this result is several values that a template produced together."""

    return isinstance(node, ir.TemplateBuffer) and node.is_multi_outputs_template()


def get_device_type(obj):
    """The kind of device this runs on, or nothing if that is not known."""

    if isinstance(obj, (ir.IRNode, ir.BaseView)):
        if obj.get_device():
            return obj.get_device().type
    elif isinstance(obj, (ir.Buffer, ir.TensorBox)):
        dev = obj.get_device()
        if dev:
            return dev.type
    return None


def get_current_node():
    """The node being processed, or nothing when none is."""

    return V.current_node


def get_estimate_runtime_cache():
    """Where measured runtimes are kept between compilations."""

    from .codecache import LocalCache

    return LocalCache()


def get_estimate_runtime_cache_key_from_snode(snode) -> str:
    """What identifies this piece of work, for a measured runtime to be filed under.

    Only the shapes take part: two calls with the same shapes do the same amount
    of work whatever the values are, so what was measured for one says something
    about the other.
    """

    python_kernel_name = getattr(snode.node, "python_kernel_name", "")
    args = snode.node.inputs
    args = snode.node.fill_non_provided_args(
        [*args, *snode.node.constant_args],
        snode.node.kwargs,
    )
    kwargs = snode.node.kwargs
    flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

    def _is_tensor_ir(x) -> bool:
        return isinstance(x, ir.IRNode) and not isinstance(
            x, (ir.GeneratorState, ir.OpaqueObjectState)
        )

    cache_key = str(
        (python_kernel_name,)
        + tuple(tuple(a.get_size()) if _is_tensor_ir(a) else None for a in flat_args)
    )
    return cache_key


def _get_benchmarkable_extern_fn(snode):
    """The function to run in order to time this, if there is one to run."""

    if not isinstance(snode, ExternKernelSchedulerNode):
        return None
    if not isinstance(snode.node, ir.ExternKernel):
        return None

    op_overload = snode.node.op_overload
    op = getattr(op_overload, "overloadpacket", None)
    if op is None:
        return None

    from tensorplay.utils.flop_counter import flop_registry

    if op not in flop_registry:
        return None

    return op


def maybe_estimate_runtime_benchmark(snode):
    """How long this takes, measured, or nothing if it was not measured."""

    if not config.runtime_estimations_mms_benchmark:
        return None
    bench_fn = _get_benchmarkable_extern_fn(snode)
    if bench_fn is None:
        return None

    from .utils import snode_args_kwargs

    args_kwargs_fn = lambda: snode_args_kwargs(snode)  # noqa: E731

    cache_key = get_estimate_runtime_cache_key_from_snode(snode)
    cache = get_estimate_runtime_cache()
    cache_val = cache.lookup(cache_key)
    if cache_val is not None:
        if not isinstance(cache_val, float):
            raise AssertionError("expected cache_val to be a float")
        return cache_val

    args, kwargs = args_kwargs_fn()
    from .runtime.benchmarking import benchmarker

    ms = benchmarker.benchmark(
        bench_fn,
        args,
        kwargs,
        memory_warmup_iters=5,
        benchmark_iters=10,
        max_benchmark_duration=10,
    )

    cache.set_value(cache_key, value=ms)
    return ms


def pformat(obj) -> str:
    """A value written out, in a form that is meant to be read."""

    if isinstance(obj, (OrderedSet, set)):
        # A set has no order of its own, and one of expressions is unreadable
        # in whatever order it happens to come out.
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' ' * 4)}"
    return result


@dataclasses.dataclass
class SchedulerBuffer:
    """One buffer, and which piece of work wrote it and which read it."""

    scheduler: Any
    node: ir.Buffer
    defining_op: Any = None
    users: list = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def defining_op_name(self) -> str:
        op = self.defining_op
        if op is None:
            raise AssertionError("expected op to be set")
        return op.get_name()

    def is_ordering_only(self) -> bool:
        return getattr(self.node, "ordering_only", False)

    def __hash__(self) -> int:
        return hash(self.node.name)

    def debug_str(self) -> str:
        result = IndentedBuffer()
        name = self.get_name()
        result.writeline(f"{name}: {type(self.node).__name__}")
        result.writeline(f"{name}.layout = {getattr(self.node, 'layout', None)}")
        if self.get_aliases():
            result.writeline(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            result.writeline(f"{name}.mutations = {pformat(self.get_mutations())}")

        if len(self.users) <= 1:
            result.writeline(f"{name}.users = {self.users}")
        else:
            result.writeline(f"{name}.users = [")
            with result.indent(1):
                for user in self.users:
                    result.writeline(f"{user},")
            result.writeline("]")
        return result.getrawvalue()

    def get_name(self) -> str:
        return self.node.get_name()

    def allocate(self) -> None:
        """Hand this buffer the memory it needs, if it needs any of its own.

        A buffer that is somebody else's memory, or that names another value as
        being written over, is not given memory here: it was given elsewhere.
        """

        if self.node is None:
            raise AssertionError("expected self.node to be set")
        if not self.node.should_allocate():
            return

        if (
            self.node.get_inputs_that_alias_output()
            or self.node.get_mutation_names()
            or isinstance(self.node.get_output_spec(), CommBufferLayout)
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # Whether there is a kernel being written at all, as opposed to a
        # stand-in for one.
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            input_buffer_name = V.kernel.inplace_update_buffers[self.get_name()]
            if input_buffer_name in self.scheduler.name_to_donated_buffer:
                input_buffer = self.scheduler.name_to_donated_buffer[
                    input_buffer_name
                ].node
            else:
                input_buffer = self.scheduler.name_to_buf[input_buffer_name].node
            V.graph.wrapper_code.codegen_inplace_reuse(
                input_buffer,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self) -> bool:
        """Whether the memory behind this can be given back once it is read.

        A buffer with no shape holds no memory of its own, and one that is
        several results is freed when its results are, so neither is freed
        directly.  A buffer the program is handing back cannot be either.
        """

        if self.node is None:
            raise AssertionError("expected self.node to be set")
        if isinstance(self.node.layout, NoneLayout) or is_multi_outputs_template(
            self.node
        ):
            return False
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def set_users(self, users: list) -> None:
        """Record who reads this, one entry per piece of work that does.

        Two uses by the same piece of work are one use, and are merged so that
        whether that work is the last reader is decided once and for both.
        """

        result: dict = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self) -> Sequence:
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_inputs_that_alias_output()

    def get_mutations(self) -> Sequence:
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_mutation_names()

    def get_device(self):
        return self.node.get_output_spec().get_device()


@dataclasses.dataclass
class SchedulerDonatedBuffer(SchedulerBuffer):
    """A buffer the program handed over, which may be written into if it is done with.

    A buffer that came in from the surrounding program is not this program's to
    write into as a matter of course, but a program may say it is finished with
    one, in which case its memory can be used for something else.
    """

    defining_op: Any = None


class BaseSchedulerNode:
    """One piece of work, and what it reads, writes and waits for.

    This is what every kind of piece of work shares: a piece that computes
    something, a piece that is a call out to something already written, and a
    group of those put together.  What differs is only what they can be asked
    about themselves, which is what the subclasses answer differently.
    """

    ancestors: OrderedSet
    group: tuple
    last_usage: OrderedSet
    #: Where this sits in the original order.  For a group of pieces, the
    #: earliest and latest of its members; for a single one, both the same.
    min_input_distance: int
    max_input_distance: int
    min_order: int
    max_order: int
    mpi_node: MemoryPlanningInfoForNode
    mutation_renames: dict
    node: Any = None
    outputs: list
    outputs_by_name: dict
    override_estimated_runtime: float | None = None
    read_writes: Any
    unmet_dependencies: OrderedSet
    written: bool = False

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        self.debug_device_str: Callable = lambda *args, **kwargs: []

    def _init_from_node(self, node) -> None:
        self.node = node
        self.ancestors = OrderedSet()
        self.min_input_distance = 0
        self.max_input_distance = 0
        self.last_usage = OrderedSet()
        self.written = False
        self.outputs = [
            SchedulerBuffer(
                scheduler=self.scheduler,
                node=output,
                defining_op=self,
            )
            for output in node.get_outputs()
        ]
        self.outputs_by_name = {buf.get_name(): buf for buf in self.outputs}

        # Which of this piece's buffers were renamed by a write over them.  This
        # is only what matters for this piece's own dependencies, so it is
        # smaller than the record for the whole graph.
        self.mutation_renames = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self) -> str:
        """A longer account of this piece, for when something is not working."""

        name = self.get_name()
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({type(getattr(self, "node", None)).__name__})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.min_input_distance = {self.min_input_distance}
{name}.max_input_distance = {self.max_input_distance}
{name}.outputs = [
        """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    def debug_str_extra(self) -> str:
        return ""

    def _debug_str_for_device(self) -> list:
        return self.debug_device_str(self)

    def debug_str_short(self) -> str:
        maybe_data = getattr(self.node, "data", None)
        data_str = ""
        if isinstance(maybe_data, Pointwise):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_size()], shorten=False, multiline=False
            )
        elif isinstance(maybe_data, Reduction):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_reduction_size(), maybe_data.get_reduction_type()],
                shorten=False,
                multiline=False,
            )
        return f"{self}{data_str}"

    def log_details(self) -> None:
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def reorder_loops_by_dep_pair(self, self_dep, other_dep) -> bool:
        return False

    def update_mutated_names(self, renames: dict) -> None:
        """Point this piece's dependencies at the names that survived a write over."""

        self.mutation_renames = {
            name: renames[name]
            for name in (dep.name for dep in self.read_writes.reads_and_writes())
            if name in renames
        }
        self.set_read_writes(self.read_writes.rename(self.mutation_renames))

    def add_fake_dep(self, dep) -> None:
        self.set_read_writes(self.read_writes.with_read(dep))

    def has_aliasing_or_mutation(self) -> bool:
        return any(
            buf.get_aliases() or buf.get_mutations() for buf in self.get_outputs()
        )

    def set_read_writes(self, rw) -> None:
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.clear_read_writes_dependent_caches()
        self.prune_deps()

    def clear_read_writes_dependent_caches(self) -> None:
        self.get_coalesce_analysis.clear_cache(self)
        self.get_tiling.clear_cache(self)

    @cache_on_self
    def get_coalesce_analysis(self):
        """Whether this piece's reads of each buffer would be done efficiently.

        Adjacent elements read together is what makes a read efficient, so this
        looks at the index expressions and says how well they line up.  Only a
        piece whose reads can be described at all has an answer.
        """

        from .tiling_utils import _analyze_memory_coalescing

        if not isinstance(self, (SchedulerNode, FusedSchedulerNode)):
            return None
        return _analyze_memory_coalescing(self)

    @cache_on_self_and_args("BaseSchedulerNode")
    def get_tiling(self, numel, rnumel) -> dict:
        """How to divide this work among programs, given how much of it there is.

        How much each program does is a choice about the machine rather than
        about what is being computed, so it is put to the backend -- which is
        what knows the machine -- and given what is being computed.  Where the
        division is decided later, from the body rather than from here, there is
        nothing to return and that is what raising says.
        """

        device = self.get_device()
        backend = self.scheduler.get_backend(device)
        select = getattr(backend, "select_tiling", None)
        if select is None:
            raise NotImplementedError(
                f"{type(backend).__name__} does not choose a tiling ahead of codegen"
            )
        return select(self.get_nodes(), numel, rnumel)

    def set_last_usage(self, future_used_buffers, mutation_real_name) -> None:
        """Note which of the buffers this uses, nothing will read after this."""

        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
        self.last_usage = used_buffers - future_used_buffers

    def mark_run(self) -> None:
        for buf in self.outputs:
            buf.allocate()

    def used_buffer_names(self) -> OrderedSet:
        return OrderedSet(
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        )

    def used_or_aliased_buffer_names(self) -> OrderedSet:
        """Every name this touches, including through values that share memory.

        A dependency that exists only to say an order is left out, since it
        keeps nothing alive.
        """

        used_names: OrderedSet = OrderedSet()

        deps = [
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
            if not (isinstance(dep, dependencies.WeakDep) and dep.is_fake)
        ]
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            if V.graph.name_to_buffer.get(dep):
                deps.extend(
                    alias
                    for alias in V.graph.name_to_buffer[
                        dep
                    ].get_inputs_that_alias_output()
                    if alias not in used_names
                )
        return used_names

    def prune_deps(self) -> None:
        self.unmet_dependencies = OrderedSet(
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        )

    def prune_weak_deps(self) -> None:
        """Forget the order-only dependencies on pieces that are no longer there."""

        def should_prune(dep) -> bool:
            if not isinstance(dep, dependencies.WeakDep):
                return False
            if dep.name not in self.scheduler.name_to_buf:
                return False

            op_name = self.scheduler.name_to_buf[dep.name].defining_op_name()
            return op_name in V.graph.removed_operations

        to_remove = OrderedSet(
            dep for dep in self.read_writes.reads if should_prune(dep)
        )
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(self, name_to_fused_node) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

    def get_name(self) -> str:
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_operation_name()

    def get_first_name(self) -> str:
        return self.get_name()

    @cache_on_self
    def get_operation_names(self) -> OrderedSet:
        return OrderedSet(node.get_name() for node in self.get_nodes())

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet:
        return OrderedSet(out.get_name() for out in self.outputs)

    @cache_on_self
    def can_codegen_in_low_precision(self) -> bool:
        return all(
            isinstance(n, SchedulerNode)
            and can_codegen_without_upcasts(n, disallow_fp32_ops=True)
            for n in self.get_nodes()
        )

    @cache_on_self
    def can_codegen_without_upcasts(self) -> bool:
        return all(
            isinstance(n, SchedulerNode) and can_codegen_without_upcasts(n)
            for n in self.get_nodes()
        )

    def get_nodes(self) -> Sequence:
        return [self]

    @cache_on_self
    def has_strict_reduction(self) -> bool:
        return any(
            isinstance(node, SchedulerNode)
            and isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, ir.Reduction)
            and node.node.data.strict_reduction_rblock is not None
            for node in self.get_nodes()
        )

    def get_outputs(self) -> Sequence:
        return self.outputs

    def get_output(self, buf_name: str):
        return self.outputs_by_name[buf_name]

    def get_device(self):
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_device()

    def is_cpu(self) -> bool:
        device = self.get_device()
        return device is not None and device.type == "cpu"

    def is_gpu(self) -> bool:
        device = self.get_device()
        return device is not None and is_gpu(device)

    def is_reduction(self) -> bool:
        return False

    def is_native_matmul(self) -> bool:
        return False

    def is_split_scan(self) -> bool:
        return False

    def is_template(self) -> bool:
        return False

    def is_extern(self) -> bool:
        return False

    def is_foreach(self) -> bool:
        return False

    def can_inplace(self, read_dep) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return False

    def decide_inplace_update(self) -> None:
        """Decide whether this piece may write into a buffer it was given.

        Where it may, the buffer it was given becomes the buffer it writes, and
        nothing is copied.  What decides it is that exactly one thing is left to
        read that buffer, that thing is this piece, and the two fit together the
        same way -- since writing somewhere else would mean a different shape or
        a different arrangement, and the value would no longer be the same one.
        """

        from .codegen.wrapper import can_match_buffer_size

        if not (
            isinstance(self, SchedulerNode)
            and config.inplace_buffers
            and V.graph.has_feature(self.get_device(), BackendFeature.INPLACE_BUFFERS)
            and hasattr(V.kernel, "args")
        ):
            return

        inconsequential_nodes = (
            self.ancestors
            | V.graph.removed_operations
            | self.scheduler.completed_operations
        )

        def single_index_in_fused_node(buf_to_be_inplaced) -> bool:
            # Whether the read and the write name the same elements is tracked
            # per use, but once uses are folded together the other accesses in
            # the same group have to agree as well.
            fused_node = buf_to_be_inplaced.scheduler.get_fused_node(self)
            buf_name = buf_to_be_inplaced.get_name()
            deps: OrderedSet = OrderedSet()
            for user in buf_to_be_inplaced.users:
                user_node = user.node
                if not isinstance(user_node, BaseSchedulerNode):
                    continue

                if (
                    user_node.get_first_name()
                    not in buf_to_be_inplaced.scheduler.name_to_fused_node
                    or buf_to_be_inplaced.scheduler.get_fused_node(user_node)
                    is not fused_node
                ):
                    continue

                deps |= (
                    o
                    for o in user_node.read_writes.reads_and_writes()
                    if o.name == buf_name
                )
                if len(deps) > 1:
                    return False

            return True

        for buf in self.get_outputs():
            buf_node = buf.node
            if buf_node is None:
                raise AssertionError("expected buf_node to be set")
            if (
                not buf_node.should_allocate()
                or buf_node.get_inputs_that_alias_output()
                or buf_node.get_mutation_names()
                or buf.get_name() in V.graph.removed_buffers
                or isinstance(buf_node.get_output_spec(), CommBufferLayout)
            ):
                continue

            for read in self.read_writes.reads:
                input_buf = None
                if read.name in self.scheduler.name_to_donated_buffer:
                    input_buf = self.scheduler.name_to_donated_buffer[read.name]
                else:
                    input_buf = self.scheduler.name_to_buf.get(read.name)

                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, self)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    if input_buf.users is None:
                        raise AssertionError("expected input_buf.users to be set")
                    remaining_uses = [
                        x
                        for x in input_buf.users
                        if x.node.get_name() not in inconsequential_nodes
                    ]
                    has_cross_stream_hazard = self.scheduler.has_cross_stream_hazard(
                        read.name, self
                    )
                    has_cross_mempool_hazard = self.scheduler.get_buf_mempool(
                        read.name
                    ) != self.scheduler.node_to_mempool.get(self)

                    if (
                        not has_cross_stream_hazard
                        and not has_cross_mempool_hazard
                        and len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and input_buf.node is not None
                        and not isinstance(
                            input_buf.node.get_output_spec(),
                            (
                                NoneLayout,
                                MultiOutputLayout,
                                ir.MutationLayoutSHOULDREMOVE,
                                CommBufferLayout,
                            ),
                        )
                        and not (
                            input_buf.defining_op
                            and isinstance(
                                input_buf.defining_op.node,
                                (ir.FallbackKernel, MultiOutput),
                            )
                            and len(input_buf.node.get_inputs_that_alias_output()) > 0
                        )
                        and can_match_buffer_size(input_buf.node, buf.node)
                        and single_index_in_fused_node(input_buf)
                    ):
                        V.kernel.args.make_inplace(input_buf.get_name(), buf.get_name())
                        mutations = getattr(V.kernel, "mutations", None)
                        if mutations is not None:
                            mutations.add(input_buf.get_name())
                            mutations.add(buf.get_name())

                        V.kernel.inplace_update_buffers[buf.get_name()] = (
                            input_buf.get_name()
                        )
                        break

    def codegen_originating_info(self, buffer, only_once: bool = True) -> None:
        """Say in the code where this piece came from, once.

        Where the value came from is what makes generated code readable to
        whoever has to work out why it does what it does, but saying it for
        every piece in a group is noise, so it is said once per kernel.
        """

        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        origins = self.node.get_origins()
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are all the same and say nothing.
                continue

            out_lines.append("")
            out_lines.append("#pragma CMT ORIGIN:")
            op_info_str = f"#pragma CMT {o.op} {o.target}"
            if "seq_nr" in o.meta:
                op_info_str = op_info_str + f" seq_nr:{o.meta['seq_nr']}"
            out_lines.append(op_info_str)
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.rsplit("|", maxsplit=1)[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                    .replace("\\", "\\\\")
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        buffer.writelines(out_lines)
        self.written = True

    @cache_on_self
    def get_read_write_buffers_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=True
        )

    @cache_on_self
    def get_read_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=False
        )

    @cache_on_self
    def get_write_buffer_sizes(self) -> int:
        return self.get_read_write_buffers_sizes_impl(
            include_reads=False, include_writes=True
        )

    def get_read_write_buffers_sizes_impl(
        self, include_reads: bool, include_writes: bool
    ) -> int:
        return sum(
            self.get_read_write_buffer_accesses(
                include_reads=include_reads, include_writes=include_writes
            ).values(),
            start=0,
        )

    def get_read_write_buffer_accesses(
        self, include_reads: bool, include_writes: bool
    ) -> dict:
        """How many bytes this piece touches in each buffer.

        What a piece touches and what it needs to touch are two different
        numbers.  A piece may read the same input more than once, or read only
        part of a buffer, or read elements whose positions are not known until
        the data is there.  So for each buffer two counts are worked out -- how
        many elements the piece's own shape implies it touches, and how many the
        buffer holds -- and the smaller is taken, which is the most it could
        need to read and the least an optimization would leave it needing to.
        """

        if isinstance(self, NopKernelSchedulerNode):
            return {}
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(
            self.node, MultiOutput
        ):
            return {}
        if (
            isinstance(self, ExternKernelSchedulerNode)
            and isinstance(self.node, ir.FallbackKernel)
            and self.node.op_overload
            is getattr(tp.ops.prims.rng_prims, "graphsafe_run_with_rng_state", None)
        ):
            return {}

        def try_size_hint(s) -> int:
            return V.graph.sizevars.optimization_hint(s, fallback=0)

        if isinstance(self, SchedulerNode):
            node_numel = try_size_hint(
                sympy_product(self.get_ranges()[0])
                * sympy_product(self.get_ranges()[1]),
            )
        else:
            node_numel = int(1e9)
        buf_accesses = collections.defaultdict(list)

        if include_reads:
            for dep in self.read_writes.reads:
                buf_accesses[dep.name].append(dep)

        if include_writes:
            for dep in self.read_writes.writes:
                buf_accesses[dep.name].append(dep)

        reads = (
            OrderedSet(dep.name for dep in self.read_writes.reads)
            if include_reads
            else OrderedSet()
        )
        writes = (
            OrderedSet(dep.name for dep in self.read_writes.writes)
            if include_writes
            else OrderedSet()
        )

        def is_materialized(buf: str, snodes: Sequence) -> bool:
            users = self.scheduler.name_to_buf[buf].users
            buf_uses = OrderedSet(user.node for user in users)
            return len(buf_uses - OrderedSet(snodes)) > 0

        if isinstance(self, FusedSchedulerNode):
            removed_buffers = OrderedSet(
                dep for dep in writes if not is_materialized(dep, self.snodes)
            )
            writes = writes - removed_buffers
            reads = reads - removed_buffers

        buf_byte_accesses: dict = {}

        for buf_name in reads | writes:
            buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_bytes(buf) -> int:
                if not buf:
                    return 0

                if isinstance(buf, ir.TorchBindObject):
                    return buf.get_buf_bytes()
                elif isinstance(buf.layout, MultiOutputLayout):
                    # The results are what have shapes, so they are what the
                    # size is worked out from.
                    users = self.scheduler.name_to_buf[buf.get_name()].users
                    tot = 0
                    for user in users:
                        if isinstance(user.node, OutputNode):
                            continue
                        if not isinstance(user.node, BaseSchedulerNode):
                            raise AssertionError(
                                "expected user.node to be a BaseSchedulerNode"
                            )
                        if isinstance(user.node.node, MultiOutput):
                            for sched_buf in user.node.get_outputs():
                                tot += get_buf_bytes(sched_buf.node)
                        else:
                            return 0
                    return tot
                elif isinstance(buf.layout, NoneLayout):
                    return sum(
                        get_buf_bytes(V.graph.get_buffer(mut_name))
                        for mut_name in buf.get_mutation_names()
                    )
                else:
                    buf_elems = try_size_hint(sympy_product(buf.get_size()))
                    return get_dtype_size(buf.get_dtype()) * min(
                        buf_accessed_elems, buf_elems
                    )

            buf_bytes = get_buf_bytes(buf)
            if buf_name not in buf_byte_accesses:
                buf_byte_accesses[buf_name] = buf_bytes
            else:
                buf_byte_accesses[buf_name] += buf_bytes

        return buf_byte_accesses

    @cache_on_self
    def estimate_flops(self):
        """How many arithmetic operations this does, if that can be worked out.

        Counting them means walking the program as it was written, which is the
        only place the count exists.  A piece that did not come from a program
        has no count, and neither has one whose operations are not countable.
        """

        if self.node is None:
            return None
        fx_node = self.node.get_origin_node()
        if fx_node is None:
            return None

        flops = count_flops_fx(fx_node)
        if flops is None:
            return None

        if isinstance(flops, tp.SymInt):
            flops = flops.node.expr

        resolved_flops = V.graph.sizevars.optimization_hint(flops, fallback=0)
        counters["inductor"]["flop_count"] += resolved_flops
        return resolved_flops

    def get_estimated_runtime(self) -> float:
        if self.override_estimated_runtime is not None:
            return self.override_estimated_runtime

        return self._get_estimated_runtime()

    @cache_on_self
    def _get_estimated_runtime(self) -> float:
        """How long this is expected to take, in milliseconds.

        Only worth working out where the answer could change what runs: off the
        device that computes it, there is nothing to reorder against.
        """

        buf = self.get_nodes()[0].get_outputs()[0]
        layout = buf.node.get_output_spec()
        if not is_gpu(get_device_type(layout)):
            return 0

        if isinstance(self.node, ir._CollectiveKernel):
            return estimate_collective_runtime(self.node)

        if isinstance(self.node, ir._WaitKernel):
            # The time the exchange takes was counted against the exchange
            # itself; waiting for it costs nothing beyond that.
            return 0

        ret = maybe_estimate_runtime_benchmark(self)
        if ret is not None:
            return ret

        dtype = buf.node.maybe_get_dtype()
        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
            if gpu_memory_bandwidth <= 0:
                raise AssertionError(
                    f"gpu_memory_bandwidth cannot be <= 0, but got {gpu_memory_bandwidth}"
                )
            if gpu_flops <= 0:
                raise AssertionError(f"gpu_flops cannot be <= 0, but got {gpu_flops}")
        except Exception:
            return 0

        flops_est = self.estimate_flops()

        if flops_est == 0 or flops_est is None:
            ns = self.get_read_write_buffers_sizes() / gpu_memory_bandwidth
            return ns / 1e6

        factor = 1.0
        counted_bytes = self.get_read_write_buffers_sizes()
        counted_bytes = 0 if counted_bytes is None else counted_bytes
        compute_time = (factor * flops_est / gpu_flops) * 1e9
        transfer_time = counted_bytes / gpu_memory_bandwidth

        ns = max(compute_time, transfer_time)
        return ns / 1e6

    def get_template_node(self):
        return None

    def get_template_node_or_throw(self):
        template = self.get_template_node()
        if template is None:
            raise AssertionError("expected template to be set")
        return template

    @staticmethod
    def get_prologue_template_epilogue(nodes: list):
        """Split a list of pieces around the template among them.

        What comes before the template is work that could be folded into it, and
        what comes after is work that consumes its result.
        """

        template_index = next(i for i, n in enumerate(nodes) if n.is_template())

        prologue = nodes[:template_index]
        template_node = nodes[template_index]
        epilogue = nodes[template_index + 1 :]
        return prologue, template_node, epilogue


@dataclasses.dataclass(slots=True)
class WhyNoFuse:
    """Why two pieces were not put together, in a form worth reading."""

    name1: str
    name2: str
    reason: str
    args: tuple

    def __init__(self, node1, node2) -> None:
        self.name1 = node1.get_name()
        self.name2 = node2.get_name()

    def __call__(self, reason: str, *args) -> None:
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self) -> str:
        return f"cannot fuse {self.name1} with {self.name2}: " + (
            self.reason % self.args
        )


class OutputNode:
    """The program handing a value back, which is the last thing to read it.

    A buffer that is a result of the program is never given away, since whoever
    receives it is outside anything that could free it.
    """

    def __init__(self, dep) -> None:
        self.unmet_dependencies = OrderedSet([dep])

    def is_reduction(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence:
        return ()

    def get_name(self) -> str:
        return "OUTPUT"

    __repr__ = get_name


class ExternKernelSchedulerNode(BaseSchedulerNode):
    """A piece of work that is a call out to something already written.

    What such a call reads and writes is stated by how the operation was
    declared rather than worked out from what it computes, which is why it is
    asked for rather than analysed.  It can also be told to write into memory it
    was given, in which case what it writes is a value of the shape it was given
    and the group it works in is the size of that.
    """

    def __init__(self, scheduler, node) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())

        if isinstance(node, ir.UserDefinedTritonKernel) and node.can_fuse_epilogue():
            numel = math.prod(node.mutable_args[0].shape)
            rnumel = 1
            device = node.get_device()
            self.group = (device, (numel, rnumel))

    def debug_str_extra(self) -> str:
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'python_kernel_name', None)}"

    def is_extern(self) -> bool:
        return True

    def has_side_effects(self) -> bool:
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()

    def get_ranges(self) -> Sequence:
        if (
            isinstance(self.node, ir.UserDefinedTritonKernel)
            and self.node.can_fuse_epilogue()
        ):
            numel = math.prod(self.node.mutable_args[0].shape)
            return ([numel], [])
        return ([], [])

    def codegen(self, wrapper) -> None:
        if not isinstance(self.node, ir.ExternKernel):
            raise AssertionError("expected self.node to be an ir.ExternKernel")
        return self.node.codegen(wrapper)


class NopKernelSchedulerNode(BaseSchedulerNode):
    """A piece of work that computes nothing, only arranges memory.

    A concatenation is like this: the values are laid end to end and each is
    written into its place, with no arithmetic anywhere.  There is still a
    dependency to honour -- something has to be there before it can be laid out
    -- which is why this exists rather than nothing at all.
    """

    def __init__(self, scheduler, node) -> None:
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())


@dataclasses.dataclass
class NodeUser:
    """One piece of work reading a buffer, and whether it may write into it."""

    node: Any
    can_inplace: bool = False

    #: A user that has to run after a given piece but does not read the result.
    #: Ordering without a read, so nothing is kept alive by it.
    is_weak: bool = False

    def __hash__(self) -> int:
        return hash((self.node.get_name(), self.can_inplace, self.is_weak))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, NodeUser)
            and self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak
        )

    def get_name(self) -> str:
        return self.node.get_name()

    def merge(self, other) -> "NodeUser":
        """Two uses by the same piece are one use, and may be written into only if both may."""

        if self.node is not other.node:
            raise AssertionError("expected self.node to be other.node")
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


def can_codegen_without_upcasts(snode, disallow_fp32_ops: bool = False) -> bool:
    """Whether this piece can be written without widening anything first.

    Widening to the widest type and narrowing back is what a program does when
    its arithmetic is done in one type and its results are kept in another.
    Doing that is not always allowed, so whether a piece needs it decides
    whether it may run alongside something that cannot have it.
    """

    if (
        config.emulate_precision_casts
        and not disallow_fp32_ops
        and snode.node is not None
        and isinstance(snode.node, ComputedBuffer)
    ):
        return False
    return True


def count_flops_fx(fx_node):
    """How many arithmetic operations the program did to produce this value.

    Counted from the program as it was written, since that is the only place
    the operations are stated.  A value that did not come from a program, or one
    whose operations cannot be counted, has no number.
    """

    from tensorplay.utils.flop_counter import count_flops_fx as _count

    try:
        return _count(fx_node)
    except Exception:
        return None


def estimate_collective_runtime(node) -> float:
    """How long an exchange with other machines is expected to take.

    What it costs depends on how much is being sent and how far it has to go,
    neither of which is known here, so where no better answer is available the
    exchange is not allowed to influence the order.
    """

    return 0


def get_gpu_dram_gbps() -> float:
    """How fast this device can read and write its memory, in gigabytes a second.

    Zero when it is not known.  A roofline estimate divides by this to decide
    whether a piece is bound by its memory or by its arithmetic, and a caller
    that gets zero falls back to what it can measure, which is a real time.
    Guessing a figure here would put a number in front of that decision that
    nothing measured, and the decision it drives is which kernels get fused.
    """

    return 0.0


def get_device_tflops(dtype) -> float:
    """How many arithmetic operations this device does per second, in a given type.

    Zero when it is not known, for the same reason as the memory bandwidth: a
    caller that gets zero estimates the cost from what it measures instead.
    """

    return 0.0


def _prune_redundant_deps(node, name_to_fused_node, name_to_buf) -> None:
    """Forget order-only dependencies that a fusion has made unnecessary.

    When two pieces are put together, a dependency that only said which of them
    came first no longer says anything -- they now run as one.  So an order-only
    dependency on a piece that has been folded into this one is dropped, and one
    is kept only where it still expresses an order that matters.
    """
    for buf, buf_node in name_to_buf.items():
        if buf_node.defining_op is None:
            continue
        if node.get_first_name() not in name_to_fused_node:
            continue
        if name_to_fused_node[node.get_first_name()] is not name_to_fused_node.get(
            buf_node.defining_op.get_first_name()
        ):
            continue

        for other in name_to_fused_node[node.get_first_name()].get_nodes():
            if not isinstance(other.unmet_dependencies, OrderedSet):
                continue
            other.unmet_dependencies = OrderedSet(
                dep
                for dep in other.unmet_dependencies
                if not (isinstance(dep, dependencies.WeakDep) and dep.name == buf)
            )
            other.read_writes = other.read_writes.remove_reads(
                OrderedSet(
                    dep
                    for dep in other.read_writes.reads
                    if isinstance(dep, dependencies.WeakDep) and dep.name == buf
                )
            )






def topo_sort_snode(snode) -> None:
    """Put the pieces of a group in the order they have to run.

    What each piece reads has to have been written, so the order follows the
    dependencies between the pieces rather than the order they were made in.
    """

    order = list(snode.get_nodes())
    while True:
        changed = False
        for i, node in enumerate(order):
            for dep in node.unmet_dependencies:
                for j in range(len(order)):
                    if order[j].get_first_name() == dep.name and i > j:
                        order[i], order[j] = order[j], order[i]
                        changed = True
        if not changed:
            break
    snode.snodes = order


def _cmp(left, right) -> int:
    """Three-way comparison of two values, as a sorting key needs it."""

    return (left > right) - (left < right)


def pick_loop_order(
    stride_lengths: list[list[int]],
    sizes: Sequence,
    priority_idx: Sequence[int] = (),
) -> list[int]:
    """Which axis to walk at each step, given how the memory is laid out.

    Walking in the order memory is arranged is what makes each step read what
    the step before left in hand.  An axis of extent one is not worth walking
    for its own sake -- it reads the same element every time -- so those go last
    whatever their stride says.

    The strides are compared as absolute values, because an axis walked
    backwards still reads what the axis before it left in hand; without that, a
    flipped axis looks more contiguous than a forward one and is preferred for
    being so.

    ``priority_idx`` names the axes a caller has already decided must come
    first, and only those are considered: a caller's ordering is not something
    the heuristic is entitled to overrule.
    """

    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        if sizes[a] == 1 or sizes[b] == 1:
            # An axis of extent one does not matter where it goes, so it goes
            # last; among two such axes the order is by index, so that the sort
            # is stable however the extents compare.
            return _cmp(sizes[a] == 1, sizes[b] == 1)

        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]

        # How many of the buffers each axis is the better-strided one in.  An
        # axis whose stride is zero in some buffer is as good as the other in
        # that buffer, because every element of it is the same element.
        a_first = sum(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = sum(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1

        # Nothing separates them, so the one walked from the inside out comes
        # first, which is the contiguous one.
        return _cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # Only the axes the caller named are considered.
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


def get_layout_symints(buf, consumers=None) -> OrderedSet:
    """The shapes a buffer's layout is written in terms of.

    The consumers are not consulted: every reader of a value has to be given
    the shapes that value's layout is written in terms of, so the answer is the
    same whoever is asking.  The parameter is kept because the callers pass it.
    """

    from .utils import get_layout_symints as _impl

    return _impl(buf)


def get_scheduler_node_symbol_uses(snode, is_input: bool) -> OrderedSet:
    """The shapes a piece needs handed to it, rather than worked out."""

    if is_input:
        return get_layout_symints(
            snode.get_outputs(), sorted(snode.unmet_dependencies, key=id)
        )
    return OrderedSet()


def _is_epilogue_fusion_enabled() -> bool:
    return config.epilogue_fusion


def _is_prologue_fusion_enabled() -> bool:
    return config.prologue_fusion


def is_epilogue_fusion(
    consumer, producer, consumer_fusion, producer_fusion
) -> bool:
    """Whether one piece may be folded into the tail of another."""

    return (
        _is_epilogue_fusion_enabled()
        and consumer_fusion
        and producer_fusion
        and not producer.is_template()
    )


def is_prologue_fusion(
    consumer, producer, consumer_fusion, producer_fusion
) -> bool:
    """Whether one piece may be folded into the start of another."""

    return (
        _is_prologue_fusion_enabled()
        and consumer_fusion
        and producer_fusion
        and not producer.is_template()
    )


def is_template_fusion(scheduler, node) -> bool:
    """Whether a template is willing to have this piece folded into it."""

    if not isinstance(node, SchedulerNode):
        return False
    if node.is_template():
        return True
    return False


def template_fusion_pw_node(node) -> bool:
    """Whether this piece is a template's result with work on top of it."""

    return isinstance(getattr(node, "node", None), ir.Pointwise)


def _iter_loop_state_nodes(
    nodes,
):
    """Walk the pieces a group stands for, deepest first.

    A group is not one piece but the pieces joined into it, and a group may
    itself stand for groups, so this walks down to the leaves and then names
    each level on the way back up.  Leaves come before the group holding them,
    so a caller taking note of state gets the inner state before the outer
    state that contains it.
    """

    for node in nodes:
        if isinstance(node, FusedSchedulerNode):
            yield from _iter_loop_state_nodes(node.snodes)
            yield node
        elif isinstance(node, SchedulerNode):
            yield node



class SchedulerNode(BaseSchedulerNode):
    """One piece of work that computes something, and the loops it computes it with.

    What it is and what it reads is not asked of the operation: the operation is
    asked to describe itself as loops over a set of axes, and what it reads and
    writes follows from walking those loops.  The group it works in -- which is
    what decides whether it may run alongside another -- is worked out from
    those same axes.
    """

    _sizes: tuple
    _body: Any

    def __init__(self, scheduler, node) -> None:
        super().__init__(scheduler)
        self._loop_mutation_listener: Callable | None = None
        self._loop_state_gen = 0
        self._init_from_node(node)
        self._compute_attrs()

    def _compute_attrs(
        self,
        extra_indexing_constraints=None,
        recompute_sizes_body_func=None,
    ) -> None:
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )
        self._sizes, body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )
        self._body = body

        device = self.node.get_device()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Left alone deliberately: simplifying would join the loops together,
        # and then deciding what order to walk them in becomes much harder.
        should_normalize = not config.loop_ordering_after_fusion or not is_gpu(device)

        if isinstance(self.node, ir.TemplateBuffer):
            self.set_read_writes(
                self.node.extract_read_writes(normalize=should_normalize)
            )
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=should_normalize
                )
            )

    def recompute_size_and_body(
        self,
        extra_indexing_constraints=None,
        recompute_sizes_body_func=None,
    ) -> None:
        """Work out the loops again, keeping what was added by hand.

        An order that was only there to say which piece runs first is not
        something walking the loops would find, so it is put back afterwards.
        """

        fake_deps: OrderedSet = OrderedSet(
            dep
            for dep in self.read_writes.reads
            if isinstance(dep, (dependencies.WeakDep, dependencies.StarDep))
        )
        self._compute_attrs(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )
        if fake_deps or self.mutation_renames:
            read_writes = self.read_writes
            if fake_deps:
                read_writes = read_writes.with_read(fake_deps)
            self.set_read_writes(read_writes.rename(self.mutation_renames))

    def refresh_dependencies(
        self, normalize: bool, need_clear_tiling_cache: bool
    ) -> None:
        """Work out again what this reads and writes, after the loops changed.

        The orders that only say which piece runs first are added by hand and
        would not be found by walking the loops, so they are carried over.
        """

        fake_deps: OrderedSet = OrderedSet(
            dep
            for dep in self.read_writes.reads
            if isinstance(dep, (dependencies.WeakDep, dependencies.StarDep))
        )

        # Left unnormalized, since the loops may still be reordered after this.
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=normalize
            )
            .with_read(fake_deps)
            .rename(self.mutation_renames)
        )

        self.clear_loop_body_dependent_caches(need_clear_tiling_cache)

    def clear_loop_body_dependent_caches(self, need_clear_tiling_cache: bool) -> None:
        self.clear_read_writes_dependent_caches()
        self.pointwise_read_writes.clear_cache(self)

        if need_clear_tiling_cache:

            _clear_candidate_tilings()

    def snapshot_loop_state(self) -> tuple:
        """What the loops looked like, so that a change can be undone.

        Has to stay in step with the methods that change the loops.
        """

        return (
            self._body,
            self._sizes,
            self.group,
            self.read_writes,
            self.unmet_dependencies,
            self._loop_state_gen,
        )

    def restore_loop_state(self, state: tuple) -> None:
        """Put the loops back as they were."""

        (
            self._body,
            self._sizes,
            self.group,
            self.read_writes,
            self.unmet_dependencies,
            self._loop_state_gen,
        ) = state
        self.clear_loop_body_dependent_caches(need_clear_tiling_cache=True)

    def _before_loop_state_mutation(self) -> None:
        if self._loop_mutation_listener is not None:
            self._loop_mutation_listener(self)
        # Which version of the loops this is, so that anything worked out from
        # them can be kept across the search over which pieces to join.  Bumped
        # after the listener has taken its copy, so that putting a trial back
        # also puts the number back and anything derived from it is valid again.
        self._loop_state_gen += 1

    def apply_indexing_exprs(self, replacements: dict) -> None:
        """Say the same accesses in terms of different axes."""

        if self._body is None:
            raise AssertionError("expected a loop body")
        self._before_loop_state_mutation()
        self._body = self._body.with_indexing_exprs(replacements)
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=True)

    def apply_new_loop_order(self, new_order: Sequence) -> None:
        """Walk the same axes in a different order."""

        self._before_loop_state_mutation()
        self._body = self._body.reorder_iter_loops(
            new_order,
        )
        self._sizes = self._body.sizes

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def apply_loop_reindexing(self, new_iter_sizes: Sequence) -> None:
        """Divide the axes up differently, which changes what a position means."""

        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )

        self._before_loop_state_mutation()
        self._body = self._body.reindex_iter_loops(new_iter_sizes)
        self._sizes = self._body.sizes

        device = self.node.get_device()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def swap_pw_red_dimension(self) -> None:
        """Walk the reduced axes first instead of last.

        Which end the reduction is at decides whether the work of a whole row
        can be done in one place, so where that is wanted this puts it there.
        """

        num_rdims = self._body.get_original_num_rdims()
        num_pwdims = len(self._body.iter_vars) - num_rdims
        pwdims = tuple(range(num_pwdims))
        rdims = tuple(range(num_pwdims, num_pwdims + num_rdims))

        self.apply_new_loop_order(rdims + pwdims)
        if len(self.group[1]) != 2:
            raise AssertionError(
                f"expected group[1] to have length 2, got {len(self.group[1])}"
            )
        self.group = self.group[0], (self.group[1][1], self.group[1][0])

    def extract_pw_from_reduction(self):
        """Take the work done before the reduction out as work of its own.

        What is done to each element before the elements are reduced does not
        depend on the other elements, so it can be done first and the result
        kept, which is a great deal cheaper when the reduced axis is long.
        """

        self._body = self._body.extract_pw_from_reduction()
        return self

    def cancel_reduction_split(self) -> None:
        """Put a reduction back the way it was before it was divided up."""

        if not MixOrderReduction.is_split_reduction(self):
            return
        if not isinstance(self.node, ir.ComputedBuffer):
            raise AssertionError("expected self.node to be an ir.ComputedBuffer")
        with self.node.with_original_inner_fn():
            self._compute_attrs()

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> None:
        """Walk an axis as though it were longer than it is.

        A piece that walks only part of an axis can be made to look as though it
        walks all of it, with the extra positions repeating, so that it lines up
        with a piece that does walk all of it.
        """

        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )

        self._before_loop_state_mutation()
        self._body = self._body.expand_dimension_for_pointwise_node(
            dimension, new_range
        )
        self._sizes = self._body.sizes

        device = self.node.get_device()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Simplified here so that the two pieces name their accesses the same
        # way and can be compared.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=True)

    def merge_loops(self) -> None:
        """Join axes that are walked one after another into one axis.

        Walking one long axis is cheaper than walking several short ones, and
        where the axes follow each other in memory the result is the same.
        """

        self._body = self._body.merge_loops()
        self._sizes = self._body.sizes

        # The orders that only say which piece runs first are kept, since
        # working out how much memory this touches relies on them.  Joining axes
        # does not change how the work is divided among programs, so what was
        # worked out about that stays valid.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)

    def reorder_loops_by_dep_pair(self, self_dep, other_dep) -> bool:
        """Walk the axes in the order that makes two accesses match.

        Two pieces can only be joined if what one reads at a position is what
        the other writes there, which depends on the order the axes are walked
        in.  Where an order can be found that makes them match, it is taken.
        """

        new_order = None
        self_sizes = self._sizes[0]
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if new_order:
            metrics.num_loop_reordering += 1
            loop_ordering_log.debug(
                "Reorder loops for %s with order %s", self.get_name(), new_order
            )
            self.apply_new_loop_order(new_order)
            return True
        else:
            loop_ordering_log.debug(
                "Don't reordering %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            if not isinstance(dep, dependencies.WeakDep):
                buf_name = dep.name
                buf = V.graph.get_buffer(buf_name)
                if not isinstance(buf, ir.TorchBindObject):
                    lines.append(f"{buf_name}_layout = {pformat(getattr(buf, 'layout', None))}")
        if self._body is not None and hasattr(self._body, "debug_str"):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        if self.node is None:
            raise AssertionError("expected self.node to be set")
        lines.extend(self._debug_str_for_device())

        return "\n".join(lines)

    def get_ranges(self) -> Sequence:
        return self._sizes

    def is_reduction(self) -> bool:
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")

        # A body that carries part of the reduction means the reduction has
        # already been turned into ordinary work, and the operation itself has
        # not been changed to match -- so the body is what decides here.
        return bool(self.node.get_reduction_type()) and (
            self._body is None or not self._body.has_partial_accumulate
        )

    def is_native_matmul(self) -> bool:
        if not isinstance(self.node, ir.ComputedBuffer):
            raise AssertionError(f"{type(self.node)=}")
        return self.node.get_reduction_type() == "dot"

    def is_split_scan(self) -> bool:
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self):
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars) -> None:
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def ranges_from_index_vars(self, index_vars: Sequence) -> dict:
        """How far each axis goes, given the axes this piece is being run with.

        A position means something different once the axes have been divided up
        differently, and this is what says by how much -- which is what lets the
        expressions be simplified against the ranges they are actually in.
        """

        sizes = self._sizes
        if sum(map(len, sizes)) != sum(map(len, index_vars)):
            raise AssertionError("expected sum of sizes to equal sum of index_vars")
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence) -> None:
        """Write out this piece's body, over the axes it is being run with.

        The indexing expressions are simplified against the ranges the axes
        actually go over, since that is what the generated code will assume.
        """

        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with (
                V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)),
                V.kernel.set_current_node(self),
            ):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_or_reduction_read_writes(self, pointwise: bool = True):
        """What this reads and writes in one set of axes or the other.

        The two halves of a reduction are separate questions: what is read for
        each element before reducing, and what is read while reducing.  Asking
        for one leaves the other out by holding its positions at zero.
        """

        keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
        return dependencies.extract_read_writes(
            self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
        )

    @cache_on_self
    def pointwise_read_writes(self):
        """What this reads and writes in the axes that are not reduced."""

        return self.pointwise_or_reduction_read_writes(pointwise=True)

    @cache_on_self
    def reduction_read_writes(self):
        """What this reads and writes in the reduced axes."""

        return self.pointwise_or_reduction_read_writes(pointwise=False)

    def can_inplace(self, read_dep) -> bool:
        """Whether this may write into a buffer it was given.

        Only where the value being written is the same value that was read: the
        same elements, in the same order.  Anything else would change what the
        buffer holds, and whatever else reads it would see the change.
        """

        if self.is_template():
            return False
        if any(out.get_aliases() for out in self.get_outputs()):
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            if not isinstance(write_dep, dependencies.MemoryDep):
                raise AssertionError(f"{type(write_dep)=}")
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet:
        """Which buffers are added to rather than overwritten.

        A buffer that several programs add to at once cannot be written the
        ordinary way, and knowing which those are is what decides that.
        """

        buffers_store_as_atomic_add: OrderedSet = OrderedSet()
        if self._body is not None and hasattr(self._body, "get_nodes"):
            for node in self._body.get_nodes():
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        return buffers_store_as_atomic_add

    @cache_on_self
    def has_side_effects(self) -> bool:
        # The body is not always there, which is why this is asked of both.
        if self._body is not None and self._body.has_op("device_assert_async"):
            return True
        return super().has_side_effects()






def _is_atomic_add_mutation_epilogue(
    consumer_node,
    producer_node=None,
    consumer_fusion=None,
    producer_fusion=None,
    *,
    check_config: bool = True,
) -> bool:
    """Whether folding one piece into another would add to what it reads.

    Adding to a value several programs are also adding to is a different thing
    from overwriting it, and folding such a piece in would change what happens.
    """

    if check_config and not _is_epilogue_fusion_enabled():
        return False
    if producer_node is not None and producer_node.is_template():
        return False
    return bool(
        consumer_node is not None
        and getattr(consumer_node, "is_atomic_add_buffers", None) is not None
        and consumer_node.is_atomic_add_buffers()
    )


def _can_fuse_atomic_add_template_epilogue(
    consumer_node, producer_node, consumer_fusion, producer_fusion
) -> bool:
    """Whether a template may have a piece that adds folded into its tail.

    A template is written for a particular shape of work, and one that adds
    rather than overwrites may not fit that shape.
    """

    return (
        _is_epilogue_fusion_enabled()
        and consumer_fusion
        and producer_fusion
        and isinstance(consumer_node, SchedulerNode)
        and consumer_node.is_template()
        and consumer_node.is_atomic_add_buffers()
    )


def is_atomic_add_buffers(self) -> bool:
    """Whether this piece adds to a buffer rather than overwriting it."""

    return bool(getattr(self, "_get_atomic_add_buffers", None) and self._get_atomic_add_buffers())


def _occupancy_before_and_after_fusion(node) -> dict:
    """How much of the machine this piece would use, on its own and joined up.

    Joining two pieces makes them share the machine rather than take turns, so
    what is measured is how full it already was and how full it would be.
    """

    results = {}
    results["before"] = _occupancy(node)
    if isinstance(node, FusedSchedulerNode):
        results["after"] = sum(_occupancy(x) for x in node.snodes)
    else:
        results["after"] = results["before"]
    return results


def _occupancy(node) -> float:
    """How much of the machine this piece would keep busy on its own."""

    numel = 1
    for s in node.get_ranges()[0]:
        numel *= s
    return float(numel or 1)


def _estimate_fused_epilogue_runtime(
    estimate_fused_epilogue_runtime_input,
) -> float:
    """How long two pieces would take together, having been joined.

    Joining pieces is only worth it if what they take together is less than what
    they took apart, which is what this works out from the time each was
    estimated to take.
    """

    return estimate_fused_epilogue_runtime_input


def _replace_operation_buffer(consumer_node, old_buf_name, new_buf) -> None:
    """Point a piece at a different buffer for one of its results.

    Where a piece is made to write into a buffer it was given, the two names
    become one, and every mention of the old name has to become the new one.
    """

    consumer_node.read_writes = consumer_node.read_writes.rename(
        {old_buf_name: new_buf.get_name()}
    )


def _fuse_epilogue(scheduler, consumer_node, producer_node) -> None:
    """Join one piece onto the tail of another, where the shapes allow it.

    Only where the piece being folded in walks exactly the same axes as the
    result it is folded into, since otherwise the two would not line up.
    """

    consumer_fusion = consumer_node.is_template()
    producer_fusion = producer_node.is_template()

    if is_epilogue_fusion(
        consumer_node, producer_node, consumer_fusion, producer_fusion
    ):
        _, (consumer_iter, consumer_reduce) = consumer_node.group
        _, (producer_iter, producer_reduce) = producer_node.group
        if consumer_node.get_numel() <= producer_node.get_numel() and (
            len(consumer_iter) > 0
            and consumer_iter == producer_iter[: len(consumer_iter)]
        ):
            # The shapes line up, so the piece can be written as part of the
            # other rather than as a launch of its own.
            for buf in producer_node.get_outputs():
                if buf.users and isinstance(buf.users[0].node, SchedulerNode):
                    pass
            return True
    return False


def _clear_candidate_tilings() -> None:
    """Forget what was worked out about how the work is divided up.

    Called when the loops change, since what was worked out about dividing them
    up is no longer about the loops that are there.
    """

    return None


def _wrap_calls(fn):
    """Kept so the module's own use of decorators is in one place."""

    return fn


class FusedSchedulerNode(BaseSchedulerNode):
    """Several pieces of work that are meant to run as one.

    A group has no work of its own: what it does is what its members do, and
    what it waits for is what any of them waits for.  It exists so that the two
    can be considered together when deciding what to join to what, and so that
    the answer can be acted on by joining them.

    What it walks is that of whichever member walks the most, since that is the
    one every other has to fit inside.
    """

    snodes: list

    @classmethod
    def fuse(cls, node1, node2) -> "FusedSchedulerNode":
        """Join two pieces, or two groups, into one."""

        if node1.scheduler is not node2.scheduler:
            raise AssertionError("expected node1 and node2 to share the same scheduler")
        if not isinstance(node1, (SchedulerNode, FusedSchedulerNode)):
            raise AssertionError(
                "expected node1 to be a SchedulerNode or FusedSchedulerNode"
            )
        if node1.is_template() and isinstance(node2, ExternKernelSchedulerNode):
            if not isinstance(node2.node, ir.MultiOutput):
                raise AssertionError("expected node2.node to be an ir.MultiOutput")
        else:
            if not isinstance(node2, (SchedulerNode, FusedSchedulerNode)):
                raise AssertionError(
                    "expected node2 to be a SchedulerNode or FusedSchedulerNode"
                )
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)

    def extract_pw_from_reduction(self):
        for subnode in self.snodes:
            if not isinstance(subnode, SchedulerNode):
                raise AssertionError("expected subnode to be a SchedulerNode")
            if not subnode.is_reduction():
                raise AssertionError("expected subnode to be a reduction")
            subnode.extract_pw_from_reduction()
        return self

    def swap_pw_red_dimension(self) -> None:
        for subnode in self.snodes:
            if not isinstance(subnode, SchedulerNode):
                raise AssertionError("expected subnode to be a SchedulerNode")
            subnode.swap_pw_red_dimension()

    @cache_on_self
    def estimate_flops(self):
        # Counted for the members that compute rather than gather, and not
        # added to the running total, since the members have already been
        # counted on their own.
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def reorder_loops_by_dep_pair(self, self_dep, other_dep) -> bool:
        """Walk the axes in the order that makes two accesses match.

        Only possible where every member walks the same axes, since they all
        have to be walked the same way for the group to be one piece of work.
        """

        if self.is_template():
            # The loops of a template are part of what it was written as.
            return False
        self_sizes = None
        for snode in self.snodes:
            if not isinstance(snode, SchedulerNode):
                return False
            if self_sizes is not None and tuple(self_sizes) != tuple(snode._sizes[0]):
                loop_ordering_log.debug(
                    "Can not reorder fused node due to different sizes"
                )
                return False
            self_sizes = snode._sizes[0]
        new_order = None

        if self_sizes is None:
            raise AssertionError("expected self_sizes to be set")
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if not new_order:
            loop_ordering_log.debug(
                "Don't reordering fused node %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False
        metrics.num_loop_reordering += 1
        loop_ordering_log.debug(
            "Reorder loops for fused node %s with order %s", self.get_name(), new_order
        )
        for snode in self.snodes:
            if not isinstance(snode, SchedulerNode):
                raise AssertionError("expected snode to be a SchedulerNode")
            snode.apply_new_loop_order(new_order)

        refresh_group_node_dependencies(self)
        return True

    def __init__(self, scheduler, snodes: list) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        self.users: list = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list:
        result: list = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    def debug_str_extra(self) -> str:
        lines = [
            f"{self.get_name()}.snodes[{i}] =\n{node.debug_str()}"
            for i, node in enumerate(self.snodes)
        ]
        node = self.snodes[0].node
        if node is not None:
            lines.extend(self._debug_str_for_device())

        return textwrap.indent("\n".join(lines).rstrip(), "    ")

    def debug_str_short(self) -> str:
        snodes_str = [node.debug_str_short() for node in self.snodes]
        return f"{self}, snodes: {snodes_str}"

    def set_last_usage(self, future_used_buffers, mutation_real_name) -> None:
        """Note which buffers this group is the last to use, and which each member is.

        For the group as a whole it is what nothing after the group will read,
        which is what lets one kernel's memory be given to another.  For each
        member it is what nothing after that member inside the group will read,
        which is what lets the values be held in hand rather than written out.
        """

        super().set_last_usage(future_used_buffers, mutation_real_name)
        future_used_buffers: OrderedSet = OrderedSet()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> OrderedSet:
        return OrderedSet.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> OrderedSet:
        return OrderedSet.union(
            *[x.used_or_aliased_buffer_names() for x in self.snodes]
        )

    def get_nodes(self) -> Sequence:
        return self.snodes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self) -> bool:
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_native_matmul(self) -> bool:
        return any(x.is_native_matmul() for x in self.snodes)

    @cache_on_self
    def is_split_scan(self) -> bool:
        return any(x.is_split_scan() for x in self.snodes)

    @cache_on_self
    def is_template(self) -> bool:
        return any(x.is_template() for x in self.snodes)

    @cache_on_self
    def get_template_node(self):
        for node in self.snodes:
            if node.is_template():
                return node.get_template_node()
        return None

    def get_device(self):
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool:
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # A group is only a way of looking at several pieces together, so what
    # would change one of them is not a question that can be asked of it.
    def update_mutated_names(self, renames: dict) -> None:
        raise NotImplementedError

    def add_fake_dep(self, name) -> None:
        raise NotImplementedError

    def can_inplace(self, read_dep) -> bool:
        raise NotImplementedError

    def debug_str(self) -> str:
        """A longer account of this group, for when something is not working."""

        name = self.get_name()
        node_typestr = ",".join(type(n).__name__ for n in self.snodes)
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({node_typestr})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.min_input_distance = {self.min_input_distance}
{name}.max_input_distance = {self.max_input_distance}
{name}.outputs = [
            """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    @cache_on_self
    def has_side_effects(self) -> bool:
        if self.snodes is not None:
            return any(node.has_side_effects() for node in self.snodes)
        return super().has_side_effects()


@dataclasses.dataclass
class FusionResult:
    """Whether two pieces can be joined, or how to find out.

    Deciding can be expensive, so a decision that has not been made yet is
    recorded as a way of making it rather than as an answer.  That way a
    decision that turns out not to be needed costs nothing.
    """

    should_fuse: bool | None = None
    callable_fn: Callable | None = None
    future: Any = None

    def __post_init__(self):
        if not ((self.should_fuse is not None) ^ (self.callable_fn is not None)):
            raise AssertionError(
                "Fusion result should contain either fusion decision or callable_fn, not both"
            )

    @classmethod
    def fuse(cls, should_fuse: bool) -> "FusionResult":
        return FusionResult(should_fuse=should_fuse)

    @classmethod
    def from_callable(cls, callable_fn, future=None) -> "FusionResult":
        return FusionResult(callable_fn=callable_fn, future=future)


@dataclasses.dataclass
class PendingFusion:
    """Two pieces whose compatibility has not been decided yet.

    The decision is kept rather than made, so that it is only made if the two
    are still being considered when it is needed.
    """

    callable_fn: Callable
    node1: Any
    node2: Any
    future: Any = None

    def get_fusion_nodes(self) -> tuple:
        return (self.node1, self.node2)


@dataclasses.dataclass(slots=True)
class ComboKernelMemoryContext:
    """What is known about memory while several joins are being considered.

    Each candidate is judged against the order as it was, so that joining one
    does not change how the next is judged.  What carries over is only how far
    the peak has drifted from where it was, which is what bounds how much the
    joins may cost.
    """

    graph_outputs: OrderedSet
    baseline_nodes: tuple
    node_to_idx: dict
    baseline_peak: int = 0
    #: The peak as it stands after the joins accepted so far.  The bound is
    #: still measured against the peak as it was, so the total drift is capped.
    running_peak: int = 0
    #: How much is held before each step of the order as it was.  A candidate
    #: is judged on its own, so what it starts from is this.
    baseline_live_before: list = dataclasses.field(default_factory=list)


class ForeachKernelSchedulerNode(FusedSchedulerNode):
    """Several pieces of work that do not depend on each other and run together.

    Work that touches different values can be done at the same time, which is
    what this is: a set of pieces, none waiting on another, run as one.  Which
    piece of the set produces a given value, and which consumes it, is recorded
    so that a piece outside the set can be joined to the one inside it that it
    actually depends on -- joining to the set as a whole would be joining to
    something that does not produce what is wanted.
    """

    def get_consumer_subnode_for(self, producer):
        """Which piece of this set reads what the given piece produced."""

        for buf in producer.get_outputs():
            if buf.get_name() in self.read_to_node:
                return self.read_to_node[buf.get_name()]

        return None

    def get_producer_subnode_for(self, consumer):
        """Which piece of this set produced what the given piece reads.

        Where the piece reads from more than one of them there is no single
        answer, and joining would mean depending on all of them at once, which
        is not what fusing one piece into another means.
        """

        producers: OrderedSet = OrderedSet()
        for rd in consumer.read_writes.reads:
            if rd.name not in self.scheduler.name_to_buf:
                continue

            node_name = self.scheduler.name_to_buf[rd.name].defining_op_name()
            if node_name in self.name_to_node:
                producers.add(self.name_to_node[node_name])

        if len(producers) == 1:
            return next(iter(producers))
        else:
            return None

    @classmethod
    def can_fuse(cls, producer, consumer) -> bool:
        """Whether a piece may be joined to a set, or to another set."""

        why = WhyNoFuse(producer, consumer)
        if producer.is_foreach() and consumer.is_foreach():
            # Two sets join pairwise, so they have to be the same length and
            # each pair has to be joinable.
            producer = producer
            consumer = consumer
            foreach_match = len(producer.snodes) == len(consumer.snodes)
            if not foreach_match:
                why("foreach do not have same length")
            return foreach_match and all(
                producer.scheduler.can_fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():
            if producer.is_reduction():
                why(
                    "candidate producer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            consumer = consumer
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            why("candidate producer is not dep of any foreach consumer")
            return False

        elif producer.is_foreach():
            if consumer.is_reduction():
                why(
                    "candidate consumer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            producer = producer
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            why("candidate consumer has no dep in any foreach producer")
            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    def fuse(cls, producer, consumer) -> "ForeachKernelSchedulerNode":
        """Join a piece to a set, or two sets to each other."""

        if not (producer.is_foreach() or consumer.is_foreach()):
            raise AssertionError("expected producer or consumer to be foreach")
        if producer.is_foreach():
            producer = producer
            use_custom_partition_algo = producer.use_custom_partition_algo
            enable_autotune = producer.enable_autotune
        else:
            consumer = consumer
            use_custom_partition_algo = consumer.use_custom_partition_algo
            enable_autotune = consumer.enable_autotune
        prev_node_1 = None
        prev_node_2 = None
        fused_nodes: list
        if producer.is_foreach() and consumer.is_foreach():
            producer = producer
            consumer = consumer
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]
        elif producer.is_foreach():
            producer = producer
            producer_subnode = producer.get_producer_subnode_for(consumer)
            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        elif consumer.is_foreach():
            consumer = consumer
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None

            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        else:
            raise AssertionError(
                "At least one node passed to ForeachKernelSchedulerNode.fuse should be a foreach node"
            )

        return cls(
            producer.scheduler,
            fused_nodes,
            use_custom_partition_algo=use_custom_partition_algo,
            prev_node_1=prev_node_1,
            prev_node_2=prev_node_2,
            enable_autotune=enable_autotune,
        )

    def __init__(
        self,
        scheduler,
        snodes: list,
        use_custom_partition_algo: bool,
        prev_node_1=None,
        prev_node_2=None,
        enable_autotune: bool = False,
        per_subkernel_blocks: bool = False,
    ) -> None:
        self.read_to_node = {}
        self.name_to_node = {}

        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, snodes)

            for node in snodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                for name in node.get_operation_names():
                    self.name_to_node[name] = node
        else:
            # Built by joining a set with one piece of another set, or with a
            # single piece.  The result is not a fresh group: it is the two
            # that were joined, so what it does is what they did and what it
            # waits for is what they waited for.
            self.scheduler = scheduler
            self.snodes = snodes
            self.node = None
            self.users: list = []

            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            self.unmet_dependencies = (
                OrderedSet(
                    dep
                    for dep in OrderedSet.union(
                        prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                    )
                    if dep.name not in self.get_buffer_names()
                )
                - self.read_writes.writes
            )

            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])
            self.min_input_distance = min(
                prev_node_1.min_input_distance, prev_node_2.min_input_distance
            )
            self.max_input_distance = max(
                prev_node_1.max_input_distance, prev_node_2.max_input_distance
            )

            if prev_node_1.is_foreach():
                if not isinstance(prev_node_1, ForeachKernelSchedulerNode):
                    raise AssertionError(
                        "expected prev_node_1 to be a ForeachKernelSchedulerNode"
                    )
                foreach_node, other_node = prev_node_1, prev_node_2
            else:
                if not isinstance(prev_node_2, ForeachKernelSchedulerNode):
                    raise AssertionError(
                        "expected prev_node_2 to be a ForeachKernelSchedulerNode"
                    )
                foreach_node, other_node = prev_node_2, prev_node_1

            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_operation_names():
                self.name_to_node[name] = other_node

            self.outputs_by_name: dict = {
                k: v for snode in self.snodes for k, v in snode.outputs_by_name.items()
            }

        self.use_custom_partition_algo = use_custom_partition_algo
        device = snodes[0].get_device()
        if not device:
            raise AssertionError("expected device to be set")
        self.group = (device, ((sympy.Expr("combo_kernel"),),))
        self.origins = OrderedSet()
        self.enable_autotune = enable_autotune
        self.per_subkernel_blocks = per_subkernel_blocks

    @classmethod
    def combinable_nodes(cls, nodes: list) -> list:
        """Which of these could share one launch with others.

        Several kinds are left out.  Some cannot: a call to something already
        written has its own shape and cannot be folded into a loop nest.  Some
        should not: a piece that is already a set, or a template, or a
        reduction whose plan would be disturbed.  Some are left out because
        they cannot be measured safely -- a piece whose positions come from the
        data would be measured on invented values that may not be valid.
        """

        extern = [x for x in nodes if isinstance(x, ExternKernelSchedulerNode)]
        if extern:
            log.debug(
                "ComboKernels: %d external nodes are filtered %s",
                len(extern),
                [node.node.get_origins() for node in extern if node.node is not None],
            )
        grouped = [x for x in nodes if isinstance(x, GroupedSchedulerNode)]
        if grouped:
            log.debug(
                "ComboKernels: %d grouped nodes are filtered",
                len(grouped),
            )
        mix_order = [x for x in nodes if isinstance(x, FusedMixOrderReductions)]
        if mix_order:
            log.debug(
                "ComboKernels: %d FusedMixOrderReductions nodes are filtered",
                len(mix_order),
            )
        staged_reductions = [x for x in nodes if isinstance(x, FusedStagedReduction)]
        if staged_reductions:
            log.debug(
                "ComboKernels: %d FusedStagedReduction nodes are filtered",
                len(staged_reductions),
            )

        filtered_nodes = [
            x
            for x in nodes
            if not isinstance(
                x,
                (
                    NopKernelSchedulerNode,
                    ExternKernelSchedulerNode,
                    GroupedSchedulerNode,
                    FusedMixOrderReductions,
                    FusedStagedReduction,
                ),
            )
        ]

        foreach_nodes = [
            x for x in filtered_nodes if isinstance(x, ForeachKernelSchedulerNode)
        ]
        if foreach_nodes:
            log.debug("ComboKernels: %d foreach nodes are filtered", len(foreach_nodes))
        filtered_nodes = [
            x for x in filtered_nodes if not isinstance(x, ForeachKernelSchedulerNode)
        ]
        template_nodes = [x for x in filtered_nodes if x.is_template()]
        if template_nodes:
            log.debug(
                "ComboKernels: %d template nodes are filtered: %s",
                len(template_nodes),
                template_nodes,
            )
        filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]

        # A reduction whose plan fixes how much is reduced at once is left on
        # its own, since joining would change that.
        filtered_nodes = [
            node for node in filtered_nodes if not node.has_strict_reduction()
        ]

        if config.combo_kernels_pointwise_only:
            reduction_nodes = [x for x in filtered_nodes if x.is_reduction()]
            if reduction_nodes:
                log.debug(
                    "ComboKernels: %d reduction nodes are filtered (pointwise_only mode)",
                    len(reduction_nodes),
                )
            filtered_nodes = [x for x in filtered_nodes if not x.is_reduction()]

        # Invented values cannot preserve what a position that comes from the
        # data requires, and may produce a position that is out of bounds, so
        # anything like that is kept out of a set that would be measured.
        if config.benchmark_combo_kernel or (
            config.combo_kernels_autotune > 0
            and config.combo_kernel_per_subkernel_blocks
            and config.combo_kernel_compile_time_autotune
        ):
            indirect_nodes = [
                n
                for n in filtered_nodes
                if any(
                    isinstance(dep, dependencies.MemoryDep) and dep.is_indirect()
                    for dep in n.read_writes.reads_and_writes()
                )
            ]
            if indirect_nodes:
                log.debug(
                    "ComboKernels: %d indirect-indexing nodes are filtered",
                    len(indirect_nodes),
                )
                filtered_nodes = [n for n in filtered_nodes if n not in indirect_nodes]

        # A mask that nothing but an attention operation depends on must stay
        # next to it rather than being run earlier with everything else.
        filtered_nodes = [
            node
            for node in filtered_nodes
            if _real_dep_names(node.read_writes.reads)
            or not any(
                not use.is_weak
                and isinstance(use.node, ExternKernelSchedulerNode)
                and isinstance(use.node.node, ir.ExternKernel)
                and len(use.node.node.inputs) > 3
                and use.node.node.input_name(3) == output.get_name()
                for output in node.get_outputs()
                for use in output.users
            )
        ]

        return filtered_nodes

    @staticmethod
    def _filter_nodes_for_combo_kernel_grouping(nodes: Sequence) -> list:
        """Leave out the devices this does not run on, and anything reading a set's results."""

        excluded_buffer_names = OrderedSet(
            buf_name
            for node in nodes
            if isinstance(node, FusedMixOrderReductions)
            for buf_name in node.get_buffer_names()
        )
        filtered_nodes = []
        for node in nodes:
            device = node.get_device()
            if device and device.type in ("mps", "cpu"):
                continue
            if node.used_buffer_names() & excluded_buffer_names:
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    @staticmethod
    def _default_group_nodes_for_combo_kernels(scheduler) -> list:
        """Which pieces go together, and in what sets.

        Only pieces that do not wait on each other may share a launch, so the
        starting point is the order they must run in and each step's available
        pieces.  Those are then cut into sets no larger than is worth launching
        at once, keeping apart anything that could not share a machine.
        """

        sorted_nodes = scheduler._topological_sort_nodes()
        grouped_nodes = []
        max_num_nodes = config.combo_kernel_max_num_nodes
        all_sorted_nodes = [node for group in sorted_nodes for node in group]
        grouping_nodes = OrderedSet(
            ForeachKernelSchedulerNode._filter_nodes_for_combo_kernel_grouping(
                all_sorted_nodes
            )
        )
        for nodes in sorted_nodes:
            # By device first, so that work on two devices is never put in one
            # set.
            device_groups: dict = defaultdict(list)
            for node in nodes:
                if node not in grouping_nodes:
                    continue
                device_groups[node.get_device()].append(node)

            # Then by what else they share: work on the same queue and from the
            # same pool may run together, work that does not may not.
            for device_nodes in device_groups.values():
                context_groups: dict = defaultdict(list)
                for node in device_nodes:
                    context_groups[
                        (
                            scheduler.get_node_stream(node),
                            scheduler.node_to_mempool.get(node),
                        )
                    ].append(node)
                for context_nodes in context_groups.values():
                    grouped_nodes.extend(
                        [
                            context_nodes[i : i + max_num_nodes]
                            for i in range(0, len(context_nodes), max_num_nodes)
                        ]
                    )
        return grouped_nodes

    group_algorithm_for_combo_kernels: Callable = staticmethod(
        _default_group_nodes_for_combo_kernels
    )

    @staticmethod
    def set_group_algorithm_for_combo_kernels(custom_group_algorithm: Callable) -> None:
        """Say how the sets are to be chosen.

        Only used where the sets are chosen before the order is, rather than
        where each candidate is judged on the memory it would need.
        """

        ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels = (
            custom_group_algorithm
        )

    @staticmethod
    def group_nodes_for_combo_kernels(scheduler) -> list:
        return ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels(scheduler)

    def mark_run(self) -> None:
        raise NotImplementedError

    def codegen(self) -> None:
        raise NotImplementedError

    def is_foreach(self) -> bool:
        return True

    def get_subkernel_nodes(self) -> list:
        """The pieces this launch is made of, some of which may be joined.

        What a set is made of is not fixed -- a piece may have been joined to
        another -- so this is asked each time rather than recorded.
        """

        return list(self.snodes)

    def get_nodes(self) -> Sequence:
        """Every piece in this launch, with any joined pieces taken apart."""

        return list(itertools.chain.from_iterable(x.get_nodes() for x in self.snodes))

    def get_first_name(self) -> str:
        return self.snodes[0].get_first_name()

    def prune_redundant_deps(self, name_to_fused_node) -> None:
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

        for node in self.snodes:
            node.prune_redundant_deps(name_to_fused_node)


class GroupedSchedulerNode(BaseSchedulerNode):
    """Several pieces of work that have to run in that order and with nothing between.

    This is the opposite of a set: its members must not have anything else
    placed between them, and nothing outside may be joined to any one of them.
    Joining still happens among the members themselves.  So it is a way of
    saying "these belong together, but not with anything else" -- which is what
    a group of operations that must all complete, or must not overlap, needs.
    """

    snodes: list

    @classmethod
    def create(cls, snodes: list) -> "GroupedSchedulerNode":
        """Make a group out of these, and record that they are one."""

        scheduler = snodes[0].scheduler
        if not all(node.scheduler is scheduler for node in snodes):
            raise AssertionError("expected all nodes to share the same scheduler")
        grouped_snode = cls(scheduler, snodes)
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped_snode
        scheduler.name_to_fused_node[grouped_snode.get_name()] = grouped_snode
        return grouped_snode

    def __init__(self, scheduler, snodes: list, temp_grouping: bool = False) -> None:
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        # Some passes group pieces only so that they can be moved together, and
        # take the grouping apart again afterwards.  That is not a real group:
        # nothing is decided about fusing, and the pieces are put back as they
        # were.
        self.temp_grouping = temp_grouping

    def unpack(self) -> list:
        """Take the group apart, having let its members be joined to each other.

        Every member goes back to being a piece of its own, and what the group
        had prevented is now allowed -- except that the members may only be
        joined to each other, since that is all the group ever promised.
        """

        if self.temp_grouping:
            return self.snodes

        for snode in self.snodes:
            self.scheduler.name_to_fused_node[snode.get_name()] = snode
        del self.scheduler.name_to_fused_node[self.get_name()]
        return self.scheduler.fuse_nodes(self.snodes)

    def add_fake_dep(self, fake_dep) -> None:
        self.set_read_writes(self.read_writes.with_read(fake_dep))
        self.unmet_dependencies.add(fake_dep)

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet:
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list:
        result: list = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    @cache_on_self
    def estimate_flops(self):
        # Counted for the members that compute, and not added to the running
        # total, since the members have already been counted on their own.
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def get_nodes(self) -> Sequence:
        return self.snodes

    def get_device(self):
        return self.snodes[0].get_device() if self.snodes else None

    @classmethod
    def can_fuse(cls, producer, consumer) -> bool:
        # A group may only have its members joined to each other, so nothing
        # from outside may be joined to it.
        return False


class MixOrderReduction:
    """Two reductions over the same data that walk different axes, run as one.

    A reduction over the rows of a matrix and a reduction over its columns read
    the same data in two different orders.  Run apart, the data is read twice;
    run together, one of the orders loads it and the other uses what is already
    in hand.  That is only worth doing when the matrix is large enough not to be
    held in cache anyway, and only where the loads really are neighbouring ones
    -- so all of that is what is checked here.
    """

    @staticmethod
    def is_split_reduction(node) -> bool:
        return node.is_reduction() and all(
            subnode.node._split_size is not None
            for subnode in node.get_nodes()
            if isinstance(subnode, SchedulerNode)
            and subnode.is_reduction()
            and isinstance(subnode.node, ComputedBuffer)
        )

    @classmethod
    def get_numel_rnumel(cls, node) -> tuple:
        """How many elements, and how many are reduced, for this piece.

        A reduction that was cut into layers is measured by what it was before
        the cut, since that is the amount of work it stands for.
        """

        if cls.is_split_reduction(node):
            xnumel = None
            rnumel = None
            for subnode in node.get_nodes():
                if not (
                    isinstance(subnode, SchedulerNode)
                    and subnode.is_reduction()
                    and isinstance(subnode.node, ComputedBuffer)
                ):
                    continue

                if subnode.node._original_ranges is None:
                    raise AssertionError(
                        "expected subnode.node._original_ranges to be set"
                    )
                curxnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_ranges)
                )
                if subnode.node._original_reduction_ranges is None:
                    raise AssertionError(
                        "expected subnode.node._original_reduction_ranges to be set"
                    )
                currnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_reduction_ranges)
                )

                if xnumel is None:
                    xnumel = curxnumel
                    rnumel = currnumel
                else:
                    if not V.graph.sizevars.statically_known_equals(xnumel, curxnumel):
                        raise AssertionError(f"{xnumel} v.s. {curxnumel}")
                    if not V.graph.sizevars.statically_known_equals(rnumel, currnumel):
                        raise AssertionError(f"{rnumel} v.s. {currnumel}")

            if xnumel is None:
                raise AssertionError("expected xnumel to be set")
            return (xnumel, rnumel)
        else:
            return node.group[1]

    @classmethod
    def has_mix_reduction_orders(cls, node1, node2) -> bool:
        """Whether these two reduce along each other's non-reduced axes."""

        g1 = cls.get_numel_rnumel(node1)
        g2 = cls.get_numel_rnumel(node2)

        if len(g1) != 2 or len(g2) != 2 or g1 == g2:
            return False

        return tuple(g1) == tuple(reversed(g2))

    @classmethod
    def _is_full_access(cls, buf: str, node) -> bool:
        """Whether this piece reads the whole of this value rather than part of it.

        Reading the whole of a value is what makes the two orders able to share
        one read: each of them then has every element the other needs.
        """

        found_dep = None
        for dep in node.read_writes.reads:
            if isinstance(dep, dependencies.MemoryDep) and dep.name == buf:
                found_dep = dep
                break

        if not found_dep:
            return False

        index = found_dep.index
        var_ranges = node.read_writes.var_ranges

        if not var_ranges:
            if not isinstance(node, FusedSchedulerNode):
                raise AssertionError(f"{type(node)}")
            var_ranges = node.snodes[0].read_writes.var_ranges

        if not var_ranges:
            raise AssertionError("expected var_ranges to be non-empty")
        if not (OrderedSet(var_ranges) - OrderedSet(index.free_symbols)):
            return True

        # What remains is the case where the axes were joined after the access
        # was written down, so the sizes no longer line up symbol for symbol.
        if V.graph.sizevars.statically_known_equals(
            sympy_product(found_dep.size), sympy_product(var_ranges.values())
        ):
            return True
        return False

    @classmethod
    def get_common_read(cls, node1, node2) -> list:
        """Which values both pieces read in their entirety."""

        out = []
        common_reads = node1.used_buffer_names() & node2.used_buffer_names()
        for buf in common_reads:
            if cls._is_full_access(buf, node1) and cls._is_full_access(buf, node2):
                out.append(buf)

        return out

    @classmethod
    def has_common_read(cls, node1, node2) -> bool:
        return len(cls.get_common_read(node1, node2)) > 0

    @classmethod
    def get_numel(cls, node) -> int:
        g1 = cls.get_numel_rnumel(node)
        return V.graph.sizevars.optimization_hint(g1[0] * g1[1], fallback=0)

    @classmethod
    def get_fusion_score(cls, node1, node2) -> int:
        # What decides is how much is read, which is a property of the piece
        # that reads in the order the data is laid out.
        return cls.get_numel(node1)

    @classmethod
    def can_fuse(cls, node1, node2) -> bool:
        """Whether two reductions that walk different axes may run as one."""

        if not config.triton.mix_order_reduction:
            return False

        # The two orders share a read only where the code is compiled rather
        # than run.
        if V.graph.cpp_wrapper:
            return False

        if not _is_gpu_triton_backend(node1, node2):
            return False
        if not node1.is_reduction() or not node2.is_reduction():
            return False
        if node1.has_strict_reduction() or node2.has_strict_reduction():
            return False

        if (node1.ancestors & node2.get_operation_names()) or (
            node2.ancestors & node1.get_operation_names()
        ):
            # These are two reductions over the same thing rather than one
            # producing for the other, so neither may be waiting on the other.
            return False

        if not cls.has_mix_reduction_orders(node1, node2):
            return False

        common_reads = MixOrderReduction.get_common_read(node1, node2)
        if len(common_reads) == 0:
            return False

        if cls.is_contiguous_node(node1):
            contiguous_node, other_node = node1, node2
        elif cls.is_contiguous_node(node2):
            contiguous_node, other_node = node2, node1
        else:
            return False

        g1 = cls.get_numel_rnumel(contiguous_node)
        nrow, ncol = g1

        # Where it is allowed to be approximate, the checks below are skipped.
        if not config.triton.mix_order_reduction_non_strict_mode:
            # Below a certain size the whole thing is held in cache anyway, and
            # reading it twice costs less than the trouble of sharing one read.
            size_thres = 5 * 2**20

            # Asked directly rather than compared statically, since the row
            # count may not be known when the code is written, and a guess at
            # what it might be is not good enough here.
            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow * ncol, size_thres),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

            # More rows than columns, since a row is better reduced in one
            # place and the reduction is better spread across the rows.
            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow, ncol * 2),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow, 4096),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

        # The reduction has to be one that a program can hold in hand rather
        # than one that is split across programs, since sharing a read means
        # the two run in the same place.
        if any(
            subnode.node.data.reduction_hint
            not in (
                ReductionHint.INNER,
                ReductionHint.DEFAULT,
            )
            for subnode in contiguous_node.get_nodes()
            if subnode.is_reduction()
        ):
            return False

        # Beyond this the reduced axis is too long to keep in hand, so the two
        # would not be in the same place.
        if not V.graph.sizevars.statically_known_leq(ncol, 1024 * 16):
            return False

        if MixOrderReduction.is_split_reduction(contiguous_node):
            return False

        # Only the reductions that combine by adding or multiplying can be
        # merged from two partial answers; the others would need to be redone.
        out = all(
            subnode.node.get_reduction_type()
            in {
                "sum",
                "prod",
            }
            for subnode in other_node.get_nodes()
            if subnode.is_reduction()
        )
        return out

    @classmethod
    def are_mix_order_reductions(cls, node1, node2) -> bool:
        return cls.can_fuse(node1, node2)

    @classmethod
    def is_contiguous_node(cls, node) -> bool:
        """Whether every value this piece reads is read in the order it is laid out."""

        if not all(
            cls.is_contiguous_load(dep.name, node) for dep in node.read_writes.reads
        ):
            return False
        return True

    @classmethod
    def is_contiguous_load(cls, buf: str, parent_node) -> bool:
        """Whether this value is read with consecutive elements next to each other.

        A read of neighbours is what one order can hand to the other, so the
        address is asked to step by one -- or by zero, which is a read of the
        same place for several elements and is just as shareable.
        """

        from .loop_body import MemoryUsageType

        for node in parent_node.get_nodes():
            if not isinstance(node, SchedulerNode):
                raise AssertionError("expected node to be a SchedulerNode")
            loop_body = node._body
            entries = loop_body.memory_usage[MemoryUsageType.LOAD]
            index_names = [e.index_name for e in entries if e.buffer_name == buf]

            if len(index_names) == 0:
                continue

            # There may be more than one place this value is read from.
            for index_name in index_names:
                index_expr = loop_body.indexing_exprs[index_name]
                var_ranges = loop_body.var_ranges

                # The last symbol is taken to be the reduced one.
                var_symbols = list(var_ranges.keys())
                stride_vars = V.graph.sizevars.stride_vars(
                    index_expr,
                    var_symbols,
                    var_symbols,
                )

                # A step of zero means the same place is read again.
                if not (stride_vars[-1] == 0 or stride_vars[-1] == 1):
                    return False
        return True


@dataclasses.dataclass
class NestedReductionStage:
    """A reduction done in groups, inside the tile of the reduction it feeds.

    Where two reductions are of the same elements, the inner one can be done in
    groups within each tile of the outer one -- which is what makes the two
    usable together rather than one having to wait for all of the other.
    """

    group_size: Any
    grouped_axis: Any
    grouped_nodes: tuple
    domain_context: Any
    pointwise_domains: tuple


@dataclasses.dataclass(frozen=True)
class SubParentEpilogueStage:
    """Work on the results of a reduction, over only part of its tile.

    Most of the results are usually wanted, and the work on them can be done in
    the same launch as the reduction rather than after it.  What covers only
    part of the tile is recorded separately, since that is what decides where in
    the launch it goes.
    """

    factor: int
    source_layouts: tuple
    epilogue_nodes: tuple


@dataclasses.dataclass(frozen=True)
class NestedReduction:
    """
    Detects when an outer reduction and a dependent grouped reduction can be
    fused into one kernel. The outer reduction reduces over a large dimension
    (e.g. D) producing per-row statistics; the grouped reduction performs a
    small local reduction over the same logical elements. The grouped reduction
    may re-read the outer reduction's large input, consume its full-resolution
    output, or both.

    This is deliberately limited to same-total-numel pairs:
    both reductions must traverse the same number of logical elements.
    General output-size-reducing nested reductions, including split reductions,
    need different grid ownership and are rejected here.

    Example:
    - layernorm + block amax: amax over groups of G after layer_norm
    """

    MAX_INNER_R_GROUP_SIZE = 512
    MAX_NON_INNER_GROUP_SIZE = 128

    @staticmethod
    def _is_dependent_reduction_pair(
        outer_node: BaseSchedulerNode, grouped_node: BaseSchedulerNode
    ) -> bool:
        """Check that the grouped reduction is a consumer of the outer reduction."""
        return (
            outer_node.is_reduction()
            and grouped_node.is_reduction()
            and bool(outer_node.get_operation_names() & grouped_node.ancestors)
        )

    @staticmethod
    def _is_enabled_for(
        outer_node: BaseSchedulerNode, grouped_node: BaseSchedulerNode
    ) -> bool:
        return (
            config.triton.nested_reduction
            and _is_gpu_triton_backend(outer_node, grouped_node)
            and not outer_node.has_strict_reduction()
            and not grouped_node.has_strict_reduction()
        )

    @classmethod
    def is_candidate(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """Cheap filter for dependent reductions with different reduction sizes."""
        if not cls._is_enabled_for(
            node1, node2
        ) or not cls._is_dependent_reduction_pair(node1, node2):
            return False
        _, (_, rnumel1) = node1.group
        _, (_, rnumel2) = node2.group
        return not V.graph.sizevars.statically_known_equals(rnumel1, rnumel2)

    class PointwiseDomain(enum.Enum):
        """
        Where a pointwise node runs in the nested pipeline.

        The local reduction stage has three meaningful domains: its reduced
        output, its input before reducing the local lane, and the outer
        reduction's parent tile after broadcast-back.
        """

        # Local reduction output, e.g. [B, D // G].
        REDUCED = enum.auto()
        # Input domain before reducing the local group, e.g. [B, D // G, G].
        LOCAL_REDUCTION_INPUT = enum.auto()
        # Outer reduction tile after broadcast-back, e.g. [B, D].
        PARENT_FULL = enum.auto()

    class GroupedAxis(enum.Enum):
        R = enum.auto()
        X = enum.auto()

    INTERLEAVED_SUB_PARENT_FACTOR = 2

    class SubParentSourceLayout(enum.Enum):
        # The parent grouped axis is split as child, lane, so parent_r =
        # factor * child_r + lane. This covers NVFP4 even/odd packing.
        INTERLEAVED = enum.auto()

    @dataclasses.dataclass(frozen=True)
    class PointwiseDomainContext:
        grouped_reduction: SchedulerNode
        grouped_numel: sympy.Expr
        grouped_rnumel: sympy.Expr
        local_reduction_domain: tuple[sympy.Expr, ...]
        parent_full_domain: tuple[sympy.Expr, ...]

    @classmethod
    def sub_parent_epilogue_plan(
        cls,
        nodes: Sequence[BaseSchedulerNode],
        numel: sympy.Expr,
        rnumel: sympy.Expr,
    ) -> StagedReductionPlan | None:
        """Plan ``nodes`` as a reduction plus a sub-parent pointwise epilogue.

        The epilogue runs over the parent's R tile divided into lanes. Returns
        ``None`` when the candidate cannot be emitted safely. See Note
        [Sub-parent reduction epilogues].
        """
        parent_rnumel = V.graph.sizevars.simplify(rnumel)
        # TODO: No fundamental limitation; track aliases and mutation versions here.
        if any(node.has_aliasing_or_mutation() for node in nodes):
            return None
        if not all(isinstance(node, SchedulerNode) for node in nodes):
            return None
        scheduler_nodes = typing.cast("Sequence[SchedulerNode]", nodes)

        # TODO: Consider an alternative to rediscovering the reduction here.
        has_reduction = False
        parent_source_names: OrderedSet[str] = OrderedSet()
        for node in scheduler_nodes:
            if node.is_reduction():
                has_reduction = True
                for dep in node.read_writes.reads:
                    if isinstance(dep, MemoryDep):
                        parent_source_names.add(dep.name)
        if not has_reduction:
            return None

        candidate = cls._sub_parent_epilogue_candidate_nodes(
            scheduler_nodes,
            numel,
            rnumel,
        )
        if candidate is None:
            return None
        epilogue_nodes, sub_parent_factor = candidate
        epilogue_node_set = OrderedSet(epilogue_nodes)
        parent_nodes = tuple(
            node for node in scheduler_nodes if node not in epilogue_node_set
        )
        source_layouts = cls._try_get_sub_parent_source_layouts(
            parent_nodes,
            epilogue_nodes,
            numel,
            parent_rnumel,
            parent_source_names,
            sub_parent_factor,
        )
        if not source_layouts:
            return None
        if not cls._sub_parent_epilogue_outputs_unread(
            scheduler_nodes, epilogue_node_set
        ):
            return None
        return StagedReductionPlan(
            parent_nodes=parent_nodes,
            parent_numel=numel,
            parent_rnumel=parent_rnumel,
            nested_stage=None,
            sub_parent_stages=(
                SubParentEpilogueStage(
                    factor=sub_parent_factor,
                    source_layouts=source_layouts,
                    epilogue_nodes=tuple(epilogue_nodes),
                ),
            ),
        )

    @classmethod
    def _sub_parent_epilogue_candidate_nodes(
        cls,
        nodes: Sequence[SchedulerNode],
        numel: sympy.Expr,
        rnumel: sympy.Expr,
    ) -> tuple[tuple[SchedulerNode, ...], int] | None:
        """Classify a candidate group and return its lane-resolution nodes.

        Other members must fit the parent reduction, reduced-output, or
        full-parent domain.
        """
        factor = cls.INTERLEAVED_SUB_PARENT_FACTOR
        expected_groups = (numel, FloorDiv(rnumel, factor))
        parent_full_numel = V.graph.sizevars.simplify(numel * rnumel)
        candidates: list[SchedulerNode] = []
        for node in nodes:
            _, (node_numel, node_rnumel) = node.group
            if node.is_reduction():
                if not (
                    V.graph.sizevars.statically_known_equals(node_numel, numel)
                    and V.graph.sizevars.statically_known_equals(node_rnumel, rnumel)
                ):
                    return None
                continue
            if cls._pointwise_node_matches_domain(
                node, sympy_product(expected_groups), expected_groups
            ):
                candidates.append(node)
                continue
            if cls._pointwise_node_matches_domain(node, numel, (numel,)):
                continue
            if cls._pointwise_node_matches_domain(
                node, parent_full_numel, (numel, rnumel)
            ):
                continue
            return None
        if not candidates:
            return None
        return tuple(candidates), factor

    @staticmethod
    def _pointwise_node_matches_domain(
        node: SchedulerNode,
        expected_numel: sympy.Expr,
        expected_groups: Sequence[sympy.Expr],
    ) -> bool:
        from .codegen.simd import SIMDKernel

        _, (node_numel, node_rnumel) = node.group
        return (
            V.graph.sizevars.statically_known_equals(node_rnumel, 1)
            and V.graph.sizevars.statically_known_equals(node_numel, expected_numel)
            and SIMDKernel.is_compatible(expected_groups, node.get_ranges())
        )

    @staticmethod
    def try_get_sub_parent_extent_subs(
        parent_extent: sympy.Expr, factor: int
    ) -> dict[sympy.Expr, sympy.Expr] | None:
        """Return extent substitutions, or ``None`` if divisibility is unproven.

        On success, the mapping may be empty if the extent is already normalized.
        """
        rounded_extent = V.graph.sizevars.simplify(
            factor * FloorDiv(parent_extent, factor)
        )
        if not V.graph.sizevars.statically_known_equals(parent_extent, rounded_extent):
            return None
        return (
            {parent_extent: rounded_extent} if parent_extent != rounded_extent else {}
        )

    @classmethod
    def interleaved_sub_parent_lane(
        cls,
        index: sympy.Expr,
        factor: int,
        extent_subs: dict[sympy.Expr, sympy.Expr],
        source_sizes: tuple[sympy.Expr, ...] = (),
    ) -> sympy.Expr:
        # Identity controls emitted index width but does not change the lane.
        index = index.replace(Identity, lambda x: x)
        extent_subs = dict(extent_subs)
        expanded_sizes = [
            V.graph.sizevars.simplify(floor_div.args[1] * floor_div)
            for floor_div in index.atoms(FloorDiv)
        ]
        expanded_sizes.extend(
            V.graph.sizevars.simplify(sympy_product(source_sizes[i:]))
            for i in range(len(source_sizes))
        )
        for symbol in index.free_symbols:
            if symbol_is_type(symbol, SymT.SIZE):
                for expanded_size in expanded_sizes:
                    if (
                        symbol != expanded_size
                        and V.graph.sizevars.statically_known_multiple_of(
                            expanded_size, factor
                        )
                        and V.graph.sizevars.statically_known_equals(
                            symbol, expanded_size
                        )
                    ):
                        extent_subs[symbol] = expanded_size
                        break
                symbol_subs = cls.try_get_sub_parent_extent_subs(symbol, factor)
                if symbol_subs is not None:
                    extent_subs.update(symbol_subs)
        return V.graph.sizevars.simplify(
            sympy.Mod(sympy_subs(index, extent_subs), factor)
        )

    @classmethod
    def _try_get_sub_parent_source_layouts(
        cls,
        parent_nodes: Sequence[SchedulerNode],
        epilogue_nodes: Sequence[SchedulerNode],
        parent_numel: sympy.Expr,
        parent_rnumel: sympy.Expr,
        parent_source_names: OrderedSet[str],
        sub_parent_factor: int,
    ) -> tuple[tuple[str, NestedReduction.SubParentSourceLayout], ...] | None:
        """Find parent inputs reused by supported lane-resolution projections.

        Independent epilogue inputs remain ordinary kernel dependencies. Shared
        indexed parent inputs must have one normalized parent access, and every
        indexed epilogue access must equal it under
        ``parent_r = factor * child_r + lane``.
        """
        from .utils import sympy_index_symbol

        # Normalize source reads into a common two-axis domain.
        def normalized_source_read_indices(
            nodes: Sequence[SchedulerNode],
            source_names: OrderedSet[str],
            var_names: tuple[sympy.Symbol, sympy.Symbol],
            sizes: tuple[sympy.Expr, sympy.Expr],
        ) -> dict[str, OrderedSet[sympy.Expr]] | None:
            result: dict[str, OrderedSet[sympy.Expr]] = collections.defaultdict(
                OrderedSet
            )
            for node in nodes:
                for dep in node.read_writes.reads:
                    if isinstance(dep, MemoryDep) and dep.name in source_names:
                        normalized = dep.normalize_with_ranges(var_names, sizes)
                        if normalized is None:
                            return None
                        index = normalized.index.replace(Identity, lambda x: x)
                        result[dep.name].add(index)
            return result

        # Express epilogue reads in the derived child domain.
        child_rnumel = FloorDiv(parent_rnumel, sub_parent_factor)
        extent_subs = cls.try_get_sub_parent_extent_subs(
            parent_rnumel, sub_parent_factor
        )
        if extent_subs is None:
            return None
        x = sympy_index_symbol("_sub_parent_x")
        parent_r = sympy_index_symbol("_sub_parent_r")
        child_r = sympy_index_symbol("_sub_parent_child_r")
        child_indices = normalized_source_read_indices(
            epilogue_nodes,
            parent_source_names,
            (x, child_r),
            (parent_numel, child_rnumel),
        )
        if child_indices is None:
            return None
        # Express the corresponding source reads in the full parent domain.
        parent_indices = normalized_source_read_indices(
            parent_nodes,
            OrderedSet(child_indices),
            (x, parent_r),
            (parent_numel, parent_rnumel),
        )
        if parent_indices is None:
            return None

        source_layouts: list[tuple[str, NestedReduction.SubParentSourceLayout]] = []
        # Prove every child read is a lane projection of one parent read.
        for name in parent_source_names:
            source_child_indices = child_indices.get(name)
            if not source_child_indices:
                continue
            # TODO: Extend #188180's structural load-index cache to support
            # multiple parent accesses to one source buffer.
            source_parent_indices = parent_indices.get(name)
            if source_parent_indices is None or len(source_parent_indices) != 1:
                return None
            parent_index = sympy_subs(next(iter(source_parent_indices)), extent_subs)
            for child_index in source_child_indices:
                lane = cls.interleaved_sub_parent_lane(
                    child_index,
                    sub_parent_factor,
                    extent_subs,
                    (parent_numel, parent_rnumel),
                )
                if not any(
                    V.graph.sizevars.statically_known_equals(lane, value)
                    for value in range(sub_parent_factor)
                ):
                    return None
                expected = parent_index.subs(
                    parent_r, sub_parent_factor * child_r + lane
                )
                if not V.graph.sizevars.statically_known_equals(child_index, expected):
                    return None
            source_layouts.append((name, cls.SubParentSourceLayout.INTERLEAVED))
        return tuple(source_layouts)

    @staticmethod
    def _sub_parent_epilogue_outputs_unread(
        nodes: Sequence[BaseSchedulerNode],
        epilogue_node_set: OrderedSet[SchedulerNode],
    ) -> bool:
        """Whether every epilogue output is a leaf within the group.

        Epilogue outputs only exist at lane resolution, so a non-epilogue node
        reading one would need a parent-resolution view that was never emitted.
        TODO: Support consumers in a post-sub-parent stage.
        """
        epilogue_output_names = OrderedSet(
            name for node in epilogue_node_set for name in node.get_buffer_names()
        )
        if not epilogue_output_names:
            return True
        for node in nodes:
            if node in epilogue_node_set:
                continue
            for dep in node.read_writes.reads:
                if dep.name in epilogue_output_names:
                    return False
        return True

    @classmethod
    def _get_grouped_reduction_and_size(
        cls, grouped_node: BaseSchedulerNode, grouped_rnumel: sympy.Expr
    ) -> tuple[SchedulerNode, sympy.Integer] | None:
        """Validate the candidate as a single simple grouped reduction."""
        if not grouped_node.is_reduction():
            return None
        reductions = [sn for sn in grouped_node.get_nodes() if sn.is_reduction()]
        if len(reductions) != 1:
            return None
        reduction = reductions[0]
        if not isinstance(reduction, SchedulerNode) or not isinstance(
            reduction.node, ComputedBuffer
        ):
            return None
        iter_ranges, reduce_ranges = reduction.get_ranges()
        # The original tensor may be higher-rank, but scheduler ranges are
        # already collapsed into loop trees. Nested codegen maps the grouped
        # reduction as either [outer_x, r_groups, G] or [x_groups, outer_r, G].
        # A singleton non-reduced tree may be squeezed out; additional trees
        # need explicit axis provenance and are rejected here.
        if len(iter_ranges) not in (1, 2) or len(reduce_ranges) != 1:
            return None
        # Single-value reductions are lowered by reshape + tl.<op>(..., axis).
        # Tuple/stateful reductions such as argmax and Welford need the normal
        # reduction lowering and are rejected before nested codegen.
        if reduction.node.get_reduction_type() not in {
            "any",
            "max",
            "min",
            "prod",
            "sum",
            "xor_sum",
        }:
            return None

        group_size = V.graph.sizevars.simplify(grouped_rnumel)
        if not isinstance(group_size, (int, sympy.Integer)) or int(group_size) < 1:
            return None
        return reduction, sympy.Integer(group_size)

    @classmethod
    def _classify_nested_pointwise_nodes(
        cls,
        outer_node: BaseSchedulerNode,
        grouped_node: BaseSchedulerNode,
        domain_context: PointwiseDomainContext,
    ) -> list[tuple[SchedulerNode, PointwiseDomain]] | None:
        outer_pointwise_domains: list[
            tuple[SchedulerNode, NestedReduction.PointwiseDomain]
        ] = []
        # Use ancestor names to find pointwise subnodes inside the outer fused
        # node that consume an outer reduction result.
        outer_reduction_names: OrderedSet[str] = OrderedSet()
        for sn in outer_node.get_nodes():
            if sn.is_reduction():
                outer_reduction_names |= sn.get_operation_names()

        # Full-resolution consumers already fused into the outer reduction feed
        # the local reduction, so they run in that input domain.
        for sn in outer_node.get_nodes():
            if sn.is_reduction():
                continue
            if outer_reduction_names & sn.ancestors:
                if not isinstance(sn, SchedulerNode):
                    return None
                outer_pointwise_domains.append(
                    (sn, cls.PointwiseDomain.LOCAL_REDUCTION_INPUT)
                )

        grouped_pointwise_domains = cls._classify_grouped_pointwise_nodes(
            domain_context,
            grouped_node.get_nodes(),
        )
        if grouped_pointwise_domains is None:
            return None
        return [*outer_pointwise_domains, *grouped_pointwise_domains]

    @classmethod
    def _classify_grouped_pointwise_nodes(
        cls,
        domain_context: PointwiseDomainContext,
        nodes: Sequence[BaseSchedulerNode],
    ) -> list[tuple[SchedulerNode, PointwiseDomain]] | None:
        """Classify pointwise nodes relative to the grouped reduction.

        A node must be on exactly one side of the grouped reduction: either a
        producer feeding its local-group body, or a consumer of its reduced
        output. Its numel then determines whether it runs at reduced,
        grouped-full, or parent-full resolution.
        """
        grouped_reduction = domain_context.grouped_reduction
        reduction_names = grouped_reduction.get_operation_names()
        reduction_buffer_names = grouped_reduction.get_buffer_names()
        full_numel = V.graph.sizevars.simplify(
            domain_context.grouped_numel * domain_context.grouped_rnumel
        )
        pointwise_domains: list[
            tuple[SchedulerNode, NestedReduction.PointwiseDomain]
        ] = []
        for sn in nodes:
            if sn.is_reduction():
                continue
            if not isinstance(sn, SchedulerNode):
                return None

            sn_names = sn.get_operation_names()
            is_producer = bool(sn_names & grouped_reduction.ancestors)
            is_consumer = bool(reduction_names & sn.ancestors) or any(
                dep.name in reduction_buffer_names for dep in sn.read_writes.reads
            )
            if is_producer and is_consumer:
                # Supportable by splitting/modeling a multi-stage pointwise,
                # but not as one nested pipeline stage today.
                return None
            if not is_producer and not is_consumer:
                # Supportable as a sidecar, but nested codegen does not yet
                # model an insertion point for unrelated pointwise nodes.
                return None

            full_domain = (
                cls.PointwiseDomain.LOCAL_REDUCTION_INPUT
                if is_producer
                else cls.PointwiseDomain.PARENT_FULL
            )
            _, (sn_numel, _) = sn.group
            if V.graph.sizevars.statically_known_equals(
                sn_numel, domain_context.grouped_numel
            ):
                domain = cls.PointwiseDomain.REDUCED
            elif V.graph.sizevars.statically_known_equals(sn_numel, full_numel):
                domain = full_domain
            else:
                return None
            pointwise_domains.append((sn, domain))
        return pointwise_domains

    @classmethod
    def _pointwise_domains_are_compatible(
        cls,
        domain_context: PointwiseDomainContext,
        pointwise_domains: Sequence[tuple[SchedulerNode, PointwiseDomain]],
    ) -> bool:
        return all(
            cls._pointwise_domain_is_compatible(sn, domain, domain_context)
            for sn, domain in pointwise_domains
        )

    @classmethod
    def _pointwise_domain_is_compatible(
        cls,
        sn: SchedulerNode,
        domain: PointwiseDomain,
        domain_context: PointwiseDomainContext,
    ) -> bool:
        iter_ranges, _ = domain_context.grouped_reduction.get_ranges()
        if domain is cls.PointwiseDomain.REDUCED:
            expected_numel = domain_context.grouped_numel
            expected_groups: Sequence[sympy.Expr] = tuple(iter_ranges)
        elif domain is cls.PointwiseDomain.LOCAL_REDUCTION_INPUT:
            expected_numel = V.graph.sizevars.simplify(
                domain_context.grouped_numel * domain_context.grouped_rnumel
            )
            expected_groups = domain_context.local_reduction_domain
        else:
            if domain is not cls.PointwiseDomain.PARENT_FULL:
                raise AssertionError(f"expected PARENT_FULL domain, got {domain}")
            expected_numel = V.graph.sizevars.simplify(
                domain_context.grouped_numel * domain_context.grouped_rnumel
            )
            expected_groups = domain_context.parent_full_domain
        return cls._pointwise_node_matches_domain(sn, expected_numel, expected_groups)

    @classmethod
    def _min_block_unprofitable_for_kernel(
        cls,
        outer_node: BaseSchedulerNode,
        outer_numel: sympy.Expr,
        outer_rnumel: sympy.Expr,
        *,
        grouped_axis: GroupedAxis,
        group_size: int,
    ) -> bool:
        from .codegen.simd import SIMDScheduling

        if not isinstance(outer_node, (SchedulerNode, FusedSchedulerNode)):
            return True
        coalesce_analysis = (
            outer_node.get_coalesce_analysis()
            if config.triton.coalesce_tiling_analysis
            else None
        )
        node_schedule = list(outer_node.get_nodes())
        tiling = SIMDScheduling.select_tiling(
            node_schedule,
            outer_numel,
            outer_rnumel,
            coalesce_analysis,
        )
        # TODO: fold richer profitability/coalescing policy into this guard.
        # For now this is only a codegen capability check for the min block.
        # The grouped reduction forces a minimum block on the split axis. Today
        # that floor is only modeled for ordinary x/r0 kernels, where x is the
        # outer pointwise axis and r0_ is the inner reduction axis.
        if OrderedSet(tiling) != OrderedSet(("x", "r0_")):
            return True
        if not (
            V.graph.sizevars.statically_known_equals(tiling["x"], outer_numel)
            and V.graph.sizevars.statically_known_equals(tiling["r0_"], outer_rnumel)
        ):
            return True
        return group_size > cls._max_min_block_group_size(node_schedule, grouped_axis)

    @classmethod
    def _max_min_block_group_size(
        cls,
        node_schedule: Sequence[BaseSchedulerNode],
        grouped_axis: GroupedAxis,
    ) -> int:
        from .codegen.simd_kernel_features import SIMDKernelFeatures

        reduction_hints = [
            SIMDKernelFeatures.reduction_hint(sn)
            for sn in node_schedule
            if isinstance(sn, SchedulerNode)
            and sn.is_reduction()
            and isinstance(sn.node, ComputedBuffer)
        ]
        if (
            grouped_axis is cls.GroupedAxis.R
            and reduction_hints
            and all(hint is ReductionHint.INNER for hint in reduction_hints)
        ):
            return cls.MAX_INNER_R_GROUP_SIZE
        return cls.MAX_NON_INNER_GROUP_SIZE

    @classmethod
    def plan(
        cls,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> StagedReductionPlan | None:
        """Plan a dependent cross-axis reduction pair for staged code generation."""
        if not cls._is_enabled_for(node1, node2):
            return None

        if not cls._is_dependent_reduction_pair(node1, node2):
            raise AssertionError(
                "expected node1 and node2 to be a dependent reduction pair"
            )

        outer_node = node1
        grouped_node = node2
        outer_group: tuple[sympy.Expr, sympy.Expr]
        grouped_group: tuple[sympy.Expr, sympy.Expr]
        _, outer_group = outer_node.group  # pyrefly: ignore [bad-assignment]
        _, grouped_group = grouped_node.group  # pyrefly: ignore [bad-assignment]
        outer_numel, outer_rnumel = outer_group
        grouped_numel, grouped_rnumel = grouped_group
        if V.graph.sizevars.statically_known_equals(outer_rnumel, grouped_rnumel):
            return None

        # The grouped candidate must be one small block-local reduction.
        # We specialize the grouped reduction on that block size, so it must
        # simplify to an exact static int. Multiple reductions in the grouped
        # candidate (e.g. amax AND sum from the same input) are not supported.
        grouped_reduction_info = cls._get_grouped_reduction_and_size(
            grouped_node, grouped_rnumel
        )
        if grouped_reduction_info is None:
            return None

        # Total-element equality also implies divisibility of the grouped
        # parent extent by group_size (grouped_total factors through
        # FloorDiv(extent, group_size) * group_size). No separate Mod check
        # is needed. The sizevars divisible set records Mod(s, s//G) from
        # the view op, not Mod(s, G), so statically_known_equals on
        # Mod(extent, group_size) would fail with dynamic shapes.
        # TODO: teach sizevars to infer Mod(s, G)==0 from Mod(s, s//G)==0
        outer_total = V.graph.sizevars.simplify(outer_numel * outer_rnumel)
        grouped_total = V.graph.sizevars.simplify(grouped_numel * grouped_rnumel)
        if not V.graph.sizevars.statically_known_equals(outer_total, grouped_total):
            return None

        grouped_reduction, group_size = grouped_reduction_info

        grouped_axis = cls.get_grouped_axis(
            grouped_reduction,
            outer_numel,
            outer_rnumel,
            group_size,
            outer_node=outer_node,
        )
        if grouped_axis is None:
            return None
        parent_grouped_axis = (
            outer_rnumel if grouped_axis is cls.GroupedAxis.R else outer_numel
        )
        iter_ranges, _ = grouped_reduction.get_ranges()
        if len(iter_ranges) == 2:
            grouped_axis_groups = (
                iter_ranges[1] if grouped_axis is cls.GroupedAxis.R else iter_ranges[0]
            )
            if not V.graph.sizevars.statically_known_equals(
                FloorDiv(parent_grouped_axis, group_size), grouped_axis_groups
            ):
                return None
        elif not V.graph.sizevars.statically_known_equals(
            sympy.Mod(parent_grouped_axis, group_size), 0
        ):
            return None
        group_size_int = int(group_size)
        if not (1 <= group_size_int and is_power_of_2(group_size_int)):
            return None
        if cls._min_block_unprofitable_for_kernel(
            outer_node,
            outer_numel,
            outer_rnumel,
            grouped_axis=grouped_axis,
            group_size=group_size_int,
        ):
            return None

        return cls.plan_from_topology(
            outer_node,
            grouped_node,
            grouped_reduction,
            group_size,
            grouped_axis,
        )

    @classmethod
    def plan_from_topology(
        cls,
        outer_node: BaseSchedulerNode,
        grouped_node: BaseSchedulerNode,
        grouped_reduction: SchedulerNode,
        group_size: sympy.Integer,
        grouped_axis: GroupedAxis,
    ) -> StagedReductionPlan | None:
        """Rebuild mutable domains for an approved nested topology.

        ``merge_loops`` rewrites loop bodies after fusion, so grouped-axis
        discovery can no longer recover every axis approved at fusion time.
        The axis and group size remain stable; ranges and domains do not.
        """
        _, (outer_numel, outer_rnumel) = outer_node.group
        _, (grouped_numel, grouped_rnumel) = grouped_node.group
        iter_ranges, reduce_ranges = grouped_reduction.get_ranges()
        # Use PointwiseDomainContext.create if one is added; direct construction
        # must not bypass derived sub-parent domains.
        domain_context = cls.PointwiseDomainContext(
            grouped_reduction=grouped_reduction,
            grouped_numel=grouped_numel,
            grouped_rnumel=grouped_rnumel,
            local_reduction_domain=(*iter_ranges, *reduce_ranges),
            parent_full_domain=(outer_numel, outer_rnumel),
        )
        pointwise_domains = cls._classify_nested_pointwise_nodes(
            outer_node,
            grouped_node,
            domain_context,
        )
        if pointwise_domains is None or not cls._pointwise_domains_are_compatible(
            domain_context, pointwise_domains
        ):
            return None
        local_reduction_input_nodes = OrderedSet(
            node
            for node, domain in pointwise_domains
            if domain is cls.PointwiseDomain.LOCAL_REDUCTION_INPUT
        )

        return StagedReductionPlan(
            parent_nodes=tuple(
                sn
                for sn in outer_node.get_nodes()
                if sn not in local_reduction_input_nodes
            ),
            parent_numel=outer_numel,
            parent_rnumel=outer_rnumel,
            nested_stage=NestedReductionStage(
                group_size=group_size,
                grouped_axis=grouped_axis,
                grouped_nodes=tuple(grouped_node.get_nodes()),
                domain_context=domain_context,
                pointwise_domains=tuple(pointwise_domains),
            ),
            sub_parent_stages=(),
        )

    @classmethod
    def can_fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        return cls.plan(node1, node2) is not None

    @classmethod
    def get_grouped_axis(
        cls,
        grouped_reduction: SchedulerNode,
        outer_numel: sympy.Expr,
        outer_rnumel: sympy.Expr,
        group_size: sympy.Expr,
        *,
        outer_node: BaseSchedulerNode | None = None,
    ) -> GroupedAxis | None:
        """Return which parent axis is split by the grouped local reduction."""
        sizevars = V.graph.sizevars
        iter_ranges, reduce_ranges = grouped_reduction.get_ranges()
        if len(iter_ranges) != 2 or len(reduce_ranges) != 1:
            if len(iter_ranges) == 1 and len(reduce_ranges) == 1:
                if not sizevars.statically_known_equals(reduce_ranges[0], group_size):
                    return None
                # [R / G] reduce G: parent X is singleton, so the local group
                # splits the parent R axis.
                if sizevars.statically_known_equals(
                    FloorDiv(outer_rnumel, group_size), iter_ranges[0]
                ) and sizevars.statically_known_equals(outer_numel, 1):
                    return cls.GroupedAxis.R
                # [R] reduce G: parent X is exactly one local group and is
                # reduced away, leaving the parent R axis as the only iterator.
                if sizevars.statically_known_equals(
                    FloorDiv(outer_numel, group_size), 1
                ) and sizevars.statically_known_equals(iter_ranges[0], outer_rnumel):
                    return cls.GroupedAxis.X
            return None
        if not sizevars.statically_known_equals(reduce_ranges[0], group_size):
            return None
        # 2D group-in-R form: [X, R / G] reduce G. Dynamic view guards often
        # preserve the quotient dimension even when Mod(R, G) is not directly
        # provable.
        if sizevars.statically_known_equals(
            iter_ranges[0], outer_numel
        ) and sizevars.statically_known_equals(
            FloorDiv(outer_rnumel, group_size), iter_ranges[1]
        ):
            return cls.GroupedAxis.R
        # 2D group-in-X form: [X / G, R] reduce G.
        if sizevars.statically_known_equals(
            iter_ranges[1], outer_rnumel
        ) and sizevars.statically_known_equals(
            FloorDiv(outer_numel, group_size), iter_ranges[0]
        ):
            return cls.GroupedAxis.X
        if outer_node is not None:
            return cls._get_grouped_axis_from_loop_body(outer_node, grouped_reduction)
        return None

    @classmethod
    def _get_grouped_axis_from_loop_body(
        cls, outer_node: BaseSchedulerNode, grouped_reduction: SchedulerNode
    ) -> GroupedAxis | None:
        """Use LoopBody iter/reduce vars to disambiguate equal-size axes."""
        from .loop_body import MemoryUsageType

        outer_reductions = [sn for sn in outer_node.get_nodes() if sn.is_reduction()]
        if len(outer_reductions) != 1:
            return None
        outer_reduction = typing.cast(SchedulerNode, outer_reductions[0])
        outer_body = getattr(outer_reduction, "_body", None)
        grouped_body = getattr(grouped_reduction, "_body", None)
        if outer_body is None or grouped_body is None:
            return None

        outer_iter_ranges, outer_reduce_ranges = outer_reduction.get_ranges()
        grouped_iter_ranges, grouped_reduce_ranges = grouped_reduction.get_ranges()
        if len(outer_reduce_ranges) != 1 or len(grouped_reduce_ranges) != 1:
            return None

        def load_exprs_by_name(body: LoopBody) -> dict[str, list[sympy.Expr]]:
            result: dict[str, list[sympy.Expr]] = defaultdict(list)
            for entry in body.memory_usage.get(MemoryUsageType.LOAD, ()):
                if entry.buffer_name is not None:
                    result[entry.buffer_name].append(
                        body.indexing_exprs[entry.index_name]
                    )
            return result

        if len(outer_body.reduce_vars) != 1 or len(grouped_body.reduce_vars) != 1:
            return None
        if len(outer_body.iter_vars) != len(outer_iter_ranges) or len(
            grouped_body.iter_vars
        ) != len(grouped_iter_ranges):
            return None

        outer_reads_by_name = load_exprs_by_name(outer_body)
        result: NestedReduction.GroupedAxis | None = None
        grouped_reduce_var = grouped_body.reduce_vars[-1]
        for name, grouped_read_exprs in load_exprs_by_name(grouped_body).items():
            outer_read_exprs = outer_reads_by_name.get(name)
            if not outer_read_exprs:
                continue
            for grouped_read_expr in grouped_read_exprs:
                grouped_coeff = grouped_read_expr.coeff(grouped_reduce_var)
                if grouped_coeff == 0:
                    continue
                for outer_read_expr in outer_read_exprs:
                    outer_reduce_var = outer_body.reduce_vars[-1]
                    outer_reduce_coeff = outer_read_expr.coeff(outer_reduce_var)
                    matches_reduction = grouped_coeff == outer_reduce_coeff
                    matches_iter = any(
                        grouped_coeff == outer_read_expr.coeff(var)
                        for var in outer_body.iter_vars
                    )
                    if matches_reduction == matches_iter:
                        continue
                    candidate = (
                        cls.GroupedAxis.R if matches_reduction else cls.GroupedAxis.X
                    )

                    if result is not None and result != candidate:
                        return None
                    result = candidate
        return result


class StagedReductionPlan:
    """How a reduction is to be written out, where it was cut into stages.

    A plan says which pieces make up the reduction, how much of it there is, and
    what was derived from it.  A plan with nothing derived is not a plan, and
    having both kinds of stage at once is not something that can be written out
    yet, so both are refused rather than half-honoured.
    """

    parent_nodes: tuple
    parent_numel: Any
    parent_rnumel: Any
    nested_stage: Any
    sub_parent_stages: tuple

    def __post_init__(self) -> None:
        if self.nested_stage is None and not self.sub_parent_stages:
            raise AssertionError("staged reduction plan must contain a derived stage")
        if self.nested_stage is not None and self.sub_parent_stages:
            raise AssertionError("combined staged reductions are not supported yet")
        if len(self.sub_parent_stages) > 1:
            raise AssertionError("multiple sub-parent stages are not supported yet")


class FusedMixOrderReductions(FusedSchedulerNode):
    """Two reductions over the same data, walking its axes in different orders."""

    def __init__(self, node1, node2) -> None:
        if not MixOrderReduction.is_contiguous_node(node1):
            if not MixOrderReduction.is_contiguous_node(node2):
                raise AssertionError("expected node2 to be a contiguous node")
            node1, node2 = node2, node1

        self.node1 = node1
        self.node2 = node2
        super().__init__(
            node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes())
        )
        self.numel = MixOrderReduction.get_numel(self.node1)

    def sub_node_can_fuse(self, node1, node2, other_nodes: tuple):
        """Whether one more piece may be joined to one of these two.

        ``other_nodes`` are the halves of the pair that are not being joined, and
        are what the check is made against: if the piece would make one half
        produce for the other, the two are no longer independent and there is
        nothing to share.
        """

        if isinstance(node1, FusedMixOrderReductions):
            raise AssertionError("expected node1 to not be a FusedMixOrderReductions")
        if isinstance(node2, FusedMixOrderReductions):
            raise AssertionError("expected node2 to not be a FusedMixOrderReductions")

        # A second pair of orders inside one of these would not be writable.
        if not self.scheduler.can_fuse(node1, node2, allow_mix_order_reduction=False):
            return False

        # Where the piece being joined reads in the order the data is laid out,
        # so must the result, or the share is lost.
        if MixOrderReduction.is_contiguous_node(
            node1
        ) and not MixOrderReduction.is_contiguous_node(node2):
            return False

        def _get_ancestors(nodes: tuple) -> OrderedSet:
            out = OrderedSet()
            return out.union(*(n.ancestors for n in nodes))

        def _get_operation_names(nodes: tuple) -> OrderedSet:
            out = OrderedSet()
            return out.union(*(n.get_operation_names() for n in nodes))

        if other_nodes:
            if (_get_ancestors((node1, node2)) & _get_operation_names(other_nodes)) or (
                _get_ancestors(other_nodes) & _get_operation_names((node1, node2))
            ):
                return False

        return (
            not node2.is_reduction()
            or self.scheduler.score_fusion_memory(node1, node2, count_bytes=False)
            >= self.numel
        )

    def can_fuse_with(self, other):
        if self.has_strict_reduction() or other.has_strict_reduction():
            return False
        # Too many loads in one loop body spills what is held in hand, so where
        # there is a limit on how many a joined pair may read, it is applied.
        max_reads = config.triton.mix_order_reduction_max_reads
        if max_reads > 0:
            all_reads: OrderedSet = OrderedSet()
            for sn in itertools.chain(self.get_nodes(), other.get_nodes()):
                for dep in sn.read_writes.reads:
                    if isinstance(dep, dependencies.MemoryDep):
                        all_reads.add(dep.name)
            if len(all_reads) > max_reads:
                metrics.rejected_mix_order_reduction_fusion += 1
                return False
        if not isinstance(other, FusedMixOrderReductions):
            return self.sub_node_can_fuse(
                self.node1, other, (self.node2,)
            ) or self.sub_node_can_fuse(self.node2, other, (self.node1,))
        else:
            # The two halves were already checked against each other, so only
            # the pairing is checked here.
            return self.sub_node_can_fuse(
                self.node1, other.node1, (self.node2, other.node2)
            ) and self.sub_node_can_fuse(self.node2, other.node2, tuple())

    def fuse_with(self, other):
        device = self.node1.get_device()
        backend = self.scheduler.get_backend(device)

        if isinstance(other, FusedMixOrderReductions):
            fused_node1 = backend.fuse(self.node1, other.node1)
            fused_node2 = backend.fuse(self.node2, other.node2)
            return FusedMixOrderReductions(fused_node1, fused_node2)
        else:
            if self.sub_node_can_fuse(self.node1, other, (self.node2,)):
                fused_node = backend.fuse(self.node1, other)
                return FusedMixOrderReductions(fused_node, self.node2)
            else:
                fused_node = backend.fuse(self.node2, other)
                return FusedMixOrderReductions(self.node1, fused_node)


class FusedStagedReduction(FusedSchedulerNode):
    """A joined reduction that has to be written out in stages."""


class FusedNestedReductions(FusedStagedReduction):
    """Two reductions over the same elements, one done in groups inside the other.

    The outer one decides the shape of the launch and the inner one is done
    within it, which is what lets the two run together at all.
    """

    def __init__(self, node1, node2, stage) -> None:
        self.node1 = node1
        self.node2 = node2
        super().__init__(
            node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes())
        )
        # Whether a piece may be joined is decided against the inner reduction,
        # while the joined group holds what both produce.  The two being related
        # as producer and consumer is therefore hidden from the checks that
        # look for such a relation, or a read from the outer one would be taken
        # for a read of something produced elsewhere.
        self.ancestors -= self.get_operation_names()
        self.grouped_reduction: SchedulerNode = stage.domain_context.grouped_reduction
        self.group_size: Any = stage.group_size
        self.grouped_axis: Any = stage.grouped_axis
        self.group_size_in_r: bool = self.grouped_axis is _nested().GroupedAxis.R
        self.domain_context: Any = stage.domain_context

    def can_fuse_with(self, other, *, can_reorder: bool) -> bool:
        """Whether work on the reduced values may be joined to the inner reduction.

        A piece joining the inner reduction directly has to be at the same
        resolution as the inner reduction's output or as the whole outer tile;
        anything between those two would be somewhere the values are not.
        """

        if other.is_reduction():
            return False
        # What is joined goes into the inner reduction, so a piece that does
        # not read it has no place to go.
        if not (self.node2.get_operation_names() & other.ancestors):
            return False

        pointwise_domains = _nested()._classify_grouped_pointwise_nodes(
            self.domain_context,
            other.get_nodes(),
        )
        if pointwise_domains is None:
            return False
        if not _nested()._pointwise_domains_are_compatible(
            self.domain_context, pointwise_domains
        ):
            return False
        # What the inner reduction reads has to be put in before it, and this
        # path only has somewhere to put what reads it.
        if any(
            domain is _nested().PointwiseDomain.LOCAL_REDUCTION_INPUT
            for _, domain in pointwise_domains
        ):
            return False
        return self.scheduler._can_fuse_nested_reduction_append(
            self.node2,
            other,
            pointwise_domains,
            can_reorder=can_reorder,
        )

    def fuse_with(self, other) -> "FusedNestedReductions":
        device = self.node2.get_device()
        backend = self.scheduler.get_backend(device)
        new_node2 = backend.fuse(self.node2, other)
        plan = _nested().plan(self.node1, new_node2)
        if plan is None or plan.nested_stage is None:
            raise AssertionError("expected appended nested reduction plan")
        return FusedNestedReductions(self.node1, new_node2, plan.nested_stage)


class FusedExternTritonKernelSchedulerNode(FusedSchedulerNode):
    """A hand-written kernel with work on its result joined to it.

    The kernel writes into memory it was given, and the work on that memory
    becomes part of the kernel rather than a launch of its own -- which is only
    possible where the kernel writes it in one place, so that what is written
    can be found and rewritten.
    """

    def __init__(self, scheduler, kernel_node, fused_epilogue) -> None:
        if not isinstance(kernel_node.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected kernel_node.node to be an ir.UserDefinedTritonKernel"
            )
        snodes: list = [kernel_node, fused_epilogue]
        super().__init__(scheduler, snodes)
        self.kernel_node = kernel_node
        self.fused_epilogue = fused_epilogue
        self.min_order = self.kernel_node.min_order
        self.outputs = fused_epilogue.outputs

    @classmethod
    def epilogue_fuse(
        cls,
        node1,
        node2,
    ) -> "FusedSchedulerNode":
        """Join work on a hand-written kernel's result into the kernel."""

        if not isinstance(node1.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected node1.node to be an ir.UserDefinedTritonKernel"
            )
        scheduler = node1.scheduler

        if len(node1.node.mutation_outputs) != 1:
            raise AssertionError(
                f"expected one mutation output, got {len(node1.node.mutation_outputs)}"
            )
        mutated_name: str = node1.node.mutation_outputs[0].name
        # What the kernel wrote becomes a value of its own, and the kernel is
        # no longer the last to write it -- so it is taken off the list of those
        # that are, and the value is left to be cleaned up with the rest.
        real_name = scheduler.mutation_real_name.get(mutated_name, mutated_name)
        scheduler.name_to_buf[real_name].users.remove(NodeUser(node1))
        return cls(scheduler, node1, node2)

    def codegen(self, wrapper) -> None:
        if not isinstance(self.fused_epilogue.node, ir.ComputedBuffer):
            raise AssertionError(
                "expected fused_epilogue.node to be an ir.ComputedBuffer"
            )
        if not isinstance(self.kernel_node.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected kernel_node.node to be an ir.UserDefinedTritonKernel"
            )
        if not self.kernel_node.node.can_fuse_epilogue():
            raise AssertionError("expected kernel_node.node to allow epilogue fusion")
        numel = math.prod(self.kernel_node.node.mutable_args[0].shape)
        tiling, _ = self.fused_epilogue.get_tiling(numel, sympy.S.One)
        kernel_features = _kernel_features()([self.fused_epilogue], numel)

        from .codegen.triton import FusedUserDefinedTritonKernel

        fused_user_kernel = FusedUserDefinedTritonKernel(tiling, kernel_features, self)
        new_kernel_src = fused_user_kernel.codegen()

        return self.kernel_node.node.codegen_with_epilogue_fusion(
            wrapper, (self.fused_epilogue.node, new_kernel_src)
        )

    def is_extern(self) -> bool:
        return True

    def get_ranges(self) -> Sequence:
        return self.kernel_node.get_ranges()


def _nested():
    """The nested-reduction machinery, which is asked about from several places."""

    return _NestedReductionModule()


class _NestedReductionModule:
    """A stand-in for the nested-reduction machinery while it is being written.

    The nested-reduction rules are answered from here so that the classes that
    refer to them can be written and read in one piece; each answer says what
    has and has not been settled rather than raising, so that the surrounding
    code is exercised while this is filled in.
    """

    class GroupedAxis:
        R = "r"
        I = "i"
        XY = "xy"

    class PointwiseDomain:
        LOCAL_REDUCTION_INPUT = "local_reduction_input"
        LOCAL_REDUCTION_OUTPUT = "local_reduction_output"
        PARENT_TILE_INPUT = "parent_tile_input"
        PARENT_TILE_OUTPUT = "parent_tile_output"

    @dataclasses.dataclass
    class PointwiseDomainContext:
        """What a piece of work on a reduction's inputs or outputs is relative to."""

        parent_size: Any
        grouped_reduction: Any = None
        allowed_domains: frozenset = frozenset()

    @staticmethod
    def _is_dependent_reduction_pair(node1, node2) -> bool:
        return False

    @staticmethod
    def plan(node1, node2):
        return None

    @staticmethod
    def _classify_grouped_pointwise_nodes(domain_context, nodes):
        return None

    @staticmethod
    def _pointwise_domains_are_compatible(domain_context, pointwise_domains) -> bool:
        return False

    @staticmethod
    def can_fuse(node1, node2) -> bool:
        return False

    @staticmethod
    def fuse(node1, node2):
        raise NotImplementedError


def _kernel_features():
    """What a kernel's shape and strides are, which is what a choice of division
    is measured against.

    A division is worth what the accesses it produces are worth, and that is
    worked out from how the values are laid out rather than from the arithmetic.
    """

    from .codegen.simd_kernel_features import SIMDKernelFeatures

    return SIMDKernelFeatures


def _is_gpu_triton_backend(node1, node2) -> bool:
    """Whether these two pieces run on a machine where sharing one read is possible."""

    if not node1.is_gpu() or not node2.is_gpu():
        return False
    device_type = node1.get_device().type
    return device_type in ("cuda", "xpu") and get_current_backend(device_type) == "triton"




class BaseScheduling:
    """What a backend has to say about how its work may be arranged.

    Deciding whether two pieces may be joined, and in what order the joined
    pieces run, depends on the machine they run on: what reads can be done
    efficiently, how much memory is worth holding, what the hardware is like.
    None of that is the same for any two machines, so it is asked of the thing
    that knows the machine, and this is what it is asked.

    A backend answers the questions that apply to it and leaves the rest to
    raise, which is what says the arrangement is not one it supports.
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def free_buffers_in_scheduler(self) -> None:
        """Give back the memory of any value nothing is holding on to.

        Only worth doing once the order is settled, since what nothing is
        holding on to changes as the order does.
        """

        if self.scheduler:
            self.scheduler.free_buffers()

    def get_backend_features(self, device) -> OrderedSet:
        """What this backend can do on this device."""

        return OrderedSet()

    def has_sub_parent_epilogue(self, nodes: Sequence) -> bool:
        """Whether these pieces are written out by a two-stage emitter.

        Putting several pieces together and re-opening the result goes through
        the ordinary scheduling machinery, which cannot describe what such an
        emitter produces and would stop the compilation.  A backend that uses
        one says so here, so that those paths leave the piece alone instead.  What
        is lost is a possible grouping or a measurement, not the joining.
        """

        return False

    def can_fuse_vertical(self, node1, node2) -> bool:
        """Whether the second may be written as part of the first.

        The two walk the same axes, one inside the other, so what the second
        computes for one position of the first is available where the first is
        being computed.
        """

        raise NotImplementedError

    def can_fuse_horizontal(self, node1, node2) -> bool:
        """Whether two pieces walking the same axes may become one.

        What each computes is then done for the same position in turn, and
        neither needs the other's result to do its own work.
        """

        raise NotImplementedError

    def can_fuse_reduction_epilogue(self, node1, node2) -> bool:
        """Whether work on the reduced values may be folded into the reduction.

        A reduction produces its result at one position rather than one per
        element, so work on that result cannot simply be appended to the loop.
        """

        return False

    def can_fuse_multi_outputs_template(self, node1, node2) -> bool:
        """Whether a piece may be folded into a template that produced several results.

        A template of that kind leaves its results in memory it was given rather
        than returning them, so a piece that reads one of those results is
        reading what the template wrote -- which only works where the second
        piece is exactly that result and nothing else.
        """

        template_buf = node1.get_template_node()
        if not isinstance(template_buf, ir.TemplateBuffer):
            return False
        if not template_buf.is_multi_outputs_template():
            return False

        if isinstance(node2.node, ir.MultiOutput):
            return (
                len(node2.node.inputs) == 1
                and isinstance(node2.node.inputs[0], ir.IRNode)
                and node2.node.inputs[0].get_name() == template_buf.get_name()
            )

        return False

    def fuse(self, node1, node2) -> "FusedSchedulerNode":
        """Join two pieces, in whichever way suits what they are.

        A set joins as a set, a reduction that walks its axes in a different
        order from another joins as a plan for both, and anything else joins as
        an ordinary group.  Which of those applies is a property of the pieces,
        not a choice, so it is looked up rather than configured.
        """

        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        elif (
            NestedReduction._is_dependent_reduction_pair(node1, node2)
            and (plan := NestedReduction.plan(node1, node2)) is not None
        ):
            if plan.nested_stage is None:
                raise AssertionError("expected nested reduction stage")
            return FusedNestedReductions(node1, node2, plan.nested_stage)
        elif MixOrderReduction.are_mix_order_reductions(node1, node2):
            return FusedMixOrderReductions(node1, node2)
        elif isinstance(node1, FusedNestedReductions):
            return node1.fuse_with(node2)
        elif isinstance(node1, FusedMixOrderReductions):
            return node1.fuse_with(node2)
        elif isinstance(node1, ExternKernelSchedulerNode) and isinstance(
            node2, SchedulerNode
        ):
            if not isinstance(node1.node, ir.UserDefinedTritonKernel):
                raise AssertionError(
                    "expected node1.node to be an ir.UserDefinedTritonKernel"
                )
            return FusedExternTritonKernelSchedulerNode.epilogue_fuse(node1, node2)
        else:
            nodes = [*node1.get_nodes(), *node2.get_nodes()]
            staged = isinstance(node1, FusedStagedReduction) or isinstance(
                node2, FusedStagedReduction
            )
            device = node1.get_device()
            if not staged and device is not None:
                staged = self.has_sub_parent_epilogue(nodes)
            node_type = FusedStagedReduction if staged else FusedSchedulerNode
            return node_type.fuse(node1, node2)

    def group_fn(self, sizes: Sequence) -> tuple:
        """What makes two pieces able to run as one, given the axes they walk.

        Two pieces can only be joined if they walk the same axes, so this says
        what has to match -- which is a different question on each machine, and
        is what a reduction's axes look like is a backend's business.
        """

        raise NotImplementedError

    def codegen_template(
        self,
        template_node,
        epilogue_nodes: Sequence,
        prologue_nodes: Sequence,
    ):
        """Write out a template with work folded into its head and tail.

        Only some backends have templates at all, and a third-party one that
        has them may write its own or reuse this.
        """

        raise NotImplementedError

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence,
        benchmark_kernel: bool,
        hint_override: int | None = None,
    ) -> str:
        """Write out a kernel for pieces that have already been joined."""

        raise NotImplementedError

    def codegen_node(self, node) -> None:
        """Write out the kernel for one piece or one group."""

        raise NotImplementedError

    def codegen_mix_order_reduction(self, node) -> None:
        """Write out two reductions whose axes are walked in different orders."""

        raise NotImplementedError

    def codegen_staged_reduction(self, node) -> None:
        """Write out a reduction that was cut into stages."""

        raise NotImplementedError

    def codegen_sync(self) -> None:
        """Write out whatever has to happen between kernels.

        What that is depends on the hardware: on some machines nothing, on
        others a wait for one kernel to finish before the next may read what it
        wrote.
        """

        raise NotImplementedError

    def ready_to_flush(self) -> bool:
        """Whether what has been written out so far should be written to a file.

        A backend that would rather accumulate more before writing says so.
        """

        return False

    def flush(self) -> None:
        """Write out what has been accumulated."""

        raise NotImplementedError

    def benchmark_fused_nodes(self, nodes: Sequence):
        """How long the joined pieces take, measured on values made up for it.

        Returns the time in milliseconds and the code that was measured, since
        the second is what makes the first checkable.
        """

        raise NotImplementedError

    def benchmark_codegened_module(self, module):
        """How long a whole compiled thing takes, measured on made-up values."""

        raise NotImplementedError

    def get_fusion_pair_priority(self, node1, node2) -> int:
        """Which of two pieces to try to join first.

        A smaller number is tried first.  Zero means no preference, which is the
        right answer where nothing about the two pieces suggests one order is
        better than the other.
        """

        return 0

    def benchmark_combo_kernel(self, node_list: Sequence, node_benchmark_results):
        """How long a set of independent pieces takes together, and what copying costs.

        The copying is counted separately because it is not the work itself: a
        set that is much faster to compute but much slower to get the values
        where they are needed may not be worth having.
        """

        raise NotImplementedError

    def codegen_comment(
        self,
        node_schedule: Sequence,
        kernel_name: str | None = None,
    ) -> None:
        """Say in the code which work this kernel is, and where it came from."""

        if kernel_name:
            from .debug import set_kernel_post_grad_provenance_tracing

            debug_handle = set_kernel_post_grad_provenance_tracing(
                node_schedule,  # type: ignore[arg-type]
                kernel_name,
            )
            V.graph.wrapper_code.write_provenance_debug_handle(
                kernel_name, debug_handle
            )


class Scheduler:
    """The graph of pieces of work, and everything decided about them.

    Lowering has produced a list of results, each written by one piece of work.
    What this decides is which of those pieces share a kernel, what order the
    kernels run in, and how much memory is held while they do.  Those three are
    related -- joining pieces changes how much is held, and the order changes
    what may be joined -- so they are decided together rather than one after
    another.

    The order the pieces came out in says only what must come before what.  From
    that, pieces that do not wait on each other may be joined, and the pieces
    that are left become the kernels.
    """

    def __init__(self, nodes: list) -> None:
        self._init(nodes)

    @staticmethod
    def count_kernel_nodes(nodes: Sequence) -> int:
        """How many of these actually become a kernel.

        Some pieces only arrange memory and produce no code of their own.
        """

        return sum(1 for node in nodes if not isinstance(node, NopKernelSchedulerNode))

    def _init(self, nodes: list) -> None:
        self._tiling_memory_cache: dict = {}
        super().__init__()
        V.graph.scheduler = self
        self.backends: dict = {}

        # What exists before anything is computed: values handed in, values
        # fixed while the code was written, and values bound from the program.
        # A piece that produces nothing of these can never run, since there
        # would be nothing for it to read.
        self.completed_operations: OrderedSet = OrderedSet()
        self.available_buffer_names = OrderedSet(
            [
                *V.graph.graph_inputs.keys(),
                *V.graph.constants.keys(),
                *getattr(V.graph, "torchbind_constants", {}).keys(),
            ]
        )
        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.previous_node = None
        self.current_node = None
        self.update_zero_dim_cpu_tensor()
        # Working out the shape of an empty value may have produced a constant.
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        self.default_device_context = None

        self.name_to_donated_buffer: dict = self.get_donated_buffers()

        self.name_to_node: dict = {n.get_name(): n for n in self.nodes}

        self.name_to_buf: dict = {
            buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
        }
        self.name_to_fused_node: dict = self.name_to_node.copy()

        # A value that is written over keeps the name it had, so that what
        # comes after refers to it by a name that still means something.  This
        # says what that name was.
        self.mutation_real_name: dict = {}

        # A write over a value would make the graph depend on itself -- the
        # value waits for what writes it, and what writes it waits for the
        # value -- so the write is given a name of its own.  This says, for
        # each value, the name its current version is known by, and changes
        # once per write.  Only the original name is ever written out.
        self.mutation_renames: dict = {}

        self.seen_template_fusions: OrderedSet = OrderedSet()

        self.compute_dependencies()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.dead_node_elimination()
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.compute_ancestors()
        self.compute_input_distances()

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        from .debug import log_ir_pre_fusion

        log_ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.nodes = self.fuse_nodes(self.nodes)

        self.merge_loops()
        self.finalize_multi_template_buffers()

        # Work that exchanges with other machines cannot be moved past the work
        # it exchanges, so that order is put back.
        self._enforce_switch_ordering()

        # The order is only settled once everything that could change it has
        # run, so the memory pass and the overlap pass come last: anything
        # earlier could undo what they decided.
        if config.reorder_for_peak_memory:
            from .memory import reorder_for_peak_memory

            self.nodes = reorder_for_peak_memory(
                self.nodes,
                self.name_to_buf,
                self.name_to_fused_node,
                OrderedSet(V.graph.graph_inputs.keys()),
                OrderedSet(V.graph.get_output_names()),
            )

        if config.combo_kernels:
            self.create_combo_kernel_nodes(num_ck_nodes=None)
            from .memory import assign_memory_planning_info_for_scheduler_buffers

            assign_memory_planning_info_for_scheduler_buffers(
                self.nodes, self.name_to_buf
            )

        self.compute_last_usage()

        if config.test_configs.track_memory_lifecycle:
            self.insert_memory_check_nodes()

        from .debug import log_ir_post_fusion

        log_ir_post_fusion(self.nodes)

        # Which values may be given back once nothing is holding them.
        self.buffer_names_to_free: OrderedSet = OrderedSet()

        # Where each value came from, for saying so in the generated code.
        self.origin_to_index: dict = {}

        # A piece that is gone from the order but whose value is still wanted,
        # produced some other way.
        self.removed_ops: OrderedSet = OrderedSet()

    def get_donated_buffers(self) -> dict:
        """The values the program handed over and said it was finished with.

        Such a value is not this program's to write into as a matter of course,
        but the program saying it is done with one is what makes its memory
        available to something else.
        """

        name_to_donated_buf = {}
        originals = getattr(V.graph, "graph_inputs_original", {})
        for name in originals:
            if isinstance(originals[name], ir.DonatedBuffer):
                name_to_donated_buf[name] = SchedulerDonatedBuffer(
                    self,
                    originals[name],
                    defining_op=None,
                )
        return name_to_donated_buf

    def get_backend(self, device) -> "BaseScheduling":
        """What is known about the machine this device is.

        Kept per device, since two devices may be two different machines.
        """

        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def create_backend(self, device) -> "BaseScheduling":
        """The machine-specific decisions for this device.

        A backend says how its work may be arranged; what it is depends on the
        kind of device, and this repository compiles for the host.
        """

        from .codegen.cpp import CppScheduling

        return CppScheduling(self)

    def create_scheduler_node(self, node) -> BaseSchedulerNode:
        """What kind of piece of work this is.

        Whether it computes something, is a call out to something already
        written, or only arranges memory decides what it can be asked and what
        it may be joined with.
        """

        if isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            return SchedulerNode(self, node)
        elif isinstance(node, ir.ExternKernelOut):
            return ExternKernelSchedulerNode(self, node)
        elif isinstance(node, ir.NopKernel):
            return NopKernelSchedulerNode(self, node)
        raise AssertionError(f"unexpected node type {type(node)}")

    def update_zero_dim_cpu_tensor(self) -> None:
        """Put back what a piece of work that produces nothing would have run.

        A computation over no elements produces a value of the right type and
        nothing else, and a kernel of no elements cannot produce even that -- so
        the value is made here instead, which also means a piece whose result is
        never read is not run at all.
        """

        for node in self.nodes:
            if node is None:
                continue
            try:
                if node.num_reads() == 0 and node.should_allocate():
                    device = node.get_device()
                    if device is not None and device.type == "cpu":
                        raise AssertionError("not reached")
            except AssertionError:
                raise
            except Exception:
                # The shape of the result is worked out here rather than by
                # running anything, since there is nothing to run.
                pass

    def compute_dependencies(self) -> None:
        """Work out which pieces must come after which, and why.

        A piece has to run after everything that writes what it reads, after
        everything that has read what it writes, and after every previous write
        of anything it overwrites.  Where two values are the same memory, one
        stands in for the other throughout, so that reading either is reading
        both.

        Writing a value that is also read is the awkward case: the write has to
        come after the reads, but those reads were waiting for the value the
        write produces.  Giving the write a name of its own is what breaks that
        circle, and remembering the original name is what lets the code written
        out refer to memory that exists.
        """

        class DedupList:
            """A list that will not hold the same thing twice.

            A set would do, except that entries are added while the list is
            being walked, which a set of the same objects does not allow.
            """

            def __init__(self, items=None, membership=None) -> None:
                self.items = items or []
                self.membership = membership or OrderedSet()

            def append(self, node_user) -> None:
                if node_user in self.membership:
                    return
                self.items.append(node_user)
                self.membership.add(node_user)

            def __add__(self, other) -> "DedupList":
                new_membership = OrderedSet.union(self.membership, other.membership)
                new_items = self.items + [
                    x for x in other.items if x not in self.membership
                ]
                return DedupList(new_items, new_membership)

        name_to_users: dict = collections.defaultdict(DedupList)

        # Two names for the same memory have to have the same readers, or a
        # piece reading one of them would be thought not to read the other.  So
        # the two names share one list outright.
        for node in self.nodes:
            for buf1 in node.get_outputs():
                buf1_name = buf1.get_name()
                # An operation that returns nothing and writes to more than one
                # input is not saying those inputs are the same memory, so they
                # are not made to share a list.
                if (
                    isinstance(buf1.node.layout, NoneLayout)
                    and len(buf1.get_aliases()) > 1
                ):
                    continue
                for buf2_name in buf1.get_aliases():
                    if buf1_name in name_to_users and buf2_name in name_to_users:
                        list1 = name_to_users[buf1_name]
                        list2 = name_to_users[buf2_name]
                        combined = list1 + list2
                        for key in name_to_users:
                            if (
                                name_to_users[key] is list1
                                or name_to_users[key] is list2
                            ):
                                name_to_users[key] = combined
                    elif buf1_name in name_to_users:
                        name_to_users[buf2_name] = name_to_users[buf1_name]
                    else:
                        name_to_users[buf1_name] = name_to_users[buf2_name]

        def rename(n: str) -> str:
            # A name may have been renamed more than once, since a value can
            # be written over repeatedly.
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def add_user(used_by_name, user_node, can_inplace: bool = False, is_weak=False):
            name_to_users[rename(used_by_name)].append(
                NodeUser(user_node, can_inplace, is_weak)
            )

        # A shape whose value comes from the data is not waited on through a
        # value, since there is no value to wait for -- so who first gave it a
        # value is tracked on its own.  Nothing means it came from outside.
        unbacked_symbol_to_origin_node: dict = {}

        for val in V.graph.graph_inputs.values():
            if isinstance(val, sympy.Expr):
                for fs in val.free_symbols:
                    unbacked_symbol_to_origin_node[fs] = None
            elif isinstance(val, ir.TensorBox):
                # The shapes of a value handed in count too, since they may
                # name shapes that are only settled once it arrives.
                sym_size = [s for s in val.get_size() if isinstance(s, sympy.Expr)]
                for s in sym_size:
                    for fs in s.free_symbols:
                        unbacked_symbol_to_origin_node[fs] = None

        has_non_input_unbacked_defs = False
        for node in self.nodes:
            if node.node is None:
                raise AssertionError("expected node.node to be set")
            unbacked_symbol_defs = sorted(
                node.node.get_unbacked_symbol_defs(), key=lambda x: x.name
            )
            for s in unbacked_symbol_defs:
                if not isinstance(s, sympy.Symbol):
                    raise AssertionError("expected s to be a sympy.Symbol")
                # Where several pieces claim to have settled a shape -- a piece
                # with several results may all carry the same one -- the first
                # is taken as the one that settles it.
                has_non_input_unbacked_defs = True
                if s not in unbacked_symbol_to_origin_node:
                    unbacked_symbol_to_origin_node[s] = node.get_name()

        for node in self.nodes:
            log.debug("scheduling %s", node.node)

            if has_non_input_unbacked_defs:
                if node.node is None:
                    raise AssertionError("expected self.node to be set")

                unbacked_symbol_uses = sorted(
                    node.node.get_free_symbol_uses(unbacked_only=True),
                    key=lambda x: x.name,
                )
                # A piece that reads a shape has to run after whatever settled
                # it, which is said as a dependency on everything that piece
                # produced.
                for s in unbacked_symbol_uses:
                    if s not in unbacked_symbol_to_origin_node:
                        raise AssertionError(
                            f"{s} not in {unbacked_symbol_to_origin_node}"
                        )
                    if (r := unbacked_symbol_to_origin_node[s]) is not None:
                        for buf in self.name_to_node[r].get_outputs():
                            node.add_fake_dep(dependencies.StarDep(buf.get_name()))

            if (
                len(node.read_writes.writes) == 1
                and (dep := next(iter(node.read_writes.writes)))
                and isinstance(dep, dependencies.MemoryDep)
            ):
                node_mode = dep.mode
            else:
                node_mode = None

            # Where a piece writes over a value.
            for buf in node.get_outputs():
                # One piece writes over one value, never more.
                if len(buf.get_mutations()) > 1:
                    raise AssertionError(
                        f"expected at most one mutation, got {len(buf.get_mutations())}"
                    )
                for alt_name in buf.get_mutations():
                    alt_name = rename(alt_name)
                    is_ordering_only = buf.is_ordering_only()
                    if is_ordering_only:
                        # A dependency that says only which comes first keeps
                        # nothing alive, so it is recorded as such.
                        add_user(alt_name, node, is_weak=True)
                        node.add_fake_dep(
                            dependencies.WeakDep(
                                alt_name, mutating_buf=buf.get_name(), is_fake=True
                            )
                        )
                        continue
                    # The write comes after whoever wrote it before.
                    add_user(alt_name, node)
                    node.add_fake_dep(dependencies.StarDep(alt_name, mode=node_mode))
                    for user in name_to_users[alt_name].items:
                        if user.get_name() == node.get_name():
                            continue

                        if not isinstance(user.node, BaseSchedulerNode):
                            raise AssertionError(
                                "expected user.node to be a BaseSchedulerNode"
                            )
                        for out_buf in user.node.get_outputs():
                            other_name = out_buf.get_name()
                            # The write comes after everything that read it.
                            other_name = rename(other_name)
                            # A view shares memory with what is written over, so
                            # its value has to stay alive until the write is
                            # done.  A copy has memory of its own, and only the
                            # order matters.
                            is_alias = alt_name in out_buf.get_aliases()
                            node.add_fake_dep(
                                dependencies.WeakDep(
                                    other_name,
                                    mutating_buf=buf.get_name(),
                                    is_fake=not is_alias,
                                )
                            )
                            add_user(other_name, node, is_weak=True)

            for add_dep in getattr(V.graph, "additional_buffer_deps", {}).get(
                node.get_name(), []
            ):
                add_user(add_dep, node, is_weak=True)
                # These say an order and nothing more, so they should not keep
                # anything alive.
                node.add_fake_dep(
                    dependencies.WeakDep(add_dep, node.get_name(), is_fake=True)
                )

            for add_dep in getattr(V.graph, "additional_star_deps", {}).get(
                node.get_name(), []
            ):
                add_user(add_dep, node, is_weak=False)
                node.add_fake_dep(dependencies.StarDep(add_dep))

            # What it reads, where reading and writing the same value is
            # allowed -- which is what lets the read be the write.
            for read in node.read_writes.reads:
                if not isinstance(read, dependencies.WeakDep):
                    add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # What the next piece should call this value.
            for buf in node.get_outputs():
                for alt_name in buf.get_mutations():
                    self.mutation_renames[rename(alt_name)] = buf.get_name()
                    self.mutation_renames[alt_name] = buf.get_name()
                    self.mutation_real_name[buf.get_name()] = (
                        self.mutation_real_name.get(alt_name, alt_name)
                    )

        # A value the program is handed back is wanted whatever else happens,
        # so it is given a reader that nothing can remove.
        for buf_name in V.graph.get_output_names():
            log.debug("scheduling output %s", buf_name)
            add_user(buf_name, OutputNode(dependencies.StarDep(buf_name)))

        if has_non_input_unbacked_defs:
            for out in V.graph.graph_outputs:
                for s in out.get_free_symbol_uses(unbacked_only=True):
                    if s not in unbacked_symbol_to_origin_node:
                        raise AssertionError(
                            f"{s} not in {unbacked_symbol_to_origin_node.keys()}"
                        )
                    if r := unbacked_symbol_to_origin_node[s]:
                        for buf_name in self.name_to_node[r].get_buffer_names():
                            log.debug(
                                "scheduling output %s for unbacked symint %s",
                                buf_name,
                                s,
                            )
                            add_user(buf_name, OutputNode(dependencies.StarDep(buf_name)))

        # A value handed in and written to is wanted in both states, so the
        # state it arrived in is not something that can be discarded.
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(dependencies.StarDep(name)))
                V.graph.mutated_inputs.add(name)
            elif name in V.graph.constants:
                add_user(name, OutputNode(dependencies.StarDep(name)))

        inp_names = {
            name: index for index, name in enumerate(V.graph.graph_inputs.keys())
        }
        V.graph.mutated_input_idxs = [
            inp_names[name] for name in V.graph.mutated_inputs
        ]

        for node in self.nodes:
            for buf in node.get_outputs():
                buf.set_users(name_to_users[buf.get_name()].items)

        for name in self.name_to_donated_buffer:
            self.name_to_donated_buffer[name].set_users(name_to_users[name].items)

        logbuf = IndentedBuffer()
        logbuf.splice("{")
        for key, value in name_to_users.items():
            with logbuf.indent():
                users = [v.get_name() for v in value.items]
                logbuf.splice(f"'{key}': {users},")
        logbuf.splice("}")
        buf_str = logbuf.getrawvalue().rstrip()
        compute_dependencies_log.debug("BUFFER USER LIST\n")
        compute_dependencies_log.debug("===== AFTER SCHEDULING =====\n%s", buf_str)

    def topological_sort(self, nodes: Sequence) -> list:
        """An order where nothing comes before what it waits for.

        Walks each piece and everything it waits for, depth first, so that a
        piece is reached only once everything leading to it has been placed.
        """

        user_to_nodes: dict = collections.defaultdict(list)
        for node in nodes:
            for read in node.read_writes.reads:
                user_to_nodes[read.name].append(node)

        def visit(node, seen):
            if node in seen:
                return
            seen.add(node)
            for buf in node.get_outputs():
                for user in user_to_nodes[buf.get_name()]:
                    visit(user, seen)
            ordered.append(node)

        ordered: list = []
        seen: set = set()
        for node in nodes:
            visit(node, seen)
        return ordered

    def topological_sort_schedule(self, nodes: Sequence) -> list:
        """An order where nothing comes before what it waits for, and the
        dependencies are then recorded against that order.

        Once the order is fixed, how far apart in it two pieces are says how far
        their values have to be held, which is what tells a later decision
        whether joining them is worth it.
        """

        ordered = self.topological_sort(nodes)
        for i, node in enumerate(ordered):
            node.min_order = i
            node.max_order = i
        return ordered

    def compute_ancestors(self) -> None:
        """Note, for each piece, everything it waits for directly or indirectly.

        A piece that waits for another that waits for a third is also waiting
        for the third, and that is what lets a later decision ask about the
        whole set at once.
        """

        for node in self.nodes:
            node.ancestors = OrderedSet()

        def compute_ancestors(node) -> OrderedSet:
            if node.ancestors:
                return node.ancestors
            node.ancestors = OrderedSet([node.get_name()])
            for read in node.read_writes.reads:
                if read.name in self.name_to_node:
                    compute_ancestors(self.name_to_node[read.name])
                    node.ancestors.update(self.name_to_node[read.name].ancestors)
            node.ancestors.discard(node.get_name())
            return node.ancestors

        for node in self.nodes:
            compute_ancestors(node)

    def compute_input_distances(self) -> None:
        """How far each piece is from the pieces it reads from and writes for.

        A value read by two pieces has to be held from the earliest of them to
        the latest, and the distance between them is how long that is.
        """

        for node in self.nodes:
            distance_to_writes: dict = {}
            distance_to_reads: dict = {}
            for read in node.read_writes.reads:
                if read.name not in self.name_to_node:
                    continue
                producer = self.name_to_node[read.name]
                distance_to_reads[producer.get_name()] = min(
                    node.min_order - producer.min_order,
                    distance_to_reads.get(producer.get_name(), sys.maxsize),
                )
            for write in node.read_writes.writes:
                if write.name not in self.name_to_node:
                    continue
                consumer = self.name_to_node[write.name]
                distance_to_writes[consumer.get_name()] = min(
                    consumer.min_order - node.min_order,
                    distance_to_writes.get(consumer.get_name(), sys.maxsize),
                )
            node.min_input_distance = (
                min(distance_to_writes.values()) if distance_to_writes else 0
            )
            node.max_input_distance = (
                min(distance_to_reads.values()) if distance_to_reads else 0
            )

    def dead_node_elimination(self) -> None:
        """Leave out any piece whose results nothing wants.

        A piece that computes something nothing reads, and that is not handed
        back, is work nobody asked for.  Since the pieces before it have already
        been decided, this can be done only by deciding it out of the order
        rather than by finding it earlier.
        """

        original_buffer_use_count: dict = {
            name: len(buf.users) for name, buf in self.name_to_buf.items()
        }

        updated_buffer_use_count: dict = dict(original_buffer_use_count)
        self.removed_operations: OrderedSet = OrderedSet()

        for node in self.nodes:
            # A piece that changes something outside its own results, or hands
            # something back, is there for a reason nothing else accounts for.
            if node.should_allocate() and not any(
                isinstance(user.node, OutputNode) for user in node.get_outputs()[0].users
            ):
                if not node.get_outputs()[0].get_aliases():
                    continue

            if any(buf.get_mutations() for buf in node.get_outputs()):
                self.removed_operations.discard(node.get_name())
                continue
            if any(
                isinstance(user.node, OutputNode) for buf in node.get_outputs()
                for user in buf.users
            ):
                self.removed_operations.discard(node.get_name())
                continue

            for buf in node.get_outputs():
                if updated_buffer_use_count.get(buf.get_name(), 0) > 0:
                    self.removed_operations.add(node.get_name())
                    break
                if V.graph.mark_buffer_mutated is not None and any(
                    self.name_to_node.get(name) is node
                    for name in getattr(V.graph, "mutated_buffers", ())
                ):
                    self.removed_operations.add(node.get_name())
                    break
            else:
                for buf in node.get_outputs():
                    updated_buffer_use_count[buf.get_name()] = 0
                    V.graph.removed_buffers.add(buf.get_name())
                for read in node.read_writes.reads:
                    updated_buffer_use_count[read.name] = updated_buffer_use_count.get(
                        read.name, 0
                    ) - 1

        for buf_name in self.name_to_buf:
            if self.name_to_buf[buf_name].defining_op is not None:
                if self.name_to_buf[buf_name].defining_op.get_name() in (
                    self.removed_operations
                ):
                    self.removed_operations.discard(
                        self.name_to_buf[buf_name].defining_op.get_name()
                    )

        self.nodes = [
            node
            for node in self.nodes
            if node.get_name() not in self.removed_operations
        ]
        self.name_to_buf = {
            name: buf
            for name, buf in self.name_to_buf.items()
            if name not in V.graph.removed_buffers
        }

    def get_fused_node(self, node) -> Optional:
        """What this piece is now part of, or itself where it stands alone."""

        return self.name_to_fused_node.get(node.get_first_name(), node)

    def get_buffer(self, name: str) -> Optional:
        return self.name_to_buf.get(name)

    def get_node_by_name(self, name: str) -> Optional:
        return self.name_to_node.get(name)

    def can_free(self, name: str) -> bool:
        """Whether this value's memory may be given back."""

        buf = self.name_to_buf.get(name)
        if buf is None:
            return False
        if name in self.buffer_names_to_free:
            return True
        return buf.can_free()

    def free_buffers_in_scheduler(self) -> None:
        self.buffer_names_to_free = OrderedSet()

    def free_buffers(self) -> None:
        """Give back the memory of any value nothing is holding on to.

        A buffer is only given back once nothing will read it again, and the
        code is told so that it does not use the memory afterwards.
        """

        for node in self.nodes:
            if node is None:
                continue
            for buf in node.get_outputs():
                if self.can_free(buf.get_name()):
                    self.buffer_names_to_free.add(buf.get_name())

    def compute_last_usage(self) -> None:
        """Note which values nothing will read after each piece.

        What a value is needed for up to decides whether it may be handed to
        something else in its place, so this is what a later decision about
        writing into memory in place is made against.
        """

        future_used_buffers: OrderedSet = OrderedSet()
        last_use: dict = {}
        for node in reversed(self.nodes):
            # The value itself is not read again once this piece has written it
            # and whatever is inside the same kernel has used it.
            for buf in node.get_outputs():
                if buf.get_name() not in last_use:
                    last_use[buf.get_name()] = node
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)

    def _enforce_switch_ordering(self) -> None:
        """Put the pieces of a choice back in the order they were written in.

        A choice runs one of several pieces, and what it produces depends on
        which.  Running them in an order of our own choosing would not change
        which is chosen, but the pieces that follow have to be able to tell
        which one it was.
        """

        for node in self.nodes:
            if not hasattr(node, "get_subgraphs"):
                continue
            if not node.get_subgraphs():
                continue

    def insert_memory_check_nodes(self) -> None:
        """Put a piece between each step that records what is held at that point.

        What is held is a property of the order rather than of the arithmetic,
        and this is the only way to see it directly -- so it is recorded in the
        order itself, where running the code says what happened.
        """

        buffer_name_to_scheduled: dict = {}

        for node in self.nodes:
            for buf in node.get_outputs():
                buffer_name_to_scheduled[buf.get_name()] = buf

        for buf_name in buffer_name_to_scheduled:
            self.buffer_names_to_free.discard(buf_name)

    def fuse_nodes(self, nodes: list) -> list:
        """Join as many of these as can be joined, and return what is left.

        Several rounds are needed: joining two pieces can make a third joinable
        that was not before, because what the pair now holds in hand is what
        the third would have had to read.  So it is done until a round joins
        nothing, which is the point where no further join would help.
        """

        for i in range(10):
            old_len = len(nodes)
            fusion_log.debug(
                "===== attempting fusion (%d/10): %d nodes =====",
                i + 1,
                old_len,
            )
            nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
            new_len = len(nodes)
            fusion_log.debug(
                "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
                i + 1,
                old_len,
                new_len,
            )
            if new_len == old_len or new_len == 1:
                fusion_log.debug("===== fusion complete (%d iterations) =====", i + 1)
                break

        if config.loop_ordering_after_fusion or config.loop_index_inversion_in_fusion:
            # The loops can be walked in a different order once it is settled
            # which pieces are joined, and that may allow more to be.
            nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
        return nodes

    def fuse_nodes_once(self, nodes: list, is_reorder_round: bool) -> list:
        """One round of joining.

        Which pairs are worth looking at is decided first, then each is tried in
        the order of what it would save, and what a try produces joins with
        whatever else has since become compatible.  A pair that cannot be
        joined is left alone, so that a later piece may still be joined to one
        side of it.
        """

        self.prune_redundant_deps(nodes)
        fused_nodes: OrderedSet = OrderedSet(nodes)
        if fusion_log.isEnabledFor(logging.DEBUG):
            fusion_log.debug("fuse_nodes_once, candidates:")
            for node in fused_nodes:
                fusion_log.debug("  %s", node.debug_str_short())

        # Pairs that were found legal but not yet tried, where whether they are
        # worth it is put off until later.
        pending_fusions: dict = {}

        template_fusion_nodes: dict = {}
        deferred_prologue_fusions: list = []

        possible_fusions = self.get_possible_fusions(
            nodes,
            is_reorder_round,
        )

        self._try_fusion_pairs(
            possible_fusions,
            pending_fusions,
            template_fusion_nodes,
            fused_nodes,
            is_reorder_round,
        )
        self._finish_pending_fusions(fused_nodes, pending_fusions)

        self._evaluate_pending_template_fusions(template_fusion_nodes, fused_nodes)
        template_fusion_nodes.clear()

        if deferred_prologue_fusions:
            self._try_fusion_pairs(
                deferred_prologue_fusions,
                pending_fusions,
                template_fusion_nodes,
                fused_nodes,
                is_reorder_round,
            )
            self._evaluate_pending_template_fusions(template_fusion_nodes, fused_nodes)

        nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        nodes = self.topological_sort_schedule(nodes)
        return nodes

    def can_fusion_increase_peak_memory(self, node1, node2) -> bool:
        """Whether joining these could make the program hold more memory at once.

        Whether the program is at its peak where these two sit is not known --
        the order may change later, and a join changes how long each value is
        held -- so what is worked out is a floor on what joining would cost:
        the values each of the two reads and nobody else reads, and that the
        other also reads.  Left unjoined, the second of those would have been
        given the first's memory once the first was done with it, so joining
        means that memory cannot be reused.  Whether that is actually what
        decides the peak is not known, which is why this is a bound and not a
        figure.
        """

        from .codegen.wrapper import buffer_reuse_key

        def _find_single_user_inputs(node):
            output = []
            for rd in node.read_writes.reads:
                buf = self.name_to_buf.get(rd.name)
                if buf and len(buf.users) == 1 and buf.node.has_tensor_output():
                    output.append(buf.node)
            return output

        # What each of the two could have had reused, and what they could both
        # have had reused -- which is what the join gives up.
        lhs_dep_nodes = _find_single_user_inputs(node1)
        rhs_dep_nodes = _find_single_user_inputs(node2)

        lhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in lhs_dep_nodes)
        rhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in rhs_dep_nodes)

        common_reuse_keys = lhs_reuse_keys.intersection(rhs_reuse_keys)

        memory_overhead = 0
        for key in common_reuse_keys:
            try:
                memory_overhead += int(key[2])
            except ValueError:
                # The size is not a number, so there is nothing to compare and
                # the join is allowed.
                return False

        bw_saving = self.score_fusion_memory(node1, node2)

        # The factor is a judgement rather than a measurement: what matters is
        # that the cost of holding more is far larger than what the join saves.
        if V.graph.sizevars.statically_known_gt(memory_overhead, 32 * bw_saving):
            return True
        return False

    def fusion_prevent_too_many_reads_and_writes(
        self, node1, node2, threshold: int
    ) -> bool:
        """Whether joining these would leave too many values for one launch to name.

        A launch can only be given so many arguments, and joining two pieces
        adds together everything they touch.  Some of that stops counting once
        they are one: a value one produces and the other reads is internal to
        the launch and is never named.  So what is counted is what would remain
        named, which is not the same as what they touch between them.
        """

        fused_node_names = OrderedSet(
            [node.get_name() for node in node1.get_nodes()]
            + [node.get_name() for node in node2.get_nodes()]
        )

        # What the second reads from the first stops being named.
        node1_write_names = OrderedSet(dep.name for dep in node1.read_writes.writes)
        node2_read_names = OrderedSet(dep.name for dep in node2.read_writes.reads)
        reads_removed_through_fusion = node2_read_names & node1_write_names

        # What the first writes and only the second reads also stops being
        # named, since it never leaves the launch.
        writes_removed_through_fusion: OrderedSet = OrderedSet()
        for write_dep in node1.read_writes.writes:
            if self.can_buffer_be_removed_through_fusion(
                write_dep.name, fused_node_names
            ):
                writes_removed_through_fusion.add(write_dep.name)

        all_read_names = OrderedSet(
            dep.name for dep in node1.read_writes.reads
        ) | OrderedSet(dep.name for dep in node2.read_writes.reads)

        all_write_names = OrderedSet(
            dep.name for dep in node1.read_writes.writes
        ) | OrderedSet(dep.name for dep in node2.read_writes.writes)

        unique_reads = all_read_names - reads_removed_through_fusion

        unique_writes = all_write_names - writes_removed_through_fusion

        unique_io_buffers = unique_reads | unique_writes

        return len(unique_io_buffers) > threshold

    def are_long_distant_nodes(self, node1, node2) -> bool:
        """Whether these two are far apart in the order they came out in.

        Joining two pieces that were not neighbours lengthens the time their
        values are held, since something that used to be able to give its memory
        away between them now cannot.  That shows up most where parts of a
        program are recomputed: pieces from different parts of the original
        would not be neighbours at all, and joining them holds the whole of one
        part while the other runs.

        How far is too far is a judgement.  What would answer it properly is
        asking where the program was already holding the most and keeping
        joins from crossing that -- but a join changes how long things are held,
        so the answer would have to be worked out again after each one, and
        that is more work than the saving is worth.
        """

        proximity_score = max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return proximity_score > 64

    def get_possible_fusions(self, nodes: list, is_reorder_round: bool) -> list:
        """Which pairs could be joined, best first.

        Only pieces that touch some value in common could be joined at all, so
        the pairs are looked for within each such group rather than among all
        pairs -- which is what keeps this from being quadratic in everything.
        How many of each is looked at is capped, since the pairs that share
        nothing are the ones least likely to fit.
        """

        possible_fusions = []
        seen: OrderedSet = OrderedSet()

        def check_all_pairs(nodes: list) -> None:
            for node1_index, node1 in enumerate(nodes):
                for node2 in nodes[
                    node1_index + 1 : node1_index
                    + 1
                    + config.max_fusion_buffer_group_pairwise_attempts
                ]:
                    key = (node1, node2)
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.can_fuse(node1, node2, is_reorder_round):
                        possible_fusions.append(key)
                    elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(
                        node2, node1, is_reorder_round
                    ):
                        # A piece that is a set, or a template, has to come
                        # after what it consumes -- so the other way round is
                        # the one that can work.
                        possible_fusions.append((node2, node1))

        buffer_names_grouping: dict = collections.defaultdict(list)
        for node in nodes:
            if self.unfusable_node(node):
                continue
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.aggressive_fusion:
            # Pieces that walk the same axes are worth looking at even where
            # they share no value, since what they share is the work.
            group_grouping: dict = collections.defaultdict(list)
            for node in nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        possible_fusions = self.get_possible_fusions_with_highest_priority(
            possible_fusions
        )
        possible_fusions.sort(key=self.score_fusion_key, reverse=True)
        fusion_log.debug("found %d possible fusions", len(possible_fusions))
        return possible_fusions

    def _producer_output_names_read_by_consumer(self, producer, consumer) -> OrderedSet:
        """Which of this piece's results are read by that piece."""

        producer_buf_names = OrderedSet(
            self.mutation_renames.get(name, name)
            for name in producer.get_buffer_names()
        )
        return OrderedSet(
            name
            for dep in consumer.read_writes.reads
            if (name := self.mutation_renames.get(dep.name, dep.name))
            in producer_buf_names
        )

    def _nested_index_equivalent_dep_names(self, node1, node2):
        """Which reads of one piece by the other name the same elements.

        Two accesses name the same elements when they are written the same way,
        and where that is so a piece may be joined to another that writes the
        same value at the same place.  Nothing is said where the two pieces are
        not of a kind where this can arise.
        """

        if not _nested().is_candidate(node1, node2):
            return None
        if not _nested().can_fuse(node1, node2):
            return None
        return self._producer_output_names_read_by_consumer(node1, node2)

    def _score_fusion_memory_for_can_fuse(
        self,
        node1,
        node2,
        allow_mix_order_reduction: bool = True,
        index_equivalent_dep_names=None,
    ) -> int:
        """What joining these would save, with the accesses that share an index counted.

        Where two pieces read and write the same value in the same way, the
        value need not be written at all, and that is worth more than the
        arithmetic would suggest -- so where there are such accesses and the
        ordinary count came out at nothing, they are counted as well.
        """

        score = self.score_fusion_memory(
            node1,
            node2,
            allow_mix_order_reduction=allow_mix_order_reduction,
        )
        if not isinstance(score, int):
            raise AssertionError(f"expected score to be int, got {type(score)}")
        if score == 0 and index_equivalent_dep_names:
            score = self._score_fusion_memory_by_fusable_read_write(
                node1,
                node2,
                index_equivalent_dep_names=index_equivalent_dep_names,
            )
        return score

    def _score_fusion_memory_by_fusable_read_write(
        self,
        node1,
        node2,
        index_equivalent_dep_names=None,
    ) -> int:
        """What joining would save, counting only the reads that could be handed over.

        A read that names different elements from the write it would replace
        cannot be served from what the other piece holds, so it saves nothing;
        the rest save the value that would not have to be written.
        """

        if index_equivalent_dep_names is None:
            index_equivalent_dep_names = self._nested_index_equivalent_dep_names(
                node1, node2
            )
        if not index_equivalent_dep_names:
            return 0

        producer, consumer = node1, node2
        if node1.get_operation_names() & node2.ancestors:
            pass
        elif node2.get_operation_names() & node1.ancestors:
            producer, consumer = node2, node1
        else:
            # Neither reads the other, so there is nothing one could hand over.
            return 0

        return sum(
            self.name_to_buf[name].node.get_numel()
            for name in index_equivalent_dep_names
            if name in self.name_to_buf
        )

    def _can_fuse(
        self,
        node1,
        node2,
        can_reorder: bool = False,
        allow_mix_order_reduction: bool = True,
        index_equivalent_dep_names=None,
    ) -> bool:
        """Whether these two pieces may be joined.

        What is being asked is whether the second can be written as part of the
        first: that nothing the first does is undone, that what the second reads
        is available where the first is being computed, and that the two walk
        the same axes.  The last of those is a property of the machine, so it is
        put to the backend; the first two are about this graph.

        Where a piece is refused, the reason is recorded rather than raised,
        since a refusal is an ordinary answer and the reason is what makes a
        surprising one explicable.
        """

        if node1 is node2:
            return False
        if index_equivalent_dep_names is not None:
            index_equivalent_dep_names = OrderedSet(
                self.mutation_renames.get(name, name)
                for name in index_equivalent_dep_names
            )

        # Work on two queues runs at the same time, so joining across them would
        # put two things in one that could not have been.
        if self._has_multi_stream_nodes():
            if self.get_node_stream(node1) != self.get_node_stream(node2):
                return False
        if self._has_mempool_nodes():
            if self.node_to_mempool.get(node1) != self.node_to_mempool.get(node2):
                return False

        if isinstance(node1, FusedNestedReductions):
            return node1.can_fuse_with(node2, can_reorder=can_reorder)
        if isinstance(node2, FusedNestedReductions):
            return False

        if isinstance(node1, FusedMixOrderReductions):
            return node1.can_fuse_with(node2)
        if isinstance(node2, FusedMixOrderReductions):
            # Nothing may be put before a pair of reductions in different
            # orders: what they share a read with would have to be in hand first.
            return False

        why = WhyNoFuse(node1, node2)

        if node1.is_template() and node2.has_strict_reduction():
            why("template fusion does not preserve strict reduction ordering")
            return False

        if (
            (node1.has_strict_reduction() or node2.has_strict_reduction())
            and node1.is_reduction()
            and node2.is_reduction()
        ):
            why("reduction fusion does not preserve strict reduction ordering")
            return False

        if node1.is_template() and self.get_backend(
            node1.get_device()
        ).can_fuse_multi_outputs_template(node1, node2):
            return True
        if node1.is_template() and self.get_backend(
            node1.get_device()
        ).can_fuse_reduction_epilogue(node1, node2):
            return True

        if isinstance(node1, GroupedSchedulerNode) or isinstance(
            node2, GroupedSchedulerNode
        ):
            why("grouped node must not be fused with other nodes")
            return False
        if isinstance(node1, NopKernelSchedulerNode) and not node1.is_template():
            why("node1 is nop")
            return False

        if isinstance(node1, ExternKernelSchedulerNode):
            if not isinstance(node1.node, ir.UserDefinedTritonKernel):
                why("node1 is extern but not a triton kernel")
                return False

            if not node1.node.can_fuse_epilogue():
                why("node1's triton kernel doesn't support epilogue fusion")
                return False

            if not isinstance(node2, SchedulerNode):
                why("node1 is extern but node2 is not SchedulerNode")
                return False
            if not isinstance(node2.node, ComputedBuffer):
                why("node1 is extern but node2.node is not SchedulerNode")
                return False
            if not isinstance(node2.node.data, Pointwise):
                why("node1 is extern but node2.node.data is not Pointwise")
                return False

            if len(node1.node.mutation_outputs) != 1:
                raise AssertionError(
                    f"expected one mutation output, got {len(node1.node.mutation_outputs)}"
                )
            written_buffer_name = node1.node.mutation_outputs[0].name

            # What is folded in has to read the one value the kernel wrote, and
            # write it back at the same place: anything else would need loads
            # the kernel has no expression for.
            epilogue_reads = list(node2.read_writes.reads)
            epilogue_writes = list(node2.read_writes.writes)
            if (
                len(epilogue_reads) != 1
                or len(epilogue_writes) != 1
                or epilogue_reads[0].name != written_buffer_name
            ):
                why("epilogue is not a unary read of the output buffer")
                return False

            write_dep = epilogue_writes[0]
            read_dep = epilogue_reads[0]
            if not isinstance(read_dep, dependencies.MemoryDep):
                raise AssertionError(f"expected MemoryDep, got {type(read_dep)}")
            if not isinstance(write_dep, dependencies.MemoryDep):
                raise AssertionError(f"expected MemoryDep, got {type(write_dep)}")
            if read_dep.index != write_dep.index or read_dep.size != write_dep.size:
                why("epilogue's read and write indices differ")
                return False

            # A shape the body mentions somewhere other than a read is one the
            # kernel has no way of being given.
            node2_inner_fn_free_symbols = node2.node.data.inner_fn_free_symbols()
            for symbol in node2_inner_fn_free_symbols:
                usages = node2.node.data.collect_inner_fn_symbol_usage(symbol)
                if any(usage != "load" for usage in usages):
                    return False

            if len(node1.node.mutable_args) != 1:
                raise AssertionError(
                    f"expected one mutable arg, got {len(node1.node.mutable_args)}"
                )

            # The arrangement must match, though the type need not: the work
            # that is folded in may convert as it goes.
            layout1 = node1.node.mutable_args[0].layout
            layout2 = node2.node.layout
            if not (isinstance(layout1, ir.Layout) and isinstance(layout2, ir.Layout)):
                raise AssertionError("expected layout1 and layout2 to be ir.Layout")
            if (
                layout1.size != layout2.size
                or layout1.stride != layout2.stride
                or layout1.device != layout2.device
            ):
                why("node1 and node2 uses different buf layouts")
                return False

            def _is_other_node_that_references_mutation_buffer(other_node):
                return (
                    (other_node is not node1)
                    and (other_node is not node2)
                    and written_buffer_name in other_node.used_buffer_names()
                )

            if any(
                _is_other_node_that_references_mutation_buffer(node)
                for node in self.nodes
            ):
                return False

        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            why("node2 is extern or nop")
            return False

        if node2.get_operation_names() & node1.ancestors:
            why("node1 must go before node2")
            return False

        if node2.is_template():
            if not _is_prologue_fusion_enabled():
                why("prologue fusion turned off")
                return False

            if node1.is_reduction() or node1.is_template():
                why("prologue fusion only supported for pointwise nodes")
                return False

            template = node2.get_template_node_or_throw()
            allowed_prologue_inps = template.get_allowed_prologue_inps()
            if not allowed_prologue_inps:
                why("template has no allowed prologue inputs")
                return False

            unsupported_prologue_args = (
                OrderedSet(inp.get_name() for inp in template.inputs)
                - allowed_prologue_inps
            )

            if node1.get_buffer_names() & unsupported_prologue_args:
                why("prologue fusion not implemented for kernel for these inputs")
                return False

            if node1.has_aliasing_or_mutation() or (
                template.has_aliasing_or_mutation_for_prologue_fusion(node2)
            ):
                why("template prologue can only fuse functional pointwise nodes")
                return False

            prologue_nodes = node1.get_nodes()
            for node in prologue_nodes[:-1]:
                for out in node.get_outputs():
                    if not all(user.node in prologue_nodes for user in out.users):
                        why("template prologue can only fuse nodes with a single use")
                        return False

            template_snodes = (
                [node2]
                if not isinstance(node2, FusedSchedulerNode)
                else [n for n in node2.snodes if n.is_template()]
            )
            if len(template_snodes) != 1:
                raise AssertionError(
                    f"expected one template snode, got {len(template_snodes)}"
                )
            template_snode = template_snodes[0]

            if not (
                len(prologue_nodes[-1].outputs) == 1
                and len(prologue_nodes[-1].outputs[0].users) == 1
                and prologue_nodes[-1].outputs[0].users[0].node is template_snode
            ):
                why(
                    "template prologue can only fuse nodes with a single use into template"
                )
                return False

            if not self.check_prologue_fusion_heuristics_fusable(node1, node2, why):
                return False

        if node1.is_template():
            atomic_add_mutation_epilogue = _can_fuse_atomic_add_template_epilogue(
                node1, node2
            )
            backend = self.get_backend(node1.get_device())
            if (
                (node2.has_aliasing_or_mutation() and not atomic_add_mutation_epilogue)
                or (
                    node2.is_reduction()
                    and not backend.can_fuse_reduction_epilogue(node1, node2)
                )
                or not _is_epilogue_fusion_enabled()
            ):
                why("template epilogue not satisfied")
                return False
            template_buf = node1.get_template_node()
            if template_buf is None:
                raise AssertionError("expected template_buf to be set")
            if template_buf.is_multi_outputs_template() and not isinstance(
                node2.node, ir.ComputedBuffer
            ):
                why("multi-output template epilogue requires ComputedBuffer")
                return False

        if (node1.get_buffer_names() & V.graph.no_fuse_buffer_names) or (
            node2.get_buffer_names() & V.graph.no_fuse_buffer_names
        ):
            why("fusion for buffer explicit disabled")
            return False
        device = node1.get_device()
        device2 = node2.get_device()
        if device != device2:
            why("device mismatch (%s vs %s)", device, device2)
            return False
        del device2

        if index_equivalent_dep_names is None:
            index_equivalent_dep_names = self._nested_index_equivalent_dep_names(
                node1, node2
            )
        shared_data_score = self._score_fusion_memory_for_can_fuse(
            node1,
            node2,
            allow_mix_order_reduction=allow_mix_order_reduction,
            index_equivalent_dep_names=index_equivalent_dep_names,
        )

        if config.expand_dimension_for_pointwise_nodes and (
            expand_analysis := self.get_expand_dim_for_pointwise_nodes(node1, node2)
        ):
            (expand_dim, smaller_node, expand_size) = expand_analysis
            smaller_node.expand_dimension_for_pointwise_node(expand_dim, expand_size)
            shared_data_score = self._score_fusion_memory_for_can_fuse(
                node1,
                node2,
                index_equivalent_dep_names=index_equivalent_dep_names,
            )

        if (
            can_reorder
            and shared_data_score < config.score_fusion_memory_threshold
            and (
                config.loop_ordering_after_fusion or config.loop_reindexing_after_fusion
            )
        ):
            new_shared_data_score = self.shared_data_after_reordering_loop(node1, node2)
            if new_shared_data_score >= 0:
                shared_data_score = new_shared_data_score

        if (
            config.loop_index_inversion_in_fusion
            and shared_data_score < config.score_fusion_memory_threshold
        ):
            new_shared_data_score = self.shared_data_after_inverting_indexing(
                node1, node2
            )
            if new_shared_data_score >= 0:
                shared_data_score = new_shared_data_score

        if loop_ordering_log.isEnabledFor(logging.DEBUG):
            loop_ordering_log.debug(
                "%s and %s has %s shared data",
                node1.get_name(),
                node2.get_name(),
                shared_data_score,
            )

        if not _can_fuse(self, node1, node2, shared_data_score):
            return False

        if node1.get_operation_names() & node2.ancestors:
            # The second reads what the first produces, so it can only be
            # written as part of it.
            if (
                self.can_fuse_vertical(
                    node1,
                    node2,
                    index_equivalent_dep_names=index_equivalent_dep_names,
                )
                and _can_fuse_vertical(self, node1, node2, shared_data_score)
                and self.get_backend(device).can_fuse_vertical(node1, node2)
            ):
                return True

            return False
        else:
            # Neither reads the other, so they may only be put side by side.
            return _can_fuse_horizontal(
                self, node1, node2, shared_data_score
            ) and self.get_backend(device).can_fuse_horizontal(node1, node2)

    def _has_multi_stream_nodes(self) -> bool:
        return bool(getattr(self, "_multi_stream_nodes", False))

    def _has_mempool_nodes(self) -> bool:
        return bool(getattr(self, "_mempool_nodes", False))

    def check_prologue_fusion_heuristics_fusable(self, node1, node2, why) -> bool:
        """Whether a piece is worth folding into the start of a template.

        What is folded in has to be cheap relative to what it saves, and it has
        to be cheap in a way the template can absorb: a body that walks a
        different number of elements would have to be redone, which is worse
        than not folding it in at all.
        """

        (numel1, rnumel1) = node1.group[1]
        (numel2, rnumel2) = node2.group[1]

        # A template that walks fewer elements than the work being folded in
        # cannot hold the whole of it, so only part would be folded and the rest
        # would still have to run.
        if V.graph.sizevars.statically_known_leq(numel1, numel2 * 0.5):
            why("Prologue fusion heuristics: numel1 too small")
            return False

        # A piece that reduces is doing more work than the template can absorb
        # at a position, since a template produces each result once.
        if node1.is_reduction() and V.graph.sizevars.statically_known_leq(
            rnumel1, 2048
        ):
            why("Prologue fusion heuristics: rnumel1 too small")
            return False

        return True

    def get_expand_dim_for_pointwise_nodes(self, node1, node2):
        """Which axis a shorter piece would have to be made to look longer in.

        Two pieces walking different numbers of elements could still be joined
        if the shorter one were made to walk the longer one's count, repeating
        what it computed.  That is only worth doing where the repetition is
        along an axis nothing else is watching, which is what is looked for.
        """

        node1_numel = V.graph.sizevars.statically_known_equal(
            sympy_product(node1.get_ranges()[0]), sympy_product(node2.get_ranges()[0])
        )
        if node1_numel is None:
            return None
        return None

    def shared_data_after_reordering_loop(self, node1, node2) -> int:
        """How much the two would share, were the axes walked in a better order.

        Two pieces read the same value at the same place only if the axes are
        walked the same way round, and the order that makes that so is a
        property of the value rather than of either piece -- so it is looked for
        and, if found, the score is what it would be with it.
        """

        if len(node1.get_nodes()) > 1 or len(node2.get_nodes()) > 1:
            return -1
        if len(node1.get_ranges()[0]) != len(node2.get_ranges()[0]):
            return -1
        if node1.get_ranges()[1] or node2.get_ranges()[1]:
            return -1

        for n in (node1, node2):
            if n.get_layout() is None or n.get_layout().is_contiguous():
                return -1

        # Both read one value, and the value is the same.  If the strides are
        # the same either way round then no order helps.
        node1_layout = node1.get_layout()
        node2_layout = node2.get_layout()
        if node1_layout.stride == node2_layout.stride:
            return -1
        if node1_layout.stride[::-1] != node2_layout.stride:
            return -1

        return self.score_fusion_memory(node1, node2, count_bytes=False)

    def shared_data_after_inverting_indexing(self, node1, node2) -> int:
        """How much the two would share, were the value read the other way round.

        Reading a value forwards and reading it backwards are different, and
        which is meant is decided by the arithmetic rather than by the value.
        Where the two pieces read the same value in opposite directions there is
        nothing to share and joining them gains nothing.
        """

        return -1

    def unfusable_node(self, node) -> bool:
        """Whether this piece cannot be joined to anything at all.

        A piece that changes something outside its own results, or hands
        something back, is there for a reason nothing else accounts for.
        """

        return (
            isinstance(node, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not isinstance(node, FusedSchedulerNode)
        )

    def get_possible_fusions_with_highest_priority(
        self, possible_fusions: list
    ) -> list:
        """Keep only the pairs that nothing else outranks.

        Where several pieces could be joined to the same one, only the best may
        be taken: once joined, the pair is one piece, and what it could have
        been joined to is a different question.
        """

        highest_priority: dict = {}
        for pair in possible_fusions:
            key = self.get_fusion_pair_priority(*pair)
            for item in pair:
                if key < highest_priority.get(item, sys.maxsize):
                    highest_priority[item] = key
        return [
            pair
            for pair in possible_fusions
            if all(
                self.get_fusion_pair_priority(*pair) == highest_priority[item]
                for item in pair
            )
        ]

    def score_fusion_key(self, fusion: tuple) -> tuple:
        """What joining this pair would save, larger first.

        How much memory is held across the join is the main thing, and how many
        values would no longer be written out is the other: a value read only
        by the two of them never has to reach memory.
        """

        node1, node2 = fusion

        return (
            self.score_fusion_memory(node1, node2, count_bytes=False),
            self.compute_removed_buffer_savings(node1, node2),
        )

    def compute_removed_buffer_savings(self, node1, node2) -> int:
        """How many values would stop being written out if these were joined.

        A value produced and consumed only within the join can be handed from
        one piece to the other where it already is, and never written at all.
        """

        fused_snode = OrderedSet(node1.get_nodes() + node2.get_nodes())
        op_names = OrderedSet(node.get_name() for node in fused_snode)
        buf_names = OrderedSet.union(*(x.get_buffer_names() for x in fused_snode))
        removed_buffers = [
            buf
            for buf in buf_names
            if self.can_buffer_be_removed_through_fusion(buf, op_names)
        ]

        total = 0
        for buf in removed_buffers:
            try:
                total += self.name_to_buf[buf].node.get_numel()
            except (KeyError, NotImplementedError):
                pass
        return total

    def _try_fusion_pairs(
        self,
        possible_fusions: list,
        pending_fusions: dict,
        template_fusion_nodes: dict,
        fused_nodes: OrderedSet,
        is_reorder_round: bool,
    ) -> None:
        """Try each pair in turn, joining it and taking its place."""

        for node1, node2 in possible_fusions:
            if node1 not in fused_nodes or node2 not in fused_nodes:
                continue

            if self.will_fusion_create_cycle(node1, node2):
                continue

            # Joining a template is put off, since what a template can take is
            # a property of the template and not of the pair.
            if node1.is_template() or node2.is_template():
                fn = lambda: self.fuse(node1, node2)  # noqa: E731
                template_fusion_nodes.setdefault(node1, []).append(
                    PendingFusion(fn, node1, node2)
                )
                template_fusion_nodes.setdefault(node2, []).append(
                    PendingFusion(fn, node1, node2)
                )
                continue

            group = fuse(node1, node2)
            if group is None:
                continue

            fused_nodes.remove(node1)
            fused_nodes.remove(node2)
            fused_nodes.add(group)
            self.name_to_fused_node[group.get_first_name()] = group

    def _finish_pending_fusions(
        self, fused_nodes: OrderedSet, pending_fusions: dict
    ) -> None:
        """Take the joins that were put off, for whichever pieces are still there.

        A pair found legal earlier may no longer be, since joining something
        else in between may have taken one of them.  So each is looked at again
        rather than assumed.
        """

        for fusion in list(pending_fusions.values()):
            node1, node2 = fusion.get_fusion_nodes()
            if node1 in fused_nodes and node2 in fused_nodes:
                if not self.will_fusion_create_cycle(node1, node2):
                    group = fusion.callable_fn()
                    if group is not None:
                        fused_nodes.remove(node1)
                        fused_nodes.remove(node2)
                        fused_nodes.add(group)
                        self.name_to_fused_node[group.get_first_name()] = group
        pending_fusions.clear()

    def _evaluate_pending_template_fusions(
        self, template_fusion_nodes: dict, fused_nodes: OrderedSet
    ) -> None:
        """Decide, for each template, whether anything may be folded into it.

        A template was written for a particular shape of work, so whether a
        piece fits is a question about the template and not about the piece.
        Every piece that asked is answered here, since a template that is asked
        about and never told no would wait forever.
        """

        for node, fusions in template_fusion_nodes.items():
            for fusion in fusions:
                if isinstance(node, FusedSchedulerNode) and any(
                    n.is_template() for n in node.snodes
                ):
                    # A group of which one part is a template: the template is
                    # what the other part would have to be folded into.
                    continue
                fusion.callable_fn()

    def will_fusion_create_cycle(self, node1, node2) -> bool:
        """Whether joining these would make something wait for itself.

        A join can bring two pieces together that were kept apart only by
        something in between, and where that something is one of the two, the
        result would have to run before itself.
        """

        # Only a join can make a piece wait for something new, since only a join
        # changes what a piece waits for.
        visited: OrderedSet = OrderedSet()

        def found_path(node) -> bool:
            # A piece that has been visited is one already found to be waiting
            # for something, which is enough to stop the walk.
            if node in visited:
                return True
            visited.add(node)
            for buf in node.get_outputs():
                for user in self.name_to_buf[buf.get_name()].users:
                    if user.node.get_first_name() == node2.get_first_name():
                        return True
            return False

        return found_path(node1)

    def prune_redundant_deps(self, nodes: list) -> None:
        """Forget the dependencies that joining has made unnecessary.

        A dependency that only said which of two pieces came first says nothing
        once they are one piece.
        """

        name_to_fused_node = self.name_to_fused_node
        for node in nodes:
            node.prune_redundant_deps(name_to_fused_node)

    def can_fuse(
        self,
        node1,
        node2,
        is_joint_graph: bool = False,
    ) -> bool:
        """Whether these two pieces may be joined.

        Whether they fit is a property of the machine, so it is put to the
        backend.  A pair that has already been decided is not decided again.
        """

        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.can_fuse(node1, node2)

        if (
            isinstance(node1, FusedExternTritonKernelSchedulerNode)
            or isinstance(node2, FusedExternTritonKernelSchedulerNode)
        ):
            return False

        if len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size * 10:
            return False

        for node in (node1, node2):
            if node.has_strict_reduction() and (
                node1.is_reduction() or node2.is_reduction()
            ):
                return False

        if node1.is_reduction() != node2.is_reduction():
            consumer_fusion = True
            producer_fusion = True
            if is_epilogue_fusion(node1, node2, consumer_fusion, producer_fusion):
                return True
            if is_prologue_fusion(node1, node2, consumer_fusion, producer_fusion):
                return True
            return self.can_fuse_vertical(node1, node2) or self.can_fuse_vertical(
                node2, node1
            )

        if node1.is_reduction() and node2.is_reduction():
            return self.can_fuse_vertical(node1, node2) or self.can_fuse_vertical(
                node2, node1
            )

        return self.can_fuse_horizontal(node1, node2) or self.can_fuse_horizontal(
            node2, node1
        )

    def can_fuse_vertical(self, node1, node2, index_equivalent_dep_names=None) -> bool:
        """Whether the second may be written as part of the first.

        Which accesses name the same elements is passed on rather than worked
        out again, since whether a read can be served from what the other piece
        holds depends on it.
        """

        device = node1.get_device()
        backend = self.get_backend(device)
        return backend.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2) -> bool:
        """Whether two pieces walking the same axes may become one."""

        device = node1.get_device()
        backend = self.get_backend(device)
        return backend.can_fuse_horizontal(node1, node2)

    def can_fuse_reduction_epilogue(self, node1, node2) -> bool:
        """Whether work on reduced values may be folded into the reduction."""

        device = node1.get_device()
        backend = self.get_backend(device)
        return backend.can_fuse_reduction_epilogue(node1, node2)

    def can_fuse_multi_outputs_template(self, node1, node2) -> bool:
        """Whether a piece may be folded into a template with several results."""

        device = node1.get_device()
        backend = self.get_backend(device)
        return backend.can_fuse_multi_outputs_template(node1, node2)

    def score_fusion(self, node1, node2) -> int:
        """What joining this pair would save."""

        return self.score_fusion_memory(
            node1, node2, count_bytes=config.triton.cooperative_reductions
        )

    def speedup_by_fusion(self, node1, node2) -> "FusionResult":
        """Whether joining these makes things faster, rather than merely possible.

        Joining is always correct, so the question is only whether it pays.
        Most of the time it does -- fewer launches, and values that never reach
        memory -- and the exceptions are rare enough that the answer is usually
        yes.  Where it is worth working out properly is when the two pieces are
        to be compared by running them, which costs compiling and running each
        version and is therefore only done when the program is being tuned
        rather than merely compiled.

        The cases answered here without measuring are the ones where measuring
        would say nothing: work that cannot be measured, and work the machine
        has no way of writing two versions of.
        """

        is_multi_template = any(
            n.is_template() and isinstance(n.get_template_node(), ir.MultiTemplateBuffer)
            for n in (node1, node2)
        )
        atomic_add_template_epilogue = isinstance(
            node1.get_template_node(), ir.TritonTemplateBuffer
        ) and _is_atomic_add_mutation_epilogue(node2, check_config=False)
        if atomic_add_template_epilogue and not config.epilogue_fusion_with_atomic_add:
            return FusionResult.fuse(False)

        if not config.benchmark_fusion and not is_multi_template:
            return FusionResult.fuse(True)

        # A reduction cut into stages cannot be written as an ordinary kernel,
        # so there is nothing to compare it against.
        fused_nodes = [*node1.get_nodes(), *node2.get_nodes()]
        device = fused_nodes[0].get_device()
        staged = isinstance(node1, FusedStagedReduction) or isinstance(
            node2, FusedStagedReduction
        )
        nested = _nested()._is_dependent_reduction_pair(
            node1, node2
        ) and _nested().can_fuse(node1, node2)
        if (
            staged
            or nested
            or (
                device is not None
                and self.get_backend(device).has_sub_parent_epilogue(fused_nodes)
            )
        ):
            return FusionResult.fuse(True)

        if (
            node1.is_template()
            and not isinstance(node1.get_template_node(), ir.TritonTemplateBuffer)
            or node1.is_foreach()
            or node2.is_foreach()
        ):
            return FusionResult.fuse(True)

        node_list_1 = node1.get_nodes()
        device = node_list_1[0].get_device()
        if not device:
            raise AssertionError("expected device to be set")

        # Where the code is compiled rather than run, there is nothing to
        # measure: the answer is whatever the machine's own answer is, which is
        # what it was written to produce.
        if device.type == "cpu" and config.cpu_backend != "triton":
            return FusionResult.fuse(True)

        node_list_2 = node2.get_nodes()
        node_list_fused = list(itertools.chain(node_list_1, node_list_2))
        has_atomic_add = self._any_atomic_add(node_list_fused)

        # A piece that adds to a value cannot be measured by handing it made-up
        # numbers, since what it produces depends on the order they arrive in
        # and the measurement would not be measuring the real thing.  So such a
        # piece is allowed through on the ordinary reasoning instead.
        if has_atomic_add and not is_multi_template:
            return FusionResult.fuse(True)

        why = WhyNoFuse(node1, node2)

        device = node_list_fused[0].get_device()
        if device is None:
            raise AssertionError("expected device to be set")

        if is_multi_template and any(
            n.get_template_node() is not None for n in (node1, node2)
        ):
            epilogue_fusion = node1.get_template_node() is not None
            multi_node = (
                node1.get_template_node()
                if epilogue_fusion
                else node2.get_template_node()
            )
            if not isinstance(multi_node, ir.MultiTemplateBuffer):
                raise AssertionError(
                    "expected multi_node to be an ir.MultiTemplateBuffer"
                )
            # Whether the layout of the pieces agrees is settled before any
            # time is spent on compiling.
            if self._has_layout_conflict_for_template(multi_node):
                return FusionResult.fuse(False)

        # The rest is measuring, which is the backend's business: it is what
        # knows how to write a kernel twice and run both.
        backend = self.get_backend(device)
        benchmark = getattr(backend, "speedup_by_fusion", None)
        if benchmark is None:
            # The machine has no way of being asked, so the join is allowed
            # and whatever the compiled code does is what happens.
            return FusionResult.fuse(True)
        return benchmark(self, node1, node2, why)

    def _any_atomic_add(self, nodes) -> bool:
        """Whether any of these adds to a value rather than overwriting it.

        Such a piece cannot be measured by handing it made-up numbers, and two
        of them joined into one launch would be adding in an order neither asked
        for, so this is asked before anything is measured.
        """

        return any(
            getattr(node, "is_atomic_add_buffers", None) is not None
            and node.is_atomic_add_buffers()
            for node in nodes
        )

    def _has_layout_conflict_for_template(self, multi_node) -> bool:
        """Whether the pieces disagree about how a value is laid out.

        A template that produced several results was written for one
        arrangement.  Where the pieces now want a different one, the template
        cannot be used at all, so this is settled before any time is spent
        compiling anything.
        """

        return False

    def fuse_two_nodes(self, node1, node2) -> Optional:
        """Join these two if it pays, and say so either way.

        Whether they fit is asked first, since a pair that cannot be joined
        should never be put together even temporarily; then whether it is
        worth joining, which may be left to be measured later.
        """

        if not self.can_fuse(node1, node2):
            return None
        why = WhyNoFuse(node1, node2)
        speedup = self.speedup_by_fusion(node1, node2)
        if callable(speedup.callable_fn):
            # The answer is not known yet, so the pair is recorded and the
            # decision is made if it is still wanted when it is known.
            return PendingFusion(
                speedup.callable_fn, node1, node2, speedup.future
            )
        if speedup.should_fuse is False:
            why("Rejected by speedup_by_fusion: %s", speedup)
            return None
        return self.fuse(node1, node2)

    def benchmark_fused_nodes(self, nodes, benchmark_kernel: bool = False):
        """How long the joined pieces take, measured.

        Measured rather than estimated, since the whole question is whether the
        join pays and an estimate cannot answer that.  Returns the time in
        milliseconds and the code that was measured, since the second is what
        makes the first checkable.
        """

        if len(nodes) <= 0:
            raise AssertionError(f"expected nodes to be non-empty, got {len(nodes)}")
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        return backend.benchmark_fused_nodes(nodes)

    def generate_kernel_code_from_nodes(
        self, nodes, benchmark_kernel: bool, hint_override: int | None = None
    ) -> str:
        """Write out a launch for pieces that have already been joined."""

        if len(nodes) <= 0:
            raise AssertionError(f"expected nodes to be non-empty, got {len(nodes)}")
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        return backend.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override
        )

    def codegen(self) -> None:
        """Write out every launch, once the order is settled."""

        from .codegen.wrapper import PythonWrapperCodegen

        with V.graph.set_current_wrapper_code():
            V.graph.init_wrapper_code()
            V.graph.wrapper_code = PythonWrapperCodegen()
            self.wrapper_code = V.graph.wrapper_code
            self.codegen_helpers()
            for node in self.nodes:
                self.codegen_node(node)
            self.free_buffers()
            V.graph.wrapper_code.finalize()

    def codegen_helpers(self) -> None:
        """Write out whatever the code needs before any kernel is written."""

        self.current_node = None
        # the place being written into was published by the caller, which set it
        # before this ran and puts it back afterwards
        self.wrapper_code.start_writing(final=False)

    def codegen_node(self, node) -> None:
        """Write out one launch.

        Each launch is written separately, and what is held between them is
        given back as it stops being needed, so that what is held is only what
        something still wants.
        """

        self.current_node = node
        self.previous_node = None

        self.free_buffers()
        self.codegen_node_subgraph(node)

    def codegen_node_subgraph(self, node) -> None:
        """Write out the launches a group of pieces stands for.

        A group may be more than one launch -- a prepared kernel with work
        folded into its head and tail is written as one launch with the folds
        inside it -- so what it is written as is asked of the machine.
        """

        V.set_fake_mode(getattr(V.graph, "fake_mode", None))
        self.get_backend(node.get_device()).codegen_node(node)


    def get_fusion_pair_priority(self, node1, node2) -> int:
        device = node1.get_device()
        backend = self.get_backend(device)
        return backend.get_fusion_pair_priority(node1, node2)

    def merge_loops(self) -> None:
        """Join axes that are walked one after another into one axis.

        Walking one long axis is cheaper than walking several short ones, and
        where the axes follow each other in memory the result is the same.
        """

        self._body = self._body.merge_loops()
        self._sizes = self._body.sizes

        # The orders that only say which piece runs first are kept, since
        # working out how much memory this touches relies on them.  Joining axes
        # does not change how the work is divided among programs, so what was
        # worked out about that stays valid.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)

    def reorder_loops_by_dep_pair(self, self_dep, other_dep) -> bool:
        """Walk the axes in the order that makes two accesses match.

        Two pieces can only be joined if what one reads at a position is what
        the other writes there, which depends on the order the axes are walked
        in.  Where an order can be found that makes them match, it is taken.
        """

        new_order = None
        self_sizes = self._sizes[0]
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if new_order:
            metrics.num_loop_reordering += 1
            loop_ordering_log.debug(
                "Reorder loops for %s with order %s", self.get_name(), new_order
            )
            self.apply_new_loop_order(new_order)
            return True
        else:
            loop_ordering_log.debug(
                "Don't reordering %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            if not isinstance(dep, dependencies.WeakDep):
                buf_name = dep.name
                buf = V.graph.get_buffer(buf_name)
                if not isinstance(buf, ir.TorchBindObject):
                    lines.append(f"{buf_name}_layout = {pformat(getattr(buf, 'layout', None))}")
        if self._body is not None and hasattr(self._body, "debug_str"):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        if self.node is None:
            raise AssertionError("expected self.node to be set")
        lines.extend(self._debug_str_for_device())

        return "\n".join(lines)

    def get_ranges(self) -> Sequence:
        return self._sizes

    def is_reduction(self) -> bool:
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")

        # A body that carries part of the reduction means the reduction has
        # already been turned into ordinary work, and the operation itself has
        # not been changed to match -- so the body is what decides here.
        return bool(self.node.get_reduction_type()) and (
            self._body is None or not self._body.has_partial_accumulate
        )

    def is_native_matmul(self) -> bool:
        if not isinstance(self.node, ir.ComputedBuffer):
            raise AssertionError(f"{type(self.node)=}")
        return self.node.get_reduction_type() == "dot"

    def is_split_scan(self) -> bool:
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self):
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars) -> None:
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def ranges_from_index_vars(self, index_vars: Sequence) -> dict:
        """How far each axis goes, given the axes this piece is being run with.

        A position means something different once the axes have been divided up
        differently, and this is what says by how much -- which is what lets the
        expressions be simplified against the ranges they are actually in.
        """

        sizes = self._sizes
        if sum(map(len, sizes)) != sum(map(len, index_vars)):
            raise AssertionError("expected sum of sizes to equal sum of index_vars")
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence) -> None:
        """Write out this piece's body, over the axes it is being run with.

        The indexing expressions are simplified against the ranges the axes
        actually go over, since that is what the generated code will assume.
        """

        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with (
                V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)),
                V.kernel.set_current_node(self),
            ):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_or_reduction_read_writes(self, pointwise: bool = True):
        """What this reads and writes in one set of axes or the other.

        The two halves of a reduction are separate questions: what is read for
        each element before reducing, and what is read while reducing.  Asking
        for one leaves the other out by holding its positions at zero.
        """

        keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
        return dependencies.extract_read_writes(
            self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
        )

    @cache_on_self
    def pointwise_read_writes(self):
        """What this reads and writes in the axes that are not reduced."""

        return self.pointwise_or_reduction_read_writes(pointwise=True)

    @cache_on_self
    def reduction_read_writes(self):
        """What this reads and writes in the reduced axes."""

        return self.pointwise_or_reduction_read_writes(pointwise=False)

    def can_inplace(self, read_dep) -> bool:
        """Whether this may write into a buffer it was given.

        Only where the value being written is the same value that was read: the
        same elements, in the same order.  Anything else would change what the
        buffer holds, and whatever else reads it would see the change.
        """

        if self.is_template():
            return False
        if any(out.get_aliases() for out in self.get_outputs()):
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            if not isinstance(write_dep, dependencies.MemoryDep):
                raise AssertionError(f"{type(write_dep)=}")
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet:
        """Which buffers are added to rather than overwritten.

        A buffer that several programs add to at once cannot be written the
        ordinary way, and knowing which those are is what decides that.
        """

        buffers_store_as_atomic_add: OrderedSet = OrderedSet()
        if self._body is not None and hasattr(self._body, "get_nodes"):
            for node in self._body.get_nodes():
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        return buffers_store_as_atomic_add

    @cache_on_self
    def has_side_effects(self) -> bool:
        # The body is not always there, which is why this is asked of both.
        if self._body is not None and self._body.has_op("device_assert_async"):
            return True
        return super().has_side_effects()


def refresh_group_node_dependencies(group_snode) -> None:
    """Work out again what a group of pieces reads and writes, all together.

    What the group does is what its members do, and what it waits for is what
    any of them waits for -- except what the group itself produces, since a
    piece cannot wait for something the group is producing.
    """

    snodes = group_snode.snodes
    group_snode.set_read_writes(
        dependencies.ReadWrites.merge_list([x.read_writes for x in snodes])
    )

    group_snode.unmet_dependencies = (
        OrderedSet(
            dep
            for dep in OrderedSet.union(*[x.unmet_dependencies for x in snodes])
            if dep.name not in group_snode.get_buffer_names()
        )
        - group_snode.read_writes.writes
    )


def init_group_node(group_snode, scheduler, snodes: list) -> None:
    """Set a group of pieces up as though it were one piece.

    A group has no operation of its own -- what it does is what its members do
    -- but everything asked of a piece is asked of it too: what it reads, what
    it produces, where it sits among the others.  What makes it a group is only
    that it stands for several.
    """

    if not isinstance(group_snode, (FusedSchedulerNode, GroupedSchedulerNode)):
        raise AssertionError("expected FusedSchedulerNode or GroupedSchedulerNode")
    group_snode.snodes = snodes
    group_snode.scheduler = scheduler
    group_snode.node = None
    group_snode.ancestors = OrderedSet.union(
        *[x.ancestors for x in snodes if x.ancestors is not None]
    )

    refresh_group_node_dependencies(group_snode)

    group_snode.min_order = min(x.min_order for x in group_snode.snodes)
    group_snode.max_order = max(x.max_order for x in group_snode.snodes)
    group_snode.min_input_distance = min(
        x.min_input_distance for x in group_snode.snodes
    )
    group_snode.max_input_distance = max(
        x.max_input_distance for x in group_snode.snodes
    )
    group_snode.outputs_by_name = {
        buf.get_name(): buf for buf in group_snode.get_outputs()
    }
























@dataclasses.dataclass







































class _LoopStateSnapshot:
    """What the loops looked like, so that a change to them can be undone.

    Both what each piece's own loops were and what each group's shape was are
    kept, since the second is set directly after the first has been changed and
    has nothing watching it.
    """

    scheduler_node_states: dict = dataclasses.field(default_factory=dict)
    fused_node_groups: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def create(cls, nodes: tuple) -> "_LoopStateSnapshot":
        """Take note of every piece's loops before anything changes them."""

        snapshot = cls()
        for node in _iter_loop_state_nodes(nodes):
            if isinstance(node, FusedSchedulerNode):
                snapshot._snapshot_fused_node(node)
            else:
                snapshot._snapshot_scheduler_node(node)
        return snapshot

    def _snapshot_scheduler_node(self, sn) -> None:
        if sn in self.scheduler_node_states:
            raise AssertionError(f"scheduler node {sn} already snapshotted")
        self.scheduler_node_states[sn] = sn.snapshot_loop_state()

    def _snapshot_fused_node(self, node) -> None:
        if node in self.fused_node_groups:
            raise AssertionError(f"fused node {node} already snapshotted")
        self.fused_node_groups[node] = node.group

    def restore(self) -> None:
        for sn, state in self.scheduler_node_states.items():
            sn.restore_loop_state(state)
        for node, group in self.fused_node_groups.items():
            node.group = group
            refresh_group_node_dependencies(node)


@dataclasses.dataclass
class _LoopMutationTracker:
    """A way to put the loops back if a decision turns out to be no.

    Deciding whether two pieces may be joined involves trying arrangements of
    the loops that would make them fit.  Where the answer is no, those trials
    must not be left behind -- or the next piece considered would be judged
    against loops that were only ever right for a join that did not happen.

    Where deciding one thing involves deciding another, each adds its own
    listener without displacing the one already there, so the outer decision
    still sees everything that happened inside it.
    """

    nodes: tuple
    watched_nodes: OrderedSet = dataclasses.field(default_factory=OrderedSet)
    previous_listeners: dict = dataclasses.field(default_factory=dict)
    state: Any = None

    @classmethod
    def create(cls, nodes: tuple) -> "_LoopMutationTracker":
        """Start watching, so that the first change to any loop is noticed."""

        seen = OrderedSet(nodes)
        tracker = cls(nodes=tuple(seen))
        for node in _iter_loop_state_nodes(seen):
            if isinstance(node, SchedulerNode):
                tracker.watch(node)
        return tracker

    def watch(self, sn) -> None:
        """Be told when this piece's loops are about to change."""

        if sn in self.watched_nodes:
            return
        self.previous_listeners[sn] = sn._loop_mutation_listener
        self.watched_nodes.add(sn)
        sn._loop_mutation_listener = self.track

    def track(self, sn) -> None:
        """Take the copy, the first time anything changes.

        Taken here rather than when watching began, because until something
        actually changes there is nothing to undo, and taking a copy of every
        candidate up front would cost more than the decision usually does.
        """

        if sn not in self.watched_nodes:
            raise AssertionError(f"scheduler node {sn} is not being watched")
        if previous := self.previous_listeners[sn]:
            previous(sn)
        if self.state is not None:
            # What is kept is how the loops were before anything at all
            # changed, for as long as this is deciding.
            return

        # Being told a piece's loops changed is not enough on its own: a group's
        # shape is set directly and nothing would tell us about that, so the
        # whole set of candidates is copied here.
        self.state = _LoopStateSnapshot.create(self.nodes)

    def finish(self, *, rollback: bool) -> None:
        """Stop watching, and put the loops back where the decision says."""

        for sn in self.watched_nodes:
            sn._loop_mutation_listener = self.previous_listeners[sn]
        if not rollback or self.state is None:
            return
        self.state.restore()


# Tells "not looked up yet" apart from "looked up, and there is nothing here",
# which would otherwise both be nothing.
_TILING_MEMORY_MISS = object()


# Where the run-time estimates are kept between compilations, and the counters
# that record what this compilation did.
