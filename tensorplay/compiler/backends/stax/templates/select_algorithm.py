"""Choosing what runs: the built choice, the deferred pairing, and the operation itself.

Everything a call can be answered by is a choice, and a choice answers the
same questions: where its result lands, how to run it, how to measure it, and
whether it fits this call at all.  The operation the templates are measured
against is one of those choices rather than a fallback taken when measurement
fails, which is why it is registered here beside the rest.
"""
from __future__ import annotations

import contextlib
import dataclasses
from collections import defaultdict
import functools
import json
import sys
import hashlib
import itertools
import logging
import operator
import os
import sympy
import textwrap
from unittest.mock import patch
from typing import NamedTuple

from typing import Any, Callable, Iterator

import tensorplay as tp

from .. import config
from ..codegen.common import KernelTemplate
from ..kernel_inputs import KernelInputs
from ..ir import (
    BaseView,
    Buffer as _Buffer,
    ComputedBuffer,
    ExternKernel,
    FlexibleLayout,
    IRNode,
    InputBuffer,
    Layout,
    ReinterpretView,
    compute_required_storage_length,
)
from ..loops import V
from ..ir import ChoiceCaller, TritonTemplateCallerBase
from .mm_common import use_aten_gemm_kernels
from ..loops import contiguous_strides, dtype_name
from ..heuristics.template.params import DictKernelTemplateParams, KernelTemplateParams
from ..runtime.triton_helpers import get_constexprs
from ..heuristics.registry import (
    get_template_heuristic as registry_get_template_heuristic,
)
from .triton import CHOICES
from ..codegen.subgraph import SubgraphChoiceCaller
from ..codegen.simd import IterationRangesEntry, IterationRangesRoot
from ..codegen.common import CSE, IndentedBuffer, OpOverrides
from ..codegen.triton import IndexingOptions, texpr, TritonSymbols
from ..utils import (
    get_dtype_size,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    triton_type,
    triton_type_to_torch,
    unique,
)
from ..kernel_scheduler import count_flops_fx
from ..ops_handler import StoreMode
from ..heuristics.template.base import SymbolicGridFn
from ..runtime.triton_heuristics import FixedGrid
from ..codegen.wrapper import pexpr
from tensorplay.graph.experimental.sympy_functions import OrderedSet
from ..autotune_process import TritonBenchmarkRequest
from ..utils import do_bench_using_profiling
from pathlib import Path
from tensorplay.utils._filelock import FileLock
from ..compile_log import timed_block, trace_structured
from ..codecache import PersistentCache
from ..utils import counters, restore_stdout_stderr
from ..codegen.triton import TritonKernel
from ..codegen.simd_kernel_features import SIMDKernelFeatures
from ..codegen.common import WorkspaceArg
from ..loop_body import identity
from ..utils import FakeIndentedBuffer, sympy_product
from ..runtime.hints import TritonMeta
from ..autotune_process import PrecompileThreadPool, use_pipelined_autotuning
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from tensorplay.testing import assert_close

#: Whether a candidate's result is checked against what the operation's own
#: kernel produced.  On by default because a template that computes the wrong
#: number faster is not an answer, and the check costs one run per candidate.
VERIFY = os.environ.get("TP_AUTOTUNE_VERIFY", "1") == "1"

#: Where this module's messages go.  A measurement is worth a line and a
#: measurement that was thrown away is worth none, so the numbers are here
#: rather than printed.
# A verbosity switch for the tuning itself, kept beside the tuning rather than
# in the settings, because it is read only here and only while tuning.
DEBUG = False

log = logging.getLogger(__name__)


class GeneratedCodeCacheEntry(NamedTuple):
    """What one rendering of a template produced."""

    code: str
    extra: str
    events: list


class GeneratedCodeCache:
    """Rendered code, kept per template so the same call is not rendered twice.

    The key is everything the rendering depended on: which values were read,
    how the launch was shaped, and what was imposed on every configuration.
    A key that cannot be built is no key at all, and the entry is then neither
    read nor written -- a partially known call must not be answered from a
    cache entry that some other call wrote.
    """

    def __init__(self, *args, **kwargs):
        self._cache: dict[str, GeneratedCodeCacheEntry] = {}

    def cache_clear(self) -> None:
        self._cache.clear()

    def __repr__(self):
        return repr(self._cache)

    def make_key(self, **keyed) -> str | None:
        """The identity of one rendering, or ``None`` when it cannot be pinned.

        Every part has to be known for the entry to be reusable, so a part
        that is not is what makes the whole key unusable rather than being
        skipped.
        """

        if any(part is None for part in keyed.values()):
            return None
        return "|".join(f"{name}={part!r}" for name, part in sorted(keyed.items()))

    def get_entry(self, cache_key: str | None) -> GeneratedCodeCacheEntry | None:
        if cache_key is None:
            return None
        return self._cache.get(cache_key, None)

    def put_entry(
        self,
        cache_key: str | None,
        code: str,
        extra: str,
        events: list,
    ) -> None:
        if cache_key is None:
            return
        self._cache[cache_key] = GeneratedCodeCacheEntry(code, extra, events)


_registered_caches: list = []


def clear_on_fresh_cache(obj: Any) -> Any:
    """Register a cache to be emptied whenever the caches are declared fresh.

    A template that renders the same call twice is only wasteful until the
    second rendering disagrees with the first, which it can when the region
    changed underneath it.  So every cache a template holds is emptied
    together, rather than each template deciding on its own.
    """

    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def fresh_caches() -> None:
    """Empty every registered cache."""

    for obj in _registered_caches:
        obj.cache_clear()


class KernelTemplateChoice:
    """One template, one configuration, and the choice they make -- eventually.

    The kernel is built the first time the choice is asked for and then kept,
    including when the build turns out not to apply: a configuration that does
    not fit this call is an answer, not a retry.
    """

    def __init__(
        self,
        template: KernelTemplate,
        params: KernelTemplateParams,
        extra_kwargs: dict[str, Any],
        layout: Layout | None,
        inputs: KernelInputs,
        plain_launch: Any = None,
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = dict(extra_kwargs)
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}
        self._plain_launch = plain_launch

    @property
    def key(self) -> tuple:
        """What identifies this choice across processes."""

        return (
            self.template.uid,
            repr(self.params.to_serializeable_dict()),
            repr(self.layout),
            str(self.inputs.device),
        )

    @property
    def choice(self):
        """The built choice, made on the first ask and kept afterwards.

        Building is deferred to here because most configurations are never
        reached: a table is offered whole, one entry is measured and wins, and
        the rest are discarded without ever being built.  A refusal is kept just
        like a success, because a configuration that does not apply to this call
        is an answer rather than something to retry.
        """

        return self.resolve(self._plain_launch)

    def resolve(self, plain_launch):
        """The launcher for this configuration, or ``None`` when it does not fit."""

        if not hasattr(self, "_resolved"):
            self._resolved = True
            try:
                self._choice = self.template.choice_or_none(
                    **self.params.to_kwargs(),
                    **self.extra_kwargs,
                    layout=self.layout,
                    input_nodes=self.inputs.nodes(),
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice

    def __repr__(self) -> str:
        return f"KernelTemplateChoice({self.template.name}, {self.params!r})"
def make_ktc_generator(
    template: KernelTemplate,
    cs: Iterator[KernelTemplateParams],
    extra_kwargs: dict[str, Any],
    overrides: dict[str, Any],
    layout: Layout | None,
    inputs: KernelInputs,
) -> Iterator[KernelTemplateChoice]:
    """One deferred choice per configuration, with the overrides folded in.

    The overrides are what a caller imposes on every configuration -- the
    parts of the call that are not a choice -- and folding them into the
    parameters here means a template reads one dict.
    """

    for params in cs:
        merged = {**params.to_kwargs(), **overrides}
        yield KernelTemplateChoice(
            template=template,
            params=DictKernelTemplateParams(merged),
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
class KernelNamespace:
    """A place to hang the operations a generated wrapper calls by name.

    The names have to survive into generated text, so they are held on an
    object rather than only in the table that made them: what codegen writes
    is a reference to this, and what it resolves to at run time is whatever was
    registered under that name.
    """


#: The operations a generated wrapper reaches by name.
extern_kernels = KernelNamespace()

#: Stands in for a kernel's workspace in an argument list that has to cross a
#: process boundary, where a workspace cannot be sent.  The reader puts a real
#: one in its place.
WORKSPACE_ARG_PLACEHOLDER = "ws_placeholder"


def call_operation(name, kernel, layout, kwargs=None, has_out_variant=True):
    """One call of an operation, registering the operation if it is new.

    A launcher the compiler was handed is an operation like any other: it just
    happens to have arrived already built rather than named.  It is registered
    the same way, so that everything downstream -- the trace, the generated
    call, the identity a cache keys on -- sees one kind of thing.
    """

    if kernel is None:
        return None
    choice = ExternKernelChoice.lookup(name)
    if choice is None or choice.kernel is not kernel:
        choice = ExternKernelChoice(
            kernel, name=name, has_out_variant=has_out_variant
        )
    return choice.bind(input_nodes=(), layout=layout, **(kwargs or {}))


class ExternKernelCaller(ChoiceCaller):
    """A call to an operation that already exists, built and ready to measure.

    The choice it was made from is kept rather than copied apart, because
    everything still needed after the call is built -- which overload the
    operation is, whether it writes into a buffer the caller supplied or one it
    allocates, the name a trace shows it under -- is a property of the
    operation, and is the same however many times it is called.
    """

    def __init__(
        self,
        choice,
        input_nodes=(),
        layout=None,
        kwargs=None,
        *,
        has_out_variant=True,
    ):
        super().__init__(choice.name, input_nodes, layout, description="")
        self.choice = choice
        self.kwargs = kwargs or {}
        #: Whether the operation writes into memory the caller named or memory
        #: it allocates is a property of the operation, and it decides what kind
        #: of node describing this call turns out to be.
        self.has_out_variant = has_out_variant
        self.gm = choice.gm
        self._callable = choice.to_callable()

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of the operation takes.

        Where the operation writes into a buffer the caller supplied, that
        buffer is named rather than appended: the operation's own signature
        says which of its arguments it is, and appending it would be a
        statement about the order rather than the name.  An operation that
        returns its result is simply called.
        """

        from ..runtime.stax_autotune import bench_launch

        algo = self.to_callable()
        if self.has_out_variant and out is not None:
            def launch(_operands, _algo=algo, _args=args, _out=out):
                return _algo(*_args, out=_out)
        else:
            def launch(_operands, _algo=algo, _args=args):
                return _algo(*_args)
        return bench_launch(launch, [])

    def __str__(self) -> str:
        return f"ExternKernelCaller({self.choice.call_name()})"

    def call_name(self) -> str:
        """The name a trace shows this call under."""

        return self.choice.call_name()

    def to_callable(self):
        return self.choice.to_callable()

    def hash_key(self) -> str:
        """What identifies this call: the operation, and the arguments beside it.

        Two calls of one operation with the same arguments are the same call,
        so the key is the operation's own identity followed by the arguments
        written out in a fixed order.
        """

        return "-".join(
            [
                self.choice.name,
                *[
                    f"{kwarg}={self.kwargs[kwarg]!r}"
                    for kwarg in sorted(self.kwargs.keys())
                ],
                self.choice.hash_key(),
            ]
        )

    def output_node(self):
        """The result of making this choice, as a value the rest can read.

        A choice that is a library call has no source to render, so asking it
        for its result is asking it to describe the call: what is written, what
        is passed, and where the answer goes.  The node that comes back is a
        description rather than a number, and becomes a number when the region
        is finally written out -- which is why it can be made now and measured
        later without being measured twice.
        """

        from .. import ir

        if self.choice.use_fallback_kernel:
            # The operation is one that was described rather than wrapped, so
            # what the call needs to be described comes from the overload
            # itself.  A description that names no overload names nothing.
            if self.choice.op_overload is None:
                raise AssertionError(
                    "a call described as a fallback has to say which overload "
                    "it is, or there is nothing to call"
                )
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
        elif self.choice.kernel_creator is not None:
            inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
        else:
            cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
            inner = cls(
                layout=self.layout,
                inputs=self.input_nodes,
                python_kernel_name=self.choice.call_name(),
                cpp_kernel_name=self.choice.cpp_kernel_name,
                ordered_kwargs_for_cpp_kernel=(
                    self.choice.ordered_kwargs_for_cpp_kernel
                ),
                op_overload=self.choice.op_overload,
                kwargs=self.kwargs,
            )
        if "ktc" in self.annotations:
            inner.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(inner)

    def info_dict(self) -> dict:
        """What is worth writing down about this choice."""

        return {
            "backend": "extern",
            "kernel_call_name": self.choice.call_name(),
        }

    def autoheuristic_id(self) -> str:
        return f"extern_{self.choice.name}"


def template_subgraph_index_dtype_nodes(
    subgraphs,
) -> tuple[Any, ...]:
    """The buffers a set of subgraphs read or write, for choosing an index width.

    A kernel that indexes any of its operands in 64 bits has to do all of it
    in 64 bits, so the whole set has to be looked at before a width is chosen.
    """

    if subgraphs is None:
        return ()

    nodes: list[Any] = []
    seen_names: OrderedSet[str] = OrderedSet()
    pending: list[Any] = list(reversed(subgraphs))
    while pending:
        subgraph = pending.pop()
        if isinstance(subgraph, (list, tuple)):
            pending.extend(reversed(subgraph))
            continue
        if subgraph is None:
            continue
        for dep in subgraph.get_read_writes().reads_and_writes():
            if dep.name in seen_names:
                continue
            buffer = V.graph.try_get_buffer(dep.name)
            if buffer is not None:
                nodes.append(buffer)
                seen_names.add(dep.name)
    return tuple(nodes)


class ModificationWrapper(V.WrapperHandler):
    """Stands in for the graph's own operations while a subgraph is rewritten.

    A subgraph that a template asks to have modified is not lowered as it was
    written: some of its inputs are the caller's own values rather than the
    kernel's operands, and one of its results is wanted as a value rather than
    as a store.  Lowering it through this wrapper turns those into the reads
    and the assignment the modified form needs, and leaves everything else to
    be lowered as usual.
    """

    def __init__(
        self,
        kernel,
        subgraph_number: int,
        fixed_inputs: dict[str, Any],
        mask: str | None,
        input_shapes: dict[str, tuple[str, ...]] | None = None,
        input_dtypes=None,
    ):
        super().__init__(V.ops)
        self.name = f"PlaceholderSubstitution_{subgraph_number}"
        self.kernel = kernel
        self.fixed_inputs = fixed_inputs
        self.mask = mask
        self.input_shapes = input_shapes or {}
        self.input_dtypes = input_dtypes or {}
        extra_input_shapes = self.input_shapes.keys() - self.fixed_inputs.keys()
        extra_input_dtypes = self.input_dtypes.keys() - self.fixed_inputs.keys()
        if extra_input_shapes:
            raise AssertionError(
                f"input_shapes keys must match fixed inputs: {extra_input_shapes}"
            )
        if extra_input_dtypes:
            raise AssertionError(
                f"input_dtypes keys must match fixed inputs: {extra_input_dtypes}"
            )

    def load(self, name: str, index: sympy.Expr):
        """Read something the subgraph wanted: an operand, or the caller's value."""

        if name not in self.fixed_inputs:
            index_str = self._process_indexing(index)
            var = self._add_kernel_input(name)
            buffer = V.graph.get_buffer(name)
            var_dtype = buffer.dtype
            line = f"tl.load({var} + {index_str})"

            if (
                var_dtype in (tp.float16, tp.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                var_dtype = tp.float32

            out = self.kernel.cse.generate(
                self.kernel.compute,
                line,
                dtype=var_dtype,
                shape=TritonSymbols.get_block_shape(index),
            )
            return out

        shape = self.input_shapes.get(name, ())
        return self.kernel.cse.generate(
            self.kernel.compute,
            f"({self.fixed_inputs[name]})",
            dtype=self._fixed_input_dtype(name),
            shape=shape,
        )

    def _index_dtype(self) -> tp.dtype:
        return tp.int64 if self.kernel.index_dtype == "tl.int64" else tp.int32

    def _normalize_input_dtype(self, dtype):
        if isinstance(dtype, tp.dtype):
            return dtype
        if dtype == "index":
            return self._index_dtype()
        raise AssertionError(f"Unexpected fixed input dtype: {dtype}")

    def _fixed_input_dtype(self, name: str) -> tp.dtype:
        """What type a caller-supplied value is, worked out from whatever it is.

        The caller may name the type, or hand over a value whose type is
        already known, or hand over a plain Python value -- and a plain bool is
        not a one-byte integer as far as a kernel is concerned.
        """

        if name in self.input_dtypes:
            return self._normalize_input_dtype(self.input_dtypes[name])

        value = self.fixed_inputs[name]
        if isinstance(value, CSEVariable) and value.dtype is not None:
            return value.dtype
        if isinstance(value, str):
            cse_value = self.kernel.cse.varname_map.get(value)
            if cse_value is not None and cse_value.dtype is not None:
                return cse_value.dtype
        if isinstance(value, bool):
            return tp.bool
        if isinstance(value, float):
            return tp.float32
        return tp.float32

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """An index that was named rather than computed."""

        return sympy_index_symbol(str(index_var))

    def store(
        self, name: str, index: sympy.Expr, value: "CSEVariable", mode: StoreMode = None
    ) -> str:
        """An accumulation into a buffer, which is the only store allowed here.

        Everything else in a subgraph is a read or a result; a store would be
        a write the caller did not ask for and cannot see.  An accumulation is
        different -- it is how a subgraph contributes to something the caller
        already owns -- but it is still only allowed where the caller said
        which of the elements are really there.
        """

        if self.mask is None:
            raise AssertionError("Mask is required for inner stores in modifications")
        if mode != "atomic_add":
            raise AssertionError("Only atomic_add is supported for inner stores")

        buf_name = self._add_kernel_input(name)
        index_str = self._broadcast_index(index, f"{value}.shape")
        return f"tl.atomic_add({buf_name} + {index_str}, {value}, {self.mask}, sem='relaxed')"

    def _add_kernel_input(self, name: str) -> str:
        return self.kernel.args.input(name)

    def _process_indexing(self, index: sympy.Expr) -> str:
        return self.kernel.kexpr(self.kernel.rename_indexing(index))

    def _broadcast_index(self, index: sympy.Expr, shape: str) -> str:
        """An index written so it has the shape of what it indexes into.

        A single index and a number are both one element, and neither of them
        has a shape the runtime can broadcast on its own, so each is given one.
        """

        index = sympy.sympify(index)
        index_str = self._process_indexing(index)
        index_shape = TritonSymbols.get_block_shape(index)
        if (
            index_shape
            and len(index_shape) == 1
            and all(
                V.graph.sizevars.statically_known_equals(sympy.sympify(d), 1)
                for d in index_shape
            )
        ):
            return f"tl.broadcast_to(tl.reshape({index_str}, []), {shape})"
        if not index_shape and len(index.free_symbols) == 0:
            return f"tl.full({shape}, {index_str}, INDEX_DTYPE)"
        return f"tl.broadcast_to({index_str}, {shape})"


# Function name, followed by args and kwargs.
RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]


@dataclasses.dataclass()
class SubgraphInfo:
    """What is needed to lower one subgraph of a kernel written as a template.

    The template's text is written once and then rendered once, but the parts
    of it that belong to a subgraph are worked out while that subgraph is
    lowered.  So each subgraph keeps its own text and its own bookkeeping, and
    a kernel swaps those in while it works on one subgraph and swaps them back
    when it is done.
    """

    body: IndentedBuffer
    template_mask: str | None = None
    template_out_shape: str | tuple[str] | None = None
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: V.WrapperHandler | None = None
    cse: CSE[Any, str] | None = None

    # Only carried over when they were made, because a subgraph that made
    # none of them shares the kernel's rather than replacing it with nothing.
    range_trees: list[IterationRangesRoot] | None = None
    range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry] | None = None
    numels: dict[str, sympy.Expr] | None = None

    # Maps a range-tree root's name to the name the prologue gave it.
    root_var_renames: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.only_copy_if_non_none_fields = (
            "range_trees",
            "range_tree_nodes",
            "numels",
            "cse",
        )

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class ExternKernelChoice:
    """An operation that can hold its own against a kernel, as a choice.

    The operation every template is measured against is not a fallback taken
    when measurement fails -- it is one of the things being measured, and it
    wins whenever no kernel beats it.  So it is registered here, beside the
    templates, under a name codegen can refer to, and it answers the same
    questions they do; a caller can hand a list of either to the same
    enumeration without caring which is which.

    Each instance registers once under its name.  Registering the same
    callable twice is tolerated, because a module can be initialised twice in
    one process; registering a *different* callable under a name already taken
    is refused, because that is a collision rather than a re-registration.
    """

    _registry: dict[str, "ExternKernelChoice"] = {}

    def __init__(
        self,
        kernel: Callable[..., Any],
        cpp_kernel: str | None = None,
        *,
        name: str | None = None,
        has_out_variant: bool = True,
        op_overload: Any = None,
        use_fallback_kernel: bool = False,
        kernel_creator: Callable[..., Any] | None = None,
    ):
        name = name or getattr(kernel, "__name__", None)
        if not callable(kernel):
            raise AssertionError("an extern choice has to wrap something callable")
        existing = getattr(extern_kernels, name, None)
        if existing is not None and existing is not kernel:
            raise AssertionError(f"duplicate extern kernel: {name}")
        self.kernel = kernel
        self.name = name
        #: The name the same operation goes by when it is emitted as C++, and
        #: the graph it stands for when the choice is a whole region.  Both are
        #: absent for an ordinary library call, and both are recorded rather
        #: than inferred: a name that had to be guessed would be a name that
        #: could be guessed wrong.
        self.cpp_kernel_name = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator
        self.ordered_kwargs_for_cpp_kernel: tuple = ()
        #: There is no source for an operation that is not written out here, so
        #: there is nothing to hash: the operation is the kernel, and the name
        #: is the whole of its identity.
        self.src_hash: str | None = None
        self.gm: Any = None
        ExternKernelChoice._registry[name] = self

    @classmethod
    def lookup(cls, name: str) -> "ExternKernelChoice | None":
        return cls._registry.get(name)

    def to_callable(self) -> Callable[..., Any]:
        """The operation as it is run when the result goes somewhere named.

        The name under which the operation is registered is the function a
        caller would reach for, which for an operation that can write into a
        buffer the caller supplied is not the same as the form of it that
        takes that buffer: the first returns its result and the second is
        handed where to put it.  A measurement hands it a buffer, so what it is
        given is the form that takes one -- and an operation with no such form
        is the function itself.
        """

        if self.op_overload is not None:
            return self.op_overload
        return getattr(extern_kernels, self.name)

    def call_name(self) -> str:
        """The name a trace shows this operation under."""

        return f"extern_kernels.{self.name}"

    def hash_key(self) -> str:
        """What identifies the operation itself.

        The name is the identity, but two operations can share one across
        different builds of the library, so the function behind the name and
        its source are hashed as well where the source can be read at all.
        """

        import inspect

        from ..codecache import code_hash

        fn = self.to_callable()
        parts = [
            self.name,
            getattr(fn, "__name__", ""),
            getattr(fn, "__module__", ""),
        ]
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            pass
        return code_hash("-".join(parts))

    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=(), **kwargs):
        """This operation, as one particular call of it."""

        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            self,
            input_nodes,
            layout,
            kwargs,
            has_out_variant=self.has_out_variant,
        )

    @property
    def uid(self) -> str:
        """Namespaced by kind, so two kinds may share a name."""

        return f"framework::{self.name}"

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The operation itself, as the choice it always is.

        Carries across what the call needs in order to be described rather than
        merely named: the values it is handed, and whether it writes into a
        buffer the caller supplied or one it allocates.
        """

        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        # Convenience function to match the template interface, so that
        # templates and operations can be treated the same when generating
        # choice callers.
        if "input_nodes" not in kwargs:
            raise AssertionError("input_nodes argument required")
        if "layout" not in kwargs:
            raise AssertionError("layout argument required")
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        choices.append(
            self.bind(input_nodes=input_nodes, layout=layout, **kwargs)
        )
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        return self.choice_or_none(**kwargs)

    def __repr__(self) -> str:
        return f"ExternKernelChoice({self.name})"


class TritonChoiceCaller(ChoiceCaller):
    """A choice whose kernel is emitted as source for a streaming backend.

    What makes it a separate kind rather than a flag is the hash key: a kernel
    emitted as source is recognised in a cache by the source itself, so a
    change to the emitter invalidates every stored decision that named it,
    with no separate version to keep in step.
    """

    def __init__(self, name, input_nodes=(), layout=None, description="",
                 source: str = "", src_hash: str | None = None,
                 launcher_args: tuple | None = None,
                 num_stages: int = 2, num_warps: int = 4,
                 config: dict | None = None, operands: dict | None = None,
                 inductor_meta: dict | None = None, template: Any = None,
                 make_kernel_render: Callable[[], Any] | None = None,
                 mutated_inputs: tuple = (),
                 allowed_prologue_inps: Any = None):
        super().__init__(name, input_nodes, layout, description)
        self.source = source
        self._src_hash = src_hash
        #: How many steps of the contraction are held at once, and how many
        #: programs run.  How many are live at once is what decides how much
        #: memory the kernel needs, so a configuration that does not fit the
        #: device is tried again with fewer of them.
        self.stages = num_stages
        self.warps = num_warps
        #: The configuration this was built from, and what it was written
        #: against, kept so that the same text can be compiled again with one
        #: thing changed rather than guessed at again from the shape.
        self.config = dict(config or {})
        self.operands = dict(operands or {})
        self.inductor_meta = dict(inductor_meta or {})
        self.template = template
        #: The text this kernel's body is written out by, the inputs it is
        #: allowed to read, and the inputs it writes rather than reads.  Carried
        #: because a kernel written from a template returns nothing: its result
        #: is a buffer that will be written, and writing it needs the text.
        self.make_kernel_render = make_kernel_render
        self.mutated_inputs = mutated_inputs
        self.allowed_prologue_inps = allowed_prologue_inps
        #: What a launcher written from this kernel is handed, in the order it
        #: declares: the operands of the call, then the extents and strides the
        #: kernel was written to be given, then the stream.  A measurement is
        #: handed the operands and has to say the rest, and it can only say the
        #: rest if the caller settled it when the launcher was written -- which
        #: is the one place that knows the signature.
        self.launcher_args = launcher_args

    def bind(self, launcher: Callable[..., Any]) -> "TritonChoiceCaller":
        """Attach the launcher written from this kernel, and hand back the choice.

        The launcher is what a measurement runs and what a launch runs, and it
        is not the kernel: what a compile hands back cannot be called on its
        own, being callable only from inside a launch it has set up itself --
        with the grid worked out and the arguments in the order its signature
        declares.
        """

        self._callable = launcher
        return self

    def call_name(self) -> str:
        """The name a trace shows this kernel under."""

        return self.name

    def to_callable(self) -> Callable[..., Any]:
        """The launcher, which is what this choice is run through."""

        if self._callable is None:
            raise NotImplementedError(f"{self.name} has no kernel bound to it")
        return self._callable

    def info_dict(self) -> dict:
        """What is worth writing down about this choice.

        A measurement that will be looked at later is worth reading later, and
        a row of timings with no configuration beside it says only that
        something was faster.  What the configuration was -- the tile, the
        warps and stages, and the handful of values the body was told -- is
        what turns a number back into a thing that can be chosen again.
        """

        config = self.config or {}
        return {
            "tile_shape": str(
                (
                    config.get("BLOCK_M", -1),
                    config.get("BLOCK_K", -1),
                    config.get("BLOCK_N", -1),
                )
            ),
            "num_stages": self.stages,
            "num_warps": self.warps,
            "GROUP_M": config.get("GROUP_M", -1),
            "allow_tf32": str(config.get("ALLOW_TF32")),
            "acc_type": str(config.get("ACC_TYPE")),
            "matrix_instr_nonkdim": config.get("matrix_instr_nonkdim", 0),
            "waves_per_eu": config.get("waves_per_eu", 0),
            "kpack": config.get("kpack", 2),
            "epilogue_subtile": config.get("EPILOGUE_SUBTILE", 0),
        }

    def output_node(self) -> Any:
        """The result this choice produced, as a value the rest can read.

        A kernel written from a template returns nothing -- what it computed is
        left in memory it was given -- so the result is a buffer that will be
        written rather than a value that was returned, and the text that writes
        it is carried alongside so that whoever writes the region out has it.
        """

        from .. import ir

        buffer = ir.TritonTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            mutated_inputs=self.mutated_inputs,
            allowed_prologue_inps=self.allowed_prologue_inps,
        )
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)

    def hash_key(self) -> str:
        parts = [self.name, self.description]
        if self.layout is not None:
            parts.append(repr(self.layout))
        digest = self._src_hash
        if digest is None and self.source:
            digest = hashlib.sha1(self.source.encode()).hexdigest()[:16]
        if digest is not None:
            parts.append(digest)
        return ":".join(parts)

    def _with_fewer_stages(self) -> "TritonChoiceCaller":
        """The same configuration, holding fewer steps at once.

        What a kernel needs in memory is what it is working on at once, so the
        same tile with fewer steps live is the same computation with a smaller
        working set.  It is a different kernel and therefore a different
        configuration, and it is measured as one rather than folded into the
        first one's answer.
        """

        config = {**self.config, "num_stages": max(1, int(self.stages) - 1)}
        meta = {**self.inductor_meta}
        result = _compile_rendered(
            self.template,
            self.source,
            config={k: v for k, v in config.items() if k != "num_stages"},
            constants={
                k: v
                for k, v in config.items()
                if k not in ("num_stages", "num_warps", "layout",
                             "input_nodes", "out_size")
            },
            inductor_meta=meta,
        )
        if result is None:
            raise NotImplementedError(
                "this configuration does not fit even with a single step"
            )
        result.kernel._init_handles()
        launcher = result.make_launcher()
        launcher.__name__ = self.name
        return TritonChoiceCaller(
            name=self.name,
            input_nodes=self.input_nodes,
            layout=self.layout,
            description=self.description,
            source=self.source,
            src_hash=self._src_hash,
            launcher_args=self.launcher_args,
            num_stages=int(config["num_stages"]),
            num_warps=int(self.warps),
            config=config,
            operands=self.operands,
            inductor_meta=meta,
            template=self.template,
        ).bind(launcher)

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of this choice takes.

        A measurement is handed the operands of the call and nothing else: the
        extents and strides the kernel was written to be given, and the stream
        it is launched on, are settled when the launcher is written, and are
        carried alongside it.  What is measured is therefore a launch made the
        way a real one is made.
        """

        from ..runtime.stax_autotune import bench_launch

        from ..runtime.triton_compat import OutOfResources

        algo = self.to_callable()
        # The result is written into a buffer the caller named, and in the
        # signature that buffer is among the pointers -- so it goes with the
        # operands rather than at the end.  What follows the operands is what
        # the kernel was written to be given and the stream it runs on, in the
        # order it declares them.
        operands = [*args, *([out] if out is not None else []),
                    *(self.launcher_args or ())]
        def launch(these, _algo=algo):
            return _algo(*these)

        try:
            return bench_launch(launch, operands)
        except OutOfResources:
            # The kernel is too large for the shared memory this device has.
            # A tile that is too large is a configuration that cannot be
            # measured here, which is an answer about this machine rather than
            # a failure: the other candidates are still answers.  The only
            # thing worth trying is the same tile with fewer steps held at
            # once, since that is what decides how much memory is live.
            if int(self.stages) > 1:
                return self._with_fewer_stages().benchmark(*args, out=out)
            raise NotImplementedError(
                "this configuration does not fit the shared memory on this "
                "device, even with a single step"
            )

    def autoheuristic_id(self) -> str:
        return "triton_template"

class TritonTemplateKernel(TritonKernel):
    """One built kernel: its source, the symbol it defines, and how to launch.

    Kept apart from the template that describes it because the two have
    different lifetimes.  A template is written once and describes a family of
    launches; a kernel is built from one piece of source and is then the thing
    every launch of that source shares.  Two configurations whose source comes
    out identical therefore share one kernel, which is why what is remembered
    is the hash of the source and not the configuration that produced it.
    """

    #: One kernel per distinct source, however many templates asked for it.
    _memo: dict = {}

    def __init__(
        self,
        kernel_name,
        input_nodes: tuple[ir.IRNode, ...],
        output_node,
        defines,
        num_stages,
        num_warps,
        grid_fn,
        meta,
        call_sizes,
        num_consumer_groups=0,
        num_buffers_warp_spec=0,
        use_jit=False,
        tma_store=False,
        tma_load_for_template_epilogue=False,
        transpose_discontiguous_tensor_descriptors_override=None,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        subgraphs: list[ir.ComputedBuffer] | None = None,
        workspace_arg: WorkspaceArg | None = None,
        prologue_loads_all_inputs=False,
        hint_override: int | None = None,
        triton_meta: TritonMeta | None = None,
        always_freeze_layout: bool = False,
        index_dtype_override: str | None = None,
    ) -> None:
        tma_2d = tma_store or tma_load_for_template_epilogue
        if tma_store:
            pass
        numel = sympy_product(output_node.get_size())
        if tma_2d:
            if len(output_node.get_size()) != 2:
                raise AssertionError(
                    "TMA load/store only supported for 2D with templates"
                )
            tiling = {
                "x": output_node.get_size()[0],
                "y": output_node.get_size()[1],
                "r0_": sympy.S.One,
            }
        else:
            tiling = {
                "x": numel,
                "r0_": sympy.S.One,
            }
        super().__init__(
            tiling,
            features=SIMDKernelFeatures([], numel),
            hint_override=hint_override,
        )
        if tma_2d:
            # By default `construct_range_trees` will return the range_trees in the order
            # ["z", "y", "x", "r0_", "r1_"] (see simd.py:all_prefixes)
            # and this order defines what the kernel block shape will be. So if the template
            # input / output has requested e.g. ["x", "y"], `construct_range_trees` will still return the
            # trees in the order ["y", "x"]. This would mean that the template would need to transpose
            # the loaded value.
            # The below sorts the range trees according to that required by the caller
            prefix_to_range_tree = {rt.prefix: rt for rt in self.range_trees}
            pw_sorted_range_trees = []
            reduction_idx = None
            for i, prefix in enumerate(tiling):
                rt = prefix_to_range_tree[prefix]

                if rt.is_reduction:
                    reduction_idx = i
                    break
                rt.index = i
                rt.grid_dim = i
                rt.tensor_dim = i
                pw_sorted_range_trees.append(rt)
            self.range_trees = pw_sorted_range_trees + self.range_trees[reduction_idx:]

        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}  # type: ignore[var-annotated]
        self.defines = defines
        self.kernel_name = kernel_name
        self.use_jit = use_jit
        self.tma_store = tma_store
        self.tma_load_for_template_epilogue = tma_load_for_template_epilogue
        self.transpose_discontiguous_tensor_descriptors_override = (
            transpose_discontiguous_tensor_descriptors_override
        )
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.grid_fn = grid_fn
        self.meta = meta
        self.call_sizes = call_sizes
        # for templates with fixed epilogues
        self.prefix_args = prefix_args
        self.suffix_args = suffix_args
        # pyrefly: ignore [invalid-type-var]
        self.epilogue_fn = epilogue_fn
        self.render_hooks = {}  # type: ignore[var-annotated]
        self.triton_meta: TritonMeta | None = triton_meta
        self._index_dtype_override = index_dtype_override
        # For Templated Attention this can be a list of ir.Subgraph
        self.subgraphs: list[ir.ComputedBuffer] | None = subgraphs

        # Some templates use extra global memory as a workspace
        self.workspace_arg = workspace_arg
        if workspace_arg is not None:
            self.args.workspace_args.append(workspace_arg)

        # The following attributes (body, template_mask, output_val) are all
        # used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        self.subgraph_bodies: dict[str, SubgraphInfo] = {}

        # input buffers which we are allowed to prologue fuse into
        self.prologue_supported_inputs: OrderedSet[str] = OrderedSet()

        # input buffers which we are fusing into
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        # input buffers which we are fusing into, which preserve a zero mask
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()

        # The following attributes are all used for triton kernel codegen.
        # They are swapped onto the TritonTemplateKernel object by
        # `set_subgraph_body`
        # NB: the names here must match the fields in SubgraphInfo
        self.body: IndentedBuffer = FakeIndentedBuffer()
        self.compute: IndentedBuffer = FakeIndentedBuffer()
        self.indexing_code: IndentedBuffer = FakeIndentedBuffer()
        self.loads: IndentedBuffer = FakeIndentedBuffer()
        self.stores: IndentedBuffer = FakeIndentedBuffer()
        self.template_mask: str | None = None
        self.template_out_shape: str | tuple[str] | None = None
        self.ops_handler: V.WrapperHandler | None = None  # type: ignore[name-defined]
        self.root_var_renames: dict[str, str] = {}

        # When caching is enabled, the generated code is not dependent on the input nodes names, or
        # symbolic sizes names.
        # However, some of the variables returned by generate_and_load that are computed during the
        # triton template expansions (code generation) are dependent on those.
        # In order to cache the code generation and avoid redoing it for similar inputs that varies only by
        # input names or symbol names, we do a record and replay method.
        # During template expansions we record all function calls that change input_dependent_preserved_state
        # and replay them on a cache hit to regenerate them.
        self.cached_replay_events: RecordedEventsType | None = None

        # Update each time an input is marked frozen, used to replay the freezing of inputs on a cache hit.
        self.frozen_layouts_cnt = 0

        # When prologue_loads_all_inputs is true, prologue_supported_inputs is populated during def_kernel
        # by adding all inputs.
        self.prologue_loads_all_inputs = prologue_loads_all_inputs

        # When always_freeze_layout is True, get_stride_and_maybe_freeze_layout will
        # always freeze the layout immediately, bypassing layout constraints.
        # Set by templates that need the layout settled before anything is
        # loaded, which the attention templates do because a descriptor is
        # written against a layout rather than against a shape.
        self.always_freeze_layout = always_freeze_layout

        # Extra functions to be exposed during partial template rendering.
        self.extra_template_env_fns: list[Callable[..., Any]] = []

        # Tracking for intermediate variables
        self.tmp_var_ctr = itertools.count()

    @property

    @classmethod





    @property
    def index_dtype(self) -> str:
        """How wide this kernel indexes, which the caller may have said.

        A caller that knows the largest tensor this kernel will see can say so
        and get narrower indexing than the default would pick.
        """

        if self._index_dtype_override is not None:
            return self._index_dtype_override
        return super().index_dtype

    def _gen_tmp_var(self) -> str:
        """A name nothing else in this kernel is using."""

        return f"_tmp_var{next(self.tmp_var_ctr)}"

    def input_dependent_preserved_state(self) -> str:
        """The part of the kernel's state that a shape decides.

        Everything here is a function of the shapes, so it does not have to be
        written down again when a cached kernel is replayed against the same
        shapes.  The output buffers are left out on purpose: nothing reads them
        back, so carrying them would only make two runs of the same shapes look
        different.
        """

        return repr(
            [
                self.args.input_buffers,
                self.args.sizevars,
                self.args.workspace_args,
                self.prologue_supported_inputs,
                self.frozen_layouts_cnt,
            ]
        )

    def record_input_dependent_tracked_event(self) -> Callable[..., Any]:
        """Note calls that changed a shape-dependent part of the state.

        A kernel built once and run against several shapes is remembered by
        what its shapes decided.  Wrapping the calls that make those decisions
        is how a replay learns which of them to make again.
        """

        def decorator(fn) -> Callable[..., Any]:
            def wrapper(*args, **kwargs) -> Any:
                pre_state = self.input_dependent_preserved_state()
                result = fn(*args, **kwargs)
                post_state = self.input_dependent_preserved_state()
                if pre_state != post_state:
                    if self.cached_replay_events is None:
                        raise AssertionError("cached_replay_events must not be None")
                    self.cached_replay_events.append((fn.__name__, [*args], {**kwargs}))
                return result

            return wrapper

        return decorator

    def replay_cached_events(self, events) -> None:
        """Make again the decisions that the shapes decided last time."""

        for f, args, kwargs in events:
            getattr(self, f)(*args, **kwargs)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        """Work on one subgraph's text, and put the rest back afterwards.

        Everything a subgraph decides is decided against the text and the
        bookkeeping it owns, so both are put in place for the length of the
        work and taken back out after it.
        """

        if not all(
            hasattr(self, field.name) for field in dataclasses.fields(SubgraphInfo)
        ):
            raise AssertionError("missing expected SubgraphInfo fields on self")
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(SubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            raise AssertionError(body_name)

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in subgraph.only_copy_if_non_none_fields:
                continue
            setattr(self, key, value)

        context = (
            contextlib.nullcontext
            if not self.ops_handler
            else lambda: V.set_ops_handler(self.ops_handler(V.get_ops_handler()))
        )
        with context():
            yield
        self.subgraph_bodies[body_name] = SubgraphInfo(
            **{
                key.name: getattr(self, key.name)
                for key in dataclasses.fields(SubgraphInfo)
            }
        )
        for key, value in old_state.items():
            setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, clear_cse: bool = False):
        """A subgraph that has not been worked on yet gets its text here.

        The name has to be new, because working on a subgraph that already has
        text would put two lowerings in one place.
        """

        if body_name in self.subgraph_bodies:
            raise AssertionError(f"subgraph body {body_name} already exists")
        self.subgraph_bodies[body_name] = SubgraphInfo(
            IndentedBuffer(), None, None, cse=self.cse.clone() if clear_cse else None
        )
        with self.set_subgraph_body(body_name):
            yield

    def _make_independent_subgraph(self, subgraph_name, numel, **extra_fields):
        """A subgraph that ranges over its own numbers, not the kernel's.

        Used by epilogue and prologue hooks that have to do their own
        bookkeeping rather than share the kernel's.
        """

        groups = {"x": V.graph.sizevars.simplify(numel), "r0_": sympy.S.One}
        self.subgraph_bodies[subgraph_name] = SubgraphInfo(
            body=IndentedBuffer(),
            cse=self.cse.clone(),
            range_trees=self.construct_range_trees(
                pid_cache=None,
                inside_reduction=False,
                is_reduction=False,
                numels=groups,
                no_x_dim=False,
            ),
            range_tree_nodes={},
            numels=groups,
            **extra_fields,
        )

    def _setup_contiguous_index_state(
        self,
        indices: list[str],
        index_symbols: list[sympy.Symbol],
        lengths: list[sympy.Expr],
        mask: str | None,
        xindex_name: str = "xindex",
    ) -> sympy.Expr:
        """Name the output's dimensions and give the whole output one number.

        The result is one number per element of the result, and having one
        means the load and the store of a whole kernel can be ordered by it.
        Naming the dimensions is what lets the later text ask for a position
        by dimension rather than by arithmetic.

        Returns the expression for that one number.
        """

        for name, range_tree_entry in zip(
            indices, self.range_trees[0].construct_entries(lengths)
        ):
            range_tree_entry.set_name(name)
        contiguous_index = sympy_dot(
            FlexibleLayout.contiguous_strides(lengths), index_symbols
        )
        contiguous_index = self.rename_indexing(contiguous_index)
        self.body.writeline(f"{xindex_name} = " + texpr(contiguous_index))
        xindex_entry = self.range_trees[0].lookup(sympy.S.One, sympy_product(lengths))
        old_symbol = xindex_entry.symbol()
        xindex_entry.set_name(xindex_name)
        if self.range_tree_nodes.get(old_symbol) is xindex_entry:
            del self.range_tree_nodes[old_symbol]
        self.range_tree_nodes[xindex_entry.symbol()] = xindex_entry
        self.template_mask = mask
        self.template_indices = indices
        return contiguous_index

    def _make_codegen_hook(
        self, subgraph_name: str, indent_width: int = 0
    ) -> Callable[[], str]:
        """A hook that lowers a subgraph when the template's text asks for it.

        The text is laid out before the subgraphs in it are lowered, so where
        each one goes is written down as a placeholder and this fills it in.
        """

        def hook():
            with self.set_subgraph_body(subgraph_name):
                self.codegen_body()
                self.cse.invalidate(OrderedSet())
                result = self.body.getvalue()
                if indent_width:
                    result = textwrap.indent(
                        textwrap.dedent(result), " " * indent_width
                    )
                return result.strip()

        return hook

    def need_numel_args(self):
        return False

    def estimate_kernel_num_bytes(self):
        """An upper bound on the bytes this kernel touches.

        A value that is both read and written is counted twice, because it is
        both read and written.
        """

        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        num_bytes = []
        for i, inp in enumerate(itertools.chain(self.input_nodes, (self.output_node,))):
            size = V.graph.sizevars.optimization_hints(inp.get_size(), fallback=0)
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def estimate_flops(self) -> int:
        """How much arithmetic this kernel does, for weighing it against others."""

        for node in self.input_nodes:
            for fx_node in node._current_origins:
                f = count_flops_fx(fx_node)
                if f is not None:
                    if isinstance(f, tp.SymInt):
                        f = f.node.expr
                    return V.graph.sizevars.optimization_hint(f, fallback=0)
        return 0

    def jit_lines(self):
        """Render decorators and metadata for the generated Triton template."""

        if self.use_jit:
            return "@triton.jit"

    def _register_hook(
        self,
        hook_name: str,
        hook_fn,
        *,
        allow_overwriting: bool = False,
    ) -> str:
        """Remember what is to go where a placeholder stands.

        The name is the string the rendered text will carry and the function is
        what fills it in, so a name already in use means two different things
        want the same place -- which is only ever meant if the caller said so.
        """

        if not allow_overwriting:
            if hook_name in self.render_hooks:
                raise AssertionError(
                    f"Tried to register the hook {hook_name} multiple times. If "
                    "desired, pass allow_overwriting=True to _register_hook"
                )
        self.render_hooks[hook_name] = hook_fn
        return hook_name

    def _register_extra_template_env_fns(self, *fns: Callable[..., Any]):
        """Add more names to the template's rendering, beyond the usual ones.

        A function added this way may itself register a hook, so a template can
        carry a part of its text that nothing else knows how to produce.
        """

        self.extra_template_env_fns.extend(fns)

    def gen_argdefs(self):
        """The parameters the entry point takes, once the whole body is known.

        A template can decide to ask for another argument while it renders, so
        the list cannot be written until the whole body has been asked.
        """

        def hook():
            arg_defs, *_ = self.args.python_argdefs()
            return f"{', '.join(x.full_name() for x in arg_defs)}"

        return self._register_hook("<ARGDEFS>", hook, allow_overwriting=True)

    def gen_defines(self):
        return self.defines

    def def_kernel(self, *argnames):
        """The entry point: its parameters, and the names the body calls them by.

        The body names its operands, and those names are what it says.  Each
        one is bound to the argument that carries the operand's storage, and
        the binding is written out where the body can rely on it.

        A name given here is the name the body uses; it is not looked for
        among the kernel's arguments, because the kernel's arguments are in
        the order the graph gave them and the body is free to name them as it
        likes.
        """

        if not all(isinstance(x, str) for x in argnames):
            raise AssertionError("all argnames must be str")
        renames = IndentedBuffer(initial_indent=1)

        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        if len(argnames) != len(named_args):
            raise AssertionError(
                (
                    len(argnames),
                    len(named_args),
                    self.prefix_args,
                    len(self.input_nodes),
                )
            )

        for input_node in self.input_nodes[: self.prefix_args]:
            self.args.input(input_node.get_name())

        for name, input_node in zip(argnames, named_args):
            arg_name = f"arg_{name}"
            self.named_input_nodes[name] = input_node
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input_buffers[input_node.get_name()] = arg_name

        # The args may be duplicated, so renaming must be after args are
        # de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            if self.prologue_loads_all_inputs:
                self.prologue_supported_inputs.add(input_node.get_name())
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input(input_node.get_name())

        def hook():
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(self.gen_common_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(
                f"def {self.kernel_name}({', '.join(x.full_name() for x in arg_defs)}):"
            )
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
                self.codegen_prologue(code)
            return code.getvalue()

        return self._register_hook("<DEF_KERNEL>", hook)

    def size(self, name: str | None, index: int):
        """The extent of a dimension, as the body should write it.

        Asking with no name gives the result's own extent.  When the kernel
        indexes in 64 bits the extent is widened on the way out, because an
        extent computed in 32 bits can wrap around on a large tensor and turn
        into a wrong answer rather than an error.
        """

        if not isinstance(index, int):
            raise AssertionError(f"expected index to be int, got {type(index)}")
        if name is None:
            val = self.output_node.get_size()[index]
        else:
            if not isinstance(name, str):
                raise AssertionError(f"expected name to be str, got {type(name)}")
            val = self.named_input_nodes[name].get_size()[index]
        result = texpr(self.rename_indexing(val))
        if self.index_dtype == "tl.int64":
            return f"tl.full([], {result}, dtype=INDEX_DTYPE)"
        return result

    def stride(self, name, index=None):
        """The strides of a tensor, or one of them, as the body should write it.

        Asking with no name gives the result's own strides.  Asking for one
        dimension gives that dimension's stride, and asking for none of them
        gives the whole list, which is what a body needs when it is doing its
        own arithmetic over a tensor it did not load elementwise.
        """

        if name is None:
            val = self.output_node.get_stride()
        else:
            if not isinstance(name, str):
                raise AssertionError(f"expected name to be str, got {type(name)}")
            val = self.get_stride_and_maybe_freeze_layout(self.named_input_nodes[name])

        if isinstance(index, int):
            return texpr(self.rename_indexing(val[index]))
        return ", ".join([texpr(self.rename_indexing(i)) for i in val])

    def render(self, template, kwargs, record_input_dependent_tracked_event=False):
        """Lay out the template's text, and hand back what is still open.

        Every part of the text that is not known yet is written down as a
        placeholder by the hook that stands for it, so what comes back is text
        with placeholders in it plus the list of what fills them.  The caller
        fills them and asks again.

        Asking to record what the shapes decided wraps each hook, so a kernel
        remembered across shapes learns which decisions to make again.
        """

        if record_input_dependent_tracked_event:
            self.cached_replay_events = []

        template_env = {
            fn.__name__: (
                self.record_input_dependent_tracked_event()(fn)
                if record_input_dependent_tracked_event
                else fn
            )
            for fn in [
                self.def_kernel,
                self.size,
                self.stride,
                self.store_output,
                self.load_input,
                self.make_load,
                self.modification,
                self.gen_argdefs,
                self.gen_defines,
                *self.extra_template_env_fns,
            ]
        }
        return PartialRender(
            template.render(**template_env, **kwargs),
            self.render_hooks,
        )

    def _handle_scatter_graph(self, scatter_graph):
        """One scatter's contribution, as an assignment rather than a store.

        A scatter writes into a gradient the caller allocated for it, and that
        gradient is laid out contiguously whatever the scatter's own layout is,
        so the position of an element is worked out against the caller's
        strides rather than the scatter's.
        """

        if not isinstance(scatter_graph, ComputedBuffer):
            raise AssertionError(
                f"scatter_graph must be an instance of ComputeBuffer but got "
                f"{type(scatter_graph)}"
            )

        def contiguous_strides(x):
            # We always create a fresh contiguous grad for scattering into
            return sum(
                x_i * stride for x_i, stride in zip(x, scatter_graph.get_stride())
            )

        return scatter_graph.data.store_output(  # type: ignore[attr-defined]
            scatter_graph.name, contiguous_strides, []
        )

    def make_load(self, name, indices, mask):
        """A read a body writes out itself, for a body doing its own arithmetic.

        `load_input` leaves a placeholder and has the read worked out for it,
        which is right for most bodies.  A body that is computing its own
        index -- one that is gathering, or indexing a tensor it did not read
        elementwise -- needs the read as text it can put inside its own
        expression, so this hands it over.
        """

        if not isinstance(indices, (list, tuple)):
            raise AssertionError(
                f"expected indices to be list or tuple, got {type(indices)}"
            )
        if not isinstance(name, str):
            raise AssertionError(f"expected name to be str, got {type(name)}")
        if not isinstance(mask, str):
            raise AssertionError(f"expected mask to be str, got {type(mask)}")
        stride = self.get_stride_and_maybe_freeze_layout(self.named_input_nodes[name])
        indices = list(map(OpOverrides.paren, indices))
        if len(indices) != len(stride):
            raise AssertionError(
                f"expected len(indices) == len(stride), got {len(indices)} and {len(stride)}"
            )
        index = " + ".join(
            f"{texpr(self.rename_indexing(s))} * {i}" for s, i in zip(stride, indices)
        )
        return f"tl.load({name} + ({index}), {mask}, other=0.0)"

    def indexing(
        self,
        index: sympy.Expr,
        *,
        dense_indexing=False,
        copy_shape=None,
        override_mask=None,
        block_ptr=False,
        tma_compatibility_checker=None,
        mask_constant_index=False,
        allow_reduction_invariant_indexing=False,
    ):
        """Index as this kernel indexes, rather than as a kernel normally would.

        A body's own mask and the shape it wants broadcast to are the ones the
        surrounding read or write established, so those are what an index
        inside the body has to agree with.
        """

        return super().indexing(
            index,
            dense_indexing=False,
            # We pass template_out as the shape to broadcast the indexing to as
            # the mask might be broadcast to the output shape
            copy_shape=self.template_out_shape,
            override_mask=self.template_mask,
            block_ptr=block_ptr,
            tma_compatibility_checker=tma_compatibility_checker,
            mask_constant_index=mask_constant_index,
            allow_reduction_invariant_indexing=allow_reduction_invariant_indexing,
        )

    def codegen_range_tree(self):
        pass  # ignore default codegen

    def _compute_fusion_metadata(
        self, scheduling, epilogue_nodes, prologue_nodes, buf_name_to_prologue_group
    ):
        """Decide, before rendering, which fold goes with which write.

        The default is the blunt one: every folded-in operation goes with every
        write.  A subclass narrows that when it knows which output each fold
        belongs to.
        """

        self._epilogue_nodes_by_subgraph: defaultdict[int, list[Any]] = defaultdict(
            lambda: epilogue_nodes
        )
        self._unfused_epilogues: list[Any] = []
        self._prologue_sources: dict[str, frozenset[str]] = {}

    def codegen_template_body(
        self,
        scheduling,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        buf_name_to_prologue_group,
        prologue_preserves_zero_mask_fn,
        render,
    ) -> str:
        """The whole kernel, as source.

        Laying the template out and filling in what it left open are two
        passes, and the folds go in between: a read that had something folded
        into it is not finished until that fold is lowered into it, and a write
        is not finished until the folds that end in it are.
        """

        self._compute_fusion_metadata(
            scheduling, epilogue_nodes, prologue_nodes, buf_name_to_prologue_group
        )
        with self:
            partial_code = render()

            num_store_subgraphs = self.get_store_output_count()
            for i in range(num_store_subgraphs):
                subgraph_name = self._get_store_output_subgraph_name(i)
                with self.set_subgraph_body(subgraph_name):
                    for node in self._epilogue_nodes_by_subgraph[i]:
                        node.codegen(self.split_and_set_ranges(node.get_ranges()))
                    self.cse.invalidate(OrderedSet())

            self.codegen_prologues_in_subgraphs(
                buf_name_to_prologue_group, prologue_preserves_zero_mask_fn
            )

        partial_code = self._finalize_partial_render(partial_code)

        # Template hooks must be finalised after kernel.remove_kernel_local_buffers
        # is called (this is called when the kernel context is exited above), and when
        # the kernel handler is set (as below). This is because the hooks may add
        # DeferredLine type lines, which preclude lines involving buffers that have
        # been removed

        # finalize must be called after adding epilogue above
        with V.set_kernel_handler(self):
            if isinstance(partial_code, str):
                src_code = partial_code
            else:
                # This is used to calculate flops in TritonTemplateKernels
                with IRNode.current_origins(template_node.node.origins):
                    partial_code.finalize_hook("<DEF_KERNEL>")
                partial_code.finalize_hook("<ARGDEFS>", strict=False)

                for input_name in self.named_input_nodes:
                    subgraph_name = f"<LOAD_INPUT_{input_name}>"
                    partial_code.finalize_hook(subgraph_name, strict=False)

                num_store_subgraphs = self.get_store_output_count()
                for i in range(num_store_subgraphs):
                    subgraph_name = self._get_store_output_subgraph_name(i)
                    partial_code.finalize_hook(subgraph_name)

                # Ensure all hooks are finalized before the kernel is defined.
                # Note: some of these hooks may have been registered by a kernel
                # subclass
                src_code = partial_code.finalize_remaining()

        return src_code

    def codegen_prologues_in_subgraphs(
        self, buf_name_to_prologue_group, prologue_preserves_zero_mask_fn
    ):
        """Lower whatever was folded into each read, into that read.

        A read whose operand had something folded into it is not the read the
        body asked for -- it is that read with the fold applied -- so the fold
        is lowered here, into the same subgraph the read was made in.

        Whether the fold can be done without widening the operand to 32 bits
        decides the widening, and the answer differs per operand, so the
        setting is changed around each one.
        """

        for input_name, buffer in self.named_input_nodes.items():
            subgraph_name = f"<LOAD_INPUT_{input_name}>"
            prologue_group = buf_name_to_prologue_group.get(buffer.get_name(), [])
            if not prologue_group:
                continue
            can_codegen_without_upcast = all(
                p_n.can_codegen_without_upcasts() for p_n in prologue_group
            )
            with config.patch(
                "triton.codegen_upcast_to_fp32", not can_codegen_without_upcast
            ):
                with self.set_subgraph_body(subgraph_name):
                    for prologue_node in prologue_group:
                        if (
                            len(prologue_node.get_buffer_names()) == 1
                            and len(prologue_group) == 1
                        ):
                            if prologue_preserves_zero_mask_fn(prologue_node):
                                self.prologue_fused_inputs_preserve_zero |= (
                                    prologue_node.get_buffer_names()
                                )
                        prologue_node.codegen(
                            self.split_and_set_ranges(prologue_node.get_ranges())
                        )
                    self.cse.invalidate(OrderedSet())

    def _finalize_partial_render(self, partial_code):
        """A last look at the text before the placeholders are filled in.

        A subclass that wants to supply the whole source itself -- rather than
        have the placeholders filled in -- returns a replacement here.
        """

        return partial_code

    def modification(
        self,
        subgraph_number: int,
        output_name: str | None,
        mask: str | None = None,
        input_shapes: dict[str, tuple[str, ...]] | None = None,
        input_dtypes=None,
        **fixed_inputs,
    ) -> str:
        """Lower one subgraph, with the caller's values standing in for its inputs.

        A template asks for this where it wants part of itself rewritten: the
        subgraph named is lowered as usual, except that the values passed here
        are what it reads instead of the kernel's operands, and its result is
        assigned to `output_name` rather than stored.

        Passing no output name means the subgraph only contributes stores -- a
        scatter into something the caller owns -- and passing a name when it
        produced nothing, or nothing when it produced something, is a mistake
        in the template rather than something to guess at.
        """

        num = 0
        out = None
        scatters = []
        while f"mod_{subgraph_number}_{num}" in self.subgraph_bodies:
            num += 1
        with self.create_subgraph_body(f"mod_{subgraph_number}_{num}"):
            subgraph = self._get_subgraph(subgraph_number)
            modification_handler = ModificationWrapper(
                self,
                subgraph_number,
                fixed_inputs,
                mask,
                input_shapes,
                input_dtypes,
            )
            with V.set_ops_handler(modification_handler):
                if not isinstance(subgraph, (ComputedBuffer, list)):
                    raise AssertionError(
                        f"Expected the subgraph to be a ComputedBuffer or a "
                        f"List[ComputedBuffer], got {type(subgraph)}"
                    )
                # Handle scatter stores
                if isinstance(subgraph, list):
                    for scatter_graph in subgraph:
                        scatters.append(self._handle_scatter_graph(scatter_graph))
                elif isinstance(subgraph.data, InputBuffer):
                    out = subgraph.data.make_loader()(())
                else:
                    out = subgraph.data.inner_fn(())

            self.codegen_body()
            if output_name is not None:
                if not isinstance(output_name, str):
                    raise AssertionError(
                        f"expected output_name to be str, got {type(output_name)}"
                    )
                if out is None:
                    raise AssertionError("out must not be None")
                self.body.writeline(f"{output_name} = {out.value}")
            else:
                if out is not None:
                    raise AssertionError("out must be None when output_name is None")
                for scatter in scatters:
                    self.body.writeline(str(scatter))

            body_val = self.body.getvalue()
            self.cse.invalidate(OrderedSet())
            return body_val

    def _get_subgraph(self, subgraph_number: int):
        """The subgraph a hook was asked about, checked that it is there.

        The body is emptied of everything else, because what a modification
        adds is the whole of what its subgraph contributes and text left over
        from before would be added to it by accident.
        """

        if not isinstance(subgraph_number, int):
            raise AssertionError(
                f"expected subgraph_number to be int, got {type(subgraph_number)}"
            )
        if not isinstance(self.subgraphs, list):
            raise AssertionError(
                f"expected self.subgraphs to be list, got {type(self.subgraphs)}"
            )
        if subgraph_number >= len(self.subgraphs):
            raise AssertionError(
                f"Invalid subgraph number provided to create_modification, "
                f"{subgraph_number} must be < {len(self.subgraphs)}"
            )
        if self.body.getvalue() != "":
            raise AssertionError("Body should be clear before adding a modification")
        return self.subgraphs[subgraph_number]

    def load_input(
        self,
        input_name: str,
        output_name: str,
        indices: list[Any] | tuple[Any],
        mask: str | None = None,
        other: float | int | None = 0.0,
        indent_width: int = 4,
        index_shape: tuple[str] | None = None,
    ) -> str:
        """Read one of the kernel's operands, and leave a placeholder for it.

        The body cannot say where to read from, because the body does not know
        the operand's layout -- the kernel does.  So the body names the operand
        and one thing per dimension, and the read itself is left as a
        placeholder and filled in once the layout is known.

        A `mask` says which of those are really there.  `other` is what the
        ones that are not read as, which is not the same as zero once
        something has been folded into this operand: a prologue that adds a
        bias would otherwise add it to the padding too, so the mask is applied
        again after the fold.

        A body whose indices wrap around rather than run off the end (the
        matrix-multiply templates do) passes no mask, and that is not the same
        as passing one that says everything is there -- so "no mask" is
        recorded as its own value rather than as nothing.
        """

        input_node = self.named_input_nodes[input_name]
        if not self.prologue_loads_all_inputs:
            self.prologue_supported_inputs.add(input_node.get_name())

        tilings = (sympy_product(input_node.get_size()), sympy.Integer(1))
        groups = {
            "x": tilings[0],
            "r0_": tilings[1],
        }

        range_trees = self.construct_range_trees(
            pid_cache=None,
            inside_reduction=False,
            is_reduction=False,
            numels=groups,
            no_x_dim=False,
        )
        load_code = None

        with self.create_subgraph_body(f"<LOAD_INPUT_{input_name}>"):
            if not isinstance(indices, (list, tuple)):
                raise AssertionError(
                    f"expected indices to be list or tuple, got {type(indices)}"
                )
            if not isinstance(output_name, str):
                raise AssertionError(
                    f"expected output_name to be str, got {type(output_name)}"
                )
            if not isinstance(mask, (str, type(None))):
                raise AssertionError(
                    f"expected mask to be str or None, got {type(mask)}"
                )
            self.range_trees = range_trees
            self.numels = {k: V.graph.sizevars.simplify(v) for k, v in groups.items()}
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]

            lengths = [V.graph.sizevars.simplify(s) for s in input_node.get_size()]
            if len(indices) != len(lengths):
                raise AssertionError(
                    f"expected len(indices) == len(lengths), got {len(indices)} and {len(lengths)}"
                )

            # MM templates use out-of-bounds wrapping (e.g. `rm % M`) so no mask
            # is needed on the load.  Pass "None" when mask is unset to override
            # the mask that would otherwise be inherited.
            contiguous_index = self._setup_contiguous_index_state(
                indices,
                index_symbols,
                lengths,
                mask=mask if mask is not None else "None",
            )
            self.template_out_shape = index_shape if index_shape else "xindex"
            self.cse.invalidate(OrderedSet())

            template_mask = self.template_mask

            class StoreOutputSubstitution(V.WrapperHandler):
                """Stands in for a store while this load is being read.

                A prologue is lowered as a store into the operand it reads, and
                that is not a store to memory -- it is the value this load is
                supposed to produce.  So the store is caught here and turned
                into an assignment, reapplying the mask, because what was read
                as zero for the elements past the end is not zero once
                something has been folded in.
                """

                name = "StoreOutputSubstitution"

                def store(
                    self,
                    name: str,
                    index: sympy.Expr,
                    value: "CSEVariable",
                    mode=None,
                ):
                    V.kernel.store_buffer_names.add(name)
                    V.kernel.cse.store_cache[name] = value
                    if name in V.kernel.prologue_fused_inputs:
                        # We load masked out values with 0, then apply a prologue.
                        # The masked out values may not necessarily be 0 any more
                        # so we need to reapply the mask.
                        value_dtype = value.dtype
                        value_str = str(value)
                        if template_mask != "None" and (
                            name not in V.kernel.prologue_fused_inputs_preserve_zero
                            or other != 0
                        ):
                            value_str = (
                                f"tl.where({template_mask}, {value_str}, {other})"
                            )

                        if value_dtype != V.graph.get_buffer(name).dtype:
                            value_str = f"{value_str}.to({triton_type(V.graph.get_buffer(name).dtype)})"

                        V.kernel.compute.writeline(
                            f"{output_name} = {value_str}.broadcast_to(xindex.shape)"
                        )

            self.ops_handler = StoreOutputSubstitution

            input_node = self.named_input_nodes[input_name]
            if isinstance(input_node.layout, FlexibleLayout):
                # This will set a layout constraint on the template
                self.get_stride_and_maybe_freeze_layout(input_node)
                with patch.object(FlexibleLayout, "allow_indexing", True):
                    output_index = input_node.make_indexer()(index_symbols)
            else:
                output_index = input_node.make_indexer()(index_symbols)

            # in def_kernel above we define the inputs with the storage offset adjusted
            # creating the load in input_node.make_indexer() will also adjust by storage offset
            # so subtract here to not double increment
            if not V.graph.sizevars.statically_known_equals(
                input_node.layout.offset, 0
            ):
                output_index = output_index - self.rename_indexing(
                    input_node.get_layout().offset
                )

            output_index = self.rename_indexing(output_index)

            if output_index == contiguous_index:
                output_index_str = "xindex"
            else:
                out_indexing = self.indexing(
                    output_index,
                    copy_shape=self.template_out_shape,
                    override_mask=self.template_mask,
                )
                if not isinstance(out_indexing, IndexingOptions):
                    raise AssertionError(
                        f"expected out_indexing to be IndexingOptions, got {type(out_indexing)}"
                    )
                output_index_str = (
                    f"({out_indexing.index_str}).broadcast_to(xindex.shape)"
                )

            # Generate load code
            load_code = f"{output_name} = tl.load({input_name} + ({output_index_str})"

            if mask:
                load_code += f", mask={mask}, other={other})"
            else:
                load_code += ")"

        hook_key = f"<LOAD_INPUT_{input_name}>"

        def hook():
            with self.set_subgraph_body(hook_key):
                self.cse.invalidate(OrderedSet())
                self.codegen_body()
                self.cse.invalidate(OrderedSet())
                if input_node.get_name() not in self.prologue_fused_inputs:
                    if load_code is None:
                        raise AssertionError("load_code must not be None")
                    self.body.writeline(load_code)

                result = self.body.getvalue()
                if indent_width:
                    result = textwrap.indent(result, " " * indent_width)
                return result.strip()

        return self._register_hook(hook_key, hook)

    def additional_call_args_and_types(self):
        """What the launch has to work out before it can start.

        How many programs to start is usually known only once the shapes are,
        so the work is done by the launch rather than baked in.  When the count
        cannot be worked out from the sizes alone, the grid is handed in as
        arguments instead -- but a grid that reduces to a fixed number of
        values has to be settled at compile time, so a kernel that only some of
        its shapes can settle is asking for trouble.
        """

        if isinstance(self.grid_fn, SymbolicGridFn):
            grid_args = self.grid_fn.sympy_call(*self.call_sizes, self.meta)
            if len(grid_args) not in (0, 3):
                raise AssertionError("grid_fn should return 3 values")
            return (grid_args, map(type, grid_args))
        elif all(isinstance(x, (int, sympy.Integer)) for x in self.call_sizes):
            grid_args = self.grid_fn(*map(int, self.call_sizes), self.meta)
            if len(grid_args) not in (0, 3):
                raise AssertionError("grid_fn should return 3 values")
            return (grid_args, map(type, grid_args))
        return ((), ())

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True):
        """Write the line that starts this kernel."""

        wrapper = V.graph.wrapper_code
        _, call_args, _, arg_types = self.args.python_argdefs()

        additional_call_args, additional_arg_types = (
            self.additional_call_args_and_types()
        )

        if not additional_call_args:
            if V.graph.cpp_wrapper:
                raise AssertionError("cpp_wrapper requires SymbolicGridFn")
            wrapper.add_import_once(f"import {self.grid_fn.__module__}")
            meta = wrapper.add_meta_once(self.meta)
            fn_name = f"{self.grid_fn.__module__}.{self.grid_fn.__name__}"
            call_args.append(
                f"*{fn_name}({', '.join(map(pexpr, self.call_sizes))}, {meta})"
            )
            arg_types.append(None)

        call_args.extend(additional_call_args)
        arg_types.extend(additional_arg_types)

        if self.workspace_arg is not None:
            wrapper.generate_workspace_allocation(self.workspace_arg)

        # Use FixedGrid which properly handles grid values passed as arguments
        inductor_meta = FixedGrid.setup_grid_as_args() if additional_call_args else None
        wrapper.generate_kernel_call(
            name,
            call_args,
            arg_types=arg_types,
            triton_meta=self.triton_meta,
            inductor_meta=inductor_meta,
            triton=True,
        )
        self._emit_post_kernel_code(wrapper, name)
        if self.workspace_arg is not None:
            wrapper.generate_workspace_deallocation(self.workspace_arg)

    def _emit_post_kernel_code(self, wrapper, kernel_name: str) -> None:
        """Hook for subclasses to emit code after kernel call, before workspace dealloc."""

        pass

    def kernel_benchmark_extra_args(self) -> list[str]:
        """The grid, for the sake of measuring the kernel.

        How many programs to start does not affect what the kernel computes, so
        it is settled here rather than being carried through as something the
        measurement has to be told.
        """

        # Grid args are only used for benchmarking, not correctness
        return [
            str(x)
            for x in self.grid_fn(
                *V.graph.sizevars.optimization_hints(self.call_sizes), self.meta
            )
        ]

    def get_stride_and_maybe_freeze_layout(self, node) -> list[int]:
        """The strides a body should see for one of the kernel's operands.

        An operand whose layout is still open could be given any strides, so a
        body that hard-codes them is only correct if the layout is settled
        first.  Where there is a call into the framework to fall back on, the
        strides are computed as if it were settled and written down as a
        constraint instead -- so a second template that wants different strides
        is caught here rather than by a kernel that quietly reads the wrong
        elements.

        A view is left alone: its strides are already decided by the layout it
        is a view of, and a view does not have a name of its own to record a
        constraint against.
        """

        # realizing for safety
        ExternKernel.realize_input(node)
        layout = node.data.layout
        node_name = node.get_name()

        if isinstance(layout, FlexibleLayout) and not isinstance(
            node, ReinterpretView
        ):
            if not use_aten_gemm_kernels() or self.always_freeze_layout:
                # No framework fallback available, or the caller has said to
                # always freeze, so settle it now.
                node.data.freeze_layout()
            else:
                # Compute what the strides WOULD be if frozen, without freezing.
                fixed_layout_copy = layout.get_fixed_layout_without_freezing()
                existing = V.graph.buffer_layout_constraints.get(node_name)
                if existing is not None and existing != fixed_layout_copy:
                    raise AssertionError(
                        f"Layout constraint mismatch for {node_name}: "
                        f"existing {existing} vs new {fixed_layout_copy}"
                    )
                else:
                    V.graph.buffer_layout_constraints[node_name] = fixed_layout_copy

                return list(fixed_layout_copy.stride)
        # Already frozen or not an open layout, just return current strides
        return node.get_stride()

    def _generate_index_from_tma_index(
        self,
        output_name: str,
        offset_name: str,
        tma_index: sympy.Symbol,
        block_size: str,
        dim: int,
        num_dims: int,
        block_name: str | None = None,
    ) -> list[str]:
        """Turn a block descriptor's offset back into an index a load can use.

        A block descriptor is handed an offset and works out the addresses
        itself, which is efficient but leaves nothing to do arithmetic on.  So
        the offset is written out as well, and a body that needs to fuse into
        what the descriptor loaded gets the index the descriptor implies.

        The name given for the block is fixed for the whole kernel rather than
        per call, because it has to be a constant declared once at the top.
        """

        if block_name:
            if block_name in self.prologue_cache:
                if self.prologue_cache[block_name] != block_size:
                    raise AssertionError(
                        f"Constant {block_name} must be used for all stores"
                    )
            else:
                self.prologue_cache[block_name] = block_size
                self.prologue.writeline(f"{block_name}: tl.constexpr = {block_size}")
        else:
            block_name = block_size
        line0 = f"{offset_name} = {texpr(tma_index)}"
        expr = f"({offset_name} + tl.arange(0, {block_name}))"
        prefix_none = "".join(["None, "] * dim)
        suffix_none = ", ".join(["None"] * (num_dims - (dim + 1)))
        line1 = f"{output_name} = {expr}[{prefix_none}:, {suffix_none}]"
        return [line0, line1]

    def _generated_mask_for_tma(
        self,
        index_name: str,
        shape_val: str,
        output_name: str,
    ) -> str:
        """The line that says which of a block's elements are really there.

        A block is a fixed number of elements whichever way it is addressed,
        so the ones past the end of the tensor have to be told apart from the
        ones inside it.  A body that fuses on the loaded value needs that
        distinction as a value, not as the descriptor's own handling of it.
        """

        return f"{output_name} = {index_name} < {shape_val}"

    def store_output(
        self,
        indices: list[Any] | tuple[Any],
        val: str,
        mask: str | None = None,
        indent_width: int = 4,
        val_shape: tuple[str] | None = None,
        block_indexing: bool = False,
    ) -> str:
        """Write the result out, and fuse anything that was folded into it.

        The body's text cannot say where the result is written, because the
        body does not know the result's layout -- the kernel does, and decides
        it while the subgraph is being lowered.  So this leaves a placeholder
        where the write goes and fills it in after.

        `indices` names one thing per dimension of the result.  `val` is what
        is written, and writing it at those indices is what "the result"
        means.  A `mask` says which of those are really there, for a result
        whose extent is not a multiple of the block.

        `val_shape` is the shape of `val` when the body is working in blocks,
        which is not the result's own shape.  `block_indexing` says the
        indices are offsets into a block rather than the indices themselves.
        """

        subgraph_idx = next(self.store_output_ctr)
        subgraph_name = self._get_store_output_subgraph_name(subgraph_idx)
        with self.create_subgraph_body(subgraph_name, clear_cse=True):
            if not isinstance(indices, (list, tuple)):
                raise AssertionError(
                    f"expected indices to be list or tuple, got {type(indices)}"
                )
            if not isinstance(val, str):
                raise AssertionError(f"expected val to be str, got {type(val)}")
            if not isinstance(mask, (str, type(None))):
                raise AssertionError(
                    f"expected mask to be str or None, got {type(mask)}"
                )
            if not isinstance(val_shape, (tuple, type(None))):
                raise AssertionError(
                    f"expected val_shape to be tuple or None, got {type(val_shape)}"
                )
            if not isinstance(block_indexing, bool):
                raise AssertionError(
                    f"expected block_indexing to be bool, got {type(block_indexing)}"
                )
            if self.template_mask is not None:
                raise AssertionError("template_mask must be None")
            indices = list(map(OpOverrides.paren, indices))
            index_symbols = [sympy.Symbol(x, integer=True) for x in indices]
            lengths = [
                V.graph.sizevars.simplify(s) for s in self.output_node.get_size()
            ]
            if len(indices) != len(lengths):
                raise AssertionError(
                    f"expected len(indices) == len(lengths), got {len(indices)} and {len(lengths)}"
                )

            output_layout = self.output_node.get_layout()
            self.template_out = val
            if block_indexing:
                if not val_shape:
                    raise AssertionError(
                        "Blocking indexing requires passing in val_shape"
                    )
                if len(val_shape) != 2:
                    raise AssertionError(
                        "Blocking indexing only supports 2D data at this time"
                    )
                if mask:
                    raise AssertionError("Mask is not supported with blocking indexing")
                intermediate_lines: list[str] = []
                epilogue_index_symbols: list[sympy.Symbol] = []
                if self.tma_store or self.tma_load_for_template_epilogue:
                    val_shape_copy = list(val_shape)
                    for i, range_tree in enumerate(self.range_trees[:-1]):
                        name = range_tree.name
                        symbol = range_tree.symbol()
                        epilogue_index_symbols.append(symbol)
                        lookup_output = range_tree.lookup(sympy.S.One, lengths[i])
                        old_symbol = lookup_output.symbol()
                        lookup_output.set_name(name)
                        # Update var_list and var_range
                        range_tree.var_list[range_tree.var_list.index(old_symbol)] = (
                            symbol
                        )
                        range_val = range_tree.var_ranges[old_symbol]
                        del range_tree.var_ranges[old_symbol]
                        range_tree.var_ranges[symbol] = range_val
                        # Keep block-shape inference metadata in sync with the
                        # renamed epilogue range symbols used below.
                        if self.range_tree_nodes.get(old_symbol) is lookup_output:
                            del self.range_tree_nodes[old_symbol]
                        self.range_tree_nodes[symbol] = lookup_output
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                name,
                                "xoffset" if name == "xindex" else "yoffset",
                                index_symbols[i],
                                val_shape[i],
                                i,
                                len(val_shape),
                                block_name=range_tree.symt.name,
                            )
                        )
                        # Generate the xmask and ymask
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                name,
                                self.size(None, i),
                                "xmask" if name == "xindex" else "ymask",
                            )
                        )
                        # Update the val_shape information to use consistent naming
                        # after the remapping.
                        val_shape_copy[i] = range_tree.symt.name
                    val_shape = tuple(val_shape_copy)
                else:
                    mask_vars: list[str] = []
                    for i, (index, shape) in enumerate(zip(index_symbols, val_shape)):
                        index_name = self._gen_tmp_var()
                        offset_name = self._gen_tmp_var()
                        intermediate_lines.extend(
                            self._generate_index_from_tma_index(
                                index_name,
                                offset_name,
                                index,
                                shape,
                                i,
                                len(index_symbols),
                            )
                        )
                        epilogue_index_symbols.append(
                            sympy.Symbol(index_name, integer=True)
                        )
                        mask_name = self._gen_tmp_var()
                        intermediate_lines.append(
                            self._generated_mask_for_tma(
                                index_name,
                                self.size(None, i),
                                mask_name,
                            )
                        )
                        mask_vars.append(mask_name)
                    final_mask_var = self._gen_tmp_var()
                    final_mask_rhs = " & ".join(
                        f"{mask_name}" for mask_name in mask_vars
                    )
                    intermediate_lines.append(f"{final_mask_var} = {final_mask_rhs}")
                    self.template_mask = final_mask_var
                index_symbols = epilogue_index_symbols
                contiguous_index = sympy_dot(output_layout.stride, index_symbols)
                if not (self.tma_store or self.tma_load_for_template_epilogue):
                    # Convert to just use xindex.
                    contiguous_index = self.rename_indexing(contiguous_index)
                    intermediate_lines.append(f"xindex = {texpr(contiguous_index)}")
                    self.range_trees[0].lookup(
                        sympy.S.One, sympy_product(lengths)
                    ).set_name("xindex")
                index_symbols = epilogue_index_symbols
                output_index = contiguous_index
                # Write out the intermediate lines
                for line in intermediate_lines:
                    self.body.writeline(line)
            else:
                if self.tma_store:
                    raise AssertionError("TMA store requires block indexing")
                contiguous_index = self._setup_contiguous_index_state(
                    indices, index_symbols, lengths, mask
                )
                output_index = self.output_node.get_layout().make_indexer()(
                    index_symbols
                )
                output_index = self.rename_indexing(output_index)
                if output_index == contiguous_index:
                    output_index = sympy.Symbol("xindex", integer=True)

            self.template_out_shape = val_shape if val_shape else val
            acc_dtype = (
                triton_type_to_torch(self.meta["ACC_TYPE"])
                if "ACC_TYPE" in self.meta
                else tp.float32
            )
            output_dtype = self.output_node.get_dtype()

            epilogue_args = [
                V.kernel.cse.namedvar(val, dtype=acc_dtype, shape=val_shape)
            ]
            epilogue_nodes_by_subgraph = getattr(
                self, "_epilogue_nodes_by_subgraph", None
            )
            has_epilogue_fusion = bool(
                epilogue_nodes_by_subgraph[subgraph_idx]
                if epilogue_nodes_by_subgraph is not None
                else False
            )
            for input_node in itertools.chain(
                self.input_nodes[: self.prefix_args],
                self.input_nodes[len(self.input_nodes) - self.suffix_args :],
            ):
                input_node.freeze_layout()
                epilogue_arg = V.kernel.cse.generate(
                    self.compute,
                    input_node.make_loader()(index_symbols),
                    dtype=acc_dtype,
                    shape=input_node.get_size(),
                )
                epilogue_args.append(epilogue_arg)
                # We update frozen_layouts_cnt in order to replay this function
                # on a cache hit.
                self.frozen_layouts_cnt += 1

            # Apply the template's manual epilogue (e.g., bias add for addmm)
            epilogue_result = self.epilogue_fn(*epilogue_args)

            # When acc_dtype differs from output_dtype and there are fused
            # epilogue ops, emulate unfused numerics by truncating AFTER the
            # manual epilogue (so bias add happens in full precision)
            if acc_dtype != output_dtype and has_epilogue_fusion:
                epilogue_result = V.ops.to_dtype(
                    epilogue_result,
                    output_dtype,
                    src_dtype=acc_dtype,
                    use_compute_types=False,
                )
                epilogue_result = V.ops.to_dtype(
                    epilogue_result, acc_dtype, src_dtype=output_dtype
                )

            V.ops.store(
                self.output_node.get_name(),
                output_index,
                epilogue_result,
                mode="tma" if self.tma_store else None,
            )
            self.codegen_body()

        return self._register_hook(
            subgraph_name, self._make_codegen_hook(subgraph_name, indent_width)
        )


def _compile_rendered(template, source: str, config: dict, constants: dict,
                      inductor_meta: dict):
    """The rendered text, compiled the way a configuration is compiled.

    Two things are handed to the runtime: the signature, which is the name and
    type of every argument, and the constants, which are the values of the
    arguments that are not read because they were fixed before the kernel ran.
    The signature is worked out from the operands the kernel was written
    against -- a pointer's type is the operand's type, an extent's type is
    however wide it has to be -- and anything the kernel declares as a constant
    goes among the constants rather than among the types.

    A configuration that will not compile is a configuration that cannot be
    measured rather than a failure of the whole table: the other candidates are
    still answers, and this one is not one of them.  So a refusal is answered
    with nothing and the caller moves on.
    """

    from ..codegen.triton_conv import _build
    from ..runtime.hints import DeviceProperties
    from ..runtime.triton_compat import Config
    from ..runtime.triton_heuristics import CachingAutotuner

    try:
        fn = _build(
            source, f"<tensorplay-stax-{template.name}>", template.symbol
        )
        # What the kernel takes is what its own signature says.  A value
        # chosen when the text was written is an argument the compiler folds,
        # so it is in the signature as a constant and its value is among the
        # constants -- the two halves of saying the same thing.
        signature = _signature_of(fn.arg_names, template.operands)
        # A constant is a value, not a spelling of one.  Several of the values
        # a configuration carries are named in the language the kernel is
        # written in -- the index type, the type a sum is accumulated in -- and
        # the compiler wants the thing itself rather than the name of it.
        all_constants = {
            name: _language_type(value) if _is_language_name(value) else value
            for name, value in constants.items()
        }
        all_constants.setdefault("INDEX_DTYPE", _language_type(template.index_dtype))
        triton_meta = {
            "device": DeviceProperties.create(inductor_meta["device"]),
            "signature": signature,
            "constants": {
                name: value
                for name, value in all_constants.items()
                if name in signature
            },
        }
        tuner = CachingAutotuner(
            fn,
            triton_meta,
            [
                Config(
                    {
                        k: v
                        for k, v in config.items()
                        if k not in signature
                    },
                    num_warps=int(config.get("num_warps", 4)),
                    num_stages=int(config.get("num_stages", 2)),
                )
            ],
            save_cache_hook=None,
            mutated_arg_names=[],
            optimize_mem=False,
            heuristic_type=None,
            inductor_meta=inductor_meta,
        )
        return tuner._precompile_config(tuner.configs[0])
    except Exception:
        log.info(
            "could not compile %s at this configuration",
            template.name, exc_info=True,
        )
        return None


def _param_name(name: str) -> str:
    """The name the parameter carrying an operand goes by.

    A body refers to an operand by one name; the parameter that carries it is
    named after that, with a pointer's suffix when the body did not already
    use one.  One spelling, used by the signature, by the extents and strides
    it carries, and by anything asking about them, so that nothing has to
    translate between them.
    """

    return name if name.endswith("_ptr") else f"{name}_ptr"


def _launcher_tail(layout, input_nodes, arg_names) -> tuple:
    """What a launcher is handed after the operands, in the signature's order.

    The signature names the extents and the strides, and the order it names
    them in is the order they are given in -- a launcher is a call written
    before the values were known, so it cannot work that out for itself.  The
    names say which operand each belongs to, so each is looked up in that
    operand's own extents rather than counted off in sequence, and the stream
    the launch runs on is the last of them.
    """

    import tensorplay as tp

    def extents_of(node):
        """An operand's extents and strides, whichever kind of thing it is.

        A value the compiler has lowered knows its own extents and strides; one
        handed over from outside knows only its shape, and a value that knows
        neither is recorded as knowing nothing, so that asking about it fails
        here rather than at a launch.
        """

        if hasattr(node, "get_size"):
            return [int(v) for v in node.get_size()], [
                int(v) for v in node.get_stride()
            ]
        def read(name):
            """A value that may be spelled as an attribute or as a method.

            One describes itself by its shape and its strides; the other is
            asked.  Both spellings mean the same thing, so which one a value
            uses is not something the caller has to know.
            """

            value = getattr(node, name, None)
            if value is None:
                return None
            return value() if callable(value) else value

        shape = read("shape")
        if shape is None:
            shape = read("size")
        if shape is not None:
            shape = [int(v) for v in shape]
            known = read("stride")
            if known is not None:
                return shape, [int(v) for v in known]
            strides: list[int] = []
            running = 1
            for extent in reversed(shape):
                strides.append(running)
                running *= max(extent, 1)
            return shape, list(reversed(strides))
        raise AssertionError(
            f"an operand of this launch knows neither its shape nor its "
            f"extents: {node!r}"
        )

    by_name: dict = {}
    for index, node in enumerate(input_nodes):
        letter = chr(ord("A") + index)
        by_name[letter] = node
    by_name["C"] = layout
    sizes_of: dict = {}
    strides_of: dict = {}
    for letter, node in by_name.items():
        sizes_of[letter], strides_of[letter] = extents_of(node)

    tail: list = []
    size_seen: dict = {}
    stride_seen: dict = {}
    for name in arg_names:
        if name.startswith("size_"):
            # The name says which operand and which of its extents: the operand
            # is the letters up to the last digit, and the extent is which
            # digit it ends with.  The operand is named by the parameter, so
            # whichever spelling the operands were declared under is the one
            # looked for.
            # The name is the operand's own name with its pointer's suffix and
            # the index of the extent, so the operand is what is left once
            # those are taken off -- in whichever of the two spellings the
            # operands were declared.
            stem = name[len("size_"):-1]
            operand = next(
                (
                    letter
                    for letter in by_name
                    if stem.lower() in (letter.lower(), letter.lower() + "_ptr")
                ),
                None,
            )
            if operand is None:
                raise AssertionError(
                    f"the signature names an extent of {stem!r}, which is not "
                    f"one of this kernel's operands: {sorted(by_name)}"
                )
            rank = size_seen.get(operand, 0)
            size_seen[operand] = rank + 1
            tail.append(sizes_of[operand][rank])
        elif name.startswith("stride_"):
            stem = name[len("stride_"):-1]
            operand = next(
                (
                    letter
                    for letter in by_name
                    if stem.lower() in (letter.lower(), letter.lower() + "_ptr")
                ),
                None,
            )
            if operand is None:
                raise AssertionError(
                    f"the signature names a stride of {stem!r}, which is not "
                    f"one of this kernel's operands: {sorted(by_name)}"
                )
            rank = stride_seen.get(operand, 0)
            stride_seen[operand] = rank + 1
            tail.append(strides_of[operand][rank])

    stream = None
    device = getattr(layout, "device", None)
    if device is not None and getattr(device, "type", "cpu") != "cpu":
        try:
            stream = tp.cuda.current_stream(device).cuda_stream
        except Exception:
            stream = None
    tail.append(stream)
    return tuple(tail)


def _cdiv(a, b):
    """The number of whole steps of ``b`` that cover ``a``."""

    return -(-a // b)


def _call_sizes(layout, input_nodes, constexprs):
    """The extents a grid is a function of, in the order it declares them.

    A grid is a function of what is being computed, so the extents come from
    the result's layout and from the operands' -- and the tile comes from the
    configuration, which the caller passes alongside.
    """

    sizes: list = []
    for index in (0, 1):
        node = input_nodes[index] if len(input_nodes) > index else None
        sizes.append(
            int(node.get_size()[0])
            if node is not None and hasattr(node, "get_size")
            else int(layout.size[0])
        )
    return tuple(sizes)


def _is_language_name(value) -> bool:
    """Whether this value is written as a name in the kernel-writing language.

    A type the body is told about -- the width an index is narrowed to, the
    type a sum is accumulated in -- travels through the configuration as the
    name the body knows it by, because that is the only spelling a template
    can carry.  The compiler wants the type itself, so such a name is
    recognised here rather than by every value being looked up.
    """

    return isinstance(value, str) and value.startswith("tl.")


def _language_type(name: str):
    """A type as the kernel-writing language spells it, not as text.

    A constant handed to the compiler is a value, not a spelling of one: the
    text "tl.int64" is a name to look up, and the type it names is what a
    kernel narrows an index to.
    """

    from ..runtime.triton_compat import tl

    return getattr(tl, str(name).split(".")[-1], None)


def _signature_of(arg_names, operands: dict) -> dict:
    """What the runtime is told about each argument of a rendered kernel.

    An argument is one of three things, and the three are said differently: a
    pointer says the type of what it points at, an extent or a stride says how
    wide it has to be to hold any value the computation can produce, and a
    constant says nothing at all -- its value is among the constants, and its
    type is that it is not read.
    """

    from ..utils import _type_of

    signature: dict = {}
    for name in arg_names:
        if name.endswith("_ptr"):
            # The parameter is named after the operand, which may or may not
            # already carry the suffix itself; the operand is whichever of the
            # two spellings was declared.
            stem = name[:-4]
            operand = stem if stem in operands else name
            dtype = (operands.get(operand) or {}).get("dtype")
            # An operand whose type is not known is said as a byte pointer
            # rather than guessed at: the type decides how the kernel reads
            # what it points at, and a wrong one is a wrong answer rather than
            # a refused compile.
            resolved = getattr(tp, dtype, None) if dtype else None
            signature[name] = _type_of(resolved) if resolved is not None else "*i8"
        elif name.startswith(("size_", "stride_")):
            signature[name] = "i64"
        else:
            signature[name] = "constexpr"
    return signature


class TritonTemplate(KernelTemplate):
    """A template whose body is source text and whose launch is a grid.

    Everything a choice needs is settled by the time the template is built: what
    it is called, what it is made of, and how it is launched.  Holding the text
    rather than a reference to a builder is what lets two templates that wrote
    the same text be recognised as the same kernel, and it is why the text's
    hash -- not the template's -- is what identifies the kernel downstream.

    A template built twice under one name with one source is one template
    arriving twice, so the second is the same object rather than a rival: a
    module reached along two paths in one process should not look like two
    different kernels competing to be measured.
    """

    #: The kind of kernel this template builds; a subclass may name another.
    kernel_type: type[Any] = TritonTemplateKernel
    #: Whether a build is checked against the cache instead of trusted to it.
    #: Off by default because checking costs a second build, and a second
    #: build is exactly what a caller in a hurry is trying to avoid.
    test_cache: bool = False
    index_counter = itertools.count()
    all_templates: dict = {}

    @staticmethod
    def jinja2_env():
        """The environment kernel sources are rendered in.

        Strict, because a kernel is not a document: a name the template did
        not define is a bug in the template or in what was passed to it, and
        rendering around it would produce a kernel that runs and is wrong.
        """

        import jinja2

        return jinja2.Environment(undefined=jinja2.StrictUndefined)

    @classmethod
    def from_file(cls, name: str, grid=None, path=None, symbol: str = "",
                  file: str | None = None, **kwargs: Any):
        """A template whose source lives in a file, rendered when it is used.

        A kernel is long enough that keeping it inside a string makes the two
        things that matter -- the code and the parts that vary -- hard to tell
        apart.  A file lets the varying parts be marked as varying, and lets an
        editor see the kernel as code.

        The file's name and the template's name are two separate facts: a file
        is named for the dialect it is written in, while a template answers to
        what the operation is called.  So the file is named by ``file`` when the
        two differ, and otherwise by the template's own name.
        """

        import jinja2

        try:
            import jinja2  # noqa: F811
        except ImportError as e:  # pragma: no cover - depends on environment
            raise AssertionError("requires jinja2") from e
        if path is None:
            from ..codegen import templates as _tpl_dir

            path = _tpl_dir.__path__[0] + f"/{file or name}.py.jinja"
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        return cls(name, grid=grid, source=source, symbol=symbol, **kwargs)

    def render(self, **params: Any) -> str:
        """This kernel's source with the given decisions filled in."""

        return self.jinja2_env().from_string(self.source).render(**params)

    def render_with(self, kernel_args, **params: Any) -> str:
        """Render with the operands in scope by name rather than by variable.

        A body asks for what it needs -- ``size("A", -1)``, ``def_kernel("A",
        "B")`` -- without being handed an object to ask.  The names resolve
        against the operands the kernel is written for, so the same text serves
        every shape those operands may take, and a body that refers to an
        operand nobody declared fails here rather than at launch.

        The index type travels with the operands rather than with the body: it
        is a property of the launch, and a body that narrows an index -- to hand
        a descriptor a 32-bit offset, say -- has to name the type it narrows
        to.  So it is in scope under the name the bodies write it by, and a
        body that uses it without an operand record to say which type fails
        here rather than in the kernel.
        """

        env = self.jinja2_env()
        # The body names its operands but not itself: the entry point's name is
        # a property of the template, so it is supplied here rather than asked
        # for, and a body cannot introduce a second name for the same function.
        kernel_args.symbol = self.symbol
        env.globals["def_kernel"] = kernel_args.def_kernel
        env.globals["size"] = kernel_args.size
        env.globals["stride"] = kernel_args.stride
        env.globals["dtype"] = getattr(kernel_args, "dtype", None)
        return env.from_string(self.source).render(
            index_dtype=kernel_args.index_dtype, **params
        )

    def __init__(
        self,
        name: str,
        grid=None,
        source: str = "",
        symbol: str = "",
        debug: bool = False,
        cache_codegen_enabled_for_template: bool = False,
        prologue_loads_all_inputs: bool = False,
        always_freeze_layout: bool = False,
    ) -> None:
        super().__init__(name, hash=hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.grid = grid
        self.source = source
        #: The name the source defines its entry point under.  Named rather
        #: than assumed, because the symbol is part of the text: a source that
        #: defines something else needs no rewriting to be used here.
        self.symbol = symbol or name
        self.debug = debug
        self._cache_codegen_enabled_for_template = cache_codegen_enabled_for_template
        self._generated_code_cache: GeneratedCodeCache = clear_on_fresh_cache(
            GeneratedCodeCache()
        )
        self.prologue_loads_all_inputs = prologue_loads_all_inputs
        self.always_freeze_layout = always_freeze_layout
        #: The operands the last rendering was written against, by name, with
        #: the geometry and the type of each.  Filled in when a rendering
        #: happens and read when that rendering is compiled.
        self.operands: dict = {}
        #: The width an index is narrowed to, as the language spells it.  It is
        #: a property of the launch rather than of the text, and the compiled
        #: form has to be given it as well as the text.
        self.index_dtype: str = "tl.int64"
        existing = self.all_templates.get(name)
        if existing is not None and existing.src_hash != self.src_hash:
            raise AssertionError(
                f"two different kernels are both called {name}"
            )
        self.all_templates[name] = self

    @property
    def uid(self) -> str:
        """Namespaced by kind, so two kinds may share a name."""

        return f"triton::{self.name}"

    @property
    def template(self) -> "TritonTemplateKernel":
        """This template's source in the form that can be run.

        What a caller wants from a template that owns source text is the thing
        the text becomes, and that is the same for every configuration of this
        template -- so it is made once here and every configuration shares it.
        That sharing is the point: two configurations that wrote the same text
        are the same kernel, and a cache that could tell them apart would be
        keeping two of something there is only one of.
        """

        return self.kernel_type.get(self.name, self.source, self.symbol, self.grid)

    def argnames(self) -> list[str]:
        """The operand names the body asked for, in the order it asked for them.

        A body names its operands in the kernel it writes, and those names are
        what the rest of the template has to call them by: a body that asks for
        a name no operand was given fails here rather than producing a kernel
        that reads the wrong buffer.  Read out of the source rather than
        declared beside it, because a name written in the body and a name
        recorded elsewhere can disagree and then the disagreement is invisible.
        """

        import re

        found = re.search(r"def_kernel\(([^)]*)\)", self.source)
        if found is None:
            return ["A"]
        names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", found.group(1))
        return names or ["A"]

    def generate(self, **kwargs: Any):
        """The choice this template makes for one set of arguments.

        Refuses rather than raising when the arguments do not describe a launch
        this template can perform, so that a whole table can be offered at once
        and the entries that do not apply dropped without unwinding the rest.
        """

        if self.source == "":
            return None
        grid_fn = self.grid
        layout = kwargs.get("layout")
        input_nodes = kwargs.get("input_nodes") or ()
        if layout is not None:
            meta = {k: v for k, v in kwargs.items()
                    if k not in ("layout", "input_nodes")}
            meta.setdefault("out_size", tuple(int(s) for s in layout.size))
            names = self.argnames()[:len(input_nodes)] or list(self.argnames())[:1]
            operands = {}
            for name, node in zip(names, input_nodes):
                # A lowered value knows its own extents and strides; something
                # handed over from outside knows only its shape, and a value
                # that knows neither is recorded as knowing nothing so that a
                # body asking about it fails at render rather than at launch.
                shape = node.get_size() if hasattr(node, "get_size") else getattr(node, "shape", ())
                if hasattr(node, "get_stride"):
                    stride = node.get_stride()
                else:
                    stride = contiguous_strides(tuple(int(v) for v in shape))
                dt = node.get_dtype() if hasattr(node, "get_dtype") else getattr(node, "dtype", "float32")
                operands[name] = {
                    "shape": tuple(int(v) for v in shape),
                    "stride": tuple(int(v) for v in stride),
                    "dtype": str(dtype_name(dt)),
                }
            outputs = {
                "C": {
                    "shape": tuple(int(s) for s in layout.size),
                    "stride": tuple(int(s) for s in layout.stride),
                    # The result's own type, and the type of what it is read
                    # from: a product reads two operands and writes their type.
                    "dtype": str(
                        dtype_name(
                            layout.dtype
                            if layout.dtype is not None
                            else next(
                                (
                                    node.get_dtype()
                                    for node in input_nodes
                                    if hasattr(node, "get_dtype")
                                ),
                                None,
                            )
                        )
                    ),
                }
            }
            meta.update(kwargs)
            # What the kernel is written against and what the launch is written
            # against are two different lists.  A body names a tile, an
            # accumulation type, a bound: values chosen when the text was
            # written, which are parameters of the kernel because they cannot
            # change while it runs.  Everything else -- how the result is laid
            # out, what it is called, how many programs to start -- describes
            # the launch and is handed to the launch instead, because putting
            # it in the signature would make the kernel take an argument it
            # never reads, and a launch option the runtime also wants would be
            # asked for twice under the same name.
            # A configuration is what the kernel was written for; what the call
            # is about -- where its result goes, which values it was handed --
            # is what the launch is for.  The first is a constant of the module
            # the kernel is defined in; the second is the launch's own business
            # and would be a name the body could read but has no meaning for.
            constexprs = {
                k: v for k, v in kwargs.items()
                if k not in ("layout", "input_nodes", "out_size")
            }
            # Kept on the template for the compile that follows: what each
            # operand is decides the type of the argument the kernel takes for
            # it, and the operands are settled here rather than asked again.
            self.operands = {**operands, **outputs}
            rendered = self.render_with(
                KernelArgs(operands, outputs, constexprs), **meta
            )
        else:
            meta = {k: v for k, v in kwargs.items()
                    if k not in ("layout", "input_nodes")}
            rendered = self.render(**meta)
        # The rendered text has to be compiled before it can be launched, and
        # the compiled form cannot be called directly: what the runtime hands
        # back is only callable from inside a launch it has set up itself, with
        # the grid worked out, the arguments put in the order its signature
        # declares, and the stream passed along.  So the launch is written out
        # from the compiled form together with what it was compiled for, and
        # that is what the choice is bound to.
        # How many programs to start is a function of the extents being
        # computed and of the tile they are split into, so it is worked out
        # here, once, rather than asked of the kernel at every launch.  The
        # kernel takes operands; how many programs to start is not something an
        # operand can say.
        # The configuration as the grid function takes it: the values chosen
        # for this call, and nothing about the call itself -- a grid is a
        # function of what is being computed and of the tile it is split into,
        # and where the result lands is neither.
        extents = self.grid(
            *_call_sizes(layout, input_nodes, constexprs),
            {
                k: v
                for k, v in constexprs.items()
                if k not in ("layout", "input_nodes", "out_size")
            },
        )
        # What the launch is written against, kept so that a retry with one
        # thing changed can be compiled from the same record rather than
        # worked out again.
        inductor_meta = {
            "kernel_name": self.uid,
            "device": layout.device if layout is not None else "cuda",
            "grid_type": "FixedGrid",
            "fixed_grid": tuple(int(v) for v in extents),
        }
        result = _compile_rendered(
            self,
            rendered,
            config={
                k: v for k, v in kwargs.items()
                if k not in ("layout", "input_nodes", "out_size")
            },
            constants=dict(constexprs),
            inductor_meta={
                "kernel_name": self.uid,
                "device": layout.device if layout is not None else "cuda",
                "grid_type": "FixedGrid",
                "fixed_grid": tuple(int(v) for v in extents),
            },
        )
        if result is None:
            return None
        try:
            # Loading the binary is where a kernel that does not fit the device
            # says so -- a tile whose working set is larger than the shared
            # memory there is, or a register count the device cannot hold.  A
            # configuration that does not fit is one that cannot be measured,
            # not a failure of the table: the other candidates are still
            # answers, so it is dropped and the caller moves on.
            result.kernel._init_handles()
            launcher = result.make_launcher()
        except Exception:
            log.info(
                "could not load %s at this configuration", self.uid, exc_info=True
            )
            return None
        launcher.__name__ = self.uid
        # What a launcher written from this kernel is handed after the
        # operands: the extents and strides it declares, in that order, and the
        # stream it is launched on.  Written once here, where the signature is
        # known, rather than at each measurement.
        launcher_args = _launcher_tail(
                layout, input_nodes, result.kernel.src.fn.arg_names
            )
        caller = TritonChoiceCaller(
            name=self.uid,
            input_nodes=input_nodes,
            layout=layout,
            description=repr(sorted((k, repr(v)) for k, v in kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
            launcher_args=launcher_args,
            num_stages=int(kwargs.get("num_stages", 2)),
            num_warps=int(kwargs.get("num_warps", 4)),
            config={k: v for k, v in kwargs.items()},
            operands=dict(self.operands),
            inductor_meta=inductor_meta,
            template=self,
        )
        # A grid is a function of the extents being computed and of the tile
        # they were split into, so both are computed once here rather than
        # asked of the kernel at every launch: the kernel takes operands, and
        # how many programs to start is not something an operand can say.
        return caller.bind(launcher)

    def choice_or_none(self, **kwargs: Any):
        """This template's choice for one configuration, or nothing if it does not fit.

        The question is asked the same way of every candidate in a list, whether
        it is a template the compiler could write or an operation the framework
        already has, so that a list is offered as a whole and the entries that
        do not apply to this call are dropped without unwinding the rest.
        """

        try:
            return self.generate(**kwargs)
        except NotImplementedError:
            return None

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        """Add this template's choice to a list, or report why there is none."""

        try:
            choice = self.generate(**kwargs)
            if choice is not None:
                choices.append(choice)
            return None
        except NotImplementedError as e:
            return e

    def generate_and_load(self, generate_with_caching: bool = True, **kwargs: Any):
        """Write this configuration's kernel and load it.

        Generating and loading are one step here rather than two, because the
        load is what the generate is for.  Whether the written form is kept at
        all is the caller's decision: a caller enumerating a table wants every
        entry written, because it is about to measure them, while a caller
        measuring a second time would rather have a fresh one.
        """

        built = self.generate(**kwargs)
        if self.test_cache and built is not None:
            # A cache that hands back something different from what it was
            # given is worse than no cache, because the difference is invisible
            # until a number is wrong.  So with the switch on, the same call is
            # made twice and the two answers compared.
            again = self.generate(**kwargs)
            if getattr(again, "src_hash", None) != getattr(built, "src_hash", None):
                raise AssertionError(
                    f"{self.name} built two different ways from one source"
                )
        return built


class TritonTemplateCaller(TritonTemplateCallerBase):
    """A choice whose kernel is written from a template at launch time.

    What it holds is not a callable but the two things a callable would have
    been made of: the request that says how to build and time the kernel, and
    the text that builds it.  Keeping them apart is what lets the same kernel
    be timed more than once under different conditions, and then launched
    without being timed again.

    Its identity in a cache is the name of the kernel without its
    configuration, joined to the key of the request -- so that two choices
    differing only in configuration are recognised as the same kernel, and a
    change to the request that builds it invalidates both.
    """

    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,
        description,
        bmreq,
        log_info: dict | None = None,
        mutated_inputs=None,
        workspace_arg=None,
        allowed_prologue_inps: OrderedSet | None = None,
        hint_override: int | None = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.make_kernel_render = make_kernel_render
        self.bmreq: TritonBenchmarkRequest = bmreq
        if log_info is None:
            log_info = {}
        self.log_info: dict = log_info
        self.log_info.update(
            {
                "backend": "Triton",
                "num_stages": self.bmreq.num_stages,
                "num_warps": self.bmreq.num_warps,
            }
        )
        self.mutated_inputs = mutated_inputs
        self.workspace_arg = workspace_arg
        self.allowed_prologue_inps = (
            allowed_prologue_inps if allowed_prologue_inps is not None else OrderedSet()
        )
        self.hint_override = hint_override

        self.n_regs = None

    def benchmark(self, *args, out):
        """Time this kernel on the arguments it will actually be given.

        Timed through a profiler rather than by the clock when that is
        configured, because a clock on the host cannot tell a kernel's own time
        from the time the launch around it spent; but not when the launch is
        being captured whole, where a profiler would measure the capture.
        """
        if self.bmreq is None:
            raise AssertionError("self.bmreq must not be None")
        if (
            config.profile_bandwidth_with_do_bench_using_profiling
            and not self._benchmark_with_cudagraphs
        ):
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        self.bmreq.benchmark_with_cudagraphs = self._benchmark_with_cudagraphs
        return self.bmreq.benchmark(*args, out=out)

    def precompile(self):
        """Build the kernel without running it, and keep what the build said.

        The register count is what the build reports and a run cannot: it is a
        property of the machine code, which exists before the machine code is
        asked to do anything.
        """
        if self.bmreq is None:
            raise AssertionError("self.bmreq must not be None")
        self.bmreq.precompile()

        self.n_regs = self.bmreq.n_regs

    def __str__(self) -> str:
        return f"TritonTemplateCaller({self.bmreq.module_path}, {self.description})"

    def call_name(self):
        return f"template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        """The buffer this choice writes, rather than a value it returns.

        A kernel written from a template produces its result by writing into
        something, so what a choice offers the rest of the graph is that
        something and the text that fills it -- not a value.
        """

        from .. import ir

        buffer = ir.TritonTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            mutated_inputs=self.mutated_inputs,
            allowed_prologue_inps=self.allowed_prologue_inps,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)

    def info_dict(self) -> dict:
        """What the autotune log records about this candidate."""
        return self.log_info

    def get_make_kernel_render(self):
        return self.make_kernel_render

    def autoheuristic_id(self):
        """A name that says the shape of the work, not the name it was given.

        Two candidates that tile the same way and differ only in how many
        steps are held at once are different tunings of the same work, and
        naming them by the tiling is what lets a stored decision be found
        again by a kernel that was written under a different name.
        """
        type_name = "triton"
        info = self.info_dict()
        tile = info["tile_shape"]
        tile_vals = eval(tile)
        BLOCK_M = tile_vals[0]
        BLOCK_K = tile_vals[1]
        BLOCK_N = tile_vals[2]
        num_stages = info["num_stages"]
        num_warps = info["num_warps"]
        return f"type={type_name}_BLOCK-M={BLOCK_M}_BLOCK-K={BLOCK_K}_BLOCK-N={BLOCK_N}_numstages={num_stages}_numwarps={num_warps}"


class KernelArgs:
    """The operands a kernel is written against, held by name.

    A kernel's body refers to its operands by name and asks for their extents
    and strides rather than being handed them: a body that reads ``size("A",
    -2)`` says what it wants, and the answer comes from whoever holds the
    operand.  That separation is what lets one body serve several shapes -- a
    batched product and a plain one differ in which extents they ask for, not in
    the text between the asks.

    The signature is emitted here too, from the same names, so a body cannot
    disagree with its own parameters: the arguments it may refer to and the
    arguments it is given are the same list, read twice.
    """

    def __init__(self, inputs: dict, outputs: dict, constexprs: dict | None = None,
                 index_dtype: str = "tl.int64"):
        self.inputs = {k: dict(v) for k, v in inputs.items()}
        self.outputs = {k: dict(v) for k, v in outputs.items()}
        self.constexprs = dict(constexprs or {})
        self.index_dtype = index_dtype
        #: The constants the body reads by name, written into the text the
        #: first time a signature is asked for.
        self.defines = ""
        #: The entry point's name, supplied by whatever renders this.
        self.symbol = ""

    def call_sizes(self) -> tuple:
        """The extents a launch is sized by, and the configuration it was given.

        A grid is a function of what is being computed and of the tile the
        computation was split into, so both are handed together: a grid asked
        for the extents alone would have to guess the tile, and one asked for
        the tile alone would have to guess the extents.  The extents come first
        because that is the order a grid function declares them in.
        """

        meta = {k: v for k, v in self.constexprs.items()}
        for name, spec in self.outputs.items():
            meta.setdefault(name, spec)
        return tuple(spec["shape"] for name, spec in self.outputs.items()), meta

    def _operand(self, name: str) -> dict:
        # An operand is declared under one name and a body may ask about it
        # under either that name or the name its value is read by; both mean
        # the same operand, so both are looked for before either is refused.
        for candidate in (name, name.removesuffix("_ptr"), f"{name}_ptr"):
            if candidate in self.inputs:
                return self.inputs[candidate]
            if candidate in self.outputs:
                return self.outputs[candidate]
        raise KeyError(
            f"{name} is not one of this kernel's operands: "
            f"{sorted(self.inputs) + sorted(self.outputs)}"
        )

    def size(self, name, dim: int) -> str:
        """One extent of one operand, as the kernel's index type.

        Negative dimensions count from the end, which is what lets one body ask
        for ``-1`` and mean the contiguous axis whatever the rank is.  Naming
        nothing asks about the result rather than an input, which is what a body
        wants when it is working out the shape of what it is producing.
        """

        if name is None:
            name, spec = next(iter(self.outputs.items()))
            shape = spec["shape"]
        else:
            shape = self._operand(name)["shape"]
        index = dim if dim >= 0 else len(shape) + dim
        if not 0 <= index < len(shape):
            raise IndexError(
                f"{name} has {len(shape)} dimensions; {dim} is not one of them"
            )
        return f"size_{_param_name(name)}{index}"

    def stride(self, name, dim: int) -> str:
        """One stride of one operand, by name rather than by position.

        Strides are passed as arguments rather than baked in, so a body that
        asks for one does not have to be re-rendered per layout: the kernel is
        told the layout at launch, and the same text serves every layout.

        Naming nothing asks about the result rather than an input, which is what
        a body wants when it is working out how its own result is addressed.
        """

        if name is None:
            name = next(iter(self.outputs))
        return f"stride_{_param_name(name)}{dim}"

    def def_kernel(self, *names: str, symbol: str | None = None, outputs: tuple = ()) -> str:
        """The decorator and the signature, written from the operand names.

        Every operand named here becomes a parameter, in the order named, and
        every operand *not* named is not a parameter -- so a body that refers
        to something it did not ask for fails at render time rather than at
        launch, where the mistake would look like a kernel fault.
        """

        symbol = symbol or self.symbol or "_kernel"
        parts = list(names) + list(outputs)

        for name in parts:
            self._operand(name)
        # One spelling per name, used everywhere: the parameter that carries an
        # operand, the extents it has, and the strides it is laid out with are
        # all named after it, so a body that asks for one of them is asking
        # about the operand and nothing has to translate between spellings.
        strides = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            strides += [f"stride_{_param_name(name)}{d}" for d in range(rank)]
        sizes = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            sizes += [f"size_{_param_name(name)}{d}" for d in range(rank)]
        # A value chosen when the text was written -- a tile, an accumulation
        # type, a bound -- cannot change while the kernel runs, and the kernel
        # is compiled once for the configuration it came from.  So it is a
        # parameter the compiler may fold rather than one passed on every
        # launch, and it is annotated as such where the compiler reads that.
        # It has to be a parameter rather than something the body reads from
        # the module it is defined in: a global is not a value the body can
        # fold, and the arithmetic on a tile would be worked out at every step
        # instead of once.
        names = [*names, *outputs]
        # The index type travels with the operands rather than with the body: it
        # is a property of the launch, and a body that narrows an index has to
        # name the type it narrows to.  It is a parameter for the same reason
        # the rest are -- the compiler folds it, and the body reads it by name.
        tail = [f"{n}: tl.constexpr" for n in self.constexprs]
        tail.append("INDEX_DTYPE: tl.constexpr")
        # An operand is declared under one name and referred to by it.  A body
        # that says "A" means the parameter named A, so that is the name given
        # -- a short name is spelled the way the body spells it, and one the
        # body spells with its pointer suffix keeps it.  A body that goes on to
        # assign a name it was not given is therefore assigning a name of its
        # own, which is what it meant to do.
        all_params = [_param_name(n) for n in names] + sizes + strides + tail
        binds = "".join(
            f"    {n} = {_param_name(n)}\n"
            for n in names
            if not n.endswith("_ptr")
        )
        return (
            "@triton.jit\n"
            f"def {symbol}({', '.join(all_params)}):\n"
            f"{binds}"
        )



def rand_strided(
    size,
    stride,
    dtype=tp.float32,
    device="cpu",
    extra_size: int = 0,
):
    """A value of a given geometry, holding values rather than nothing.

    Filled rather than left empty because a candidate may read the place it is
    going to write -- accumulating into it, say -- and an empty place would make
    every candidate that does so compute from whatever was in memory, which is
    not the same measurement twice running.  Random rather than fixed for the
    same reason: a candidate that branches on what it reads would be measured on
    one branch, and there is no way to know which.

    The buffer behind it is as large as the geometry reaches and no larger, since
    a strided view of a smaller buffer would read past it.
    """

    needed_size = extra_size
    if all(s > 0 for s in size):
        # only need to allocate if all sizes are non-zero
        needed_size += (
            sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
        )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            # there is no normal distribution for the eight-bit float types, so
            # the values are made in a wider type and cast: which values they are
            # does not matter, only that they are values
            buffer = tp.randn(needed_size, dtype=tp.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = tp.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = tp.zeros(size=[needed_size], dtype=dtype, device=device)
    return tp.as_strided(buffer, size, stride)


class NoValidChoicesError(RuntimeError):
    """Nothing was offered to choose from.

    Its own kind of failure because the way out of it is not a fallback: a call
    with no candidates has no correct answer, and the only thing to do about it
    is to offer something.  So the message says what was empty rather than
    reporting a crash somewhere downstream of it.
    """


class BenchmarkTensors(NamedTuple):
    """The values one measurement is made against.

    Inputs and the place the result goes, held together because a measurement
    that used one call's inputs and another's output would be measuring neither.
    """

    inputs: list
    output: Any

    def unpack(self) -> tuple[list, Any]:
        return self.inputs, self.output


class AutotuneArgs(NamedTuple):
    """Everything a measurement needs that is not the choice being measured.

    Two sets of inputs travel here rather than one, because the operation's own
    kernel and a template are sometimes handed the same values in different
    shapes -- a bias of one value against a bias of a row, say.  Measuring both
    against the same values means the difference in their times is a difference
    in their kernels and not in what they were handed.

    The expected result is here too, because a candidate has to be checked as
    well as timed: the fastest wrong answer is not an answer.
    """

    example_inputs: list
    example_inputs_extern: list
    out: Any
    out_extern: Any
    expected: Any = None

    def verify(self, **kwargs):
        """Check the answer, not only how fast it arrived.

        A candidate that is fastest and wrong is not a candidate, and a timing
        cannot tell the two apart -- so what was produced is compared with what
        was expected, and the comparison is the one a reader would make.
        """

        assert_close(self.out_extern, self.expected, **kwargs)

    @staticmethod
    def from_choice_args(
        example_inputs, example_inputs_extern, out, out_extern, expected
    ) -> "AutotuneArgs":
        return AutotuneArgs(
            example_inputs, example_inputs_extern, out, out_extern, expected
        )

    def get_benchmark_tensors(self, is_extern: bool) -> BenchmarkTensors:
        if is_extern:
            return BenchmarkTensors(self.example_inputs_extern, self.out_extern)
        return BenchmarkTensors(self.example_inputs, self.out)


def create_inputs_key(input_nodes) -> str:
    """What the inputs are, said so that two calls with the same inputs agree."""

    return repr([AlgorithmSelectorCache.key_of(x) for x in input_nodes])


def create_precompile_key(name: str, inputs_key: str, choices: list) -> str:
    """What a set of candidates was built for, as one string.

    The operation, the inputs, the precision and every candidate's own key.
    All four are in it because a build is only reusable for the same question:
    the same candidates asked about different inputs are different kernels,
    and the same ones measured at another precision are not comparable.
    """

    return ":".join(
        [
            name,
            inputs_key,
            tp.get_float32_matmul_precision(),
        ]
        + [choice.kernel_hash_key() for choice in choices]
    )


def should_use_layout_constraints(x: Any) -> bool:
    """Whether a value's layout was pinned rather than left to be inferred.

    Asked per value because one call's inputs can be pinned and another's not,
    and a stored decision made under a pin must not be reused without it.
    """

    return getattr(x, "layout", None) is not None and getattr(
        getattr(x, "layout", None), "stride", None
    ) is not None


def get_strides_with_layout_constraints(x: Any):
    """A value's strides, from its layout when it has one and its shape when not.

    The two say different things: a layout's strides are where the value is, and
    a shape's are how it would be laid out.  A measurement made against one and
    stored against the other would be reused for a call it was not made for.
    """

    if should_use_layout_constraints(x):
        return x.layout.stride
    return x.get_stride()


def filter_choices_by_name_regex(choices: list) -> list:
    """The candidates whose names are wanted, when a name pattern was given.

    A pattern is a way of asking for one candidate to be measured without the
    others being taken off the list, so a pattern that matched nothing leaves the
    list as it was rather than leaving it empty.
    """

    import re

    regex = config.autotune_choice_name_regex
    if regex is None:
        return choices
    matcher = re.compile(regex)
    return [choice for choice in choices if matcher.search(choice.name)]


def filter_choices_by_desc_regex(choices: list) -> list:
    """The candidates whose descriptions are wanted, for the same reason."""

    import re

    regex = config.autotune_choice_desc_regex
    if regex is None:
        return choices
    matcher = re.compile(regex)
    return [choice for choice in choices if matcher.search(choice.description)]


def _autotune_metadata(input_nodes) -> dict[str, str]:
    """Say what the values being tuned for look like, in words.

    Two searches that timed the same kernel on differently-shaped values are
    not comparable, and neither are two records of the same search.  So the
    shape, the layout and the strides go into the record of the search itself,
    both as they are and as the size solver would eventually settle them --
    because which of the two a search actually ran under is itself worth
    knowing.
    """

    return {
        "autotune_strides": ", ".join([str(n.get_stride()) for n in input_nodes]),
        "autotune_dtypes": ", ".join([str(n.get_dtype()) for n in input_nodes]),
        "autotune_shape": ", ".join(
            ["x".join(map(str, n.get_size())) for n in input_nodes]
        ),
        "autotune_offset": ", ".join([str(n.get_layout().offset) for n in input_nodes]),
        "autotune_strides_hinted": ", ".join(
            [
                str(V.graph.sizevars.optimization_hints(n.get_stride()))
                for n in input_nodes
            ]
        ),
        "autotune_shape_hinted": ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.optimization_hints(n.get_size()),
                    )
                )
                for n in input_nodes
            ]
        ),
    }


def _log_autotune_choices_stats(
    event_name: str, timings: dict
) -> None:
    """Report how the candidates ranked, not only which one won.

    The winning time says what was chosen and nothing about whether the search
    earned it: a search that barely improved on the first candidate it tried
    and one that ruled out nine near-misses look identical from the winner
    alone.  So where the best template candidate sat in the ranking is recorded
    alongside it, together with the time it would have taken had it been
    chosen -- which is the number that says what the search was worth.
    """

    if not timings:
        return None

    metadata: dict = {
        "num_choices": len(timings),
        "num_triton_choices": len(
            [c for c in timings if isinstance(c, TritonTemplateCaller)]
        ),
    }

    sorted_choices = sorted(timings, key=timings.__getitem__)
    best_choice = sorted_choices[0]
    metadata["best_kernel"] = best_choice.name
    if best_choice.description:
        metadata["best_kernel_desc"] = best_choice.description
    metadata["best_time"] = timings[best_choice]

    best_triton_pos = next(
        (
            i
            for i, choice in enumerate(sorted_choices)
            if isinstance(choice, TritonTemplateCaller)
        ),
        None,
    )
    if best_triton_pos is not None:
        metadata["best_triton_pos"] = best_triton_pos
        best_triton_kernel = sorted_choices[best_triton_pos]
        if best_triton_pos != 0:
            metadata["best_triton_time"] = timings[best_triton_kernel]
            metadata["best_triton_kernel"] = best_triton_kernel.name
            if best_triton_kernel.description:
                metadata["best_triton_kernel_desc"] = best_triton_kernel.description

    payload = json.dumps(metadata, default=str)
    trace_structured(event_name, metadata)
    sys.stderr.write(f"Autotune Choices Stats:\n{payload}\n")


@functools.cache
def get_mm_log_filename() -> str | None:
    """Where matrix-multiplication tunings are recorded, if they are being."""
    mm_file_name = os.environ.get("TP_MM_LOGGING_FILE", None)
    if not mm_file_name:
        return None

    if "json" not in mm_file_name:
        mm_file_name = f"{mm_file_name}.json"

    return mm_file_name


@functools.cache
def get_omni_attention_log_filename() -> str | None:
    """Where omni-attention tunings are recorded, if they are being."""
    file_name = os.environ.get("TP_OMNI_ATTENTION_LOGGING_FILE", None)
    if not file_name:
        return None

    return str(Path(file_name).with_suffix(".json"))


@functools.cache
def get_conv_log_filename() -> str | None:
    """Where convolution tunings are recorded, if they are being."""
    conv_file_name = os.environ.get("TP_CONV_LOGGING_FILE", None)
    if not conv_file_name:
        return None

    return str(Path(conv_file_name).with_suffix(".json"))


def append_to_log(filename, data):
    """Add one record to a log of tunings, without losing another's.

    Several builds can be writing at once -- one per process on a machine, and
    a machine can be building for several -- so the read and the write are
    taken together under a lock, and a file that is not there yet or is not
    readable as a list starts as an empty list rather than failing the build
    that happened to run second.
    """

    lock_file = filename.replace(".json", ".lock")
    lock = FileLock(lock_file)
    with lock:
        try:
            with open(filename) as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append(data)

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)


def _classify_kernel_operation(
    name: str, choices: list, input_nodes
) -> str:
    """Say which kind of operation a set of candidates is for.

    Returns one of: "mm", "conv", "omni", or "other".

    A tuning is told to record itself under one of these names, and being
    wrong about which would file a convolution's timings under matrix
    multiplication -- so this is worked out from what the candidates are where
    that is possible, from the shapes where it is not, and from the name only
    as a last resort.  The name is matched in full rather than by substring,
    because a name that happens to contain "mm" is not a matrix
    multiplication.
    """
    # First, try to classify from choice types
    if choices:
        for choice in choices:
            if isinstance(choice, TritonTemplateCaller):
                # Extract template name (e.g., "mm" from "mm_1", "convolution2d" from "convolution2d_3")
                template_name = choice.name.rsplit("_", 1)[0]

                # Check known template patterns
                if template_name in (
                    "mm",
                    "bmm",
                    "mm_persistent_tma",
                    "grouped_mm",
                    "scaled_grouped_mm",
                    "mm_plus_mm",
                    "blackwell_ws_persistent_device_tma",
                    "scaled_mm_device_tma_main_loop_scaling",
                ):
                    return "mm"
                elif template_name in ("convolution2d", "convolution3d"):
                    return "conv"
                elif template_name.startswith("omni_"):
                    return "omni"

            elif isinstance(choice, ExternKernelCaller):
                # Check extern kernel names
                choice_name = choice.name
                if choice_name in (
                    "mm",
                    "bmm",
                    "addmm",
                    "baddbmm",
                    "_int_mm",
                    "_scaled_mm",
                ):
                    return "mm"
                elif "conv" in choice_name:
                    return "conv"

    # Second, use input shape heuristics for additional validation
    if len(input_nodes) >= 2:
        try:
            input_0_shape = input_nodes[0].get_size()
            input_1_shape = input_nodes[1].get_size()

            # Matrix multiplication patterns
            if len(input_0_shape) == 2 and len(input_1_shape) == 2:
                return "mm"
            elif len(input_0_shape) == 3 and len(input_1_shape) == 3:
                return "mm"  # bmm

            # Convolution patterns: input NCHW/NCDHW, weight OIHW/OIDHW
            elif len(input_0_shape) in (4, 5) and len(input_1_shape) in (4, 5):
                # Could be conv or omni attention, prefer template name if available
                if len(input_0_shape) == 4 and len(input_1_shape) == 4:
                    # Check if it looks like conv (channel dims match)
                    # Conv: input[N,C,H,W] @ weight[O,C,kH,kW] where input[1] == weight[1]
                    try:
                        if input_0_shape[1] == input_1_shape[1]:
                            return "conv"
                    except (IndexError, TypeError):
                        pass

        except (ValueError, IndexError, AttributeError):
            pass

    # Last resort: exact name matching (not substring to avoid false positives)
    name_lower = name.lower()
    if name_lower in ("mm", "bmm", "addmm", "baddbmm"):
        return "mm"
    elif name_lower in (
        "convolution",
        "convolution2d",
        "convolution3d",
        "conv2d",
        "conv3d",
    ):
        return "conv"
    elif name_lower.startswith("omni_"):
        return "omni"

    return "other"


@dataclasses.dataclass(frozen=True, slots=True)
class PrecompileFunction:
    """A precompile that can be found again, and can be thrown away.

    Two lowerings of the same operation reach the same candidates, and
    building them twice costs the most expensive thing a build does.  So the
    build is kept under a key that says what it was for -- which also means a
    build can be dropped when what it was for stops being wanted, rather than
    being kept for a process that will never ask again.
    """

    fn: Callable[[], dict]
    precompile_key: str | None = None

    def __call__(self) -> dict:
        """Build, and say what it cost -- callable so a caller need not unwrap it.

        What the key is for is bookkeeping; a caller that wants the build does
        not have a reason to know there is a key at all.
        """

        return self.fn()


def get_num_workers() -> int:
    """How many candidates may be built at once."""
    return config.compile_threads


def _log_autotune_exceptions(exceptions: list) -> None:
    """Say which candidates could not be built, and why.

    A candidate that could not be built is not a candidate, and the search
    carries on without it -- so the only way to know one was dropped is this
    record.  What went wrong is reduced to its type and its first line: the
    rest of a failure is usually the same failure repeated per thread, and a
    record that grows with the thread count is one nobody reads.
    """

    if not exceptions:
        return

    for choice, exc in exceptions:
        data = {
            "choice_type": "triton" if isinstance(choice, TritonTemplateCaller) else "other",
            "choice": getattr(choice, "description", None),
            "exception_message": str(exc),
        }
        exc_type_match = re.search(r"(\w+):", str(exc))
        if exc_type_match:
            data["exception"] = exc_type_match.group(1)
        trace_structured("autotune_exception", data)
        log.warning(
            "Exception %s for benchmark choice %s", exc, choice, exc_info=exc
        )


class AlgorithmSelectorCache(PersistentCache):
    """Which candidate ran, remembered from one call to the next.

    A measurement is expensive and a call is not, so what was measured is kept
    and looked up rather than measured again.  The key is what the inputs *are*
    -- their device, their types, their sizes, their strides -- and deliberately
    not where the result lands: two calls with the same inputs and different
    destinations pick the same way of computing it, and a decision that
    distinguished them would be a decision made twice for no reason.

    The candidates themselves are also filtered before they are measured, by a
    coarse pass and then a fine one, because measuring every configuration of
    every template to find the best one would cost more than the compile it is
    meant to save.
    """

    #: The parts of a convolution's description that say which convolution was
    #: timed, as opposed to how the search happened to be run.  Recorded per
    #: candidate so that two records of differently-shaped convolutions are
    #: not read as two measurements of the same one.
    CONV_TUNABLE_KEYS = [
        "KERNEL_H",
        "KERNEL_W",
        "KERNEL_D",
        "STRIDE_H",
        "STRIDE_W",
        "STRIDE_D",
        "PADDING_H",
        "PADDING_W",
        "PADDING_D",
        "GROUPS",
        "UNROLL",
    ]

    #: The same idea for omni attention: the block sizes and launch shape that
    #: say what was timed, kept in the order they are first written so that a
    #: record reads the same whichever build wrote it.
    OMNI_ATTENTION_TUNABLE_KEYS = tuple(
        dict.fromkeys(
            [
                "num_warps",
                "num_stages",
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_M1",
                "BLOCK_N1",
                "BLOCK_M2",
                "BLOCK_N2",
                "USE_TMA",
                "kpack",
                "matrix_instr_nonkdim",
                "waves_per_eu",
            ]
        )
    )

    def __init__(self) -> None:
        # A lowering is not necessarily the first thing to ask about a given set
        # of inputs, so the record of what was precompiled for those inputs is
        # shared by every lowering that reaches them.
        self.prescreening_cache: dict[str, list] = {}
        self.precompile_cache: dict[str, PrecompileFunction] = {}
        self.feedback_saver_fns: list = []
        self.preprocessing_fns: list = [
            filter_choices_by_name_regex,
            filter_choices_by_desc_regex,
        ]
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        self.prescreening_cache.clear()

    def pick_deterministic_choice(self, choices: list) -> ChoiceCaller:
        """The candidate to use when nothing may be measured.

        The operation's own kernel when it is among the candidates, because it is
        the one whose answer does not depend on which machine this is.  A caller
        that asked for determinism is asking for the same answer twice, and the
        only candidate that gives it is the one that was written once.
        """

        if len(choices) < 2:
            raise AssertionError(f"expected at least 2 choices, got {len(choices)}")
        externs = [
            choice for choice in choices if isinstance(choice, ExternKernelCaller)
        ]
        if len(externs) > 0:
            return externs[0]
        else:
            return choices[0]

    @staticmethod
    def _is_extern(choice: ChoiceCaller) -> bool:
        """Whether a candidate is the operation's own rather than a template's.

        Asked per candidate because the two are measured differently: one has to
        be built and the other is already there, and a pass that treated them the
        same would spend the difference on nothing.
        """

        return isinstance(choice, (ExternKernelCaller, SubgraphChoiceCaller))

    @staticmethod
    def key_of(node):
        """The pieces of a value that a stored decision must agree with.

        Its device, its type, its sizes, its strides and where it starts.  What
        is left out is as deliberate as what is in: a decision does not depend on
        which node the value came from, nor on its name, nor on where the result
        of the call goes -- so two calls that differ only in those share a
        measurement, which is the whole reason for keeping one.
        """

        sizevars = V.graph.sizevars
        return (
            node.get_device().type,
            str(node.get_dtype()),
            *sizevars.optimization_hints(node.get_size()),
            *V.graph.sizevars.optimization_hints(
                get_strides_with_layout_constraints(node)
            ),
            sizevars.optimization_hint(node.get_layout().offset),
        )

    def create_no_valid_choices(self, name: str, reason: str) -> NoValidChoicesError:
        """The failure for a call nothing was offered for, saying what was empty."""

        return NoValidChoicesError(
            f"No choices to select. Provided reason: {reason} "
            f"please consider adding the operation's own kernel to the "
            f"candidates for {name} to allow at least one choice. "
        )

    def __call__(
        self,
        name,
        choices: list,
        input_nodes,
        layout,
        input_gen_fns: dict | None = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template: bool = False,
        best_config_future=None,
        is_collective: bool = False,
        min_speedup_threshold: float = 1.0,
        benchmark_with_cudagraphs: bool = False,
    ):
        """The node and the candidate that ran, out of everything offered.

        One candidate is not a choice, so it is used without being measured: a
        measurement costs more than the answer and would find that the only
        candidate is the fastest, which is what was already known.  The rest are
        measured, and the fastest becomes the answer unless determinism was asked
        for, in which case the operation's own kernel does.
        """

        for preprocessing_fn in self.preprocessing_fns:
            choices = preprocessing_fn(choices)

        if benchmark_with_cudagraphs:
            for choice in choices:
                choice._benchmark_with_cudagraphs = True

        # A candidate that needs its inputs to be particular values cannot be
        # measured against values that were made up, so the deferred path is off
        # for those and the answer is one template rather than several.
        if input_gen_fns is not None:
            return_multi_template = False

        if len(choices) == 0:
            raise self.create_no_valid_choices(name, "No choices exist for backend.")
        log.debug("Max autotune selects from %s choices.", len(choices))

        if len(choices) == 1:
            node = choices[0].output_node()
            return node, choices[0]

        if config.deterministic:
            choice = self.pick_deterministic_choice(choices)
            node = choice.output_node()
            return node, choice

        if is_collective:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns, is_collective=True
            )
        elif config.autotune_in_subproc:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns
            )
        else:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns
            )

        if min_speedup_threshold > 1.0:
            best_choice = min(benchmark_results, key=benchmark_results.get)
            if min_speedup_threshold * benchmark_results[best_choice] > benchmark_results[
                min(
                    (c for c in benchmark_results if self._is_extern(c)),
                    key=benchmark_results.get,
                    default=best_choice,
                )
            ]:
                # nothing beat the operation's own kernel by enough to be worth
                # the kernel that would have to be trusted instead
                best_choice = min(
                    (c for c in benchmark_results if self._is_extern(c)),
                    key=benchmark_results.get,
                    default=best_choice,
                )
            node = best_choice.output_node()
            return node, best_choice

        node = self.do_autotuning(
            name, input_nodes, layout, input_gen_fns, choices, benchmark_results
        )
        return node, benchmark_results

    def do_autotuning(
        self,
        name,
        input_nodes,
        layout,
        input_gen_fns,
        choices,
        benchmark_results,
    ):
        """The node the fastest candidate produced, or the operation's own.

        The floor is the operation's own kernel among the measured ones: a
        template that cannot be shown to be faster than the thing it replaces is
        not worth compiling, and the number it was measured at is what "faster"
        is compared against rather than a constant.
        """

        best = min(
            benchmark_results,
            key=lambda choice: (
                not self._is_extern(choice),
                benchmark_results[choice],
            ),
        )
        return best.output_node()

    def benchmark(
        self,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
        is_collective: bool = False,
    ):
        """How long each candidate takes, measured against the same values.

        Every candidate is measured on one set of values, made once and handed to
        each in turn, because a candidate measured against different values is
        not comparable to one measured against these.
        """

        log.debug("Starting autotuning")
        return self.benchmark_in_current_process(
            choices,
            input_nodes,
            layout,
            input_gen_fns,
            hint_override=hint_override,
            is_collective=is_collective,
        )

    @classmethod
    def benchmark_in_current_process(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
        is_collective: bool = False,
    ) -> dict:
        """Measure the candidates here, where the values already are."""

        autotune_args = cls.get_inputs(
            choices, input_nodes, layout, input_gen_fns, hint_override=hint_override
        )
        return cls.benchmark_choices(choices, autotune_args, is_collective=is_collective)

    @classmethod
    def benchmark_choice(cls, choice: ChoiceCaller, autotune_args: AutotuneArgs) -> float:
        """How long one candidate takes on the values the others were measured on.

        The output is cleared first because a candidate that only partly writes
        it would otherwise be timed for writing the parts it did write, and the
        rest of the previous result would be counted as its work.
        """

        benchmark_tensors = autotune_args.get_benchmark_tensors(cls._is_extern(choice))
        inputs, output = benchmark_tensors.unpack()
        output.zero_()
        result = choice.benchmark(*inputs, out=output)
        device_type = next(
            (tensor.device.type for tensor in inputs if tensor.device.type == "cuda"),
            "cuda",
        )
        if device_type == "cuda" and tp.cuda.is_available():
            tp.cuda.synchronize()  # shake out any device errors
        return result

    @classmethod
    def benchmark_choices(
        cls,
        choices,
        autotune_args: AutotuneArgs,
        is_collective: bool = False,
    ) -> dict:
        """Every candidate's time, and the first one's failure if one fails.

        A candidate that cannot be built is a measurement of infinity rather than
        a failure of the whole thing: the other candidates are still answers, and
        the one that broke is not one of them.
        """

        timings: dict = {}
        for choice in choices:
            try:
                result = cls.benchmark_choice(choice, autotune_args)
                timings[choice] = float(result)
            except Exception:
                timings[choice] = float("inf")
        return timings

    @classmethod
    def get_inputs(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns: dict | None,
        hint_override: int | None = None,
    ) -> AutotuneArgs:
        """The values every candidate is measured against, made once.

        Made from the nodes rather than from the program, because the program is
        being compiled and has no values in it yet.  A candidate that asked for
        particular values has them made the way it asked, and one that did not is
        measured against values that are merely of the right shape -- which is
        enough to time it and not enough to be sure it is right, which is why the
        operation's own result is what a timed template is checked against.
        """

        if input_gen_fns is None:
            input_gen_fns = {}

        # one value per name, so that two nodes reading the same buffer are
        # measured against the same buffer and a candidate cannot win by being
        # handed something easier than its neighbour was
        unique_example_inputs = {
            x.get_name(): input_gen_fns.get(
                i,
                lambda x: cls.benchmark_example_value(x, hint_override=hint_override),
            )(x)
            for i, x in enumerate(input_nodes)
        }
        example_inputs = list(unique_example_inputs.values())

        extern_choice = next(
            (choice for choice in choices if cls._is_extern(choice)), None
        )
        extern_input_nodes = input_nodes
        if extern_choice is not None:
            if len(extern_choice.input_nodes) != len(input_nodes):
                raise AssertionError(
                    "extern_choice.input_nodes length must match input_nodes: "
                    f"{len(extern_choice.input_nodes)} != {len(input_nodes)}"
                )
            extern_input_nodes = extern_choice.input_nodes

        # The operation's own kernel is handed each value at the geometry its own
        # node says, which is not always the geometry the templates were handed:
        # a bias is one value to one and a row to the other.  So the two are kept
        # apart, and each is viewed onto the storage it needs -- a value whose
        # stride reaches past what was made for it would read past the end, so a
        # larger buffer is made rather than a kernel being handed a view it
        # cannot safely write.
        example_inputs_extern: list = []
        for i, input_node in enumerate(extern_input_nodes):
            input_tensor = unique_example_inputs[input_node.get_name()]
            base = input_tensor

            if i in input_gen_fns:
                sizes = tuple(input_tensor.size())
                strides = tuple(input_tensor.stride())
                storage_offset = input_tensor.storage_offset()
            else:
                sizes = V.graph.sizevars.optimization_hints_with_override(
                    input_node.get_size(),
                    hint_override=hint_override,
                )
                strides = V.graph.sizevars.optimization_hints_with_override(
                    get_strides_with_layout_constraints(input_node),
                    hint_override=hint_override,
                )
                storage_offset = V.graph.sizevars.optimization_hint_with_override(
                    input_node.get_layout().offset,
                    hint_override=hint_override,
                )

            needed_size = compute_required_storage_length(
                sizes, strides, int(storage_offset)
            )
            current_size = base.untyped_storage().size() // base.element_size()
            if needed_size > current_size:
                new_base = tp.empty(
                    needed_size, dtype=base.dtype, device=base.device
                )
                base = new_base.as_strided(
                    base.size(), base.stride(), base.storage_offset()
                )

            example_inputs_extern.append(
                tp.as_strided(base, sizes, strides, storage_offset)
            )

        out = cls.benchmark_example_value(layout, hint_override=hint_override)

        # the place the result goes may reach past what was made for it, and what
        # was made is what has to be big enough for what a kernel writes
        out_base = out
        out_offset = V.graph.sizevars.optimization_hint(layout.offset, fallback=0)
        needed_out_size = compute_required_storage_length(
            out.size(), out.stride(), out_offset
        )
        current_out_size = out_base.untyped_storage().size() // out_base.element_size()
        if needed_out_size > current_out_size:
            new_out_base = tp.empty(
                needed_out_size, dtype=out_base.dtype, device=out_base.device
            )
            out_base = new_out_base.as_strided(
                new_out_base.size(),
                new_out_base.stride(),
                new_out_base.storage_offset(),
            )

        out_extern = tp.as_strided(out_base, out.size(), out.stride(), out_offset)
        # What the answer is checked against, when it is going to be checked.
        # Produced by running a candidate rather than by trusting one, so it is
        # only produced when there is a candidate that can be run: the list
        # holds an operation the framework already has, whose kernel is a name
        # rather than something to call, and asking that for a result is asking
        # it to do the one thing it does not do.  Left as nothing, the check
        # simply does not happen rather than happening against a wrong answer.
        expected = None
        if VERIFY and choices:
            for candidate in choices:
                try:
                    candidate.benchmark(*example_inputs, out=out)
                except NotImplementedError:
                    continue
                expected = out.clone()
                break
        return AutotuneArgs.from_choice_args(
            example_inputs, example_inputs_extern, out, out_extern, expected
        )

    @classmethod
    def generate_example_value(
        cls, size, stride, device, dtype, extra_size, allocation_size=None
    ):
        """A value for a result, of the result's geometry and holding values.

        The random state is saved and put back because making a value here would
        otherwise move the program's own random numbers along: a program compiled
        twice would then get different results, and a measurement is not worth a
        result that depends on whether anything was timed first.

        A result that is larger than the part of it that is written is made at the
        larger size and then viewed at the smaller one, because a kernel that
        writes the whole of what it was given may write past the part that is read
        and there has to be somewhere for that to land.
        """

        state = tp.get_rng_state()
        try:
            if allocation_size is None or allocation_size == size:
                return rand_strided(
                    size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                )
            else:
                return rand_strided(
                    allocation_size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                ).as_strided(size, stride)
        finally:
            tp.set_rng_state(state)

    @staticmethod
    def benchmark_example_value(node, hint_override: int | None = None):
        """One node made into a value it can be measured on.

        A layout counts as a node here, because a result's geometry is known
        before anything has been computed and is what the place to write has to
        be made from.  A view is unwrapped first, because what a view holds is
        not what was read and a value made from the view's own description would
        be a different value.
        """

        if isinstance(node, Layout):
            node = _Buffer(name="fake", layout=node)
        original_dtype = node.get_dtype()
        if isinstance(node, BaseView):
            node = node.unwrap_view()

        result = AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.optimization_hints_with_override(
                node.get_size(),
                hint_override=hint_override,
            ),
            V.graph.sizevars.optimization_hints_with_override(
                get_strides_with_layout_constraints(node),
                hint_override=hint_override,
            ),
            node.get_device(),
            node.get_dtype(),
            V.graph.sizevars.optimization_hint_with_override(
                node.layout.offset,
                hint_override=hint_override,
            ),
            V.graph.sizevars.optimization_hints_with_override(
                V.graph.get_allocation_size(node),
                hint_override=hint_override,
            ),
        )
        if result.dtype != original_dtype:
            result = result.view(original_dtype)
        return result

    def prescreen_choices(self, choices: list, name: str) -> list:
        """The candidates worth measuring at all, by a cheap pass.

        Measured rather than timed, because a candidate that is wrong cannot be
        the answer however fast it is, and finding that out by measurement would
        be too late.
        """

        return list(choices)

    def prune_choices_postscreen(self, choices: list, name: str, pruned: list) -> None:
        """Which candidates the coarse pass took out, recorded so it is not redone."""

        inputs_key = create_inputs_key([])
        self.prescreening_cache[inputs_key] = [
            choice.name for choice in pruned
        ]

    def log_results(self, name: str, choices: list, benchmark_results: dict) -> None:
        """What each candidate was measured at, when the answer is worth recording.

        Logged rather than returned because a caller that wanted the numbers
        would have had to measure everything, and the numbers are for whoever
        asks afterwards whether the chosen candidate was the right one.
        """

        if not log.isEnabledFor(logging.INFO):
            return
        log.info(
            "Autotuning %s over %s candidates:\n%s",
            name,
            len(choices),
            "\n".join(
                "\t%s: %s"
                % (
                    choice,
                    (
                        f"{benchmark_results[choice] * 1e3:6.1f}us"
                        if choice in benchmark_results
                        else "N/A"
                    ),
                )
                for choice in choices
            ),
        )

    def add_feedback_saver(self, fn) -> None:
        self.feedback_saver_fns.append(fn)

    def clear_feedback_savers(self) -> None:
        self.feedback_saver_fns.clear()

    def add_preprocessing_fn(self, fn) -> None:
        self.preprocessing_fns.append(fn)

    def _register_default_preprocessing_fns(self) -> None:
        """Put back the two filters that every search starts from.

        Kept apart from whatever a caller added so that clearing the caller's
        can leave these in place; a search with neither filter would time
        candidates that were never going to be chosen.
        """

        self.add_preprocessing_fn(filter_choices_by_name_regex)
        self.add_preprocessing_fn(filter_choices_by_desc_regex)

    @classmethod
    def benchmark_in_sub_process(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
    ):
        """Measure the candidates that could take this process with them.

        A candidate that reads what it should not does not raise; it corrupts
        memory, and the process stops.  So a candidate whose failure would be
        silent is measured in a process of its own, and comes back as a time
        or as no time at all -- and a candidate that is already known to be
        safe is measured here, because paying for a process to measure
        something that cannot fail is not worth it.

        The ones that cannot fail here are measured first, so that a process
        lost to a bad candidate does not also cost the times of the good ones.
        """

        from ..autotune_process import AsyncAutotuner
        from ..codegen.cutedsl.cutedsl_template import CuteDSLTemplateCaller

        # A candidate that calls straight into the library cannot corrupt
        # anything, so measuring it here costs nothing to be safe about.
        extern = [c for c in choices if cls._is_extern(c)]
        non_cutlass = [
            c
            for c in choices
            if not cls._is_extern(c) and not isinstance(c, CuteDSLTemplateCaller)
        ]
        cutlass = [c for c in choices if isinstance(c, CuteDSLTemplateCaller)]

        timings = cls.benchmark_in_current_process(
            extern, input_nodes, layout, input_gen_fns, hint_override=hint_override
        )
        # Order the ones measured elsewhere so that valid timings are collected
        # before any candidate that can crash its process has a chance to.
        remote = non_cutlass + cutlass
        if remote:
            inputs_key = str(
                [getattr(n, "get_size", lambda: None)() for n in input_nodes]
            )
            AsyncAutotuner.start(remote, inputs_key)
            timings.update(AsyncAutotuner.get_results(remote, inputs_key))
        return timings

    @classmethod
    def make_benchmark_fn(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
        is_collective=False,
    ):
        """Say how this batch of candidates should be measured, without measuring it.

        Whether a candidate is measured here or elsewhere is decided before any
        of them runs, because the answer is a property of what the candidates
        are rather than of what happened when one of them ran.
        """

        from ..codegen.cutedsl.cutedsl_template import CuteDSLTemplateCaller

        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        has_cutlass = any(isinstance(c, CuteDSLTemplateCaller) for c in choices)

        # Collective ops must use current process
        if is_collective:
            return functools.partial(
                cls.benchmark_in_current_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
                hint_override=hint_override,
                is_collective=is_collective,
            )
        # A candidate that can leave the device unusable is always measured
        # elsewhere, whether or not measuring elsewhere was asked for.
        elif config.autotune_in_subproc or has_cutlass:
            return functools.partial(
                cls.benchmark_in_sub_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
                hint_override=hint_override,
            )
        else:
            return functools.partial(
                cls.benchmark_in_current_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
                hint_override=hint_override,
            )

    @staticmethod
    def maybe_log_mm_results(
        name: str, input_nodes: list, timings: dict
    ) -> None:
        """Record what a matrix-multiplication search found, if asked to.

        The extents are the key rather than part of the record: the same three
        numbers describe every measurement of the same operation, so a lookup
        by them says how this one compared with the ones measured before.
        """

        mm_filename = get_mm_log_filename()
        if not mm_filename:
            return

        # Classify operation to ensure it's actually an MM operation
        choices_list = list(timings.keys())
        operation_type = _classify_kernel_operation(name, choices_list, input_nodes)
        if operation_type != "mm":
            return

        if len(input_nodes) < 2:
            return

        M, K = input_nodes[-2].get_size()[:2]
        N = input_nodes[-1].get_size()[-1]

        def get_choice_info(choice):
            if isinstance(choice, ExternKernelCaller):
                return {"type": "extern", "time": timings[choice]}

            if isinstance(choice, TritonTemplateCaller):
                info = choice.info_dict()
                tile = info["tile_shape"]

                tile_vals = eval(tile)
                BLOCK_M = tile_vals[0]
                BLOCK_K = tile_vals[1]
                BLOCK_N = tile_vals[2]

                return {
                    "type": "triton",
                    "time": timings[choice],
                    "BLOCK_M": BLOCK_M,
                    "BLOCK_K": BLOCK_K,
                    "BLOCK_N": BLOCK_N,
                    "num_stages": info["num_stages"],
                    "num_warps": info["num_warps"],
                    "waves_per_eu": info.get("waves_per_eu", 0),
                    "matrix_instr_nonkdim": info.get("matrix_instr_nonkdim", 0),
                    "kpack": info.get("kpack", 2),
                }
            return None

        out_dict = {
            str((M, K, N)): [get_choice_info(choice) for choice in timings],
            "kernel_type": name,
        }

        append_to_log(mm_filename, out_dict)

    @staticmethod
    def maybe_log_conv_results(
        name: str, input_nodes: list, timings: dict
    ) -> None:
        """Record what a convolution search found, if asked to.

        Everything the candidate said about itself is kept, rather than a fixed
        set of fields: what a convolution can be varied by is not a short list,
        and a record that dropped the rest would not say which convolution was
        measured.  A value that cannot be written as JSON is written as its
        text rather than dropped, since a field as text still names what it was.
        """

        conv_filename = get_conv_log_filename()
        if not conv_filename:
            return

        # Classify operation to ensure it's actually a conv operation
        choices_list = list(timings.keys())
        operation_type = _classify_kernel_operation(name, choices_list, input_nodes)
        if operation_type != "conv":
            return

        if len(input_nodes) < 2:
            return

        x_size = input_nodes[0].get_size()
        w_size = input_nodes[1].get_size()

        def get_conv_choice_info(choice):
            if choice not in timings:
                return None
            info = choice.info_dict()

            # Start with timing and backend type
            result = {
                "time": timings[choice],
                "backend": info.get("backend", "unknown"),
            }

            # Add all parameters from info_dict
            for key, value in info.items():
                if key != "backend":  # Already added
                    try:
                        json.dumps(value)  # Test if serializable
                        result[key] = value
                    except (TypeError, ValueError):
                        result[key] = str(value)

            return result

        out_dict = {
            "input_shape": str(x_size),
            "weight_shape": str(w_size),
            "choices": [
                get_conv_choice_info(choice)
                for choice in timings
                if get_conv_choice_info(choice) is not None
            ],
            "kernel_type": name,
        }

        append_to_log(conv_filename, out_dict)

    @staticmethod
    def get_omni_attention_choice_info(choice, timings: dict) -> dict:
        """What one candidate contributed to an omni-attention record."""

        if isinstance(choice, ExternKernelCaller):
            return {"type": "extern", "time": timings[choice]}

        if not isinstance(choice, TritonTemplateCaller):
            raise AssertionError(
                f"expected choice to be TritonTemplateCaller, got {type(choice)}"
            )

        info = choice.info_dict()
        result = {
            "type": "triton",
            "time": timings[choice],
        }

        for key in AlgorithmSelectorCache.OMNI_ATTENTION_TUNABLE_KEYS:
            if key in info:
                result[key] = info[key]

        return result

    @staticmethod
    def _omni_attention_log_dim(dim) -> int:
        """One extent as a number, settling a symbolic one if it can be.

        A record is read by someone looking for two runs of the same shape, so
        an extent that is still symbolic is resolved to what the solver settled
        it to.  One that cannot be resolved is a shape this record cannot
        describe, and saying so is better than writing an expression that will
        not read back as a number.
        """

        if isinstance(dim, tp.SymInt):
            dim = dim.node.expr

        if type(dim) is int or isinstance(dim, sympy.Integer):
            return int(dim)

        if isinstance(dim, sympy.Expr):
            return V.graph.sizevars.optimization_hint(dim)

        raise TypeError(
            f"Unexpected omni attention log dimension type {type(dim).__name__}: {dim}"
        )

    @staticmethod
    def _omni_attention_log_shape(size) -> str:
        """A whole shape as text, with every extent settled to a number."""

        dims = [AlgorithmSelectorCache._omni_attention_log_dim(dim) for dim in size]
        return f"[{', '.join(map(str, dims))}]"

    @staticmethod
    def maybe_log_omni_attention_results(
        name: str, input_nodes: list, timings: dict
    ) -> None:
        """Record what an omni-attention search found, if asked to.

        Which of the three passes a candidate belongs to is part of the record
        rather than something a reader has to infer from its name, because the
        three take different shapes: the backward pass is handed the scores it
        produced, and the decoding pass has no sequence axis to speak of, so a
        record without that would have shapes that cannot be compared.
        """

        omni_attention_filename = get_omni_attention_log_filename()
        # Support both omni_attention and omni_decoding
        if not omni_attention_filename or (
            "omni_attention" not in name and "omni_decoding" not in name
        ):
            return

        if len(input_nodes) < 3:
            return

        query_size = input_nodes[0].get_size()
        key_size = input_nodes[1].get_size()
        value_size = input_nodes[2].get_size()

        # Handle both 4D (forward/backward) and 5D (decode) tensor formats
        # 4D: [B, H, seq_len, head_dim]
        # 5D: [B, H, 1, 1, head_dim] (decode mode has extra dimension)
        if len(query_size) == 5:
            # Decode mode with 5D tensors
            B = query_size[0]
            Hq = query_size[1]
            # query_size[2] and query_size[3] are both 1 for decode
            seq_len_q = query_size[2]  # This will be 1
            qk_head_dim = query_size[4]  # Head dim is at index 4 for 5D
            Hkv = key_size[1]
            seq_len_kv = key_size[2]
            v_head_dim = value_size[4] if len(value_size) == 5 else value_size[3]
        else:
            # Forward/backward mode with 4D tensors
            B = query_size[0]
            Hq = query_size[1]
            seq_len_q = query_size[2]
            qk_head_dim = query_size[3]
            Hkv = key_size[1]
            seq_len_kv = key_size[2]
            v_head_dim = value_size[3]

        kernel_type = (
            "backward"
            if "backward" in name
            else ("decode" if "decoding" in name else "forward")
        )

        # Create shape info dictionary
        shape_info = {
            "kernel_type": kernel_type,
            "B": AlgorithmSelectorCache._omni_attention_log_dim(B),
            "Hq": AlgorithmSelectorCache._omni_attention_log_dim(Hq),
            "Hkv": AlgorithmSelectorCache._omni_attention_log_dim(Hkv),
            "seq_len_q": AlgorithmSelectorCache._omni_attention_log_dim(seq_len_q),
            "seq_len_kv": AlgorithmSelectorCache._omni_attention_log_dim(seq_len_kv),
            "qk_head_dim": AlgorithmSelectorCache._omni_attention_log_dim(qk_head_dim),
            "v_head_dim": AlgorithmSelectorCache._omni_attention_log_dim(v_head_dim),
        }

        sorted_choices = sorted(timings, key=timings.__getitem__)

        # Include shape info in each choice
        choices_with_shapes = []
        for choice in sorted_choices:
            choice_info = AlgorithmSelectorCache.get_omni_attention_choice_info(
                choice, timings
            )
            # Merge shape info with choice info
            choice_info.update(shape_info)
            choices_with_shapes.append(choice_info)

        out_dict = {
            "query_shape": AlgorithmSelectorCache._omni_attention_log_shape(query_size),
            "key_shape": AlgorithmSelectorCache._omni_attention_log_shape(key_size),
            "value_shape": AlgorithmSelectorCache._omni_attention_log_shape(value_size),
            "kernel_type": kernel_type,
            "choices": choices_with_shapes,
        }
        append_to_log(omni_attention_filename, out_dict)

    def make_precompile_fn(
        self,
        choices,
        name: str,
        inputs_key: str,
        precompilation_timeout_seconds: int | None = 60 * 60,
    ) -> Callable[[], dict]:
        """Build the candidates ahead of being measured, and say what it cost.

        Building a candidate is most of what a search costs, and it is worth
        doing before the measuring rather than during it: a search that
        measures candidates one at a time spends its time waiting on a single
        build while the rest of the machine is idle.  So they are built
        together first, and the time each took is what the search reports
        rather than the time the build appeared to cost.

        Several answers to "do this need doing" come back as doing nothing,
        each for its own reason: a search that is not timed at all, a set of
        candidates already timed on another machine, a set already built for
        these inputs by another lowering, and a machine with no spare workers.
        Building twice is the one outcome worth spending effort to avoid, and
        a candidate that differs only in what it will be handed at run time
        builds to the same thing -- so those are built once.
        """

        log.debug("Starting precompilation")

        def no_op(*args, **kwargs) -> dict:
            return {}

        if (
            precompilation_timeout_seconds is None
            or precompilation_timeout_seconds <= 0
        ):
            log.debug("Precompilation timeout is None or <= 0, returning no_op")
            return no_op

        num_workers = min(get_num_workers(), len(choices))

        if num_workers <= 0:
            return no_op

        # check local and global cache before precompiling
        timings = self.lookup(
            choices,
            name,
            inputs_key,
            benchmark=None,
        )

        if timings and len(timings) == len(choices):
            # compilation in precompile stage is much cheaper than that in
            # autotuning stage
            log.debug("Found all %d timings in cache, returning no_op", len(timings))
            return no_op

        precompile_key = create_precompile_key(name, inputs_key, choices)
        if precompile_func := self.precompile_cache.get(precompile_key):
            log.debug("Precompile function found in cache, returning it")
            return precompile_func

        log.info(
            "Multithreaded precompilation for %d choices using %d worker threads",
            len(choices),
            num_workers,
        )

        # Because threads inherit global state, a pool can race and leave the
        # output streams somewhere other than where it found them; so each
        # build puts them back rather than assuming they were left alone.
        def precompile_with_captured_stdout(choice):
            log.debug("Precompiling choice with captured stdout: %s", choice)
            start_ns = time.time_ns()
            with restore_stdout_stderr():
                choice.precompile()
            elapsed_ns = time.time_ns() - start_ns
            return None, elapsed_ns // 1000

        def on_complete(future):
            if not future.exception():
                _, precompile_elapsed_us = future.result()
                elapsed_seconds = precompile_elapsed_us / 1e6
                elapsed_times[future] = elapsed_seconds
                log.debug(
                    "Precompilation complete for future: %s, elapsed time: %.02fs",
                    future,
                    elapsed_seconds,
                )

        if use_pipelined_autotuning():
            executor = PrecompileThreadPool.get_instance()
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)

        futures: dict = {}
        elapsed_times: dict = {}

        # Some choices only differ in runtime arguments, so we
        # skip a choice if it has the same hash as a previously seen choice
        seen_choices: OrderedSet = OrderedSet()

        for c in choices:
            # Skip choices which we have already issued a precompile
            if c.kernel_hash_key() in seen_choices:
                log.debug("Skipping already seen choice: %s", c)
                continue
            else:
                seen_choices.add(c.kernel_hash_key())

            if hasattr(c, "precompile"):
                future = executor.submit(precompile_with_captured_stdout, c)
                log.debug("Submitted precompile for choice: %s", c)

                future.add_done_callback(on_complete)
                futures[future] = c

        @functools.cache
        @restore_stdout_stderr()
        def wait_on_futures() -> dict:
            """Wait for every build, and return what each one cost.

            Waiting twice returns the same answer rather than building twice,
            which is why this is remembered: a search that asks how long its
            candidates took, and then asks again, is asking the same question.
            """
            log.debug("Waiting on futures")
            counters["inductor"]["select_algorithm_precompile"] += 1
            exceptions: list = []
            try:
                for future in as_completed(
                    futures,
                    timeout=precompilation_timeout_seconds,
                ):
                    if e := future.exception():
                        counters["inductor"][
                            "select_algorithm_num_precompilation_exceptions"
                        ] += 1
                        exceptions.append((futures[future], e))
                        log.exception(
                            "Exception %s for benchmark choice %s",
                            e,
                            futures[future],
                            exc_info=e,
                        )
                        futures[future].mark_failed()
                    else:
                        counters["inductor"]["select_algorithm_num_precompiles"] += 1
                        log.info(
                            "Precompiling benchmark choice %s took %.02fs",
                            futures.get(future),
                            elapsed_times.get(future),
                        )
            except TimeoutError:
                # A build that has not finished in an hour is a build that is
                # not going to; the whole build is not abandoned for it, and
                # the candidates still building are treated as candidates that
                # could not be built.
                completed_futures = OrderedSet([f for f in futures if f.done()])
                remaining_futures = OrderedSet(futures.keys()) - completed_futures

                log.warning(
                    "Precompilation timeout after %ds: %d of %d futures did not complete",
                    precompilation_timeout_seconds,
                    len(remaining_futures),
                    len(futures),
                )

                # Mark remaining futures as failed and log them
                for future in remaining_futures:
                    choice = futures[future]
                    log.warning(
                        "Marking choice as failed due to timeout: %s",
                        choice,
                    )
                    choice.mark_failed()
                    # Add timeout exception to the exceptions list
                    timeout_exc = TimeoutError(
                        f"Precompilation timed out after {precompilation_timeout_seconds}s"
                    )
                    exceptions.append((choice, timeout_exc))
            if exceptions:
                _log_autotune_exceptions(exceptions)

            if not use_pipelined_autotuning():
                executor.shutdown(wait=True)

            # Build and return dict mapping choices to their precompilation times
            precompile_times: dict = {}
            for future, choice in futures.items():
                if future in elapsed_times:
                    precompile_times[choice] = elapsed_times[future]
            return precompile_times

        precompile_fn = PrecompileFunction(wait_on_futures, precompile_key)
        self.precompile_cache[precompile_key] = precompile_fn

        return precompile_fn

    def autotune(
        self,
        name,
        input_nodes,
        layout,
        input_gen_fns,
        choices,
        hint_override: int | None = None,
        is_collective=False,
        precompile_key: str | None = None,
    ):
        """Time every candidate and keep the fastest answer.

        The search is timed as a whole, because a search that takes longer than
        the kernel it is choosing for has cost more than it saved -- and that
        is only visible if the search is measured rather than the candidates
        alone.  What it was measured on is recorded with it, since two
        searches over differently-shaped values are not comparable.
        """

        log.debug("Starting autotuning")

        with timed_block(f"{name}_template_autotuning"):
            trace_structured(f"{name}_template_autotuning", _autotune_metadata(input_nodes))
            benchmark_results = self.benchmark(
                choices,
                input_nodes,
                layout,
                input_gen_fns,
                hint_override=hint_override,
                is_collective=is_collective,
                precompile_key=precompile_key,
            )
            if config.max_autotune_report_choices_stats:
                _log_autotune_choices_stats(
                    f"{name}_template_autotuning", benchmark_results
                )
            return benchmark_results

    def clear_preprocessing_fns(self) -> None:
        """The candidates are filtered by name and by description before they are
        measured; those two are put back and anything a caller added is dropped,
        so a filtering that has gone stale cannot keep narrowing the list.
        """

        self.preprocessing_fns.clear()
        self.preprocessing_fns.extend(
            (filter_choices_by_name_regex, filter_choices_by_desc_regex)
        )


_ALGORITHM_SELECTOR_CACHE: AlgorithmSelectorCache | None = None


def get_algorithm_selector_cache() -> AlgorithmSelectorCache:
    """The one record of what was chosen, made when it is first needed.

    One rather than one per call because a measurement is only worth keeping if
    the next call can find it, and two records would each hold half the answers.
    """

    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    return _ALGORITHM_SELECTOR_CACHE


def autotune_select_algorithm(*args, **kwargs):
    """Measure the candidates and hand back the one that ran.

    Every default a caller did not give is filled in here rather than at each
    call site, so that a caller who did not think about a setting gets the same
    one as a caller who did.
    """

    cache = get_algorithm_selector_cache()
    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = config.benchmark_epilogue_fusion
    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = 60 * 60
    return cache(*args, **kwargs)


def realize_inputs(*args):
    """The values as they must be handed to something outside this compiler.

    Realized and made to run along their last axis, because a caller that is not
    this compiler reads the values it is given and cannot be told what a stride
    means.
    """

    if len(args) == 1:
        realized = args[0]
        if not realized.has_tensor_output():
            return realized
        return realized.require_stride1()
    return [realize_inputs(x) for x in args]


def get_ktc(
    kernel_inputs: KernelInputs,
    template: Any,
    op_name: str,
    kwarg_overrides: dict | None = None,
):
    """The deferred candidates of one template for one call.

    Deferred because a table is offered whole and most of it is never built: the
    configurations are produced as a generator, and the kernel behind one is only
    built if that configuration is reached.  Building them all to choose between
    them would cost more than choosing does.

    The heuristic is asked three separate questions -- which configurations fit,
    what the inputs become, and what the kernel is told that is not a choice --
    because the answers change independently: a template may fit more shapes
    than it has useful configurations for, and may need to be told something
    about the call that is not one of its configurations.
    """

    device_type = kernel_inputs.device_type
    if device_type is None:
        raise AssertionError("get_ktc requires a valid device type")
    heuristic = get_template_heuristic(template, device_type, op_name)
    cs = heuristic.get_template_configs(kernel_inputs, op_name)
    inputs_val = heuristic.adjust_kernel_inputs(kernel_inputs, op_name)
    extra_kwargs = heuristic.get_extra_kwargs(kernel_inputs, op_name)
    overrides = kwarg_overrides or {}
    return make_ktc_generator(
        template=template,
        cs=cs,
        extra_kwargs=extra_kwargs,
        overrides=overrides,
        layout=kernel_inputs.output_layout()
        if hasattr(kernel_inputs, "output_layout")
        else None,
        inputs=inputs_val,
    )


def get_template_heuristic(template: Any, device_type: str, op_name: str):
    """The rule that says which configurations of a template are worth trying.

    Asked for by what a template calls itself, not by the template, so that a
    rule lives in the table beside every other rule rather than hanging off the
    thing it is a rule about.  A template and an operation's own kernel can then
    be told apart by name alone, which is what lets both go through the same
    question and the same answer.
    """

    return registry_get_template_heuristic(template.uid, device_type, op_name)


def get_template_configs(
    kernel_inputs: KernelInputs,
    templates: list,
    op_name: str,
    kwarg_overrides: dict | None = None,
) -> list:
    """The candidates of every template offered, as the built things among them.

    A template that declines a configuration contributes nothing rather than a
    refusal, and a template that declines all of them leaves the list with the
    other templates' candidates in it -- so a call is measured against whatever
    can actually be measured, and a template's absence is not a failure of the
    call.
    """

    if kwarg_overrides is None:
        kwarg_overrides = {}
    input_tensors = (
        kernel_inputs.nodes() if hasattr(kernel_inputs, "nodes") else []
    )
    if len(input_tensors) < 2 and not input_tensors:
        raise ValueError(
            f"Need at least 2 input tensors, got {len(input_tensors)}"
        )
    template_choices = {}
    for template in templates:
        template_choices[template.uid] = get_ktc(
            kernel_inputs, template, op_name, kwarg_overrides.get(template.uid, {})
        )
    adjusted_choices = _finalize_template_configs(
        template_choices, kernel_inputs, templates, op_name, kwarg_overrides
    )
    if _need_to_fix_layout(adjusted_choices, op_name):
        layout = (
            kernel_inputs.output_layout(flexible=False)
            if hasattr(kernel_inputs, "output_layout")
            else None
        )
        for ktc in adjusted_choices:
            ktc.layout = layout
            if hasattr(ktc, "_choice"):
                del ktc._choice
    return [ktc.choice for ktc in adjusted_choices if ktc.choice is not None]


def _finalize_template_configs(
    template_choices: dict,
    kernel_inputs: KernelInputs,
    templates: list,
    op_name: str,
    kwarg_overrides: dict | None = None,
) -> list:
    """Every template's candidates in one list, after any adjustment.

    Split out so that a caller may change the candidates before they are built,
    which is the only time a change is cheap: the incoming candidates are
    generators, and becoming a built kernel is what costs.
    """

    choices: list = []
    for choice_gen in template_choices.values():
        choices.extend(choice_gen)
    return choices


def _need_to_fix_layout(adjusted_choices: list, op_name: str) -> bool:
    """Whether the candidates have to be measured against a fixed result geometry.

    A candidate that can take a flexible geometry will take whichever was asked
    for, and one that cannot has to be told.  So this is asked only when it would
    change something: a measurement against a geometry no candidate can produce
    is a measurement of nothing.
    """

    if len(adjusted_choices) > 0:
        if adjusted_choices[0].inputs.device_type == "mps" and op_name not in (
            "mm",
            "addmm",
        ):
            return True
    if not (config.max_autotune or config.max_autotune_gemm):
        return False
    # a measured candidate is a built kernel, and a built kernel's geometry is
    # part of what it was built for
    return any(
        (not isinstance(ktc.template, ExternKernelCaller) for ktc in adjusted_choices)
    )



class PartialRender:
    """Some of a template's parts are written last but belong at the start.

    A template's shape -- what its kernel looks like, what it loads, what it
    stores -- is settled by walking the region, and the region is walked after
    the template's own text is laid out.  So a part that is not known when the
    text is laid out is written into the text as a placeholder, and filled in
    once the walk has said what it is.

    Asking for the code before every placeholder has been filled in is an error
    rather than a partially-built source: a caller that compiles what it gets
    would compile a program with a placeholder in it, and the failure would be
    the compiler's rather than here.
    """

    HookFn = Callable[[], str]

    def __init__(self, code: str, replacement_hooks: dict) -> None:
        super().__init__()
        self._code: str = code
        self.replacement_hooks: dict = dict(replacement_hooks)

    @property
    def code(self) -> str:
        """The source, once every placeholder has been filled in.

        Raises while any is still open, naming the ones that are: a caller
        that has not finished cannot tell which part it is missing, and would
        otherwise get source that looks complete.
        """

        remaining = [
            key for key, fn in self.replacement_hooks.items() if fn is not None
        ]
        if remaining:
            raise AssertionError(
                f"these placeholders have not been filled in: {remaining}"
            )
        return self._code

    def _replace_placeholder(self, hook_key: str, result: str) -> str:
        """Put one filled-in part where its placeholder stands.

        A placeholder on a line of its own and one in the middle of a line are
        different problems.  On its own, the text replaces the line and takes
        the line's indentation with it -- but only for a result written flush
        left, because a result that carries its own indentation is saying
        where it belongs and re-indenting it would put it somewhere it does not
        compile.  In the middle of a line, the text goes in as it is, because
        there is no line whose indentation it could take.

        A result with nothing in it removes the line.  That is how a part that
        turned out to be unnecessary says so: the piece was not needed, so
        there is nothing to write and the line it was on should not be there
        either.
        """

        if hook_key not in self._code:
            return self._code

        if not (result and result.strip()):
            lines = self._code.split("\n")
            return "\n".join(
                line for line in lines if line.strip() != hook_key
            )

        lines = self._code.split("\n")
        new_lines = []
        for line in lines:
            if line.strip() == hook_key:
                indent = line[: len(line) - len(line.lstrip())]
                result_lines = result.strip("\n").split("\n")
                non_empty = [one for one in result_lines if one.strip()]
                written_flush_left = bool(non_empty) and all(
                    not one[0].isspace() for one in non_empty
                )
                if written_flush_left:
                    new_lines.append(
                        "\n".join(
                            indent + one if one.strip() else one
                            for one in result_lines
                        ).rstrip()
                    )
                else:
                    new_lines.append(line.replace(hook_key, result))
            elif hook_key in line:
                new_lines.append(line.replace(hook_key, result))
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def finalize_hook(self, hook_key: str, strict: bool = True) -> None:
        """Fill in one placeholder, by name.

        A placeholder that was never registered is a mistake in the caller
        rather than something to tolerate, unless it says otherwise: it means
        the part it expected to write is one this render does not have.

        A placeholder that has already been filled in cannot be filled in
        again, because the text it stood for is gone; saying so is better than
        writing the second answer into a source that no longer has the first
        place to put it.
        """

        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} is not a placeholder in this render"
                )
            return

        hook = self.replacement_hooks[hook_key]
        if hook is None:
            raise AssertionError(f"{hook_key} has already been filled in")
        self._code = self._replace_placeholder(hook_key, hook())
        self.replacement_hooks[hook_key] = None

    def finalize_remaining(self) -> str:
        """Fill in whatever is still open, and return the source.

        For a caller that fills in one placeholder itself -- because it is the
        one that knows what that part is -- and wants the rest done for it.
        Unlike filling them all in, this skips the ones already done.
        """

        for key, fn in list(self.replacement_hooks.items()):
            if fn is not None:
                self.finalize_hook(key)
        return self.code

    def finalize_all(self) -> str:
        """Fill in every placeholder that has not been filled in yet.

        Unlike filling in the remaining ones, this is an error if one has
        already been done: a caller that has filled one in itself and then says
        "fill in the rest" means the rest, not all of them.
        """

        for key in self.replacement_hooks:
            self.finalize_hook(key)
        return self.code
