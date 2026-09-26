"""A loop nest captured as a graph, so that it can be looked at rather than run.

Running a body and capturing what it did is the only way to see which buffers
it touches and at which positions, and that is what every decision about loop
order, fusion and vector width rests on.  Capturing costs a trace, so a body
that is only being reordered or re-split is copied rather than re-captured.

The positions themselves are held apart from the graph.  A position is a
symbolic expression, and holding it once under a name is what lets the same
position be recognised when two bodies that compute it differently are
compared, and what lets an index be rewritten when the loops are.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import itertools
import re
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, TypeVar

import sympy

import tensorplay as tp
from tensorplay.graph import Graph as FxGraph
from tensorplay.graph import GraphModule
from tensorplay.graph import Interpreter as FxInterpreter
from tensorplay.graph import Proxy
from tensorplay.graph.proxy import Scope, TracerBase

from tensorplay.graph.experimental.sympy_functions import (
    Mod,
    SymT,
)

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from tensorplay.graph.symbolic_trace import symbolic_trace

from .loops import ops, V
from .ops_handler import DefaultHandler, OpsHandler, WrapperHandler
from .utils import (
    cache_on_self,
    sympy_index_symbol_with_prefix,
    flatten_index,
    decompose_index,
    flatten_index,
    reduction_num_outputs,
    sympy_subs,
)


T = TypeVar("T")


def identity(x):
    return x


class InterpreterShim(FxInterpreter):
    """Runs a captured graph, standing in for the handler doing the work.

    The graph is run against whatever handler is current, so that the body can
    be captured once and then run through any handler -- one that records what
    it touches, or one that writes the code.
    """

    @staticmethod
    @functools.cache
    def _dummy_gm():
        return symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # The parent is given a placeholder module on purpose: building a real
        # one would generate code for it, which is the most expensive part of
        # running a graph and is not what is wanted here.
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__
        self.current_node = None

    def run_node(self, n):
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)


class LightTracer(TracerBase):
    """A tracer that records a body without carrying anything else along.

    A body is not a module and holds no parameters, so the tracing machinery for
    those is not wanted here; what is wanted is a graph and somewhere to put
    the placeholders.
    """

    def __init__(self):
        super().__init__()
        self.graph = FxGraph(tracer_cls=self.__class__)
        self.scope = Scope("", None)
        self.module_stack = {}
        self.node_name_to_scope = {}


class MemoryEntry(NamedTuple):
    """One recorded touch of a buffer: which position, which buffer, and how."""

    index_name: str  # LoopBody.indexing_exprs[index_name]
    buffer_name: str | None
    mode: str | None  # V.ops.store(..., mode=mode)


class MemoryUsageType(enum.Enum):
    """The kinds of touch that are recorded, one for each that generates code."""

    LOAD = enum.auto()
    LOAD_SEED = enum.auto()
    STORE = enum.auto()
    STORE_REDUCTION = enum.auto()
    INDEX_EXPR = enum.auto()
    CHECK_BOUNDS = enum.auto()
    BUCKETIZE = enum.auto()


class LoopBody:
    """A body captured as a graph, with its positions held beside it.

    A body is captured either by tracing what a callable does, or by copying an
    existing body under different loop variables.  The second is far cheaper and
    is what every reordering and re-splitting uses, so it is a path of its own
    rather than something the tracing path happens to handle.
    """

    indexing_exprs: dict
    submodules: dict
    subblocks: dict
    indirect_vars: list
    indirect_var_ranges: dict
    root_block: Any
    memory_usage: dict
    op_counts: collections.Counter

    # defined only temporarily
    indexing_exprs_name: dict

    @staticmethod
    def _wrap_int_to_sympy_integer(expr):
        # A static size can reach a position as a plain Python integer.
        if type(expr) is int:
            return sympy.Integer(expr)
        return expr

    def __init__(
        self,
        fn,
        args,
        var_ranges,
        iter_vars,
        reduce_vars,
        allow_same_symbol_in_index=False,
    ):
        super().__init__()

        _flat_sizes = tuple(var_ranges.values())
        self.sizes = (
            _flat_sizes[: len(iter_vars)],
            _flat_sizes[len(iter_vars) :],
        )

        self.iter_vars = iter_vars
        self.reduce_vars = reduce_vars
        self.var_ranges = var_ranges

        if isinstance(fn, LoopBody):
            self._init_with_copy(fn, args, allow_same_symbol_in_index)
        else:
            self._init_with_tracing(fn, args)

        self.indexing = None

    def get_original_num_rdims(self) -> int:
        if not self.has_partial_accumulate:
            raise AssertionError("Expected has_partial_accumulate to be set")
        node = self.root_block.graph.find_nodes(
            op="call_method", target="partial_accumulate"
        )[0]
        meta = node.args[-1]
        return meta["num_reduction_dims"]

    def extract_pw_from_reduction(self):
        """Take the pointwise part out of a reduction, leaving the accumulate.

        A reduction is a pointwise step followed by a step that folds what the
        pointwise step produced.  Separating them is what lets the first be
        reordered and fused like any other pointwise body, with only the second
        left to run over the reduced loops.
        """

        self.root_block = self.root_block.extract_pw_from_reduction()
        self.has_partial_accumulate = True
        self.iter_vars = self.iter_vars + self.reduce_vars
        self.reduce_vars = []
        self.sizes = (self.sizes[0] + self.sizes[1], tuple())
        return self

    def _init_with_tracing(self, fn, args):
        """Capture a body by tracing what an arbitrary callable does."""

        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.indirect_var_ranges: dict = {}
        self.memory_usage = {t: [] for t in MemoryUsageType}
        self.op_counts = collections.Counter()
        self.root_block = LoopBodyBlock(self, fn, args)  # traces
        self.has_partial_accumulate = bool(
            self.root_block.graph.find_nodes(
                op="call_method", target="partial_accumulate"
            )
        )
        del self.indexing_exprs_name  # not used after _init_with_tracing

    def _init_with_copy(self, other: LoopBody, args, allow_same_symbol_in_index):
        """Capture by copying an existing body under different loop variables.

        Tracing is slow, and reordering or re-splitting a body produces another
        body that differs only in which loop variables it is written in, so
        that case is copied instead of traced again.
        """

        indexing_exprs = other.indexing_from_args(args, allow_same_symbol_in_index)
        self.indexing_exprs = {
            name: self._wrap_int_to_sympy_integer(
                V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
            )
            for name, expr in indexing_exprs.items()
        }
        self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
        self.indirect_vars = other.indirect_vars
        self.indirect_var_ranges = other.indirect_var_ranges
        self.memory_usage = other.memory_usage
        self.op_counts = other.op_counts
        self.root_block = other.root_block.clone(self)
        self.has_partial_accumulate = other.has_partial_accumulate

        submodules = {**other.submodules}
        submodules.pop("get_index")
        self.submodules = {
            "get_index": self.get_index,
            **{k: v.clone(self) for k, v in submodules.items()},
        }

    def has_op(self, name: str):
        return self.op_counts.get(name, 0) > 0

    def merge_loops(self) -> "LoopBody":
        """This body with its loops merged as far as the positions allow.

        The two groups are merged separately, because the loops that are
        iterated and the loops that is reduced are not interchangeable: merging
        across the boundary would put a reduced loop inside an iterated one.
        """

        old_body = self
        old_sizes = self.sizes
        old_iter_vars, old_reduce_vars = old_body.vars
        old_iter_sizes, old_reduce_sizes = old_sizes

        index_exprs = [*old_body.indexing_exprs.values()]

        iter_sizes, iter_reindex, _ = V.graph.sizevars._simplify_loops(
            old_iter_vars,
            old_iter_sizes,
            index_prevent_reordering(index_exprs, old_iter_vars, old_iter_sizes),
        )

        reduce_sizes, reduce_reindex, _ = V.graph.sizevars._simplify_loops(
            old_reduce_vars,
            old_reduce_sizes,
            index_prevent_reordering(index_exprs, old_reduce_vars, old_reduce_sizes),
        )

        if iter_sizes == old_iter_sizes and reduce_sizes == old_reduce_sizes:
            return old_body

        (
            (
                iter_vars,
                reduce_vars,
            ),
            var_ranges,
        ) = dependencies.index_vars_no_squeeze(iter_sizes, reduce_sizes, prefix="p")
        new_body = LoopBody(
            old_body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )

        return new_body

    def with_indexing_exprs(self, replacements: dict) -> "LoopBody":
        """This body with some of its positions replaced."""

        iter_vars, reduce_vars = self.vars
        new_body = LoopBody(
            self,
            (iter_vars, reduce_vars),
            self.var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )
        new_body.indexing_exprs.update(replacements)
        return new_body

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> "LoopBody":
        """A body run over a longer loop, wrapping the position back round.

        A pointwise node that has been unrolled into a longer loop reaches
        elements past the end of what it was written for, and taking the
        position modulo its original extent is what makes those elements repeat
        the ones that were meant, rather than read past the end.
        """

        old_body = self
        old_sizes = self.sizes

        iter_size, reduce_size = old_sizes
        original_range = iter_size[dimension]
        new_iter_size = list(iter_size)
        new_iter_size[dimension] = new_range
        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",
        )

        def new_body(*indices):
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]

            new_iter_idx = list(iter_idx)
            new_iter_idx[dimension] = Mod(iter_idx[dimension], original_range)

            return old_body(new_iter_idx, reduce_idx)

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        # The original prefix is used again so that this can be done more than
        # once over.
        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",
        )
        new_body = LoopBody(
            loop_body, (iter_vars2, reduce_vars2), var_ranges2, iter_vars2, reduce_vars2
        )
        return new_body

    def reindex_iter_loops(self, new_iter_sizes) -> "LoopBody":
        """The same body, with its iterated loops split a different way.

        A shape of one thousand and twenty-four by eight thousand nineteen
        hundred and ninety-two and one of sixty-five thousand five hundred and
        thirty-six by one hundred and twenty-eight cover the same elements, and
        which of the two is the better loop order depends on what is being
        computed.  The old variables are recovered from the new ones by taking
        the flat position apart again.
        """

        old_body = self
        old_iter_sizes = self.sizes[0]
        reduce_sizes = self.sizes[1]

        new_sizes = (list(new_iter_sizes), list(reduce_sizes))

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",
        )

        def new_body(*indices):
            index = [*itertools.chain.from_iterable(indices)]
            new_iter_idx = index[: len(new_iter_sizes)]
            reduce_idx = index[len(new_iter_sizes) :]
            flat = flatten_index(new_iter_idx, new_iter_sizes)
            old_iter_idx = decompose_index(flat, old_iter_sizes)
            return old_body(old_iter_idx, list(reduce_idx))

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",
        )
        return LoopBody(
            loop_body,
            (iter_vars2, reduce_vars2),
            var_ranges2,
            iter_vars2,
            reduce_vars2,
        )

    def reorder_iter_loops(self, new_order) -> "LoopBody":
        """The same body, with its iterated loops in another order."""

        from .ir import same_reorder

        old_body = self
        old_sizes = self.sizes
        if not len(old_sizes[0]) == len(new_order):
            raise AssertionError(
                f"Expected old_sizes[0] length {len(old_sizes[0])} to equal "
                f"new_order length {len(new_order)}"
            )
        reorder_fn = same_reorder(new_order)

        iter_size, reduce_size = old_sizes
        new_iter_size = reorder_fn(iter_size)

        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",
        )

        inverse_order = {b: a for a, b in enumerate(new_order)}
        inverse_order = [inverse_order[i] for i in range(len(new_order))]

        def new_body(*indices):
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]
            iter_idx = [iter_idx[i] for i in inverse_order]
            return old_body(iter_idx, reduce_idx, allow_same_symbol_in_index=True)

        return LoopBody(
            new_body,
            (iter_vars, reduce_vars),
            var_ranges,
            iter_vars,
            reduce_vars,
        )

    @property
    def vars(self):
        if self.iter_vars is None:
            raise AssertionError("iter_vars is None")
        if self.reduce_vars is None:
            raise AssertionError("reduce_vars is None")
        return self.iter_vars, self.reduce_vars

    def indexing_from_args(
        self,
        indices,
        allow_same_symbol_in_index: bool = False,
    ) -> dict:
        """This body's index expressions, written in the variables it was given.

        A body's positions are written in the body's own variables.  What reads
        them wants them in the caller's, because the caller's variables are the
        ones the enclosing nest iterates over.  Substituting one set for the
        other is the whole of the translation.
        """

        index = [*itertools.chain.from_iterable(indices)]
        if len(index) != len(self.var_ranges):
            raise AssertionError(
                f"index length mismatch: {len(index)} positions for "
                f"{len(self.var_ranges)} variables"
            )
        if not allow_same_symbol_in_index and any(
            variable in self.var_ranges for variable in index
        ):
            raise AssertionError(
                f"a position is already one of this body's own variables, so "
                f"substituting would leave it meaning two things: "
                f"{self.var_ranges} and {indices}"
            )
        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        # Imported here so that this module does not have to know what a bound
        # is in order to be used.
        from .bounds import BoundVars

        return BoundVars(self)

    def get_read_expr(self, buffer_name):
        # Reversed to match what the callers of this used to be handed.
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_write_expr(self, buffer_name):
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_read_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in self.memory_usage[MemoryUsageType.LOAD]
        ]

    def get_all_read_expr(self, buffer_name):
        # Reversed to match what the callers of this used to be handed.
        out = []
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def get_write_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in itertools.chain(
                self.memory_usage[MemoryUsageType.STORE],
                self.memory_usage[MemoryUsageType.STORE_REDUCTION],
            )
        ]

    def get_all_write_expr(self, buffer_name):
        out = []
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def is_memory_copy(self) -> bool:
        """Whether this body only reads somewhere and writes somewhere else.

        A body that does nothing else is a copy even when the two places are
        laid out differently, since the work is the same either way and what
        differs is only how the elements are reached.
        """

        return (
            len(self.memory_usage[MemoryUsageType.LOAD]) == 1
            and len(self.memory_usage[MemoryUsageType.STORE]) == 1
            and len(self.submodules) == 1  # get_index
            and self.root_block.contains_only_ops(("load", "store"))
        )

    __repr__ = debug_str

    def add_index_expr(
        self,
        expr,
        mtype: MemoryUsageType,
        buffer_name: str | None = None,
        mode: str | None = None,
    ):
        """Record a position under a name, reusing the name if it is already there.

        Naming a position rather than writing it out at each use is what lets
        two uses of one position be recognised as such, and what lets the
        position be rewritten in one place when the loops change.
        """

        expr = self._wrap_int_to_sympy_integer(expr)
        name = self.indexing_exprs_name.get(expr)
        if not name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        self.memory_usage[mtype].append(MemoryEntry(name, buffer_name, mode))
        return name

    def add_submodule(self, block, prefix):
        """Record a sub-block, which the graph refers to as a module.

        These are not modules in the ordinary sense; a sub-block of the body
        becomes a call to one, and a prefix that is already a name is kept so
        that a sub-block named for what it is does not also get a number.
        """

        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        """A variable standing for a position that came out of the data.

        The extent it runs over is recorded, because the position is not known
        until the data is read and anything that has to reason about the loop
        has to know how far it runs.
        """

        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))
        if var in self.indirect_var_ranges:
            raise AssertionError(f"Indirect var {var} already in indirect_var_ranges")
        self.indirect_vars.append(var)
        self.indirect_var_ranges[var] = size
        return var

    def replace_indirect(self, old, new):
        """Put a different variable in place of one used for indirect indexing."""

        if str(old) == str(new):
            return
        if self.indexing is None:
            raise AssertionError("indexing must be set before replace_indirect")
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        if self.indexing is None:
            raise AssertionError("indexing must be set before get_index")
        return self.indexing[name]

    def indexing_from_args(self, indices, allow_same_symbol_in_index=False):
        """The positions of this body, written in terms of the loops given.

        The two sets of loops must be the same length as the loops the body was
        captured with, or the positions would be written in terms of something
        else.  A variable that appears both as a loop and in a position is
        refused unless the caller says otherwise, because a position that
        mentions the loop it is indexed by is a different position.
        """

        index = [*itertools.chain.from_iterable(indices)]
        if not len(index) == len(self.var_ranges):
            raise AssertionError(f"Index length mismatch: {index} vs {self.var_ranges}")
        if not allow_same_symbol_in_index and not all(
            v not in self.var_ranges for v in index
        ):
            raise AssertionError(
                f"Same symbol found in index: {self.var_ranges=}, {indices=}"
            )

        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices, allow_same_symbol_in_index=False):
        self.indexing = self.indexing_from_args(indices, allow_same_symbol_in_index)
        result = self.root_block()
        self.indexing = None
        return result

    def bind_set_indirect_shim(self, var, size, check, wrap_neg):
        """A call the graph makes to learn which position a variable stands for.

        The value is read out of the data during the run, so it is not known
        when the body is captured; the call is what tells the body afterwards.
        """

        def set_indirect(new_var):
            self.replace_indirect(
                var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
            )

        set_indirect.clone = functools.partial(
            LoopBody.bind_set_indirect_shim,
            var=var,
            size=size,
            check=check,
            wrap_neg=wrap_neg,
        )
        return set_indirect

    def bind_scan_shim(self, combine_fn):
        """A call the graph makes to run a scan over the loops."""

        def shim(dtypes, values):
            return V.ops.scan(dtypes, combine_fn, values)

        shim.clone = functools.partial(LoopBody.bind_scan_shim, combine_fn=combine_fn)
        return shim

    def bind_masked_shim(self, name):
        """A call the graph makes to run the part of the body a mask lets through."""

        def shim(mask, other):
            return V.ops.masked(mask, self.subblocks[name], other)

        shim.clone = functools.partial(LoopBody.bind_masked_shim, name=name)
        return shim


class LoopBodyBlock:
    """One block of a captured body, as a graph.

    Usually a body and a block are one to one, but where an operation applies to
    part of a body only -- a masked operation, say -- the part it does not apply
    to becomes a block of its own.
    """

    def __init__(self, body: LoopBody, fn, args: list):
        self.body = body

        tracer = LightTracer()
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation

        handler: Any = CountOps(
            CaptureIndexing(
                proxy_ops,
                body,
                tracer,
            ),
            body.op_counts,
        )
        if config.constant_and_index_propagation:
            handler = IndexPropagation(
                handler, self.body.var_ranges, self.body.indirect_var_ranges
            )

        with V.set_ops_handler(handler):
            # The extra call is what lets the handler above see what the body
            # returned.
            ops.output(fn(*args))
        self.graph = tracer.graph

    def extract_pw_from_reduction(self):
        """Rewrite this block so the pointwise part stands on its own.

        The fold that the reduction ends with is replaced by a call that keeps
        the intermediate values instead, so that what came before it can be
        looked at as a body in its own right.
        """

        red = None
        store = None
        for node in self.graph.nodes:
            if node.target == "reduction":
                if red:
                    raise AssertionError("Found multiple reduction nodes")
                red = node
            if node.target == "store_reduction":
                if store:
                    raise AssertionError("Found multiple store_reduction nodes")
                store = node
        if not red:
            raise AssertionError("No reduction node found in graph")
        if not store:
            raise AssertionError("No store_reduction node found in graph")
        reduction_type = red.args[-2]
        red_arg = red.args[-1]
        buf = store.args[1]
        ops = store.args[0]

        extra_meta = {
            "num_reduction_dims": len(self.body.reduce_vars),
        }
        with self.graph.inserting_after(store):
            self.graph.call_method(
                "partial_accumulate", (ops, buf, reduction_type, red_arg, extra_meta)
            )
        self.graph.erase_node(store)
        self.graph.erase_node(red)
        return self

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules

        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        code = GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # The trailing deletions are dropped so that the output reads as the
            # body rather than as what the interpreter needs.
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )

    def contains_only_ops(self, allowed_ops) -> bool:
        return all(
            node.target in allowed_ops
            for node in self.graph.find_nodes(op="call_method")
        )

    def clone(self, body: LoopBody):
        """A copy of this block that belongs to another body."""

        copy = LoopBodyBlock.__new__(LoopBodyBlock)
        copy.__dict__.update({**self.__dict__, "body": body})
        return copy


class CountOps(DefaultHandler):
    """Counts what the body does, on its way to the handler that records it."""

    def __init__(self, inner: OpsHandler, counts: collections.Counter):
        self._inner = inner
        self._counts = counts

    def _default(self, name: str, args: tuple, kwargs: dict):
        self._counts[name] += 1
        return getattr(self._inner, name)(*args, **kwargs)


class CaptureIndexing(WrapperHandler):
    """Records each position the body uses, and what it uses it for.

    A position is simplified before it is recorded, so that two ways of writing
    the same position are recorded as one, and then handed to the body as a
    reference rather than as an expression, so that the graph holds positions by
    name.
    """

    name = "CaptureIndexing"

    def __init__(
        self,
        inner: OpsHandler,
        body: LoopBody,
        tracer: "LightTracer",
    ):
        super().__init__(inner)
        self.body = body
        self.tracer = tracer

    def _add_index(self, expr, mtype: MemoryUsageType, **kwargs):
        return self.tracer.create_proxy(
            "call_module",
            "get_index",
            (self.body.add_index_expr(expr, mtype, **kwargs),),
            {},
        )

    def _simplify(self, expr):
        return V.graph.sizevars.simplify_with_ranges(expr, self.body.var_ranges)

    def load(self, name: str, index):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
        return self._inner.load(name, index)

    def load_seed(self, name: str, index: int):
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")
        self.body.add_index_expr(
            sympy.Integer(index), MemoryUsageType.LOAD_SEED, buffer_name=name
        )
        return self._inner.load_seed(name, index)

    def store(self, name, index, value, mode=None):
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE, buffer_name=name, mode=mode
        )
        return self._inner.store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.STORE_REDUCTION, buffer_name=name)
        return self._inner.store_reduction(name, index, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
        num_outputs = reduction_num_outputs(reduction_type)
        if num_outputs > 1:
            return tuple(result[i] for i in range(num_outputs))
        return result

    def index_expr(self, index, dtype):
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.index_expr(index, dtype)

    def value_expr(self, index, dtype):
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.value_expr(index, dtype)

    def check_bounds(self, index, size, lower, upper):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.CHECK_BOUNDS)
        size = self.body._wrap_int_to_sympy_integer(size)
        size = self._add_index(size, MemoryUsageType.CHECK_BOUNDS)
        return self._inner.check_bounds(index, size, lower, upper)

    def bucketize(
        self,
        values: T,
        boundaries,
        boundary_indices: T,
        indexing_dtype,
        right,
        sorter=None,
        sorter_indices: T | None = None,
    ) -> T:
        """The boundaries a bucketize is done against, recorded as positions."""

        boundaries = (
            boundaries[0],
            self._add_index(
                boundaries[1],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[2],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[3],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
        )
        if sorter is not None:
            sorter = (
                sorter[0],
                self._add_index(
                    sorter[1], MemoryUsageType.BUCKETIZE, buffer_name=sorter[0]
                ),
            )

        return self._inner.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )

    def masked(self, mask_proxy, masked_body, other_proxy):
        """Capture the part of the body a mask lets through, as a block of its own.

        What the mask excludes is not thrown away: it becomes a block that the
        graph calls, because what to do about the excluded elements is a
        decision the body still has to be able to make.
        """

        name = self.body.add_submodule(None, "masked_subblock")
        self.body.submodules[name] = self.body.bind_masked_shim(name)
        self.body.subblocks[name] = LoopBodyBlock(self.body, masked_body, [])
        return self.tracer.create_proxy(
            "call_module", name, (mask_proxy, other_proxy), {}
        )

    def scan(
        self,
        dtype_proxy,
        combine_fn,
        value_proxy,
    ):
        """Capture a scan, which is a call the graph makes to run over the loops."""

        shim = self.body.bind_scan_shim(combine_fn)
        name = self.body.add_submodule(shim, "scan")
        result = self.tracer.create_proxy(
            "call_module",
            name,
            (dtype_proxy, value_proxy),
            {},
        )
        # A proxy can be indexed, but some callers want a tuple or a list.
        return tuple(result[i] for i in range(len(value_proxy)))

    def sort(self, dtypes, values, stable, descending):
        result = self._inner.sort(dtypes, values, stable, descending)
        # A proxy can be indexed, but some callers want a tuple or a list.
        return tuple(result[i] for i in range(len(values)))

    def frexp(self, value_proxy):
        result = self._inner.frexp(value_proxy)
        # A proxy can be indexed, but some callers want a tuple or a list.
        return (result[0], result[1])

    def indirect_indexing(self, index_proxy, size, check=True, wrap_neg=True):
        """Carry a position out of the data and into the formulas.

        A position read from a tensor is not known while the body is being
        captured, so a variable stands for it and a call in the graph says what
        the variable turned out to be.
        """

        var = self.body.add_indirect(size)
        set_indirect = self.body.bind_set_indirect_shim(var, size, check, wrap_neg)
        self.tracer.create_proxy(
            "call_module",
            self.body.add_submodule(set_indirect, f"set_{var}"),
            (index_proxy,),
            {},
        )
        return var

    def output(self, *result):
        self.tracer.create_proxy("output", "output", result, {})
