"""What a body reads and writes, and where.

A body is recorded rather than run: every load, store and index it makes is
collected as a dependency naming a buffer, the position it touches, and the
loops that position is written in terms of.  Two bodies that differ only in the
order of their loops produce different sets, which is what makes the order part
of what is being compared.
"""

import abc
import dataclasses
import itertools
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import sympy

import tensorplay as tp

from tensorplay.graph.experimental.sympy_functions import OrderedSet, SymT
from tensorplay.graph.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
)

from .codegen.common import index_prevent_reordering
from .ops_handler import (
    DefaultHandler,
    WrapperHandler as _WrapperHandler,
    KernelFormatterHandler,
    MockHandler,
    V,
)
from .utils import (
    decompose_index,
    get_dtype_size,
    get_free_symbols,
    reduction_num_outputs,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    VarRanges,
)


T = TypeVar("T")

log = logging.getLogger(__name__)
is_indirect = re.compile(r"indirect|tmp").search


class Dep(abc.ABC):
    """One thing a body depends on."""

    name: str
    index: sympy.Expr

    @abc.abstractmethod
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        pass

    @abc.abstractmethod
    def rename(self, renames: dict[str, str]):
        pass

    @abc.abstractmethod
    def get_numel(self) -> sympy.Expr:
        pass

    @abc.abstractmethod
    def numbytes_hint(self) -> int:
        pass

    @abc.abstractmethod
    def numel_hint(self) -> int:
        pass

    @abc.abstractmethod
    def has_unbacked_symbols(self) -> bool:
        pass

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        pass

    def normalize_with_stride_order(self, prefix: str = "t"):
        return self


@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    """One place in a buffer, reached through an index over a domain of loops."""

    name: str
    index: sympy.Expr
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]
    mode: str | None = None

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.index, unbacked_only)
            | get_free_symbols(self.size, unbacked_only)
            | get_free_symbols(self.var_names, unbacked_only)
        )

    def __repr__(self) -> str:
        maybe_mode = ""
        if self.mode is not None:
            maybe_mode = f", {self.mode}"
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges}{maybe_mode})"

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    def decide_loop_order_to_match(self, other: "MemoryDep"):
        """The loop order that would make this access the other one, if any.

        Two accesses reach the same set of places when one is the other with
        its loops permuted, and the permutation is what says the two bodies
        differ only in loop order.  An access that does not mention every loop
        is a broadcast, and a broadcast has a zero stride that says nothing
        about the order, so one of those gives no answer.
        """

        if self.num_vars != other.num_vars:
            raise AssertionError(
                f"expected num_vars to match, got {self.num_vars} and {other.num_vars}"
            )

        # A broadcast has a zero stride, which makes the order impossible to
        # read off, so an access that mentions fewer loops than it has gives
        # no answer.
        if self.num_vars != len(self.index.free_symbols):
            return None
        if other.num_vars != len(other.index.free_symbols):
            return None

        # An empty buffer has no strides worth comparing, and a dimension of
        # one gives several loops the same stride, which ties them.
        if any(s == 0 or s == 1 for s in itertools.chain(self.size, other.size)):
            return None

        self_strides = V.graph.sizevars.stride_hints(self.index, self.var_names)
        other_strides = V.graph.sizevars.stride_hints(other.index, other.var_names)

        # Two strides can come out equal for an index that is not simply a
        # linear one, in which case the order is not readable from the strides
        # either.
        if len(OrderedSet(self_strides)) != len(self_strides) or len(
            OrderedSet(other_strides)
        ) != len(other_strides):
            log.debug(
                "unable to decide loop order. self_dep=%s v.s. other_dep=%s, self_strides=%s v.s. other_strides=%s",
                self,
                other,
                self_strides,
                other_strides,
            )
            return None

        if OrderedSet(self_strides) != OrderedSet(other_strides):
            return None

        stride_to_index = {s: i for i, s in enumerate(self_strides)}
        order = [stride_to_index[s] for s in other_strides]

        if OrderedSet(order) != OrderedSet(range(self.num_vars)):
            raise AssertionError(
                f"expected order to be a permutation of range({self.num_vars}), got {order}"
            )
        return order

    def get_offset(self) -> sympy.Expr:
        """The first place this access reaches, which is where every variable is zero."""

        return sympy_subs(self.index, dict.fromkeys(self.var_names, 0))

    def normalize(self) -> "MemoryDep":
        """The same access with its loops merged, without moving them."""

        return MemoryDep(
            self.name,
            *_RecordLoadStoreInner._normalize(self.index, self.ranges),
            self.mode,
        )

    def normalize_with_ranges(
        self,
        var_names: tuple[sympy.Symbol, ...],
        sizes: tuple[sympy.Expr, ...],
    ):
        """The same access, written over a different domain of loops.

        Both domains list their dimensions outermost first and correspond by
        linearized order, and nothing is assumed about how the access reaches
        memory.  ``None`` comes back when the correspondence cannot be worked
        out, which happens for an access through indirect indexes and for one
        whose extents do not multiply to the same number.
        """

        if len(var_names) != len(sizes):
            raise AssertionError("var_names and sizes must have equal length")
        if self.is_indirect():
            return None
        if not self.var_names:
            # A position that does not move is the same in every domain.
            return MemoryDep(self.name, self.index, var_names, sizes, self.mode)
        if not V.graph.sizevars.statically_known_equals(
            sympy_product(self.size), sympy_product(sizes)
        ):
            return None

        from .codegen.simd import CantSplit, SIMDKernel

        def split_values(
            *new_ranges: Sequence[sympy.Expr],
        ) -> list[list[sympy.Expr]]:
            return [
                decompose_index(value, ranges)
                for value, ranges in zip(var_names, new_ranges, strict=True)
            ]

        try:
            (source_indices,) = SIMDKernel.map_kernel_groups_to_node_sizes(
                sizes, (self.size,), split_values
            )
        except CantSplit:
            return None
        replacements = dict(zip(self.var_names, source_indices, strict=True))
        var_ranges = dict(zip(var_names, sizes, strict=True))
        index = V.graph.sizevars.simplify_with_ranges(
            sympy_subs(self.index, replacements), var_ranges
        )
        return MemoryDep(self.name, index, var_names, sizes, self.mode)

    def normalize_with_stride_order(self, prefix: str = "t") -> "MemoryDep":
        """The same access with its loops in the order the strides suggest.

        Two accesses that are not equal may still be equal once both are put in
        stride order, and that is what tells apart a difference in loop order
        from a difference in what was computed.
        """

        from .ir import same_reorder

        strides = V.graph.sizevars.stride_hints(self.index, self.var_names)

        # The outermost loop is the one with the largest stride.
        order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        stride_reorder = same_reorder(order)
        sizes = self.size
        var_names = self.var_names

        new_reordered_sizes = stride_reorder(sizes)
        new_reordered_var_names = stride_reorder(var_names)

        new_simplified_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            new_reordered_var_names,
            new_reordered_sizes,
            index_prevent_reordering(
                [self.index], new_reordered_var_names, new_reordered_sizes
            ),
        )

        # Fresh symbols under the given prefix, so that this access cannot be
        # confused with one that was never reordered.
        var_ranges, add_var = var_builder(prefix)
        replacement = dict(
            zip(
                new_reordered_var_names,
                reindex([add_var(x) for x in new_simplified_sizes]),
            )
        )
        new_index = sympy_subs(sympy.expand(self.index), replacement)

        out = MemoryDep(
            self.name, new_index, tuple(var_ranges.keys()), tuple(var_ranges.values())
        )
        return out

    @property
    def ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        """What each loop variable of this access runs over."""

        return dict(zip(self.var_names, self.size))

    def simplify_with_ranges(self) -> "MemoryDep":
        return MemoryDep(
            name=self.name,
            index=V.graph.sizevars.simplify_with_ranges(self.index, self.ranges),
            var_names=self.var_names,
            size=self.size,
            mode=self.mode,
        )

    def get_numel(self) -> sympy.Expr:
        """How many elements this access reaches.

        Only the loops the index actually mentions are counted, because a loop
        it does not mention moves the access nowhere.
        """

        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars: OrderedSet = OrderedSet(self.index.free_symbols)
            numel = sympy.S.One
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(
                renames[self.name],
                self.index,
                var_names=self.var_names,
                size=self.size,
                mode=self.mode,
            )
        return self

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(
                self.get_numel(), fallback=0
            ) * get_dtype_size(V.graph.get_dtype(self.name))
        except NotImplementedError:  # NoneLayout
            return 0

    def numel_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(self.get_numel(), fallback=0)
        except NotImplementedError:  # NoneLayout
            return 0

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        if isinstance(self.index, sympy.Integer):
            return True
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def stride1_for_last_dim(self, result_for_complex_expression: bool = True) -> bool:
        """Whether the innermost loop's stride is one.

        An innermost stride of one is what lets a run of elements be moved at
        once, so an access where it is anything larger is worth knowing about
        even when the index is too tangled to say more than that.
        """

        if len(self.var_names) == 0:
            return True

        terms = self.index.args if isinstance(self.index, sympy.Add) else [self.index]

        last_sym = self.var_names[-1]
        for term in terms:
            if term == last_sym:
                return True

            # A stride of more than one on the innermost loop means the run is
            # broken up, which is the case worth refusing.
            if (
                isinstance(term, sympy.Mul)
                and len(term.args) == 2
                and term.args[1] == last_sym
                and isinstance(term.args[0], (int, sympy.Integer))
                and term.args[0] > 1
            ):
                return False

        return result_for_complex_expression

    def is_scalar(self) -> bool:
        """Whether this access is one position rather than a run of them."""

        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and not self.is_indirect()
        return isinstance(self.index, (int, sympy.Integer))

    def is_indirect(self) -> bool:
        return any(is_indirect(v.name) for v in self.index.free_symbols)


@dataclasses.dataclass(frozen=True)
class StarDep(Dep):
    """A dependence on a whole buffer, wherever in it the access lands."""

    name: str
    mode: str | None = None

    @property
    def index(self) -> sympy.Expr:
        raise NotImplementedError("StarDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name], self.mode)
        return self

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(
                self.get_numel(), fallback=0
            ) * get_dtype_size(V.graph.get_dtype(self.name))
        except NotImplementedError:
            return 0

    def numel_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(self.get_numel(), fallback=0)
        except NotImplementedError:
            return 0

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return False

    def is_indirect(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class WeakDep(Dep):
    """A dependence on a buffer only to say an order, not to read a value.

    Two bodies that touch the same buffer have to run in a definite order, and
    this is how that order is recorded without saying which places are involved.
    It is also used to hold two things apart that would otherwise be free to
    swap, and a dependence invented only for that purpose is marked so, because
    such a dependence is only worth keeping if whatever forced it is still
    there.
    """

    name: str
    mutating_buf: str
    is_fake: bool = False

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @property
    def index(self) -> sympy.Expr:
        raise NotImplementedError("WeakDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return sympy.S.One

    def rename(self, renames: dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name], self.mutating_buf, self.is_fake)
        return self

    def numbytes_hint(self) -> int:
        return 1  # Purely inserted for ordering, not an actual dep

    def numel_hint(self) -> int:
        return 1  # Purely inserted for ordering, not an actual dep

    def has_unbacked_symbols(self) -> bool:
        return False

    def is_contiguous(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class IndexExprDep:
    """A position computed rather than read or written."""

    index: sympy.Expr
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]


@dataclasses.dataclass
class ReadWrites:
    """Everything one body reads, writes, and computes a position for."""

    reads: OrderedSet
    writes: OrderedSet
    index_exprs: OrderedSet
    range_vars: list[sympy.Expr] | None = None
    var_ranges: VarRanges | None = None

    def rename(self, renames: dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            OrderedSet(dep.rename(renames) for dep in self.reads),
            OrderedSet(dep.rename(renames) for dep in self.writes),
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def with_read(self, dep):
        """This, plus a dependence on a read that is not a place in a buffer.

        A buffer that is read is a position, but a buffer whose whole contents
        are needed says nothing about where in it, so only those two kinds may
        be added this way.
        """

        if not isinstance(dep, (WeakDep, StarDep, OrderedSet)):
            raise AssertionError(
                f"expected WeakDep, StarDep, or OrderedSet, got {type(dep)}"
            )
        if not isinstance(dep, OrderedSet):
            dep = OrderedSet([dep])
        return ReadWrites(
            OrderedSet.union(self.reads, dep),
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def merge(self, other: "ReadWrites") -> "ReadWrites":
        """Both, with anything written by one no longer counted as read by it.

        A buffer that is written is not also read: the write is what leaves it
        holding something, and counting the read as well would make the buffer
        look like it had to be there beforehand.
        """

        reads = OrderedSet.union(self.reads, other.reads)
        writes = OrderedSet.union(self.writes, other.writes)
        index_exprs = OrderedSet.union(self.index_exprs, other.index_exprs)
        return ReadWrites(reads - writes, writes, index_exprs)

    @staticmethod
    def merge_list(read_writes: list["ReadWrites"]) -> "ReadWrites":
        all_writes = OrderedSet.union(*[rw.writes for rw in read_writes])
        all_reads = OrderedSet.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = OrderedSet.union(*[rw.index_exprs for rw in read_writes])
        return ReadWrites(all_reads, all_writes, all_index_exprs)

    def remove_reads(self, rem_reads: OrderedSet) -> "ReadWrites":
        return ReadWrites(
            self.reads - rem_reads,
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def reads_and_writes(self) -> Iterable[Dep]:
        return itertools.chain(self.reads, self.writes)

    def buffer_names(self, ignore_integer_index: bool = True) -> OrderedSet:
        """The names of the buffers this body touches.

        A position that is a plain number is a seed rather than a place in a
        buffer, so it is left out unless the caller asks for it.
        """

        names: OrderedSet = OrderedSet()
        for dep in self.reads_and_writes():
            if not isinstance(dep, MemoryDep):
                continue
            if not ignore_integer_index or not isinstance(
                dep.index, (int, sympy.Integer)
            ):
                names.add(dep.name)
        return names

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet:
        result: OrderedSet = OrderedSet()

        for dep in self.reads_and_writes():
            result |= dep.get_free_symbol_uses(unbacked_only)
        return result


def var_builder(prefix: str):
    """A fresh loop variable per extent, and the record of what each runs over.

    The names come from one counter so that two calls never produce the same
    name, and the record is what says which extent belongs to which name.
    """

    cnt = itertools.count()
    var_ranges: VarRanges = {}

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        v = sympy_index_symbol(f"{prefix}{next(cnt)}")
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def canonicalization_prefix() -> str:
    """The prefix the symbols a normalized access gets are named under."""

    return "c"


class _RecordLoadStoreInner(MockHandler):
    """Where the loads, stores and computed positions of one body are recorded.

    A position is put into the simplest form its loops allow before it is
    recorded, so that two bodies reaching the same places are recorded the same
    way whatever order they happened to compute the position in.
    """

    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        super().__init__()
        self._reads: OrderedSet = OrderedSet()
        self._writes: OrderedSet = OrderedSet()
        self._index_exprs: OrderedSet = OrderedSet()
        self._var_ranges: VarRanges = var_ranges
        self._should_normalize: bool = normalize

    @staticmethod
    def drop_unused_symbols(
        index,
        var_names: list,
        sizes: list,
    ) -> None:
        """Let go of the loops at the end that the position never mentions.

        A reduction keeps the loop it reduces over among its extents and
        whatever comes after it does not, so those trailing loops are dropped
        rather than left to make two records of one access differ in how many
        loops they have.
        """

        if not isinstance(index, sympy.Expr):
            # An index can be a plain number.
            return
        free = index.free_symbols
        while var_names and var_names[-1] not in free:
            var_names.pop()
            sizes.pop()

    @classmethod
    def _normalize(cls, index, var_ranges):
        """The position with its loops merged and renumbered from scratch.

        Merging the loops is not always enough on its own, because one
        position's form can get in the way of another's.  The position is
        expanded and the loops are then numbered afresh, since what came back
        as d0, d1, d2 may be only d0, d2 -- which would otherwise look like a
        different access.
        """

        index_vars = [*var_ranges.keys()]
        sizes = tuple(var_ranges.values())
        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars,
            sizes,
            index_prevent_reordering([index], index_vars, sizes),
        )

        new_vars, add_var = var_builder(canonicalization_prefix())
        replacement = dict(
            zip(index_vars, reindex([add_var(x) for x in new_sizes]))
        )
        index = sympy_subs(sympy.expand(index), replacement)

        new_vars = [*new_vars.keys()]
        new_sizes = [*new_sizes]
        cls.drop_unused_symbols(index, new_vars, new_sizes)
        return index, tuple(new_vars), tuple(new_sizes)

    def canonicalize(self, index):
        if not self._should_normalize:
            sizes = [V.graph.sizevars.simplify(x) for x in self._var_ranges.values()]
            var_names = [k for k, v in zip(self._var_ranges.keys(), sizes) if v != 1]
            sizes = [v for v in sizes if v != 1]

            self.drop_unused_symbols(index, var_names, sizes)

            return index, tuple(var_names), tuple(sizes)
        var_ranges = {k: V.graph.sizevars.simplify(v) for k, v in self._var_ranges.items()}
        return self._normalize(index, var_ranges)

    def load(self, name: str, index) -> None:
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))

    def load_seed(self, name: str, index) -> None:
        if not isinstance(index, int):
            raise AssertionError(f"expected index to be int, got {type(index)}")
        self.load(name, sympy.Integer(index))

    def store(self, name: str, index, value, mode: str | None = None) -> None:
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))

    def store_reduction(self, name: str, index, value) -> None:
        self.store(name, index, f"store_reduction({value})")

    def index_expr(self, index, dtype) -> None:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))

    def value_expr(self, index, dtype) -> None:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right,
        sorter=None,
        sorter_indices=None,
    ) -> None:
        """The boundaries a bucketize reads, and the order it reads them by."""

        self._reads.add(StarDep(boundaries[0]))
        if sorter is not None:
            self._reads.add(StarDep(sorter[0]))


class RecordLoadStore(KernelFormatterHandler):
    """Records a body as it runs, standing in for the code that would be written."""

    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        super().__init__(parent_handler=parent_handler)


def index_vars_no_squeeze(*argsizes, prefix: str):
    """A loop variable per extent of each shape, with nothing dropped."""

    var_ranges, add_var = var_builder(prefix)
    args = [list(map(add_var, size)) for size in argsizes]
    return args, var_ranges


def index_vars_squeeze(*argsizes, prefix: str = "d"):
    """A loop variable per extent, with the extents of one dropped.

    An extent of one makes a loop that can only run once, which says nothing
    about the order the others run in, so it is left out.
    """

    from .ir import SqueezeView

    var_ranges, add_var = var_builder(prefix)
    args = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges


def extract_read_writes(
    fn: Callable,
    *argsizes,
    normalize: bool = False,
    prefix: str = "d",
    hidden_args: Sequence[list] = (),
) -> ReadWrites:
    """What a body reads and writes, with one loop variable per extent.

    A body that is already a recorded loop body is asked directly; anything
    else is run once with a handler standing in for the code, which is slower
    but works whatever the body is.
    """

    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)

    from .loop_body import LoopBody

    if isinstance(fn, LoopBody):
        from .loop_body import extract_loop_body_with_args

        inner = extract_loop_body_with_args(
            fn,
            [*args, *hidden_args],
            var_ranges,
            normalize,
        )
    else:
        # Running the body with a handler in place of the code.
        rw = RecordLoadStore(var_ranges, normalize=normalize)
        with V.set_ops_handler(rw):
            fn(*args, *hidden_args)
        inner = rw.parent_handler

    if normalize:
        # Normalizing can change how many loops there are.
        range_vars = []
    else:
        range_vars = [*itertools.chain.from_iterable(args)]

    return ReadWrites(
        OrderedSet(inner._reads),
        OrderedSet(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
    )


class FreeSymbolsOpsHandler(DefaultHandler):
    """Collects every shape symbol a body mentions, and where it mentions it."""

    def __init__(self):
        self.symbols: OrderedSet = OrderedSet()

    def free_symbols(self, expr, hint: int | None = None):
        result = (
            OrderedSet(free_symbols(expr, hint))
            if isinstance(expr, sympy.Expr)
            else OrderedSet(*expr)
        )
        self.symbols |= result
        return result

    def handle(self, expr):
        if isinstance(expr, sympy.Expr):
            self.free_symbols(expr)
        elif isinstance(expr, (list, tuple)):
            for e in expr:
                self.handle(e)
        return expr


def extract_free_symbols(*args) -> OrderedSet:
    """Every shape symbol mentioned anywhere inside these arguments."""

    handler = FreeSymbolsOpsHandler()
    V.op.handle(args)
    return handler.symbols


class SymbolUsageCollectorOpsHandler(_WrapperHandler):
    """Which operations of a body mention a given shape.

    Knowing that a shape reaches a body only through one operation is what lets
    a decision be made about that shape alone -- whether it has to be materialized
    for that one use, say -- without asking what the rest of the body does.
    """

    usages: OrderedSet

    def __init__(self, symbol) -> None:
        super().__init__(MockHandler())
        self.symbol = symbol
        self.usages = OrderedSet()

    def _default(self, name: str, args: tuple, kwargs: dict):
        used_here = self.symbol in args or self.symbol in kwargs.values()
        if used_here:
            self.usages.add(name)
        return getattr(self._inner, name)(*args, **kwargs)
