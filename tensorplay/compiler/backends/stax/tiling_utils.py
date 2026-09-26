"""Deciding which pieces of work can share a kernel, and how to divide it up.

Two pieces of work can run together when the loops they each walk can be made
to be the same loops.  A piece that walks two elements next to each other is
worth fusing with another that does the same, since the second reads what the
first has in hand; a piece that strides through memory is not, since fusing
would mean making both stride.

Once it is decided that pieces run together, something still has to say how many
programs each runs and over which axes.  That is a question about the machine
rather than about the arithmetic, so it is answered here and handed on.

What this module works out, in order: the axes a piece's reads and writes are
expressed in once everything is described in one set of axes; which of those
axes make consecutive elements land next to each other in memory, and how much
that is worth; and, where some access does not, whether re-dividing one axis
would fix it.
"""

from __future__ import annotations

import dataclasses
import itertools
import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload, Union

import sympy

from tensorplay.graph.experimental.sympy_functions import (
    FloorDiv,
    Identity,
    ModularIndexing,
    SymT,
    symbol_is_type,
)
from tensorplay.graph.experimental.sympy_solve import try_solve

from .dependencies import index_vars_no_squeeze, ReadWrites
from .ir import OrderedSet
from .loops import V
from .utils import sympy_product, sympy_subs

import logging

loop_tiling_log = logging.getLogger(f"{__name__}.loop_tiling")

#: The smallest piece worth re-dividing an axis into.  Below this the extra
#: program is not worth its own launch.
MIN_TILING_BLOCK = 8


class CantSplit(Exception):
    """Raised when an axis cannot be divided the way a division was asked for.

    How much of the axis each program walks has to divide it exactly, or the
    last program would walk past the end.  Where the shapes are known this is
    decided while writing the code; where they are only known at run time it may
    turn out not to hold, and this is what says so.
    """

    def __init__(self, expr, remaining):
        super().__init__()
        self.expr = expr
        self.remaining = remaining

    def __str__(self):
        return f"{self.expr} not divisible by {self.remaining}"


def _split_iteration_ranges(groups, lengths):
    """Divide each group of axes so that the pieces line up with the lengths.

    A group of axes is walked as one flat range of positions, and a length says
    how that range has to be broken up.  A length that does not divide the group
    cannot be honoured, which is what :class:`CantSplit` says.  What comes back
    is the axes each program walks, and for each of the lengths how to recover
    its position from those -- the mapping back, which is what lets a piece
    written in one set of axes be run in another.
    """

    if all(len(length) == 0 for length in lengths):
        return [[] for group in groups], []

    sv = V.graph.sizevars
    new_ranges: list = [[] for _ in groups]
    remaining = [sv.simplify(g) for g in groups]
    var_count = itertools.count()

    def add_range(i: int, expr):
        expr = sv.simplify(expr)
        if not sv.statically_known_multiple_of(remaining[i], expr):
            raise CantSplit(remaining[i], expr)
        # The last one out is the one that divides, so that the result is exact.
        remaining[i] = FloorDiv(remaining[i], expr)
        new_ranges[i].append(expr)
        return next(var_count)

    def make_combined(sizes: list, idxs: list):
        """Build the expression that turns the new axes back into the old one.

        Walking an axis of length ``s1`` and then one of length ``s2`` visits
        the positions the two-level walk would, so the position is recovered by
        mixing the two from the outside in.
        """

        if len(idxs) != len(sizes) + 1:
            raise AssertionError(
                f"expected len(idxs) == len(sizes) + 1, "
                f"got {len(idxs)} and {len(sizes) + 1}"
            )

        def getter(flat_vars: list):
            expr = flat_vars[idxs[0]]
            for s, idx in zip(sizes, idxs[1:]):
                expr = s * expr + flat_vars[idx]
            return expr

        return getter

    return_getters_groups = []
    current_group = 0
    for length_group in lengths:
        return_getters = []
        for size in length_group:
            if sv.statically_known_equals(size, 1):
                return_getters.append(lambda _: sympy.S.Zero)
                continue

            while current_group < len(remaining) and sv.statically_known_equals(
                remaining[current_group],
                1,
            ):
                current_group += 1

            # A piece whose flat walk spans three consecutive groups is taken
            # apart across all three, which is what a batched product or a
            # nested reduction needs.
            if current_group + 2 < len(remaining) and sv.statically_known_gt(
                size, remaining[current_group] * remaining[current_group + 1]
            ):
                if not sv.statically_known_multiple_of(
                    size, remaining[current_group] * remaining[current_group + 1]
                ):
                    raise CantSplit(
                        size,
                        remaining[current_group] * remaining[current_group + 1],
                    )

                size1 = remaining[current_group]
                size2 = remaining[current_group + 1]
                size3 = FloorDiv(size, size1 * size2)
                return_getters.append(
                    make_combined(
                        [size2, size3],
                        [
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                            add_range(current_group + 2, size3),
                        ],
                    )
                )

            # Two groups: the length is taken apart across both of them.
            elif current_group + 1 < len(remaining) and (
                sv.statically_known_gt(size, remaining[current_group])
                # Whether one is greater than another cannot always be decided
                # for shapes that are only known at run time, since either could
                # be zero.  A length that is a multiple of a group and at least
                # one whole group long must be greater than that group, since
                # the multiple cannot be zero.
                or sv.statically_known_gt(FloorDiv(size, remaining[current_group]), 1)
            ):
                if not sv.statically_known_multiple_of(
                    size, remaining[current_group]
                ):
                    raise CantSplit(size, remaining[current_group])

                size1 = remaining[current_group]
                size2 = FloorDiv(size, remaining[current_group])
                return_getters.append(
                    make_combined(
                        [size2],
                        [
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        ],
                    )
                )
            else:
                if current_group >= len(remaining):
                    raise CantSplit(size, 0)
                return_getters.append(operator.itemgetter(add_range(current_group, size)))
        return_getters_groups.append(return_getters)

    if not all(V.graph.sizevars.guarding_hint_or_throw(s) == 1 for s in remaining):
        # Something is left over, which means the piece's walk does not tile
        # onto the groups at all -- a piece with fewer rows than the group it is
        # being folded into, for instance.  That is not a shape to be worked
        # around: this fusion is not possible, and the caller falls back to
        # writing the pieces separately.
        raise CantSplit(remaining, lengths)
    return new_ranges, return_getters_groups


def prepare_split_iteration_lengths(groups, lengths, reduction_numel=sympy.S.One):
    """Fill in the reduced part of the lengths when it was not given.

    A piece that reduces is said in terms of what it walks without the reduced
    axis, since that is what its output has.  Here the missing part is put back
    from what is known about how much is being reduced.
    """

    sizevars = V.graph.sizevars
    if len(lengths[1]) == 0 and (
        not sizevars.statically_known_equals(reduction_numel, sympy.S.One)
        and sizevars.statically_known_equals(
            sympy_product(groups),
            sympy_product(lengths[0]) * reduction_numel,
        )
    ):
        return (lengths[0], [reduction_numel])
    return lengths


def solve_for_zero(expr):
    """For an expression with one free symbol, the value that makes it zero.

    Knowing what one axis would have to be for an access to be all at one place
    is what tells us whether re-dividing that axis could make the access
    efficient.  An expression that has no such value -- one that is already a
    constant, or one that a whole number of steps would never bring to zero --
    is not one that can be fixed this way.
    """

    if expr.is_constant():
        return None
    elif isinstance(expr, FloorDiv):
        return None

    if len(expr.free_symbols) != 1:
        raise AssertionError(
            f"expected exactly 1 free symbol, got {len(expr.free_symbols)}"
        )
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out[1]


def solve_for_tiling(expr):
    """For an expression with one free symbol, what to divide that axis by.

    Dividing an axis by some factor and adding an axis of that factor means a
    position is now both a coarse and a fine one, so what was ``x`` becomes
    ``x * y``.  Two neighbouring fine positions then differ by one, which is
    what makes the access efficient.  So what is looked for is a factor making
    the expression one, and the expression is checked at both ends of the range
    since a division that is only approximately invertible will pass one and
    fail the other.
    """

    if len(expr.free_symbols) != 1:
        return None

    free_symbol = next(iter(expr.free_symbols))

    def _solve_simple_expr(expr):
        if expr.has(ModularIndexing) or expr.has(FloorDiv):
            # The division is only approximately undone, so what is left cannot
            # be solved for.
            return None
        if len(expr.free_symbols) != 1:
            return None

        out = try_solve(sympy.Eq(expr, 1), free_symbol)
        if not out or not out[1].is_constant():
            return None
        return out[1]

    # Solving is limited once a division is involved but works well without.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        return _solve_simple_expr(expr)

    required_values = []
    eq_1_expressions = []

    # Terms that can be made zero and terms that can be made one are collected
    # separately, since they are solved for differently.  Every product has to
    # be made zero, or the whole thing cannot be.
    for arg in sympy.Add.make_args(expr):
        if isinstance(arg, sympy.Mul):
            seen = False
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None:
                    continue

                if not out.is_constant():
                    raise AssertionError(f"expected constant, got {out}")
                seen = True
                required_values.append(out)

            if not seen:
                return None
        else:
            eq_1_expressions.append(arg)

    if not eq_1_expressions:
        return None

    eq_1_expr = sum(eq_1_expressions)

    def indexing_div_rep(x, y, z=None):
        return x / y

    # The divisions are only approximately undone, so what comes out is worked
    # out from bottom up and swept twice: the second sweep catches the divisions
    # that undoing reintroduces.  Whatever is left makes the check below fail.
    eq_1_expr_simplified = eq_1_expr
    for _ in range(2):
        eq_1_expr_simplified = eq_1_expr_simplified.replace(
            ModularIndexing, indexing_div_rep, simultaneous=False
        ).replace(FloorDiv, indexing_div_rep, simultaneous=False)

    out = _solve_simple_expr(eq_1_expr_simplified)

    # Since the divisions were only approximately undone, the answer is checked
    # against the expression as it was.
    if not out or sympy_subs(eq_1_expr, {free_symbol: out}) != 1:
        return None

    required_values.append(out)

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_broadcast_var(index, var_ranges):
    """Which axis this access is the same for, whatever that axis is.

    Many axes reading one place is not efficient in the sense of neighbouring
    elements being neighbours, but it does keep the access inside one cache
    line, which is most of what that is worth.  So it is found and scored as if
    it were efficient.
    """

    # A rough answer is had by evaluating at one and at zero.
    variables: dict = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        if v not in index.free_symbols:
            continue

        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        # The same place however far this axis goes is what broadcasting is.
        if new_val == zero_index:
            return v
        variables[v] = 0

    return None


def find_coalesced_var(index, var_ranges):
    """Which axis this access steps through one element at a time.

    A symbol standing on its own as a whole term is the easy case.  Failing
    that, an axis is tried by checking whether stepping it moves the address by
    one, both from the start and from one place in, since some expressions only
    look that way near the beginning.
    """

    top_level_terms = sympy.Add.make_args(index)
    for v in var_ranges:
        if v in top_level_terms:
            return v

    variables: dict = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges:
        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        if new_val - zero_index == 1:
            variables[v] = 2
            # Some expressions step by one at the start and not afterwards.
            if (sympy_subs(index, variables) - new_val) == 1:
                return v
        variables[v] = 0

    return None


def has_indirect_access(memory_expr) -> bool:
    """Whether this access is at a position that comes from the data.

    Such an access cannot be said to be efficient or not, since the positions
    are not known until the data is there.
    """

    return any(symbol_is_type(s, SymT.INDIRECT) for s in memory_expr.free_symbols)


@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """The reads and writes of several pieces, all in one set of axes.

    Pieces written in different axes have to be described in the same axes
    before anything can be said about whether they fit together.  This is that
    common description: which axes, which of them are reduced, and what each
    buffer is read or written at in terms of those.
    """

    index_vars: OrderedSet
    reduce_vars: OrderedSet
    reads: dict
    writes: dict
    var_ranges: dict


@dataclasses.dataclass(frozen=True)
class _FusedNodeView:
    """Several pieces looked at as one, for the sake of asking about them together."""

    nodes: Sequence
    read_writes: ReadWrites
    group: Any

    def get_nodes(self) -> Sequence:
        return self.nodes

    def get_buffer_names(self) -> OrderedSet:
        return OrderedSet.union(*(node.get_buffer_names() for node in self.nodes))

    def get_operation_names(self) -> OrderedSet:
        return OrderedSet(node.get_name() for node in self.nodes)


@overload
def get_pw_red_splits(
    n,
    pointwise_numel,
    red_numel,
    none_if_not_divisible: Literal[True],
) -> tuple | None: ...


@overload
def get_pw_red_splits(
    n,
    pointwise_numel,
    red_numel,
    none_if_not_divisible: Literal[False] = False,
) -> tuple: ...


def get_pw_red_splits(n, pointwise_numel, red_numel, none_if_not_divisible=False):
    """Split a piece's axes into the part walked once and the part reduced.

    A piece walks some axes once and reduces over others, and the two are often
    written as one flat set of axes.  Here they are told apart: the boundary is
    where the product of the axes from the end reaches how much is reduced.
    Where the shapes are numbers the boundary is exact; where they are not, the
    caller may ask for nothing rather than a guess.
    """

    if n.is_reduction() or V.graph.sizevars.statically_known_equals(
        sympy_product(n._body.sizes[0]), pointwise_numel
    ):
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )

    if get_hint(sympy_product(n._body.sizes[0])) != get_hint(
        pointwise_numel * red_numel
    ):
        raise AssertionError(
            "expected pointwise sizes to match pointwise_numel * red_numel"
        )
    i = len(n._body.sizes[0]) - 1
    prod = 1
    while i >= 0:
        prod *= n._body.sizes[0][i]
        if prod == red_numel:
            break
        i -= 1

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)

    if none_if_not_divisible:
        return None
    else:
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )


class NodeSplitGetter:
    """A division of the axes that every piece in a group can be run in.

    Each piece may be willing to be run in a division, or not.  What is looked
    for is one that all of them accept, and where none is acceptable the widest
    one is tried first, since more axes is more freedom about how the work is
    divided among programs.
    """

    def __init__(self, node) -> None:
        self.node = node
        self.pointwise_numel = node.group[1][0]
        self.red_numel = node.group[1][1]

        self.pw_split_options: dict = defaultdict(OrderedSet)
        self.red_split_options: dict = defaultdict(OrderedSet)

        self.reduction_split: tuple = ()
        self.all_node_sizes: OrderedSet = OrderedSet()

        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, _scheduler_module().SchedulerNode):
                continue

            # A piece whose axes cannot be divided that way is not a candidate,
            # but its size still has to be checked.
            maybe_splits = get_pw_red_splits(
                n, self.pointwise_numel, self.red_numel, none_if_not_divisible=True
            )
            if maybe_splits is None:
                self.all_node_sizes.add(n._body.sizes)
                continue

            (_, n_pw_splits), (_, n_red_splits) = maybe_splits

            n_pw_splits, n_red_splits = prepare_split_iteration_lengths(
                fused_group, (n_pw_splits, n_red_splits), self.red_numel
            )

            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))
            self.red_split_options[len(n_red_splits)].add(tuple(n_red_splits))

            if n_red_splits != ():
                self.reduction_split = (sympy_product(n_red_splits),)

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        self.seen_pw_splits: OrderedSet = OrderedSet()

    def get_node_splits(self) -> tuple:
        """A division of the axes that every piece here accepts."""

        if len(self.all_node_sizes) == 1:
            return next(iter(self.all_node_sizes))

        if len(self.pw_split_options) == 0:
            return ((self.pointwise_numel,), (self.red_numel,))

        max_pw_split = max(self.pw_split_options.keys())
        max_red_split = max(self.red_split_options.keys())

        def add_combined_split_options(split_options: dict, curr_length: int) -> None:
            """Offer the divisions with two neighbouring axes joined together.

            Joining two axes is another way of dividing the same work, and it
            gives divisions that were not offered to begin with.
            """

            for split in split_options[curr_length]:
                for i in range(len(split) - 1):
                    new_split = tuple(
                        split[0:i] + (sympy_product(split[i : i + 2]),) + split[i + 2 :]
                    )
                    split_options[len(new_split)].add(new_split)

        max_total_splits = max_pw_split + max_red_split
        for curr_iter, total_splits in enumerate(range(max_total_splits, 0, -1)):
            for pw_split_len in range(total_splits, 0, -1):
                for pw_split in self.pw_split_options[pw_split_len]:
                    for red_split in self.red_split_options[
                        total_splits - pw_split_len
                    ]:
                        if out := self.try_split(pw_split, red_split):
                            return out

            add_combined_split_options(self.pw_split_options, max_pw_split - curr_iter)
            add_combined_split_options(
                self.red_split_options, max_red_split - curr_iter
            )

        return ((self.pointwise_numel,), (self.red_numel,))

    def try_split(self, pw, red):
        """Whether this division works for every piece, and a longer one if not.

        A division that has to cut one axis in two produces a finer division
        than was asked for.  Where that happens the finer one is tried instead,
        since it is what the pieces can actually be run in.
        """

        if pw in self.seen_pw_splits:
            return None
        self.seen_pw_splits.add(pw)

        for n_pw, n_red in self.all_node_sizes:
            try:
                groups = pw + red
                lengths = (n_pw, n_red)
                splits, getters = _split_iteration_ranges(groups, lengths)
            except CantSplit:
                return None

            if len(getters) != 2:
                raise AssertionError(f"expected 2 getters, got {len(getters)}")
            pw_group_splits = splits[: len(pw)]
            flattened_pw_splits = tuple(itertools.chain.from_iterable(pw_group_splits))
            if flattened_pw_splits != pw:
                if out := self.try_split(flattened_pw_splits, red):
                    return out

        return pw, red


def apply_var_mapping(
    iter_vars: list,
    red_vars: list,
    norm_pw_vars: list,
    norm_red_vars: list,
    new_ranges: list,
    return_getters_groups: list,
) -> dict:
    """What each original axis became, once everything is in the new axes.

    Dividing axes produces new axes, and a position in the old ones is a mixture
    of the new.  Each new axis is given a name, the mappings back are applied to
    those names, and what is left says which new axes each old axis became.
    """

    # What comes back from dividing the axes is the axes each program walks, and
    # for each of the original lengths a way of recovering its position from
    # them.  Flattened, a position is a mixture of the new axes -- six times six
    # is the first plus six times the second -- and that is what is built here.
    num_vars = sum(len(s) for s in new_ranges)
    flat_vars = sympy.symbols(f"v_0:{num_vars}")
    count = 0

    if len(iter_vars) == 0 and len(red_vars) == 0:
        return {}

    if len(new_ranges) != len(norm_pw_vars + norm_red_vars):
        raise AssertionError(
            f"expected len(new_ranges) == len(norm_pw_vars + norm_red_vars), "
            f"got {len(new_ranges)} and {len(norm_pw_vars + norm_red_vars)}"
        )
    apply_groups = []
    for group in return_getters_groups:
        apply_groups.append([g(flat_vars) for g in group])

    iter_vars_to_flat_vars = {}
    for i, (group, var_group) in enumerate(
        zip(apply_groups, (iter_vars, red_vars), strict=True)
    ):
        # A piece with a reduced axis of one has nothing to map there, since the
        # reduced axis was filled in while dividing.
        if len(group) != len(var_group):
            if i != 1:
                raise AssertionError(f"expected i == 1, got {i}")
            if len(var_group) != 0:
                raise AssertionError(
                    f"expected empty var_group, got len {len(var_group)}"
                )
            continue

        iter_vars_to_flat_vars.update({v: g for g, v in zip(group, var_group)})

    count = 0
    flat_vars_to_new_vars = {}
    for new_range, new_var in zip(
        new_ranges, norm_pw_vars + norm_red_vars, strict=True
    ):
        range_vars = []
        for _ in range(len(new_range)):
            range_vars.append(flat_vars[count])
            count += 1

        prod = 1
        for i in range(len(new_range) - 1, -1, -1):
            flat_vars_to_new_vars[range_vars[i]] = new_var * prod
            prod = new_range[i] * prod

    return {
        k: sympy_subs(v, flat_vars_to_new_vars)
        for k, v in iter_vars_to_flat_vars.items()
    }


def extract_normalized_read_writes(node):
    """The reads and writes of these pieces, all in one set of axes.

    What comes back is nothing where the shapes are not known well enough to
    say whether the pieces can be run together at all.
    """

    reads: dict = defaultdict(OrderedSet)
    writes: dict = defaultdict(OrderedSet)

    all_output_names = node.get_buffer_names()
    op_names = node.get_operation_names()
    outputs: OrderedSet = OrderedSet()
    removed_buffers: OrderedSet = OrderedSet()
    for buf_name in all_output_names:
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(buf_name, op_names):
            removed_buffers.add(buf_name)
        else:
            outputs.add(buf_name)

    inputs = OrderedSet(
        dep.name for dep in node.read_writes.reads if dep.name not in removed_buffers
    )

    pointwise_numel = node.group[1][0]
    red_numel = node.group[1][1]

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # A different prefix, so these axes can be told from the pieces' own.
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )

    for n in list(node.get_nodes()):
        if not isinstance(n, _scheduler_module().SchedulerNode):
            continue

        body = n._body

        n_reads: dict = defaultdict(OrderedSet)
        n_writes: dict = defaultdict(OrderedSet)

        for inp in inputs:
            for expr in body.get_all_read_expr(inp):
                n_reads[expr].add(inp)

        for out in outputs:
            for expr in body.get_all_write_expr(out):
                n_writes[expr].add(out)

        if not n_reads and not n_writes:
            continue

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(
            n, pointwise_numel, red_numel
        )

        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = prepare_split_iteration_lengths(groups, lengths, red_numel)
        try:
            new_ranges, return_getters_groups = _split_iteration_ranges(groups, lengths)
        except CantSplit as e:
            # Where the shapes are known, not being able to divide them is a
            # mistake.  Where they are not, it is a reason to say nothing.
            if not (pointwise_numel.free_symbols or red_numel.free_symbols):
                raise AssertionError(
                    "expected dynamic shapes (free symbols) when split fails"
                ) from e
            return None

        var_map = apply_var_mapping(
            iter_vars,
            red_vars,
            norm_pw_vars,
            norm_red_vars,
            new_ranges,
            return_getters_groups,
        )

        # Wrapping in a function that does nothing keeps the expressions from
        # being simplified into a plain number, which would lose the fact that
        # they came from an axis.
        def remove_identity(expr):
            return expr.replace(Identity, lambda x: x)

        n_reads_new = {
            sympy_subs(remove_identity(read), var_map): v for read, v in n_reads.items()
        }
        n_writes_new = {
            sympy_subs(remove_identity(write), var_map): v
            for write, v in n_writes.items()
        }

        for expr, buf_names in n_reads_new.items():
            reads[expr] |= buf_names

        for expr, buf_names in n_writes_new.items():
            writes[expr] |= buf_names

    reads = {
        V.graph.sizevars.simplify_with_ranges(r, ranges): v for r, v in reads.items()
    }
    writes = {
        V.graph.sizevars.simplify_with_ranges(w, ranges): v for w, v in writes.items()
    }

    fused_out = FusedNormalizedReadsWrites(
        norm_pw_vars,
        norm_red_vars,
        reads,
        writes,
        ranges,
    )
    loop_tiling_log.info("Normalized Fused reads: %s", fused_out)
    return fused_out


def get_score(addr, var_ranges, buf_names) -> int:
    """How much memory this access is expected to touch."""

    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v)
        # A position that comes from the data has no size to work from.
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .loops import V

    return V.graph.sizevars.optimization_hint(sympy_product(var_sizes))


def try_get_buf_size(buf_name):
    """How many elements this buffer holds, or nothing if that is not known."""

    buf = V.graph.try_get_buffer(buf_name)
    if not buf:
        return None
    return V.graph.sizevars.optimization_hint(sympy_product(buf.get_size()))


def get_hint(v) -> int:
    """This number, or what it is expected to be."""

    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.optimization_hint(v)


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """Dividing one axis by some factor, and what that would be worth."""

    var: sympy.Symbol
    tiling_factor: int
    score: int


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    """Which axes make accesses efficient, and which do not.

    The score is not strictly an amount of memory: a write is worth twice a
    read, since getting one right is worth more than getting the other right.
    """

    coalesced_by_var: dict
    uncoalesced_addrs: dict
    norm_read_writes: FusedNormalizedReadsWrites
    suggested_split: VarTiling | None = None


def _analyze_memory_coalescing(fused_node):
    """Which accesses are efficient, and whether re-dividing an axis would help.

    Each access is looked at to see which axis, if any, it steps through one
    element at a time, and what it is worth in memory.  Where some access has no
    such axis, each axis is tried in turn: if dividing it by some factor would
    make that access efficient, and the factor is worth having, that is what is
    suggested.
    """

    norm_read_writes = extract_normalized_read_writes(fused_node)

    if norm_read_writes is None:
        return None

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges

    coalesced_by_var: dict = Counter()
    uncoalesced_addrs: dict = Counter()

    # Only where nothing is being reduced, since a reduced axis is not what
    # decides whether neighbouring elements are neighbours.
    index_vars = norm_read_writes.index_vars
    reduce_vars = norm_read_writes.reduce_vars
    innermost_var = (
        next(reversed(index_vars)) if index_vars and not reduce_vars else None
    )

    for is_read, (memory_expr, buf_names) in itertools.chain(
        ((True, item) for item in reads.items()),
        ((False, item) for item in writes.items()),
    ):
        size = get_score(memory_expr, var_ranges, buf_names)
        if size == 0:
            continue

        # A position that comes from the data cannot be efficient or not.
        indirect_expr = has_indirect_access(memory_expr)

        if indirect_expr:
            maybe_coalesced_var = None
        else:
            maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
            # Many axes reading one place is not strictly efficient, but it
            # does keep the access within a cache line, which is most of what
            # being efficient is worth here.
            if maybe_coalesced_var is None:
                maybe_coalesced_var = find_broadcast_var(memory_expr, var_ranges)

        total_score = 0
        for buf_name in buf_names:
            if (buf := V.graph.try_get_buffer(buf_name)) and (
                buf_size := try_get_buf_size(buf_name)
            ):
                # At most the whole buffer is read, whatever the access says.
                total_score += min(buf_size, size) * buf.dtype.itemsize

        # A write that is done efficiently saves more than a read that is.
        total_score *= 1 if is_read else 2

        if maybe_coalesced_var:
            # Whether this is already efficient when walked as one axis, since
            # then it needs no dividing.  The axis that is walked fastest is
            # left out: consecutive programs walk it, so being efficient in it
            # is not a choice.
            already_coalesced_1d = False
            if innermost_var is not None and maybe_coalesced_var != innermost_var:
                # Stepping at two points, so that an expression which only looks
                # efficient at the start is not taken for one that is.
                subs = dict.fromkeys(var_ranges, 0)
                try:
                    val_0 = sympy_subs(memory_expr, subs)
                    subs[innermost_var] = 1
                    val_1 = sympy_subs(memory_expr, subs)
                    stride_01 = val_1 - val_0
                    if stride_01 in (0, 1):
                        subs[innermost_var] = 2
                        val_2 = sympy_subs(memory_expr, subs)
                        stride_12 = val_2 - val_1
                        if stride_12 in (0, 1):
                            already_coalesced_1d = True
                except (ZeroDivisionError, TypeError):
                    pass

            if not already_coalesced_1d:
                coalesced_by_var[maybe_coalesced_var] += total_score
            else:
                coalesced_by_var[innermost_var] += total_score
        else:
            uncoalesced_addrs[memory_expr] += total_score

    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    # Which axis, divided by what, would be worth the most.
    tiling_scores: dict = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        if has_indirect_access(uncoalesced_expr):
            continue

        expr_subs = dict.fromkeys(var_ranges.keys(), 0)
        for v in uncoalesced_expr.free_symbols & var_ranges.keys():
            if v not in var_ranges:
                continue
            if addr_score == 0:
                continue

            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0

            if len(single_var_expr.free_symbols) != 1:
                continue

            tiling_factor = solve_for_tiling(single_var_expr)

            if (
                tiling_factor is None
                or not tiling_factor.is_constant()
                or not tiling_factor.is_integer
            ):
                continue

            tiling_factor = int(tiling_factor)
            if not V.graph.sizevars.statically_known_lt(tiling_factor, var_ranges[v]):
                continue

            if not all(
                V.graph.sizevars.statically_known_lt(MIN_TILING_BLOCK, block)
                for block in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            tiling_scores[v][tiling_factor] += addr_score

    if len(tiling_scores) == 0:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    best_tiling: tuple | None = None
    best_tiling_score = 0

    for var, tiling_counter in tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            if tile_score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = tile_score

    if best_tiling is None:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var,
            uncoalesced_addrs=uncoalesced_addrs,
            norm_read_writes=norm_read_writes,
        )

    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        uncoalesced_addrs=uncoalesced_addrs,
        norm_read_writes=norm_read_writes,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], best_tiling_score),
    )


def analyze_memory_coalescing_for_nodes(nodes):
    """Which accesses are efficient, for these pieces looked at as one.

    Where they are already a group, the answer the group recorded is used, since
    that is what the rest of the work is written against.
    """

    if not nodes:
        return None

    node_types = (_scheduler_module().FusedSchedulerNode, _scheduler_module().SchedulerNode)
    if not all(isinstance(node, node_types) for node in nodes):
        return None

    if len(nodes) == 1:
        return nodes[0].get_coalesce_analysis()

    graph_scheduler = getattr(V.graph, "scheduler", None)
    if graph_scheduler is not None:
        fused_node = graph_scheduler.name_to_fused_node.get(nodes[0].get_first_name())
        if fused_node is not None:
            fused_nodes = list(fused_node.get_nodes())
            if len(fused_nodes) == len(nodes) and all(
                fused is node for fused, node in zip(fused_nodes, nodes, strict=True)
            ):
                return fused_node.get_coalesce_analysis()

    return _analyze_memory_coalescing(
        _FusedNodeView(
            nodes=nodes,
            read_writes=ReadWrites.merge_list([node.read_writes for node in nodes]),
            group=max(nodes, key=lambda node: int(node.is_reduction())).group,
        )
    )


def _scheduler_module():
    """The scheduler, which is imported here because the two refer to each other."""

    from . import kernel_scheduler

    return kernel_scheduler
