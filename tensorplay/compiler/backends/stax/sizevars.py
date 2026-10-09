"""What is known about the extents a call was given.

A layout question -- is this stride at least that one, is this extent a
multiple of that one -- has to be answered before any code is written, and the
answer is a fact about the shapes rather than about the program.  That is what
this holds: the shape environment of the region being compiled, and the
predicates over it that let a caller ask a question and get a definite answer
without a guard, plus the value of asking it when the answer is not available.

The predicates come in pairs on purpose.  A caller that is choosing between
two ways to write the same kernel wants the weaker question -- is this
divisible, ignoring whether the shape happens to be divisible this time -- so
that its choice is still correct for a shape it has not seen.  A caller that
is emitting a guard wants the stronger question about the shape in hand.
"""

from __future__ import annotations

import operator
import dataclasses
import functools
import itertools
import logging
from typing import Any

import sympy
from sympy import Expr

from tensorplay.graph.experimental._config import backed_size_oblivious
from tensorplay.graph.experimental._size_hinting import (
    _guarding_hint_or_throw_base,
    _maybe_realize_expr,
    _optimization_hint_base,
)
from tensorplay.graph.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    ShapeEnv,
)
from .codegen.index_expr import Expr as IndexExpr
from .loops import V
from .ops_handler import WrapperHandler
from .utils import (
    has_free_symbols,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_subs,
)
from tensorplay.graph.experimental.sympy_functions import (
    is_power_of_2,
    FloorDiv,
    IntInfinity,
    int_oo,
    ModularIndexing,
    Mod,
    OrderedSet,
    safe_gcd,
    simple_floordiv_gcd,
    symbol_is_type,
    SymT,
    ValueRanges,
)


log = logging.getLogger(__name__)


def _size_expr(value):
    if isinstance(value, IndexExpr):
        return value.to_sympy()
    return value

def free_symbols_of(*values: Any) -> set:
    """Every free symbol appearing in the values."""

    out: set = set()
    for value in values:
        if isinstance(value, (Expr, sympy.Basic)):
            out |= value.free_symbols
        elif isinstance(value, (list, tuple, set, frozenset)):
            out |= free_symbols_of(*value)
        elif isinstance(value, dict):
            out |= free_symbols_of(*value.keys(), *value.values())
    return out


class SizeVarAllocator:
    """The shape environment, plus the questions asked of it while lowering.

    A lowering asks two kinds of question.  One is about the value the shape
    has now, and may be answered by evaluating an expression; the other is
    about what may be assumed of the shape in general, and may only be
    answered by what the environment already knows, because a fact that holds
    only for this shape must not be baked into the code.
    """

    def __init__(self, shape_env: ShapeEnv | None = None) -> None:
        super().__init__()
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.backed_var_to_val = self.shape_env.backed_var_to_val
        self.var_to_hint_override = self.shape_env.var_to_hint_override
        self.replacements: dict = self.shape_env.replacements
        self.unbacked_replacements: dict | None = None
        # Dynamic extents whose value is a whole expression are precomputed on
        # the host and passed in as an argument, so a kernel refers to the
        # argument instead of repeating the expression.  Such an argument
        # cannot be guarded on, so a guard on an expression that may already
        # have been precomputed has to be phrased on the expression itself,
        # which is what the inverse map is for.
        self.precomputed_replacements: dict = {}
        self.inv_precomputed_replacements: dict = {}
        # The same expression is asked for a hint over and over while lowering,
        # and every miss costs a substitution the algebra does not remember, so
        # the answers are kept until the replacements change underneath them.
        self._optimization_hint_cache = self._lru_cache(
            self._optimization_hint_uncached
        )
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        """The expression with the known values of the shapes substituted in."""

        return sympy.expand(expr).xreplace(self.replacements)

    def guarding_hints_or_throw(self, exprs) -> tuple:
        """The values each expression is known to have, as plain integers.

        This is a guard: an expression whose value is not already known is
        refused rather than guessed, because the caller is about to compare
        the values it gets and a guess would decide that comparison.
        """

        return tuple(self.shape_env.guarding_hint_or_throw(x) for x in exprs)

    def _lru_cache(self, fn, maxsize=None):
        """Wrap a function in a cache that empties when the replacements change.

        Anything derived from the replacements is wrong the moment one of them
        changes, so the cache is dropped rather than left to answer from what
        it remembers.
        """

        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal prior_len
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)

        return wrapper

    def make_stride_vars_cache(self):
        """The strides of an index, cached until the replacements change."""

        cache = self._lru_cache(self._stride_vars)

        def stride_vars(
            index: Expr,
            vars: Sequence,
            support_vars: Sequence = None,
        ) -> list:
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence,
        support_vars: Sequence,
    ) -> list:
        """An index taken apart into the stride of each loop.

        The stride of a loop is how far the index moves when that loop's
        variable moves by one, which is read by setting every other variable to
        zero and subtracting.  This is only the right answer for a plainly
        strided offset: an index that wraps partway along, such as ten times a
        modular indexing of one, has a negative stride here, because the wrap is
        not something a stride can express.
        """

        strides = []
        index = self.simplify(index)
        # Take out the part of the index that does not move with any loop.
        index = index - sympy_subs(
            index, {v: sympy.S.Zero for v in support_vars if v != 0}
        )
        for i in range(len(vars)):
            # Drop every loop but this one, by setting it to zero.
            index_dim = sympy_subs(
                index,
                {
                    support_vars[j]: sympy.S.Zero
                    for j in range(len(support_vars))
                    if vars[i] != support_vars[j] and support_vars[j] != 0
                },
            )
            v = vars[i]
            if v == 0:
                strides.append(sympy.S.Zero)
            else:
                strides.append(
                    sympy_subs(index_dim, {v: sympy.S.One})
                    - sympy_subs(index_dim, {v: sympy.S.Zero})
                )
        return strides

    def offset_var(self, index: Expr, vars: Sequence) -> Expr:
        """The part of an index that does not move with any loop."""

        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.S.Zero for v in vars if v != 0})

    def stride_hints(
        self,
        index: Expr,
        vars: Sequence,
        support_vars: Sequence = None,
    ) -> list:
        """Concrete guesses at the stride of each loop, for choosing between layouts.

        A position reached through a vector of indirect indexes has no stride
        that can be read off the index, so those are treated as not moving at
        all rather than left to produce a nonsense number.
        """

        for v in index.free_symbols:
            if symbol_is_type(v, SymT.INDIRECT):
                index = sympy_subs(index, {v: 0})
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            result.append(self.optimization_hint(s, fallback=0))
        return result

    def stride_order(self, index: Expr, vars: list) -> list:
        """The loops from smallest stride to largest, with the broadcasts first.

        A loop of stride zero moves nothing, so where it goes in the order says
        nothing about the order the rest must run in; putting those first is
        what makes the rest of the list readable as an order.
        """

        strides = tuple(map(abs, self.stride_hints(index, vars)))
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def make_simplify_with_ranges_cache(self):
        """Simplification that knows the extents, cached until they change."""

        cache: dict = {}
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges) -> Expr:
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
                if result != expr:
                    cache[(result, *var_ranges.items())] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        """Loop merging, cached until the replacements change."""

        cache: dict = {}
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key)
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result

        return simplify_loops

    def _simplify_with_ranges(self, expr: Expr, var_ranges) -> Expr:
        """Simplify an index, knowing how far each of its loops runs.

        Knowing the extents is what lets a division be decided: a loop that
        runs from zero to fewer than the divisor cannot divide anything, so the
        division is zero, and a position taken modulo something larger than the
        whole range is the position itself.
        """

        expr = _size_expr(expr)
        var_ranges = {k: _size_expr(v) for k, v in var_ranges.items()}

        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        var_to_range = dict(self.shape_env.var_to_range)
        var_to_range.update(
            {
                k: ValueRanges(
                    0, max(0, v - 1) if not has_free_symbols([v]) else IntInfinity()
                )
                for k, v in var_ranges.items()
            }
        )
        for var in expr.free_symbols:
            if var not in var_to_range:
                var_to_range[var] = ValueRanges(0, IntInfinity())

        var_to_range_tuple = tuple(var_to_range.items())

        axioms = []
        for var, upper_bound in var_ranges.items():
            axioms.append(0 <= var)
            axioms.append(var < upper_bound)
        axioms = tuple(axioms) + self.shape_env.get_axioms()

        def statically_known(expr):
            evaluated = self.shape_env._maybe_evaluate_static(
                expr,
                axioms=axioms,
                var_to_range=var_to_range_tuple,
            )
            return bool(evaluated)

        def remove_zero_terms(base, divisor):
            """Drop terms that are certainly zero because their loop is short.

            A loop of fewer than the divisor has no multiple of the divisor in
            it, so the part of the index that would carry one is zero however
            the loop runs.
            """

            if not statically_known(base >= 0):
                return base

            for v in base.free_symbols:
                if v in var_ranges:
                    rest = sympy.Wild("_rest", exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = safe_gcd(m[rest], divisor)
                        if statically_known(v < gcd):
                            base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            base = remove_zero_terms(base, divisor)
            if statically_known(base >= 0) and statically_known(base < divisor):
                return sympy.S.Zero
            # Taking the divisor of a position that is already taken modulo
            # something the divisor divides is the same as taking it modulo
            # that something divided by the divisor.
            if isinstance(base, ModularIndexing) and isinstance(divisor, sympy.Integer):
                b, d1, m = base.args
                if m % divisor == 0:
                    return ModularIndexing(b, d1 * divisor, FloorDiv(m, divisor))
            return FloorDiv(base, divisor)

        def visit_modular_indexing(base, divisor, modulus):
            base = remove_zero_terms(base, divisor)

            if isinstance(base, ModularIndexing):
                inner_base, inner_divisor, inner_modulus = base.args
                period = divisor * modulus
                # The common case has constant numbers, and the analysis below
                # is not worth its cost on symbolic ones.
                if all(
                    isinstance(value, sympy.Integer) and value > 0
                    for value in (inner_modulus, divisor, modulus)
                ) and self.statically_known_multiple_of(inner_modulus, period):
                    return ModularIndexing(inner_base, inner_divisor * divisor, modulus)

            can_remove_mod = statically_known(base >= 0) and statically_known(
                base < modulus * divisor
            )

            if can_remove_mod:
                return FloorDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)

        if expr.has(ModularIndexing):
            expr = expr.replace(
                ModularIndexing(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                    sympy.Wild("modulus", integer=True),
                ),
                visit_modular_indexing,
            )

        if expr.has(FloorDiv):
            expr = expr.replace(
                FloorDiv(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                ),
                visit_indexing_div,
            )

        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(
        self,
        index_vars: list,
        sizes,
        index_formulas,
    ):
        """Remove as many loops as the indexes allow.

        Two things are done, repeatedly until neither applies: a loop of extent
        one is dropped, since it can only run once and orders nothing, and two
        adjacent loops are merged when every index moves along them in a way
        that is the same as running one loop over their product.
        """

        sizes = list(map(self.simplify, sizes))

        strides = [
            # An index formula may be a relation rather than a position, and a
            # relation has no stride, so it counts as not moving at all.  It
            # can still stop two loops from merging, because the test below
            # substitutes into it and reads the result.
            (
                self.stride_vars(x, index_vars)
                if isinstance(x, sympy.Expr)
                else [0] * len(index_vars)
            )
            for x in index_formulas
        ]
        if len(sizes) != len(strides[0]):
            raise AssertionError((len(sizes), len(strides[0])))

        for i in range(len(sizes)):
            if sizes[i] == 1:
                # remove dim
                sizes[i] = None

        def can_merge_dims(a, b):
            for k in range(len(strides)):
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(
                    strides[k][b]
                ):
                    # The cheap test passed, so try the one that cannot be
                    # fooled: run the two loops as one and see whether every
                    # index comes out the same.
                    va = index_vars[a]
                    vb = index_vars[b]
                    m1 = sympy_index_symbol("_merge_tester1")
                    m2 = sympy_index_symbol("_merge_tester2")
                    # Neither variable may be set to zero here: an index that
                    # contains the product of the two would come out zero under
                    # either substitution and the two would look equal.
                    expr1 = sympy_subs(index_formulas[k], {va: m1 * sizes[a], vb: m2})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: (m1 + m2)})
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True

        changed = True
        while changed:
            changed = False
            for i, j in itertools.product(
                reversed(range(len(sizes))), reversed(range(len(sizes)))
            ):
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        def reindex(index):
            it = list(reversed(index))
            new_index = []
            for size in sizes:
                if size is None:
                    new_index.append(sympy.S.Zero)
                else:
                    new_index.append(it.pop())
            if it:
                raise AssertionError(f"expected all entries consumed, got {it}")
            return new_index

        def prune(index):
            if len(index) != len(sizes):
                raise AssertionError(
                    f"expected len(index) == len(sizes), got {len(index)} != {len(sizes)}"
                )
            return [i for i, s in zip(index, sizes) if s is not None]

        return [x for x in sizes if x is not None], reindex, prune

    # The statically_known_* family never guards.  A question they answer yes
    # to is a fact that may be acted on; a question they cannot answer is
    # simply left alone, because adding a guard would be a different and much
    # stronger thing to do.
    def statically_known_true(self, expr) -> bool:
        """Whether the relation holds, or can be shown to hold, without guarding."""

        return statically_known_true(self.shape_env, expr)

    def statically_known_equals(self, left, right) -> bool:
        """Whether the two are the same for every shape the code will be handed."""

        return self.statically_known_true(
            sympy.Eq(_size_expr(left), _size_expr(right))
        )

    def statically_known_list_equals(self, left, right) -> bool:
        """Whether two lists are the same element for element, for every shape."""

        return len(left) == len(right) and all(
            self.statically_known_equals(l, r) for l, r in zip(left, right)
        )

    def statically_known_leq(self, left, right) -> bool:
        """Whether one is no larger than the other, for every shape."""

        expr = _size_expr(left) <= _size_expr(right)
        return self.statically_known_true(expr)

    def statically_known_geq(self, left, right) -> bool:
        """Whether one is no smaller than the other, for every shape."""

        expr = _size_expr(left) >= _size_expr(right)
        return self.statically_known_true(expr)

    def statically_known_lt(self, left, right) -> bool:
        """Whether one is strictly smaller than the other, for every shape."""

        expr = _size_expr(left) < _size_expr(right)
        return self.statically_known_true(expr)

    def statically_known_gt(self, left, right) -> bool:
        """Whether one is strictly larger than the other, for every shape."""

        expr = _size_expr(left) > _size_expr(right)
        return self.statically_known_true(expr)

    def _is_multiple_of(self, numerator, denominator: int) -> bool:
        """Whether a numerator is certainly a multiple of a denominator.

        The structure of the expression is read first, because that is cheap and
        settles most cases; only what the structure leaves open is put to the
        algebra, which is far more expensive.
        """

        # A number is its own answer.
        if isinstance(numerator, (int, sympy.Integer)):
            return int(numerator) % denominator == 0

        # In a product, one factor being a multiple makes the product one.
        if isinstance(numerator, sympy.Mul):
            for factor in numerator.args:
                if self._is_multiple_of(factor, denominator):
                    return True
            const = 1
            for factor in numerator.args:
                if isinstance(factor, (int, sympy.Integer)):
                    const *= int(factor)
            if const != 1 and const % denominator == 0:
                return True

        # In a sum, every term being a multiple makes the sum one.
        if isinstance(numerator, sympy.Add):
            if all(self._is_multiple_of(term, denominator) for term in numerator.args):
                return True

        # A division is a multiple when what it divides is a multiple of a
        # correspondingly larger multiple.
        if isinstance(numerator, FloorDiv):
            a, b = numerator.args
            if isinstance(b, (int, sympy.Integer)):
                if self._is_multiple_of(a, int(b) * denominator):
                    return True

        # A remainder is a difference of a multiple and a multiple, so it is a
        # multiple when both of those are.
        if isinstance(numerator, (Mod, sympy.Mod)):
            a, b = numerator.args
            if self._is_multiple_of(a, denominator) and self._is_multiple_of(
                b, denominator
            ):
                return True

        # A cheap common factor before the expensive fallback.
        gcd = simple_floordiv_gcd(numerator, sympy.Integer(denominator))
        if isinstance(gcd, (int, sympy.Integer)) and int(gcd) % denominator == 0:
            return True

        # The algebra decides, unless the expression is wide enough that the
        # cost of asking outweighs what the answer is worth.
        if len(free_symbols_of(numerator)) > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS:
            return False
        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)

    def statically_known_multiple_of(self, numerator, denominator) -> bool:
        """Whether the numerator is certainly a multiple of the denominator."""

        # Compared by identity first: building an equation between two wide
        # expressions costs more than everything else here put together.
        if numerator is denominator or numerator == denominator:
            return True

        if isinstance(denominator, (int, sympy.Integer)):
            return self._is_multiple_of(numerator, int(denominator))

        if numerator == 0:
            return True

        if (
            len(free_symbols_of([numerator, denominator]))
            > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS
        ):
            return False

        def gcd_covers_denominator(gcd) -> bool:
            return gcd != 1 and (
                gcd == denominator or self.simplify(gcd - denominator) == 0
            )

        # A common factor that already covers the denominator proves it.  This
        # is checked before the algebra because the algebra can miss a sum
        # whose factors cancel.
        try:
            if gcd_covers_denominator(simple_floordiv_gcd(numerator, denominator)):
                return True
            if gcd_covers_denominator(safe_gcd(numerator, denominator)):
                return True
        except sympy.PolynomialError:
            pass

        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)

    def statically_known_power_of_2(self, expr) -> bool:
        """Whether a value is known to be a power of two."""

        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))

    def expect_true(self, expr) -> bool:
        """Make sure a relation the caller already believes holds does hold.

        This does not insist the relation is true -- a caller that was wrong
        about it still gets code -- but it does not let it go by unrecorded
        either: a guard or a runtime assertion is put in place so that a shape
        which does not satisfy it is caught rather than miscompiled.
        """

        if not self.statically_known_true(expr):
            return self.shape_env.guard_or_defer_runtime_assert(
                expr, "sizevars.expect_true"
            )
        return True

    def check(self, expr) -> None:
        """Refuse to go on unless a relation the caller relies on does hold.

        The relation is stated in terms of the original expressions rather than
        of the arguments they were replaced by, since it is the original
        expression that the caller can be held to.
        """

        expr = sympy_subs(expr, self.inv_precomputed_replacements)
        if not self.expect_true(expr):
            raise AssertionError(f"expect_true failed for {expr}")

    def check_equals(self, left, right):
        """Refuse to go on unless the two are equal, and hand the first back."""

        self.check(sympy.Eq(left, right))
        return left

    def check_equals_and_simplify(self, left, right):
        """Refuse to go on unless the two are equal, and hand back the first in
        terms of the expressions the caller passed rather than the arguments."""

        self.check(sympy.Eq(left, right))
        return sympy_subs(left, self.inv_precomputed_replacements)

    def check_leq(self, left, right) -> None:
        """Refuse to go on unless one is no larger than the other."""

        self.check(sympy.Le(left, right))

    def check_lt(self, left, right) -> None:
        """Refuse to go on unless one is strictly smaller than the other."""

        self.check(sympy.Lt(left, right))

    def guard_or_false(self, left):
        """The truth of a relation, or false when it cannot be decided.

        A relation that holds only for the shape in hand is not a fact about
        the code, so when it cannot be settled generally the answer is the one
        that is safe without a guard.
        """

        # A claim that is already settled -- a comparison of two numbers
        # rather than of two extents -- is answered with itself.  Only a
        # claim about extents needs deciding against what is known about
        # them, and treating a settled answer as one to be decided is how a
        # plain true becomes a question with no answer.
        if not isinstance(left, sympy.logic.boolalg.Boolean):
            if not isinstance(left, bool):
                raise AssertionError(f"Expected bool, got {type(left)}")
            return left


        if backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return False
        return self.evaluate_expr(left, fallback_value=False)

    def guard_or_true(self, left):
        """The truth of a relation, or true when it cannot be decided."""

        # A claim that is already settled -- a comparison of two numbers
        # rather than of two extents -- is answered with itself.  Only a
        # claim about extents needs deciding against what is known about
        # them, and treating a settled answer as one to be decided is how a
        # plain true becomes a question with no answer.
        if not isinstance(left, sympy.logic.boolalg.Boolean):
            if not isinstance(left, bool):
                raise AssertionError(f"Expected bool, got {type(left)}")
            return left


        if backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return True
        return self.evaluate_expr(left, fallback_value=True)

    def evaluate_expr(
        self,
        left,
        size_oblivious: bool = False,
        fallback_value=None,
    ) -> bool:
        """What a relation is, guarding on it being that.

        The relation must be written as an expression rather than as a
        comparison, so that a shape is never compared by its own equality and
        quietly accepted.
        """

        if not isinstance(left, (Expr, sympy.logic.boolalg.Boolean)):
            raise AssertionError(type(left))
        return self.shape_env.evaluate_expr(
            sympy.sympify(left),
            size_oblivious=size_oblivious,
            fallback_value=fallback_value,
        )

    def is_size_one_or_false(self, size) -> bool:
        """Whether an extent is one, or a shape that is not known to be one.

        A shape that came out of the data is not asked about, because answering
        would mean waiting for a value that is not there yet.
        """

        return self.guard_or_false(sympy.Eq(size, 1))

    def evaluate_min(self, left, right):
        """The smaller of two extents, and a guard saying which it was."""

        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)
        if self.guard_or_false(sympy.Le(left, right)):
            return left
        if self.guard_or_false(sympy.Le(right, left)):
            return right

        # A common factor that is one of the two means the other is a multiple
        # of it, and a multiple of a positive number is not smaller.
        gcd = sympy.gcd(left, right)
        if left == gcd:
            return left
        if right == gcd:
            return right

        # A minimum of one term against a known bound can be settled by which
        # term is under that bound, which the algebra does not do on its own.
        for lhs, rhs in [(left, right), (right, left)]:

            def le_rhs(a):
                return self.guard_or_false(sympy.Le(a, rhs))

            if isinstance(lhs, (sympy.Min,)) and any(le_rhs(a) for a in lhs.args):
                return lhs
            if isinstance(lhs, (sympy.Max,)) and all(le_rhs(a) for a in lhs.args):
                return lhs

        raise TypeError(
            f"evaluate_min({left}, {right}) with unbacked symints"
        ) from None

    def evaluate_max(self, left, right):
        """The larger of two extents, and a guard saying which it was."""

        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)
        if self.guard_or_false(sympy.Ge(left, right)):
            return left
        if self.guard_or_false(sympy.Ge(right, left)):
            return right

        gcd = sympy.gcd(left, right)
        if left == gcd:
            return left
        if right == gcd:
            return right

        for lhs, rhs in [(left, right), (right, left)]:

            def ge_rhs(a):
                return self.guard_or_false(sympy.Ge(a, rhs))

            if isinstance(lhs, (sympy.Max,)) and any(ge_rhs(a) for a in lhs.args):
                return lhs
            if isinstance(lhs, (sympy.Min,)) and all(ge_rhs(a) for a in lhs.args):
                return lhs

        raise TypeError(
            f"evaluate_max({left}, {right}) with unbacked symints"
        ) from None

    def free_symbols(self) -> set:
        """Every shape the region has a value for and has not replaced away.

        A shape whose value has been substituted into the expressions is no
        longer a free one, and listing it would suggest it still varies.
        """

        return OrderedSet(self.backed_var_to_val.keys()) - OrderedSet(
            self.replacements.keys()
        )

    def remove_precomputed_replacements(self, expr):
        """The expression with the arguments put back in place of what they stand for.

        A precomputed extent is an argument, and the caller may only be asked
        about the expression it was computed from, so the substitution goes the
        other way here.
        """

        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):
            return sympy_subs(expr, self.inv_precomputed_replacements)
        return expr

    def lookup_precomputed_size(self, expr):
        """The argument a precomputed extent is passed as, or the extent itself.

        Anything already a plain value, a symbol or a number is left alone, and
        so is an expression that has not been precomputed.
        """

        if (
            isinstance(expr, (int, sympy.Symbol, sympy.Number))
            or expr.is_number
            or expr.is_symbol
        ):
            return expr
        expr = self.remove_precomputed_replacements(expr)
        if expr not in self.precomputed_replacements:
            sym = sympy_index_symbol_with_prefix(
                SymT.PRECOMPUTED_SIZE, len(self.precomputed_replacements)
            )
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
            return sym
        return self.precomputed_replacements[expr]

    def all_unbacked_explicitly_hinted(self, exprs) -> bool:
        """Whether every shape that came out of the data has a value given for it.

        With no such shape at all the answer is yes, since there is nothing
        left to want a value for.
        """

        unbacked = free_unbacked_symbols(exprs)
        return unbacked.issubset(self.var_to_hint_override.keys())

    def optimization_hint_with_override(self, expr, hint_override) -> int:
        """A value to stand in for an extent, preferring one the caller gave.

        A shape that is already known keeps its own value whatever the caller
        suggested: a suggestion is for a shape that has no value yet, and
        overriding a known one would be acting on the wrong number.
        """

        simplified = _maybe_realize_expr(self.simplify(expr), None)

        if simplified is not None:
            return simplified

        if hint_override is not None:
            return hint_override

        return self.optimization_hint(expr)

    def optimization_hints_with_override(self, exprs, hint_override) -> tuple:
        """The same, for a sequence of extents."""

        return tuple(
            self.optimization_hint_with_override(e, hint_override) for e in exprs
        )

    def guarding_hint_or_throw(self, expr):
        """The value an extent is known to have, refusing rather than guessing."""

        return _guarding_hint_or_throw_base(
            self.shape_env, expr, self.inv_precomputed_replacements
        )

    def guard_int(self, *args):
        """Ask the environment to vouch for a shape being the value given.

        A shape that is a plain number is returned as it is; anything else is
        the environment's to answer, since a number standing in for a shape is
        what this call is asking it to confirm.
        """

        if len(args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(args)}")
        (x,) = args
        if not isinstance(x, (Expr, int)):
            raise AssertionError(f"expected Expr or int, got {x}")
        if isinstance(x, int):
            return x
        return self.shape_env.guard_int(x)

    def guard_int_seq(self, *args):
        """The same, for a sequence of shapes."""

        if len(args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(args)}")
        (x,) = args
        if not isinstance(x, (tuple, list, Expr)):
            raise AssertionError(f"expected tuple, list, or Expr but got {x}")
        if isinstance(x, Expr):
            return [self.guard_int(x)]
        return [self.guard_int(xi) for xi in x]

    def to_symint_or_int(self, x):
        """Wrap an expression so layout decisions record their constraints."""

        if isinstance(x, int):
            return x
        if not isinstance(x, Expr):
            raise AssertionError(f"expected Expr or int, got {x}")
        x = self.remove_precomputed_replacements(self.simplify(x))
        if isinstance(x, (int, sympy.Integer)):
            return int(x)
        try:
            hint = self.guarding_hint_or_throw(x)
        except Exception:
            hint = None
        return self.shape_env.create_symintnode(x, hint=hint)

    def to_symints_or_ints(self, x):
        """The values of a sequence of shapes, one at a time."""

        if isinstance(x, (list, tuple)):
            return [self.to_symint_or_int(xi) for xi in x]
        if isinstance(x, Expr):
            return self.to_symint_or_int(x)
        raise AssertionError(f"expected list, tuple, or Expr but got {x}")

    def expand_floor_div(
        self,
        index: sympy.Expr,
        candidate_vars: Iterable[sympy.Symbol] | None = None,
    ) -> bool | tuple[sympy.Expr, sympy.Expr]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization cannot be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """
        candidate_vars_set = (
            OrderedSet(candidate_vars) if candidate_vars is not None else None
        )

        def parse_static_term(
            term: sympy.Expr,
        ) -> tuple[sympy.Expr, sympy.Symbol] | None:
            if not isinstance(term, sympy.Mul):
                return None
            # For dynamic shape, term like '2*s1*x1' has 3 child nodes.
            # Without candidate loop vars we cannot tell shape and loop
            # symbols apart, so preserve the historical conservative path.
            if len(term.args) != 2:
                return None
            factor, var = term.args
            if not isinstance(factor, sympy.Integer) or not isinstance(
                var, sympy.Symbol
            ):
                return None
            return factor, var

        def parse_term(term: sympy.Expr) -> tuple[sympy.Expr, sympy.Symbol] | None:
            if candidate_vars_set is None:
                parsed = parse_static_term(term)
                if parsed is None:
                    return None
                factor, var = parsed
            else:
                if isinstance(term, sympy.Symbol):
                    factor = sympy.S.One
                    var = term
                elif isinstance(term, sympy.Mul):
                    vars_in_term = [
                        symbol
                        for symbol in term.free_symbols
                        if symbol in candidate_vars_set
                    ]
                    if len(vars_in_term) == 0:
                        parsed = parse_static_term(term)
                        if parsed is None:
                            return None
                        factor, var = parsed
                    elif len(vars_in_term) == 1:
                        var = vars_in_term[0]
                        factor = sympy.cancel(term / var)
                        if factor.has(*candidate_vars_set):
                            return None
                        if factor.is_integer is not True:
                            return None
                    else:
                        return None
                else:
                    return None

                if var not in candidate_vars_set:
                    return None

            # It's easier to reason about the correctness of the transformation
            # for non-negative integers.
            if not self.statically_known_geq(var, 0):
                return None
            return factor, var

        if not isinstance(index, sympy.Add):
            return False
        terms = index.args

        if len(terms) < 2:
            return False
        floor_div_index = -1
        varlist = []
        factorlist = []
        for idx, term in enumerate(terms):
            if parsed := parse_term(term):
                factor, var = parsed
                varlist.append(var)
                factorlist.append(factor)
            elif isinstance(term, FloorDiv):
                var, factor = term.args
                if not isinstance(factor, sympy.Integer) or not isinstance(
                    var, sympy.Symbol
                ):
                    return False
                if candidate_vars_set is not None and var not in candidate_vars_set:
                    return False
                if not self.statically_known_geq(var, 0):
                    return False
                if floor_div_index >= 0:
                    # can not handle multi FloorDiv yet
                    return False

                floor_div_index = idx
                varlist.append(var)
                # this factor is denominator
                factorlist.append(factor)
            else:
                return False

        if floor_div_index < 0:
            return False

        # Construct the new expression and remember the denominator
        denominator = factorlist[floor_div_index]
        new_index = sympy.S.Zero

        for var, factor, idx in zip(varlist, factorlist, itertools.count()):
            if idx == floor_div_index:
                new_index += var
            else:
                new_index += (factor * denominator) * var

        return new_index, denominator

    def combine_modular_indexing_pairs(self, expr):
        """One position written two ways, put back into one.

        A view of a view writes the same position as a remainder and as which
        group it is, and the two can be written as a single wider remainder.
        """

        if isinstance(expr, sympy.Add):
            return sympy.Add(
                *(self.combine_modular_indexing_pairs(x) for x in expr.args)
            )
        if expr.has(ModularIndexing):
            expr = join_dimensions(expr)
        return expr

    def replace_backed_symbols_with_hints(self, expr):
        """The expression with each known shape replaced by the value it has.

        A shape that has no value yet is left as it is, since there is nothing
        to replace it with.  A plain number is handed back as it is: there is
        nothing in it to replace, and a caller holding a number already has the
        value this would have substituted.
        """

        if isinstance(expr, int):
            return expr

        replacements = {}
        for symbol in free_symbols_of(expr):
            if symbol in self.backed_var_to_val:
                replacements[symbol] = self.backed_var_to_val[symbol]
        return expr.xreplace(replacements)

    def analyze_lane_contiguity(self, expr: Expr, lane_var) -> LaneContiguity:
        """How an index moves from one lane of a vector to the next.

        Anything this cannot account for is reported as unknown rather than
        guessed at, because a guess that is wrong about contiguity is a wrong
        memory access.  What comes out describes the pattern of the index only:
        whether the extent, the stride and where the load starts actually permit
        a vector load is for the caller to establish.
        """

        expr = self.simplify(expr)
        if lane_var not in expr.free_symbols:
            return LaneContiguity(stride=0, uniform_width=int_oo)
        stride = stride_at(expr, lane_var)
        match expr:
            case _ if self.statically_known_equals(stride, 1):
                return LaneContiguity(contiguous_width=int_oo, stride=1)
            case _ if self.statically_known_equals(stride, 0):
                return LaneContiguity(stride=0, uniform_width=int_oo)
            case sympy.Add():
                return self._analyze_lane_contiguity_add(expr.args, lane_var)
            case sympy.Mul():
                return self._analyze_lane_contiguity_mul(expr.args, lane_var)
            case FloorDiv():
                return self._analyze_lane_contiguity_floor_div(
                    expr.args[0], expr.args[1], lane_var
                )
            case ModularIndexing():
                return self._analyze_lane_contiguity_modular_indexing(expr, lane_var)
            case _ if isinstance(expr, (Mod, sympy.Mod)):
                return self._analyze_lane_contiguity_mod(
                    expr.args[0], expr.args[1], lane_var
                )
            case _:
                return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_add(self, args, lane_var) -> LaneContiguity:
        """A sum, when at most one of its terms moves across the lanes.

        Two terms that both move would have to be added to know the pattern,
        and that is more than the pattern of either term says, so such a sum is
        reported as unknown.
        """

        uniform_width: Width | None = int_oo
        varying_result: LaneContiguity | None = None
        for arg in args:
            arg_result = self.analyze_lane_contiguity(arg, lane_var)
            if arg_result.unknown:
                return LaneContiguity(unknown=True)
            if arg_result.uniform:
                uniform_width = _min_width(uniform_width, arg_result.uniform_width)
            elif varying_result is None:
                varying_result = arg_result
            else:
                return LaneContiguity(unknown=True)
        if varying_result is None:
            return LaneContiguity(stride=0, uniform_width=uniform_width)
        return LaneContiguity(
            contiguous_width=_min_width(varying_result.contiguous_width, uniform_width),
            stride=varying_result.stride,
        )

    def _analyze_lane_contiguity_mul(self, args, lane_var) -> LaneContiguity:
        """A product, when exactly one of its factors moves across the lanes.

        A factor that is the same for every lane scales the pattern of the one
        that moves: it multiplies the stride, and a stride of more than one
        means the lanes are no longer consecutive however wide the moving part
        was.
        """

        uniform_factor: Expr = sympy.S.One
        lane_factors = []
        for arg in args:
            if lane_var in arg.free_symbols:
                lane_factors.append(arg)
            else:
                uniform_factor *= arg
        if len(lane_factors) != 1:
            return LaneContiguity(unknown=True)
        child_result = self.analyze_lane_contiguity(lane_factors[0], lane_var)
        if child_result.unknown or child_result.stride is None:
            return LaneContiguity(unknown=True)
        stride = self.simplify(uniform_factor * child_result.stride)
        if self.statically_known_equals(stride, 0):
            return LaneContiguity(
                stride=0,
                uniform_width=child_result.uniform_width,
            )
        return LaneContiguity(
            contiguous_width=child_result.contiguous_width
            if self.statically_known_equals(stride, 1)
            else None,
            stride=stride,
        )

    def _analyze_lane_contiguity_modular_indexing(self, expr: Expr, lane_var):
        """A position taken within each group, read as a division or a remainder."""

        base, divisor, modulus = expr.args
        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        if int(divisor) != 1:
            return self._analyze_lane_contiguity_floor_div(base, divisor, lane_var)
        return self._analyze_lane_contiguity_mod(base, modulus, lane_var)

    def _analyze_lane_contiguity_floor_div(self, base: Expr, divisor: Expr, lane_var):
        """How many lanes a division by a constant is the same for.

        A division is the same for every lane of a group only if the lanes all
        land in the same group, which is what dividing by a power of two that
        the position is a multiple of guarantees.
        """

        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        divisor_int = int(divisor)
        if divisor_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(divisor_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(stride=0, uniform_width=width)
            width //= 2
        return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_mod(self, base: Expr, modulus: Expr, lane_var):
        """How many lanes a remainder is consecutive for.

        The remainder of a run is consecutive only while the run does not wrap:
        lanes zero to three under a modulus of four are consecutive, but lanes
        two to five come back as two, three, zero, one.  So the group has to
        start on a multiple of the width, and the width has to divide the
        modulus.
        """

        if not isinstance(modulus, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        modulus_int = int(modulus)
        if modulus_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(modulus_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(contiguous_width=width, stride=1)
            width //= 2
        return LaneContiguity(unknown=True)

    def optimization_hint(self, expr, fallback: int | None = None) -> int:
        """A concrete value to use for an extent while choosing between layouts.

        This is a hint and not a value: it exists so a choice between two ways
        of writing the same kernel can be made before the shape is known, and
        so it must never be used to decide anything about correctness.  When
        the environment has nothing to suggest, the caller's fallback is what
        the choice is made on.
        """

        return self._optimization_hint_cache(expr, fallback)

    def _optimization_hint_uncached(self, expr, fallback):
        """A concrete value to stand in for an extent, computed from scratch.

        A complex expression is refused, because a tensor extent is a count of
        elements and cannot be complex.  An unbounded one becomes the largest
        value an extent can take, which keeps arithmetic on it from wrapping.
        An undefined one has no value to stand in, so the caller's fallback is
        what is used.
        """

        return _optimization_hint_base(
            self.shape_env, expr, self.inv_precomputed_replacements, fallback
        )

    def optimization_hints(self, exprs, fallback: int | None = None):
        """A concrete value for each of the extents, where there is one."""

        return tuple(
            self.optimization_hint(x, fallback) if isinstance(x, Expr) else x
            for x in exprs
        )

    def evaluate_expr(
        self,
        left: Expr | sympy.logic.boolalg.Boolean,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
    ) -> bool:
        """The truth of a claim about the extents, or the fallback answer."""

        if not isinstance(left, (Expr, sympy.logic.boolalg.Boolean)):
            raise AssertionError(type(left))
        return self.shape_env.evaluate_expr(
            sympy.sympify(left),
            size_oblivious=size_oblivious,
            fallback_value=fallback_value,
        )

    def guard_or_false(self, left):
        """The claim, or false when it cannot be settled.

        Used where the caller has a way to carry on when the claim does not
        hold, so a claim it cannot settle is answered no rather than refused.
        """

        # A claim that is already settled -- a comparison of two numbers
        # rather than of two extents -- is answered with itself.  Only a
        # claim about extents needs deciding against what is known about
        # them, and treating a settled answer as one to be decided is how a
        # plain true becomes a question with no answer.
        if not isinstance(left, sympy.logic.boolalg.Boolean):
            if not isinstance(left, bool):
                raise AssertionError(f"Expected bool, got {type(left)}")
            return left


        if backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return False
        return self.evaluate_expr(left, fallback_value=False)

    def guard_or_true(self, left):
        """The claim, or true when it cannot be settled.

        The mirror of the above, for the caller that carries on when the claim
        holds rather than when it does not.
        """

        # A claim that is already settled -- a comparison of two numbers
        # rather than of two extents -- is answered with itself.  Only a
        # claim about extents needs deciding against what is known about
        # them, and treating a settled answer as one to be decided is how a
        # plain true becomes a question with no answer.
        if not isinstance(left, sympy.logic.boolalg.Boolean):
            if not isinstance(left, bool):
                raise AssertionError(f"Expected bool, got {type(left)}")
            return left


        if backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return True
        return self.evaluate_expr(left, fallback_value=True)

    def statically_known_equals(self, left: Expr | int, right: Expr | int) -> bool:
        """Whether it is sound to proceed as if the two were the same value."""

        return statically_known_true(
            self.shape_env, sympy.Eq(_size_expr(left), _size_expr(right))
        )

    def _is_multiple_of(self, numerator: Expr, denominator: int) -> bool:
        """Whether the numerator is provably a multiple of the denominator.

        The structure of the expression is walked first, because a sum whose
        terms are all multiples is one, a product with one such factor is one,
        and a division whose numerator is a multiple of a correspondingly
        larger denominator is one -- and each of those is a fact the
        environment does not have to be asked about at all.
        """

        if isinstance(numerator, (int, sympy.Integer)):
            return int(numerator) % denominator == 0

        if isinstance(numerator, sympy.Mul):
            for factor in numerator.args:
                if self._is_multiple_of(factor, denominator):
                    return True
            const = 1
            for factor in numerator.args:
                if isinstance(factor, (int, sympy.Integer)):
                    const *= int(factor)
            if const != 1 and const % denominator == 0:
                return True

        if isinstance(numerator, sympy.Add):
            if all(self._is_multiple_of(term, denominator) for term in numerator.args):
                return True

        if isinstance(numerator, FloorDiv):
            a, b = numerator.args
            if isinstance(b, (int, sympy.Integer)):
                if self._is_multiple_of(a, int(b) * denominator):
                    return True

        if isinstance(numerator, Mod):
            a, b = numerator.args
            if self._is_multiple_of(a, denominator) and self._is_multiple_of(
                b, denominator
            ):
                return True

        return statically_known_true(
            self.shape_env, sympy.Eq(sympy.Mod(numerator, denominator), 0)
        )

    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Expr | int
    ) -> bool:
        """Whether it is sound to treat the numerator as a multiple.

        Equality of the two is answered by structure rather than by building
        the equation out of them, because the equation is a measurable cost on
        a wide expression and the answer is already in front of us.
        """

        if numerator is denominator or numerator == denominator:
            return True

        if isinstance(denominator, (int, sympy.Integer)):
            return self._is_multiple_of(numerator, int(denominator))

        if numerator == 0:
            return True

        if len(free_symbols_of(numerator, denominator)) > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS:
            return False

        def gcd_covers_denominator(gcd: sympy.Basic) -> bool:
            return gcd != 1 and (
                gcd == denominator or self.simplify(gcd - denominator) == 0
            )

        try:
            if gcd_covers_denominator(simple_floordiv_gcd(numerator, denominator)):
                return True
            if gcd_covers_denominator(safe_gcd(numerator, denominator)):
                return True
        except sympy.PolynomialError:
            pass

        expr = sympy.Eq(sympy.Mod(numerator, denominator), 0)
        return statically_known_true(self.shape_env, expr)

Width = int | IntInfinity


@dataclasses.dataclass(frozen=True)
class LaneContiguity:
    """How an index expression moves from one lane of a vector to the next.

    This says only how the position varies across the lanes.  It does not say
    a vector load is allowed: that also needs the extent of the dimension, its
    stride, and the alignment of where the load starts to line up, and a caller
    that has not proved those must not act on this alone.  An unbounded width
    means the expression puts no limit on how many lanes hold the same value
    or run consecutively.
    """

    contiguous_width: Width | None = None
    uniform_width: Width | None = None
    stride: int | Expr | None = None
    unknown: bool = False

    @property
    def uniform(self) -> bool:
        """Whether every lane holds the same position."""

        return self.uniform_width is not None

    def is_contiguous_for(self, width: int) -> bool:
        """Whether the positions run consecutively for at least this many lanes."""

        return _width_covers(self.contiguous_width, width)

    def is_uniform_for(self, width: int) -> bool:
        """Whether the same position repeats for at least this many lanes."""

        return _width_covers(self.uniform_width, width)


def _width_covers(max_width: Width | None, width: int) -> bool:
    return max_width is not None and (max_width == int_oo or max_width >= width)


def _min_width(lhs: Width | None, rhs: Width | None) -> Width | None:
    if lhs is None or rhs is None:
        return None
    if lhs == int_oo:
        return rhs
    if rhs == int_oo:
        return lhs
    return min(lhs, rhs)


def _largest_power_of_2_factor(n: int) -> int:
    return n & -n


#: Above this many symbols the reasoning that scales badly with the number of
#: them is skipped: the polynomial conversions inside the algebra dominate the
#: time spent once the expressions get wide, and the answer is only a hint.
_MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS = 20


def join_dimensions(expr: Expr) -> Expr:
    """Merge the dimensions of a sum that are two views of the same position.

    A view of a view writes the same position twice, once as an offset within a
    group and once as which group it is.  Two such terms can be written as one,
    and doing so is what lets a later step see the position as a single
    expression rather than as a sum that has to be reasoned about term by term.
    """

    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    return _join_dimensions_cached(expr)


@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    """One pass of the dimension merging, on a sum that has a modular index in it.

    A position that is an offset within a group of a size, added to a multiple
    of that same size times which group it is, is one position written two
    ways -- as a wider modular index, or as a plain division once the two agree
    on the group size.
    """

    if not isinstance(expr, sympy.Add):
        raise AssertionError(f"expected sympy.Add, got {type(expr)}")

    scale = sympy.Wild("scale", exclude=[0], integer=True)
    base = sympy.Wild("base", integer=True)
    divisor = sympy.Wild("divisor", integer=True)
    mod1 = sympy.Wild("modulus", integer=True)
    mod2 = sympy.Wild("modulus2", integer=True)
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale]
                    * m1[mod1]
                    * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2)
                )
                if m2 and term1 != term2:
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale]
                        * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2])
                    )
                    return expr
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1])
                )
                if m2 is not None:
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * FloorDiv(m1[base], m1[divisor])
                    )
                    return expr
    return expr


def statically_known_true(
    shape_env: ShapeEnv,
    expr,
    axioms: tuple = None,
    var_to_range: tuple = None,
) -> bool:
    """Whether a relation holds, or can be shown to hold, without guarding.

    A hint that already makes the relation false settles it: something false
    under the values the shapes are known to have cannot be true always, and
    saying so avoids the far more expensive evaluation below.  A hint that
    makes it true settles nothing, because a value can satisfy it and others
    cannot, so that case still has to be reasoned about.
    """

    if expr in (True, False):
        return bool(expr)

    try:
        hinted = expr.xreplace(shape_env.backed_var_to_val)
        if hinted is sympy.S.false:
            return False
    except Exception:
        pass

    try:
        simplified = shape_env._maybe_evaluate_static(
            expr,
            axioms=axioms,
            var_to_range=var_to_range,
        )
        if simplified is not None:
            return bool(simplified)
    except Exception:
        log.debug("Could not simplify  %s", expr, exc_info=True)

    return False


def stride_at(index, var):
    """How much an index moves when a loop variable moves by one.

    The answer is the index with the variable advanced by one, less the index
    itself, which is the stride the loop is walking the buffer by.  An index
    that does not mention the variable at all has no stride with respect to it,
    which is zero rather than an error: the variable simply does not appear in
    that address.
    """

    if not index.has(var):
        # An index that does not mention the variable has no dependence on it,
        # which is a stride of zero rather than a question that cannot be asked.
        return sympy.S.Zero
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)
    return sympy.simplify(new_index - index)


@functools.lru_cache
def simplify_index_in_vec_range(index, var, vec_length: int):
    """The index in the form that is constant across a whole vector.

    A vectorized loop steps by a whole vector, so within one vector the variable
    runs over a known number of values and the parts of the index that divide by
    something the vector divides evenly do not change across it.  Those parts
    are replaced by names of their own, which is what makes it possible to ask
    whether the index is contiguous across a vector without knowing the values
    the variable takes.

    The result is for deciding, not for emitting: it names things that do not
    exist in the program.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    if not index.has(sympy.Rel):
        index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(index, var, vec_length: int | None = None):
    """The stride across a vector, which is the stride of the simplified index."""

    if vec_length:
        index = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index, var)


class SimplifyIndexing(WrapperHandler):
    """Every index expression put in its simplest form, given the axes it is over.

    An index like ``x // 4 * 4 + 3`` is a position in memory, and what it
    simplifies to depends on how far the axis actually goes: if it only reaches
    ten, most of the expression is always the same number.  Wrapping the
    operations so that each index is simplified as it arrives means that is
    worked out once per place rather than at every step of the reasoning about
    it.
    """

    def __init__(self, inner, var_ranges) -> None:
        super().__init__(inner)
        self.name = "SimplifyIndexing"
        self._simplify = (
            lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        )

    def load(self, name: str, index):
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)

    def value_expr(self, index, dtype):
        return self._inner.value_expr(self._simplify(index), dtype)

    def check_bounds(self, index, size, lower, upper):
        return self._inner.check_bounds(self._simplify(index), size, lower, upper)
