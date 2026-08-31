from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Any

import sympy

from .symbolic_shapes import GuardOnDataDependentSymNode, has_free_unbacked_symbols

log = logging.getLogger(__name__)

SYMPY_FACTOR_MAX_FREE_SYMBOLS = 50
SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT = 1024

__all__ = [
    "CanonicalExprFinder",
    "SYMPY_FACTOR_MAX_FREE_SYMBOLS",
    "SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT",
    "_get_unbacked_replacements",
    "_guarding_hint_or_throw_base",
    "_hint_bounds_from_runtime_asserts",
    "_maybe_realize_expr",
    "_optimization_hint_base",
    "_sub_unbacked_exprs",
    "_sympy_subs",
]


def to_symbol(replaced: sympy.Expr, replacement: sympy.Expr | str) -> sympy.Expr:
    if not isinstance(replaced, sympy.Expr):
        raise AssertionError(f"expected a symbolic expression, got {type(replaced)}")
    if isinstance(replacement, str):
        return sympy.Symbol(
            replacement,
            integer=getattr(replaced, "is_integer", None),
            nonnegative=getattr(replaced, "is_nonnegative", None),
        )
    return replacement


class CanonicalExprFinder:
    def __init__(self, eq_graph: dict[sympy.Expr, Iterable[sympy.Expr]]) -> None:
        self.eq_graph = {key: set(value) for key, value in eq_graph.items()}
        self.expressions = list(self.eq_graph)
        self.reverse_expressions = {
            expression: index for index, expression in enumerate(self.expressions)
        }
        self.leader = list(range(len(self.expressions)))
        self.size = [1] * len(self.expressions)
        self._build_canonical_expr_mapping()

    def _build_canonical_expr_mapping(self) -> None:
        for expression, edges in self.eq_graph.items():
            for adjacent in edges:
                self.union_expr(expression, adjacent)

    def union_expr(self, first: sympy.Expr, second: sympy.Expr) -> bool:
        if first not in self.reverse_expressions:
            self.reverse_expressions[first] = len(self.expressions)
            self.expressions.append(first)
            self.leader.append(len(self.leader))
            self.size.append(1)
        if second not in self.reverse_expressions:
            self.reverse_expressions[second] = len(self.expressions)
            self.expressions.append(second)
            self.leader.append(len(self.leader))
            self.size.append(1)
        return self.union(self.reverse_expressions[first], self.reverse_expressions[second])

    def union(self, first: int, second: int) -> bool:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False
        leader, follower = self.choose_leader(root_first, root_second)
        self.leader[follower] = leader
        self.size[leader] += self.size[follower]
        return True

    def find(self, index: int) -> int:
        if self.leader[index] != index:
            self.leader[index] = self.find(self.leader[index])
        return self.leader[index]

    def find_expr(self, expression: sympy.Expr) -> sympy.Expr:
        return self.expressions[self.find(self.reverse_expressions[expression])]

    def choose_leader(self, first: int, second: int) -> tuple[int, int]:
        left = self.expressions[first]
        right = self.expressions[second]
        left_unbacked = has_free_unbacked_symbols(left)
        right_unbacked = has_free_unbacked_symbols(right)
        if left_unbacked != right_unbacked:
            return (second, first) if left_unbacked else (first, second)
        if left.has(right):
            return second, first
        if right.has(left):
            return first, second
        left_degree = len(self.eq_graph.get(left, ()))
        right_degree = len(self.eq_graph.get(right, ()))
        if left_degree != right_degree:
            return (first, second) if left_degree > right_degree else (second, first)
        if self.size[first] != self.size[second]:
            return (first, second) if self.size[first] > self.size[second] else (second, first)
        return (first, second) if left.compare(right) == -1 else (second, first)


def _sympy_subs(
    expr: sympy.Basic, replacements: dict[sympy.Expr, Any]
) -> sympy.Basic:
    converted: dict[sympy.Expr, Any] = {}
    for key, value in replacements.items():
        if isinstance(value, str):
            converted[key] = sympy.Symbol(
                value,
                integer=getattr(key, "is_integer", None),
                nonnegative=getattr(key, "is_nonnegative", None),
            )
        else:
            converted[key] = value
    return sympy.sympify(expr).xreplace(converted)


def _maybe_realize_expr(
    expr: sympy.Basic, nan_fallback: int | None
) -> int | bool | None:
    expr = sympy.sympify(expr)
    if expr is sympy.true:
        return True
    if expr is sympy.false:
        return False
    if expr in (sympy.oo, -sympy.oo):
        return sys.maxsize if expr is sympy.oo else -sys.maxsize
    if expr is sympy.nan or expr.has(sympy.nan):
        return nan_fallback
    if expr.has(sympy.I):
        raise ValueError(f"symbolic expression is complex: {expr}")
    if not expr.free_symbols and getattr(expr, "is_integer", False):
        return int(expr)
    if not expr.free_symbols and getattr(expr, "is_float", False):
        return float(expr)  # type: ignore[return-value]
    return None


def _guarding_hint_or_throw_base(
    shape_env: Any,
    expr: sympy.Expr | sympy.Basic | int | bool,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol] | None = None,
) -> int | bool:
    current = shape_env.replace(expr)
    if precomputed_replacements:
        current = _sympy_subs(current, precomputed_replacements)
    current = _sympy_subs(current, getattr(shape_env, "backed_var_to_val", {}))
    realized = _maybe_realize_expr(current, None)
    if realized is not None:
        return realized
    if has_free_unbacked_symbols(current):
        raise GuardOnDataDependentSymNode(
            current, f"cannot resolve symbolic guard {current}"
        )
    current = _sympy_subs(current, getattr(shape_env, "var_to_hint_override", {}))
    realized = _maybe_realize_expr(current, None)
    if realized is None:
        raise GuardOnDataDependentSymNode(
            current, f"cannot resolve symbolic guard {current}"
        )
    return realized


def _get_unbacked_replacements(shape_env: Any) -> dict[sympy.Expr, sympy.Expr]:
    cached = getattr(shape_env, "_unbacked_replacements", None)
    if cached is not None:
        return cached
    graph: dict[sympy.Expr, set[sympy.Expr]] = {}
    for assertions in getattr(shape_env, "deferred_runtime_asserts", {}).values():
        for assertion in assertions:
            expression = getattr(assertion, "expr", assertion)
            if not isinstance(expression, sympy.Equality):
                continue
            left, right = sympy.sympify(expression.lhs), sympy.sympify(expression.rhs)
            if not (has_free_unbacked_symbols(left) or has_free_unbacked_symbols(right)):
                continue
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)

    finder = CanonicalExprFinder(graph)
    replacements: dict[sympy.Expr, sympy.Expr] = {}
    for expression in finder.expressions:
        canonical = finder.find_expr(expression)
        if expression != canonical:
            replacements[expression] = canonical
    shape_env._unbacked_replacements = replacements
    return replacements


@lru_cache(maxsize=1024)
def _sub_unbacked_exprs(shape_env: Any, expr: sympy.Expr) -> sympy.Expr:
    current = sympy.sympify(expr)
    replacements = _get_unbacked_replacements(shape_env)
    for _ in range(30):
        updated = current.xreplace(replacements)
        if updated == current:
            break
        current = sympy.factor(updated) if len(updated.free_symbols) <= SYMPY_FACTOR_MAX_FREE_SYMBOLS else updated
    current = _sympy_subs(current, getattr(shape_env, "backed_var_to_val", {}))
    current = _sympy_subs(current, getattr(shape_env, "var_to_hint_override", {}))
    return current


def _hint_bounds_from_runtime_asserts(
    shape_env: Any, symbol: sympy.Symbol
) -> tuple[int | None, int | None]:
    lower: int | None = None
    upper: int | None = None
    backed = getattr(shape_env, "backed_var_to_val", {})
    for assertion in getattr(shape_env, "deferred_runtime_asserts", {}).get(symbol, ()):
        expression = sympy.sympify(getattr(assertion, "expr", assertion)).xreplace(backed)
        if symbol not in expression.free_symbols:
            continue
        if isinstance(expression, (sympy.LessThan, sympy.StrictLessThan)):
            difference = expression.lhs - expression.rhs
            strict = isinstance(expression, sympy.StrictLessThan)
        elif isinstance(expression, (sympy.GreaterThan, sympy.StrictGreaterThan)):
            difference = expression.rhs - expression.lhs
            strict = isinstance(expression, sympy.StrictGreaterThan)
        else:
            continue
        coefficient = difference.coeff(symbol)
        constant = difference.subs(symbol, 0)
        if not (coefficient.is_Integer and constant.is_Integer) or coefficient == 0:
            continue
        threshold = -sympy.Rational(constant, coefficient)
        if coefficient > 0:
            bound = int(sympy.ceiling(threshold) - 1) if strict else int(sympy.floor(threshold))
            upper = bound if upper is None else min(upper, bound)
        else:
            bound = int(sympy.floor(threshold) + 1) if strict else int(sympy.ceiling(threshold))
            lower = bound if lower is None else max(lower, bound)
    return lower, upper


def _optimization_hint_base(
    shape_env: Any,
    expr: sympy.Expr | int,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol] | None = None,
    fallback: int | None = None,
) -> int:
    fallback = int(
        fallback
        if fallback is not None
        else getattr(shape_env, "unbacked_symint_fallback", 2)
    )
    original = sympy.sympify(expr)
    current = shape_env.replace(original)
    if precomputed_replacements:
        current = _sympy_subs(current, precomputed_replacements)
    current = _sympy_subs(current, getattr(shape_env, "backed_var_to_val", {}))
    current = _sympy_subs(current, getattr(shape_env, "var_to_hint_override", {}))
    realized = _maybe_realize_expr(current, fallback)
    if realized is not None:
        return int(realized)
    current = _sub_unbacked_exprs(shape_env, original)
    substitutions: dict[sympy.Symbol, int] = {}
    for symbol in current.free_symbols:
        if not getattr(shape_env, "is_unbacked_symint", lambda _: str(symbol).startswith("u"))(symbol):
            continue
        lower, upper = _hint_bounds_from_runtime_asserts(shape_env, symbol)
        value = fallback
        ranges = getattr(shape_env, "var_to_range", {}).get(symbol)
        if ranges is not None:
            if isinstance(ranges.lower, (int, sympy.Integer)):
                value = max(value, int(ranges.lower))
            if isinstance(ranges.upper, (int, sympy.Integer)):
                value = min(value, int(ranges.upper))
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        substitutions[symbol] = value
    realized = _maybe_realize_expr(current.xreplace(substitutions), fallback)
    if realized is None:
        raise RuntimeError(f"failed to produce a size hint for {expr}")
    return int(realized)
