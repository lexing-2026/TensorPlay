"""Symbolic sizes for the input dimensions an export declares dynamic.

A dimension declared dynamic is not fixed by the example the program is
captured with, so reading it must not hand the program the example's number.
Each such dimension stands for a symbol.  Reading it yields a value of the
graph whose example is that symbol, carrying the example size as its hint;
arithmetic on it stays symbolic, and a Python decision on it -- a branch, an
``int()`` -- is taken from the hint and kept as a guard: a condition the
captured program only holds under.

Once capture ends every guard is checked against the declared ranges.  A guard
the ranges imply is dropped.  One they do not imply is an error for a named
:class:`Dim` (with the fix that would make the declaration true), an error for
``Dim.DYNAMIC`` when it fixes the dimension to one size, and otherwise becomes a
runtime assertion of the captured program.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import sympy
from sympy.printing.pycode import pycode

from ..graph.experimental.sym_node import SymNode
from ..graph.experimental.symbolic_shapes import ShapeEnv
from ..graph.experimental.sympy_functions import ValueRanges, int_oo
from ..graph.node import Node
from .dynamic_shapes import (
    ConstraintsExceededError,
    Dim,
    _DerivedDim,
    _DimHint,
    _DimHintType,
    _StaticDim,
    _linear_name,
)

#: A dimension that varies is taken to be at least this large when deciding
#: whether a guard is one the declaration is meant to allow: guards that only
#: exclude sizes 0 and 1 (``x.shape[0] != 1``, ``x.shape[0] > 1``) are kept as
#: runtime assertions instead of rejecting the declaration.
_SIZE_FLOOR = 2


def _bounded(value: Any) -> bool:
    return value not in (sympy.oo, int_oo)


def render(fact: sympy.Basic) -> str:
    """``fact`` as the Python condition it stands for."""

    text = pycode(fact).replace("math.", "")
    depth = 0
    for index, char in enumerate(text):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0 and index < len(text) - 1:
            return text
    return text[1:-1] if text.startswith("(") else text


@dataclasses.dataclass(frozen=True)
class _Symbol:
    symbol: sympy.Symbol
    #: How messages name the dimension: the Dim's name, or ``x.size()[0]``.
    label: str
    #: The named Dim (the root of a derived one) or the dim hint behind it.
    spec: Any

    @property
    def named(self) -> bool:
        return isinstance(self.spec, Dim)


@dataclasses.dataclass(frozen=True)
class Guard:
    """A condition the captured program was traced under."""

    fact: sympy.Basic
    #: The graph value the decision was taken on.
    node: Node
    #: What that value was on the example; the program holds while it still is.
    expected: Any


class SymbolicDims:
    """The symbols behind an export's dynamic dimensions and the guards on them."""

    def __init__(self) -> None:
        self.env = ShapeEnv(specialize_zero_one=False, duck_shape=False)
        self.symbols: dict[sympy.Symbol, _Symbol] = {}
        #: input name -> axis -> (size expression, example size)
        self.sites: dict[str, dict[int, tuple[sympy.Expr, int]]] = {}
        self.guards: list[Guard] = []

    @classmethod
    def from_spec(
        cls, inputs: Mapping[str, Any], dynamic_shapes: Mapping[str, Any]
    ) -> SymbolicDims | None:
        """Symbols for every dimension ``dynamic_shapes`` leaves free.

        ``dynamic_shapes`` is the normalized specification, keyed by argument
        name.  Only top-level tensor arguments are sized symbolically; the
        extents of tensors nested in containers are read as graph values
        already.
        """

        dims = cls()
        for name, spec in (dynamic_shapes or {}).items():
            value = inputs.get(name)
            shape = getattr(value, "shape", None)
            if shape is None or spec is None or not hasattr(value, "numpy"):
                continue
            sizes = tuple(int(size) for size in shape)
            entries = spec.items() if isinstance(spec, dict) else enumerate(spec)
            for axis, dim in entries:
                expr = dims._expression(name, axis, dim)
                if expr is not None:
                    dims.sites.setdefault(name, {})[axis] = (expr, sizes[axis])
        return dims if dims.sites else None

    def _expression(self, name: str, axis: int, dim: Any) -> sympy.Expr | None:
        if dim is None or isinstance(dim, (int, _StaticDim)):
            return None
        if isinstance(dim, _DimHint):
            if dim.type is _DimHintType.STATIC:
                return None
            return self._symbol(f"{name}.size()[{axis}]", dim, dim.min, dim.max)
        if isinstance(dim, _DerivedDim):
            root = self._symbol(dim.root.__name__, dim.root, dim.root.min, dim.root.max)
            return dim.scale * root + dim.offset
        if isinstance(dim, Dim):
            return self._symbol(dim.__name__, dim, dim.min, dim.max)
        return None

    def _symbol(self, label: str, spec: Any, lower: int | None, upper: int | None) -> sympy.Symbol:
        symbol = sympy.Symbol(label, integer=True, nonnegative=True)
        if symbol not in self.symbols:
            self.symbols[symbol] = _Symbol(symbol, label, spec)
            self.env.var_to_range[symbol] = ValueRanges(
                0 if lower is None else lower, sympy.oo if upper is None else upper
            )
        return symbol

    # -- capture ------------------------------------------------------------

    def decide(self, node: Node, value: SymNode, kind: str) -> Any:
        """Answer a Python decision on ``value`` from its hint and keep it."""

        hint = value.hint
        expr = value.expr
        if kind == "bool":
            outcome = bool(hint)
            if value.pytype is not bool:
                expr = sympy.Ne(expr, 0)
            fact = expr if outcome else sympy.Not(expr)
        else:
            outcome = float(hint) if kind == "float" else int(hint)
            fact = sympy.Eq(expr, outcome)
        if fact is not sympy.true:
            self.guards.append(Guard(fact, node, outcome))
        return outcome

    # -- after capture --------------------------------------------------------

    def settle(self) -> list[Guard]:
        """Check every guard against the declaration.

        Returns the guards the captured program must assert at runtime and
        raises :class:`ConstraintsExceededError` for any the declaration
        forbids.
        """

        violated: list[str] = []
        problems: list[str] = []
        fixes: dict[str, str] = {}
        held: list[Guard] = []
        for guard in self.guards:
            fact = guard.fact
            if self._decide(fact) is True:
                continue
            symbols = [self.symbols[s] for s in fact.free_symbols if s in self.symbols]
            if self._decide(fact, floor=_SIZE_FLOOR) is not True:
                named = [symbol for symbol in symbols if symbol.named]
                if named:
                    violated.extend(symbol.label for symbol in named)
                    problems.append(self._describe(fact, named))
                    fixes.update(self._suggest(fact))
                    continue
                fixed = self._fixed_value(fact)
                strict = [
                    symbol
                    for symbol in symbols
                    if symbol.spec.type is _DimHintType.DYNAMIC
                ]
                if fixed is not None and strict:
                    violated.extend(symbol.label for symbol in strict)
                    problems.append(
                        f"  - {strict[0].label} was declared Dim.DYNAMIC, but the "
                        f"program fixes it to {fixed} (it was captured under "
                        f"{render(fact)}); declare it static, or Dim.AUTO to let "
                        "export decide"
                    )
                    continue
            self._narrow(fact)
            held.append(guard)
        if problems:
            message = (
                f"Constraints violated ({', '.join(sorted(set(violated)))})! The "
                "captured program holds only for some of the declared sizes:\n"
                + "\n".join(problems)
            )
            if fixes:
                message += "\nSuggested fixes:\n" + "\n".join(
                    f"    {name} = {fix}" for name, fix in fixes.items()
                )
            raise ConstraintsExceededError(message)
        return held

    def _ranges(self, floor: int = 0) -> tuple[tuple[sympy.Symbol, ValueRanges], ...]:
        return tuple(
            (
                symbol,
                ValueRanges(max(known.lower, floor), known.upper)
                if known.upper >= floor
                else known,
            )
            for symbol, known in self.env.var_to_range.items()
        )

    def _decide(self, fact: sympy.Basic, floor: int = 0) -> bool | None:
        """Whether the declared ranges (each at least ``floor``) settle ``fact``."""

        if fact in (sympy.true, sympy.false):
            return bool(fact)
        if isinstance(fact, sympy.Ne):
            equal = self._decide(sympy.Eq(*fact.args), floor)
            return None if equal is None else not equal
        if isinstance(fact, (sympy.And, sympy.Or)):
            parts = [self._decide(part, floor) for part in fact.args]
            if isinstance(fact, sympy.And):
                return False if False in parts else None if None in parts else True
            return True if True in parts else None if None in parts else False
        result = self.env._maybe_evaluate_static(fact, var_to_range=self._ranges(floor))
        if result is None:
            return None
        return bool(result)

    def _fixed_value(self, fact: sympy.Basic) -> int | None:
        if not isinstance(fact, sympy.Eq) or len(fact.free_symbols) != 1:
            return None
        (symbol,) = fact.free_symbols
        solutions = sympy.solve(fact, symbol)
        if len(solutions) == 1 and solutions[0].is_integer:
            return int(solutions[0])
        return None

    def _interval(self, fact: sympy.Basic) -> tuple[sympy.Symbol, int, Any] | None:
        """The bounds ``fact`` puts on its one symbol, if it is a plain bound."""

        if not isinstance(fact, (sympy.Lt, sympy.Le, sympy.Gt, sympy.Ge)):
            return None
        if len(fact.free_symbols) != 1:
            return None
        (symbol,) = fact.free_symbols
        try:
            interval = sympy.solve_univariate_inequality(fact, symbol, relational=False)
        except (NotImplementedError, ValueError, TypeError):
            return None
        if not isinstance(interval, sympy.Interval):
            return None
        lower, upper = interval.start, interval.end
        if _bounded(lower):
            lower = int(sympy.ceiling(lower)) + (
                1 if interval.left_open and lower.is_integer else 0
            )
        else:
            lower = 0
        if _bounded(upper):
            upper = int(sympy.floor(upper)) - (
                1 if interval.right_open and upper.is_integer else 0
            )
        else:
            upper = int_oo
        return symbol, lower, upper

    def _narrow(self, fact: sympy.Basic) -> None:
        bound = self._interval(fact)
        if bound is None:
            return
        symbol, lower, upper = bound
        known = self.env.var_to_range[symbol]
        lower = max(known.lower, lower)
        upper = upper if not _bounded(known.upper) else min(known.upper, upper)
        if lower <= upper:
            self.env.var_to_range[symbol] = ValueRanges(lower, upper)

    def _describe(self, fact: sympy.Basic, named: list[_Symbol]) -> str:
        spans = ", ".join(
            f"{symbol.label} in {self._render_range(symbol)}" for symbol in named
        )
        return (
            f"  - Not all values of {spans} satisfy the guard {render(fact)}, "
            "which the program was captured under"
        )

    def _render_range(self, symbol: _Symbol) -> str:
        known = self.env.var_to_range[symbol.symbol]
        upper = known.upper if _bounded(known.upper) else "inf"
        return f"[{known.lower}, {upper}]"

    def _suggest(self, fact: sympy.Basic) -> dict[str, str]:
        """Declaration fixes that make ``fact`` hold for every declared size."""

        symbols = sorted(fact.free_symbols, key=str)
        if len(symbols) == 1:
            symbol = symbols[0]
            name = self.symbols[symbol].label
            fixed = self._fixed_value(fact)
            if fixed is not None:
                return {name: str(fixed)}
            bound = self._interval(fact)
            if bound is None:
                return {}
            _, lower, upper = bound
            known = self.env.var_to_range[symbol]
            lower = max(int(known.lower), lower)
            if _bounded(known.upper):
                upper = min(int(known.upper), upper)
            maximum = int(upper) if _bounded(upper) else None
            return {name: repr(Dim(name, min=lower, max=maximum))}
        if len(symbols) == 2 and isinstance(fact, sympy.Eq):
            for derived, root in (symbols, symbols[::-1]):
                solutions = sympy.solve(fact, derived)
                if len(solutions) != 1:
                    continue
                expression = sympy.Poly(solutions[0], root)
                if expression.degree() != 1:
                    continue
                scale, offset = expression.all_coeffs()
                if scale.is_integer and scale > 0 and offset.is_integer:
                    root_name = self.symbols[root].label
                    return {
                        self.symbols[derived].label: _linear_name(
                            root_name, int(scale), int(offset)
                        )
                    }
        return {}
