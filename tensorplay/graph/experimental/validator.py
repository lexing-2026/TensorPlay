from __future__ import annotations

import builtins
import functools
import math
import operator
from collections.abc import Callable
from typing import Any, TypeVar

import sympy

from .. import config
from ..interpreter import Interpreter
from ..node import Node
from .sym_node import SymNode

try:
    import z3 as _z3  # type: ignore[import-not-found]
except ImportError:
    _z3 = None

_R = TypeVar("_R")

__all__ = [
    "BisectValidationException",
    "PopulateValidator",
    "SympyToZ3",
    "TranslationValidator",
    "ValidationException",
    "bisect",
    "translation_validation_enabled",
    "translation_validation_timeout",
    "z3op",
    "z3str",
]


def z3str(value: Any) -> str:
    """Render a solver expression without requiring a solver installation."""

    sexpr = getattr(value, "sexpr", None)
    if callable(sexpr):
        return str(sexpr())
    return str(value)


def _as_expr(value: Any) -> Any:
    if isinstance(value, SymNode):
        return sympy.sympify(value.expr)
    if isinstance(value, sympy.Basic):
        return value
    if isinstance(value, bool):
        return sympy.true if value else sympy.false
    if isinstance(value, (int, float)):
        return sympy.sympify(value)
    return value


def _apply_operator(op: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    if op in {operator.and_, operator.or_}:
        fn = sympy.And if op is operator.and_ else sympy.Or
        return fn(*args)
    if op is operator.not_:
        return sympy.Not(args[0])
    if op is operator.xor:
        return sympy.Xor(*args)
    if op is min:
        return sympy.Min(*args)
    if op is max:
        return sympy.Max(*args)
    if op is abs or op is operator.abs:
        return sympy.Abs(args[0])
    if op is math.floor:
        return sympy.floor(args[0])
    if op is math.ceil:
        return sympy.ceiling(args[0])
    if op is math.trunc:
        return sympy.floor(args[0]) if args[0].is_nonnegative else sympy.ceiling(args[0])
    if op is builtins.round:
        if len(args) == 1:
            return sympy.floor(args[0] + sympy.Rational(1, 2))
        return sympy.floor(args[0] * 10 ** args[1] + sympy.Rational(1, 2)) / 10 ** args[1]
    return op(*args)


def z3op(op: Callable[..., Any], validator: "TranslationValidator") -> Callable[..., Any]:
    """Lift a scalar callable into the validator expression domain."""

    @functools.wraps(op)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        values = tuple(_as_expr(value) for value in args)
        keyword_values = {key: _as_expr(value) for key, value in kwargs.items()}
        try:
            return _apply_operator(op, values) if not keyword_values else op(*values, **keyword_values)
        except (TypeError, ValueError):
            return op(*values, **keyword_values)

    del validator
    return wrapper


class PopulateValidator(Interpreter):
    """Evaluate graph scalar expressions into a validation context."""

    def __init__(self, graph: Any, validator: "TranslationValidator") -> None:
        self.validator = validator
        module = getattr(graph, "owning_module", None)
        if module is None:
            from ..graph_module import GraphModule

            module = GraphModule({}, graph)
        super().__init__(module, garbage_collect_values=False, graph=graph)

    def placeholder(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        del args, kwargs
        node = self.last_node
        symbol = node.meta.get("symbol") if node is not None else None
        if not isinstance(symbol, sympy.Symbol):
            symbol = sympy.Symbol(str(target), integer=True)
        return self.validator.add_var(symbol, int)

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if getattr(target, "_tensorplay_assertion", False):
            if len(args) != 1:
                raise ValueError("assertion target requires one argument")
            self.validator.add_source_expr(args[0])
            return None
        return z3op(target, self.validator)(*args, **kwargs)


class SympyToZ3:
    """Translate symbolic expressions to the active validation representation."""

    OPERATOR_HANDLES = {"add", "mul", "eq", "ne", "lt", "gt", "le", "ge"}

    def __init__(self, validator: "TranslationValidator") -> None:
        self._validator = validator

    def constant(self, value: int | float | bool, dtype: type) -> Any:
        del dtype
        return _as_expr(value)

    def to_dtype(self, value: Any, dtype: type) -> Any:
        if dtype is float:
            return sympy.Float(value)
        if dtype is int:
            return sympy.Integer(value)
        return value

    def trunc_to_int(self, value: Any, dtype: type) -> Any:
        del dtype
        return sympy.floor(value)

    def round_to_int(self, value: Any, dtype: type) -> Any:
        del dtype
        return _apply_operator(round, (value,))

    def int_truediv(self, numerator: Any, denominator: Any) -> Any:
        return numerator / denominator

    truediv = int_truediv

    def floordiv(self, numerator: Any, denominator: Any) -> Any:
        return sympy.floor(numerator / denominator)

    div = floordiv

    def pow(self, base: Any, exponent: Any) -> Any:
        return base**exponent

    pow_by_natural = pow

    def mod(self, left: Any, right: Any) -> Any:
        return sympy.Mod(left, right)

    python_mod = mod

    def ceil_to_int(self, value: Any, dtype: type) -> Any:
        del dtype
        return sympy.ceiling(value)

    floor_to_int = trunc_to_int

    def __getattr__(self, name: str) -> Any:
        replacements = {
            "and_": sympy.And,
            "or_": sympy.Or,
            "not_": sympy.Not,
            "minimum": sympy.Min,
            "maximum": sympy.Max,
            "bitwise_and": operator.and_,
            "bitwise_or": operator.or_,
            "lshift": operator.lshift,
            "rshift": operator.rshift,
            "floor": sympy.floor,
            "ceil": sympy.ceiling,
        }
        if name in replacements:
            return replacements[name]
        if name in self.OPERATOR_HANDLES:
            return getattr(operator, name)
        raise AttributeError(name)

    def run(self, expr: sympy.Basic) -> Any:
        return sympy.sympify(expr)


class TranslationValidator:
    """Check that target guard expressions imply source guard expressions."""

    def __init__(self) -> None:
        self.symbols: dict[sympy.Symbol, Any] = {}
        self._source_exprs: set[Any] = set()
        self._target_exprs: set[Any] = set()
        self._assertions: set[Any] = set()

    def z3var(self, symbol: sympy.Symbol) -> Any:
        if symbol not in self.symbols:
            raise KeyError(f"symbol {symbol} is not registered")
        return self.symbols[symbol]

    def add_var(self, symbol: sympy.Symbol, type: type) -> Any:
        if symbol in self.symbols:
            return self.symbols[symbol]
        if type is int:
            value = sympy.Symbol(str(symbol), integer=True)
        elif type is float:
            value = sympy.Symbol(str(symbol), real=True)
        elif type is bool:
            value = sympy.Symbol(str(symbol), boolean=True)
        else:
            raise TypeError(f"unsupported symbolic type {type!r}")
        self.symbols[symbol] = value
        return value

    def _check_free_symbols(self, expr: Any) -> None:
        for symbol in sympy.sympify(expr).free_symbols:
            self.symbols.setdefault(symbol, symbol)

    def to_z3_boolean_expr(self, expr: Any) -> Any:
        result = SympyToZ3(self).run(sympy.sympify(expr))
        if result in (sympy.true, sympy.false) or getattr(result, "is_Boolean", False):
            return result
        raise TypeError(f"expected boolean expression, got {result!r}")

    def add_source_expr(self, expr: Any) -> None:
        self._check_free_symbols(expr)
        self._source_exprs.add(self.to_z3_boolean_expr(expr))

    def add_target_expr(self, expr: Any) -> None:
        self._check_free_symbols(expr)
        self._target_exprs.add(self.to_z3_boolean_expr(expr))

    def add_assertion(self, expr: Any) -> None:
        self._check_free_symbols(expr)
        self._assertions.add(self.to_z3_boolean_expr(expr))

    def validate(self) -> None:
        if not self._source_exprs or not self._target_exprs:
            return
        source = sympy.And(*self._source_exprs)
        target = sympy.And(*self._target_exprs)
        assumptions = sympy.And(*self._assertions) if self._assertions else sympy.true
        counterexample = sympy.And(assumptions, target, sympy.Not(source))
        simplified = sympy.simplify_logic(counterexample, force=True)
        if simplified is sympy.false:
            return
        try:
            model = sympy.satisfiable(counterexample)
        except Exception as exc:
            raise RuntimeError("symbolic guard validation could not be decided") from exc
        if model is False:
            return
        failed = [expr for expr in self._source_exprs if model.get(expr, expr) is not True]
        raise ValidationException(model, self._assertions, self._target_exprs, failed)


def translation_validation_enabled() -> bool:
    enabled = bool(getattr(config, "translation_validation", False))
    if enabled and _z3 is None:
        return True
    return enabled


def translation_validation_timeout() -> int:
    return int(getattr(config, "translation_validation_timeout", 0))


class ValidationException(RuntimeError):
    def __init__(
        self,
        model: Any,
        assertions: Any,
        target_exprs: Any,
        failed_source_exprs: Any,
    ) -> None:
        self.model = model
        self.assertions = assertions
        self.target_exprs = target_exprs
        self.failed_source_exprs = failed_source_exprs
        super().__init__(self._render())

    def _render(self) -> str:
        def lines(values: Any) -> str:
            return "\n".join(f"  ==> {value}" for value in sorted(map(str, values)))

        return (
            "symbolic validation failed.\n\n"
            f"Model:\n{lines(self.model.items() if hasattr(self.model, 'items') else self.model)}\n\n"
            f"Assertions:\n{lines(self.assertions)}\n\n"
            f"Target expressions:\n{lines(self.target_exprs)}\n\n"
            f"Failed source expressions:\n{lines(self.failed_source_exprs)}"
        )


class BisectValidationException(RuntimeError):
    def __init__(
        self,
        validation_exc: ValidationException,
        expr: Any,
        failed_action: str,
        traced_node: Node,
    ) -> None:
        self.validation_exc = validation_exc
        self.expr = expr
        self.failed_action = failed_action
        self.traced_node = traced_node
        super().__init__(
            f"symbolic validation failed while {failed_action}: {expr}\n\n"
            f"Failure occurred at node:\n    {traced_node.format_node()}\n\n"
            f"{validation_exc}"
        )


def bisect(shape_env: Any) -> None:
    """Replay recorded state events and report the first failing validation."""

    events = list(getattr(shape_env, "events", ()))
    if not events:
        return
    validator = getattr(shape_env, "validator", None)
    if isinstance(validator, TranslationValidator):
        validator.validate()
        return
    check = getattr(shape_env, "validate", None)
    if callable(check):
        check()

