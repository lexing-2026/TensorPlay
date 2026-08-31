from __future__ import annotations

import math
import operator
from typing import Any, Callable, cast

import sympy


def _type_name(kind: type) -> str:
    return {int: "int", float: "float", bool: "bool"}.get(kind, kind.__name__)


def _to_symtype(kind: type) -> type:
    return {int: SymInt, float: SymFloat, bool: SymBool}.get(kind, kind)


def _as_expr(value: Any) -> sympy.Basic:
    if isinstance(value, SymNode):
        value = value.expr
    return sympy.sympify(value)


def _primitive(value: Any) -> Any:
    if value in (sympy.true, sympy.false):
        return bool(value)
    if isinstance(value, sympy.Integer):
        return int(value)
    if isinstance(value, sympy.Float):
        return float(value)
    return value


def _binary_expr(
    symbol: str,
    left: Any,
    right: Any,
    function: Callable[[Any, Any], Any],
) -> sympy.Basic:
    left_expr = _as_expr(left)
    right_expr = _as_expr(right)
    try:
        if symbol == "+":
            return left_expr + right_expr
        if symbol == "-":
            return left_expr - right_expr
        if symbol == "*":
            return left_expr * right_expr
        if symbol == "%":
            return sympy.Mod(left_expr, right_expr)
        if symbol in {"**", "pow"}:
            return left_expr**right_expr
        if symbol == "/":
            return left_expr / right_expr
        if symbol == "int_truediv":
            return sympy.floor(left_expr / right_expr)
        if symbol == "//":
            return sympy.floor(left_expr / right_expr)
        if symbol == "==":
            return sympy.Eq(left_expr, right_expr)
        if symbol == "!=":
            return sympy.Ne(left_expr, right_expr)
        if symbol == ">":
            return sympy.Gt(left_expr, right_expr)
        if symbol == "<":
            return sympy.Lt(left_expr, right_expr)
        if symbol == "<=":
            return sympy.Le(left_expr, right_expr)
        if symbol == ">=":
            return sympy.Ge(left_expr, right_expr)
        if symbol == "and":
            return sympy.And(left_expr, right_expr)
        if symbol == "or":
            return sympy.Or(left_expr, right_expr)
        if symbol == "xor":
            return sympy.Xor(left_expr, right_expr)
        if symbol == "min":
            return sympy.Min(left_expr, right_expr)
        if symbol == "max":
            return sympy.Max(left_expr, right_expr)
        if symbol == "<<":
            return sympy.Function("lshift")(left_expr, right_expr)
        if symbol == ">>":
            return sympy.Function("rshift")(left_expr, right_expr)
        return sympy.sympify(function(left_expr, right_expr))
    except (TypeError, ValueError):
        return sympy.Function(symbol.replace(" ", "_"))(left_expr, right_expr)


def _unary_expr(
    symbol: str,
    value: Any,
    function: Callable[[Any], Any],
) -> sympy.Basic:
    expr = _as_expr(value)
    try:
        if symbol == "-":
            return -expr
        if symbol == "+":
            return +expr
        if symbol == "abs":
            return sympy.Abs(expr)
        if symbol == "floor":
            return sympy.floor(expr)
        if symbol == "ceil":
            return sympy.ceiling(expr)
        if symbol == "trunc":
            return sympy.Function("trunc")(expr)
        if symbol == "not":
            return sympy.Not(expr)
        if symbol == "is_integer":
            return sympy.Function("is_integer")(expr)
        if symbol == "float":
            return sympy.Float(expr)
        if symbol == "int":
            return sympy.floor(expr)
        return sympy.sympify(function(expr))
    except (TypeError, ValueError):
        return sympy.Function(symbol.replace(" ", "_"))(expr)


class SymNode:
    """A symbolic scalar with an optional concrete hint and evaluator."""

    __slots__ = ("_expr", "shape_env", "pytype", "_hint", "_evaluate")

    def __init__(
        self,
        expr: Any,
        shape_env: Any = None,
        pytype: type = int,
        hint: Any = None,
        evaluate: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        self._hint = hint
        self._evaluate = evaluate

    @property
    def expr(self) -> Any:
        return self._expr

    @property
    def hint(self) -> Any:
        return self._hint

    def has_hint(self) -> bool:
        return self._hint is not None

    def with_shape_env(self, shape_env: Any) -> "SymNode":
        return type(self)(
            self._expr,
            shape_env,
            self.pytype,
            self._hint,
            self._evaluate,
        )

    def _value_eq(self, other: object) -> bool:
        return (
            isinstance(other, SymNode)
            and self.pytype is other.pytype
            and self._expr == other._expr
            and self.shape_env is other.shape_env
        )

    def _value_hash(self) -> int:
        return hash((self.pytype, repr(self._expr), id(self.shape_env)))

    def __hash__(self) -> int:
        return self._value_hash()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymNode):
            return self.eq(other)
        return self.eq(other)

    def __bool__(self) -> bool:
        return bool(self.guard_bool())

    def __int__(self) -> int:
        return int(self.guard_int())

    def __float__(self) -> float:
        return float(self.guard_float())

    def __index__(self) -> int:
        return int(self.guard_int())

    def _coerce(self, value: Any) -> SymNode:
        if isinstance(value, SymNode):
            return value
        kind = bool if isinstance(value, bool) else float if isinstance(value, float) else int
        cls = SymBool if kind is bool else SymFloat if kind is float else SymInt
        return cls(value, self.shape_env, kind, value)

    def _binary(
        self,
        other: Any,
        symbol: str,
        function: Callable[[Any, Any], Any],
        result_type: type | None = None,
        reverse: bool = False,
    ) -> SymNode:
        rhs = self._coerce(other)
        left, right = (rhs, self) if reverse else (self, rhs)
        expr = _binary_expr(symbol, left.expr, right.expr, function)
        kind = result_type or (
            float if float in (left.pytype, right.pytype) else int
        )
        hint = None
        if left.has_hint() and right.has_hint():
            try:
                hint = kind(function(left.hint, right.hint))
            except Exception:
                hint = None

        def evaluate(values: dict[str, Any]) -> Any:
            return function(left.evaluate(values), right.evaluate(values))

        cls = SymBool if kind is bool else SymFloat if kind is float else SymInt
        return cls(expr, self.shape_env, kind, hint, evaluate)

    def _unary(
        self,
        symbol: str,
        function: Callable[[Any], Any],
        result_type: type | None = None,
    ) -> SymNode:
        hint = None
        if self.has_hint():
            try:
                hint = function(self.hint)
            except Exception:
                pass

        def evaluate(values: dict[str, Any]) -> Any:
            return function(self.evaluate(values))

        kind = result_type or self.pytype
        cls = SymBool if kind is bool else SymFloat if kind is float else SymInt
        return cls(_unary_expr(symbol, self.expr, function), self.shape_env, kind, hint, evaluate)

    def add(self, other: Any) -> SymNode:
        return self._binary(other, "+", operator.add)

    def sub(self, other: Any) -> SymNode:
        return self._binary(other, "-", operator.sub)

    def mul(self, other: Any) -> SymNode:
        return self._binary(other, "*", operator.mul)

    def mod(self, other: Any) -> SymNode:
        return self._binary(other, "%", operator.mod)

    def float_pow(self, other: Any) -> SymNode:
        return self._binary(other, "**", operator.pow, float)

    def pow_by_natural(self, other: Any) -> SymNode:
        return self._binary(other, "**", operator.pow, int)

    def pow(self, other: Any) -> SymNode:
        return self._binary(other, "**", operator.pow)

    def float_truediv(self, other: Any) -> SymNode:
        return self._binary(other, "/", operator.truediv, float)

    def int_truediv(self, other: Any) -> SymNode:
        return self._binary(other, "int_truediv", operator.truediv, int)

    def int_floordiv(self, other: Any) -> SymNode:
        return self._binary(other, "//", operator.floordiv, int)

    def xor(self, other: Any) -> SymBool:
        return self._binary(other, "xor", operator.xor, bool)  # type: ignore[return-value]

    def bitwise_and(self, other: Any) -> SymInt:
        return self._binary(other, "&", operator.and_, int)  # type: ignore[return-value]

    def bitwise_or(self, other: Any) -> SymInt:
        return self._binary(other, "|", operator.or_, int)  # type: ignore[return-value]

    def bitwise_xor(self, other: Any) -> SymInt:
        return self._binary(other, "^", operator.xor, int)  # type: ignore[return-value]

    def sym_or(self, other: Any) -> SymBool:
        return self.or_(other)

    def sym_and(self, other: Any) -> SymBool:
        return self.and_(other)

    def lshift(self, other: Any) -> SymInt:
        return self._binary(other, "<<", operator.lshift, int)  # type: ignore[return-value]

    def rshift(self, other: Any) -> SymInt:
        return self._binary(other, ">>", operator.rshift, int)  # type: ignore[return-value]

    def neg(self) -> SymNode:
        return self._unary("-", operator.neg)

    def pos(self) -> SymNode:
        return self._unary("+", operator.pos)

    def abs(self) -> SymNode:
        return self._unary("abs", abs)

    def round(self, ndigits: int | None = None) -> SymNode:
        if ndigits is None:
            return self._unary("round", round)
        return self._binary(ndigits, "round", lambda value, digits: round(value, digits), self.pytype)

    def trunc(self) -> SymInt:
        return self._unary("trunc", math.trunc, int)  # type: ignore[return-value]

    def floor(self) -> SymInt:
        return self._unary("floor", math.floor, int)  # type: ignore[return-value]

    def ceil(self) -> SymInt:
        return self._unary("ceil", math.ceil, int)  # type: ignore[return-value]

    def is_integer(self) -> SymBool:
        return self._unary("is_integer", lambda value: float(value).is_integer(), bool)  # type: ignore[return-value]

    def sym_float(self) -> SymFloat:
        return self._unary("float", float, float)  # type: ignore[return-value]

    def sym_int(self) -> SymInt:
        return self._unary("int", int, int)  # type: ignore[return-value]

    def sym_min(self, other: Any) -> SymNode:
        return self._binary(other, "min", min)

    def sym_max(self, other: Any) -> SymNode:
        return self._binary(other, "max", max)

    def sym_ite(self, then_value: Any, else_value: Any) -> SymNode:
        return sym_ite(self, then_value, else_value)

    def is_non_overlapping_and_dense_indicator(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymInt:
        if len(sizes) != len(strides):
            return SymInt.wrap_int(0, self.shape_env)
        pairs = sorted(
            zip(sizes, strides),
            key=lambda item: item[1].hint if item[1].has_hint() else 0,
        )
        expected: SymNode = SymInt.wrap_int(1, self.shape_env)
        for size, stride in pairs:
            if size.has_hint() and size.hint == 1:
                continue
            if stride.has_hint() and expected.has_hint() and stride.hint != expected.hint:
                return SymInt.wrap_int(0, self.shape_env)
            expected = expected * size
        return SymInt.wrap_int(1, self.shape_env)

    def is_non_overlapping_and_dense(self, sizes: list[SymNode], strides: list[SymNode]) -> SymBool:
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(1)

    def is_contiguous(self, sizes: list[SymNode], strides: list[SymNode]) -> SymBool:
        return _layout_predicate(self, sizes, strides, "contiguous")

    def is_channels_last_contiguous_2d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymBool:
        return _layout_predicate(self, sizes, strides, "channels_last_2d")

    def is_channels_last_contiguous_3d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymBool:
        return _layout_predicate(self, sizes, strides, "channels_last_3d")

    def is_channels_last_strides_2d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymBool:
        return _layout_predicate(self, sizes, strides, "channels_last_strides_2d")

    def is_channels_last_strides_3d(
        self, sizes: list[SymNode], strides: list[SymNode]
    ) -> SymBool:
        return _layout_predicate(self, sizes, strides, "channels_last_strides_3d")

    def eq(self, other: Any) -> SymBool:
        return self._binary(other, "==", operator.eq, bool)  # type: ignore[return-value]

    def ne(self, other: Any) -> SymBool:
        return self._binary(other, "!=", operator.ne, bool)  # type: ignore[return-value]

    def gt(self, other: Any) -> SymBool:
        return self._binary(other, ">", operator.gt, bool)  # type: ignore[return-value]

    def lt(self, other: Any) -> SymBool:
        return self._binary(other, "<", operator.lt, bool)  # type: ignore[return-value]

    def le(self, other: Any) -> SymBool:
        return self._binary(other, "<=", operator.le, bool)  # type: ignore[return-value]

    def ge(self, other: Any) -> SymBool:
        return self._binary(other, ">=", operator.ge, bool)  # type: ignore[return-value]

    def and_(self, other: Any) -> SymBool:
        return self._binary(other, "and", lambda a, b: bool(a and b), bool)  # type: ignore[return-value]

    def or_(self, other: Any) -> SymBool:
        return self._binary(other, "or", lambda a, b: bool(a or b), bool)  # type: ignore[return-value]

    def sym_not(self) -> SymBool:
        return self._unary("not", operator.not_)  # type: ignore[return-value]

    def sym_sum(self, values: list[SymNode]) -> SymInt:
        result: SymNode = SymInt.wrap_int(0, self.shape_env)
        for value in values:
            result = result + value
        return result  # type: ignore[return-value]

    def clone(self) -> SymNode:
        return type(self)(self._expr, self.shape_env, self.pytype, self._hint, self._evaluate)

    def str(self) -> str:
        return str(self._expr)

    def _graph_repr(self) -> str:
        return self.str()

    def evaluate(self, values: dict[str, Any] | None = None) -> Any:
        if self._evaluate is not None:
            return self._evaluate(values or {})
        if self.has_hint():
            return self._hint
        if isinstance(self._expr, str):
            try:
                return eval(self._expr, {"__builtins__": {}}, values or {})
            except Exception as exc:
                raise TypeError(f"cannot evaluate symbolic expression {self._expr!r}") from exc
        if isinstance(self._expr, sympy.Basic):
            substitutions = {
                sympy.Symbol(str(key)): value for key, value in (values or {}).items()
            }
            result = self._expr.subs(substitutions)
            if not result.free_symbols:
                return _primitive(result)
        return self._expr

    def _guard(self, expected: type) -> Any:
        value = self.evaluate()
        if not isinstance(value, expected):
            raise TypeError(f"expected {_type_name(expected)}, got {type(value).__name__}")
        return value

    def maybe_as_int(self) -> int | None:
        try:
            if self.is_symbolic():
                return None
            return int(self._guard(int))
        except (TypeError, ValueError):
            return None

    def maybe_as_float(self) -> float | None:
        try:
            return None if self.is_symbolic() else float(self.evaluate())
        except (TypeError, ValueError):
            return None

    def maybe_as_bool(self) -> bool | None:
        try:
            if self.is_symbolic():
                return None
            value = self.evaluate()
            return value if isinstance(value, bool) else None
        except (TypeError, ValueError):
            return None

    def is_int(self) -> bool:
        return self.pytype is int

    def is_float(self) -> bool:
        return self.pytype is float

    def is_bool(self) -> bool:
        return self.pytype is bool

    def is_symbolic(self) -> bool:
        if isinstance(self._expr, str):
            return True
        return bool(getattr(self._expr, "free_symbols", ()))

    def is_constant(self) -> bool:
        return not self.is_symbolic()

    def guard_int(self, *_: Any) -> int:
        value = self._guard(int)
        return int(value)

    def guard_float(self, *_: Any) -> float:
        return float(self.evaluate())

    def guard_bool(self, *_: Any) -> bool:
        value = _primitive(self.evaluate())
        if isinstance(value, sympy.logic.boolalg.BooleanAtom):
            return bool(value)
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got {type(value).__name__}")
        return value

    def expect_true(self, *_: Any) -> bool:
        return self.guard_bool()

    def statically_known_true(self, *_: Any) -> bool:
        return self.has_hint() and bool(self._hint)

    def statically_known_false(self, *_: Any) -> bool:
        return self.has_hint() and not bool(self._hint)

    def guard_size_oblivious(self, *_: Any) -> bool:
        return self.guard_bool()

    def guard_or_false(self, *_: Any) -> bool:
        try:
            return self.guard_bool()
        except Exception:
            return False

    def guard_or_true(self, *_: Any) -> bool:
        try:
            return self.guard_bool()
        except Exception:
            return True

    def int_(self) -> int:
        return self.guard_int()

    def bool_(self) -> bool:
        return self.guard_bool()

    def nested_int(self) -> None:
        return None

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"SymNode({self._expr!r}, pytype={_type_name(self.pytype)})"

    def __add__(self, other: Any) -> SymNode:
        return self.add(other)

    def __radd__(self, other: Any) -> SymNode:
        return self._binary(other, "+", operator.add, reverse=True)

    def __sub__(self, other: Any) -> SymNode:
        return self.sub(other)

    def __rsub__(self, other: Any) -> SymNode:
        return self._binary(other, "-", operator.sub, reverse=True)

    def __mul__(self, other: Any) -> SymNode:
        return self.mul(other)

    def __rmul__(self, other: Any) -> SymNode:
        return self._binary(other, "*", operator.mul, reverse=True)

    def __truediv__(self, other: Any) -> SymNode:
        return self.float_truediv(other)

    def __rtruediv__(self, other: Any) -> SymNode:
        return self._binary(other, "/", operator.truediv, float, reverse=True)

    def __floordiv__(self, other: Any) -> SymNode:
        return self.int_floordiv(other)

    def __rfloordiv__(self, other: Any) -> SymNode:
        return self._binary(other, "//", operator.floordiv, int, reverse=True)

    def __mod__(self, other: Any) -> SymNode:
        return self.mod(other)

    def __rmod__(self, other: Any) -> SymNode:
        return self._binary(other, "%", operator.mod, reverse=True)

    def __pow__(self, other: Any) -> SymNode:
        return self.pow(other)

    def __rpow__(self, other: Any) -> SymNode:
        return self._binary(other, "**", operator.pow, reverse=True)

    def __neg__(self) -> SymNode:
        return self.neg()

    def __pos__(self) -> SymNode:
        return self.pos()

    def __lt__(self, other: Any) -> SymBool:
        return self.lt(other)

    def __le__(self, other: Any) -> SymBool:
        return self.le(other)

    def __gt__(self, other: Any) -> SymBool:
        return self.gt(other)

    def __ge__(self, other: Any) -> SymBool:
        return self.ge(other)

    def __and__(self, other: Any) -> SymNode:
        return self.bitwise_and(other)

    def __rand__(self, other: Any) -> SymNode:
        return self._binary(other, "&", operator.and_, reverse=True)

    def __or__(self, other: Any) -> SymNode:
        return self.bitwise_or(other)

    def __ror__(self, other: Any) -> SymNode:
        return self._binary(other, "|", operator.or_, reverse=True)

    def __xor__(self, other: Any) -> SymNode:
        return self.bitwise_xor(other)

    def __rxor__(self, other: Any) -> SymNode:
        return self._binary(other, "^", operator.xor, reverse=True)

    def __lshift__(self, other: Any) -> SymInt:
        return self.lshift(other)

    def __rlshift__(self, other: Any) -> SymInt:
        return self._binary(other, "<<", operator.lshift, int, reverse=True)  # type: ignore[return-value]

    def __rshift__(self, other: Any) -> SymInt:
        return self.rshift(other)

    def __rrshift__(self, other: Any) -> SymInt:
        return self._binary(other, ">>", operator.rshift, int, reverse=True)  # type: ignore[return-value]

    def __invert__(self) -> SymBool:
        return self.sym_not()

    def truediv(self, other: Any) -> SymFloat:
        return self.float_truediv(other)  # type: ignore[return-value]

    def floordiv(self, other: Any) -> SymInt:
        return self.int_floordiv(other)  # type: ignore[return-value]

    @classmethod
    def wrap_int(cls, value: Any, shape_env: Any = None) -> SymInt:
        return SymInt(sympy.sympify(value), shape_env, int, value)

    @classmethod
    def wrap_float(cls, value: Any, shape_env: Any = None) -> SymFloat:
        return SymFloat(sympy.sympify(value), shape_env, float, value)

    @classmethod
    def wrap_bool(cls, value: Any, shape_env: Any = None) -> SymBool:
        return SymBool(sympy.true if value else sympy.false, shape_env, bool, value)


class SymInt(SymNode):
    """Symbolic integer scalar."""


class SymFloat(SymNode):
    """Symbolic floating-point scalar."""


class SymBool(SymNode):
    """Symbolic boolean scalar."""


SymTypes = (SymInt, SymFloat, SymBool)


def _layout_predicate(
    owner: SymNode,
    sizes: list[SymNode],
    strides: list[SymNode],
    layout: str,
) -> SymBool:
    def check(size_values: list[Any], stride_values: list[Any]) -> bool:
        if len(size_values) != len(stride_values):
            return False
        if layout == "contiguous":
            expected = 1
            for size, stride in zip(reversed(size_values), reversed(stride_values)):
                if size != 1 and stride != expected:
                    return False
                expected *= size
            return True
        if layout.startswith("channels_last"):
            rank = 4 if layout.endswith("2d") else 5
            if len(size_values) != rank:
                return False
            order = [0, 2, 3, 1] if rank == 4 else [0, 2, 3, 4, 1]
            expected = 1
            wanted = [0] * rank
            for axis in reversed(order):
                wanted[axis] = expected
                expected *= size_values[axis]
            return all(size == 1 or stride == wanted[index] for index, (size, stride) in enumerate(zip(size_values, stride_values)))
        return False

    if all(value.has_hint() for value in (*sizes, *strides)):
        return SymBool.wrap_bool(check([value.hint for value in sizes], [value.hint for value in strides]), owner.shape_env)
    size_exprs = [_as_expr(value) for value in sizes]
    stride_exprs = [_as_expr(value) for value in strides]
    expression = {
        "contiguous": sympy_is_contiguous,
        "channels_last_2d": sympy_is_channels_last_contiguous_2d,
        "channels_last_3d": sympy_is_channels_last_contiguous_3d,
        "channels_last_strides_2d": sympy_is_channels_last_strides_2d,
        "channels_last_strides_3d": sympy_is_channels_last_strides_3d,
    }.get(layout, lambda a, b: sympy.Function(layout)(sympy.Tuple(*a), sympy.Tuple(*b)))(size_exprs, stride_exprs)
    return SymBool(expression, owner.shape_env, bool, None)


class _DynamicScalar:
    def __new__(cls, *args: Any) -> Any:
        if cls is _DynamicScalar:
            raise TypeError("_DynamicScalar is an abstract base class")
        return super().__new__(cls, *args)


class DynamicInt(_DynamicScalar, int):
    """An integer wrapper that keeps dynamic-input intent in repr output."""

    def __new__(cls, value: int) -> "DynamicInt":
        if not isinstance(value, int):
            raise TypeError(f"expected int, got {type(value).__name__}")
        return cast(DynamicInt, super().__new__(cls, int(value)))

    def __repr__(self) -> str:
        return f"DynamicInt({int(self)})"

    def __floordiv__(self, other: int) -> "DynamicInt":
        return DynamicInt(int(self) // other)

    def __rfloordiv__(self, other: int) -> "DynamicInt":
        return DynamicInt(other // int(self))

    def __pow__(self, other: int, modulo: int | None = None) -> Any:
        result = pow(int(self), other, modulo) if modulo is not None else int(self) ** other
        return DynamicInt(result) if isinstance(result, int) else result

    def __rpow__(self, other: int, modulo: int | None = None) -> Any:
        result = pow(other, int(self), modulo) if modulo is not None else other ** int(self)
        return DynamicInt(result) if isinstance(result, int) else result


def to_node(value: Any, shape_env: Any = None) -> SymNode:
    if isinstance(value, SymNode):
        return value
    if isinstance(value, bool):
        return SymBool.wrap_bool(value, shape_env)
    if isinstance(value, float):
        return SymFloat.wrap_float(value, shape_env)
    return SymInt.wrap_int(value, shape_env)


def wrap_node(value: Any, shape_env: Any = None) -> SymNode:
    return to_node(value, shape_env)


def sym_min(a: Any, b: Any) -> SymNode:
    left, right = to_node(a), to_node(b)
    return left._binary(right, "min", min)


def sym_max(a: Any, b: Any) -> SymNode:
    left, right = to_node(a), to_node(b)
    return left._binary(right, "max", max)


def sym_ite(condition: Any, true_value: Any, false_value: Any) -> SymNode:
    shape_env = condition.shape_env if isinstance(condition, SymNode) else None
    cond = to_node(condition, shape_env)
    true_node, false_node = to_node(true_value, shape_env), to_node(false_value, shape_env)
    hint = None
    if cond.has_hint():
        hint = true_node.hint if cond.hint else false_node.hint
    result_type = true_node.pytype
    result_class = SymBool if result_type is bool else SymFloat if result_type is float else SymInt
    return result_class(
        sympy.Piecewise((_as_expr(true_node), _as_expr(cond)), (_as_expr(false_node), True)),
        shape_env,
        result_type,
        hint,
        lambda values: true_node.evaluate(values) if cond.evaluate(values) else false_node.evaluate(values),
    )


def method_to_operator(method: str) -> Callable[..., Any]:
    operators: dict[str, Callable[..., Any]] = {
        "pos": operator.pos,
        "abs": operator.abs,
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "mod": operator.mod,
        "float_pow": operator.pow,
        "pow": operator.pow,
        "float_truediv": operator.truediv,
        "int_truediv": operator.truediv,
        "int_floordiv": operator.floordiv,
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "lt": operator.lt,
        "le": operator.le,
        "ge": operator.ge,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
        "bitwise_and": operator.and_,
        "bitwise_or": operator.or_,
        "bitwise_xor": operator.xor,
        "lshift": operator.lshift,
        "rshift": operator.rshift,
        "neg": operator.neg,
        "sym_not": operator.not_,
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "sym_min": min,
        "sym_max": max,
    }
    try:
        return operators[method]
    except KeyError as exc:
        raise KeyError(f"unknown symbolic operation {method!r}") from exc


magic_methods = {
    name: method_to_operator(name)
    for name in (
        "pos",
        "abs",
        "add",
        "sub",
        "mul",
        "mod",
        "float_pow",
        "float_truediv",
        "int_truediv",
        "int_floordiv",
        "eq",
        "ne",
        "gt",
        "lt",
        "le",
        "ge",
        "and",
        "or",
        "xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "lshift",
        "rshift",
        "neg",
        "sym_not",
        "floor",
        "ceil",
        "trunc",
        "sym_min",
        "sym_max",
    )
}


def _get_sym_node_fn(name: str) -> Callable[[SymNode], SymNode]:
    def fn(node: SymNode) -> SymNode:
        return node._unary(name, getattr(math, name), float)

    return fn


def _sympy_float_truediv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return a / b


def _sympy_int_truediv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.floor(a / b)


def _sympy_floordiv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.floor(a / b)


def _sympy_mod(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Mod(a, b)


def _sympy_pow_by_natural(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return a**b


def _sympy_float_pow(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return a**b


def _sympy_and(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.And(a, b)


def _sympy_or(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Or(a, b)


def _sympy_xor(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Xor(a, b)


def _sympy_lshift(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Function("lshift")(a, b)


def _sympy_rshift(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Function("rshift")(a, b)


def _binary_search_insert_arg(ordered_args: list[sympy.Basic], new_arg: sympy.Basic) -> list[sympy.Basic] | None:
    if new_arg in ordered_args:
        return None
    ordered_args.append(new_arg)
    ordered_args.sort(key=str)
    return ordered_args


def make_optimized(ordered_args: list[sympy.Basic]) -> tuple[bool, sympy.Basic]:
    return True, sympy.Add(*ordered_args)


def _optimized_add(
    lhs: sympy.Basic,
    rhs: sympy.Basic,
    lhs_is_optimized_summation: bool = False,
    rhs_is_optimized_summation: bool = False,
) -> tuple[bool, sympy.Basic]:
    del lhs_is_optimized_summation, rhs_is_optimized_summation
    result = sympy.Add(lhs, rhs)
    return isinstance(result, sympy.Add) and all(item.is_Symbol for item in result.args), result


def _bitwise_and(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Function("bitwise_and")(a, b)


def _bitwise_or(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Function("bitwise_or")(a, b)


def _bitwise_xor(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Function("bitwise_xor")(a, b)


def _floor_ceil_helper(a: sympy.Basic, fn: Callable[..., sympy.Basic]) -> sympy.Basic:
    return fn(a)


def _sympy_floor(a: sympy.Basic) -> sympy.Basic:
    return sympy.floor(a)


def _sympy_trunc(a: sympy.Basic) -> sympy.Basic:
    return sympy.Function("trunc")(a)


def _sympy_ceil(a: sympy.Basic) -> sympy.Basic:
    return sympy.ceiling(a)


def _sympy_eq(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Eq(a, b)


def _sympy_ne(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Ne(a, b)


def _sympy_gt(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Gt(a, b)


def _sympy_lt(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Lt(a, b)


def _sympy_le(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Le(a, b)


def _sympy_ge(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Ge(a, b)


def _sympy_min(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Min(a, b)


def _sympy_max(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return sympy.Max(a, b)


def _sympy_ite(a: sympy.Basic, t: sympy.Basic, f: sympy.Basic) -> sympy.Basic:
    return sympy.Piecewise((t, a), (f, True))


def _get_sym_math_fn(name: str) -> Callable[[sympy.Basic], sympy.Basic]:
    return lambda value: sympy.Function(name)(value)


def _sympy_abs(a: sympy.Basic) -> sympy.Basic:
    return sympy.Abs(a)


def _sympy_round(number: sympy.Basic, ndigits: sympy.Basic | None = None) -> sympy.Basic:
    return sympy.Function("round")(number) if ndigits is None else sympy.Function("round")(number, ndigits)


def _sympy_sym_float(a: sympy.Basic) -> sympy.Basic:
    return sympy.Function("to_float")(a)


def _sympy_is_integer(a: sympy.Basic) -> sympy.Basic:
    return sympy.Eq(sympy.floor(a), a)


def sympy_is_contiguous(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    return sympy_is_contiguous_generic(sizes, strides, list(range(len(sizes) - 1, -1, -1)))


def sympy_is_contiguous_generic(sizes: list[sympy.Basic], strides: list[sympy.Basic], dim_order: list[int]) -> sympy.Basic:
    if len(sizes) != len(strides) or len(dim_order) != len(sizes):
        return sympy.false
    result: sympy.Basic = sympy.true
    expected: sympy.Basic = sympy.Integer(1)
    for index in dim_order:
        result = sympy.And(result, sympy.Or(sympy.Eq(sizes[index], 1), sympy.Eq(strides[index], expected)))
        expected *= sizes[index]
    return sympy.Or(result, *[sympy.Eq(size, 0) for size in sizes])


def sympy_is_channels_last_contiguous_2d(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_contiguous_3d(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])


def sympy_is_channels_last_strides_generic(sizes: list[sympy.Basic], strides: list[sympy.Basic], dim_order: list[int]) -> sympy.Basic:
    if len(sizes) != len(strides) or len(dim_order) != len(sizes):
        return sympy.false
    result: sympy.Basic = sympy.Ne(strides[1], 0) if len(strides) > 1 else sympy.true
    minimum: sympy.Basic = sympy.Integer(0)
    for index in dim_order:
        result = sympy.And(result, sympy.Ne(sizes[index], 0), strides[index] >= minimum)
        minimum = strides[index] * sympy.Max(sizes[index], 1)
    return result


def sympy_is_channels_last_strides_2d(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])


def sympy_is_channels_last_strides_3d(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])


def sympy_is_non_overlapping_and_dense_indicator(sizes: list[sympy.Basic], strides: list[sympy.Basic]) -> sympy.Basic:
    ordered = sorted(zip(sizes, strides), key=lambda item: str(item[1]))
    expected: sympy.Basic = sympy.Integer(1)
    result: sympy.Basic = sympy.true
    for size, stride in ordered:
        result = sympy.And(result, sympy.Or(sympy.Eq(size, 1), sympy.Eq(stride, expected)))
        expected *= size
    return sympy.Piecewise((1, result), (0, True))


def dynamic_int_impl(value: Any) -> DynamicInt:
    return DynamicInt(int(value))


def unary_magic_impl(node: SymNode, method: str) -> SymNode:
    return getattr(node, method)()


def binary_magic_impl(node: SymNode, method: str, other: Any) -> SymNode:
    return getattr(node, method)(other)


def rbinary_magic_impl(node: SymNode, method: str, other: Any) -> SymNode:
    if method not in magic_methods:
        raise KeyError(f"unknown symbolic operation {method!r}")
    symbols = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "mod": "%",
        "float_pow": "**",
        "pow": "**",
        "float_truediv": "/",
        "int_truediv": "int_truediv",
        "int_floordiv": "//",
        "eq": "==",
        "ne": "!=",
        "gt": ">",
        "lt": "<",
        "le": "<=",
        "ge": ">=",
        "and": "and",
        "or": "or",
        "xor": "xor",
        "bitwise_and": "&",
        "bitwise_or": "|",
        "bitwise_xor": "^",
        "lshift": "<<",
        "rshift": ">>",
        "sym_min": "min",
        "sym_max": "max",
    }
    symbol = symbols.get(method)
    if symbol is None:
        return getattr(node, method)(other)
    result_type = bool if method in {"eq", "ne", "gt", "lt", "le", "ge", "and", "or", "xor"} else None
    if method in {"float_pow", "float_truediv"}:
        result_type = float
    elif method in {"int_truediv", "int_floordiv", "bitwise_and", "bitwise_or", "bitwise_xor", "lshift", "rshift"}:
        result_type = int
    return node._binary(other, symbol, magic_methods[method], result_type, reverse=True)


def sizes_strides_impl(node: SymNode, method: str, sizes: list[SymNode], strides: list[SymNode]) -> SymNode:
    return getattr(node, method)(sizes, strides)


def sizes_strides_user(node: SymNode, method: str, sizes: list[SymNode], strides: list[SymNode]) -> SymNode:
    return sizes_strides_impl(node, method, sizes, strides)


def sym_ite_impl(condition: Any, true_value: Any, false_value: Any) -> SymNode:
    return sym_ite(condition, true_value, false_value)


def sym_ite_magic_impl(node: SymNode, true_value: Any, false_value: Any) -> SymNode:
    return sym_ite(node, true_value, false_value)


def round_impl(node: SymNode, ndigits: int | None = None) -> SymNode:
    return node.round(ndigits)


def round_magic_impl(node: SymNode, ndigits: Any = None) -> SymNode:
    return node.round(ndigits)


def promote(node: SymNode, other: Any) -> tuple[SymNode, SymNode]:
    return node, node._coerce(other)


def promote2(left: Any, right: Any) -> tuple[SymNode, SymNode]:
    node = to_node(left)
    return node, node._coerce(right)


def get_id(value: Any) -> int:
    return id(value)


def get_constant(value: Any) -> Any:
    return value.hint if isinstance(value, SymNode) and value.has_hint() else value


def capture_provenance(value: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return value


def setattrs(value: Any, attrs: dict[str, Any]) -> Any:
    for key, item in attrs.items():
        setattr(value, key, item)
    return value


def is_nested_int(value: Any) -> bool:
    return isinstance(value, SymInt)


def uninteresting_files() -> set[str]:
    return set()


def compute_hint(value: SymNode) -> Any:
    return value.hint if value.has_hint() else value.evaluate()


def wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


def fn(function: Callable[..., Any]) -> Callable[..., Any]:
    return function


__all__ = [
    "DynamicInt",
    "magic_methods",
    "method_to_operator",
    "SymBool",
    "SymFloat",
    "SymInt",
    "SymNode",
    "sym_ite",
    "sym_max",
    "sym_min",
    "to_node",
    "wrap_node",
]
