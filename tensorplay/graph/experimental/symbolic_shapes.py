from __future__ import annotations

import functools
import itertools
import math
import operator
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, TypeAlias

import sympy
from sympy.printing.pycode import PythonCodePrinter

from ..graph import Graph
from ..interpreter import Interpreter
from ..node import Node
from .recording import (
    ShapeEnvEvent,
    record_shapeenv_event,
    replay_shape_env_events,
    shape_env_check_state_equal,
)
from .sym_node import SymBool, SymFloat, SymInt, SymNode, sym_ite

Int: TypeAlias = SymInt | int
Scalar: TypeAlias = SymInt | SymFloat | SymBool | int | float | bool

SHAPEENV_EVENT_KEY = "shapeenv_event"
CURRENT_NODE_KEY = "current_node"

__all__ = [
    "CallMethodKey",
    "SequenceKey",
    "Constraint",
    "ConstraintViolationError",
    "ConvertIntKey",
    "DimDynamic",
    "DivideByKey",
    "EqualityConstraint",
    "GuardOnDataDependentSymNode",
    "InnerTensorKey",
    "PendingUnbackedSymbolNotFound",
    "RelaxedUnspecConstraint",
    "RuntimeAssert",
    "SHAPEENV_EVENT_KEY",
    "CURRENT_NODE_KEY",
    "ShapeEnv",
    "ShapeEnvSettings",
    "ShapeGuard",
    "ShapeGuardPrinter",
    "ShapeGuardPythonPrinter",
    "SymExprPrinter",
    "LoggingShapeGuardPrinter",
    "DynamicDimConstraintPrinter",
    "DimConstraints",
    "Specialization",
    "StatefulSymbolicContext",
    "StatelessSymbolicContext",
    "StrictMinMaxConstraint",
    "SubclassSymbolicContext",
    "SymIntEqByExpr",
    "SymIntSymbolicContext",
    "SymbolicContext",
    "TrackedFake",
    "ValueRanges",
    "ValueRangesSLoc",
    "SYMPY_INTERP",
    "bind_symbols",
    "canonicalize_bool_expr",
    "compute_unbacked_bindings",
    "cast_symbool_to_symint_guardless",
    "check_consistent",
    "constrain_range",
    "constrain_unify",
    "create_contiguous",
    "eval_guards",
    "eval_is_non_overlapping_and_dense",
    "expect_true",
    "free_symbols",
    "guard_bool",
    "guard_float",
    "guard_int",
    "guard_or_false",
    "guard_or_true",
    "guard_scalar",
    "guard_size_oblivious",
    "guarding_hint_or_throw",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "has_guarding_hint",
    "has_static_value",
    "has_symbolic_sizes_strides",
    "is_accessor_node",
    "is_concrete_bool",
    "is_concrete_float",
    "is_concrete_int",
    "is_nested_int",
    "is_symbol_binding_graph_node",
    "is_symbolic",
    "optimization_hint",
    "rebind_unbacked",
    "resolve_unbacked_bindings",
    "safe_expand",
    "statically_known_false",
    "statically_known_true",
    "sym_and",
    "sym_eq",
    "sym_or",
    "PropagateUnbackedSymInts",
]


class ConstraintViolationError(RuntimeError):
    pass


class GuardOnDataDependentSymNode(RuntimeError):
    def __init__(self, cond: Any, *args: Any) -> None:
        super().__init__(*args)
        self.cond = cond


class PendingUnbackedSymbolNotFound(RuntimeError):
    pass


class _ShapeEnvGuardError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Source:
    name: str
    idx: int | None = None


@dataclass(frozen=True, slots=True)
class ShapeGuard:
    expr: sympy.Basic
    source: Any = None
    size_oblivious: bool = False


@dataclass(frozen=True, slots=True)
class _ShapeGuardsHelper:
    exprs: list[str]


@dataclass(frozen=True, slots=True)
class ValueRanges:
    lower: Any
    upper: Any

    @classmethod
    def unknown(cls) -> "ValueRanges":
        return cls(-sympy.oo, sympy.oo)

    @property
    def is_int(self) -> bool:
        return all(isinstance(value, (int, sympy.Integer)) for value in (self.lower, self.upper))

    @property
    def is_float(self) -> bool:
        return not self.is_int

    def is_singleton(self) -> bool:
        return self.lower == self.upper


@dataclass(slots=True)
class ValueRangesSLoc:
    lower: Any
    upper: Any


@dataclass(frozen=True, slots=True)
class ShapeEnvSettings:
    allow_scalar_outputs: bool
    allow_dynamic_output_shape_ops: bool
    assume_static_by_default: bool
    specialize_zero_one: bool
    duck_shape: bool
    prefer_deferred_runtime_asserts_over_guards: bool
    trace_asserts: bool


class DimDynamic(Enum):
    DYNAMIC = 0
    DUCK = 1
    STATIC = 2
    UNBACKED = 3
    INFER_STRIDE = 4


@dataclass(frozen=True, slots=True)
class Constraint:
    warn_only: bool = False


@dataclass(frozen=True, slots=True)
class StrictMinMaxConstraint(Constraint):
    vr: ValueRanges = field(default_factory=ValueRanges.unknown)

    def render(self, source: Any) -> str:
        return f"{self.vr.lower} <= {_source_name(source)} <= {self.vr.upper}"


@dataclass(frozen=True, slots=True)
class RelaxedUnspecConstraint(Constraint):
    def render(self, source: Any) -> str:
        return f"RelaxedUnspecConstraint({_source_name(source)})"


DimConstraint = StrictMinMaxConstraint | RelaxedUnspecConstraint | None


@dataclass(frozen=True, slots=True)
class EqualityConstraint:
    source_pairs: list[tuple[Any, Any]]
    derived_equalities: list[tuple[Any, Any, Callable[[sympy.Expr], sympy.Expr]]]
    phantom_symbols: list[sympy.Symbol]
    relaxed_sources: set[Any]
    _parents: dict[Any, Any] = field(init=False, repr=False)
    _defs: dict[Any, sympy.Expr] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parents: dict[Any, Any] = {}
        definitions: dict[Any, sympy.Expr] = {}
        object.__setattr__(self, "_parents", parents)
        object.__setattr__(self, "_defs", definitions)
        for first, second in self.source_pairs:
            self._union(self._find(first), self._find(second))
        for source, root, function in self.derived_equalities:
            root_expr = root if isinstance(root, sympy.Basic) else self._rewrite(root)
            definitions[self._find(source)] = function(root_expr)

    def _find(self, source: Any) -> Any:
        parent = self._parents.get(source)
        return source if parent is None else self._find(parent)

    def _union(self, first: Any, second: Any) -> None:
        if first != second:
            self._parents[first] = second

    def _rewrite(self, source: Any) -> sympy.Expr:
        root = self._find(source)
        return self._defs.get(root, sympy.Symbol(_source_name(root)))

    def is_equal(self, first: Any, second: Any) -> bool:
        left, right = self._find(first), self._find(second)
        return left in self.relaxed_sources or right in self.relaxed_sources or left == right or self._rewrite(first) == self._rewrite(second)

    def is_derived(self, source: Any, symbol_source: Any, function: Callable[[sympy.Expr], sympy.Expr]) -> bool:
        return self._rewrite(source) == function(self._rewrite(symbol_source))


@dataclass(frozen=True, slots=True)
class SymbolicContext:
    pass


@dataclass(frozen=True, slots=True)
class SymIntSymbolicContext(SymbolicContext):
    constraint: DimConstraint = None


@dataclass(frozen=True, slots=True)
class StatelessSymbolicContext(SymbolicContext):
    dynamic_sizes: list[DimDynamic]
    dynamic_strides: list[DimDynamic] | None = None
    constraint_sizes: list[DimConstraint] | None = None
    constraint_strides: list[DimConstraint] | None = None
    specialize_on: list[list[Callable[..., Any]]] | None = None
    view_base_context: SymbolicContext | None = None
    shape_ids: dict[int, str | None] | None = None
    unbacked_bounds: dict[int, tuple[int | None, int | None]] | None = None

    def __post_init__(self) -> None:
        size_count = len(self.dynamic_sizes)
        if self.dynamic_strides is None:
            object.__setattr__(self, "dynamic_strides", [DimDynamic.INFER_STRIDE] * size_count)
        if self.constraint_sizes is None:
            object.__setattr__(self, "constraint_sizes", [None] * size_count)
        if self.constraint_strides is None:
            object.__setattr__(self, "constraint_strides", [None] * size_count)
        if self.specialize_on is None:
            object.__setattr__(self, "specialize_on", [[] for _ in range(size_count)])
        if any(value not in {DimDynamic.INFER_STRIDE, DimDynamic.DYNAMIC, DimDynamic.DUCK} for value in self.dynamic_strides or ()):
            raise ValueError("dynamic stride policy is invalid")


@dataclass(frozen=True, slots=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    tensor_source: Any = None
    shape_env_to_source_to_symbol_cache: dict[int, dict[str, sympy.Expr]] = field(default_factory=dict)
    excluded_sizes: tuple[int | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    inner_contexts: dict[str, SymbolicContext] = field(default_factory=dict)
    track_outer_size_stride: bool = True


@dataclass(slots=True)
class TrackedFake:
    fake: Any
    source: Any
    symbolic_context: SymbolicContext | None

    def __hash__(self) -> int:
        return hash((id(self.fake), _source_name(self.source)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrackedFake) and self.fake is other.fake and _source_name(self.source) == _source_name(other.source)


@dataclass(frozen=True, slots=True)
class Specialization:
    source: Any
    check_fn: Callable[[int], bool]


@dataclass(frozen=True, slots=True)
class ConvertIntKey:
    def __str__(self) -> str:
        return ".cast_symbool_to_symint_guardless()"

    def get(self, value: bool | SymBool) -> int | SymInt:
        return cast_symbool_to_symint_guardless(value)


@dataclass(frozen=True, slots=True)
class CallMethodKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}()"

    def get(self, value: Any) -> Any:
        return getattr(value, self.name)()


@dataclass(frozen=True, slots=True)
class SequenceKey:
    index: int

    def __str__(self) -> str:
        return f"[{self.index}]"

    def get(self, value: Sequence[Any]) -> Any:
        return value[self.index]


@dataclass(frozen=True, slots=True)
class InnerTensorKey:
    inner_name: str

    def __str__(self) -> str:
        return f".{self.inner_name}"

    def get(self, value: Any) -> Any:
        return getattr(value, self.inner_name)


@dataclass(frozen=True, slots=True)
class DivideByKey:
    divisor: int | SymInt

    def __str__(self) -> str:
        return f".__floordiv__({self.divisor})"

    def get(self, value: int) -> int:
        return value // self.divisor


class SymIntEqByExpr:
    def __init__(self, value: SymInt | int) -> None:
        self.value = _expr(value)

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SymIntEqByExpr) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True, slots=True)
class RuntimeAssert:
    expr: sympy.Basic
    msg: str = field(repr=False)
    stack: Any = field(repr=False, default=None)


def _source_name(source: Any) -> str:
    return str(getattr(source, "name", source))


def _expr(value: Any) -> sympy.Basic:
    if isinstance(value, SymNode):
        return sympy.sympify(value.expr)
    if isinstance(value, sympy.Basic):
        return value
    return sympy.sympify(value)


def _primitive(value: Any) -> Any:
    if value in (sympy.true, sympy.false):
        return bool(value)
    if isinstance(value, sympy.Integer):
        return int(value)
    if isinstance(value, sympy.Float):
        return float(value)
    return value


def guarding_hint_or_throw(value: SymNode | int | bool) -> int | bool:
    if isinstance(value, SymNode):
        if value.has_hint():
            return value.hint
        if value.shape_env is None:
            raise AssertionError("a shape environment is required")
        return value.shape_env.guarding_hint_or_throw(value.expr)
    if isinstance(value, (bool, int)):
        return value
    raise TypeError(f"expected integer or boolean, got {type(value).__name__}")


def optimization_hint(value: SymInt | int, fallback: int | None = None) -> int:
    if isinstance(value, SymNode):
        if value.has_hint():
            return int(value.hint)
        if value.shape_env is None:
            if fallback is None:
                raise RuntimeError("a shape environment is required")
            return fallback
        return value.shape_env.optimization_hint(value.expr, fallback=fallback)
    if type(value) is not int:
        raise TypeError(f"expected integer, got {type(value).__name__}")
    return value


def has_guarding_hint(value: Scalar) -> bool:
    return not isinstance(value, SymNode) or value.has_hint()


def is_concrete_int(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return isinstance(value, SymInt) and not free_symbols(value.expr)


def is_concrete_float(value: Any) -> bool:
    return isinstance(value, float) or isinstance(value, SymFloat) and not free_symbols(value.expr)


def is_concrete_bool(value: Any) -> bool:
    return isinstance(value, bool) or isinstance(value, SymBool) and not free_symbols(value.expr)


def has_static_value(value: Scalar) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    return isinstance(value, SymNode) and not free_symbols(value.expr)


def is_symbolic(value: Scalar) -> bool:
    return isinstance(value, SymNode) and bool(free_symbols(value.expr))


def is_nested_int(value: Any) -> bool:
    return isinstance(value, SymInt)


def has_symbolic_sizes_strides(value: Any) -> bool:
    if getattr(value, "_has_symbolic_sizes_strides", False):
        return True
    for name in ("shape", "size", "stride", "storage_offset"):
        item = getattr(value, name, None)
        if callable(item):
            try:
                item = item()
            except TypeError:
                continue
        if has_free_symbols(item):
            return True
    return False


def create_contiguous(shape: Sequence[Int]) -> list[Int]:
    result: list[Int] = [1]
    for dim in reversed(shape[:-1]):
        result.append(dim * result[-1])
    return list(reversed(result))


def free_symbols(value: Any) -> set[sympy.Symbol]:
    if isinstance(value, SymNode):
        return set(_expr(value).free_symbols)
    if isinstance(value, sympy.Basic):
        return set(value.free_symbols)
    if isinstance(value, (tuple, list, set, frozenset)):
        result: set[sympy.Symbol] = set()
        for item in value:
            result.update(free_symbols(item))
        return result
    if isinstance(value, dict):
        result: set[sympy.Symbol] = set()
        for item in value.values():
            result.update(free_symbols(item))
        return result
    return set()


def has_free_symbols(value: Any) -> bool:
    return bool(free_symbols(value))


def has_free_unbacked_symbols(value: Any) -> bool:
    return any(str(symbol).startswith("u") for symbol in free_symbols(value))


def canonicalize_bool_expr(expr: Any) -> Any:
    return sympy.simplify_logic(_expr(expr), force=True)


def guard_size_oblivious(expr: SymBool | bool) -> bool:
    return guard_bool(expr) if isinstance(expr, SymNode) else bool(expr)


def statically_known_true(value: SymBool | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.has_hint() and bool(value.hint) or not free_symbols(value.expr) and bool(_primitive(_expr(value)))


def statically_known_false(value: SymBool | bool) -> bool:
    if isinstance(value, bool):
        return not value
    return value.has_hint() and not bool(value.hint) or not free_symbols(value.expr) and not bool(_primitive(_expr(value)))


def sym_and(value: Any, *others: Any) -> Any:
    result = value
    for other in others:
        if isinstance(result, SymNode):
            result = result.and_(other)
        else:
            result = bool(result) and bool(other)
    return result


def sym_or(value: Any, *others: Any) -> Any:
    result = value
    for other in others:
        if isinstance(result, SymNode):
            result = result.or_(other)
        else:
            result = bool(result) or bool(other)
    return result


def sym_eq(left: Any, right: Any) -> Any:
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return False
        return functools.reduce(sym_and, (sym_eq(a, b) for a, b in zip(left, right)), True)
    if isinstance(left, SymNode):
        return left.eq(right)
    if isinstance(right, SymNode):
        return right.eq(left)
    return left == right


def guard_bool(value: SymBool | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value.guard_bool())


def guard_int(value: SymInt | int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, SymInt):
        raise TypeError(f"expected integer, got {type(value).__name__}")
    return int(value.guard_int())


def guard_float(value: SymFloat | float) -> float:
    if isinstance(value, float):
        return value
    if not isinstance(value, SymFloat):
        raise TypeError(f"expected float, got {type(value).__name__}")
    return float(value.guard_float())


def guard_scalar(value: Scalar) -> bool | int | float:
    if isinstance(value, (SymBool, bool)):
        return guard_bool(value)
    if isinstance(value, (SymInt, int)) and not isinstance(value, bool):
        return guard_int(value)
    if isinstance(value, (SymFloat, float)):
        return guard_float(value)
    raise TypeError(f"unsupported scalar {type(value).__name__}")


def guard_or_false(value: SymBool | bool) -> bool:
    try:
        return guard_bool(value)
    except GuardOnDataDependentSymNode:
        return False


def guard_or_true(value: SymBool | bool) -> bool:
    try:
        return guard_bool(value)
    except GuardOnDataDependentSymNode:
        return True


def constrain_range(value: SymInt | int, *, min: int | None, max: int | None = None) -> None:
    lower = -math.inf if min is None else min
    upper = math.inf if max is None else max
    if lower > upper:
        raise ValueError("minimum cannot exceed maximum")
    if isinstance(value, int) and not isinstance(value, bool):
        if not lower <= value <= upper:
            raise ValueError(f"value {value} is outside [{lower}, {upper}]")
        return
    if not isinstance(value, SymInt) or value.shape_env is None:
        raise TypeError("range constraints require a symbolic integer with an environment")
    value.shape_env._constrain_range(value.expr, lower, upper)


def constrain_unify(left: SymInt | int, right: SymInt | int) -> None:
    if isinstance(left, SymNode) and left.shape_env is not None:
        left.shape_env._constrain_unify(left.expr, right.expr if isinstance(right, SymNode) else right)
        return
    if left != right:
        raise ConstraintViolationError(f"cannot unify {left!r} and {right!r}")


def expect_true(value: SymBool | bool, skip: int = 0) -> bool:
    del skip
    return guard_bool(value)


def cast_symbool_to_symint_guardless(value: bool | SymBool) -> int | SymInt:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, SymBool) or value.shape_env is None:
        raise TypeError("expected a symbolic boolean with an environment")
    return value.shape_env.create_symintnode(sympy.Piecewise((1, _expr(value)), (0, True)), hint=value.hint)


def eval_is_non_overlapping_and_dense(sizes: Sequence[int], strides: Sequence[int]) -> bool:
    if len(sizes) != len(strides):
        return False
    if len(sizes) == 1:
        return strides[0] == 1 or sizes[0] < 2
    expected = 1
    for size, stride in sorted(zip(sizes, strides), key=lambda item: item[1]):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def is_accessor_node(node: Node) -> bool:
    return node.op in {"get_attr", "call_method"} or node.op == "call_function" and node.target is operator.getitem


def is_symbol_binding_graph_node(node: Node) -> sympy.Symbol | None:
    symbol = node.meta.get("symbol")
    return symbol if isinstance(symbol, sympy.Symbol) else None


def check_consistent(new: Any, old: Any) -> None:
    if isinstance(new, (int, float, bool, SymNode)):
        if isinstance(new, SymNode):
            if new.has_hint() and new.hint != old:
                raise AssertionError(f"{new.hint} != {old}")
        elif new != old:
            raise AssertionError(f"{new} != {old}")
        return
    new_shape = getattr(new, "shape", None)
    old_shape = getattr(old, "shape", None)
    if new_shape is not None and old_shape is not None and tuple(new_shape) != tuple(old_shape):
        raise AssertionError(f"{new_shape} != {old_shape}")


def resolve_unbacked_bindings(shape_env: "ShapeEnv | None", bindings: dict[sympy.Symbol, Any] | None) -> dict[sympy.Symbol, Any] | None:
    if bindings is None:
        return None
    if shape_env is None:
        raise AssertionError("shape_env is required")
    return {shape_env.unbacked_renamings.get(key, key): value for key, value in bindings.items()}


def rebind_unbacked(shape_env: ShapeEnv | None, node: Node, result: Any) -> None:
    del result
    if shape_env is None or node.op == "placeholder":
        return
    bindings = resolve_unbacked_bindings(shape_env, node.meta.get("unbacked_bindings"))
    if bindings:
        node.meta["unbacked_bindings"] = bindings


def compute_unbacked_bindings(
    shape_env: ShapeEnv | None,
    example_value: Any,
    old_example_value: Any | None = None,
    peek: bool = False,
) -> dict[sympy.Symbol, tuple[Any, ...]] | None:
    """Find fresh unbacked symbols in a result and record their access paths."""

    del old_example_value
    if shape_env is None:
        return None
    pending = set(shape_env.pending_fresh_unbacked_symbols)
    if not pending:
        return None
    result: dict[sympy.Symbol, tuple[Any, ...]] = {}

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        expression = _expr(value) if isinstance(value, SymNode) else None
        if isinstance(expression, sympy.Symbol) and expression in pending:
            result[expression] = path
            pending.remove(expression)
            return
        if (
            isinstance(value, SymBool)
            and isinstance(expression, sympy.Equality)
            and expression.rhs == 1
            and isinstance(expression.lhs, sympy.Symbol)
            and expression.lhs in pending
        ):
            result[expression.lhs] = path + (ConvertIntKey(),)
            pending.remove(expression.lhs)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, path + (SequenceKey(index),))
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + (key,))
            return
        shape = getattr(value, "shape", None)
        if shape is not None:
            shape = shape() if callable(shape) else shape
            visit(shape, path + (CallMethodKey("size"),))
        stride = getattr(value, "stride", None)
        if stride is not None:
            stride = stride() if callable(stride) else stride
            visit(stride, path + (CallMethodKey("stride"),))
        offset = getattr(value, "storage_offset", None)
        if offset is not None:
            offset = offset() if callable(offset) else offset
            visit(offset, path + (CallMethodKey("storage_offset"),))

    visit(example_value, ())
    if not peek:
        shape_env.pending_fresh_unbacked_symbols = [
            symbol for symbol in shape_env.pending_fresh_unbacked_symbols if symbol in pending
        ]
    if pending and not peek:
        raise PendingUnbackedSymbolNotFound(
            f"fresh symbols {sorted(map(str, pending))} were not found in the result"
        )
    return result or None


def safe_expand(expr: Any) -> Any:
    return sympy.expand(_expr(expr))


def is_symbol_binding_node(node: Node) -> sympy.Symbol | None:
    return is_symbol_binding_graph_node(node)


def _shape_value(value: Any, name: str, default: Any = None) -> Any:
    item = getattr(value, name, default)
    return item() if callable(item) else item


SYMPY_INTERP: dict[str, Any] = {
    "abs": abs,
    "ceiling": sympy.ceiling,
    "floor": sympy.floor,
    "Max": sympy.Max,
    "Min": sympy.Min,
    "Piecewise": sympy.Piecewise,
    "sym_ite": sym_ite,
    "cast_symbool_to_symint_guardless": cast_symbool_to_symint_guardless,
}


class ShapeEnv:
    """Track symbolic dimensions, value ranges, substitutions, and guards."""

    def __init__(
        self,
        *,
        should_record_events: bool | None = None,
        tracked_fakes: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._init(**kwargs)
        self._translation_validation_enabled = False
        self.should_record_events = bool(should_record_events)
        self.check_recorded_events = False
        self.is_recording = False
        self.tracked_fakes = tracked_fakes
        self.events: list[ShapeEnvEvent] = []
        if self.should_record_events:
            replay_kwargs = dict(kwargs)
            replay_kwargs["should_record_events"] = False
            self.events.append(ShapeEnvEvent(type(self), kwargs=replay_kwargs))

    def _init(
        self,
        *,
        allow_scalar_outputs: bool = True,
        allow_dynamic_output_shape_ops: bool = True,
        assume_static_by_default: bool = False,
        specialize_zero_one: bool = True,
        duck_shape: bool | None = None,
        prefer_deferred_runtime_asserts_over_guards: bool = False,
        trace_asserts: bool = False,
        **_: Any,
    ) -> None:
        self.settings = ShapeEnvSettings(
            allow_scalar_outputs,
            allow_dynamic_output_shape_ops,
            assume_static_by_default,
            specialize_zero_one,
            True if duck_shape is None else duck_shape,
            prefer_deferred_runtime_asserts_over_guards,
            trace_asserts,
        )
        self.graph = Graph()
        self.name_to_node: dict[str, Node] = {}
        self.guards: list[ShapeGuard] = []
        self.deferred_runtime_asserts: dict[sympy.Symbol, list[RuntimeAssert]] = defaultdict(list)
        self.var_to_val: dict[sympy.Symbol, sympy.Basic] = {}
        self.backed_var_to_val = self.var_to_val
        self.var_to_range: dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_hint_override: dict[sympy.Symbol, int | float | bool] = {}
        self.var_to_sources: dict[sympy.Symbol, list[Any]] = defaultdict(list)
        self.source_to_var: dict[str, sympy.Symbol] = {}
        self.var_to_stack: dict[sympy.Symbol, Any] = {}
        self.replacements: dict[sympy.Symbol, sympy.Expr] = {}
        self.unbacked_renamings: dict[sympy.Symbol, sympy.Symbol] = {}
        self.pending_fresh_unbacked_symbols: list[sympy.Symbol] = []
        self.unbacked_inputs: set[sympy.Symbol] = set()
        self.unbacked_symint_counter = 0
        self.symbol_counter = itertools.count()
        self._version = 0
        self._frozen = False
        self._suppress_guards = 0
        self._runtime_asserts_frozen = False
        self._duck_symbols: dict[int, sympy.Symbol] = {}
        self.size_like: set[sympy.Symbol] = set()
        self.counter: dict[str, int] = defaultdict(int)
        self.var_to_hint = self.var_to_hint_override
        self.input_contexts: list[SymbolicContext] = []
        self._shape_id_to_unbacked_symbol: dict[str, sympy.Symbol] = {}

    def allow_scalar_outputs(self) -> bool:
        return self.settings.allow_scalar_outputs

    def allow_dynamic_output_shape_ops(self) -> bool:
        return self.settings.allow_dynamic_output_shape_ops

    def assume_static_by_default(self) -> bool:
        return self.settings.assume_static_by_default

    def specialize_zero_one(self) -> bool:
        return self.settings.specialize_zero_one

    def duck_shape(self) -> bool:
        return self.settings.duck_shape

    def prefer_deferred_runtime_asserts_over_guards(self) -> bool:
        return self.settings.prefer_deferred_runtime_asserts_over_guards

    @contextmanager
    def _recording(self) -> Iterable[None]:
        old = self.is_recording
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = old

    def _snapshot_tracked_fakes(self) -> list[Any] | None:
        return list(self.tracked_fakes) if self.tracked_fakes is not None else None

    def _last_event_index(self) -> int:
        return len(self.events) - 1

    def check_equal(self, other: "ShapeEnv") -> None:
        shape_env_check_state_equal(
            self,
            other,
            ("graph", "events", "tracked_fakes", "is_recording"),
        )

    def _check_frozen(self, expr: Any, concrete_val: Any = None) -> None:
        if self._frozen and self._suppress_guards == 0:
            raise _ShapeEnvGuardError(f"guard attempted while the shape environment is frozen: {expr} == {concrete_val}")

    @contextmanager
    def suppress_guards(self) -> Iterable[None]:
        self._suppress_guards += 1
        try:
            yield
        finally:
            self._suppress_guards -= 1

    @contextmanager
    def error_on_new_guards(self) -> Iterable[None]:
        with self.suppress_guards():
            yield

    def freeze(self) -> None:
        self._frozen = True

    def freeze_runtime_asserts(self) -> None:
        self._runtime_asserts_frozen = True

    def _source_symbol_name(self, source: Any) -> str:
        return _source_name(source)

    def _new_symbol(self, prefix: str = "s", *, positive: bool | None = True) -> sympy.Symbol:
        index = next(self.symbol_counter)
        name = f"{prefix}{index}"
        return sympy.Symbol(name, integer=True, positive=positive)

    @record_shapeenv_event()
    def create_symbol(
        self,
        val: int | float | SymInt | SymFloat,
        source: Any,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,
        positive: bool | None = True,
        do_not_specialize_zero_one: bool = False,
        symbolic_context: SymbolicContext | None = None,
    ) -> sympy.Expr:
        hint = optimization_hint(val, fallback=None) if isinstance(val, SymNode) else val
        if isinstance(hint, SymNode):
            hint = hint.hint
        if hint is None:
            raise ValueError("symbol creation requires a concrete hint")
        if constraint_dim is not None and isinstance(constraint_dim, StrictMinMaxConstraint):
            if constraint_dim.vr.is_singleton():
                dynamic_dim = DimDynamic.STATIC
            else:
                self._apply_range_hint(constraint_dim.vr, hint)
        if dynamic_dim is DimDynamic.STATIC or (
            self.specialize_zero_one() and hint in (0, 1) and not do_not_specialize_zero_one
        ):
            return sympy.Integer(hint)
        source_name = self._source_symbol_name(source)
        if source_name in self.source_to_var:
            return self.source_to_var[source_name]
        if dynamic_dim is DimDynamic.DUCK and self.duck_shape() and hint in self._duck_symbols:
            symbol = self._duck_symbols[hint]
        else:
            symbol = self._new_symbol("u" if dynamic_dim is DimDynamic.UNBACKED else "s", positive=positive)
        self.source_to_var[source_name] = symbol
        self.var_to_sources[symbol].append(source)
        if dynamic_dim is DimDynamic.UNBACKED:
            self.unbacked_inputs.add(symbol)
            self.pending_fresh_unbacked_symbols.append(symbol)
            self.var_to_range[symbol] = ValueRanges(0, sympy.oo)
        else:
            self.backed_var_to_val[symbol] = sympy.sympify(hint)
            lower = 0 if positive is True else -sympy.oo
            self.var_to_range[symbol] = ValueRanges(lower, sympy.oo)
            self.var_to_hint_override[symbol] = hint
            self._duck_symbols.setdefault(int(hint), symbol)
            if self._suppress_guards == 0:
                self.guards.append(ShapeGuard(sympy.Eq(symbol, hint), source))
        if constraint_dim is not None and isinstance(constraint_dim, StrictMinMaxConstraint):
            self._constrain_range(symbol, constraint_dim.vr.lower, constraint_dim.vr.upper)
        self.counter["create_symbol"] += 1
        return symbol

    def create_unspecified_symbol(
        self,
        val: int | float | SymInt | SymFloat,
        source: Any,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,
        symbolic_context: SymbolicContext | None = None,
    ) -> sympy.Expr:
        return self.create_symbol(
            val,
            source,
            dynamic_dim,
            constraint_dim,
            positive=None,
            do_not_specialize_zero_one=True,
            symbolic_context=symbolic_context,
        )

    def _apply_range_hint(self, value_range: ValueRanges, hint: Any) -> None:
        if not value_range.lower <= hint <= value_range.upper:
            raise ConstraintViolationError(f"value {hint} is outside {value_range}")

    def _node_evaluator(self, expr: sympy.Basic) -> Callable[[dict[str, Any]], Any]:
        def evaluate(values: dict[str, Any]) -> Any:
            substitutions: dict[Any, Any] = dict(self.backed_var_to_val)
            substitutions.update({sympy.Symbol(str(key)): value for key, value in values.items()})
            result = self.replace(expr).subs(substitutions)
            if result.free_symbols:
                raise GuardOnDataDependentSymNode(expr, f"cannot evaluate {expr}")
            return _primitive(result)

        return evaluate

    @record_shapeenv_event()
    def create_symintnode(
        self,
        sym: sympy.Expr,
        *,
        hint: int | float | bool | SymInt | None = None,
        source: Any | None = None,
    ) -> int | SymInt:
        del source
        expr = _expr(sym)
        if isinstance(expr, sympy.Integer):
            return int(expr)
        actual_hint = hint.hint if isinstance(hint, SymNode) else hint
        if actual_hint is None:
            static = self._maybe_evaluate_static(expr)
            actual_hint = static if isinstance(static, int) else None
        return SymInt(expr, self, int, actual_hint, self._node_evaluator(expr))

    @record_shapeenv_event()
    def create_symfloatnode(
        self,
        sym: sympy.Expr,
        *,
        hint: int | float | bool | SymFloat | None = None,
        source: Any | None = None,
    ) -> float | SymFloat:
        del source
        expr = _expr(sym)
        if isinstance(expr, sympy.Float):
            return float(expr)
        actual_hint = hint.hint if isinstance(hint, SymNode) else hint
        return SymFloat(expr, self, float, actual_hint, self._node_evaluator(expr))

    def create_symboolnode(self, sym: sympy.Expr) -> SymBool:
        expr = _expr(sym)
        return SymBool(expr, self, bool, None, self._node_evaluator(expr))

    @record_shapeenv_event()
    def create_unbacked_symint(self, source: Any | None = None) -> SymInt:
        symbol = self._new_symbol("u", positive=True)
        self.unbacked_symint_counter += 1
        self.unbacked_inputs.add(symbol)
        self.pending_fresh_unbacked_symbols.append(symbol)
        self.var_to_range[symbol] = ValueRanges(0, sympy.oo)
        if source is not None:
            self.var_to_sources[symbol].append(source)
            self.source_to_var[_source_name(source)] = symbol
        return SymInt(symbol, self, int, None, self._node_evaluator(symbol))

    @record_shapeenv_event()
    def create_unbacked_symfloat(self) -> SymFloat:
        symbol = self._new_symbol("u", positive=None)
        return SymFloat(symbol, self, float, None, self._node_evaluator(symbol))

    @record_shapeenv_event()
    def create_unbacked_symbool(self) -> SymBool:
        symbol = self._new_symbol("u", positive=True)
        self.var_to_range[symbol] = ValueRanges(0, 1)
        expr = sympy.Eq(symbol, 1)
        return SymBool(expr, self, bool, None, self._node_evaluator(expr))

    def create_unspecified_symint_and_symbol(
        self,
        value: int,
        source: Any,
        dynamic_dim: DimDynamic,
        excluded_value: int | None = None,
    ) -> int | SymInt:
        symbol = self.create_unspecified_symbol(value, source, dynamic_dim)
        if excluded_value is not None and isinstance(symbol, sympy.Symbol):
            self.exclusion_constraints = getattr(self, "exclusion_constraints", [])
            self.exclusion_constraints.append((symbol, excluded_value))
        return self.create_symintnode(symbol, hint=value, source=source)

    def _create_symbolic_sizes_strides_storage_offset(
        self,
        sizes: Sequence[Any],
        strides: Sequence[Any],
        storage_offset: Any,
        source: Any,
        symbolic_context: StatelessSymbolicContext | None = None,
    ) -> tuple[tuple[Int, ...], tuple[Int, ...], Int]:
        dynamic_sizes = symbolic_context.dynamic_sizes if symbolic_context is not None else [DimDynamic.STATIC if self.assume_static_by_default() else DimDynamic.DUCK] * len(sizes)
        dynamic_strides = symbolic_context.dynamic_strides if symbolic_context is not None else [DimDynamic.INFER_STRIDE] * len(strides)
        constraint_sizes = symbolic_context.constraint_sizes if symbolic_context is not None else [None] * len(sizes)
        symbolic_sizes: list[Int] = []
        for index, value in enumerate(sizes):
            if isinstance(value, SymNode):
                symbolic_sizes.append(value)
            else:
                symbolic_sizes.append(
                    self.create_symintnode(
                        self.create_symbol(
                            int(value),
                            _Source(f"{_source_name(source)}.size[{index}]", index),
                            dynamic_sizes[index] if index < len(dynamic_sizes) else DimDynamic.DUCK,
                            constraint_sizes[index] if index < len(constraint_sizes) else None,
                        ),
                        hint=int(value),
                    )
                )
        symbolic_strides: list[Int] = []
        for index, value in enumerate(strides):
            policy = dynamic_strides[index] if index < len(dynamic_strides) else DimDynamic.INFER_STRIDE
            if policy is DimDynamic.INFER_STRIDE:
                symbolic_strides.append(int(value))
            else:
                symbolic_strides.append(
                    self.create_symintnode(
                        self.create_symbol(
                            int(value),
                            _Source(f"{_source_name(source)}.stride[{index}]", index),
                            policy,
                        ),
                        hint=int(value),
                    )
                )
        offset = int(storage_offset) if not isinstance(storage_offset, SymNode) else storage_offset
        return tuple(symbolic_sizes), tuple(symbolic_strides), offset

    def create_symbolic_sizes_strides_storage_offset(
        self,
        ex: Any,
        source: Any,
        *,
        symbolic_context: SymbolicContext | None = None,
    ) -> tuple[tuple[Int, ...], tuple[Int, ...], Int]:
        sizes = _shape_value(ex, "size", _shape_value(ex, "shape", ()))
        strides = _shape_value(ex, "stride", create_contiguous(tuple(sizes)))
        offset = _shape_value(ex, "storage_offset", 0)
        return self._create_symbolic_sizes_strides_storage_offset(
            tuple(sizes), tuple(strides), offset, source, symbolic_context if isinstance(symbolic_context, StatelessSymbolicContext) else None
        )

    def _constrain_range(self, expr: sympy.Expr, min: int | float, max: int | float) -> None:
        self._check_frozen(expr, (min, max))
        symbol = next(iter(_expr(expr).free_symbols), None)
        if symbol is None:
            value = _primitive(_expr(expr))
            if not min <= value <= max:
                raise ConstraintViolationError(f"{value} is outside [{min}, {max}]")
            return
        old = self.var_to_range.get(symbol, ValueRanges(-sympy.oo, sympy.oo))
        lower = max_value(old.lower, min)
        upper = min_value(old.upper, max)
        if lower > upper:
            raise ConstraintViolationError(f"range for {symbol} became empty")
        self.var_to_range[symbol] = ValueRanges(lower, upper)
        self._version += 1

    def _constrain_range_for_size(self, expr: sympy.Expr, min: int | None = 0, max: int | None = None) -> None:
        self._constrain_range(expr, 0 if min is None else min, sympy.oo if max is None else max)

    def _constrain_is_bounded(self, expr: sympy.Expr, upper_bound: int) -> None:
        self._constrain_range(expr, 0, upper_bound)

    def _constrain_unify(self, left: Any, right: Any) -> None:
        left_expr, right_expr = _expr(left), _expr(right)
        self._check_frozen(left_expr, right_expr)
        if left_expr == right_expr:
            return
        if isinstance(left_expr, sympy.Symbol):
            self.replacements[left_expr] = right_expr
        elif isinstance(right_expr, sympy.Symbol):
            self.replacements[right_expr] = left_expr
        else:
            self.guards.append(ShapeGuard(sympy.Eq(left_expr, right_expr)))
        self._version += 1

    def add_backed_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        if expr in self.backed_var_to_val:
            raise AssertionError(f"{expr} already exists")
        self.backed_var_to_val[expr] = sympy.Integer(val)

    def add_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        self.add_backed_var_to_val(expr, val)

    def set_real_tensor_prop_unbacked_vals(self, symbol: sympy.Symbol, value: Any) -> None:
        if not isinstance(symbol, sympy.Symbol):
            raise TypeError(f"expected a symbol, got {type(symbol).__name__}")
        self.var_to_hint_override[symbol] = value

    @contextmanager
    def patch_source_specialization(
        self, source: Any, check_fn: Callable[[sympy.Symbol], Any]
    ) -> Iterable[None]:
        name = _source_name(source)
        symbol = self.source_to_var.get(name)
        if symbol is None:
            raise KeyError(f"no symbol is registered for source {name!r}")
        expression = _expr(check_fn(symbol))
        previous_guards = list(self.guards)
        self.guards.append(ShapeGuard(expression, source))
        try:
            yield
        finally:
            self.guards = previous_guards

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        return str(symbol).startswith("u")

    def _find(self, expr: sympy.Symbol) -> sympy.Expr:
        seen: set[sympy.Symbol] = set()
        current: sympy.Expr = expr
        while isinstance(current, sympy.Symbol) and current in self.replacements and current not in seen:
            seen.add(current)
            current = self.replacements[current]
        return current

    def replace(self, expr: Any) -> sympy.Expr:
        result = _expr(expr)
        for _ in range(len(self.replacements) + 1):
            updated = result.xreplace(self.replacements)
            if updated == result:
                break
            result = updated
        return result

    def _maybe_evaluate_static(self, expr: Any, *, compute_hint: bool = False, axioms: Iterable[Any] = (), size_oblivious: bool = False) -> Any:
        del compute_hint, axioms, size_oblivious
        result = self.replace(expr).subs(self.backed_var_to_val)
        if not result.free_symbols:
            return _primitive(result)
        return None

    def simplify(self, expr: Any, **_: Any) -> sympy.Expr:
        return sympy.simplify(self.replace(expr))

    def bound_sympy(self, expr: sympy.Expr, size_oblivious: bool = False) -> ValueRanges:
        del size_oblivious
        result = self.replace(expr)
        if not result.free_symbols:
            value = _primitive(result)
            return ValueRanges(value, value)
        lower, upper = -sympy.oo, sympy.oo
        if isinstance(result, sympy.Symbol) and result in self.var_to_range:
            return self.var_to_range[result]
        return ValueRanges(lower, upper)

    def guarding_hint_or_throw(self, expr: sympy.Expr | int) -> int | bool:
        value = self._maybe_evaluate_static(expr)
        if value is None:
            expression = self.replace(expr)
            if isinstance(expression, sympy.Symbol) and expression in self.var_to_hint_override:
                return self.var_to_hint_override[expression]
            raise GuardOnDataDependentSymNode(expression, f"cannot guard {expression}")
        self.guards.append(ShapeGuard(sympy.Eq(self.replace(expr), value)))
        return value

    def has_guarding_hint(self, expr: sympy.Expr) -> bool:
        return self._maybe_evaluate_static(expr) is not None or self.replace(expr) in self.var_to_hint_override

    def optimization_hint(self, expr: sympy.Expr, fallback: int | None = None) -> int:
        value = self._maybe_evaluate_static(expr)
        if value is not None:
            return int(value)
        expression = self.replace(expr)
        if expression in self.var_to_hint_override:
            return int(self.var_to_hint_override[expression])
        if fallback is not None:
            return fallback
        raise RuntimeError(f"no optimization hint for {expression}")

    def size_hint(self, expr: sympy.Expr, fallback: int | None = None) -> int:
        return self.optimization_hint(expr, fallback)

    def evaluate_sym_node(self, node: SymNode, **_: Any) -> Any:
        return self.guarding_hint_or_throw(node.expr)

    def evaluate_expr(self, expr: Any, values: Mapping[Any, Any] | None = None) -> Any:
        substitutions = dict(self.backed_var_to_val)
        if values:
            substitutions.update({sympy.Symbol(str(key)): value for key, value in values.items()})
        result = self.replace(expr).subs(substitutions)
        if result.free_symbols:
            raise GuardOnDataDependentSymNode(result, f"cannot evaluate {result}")
        return _primitive(result)

    def evaluate_symexpr(self, code: str) -> int | float | bool:
        namespace = dict(SYMPY_INTERP)
        namespace.update({str(symbol): _primitive(value) for symbol, value in self.backed_var_to_val.items()})
        return eval(code, {"__builtins__": {}}, namespace)

    def deserialize_symexpr(self, code: str) -> SymInt | SymFloat | SymBool:
        value = eval(code, {"__builtins__": {}}, {str(symbol): self.create_symintnode(symbol, hint=int(val)) for symbol, val in self.backed_var_to_val.items()})
        if isinstance(value, (SymInt, SymFloat, SymBool)):
            return value
        return self.create_symintnode(_expr(value), hint=value if isinstance(value, int) else None)

    def produce_guards_verbose(
        self,
        placeholders: Sequence[Any],
        sources: Sequence[Any],
        source_ref: Callable[[Any], str] = _source_name,
        *,
        guards: list[ShapeGuard] | None = None,
        input_contexts: Sequence[SymbolicContext] | None = None,
        equalities_inputs: EqualityConstraint | None = None,
        _simplified: bool = False,
        ignore_static: bool = True,
        langs: tuple[str, ...] = ("python", "verbose_python"),
    ) -> list[_ShapeGuardsHelper]:
        del placeholders, input_contexts, equalities_inputs, _simplified, langs
        selected = self.guards if guards is None else guards
        expressions: list[str] = []
        for guard in selected:
            simplified = self.simplify(guard.expr)
            if ignore_static and not free_symbols(simplified):
                continue
            expressions.append(str(simplified))
        if sources and not expressions:
            expressions = [str(source_ref(source)) for source in ()]
        return [_ShapeGuardsHelper(expressions), _ShapeGuardsHelper(list(expressions))]

    def produce_guards(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.produce_guards_verbose(*args, **kwargs, langs=("python",))[0].exprs

    def produce_guards_expression(self, placeholders: Sequence[Any], *, guards: list[ShapeGuard] | None = None, ignore_static: bool = True) -> str | None:
        del placeholders
        expressions = self.produce_guards_verbose((), (), guards=guards, ignore_static=ignore_static)[0].exprs
        return " and ".join(expressions) if expressions else None

    def evaluate_guards_expression(self, code: str, args: Sequence[Any]) -> bool:
        return bool(eval(code, {"__builtins__": {}}, {"L": dict(zip((f"t{i}" for i in range(len(args))), args))}))

    def evaluate_guards_for_args(self, placeholders: Sequence[Any], args: Sequence[Any], *, ignore_static: bool = True) -> bool:
        bindings = self.bind_symbols(placeholders, args)
        for guard in self.guards:
            expression = self.replace(guard.expr)
            if ignore_static and not expression.free_symbols:
                continue
            expression = expression.subs(bindings)
            if expression.free_symbols:
                raise GuardOnDataDependentSymNode(
                    expression, f"cannot evaluate guard {expression}"
                )
            value = _primitive(expression)
            if not bool(value):
                return False
        return True

    def bind_symbols(self, placeholders: Sequence[Any], args: Sequence[Any]) -> dict[sympy.Symbol, int]:
        bindings: dict[sympy.Symbol, int] = {}

        def bind_expression(value: Any, concrete: Any) -> None:
            if isinstance(concrete, SymNode):
                concrete = concrete.hint
            if not isinstance(concrete, int) or isinstance(concrete, bool):
                return
            expression = self.replace(_expr(value))
            symbols = expression.free_symbols
            if not symbols:
                if _primitive(expression) != concrete:
                    raise AssertionError(f"{expression} != {concrete}")
                return
            if len(symbols) != 1:
                return
            symbol = next(iter(symbols))
            solutions = sympy.solve(sympy.Eq(expression, concrete), symbol, dict=True)
            if len(solutions) != 1 or symbol not in solutions[0]:
                return
            solution = solutions[0][symbol]
            if solution.free_symbols:
                return
            solution = _primitive(solution)
            if not isinstance(solution, int) or isinstance(solution, bool):
                return
            previous = bindings.get(symbol)
            if previous is not None and previous != solution:
                raise AssertionError(f"{previous} != {solution}")
            bindings[symbol] = solution

        def bind_pair(value: Any, concrete: Any) -> None:
            if isinstance(value, SymNode):
                bind_expression(value, concrete)
                return
            if isinstance(value, (tuple, list)) and isinstance(concrete, (tuple, list)):
                for symbolic, actual in zip(value, concrete):
                    bind_pair(symbolic, actual)
                return
            if isinstance(value, dict) and isinstance(concrete, dict):
                for key, symbolic in value.items():
                    if key in concrete:
                        bind_pair(symbolic, concrete[key])
                return
            if isinstance(value, (int, sympy.Basic)):
                bind_expression(value, concrete)

        for placeholder, argument in zip(placeholders, args):
            if isinstance(placeholder, SymNode):
                bind_expression(placeholder, argument)
                continue
            placeholder_shape = _shape_value(placeholder, "shape", None)
            argument_shape = _shape_value(argument, "shape", None)
            if placeholder_shape is not None and argument_shape is not None:
                bind_pair(placeholder_shape, argument_shape)
            placeholder_size = _shape_value(placeholder, "size", None)
            argument_size = _shape_value(argument, "size", None)
            if callable(getattr(placeholder, "size", None)) and callable(getattr(argument, "size", None)):
                placeholder_size = placeholder.size()
                argument_size = argument.size()
            if placeholder_size is not None and argument_size is not None:
                bind_pair(placeholder_size, argument_size)
            placeholder_stride = _shape_value(placeholder, "stride", None)
            argument_stride = _shape_value(argument, "stride", None)
            if callable(getattr(placeholder, "stride", None)) and callable(getattr(argument, "stride", None)):
                placeholder_stride = placeholder.stride()
                argument_stride = argument.stride()
            if placeholder_stride is not None and argument_stride is not None:
                bind_pair(placeholder_stride, argument_stride)
            placeholder_offset = _shape_value(placeholder, "storage_offset", None)
            argument_offset = _shape_value(argument, "storage_offset", None)
            if placeholder_offset is not None and argument_offset is not None:
                bind_pair(placeholder_offset, argument_offset)
        return bindings

    def get_pruned_guards(self, symints: Sequence[SymInt]) -> list[ShapeGuard]:
        symbols = set().union(*(free_symbols(value) for value in symints))
        return [guard for guard in self.guards if free_symbols(guard.expr) <= symbols]

    def get_nontrivial_guards(self) -> list[sympy.Basic]:
        return [guard.expr for guard in self.guards if self._maybe_evaluate_static(guard.expr) is None]

    def format_guards(self, verbose: bool = False) -> str:
        del verbose
        return "\n".join(f" - {guard.expr}" for guard in self.guards)

    def get_axioms(self, symbols: tuple[sympy.Symbol, ...] | None = None, compute_hint: bool = False) -> tuple[sympy.Basic, ...]:
        del compute_hint
        if symbols is None:
            return tuple(guard.expr for guard in self.guards)
        return tuple(guard.expr for guard in self.guards if free_symbols(guard.expr) & set(symbols))

    def get_implications(self, expr: Any) -> tuple[sympy.Basic, ...]:
        return tuple(guard.expr for guard in self.guards if free_symbols(guard.expr) & free_symbols(expr))

    def cleanup(self) -> None:
        self.guards = list(dict.fromkeys(self.guards))

    def guard_or_defer_runtime_assert(
        self, expression: Any, message: str, node: Node | None = None
    ) -> bool:
        del node
        value = self._maybe_evaluate_static(expression)
        if value is not None:
            if not bool(value):
                raise ConstraintViolationError(message)
            return True
        expression = self.replace(expression)
        if (
            not self.prefer_deferred_runtime_asserts_over_guards()
            and expression.free_symbols <= self.backed_var_to_val.keys()
        ):
            self.guards.append(ShapeGuard(expression))
            return True
        symbol = next(iter(expression.free_symbols), None)
        if symbol is None:
            raise ConstraintViolationError(message)
        if self._runtime_asserts_frozen:
            raise _ShapeEnvGuardError("runtime assertions are frozen")
        self.deferred_runtime_asserts[symbol].append(
            RuntimeAssert(expression, message)
        )
        return True

    def constrain_symbol_range(
        self, symbol: sympy.Symbol, compiler_min: int, compiler_max: int
    ) -> None:
        if compiler_min > compiler_max:
            raise ValueError("minimum cannot exceed maximum")
        self._constrain_range(symbol, compiler_min, compiler_max)


def max_value(left: Any, right: Any) -> Any:
    try:
        return max(left, right)
    except TypeError:
        return right if left == -sympy.oo else left


def min_value(left: Any, right: Any) -> Any:
    try:
        return min(left, right)
    except TypeError:
        return right if left == sympy.oo else left


def graph_placeholder_vals(graph_module: Any) -> list[Any]:
    return [node.meta.get("val") for node in graph_module.graph.nodes if node.op == "placeholder"]


def graph_placeholder_targets(graph_module: Any) -> list[str]:
    return [node.target for node in graph_module.graph.nodes if node.op == "placeholder"]


def eval_guards(graph_module: Any, *args: Any, ignore_static: bool = True) -> bool:
    shape_env = getattr(graph_module, "shape_env", None)
    if shape_env is None:
        raise AssertionError("graph module has no shape environment")
    return shape_env.evaluate_guards_for_args(graph_placeholder_vals(graph_module), args, ignore_static=ignore_static)


def bind_symbols(graph_module: Any, *args: Any) -> dict[sympy.Symbol, int]:
    shape_env = getattr(graph_module, "shape_env", None)
    if shape_env is None:
        raise AssertionError("graph module has no shape environment")
    return shape_env.bind_symbols(graph_placeholder_vals(graph_module), args)


class SymExprPrinter(PythonCodePrinter):
    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))


class ShapeGuardPythonPrinter(PythonCodePrinter):
    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Any]] | None = None,
        source_ref: Callable[[Any], str] | None = None,
        var_to_sources: Mapping[sympy.Symbol, list[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.symbol_to_source = symbol_to_source or {}
        self.source_ref = source_ref or _source_name
        self.var_to_sources = var_to_sources or {}
        self._print_cache: dict[sympy.Basic, str] = {}

    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        sources = self.symbol_to_source.get(expr) or self.var_to_sources.get(expr)
        if sources:
            return self.print_source(sources[0])
        return str(expr)

    def _print_Max(self, expr: sympy.Basic) -> str:
        return self._fold_call("max", expr.args)

    def _print_Min(self, expr: sympy.Basic) -> str:
        return self._fold_call("min", expr.args)

    def _fold_call(self, name: str, args: Sequence[sympy.Basic]) -> str:
        if not args:
            raise ValueError(f"{name} requires at least one argument")
        result = self.doprint(args[0])
        for arg in args[1:]:
            result = f"{name}({result}, {self.doprint(arg)})"
        return result

    def print_source(self, source: Any) -> str:
        return self.source_ref(source)

    def doprint(self, expr: sympy.Basic) -> str:
        cached = self._print_cache.get(expr)
        if cached is None:
            cached = super().doprint(expr)
            self._print_cache[expr] = cached
        return cached


class ShapeGuardPrinter(ShapeGuardPythonPrinter):
    pass


class LoggingShapeGuardPrinter(ShapeGuardPythonPrinter):
    def __init__(self, var_to_sources: Mapping[sympy.Symbol, list[Any]]) -> None:
        super().__init__(var_to_sources, _source_name, var_to_sources)


class DynamicDimConstraintPrinter(PythonCodePrinter):
    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Any]],
        source_name_to_debug_name: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name or {}

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        sources = self.symbol_to_source.get(expr)
        if not sources:
            raise AssertionError(f"unknown dimension symbol {expr}")
        return _source_name(sources[0])


class DimConstraints:
    """Collect and solve scalar constraints attached to symbolic dimensions."""

    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Any]],
        var_to_val: Mapping[sympy.Symbol, Any],
        marked_dynamic: set[sympy.Symbol],
        source_name_to_debug_name: Mapping[str, str] | None = None,
    ) -> None:
        self._var_to_val = dict(var_to_val)
        self._marked_dynamic = set(marked_dynamic)
        self._dcp = DynamicDimConstraintPrinter(
            symbol_to_source, source_name_to_debug_name
        )
        self._univariate_inequalities: dict[sympy.Symbol, set[sympy.Basic]] = defaultdict(set)
        self._symbols_with_equalities: set[sympy.Symbol] = set()
        self._substitutions: dict[sympy.Symbol, sympy.Basic] = {}
        self._multivariate_inequalities: set[sympy.Basic] = set()
        self._symbolic_equivalences: list[tuple[Any, sympy.Basic]] = []
        self._static_results: set[str] = set()
        self._dynamic_results: set[str] = set()
        self._inconsistencies: list[str] = []

    @property
    def static_results(self) -> set[str]:
        return set(self._static_results)

    @property
    def dynamic_results(self) -> set[str]:
        return set(self._dynamic_results)

    def _raise_inconsistencies(self) -> None:
        if self._inconsistencies:
            message = "\n".join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f"inconsistent dimension constraints:\n{message}")

    def add(self, expression: Any) -> bool:
        expr_value = _expr(expression)
        reduced = expr_value.xreplace(self._var_to_val)
        if reduced == sympy.true:
            return True
        if reduced == sympy.false:
            self._inconsistencies.append(f"{expr_value} is false")
        symbols = expr_value.free_symbols
        if not symbols:
            raise AssertionError(f"constraint has no free symbols: {expr_value}")
        if len(symbols) != 1 or isinstance(expr_value, (sympy.And, sympy.Or, sympy.Ne)):
            self._multivariate_inequalities.add(expr_value)
            return False
        symbol = next(iter(symbols))
        self._univariate_inequalities[symbol].add(expr_value)
        if isinstance(expr_value, sympy.Equality):
            self._symbols_with_equalities.add(symbol)
        return False

    def add_equality(self, source: Any, expression: Any) -> None:
        expr_value = _expr(expression)
        if not expr_value.free_symbols:
            self._static_results.add(f"{_source_name(source)} == {expr_value}")
        else:
            self._symbolic_equivalences.append((source, expr_value))

    def _source_for(self, symbol: sympy.Symbol) -> str:
        sources = self._dcp.symbol_to_source.get(symbol)
        return _source_name(sources[0]) if sources else str(symbol)

    def solve(self) -> None:
        self._raise_inconsistencies()
        while self._symbols_with_equalities:
            symbol = self._symbols_with_equalities.pop()
            expressions = self._univariate_inequalities.pop(symbol)
            try:
                reduced = sympy.reduce_inequalities(expressions, symbol)
            except (NotImplementedError, ValueError):
                self._dynamic_results.update(self._dcp.doprint(expr) for expr in expressions)
                continue
            solutions = sympy.solve(reduced, symbol, dict=True)
            if len(solutions) != 1 or symbol not in solutions[0]:
                self._dynamic_results.add(self._dcp.doprint(reduced))
                continue
            value = sympy.sympify(solutions[0][symbol])
            self._substitutions[symbol] = value
            self._static_results.add(f"{self._source_for(symbol)} == {value}")
            pending = list(self._multivariate_inequalities)
            self._multivariate_inequalities.clear()
            for expression in pending:
                self.add(expression.xreplace({symbol: value}))
            self._raise_inconsistencies()

        for symbol, expressions in self._univariate_inequalities.items():
            substitutions = {symbol: self._substitutions[symbol]} if symbol in self._substitutions else {}
            for expression in expressions:
                reduced_expression = expression.xreplace(substitutions)
                if reduced_expression in (sympy.true, sympy.false):
                    if reduced_expression is sympy.false:
                        self._inconsistencies.append(str(expression))
                    continue
                try:
                    reduced_expression = sympy.reduce_inequalities([reduced_expression], symbol)
                except (NotImplementedError, ValueError):
                    pass
                self._dynamic_results.add(self._dcp.doprint(reduced_expression))

        for source, expression in self._symbolic_equivalences:
            expression = expression.xreplace(self._substitutions)
            if not expression.free_symbols:
                self._static_results.add(f"{_source_name(source)} == {expression}")
            else:
                self._dynamic_results.add(
                    f"{_source_name(source)} == {self._dcp.doprint(expression)}"
                )
        self._raise_inconsistencies()

    def forced_specializations(self) -> dict[str, sympy.Basic]:
        return {
            self._source_for(symbol): value
            for symbol, value in self._substitutions.items()
            if symbol in self._marked_dynamic
        }

    @classmethod
    def _is_supported_congruence(cls, expression: sympy.Basic) -> bool:
        return bool(expression.has(sympy.Mod))


class PropagateUnbackedSymInts(Interpreter):
    """Propagate recorded symbolic bindings while interpreting a graph."""

    def __init__(self, module: Any, *args: Any, shape_env: ShapeEnv | None = None, **kwargs: Any) -> None:
        super().__init__(module, *args, **kwargs)
        self.shape_env = shape_env or getattr(module, "shape_env", None)

    def run_node(self, node: Node) -> Any:
        result = super().run_node(node)
        rebind_unbacked(self.shape_env, node, result)
        return result
