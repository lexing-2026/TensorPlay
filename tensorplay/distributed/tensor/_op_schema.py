"""Structured records used by distributed placement propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Iterable, Mapping, Sequence

from ._dtensor_spec import DTensorSpec, TensorMeta

__all__ = [
    "OpInfo",
    "OpSpec",
    "OpSchema",
    "OpStrategy",
    "OutputSharding",
    "PlacementStrategy",
    "RuntimeSchemaInfo",
    "StrategyType",
    "TupleStrategy",
]


def _rebuild_tensor_from_dtensor_meta(arg: DTensorSpec) -> Any:
    if arg.tensor_meta is None:
        raise AssertionError("DTensorSpec does not contain tensor metadata")
    import tensorplay

    meta = arg.tensor_meta
    factory = getattr(tensorplay, "empty_strided", None)
    if factory is not None:
        return factory(meta.shape, meta.stride, dtype=meta.dtype)
    return tensorplay.empty(meta.shape, dtype=meta.dtype)


def _tree_leaves(value: Any) -> list[Any]:
    if isinstance(value, TensorMeta):
        return [value]
    if isinstance(value, Mapping):
        result: list[Any] = []
        for item in value.values():
            result.extend(_tree_leaves(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_tree_leaves(item))
        return result
    return [value]


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, TensorMeta):
        return function(value)
    if isinstance(value, dict):
        return {key: _tree_map(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(item, function) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, function) for item in value]
    return function(value)


def _tree_map_only(
    value: Any, target: type[Any], function: Callable[[Any], Any]
) -> Any:
    if isinstance(value, target):
        return function(value)
    if isinstance(value, TensorMeta):
        return value
    if isinstance(value, dict):
        return {
            key: _tree_map_only(item, target, function)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_tree_map_only(item, target, function) for item in value)
    if isinstance(value, list):
        return [_tree_map_only(item, target, function) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _pretty_print_spec(spec: Any) -> str:
    if spec is None:
        return "None"
    if isinstance(spec, DTensorSpec):
        return "".join(str(placement) for placement in spec.placements)
    if isinstance(spec, (tuple, list)):
        return "(" + ", ".join(_pretty_print_spec(item) for item in spec) + ")"
    return str(spec)


@dataclass
class OpSpec:
    """A valid output layout and optional input layouts for one operation."""

    output_specs: Any
    input_specs: Sequence[DTensorSpec] | None = None
    redistribute_cost: list[list[float]] | None = None

    @cached_property
    def output_spec(self) -> DTensorSpec:
        if isinstance(self.output_specs, DTensorSpec):
            return self.output_specs
        raise ValueError(
            "output_spec requires one distributed tensor result; "
            f"got {self.output_specs!r}"
        )

    def input_spec(self, index: int = 0) -> DTensorSpec:
        if self.input_specs is None:
            raise ValueError("input_specs is not set")
        if index < 0 or index >= len(self.input_specs):
            raise IndexError(
                f"input index {index} is outside {len(self.input_specs)} inputs"
            )
        return self.input_specs[index]

    @property
    def mesh(self) -> Any:
        outputs = self.output_specs
        if isinstance(outputs, DTensorSpec):
            return outputs.mesh
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if isinstance(value, DTensorSpec):
                    return value.mesh
        if self.input_specs:
            return self.input_specs[0].mesh
        raise ValueError("strategy has no distributed tensor specification")

    def __str__(self) -> str:
        prefix = ""
        if self.input_specs is not None:
            prefix = f"{_pretty_print_spec(tuple(self.input_specs))} -> "
        return prefix + _pretty_print_spec(self.output_specs)

    def __hash__(self) -> int:
        if self.output_specs is None:
            output_hash = hash(None)
        elif isinstance(self.output_specs, DTensorSpec):
            output_hash = hash(self.output_specs)
        else:
            output_hash = hash(tuple(self.output_specs))
        input_hash = hash(tuple(self.input_specs)) if self.input_specs else 0
        return hash((output_hash, input_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSpec):
            return False
        return (
            self.output_specs == other.output_specs
            and self.input_specs == other.input_specs
        )


PlacementStrategy = OpSpec


class StrategyType:
    """Base type for operation strategy collections."""


class OpStrategy(StrategyType):
    """A collection of alternative placement strategies."""

    def __init__(self, strategies: Iterable[PlacementStrategy]) -> None:
        self.strategies = list(strategies)

    def __str__(self) -> str:
        if not self.strategies:
            return "[]"
        return f"[{', '.join(str(strategy) for strategy in self.strategies)}] @ mesh: {self.mesh_shape}"

    def max_num_shards(self) -> int:
        if not self.strategies:
            return 0
        return max(strategy.output_spec.num_shards for strategy in self.strategies)

    @property
    def mesh(self) -> Any:
        if not self.strategies:
            raise ValueError("strategy has no choices")
        return self.strategies[0].mesh

    @property
    def mesh_shape(self) -> Any:
        if not self.strategies:
            return ()
        output = self.strategies[0].output_specs
        if isinstance(output, DTensorSpec):
            return getattr(output.mesh, "shape", ())
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, DTensorSpec):
                    return getattr(item.mesh, "shape", ())
        return ()

    @property
    def ndim(self) -> int:
        return self.strategies[0].output_spec.ndim

    @property
    def shape(self) -> Any:
        return self.strategies[0].output_spec.shape

    @property
    def tensor_meta(self) -> TensorMeta:
        if not self.strategies or self.strategies[0].output_spec.tensor_meta is None:
            raise ValueError("strategy output has no tensor metadata")
        return self.strategies[0].output_spec.tensor_meta

    def __hash__(self) -> int:
        return hash(tuple(self.strategies))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpStrategy) and self.strategies == other.strategies


class TupleStrategy(StrategyType):
    """Strategies for operations returning several tensor-like values."""

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        self.childs = tuple(childs)

    @property
    def children(self) -> tuple[StrategyType, ...]:
        return self.childs

    @property
    def childs(self) -> tuple[StrategyType, ...]:
        return self._children

    @childs.setter
    def childs(self, value: Sequence[StrategyType]) -> None:
        self._children = tuple(value)

    def child_mesh(self, index: int) -> Any:
        child = self.children[index]
        if not isinstance(child, OpStrategy):
            raise TypeError("tuple strategy child is not an operation strategy")
        return child.mesh

    def __str__(self) -> str:
        return "TupleStrategy(" + ", ".join(str(child) for child in self.childs) + ")"

    def __hash__(self) -> int:
        return hash(tuple(self.children))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleStrategy) and self.children == other.children


@dataclass
class RuntimeSchemaInfo:
    """Controls which non-tensor arguments participate in propagation caching."""

    static_argnum: int = 100
    static_kwargkey: list[str] | None = None
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """Operation inputs with distributed tensor values replaced by specifications."""

    func: Any
    args_schema: tuple[Any, ...] = ()
    kwargs_schema: dict[str, Any] = field(default_factory=dict)
    schema_info: RuntimeSchemaInfo | None = None
    _comparison_key: tuple[Any, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __init__(
        self,
        func: Any = None,
        args_schema: Any = (),
        kwargs_schema: Any = (),
        schema_info: RuntimeSchemaInfo | None = None,
        *,
        op: Any = None,
    ) -> None:
        if func is None:
            func = op
        elif op is not None and func != op:
            raise TypeError("func and op identify different operations")
        if isinstance(kwargs_schema, Mapping):
            kwargs = dict(kwargs_schema)
        else:
            kwargs = dict(kwargs_schema)
        self.func = func
        self.args_schema = tuple(args_schema)
        self.kwargs_schema = kwargs
        self.schema_info = schema_info
        self.__post_init__()

    @property
    def op(self) -> Any:
        return self.func

    @op.setter
    def op(self, value: Any) -> None:
        self.func = value

    @property
    def args(self) -> tuple[Any, ...]:
        return self.args_schema

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs_schema)

    @property
    def args_spec(self) -> tuple[DTensorSpec, ...]:
        values = (
            _tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else list(self.args_schema)
        )
        return tuple(value for value in values if isinstance(value, DTensorSpec))

    @property
    def args_strategy(self) -> tuple[OpStrategy, ...]:
        values = (
            _tree_leaves(self.args_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else list(self.args_schema)
        )
        return tuple(value for value in values if isinstance(value, OpStrategy))

    @property
    def kwargs_strategy(self) -> tuple[OpStrategy, ...]:
        values = (
            _tree_leaves(self.kwargs_schema)
            if self.schema_info is not None and self.schema_info.needs_pytree
            else list(self.kwargs_schema.values())
        )
        return tuple(value for value in values if isinstance(value, OpStrategy))

    @property
    def args_meta(self) -> tuple[Any, ...]:
        def convert(value: Any) -> Any:
            if isinstance(value, TensorMeta):
                return value
            if isinstance(value, OpStrategy):
                return value.tensor_meta
            if isinstance(value, TupleStrategy):
                return tuple(convert(child) for child in value.children)
            if isinstance(value, (tuple, list)):
                converted = [convert(child) for child in value]
                return type(value)(converted)
            if isinstance(value, dict):
                return {key: convert(child) for key, child in value.items()}
            return value

        return tuple(convert(value) for value in self.args_schema)

    @property
    def kwargs_meta(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, TensorMeta):
                return value
            if isinstance(value, OpStrategy):
                return value.tensor_meta
            if isinstance(value, TupleStrategy):
                return tuple(convert(child) for child in value.children)
            if isinstance(value, (tuple, list)):
                converted = [convert(child) for child in value]
                return type(value)(converted)
            if isinstance(value, dict):
                return {key: convert(child) for key, child in value.items()}
            return value

        return {key: convert(value) for key, value in self.kwargs_schema.items()}

    @property
    def has_symints(self) -> bool:
        for spec in self.args_spec:
            if spec.tensor_meta is None:
                continue
            if any(
                type(value).__name__ in {"SymInt", "SymFloat", "SymBool"}
                for value in spec.shape
            ):
                return True
        return False

    def __post_init__(self) -> None:
        self._recompute_comparison_key()

    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        value = self.args_schema[arg_idx]
        if isinstance(value, DTensorSpec):
            return True
        if isinstance(value, (list, tuple)):
            return all(item is None or isinstance(item, DTensorSpec) for item in value)
        return False

    def _returns(self) -> Sequence[Any]:
        schema = getattr(self.func, "_schema", None)
        returns = getattr(schema, "returns", ())
        return tuple(returns or ())

    @staticmethod
    def _is_tensor_type(value: Any) -> bool:
        name = getattr(value, "__name__", type(value).__name__).lower()
        return "tensor" in name

    def return_type_tuple_tensor_like(self) -> bool:
        returns = self._returns()
        if len(returns) <= 1:
            return False
        return self._is_tensor_type(getattr(returns[0], "type", returns[0]))

    def return_type_tensor(self) -> bool:
        returns = self._returns()
        return bool(returns) and self._is_tensor_type(getattr(returns[0], "type", returns[0]))

    def return_type_list_tensor_like(self) -> bool:
        returns = self._returns()
        if len(returns) != 1:
            return False
        value = getattr(returns[0], "type", returns[0])
        return "list" in getattr(value, "__name__", type(value).__name__).lower()

    def get_mesh_from_args(self, validate: bool = True) -> Any:
        mesh = None
        values = tuple(self.args_schema) + tuple(self.kwargs_schema.values())
        for value in values:
            candidates = _tree_leaves(value)
            for candidate in candidates:
                if isinstance(candidate, DTensorSpec):
                    mesh = candidate.mesh
                    break
                if isinstance(candidate, OpStrategy):
                    mesh = candidate.mesh
                    break
            if mesh is not None:
                break
        if mesh is None:
            raise ValueError("cannot find a device mesh in operation arguments")
        if validate:
            for value in values:
                for candidate in _tree_leaves(value):
                    candidate_mesh = None
                    if isinstance(candidate, DTensorSpec):
                        candidate_mesh = candidate.mesh
                    elif isinstance(candidate, OpStrategy):
                        candidate_mesh = candidate.mesh
                    if candidate_mesh is not None and candidate_mesh != mesh:
                        raise ValueError("distributed operation arguments use different meshes")
        return mesh

    def is_inplace_op(self) -> bool:
        name = str(getattr(self.func, "__name__", self.func)).rsplit(".", 1)[-1]
        return name.endswith("_")

    def is_out_variant_op(self) -> bool:
        return "out" in self.kwargs_schema

    def is_view_op(self) -> bool:
        return bool(getattr(self.func, "is_view", False))

    def __hash__(self) -> int:
        if self._comparison_key is None:
            self._recompute_comparison_key()
        return hash((self.func, self._comparison_key))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSchema) or self.func != other.func:
            return False
        if len(self.args_schema) != len(other.args_schema):
            return False
        if self._comparison_key is None:
            self._recompute_comparison_key()
        if other._comparison_key is None:
            other._recompute_comparison_key()
        return self._comparison_key == other._comparison_key

    def _recompute_comparison_key(self) -> None:
        info = self.schema_info
        static_argnum = len(self.args_schema) if info is None else info.static_argnum
        args = tuple(
            _freeze(value)
            for index, value in enumerate(self.args_schema)
            if self.arg_type_tensor_or_tensor_list_like(index) or index >= static_argnum
        )
        if info is not None and info.static_kwargkey is not None:
            kwargs = tuple(
                (key, _freeze(self.kwargs_schema.get(key)))
                for key in info.static_kwargkey
            )
        else:
            kwargs = ()
        self._comparison_key = (args, kwargs)

    @staticmethod
    def _fake_tensor(spec: DTensorSpec) -> Any:
        if spec.tensor_meta is None:
            raise ValueError("DTensorSpec does not contain tensor metadata")
        import tensorplay

        meta: TensorMeta = spec.tensor_meta
        factory = getattr(tensorplay, "empty_strided", None)
        if factory is not None:
            return factory(meta.shape, meta.stride, dtype=meta.dtype)
        return tensorplay.empty(meta.shape, dtype=meta.dtype)

    def gen_fake_args(self) -> tuple[Any, ...]:
        return tuple(
            _tree_map_only(value, DTensorSpec, _rebuild_tensor_from_dtensor_meta)
            for value in self.args_schema
        )

    def gen_fake_kwargs(self) -> dict[str, Any]:
        return {
            key: _tree_map_only(value, DTensorSpec, _rebuild_tensor_from_dtensor_meta)
            for key, value in self.kwargs_schema.items()
        }

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        replacements = iter(self.args_spec)

        def replace(value: Any) -> Any:
            if isinstance(value, DTensorSpec):
                return next(replacements)
            if isinstance(value, tuple):
                return tuple(replace(item) for item in value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        self.args_schema = tuple(replace(value) for value in origin_schema.args_schema)
        self.kwargs_schema = dict(origin_schema.kwargs_schema)
        self.schema_info = origin_schema.schema_info
        self._recompute_comparison_key()

    def __repr__(self) -> str:
        return (
            f"OpSchema(op={self.func!r}, args_schema={self.args_schema!r}, "
            f"kwargs_schema={self.kwargs_schema!r})"
        )

    def __str__(self) -> str:
        values = ", ".join(_pretty_print_spec(value) for value in self.args_schema)
        return f"Op(op={self.func!r}, args_schema=({values}))"


@dataclass
class OutputSharding:
    """Result of placement propagation and an optional redistribution request."""

    output_spec: Any
    redistribute_schema: OpSchema | None = None
    needs_redistribute: bool = False
    schema_suggestions: tuple[Any, ...] = ()
    failed_reason: str | None = None

    @property
    def mesh(self) -> Any:
        if isinstance(self.output_spec, DTensorSpec):
            return self.output_spec.mesh
        if isinstance(self.output_spec, (tuple, list)):
            for value in self.output_spec:
                if isinstance(value, DTensorSpec):
                    return value.mesh
        raise ValueError(
            f"cannot determine mesh from output spec {self.output_spec!r}"
        )


@dataclass
class OpInfo:
    """Runtime inputs and the placement result for one operation."""

    mesh: Any = None
    schema: OpSchema | None = None
    flat_args_schema: list[Any] = field(default_factory=list)
    local_args: Sequence[Any] = ()
    local_kwargs: dict[str, Any] = field(default_factory=dict)
    args_tree_spec: Any = None
    output_sharding: OutputSharding | None = None
