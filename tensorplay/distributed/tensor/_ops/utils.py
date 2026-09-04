"""Shared registration and dimension utilities for placement rules."""

from __future__ import annotations

import math
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .._api import DTensor
from .._collective_utils import redistribute_cost
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSpec, OpStrategy, OutputSharding, TupleStrategy
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)

__all__ = [
    "as_list",
    "generate_redistribute_costs",
    "expand_to_full_mesh_op_strategy",
    "infer_broadcast_dims_map",
    "is_tensor_dim_sharded",
    "is_tensor_evenly_shardable",
    "is_tensor_evenly_shardable_on_dim",
    "is_tensor_partial",
    "is_tensor_shardable",
    "map_placements_after_broadcast",
    "normalize_dim",
    "normalize_dims",
    "prod",
    "register_op_strategy",
    "register_prop_rule",
    "replicate_op_strategy",
    "shift_shard_dims_after_insert",
    "shift_shard_dims_after_remove",
]

_PROPAGATION_RULES: dict[Any, Callable[..., Any]] = {}
_STRATEGY_RULES: dict[Any, Callable[..., Any]] = {}
_NAMED_PROPAGATION_RULES: dict[str, Callable[..., Any]] = {}
_NAMED_STRATEGY_RULES: dict[str, Callable[..., Any]] = {}
_BUILTINS_READY = False


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, (DTensor, DTensorSpec))


def _get_registration_wrapper(table: dict[Any, Callable[..., Any]], operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        table[operation] = function
        return function

    return register


def register_prop_rule(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _get_registration_wrapper(_PROPAGATION_RULES, operation)


def register_op_strategy(operation: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _get_registration_wrapper(_STRATEGY_RULES, operation)


def _operation_name(operation: Any) -> str:
    if isinstance(operation, str):
        value = operation
    else:
        value = str(
            getattr(
                operation,
                "__name__",
                getattr(operation, "name", type(operation).__name__),
            )
        )
    value = value.rsplit(".", 1)[-1]
    for suffix in ("_default", "_out", "_functional"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def _schema_values(schema: Any) -> list[Any]:
    if hasattr(schema, "args"):
        values = (
            getattr(schema, "args"),
            getattr(schema, "kwargs", {}),
        )
    else:
        values = schema
    result: list[Any] = []

    def visit(value: Any) -> None:
        if _is_tensor_like(value):
            result.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(values)
    return result


def _schema_argument(schema: Any, index: int, name: str, default: Any = None) -> Any:
    if hasattr(schema, "kwargs") and name in schema.kwargs:
        return schema.kwargs[name]
    values = getattr(schema, "args", schema)
    if isinstance(values, (tuple, list)) and index < len(values):
        return values[index]
    return default


def _reduction_rule(schema: Any) -> Any:
    from ._math_ops import _get_norm_reduction_op, map_placements_after_reduction

    values = _schema_values(schema)
    if not values:
        return OutputSharding(None, failed_reason="no distributed tensor input")
    value = values[0]
    name = _operation_name(getattr(schema, "op", "sum"))
    if name == "norm":
        ord_value = schema.kwargs.get("ord", _schema_argument(schema, 1, "ord", 2))
        dim = schema.kwargs.get("dim", _schema_argument(schema, 2, "dim", None))
        keepdim = bool(
            schema.kwargs.get("keepdim", _schema_argument(schema, 3, "keepdim", False))
        )
        reduction_op = _get_norm_reduction_op(2 if ord_value is None else ord_value)
    else:
        dim = _schema_argument(schema, 1, "dim", None)
        keepdim = bool(_schema_argument(schema, 2, "keepdim", False))
        reduction_op = None if name in {"all", "any", "var", "std"} else name
    return OutputSharding(
        map_placements_after_reduction(
            DTensorSpec(value.device_mesh, value.placements, getattr(value, "tensor_meta", None)),
            dim,
            keepdim,
            reduction_op,
        )
    )


def _matrix_rule(schema: Any) -> Any:
    from ._matrix_ops import linear_single_dim_strategy, mm_single_dim_strategy

    values = _schema_values(schema)
    name = _operation_name(getattr(schema, "op", "mm"))
    bias = None
    if name in {"addmm", "baddbmm"}:
        names = ("input", "mat1", "mat2") if name == "addmm" else ("input", "batch1", "batch2")
        raw_bias = _schema_argument(schema, 0, names[0], None)
        raw_left = _schema_argument(schema, 1, names[1], None)
        raw_right = _schema_argument(schema, 2, names[2], None)
        if not _is_tensor_like(raw_left) or not _is_tensor_like(raw_right):
            return OutputSharding(None, failed_reason="matrix operation needs bias and two tensors")
        bias = raw_bias if _is_tensor_like(raw_bias) else None
        left, right = raw_left, raw_right
    elif len(values) >= 2:
        left, right = values[:2]
        if name == "linear" and len(values) >= 3:
            bias = values[2]
    else:
        return OutputSharding(None, failed_reason="matrix operation needs two tensors")
    if name == "linear":
        return OutputSharding(linear_single_dim_strategy(left, right, bias=bias))
    return OutputSharding(mm_single_dim_strategy(left, right, bias=bias))


def _single_input_rule(schema: Any) -> Any:
    values = _schema_values(schema)
    if not values:
        return OutputSharding(None, failed_reason="no distributed tensor input")
    value = values[0]
    return OutputSharding(
        DTensorSpec(value.device_mesh, value.placements, getattr(value, "tensor_meta", None))
    )


def _install_builtin_rules() -> None:
    global _BUILTINS_READY
    if _BUILTINS_READY:
        return
    _BUILTINS_READY = True
    from ._conv_ops import convolution_backward_rules, convolution_rules
    from ._embedding_ops import embedding_dense_backward_strategy, embedding_strategy
    from ._experimental_ops import slice_backward_rules
    from ._math_ops import register_math_ops
    from ._random_ops import register_random_ops
    from ._tensor_ops import register_tensor_ops
    from ._pointwise_ops import register_pointwise_ops
    from ._view_ops import register_view_ops

    reduction_names = {
        "sum", "mean", "prod", "amin", "amax", "min", "max", "all", "any",
        "var", "std", "norm",
    }
    for name in reduction_names:
        _NAMED_PROPAGATION_RULES[name] = _reduction_rule
    for name in {"mm", "matmul", "bmm", "addmm", "baddbmm", "linear"}:
        _NAMED_PROPAGATION_RULES[name] = _matrix_rule
    for name in {"to"}:
        _NAMED_PROPAGATION_RULES[name] = _single_input_rule
    for name in {"convolution", "convolution_backward"}:
        _NAMED_PROPAGATION_RULES[name] = (
            convolution_backward_rules if name.endswith("backward") else convolution_rules
        )
    _NAMED_STRATEGY_RULES.update({
        "embedding": embedding_strategy,
        "embedding_dense_backward": embedding_dense_backward_strategy,
        "slice_backward": slice_backward_rules,
    })
    register_tensor_ops()
    register_math_ops()
    register_pointwise_ops()
    register_random_ops()
    register_view_ops()
    from .autogen import auto_register_op_variants

    auto_register_op_variants()


def _lookup_builtin_rule(operation: Any) -> tuple[str, Callable[..., Any] | None]:
    _install_builtin_rules()
    name = _operation_name(operation)
    rule = _NAMED_PROPAGATION_RULES.get(name)
    if rule is not None:
        return "prop", rule
    rule = _NAMED_STRATEGY_RULES.get(name)
    if rule is not None:
        return "strategy", rule
    return "", None


def replicate_op_strategy(op_schema: Any) -> Any:
    inputs = getattr(op_schema, "args", ())
    template = next((value for value in inputs if _is_tensor_like(value)), None)
    if template is None:
        return None
    return DTensorSpec(template.device_mesh, tuple(Replicate() for _ in template.placements), None)


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (tuple, list)) else [value]


def normalize_dim(dim: int, ndim: int) -> int:
    if type(dim) is not int:
        raise TypeError("dimension must be an integer")
    result = dim + ndim if dim < 0 else dim
    if result < 0 or result >= ndim:
        raise IndexError(f"dimension {dim} is outside rank {ndim}")
    return result


def normalize_dims(dims: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(ndim))
    values = (dims,) if isinstance(dims, int) else tuple(dims)
    result = tuple(normalize_dim(value, ndim) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("dimensions must be unique")
    return result


def shift_shard_dims_after_insert(
    placements: Sequence[Placement], insert_dim: int = 0
) -> tuple[Placement, ...]:
    result: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard) and placement.dim >= insert_dim:
            result.append(
                _StridedShard(placement.dim + 1, split_factor=placement.split_factor)
            )
        elif isinstance(placement, Shard) and placement.dim >= insert_dim:
            result.append(Shard(placement.dim + 1))
        else:
            result.append(placement)
    return tuple(result)


def shift_shard_dims_after_remove(
    placements: Sequence[Placement], remove_dim: int = 0
) -> tuple[Placement, ...]:
    result: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard) and placement.dim > remove_dim:
            result.append(
                _StridedShard(placement.dim - 1, split_factor=placement.split_factor)
            )
        elif isinstance(placement, Shard) and placement.dim > remove_dim:
            result.append(Shard(placement.dim - 1))
        else:
            result.append(placement)
    return tuple(result)


def prod(values: Iterable[int]) -> int:
    return math.prod(values)


def is_tensor_shardable(
    shape: Sequence[int],
    spec: DTensorSpec,
    allow_unbacked_sharding: bool | None = None,
    *,
    dim: int | None = None,
) -> bool:
    if type(allow_unbacked_sharding) is int and dim is None:
        dim = int(allow_unbacked_sharding)
        allow_unbacked_sharding = None
    if allow_unbacked_sharding not in (None, True, False):
        raise ValueError("allow_unbacked_sharding must be None, True, or False")
    shard_count: dict[int, int] = {}
    for mesh_dim, placement in enumerate(spec.placements):
        if not _is_shard_like(placement):
            continue
        tensor_dim = int(placement.dim)
        if tensor_dim < 0:
            tensor_dim += len(shape)
        if tensor_dim < 0 or tensor_dim >= len(shape):
            return False
        if dim is not None and tensor_dim != dim:
            continue
        count = shard_count.get(tensor_dim, 1) * int(spec.mesh.size(mesh_dim))
        shard_count[tensor_dim] = count
        required = count
        if isinstance(placement, _StridedShard):
            required *= int(placement.split_factor)
        try:
            size = int(shape[tensor_dim])
        except (TypeError, ValueError):
            if allow_unbacked_sharding is True:
                continue
            return allow_unbacked_sharding is None
        if size < required:
            return False
    return True


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    return all(is_tensor_evenly_shardable_on_dim(shape, spec, dim) for dim in range(len(shape)))


def is_tensor_evenly_shardable_on_dim(shape: Sequence[int], spec: DTensorSpec, dim: int) -> bool:
    dim = normalize_dim(dim, len(shape))
    factor = 1
    for index, placement in enumerate(spec.placements):
        if _is_shard_like(placement) and placement.dim == dim:
            factor *= int(spec.mesh.size(index))
            if isinstance(placement, _StridedShard):
                factor *= int(placement.split_factor)
    return factor == 0 or int(shape[dim]) % factor == 0


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    return any(_is_shard_like(placement) and placement.dim == dim for placement in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    return any(isinstance(placement, Partial) for placement in spec.placements)


def infer_broadcast_dims_map(
    common_shape: Sequence[int], input_shape: Sequence[int]
) -> tuple[int, ...]:
    if len(input_shape) > len(common_shape):
        raise ValueError("input rank cannot exceed broadcast rank")
    result = [-1] * len(common_shape)
    for index in range(-1, -1 - len(input_shape), -1):
        source = int(input_shape[index])
        target = int(common_shape[index])
        if source not in (1, target):
            raise ValueError("shapes are not broadcastable")
        if source == target:
            result[len(common_shape) + index] = len(input_shape) + index
    return tuple(result)


def map_placements_after_broadcast(
    placements: Sequence[Any],
    shape: Sequence[int],
    broadcast_dims_map: Sequence[int],
) -> tuple[Any, ...]:
    result = []
    for placement in placements:
        if _is_shard_like(placement):
            shard_dim = normalize_dim(placement.dim, len(shape))
            input_dim = broadcast_dims_map[shard_dim]
            result.append(
                type(placement)(input_dim, placement.split_factor)
                if input_dim != -1 and isinstance(placement, _StridedShard)
                else Shard(input_dim)
                if input_dim != -1
                else Replicate()
            )
        else:
            result.append(placement)
    return tuple(result)


def generate_redistribute_costs(
    current: DTensorSpec | OpStrategy, target: DTensorSpec
) -> float | list[float]:
    if isinstance(current, OpStrategy):
        return [
            float(redistribute_cost(strategy.output_spec, target))
            for strategy in current.strategies
        ]
    return float(redistribute_cost(current, target))


def _strategy_leaves(value: Any) -> list[OpStrategy]:
    if isinstance(value, OpStrategy):
        return [value]
    if isinstance(value, TupleStrategy):
        result: list[OpStrategy] = []
        for child in value.children:
            result.extend(_strategy_leaves(child))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(_strategy_leaves(child))
        return result
    if isinstance(value, dict):
        result = []
        for child in value.values():
            result.extend(_strategy_leaves(child))
        return result
    return []


def expand_to_full_mesh_op_strategy(
    mesh: Any,
    op_schema: Any,
    single_mesh_dim_strategies: list[list[Placement | Any | None]],
    *,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    input_index: int = 1,
    inplace_op: bool = False,
    allow_unbacked_sharding: bool | None = None,
    allow_uneven_sharding: bool = False,
    full_mesh_strategy_filter: Callable[
        [list[DTensorSpec], DTensorSpec | tuple[DTensorSpec | None, ...]], bool
    ]
    | None = None,
    different_mesh_args: list[int] | None = None,
) -> OpStrategy:
    """Expand placement rules for one mesh dimension across the full mesh."""
    from .single_dim_strategy import _ShardingPlaceholder

    if not single_mesh_dim_strategies:
        raise ValueError("at least one single-dimension strategy is required")
    mesh_ndim_value = getattr(mesh, "ndim")
    mesh_ndim = int(mesh_ndim_value() if callable(mesh_ndim_value) else mesh_ndim_value)
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh_ndim
    args_strategy = _strategy_leaves(getattr(op_schema, "args_schema", ()))
    kwargs_strategy = _strategy_leaves(
        getattr(op_schema, "kwargs_schema", {})
    )
    input_args_strategy = args_strategy + kwargs_strategy
    if not input_args_strategy:
        raise ValueError("a strategy expansion requires at least one tensor input")

    blocking_placements: tuple[Placement, ...] | None = None
    strategies: list[OpSpec] = []
    for strategy_combination in itertools.product(*all_mesh_dim_strategies):
        spec_list: list[DTensorSpec | None] = []
        output_meta_index = 0
        input_meta_index = 0
        for position, per_dim in enumerate(zip(*strategy_combination)):
            if any(value is None for value in per_dim):
                if any(value is not None for value in per_dim):
                    raise ValueError("a strategy position cannot mix missing layouts")
                spec_list.append(None)
                continue
            if not all(isinstance(value, Placement) for value in per_dim):
                raise TypeError(
                    f"unresolved placement in expanded strategy: {per_dim!r}"
                )
            tensor_meta: TensorMeta | None = None
            if position < input_index:
                if isinstance(output_tensor_meta, TensorMeta):
                    tensor_meta = output_tensor_meta
                elif isinstance(output_tensor_meta, (tuple, list)) and output_meta_index < len(output_tensor_meta):
                    tensor_meta = output_tensor_meta[output_meta_index]
                output_meta_index += 1
            elif input_meta_index < len(input_args_strategy):
                tensor_meta = input_args_strategy[input_meta_index].tensor_meta
                input_meta_index += 1
            spec_list.append(DTensorSpec(mesh, tuple(per_dim), tensor_meta=tensor_meta))

        for spec in spec_list:
            if spec is None:
                continue
            partial_kinds = {
                (type(value), getattr(value, "reduce_op", None))
                for value in spec.placements
                if isinstance(value, Partial)
            }
            if len(partial_kinds) > 1:
                reduce_ops = {value[1] for value in partial_kinds}
                partial_types = {value[0] for value in partial_kinds}
                if not (len(partial_types) == 1 and reduce_ops == {"sum", "avg"}):
                    break
        else:
            input_specs = [
                value
                for value in spec_list[input_index:]
                if isinstance(value, DTensorSpec)
            ]
            if len(input_specs) != len(input_args_strategy):
                raise ValueError(
                    f"expanded inputs {len(input_specs)} do not match strategies {len(input_args_strategy)}"
                )
            if different_mesh_args is not None:
                for index in different_mesh_args:
                    if index >= len(input_args_strategy):
                        continue
                    original = input_args_strategy[index].strategies[0].output_spec
                    if original.mesh != mesh:
                        if not all(isinstance(value, Replicate) for value in original.placements):
                            raise ValueError("a cross-mesh input must be replicated")
                        input_specs[index] = DTensorSpec(
                            original.mesh,
                            original.placements,
                            tensor_meta=original.tensor_meta,
                        )
            self_spec = input_args_strategy[0].strategies[0].output_spec
            if inplace_op and (
                self_spec.placements != input_specs[0].placements
                or (
                    spec_list and spec_list[0] is not None
                    and spec_list[0].placements != self_spec.placements
                )
            ):
                if blocking_placements is None:
                    blocking_placements = self_spec.placements
                continue

            if input_index == 0:
                output_specs: Any = None
            elif input_index > 1:
                output_specs = tuple(spec_list[:input_index])
            elif spec_list[0] is None:
                continue
            else:
                output_specs = spec_list[0]
            if isinstance(output_tensor_meta, Sequence) and not isinstance(output_tensor_meta, TensorMeta):
                if getattr(op_schema, "return_type_list_tensor_like", lambda: False)():
                    output_specs = tuple(spec_list[:input_index])

            if not all(
                is_tensor_shardable(
                    spec.shape,
                    target,
                    allow_unbacked_sharding=allow_unbacked_sharding,
                )
                or (
                    allow_uneven_sharding
                    and source.strategies[0].output_spec.placements == target.placements
                )
                for source, target, spec in zip(input_args_strategy, input_specs, input_specs)
            ):
                continue
            if full_mesh_strategy_filter is not None and output_specs is not None:
                if not full_mesh_strategy_filter(input_specs, output_specs):
                    continue
            costs = [
                list(generate_redistribute_costs(source, target))
                for source, target in zip(input_args_strategy, input_specs)
            ]
            strategies.append(
                OpSpec(
                    output_specs=output_specs,
                    input_specs=input_specs,
                    redistribute_cost=costs,
                )
            )

    if not strategies and blocking_placements is not None:
        raise RuntimeError(
            f"{getattr(op_schema, 'op', op_schema)}: in-place placement change is unsupported for {blocking_placements}"
        )
    if not strategies:
        raise RuntimeError(
            f"{getattr(op_schema, 'op', op_schema)}: no valid expanded placement strategy"
        )
    return OpStrategy(strategies)
