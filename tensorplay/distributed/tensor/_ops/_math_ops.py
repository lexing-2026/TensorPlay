"""Placement rules for reductions and normalization operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OpStrategy, PlacementStrategy
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)
from .single_dim_strategy import _ShardingPlaceholder
from .utils import (
    _is_tensor_like,
    _operation_name,
    as_list,
    generate_redistribute_costs,
    is_tensor_evenly_shardable_on_dim,
    normalize_dim,
    normalize_dims,
)

__all__ = [
    "Reduction",
    "NormReduction",
    "_NormPartial",
    "common_reduction_strategy",
    "get_placement_from_reduction_op",
    "map_placements_after_reduction",
    "replicate_reduction_dims",
    "_infer_reduction_dims",
    "_infer_reduce_dims_map",
    "_replicate_dims_start_at",
    "_skip_dim",
    "_reduction_single_dim_strategy",
    "linear_reduction_single_dim_strategy",
    "mean_single_dim_strategy",
    "bool_reduction_single_dim_strategy",
    "max_min_dim_single_dim_strategy",
    "argmax_argmin_single_dim_strategy",
    "dim_reduction_with_indices_strategy",
    "kthvalue_strategy",
    "cummax_cummin_single_dim_strategy",
    "std_var_single_dim_strategy",
    "vector_norm_single_dim_strategy",
    "powsum_single_dim_strategy",
    "pooling_single_dim_strategy",
    "softmax_single_dim_strategy",
    "softmax_backward_single_dim_strategy",
    "topk_single_dim_strategy",
    "sort_default_single_dim_strategy",
    "sort_stable_single_dim_strategy",
    "histc_single_dim_strategy",
    "logsumexp_single_dim_strategy",
    "linalg_batch_dim_strategy",
    "linalg_pinv_strategy",
    "linalg_cross_strategy",
    "interp_upsample_1out_1in_strategy",
    "interp_pool_1out_2in_strategy",
    "pool_backward_strategy",
    "grid_sampler_strategy",
    "grid_sampler_backward_strategy",
    "batch_norm_strategy",
    "batch_norm_backward_strategy",
    "group_norm_strategy",
]


class Reduction(str, Enum):
    NONE = "none"
    SUM = "sum"
    PROD = "product"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


@dataclass(frozen=True)
class NormReduction:
    """Describe a norm whose local reduction needs a power transform."""

    norm_type: int | float


@dataclass(frozen=True)
class _NormPartial(Partial):
    """A pending p-norm reduction."""

    norm_type: int | float = 2

    def __init__(self, norm_type: int | float = 2) -> None:
        object.__setattr__(self, "reduce_op", "sum")
        object.__setattr__(self, "norm_type", norm_type)

    def __hash__(self) -> int:
        return 1 + hash(self.norm_type)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NormPartial) and self.norm_type == other.norm_type

    def __repr__(self) -> str:
        return f"_NormPartial({self.norm_type})"

    def __str__(self) -> str:
        return f"_NormP({self.norm_type})"

    def _partition_value(self, value: Any, mesh: Any, mesh_dim: int) -> Any:
        return value / (int(mesh.size(mesh_dim)) ** (1.0 / self.norm_type))

    def _pre_reduce_transform(self, value: Any) -> Any:
        return value ** self.norm_type

    def _post_reduce_transform(self, value: Any) -> Any:
        return value ** (1.0 / self.norm_type)


ReductionOpType = NormReduction | Reduction | str | None


def _reduction_value(reduction_op: ReductionOpType) -> str | NormReduction | None:
    if isinstance(reduction_op, NormReduction):
        return reduction_op
    value = getattr(reduction_op, "value", reduction_op)
    if value is None:
        return None
    value = str(value)
    return {
        "mean": "avg",
        "prod": "product",
        "amin": "min",
        "amax": "max",
        "all": "product",
        "any": "sum",
    }.get(value, value)


def _infer_reduction_dims(dims_arg: object, ndim: int) -> list[int] | None:
    if dims_arg is None:
        return None
    dims = as_list(dims_arg)
    if ndim == 0 and dims in ([], [0], [-1]):
        return None
    return list(normalize_dims(dims, ndim))


def _infer_reduce_dims_map(
    reduction_dims: Sequence[int], input_ndim: int, keep_dim: bool = False
) -> list[int]:
    reduction_set = set(reduction_dims)
    result: list[int] = []
    next_dim = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_set and not keep_dim:
            result.append(-1)
        else:
            result.append(next_dim)
            next_dim += 1
    return result


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> tuple[Placement, ...]:
    return tuple(
        Replicate()
        if placement.is_partial()
        or (_is_shard_like(placement) and placement.dim >= start_dim)
        else placement
        for placement in placements
    )


def _skip_dim(
    placements: tuple[Placement, ...], skipped_dim: int
) -> tuple[Placement, ...]:
    result: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard) and placement.dim >= skipped_dim:
            result.append(
                _StridedShard(placement.dim - 1, split_factor=placement.split_factor)
            )
        elif isinstance(placement, Shard) and placement.dim >= skipped_dim:
            result.append(Shard(placement.dim - 1))
        else:
            result.append(placement)
    return tuple(result)


def replicate_reduction_dims(
    placements: Sequence[Placement], reduction_dims: Sequence[int]
) -> tuple[Placement, ...]:
    reduced = set(reduction_dims)
    result: list[Placement] = []
    for placement in placements:
        if placement.is_partial() or (
            _is_shard_like(placement) and placement.dim in reduced
        ):
            result.append(Replicate())
        else:
            result.append(placement)
    return tuple(result)


def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement:
    operation = _reduction_value(reduction_op)
    if operation is None or operation == "none":
        return Replicate()
    if isinstance(operation, NormReduction):
        if operation.norm_type == 0:
            return Partial("sum")
        return _NormPartial(operation.norm_type)
    if operation not in Partial.ALL_REDUCE_OPS:
        raise ValueError(f"unsupported reduction operation {operation!r}")
    return Partial(operation)


def _output_tensor_meta(
    spec: DTensorSpec, reduction_dims: Sequence[int], keep_dim: bool
) -> TensorMeta | None:
    if spec.tensor_meta is None:
        return None
    reduced = set(reduction_dims)
    if keep_dim:
        shape = tuple(
            1 if index in reduced else size
            for index, size in enumerate(spec.shape)
        )
    else:
        shape = tuple(
            size for index, size in enumerate(spec.shape) if index not in reduced
        )
    stride = [1] * len(shape)
    running = 1
    for index in reversed(range(len(shape))):
        stride[index] = running
        running *= int(shape[index])
    return type(spec.tensor_meta)(shape, tuple(stride), spec.tensor_meta.dtype)


def _is_evenly_sharded_on_dim(spec: DTensorSpec, dim: int) -> bool:
    if spec.tensor_meta is None:
        return True
    factor = 1
    for mesh_dim, placement in enumerate(spec.placements):
        if _is_shard_like(placement) and placement.dim == dim:
            factor *= int(spec.mesh.size(mesh_dim))
            if isinstance(placement, _StridedShard):
                factor *= placement.split_factor
    return int(spec.shape[dim]) % factor == 0


def map_placements_after_reduction(
    placements_or_spec: DTensorSpec | Sequence[Placement],
    reduction_dims: int | Sequence[int] | None,
    reduction_dims_map_or_keep_dim: Sequence[int] | bool | None = None,
    reduction_op: ReductionOpType = Reduction.SUM,
    *,
    keepdim: bool | None = None,
) -> DTensorSpec | tuple[Placement, ...]:
    """Map layouts through a dimension-reducing operation.

    The specification form also updates tensor metadata. The placement-only form
    accepts an explicit input-to-output dimension map for strategy generation.
    """
    is_spec = isinstance(placements_or_spec, DTensorSpec)
    spec = placements_or_spec if is_spec else None
    placements = tuple(spec.placements if spec is not None else placements_or_spec)
    if isinstance(reduction_dims_map_or_keep_dim, bool):
        effective_keep_dim = reduction_dims_map_or_keep_dim
        explicit_map: Sequence[int] | None = None
    else:
        explicit_map = reduction_dims_map_or_keep_dim
        effective_keep_dim = bool(keepdim) if keepdim is not None else False

    if spec is not None:
        dims = _infer_reduction_dims(reduction_dims, spec.ndim)
        reduced = list(range(spec.ndim)) if dims is None else dims
        if explicit_map is None:
            explicit_map = _infer_reduce_dims_map(
                reduced, spec.ndim, effective_keep_dim
            )
        elif keepdim is None:
            effective_keep_dim = -1 not in explicit_map
    else:
        if explicit_map is None:
            if reduction_dims is None:
                raise ValueError("a dimension map is required without tensor metadata")
            dims = (
                list(reduction_dims)
                if not isinstance(reduction_dims, int)
                else [reduction_dims]
            )
            ndim = (
                max(
                    (
                        placement.dim
                        for placement in placements
                        if _is_shard_like(placement)
                    ),
                    default=-1,
                )
                + 1
            )
            explicit_map = _infer_reduce_dims_map(dims, ndim, effective_keep_dim)
        reduced = (
            list(reduction_dims or ())
            if not isinstance(reduction_dims, int)
            else [reduction_dims]
        )

    reduced_set = set(reduced)
    operation = _reduction_value(reduction_op)
    mapped: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            mapped.append(placement)
            continue
        if not _is_shard_like(placement):
            raise TypeError(f"unsupported placement {placement!r}")
        if placement.dim < 0 or placement.dim >= len(explicit_map):
            raise ValueError(f"shard dimension {placement.dim} is outside the input rank")
        output_dim = explicit_map[placement.dim]
        if placement.dim in reduced_set or output_dim == -1:
            if operation == "avg" and spec is not None and not _is_evenly_sharded_on_dim(spec, placement.dim):
                mapped.append(Replicate())
            else:
                mapped.append(get_placement_from_reduction_op(operation))
        elif isinstance(placement, _StridedShard):
            mapped.append(
                _StridedShard(output_dim, split_factor=placement.split_factor)
            )
        else:
            mapped.append(Shard(output_dim))

    result = tuple(mapped)
    if spec is None:
        return result
    meta = _output_tensor_meta(spec, reduced, effective_keep_dim)
    return DTensorSpec(spec.mesh, result, meta, shard_order=spec.shard_order)


def _spec_from_value(value: Any) -> DTensorSpec | None:
    if isinstance(value, DTensorSpec):
        return value
    if isinstance(value, DTensor):
        stride = value.stride() if callable(value.stride) else value.stride
        return DTensorSpec(
            value.device_mesh,
            value.placements,
            TensorMeta(tuple(value.shape), tuple(stride), value.dtype),
        )
    if isinstance(value, OpStrategy):
        if not value.strategies:
            return None
        return value.strategies[0].output_spec
    return None


def _find_spec(value: Any) -> DTensorSpec | None:
    spec = _spec_from_value(value)
    if spec is not None:
        return spec
    if isinstance(value, dict):
        for child in value.values():
            spec = _find_spec(child)
            if spec is not None:
                return spec
    if isinstance(value, (tuple, list)):
        for child in value:
            spec = _find_spec(child)
            if spec is not None:
                return spec
    return None


def _common_reduction_strategy_from_op_strategy(
    input_strategy: OpStrategy,
    reduce_dims: int | Sequence[int] | None,
    keep_dim: bool,
    reduction_linear: bool,
    reduction_op: ReductionOpType,
) -> OpStrategy:
    result = OpStrategy([])
    if not input_strategy.strategies:
        return result
    first = input_strategy.strategies[0].output_spec
    dims = _infer_reduction_dims(reduce_dims, first.ndim)
    reduced = list(range(first.ndim)) if dims is None else dims
    operation = _reduction_value(reduction_op)
    for candidate in input_strategy.strategies:
        output_spec = candidate.output_spec
        linear = reduction_linear
        if operation == "avg" and output_spec.tensor_meta is not None:
            linear = all(
                is_tensor_evenly_shardable_on_dim(output_spec.shape, output_spec, dim)
                for dim in reduced
            )
        if linear and isinstance(operation, str):
            for placement in output_spec.placements:
                if isinstance(placement, Partial) and placement.reduce_op != operation:
                    linear = False
                    break
        input_placements = (
            output_spec.placements
            if linear
            else replicate_reduction_dims(output_spec.placements, reduced)
        )
        input_spec = DTensorSpec(
            output_spec.mesh,
            input_placements,
            output_spec.tensor_meta,
            shard_order=output_spec.shard_order,
        )
        mapped = map_placements_after_reduction(
            input_spec, reduced, keep_dim, operation
        )
        assert isinstance(mapped, DTensorSpec)
        cost = generate_redistribute_costs(output_spec, input_spec)
        result.strategies.append(
            PlacementStrategy(
                output_specs=mapped,
                input_specs=(input_spec,),
                redistribute_cost=[[float(cost)]],
            )
        )
    return result


def common_reduction_strategy(
    input_or_schema: Any,
    reduce_dims: int | Sequence[int] | None = None,
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = Reduction.SUM,
    *,
    keepdim: bool | None = None,
) -> Any:
    """Build either a full strategy collection or one propagated output spec."""
    if keepdim is not None:
        keep_dim = bool(keepdim)
    if isinstance(input_or_schema, OpStrategy):
        return _common_reduction_strategy_from_op_strategy(
            input_or_schema,
            reduce_dims,
            keep_dim,
            reduction_linear,
            reduction_op,
        )
    value = _find_spec(getattr(input_or_schema, "args", input_or_schema))
    if value is None:
        return None
    if reduce_dims is None and hasattr(input_or_schema, "kwargs"):
        reduce_dims = input_or_schema.kwargs.get("dim")
        keep_dim = bool(input_or_schema.kwargs.get("keepdim", keep_dim))
    return map_placements_after_reduction(value, reduce_dims, keep_dim, reduction_op)


def _argument_meta(value: Any) -> TensorMeta | None:
    if isinstance(value, TensorMeta):
        return value
    spec = _spec_from_value(value)
    return None if spec is None else spec.tensor_meta


def _first_meta(args_schema: Sequence[Any]) -> TensorMeta:
    meta = _argument_meta(args_schema[0]) if args_schema else None
    if meta is None:
        raise AssertionError("the first operation argument must carry tensor metadata")
    return meta


def _schema_dim(
    args_schema: Sequence[Any],
    kwargs_schema: dict[str, Any],
    index: int,
    default: Any = None,
) -> Any:
    return kwargs_schema.get(
        "dim", args_schema[index] if len(args_schema) > index else default
    )


def _schema_keepdim(
    args_schema: Sequence[Any], kwargs_schema: dict[str, Any], index: int = 2
) -> bool:
    return bool(
        kwargs_schema.get(
            "keepdim", args_schema[index] if len(args_schema) > index else False
        )
    )


def _reduction_single_dim_strategy(
    args_schema: Sequence[Any],
    reduction_dims: Sequence[int] | None,
    keep_dim: bool,
    reduction_linear: bool,
    reduction_op: ReductionOpType,
    extra_partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = len(_first_meta(args_schema).shape)
    reduced = set(range(ndim) if reduction_dims is None else reduction_dims)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim in reduced:
            if reduction_linear and _reduction_value(reduction_op) != "avg":
                strategies.append(
                    [
                        get_placement_from_reduction_op(reduction_op),
                        _ShardingPlaceholder(dim),
                    ]
                )
        else:
            output_dim = dim if keep_dim else dim - sum(old < dim for old in reduced)
            strategies.append(
                [_ShardingPlaceholder(output_dim), _ShardingPlaceholder(dim)]
            )
    if reduction_linear and not isinstance(reduction_op, NormReduction):
        partial = get_placement_from_reduction_op(reduction_op)
        if isinstance(partial, Partial):
            strategies.append([partial, partial])
    if extra_partial_rules:
        strategies.extend(extra_partial_rules)
    return strategies


def linear_reduction_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    name = _operation_name(operation)
    reduction_op = {
        "all": "product",
        "any": "sum",
        "sum": "sum",
        "prod": "product",
        "max": "max",
        "min": "min",
        "amax": "max",
        "amin": "min",
        "nansum": "sum",
    }.get(name)
    if reduction_op is None:
        raise KeyError(f"unsupported linear reduction {operation!r}")
    meta = _first_meta(args_schema)
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), len(meta.shape)
    )
    return _reduction_single_dim_strategy(
        args_schema,
        dims,
        _schema_keepdim(args_schema, kwargs_schema),
        True,
        reduction_op,
    )


def mean_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), len(meta.shape)
    )
    reduced = set(range(len(meta.shape)) if dims is None else dims)
    keep_dim = _schema_keepdim(args_schema, kwargs_schema)
    result: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(meta.shape)):
        if dim in reduced:
            result.append([Partial("avg"), _ShardingPlaceholder(dim)])
        else:
            output_dim = dim if keep_dim else dim - sum(old < dim for old in reduced)
            result.append(
                [_ShardingPlaceholder(output_dim), _ShardingPlaceholder(dim)]
            )
    result.append([Partial("avg"), Partial("avg")])
    return result


def bool_reduction_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), len(meta.shape)
    )
    return _reduction_single_dim_strategy(
        args_schema, dims, _schema_keepdim(args_schema, kwargs_schema), False, "sum"
    )


def _shard_non_reduction_dim(
    args_schema: Sequence[Any], dim: int, keep_dim: bool, n_outputs: int
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = len(_first_meta(args_schema).shape)
    dim = normalize_dim(dim, ndim)
    result: list[list[Placement | _ShardingPlaceholder]] = []
    for input_dim in range(ndim):
        if input_dim == dim:
            continue
        output_dim = input_dim if keep_dim or input_dim < dim else input_dim - 1
        result.append(
            [_ShardingPlaceholder(output_dim)] * n_outputs
            + [_ShardingPlaceholder(input_dim)]
        )
    return result


def max_min_dim_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = int(_schema_dim(args_schema, kwargs_schema, 1, -1))
    return _shard_non_reduction_dim(
        args_schema, dim, _schema_keepdim(args_schema, kwargs_schema), 2
    )


def _argmax_argmin_reduction_dims(
    args_schema: Sequence[Any], kwargs_schema: dict[str, Any], ndim: int
) -> list[int]:
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), ndim
    )
    return list(range(ndim)) if dims is None else dims


def argmax_argmin_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = len(_first_meta(args_schema).shape)
    reduced = set(_argmax_argmin_reduction_dims(args_schema, kwargs_schema, ndim))
    keep_dim = _schema_keepdim(args_schema, kwargs_schema)
    if len(reduced) == ndim:
        return []
    result: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim in reduced:
            continue
        output_dim = dim if keep_dim else dim - sum(old < dim for old in reduced)
        result.append([_ShardingPlaceholder(output_dim), _ShardingPlaceholder(dim)])
    return result


def dim_reduction_with_indices_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = int(_schema_dim(args_schema, kwargs_schema, 1, -1))
    return _shard_non_reduction_dim(
        args_schema, dim, _schema_keepdim(args_schema, kwargs_schema), 2
    )


def kthvalue_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = int(_schema_dim(args_schema, kwargs_schema, 2, -1))
    return _shard_non_reduction_dim(
        args_schema, dim, _schema_keepdim(args_schema, kwargs_schema, 3), 2
    )


def _shard_except_dim_strategy(
    args_schema: Sequence[Any], active_dim: int, n_placements: int
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = len(_first_meta(args_schema).shape)
    active_dim = normalize_dim(active_dim, ndim)
    return [
        [_ShardingPlaceholder(dim)] * n_placements
        for dim in range(ndim)
        if dim != active_dim
    ]


def cummax_cummin_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _shard_except_dim_strategy(
        args_schema, int(_schema_dim(args_schema, kwargs_schema, 1, -1)), 3
    )


def std_var_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), len(meta.shape)
    )
    return _reduction_single_dim_strategy(
        args_schema, dims, _schema_keepdim(args_schema, kwargs_schema), False, "sum"
    )


def _get_norm_reduction_op(norm_type: int | float | str) -> ReductionOpType:
    if norm_type in (float("inf"), "inf"):
        return "max"
    if norm_type in (float("-inf"), "-inf"):
        return "min"
    if not isinstance(norm_type, (int, float)):
        raise TypeError("norm order must be numeric or an infinity marker")
    return NormReduction(norm_type)


def vector_norm_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    norm_type = args_schema[1] if len(args_schema) > 1 else kwargs_schema.get("ord", 2)
    dim = args_schema[2] if len(args_schema) > 2 else kwargs_schema.get("dim")
    keep_dim = args_schema[3] if len(args_schema) > 3 else kwargs_schema.get("keepdim", False)
    dims = _infer_reduction_dims(dim, len(meta.shape))
    return _reduction_single_dim_strategy(
        args_schema,
        dims,
        bool(keep_dim),
        True,
        _get_norm_reduction_op(norm_type),
    )


def powsum_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    dim = args_schema[2] if len(args_schema) > 2 else kwargs_schema.get("dim")
    keep_dim = args_schema[3] if len(args_schema) > 3 else kwargs_schema.get("keepdim", False)
    dims = _infer_reduction_dims(dim, len(meta.shape))
    return _reduction_single_dim_strategy(args_schema, dims, bool(keep_dim), True, "sum")


def pooling_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = len(_first_meta(args_schema).shape)
    name = _operation_name(operation)
    result = [[_ShardingPlaceholder(0)] * 2]
    if name.startswith("avg_pool") or name.startswith("adaptive_avg_pool"):
        result.extend([[Partial("sum")] * 2, [Partial("avg")] * 2])
    if ndim >= 4:
        result.append([_ShardingPlaceholder(1)] * 2)
    return result


def softmax_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _shard_except_dim_strategy(
        args_schema, int(_schema_dim(args_schema, kwargs_schema, 1, -1)), 2
    )


def softmax_backward_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _shard_except_dim_strategy(
        args_schema, int(_schema_dim(args_schema, kwargs_schema, 2, -1)), 3
    )


def topk_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _shard_except_dim_strategy(
        args_schema, int(_schema_dim(args_schema, kwargs_schema, 2, -1)), 3
    )


def sort_default_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _shard_except_dim_strategy(
        args_schema, int(_schema_dim(args_schema, kwargs_schema, 1, -1)), 3
    )


def sort_stable_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return sort_default_single_dim_strategy(operation, args_schema, kwargs_schema)


def histc_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    if len(args_schema) != 4:
        return []
    ndim = len(_first_meta(args_schema).shape)
    return [[Partial("sum"), _ShardingPlaceholder(dim)] for dim in range(ndim)]


def logsumexp_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    meta = _first_meta(args_schema)
    dims = _infer_reduction_dims(
        _schema_dim(args_schema, kwargs_schema, 1), len(meta.shape)
    )
    return _reduction_single_dim_strategy(
        args_schema, dims, _schema_keepdim(args_schema, kwargs_schema), False, "sum"
    )


def _linalg_batch_dim_strategies(
    ndim: int, n_placements: int
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [
        [_ShardingPlaceholder(dim)] * n_placements
        for dim in range(max(0, ndim - 2))
    ]


def _get_ndim(tensor_meta: Any) -> int:
    meta = _argument_meta(tensor_meta)
    if meta is None:
        raise AssertionError("a tensor metadata value is required")
    return len(meta.shape)


def linalg_batch_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return _linalg_batch_dim_strategies(
        _get_ndim(args_schema[0]), 1 + sum(_argument_meta(arg) is not None for arg in args_schema)
    )


def linalg_pinv_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    extra = sum(
        _argument_meta(kwargs_schema.get(name)) is not None
        for name in ("atol", "rtol")
    )
    return [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        + [Replicate()] * extra
        for dim in range(max(0, ndim - 2))
    ]


def linalg_cross_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    cross_dim = int(kwargs_schema.get("dim", -1)) % ndim
    return [
        [_ShardingPlaceholder(dim)] * 3
        for dim in range(ndim)
        if dim != cross_dim
    ]


def interp_upsample_1out_1in_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [
        [_ShardingPlaceholder(0)] * 2,
        [_ShardingPlaceholder(1)] * 2,
        [Partial("sum")] * 2,
        [Partial("avg")] * 2,
    ]


def interp_pool_1out_2in_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [[_ShardingPlaceholder(0)] * 3, [_ShardingPlaceholder(1)] * 3]


def pool_backward_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    result = [[_ShardingPlaceholder(0)] * 4]
    if _get_ndim(args_schema[0]) >= 4:
        result.append([_ShardingPlaceholder(1)] * 4)
    result.extend(
        [
            [Partial("sum"), Partial("sum"), Replicate(), Replicate()],
            [Partial("avg"), Partial("avg"), Replicate(), Replicate()],
        ]
    )
    return result


def grid_sampler_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [
        [_ShardingPlaceholder(0)] * 3,
        [Partial("sum"), Partial("sum"), Replicate()],
        [Partial("avg"), Partial("avg"), Replicate()],
    ]


def grid_sampler_backward_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    return [[_ShardingPlaceholder(0)] * 5]


def batch_norm_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    tensor_inputs = sum(_argument_meta(value) is not None for value in args_schema)
    return [
        [_ShardingPlaceholder(1), _ShardingPlaceholder(0), _ShardingPlaceholder(0), _ShardingPlaceholder(1)]
        + [_ShardingPlaceholder(0)] * max(0, tensor_inputs - 1)
    ]


def batch_norm_backward_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    tensor_inputs = sum(_argument_meta(value) is not None for value in args_schema)
    return [
        [_ShardingPlaceholder(1), _ShardingPlaceholder(0), _ShardingPlaceholder(0), _ShardingPlaceholder(1)]
        + [_ShardingPlaceholder(0)] * max(0, tensor_inputs - 2)
    ]


def group_norm_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    tensor_inputs = sum(_argument_meta(value) is not None for value in args_schema)
    return [
        [_ShardingPlaceholder(0)] * 4 + [Replicate()] * max(0, tensor_inputs - 1)
    ]
