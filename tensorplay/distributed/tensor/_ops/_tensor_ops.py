"""Placement rules for tensor indexing, shape creation, and layout-preserving ops."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Sized
from typing import Any

from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from ..placement_types import (
    _MaskPartial,
    _StridedShard,
    _is_shard_like,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from .single_dim_strategy import _ShardingPlaceholder
from .utils import (
    generate_redistribute_costs,
    is_tensor_dim_sharded,
    normalize_dim,
    shift_shard_dims_after_insert,
    shift_shard_dims_after_remove,
)

__all__ = [
    "_derive_follow_placements_from_tuple_strategy",
    "_index_dim_strategy",
    "_partial_needs_reduce_for_dtype_cast",
    "_pass_through_partials",
    "_shard_inactive_dims",
    "_to_copy_single_dim_strategy",
    "bucketize_single_dim_strategy",
    "cat_single_dim_strategy",
    "create_like_single_dim_strategy",
    "diagonal_scatter_single_dim_strategy",
    "equal_single_dim_strategy",
    "eye_out_single_dim_strategy",
    "fft_single_dim_strategy",
    "flip_single_dim_strategy",
    "gather_single_dim_strategy",
    "gen_unbind_strategy",
    "index_fill_scalar_single_dim_strategy",
    "index_fill_tensor_single_dim_strategy",
    "index_put_single_dim_strategy",
    "index_reduce_single_dim_strategy",
    "index_select_single_dim_strategy",
    "index_single_dim_strategy",
    "local_scalar_dense_single_dim_strategy",
    "new_factory_single_dim_strategy",
    "propagate_single_input_single_dim_strategy",
    "roll_single_dim_strategy",
    "scatter_add_single_dim_strategy",
    "scatter_single_dim_strategy",
    "select_backward_single_dim_strategy",
    "select_int_single_dim_strategy",
    "select_scatter_single_dim_strategy",
    "slice_scatter_single_dim_strategy",
    "slice_single_dim_strategy",
    "split_single_dim_strategy",
    "stack_strategy",
    "register_tensor_ops",
]

_PARTIAL_PASS_THROUGH_REDUCE_OPS = ("sum", "avg", "min", "max")


def _meta(value: Any) -> TensorMeta:
    if not isinstance(value, TensorMeta):
        raise AssertionError(f"expected tensor metadata, got {type(value)}")
    return value


def _int_like(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _known_bool(value: Any) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _dtype_is_bool(value: Any) -> bool:
    return str(value).lower().split(".")[-1] in {"bool", "boolean"}


def _dtype_is_floating(value: Any) -> bool:
    text = str(value).lower()
    return any(name in text for name in ("float", "double", "bfloat", "half"))


def _broadcast_shape(shapes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if not shapes:
        return ()
    reversed_shapes = [tuple(reversed(tuple(shape))) for shape in shapes]
    result: list[int] = []
    for dim in range(max(len(shape) for shape in reversed_shapes)):
        values = [shape[dim] for shape in reversed_shapes if dim < len(shape)]
        values = [int(value) for value in values if int(value) != 1]
        if values and any(value != values[0] for value in values):
            raise ValueError("index shapes are not broadcastable")
        result.append(values[0] if values else 1)
    return tuple(reversed(result))


def propagate_single_input_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    strategies = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
    ]
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def _partial_needs_reduce_for_dtype_cast(
    reduce_op: str, src_dtype: Any, target_dtype: Any | None
) -> bool:
    if target_dtype is None or src_dtype == target_dtype:
        return False
    if _dtype_is_bool(target_dtype):
        return True
    if reduce_op in ("max", "min"):
        return False
    return _dtype_is_floating(src_dtype) and not _dtype_is_floating(target_dtype)


def _to_copy_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op
    input_meta = _meta(args_schema[0])
    target_dtype = kwargs_schema.get("dtype")
    strategies = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
    ]
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
        if not _partial_needs_reduce_for_dtype_cast(
            reduce_op, input_meta.dtype, target_dtype
        )
    )
    return strategies


def equal_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    left = _meta(args_schema[0])
    right = _meta(args_schema[1])
    return [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(min(len(left.shape), len(right.shape)))
    ]


def create_like_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op
    input_meta = _meta(args_schema[0])
    out_names = [
        name for name, value in kwargs_schema.items() if isinstance(value, TensorMeta)
    ]
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        strategies.append(
            [placement, placement]
            + [placement if name == "out" else Replicate() for name in out_names]
        )
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append(
            [Replicate(), Partial(reduce_op)]
            + [Replicate() for _ in out_names]
        )
    return strategies


def new_factory_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del kwargs_schema
    input_meta = _meta(args_schema[0])
    output_shape = tuple(args_schema[1])
    same_shape = input_meta.shape == output_shape
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        strategies.append(
            [placement, placement] if same_shape else [Replicate(), placement]
        )
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append(
            [
                Partial(reduce_op) if same_shape and "empty" in str(op) else Replicate(),
                Partial(reduce_op),
            ]
        )
    return strategies


def bucketize_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    strategies = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim), Replicate()]
        for dim in range(len(input_meta.shape))
    ]
    strategies.append([Partial("sum"), Replicate(), _ShardingPlaceholder(0)])
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op), Replicate()]
        for reduce_op in ("max", "min")
    )
    return strategies


def select_int_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    selected_dim = normalize_dim(int(args_schema[1]), len(input_meta.shape))
    strategies = [
        [
            _ShardingPlaceholder(dim if dim < selected_dim else dim - 1),
            _ShardingPlaceholder(dim),
        ]
        for dim in range(len(input_meta.shape))
        if dim != selected_dim
    ]
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def select_backward_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    grad_meta = _meta(args_schema[0])
    input_sizes = args_schema[1]
    dim = normalize_dim(int(args_schema[2]), len(input_sizes))
    strategies = [
        [
            _ShardingPlaceholder(grad_dim if grad_dim < dim else grad_dim + 1),
            _ShardingPlaceholder(grad_dim),
        ]
        for grad_dim in range(len(grad_meta.shape))
    ]
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def slice_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    values = list(args_schema) + [None, 0, None, None, 1]
    input_meta, dim, start, end, step = values[:5]
    input_meta = _meta(input_meta)
    if not _int_like(dim):
        raise AssertionError(f"expected integer dimension, got {type(dim)}")
    slice_dim = normalize_dim(dim, len(input_meta.shape))
    start = 0 if start is None else start
    if not _int_like(start) or (end is not None and not _int_like(end)):
        raise AssertionError("slice bounds must be integer-like")
    if not _int_like(step):
        raise AssertionError("slice step must be integer-like")
    if end is None or _known_bool(end > input_meta.shape[slice_dim]):
        end = input_meta.shape[slice_dim]
    if _known_bool(start < 0):
        start += input_meta.shape[slice_dim]
    if _known_bool(end < 0):
        end += input_meta.shape[slice_dim]
    full_slice = _known_bool(
        start == 0 and end == input_meta.shape[slice_dim] and step == 1
    )
    strategies = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
        if dim != slice_dim or full_slice
    ]
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def slice_scatter_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    _meta(args_schema[1])
    dim = int(args_schema[2]) if len(args_schema) > 2 else 0
    slice_dim = normalize_dim(dim, len(input_meta.shape))
    strategies = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
        if dim != slice_dim
    ]
    strategies.extend(
        [Partial(reduce_op)] * 3 for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def select_scatter_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    dim = normalize_dim(int(args_schema[2]), len(input_meta.shape))
    return [
        [
            _ShardingPlaceholder(value),
            _ShardingPlaceholder(value),
            _ShardingPlaceholder(value if value < dim else value - 1),
        ]
        for value in range(len(input_meta.shape))
        if value != dim
    ]


def diagonal_scatter_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    ndim = len(input_meta.shape)
    dim1 = normalize_dim(int(args_schema[3]) if len(args_schema) > 3 else 0, ndim)
    dim2 = normalize_dim(int(args_schema[4]) if len(args_schema) > 4 else 1, ndim)
    low, high = min(dim1, dim2), max(dim1, dim2)
    return [
        [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim - (dim > low) - (dim > high)),
        ]
        for dim in range(ndim)
        if dim not in (dim1, dim2)
    ]


def local_scalar_dense_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, args_schema, kwargs_schema
    return []


def scatter_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta, dim, index_meta = args_schema[:3]
    input_meta = _meta(input_meta)
    index_meta = _meta(index_meta)
    scatter_dim = normalize_dim(int(dim), len(input_meta.shape))
    src_meta = args_schema[3] if len(args_schema) > 3 else None
    src_shape = src_meta.shape if isinstance(src_meta, TensorMeta) else None
    count = 4 if src_shape is not None else 3
    if len(input_meta.shape) != len(index_meta.shape):
        return []
    strategies = []
    for dim in range(len(input_meta.shape)):
        if dim == scatter_dim or input_meta.shape[dim] != index_meta.shape[dim]:
            continue
        if src_shape is not None and (
            len(src_shape) != len(index_meta.shape) or src_shape[dim] != index_meta.shape[dim]
        ):
            continue
        placement = _ShardingPlaceholder(dim)
        strategies.append([placement] * count)
    return strategies


def scatter_add_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta, dim, index_meta = args_schema[:3]
    input_meta = _meta(input_meta)
    index_meta = _meta(index_meta)
    scatter_dim = normalize_dim(int(dim), len(input_meta.shape))
    if len(input_meta.shape) != len(index_meta.shape):
        return []
    strategies = []
    for dim in range(len(input_meta.shape)):
        if dim != scatter_dim and input_meta.shape[dim] == index_meta.shape[dim]:
            placement = _ShardingPlaceholder(dim)
            strategies.append([placement, placement, placement, placement])
    return strategies


def gather_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta, dim, index_meta = args_schema[:3]
    input_meta = _meta(input_meta)
    index_meta = _meta(index_meta)
    gather_dim = normalize_dim(int(dim), len(input_meta.shape))
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if gather_dim < len(index_meta.shape) and index_meta.shape[gather_dim] == 1:
        mask = _MaskPartial(offset_shape=input_meta.shape, offset_dim=gather_dim)
        strategies.append([mask, Shard(gather_dim), mask])
    if gather_dim < len(index_meta.shape):
        strategies.append([Shard(gather_dim), Replicate(), Shard(gather_dim)])
    if len(input_meta.shape) == len(index_meta.shape):
        for dim in range(len(input_meta.shape)):
            if dim != gather_dim:
                placement = _ShardingPlaceholder(dim)
                strategies.append([placement, placement, placement])
    return strategies


def _derive_follow_placements_from_tuple_strategy(
    op: Any, tuple_strategy: TupleStrategy
) -> Sequence[Placement]:
    del op

    def merge(current: Placement, new: Placement) -> Placement:
        if current == new:
            return current
        if current.is_partial():
            if _is_shard_like(new):
                return new
            if new.is_partial():
                return Replicate()
            return current
        if _is_shard_like(current):
            return Replicate() if _is_shard_like(new) else current
        return new

    follow: list[Placement] | None = None
    mesh = tuple_strategy.child_mesh(0)
    for child in tuple_strategy.children:
        if not isinstance(child, OpStrategy):
            raise AssertionError(f"expected operation strategy, got {type(child)}")
        if child.mesh != mesh:
            raise ValueError("tuple strategy inputs use different meshes")
        for strategy in child.strategies:
            placements = strategy.output_spec.placements
            if follow is None:
                follow = list(placements)
                continue
            for mesh_dim in range(int(mesh.ndim)):
                follow[mesh_dim] = merge(follow[mesh_dim], placements[mesh_dim])
    if follow is None:
        raise AssertionError("tuple strategy has no placements")
    return follow


def stack_strategy(op_schema: OpSchema) -> StrategyType:
    tuple_strategy = op_schema.args_schema[0]
    if not isinstance(tuple_strategy, TupleStrategy):
        raise AssertionError(f"expected tuple strategy, got {type(tuple_strategy)}")
    input_strategies = []
    for child in tuple_strategy.children:
        if not isinstance(child, OpStrategy):
            raise AssertionError(f"expected operation strategy, got {type(child)}")
        input_strategies.append(child)
    first = input_strategies[0]
    dim = int(op_schema.args_schema[1]) if len(op_schema.args_schema) > 1 else 0
    dim = normalize_dim(dim, first.ndim + 1)
    mesh = first.mesh
    follow = _derive_follow_placements_from_tuple_strategy(op_schema.op, tuple_strategy)
    input_specs = tuple(DTensorSpec(mesh, tuple(follow)) for _ in input_strategies)
    output_spec = DTensorSpec(mesh, tuple(shift_shard_dims_after_insert(follow, dim)))
    return OpStrategy(
        [
            OpSpec(
                output_specs=output_spec,
                input_specs=input_specs,
                redistribute_cost=[
                    generate_redistribute_costs(strategy, spec)
                    for strategy, spec in zip(input_strategies, input_specs)
                ],
            )
        ]
    )


def cat_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_list = args_schema[0]
    if not isinstance(input_list, (tuple, list)) or not input_list:
        raise AssertionError("cat requires a non-empty tensor list")
    input_list = tuple(_meta(value) for value in input_list)

    def legacy_empty(meta: TensorMeta) -> bool:
        return len(meta.shape) == 1 and _known_bool(meta.shape[0] == 0)

    empty = tuple(legacy_empty(meta) for meta in input_list)
    real_ndims = {
        len(meta.shape) for meta, is_empty in zip(input_list, empty) if not is_empty
    }
    if len(real_ndims) > 1:
        raise AssertionError("non-empty cat inputs must have equal rank")
    common_ndim = next(iter(real_ndims), 1)
    cat_dim = int(args_schema[1]) if len(args_schema) > 1 else 0
    cat_dim = normalize_dim(cat_dim, common_ndim)
    strategies = []
    for dim in range(common_ndim):
        if dim != cat_dim:
            strategies.append(
                [_ShardingPlaceholder(dim)]
                + [
                    Replicate() if is_empty else _ShardingPlaceholder(dim)
                    for is_empty in empty
                ]
            )
    strategies.extend(
        [Partial(reduce_op)]
        + [
            Replicate() if is_empty else Partial(reduce_op)
            for is_empty in empty
        ]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def index_select_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    values_meta, dim, index_meta = args_schema
    values_meta = _meta(values_meta)
    _meta(index_meta)
    dim = normalize_dim(int(dim), len(values_meta.shape))
    strategies = [
        [_ShardingPlaceholder(value), _ShardingPlaceholder(value), Replicate()]
        for value in range(len(values_meta.shape))
        if value != dim
    ]
    strategies.append([_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(0)])
    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op), Replicate()]
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def index_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    values_meta = _meta(args_schema[0])
    multi_indices_meta = args_schema[1]
    if not isinstance(multi_indices_meta, (list, tuple)):
        raise AssertionError("index metadata must be a sequence")
    indexed_dims = [
        dim for dim, index_meta in enumerate(multi_indices_meta) if index_meta is not None
    ]
    non_indexed_dims = [
        dim for dim in range(len(values_meta.shape)) if dim not in set(indexed_dims)
    ]
    index_metas = [index_meta for index_meta in multi_indices_meta if index_meta is not None]
    if not all(isinstance(index_meta, TensorMeta) for index_meta in index_metas):
        raise AssertionError("index metadata must contain tensor metadata")
    broadcast_shape = _broadcast_shape([index_meta.shape for index_meta in index_metas])
    broadcast_ndim = len(broadcast_shape)
    num_indices = len(indexed_dims)
    consecutive = all(
        indexed_dims[index + 1] - indexed_dims[index] == 1
        for index in range(len(indexed_dims) - 1)
    )
    insert_dim = indexed_dims[0] if consecutive and indexed_dims else 0

    def output_dim(input_dim: int) -> int:
        if input_dim < insert_dim:
            return input_dim
        return input_dim + broadcast_ndim - sum(
            indexed_dim < input_dim for indexed_dim in indexed_dims
        )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for input_dim in non_indexed_dims:
        row: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(output_dim(input_dim)),
            _ShardingPlaceholder(input_dim),
        ]
        row.extend([Replicate()] * num_indices)
        strategies.append(row)

    for broadcast_dim in range(broadcast_ndim):
        per_tensor: list[tuple[int, int]] = []
        for index_meta in index_metas:
            offset = broadcast_ndim - len(index_meta.shape)
            if broadcast_dim < offset:
                per_tensor.append((-1, 1))
            else:
                tensor_dim = broadcast_dim - offset
                per_tensor.append((tensor_dim, int(index_meta.shape[tensor_dim])))
        if all(size == 1 for _, size in per_tensor):
            continue
        row = [_ShardingPlaceholder(broadcast_dim + insert_dim), Replicate()]
        row.extend(
            _ShardingPlaceholder(tensor_dim) if size > 1 else Replicate()
            for tensor_dim, size in per_tensor
        )
        strategies.append(row)

    strategies.extend(
        [Partial(reduce_op), Partial(reduce_op)]
        + [Replicate()] * num_indices
        for reduce_op in Partial.LINEAR_REDUCE_OPS
    )
    return strategies


def index_put_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    self_meta = _meta(args_schema[0])
    indices_meta = args_schema[1]
    values_meta = _meta(args_schema[2])
    if not isinstance(indices_meta, (tuple, list)):
        raise AssertionError("index metadata must be a sequence")
    indexed_dims = {dim for dim, value in enumerate(indices_meta) if value is not None}
    non_indexed_dims = [
        dim for dim in range(len(self_meta.shape)) if dim not in indexed_dims
    ]
    index_shapes = [
        value.shape for value in indices_meta if isinstance(value, TensorMeta)
    ]
    broadcast_ndim = len(_broadcast_shape(index_shapes)) if index_shapes else 0
    num_indices = sum(value is not None for value in indices_meta)
    sorted_indexed = sorted(indexed_dims)
    consecutive = len(sorted_indexed) <= 1 or (
        sorted_indexed[-1] - sorted_indexed[0] + 1 == len(sorted_indexed)
    )
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for index, self_dim in enumerate(non_indexed_dims):
        if consecutive and sorted_indexed:
            first_indexed = sorted_indexed[0]
            result_dim = (
                self_dim
                if self_dim < first_indexed
                else self_dim - num_indices + broadcast_ndim
            )
        else:
            result_dim = broadcast_ndim + index
        result_ndim = broadcast_ndim + len(non_indexed_dims)
        values_dim = result_dim - (result_ndim - len(values_meta.shape))
        if values_dim < 0 or values_meta.shape[values_dim] == 1:
            values_placement: Placement | _ShardingPlaceholder = Replicate()
        else:
            values_placement = _ShardingPlaceholder(values_dim)
        strategies.append(
            [
                _ShardingPlaceholder(self_dim),
                _ShardingPlaceholder(self_dim),
                *([Replicate()] * num_indices),
                values_placement,
            ]
        )
    strategies.append(
        [
            Partial(),
            Partial(),
            *([Replicate()] * num_indices),
            Partial(),
        ]
    )
    return strategies


def _index_dim_strategy(
    args_schema: tuple[Any, ...],
    shard_row: Callable[[int], list[Placement | _ShardingPlaceholder]],
    partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = _meta(args_schema[0])
    dim = normalize_dim(int(args_schema[1]), len(self_meta.shape))
    strategies = [shard_row(value) for value in range(len(self_meta.shape)) if value != dim]
    if partial_rules:
        strategies.extend(partial_rules)
    return strategies


def index_fill_scalar_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    return _index_dim_strategy(
        args_schema,
        lambda dim: [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
            Replicate(),
        ],
        [
            [Partial(reduce_op), Partial(reduce_op), Replicate()]
            for reduce_op in ("avg", "max", "min")
        ],
    )


def index_fill_tensor_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    return _index_dim_strategy(
        args_schema,
        lambda dim: [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
            Replicate(),
            Replicate(),
        ],
        [
            [Partial(reduce_op), Partial(reduce_op), Replicate(), Partial(reduce_op)]
            for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
        ],
    )


def index_reduce_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    return _index_dim_strategy(
        args_schema,
        lambda dim: [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
            Replicate(),
            _ShardingPlaceholder(dim),
        ],
    )


def split_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    split_value = args_schema[1]
    split_dim = int(args_schema[2]) if len(args_schema) > 2 else 0
    split_dim = normalize_dim(split_dim, len(input_meta.shape))
    if _int_like(split_value):
        if split_value <= 0:
            raise AssertionError("split size must be positive")
        output_sizes = [
            split_value
        ] * (int(input_meta.shape[split_dim]) // split_value)
        remainder = int(input_meta.shape[split_dim]) % split_value
        if remainder:
            output_sizes.append(remainder)
    elif isinstance(split_value, Sized):
        output_sizes = list(split_value)
    else:
        raise AssertionError("split sections must be sized")
    num_outputs = len(output_sizes)
    strategies = [
        [_ShardingPlaceholder(dim)] * (num_outputs + 1)
        for dim in range(len(input_meta.shape))
        if dim != split_dim
    ]
    strategies.extend(
        [Partial(reduce_op)] * (num_outputs + 1)
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    )
    return strategies


def gen_unbind_strategy(op_schema: OpSchema) -> StrategyType:
    input_strategy = op_schema.args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"expected operation strategy, got {type(input_strategy)}")
    input_dim = int(op_schema.args_schema[1]) if len(op_schema.args_schema) > 1 else 0
    input_dim = normalize_dim(input_dim, input_strategy.ndim)
    mesh = input_strategy.mesh
    result = OpStrategy([])
    for candidate in input_strategy.strategies:
        input_spec = candidate.output_spec
        if is_tensor_dim_sharded(input_spec, input_dim):
            raise RuntimeError("unbind cannot remove a sharded dimension")
        output_placements = shift_shard_dims_after_remove(input_spec.placements, input_dim)
        output_specs = tuple(
            DTensorSpec(mesh, tuple(output_placements))
            for _ in range(int(input_strategy.shape[input_dim]))
        )
        result.strategies.append(
            OpSpec(
                output_specs=output_specs,
                input_specs=(input_spec,),
                redistribute_cost=[[0.0] * len(input_strategy.strategies)],
            )
        )
    return result


def eye_out_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, args_schema
    if "out" in kwargs_schema:
        _meta(kwargs_schema["out"])
    return []


def _pass_through_partials(
    num_inputs: int = 1,
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [
        [Partial(reduce_op)] * (1 + num_inputs)
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
    ]


def _shard_inactive_dims(
    ndim: int, active_dims: set[int], num_inputs: int = 1
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [
        [_ShardingPlaceholder(dim)] * (1 + num_inputs)
        for dim in range(ndim)
        if dim not in active_dims
    ]


def roll_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    raw_dims = args_schema[2] if len(args_schema) > 2 else []
    if not raw_dims:
        raw_dims = list(range(len(input_meta.shape)))
    active_dims = {normalize_dim(int(dim), len(input_meta.shape)) for dim in raw_dims}
    return _shard_inactive_dims(len(input_meta.shape), active_dims) + _pass_through_partials()


def flip_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    raw_dims = args_schema[1]
    active_dims = {normalize_dim(int(dim), len(input_meta.shape)) for dim in raw_dims}
    return _shard_inactive_dims(len(input_meta.shape), active_dims) + _pass_through_partials()


def fft_single_dim_strategy(
    op: Any, args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    del op, kwargs_schema
    input_meta = _meta(args_schema[0])
    active_dims = {normalize_dim(int(dim), len(input_meta.shape)) for dim in args_schema[1]}
    return _shard_inactive_dims(len(input_meta.shape), active_dims) + _pass_through_partials()


def _register_single_dim(
    names: Sequence[str],
    function: Callable[..., Any],
    schema_info: RuntimeSchemaInfo | None = None,
    *,
    allow_unbacked_sharding: bool | None = None,
    allow_uneven_sharding: bool = False,
) -> None:
    from .single_dim_strategy import register_single_dim_strategy

    register_single_dim_strategy(
        tuple(names),
        schema_info=schema_info,
        allow_unbacked_sharding=allow_unbacked_sharding,
        allow_uneven_sharding=allow_uneven_sharding,
    )(function)


_TENSOR_OPS_READY = False


def register_tensor_ops() -> None:
    global _TENSOR_OPS_READY
    if _TENSOR_OPS_READY:
        return
    _TENSOR_OPS_READY = True

    common = (
        "clone",
        "contiguous",
        "detach",
        "detach_",
        "alias",
        "fill_",
        "view_dtype",
        "zero_",
        "view_of",
    )
    _register_single_dim(
        common,
        propagate_single_input_single_dim_strategy,
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("_to_copy", "to_copy"),
        _to_copy_single_dim_strategy,
        RuntimeSchemaInfo(static_kwargkey=["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("equal", "is_same_size"),
        equal_single_dim_strategy,
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("empty_like",),
        create_like_single_dim_strategy,
        RuntimeSchemaInfo(1, ["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("ones_like", "rand_like", "randn_like", "zeros_like"),
        create_like_single_dim_strategy,
        RuntimeSchemaInfo(1, ["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("full_like",),
        create_like_single_dim_strategy,
        RuntimeSchemaInfo(2, ["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("randint_like", "randint_like_low_dtype"),
        create_like_single_dim_strategy,
        RuntimeSchemaInfo(3, ["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("new_empty", "new_full", "new_ones", "new_zeros", "new_empty_strided"),
        new_factory_single_dim_strategy,
        RuntimeSchemaInfo(1, ["dtype"]),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(("bucketize",), bucketize_single_dim_strategy)
    _register_single_dim(
        ("select", "select_int"),
        select_int_single_dim_strategy,
        RuntimeSchemaInfo(1),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("select_backward",),
        select_backward_single_dim_strategy,
        RuntimeSchemaInfo(1),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("slice", "slice_tensor"),
        slice_single_dim_strategy,
        RuntimeSchemaInfo(1),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("slice_scatter",),
        slice_scatter_single_dim_strategy,
        RuntimeSchemaInfo(2),
        allow_unbacked_sharding=True,
        allow_uneven_sharding=True,
    )
    _register_single_dim(
        ("select_scatter",), select_scatter_single_dim_strategy, RuntimeSchemaInfo(1)
    )
    _register_single_dim(
        ("diagonal_scatter",), diagonal_scatter_single_dim_strategy, RuntimeSchemaInfo(1)
    )
    _register_single_dim(("_local_scalar_dense", "local_scalar_dense"), local_scalar_dense_single_dim_strategy)
    _register_single_dim(("scatter", "scatter_", "scatter_value", "scatter_src"), scatter_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("scatter_add", "scatter_add_"), scatter_add_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("gather",), gather_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("cat",), cat_single_dim_strategy, RuntimeSchemaInfo(1, needs_pytree=True), allow_unbacked_sharding=False)
    _register_single_dim(("index_select",), index_select_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("index",), index_single_dim_strategy, RuntimeSchemaInfo(needs_pytree=True))
    _register_single_dim(("index_put", "index_put_", "_index_put_impl_"), index_put_single_dim_strategy, RuntimeSchemaInfo(needs_pytree=True))
    _register_single_dim(("index_fill_scalar", "index_fill_scalar_"), index_fill_scalar_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("index_fill_tensor", "index_fill_tensor_"), index_fill_tensor_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("index_reduce", "index_reduce_"), index_reduce_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("split", "split_tensor", "split_with_sizes", "split_with_sizes_copy"), split_single_dim_strategy, RuntimeSchemaInfo(1), allow_unbacked_sharding=False)
    _register_single_dim(("eye_out",), eye_out_single_dim_strategy, RuntimeSchemaInfo(static_kwargkey=["out"]), allow_unbacked_sharding=True, allow_uneven_sharding=True)
    _register_single_dim(("roll",), roll_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("flip",), flip_single_dim_strategy, RuntimeSchemaInfo(1))
    _register_single_dim(("fft_c2c", "fft_r2c", "fft_c2r"), fft_single_dim_strategy, RuntimeSchemaInfo(1))

    dispatcher = getattr(DTensor, "_op_dispatcher", None)
    if dispatcher is None:
        raise RuntimeError("distributed tensor dispatcher is not initialized")
    propagator = dispatcher.sharding_propagator
    stack_info = RuntimeSchemaInfo(1, needs_pytree=True)
    propagator.register_op_strategy("stack", stack_strategy, stack_info)
    propagator.register_op_strategy("unbind", gen_unbind_strategy, RuntimeSchemaInfo(1))
