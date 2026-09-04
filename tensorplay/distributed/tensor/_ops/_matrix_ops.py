"""Placement rules for matrix products, contractions, and attention kernels."""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

from .._api import DTensor
from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OpStrategy, RuntimeSchemaInfo
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
    _is_shard_like,
)
from ._einsum_strategy import EinsumDims
from .single_dim_strategy import _ShardingPlaceholder, register_single_dim_strategy
from .utils import _is_tensor_like, _operation_name, normalize_dim, prod

__all__ = [
    "transpose_single_dim_strategy",
    "_scaled_mm_scale_placement",
    "gen_single_dim_einsum_strategies",
    "dot_single_dim_strategy",
    "mm_single_dim_strategy",
    "linear_single_dim_strategy",
    "addmm_single_dim_strategy",
    "bmm_single_dim_strategy",
    "baddbmm_single_dim_strategy",
    "scaled_mm_single_dim_strategy",
    "_scaled_dot_product_flash_attention_base_strategies",
    "scaled_dot_product_flash_attention_single_dim_strategy",
    "_scaled_dot_product_flash_attention_backward_base_strategies",
    "scaled_dot_product_flash_attention_backward_single_dim_strategy",
    "_scaled_dot_product_efficient_attention_base_strategies",
    "scaled_dot_product_efficient_attention_single_dim_strategy",
    "_scaled_dot_product_efficient_attention_backward_base_strategies",
    "scaled_dot_product_efficient_attention_backward_single_dim_strategy",
    "_scaled_dot_product_cudnn_attention_base_strategies",
    "scaled_dot_product_cudnn_attention_single_dim_strategy",
    "_scaled_dot_product_cudnn_attention_backward_base_strategies",
    "scaled_dot_product_cudnn_attention_backward_single_dim_strategy",
    "constant_pad_nd_single_dim_strategy",
    "_valid_grouped_mm_strides",
    "grouped_mm_single_dim_strategy",
]


def _spec(value: Any) -> DTensorSpec | None:
    if isinstance(value, DTensorSpec):
        return value
    if isinstance(value, DTensor):
        stride = value.stride() if callable(value.stride) else value.stride
        return DTensorSpec(
            value.device_mesh,
            value.placements,
            TensorMeta(tuple(value.shape), tuple(stride), value.dtype),
        )
    return None


def _meta(value: Any) -> TensorMeta | None:
    if isinstance(value, TensorMeta):
        return value
    if hasattr(value, "strategies"):
        strategies = getattr(value, "strategies")
        if strategies:
            return _meta(strategies[0].output_spec)
    value_spec = _spec(value)
    return None if value_spec is None else value_spec.tensor_meta


def _placeholder_or_shard(dim: int, source: Any) -> Any:
    if isinstance(source, _StridedShard):
        return _StridedShard(dim, split_factor=source.split_factor)
    if isinstance(source, Shard):
        return Shard(dim)
    return _ShardingPlaceholder(dim)


def _transpose_strategy(args_schema: Sequence[Any]) -> list[list[Any]]:
    input_meta = _meta(args_schema[0]) if args_schema else None
    if input_meta is None:
        raise AssertionError("transpose requires tensor metadata")
    ndim = len(input_meta.shape)
    if ndim <= 1:
        strategies = [[_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)] for dim in range(ndim)]
    else:
        strategies = [
            [_ShardingPlaceholder(1), _ShardingPlaceholder(0)],
            [_ShardingPlaceholder(0), _ShardingPlaceholder(1)],
        ]
    strategies.extend(
        [[Partial(reduce_op), Partial(reduce_op)] for reduce_op in ("sum", "avg", "max", "min")]
    )
    return strategies


def transpose_single_dim_strategy(
    value: Any, dim0: Any = 0, dim1: Any = 1
) -> DTensorSpec | list[list[Any]]:
    if _is_tensor_like(value):
        spec = _spec(value)
        if spec is None:
            raise AssertionError("transpose requires a distributed tensor")
        first = normalize_dim(int(dim0), spec.ndim)
        second = normalize_dim(int(dim1), spec.ndim)
        permutation = list(range(spec.ndim))
        permutation[first], permutation[second] = permutation[second], permutation[first]
        placements = tuple(
            _placeholder_or_shard(permutation.index(placement.dim), placement)
            if _is_shard_like(placement)
            else placement
            for placement in spec.placements
        )
        shape = tuple(spec.shape[index] for index in permutation) if spec.tensor_meta else None
        stride = tuple(spec.stride[index] for index in permutation) if spec.tensor_meta else None
        meta = None if shape is None else TensorMeta(shape, stride or (), spec.tensor_meta.dtype)
        return DTensorSpec(
            spec.mesh,
            placements,
            meta,
            use_strided_shard_as_shard_order=spec.use_strided_shard_as_shard_order,
        )
    return _transpose_strategy(dim0 if isinstance(dim0, (tuple, list)) else (),)


def _scaled_mm_scale_placement(
    data_placement: Placement | _ShardingPlaceholder,
    scale_shape: Sequence[int],
    contracting_dim: int,
) -> Placement | _ShardingPlaceholder | None:
    if prod(scale_shape) == 1:
        return Replicate()
    if len(scale_shape) != 1:
        return data_placement
    if isinstance(data_placement, _ShardingPlaceholder):
        return None if data_placement.dim == contracting_dim else _ShardingPlaceholder(0)
    if _is_shard_like(data_placement):
        return None if data_placement.dim == contracting_dim else _placeholder_or_shard(0, data_placement)
    return Replicate() if isinstance(data_placement, (Replicate, Partial)) else data_placement


def _bias_dim_map(output_ndim: int, bias_shape: Sequence[int]) -> tuple[int | None, ...]:
    bias_ndim = len(bias_shape)
    padding = output_ndim - bias_ndim
    return tuple(
        None
        if index < padding or int(bias_shape[index - padding]) == 1
        else index - padding
        for index in range(output_ndim)
    )


def gen_single_dim_einsum_strategies(
    equation: str,
    *,
    bias_shape: Sequence[int] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    dims = EinsumDims.parse_dims(input_dims, output_dim)
    bias_map = None if bias_shape is None else _bias_dim_map(len(output_dim), bias_shape)

    def bias_placement(placement: Placement | _ShardingPlaceholder) -> Placement | _ShardingPlaceholder:
        if bias_map is None:
            return placement
        if isinstance(placement, _ShardingPlaceholder):
            mapped = bias_map[placement.dim]
            return Replicate() if mapped is None else _ShardingPlaceholder(mapped)
        return placement

    def with_bias(values: list[Placement | _ShardingPlaceholder]) -> list[Placement | _ShardingPlaceholder]:
        if bias_shape is None:
            return values
        return [values[0], bias_placement(values[0]), *values[1:]]

    result: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in dims.batch_dims:
        result.append(
            with_bias(
                [_ShardingPlaceholder(output_dim.index(dim))]
                + [_ShardingPlaceholder(item.index(dim)) for item in input_dims]
            )
        )
    for dim in dims.contracting_dims:
        result.append(
            with_bias(
                [Partial("sum")]
                + [_ShardingPlaceholder(item.index(dim)) for item in input_dims]
            )
        )
    for dim in dims.lhs_out_only_dims:
        result.append(
            with_bias(
                [
                    _ShardingPlaceholder(output_dim.index(dim)),
                    _ShardingPlaceholder(input_dims[0].index(dim)),
                    Replicate(),
                ]
            )
        )
    for dim in dims.rhs_out_only_dims:
        result.append(
            with_bias(
                [
                    _ShardingPlaceholder(output_dim.index(dim)),
                    Replicate(),
                    _ShardingPlaceholder(input_dims[1].index(dim)),
                ]
            )
        )
    for reduce_op in Partial.LINEAR_REDUCE_OPS:
        result.append(with_bias([Partial(reduce_op), Partial(reduce_op), Replicate()]))
        result.append(with_bias([Partial(reduce_op), Replicate(), Partial(reduce_op)]))
    if not dims.contracting_dims and not dims.lhs_out_only_dims and not dims.rhs_out_only_dims:
        for reduce_op in Partial.LINEAR_REDUCE_OPS:
            result.append(with_bias([Partial(reduce_op)] * (len(input_dims) + 1)))
    return result


def _matrix_output_shape(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    left_shape, right_shape = tuple(left), tuple(right)
    if not left_shape or not right_shape:
        raise ValueError("matrix operands must have at least one dimension")
    if len(left_shape) == 1 and len(right_shape) == 1:
        if left_shape[0] != right_shape[0]:
            raise ValueError("vector operands must have the same length")
        return ()
    if len(left_shape) == 1:
        if left_shape[0] != right_shape[-2]:
            raise ValueError("matrix operands have incompatible contraction dimensions")
        return tuple(right_shape[:-2]) + (right_shape[-1],)
    if len(right_shape) == 1:
        if left_shape[-1] != right_shape[0]:
            raise ValueError("matrix operands have incompatible contraction dimensions")
        return tuple(left_shape[:-2]) + (left_shape[-2],)
    if left_shape[-1] != right_shape[-2]:
        raise ValueError("matrix operands have incompatible contraction dimensions")
    batch = _broadcast_shapes(left_shape[:-2], right_shape[:-2])
    return batch + (left_shape[-2], right_shape[-1])


def _broadcast_shapes(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for index in range(1, max(len(left), len(right)) + 1):
        a = left[-index] if index <= len(left) else 1
        b = right[-index] if index <= len(right) else 1
        if a not in (1, b) and b not in (1, a):
            raise ValueError(f"matrix batch dimensions {left} and {right} are not broadcastable")
        result.append(max(int(a), int(b)))
    return tuple(reversed(result))


def _matrix_output_dim(operand: str, input_dim: int, left_ndim: int, right_ndim: int) -> int | None:
    batch_ndim = max(left_ndim - 2, right_ndim - 2)
    if operand == "left":
        if left_ndim == 1 or input_dim == left_ndim - 1:
            return None
        if input_dim == left_ndim - 2:
            return batch_ndim
        return input_dim + batch_ndim - (left_ndim - 2)
    if right_ndim == 1 or input_dim == right_ndim - 2:
        return None
    if input_dim == right_ndim - 1:
        return batch_ndim + (0 if left_ndim == 1 else 1)
    return input_dim + batch_ndim - (right_ndim - 2)


def _merge_matrix_placement(current: Any, candidate: Any) -> Any:
    if current is None:
        return candidate
    if current == candidate:
        return current
    if isinstance(current, Partial) and isinstance(candidate, Partial):
        return current if current.reduce_op == candidate.reduce_op else Replicate()
    return Replicate()


def _bias_output_dim(bias_ndim: int, bias_dim: int, output_ndim: int) -> int:
    return output_ndim - bias_ndim + bias_dim


def _shape_dim(value: DTensor | DTensorSpec, dim: int) -> int | None:
    try:
        return int(value.shape[dim])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _matrix_output_spec(
    left: DTensor | DTensorSpec,
    right: DTensor | DTensorSpec,
    placements: Sequence[Placement],
) -> DTensorSpec:
    left_spec, right_spec = _spec(left), _spec(right)
    if left_spec is None or right_spec is None:
        raise TypeError("matrix operands must be distributed tensor values")
    if left_spec.tensor_meta is None or right_spec.tensor_meta is None:
        return DTensorSpec(left_spec.mesh, tuple(placements), None)
    shape = _matrix_output_shape(left_spec.shape, right_spec.shape)
    stride = [1] * len(shape)
    running = 1
    for index in reversed(range(len(shape))):
        stride[index] = running
        running *= int(shape[index])
    return DTensorSpec(
        left_spec.mesh,
        tuple(placements),
        TensorMeta(shape, tuple(stride), left_spec.tensor_meta.dtype),
    )


def _mm_output_spec(
    left: DTensor | DTensorSpec,
    right: DTensor | DTensorSpec,
    bias: DTensor | DTensorSpec | None = None,
) -> DTensorSpec:
    left_spec, right_spec = _spec(left), _spec(right)
    if left_spec is None or right_spec is None:
        raise TypeError("matrix operands must be distributed tensor values")
    if left_spec.mesh != right_spec.mesh:
        raise ValueError("matrix operands must use the same mesh")
    left_ndim, right_ndim = left_spec.ndim, right_spec.ndim
    output_ndim = len(_matrix_output_shape(left_spec.shape, right_spec.shape)) if left_spec.tensor_meta and right_spec.tensor_meta else max(left_ndim, right_ndim)
    placements: list[Any] = [None] * len(left_spec.placements)
    conflicts = [False] * len(placements)

    def add_operand(operand: str, value: DTensor | DTensorSpec) -> None:
        value_spec = _spec(value)
        if value_spec is None or len(value_spec.placements) != len(placements):
            raise ValueError("matrix operands must use the same mesh rank")
        ndim = left_ndim if operand == "left" else right_ndim
        contracting_dim = ndim - 1 if operand == "left" else 0
        for mesh_dim, source in enumerate(value_spec.placements):
            if conflicts[mesh_dim] or isinstance(source, Replicate):
                continue
            if isinstance(source, Partial):
                candidate = source
            elif _is_shard_like(source):
                if source.dim < 0 or source.dim >= ndim:
                    conflicts[mesh_dim] = True
                    placements[mesh_dim] = Replicate()
                    continue
                if source.dim == contracting_dim:
                    candidate = Partial("sum")
                else:
                    is_batch = ndim >= 2 and source.dim < ndim - 2
                    if is_batch and _shape_dim(value, source.dim) == 1:
                        continue
                    output_dim = _matrix_output_dim(operand, source.dim, left_ndim, right_ndim)
                    if output_dim is None:
                        conflicts[mesh_dim] = True
                        placements[mesh_dim] = Replicate()
                        continue
                    candidate = _placeholder_or_shard(output_dim, source)
            else:
                continue
            merged = _merge_matrix_placement(placements[mesh_dim], candidate)
            if isinstance(merged, Replicate) and placements[mesh_dim] is not None and merged != placements[mesh_dim]:
                conflicts[mesh_dim] = True
            placements[mesh_dim] = merged

    add_operand("left", left)
    add_operand("right", right)

    if bias is not None:
        bias_spec = _spec(bias)
        if bias_spec is None or bias_spec.mesh != left_spec.mesh:
            raise ValueError("matrix bias must use the same mesh")
        bias_ndim = bias_spec.ndim
        if bias_ndim > output_ndim:
            raise ValueError("matrix bias rank exceeds the matrix output rank")
        for mesh_dim, source in enumerate(bias_spec.placements):
            if conflicts[mesh_dim] or isinstance(source, Replicate):
                continue
            if isinstance(source, Partial):
                candidate = source
            elif _is_shard_like(source):
                if source.dim < 0 or source.dim >= bias_ndim:
                    conflicts[mesh_dim] = True
                    placements[mesh_dim] = Replicate()
                    continue
                if _shape_dim(bias, source.dim) == 1:
                    continue
                candidate = _placeholder_or_shard(_bias_output_dim(bias_ndim, source.dim, output_ndim), source)
            else:
                continue
            merged = _merge_matrix_placement(placements[mesh_dim], candidate)
            if isinstance(merged, Replicate) and placements[mesh_dim] is not None and merged != placements[mesh_dim]:
                conflicts[mesh_dim] = True
            placements[mesh_dim] = merged

    return _matrix_output_spec(left, right, tuple(value if value is not None else Replicate() for value in placements))


def mm_single_dim_strategy(
    left: Any, right: Any = None, bias: Any = None
) -> DTensorSpec | list[list[Placement | _ShardingPlaceholder]]:
    if _is_tensor_like(left) and _is_tensor_like(right):
        return _mm_output_spec(left, right, bias=bias)
    return gen_single_dim_einsum_strategies("mk,kn->mn")


def linear_single_dim_strategy(
    input_value: DTensor | DTensorSpec,
    weight: DTensor | DTensorSpec,
    bias: DTensor | DTensorSpec | None = None,
) -> DTensorSpec:
    weight_t = transpose_single_dim_strategy(weight, 0, 1)
    if not isinstance(weight_t, DTensorSpec):
        raise AssertionError("linear weight transpose did not produce a specification")
    return _mm_output_spec(input_value, weight_t, bias=bias)


def dot_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("i,i->")


def addmm_single_dim_strategy(*args: Any, **kwargs: Any) -> Any:
    if args and _is_tensor_like(args[0]):
        return _mm_output_spec(args[1], args[2], bias=args[0])
    return gen_single_dim_einsum_strategies(
        "mk,kn->mn", bias_shape=_meta(args[1][0]).shape if len(args) > 1 and _meta(args[1][0]) else None
    )


def bmm_single_dim_strategy(*args: Any, **kwargs: Any) -> Any:
    if args and _is_tensor_like(args[0]):
        return _mm_output_spec(args[0], args[1])
    return gen_single_dim_einsum_strategies("bmk,bkn->bmn")


def baddbmm_single_dim_strategy(*args: Any, **kwargs: Any) -> Any:
    if args and _is_tensor_like(args[0]):
        return _mm_output_spec(args[1], args[2], bias=args[0])
    bias_meta = _meta(args[1][0]) if len(args) > 1 and args[1] else None
    return gen_single_dim_einsum_strategies(
        "bmk,bkn->bmn", bias_shape=None if bias_meta is None else bias_meta.shape
    )


def scaled_mm_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Placement | _ShardingPlaceholder]]:
    scale_self = _meta(args_schema[2]) if len(args_schema) > 2 else None
    scale_mat2 = _meta(args_schema[3]) if len(args_schema) > 3 else None
    if scale_self is None or scale_mat2 is None:
        raise AssertionError("scaled matrix multiplication requires scale metadata")
    result = []
    for strategy in gen_single_dim_einsum_strategies("mk,kn->mn"):
        self_scale = _scaled_mm_scale_placement(strategy[1], scale_self.shape, 1)
        mat2_scale = _scaled_mm_scale_placement(strategy[2], scale_mat2.shape, 0)
        if self_scale is not None and mat2_scale is not None:
            result.append([*strategy, self_scale, mat2_scale])
    return result


def _scaled_dot_product_flash_attention_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    return_debug_mask = len(op_schema.args_schema) >= 6 and bool(
        op_schema.args_schema[5]
    )
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(q_input_strategy)}")

    debug_attn_mask_sharding: Placement = (
        Replicate() if not return_debug_mask else Shard(1)
    )
    return [
        [
            Replicate(),
            Replicate(),
            None,
            None,
            None,
            None,
            Replicate(),
            None,
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
        ],
        [
            Shard(1),
            Shard(1),
            None,
            None,
            None,
            None,
            Replicate(),
            None,
            debug_attn_mask_sharding,
            Shard(1),
            Shard(1),
            Shard(1),
        ],
        [
            Shard(0),
            Shard(0),
            None,
            None,
            None,
            None,
            Replicate(),
            None,
            Shard(0) if return_debug_mask else Replicate(),
            Shard(0),
            Shard(0),
            Shard(0),
        ],
    ]


@register_single_dim_strategy(
    "_scaled_dot_product_flash_attention", schema_info=RuntimeSchemaInfo(5)
)
def scaled_dot_product_flash_attention_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(q_meta)}")

    return_debug_mask = len(args_schema) >= 6 and bool(args_schema[5])
    debug_attn_mask_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if return_debug_mask else Replicate()
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if return_debug_mask else Replicate()
    )
    return [
        [
            _ShardingPlaceholder(1),
            _ShardingPlaceholder(1),
            None,
            None,
            None,
            None,
            Replicate(),
            None,
            debug_attn_mask_head,
            _ShardingPlaceholder(1),
            _ShardingPlaceholder(1),
            _ShardingPlaceholder(1),
        ],
        [
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),
            None,
            None,
            None,
            None,
            Replicate(),
            None,
            debug_attn_mask_batch,
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),
        ],
    ]


def _scaled_dot_product_flash_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(q_input_strategy)}")
    num_tensor_inputs = sum(
        isinstance(arg_spec, OpStrategy) for arg_spec in op_schema.args_schema
    )
    if num_tensor_inputs < 6:
        raise AssertionError(
            f"expected at least 6 tensor inputs, got {num_tensor_inputs}"
        )

    all_replicate = [Replicate()] * (3 + num_tensor_inputs)
    num_heads_dim_sharding = [
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
    ]
    num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    batch_dim_sharding = [Shard(0)] * 9
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    return [all_replicate, num_heads_dim_sharding, batch_dim_sharding]


@register_single_dim_strategy("_scaled_dot_product_flash_attention_backward")
def scaled_dot_product_flash_attention_backward_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    if num_tensor_inputs < 6:
        raise AssertionError(
            f"expected at least 6 tensor inputs, got {num_tensor_inputs}"
        )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
    ]
    num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    batch_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
    ]
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    return [num_heads_dim_sharding, batch_dim_sharding]


def _scaled_dot_product_efficient_attention_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(q_input_strategy)}")
    has_attn_bias = op_schema.args_schema[3] is not None
    compute_log_sumexp = bool(op_schema.args_schema[4])

    all_replicate = [
        Replicate(),
        Replicate(),
        None,
        None,
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())
    head = [
        Shard(1),
        Shard(1) if compute_log_sumexp else Replicate(),
        None,
        None,
        Shard(1),
        Shard(1),
        Shard(1),
    ]
    batch = [
        Shard(0),
        Shard(0) if compute_log_sumexp else Replicate(),
        None,
        None,
        Shard(0),
        Shard(0),
        Shard(0),
    ]
    if has_attn_bias:
        head.append(Shard(1))
        batch.append(Shard(0))
    return [all_replicate, head, batch]


@register_single_dim_strategy(
    "_scaled_dot_product_efficient_attention", schema_info=RuntimeSchemaInfo(4)
)
def scaled_dot_product_efficient_attention_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(q_meta)}")
    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = bool(args_schema[4])
    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )
    head: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),
        logsumexp_head,
        None,
        None,
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
    ]
    batch: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        logsumexp_batch,
        None,
        None,
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
    ]
    if has_attn_bias:
        head.append(_ShardingPlaceholder(1))
        batch.append(_ShardingPlaceholder(0))
    return [head, batch]


def _scaled_dot_product_efficient_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(q_input_strategy)}")
    has_attn_bias = op_schema.args_schema[4] is not None
    all_replicate = [Replicate()] * (12 + int(has_attn_bias))
    if not has_attn_bias:
        all_replicate[3] = None

    head: list[Placement | None] = [
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1) if has_attn_bias else None,
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
        Shard(1),
    ]
    if has_attn_bias:
        head.insert(8, Shard(1))
    head.extend([Replicate(), Replicate()])

    batch: list[Placement | None] = [
        Shard(0),
        Shard(0),
        Shard(0),
        Shard(0) if has_attn_bias else None,
        Shard(0),
        Shard(0),
        Shard(0),
        Shard(0),
        Shard(0),
        Shard(0),
    ]
    if has_attn_bias:
        batch.insert(8, Shard(0))
    batch.extend([Replicate(), Replicate()])
    return [all_replicate, head, batch]


@register_single_dim_strategy("_scaled_dot_product_efficient_attention_backward")
def scaled_dot_product_efficient_attention_backward_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    has_attn_bias = args_schema[4] is not None
    head: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1) if has_attn_bias else None,
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
    ]
    if has_attn_bias:
        head.append(_ShardingPlaceholder(1))
    head.extend(
        [
            _ShardingPlaceholder(1),
            _ShardingPlaceholder(1),
            Replicate(),
            Replicate(),
        ]
    )
    batch: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0) if has_attn_bias else None,
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
    ]
    if has_attn_bias:
        batch.append(_ShardingPlaceholder(0))
    batch.extend(
        [
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),
            Replicate(),
            Replicate(),
        ]
    )
    return [head, batch]


def _scaled_dot_product_cudnn_attention_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    query_strategy, _, _, attn_bias_strategy, compute_log_sumexp, *rest_args = (
        op_schema.args_schema
    )
    return_debug_mask = len(op_schema.args_schema) >= 8 and bool(rest_args[2])
    has_attn_bias = attn_bias_strategy is not None
    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(query_strategy)}")

    debug_attn_mask_sharding: Placement | None = (
        Replicate() if return_debug_mask else None
    )
    all_replicate: list[Placement | None] = [
        Replicate(),
        Replicate(),
        None,
        None,
        None,
        None,
        None,
        None,
        debug_attn_mask_sharding,
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())

    head: list[Placement | None] = [
        Shard(1),
        Shard(1) if compute_log_sumexp else Replicate(),
        None,
        None,
        None,
        None,
        None,
        None,
        Shard(1) if return_debug_mask else None,
        Shard(1),
        Shard(1),
        Shard(1),
    ]
    if has_attn_bias:
        head.append(Shard(1))

    batch: list[Placement | None] = [
        Shard(0),
        Shard(0) if compute_log_sumexp else Replicate(),
        None,
        None,
        None,
        None,
        None,
        None,
        Shard(0) if return_debug_mask else None,
        Shard(0),
        Shard(0),
        Shard(0),
    ]
    if has_attn_bias:
        batch.append(Shard(0))
    return [all_replicate, head, batch]


@register_single_dim_strategy(
    "_scaled_dot_product_cudnn_attention", schema_info=RuntimeSchemaInfo(4)
)
def scaled_dot_product_cudnn_attention_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    query_meta = args_schema[0]
    if not isinstance(query_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(query_meta)}")
    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = bool(args_schema[4])
    return_debug_mask = len(args_schema) >= 8 and bool(args_schema[7])
    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_head: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(1) if return_debug_mask else None
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(0) if return_debug_mask else None
    )
    head: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),
        logsumexp_head,
        None,
        None,
        None,
        None,
        None,
        None,
        debug_attn_mask_head,
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
    ]
    if has_attn_bias:
        head.append(_ShardingPlaceholder(1))
    batch: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        logsumexp_batch,
        None,
        None,
        None,
        None,
        None,
        None,
        debug_attn_mask_batch,
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
    ]
    if has_attn_bias:
        batch.append(_ShardingPlaceholder(0))
    return [head, batch]


def _scaled_dot_product_cudnn_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[list[Any]]:
    if len(op_schema.args_schema) < 15:
        raise AssertionError(
            f"expected at least 15 args_schema, got {len(op_schema.args_schema)}"
        )
    has_attn_bias = op_schema.args_schema[8] is not None
    has_scale = len(op_schema.args_schema) >= 16 and False
    query_strategy = op_schema.args_schema[1]
    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"expected OpStrategy, got {type(query_strategy)}")

    all_replicate: list[Placement | None] = [Replicate()] * 3
    all_replicate.extend([Replicate()] * 6)
    all_replicate.extend([Replicate(), Replicate()])
    all_replicate.append(Replicate() if has_attn_bias else None)
    all_replicate.extend([None] * 6)
    if has_scale:
        all_replicate.append(None)

    head: list[Placement | None] = [Shard(1)] * 3
    head.extend([Shard(1)] * 4)
    head.extend([Shard(1), Shard(1)])
    head.extend([Replicate(), Replicate()])
    head.append(Shard(1) if has_attn_bias else None)
    head.extend([None] * 6)
    if has_scale:
        head.append(None)

    batch: list[Placement | None] = [Shard(0)] * 3
    batch.extend([Shard(0)] * 4)
    batch.extend([Shard(0), Shard(0)])
    batch.extend([Replicate(), Replicate()])
    batch.append(Shard(0) if has_attn_bias else None)
    batch.extend([None] * 6)
    if has_scale:
        batch.append(None)
    return [all_replicate, head, batch]


@register_single_dim_strategy("_scaled_dot_product_cudnn_attention_backward")
def scaled_dot_product_cudnn_attention_backward_single_dim_strategy(
    _op: Any,
    args_schema: Sequence[Any],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    import tensorplay as tp

    if len(args_schema) < 15:
        raise AssertionError(f"expected at least 15 args, got {len(args_schema)}")
    for arg in args_schema[:6]:
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"expected TensorMeta, got {type(arg)}")

    philox_placements: list[Placement] = []
    for arg in args_schema[6:8]:
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, tp.Tensor):
            raise AssertionError(f"expected TensorMeta or Tensor, got {type(arg)}")

    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, tp.Tensor)):
        raise AssertionError(
            f"expected TensorMeta or Tensor, got {type(args_schema[8])}"
        )

    cum_seq_placements: list[None] = []
    for arg in args_schema[9:11]:
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, tp.Tensor):
            continue
        else:
            raise AssertionError(f"expected TensorMeta or Tensor, got {type(arg)}")

    head: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
        _ShardingPlaceholder(1),
    ]
    head.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        head.append(_ShardingPlaceholder(1))
    head.extend(cum_seq_placements)

    batch: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
    ]
    batch.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        batch.append(_ShardingPlaceholder(0))
    batch.extend(cum_seq_placements)
    return [head, batch]


def constant_pad_nd_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Any]]:
    meta = _meta(args_schema[0])
    if meta is None:
        raise AssertionError("padding requires tensor metadata")
    pad = args_schema[1] if len(args_schema) > 1 else ()
    padded_dims = {
        len(meta.shape) - 1 - index
        for index in range(len(pad) // 2)
        if pad[2 * index] != 0 or pad[2 * index + 1] != 0
    }
    result = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(meta.shape))
        if dim not in padded_dims
    ]
    value = args_schema[2] if len(args_schema) > 2 else 0
    reduce_ops = ("sum", "avg", "max", "min") if all(item == 0 for item in pad) or value == 0 else ("avg", "max", "min")
    result.extend([[Partial(op), Partial(op)] for op in reduce_ops])
    return result


def _valid_grouped_mm_strides(
    mesh: Any,
    op_schema: OpSchema,
    input_specs: Sequence[DTensorSpec],
    output_specs: Any,
) -> bool:
    del mesh, op_schema, output_specs
    for spec in input_specs[:2]:
        if spec.tensor_meta is None or len(spec.shape) < 2:
            return False
        stride = spec.stride
        if stride[-1] == 1 and stride[-2] >= max(1, spec.shape[-1]):
            continue
        if stride[-2] == 1 and stride[-1] >= max(1, spec.shape[-2]):
            continue
        return False
    return True


def grouped_mm_single_dim_strategy(
    operation: Any, args_schema: Sequence[Any], kwargs_schema: dict[str, Any]
) -> list[list[Any]]:
    mat1 = _meta(args_schema[0])
    mat2 = _meta(args_schema[1])
    if mat1 is None or mat2 is None:
        raise AssertionError("grouped matrix multiplication requires tensor metadata")
    tail = [Replicate()] if len(args_schema) > 2 and args_schema[2] is not None else []
    result: list[list[Any]] = [
        [Partial("sum"), Partial("sum"), Replicate(), *tail],
        [Partial("sum"), Replicate(), Partial("sum"), *tail],
    ]
    n1, n2 = len(mat1.shape), len(mat2.shape)
    if n1 == 2 and n2 == 3:
        result.extend([
            [_ShardingPlaceholder(1), Replicate(), _ShardingPlaceholder(2), *tail],
            [Partial("sum"), _ShardingPlaceholder(1), _ShardingPlaceholder(1), *tail],
        ])
    elif n1 == 3 and n2 == 2:
        result.extend([
            [Partial("sum"), _ShardingPlaceholder(2), _ShardingPlaceholder(0), *tail],
            [_ShardingPlaceholder(0), _ShardingPlaceholder(1), Replicate(), *tail],
        ])
    elif n1 == 2 and n2 == 2:
        result.extend([
            [_ShardingPlaceholder(2), Replicate(), _ShardingPlaceholder(1), *tail],
            [_ShardingPlaceholder(1), _ShardingPlaceholder(0), Replicate(), *tail],
        ])
    elif n1 == 3 and n2 == 3:
        result.extend([
            [_ShardingPlaceholder(2), Replicate(), _ShardingPlaceholder(2), *tail],
            [_ShardingPlaceholder(1), _ShardingPlaceholder(1), Replicate(), *tail],
            [Partial("sum"), _ShardingPlaceholder(2), _ShardingPlaceholder(1), *tail],
            [_ShardingPlaceholder(0), _ShardingPlaceholder(0), _ShardingPlaceholder(0), *tail],
        ])
    return result
