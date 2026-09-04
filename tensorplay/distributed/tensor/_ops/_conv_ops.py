"""Placement propagation for convolution-shaped operations."""

from __future__ import annotations

from typing import Any

from .._dtensor_spec import DTensorSpec, TensorMeta
from .._op_schema import OpSchema, OutputSharding, RuntimeSchemaInfo
from ..placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    _StridedShard,
)
from .single_dim_strategy import _ShardingPlaceholder, register_single_dim_strategy

__all__ = [
    "_convolution_full_mesh_strategy_filter",
    "_supports_last_dim_sharding",
    "convolution_backward_rules",
    "convolution_backward_single_dim_strategy",
    "convolution_rules",
    "convolution_single_dim_strategy",
]


def _known_bool(value: Any) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    if bias_spec is not None:
        assert isinstance(bias_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    assert isinstance(stride, list)
    assert isinstance(padding, list)
    assert isinstance(dilation, list)
    output_spatial_shape = [
        (dim + 2 * padding[index] - dilation[index] * (weight_shape[index + 1] - 1) - 1)
        // stride[index]
        + 1
        for index, dim in enumerate(in_shape[2:])
    ]
    output_shape = [in_shape[0], weight_shape[0], *output_spatial_shape]
    output_stride = [1]
    for index in range(1, len(output_shape)):
        output_stride.insert(0, output_stride[0] * output_shape[-index])
    tensor_meta = TensorMeta(
        output_shape,
        output_stride,
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            input_spec.dim_map,
            input_spec.sums,
            tensor_meta=tensor_meta,
        )
    )


def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    if bias_shape_opt is not None:
        assert isinstance(bias_shape_opt, list)
    assert input_spec.tensor_meta is not None
    weight_tensor_meta = weight_spec.tensor_meta
    bias_tensor_meta = (
        TensorMeta(
            tuple(bias_shape_opt),
            (1,),
            input_spec.tensor_meta.dtype,
        )
        if bias_shape_opt is not None
        else None
    )

    grad_input_spec = input_spec
    grad_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1] * len(weight_tensor_meta.shape) if weight_tensor_meta is not None else [],
        [0],
        tensor_meta=weight_tensor_meta,
    )
    grad_bias_spec = (
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=bias_tensor_meta,
        )
        if bias_tensor_meta is not None
        else None
    )
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])


def _supports_last_dim_sharding(
    input_meta: TensorMeta,
    weight_meta: TensorMeta,
    stride: list[Any],
    padding: list[Any],
    dilation: list[Any],
    transposed: bool,
) -> bool:
    ndim = len(input_meta.shape)
    return (
        ndim == len(weight_meta.shape)
        and ndim in (3, 4, 5)
        and not transposed
        and _known_bool(padding[-1] == 0)
        and _known_bool(dilation[-1] == 1)
        and _known_bool(stride[-1] == weight_meta.shape[-1])
    )


def _operation_key(operation: Any) -> str:
    value = getattr(operation, "__name__", None)
    if value is None:
        value = getattr(operation, "name", None)
        if callable(value):
            value = value()
    if value is None:
        value = operation
    text = str(value)
    if text.endswith(".default"):
        text = text[: -len(".default")]
    return text.rsplit(".", 1)[-1].rsplit("::", 1)[-1]


def _convolution_full_mesh_strategy_filter(
    mesh: Any,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    _output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    operation = _operation_key(op_schema.op)
    if operation == "convolution":
        input_index, stride_index = 0, 3
    elif operation == "convolution_backward":
        input_index, stride_index = 1, 4
    else:
        raise AssertionError(f"unexpected convolution operation {op_schema.op}")

    input_spec = input_specs[input_index]
    last_dim = len(input_spec.shape) - 1
    last_dim_mesh_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(input_spec.placements)
        if placement.is_shard() and placement.dim == last_dim
    ]
    if not last_dim_mesh_dims:
        return True
    if len(last_dim_mesh_dims) != 1:
        return False

    last_dim_mesh_dim = last_dim_mesh_dims[0]
    last_dim_placement = input_spec.placements[last_dim_mesh_dim]
    if not isinstance(last_dim_placement, Shard) or isinstance(
        last_dim_placement, _StridedShard
    ):
        return False

    stride = op_schema.args_schema[stride_index]
    if not isinstance(stride, list):
        raise AssertionError(f"expected stride list, got {type(stride)}")
    return _known_bool(
        input_spec.shape[last_dim]
        % (stride[-1] * mesh.size(last_dim_mesh_dim))
        == 0
    )


@register_single_dim_strategy(
    ["convolution"],
    schema_info=RuntimeSchemaInfo(2),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_single_dim_strategy(
    _op: Any,
    args_schema: tuple[Any, ...],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[1]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(weight_meta)}")
    bias_meta = args_schema[2]
    stride, padding, dilation = args_schema[3:6]
    transposed = args_schema[6]
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        Replicate(),
    ]
    if bias_meta is not None:
        rule.append(Replicate())
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        last_dim_rule: list[Placement | _ShardingPlaceholder | None] = [
            Shard(last_dim),
            Shard(last_dim),
            Replicate(),
        ]
        if bias_meta is not None:
            last_dim_rule.append(Replicate())
        strategies.append(last_dim_rule)

    return strategies


@register_single_dim_strategy(
    ["convolution_backward"],
    schema_info=RuntimeSchemaInfo(3),
    full_mesh_strategy_filter=_convolution_full_mesh_strategy_filter,
)
def convolution_backward_single_dim_strategy(
    _op: Any,
    args_schema: tuple[Any, ...],
    _kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(input_meta)}")
    weight_meta = args_schema[2]
    if not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"expected TensorMeta, got {type(weight_meta)}")
    bias_sizes = args_schema[3]
    stride, padding, dilation = args_schema[4:7]
    transposed = args_schema[7]
    has_bias = bias_sizes is not None
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),
        Partial("sum"),
        Partial("sum") if has_bias else None,
        _ShardingPlaceholder(0),
        _ShardingPlaceholder(0),
        Replicate(),
    ]
    strategies = [rule]

    if _supports_last_dim_sharding(
        input_meta, weight_meta, stride, padding, dilation, transposed
    ):
        last_dim = len(input_meta.shape) - 1
        strategies.append(
            [
                Shard(last_dim),
                Partial("sum"),
                Partial("sum") if has_bias else None,
                Shard(last_dim),
                Shard(last_dim),
                Replicate(),
            ]
        )

    return strategies
