"""Distributed cross-entropy execution for class-sharded logits."""

from __future__ import annotations

import contextlib
from typing import Any, cast

import tensorplay as tp
from tensorplay.primitives.common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    elementwise_dtypes,
)
from tensorplay.autograd.function import Function

from ... import _functional_collectives as funcol
from .._api import DTensor
from .._dtensor_spec import TensorMeta
from .._ops._embedding_ops import _MaskPartial
from .._ops._math_ops import Reduction, _skip_dim, replicate_reduction_dims
from .._ops.utils import normalize_dim
from ..placement_types import Partial, Placement, Replicate, Shard

__all__ = ["loss_parallel"]


def _mesh_ndim(mesh: Any) -> int:
    value = getattr(mesh, "ndim")
    return int(value() if callable(value) else value)


def _reduction_name(value: Any) -> str:
    if isinstance(value, Reduction):
        name = value.value
    elif isinstance(value, str):
        name = value
    elif type(value) is int:
        name = {0: "none", 1: "mean", 2: "sum"}.get(value, "")
    else:
        name = ""
    if name not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported loss reduction: {value!r}")
    return name


def _find_all_reduce_mesh_dim(
    placements: tuple[Placement, ...], dim: int
) -> int:
    shard_mesh_dims = [
        index
        for index, placement in enumerate(placements)
        if isinstance(placement, Shard) and placement.dim == dim
    ]
    if len(shard_mesh_dims) != 1:
        raise ValueError(
            "loss_parallel requires exactly one mesh dimension to shard "
            f"tensor dimension {dim}; got {placements}"
        )
    mesh_dim = shard_mesh_dims[0]
    for index, placement in enumerate(placements):
        if index != mesh_dim and not isinstance(placement, (Shard, Replicate)):
            raise ValueError(
                "loss_parallel accepts only Shard or Replicate on non-class "
                f"mesh dimensions; got {placement} at {index}"
            )
    return mesh_dim


def _cast_to_dtensor(
    tensor: Any, placements: tuple[Placement, ...], mesh: Any
) -> DTensor:
    if isinstance(tensor, DTensor):
        if tensor.placements == placements:
            return tensor
        raise RuntimeError(
            f"expected placements {placements}, got {tensor.placements}"
        )
    if isinstance(tensor, tp.Tensor):
        if any(isinstance(placement, Shard) for placement in placements):
            raise ValueError(
                "a plain tensor cannot represent a sharded loss operand; "
                f"use a distributed tensor for placements {placements}"
            )
        return DTensor.from_local(
            tensor, device_mesh=mesh, placements=placements, run_check=False
        )
    raise TypeError(f"unsupported loss operand type: {type(tensor)!r}")


def _propagate_tensor_meta(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> TensorMeta | None:
    try:
        op_info = DTensor._op_dispatcher.unwrap_to_op_info(
            operation, args, kwargs
        )
        schema = op_info.schema
        if schema is None:
            return None
        tensor_meta = (
            DTensor._op_dispatcher.sharding_propagator._propagate_tensor_meta(
                schema
            )
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if isinstance(tensor_meta, TensorMeta):
        return tensor_meta
    if isinstance(tensor_meta, (tuple, list)) and tensor_meta:
        first = tensor_meta[0]
        if isinstance(first, TensorMeta):
            return first
    return None


def _log_softmax(
    value: Any,
    dim: int,
    half_to_float: bool,
    requested_dtype: Any,
    mesh: Any,
    mesh_dim: int,
) -> Any:
    computation_dtype, result_dtype = elementwise_dtypes(
        value,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )
    if requested_dtype is not None:
        result_dtype = requested_dtype
        value = value.to(dtype=requested_dtype)
    value = value.to(dtype=computation_dtype).contiguous()
    if value.numel() == 0:
        shifted = value
    else:
        value_max = tp.amax(value, dim, keepdim=True)
        value_max = funcol.all_reduce(
            value_max, reduce_op="max", group=(mesh, mesh_dim)
        )
        value_max = funcol.wait_tensor(value_max)
        shifted = value - value_max
    sum_exp = tp.sum(tp.exp(shifted), dim, keepdim=True)
    sum_exp = funcol.all_reduce(
        sum_exp, reduce_op="sum", group=(mesh, mesh_dim)
    )
    sum_exp = funcol.wait_tensor(sum_exp)
    result = shifted - tp.log(sum_exp)
    if not half_to_float:
        result = result.to(dtype=result_dtype)
    return result


def _log_softmax_handler(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> DTensor:
    value = cast(DTensor, args[0])
    dim = normalize_dim(cast(int, args[1]), value.ndim)
    option = args[2] if len(args) > 2 else tp.undefined
    if isinstance(option, bool):
        half_to_float = option
        requested_dtype = None
    else:
        half_to_float = False
        requested_dtype = None if option in (None, tp.undefined) else option
    spec = DTensor._op_dispatcher._spec_from_dtensor(value)
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, dim)
    result = _log_softmax(
        value.to_local(),
        dim,
        half_to_float,
        requested_dtype,
        spec.mesh,
        mesh_dim,
    )
    output_meta = _propagate_tensor_meta(operation, args, kwargs)
    if output_meta is None:
        output_meta = TensorMeta(value.shape, value.stride(), result.dtype)
    else:
        output_meta = TensorMeta(
            output_meta.shape, output_meta.stride, result.dtype
        )
    return DTensor(
        result,
        spec.mesh,
        spec.placements,
        shape=output_meta.shape,
        stride=output_meta.stride,
    )


def _log_softmax_backward_handler(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> DTensor:
    del operation, kwargs
    grad_output = cast(DTensor, args[0])
    input_dtype = args[3]
    return grad_output.to(input_dtype)


def _nll_loss_forward(
    value: Any,
    target: Any,
    weight: Any,
    local_weight: Any,
    reduction: str,
    ignore_index: int,
    input_shape: tuple[int, ...],
    channel_dim: int,
    mesh: Any,
    mesh_dim: int,
) -> tuple[Any, Any]:
    ndim = int(value.dim())

    def weight_view(current: Any) -> Any:
        if ndim > 1:
            shape = [1] * ndim
            shape[channel_dim] = current.shape[0]
            return current.view(shape)
        return current

    if weight is not None:
        if local_weight is None:
            raise AssertionError("local class weights are required")
        value = value * weight_view(local_weight)
    safe_target = tp.where(target != ignore_index, target, 0)
    safe_target_with_dim = safe_target.unsqueeze(channel_dim)
    partial = _MaskPartial(
        offset_shape=input_shape,
        offset_dim=channel_dim,
    )
    partitioned_target = partial._partition_value(
        safe_target_with_dim, mesh, mesh_dim
    )
    selected = tp.gather(value, channel_dim, partitioned_target)
    reduced = partial._reduce_value(selected, mesh, mesh_dim)
    reduced = funcol.wait_tensor(reduced)
    result = -reduced.squeeze(channel_dim)
    result = tp.where(target != ignore_index, result, 0)

    if reduction == "none" and ndim > 1:
        return result, value.new_full((), 0.0)

    if weight is not None:
        full_weight = weight_view(weight)
        shape = list(value.shape)
        shape[channel_dim] = -1
        expanded_weight = full_weight.expand(shape)
        weight_sum = tp.gather(
            expanded_weight, channel_dim, safe_target_with_dim
        ).squeeze(channel_dim)
        weight_sum = tp.where(target != ignore_index, weight_sum, 0)
        total_weight = weight_sum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(
            dtype=value.dtype, device=value.device
        )

    if reduction == "sum":
        result = result.sum()
    elif reduction == "mean":
        result = result.sum() / total_weight
    return result, total_weight


def _output_meta(
    meta: TensorMeta | None,
    result: Any,
    target: DTensor,
    reduction: str,
    input_ndim: int,
) -> TensorMeta:
    if meta is not None:
        return TensorMeta(meta.shape, meta.stride, result.dtype)
    if reduction == "none" and input_ndim > 1:
        return TensorMeta(target.shape, target.stride(), result.dtype)
    return TensorMeta((), (), result.dtype)


def _nll_loss_forward_handler(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[DTensor, Any]:
    value = cast(DTensor, args[0])
    target_value = args[1]
    weight_value = args[2]
    reduction = _reduction_name(args[3])
    ignore_index = int(args[4])
    channel_dim = 1 if value.ndim >= 2 else 0
    spec = DTensor._op_dispatcher._spec_from_dtensor(value)
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate = (Replicate(),) * _mesh_ndim(spec.mesh)
    target = _cast_to_dtensor(target_value, target_placements, spec.mesh)
    weight = (
        None
        if weight_value is None
        else _cast_to_dtensor(weight_value, all_replicate, spec.mesh)
    )
    local_weight = None
    if weight is not None:
        sharded_weight_placements = tuple(
            Shard(0) if index == mesh_dim else Replicate()
            for index in range(_mesh_ndim(spec.mesh))
        )
        local_weight = weight.redistribute(
            spec.mesh, sharded_weight_placements
        ).to_local()
        if local_weight.shape[0] != value.to_local().shape[channel_dim]:
            raise AssertionError("class weight shape does not match logits")

    if reduction == "none":
        output_placements = target_placements
    else:
        if reduction == "mean" and _mesh_ndim(spec.mesh) > 1:
            raise NotImplementedError(
                "mean loss reduction requires a one-dimensional mesh"
            )
        output_placements = tuple(
            Replicate()
            if index == mesh_dim
            else Partial()
            if isinstance(placement, Shard)
            else placement
            for index, placement in enumerate(spec.placements)
        )

    meta_args = list(args)
    meta_args[1] = target
    meta_args[2] = weight
    output_meta = _propagate_tensor_meta(operation, tuple(meta_args), kwargs)
    result, total_weight = _NLLLossFunction.apply(
        value.to_local(),
        target.to_local(),
        None
        if weight is None
        else weight.to_local().detach(),
        None if local_weight is None else local_weight.detach(),
        reduction,
        ignore_index,
        channel_dim,
        _NLLLossFunction._MeshRef(spec.mesh, tuple(value.shape)),
        mesh_dim,
    )
    output_meta = _output_meta(
        output_meta, result, target, reduction, value.ndim
    )
    return (
        DTensor(
            result,
            spec.mesh,
            output_placements,
            shape=output_meta.shape,
            stride=output_meta.stride,
        ),
        total_weight,
    )


def _nll_loss_and_log_softmax_backward(
    grad_output: Any,
    value: Any,
    target: Any,
    weight: Any,
    reduction: str,
    ignore_index: int,
    total_weight: Any,
    input_shape: tuple[int, ...],
    channel_dim: int,
    mesh: Any,
    mesh_dim: int,
    fuse_log_softmax: bool = True,
) -> Any:
    channel_dim = 0 if value.dim() < 2 else 1
    if reduction == "mean":
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    safe_target = tp.where(target != ignore_index, target, 0)
    grad_input = tp.zeros_like(value)
    partial = _MaskPartial(
        offset_shape=input_shape,
        offset_dim=channel_dim,
    )
    flat_target = safe_target.squeeze(channel_dim).flatten()
    masked_target = partial._partition_value(flat_target, mesh, mesh_dim)
    if partial.mask_buffer.data is None:
        raise AssertionError("loss target mask was not materialized")
    grad_update = partial.mask_buffer.data.to(grad_input.dtype) - 1.0
    indices = tp.arange(
        masked_target.shape[0], device=masked_target.device
    )
    if value.dim() == 1:
        grad_input[masked_target] = grad_update
    elif value.dim() == 2:
        grad_input[indices, masked_target] = grad_update
    else:
        transposed = grad_input.transpose(channel_dim, -1)
        intermediate_shape = transposed.shape
        flattened = transposed.reshape(-1, value.shape[channel_dim])
        flattened[indices, masked_target] = grad_update
        grad_input = flattened.view(intermediate_shape).transpose(
            channel_dim, -1
        )

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        shape = [1] * value.dim()
        shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(shape)
        expanded_shape = list(value.shape)
        expanded_shape[channel_dim] = -1
        expanded_weight = weight.expand(expanded_shape)
        target_weight = tp.gather(
            expanded_weight, channel_dim, target
        )
        grad_output = grad_output * target_weight

    grad_output = tp.where(target != ignore_index, grad_output, 0)
    if fuse_log_softmax:
        return (grad_input + tp.exp(value)) * grad_output
    return grad_input * grad_output


class _NLLLossFunction(Function):
    class _MeshRef:
        __slots__ = ("mesh", "input_shape")

        def __init__(self, mesh: Any, input_shape: tuple[int, ...]) -> None:
            self.mesh = mesh
            self.input_shape = input_shape

    @staticmethod
    def forward(
        context: Any,
        value: Any,
        target: Any,
        weight: Any,
        local_weight: Any,
        reduction: str,
        ignore_index: int,
        channel_dim: int,
        mesh_ref: Any,
        mesh_dim: int,
    ) -> tuple[Any, Any]:
        context.reduction = reduction
        context.ignore_index = ignore_index
        context.input_shape = mesh_ref.input_shape
        context.channel_dim = channel_dim
        context.mesh = mesh_ref.mesh
        context.mesh_dim = mesh_dim
        context.save_for_backward(value, target, weight, local_weight)
        result, total_weight = _nll_loss_forward(
            value,
            target,
            weight,
            local_weight,
            reduction,
            ignore_index,
            context.input_shape,
            channel_dim,
            context.mesh,
            mesh_dim,
        )
        context.total_weight = total_weight
        return result, total_weight

    @staticmethod
    def backward(
        context: Any, grad_output: Any, grad_total_weight: Any
    ) -> tuple[Any, ...]:
        del grad_total_weight
        value, target, weight, _local_weight = context.saved_tensors
        grad_value = _nll_loss_and_log_softmax_backward(
            grad_output,
            value,
            target,
            weight,
            context.reduction,
            context.ignore_index,
            context.total_weight,
            context.input_shape,
            context.channel_dim,
            context.mesh,
            context.mesh_dim,
            fuse_log_softmax=False,
        )
        return (
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _nll_loss_backward_handler(
    operation: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> DTensor:
    grad_output = cast(DTensor, args[0])
    value = cast(DTensor, args[1])
    target_value = args[2]
    weight_value = args[3]
    reduction = _reduction_name(args[4])
    ignore_index = int(args[5])
    total_weight_value = args[6]
    channel_dim = 1 if value.ndim >= 2 else 0
    spec = DTensor._op_dispatcher._spec_from_dtensor(value)
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate = (Replicate(),) * _mesh_ndim(spec.mesh)
    target = _cast_to_dtensor(target_value, target_placements, spec.mesh)
    weight = (
        None
        if weight_value is None
        else _cast_to_dtensor(weight_value, all_replicate, spec.mesh)
    )
    if reduction == "none":
        grad_output = grad_output.redistribute(
            spec.mesh, target_placements
        )

    meta_args = list(args)
    meta_args[0] = grad_output
    meta_args[2] = target
    meta_args[3] = weight
    meta_args[6] = _cast_to_dtensor(
        total_weight_value, all_replicate, spec.mesh
    )
    output_meta = _propagate_tensor_meta(operation, tuple(meta_args), kwargs)
    total_weight = (
        total_weight_value.to_local()
        if isinstance(total_weight_value, DTensor)
        else total_weight_value
    )
    result = _nll_loss_and_log_softmax_backward(
        grad_output.to_local(),
        value.to_local(),
        target.to_local(),
        None if weight is None else weight.to_local(),
        reduction,
        ignore_index,
        total_weight,
        tuple(value.shape),
        channel_dim,
        spec.mesh,
        mesh_dim,
    )
    output_meta = _output_meta(
        output_meta, result, target, "sum", value.ndim
    )
    return DTensor(
        result,
        spec.mesh,
        spec.placements,
        shape=output_meta.shape,
        stride=output_meta.stride,
    )


_CUSTOM_LOSS_HANDLERS = {
    "log_softmax": _log_softmax_handler,
    "_log_softmax": _log_softmax_handler,
    "_log_softmax_backward_data": _log_softmax_backward_handler,
    "nll_loss": _nll_loss_forward_handler,
    "nll_loss_forward": _nll_loss_forward_handler,
    "nll_loss2d": _nll_loss_forward_handler,
    "nll_loss2d_forward": _nll_loss_forward_handler,
    "nll_loss_backward": _nll_loss_backward_handler,
    "nll_loss2d_backward": _nll_loss_backward_handler,
}


def _enable_custom_loss_ops() -> None:
    DTensor._op_dispatcher._custom_op_handlers.update(_CUSTOM_LOSS_HANDLERS)


def _disable_custom_loss_ops() -> None:
    for operation in _CUSTOM_LOSS_HANDLERS:
        DTensor._op_dispatcher._custom_op_handlers.pop(operation, None)


@contextlib.contextmanager
def loss_parallel():
    _enable_custom_loss_ops()
    try:
        yield
    finally:
        _disable_custom_loss_ops()
