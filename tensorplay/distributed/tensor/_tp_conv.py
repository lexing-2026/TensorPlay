"""Tensor-parallel convolution execution paths."""

from __future__ import annotations

from typing import Any, cast

import tensorplay
import tensorplay.distributed as dist

from ._api import DTensor

__all__ = ["convolution_backward_handler", "convolution_handler", "tp_convolution"]


def _requires_data_exchange(padding: Any) -> bool:
    return padding[1] != 0


def _is_supported(
    input_size: Any,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
) -> bool:
    if dilation[1] != 1:
        raise RuntimeError("dilation must be 1 for tensor-parallel convolution")
    if padding[1] != 0:
        if stride[1] != 1:
            raise RuntimeError(
                "stride must be 1 when tensor-parallel convolution uses padding"
            )
        if kernel_size[3] // 2 > input_size[3]:
            raise RuntimeError(
                "half the convolution kernel width must fit the input width"
            )
    elif not (input_size[3] % stride[1] == 0 and stride[1] == kernel_size[3]):
        raise RuntimeError(
            "an unpadded tensor-parallel convolution requires divisible input width "
            "and stride equal to kernel width"
        )
    return True


def _ring_send_recv_construct(
    in_tensor: Any,
    d1: int,
    d2: int,
    left: int,
    right: int,
    rank: int,
    size: int,
) -> Any:
    send_to_right = in_tensor[:, :, :, -d1:].contiguous()
    send_to_left = in_tensor[:, :, :, :d2].contiguous()
    recv_from_right = tensorplay.zeros_like(send_to_left)
    recv_from_left = tensorplay.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    requests = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for request in requests:
        request.wait()

    if rank == 0:
        in_tensor = tensorplay.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = tensorplay.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = tensorplay.cat(
            [recv_from_left, in_tensor, recv_from_right], dim=-1
        )
    return in_tensor


def _ring_send_recv_aggregate(
    grad_in_tensor: Any,
    d1: int,
    d2: int,
    left: int,
    right: int,
    rank: int,
    size: int,
) -> Any:
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    recv_from_right = tensorplay.zeros_like(send_to_left)
    recv_from_left = tensorplay.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    requests = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for request in requests:
        request.wait()

    if rank == 0:
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        grad_in_tensor[:, :, :, -d1:] = tensorplay.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
    elif rank == size - 1:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        grad_in_tensor[:, :, :, :d2] = tensorplay.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    else:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        grad_in_tensor[:, :, :, -d1:] = tensorplay.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
        grad_in_tensor[:, :, :, :d2] = tensorplay.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    return grad_in_tensor


def tp_convolution(
    op_call: Any,
    local_tensor_args: tuple[Any, ...],
    local_tensor_kwargs: dict[str, Any],
) -> Any:
    assert getattr(op_call, "__name__", None) == "convolution"
    assert len(local_tensor_args) == 9

    rank = dist.get_rank()
    size = dist.get_world_size()
    in_tensor = cast(Any, local_tensor_args[0])
    weight = cast(Any, local_tensor_args[1])
    stride, padding, dilation = local_tensor_args[3:6]

    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    assert isinstance(padding, list)

    if not _requires_data_exchange(padding):
        return op_call(*local_tensor_args, **local_tensor_kwargs)

    d = weight.shape[3] - 1
    d1 = d // 2
    d2 = d - d1
    assert d1 + d2 == d
    right = (rank + 1) % size
    left = (rank - 1 + size) % size

    in_tensor = _ring_send_recv_construct(
        in_tensor, d1, d2, left, right, rank, size
    )

    local_args = list(local_tensor_args)
    local_args[0] = in_tensor
    local_results = op_call(*tuple(local_args), **local_tensor_kwargs)
    if isinstance(local_results, tuple):
        local_results = local_results[0]

    padding_w = padding[1]
    width = local_results.size(3)
    if rank == 0:
        local_results = local_results[:, :, :, : width - padding_w]
    elif rank == size - 1:
        local_results = local_results[:, :, :, padding_w:]
    else:
        local_results = local_results[:, :, :, padding_w : width - padding_w]
    return local_results


def tp_convolution_backward(
    op_call: Any,
    local_tensor_args: tuple[Any, ...],
    local_tensor_kwargs: dict[str, Any],
) -> Any:
    assert getattr(op_call, "__name__", None) == "convolution_backward"
    assert len(local_tensor_args) == 11

    rank = dist.get_rank()
    size = dist.get_world_size()
    grad_out_tensor = cast(Any, local_tensor_args[0])
    in_tensor = cast(Any, local_tensor_args[1])
    weight = cast(Any, local_tensor_args[2])
    stride, padding, dilation = local_tensor_args[4:7]

    assert _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation)
    assert isinstance(padding, list)

    if not _requires_data_exchange(padding):
        return op_call(*local_tensor_args, **local_tensor_kwargs)

    d = weight.shape[3] - 1
    d1 = d // 2
    d2 = d - d1
    assert d1 + d2 == d
    right = (rank + 1) % size
    left = (rank - 1 + size) % size

    in_tensor = _ring_send_recv_construct(
        in_tensor, d1, d2, left, right, rank, size
    )

    padding_w = padding[1]
    if rank == 0:
        grad_out_tensor = tensorplay.pad(
            grad_out_tensor, [0, padding_w], "constant", 0
        )
    elif rank == size - 1:
        grad_out_tensor = tensorplay.pad(
            grad_out_tensor, [padding_w, 0], "constant", 0
        )
    else:
        grad_out_tensor = tensorplay.pad(
            grad_out_tensor, [padding_w, padding_w], "constant", 0
        )

    local_args = list(local_tensor_args)
    local_args[0] = grad_out_tensor
    local_args[1] = in_tensor
    local_results = op_call(*tuple(local_args), **local_tensor_kwargs)
    grad_in_tensor = _ring_send_recv_aggregate(
        local_results[0], d1, d2, left, right, rank, size
    )

    local_results = list(local_results)
    local_results[0] = grad_in_tensor
    return tuple(local_results)


def convolution_handler(
    op_call: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    dispatcher = DTensor._op_dispatcher
    op_info = dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "convolution output sharding is required"
    local_results = tp_convolution(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )
    return dispatcher.wrap(local_results, output_sharding.output_spec)


def convolution_backward_handler(
    op_call: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    args = list(args)
    assert isinstance(args[0], DTensor) and isinstance(args[1], DTensor)
    args[0] = args[0].redistribute(args[1].device_mesh, args[1].placements)
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(
        op_call, tuple(args), kwargs
    )
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "convolution backward output sharding is required"
    local_results = tp_convolution_backward(
        op_call, tuple(op_info.local_args), op_info.local_kwargs
    )
    return DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )
