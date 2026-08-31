"""Tensor-parallel convolution helpers."""

from __future__ import annotations

from typing import Any

import tensorplay

__all__ = ["convolution_backward_handler", "convolution_handler", "tp_convolution"]


def _requires_data_exchange(padding: Any, dim_map: Any) -> bool:
    return bool(padding and any(value != -1 for value in dim_map[1:]))


def _is_supported(input_size: Any, kernel_size: Any, stride: Any, padding: Any, dilation: Any) -> bool:
    if dilation[-1] != 1:
        raise RuntimeError("tensor-parallel convolution requires unit dilation on the partitioned axis")
    if padding[-1] and stride[-1] != 1:
        raise RuntimeError("a padded partitioned convolution requires unit stride")
    if not padding[-1] and stride[-1] != kernel_size[-1]:
        raise RuntimeError("an unpadded partitioned convolution requires stride equal to kernel size")
    if input_size[-1] <= 0:
        raise ValueError("convolution input size must be positive")
    return True


def tp_convolution(op_call: Any, local_tensor_args: tuple[Any, ...], local_tensor_kwargs: dict[str, Any], dim_map: list[int]) -> Any:
    if not callable(op_call):
        raise TypeError("convolution operation must be callable")
    del dim_map
    return op_call(*local_tensor_args, **local_tensor_kwargs)


def tp_convolution_backward(op_call: Any, local_tensor_args: tuple[Any, ...], local_tensor_kwargs: dict[str, Any], dim_map: list[int]) -> Any:
    return tp_convolution(op_call, local_tensor_args, local_tensor_kwargs, dim_map)


def convolution_handler(op_call: Any, *args: Any, **kwargs: Any) -> Any:
    dim_map = kwargs.pop("dim_map", [])
    return tp_convolution(op_call, args, kwargs, dim_map)


def convolution_backward_handler(op_call: Any, *args: Any, **kwargs: Any) -> Any:
    dim_map = kwargs.pop("dim_map", [])
    return tp_convolution_backward(op_call, args, kwargs, dim_map)
