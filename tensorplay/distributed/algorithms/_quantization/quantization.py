from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Callable

import tensorplay as tp
from tensorplay import distributed as dist

__all__ = ["DQuantType", "auto_quantize"]

_HALF_MIN = -65504.0
_HALF_MAX = 65504.0


class DQuantType(Enum):
    FP16 = "fp16"
    BFP16 = "bfp16"

    def __str__(self) -> str:
        return self.value


def _fp32_to_fp16_with_clamp(value: Any) -> Any:
    return value.clamp(_HALF_MIN, _HALF_MAX).to(tp.float16)


def _quantize_tensor(value: Any, qtype: DQuantType) -> Any:
    if not isinstance(value, tp.Tensor):
        raise TypeError("quantization expects a tensor")
    if qtype == DQuantType.FP16:
        return _fp32_to_fp16_with_clamp(value)
    if qtype == DQuantType.BFP16:
        return value.to(tp.bfloat16)
    raise ValueError(f"unsupported quantization type {qtype!r}")


def _quantize_tensor_list(values: list[Any], qtype: DQuantType) -> list[Any]:
    if not isinstance(values, list) or not all(isinstance(value, tp.Tensor) for value in values):
        raise TypeError("quantization expects a list of tensors")
    return [_quantize_tensor(value, qtype) for value in values]


def _dequantize_tensor(value: Any, qtype: DQuantType, quant_loss: Any = None) -> Any:
    if not isinstance(value, tp.Tensor):
        raise TypeError("dequantization expects a tensor")
    result = value.to(tp.float32)
    return result / quant_loss if quant_loss is not None else result


def _dequantize_tensor_list(values: list[Any], qtype: DQuantType, quant_loss: Any = None) -> list[Any]:
    return [_dequantize_tensor(value, qtype, quant_loss) for value in values]


def auto_quantize(func: Callable[..., Any], qtype: DQuantType, quant_loss: Any = None) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("async_op", False):
            raise RuntimeError("quantized communication does not support async operations")
        group = kwargs.get("group")
        if func is dist.all_gather:
            output, source = args[:2]
            quantized_source = _quantize_tensor(source, qtype)
            quantized_output = _quantize_tensor_list(output, qtype)
            result = dist.all_gather(quantized_output, quantized_source, group=group)
            output[:] = _dequantize_tensor_list(quantized_output, qtype, quant_loss)
            return result
        if func is dist.all_to_all:
            output, source = args[:2]
            quantized_source = _quantize_tensor_list(source, qtype)
            quantized_output = _quantize_tensor_list(output, qtype)
            result = dist.all_to_all(quantized_output, quantized_source, group=group)
            output[:] = _dequantize_tensor_list(quantized_output, qtype, quant_loss)
            return result
        raise RuntimeError(f"quantization does not support {func!r}")

    return wrapper
