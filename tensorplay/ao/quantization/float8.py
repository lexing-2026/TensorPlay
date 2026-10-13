"""FP8 weight-only and mixed-precision training built on native FP8 ops.

The native ``quantize_affine_float8`` / ``dequantize_affine_float8`` /
``choose_scale_float8`` primitives provide per-block absmax FP8 quantization
on CPU and CUDA.  This module wraps them into trainable linear modules:

* :class:`Float8WeightOnlyQuantizedTrainingLinear` quantizes the weight to
  FP8 row-wise on every forward and runs the matmul in the input precision
  (straight-through estimator keeps the full-precision weight trainable).
* :class:`Float8MixedPrecisionTrainingLinear` additionally quantizes the
  activation per-token on every forward, giving FP8 mixed-precision training
  semantics while the optimizer keeps a full-precision parameter.

Both expose the same ``float8_weight_only()`` and
``float8_dynamic_activation_float8_weight()`` config entry points, built
entirely on native quantization ops.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Float8WeightOnlyConfig",
    "Float8DynamicActivationFloat8WeightConfig",
    "float8_weight_only",
    "float8_dynamic_activation_float8_weight",
    "Float8WeightOnlyQuantizedTrainingLinear",
    "Float8MixedPrecisionTrainingLinear",
]


@dataclass
class Float8WeightOnlyConfig:
    """Configuration for FP8 weight-only quantization.

    Attributes:
        dtype: one of ``tp.float8_e4m3fn`` / ``tp.float8_e4m3fnuz`` /
            ``tp.float8_e5m2`` / ``tp.float8_e5m2fnuz``.
    """

    dtype: str = "float8_e4m3fn"


@dataclass
class Float8DynamicActivationFloat8WeightConfig:
    """Configuration for FP8 dynamic-activation / FP8-weight training.

    Attributes:
        dtype: FP8 format used for both activations and weights.
    """

    dtype: str = "float8_e4m3fn"


# Backwards-compatible aliases matching the workflow config names.
float8_weight_only = Float8WeightOnlyConfig
float8_dynamic_activation_float8_weight = Float8DynamicActivationFloat8WeightConfig


def _fp8_dtype(config_or_str) -> "tp.DType":
    name = getattr(config_or_str, "dtype", config_or_str)
    return getattr(tp, name if isinstance(name, str) else name.name)


def _quantize_fp8_rowwise(
    tensor: tp.Tensor, fp8_dtype: "tp.DType", eps: float = 1e-12
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Absmax FP8 quantization with per-row scales.

    Returns ``(fp8_codes, scale)`` with ``scale`` of shape ``(M, 1)`` chosen so
    that ``tensor / scale`` saturates exactly at the FP8 format maximum.
    """
    flat = tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() > 2 else tensor
    if flat.dim() != 2:
        raise ValueError("_quantize_fp8_rowwise expects a 2-D tensor")
    K = flat.shape[1]
    scale = tp._C.choose_scale_float8(flat, [1, K], fp8_dtype)
    scale = scale.clamp(min=eps)
    codes = tp._C.quantize_affine_float8(flat, scale, fp8_dtype)
    return codes.reshape(tensor.shape), scale.reshape(list(tensor.shape[:-1]) + [1])


def _dequantize_fp8(
    codes: tp.Tensor, scale: tp.Tensor, output_dtype: "tp.DType"
) -> tp.Tensor:
    return tp._C.dequantize_affine_float8(codes, scale, output_dtype)


def _fp8_weight_only_linear(
    input: tp.Tensor, weight: tp.Tensor, bias: Optional[tp.Tensor], fp8_dtype
) -> tp.Tensor:
    """Weight-only FP8 matmul with straight-through estimator."""
    q, scale = _quantize_fp8_rowwise(weight.detach(), fp8_dtype)
    dq = _dequantize_fp8(q, scale, input.dtype)
    out = input @ dq.T.contiguous()
    if bias is not None:
        out = out + bias
    return out


class _Float8WeightOnlyLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, fp8_dtype):
        q, scale = _quantize_fp8_rowwise(weight, fp8_dtype)
        dq = _dequantize_fp8(q, scale, input.dtype)
        ctx.save_for_backward(input, weight, dq)
        ctx.bias = bias is not None
        out = input @ dq.T.contiguous()
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, dq = ctx.saved_tensors
        grad_input = grad_output @ dq.to(grad_output.dtype)
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias, None


class _Float8MixedPrecisionLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, fp8_dtype):
        # Weight: per-row FP8.
        q_w, s_w = _quantize_fp8_rowwise(weight, fp8_dtype)
        dq_w = _dequantize_fp8(q_w, s_w, input.dtype)
        # Activation: per-token FP8 (block = full row).
        flat = input.reshape(-1, input.shape[-1])
        q_a, s_a = _quantize_fp8_rowwise(flat, fp8_dtype)
        dq_a = _dequantize_fp8(q_a, s_a, input.dtype).reshape(input.shape)
        ctx.save_for_backward(input, weight, dq_w, s_w)
        ctx.bias = bias is not None
        out = dq_a @ dq_w.T.contiguous()
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, dq_w, s_w = ctx.saved_tensors
        # STE: activation gradient flows through the dequantized weight.
        grad_input = grad_output @ dq_w.to(grad_output.dtype)
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias, None


class Float8WeightOnlyQuantizedTrainingLinear(nn.Linear):
    """Linear module with FP8 weight-only quantization on every forward."""

    def __init__(self, *args, dtype: str = "float8_e4m3fn", **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_dtype = dtype

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Float8WeightOnlyLinear.apply(
            input, self.weight, self.bias, _fp8_dtype(self.fp8_dtype)
        )


class Float8MixedPrecisionTrainingLinear(nn.Linear):
    """Linear module with FP8 dynamic-activation / FP8-weight training."""

    def __init__(self, *args, dtype: str = "float8_e4m3fn", **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_dtype = dtype

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Float8MixedPrecisionLinear.apply(
            input, self.weight, self.bias, _fp8_dtype(self.fp8_dtype)
        )


@register_quantize_module_handler(Float8WeightOnlyConfig)
def _float8_weight_only_transform(module, config):
    """Replace matching ``nn.Linear`` modules with FP8 weight-only linear."""
    module.__class__ = Float8WeightOnlyQuantizedTrainingLinear
    module.fp8_dtype = config.dtype
    return module


@register_quantize_module_handler(Float8DynamicActivationFloat8WeightConfig)
def _float8_mixed_precision_transform(module, config):
    """Replace matching ``nn.Linear`` modules with FP8 mixed-precision linear."""
    module.__class__ = Float8MixedPrecisionTrainingLinear
    module.fp8_dtype = config.dtype
    return module