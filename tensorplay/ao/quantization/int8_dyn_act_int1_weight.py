"""INT8 dynamic-activation / 1-bit weight linear built on native ops.

The forward dynamically quantizes activations per row to int8 and the
weight per row to 1-bit sign codes {-1, +1} with a per-row absmax scale,
then runs the product through the native signed integer matmul path (int8
tensor cores on CUDA, SIMD quantized linear on CPU).  A straight-through
estimator keeps the full-precision weight trainable.
"""

from dataclasses import dataclass

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.bitnet import quantize_int8_rowwise
from tensorplay.ao.quantization.int8_mixed_precision import _int8_mm
from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Int8DynamicActivationBinaryWeightConfig",
    "int8_dynamic_activation_binary_weight",
    "Int8DynActBinaryWeightLinear",
]


def _quantize_binary_weight_signed(weight: tp.Tensor):
    """Row-wise 1-bit quantization to strict sign codes.

    Returns ``(codes, scale)`` with ``codes`` of shape ``(N, K)`` in
    ``{-1, +1}`` and per-row ``scale`` of shape ``(N,)`` equal to the row
    absmax.
    """
    return tp._C._quantize_binary_rowwise(weight, 1e-8)


class _Int8DynActBinaryWeightLinear(Function):
    """Straight-through estimator for int8-dynamic-activation / 1-bit weight."""

    @staticmethod
    def forward(ctx, input, weight):
        flat = input.reshape(-1, input.shape[-1])
        x_i8, x_scale = quantize_int8_rowwise(flat)
        w_codes, w_scale = _quantize_binary_weight_signed(weight)
        ctx.save_for_backward(input, weight)
        out = _int8_mm(x_i8, w_codes, x_scale, w_scale)
        return out.reshape(*input.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_input = grad_output @ weight.to(grad_output.dtype)
        return grad_input, grad_weight


class Int8DynActBinaryWeightLinear(nn.Linear):
    """Linear module with int8 dynamic activations and 1-bit weights."""

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int8DynActBinaryWeightLinear.apply(input, self.weight)


@dataclass
class Int8DynamicActivationBinaryWeightConfig:
    """Configuration for INT8 dynamic-activation / 1-bit weight quantization."""


# Backwards-compatible alias matching the workflow config name.
int8_dynamic_activation_binary_weight = Int8DynamicActivationBinaryWeightConfig


@register_quantize_module_handler(Int8DynamicActivationBinaryWeightConfig)
def _int8_dyn_act_binary_transform(module, config):
    """Replace an ``nn.Linear`` module with an 8da1w linear."""
    module.__class__ = Int8DynActBinaryWeightLinear
    return module