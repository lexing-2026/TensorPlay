"""INT8 dynamic-activation / INT2 weight linear built on native ops.

The forward dynamically quantizes activations per row to int8 and the
weight per row to int2 (signed codes in [-2, 1]), then runs the product
through the native signed integer matmul path (int8 tensor cores on CUDA,
SIMD quantized linear on CPU).  The result is rescaled by the per-row
activation and weight scales.  A straight-through estimator keeps the
full-precision weight trainable.
"""

from dataclasses import dataclass

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.bitnet import quantize_int8_rowwise
from tensorplay.ao.quantization.int8_mixed_precision import _int8_mm
from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Int8DynamicActivationInt2WeightConfig",
    "int8_dynamic_activation_int2_weight",
    "Int8DynActInt2WeightLinear",
]


def _quantize_int2_weight_signed(weight: tp.Tensor):
    """Row-wise symmetric int2 quantization to signed codes.

    Returns ``(codes, scale)`` with ``codes`` of shape ``(N, K)`` in
    ``[-2, 1]`` and per-row ``scale`` of shape ``(N,)``.
    """
    return tp._C._quantize_int2_rowwise(weight, 1e-8)


class _Int8DynActInt2WeightLinear(Function):
    """Straight-through estimator for int8-dynamic-activation / int2 weight."""

    @staticmethod
    def forward(ctx, input, weight, groupsize):
        del groupsize  # per-row weight scale; kept for config compatibility
        flat = input.reshape(-1, input.shape[-1])
        x_i8, x_scale = quantize_int8_rowwise(flat)
        w_codes, w_scale = _quantize_int2_weight_signed(weight)
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
        return grad_input, grad_weight, None


class Int8DynActInt2WeightLinear(nn.Linear):
    """Linear module with int8 dynamic activations and int2 weights."""

    def __init__(self, *args, groupsize: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int8DynActInt2WeightLinear.apply(
            input, self.weight, self.groupsize
        )


@dataclass
class Int8DynamicActivationInt2WeightConfig:
    """Configuration for INT8 dynamic-activation / INT2 weight quantization.

    Attributes:
        groupsize: kept for workflow compatibility; the weight scale is
            per-row in this kernel path.
    """

    groupsize: int = 128


# Backwards-compatible alias matching the workflow config name.
int8_dynamic_activation_int2_weight = Int8DynamicActivationInt2WeightConfig


@register_quantize_module_handler(Int8DynamicActivationInt2WeightConfig)
def _int8_dyn_act_int2_transform(module, config):
    """Replace an ``nn.Linear`` module with an 8da2w linear."""
    module.__class__ = Int8DynActInt2WeightLinear
    module.groupsize = config.groupsize
    return module