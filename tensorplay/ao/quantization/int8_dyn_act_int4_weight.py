"""INT8 dynamic-activation / INT4 weight linear built on native ops.

The forward dynamically quantizes activations per token to int8 and the
weight per group to int4 (signed codes in [-8, 7]), then runs the product
through the native signed integer matmul path (int8 tensor cores on CUDA,
SIMD quantized linear on CPU).  The result is rescaled by the per-token and
per-group scales, so the output matches a float linear much more closely
than an int4-only weight path while keeping the compute on tensor cores.
A straight-through estimator keeps the full-precision weight trainable.
"""

from dataclasses import dataclass
from typing import Optional

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.bitnet import quantize_int8_rowwise
from tensorplay.ao.quantization.int8_mixed_precision import _int8_mm
from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Int8DynamicActivationInt4WeightConfig",
    "int8_dynamic_activation_int4_weight",
    "Int8DynActInt4WeightLinear",
]


@dataclass
class Int8DynamicActivationInt4WeightConfig:
    """Configuration for INT8 dynamic-activation / INT4 weight quantization.

    Attributes:
        groupsize: number of input columns sharing one weight scale.
    """

    groupsize: int = 128


# Backwards-compatible alias matching the workflow config name.
int8_dynamic_activation_int4_weight = Int8DynamicActivationInt4WeightConfig


def _quantize_int4_weight_signed(weight: tp.Tensor):
    """Row-wise symmetric int4 quantization to signed codes.

    Returns ``(codes, scale)`` with ``codes`` of shape ``(N, K)`` in
    ``[-8, 7]`` and per-row ``scale`` of shape ``(N,)``.  A per-row scale
    keeps the product on the native integer matmul path (int8 tensor cores);
    group-wise scales would need a per-group split of the integer GEMM.
    """
    N, K = weight.shape
    scale = weight.abs().amax(dim=-1) / 7.0
    scale = scale.clamp(min=1e-8)
    codes = (weight / scale.unsqueeze(-1)).round().clamp(-8, 7)
    codes = codes.to(tp.int8).reshape(N, K)
    return codes, scale


class _Int8DynActInt4WeightLinear(Function):
    """Straight-through estimator for 8da4w."""

    @staticmethod
    def forward(ctx, input, weight, groupsize):
        del groupsize  # per-row weight scale; kept for config compatibility
        flat = input.reshape(-1, input.shape[-1])
        x_i8, x_scale = quantize_int8_rowwise(flat)
        w_codes, w_scale = _quantize_int4_weight_signed(weight)
        ctx.save_for_backward(input, weight)
        out = _int8_mm(
            x_i8.contiguous(),
            w_codes.contiguous(),
            x_scale.contiguous(),
            w_scale.contiguous(),
        )
        return out.reshape(*input.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_input = grad_output @ weight.to(grad_output.dtype)
        return grad_input, grad_weight, None


class Int8DynActInt4WeightLinear(nn.Linear):
    """Linear module with int8 dynamic activations and int4 weights."""

    def __init__(self, *args, groupsize: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int8DynActInt4WeightLinear.apply(
            input, self.weight, self.groupsize
        )


@register_quantize_module_handler(Int8DynamicActivationInt4WeightConfig)
def _int8_dyn_act_int4_transform(module, config):
    """Replace an ``nn.Linear`` module with an 8da4w linear."""
    module.__class__ = Int8DynActInt4WeightLinear
    module.groupsize = config.groupsize
    return module