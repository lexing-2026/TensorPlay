"""INT4 weight-only quantization built on native int4pack ops.

The native ``_convert_weight_to_int4pack`` family packs int4 codes two per
byte and ``_weight_int4pack_mm`` / ``_weight_int4pack_mm_for_cpu`` run the
group-quantized 4-bit matmul (CUDA tensor-core path and SIMD CPU path).
This module builds the ``int4_weight_only()`` workflow:

* :class:`WeightOnlyInt4Linear` stores the packed int4 weight and per-group
  scales/zeros and is produced by ``quantize_(model, int4_weight_only())``.
* :class:`Int4WeightOnlyQuantizedTrainingLinear` keeps a full-precision
  weight, packs on every forward and uses a straight-through estimator, so
  the int4 codes/scales follow the current weight while gradients flow to
  the full-precision parameter.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Int4WeightOnlyConfig",
    "Int4WeightOnlyQuantizedTrainingConfig",
    "int4_weight_only",
    "quantize_int4_weight_only",
    "WeightOnlyInt4Linear",
    "Int4WeightOnlyQuantizedLinear",
    "Int4WeightOnlyQuantizedTrainingLinear",
]


@dataclass
class Int4WeightOnlyConfig:
    """Configuration for INT4 weight-only quantization.

    Attributes:
        groupsize: number of input columns sharing one scale/zero.
    """

    groupsize: int = 128


@dataclass
class Int4WeightOnlyQuantizedTrainingConfig:
    """Configuration for INT4 weight-only quantization training.

    Attributes:
        groupsize: number of input columns sharing one scale/zero.
    """

    groupsize: int = 128


# Backwards-compatible alias matching the workflow config name.
int4_weight_only = Int4WeightOnlyConfig


def _quantize_int4_groupwise(
    weight: tp.Tensor, groupsize: int
) -> Tuple[tp.Tensor, tp.Tensor, tp.Tensor]:
    """Symmetric group-wise int4 quantization of a 2-D weight.

    Returns ``(codes, scale, zero)`` with ``codes`` of shape ``(N, K)`` in
    ``[0, 15]`` (signed code + 8), ``scale``/``zero`` of shape ``(N, K/g)``.
    """
    N, K = weight.shape
    if K % groupsize != 0:
        raise ValueError(
            f"int4 weight-only requires K={K} divisible by groupsize={groupsize}"
        )
    groups = K // groupsize
    w = weight.reshape(N, groups, groupsize)
    scale = w.abs().amax(dim=-1) / 8.0
    scale = scale.clamp(min=1e-8)
    codes = (w / scale.unsqueeze(-1)).round().clamp(-8, 7) + 8
    codes = codes.to(tp.int32).reshape(N, K)
    zero = tp.zeros_like(scale)
    return codes, scale, zero


def _pack_int4_weight(
    weight: tp.Tensor, groupsize: int
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Quantize and pack a float weight into ``(packed_uint8, scale_zero)``."""
    codes, scale, zero = _quantize_int4_groupwise(weight, groupsize)
    # Canonical (K/group, N, 2) layout shared by the CPU and CUDA int4pack
    # matmul ops.
    scale_zero = tp.stack([scale.t(), zero.t()], dim=-1)
    if weight.device.type == "cuda":
        packed = tp._C._convert_weight_to_int4pack(codes, 2)
    else:
        packed = tp._C._convert_weight_to_int4pack_for_cpu(codes, 2)
    return packed, scale_zero


def _int4_mm(
    input: tp.Tensor, packed: tp.Tensor, scale_zero: tp.Tensor, groupsize: int
) -> tp.Tensor:
    if input.device.type == "cuda":
        return tp._C._weight_int4pack_mm(input, packed, groupsize, scale_zero)
    return tp._C._weight_int4pack_mm_for_cpu(input, packed, groupsize, scale_zero)


class _Int4WeightOnlyLinear(Function):
    """Straight-through estimator for INT4 weight-only training."""

    @staticmethod
    def forward(ctx, input, weight, groupsize):
        codes, scale, zero = _quantize_int4_groupwise(weight, groupsize)
        scale_zero = tp.stack([scale.t(), zero.t()], dim=-1)
        packed = (
            tp._C._convert_weight_to_int4pack(codes, 2)
            if input.device.type == "cuda"
            else tp._C._convert_weight_to_int4pack_for_cpu(codes, 2)
        )
        ctx.save_for_backward(input, weight)
        ctx.groupsize = groupsize
        out = _int4_mm(input, packed, scale_zero, groupsize)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # STE: gradient flows to the full-precision weight.
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_input = grad_output @ weight.to(grad_output.dtype)
        return grad_input, grad_weight, None


class WeightOnlyInt4Linear(nn.Module):
    """Inference linear with a packed INT4 weight and per-group scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        groupsize: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.register_buffer("weight", tp.zeros((out_features, in_features // 2), dtype=tp.uint8))
        self.register_buffer(
            "scales_and_zeros",
            tp.zeros((in_features // groupsize, out_features, 2), dtype=tp.float32),
        )
        if bias:
            self.register_buffer("bias", tp.zeros(out_features, dtype=tp.float32))
        else:
            self.bias = None

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        act_pre_scale = getattr(self, "act_pre_scale", None)
        if act_pre_scale is not None:
            input = input * act_pre_scale
        out = _int4_mm(input, self.weight, self.scales_and_zeros, self.groupsize)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(cls, linear: nn.Linear, groupsize: int = 128) -> "WeightOnlyInt4Linear":
        if linear.bias is not None and linear.bias.dim() != 1:
            raise ValueError("int4 weight-only requires a 1-D bias or no bias")
        packed, scale_zero = _pack_int4_weight(linear.weight.detach(), groupsize)
        mod = cls(
            linear.in_features,
            linear.out_features,
            groupsize=groupsize,
            bias=linear.bias is not None,
        )
        mod.weight = packed
        mod.scales_and_zeros = scale_zero
        if linear.bias is not None:
            mod.bias = linear.bias.detach()
        return mod


class Int4WeightOnlyQuantizedTrainingLinear(nn.Linear):
    """Linear module with INT4 weight-only quantization and STE training."""

    def __init__(self, *args, groupsize: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int4WeightOnlyLinear.apply(input, self.weight, self.groupsize)


@register_quantize_module_handler(Int4WeightOnlyConfig)
def _int4_weight_only_transform(module, config):
    """Replace an ``nn.Linear`` module with a packed INT4 weight-only linear."""
    return WeightOnlyInt4Linear.from_float(module, config.groupsize)


@register_quantize_module_handler(Int4WeightOnlyQuantizedTrainingConfig)
def _int4_training_transform(module, config):
    """Replace an ``nn.Linear`` module with an INT4 weight-only training linear."""
    module.__class__ = Int4WeightOnlyQuantizedTrainingLinear
    module.groupsize = config.groupsize
    return module


# Compatibility alias: the inference module is the packed linear.
Int4WeightOnlyQuantizedLinear = WeightOnlyInt4Linear


def quantize_int4_weight_only(
    model: "tp.nn.Module", groupsize: int = 128
) -> "tp.nn.Module":
    """Convert every ``nn.Linear`` in ``model`` to INT4 weight-only."""
    from tensorplay.ao.quantization.quantize_api import quantize_

    return quantize_(model, int4_weight_only(groupsize=groupsize))

