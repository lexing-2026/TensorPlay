"""INT2 weight-only quantization built on manual 2-bit packing.

A 2-bit signed weight has four representable levels per element.  The codes
are quantized group-wise (``{-2, -1, 0, 1} * scale``) and packed two bits at
a time into uint8 storage, four codes per byte, so the quantized weight
occupies a quarter of the float memory.  This module builds the
``int2_weight_only()`` workflow:

* :class:`WeightOnlyInt2Linear` stores the packed 2-bit weight and per-group
  scales/zeros and is produced by ``quantize_(model, int2_weight_only())``.
* :class:`Int2WeightOnlyQuantizedTrainingLinear` keeps a full-precision
  weight, quantizes on every forward and uses a straight-through estimator,
  so the int2 codes/scales follow the current weight while gradients flow to
  the full-precision parameter.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "Int2WeightOnlyConfig",
    "Int2WeightOnlyQuantizedTrainingConfig",
    "int2_weight_only",
    "quantize_int2_weight_only",
    "pack_int2",
    "unpack_int2",
    "WeightOnlyInt2Linear",
    "Int2WeightOnlyQuantizedTrainingLinear",
]


def _quantize_int2_groupwise(
    weight: tp.Tensor, groupsize: int, eps: float = 1e-8
) -> Tuple[tp.Tensor, tp.Tensor, tp.Tensor]:
    """Symmetric group-wise int2 quantization of a 2-D weight.

    Returns ``(codes, scale, zero)`` with ``codes`` an int8 tensor of shape
    ``(N, K)`` holding the four levels {0, 1, 2, 3}, ``scale`` of shape
    ``(N, K/groupsize)`` and ``zero`` of shape ``(N, K/groupsize)``.  The
    representable values are ``(code - 2) * scale``, i.e. ``{-2, -1, 0, 1}``
    times the group scale, so ``scale = max(|w|) / 2``.
    """
    N, K = weight.shape
    if K % groupsize != 0:
        raise ValueError(
            f"int2 weight-only requires K={K} divisible by groupsize={groupsize}"
        )
    groups = K // groupsize
    w = weight.reshape(N, groups, groupsize)
    scale = w.abs().amax(dim=-1) / 2.0
    scale = scale.clamp(min=eps)
    codes = (w / scale.unsqueeze(-1)).round() + 2.0
    codes = codes.clamp(0, 3).to(tp.int8).reshape(N, K)
    zero = tp.full_like(scale, 2.0)
    return codes, scale, zero


def pack_int2(codes: tp.Tensor) -> tp.Tensor:
    """Pack int8 codes {0, 1, 2, 3} into uint8 storage, four per byte.

    The last dimension is padded with zeros to a multiple of four.  Each
    2-bit field holds ``code & 0b11``; the first element of each group
    occupies the two most significant bits, matching the packed 2-bit layout
    of the native ``quint2x4`` dtype.
    """
    x = codes.to(tp.int8)
    columns = x.shape[-1]
    padded = columns
    if columns % 4:
        padded = columns + (4 - columns % 4)
        pad = padded - columns
        x = tp.cat(
            [x, tp.zeros((*x.shape[:-1], pad), dtype=tp.int8, device=x.device)],
            dim=-1,
        )
    x = x.to(tp.int32)
    high = (x[..., 0::4] & 0b11) << 6
    upper_mid = (x[..., 1::4] & 0b11) << 4
    lower_mid = (x[..., 2::4] & 0b11) << 2
    low = x[..., 3::4] & 0b11
    packed = (high | upper_mid | lower_mid | low).to(tp.uint8)
    return packed.reshape(*x.shape[:-1], (padded + 3) // 4)


def unpack_int2(packed: tp.Tensor, columns: int) -> tp.Tensor:
    """Unpack uint8 2-bit fields back into int8 codes {0, 1, 2, 3}.

    ``columns`` is the original number of logical elements along the packed
    dimension; trailing padded columns are dropped.
    """
    x = packed.to(tp.uint8)
    fields = tp.stack([x >> 6, (x << 2) >> 6, (x << 4) >> 6, (x << 6) >> 6], dim=-1)
    codes = fields.reshape(*packed.shape[:-1], -1)
    if codes.shape[-1] > columns:
        codes = codes[..., :columns]
    return codes


def _dequantize_int2(
    packed: tp.Tensor, scale: tp.Tensor, zero: tp.Tensor, columns: int
) -> tp.Tensor:
    """Dequantize packed int2 codes back to float32."""
    groupsize = columns // scale.shape[-1]
    unpacked = unpack_int2(packed, columns).to(tp.float32)
    z = zero.repeat_interleave(groupsize, dim=-1)
    s = scale.repeat_interleave(groupsize, dim=-1)
    return (unpacked - z) * s


def _int2_mm(
    input: tp.Tensor, packed: tp.Tensor, scale: tp.Tensor, zero: tp.Tensor,
    columns: int,
) -> tp.Tensor:
    """INT2 matmul: unpack the packed weight, dequantize, run a float GEMM."""
    dq = _dequantize_int2(packed, scale, zero, columns)
    return input @ dq.T.contiguous()


class WeightOnlyInt2Linear(nn.Module):
    """Inference linear with a packed INT2 weight and per-group scales."""

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
        self.register_buffer(
            "weight", tp.zeros((out_features, (in_features + 3) // 4), dtype=tp.uint8)
        )
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
        scale = self.scales_and_zeros[..., 0].t().contiguous()
        zero = self.scales_and_zeros[..., 1].t().contiguous()
        out = _int2_mm(input, self.weight, scale, zero, self.in_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(cls, linear: nn.Linear, groupsize: int = 128) -> "WeightOnlyInt2Linear":
        """Build an INT2 weight-only linear from a float ``nn.Linear``."""
        if linear.bias is not None and linear.bias.dim() != 1:
            raise ValueError("int2 weight-only requires a 1-D bias or no bias")
        codes, scale, zero = _quantize_int2_groupwise(
            linear.weight.detach(), groupsize
        )
        mod = cls(
            linear.in_features,
            linear.out_features,
            groupsize=groupsize,
            bias=linear.bias is not None,
        )
        mod.weight = pack_int2(codes)
        mod.scales_and_zeros = tp.stack([scale.t(), zero.t()], dim=-1)
        if linear.bias is not None:
            mod.bias = linear.bias.detach()
        return mod


class _Int2WeightOnlyLinear(Function):
    """Straight-through estimator for INT2 weight-only training."""

    @staticmethod
    def forward(ctx, input, weight, groupsize):
        codes, scale, zero = _quantize_int2_groupwise(weight, groupsize)
        z = zero.repeat_interleave(groupsize, dim=-1)
        s = scale.repeat_interleave(groupsize, dim=-1)
        dq = (codes.to(tp.float32) - z) * s
        ctx.save_for_backward(input, weight)
        out = input @ dq.T.contiguous()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # STE: gradients flow to the full-precision weight.
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_input = grad_output @ weight.to(grad_output.dtype)
        return grad_input, grad_weight, None


class Int2WeightOnlyQuantizedTrainingLinear(nn.Linear):
    """Linear module with INT2 weight-only quantization and STE training."""

    def __init__(self, *args, groupsize: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int2WeightOnlyLinear.apply(input, self.weight, self.groupsize)


@dataclass
class Int2WeightOnlyConfig:
    """Configuration for INT2 weight-only quantization.

    Attributes:
        groupsize: number of input columns sharing one scale/zero.
    """

    groupsize: int = 128


@dataclass
class Int2WeightOnlyQuantizedTrainingConfig:
    """Configuration for INT2 weight-only quantization training.

    Attributes:
        groupsize: number of input columns sharing one scale/zero.
    """

    groupsize: int = 128


# Backwards-compatible alias matching the workflow config name.
int2_weight_only = Int2WeightOnlyConfig


@register_quantize_module_handler(Int2WeightOnlyConfig)
def _int2_weight_only_transform(module, config):
    """Replace an ``nn.Linear`` module with a packed INT2 weight-only linear."""
    return WeightOnlyInt2Linear.from_float(module, config.groupsize)


@register_quantize_module_handler(Int2WeightOnlyQuantizedTrainingConfig)
def _int2_training_transform(module, config):
    """Replace an ``nn.Linear`` module with an INT2 weight-only training linear."""
    module.__class__ = Int2WeightOnlyQuantizedTrainingLinear
    module.groupsize = config.groupsize
    return module


def quantize_int2_weight_only(
    model: "tp.nn.Module", groupsize: int = 128
) -> "tp.nn.Module":
    """Convert every ``nn.Linear`` in ``model`` to INT2 weight-only."""
    from tensorplay.ao.quantization.quantize_api import quantize_

    return quantize_(model, int2_weight_only(groupsize=groupsize))