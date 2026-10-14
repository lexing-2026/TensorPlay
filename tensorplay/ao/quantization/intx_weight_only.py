"""Generic n-bit weight-only quantization (1 <= bits <= 8).

This is the framework-facing counterpart of the generic low-bit config in
upstream low-bit tooling: it quantizes weights to ``bits`` levels per
element with per-group affine scales and zeros, packs the codes into uint8
storage (``8 // bits`` codes per byte), and provides inference and
straight-through-estimator training linear modules.

* :class:`WeightOnlyIntxLinear` stores the packed n-bit weight and per-group
  scales/zeros and is produced by ``quantize_(model, intx_weight_only(...))``.
* :class:`IntxWeightOnlyQuantizedTrainingLinear` keeps a full-precision
  weight, quantizes on every forward and uses a straight-through estimator.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "IntxWeightOnlyConfig",
    "intx_weight_only",
    "quantize_intx_weight_only",
    "pack_intx",
    "unpack_intx",
    "WeightOnlyIntxLinear",
    "IntxWeightOnlyQuantizedTrainingLinear",
]


def pack_intx(codes: tp.Tensor, bits: int) -> tp.Tensor:
    """Pack int8 codes in ``[0, 2**bits - 1]`` into uint8 storage.

    ``8 // bits`` codes share one storage byte; the first code of each group
    occupies the most significant bits (matching the packed 2-bit layout).
    The last dimension is padded with zeros to a multiple of ``8 // bits``.
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    per_byte = 8 // bits
    x = codes.to(tp.int32)
    columns = x.shape[-1]
    padded = columns
    if columns % per_byte:
        padded = columns + (per_byte - columns % per_byte)
        pad = padded - columns
        x = tp.cat(
            [x, tp.zeros((*x.shape[:-1], pad), dtype=tp.int32, device=x.device)],
            dim=-1,
        )
    packed = tp.zeros((*x.shape[:-1], padded // per_byte), dtype=tp.uint8, device=x.device)
    for i in range(per_byte):
        shift = 8 - (i + 1) * bits
        packed = packed | ((x[..., i::per_byte] & ((1 << bits) - 1)) << shift)
    return packed.to(tp.uint8)


def unpack_intx(packed: tp.Tensor, bits: int, columns: int) -> tp.Tensor:
    """Unpack uint8 n-bit fields back into int8 codes ``[0, 2**bits - 1]``.

    ``columns`` is the original number of logical elements along the packed
    dimension; trailing padded columns are dropped.
    """
    per_byte = 8 // bits
    x = packed.to(tp.uint8)
    codes = tp.zeros(
        (*packed.shape[:-1], packed.shape[-1] * per_byte),
        dtype=tp.uint8,
        device=packed.device,
    )
    for i in range(per_byte):
        shift = 8 - (i + 1) * bits
        codes[..., i::per_byte] = (x >> shift) & ((1 << bits) - 1)
    if codes.shape[-1] > columns:
        codes = codes[..., :columns]
    return codes.to(tp.int8)


def _quantize_intx_groupwise(
    weight: tp.Tensor, groupsize: int, bits: int, eps: float = 1e-8
) -> Tuple[tp.Tensor, tp.Tensor, tp.Tensor]:
    """Affine group-wise n-bit quantization of a 2-D weight.

    Returns ``(codes, scale, zero)`` with ``codes`` an int8 tensor holding
    levels in ``[0, 2**bits - 1]``, ``scale``/``zero`` of shape
    ``(N, K/groupsize)``.  Each group maps ``min`` to code 0 and ``max`` to
    code ``2**bits - 1``.
    """
    N, K = weight.shape
    if K % groupsize != 0:
        raise ValueError(
            f"intx weight-only requires K={K} divisible by groupsize={groupsize}"
        )
    groups = K // groupsize
    w = weight.reshape(N, groups, groupsize)
    min_vals = w.min(dim=-1).values
    max_vals = w.max(dim=-1).values
    scale = (max_vals - min_vals) / float((1 << bits) - 1)
    scale = scale.clamp(min=eps)
    zero = (-min_vals / scale).round().clamp(0, (1 << bits) - 1)
    codes = ((w - min_vals.unsqueeze(-1)) / scale.unsqueeze(-1)).round()
    codes = codes.clamp(0, (1 << bits) - 1).to(tp.int8).reshape(N, K)
    return codes, scale, zero


def _dequantize_intx(
    packed: tp.Tensor, scale: tp.Tensor, zero: tp.Tensor, bits: int, columns: int
) -> tp.Tensor:
    """Dequantize packed n-bit codes back to float32.

    The group scale/zero are broadcast against a 3-D view of the unpacked
    codes instead of being repeated to the full matrix, which avoids two
    large intermediate copies per forward.
    """
    groupsize = columns // scale.shape[-1]
    codes = unpack_intx(packed, bits, columns).to(tp.float32)
    codes = codes.reshape(*codes.shape[:-1], scale.shape[-1], groupsize)
    return ((codes - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(
        *codes.shape[:-2], -1
    )


def _intx_mm(
    input: tp.Tensor, packed: tp.Tensor, scale: tp.Tensor, zero: tp.Tensor,
    bits: int, columns: int,
) -> tp.Tensor:
    """n-bit matmul: unpack, dequantize, run a float GEMM."""
    dq = _dequantize_intx(packed, scale, zero, bits, columns)
    return input @ dq.T.contiguous()


class WeightOnlyIntxLinear(nn.Module):
    """Inference linear with a packed n-bit weight and per-group scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 2,
        groupsize: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        per_byte = 8 // bits
        self.register_buffer(
            "weight",
            tp.zeros((out_features, (in_features + per_byte - 1) // per_byte), dtype=tp.uint8),
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
        out = _intx_mm(input, self.weight, scale, zero, self.bits, self.in_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(
        cls, linear: nn.Linear, bits: int = 2, groupsize: int = 128
    ) -> "WeightOnlyIntxLinear":
        """Build an n-bit weight-only linear from a float ``nn.Linear``."""
        if linear.bias is not None and linear.bias.dim() != 1:
            raise ValueError("intx weight-only requires a 1-D bias or no bias")
        codes, scale, zero = _quantize_intx_groupwise(
            linear.weight.detach(), groupsize, bits
        )
        mod = cls(
            linear.in_features,
            linear.out_features,
            bits=bits,
            groupsize=groupsize,
            bias=linear.bias is not None,
        )
        mod.weight = pack_intx(codes, bits)
        mod.scales_and_zeros = tp.stack([scale.t(), zero.t()], dim=-1)
        if linear.bias is not None:
            mod.bias = linear.bias.detach()
        return mod


class _IntxWeightOnlyLinear(Function):
    """Straight-through estimator for n-bit weight-only training."""

    @staticmethod
    def forward(ctx, input, weight, groupsize, bits):
        codes, scale, zero = _quantize_intx_groupwise(weight, groupsize, bits)
        codes_f = codes.to(tp.float32).reshape(
            *codes.shape[:-1], scale.shape[-1], groupsize
        )
        dq = ((codes_f - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(
            *codes.shape
        )
        ctx.save_for_backward(input, weight)
        out = input @ dq.T.contiguous()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(
            -1, weight.shape[1]
        )
        grad_input = grad_output @ weight.to(grad_output.dtype)
        return grad_input, grad_weight, None, None


class IntxWeightOnlyQuantizedTrainingLinear(nn.Linear):
    """Linear module with n-bit weight-only quantization and STE training."""

    def __init__(self, *args, bits: int = 2, groupsize: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.groupsize = groupsize

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _IntxWeightOnlyLinear.apply(input, self.weight, self.groupsize, self.bits)


@dataclass
class IntxWeightOnlyConfig:
    """Configuration for n-bit weight-only quantization.

    Attributes:
        bits: number of bits per weight element, in ``[1, 8]``.
        groupsize: number of input columns sharing one scale/zero.
    """

    bits: int = 2
    groupsize: int = 128

    def __post_init__(self):
        if not (1 <= self.bits <= 8):
            raise ValueError(f"bits must be in [1, 8], got {self.bits}")
        if self.groupsize <= 0:
            raise ValueError(f"groupsize must be positive, got {self.groupsize}")


# Backwards-compatible alias matching the workflow config name.
intx_weight_only = IntxWeightOnlyConfig


@register_quantize_module_handler(IntxWeightOnlyConfig)
def _intx_weight_only_transform(module, config):
    """Replace an ``nn.Linear`` module with a packed n-bit weight-only linear."""
    return WeightOnlyIntxLinear.from_float(module, config.bits, config.groupsize)


def quantize_intx_weight_only(
    model: "tp.nn.Module", bits: int = 2, groupsize: int = 128
) -> "tp.nn.Module":
    """Convert every ``nn.Linear`` in ``model`` to n-bit weight-only."""
    from tensorplay.ao.quantization.quantize_api import quantize_

    return quantize_(model, intx_weight_only(bits=bits, groupsize=groupsize))
