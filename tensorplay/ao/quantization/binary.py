"""1-bit binary weight-only quantization for linear layers.

A binary weight stores one sign bit per element; the magnitude is captured by
a per-row scale equal to the row mean absolute value (the classic binary
weight-network scaling).  ``weight`` is recovered as ``sign * scale``.

* :class:`WeightOnlyBinaryLinear` stores the packed sign bits and per-row
  scales and is produced by ``quantize_(model, binary_weight_only())``.
* :class:`BinaryWeightOnlyQuantizedTrainingLinear` keeps a full-precision
  weight, binarizes on every forward and uses a straight-through estimator,
  so the sign bits follow the current weight while gradients flow to the
  full-precision parameter.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorplay as tp
from tensorplay import nn
from tensorplay.autograd import Function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler

__all__ = [
    "BinaryWeightOnlyConfig",
    "BinaryWeightOnlyQuantizedTrainingConfig",
    "binary_weight_only",
    "quantize_binary_weight_only",
    "pack_binary",
    "unpack_binary",
    "WeightOnlyBinaryLinear",
    "BinaryWeightOnlyQuantizedTrainingLinear",
]


def _quantize_binary_weight(
    weight: tp.Tensor, eps: float = 1e-8
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Binarize a 2-D weight to ``(sign_bits, scale)``.

    ``sign_bits`` is a ``uint8`` tensor with values in {0, 1} (0 -> -1,
    1 -> +1) of shape ``(N, K)`` and ``scale`` of shape ``(N, 1)`` holds the
    per-row mean absolute value.
    """
    scale = weight.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    sign_bits = (weight >= 0).to(tp.uint8)
    return sign_bits, scale


def pack_binary(codes: tp.Tensor) -> tp.Tensor:
    """Pack {0, 1} sign codes into uint8 storage, eight codes per byte.

    The trailing dimension is padded with zeros to a multiple of eight; code
    ``i`` of each group occupies bit ``i % 8`` (least significant first),
    matching the native packed 1-bit layout.
    """
    x = codes.to(tp.uint8)
    columns = x.shape[-1]
    padded = columns
    if columns % 8:
        padded = columns + (8 - columns % 8)
        pad = padded - columns
        x = tp.cat(
            [x, tp.zeros((*x.shape[:-1], pad), dtype=tp.uint8, device=x.device)],
            dim=-1,
        )
    fields = tp.stack(
        [((x[..., i::8] & 1) << i) for i in range(8)], dim=-1
    )
    # the eight 1-bit fields do not overlap within a byte, so OR == sum
    return fields.sum(dim=-1).to(tp.uint8)


def unpack_binary(packed: tp.Tensor, columns: int) -> tp.Tensor:
    """Unpack uint8 1-bit fields back into {0, 1} codes.

    ``columns`` is the original number of logical elements along the packed
    dimension; trailing padded columns are dropped.
    """
    x = packed.to(tp.uint8)
    fields = tp.stack([(x >> i) & 1 for i in range(8)], dim=-1)
    codes = fields.reshape(*packed.shape[:-1], -1)
    if codes.shape[-1] > columns:
        codes = codes[..., :columns]
    return codes


def _binary_mm(
    input: tp.Tensor, packed: tp.Tensor, scale: tp.Tensor, columns: int
) -> tp.Tensor:
    """1-bit matmul: unpack sign bits, scale per row, run a float GEMM."""
    sign_bits = unpack_binary(packed, columns).to(input.dtype)
    codes = sign_bits * 2.0 - 1.0  # {0, 1} -> {-1, +1}
    dq = codes * scale
    return input @ dq.T.contiguous()


class WeightOnlyBinaryLinear(nn.Module):
    """Inference linear with a packed 1-bit weight and per-row scales.

    Set ``cache_unpacked=True`` to keep the unpacked sign codes in a
    non-persistent buffer; this removes the per-forward unpacking cost at the
    price of a larger resident weight (one byte per element instead of one
    bit), which is a good trade for latency-sensitive inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        cache_unpacked: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cache_unpacked = cache_unpacked
        self.register_buffer(
            "weight", tp.zeros((out_features, (in_features + 7) // 8), dtype=tp.uint8)
        )
        self.register_buffer(
            "scale", tp.zeros((out_features, 1), dtype=tp.float32)
        )
        self.register_buffer("_unpacked_codes", None, persistent=False)
        if bias:
            self.register_buffer("bias", tp.zeros(out_features, dtype=tp.float32))
        else:
            self.bias = None

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        act_pre_scale = getattr(self, "act_pre_scale", None)
        if act_pre_scale is not None:
            input = input * act_pre_scale
        if self._unpacked_codes is not None:
            codes = self._unpacked_codes.to(input.dtype) * 2.0 - 1.0
            dq = codes * self.scale
            out = input @ dq.T.contiguous()
        else:
            out = _binary_mm(input, self.weight, self.scale, self.in_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(
        cls, linear: nn.Linear, cache_unpacked: bool = False
    ) -> "WeightOnlyBinaryLinear":
        """Build a 1-bit weight-only linear from a float ``nn.Linear``."""
        if linear.bias is not None and linear.bias.dim() != 1:
            raise ValueError("binary weight-only requires a 1-D bias or no bias")
        sign_bits, scale = _quantize_binary_weight(linear.weight.detach())
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            cache_unpacked=cache_unpacked,
        )
        mod.weight = pack_binary(sign_bits)
        mod.scale = scale
        if cache_unpacked:
            mod._unpacked_codes = sign_bits
        if linear.bias is not None:
            mod.bias = linear.bias.detach()
        return mod


class _BinaryWeightOnlyLinear(Function):
    """Straight-through estimator for 1-bit weight-only training."""

    @staticmethod
    def forward(ctx, input, weight, eps):
        sign_bits, scale = _quantize_binary_weight(weight, eps)
        codes = sign_bits.to(input.dtype) * 2.0 - 1.0
        dq = codes * scale
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


class BinaryWeightOnlyQuantizedTrainingLinear(nn.Linear):
    """Linear module with 1-bit weight-only quantization and STE training."""

    def __init__(self, *args, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _BinaryWeightOnlyLinear.apply(input, self.weight, self.eps)


@dataclass
class BinaryWeightOnlyConfig:
    """Configuration for 1-bit weight-only quantization."""


@dataclass
class BinaryWeightOnlyQuantizedTrainingConfig:
    """Configuration for 1-bit weight-only quantization training."""


# Backwards-compatible alias matching the workflow config name.
binary_weight_only = BinaryWeightOnlyConfig


@register_quantize_module_handler(BinaryWeightOnlyConfig)
def _binary_weight_only_transform(module, config):
    """Replace an ``nn.Linear`` module with a packed 1-bit weight-only linear."""
    cache = getattr(module, "_binary_cache_unpacked", False)
    return WeightOnlyBinaryLinear.from_float(module, cache_unpacked=cache)


@register_quantize_module_handler(BinaryWeightOnlyQuantizedTrainingConfig)
def _binary_training_transform(module, config):
    """Replace an ``nn.Linear`` module with a 1-bit weight-only training linear."""
    module.__class__ = BinaryWeightOnlyQuantizedTrainingLinear
    return module


def quantize_binary_weight_only(model: "tp.nn.Module") -> "tp.nn.Module":
    """Convert every ``nn.Linear`` in ``model`` to 1-bit weight-only."""
    from tensorplay.ao.quantization.quantize_api import quantize_

    return quantize_(model, binary_weight_only())