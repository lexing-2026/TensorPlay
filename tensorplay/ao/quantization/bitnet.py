"""BitNet b1.58 style ternary quantization built on native quantized ops.

The weight is quantized to ternary codes in {-1, 0, 1} scaled by the tensor
absmean. Codes are stored packed at two bits per element (four per byte) using
the same bit layout as the reference BitNet packer, so the packed buffers are
interchangeable. Activations are quantized row-wise to signed 8-bit codes. The
matmul runs through the native signed integer linear path (int8 tensor cores on
accelerators) and is rescaled by the row and tensor scales.

Training uses a straight-through estimator: the forward executes the quantized
recipe exactly, while the backward passes gradients to the float weight and
activation as if the round/clamp steps were the identity (scaled so the
gradient magnitudes match the integer matmul, following the reference BitNet
training recipe). This makes ``BitNetLinear`` trainable with any optimizer.
"""

import math
from typing import Optional, Tuple

import tensorplay as tp
from tensorplay import nn, _C
from tensorplay.ao.nn.quantized.packed import PackedWeight
from tensorplay.autograd import Function

__all__ = [
    "get_bitnet_scale",
    "quantize_bitnet_weight",
    "pack_ternary",
    "unpack_ternary",
    "quantize_int8_rowwise",
    "bitnet_linear",
    "bitnet_linear_packed",
    "BitNetLinear",
]


def get_bitnet_scale(weight: tp.Tensor, eps: float = 1e-5) -> tp.Tensor:
    """Return the per-tensor absmean scale used by BitNet b1.58.

    The scale is ``mean(|W|)`` clamped away from zero so the reciprocal is
    finite for all-zero weights.
    """
    return weight.abs().mean().clamp(min=eps)


def quantize_bitnet_weight(
    weight: tp.Tensor,
    scale: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Quantize a float weight to ternary codes and an absmean scale.

    Returns ``(codes, scale)`` where ``codes`` has values in {-1, 0, 1} stored
    as int8 and ``scale`` is a rank-0 float32 tensor.
    """
    scale = get_bitnet_scale(weight, eps) if scale is None else scale.clamp(min=eps)
    codes = (weight.float() / scale).round().clamp(-1, 1).to(tp.int8)
    return codes, scale


def pack_ternary(codes: tp.Tensor) -> tp.Tensor:
    """Pack int8 ternary codes {-1, 0, 1} into uint8 storage, four per byte.

    The last dimension is padded with zeros to a multiple of four. Each 2-bit
    field holds ``code & 0b11`` (``-1`` -> ``0b11``, ``0`` -> ``0``,
    ``1`` -> ``0b01``), with the first element of each group occupying the two
    most significant bits, matching the reference BitNet packer.
    """
    x = codes.to(tp.int8)
    columns = x.shape[-1]
    padded = columns
    if columns % 4:
        padded = columns + (4 - columns % 4)
        pad = padded - columns
        x = tp.cat([x, tp.zeros((*x.shape[:-1], pad), dtype=tp.int8, device=x.device)], dim=-1)
    x = x.to(tp.int32)
    high = (x[..., 0::4] & 0b11) << 6
    upper_mid = (x[..., 1::4] & 0b11) << 4
    lower_mid = (x[..., 2::4] & 0b11) << 2
    low = x[..., 3::4] & 0b11
    packed = (high | upper_mid | lower_mid | low).to(tp.uint8)
    return packed.reshape(*x.shape[:-1], (padded + 3) // 4)


def unpack_ternary(packed: tp.Tensor, columns: int) -> tp.Tensor:
    """Unpack uint8 2-bit fields back into int8 ternary codes {-1, 0, 1}.

    ``columns`` is the original number of logical elements along the packed
    dimension; trailing padded columns are dropped.
    """
    x = packed.to(tp.int8)
    fields = tp.stack([x >> 6, (x << 2) >> 6, (x << 4) >> 6, (x << 6) >> 6], dim=-1)
    codes = fields.reshape(*packed.shape[:-1], -1)
    if codes.shape[-1] > columns:
        codes = codes[..., :columns]
    return codes


def quantize_int8_rowwise(
    input: tp.Tensor, eps: float = 1e-5
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Quantize the trailing dimension of ``input`` to int8 with row scales.

    For each row ``x`` the scale is ``max(|x|) / 127`` and the codes are
    ``round(x / scale)`` clamped to [-127, 127]. Returns ``(codes, scales)``
    with ``codes`` int8 and ``scales`` the per-row float32 scale.
    """
    absmax = input.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    row_scale = absmax / 127.0
    codes = (input.float() / row_scale).round().clamp(-127, 127).to(tp.int8)
    return codes, row_scale


class _BitNetLinearFunction(Function):
    """Straight-through estimator for the BitNet b1.58 linear.

    The forward follows :func:`bitnet_linear` exactly: row-wise int8
    activation codes, ternary weight codes, a native signed integer matmul,
    and rescaling by the row and tensor scales. The backward treats the
    quantizers as the identity:

        d loss / d input  = (grad_output @ codes) * tensor_scale
        d loss / d weight = grad_output.T @ (codes_x * row_scale)
        d loss / d bias   = grad_output.sum(0)

    The row and tensor scales cancel with the quantizer derivatives so the
    gradients have the same magnitude as the integer matmul, matching the
    reference BitNet training recipe.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, scale, eps):
        batch_dims = input.shape[:-1]
        x = input.reshape(-1, weight.shape[1])
        x_i8, row_scale = quantize_int8_rowwise(x, eps)
        codes, tensor_scale = quantize_bitnet_weight(weight, scale, eps)
        ctx.save_for_backward(x_i8, row_scale, codes, tensor_scale)
        if x.device.type == 'cuda':
            # Native int8 tensor-core path: no re-packing needed per step.
            out = tp.ops.tp._int_mm(x_i8, codes.t().contiguous())
        else:
            # Native SIMD quantized linear path for CPU.
            x_q = _C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
            w_q = _C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
            packed_weight = PackedWeight(w_q, None)
            out = packed_weight.run(x_q, 1.0, 0, output_float=True)
        out = out * row_scale * tensor_scale
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_dims, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x_i8, row_scale, codes, tensor_scale = ctx.saved_tensors
        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.reshape(-1, codes.shape[0])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output @ codes.to(grad_output.dtype)) * tensor_scale
            grad_input = grad_input.reshape(*batch_dims, codes.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ (x_i8 * row_scale)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None


def bitnet_linear(
    input: tp.Tensor,
    weight: tp.Tensor,
    bias: Optional[tp.Tensor] = None,
    scale: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> tp.Tensor:
    """BitNet b1.58 linear: ternary weight times int8 row-wise activations.

    ``weight`` has shape ``(out_features, in_features)``. The forward computes

        y = round_clip(x / row_absmax * 127) @ round_clip(W / mean(|W|))
            * row_scale * weight_scale + bias

    using the native signed integer linear path for the matmul, and the
    straight-through estimator provides gradients for training. The returned
    tensor has the float dtype of ``input``.
    """
    return _BitNetLinearFunction.apply(input, weight, bias, scale, eps)


def bitnet_linear_packed(
    input: tp.Tensor,
    packed: tp.Tensor,
    scale: tp.Tensor,
    in_features: int,
    bias: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> tp.Tensor:
    """BitNet b1.58 linear from pre-packed ternary weights.

    ``packed`` is the output of :func:`pack_ternary` (uint8, four codes per
    byte), ``scale`` is the absmean tensor scale and ``in_features`` the number
    of input channels. This is the inference path: the weight quantization and
    packing happen once ahead of time.
    """
    batch_dims = input.shape[:-1]
    x = input.reshape(-1, in_features)
    x_i8, row_scale = quantize_int8_rowwise(x, eps)
    codes = unpack_ternary(packed, in_features)
    x_q = _C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
    w_q = _C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
    packed_weight = PackedWeight(w_q, None)
    out = packed_weight.run(x_q, 1.0, 0, output_float=True)
    out = out * row_scale * scale
    if bias is not None:
        out = out + bias
    return out.reshape(*batch_dims, packed.shape[0])


class BitNetLinear(nn.Module):
    """Linear layer with BitNet b1.58 ternary weights and int8 activations.

    The weight parameter stays in float; every forward quantizes it to ternary
    codes with the absmean scale and quantizes the activation row-wise to int8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(tp.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(tp.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return bitnet_linear(input, self.weight, self.bias)