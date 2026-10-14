"""Quantized activation modules.

Each module computes its activation on the dequantized values of an affine
Int8 tensor and requantizes into declared output qparams, except relu-style
ops whose grid is order-preserving, so the output inherits the input qparams.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay import _C
from tensorplay._C import (
    quantized_elu as _quantized_elu,
    quantized_hardswish as _quantized_hardswish,
    quantized_hardsigmoid as _quantized_hardsigmoid,
    quantized_leaky_relu as _quantized_leaky_relu,
    quantized_prelu as _quantized_prelu,
    quantized_relu as _quantized_relu,
    quantized_relu6 as _quantized_relu6,
    quantized_sigmoid as _quantized_sigmoid,
    quantized_softmax as _quantized_softmax,
    quantized_tanh as _quantized_tanh,
)

__all__ = [
    "ReLU",
    "ReLU6",
    "ELU",
    "LeakyReLU",
    "Hardswish",
    "Hardsigmoid",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "PReLU",
]


def _activation_qparams(mod, scale, zero_point):
    """Resolve (scale, zero_point) from explicit values or a calibrated
    ``activation_post_process`` attribute on ``mod``."""
    if scale is not None and zero_point is not None:
        return float(scale), int(zero_point)
    post_process = getattr(mod, "activation_post_process", None)
    if post_process is None:
        raise ValueError(
            f"{type(mod).__name__} requires scale and zero_point, or a "
            "calibrated activation_post_process")
    s, z = post_process.calculate_qparams()
    return float(s), int(z)


class ReLU(nn.ReLU):
    """Rectified linear unit on a quantized tensor.

    The output carries the input scale and zero point: the negative
    half-space of the affine grid is exactly the code segment below the zero
    point, so the op is an integer maximum.
    """

    def forward(self, input):
        return _quantized_relu(input)

    def _get_name(self):
        return "QuantizedReLU"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return ReLU(mod.inplace)


class ReLU6(nn.ReLU6):
    """Rectified linear unit clamped at 6 on a quantized tensor.

    The output carries the input scale and zero point; the upper bound is the
    grid position of the real value 6.
    """

    def forward(self, input):
        return _quantized_relu6(input)

    def _get_name(self):
        return "QuantizedReLU6"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return ReLU6(mod.inplace)


class ELU(nn.ELU):
    """Exponential linear unit on a quantized tensor.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """

    def __init__(self, scale, zero_point, alpha=1.0):
        super().__init__(alpha)
        self.scale = scale
        self.zero_point = zero_point
        self.alpha = alpha

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_elu(
            input, scale, zero_point, alpha=self.alpha)

    def _get_name(self):
        return "QuantizedELU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.alpha)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.alpha)


class LeakyReLU(nn.LeakyReLU):
    """Leaky rectified linear unit on a quantized tensor.

    Values below zero are multiplied by ``negative_slope``; the result is
    requantized into the declared output qparams.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: slope applied to negative values
        inplace: accepted for interface compatibility; the computation is
            always out-of-place
    """

    def __init__(self, scale, zero_point, negative_slope=1e-2, inplace=False):
        super().__init__(negative_slope, inplace)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_leaky_relu(input, self.negative_slope, scale, zero_point)

    def _get_name(self):
        return "QuantizedLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)


class Hardswish(nn.Hardswish):
    """HardSwish activation on a quantized tensor.

    Computes ``x * clamp(x + 3, 0, 6) / 6`` in the dequantized domain and
    requantizes into the declared output qparams.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_hardswish(input, scale, zero_point)

    def _get_name(self):
        return "QuantizedHardswish"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class Hardsigmoid(nn.Hardsigmoid):
    """HardSigmoid activation on a quantized tensor.

    Computes ``clamp(x / 6 + 1 / 2, 0, 1)`` in the dequantized domain and
    requantizes into the declared output qparams.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_hardsigmoid(input, scale, zero_point)

    def _get_name(self):
        return "QuantizedHardSigmoid"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class Sigmoid(nn.Sigmoid):
    """Logistic sigmoid on a quantized tensor.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_sigmoid(input, scale, zero_point)

    def _get_name(self):
        return "QuantizedSigmoid"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class Tanh(nn.Tanh):
    """Hyperbolic tangent on a quantized tensor.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        scale, zero_point = _activation_qparams(self, self.scale, self.zero_point)
        return _quantized_tanh(input, scale, zero_point)

    def _get_name(self):
        return "QuantizedTanh"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))


class Softmax(nn.Softmax):
    """Softmax on a quantized tensor, requantized to declared qparams."""

    def __init__(self, dim=None, scale=1.0, zero_point=0):
        super().__init__(dim=dim)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        dim = self.dim
        if dim is None:
            dim = input.dim() - 1
        return _quantized_softmax(input, dim, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedSoftmax"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = _activation_qparams(mod, None, None)
        return cls(mod.dim, scale, zero_point)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.dim, float(scale), int(zero_point))


class PReLU(nn.Module):
    """Parametric rectifier on a quantized tensor with a quantized slope."""

    def __init__(self, output_scale, output_zero_point, num_parameters=1):
        super().__init__()
        self.num_parameters = num_parameters
        self.scale = output_scale
        self.zero_point = output_zero_point
        w = tensorplay.randn(num_parameters, dtype=tensorplay.float32)
        qw = tensorplay.quantize_per_tensor(w, 1.0, 0, tensorplay.quint8)
        self.set_weight(qw)

    def set_weight(self, w):
        self.weight = w

    def forward(self, input):
        return _quantized_prelu(input, self.weight, self.scale, self.zero_point)

    def _get_name(self):
        return "QuantizedPReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = _activation_qparams(mod, None, None)
        result = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = tensorplay.quantize_per_tensor(float_wt, float(wt_scale), int(wt_zp), tensorplay.quint8)
        result.set_weight(qweight)
        return result

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        result = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = tensorplay.quantize_per_tensor(float_wt, float(wt_scale), int(wt_zp), tensorplay.quint8)
        result.set_weight(qweight)
        return result


class MultiheadAttention(nn.Module):
    """Quantized multihead attention converted from an observed module.

    The projection layers are native quantized linear modules; the attention
    output is requantized with the module's declared output qparams.
    """

    _FLOAT_MODULE = None

    def __init__(self, linear_Q, linear_K, linear_V, out_proj, scale, zero_point,
                 batch_first=False):
        super().__init__()
        self.linear_Q = linear_Q
        self.linear_K = linear_K
        self.linear_V = linear_V
        self.out_proj = out_proj
        self.scale = scale
        self.zero_point = zero_point
        self.embed_dim = linear_Q.out_features
        self.num_heads = getattr(linear_Q, "_num_heads", 1)
        self.batch_first = batch_first

    def _get_name(self):
        return "QuantizedMultiheadAttention"

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if is_causal:
            raise AssertionError("causal mask not supported by AO MHA module")
        if self.training:
            raise AssertionError("quantized MHA can only be used in eval mode")

        def run_linear(linear, x):
            if not x.is_quantized():
                x = _C.quantize_per_tensor(x, 1.0, 0, tensorplay.qint8)
            return linear(x).dequantize()

        q = run_linear(self.linear_Q, query)
        k = run_linear(self.linear_K, key)
        v = run_linear(self.linear_V, value)
        # Normalize to (N, L, E).
        if not self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        head_dim = self.embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5
        # Reshape to (N, H, L, D).
        q = q.reshape(q.size(0), q.size(1), self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(k.size(0), k.size(1), self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(v.size(0), v.size(1), self.num_heads, head_dim).transpose(1, 2)
        attn = tensorplay.matmul(q, k.transpose(-2, -1)) * scaling
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = tensorplay.softmax(attn, dim=-1)
        attn_output = tensorplay.matmul(attn, v).transpose(1, 2).reshape(q.size(0), q.size(2), self.embed_dim)
        from tensorplay.nn import functional as F
        out_w = self.out_proj.weight().dequantize() if self.out_proj.weight().is_quantized() else self.out_proj.weight()
        attn_output = F.linear(attn_output, out_w, self.out_proj.bias)
        attn_output = _C.quantize_per_tensor(attn_output, self.scale, self.zero_point, tensorplay.qint8)
        if need_weights:
            attn_weights = attn.mean(dim=1) if average_attn_weights else attn
            return attn_output, attn_weights
        return attn_output, None

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        raise NotImplementedError(
            "It looks like you are trying to convert a non-observed MHA module. "
            "Please use the quantizable MHA with the eager quantization workflow."
        )

    @classmethod
    def from_observed(cls, other):
        from tensorplay.ao.quantization import convert

        converted = convert(other, inplace=False, remove_qconfig=True)
        post = getattr(other, "activation_post_process", None)
        if post is None:
            post = getattr(other.out_proj, "activation_post_process", None)
        if post is None:
            scale, zero_point = 1.0, 0
        else:
            scale, zero_point = post.calculate_qparams()
        result = cls(converted.linear_Q, converted.linear_K, converted.linear_V,
                     converted.out_proj, float(scale), int(zero_point),
                     batch_first=getattr(other, "batch_first", False))
        result.eval()
        return result
