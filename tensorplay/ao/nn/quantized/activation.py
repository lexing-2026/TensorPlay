"""Quantized activation modules.

Each module computes its activation on the dequantized values of an affine
Int8 tensor and requantizes into declared output qparams, except relu-style
ops whose grid is order-preserving, so the output inherits the input qparams.
"""

from __future__ import annotations

import tensorplay
from tensorplay import nn
from tensorplay._C import (
    quantized_elu as _quantized_elu,
    quantized_hardswish as _quantized_hardswish,
    quantized_hardsigmoid as _quantized_hardsigmoid,
    quantized_leaky_relu as _quantized_leaky_relu,
    quantized_relu as _quantized_relu,
    quantized_relu6 as _quantized_relu6,
    quantized_sigmoid as _quantized_sigmoid,
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
