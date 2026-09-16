"""Functional quantized operators.

Each function forwards to its native operator, reading the operand qparams
that the tensor carries and taking the output qparams explicitly.
"""

from __future__ import annotations

from tensorplay._C import (
    q_scale as _q_scale,
    q_zero_point as _q_zero_point,
    quantized_add as _quantized_add,
    quantized_cat as _quantized_cat,
    quantized_cat_relu as _quantized_cat_relu,
    quantized_clamp as _quantized_clamp,
    quantized_div as _quantized_div,
    quantized_elu as _quantized_elu,
    quantized_hardswish as _quantized_hardswish,
    quantized_hardsigmoid as _quantized_hardsigmoid,
    quantized_leaky_relu as _quantized_leaky_relu,
    quantized_mul as _quantized_mul,
    quantized_relu as _quantized_relu,
    quantized_relu6 as _quantized_relu6,
    quantized_sigmoid as _quantized_sigmoid,
    quantized_sub as _quantized_sub,
    quantized_tanh as _quantized_tanh,
)

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "cat",
    "clamp",
    "relu",
    "relu6",
    "leaky_relu",
    "elu",
    "hardswish",
    "hardsigmoid",
    "sigmoid",
    "tanh",
]


def _qparams(x):
    if not x.is_quantized():
        raise TypeError(
            "quantized functional op(): operands must be quantized tensors")
    return _q_scale(x), _q_zero_point(x)


def add(a, b, scale, zero_point):
    """Quantized addition; output qparams given explicitly."""
    a_scale, a_zero_point = _qparams(a)
    b_scale, b_zero_point = _qparams(b)
    return _quantized_add(a, b, a_scale, a_zero_point, b_scale, b_zero_point,
                          scale, zero_point)


def sub(a, b, scale, zero_point):
    """Quantized subtraction; output qparams given explicitly."""
    a_scale, a_zero_point = _qparams(a)
    b_scale, b_zero_point = _qparams(b)
    return _quantized_sub(a, b, a_scale, a_zero_point, b_scale, b_zero_point,
                          scale, zero_point)


def mul(a, b, scale, zero_point):
    """Quantized multiplication; output qparams given explicitly."""
    a_scale, a_zero_point = _qparams(a)
    b_scale, b_zero_point = _qparams(b)
    return _quantized_mul(a, b, a_scale, a_zero_point, b_scale, b_zero_point,
                          scale, zero_point)


def div(a, b, scale, zero_point):
    """Quantized division; output qparams given explicitly."""
    a_scale, a_zero_point = _qparams(a)
    b_scale, b_zero_point = _qparams(b)
    return _quantized_div(a, b, a_scale, a_zero_point, b_scale, b_zero_point,
                          scale, zero_point)


def cat(tensors, dim=0, scale=1.0, zero_point=0):
    """Quantized concatenation along ``dim``.

    Output qparams may be omitted when every input already shares the same
    affine parameters, in which case they default to the first input's.
    """
    if scale is None or zero_point is None:
        return _quantized_cat(tensors, dim, None, None)
    return _quantized_cat(tensors, dim, float(scale), int(zero_point))


def clamp(x, scale, zero_point, min=None, max=None):
    """Quantized clamp on the dequantized domain."""
    x_scale, x_zero_point = _qparams(x)
    return _quantized_clamp(x, x_scale, x_zero_point, scale, zero_point,
                            min=min, max=max)


def relu(x):
    """Quantized rectifier; the output inherits the input qparams."""
    return _quantized_relu(x)


def relu6(x):
    """Quantized rectifier clamped at 6; the output inherits the input
    qparams."""
    return _quantized_relu6(x)


def leaky_relu(x, negative_slope, scale, zero_point):
    """Quantized leaky rectifier."""
    return _quantized_leaky_relu(x, negative_slope, scale, zero_point)


def elu(x, scale, zero_point, alpha=1.0):
    """Quantized exponential linear unit."""
    return _quantized_elu(x, scale, zero_point, alpha)


def hardswish(x, scale, zero_point):
    """Quantized hardswish."""
    return _quantized_hardswish(x, scale, zero_point)


def hardsigmoid(x, scale, zero_point):
    """Quantized hardsigmoid."""
    return _quantized_hardsigmoid(x, scale, zero_point)


def sigmoid(x, scale, zero_point):
    """Quantized sigmoid."""
    return _quantized_sigmoid(x, scale, zero_point)


def tanh(x, scale, zero_point):
    """Quantized hyperbolic tangent."""
    return _quantized_tanh(x, scale, zero_point)
