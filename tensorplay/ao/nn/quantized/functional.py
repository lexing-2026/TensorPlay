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


def _result(value, out):
    if out is None:
        return value
    if out.dtype != value.dtype or out.device != value.device or tuple(out.shape) != tuple(value.shape):
        raise ValueError('Output shape, dtype and device must match the result')
    out.copy_(value)
    return out


def _binary(a, b, scale, zero_point, operation, relu=False, out=None):
    from tensorplay import _C
    if out is not None:
        scale, zero_point = _qparams(out)
    return _result(_C.quantized_binary(a, b, float(scale), int(zero_point), operation, relu), out)


def add(a, b, scale=None, zero_point=None, out=None):
    if isinstance(b, (int, float)):
        return add_scalar(a, b, out=out)
    return _binary(a, b, scale, zero_point, 0, out=out)


def sub(a, b, scale=None, zero_point=None, out=None):
    return _binary(a, b, scale, zero_point, 1, out=out)


def mul(a, b, scale=None, zero_point=None, out=None):
    if isinstance(b, (int, float)):
        return mul_scalar(a, b, out=out)
    return _binary(a, b, scale, zero_point, 2, out=out)


def div(a, b, scale=None, zero_point=None, out=None):
    return _binary(a, b, scale, zero_point, 3, out=out)


def add_relu(a, b, scale=None, zero_point=None, out=None):
    if isinstance(b, (int, float)):
        return add_scalar_relu(a, b, out=out)
    return _binary(a, b, scale, zero_point, 0, True, out)


def mul_relu(a, b, scale=None, zero_point=None, out=None):
    if isinstance(b, (int, float)):
        return mul_scalar_relu(a, b, out=out)
    return _binary(a, b, scale, zero_point, 2, True, out)


def add_scalar(x, value, out=None):
    from tensorplay import _C
    return _result(_C.quantized_scalar(x, float(value), 0), out)


def add_scalar_relu(x, value, out=None):
    from tensorplay import _C
    return _result(_C.quantized_scalar(x, float(value), 0, True), out)


def mul_scalar(x, value, out=None):
    from tensorplay import _C
    return _result(_C.quantized_scalar(x, float(value), 2), out)


def mul_scalar_relu(x, value, out=None):
    from tensorplay import _C
    return _result(_C.quantized_scalar(x, float(value), 2, True), out)


def matmul(a, b, scale, zero_point):
    from tensorplay import _C
    return _C.quantized_matmul(a, b, float(scale), int(zero_point))


def softmax(x, dim, scale, zero_point):
    from tensorplay import _C
    return _C.quantized_softmax(x, int(dim), float(scale), int(zero_point))


def prelu(x, weight, scale, zero_point):
    from tensorplay import _C
    return _C.quantized_prelu(x, weight, float(scale), int(zero_point))


def threshold(x, threshold, value):
    from tensorplay import _C
    return _C.quantized_unary(x, *_qparams(x), 6, float(threshold), float(value))


def gelu(x, approximate='none'):
    from tensorplay import _C
    if approximate not in ('none', 'tanh'):
        raise ValueError('GELU approximation must be none or tanh')
    return _C.quantized_unary(x, *_qparams(x), 8 if approximate == 'none' else 9)


def celu(x, scale, zero_point, alpha=1.0):
    if alpha == 0:
        raise ValueError('CELU alpha must be nonzero')
    return _quantized_elu(x, float(scale), int(zero_point), float(alpha), 1., 1. / float(alpha))


def dropout(x, scale, zero_point, p=0.5, training=False):
    from tensorplay import _C
    return _C.quantized_dropout(x, float(scale), int(zero_point), float(p), bool(training))


def _spatial(value, rank):
    return list(value) if isinstance(value, (tuple, list)) else [int(value)] * rank


def _avg_pool(x, rank, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    from tensorplay import _C
    return _C.quantized_avg_pool(x, _spatial(kernel_size, rank),
        [] if stride is None else _spatial(stride, rank), _spatial(padding, rank),
        bool(ceil_mode), bool(count_include_pad), divisor_override)


def avg_pool1d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return _avg_pool(x, 1, kernel_size, stride, padding, ceil_mode, count_include_pad, None)


def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(x, 2, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


def avg_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(x, 3, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


def _adaptive_pool(x, rank, output_size):
    from tensorplay import _C
    size = _spatial(output_size, rank)
    size = [x.size(x.dim() - rank + d) if value is None else int(value) for d, value in enumerate(size)]
    return _C.quantized_adaptive_avg_pool(x, size)


def adaptive_avg_pool1d(x, output_size):
    return _adaptive_pool(x, 1, output_size)


def adaptive_avg_pool2d(x, output_size):
    return _adaptive_pool(x, 2, output_size)


def adaptive_avg_pool3d(x, output_size):
    return _adaptive_pool(x, 3, output_size)


__all__ += ['add_relu', 'mul_relu', 'add_scalar', 'add_scalar_relu', 'mul_scalar',
            'mul_scalar_relu', 'matmul', 'softmax', 'prelu', 'threshold', 'gelu',
            'celu', 'dropout', 'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
            'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d']


def rowwise_prepack(weight, bit_width=8, optimized_qparams=False, nbins=200, ratio=0.16, rowwise_min_max=None):
    from tensorplay import _C
    return _C._quantized_rowwise_pack(weight, int(bit_width), bool(optimized_qparams), int(nbins), float(ratio), rowwise_min_max)


def rowwise_unpack(weight, bit_width=8):
    from tensorplay import _C
    return _C._quantized_rowwise_unpack(weight, int(bit_width))


def embedding(weight, indices, bit_width=8):
    from tensorplay import _C
    return _C._quantized_embedding(weight, indices, int(bit_width))


def embedding_bag(weight, indices, offsets=None, bit_width=8, scale_grad_by_freq=False,
                  mode=0, pruned_weights=False, per_sample_weights=None,
                  compressed_indices_mapping=None, include_last_offset=False):
    from tensorplay import _C
    return _C._quantized_embedding_bag(weight, indices, int(bit_width), offsets,
        per_sample_weights, compressed_indices_mapping, bool(include_last_offset), bool(pruned_weights))


__all__ += ['rowwise_prepack', 'rowwise_unpack', 'embedding', 'embedding_bag']
