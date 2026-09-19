"""Decompositions of operator overloads onto other operators.

Each function computes its operator from simpler operators with the same
numerical contract (dtype promotion, broadcasting, special values).  The
in-place overload of an operator reuses the functional decomposition and
writes the result into ``self``; ``out=`` overloads are served by the
registry, which writes the result into the destination.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE

from . import register_decomposition

ops = getattr(tp.ops, NATIVE_NAMESPACE)


def _inplace(functional: Callable[..., Any]) -> Callable[..., Any]:
    def inplace(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self.copy_(functional(self, *args, **kwargs))

    inplace.__name__ = functional.__name__ + "_"
    return inplace


def _register_with_inplace(functional_ops: Any, inplace_ops: Any):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        register_decomposition(functional_ops)(fn)
        register_decomposition(inplace_ops)(_inplace(fn))
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Pointwise arithmetic
# ---------------------------------------------------------------------------


@_register_with_inplace(ops.addcmul.default, ops.addcmul_.default)
def addcmul(self, tensor1, tensor2, *, value=1):
    return self + value * tensor1 * tensor2


@_register_with_inplace(ops.addcdiv.default, ops.addcdiv_.default)
def addcdiv(self, tensor1, tensor2, *, value=1):
    return self + value * tensor1 / tensor2


@register_decomposition([ops.rsub.Tensor, ops.rsub.Scalar])
def rsub(self, other, alpha=1):
    return other - alpha * self


@_register_with_inplace([ops.clamp_min.default, ops.clamp_min.Tensor],
                        [ops.clamp_min_.default, ops.clamp_min_.Tensor])
def clamp_min(self, min):
    return tp.clamp(self, min=min)


@_register_with_inplace([ops.clamp_max.default, ops.clamp_max.Tensor],
                        [ops.clamp_max_.default, ops.clamp_max_.Tensor])
def clamp_max(self, max):
    return tp.clamp(self, max=max)


@_register_with_inplace(ops.deg2rad.default, ops.deg2rad_.default)
def deg2rad(self):
    if self.is_complex():
        raise RuntimeError("deg2rad is not supported for complex tensors.")
    return self * (math.pi / 180.0)


@_register_with_inplace(ops.rad2deg.default, ops.rad2deg_.default)
def rad2deg(self):
    if self.is_complex():
        raise RuntimeError("rad2deg is not supported for complex tensors.")
    return self * (180.0 / math.pi)


@_register_with_inplace(ops.frac.default, ops.frac_.default)
def frac(self):
    return self - tp.trunc(self)


@_register_with_inplace(ops.sgn.default, ops.sgn_.default)
def sgn(self):
    if self.is_complex():
        magnitude = self.abs()
        return tp.where(magnitude == 0, tp.zeros_like(self), self / magnitude)
    return tp.sign(self)


@_register_with_inplace(ops.sinc.default, ops.sinc_.default)
def sinc(self):
    scaled = self * math.pi
    return tp.where(self == 0, tp.ones_like(scaled), tp.sin(scaled) / scaled)


@_register_with_inplace(ops.heaviside.default, ops.heaviside_.default)
def heaviside(self, values):
    positive = (self > 0).to(tp.result_type(self, values))
    return tp.where(self == 0, values, positive)


@register_decomposition([ops.lerp.default, ops.lerp.Scalar, ops.lerp.Tensor])
def lerp(self, end, weight):
    # Interpolate from whichever end point is nearer to ``weight`` so the
    # arithmetic stays close to zero: w >= 0.5 uses (w - 1) * (end - start) + end.
    if not isinstance(weight, tp.Tensor):
        weight = tp.full((), weight, dtype=self.dtype, device=self.device)
    far = weight.abs() >= 0.5
    coeff = tp.where(far, weight - 1, weight)
    base = tp.where(far, end, self)
    return coeff * (end - self) + base


register_decomposition([ops.lerp_.Scalar, ops.lerp_.Tensor])(_inplace(lerp))


@register_decomposition(ops.logaddexp.default)
def logaddexp(self, other):
    first = self.real if self.is_complex() else self
    second = other.real if other.is_complex() else other
    mask = first >= second
    high = tp.where(mask, self, other)
    low = tp.where(mask, other, self)
    # Equal infinities would give inf - inf; the result is that infinity.
    same_inf = ~tp.isfinite(first) & (first == second)
    if self.is_complex() or other.is_complex():
        low_real = low.real if low.is_complex() else low
        inf_values = tp.where(low_real < 0, low, tp.log(tp.exp(low) + tp.exp(high)))
        values = tp.where(same_inf, inf_values, high + tp.log1p(tp.exp(low - high)))
        nan = tp.full((), complex(float("nan"), float("nan")), dtype=values.dtype, device=values.device)
        return tp.where(tp.isnan(low), nan, values)
    return tp.where(same_inf, self, high + tp.log1p(tp.exp(low - high)))


@register_decomposition(ops.logaddexp2.default)
def logaddexp2(self, other):
    if self.is_complex() or other.is_complex():
        raise RuntimeError("logaddexp2 doesn't support complex dtypes")
    mask = self >= other
    high = tp.where(mask, self, other)
    low = tp.where(mask, other, self)
    same_inf = tp.isinf(self) & (self == other)
    result = high + tp.log1p(tp.exp2(low - high)) * (1.0 / math.log(2))
    return tp.where(same_inf, self, result)


@_register_with_inplace(
    [ops.xlogy.default, ops.xlogy.Tensor, ops.xlogy.Scalar_Other],
    [ops.xlogy_.default, ops.xlogy_.Tensor, ops.xlogy_.Scalar_Other],
)
def xlogy(self, other):
    if not isinstance(other, tp.Tensor):
        other = tp.full((), other, dtype=self.dtype, device=self.device)
    product = self * tp.log(other)
    zero = tp.zeros((), dtype=product.dtype, device=product.device)
    nan = tp.full((), float("nan"), dtype=product.dtype, device=product.device)
    return tp.where(tp.isnan(other), nan, tp.where(self == 0, zero, product))


@register_decomposition(ops.xlogy.Scalar_Self)
def xlogy_scalar_self(self, other):
    return xlogy(tp.full((), self, dtype=other.dtype, device=other.device), other)


@_register_with_inplace(ops.nan_to_num.default, ops.nan_to_num_.default)
def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
    if not (self.is_floating_point() or self.is_complex()):
        return self.clone()
    info = tp.finfo(self.dtype)
    nan = 0.0 if nan is None else nan
    posinf = info.max if posinf is None else posinf
    neginf = info.min if neginf is None else neginf
    result = tp.where(tp.isnan(self), tp.full_like(self, nan), self)
    result = tp.where(tp.isneginf(self), tp.full_like(self, neginf), result)
    return tp.where(tp.isposinf(self), tp.full_like(self, posinf), result)


@_register_with_inplace(ops.logit.default, ops.logit_.default)
def logit(self, eps=None):
    x = self if eps is None else tp.clamp(self, eps, 1.0 - eps)
    return tp.log(x / (1 - x))


# ---------------------------------------------------------------------------
# Activations and their backward formulas
# ---------------------------------------------------------------------------


@_register_with_inplace(ops.silu.default, ops.silu_.default)
def silu(self):
    return self * tp.sigmoid(self)


@register_decomposition([ops.silu_backward.default, ops.silu_backward.grad_input])
def silu_backward(grad_output, self):
    sigmoid = tp.sigmoid(self)
    return grad_output * (sigmoid * (1 + self * (1 - sigmoid)))


@_register_with_inplace(ops.mish.default, ops.mish_.default)
def mish(self):
    return self * tp.tanh(tp.nn.functional.softplus(self))


@register_decomposition(ops.mish_backward.default)
def mish_backward(grad_output, self):
    sigmoid = tp.sigmoid(self)
    tanh_softplus = tp.tanh(tp.nn.functional.softplus(self))
    return grad_output * (tanh_softplus + self * sigmoid * (1 - tanh_softplus * tanh_softplus))


@_register_with_inplace(ops.celu.default, ops.celu_.default)
def celu(self, alpha=1.0):
    if alpha == 0:
        raise RuntimeError("ZeroDivisionError: alpha cannot be 0 for CELU")
    return tp.where(self > 0, self, alpha * tp.expm1(self / alpha))


@register_decomposition([ops.elu_backward.default, ops.elu_backward.grad_input])
def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    negcoef = alpha * scale
    poscoef = scale
    if is_result:
        negative = grad_output * input_scale * (self_or_result + negcoef)
    else:
        negative = grad_output * input_scale * negcoef * tp.exp(self_or_result * input_scale)
    return tp.where(self_or_result <= 0, negative, grad_output * poscoef)


@_register_with_inplace(ops.hardsigmoid.default, ops.hardsigmoid_.default)
def hardsigmoid(self):
    return tp.clamp(self + 3, 0, 6) / 6


@register_decomposition([ops.hardsigmoid_backward.default, ops.hardsigmoid_backward.grad_input])
def hardsigmoid_backward(grad_output, self):
    inside = (self > -3.0) & (self < 3.0)
    return tp.where(inside, grad_output / 6.0, tp.zeros_like(grad_output))


@_register_with_inplace(ops.hardswish.default, ops.hardswish_.default)
def hardswish(self):
    return self * tp.clamp(self + 3, 0, 6) / 6


@register_decomposition(ops.hardswish_backward.default)
def hardswish_backward(grad_output, self):
    middle = tp.where(self < 3.0, grad_output * ((self / 3.0) + 0.5), grad_output)
    return tp.where(self <= -3.0, tp.zeros_like(grad_output), middle)


@register_decomposition([ops.hardtanh_backward.default, ops.hardtanh_backward.grad_input])
def hardtanh_backward(grad_output, self, min_val, max_val):
    outside = (self <= min_val) | (self >= max_val)
    return tp.where(outside, tp.zeros_like(grad_output), grad_output)


@register_decomposition(ops.hardshrink.default)
def hardshrink(self, lambd=0.5):
    return tp.where(tp.abs(self) <= lambd, tp.zeros_like(self), self)


@register_decomposition(ops.softshrink.default)
def softshrink(self, lambd=0.5):
    limit = tp.finfo(self.dtype).max
    if not 0 <= lambd <= limit:
        raise RuntimeError(
            f"lambda must be in range [0, {limit}] for input dtype {self.dtype}, but found {lambd}"
        )
    # Multiplying by zero keeps NaN inputs NaN.
    return tp.where(tp.abs(self) > lambd, self - tp.sign(self) * lambd, self * 0)


@register_decomposition(
    [ops.leaky_relu_backward.default, ops.leaky_relu_backward.grad_input]
)
def leaky_relu_backward(grad_output, self, negative_slope, self_is_result):
    return tp.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition([ops.gelu_backward.default, ops.gelu_backward.grad_input])
def gelu_backward(grad_output, self, approximate="none"):
    if approximate == "tanh":
        kappa = math.sqrt(2.0 / math.pi) * 0.044715
        beta = math.sqrt(2.0 / math.pi)
        inner = beta * self + kappa * self * self * self
        tanh_inner = tp.tanh(inner)
        left = 0.5 * self
        right = 1 + tanh_inner
        dinner = beta + 3 * kappa * self * self
        return grad_output * (0.5 * right + left * (1 - tanh_inner * tanh_inner) * dinner)
    cdf = 0.5 * (1 + tp.erf(self * (1.0 / math.sqrt(2.0))))
    pdf = tp.exp(-0.5 * self * self) * (1.0 / math.sqrt(2.0 * math.pi))
    return grad_output * (cdf + self * pdf)


@register_decomposition([ops.softplus.default])
def softplus(self, beta=1, threshold=20):
    scaled = self * beta
    return tp.where(scaled > threshold, self, tp.log1p(tp.exp(scaled)) / beta)


@register_decomposition([ops.softplus_backward.default, ops.softplus_backward.grad_input])
def softplus_backward(grad_output, self, beta, threshold):
    scaled = self * beta
    z = tp.exp(scaled)
    return tp.where(scaled > threshold, grad_output, grad_output * z / (z + 1.0))


@_register_with_inplace(ops.threshold.default, ops.threshold_.default)
def threshold(self, threshold, value):
    return tp.where(self <= threshold, tp.full_like(self, value), self)


@register_decomposition([ops.threshold_backward.default, ops.threshold_backward.grad_input])
def threshold_backward(grad_output, self, threshold):
    return tp.where(self <= threshold, tp.zeros_like(grad_output), grad_output)


@register_decomposition([ops.sigmoid_backward.default, ops.sigmoid_backward.grad_input])
def sigmoid_backward(grad_output, output):
    return grad_output * (output * (1 - output)).conj()


@register_decomposition([ops.tanh_backward.default, ops.tanh_backward.grad_input])
def tanh_backward(grad_output, output):
    return grad_output * (1 - output * output).conj()


@register_decomposition([ops.logit_backward.default, ops.logit_backward.grad_input])
def logit_backward(grad_output, self, eps=None):
    if eps is None:
        inside = (self >= 0.0) & (self <= 1.0)
        return tp.where(inside, grad_output / (self * (1.0 - self)), tp.full_like(self, float("nan")))
    inside = (self >= eps) & (self <= 1.0 - eps)
    return tp.where(inside, grad_output / (self * (1.0 - self)), tp.zeros_like(self))


@register_decomposition(ops.glu.default)
def glu(self, dim=-1):
    dim = dim % self.dim()
    if int(self.shape[dim]) % 2 != 0:
        raise RuntimeError(f"Halving dimension must be even, but dimension {dim} is size {int(self.shape[dim])}")
    first, second = tp.chunk(self, 2, dim=dim)
    return first * tp.sigmoid(second)


# ---------------------------------------------------------------------------
# Products and reductions over small structures
# ---------------------------------------------------------------------------


@register_decomposition(ops.mv.default)
def mv(self, vec):
    if self.dim() != 2 or vec.dim() != 1:
        raise RuntimeError(f"matrix @ vector expected, got {self.dim()}, {vec.dim()}")
    if int(self.shape[1]) != int(vec.shape[0]):
        raise RuntimeError(
            f"size mismatch, got input ({int(self.shape[0])}x{int(self.shape[1])}), vec ({int(vec.shape[0])})"
        )
    return (self * vec).sum(dim=1)


def _dot_check(self, other):
    if self.dim() != 1 or other.dim() != 1:
        raise RuntimeError(f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors")
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"dot : expected both vectors to have same dtype, but found {self.dtype} and {other.dtype}"
        )
    if self.numel() != other.numel():
        raise RuntimeError(
            f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have the "
            f"same number of elements, but got {self.numel()} and {other.numel()} elements respectively"
        )


@register_decomposition(ops.dot.default)
def dot(self, tensor):
    _dot_check(self, tensor)
    return (self * tensor).sum()


@register_decomposition(ops.vdot.default)
def vdot(self, other):
    _dot_check(self, other)
    return (self.conj() * other).sum()


@register_decomposition(ops.trace.default)
def trace(self):
    if self.dim() != 2:
        raise RuntimeError(f"expected a matrix, but got tensor with dim {self.dim()}")
    return tp.diagonal(self).sum()


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------


def _dtype_or(dtype, fallback):
    if dtype is None or dtype == tp.DType.undefined:
        return fallback
    return dtype


@register_decomposition(ops.zeros.default)
def zeros(size, *, dtype=None, device=None, pin_memory=False, requires_grad=False):
    return tp.full(tuple(size), 0, dtype=_dtype_or(dtype, tp.get_default_dtype()), device=device)


@register_decomposition(ops.ones.default)
def ones(size, *, dtype=None, device=None, pin_memory=False, requires_grad=False):
    return tp.full(tuple(size), 1, dtype=_dtype_or(dtype, tp.get_default_dtype()), device=device)


@register_decomposition(ops.zeros_like.default)
def zeros_like(self, *, dtype=None, device=None, requires_grad=False):
    return tp.full_like(self, 0, dtype=_dtype_or(dtype, self.dtype), device=device or self.device)


@register_decomposition(ops.ones_like.default)
def ones_like(self, *, dtype=None, device=None, requires_grad=False):
    return tp.full_like(self, 1, dtype=_dtype_or(dtype, self.dtype), device=device or self.device)


@register_decomposition(ops.empty_like.default)
def empty_like(self, *, dtype=None, device=None, requires_grad=False):
    return tp.empty(tuple(self.shape), dtype=_dtype_or(dtype, self.dtype), device=device or self.device)


def _new_full(self, size, value, dtype, device):
    return tp.full(tuple(size), value, dtype=_dtype_or(dtype, self.dtype), device=device or self.device)


@register_decomposition(ops.new_empty.default)
def new_empty(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return tp.empty(tuple(size), dtype=_dtype_or(dtype, self.dtype), device=device or self.device)


@register_decomposition(ops.new_full.default)
def new_full(self, size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _new_full(self, size, fill_value, dtype, device)


@register_decomposition(ops.new_zeros.default)
def new_zeros(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _new_full(self, size, 0, dtype, device)


@register_decomposition(ops.new_ones.default)
def new_ones(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _new_full(self, size, 1, dtype, device)


@_register_with_inplace(ops.fill.Scalar, ops.fill_.Scalar)
def fill_scalar(self, value):
    return tp.full_like(self, value)


@_register_with_inplace(ops.fill.Tensor, ops.fill_.Tensor)
def fill_tensor(self, value):
    if value.dim() != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.dim()} dimensions"
        )
    return value.to(self.dtype).expand(tuple(self.shape)).clone()


@register_decomposition(ops.zero_.default)
def zero_(self):
    return self.copy_(tp.zeros_like(self))


def _eye(n, m, dtype, device):
    if n < 0:
        raise RuntimeError(f"n must be greater or equal to 0, got {n}")
    if m < 0:
        raise RuntimeError(f"m must be greater or equal to 0, got {m}")
    rows = tp.arange(n, device=device).unsqueeze(-1)
    cols = tp.arange(m, device=device)
    return (rows == cols).to(dtype)


@register_decomposition(ops.eye.default)
def eye(n, m=-1, *, dtype=tp.float32, device=None, requires_grad=False):
    return _eye(n, n if m < 0 else m, dtype, device)


@register_decomposition(ops.eye.m)
def eye_m(n, m, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _eye(n, m, _dtype_or(dtype, tp.get_default_dtype()), device)


def _linspace(start, end, steps, dtype, device):
    if steps < 0:
        raise RuntimeError("number of steps must be non-negative")
    if steps == 0:
        return tp.full((0,), 0, dtype=dtype, device=device)
    if steps == 1:
        return tp.full((1,), start, dtype=dtype, device=device)
    # Low-precision results accumulate in float32; integers in float32 too,
    # since the step is fractional.
    compute = tp.float64 if dtype in (tp.float64, tp.complex128) else tp.float32
    if dtype in (tp.complex64, tp.complex128):
        compute = dtype
    step = (end - start) / (steps - 1)
    index = tp.arange(steps, device=device)
    # Anchor each half at its own end point: both ends come out exact.
    forward = start + step * index.to(compute)
    backward = end - step * (steps - 1 - index).to(compute)
    return tp.where(index < steps / 2, forward, backward).to(dtype)


def _scalar(value, name="linspace"):
    if isinstance(value, tp.Tensor):
        if value.dim() != 0:
            raise RuntimeError(f"{name} only supports 0-dimensional start and end tensors")
        return value.item()
    return value


@register_decomposition(ops.linspace.default)
def linspace(start, end, steps, *, dtype=tp.float32, device=None, requires_grad=False):
    return _linspace(_scalar(start), _scalar(end), steps, dtype, device)


@register_decomposition([ops.linspace.Tensor_Tensor, ops.linspace.Tensor_Scalar, ops.linspace.Scalar_Tensor])
def linspace_tensor(start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None):
    start, end = _scalar(start), _scalar(end)
    if isinstance(start, complex) or isinstance(end, complex):
        fallback = tp.complex128 if tp.get_default_dtype() == tp.float64 else tp.complex64
        dtype = _dtype_or(dtype, fallback)
        if not dtype.is_complex:
            raise RuntimeError(f"linspace(): inferred dtype {fallback} can't be safely cast to passed dtype {dtype}")
    return _linspace(start, end, steps, _dtype_or(dtype, tp.get_default_dtype()), device)


def _logspace(start, end, steps, base, dtype, device):
    start, end = _scalar(start, "logspace"), _scalar(end, "logspace")
    if not (dtype.is_floating_point or dtype.is_complex):
        # Integer results come from integer exponents.
        start, end = int(start), int(end)
    if base < 0:
        raise NotImplementedError("logspace with a negative base")
    exponent = _linspace(start, end, steps, tp.float64, device)
    return tp.pow(tp.full_like(exponent, base), exponent).to(dtype)


@register_decomposition(ops.logspace.default)
def logspace(start, end, steps, base=10.0, *, dtype=tp.float32, device=None, requires_grad=False):
    return _logspace(start, end, steps, base, dtype, device)


@register_decomposition([ops.logspace.Tensor_Tensor, ops.logspace.Tensor_Scalar, ops.logspace.Scalar_Tensor])
def logspace_tensor(start, end, steps, base=10.0, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _logspace(start, end, steps, base, _dtype_or(dtype, tp.get_default_dtype()), device)


def _hann(window_length, periodic, dtype, device):
    dtype = _dtype_or(dtype, tp.get_default_dtype())
    if window_length == 0:
        return tp.empty(0, dtype=dtype, device=device)
    if window_length == 1:
        return tp.ones(1, dtype=dtype, device=device)
    denominator = window_length if periodic else window_length - 1
    n = tp.arange(window_length, dtype=tp.float64, device=device)
    return (0.5 - 0.5 * tp.cos(n * (2.0 * math.pi / denominator))).to(dtype)


@register_decomposition(ops.hann_window.default)
def hann_window(window_length, periodic=True, dtype=None):
    return _hann(window_length, periodic, dtype, None)


@register_decomposition(ops.hann_window.periodic)
def hann_window_periodic(window_length, periodic, *, dtype=None, layout=None, device=None, pin_memory=None):
    return _hann(window_length, periodic, dtype, device)


# ---------------------------------------------------------------------------
# Views onto other views, and their copying variants
# ---------------------------------------------------------------------------


@register_decomposition(ops.t.default)
def t(self):
    if self.dim() > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {self.dim()}D")
    if self.dim() < 2:
        return ops.alias.default(self)
    return tp.permute(self, (1, 0))


@register_decomposition([ops.transpose.default, ops.transpose.int])
def transpose(self, dim0, dim1):
    ndim = max(self.dim(), 1)
    dims = list(range(self.dim()))
    if not dims:
        return ops.alias.default(self)
    a, b = dim0 % ndim, dim1 % ndim
    dims[a], dims[b] = dims[b], dims[a]
    return tp.permute(self, tuple(dims))


@register_decomposition(ops._unsafe_view.default)
def _unsafe_view(self, size):
    return ops.view.default(self, list(size))


@register_decomposition(ops._reshape_alias.default)
def _reshape_alias(self, size, stride):
    return ops.view.default(self, list(size))


@register_decomposition(ops.expand_as.default)
def expand_as(self, other):
    return self.expand(tuple(other.shape))


@register_decomposition(ops.detach.default)
def detach(self):
    return ops.alias.default(self)


def _copy_of(view):
    return view.clone()


@register_decomposition(ops.alias_copy.default)
def alias_copy(self):
    return _copy_of(ops.alias.default(self))


@register_decomposition(ops.t_copy.default)
def t_copy(self):
    return _copy_of(t(self))


@register_decomposition(ops.transpose_copy.int)
def transpose_copy(self, dim0, dim1):
    return _copy_of(transpose(self, dim0, dim1))


@register_decomposition(ops.squeeze_copy.default)
def squeeze_copy(self):
    return _copy_of(self.squeeze())


@register_decomposition(ops.squeeze_copy.dim)
def squeeze_copy_dim(self, dim):
    return _copy_of(self.squeeze(dim))


@register_decomposition(ops.squeeze_copy.dims)
def squeeze_copy_dims(self, dim):
    return _copy_of(self.squeeze(tuple(dim)))


@register_decomposition(ops.unsqueeze_copy.default)
def unsqueeze_copy(self, dim):
    return _copy_of(self.unsqueeze(dim))


@register_decomposition(ops.diagonal_copy.default)
def diagonal_copy(self, offset=0, dim1=0, dim2=1):
    return _copy_of(tp.diagonal(self, offset, dim1, dim2))


@register_decomposition(ops.unfold_copy.default)
def unfold_copy(self, dimension, size, step):
    return _copy_of(self.unfold(dimension, size, step))


@register_decomposition(ops.expand_copy.default)
def expand_copy(self, size, *, implicit=False):
    return _copy_of(self.expand(tuple(size)))


def _split_sizes(length, split_size):
    if split_size <= 0:
        if length == 0 and split_size == 0:
            return [0]
        raise RuntimeError(f"split_size must be a positive integer, but got {split_size}")
    sizes = [split_size] * (length // split_size)
    if length % split_size or not sizes:
        sizes.append(length % split_size if sizes else length)
    return sizes


def _split_with_sizes(self, sizes, dim):
    dim = dim % max(self.dim(), 1)
    total = int(self.shape[dim])
    if sum(sizes) != total:
        raise RuntimeError(
            f"split_with_sizes expects split_sizes to sum exactly to {total}, but got {list(sizes)}"
        )
    pieces = []
    start = 0
    for size in sizes:
        pieces.append(tp.narrow(self, dim, start, size))
        start += size
    return pieces


@register_decomposition([ops.split.default, ops.split.Tensor])
def split(self, split_size, dim=0):
    dim_size = int(self.shape[dim])
    if split_size < 0:
        raise RuntimeError(f"split expects split_size be non-negative, but got split_size={split_size}")
    if dim_size == 0:
        return [ops.alias.default(self)]
    if split_size == 0:
        raise RuntimeError(
            f"split_size can only be 0 if dimension size is 0, but got dimension size of {dim_size}"
        )
    return _split_with_sizes(self, _split_sizes(dim_size, split_size), dim)


@register_decomposition([ops.split.sizes])
def split_sizes(self, split_sizes, dim=0):
    return _split_with_sizes(self, list(split_sizes), dim)


@register_decomposition(ops.unsafe_split.Tensor)
def unsafe_split(self, split_size, dim=0):
    return split(self, split_size, dim)


@register_decomposition(ops.unsafe_split_with_sizes.default)
def unsafe_split_with_sizes(self, split_sizes, dim=0):
    return _split_with_sizes(self, list(split_sizes), dim)


@register_decomposition(ops.split_with_sizes_copy.default)
def split_with_sizes_copy(self, split_sizes, dim=0):
    return [piece.clone() for piece in _split_with_sizes(self, list(split_sizes), dim)]


@register_decomposition([ops.unbind.default, ops.unbind.int])
def unbind(self, dim=0):
    if self.dim() == 0:
        raise IndexError("Dimension specified as 0 but tensor has no dimensions")
    dim = dim % self.dim()
    return [tp.select(self, dim, index) for index in range(int(self.shape[dim]))]


@register_decomposition([ops.narrow.default])
def narrow(self, dim, start, length):
    if self.dim() == 0:
        raise RuntimeError("narrow() cannot be applied to a 0-dim tensor.")
    if length < 0:
        raise RuntimeError("narrow(): length must be non-negative.")
    dim = dim % self.dim()
    size = int(self.shape[dim])
    if not -size <= start <= size:
        raise IndexError(f"start out of range (expected to be in range of [{-size}, {size}], but got {start})")
    start = start + size if start < 0 else start
    if start > size - length:
        raise RuntimeError(f"start ({start}) + length ({length}) exceeds dimension size ({size}).")
    return ops.slice.Tensor(self, dim, start, start + length, 1)


@register_decomposition(ops.narrow.Tensor)
def narrow_tensor(self, dim, start, length):
    if start.dim() != 0 or start.is_floating_point() or start.is_complex() or start.dtype == tp.bool:
        raise RuntimeError("start must be an 0-dim integral Tensor.")
    return narrow(self, dim, int(start.item()), length)


@register_decomposition(ops.stack.default)
def stack(tensors, dim=0):
    tensors = list(tensors)
    if not tensors:
        raise RuntimeError("stack expects a non-empty TensorList")
    ndim = tensors[0].dim() + 1
    dim = dim % ndim
    return tp.cat([tensor.unsqueeze(dim) for tensor in tensors], dim)


def _roll_one(self, shift, dim):
    size = int(self.shape[dim])
    if size == 0:
        return self.clone()
    shift %= size
    if shift == 0:
        return self.clone()
    return tp.cat(
        [tp.narrow(self, dim, size - shift, shift), tp.narrow(self, dim, 0, size - shift)], dim
    )


@register_decomposition(ops.roll.default)
def roll(self, shifts, dims=()):
    shifts = list(shifts) if isinstance(shifts, (list, tuple)) else [shifts]
    dims = list(dims) if isinstance(dims, (list, tuple)) else [dims]
    if self.numel() == 0:
        return self.clone()
    if self.dim() == 0 and dims:
        raise IndexError(f"Dimension specified as {dims[0]} but tensor has no dimensions")
    if not shifts:
        raise RuntimeError("`shifts` required")
    if not dims:
        if len(shifts) != 1:
            raise RuntimeError(f"shifts and dimensions must align. shifts: {len(shifts)}, dims: 0")
        return _roll_one(self.flatten(), shifts[0], 0).view(tuple(self.shape))
    if len(shifts) != len(dims):
        raise RuntimeError(f"shifts and dimensions must align. shifts: {len(shifts)}, dims: {len(dims)}")
    result = self
    for shift, dim in zip(shifts, dims):
        result = _roll_one(result, shift, dim % max(self.dim(), 1))
    return result


@register_decomposition(ops.rot90.default)
def rot90(self, k=1, dims=(0, 1)):
    dims = list(dims)
    if len(dims) != 2:
        raise RuntimeError(f"expected total rotation dims == 2, but got dims = {len(dims)}")
    ndim = self.dim()
    if ndim < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {ndim}")
    d0, d1 = dims[0] % ndim, dims[1] % ndim
    if d0 == d1:
        raise RuntimeError(f"expected rotation dims to be different, but got dim0 = {d0} and dim1 = {d1}")
    k %= 4
    if k == 1:
        return tp.flip(self, (d1,)).transpose(d0, d1)
    if k == 2:
        return tp.flip(self, (d0, d1))
    if k == 3:
        return tp.flip(self, (d0,)).transpose(d0, d1)
    return self.clone()


@register_decomposition(ops.take.default)
def take(self, index):
    flat = self.reshape(-1)
    numel = flat.shape[0]
    wrapped = tp.where(index < 0, index + numel, index)
    return tp.index_select(flat, 0, wrapped.reshape(-1)).view(tuple(index.shape))


def _triangle_mask(self, diagonal, lower):
    if self.dim() < 2:
        name = "tril" if lower else "triu"
        raise RuntimeError(f"{name}: input tensor must have at least 2 dimensions")
    rows, cols = int(self.shape[-2]), int(self.shape[-1])
    row = tp.arange(rows, device=self.device).unsqueeze(-1)
    col = tp.arange(cols, device=self.device)
    offset = col - row
    return offset <= diagonal if lower else offset >= diagonal


@_register_with_inplace(ops.tril.default, ops.tril_.default)
def tril(self, diagonal=0):
    return tp.where(_triangle_mask(self, diagonal, True), self, tp.zeros_like(self)).contiguous()


@_register_with_inplace(ops.triu.default, ops.triu_.default)
def triu(self, diagonal=0):
    return tp.where(_triangle_mask(self, diagonal, False), self, tp.zeros_like(self)).contiguous()


@register_decomposition(ops.diag_embed.default)
def diag_embed(self, offset=0, dim1=-2, dim2=-1):
    ndim = self.dim() + 1
    dim1, dim2 = dim1 % ndim, dim2 % ndim
    if dim1 == dim2:
        raise RuntimeError(f"diagonal dimensions cannot be identical {dim1}, {dim2}")
    n = int(self.shape[-1]) + abs(offset)
    batch = list(self.shape[:-1])
    shape = []
    source = iter(batch)
    for position in range(ndim):
        shape.append(n if position in (dim1, dim2) else next(source))
    result = tp.zeros(tuple(shape), dtype=self.dtype, device=self.device)
    return ops.diagonal_scatter.default(result, self, offset, dim1, dim2)


@register_decomposition(ops.block_diag.default)
def block_diag(tensors):
    blocks = []
    for tensor in tensors:
        if tensor.dim() == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() != 2:
            raise RuntimeError(
                f"block_diag: Input tensors must have 2 or fewer dimensions. Input {len(blocks)} has {tensor.dim()} dimensions"
            )
        blocks.append(tensor)
    rows = sum(int(b.shape[0]) for b in blocks)
    cols = sum(int(b.shape[1]) for b in blocks)
    dtype = blocks[0].dtype
    for b in blocks[1:]:
        dtype = tp.promote_types(dtype, b.dtype)
    device = blocks[0].device if blocks else None
    result = tp.zeros((rows, cols), dtype=dtype, device=device)
    r = c = 0
    for block in blocks:
        h, w = int(block.shape[0]), int(block.shape[1])
        band = ops.slice_scatter.default(
            tp.narrow(result, 0, r, h), block.to(dtype), 1, c, c + w, 1
        )
        result = ops.slice_scatter.default(result, band, 0, r, r + h, 1)
        r += h
        c += w
    return result


# ---------------------------------------------------------------------------
# Backward formulas of views: scatter the gradient into zeros
# ---------------------------------------------------------------------------


@register_decomposition(ops.select_scatter.default)
def select_scatter(self, src, dim, index):
    dim = dim % max(self.dim(), 1)
    size = int(self.shape[dim])
    index = index + size if index < 0 else index
    positions = tp.arange(size, device=self.device)
    shape = [1] * self.dim()
    shape[dim] = size
    mask = (positions == index).view(tuple(shape))
    return tp.where(mask, src.unsqueeze(dim).to(self.dtype), self)


@register_decomposition(ops.select_backward.default)
def select_backward(grad_output, self, dim, index):
    return select_scatter(tp.zeros_like(self, dtype=grad_output.dtype), grad_output, dim, index)


@register_decomposition(ops.slice_backward.default)
def slice_backward(grad_output, self, dim=0, start=None, end=None, step=1):
    zeros_ = tp.zeros_like(self, dtype=grad_output.dtype)
    return ops.slice_scatter.default(zeros_, grad_output, dim, start, end, step)


@register_decomposition(ops.diagonal_backward.default)
def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    zeros_ = tp.zeros(tuple(input_sizes), dtype=grad_output.dtype, device=grad_output.device)
    return ops.diagonal_scatter.default(zeros_, grad_output, offset, dim1, dim2)


@register_decomposition(ops.unfold_backward.default)
def unfold_backward(grad_output, input_sizes, dim, size, step):
    if step <= 0:
        raise ValueError(f"step is {step} but must be > 0")
    ndim = len(input_sizes)
    dim = dim % max(ndim, 1)
    windows = int(grad_output.shape[dim]) if ndim else 1
    index = (tp.arange(windows, device=grad_output.device).unsqueeze(-1) * step
             + tp.arange(size, device=grad_output.device))
    # grad_output: input shape with dim -> windows plus a trailing window axis.
    moved = grad_output.movedim(dim, -2) if ndim else grad_output
    moved = moved.reshape(tuple(moved.shape[:-2]) + (windows * size,))
    result = tp.zeros(tuple(input_sizes), dtype=grad_output.dtype, device=grad_output.device)
    target = result.movedim(dim, -1) if ndim else result
    target = target.index_add(-1, index.reshape(-1), moved)
    return target.movedim(-1, dim) if ndim else target.reshape(())


# ---------------------------------------------------------------------------
# Masked and indexed fills
# ---------------------------------------------------------------------------


@_register_with_inplace(
    [ops.masked_fill.default, ops.masked_fill.Scalar, ops.masked_fill.Tensor],
    [ops.masked_fill_.default, ops.masked_fill_.Scalar, ops.masked_fill_.Tensor],
)
def masked_fill(self, mask, value):
    if isinstance(value, tp.Tensor):
        if value.dim() != 0:
            raise RuntimeError(
                "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
                f"with {value.dim()} dimension(s)."
            )
        if value.is_complex() and not self.is_complex():
            raise RuntimeError(f"could not convert to type {self.dtype} without overflow")
        value = value.to(self.dtype)
    else:
        if isinstance(value, complex) and not self.is_complex():
            raise RuntimeError(f"could not convert to type {self.dtype} without overflow")
        value = tp.full((), value, dtype=self.dtype, device=self.device)
    return tp.where(mask, value, self).contiguous()


def _along(index, dim, like):
    shape = [1] * like.dim()
    shape[dim] = -1
    return index.reshape(tuple(shape)).expand(tuple(like.shape))


@_register_with_inplace(ops.index_add.default, ops.index_add_.default)
def index_add(self, dim, index, source):
    dim = dim % max(self.dim(), 1)
    return self.scatter_add(dim, _along(index, dim, source), source.to(self.dtype))


@_register_with_inplace(ops.index_copy.default, ops.index_copy_.default)
def index_copy(self, dim, index, source):
    dim = dim % max(self.dim(), 1)
    return self.scatter(dim, _along(index, dim, source), source.to(self.dtype))


@_register_with_inplace(
    [ops.index_fill.Scalar, ops.index_fill.Tensor, ops.index_fill.int_Scalar, ops.index_fill.int_Tensor],
    [ops.index_fill_.Scalar, ops.index_fill_.Tensor, ops.index_fill_.int_Scalar, ops.index_fill_.int_Tensor],
)
def index_fill(self, dim, index, value):
    if index.dim() > 1:
        raise RuntimeError(f"Index should have dimension 1 or 0 (got {index.dim()})")
    if isinstance(value, tp.Tensor) and value.dim() != 0:
        raise RuntimeError(
            f"Only supports 0-dimensional value tensor. Got a tensor with {value.dim()} dimensions."
        )
    dim = dim % max(self.dim(), 1)
    size = int(self.shape[dim]) if self.dim() else 1
    wrapped = tp.where(index < 0, index + size, index).reshape(-1)
    hit = (tp.arange(size, device=self.device).unsqueeze(-1) == wrapped).any(-1)
    shape = [1] * self.dim()
    if shape:
        shape[dim] = size
    mask = hit.reshape(tuple(shape))
    return masked_fill(self, mask, value)


# ---------------------------------------------------------------------------
# Reductions and statistics
# ---------------------------------------------------------------------------


@register_decomposition([ops.count_nonzero.default, ops.count_nonzero.dim_IntList])
def count_nonzero(self, dim=()):
    dims = tuple(dim) if dim is not None else ()
    nonzero = (self != 0).to(tp.int64)
    return nonzero.sum() if not dims else nonzero.sum(dim=dims)


@register_decomposition(ops.nansum.default)
def nansum(self, dim=(), keepdim=False, *, dtype=None):
    cleaned = tp.where(tp.isnan(self), tp.zeros_like(self), self) if self.is_floating_point() else self
    if dtype is not None:
        cleaned = cleaned.to(dtype)
    dims = tuple(dim) if dim is not None else ()
    return cleaned.sum() if not dims and not keepdim else cleaned.sum(dim=dims or tuple(range(self.dim())), keepdim=keepdim)


@register_decomposition(ops.isposinf.default)
def isposinf(self):
    if self.is_complex():
        raise TypeError(f"Complex dtype is not supported for isposinf, got dtype {self.dtype}")
    if self.is_floating_point():
        return self == math.inf
    return tp.zeros_like(self, dtype=tp.bool)


@register_decomposition(ops.isneginf.default)
def isneginf(self):
    if self.is_complex():
        raise TypeError(f"Complex dtype is not supported for isneginf, got dtype {self.dtype}")
    if self.is_floating_point():
        return self == -math.inf
    return tp.zeros_like(self, dtype=tp.bool)


@register_decomposition([ops.isin.Tensor_Tensor, ops.isin.Tensor_Scalar, ops.isin.Scalar_Tensor])
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    if not isinstance(elements, tp.Tensor):
        elements = tp.full((), elements, dtype=test_elements.dtype, device=test_elements.device)
    if not isinstance(test_elements, tp.Tensor):
        test_elements = tp.full((1,), test_elements, dtype=elements.dtype, device=elements.device)
    hit = (elements.unsqueeze(-1) == test_elements.reshape(-1)).any(-1)
    return tp.logical_not(hit) if invert else hit


def _var(self, dims, correction, keepdim):
    dims = tuple(range(self.dim())) if dims is None or len(dims) == 0 else tuple(dims)
    count = 1
    for dim in dims:
        count *= int(self.shape[dim])
    mean = self.mean(dim=dims, keepdim=True)
    centered = self - mean
    squared = (centered * centered.conj()).real if self.is_complex() else centered * centered
    total = squared.sum(dim=dims, keepdim=keepdim)
    return total / max(0, count - correction)


def _correction(correction, unbiased=True):
    if correction is None:
        return 1 if unbiased else 0
    return correction


def _std(self, dims, correction, keepdim):
    # Low-precision inputs reduce in float32; complex inputs give real results.
    result_dtype = self.abs().dtype if self.is_complex() else self.dtype
    opmath = _computation_dtype(self.dtype)
    return tp.sqrt(_var(self.to(opmath), dims, correction, keepdim)).to(result_dtype)


@register_decomposition(ops.std.default)
def std(self, correction=1):
    return _std(self, None, correction, False)


@register_decomposition(ops.std.dim)
def std_dim(self, dim, correction=1, keepdim=False):
    return _std(self, dim, correction, keepdim)


@register_decomposition(ops.std.correction)
def std_correction(self, dim=None, *, correction=None, keepdim=False):
    return _std(self, dim, _correction(correction), keepdim)


def _std_mean(self, dims, correction, keepdim):
    std_ = _std(self, dims, correction, keepdim)
    dims = tuple(range(self.dim())) if dims is None or len(dims) == 0 else tuple(dims)
    mean = self.to(_computation_dtype(self.dtype)).mean(dim=dims, keepdim=keepdim)
    return std_, mean.to(self.dtype)


@register_decomposition(ops.std_mean.default)
def std_mean(self, dim=(), unbiased=True, keepdim=False):
    return _std_mean(self, dim, _correction(None, unbiased), keepdim)


@register_decomposition(ops.std_mean.dim)
def std_mean_dim(self, dim, unbiased=True, keepdim=False):
    return _std_mean(self, dim, _correction(None, unbiased), keepdim)


@register_decomposition(ops.std_mean.correction)
def std_mean_correction(self, dim=None, *, correction=None, keepdim=False):
    return _std_mean(self, dim, _correction(correction), keepdim)


# ---------------------------------------------------------------------------
# Activations with in-place forms, matrix products, reductions
# ---------------------------------------------------------------------------


@_register_with_inplace(ops.leaky_relu.default, ops.leaky_relu_.default)
def leaky_relu(self, negative_slope=0.01):
    return tp.where(self > 0, self, self * negative_slope)


@_register_with_inplace(ops.elu.default, ops.elu_.default)
def elu(self, alpha=1, scale=1, input_scale=1):
    negative = tp.expm1(self * input_scale) * (alpha * scale)
    return tp.where(self > 0, self * scale, negative)


@_register_with_inplace(ops.gelu.default, ops.gelu_.default)
def gelu(self, approximate="none"):
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (self + 0.044715 * self * self * self)
        return 0.5 * self * (1 + tp.tanh(inner))
    if approximate != "none":
        raise RuntimeError("approximate argument must be either none or tanh.")
    return 0.5 * self * (1 + tp.erf(self * (1.0 / math.sqrt(2.0))))


@_register_with_inplace(ops.hardtanh.default, ops.hardtanh_.default)
def hardtanh(self, min_val=-1, max_val=1):
    if self.dtype == tp.bool:
        raise NotImplementedError("Bool inputs not supported for hardtanh")
    if not (self.is_floating_point() or self.is_complex()):
        # Integer inputs keep their dtype: the bounds are truncated.
        min_val, max_val = int(min_val), int(max_val)
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    return tp.clamp(self, min_val, max_val)


@register_decomposition(ops.log_sigmoid_forward.default)
def log_sigmoid_forward(self):
    # log(sigmoid(x)) = min(x, 0) - log1p(exp(-|x|)); the buffer keeps
    # exp(-|x|) for the backward formula.
    buffer = tp.exp(-tp.abs(self))
    return tp.minimum(self, tp.zeros_like(self)) - tp.log1p(buffer), buffer


@register_decomposition([ops.log_sigmoid_backward.default])
def log_sigmoid_backward(grad_output, self, buffer=None):
    # An empty buffer means the forward kept nothing; recompute exp(-|x|).
    if buffer is None or buffer.numel() == 0:
        buffer = tp.exp(-tp.abs(self))
    in_negative = self < 0
    max_deriv = tp.where(in_negative, tp.ones_like(self), tp.zeros_like(self))
    sign = tp.where(in_negative, tp.ones_like(self), -tp.ones_like(self))
    return grad_output * (max_deriv - sign * (buffer / (1 + buffer)))


@register_decomposition(ops.glu_backward.default)
def glu_backward(grad_output, self, dim=-1):
    if self.dim() <= 0:
        raise RuntimeError("glu does not support 0-dimensional tensors")
    dim = dim % self.dim()
    if int(self.shape[dim]) % 2 != 0:
        raise RuntimeError(f"Halving dimension must be even, but dimension {dim} is size {int(self.shape[dim])}")
    first, second = tp.chunk(self, 2, dim=dim)
    gate = tp.sigmoid(second)
    grad_first = grad_output * gate
    grad_second = grad_output * first * gate * (1 - gate)
    return tp.cat([grad_first, grad_second], dim)


@_register_with_inplace([ops.floor_divide.default, ops.floor_divide.Scalar],
                        [ops.floor_divide_.Tensor, ops.floor_divide_.Scalar])
def floor_divide(self, other):
    return tp.div(self, other, rounding_mode="floor")


@_register_with_inplace(ops.baddbmm.default, ops.baddbmm_.default)
def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    if not self.is_floating_point() and not self.is_complex():
        beta, alpha = int(beta), int(alpha)
    result = tp.bmm(batch1, batch2)
    if alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
    if beta != 1:
        self = self * beta
    return self + result


@register_decomposition(ops.baddbmm.dtype)
def baddbmm_dtype(self, batch1, batch2, out_dtype, *, beta=1, alpha=1):
    product = tp.bmm(batch1.to(tp.float32), batch2.to(tp.float32))
    result = alpha * product if beta == 0 else beta * self.to(tp.float32) + alpha * product
    return result.to(out_dtype)


@_register_with_inplace(ops.addr.default, ops.addr_.default)
def addr(self, vec1, vec2, *, beta=1, alpha=1):
    if vec1.dim() != 1:
        raise RuntimeError(f"addr: Expected 1-D argument vec1, but got {vec1.dim()}-D")
    if vec2.dim() != 1:
        raise RuntimeError(f"addr: Expected 1-D argument vec2, but got {vec2.dim()}-D")
    all_bool = self.dtype == tp.bool and vec1.dtype == tp.bool and vec2.dtype == tp.bool
    for arg, name in ((alpha, "alpha"), (beta, "beta")):
        if isinstance(arg, bool) and not all_bool:
            raise RuntimeError(f"Boolean {name} only supported for Boolean results.")
    self = self.expand(int(vec1.shape[0]), int(vec2.shape[0]))
    outer = vec1.unsqueeze(-1) * vec2.unsqueeze(0)
    if self.dtype == tp.bool:
        for arg, name in ((beta, "beta"), (alpha, "alpha")):
            if not isinstance(arg, (bool, int)):
                raise RuntimeError(f"expected bool/int {name} but got {type(arg)}")
        product = outer if alpha else tp.full_like(self, False)
        return product if not beta else tp.logical_or(self, product)
    if beta == 0:
        return alpha * outer
    return beta * self + alpha * outer


@register_decomposition([ops.all.default, ops.all.dim, ops.all.dims])
def all_(self, dim=None, keepdim=False):
    as_bool = self if self.dtype == tp.bool else self != 0
    failing = tp.logical_not(as_bool).to(tp.int64)
    dims = () if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    if dims:
        result = failing.sum(dim=dims, keepdim=keepdim) == 0
    else:
        result = failing.sum() == 0
        result = result.reshape((1,) * self.dim()) if keepdim else result
    return result.to(tp.uint8) if self.dtype == tp.uint8 else result


@register_decomposition(ops.aminmax.default)
def aminmax(self, dim=(), keepdim=False):
    if isinstance(dim, int):
        dim = (dim,)
    dims = tuple(dim) if dim is not None else ()
    if not dims:
        return self.amin(), self.amax()
    return self.amin(dim=dims, keepdim=keepdim), self.amax(dim=dims, keepdim=keepdim)


@register_decomposition(ops.linalg_cross.default)
def linalg_cross(self, other, *, dim=-1):
    if self.dim() != other.dim():
        raise RuntimeError("linalg.cross: inputs must have the same number of dimensions.")
    if int(self.shape[dim]) != 3 or int(other.shape[dim]) != 3:
        raise RuntimeError(
            f"linalg.cross: inputs dim {dim} must have length 3, got {int(self.shape[dim])} and {int(other.shape[dim])}"
        )
    self, other = tp.broadcast_tensors(self, other)
    dim = dim % self.dim()
    a0, a1, a2 = (tp.select(self, dim, i) for i in range(3))
    b0, b1, b2 = (tp.select(other, dim, i) for i in range(3))
    return tp.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dim)


@_register_with_inplace(ops.mvlgamma.default, ops.mvlgamma_.default)
def mvlgamma(self, p):
    if p < 1:
        raise RuntimeError(f"p has to be greater than or equal to 1, but got {p}")
    offsets = tp.arange(p, dtype=self.dtype, device=self.device) * -0.5
    terms = tp.lgamma(self.unsqueeze(-1) + offsets)
    return terms.sum(-1) + (p * (p - 1) / 4.0) * math.log(math.pi)


@_register_with_inplace(ops.renorm.default, ops.renorm_.default)
def renorm(self, p, dim, maxnorm):
    if isinstance(p, complex):
        raise RuntimeError("renorm: p must be real-valued")
    if p <= 0:
        raise RuntimeError("renorm: non-positive norm not supported")
    if isinstance(maxnorm, complex):
        raise RuntimeError("renorm: maxnorm must be real-valued")
    if maxnorm < 0:
        raise RuntimeError(f"renorm: expected maxnorm to be >= 0 but got {maxnorm}")
    if self.dim() <= 1:
        raise RuntimeError(f"renorm: input needs at least 2 dimensions, got {self.dim()} dimensions")
    dim = dim % self.dim()
    reduce = [d for d in range(self.dim()) if d != dim]
    acc = _computation_dtype(self.dtype)
    norms = ops.linalg_vector_norm.default(self, p, reduce, True, dtype=acc if acc != self.dtype else None)
    factor = tp.where(norms > maxnorm, maxnorm / (norms + 1e-7), tp.ones_like(norms))
    return (self * factor.to(self.dtype)).contiguous()


@register_decomposition(ops._lazy_clone.default)
def _lazy_clone(self):
    return self.clone()


@register_decomposition([ops.special_xlog1py.default, ops.special_xlog1py.other_scalar, ops.special_xlog1py.self_scalar])
def special_xlog1py(self, other):
    if not isinstance(self, tp.Tensor):
        self = tp.full((), self, dtype=other.dtype, device=other.device)
    if not isinstance(other, tp.Tensor):
        other = tp.full((), other, dtype=self.dtype, device=self.device)
    product = self * tp.log1p(other)
    zero = tp.zeros((), dtype=product.dtype, device=product.device)
    nan = tp.full((), float("nan"), dtype=product.dtype, device=product.device)
    return tp.where(tp.isnan(other), nan, tp.where(self == 0, zero, product))


@register_decomposition(ops.special_entr.default)
def special_entr(self):
    negative = tp.full_like(self, -math.inf)
    value = -self * tp.log(self)
    result = tp.where(self == 0, tp.zeros_like(self), value)
    result = tp.where(self < 0, negative, result)
    return tp.where(tp.isnan(self), self, result)


@register_decomposition(ops.special_log_ndtr.default)
def special_log_ndtr(self):
    scaled = self * (1.0 / math.sqrt(2.0))
    upper = tp.log1p(-tp.special.erfc(scaled) * 0.5)
    lower = tp.log(tp.special.erfcx(-scaled) * 0.5) - scaled * scaled
    return tp.where(self < 1.0, lower, upper)


# ---------------------------------------------------------------------------
# Softmax backward formulas
# ---------------------------------------------------------------------------


@register_decomposition(ops._softmax_backward_data.default)
def _softmax_backward_data(grad_output, output, dim, input_dtype):
    product = grad_output * output
    result = product - output * product.sum(dim=dim, keepdim=True)
    return (result.to(input_dtype) if grad_output.dtype != input_dtype else result).contiguous()


@register_decomposition(ops._log_softmax_backward_data.default)
def _log_softmax_backward_data(grad_output, output, dim, input_dtype):
    result = grad_output - tp.exp(output) * grad_output.sum(dim=dim, keepdim=True)
    return result.to(input_dtype) if grad_output.dtype != input_dtype else result


@register_decomposition(ops._safe_softmax.default)
def _safe_softmax(self, dim, dtype=None):
    source = self.to(dtype) if dtype is not None else self
    # Rows that are entirely -inf produce zeros instead of NaN.
    all_masked = (source == -math.inf).all(dim=dim, keepdim=True)
    softmax = tp.softmax(source, dim)
    return tp.where(all_masked, tp.zeros_like(softmax), softmax)


# ---------------------------------------------------------------------------
# Losses (reduction: 0 none, 1 mean, 2 sum)
# ---------------------------------------------------------------------------


def _reduce(loss, reduction):
    if reduction == 1:
        return loss.mean()
    if reduction == 2:
        return loss.sum()
    return loss


def _reduction_scale(grad_output, numel, reduction):
    return grad_output / numel if reduction == 1 else grad_output


@register_decomposition(ops.mse_loss.default)
def mse_loss(self, target, reduction=1):
    difference = self - target
    return _reduce(difference * difference, reduction)


@register_decomposition(ops.mse_loss_backward.default)
def mse_loss_backward(grad_output, self, target, reduction=1):
    scale = 2.0 / self.numel() if reduction == 1 else 2.0
    return scale * (self - target) * grad_output


@register_decomposition(ops.l1_loss.default)
def l1_loss(input, target):
    return tp.abs(input - target).mean()


@register_decomposition(ops.smooth_l1_loss.default)
def smooth_l1_loss(self, target, reduction=1, beta=1.0):
    difference = tp.abs(self - target)
    if beta == 0:
        return _reduce(difference, reduction)
    loss = tp.where(difference < beta, 0.5 * difference * difference / beta, difference - 0.5 * beta)
    return _reduce(loss, reduction)


@register_decomposition(ops.smooth_l1_loss_backward.default)
def smooth_l1_loss_backward(grad_output, self, target, reduction, beta):
    norm = 1.0 / self.numel() if reduction == 1 else 1.0
    x = self - target
    norm_grad = norm * grad_output
    inner = norm_grad * x / beta if beta != 0 else tp.zeros_like(x)
    return tp.where(tp.abs(x) < beta, inner, norm_grad * tp.sign(x))


@register_decomposition(ops.huber_loss.default)
def huber_loss(self, target, reduction=1, delta=1.0):
    if delta <= 0:
        raise RuntimeError("huber_loss does not support non-positive values for delta.")
    difference = tp.abs(self - target)
    loss = tp.where(difference < delta, 0.5 * difference * difference, delta * (difference - 0.5 * delta))
    return _reduce(loss, reduction)


@register_decomposition(ops.huber_loss_backward.default)
def huber_loss_backward(grad_output, self, target, reduction, delta):
    norm = 1.0 / self.numel() if reduction == 1 else 1.0
    x = self - target
    return tp.where(
        x < -delta,
        -norm * grad_output * delta,
        tp.where(x > delta, norm * grad_output * delta, norm * x * grad_output),
    )


@register_decomposition(ops.binary_cross_entropy.default)
def binary_cross_entropy(self, target, weight=None, reduction=1):
    # Logarithms are clamped at -100 so saturated probabilities stay finite.
    floor = tp.full((), -100, dtype=self.dtype, device=self.device)
    loss = (target - 1) * tp.maximum(tp.log1p(-self), floor) - target * tp.maximum(tp.log(self), floor)
    if weight is not None:
        loss = loss * weight
    return _reduce(loss, reduction)


@register_decomposition(ops.binary_cross_entropy_backward.default)
def binary_cross_entropy_backward(grad_output, self, target, weight=None, reduction=1):
    epsilon = 1e-12
    grad = grad_output * (self - target) / tp.clamp(self * (1 - self), min=epsilon)
    if weight is not None:
        grad = grad * weight
    return grad / self.numel() if reduction == 1 else grad


@register_decomposition(ops.binary_cross_entropy_with_logits.default)
def binary_cross_entropy_with_logits(self, target, weight=None, pos_weight=None):
    if pos_weight is not None:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * self - log_weight * tp.nn.functional.logsigmoid(self)
    else:
        loss = (1 - target) * self - tp.nn.functional.logsigmoid(self)
    if weight is not None:
        loss = loss * weight
    return loss.to(target.dtype).mean()


@register_decomposition(ops.soft_margin_loss.default)
def soft_margin_loss(input, target):
    return tp.log1p(tp.exp(-input * target)).mean()


@register_decomposition(ops.soft_margin_loss_backward.default)
def soft_margin_loss_backward(grad_output, self, target, reduction):
    grad_input = target * grad_output * (tp.sigmoid(target * self) - 1)
    return grad_input / self.numel() if reduction == 1 else grad_input


# ---------------------------------------------------------------------------
# Layout shuffles, PReLU, weight norm, chunked concatenation
# ---------------------------------------------------------------------------


@register_decomposition(ops.pixel_shuffle.default)
def pixel_shuffle(self, upscale_factor):
    if self.dim() < 3:
        raise RuntimeError(
            f"pixel_shuffle expects input to have at least 3 dimensions, but got input with {self.dim()} dimension(s)"
        )
    r = upscale_factor
    *batch, c, h, w = (int(s) for s in self.shape)
    if c % (r * r):
        raise RuntimeError(
            "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of upscale_factor"
        )
    oc = c // (r * r)
    x = self.reshape(tuple(batch) + (oc, r, r, h, w))
    n = len(batch)
    x = x.permute(tuple(range(n)) + (n, n + 3, n + 1, n + 4, n + 2))
    return x.reshape(tuple(batch) + (oc, h * r, w * r))


@register_decomposition(ops.pixel_unshuffle.default)
def pixel_unshuffle(self, downscale_factor):
    if self.dim() < 3:
        raise RuntimeError(
            f"pixel_unshuffle expects input to have at least 3 dimensions, but got input with {self.dim()} dimension(s)"
        )
    r = downscale_factor
    *batch, c, h, w = (int(s) for s in self.shape)
    if h % r or w % r:
        raise RuntimeError("pixel_unshuffle expects height and width to be divisible by downscale_factor")
    x = self.reshape(tuple(batch) + (c, h // r, r, w // r, r))
    n = len(batch)
    x = x.permute(tuple(range(n)) + (n, n + 2, n + 4, n + 1, n + 3))
    return x.reshape(tuple(batch) + (c * r * r, h // r, w // r))


@register_decomposition(ops.channel_shuffle.default)
def channel_shuffle(self, groups):
    if self.dim() <= 2:
        raise RuntimeError(
            f"channel_shuffle expects input with > 2 dims, but got input with sizes {list(self.shape)}"
        )
    n, c = int(self.shape[0]), int(self.shape[1])
    if groups <= 0:
        raise RuntimeError(f"Number of groups to divide channels in must be positive. Value of groups:{groups}")
    if c % groups:
        raise RuntimeError(f"Number of channels must be divisible by groups. Got {c} channels and {groups} groups.")
    if self.numel() == 0:
        return ops.alias.default(self)
    rest = tuple(int(s) for s in self.shape[2:])
    x = self.reshape((n, groups, c // groups) + rest)
    return x.transpose(1, 2).reshape((n, c) + rest).contiguous()


@register_decomposition(ops._prelu_kernel.default)
def _prelu_kernel(self, weight):
    # ``weight`` arrives already broadcastable against ``self``.
    return tp.where(self > 0, self, weight * self)


@register_decomposition(ops._prelu_kernel_backward.default)
def _prelu_kernel_backward(grad_output, self, weight):
    # Both gradients come back in the broadcast shape; the caller reduces
    # the weight gradient to the weight's own shape.
    grad_input = tp.where(self > 0, grad_output, weight * grad_output)
    grad_weight = tp.where(self > 0, tp.zeros_like(self), self * grad_output)
    return grad_input, grad_weight


@register_decomposition(ops._weight_norm_interface.default)
def _weight_norm_interface(v, g, dim=0):
    keep = [d for d in range(v.dim()) if d != dim % max(v.dim(), 1)]
    norm_dtype = tp.float32 if g.dtype == tp.bfloat16 else None
    norm = ops.linalg_vector_norm.default(v, 2, keep, True, dtype=norm_dtype)
    return v * (g / norm.to(g.dtype)), norm


@register_decomposition(ops._chunk_cat.default)
def _chunk_cat(tensors, dim, num_chunks):
    pieces = []
    for tensor in tensors:
        d = dim % max(tensor.dim(), 1)
        size = int(tensor.shape[d])
        chunk = -(-size // num_chunks)
        padded = size if size % num_chunks == 0 else chunk * num_chunks
        if padded != size:
            pad_shape = list(tensor.shape)
            pad_shape[d] = padded - size
            tensor = tp.cat([tensor, tp.zeros(tuple(pad_shape), dtype=tensor.dtype, device=tensor.device)], d)
        shape = list(tensor.shape[:d]) + [num_chunks, -1]
        pieces.append(tensor.reshape(tuple(shape)))
    return tp.cat(pieces, dim + 1 if dim >= 0 else dim)


@register_decomposition(ops.embedding_dense_backward.default)
def embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    result_dtype = grad_output.dtype
    grad_output = grad_output.to(_computation_dtype(result_dtype))
    indices = indices.to(tp.int64)
    # Out-of-range indices contribute nothing.
    valid = (indices >= 0) & (indices < num_weights)
    if scale_grad_by_freq:
        counts = tp.zeros((num_weights,), dtype=indices.dtype, device=indices.device)
        counts = _unsafe_masked_index_put_accumulate(counts, valid, [indices], tp.ones_like(indices))
        scale = _unsafe_masked_index(counts, valid, [indices], 1)
        grad_output = grad_output / scale.unsqueeze(-1)
    mask = valid & (indices != padding_idx)
    mask = mask.reshape(tuple(mask.shape) + (1,) * (grad_output.dim() - mask.dim()))
    trailing = tuple(int(s) for s in grad_output.shape[indices.dim():])
    grad_weight = tp.zeros((num_weights,) + trailing, dtype=grad_output.dtype, device=grad_output.device)
    return _unsafe_masked_index_put_accumulate(grad_weight, mask, [indices], grad_output).to(result_dtype)


# ---------------------------------------------------------------------------
# Normalization backward formulas
# ---------------------------------------------------------------------------


def _computation_dtype(dtype):
    """Reduced-precision floats compute in float32 and round once at the end."""

    return tp.float32 if dtype in (tp.float16, tp.bfloat16) else dtype


def _cast(value, dtype):
    return None if value is None else value.to(dtype)


def _to_rank(value, ndim):
    while value.dim() < ndim:
        value = value.unsqueeze(-1)
    return value


def _stat_shape(stat, input, axis):
    """Per-row statistics shaped to broadcast over the normalized dims.

    Row statistics arrive either kept-dimension (outer dims then ones) or
    flattened to one value per row; both hold one value per outer index.
    """

    shape = tuple(int(s) for s in input.shape[:axis]) + (1,) * (input.dim() - axis)
    return stat.reshape(shape)


@register_decomposition(ops.native_batch_norm_backward.default)
def native_batch_norm_backward(grad_out, input, weight, running_mean, running_var,
                               save_mean, save_invstd, train, eps, output_mask):
    if input.dim() < 2:
        raise RuntimeError(f"rank of the input must be at least 2, got {input.dim()}")
    compute = _computation_dtype(input.dtype)
    weight_dtype = weight.dtype if weight is not None else input.dtype
    grad_out_c, input_c, weight_c = (_cast(x, compute) for x in (grad_out, input, weight))
    if train:
        mean, invstd = _cast(save_mean, compute), _cast(save_invstd, compute)
    else:
        mean = _cast(running_mean, compute)
        invstd = tp.rsqrt(_cast(running_var, compute) + eps)
    channels = int(input.shape[1])
    broadcast = [1] * input.dim()
    broadcast[1] = channels
    reduce = [d for d in range(input.dim()) if d != 1]
    count = input.numel() / channels
    mean_b = mean.reshape(broadcast)
    grad_output_sum = grad_out_c.sum(dim=reduce)
    dot_p = (grad_out_c * (input_c - mean_b)).sum(dim=reduce)
    grad_mean = (grad_output_sum / count).reshape(broadcast)
    proj_scale = (dot_p / count * invstd * invstd).reshape(broadcast)
    scale = invstd if weight_c is None else invstd * weight_c
    grad_scale = scale.reshape(broadcast)
    if train:
        grad_input = ((grad_out_c - (input_c - mean_b) * proj_scale) - grad_mean) * grad_scale
    else:
        grad_input = grad_out_c * grad_scale
    grad_weight = dot_p * invstd if output_mask[1] else None
    grad_bias = grad_output_sum if output_mask[2] else None
    return (
        grad_input.to(input.dtype) if output_mask[0] else None,
        _cast(grad_weight, weight_dtype),
        _cast(grad_bias, weight_dtype),
    )


@register_decomposition(ops.cudnn_batch_norm_backward.default)
def cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var,
                              save_mean, save_var, epsilon, reserveSpace):
    return native_batch_norm_backward(
        grad_output, input, weight, running_mean, running_var, save_mean, save_var,
        True, epsilon, [True, True, True],
    )


@register_decomposition(ops.miopen_batch_norm_backward.default)
def miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var,
                               save_mean, save_var, epsilon):
    return native_batch_norm_backward(
        grad_output, input, weight, running_mean, running_var, save_mean, save_var,
        True, epsilon, [True, True, True],
    )


@register_decomposition(ops.cudnn_batch_norm.default)
def cudnn_batch_norm(input, weight, bias, running_mean, running_var, training,
                     exponential_average_factor, epsilon):
    output, save_mean, save_invstd = ops.native_batch_norm.default(
        input, weight, bias, running_mean, running_var, training,
        exponential_average_factor, epsilon,
    )
    reserve = tp.empty(0, dtype=tp.uint8, device=input.device)
    if not training:
        # Inference keeps no statistics for a backward pass.
        save_mean = tp.empty(0, dtype=input.dtype, device=input.device)
        save_invstd = tp.empty(0, dtype=input.dtype, device=input.device)
    return output, save_mean, save_invstd, reserve


@register_decomposition(ops.native_layer_norm_backward.default)
def native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight,
                               bias, output_mask):
    compute = _computation_dtype(input.dtype)
    grad_out_c, input_c, weight_c = (_cast(x, compute) for x in (grad_out, input, weight))
    axis = input.dim() - len(normalized_shape)
    inner = list(range(axis, input.dim()))
    outer = list(range(axis))
    n_inner = 1
    for size in input.shape[axis:]:
        n_inner *= int(size)
    n_outer = 1
    for size in input.shape[:axis]:
        n_outer *= int(size)
    if n_inner == 0 or n_outer == 0:
        return (
            tp.zeros_like(input) if output_mask[0] else None,
            input.new_zeros(tuple(input.shape[axis:])) if output_mask[1] else None,
            input.new_zeros(tuple(input.shape[axis:])) if output_mask[2] else None,
        )
    mean = _stat_shape(mean.to(compute), input, axis)
    rstd = _stat_shape(rstd.to(compute), input, axis)
    x_hat = (input_c - mean) * rstd
    grad_x_hat = grad_out_c * weight_c if weight_c is not None else grad_out_c
    a = grad_x_hat * n_inner
    b = grad_x_hat.sum(dim=inner, keepdim=True)
    c = x_hat * (grad_x_hat * x_hat).sum(dim=inner, keepdim=True)
    d_input = (rstd / n_inner) * (a - b - c) if output_mask[0] else None
    d_weight = d_bias = None
    if output_mask[1] and weight_c is not None:
        product = grad_out_c * x_hat
        d_weight = product.sum(dim=outer) if outer else product
    if output_mask[2] and bias is not None:
        d_bias = grad_out_c.sum(dim=outer) if outer else grad_out_c.clone()
    return (
        _cast(d_input, input.dtype),
        _cast(d_weight, weight.dtype if weight is not None else input.dtype),
        _cast(d_bias, bias.dtype if bias is not None else input.dtype),
    )


@register_decomposition(ops.native_group_norm_backward.default)
def native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask):
    compute = _computation_dtype(input.dtype)
    grad_out_c, input_c = grad_out.to(compute), input.to(compute)
    mean_c, rstd_c = mean.to(compute), rstd.to(compute)
    weight_c = _cast(weight, compute)
    cpg = C // group
    if C != cpg * group:
        raise RuntimeError(
            f"Expect number of channels {C} to be evenly-divisible by number of groups {group}"
        )
    ds = (grad_out_c * input_c).reshape(N, C, HxW).sum(dim=[2])
    db = grad_out_c.reshape(N, C, HxW).sum(dim=[2])
    d_input = d_gamma = d_bias = None
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        if weight_c is not None:
            ds_val = (ds * weight_c.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = (db * weight_c.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = rstd_c.unsqueeze(-1) * weight_c.reshape(1, group, cpg)
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = rstd_c.unsqueeze(-1) * tp.ones((1, group, cpg), dtype=compute, device=rstd.device)
        c2 = (db_val * mean_c - ds_val) * rstd_c * rstd_c * rstd_c * s
        c3 = -c2 * mean_c - db_val * rstd_c * s
        d_input = (
            grad_out_c.reshape(N, group, cpg, HxW) * c1.unsqueeze(-1)
            + input_c.reshape(N, group, cpg, HxW) * _to_rank(c2, 4)
            + _to_rank(c3, 4)
        ).reshape(tuple(input.shape)).to(input.dtype)
    if output_mask[1]:
        d_gamma = (
            ((ds.reshape(N, group, cpg) - db.reshape(N, group, cpg) * mean_c.unsqueeze(-1))
             * rstd_c.unsqueeze(-1)).sum(dim=[0]).reshape(C)
        ).to(weight.dtype if weight is not None else input.dtype)
    if output_mask[2]:
        d_bias = db.sum(dim=[0]).to(weight.dtype if weight is not None else input.dtype)
    return d_input, d_gamma, d_bias


@register_decomposition(ops._fused_rms_norm.default)
def _fused_rms_norm(input, normalized_shape, weight, eps):
    compute = _computation_dtype(input.dtype)
    axis = input.dim() - len(normalized_shape)
    dims = list(range(axis, input.dim()))
    x = input.to(compute)
    eps = tp.finfo(compute).eps if eps is None else eps
    rstd = tp.rsqrt((x * x).mean(dim=dims, keepdim=True) + eps)
    output = x * rstd
    if weight is not None:
        output = output * weight.to(compute)
    return output.to(input.dtype), rstd


@register_decomposition(ops._fused_rms_norm_backward.default)
def _fused_rms_norm_backward(grad_out, input, normalized_shape, rstd, weight, output_mask):
    compute = _computation_dtype(input.dtype)
    grad_out_c, input_c, weight_c = (_cast(x, compute) for x in (grad_out, input, weight))
    axis = input.dim() - len(normalized_shape)
    inner = list(range(axis, input.dim()))
    outer = list(range(axis))
    n_inner = 1
    for size in input.shape[axis:]:
        n_inner *= int(size)
    rstd = _stat_shape(rstd.to(compute), input, axis)
    grad_x_hat = grad_out_c * weight_c if weight_c is not None else grad_out_c
    x_hat = input_c * rstd
    d_input = d_weight = None
    if output_mask[0]:
        total = (x_hat * grad_x_hat).sum(dim=inner, keepdim=True)
        d_input = ((grad_x_hat - (x_hat / n_inner) * total) * rstd).to(input.dtype)
    if output_mask[1] and weight_c is not None:
        product = grad_out_c * x_hat
        d_weight = (product.sum(dim=outer) if outer else product).to(weight.dtype)
    return d_input, d_weight


@register_decomposition(ops.native_dropout_backward.default)
def native_dropout_backward(grad_output, mask, scale):
    return grad_output * (mask.to(grad_output.dtype) * scale)


# ---------------------------------------------------------------------------
# Negative log likelihood
# ---------------------------------------------------------------------------


def _nll_forward(self, target, weight, reduction, ignore_index):
    ndim = self.dim()
    if not 0 < ndim <= 2:
        raise RuntimeError(f"input tensor should be 1D or 2D, got {ndim}D")
    channel_dim = 1 if ndim > 1 else 0
    w = None
    if weight is not None:
        shape = [1] * ndim
        shape[channel_dim] = int(weight.shape[0])
        w = weight.reshape(tuple(shape)) if ndim > 1 else weight
        self = self * w
    keep = target != ignore_index
    safe_target = tp.where(keep, target, tp.zeros_like(target))
    gathered = tp.gather(self, channel_dim, safe_target.unsqueeze(channel_dim)).squeeze(channel_dim)
    result = tp.where(keep, -gathered, tp.zeros_like(gathered))
    if reduction == 0 and ndim > 1:
        return result, tp.zeros((), dtype=self.dtype, device=self.device)
    if w is not None:
        wsum = tp.gather(w.expand(tuple(self.shape)), channel_dim,
                         safe_target.unsqueeze(channel_dim)).squeeze(channel_dim)
        total_weight = tp.where(keep, wsum, tp.zeros_like(wsum)).sum()
    else:
        total_weight = keep.sum().to(self.dtype)
    if reduction == 2:
        result = result.sum()
    elif reduction == 1:
        result = result.sum() / total_weight
    return result, total_weight


@register_decomposition(ops.nll_loss_forward.default)
def nll_loss_forward(self, target, weight, reduction, ignore_index):
    return _nll_forward(self, target, weight, reduction, ignore_index)


@register_decomposition(ops.nll_loss2d_forward.default)
def nll_loss2d_forward(self, target, weight, reduction, ignore_index):
    # (N, C, H, W) scores against (N, H, W) targets: classes move last and
    # the spatial positions become batch entries.
    channels = int(self.shape[1])
    flat_self = self.movedim(1, -1).reshape(-1, channels)
    flat_target = target.reshape(-1)
    output, total_weight = _nll_forward(flat_self, flat_target, weight, reduction, ignore_index)
    if reduction == 0:
        output = output.reshape(tuple(target.shape))
    return output, total_weight


def _nll_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight):
    channel_dim = 0 if self.dim() < 2 else 1
    if reduction == 1:
        grad_output = grad_output / total_weight
    if self.dim() == 1 and target.dim() > 0:
        target = target[0]
    target = target.unsqueeze(channel_dim)
    keep = target != ignore_index
    safe_target = tp.where(keep, target, tp.zeros_like(target))
    grad_input = tp.zeros_like(self).scatter(channel_dim, safe_target, -1.0)
    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)
    if weight is not None:
        shape = [1] * self.dim()
        shape[channel_dim] = int(weight.shape[0])
        grad_output = grad_output * weight.reshape(tuple(shape))
    grad_output = tp.where(keep, grad_output, tp.zeros_like(grad_output))
    return grad_input * grad_output


@register_decomposition(ops.nll_loss_backward.default)
def nll_loss_backward(grad_output, self, target, weight=None, reduction=1, ignore_index=-100,
                      total_weight=None):
    if total_weight is None:
        _, total_weight = _nll_forward(self, target, weight, reduction, ignore_index)
    return _nll_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight)


@register_decomposition(ops.nll_loss2d_backward.default)
def nll_loss2d_backward(grad_output, self, target, weight=None, reduction=1, ignore_index=-100,
                        total_weight=None):
    channels = int(self.shape[1])
    flat_self = self.movedim(1, -1).reshape(-1, channels)
    flat_target = target.reshape(-1)
    if total_weight is None:
        _, total_weight = _nll_forward(flat_self, flat_target, weight, reduction, ignore_index)
    flat_grad = grad_output.reshape(-1) if reduction == 0 else grad_output
    grad = _nll_backward(flat_grad, flat_self, flat_target, weight, reduction, ignore_index, total_weight)
    spatial = tuple(target.shape) + (channels,)
    return grad.reshape(spatial).movedim(-1, 1)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def _pad_indices(padding, dims, sizes, device, reflect):
    """Source index per output position along each padded dim.

    ``padding`` lists (left, right) pairs starting from the last dimension.
    Reflection folds back at the border element; replication repeats it.
    """

    indices = []
    for i in range(dims):
        left = padding[2 * (dims - 1 - i)]
        right = padding[2 * (dims - 1 - i) + 1]
        size = int(sizes[i])
        position = tp.arange(-left, size + right, device=device)
        if reflect:
            if left >= size or right >= size:
                raise RuntimeError(
                    f"Argument #4: Padding size should be less than the corresponding input dimension, "
                    f"but got: padding ({left}, {right}) at dimension {i} of input of size {size}"
                )
            span = size - 1
            folded = tp.abs(position)
            source = span - tp.abs(folded - span)
        else:
            source = tp.clamp(position, 0, size - 1)
        indices.append(source)
    return indices


def _pad_forward(self, padding, reflect):
    dims = len(padding) // 2
    if self.dim() not in (dims + 1, dims + 2):
        kind = "reflection" if reflect else "replication"
        raise RuntimeError(f"{kind}_pad{dims}d requires {dims + 1}D or {dims + 2}D input")
    offset = self.dim() - dims
    result = self
    for i, index in enumerate(_pad_indices(padding, dims, self.shape[offset:], self.device, reflect)):
        result = tp.index_select(result, offset + i, index)
    return result.contiguous()


def _pad_backward(grad_output, self, padding, reflect):
    dims = len(padding) // 2
    offset = self.dim() - dims
    indices = _pad_indices(padding, dims, self.shape[offset:], self.device, reflect)
    grad = grad_output
    shape = list(grad_output.shape)
    for i in reversed(range(dims)):
        shape[offset + i] = int(self.shape[offset + i])
        zeros_ = tp.zeros(tuple(shape), dtype=grad.dtype, device=grad.device)
        grad = zeros_.index_add(offset + i, indices[i], grad)
    return grad


@register_decomposition([ops.reflection_pad1d.default, ops.reflection_pad2d.default, ops.reflection_pad3d.default])
def reflection_pad(self, padding):
    return _pad_forward(self, list(padding), True)


@register_decomposition([ops.replication_pad1d.default, ops.replication_pad2d.default, ops.replication_pad3d.default])
def replication_pad(self, padding):
    return _pad_forward(self, list(padding), False)


@register_decomposition([
    ops.reflection_pad1d_backward.default,
    ops.reflection_pad2d_backward.default,
    ops.reflection_pad3d_backward.default,
])
def reflection_pad_backward(grad_output, self, padding):
    return _pad_backward(grad_output, self, list(padding), True)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _upsample_scale(in_size, out_size, align_corners, scale):
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0.0
    return 1.0 / scale if scale is not None and scale > 0 else in_size / out_size


def _upsample_output_size(input, output_size, scale_factors):
    spatial = [int(s) for s in input.shape[2:]]
    if output_size is not None:
        if scale_factors is not None:
            raise RuntimeError("Must specify exactly one of output_size and scale_factors")
        return list(output_size)
    if scale_factors is None:
        raise RuntimeError("Must specify exactly one of output_size and scale_factors")
    return [int(math.floor(size * float(factor))) for size, factor in zip(spatial, scale_factors)]


def _upsample_linear(input, output_size, align_corners, scales):
    dtype = input.dtype if input.is_floating_point() else tp.get_default_dtype()
    result = input.to(dtype)
    spatial = input.dim() - 2
    for d in range(spatial):
        axis = 2 + d
        in_size, out_size = int(input.shape[axis]), int(output_size[d])
        scale = _upsample_scale(in_size, out_size, align_corners, scales[d])
        position = tp.arange(out_size, device=input.device).to(dtype)
        source = position * scale if align_corners else (position + 0.5) * scale - 0.5
        source = tp.clamp(source, min=0.0)
        low = source.to(tp.int64)
        high = tp.clamp(low + 1, max=in_size - 1)
        weight = tp.clamp(source - low.to(dtype), 0.0, 1.0)
        shape = [1] * input.dim()
        shape[axis] = out_size
        weight = weight.reshape(tuple(shape))
        first = tp.index_select(result, axis, low)
        second = tp.index_select(result, axis, high)
        result = first + (second - first) * weight
    return result


@register_decomposition(ops.upsample_linear1d.default)
def upsample_linear1d(self, output_size, align_corners, scales=None):
    return _upsample_linear(self, list(output_size), align_corners, [scales])


@register_decomposition(ops.upsample_bilinear2d.default)
def upsample_bilinear2d(self, output_size, align_corners, scales_h=None, scales_w=None):
    return _upsample_linear(self, list(output_size), align_corners, [scales_h, scales_w])


@register_decomposition(ops.upsample_trilinear3d.default)
def upsample_trilinear3d(self, output_size, align_corners, scales_d=None, scales_h=None, scales_w=None):
    return _upsample_linear(self, list(output_size), align_corners, [scales_d, scales_h, scales_w])


@register_decomposition([ops.upsample_linear1d.vec, ops.upsample_bilinear2d.vec, ops.upsample_trilinear3d.vec])
def upsample_linear_vec(input, output_size, align_corners, scale_factors):
    size = _upsample_output_size(input, output_size, scale_factors)
    scales = list(scale_factors) if scale_factors else [None] * len(size)
    return _upsample_linear(input, size, align_corners, scales)


def _nearest_source(in_size, out_size, scale, device):
    step = in_size / (in_size * scale) if scale is not None and scale > 0 else in_size / out_size
    position = tp.arange(out_size, dtype=tp.float32, device=device)
    return (position * step).to(tp.int64)


@register_decomposition(ops.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(grad_output, output_size, input_size, scales_h=None, scales_w=None):
    grad = grad_output
    scales = [scales_h, scales_w]
    for d in reversed(range(2)):
        axis = grad.dim() - 2 + d
        source = _nearest_source(int(input_size[axis]), int(output_size[d]), scales[d], grad.device)
        shape = list(grad.shape)
        shape[axis] = int(input_size[axis])
        grad = tp.zeros(tuple(shape), dtype=grad.dtype, device=grad.device).index_add(axis, source, grad)
    return grad


# ---------------------------------------------------------------------------
# Sliding blocks and unpooling
# ---------------------------------------------------------------------------


def _block_indices(size, kernel, dilation, padding, stride, device):
    blocks = size + 2 * padding - dilation * (kernel - 1)
    starts = tp.arange(0, blocks, stride, device=device).unsqueeze(0)
    offsets = tp.arange(0, kernel * dilation, dilation, device=device).unsqueeze(-1)
    return starts + offsets  # (kernel, blocks)


def _check_positive(values, name, strict=True):
    ok = all(v > 0 for v in values) if strict else all(v >= 0 for v in values)
    if not ok:
        raise RuntimeError(f"{name} should be greater than zero, but got {list(values)}")


@register_decomposition(ops.im2col.default)
def im2col(self, kernel_size, dilation=(), padding=(), stride=()):
    kernel_size, dilation = list(kernel_size), list(dilation) or [1, 1]
    padding, stride = list(padding) or [0, 0], list(stride) or [1, 1]
    _check_positive(kernel_size, "kernel_size")
    _check_positive(dilation, "dilation")
    _check_positive(padding, "padding", strict=False)
    _check_positive(stride, "stride")
    batched = self.dim() == 4
    x = self if batched else self.unsqueeze(0)
    n, c, h, w = (int(s) for s in x.shape)
    rows = _block_indices(h, kernel_size[0], dilation[0], padding[0], stride[0], x.device)
    cols = _block_indices(w, kernel_size[1], dilation[1], padding[1], stride[1], x.device)
    padded = tp.nn.functional.pad(x, (padding[1], padding[1], padding[0], padding[0]))
    # Gather rows then columns: (n, c, kh, bh, W) -> (n, c, kh, bh, kw, bw).
    kh, bh = int(rows.shape[0]), int(rows.shape[1])
    kw, bw = int(cols.shape[0]), int(cols.shape[1])
    gathered = tp.index_select(padded, 2, rows.reshape(-1)).reshape(n, c, kh, bh, -1)
    gathered = tp.index_select(gathered, 4, cols.reshape(-1)).reshape(n, c, kh, bh, kw, bw)
    output = gathered.permute(0, 1, 2, 4, 3, 5).reshape(n, c * kh * kw, bh * bw)
    return output if batched else output.squeeze(0)


@register_decomposition(ops.col2im.default)
def col2im(self, output_size, kernel_size, dilation=(), padding=(), stride=()):
    output_size, kernel_size = list(output_size), list(kernel_size)
    dilation, padding = list(dilation) or [1, 1], list(padding) or [0, 0]
    stride = list(stride) or [1, 1]
    _check_positive(kernel_size, "kernel_size")
    _check_positive(dilation, "dilation")
    _check_positive(padding, "padding", strict=False)
    _check_positive(stride, "stride")
    _check_positive(output_size, "output_size")
    batched = self.dim() == 3
    x = self if batched else self.unsqueeze(0)
    kh, kw = kernel_size
    rows = _block_indices(output_size[0], kh, dilation[0], padding[0], stride[0], x.device)
    cols = _block_indices(output_size[1], kw, dilation[1], padding[1], stride[1], x.device)
    bh, bw = int(rows.shape[1]), int(cols.shape[1])
    n = int(x.shape[0])
    c = int(x.shape[1]) // (kh * kw)
    if int(x.shape[-1]) != bh * bw:
        raise RuntimeError(
            f"Given output_size={output_size}, kernel_size={kernel_size}, dilation={dilation}, "
            f"padding={padding}, stride={stride}, expected input.size(-1) to be {bh * bw} "
            f"but got {int(x.shape[-1])}."
        )
    ph, pw = output_size[0] + 2 * padding[0], output_size[1] + 2 * padding[1]
    blocks = x.reshape(n, c, kh, kw, bh, bw).permute(0, 1, 2, 4, 3, 5)  # (n, c, kh, bh, kw, bw)
    flat_index = (rows.reshape(kh, bh, 1, 1) * pw + cols.reshape(1, 1, kw, bw)).reshape(-1)
    output = tp.zeros((n, c, ph * pw), dtype=x.dtype, device=x.device)
    output = output.index_add(2, flat_index, blocks.reshape(n, c, -1))
    output = output.reshape(n, c, ph, pw)
    output = output[:, :, padding[0]:padding[0] + output_size[0], padding[1]:padding[1] + output_size[1]]
    output = output.contiguous()
    return output if batched else output.squeeze(0)


def _max_unpool(self, indices, output_size, dims):
    output_shape = list(self.shape[:-dims]) + [int(s) for s in output_size]
    if any(s == 0 for s in output_shape):
        return tp.zeros(tuple(output_shape), dtype=self.dtype, device=self.device)
    leading = 1
    for size in self.shape[:-dims]:
        leading *= int(size)
    area = 1
    for size in output_size:
        area *= int(size)
    base_shape = [1] * self.dim()
    base_shape[:-dims] = [int(s) for s in self.shape[:-dims]]
    base = tp.arange(leading, device=self.device).reshape(tuple(base_shape)) * area
    flat_index = (indices + base).reshape(-1)
    output = tp.zeros(leading * area, dtype=self.dtype, device=self.device)
    output = output.index_copy(0, flat_index, self.reshape(-1))
    return output.reshape(tuple(output_shape))


@register_decomposition(ops.max_unpool2d.default)
def max_unpool2d(self, indices, output_size):
    if self.dim() not in (3, 4):
        raise RuntimeError(
            f"Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with {self.dim()} dimensions."
        )
    return _max_unpool(self, indices, list(output_size), 2)


@register_decomposition(ops.max_unpool3d.default)
def max_unpool3d(self, indices, output_size, stride, padding):
    if self.dim() not in (4, 5):
        raise RuntimeError(
            f"Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with {self.dim()} dimensions."
        )
    return _max_unpool(self, indices, list(output_size), 3)


# ---------------------------------------------------------------------------
# Unchecked indexing
# ---------------------------------------------------------------------------


@register_decomposition(ops._unsafe_index.Tensor)
def _unsafe_index(self, indices):
    return ops.index.Tensor(self, list(indices))


@register_decomposition(ops._unsafe_index_put.default)
def _unsafe_index_put(self, indices, values, accumulate=False):
    return ops.index_put.default(self, list(indices), values, accumulate)


def _check_masked_index_args(mask, indices):
    for index in indices:
        if index is not None and index.dtype not in (tp.int64, tp.int32):
            raise RuntimeError("tensors used as indices must be long or int tensors")
    if mask.dtype != tp.bool:
        raise RuntimeError("tensors used as masks must be bool tensors")


@register_decomposition(ops._unsafe_masked_index.default)
def _unsafe_masked_index(self, mask, indices, fill):
    _check_masked_index_args(mask, indices)
    indices = list(indices)
    if self.numel() == 0:
        shape = tuple(ops.index.Tensor(tp.empty(tuple(self.shape), dtype=self.dtype), indices).shape)
        return tp.full(shape, fill, dtype=self.dtype, device=self.device)
    clamped = [
        None if index is None else tp.clamp(index, 0, int(self.shape[i]) - 1)
        for i, index in enumerate(indices)
    ]
    return ops.index.Tensor(self, clamped).masked_fill(~mask, fill)


@register_decomposition(ops._unsafe_masked_index_put_accumulate.default)
def _unsafe_masked_index_put_accumulate(self, mask, indices, values):
    _check_masked_index_args(mask, indices)
    if self.numel() == 0:
        return self.clone()
    clamped = [
        None if index is None else tp.clamp(index, -int(self.shape[i]), int(self.shape[i]) - 1)
        for i, index in enumerate(indices)
    ]
    return ops._unsafe_index_put.default(self, clamped, values.masked_fill(~mask, 0), True)


# ---------------------------------------------------------------------------
# Margin losses
# ---------------------------------------------------------------------------


@register_decomposition(ops.multi_margin_loss.default)
def multi_margin_loss(self, target, p=1, margin=1, weight=None, reduction=1):
    input = tp.atleast_2d(self)
    target = tp.atleast_1d(target)
    nframe, dim = int(input.shape[0]), int(input.shape[1])
    if p not in (1, 2):
        raise RuntimeError("only p == 1 and p == 2 supported")
    if input.dim() != 2 or dim == 0:
        raise RuntimeError(
            f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {tuple(input.shape)}"
        )
    if target.dim() != 1 or target.numel() != nframe:
        raise RuntimeError(f"inconsistent target size, expected {nframe} but got {tuple(target.shape)}")
    if weight is not None:
        weight = tp.atleast_1d(weight)
        if weight.dim() != 1 or weight.numel() != dim:
            raise RuntimeError(f"inconsistent weight size, expected {dim} but got {tuple(weight.shape)}")
    column = target.unsqueeze(1)
    picked = tp.gather(input, 1, column)
    z = tp.clamp(margin - picked + input, min=0)
    z = z if p == 1 else z * z
    if weight is not None:
        z = z * tp.index_select(weight, 0, target).unsqueeze(1)
    classes = tp.arange(dim, device=input.device)
    z = tp.where(classes != column, z, tp.zeros((), dtype=z.dtype, device=z.device))
    if reduction == 1:
        return z.mean()
    if reduction == 2:
        return z.sum() / dim
    return z.mean(dim=1)


@register_decomposition(ops.multilabel_margin_loss_forward.default)
def multilabel_margin_loss_forward(self, target, reduction):
    input_shape, target_shape = tuple(self.shape), tuple(target.shape)
    input = tp.atleast_2d(self)
    target = tp.atleast_2d(target)
    dim = int(input.shape[1])
    if len(input_shape) > 2 or dim == 0:
        raise RuntimeError(
            f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {input_shape}"
        )
    if len(target_shape) > 2 or target_shape != input_shape:
        raise RuntimeError(f"inconsistent target size: {target_shape} for input of size: {input_shape}")
    # Labels end at the first -1.
    positions = tp.arange(dim, device=target.device)
    end = tp.amin(tp.where(target == -1, positions, dim), dim=[-1], keepdim=True)
    valid = positions < end
    picked = tp.gather(input, -1, tp.where(valid, target, 0))
    labels = tp.where(valid, target, -1)
    is_target = (positions == labels.unsqueeze(-1)).any(dim=1)
    z = tp.clamp(1.0 - picked.t().unsqueeze(-1) + input, min=0) / dim
    zero = tp.zeros((), dtype=z.dtype, device=z.device)
    z = tp.where(is_target, zero, z)
    z = tp.where(valid.t().unsqueeze(-1), z, zero)
    if reduction == 1:
        z = z.sum(dim=(0, -1)).mean()
    elif reduction == 2:
        z = z.sum()
    else:
        z = z.sum(dim=(0, -1))
    return z, is_target.to(self.dtype).reshape(target_shape)


# ---------------------------------------------------------------------------
# Sampling grids, random draws, ranges, norms
# ---------------------------------------------------------------------------


def _linspace_from_neg_one(steps, align_corners, dtype, device):
    if steps <= 1:
        return tp.zeros((), dtype=dtype, device=device)
    bound = 1.0 if align_corners else (steps - 1) / steps
    return tp.linspace(-bound, bound, steps, dtype=dtype, device=device)


@register_decomposition(ops.affine_grid_generator.default)
def affine_grid_generator(theta, size, align_corners):
    size = [int(s) for s in size]
    if len(size) not in (4, 5):
        raise RuntimeError("affine_grid_generator needs 4d (spatial) or 5d (volumetric) inputs.")
    dtype = _computation_dtype(theta.dtype)
    theta_c = theta.to(dtype)
    spatial = size[2:]
    rank = len(spatial)
    # Homogeneous base grid: coordinates ordered (x, y[, z], 1).
    components = []
    for axis, steps in enumerate(reversed(spatial)):
        shape = [1] * rank + [1]
        shape[rank - 1 - axis] = steps
        components.append(_linspace_from_neg_one(steps, align_corners, dtype, theta.device).reshape(tuple(shape)))
    components.append(tp.ones(tuple([1] * (rank + 1)), dtype=dtype, device=theta.device))
    full = tuple(spatial) + (1,)
    base = tp.cat([c.expand(full) for c in components], dim=-1)
    grid = (base.reshape(-1, rank + 1, 1) * theta_c.transpose(-2, -1).unsqueeze(1)).sum(dim=-2)
    return grid.reshape(size[0], *spatial, rank).to(theta.dtype)


@register_decomposition(ops.rrelu_with_noise_backward.default)
def rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training, self_is_result):
    if training:
        return grad_output * noise
    return ops.leaky_relu_backward.default(grad_output, self, (lower + upper) / 2, self_is_result)


@register_decomposition(ops.bernoulli.default)
def bernoulli(self, *, generator=None):
    draws = tp.rand(tuple(self.shape), dtype=tp.float32, device=self.device, generator=generator)
    return (draws < self).to(self.dtype)


@register_decomposition(ops.arange.start)
def arange_start(start, end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return ops.arange.start_step(start, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)


@register_decomposition(ops.arange.end)
def arange_end(end, *, dtype=None, device=None, requires_grad=False):
    dtype = None if dtype is None or dtype == tp.DType.undefined else dtype
    return ops.arange.start_step(0, end, 1, dtype=dtype, device=device)


@register_decomposition(ops.linalg_vector_norm.default)
def linalg_vector_norm(self, ord=2, dim=None, keepdim=False, *, dtype=None):
    if not (self.is_floating_point() or self.is_complex()):
        raise RuntimeError(
            f"linalg.vector_norm: Expected a floating point or complex tensor as input. Got {self.dtype}"
        )
    ord = float(ord)
    if dim is not None and not isinstance(dim, (list, tuple)):
        dim = [dim]
    if dim is not None and len(dim) == 0:
        dim = None
    if dim is not None:
        dim = [d % self.dim() if self.dim() else 0 for d in dim]
    if ord < 0 or math.isinf(ord):
        reduced = range(self.dim()) if dim is None else dim
        if self.numel() == 0 and (dim is None or any(int(self.shape[d]) == 0 for d in reduced)):
            raise RuntimeError(
                f"linalg.vector_norm cannot compute the {ord} norm on an empty tensor "
                "because the operation does not have an identity"
            )
    result_dtype = dtype
    if result_dtype is None:
        result_dtype = self.abs().dtype if self.is_complex() else self.dtype
    computation_dtype = _computation_dtype(result_dtype)
    dims = list(range(self.dim())) if dim is None else list(dim)
    if ord == 0.0:
        return (self != 0).to(result_dtype).sum(dim=dims, keepdim=keepdim) if dims else (self != 0).to(result_dtype)
    if math.isinf(ord):
        magnitude = self.abs()
        if not dims:
            return magnitude.to(result_dtype)
        reduced = tp.amax(magnitude, dim=dims, keepdim=keepdim) if ord > 0 else tp.amin(magnitude, dim=dims, keepdim=keepdim)
        return reduced.to(result_dtype)
    x = self.to(computation_dtype) if not self.is_complex() else self
    x = x.abs()
    if not dims:
        return x.to(result_dtype).contiguous()
    return tp.pow(tp.pow(x, ord).sum(dim=dims, keepdim=keepdim), 1.0 / ord).to(result_dtype)
