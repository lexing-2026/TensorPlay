"""Decompositions of operator overloads onto other operators.

Each function computes its operator from simpler operators with the same
numerical contract (dtype promotion, broadcasting, special values).  The
in-place overload of an operator reuses the functional decomposition and
writes the result into ``self``; ``out=`` overloads are served by the
registry, which writes the result into the destination.
"""

from __future__ import annotations

import math
import operator
import sys
from collections.abc import Callable
from functools import partial, reduce
from typing import Any

import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE
from tensorplay.primitives.common import (
    canonicalize_dim,
    canonicalize_dims,
    infer_size,
    is_contiguous_or_false,
    make_contiguous_strides_for,
)

from . import register_decomposition

ops = getattr(tp.ops, NATIVE_NAMESPACE)
prims = tp.ops.prims


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


# ---------------------------------------------------------------------------
# matrix products
# ---------------------------------------------------------------------------


def should_fold(tensor1, tensor2, is_out: bool) -> bool:
    """Whether a product of a matrix by a stack of them is one product.

    A product of an ``n``-by-``m`` by an ``m``-by-``p`` is a product; a product of
    a stack of them by a matrix is a stack of products, and can be computed as
    one product by folding the stack into the first matrix's rows.  Whether that
    is worth doing is the question here, because folding copies when the stack is
    not one run of memory, and a copy is more than the saving.

    Folding is refused when it would read a stride the eager path would not have
    read, and accepted when it would: a matrix whose rows are not contiguous is
    the case where folding would gather, and gathering to save a loop is a loss.
    A stack that needs its gradient is folded regardless, because the gradient of
    a fold is a different shape and a caller asking for one is asking for the
    fold to have been possible.
    """

    from tensorplay.graph.experimental.symbolic_shapes import guard_or_false

    # The one with more dimensions decides, since folding is only ever from a
    # stack to something without a stack.
    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)

    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    if t2.requires_grad and not is_out:
        return True
    if tensor1.ndim == 2:
        return False
    from tensorplay.functional import sym_numel

    if guard_or_false(sym_numel(t1) == 0):
        return True

    t1_shape = t1.shape
    t1_stride = t1.stride()

    # Contiguous apart from any axis of extent one, which addresses as many
    # elements as it skips and so makes no difference to whether the rows follow
    # one another.
    expected_stride = [1]
    for size in reversed(t1_shape[1:]):
        expected_stride.append(size * expected_stride[-1])
    return all(
        guard_or_false(size == 1) or guard_or_false(left == right)
        for left, right, size in zip(
            t1_stride, list(reversed(expected_stride)), t1_shape
        )
    )


@register_decomposition([ops.matmul.default, ops.matmul.out])
def matmul(tensor1, tensor2, *, is_out=False):
    """A product of whatever shapes were given, as the products it is made of.

    One operation covers a vector by a vector, a matrix by a vector, a stack of
    matrices by a vector, and a stack by a stack, and which of those it is can
    only be told from the shapes.  So it is taken apart here into the products
    that do have one meaning each, and everything downstream of here sees one of
    those rather than a case.
    """

    from tensorplay.graph.experimental.symbolic_shapes import guard_or_true

    dim_tensor1 = tensor1.dim()
    dim_tensor2 = tensor2.dim()
    if dim_tensor1 == 0 or dim_tensor2 == 0:
        raise AssertionError(
            f"matmul does not support 0-dimensional tensors, got dims: "
            f"{dim_tensor1} and {dim_tensor2}"
        )
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return tp.dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return tp.mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return tp.squeeze(tp.mm(tp.unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        return tp.mm(tensor1, tensor2)
    elif should_fold(tensor1, tensor2, is_out):
        # dim_tensor1 >= 3 and the other has at most two dimensions, and the
        # strides allow it: the stack of matrices becomes one matrix by having
        # its stack axis folded into its rows, so the product is one product.
        transpose = dim_tensor2 > dim_tensor1
        t1 = tensor2.mT if transpose else tensor1
        t2 = tensor2 if not transpose else (tensor1.t() if dim_tensor1 == 2 else tensor1)

        sizes_1 = t1.shape
        output_shape = list(sizes_1[:-1])
        folded_dim1 = reduce(operator.mul, output_shape)

        t2_is_matrix = t2.dim() == 2
        if t2_is_matrix:
            output_shape.append(t2.shape[1])

        # A reshape rather than a view because the last extent may be zero, and
        # a view cannot say which extent a -1 stands for when one of them is.
        t1_folded = t1.reshape(folded_dim1, sizes_1[-1])
        if t2_is_matrix:
            output = tp.ops.tp._unsafe_view(t1_folded.mm(t2), output_shape)
            return output.mT.contiguous() if transpose else output
        return tp.ops.tp._unsafe_view(t1_folded.mv(t2), output_shape)

    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # A stack by a stack.  The two stacks need not agree: one may have a
        # single entry where the other has many, which is a broadcast rather than
        # a product of equals.
        n = tensor1.size(-2) if dim_tensor1 > 1 else 1
        m1 = tensor1.size(-1)
        batch_tensor1 = tensor1.shape[:-2]
        m2 = tensor2.size(-2) if dim_tensor2 > 1 else tensor2.size(-1)
        p = tensor2.size(-1) if dim_tensor2 > 1 else 1

        batch_tensor2: list = []
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2.size(i))

        # The same folding as above, decided by the stack axes disagreeing: a
        # stack of one against a stack of many can drop the stack of one, which
        # is only a rewrite of the shape rather than a gather.
        if (
            dim_tensor1 == 3
            and dim_tensor2 == 3
            and guard_or_true(batch_tensor1[0] != batch_tensor2[0])
        ):
            if guard_or_false(batch_tensor1[0] == 1) and tensor1.requires_grad:
                return matmul(tensor1.squeeze(0), tensor2)
            if guard_or_false(batch_tensor2[0] == 1) and tensor2.requires_grad:
                return matmul(tensor1, tensor2.squeeze(0))

        expand_batch_portion = list(
            tp.broadcast_shapes(tuple(batch_tensor1), tuple(batch_tensor2))
        )
        tensor1_expand_size = expand_batch_portion + [n, m1]
        expand_batch_product = reduce(operator.mul, expand_batch_portion)

        tensor1_expanded = tensor1.broadcast_to(tensor1_expand_size) \
            if expand_batch_product > 1 else tensor1
        tensor2_expanded = tensor2.broadcast_to(expand_batch_portion + [m2, p]) \
            if expand_batch_product > 1 else tensor2

        # The folded pair is a stack of matrices, so the product of it is a
        # product of a stack -- a different operation from the one being
        # decomposed.  Reaching for this operation here would ask the
        # question that is being answered, and the answer would be asked
        # again.
        return tensor1_expanded.reshape(expand_batch_product, n, m1).bmm(
            tensor2_expanded.reshape(expand_batch_product, m2, p)
        ).reshape(expand_batch_portion + [n, p])

    raise RuntimeError(
        f"matmul: unable to compute the product of {dim_tensor1}-dimensional "
        f"and {dim_tensor2}-dimensional values"
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


def _tensor_split_sections(self, sections, dim):
    if self.dim() == 0:
        raise RuntimeError(
            "tensor_split expected at least a 1-dimensional tensor, but got a tensor with 0 dims"
        )
    if sections <= 0:
        raise RuntimeError(f"number of sections must be larger than 0, got {sections}")
    dim = dim % self.dim()
    length = int(self.shape[dim])
    base, extras = divmod(length, sections)
    sizes = [base + (1 if index < extras else 0) for index in range(sections)]
    return _split_with_sizes(self, sizes, dim)


def _tensor_split_boundaries(self, boundaries, dim):
    if self.dim() == 0:
        raise RuntimeError(
            "tensor_split expected at least a 1-dimensional tensor, but got a tensor with 0 dims"
        )
    dim = dim % self.dim()
    length = int(self.shape[dim])
    pieces = []
    start = 0
    for boundary in [*boundaries, length]:
        end = int(boundary)
        if end < 0:
            end += length
        end = min(max(end, 0), length)
        start = min(max(start, 0), length)
        pieces.append(tp.narrow(self, dim, start, max(end - start, 0)))
        start = end
    return pieces


@register_decomposition(ops.tensor_split.sections)
def tensor_split_sections(self, sections, dim=0):
    return _tensor_split_sections(self, sections, dim)


@register_decomposition(ops.tensor_split.indices)
def tensor_split_indices(self, indices, dim=0):
    if isinstance(indices, tp.Tensor):
        indices = indices.tolist()
    return _tensor_split_boundaries(self, list(indices), dim)


@register_decomposition(ops.tensor_split.tensor_indices_or_sections)
def tensor_split_tensor(self, tensor_indices_or_sections, dim=0):
    if tensor_indices_or_sections.device.type != "cpu":
        raise RuntimeError(
            "tensor_split expected tensor_indices_or_sections to be on cpu, but it's on "
            f"{tensor_indices_or_sections.device.type}"
        )
    if tensor_indices_or_sections.dtype != tp.int64:
        raise RuntimeError(
            "tensor_split expected tensor_indices_or_sections to have dtype of long, but got "
            f"{tensor_indices_or_sections.dtype}"
        )
    if tensor_indices_or_sections.dim() > 1:
        raise RuntimeError(
            "tensor_split expected tensor_indices_or_sections to be a zero-dimensional or "
            f"one-dimensional tensor, but got a tensor with {tensor_indices_or_sections.dim()} dims"
        )
    boundaries = tensor_indices_or_sections.tolist()
    if tensor_indices_or_sections.dim() == 0:
        boundaries = [boundaries]
    return _tensor_split_boundaries(self, boundaries, dim)


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
    # Both operands are given the output's extents rather than left to be
    # matched up while the expression is written: a reader of a value is handed
    # one index for the whole output, so an operand of fewer extents would have
    # to be widened before it is read at all.
    row = tp.arange(rows, device=self.device).unsqueeze(-1).expand(rows, cols)
    col = tp.arange(cols, device=self.device).expand(rows, cols)
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
# Reductions
# ---------------------------------------------------------------------------


def _reduction_dims(a, dim):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return list(range(a.dim()))
    if isinstance(dim, int):
        return [canonicalize_dim(a.dim(), dim)]
    return canonicalize_dims(a.dim(), list(dim))


def _kept_shape(a, dims):
    return [int(a.shape[i]) if i not in dims else 1 for i in range(a.dim())]


def _int_like_dtype(a):
    # Bool and integer sums and products accumulate in int64.
    if a.dtype == tp.bool or (not a.dtype.is_floating_point and not a.dtype.is_complex):
        return tp.int64
    return a.dtype


@register_decomposition([ops.sum.default, ops.sum.dim_IntList, ops.sum.IntList_out])
def sum_reduce(a, dim=None, keepdim=False, *, dtype=None):
    dims = _reduction_dims(a, dim)
    result = prims.sum(a.to(dtype if dtype is not None else _int_like_dtype(a)), dims)
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    return result


@register_decomposition([ops.prod.default, ops.prod.dim_int, ops.prod.dim_IntList,
                         ops.prod.int_out])
def prod_reduce(a, dim=None, keepdim=False, *, dtype=None):
    dims = _reduction_dims(a, dim)
    result = prims.prod(a.to(dtype if dtype is not None else _int_like_dtype(a)), dims)
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    return result


@register_decomposition([ops.mean.default, ops.mean.dim, ops.mean.out, ops.mean.dtype_out])
def mean_reduce(a, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = a.dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "mean(): could not infer output dtype. Input dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    dims = _reduction_dims(a, dim)
    total = prims.sum(a.to(_computation_dtype(a.dtype)), dims)
    nelem = 1
    for d in dims:
        nelem *= int(a.shape[d])
    result = total / nelem
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    return result.to(dtype)


def _var_reduce_common(a, dim, correction, keepdim):
    result_dtype = a.abs().dtype if a.is_complex() else a.dtype
    result = _var(a.to(_computation_dtype(a.dtype)), tuple(_reduction_dims(a, dim)),
                  correction, keepdim)
    return result.to(result_dtype)


@register_decomposition(ops.var.default)
def var_reduce(a, correction=1):
    return _var_reduce_common(a, None, correction, False)


@register_decomposition(ops.var.dim)
def var_reduce_dim(a, dim, correction=1, keepdim=False):
    return _var_reduce_common(a, dim, correction, keepdim)


@register_decomposition([ops.var.correction, ops.var.out])
def var_reduce_correction(a, dim=None, *, correction=None, keepdim=False):
    return _var_reduce_common(a, dim, _correction(correction), keepdim)


def _var_mean_reduce_common(a, dim, correction, keepdim):
    return _var_reduce_common(a, dim, correction, keepdim), mean_reduce(a, dim, keepdim)


@register_decomposition(ops.var_mean.default)
def var_mean_reduce(a, dim=[], unbiased=True, keepdim=False):
    return _var_mean_reduce_common(a, dim, _correction(None, unbiased), keepdim)


@register_decomposition(ops.var_mean.dim)
def var_mean_reduce_dim(a, dim, unbiased=True, keepdim=False):
    return _var_mean_reduce_common(a, dim, _correction(None, unbiased), keepdim)


@register_decomposition(ops.var_mean.correction)
def var_mean_reduce_correction(a, dim=None, *, correction=None, keepdim=False):
    return _var_mean_reduce_common(a, dim, _correction(correction), keepdim)


@register_decomposition([ops.amax.default, ops.amax.out])
def amax_reduce(a, dim=[], keepdim=False):
    dims = _reduction_dims(a, dim)
    result = prims.amax(a, dims)
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    return result


@register_decomposition([ops.amin.default, ops.amin.out])
def amin_reduce(a, dim=[], keepdim=False):
    dims = _reduction_dims(a, dim)
    result = prims.amin(a, dims)
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    return result


@register_decomposition([ops.any.default, ops.any.dim, ops.any.dims, ops.any.out,
                         ops.any.dims_out, ops.any.all_out])
def any_reduce(a, dim=None, keepdim=False):
    dims = _reduction_dims(a, dim)
    result = ops.ne.Scalar(prims.sum(a.to(tp.int64), dims), 0)
    if keepdim:
        result = ops.reshape.default(result, _kept_shape(a, dims))
    # A uint8 answer preserves the legacy mask spelling.
    return result.to(tp.uint8) if a.dtype == tp.uint8 else result


def _cumsumprod_common(reduce_fn, init, a, dim, dtype):
    ndim = a.dim()
    dim = canonicalize_dim(ndim, dim)
    if ndim == 0:
        return a if dtype is None else a.to(dtype)
    # A row is included in the answer at position j when its index i <= j:
    # the triangular mask turns the running scan into one masked reduction.
    a_usq = a.unsqueeze(dim + 1)
    rg = tp.arange(int(a_usq.shape[dim]), device=a.device)
    mask = rg.unsqueeze(1) <= rg
    for _ in range(ndim - dim - 1):
        mask = mask.unsqueeze(-1)
    masked = ops.where.ScalarOther(mask, a_usq, init)
    return reduce_fn(masked, [dim], False, dtype=dtype)


@register_decomposition(ops.cumsum.default)
def cumsum_reduce(a, dim=0, dtype=None):
    return _cumsumprod_common(sum_reduce, 0, a, dim, dtype)


@register_decomposition(ops.cumprod.default)
def cumprod_reduce(a, dim, dtype=None):
    return _cumsumprod_common(prod_reduce, 1, a, dim, dtype)


_QUANTILE_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def _quantile_impl(self, q, dim, keepdim, interpolation, ignore_nan):
    name = "nanquantile()" if ignore_nan else "quantile()"
    if self.numel() == 0:
        raise RuntimeError(f"{name} input tensor must be non-empty")
    if not isinstance(q, tp.Tensor):
        q = tp.full((), q, dtype=self.dtype, device=self.device)
    if q.dim() > 1:
        raise RuntimeError(f"{name} q must be a scalar or 1D tensor")
    if self.dtype not in (tp.float32, tp.float64):
        raise RuntimeError(f"{name} input tensor must be either float or double dtype")
    if q.dtype != self.dtype:
        raise RuntimeError(f"{name} q tensor must be same dtype as the input tensor")
    if q.device != self.device:
        raise RuntimeError(f"{name} q tensor must be on the same device as the input tensor")
    if interpolation not in _QUANTILE_INTERPOLATIONS:
        raise RuntimeError(
            f"{name} interpolation must be one of linear, lower, higher, midpoint or "
            f"nearest, but got {interpolation}"
        )

    scalar_q = q.dim() == 0
    q_extent = 1 if scalar_q else int(q.numel())

    if dim is None:
        wrapped_dim = None
    elif self.dim() == 0:
        raise IndexError("Dimension specified as 0 but tensor has no dimensions")
    else:
        wrapped_dim = dim % self.dim()

    out_shape = []
    if dim is not None:
        out_shape = list(self.shape)
        if keepdim:
            out_shape[wrapped_dim] = 1
        else:
            del out_shape[wrapped_dim]
    elif keepdim:
        out_shape = [1] * self.dim()
    if not scalar_q:
        out_shape.insert(0, q_extent)

    if dim is None:
        reduced = self.reshape(-1)
    elif wrapped_dim == self.dim() - 1:
        reduced = self
    else:
        reduced = self.unsqueeze(-1).transpose(wrapped_dim, -1)

    # The reduction runs over the last axis of the view; every remaining axis
    # is a row the quantiles are read from, and the reduction extent joins
    # them as one more trailing axis.
    view_shape = list(out_shape)
    if scalar_q:
        view_shape.insert(0, 1)
    in_shape = list(view_shape[1:])
    in_shape.append(int(reduced.shape[-1]))
    reduced = reduced.reshape(in_shape)

    last_index = int(reduced.shape[-1]) - 1
    if ignore_nan:
        # Rows carry their rank by their non-NaN count, so a row that is all
        # NaN ranks at zero and reads a NaN back; a partly NaN row skips the
        # NaN entries in both count and order.
        non_nan = tp.logical_not(tp.isnan(reduced)).sum(-1, keepdim=True)
        ranks = q.to(tp.float64) * (non_nan - 1).to(tp.float64)
        ranks = tp.masked_fill(ranks, ranks < 0, 0)
    else:
        # A NaN anywhere in a row pins every rank of that row to the last
        # position, so the quantile read there is NaN.
        nan_anywhere = tp.isnan(reduced).any(-1, keepdim=True)
        q_ranks = q.to(tp.float64) * float(last_index)
        rank_shape = tuple(nan_anywhere.shape[:-1]) + (q_extent,)
        ranks = tp.masked_fill(q_ranks.expand(rank_shape), nan_anywhere, float(last_index))

    if interpolation == "lower":
        ranks = ranks.floor()
    elif interpolation == "higher":
        ranks = ranks.ceil()
    elif interpolation == "nearest":
        ranks = ranks.round()

    ranks_below = ranks.to(tp.int64)
    interpolate = interpolation in ("linear", "midpoint")
    if interpolate:
        if interpolation == "midpoint":
            weights = tp.full_like(ranks_below, 0.5, dtype=self.dtype)
        else:
            weights = (ranks - ranks_below).to(self.dtype)
        ranks_above = ranks.ceil().to(tp.int64)

    ordered = tp.sort(reduced).values
    values = ordered.gather(-1, ranks_below)
    if interpolate:
        values = tp.lerp(values, ordered.gather(-1, ranks_above), weights)

    if scalar_q:
        values = values.squeeze(-1)
    else:
        values = values.unsqueeze(0).transpose(0, -1).squeeze(-1)
    return values


@register_decomposition(ops.quantile.default)
def quantile(self, q, dim=None, keepdim=False, *, interpolation="linear"):
    return _quantile_impl(self, q, dim, keepdim, interpolation, False)


@register_decomposition(ops.nanquantile.default)
def nanquantile(self, q, dim=None, keepdim=False, *, interpolation="linear"):
    return _quantile_impl(self, q, dim, keepdim, interpolation, True)


@register_decomposition(ops.instance_norm.default)
def instance_norm(input, weight=None, bias=None, running_mean=None, running_var=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5):
    if not use_input_stats and (running_mean is None or running_var is None):
        raise RuntimeError(
            "Expected running_mean and running_var to be defined when use_input_stats is false"
        )
    if input.dim() < 3:
        raise RuntimeError(
            f"instance_norm: input must have at least 3 dimensions, but got {input.dim()} dims"
        )

    batch = int(input.shape[0])
    channels = int(input.shape[1])
    # Every sample's every channel is one normalization group of its own, so
    # the samples are folded into the channel axis and each channel's
    # per-sample statistic stands alone: the merged call is one batch norm
    # whose channels are the (sample, channel) pairs.
    merged = [1, batch * channels] + list(input.shape[2:])

    def repeated(stats):
        if stats is None:
            return None
        return stats.repeat([batch] + [1] * (stats.dim() - 1))

    # Mixed running-stat dtypes follow the weight's dtype when one is given.
    if weight is not None:
        if running_mean is not None and running_mean.dtype != weight.dtype:
            running_mean = running_mean.to(weight.dtype)
        if running_var is not None and running_var.dtype != weight.dtype:
            running_var = running_var.to(weight.dtype)

    input_reshaped = input.contiguous().view(merged)
    running_mean_ = repeated(running_mean)
    running_var_ = repeated(running_var)
    out = ops.batch_norm.default(
        input_reshaped, repeated(weight), repeated(bias),
        running_mean_, running_var_,
        use_input_stats, momentum, eps,
    )
    if use_input_stats:
        if running_mean is not None:
            running_mean.copy_(running_mean_.view(batch, channels).mean(0))
        if running_var is not None:
            running_var.copy_(running_var_.view(batch, channels).mean(0))
    return out.view(list(input.shape))


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


def _sdpa_gqa_key_value(query, key, value, enable_gqa):
    """Key and value heads repeated so each query head has one of its own.

    With grouped-query attention several query heads read the same key and
    value head, so the key and value are given as many heads as the query has
    before the scores are taken.
    """
    if not enable_gqa or key.dim() < 3 or query.dim() < 3:
        return key, value
    hq, hk = query.shape[-3], key.shape[-3]
    if hq == hk:
        return key, value
    if hq % hk != 0:
        raise ValueError(
            "sdpa: enable_gqa requires the query head count to be divisible "
            f"by the key/value head count, got {hq} and {hk}"
        )
    repeats = [1] * key.dim()
    repeats[-3] = hq // hk
    return key.repeat(repeats), value.repeat(repeats)


def _sdpa_math_attention(query, key, value, attn_mask, dropout_p, is_causal,
                         scale, enable_gqa):
    """Attention written out of the operations it is made of.

    The scale folds into the query before the scores are taken rather than
    dividing them after, which is what a caller asking for a scale means; the
    mask is what is added to the scores, so a boolean mask selects and a
    floating one offsets; and the causal mask is the upper triangle of the
    score matrix set to nothing.
    """
    key, value = _sdpa_gqa_key_value(query, key, value, enable_gqa)
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    scores = tp.matmul(query * scale, key.transpose(-2, -1))
    if is_causal:
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        keep = tp.ones([t_q, t_k], dtype=tp.bool, device=scores.device).tril(
            diagonal=t_k - t_q
        )
        scores = tp.where(keep, scores, -math.inf)
    if attn_mask is not None:
        if attn_mask.dtype == tp.bool:
            scores = tp.where(attn_mask, scores, -math.inf)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)
    probs = _safe_softmax(scores, -1)
    if dropout_p:
        # Dropping is written as the operation that keeps what it kept, which
        # already rescales what it kept so that what survives averages to what
        # was there.
        probs = ops.native_dropout.default(probs, dropout_p, True)[0]
    return tp.matmul(probs, value)


def _can_use_cpu_flash(query, key, value, attn_mask, dropout_p, is_causal,
                       enable_gqa):
    """Whether the fused cpu flash kernel covers the call.

    The kernel accepts four-dimensional float inputs with matching head
    sizes, no dropout, and an optional two- or four-dimensional mask of the
    same precision as the query (or float for a float query).  Calls outside
    that envelope fall back to the math path.
    """
    if query.device.type != "cpu":
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if dropout_p != 0.0:
        return False
    if query.dtype not in (tp.float16, tp.bfloat16, tp.float32, tp.float64):
        return False
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False
    if query.size(-1) != key.size(-1) or query.size(-1) != value.size(-1):
        return False
    if enable_gqa and query.size(-3) != key.size(-3):
        return False
    if is_causal and query.size(-2) != key.size(-2):
        return False
    if attn_mask is not None:
        if attn_mask.requires_grad:
            return False
        if attn_mask.dim() not in (2, 4):
            return False
        if attn_mask.dtype != tp.float32 and attn_mask.dtype != query.dtype:
            return False
    return True


@register_decomposition(ops.scaled_dot_product_attention.default)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, *, scale=None, enable_gqa=False):
    if _can_use_cpu_flash(query, key, value, attn_mask, dropout_p, is_causal,
                          enable_gqa):
        output, _ = ops._scaled_dot_product_flash_attention_for_cpu.default(
            query, key, value, dropout_p, is_causal, attn_mask=attn_mask,
            scale=scale,
        )
        return output
    return _sdpa_math_attention(query, key, value, attn_mask, dropout_p,
                                is_causal, scale, enable_gqa)


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


def _batch_norm_stat_dims(input):
    # Channel 1 plus every spatial dim: the batch norm statistics reduce the
    # batch and the spatial extent, keeping one value per channel.
    return [0] + list(range(2, input.dim()))


def _batch_norm_stat_shape(input, dims):
    return [int(s) for i, s in enumerate(input.shape) if i not in dims]


def _batch_norm_helper(input, weight, bias, running_mean, running_var,
                       training, momentum, eps, functional):
    dims = _batch_norm_stat_dims(input)
    compute = _computation_dtype(input.dtype)
    new_running_mean = running_mean
    new_running_var = running_var
    if training:
        input_acc = input.to(dtype=compute)
        biased_var, mean = ops.var_mean.default(
            input_acc, dims, unbiased=False, keepdim=True
        )
        rstd = tp.rsqrt(biased_var + eps)
        output = (input_acc - mean) * rstd
        save_mean = ops.reshape.default(mean, _batch_norm_stat_shape(input, dims))
        save_rstd = ops.reshape.default(rstd, _batch_norm_stat_shape(input, dims))
        if running_mean is not None:
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
            if not functional:
                running_mean.copy_(new_running_mean)
        if running_var is not None:
            n = input.numel() / int(input.shape[1])
            # The running spread keeps the unbiased estimate, recovered from
            # the biased one by the factor n/(n-1).
            unbiased_var = ops.reshape.default(
                biased_var, _batch_norm_stat_shape(input, dims)
            ) * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
            if not functional:
                running_var.copy_(new_running_var)
    else:
        if running_mean is None or running_var is None:
            raise RuntimeError("running_mean and running_var must not be None in eval mode")
        running_mean = running_mean.to(dtype=compute, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=compute, copy=True)
        new_running_var = running_var
        mean = running_mean
        invstd = tp.rsqrt(running_var + eps)
        # Backends outside the CPU path reuse the running statistics as the
        # saved ones; the CPU kernel reports empty saved stats because the
        # backward pass recomputes the inverse spread from running_var.
        if input.device.type != "cpu":
            save_mean = running_mean
            save_rstd = invstd
        else:
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        mean = _to_rank(mean, input.dim() - 1)
        invstd = _to_rank(invstd, input.dim() - 1)
        output = (input - mean) * invstd

    if weight is not None:
        output = output * _to_rank(ops.reshape.default(weight, [-1]), input.dim() - 1)
    if bias is not None:
        output = output + _to_rank(ops.reshape.default(bias, [-1]), input.dim() - 1)

    if input.device.type == "cpu":
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    )


@register_decomposition(ops.native_batch_norm.default)
def native_batch_norm(input, weight, bias, running_mean, running_var,
                      training, momentum, eps):
    output, save_mean, save_rstd, _, _ = _batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(ops._native_batch_norm_legit.default)
def _native_batch_norm_legit(input, weight, bias, running_mean, running_var,
                             training, momentum, eps):
    output, save_mean, save_rstd, _, _ = _batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(ops._native_batch_norm_legit.no_stats)
def _native_batch_norm_legit_no_stats(input, weight, bias, training, momentum, eps):
    output, save_mean, save_rstd, _, _ = _batch_norm_helper(
        input, weight, bias, None, None, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(ops._native_batch_norm_legit_no_training.default)
def _native_batch_norm_legit_no_training(input, weight, bias, running_mean,
                                         running_var, momentum, eps):
    return _native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )


def _batch_norm_reserve(input):
    # Placeholder the cudnn kernels size their forward-state scratch from;
    # the composite walk needs no state, so it stays empty.
    return tp.empty(0, dtype=tp.uint8, device=input.device)


@register_decomposition(ops._batch_norm_with_update.default)
def _batch_norm_with_update(input, weight, bias, running_mean, running_var,
                            momentum, eps):
    output, save_mean, save_rstd, _, _ = _batch_norm_helper(
        input, weight, bias, running_mean, running_var, True, momentum, eps, False
    )
    return output, save_mean, save_rstd, _batch_norm_reserve(input)


@register_decomposition(ops._batch_norm_no_update.default)
def _batch_norm_no_update(input, weight, bias, running_mean, running_var,
                          momentum, eps):
    output, save_mean, save_rstd, _, _ = _batch_norm_helper(
        input, weight, bias, running_mean, running_var, False, momentum, eps, False
    )
    return output, save_mean, save_rstd, _batch_norm_reserve(input)


@register_decomposition(ops.native_layer_norm.default)
def native_layer_norm(input, normalized_shape, weight, bias, eps):
    normalized_shape = tuple(int(s) for s in normalized_shape)
    normalized_ndim = len(normalized_shape)
    if normalized_ndim < 1:
        raise RuntimeError(
            "Expected normalized_shape to be at least 1-dimensional, i.e., "
            f"containing at least one element, but got normalized_shape = {list(normalized_shape)}"
        )
    if weight is not None and tuple(weight.shape) != normalized_shape:
        raise RuntimeError(
            "Expected weight to be of same shape as normalized_shape, but got "
            f"weight of shape {tuple(weight.shape)} and normalized_shape = {list(normalized_shape)}"
        )
    if bias is not None and tuple(bias.shape) != normalized_shape:
        raise RuntimeError(
            "Expected bias to be of same shape as normalized_shape, but got "
            f"bias of shape {tuple(bias.shape)} and normalized_shape = {list(normalized_shape)}"
        )
    if input.dim() < normalized_ndim or tuple(input.shape[input.dim() - normalized_ndim:]) != normalized_shape:
        raise RuntimeError(
            f"Given normalized_shape={list(normalized_shape)}, expected input with shape "
            f"{list(normalized_shape)}, but got input of size {tuple(input.shape)}"
        )
    if input.is_complex():
        raise RuntimeError("native_layer_norm does not support complex inputs")

    input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    axis = input.dim() - normalized_ndim
    dims = list(range(axis, input.dim()))
    compute = _computation_dtype(input.dtype)
    a_acc = input.to(dtype=compute)
    biased_var, mean = ops.var_mean.default(a_acc, dims, unbiased=False, keepdim=True)
    rstd = tp.rsqrt(biased_var + eps)
    out = (a_acc - mean) * rstd

    if weight is None and bias is not None:
        out = out + bias
    elif weight is not None and bias is None:
        out = out * weight
    elif weight is not None and bias is not None:
        out = out * weight + bias

    out = out.to(dtype=input.dtype)
    # The row statistics keep the computation precision the walk ran in; only
    # the normalized output drops back to the stored precision.
    return out, ops.reshape.default(mean, [-1]), ops.reshape.default(rstd, [-1])


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


@register_decomposition(ops.native_group_norm.default)
def native_group_norm(input, weight, bias, N, C, HxW, group, eps):
    """Each group of channels normalized by its own mean and spread.

    The channels of a group and the spatial positions inside them are the axes
    the statistics are taken over, so the group is made an axis of its own and
    the two that follow it are reduced together.  The spread is the biased one
    (dividing by the count rather than by one less than it) because the count is
    the whole group, not a sample of it, and the inverse spread is taken after
    the epsilon has been added so a constant group stays finite.

    The two statistics come back with the reduced axes squeezed out, so each is
    one value per group and broadcasts against the input again.
    """
    cpg = C // group
    # [N, group, channels-per-group, spatial] so the group's own channels and
    # the positions inside them are the last two axes.  The input is read in
    # its stored precision and widened inside the walk that reduces it, so no
    # float copy of the whole tensor is materialised for the statistics.
    grouped = input.reshape(N, group, cpg, HxW)
    var, mean = tp.var_mean(grouped, dim=[2, 3], unbiased=False, keepdim=True)
    rstd = (var + eps).rsqrt()

    out = (grouped - mean) * rstd
    out = out.reshape(input.shape)
    # one coefficient per channel: the channel axis is kept, every other axis
    # is a singleton, so each channel is scaled on its own.
    per_channel = [1, C] + [1] * (input.dim() - 2)
    if weight is not None:
        out = out * weight.reshape(per_channel)
    if bias is not None:
        out = out + bias.reshape(per_channel)
    return (
        _cast(out, input.dtype),
        mean.squeeze((2, 3)),
        rstd.squeeze((2, 3)),
    )


@register_decomposition(ops.native_group_norm_backward.default)
def native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask):
    cpg = C // group
    if C != cpg * group:
        raise RuntimeError(
            f"Expect number of channels {C} to be evenly-divisible by number of groups {group}"
        )
    ds = (grad_out * input).reshape(N, C, HxW).sum(dim=[2])
    db = grad_out.reshape(N, C, HxW).sum(dim=[2])
    d_input = d_gamma = d_bias = None
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        if weight is not None:
            ds_val = (ds * weight.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = (db * weight.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = rstd.unsqueeze(-1) * weight.reshape(1, group, cpg)
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = rstd.unsqueeze(-1) * tp.ones((1, group, cpg), dtype=rstd.dtype, device=rstd.device)
        c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
        c3 = -c2 * mean - db_val * rstd * s
        d_input = (
            grad_out.reshape(N, group, cpg, HxW) * c1.unsqueeze(-1)
            + input.reshape(N, group, cpg, HxW) * _to_rank(c2, 4)
            + _to_rank(c3, 4)
        ).reshape(tuple(input.shape)).to(input.dtype)
    if output_mask[1]:
        d_gamma = (
            ((ds.reshape(N, group, cpg) - db.reshape(N, group, cpg) * mean.unsqueeze(-1))
             * rstd.unsqueeze(-1)).sum(dim=[0]).reshape(C)
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


# Dropout is one operation to the caller and two to the machine: the values
# that were kept, and the record of which those were, which the backward needs
# and which the caller has no use for.  So what the caller gets is written in
# terms of the pair, and the pair is what the graph below records.
@register_decomposition(ops.dropout.default)
def dropout(input, p: float, train: bool | None = None):
    if train and p != 0:
        return ops.native_dropout.default(input, p, train)[0]
    return input


@register_decomposition(ops.native_dropout.default)
def native_dropout(input, p: float, train: bool | None = None):
    if train and p != 0:
        if p == 1:
            return (tp.zeros_like(input), tp.zeros_like(input, dtype=tp.bool))
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type Long"
            )
        bool_mask = tp.rand_like(input) > p
        res = bool_mask * input * float(1.0 / (1.0 - p))
        return (res, bool_mask)
    return (input, tp.ones_like(input, dtype=tp.bool))


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


def _upsample_nearest_indices(input, output_size, scales, exact):
    # Per-axis gather indices.  The pixel-exact variant centres samples on
    # half-pixels; the plain variant floors the scaled position.  Truncation
    # toward zero matches rounding because the scaled position never goes
    # below -0.5.
    offset = 0.5 if exact else 0.0
    num_spatial = len(output_size)
    indices = []
    for d in range(num_spatial):
        osize = output_size[d]
        isize = int(input.shape[-num_spatial + d])
        scale = (
            isize / (isize * scales[d])
            if scales[d] is not None and scales[d] > 0
            else isize / osize
        )
        position = tp.arange(osize, dtype=tp.float32, device=input.device)
        idx = ((position + offset) * scale).to(tp.int64)
        for _ in range(num_spatial - 1 - d):
            idx = idx.unsqueeze(-1)
        indices.append(idx)
    return indices


def _upsample_nearest(input, output_size, scales, exact=False):
    indices = _upsample_nearest_indices(input, output_size, scales, exact)
    return ops.index.Tensor(input, [None, None] + indices).contiguous()


@register_decomposition(ops.upsample_nearest1d.default)
def upsample_nearest1d(self, output_size, scales=None):
    return _upsample_nearest(self, list(output_size), [scales])


@register_decomposition(ops._upsample_nearest_exact1d.default)
def upsample_nearest_exact1d(self, output_size, scales=None):
    return _upsample_nearest(self, list(output_size), [scales], exact=True)


@register_decomposition(ops.upsample_nearest2d.default)
def upsample_nearest2d(self, output_size, scales_h=None, scales_w=None):
    return _upsample_nearest(self, list(output_size), [scales_h, scales_w])


@register_decomposition(ops._upsample_nearest_exact2d.default)
def upsample_nearest_exact2d(self, output_size, scales_h=None, scales_w=None):
    return _upsample_nearest(self, list(output_size), [scales_h, scales_w], exact=True)


@register_decomposition(ops.upsample_nearest3d.default)
def upsample_nearest3d(self, output_size, scales_d=None, scales_h=None, scales_w=None):
    return _upsample_nearest(self, list(output_size), [scales_d, scales_h, scales_w])


@register_decomposition(ops._upsample_nearest_exact3d.default)
def upsample_nearest_exact3d(self, output_size, scales_d=None, scales_h=None, scales_w=None):
    return _upsample_nearest(self, list(output_size), [scales_d, scales_h, scales_w], exact=True)


@register_decomposition([ops.upsample_nearest1d.vec, ops.upsample_nearest2d.vec, ops.upsample_nearest3d.vec])
def upsample_nearest_vec(input, output_size, scale_factors):
    size = _upsample_output_size(input, output_size, scale_factors)
    scales = list(scale_factors) if scale_factors else [None] * len(size)
    return _upsample_nearest(input, size, scales)


@register_decomposition([ops._upsample_nearest_exact1d.vec, ops._upsample_nearest_exact2d.vec, ops._upsample_nearest_exact3d.vec])
def upsample_nearest_exact_vec(input, output_size, scale_factors):
    size = _upsample_output_size(input, output_size, scale_factors)
    scales = list(scale_factors) if scale_factors else [None] * len(size)
    return _upsample_nearest(input, size, scales, exact=True)


def _cubic_convolution1(x, a):
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0


def _cubic_convolution2(x, a):
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a


def _cubic_coefficients(t):
    # Weights for the four neighbours at offsets -1..2; the kernel uses
    # a = -0.75.  Values beyond one pixel use the outer lobe formula.
    a = -0.75
    return (
        _cubic_convolution2(t + 1.0, a),
        _cubic_convolution1(t, a),
        _cubic_convolution1(1.0 - t, a),
        _cubic_convolution2(2.0 - t, a),
    )


def _sum_tensors(tensors):
    return reduce(lambda a, b: a + b, tensors)


def _weight_precision(weights):
    # Fixed-point shift that keeps uint8 accumulation exact in int16.
    stacked = ops.stack.default(list(weights), 0)
    max_weight = ops.amax.default(stacked, list(range(stacked.dim())))
    precisions = tp.arange(22, device=max_weight.device)
    values = 0.5 + max_weight * ops.bitwise_left_shift.Scalar_Tensor(1, precisions + 1)
    mask = values >= (1 << 15)
    return 22 - ops.sum.default(mask.to(tp.int64))


def _sum_weights_uint8(sources, weights, precision):
    total = _sum_tensors(
        s.to(tp.int32) * c.to(tp.int32) for s, c in zip(sources, weights)
    ) + ops.bitwise_left_shift.Scalar_Tensor(1, precision - 1)
    total = ops.bitwise_right_shift.Tensor(total, precision)
    return ops.clamp.default(total, 0, 255).to(tp.uint8)


@register_decomposition(ops.upsample_bicubic2d.default)
def upsample_bicubic2d(self, output_size, align_corners, scales_h=None, scales_w=None):
    in_h, in_w = int(self.shape[-2]), int(self.shape[-1])
    out_h, out_w = list(output_size)
    h_scale = _upsample_scale(in_h, out_h, align_corners, scales_h)
    w_scale = _upsample_scale(in_w, out_w, align_corners, scales_w)
    compute = _computation_dtype(self.dtype) if self.is_floating_point() else tp.get_default_dtype()
    x = self.to(compute)

    i = tp.arange(out_h, device=self.device).to(compute)
    j = tp.arange(out_w, device=self.device).to(compute)
    y_float = i * h_scale if align_corners else (i + 0.5) * h_scale - 0.5
    x_float = j * w_scale if align_corners else (j + 0.5) * w_scale - 0.5
    y_float = y_float.unsqueeze(-1)

    y = y_float.floor()
    xw = x_float.floor()
    yscale = tp.clamp(y_float - y, 0.0, 1.0)
    xscale = tp.clamp(x_float - xw, 0.0, 1.0)
    y = y.to(tp.int64)
    xw = xw.to(tp.int64)

    weights_x = _cubic_coefficients(xscale)
    weights_y = _cubic_coefficients(yscale)

    precision_x = precision_y = None
    if self.dtype == tp.uint8:
        precision_x = _weight_precision(weights_x)
        precision_y = _weight_precision(weights_y)
        weights_x = [
            (w * ops.bitwise_left_shift.Scalar_Tensor(1, precision_x)
             + ops.sign.default(w) * 0.5).to(tp.int16)
            for w in weights_x
        ]
        weights_y = [
            (w * ops.bitwise_left_shift.Scalar_Tensor(1, precision_y)
             + ops.sign.default(w) * 0.5).to(tp.int16)
            for w in weights_y
        ]

    def load_bounded(ys, xs):
        y_idx = tp.clamp(ys, 0, in_h - 1)
        x_idx = tp.clamp(xs, 0, in_w - 1)
        return ops.index.Tensor(x, [None, None, y_idx, x_idx])

    def interp_x(ys):
        sources = [load_bounded(ys, xw + o) for o in (-1, 0, 1, 2)]
        if self.dtype == tp.uint8:
            return _sum_weights_uint8(sources, weights_x, precision_x)
        return _sum_tensors(s * w for s, w in zip(sources, weights_x))

    sources_y = [interp_x(y + o) for o in (-1, 0, 1, 2)]
    if self.dtype == tp.uint8:
        result = _sum_weights_uint8(sources_y, weights_y, precision_y)
    else:
        result = _sum_tensors(s * w for s, w in zip(sources_y, weights_y))
    return result.to(self.dtype).contiguous()


@register_decomposition(ops.upsample_bicubic2d.vec)
def upsample_bicubic2d_vec(input, output_size, align_corners, scale_factors):
    size = _upsample_output_size(input, output_size, scale_factors)
    scales = list(scale_factors) if scale_factors else [None, None]
    return upsample_bicubic2d(input, size, align_corners, scales[0], scales[1])


def _upsample_aa_default(op, input, output_size, align_corners, scale_factors):
    size = _upsample_output_size(input, output_size, scale_factors)
    scales = list(scale_factors) if scale_factors else [None, None]
    return op(input, size, align_corners, scales[0], scales[1])


@register_decomposition(ops._upsample_bilinear2d_aa.vec)
def _upsample_bilinear2d_aa_vec(input, output_size, align_corners, scale_factors):
    return _upsample_aa_default(ops._upsample_bilinear2d_aa.default, input, output_size, align_corners, scale_factors)


@register_decomposition(ops._upsample_bicubic2d_aa.vec)
def _upsample_bicubic2d_aa_vec(input, output_size, align_corners, scale_factors):
    return _upsample_aa_default(ops._upsample_bicubic2d_aa.default, input, output_size, align_corners, scale_factors)


@register_decomposition(ops._upsample_lanczos2d_aa.vec)
def _upsample_lanczos2d_aa_vec(input, output_size, align_corners, scale_factors):
    return _upsample_aa_default(ops._upsample_lanczos2d_aa.default, input, output_size, align_corners, scale_factors)


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


# ---------------------------------------------------------------------------
# Operations the framework computes whole, written as the operations they are
# made of.  A compiled region lowers the parts, where a whole call would be a
# boundary nothing fuses across.
# ---------------------------------------------------------------------------


register_decomposition(ops.entr.default)(special_entr)
register_decomposition(ops.xlog1py.default)(special_xlog1py)


@register_decomposition(ops.index_select.default)
def index_select(self, dim, index):
    if index.dim() > 1:
        raise RuntimeError(f"index_select(): Index is supposed to be a vector, got {index.dim()} dims")
    if index.dim() == 0:
        index = index.unsqueeze(0)
    if self.dim() == 0:
        return self.reshape(1).expand(tuple(index.shape)).contiguous()
    dim = dim % self.dim()
    return ops.index.Tensor(self, [None] * dim + [index]).contiguous()


def _norm(self, p, dim, keepdim, dtype):
    if p is None or p == "fro":
        p = 2
    return ops.linalg_vector_norm.default(self, p, dim, keepdim, dtype=dtype)


@register_decomposition([ops.norm.default, ops.norm.Scalar])
def norm(self, p=2):
    return _norm(self, p, None, False, None)


@register_decomposition(ops.norm.dim)
def norm_dim(self, dim, p=2.0, keepdim=False):
    return _norm(self, p, list(dim), keepdim, None)


@register_decomposition(ops.norm.ScalarOpt_dim)
def norm_scalaropt_dim(self, p, dim, keepdim=False):
    return _norm(self, p, list(dim), keepdim, None)


@register_decomposition(ops.norm.ScalarOpt_dtype)
def norm_scalaropt_dtype(self, p, *, dtype):
    return _norm(self, p, None, False, dtype)


@register_decomposition(ops.norm.ScalarOpt_dim_dtype)
def norm_scalaropt_dim_dtype(self, p, dim, keepdim, *, dtype):
    return _norm(self, p, list(dim), keepdim, dtype)


@register_decomposition(ops.mT.default)
def mT(self):
    if self.dim() < 2:
        raise RuntimeError(f"tensor.mT is only supported on matrices or batches of matrices. Got {self.dim()}-D tensor.")
    return self.transpose(-2, -1)


@register_decomposition([ops.mH.default, ops.adjoint.default])
def mH(self):
    if self.dim() < 2:
        raise RuntimeError(f"tensor.mH is only supported on matrices or batches of matrices. Got {self.dim()}-D tensor.")
    transposed = self.transpose(-2, -1)
    return transposed.conj() if self.is_complex() else transposed


@register_decomposition(ops.log_sigmoid.default)
def log_sigmoid(self):
    return ops.log_sigmoid_forward.default(self)[0]


@register_decomposition(ops.logsumexp.default)
def logsumexp(self, dim, keepdim=False):
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    if not (self.is_floating_point() or self.is_complex()):
        self = self.to(tp.get_default_dtype())
    if self.numel() == 0:
        return tp.log(tp.sum(tp.exp(self), dims, keepdim))
    # The largest value is taken out before exponentiating so nothing
    # overflows; an infinite largest value is not taken out, since inf - inf
    # is no number.
    maxes = tp.amax(self, dims, keepdim=True)
    maxes = tp.masked_fill(maxes, maxes.abs() == math.inf, 0)
    total = tp.sum(tp.exp(self - maxes), dims, keepdim)
    return tp.log(total) + maxes.reshape(tuple(total.shape))


@register_decomposition(ops.round.decimals)
def round_decimals(self, *, decimals=0):
    if not self.is_floating_point():
        return NotImplemented
    if decimals >= 0:
        scale = 10.0 ** decimals
        return tp.round(self * scale) / scale
    scale = 10.0 ** (-decimals)
    return tp.round(self / scale) * scale


@register_decomposition(ops.isclose.default)
def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if self.dtype != other.dtype:
        raise RuntimeError(f"{self.dtype} did not match {other.dtype}")
    if rtol < 0:
        raise RuntimeError(f"rtol must be greater than or equal to zero, but got {rtol}")
    if atol < 0:
        raise RuntimeError(f"atol must be greater than or equal to zero, but got {atol}")
    close = self == other
    if equal_nan and (self.is_floating_point() or self.is_complex()):
        close = close | (tp.isnan(self) & tp.isnan(other))
    if atol == 0 and rtol == 0:
        return close
    if not (self.is_floating_point() or self.is_complex()):
        self = self.to(tp.get_default_dtype())
        other = other.to(tp.get_default_dtype())
    allowed = atol + tp.abs(other * rtol)
    actual = tp.abs(self - other)
    return close | (tp.isfinite(actual) & (actual <= allowed))


@register_decomposition([ops.swapaxes.default, ops.swapdims.default])
def swapaxes(self, axis0, axis1):
    return self.transpose(axis0, axis1)


@register_decomposition([ops.movedim.default, ops.movedim.intlist, ops.movedim.int,
                         ops.moveaxis.intlist, ops.moveaxis.int])
def movedim(self, source, destination):
    src = [source] if isinstance(source, int) else list(source)
    dst = [destination] if isinstance(destination, int) else list(destination)
    if len(src) != len(dst):
        raise RuntimeError(
            f"movedim: Invalid source or destination dims: source ({src} dims) should "
            f"contain the same number of dims as destination ({dst} dims)"
        )
    ndim = self.dim()
    if ndim == 0:
        return self.view(())
    src = [s % ndim for s in src]
    dst = [d % ndim for d in dst]
    order = [-1] * ndim
    for s, d in zip(src, dst):
        order[d] = s
    rest = iter(i for i in range(ndim) if i not in src)
    return self.permute([o if o != -1 else next(rest) for o in order])


@register_decomposition(ops.outer.default)
def outer(self, vec2):
    if self.dim() != 1:
        raise RuntimeError(f"outer: Expected 1-D argument self, but got {self.dim()}-D")
    if vec2.dim() != 1:
        raise RuntimeError(f"outer: Expected 1-D argument vec2, but got {vec2.dim()}-D")
    return self.reshape(-1, 1) * vec2


@register_decomposition(ops.atleast_1d.default)
def atleast_1d(self):
    return self.reshape(1) if self.dim() == 0 else self


@register_decomposition(ops.atleast_2d.default)
def atleast_2d(self):
    if self.dim() == 0:
        return self.reshape(1, 1)
    if self.dim() == 1:
        return self.unsqueeze(0)
    return self


@register_decomposition(ops.atleast_3d.default)
def atleast_3d(self):
    if self.dim() == 0:
        return self.reshape(1, 1, 1)
    if self.dim() == 1:
        return self.unsqueeze(0).unsqueeze(-1)
    if self.dim() == 2:
        return self.unsqueeze(-1)
    return self


@register_decomposition(ops.atleast_1d.Sequence)
def atleast_1d_sequence(tensors):
    return [atleast_1d(t) for t in tensors]


@register_decomposition(ops.atleast_2d.Sequence)
def atleast_2d_sequence(tensors):
    return [atleast_2d(t) for t in tensors]


@register_decomposition(ops.atleast_3d.Sequence)
def atleast_3d_sequence(tensors):
    return [atleast_3d(t) for t in tensors]


@register_decomposition(ops.hstack.default)
def hstack(tensors):
    tensors = [atleast_1d(t) for t in tensors]
    return tp.cat(tensors, 0 if tensors[0].dim() == 1 else 1)


@register_decomposition(ops.vstack.default)
def vstack(tensors):
    return tp.cat([atleast_2d(t) for t in tensors], 0)


@register_decomposition(ops.unflatten.int)
def unflatten(self, dim, sizes):
    if self.dim() == 0:
        raise RuntimeError("Cannot unflatten a 0-d tensor")
    dim = dim % self.dim()
    shape = list(self.shape)
    return self.view(tuple(shape[:dim] + list(sizes) + shape[dim + 1:]))


@register_decomposition(ops.tile.default)
def tile(self, dims):
    dims = list(dims)
    if len(dims) < self.dim():
        dims = [1] * (self.dim() - len(dims)) + dims
    return self.repeat(dims)


@register_decomposition(ops.diag.default)
def diag(self, diagonal=0):
    if self.dim() == 1:
        return tp.diag_embed(self, diagonal)
    if self.dim() == 2:
        return tp.diagonal(self, diagonal).clone()
    raise RuntimeError(f"diag(): Supports 1D or 2D tensors. Got {self.dim()}D")


@register_decomposition(ops.kron.default)
def kron(self, other):
    ndim = max(self.dim(), other.dim())
    a = self.reshape((1,) * (ndim - self.dim()) + tuple(self.shape))
    b = other.reshape((1,) * (ndim - other.dim()) + tuple(other.shape))
    # Each axis of the result is an axis of ``a`` with an axis of ``b`` nested
    # inside it.
    a_spread = a.reshape(tuple(s for size in a.shape for s in (size, 1)))
    b_spread = b.reshape(tuple(s for size in b.shape for s in (1, size)))
    return (a_spread * b_spread).reshape(tuple(sa * sb for sa, sb in zip(a.shape, b.shape)))


@register_decomposition(ops.diff.default)
def diff(self, n=1, dim=-1, prepend=None, append=None):
    if self.dim() == 0:
        raise RuntimeError("diff expects input to be at least one-dimensional")
    if n < 0:
        raise RuntimeError(f"order must be non-negative but got {n}")
    dim = dim % self.dim()
    parts = []
    for extra in (prepend,):
        if extra is not None:
            parts.append(extra)
    parts.append(self)
    if append is not None:
        parts.append(append)
    x = tp.cat(parts, dim) if len(parts) > 1 else self
    for _ in range(n):
        length = int(x.shape[dim])
        if length == 0:
            break
        later, earlier = x.narrow(dim, 1, length - 1), x.narrow(dim, 0, length - 1)
        x = tp.logical_xor(later, earlier) if x.dtype == tp.bool else later - earlier
    return x.contiguous() if x is not self else x.clone()


@register_decomposition(ops.take_along_dim.default)
def take_along_dim(self, indices, dim=None):
    if dim is None:
        return tp.gather(self.reshape(-1), 0, indices.reshape(-1))
    if self.dim() != indices.dim():
        raise RuntimeError(
            "take_along_dim(): input and indices should have the same number of dimensions, "
            f"but got {self.dim()} dimensions for input, and {indices.dim()} dimensions for indices"
        )
    dim = dim % self.dim()
    shape = [1 if d == dim else max(int(a), int(b)) for d, (a, b) in enumerate(zip(self.shape, indices.shape))]
    self_shape = list(shape)
    self_shape[dim] = int(self.shape[dim])
    index_shape = list(shape)
    index_shape[dim] = int(indices.shape[dim])
    return tp.gather(self.expand(tuple(self_shape)), dim, indices.expand(tuple(index_shape)))


@register_decomposition(ops.nanmean.default)
def nanmean(self, dim=None, keepdim=False, *, dtype=None):
    if not (self.is_floating_point() or self.is_complex()):
        raise RuntimeError(f"nanmean(): expected input to have floating point or complex dtype but got {self.dtype}")
    dims = [] if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    values = self if dtype is None else self.to(dtype)
    counted = tp.logical_not(tp.isnan(values))
    total = ops.nansum.default(values, dims, keepdim)
    count = counted.sum(dims, keepdim) if dims else counted.sum()
    return total / count


@register_decomposition(ops.argsort.default)
def argsort(self, dim=-1, descending=False):
    return tp.sort(self, dim, descending)[1]


@register_decomposition(ops.argsort.stable)
def argsort_stable(self, *, stable, dim=-1, descending=False):
    return ops.sort.stable(self, stable=stable, dim=dim, descending=descending)[1]


@register_decomposition(ops.msort.default)
def msort(self):
    return tp.sort(self, 0)[0]


@register_decomposition(ops.rms_norm.default)
def rms_norm(input, normalized_shape, weight=None, eps=None):
    count = len(normalized_shape)
    if count > input.dim():
        raise RuntimeError("rms_norm: normalized_shape dim larger than input dim")
    if tuple(input.shape[input.dim() - count:]) != tuple(normalized_shape):
        raise RuntimeError("rms_norm: Input shape mismatch with normalized_shape")
    # Computed in single precision for the half-precision types and stored
    # back in the input's; an unset epsilon is that of the computation type.
    compute = tp.float64 if input.dtype == tp.float64 else tp.float32
    if eps is None:
        eps = 2.220446049250313e-16 if compute == tp.float64 else 1.1920928955078125e-07
    dims = list(range(input.dim() - count, input.dim()))
    x = input.to(compute)
    scaled = x * tp.rsqrt((x * x).mean(dims, keepdim=True) + eps)
    if weight is not None:
        scaled = scaled * weight.to(compute)
    return scaled.to(input.dtype)


@register_decomposition(ops.tp_l1_loss.default)
def tp_l1_loss(input, target, reduction=1):
    return _reduce(tp.abs(input - target), reduction)


@register_decomposition(ops.tp_kl_div.default)
def tp_kl_div(input, target, reduction=1, log_target=False):
    if log_target:
        loss = tp.exp(target) * (target - input)
    else:
        loss = tp.xlogy(target, target) - target * input
    return _reduce(loss, reduction)


@register_decomposition(ops.tp_margin_ranking_loss.default)
def tp_margin_ranking_loss(input1, input2, target, margin=0.0, reduction=1):
    return _reduce(tp.clamp(-target * (input1 - input2) + margin, min=0), reduction)


@register_decomposition(ops.tp_soft_margin_loss.default)
def tp_soft_margin_loss(input, target, reduction=1):
    return _reduce(tp.log1p(tp.exp(-target * input)), reduction)


@register_decomposition(ops.tp_hinge_embedding_loss.default)
def tp_hinge_embedding_loss(input, target, margin=1.0, reduction=1):
    zeros = tp.zeros_like(input)
    margin_part = tp.where(target != 1, tp.clamp(margin - input, min=0), zeros)
    self_part = tp.where(target != -1, input, zeros)
    return _reduce(margin_part + self_part, reduction)


@register_decomposition(ops.tp_poisson_nll_loss.default)
def tp_poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-08, reduction=1):
    if log_input:
        loss = tp.exp(input) - target * input
    else:
        loss = input - target * tp.log(input + eps)
    if full:
        stirling = target * tp.log(target) - target + 0.5 * tp.log(2 * math.pi * target)
        loss = loss + tp.masked_fill(stirling, target <= 1, 0)
    return _reduce(loss, reduction)


# ---------------------------------------------------------------------------
# Writing forms of the bitwise, logical and activation operations
#
# An in-place overload computes what the non-writing one computes and writes
# the answer into the value it was called on; the operation behind it is the
# same one, and only where the answer goes differs.
# ---------------------------------------------------------------------------


@register_decomposition([ops.bitwise_and_.Scalar, ops.bitwise_and_.Tensor])
def bitwise_and_(self, other):
    return self.copy_(ops.bitwise_and(self, other))


@register_decomposition([ops.bitwise_or_.Scalar, ops.bitwise_or_.Tensor])
def bitwise_or_(self, other):
    return self.copy_(ops.bitwise_or(self, other))


@register_decomposition([ops.bitwise_xor_.Scalar, ops.bitwise_xor_.Tensor])
def bitwise_xor_(self, other):
    return self.copy_(ops.bitwise_xor(self, other))


@register_decomposition(ops.bitwise_not_.default)
def bitwise_not_(self):
    return self.copy_(ops.bitwise_not(self))


@register_decomposition(
    [ops.bitwise_left_shift_.Tensor, ops.bitwise_left_shift_.Tensor_Scalar]
)
def bitwise_left_shift_(self, other):
    return self.copy_(ops.bitwise_left_shift(self, other))


@register_decomposition(
    [ops.bitwise_right_shift_.Tensor, ops.bitwise_right_shift_.Tensor_Scalar]
)
def bitwise_right_shift_(self, other):
    return self.copy_(ops.bitwise_right_shift(self, other))


@register_decomposition(ops.logical_and_.default)
def logical_and_(self, other):
    return self.copy_(ops.logical_and(self, other))


@register_decomposition(ops.logical_not_.default)
def logical_not_(self):
    return self.copy_(ops.logical_not(self))


@register_decomposition(ops.logical_or_.default)
def logical_or_(self, other):
    return self.copy_(ops.logical_or(self, other))


@register_decomposition(ops.logical_xor_.default)
def logical_xor_(self, other):
    return self.copy_(ops.logical_xor(self, other))


@register_decomposition(ops.relu_.default)
def relu_(self):
    return self.copy_(ops.relu(self))


@register_decomposition(ops.sigmoid_.default)
def sigmoid_(self):
    return self.copy_(ops.sigmoid(self))


@register_decomposition([ops.__iand__.Scalar, ops.__iand__.Tensor])
def __iand__(self, other):
    return self.copy_(ops.__and__(self, other))


@register_decomposition([ops.__ior__.Scalar, ops.__ior__.Tensor])
def __ior__(self, other):
    return self.copy_(ops.__or__(self, other))


@register_decomposition([ops.__ixor__.Scalar, ops.__ixor__.Tensor])
def __ixor__(self, other):
    return self.copy_(ops.__xor__(self, other))


@register_decomposition([ops.__ilshift__.Scalar, ops.__ilshift__.Tensor])
def __ilshift__(self, other):
    return self.copy_(ops.__lshift__(self, other))


@register_decomposition([ops.__irshift__.Scalar, ops.__irshift__.Tensor])
def __irshift__(self, other):
    return self.copy_(ops.__rshift__(self, other))


# ---------------------------------------------------------------------------
# A value carried under a new name
#
# Lifting retags a value as one a graph owns: the values and the sharing
# underneath are the ones it already has, so what comes back is the value
# itself.
# ---------------------------------------------------------------------------


@register_decomposition([ops.lift.default, ops.lift_fresh.default])
def lift(x):
    return ops.alias(x)


# ---------------------------------------------------------------------------
# A shape read at a layout of its own, a scatter into such a shape, and a
# place made to a shape and a layout
# ---------------------------------------------------------------------------


@register_decomposition(ops.as_strided_copy.default)
def as_strided_copy(self, size, stride, storage_offset=None):
    return ops.as_strided(self, size, stride, storage_offset).clone(
        memory_format=tp.contiguous_format
    )


@register_decomposition(ops.as_strided_scatter.default)
def as_strided_scatter(input, src, size, stride, storage_offset=None):
    # The distances are measured from the start of the storage, so an offset
    # nobody stated is the start of it rather than a question about the input.
    offset = 0 if storage_offset is None else storage_offset
    return prims.as_strided_scatter(input, src, size, stride, offset)


@register_decomposition(ops.new_empty_strided.default)
def new_empty_strided(
    self, size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    if layout is not None and layout != tp.strided:
        raise NotImplementedError(f"layout={layout}")
    return ops.empty_strided(
        size,
        stride,
        dtype=_dtype_or(dtype, self.dtype),
        device=device or self.device,
        pin_memory=bool(pin_memory),
    )


# ---------------------------------------------------------------------------
# A draw of normal values, written as a read of a distribution at a shape
# ---------------------------------------------------------------------------


@register_decomposition(ops.randn.default)
def randn(size, *, dtype=None, device=None, requires_grad=False):
    return prims.normal(
        size,
        mean=0.0,
        std=1.0,
        dtype=_dtype_or(dtype, tp.get_default_dtype()),
        device=device or tp.get_default_device(),
        requires_grad=requires_grad,
    )


# ---------------------------------------------------------------------------
# How many positions a value has, as a number rather than as a read
# ---------------------------------------------------------------------------


@register_decomposition(ops.sym_numel.default)
def sym_numel(t):
    return reduce(operator.mul, t.shape, 1)


# ---------------------------------------------------------------------------
# Writing forms of the arithmetic, comparison and rounding operations
#
# The same pattern as the writing forms above: an in-place overload computes
# what the non-writing one computes and copies the answer into the value it
# was called on.  The eager kernels reject writes that would change the value
# of a dtype the target cannot hold; the copy here carries the same result,
# which is why no additional promotion decision is made on the way in.
# ---------------------------------------------------------------------------


@register_decomposition([ops.add_.Scalar, ops.add_.Tensor])
def add_(self, other, alpha=1):
    return self.copy_(ops.add(self, other, alpha=alpha))


@register_decomposition([ops.sub_.Scalar, ops.sub_.Tensor])
def sub_(self, other, alpha=1):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.sub.Tensor(self, other, alpha=alpha))
    return self.copy_(ops.sub.Scalar(self, other, alpha=alpha))


@register_decomposition([ops.mul_.Scalar, ops.mul_.Tensor])
def mul_(self, other):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.mul.Tensor(self, other))
    return self.copy_(ops.mul.Scalar(self, other))


@register_decomposition([ops.div_.Scalar, ops.div_.Tensor])
def div_(self, other):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.div.Tensor(self, other))
    return self.copy_(ops.div.Scalar(self, other))


@register_decomposition([ops.div_.Scalar_mode, ops.div_.Tensor_mode])
def div_mode_(self, other, rounding_mode=None):
    if isinstance(other, tp.Tensor):
        return self.copy_(
            ops.div.Tensor_mode(self, other, rounding_mode=rounding_mode)
        )
    return self.copy_(ops.div.Scalar_mode(self, other, rounding_mode=rounding_mode))


@register_decomposition([ops.true_divide_.Scalar, ops.true_divide_.Tensor])
def true_divide_(self, other):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.true_divide.Tensor(self, other))
    return self.copy_(ops.true_divide.Scalar(self, other))


@register_decomposition([ops.remainder_.Scalar, ops.remainder_.Tensor])
def remainder_(self, other):
    return self.copy_(ops.remainder(self, other))


@register_decomposition([ops.fmod_.Scalar, ops.fmod_.Tensor])
def fmod_(self, other):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.fmod.Tensor(self, other))
    return self.copy_(ops.fmod.Scalar(self, other))


@register_decomposition([ops.pow_.Scalar, ops.pow_.Tensor])
def pow_(self, exponent):
    return self.copy_(ops.pow(self, exponent))


@register_decomposition([ops.float_power_.Scalar, ops.float_power_.Tensor])
def float_power_(self, exponent):
    return self.copy_(ops.float_power(self, exponent))


@register_decomposition([ops.copysign_.Scalar, ops.copysign_.Tensor])
def copysign_(self, other):
    if isinstance(other, tp.Tensor):
        return self.copy_(ops.copysign.Tensor(self, other))
    return self.copy_(ops.copysign.Scalar(self, other))


@register_decomposition(ops.atan2_.default)
def atan2_(self, other):
    return self.copy_(ops.atan2(self, other))


@register_decomposition(ops.hypot_.default)
def hypot_(self, other):
    return self.copy_(ops.hypot(self, other))


@register_decomposition(ops.ldexp_.default)
def ldexp_(self, other):
    return self.copy_(ops.ldexp(self, other))


@register_decomposition(ops.nextafter_.default)
def nextafter_(self, other):
    return self.copy_(ops.nextafter(self, other))


@register_decomposition(ops.gcd_.default)
def gcd_(self, other):
    return self.copy_(ops.gcd(self, other))


@register_decomposition(ops.lcm_.default)
def lcm_(self, other):
    return self.copy_(ops.lcm(self, other))


@register_decomposition(ops.igamma_.default)
def igamma_(self, other):
    return self.copy_(ops.igamma(self, other))


@register_decomposition(ops.igammac_.default)
def igammac_(self, other):
    return self.copy_(ops.igammac(self, other))


@register_decomposition([ops.eq_.Scalar, ops.eq_.Tensor])
def eq_(self, other):
    return self.copy_(ops.eq(self, other))


@register_decomposition([ops.ne_.Scalar, ops.ne_.Tensor])
def ne_(self, other):
    return self.copy_(ops.ne(self, other))


@register_decomposition([ops.lt_.Scalar, ops.lt_.Tensor])
def lt_(self, other):
    return self.copy_(ops.lt(self, other))


@register_decomposition([ops.le_.Scalar, ops.le_.Tensor])
def le_(self, other):
    return self.copy_(ops.le(self, other))


@register_decomposition([ops.gt_.Scalar, ops.gt_.Tensor])
def gt_(self, other):
    return self.copy_(ops.gt(self, other))


@register_decomposition([ops.ge_.Scalar, ops.ge_.Tensor])
def ge_(self, other):
    return self.copy_(ops.ge(self, other))


@register_decomposition([ops.clamp_.default, ops.clamp_.Tensor])
def clamp_(self, min=None, max=None):
    return self.copy_(ops.clamp(self, min, max))


# ---------------------------------------------------------------------------
# Writing forms of the unary mathematical functions
# ---------------------------------------------------------------------------


@register_decomposition(ops.abs_.default)
def abs_(self):
    return self.copy_(ops.abs(self))


@register_decomposition(ops.neg_.default)
def neg_(self):
    return self.copy_(ops.neg(self))


@register_decomposition(ops.reciprocal_.default)
def reciprocal_(self):
    return self.copy_(ops.reciprocal(self))


@register_decomposition(ops.sqrt_.default)
def sqrt_(self):
    return self.copy_(ops.sqrt(self))


@register_decomposition(ops.rsqrt_.default)
def rsqrt_(self):
    return self.copy_(ops.rsqrt(self))


@register_decomposition(ops.square_.default)
def square_(self):
    return self.copy_(ops.square(self))


@register_decomposition(ops.sign_.default)
def sign_(self):
    return self.copy_(ops.sign(self))


@register_decomposition(ops.exp_.default)
def exp_(self):
    return self.copy_(ops.exp(self))


@register_decomposition(ops.exp2_.default)
def exp2_(self):
    return self.copy_(ops.exp2(self))


@register_decomposition(ops.expm1_.default)
def expm1_(self):
    return self.copy_(ops.expm1(self))


@register_decomposition(ops.log_.default)
def log_(self):
    return self.copy_(ops.log(self))


@register_decomposition(ops.log2_.default)
def log2_(self):
    return self.copy_(ops.log2(self))


@register_decomposition(ops.log10_.default)
def log10_(self):
    return self.copy_(ops.log10(self))


@register_decomposition(ops.log1p_.default)
def log1p_(self):
    return self.copy_(ops.log1p(self))


@register_decomposition(ops.floor_.default)
def floor_(self):
    return self.copy_(ops.floor(self))


@register_decomposition(ops.ceil_.default)
def ceil_(self):
    return self.copy_(ops.ceil(self))


@register_decomposition(ops.trunc_.default)
def trunc_(self):
    return self.copy_(ops.trunc(self))


@register_decomposition(ops.round_.default)
def round_(self):
    return self.copy_(ops.round(self))


@register_decomposition(ops.round_.decimals)
def round_decimals_(self, decimals):
    return self.copy_(ops.round.decimals(self, decimals=decimals))


@register_decomposition(ops.sin_.default)
def sin_(self):
    return self.copy_(ops.sin(self))


@register_decomposition(ops.cos_.default)
def cos_(self):
    return self.copy_(ops.cos(self))


@register_decomposition(ops.tan_.default)
def tan_(self):
    return self.copy_(ops.tan(self))


@register_decomposition(ops.asin_.default)
def asin_(self):
    return self.copy_(ops.asin(self))


@register_decomposition(ops.acos_.default)
def acos_(self):
    return self.copy_(ops.acos(self))


@register_decomposition(ops.atan_.default)
def atan_(self):
    return self.copy_(ops.atan(self))


@register_decomposition(ops.sinh_.default)
def sinh_(self):
    return self.copy_(ops.sinh(self))


@register_decomposition(ops.cosh_.default)
def cosh_(self):
    return self.copy_(ops.cosh(self))


@register_decomposition(ops.tanh_.default)
def tanh_(self):
    return self.copy_(ops.tanh(self))


@register_decomposition(ops.asinh_.default)
def asinh_(self):
    return self.copy_(ops.asinh(self))


@register_decomposition(ops.acosh_.default)
def acosh_(self):
    return self.copy_(ops.acosh(self))


@register_decomposition(ops.atanh_.default)
def atanh_(self):
    return self.copy_(ops.atanh(self))


@register_decomposition(ops.erf_.default)
def erf_(self):
    return self.copy_(ops.erf(self))


@register_decomposition(ops.erfc_.default)
def erfc_(self):
    return self.copy_(ops.erfc(self))


@register_decomposition(ops.erfinv_.default)
def erfinv_(self):
    return self.copy_(ops.erfinv(self))


@register_decomposition(ops.digamma_.default)
def digamma_(self):
    return self.copy_(ops.digamma(self))


@register_decomposition(ops.lgamma_.default)
def lgamma_(self):
    return self.copy_(ops.lgamma(self))


@register_decomposition(ops.i0_.default)
def i0_(self):
    return self.copy_(ops.i0(self))


@register_decomposition(ops.conj_physical_.default)
def conj_physical_(self):
    return self.copy_(ops.conj_physical(self))


# ---------------------------------------------------------------------------
# Writing forms of the running sums, the products of small matrix chains and
# the gated activation
# ---------------------------------------------------------------------------


@register_decomposition(ops.cumsum_.default)
def cumsum_(self, dim=0, dtype=None):
    return self.copy_(ops.cumsum(self, dim, dtype=dtype))


@register_decomposition(ops.cumprod_.default)
def cumprod_(self, dim=0, dtype=None):
    return self.copy_(ops.cumprod(self, dim, dtype=dtype))


@register_decomposition(ops.addbmm_.default)
def addbmm_(self, batch1, batch2, beta=1, alpha=1):
    return self.copy_(ops.addbmm(self, batch1, batch2, beta=beta, alpha=alpha))


@register_decomposition(ops.addmm_.default)
def addmm_(self, mat1, mat2, beta=1, alpha=1):
    return self.copy_(ops.addmm(self, mat1, mat2, beta=beta, alpha=alpha))


@register_decomposition(ops.addmv_.default)
def addmv_(self, mat, vec, beta=1, alpha=1):
    return self.copy_(ops.addmv(self, mat, vec, beta=beta, alpha=alpha))


@register_decomposition(ops.selu_.default)
def selu_(self):
    return self.copy_(ops.selu(self))


# ---------------------------------------------------------------------------
# Writing forms of the reads and writes at positions
#
# Each is the non-writing operation's answer copied into the value it was
# called on, exactly like the arithmetic forms above; the write at a position
# needs no decision of its own because the read at that position already made
# every one.
# ---------------------------------------------------------------------------


@register_decomposition(ops.scatter_.src)
def scatter_src_(self, dim, index, src):
    return self.copy_(ops.scatter.src(self, dim, index, src))


@register_decomposition(ops.scatter_.value)
def scatter_value_(self, dim, index, value):
    return self.copy_(ops.scatter.value(self, dim, index, value))


@register_decomposition(ops.scatter_.reduce)
def scatter_reduce_(self, dim, index, src, reduce):
    return self.copy_(ops.scatter.reduce(self, dim, index, src, reduce=reduce))


@register_decomposition(ops.scatter_.value_reduce)
def scatter_value_reduce_(self, dim, index, value, reduce):
    return self.copy_(
        ops.scatter.value_reduce(self, dim, index, value, reduce=reduce)
    )


@register_decomposition(ops.scatter_add_.default)
def scatter_add_(self, dim, index, src):
    return self.copy_(ops.scatter_add(self, dim, index, src))


@register_decomposition(ops.scatter_reduce_.two)
def scatter_reduce_two_(self, dim, index, src, reduce, include_self=True):
    return self.copy_(
        ops.scatter_reduce(self, dim, index, src, reduce, include_self=include_self)
    )


@register_decomposition(ops.index_put_.default)
def index_put_(self, indices, values, accumulate=False):
    return self.copy_(ops.index_put(self, indices, values, accumulate=accumulate))


@register_decomposition(ops.index_reduce_.default)
def index_reduce_(self, dim, index, source, reduce, include_self=True):
    return self.copy_(
        ops.index_reduce(self, dim, index, source, reduce, include_self=include_self)
    )


# ---------------------------------------------------------------------------
# Writing forms of the distributions
#
# A fill reads a draw and leaves it in the value it was called on.  The uniform
# reads come out of the uniform prim at the value's own layout; the rest are
# the standard inverses of the distributions' distribution functions applied
# to a uniform or a normal draw.  A generator cannot be carried into the read,
# so one handed here is refused rather than silently ignored.
# ---------------------------------------------------------------------------


def _no_generator(op, generator):
    if generator is not None:
        raise AssertionError(f"{op} does not accept a generator when written as its parts")


@register_decomposition(ops.uniform_.default)
def uniform_(self, low=0, high=1, generator=None):
    _no_generator("uniform_", generator)
    return self.copy_(
        prims.uniform(
            list(self.shape),
            low=low,
            high=high,
            dtype=self.dtype,
            device=self.device,
            stride=list(self.stride()),
        )
    )


@register_decomposition(ops.normal_.default)
def normal_(self, mean=0, std=1, generator=None):
    if isinstance(mean, tp.Tensor) or isinstance(std, tp.Tensor):
        return self.copy_(ops.normal(mean, std, generator=generator))
    return self.copy_(ops.normal(mean, std, list(self.shape), generator=generator))


@register_decomposition(ops.cauchy_.default)
def cauchy_(self, median=0, sigma=1, generator=None):
    _no_generator("cauchy_", generator)
    u = ops.rand_like(self)
    return self.copy_(median + sigma * ops.tan(math.pi * (u - 0.5)))


@register_decomposition(ops.exponential_.default)
def exponential_(self, lambd=1, generator=None):
    _no_generator("exponential_", generator)
    u = ops.rand_like(self)
    # The draw lands in (0, 1); where it rounds to 1 the logarithm would be 0
    # and the inverse exponential would be 0, so the log is clamped to -eps/2.
    eps = tp.finfo(u.dtype).eps / 2
    log_u = prims.where(u >= 1.0 - eps, -eps, ops.log(u))
    return self.copy_(-log_u / lambd)


@register_decomposition(ops.geometric_.default)
def geometric_(self, p, generator=None):
    _no_generator("geometric_", generator)
    u = ops.rand_like(self)
    return self.copy_(ops.floor(ops.log1p(-u) / math.log1p(-p)) + 1)


@register_decomposition(ops.log_normal_.default)
def log_normal_(self, mean=1, std=2, generator=None):
    _no_generator("log_normal_", generator)
    n = ops.randn_like(self)
    return self.copy_(ops.exp(std * n + mean))


# ---------------------------------------------------------------------------
# Views and their copies
#
# A view is expressed as a strided window on the buffer it came from: sizes,
# strides and a storage offset.  The copy form of a view op materializes the
# same window into a fresh contiguous buffer.
# ---------------------------------------------------------------------------


def _view_strided(a, size, stride, storage_offset):
    return a.as_strided(list(size), list(stride), storage_offset)


@register_decomposition(ops.alias.default)
def alias(a):
    return prims.view_of(a)


@register_decomposition(ops.clone.default)
def clone(a, *, memory_format=None):
    return prims.clone(a, memory_format=memory_format)


@register_decomposition(ops.unsqueeze.default)
def unsqueeze(a, dim):
    # The new axis has length one, so a zero stride addresses it correctly.
    dim = canonicalize_dim(a.ndim + 1, dim)
    size = list(a.shape)
    stride = list(a.stride())
    size.insert(dim, 1)
    stride.insert(dim, 0)
    return _view_strided(a, size, stride, a.storage_offset())


def _squeeze_all(a):
    dims = tuple(i for i, length in enumerate(a.shape) if length == 1)
    if dims:
        return prims.squeeze(a, list(dims))
    return prims.view_of(a)


@register_decomposition([ops.squeeze.default, ops.squeeze.dim, ops.squeeze.dims])
def squeeze(a, dim=None):
    if dim is None:
        return _squeeze_all(a)
    if a.ndim == 0:
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        if dims not in ((), (0,)):
            raise RuntimeError(f"Expected dims to be empty or (0,) for 0-dim tensor, got {dims}")
        return prims.view_of(a)
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    dims = canonicalize_dims(a.ndim, dims)
    # Axes that are not of length one pass through untouched.
    dims = tuple(d for d in dims if a.shape[d] == 1)
    if not dims:
        return prims.view_of(a)
    if len(dims) == 1:
        return prims.squeeze(a, list(dims))
    for d in sorted(dims, reverse=True):
        a = prims.squeeze(a, [d])
    return a


@register_decomposition(ops.permute.default)
def permute(a, dims):
    return prims.transpose(a, canonicalize_dims(a.ndim, dims))


@register_decomposition(ops.expand.default)
def expand(a, size, *, implicit=False):
    if len(size) < len(a.shape):
        raise RuntimeError("expand: the requested shape has too few dimensions!")
    offset = len(size) - len(a.shape)
    shape_ = list(size)
    for idx, x in enumerate(a.shape):
        requested = shape_[idx + offset]
        if requested == -1:
            # A request of -1 keeps the incoming length; a leading axis the
            # tensor does not reach has nothing to keep.
            shape_[idx + offset] = x
        else:
            if x != 1 and requested != x:
                raise RuntimeError(
                    f"expand: attempting to expand a dimension of length {x} -> {requested}!"
                )
            shape_[idx + offset] = requested
    for i in range(offset):
        if shape_[i] == -1:
            raise RuntimeError(
                f"The expanded size of the tensor ({shape_[i]}) isn't allowed "
                f"in a leading, non-existing dimension {i}"
            )
    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, offset + len(a.shape)))
    )


@register_decomposition(ops.flip.default)
def flip(a, dims=()):
    dims = canonicalize_dims(a.ndim, dims)
    if len(set(dims)) != len(dims):
        raise RuntimeError("flip: dims may not repeat")
    if a.ndim == 0:
        return prims.clone(a)
    return prims.rev(a, list(dims))


@register_decomposition(ops.slice.Tensor)
def slice_forward(self, dim=0, start=None, end=None, step=1):
    if self.ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = canonicalize_dim(self.ndim, dim)
    sizes = list(self.shape)
    strides = list(self.stride())
    if step <= 0:
        raise RuntimeError("slice step must be positive")
    start_val = 0 if start is None else start
    end_val = sys.maxsize if end is None else end
    if start_val < 0:
        start_val += sizes[dim]
    if end_val < 0:
        end_val += sizes[dim]
    if start_val < 0:
        start_val = 0
    elif start_val > sizes[dim]:
        start_val = sizes[dim]
    if end_val == sys.maxsize:
        end_val = sizes[dim]
    elif end_val < start_val:
        end_val = start_val
    elif end_val > sizes[dim]:
        end_val = sizes[dim]
    storage_offset = self.storage_offset() + start_val * strides[dim]
    length = end_val - start_val
    sizes[dim] = (length + step - 1) // step
    strides[dim] *= step
    return _view_strided(self, sizes, strides, storage_offset)


@register_decomposition(ops.slice_scatter.default)
def slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    dim = canonicalize_dim(self.ndim, dim)
    result = _clone_preserving_strides(self)
    window = slice_forward(result, dim, start, end, step)
    if tuple(window.shape) != tuple(src.shape):
        raise RuntimeError(
            f"expected src to have a size equal to the target slice. "
            f"src size = {tuple(src.shape)}, slice size = {tuple(window.shape)}"
        )
    prims.copy_to(window, src)
    return result


@register_decomposition(ops.split_with_sizes.default)
def split_with_sizes(self, split_sizes, dim=0):
    for split_size in split_sizes:
        if split_size < 0:
            raise RuntimeError(
                "split_with_sizes expects split_sizes have only non-negative entries"
            )
    dim = canonicalize_dim(self.ndim, dim)
    total = sum(split_sizes)
    if total != self.shape[dim]:
        raise RuntimeError(
            f"Split sizes add up to {total} but got the tensor's size of {self.shape[dim]}"
        )
    splits = []
    offset = self.storage_offset()
    stride = list(self.stride())
    for split_size in split_sizes:
        new_shape = list(self.shape)
        new_shape[dim] = split_size
        splits.append(_view_strided(self, new_shape, stride, offset))
        offset += stride[dim] * split_size
    return splits


def _get_unfold_shape_stride(a_shape, a_stride, dimension, size, step):
    a_ndim = len(a_shape)
    dim = canonicalize_dim(a_ndim, dimension, wrap_scalar=True)
    max_size = 1 if a_ndim == 0 else a_shape[dim]
    last_stride = 1 if a_ndim == 0 else a_stride[dim]
    if size > max_size:
        raise RuntimeError(
            f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}"
        )
    if step <= 0:
        raise RuntimeError(f"Step is {step} but must be > 0")
    shape = list(a_shape)
    strides = list(a_stride)
    shape.append(size)
    strides.append(last_stride)
    if dim < a_ndim:
        shape[dim] = (shape[dim] - size) // step + 1
        strides[dim] *= step
    return shape, strides


@register_decomposition(ops.unfold.default)
def unfold(self, dimension, size, step):
    shape, strides = _get_unfold_shape_stride(
        self.shape, self.stride(), dimension, size, step
    )
    return _view_strided(self, shape, strides, self.storage_offset())


@register_decomposition(ops.diagonal.default)
def diagonal(self, offset=0, dim1=0, dim2=1):
    num_dims = self.ndim
    dim1 = canonicalize_dim(num_dims, dim1)
    dim2 = canonicalize_dim(num_dims, dim2)
    if dim1 == dim2:
        raise RuntimeError(f"diagonal dimensions cannot be identical {dim1}, {dim2}")
    storage_offset = self.storage_offset()
    if offset >= 0:
        diag_size = tp.sym_max(tp.sym_min(self.shape[dim1], self.shape[dim2] - offset), 0)
        storage_offset += offset * self.stride()[dim2]
    else:
        diag_size = tp.sym_max(tp.sym_min(self.shape[dim1] + offset, self.shape[dim2]), 0)
        storage_offset -= offset * self.stride()[dim1]
    sizes = [s for i, s in enumerate(self.shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)
    strides = [s for i, s in enumerate(self.stride()) if i not in (dim1, dim2)]
    strides.append(self.stride()[dim1] + self.stride()[dim2])
    return _view_strided(self, sizes, strides, storage_offset)


def _clone_preserving_strides(a):
    buffer = prims.empty_strided(
        list(a.shape),
        list(a.stride()),
        dtype=a.dtype,
        device=a.device,
        requires_grad=False,
    )
    prims.copy_to(buffer, a)
    return buffer


@register_decomposition(ops.diagonal_scatter.default)
def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    # Internal overlap (a length above one carried on a zero stride) forbids
    # keeping the strides: the writes below would land on aliased positions,
    # so the scatter goes through a contiguous buffer instead.
    if any(sz > 1 and st == 0 for sz, st in zip(input.shape, input.stride())):
        out = prims.clone(input, memory_format=tp.contiguous_format)
    else:
        out = _clone_preserving_strides(input)
    diag = diagonal(out, offset, dim1, dim2)
    if tuple(diag.shape) != tuple(src.shape):
        raise RuntimeError(
            f"expected src to have a size equal to the diagonal of the input."
            f"Got {tuple(src.shape)} for a diagonal of shape {tuple(diag.shape)}"
        )
    prims.copy_to(diag, src)
    return out


def _dims_collapsible(a, start, end):
    """Whether collapsing ``start..end`` (inclusive) keeps the buffer a view.

    Length-one axes impose nothing on the strides, so only the axes longer
    than one take part: walking inward, each stride must equal the next
    stride scaled by the sizes it spans.
    """
    shape = list(a.shape)
    strides = list(a.stride())
    pairs = [
        (shape[i], strides[i]) for i in range(start, end + 1) if shape[i] != 1
    ]
    for (outer_size, outer_stride), (inner_size, inner_stride) in zip(
        pairs, pairs[1:]
    ):
        if outer_stride != inner_stride * inner_size:
            return False
    return True


def _reshape_view_core_alg(a, shape):
    # Dimensions of the requested shape are built left to right: incoming
    # axes are collapsed until one stretch covers a requested length, which
    # is then carved out with a split.  Tail length-one axes are appended by
    # splitting the last axis so the inner stride stays the original one.
    idx = 0
    a_ = a
    for length in shape:
        if idx >= a_.ndim:
            if length != 1:
                raise RuntimeError(
                    f"Cannot unsqueeze dimension with length {length}, expected 1"
                )
            last_dim = a_.ndim - 1
            a_ = prims.split_dim(a_, last_dim, a_.shape[last_dim])
            idx += 1
            continue
        if length == a_.shape[idx]:
            idx += 1
            continue
        accum = a_.shape[idx]
        end = idx
        while accum % length != 0:
            end += 1
            if end >= a_.ndim:
                raise RuntimeError(
                    f"Cannot view a tensor with shape {list(a.shape)} and strides "
                    f"{list(a.stride())} as a tensor with shape {list(shape)}!"
                )
            accum *= a_.shape[end]
        if end != idx:
            if not _dims_collapsible(a_, idx, end):
                raise RuntimeError(
                    f"view size is not compatible with input tensor's size and stride "
                    f"(at dimension {idx}, required length {length})"
                )
            a_ = prims.collapse_view(a_, idx, end)
        if accum != length:
            a_ = prims.split_dim(a_, idx, length)
        idx += 1
    while idx < a_.ndim:
        if a_.shape[idx] != 1:
            raise RuntimeError(
                f"a.size({idx}) expected to be 1 but got {a_.shape[idx]}"
            )
        a_ = prims.squeeze(a_, [idx])
    if a_ is a:
        return prims.view_of(a)
    return a_


@register_decomposition(ops.view.default)
def view(a, size):
    shape = infer_size(list(size), a.numel())
    if a.numel() == 0:
        return _view_strided(a, shape, make_contiguous_strides_for(shape), a.storage_offset())
    if a.ndim == 0:
        _a = a
        for length in shape:
            if length != 1:
                raise RuntimeError(
                    f"Cannot reshape 0-dim tensor: shape dimension must be 1, got {length}"
                )
            _a = unsqueeze(_a, -1)
        return _a if _a is not a else prims.view_of(a)
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            if length != 1:
                raise RuntimeError(
                    f"Cannot reshape to 0-dim tensor: shape dimension must be 1, got {length}"
                )
            _a = squeeze(_a, -1)
        return _a if _a is not a else prims.view_of(a)
    if is_contiguous_or_false(a):
        if len(shape) == 1 and a.ndim > 1:
            return _view_strided(a, [a.numel()], [1], a.storage_offset())
        if len(shape) == 2 and a.ndim == 1:
            return _view_strided(a, shape, [shape[1], 1], a.storage_offset())
    shape_numel = reduce(operator.mul, shape, 1)
    if a.numel() != shape_numel:
        raise RuntimeError(
            f"shape '{list(shape)}' is invalid for input of size {a.numel()}"
        )
    return _reshape_view_core_alg(a, shape)


@register_decomposition(ops.cat.default)
def cat(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError("cat expects at least one tensor, but received zero!")
    # A 1-D zero-length input may ride along with any rank; every other
    # input must agree on the rank of the output.
    example = next((t for t in tensors if t.ndim != 1), tensors[0])
    for i, t in enumerate(tensors):
        if t.ndim != 1 and t.ndim != example.ndim:
            raise RuntimeError(
                f"Number of dimensions of tensors must match.  Expected "
                f"{example.ndim}-D tensors, but got {t.ndim}-D for tensor number {i} in the list"
            )
    filtered = []
    for i, t in enumerate(tensors):
        if len(example.shape) != len(t.shape):
            if t.ndim != 1:
                raise AssertionError(f"tensor.ndim should be 1 at this point, got {t.ndim}")
            if t.shape[0] != 0:
                raise RuntimeError(
                    f"Number of dimensions of tensors must match.  Expected "
                    f"{example.ndim}-D tensors, but got 1-D for tensor number {i} in the list"
                )
        else:
            if t.ndim == 1 and t.shape[0] == 0:
                continue
            filtered.append(t)
    if len(filtered) == 0:
        t = tensors[0]
        return prims.empty_strided(
            [0], [1], dtype=t.dtype, device=t.device, requires_grad=False
        )
    dim = canonicalize_dim(filtered[0].ndim, dim)
    return prims.clone(prims.cat(filtered, dim), memory_format=tp.contiguous_format)


@register_decomposition([ops.meshgrid.default, ops.meshgrid.indexing])
def meshgrid(tensors, indexing="ij"):
    if len(tensors) == 0:
        raise RuntimeError("meshgrid expects a non-empty TensorList")
    for i in range(len(tensors) - 1):
        if tensors[i].dtype != tensors[i + 1].dtype:
            raise RuntimeError("meshgrid expects all tensors to have the same dtype")
        if tensors[i].device != tensors[i + 1].device:
            raise RuntimeError("meshgrid expects all tensors to have the same device")
    swap_first_two = False
    if indexing == "xy":
        swap_first_two = len(tensors) >= 2
        if swap_first_two:
            tensors = (tensors[1], tensors[0], *tensors[2:])
    elif indexing != "ij":
        raise RuntimeError(
            f'meshgrid: indexing must be one of "xy" or "ij", but received: {indexing}'
        )
    for t in tensors:
        if t.ndim > 1:
            raise RuntimeError(f"meshgrid: Expected 0D or 1D tensor in the tensor list but got: {t}")
    result_shape = [t.numel() for t in tensors]
    grids = []
    for i, t in enumerate(tensors):
        if t.ndim == 0:
            t = unsqueeze(t, 0)
        grids.append(prims.broadcast_in_dim(t, result_shape, (i,)))
    if swap_first_two:
        grids[0], grids[1] = grids[1], grids[0]
    return grids


@register_decomposition(ops.constant_pad_nd.default)
def constant_pad_nd(self, pad, value=0):
    if len(pad) % 2 != 0:
        raise RuntimeError(f"Length of pad must be even but instead it equals {len(pad)}")
    input_sizes = list(self.shape)
    l_inp = len(input_sizes)
    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad
    if l_inp < l_pad:
        raise RuntimeError(
            "Length of pad should be no more than twice the number of "
            f"dimensions of the input. Pad length is {len(pad)} while the input has "
            f"{l_inp} dimensions."
        )
    c_input = self
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] < 0:
            c_input = ops.narrow.default(
                c_input, i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx]
            )
        if pad[pad_idx + 1] < 0:
            c_input = ops.narrow.default(
                c_input, i, 0, c_input.shape[i] + pad[pad_idx + 1]
            )
    if all(p < 0 for p in pad):
        return prims.clone(c_input)
    if value == 0 and self.dtype == tp.bool:
        value = False
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        left = max(pad[pad_idx], 0)
        right = max(pad[pad_idx + 1], 0)
        if left == 0 and right == 0:
            continue
        parts = []
        if left > 0:
            left_shape = list(c_input.shape)
            left_shape[i] = left
            parts.append(tp.full(left_shape, value, dtype=self.dtype, device=self.device))
        parts.append(c_input)
        if right > 0:
            right_shape = list(c_input.shape)
            right_shape[i] = right
            parts.append(tp.full(right_shape, value, dtype=self.dtype, device=self.device))
        c_input = ops.cat.default(parts, i)
    return prims.clone(c_input, memory_format=tp.contiguous_format)


@register_decomposition(ops.repeat.default)
def repeat(a, repeats):
    if len(repeats) < len(a.shape):
        raise RuntimeError(
            "Number of dimensions of repeat dims can not be smaller than "
            "number of dimensions of tensor"
        )
    if len(repeats) == 0:
        return prims.clone(a)
    num_new_dimensions = len(repeats) - a.ndim
    padded_shape = [1] * num_new_dimensions + list(a.shape)
    target_shape = [
        padded_size * repeat_size
        for padded_size, repeat_size in zip(padded_shape, repeats)
    ]
    if 0 in repeats:
        return prims.empty_strided(
            target_shape,
            make_contiguous_strides_for(target_shape),
            dtype=a.dtype,
            device=a.device,
            requires_grad=False,
        )
    # The tiled buffer is first described as an interleaved layout (one axis
    # per tile count and one per incoming length, the length axes appended
    # in order), then read in stride order and re-strided to the target.
    urtensor_shape = list(target_shape)
    urtensor_stride = make_contiguous_strides_for(target_shape)
    for dim, dim_size in enumerate(padded_shape):
        urtensor_shape, urtensor_stride = _get_unfold_shape_stride(
            urtensor_shape, urtensor_stride, dim, dim_size, max(dim_size, 1)
        )
    enumerated_stride = sorted(
        enumerate(urtensor_stride), key=operator.itemgetter(1), reverse=True
    )
    permute_order = [i for i, _ in enumerated_stride]
    repeat_xtensor = expand(a, urtensor_shape)
    cloned_result = prims.clone(repeat_xtensor, memory_format=tp.contiguous_format)
    permuted_result = permute(cloned_result, permute_order)
    if not is_contiguous_or_false(permuted_result):
        permuted_result = prims.clone(
            permuted_result, memory_format=tp.contiguous_format
        )
    return _view_strided(
        permuted_result,
        target_shape,
        make_contiguous_strides_for(target_shape),
        permuted_result.storage_offset(),
    )


def _trilu_checks(name, row, col, dtype):
    if row < 0:
        raise RuntimeError(f"row must be non-negative, got {row}")
    if col < 0:
        raise RuntimeError(f"col must be non-negative, got {col}")
    if dtype not in (tp.int32, tp.int64):
        raise RuntimeError(f'"{name}" not implemented for {dtype}')


def _get_tril_sizes(row, col, offset):
    if row == 0 or col == 0:
        return 0, 0, 0
    m_first_row = min(col, 1 + offset) if offset > 0 else int(row + offset > 0)
    m_last_row = max(0, min(col, row + offset))
    n_row_all = max(0, min(row, row + offset))
    n_row_trapezoid = m_last_row - m_first_row + 1
    # Elements in the top trapezoid: rows of lengths m_first_row..m_last_row.
    trapezoid_size = (m_first_row + m_last_row) * n_row_trapezoid // 2
    diff_row = n_row_all - n_row_trapezoid
    rectangle_size = max(0, diff_row * col)
    return trapezoid_size, rectangle_size, m_first_row


def _get_triu_sizes(row, col, offset):
    if row == 0 or col == 0:
        return 0, 0, 0
    m_first_row = max(0, col - offset) if offset > 0 else col
    rectangle_size = max(0, min(row, -offset) * col)
    trapezoid_size_tril, rectangle_size_tril, _ = _get_tril_sizes(row, col, offset - 1)
    triu_size = row * col - (trapezoid_size_tril + rectangle_size_tril)
    trapezoid_size = triu_size - rectangle_size
    return trapezoid_size, rectangle_size, m_first_row


@register_decomposition(ops.tril_indices.default)
def tril_indices(row, col, offset=0, *, dtype=tp.int64, device=None, pin_memory=False):
    _trilu_checks("tril_indices", row, col, dtype)
    trapezoid_size, rectangle_size, m_first_row = _get_tril_sizes(row, col, offset)
    row_offset = max(0, -offset)

    # Linear positions of the top trapezoid invert back to (row, column):
    # the cumulative count up to row r is r*(2*m_first_row + r - 1)/2, so the
    # row is the floor of the positive root of r^2 + (2*m_first_row - 1)*r - 2x = 0.
    xs1 = ops.arange.start_step(0, trapezoid_size, 1, dtype=tp.float64, device=device)
    b = m_first_row - 0.5
    row_inds1 = tp.floor(-b + tp.sqrt(b * b + 2 * xs1))
    col_inds1 = tp.floor(xs1 - (2 * m_first_row - 1 + row_inds1) * row_inds1 * 0.5)
    row_inds1 = prims.convert_element_type(row_inds1 + row_offset, dtype)
    col_inds1 = prims.convert_element_type(col_inds1, dtype)

    # The bottom rectangle is a full row of columns per position, starting
    # one row below where the trapezoid's widest row ended.
    xs2 = ops.arange.start_step(0, rectangle_size, 1, dtype=dtype, device=device)
    row_inds2 = xs2 // col + (col - m_first_row + 1 + row_offset)
    col_inds2 = xs2 % col

    return ops.stack.default(
        (ops.cat.default((row_inds1, row_inds2)), ops.cat.default((col_inds1, col_inds2)))
    )


@register_decomposition(ops.triu_indices.default)
def triu_indices(row, col, offset=0, *, dtype=tp.int64, device=None, pin_memory=False):
    _trilu_checks("triu_indices", row, col, dtype)
    trapezoid_size, rectangle_size, m_first_row = _get_triu_sizes(row, col, offset)
    col_offset = max(0, offset)

    # The top rectangle is a full row of columns per position.
    xs2 = ops.arange.start_step(0, rectangle_size, 1, dtype=dtype, device=device)
    row_inds2 = xs2 // col
    col_inds2 = xs2 % col

    # Bottom trapezoid: rows of lengths m_first_row downward; the row solves
    # r^2 - (2*m_first_row - 1)*r + 2x = 0 counted from the rectangle's end.
    xs1 = ops.arange.start_step(0, trapezoid_size, 1, dtype=tp.float64, device=device)
    b = -0.5 - m_first_row
    row_inds1 = tp.floor(-b - tp.sqrt(b * b - 2 * xs1))
    col_inds1 = tp.floor(xs1 - ((2 * m_first_row - 1 - row_inds1) * row_inds1) * 0.5)
    row_inds1 = prims.convert_element_type(row_inds1, dtype)
    col_inds1 = prims.convert_element_type(col_inds1, dtype)

    if col:
        row_inds1 = row_inds1 + (rectangle_size // col)
    col_inds1 = col_inds1 + col_offset

    return ops.stack.default(
        (ops.cat.default((row_inds2, row_inds1)), ops.cat.default((col_inds2, col_inds1)))
    )


@register_decomposition(ops.empty_strided.default)
def empty_strided(size, stride, *, dtype=None, device=None, pin_memory=False):
    return prims.empty_strided(
        list(size),
        list(stride),
        dtype=dtype if dtype is not None else tp.get_default_dtype(),
        device=device if device is not None else tp.get_default_device(),
        requires_grad=False,
    )


@register_decomposition(ops.pad_sequence.default)
def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side="right"):
    if len(sequences) == 0:
        raise RuntimeError("received an empty list of sequences")
    if padding_side not in ("left", "right"):
        raise RuntimeError(
            f"Expected padding_side to be one of left or right, but got {padding_side}."
        )
    sequences_size = len(sequences)
    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])
    max_len = reduce(tp.sym_max, (x.shape[0] for x in sequences))
    out_dims = (
        (sequences_size, max_len) if batch_first else (max_len, sequences_size)
    ) + trailing_dims
    out = ops.new_full.default(sequences[0], list(out_dims), padding_value)
    dim_paddings = (0, 0) * len(trailing_dims)
    for i in range(sequences_size):
        currseq = prims.convert_element_type(sequences[i], out.dtype)
        pad_amount = max_len - currseq.shape[0]
        if padding_side == "right":
            row = ops.constant_pad_nd.default(
                currseq, dim_paddings + (0, pad_amount), padding_value
            )
        else:
            row = ops.constant_pad_nd.default(
                currseq, dim_paddings + (pad_amount, 0), padding_value
            )
        out = ops.select_scatter.default(out, row, 0 if batch_first else 1, i)
    return out


@register_decomposition(ops.permute_copy.default)
def permute_copy(self, dims):
    return prims.clone(permute(self, dims), memory_format=tp.contiguous_format)


@register_decomposition(ops.narrow_copy.default)
def narrow_copy(self, dim, start, length):
    return prims.clone(
        ops.narrow.default(self, dim, start, length),
        memory_format=tp.contiguous_format,
    )


@register_decomposition(ops.view_copy.default)
def view_copy(self, size):
    return prims.clone(view(self, list(size)), memory_format=tp.contiguous_format)


# ---------------------------------------------------------------------------
# Activations, losses, and the backward passes built on scatter
#
# Reduction labels follow the eager convention: 0 keeps one value per
# position, 1 divides by the total weight, 2 adds everything up.
# ---------------------------------------------------------------------------

_RED_NONE, _RED_MEAN, _RED_SUM = 0, 1, 2


@register_decomposition(ops.sigmoid.default)
def sigmoid(self):
    compute = _computation_dtype(self.dtype)
    x = prims.convert_element_type(self, compute)
    return prims.convert_element_type(1 / (1 + tp.exp(-x)), self.dtype)


@register_decomposition(ops.tanh.default)
def tanh(self):
    return prims.tanh(self)


@register_decomposition(ops.atanh.default)
def atanh(self):
    return prims.atanh(self)


@register_decomposition(ops.relu.default)
def relu(self):
    return ops.clamp_min.default(self, 0)


@register_decomposition(ops.relu6.default)
def relu6(self):
    return ops.clamp.default(self, 0, 6)


_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


@register_decomposition(ops.selu.default)
def selu(self):
    return ops.elu.default(self, _SELU_ALPHA, _SELU_SCALE, 1.0)


# The fill value and the normalization factor keep the kept entries' second
# moment at one and send the dropped entries to a fixed negative point.
_ALPHA_DROPOUT_SATURATION = 1.7580993408473766


@register_decomposition(ops.alpha_dropout.default)
def alpha_dropout(input, p=0.5, train=True):
    if p < 0 or p > 1:
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")
    if p == 0 or not train or input.numel() == 0:
        return input
    if p == 1:
        return input * 0
    alpha = _ALPHA_DROPOUT_SATURATION
    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    compute = _computation_dtype(input.dtype)
    x = prims.convert_element_type(input, compute)
    keep = prims.convert_element_type(ops.rand_like.default(input) > p, compute)
    saturation = alpha * a
    b = (keep - 1) * saturation + saturation * p
    return prims.convert_element_type(x * (keep * a) + b, input.dtype)


@register_decomposition(ops.prelu.default)
def prelu(self, weight):
    if weight.numel() == 1:
        if self.ndim == 0 and weight.ndim == 1:
            weight = weight.reshape(())
        return ops.where.self(self > 0, self, self * weight)
    if weight.ndim != 1:
        raise RuntimeError("prelu: weight must be a scalar or a vector")
    if self.ndim < 2:
        raise RuntimeError("prelu: per-channel weights need at least 2 dimensions")
    if weight.shape[0] != self.shape[1]:
        raise RuntimeError(
            f"prelu: weight of shape {list(weight.shape)} cannot be broadcast "
            f"to input of shape {list(self.shape)}"
        )
    shape = [1] * self.ndim
    shape[1] = weight.shape[0]
    return ops.where.self(self > 0, self, self * weight.reshape(shape))


@register_decomposition(ops.margin_ranking_loss.default)
def margin_ranking_loss(input1, input2, target, margin=0):
    return ops.clamp.default(margin - target * (input1 - input2), 0).mean()


@register_decomposition(ops.hinge_embedding_loss.default)
def hinge_embedding_loss(input, target, margin=1.0):
    zeros = ops.zeros_like.default(input)
    margin_part = ops.where.self(target != 1, ops.clamp_min.default(margin - input, 0), zeros)
    self_part = ops.where.self(target != -1, input, zeros)
    return (margin_part + self_part).mean()


@register_decomposition(ops.nll_loss.default)
def nll_loss(self, target, weight=None, reduction=_RED_MEAN, ignore_index=-100):
    if not (self.ndim > 0 and self.ndim <= 2):
        raise RuntimeError(f"input tensor should be 1D or 2D, got {self.ndim}D")
    if target.ndim > 1:
        raise RuntimeError(
            f"0D or 1D target tensor expected, multi-target not supported, got {target.ndim}D"
        )
    if self.ndim != 1 or target.ndim != 0:
        if self.shape[0] != target.shape[0]:
            raise RuntimeError(
                f"size mismatch (got input: {list(self.shape)}, target: {list(target.shape)})"
            )
    n_classes = self.shape[-1]
    if weight is not None and not (weight.ndim == 1 and weight.numel() == n_classes):
        raise RuntimeError(
            f"weight tensor should be defined either for all {n_classes} classes or no classes "
            f"but got weight tensor of shape: {list(weight.shape)}"
        )
    return _nll_forward(self, target, weight, reduction, ignore_index)


def _softmax_amax(x, dim):
    # amax has no identity, so an empty axis cannot be reduced.  One -inf
    # rides along: it never wins for a non-empty axis and gives the empty
    # axis the identity of max.
    d = canonicalize_dim(x.ndim, dim)
    if x.shape[d] == 0:
        pad_shape = list(x.shape)
        pad_shape[d] = 1
        x = ops.cat.default((x, ops.new_full.default(x, pad_shape, float("-inf"))), dim)
    return ops.amax.default(x, [d], True)


@register_decomposition(ops._softmax.default)
def _softmax(self, dim, half_to_float):
    if half_to_float and self.dtype != tp.float16:
        raise RuntimeError(f"half_to_float is True but self.dtype is {self.dtype}, expected float16")
    compute = _computation_dtype(self.dtype)
    x = prims.convert_element_type(self, compute)
    x_max = _softmax_amax(x, dim)
    unnormalized = tp.exp(x - x_max)
    result = unnormalized / ops.sum.dim_IntList(unnormalized, [dim], True)
    if not half_to_float:
        result = prims.convert_element_type(result, self.dtype)
    return result


@register_decomposition(ops._log_softmax.default)
def _log_softmax(self, dim, half_to_float):
    if half_to_float and self.dtype != tp.float16:
        raise RuntimeError(f"half_to_float is True but self.dtype is {self.dtype}, expected float16")
    compute = _computation_dtype(self.dtype)
    x = prims.convert_element_type(self, compute)
    x_max = _softmax_amax(x, dim)
    shifted = x - x_max
    shifted_logsumexp = tp.log(ops.sum.dim_IntList(tp.exp(shifted), [dim], True))
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = prims.convert_element_type(result, self.dtype)
    return result


@register_decomposition(ops._fused_dropout.default)
def _fused_dropout(self, p, generator=None):
    if generator is not None:
        raise AssertionError(f"generator must be None for _fused_dropout, got {generator}")
    compute = _computation_dtype(self.dtype)
    x = prims.convert_element_type(self, compute)
    mask = prims.convert_element_type(ops.rand_like.default(x) < p, tp.uint8)
    scale = 1.0 / p
    res = prims.convert_element_type(mask, compute) * x * scale
    return (prims.convert_element_type(res, self.dtype), mask)


@register_decomposition(ops.embedding.default)
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if weight.ndim != 2:
        raise RuntimeError(f"'weight' must be 2-D, got {weight.ndim}-D")
    if indices.ndim == 0:
        # index_select requires a vector, so a scalar index goes in as a
        # one-element vector and the lookup result loses the row axis.
        out = ops.index_select.default(weight, 0, ops.reshape.default(indices, [-1]))
        return squeeze(out, 0)
    if indices.ndim == 1:
        return ops.index_select.default(weight, 0, indices)
    flat = ops.index_select.default(weight, 0, ops.reshape.default(indices, [-1]))
    return view(flat, list(indices.shape) + list(weight.shape[1:]))


@register_decomposition(ops.batch_norm_backward.default)
def batch_norm_backward(grad_output, input, weight=None, running_mean=None, running_var=None, training=True, eps=1e-5):
    if input.ndim < 2:
        raise RuntimeError(f"rank of the input must be at least 2, got {input.ndim}")
    input_dtype = input.dtype
    weight_dtype = weight.dtype if weight is not None else input_dtype
    compute = _computation_dtype(input_dtype)
    grad_out = prims.convert_element_type(grad_output, compute)
    x = prims.convert_element_type(input, compute)

    axis = 1
    num_features = reduce(operator.mul, input.shape, 1) / input.shape[axis]
    reduction_axes = [i for i in range(input.ndim) if i != axis]
    broadcast_mask = [1] * input.ndim
    broadcast_mask[axis] = input.shape[axis]

    def per_channel(t):
        return ops.reshape.default(t, broadcast_mask)

    if training:
        mean = ops.mean.dim(x, reduction_axes, True)
        var = ops.mean.dim(x * x, reduction_axes, True) - mean * mean
        invstd = ops.rsqrt.default(var + eps)
    else:
        mean = per_channel(
            running_mean if running_mean is not None else ops.zeros(input.shape[axis])
        ).to(compute)
        var = running_var if running_var is not None else ops.zeros(input.shape[axis])
        invstd = ops.rsqrt.default(per_channel(var).to(compute) + eps)

    norm = 1.0 / num_features
    grad_output_sum = ops.sum.dim_IntList(grad_out, reduction_axes, True)
    dot_p = ops.sum.dim_IntList(grad_out * (x - mean), reduction_axes, True)

    grad_mean = grad_output_sum * norm
    proj_scale = dot_p * norm * invstd * invstd
    if weight is None:
        grad_scale = invstd * 1.0
    else:
        grad_scale = invstd * per_channel(prims.convert_element_type(weight, compute))

    if training:
        proj = (x - mean) * proj_scale
        grad_input = ((grad_out - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out * grad_scale

    grad_weight = dot_p * invstd if weight is not None else None
    grad_bias = grad_output_sum if weight is not None else None

    grad_input = prims.convert_element_type(grad_input, grad_output.dtype)
    if grad_weight is not None:
        grad_weight = prims.convert_element_type(squeeze(grad_weight, reduction_axes), weight_dtype)
    if grad_bias is not None:
        grad_bias = prims.convert_element_type(squeeze(grad_bias, reduction_axes), weight_dtype)
    return grad_input, grad_weight, grad_bias


_INT_DTYPE_BITS = {
    tp.int8: 8, tp.uint8: 8, tp.int16: 16, tp.int32: 32, tp.int64: 64,
}

_INT_DTYPE_MIN = {
    tp.int8: -(1 << 7), tp.uint8: 0, tp.int16: -(1 << 15),
    tp.int32: -(1 << 31), tp.int64: -(1 << 63),
}


def _pooling_output_shape(input_size, kernel_size, pad, stride, dilation, ceil_mode):
    if stride == 0:
        raise RuntimeError("stride should not be zero")
    if pad < 0:
        raise RuntimeError(f"pad must be non-negative, but got pad: {pad}")
    if pad > ((kernel_size - 1) * dilation + 1) // 2:
        raise RuntimeError(
            f"pad should be at most half of effective kernel size, but got pad={pad}, "
            f"kernel_size={kernel_size} and dilation={dilation}"
        )
    if not ceil_mode:
        return (input_size + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
    output_size = (
        input_size + 2 * pad - dilation * (kernel_size - 1) - 1 + stride - 1
    ) // stride + 1
    if (output_size - 1) * stride >= input_size + pad:
        output_size -= 1
    return output_size


def _max_pool_nd_with_indices(self, kernel_size, stride, padding, dilation,
                              ceil_mode, n_dim):
    def expand(value, default=None):
        if not isinstance(value, (list, tuple)):
            return [value] * n_dim
        if not value:
            if default is None:
                raise RuntimeError("an empty size list requires a default")
            return list(default)
        return value * n_dim if len(value) == 1 else list(value)

    ks = expand(kernel_size)
    st = expand(stride, ks)
    pa = expand(padding, [0] * n_dim)
    di = expand(dilation, [1] * n_dim)

    if self.dim() not in (n_dim + 1, n_dim + 2):
        raise RuntimeError(f"Expected {n_dim + 1}D or {n_dim + 2}D input, got {self.dim()}D")

    is_batched = self.dim() == n_dim + 2
    if not is_batched:
        self = unsqueeze(self, 0)

    input_sizes = [int(s) for s in self.shape[-n_dim:]]
    output_sizes = [
        _pooling_output_shape(input_sizes[d], ks[d], pa[d], st[d], di[d], ceil_mode)
        for d in range(n_dim)
    ]

    # Padding is placed with a sentinel no real element can reach so it never
    # wins the max.  For integers the type minimum is a reachable value and
    # would tie in the argmax, so the input is widened first and the sentinel
    # sits one below the minimum of the original type.
    dtype = self.dtype
    promoted = False
    if dtype == tp.bool:
        fill_value = 0.0
    elif dtype.is_floating_point:
        fill_value = float("-inf")
    else:
        bits = _INT_DTYPE_BITS[dtype]
        if bits < 32:
            self = self.to(dtype=tp.int32)
        elif bits < 64:
            self = self.to(dtype=tp.int64)
        fill_value = float(_INT_DTYPE_MIN[dtype] - 1) if bits < 64 else float(_INT_DTYPE_MIN[dtype])
        promoted = True

    # Ceil mode lets the last window overhang the (padded) extent, so the
    # right side of each dim may need more than the symmetric padding.
    pad_args = []
    for d in reversed(range(n_dim)):
        needed = (output_sizes[d] - 1) * st[d] + (ks[d] - 1) * di[d] + 1
        right = max(0, needed - (input_sizes[d] + 2 * pa[d])) + pa[d]
        pad_args.extend([pa[d], right])
    x = ops.constant_pad_nd.default(self, pad_args, fill_value)

    # One index plane per spatial dim: entry [out_pos, ker_pos] points at the
    # input position that window element reads.  The planes broadcast into the
    # Cartesian product of all dims, so the gather below pulls every pooling
    # window at once.
    idx_tensors = []
    for d in range(n_dim):
        out_pos = tp.arange(output_sizes[d], dtype=tp.int64, device=self.device).unsqueeze(1)
        ker_pos = tp.arange(ks[d], dtype=tp.int64, device=self.device).unsqueeze(0)
        idx = out_pos * st[d] + ker_pos * di[d]
        shape = [1] * (2 * n_dim)
        shape[2 * d] = output_sizes[d]
        shape[2 * d + 1] = ks[d]
        idx_tensors.append(ops.reshape.default(idx, shape))

    # (N, C, out_0, k_0, out_1, k_1, ..., out_{n-1}, k_{n-1})
    windows = ops.index.Tensor(x, [None, None] + idx_tensors)

    perm = [0, 1]
    perm += [2 + 2 * d for d in range(n_dim)]
    perm += [3 + 2 * d for d in range(n_dim)]
    windows = ops.permute.default(windows, perm)
    out_shape = [int(s) for s in windows.shape[:2 + n_dim]]
    windows = ops.reshape.default(windows, out_shape + [-1])
    values, local_argmax = ops.max.dim(windows, -1)

    # Expand the flat position inside the window into one offset per dim.
    kernel_pos = []
    remaining = local_argmax
    for d in range(n_dim):
        divisor = 1
        for dd in range(d + 1, n_dim):
            divisor *= ks[dd]
        kernel_pos.append(remaining // divisor)
        remaining = remaining % divisor

    # Fold per-dim input coordinates (window origin plus kernel offset, minus
    # padding) into the flat index the backward pass scatters through.
    orig_coords = []
    for d in range(n_dim):
        shape = [1] * (2 + n_dim)
        shape[2 + d] = output_sizes[d]
        out_pos = tp.arange(output_sizes[d], dtype=tp.int64, device=self.device).reshape(shape)
        orig_coords.append(out_pos * st[d] + kernel_pos[d] * di[d] - pa[d])

    flat_indices = orig_coords[0]
    for d in range(1, n_dim):
        flat_indices = flat_indices * input_sizes[d] + orig_coords[d]

    if promoted:
        values = values.to(dtype)
    if not is_batched:
        values = values.squeeze(0)
        flat_indices = flat_indices.squeeze(0)
    return values, flat_indices


@register_decomposition(ops.max_pool2d_with_indices.default)
def max_pool2d_with_indices(self, kernel_size, stride=[], padding=[],
                            dilation=[], ceil_mode=False):
    return _max_pool_nd_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode, 2
    )


@register_decomposition(ops.max_pool3d_with_indices.default)
def max_pool3d_with_indices(self, kernel_size, stride=[], padding=[],
                            dilation=[], ceil_mode=False):
    return _max_pool_nd_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode, 3
    )


@register_decomposition(ops.adaptive_max_pool2d.default)
def adaptive_max_pool2d(input, output_size):
    if input.dim() not in (3, 4):
        raise RuntimeError(
            f"adaptive_max_pool2d(): Expected 3D or 4D tensor, but got {input.dim()}D"
        )
    for i in range(1, input.dim()):
        if int(input.shape[i]) <= 0:
            raise RuntimeError(
                "adaptive_max_pool2d(): Expected input to have non-zero size for "
                f"non-batch dimensions, but input has sizes {tuple(input.shape)} "
                f"with dimension {i} being empty"
            )

    h_in = int(input.shape[-2])
    w_in = int(input.shape[-1])
    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        output_shape = [int(s) for s in input.shape[:-2]] + [h_out, w_out]
        return input.new_empty(output_shape)

    # Global pooling: one max over the whole spatial extent.
    if h_out == 1 and w_out == 1:
        return ops.amax.default(input, [-2, -1], True)

    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return ops.max_pool2d.default(input, kernel_size)

    return NotImplemented


@register_decomposition(ops.adaptive_max_pool3d.default)
def adaptive_max_pool3d(input, output_size):
    if input.dim() not in (4, 5):
        raise RuntimeError(
            f"adaptive_max_pool3d(): Expected 4D or 5D tensor, but got {input.dim()}D"
        )
    for i in range(1, input.dim()):
        if int(input.shape[i]) <= 0:
            raise RuntimeError(
                "adaptive_max_pool3d(): Expected input to have non-zero size for "
                f"non-batch dimensions, but input has sizes {tuple(input.shape)} "
                f"with dimension {i} being empty"
            )

    d_in = int(input.shape[-3])
    h_in = int(input.shape[-2])
    w_in = int(input.shape[-1])
    d_out, h_out, w_out = output_size

    if d_out == 0 or h_out == 0 or w_out == 0:
        output_shape = [int(s) for s in input.shape[:-3]] + [d_out, h_out, w_out]
        return input.new_empty(output_shape)

    if d_out == 1 and h_out == 1 and w_out == 1:
        return ops.amax.default(input, [-3, -2, -1], True)

    if d_in % d_out == 0 and h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [d_in // d_out, h_in // h_out, w_in // w_out]
        return ops.max_pool3d.default(input, kernel_size)

    return NotImplemented


@register_decomposition(ops.max_pool2d_with_indices_backward.default)
def max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride=[], padding=[], dilation=[], ceil_mode=False, indices=None):
    if indices is None:
        indices = ops.max_pool2d_with_indices.default(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )[1]
    is_batched = self.ndim == 4
    if not is_batched:
        self = unsqueeze(self, 0)
        grad_output = unsqueeze(grad_output, 0)
        indices = unsqueeze(indices, 0)

    batch_size, channels = self.shape[0], self.shape[1]
    in_height, in_width = self.shape[-2], self.shape[-1]
    out_height, out_width = grad_output.shape[-2], grad_output.shape[-1]

    # Overlapping windows pile several gradients onto one input position,
    # so reduced precision accumulates in float32.
    accum = tp.float32 if grad_output.dtype in (tp.float16, tp.bfloat16) else grad_output.dtype
    grad_input_flat = ops.zeros(
        [batch_size * channels, in_height * in_width], dtype=accum, device=grad_output.device
    )
    grad_output_flat = ops.reshape.default(grad_output, [batch_size * channels, out_height * out_width])
    indices_flat = ops.reshape.default(indices, [batch_size * channels, out_height * out_width])
    if accum != grad_output.dtype:
        grad_output_flat = prims.convert_element_type(grad_output_flat, accum)
    grad_input_flat = ops.scatter_add.default(grad_input_flat, 1, indices_flat, grad_output_flat)
    grad_input = ops.reshape.default(grad_input_flat, [batch_size, channels, in_height, in_width])
    if accum != grad_output.dtype:
        grad_input = prims.convert_element_type(grad_input, grad_output.dtype)
    if not is_batched:
        grad_input = squeeze(grad_input, 0)
    return grad_input


def _gather_rnn_params(params, has_biases, has_projections=False):
    # The flat parameter list holds one group per direction per layer:
    # [w_ih, w_hh, b_ih, b_hh] with biases, [w_ih, w_hh] without; a
    # projection layer appends w_hr to each group.
    if has_biases and has_projections:
        group_size = 5
    elif has_biases:
        group_size = 4
    elif has_projections:
        group_size = 3
    else:
        group_size = 2
    if len(params) % group_size != 0:
        raise RuntimeError(
            f"len(params)={len(params)} is not divisible by group_size={group_size}"
        )
    return [tuple(params[i : i + group_size]) for i in range(0, len(params), group_size)]


def _one_layer_rnn(inp, hidden, params, has_biases, hidden_fn, reverse=False):
    ih_weight, hh_weight = params[0], params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    precomputed_input = ops.linear.default(inp, ih_weight, ih_bias)
    if reverse:
        precomputed_input = prims.rev(precomputed_input, [0])
    cur_hidden = unsqueeze(hidden, 0)
    step_output = []
    for i in ops.unbind.default(precomputed_input, 0):
        cur_hidden = hidden_fn(i, cur_hidden, hh_weight, hh_bias)
        step_output.append(cur_hidden)
    if reverse:
        step_output.reverse()
    out = ops.cat.default(step_output, 0)
    return out, squeeze(cur_hidden, 0)


def _rnn_helper(input, hidden, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, layer_fn):
    if batch_first:
        input = transpose(input, 0, 1)
    final_hiddens = []
    for i in range(num_layers):
        if bidirectional:
            cur_params, cur_hidden = params[2 * i], hidden[2 * i]
            bidir_params, bidir_hidden = params[2 * i + 1], hidden[2 * i + 1]
        else:
            cur_params, cur_hidden = params[i], hidden[i]
            bidir_params, bidir_hidden = None, None

        fwd_inp, fwd_hidden = layer_fn(input, cur_hidden, cur_params, has_biases)
        final_hiddens.append(fwd_hidden)

        if bidirectional:
            bwd_inp, bwd_hidden = layer_fn(
                input, bidir_hidden, bidir_params, has_biases, reverse=True
            )
            final_hiddens.append(bwd_hidden)
            input = ops.cat.default([fwd_inp, bwd_inp], fwd_inp.ndim - 1)
        else:
            input = fwd_inp

        if dropout != 0 and train and i < num_layers - 1:
            input = ops.dropout.default(input, dropout, True)

    if batch_first:
        input = transpose(input, 0, 1)
    return input, final_hiddens


def _rnn_cell(hidden_fn):
    def cell(i, cur_hidden, hh_weight, hh_bias):
        return hidden_fn(ops.linear.default(cur_hidden, hh_weight, hh_bias) + i)

    return cell


def _gru_cell(inp, cur_hidden, hh_weight, hh_bias):
    # inp holds the three precomputed input gates (batch, 3*hidden); the
    # hidden gates come from the current state and reset scales them.
    chunked_igates = tp.chunk(inp, 3, dim=1)
    chunked_hgates = tp.chunk(ops.linear.default(cur_hidden, hh_weight, hh_bias), 3, dim=2)
    reset_gate = tp.sigmoid(chunked_hgates[0] + chunked_igates[0])
    input_gate = tp.sigmoid(chunked_hgates[1] + chunked_igates[1])
    new_gate = tp.tanh(chunked_igates[2] + chunked_hgates[2] * reset_gate)
    return (cur_hidden - new_gate) * input_gate + new_gate


def _lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight):
    gates = ops.linear.default(hx, hh_weight, hh_bias) + inp
    in_gate, forget_gate, cell_gate, out_gate = tp.chunk(gates, 4, dim=2)
    in_gate = tp.sigmoid(in_gate)
    forget_gate = tp.sigmoid(forget_gate)
    cell_gate = tp.tanh(cell_gate)
    out_gate = tp.sigmoid(out_gate)
    cy = forget_gate * cx + in_gate * cell_gate
    hy = out_gate * tp.tanh(cy)
    if hr_weight is not None:
        hy = ops.linear.default(hy, hr_weight, None)
    return hy, cy


def _one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    ih_weight, hh_weight = params[0], params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    # A projection layer carries w_hr as the last group entry: position 4
    # with biases, position 2 without.
    hr_weight = params[4] if len(params) == 5 else params[2] if len(params) == 3 else None

    hx = unsqueeze(hidden[0], 0)
    cx = unsqueeze(hidden[1], 0)
    precomputed_input = ops.linear.default(inp, ih_weight, ih_bias)
    if reverse:
        precomputed_input = prims.rev(precomputed_input, [0])
    step_output = []
    for i in ops.unbind.default(precomputed_input, 0):
        hx, cx = _lstm_cell(i, hx, cx, hh_weight, hh_bias, hr_weight)
        step_output.append(hx)
    if reverse:
        step_output.reverse()
    out = ops.cat.default(step_output, 0)
    return out, (squeeze(hx, 0), squeeze(cx, 0))


@register_decomposition(ops.rnn_tanh.input)
def rnn_tanh_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    hidden = list(ops.unbind.default(hx, 0))
    params = _gather_rnn_params(params, has_biases)
    out, final_hiddens = _rnn_helper(
        input, hidden, params, has_biases, num_layers, dropout, train,
        bidirectional, batch_first, partial(_one_layer_rnn, hidden_fn=_rnn_cell(prims.tanh)),
    )
    return out, ops.stack.default(final_hiddens, 0)


@register_decomposition(ops.rnn_relu.input)
def rnn_relu_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    hidden = list(ops.unbind.default(hx, 0))
    params = _gather_rnn_params(params, has_biases)
    out, final_hiddens = _rnn_helper(
        input, hidden, params, has_biases, num_layers, dropout, train,
        bidirectional, batch_first, partial(_one_layer_rnn, hidden_fn=_rnn_cell(ops.relu.default)),
    )
    return out, ops.stack.default(final_hiddens, 0)


@register_decomposition(ops.lstm.input)
def lstm_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    if len(hx) != 2:
        raise RuntimeError(f"lstm expects two hidden states, got {len(hx)}")
    has_projections = hx[0].size(2) != hx[1].size(2)
    params = _gather_rnn_params(params, has_biases, has_projections)
    hidden = list(zip(hx[0], hx[1]))
    out, final_hiddens = _rnn_helper(
        input, hidden, params, has_biases, num_layers, dropout, train,
        bidirectional, batch_first, _one_layer_lstm,
    )
    final_h = list(zip(*final_hiddens))
    return out, ops.stack.default(final_h[0], 0), ops.stack.default(final_h[1], 0)


@register_decomposition(ops.gru.input)
def gru_input(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    params = _gather_rnn_params(params, has_biases)
    hidden = list(ops.unbind.default(hx, 0))
    out, final_hiddens = _rnn_helper(
        input, hidden, params, has_biases, num_layers, dropout, train,
        bidirectional, batch_first, partial(_one_layer_rnn, hidden_fn=_gru_cell),
    )
    return out, ops.stack.default(final_hiddens, 0)


# The special-function spellings share one computation layer: the base names
# and their "special_" aliases route to the same prims, which promote integer
# inputs to a floating type before evaluating.


@register_decomposition(ops.bessel_j0.default)
def bessel_j0(self):
    return prims.bessel_j0(self)


@register_decomposition(ops.bessel_j1.default)
def bessel_j1(self):
    return prims.bessel_j1(self)


@register_decomposition(ops.spherical_bessel_j0.default)
def spherical_bessel_j0(self):
    return prims.spherical_bessel_j0(self)


@register_decomposition(ops.digamma.default)
def digamma(self):
    return prims.digamma(self)


@register_decomposition(ops.erf.default)
def erf(self):
    return prims.erf(self)


@register_decomposition(ops.erfc.default)
def erfc(self):
    return prims.erfc(self)


@register_decomposition(ops.erfinv.default)
def erfinv(self):
    return prims.erf_inv(self)


@register_decomposition(ops.i0.default)
def i0(self):
    return prims.bessel_i0(self)


@register_decomposition(ops.i0e.default)
def i0e(self):
    return prims.bessel_i0e(self)


@register_decomposition(ops.i1.default)
def i1(self):
    return prims.bessel_i1(self)


@register_decomposition(ops.i1e.default)
def i1e(self):
    return prims.bessel_i1e(self)


@register_decomposition(ops.lgamma.default)
def lgamma(self):
    return prims.lgamma(self)


@register_decomposition(ops.igamma.default)
def igamma(self, other):
    return prims.igamma(self, other)


@register_decomposition(ops.igammac.default)
def igammac(self, other):
    return prims.igammac(self, other)


@register_decomposition(ops.zeta.default)
def zeta(self, other):
    return prims.zeta(self, other)


@register_decomposition(ops.special_bessel_j0.default)
def special_bessel_j0(self):
    return prims.bessel_j0(self)


@register_decomposition(ops.special_bessel_j1.default)
def special_bessel_j1(self):
    return prims.bessel_j1(self)


@register_decomposition(ops.special_spherical_bessel_j0.default)
def special_spherical_bessel_j0(self):
    return prims.spherical_bessel_j0(self)


@register_decomposition(ops.special_erfcx.default)
def special_erfcx(self):
    return prims.erfcx(self)


@register_decomposition(ops.special_i0e.default)
def special_i0e(self):
    return prims.bessel_i0e(self)


@register_decomposition(ops.special_i1.default)
def special_i1(self):
    return prims.bessel_i1(self)


@register_decomposition(ops.special_i1e.default)
def special_i1e(self):
    return prims.bessel_i1e(self)


@register_decomposition(ops.special_ndtri.default)
def special_ndtri(self):
    return prims.ndtri(self)


@register_decomposition(ops.special_ndtr.default)
def special_ndtr(self):
    # M_SQRT1_2 is the value of 1 / sqrt(2).
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    scaled = self * M_SQRT1_2
    return (1 + ops.erf(scaled)) * 0.5


@register_decomposition(ops.special_zeta.default)
def special_zeta(self, other):
    return prims.zeta(self, other)


# The FFT transforms decompose onto three prims (real-to-complex,
# complex-to-complex, complex-to-real) wrapped in type promotion, input
# resizing, and normalization.  A transform axis of size -1 in `n`/`s`
# means "keep the current length"; an omitted `n`/`s` resizes nothing.

_FFT_NORM_VALUES = {None, "forward", "backward", "ortho"}


def _fft_apply_norm(x, norm, signal_numel, forward):
    if norm not in _FFT_NORM_VALUES:
        raise RuntimeError(f"Invalid normalization mode: {norm}")
    if norm == "ortho":
        return x * (1 / math.sqrt(signal_numel))
    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return x * (1 / signal_numel) if normalize else x


def _fft_promote_type(dtype, require_complex):
    if dtype.is_complex:
        return dtype
    if not dtype.is_floating_point:
        dtype = tp.get_default_dtype()
    if dtype not in (tp.float32, tp.float64):
        raise RuntimeError(
            f"fft transforms expect float32 or float64 inputs, but got {dtype}"
        )
    if require_complex:
        dtype = tp.complex128 if dtype == tp.float64 else tp.complex64
    return dtype


def _fft_maybe_promote(t, require_complex=False):
    new_type = _fft_promote_type(t.dtype, require_complex)
    return prims.convert_element_type(t, new_type) if new_type != t.dtype else t


def _fft_resize_input(x, dims, sizes):
    # Grow or trim x along dims so each size matches; growth zero-pads the
    # end of the axis, trim keeps the leading elements.
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue
        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_amount[len(pad_amount) - 2 * dims[i] - 1] = sizes[i] - x_sizes[dims[i]]
        if x_sizes[dims[i]] > sizes[i]:
            x = ops.narrow.default(x, dims[i], 0, sizes[i])
    return ops.constant_pad_nd.default(x, pad_amount, 0) if must_copy else x


def _fft_c2r(input, n, dim, norm, forward):
    input = _fft_maybe_promote(input, require_complex=True)
    dims = (canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    if last_dim_size < 1:
        raise RuntimeError(f"Invalid number of data points ({last_dim_size}) specified")
    if n is not None:
        input = _fft_resize_input(input, dims, (last_dim_size // 2 + 1,))
    if forward:
        input = prims.conj(input)
    output = prims.fft_c2r(input, dim=list(dims), last_dim_size=last_dim_size)
    # The c2r prim routes to the user-facing inverse kernel, which already
    # normalizes by the output length; undo it for a bare transform.
    output = output * last_dim_size
    return _fft_apply_norm(output, norm, last_dim_size, forward)


def _fft_r2c(input, n, dim, norm, forward, onesided):
    if input.dtype.is_complex:
        raise RuntimeError(
            f"fft expects a floating point input tensor, but got {input.dtype}"
        )
    input = _fft_maybe_promote(input)
    dims = (canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    if dim_size < 1:
        raise RuntimeError(f"Invalid number of data points ({dim_size}) specified")
    if n is not None:
        input = _fft_resize_input(input, dims, (n,))
    ret = prims.fft_r2c(input, dim=list(dims), onesided=onesided)
    ret = _fft_apply_norm(ret, norm, dim_size, forward)
    return ret if forward else prims.conj(ret)


def _fft_c2c(input, n, dim, norm, forward):
    if not input.dtype.is_complex:
        raise RuntimeError(
            f"fft expects a complex input tensor, but got {input.dtype}"
        )
    dims = (canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    if dim_size < 1:
        raise RuntimeError(f"Invalid number of data points ({dim_size}) specified")
    if n is not None:
        input = _fft_resize_input(input, dims, (n,))
    ret = prims.fft_c2c(input, dim=list(dims), forward=forward)
    if not forward:
        # The complex transform prim routes to the user-facing kernels, whose
        # inverse already carries the 1/N factor; undo it so the caller sees a
        # bare transform and applies the requested normalization exactly once.
        ret = ret * dim_size
    return _fft_apply_norm(ret, norm, dim_size, forward)


def _fft_canonicalize_shape_and_dims(input, shape, dim):
    input_dim = input.ndim
    input_sizes = input.shape
    ret_dims = None
    if dim is not None:
        if not isinstance(dim, (list, tuple)):
            dim = (dim,)
        ret_dims = canonicalize_dims(input_dim, list(dim), wrap_scalar=False)
        if len(set(ret_dims)) != len(ret_dims):
            raise RuntimeError("FFT dims must be unique")
    if shape is not None:
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)
        if dim is not None and len(dim) != len(shape):
            raise RuntimeError(
                "When given, dim and shape arguments must have the same length"
            )
        transform_ndim = len(shape)
        if transform_ndim > input_dim:
            raise RuntimeError(
                f"Got shape with {transform_ndim} values but input tensor "
                f"only has {input_dim} dimensions."
            )
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))
        ret_shape = tuple(
            s if s != -1 else input_sizes[d] for s, d in zip(shape, ret_dims)
        )
    elif dim is None:
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        ret_shape = tuple(input_sizes[d] for d in ret_dims)
    for n in ret_shape:
        if n <= 0:
            raise RuntimeError(f"Invalid number of data points ({n}) specified")
    return ret_shape, ret_dims


def _fft_prod(xs):
    prod = 1
    for x in xs:
        prod *= x
    return prod


def _fft_fftn_c2c(input, shape, dim, norm, forward):
    if not input.dtype.is_complex:
        raise RuntimeError(
            f"fftn expects a complex input tensor, but got {input.dtype}"
        )
    x = _fft_resize_input(input, dim, shape)
    # A multi-axis transform is the composition of one-axis transforms; the
    # transform prim handles a single axis per call.
    ret = x
    signal_numel = 1
    for d in dim:
        ret = prims.fft_c2c(ret, dim=[d], forward=forward)
        signal_numel *= ret.shape[d]
    if not forward:
        # Undo the inverse kernel's built-in 1/N before caller-side norm.
        ret = ret * signal_numel
    return _fft_apply_norm(ret, norm, signal_numel, forward)


@register_decomposition(ops.fft_fft.default)
def fft_fft(input, n=-1, dim=-1, norm="backward"):
    n = None if n == -1 else n
    if input.dtype.is_complex:
        return _fft_c2c(input, n, dim, norm, forward=True)
    return _fft_r2c(input, n, dim, norm, forward=True, onesided=False)


@register_decomposition(ops.fft_ifft.default)
def fft_ifft(input, n=-1, dim=-1, norm="backward"):
    n = None if n == -1 else n
    if input.dtype.is_complex:
        return _fft_c2c(input, n, dim, norm, forward=False)
    return _fft_r2c(input, n, dim, norm, forward=False, onesided=False)


@register_decomposition(ops.fft_rfft.default)
def fft_rfft(input, n=-1, dim=-1, norm="backward"):
    return _fft_r2c(input, None if n == -1 else n, dim, norm, forward=True, onesided=True)


@register_decomposition(ops.fft_irfft.default)
def fft_irfft(input, n=-1, dim=-1, norm="backward"):
    return _fft_c2r(input, None if n == -1 else n, dim, norm, forward=False)


@register_decomposition(ops.fft_hfft.default)
def fft_hfft(input, n=None, dim=-1, norm=None):
    return _fft_c2r(input, n, dim, norm, forward=True)


@register_decomposition(ops.fft_ihfft.default)
def fft_ihfft(input, n=None, dim=-1, norm=None):
    return _fft_r2c(input, n, dim, norm, forward=False, onesided=True)


@register_decomposition(ops.fft_fftn.default)
def fft_fftn(input, s=None, dim=None, norm=None):
    shape, dims = _fft_canonicalize_shape_and_dims(input, s, dim)
    x = _fft_maybe_promote(input, require_complex=True)
    return _fft_fftn_c2c(x, shape, dims, norm, forward=True)


@register_decomposition(ops.fft_ifftn.default)
def fft_ifftn(input, s=None, dim=None, norm=None):
    shape, dims = _fft_canonicalize_shape_and_dims(input, s, dim)
    x = _fft_maybe_promote(input, require_complex=True)
    return _fft_fftn_c2c(x, shape, dims, norm, forward=False)


@register_decomposition(ops.fft_rfftn.default)
def fft_rfftn(input, s=None, dim=None, norm=None):
    if input.dtype.is_complex:
        raise RuntimeError(
            f"rfftn expects a real-valued input tensor, but got {input.dtype}"
        )
    shape, dims = _fft_canonicalize_shape_and_dims(input, s, dim)
    input = _fft_maybe_promote(input)
    input = _fft_resize_input(input, dims, shape)
    # The real transform runs first while the data is still real; the other
    # axes are ordinary complex transforms of the truncated spectrum (the
    # axes are independent, so the order does not matter).
    out = prims.fft_r2c(input, dim=[dims[-1]], onesided=True)
    for d in dims[:-1]:
        out = prims.fft_c2c(out, dim=[d], forward=True)
    return _fft_apply_norm(out, norm, _fft_prod(shape), forward=True)


@register_decomposition(ops.fft_ihfftn.default)
def fft_ihfftn(input, s=None, dim=None, norm=None):
    if input.dtype.is_complex:
        raise RuntimeError(
            f"ihfftn expects a real-valued input tensor, but got {input.dtype}"
        )
    shape, dims = _fft_canonicalize_shape_and_dims(input, s, dim)
    if len(shape) == 0:
        raise RuntimeError("ihfftn must transform at least one axis")
    input = _fft_maybe_promote(input)
    input = _fft_resize_input(input, dims, shape)
    tmp = prims.fft_r2c(input, dim=[dims[-1]], onesided=True)
    if len(dims) == 1:
        tmp = _fft_apply_norm(tmp, norm, shape[0], forward=False)
        return prims.conj(tmp)
    tmp = prims.conj_physical(tmp)
    signal_numel = 1
    for d in dims[:-1]:
        tmp = prims.fft_c2c(tmp, dim=[d], forward=False)
        # Undo the inverse kernel's built-in 1/N per axis.
        signal_numel *= tmp.shape[d]
    tmp = tmp * signal_numel
    return _fft_apply_norm(tmp, norm, _fft_prod(shape), forward=False)


def _fft_canonicalize_c2r_shape_and_dims(input, s, dim):
    shape, dims = _fft_canonicalize_shape_and_dims(input, s, dim)
    if len(shape) == 0:
        raise RuntimeError("irfftn must transform at least one axis")
    if s is None or s[-1] == -1:
        last_dim_size = 2 * (input.shape[dims[-1]] - 1)
    else:
        last_dim_size = shape[-1]
    if last_dim_size < 1:
        raise RuntimeError(f"Invalid number of data points ({last_dim_size}) specified")
    shape_list = list(shape)
    shape_list[-1] = last_dim_size // 2 + 1
    return tuple(shape_list), dims, last_dim_size


@register_decomposition(ops.fft_irfftn.default)
def fft_irfftn(input, s=None, dim=None, norm=None):
    shape, dims, last_dim_size = _fft_canonicalize_c2r_shape_and_dims(input, s, dim)
    input = _fft_maybe_promote(input, require_complex=True)
    input = _fft_resize_input(input, dims, shape)
    tmp = input
    for d in dims[:-1]:
        tmp = prims.fft_c2c(tmp, dim=[d], forward=False)
        # Undo the inverse kernel's built-in 1/N per axis.
        tmp = tmp * tmp.shape[d]
    out = prims.fft_c2r(tmp, dim=[dims[-1]], last_dim_size=last_dim_size)
    out = out * last_dim_size
    return _fft_apply_norm(out, norm, _fft_prod(out.shape[d] for d in dims), forward=False)


@register_decomposition(ops.fft_hfftn.default)
def fft_hfftn(input, s=None, dim=None, norm=None):
    shape, dims, last_dim_size = _fft_canonicalize_c2r_shape_and_dims(input, s, dim)
    input = _fft_maybe_promote(input, require_complex=True)
    input = _fft_resize_input(input, dims, shape)
    tmp = input
    for d in dims[:-1]:
        tmp = prims.fft_c2c(tmp, dim=[d], forward=True)
    tmp = _fft_apply_norm(tmp, norm, _fft_prod(shape[:-1]), forward=True)
    tmp = prims.conj_physical(tmp)
    out = prims.fft_c2r(tmp, dim=[dims[-1]], last_dim_size=last_dim_size)
    out = out * last_dim_size
    return _fft_apply_norm(out, norm, last_dim_size, forward=True)


@register_decomposition(ops.fft_fft2.default)
def fft_fft2(input, s=None, dim=(-2, -1), norm="backward"):
    return fft_fftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_ifft2.default)
def fft_ifft2(input, s=None, dim=(-2, -1), norm="backward"):
    return fft_ifftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_rfft2.default)
def fft_rfft2(input, s=None, dim=(-2, -1), norm="backward"):
    return fft_rfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_irfft2.default)
def fft_irfft2(input, s=None, dim=(-2, -1), norm="backward"):
    return fft_irfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_hfft2.default)
def fft_hfft2(input, s=None, dim=(-2, -1), norm=None):
    return fft_hfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_ihfft2.default)
def fft_ihfft2(input, s=None, dim=(-2, -1), norm=None):
    return fft_ihfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(ops.fft_fftshift.default)
def fft_fftshift(input, dim=None):
    dims = list(range(input.ndim)) if dim is None else (
        [dim] if not isinstance(dim, (list, tuple)) else list(dim)
    )
    shift = [input.shape[d] // 2 for d in dims]
    return ops.roll.default(input, shift, dims)


@register_decomposition(ops.fft_ifftshift.default)
def fft_ifftshift(input, dim=None):
    dims = list(range(input.ndim)) if dim is None else (
        [dim] if not isinstance(dim, (list, tuple)) else list(dim)
    )
    shift = [(input.shape[d] + 1) // 2 for d in dims]
    return ops.roll.default(input, shift, dims)


# ---------------------------------------------------------------------------
# Predicates, scalar reads, factories and BLAS glue
# ---------------------------------------------------------------------------


@register_decomposition(ops.is_same_size.default)
def is_same_size(a, b):
    return a.shape == b.shape


@register_decomposition(ops.is_complex.default)
def is_complex(input):
    return input.dtype.is_complex


@register_decomposition(ops.isreal.default)
def isreal(a):
    if a.dtype.is_complex:
        return ops.eq.Scalar(ops.imag.default(a), 0)
    return ops.ones_like.default(a, dtype=tp.bool)


@register_decomposition(ops.item.default)
def item(a):
    if a.numel() != 1:
        raise RuntimeError(
            f"Can't convert a tensor with {a.numel()} elements to a number"
        )
    # Read the element without dispatching item again: the read is the very
    # operation this decomposition replaces.
    return a.reshape(1).tolist()[0]


def _where_promoted(pred, a, b):
    if pred.dtype != tp.bool:
        raise RuntimeError(f"expected predicate to be bool, got {pred.dtype}")
    target = tp.result_type(a, b)
    a = ops.to.dtype(a, target)
    b = ops.to.dtype(b, target)
    if target == tp.bool:
        return ops.logical_or.default(
            ops.logical_and.default(a, pred),
            ops.logical_and.default(b, ops.logical_not.default(pred)),
        )
    # Selection as two zero-filled halves; the fill is exact for every value
    # except that a negative zero on the taken side comes out positive.
    return ops.masked_fill.Scalar(
        a, ops.logical_not.default(pred), 0
    ) + ops.masked_fill.Scalar(b, pred, 0)


@register_decomposition(ops.where.self)
def where_self(condition, self, other):
    return _where_promoted(condition, self, other)


@register_decomposition(ops.where.ScalarSelf)
def where_scalar_self(condition, self, other):
    return _where_promoted(
        condition, ops.scalar_tensor.default(self, dtype=other.dtype), other
    )


@register_decomposition(ops.where.ScalarOther)
def where_scalar_other(condition, self, other):
    return _where_promoted(
        condition, self, ops.scalar_tensor.default(other, dtype=self.dtype)
    )


@register_decomposition(ops.where.Scalar)
def where_scalar(condition, self, other):
    def weak(value):
        if isinstance(value, bool):
            return tp.bool
        if isinstance(value, int):
            return tp.int64
        if isinstance(value, float):
            return tp.float64
        return tp.complex128

    return _where_promoted(
        condition,
        ops.scalar_tensor.default(self, dtype=weak(self)),
        ops.scalar_tensor.default(other, dtype=weak(other)),
    )


@register_decomposition(ops.where.default)
def where(condition):
    # Flat row indices per coordinate: the nonzero rows already come out
    # sorted, so selecting column d yields the indices along dimension d.
    flat = ops.nonzero.default(condition)
    return [ops.select.int(flat, 1, d) for d in range(condition.dim())]


def _broadcast_shapes(a, b):
    sa = tuple(int(s) for s in a)
    sb = tuple(int(s) for s in b)
    n = max(len(sa), len(sb))
    sa = (1,) * (n - len(sa)) + sa
    sb = (1,) * (n - len(sb)) + sb
    shape = []
    for x, y in zip(sa, sb):
        if x != y and x != 1 and y != 1:
            raise RuntimeError(
                f"The size of tensor a ({x}) must match the size of tensor b "
                f"({y}) at non-singleton dimension {len(shape)}"
            )
        shape.append(max(x, y))
    return shape


_COMPLEX_OF = {
    tp.float16: tp.complex32,
    tp.float32: tp.complex64,
    tp.float64: tp.complex128,
}


def _complex_from_parts(real, imag):
    if real.dtype not in _COMPLEX_OF or imag.dtype not in _COMPLEX_OF:
        raise RuntimeError(
            "Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        )
    if real.dtype != imag.dtype:
        raise RuntimeError(
            f"Expected object of scalar type {real.dtype} but got scalar type "
            f"{imag.dtype} for second argument"
        )
    shape = _broadcast_shapes(real.shape, imag.shape)
    interleaved = ops.stack.default(
        (ops.expand.default(real, shape), ops.expand.default(imag, shape)), -1
    )
    # The reinterpreting view folds the trailing pair into a size-one dim;
    # reshape it back to the broadcast shape.
    return ops.reshape.default(
        ops.view.dtype(interleaved, _COMPLEX_OF[real.dtype]), shape
    )


@register_decomposition(ops.complex.default)
def complex_op(real, imag):
    return _complex_from_parts(real, imag)


@register_decomposition(ops.polar.default)
def polar(abs, angle):
    return _complex_from_parts(abs * ops.cos.default(angle), abs * ops.sin.default(angle))


@register_decomposition(ops.conj_physical.default)
def conj_physical(input):
    if not input.dtype.is_complex:
        return input
    return prims.conj_physical(input)


def _number_dtype(value):
    if isinstance(value, bool):
        return tp.bool
    if isinstance(value, int):
        return tp.int64
    if isinstance(value, float):
        return tp.get_default_dtype()
    return tp.complex64


@register_decomposition([ops.full.default, ops.full.out])
def full(size, fill_value, *, dtype=None, device=None, pin_memory=False,
         requires_grad=False):
    dtype = dtype if dtype is not None else _number_dtype(fill_value)
    e = ops.empty.default(
        list(size), dtype=dtype, device=device, pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return ops.fill.Scalar(e, fill_value)


@register_decomposition([ops.bucketize.Tensor, ops.bucketize.Scalar])
def bucketize(self, boundaries, *, out_int32=False, right=False):
    if boundaries.dim() != 1:
        raise RuntimeError(
            f"boundaries tensor must be 1 dimension but got dim({boundaries.dim()})"
        )
    if not isinstance(self, tp.Tensor):
        self = ops.scalar_tensor.default(self, dtype=boundaries.dtype)
    out_dtype = tp.int32 if out_int32 else tp.int64
    n_boundaries = int(boundaries.shape[-1])
    if n_boundaries == 0:
        return ops.zeros_like.default(self, dtype=out_dtype)
    # Each step of the binary search runs over every element at once; the
    # iteration count is the search depth, with a flag tensor freezing the
    # elements whose search has already terminated.
    start = ops.zeros.default(list(self.shape), dtype=tp.int64, device=self.device)
    end = start + n_boundaries
    mid = start + ops.floor_divide.Scalar(end - start, 2)
    mid_val = ops.index.Tensor(boundaries, [mid])
    cond_mid = mid_val > self if right else mid_val >= self
    start = ops.where.self(cond_mid, start, mid + 1)
    if n_boundaries > 1:
        cond_update = ops.ones_like.default(self, dtype=tp.bool)
        for _ in range(int(math.log2(n_boundaries))):
            end = ops.where.self(cond_mid & cond_update, mid, end)
            cond_update = start < end
            mid = ops.where.ScalarOther(
                cond_update, start + ops.floor_divide.Scalar(end - start, 2), 0
            )
            mid_val = ops.index.Tensor(boundaries, [mid])
            cond_mid = mid_val > self if right else mid_val >= self
            start = ops.where.self(
                ops.logical_not.default(cond_mid) & cond_update, mid + 1, start
            )
    return ops.to.dtype(start, out_dtype)


@register_decomposition([ops.addmm.default, ops.addmm.out])
def addmm(self, mat1, mat2, *, beta=1, alpha=1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * ops.mm.default(mat1, mat2)
    if beta == 0:
        return out
    # The product is the leading addend so the result carries its contiguous
    # layout rather than the possibly strided ``self``.
    return out + beta * self


@register_decomposition([ops.addmm.dtype, ops.addmm.dtype_out])
def addmm_dtype(self, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    out = alpha * ops.mm.dtype(mat1, mat2, out_dtype)
    if beta == 0:
        return out
    return out + beta * ops.to.dtype(self, out_dtype)


@register_decomposition([ops._addmm_activation.default, ops._addmm_activation.out])
def _addmm_activation(self, mat1, mat2, *, beta=1, alpha=1, use_gelu=False):
    out = addmm(self, mat1, mat2, beta=beta, alpha=alpha)
    if use_gelu:
        return ops.gelu.default(out)
    return ops.relu.default(out)


@register_decomposition([ops.addmv.default, ops.addmv.out])
def addmv(self, mat, vec, *, beta=1, alpha=1):
    if self.dtype != mat.dtype or mat.dtype != vec.dtype:
        raise RuntimeError(
            f"addmv input tensors must have the same dtype, but got {self.dtype}, "
            f"{mat.dtype}, and {vec.dtype}"
        )
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * ops.mv.default(mat, vec)
    if beta == 0:
        return out
    if out.numel() == 0:
        return beta * self
    return out + beta * self


@register_decomposition(ops.dist.default)
def dist(self, other, p=2):
    return ops.norm.Scalar(self - other, p)


@register_decomposition(ops._euclidean_dist.default)
def _euclidean_dist(x1, x2):
    # |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, evaluated as one matmul over rows
    # extended with the norms and a column of ones.
    x1_norm = ops.sum.dim_IntList(ops.pow.Tensor_Scalar(x1, 2), [-1], True)
    x1_pad = ops.ones_like.default(x1_norm)
    x2_norm = ops.sum.dim_IntList(ops.pow.Tensor_Scalar(x2, 2), [-1], True)
    x2_pad = ops.ones_like.default(x2_norm)
    x1_ = ops.cat.default((x1 * -2, x1_norm, x1_pad), -1)
    x2_ = ops.cat.default((x2, x2_pad, x2_norm), -1)
    result = ops.matmul.default(x1_, ops.transpose.default(x2_, -1, -2))
    return ops.sqrt.default(ops.clamp.default(result, min=0))


@register_decomposition(ops._to_copy.default)
def _to_copy(self, *, dtype=None, layout=None, device=None, pin_memory=None,
             non_blocking=False, memory_format=None):
    if layout is not None and getattr(layout, "name", layout) != "strided":
        raise RuntimeError(f"layout must be None or the strided layout, got {layout}")
    if pin_memory:
        raise RuntimeError("pin_memory=True is not supported in _to_copy decomposition")
    if dtype is None and device is None and memory_format is None:
        return self.clone()
    x = self
    if device is not None and device != x.device:
        # When both dtype and device change, convert on the source device when
        # moving to the host, and on the destination device otherwise.
        if dtype is not None and device.type == "cpu":
            x = prims.convert_element_type(x, dtype)
            dtype = None
        x = prims.device_put(x, device, non_blocking)
    if dtype is not None:
        x = prims.convert_element_type(x, dtype)
    if memory_format is not None:
        x = ops.clone.default(x, memory_format=memory_format)
    return x


@register_decomposition(ops._adaptive_avg_pool2d.default)
def _adaptive_avg_pool2d(input, output_size):
    if input.dim() not in (3, 4):
        raise RuntimeError(
            f"adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {input.dim()}"
        )
    if any(d == 0 for d in input.shape[-2:]):
        raise RuntimeError(
            "adaptive_avg_pool2d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(input.shape)}."
        )
    out_h, out_w = int(output_size[-2]), int(output_size[-1])

    # When every window has the same size the pooling is a strided average.
    if int(input.shape[-2]) % out_h == 0 and int(input.shape[-1]) % out_w == 0:
        stride = [int(input.shape[-2]) // out_h, int(input.shape[-1]) // out_w]
        kernel = [
            int(input.shape[-2]) - (out_h - 1) * stride[0],
            int(input.shape[-1]) - (out_w - 1) * stride[1],
        ]
        return ops.avg_pool2d.default(input, kernel, stride)

    def start_index(a, b, c):
        return ops.floor_divide.Scalar(a * c, b)

    def end_index(a, b, c):
        return ops.floor_divide.Scalar((a + 1) * c + b - 1, b)

    def compute_idx(in_size, out_size):
        orange = ops.arange.end(out_size, dtype=tp.int64)
        i0 = start_index(orange, out_size, in_size)
        # Window lengths vary unless the split is uniform; the longest one is
        # known analytically, so the index plane is padded to that width.
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1
        range_max = ops.arange.end(maxlength, dtype=tp.int64)
        idx = i0.unsqueeze(-1) + range_max
        if adaptive:
            # Rows shorter than the pad width read a clamped last position;
            # the mask below drops those reads from the average.
            idx = ops.minimum.default(
                idx, ops.scalar_tensor.default(in_size - 1, dtype=idx.dtype)
            )
            length = end_index(orange, out_size, in_size) - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    idxh, length_h, range_max_h, adaptive_h = compute_idx(
        int(input.shape[-2]), out_h
    )
    idxw, length_w, range_max_w, adaptive_w = compute_idx(
        int(input.shape[-1]), out_w
    )

    vals = ops.index.Tensor(input, [None, None, _to_rank(idxh, 4), idxw])
    if not adaptive_h and not adaptive_w:
        return ops.mean.dim(vals, [-3, -1])

    def maybe_mask(vals, length, range_max, dim):
        if isinstance(length, int):
            return vals, length
        mask = range_max >= length.unsqueeze(-1)
        if dim == -2:
            mask = _to_rank(mask, 4)
        vals = ops.masked_fill.Scalar(vals, mask, 0.0)
        return vals, _to_rank(length, -dim)

    vals, length_h = maybe_mask(vals, length_h, range_max_h, -2)
    vals, length_w = maybe_mask(vals, length_w, range_max_w, -1)
    total = ops.sum.dim_IntList(vals, [3, 5])
    return total / (length_h * length_w)


# ---------------------------------------------------------------------------
# Distances, random reads and grid sampling
# ---------------------------------------------------------------------------


@register_decomposition(ops.pairwise_distance.default)
def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    return ops.linalg_vector_norm.default(x1 - x2 + eps, p, [-1], keepdim)


@register_decomposition(ops.pdist.default)
def pdist(a, p=2):
    if a.dim() != 2:
        raise RuntimeError(f"pdist only supports 2D tensors, got: {a.dim()}D")
    if p < 0:
        raise RuntimeError("pdist only supports non-negative p values")
    if p == 2:
        # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, read off one gram matrix.
        aTa = ops.mm.default(a, ops.transpose.default(a, 0, 1))
        aTa_diag = ops.diag.default(aTa)
        t = ops.sqrt.default(
            ops.clamp.default(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)
        )
    else:
        t = ops.linalg_vector_norm.default(a.unsqueeze(1) - a, p, [2], False)
    i = ops.triu_indices.default(
        int(t.shape[0]), int(t.shape[1]), 1, dtype=tp.int64, device=a.device
    )
    rows = ops.select.int(i, 0, 0) * int(t.shape[0]) + ops.select.int(i, 0, 1)
    return ops.index_select.default(ops.reshape.default(t, [-1]), 0, rows)


def _normal_impl(mean, std, size=None, *, generator=None, dtype=None,
                 device=None, layout=None, pin_memory=None):
    _no_generator("normal", generator)
    if layout is not None and getattr(layout, "name", layout) != "strided":
        raise RuntimeError(f"layout must be None or the strided layout, got {layout}")
    if not isinstance(std, tp.Tensor) and std < 0:
        raise RuntimeError(f"normal expects std >= 0.0, but found std {std}")
    if size is None:
        tensors = [t for t in (mean, std) if isinstance(t, tp.Tensor)]
        if not tensors:
            raise RuntimeError(
                "normal expects that either mean or std is a tensor, or size is defined"
            )
        shape = tensors[0].shape
        for t in tensors[1:]:
            shape = _broadcast_shapes(shape, t.shape)
        target = tp.result_type(mean, std)
        device = tensors[0].device
    else:
        if isinstance(mean, tp.Tensor) or isinstance(std, tp.Tensor):
            raise RuntimeError(
                "normal expects mean and std to be scalars when size is defined"
            )
        shape = list(size)
        target = dtype if dtype is not None else tp.get_default_dtype()
        device = device if device is not None else tp.get_default_device()
    samples = prims.normal(
        list(shape), mean=0.0, std=1.0, dtype=target, device=device,
        requires_grad=False,
    )
    return std * samples + mean


@register_decomposition(ops.normal.Tensor_Tensor)
def normal_tensor_tensor(mean, std, *, generator=None):
    return _normal_impl(mean, std, generator=generator)


@register_decomposition(ops.normal.Tensor_float)
def normal_tensor_float(mean, std=1, *, generator=None):
    return _normal_impl(mean, std, generator=generator)


@register_decomposition(ops.normal.float_Tensor)
def normal_float_tensor(mean, std, *, generator=None):
    return _normal_impl(mean, std, generator=generator)


@register_decomposition(ops.normal.float_float)
def normal_float_float(mean=0, std=1, size=None, *, generator=None, dtype=None,
                       layout=None, device=None, pin_memory=None):
    return _normal_impl(mean, std, size, generator=generator, dtype=dtype,
                        device=device, layout=layout, pin_memory=pin_memory)


@register_decomposition(ops.normal_functional.default)
def normal_functional(self, mean=0, std=1, *, generator=None):
    return _normal_impl(
        mean, std, list(self.shape), generator=generator, dtype=self.dtype,
        device=self.device,
    )


@register_decomposition(ops.grid_sampler_2d.default)
def grid_sampler_2d(a, grid, interpolation_mode, padding_mode, align_corners):
    if interpolation_mode not in (0, 1, 2):
        raise RuntimeError(f"Invalid interpolation mode {interpolation_mode}")
    if padding_mode not in (0, 1, 2):
        raise RuntimeError(f"Invalid padding mode {padding_mode}")

    def unnormalize(coords, size):
        # [-1, 1] maps onto [0, size - 1] with corner alignment and onto
        # [-0.5, size - 0.5] without it.
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    def reflect_coordinates(coords, twice_low, twice_high):
        if twice_low == twice_high:
            return ops.zeros_like.default(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = ops.fmod.Scalar(coords2, coords_span)
        flips = ops.floor.default(coords2 / coords_span).to(tp.int8)
        return ops.where.self(
            (flips & 1) == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords, size):
        if padding_mode == 0:
            return coords
        if padding_mode == 1:
            return tp.clamp(coords, 0, size - 1)
        if align_corners:
            coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
        else:
            coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
        return tp.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords, size):
        return compute_coordinates(unnormalize(coords, size), size)

    N, C, iH, iW = a.shape
    _, oH, oW, two = grid.shape
    if two != 2:
        raise RuntimeError(f"grid last dimension must be 2 (for x,y coords), got {two}")

    def in_bounds_cond(xs, ys):
        return ops.logical_and.default(
            0 <= xs,
            ops.logical_and.default(
                xs < iW, ops.logical_and.default(0 <= ys, ys < iH)
            ),
        )

    n_idx = ops.reshape.default(tp.arange(N, device=a.device), [N, 1, 1, 1])
    c_idx = ops.reshape.default(tp.arange(C, device=a.device), [1, C, 1, 1])

    def clip(xs, ys, ws):
        # Reads outside the image land at (0, 0) with weight zero, so the
        # gather stays in bounds and contributes nothing.
        cond = in_bounds_cond(xs, ys)
        out = []
        for t in (ops.to.dtype(xs, tp.int64), ops.to.dtype(ys, tp.int64), ws):
            if isinstance(t, tp.Tensor):
                picked = ops.where.ScalarOther(cond, t, 0)
            else:
                picked = ops.where.Scalar(cond, t, 0)
            out.append(ops.reshape.default(picked, [N, 1, oH, oW]))
        return tuple(out)

    def get_summand(ix, iy, w):
        idx_x, idx_y, w_ = clip(ix, iy, w)
        return ops.index.Tensor(a, [n_idx, c_idx, idx_y, idx_x]) * w_

    x = ops.select.int(grid, -1, 0)
    y = ops.select.int(grid, -1, 1)

    if interpolation_mode == 0:
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)
        ix_nw, iy_nw = ops.floor.default(ix), ops.floor.default(iy)
        ix_ne, iy_sw = ix_nw + 1, iy_nw + 1
        w_nw = (ix_ne - ix) * (iy_sw - iy)
        w_ne = (ix - ix_nw) * (iy_sw - iy)
        w_sw = (ix_ne - ix) * (iy - iy_nw)
        w_se = (ix - ix_nw) * (iy - iy_nw)
        return _sum_tensors(
            get_summand(ix, iy, w)
            for ix, iy, w in (
                (ix_nw, iy_nw, w_nw),
                (ix_ne, iy_nw, w_ne),
                (ix_nw, iy_sw, w_sw),
                (ix_ne, iy_sw, w_se),
            )
        )
    if interpolation_mode == 1:
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)
        return get_summand(ops.round.default(ix), ops.round.default(iy), 1)
    # Bicubic: the grid drives the sample positions directly; reflection and
    # clamping happen per read.
    ix = unnormalize(x, iW)
    iy = unnormalize(y, iH)
    ix_nw, iy_nw = ops.floor.default(ix), ops.floor.default(iy)
    # The grid stays unexpanded, so the fractional parts carry a singleton
    # channel dim to broadcast against the gathered values.
    tx = (ix - ix_nw).unsqueeze(1)
    ty = (iy - iy_nw).unsqueeze(1)

    def get_value_bounded(ix_, iy_):
        return get_summand(
            compute_coordinates(ix_, iW), compute_coordinates(iy_, iH), 1
        )

    def get_coeff(ofs):
        iy_ofs = iy_nw + (ofs - 1)
        cs = tuple(
            get_value_bounded(ix_nw + delta, iy_ofs) for delta in (-1, 0, 1, 2)
        )
        return _sum_tensors(s * w for s, w in zip(cs, _cubic_coefficients(tx)))

    coeffs = tuple(get_coeff(ofs) for ofs in range(4))
    return _sum_tensors(s * w for s, w in zip(coeffs, _cubic_coefficients(ty)))
