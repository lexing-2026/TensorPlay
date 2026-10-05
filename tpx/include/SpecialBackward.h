#pragma once
// Backward of the first-order Bessel functions.
//
// Their derivatives divide by x.  At the origin that quotient is 0/0 while
// the derivative has a finite limit, so the origin is replaced by a harmless
// stand-in before the division and the limit is selected there afterwards;
// selecting with where keeps the stand-in's value out of the result and out
// of a recorded second derivative.  A NaN input fails the |x| <= eps test and
// keeps propagating NaN.

#include "Autograd.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <limits>

namespace tensorplay {
namespace tpx {

namespace special_bwd_detail {

// The machine epsilon of the input's own type, which bounds the band of
// inputs the quotient cannot resolve.
inline double epsilon_of(DType dtype) {
    switch (dtype) {
        case DType::Float64: return std::numeric_limits<double>::epsilon();
        case DType::Float16: return 0.0009765625;
        case DType::BFloat16: return 0.0078125;
        default: return static_cast<double>(std::numeric_limits<float>::epsilon());
    }
}

// grad * (lead(x) - result * (shift(x) + 1/x)), with `limit` at |x| <= eps.
template <typename Lead, typename Shift>
Tensor divided_by_x_backward(const Tensor& grad, const Tensor& self, const Tensor& result,
                             Lead lead, Shift shift, double limit) {
    const Scalar eps(epsilon_of(self.dtype()));
    const Tensor tiny = ops::le(ops::abs(self), eps);
    const Tensor safe = ops::where(tiny, eps, self);
    const Tensor slope = ops::sub(lead(safe), ops::mul(result, shift(safe)));
    return ops::mul(grad, ops::where(tiny, Scalar(limit), slope));
}

}  // namespace special_bwd_detail

// I1'(x) = I0(x) - I1(x) / x, with the limit 1/2 at the origin.
inline Tensor i1_backward(const Tensor& grad, const Tensor& self, const Tensor& result) {
    return special_bwd_detail::divided_by_x_backward(
        grad, self, result, [](const Tensor& x) { return ops::i0(x); },
        [](const Tensor& x) { return ops::reciprocal(x); }, 0.5);
}

// d/dx e^-|x| I1(x) = e^-|x| I0(x) - e^-|x| I1(x) (sgn x + 1/x), with the
// limit 1/2 at the origin.
inline Tensor i1e_backward(const Tensor& grad, const Tensor& self, const Tensor& result) {
    return special_bwd_detail::divided_by_x_backward(
        grad, self, result, [](const Tensor& x) { return ops::i0e(x); },
        [](const Tensor& x) { return ops::add(ops::sgn(x), ops::reciprocal(x)); }, 0.5);
}

// J1'(x) = J0(x) - J1(x) / x, with the limit 1/2 at the origin.
inline Tensor bessel_j1_backward(const Tensor& grad, const Tensor& self, const Tensor& result) {
    return special_bwd_detail::divided_by_x_backward(
        grad, self, result, [](const Tensor& x) { return ops::bessel_j0(x); },
        [](const Tensor& x) { return ops::reciprocal(x); }, 0.5);
}

// Y1'(x) = Y0(x) - Y1(x) / x.  Both terms are -inf at the origin, where the
// one-sided limit of the derivative is +inf.
inline Tensor bessel_y1_backward(const Tensor& grad, const Tensor& self, const Tensor& result) {
    const Tensor slope =
        ops::sub(ops::bessel_y0(self), ops::mul(result, ops::reciprocal(self)));
    return ops::mul(grad, ops::where(ops::eq(self, Scalar(0.0)),
                                     Scalar(std::numeric_limits<double>::infinity()), slope));
}

}  // namespace tpx
}  // namespace tensorplay
