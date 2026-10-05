#pragma once
// Second derivatives through the activation and softmax backward kernels.
//
// Under create_graph the first backward pass runs kernels such as
// sigmoid_backward on tensors that belong to a graph.  The helpers below
// differentiate those kernels with respect to the activation's input (or
// output); the formulas that use them sit next to the kernels' schemas in
// derivatives.yaml.  Each is written with differentiable operations, so a
// third pass records as well.

#include "Autograd.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>

namespace tensorplay {
namespace tpx {

namespace activation_bwd_detail {

constexpr double kPi = 3.14159265358979323846;

// A 0/1 mask in the type of `like`, for multiplying a gradient.
inline Tensor mask_like(const Tensor& mask, const Tensor& like) {
    return ops::to(mask, like.dtype());
}

}  // namespace activation_bwd_detail

// softmax_backward(gO, y) = y * (gO - sum(gO * y)) differentiated in y,
// applied to the incoming gradient `grad`.
inline Tensor softmax_double_backward(const Tensor& grad, const Tensor& grad_output,
                                      int64_t dim, const Tensor& output) {
    const std::vector<int64_t> dims{dim};
    return ops::sub(
        ops::sub(ops::mul(grad_output, grad),
                 ops::mul(ops::sum(ops::mul(output, grad_output), dims, true), grad)),
        ops::mul(grad_output, ops::sum(ops::mul(output, grad), dims, true)));
}

// log_softmax_backward(gO, y) = gO - exp(y) * sum(gO): its transpose in gO
// and its slope in y, applied to the incoming gradient `grad`.
inline Tensor log_softmax_double_backward_grad_output(const Tensor& grad, int64_t dim,
                                                      const Tensor& output) {
    return ops::sub(grad, ops::sum(ops::mul(grad, ops::exp(output)),
                                   std::vector<int64_t>{dim}, true));
}

inline Tensor log_softmax_double_backward(const Tensor& grad, const Tensor& grad_output,
                                          int64_t dim, const Tensor& output) {
    return ops::neg(ops::mul(
        ops::mul(ops::sum(grad_output, std::vector<int64_t>{dim}, true), ops::exp(output)),
        grad));
}

// elu_backward differentiated in its saved input or output: only the
// negative side, where the slope is exponential, has a second derivative.
inline Tensor elu_double_backward(const Tensor& grad, const Tensor& grad_output,
                                  const Scalar& alpha, const Scalar& scale,
                                  const Scalar& input_scale, bool is_result,
                                  const Tensor& self_or_result) {
    const Tensor negative = activation_bwd_detail::mask_like(
        ops::lt(self_or_result, Scalar(0)), grad);
    const Tensor scaled = ops::mul(ops::mul(grad, grad_output), input_scale);
    if (is_result) {
        return ops::mul(scaled, negative);
    }
    return ops::mul(
        ops::elu_backward(scaled, alpha, scale, input_scale, false, self_or_result),
        negative);
}

// gelu''(x) times the incoming gradient and the first pass's gradient.
inline Tensor gelu_double_backward(const Tensor& grad, const Tensor& grad_output,
                                   const Tensor& self, const std::string& approximate) {
    if (approximate == "tanh") {
        // gelu(x) = f g' with f = x / 2, g = 1 + tanh(u), u = b (x + k x^3).
        const double beta = std::sqrt(2.0 / activation_bwd_detail::kPi);
        const double kappa = 0.044715;
        const Tensor inner =
            ops::mul(ops::add(self, ops::mul(ops::pow(self, Scalar(3)), Scalar(kappa))),
                     Scalar(beta));
        const Tensor tanh_inner = ops::tanh(inner);
        const Tensor sech_inner = ops::reciprocal(ops::cosh(inner));
        const Tensor f = ops::mul(self, Scalar(0.5));
        const Tensor g = ops::rsub(ops::mul(tanh_inner, tanh_inner), Scalar(1));
        const Tensor h = ops::mul(
            ops::add(ops::mul(ops::mul(self, self), Scalar(3 * kappa)), Scalar(1)),
            Scalar(beta));
        const Tensor f_prime_gh = ops::mul(ops::mul(g, h), Scalar(0.5));
        const Tensor g_prime = ops::mul(
            ops::mul(ops::mul(sech_inner, Scalar(2)), ops::neg(ops::mul(sech_inner, tanh_inner))),
            h);
        const Tensor g_prime_fh = ops::mul(ops::mul(f, h), g_prime);
        const Tensor h_prime = ops::mul(self, Scalar(6 * kappa * beta));
        const Tensor h_prime_fg = ops::mul(ops::mul(f, g), h_prime);
        const Tensor second = ops::add(
            ops::add(ops::mul(f_prime_gh, Scalar(2)), g_prime_fh), h_prime_fg);
        return ops::mul(ops::mul(grad, grad_output), second);
    }
    // gelu(x) = x Phi(x): gelu''(x) = 2 phi(x) - x^2 phi(x).
    const double beta = 1.0 / std::sqrt(2.0 * activation_bwd_detail::kPi);
    const Tensor input_sq = ops::mul(self, self);
    const Tensor pdf = ops::mul(ops::exp(ops::mul(input_sq, Scalar(-0.5))), Scalar(beta));
    const Tensor second = ops::sub(ops::mul(pdf, Scalar(2)), ops::mul(input_sq, pdf));
    return ops::mul(ops::mul(grad, grad_output), second);
}

// softplus'(x) = sigmoid(beta x) below the threshold; its slope in x.
inline Tensor softplus_double_backward(const Tensor& grad, const Tensor& self,
                                       const Scalar& beta, const Scalar& threshold) {
    const Tensor x = ops::mul(self, beta);
    const Tensor below = activation_bwd_detail::mask_like(ops::lt(x, threshold), grad);
    return ops::mul(ops::mul(ops::sigmoid_backward(grad, ops::sigmoid(x)), below), beta);
}

// log_sigmoid'(x) = 1 - sigmoid(x), whose slope is (sigmoid(x) - 1) sigmoid(x).
inline Tensor log_sigmoid_double_backward(const Tensor& grad, const Tensor& self) {
    const Tensor z = ops::sigmoid(self);
    return ops::mul(ops::mul(grad, ops::sub(z, Scalar(1))), z);
}

// hardswish'(x) = (2x + 3) / 6 on (-3, 3): a constant slope of 1/3 there.
inline Tensor hardswish_double_backward(const Tensor& grad, const Tensor& grad_output,
                                        const Tensor& self) {
    const Tensor inside = ops::logical_and(ops::gt(self, Scalar(-3.0)),
                                           ops::lt(self, Scalar(3.0)));
    return ops::where(inside, ops::div(ops::mul(grad, grad_output), Scalar(3.0)),
                      Scalar(0.0));
}

// glu_backward differentiated in the input: the gate's second half carries
// the sigmoid's curvature, the first half only the gate's slope.
inline Tensor glu_double_backward(const Tensor& grad, const Tensor& grad_output,
                                  const Tensor& self, int64_t dim) {
    if (dim < 0) dim += self.dim();
    const int64_t half = self.size(dim) / 2;
    const Tensor first_half = ops::narrow(self, dim, 0, half);
    const Tensor second_half = ops::narrow(self, dim, half, half);
    const Tensor sig = ops::sigmoid(second_half);
    const Tensor one_sub_sig = ops::rsub(sig, Scalar(1));
    const Tensor sig_one_sub_sig = ops::mul(sig, one_sub_sig);
    const Tensor gg_first = ops::narrow(grad, dim, 0, half);
    const Tensor gg_second = ops::narrow(grad, dim, half, half);
    const Tensor gi_first = ops::mul(ops::mul(gg_second, grad_output), sig_one_sub_sig);
    const Tensor second_order = ops::sub(ops::mul(sig_one_sub_sig, one_sub_sig),
                                         ops::mul(sig, sig_one_sub_sig));
    const Tensor gi_second = ops::add(
        ops::mul(ops::mul(ops::mul(gg_second, first_half), grad_output), second_order),
        ops::mul(ops::mul(gg_first, grad_output), sig_one_sub_sig));
    return ops::cat({gi_first, gi_second}, dim);
}

// glu_backward is linear in the incoming gradient; its transpose folds the
// two halves of the kernel's slope back onto the output's shape.
inline Tensor glu_double_backward_grad_output(const Tensor& grad, const Tensor& self,
                                              int64_t dim) {
    if (dim < 0) dim += self.dim();
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
    sizes[dim] /= 2;
    const Tensor ones = ops::ones(sizes, self.dtype(), self.device());
    const Tensor tmp = ops::mul(grad, ops::glu_backward(ones, self, dim));
    return ops::add(ops::narrow(tmp, dim, 0, sizes[dim]),
                    ops::narrow(tmp, dim, sizes[dim], sizes[dim]));
}

// The backward of silu, mish and logit: the fused kernel when the pass
// records nothing, differentiable arithmetic when it does (create_graph).
inline Tensor silu_backward_recordable(const Tensor& grad, const Tensor& self) {
    if (!GradMode::is_enabled()) {
        return ops::silu_backward(grad, self);
    }
    // grad * s * (1 + x (1 - s)) with s = sigmoid(x).
    const Tensor s = ops::sigmoid(self);
    return ops::mul(ops::mul(grad, s),
                    ops::add(ops::mul(self, ops::rsub(s, Scalar(1))), Scalar(1)));
}

inline Tensor mish_backward_recordable(const Tensor& grad, const Tensor& self) {
    if (!GradMode::is_enabled()) {
        return ops::mish_backward(grad, self);
    }
    // grad * (t + x sigmoid(x) (1 - t^2)) with t = tanh(softplus(x)).
    const Tensor s = ops::sigmoid(self);
    const Tensor t = ops::tanh(ops::softplus(self, Scalar(1), Scalar(20)));
    return ops::mul(grad, ops::add(t, ops::mul(ops::mul(self, s),
                                               ops::rsub(ops::mul(t, t), Scalar(1)))));
}

inline Tensor logit_backward_recordable(const Tensor& grad, const Tensor& self,
                                        const std::optional<Scalar>& eps) {
    if (!GradMode::is_enabled()) {
        return ops::logit_backward(grad, self, eps);
    }
    // grad / (x (1 - x)) inside the clamped band; zero outside it with eps,
    // NaN outside [0, 1] without.
    const double lo = eps.has_value() ? eps->toDouble() : 0.0;
    const double hi = 1.0 - lo;
    const Tensor inside = ops::logical_and(ops::ge(self, Scalar(lo)), ops::le(self, Scalar(hi)));
    const Tensor slope = ops::div(grad, ops::mul(self, ops::rsub(self, Scalar(1))));
    const Scalar outside = eps.has_value()
        ? Scalar(0.0) : Scalar(std::numeric_limits<double>::quiet_NaN());
    return ops::where(inside, slope, outside);
}

}  // namespace tpx
}  // namespace tensorplay
