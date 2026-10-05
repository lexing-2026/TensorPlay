#pragma once
// Gradients for the loss targets and second derivatives through the loss
// backward kernels.
//
// Each elementwise loss l(x, t) reduces to none / mean / sum, and its backward
// kernel is  scale * grad_output * l'(x, t)  with scale = 1 / numel for mean.
// The kernel is linear in grad_output, so the grad_output slot of its
// derivative is the transpose (a reduction of slope * grad), and the other
// slots differentiate the slope.  Every helper is written with differentiable
// operations so a third pass records as well.

#include "Autograd.h"
#include "GradMode.h"
#include "Node.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <optional>
#include <tuple>

namespace tensorplay {
namespace tpx {

namespace loss_bwd_detail {

// A 0/1 mask in the type of `like`, for multiplying a gradient.
inline Tensor mask_like(const Tensor& mask, const Tensor& like) {
    return ops::to(mask, like.dtype());
}

// The factor a mean reduction puts on every element's gradient.
inline Tensor mean_scaled(const Tensor& t, int64_t reduction, int64_t numel) {
    if (reduction == 1) return ops::div(t, Scalar(static_cast<double>(numel)));
    return t;
}

inline Tensor weighted(const Tensor& t, const std::optional<Tensor>& weight) {
    if (weight.has_value() && weight->defined()) return ops::mul(t, *weight);
    return t;
}

}  // namespace loss_bwd_detail

// Transpose of a backward kernel in its grad_output: the kernel computes
// grad_output * slope (per element, then divided by numel for mean), so for a
// reduced loss the gradient of the 0-d grad_output is the sum over elements.
inline Tensor loss_grad_output(const Tensor& grad, const Tensor& slope, int64_t reduction) {
    const Tensor product = ops::mul(grad, slope);
    if (reduction == 0) return product;
    return loss_bwd_detail::mean_scaled(ops::sum(product), reduction, grad.numel());
}

// d/d self of the backward kernels whose slope is constant (mse).
inline Tensor mse_loss_double_backward(const Tensor& grad, int64_t reduction) {
    return loss_bwd_detail::mean_scaled(ops::mul(grad, Scalar(2.0)), reduction, grad.numel());
}

// smooth_l1 slope is (x - t) / beta inside the quadratic zone, sign outside.
inline Tensor smooth_l1_loss_double_backward(const Tensor& grad, const Tensor& self,
                                             const Tensor& target, int64_t reduction,
                                             double beta) {
    if (beta == 0.0) return ops::zeros_like(grad);
    const Tensor inside = loss_bwd_detail::mask_like(
        ops::lt(ops::abs(ops::sub(self, target)), Scalar(beta)), grad);
    return loss_bwd_detail::mean_scaled(ops::div(ops::mul(grad, inside), Scalar(beta)),
                                        reduction, grad.numel());
}

// huber slope is x - t inside the quadratic zone, delta * sign outside.
inline Tensor huber_loss_double_backward(const Tensor& grad, const Tensor& self,
                                         const Tensor& target, int64_t reduction,
                                         double delta) {
    const Tensor inside = loss_bwd_detail::mask_like(
        ops::lt(ops::abs(ops::sub(self, target)), Scalar(delta)), grad);
    return loss_bwd_detail::mean_scaled(ops::mul(grad, inside), reduction, grad.numel());
}

// binary_cross_entropy: slope (x - t) / max(x (1 - x), 1e-12), times the weight.
inline Tensor bce_denominator(const Tensor& self) {
    return ops::clamp_min(ops::mul(self, ops::rsub(self, Scalar(1.0))), Scalar(1e-12));
}

// d loss / d target = log(1 - x) - log(x), each logarithm clamped at -100.
inline Tensor binary_cross_entropy_target_backward(const Tensor& grad, const Tensor& self,
                                                   const std::optional<Tensor>& weight,
                                                   int64_t reduction) {
    const Tensor log_x = ops::clamp_min(ops::log(self), Scalar(-100.0));
    const Tensor log_1mx = ops::clamp_min(ops::log(ops::rsub(self, Scalar(1.0))), Scalar(-100.0));
    Tensor g = ops::mul(grad, ops::sub(log_1mx, log_x));
    g = loss_bwd_detail::weighted(g, weight);
    return loss_bwd_detail::mean_scaled(g, reduction, self.numel());
}

// d slope / d x = (x^2 - 2 x t + t) / (x (1 - x))^2.
inline Tensor binary_cross_entropy_double_backward(const Tensor& grad, const Tensor& self,
                                                   const Tensor& target,
                                                   const std::optional<Tensor>& weight,
                                                   int64_t reduction) {
    const Tensor numerator = ops::add(
        ops::sub(ops::mul(self, self), ops::mul(ops::mul(self, target), Scalar(2.0))), target);
    const Tensor denom = bce_denominator(self);
    Tensor g = ops::mul(grad, ops::div(numerator, ops::mul(denom, denom)));
    g = loss_bwd_detail::weighted(g, weight);
    return loss_bwd_detail::mean_scaled(g, reduction, self.numel());
}

// d slope / d t = -1 / (x (1 - x)).
inline Tensor binary_cross_entropy_double_backward_target(const Tensor& grad, const Tensor& self,
                                                          const std::optional<Tensor>& weight,
                                                          int64_t reduction) {
    Tensor g = ops::neg(ops::div(grad, bce_denominator(self)));
    g = loss_bwd_detail::weighted(g, weight);
    return loss_bwd_detail::mean_scaled(g, reduction, self.numel());
}

// kl_div: the pointwise loss is t log t - t x (a zero target contributes
// nothing) or exp(t) (t - x) when the target holds log-probabilities.
inline Tensor kl_div_target_backward(const Tensor& grad, const Tensor& input,
                                     const Tensor& target, int64_t reduction, bool log_target) {
    Tensor slope;
    if (log_target) {
        slope = ops::mul(ops::exp(target),
                         ops::add(ops::sub(target, input), Scalar(1.0)));
    } else {
        const Tensor nonzero = ops::ne(target, Scalar(0));
        const Tensor safe = ops::where(nonzero, target, Scalar(1.0));
        slope = ops::mul(ops::sub(ops::add(ops::log(safe), Scalar(1.0)), input),
                         loss_bwd_detail::mask_like(nonzero, input));
    }
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input.numel());
}

// d (kl_div slope in x) / d t = -[t != 0] or -exp(t).
inline Tensor kl_div_double_backward_target(const Tensor& grad, const Tensor& input,
                                            const Tensor& target, int64_t reduction,
                                            bool log_target) {
    const Tensor slope = log_target
        ? ops::neg(ops::exp(target))
        : ops::neg(loss_bwd_detail::mask_like(ops::ne(target, Scalar(0)), input));
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input.numel());
}

// soft margin: loss log(1 + exp(-t x)), slope -t sigmoid(-t x).
inline Tensor soft_margin_loss_target_backward(const Tensor& grad, const Tensor& input,
                                               const Tensor& target, int64_t reduction) {
    const Tensor s = ops::sigmoid(ops::neg(ops::mul(input, target)));
    return loss_bwd_detail::mean_scaled(ops::mul(grad, ops::neg(ops::mul(input, s))),
                                        reduction, input.numel());
}

// d slope / d x = t^2 s (1 - s) with s = sigmoid(-t x).
inline Tensor soft_margin_loss_double_backward(const Tensor& grad, const Tensor& input,
                                               const Tensor& target, int64_t reduction) {
    const Tensor s = ops::sigmoid(ops::neg(ops::mul(input, target)));
    const Tensor curvature = ops::mul(ops::mul(target, target), ops::mul(s, ops::rsub(s, Scalar(1.0))));
    return loss_bwd_detail::mean_scaled(ops::mul(grad, curvature), reduction, input.numel());
}

// d slope / d t = -s + t x s (1 - s).
inline Tensor soft_margin_loss_double_backward_target(const Tensor& grad, const Tensor& input,
                                                      const Tensor& target, int64_t reduction) {
    const Tensor s = ops::sigmoid(ops::neg(ops::mul(input, target)));
    const Tensor slope = ops::add(
        ops::neg(s), ops::mul(ops::mul(ops::mul(target, input), s), ops::rsub(s, Scalar(1.0))));
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input.numel());
}

// poisson nll: loss exp(x) - t x (log input) or x - t log(x + eps), plus the
// Stirling term t log t - t + log(2 pi t) / 2 where t > 1 when `full`.
inline Tensor poisson_nll_loss_target_backward(const Tensor& grad, const Tensor& input,
                                               const Tensor& target, bool log_input, bool full,
                                               double eps, int64_t reduction) {
    Tensor slope = log_input ? ops::neg(input)
                             : ops::neg(ops::log(ops::add(input, Scalar(eps))));
    if (full) {
        const Tensor positive = ops::gt(target, Scalar(1));
        const Tensor safe = ops::where(positive, target, Scalar(1.0));
        const Tensor stirling = ops::add(ops::log(safe), ops::div(ops::reciprocal(safe), Scalar(2.0)));
        slope = ops::add(slope, ops::mul(stirling, loss_bwd_detail::mask_like(positive, input)));
    }
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input.numel());
}

// d slope / d x = exp(x) or t / (x + eps)^2.
inline Tensor poisson_nll_loss_double_backward(const Tensor& grad, const Tensor& input,
                                               const Tensor& target, bool log_input,
                                               double eps, int64_t reduction) {
    Tensor curvature;
    if (log_input) {
        curvature = ops::exp(input);
    } else {
        const Tensor shifted = ops::add(input, Scalar(eps));
        curvature = ops::div(target, ops::mul(shifted, shifted));
    }
    return loss_bwd_detail::mean_scaled(ops::mul(grad, curvature), reduction, input.numel());
}

// d slope / d t = -1 or -1 / (x + eps).
inline Tensor poisson_nll_loss_double_backward_target(const Tensor& grad, const Tensor& input,
                                                      bool log_input, double eps,
                                                      int64_t reduction) {
    const Tensor slope = log_input
        ? ops::full_like(input, Scalar(-1.0))
        : ops::neg(ops::reciprocal(ops::add(input, Scalar(eps))));
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input.numel());
}

// margin ranking: loss max(0, -t (x1 - x2) + margin); the target only moves
// the loss while the hinge is active.
inline Tensor margin_ranking_active(const Tensor& input1, const Tensor& input2,
                                    const Tensor& target, double margin) {
    const Tensor pre = ops::add(ops::neg(ops::mul(ops::sub(input1, input2), target)),
                                Scalar(margin));
    return loss_bwd_detail::mask_like(ops::gt(pre, Scalar(0.0)), input1);
}

inline Tensor margin_ranking_loss_target_backward(const Tensor& grad, const Tensor& input1,
                                                  const Tensor& input2, const Tensor& target,
                                                  double margin, int64_t reduction) {
    const Tensor active = margin_ranking_active(input1, input2, target, margin);
    const Tensor slope = ops::mul(ops::sub(input2, input1), active);
    return loss_bwd_detail::mean_scaled(ops::mul(grad, slope), reduction, input1.numel());
}

// cosine embedding: d loss / d cos for each row, with the mean's 1 / rows.
inline Tensor cosine_loss_slope(const Tensor& cosine, const Tensor& target, double margin,
                                int64_t reduction) {
    const Tensor positive = loss_bwd_detail::mask_like(ops::eq(target, Scalar(1)), cosine);
    const Tensor negative = loss_bwd_detail::mask_like(ops::eq(target, Scalar(-1)), cosine);
    const Tensor above = loss_bwd_detail::mask_like(ops::gt(cosine, Scalar(margin)), cosine);
    const Tensor slope = ops::sub(ops::mul(negative, above), positive);
    return loss_bwd_detail::mean_scaled(slope, reduction, cosine.numel());
}

// The second-derivative nodes of the two-output backward kernels.  Their
// outputs are both gradients, so the engine delivers one incoming gradient
// per output; they are listed in the MANUAL_DERIVATIVES table of
// tools/codegen/gen_autograd.py.
struct TpMarginRankingLossBackwardBackward : public Node {
    SavedVariable grad_output_;
    SavedVariable input1_;
    SavedVariable input2_;
    SavedVariable target_;
    double margin_;
    int64_t reduction_;

    TpMarginRankingLossBackwardBackward(Tensor grad_output, Tensor input1, Tensor input2,
                                        Tensor target, double margin, int64_t reduction)
        : grad_output_(std::move(grad_output)), input1_(std::move(input1)),
          input2_(std::move(input2)), target_(std::move(target)), margin_(margin),
          reduction_(reduction) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor gg1 = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor gg2 = inputs.size() > 1 ? inputs[1] : Tensor();
        if (!gg1.defined() && !gg2.defined()) return {Tensor(), Tensor(), Tensor(), Tensor()};
        const Tensor grad_output = grad_output_.unpack();
        const Tensor input1 = input1_.unpack();
        const Tensor input2 = input2_.unpack();
        const Tensor target = target_.unpack();
        if (!grad_output.defined() || !input1.defined() || !input2.defined() || !target.defined()) {
            return {Tensor(), Tensor(), Tensor(), Tensor()};
        }
        // The kernel returns (-h, h) for h = active * t * g * scale, so the two
        // incoming gradients act together as u = gg2 - gg1 on h.
        Tensor u;
        if (gg1.defined() && gg2.defined()) u = ops::sub(gg2, gg1);
        else if (gg2.defined()) u = gg2;
        else u = ops::neg(gg1);

        const Tensor active = margin_ranking_active(input1, input2, target, margin_);
        variable_list grads;
        if (should_compute_output(0)) {
            grads.push_back(loss_grad_output(u, ops::mul(active, target), reduction_));
        } else {
            grads.push_back(Tensor());
        }
        grads.push_back(should_compute_output(1) ? ops::zeros_like(input1) : Tensor());
        grads.push_back(should_compute_output(2) ? ops::zeros_like(input2) : Tensor());
        if (should_compute_output(3)) {
            grads.push_back(loss_bwd_detail::mean_scaled(
                ops::mul(ops::mul(u, active), grad_output), reduction_, input1.numel()));
        } else {
            grads.push_back(Tensor());
        }
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input1_.reset_data();
        input2_.reset_data();
        target_.reset_data();
    }
};

struct TpCosineEmbeddingLossBackwardBackward : public Node {
    SavedVariable grad_output_;
    SavedVariable input1_;
    SavedVariable input2_;
    SavedVariable target_;
    double margin_;
    int64_t reduction_;

    TpCosineEmbeddingLossBackwardBackward(Tensor grad_output, Tensor input1, Tensor input2,
                                          Tensor target, double margin, int64_t reduction)
        : grad_output_(std::move(grad_output)), input1_(std::move(input1)),
          input2_(std::move(input2)), target_(std::move(target)), margin_(margin),
          reduction_(reduction) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        const Tensor gg1 = inputs.size() > 0 ? inputs[0] : Tensor();
        const Tensor gg2 = inputs.size() > 1 ? inputs[1] : Tensor();
        if (!gg1.defined() && !gg2.defined()) return {Tensor(), Tensor(), Tensor(), Tensor()};
        const Tensor grad_output = grad_output_.unpack();
        const Tensor a = input1_.unpack();
        const Tensor b = input2_.unpack();
        const Tensor target = target_.unpack();
        if (!grad_output.defined() || !a.defined() || !b.defined() || !target.defined()) {
            return {Tensor(), Tensor(), Tensor(), Tensor()};
        }
        const Tensor u = gg1.defined() ? gg1 : ops::zeros_like(a);
        const Tensor v = gg2.defined() ? gg2 : ops::zeros_like(b);

        // The kernel returns w * (dcos/da, dcos/db) with w = dloss/dcos * g.
        // With A = |a|^2, B = |b|^2, D = sqrt(A B) and c = <a, b> / D, the
        // pair (u, v) moves cos by
        //   J = q / D - c m,  q = <u, b> + <v, a>,  m = <u, a> / A + <v, b> / B,
        // and the derivative of w * J in a and b is below.
        const std::vector<int64_t> rows{1};
        auto row = [&](const Tensor& t) { return ops::sum(t, rows, false); };
        auto col = [&](const Tensor& t) { return ops::unsqueeze(t, 1); };
        const Tensor A = ops::add(row(ops::mul(a, a)), Scalar(1e-12));
        const Tensor B = ops::add(row(ops::mul(b, b)), Scalar(1e-12));
        const Tensor D = ops::sqrt(ops::mul(A, B));
        const Tensor c = ops::div(row(ops::mul(a, b)), D);
        const Tensor q = ops::add(row(ops::mul(u, b)), row(ops::mul(v, a)));
        const Tensor r = row(ops::mul(u, a));
        const Tensor s = row(ops::mul(v, b));
        const Tensor m = ops::add(ops::div(r, A), ops::div(s, B));
        const Tensor J = ops::sub(ops::div(q, D), ops::mul(c, m));
        const Tensor slope = cosine_loss_slope(c, target, margin_, reduction_);

        variable_list grads;
        if (should_compute_output(0)) {
            const Tensor prod = ops::mul(slope, J);
            grads.push_back(reduction_ == 0 ? prod : ops::sum(prod));
        } else {
            grads.push_back(Tensor());
        }
        const Tensor w = col(ops::mul(slope, grad_output));
        if (should_compute_output(1)) {
            // v / D - q a / (A D) - m (b / D - c a / A) - c (u / A - 2 r a / A^2)
            Tensor t = ops::sub(ops::div(v, col(D)),
                                ops::mul(a, col(ops::div(q, ops::mul(A, D)))));
            t = ops::sub(t, ops::mul(ops::sub(ops::div(b, col(D)),
                                              ops::mul(a, col(ops::div(c, A)))), col(m)));
            t = ops::sub(t, ops::mul(ops::sub(ops::div(u, col(A)),
                                              ops::mul(a, col(ops::div(ops::mul(r, Scalar(2.0)),
                                                                       ops::mul(A, A))))),
                                     col(c)));
            grads.push_back(ops::mul(t, w));
        } else {
            grads.push_back(Tensor());
        }
        if (should_compute_output(2)) {
            Tensor t = ops::sub(ops::div(u, col(D)),
                                ops::mul(b, col(ops::div(q, ops::mul(B, D)))));
            t = ops::sub(t, ops::mul(ops::sub(ops::div(a, col(D)),
                                              ops::mul(b, col(ops::div(c, B)))), col(m)));
            t = ops::sub(t, ops::mul(ops::sub(ops::div(v, col(B)),
                                              ops::mul(b, col(ops::div(ops::mul(s, Scalar(2.0)),
                                                                       ops::mul(B, B))))),
                                     col(c)));
            grads.push_back(ops::mul(t, w));
        } else {
            grads.push_back(Tensor());
        }
        // The target only selects which branch of the loss is taken.
        grads.push_back(should_compute_output(3) ? ops::zeros_like(target) : Tensor());
        return grads;
    }

    void release_variables() override {
        Node::release_variables();
        grad_output_.reset_data();
        input1_.reset_data();
        input2_.reset_data();
        target_.reset_data();
    }
};

}  // namespace tpx
}  // namespace tensorplay
