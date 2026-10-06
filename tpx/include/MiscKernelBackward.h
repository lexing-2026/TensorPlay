#pragma once
// Second derivatives through the spectral, integration, scatter/index
// reduction and covariance backward kernels.  The scaled-dot-product
// attention backwards differentiate in SdpaBackward.h.
//
// Under create_graph the first backward pass runs these kernels on tensors
// that belong to a graph.  The formulas next to the kernels' schemas in
// derivatives.yaml use the helpers below; each is written with
// differentiable operations, so a third pass records as well.

#include "Autograd.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {

// The extent of `dim` (which may count from the end) in `sizes`.
inline int64_t size_along(const std::vector<int64_t>& sizes, int64_t dim) {
    const int64_t nd = static_cast<int64_t>(sizes.size());
    return sizes[static_cast<size_t>(dim < 0 ? dim + nd : dim)];
}

// The derivative of a multi-output backward kernel that has none: a pass
// that needs a gradient through the kernel raises, so the missing term is
// never mistaken for zero.  A pass that does not lead through the kernel's
// inputs never asks.
struct UndifferentiatedKernelBackward : public Node {
    UndifferentiatedKernelBackward(size_t outputs, const char* kernel)
        : outputs_(outputs), kernel_(kernel) {}

    size_t num_inputs() const override { return outputs_; }

    variable_list apply(variable_list&& /*inputs*/) override {
        for (size_t i = 0; i < next_edges().size(); ++i) {
            if (should_compute_output(i)) not_implemented_grad(kernel_);
        }
        return variable_list(next_edges().size());
    }

private:
    size_t outputs_;
    const char* kernel_;
};

#define TPX_UNDIFFERENTIATED_KERNEL_NODE(node, outputs, kernel)     \
    struct node : public UndifferentiatedKernelBackward {           \
        node() : UndifferentiatedKernelBackward(outputs, kernel) {} \
    };

TPX_UNDIFFERENTIATED_KERNEL_NODE(DeformConv2dBackwardBackward, 5, "deform_conv2d_backward")
TPX_UNDIFFERENTIATED_KERNEL_NODE(FlashAttentionBackwardBackward, 3, "_flash_attention_backward")
TPX_UNDIFFERENTIATED_KERNEL_NODE(EfficientAttentionBackwardBackward, 4,
                                 "_efficient_attention_backward")

#undef TPX_UNDIFFERENTIATED_KERNEL_NODE

// ---------------------------------------------------------------------------
// A value whose history raises when differentiated
// ---------------------------------------------------------------------------

// Stands between a value and the graph that produced it when the value's own
// derivative is not available for some inputs: the first pass uses the
// value, a pass that differentiates through it raises.
struct DelayedErrorBackward : public Node {
    explicit DelayedErrorBackward(std::string message) : message_(std::move(message)) {}

    variable_list apply(variable_list&& /*inputs*/) override {
        if (should_compute_output(0)) TP_THROW(NotImplementedError, message_);
        return {Tensor()};
    }

private:
    std::string message_;
};

inline Tensor delayed_error(const Tensor& value, std::string message) {
    if (!GradMode::is_enabled() || !value.requires_grad()) return value;
    auto node = std::make_shared<DelayedErrorBackward>(std::move(message));
    node->add_next_edge_list(collect_next_edges(value));
    Tensor out = value.detach();
    impl::set_requires_grad(out, true);
    impl::set_grad_fn(out, std::move(node));
    return out;
}

namespace misc_bwd_detail {

inline int64_t wrap(int64_t dim, int64_t ndim) { return dim < 0 ? dim + ndim : dim; }

inline Tensor as_values(const Tensor& mask, const Tensor& like) {
    return ops::to(mask, like.dtype());
}

}  // namespace misc_bwd_detail

// ---------------------------------------------------------------------------
// scatter_reduce / index_reduce
// ---------------------------------------------------------------------------
//
// The gradients for every reduction, written with differentiable operations.
// `place` scatters values of the source's shape into the destination's shape
// and `pick` gathers them back, so one body serves scatter_reduce (scatter_add
// / gather along `dim` with an index of the source's shape) and index_reduce
// (index_add / index_select with a 1-D index).  The reduced result is
// recomputed through the recorded forward, so derivatives through it reach
// self and the source as well.

template <typename Place, typename Pick, typename Reduce, typename Clear>
std::pair<Tensor, Tensor> reduce_into_grads(const Tensor& grad, const Tensor& self,
                                            const Tensor& src, const std::string& reduce,
                                            bool include_self, Place place, Pick pick,
                                            Reduce reduce_fn, Clear clear, const char* what) {
    using misc_bwd_detail::as_values;
    Tensor grad_self;
    Tensor grad_src;
    if (reduce == "sum") {
        grad_self = grad;
        grad_src = pick(grad);
    } else if (reduce == "prod") {
        const Tensor result = reduce_fn(self, src);
        const Tensor masked_self = ops::masked_fill(self, ops::eq(self, Scalar(0)), Scalar(1));
        grad_self = ops::div(ops::mul(grad, reduce_fn(masked_self, src)), masked_self);
        const Tensor src_zero = ops::eq(src, Scalar(0));
        const Tensor src_num_zeros =
            pick(place(ops::zeros_like(self), as_values(src_zero, self)));
        const Tensor src_single_zero =
            ops::bitwise_and(src_zero, ops::eq(src_num_zeros, Scalar(1)));
        // A source element that is the only zero reaching its slot gets the
        // product of everything else there, which the plain quotient loses.
        const Tensor masked_src = ops::masked_fill(src, src_single_zero, Scalar(1));
        grad_src = ops::where(
            src_single_zero, pick(ops::mul(grad, reduce_fn(self, masked_src))),
            ops::div(pick(ops::mul(grad, result)),
                     ops::masked_fill(src, src_zero, Scalar(1))));
        if (ops::any(ops::gt(src_num_zeros, Scalar(1))).item<bool>()) {
            grad_src = delayed_error(
                grad_src, std::string(what) +
                              ": the second derivative with respect to the source is not "
                              "implemented when more than one zero reaches the same slot");
        }
    } else if (reduce == "mean") {
        Tensor counts = include_self ? ops::ones_like(grad) : ops::zeros_like(grad);
        counts = place(counts, ops::ones_like(src));
        counts = ops::masked_fill(counts, ops::eq(counts, Scalar(0)), Scalar(1));
        grad_self = ops::div(grad, counts);
        grad_src = ops::div(pick(grad), pick(counts));
    } else if (reduce == "amax" || reduce == "amin") {
        // Ties share the gradient evenly.
        const Tensor result = reduce_fn(self, src);
        const Tensor value = pick(result);
        const Tensor self_is_result = as_values(ops::eq(self, result), self);
        const Tensor src_is_result = as_values(ops::eq(src, value), self);
        const Tensor share = ops::div(grad, place(self_is_result, src_is_result));
        grad_self = ops::mul(self_is_result, share);
        grad_src = ops::mul(src_is_result, pick(share));
    } else {
        TP_THROW(ValueError, what, ": expected reduce to be one of sum, prod, mean, amax "
                 "or amin, got ", reduce);
    }
    if (!include_self) grad_self = clear(grad_self);
    return {grad_self, grad_src};
}

inline std::pair<Tensor, Tensor> scatter_reduce_grads(const Tensor& grad, const Tensor& self,
                                                      int64_t dim, const Tensor& index,
                                                      const Tensor& src, const std::string& reduce,
                                                      bool include_self) {
    return reduce_into_grads(
        grad, self, src, reduce, include_self,
        [&](const Tensor& dst, const Tensor& v) { return ops::scatter_add(dst, dim, index, v); },
        [&](const Tensor& t) { return ops::gather(t, dim, index); },
        [&](const Tensor& s, const Tensor& v) {
            return ops::scatter_reduce(s, dim, index, v, reduce, include_self);
        },
        [&](const Tensor& t) { return ops::scatter(t, dim, index, Scalar(0)); },
        "scatter_reduce");
}

inline std::pair<Tensor, Tensor> index_reduce_grads(const Tensor& grad, const Tensor& self,
                                                    int64_t dim, const Tensor& index,
                                                    const Tensor& source, const std::string& reduce,
                                                    bool include_self) {
    return reduce_into_grads(
        grad, self, source, reduce, include_self,
        [&](const Tensor& dst, const Tensor& v) { return ops::index_add(dst, dim, index, v); },
        [&](const Tensor& t) { return ops::index_select(t, dim, index); },
        [&](const Tensor& s, const Tensor& v) {
            return ops::index_reduce(s, dim, index, v, reduce, include_self);
        },
        [&](const Tensor& t) { return ops::index_fill(t, dim, index, Scalar(0)); },
        "index_reduce");
}

// The fused kernels when nothing records; the differentiable composition
// when a pass with create_graph does.
inline Tensor scatter_reduce_backward_self_recordable(
    const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index,
    const Tensor& src, const std::string& reduce, bool include_self) {
    if (!GradMode::is_enabled()) {
        return ops::_scatter_reduce_backward_self(grad, self, dim, index, src, reduce, include_self);
    }
    return scatter_reduce_grads(grad, self, dim, index, src, reduce, include_self).first;
}

inline Tensor scatter_reduce_backward_src_recordable(
    const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index,
    const Tensor& src, const std::string& reduce, bool include_self) {
    if (!GradMode::is_enabled()) {
        return ops::_scatter_reduce_backward_src(grad, self, dim, index, src, reduce, include_self);
    }
    return scatter_reduce_grads(grad, self, dim, index, src, reduce, include_self).second;
}

inline Tensor index_reduce_backward_self_recordable(
    const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index,
    const Tensor& source, const std::string& reduce, bool include_self) {
    if (!GradMode::is_enabled()) {
        return ops::_index_reduce_backward_self(grad, self, dim, index, source, reduce, include_self);
    }
    return index_reduce_grads(grad, self, dim, index, source, reduce, include_self).first;
}

inline Tensor index_reduce_backward_src_recordable(
    const Tensor& grad, const Tensor& self, int64_t dim, const Tensor& index,
    const Tensor& source, const std::string& reduce, bool include_self) {
    if (!GradMode::is_enabled()) {
        return ops::_index_reduce_backward_src(grad, self, dim, index, source, reduce, include_self);
    }
    return index_reduce_grads(grad, self, dim, index, source, reduce, include_self).second;
}

// ---------------------------------------------------------------------------
// trapezoid / cumulative_trapezoid with respect to the sample points
// ---------------------------------------------------------------------------
//
// With spacings s_k = x_{k+1} - x_k the integral is sum_k s_k (y_k + y_{k+1}) / 2,
// so the gradient of each spacing is the incoming gradient times the
// trapezoid's mean height (summed over the outputs that include it, for the
// cumulative form), and point j collects that of spacing j-1 minus that of
// spacing j.  A 1-D x lies along `dim` and collects over the other dims.

namespace misc_bwd_detail {

inline Tensor spacing_to_points(const Tensor& g_seg, const Tensor& x, int64_t d) {
    const Tensor edge = ops::zeros_like(ops::narrow(g_seg, d, 0, 1));
    Tensor gx = ops::sub(ops::cat({edge, g_seg}, d), ops::cat({g_seg, edge}, d));
    if (x.dim() == 1) {
        std::vector<int64_t> others;
        for (int64_t i = 0; i < gx.dim(); ++i) {
            if (i != d) others.push_back(i);
        }
        if (!others.empty()) gx = ops::sum(gx, others, false);
        return gx;
    }
    return ops::sum_to_size(gx, x.sizes());
}

inline Tensor mean_heights(const Tensor& y, int64_t d) {
    const int64_t n = y.size(d);
    return ops::mul(ops::add(ops::narrow(y, d, 0, n - 1), ops::narrow(y, d, 1, n - 1)),
                    Scalar(0.5));
}

}  // namespace misc_bwd_detail

// d <grad, trapezoid(y, x)> / dx.
inline Tensor trapezoid_x_backward(const Tensor& grad, const Tensor& y,
                                   const std::optional<Tensor>& x, int64_t dim) {
    if (!x.has_value() || !x->defined()) return Tensor();
    const int64_t d = misc_bwd_detail::wrap(dim, y.dim());
    if (y.size(d) < 2) return ops::zeros_like(*x);
    const Tensor g_seg = ops::mul(ops::unsqueeze(grad, d), misc_bwd_detail::mean_heights(y, d));
    return misc_bwd_detail::spacing_to_points(g_seg, *x, d);
}

// d <grad, cumulative_trapezoid(y, x)> / dx.
inline Tensor cumulative_trapezoid_x_backward(const Tensor& grad, const Tensor& y,
                                              const std::optional<Tensor>& x, int64_t dim) {
    if (!x.has_value() || !x->defined()) return Tensor();
    const int64_t d = misc_bwd_detail::wrap(dim, y.dim());
    if (y.size(d) < 2) return ops::zeros_like(*x);
    const std::vector<int64_t> along{d};
    const Tensor later = ops::flip(ops::cumsum(ops::flip(grad, along), d), along);
    const Tensor g_seg = ops::mul(later, misc_bwd_detail::mean_heights(y, d));
    return misc_bwd_detail::spacing_to_points(g_seg, *x, d);
}

// ---------------------------------------------------------------------------
// cov / corrcoef
// ---------------------------------------------------------------------------
//
// cov(x) = xc diag(w) xc^T / N with w = fweights * aweights, xc = x minus its
// w-weighted mean and N = sum(w) - correction * sum(w * aweights) / sum(w)
// (sum(w) - correction without aweights).  The mean's own variation drops out
// because sum_j w_j xc_j = 0, leaving dx = (G + G^T) xc diag(w) / N.

namespace misc_bwd_detail {

struct CovParts {
    Tensor x2;      // observations as columns, 2-D
    Tensor w;       // per-observation weights (undefined when unweighted)
    Tensor centered;
    Tensor norm;    // N, a 0-d tensor
    Tensor aw;      // aweights in the input's dtype (undefined when absent)
};

inline CovParts cov_parts(const Tensor& self, int64_t correction,
                          const std::optional<Tensor>& fweights,
                          const std::optional<Tensor>& aweights) {
    CovParts p;
    p.x2 = self.dim() < 2 ? ops::reshape(self, {1, -1}) : self;
    const bool has_f = fweights.has_value() && fweights->defined();
    const bool has_a = aweights.has_value() && aweights->defined();
    if (has_f) p.w = ops::to(*fweights, p.x2.dtype());
    if (has_a) {
        // Recorded even across a dtype change, so the weights' own gradient
        // flows back.
        p.aw = tpx::to(*aweights, p.x2.dtype());
        p.w = p.w.defined() ? ops::mul(p.w, p.aw) : p.aw;
    }
    const std::vector<int64_t> obs{1};
    Tensor mean;
    Tensor w_sum;
    if (p.w.defined()) {
        w_sum = ops::sum(p.w);
        mean = ops::div(ops::sum(ops::mul(p.x2, p.w), obs, true), w_sum);
    } else {
        w_sum = ops::scalar_tensor(Scalar(p.x2.size(1)), p.x2.dtype(), p.x2.device());
        mean = ops::mean(p.x2, obs, true);
    }
    p.centered = ops::sub(p.x2, mean);
    if (has_a && correction != 0) {
        p.norm = ops::sub(w_sum, ops::div(ops::mul(ops::sum(ops::mul(p.w, p.aw)),
                                                   Scalar(correction)),
                                          w_sum));
    } else {
        p.norm = ops::sub(w_sum, Scalar(correction));
    }
    return p;
}

// G of the 2-D covariance from the incoming gradient (0-d for one variable).
inline Tensor cov_grad_2d(const Tensor& grad) {
    return grad.dim() < 2 ? ops::reshape(grad, {1, 1}) : grad;
}

}  // namespace misc_bwd_detail

inline Tensor cov_backward_composite(const Tensor& grad, const Tensor& self, int64_t correction,
                                     const std::optional<Tensor>& fweights,
                                     const std::optional<Tensor>& aweights) {
    const auto p = misc_bwd_detail::cov_parts(self, correction, fweights, aweights);
    const Tensor g = misc_bwd_detail::cov_grad_2d(grad);
    const Tensor sym = ops::add(g, ops::transpose(g, 0, 1));
    Tensor weighted = p.w.defined() ? ops::mul(p.centered, p.w) : p.centered;
    Tensor gx = ops::div(ops::matmul(sym, weighted), p.norm);
    return self.dim() < 2 ? ops::reshape(gx, self.sizes()) : gx;
}

inline Tensor cov_backward_recordable(const Tensor& grad, const Tensor& self, int64_t correction,
                                      const std::optional<Tensor>& fweights,
                                      const std::optional<Tensor>& aweights) {
    if (!GradMode::is_enabled()) {
        return ops::_cov_backward(grad, self, correction, fweights, aweights);
    }
    return cov_backward_composite(grad, self, correction, fweights, aweights);
}

// d <G, cov> / d aweights_k with L = <G, cov>, fw the frequency weights,
// w = fw * aw, s = sum(w), q = sum(w * aw) and c the correction:
//   fw_k xc_k^T G xc_k / N - (L / N) (fw_k - 2 c w_k / s + c fw_k q / s^2).
inline Tensor cov_aweights_backward(const Tensor& grad, const Tensor& self, int64_t correction,
                                    const std::optional<Tensor>& fweights,
                                    const std::optional<Tensor>& aweights) {
    if (!aweights.has_value() || !aweights->defined()) return Tensor();
    const auto p = misc_bwd_detail::cov_parts(self, correction, fweights, aweights);
    const Tensor g = misc_bwd_detail::cov_grad_2d(grad);
    const Tensor& aw = p.aw;
    const Tensor fw = (fweights.has_value() && fweights->defined())
                          ? ops::to(*fweights, aw.dtype())
                          : ops::ones_like(aw);
    const std::vector<int64_t> vars{0};
    // xc_k^T G xc_k for every observation k.
    const Tensor quad = ops::sum(ops::mul(p.centered, ops::matmul(g, p.centered)), vars, false);
    const Tensor cov2 = ops::div(ops::matmul(ops::mul(p.centered, p.w),
                                             ops::transpose(p.centered, 0, 1)),
                                 p.norm);
    const Tensor loss = ops::sum(ops::mul(g, cov2));
    const Tensor s = ops::sum(p.w);
    const Tensor q = ops::sum(ops::mul(p.w, aw));
    Tensor dnorm = fw;
    if (correction != 0) {
        dnorm = ops::add(ops::sub(fw, ops::div(ops::mul(p.w, Scalar(2 * correction)), s)),
                         ops::div(ops::mul(ops::mul(fw, q), Scalar(correction)), ops::mul(s, s)));
    }
    return ops::sub(ops::div(ops::mul(fw, quad), p.norm),
                    ops::mul(ops::div(loss, p.norm), dnorm));
}

// corrcoef(x) = C / (s s^T) with s = sqrt(diag C), clipped to [-1, 1]:
//   dC = G' / (s s^T) - diag((rowsum(G' R) + colsum(G' R)) / (2 C_ii)),
// where G' keeps the entries the clip let through.
inline Tensor corrcoef_backward_composite(const Tensor& grad, const Tensor& self) {
    const Tensor c = ops::cov(self);
    if (c.dim() < 2) {
        // One variable: the coefficient is the constant c / c.
        return ops::zeros_like(self);
    }
    const Tensor diag = ops::diagonal(c, 0, 0, 1);
    const Tensor s = ops::sqrt(diag);
    const Tensor outer = ops::mul(ops::unsqueeze(s, 1), ops::unsqueeze(s, 0));
    const Tensor r = ops::div(c, outer);
    const Tensor kept = ops::logical_and(ops::ge(r, Scalar(-1)), ops::le(r, Scalar(1)));
    const Tensor g = ops::mul(grad, misc_bwd_detail::as_values(kept, grad));
    const Tensor gr = ops::mul(g, r);
    const Tensor shared = ops::div(ops::add(ops::sum(gr, std::vector<int64_t>{1}, false),
                                            ops::sum(gr, std::vector<int64_t>{0}, false)),
                                   ops::mul(diag, Scalar(2)));
    const Tensor grad_c = ops::sub(ops::div(g, outer), ops::diag_embed(shared, 0, 0, 1));
    return cov_backward_composite(grad_c, self, 1, std::nullopt, std::nullopt);
}

inline Tensor corrcoef_backward_recordable(const Tensor& grad, const Tensor& self) {
    if (!GradMode::is_enabled()) return ops::_corrcoef_backward(grad, self);
    return corrcoef_backward_composite(grad, self);
}

// ---------------------------------------------------------------------------
// The primitive transforms
// ---------------------------------------------------------------------------

// The one-sided real-to-complex transform is the complex transform of the
// real signal with the upper half of the last transformed dimension dropped.
// Its adjoint pads that half back with zeros, runs the complex transform of
// the other direction (same scaling) and keeps the real part.
inline Tensor fft_r2c_backward(const Tensor& grad, const std::vector<int64_t>& dim,
                               int64_t normalization, bool onesided,
                               int64_t last_dim_size) {
    Tensor full = grad;
    if (onesided) {
        const int64_t ndim = grad.dim();
        const int64_t last = misc_bwd_detail::wrap(dim.back(), ndim);
        const int64_t missing = last_dim_size - grad.size(last);
        if (missing > 0) {
            std::vector<int64_t> pad(static_cast<size_t>(2 * (ndim - last)), 0);
            pad.back() = missing;
            full = ops::constant_pad_nd(grad, pad, 0);
        }
    }
    return ops::real(ops::_fft_c2c(full, dim, normalization, /*forward=*/false));
}

// The complex-to-real transform reads a one-sided spectrum and fills the
// rest by conjugate symmetry.  Its adjoint is the one-sided real-to-complex
// transform, with every entry that also stands for its mirrored partner
// (indices 1 .. n - (n / 2 + 1) along the last dimension) counted twice.
inline Tensor fft_c2r_backward(const Tensor& grad, const std::vector<int64_t>& dim,
                               int64_t normalization) {
    Tensor result = ops::_fft_r2c(grad, dim, normalization, /*onesided=*/true);
    const int64_t ndim = grad.dim();
    const int64_t last = misc_bwd_detail::wrap(dim.back(), ndim);
    const int64_t half = result.size(last);
    const int64_t doubled = grad.size(last) - half;
    if (doubled <= 0) return result;
    Tensor weight = ops::ones({half}, grad.dtype(), grad.device());
    Tensor mirrored = ops::narrow(weight, 0, 1, doubled);
    ops::fill_(mirrored, 2);
    std::vector<int64_t> shape(static_cast<size_t>(ndim), 1);
    shape[static_cast<size_t>(last)] = half;
    return ops::mul(result, ops::view(weight, shape));
}

// ---------------------------------------------------------------------------
// stft with respect to the window
// ---------------------------------------------------------------------------

// d <grad, stft(self, window)> / d window: the adjoint of each frame's
// transform applied to the incoming gradient, times the (padded) frame it
// came from, summed over frames and the batch, then cropped to the window's
// own length (stft centers a shorter window inside n_fft).  Written with
// differentiable operations, so it records under create_graph.
inline Tensor stft_window_backward(const Tensor& grad, const Tensor& self, int64_t n_fft,
                                   std::optional<int64_t> hop_length,
                                   std::optional<int64_t> win_length,
                                   const std::optional<Tensor>& window, bool center,
                                   const std::string& pad_mode, bool normalized, bool onesided) {
    if (!window.has_value() || !window->defined()) return Tensor();
    const int64_t hop = hop_length.value_or(n_fft >> 2);
    const int64_t win = win_length.value_or(n_fft);
    Tensor x = self.dim() == 1 ? ops::unsqueeze(self, 0) : self;
    const Tensor g = grad.dim() == 2 ? ops::unsqueeze(grad, 0) : grad;  // (batch, freq, frames)
    if (center) {
        const int64_t p = n_fft / 2;
        x = ops::pad(x, {p, p}, pad_mode);
    }
    const Tensor frames = ops::unfold(x, 1, n_fft, hop);  // (batch, frames, n_fft)
    const Tensor per_frame = ops::transpose(g, 1, 2);     // (batch, frames, freq)
    const std::string norm = normalized ? "ortho" : "backward";
    const Tensor time_grad = onesided
        ? ops::fft_rfft_backward(per_frame, frames, 2, norm, n_fft)
        : ops::fft_fft_backward(per_frame, frames, 2, norm);
    Tensor gw = ops::sum(ops::mul(time_grad, frames), std::vector<int64_t>{0, 1}, false);
    if (win < n_fft) gw = ops::narrow(gw, 0, (n_fft - win) / 2, win);
    return tpx::to(gw, window->dtype());
}

}  // namespace tpx
}  // namespace tensorplay
