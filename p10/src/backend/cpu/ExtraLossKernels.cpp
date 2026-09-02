// Mean-reduced losses kept outside the TensorIterator family:
// binary_cross_entropy_with_logits (stable softplus form),
// hinge_embedding_loss, margin_ranking_loss, l1_loss, kl_div,
// cosine_embedding_loss, soft_margin_loss, triplet_margin_loss,
// poisson_nll_loss, multilabel_soft_margin_loss.
//
// multi_margin_loss / multilabel_margin_loss live in MarginLossKernels.cpp;
// ctc_loss / gaussian_nll_loss intentionally omitted (heavy dispatch
// surface) — noted for a later pass.

#include "Tensor.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"

#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace tensorplay {
namespace cpu {

namespace {

inline double softplus(double y) { return std::max(y, 0.0) + std::log1p(std::exp(-std::fabs(y))); }

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline std::pair<Tensor, Tensor> bcast2(const Tensor& a, const Tensor& b) {
    auto shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(DType::Float64),
            b.expand(shape).contiguous().to(DType::Float64)};
}

inline Tensor scalar_from(double v, DType dt, const Device& dev) {
    return Tensor::full({}, Scalar(v),
                        dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
}

} // anonymous namespace

Tensor binary_cross_entropy_with_logits_cpu(const Tensor& self, const Tensor& target,
                                            const std::optional<Tensor>& weight_opt,
                                            const std::optional<Tensor>& pos_weight_opt) {
    // Stable form: l = w*(pw*t*softplus(-x) + (1-t)*softplus(x)) with
    // softplus(y) = max(y,0) + log1p(exp(-|y|)).
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    Tensor x = self.contiguous().to(DType::Float64);
    Tensor t = target.contiguous().to(DType::Float64).expand(shape_of(x)).contiguous();
    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    Tensor w = has_w ? weight.to(DType::Float64).expand(shape_of(x)).contiguous()
                     : Tensor::zeros({}, DType::Float64, Device(DeviceType::CPU));
    Tensor pw = has_pw ? pos_weight.to(DType::Float64).expand(shape_of(x)).contiguous()
                       : Tensor::zeros({}, DType::Float64, Device(DeviceType::CPU));
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    const double* wp = has_w ? w.data_ptr<double>() : nullptr;
    const double* pwp = has_pw ? pw.data_ptr<double>() : nullptr;
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double xv = xp[i], tv = tp[i];
        double wi = wp ? wp[i] : 1.0;
        double pi = pwp ? pwp[i] : 1.0;
        total += wi * (pi * tv * softplus(-xv) + (1.0 - tv) * softplus(xv));
    }
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, self.device());
}

Tensor hinge_embedding_loss_cpu(const Tensor& input, const Tensor& target, Scalar margin) {
    // target == 1 -> x ; else relu(margin - x); mean.
    Tensor x = input.contiguous().to(DType::Float64);
    Tensor t = target.contiguous().to(DType::Float64).expand(shape_of(x)).contiguous();
    double mg = margin.toDouble();
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += (tp[i] == 1.0) ? xp[i] : std::max(0.0, mg - xp[i]);
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, input.device());
}

Tensor margin_ranking_loss_cpu(const Tensor& input1, const Tensor& input2,
                               const Tensor& target, Scalar margin) {
    // mean(relu(margin - target*(x1 - x2)))
    Tensor a = input1.contiguous().to(DType::Float64);
    Tensor b = input2.contiguous().to(DType::Float64).expand(shape_of(a)).contiguous();
    Tensor tg = target.contiguous().to(DType::Float64).expand(shape_of(a)).contiguous();
    double mg = margin.toDouble();
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    const double* gp = tg.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += std::max(0.0, mg - gp[i] * (ap[i] - bp[i]));
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = input1.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, input1.device());
}

Tensor l1_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor a = pr.first, b = pr.second;
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) total += std::fabs(ap[i] - bp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor kl_div_cpu(const Tensor& input, const Tensor& target) {
    // input log-probs; target probs; mean(t*(log t - x)).
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        if (tp[i] > 0) total += tp[i] * (std::log(tp[i]) - xp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor cosine_embedding_loss_cpu(const Tensor& x1, const Tensor& x2, const Tensor& target,
                                 Scalar margin) {
    Tensor a = x1.contiguous().to(DType::Float64);
    Tensor b = x2.contiguous().to(DType::Float64);
    Tensor tg = target.contiguous().to(DType::Float64);
    int64_t N = a.size(0), D = a.size(1);
    if (tg.dim() == 0) tg = tg.expand({N}).contiguous();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    const double* gp = tg.data_ptr<double>();
    double mg = margin.toDouble();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        double dot = 0, na = 0, nbv = 0;
        for (int64_t j = 0; j < D; ++j) {
            dot += ap[i * D + j] * bp[i * D + j];
            na += ap[i * D + j] * ap[i * D + j];
            nbv += bp[i * D + j] * bp[i * D + j];
        }
        double cosv = dot / (std::sqrt(na) * std::sqrt(nbv) + 1e-12);
        total += (gp[i] == 1.0) ? 1.0 - cosv : std::max(0.0, cosv - mg);
    }
    return scalar_from(N ? total / N : 0.0, x1.dtype(), x1.device());
}

Tensor soft_margin_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += softplus(-tp[i] * xp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor triplet_margin_loss_cpu(const Tensor& anchor, const Tensor& positive,
                               const Tensor& negative, Scalar margin, double p) {
    int64_t N = anchor.size(0), D = anchor.size(1);
    Tensor a = anchor.contiguous().to(DType::Float64);
    Tensor pp2 = positive.contiguous().to(DType::Float64);
    Tensor nn2 = negative.contiguous().to(DType::Float64);
    const double* ap = a.data_ptr<double>();
    const double* ppos = pp2.data_ptr<double>();
    const double* pneg = nn2.data_ptr<double>();
    double mg = margin.toDouble();
    auto dist = [&](const double* u, const double* v) {
        if (p == std::numeric_limits<double>::infinity()) {
            double mx = 0;
            for (int64_t j = 0; j < D; ++j) mx = std::max(mx, std::fabs(u[j] - v[j]));
            return mx;
        }
        double s2 = 0;
        for (int64_t j = 0; j < D; ++j) s2 += std::pow(std::fabs(u[j] - v[j]), p);
        return std::pow(s2, 1.0 / p);
    };
    double total = 0;
    for (int64_t i = 0; i < N; ++i)
        total += std::max(0.0, dist(ap + i * D, ppos + i * D) -
                               dist(ap + i * D, pneg + i * D) + mg);
    return scalar_from(N ? total / N : 0.0, anchor.dtype(), anchor.device());
}

Tensor poisson_nll_loss_cpu(const Tensor& input, const Tensor& target, bool log_input,
                            bool full, double eps) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, z = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* zp = z.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double xv = xp[i], zv = zp[i];
        double l2 = log_input ? (std::exp(xv) - zv * xv)
                              : (xv - zv * std::log(std::exp(xv) + eps));
        if (full && zv > 0) l2 += zv * std::log(zv) - std::lgamma(zv + 1.0);
        total += l2;
    }
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor multilabel_soft_margin_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t N = x.size(0), C = x.size(1);
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        double row = 0;
        for (int64_t c = 0; c < C; ++c) {
            double xv = xp[i * C + c], tv = tp[i * C + c];
            row += tv * -softplus(-xv) + (1.0 - tv) * -softplus(xv);
        }
        total += -row / C;
    }
    return scalar_from(N ? total / N : 0.0, input.dtype(), input.device());
}

TENSORPLAY_LIBRARY_IMPL(CPU, ExtraLosses) {
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cpu);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_cpu);
    m.impl("margin_ranking_loss", margin_ranking_loss_cpu);
    m.impl("l1_loss", l1_loss_cpu);
    m.impl("kl_div", kl_div_cpu);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_cpu);
    m.impl("soft_margin_loss", soft_margin_loss_cpu);
    m.impl("triplet_margin_loss", triplet_margin_loss_cpu);
    m.impl("poisson_nll_loss", poisson_nll_loss_cpu);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
