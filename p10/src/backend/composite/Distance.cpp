// Composite kernel: cosine_similarity.
// (clamped at eps) then dot.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Autograd.h"

#include <cstdint>
#include <limits>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor cosine_similarity_native(const Tensor& x1, const Tensor& x2,
                                int64_t dim, double eps) {
    const DType common = promoteTypes(x1.dtype(), x2.dtype());
    if (!isFloatingType(common)) {
        TP_THROW(RuntimeError,
                 "expected common dtype to be floating point, yet common dtype is ",
                 toString(common));
    }
    if (!(eps >= 0)) {
        TP_THROW(RuntimeError, "eps must be non-negative, got: ", eps);
    }
    // The conversions record themselves, so a gradient reaches each operand
    // in its own type.
    Tensor a = tpx::to(x1, common);
    Tensor b = tpx::to(x2, common);
    const Tensor n1 = ops::clamp_min(ops::norm(a, {dim}, 2.0, true), Scalar(eps));
    const Tensor n2 = ops::clamp_min(ops::norm(b, {dim}, 2.0, true), Scalar(eps));
    return ops::sum(ops::mul(ops::div(a, n1), ops::div(b, n2)), {dim}, false);
}

// Pairwise p-norm distances.  2-D (N, D) x (M, D) and batched 3-D
// (B, N, D) x (B, M, D) inputs are supported; the pairwise difference tensor
// is (B, N, M, D) and the norm reduces over the last axis.  p in {0, 1, 2,
// inf} takes direct reductions, any other positive p composes
// sum(|d|^p)^(1/p).  compute_mode only selects between mathematically
// equivalent evaluation orders for p == 2 and is accepted for signature
// compatibility.
Tensor cdist_native(const Tensor& x1, const Tensor& x2, double p,
                    std::optional<int64_t> /*compute_mode*/) {
    if (x1.dim() < 2 || x1.dim() != x2.dim() || x1.dim() > 3) {
        TP_THROW(RuntimeError,
                 "cdist(): expects 2-D or matching-batch 3-D inputs, got ",
                 x1.dim(), " and ", x2.dim(), " dims");
    }
    const DType common = promoteTypes(x1.dtype(), x2.dtype());
    if (!isFloatingType(common)) {
        TP_THROW(RuntimeError,
                 "cdist(): expected floating-point inputs, got ",
                 toString(common));
    }
    // The conversions record themselves, so a gradient reaches each operand
    // in its own type.
    Tensor a = tpx::to(x1, common);
    Tensor b = tpx::to(x2, common);
    const bool batched = a.dim() == 3;
    if (!batched) {
        a = ops::unsqueeze(a, 0);
        b = ops::unsqueeze(b, 0);
    }
    const Tensor diff = ops::sub(ops::unsqueeze(a, 2), ops::unsqueeze(b, 1));
    Tensor d;
    if (p == 2) {
        d = ops::sqrt(ops::sum(ops::mul(diff, diff), {-1}, false));
    } else if (p == 1) {
        d = ops::sum(ops::abs(diff), {-1}, false);
    } else if (p == 0) {
        d = ops::sum(ops::ne(diff, Scalar(0)).to(common), {-1}, false);
    } else if (p == std::numeric_limits<double>::infinity()) {
        d = ops::amax(ops::abs(diff), {-1}, false);
    } else if (p > 0) {
        d = ops::pow(ops::sum(ops::pow(ops::abs(diff), Scalar(p)), {-1}, false),
                     Scalar(1.0 / p));
    } else {
        TP_THROW(NotImplementedError,
                 "cdist(): composite kernel supports p in [0, inf], got ", p);
    }
    return batched ? d : ops::squeeze(d, 0);
}

namespace {

// How each p-norm distance moves with the first point of its pair, per
// coordinate: sign(diff) |diff|^(p-1) / dist^(p-1), with p = 0, 1, 2 and
// infinity written out.  A zero distance passes nothing back.
Tensor distance_slope(const Tensor& diff, const Tensor& dist, double p) {
    if (p == 0) return ops::zeros_like(diff);
    if (p == 1) return ops::sign(diff);
    const Tensor at_zero = ops::eq(dist, Scalar(0));
    const Tensor none = ops::zeros_like(diff);
    if (p == 2) return ops::where(at_zero, none, ops::div(diff, dist));
    if (p == std::numeric_limits<double>::infinity()) {
        return ops::mul(ops::sign(diff), ops::eq(ops::abs(diff), dist).to(diff.dtype()));
    }
    const Tensor slope = ops::div(
        ops::mul(ops::sign(diff), ops::pow(ops::abs(diff), Scalar(p - 1))),
        ops::pow(dist, Scalar(p - 1)));
    return ops::where(at_zero, none, slope);
}

} // namespace

// The gradient of cdist with respect to x1: each row of x1 gathers the slopes
// of its distances to every row of x2, weighted by the gradient arriving for
// each distance.  (The gradient for x2 is the same question asked the other
// way round.)
Tensor _cdist_backward_native(const Tensor& grad, const Tensor& x1, const Tensor& x2,
                              double p, const Tensor& cdist) {
    const Tensor diff = ops::sub(ops::unsqueeze(x1, -2), ops::unsqueeze(x2, -3));
    const Tensor slope = distance_slope(diff, ops::unsqueeze(cdist, -1), p);
    return ops::sum(ops::mul(slope, ops::unsqueeze(grad, -1)), {-2}, false);
}

// pdist lists the distances of row pairs (i, j), i < j, in row order; both
// rows of a pair move with it, so the condensed gradient and distances are
// laid out as symmetric matrices and differentiated as cdist of the rows
// against themselves.
Tensor _pdist_backward_native(const Tensor& grad, const Tensor& self, double p,
                              const Tensor& pdist) {
    const int64_t n = self.size(0);
    if (n < 2 || grad.numel() == 0) return ops::zeros_like(self);
    const Tensor pairs = ops::triu_indices(n, n, 1, DType::Int64, self.device());
    const std::vector<std::optional<Tensor>> at = {ops::select(pairs, 0, 0),
                                                   ops::select(pairs, 0, 1)};
    const Tensor blank = ops::zeros({n, n}, grad.dtype(), grad.device());
    const Tensor g = ops::index_put(blank, at, grad);
    const Tensor d = ops::index_put(blank, at, pdist);
    return _cdist_backward_native(ops::add(g, ops::transpose(g, 0, 1)), self, self, p,
                                  ops::add(d, ops::transpose(d, 0, 1)));
}

TENSORPLAY_LIBRARY_IMPL(Composite, DistanceComposite) {
    m.impl("_cdist_backward", _cdist_backward_native);
    m.impl("_pdist_backward", _pdist_backward_native);
    m.impl("cosine_similarity", cosine_similarity_native);
    m.impl("cdist", cdist_native);
}

} // namespace composite
} // namespace tensorplay
