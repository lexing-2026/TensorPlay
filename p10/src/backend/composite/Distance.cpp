// Composite kernel: cosine_similarity.
// (clamped at eps) then dot.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

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
    Tensor a = x1.dtype() == common ? x1 : x1.to(common);
    Tensor b = x2.dtype() == common ? x2 : x2.to(common);
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
    Tensor a = x1.dtype() == common ? x1 : x1.to(common);
    Tensor b = x2.dtype() == common ? x2 : x2.to(common);
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

TENSORPLAY_LIBRARY_IMPL(Composite, DistanceComposite) {
    m.impl("cosine_similarity", cosine_similarity_native);
    m.impl("cdist", cdist_native);
}

} // namespace composite
} // namespace tensorplay
