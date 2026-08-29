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

Tensor cdist_native(const Tensor& x1, const Tensor& x2, double p) {
    if (x1.dim() != 2 || x2.dim() != 2) {
        TP_THROW(NotImplementedError,
                 "cdist(): composite kernel expects 2-D inputs, got ",
                 x1.dim(), " and ", x2.dim(), " dims");
    }
    const DType common = promoteTypes(x1.dtype(), x2.dtype());
    Tensor a = x1.dtype() == common ? x1 : x1.to(common);
    Tensor b = x2.dtype() == common ? x2 : x2.to(common);
    const Tensor diff = ops::sub(ops::unsqueeze(a, 1), ops::unsqueeze(b, 0));
    if (p == 2) {
        return ops::sqrt(ops::sum(ops::mul(diff, diff), {-1}, false));
    }
    if (p == 1) {
        return ops::sum(ops::abs(diff), {-1}, false);
    }
    if (p == 0) {
        return ops::sum(ops::ne(diff, Scalar(0)).to(common), {-1}, false);
    }
    if (p == std::numeric_limits<double>::infinity()) {
        return ops::amax(ops::abs(diff), {-1}, false);
    }
    TP_THROW(NotImplementedError,
             "cdist(): composite kernel supports p in {0, 1, 2, inf}, got ",
             p);
}

TENSORPLAY_LIBRARY_IMPL(Composite, DistanceComposite) {
    m.impl("cosine_similarity", cosine_similarity_native);
    m.impl("cdist", cdist_native);
}

} // namespace composite
} // namespace tensorplay
