// Composite kernel: histc.
// Linear bin mapping with bincount accumulation.  Boundary semantics keep the
// rightmost bin closed, while NaNs and out-of-range values are skipped.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void histc_expand_constant_range(DType dtype, double& lo, double& hi) {
    switch (dtype) {
        case DType::Float64:
            lo = std::min(
                std::nexttoward(lo, std::numeric_limits<double>::lowest()),
                lo - 1.0);
            hi = std::max(
                std::nexttoward(hi, std::numeric_limits<double>::max()),
                hi + 1.0);
            break;
        case DType::Float32:
            lo = std::min(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(lo),
                    std::numeric_limits<float>::lowest())),
                lo - 1.0);
            hi = std::max(
                static_cast<double>(std::nexttoward(
                    static_cast<float>(hi),
                    std::numeric_limits<float>::max())),
                hi + 1.0);
            break;
        default:
            lo -= 1.0;
            hi += 1.0;
            break;
    }
}

} // anonymous namespace

Tensor histc_native(const Tensor& self, int64_t bins, const Scalar& min,
                    const Scalar& max) {
    if (bins <= 0) TP_THROW(RuntimeError, "histc(): bins must be positive");
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto extrema = ops::aminmax(self);
        lo = std::get<0>(extrema).item().toDouble();
        hi = std::get<1>(extrema).item().toDouble();
    }
    if (lo == hi) {
        histc_expand_constant_range(self.dtype(), lo, hi);
        histc_expand_constant_range(self.dtype(), lo, hi);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) {
        TP_THROW(RuntimeError, "histc: max must be larger than min");
    }

    const Tensor flat = ops::reshape(self, {-1});
    const Tensor in_range = ops::logical_and(ops::ge(flat, Scalar(lo)),
                                             ops::le(flat, Scalar(hi)));
    const Tensor safe = Tensor::where(in_range, flat, Tensor::zeros_like(flat));
    Tensor idx = ops::div(
        ops::mul(ops::sub(safe, Scalar(lo)), Scalar(bins)),
        Scalar(hi - lo)).to(DType::Int64);
    idx = ops::clamp(idx, Scalar(int64_t(0)), Scalar(bins - 1));
    const Tensor counted = ops::masked_select(idx, in_range);
    const Tensor counts = ops::bincount(counted, std::nullopt, bins);
    return counts.to(self.dtype());
}

TENSORPLAY_LIBRARY_IMPL(Composite, HistogramComposite) {
    m.impl("histc", histc_native);
}

} // namespace composite
} // namespace tensorplay
