// Composite kernel: histc.
// assignment via bucketize, counting via bincount.  Boundary semantics match
// bin closed on the right, NaNs and out-of-range values skipped.

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
        lo = std::min(std::nextafter(lo, -std::numeric_limits<double>::infinity()), lo - 1.0);
        hi = std::max(std::nextafter(hi, std::numeric_limits<double>::infinity()), hi + 1.0);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) {
        TP_THROW(RuntimeError, "histc: max must be larger than min");
    }

    const Tensor flat = ops::reshape(self, {-1});
    const Tensor x64 = flat.to(DType::Float64);
    const Tensor edges = ops::linspace(Scalar(lo), Scalar(hi), bins + 1,
                                       DType::Float64, self.device());
    // right=true: number of edges <= x; subtract one for the bin index.
    Tensor idx = ops::sub(ops::bucketize(x64, edges, false, true), Scalar(int64_t(1)));
    idx = ops::clamp(idx, Scalar(int64_t(0)), Scalar(bins - 1));
    const Tensor in_range = ops::logical_and(ops::ge(x64, Scalar(lo)),
                                             ops::le(x64, Scalar(hi)));
    const Tensor counted = ops::masked_select(idx, in_range);
    const Tensor counts = ops::bincount(counted, std::nullopt, bins);
    return counts.to(self.dtype());
}

TENSORPLAY_LIBRARY_IMPL(Composite, HistogramComposite) {
    m.impl("histc", histc_native);
}

} // namespace composite
} // namespace tensorplay
