// Composite kernels: trapz.x / trapz.dx.
// trapezoid.

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor trapz_x_native(const Tensor& y, const Tensor& x, int64_t dim) {
    return ops::trapezoid(y, std::optional<Tensor>(x), Scalar(1), dim);
}

Tensor trapz_dx_native(const Tensor& y, double dx, int64_t dim) {
    return ops::trapezoid(y, std::nullopt, Scalar(dx), dim);
}

TENSORPLAY_LIBRARY_IMPL(Composite, IntegrationComposite) {
    m.impl("trapz.x", trapz_x_native);
    m.impl("trapz.dx", trapz_dx_native);
}

} // namespace composite
} // namespace tensorplay
