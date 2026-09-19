// Composite kernels: rrelu / rrelu_.
// Both draw into a fresh noise buffer through rrelu_with_noise, which applies
// uniform slopes in training and the midpoint slope otherwise.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_rrelu_bounds(const Scalar& lower, const Scalar& upper, bool check_finite) {
    const double l = lower.toDouble();
    const double u = upper.toDouble();
    if (check_finite) {
        if (!std::isfinite(l)) {
            TP_THROW(RuntimeError, "rrelu: lower bound must be finite, got ", l);
        }
        if (!std::isfinite(u)) {
            TP_THROW(RuntimeError, "rrelu: upper bound must be finite, got ", u);
        }
    }
    if (!(l <= u)) {
        TP_THROW(RuntimeError,
                 "Lower bound should be less than or equal to the upper bound");
    }
}

} // anonymous namespace

Tensor rrelu_native(const Tensor& self, const Scalar& lower, const Scalar& upper,
                    bool training, std::optional<Generator> generator) {
    check_rrelu_bounds(lower, upper, true);
    Tensor noise = ops::empty_like(self);
    return ops::rrelu_with_noise(self, noise, lower, upper, training, generator);
}

Tensor& rrelu__native(Tensor& self, const Scalar& lower, const Scalar& upper,
                      bool training, std::optional<Generator> generator) {
    check_rrelu_bounds(lower, upper, false);
    Tensor noise = ops::empty_like(self);
    ops::rrelu_with_noise_(self, noise, lower, upper, training, generator);
    return self;
}

TENSORPLAY_LIBRARY_IMPL(Composite, ActivationComposite) {
    m.impl("rrelu", rrelu_native);
    m.impl("rrelu_", rrelu__native);
}

} // namespace composite
} // namespace tensorplay
