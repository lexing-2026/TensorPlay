// Composite kernels: rrelu / rrelu_.
// leaky_relu((l+u)/2); training draws uniform slopes and routes through
// rrelu_with_noise (the TP kernel consumes caller-generated noise).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor rrelu_impl(const Tensor& self, const Scalar& lower, const Scalar& upper,
                  bool training, bool check_bounds) {
    const double l = lower.toDouble();
    const double u = upper.toDouble();
    if (check_bounds) {
        if (!std::isfinite(l)) {
            TP_THROW(RuntimeError, "rrelu: lower bound must be finite, got ", l);
        }
        if (!std::isfinite(u)) {
            TP_THROW(RuntimeError, "rrelu: upper bound must be finite, got ", u);
        }
        if (!(l <= u)) {
            TP_THROW(RuntimeError,
                     "Lower bound should be less than or equal to the upper bound");
        }
    }
    Tensor noise = ops::empty_like(self);
    if (training) {
        ops::uniform_(noise, l, u);
        return ops::rrelu_with_noise(self, noise, lower, upper, true);
    }
    return ops::leaky_relu(self, Scalar((l + u) / 2.0));
}

} // anonymous namespace

Tensor rrelu_native(const Tensor& self, const Scalar& lower,
                    const Scalar& upper, bool training) {
    return rrelu_impl(self, lower, upper, training, true);
}

Tensor& rrelu__native(Tensor& self, const Scalar& lower, const Scalar& upper,
                      bool training) {
    Tensor result = rrelu_impl(self, lower, upper, training, false);
    if (result.impl() != self.impl()) ops::copy_(self, result);
    return self;
}

TENSORPLAY_LIBRARY_IMPL(Composite, ActivationComposite) {
    m.impl("rrelu", rrelu_native);
    m.impl("rrelu_", rrelu__native);
}

} // namespace composite
} // namespace tensorplay
