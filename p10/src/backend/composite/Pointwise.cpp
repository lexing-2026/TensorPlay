// Composite kernels: numerically-special elementwise ops -- xlogy, ldexp,
// fmax/fmin, float_power, mvlgamma, conj_physical and the negative_ alias.
//
//   xlogy:      x * log(y), with the x == 0 branch collapsing every y
//               (including 0 / inf / nan) to 0.
//   ldexp:      x * 2^y; the exponent is evaluated in the input dtype so the
//               power of two is exact.
//   fmax/fmin:  maximum/minimum with NaN treated as missing -- whenever one
//               operand is NaN the other is returned verbatim.
//   float_power: exponentiation evaluated in Float64, cast back to the
//               natural result type (integral inputs promote to Float32).
//   mvlgamma:   sum_{i=0..p-1} lgamma(x - i/2), elementwise.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor xlogy_impl(const Tensor& self, const Tensor& other) {
    return ops::where(ops::eq(self, Scalar(0)), Scalar(0),
                      ops::mul(self, ops::log(other)));
}

Tensor fmaxmin_impl(const Tensor& self, const Tensor& other, bool max) {
    const Tensor nan_self = ops::isnan(self);
    const Tensor nan_other = ops::isnan(other);
    const Tensor base = max ? ops::maximum(self, other)
                            : ops::minimum(self, other);
    return ops::where(nan_self, other,
                      ops::where(nan_other, self, base));
}

} // anonymous namespace

Tensor xlogy_native(const Tensor& self, const Tensor& other) {
    return xlogy_impl(self, other);
}

Tensor& xlogy__native(Tensor& self, const Tensor& other) {
    ops::copy_(self, xlogy_impl(self, other));
    return self;
}

Tensor ldexp_native(const Tensor& self, const Tensor& other) {
    return ops::mul(self, ops::exp2(other.to(self.dtype())));
}

Tensor& ldexp__native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ldexp_native(self, other));
    return self;
}

Tensor fmax_native(const Tensor& self, const Tensor& other) {
    return fmaxmin_impl(self, other, true);
}

Tensor fmin_native(const Tensor& self, const Tensor& other) {
    return fmaxmin_impl(self, other, false);
}

Tensor float_power_native(const Tensor& self, const Tensor& exponent) {
    const DType rt = ops::result_type(self, exponent);
    const DType out = (isFloatingType(rt) || isComplexType(rt)) ? rt
                                                                : DType::Float32;
    return ops::pow(self.to(DType::Float64), exponent.to(DType::Float64))
        .to(out);
}

Tensor mvlgamma_native(const Tensor& self, int64_t p) {
    if (p < 1) {
        TP_THROW(RuntimeError, "multigammaln requires p >= 1, but got ", p);
    }
    // sum_i lgamma(x - i/2) plus the pi normalization of the multivariate
    // gamma: p*(p-1)/4 * log(pi).
    Tensor acc = ops::lgamma(self);
    for (int64_t i = 1; i < p; ++i) {
        acc = ops::add(acc, ops::lgamma(ops::sub(self, Scalar(i / 2.0))));
    }
    const double pi_norm = static_cast<double>(p) * (p - 1) / 4.0 *
                           std::log(std::acos(-1.0));
    return ops::add(acc, Scalar(pi_norm));
}

Tensor conj_physical_native(const Tensor& self) {
    return ops::clone(ops::conj(self), kContiguous);
}

Tensor& conj_physical__native(Tensor& self) {
    ops::copy_(self, ops::conj(self));
    return self;
}

Tensor& negative__native(Tensor& self) {
    return ops::neg_(self);
}

TENSORPLAY_LIBRARY_IMPL(Composite, PointwiseComposite) {
    m.impl("xlogy", xlogy_native);
    m.impl("xlogy_", xlogy__native);
    m.impl("ldexp", ldexp_native);
    m.impl("ldexp_", ldexp__native);
    m.impl("fmax", fmax_native);
    m.impl("fmin", fmin_native);
    m.impl("float_power", float_power_native);
    m.impl("mvlgamma", mvlgamma_native);
    m.impl("conj_physical", conj_physical_native);
    m.impl("conj_physical_", conj_physical__native);
    m.impl("negative_", negative__native);
}

} // namespace composite
} // namespace tensorplay
