// Shape-only kernels for the assorted value-free ops on the meta device:
// alpha dropout, the histogram contract, and the floating decomposition
// frexp.
//
// Alpha dropout samples a keep mask and rescales the input; only the
// geometries of the (output, mask) pair are observable.  The histogram bin
// count is the answer's whole extent, so the bin validations are scalar
// checks that run unchanged; deriving the span from the stored extrema when
// min equals max reads element data, so that path answers the shape and
// leaves the span validation to a materializing backend.

#include "MetaCommon.h"

#include "DType.h"
#include "Exception.h"
#include "Scalar.h"

#include <cmath>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

std::tuple<Tensor, Tensor> native_alpha_dropout_kernel(const Tensor& input,
                                                       double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "alpha_dropout: p must be in [0, 1)");
    }
    // The keep mask is sampled in single precision regardless of the input;
    // the affine rescale lands on the promoted type.
    return {empty_meta(shape_of(input),
                       promoteTypes(input.dtype(), DType::Float32), input.device()),
            empty_meta(shape_of(input), DType::Float32, input.device())};
}

Tensor histc_kernel(const Tensor& self, int64_t bins, const Scalar& min,
                    const Scalar& max) {
    if (bins <= 0) {
        TP_THROW(RuntimeError, "histc(): bins must be positive");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    const double lo = min.toDouble();
    const double hi = max.toDouble();
    if (lo == hi) {
        // An equal range derives the span from the stored extrema, which a
        // metadata-only device cannot read; answer the shape and leave the
        // span validation to a materializing backend.
        return empty_meta({bins}, self.dtype(), self.device());
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi, "] is not finite");
    }
    if (!(lo < hi)) {
        TP_THROW(RuntimeError, "histc: max must be larger than min");
    }
    return empty_meta({bins}, self.dtype(), self.device());
}

std::tuple<Tensor, Tensor> frexp_kernel(const Tensor& self) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "frexp(): only supports floating-point dtypes");
    }
    // The mantissa in [0.5, 1) (or exactly 0) carries the input dtype and the
    // exponent plane is 32-bit.
    return {empty_meta(shape_of(self), self.dtype(), self.device()),
            empty_meta(shape_of(self), DType::Int32, self.device())};
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaMisc) {
    m.impl("native_alpha_dropout", native_alpha_dropout_kernel);
    m.impl("histc", histc_kernel);
    m.impl("frexp", frexp_kernel);
}

} // namespace meta
} // namespace tensorplay
