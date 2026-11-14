// Shape-only kernels for vector norms on the meta device.
//
// A norm reduces the operand (over every dimension, or over the named ones)
// to a real-valued result: complex inputs answer in their real component
// type, and floating inputs keep their own type.  Only the reduction
// geometry is observable here.

#include "MetaCommon.h"

#include "DType.h"
#include "TypeProperties.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// The value type of a norm.  A floating or complex operand answers directly
// (the complex planes answer in their real component type).  A whole-number
// operand runs the norm as abs, a power, a sum and a root: an infinite order
// short-circuits to the largest absolute value and keeps the operand type, a
// negative whole-number order is refused, and the chained powers answer in
// the operand type only when both the order and its reciprocal are whole
// numbers (order 1); otherwise single precision.  A boolean operand has no
// power form: an infinite order reads as the largest element, anything else
// is refused.
DType norm_answer_dtype(const Tensor& self, double p) {
    const DType dt = self.dtype();
    if (isComplexType(dt)) return toRealValueType(dt);
    if (isFloatingType(dt)) return dt;
    if (std::isinf(p)) return dt;
    if (dt == DType::Bool) {
        TP_THROW(TypeError,
                 "pow() received an invalid combination of arguments");
    }
    if (p == std::floor(p)) {
        if (p < 0) {
            TP_THROW(RuntimeError,
                     "Integers to negative integer powers are not allowed.");
        }
        if (std::floor(1.0 / p) == 1.0 / p) return dt;
    }
    return DType::Float32;
}

Tensor norm_kernel(const Tensor& self, double p) {
    return empty_meta({}, norm_answer_dtype(self, p), self.device());
}

Tensor norm_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       double p, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim),
                      norm_answer_dtype(self, p), self.device());
}

// A distance is the norm of the difference: the subtraction promotes the pair
// (boolean operands have no subtraction), the norm answers at the pair's real
// type, and a whole-number pair answers at the default real type.
Tensor dist_kernel(const Tensor& self, const Tensor& other, const Scalar& /*p*/) {
    check_all_meta(self, other);
    const DType pair = promoteTypes(self.dtype(), other.dtype());
    if (pair == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    DType out = pair;
    if (isComplexType(pair)) {
        out = toRealValueType(pair);
    } else if (!isFloatingType(pair)) {
        out = DType::Float32;
    }
    return empty_meta({}, out, self.device());
}

// Renormalization rescales slices in place in value space; the geometry and
// the value type stay as they were.  The norm order and the clamp bound are
// validated the way the computing path validates them.
Tensor renorm_kernel(const Tensor& self, const Scalar& p, int64_t dim,
                     const Scalar& maxnorm) {
    if (p.isComplex()) {
        TP_THROW(TypeError, "renorm: p must be real-valued");
    }
    if (!(p.toDouble() > 0.0)) {
        TP_THROW(ValueError, "renorm: norm order must be positive");
    }
    if (maxnorm.isComplex()) {
        TP_THROW(TypeError, "renorm: maxnorm must be real-valued");
    }
    if (!(maxnorm.toDouble() >= 0.0)) {
        TP_THROW(ValueError, "renorm: maxnorm must be non-negative");
    }
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "renorm: input must have a floating or complex dtype");
    }
    if (self.dim() <= 1) {
        TP_THROW(RuntimeError, "renorm: input must have at least 2 dimensions");
    }
    wrap_dim(dim, self.dim());
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& renorm_inplace_kernel(Tensor& self, const Scalar& p, int64_t dim,
                              const Scalar& maxnorm) {
    renorm_kernel(self, p, dim, maxnorm);
    return self;
}

// The pairwise distances condense a matrix of points into one distance per
// unordered pair: the upper triangle holds N * (N - 1) / 2 entries, which is
// also the answer for the empty and single-row inputs where it is zero.  The
// distances accumulate at single precision, so a half-width operand answers
// in single precision.
Tensor pdist_kernel(const Tensor& self, double p) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError, "pdist: only floating-point tensors are supported");
    }
    if (self.dim() != 2) {
        TP_THROW(RuntimeError, "pdist only supports 2D tensors, got: ",
                 self.dim(), "D");
    }
    if (p < 0.0) {
        TP_THROW(RuntimeError, "pdist only supports non-negative p values");
    }
    const DType dt = (self.dtype() == DType::Float16 ||
                      self.dtype() == DType::BFloat16)
                         ? DType::Float32
                         : self.dtype();
    const int64_t n = self.size(0);
    return empty_meta({n * (n - 1) / 2}, dt, self.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaNorm) {
    m.impl("norm", norm_kernel);
    m.impl("norm.dim", norm_dim_kernel);
    m.impl("dist", dist_kernel);
    m.impl("renorm", renorm_kernel);
    m.impl("renorm_", renorm_inplace_kernel);
    m.impl("pdist", pdist_kernel);
}

} // namespace meta
} // namespace tensorplay