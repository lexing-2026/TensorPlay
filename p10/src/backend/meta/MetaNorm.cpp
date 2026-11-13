// Shape-only kernels for vector norms on the meta device.
//
// A norm reduces the operand (over every dimension, or over the named ones)
// to a real-valued result: complex inputs answer in their real component
// type, and floating inputs keep their own type.  Only the reduction
// geometry is observable here.

#include "MetaCommon.h"

#include "DType.h"
#include "TypeProperties.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

DType norm_answer_dtype(const Tensor& self) {
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(RuntimeError,
                 "norm(): input dtype should be either floating point or "
                 "complex. Got ", toString(self.dtype()), " instead.");
    }
    return isComplexType(self.dtype()) ? toRealValueType(self.dtype())
                                       : self.dtype();
}

Tensor norm_kernel(const Tensor& self, double /*p*/) {
    return empty_meta({}, norm_answer_dtype(self), self.device());
}

Tensor norm_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       double /*p*/, bool keepdim) {
    return empty_meta(reduce_out_shape(self, dim, keepdim),
                      norm_answer_dtype(self), self.device());
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

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaNorm) {
    m.impl("norm", norm_kernel);
    m.impl("norm.dim", norm_dim_kernel);
    m.impl("dist", dist_kernel);
    m.impl("renorm", renorm_kernel);
    m.impl("renorm_", renorm_inplace_kernel);
}

} // namespace meta
} // namespace tensorplay