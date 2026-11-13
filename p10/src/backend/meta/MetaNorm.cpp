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

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaNorm) {
    m.impl("norm", norm_kernel);
    m.impl("norm.dim", norm_dim_kernel);
}

} // namespace meta
} // namespace tensorplay