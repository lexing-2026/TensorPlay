// Shape-only kernels for comparisons and selection on the meta device.
// A comparison answers a boolean tensor over the broadcast geometry of
// its operands; where and the minimum/maximum family answer the
// broadcast geometry with the operands' promotion.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace tensorplay {
namespace meta {
namespace {
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Comparisons and selection.
// ---------------------------------------------------------------------------
#define META_COMPARISON_KERNEL(NAME)                                          \
    Tensor NAME##_tensor_kernel(const Tensor& self, const Tensor& other) {     \
        check_all_meta(self, other);                                          \
        return empty_meta(broadcast2(self, other), DType::Bool, self.device());\
    }                                                                         \
    Tensor NAME##_scalar_kernel(const Tensor& self, const Scalar& /*other*/) {\
        return empty_meta(shape_of(self), DType::Bool, self.device());         \
    }

META_COMPARISON_KERNEL(eq)
META_COMPARISON_KERNEL(ne)
META_COMPARISON_KERNEL(lt)
META_COMPARISON_KERNEL(le)
META_COMPARISON_KERNEL(gt)
META_COMPARISON_KERNEL(ge)
#undef META_COMPARISON_KERNEL

Tensor isclose_kernel(const Tensor& self, const Tensor& other, double rtol,
                      double atol, bool /*equal_nan*/) {
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, toString(self.dtype()), " did not match ",
                 toString(other.dtype()));
    }
    if (rtol < 0) {
        TP_THROW(RuntimeError,
                 "rtol must be greater than or equal to zero, but got ", rtol);
    }
    if (atol < 0) {
        TP_THROW(RuntimeError,
                 "atol must be greater than or equal to zero, but got ", atol);
    }
    return empty_meta(broadcast2(self, other), DType::Bool, self.device());
}

Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
    const std::vector<int64_t> out =
        broadcast_shapes(broadcast2(condition, self), shape_of(other));
    return empty_meta(out, promoteTypes(self.dtype(), other.dtype()),
                      self.device());
}

Tensor where_scalar_self_cpu(const Tensor& condition, const Scalar& self,
                             const Tensor& other) {
    const std::vector<int64_t> out = broadcast2(condition, other);
    return empty_meta(out, result_type(self, other.dtype()), other.device());
}

Tensor where_scalar_other_cpu(const Tensor& condition, const Tensor& self,
                              const Scalar& other) {
    const std::vector<int64_t> out = broadcast2(condition, self);
    return empty_meta(out, result_type(other, self.dtype()), self.device());
}

// Two scalars select the selection dtype directly: complex wins, then
// float (widening a double to keep its precision), then Int64.
DType where_scalar_dtype(const Scalar& self, const Scalar& other) {
    if (self.isComplex() || other.isComplex()) {
        return promoteTypes(self.dtype(), other.dtype());
    }
    if (self.isFloatingPoint() || other.isFloatingPoint()) {
        return self.dtype() == DType::Float64 || other.dtype() == DType::Float64
                   ? DType::Float64
                   : DType::Float32;
    }
    return DType::Int64;
}

Tensor where_scalar_scalar_cpu(const Tensor& condition, const Scalar& self,
                               const Scalar& other) {
    return empty_meta(shape_of(condition), where_scalar_dtype(self, other),
                      condition.device());
}

Tensor maximum_cpu(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

Tensor minimum_cpu(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaCompare) {
    m.impl("eq.Tensor", eq_tensor_kernel);
    m.impl("eq.Scalar", eq_scalar_kernel);
    m.impl("ne.Tensor", ne_tensor_kernel);
    m.impl("ne.Scalar", ne_scalar_kernel);
    m.impl("lt.Tensor", lt_tensor_kernel);
    m.impl("lt.Scalar", lt_scalar_kernel);
    m.impl("le.Tensor", le_tensor_kernel);
    m.impl("le.Scalar", le_scalar_kernel);
    m.impl("gt.Tensor", gt_tensor_kernel);
    m.impl("gt.Scalar", gt_scalar_kernel);
    m.impl("ge.Tensor", ge_tensor_kernel);
    m.impl("ge.Scalar", ge_scalar_kernel);
    // The word forms of the comparison primitives answer the same boolean
    // geometry as their symbolic names.
    m.impl("greater", gt_tensor_kernel);
    m.impl("greater_equal", ge_tensor_kernel);
    m.impl("less", lt_tensor_kernel);
    m.impl("less_equal", le_tensor_kernel);
    m.impl("not_equal", ne_tensor_kernel);
    m.impl("isclose", isclose_kernel);
    m.impl("where.self", where_cpu);
    m.impl("where.ScalarSelf", where_scalar_self_cpu);
    m.impl("where.ScalarOther", where_scalar_other_cpu);
    m.impl("where.Scalar", where_scalar_scalar_cpu);
    m.impl("maximum", maximum_cpu);
    m.impl("minimum", minimum_cpu);
}

} // namespace meta
} // namespace tensorplay
