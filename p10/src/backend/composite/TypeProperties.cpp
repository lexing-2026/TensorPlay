// Composite kernels: can_cast / promote_types / result_type (4 overloads) /
// is_conj / is_neg.
// fast path).

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// category (dim'd tensors > 0-dim tensors > wrapped scalars).
DType combine_categories(DType higher, DType lower) {
    if (higher == DType::Undefined) return lower;
    if (lower == DType::Undefined) return higher;
    if (isComplexType(higher)) return higher;
    if (isComplexType(lower)) {
        return isFloatingType(higher) ? toComplexType(higher) : lower;
    }
    if (isFloatingType(higher)) return higher;
    if (higher == DType::Bool || isFloatingType(lower)) {
        return promoteTypes(higher, lower);
    }
    return higher;
}

// Wrapped numbers (Scalar arguments) contribute only when float/complex, and
// update_result_type_state(Scalar)).
DType scalar_wrapped_dtype(const Scalar& s) {
    if (s.isComplex()) return DType::ComplexFloat;
    if (s.isFloatingPoint()) return DType::Float32;
    return DType::Undefined;
}

DType result_type_state(const Tensor* t1, const Tensor* t2,
                        const Scalar* s1, const Scalar* s2) {
    DType dim_result = DType::Undefined;
    DType zero_result = DType::Undefined;
    DType wrapped_result = DType::Undefined;
    const Tensor* tensors[2] = {t1, t2};
    for (const Tensor* t : tensors) {
        if (!t) continue;
        const DType dt = t->dtype();
        if (t->dim() > 0) {
            dim_result = dim_result == DType::Undefined
                             ? dt : promoteTypes(dim_result, dt);
        } else {
            zero_result = zero_result == DType::Undefined
                              ? dt : promoteTypes(zero_result, dt);
        }
    }
    const Scalar* scalars[2] = {s1, s2};
    for (const Scalar* s : scalars) {
        if (!s) continue;
        wrapped_result = combine_categories(wrapped_result,
                                            scalar_wrapped_dtype(*s));
    }
    return combine_categories(dim_result,
                              combine_categories(zero_result, wrapped_result));
}

} // anonymous namespace

bool can_cast_native(DType from_, DType to) {
    return promoteTypes(from_, to) == to;
}

DType promote_types_native(DType type1, DType type2) {
    return promoteTypes(type1, type2);
}

DType result_type_tensor_native(const Tensor& tensor, const Tensor& other) {
    return result_type_state(&tensor, &other, nullptr, nullptr);
}

DType result_type_scalar_native(const Tensor& tensor, const Scalar& other) {
    return result_type_state(&tensor, nullptr, nullptr, &other);
}

DType result_type_scalar_tensor_native(const Scalar& scalar,
                                       const Tensor& tensor) {
    return result_type_state(&tensor, nullptr, &scalar, nullptr);
}

DType result_type_scalar_scalar_native(const Scalar& scalar1,
                                       const Scalar& scalar2) {
    // tensors with their natural dtypes.
    return promoteTypes(scalar_natural_dtype(scalar1),
                        scalar_natural_dtype(scalar2));
}

bool is_conj_native(const Tensor& /*self*/) { return false; }

bool is_neg_native(const Tensor& /*self*/) { return false; }

bool is_distributed_native(const Tensor& /*self*/) { return false; }

bool is_floating_point_native(const Tensor& self) {
    return isFloatingType(self.dtype());
}

bool is_inference_native(const Tensor& self) {
    const auto impl = self.unsafeGetTensorImpl();
    return impl && impl->is_inference();
}

TENSORPLAY_LIBRARY_IMPL(Composite, TypePropertiesComposite) {
    m.impl("can_cast", can_cast_native);
    m.impl("promote_types", promote_types_native);
    m.impl("result_type.Tensor", result_type_tensor_native);
    m.impl("result_type.Scalar", result_type_scalar_native);
    m.impl("result_type.Scalar_Tensor", result_type_scalar_tensor_native);
    m.impl("result_type.Scalar_Scalar", result_type_scalar_scalar_native);
    m.impl("is_distributed", is_distributed_native);
    m.impl("is_floating_point", is_floating_point_native);
    m.impl("is_inference", is_inference_native);
    m.impl("is_conj", is_conj_native);
    m.impl("is_neg", is_neg_native);
}

} // namespace composite
} // namespace tensorplay
