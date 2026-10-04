// Composite kernels: can_cast / promote_types / result_type (4 overloads) /
// is_conj / is_neg.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// The result type of up to two tensors and two scalars, by category: tensors
// with dimensions over 0-dim tensors over wrapped scalars, and within a
// category by promotion.  A scalar takes part as a wrapped number -- a real
// one as the default real type, a complex one as the default complex type,
// whole numbers and truths as themselves.
DType result_type_state(const Tensor* t1, const Tensor* t2,
                        const Scalar* s1, const Scalar* s2) {
    native::ResultTypeState state{};
    const Tensor* tensors[2] = {t1, t2};
    for (const Tensor* t : tensors) {
        if (t != nullptr) state = native::update_result_type_state(*t, state);
    }
    const Scalar* scalars[2] = {s1, s2};
    for (const Scalar* s : scalars) {
        if (s != nullptr) state = native::update_result_type_state(*s, state);
    }
    return native::result_type(state);
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
    // Two wrapped numbers and nothing else: their own promotion, with a real
    // or complex one taken as the default type of its kind.
    return result_type_state(nullptr, nullptr, &scalar1, &scalar2);
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
