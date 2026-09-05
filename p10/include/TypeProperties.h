#pragma once

// Result-type promotion state machine used by TensorIterator's common dtype
// computation (and reusable wherever the [ResultType] semantics are needed).
//
// The iteration state tracks three independent categories:
//   - dimResult: tensors with at least one dimension
//   - zeroResult: 0-dim tensors that are not wrapped numbers
//   - wrappedResult: 0-dim tensors auto-wrapped from C++/Python numbers
//
// Wrapped numbers participate only when no plain operand defines a category:
// 'float32_tensor + 2' stays float32, while '2 + 3' promotes the integers to
// the default float dtype.  When a tensor operand and a wrapped number both
// appear, the tensor's category wins through combine_categories' precedence.

#include "DType.h"
#include "Context.h"
#include "Tensor.h"
#include "TypePromotion.h"

namespace tensorplay {
namespace native {

struct ResultTypeState {
    ScalarType dimResult = ScalarType::Undefined;
    ScalarType wrappedResult = ScalarType::Undefined;
    ScalarType zeroResult = ScalarType::Undefined;
};

inline ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
    if (a == ScalarType::Undefined) {
        return b;
    }
    if (b == ScalarType::Undefined) {
        return a;
    }
    return promoteTypes(a, b);
}

inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
    if (isComplexType(higher)) {
        return higher;
    } else if (isComplexType(lower)) {
        // preserve value type of higher if it is floating type.
        if (isFloatingType(higher)) {
            const ScalarType complex_type = toComplexType(higher);
            if (complex_type == ScalarType::Undefined) {
                TP_THROW(TypeError, "Cannot promote ", toString(higher),
                         " with a complex dtype");
            }
            return complex_type;
        }
        // in case of integral input
        // lower complex takes precedence.
        return lower;
    } else if (isFloatingType(higher)) {
        return higher;
    }
    if (higher == ScalarType::Bool || isFloatingType(lower)) {
        return promote_skip_undefined(higher, lower);
    }
    if (higher != ScalarType::Undefined) {
        return higher;
    }
    return lower;
}

inline ResultTypeState update_result_type_state(const Tensor& tensor,
                                                const ResultTypeState& in_state) {
    if (!tensor.defined()) {
        return in_state;
    }
    ResultTypeState new_state = in_state;
    const bool is_wrapped_number = tensor.unsafeGetTensorImpl()->is_wrapped_number();
    ScalarType current = tensor.dtype();
    if (is_wrapped_number) {
        if (isComplexType(current)) {
            current = globalContext().defaultComplexDType();
        } else if (isFloatingType(current)) {
            current = globalContext().defaultDType();
        }
    }
    if (tensor.dim() > 0) {
        new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
    } else if (is_wrapped_number) {
        new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
    } else {
        new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
    }
    return new_state;
}

inline ResultTypeState update_result_type_state(const Scalar& scalar,
                                                const ResultTypeState& in_state) {
    ResultTypeState new_state = in_state;
    ScalarType current = scalar.type();
    if (isComplexType(current)) {
        current = globalContext().defaultComplexDType();
    } else if (isFloatingType(current)) {
        current = globalContext().defaultDType();
    }
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
    return new_state;
}

inline ScalarType result_type(const ResultTypeState& in_state) {
    return combine_categories(
        in_state.dimResult,
        combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

inline ScalarType result_type(const Tensor& tensor, const Tensor& other) {
    ResultTypeState state = {};
    state = update_result_type_state(tensor, state);
    return result_type(update_result_type_state(other, state));
}

}  // namespace native
}  // namespace tensorplay
