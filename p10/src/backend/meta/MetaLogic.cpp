// Shape-only kernels for logical and bitwise operators on the meta device.
//
// A logical operator always answers a boolean tensor over the broadcast
// geometry of its operands, whatever their types.  A bitwise operator
// promotes its integer or boolean operands and answers in the promoted type;
// the shifts refuse booleans.  Neither reads element data.

#include "MetaCommon.h"

#include "DType.h"
#include "Scalar.h"
#include "TypeProperties.h"

#include <vector>

namespace tensorplay {
namespace meta {

namespace {

Tensor logical_unary_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), DType::Bool, self.device());
}

Tensor logical_binary_kernel(const Tensor& a, const Tensor& b) {
    check_all_meta(a, b);
    return empty_meta(broadcast2(a, b), DType::Bool, a.device());
}

Tensor& logical_inplace_kernel(Tensor& self) {
    return self;
}

// ---------------------------------------------------------------------------
// Bitwise operators.
// ---------------------------------------------------------------------------

void check_bitwise(const Tensor& tensor, const char* name) {
    if (tensor.dtype() == DType::Bool || isIntegralType(tensor.dtype())) return;
    TP_THROW(TypeError, name, ": only integral and boolean types are supported");
}

DType bitwise_result_dtype(DType dtype, const char* name) {
    if (dtype != DType::Bool && !isIntegralType(dtype)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    return dtype;
}

Tensor bitwise_not_kernel(const Tensor& self) {
    check_bitwise(self, "bitwise_not");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor bitwise_tensor_op(const Tensor& a, const Tensor& b, const char* name) {
    check_all_meta(a, b);
    check_bitwise(a, name);
    check_bitwise(b, name);
    return empty_meta(broadcast2(a, b),
                      bitwise_result_dtype(native::result_type(a, b), name),
                      a.device());
}

Tensor bitwise_scalar_op(const Tensor& self, const Scalar& other,
                         const char* name) {
    check_bitwise(self, name);
    return empty_meta(shape_of(self),
                      bitwise_result_dtype(result_type(other, self.dtype()), name),
                      self.device());
}

Tensor bitwise_scalar_tensor_op(const Scalar& self, const Tensor& other,
                                const char* name) {
    check_bitwise(other, name);
    return empty_meta(shape_of(other),
                      bitwise_result_dtype(result_type(self, other.dtype()), name),
                      other.device());
}

// A shift promotes like the other bitwise operators but has no meaning for a
// boolean value, so a boolean operand is refused after promotion.
Tensor shift_tensor_op(const Tensor& a, const Tensor& b, const char* name) {
    check_all_meta(a, b);
    check_bitwise(a, name);
    check_bitwise(b, name);
    const DType dtype = bitwise_result_dtype(native::result_type(a, b), name);
    if (dtype == DType::Bool) {
        TP_THROW(TypeError, name, ": unsupported dtype");
    }
    return empty_meta(broadcast2(a, b), dtype, a.device());
}

Tensor shift_scalar_op(const Tensor& self, const Scalar& other,
                       const char* name) {
    check_bitwise(self, name);
    const DType dtype = bitwise_result_dtype(result_type(other, self.dtype()), name);
    if (dtype == DType::Bool) {
        TP_THROW(TypeError, name, ": unsupported dtype");
    }
    return empty_meta(shape_of(self), dtype, self.device());
}

Tensor shift_scalar_tensor_op(const Scalar& self, const Tensor& other,
                              const char* name) {
    check_bitwise(other, name);
    const DType dtype = bitwise_result_dtype(result_type(self, other.dtype()), name);
    if (dtype == DType::Bool) {
        TP_THROW(TypeError, name, ": unsupported dtype");
    }
    return empty_meta(shape_of(other), dtype, other.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaLogic) {
    // Logical operators answer booleans over the broadcast geometry.
    m.impl("logical_not", logical_unary_kernel);
    m.impl("logical_and", logical_binary_kernel);
    m.impl("logical_or", logical_binary_kernel);
    m.impl("logical_xor", logical_binary_kernel);
    m.impl("logical_not_", logical_inplace_kernel);
    m.impl("logical_and_", logical_inplace_kernel);
    m.impl("logical_or_", logical_inplace_kernel);
    m.impl("logical_xor_", logical_inplace_kernel);

    // Bitwise operators promote their operands to an integer (or boolean)
    // type and keep that type.
    m.impl("bitwise_not", bitwise_not_kernel);
    m.impl("bitwise_not_", logical_inplace_kernel);
    m.impl("bitwise_and.Tensor", [](const Tensor& a, const Tensor& b) {
        return bitwise_tensor_op(a, b, "bitwise_and");
    });
    m.impl("bitwise_or.Tensor", [](const Tensor& a, const Tensor& b) {
        return bitwise_tensor_op(a, b, "bitwise_or");
    });
    m.impl("bitwise_xor.Tensor", [](const Tensor& a, const Tensor& b) {
        return bitwise_tensor_op(a, b, "bitwise_xor");
    });
    m.impl("bitwise_and.Scalar", [](const Tensor& self, const Scalar& other) {
        return bitwise_scalar_op(self, other, "bitwise_and");
    });
    m.impl("bitwise_or.Scalar", [](const Tensor& self, const Scalar& other) {
        return bitwise_scalar_op(self, other, "bitwise_or");
    });
    m.impl("bitwise_xor.Scalar", [](const Tensor& self, const Scalar& other) {
        return bitwise_scalar_op(self, other, "bitwise_xor");
    });
    m.impl("bitwise_and.Scalar_Tensor",
           [](const Scalar& self, const Tensor& other) {
               return bitwise_scalar_tensor_op(self, other, "bitwise_and");
           });
    m.impl("bitwise_or.Scalar_Tensor",
           [](const Scalar& self, const Tensor& other) {
               return bitwise_scalar_tensor_op(self, other, "bitwise_or");
           });
    m.impl("bitwise_xor.Scalar_Tensor",
           [](const Scalar& self, const Tensor& other) {
               return bitwise_scalar_tensor_op(self, other, "bitwise_xor");
           });
    m.impl("bitwise_left_shift.Tensor", [](const Tensor& a, const Tensor& b) {
        return shift_tensor_op(a, b, "bitwise_left_shift");
    });
    m.impl("bitwise_right_shift.Tensor", [](const Tensor& a, const Tensor& b) {
        return shift_tensor_op(a, b, "bitwise_right_shift");
    });
    m.impl("bitwise_left_shift.Tensor_Scalar",
           [](const Tensor& self, const Scalar& other) {
               return shift_scalar_op(self, other, "bitwise_left_shift");
           });
    m.impl("bitwise_right_shift.Tensor_Scalar",
           [](const Tensor& self, const Scalar& other) {
               return shift_scalar_op(self, other, "bitwise_right_shift");
           });
    m.impl("bitwise_left_shift.Scalar_Tensor",
           [](const Scalar& self, const Tensor& other) {
               return shift_scalar_tensor_op(self, other, "bitwise_left_shift");
           });
    m.impl("bitwise_right_shift.Scalar_Tensor",
           [](const Scalar& self, const Tensor& other) {
               return shift_scalar_tensor_op(self, other, "bitwise_right_shift");
           });
}

} // namespace meta
} // namespace tensorplay
