#include "Scalar.h"
#include "TypePromotion.h"
#include "Exception.h"
#include <cmath>
#include <sstream>

namespace tensorplay {

DType Scalar::promote_types(DType a, DType b) {
    return promoteTypes(a, b);
}

// Helper macro for binary ops
#define SCALAR_BINARY_OP(OP, NAME) \
Scalar Scalar::operator OP(const Scalar& other) const { \
    DType result_dtype = promote_types(type_, other.type_); \
    if (result_dtype == DType::Float64) { \
        double v1 = this->to<double>(); \
        double v2 = other.to<double>(); \
        return Scalar(v1 OP v2); \
    } else if (result_dtype == DType::Float32) { \
        float v1 = this->to<float>(); \
        float v2 = other.to<float>(); \
        return Scalar(v1 OP v2); \
    } else if (result_dtype == DType::Int64) { \
        int64_t v1 = this->to<int64_t>(); \
        int64_t v2 = other.to<int64_t>(); \
        return Scalar(v1 OP v2); \
    } else if (result_dtype == DType::Int32) { \
        int32_t v1 = this->to<int32_t>(); \
        int32_t v2 = other.to<int32_t>(); \
        return Scalar(v1 OP v2); \
    } else if (result_dtype == DType::Bool) { \
        bool v1 = this->to<bool>(); \
        bool v2 = other.to<bool>(); \
        /* Arithmetic on bools usually promotes to int in C++, mimicking PyTorch behavior */ \
        /* PyTorch: True + True = 2 (Long). Bool + Bool -> Long usually? */ \
        /* For now, let's cast to int64 if operation is arithmetic */ \
        return Scalar(static_cast<int64_t>(v1) OP static_cast<int64_t>(v2)); \
    } \
    TP_THROW(TypeError, "Unsupported scalar types for " #NAME); \
}

SCALAR_BINARY_OP(+, add)
SCALAR_BINARY_OP(-, sub)
SCALAR_BINARY_OP(*, mul)

// Division is special (float division vs integer division)
Scalar Scalar::operator/(const Scalar& other) const {
    DType result_dtype = promote_types(type_, other.type_);
    // In PyTorch, division usually results in float, unless floor_divide
    // Here we implement standard C++ division behavior but promoted
    
    if (isFloatingPoint() || other.isFloatingPoint()) {
        if (result_dtype == DType::Float64) {
            return Scalar(this->to<double>() / other.to<double>());
        } else {
            return Scalar(this->to<float>() / other.to<float>());
        }
    } else {
        // Integer division
        // Check for division by zero?
        if (result_dtype == DType::Int64) {
             return Scalar(this->to<int64_t>() / other.to<int64_t>());
        } else {
             return Scalar(this->to<int32_t>() / other.to<int32_t>());
        }
    }
}

#undef SCALAR_BINARY_OP

// Comparisons
#define SCALAR_COMPARE_OP(OP) \
bool Scalar::operator OP(const Scalar& other) const { \
    DType common_dtype = promote_types(type_, other.type_); \
    if (common_dtype == DType::Float64) { \
        return this->to<double>() OP other.to<double>(); \
    } else if (common_dtype == DType::Float32) { \
        return this->to<float>() OP other.to<float>(); \
    } else if (common_dtype == DType::Int64) { \
        return this->to<int64_t>() OP other.to<int64_t>(); \
    } else if (common_dtype == DType::Int32) { \
        return this->to<int32_t>() OP other.to<int32_t>(); \
    } else if (common_dtype == DType::Bool) { \
        return this->to<bool>() OP other.to<bool>(); \
    } \
    return false; \
}

SCALAR_COMPARE_OP(==)
SCALAR_COMPARE_OP(!=)
SCALAR_COMPARE_OP(>)
SCALAR_COMPARE_OP(<)

#undef SCALAR_COMPARE_OP

} // namespace tensorplay
