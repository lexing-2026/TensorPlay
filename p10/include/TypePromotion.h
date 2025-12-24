#pragma once

#include "DType.h"
#include "Scalar.h"
#include <algorithm>

namespace tensorplay {

// Simplified type promotion rules
inline DType promoteTypes(DType type1, DType type2) {
    if (type1 == type2) {
        return type1;
    }
    
    // If either is undefined, return undefined (or handle error)
    if (type1 == DType::Undefined || type2 == DType::Undefined) {
        return DType::Undefined;
    }

    // Priority: Complex > Float > Int > Bool
    // Within category: larger size wins
    
    bool is_complex1 = (type1 == DType::ComplexFloat || type1 == DType::ComplexDouble);
    bool is_complex2 = (type2 == DType::ComplexFloat || type2 == DType::ComplexDouble);

    if (is_complex1 && is_complex2) {
        return (elementSize(type1) >= elementSize(type2)) ? type1 : type2;
    }
    if (is_complex1) return type1;
    if (is_complex2) return type2;
    
    bool is_float1 = isFloatingType(type1);
    bool is_float2 = isFloatingType(type2);
    
    if (is_float1 && is_float2) {
        // Both float, pick larger
        return (elementSize(type1) >= elementSize(type2)) ? type1 : type2;
    }
    
    if (is_float1) return type1;
    if (is_float2) return type2;
    
    bool is_int1 = isIntegralType(type1, false);
    bool is_int2 = isIntegralType(type2, false);
    
    if (is_int1 && is_int2) {
        // Both int, pick larger (simplified: assume all signed for now or standard promotion)
        // For now, let's just pick the one with larger size
        return (elementSize(type1) >= elementSize(type2)) ? type1 : type2;
    }
    
    // Bool vs Int -> Int
    if (type1 == DType::Bool && is_int2) return type2;
    if (is_int1 && type2 == DType::Bool) return type1;
    
    // Bool vs Bool -> Bool
    if (type1 == DType::Bool && type2 == DType::Bool) return type1;
    
    return DType::Undefined; // Should not reach here for supported types
}

// Result type of Tensor + Scalar
// Simplified: If scalar is float and tensor is int, result is float. Otherwise tensor type wins.
inline DType result_type(const Scalar& scalar, DType tensorType) {
    if (isFloatingType(tensorType)) {
        return tensorType;
    }
    if (scalar.isFloatingPoint()) {
        // Int Tensor + Float Scalar -> Float Tensor (usually Float32 default unless tensor is Double)
        return DType::Float32; 
    }
    return tensorType;
}

} // namespace tensorplay
