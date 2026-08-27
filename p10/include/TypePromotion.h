#pragma once

#include "DType.h"
#include "Scalar.h"
#include <algorithm>

namespace tensorplay {

// Promotion rules for the dtypes implemented by TensorPlay.  The ordering is
// based on c10::promoteTypes in the vendored PyTorch tree.  In particular,
// uint16/uint32/uint64 are storage dtypes, but PyTorch deliberately does not
// define integral promotion for them; only promotion with a floating dtype is
// accepted.
inline DType promoteTypes(DType type1, DType type2) {
    if (type1 == type2) {
        return type1;
    }
    
    // If either is undefined, return undefined (or handle error)
    if (type1 == DType::Undefined || type2 == DType::Undefined) {
        return DType::Undefined;
    }

    const bool barebones_unsigned =
        type1 == DType::UInt16 || type1 == DType::UInt32 || type1 == DType::UInt64 ||
        type2 == DType::UInt16 || type2 == DType::UInt32 || type2 == DType::UInt64;
    if (barebones_unsigned) {
        if (isFloatingType(type1)) return type1;
        if (isFloatingType(type2)) return type2;
        TP_THROW(TypeError, "Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ",
                 toString(type1), " and ", toString(type2));
    }

    // The remaining table is the standard PyTorch table for
    // {uint8, int8..int64, float16, float32, float64, complex32..complex128,
    // bool, bfloat16, bcomplex32}.
    const bool is_complex1 = isComplexType(type1);
    const bool is_complex2 = isComplexType(type2);
    if (is_complex1 || is_complex2) {
        // ComplexDouble dominates every supported dtype.
        if (type1 == DType::ComplexDouble || type2 == DType::ComplexDouble) {
            return DType::ComplexDouble;
        }

        // ComplexFloat is the result of complex32 with bfloat16/float32, and
        // dominates all integral, bool, and float32 inputs.  float64 still
        // wins (c10: promote_types(double, complex64) == complex128).
        if (type1 == DType::ComplexFloat || type2 == DType::ComplexFloat) {
            if (type1 == DType::Float64 || type2 == DType::Float64) {
                return DType::ComplexDouble;
            }
            return DType::ComplexFloat;
        }

        // At this point the only ordinary complex type is complex32.  A
        // bcomplex32 input promotes to complex64 with float16/complex32, but
        // remains bcomplex32 with integral/bool inputs.
        const DType other = is_complex1 ? type2 : type1;
        const DType complex_type = is_complex1 ? type1 : type2;
        if ((type1 == DType::ComplexHalf && type2 == DType::BComplex32) ||
            (type2 == DType::ComplexHalf && type1 == DType::BComplex32)) {
            return DType::ComplexFloat;
        }
        if (complex_type == DType::BComplex32) {
            if (other == DType::Float16 || other == DType::ComplexHalf) {
                return DType::ComplexFloat;
            }
            if (other == DType::Float32) return DType::ComplexFloat;
            if (other == DType::Float64) return DType::ComplexDouble;
            return DType::BComplex32;
        }

        // ComplexHalf + bfloat16 is complex64; all other non-double inputs
        // fit in complex32.
        if (other == DType::BFloat16) return DType::ComplexFloat;
        if (other == DType::Float32) return DType::ComplexFloat;
        if (other == DType::Float64) return DType::ComplexDouble;
        return DType::ComplexHalf;
    }

    const bool is_float1 = isFloatingType(type1);
    const bool is_float2 = isFloatingType(type2);
    if (is_float1 || is_float2) {
        if (type1 == DType::Float64 || type2 == DType::Float64) return DType::Float64;
        if (type1 == DType::Float32 || type2 == DType::Float32) return DType::Float32;
        // PyTorch promotes half + bfloat16 to float32, unlike a size-only
        // rule (both occupy two bytes).
        if ((type1 == DType::Float16 && type2 == DType::BFloat16) ||
            (type2 == DType::Float16 && type1 == DType::BFloat16)) {
            return DType::Float32;
        }
        if (type1 == DType::BFloat16 || type2 == DType::BFloat16) return DType::BFloat16;
        return DType::Float16;
    }

    // Bool is a distinct category in PyTorch, but bool + integral uses the
    // integral operand.  uint8 + int8 is the one asymmetric integer case and
    // promotes to int16.
    if (type1 == DType::Bool) return type2;
    if (type2 == DType::Bool) return type1;
    if (type1 == DType::UInt8) {
        if (type2 == DType::UInt8) return DType::UInt8;
        if (type2 == DType::Int8 || type2 == DType::Int16) return DType::Int16;
        if (type2 == DType::Int32) return DType::Int32;
        if (type2 == DType::Int64) return DType::Int64;
    }
    if (type2 == DType::UInt8) {
        if (type1 == DType::Int8 || type1 == DType::Int16) return DType::Int16;
        if (type1 == DType::Int32) return DType::Int32;
        if (type1 == DType::Int64) return DType::Int64;
    }

    if (type1 == DType::Int64 || type2 == DType::Int64) return DType::Int64;
    if (type1 == DType::Int32 || type2 == DType::Int32) return DType::Int32;
    if (type1 == DType::Int16 || type2 == DType::Int16) return DType::Int16;
    if (type1 == DType::Int8 || type2 == DType::Int8) return DType::Int8;
    if (type1 == DType::UInt8 && type2 == DType::UInt8) return DType::UInt8;

    return DType::Undefined;
}

// Result type of Tensor + Scalar
// Simplified: If scalar is float and tensor is int, result is float. Otherwise tensor type wins.
inline DType result_type(const Scalar& scalar, DType tensorType) {
    if (isFloatingOrComplexType(tensorType)) {
        return tensorType;
    }
    if (scalar.isComplex()) {
        // Int Tensor + Complex Scalar -> Complex64 Tensor
        return DType::ComplexFloat;
    }
    if (scalar.isFloatingPoint()) {
        // Int Tensor + Float Scalar -> Float Tensor (usually Float32 default unless tensor is Double)
        return DType::Float32;
    }
    return tensorType;
}

} // namespace tensorplay
