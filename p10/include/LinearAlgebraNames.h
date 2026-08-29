#pragma once

#include "DType.h"
#include <string>

namespace tensorplay {

// Human-readable dtype spellings used by matrix-operation diagnostics and
// general operation errors.
inline const char* c10_style_dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool: return "bool";
        case DType::UInt8: return "unsigned char";
        case DType::Int8: return "signed char";
        case DType::UInt16: return "unsigned short int";
        case DType::Int16: return "short int";
        case DType::UInt32: return "unsigned int";
        case DType::Int32: return "int";
        case DType::UInt64: return "unsigned long int";
        case DType::Int64: return "long int";
        case DType::Float16: return "c10::Half";
        case DType::BFloat16: return "c10::BFloat16";
        case DType::Float32: return "float";
        case DType::Float64: return "double";
        case DType::ComplexHalf: return "c10::complex<c10::Half>";
        case DType::ComplexFloat: return "c10::complex<float>";
        case DType::ComplexDouble: return "c10::complex<double>";
        case DType::BComplex32: return "c10::complex<c10::BFloat16>";
        default: return "Unknown";
    }
}

inline const char* pretty_dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool: return "Bool";
        case DType::UInt8: return "Byte";
        case DType::Int8: return "Char";
        case DType::UInt16: return "UInt16";
        case DType::Int16: return "Short";
        case DType::UInt32: return "UInt32";
        case DType::Int32: return "Int";
        case DType::UInt64: return "UInt64";
        case DType::Int64: return "Long";
        case DType::Float16: return "Half";
        case DType::BFloat16: return "BFloat16";
        case DType::Float32: return "Float";
        case DType::Float64: return "Double";
        case DType::ComplexHalf: return "ComplexHalf";
        case DType::ComplexFloat: return "ComplexFloat";
        case DType::ComplexDouble: return "ComplexDouble";
        case DType::BComplex32: return "ComplexBFloat16";
        default: return "Undefined";
    }
}

} // namespace tensorplay
