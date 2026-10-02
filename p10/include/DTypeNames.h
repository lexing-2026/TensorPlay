#pragma once

#include "DType.h"

namespace tensorplay {

// Two ways an error message names a dtype, besides the dtype's own name.
//
// A message about the elements two operands hold names the element as the
// language spells its type ("float != double"); a message about which scalar
// kind an operation wanted names the kind ("expected scalar type Float but
// found Double").  Which one a message uses is part of its wording, so both
// spellings live here and every message takes its own from one place.

// The element as its C++ type is spelled.
inline const char* elementTypeName(DType dtype) {
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
        case DType::Float16: return "tensorplay::Half";
        case DType::BFloat16: return "tensorplay::BFloat16";
        case DType::Float32: return "float";
        case DType::Float64: return "double";
        case DType::ComplexHalf: return "tensorplay::complex<tensorplay::Half>";
        case DType::ComplexFloat: return "tensorplay::complex<float>";
        case DType::ComplexDouble: return "tensorplay::complex<double>";
        case DType::BComplex32:
            return "tensorplay::complex<tensorplay::BFloat16>";
        default: return "Unknown";
    }
}

// The scalar kind by its short name.
inline const char* scalarTypeName(DType dtype) {
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
