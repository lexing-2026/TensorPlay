#pragma once

#include "DType.h"
#include "Scalar.h"
#include <array>

namespace tensorplay {

namespace detail {

constexpr auto kPromoteUInt8 = ScalarType::UInt8;
constexpr auto kPromoteInt8 = ScalarType::Int8;
constexpr auto kPromoteInt16 = ScalarType::Int16;
constexpr auto kPromoteInt32 = ScalarType::Int32;
constexpr auto kPromoteInt64 = ScalarType::Int64;
constexpr auto kPromoteFloat16 = ScalarType::Float16;
constexpr auto kPromoteFloat32 = ScalarType::Float32;
constexpr auto kPromoteFloat64 = ScalarType::Float64;
constexpr auto kPromoteComplexHalf = ScalarType::ComplexHalf;
constexpr auto kPromoteComplexFloat = ScalarType::ComplexFloat;
constexpr auto kPromoteComplexDouble = ScalarType::ComplexDouble;
constexpr auto kPromoteBool = ScalarType::Bool;
constexpr auto kPromoteBFloat16 = ScalarType::BFloat16;
constexpr auto kPromoteBComplex32 = ScalarType::BComplex32;

constexpr std::array<ScalarType, 14> kPromoteIndexToDType = {
    kPromoteUInt8,       kPromoteInt8,        kPromoteInt16,
    kPromoteInt32,       kPromoteInt64,       kPromoteFloat16,
    kPromoteFloat32,     kPromoteFloat64,     kPromoteComplexHalf,
    kPromoteComplexFloat, kPromoteComplexDouble, kPromoteBool,
    kPromoteBFloat16,    kPromoteBComplex32};

constexpr std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)>
makePromoteDTypeToIndex() {
    std::array<int64_t, static_cast<size_t>(ScalarType::NumOptions)> inverse{};
    for (auto& value : inverse) {
        value = -1;
    }
    for (size_t index = 0; index < kPromoteIndexToDType.size(); ++index) {
        inverse[static_cast<size_t>(kPromoteIndexToDType[index])] =
            static_cast<int64_t>(index);
    }
    return inverse;
}

constexpr auto kPromoteDTypeToIndex = makePromoteDTypeToIndex();

constexpr std::array<std::array<ScalarType, kPromoteIndexToDType.size()>,
                     kPromoteIndexToDType.size()>
    kPromoteTypesLookup = {{
        {{kPromoteUInt8, kPromoteInt16, kPromoteInt16, kPromoteInt32,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteUInt8, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteInt16, kPromoteInt8, kPromoteInt16, kPromoteInt32,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteInt8, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteInt16, kPromoteInt16, kPromoteInt16, kPromoteInt32,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteInt16, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteInt32, kPromoteInt32, kPromoteInt32, kPromoteInt32,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteInt32, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteInt64, kPromoteInt64, kPromoteInt64, kPromoteInt64,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteInt64, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteFloat16, kPromoteFloat16, kPromoteFloat16,
          kPromoteFloat16, kPromoteFloat16, kPromoteFloat16,
          kPromoteFloat32, kPromoteFloat64, kPromoteComplexHalf,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteFloat16,
          kPromoteFloat32, kPromoteComplexFloat}},
        {{kPromoteFloat32, kPromoteFloat32, kPromoteFloat32,
          kPromoteFloat32, kPromoteFloat32, kPromoteFloat32,
          kPromoteFloat32, kPromoteFloat64, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteFloat32,
          kPromoteFloat32, kPromoteComplexFloat}},
        {{kPromoteFloat64, kPromoteFloat64, kPromoteFloat64,
          kPromoteFloat64, kPromoteFloat64, kPromoteFloat64,
          kPromoteFloat64, kPromoteFloat64, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble, kPromoteFloat64,
          kPromoteFloat64, kPromoteComplexDouble}},
        {{kPromoteComplexHalf, kPromoteComplexHalf, kPromoteComplexHalf,
          kPromoteComplexHalf, kPromoteComplexHalf, kPromoteComplexHalf,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteComplexHalf,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteComplexHalf,
          kPromoteComplexFloat, kPromoteComplexFloat}},
        {{kPromoteComplexFloat, kPromoteComplexFloat, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexFloat, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexFloat}},
        {{kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble, kPromoteComplexDouble,
          kPromoteComplexDouble}},
        {{kPromoteUInt8, kPromoteInt8, kPromoteInt16, kPromoteInt32,
          kPromoteInt64, kPromoteFloat16, kPromoteFloat32, kPromoteFloat64,
          kPromoteComplexHalf, kPromoteComplexFloat, kPromoteComplexDouble,
          kPromoteBool, kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteBFloat16, kPromoteBFloat16, kPromoteBFloat16,
          kPromoteBFloat16, kPromoteBFloat16, kPromoteFloat32,
          kPromoteFloat32, kPromoteFloat64, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteBFloat16,
          kPromoteBFloat16, kPromoteBComplex32}},
        {{kPromoteBComplex32, kPromoteBComplex32, kPromoteBComplex32,
          kPromoteBComplex32, kPromoteBComplex32, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteComplexFloat,
          kPromoteComplexFloat, kPromoteComplexDouble, kPromoteBComplex32,
          kPromoteBComplex32, kPromoteBComplex32}},
    }};

}  // namespace detail

inline DType promoteTypes(DType type1, DType type2) {
    if (type1 == DType::Undefined || type2 == DType::Undefined) {
        return DType::Undefined;
    }

    if (type1 == type2) {
        return type1;
    }

    if (isQIntType(type1) || isQIntType(type2)) {
        TP_THROW(TypeError,
                 "promoteTypes with quantized numbers is not handled yet; "
                 "offending types: ",
                 toString(type1), " ", toString(type2));
    }

    if (isFloat8Type(type1) || isFloat8Type(type2)) {
        TP_THROW(TypeError,
                 "Promotion for Float8 Types is not supported, attempted to "
                 "promote ",
                 toString(type1), " and ", toString(type2));
    }

    const bool barebones_unsigned =
        type1 == DType::UInt16 || type1 == DType::UInt32 || type1 == DType::UInt64 ||
        type2 == DType::UInt16 || type2 == DType::UInt32 || type2 == DType::UInt64;
    if (barebones_unsigned) {
        if (isFloatingType(type1)) {
            return type1;
        }
        if (isFloatingType(type2)) {
            return type2;
        }
        TP_THROW(TypeError, "Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ",
                 toString(type1), " and ", toString(type2));
    }

    const int64_t ix1 = detail::kPromoteDTypeToIndex[
        static_cast<size_t>(type1)];
    const int64_t ix2 = detail::kPromoteDTypeToIndex[
        static_cast<size_t>(type2)];
    TP_CHECK(ix1 >= 0 && ix2 >= 0,
             "promoteTypes received an unsupported scalar type");
    return detail::kPromoteTypesLookup[static_cast<size_t>(ix1)]
                                      [static_cast<size_t>(ix2)];
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
