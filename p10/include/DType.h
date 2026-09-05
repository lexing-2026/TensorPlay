#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <type_traits>

#include "BFloat16.h"
#include "Exception.h"
#include "Float8_e4m3fn.h"
#include "Float8_e4m3fnuz.h"
#include "Float8_e5m2.h"
#include "Float8_e5m2fnuz.h"
#include "Float8_e8m0fnu.h"
#include "Half.h"

#ifdef UInt8
#undef UInt8
#endif
#ifdef Int8
#undef Int8
#endif
#ifdef UInt16
#undef UInt16
#endif
#ifdef Int16
#undef Int16
#endif
#ifdef UInt32
#undef UInt32
#endif
#ifdef Int32
#undef Int32
#endif
#ifdef UInt64
#undef UInt64
#endif
#ifdef Int64
#undef Int64
#endif
#ifdef Float32
#undef Float32
#endif
#ifdef Float64
#undef Float64
#endif
#ifdef Bool
#undef Bool
#endif
#ifdef Byte
#undef Byte
#endif
#ifdef Char
#undef Char
#endif
#ifdef Short
#undef Short
#endif
#ifdef Int
#undef Int
#endif
#ifdef Long
#undef Long
#endif
#ifdef Half
#undef Half
#endif
#ifdef Float
#undef Float
#endif
#ifdef Double
#undef Double
#endif
#ifdef small
#undef small
#endif

namespace tensorplay {

#define TENSORPLAY_FORALL_SCALAR_TYPES(_) \
    _(uint8_t, UInt8)                     \
    _(int8_t, Int8)                       \
    _(int16_t, Int16)                     \
    _(int32_t, Int32)                     \
    _(int64_t, Int64)                     \
    _(uint16_t, UInt16)                   \
    _(uint32_t, UInt32)                   \
    _(uint64_t, UInt64)                   \
    _(float, Float32)                     \
    _(double, Float64)                    \
    _(tensorplay::Half, Float16)          \
    _(tensorplay::BFloat16, BFloat16)     \
    _(bool, Bool)

#define TENSORPLAY_FORALL_INT_TYPES(_) \
    _(uint8_t, UInt8)                    \
    _(int8_t, Int8)                      \
    _(int16_t, Int16)                    \
    _(int32_t, Int32)                    \
    _(int64_t, Int64)                    \
    _(uint16_t, UInt16)                  \
    _(uint32_t, UInt32)                  \
    _(uint64_t, UInt64)

enum class ScalarType : int8_t {
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Float16,
    BFloat16,
    Bool,
    ComplexFloat,
    ComplexDouble,
    ComplexHalf,
    BComplex32,
    Float8_e4m3fn,
    Float8_e5m2,
    QInt8,
    QUInt8,
    QInt32,
    Undefined,
    Float8_e4m3fnuz,
    Float8_e5m2fnuz,
    Float8_e8m0fnu,
    NumOptions,

    Byte = UInt8,
    Char = Int8,
    Short = Int16,
    Int = Int32,
    Long = Int64,
    Half = Float16,
    Float = Float32,
    Double = Float64
};

using DType = ScalarType;

inline bool isQuantizedType(ScalarType type) {
    return type == ScalarType::QInt8 || type == ScalarType::QUInt8 ||
           type == ScalarType::QInt32;
}

inline bool isQIntType(ScalarType type) {
    return isQuantizedType(type);
}

inline bool isQuantizedSigned(ScalarType type) {
    return type == ScalarType::QInt8 || type == ScalarType::QInt32;
}

inline bool isIntegralType(ScalarType type, bool includeBool = false) {
    const bool integral =
        type == ScalarType::UInt8 || type == ScalarType::Int8 ||
        type == ScalarType::Int16 || type == ScalarType::Int32 ||
        type == ScalarType::Int64 || type == ScalarType::UInt16 ||
        type == ScalarType::UInt32 || type == ScalarType::UInt64;
    return integral || (includeBool && type == ScalarType::Bool);
}

inline bool isFloat8Type(ScalarType type) {
    return type == ScalarType::Float8_e4m3fn ||
           type == ScalarType::Float8_e5m2 ||
           type == ScalarType::Float8_e4m3fnuz ||
           type == ScalarType::Float8_e5m2fnuz ||
           type == ScalarType::Float8_e8m0fnu;
}

inline bool isReducedFloatingType(ScalarType type) {
    return type == ScalarType::Float16 || type == ScalarType::BFloat16 ||
           isFloat8Type(type);
}

inline bool isFloatingType(ScalarType type) {
    return type == ScalarType::Float32 || type == ScalarType::Float64 ||
           isReducedFloatingType(type);
}

inline bool isComplexType(ScalarType type) {
    return type == ScalarType::ComplexHalf ||
           type == ScalarType::ComplexFloat ||
           type == ScalarType::ComplexDouble ||
           type == ScalarType::BComplex32;
}

inline bool isFloatingOrComplexType(ScalarType type) {
    return isFloatingType(type) || isComplexType(type);
}

inline bool isUnsignedType(ScalarType type) {
    return type == ScalarType::UInt8 || type == ScalarType::UInt16 ||
           type == ScalarType::UInt32 || type == ScalarType::UInt64 ||
           type == ScalarType::QUInt8 || type == ScalarType::Float8_e8m0fnu;
}

inline bool isBooleanType(ScalarType type) {
    return type == ScalarType::Bool;
}

inline bool isSignedType(ScalarType type) {
    if (isQuantizedType(type)) {
        TP_THROW(TypeError, "isSignedType is not supported for quantized types");
    }
    switch (type) {
        case ScalarType::Int8:
        case ScalarType::Int16:
        case ScalarType::Int32:
        case ScalarType::Int64:
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::ComplexHalf:
        case ScalarType::ComplexFloat:
        case ScalarType::ComplexDouble:
        case ScalarType::BComplex32:
        case ScalarType::Float8_e4m3fn:
        case ScalarType::Float8_e5m2:
        case ScalarType::Float8_e4m3fnuz:
        case ScalarType::Float8_e5m2fnuz:
            return true;
        case ScalarType::UInt8:
        case ScalarType::UInt16:
        case ScalarType::UInt32:
        case ScalarType::UInt64:
        case ScalarType::Bool:
            return false;
        case ScalarType::Float8_e8m0fnu:
            return false;
        case ScalarType::QInt8:
        case ScalarType::QUInt8:
        case ScalarType::QInt32:
            break;
        case ScalarType::Undefined:
        case ScalarType::NumOptions:
            break;
    }
    TP_THROW(TypeError, "isSignedType is undefined for scalar type ",
             static_cast<int>(type));
}

inline ScalarType toUnderlyingStorageType(ScalarType type) {
    switch (type) {
        case ScalarType::QInt8:
            return ScalarType::Int8;
        case ScalarType::QUInt8:
            return ScalarType::UInt8;
        case ScalarType::QInt32:
            return ScalarType::Int32;
        default:
            return type;
    }
}

inline ScalarType toUnderlying(ScalarType type) {
    return toUnderlyingStorageType(type);
}

inline bool isUnderlying(ScalarType type, ScalarType quantized_type) {
    return type == toUnderlyingStorageType(quantized_type);
}

inline ScalarType toQIntType(ScalarType type) {
    switch (type) {
        case ScalarType::UInt8:
            return ScalarType::QUInt8;
        case ScalarType::Int8:
            return ScalarType::QInt8;
        case ScalarType::Int32:
            return ScalarType::QInt32;
        default:
            return type;
    }
}

inline constexpr bool isBarebonesUnsignedType(ScalarType type) {
    return type == ScalarType::UInt16 || type == ScalarType::UInt32 ||
           type == ScalarType::UInt64;
}

inline ScalarType toRealValueType(ScalarType type) {
    switch (type) {
        case ScalarType::ComplexHalf:
            return ScalarType::Float16;
        case ScalarType::ComplexFloat:
            return ScalarType::Float32;
        case ScalarType::ComplexDouble:
            return ScalarType::Float64;
        case ScalarType::BComplex32:
            return ScalarType::BFloat16;
        default:
            return type;
    }
}

inline ScalarType toComplexType(ScalarType type) {
    switch (type) {
        case ScalarType::Float16:
            return ScalarType::ComplexHalf;
        case ScalarType::Float32:
            return ScalarType::ComplexFloat;
        case ScalarType::Float64:
            return ScalarType::ComplexDouble;
        case ScalarType::BFloat16:
            return ScalarType::BComplex32;
        case ScalarType::ComplexHalf:
        case ScalarType::ComplexFloat:
        case ScalarType::ComplexDouble:
        case ScalarType::BComplex32:
            return type;
        default:
            return ScalarType::Undefined;
    }
}

template <typename T>
struct is_complex_type : std::false_type {};

template <typename T>
struct is_complex_type<std::complex<T>> : std::true_type {
    using value_type = T;
};

template <typename T>
inline constexpr bool is_complex_type_v =
    is_complex_type<std::remove_cv_t<std::remove_reference_t<T>>>::value;

#define TENSORPLAY_FORALL_COMPLEX_TYPES(_)         \
    _(std::complex<tensorplay::Half>, ComplexHalf) \
    _(std::complex<float>, ComplexFloat)           \
    _(std::complex<double>, ComplexDouble)         \
    _(std::complex<tensorplay::BFloat16>, BComplex32)

#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES(_)                  \
    TENSORPLAY_FORALL_COMPLEX_TYPES(_)

#define TENSORPLAY_FORALL_FP8_TYPES(_)       \
    _(tensorplay::Float8_e4m3fn, Float8_e4m3fn)   \
    _(tensorplay::Float8_e5m2, Float8_e5m2)       \
    _(tensorplay::Float8_e4m3fnuz, Float8_e4m3fnuz) \
    _(tensorplay::Float8_e5m2fnuz, Float8_e5m2fnuz) \
    _(tensorplay::Float8_e8m0fnu, Float8_e8m0fnu)

#define TENSORPLAY_FORALL_FLOAT8_TYPES(_) TENSORPLAY_FORALL_FP8_TYPES(_)

#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_)             \
    TENSORPLAY_FORALL_FP8_TYPES(_)

#define TENSORPLAY_FORALL_QINT_TYPES(_) \
    _(int8_t, QInt8)                   \
    _(uint8_t, QUInt8)                 \
    _(int32_t, QInt32)

#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(_)       \
    TENSORPLAY_FORALL_QINT_TYPES(_)

inline const char* toString(ScalarType type) {
#define TENSORPLAY_DTYPE_NAME(cpp_type, scalar_name) \
    case ScalarType::scalar_name: return #scalar_name;
    switch (type) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(
            TENSORPLAY_DTYPE_NAME)
        TENSORPLAY_FORALL_QINT_TYPES(TENSORPLAY_DTYPE_NAME)
        case ScalarType::Undefined:
            return "Undefined";
        default:
            return "UNKNOWN_SCALAR";
    }
#undef TENSORPLAY_DTYPE_NAME
}

inline size_t elementSize(ScalarType type) {
#define TENSORPLAY_DTYPE_SIZE(cpp_type, scalar_name) \
    case ScalarType::scalar_name: return sizeof(cpp_type);
    switch (type) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(
            TENSORPLAY_DTYPE_SIZE)
        default:
            return 0;
    }
#undef TENSORPLAY_DTYPE_SIZE
}

inline ScalarType opaqueScalarType(ScalarType type) {
    switch (elementSize(type)) {
        case 1:
            return ScalarType::UInt8;
        case 2:
            return ScalarType::UInt16;
        case 4:
            return ScalarType::UInt32;
        case 8:
            return ScalarType::UInt64;
        case 16:
            return ScalarType::ComplexDouble;
        default:
            return ScalarType::Undefined;
    }
}

inline bool canCast(ScalarType from, ScalarType to) {
    if (isComplexType(from) && !isComplexType(to)) {
        return false;
    }
    if (isFloatingType(from) && isIntegralType(to, false)) {
        return false;
    }
    if (from != ScalarType::Bool && to == ScalarType::Bool) {
        return false;
    }
    return true;
}

inline constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

}  // namespace tensorplay

#include "TypeTraits.h"
