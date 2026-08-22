#pragma once
#include <cstdint>
#include <type_traits>
#include <complex>
#include <string>

#include "Half.h"
#include "BFloat16.h"
#include "Float8.h"

// MSVC workaround: Avoid macro expansion issues in enum definition
// Undefine common Windows/System macros that might conflict with our enum names
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
#ifdef small
#undef small
#endif

namespace tensorplay {

// Macro to define all supported scalar types
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

// ScalarType enum
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
    Undefined,
    NumOptions
};

inline bool isIntegralType(ScalarType t, bool includeBool = false) {
    bool isIntegral = (t == ScalarType::UInt8 || t == ScalarType::Int8 ||
                      t == ScalarType::Int16 || t == ScalarType::Int32 ||
                      t == ScalarType::Int64 || t == ScalarType::UInt16 ||
                      t == ScalarType::UInt32 || t == ScalarType::UInt64);
    
    return isIntegral || (includeBool && t == ScalarType::Bool);
}

inline bool isFloatingType(ScalarType t) {
    return (t == ScalarType::Float32 || t == ScalarType::Float64 ||
            t == ScalarType::Float16 || t == ScalarType::BFloat16);
}

inline bool isReducedFloatingType(ScalarType t) {
    return t == ScalarType::Float16 || t == ScalarType::BFloat16;
}

inline bool isComplexType(ScalarType t) {
    return t == ScalarType::ComplexHalf ||
           t == ScalarType::ComplexFloat ||
           t == ScalarType::ComplexDouble ||
           t == ScalarType::BComplex32;
}

inline bool isFloatingOrComplexType(ScalarType t) {
    return isFloatingType(t) || isComplexType(t);
}

inline bool isSignedType(ScalarType t) {
    switch (t) {
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
            return true;
        default:
            return false;
    }
}

inline ScalarType toRealValueType(ScalarType t) {
    switch (t) {
        case ScalarType::ComplexHalf: return ScalarType::Float16;
        case ScalarType::ComplexFloat: return ScalarType::Float32;
        case ScalarType::ComplexDouble: return ScalarType::Float64;
        case ScalarType::BComplex32: return ScalarType::BFloat16;
        default: return t;
    }
}

inline ScalarType toComplexType(ScalarType t) {
    switch (t) {
        case ScalarType::Float16: return ScalarType::ComplexHalf;
        case ScalarType::Float32: return ScalarType::ComplexFloat;
        case ScalarType::Float64: return ScalarType::ComplexDouble;
        case ScalarType::BFloat16: return ScalarType::BComplex32;
        case ScalarType::ComplexHalf:
        case ScalarType::ComplexFloat:
        case ScalarType::ComplexDouble:
        case ScalarType::BComplex32:
            return t;
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
inline constexpr bool is_complex_type_v = is_complex_type<T>::value;

// Extended macro including complex types
#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES(_)                  \
    _(std::complex<tensorplay::Half>, ComplexHalf)      \
    _(std::complex<float>, ComplexFloat)               \
    _(std::complex<double>, ComplexDouble)             \
    _(std::complex<tensorplay::BFloat16>, BComplex32)

// Float8 family: opt-in tier -- only conversion/copy/item paths dispatch on
// these initially (mirrors ATen keeping float8 out of the generic kernel
// lists until per-op support lands).
#define TENSORPLAY_FORALL_FP8_TYPES(_) \
    _(tensorplay::Float8_e4m3fn, Float8_e4m3fn)   \
    _(tensorplay::Float8_e5m2, Float8_e5m2)

#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_)             \
    TENSORPLAY_FORALL_FP8_TYPES(_)

// DType is an alias for ScalarType for compatibility
using DType = ScalarType;

// Type traits for mapping C++ types to ScalarType
template <typename T>
struct TypeTraits {
    static constexpr ScalarType scalar_type = ScalarType::Undefined;
};

#define SPECIALIZE_TYPE_TRAITS(ctype, name)                              \
    template <>                                                          \
    struct TypeTraits<ctype> {                                           \
        static constexpr ScalarType scalar_type = ScalarType::name;      \
        static constexpr ScalarType dtype = ScalarType::name;            \
    };
TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(SPECIALIZE_TYPE_TRAITS)
#undef SPECIALIZE_TYPE_TRAITS

// Helper functions for ScalarType
inline const char* toString(ScalarType t) {
#define DEFINE_CASE(ctype, name) \
    case ScalarType::name:       \
        return #name;

    switch (t) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
        case ScalarType::Undefined:
            return "Undefined";
        default:
            return "UNKNOWN_SCALAR";
    }
#undef DEFINE_CASE
}

inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE(ctype, name) \
    case ScalarType::name:            \
        return sizeof(ctype);

    switch (t) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(CASE_ELEMENTSIZE)
        default:
            return 0;
    }
#undef CASE_ELEMENTSIZE
}


// Check if a C++ type is compatible with a DType
template <typename T>
inline bool is_compatible_with(ScalarType t) {
    return TypeTraits<T>::dtype == t;
}

} // namespace tensorplay
