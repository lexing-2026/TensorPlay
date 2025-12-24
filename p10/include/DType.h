#pragma once
#include <cstdint>
#include <type_traits>
#include <complex>
#include <string>

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
    Bool,
    ComplexFloat,
    ComplexDouble,
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
    return (t == ScalarType::Float32 || t == ScalarType::Float64 || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble);
}

// Extended macro including complex types
#define TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
    TENSORPLAY_FORALL_SCALAR_TYPES(_)                  \
    _(std::complex<float>, ComplexFloat)               \
    _(std::complex<double>, ComplexDouble)

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
