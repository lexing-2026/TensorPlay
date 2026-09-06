#pragma once

#include "DType.h"

#include <complex>
#include <type_traits>

namespace tensorplay {

template <typename T>
struct TypeTraits {
    static constexpr ScalarType scalar_type = ScalarType::Undefined;
    static constexpr ScalarType dtype = ScalarType::Undefined;
};

#define TENSORPLAY_DEFINE_TYPE_TRAIT(cpp_type, scalar_name)                  \
    template <>                                                              \
    struct TypeTraits<cpp_type> {                                             \
        static constexpr ScalarType scalar_type = ScalarType::scalar_name;   \
        static constexpr ScalarType dtype = ScalarType::scalar_name;         \
    };

TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(
    TENSORPLAY_DEFINE_TYPE_TRAIT)

#undef TENSORPLAY_DEFINE_TYPE_TRAIT

// The complex element types come from the list above, which already names
// them in this spelling; repeating them here would redefine the traits.

template <typename T>
struct TypeTraits<const T> : TypeTraits<T> {};

template <typename T>
struct TypeTraits<volatile T> : TypeTraits<T> {};

template <typename T>
struct TypeTraits<const volatile T> : TypeTraits<T> {};

template <typename T>
struct TypeTraits<T&> : TypeTraits<T> {};

template <typename T>
struct TypeTraits<T&&> : TypeTraits<T> {};

template <typename T>
struct CppTypeToScalarType
    : std::integral_constant<
          ScalarType,
          TypeTraits<std::remove_cv_t<std::remove_reference_t<T>>>::
              scalar_type> {};

template <typename T>
inline constexpr ScalarType CppTypeToScalarType_v =
    CppTypeToScalarType<T>::value;

template <typename T>
inline bool is_compatible_with(ScalarType type) {
    return CppTypeToScalarType<T>::value == type;
}

namespace detail {

template <typename T>
inline constexpr bool is_reduced_cpp_type_v =
    std::is_same_v<std::decay_t<T>, Half> ||
    std::is_same_v<std::decay_t<T>, BFloat16> || detail::is_float8_v<T>;

template <typename destination_t, typename source_t>
inline TP_F8_HOST_DEVICE destination_t scalar_cast(source_t value) {
    using destination_value_t = std::decay_t<destination_t>;
    using source_value_t = std::decay_t<source_t>;

    if constexpr (std::is_same_v<destination_value_t, source_value_t>) {
        return static_cast<destination_t>(value);
    } else if constexpr (is_complex_type_v<destination_value_t>) {
        using component_t = typename is_complex_type<destination_value_t>::value_type;
        if constexpr (is_complex_type_v<source_value_t>) {
            return destination_t(scalar_cast<component_t>(value.real()),
                                 scalar_cast<component_t>(value.imag()));
        } else {
            return destination_t(scalar_cast<component_t>(value), component_t(0));
        }
    } else if constexpr (is_complex_type_v<source_value_t>) {
        if constexpr (std::is_same_v<destination_value_t, bool>) {
            return static_cast<destination_t>(
                static_cast<bool>(value.real()) ||
                static_cast<bool>(value.imag()));
        } else {
            return scalar_cast<destination_t>(value.real());
        }
    } else if constexpr (is_reduced_cpp_type_v<destination_t>) {
        return destination_t(static_cast<float>(value));
    } else if constexpr (is_reduced_cpp_type_v<source_t>) {
        return static_cast<destination_t>(static_cast<float>(value));
    } else {
        return static_cast<destination_t>(value);
    }
}

}  // namespace detail

template <typename T>
struct is_float8 : detail::is_float8<std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_float8_v = is_float8<T>::value;

namespace impl {

template <ScalarType type>
struct ScalarTypeToCPPType;

#define TENSORPLAY_DEFINE_CPP_TYPE(cpp_type, scalar_name)                    \
    template <>                                                               \
    struct ScalarTypeToCPPType<ScalarType::scalar_name> {                     \
        using type = cpp_type;                                                \
    };

TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(
    TENSORPLAY_DEFINE_CPP_TYPE)

#undef TENSORPLAY_DEFINE_CPP_TYPE

template <>
struct ScalarTypeToCPPType<ScalarType::QInt8> {
    using type = int8_t;
};

template <>
struct ScalarTypeToCPPType<ScalarType::QUInt8> {
    using type = uint8_t;
};

template <>
struct ScalarTypeToCPPType<ScalarType::QInt32> {
    using type = int32_t;
};

template <ScalarType type>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<type>::type;

}  // namespace impl

template <ScalarType type>
using ScalarTypeToCPPType = impl::ScalarTypeToCPPType<type>;

template <ScalarType type>
using ScalarTypeToCPPTypeT = impl::ScalarTypeToCPPTypeT<type>;

template <ScalarType type>
using TypeTraitsOf = ScalarTypeToCPPType<type>;

#define TENSORPLAY_DEFINE_DTYPE_CONSTANT(cpp_type, scalar_name)              \
    inline constexpr DType k##scalar_name = DType::scalar_name;

TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(
    TENSORPLAY_DEFINE_DTYPE_CONSTANT)

#undef TENSORPLAY_DEFINE_DTYPE_CONSTANT

inline constexpr DType kQInt8 = DType::QInt8;
inline constexpr DType kQUInt8 = DType::QUInt8;
inline constexpr DType kQInt32 = DType::QInt32;
inline constexpr DType kUndefined = DType::Undefined;
inline constexpr DType kByte = DType::UInt8;
inline constexpr DType kChar = DType::Int8;
inline constexpr DType kShort = DType::Int16;
inline constexpr DType kInt = DType::Int32;
inline constexpr DType kLong = DType::Int64;
inline constexpr DType kHalf = DType::Float16;
inline constexpr DType kFloat = DType::Float32;
inline constexpr DType kDouble = DType::Float64;

}  // namespace tensorplay
