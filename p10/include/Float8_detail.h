#pragma once

#include "Half.h"

#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TP_F8_HOST_DEVICE __host__ __device__
#else
#define TP_F8_HOST_DEVICE
#endif

namespace tensorplay {
namespace detail {

inline TP_F8_HOST_DEVICE uint32_t fp32_to_bits(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(value);
#else
    return TP_BIT_CAST(uint32_t, value);
#endif
}

inline TP_F8_HOST_DEVICE float fp32_from_bits(uint32_t bits) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(bits);
#else
    return TP_BIT_CAST(float, bits);
#endif
}

inline TP_F8_HOST_DEVICE uint32_t count_leading_zeros(uint32_t value) {
    if (value == 0) {
        return sizeof(uint32_t) * 8;
    }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __clz(value);
#elif defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanReverse(&index, static_cast<unsigned long>(value));
    return static_cast<uint32_t>(index) ^ 31u;
#else
    return static_cast<uint32_t>(__builtin_clz(value));
#endif
}

inline TP_F8_HOST_DEVICE float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;

    if (exponent == 0) {
        if (mantissa == 0) {
            return fp32_from_bits(sign);
        }
        uint32_t exponent_bits = 127 - 15 + 1;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent_bits;
        }
        mantissa &= 0x3ffu;
        return fp32_from_bits(
            sign | (exponent_bits << 23) | (mantissa << 13));
    }

    if (exponent == 0x1fu) {
        if (mantissa != 0) {
            mantissa |= 0x200u;
        }
        return fp32_from_bits(sign | 0x7f800000u | (mantissa << 13));
    }

    return fp32_from_bits(
        sign | ((exponent - 15 + 127) << 23) | (mantissa << 13));
}

template <typename T>
struct is_float8 : std::false_type {};

template <typename T>
inline constexpr bool is_float8_v = is_float8<std::decay_t<T>>::value;

}  // namespace detail

inline TP_F8_HOST_DEVICE uint32_t fp8_fp32_to_bits(float value) {
    return detail::fp32_to_bits(value);
}

inline TP_F8_HOST_DEVICE float fp8_bits_to_fp32(uint32_t bits) {
    return detail::fp32_from_bits(bits);
}

}  // namespace tensorplay

#define TP_F8_DEFINE_ARITHMETIC(Type)                                         \
    inline TP_F8_HOST_DEVICE Type operator-(const Type& value) {              \
        return Type(-static_cast<float>(value));                              \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator+(const Type& a, const Type& b) {   \
        return Type(static_cast<float>(a) + static_cast<float>(b));            \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator-(const Type& a, const Type& b) {   \
        return Type(static_cast<float>(a) - static_cast<float>(b));            \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator*(const Type& a, const Type& b) {   \
        return Type(static_cast<float>(a) * static_cast<float>(b));            \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator/(const Type& a, const Type& b) {   \
        return Type(static_cast<float>(a) / static_cast<float>(b));            \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type& operator+=(Type& a, const Type& b) {        \
        a = a + b;                                                             \
        return a;                                                              \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type& operator-=(Type& a, const Type& b) {        \
        a = a - b;                                                             \
        return a;                                                              \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type& operator*=(Type& a, const Type& b) {        \
        a = a * b;                                                             \
        return a;                                                              \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type& operator/=(Type& a, const Type& b) {        \
        a = a / b;                                                             \
        return a;                                                              \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator+(Type a, float b) {                \
        return static_cast<float>(a) + b;                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator-(Type a, float b) {                \
        return static_cast<float>(a) - b;                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator*(Type a, float b) {                \
        return static_cast<float>(a) * b;                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator/(Type a, float b) {                \
        return static_cast<float>(a) / b;                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator+(float a, Type b) {                \
        return a + static_cast<float>(b);                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator-(float a, Type b) {                \
        return a - static_cast<float>(b);                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator*(float a, Type b) {                \
        return a * static_cast<float>(b);                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float operator/(float a, Type b) {                \
        return a / static_cast<float>(b);                                      \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float& operator+=(float& a, const Type& b) {      \
        return a += static_cast<float>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float& operator-=(float& a, const Type& b) {      \
        return a -= static_cast<float>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float& operator*=(float& a, const Type& b) {      \
        return a *= static_cast<float>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE float& operator/=(float& a, const Type& b) {      \
        return a /= static_cast<float>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator+(Type a, double b) {              \
        return static_cast<double>(a) + b;                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator-(Type a, double b) {              \
        return static_cast<double>(a) - b;                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator*(Type a, double b) {              \
        return static_cast<double>(a) * b;                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator/(Type a, double b) {              \
        return static_cast<double>(a) / b;                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator+(double a, Type b) {              \
        return a + static_cast<double>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator-(double a, Type b) {              \
        return a - static_cast<double>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator*(double a, Type b) {              \
        return a * static_cast<double>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE double operator/(double a, Type b) {              \
        return a / static_cast<double>(b);                                     \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator+(Type a, int b) {                   \
        return a + Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator-(Type a, int b) {                   \
        return a - Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator*(Type a, int b) {                   \
        return a * Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator/(Type a, int b) {                   \
        return a / Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator+(int a, Type b) {                   \
        return Type(static_cast<float>(a)) + b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator-(int a, Type b) {                   \
        return Type(static_cast<float>(a)) - b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator*(int a, Type b) {                   \
        return Type(static_cast<float>(a)) * b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator/(int a, Type b) {                   \
        return Type(static_cast<float>(a)) / b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator+(Type a, int64_t b) {               \
        return a + Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator-(Type a, int64_t b) {               \
        return a - Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator*(Type a, int64_t b) {               \
        return a * Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator/(Type a, int64_t b) {               \
        return a / Type(static_cast<float>(b));                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator+(int64_t a, Type b) {               \
        return Type(static_cast<float>(a)) + b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator-(int64_t a, Type b) {               \
        return Type(static_cast<float>(a)) - b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator*(int64_t a, Type b) {               \
        return Type(static_cast<float>(a)) * b;                               \
    }                                                                          \
    inline TP_F8_HOST_DEVICE Type operator/(int64_t a, Type b) {               \
        return Type(static_cast<float>(a)) / b;                                \
    }
