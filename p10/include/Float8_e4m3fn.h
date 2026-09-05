#pragma once

#include "Float8_detail.h"

namespace tensorplay {
namespace detail {

inline TP_F8_HOST_DEVICE float fp8e4m3fn_to_fp32_value(uint8_t input) {
    const uint32_t word = static_cast<uint32_t>(input) << 24;
    const uint32_t sign = word & UINT32_C(0x80000000);
    const uint32_t nonsign = word & UINT32_C(0x7fffffff);

    uint32_t renorm_shift =
        nonsign != 0 ? count_leading_zeros(nonsign) : sizeof(uint32_t) * 8;
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

    const int32_t inf_nan_mask =
        ((static_cast<int32_t>(nonsign + UINT32_C(0x01000000)) >> 8) &
         INT32_C(0x7f800000));
    const int32_t zero_mask =
        static_cast<int32_t>(nonsign - 1) >> 31;
    const uint32_t result =
        sign |
        ((((nonsign << renorm_shift >> 4) +
           ((UINT32_C(0x78) - renorm_shift) << 23)) |
          static_cast<uint32_t>(inf_nan_mask)) &
         ~static_cast<uint32_t>(zero_mask));
    return fp32_from_bits(result);
}

inline TP_F8_HOST_DEVICE uint8_t fp8e4m3fn_from_fp32_value(float value) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

    uint32_t f_bits = fp32_to_bits(value);
    uint8_t result = 0;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;

    if (f_bits >= fp8_max) {
        result = f_bits > UINT32_C(0x7f800000) ? UINT8_C(0x7f)
                                               : UINT8_C(0x7e);
    } else if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = fp32_to_bits(
            fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
        const uint8_t mantissa_odd = static_cast<uint8_t>((f_bits >> 20) & 1);
        f_bits += ((UINT32_C(7) - UINT32_C(127)) << 23) + UINT32_C(0x7ffff);
        f_bits += mantissa_odd;
        result = static_cast<uint8_t>(f_bits >> 20);
        if (result == UINT8_C(0x7f)) {
            result = UINT8_C(0x7e);
        }
    }

    return result | static_cast<uint8_t>(sign >> 24);
}

}  // namespace detail

using detail::fp8e4m3fn_from_fp32_value;
using detail::fp8e4m3fn_to_fp32_value;

struct alignas(1) Float8_e4m3fn {
    union {
        uint8_t x;
        uint8_t x_;
    };

    struct from_bits_t {};

    static constexpr TP_F8_HOST_DEVICE from_bits_t from_bits() {
        return from_bits_t();
    }

    Float8_e4m3fn() = default;

    constexpr TP_F8_HOST_DEVICE Float8_e4m3fn(
        uint8_t bits,
        from_bits_t)
        : x(bits) {}

    static constexpr TP_F8_HOST_DEVICE Float8_e4m3fn from_bits(uint8_t bits) {
        return Float8_e4m3fn(bits, from_bits());
    }

    inline TP_F8_HOST_DEVICE Float8_e4m3fn(float value)
        : x(detail::fp8e4m3fn_from_fp32_value(value)) {}

    inline TP_F8_HOST_DEVICE operator float() const {
        return detail::fp8e4m3fn_to_fp32_value(x);
    }

    inline TP_F8_HOST_DEVICE bool isnan() const {
        return (x & UINT8_C(0x7f)) == UINT8_C(0x7f);
    }

    inline TP_F8_HOST_DEVICE bool isinf() const {
        return false;
    }
};

namespace detail {
template <>
struct is_float8<Float8_e4m3fn> : std::true_type {};
}  // namespace detail

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e4m3fn& value) {
    out << static_cast<float>(value);
    return out;
}

TP_F8_DEFINE_ARITHMETIC(Float8_e4m3fn)

}  // namespace tensorplay

namespace std {

template <>
class numeric_limits<tensorplay::Float8_e4m3fn> {
 public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr auto has_denorm = true;
    static constexpr auto has_denorm_loss = true;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 4;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 3;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -5;
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 8;
    static constexpr int max_exponent10 = 2;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = false;

    static constexpr tensorplay::Float8_e4m3fn min() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x08), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn lowest() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0xfe), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn max() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x7e), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn epsilon() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x20), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn round_error() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x30), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn quiet_NaN() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x7f), tensorplay::Float8_e4m3fn::from_bits());
    }
    static constexpr tensorplay::Float8_e4m3fn denorm_min() {
        return tensorplay::Float8_e4m3fn(
            UINT8_C(0x01), tensorplay::Float8_e4m3fn::from_bits());
    }
};

}  // namespace std
