#pragma once

#include "Float8_fnuz_cvt.h"

namespace tensorplay {
namespace detail {

inline TP_F8_HOST_DEVICE uint8_t fp8e5m2fnuz_from_fp32_value(float value) {
    constexpr uint32_t fnuz_max = UINT32_C(0x8f) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

    uint32_t f_bits = fp32_to_bits(value);
    uint32_t result = 0;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;

    if (f_bits >= fnuz_max) {
        return UINT8_C(0x80);
    }

    if (f_bits < (UINT32_C(0x70) << 23)) {
        f_bits = fp32_to_bits(
            fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
        result = f_bits - denorm_mask;
        if (result == 0) {
            return 0;
        }
    } else {
        const uint8_t mantissa_odd = static_cast<uint8_t>((f_bits >> 21) & 1);
        f_bits += ((UINT32_C(16) - UINT32_C(127)) << 23) + UINT32_C(0xfffff);
        f_bits += mantissa_odd;
        result = static_cast<uint8_t>(f_bits >> 21);
    }

    return static_cast<uint8_t>(result | (sign >> 24));
}

}  // namespace detail

using detail::fp8e5m2fnuz_from_fp32_value;

struct alignas(1) Float8_e5m2fnuz {
    union {
        uint8_t x;
        uint8_t x_;
    };

    struct from_bits_t {};

    static constexpr TP_F8_HOST_DEVICE from_bits_t from_bits() {
        return from_bits_t();
    }

    Float8_e5m2fnuz() = default;

    constexpr TP_F8_HOST_DEVICE Float8_e5m2fnuz(uint8_t bits, from_bits_t)
        : x(bits) {}

    static constexpr TP_F8_HOST_DEVICE Float8_e5m2fnuz from_bits(uint8_t bits) {
        return Float8_e5m2fnuz(bits, from_bits());
    }

    inline TP_F8_HOST_DEVICE Float8_e5m2fnuz(float value)
        : x(detail::fp8e5m2fnuz_from_fp32_value(value)) {}

    inline TP_F8_HOST_DEVICE operator float() const {
        return detail::fp8_fnuz_to_fp32_value<5, 2>(x);
    }

    inline TP_F8_HOST_DEVICE bool isnan() const {
        return x == UINT8_C(0x80);
    }

    inline TP_F8_HOST_DEVICE bool isinf() const {
        return false;
    }
};

namespace detail {
template <>
struct is_float8<Float8_e5m2fnuz> : std::true_type {};
}  // namespace detail

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e5m2fnuz& value) {
    out << static_cast<float>(value);
    return out;
}

TP_F8_DEFINE_ARITHMETIC(Float8_e5m2fnuz)

}  // namespace tensorplay

namespace std {

template <>
class numeric_limits<tensorplay::Float8_e5m2fnuz> {
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
    static constexpr int digits = 3;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -14;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before =
        numeric_limits<float>::tinyness_before;

    static constexpr tensorplay::Float8_e5m2fnuz min() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x04), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz max() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x7f), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz lowest() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0xff), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz epsilon() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x34), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz round_error() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x38), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz infinity() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x80), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz quiet_NaN() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x80), tensorplay::Float8_e5m2fnuz::from_bits());
    }
    static constexpr tensorplay::Float8_e5m2fnuz denorm_min() {
        return tensorplay::Float8_e5m2fnuz(
            UINT8_C(0x01), tensorplay::Float8_e5m2fnuz::from_bits());
    }
};

}  // namespace std
