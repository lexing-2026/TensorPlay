#pragma once

#include "Float8_detail.h"

namespace tensorplay {
namespace detail {

inline TP_F8_HOST_DEVICE uint8_t fp8e8m0fnu_from_fp32_value(float value) {
    const uint32_t bits = fp32_to_bits(value);
    uint32_t exponent = (bits >> 23) & UINT32_C(0xff);

    if (exponent == UINT32_C(0xff)) {
        return UINT8_C(0xff);
    }

    const uint8_t guard = (bits & UINT32_C(0x400000)) != 0;
    const uint8_t round = (bits & UINT32_C(0x200000)) != 0;
    const uint8_t sticky = (bits & UINT32_C(0x1fffff)) != 0;
    const uint8_t least_bit = exponent != 0;

    if (guard && (round || sticky || least_bit)) {
        ++exponent;
    }
    return static_cast<uint8_t>(exponent);
}

}  // namespace detail

using detail::fp8e8m0fnu_from_fp32_value;

struct alignas(1) Float8_e8m0fnu {
    union {
        uint8_t x;
        uint8_t x_;
    };

    struct from_bits_t {};

    static constexpr TP_F8_HOST_DEVICE from_bits_t from_bits() {
        return from_bits_t();
    }

    Float8_e8m0fnu() = default;

    constexpr TP_F8_HOST_DEVICE Float8_e8m0fnu(uint8_t bits, from_bits_t)
        : x(bits) {}

    static constexpr TP_F8_HOST_DEVICE Float8_e8m0fnu from_bits(uint8_t bits) {
        return Float8_e8m0fnu(bits, from_bits());
    }

    inline TP_F8_HOST_DEVICE Float8_e8m0fnu(float value)
        : x(detail::fp8e8m0fnu_from_fp32_value(value)) {}

    inline TP_F8_HOST_DEVICE operator float() const {
        if (x == 0) {
            return detail::fp32_from_bits(UINT32_C(0x00400000));
        }
        if (isnan()) {
            return detail::fp32_from_bits(UINT32_C(0x7f800001));
        }
        return detail::fp32_from_bits(static_cast<uint32_t>(x) << 23);
    }

    inline TP_F8_HOST_DEVICE bool isnan() const {
        return x == UINT8_C(0xff);
    }
};

namespace detail {
template <>
struct is_float8<Float8_e8m0fnu> : std::true_type {};
}  // namespace detail

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e8m0fnu& value) {
    out << static_cast<float>(value);
    return out;
}

}  // namespace tensorplay

namespace std {

template <>
class numeric_limits<tensorplay::Float8_e8m0fnu> {
 public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr auto has_denorm = false;
    static constexpr auto has_denorm_loss = false;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 1;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 1;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -126;
    static constexpr int min_exponent10 = -38;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = false;

    static constexpr tensorplay::Float8_e8m0fnu min() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0x00), tensorplay::Float8_e8m0fnu::from_bits());
    }
    static constexpr tensorplay::Float8_e8m0fnu lowest() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0x00), tensorplay::Float8_e8m0fnu::from_bits());
    }
    static constexpr tensorplay::Float8_e8m0fnu max() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0xfe), tensorplay::Float8_e8m0fnu::from_bits());
    }
    static constexpr tensorplay::Float8_e8m0fnu epsilon() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0x7f), tensorplay::Float8_e8m0fnu::from_bits());
    }
    static constexpr tensorplay::Float8_e8m0fnu round_error() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0x7e), tensorplay::Float8_e8m0fnu::from_bits());
    }
    static constexpr tensorplay::Float8_e8m0fnu quiet_NaN() {
        return tensorplay::Float8_e8m0fnu(
            UINT8_C(0xff), tensorplay::Float8_e8m0fnu::from_bits());
    }
};

}  // namespace std
