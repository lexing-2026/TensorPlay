#pragma once

#include "Float8_detail.h"

namespace tensorplay {
namespace detail {

template <uint32_t exponent_width, uint32_t mantissa_width>
inline TP_F8_HOST_DEVICE float fp8_fnuz_to_fp32_value(uint8_t value) {
    static_assert(
        (exponent_width == 4 && mantissa_width == 3) ||
        (exponent_width == 5 && mantissa_width == 2));

    constexpr uint32_t output_exponent_width = 8;
    constexpr uint32_t output_mantissa_width = 23;

    if (value == 0) {
        return 0.0f;
    }
    if (value == 0x80) {
        return fp32_from_bits(0x7f800001u);
    }

    uint32_t mantissa = value & ((1u << mantissa_width) - 1u);
    uint32_t exponent = (value & 0x7fu) >> mantissa_width;

    if (exponent == 0) {
        uint32_t renorm_shift = count_leading_zeros(mantissa);
        uint32_t shift = 1u + renorm_shift - (32u - mantissa_width);
        mantissa <<= shift;
        exponent += 1u - shift;
        mantissa &= (1u << mantissa_width) - 1u;
    }

    const uint32_t exponent_offset =
        (1u << (output_exponent_width - 1u)) -
        (1u << (exponent_width - 1u));
    exponent += exponent_offset - 1u;
    mantissa <<= output_mantissa_width - mantissa_width;

    const uint32_t sign = value >> 7;
    return fp32_from_bits(
        (sign << 31) | (exponent << 23) | mantissa);
}

}  // namespace detail
}  // namespace tensorplay
