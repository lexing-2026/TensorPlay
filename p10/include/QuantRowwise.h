#pragma once
#include "Half.h"
#include <cstdint>

#if defined(__CUDACC__)
#define TP_QR_HD __host__ __device__
#else
#define TP_QR_HD
#endif
namespace tensorplay::quant {
TP_QR_HD inline float row_float(const uint8_t* p) {
    union { uint32_t bits; float value; } v;
    v.bits = static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8
           | static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
    return v.value;
}
TP_QR_HD inline float row_half(const uint8_t* p) {
    return static_cast<float>(Half(static_cast<uint16_t>(p[0] | static_cast<uint16_t>(p[1]) << 8), Half::from_bits()));
}
TP_QR_HD inline void put_float(uint8_t* p, float value) {
    union { uint32_t bits; float value; } v; v.value = value;
    for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v.bits >> (i * 8));
}
TP_QR_HD inline void put_half(uint8_t* p, float value) {
    const Half v(value); p[0] = static_cast<uint8_t>(v.x); p[1] = static_cast<uint8_t>(v.x >> 8);
}
TP_QR_HD inline void row_grid(const uint8_t* row, int64_t width, int bits, float& scale, float& bias) {
    const uint8_t* tail = row + width - (bits == 8 ? 8 : 4);
    scale = bits == 8 ? row_float(tail) : row_half(tail);
    bias = bits == 8 ? row_float(tail + 4) : row_half(tail + 2);
}
TP_QR_HD inline uint8_t row_code(const uint8_t* row, int64_t col, int bits) {
    return (row[col / (8 / bits)] >> ((col % (8 / bits)) * bits)) & ((1 << bits) - 1);
}
}
#undef TP_QR_HD
