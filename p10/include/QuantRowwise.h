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
#ifdef __CUDA_ARCH__
    const uint32_t original=__float_as_uint(value),doubled=original+original;
    float base=__fmul_rn(__fmul_rn(::fabsf(value),__uint_as_float(239u<<23)),__uint_as_float(17u<<23));
    uint32_t bias=doubled & 0xff000000u;
    if (bias<0x71000000u) bias=0x71000000u;
    base=__fadd_rn(__uint_as_float((bias>>1)+0x07800000u),base);
    const uint32_t rounded=__float_as_uint(base);
    const uint32_t bits=(original>>16 & 0x8000u) | (doubled>0xff000000u ? 0x7e00u
        : ((rounded>>13 & 0x7c00u)+(rounded & 0xfffu)));
#else
    const uint32_t bits=Half(value).x;
#endif
    p[0]=static_cast<uint8_t>(bits & 255u);
    p[1]=static_cast<uint8_t>(bits >> 8);
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
