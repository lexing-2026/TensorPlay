#pragma once

// The crossing between a reduced width and single precision.
//
// A formula at a reduced width is computed in single precision and stored at
// the reduced width, so every step of the formula crosses between the two
// widths.  Done in software that crossing costs a handful of instructions per
// element and stands in the way of the walk vectorizing; the hardware
// conversion does the same crossing in one instruction.
//
// The hardware rounding and the software rounding agree: both round to
// nearest with ties to even, both carry a value too small for the layout to
// zero, and both leave an infinity or a not-a-number as one.  A
// not-a-number keeps its standing rather than its payload.
//
// Design notes:
// * The crossings below are single translation-unit-local functions built
//   with GCC per-function target attributes ("avx2", "f16c"), so p10 keeps
//   compiling without ISA flags while the hot loops get real AVX2 codegen.
//   Dispatch happens once per call through f16c_available(); a machine
//   without the hardware conversion keeps the software crossing.
// * The layout is read and written as the sixteen bits it occupies, so a
//   walk needs no knowledge of the type that holds them.
// * The two reduced widths cross differently.  The half layout crosses
//   through the hardware conversion; the other one is a shift of the
//   single-precision pattern, and its rounding to nearest with ties to even
//   is the same shift with the tie broken by the bit that is about to be
//   dropped.

#include "BFloat16.h"
#include "Half.h"
#include "cpu/vec/intrinsics.h"

#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TP_VECREDUCED_X86 1
#endif

namespace tensorplay {
namespace cpu {
namespace vecreduced {

#if defined(TP_VECREDUCED_X86)

inline bool f16c_available() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0 &&
                           __builtin_cpu_supports("f16c") != 0;
    return ok;
}

// The half layout, held in single precision: eight elements at a time.
__attribute__((target("avx2,f16c"), always_inline))
inline __m256 widen8(const uint16_t* p) {
    return _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// The single-precision answer, held in the half layout.
__attribute__((target("avx2,f16c"), always_inline))
inline __m128i narrow8(__m256 v) {
    return _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// The other reduced width, held in single precision: eight elements at a
// time.  The layout is the upper half of the single-precision pattern.
__attribute__((target("avx2"), always_inline))
inline __m256 widen_bf16_8(const uint16_t* p) {
    // The pattern is moved into the upper half of the single-precision
    // pattern; it is a move of bits, not a number to be converted.
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p))), 16));
}

// The eight answers held in `v`, gathered into the low half of the result.
__attribute__((target("avx2"), always_inline))
inline __m256i narrow_bf16_8(__m256 v) {
    const __m256i bits = _mm256_castps_si256(v);
    const __m256i tie = _mm256_and_si256(
        _mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    const __m256i rounded = _mm256_add_epi32(
        _mm256_add_epi32(bits, _mm256_set1_epi32(0x7FFF)), tie);
    // A not-a-number keeps its standing instead of being rounded: a payload
    // that fills the low bits would carry into the exponent and sign and
    // land on a number of some order.
    const __m256i ordered =
        _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_ORD_Q));
    return _mm256_blendv_epi8(_mm256_set1_epi32(0x7FC0),
                              _mm256_srli_epi32(rounded, 16), ordered);
}

// The single-precision answer, held in the other reduced width.  The
// rounding is to nearest with ties to even: the bit that is about to be
// dropped breaks the tie.
__attribute__((target("avx2"), always_inline))
inline void store_bf16_8(uint16_t* p, __m256 v) {
    // The answer is the upper half of each lane.  Moving those halves down
    // leaves numbers that are not signed, so the packing that gathers them
    // does not read a negative answer as a number to be held in range.
    const __m256i packed = _mm256_packus_epi32(
        narrow_bf16_8(v), _mm256_setzero_si256());
    // The packing spreads the answers over the two halves of the register;
    // the halves are put back in order, so the answers are the eight
    // sixteen-bit words that follow one another.
    _mm_storeu_si128(
        reinterpret_cast<__m128i*>(p),
        _mm256_castsi256_si128(
            _mm256_permute4x64_epi64(packed, 0x08)));
}

// Sixteen answers landed by one store: the two packs are gathered into the
// quarters of one register, each pack keeping its own order.
__attribute__((target("avx2"), always_inline))
inline void store_bf16_16(uint16_t* p, __m256 lo, __m256 hi) {
    const __m256i packed = _mm256_packus_epi32(narrow_bf16_8(lo),
                                               narrow_bf16_8(hi));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(p),
        _mm256_permute4x64_epi64(packed, 0xD8));
}

#else

// A machine without the hardware conversion keeps the software crossing.
inline bool f16c_available() { return false; }

#endif  // TP_VECREDUCED_X86

// One element, crossing in software.  The tail of a walk -- the elements
// that do not fill a step -- is answered this way.
inline float widen_one(uint16_t bits) {
    return static_cast<float>(
        tensorplay::Half(bits, tensorplay::Half::from_bits()));
}

inline uint16_t narrow_one(float value) {
    return tensorplay::Half(value).x;
}

inline float widen_bf16_one(uint16_t bits) {
    return tensorplay::detail::bfloat16_to_float_bits(bits);
}

inline uint16_t narrow_bf16_one(float value) {
    return tensorplay::detail::float_to_bfloat16_bits(value);
}

}  // namespace vecreduced
}  // namespace cpu
}  // namespace tensorplay
