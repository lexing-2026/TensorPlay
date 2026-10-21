#pragma once

// The crossing between the reduced width and single precision.
//
// A formula at the reduced width is computed in single precision and stored
// at the reduced width, so every step of the formula crosses between the two
// widths.  Done in software that crossing costs a handful of instructions
// per element and stands in the way of the walk vectorizing; the hardware
// conversion does the same crossing in one instruction.
//
// The hardware rounding and the software rounding agree: both round to
// nearest with ties to even, both carry a value too small for the layout
// to zero, and both leave an infinity or a not-a-number as one.  A
// not-a-number keeps its standing rather than its payload, which is what
// the reference runtime does as well.
//
// Design notes:
// * The crossings below are single translation-unit-local functions built
//   with GCC per-function target attributes ("avx2", "f16c"), so p10 keeps
//   compiling without ISA flags while the hot loops get real AVX2 codegen.
//   Dispatch happens once per call through f16c_available(); a machine
//   without the hardware conversion keeps the software crossing.
// * The layout is read and written as the sixteen bits it occupies, so a
//   walk needs no knowledge of the type that holds them.

#include "Half.h"
#include "cpu/vec/intrinsics.h"

#include <cstdint>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TP_VECHALF_X86 1
#endif

namespace tensorplay {
namespace cpu {
namespace vechalf {

#if defined(TP_VECHALF_X86)

inline bool f16c_available() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0 &&
                           __builtin_cpu_supports("f16c") != 0;
    return ok;
}

// The reduced layout, held in single precision: eight elements at a time.
__attribute__((target("avx2,f16c"), always_inline))
inline __m256 widen8(const uint16_t* p) {
    return _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// The single-precision answer, held in the reduced layout.
__attribute__((target("avx2,f16c"), always_inline))
inline __m128i narrow8(__m256 v) {
    return _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// The narrowed answer, widened again: the shape of "the answer of the
// previous step, carried into the next one".
__attribute__((target("avx2,f16c"), always_inline))
inline __m256 carried(__m256 v) {
    return _mm256_cvtph_ps(narrow8(v));
}

#else

// A machine without the hardware conversion keeps the software crossing.
inline bool f16c_available() { return false; }

#endif  // TP_VECHALF_X86

// One element, crossing in software.  The tail of a walk -- the elements
// that do not fill a step -- is answered this way.
inline float widen_one(uint16_t bits) {
    return static_cast<float>(
        tensorplay::Half(bits, tensorplay::Half::from_bits()));
}

inline uint16_t narrow_one(float value) {
    return tensorplay::Half(value).x;
}

}  // namespace vechalf
}  // namespace cpu
}  // namespace tensorplay
