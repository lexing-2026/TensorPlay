#pragma once

// Vectorized fast paths for the pairwise float maps whose answer costs a
// library call per element: hypot, atan2 and the logarithmic sum.
//
// The scalar loop answers these in double precision whatever the tensor's
// own width is, so a Float32 answer pays for a double-precision call and
// then narrows -- the call is the whole cost of the map.  The paths below
// answer at the tensor's width, several elements at a time, through the
// vendored SLEEF vector math.
//
// Design notes:
// * Every kernel here is a translation-unit-local function built with GCC
//   per-function target attributes ("avx2", "fma"), so p10 keeps compiling
//   without ISA flags while the hot loops get real AVX2 codegen.  Dispatch
//   happens once per call through avx2_available(); below-AVX2 machines
//   transparently keep the scalar behaviour.
// * The formulas are the ones the scalar loops use, in the same order, so
//   the vector answer and the scalar answer agree element for element.
//   The logarithmic sum keeps the max-plus-log1p form: the larger operand
//   is the base and the smaller one is folded in through exp and log1p,
//   which is what keeps the sum from overflowing.
// * The vector maximum hands back the second operand when either side is
//   NaN, where the scalar maximum hands back the first.  Both answers are
//   NaN in the end -- the folded term carries the NaN across -- so the
//   difference does not reach the result.
// * Two equal infinities give a difference of NaN, which the vector fold
//   would turn into NaN where the answer is the infinity itself.  The base
//   is therefore selected back over the fold when both operands are the
//   same infinity.

#include "cpu/vec/intrinsics.h"
#include "cpu/VecReducedMath.h"
#include <cstdint>
#include <cmath>
#include <limits>

#include "cpu/vec/SleefShims.h"

#if defined(__x86_64__) || defined(__i386__)
#define TP_VECBINARY_X86 1

// The truncated remainder of two vectors.  The dispatcher picks the widest
// body the running CPU has; the answer itself is exact, so several operands
// share one call instead of one call per element.
#ifdef __cplusplus
extern "C" {
#endif
__m256 Sleef_fmodf8(__m256, __m256);
__m256d Sleef_fmodd4(__m256d, __m256d);
#ifdef __cplusplus
}
#endif
#endif

#if defined(TP_VECBINARY_X86) && defined(__x86_64__) \
    && (defined(__GNUC__) || defined(__clang__))
#define TP_VECBINARY_SLEEF 1
#endif

namespace tensorplay {
namespace cpu {
namespace vecbinary {

// Which pairwise map the chunk answers.
enum class BOp : int { Hypot, Atan2, LogAddExp, Remainder, Fmod };

// Two equal infinities cancel in the difference the fold takes, so the
// folded term would be a not-a-number there; the answer is the infinity
// itself.  Every other pair follows the folded form, not-a-number included.
inline float log_add_exp_value(float x, float y) {
    if (std::isinf(x) && x == y) return x;
    const float m = std::max(x, y);
    return m + std::log1p(std::exp(-std::fabs(x - y)));
}

inline double log_add_exp_value(double x, double y) {
    if (std::isinf(x) && x == y) return x;
    const double m = std::max(x, y);
    return m + std::log1p(std::exp(-std::fabs(x - y)));
}

// The answer of the map at the width of the operands, one element at a
// time.  It stands in for the tail of a chunk.
template <BOp op>
inline float scalar_f32_of(float x, float y);

template <>
inline float scalar_f32_of<BOp::Hypot>(float x, float y) {
    return std::hypot(x, y);
}

template <>
inline float scalar_f32_of<BOp::Atan2>(float x, float y) {
    return std::atan2(x, y);
}

template <>
inline float scalar_f32_of<BOp::LogAddExp>(float x, float y) {
    return log_add_exp_value(x, y);
}

template <>
inline float scalar_f32_of<BOp::Remainder>(float x, float y) {
    const float mod = std::fmod(x, y);
    return (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) ? mod + y : mod;
}

template <>
inline float scalar_f32_of<BOp::Fmod>(float x, float y) {
    return std::fmod(x, y);
}

inline bool avx2_available() {
#if defined(TP_VECBINARY_X86)
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
#else
    return false;
#endif
}

#if defined(TP_VECBINARY_SLEEF)

// The scalar answer of the same map, used for the tail of a chunk and for
// the lanes the vector fold cannot settle.
inline float scalar_f32(BOp op, float x, float y) {
    switch (op) {
        case BOp::Hypot:
            return std::hypot(x, y);
        case BOp::Atan2:
            return std::atan2(x, y);
        case BOp::LogAddExp:
            return log_add_exp_value(x, y);
        case BOp::Remainder: {
            const float mod = std::fmod(x, y);
            return (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) ? mod + y : mod;
        }
        case BOp::Fmod:
            return std::fmod(x, y);
    }
    return 0.0f;
}

inline double scalar_f64(BOp op, double x, double y) {
    switch (op) {
        case BOp::Hypot:
            return std::hypot(x, y);
        case BOp::Atan2:
            return std::atan2(x, y);
        case BOp::LogAddExp:
            return log_add_exp_value(x, y);
        case BOp::Remainder: {
            const double mod = std::fmod(x, y);
            return (mod != 0.0 && ((y < 0.0) != (mod < 0.0))) ? mod + y : mod;
        }
        case BOp::Fmod:
            return std::fmod(x, y);
    }
    return 0.0;
}

// The answer at the reduced width: the operands are widened, the map is
// answered in single precision, and the answer is narrowed once.  This is
// what the flat loop answers for a reduced-width pair, which evaluates in
// single precision and narrows once at the end.
inline uint16_t scalar_f16(BOp op, uint16_t a, uint16_t b) {
    const float x = tensorplay::cpu::vecreduced::widen_one(a);
    const float y = tensorplay::cpu::vecreduced::widen_one(b);
    switch (op) {
        case BOp::Hypot:
            return tensorplay::cpu::vecreduced::narrow_one(std::hypot(x, y));
        case BOp::Atan2:
            return tensorplay::cpu::vecreduced::narrow_one(std::atan2(x, y));
        case BOp::LogAddExp:
            return tensorplay::cpu::vecreduced::narrow_one(
                log_add_exp_value(x, y));
        case BOp::Remainder: {
            const float mod = std::fmod(x, y);
            const float ans =
                (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) ? mod + y : mod;
            return tensorplay::cpu::vecreduced::narrow_one(ans);
        }
        case BOp::Fmod:
            return tensorplay::cpu::vecreduced::narrow_one(std::fmod(x, y));
    }
    return 0;
}

// The other reduced width answers the same way; only the crossing differs.
inline uint16_t scalar_bf16(BOp op, uint16_t a, uint16_t b) {
    const float x = tensorplay::cpu::vecreduced::widen_bf16_one(a);
    const float y = tensorplay::cpu::vecreduced::widen_bf16_one(b);
    switch (op) {
        case BOp::Hypot:
            return tensorplay::cpu::vecreduced::narrow_bf16_one(
                std::hypot(x, y));
        case BOp::Atan2:
            return tensorplay::cpu::vecreduced::narrow_bf16_one(
                std::atan2(x, y));
        case BOp::LogAddExp:
            return tensorplay::cpu::vecreduced::narrow_bf16_one(
                log_add_exp_value(x, y));
        case BOp::Remainder: {
            const float mod = std::fmod(x, y);
            const float ans =
                (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) ? mod + y : mod;
            return tensorplay::cpu::vecreduced::narrow_bf16_one(ans);
        }
        case BOp::Fmod:
            return tensorplay::cpu::vecreduced::narrow_bf16_one(
                std::fmod(x, y));
    }
    return 0;
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 log_add_exp_f32(__m256 x, __m256 y) {
    const __m256 signbit = _mm256_set1_ps(-0.0f);
    const __m256 inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 m = _mm256_max_ps(x, y);
    const __m256 diff = _mm256_andnot_ps(signbit, _mm256_sub_ps(x, y));
    // Flipping the sign bit is the same answer as subtracting from zero and
    // one instruction shorter on the chain that feeds the exponential.
    const __m256 folded = tensorplay::tpsleef::log1p(
        tensorplay::tpsleef::exp(_mm256_xor_ps(diff, signbit)));
    const __m256 sum = _mm256_add_ps(m, folded);
    // Two equal infinities cancel in the difference above, so the folded
    // term is a not-a-number there; the answer is the infinity itself.
    const __m256 same_inf = _mm256_and_ps(
        _mm256_cmp_ps(x, y, _CMP_EQ_OQ),
        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x), inf, _CMP_EQ_OQ));
    return _mm256_blendv_ps(sum, x, same_inf);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d log_add_exp_f64(__m256d x, __m256d y) {
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d m = _mm256_max_pd(x, y);
    const __m256d diff = _mm256_andnot_pd(signbit, _mm256_sub_pd(x, y));
    const __m256d folded = tensorplay::tpsleef::log1p(
        tensorplay::tpsleef::exp(_mm256_xor_pd(diff, signbit)));
    const __m256d sum = _mm256_add_pd(m, folded);
    const __m256d same_inf = _mm256_and_pd(
        _mm256_cmp_pd(x, y, _CMP_EQ_OQ),
        _mm256_cmp_pd(_mm256_andnot_pd(signbit, x), inf, _CMP_EQ_OQ));
    return _mm256_blendv_pd(sum, x, same_inf);
}

// A quotient far beyond the exponent range leaves the vector fmod a
// not-a-number even though both operands are finite and the divisor is not
// zero; a finite divisor always leaves a finite remainder, so those lanes
// take the exact path one at a time.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 fmod_pair_f32(__m256 x, __m256 y) {
    __m256 r = Sleef_fmodf8(x, y);
    const __m256 finite = _mm256_and_ps(_mm256_cmp_ps(x, x, _CMP_ORD_Q),
                                        _mm256_cmp_ps(y, y, _CMP_ORD_Q));
    const __m256 nonzero =
        _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    const __m256 broken = _mm256_and_ps(
        _mm256_cmp_ps(r, r, _CMP_UNORD_Q), _mm256_and_ps(finite, nonzero));
    const int mask = _mm256_movemask_ps(broken);
    if (mask != 0) {
        alignas(32) float xs[8], ys[8], rs[8];
        _mm256_store_ps(xs, x);
        _mm256_store_ps(ys, y);
        _mm256_store_ps(rs, r);
        for (int lane = 0; lane < 8; ++lane) {
            if (mask & (1 << lane)) {
                rs[lane] = std::fmod(xs[lane], ys[lane]);
            }
        }
        r = _mm256_load_ps(rs);
    }
    return r;
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d fmod_pair_f64(__m256d x, __m256d y) {
    __m256d r = Sleef_fmodd4(x, y);
    const __m256d finite = _mm256_and_pd(_mm256_cmp_pd(x, x, _CMP_ORD_Q),
                                         _mm256_cmp_pd(y, y, _CMP_ORD_Q));
    const __m256d nonzero =
        _mm256_cmp_pd(y, _mm256_setzero_pd(), _CMP_NEQ_OQ);
    const __m256d broken = _mm256_and_pd(
        _mm256_cmp_pd(r, r, _CMP_UNORD_Q), _mm256_and_pd(finite, nonzero));
    const int mask = _mm256_movemask_pd(broken);
    if (mask != 0) {
        alignas(32) double xs[4], ys[4], rs[4];
        _mm256_store_pd(xs, x);
        _mm256_store_pd(ys, y);
        _mm256_store_pd(rs, r);
        for (int lane = 0; lane < 4; ++lane) {
            if (mask & (1 << lane)) {
                rs[lane] = std::fmod(xs[lane], ys[lane]);
            }
        }
        r = _mm256_load_pd(rs);
    }
    return r;
}

// The truncated remainder answers the dividend's sign; the modulo built on
// it adds one divisor back where the two disagree, so the answer follows
// the divisor's sign instead.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 remainder_f32(__m256 x, __m256 y) {
    const __m256 mod = fmod_pair_f32(x, y);
    const __m256 zero = _mm256_setzero_ps();
    // The two less-than masks differ exactly where the divisor and the
    // truncated remainder disagree in sign.
    const __m256 mixed = _mm256_xor_ps(_mm256_cmp_ps(y, zero, _CMP_LT_OQ),
                                       _mm256_cmp_ps(mod, zero, _CMP_LT_OQ));
    const __m256 adjust = _mm256_and_ps(
        _mm256_cmp_ps(mod, zero, _CMP_NEQ_UQ), mixed);
    return _mm256_blendv_ps(mod, _mm256_add_ps(mod, y), adjust);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d remainder_f64(__m256d x, __m256d y) {
    const __m256d mod = fmod_pair_f64(x, y);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d mixed = _mm256_xor_pd(_mm256_cmp_pd(y, zero, _CMP_LT_OQ),
                                        _mm256_cmp_pd(mod, zero, _CMP_LT_OQ));
    const __m256d adjust = _mm256_and_pd(
        _mm256_cmp_pd(mod, zero, _CMP_NEQ_UQ), mixed);
    return _mm256_blendv_pd(mod, _mm256_add_pd(mod, y), adjust);
}

// The chunk answers one map, so the map is named once rather than chosen
// at every step.
template <BOp op>
__attribute__((target("avx2,fma"), noinline))
static void f32_chunk_avx2(const float* a, const float* b, float* d,
                           int64_t begin, int64_t end) {
    // Two vectors are answered per step: the two answers do not depend on
    // each other, so the machine overlaps them instead of waiting for one
    // to finish before it starts the next.
    int64_t i = begin;
    for (; i + 16 <= end; i += 16) {
        const __m256 x0 = _mm256_loadu_ps(a + i);
        const __m256 y0 = _mm256_loadu_ps(b + i);
        const __m256 x1 = _mm256_loadu_ps(a + i + 8);
        const __m256 y1 = _mm256_loadu_ps(b + i + 8);
        __m256 r0, r1;
        if constexpr (op == BOp::Hypot) {
            r0 = tensorplay::tpsleef::hypot(x0, y0);
            r1 = tensorplay::tpsleef::hypot(x1, y1);
        } else if constexpr (op == BOp::Atan2) {
            r0 = tensorplay::tpsleef::atan2(x0, y0);
            r1 = tensorplay::tpsleef::atan2(x1, y1);
        } else if constexpr (op == BOp::Remainder) {
            r0 = remainder_f32(x0, y0);
            r1 = remainder_f32(x1, y1);
        } else if constexpr (op == BOp::Fmod) {
            r0 = fmod_pair_f32(x0, y0);
            r1 = fmod_pair_f32(x1, y1);
        } else {
            r0 = log_add_exp_f32(x0, y0);
            r1 = log_add_exp_f32(x1, y1);
        }
        _mm256_storeu_ps(d + i, r0);
        _mm256_storeu_ps(d + i + 8, r1);
    }
    for (; i + 8 <= end; i += 8) {
        const __m256 x = _mm256_loadu_ps(a + i);
        const __m256 y = _mm256_loadu_ps(b + i);
        __m256 r;
        if constexpr (op == BOp::Hypot) {
            r = tensorplay::tpsleef::hypot(x, y);
        } else if constexpr (op == BOp::Atan2) {
            r = tensorplay::tpsleef::atan2(x, y);
        } else if constexpr (op == BOp::Remainder) {
            r = remainder_f32(x, y);
        } else if constexpr (op == BOp::Fmod) {
            r = fmod_pair_f32(x, y);
        } else {
            r = log_add_exp_f32(x, y);
        }
        _mm256_storeu_ps(d + i, r);
    }
    for (; i < end; ++i) {
        d[i] = scalar_f32_of<op>(a[i], b[i]);
    }
}

// The reduced width answers the map in single precision: the operands are
// widened, the map is answered there, and the answer is narrowed once.  The
// widening and the narrowing are the hardware crossing, so a reduced-width
// pair no longer pays for a pass that converts the operands before the map
// is answered.
template <BOp op>
__attribute__((target("avx2,fma,f16c"), noinline))
static void f16_chunk_avx2(const uint16_t* a, const uint16_t* b,
                           uint16_t* d, int64_t begin, int64_t end) {
    int64_t i = begin;
    // Two vectors are answered per step: the two answers do not depend on
    // each other, so the machine overlaps them instead of waiting for one
    // to finish before it starts the next.
    for (; i + 16 <= end; i += 16) {
        const __m256 x0 = tensorplay::cpu::vecreduced::widen8(a + i);
        const __m256 y0 = tensorplay::cpu::vecreduced::widen8(b + i);
        const __m256 x1 = tensorplay::cpu::vecreduced::widen8(a + i + 8);
        const __m256 y1 = tensorplay::cpu::vecreduced::widen8(b + i + 8);
        __m256 r0, r1;
        if constexpr (op == BOp::Hypot) {
            r0 = tensorplay::tpsleef::hypot(x0, y0);
            r1 = tensorplay::tpsleef::hypot(x1, y1);
        } else if constexpr (op == BOp::Atan2) {
            r0 = tensorplay::tpsleef::atan2(x0, y0);
            r1 = tensorplay::tpsleef::atan2(x1, y1);
        } else if constexpr (op == BOp::Remainder) {
            r0 = remainder_f32(x0, y0);
            r1 = remainder_f32(x1, y1);
        } else if constexpr (op == BOp::Fmod) {
            r0 = fmod_pair_f32(x0, y0);
            r1 = fmod_pair_f32(x1, y1);
        } else {
            r0 = log_add_exp_f32(x0, y0);
            r1 = log_add_exp_f32(x1, y1);
        }
        tensorplay::cpu::vecreduced::store_f16_16(d + i, r0, r1);
    }
    for (; i + 8 <= end; i += 8) {
        const __m256 x = tensorplay::cpu::vecreduced::widen8(a + i);
        const __m256 y = tensorplay::cpu::vecreduced::widen8(b + i);
        __m256 r;
        if constexpr (op == BOp::Hypot) {
            r = tensorplay::tpsleef::hypot(x, y);
        } else if constexpr (op == BOp::Atan2) {
            r = tensorplay::tpsleef::atan2(x, y);
        } else if constexpr (op == BOp::Remainder) {
            r = remainder_f32(x, y);
        } else if constexpr (op == BOp::Fmod) {
            r = fmod_pair_f32(x, y);
        } else {
            r = log_add_exp_f32(x, y);
        }
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(d + i),
            tensorplay::cpu::vecreduced::narrow8(r));
    }
    for (; i < end; ++i) {
        d[i] = scalar_f16(op, a[i], b[i]);
    }
}

// The other reduced width walks the same way: its crossing is the bit move
// in and out of the single-precision pattern instead of a hardware
// conversion, so the map is answered eight elements at a time there too.
template <BOp op>
__attribute__((target("avx2,fma"), noinline))
static void bf16_chunk_avx2(const uint16_t* a, const uint16_t* b,
                            uint16_t* d, int64_t begin, int64_t end) {
    int64_t i = begin;
    // Two vectors are answered per step: the two answers do not depend on
    // each other, so the machine overlaps them instead of waiting for one
    // to finish before it starts the next.
    for (; i + 16 <= end; i += 16) {
        const __m256 x0 = tensorplay::cpu::vecreduced::widen_bf16_8(a + i);
        const __m256 y0 = tensorplay::cpu::vecreduced::widen_bf16_8(b + i);
        const __m256 x1 =
            tensorplay::cpu::vecreduced::widen_bf16_8(a + i + 8);
        const __m256 y1 =
            tensorplay::cpu::vecreduced::widen_bf16_8(b + i + 8);
        __m256 r0, r1;
        if constexpr (op == BOp::Hypot) {
            r0 = tensorplay::tpsleef::hypot(x0, y0);
            r1 = tensorplay::tpsleef::hypot(x1, y1);
        } else if constexpr (op == BOp::Atan2) {
            r0 = tensorplay::tpsleef::atan2(x0, y0);
            r1 = tensorplay::tpsleef::atan2(x1, y1);
        } else if constexpr (op == BOp::Remainder) {
            r0 = remainder_f32(x0, y0);
            r1 = remainder_f32(x1, y1);
        } else if constexpr (op == BOp::Fmod) {
            r0 = fmod_pair_f32(x0, y0);
            r1 = fmod_pair_f32(x1, y1);
        } else {
            r0 = log_add_exp_f32(x0, y0);
            r1 = log_add_exp_f32(x1, y1);
        }
        tensorplay::cpu::vecreduced::store_bf16_8(d + i, r0);
        tensorplay::cpu::vecreduced::store_bf16_8(d + i + 8, r1);
    }
    for (; i + 8 <= end; i += 8) {
        const __m256 x = tensorplay::cpu::vecreduced::widen_bf16_8(a + i);
        const __m256 y = tensorplay::cpu::vecreduced::widen_bf16_8(b + i);
        __m256 r;
        if constexpr (op == BOp::Hypot) {
            r = tensorplay::tpsleef::hypot(x, y);
        } else if constexpr (op == BOp::Atan2) {
            r = tensorplay::tpsleef::atan2(x, y);
        } else if constexpr (op == BOp::Remainder) {
            r = remainder_f32(x, y);
        } else if constexpr (op == BOp::Fmod) {
            r = fmod_pair_f32(x, y);
        } else {
            r = log_add_exp_f32(x, y);
        }
        tensorplay::cpu::vecreduced::store_bf16_8(d + i, r);
    }
    for (; i < end; ++i) {
        d[i] = scalar_bf16(op, a[i], b[i]);
    }
}

template <BOp op>
__attribute__((target("avx2,fma"), noinline))
static void f64_chunk_avx2(const double* a, const double* b, double* d,
                           int64_t begin, int64_t end) {
    int64_t i = begin;
    for (; i + 8 <= end; i += 8) {
        const __m256d x0 = _mm256_loadu_pd(a + i);
        const __m256d y0 = _mm256_loadu_pd(b + i);
        const __m256d x1 = _mm256_loadu_pd(a + i + 4);
        const __m256d y1 = _mm256_loadu_pd(b + i + 4);
        __m256d r0, r1;
        if constexpr (op == BOp::Hypot) {
            r0 = tensorplay::tpsleef::hypot(x0, y0);
            r1 = tensorplay::tpsleef::hypot(x1, y1);
        } else if constexpr (op == BOp::Atan2) {
            r0 = tensorplay::tpsleef::atan2(x0, y0);
            r1 = tensorplay::tpsleef::atan2(x1, y1);
        } else if constexpr (op == BOp::Remainder) {
            r0 = remainder_f64(x0, y0);
            r1 = remainder_f64(x1, y1);
        } else if constexpr (op == BOp::Fmod) {
            r0 = fmod_pair_f64(x0, y0);
            r1 = fmod_pair_f64(x1, y1);
        } else {
            r0 = log_add_exp_f64(x0, y0);
            r1 = log_add_exp_f64(x1, y1);
        }
        _mm256_storeu_pd(d + i, r0);
        _mm256_storeu_pd(d + i + 4, r1);
    }
    for (; i + 4 <= end; i += 4) {
        const __m256d x = _mm256_loadu_pd(a + i);
        const __m256d y = _mm256_loadu_pd(b + i);
        __m256d r;
        if constexpr (op == BOp::Hypot) {
            r = tensorplay::tpsleef::hypot(x, y);
        } else if constexpr (op == BOp::Atan2) {
            r = tensorplay::tpsleef::atan2(x, y);
        } else if constexpr (op == BOp::Remainder) {
            r = remainder_f64(x, y);
        } else if constexpr (op == BOp::Fmod) {
            r = fmod_pair_f64(x, y);
        } else {
            r = log_add_exp_f64(x, y);
        }
        _mm256_storeu_pd(d + i, r);
    }
    for (; i < end; ++i) {
        d[i] = scalar_f64(op, a[i], b[i]);
    }
}

#endif  // TP_VECBINARY_SLEEF

}  // namespace vecbinary
}  // namespace cpu
}  // namespace tensorplay
