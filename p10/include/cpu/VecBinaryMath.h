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
// * Two operands that are both minus infinity give a difference of NaN,
//   which the vector fold would turn into NaN where the scalar answer is
//   minus infinity.  The base is therefore selected back over the fold
//   when it is minus infinity.

#include "cpu/vec/intrinsics.h"
#include "cpu/VecReducedMath.h"
#include <cstdint>
#include <cmath>
#include <limits>

#include "cpu/vec/SleefShims.h"

#if defined(__x86_64__) || defined(__i386__)
#define TP_VECBINARY_X86 1
#endif

#if defined(TP_VECBINARY_X86) && defined(__x86_64__) \
    && (defined(__GNUC__) || defined(__clang__))
#define TP_VECBINARY_SLEEF 1
#endif

namespace tensorplay {
namespace cpu {
namespace vecbinary {

// Which pairwise map the chunk answers.
enum class BOp : int { Hypot, Atan2, LogAddExp };

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
    const float m = std::max(x, y);
    if (m == -std::numeric_limits<float>::infinity()) return m;
    if (m != m) return m;  // NaN propagates
    return m + std::log1p(std::exp(-std::fabs(x - y)));
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
        case BOp::LogAddExp: {
            float m = std::max(x, y);
            if (m == -std::numeric_limits<float>::infinity()) return m;
            if (m != m) return m;
            return m + std::log1p(std::exp(-std::fabs(x - y)));
        }
    }
    return 0.0f;
}

inline double scalar_f64(BOp op, double x, double y) {
    switch (op) {
        case BOp::Hypot:
            return std::hypot(x, y);
        case BOp::Atan2:
            return std::atan2(x, y);
        case BOp::LogAddExp: {
            double m = std::max(x, y);
            if (m == -std::numeric_limits<double>::infinity()) return m;
            if (m != m) return m;
            return m + std::log1p(std::exp(-std::fabs(x - y)));
        }
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
        case BOp::LogAddExp: {
            const float m = std::max(x, y);
            if (m == -std::numeric_limits<float>::infinity()) {
                return tensorplay::cpu::vecreduced::narrow_one(m);
            }
            if (m != m) {
                return tensorplay::cpu::vecreduced::narrow_one(m);
            }
            return tensorplay::cpu::vecreduced::narrow_one(
                m + std::log1p(std::exp(-std::fabs(x - y))));
        }
    }
    return 0;
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 log_add_exp_f32(__m256 x, __m256 y) {
    const __m256 signbit = _mm256_set1_ps(-0.0f);
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    const __m256 m = _mm256_max_ps(x, y);
    const __m256 diff = _mm256_andnot_ps(signbit, _mm256_sub_ps(x, y));
    const __m256 folded = tensorplay::tpsleef::log1p(
        tensorplay::tpsleef::exp(_mm256_sub_ps(_mm256_setzero_ps(), diff)));
    const __m256 sum = _mm256_add_ps(m, folded);
    // A base of minus infinity is the answer itself: the folded term would
    // be a NaN there, since the difference of two minus infinities is one.
    return _mm256_blendv_ps(sum, m, _mm256_cmp_ps(m, neg_inf, _CMP_EQ_OQ));
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d log_add_exp_f64(__m256d x, __m256d y) {
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d neg_inf =
        _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d m = _mm256_max_pd(x, y);
    const __m256d diff = _mm256_andnot_pd(signbit, _mm256_sub_pd(x, y));
    const __m256d folded = tensorplay::tpsleef::log1p(
        tensorplay::tpsleef::exp(_mm256_sub_pd(_mm256_setzero_pd(), diff)));
    const __m256d sum = _mm256_add_pd(m, folded);
    return _mm256_blendv_pd(sum, m, _mm256_cmp_pd(m, neg_inf, _CMP_EQ_OQ));
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
        } else {
            r0 = log_add_exp_f32(x0, y0);
            r1 = log_add_exp_f32(x1, y1);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i),
                         tensorplay::cpu::vecreduced::narrow8(r0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i + 8),
                         tensorplay::cpu::vecreduced::narrow8(r1));
    }
    for (; i + 8 <= end; i += 8) {
        const __m256 x = tensorplay::cpu::vecreduced::widen8(a + i);
        const __m256 y = tensorplay::cpu::vecreduced::widen8(b + i);
        __m256 r;
        if constexpr (op == BOp::Hypot) {
            r = tensorplay::tpsleef::hypot(x, y);
        } else if constexpr (op == BOp::Atan2) {
            r = tensorplay::tpsleef::atan2(x, y);
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
