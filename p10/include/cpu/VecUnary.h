#pragma once

// Vectorized fast paths for CPU unary kernels.
//
// Design notes:
// * Every kernel here is a single translation-unit-local function built with
//   GCC per-function target attributes ("avx2", "fma", "f16c"), so p10 keeps
//   compiling without ISA flags while the hot loops get real AVX2 codegen.
//   Dispatch happens once per chunk through avx2_available(); below-AVX2
//   machines transparently keep the previous scalar behaviour.
// * Transcendentals dispatch to the vendored SLEEF vector math through
//   cpu/vec/SleefShims.h (runtime-dispatched entry points, linked into
//   libp10).  Composite activations reuse those building blocks, so the
//   fast path and the scalar tail implement the same formulas.
// * lgamma has no libmvec entry point; it uses a Lanczos expansion restricted
//   to strictly positive inputs (blocks containing non-positive or NaN values
//   fall back to the scalar std::lgamma, which handles the reflection domain).
//   PointwiseKernels.cpp element-for-element, including multiplication order,
//   so vector and fallback paths produce identical values.
// * Bounded clamps are always written min(hi, max(lo, x)) — the clamped value
//   stays the *second* operand of both instructions.  On x86 VMAXPS/VMINPS
//   return the second operand when either side is NaN (and on a -0.0/+0.0
//   tie), so the reversed form would replace NaN with the lower bound and
//   normalize -0.0 to +0.0.  The scalar ops keep NaN, so both paths agree.

#include "cpu/vec/intrinsics.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Half.h"
#include "BFloat16.h"

#include <SpecialMath.h>
#include "cpu/vec/ErfPoly.h"
#include "cpu/vec/SleefShims.h"

#if defined(__x86_64__) || defined(__i386__)
#define TP_VECUNARY_X86 1
#endif

#if defined(TP_VECUNARY_X86) && defined(__x86_64__) \
    && (defined(__GNUC__) || defined(__clang__))
#define TP_VECUNARY_SLEEF 1
#endif

namespace tensorplay {
namespace cpu {
namespace vecunary {

enum class VOp : int {
    None = 0,
    // dtype-preserving / cheap bit ops
    Abs, Neg, Sign, Square, Reciprocal, Sqrt, Rsqrt,
    Floor, Ceil, Trunc, Round, Frac, Relu,
    // transcendental (libmvec)
    Exp, Exp2, Expm1, Log, Log2, Log10, Log1p,
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
    Erf, Erfc, ErfInv, Lgamma,
    // composed: a vector walk over one of the ops above plus an elementwise
    // fix-up, so they need no kernel of their own
    Sinc, Entr, Logit, SphBesselJ0,
    // Chebyshev series evaluated across the lanes (see the Bessel block below)
    I1e,
    // library-only: no vector form below, so these run the scalar tail unless
    // the library is linked in
    Ndtr, NdtrI, Erfcx,
    // composite activations
    Sigmoid, GeluNone, GeluTanh, Silu, Mish, Selu,
    Elu, Softplus, Hardswish, Hardsigmoid, LeakyRelu,
    Hardtanh, Relu6,
};

// Runtime scalars for parameterised ops (elu, softplus, celu, leaky_relu,
// hardtanh).  Stored as double so the Float64 kernels keep full precision;
// the Float32 kernels truncate exactly like the scalar lambdas do.
struct VParams {
    double p0 = 0.0, p1 = 0.0, p2 = 0.0;
};

inline bool avx2_available() {
#if defined(TP_VECUNARY_X86)
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
#else
    return false;
#endif
}

inline bool f16c_available() {
#if defined(TP_VECUNARY_X86)
    static const bool ok = __builtin_cpu_supports("f16c") != 0;
    return ok;
#else
    return false;
#endif
}

inline bool fma_available() {
#if defined(TP_VECUNARY_X86)
    static const bool ok = __builtin_cpu_supports("fma") != 0;
    return ok;
#else
    return false;
#endif
}

// Zen4-class machines: full-width 512-bit datapath.  Runtime-checked; the
// 512-bit kernels below carry their own target attributes so p10 keeps
// compiling without global ISA flags.
inline bool avx512_available() {
#if defined(TP_VECUNARY_X86)
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
#else
    return false;
#endif
}

// f64 kernels only the AVX2 tier carries; the 512-bit double kernels have no
// case for them, so they run there (or on the scalar loop) instead.
inline bool f64_avx2_only(VOp op) {
    return op == VOp::Elu || op == VOp::Softplus || op == VOp::LeakyRelu ||
           op == VOp::Hardswish || op == VOp::Hardsigmoid;
}

// ---------------------------------------------------------------------------
// SLEEF entry points are declared in cpu/vec/SleefShims.h; the runtime
// CPU dispatch happens inside libsleef.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scalar reference implementations.  These reuse the formulas in
// PointwiseKernels.cpp one-for-one; they serve as both the non-AVX2 fallback
// and the vector-loop tail.
// ---------------------------------------------------------------------------
// sin(x)/x.  The series form is used below 0.5, where the quotient loses the
// leading digits; the caller supplies sin(x) so a vector walk can hand in the
// value it already has.
template <typename T>
inline T spherical_j0_from_sin(T x, T sinx) {
    if (std::isinf(x)) return T(0);
    if (std::fabs(x) < T(0.5)) {
        const T x2 = x * x;
        return T(1) + x2 * (T(-1.0) / T(6.0) +
               x2 * (T(1.0) / T(120.0) +
               x2 * (T(-1.0) / T(5040.0) +
               x2 * (T(1.0) / T(362880.0) +
               x2 * (T(-1.0) / T(39916800.0) +
               x2 * (T(1.0) / T(6227020800.0)))))));
    }
    return sinx / x;
}

template <typename T>
inline T scalar_apply(VOp op, VParams prm, T x) {
    constexpr T kHalf = static_cast<T>(0.5);
    switch (op) {
        case VOp::Abs: return std::abs(x);
        case VOp::Neg: return -x;
        case VOp::Sign: return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
        case VOp::Square: return x * x;
        case VOp::Reciprocal: return T(1) / x;
        case VOp::Sqrt: return std::sqrt(x);
        case VOp::Rsqrt: return static_cast<T>(1) / std::sqrt(x);
        case VOp::Floor: return std::floor(x);
        case VOp::Ceil: return std::ceil(x);
        case VOp::Trunc: return std::trunc(x);
        case VOp::Round: return std::nearbyint(x);
        case VOp::Frac: return x - std::trunc(x);
        case VOp::Relu: return x < T(0) ? T(0) : x;
        case VOp::Exp: return std::exp(x);
        case VOp::Exp2: return std::exp2(x);
        case VOp::Expm1: return std::expm1(x);
        case VOp::Log: return std::log(x);
        case VOp::Log2: return std::log2(x);
        case VOp::Log10: return std::log10(x);
        case VOp::Log1p: return std::log1p(x);
        case VOp::Sin: return std::sin(x);
        case VOp::Cos: return std::cos(x);
        case VOp::Tan: return std::tan(x);
        case VOp::Asin: return std::asin(x);
        case VOp::Acos: return std::acos(x);
        case VOp::Atan: return std::atan(x);
        case VOp::Sinh: return std::sinh(x);
        case VOp::Cosh: return std::cosh(x);
        case VOp::Tanh: return std::tanh(x);
        case VOp::Asinh: return std::asinh(x);
        case VOp::Acosh: return std::acosh(x);
        case VOp::Atanh: return std::atanh(x);
        case VOp::Erf: return std::erf(x);
        case VOp::Erfc: return std::erfc(x);
        case VOp::ErfInv:
            return static_cast<T>(special_math::calc_erfinv(static_cast<double>(x)));
        case VOp::Ndtr:
            return static_cast<T>(special_math::calc_ndtr(static_cast<double>(x)));
        case VOp::NdtrI:
            return static_cast<T>(special_math::calc_ndtri(static_cast<double>(x)));
        case VOp::Erfcx:
            return static_cast<T>(special_math::calc_erfcx(static_cast<double>(x)));
        case VOp::Sinc: {
            const T px = static_cast<T>(3.14159265358979323846) * x;
            return std::fabs(px) < T(1e-30) ? T(1) : std::sin(px) / px;
        }
        case VOp::Entr:
            return x != x ? x
                 : (x > T(0) ? -x * std::log(x)
                             : (x == T(0) ? T(0)
                                          : -std::numeric_limits<T>::infinity()));
        case VOp::Logit: {
            T p = x;
            if (prm.p0 >= 0.0)
                p = std::min(std::max(p, static_cast<T>(prm.p0)),
                             T(1) - static_cast<T>(prm.p0));
            return std::log(p / (T(1) - p));
        }
        case VOp::SphBesselJ0: return spherical_j0_from_sin(x, std::sin(x));
        case VOp::I1e: return special_math::calc_i1e(x);
        case VOp::Lgamma: return std::lgamma(x);
        case VOp::Sigmoid: return T(1) / (T(1) + std::exp(-x));
        case VOp::GeluNone: {
            const T kAlpha = static_cast<T>(0.70710678118654752440); // M_SQRT1_2
            return kHalf * x * (T(1) + std::erf(x * kAlpha));
        }
        case VOp::GeluTanh: {
            // scalar_gelu_approximated_with_tanh
            const T kBeta = static_cast<T>(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const T kKappa = static_cast<T>(0.044715);
            T x_cube = x * x * x;
            T inner = kBeta * (x + kKappa * x_cube);
            return kHalf * x * (T(1) + std::tanh(inner));
        }
        case VOp::Silu: return x / (T(1) + std::exp(-x));
        case VOp::Mish: {
            T sp = std::log(T(1) + std::exp(x));
            return x * std::tanh(sp);
        }
        case VOp::Selu: {
            constexpr double lambda_ = 1.0507009873554804934193349852946;
            constexpr double alpha_ = 1.6732632423543772848170429916717;
            return x > T(0) ? x * static_cast<T>(lambda_)
                            : static_cast<T>(alpha_ * lambda_) * std::expm1(x);
        }
        case VOp::Elu: {
            const T negcoef = static_cast<T>(prm.p0); // alpha*scale
            const T poscoef = static_cast<T>(prm.p1); // scale
            const T negipt = static_cast<T>(prm.p2);  // input_scale
            return x < T(0) ? static_cast<T>(std::expm1(x * negipt) * negcoef)
                            : x * poscoef;
        }
        case VOp::Softplus: {
            const T beta = static_cast<T>(prm.p0);
            const T threshold = static_cast<T>(prm.p1);
            return x * beta > threshold
                ? x
                : static_cast<T>(std::log1p(std::exp(static_cast<T>(x * beta))) / beta);
        }
        case VOp::Hardswish: {
            T xf = x;
            T clamped = (xf + T(3) < T(0)) ? T(0) : (xf + T(3) > T(6)) ? T(6) : xf + T(3);
            return xf * clamped / T(6);
        }
        case VOp::Hardsigmoid: {
            T xf = x;
            T v = xf + T(3);
            v = v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
            return v / T(6);
        }
        case VOp::LeakyRelu: {
            const T slope = static_cast<T>(prm.p0);
            T xf = x;
            return xf < T(0) ? slope * xf : xf;
        }
        case VOp::Hardtanh: {
            const T lo = static_cast<T>(prm.p0);
            const T hi = static_cast<T>(prm.p1);
            return x < lo ? lo : (x > hi ? hi : x);
        }
        case VOp::Relu6: { // hardtanh(0, 6)
            return x < T(0) ? T(0) : (x > T(6) ? T(6) : x);
        }
        default: return x;
    }
}

// ---------------------------------------------------------------------------
// AVX2 helpers.  Everything below either carries a target attribute itself or
// is always_inline so it folds into an attributed caller.
// ---------------------------------------------------------------------------
#ifdef TP_VECUNARY_SLEEF

constexpr float kSignBitF = -0.0f;

// Lanczos g=7, n=9 coefficients.
constexpr double kLanczos[9] = {
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
constexpr double kHalfLogTwoPi = 0.91893853320467274178032973640562;

// Natural logarithm of a float vector, and its base-2 and base-10 scalings.
//
// The input is split as 2^e * m with e = round(log2 x), so m lands in
// [0.75, 1.5).  Rounding the exponent instead of truncating it is what keeps
// inputs near 1 accurate: they reduce to e = 0 and m close to 1, and the
// result is then formed from a leading term that already is the answer rather
// than from the difference of two numbers near 0.7.
//
// Under that reduction log(m) = log(1 + u) = u * P(u) with u = m - 1 in
// [-0.25, 0.5).  The fitted P keeps its constant term at 1, so the relative
// error stays bounded as u approaches zero; the worst relative error over the
// interval is 1.2e-8, about 0.1 ulp.
//
// The base only scales P and changes the exponent term, so all three come out
// of one template.  Base 2 adds the integer exponent straight to the mantissa
// term; the other two fold the log of their base into P, which keeps the
// mantissa product free of an extra scaling multiply.
enum class LogBase { E, Two, Ten };

template <LogBase Base>
constexpr float log_poly_scale() {
    if constexpr (Base == LogBase::Two) {
        return 1.44269504088896341f;
    } else if constexpr (Base == LogBase::Ten) {
        return 0.4342944819032518f;
    } else {
        return 1.0f;
    }
}

// Fast drops the subnormal rescale and the domain fix-up; the caller then
// guarantees every lane is a positive normal float.
template <LogBase Base, bool Fast>
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log_impl(__m256 x) {
    constexpr float kScale = log_poly_scale<Base>();
    __m256i ix;
    __m256 tiny = _mm256_setzero_ps();
    if constexpr (Fast) {
        ix = _mm256_castps_si256(x);
    } else {
        // A subnormal leaves the exponent field at zero and its leading bit
        // in the mantissa, so it is scaled into the normal range first and
        // the factor is paid back out of the exponent.
        tiny = _mm256_cmp_ps(x, _mm256_set1_ps(1.1754944e-38f), _CMP_LT_OQ);
        ix = _mm256_castps_si256(
            _mm256_blendv_ps(x, _mm256_mul_ps(x, _mm256_set1_ps(1.8446744e19f)), tiny));
    }
    // Adding half a mantissa range before shifting rounds the exponent to
    // nearest.  Subtracting the exponent field from the original bit pattern
    // then leaves 0.75 <= m < 1.5.
    const __m256i biased =
        _mm256_srli_epi32(_mm256_add_epi32(ix, _mm256_set1_epi32(0x00400000)), 23);
    const __m256i ei = _mm256_sub_epi32(biased, _mm256_set1_epi32(127));
    // Removing the exponent field leaves x / 2^ei in [0.75, 1.5).  Taking the
    // bias out first keeps the mantissa rebuild to one shift and one
    // subtract; the shift is a logical one, which is still correct for a
    // negative exponent because the arithmetic is modulo 2^32.
    const __m256 m = _mm256_castsi256_ps(
        _mm256_sub_epi32(ix, _mm256_slli_epi32(ei, 23)));
    __m256 e = _mm256_cvtepi32_ps(ei);
    if constexpr (!Fast) {
        e = _mm256_sub_ps(e, _mm256_and_ps(tiny, _mm256_set1_ps(64.0f)));
    }

    const __m256 u = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 p = _mm256_set1_ps(-4.413553457e-2f * kScale);
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.032194468e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.310681637e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.448485294e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.665304757e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.999078531e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-2.499990939e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(3.333346255e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-5.000000213e-1f * kScale));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(9.999999972e-1f * kScale));

    __m256 r;
    if constexpr (Base == LogBase::Two) {
        r = _mm256_add_ps(e, _mm256_mul_ps(u, p));
    } else {
        // One FMA is enough here: the exponent term's relative error is
        // |c - c_float| / c whatever the exponent, unlike in exp where the
        // reduced argument feeds straight into the result.
        constexpr float kHi = Base == LogBase::Ten ? 0.30102999566398119521f
                                                   : 0.693147182464599609375f;
        r = _mm256_fmadd_ps(e, _mm256_set1_ps(kHi), _mm256_mul_ps(u, p));
    }

    if constexpr (!Fast) {
        // log(0) is -inf, log(+inf) is +inf, and negative or NaN inputs give
        // NaN.  max(x, 0) returns 0 for a NaN first operand, so one compare
        // covers both the negative and the unordered cases.
        const __m256 zero = _mm256_setzero_ps();
        const __m256 bad = _mm256_cmp_ps(_mm256_max_ps(x, zero), x, _CMP_NEQ_UQ);
        r = _mm256_blendv_ps(r, _mm256_set1_ps(-INFINITY),
                             _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));
        r = _mm256_blendv_ps(r, x,
                             _mm256_cmp_ps(x, _mm256_set1_ps(3.4028235e38f), _CMP_GT_OQ));
        r = _mm256_blendv_ps(r, _mm256_set1_ps(NAN), bad);
    }
    return r;
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log_f32(__m256 x) {
    return v_log_impl<LogBase::E, false>(x);
}

// Range-restricted variants for callers that have already proved every lane
// is a positive normal float.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log_f32_fast(__m256 x) {
    return v_log_impl<LogBase::E, true>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log2_f32(__m256 x) {
    return v_log_impl<LogBase::Two, false>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log2_f32_fast(__m256 x) {
    return v_log_impl<LogBase::Two, true>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log10_f32(__m256 x) {
    return v_log_impl<LogBase::Ten, false>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log10_f32_fast(__m256 x) {
    return v_log_impl<LogBase::Ten, true>(x);
}

// One pass over a chunk for any of the three bases: a block of 32 whose lanes
// are all positive normal floats takes the short chain, and anything else --
// zero, negatives, subnormals, inf, NaN -- runs the guarded version for that
// block only.
template <LogBase Base>
__attribute__((target("avx2,fma"), always_inline))
inline void log_range_loop(const float* s, float* d, int64_t n, int64_t& i) {
    const __m256 min_normal = _mm256_set1_ps(1.1754944e-38f);
    const __m256 inf = _mm256_set1_ps(INFINITY);
    for (; i + 32 <= n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(s + i);
        __m256 x1 = _mm256_loadu_ps(s + i + 8);
        __m256 x2 = _mm256_loadu_ps(s + i + 16);
        __m256 x3 = _mm256_loadu_ps(s + i + 24);
        // Ordered predicates, so a NaN lane fails the test.
        __m256 good = _mm256_and_ps(
            _mm256_and_ps(_mm256_cmp_ps(x0, min_normal, _CMP_GE_OQ),
                          _mm256_cmp_ps(x0, inf, _CMP_LT_OQ)),
            _mm256_and_ps(_mm256_cmp_ps(x1, min_normal, _CMP_GE_OQ),
                          _mm256_cmp_ps(x1, inf, _CMP_LT_OQ)));
        good = _mm256_and_ps(good, _mm256_and_ps(
            _mm256_and_ps(_mm256_cmp_ps(x2, min_normal, _CMP_GE_OQ),
                          _mm256_cmp_ps(x2, inf, _CMP_LT_OQ)),
            _mm256_and_ps(_mm256_cmp_ps(x3, min_normal, _CMP_GE_OQ),
                          _mm256_cmp_ps(x3, inf, _CMP_LT_OQ))));
        if (_mm256_movemask_ps(good) == 0xff) {
            _mm256_storeu_ps(d + i, v_log_impl<Base, true>(x0));
            _mm256_storeu_ps(d + i + 8, v_log_impl<Base, true>(x1));
            _mm256_storeu_ps(d + i + 16, v_log_impl<Base, true>(x2));
            _mm256_storeu_ps(d + i + 24, v_log_impl<Base, true>(x3));
        } else {
            _mm256_storeu_ps(d + i, v_log_impl<Base, false>(x0));
            _mm256_storeu_ps(d + i + 8, v_log_impl<Base, false>(x1));
            _mm256_storeu_ps(d + i + 16, v_log_impl<Base, false>(x2));
            _mm256_storeu_ps(d + i + 24, v_log_impl<Base, false>(x3));
        }
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(d + i, v_log_impl<Base, false>(_mm256_loadu_ps(s + i)));
    }
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_lgamma_pos_f32(__m256 x) {
    // Valid only where every lane > 0.  Reflection for x < 1:
    // lgamma(x) = lgamma(x + 1) - log(x).
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 small = _mm256_cmp_ps(x, one, _CMP_LT_OQ);
    const __m256 xp = _mm256_add_ps(x, _mm256_and_ps(small, one));

    __m256 z = _mm256_sub_ps(xp, one);
    __m256 series = _mm256_set1_ps(static_cast<float>(kLanczos[0]));
#pragma GCC unroll 8
    for (int i = 1; i < 9; ++i) {
        __m256 denom = _mm256_add_ps(z, _mm256_set1_ps(static_cast<float>(i)));
        series = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(kLanczos[i])),
                                 _mm256_div_ps(one, denom), series);
    }
    __m256 t = _mm256_add_ps(z, _mm256_set1_ps(7.5f));
    __m256 res = _mm256_sub_ps(
        _mm256_fmadd_ps(_mm256_add_ps(z, _mm256_set1_ps(0.5f)),
                        v_log_f32(t), _mm256_set1_ps(static_cast<float>(kHalfLogTwoPi))),
        t);
    res = _mm256_add_ps(res, v_log_f32(series));
    return _mm256_sub_ps(res, _mm256_and_ps(small, v_log_f32(x)));
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_lgamma_pos_f64(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d small = _mm256_cmp_pd(x, one, _CMP_LT_OQ);
    const __m256d xp = _mm256_add_pd(x, _mm256_and_pd(small, one));

    __m256d z = _mm256_sub_pd(xp, one);
    __m256d series = _mm256_set1_pd(kLanczos[0]);
#pragma GCC unroll 8
    for (int i = 1; i < 9; ++i) {
        __m256d denom = _mm256_add_pd(z, _mm256_set1_pd(static_cast<double>(i)));
        series = _mm256_fmadd_pd(_mm256_set1_pd(kLanczos[i]),
                                 _mm256_div_pd(one, denom), series);
    }
    __m256d t = _mm256_add_pd(z, _mm256_set1_pd(7.5));
    __m256d res = _mm256_sub_pd(
        _mm256_fmadd_pd(_mm256_add_pd(z, _mm256_set1_pd(0.5)),
                        tensorplay::tpsleef::log(t), _mm256_set1_pd(kHalfLogTwoPi)),
        t);
    res = _mm256_add_pd(res, tensorplay::tpsleef::log(series));
    return _mm256_sub_pd(res, _mm256_and_pd(small, tensorplay::tpsleef::log(x)));
}

// Reciprocal via rcp + Newton-Raphson refinements: _mm256_rcp_ps seeds at
// ~1.5*2^-12 rel err, two NR steps land well under float ulp -- a drop-in
// for divps when the numerator is a vector (silu) or the inf/NaN corners
// are repaired explicitly (sigmoid).
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_rcp_nr_ps(__m256 d) {
    __m256 r = _mm256_rcp_ps(d);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(d, r, _mm256_set1_ps(2.0f)));
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(d, r, _mm256_set1_ps(2.0f)));
}

// Fast float exp: x = k*ln2 + r with |r| <= ln2/2, exp(r) from a
// fifth-order polynomial and 2^k from the exponent field.
//
// The polynomial is the only serial part of the chain.  A fifth-order
// fit holds the worst-case relative error to 4.1e-7 over the reduced
// interval, about 3.4 ulp, which is the trade this path makes for the
// shorter chain; the composites that call it (sigmoid, softplus, the
// erf tail) carry a wider error budget than the bare unary entry point.
//
// The scale is split across two exponent fields so results below the
// smallest normal stay representable as denormals.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_exp_f32(__m256 x) {
    // Inputs past the underflow bound have no representable result, and
    // letting them through would drive the polynomial to -inf where the
    // answer is +0.  The floor also catches -inf; the operand order keeps
    // NaN on the second position so it passes through unchanged.
    x = _mm256_max_ps(_mm256_set1_ps(-103.97208404541015625f), x);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693145751953125f);
    const __m256 ln2_lo = _mm256_set1_ps(1.428606765330187e-6f);
    __m256i ki = _mm256_cvtps_epi32(_mm256_mul_ps(x, log2e));
    __m256 k = _mm256_cvtepi32_ps(ki);
    __m256 r = _mm256_fnmadd_ps(k, ln2_hi, x);
    r = _mm256_fnmadd_ps(k, ln2_lo, r);
    // Coefficients fit to exp on [-ln2/2, ln2/2] with the constant and
    // linear terms pinned to 1, so exp(0) comes back exactly 1.  Worst
    // relative error over that interval is 4.1e-7, about 3.4 ulp.
    __m256 p = _mm256_set1_ps(8.3572001485e-3f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1833804078e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666630825e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.9999748990e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    // The exponent used for the scale is clamped to the range the two
    // half-fields can express.  Without it a very large input wraps the
    // field and turns an overflow into a small value; the polynomial
    // keeps the unclamped exponent, so the reduced argument stays right.
    __m256i ks = _mm256_max_epi32(
        _mm256_min_epi32(ki, _mm256_set1_epi32(254)),
        _mm256_set1_epi32(-252));
    __m256i k1 = _mm256_srai_epi32(ks, 1);
    __m256i k2 = _mm256_sub_epi32(ks, k1);
    __m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(k1, _mm256_set1_epi32(127)), 23));
    __m256 s2 = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(k2, _mm256_set1_epi32(127)), 23));
    // Two multiplies rather than one: the product of both halves can
    // reach 2^128, past the largest finite float, while the polynomial
    // times the first half and then the second stays finite for inputs
    // whose result is still representable.
    return _mm256_mul_ps(_mm256_mul_ps(p, s1), s2);
}

// Range-restricted exp for callers that have already proved every lane lies
// in [-87, 88].  The result is then always a normal float, so the exponent
// field alone carries the scale: no input floor, no exponent clamp, no
// second scale multiply.  One FMA against the float-rounded ln2 leaves the
// reduced argument within 2.4e-7 of the exact one across that whole range,
// which the polynomial's own 4.1e-7 budget absorbs.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_exp_f32_fast(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2 = _mm256_set1_ps(0.693147182464599609375f);
    __m256i ki = _mm256_cvtps_epi32(_mm256_mul_ps(x, log2e));
    __m256 k = _mm256_cvtepi32_ps(ki);
    __m256 r = _mm256_fnmadd_ps(k, ln2, x);
    __m256 p = _mm256_set1_ps(8.3572001485e-3f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1833804078e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666630825e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.9999748990e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    const __m256 s = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(p, s);
}

// erf on float lanes.  Coefficients live in cpu/vec/ErfPoly.h so the vector
// class and this kernel evaluate the same function.
// Fast selects the range-restricted exp for the decay term; the caller then
// guarantees |x| <= 9.3, which keeps -x*x above the point where exp stops
// returning a normal float.
template <bool Fast>
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erf_f32_impl(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 abs_x = _mm256_andnot_ps(sign_bit, x);
    // A single five-term form covers the whole range, so there is no series
    // branch for small inputs and no blend afterwards.  The reciprocal
    // argument maps [0, inf) onto (0, 1], which lets one polynomial in t carry
    // both the linear region and the tail.
    const __m256 t = _mm256_div_ps(
        one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), abs_x, one));
    __m256 p = _mm256_set1_ps(1.061405429f);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453152027f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736f));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592f));
    p = _mm256_mul_ps(p, t);
    // exp(-x*x) decays to zero, so the tail saturates at one without any
    // explicit clamp; the sign is restored at the end.
    const __m256 arg = _mm256_xor_ps(sign_bit, _mm256_mul_ps(x, x));
    __m256 decay;
    if constexpr (Fast) {
        decay = v_exp_f32_fast(arg);
    } else {
        decay = v_exp_f32(arg);
    }
    const __m256 r = _mm256_sub_ps(one, _mm256_mul_ps(p, decay));
    return _mm256_or_ps(_mm256_and_ps(sign_bit, x), r);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erf_f32(__m256 x) {
    return v_erf_f32_impl<false>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erf_f32_fast(__m256 x) {
    return v_erf_f32_impl<true>(x);
}

// erfc on float lanes.  The tail form erfc(x) = Q(t) * t * exp(-x*x) with
// t = 1/(1 + p|x|) is evaluated directly, so the answer never comes out of
// subtracting two numbers near one and keeps its relative accuracy all the
// way down to the underflow bound.  Q is a ten-term fit holding 7e-9
// relative over t in [0.2, 1]; its value at t = 0 is the leading
// coefficient, so inputs past the fitted range decay towards zero instead of
// blowing up, and the exp factor has underflowed by then anyway.
template <bool Fast>
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erfc_f32_impl(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 abs_x = _mm256_andnot_ps(sign_bit, x);
    const __m256 t = _mm256_div_ps(
        one, _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), abs_x, one));
    __m256 q = _mm256_set1_ps(1.7310578515e-2f);
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(-4.6872685983e-2f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(-4.4495100226e-2f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(2.1182032722e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(-1.5454427326e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(2.3693767973e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(6.6187245797e-2f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(1.7168252403e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(1.7206103609e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(1.8510092770e-1f));
    q = _mm256_fmadd_ps(q, t, _mm256_set1_ps(1.8481174722e-1f));
    const __m256 sq = _mm256_mul_ps(x, x);
    // The square rounds, and that rounding is multiplied by the square
    // itself inside the exponent -- at |x| = 10 it is already 7e-6 relative.
    // Carrying the lost bit out and folding it back as a first-order factor
    // puts the argument back on the exact square.
    const __m256 sq_err = _mm256_fnmadd_ps(x, x, sq);
    const __m256 arg = _mm256_xor_ps(sign_bit, sq);
    __m256 decay;
    if constexpr (Fast) {
        decay = v_exp_f32_fast(arg);
    } else {
        decay = v_exp_f32(arg);
    }
    decay = _mm256_fmadd_ps(decay, sq_err, decay);
    const __m256 tail = _mm256_mul_ps(_mm256_mul_ps(q, t), decay);
    // erfc(-x) = 2 - erfc(x).  The subtraction is safe because the tail is at
    // most one, so two minus it never cancels.
    const __m256 neg = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    return _mm256_blendv_ps(tail, _mm256_sub_ps(_mm256_set1_ps(2.0f), tail), neg);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erfc_f32(__m256 x) {
    return v_erfc_f32_impl<false>(x);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erfc_f32_fast(__m256 x) {
    return v_erfc_f32_impl<true>(x);
}

// asin and acos on float lanes.  One polynomial serves both the direct and
// the half-angle branch: with y = x^2 for |x| <= 0.5 and y = (1-|x|)/2
// beyond, asin(|x|) is either x + x*y*P(y) or pi/2 - 2*sqrt(y)*(1 + y*P(y)).
// The square root's NaN on a negative argument is also what reports the
// domain error for |x| > 1, so no separate range check is needed.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_asin_f32(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ax = _mm256_andnot_ps(sign_bit, x);
    const __m256 small = _mm256_cmp_ps(ax, _mm256_set1_ps(0.5f), _CMP_LE_OQ);
    const __m256 y = _mm256_blendv_ps(
        _mm256_mul_ps(_mm256_sub_ps(one, ax), _mm256_set1_ps(0.5f)),
        _mm256_mul_ps(ax, ax), small);
    __m256 p = _mm256_set1_ps(3.4078566240e-2f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(7.0976971196e-4f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(2.1211881877e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(2.1733134550e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(3.0443983482e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(4.4639437168e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(7.5000096023e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.6666666565e-1f));
    const __m256 direct = _mm256_fmadd_ps(_mm256_mul_ps(ax, y), p, ax);
    const __m256 t = _mm256_sqrt_ps(y);
    const __m256 wide = _mm256_fnmadd_ps(
        _mm256_add_ps(t, t),
        _mm256_fmadd_ps(y, p, one),
        _mm256_set1_ps(1.57079632679489661923f));
    return _mm256_or_ps(_mm256_and_ps(sign_bit, x),
                        _mm256_blendv_ps(wide, direct, small));
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_acos_f32(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ax = _mm256_andnot_ps(sign_bit, x);
    // Inside |x| <= 0.5 the answer is the complement of the direct branch;
    // outside it is 2*asin(sqrt((1-|x|)/2)), reflected for negative inputs.
    const __m256 mid = _mm256_cmp_ps(ax, half, _CMP_LE_OQ);
    const __m256 y = _mm256_blendv_ps(
        _mm256_mul_ps(_mm256_sub_ps(one, ax), half),
        _mm256_mul_ps(ax, ax), mid);
    __m256 p = _mm256_set1_ps(3.4078566240e-2f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(7.0976971196e-4f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(2.1211881877e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(2.1733134550e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(3.0443983482e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(4.4639437168e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(7.5000096023e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.6666666565e-1f));
    const __m256 direct = _mm256_fmadd_ps(_mm256_mul_ps(ax, y), p, ax);
    const __m256 t = _mm256_sqrt_ps(y);
    const __m256 tail =
        _mm256_mul_ps(_mm256_add_ps(t, t), _mm256_fmadd_ps(y, p, one));
    // The polynomial was evaluated on |x|, so the complement branch needs the
    // sign put back before it can be subtracted from pi/2.
    const __m256 signed_direct =
        _mm256_or_ps(_mm256_and_ps(sign_bit, x), direct);
    __m256 r = _mm256_sub_ps(_mm256_set1_ps(1.57079632679489661923f), signed_direct);
    r = _mm256_blendv_ps(r, tail, _mm256_cmp_ps(x, half, _CMP_GT_OQ));
    r = _mm256_blendv_ps(
        r, _mm256_sub_ps(_mm256_set1_ps(3.14159265358979323846f), tail),
        _mm256_cmp_ps(x, _mm256_set1_ps(-0.5f), _CMP_LT_OQ));
    return r;
}

// atan on float lanes.  Two range steps put the argument inside
// [-tan(pi/8), tan(pi/8)]: the reciprocal for |x| > 1, then the addition
// formula (u - t8)/(1 + u*t8).  The two offsets are folded back in at the
// end, and every corner falls out of the arithmetic: a NaN propagates
// through both blends and a discarded infinity never reaches the result.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_atan_f32(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 t8 = _mm256_set1_ps(0.41421356237309504880f);
    const __m256 ax = _mm256_andnot_ps(sign_bit, x);
    const __m256 big = _mm256_cmp_ps(ax, one, _CMP_GT_OQ);
    // True divides here rather than reciprocals with Newton steps: the
    // refinement multiplies its seed by (2 - d*seed), which turns an infinite
    // input into 0 * inf and would report NaN for atan(+-inf).
    const __m256 u = _mm256_blendv_ps(ax, _mm256_div_ps(one, ax), big);
    const __m256 over = _mm256_cmp_ps(u, t8, _CMP_GT_OQ);
    const __m256 v = _mm256_div_ps(_mm256_sub_ps(u, t8),
                                   _mm256_fmadd_ps(u, t8, one));
    const __m256 w = _mm256_blendv_ps(u, v, over);
    const __m256 y = _mm256_mul_ps(w, w);
    __m256 p = _mm256_set1_ps(-2.5157289730e-2f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(5.0106401570e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-6.5007834513e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(7.6725048792e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-9.0894109278e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.1111040776e-1f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-1.4285712355e-1f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.9999999973e-1f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-3.3333333333e-1f));
    const __m256 a = _mm256_fmadd_ps(_mm256_mul_ps(w, y), p, w);
    const __m256 lifted =
        _mm256_blendv_ps(a, _mm256_add_ps(a, _mm256_set1_ps(0.39269908169872415481f)), over);
    const __m256 r = _mm256_blendv_ps(
        lifted, _mm256_sub_ps(_mm256_set1_ps(1.57079632679489661923f), lifted), big);
    return _mm256_or_ps(_mm256_and_ps(sign_bit, x), r);
}

// Tangent's polynomial on an argument already reduced into [-pi/4, pi/4].
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_tan_poly_f32(__m256 r) {
    const __m256 y = _mm256_mul_ps(r, r);
    __m256 p = _mm256_set1_ps(2.1161404219e-4f);
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(-2.7136352998e-4f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(6.3633578712e-4f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(3.3067524298e-4f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.5654777697e-3f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(3.5614434028e-3f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(8.8688866982e-3f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(2.1868825315e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(5.3968300325e-2f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(1.3333333165e-1f));
    p = _mm256_fmadd_ps(p, y, _mm256_set1_ps(3.3333333336e-1f));
    return _mm256_fmadd_ps(_mm256_mul_ps(r, y), p, r);
}

// tan on float lanes.  The reduction runs in double, the polynomial in
// float.
//
// A float reduction cannot hold this: x * 2/pi rounds to 24 bits, which at
// |x| = 1e6 is already 0.03 -- enough to round to the wrong multiple of pi/2
// for about one lane in thirty -- and even with that repaired, the reduced
// argument carries an absolute error of k * (pi/2 - pio2_hi) * 2^-24, which
// the tangent's derivative turns into a relative error near a pole that is
// far above what the function should return.  Eight lanes become two double
// vectors, the reduction lands exact to well under a float ulp, and the
// result converts back before the polynomial.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_tan_f32(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256d two_over_pi = _mm256_set1_pd(0.63661977236758134308);
    const __m256d pio2_hi = _mm256_set1_pd(1.5707963267948966);
    const __m256d pio2_lo = _mm256_set1_pd(6.123233995736766e-17);
    const __m256d x0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
    const __m256d x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
    const __m256d k0 = _mm256_round_pd(
        _mm256_mul_pd(x0, two_over_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256d k1 = _mm256_round_pd(
        _mm256_mul_pd(x1, two_over_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d d0 = _mm256_fnmadd_pd(k0, pio2_hi, x0);
    __m256d d1 = _mm256_fnmadd_pd(k1, pio2_hi, x1);
    d0 = _mm256_fnmadd_pd(k0, pio2_lo, d0);
    d1 = _mm256_fnmadd_pd(k1, pio2_lo, d1);
    const __m256 r = _mm256_set_m128(_mm256_cvtpd_ps(d1), _mm256_cvtpd_ps(d0));
    const __m256i ki = _mm256_set_m128i(_mm256_cvtpd_epi32(k1), _mm256_cvtpd_epi32(k0));
    const __m256 t = v_tan_poly_f32(r);
    // An odd multiple of pi/2 in the reduction means the answer is -1/t.
    const __m256i odd = _mm256_cmpeq_epi32(
        _mm256_and_si256(ki, _mm256_set1_epi32(1)), _mm256_set1_epi32(1));
    return _mm256_blendv_ps(
        t, _mm256_xor_ps(v_rcp_nr_ps(t), sign_bit), _mm256_castsi256_ps(odd));
}
// elu takes its three coefficients as arguments instead of reading prm itself.
// The chunk loops below broadcast the parameters once per chunk; deriving them
// inside the vector loop means a double load, a narrowing convert and a
// broadcast per vector, and each of the three also needs a store plus a reload
// around the SLEEF call.  That bookkeeping costs more than the expm1
// polynomial it feeds.  apply_f32 keeps the one-vector form.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_elu_f32(__m256 negcoef, __m256 poscoef, __m256 negipt, __m256 x) {
    const __m256 zero = _mm256_setzero_ps();
    __m256 scaled = _mm256_mul_ps(x, negipt);
    __m256 neg = _mm256_mul_ps(tensorplay::tpsleef::expm1(scaled), negcoef);
    __m256 pos = _mm256_mul_ps(x, poscoef);
    return _mm256_blendv_ps(neg, pos, _mm256_cmp_ps(x, zero, _CMP_GE_OQ));
}

// softplus takes beta and threshold pre-broadcast for the same reason as
// v_elu_f32.  The divide stays in the input's own width, which is also what
// scalar_apply does, so a chunk and its scalar tail agree bit for bit.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_softplus_f32(__m256 beta, __m256 threshold, __m256 x) {
    __m256 bx = _mm256_mul_ps(x, beta);
    __m256 sp = _mm256_div_ps(tensorplay::tpsleef::log1p(v_exp_f32(bx)), beta);
    return _mm256_blendv_ps(sp, x, _mm256_cmp_ps(bx, threshold, _CMP_GT_OQ));
}

// Composites shared by the apply dispatcher and the per-op chunk loops, so
// both shapes emit identical code.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_sigmoid_f32(__m256 x, __m256 one, __m256 signbit) {
    // rcp+NR instead of divps.  Division corners reproduced: den==inf
    // (x <= -88.7, exp overflowed) must yield +0 like 1/inf, and a NaN
    // denominator must stay NaN -- hence the two mask repairs.
    __m256 den = _mm256_add_ps(one, v_exp_f32(_mm256_xor_ps(x, signbit)));
    __m256 r = v_rcp_nr_ps(den);
    __m256 not_inf = _mm256_cmp_ps(den, _mm256_set1_ps(INFINITY), _CMP_NEQ_OQ);
    __m256 zeroed = _mm256_and_ps(not_inf, r);
    __m256 is_nan = _mm256_cmp_ps(den, den, _CMP_UNORD_Q);
    return _mm256_blendv_ps(zeroed, r, is_nan);
}

// Sigmoid for blocks whose lanes are all inside [-87, 87].  Negating a lane
// keeps the exp argument inside the range where the result is a normal
// float, so the denominator is finite and at least one and neither of the
// two division-corner repairs in the general form has anything to repair.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_sigmoid_f32_fast(__m256 x, __m256 one, __m256 signbit) {
    const __m256 den =
        _mm256_add_ps(one, v_exp_f32_fast(_mm256_xor_ps(x, signbit)));
    return v_rcp_nr_ps(den);
}

template <bool Fast>
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_gelu_none_f32_impl(__m256 x, __m256 one) {
    const __m256 kAlpha = _mm256_set1_ps(static_cast<float>(0.70710678118654752440));
    __m256 cdf = _mm256_add_ps(one, v_erf_f32_impl<Fast>(_mm256_mul_ps(kAlpha, x)));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_gelu_none_f32(__m256 x, __m256 one) {
    return v_gelu_none_f32_impl<false>(x, one);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_gelu_none_f32_fast(__m256 x, __m256 one) {
    return v_gelu_none_f32_impl<true>(x, one);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_gelu_tanh_f32(__m256 x, __m256 one) {
    const __m256 kBeta = _mm256_set1_ps(static_cast<float>(1.41421356237309504880 * 1.12837916709551257390 * 0.5));
    const __m256 kKappa = _mm256_set1_ps(0.044715f);
    __m256 x_cube = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
    __m256 inner = _mm256_mul_ps(kBeta, _mm256_add_ps(x, _mm256_mul_ps(kKappa, x_cube)));
    __m256 cdf = _mm256_add_ps(one, tensorplay::tpsleef::tanh(inner));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
}

// Applies op to one AVX2 float vector.  Always-inline: only ever folded into
// target-attributed callers below.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 apply_f32(VOp op, VParams prm, __m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 signbit = _mm256_set1_ps(kSignBitF);
    switch (op) {
        case VOp::Abs: return _mm256_andnot_ps(signbit, x);
        case VOp::Neg: return _mm256_xor_ps(x, signbit);
        case VOp::Sign: {
            __m256 pos = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ), one);
            __m256 neg = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_LT_OQ), one);
            return _mm256_sub_ps(pos, neg);
        }
        case VOp::Square: return _mm256_mul_ps(x, x);
        case VOp::Reciprocal: return _mm256_div_ps(one, x);
        case VOp::Sqrt: return _mm256_sqrt_ps(x);
        case VOp::Rsqrt: return _mm256_div_ps(one, _mm256_sqrt_ps(x));
        case VOp::Floor: return _mm256_floor_ps(x);
        case VOp::Ceil: return _mm256_ceil_ps(x);
        case VOp::Trunc: return _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        case VOp::Round: return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        case VOp::Frac: return _mm256_sub_ps(x, _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        case VOp::Relu: return _mm256_max_ps(zero, x); // NaN propagates (second operand wins)
        case VOp::Exp: return v_exp_f32(x);
        case VOp::Exp2: return tensorplay::tpsleef::exp2(x);
        case VOp::Expm1: return tensorplay::tpsleef::expm1(x);
        case VOp::Log: return v_log_f32(x);
        case VOp::Log2: return tensorplay::tpsleef::log2(x);
        case VOp::Log10: return tensorplay::tpsleef::log10(x);
        case VOp::Log1p: return tensorplay::tpsleef::log1p(x);
        case VOp::Sin: return tensorplay::tpsleef::sin(x);
        case VOp::Cos: return tensorplay::tpsleef::cos(x);
        case VOp::Tan: return tensorplay::tpsleef::tan(x);
        case VOp::Asin: return v_asin_f32(x);
        case VOp::Acos: return v_acos_f32(x);
        case VOp::Atan: return v_atan_f32(x);
        case VOp::Sinh: return tensorplay::tpsleef::sinh(x);
        case VOp::Cosh: return tensorplay::tpsleef::cosh(x);
        case VOp::Tanh: return tensorplay::tpsleef::tanh(x);
        case VOp::Asinh: return tensorplay::tpsleef::asinh(x);
        case VOp::Acosh: return tensorplay::tpsleef::acosh(x);
        case VOp::Atanh: return tensorplay::tpsleef::atanh(x);
        case VOp::Erf: return v_erf_f32(x);
        case VOp::Erfc: return tensorplay::tpsleef::erfc(x);
        case VOp::Lgamma: {
            // Callers guarantee all lanes > 0 here.
            return v_lgamma_pos_f32(x);
        }
        case VOp::Sigmoid:
            return v_sigmoid_f32(x, one, signbit);
        case VOp::GeluNone:
            return v_gelu_none_f32(x, one);
        case VOp::GeluTanh:
            return v_gelu_tanh_f32(x, one);
        case VOp::Silu: {
            // x * rcp_nr(den) matches x/den bit-for-bit in every corner here:
            // den==inf makes the NR chain NaN exactly like -inf/+inf division,
            // and finite dens get a float-exact reciprocal.
            __m256 den = _mm256_add_ps(one, v_exp_f32(_mm256_xor_ps(x, signbit)));
            return _mm256_mul_ps(x, v_rcp_nr_ps(den));
        }
        case VOp::Mish: {
            // Uses the scalar expression: log((1 + exp(x))) * tanh(...) — plain
            // log, not log1p, and overflow-to-inf semantics are preserved.
            __m256 sp = v_log_f32(_mm256_add_ps(one, v_exp_f32(x)));
            return _mm256_mul_ps(x, tensorplay::tpsleef::tanh(sp));
        }
        case VOp::Selu: {
            const __m256 lambda = _mm256_set1_ps(static_cast<float>(1.0507009873554804934193349852946));
            const __m256 alphalambda = _mm256_set1_ps(static_cast<float>(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946));
            __m256 pos = _mm256_mul_ps(x, lambda);
            __m256 neg = _mm256_mul_ps(alphalambda, tensorplay::tpsleef::expm1(x));
            return _mm256_blendv_ps(neg, pos, _mm256_cmp_ps(x, zero, _CMP_GT_OQ));
        }
        case VOp::Elu:
            return v_elu_f32(_mm256_set1_ps(static_cast<float>(prm.p0)),
                             _mm256_set1_ps(static_cast<float>(prm.p1)),
                             _mm256_set1_ps(static_cast<float>(prm.p2)), x);
        case VOp::Softplus:
            return v_softplus_f32(_mm256_set1_ps(static_cast<float>(prm.p0)),
                                  _mm256_set1_ps(static_cast<float>(prm.p1)), x);
        case VOp::Hardswish: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            __m256 y = _mm256_add_ps(x, three);
            y = _mm256_min_ps(six, _mm256_max_ps(zero, y));
            return _mm256_div_ps(_mm256_mul_ps(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            __m256 y = _mm256_add_ps(x, three);
            y = _mm256_min_ps(six, _mm256_max_ps(zero, y));
            return _mm256_div_ps(y, six);
        }
        case VOp::LeakyRelu: {
            const __m256 slope = _mm256_set1_ps(static_cast<float>(prm.p0));
            __m256 neg = _mm256_mul_ps(slope, x);
            return _mm256_blendv_ps(x, neg, _mm256_cmp_ps(x, zero, _CMP_LT_OQ));
        }
        case VOp::Hardtanh: {
            const __m256 lo = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 hi = _mm256_set1_ps(static_cast<float>(prm.p1));
            return _mm256_min_ps(hi, _mm256_max_ps(lo, x));
        }
        case VOp::Relu6:
            return _mm256_min_ps(_mm256_set1_ps(6.0f), _mm256_max_ps(zero, x));
        default: return x;
    }
}

// One branch per chunk instead of one per vector.  The cheap ops are one or
// two instructions wide, so an op test inside the loop costs more than their
// arithmetic; transcendentals and composites keep the generic loop, where the
// polynomial work dominates the dispatch either way.
#define TP_F32_SIMPLE_LOOP_8(EXPR)                                       \
    {                                                                    \
        for (; i + 32 <= n; i += 32) {                                   \
            __m256 x = _mm256_loadu_ps(s + i);                           \
            _mm256_storeu_ps(d + i, (EXPR));                             \
            x = _mm256_loadu_ps(s + i + 8);                              \
            _mm256_storeu_ps(d + i + 8, (EXPR));                          \
            x = _mm256_loadu_ps(s + i + 16);                             \
            _mm256_storeu_ps(d + i + 16, (EXPR));                         \
            x = _mm256_loadu_ps(s + i + 24);                             \
            _mm256_storeu_ps(d + i + 24, (EXPR));                         \
        }                                                                \
        for (; i + 8 <= n; i += 8) {                                     \
            __m256 x = _mm256_loadu_ps(s + i);                           \
            _mm256_storeu_ps(d + i, (EXPR));                             \
        }                                                                \
    }

__attribute__((target("avx2,fma"), noinline))
static void f32_chunk_avx2(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
    const float* s = src + b;
    float* d = dst + b;
    const int64_t n = e - b;
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 signbit = _mm256_set1_ps(kSignBitF);
    int64_t i = 0;
    switch (op) {
        case VOp::Abs:
            TP_F32_SIMPLE_LOOP_8(_mm256_andnot_ps(signbit, x));
            break;
        case VOp::Neg:
            TP_F32_SIMPLE_LOOP_8(_mm256_xor_ps(x, signbit));
            break;
        case VOp::Sign:
            TP_F32_SIMPLE_LOOP_8(
                _mm256_sub_ps(_mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ), one),
                              _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_LT_OQ), one)));
            break;
        case VOp::Square:
            TP_F32_SIMPLE_LOOP_8(_mm256_mul_ps(x, x));
            break;
        case VOp::Reciprocal:
            TP_F32_SIMPLE_LOOP_8(_mm256_div_ps(one, x));
            break;
        case VOp::Sqrt:
            TP_F32_SIMPLE_LOOP_8(_mm256_sqrt_ps(x));
            break;
        case VOp::Rsqrt:
            TP_F32_SIMPLE_LOOP_8(_mm256_div_ps(one, _mm256_sqrt_ps(x)));
            break;
        case VOp::Floor:
            TP_F32_SIMPLE_LOOP_8(_mm256_floor_ps(x));
            break;
        case VOp::Ceil:
            TP_F32_SIMPLE_LOOP_8(_mm256_ceil_ps(x));
            break;
        case VOp::Trunc:
            TP_F32_SIMPLE_LOOP_8(_mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            break;
        case VOp::Round:
            TP_F32_SIMPLE_LOOP_8(_mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            break;
        case VOp::Frac:
            TP_F32_SIMPLE_LOOP_8(
                _mm256_sub_ps(x, _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
            break;
        case VOp::Relu:
            TP_F32_SIMPLE_LOOP_8(_mm256_max_ps(zero, x));  // NaN propagates (second operand wins)
            break;
        case VOp::Hardtanh: {
            const __m256 lo = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 hi = _mm256_set1_ps(static_cast<float>(prm.p1));
            TP_F32_SIMPLE_LOOP_8(_mm256_min_ps(hi, _mm256_max_ps(lo, x)));
            break;
        }
        case VOp::Elu: {
            const __m256 negcoef = _mm256_set1_ps(static_cast<float>(prm.p0)); // alpha*scale
            const __m256 poscoef = _mm256_set1_ps(static_cast<float>(prm.p1)); // scale
            const __m256 negipt = _mm256_set1_ps(static_cast<float>(prm.p2));  // input_scale
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(d + i, v_elu_f32(negcoef, poscoef, negipt, x));
            }
            break;
        }
        case VOp::Softplus: {
            const __m256 beta = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 threshold = _mm256_set1_ps(static_cast<float>(prm.p1));
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(d + i, v_softplus_f32(beta, threshold, x));
            }
            break;
        }
        case VOp::Exp: {
            // Whole blocks that stay inside the range where exp is a normal
            // float take the short chain; a block holding anything else
            // (overflow, underflow, NaN, inf) runs the guarded version, so
            // the result does not depend on how the data is blocked.
            const __m256 exp_lo = _mm256_set1_ps(-87.0f);
            const __m256 exp_hi = _mm256_set1_ps(88.0f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                // Unordered-quiet predicates, so a NaN lane fails the range
                // test instead of being dropped by a min/max reduction.
                __m256 bad = _mm256_or_ps(
                    _mm256_or_ps(_mm256_cmp_ps(x0, exp_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x0, exp_hi, _CMP_NLE_UQ)),
                    _mm256_or_ps(_mm256_cmp_ps(x1, exp_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x1, exp_hi, _CMP_NLE_UQ)));
                bad = _mm256_or_ps(bad, _mm256_or_ps(
                    _mm256_or_ps(_mm256_cmp_ps(x2, exp_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x2, exp_hi, _CMP_NLE_UQ)),
                    _mm256_or_ps(_mm256_cmp_ps(x3, exp_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x3, exp_hi, _CMP_NLE_UQ))));
                if (_mm256_movemask_ps(bad) == 0) {
                    _mm256_storeu_ps(d + i, v_exp_f32_fast(x0));
                    _mm256_storeu_ps(d + i + 8, v_exp_f32_fast(x1));
                    _mm256_storeu_ps(d + i + 16, v_exp_f32_fast(x2));
                    _mm256_storeu_ps(d + i + 24, v_exp_f32_fast(x3));
                } else {
                    _mm256_storeu_ps(d + i, v_exp_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_exp_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_exp_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_exp_f32(x3));
                }
            }
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(d + i, v_exp_f32(_mm256_loadu_ps(s + i)));
            }
            break;
        }
        case VOp::Expm1:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::expm1(x));
            break;
        case VOp::Log:
            log_range_loop<LogBase::E>(s, d, n, i);
            break;
        case VOp::Log2:
            log_range_loop<LogBase::Two>(s, d, n, i);
            break;
        case VOp::Log10:
            log_range_loop<LogBase::Ten>(s, d, n, i);
            break;
        case VOp::Log1p:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::log1p(x));
            break;
        case VOp::Tanh:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::tanh(x));
            break;
        case VOp::Sigmoid: {
            // Blocks inside [-87, 87] cannot make the denominator overflow or
            // turn NaN, so they take the form without the corner repairs.
            const __m256 sig_lo = _mm256_set1_ps(-87.0f);
            const __m256 sig_hi = _mm256_set1_ps(87.0f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                __m256 bad = _mm256_or_ps(
                    _mm256_or_ps(_mm256_cmp_ps(x0, sig_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x0, sig_hi, _CMP_NLE_UQ)),
                    _mm256_or_ps(_mm256_cmp_ps(x1, sig_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x1, sig_hi, _CMP_NLE_UQ)));
                bad = _mm256_or_ps(bad, _mm256_or_ps(
                    _mm256_or_ps(_mm256_cmp_ps(x2, sig_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x2, sig_hi, _CMP_NLE_UQ)),
                    _mm256_or_ps(_mm256_cmp_ps(x3, sig_lo, _CMP_NGE_UQ),
                                 _mm256_cmp_ps(x3, sig_hi, _CMP_NLE_UQ))));
                if (_mm256_movemask_ps(bad) == 0) {
                    _mm256_storeu_ps(d + i, v_sigmoid_f32_fast(x0, one, signbit));
                    _mm256_storeu_ps(d + i + 8, v_sigmoid_f32_fast(x1, one, signbit));
                    _mm256_storeu_ps(d + i + 16, v_sigmoid_f32_fast(x2, one, signbit));
                    _mm256_storeu_ps(d + i + 24, v_sigmoid_f32_fast(x3, one, signbit));
                } else {
                    _mm256_storeu_ps(d + i, v_sigmoid_f32(x0, one, signbit));
                    _mm256_storeu_ps(d + i + 8, v_sigmoid_f32(x1, one, signbit));
                    _mm256_storeu_ps(d + i + 16, v_sigmoid_f32(x2, one, signbit));
                    _mm256_storeu_ps(d + i + 24, v_sigmoid_f32(x3, one, signbit));
                }
            }
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(d + i, v_sigmoid_f32(x, one, signbit));
            }
            break;
        }
        case VOp::GeluNone: {
            // The erf inside scales by 1/sqrt(2), so |x| <= 13 is what keeps
            // its decay argument above the short chain's floor.
            const __m256 limit = _mm256_set1_ps(13.0f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                __m256 bad = _mm256_or_ps(
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x0), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x1), limit, _CMP_NLE_UQ)),
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x2), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x3), limit, _CMP_NLE_UQ)));
                if (_mm256_movemask_ps(bad) == 0) {
                    _mm256_storeu_ps(d + i, v_gelu_none_f32_fast(x0, one));
                    _mm256_storeu_ps(d + i + 8, v_gelu_none_f32_fast(x1, one));
                    _mm256_storeu_ps(d + i + 16, v_gelu_none_f32_fast(x2, one));
                    _mm256_storeu_ps(d + i + 24, v_gelu_none_f32_fast(x3, one));
                } else {
                    _mm256_storeu_ps(d + i, v_gelu_none_f32(x0, one));
                    _mm256_storeu_ps(d + i + 8, v_gelu_none_f32(x1, one));
                    _mm256_storeu_ps(d + i + 16, v_gelu_none_f32(x2, one));
                    _mm256_storeu_ps(d + i + 24, v_gelu_none_f32(x3, one));
                }
            }
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(d + i, v_gelu_none_f32(x, one));
            }
            break;
        }
        case VOp::GeluTanh:
            TP_F32_SIMPLE_LOOP_8(v_gelu_tanh_f32(x, one));
            break;
        case VOp::Lgamma:
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                if (_mm256_movemask_ps(_mm256_cmp_ps(x, zero, _CMP_LE_OQ))) {
                    // Non-positive lane (or NaN): scalar fallback keeps the exact
                    // reflection-domain behaviour of std::lgamma.
                    for (int64_t j = i; j < i + 8; ++j) d[j] = scalar_apply(op, prm, s[j]);
                } else {
                    _mm256_storeu_ps(d + i, v_lgamma_pos_f32(x));
                }
            }
            break;
        case VOp::Relu6:
            TP_F32_SIMPLE_LOOP_8(_mm256_min_ps(_mm256_set1_ps(6.0f), _mm256_max_ps(zero, x)));
            break;
        case VOp::LeakyRelu: {
            const __m256 slope = _mm256_set1_ps(static_cast<float>(prm.p0));
            TP_F32_SIMPLE_LOOP_8(_mm256_blendv_ps(x, _mm256_mul_ps(slope, x),
                                                  _mm256_cmp_ps(x, zero, _CMP_LT_OQ)));
            break;
        }
        case VOp::Hardswish: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            TP_F32_SIMPLE_LOOP_8(_mm256_div_ps(
                _mm256_mul_ps(x, _mm256_min_ps(six, _mm256_max_ps(zero, _mm256_add_ps(x, three)))), six));
            break;
        }
        case VOp::Hardsigmoid: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            TP_F32_SIMPLE_LOOP_8(_mm256_div_ps(
                _mm256_min_ps(six, _mm256_max_ps(zero, _mm256_add_ps(x, three))), six));
            break;
        }
        case VOp::Sin:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::sin(x));
            break;
        case VOp::Cos:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::cos(x));
            break;
        case VOp::Tan: {
            // The two-term pi/2 reduction holds while |x| * 2/pi still has
            // enough low bits to round the multiple of pi/2 correctly; past
            // 1e6 the library's multi-precision reduction takes the block.
            const __m256 limit = _mm256_set1_ps(1.0e6f);
            // Inside [-pi/4, pi/4] the reduction is the identity, so the
            // polynomial alone answers the block and the double round trip
            // disappears.
            const __m256 quarter_pi = _mm256_set1_ps(0.78539816339744830962f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                // One magnitude for the whole block drives both range tests.
                // max drops a NaN operand, which only ever routes a NaN lane
                // to a path that returns NaN anyway.
                const __m256 m = _mm256_max_ps(
                    _mm256_max_ps(_mm256_andnot_ps(signbit, x0),
                                  _mm256_andnot_ps(signbit, x1)),
                    _mm256_max_ps(_mm256_andnot_ps(signbit, x2),
                                  _mm256_andnot_ps(signbit, x3)));
                if (_mm256_movemask_ps(_mm256_cmp_ps(m, quarter_pi, _CMP_LE_OQ)) == 0xff) {
                    _mm256_storeu_ps(d + i, v_tan_poly_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_tan_poly_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_tan_poly_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_tan_poly_f32(x3));
                    continue;
                }
                if (_mm256_movemask_ps(_mm256_cmp_ps(m, limit, _CMP_NLE_UQ)) == 0) {
                    _mm256_storeu_ps(d + i, v_tan_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_tan_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_tan_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_tan_f32(x3));
                } else {
                    _mm256_storeu_ps(d + i, tensorplay::tpsleef::tan(x0));
                    _mm256_storeu_ps(d + i + 8, tensorplay::tpsleef::tan(x1));
                    _mm256_storeu_ps(d + i + 16, tensorplay::tpsleef::tan(x2));
                    _mm256_storeu_ps(d + i + 24, tensorplay::tpsleef::tan(x3));
                }
            }
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(d + i, tensorplay::tpsleef::tan(_mm256_loadu_ps(s + i)));
            }
            break;
        }
        case VOp::Asin:
            TP_F32_SIMPLE_LOOP_8(v_asin_f32(x));
            break;
        case VOp::Acos:
            TP_F32_SIMPLE_LOOP_8(v_acos_f32(x));
            break;
        case VOp::Atan:
            TP_F32_SIMPLE_LOOP_8(v_atan_f32(x));
            break;
        case VOp::Erf: {
            // |x| <= 9 keeps -x*x inside the range where the short exp chain
            // is valid; past it erf has saturated and the guarded form runs.
            const __m256 limit = _mm256_set1_ps(9.0f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                __m256 bad = _mm256_or_ps(
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x0), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x1), limit, _CMP_NLE_UQ)),
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x2), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x3), limit, _CMP_NLE_UQ)));
                if (_mm256_movemask_ps(bad) == 0) {
                    _mm256_storeu_ps(d + i, v_erf_f32_fast(x0));
                    _mm256_storeu_ps(d + i + 8, v_erf_f32_fast(x1));
                    _mm256_storeu_ps(d + i + 16, v_erf_f32_fast(x2));
                    _mm256_storeu_ps(d + i + 24, v_erf_f32_fast(x3));
                } else {
                    _mm256_storeu_ps(d + i, v_erf_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_erf_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_erf_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_erf_f32(x3));
                }
            }
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(d + i, v_erf_f32(_mm256_loadu_ps(s + i)));
            }
            break;
        }
        case VOp::Erfc: {
            // |x| <= 9 keeps -x*x above the short exp chain's floor.
            const __m256 limit = _mm256_set1_ps(9.0f);
            for (; i + 32 <= n; i += 32) {
                __m256 x0 = _mm256_loadu_ps(s + i);
                __m256 x1 = _mm256_loadu_ps(s + i + 8);
                __m256 x2 = _mm256_loadu_ps(s + i + 16);
                __m256 x3 = _mm256_loadu_ps(s + i + 24);
                __m256 bad = _mm256_or_ps(
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x0), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x1), limit, _CMP_NLE_UQ)),
                    _mm256_or_ps(
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x2), limit, _CMP_NLE_UQ),
                        _mm256_cmp_ps(_mm256_andnot_ps(signbit, x3), limit, _CMP_NLE_UQ)));
                if (_mm256_movemask_ps(bad) == 0) {
                    _mm256_storeu_ps(d + i, v_erfc_f32_fast(x0));
                    _mm256_storeu_ps(d + i + 8, v_erfc_f32_fast(x1));
                    _mm256_storeu_ps(d + i + 16, v_erfc_f32_fast(x2));
                    _mm256_storeu_ps(d + i + 24, v_erfc_f32_fast(x3));
                } else {
                    _mm256_storeu_ps(d + i, v_erfc_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_erfc_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_erfc_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_erfc_f32(x3));
                }
            }
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(d + i, v_erfc_f32(_mm256_loadu_ps(s + i)));
            }
            break;
        }
        case VOp::Silu:
            TP_F32_SIMPLE_LOOP_8(_mm256_mul_ps(
                x, v_rcp_nr_ps(_mm256_add_ps(one, v_exp_f32(_mm256_xor_ps(x, signbit))))));
            break;
        case VOp::Selu: {
            const __m256 lambda = _mm256_set1_ps(static_cast<float>(1.0507009873554804934193349852946));
            const __m256 alphalambda = _mm256_set1_ps(static_cast<float>(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946));
            TP_F32_SIMPLE_LOOP_8(_mm256_blendv_ps(
                _mm256_mul_ps(alphalambda, tensorplay::tpsleef::expm1(x)),
                _mm256_mul_ps(x, lambda), _mm256_cmp_ps(x, zero, _CMP_GT_OQ)));
            break;
        }
        case VOp::Mish:
            TP_F32_SIMPLE_LOOP_8(_mm256_mul_ps(
                x, tensorplay::tpsleef::tanh(v_log_f32(
                       _mm256_add_ps(one, v_exp_f32(x))))));
            break;
        default:
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(d + i, apply_f32(op, prm, x));
            }
            break;
    }
#undef TP_F32_SIMPLE_LOOP_8
    for (; i < n; ++i) d[i] = scalar_apply(op, prm, s[i]);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_elu_f64(__m256d negcoef, __m256d poscoef, __m256d negipt, __m256d x) {
    const __m256d zero = _mm256_setzero_pd();
    __m256d scaled = _mm256_mul_pd(x, negipt);
    __m256d neg = _mm256_mul_pd(tensorplay::tpsleef::expm1(scaled), negcoef);
    __m256d pos = _mm256_mul_pd(x, poscoef);
    return _mm256_blendv_pd(neg, pos, _mm256_cmp_pd(x, zero, _CMP_GE_OQ));
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_softplus_f64(__m256d beta, __m256d threshold, __m256d x) {
    __m256d bx = _mm256_mul_pd(x, beta);
    __m256d sp = _mm256_div_pd(tensorplay::tpsleef::log1p(tensorplay::tpsleef::exp(bx)), beta);
    return _mm256_blendv_pd(sp, x, _mm256_cmp_pd(bx, threshold, _CMP_GT_OQ));
}

// exp on double lanes.  The argument loses k*ln2 in two parts, so the reduced
// argument keeps the bits a single subtraction would round away, and the
// series is a degree-12 fit over [-ln2/2, ln2/2] with a relative error of
// 6e-20.  The scale is applied as two half exponents: both halves stay normal
// for every k, so a result that lands in the subnormal range is rounded once,
// by the final multiply.
__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_exp_f64(__m256d x) {
    const __m256d kLog2e = _mm256_set1_pd(1.4426950408889634074);
    const __m256d ln2_hi = _mm256_set1_pd(0x1.62e42fef00000p-1);
    const __m256d ln2_lo = _mm256_set1_pd(0x1.473de6af278edp-34);
    // The argument is clamped, not the exponent: the reduction has to stay
    // consistent with the clamp, or a lane far outside the range would come
    // back with a reduced argument the series cannot represent.  The bounds
    // sit just past the overflow and underflow thresholds, so every result
    // outside them is already inf or zero.  A NaN survives both clamps,
    // because the second operand only wins when the first is not NaN.
    x = _mm256_min_pd(_mm256_set1_pd(709.9), x);
    x = _mm256_max_pd(_mm256_set1_pd(-745.2), x);
    __m256d k = _mm256_round_pd(_mm256_mul_pd(x, kLog2e),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, ln2_hi, x);
    r = _mm256_fnmadd_pd(k, ln2_lo, r);
    __m256d p = _mm256_set1_pd(2.0921579964950299e-09);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.5114869528657765e-08));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.7557251904174915e-07));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.7557224958392624e-06));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.4801587350110797e-05));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.000198412699092171));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.0013888888888871889));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.0083333333333095276));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.041666666666666692));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.16666666666666702));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.5));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    // The two halves are split in integer form rather than by rounding k/2 in
    // floating point: ceil(k/2) and the remainder still add back to k, and the
    // vector only has to be converted out once.
    const __m128i k32 = _mm256_cvtpd_epi32(k);
    const __m128i hi = _mm_srai_epi32(_mm_add_epi32(k32, _mm_set1_epi32(1)), 1);
    const __m256i bias = _mm256_set1_epi64x(1023);
    const __m256d s1 = _mm256_castsi256_pd(_mm256_slli_epi64(
        _mm256_add_epi64(_mm256_cvtepi32_epi64(hi), bias), 52));
    const __m256d s2 = _mm256_castsi256_pd(_mm256_slli_epi64(
        _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm_sub_epi32(k32, hi)), bias), 52));
    return _mm256_mul_pd(_mm256_mul_pd(p, s1), s2);
}

enum class LogBase64 { E, Two, Ten };

constexpr double log_base_scale64(LogBase64 base) {
    return base == LogBase64::Two ? 1.4426950408889634074
         : base == LogBase64::Ten ? 0.43429448190325182765
                                  : 1.0;
}

// log on double lanes.  The mantissa is folded into [sqrt(1/2), sqrt(2)) and
// its logarithm comes from the atanh series in s = (m-1)/(m+1): |s| <= 0.1716,
// so the series splits into the leading 2s plus a correction of at most 3.4e-3
// of it, and only the final add rounds at the result's own magnitude.  The
// exponent term is split in two so that e*ln2 stays exact.
template <LogBase64 Base>
__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_log_f64(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d ax = _mm256_andnot_pd(signbit, x);
    const __m256d den = _mm256_cmp_pd(ax, _mm256_set1_pd(0x1p-1022), _CMP_LT_OQ);
    const __m256d xs = _mm256_mul_pd(
        x, _mm256_blendv_pd(one, _mm256_set1_pd(0x1p54), den));
    const __m256i bits = _mm256_and_si256(_mm256_castpd_si256(xs),
                                          _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    // The biased exponent is spliced into the low bits of 2^52, so one
    // subtraction removes both the bias and the exponent offset.
    __m256d e = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(
            _mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(0x4330000000000000LL))),
        _mm256_set1_pd(0x1.00000000003ffp52));
    e = _mm256_sub_pd(e, _mm256_and_pd(den, _mm256_set1_pd(54.0)));
    __m256d m = _mm256_castsi256_pd(_mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
        _mm256_set1_epi64x(0x3FF0000000000000LL)));
    const __m256d big = _mm256_cmp_pd(m, _mm256_set1_pd(1.4142135623730951),
                                      _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), big);
    e = _mm256_add_pd(e, _mm256_and_pd(big, one));
    const __m256d s = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    const __m256d y = _mm256_mul_pd(s, s);
    __m256d q = _mm256_set1_pd(2.0 / 23.0);
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 21.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 19.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 17.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 15.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 13.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 11.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 9.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 7.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 5.0));
    q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(2.0 / 3.0));
    constexpr double kScale = log_base_scale64(Base);
    const __m256d lead = _mm256_mul_pd(s, _mm256_set1_pd(2.0 * kScale));
    const __m256d corr = _mm256_mul_pd(
        _mm256_mul_pd(s, y), _mm256_mul_pd(q, _mm256_set1_pd(kScale)));
    const __m256d logm = _mm256_add_pd(lead, corr);
    __m256d r;
    if constexpr (Base == LogBase64::Two) {
        r = _mm256_add_pd(e, logm);
    } else if constexpr (Base == LogBase64::Ten) {
        r = _mm256_fmadd_pd(e, _mm256_set1_pd(0x1.3441350900000p-2),
                            _mm256_fmadd_pd(e, _mm256_set1_pd(0x1.ef3fde623e256p-35), logm));
    } else {
        r = _mm256_fmadd_pd(e, _mm256_set1_pd(0x1.62e42fef00000p-1),
                            _mm256_fmadd_pd(e, _mm256_set1_pd(0x1.473de6af278edp-34), logm));
    }
    const __m256d inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000LL));
    const __m256d nan = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF8000000000000LL));
    // A NaN carries mantissa bits that decode as an ordinary fraction, so the
    // unordered case is folded into the same test as the negative one.
    r = _mm256_blendv_pd(r, nan, _mm256_cmp_pd(x, zero, _CMP_NGT_UQ));
    r = _mm256_blendv_pd(r, _mm256_xor_pd(inf, signbit),
                         _mm256_cmp_pd(x, zero, _CMP_EQ_OQ));
    r = _mm256_blendv_pd(r, inf, _mm256_cmp_pd(x, inf, _CMP_EQ_OQ));
    return r;
}


// The complementary error function on double lanes.  Below |x| = 1 the
// complement comes straight from the odd series in x^2 -- a degree-12 fit over
// y in [0, 1] with a relative error of 1.4e-19 -- and above it the tail is
// exp(-x^2) times a smooth factor in u = 1/x.  One polynomial cannot hold that
// factor over the whole tail: it is fitted per octave of x, three degree-14
// pieces over u in [1/2, 1], [1/4, 1/2] and [1/27, 1/4], each evaluated in its
// own [-1, 1] variable and each accurate to 2e-16.  The square's rounding is
// carried into the exponential, without which the decay loses its low bits
// once x^2 is large enough for the rounding to move the exponent.
__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_erfc_tail_f64(__m256d ax) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d sq = _mm256_mul_pd(ax, ax);
    // inf - inf would be a NaN, so the correction is masked off far outside
    // the range where the rounding of x^2 can still move the result.
    const __m256d sq_err = _mm256_and_pd(
        _mm256_fnmadd_pd(ax, ax, sq),
        _mm256_cmp_pd(ax, _mm256_set1_pd(1.0e100), _CMP_LT_OQ));
    __m256d decay = v_exp_f64(_mm256_xor_pd(sq, signbit));
    decay = _mm256_fmadd_pd(decay, sq_err, decay);
    const __m256d u = _mm256_div_pd(one, ax);

    __m256d t = _mm256_fmadd_pd(u, _mm256_set1_pd(9.391304347826086),
                                _mm256_set1_pd(-1.3478260869565217));
    __m256d q3 = _mm256_set1_pd(1.270557215101025e-12);
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-1.8649648578977482e-12));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-2.3103889145888251e-11));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(1.244198439894267e-10));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(1.7271541321439504e-11));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-3.2425988759683361e-09));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(1.3829296143901666e-08));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(4.0439111789263601e-08));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-6.8591402601231557e-07));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(1.5388279071976929e-06));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(2.7903843311676303e-05));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-0.00025153621852648002));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(-0.0012489683856544268));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(0.058308884888871725));
    q3 = _mm256_fmadd_pd(q3, t, _mm256_set1_pd(0.080162269439375572));

    t = _mm256_fmadd_pd(u, _mm256_set1_pd(8.0), _mm256_set1_pd(-3.0));
    __m256d q2 = _mm256_set1_pd(4.5993232822733419e-14);
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-1.1795930729809297e-12));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(8.4940707554143757e-12));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-3.0655432410488974e-11));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-3.5305885227141051e-11));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(1.4900161035285428e-09));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-1.2197543406237606e-08));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(4.7855642561464015e-08));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(1.365005441045398e-07));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-3.9813218669108685e-06));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(3.2632400276842944e-05));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-4.9930615067898893e-05));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(-0.0029078207025937937));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(0.059251971965833344));
    q2 = _mm256_fmadd_pd(q2, t, _mm256_set1_pd(0.19907263099386563));

    t = _mm256_fmadd_pd(u, _mm256_set1_pd(4.0), _mm256_set1_pd(-3.0));
    __m256d q1 = _mm256_set1_pd(-1.1457539506950107e-11);
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(8.9711182438166595e-11));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-4.4142424933497509e-10));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(1.6867660545836935e-09));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-4.1926104260693363e-09));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-3.9729628449742441e-09));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(1.3591667276429934e-07));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-1.0800984559415769e-06));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(5.8840374634930062e-06));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-2.1678901569770582e-05));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(9.5464777288133952e-06));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(0.00077449644430332043));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(-0.0096606295821218147));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(0.085342214674858641));
    q1 = _mm256_fmadd_pd(q1, t, _mm256_set1_pd(0.35113469402890529));

    __m256d q = _mm256_blendv_pd(
        q3, q2, _mm256_cmp_pd(u, _mm256_set1_pd(0.25), _CMP_GE_OQ));
    q = _mm256_blendv_pd(
        q, q1, _mm256_cmp_pd(u, _mm256_set1_pd(0.5), _CMP_GE_OQ));
    return _mm256_mul_pd(decay, q);
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_erfc_f64(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d ax = _mm256_andnot_pd(signbit, x);
    // The series is fitted in t = 2x^2 - 1 over [-1, 1].
    const __m256d t = _mm256_fmadd_pd(
        _mm256_mul_pd(x, x), _mm256_set1_pd(2.0), _mm256_set1_pd(-1.0));
    __m256d p = _mm256_set1_pd(1.4543887079465116e-14);
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-3.8078824063194953e-13));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(9.167619596865806e-12));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-2.0352284058157691e-10));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(4.1158005832081488e-09));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-7.5115692865917272e-08));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.223382738359897e-06));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-1.7537169441124866e-05));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(0.00021751715604159932));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-0.0022854855611441132));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(0.019852496688983708));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-0.14053608902271711));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(0.96546873866986727));
    const __m256d near = _mm256_fnmadd_pd(x, p, one);
    const __m256d tail = v_erfc_tail_f64(ax);
    __m256d r = _mm256_blendv_pd(
        tail, near, _mm256_cmp_pd(ax, one, _CMP_LE_OQ));
    r = _mm256_blendv_pd(
        r, _mm256_sub_pd(_mm256_set1_pd(2.0), tail),
        _mm256_cmp_pd(x, _mm256_set1_pd(-1.0), _CMP_LT_OQ));
    // The tail is a product of a zero decay and a signed factor, so the value
    // at +inf would otherwise come back as a negative zero.
    return _mm256_add_pd(r, _mm256_setzero_pd());
}

// erf on double lanes.  Up to 2 the quotient erf(x)/x is a degree-17 fit in
// t = x^2/2 - 1, a form with no subtraction in it, so the relative accuracy of
// the series survives to the result.  Past 2 the same quotient is reached
// through the complement, taken as the reciprocal of a degree-19 fit of
// erfc(x)^(-1/16) raised to the sixteenth power: four squarings stand in for
// the exponential and its argument reduction, and the fitted factor only spans
// 1.4 to 11 over the range.  The input is clamped at 6, where the complement is
// already below half an ulp of one, so every lane past it answers exactly 1.
// A NaN passes the clamp untouched because it is the second operand, which is
// the one the minimum instruction returns when either side is unordered.
__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_erf_f64(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d ax = _mm256_andnot_pd(signbit, x);
    const __m256d t = _mm256_fmadd_pd(_mm256_mul_pd(ax, ax),
                                      _mm256_set1_pd(0.5), _mm256_set1_pd(-1.0));
    __m256d p = _mm256_set1_pd(-1.88741637439674822e-12);
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.72188581554841830e-11));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-1.40155406333159304e-10));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.13316117879612726e-09));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-8.61766514420579149e-09));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(6.11973590542225037e-08));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-4.04509494736763476e-07));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(2.47580105876321681e-06));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-1.39462737275972513e-05));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(7.18010087814341700e-05));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-3.35143533529314627e-04));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.40509142358536342e-03));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-5.23487583615802731e-03));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.71283445718231446e-02));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-4.86627774491550924e-02));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(1.19479138609851626e-01));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(-2.61111860931245388e-01));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(6.74933236039655049e-01));
    const __m256d near = _mm256_mul_pd(x, p);
    const __m256d ac = _mm256_min_pd(_mm256_set1_pd(6.0), ax);
    const __m256d tf = _mm256_mul_pd(_mm256_sub_pd(ac, _mm256_set1_pd(4.0)),
                                     _mm256_set1_pd(0.5));
    __m256d q = _mm256_set1_pd(1.03838257208176768e-09);
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(5.71343613323906640e-09));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.39829938878930770e-08));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(1.20093813619575749e-07));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(5.85482141133101655e-07));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.71759705131416074e-06));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(1.20931192017991857e-05));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(5.15683245066669922e-05));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.10250621946793831e-04));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(8.13158231721587362e-04));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.98004904880838986e-03));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(1.02438853513442570e-02));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(3.28915725965540015e-02));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(9.74006074349214185e-02));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.62969278706932041e-01));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(6.36422385744002450e-01));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(1.33680714216341312e+00));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(2.38025074259940705e+00));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(3.16880692459380864e+00));
    q = _mm256_fmadd_pd(q, tf, _mm256_set1_pd(3.07786490674295621e+00));
    __m256d rr = _mm256_div_pd(one, q);
    rr = _mm256_mul_pd(rr, rr);
    rr = _mm256_mul_pd(rr, rr);
    rr = _mm256_mul_pd(rr, rr);
    rr = _mm256_mul_pd(rr, rr);
    const __m256d far = _mm256_sub_pd(one, rr);
    const __m256d r = _mm256_blendv_pd(
        far, near, _mm256_cmp_pd(ax, _mm256_set1_pd(2.0), _CMP_LE_OQ));
    return _mm256_or_pd(r, _mm256_and_pd(x, signbit));
}

// tanh on double lanes.  Below 0.5 the odd series in x^2 answers directly --
// a degree-11 fit over x^2 in [0, 0.25] with a relative error of 6e-20 -- and
// above it 1 - 2/(exp(2|x|)+1) has no cancellation left to lose.  Both are
// evaluated and blended, which costs one exp more than a branch and keeps the
// lanes independent.
__attribute__((target("avx2,fma"), always_inline))
inline __m256d v_tanh_f64(__m256d x) {
    const __m256d signbit = _mm256_set1_pd(-0.0);
    const __m256d ax = _mm256_andnot_pd(signbit, x);
    const __m256d y = _mm256_mul_pd(x, x);
    __m256d p = _mm256_set1_pd(-2.1859258591298787e-05);
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(8.672444658882306e-05));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(-0.00023550170267042957));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(0.0005891889958817433));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(-0.0014557048421519495));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(0.0035921145924042454));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(-0.008863234607457774));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(0.021869488495914118));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(-0.053968253967215975));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(0.13333333333331948));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(-0.33333333333333326));
    p = _mm256_fmadd_pd(p, y, _mm256_set1_pd(1.0));
    const __m256d small = _mm256_mul_pd(x, p);
    const __m256d e = v_exp_f64(_mm256_add_pd(ax, ax));
    __m256d wide = _mm256_sub_pd(_mm256_set1_pd(1.0),
                                 _mm256_div_pd(_mm256_set1_pd(2.0),
                                               _mm256_add_pd(e, _mm256_set1_pd(1.0))));
    wide = _mm256_or_pd(wide, _mm256_and_pd(x, signbit));
    return _mm256_blendv_pd(wide, small,
                            _mm256_cmp_pd(ax, _mm256_set1_pd(0.5), _CMP_LT_OQ));
}

__attribute__((target("avx2,fma"), always_inline))
inline __m256d apply_f64(VOp op, VParams prm, __m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d signbit = _mm256_set1_pd(-0.0);
    switch (op) {
        case VOp::Abs: return _mm256_andnot_pd(signbit, x);
        case VOp::Neg: return _mm256_xor_pd(x, signbit);
        case VOp::Sign: {
            __m256d pos = _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_GT_OQ), one);
            __m256d neg = _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_LT_OQ), one);
            return _mm256_sub_pd(pos, neg);
        }
        case VOp::Square: return _mm256_mul_pd(x, x);
        case VOp::Reciprocal: return _mm256_div_pd(one, x);
        case VOp::Sqrt: return _mm256_sqrt_pd(x);
        case VOp::Rsqrt: return _mm256_div_pd(one, _mm256_sqrt_pd(x));
        case VOp::Floor: return _mm256_floor_pd(x);
        case VOp::Ceil: return _mm256_ceil_pd(x);
        case VOp::Trunc: return _mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        case VOp::Round: return _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        case VOp::Frac: return _mm256_sub_pd(x, _mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        case VOp::Relu: return _mm256_max_pd(zero, x);
        case VOp::Exp: return v_exp_f64(x);
        case VOp::Exp2: return tensorplay::tpsleef::exp2(x);
        case VOp::Expm1: return tensorplay::tpsleef::expm1(x);
        case VOp::Log: return v_log_f64<LogBase64::E>(x);
        case VOp::Log2: return v_log_f64<LogBase64::Two>(x);
        case VOp::Log10: return v_log_f64<LogBase64::Ten>(x);
        case VOp::Log1p: return tensorplay::tpsleef::log1p(x);
        case VOp::Sin: return tensorplay::tpsleef::sin(x);
        case VOp::Cos: return tensorplay::tpsleef::cos(x);
        case VOp::Tan: return tensorplay::tpsleef::tan(x);
        case VOp::Asin: return tensorplay::tpsleef::asin(x);
        case VOp::Acos: return tensorplay::tpsleef::acos(x);
        case VOp::Atan: return tensorplay::tpsleef::atan(x);
        case VOp::Sinh: return tensorplay::tpsleef::sinh(x);
        case VOp::Cosh: return tensorplay::tpsleef::cosh(x);
        case VOp::Tanh: return v_tanh_f64(x);
        case VOp::Asinh: return tensorplay::tpsleef::asinh(x);
        case VOp::Acosh: return tensorplay::tpsleef::acosh(x);
        case VOp::Atanh: return tensorplay::tpsleef::atanh(x);
        case VOp::Erf: return v_erf_f64(x);
        case VOp::Erfc: return v_erfc_f64(x);
        case VOp::Lgamma: return v_lgamma_pos_f64(x);
        case VOp::Sigmoid:
            return _mm256_div_pd(one, _mm256_add_pd(one, v_exp_f64(_mm256_xor_pd(x, signbit))));
        case VOp::GeluNone: {
            const __m256d kAlpha = _mm256_set1_pd(0.70710678118654752440);
            __m256d cdf = _mm256_add_pd(one, v_erf_f64(_mm256_mul_pd(kAlpha, x)));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m256d kBeta = _mm256_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m256d kKappa = _mm256_set1_pd(0.044715);
            __m256d x_cube = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
            __m256d inner = _mm256_mul_pd(kBeta, _mm256_add_pd(x, _mm256_mul_pd(kKappa, x_cube)));
            __m256d cdf = _mm256_add_pd(one, v_tanh_f64(inner));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::Silu: {
            __m256d den = _mm256_add_pd(one, v_exp_f64(_mm256_xor_pd(x, signbit)));
            return _mm256_div_pd(x, den);
        }
        case VOp::Mish: {
            __m256d sp = v_log_f64<LogBase64::E>(_mm256_add_pd(one, v_exp_f64(x)));
            return _mm256_mul_pd(x, v_tanh_f64(sp));
        }
        case VOp::Selu: {
            const __m256d lambda = _mm256_set1_pd(1.0507009873554804934193349852946);
            const __m256d alphalambda = _mm256_set1_pd(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946);
            __m256d pos = _mm256_mul_pd(x, lambda);
            __m256d neg = _mm256_mul_pd(alphalambda, tensorplay::tpsleef::expm1(x));
            return _mm256_blendv_pd(neg, pos, _mm256_cmp_pd(x, zero, _CMP_GT_OQ));
        }
        case VOp::Elu:
            return v_elu_f64(_mm256_set1_pd(prm.p0), _mm256_set1_pd(prm.p1),
                             _mm256_set1_pd(prm.p2), x);
        case VOp::Softplus:
            return v_softplus_f64(_mm256_set1_pd(prm.p0), _mm256_set1_pd(prm.p1), x);
        case VOp::Hardswish: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            __m256d y = _mm256_add_pd(x, three);
            y = _mm256_min_pd(six, _mm256_max_pd(zero, y));
            return _mm256_div_pd(_mm256_mul_pd(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            __m256d y = _mm256_add_pd(x, three);
            y = _mm256_min_pd(six, _mm256_max_pd(zero, y));
            return _mm256_div_pd(y, six);
        }
        case VOp::LeakyRelu: {
            const __m256d slope = _mm256_set1_pd(prm.p0);
            __m256d neg = _mm256_mul_pd(slope, x);
            return _mm256_blendv_pd(x, neg, _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ));
        }
        case VOp::Hardtanh: {
            const __m256d lo = _mm256_set1_pd(prm.p0);
            const __m256d hi = _mm256_set1_pd(prm.p1);
            return _mm256_min_pd(hi, _mm256_max_pd(lo, x));
        }
        case VOp::Relu6:
            return _mm256_min_pd(_mm256_set1_pd(6.0), _mm256_max_pd(zero, x));
        default: return x;
    }
}

#define TP_F64_SIMPLE_LOOP_4(EXPR)                                        \
    {                                                                     \
        for (; i + 16 <= n; i += 16) {                                    \
            __m256d x = _mm256_loadu_pd(s + i);                           \
            _mm256_storeu_pd(d + i, (EXPR));                              \
            x = _mm256_loadu_pd(s + i + 4);                               \
            _mm256_storeu_pd(d + i + 4, (EXPR));                          \
            x = _mm256_loadu_pd(s + i + 8);                               \
            _mm256_storeu_pd(d + i + 8, (EXPR));                          \
            x = _mm256_loadu_pd(s + i + 12);                              \
            _mm256_storeu_pd(d + i + 12, (EXPR));                         \
        }                                                                 \
        for (; i + 4 <= n; i += 4) {                                      \
            __m256d x = _mm256_loadu_pd(s + i);                           \
            _mm256_storeu_pd(d + i, (EXPR));                              \
        }                                                                 \
    }

__attribute__((target("avx2,fma"), noinline))
static void f64_chunk_avx2(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
    const double* s = src + b;
    double* d = dst + b;
    const int64_t n = e - b;
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d signbit = _mm256_set1_pd(-0.0);
    int64_t i = 0;
    switch (op) {
        case VOp::Abs:
            TP_F64_SIMPLE_LOOP_4(_mm256_andnot_pd(signbit, x));
            break;
        case VOp::Neg:
            TP_F64_SIMPLE_LOOP_4(_mm256_xor_pd(x, signbit));
            break;
        case VOp::Sign:
            TP_F64_SIMPLE_LOOP_4(
                _mm256_sub_pd(_mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_GT_OQ), one),
                              _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_LT_OQ), one)));
            break;
        case VOp::Square:
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(x, x));
            break;
        case VOp::Reciprocal:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(one, x));
            break;
        case VOp::Sqrt:
            TP_F64_SIMPLE_LOOP_4(_mm256_sqrt_pd(x));
            break;
        case VOp::Rsqrt:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(one, _mm256_sqrt_pd(x)));
            break;
        case VOp::Floor:
            TP_F64_SIMPLE_LOOP_4(_mm256_floor_pd(x));
            break;
        case VOp::Ceil:
            TP_F64_SIMPLE_LOOP_4(_mm256_ceil_pd(x));
            break;
        case VOp::Trunc:
            TP_F64_SIMPLE_LOOP_4(_mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            break;
        case VOp::Round:
            TP_F64_SIMPLE_LOOP_4(_mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            break;
        case VOp::Frac:
            TP_F64_SIMPLE_LOOP_4(
                _mm256_sub_pd(x, _mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
            break;
        case VOp::Relu:
            TP_F64_SIMPLE_LOOP_4(_mm256_max_pd(zero, x));
            break;
        case VOp::Hardtanh: {
            const __m256d lo = _mm256_set1_pd(prm.p0);
            const __m256d hi = _mm256_set1_pd(prm.p1);
            TP_F64_SIMPLE_LOOP_4(_mm256_min_pd(hi, _mm256_max_pd(lo, x)));
            break;
        }
        case VOp::Elu: {
            const __m256d negcoef = _mm256_set1_pd(prm.p0); // alpha*scale
            const __m256d poscoef = _mm256_set1_pd(prm.p1); // scale
            const __m256d negipt = _mm256_set1_pd(prm.p2);  // input_scale
            for (; i + 4 <= n; i += 4) {
                __m256d x = _mm256_loadu_pd(s + i);
                _mm256_storeu_pd(d + i, v_elu_f64(negcoef, poscoef, negipt, x));
            }
            break;
        }
        case VOp::Softplus: {
            const __m256d beta = _mm256_set1_pd(prm.p0);
            const __m256d threshold = _mm256_set1_pd(prm.p1);
            for (; i + 4 <= n; i += 4) {
                __m256d x = _mm256_loadu_pd(s + i);
                _mm256_storeu_pd(d + i, v_softplus_f64(beta, threshold, x));
            }
            break;
        }
        case VOp::Exp:
            TP_F64_SIMPLE_LOOP_4(v_exp_f64(x));
            break;
        case VOp::Expm1:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::expm1(x));
            break;
        case VOp::Log:
            TP_F64_SIMPLE_LOOP_4(v_log_f64<LogBase64::E>(x));
            break;
        case VOp::Log1p:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::log1p(x));
            break;
        case VOp::Log2:
            TP_F64_SIMPLE_LOOP_4(v_log_f64<LogBase64::Two>(x));
            break;
        case VOp::Log10:
            TP_F64_SIMPLE_LOOP_4(v_log_f64<LogBase64::Ten>(x));
            break;
        case VOp::Tanh:
            TP_F64_SIMPLE_LOOP_4(v_tanh_f64(x));
            break;
        case VOp::Sigmoid:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                one, _mm256_add_pd(one, v_exp_f64(_mm256_xor_pd(x, signbit)))));
            break;
        case VOp::Silu:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                x, _mm256_add_pd(one, v_exp_f64(_mm256_xor_pd(x, signbit)))));
            break;
        case VOp::Mish:
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                x, v_tanh_f64(v_log_f64<LogBase64::E>(
                       _mm256_add_pd(one, v_exp_f64(x))))));
            break;
        case VOp::GeluNone: {
            const __m256d kAlpha = _mm256_set1_pd(0.70710678118654752440);
            const __m256d half = _mm256_set1_pd(0.5);
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                _mm256_mul_pd(half, x),
                _mm256_add_pd(one, v_erf_f64(_mm256_mul_pd(kAlpha, x)))));
            break;
        }
        case VOp::GeluTanh: {
            const __m256d kBeta = _mm256_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m256d kKappa = _mm256_set1_pd(0.044715);
            const __m256d half = _mm256_set1_pd(0.5);
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                _mm256_mul_pd(half, x),
                _mm256_add_pd(one, v_tanh_f64(_mm256_mul_pd(
                    kBeta, _mm256_add_pd(x, _mm256_mul_pd(kKappa, _mm256_mul_pd(_mm256_mul_pd(x, x), x))))))));
            break;
        }
        case VOp::Sin:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::sin(x));
            break;
        case VOp::Cos:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::cos(x));
            break;
        case VOp::Tan:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::tan(x));
            break;
        case VOp::Asin:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::asin(x));
            break;
        case VOp::Acos:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::acos(x));
            break;
        case VOp::Atan:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::atan(x));
            break;
        case VOp::Erf:
            TP_F64_SIMPLE_LOOP_4(v_erf_f64(x));
            break;
        case VOp::Erfc:
            TP_F64_SIMPLE_LOOP_4(v_erfc_f64(x));
            break;
        case VOp::Relu6:
            TP_F64_SIMPLE_LOOP_4(_mm256_min_pd(_mm256_set1_pd(6.0), _mm256_max_pd(zero, x)));
            break;
        case VOp::LeakyRelu: {
            const __m256d slope = _mm256_set1_pd(prm.p0);
            TP_F64_SIMPLE_LOOP_4(_mm256_blendv_pd(x, _mm256_mul_pd(slope, x),
                                                  _mm256_cmp_pd(x, zero, _CMP_LT_OQ)));
            break;
        }
        case VOp::Hardswish: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                _mm256_mul_pd(x, _mm256_min_pd(six, _mm256_max_pd(zero, _mm256_add_pd(x, three)))), six));
            break;
        }
        case VOp::Hardsigmoid: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                _mm256_min_pd(six, _mm256_max_pd(zero, _mm256_add_pd(x, three))), six));
            break;
        }
        default:
            for (; i + 4 <= n; i += 4) {
                __m256d x = _mm256_loadu_pd(s + i);
                if (op == VOp::Lgamma && _mm256_movemask_pd(_mm256_cmp_pd(x, zero, _CMP_LE_OQ))) {
                    for (int64_t j = i; j < i + 4; ++j) d[j] = scalar_apply(op, prm, s[j]);
                } else {
                    _mm256_storeu_pd(d + i, apply_f64(op, prm, x));
                }
            }
            break;
    }
#undef TP_F64_SIMPLE_LOOP_4
    for (; i < n; ++i) d[i] = scalar_apply(op, prm, s[i]);
}

// Reduced-precision dtypes widen to f32, run the same kernels, narrow back.
// One chunk-wide branch for the parameterised ops, as for float32: a double
// load, a narrowing convert and a broadcast per vector sit on the same ports
// as the convert pair, and there they cost more than the arithmetic they feed.
//
// Four vectors are in flight per iteration.  The widening convert, the
// narrowing convert and most transcendental steps share one execution port,
// and each element walks the whole widen-compute-narrow chain before the next
// one starts, so a single stream leaves that port idle between steps.  Four
// independent streams keep it busy, which is worth more here than the extra
// registers.
#define TP_HALF_STEP(EXPR, OFF)                                            \
    {                                                                      \
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(                        \
            reinterpret_cast<const __m128i*>(src + (OFF))));               \
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + (OFF)),          \
                         _mm256_cvtps_ph((EXPR),                           \
                                         _MM_FROUND_TO_NEAREST_INT));      \
    }

#define TP_HALF_CHUNK_8(EXPR)                                              \
    {                                                                      \
        for (; i + 32 <= n; i += 32) {                                     \
            TP_HALF_STEP(EXPR, i)                                          \
            TP_HALF_STEP(EXPR, i + 8)                                      \
            TP_HALF_STEP(EXPR, i + 16)                                     \
            TP_HALF_STEP(EXPR, i + 24)                                     \
        }                                                                  \
        for (; i + 8 <= n; i += 8) {                                       \
            TP_HALF_STEP(EXPR, i)                                          \
        }                                                                  \
    }

__attribute__((target("avx2,fma,f16c"), noinline))
static void half_chunk_avx2(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    switch (op) {
        case VOp::Hardtanh: {
            const __m256 lo = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 hi = _mm256_set1_ps(static_cast<float>(prm.p1));
            TP_HALF_CHUNK_8(_mm256_min_ps(hi, _mm256_max_ps(lo, x)))
            break;
        }
        case VOp::LeakyRelu: {
            const __m256 zero = _mm256_setzero_ps();
            const __m256 slope = _mm256_set1_ps(static_cast<float>(prm.p0));
            TP_HALF_CHUNK_8(_mm256_blendv_ps(
                x, _mm256_mul_ps(slope, x), _mm256_cmp_ps(x, zero, _CMP_LT_OQ)))
            break;
        }
        case VOp::Elu: {
            const __m256 negcoef = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 poscoef = _mm256_set1_ps(static_cast<float>(prm.p1));
            const __m256 negipt = _mm256_set1_ps(static_cast<float>(prm.p2));
            TP_HALF_CHUNK_8(v_elu_f32(negcoef, poscoef, negipt, x))
            break;
        }
        case VOp::Softplus: {
            const __m256 beta = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 threshold = _mm256_set1_ps(static_cast<float>(prm.p1));
            TP_HALF_CHUNK_8(v_softplus_f32(beta, threshold, x))
            break;
        }
        default:
            TP_HALF_CHUNK_8(apply_f32(op, prm, x))
            break;
    }
#undef TP_HALF_STEP
#undef TP_HALF_CHUNK_8
    for (; i < n; ++i) {
        float xf = _cvtsh_ss(src[i]);
        dst[i] = _cvtss_sh(scalar_apply(op, prm, xf), _MM_FROUND_TO_NEAREST_INT);
    }
}

// ---------------------------------------------------------------------------
// AVX-512 layer (Zen4 native width).  Same formulas as the AVX2 kernels
// above -- only the register width changes -- so results stay lane-identical
// ---------------------------------------------------------------------------
constexpr float kSignBitF512 = -0.0f;  // (shared constant, kept for clarity)

__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 v_lgamma_pos_f32_512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __mmask16 small = _mm512_cmp_ps_mask(x, one, _CMP_LT_OQ);
    const __m512 xp = _mm512_add_ps(x, _mm512_maskz_mov_ps(small, one));

    __m512 z = _mm512_sub_ps(xp, one);
    __m512 series = _mm512_set1_ps(static_cast<float>(kLanczos[0]));
#pragma GCC unroll 8
    for (int i = 1; i < 9; ++i) {
        __m512 denom = _mm512_add_ps(z, _mm512_set1_ps(static_cast<float>(i)));
        series = _mm512_fmadd_ps(_mm512_set1_ps(static_cast<float>(kLanczos[i])),
                                 _mm512_div_ps(one, denom), series);
    }
    __m512 t = _mm512_add_ps(z, _mm512_set1_ps(7.5f));
    __m512 res = _mm512_sub_ps(
        _mm512_fmadd_ps(_mm512_add_ps(z, _mm512_set1_ps(0.5f)),
                        tensorplay::tpsleef::log(t), _mm512_set1_ps(static_cast<float>(kHalfLogTwoPi))),
        t);
    res = _mm512_add_ps(res, tensorplay::tpsleef::log(series));
    return _mm512_sub_ps(res, _mm512_maskz_mov_ps(small, tensorplay::tpsleef::log(x)));
}

__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512d v_lgamma_pos_f64_512(__m512d x) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __mmask8 small = _mm512_cmp_pd_mask(x, one, _CMP_LT_OQ);
    const __m512d xp = _mm512_add_pd(x, _mm512_maskz_mov_pd(small, one));

    __m512d z = _mm512_sub_pd(xp, one);
    __m512d series = _mm512_set1_pd(kLanczos[0]);
#pragma GCC unroll 8
    for (int i = 1; i < 9; ++i) {
        __m512d denom = _mm512_add_pd(z, _mm512_set1_pd(static_cast<double>(i)));
        series = _mm512_fmadd_pd(_mm512_set1_pd(kLanczos[i]),
                                 _mm512_div_pd(one, denom), series);
    }
    __m512d t = _mm512_add_pd(z, _mm512_set1_pd(7.5));
    __m512d res = _mm512_sub_pd(
        _mm512_fmadd_pd(_mm512_add_pd(z, _mm512_set1_pd(0.5)),
                        tensorplay::tpsleef::log(t), _mm512_set1_pd(kHalfLogTwoPi)),
        t);
    res = _mm512_add_pd(res, tensorplay::tpsleef::log(series));
    return _mm512_sub_pd(res, _mm512_maskz_mov_pd(small, tensorplay::tpsleef::log(x)));
}

// rcp14 seeds at <=2^-14; one NR step reaches ~2^-28, i.e. float-exact for
// the sigmoid/silu denominators seen here.
__attribute__((target("avx512f"), always_inline))
inline __m512 v_rcp_nr_ps(__m512 d) {
    __m512 r = _mm512_rcp14_ps(d);
    return _mm512_mul_ps(r, _mm512_fnmadd_ps(d, r, _mm512_set1_ps(2.0f)));
}

// erf on float lanes.  Coefficients live in cpu/vec/ErfPoly.h so the vector
// class and this kernel evaluate the same function.  The AVX-512 tier has no
// inlined exponential, so the decay factor comes from the library entry point.
__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 v_erf_f32_512(__m512 x) {
    const __m512 sign_bit = _mm512_set1_ps(-0.0f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 abs_x = _mm512_andnot_ps(sign_bit, x);
    const __m512 t = _mm512_div_ps(
        one, _mm512_fmadd_ps(_mm512_set1_ps(tensorplay::vecmath::kErfTailScale), abs_x, one));
    __m512 p = _mm512_set1_ps(tensorplay::vecmath::kErfTail[4]);
    for (int i = 3; i >= 0; --i)
        p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(tensorplay::vecmath::kErfTail[i]));
    p = _mm512_mul_ps(p, t);
    const __m512 decay = tensorplay::tpsleef::exp(_mm512_xor_ps(sign_bit, _mm512_mul_ps(x, x)));
    const __m512 r = _mm512_sub_ps(one, _mm512_mul_ps(p, decay));
    return _mm512_or_ps(_mm512_and_ps(sign_bit, x), r);
}
__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 gelu_poly_f32_512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 alpha = _mm512_set1_ps(0.7071067811865476f);
    const __m512 z = _mm512_mul_ps(alpha, x);
    const __m512 q = _mm512_mul_ps(z, z);
    __m512 p = _mm512_set1_ps(0.0000473397310f);
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(-0.000664962729f));
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(0.00495391422f));
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(-0.0266572969f));
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(0.112756411f));
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(-0.376113147f));
    p = _mm512_fmadd_ps(p, q, _mm512_set1_ps(1.12837863f));
    const __m512 cdf = _mm512_add_ps(one, _mm512_mul_ps(z, p));
    return _mm512_mul_ps(_mm512_mul_ps(half, x), cdf);
}

// 512-bit twin of v_elu_f32; same reason for the coefficient arguments.
__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 v_elu_f32_512(__m512 negcoef, __m512 poscoef, __m512 negipt, __m512 x) {
    const __m512 zero = _mm512_setzero_ps();
    __m512 scaled = _mm512_mul_ps(x, negipt);
    __m512 neg = _mm512_mul_ps(tensorplay::tpsleef::expm1(scaled), negcoef);
    __m512 pos = _mm512_mul_ps(x, poscoef);
    return _mm512_mask_mov_ps(neg, _mm512_cmp_ps_mask(x, zero, _CMP_GE_OQ), pos);
}

// 512-bit twin of v_softplus_f32; same reason for the coefficient arguments.
__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 v_softplus_f32_512(__m512 beta, __m512 threshold, __m512 x) {
    __m512 bx = _mm512_mul_ps(x, beta);
    __m512 sp = _mm512_div_ps(tensorplay::tpsleef::log1p(tensorplay::tpsleef::exp(bx)), beta);
    return _mm512_mask_mov_ps(sp, _mm512_cmp_ps_mask(bx, threshold, _CMP_GT_OQ), x);
}

__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512 apply16_f32(VOp op, VParams prm, __m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 signbit = _mm512_set1_ps(kSignBitF512);
    switch (op) {
        case VOp::Abs: return _mm512_andnot_ps(signbit, x);
        case VOp::Neg: return _mm512_xor_ps(x, signbit);
        case VOp::Sign: {
            const __m512 minus_one = _mm512_set1_ps(-1.0f);
            __m512 r = _mm512_mask_mov_ps(zero,
                                          _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ),
                                          minus_one);
            return _mm512_mask_mov_ps(r,
                                      _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ),
                                      one);
        }
        case VOp::Square: return _mm512_mul_ps(x, x);
        case VOp::Reciprocal: return _mm512_div_ps(one, x);
        case VOp::Sqrt: return _mm512_sqrt_ps(x);
        case VOp::Rsqrt: return _mm512_div_ps(one, _mm512_sqrt_ps(x));
        case VOp::Floor: return _mm512_floor_ps(x);
        case VOp::Ceil: return _mm512_ceil_ps(x);
        case VOp::Trunc: return _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        case VOp::Round: return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        case VOp::Frac: return _mm512_sub_ps(x, _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        case VOp::Relu: return _mm512_max_ps(zero, x);
        case VOp::Exp: return tensorplay::tpsleef::exp(x);
        case VOp::Exp2: return tensorplay::tpsleef::exp2(x);
        case VOp::Expm1: return tensorplay::tpsleef::expm1(x);
        case VOp::Log: return tensorplay::tpsleef::log(x);
        case VOp::Log2: return tensorplay::tpsleef::log2(x);
        case VOp::Log10: return tensorplay::tpsleef::log10(x);
        case VOp::Log1p: return tensorplay::tpsleef::log1p(x);
        case VOp::Sin: return tensorplay::tpsleef::sin(x);
        case VOp::Cos: return tensorplay::tpsleef::cos(x);
        case VOp::Tan: return tensorplay::tpsleef::tan(x);
        case VOp::Asin: return tensorplay::tpsleef::asin(x);
        case VOp::Acos: return tensorplay::tpsleef::acos(x);
        case VOp::Atan: return tensorplay::tpsleef::atan(x);
        case VOp::Sinh: return tensorplay::tpsleef::sinh(x);
        case VOp::Cosh: return tensorplay::tpsleef::cosh(x);
        case VOp::Tanh: return tensorplay::tpsleef::tanh(x);
        case VOp::Asinh: return tensorplay::tpsleef::asinh(x);
        case VOp::Acosh: return tensorplay::tpsleef::acosh(x);
        case VOp::Atanh: return tensorplay::tpsleef::atanh(x);
        case VOp::Erf: return v_erf_f32_512(x);
        case VOp::Erfc: return tensorplay::tpsleef::erfc(x);
        case VOp::Lgamma:
            return v_lgamma_pos_f32_512(x);
        case VOp::Sigmoid: {
            // rcp14+NR instead of divps; see the 256-bit Sigmoid note for the
            // inf/NaN corner repairs.
            __m512 den = _mm512_add_ps(one, tensorplay::tpsleef::exp(_mm512_xor_ps(x, signbit)));
            __m512 r = v_rcp_nr_ps(den);
            __m512 zeroed = _mm512_maskz_mov_ps(
                _mm512_cmp_ps_mask(den, _mm512_set1_ps(INFINITY), _CMP_NEQ_OQ), r);
            return _mm512_mask_mov_ps(zeroed, _mm512_cmp_ps_mask(den, den, _CMP_UNORD_Q), r);
        }
        case VOp::GeluNone: {
            const __m512 kAlpha = _mm512_set1_ps(0.7071067811865476f);
            const __m512 cdf =
                _mm512_add_ps(one, v_erf_f32_512(_mm512_mul_ps(kAlpha, x)));
            return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m512 kBeta = _mm512_set1_ps(static_cast<float>(1.41421356237309504880 * 1.12837916709551257390 * 0.5));
            const __m512 kKappa = _mm512_set1_ps(0.044715f);
            __m512 x_cube = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
            __m512 inner = _mm512_mul_ps(kBeta, _mm512_add_ps(x, _mm512_mul_ps(kKappa, x_cube)));
            __m512 cdf = _mm512_add_ps(one, tensorplay::tpsleef::tanh(inner));
            return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x), cdf);
        }
        case VOp::Silu: {
            // x * rcp14_nr(den): corner-equivalent to x/den (see 256-bit note).
            __m512 den = _mm512_add_ps(one, tensorplay::tpsleef::exp(_mm512_xor_ps(x, signbit)));
            return _mm512_mul_ps(x, v_rcp_nr_ps(den));
        }
        case VOp::Mish: {
            __m512 sp = tensorplay::tpsleef::log(_mm512_add_ps(one, tensorplay::tpsleef::exp(x)));
            return _mm512_mul_ps(x, tensorplay::tpsleef::tanh(sp));
        }
        case VOp::Selu: {
            const __m512 lambda = _mm512_set1_ps(static_cast<float>(1.0507009873554804934193349852946));
            const __m512 alphalambda = _mm512_set1_ps(static_cast<float>(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946));
            __m512 pos = _mm512_mul_ps(x, lambda);
            __m512 neg = _mm512_mul_ps(alphalambda, tensorplay::tpsleef::expm1(x));
            return _mm512_mask_mov_ps(neg, _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ), pos);
        }
        case VOp::Elu:
            return v_elu_f32_512(_mm512_set1_ps(static_cast<float>(prm.p0)),
                                 _mm512_set1_ps(static_cast<float>(prm.p1)),
                                 _mm512_set1_ps(static_cast<float>(prm.p2)), x);
        case VOp::Softplus:
            return v_softplus_f32_512(_mm512_set1_ps(static_cast<float>(prm.p0)),
                                      _mm512_set1_ps(static_cast<float>(prm.p1)), x);
        case VOp::Hardswish: {
            const __m512 three = _mm512_set1_ps(3.0f);
            const __m512 six = _mm512_set1_ps(6.0f);
            __m512 y = _mm512_add_ps(x, three);
            y = _mm512_min_ps(six, _mm512_max_ps(zero, y));
            return _mm512_div_ps(_mm512_mul_ps(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m512 three = _mm512_set1_ps(3.0f);
            const __m512 six = _mm512_set1_ps(6.0f);
            __m512 y = _mm512_add_ps(x, three);
            y = _mm512_min_ps(six, _mm512_max_ps(zero, y));
            return _mm512_div_ps(y, six);
        }
        case VOp::LeakyRelu: {
            const __m512 slope = _mm512_set1_ps(static_cast<float>(prm.p0));
            __m512 neg = _mm512_mul_ps(slope, x);
            return _mm512_mask_mov_ps(x, _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ), neg);
        }
        case VOp::Hardtanh: {
            const __m512 lo = _mm512_set1_ps(static_cast<float>(prm.p0));
            const __m512 hi = _mm512_set1_ps(static_cast<float>(prm.p1));
            return _mm512_min_ps(hi, _mm512_max_ps(lo, x));
        }
        case VOp::Relu6:
            return _mm512_min_ps(_mm512_set1_ps(6.0f), _mm512_max_ps(zero, x));
        default: return x;
    }
}

#define TP_F32_SIMPLE_LOOP_16(EXPR)                                       \
    {                                                                     \
        for (; i + 64 <= n; i += 64) {                                    \
            __m512 x = _mm512_loadu_ps(s + i);                            \
            _mm512_storeu_ps(d + i, (EXPR));                              \
            x = _mm512_loadu_ps(s + i + 16);                              \
            _mm512_storeu_ps(d + i + 16, (EXPR));                         \
            x = _mm512_loadu_ps(s + i + 32);                              \
            _mm512_storeu_ps(d + i + 32, (EXPR));                         \
            x = _mm512_loadu_ps(s + i + 48);                              \
            _mm512_storeu_ps(d + i + 48, (EXPR));                         \
        }                                                                 \
        for (; i + 16 <= n; i += 16) {                                    \
            __m512 x = _mm512_loadu_ps(s + i);                            \
            _mm512_storeu_ps(d + i, (EXPR));                              \
        }                                                                 \
    }

__attribute__((target("avx512f,avx512dq"), noinline))
static void f32_chunk_avx512(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
    const float* s = src + b;
    float* d = dst + b;
    const int64_t n = e - b;
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 signbit = _mm512_set1_ps(kSignBitF512);
    const __m512 minus_one = _mm512_set1_ps(-1.0f);
    int64_t i = 0;
    switch (op) {
        case VOp::Abs:
            TP_F32_SIMPLE_LOOP_16(_mm512_andnot_ps(signbit, x));
            break;
        case VOp::Neg:
            TP_F32_SIMPLE_LOOP_16(_mm512_xor_ps(x, signbit));
            break;
        case VOp::Sign:
            TP_F32_SIMPLE_LOOP_16(
                _mm512_mask_mov_ps(
                    _mm512_mask_mov_ps(zero, _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ), minus_one),
                    _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ), one));
            break;
        case VOp::Square:
            TP_F32_SIMPLE_LOOP_16(_mm512_mul_ps(x, x));
            break;
        case VOp::Reciprocal:
            TP_F32_SIMPLE_LOOP_16(_mm512_div_ps(one, x));
            break;
        case VOp::Sqrt:
            TP_F32_SIMPLE_LOOP_16(_mm512_sqrt_ps(x));
            break;
        case VOp::Rsqrt:
            TP_F32_SIMPLE_LOOP_16(_mm512_div_ps(one, _mm512_sqrt_ps(x)));
            break;
        case VOp::Floor:
            TP_F32_SIMPLE_LOOP_16(_mm512_floor_ps(x));
            break;
        case VOp::Ceil:
            TP_F32_SIMPLE_LOOP_16(_mm512_ceil_ps(x));
            break;
        case VOp::Trunc:
            TP_F32_SIMPLE_LOOP_16(_mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            break;
        case VOp::Round:
            TP_F32_SIMPLE_LOOP_16(_mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            break;
        case VOp::Frac:
            TP_F32_SIMPLE_LOOP_16(
                _mm512_sub_ps(x, _mm512_roundscale_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
            break;
        case VOp::Relu:
            TP_F32_SIMPLE_LOOP_16(_mm512_max_ps(zero, x));
            break;
        case VOp::Hardtanh: {
            const __m512 lo = _mm512_set1_ps(static_cast<float>(prm.p0));
            const __m512 hi = _mm512_set1_ps(static_cast<float>(prm.p1));
            TP_F32_SIMPLE_LOOP_16(_mm512_min_ps(hi, _mm512_max_ps(lo, x)));
            break;
        }
        case VOp::Elu: {
            const __m512 negcoef = _mm512_set1_ps(static_cast<float>(prm.p0)); // alpha*scale
            const __m512 poscoef = _mm512_set1_ps(static_cast<float>(prm.p1)); // scale
            const __m512 negipt = _mm512_set1_ps(static_cast<float>(prm.p2));  // input_scale
            for (; i + 16 <= n; i += 16) {
                __m512 x = _mm512_loadu_ps(s + i);
                _mm512_storeu_ps(d + i, v_elu_f32_512(negcoef, poscoef, negipt, x));
            }
            break;
        }
        case VOp::Softplus: {
            const __m512 beta = _mm512_set1_ps(static_cast<float>(prm.p0));
            const __m512 threshold = _mm512_set1_ps(static_cast<float>(prm.p1));
            for (; i + 16 <= n; i += 16) {
                __m512 x = _mm512_loadu_ps(s + i);
                _mm512_storeu_ps(d + i, v_softplus_f32_512(beta, threshold, x));
            }
            break;
        }
        default:
            for (; i + 16 <= n; i += 16) {
                __m512 x = _mm512_loadu_ps(s + i);
                if (op == VOp::Lgamma && _mm512_cmplt_ps_mask(x, zero)) {
                    for (int64_t j = i; j < i + 16; ++j) d[j] = scalar_apply(op, prm, s[j]);
                } else if (op == VOp::GeluNone) {
                    const __m512 z = _mm512_mul_ps(
                        _mm512_set1_ps(0.7071067811865476f), x);
                    const __m512 az = _mm512_andnot_ps(
                        _mm512_set1_ps(kSignBitF512), z);
                    const __mmask16 core = _mm512_cmp_ps_mask(
                        az, _mm512_set1_ps(1.5f), _CMP_LE_OQ);
                    if (core == static_cast<__mmask16>(0xffff)) {
                        _mm512_storeu_ps(d + i, gelu_poly_f32_512(x));
                    } else {
                        _mm512_storeu_ps(d + i, apply16_f32(op, prm, x));
                    }
                } else {
                    _mm512_storeu_ps(d + i, apply16_f32(op, prm, x));
                }
            }
            break;
    }
#undef TP_F32_SIMPLE_LOOP_16
    for (; i < n; ++i) d[i] = scalar_apply(op, prm, s[i]);
}

__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m512d apply16_f64(VOp op, VParams prm, __m512d x) {
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d signbit = _mm512_set1_pd(-0.0);
    switch (op) {
        case VOp::Abs: return _mm512_andnot_pd(signbit, x);
        case VOp::Neg: return _mm512_xor_pd(x, signbit);
        case VOp::Sign: {
            const __m512d minus_one = _mm512_set1_pd(-1.0);
            __m512d r = _mm512_mask_mov_pd(zero,
                                           _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ),
                                           minus_one);
            return _mm512_mask_mov_pd(r,
                                      _mm512_cmp_pd_mask(x, zero, _CMP_GT_OQ),
                                      one);
        }
        case VOp::Square: return _mm512_mul_pd(x, x);
        case VOp::Reciprocal: return _mm512_div_pd(one, x);
        case VOp::Sqrt: return _mm512_sqrt_pd(x);
        case VOp::Rsqrt: return _mm512_div_pd(one, _mm512_sqrt_pd(x));
        case VOp::Floor: return _mm512_floor_pd(x);
        case VOp::Ceil: return _mm512_ceil_pd(x);
        case VOp::Trunc: return _mm512_roundscale_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        case VOp::Round: return _mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        case VOp::Frac: return _mm512_sub_pd(x, _mm512_roundscale_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
        case VOp::Relu: return _mm512_max_pd(zero, x);
        case VOp::Exp: return tensorplay::tpsleef::exp(x);
        case VOp::Exp2: return tensorplay::tpsleef::exp2(x);
        case VOp::Expm1: return tensorplay::tpsleef::expm1(x);
        case VOp::Log: return tensorplay::tpsleef::log(x);
        case VOp::Log2: return tensorplay::tpsleef::log2(x);
        case VOp::Log10: return tensorplay::tpsleef::log10(x);
        case VOp::Log1p: return tensorplay::tpsleef::log1p(x);
        case VOp::Sin: return tensorplay::tpsleef::sin(x);
        case VOp::Cos: return tensorplay::tpsleef::cos(x);
        case VOp::Tan: return tensorplay::tpsleef::tan(x);
        case VOp::Asin: return tensorplay::tpsleef::asin(x);
        case VOp::Acos: return tensorplay::tpsleef::acos(x);
        case VOp::Atan: return tensorplay::tpsleef::atan(x);
        case VOp::Sinh: return tensorplay::tpsleef::sinh(x);
        case VOp::Cosh: return tensorplay::tpsleef::cosh(x);
        case VOp::Tanh: return tensorplay::tpsleef::tanh(x);
        case VOp::Asinh: return tensorplay::tpsleef::asinh(x);
        case VOp::Acosh: return tensorplay::tpsleef::acosh(x);
        case VOp::Atanh: return tensorplay::tpsleef::atanh(x);
        case VOp::Erf: return tensorplay::tpsleef::erf(x);
        case VOp::Erfc: return tensorplay::tpsleef::erfc(x);
        case VOp::Lgamma: return v_lgamma_pos_f64_512(x);
        case VOp::Sigmoid:
            return _mm512_div_pd(one, _mm512_add_pd(one, tensorplay::tpsleef::exp(_mm512_xor_pd(x, signbit))));
        case VOp::GeluNone: {
            const __m512d kAlpha = _mm512_set1_pd(0.70710678118654752440);
            __m512d cdf = _mm512_add_pd(one, tensorplay::tpsleef::erf(_mm512_mul_pd(kAlpha, x)));
            return _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m512d kBeta = _mm512_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m512d kKappa = _mm512_set1_pd(0.044715);
            __m512d x_cube = _mm512_mul_pd(_mm512_mul_pd(x, x), x);
            __m512d inner = _mm512_mul_pd(kBeta, _mm512_add_pd(x, _mm512_mul_pd(kKappa, x_cube)));
            __m512d cdf = _mm512_add_pd(one, tensorplay::tpsleef::tanh(inner));
            return _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), x), cdf);
        }
        case VOp::Silu: {
            __m512d den = _mm512_add_pd(one, tensorplay::tpsleef::exp(_mm512_xor_pd(x, signbit)));
            return _mm512_div_pd(x, den);
        }
        case VOp::Mish: {
            __m512d sp = tensorplay::tpsleef::log(_mm512_add_pd(one, tensorplay::tpsleef::exp(x)));
            return _mm512_mul_pd(x, tensorplay::tpsleef::tanh(sp));
        }
        case VOp::Selu: {
            const __m512d lambda = _mm512_set1_pd(1.0507009873554804934193349852946);
            const __m512d alphalambda = _mm512_set1_pd(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946);
            __m512d pos = _mm512_mul_pd(x, lambda);
            __m512d neg = _mm512_mul_pd(alphalambda, tensorplay::tpsleef::expm1(x));
            return _mm512_mask_mov_pd(neg, _mm512_cmp_pd_mask(x, zero, _CMP_GT_OQ), pos);
        }
        case VOp::Hardswish: {
            const __m512d three = _mm512_set1_pd(3.0);
            const __m512d six = _mm512_set1_pd(6.0);
            __m512d y = _mm512_add_pd(x, three);
            y = _mm512_min_pd(six, _mm512_max_pd(zero, y));
            return _mm512_div_pd(_mm512_mul_pd(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m512d three = _mm512_set1_pd(3.0);
            const __m512d six = _mm512_set1_pd(6.0);
            __m512d y = _mm512_add_pd(x, three);
            y = _mm512_min_pd(six, _mm512_max_pd(zero, y));
            return _mm512_div_pd(y, six);
        }
        case VOp::Hardtanh: {
            const __m512d lo = _mm512_set1_pd(prm.p0);
            const __m512d hi = _mm512_set1_pd(prm.p1);
            return _mm512_min_pd(hi, _mm512_max_pd(lo, x));
        }
        case VOp::Relu6:
            return _mm512_min_pd(_mm512_set1_pd(6.0), _mm512_max_pd(zero, x));
        default: return x;
    }
}

#define TP_F64_SIMPLE_LOOP_8(EXPR)                                        \
    {                                                                     \
        for (; i + 32 <= n; i += 32) {                                    \
            __m512d x = _mm512_loadu_pd(s + i);                           \
            _mm512_storeu_pd(d + i, (EXPR));                              \
            x = _mm512_loadu_pd(s + i + 8);                               \
            _mm512_storeu_pd(d + i + 8, (EXPR));                          \
            x = _mm512_loadu_pd(s + i + 16);                              \
            _mm512_storeu_pd(d + i + 16, (EXPR));                         \
            x = _mm512_loadu_pd(s + i + 24);                              \
            _mm512_storeu_pd(d + i + 24, (EXPR));                         \
        }                                                                 \
        for (; i + 8 <= n; i += 8) {                                      \
            __m512d x = _mm512_loadu_pd(s + i);                           \
            _mm512_storeu_pd(d + i, (EXPR));                              \
        }                                                                 \
    }

__attribute__((target("avx512f,avx512dq"), noinline))
static void f64_chunk_avx512(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
    const double* s = src + b;
    double* d = dst + b;
    const int64_t n = e - b;
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d signbit = _mm512_set1_pd(-0.0);
    const __m512d minus_one = _mm512_set1_pd(-1.0);
    int64_t i = 0;
    switch (op) {
        case VOp::Abs:
            TP_F64_SIMPLE_LOOP_8(_mm512_andnot_pd(signbit, x));
            break;
        case VOp::Neg:
            TP_F64_SIMPLE_LOOP_8(_mm512_xor_pd(x, signbit));
            break;
        case VOp::Sign:
            TP_F64_SIMPLE_LOOP_8(
                _mm512_mask_mov_pd(
                    _mm512_mask_mov_pd(zero, _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ), minus_one),
                    _mm512_cmp_pd_mask(x, zero, _CMP_GT_OQ), one));
            break;
        case VOp::Square:
            TP_F64_SIMPLE_LOOP_8(_mm512_mul_pd(x, x));
            break;
        case VOp::Reciprocal:
            TP_F64_SIMPLE_LOOP_8(_mm512_div_pd(one, x));
            break;
        case VOp::Sqrt:
            TP_F64_SIMPLE_LOOP_8(_mm512_sqrt_pd(x));
            break;
        case VOp::Rsqrt:
            TP_F64_SIMPLE_LOOP_8(_mm512_div_pd(one, _mm512_sqrt_pd(x)));
            break;
        case VOp::Floor:
            TP_F64_SIMPLE_LOOP_8(_mm512_floor_pd(x));
            break;
        case VOp::Ceil:
            TP_F64_SIMPLE_LOOP_8(_mm512_ceil_pd(x));
            break;
        case VOp::Trunc:
            TP_F64_SIMPLE_LOOP_8(_mm512_roundscale_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            break;
        case VOp::Round:
            TP_F64_SIMPLE_LOOP_8(_mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            break;
        case VOp::Frac:
            TP_F64_SIMPLE_LOOP_8(
                _mm512_sub_pd(x, _mm512_roundscale_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
            break;
        case VOp::Relu:
            TP_F64_SIMPLE_LOOP_8(_mm512_max_pd(zero, x));
            break;
        case VOp::Hardtanh: {
            const __m512d lo = _mm512_set1_pd(prm.p0);
            const __m512d hi = _mm512_set1_pd(prm.p1);
            TP_F64_SIMPLE_LOOP_8(_mm512_min_pd(hi, _mm512_max_pd(lo, x)));
            break;
        }
        default:
            for (; i + 8 <= n; i += 8) {
                __m512d x = _mm512_loadu_pd(s + i);
                if (op == VOp::Lgamma && _mm512_cmplt_pd_mask(x, zero)) {
                    for (int64_t j = i; j < i + 8; ++j) d[j] = scalar_apply(op, prm, s[j]);
                } else {
                    _mm512_storeu_pd(d + i, apply16_f64(op, prm, x));
                }
            }
            break;
    }
#undef TP_F64_SIMPLE_LOOP_8
    for (; i < n; ++i) d[i] = scalar_apply(op, prm, s[i]);
}

// Same one-branch-per-chunk split as the Half loop: only the ops that read
// runtime parameters need it, the rest keep the generic body.
// Four vectors in flight, for the reason given at the half-precision walk:
// the widen, the arithmetic and the narrow form one long chain per vector, so
// a single stream idles the ports between steps.
#define TP_BF16_STEP(EXPR, OFF)                                           \
    {                                                                     \
        __m256 x = _mm256_castsi256_ps(_mm256_slli_epi32(                 \
            _mm256_cvtepu16_epi32(_mm_loadu_si128(                        \
                reinterpret_cast<const __m128i*>(src + (OFF)))),          \
            16));                                                         \
        __m256 y = (EXPR);                                                \
        __m256i u = _mm256_castps_si256(y);                               \
        __m256i rounding = _mm256_add_epi32(                              \
            _mm256_set1_epi32(0x7FFFu),                                   \
            _mm256_srli_epi32(                                            \
                _mm256_and_si256(u, _mm256_set1_epi32(0x10000u)), 16));   \
        __m256i r = _mm256_srli_epi32(_mm256_add_epi32(u, rounding), 16); \
        __m256i p = _mm256_shuffle_epi8(r, pack_pick);                    \
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + (OFF)),         \
                         _mm_unpacklo_epi64(_mm256_castsi256_si128(p),    \
                                            _mm256_extracti128_si256(p, 1))); \
    }

#define TP_BF16_CHUNK_8(EXPR)                                             \
    {                                                                     \
        for (; i + 32 <= n; i += 32) {                                    \
            TP_BF16_STEP(EXPR, i)                                         \
            TP_BF16_STEP(EXPR, i + 8)                                     \
            TP_BF16_STEP(EXPR, i + 16)                                    \
            TP_BF16_STEP(EXPR, i + 24)                                    \
        }                                                                 \
        for (; i + 8 <= n; i += 8) {                                      \
            TP_BF16_STEP(EXPR, i)                                         \
        }                                                                 \
    }

__attribute__((target("avx2,fma"), noinline))
static void bf16_chunk_avx2(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    const __m256i shift = _mm256_set1_epi32(16);
    (void)shift;
    // Narrowing picks the low half of every 32-bit result.  Byte indices are
    // lane-local, so each 128-bit lane contributes four bf16 values and the
    // -1 selectors zero the rest of the lane; the two halves are then
    // concatenated.  Keeping this in registers matters: a stack round trip
    // per vector costs more than the widen/compute pair it follows.
    const __m256i pack_pick = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    switch (op) {
        case VOp::Hardtanh: {
            const __m256 lo = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 hi = _mm256_set1_ps(static_cast<float>(prm.p1));
            TP_BF16_CHUNK_8(_mm256_min_ps(hi, _mm256_max_ps(lo, x)))
            break;
        }
        case VOp::LeakyRelu: {
            const __m256 zero = _mm256_setzero_ps();
            const __m256 slope = _mm256_set1_ps(static_cast<float>(prm.p0));
            TP_BF16_CHUNK_8(_mm256_blendv_ps(
                x, _mm256_mul_ps(slope, x), _mm256_cmp_ps(x, zero, _CMP_LT_OQ)))
            break;
        }
        case VOp::Elu: {
            const __m256 negcoef = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 poscoef = _mm256_set1_ps(static_cast<float>(prm.p1));
            const __m256 negipt = _mm256_set1_ps(static_cast<float>(prm.p2));
            TP_BF16_CHUNK_8(v_elu_f32(negcoef, poscoef, negipt, x))
            break;
        }
        case VOp::Softplus: {
            const __m256 beta = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 threshold = _mm256_set1_ps(static_cast<float>(prm.p1));
            TP_BF16_CHUNK_8(v_softplus_f32(beta, threshold, x))
            break;
        }
        default:
            TP_BF16_CHUNK_8(apply_f32(op, prm, x))
            break;
    }
#undef TP_BF16_STEP
#undef TP_BF16_CHUNK_8
    for (; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float xf;
        std::memcpy(&xf, &bits, 4);
        float yf = scalar_apply(op, prm, xf);
        std::memcpy(&bits, &yf, 4);
        bits = bits + 0x7FFFu + ((bits >> 16) & 1u);
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

#endif // TP_VECUNARY_SLEEF

#ifdef USE_MKL
// The vector math library exposes batch entry points that take a whole
// contiguous array per call.  The mode word is taken from mkl_vml_defines.h:
// VML_HA (0x2) selects the high-accuracy kernel, VML_ERRMODE_IGNORE (0x100)
// lets an out-of-domain lane produce NaN without entering the error callback,
// and VML_FTZDAZ_OFF (0x140000) keeps subnormals.
//
// A lane that is subnormal, infinite, NaN or outside the function's domain
// takes a per-element repair path inside the kernel: a logarithm over one
// negative lane runs four to five times slower, over one subnormal lane fifty
// times.  Screening each block for those lanes would cost a second pass over
// the input, and the kernel is compute-bound while the screen is bound by
// memory, so the two cannot overlap: the screen measured fifteen to twenty
// percent on the fast path.  Paying that on every call to shorten a path only
// unclean data takes is the wrong trade, so blocks are forwarded as they are.
//
// tanh is never forwarded: the piecewise form here is about fifteen percent
// faster than the library kernel.
extern "C" {
void vmsExp(const int, const float[], float[], const long long) noexcept;
void vmsExp2(const int, const float[], float[], const long long) noexcept;
void vmsExpm1(const int, const float[], float[], const long long) noexcept;
void vmsLog1p(const int, const float[], float[], const long long) noexcept;
void vmsSin(const int, const float[], float[], const long long) noexcept;
void vmsCos(const int, const float[], float[], const long long) noexcept;
void vmsAsin(const int, const float[], float[], const long long) noexcept;
void vmsAcos(const int, const float[], float[], const long long) noexcept;
void vmsSinh(const int, const float[], float[], const long long) noexcept;
void vmsCosh(const int, const float[], float[], const long long) noexcept;
void vmsAsinh(const int, const float[], float[], const long long) noexcept;
void vmsAcosh(const int, const float[], float[], const long long) noexcept;
void vmsAtanh(const int, const float[], float[], const long long) noexcept;
void vmsErf(const int, const float[], float[], const long long) noexcept;
void vmsErfc(const int, const float[], float[], const long long) noexcept;
void vmsErfInv(const int, const float[], float[], const long long) noexcept;
void vmsCdfNorm(const int, const float[], float[], const long long) noexcept;
void vmsCdfNormInv(const int, const float[], float[], const long long) noexcept;
void vmsErfcx(const int, const float[], float[], const long long) noexcept;
void vmdExp(const int, const double[], double[], const long long) noexcept;
void vmdExp2(const int, const double[], double[], const long long) noexcept;
void vmdLn(const int, const double[], double[], const long long) noexcept;
void vmdLog2(const int, const double[], double[], const long long) noexcept;
void vmdLog10(const int, const double[], double[], const long long) noexcept;
void vmdLog1p(const int, const double[], double[], const long long) noexcept;
void vmdExpm1(const int, const double[], double[], const long long) noexcept;
void vmdErf(const int, const double[], double[], const long long) noexcept;
void vmdErfc(const int, const double[], double[], const long long) noexcept;
void vmdErfInv(const int, const double[], double[], const long long) noexcept;
void vmdCdfNorm(const int, const double[], double[], const long long) noexcept;
void vmdCdfNormInv(const int, const double[], double[], const long long) noexcept;
void vmdErfcx(const int, const double[], double[], const long long) noexcept;
void vmdSin(const int, const double[], double[], const long long) noexcept;
void vmdCos(const int, const double[], double[], const long long) noexcept;
void vmdTan(const int, const double[], double[], const long long) noexcept;
void vmdAsin(const int, const double[], double[], const long long) noexcept;
void vmdAcos(const int, const double[], double[], const long long) noexcept;
void vmdAtan(const int, const double[], double[], const long long) noexcept;
void vmdSinh(const int, const double[], double[], const long long) noexcept;
void vmdCosh(const int, const double[], double[], const long long) noexcept;
void vmdAsinh(const int, const double[], double[], const long long) noexcept;
void vmdAcosh(const int, const double[], double[], const long long) noexcept;
void vmdAtanh(const int, const double[], double[], const long long) noexcept;
int MKL_Set_Num_Threads_Local(int) noexcept;
}

constexpr long long kVmlMode =
    0x2LL /* VML_HA */ | 0x100LL /* VML_ERRMODE_IGNORE */ |
    0x140000LL /* VML_FTZDAZ_OFF */;

using Vml32Entry = void (*)(const int, const float[], float[], const long long);
using Vml64Entry = void (*)(const int, const double[], double[], const long long);

// Single-precision functions the library evaluates faster than the kernels
// here.  log, log2, log10, sqrt, tan, atan and tanh are absent because the
// local form is at least as fast.
inline Vml32Entry vml32_entry(VOp op) {
    switch (op) {
        case VOp::Exp:   return vmsExp;
        case VOp::Exp2:  return vmsExp2;
        case VOp::Expm1: return vmsExpm1;
        case VOp::Log1p: return vmsLog1p;
        case VOp::Sin:   return vmsSin;
        case VOp::Cos:   return vmsCos;
        case VOp::Asin:  return vmsAsin;
        case VOp::Acos:  return vmsAcos;
        case VOp::Sinh:  return vmsSinh;
        case VOp::Cosh:  return vmsCosh;
        case VOp::Asinh: return vmsAsinh;
        case VOp::Acosh: return vmsAcosh;
        case VOp::Atanh: return vmsAtanh;
        case VOp::Erf:   return vmsErf;
        case VOp::Erfc:  return vmsErfc;
        case VOp::ErfInv: return vmsErfInv;
        case VOp::Ndtr:   return vmsCdfNorm;
        case VOp::NdtrI:  return vmsCdfNormInv;
        case VOp::Erfcx:  return vmsErfcx;
        default:         return nullptr;
    }
}

inline Vml64Entry vml64_entry(VOp op) {
    switch (op) {
        case VOp::Exp:   return vmdExp;
        case VOp::Exp2:  return vmdExp2;
        case VOp::Expm1: return vmdExpm1;
        case VOp::Log:   return vmdLn;
        case VOp::Log2:  return vmdLog2;
        case VOp::Log10: return vmdLog10;
        case VOp::Log1p: return vmdLog1p;
        case VOp::Sin:   return vmdSin;
        case VOp::Cos:   return vmdCos;
        case VOp::Tan:   return vmdTan;
        case VOp::Asin:  return vmdAsin;
        case VOp::Acos:  return vmdAcos;
        case VOp::Atan:  return vmdAtan;
        case VOp::Sinh:  return vmdSinh;
        case VOp::Cosh:  return vmdCosh;
        case VOp::Asinh: return vmdAsinh;
        case VOp::Acosh: return vmdAcosh;
        case VOp::Atanh: return vmdAtanh;
        case VOp::Erf:   return vmdErf;
        case VOp::Erfc:  return vmdErfc;
        case VOp::ErfInv: return vmdErfInv;
        case VOp::Ndtr:   return vmdCdfNorm;
        case VOp::NdtrI:  return vmdCdfNormInv;
        case VOp::Erfcx:  return vmdErfcx;
        default:         return nullptr;
    }
}

#endif

// No vector form below: the library is the only vector path for these, so
// without it the block runs the scalar tail rather than reaching the chunk
// kernels, whose default arm hands the input back unchanged.
inline bool vml_only_op(VOp op) {
    return op == VOp::ErfInv || op == VOp::Ndtr || op == VOp::NdtrI ||
           op == VOp::Erfcx;
}

// ---------------------------------------------------------------------------
// Public entry points: process [b, e) of contiguous src -> dst.
// ---------------------------------------------------------------------------

// Forward declaration: the composed ops below re-enter the walk with the
// building-block op they are made of.
inline void run_f32(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e);
inline void run_f64(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e);

// sinc and logit need the scaling/clamping applied before the transcendental,
// entr the result after it.  Both halves stay in the vector walk: the fix-up is
// one elementwise pass, and the transcendental is the same entry the plain op
// uses, so the composed form costs one extra pass over the block.
inline bool composed_f32(VOp op, VParams prm, const float* src, float* dst, int64_t b,
                         int64_t e) {
    switch (op) {
        case VOp::Sinc: {
            constexpr float kPi = 3.14159265358979323846f;
            for (int64_t i = b; i < e; ++i) dst[i] = kPi * src[i];
            run_f32(VOp::Sin, {}, dst, dst, b, e);
            for (int64_t i = b; i < e; ++i) {
                const float px = kPi * src[i];
                dst[i] = std::fabs(px) < 1e-30f ? 1.0f : dst[i] / px;
            }
            return true;
        }
        case VOp::Entr:
            run_f32(VOp::Log, {}, src, dst, b, e);
            for (int64_t i = b; i < e; ++i) {
                const float x = src[i];
                const float v = dst[i];
                dst[i] = x > 0.0f
                             ? -x * v
                             : (x == 0.0f ? 0.0f
                                          : (x != x ? x
                                                    : -std::numeric_limits<float>::infinity()));
            }
            return true;
        case VOp::Logit: {
            const float eps = static_cast<float>(prm.p0);
            const bool clamp = prm.p0 >= 0.0;
            for (int64_t i = b; i < e; ++i) {
                float p = src[i];
                if (clamp) p = std::min(std::max(p, eps), 1.0f - eps);
                dst[i] = p / (1.0f - p);
            }
            run_f32(VOp::Log, {}, dst, dst, b, e);
            return true;
        }
        case VOp::SphBesselJ0:
            run_f32(VOp::Sin, {}, src, dst, b, e);
            for (int64_t i = b; i < e; ++i)
                dst[i] = spherical_j0_from_sin(src[i], dst[i]);
            return true;
        default:
            return false;
    }
}

inline bool composed_f64(VOp op, VParams prm, const double* src, double* dst, int64_t b,
                         int64_t e) {
    switch (op) {
        case VOp::Sinc: {
            constexpr double kPi = 3.14159265358979323846;
            for (int64_t i = b; i < e; ++i) dst[i] = kPi * src[i];
            run_f64(VOp::Sin, {}, dst, dst, b, e);
            for (int64_t i = b; i < e; ++i) {
                const double px = kPi * src[i];
                dst[i] = std::fabs(px) < 1e-30 ? 1.0 : dst[i] / px;
            }
            return true;
        }
        case VOp::Entr:
            run_f64(VOp::Log, {}, src, dst, b, e);
            for (int64_t i = b; i < e; ++i) {
                const double x = src[i];
                const double v = dst[i];
                dst[i] = x > 0.0
                             ? -x * v
                             : (x == 0.0 ? 0.0
                                         : (x != x ? x
                                                   : -std::numeric_limits<double>::infinity()));
            }
            return true;
        case VOp::Logit: {
            const double eps = prm.p0;
            const bool clamp = prm.p0 >= 0.0;
            for (int64_t i = b; i < e; ++i) {
                double p = src[i];
                if (clamp) p = std::min(std::max(p, eps), 1.0 - eps);
                dst[i] = p / (1.0 - p);
            }
            run_f64(VOp::Log, {}, dst, dst, b, e);
            return true;
        }
        case VOp::SphBesselJ0:
            run_f64(VOp::Sin, {}, src, dst, b, e);
            for (int64_t i = b; i < e; ++i)
                dst[i] = spherical_j0_from_sin(src[i], dst[i]);
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Bessel family.
//
// These are Chebyshev series whose recurrence is serial in the coefficient
// index: one value can only advance one step at a time, so a single value is a
// pure latency chain.  The parallelism has to come from the values instead --
// eight of them (four for double) walk the coefficient list together, which
// turns latency into throughput.  A group that holds a value outside the
// series interval, or a non-finite one, takes the scalar routine; the check is
// per group, so ordinary input never pays for it.
// ---------------------------------------------------------------------------
#if defined(TP_VECUNARY_X86) && defined(__x86_64__) \
    && (defined(__GNUC__) || defined(__clang__))
#define TP_VECUNARY_SIMD_MATH 1
#endif

#ifdef TP_VECUNARY_SIMD_MATH
// The multiply, subtract and add are kept apart rather than fused: the scalar
// routine rounds three times per step, and doing the same here keeps the lanes
// bit-identical to it instead of trading a rounding step for a wider error.
__attribute__((target("avx2,fma")))
static inline __m256 chbevl_f32x8(__m256 x, const float* a, int len) {
    const __m256 zero = _mm256_setzero_ps();
    __m256 b0 = _mm256_set1_ps(a[0]), b1 = zero, b2 = zero;
    for (int i = 1; i < len; ++i) {
        b2 = b1;
        b1 = b0;
        b0 = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(x, b1), b2), _mm256_set1_ps(a[i]));
    }
    return _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_sub_ps(b0, b2));
}

// The series changes at |x| = 8: below it the argument is x/2 - 2 and the
// result scales by x, above it the argument is 32/x - 2 and the result scales
// by 1/sqrt(x).  A group that straddles the split evaluates both and blends,
// which is cheaper than dropping the whole group to the scalar routine.
__attribute__((target("avx2,fma")))
static inline __m256 i1e_small_f32x8(__m256 x, const float* a, int len) {
    const __m256 y = _mm256_sub_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5f)),
                                   _mm256_set1_ps(2.0f));
    return _mm256_mul_ps(chbevl_f32x8(y, a, len), x);
}

__attribute__((target("avx2,fma")))
static inline __m256 i1e_large_f32x8(__m256 x, const float* b, int len) {
    const __m256 y = _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(32.0f), x),
                                   _mm256_set1_ps(2.0f));
    return _mm256_div_ps(chbevl_f32x8(y, b, len), _mm256_sqrt_ps(x));
}

__attribute__((target("avx2,fma")))
static void i1e_block_f32_avx2(const float* src, float* dst, int64_t b, int64_t e) {
    const auto [small, small_len] = special_math::chebyshev_coefficients_i1e_A<float>();
    const auto [large, large_len] = special_math::chebyshev_coefficients_i1e_B<float>();
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    const __m256 eight = _mm256_set1_ps(8.0f);
    const __m256 zero = _mm256_setzero_ps();
    int64_t i = b;
    for (; i + 8 <= e; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        const __m256 x = _mm256_andnot_ps(sgn, v);
        const __m256 below = _mm256_cmp_ps(x, eight, _CMP_LE_OQ);
        const int lanes = _mm256_movemask_ps(below);
        __m256 r;
        if (lanes == 0xFF) {
            r = i1e_small_f32x8(x, small, static_cast<int>(small_len));
        } else if (lanes == 0) {
            r = i1e_large_f32x8(x, large, static_cast<int>(large_len));
        } else {
            r = _mm256_blendv_ps(i1e_large_f32x8(x, large, static_cast<int>(large_len)),
                                 i1e_small_f32x8(x, small, static_cast<int>(small_len)),
                                 below);
        }
        r = _mm256_blendv_ps(r, _mm256_sub_ps(zero, r), _mm256_cmp_ps(v, zero, _CMP_LT_OQ));
        _mm256_storeu_ps(dst + i, r);
    }
    for (; i < e; ++i) dst[i] = scalar_apply(VOp::I1e, {}, src[i]);
}
#endif

inline void i1e_block_f32(const float* src, float* dst, int64_t b, int64_t e) {
#ifdef TP_VECUNARY_SIMD_MATH
    if (avx2_available() && fma_available()) {
        i1e_block_f32_avx2(src, dst, b, e);
        return;
    }
#endif
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(VOp::I1e, {}, src[i]);
}

// The double series carries 29 coefficients, so a four-wide walk only buys a
// few times over the scalar routine; that path keeps the routine it had.
inline void i1e_block_f64(const double* src, double* dst, int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(VOp::I1e, {}, src[i]);
}

inline void run_f32(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
    if (composed_f32(op, prm, src, dst, b, e)) return;
    if (op == VOp::I1e) {
        i1e_block_f32(src, dst, b, e);
        return;
    }
#ifdef USE_MKL
    if (Vml32Entry fn = vml32_entry(op)) {
        MKL_Set_Num_Threads_Local(1);
        constexpr int64_t kSpan = int64_t(1) << 24;
        for (int64_t i = b; i < e; i += kSpan) {
            const int64_t m = (e - i) < kSpan ? (e - i) : kSpan;
            fn(static_cast<int>(m), src + i, dst + i, kVmlMode);
        }
        return;
    }
#endif
    if (vml_only_op(op)) {
        for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
        return;
    }
#ifdef TP_VECUNARY_SLEEF
#if defined(CPU_CAPABILITY_AVX512)
    // Tier-compiled copy: the ISA is guaranteed by the tier's -m flags, so
    // the runtime CPUID branch disappears entirely.
    f32_chunk_avx512(op, prm, src, dst, b, e);
    return;
#endif
    if (avx512_available()) {
        f32_chunk_avx512(op, prm, src, dst, b, e);
        return;
    }
    if (avx2_available()) {
        f32_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}


inline void run_f64(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
    if (composed_f64(op, prm, src, dst, b, e)) return;
    if (op == VOp::I1e) {
        i1e_block_f64(src, dst, b, e);
        return;
    }
#ifdef USE_MKL
    if (Vml64Entry fn = vml64_entry(op)) {
        // Every worker owns a slice of the walk, so the library must not spread
        // that slice over threads of its own.
        MKL_Set_Num_Threads_Local(1);
        // The entry points index with a 32-bit count; the cap keeps a slice
        // from ever wrapping one.
        constexpr int64_t kSpan = int64_t(1) << 24;
        for (int64_t i = b; i < e; i += kSpan) {
            const int64_t m = (e - i) < kSpan ? (e - i) : kSpan;
            fn(static_cast<int>(m), src + i, dst + i, kVmlMode);
        }
        return;
    }
#endif
    if (vml_only_op(op)) {
        for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
        return;
    }
#ifdef TP_VECUNARY_SLEEF
#if defined(CPU_CAPABILITY_AVX512)
    // Tier-compiled copy (see run_f32).  The f64_avx2_only ops have no
    // 512-bit double kernel and run on the AVX2 one.
    if (!f64_avx2_only(op)) {
        f64_chunk_avx512(op, prm, src, dst, b, e);
        return;
    }
    f64_chunk_avx2(op, prm, src, dst, b, e);
    return;
#endif
    // Elu/Softplus/LeakyRelu/Hardswish/Hardsigmoid have AVX2 double
    // kernels only; everything else goes 512-bit.
    if (avx512_available() && !f64_avx2_only(op)) {
        f64_chunk_avx512(op, prm, src, dst, b, e);
        return;
    }
    if (avx2_available()) {
        f64_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}

inline bool vec_ready() {
#ifdef TP_VECUNARY_SLEEF
    return avx2_available();
#else
    return false;
#endif
}

// fp16 path: widen to f32 (FMA/F16C), run the f32 kernels, narrow back with
// round-to-nearest-even — same semantics as tensorplay::Half(float).
// Scalar fallback goes through the Half type itself.
inline void run_f16(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    if (vml_only_op(op)) {
        const Half* hs = reinterpret_cast<const Half*>(src);
        Half* hd = reinterpret_cast<Half*>(dst);
        for (int64_t i = b; i < e; ++i)
            hd[i] = static_cast<Half>(scalar_apply(op, prm, static_cast<float>(hs[i])));
        return;
    }
#ifdef TP_VECUNARY_SLEEF
    if (avx2_available() && f16c_available()) {
        half_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    const Half* hs = reinterpret_cast<const Half*>(src);
    Half* hd = reinterpret_cast<Half*>(dst);
    for (int64_t i = b; i < e; ++i) {
        hd[i] = static_cast<Half>(scalar_apply(op, prm, static_cast<float>(hs[i])));
    }
}

// bf16 path: widen by bit shift, compute, narrow with round-to-nearest-even
// (matches tensorplay::BFloat16(float)). Scalar fallback via BFloat16 type.
inline void run_bf16(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    if (vml_only_op(op)) {
        const BFloat16* bs = reinterpret_cast<const BFloat16*>(src);
        BFloat16* bd = reinterpret_cast<BFloat16*>(dst);
        for (int64_t i = b; i < e; ++i)
            bd[i] = static_cast<BFloat16>(scalar_apply(op, prm, static_cast<float>(bs[i])));
        return;
    }
#ifdef TP_VECUNARY_SLEEF
    if (avx2_available()) {
        bf16_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    const BFloat16* bs = reinterpret_cast<const BFloat16*>(src);
    BFloat16* bd = reinterpret_cast<BFloat16*>(dst);
    for (int64_t i = b; i < e; ++i) {
        bd[i] = static_cast<BFloat16>(scalar_apply(op, prm, static_cast<float>(bs[i])));
    }
}

} // namespace vecunary
} // namespace cpu
} // namespace tensorplay
