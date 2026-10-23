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
#include <cstdint>
#include <cstring>

#include "Half.h"
#include "BFloat16.h"

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
    Exp, Expm1, Log, Log2, Log10, Log1p,
    Sin, Cos, Tan, Asin, Acos, Atan,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
    Erf, Erfc, Lgamma,
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

// Natural logarithm of a float vector.
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
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log_f32(__m256 x) {
    // A subnormal leaves the exponent field at zero and its leading bit in the
    // mantissa, so it is scaled into the normal range first and the factor is
    // paid back out of the exponent.
    const __m256 tiny = _mm256_cmp_ps(x, _mm256_set1_ps(1.1754944e-38f), _CMP_LT_OQ);
    const __m256i ix = _mm256_castps_si256(
        _mm256_blendv_ps(x, _mm256_mul_ps(x, _mm256_set1_ps(1.8446744e19f)), tiny));
    // Adding half a mantissa range before shifting rounds the exponent to
    // nearest.  Subtracting the exponent field from the original bit pattern
    // then leaves 0.75 <= m < 1.5.
    const __m256i biased =
        _mm256_srli_epi32(_mm256_add_epi32(ix, _mm256_set1_epi32(0x00400000)), 23);
    const __m256 m = _mm256_castsi256_ps(_mm256_add_epi32(
        _mm256_sub_epi32(ix, _mm256_slli_epi32(biased, 23)),
        _mm256_set1_epi32(0x3F800000)));
    __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(biased, _mm256_set1_epi32(127)));
    e = _mm256_sub_ps(e, _mm256_and_ps(tiny, _mm256_set1_ps(64.0f)));

    const __m256 u = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 p = _mm256_set1_ps(-4.413553457e-2f);
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.032194468e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.310681637e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.448485294e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.665304757e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.999078531e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-2.499990939e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(3.333346255e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-5.000000213e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(9.999999972e-1f));
    // Two-part ln2 keeps the exponent contribution accurate for large |e|;
    // the mantissa term is the whole answer when e is zero.
    __m256 r = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693145751953125f), _mm256_mul_ps(u, p));
    r = _mm256_fmadd_ps(e, _mm256_set1_ps(1.428606765330187e-6f), r);

    // log(0) is -inf, log(+inf) is +inf, and negative or NaN inputs give NaN.
    // max(x, 0) returns 0 for a NaN first operand, so one compare covers both
    // the negative and the unordered cases.
    const __m256 zero = _mm256_setzero_ps();
    const __m256 bad = _mm256_cmp_ps(_mm256_max_ps(x, zero), x, _CMP_NEQ_UQ);
    r = _mm256_blendv_ps(r, _mm256_set1_ps(-INFINITY), _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));
    r = _mm256_blendv_ps(r, x, _mm256_cmp_ps(x, _mm256_set1_ps(3.4028235e38f), _CMP_GT_OQ));
    return _mm256_blendv_ps(r, _mm256_set1_ps(NAN), bad);
}

// Range-restricted log for callers that have already proved every lane is a
// positive normal float.  Dropping the subnormal rescale and the domain
// fix-up removes five instructions; the exponent term also needs only one
// ln2 FMA here, because its relative error is |ln2 - ln2f| / ln2 = 2.7e-9
// whatever the exponent -- unlike in exp, where the reduced argument feeds
// straight into the result and the error grows with the exponent.
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_log_f32_fast(__m256 x) {
    const __m256i ix = _mm256_castps_si256(x);
    const __m256i biased =
        _mm256_srli_epi32(_mm256_add_epi32(ix, _mm256_set1_epi32(0x00400000)), 23);
    const __m256i ei = _mm256_sub_epi32(biased, _mm256_set1_epi32(127));
    // Removing the exponent field leaves x / 2^ei in [0.75, 1.5).  Taking the
    // bias out first keeps the mantissa rebuild to one shift and one
    // subtract; the shift is a logical one, which is still correct for a
    // negative exponent because the arithmetic is modulo 2^32.
    const __m256 m = _mm256_castsi256_ps(
        _mm256_sub_epi32(ix, _mm256_slli_epi32(ei, 23)));
    const __m256 e = _mm256_cvtepi32_ps(ei);
    const __m256 u = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 p = _mm256_set1_ps(-4.413553457e-2f);
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.032194468e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.310681637e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.448485294e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-1.665304757e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.999078531e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-2.499990939e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(3.333346255e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(-5.000000213e-1f));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(9.999999972e-1f));
    return _mm256_fmadd_ps(e, _mm256_set1_ps(0.693147182464599609375f),
                           _mm256_mul_ps(u, p));
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
__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_erf_f32(__m256 x) {
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
    const __m256 decay = v_exp_f32(_mm256_xor_ps(sign_bit, _mm256_mul_ps(x, x)));
    const __m256 r = _mm256_sub_ps(one, _mm256_mul_ps(p, decay));
    return _mm256_or_ps(_mm256_and_ps(sign_bit, x), r);
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

__attribute__((target("avx2,fma"), always_inline))
inline __m256 v_gelu_none_f32(__m256 x, __m256 one) {
    const __m256 kAlpha = _mm256_set1_ps(static_cast<float>(0.70710678118654752440));
    __m256 cdf = _mm256_add_ps(one, v_erf_f32(_mm256_mul_ps(kAlpha, x)));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
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
        case VOp::Expm1: return tensorplay::tpsleef::expm1(x);
        case VOp::Log: return v_log_f32(x);
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
        case VOp::Log: {
            // Same blocking rule as exp: a block of 32 whose lanes are all
            // positive normal floats takes the short chain, and anything
            // else -- zero, negatives, subnormals, inf, NaN -- runs the
            // guarded version for that block only.
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
                    _mm256_storeu_ps(d + i, v_log_f32_fast(x0));
                    _mm256_storeu_ps(d + i + 8, v_log_f32_fast(x1));
                    _mm256_storeu_ps(d + i + 16, v_log_f32_fast(x2));
                    _mm256_storeu_ps(d + i + 24, v_log_f32_fast(x3));
                } else {
                    _mm256_storeu_ps(d + i, v_log_f32(x0));
                    _mm256_storeu_ps(d + i + 8, v_log_f32(x1));
                    _mm256_storeu_ps(d + i + 16, v_log_f32(x2));
                    _mm256_storeu_ps(d + i + 24, v_log_f32(x3));
                }
            }
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(d + i, v_log_f32(_mm256_loadu_ps(s + i)));
            }
            break;
        }
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
        case VOp::GeluNone:
            TP_F32_SIMPLE_LOOP_8(v_gelu_none_f32(x, one));
            break;
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
        case VOp::Log2:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::log2(x));
            break;
        case VOp::Log10:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::log10(x));
            break;
        case VOp::Sin:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::sin(x));
            break;
        case VOp::Cos:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::cos(x));
            break;
        case VOp::Tan:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::tan(x));
            break;
        case VOp::Asin:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::asin(x));
            break;
        case VOp::Acos:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::acos(x));
            break;
        case VOp::Atan:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::atan(x));
            break;
        case VOp::Erf:
            TP_F32_SIMPLE_LOOP_8(v_erf_f32(x));
            break;
        case VOp::Erfc:
            TP_F32_SIMPLE_LOOP_8(tensorplay::tpsleef::erfc(x));
            break;
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
        case VOp::Exp: return tensorplay::tpsleef::exp(x);
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
        case VOp::Lgamma: return v_lgamma_pos_f64(x);
        case VOp::Sigmoid:
            return _mm256_div_pd(one, _mm256_add_pd(one, tensorplay::tpsleef::exp(_mm256_xor_pd(x, signbit))));
        case VOp::GeluNone: {
            const __m256d kAlpha = _mm256_set1_pd(0.70710678118654752440);
            __m256d cdf = _mm256_add_pd(one, tensorplay::tpsleef::erf(_mm256_mul_pd(kAlpha, x)));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m256d kBeta = _mm256_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m256d kKappa = _mm256_set1_pd(0.044715);
            __m256d x_cube = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
            __m256d inner = _mm256_mul_pd(kBeta, _mm256_add_pd(x, _mm256_mul_pd(kKappa, x_cube)));
            __m256d cdf = _mm256_add_pd(one, tensorplay::tpsleef::tanh(inner));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::Silu: {
            __m256d den = _mm256_add_pd(one, tensorplay::tpsleef::exp(_mm256_xor_pd(x, signbit)));
            return _mm256_div_pd(x, den);
        }
        case VOp::Mish: {
            __m256d sp = tensorplay::tpsleef::log(_mm256_add_pd(one, tensorplay::tpsleef::exp(x)));
            return _mm256_mul_pd(x, tensorplay::tpsleef::tanh(sp));
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
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::exp(x));
            break;
        case VOp::Expm1:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::expm1(x));
            break;
        case VOp::Log:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::log(x));
            break;
        case VOp::Log1p:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::log1p(x));
            break;
        case VOp::Log2:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::log2(x));
            break;
        case VOp::Log10:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::log10(x));
            break;
        case VOp::Tanh:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::tanh(x));
            break;
        case VOp::Sigmoid:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                one, _mm256_add_pd(one, tensorplay::tpsleef::exp(_mm256_xor_pd(x, signbit)))));
            break;
        case VOp::Silu:
            TP_F64_SIMPLE_LOOP_4(_mm256_div_pd(
                x, _mm256_add_pd(one, tensorplay::tpsleef::exp(_mm256_xor_pd(x, signbit)))));
            break;
        case VOp::Mish:
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                x, tensorplay::tpsleef::tanh(tensorplay::tpsleef::log(
                       _mm256_add_pd(one, tensorplay::tpsleef::exp(x))))));
            break;
        case VOp::GeluNone: {
            const __m256d kAlpha = _mm256_set1_pd(0.70710678118654752440);
            const __m256d half = _mm256_set1_pd(0.5);
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                _mm256_mul_pd(half, x),
                _mm256_add_pd(one, tensorplay::tpsleef::erf(_mm256_mul_pd(kAlpha, x)))));
            break;
        }
        case VOp::GeluTanh: {
            const __m256d kBeta = _mm256_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m256d kKappa = _mm256_set1_pd(0.044715);
            const __m256d half = _mm256_set1_pd(0.5);
            TP_F64_SIMPLE_LOOP_4(_mm256_mul_pd(
                _mm256_mul_pd(half, x),
                _mm256_add_pd(one, tensorplay::tpsleef::tanh(_mm256_mul_pd(
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
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::erf(x));
            break;
        case VOp::Erfc:
            TP_F64_SIMPLE_LOOP_4(tensorplay::tpsleef::erfc(x));
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

// ---------------------------------------------------------------------------
// Public entry points: process [b, e) of contiguous src -> dst.
// ---------------------------------------------------------------------------
inline void run_f32(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
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
