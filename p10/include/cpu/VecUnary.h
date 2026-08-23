#pragma once

// Vectorized fast paths for CPU unary kernels.
//
// Design notes:
// * Every kernel here is a single translation-unit-local function built with
//   GCC per-function target attributes ("avx2", "fma", "f16c"), so p10 keeps
//   compiling without ISA flags while the hot loops get real AVX2 codegen.
//   Dispatch happens once per chunk through avx2_available(); below-AVX2
//   machines transparently keep the previous scalar behaviour.
// * Transcendentals resolve to glibc's libmvec vector ABI (_ZGVdN8v_*f /
//   _ZGVdN4v_*, merged into libm since glibc 2.35 and already linked into
//   libp10).  Composite activations reuse those building blocks so results
//   stay bit-compatible with the scalar formulas used by PointwiseKernels.cpp.
// * lgamma has no libmvec entry point; it uses a Lanczos expansion restricted
//   to strictly positive inputs (blocks containing non-positive or NaN values
//   fall back to the scalar std::lgamma, which handles the reflection domain).
// * All scalar fallbacks mirror the ATen-aligned formulas in
//   PointwiseKernels.cpp element-for-element, including multiplication order,
//   so vector and fallback paths produce identical values.

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "Half.h"
#include "BFloat16.h"

#if defined(__x86_64__) || defined(__i386__)
#define TP_VECUNARY_X86 1
#endif

#if defined(TP_VECUNARY_X86) && defined(__GLIBC__) && defined(__x86_64__)
#define TP_VECUNARY_LIBMVEC 1
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
    Hardtanh, Relu6, Celu,
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

// ---------------------------------------------------------------------------
// libmvec declarations (glibc >= 2.35 ships these inside libm/libmvec; p10
// already links libmvec for the stax fused kernels).
// ---------------------------------------------------------------------------
#ifdef TP_VECUNARY_LIBMVEC
extern "C" {
__m256 _ZGVdN8v_acosf(__m256);
__m256 _ZGVdN8v_acoshf(__m256);
__m256 _ZGVdN8v_asinf(__m256);
__m256 _ZGVdN8v_asinhf(__m256);
__m256 _ZGVdN8v_atanf(__m256);
__m256 _ZGVdN8v_atanhf(__m256);
__m256 _ZGVdN8v_cosf(__m256);
__m256 _ZGVdN8v_coshf(__m256);
__m256 _ZGVdN8v_erff(__m256);
__m256 _ZGVdN8v_erfcf(__m256);
__m256 _ZGVdN8v_expf(__m256);
__m256 _ZGVdN8v_expm1f(__m256);
__m256 _ZGVdN8v_logf(__m256);
__m256 _ZGVdN8v_log10f(__m256);
__m256 _ZGVdN8v_log1pf(__m256);
__m256 _ZGVdN8v_log2f(__m256);
__m256 _ZGVdN8v_sinf(__m256);
__m256 _ZGVdN8v_sinhf(__m256);
__m256 _ZGVdN8v_tanf(__m256);
__m256 _ZGVdN8v_tanhf(__m256);

__m256d _ZGVdN4v_acos(__m256d);
__m256d _ZGVdN4v_acosh(__m256d);
__m256d _ZGVdN4v_asin(__m256d);
__m256d _ZGVdN4v_asinh(__m256d);
__m256d _ZGVdN4v_atan(__m256d);
__m256d _ZGVdN4v_atanh(__m256d);
__m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_cosh(__m256d);
__m256d _ZGVdN4v_erf(__m256d);
__m256d _ZGVdN4v_erfc(__m256d);
__m256d _ZGVdN4v_exp(__m256d);
__m256d _ZGVdN4v_expm1(__m256d);
__m256d _ZGVdN4v_log(__m256d);
__m256d _ZGVdN4v_log10(__m256d);
__m256d _ZGVdN4v_log1p(__m256d);
__m256d _ZGVdN4v_log2(__m256d);
__m256d _ZGVdN4v_sin(__m256d);
__m256d _ZGVdN4v_sinh(__m256d);
__m256d _ZGVdN4v_tan(__m256d);
__m256d _ZGVdN4v_tanh(__m256d);
}
#endif // TP_VECUNARY_LIBMVEC

// ---------------------------------------------------------------------------
// Scalar reference implementations.  These mirror the lambdas in
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
            return x < T(0) ? static_cast<T>(std::expm1(static_cast<T>(float(x) * float(negipt))) * float(negcoef))
                            : x * poscoef;
        }
        case VOp::Softplus: {
            const T beta = static_cast<T>(prm.p0);
            const T threshold = static_cast<T>(prm.p1);
            return x * beta > threshold
                ? x
                : static_cast<T>(std::log1p(std::exp(static_cast<float>(x * beta))) / prm.p0);
        }
        case VOp::Hardswish: {
            T xf = static_cast<T>(static_cast<float>(x));
            T clamped = (xf + T(3) < T(0)) ? T(0) : (xf + T(3) > T(6)) ? T(6) : xf + T(3);
            return xf * clamped / T(6);
        }
        case VOp::Hardsigmoid: {
            T xf = static_cast<T>(static_cast<float>(x));
            T v = xf + T(3);
            v = v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
            return v / T(6);
        }
        case VOp::LeakyRelu: {
            const T slope = static_cast<T>(prm.p0);
            T xf = static_cast<T>(static_cast<float>(x));
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
        case VOp::Celu: {
            const T a = static_cast<T>(prm.p0);
            T af = static_cast<T>(static_cast<float>(x));
            return af > T(0) ? af : static_cast<T>(static_cast<double>(a)) * (std::expm1(af / a));
        }
        default: return x;
    }
}

// ---------------------------------------------------------------------------
// AVX2 helpers.  Everything below either carries a target attribute itself or
// is always_inline so it folds into an attributed caller.
// ---------------------------------------------------------------------------
#ifdef TP_VECUNARY_LIBMVEC

constexpr float kSignBitF = -0.0f;

// Lanczos g=7, n=9 coefficients.
constexpr double kLanczos[9] = {
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
constexpr double kHalfLogTwoPi = 0.91893853320467274178032973640562;

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
                        _ZGVdN8v_logf(t), _mm256_set1_ps(static_cast<float>(kHalfLogTwoPi))),
        t);
    res = _mm256_add_ps(res, _ZGVdN8v_logf(series));
    return _mm256_sub_ps(res, _mm256_and_ps(small, _ZGVdN8v_logf(x)));
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
                        _ZGVdN4v_log(t), _mm256_set1_pd(kHalfLogTwoPi)),
        t);
    res = _mm256_add_pd(res, _ZGVdN4v_log(series));
    return _mm256_sub_pd(res, _mm256_and_pd(small, _ZGVdN4v_log(x)));
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
        case VOp::Exp: return _ZGVdN8v_expf(x);
        case VOp::Expm1: return _ZGVdN8v_expm1f(x);
        case VOp::Log: return _ZGVdN8v_logf(x);
        case VOp::Log2: return _ZGVdN8v_log2f(x);
        case VOp::Log10: return _ZGVdN8v_log10f(x);
        case VOp::Log1p: return _ZGVdN8v_log1pf(x);
        case VOp::Sin: return _ZGVdN8v_sinf(x);
        case VOp::Cos: return _ZGVdN8v_cosf(x);
        case VOp::Tan: return _ZGVdN8v_tanf(x);
        case VOp::Asin: return _ZGVdN8v_asinf(x);
        case VOp::Acos: return _ZGVdN8v_acosf(x);
        case VOp::Atan: return _ZGVdN8v_atanf(x);
        case VOp::Sinh: return _ZGVdN8v_sinhf(x);
        case VOp::Cosh: return _ZGVdN8v_coshf(x);
        case VOp::Tanh: return _ZGVdN8v_tanhf(x);
        case VOp::Asinh: return _ZGVdN8v_asinhf(x);
        case VOp::Acosh: return _ZGVdN8v_acoshf(x);
        case VOp::Atanh: return _ZGVdN8v_atanhf(x);
        case VOp::Erf: return _ZGVdN8v_erff(x);
        case VOp::Erfc: return _ZGVdN8v_erfcf(x);
        case VOp::Lgamma: {
            // Callers guarantee all lanes > 0 here.
            return v_lgamma_pos_f32(x);
        }
        case VOp::Sigmoid:
            return _mm256_div_ps(one, _mm256_add_ps(one, _ZGVdN8v_expf(_mm256_xor_ps(x, signbit))));
        case VOp::GeluNone: {
            const __m256 kAlpha = _mm256_set1_ps(static_cast<float>(0.70710678118654752440));
            __m256 cdf = _mm256_add_ps(one, _ZGVdN8v_erff(_mm256_mul_ps(kAlpha, x)));
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m256 kBeta = _mm256_set1_ps(static_cast<float>(1.41421356237309504880 * 1.12837916709551257390 * 0.5));
            const __m256 kKappa = _mm256_set1_ps(0.044715f);
            __m256 x_cube = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            __m256 inner = _mm256_mul_ps(kBeta, _mm256_add_ps(x, _mm256_mul_ps(kKappa, x_cube)));
            __m256 cdf = _mm256_add_ps(one, _ZGVdN8v_tanhf(inner));
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
        }
        case VOp::Silu: {
            __m256 den = _mm256_add_ps(one, _ZGVdN8v_expf(_mm256_xor_ps(x, signbit)));
            return _mm256_div_ps(x, den);
        }
        case VOp::Mish: {
            // Mirrors the scalar kernel: log((1 + exp(x))) * tanh(...) — plain
            // log, not log1p, and overflow-to-inf semantics are preserved.
            __m256 sp = _ZGVdN8v_logf(_mm256_add_ps(one, _ZGVdN8v_expf(x)));
            return _mm256_mul_ps(x, _ZGVdN8v_tanhf(sp));
        }
        case VOp::Selu: {
            const __m256 lambda = _mm256_set1_ps(static_cast<float>(1.0507009873554804934193349852946));
            const __m256 alphalambda = _mm256_set1_ps(static_cast<float>(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946));
            __m256 pos = _mm256_mul_ps(x, lambda);
            __m256 neg = _mm256_mul_ps(alphalambda, _ZGVdN8v_expm1f(x));
            return _mm256_blendv_ps(neg, pos, _mm256_cmp_ps(x, zero, _CMP_GT_OQ));
        }
        case VOp::Elu: {
            const __m256 negcoef = _mm256_set1_ps(static_cast<float>(prm.p0)); // alpha*scale
            const __m256 poscoef = _mm256_set1_ps(static_cast<float>(prm.p1)); // scale
            const __m256 negipt = _mm256_set1_ps(static_cast<float>(prm.p2));  // input_scale
            __m256 scaled = _mm256_mul_ps(x, negipt);
            __m256 neg = _mm256_mul_ps(_ZGVdN8v_expm1f(scaled), negcoef);
            __m256 pos = _mm256_mul_ps(x, poscoef);
            return _mm256_blendv_ps(neg, pos, _mm256_cmp_ps(x, zero, _CMP_GE_OQ));
        }
        case VOp::Softplus: {
            // Scalar path: numerator computed in float (log1p(exp(float(x*beta)))),
            // division by beta in double, one final round back to T.
            const __m256 beta = _mm256_set1_ps(static_cast<float>(prm.p0));
            const __m256 threshold = _mm256_set1_ps(static_cast<float>(prm.p1));
            const __m256d beta_d = _mm256_set1_pd(prm.p0);
            __m256 bx = _mm256_mul_ps(x, beta);
            __m256 numf = _ZGVdN8v_log1pf(_ZGVdN8v_expf(bx));
            __m256d lo_d = _mm256_div_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(numf)), beta_d);
            __m256d hi_d = _mm256_div_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(numf, 1)), beta_d);
            __m256 sp = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo_d)),
                                             _mm256_cvtpd_ps(hi_d), 1);
            return _mm256_blendv_ps(sp, x, _mm256_cmp_ps(bx, threshold, _CMP_GT_OQ));
        }
        case VOp::Hardswish: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            __m256 y = _mm256_add_ps(x, three);
            y = _mm256_min_ps(_mm256_max_ps(y, zero), six);
            return _mm256_div_ps(_mm256_mul_ps(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m256 three = _mm256_set1_ps(3.0f);
            const __m256 six = _mm256_set1_ps(6.0f);
            __m256 y = _mm256_add_ps(x, three);
            y = _mm256_min_ps(_mm256_max_ps(y, zero), six);
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
            return _mm256_min_ps(_mm256_max_ps(x, lo), hi);
        }
        case VOp::Relu6:
            return _mm256_min_ps(_mm256_max_ps(x, zero), _mm256_set1_ps(6.0f));
        case VOp::Celu: {
            const __m256 a = _mm256_set1_ps(static_cast<float>(prm.p0));
            __m256 neg = _mm256_mul_ps(a, _ZGVdN8v_expm1f(_mm256_div_ps(x, a)));
            __m256 pos = _mm256_max_ps(zero, x);
            __m256 minneg = _mm256_min_ps(neg, zero);
            return _mm256_add_ps(pos, minneg);
        }
        default: return x;
    }
}

__attribute__((target("avx2,fma"), noinline))
static void f32_chunk_avx2(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        if (op == VOp::Lgamma && _mm256_movemask_ps(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ))) {
            // Non-positive lane (or NaN): scalar fallback keeps the exact
            // reflection-domain behaviour of std::lgamma.
            for (int64_t j = i; j < i + 8; ++j) dst[j] = scalar_apply(op, prm, src[j]);
        } else {
            _mm256_storeu_ps(dst + i, apply_f32(op, prm, x));
        }
    }
    for (; i < n; ++i) dst[i] = scalar_apply(op, prm, src[i]);
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
        case VOp::Exp: return _ZGVdN4v_exp(x);
        case VOp::Expm1: return _ZGVdN4v_expm1(x);
        case VOp::Log: return _ZGVdN4v_log(x);
        case VOp::Log2: return _ZGVdN4v_log2(x);
        case VOp::Log10: return _ZGVdN4v_log10(x);
        case VOp::Log1p: return _ZGVdN4v_log1p(x);
        case VOp::Sin: return _ZGVdN4v_sin(x);
        case VOp::Cos: return _ZGVdN4v_cos(x);
        case VOp::Tan: return _ZGVdN4v_tan(x);
        case VOp::Asin: return _ZGVdN4v_asin(x);
        case VOp::Acos: return _ZGVdN4v_acos(x);
        case VOp::Atan: return _ZGVdN4v_atan(x);
        case VOp::Sinh: return _ZGVdN4v_sinh(x);
        case VOp::Cosh: return _ZGVdN4v_cosh(x);
        case VOp::Tanh: return _ZGVdN4v_tanh(x);
        case VOp::Asinh: return _ZGVdN4v_asinh(x);
        case VOp::Acosh: return _ZGVdN4v_acosh(x);
        case VOp::Atanh: return _ZGVdN4v_atanh(x);
        case VOp::Erf: return _ZGVdN4v_erf(x);
        case VOp::Erfc: return _ZGVdN4v_erfc(x);
        case VOp::Lgamma: return v_lgamma_pos_f64(x);
        case VOp::Sigmoid:
            return _mm256_div_pd(one, _mm256_add_pd(one, _ZGVdN4v_exp(_mm256_xor_pd(x, signbit))));
        case VOp::GeluNone: {
            const __m256d kAlpha = _mm256_set1_pd(0.70710678118654752440);
            __m256d cdf = _mm256_add_pd(one, _ZGVdN4v_erf(_mm256_mul_pd(kAlpha, x)));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m256d kBeta = _mm256_set1_pd(1.41421356237309504880 * 1.12837916709551257390 * 0.5);
            const __m256d kKappa = _mm256_set1_pd(0.044715);
            __m256d x_cube = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
            __m256d inner = _mm256_mul_pd(kBeta, _mm256_add_pd(x, _mm256_mul_pd(kKappa, x_cube)));
            __m256d cdf = _mm256_add_pd(one, _ZGVdN4v_tanh(inner));
            return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), cdf);
        }
        case VOp::Silu: {
            __m256d den = _mm256_add_pd(one, _ZGVdN4v_exp(_mm256_xor_pd(x, signbit)));
            return _mm256_div_pd(x, den);
        }
        case VOp::Mish: {
            __m256d sp = _ZGVdN4v_log(_mm256_add_pd(one, _ZGVdN4v_exp(x)));
            return _mm256_mul_pd(x, _ZGVdN4v_tanh(sp));
        }
        case VOp::Selu: {
            const __m256d lambda = _mm256_set1_pd(1.0507009873554804934193349852946);
            const __m256d alphalambda = _mm256_set1_pd(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946);
            __m256d pos = _mm256_mul_pd(x, lambda);
            __m256d neg = _mm256_mul_pd(alphalambda, _ZGVdN4v_expm1(x));
            return _mm256_blendv_pd(neg, pos, _mm256_cmp_pd(x, zero, _CMP_GT_OQ));
        }
        case VOp::Elu: {
            const __m256d negcoef = _mm256_set1_pd(prm.p0);
            const __m256d poscoef = _mm256_set1_pd(prm.p1);
            const __m256d negipt = _mm256_set1_pd(prm.p2);
            // Scalar f64 kernel: a = double(float(x)); expm1(float(a)*float(ipt))
            // in double, times float(negcoef); positive branch a*poscoef.
            __m256d xf = _mm256_cvtps_pd(_mm256_cvtpd_ps(x));
            __m256d scaled = _mm256_mul_pd(xf, _mm256_cvtps_pd(_mm256_cvtpd_ps(negipt)));
            __m256d neg = _mm256_mul_pd(_ZGVdN4v_expm1(scaled),
                                        _mm256_cvtps_pd(_mm256_cvtpd_ps(negcoef)));
            __m256d pos = _mm256_mul_pd(xf, poscoef);
            return _mm256_blendv_pd(neg, pos, _mm256_cmp_pd(xf, _mm256_setzero_pd(), _CMP_GE_OQ));
        }
        case VOp::Softplus: {
            // Scalar f64: a = double(float(x)); threshold test on a*beta;
            // numerator log1p(exp(float(a*beta))) computed in float, divided
            // by beta (double), rounded once to T.
            const __m256d beta = _mm256_set1_pd(prm.p0);
            const __m256d threshold = _mm256_set1_pd(prm.p1);
            __m256d xf = _mm256_cvtps_pd(_mm256_cvtpd_ps(x));
            __m256d bt = _mm256_mul_pd(xf, beta);
            __m128 btf4 = _mm256_cvtpd_ps(bt); // float(x*beta), all 4 lanes
            __m256 btf8 = _mm256_insertf128_ps(_mm256_setzero_ps(), btf4, 0);
            __m256d num = _mm256_cvtps_pd(
                _mm256_castps256_ps128(_ZGVdN8v_log1pf(_ZGVdN8v_expf(btf8))));
            __m256d sp = _mm256_div_pd(num, beta);
            return _mm256_blendv_pd(sp, x, _mm256_cmp_pd(bt, threshold, _CMP_GT_OQ));
        }
        case VOp::Hardswish: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            __m256d y = _mm256_add_pd(x, three);
            y = _mm256_min_pd(_mm256_max_pd(y, zero), six);
            return _mm256_div_pd(_mm256_mul_pd(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m256d three = _mm256_set1_pd(3.0);
            const __m256d six = _mm256_set1_pd(6.0);
            __m256d y = _mm256_add_pd(x, three);
            y = _mm256_min_pd(_mm256_max_pd(y, zero), six);
            return _mm256_div_pd(y, six);
        }
        case VOp::LeakyRelu: {
            // Scalar f64: xf = double(float(x)); slope stays full double.
            const __m256d slope = _mm256_set1_pd(prm.p0);
            __m256d xf = _mm256_cvtps_pd(_mm256_cvtpd_ps(x));
            __m256d neg = _mm256_mul_pd(slope, xf);
            return _mm256_blendv_pd(xf, neg, _mm256_cmp_pd(xf, _mm256_setzero_pd(), _CMP_LT_OQ));
        }
        case VOp::Hardtanh: {
            const __m256d lo = _mm256_set1_pd(prm.p0);
            const __m256d hi = _mm256_set1_pd(prm.p1);
            return _mm256_min_pd(_mm256_max_pd(x, lo), hi);
        }
        case VOp::Relu6:
            return _mm256_min_pd(_mm256_max_pd(x, zero), _mm256_set1_pd(6.0));
        case VOp::Celu: {
            // Scalar f64: af = double(float(x)); alpha stays full double.
            const __m256d a = _mm256_set1_pd(prm.p0);
            __m256d af = _mm256_cvtps_pd(_mm256_cvtpd_ps(x));
            __m256d neg = _mm256_mul_pd(a, _ZGVdN4v_expm1(_mm256_div_pd(af, a)));
            __m256d pos = _mm256_max_pd(_mm256_setzero_pd(), af);
            __m256d minneg = _mm256_min_pd(neg, _mm256_setzero_pd());
            return _mm256_add_pd(pos, minneg);
        }
        default: return x;
    }
}

__attribute__((target("avx2,fma"), noinline))
static void f64_chunk_avx2(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(src + i);
        if (op == VOp::Lgamma && _mm256_movemask_pd(_mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OQ))) {
            for (int64_t j = i; j < i + 4; ++j) dst[j] = scalar_apply(op, prm, src[j]);
        } else {
            _mm256_storeu_pd(dst + i, apply_f64(op, prm, x));
        }
    }
    for (; i < n; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}

// Reduced-precision dtypes widen to f32, run the same kernels, narrow back.
__attribute__((target("avx2,fma,f16c"), noinline))
static void half_chunk_avx2(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
        __m256 y = apply_f32(op, prm, x);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(y, _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < n; ++i) {
        float xf = _cvtsh_ss(src[i]);
        dst[i] = _cvtss_sh(scalar_apply(op, prm, xf), _MM_FROUND_TO_NEAREST_INT);
    }
}

__attribute__((target("avx2,fma"), noinline))
static void bf16_chunk_avx2(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    const __m256i shift = _mm256_set1_epi32(16);
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        __m256 y = apply_f32(op, prm, _mm256_castsi256_ps(w));
        // Round-to-nearest-even back to bf16, matching BFloat16(float).
        __m256i u = _mm256_castps_si256(y);
        __m256i rounding = _mm256_add_epi32(
            _mm256_set1_epi32(0x7FFFu),
            _mm256_srli_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x10000u)), 16));
        __m256i r = _mm256_srli_epi32(_mm256_add_epi32(u, rounding), 16);
        alignas(32) uint32_t lanes[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), r);
        for (int64_t j = 0; j < 8; ++j) dst[i + j] = static_cast<uint16_t>(lanes[j]);
    }
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

#endif // TP_VECUNARY_LIBMVEC

// ---------------------------------------------------------------------------
// Public entry points: process [b, e) of contiguous src -> dst.
// ---------------------------------------------------------------------------
inline void run_f32(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
#ifdef TP_VECUNARY_LIBMVEC
    if (avx2_available()) {
        f32_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}

inline void run_f64(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
#ifdef TP_VECUNARY_LIBMVEC
    if (avx2_available()) {
        f64_chunk_avx2(op, prm, src, dst, b, e);
        return;
    }
#endif
    for (int64_t i = b; i < e; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}

inline bool vec_ready() {
#ifdef TP_VECUNARY_LIBMVEC
    return avx2_available();
#else
    return false;
#endif
}

// fp16 path: widen to f32 (FMA/F16C), run the f32 kernels, narrow back with
// round-to-nearest-even — same semantics as tensorplay::Half(float).
// Scalar fallback goes through the Half type itself.
inline void run_f16(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
#ifdef TP_VECUNARY_LIBMVEC
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
#ifdef TP_VECUNARY_LIBMVEC
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
