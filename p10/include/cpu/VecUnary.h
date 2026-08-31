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

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "Half.h"
#include "BFloat16.h"

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

// f64 kernels whose scalar reference rounds intermediates through float
// (double(float(x)) games); they stay on the AVX2 path which already
// reproduces those semantics lane-for-lane.
inline bool f64_rounding_sensitive(VOp op) {
    return op == VOp::Elu || op == VOp::Softplus || op == VOp::LeakyRelu ||
           op == VOp::Celu || op == VOp::Hardswish || op == VOp::Hardsigmoid;
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
#ifdef TP_VECUNARY_SLEEF

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
                        tensorplay::tpsleef::log(t), _mm256_set1_ps(static_cast<float>(kHalfLogTwoPi))),
        t);
    res = _mm256_add_ps(res, tensorplay::tpsleef::log(series));
    return _mm256_sub_ps(res, _mm256_and_ps(small, tensorplay::tpsleef::log(x)));
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
        case VOp::Lgamma: {
            // Callers guarantee all lanes > 0 here.
            return v_lgamma_pos_f32(x);
        }
        case VOp::Sigmoid: {
            // rcp+NR instead of divps.  Division corners reproduced: den==inf
            // (x <= -88.7, exp overflowed) must yield +0 like 1/inf, and a NaN
            // denominator must stay NaN -- hence the two mask repairs.
            __m256 den = _mm256_add_ps(one, tensorplay::tpsleef::exp(_mm256_xor_ps(x, signbit)));
            __m256 r = v_rcp_nr_ps(den);
            __m256 not_inf = _mm256_cmp_ps(den, _mm256_set1_ps(INFINITY), _CMP_NEQ_OQ);
            __m256 zeroed = _mm256_and_ps(not_inf, r);
            __m256 is_nan = _mm256_cmp_ps(den, den, _CMP_UNORD_Q);
            return _mm256_blendv_ps(zeroed, r, is_nan);
        }
        case VOp::GeluNone: {
            const __m256 kAlpha = _mm256_set1_ps(static_cast<float>(0.70710678118654752440));
            __m256 cdf = _mm256_add_ps(one, tensorplay::tpsleef::erf(_mm256_mul_ps(kAlpha, x)));
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
        }
        case VOp::GeluTanh: {
            const __m256 kBeta = _mm256_set1_ps(static_cast<float>(1.41421356237309504880 * 1.12837916709551257390 * 0.5));
            const __m256 kKappa = _mm256_set1_ps(0.044715f);
            __m256 x_cube = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
            __m256 inner = _mm256_mul_ps(kBeta, _mm256_add_ps(x, _mm256_mul_ps(kKappa, x_cube)));
            __m256 cdf = _mm256_add_ps(one, tensorplay::tpsleef::tanh(inner));
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), cdf);
        }
        case VOp::Silu: {
            // x * rcp_nr(den) matches x/den bit-for-bit in every corner here:
            // den==inf makes the NR chain NaN exactly like -inf/+inf division,
            // and finite dens get a float-exact reciprocal.
            __m256 den = _mm256_add_ps(one, tensorplay::tpsleef::exp(_mm256_xor_ps(x, signbit)));
            return _mm256_mul_ps(x, v_rcp_nr_ps(den));
        }
        case VOp::Mish: {
            // Uses the scalar expression: log((1 + exp(x))) * tanh(...) — plain
            // log, not log1p, and overflow-to-inf semantics are preserved.
            __m256 sp = tensorplay::tpsleef::log(_mm256_add_ps(one, tensorplay::tpsleef::exp(x)));
            return _mm256_mul_ps(x, tensorplay::tpsleef::tanh(sp));
        }
        case VOp::Selu: {
            const __m256 lambda = _mm256_set1_ps(static_cast<float>(1.0507009873554804934193349852946));
            const __m256 alphalambda = _mm256_set1_ps(static_cast<float>(1.6732632423543772848170429916717 * 1.0507009873554804934193349852946));
            __m256 pos = _mm256_mul_ps(x, lambda);
            __m256 neg = _mm256_mul_ps(alphalambda, tensorplay::tpsleef::expm1(x));
            return _mm256_blendv_ps(neg, pos, _mm256_cmp_ps(x, zero, _CMP_GT_OQ));
        }
        case VOp::Elu: {
            const __m256 negcoef = _mm256_set1_ps(static_cast<float>(prm.p0)); // alpha*scale
            const __m256 poscoef = _mm256_set1_ps(static_cast<float>(prm.p1)); // scale
            const __m256 negipt = _mm256_set1_ps(static_cast<float>(prm.p2));  // input_scale
            __m256 scaled = _mm256_mul_ps(x, negipt);
            __m256 neg = _mm256_mul_ps(tensorplay::tpsleef::expm1(scaled), negcoef);
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
            __m256 numf = tensorplay::tpsleef::log1p(tensorplay::tpsleef::exp(bx));
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
            __m256 neg = _mm256_mul_ps(a, tensorplay::tpsleef::expm1(_mm256_div_ps(x, a)));
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
        case VOp::Elu: {
            const __m256d negcoef = _mm256_set1_pd(prm.p0);
            const __m256d poscoef = _mm256_set1_pd(prm.p1);
            const __m256d negipt = _mm256_set1_pd(prm.p2);
            // Scalar f64 kernel: a = double(float(x)); expm1(float(a)*float(ipt))
            // in double, times float(negcoef); positive branch a*poscoef.
            __m256d xf = _mm256_cvtps_pd(_mm256_cvtpd_ps(x));
            __m256d scaled = _mm256_mul_pd(xf, _mm256_cvtps_pd(_mm256_cvtpd_ps(negipt)));
            __m256d neg = _mm256_mul_pd(tensorplay::tpsleef::expm1(scaled),
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
                _mm256_castps256_ps128(tensorplay::tpsleef::log1p(tensorplay::tpsleef::exp(btf8))));
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
            __m256d neg = _mm256_mul_pd(a, tensorplay::tpsleef::expm1(_mm256_div_pd(af, a)));
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

// Softplus: numerator stays float; the /beta happens in double with one
// final round back to float, matching scalar_apply's mixed precision.
__attribute__((target("avx512f,avx512dq"), always_inline))
inline __m256 softplus_div_beta_8(__m256 v8, double beta) {
    const __m512d beta_d8 = _mm512_set1_pd(beta);
    __m512d w8 = _mm512_div_pd(_mm512_cvtps_pd(v8), beta_d8);
    __m128 flo = _mm256_cvtpd_ps(_mm512_castpd512_pd256(w8));
    __m128 fhi = _mm256_cvtpd_ps(_mm512_extractf64x4_pd(w8, 1));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(flo), fhi, 1);
}

// rcp14 seeds at <=2^-14; one NR step reaches ~2^-28, i.e. float-exact for
// the sigmoid/silu denominators seen here.
__attribute__((target("avx512f"), always_inline))
inline __m512 v_rcp_nr_ps(__m512 d) {
    __m512 r = _mm512_rcp14_ps(d);
    return _mm512_mul_ps(r, _mm512_fnmadd_ps(d, r, _mm512_set1_ps(2.0f)));
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
        case VOp::Erf: return tensorplay::tpsleef::erf(x);
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
            const __m512 z = _mm512_mul_ps(kAlpha, x);
            const __m512 az = _mm512_andnot_ps(signbit, z);
            const __m512 t = v_rcp_nr_ps(
                _mm512_add_ps(one, _mm512_mul_ps(_mm512_set1_ps(0.3275911f), az)));
            __m512 poly = _mm512_add_ps(_mm512_mul_ps(
                _mm512_set1_ps(1.061405429f), t),
                _mm512_set1_ps(-1.453152027f));
            poly = _mm512_add_ps(_mm512_mul_ps(poly, t),
                                 _mm512_set1_ps(1.421413741f));
            poly = _mm512_add_ps(_mm512_mul_ps(poly, t),
                                 _mm512_set1_ps(-0.284496736f));
            poly = _mm512_add_ps(_mm512_mul_ps(poly, t),
                                 _mm512_set1_ps(0.254829592f));
            poly = _mm512_mul_ps(poly, t);
            const __m512 erf_abs = _mm512_sub_ps(
                one, _mm512_mul_ps(poly,
                    tensorplay::tpsleef::exp(_mm512_sub_ps(_mm512_setzero_ps(),
                                                  _mm512_mul_ps(az, az)))));
            const __m512 erf = _mm512_mask_mov_ps(
                erf_abs, _mm512_cmp_ps_mask(z, zero, _CMP_LT_OQ),
                _mm512_sub_ps(zero, erf_abs));
            __m512 cdf = _mm512_add_ps(one, erf);
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
        case VOp::Elu: {
            const __m512 negcoef = _mm512_set1_ps(static_cast<float>(prm.p0));
            const __m512 poscoef = _mm512_set1_ps(prm.p1);
            const __m512 negipt = _mm512_set1_ps(prm.p2);
            __m512 scaled = _mm512_mul_ps(x, negipt);
            __m512 neg = _mm512_mul_ps(tensorplay::tpsleef::expm1(scaled), negcoef);
            __m512 pos = _mm512_mul_ps(x, poscoef);
            return _mm512_mask_mov_ps(neg, _mm512_cmp_ps_mask(x, zero, _CMP_GE_OQ), pos);
        }
        case VOp::Softplus: {
            // Scalar path: numerator computed in float (log1p(exp(float(x*beta)))),
            // division by beta in double, one final round back to T.
            const __m512 beta = _mm512_set1_ps(static_cast<float>(prm.p0));
            const __m512 threshold = _mm512_set1_ps(static_cast<float>(prm.p1));
            __m512 bx = _mm512_mul_ps(x, beta);
            __m512 numf = tensorplay::tpsleef::log1p(tensorplay::tpsleef::exp(bx));
            __m256 sp_lo = softplus_div_beta_8(_mm512_castps512_ps256(numf), prm.p0);
            __m256 sp_hi = softplus_div_beta_8(_mm512_extractf32x8_ps(numf, 1), prm.p0);
            __m512 sp = _mm512_insertf32x8(_mm512_castps256_ps512(sp_lo), sp_hi, 1);
            return _mm512_mask_mov_ps(sp, _mm512_cmp_ps_mask(bx, threshold, _CMP_GT_OQ), x);
        }
        case VOp::Hardswish: {
            const __m512 three = _mm512_set1_ps(3.0f);
            const __m512 six = _mm512_set1_ps(6.0f);
            __m512 y = _mm512_add_ps(x, three);
            y = _mm512_min_ps(_mm512_max_ps(y, zero), six);
            return _mm512_div_ps(_mm512_mul_ps(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m512 three = _mm512_set1_ps(3.0f);
            const __m512 six = _mm512_set1_ps(6.0f);
            __m512 y = _mm512_add_ps(x, three);
            y = _mm512_min_ps(_mm512_max_ps(y, zero), six);
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
            return _mm512_min_ps(_mm512_max_ps(x, lo), hi);
        }
        case VOp::Relu6:
            return _mm512_min_ps(_mm512_max_ps(x, zero), _mm512_set1_ps(6.0f));
        case VOp::Celu: {
            const __m512 a = _mm512_set1_ps(static_cast<float>(prm.p0));
            __m512 neg = _mm512_mul_ps(a, tensorplay::tpsleef::expm1(_mm512_div_ps(x, a)));
            __m512 pos = _mm512_max_ps(zero, x);
            __m512 minneg = _mm512_min_ps(neg, zero);
            return _mm512_add_ps(pos, minneg);
        }
        default: return x;
    }
}

__attribute__((target("avx512f,avx512dq"), noinline))
static void f32_chunk_avx512(VOp op, VParams prm, const float* src, float* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(src + i);
        if (op == VOp::Lgamma && _mm512_cmplt_ps_mask(x, _mm512_setzero_ps())) {
            for (int64_t j = i; j < i + 16; ++j) dst[j] = scalar_apply(op, prm, src[j]);
        } else if (op == VOp::Square) {
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(x, x));
        } else if (op == VOp::GeluNone) {
            const __m512 z = _mm512_mul_ps(
                _mm512_set1_ps(0.7071067811865476f), x);
            const __m512 az = _mm512_andnot_ps(
                _mm512_set1_ps(kSignBitF512), z);
            const __mmask16 core = _mm512_cmp_ps_mask(
                az, _mm512_set1_ps(1.5f), _CMP_LE_OQ);
            if (core == static_cast<__mmask16>(0xffff)) {
                _mm512_storeu_ps(dst + i, gelu_poly_f32_512(x));
            } else {
                _mm512_storeu_ps(dst + i, apply16_f32(op, prm, x));
            }
        } else {
            _mm512_storeu_ps(dst + i, apply16_f32(op, prm, x));
        }
    }
    for (; i < n; ++i) dst[i] = scalar_apply(op, prm, src[i]);
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
            y = _mm512_min_pd(_mm512_max_pd(y, zero), six);
            return _mm512_div_pd(_mm512_mul_pd(x, y), six);
        }
        case VOp::Hardsigmoid: {
            const __m512d three = _mm512_set1_pd(3.0);
            const __m512d six = _mm512_set1_pd(6.0);
            __m512d y = _mm512_add_pd(x, three);
            y = _mm512_min_pd(_mm512_max_pd(y, zero), six);
            return _mm512_div_pd(y, six);
        }
        case VOp::Hardtanh: {
            const __m512d lo = _mm512_set1_pd(prm.p0);
            const __m512d hi = _mm512_set1_pd(prm.p1);
            return _mm512_min_pd(_mm512_max_pd(x, lo), hi);
        }
        case VOp::Relu6:
            return _mm512_min_pd(_mm512_max_pd(x, zero), _mm512_set1_pd(6.0));
        default: return x;
    }
}

__attribute__((target("avx512f,avx512dq"), noinline))
static void f64_chunk_avx512(VOp op, VParams prm, const double* src, double* dst, int64_t b, int64_t e) {
    src += b;
    dst += b;
    int64_t n = e - b;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d x = _mm512_loadu_pd(src + i);
        if (op == VOp::Lgamma && _mm512_cmplt_pd_mask(x, _mm512_setzero_pd())) {
            for (int64_t j = i; j < i + 8; ++j) dst[j] = scalar_apply(op, prm, src[j]);
        } else if (op == VOp::Square) {
            _mm512_storeu_pd(dst + i, _mm512_mul_pd(x, x));
        } else {
            _mm512_storeu_pd(dst + i, apply16_f64(op, prm, x));
        }
    }
    for (; i < n; ++i) dst[i] = scalar_apply(op, prm, src[i]);
}

__attribute__((target("avx2,fma"), noinline))
static void bf16_chunk_avx2(VOp op, VParams prm, const uint16_t* src, uint16_t* dst, int64_t b, int64_t e) {
    const __m256i shift = _mm256_set1_epi32(16);
    (void)shift;
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
    // Tier-compiled copy (see run_f32).  The f64_rounding_sensitive ops keep
    // their AVX2-only contract: route them through the scalar loop here the
    // same way the runtime path would.
    if (!f64_rounding_sensitive(op)) {
        f64_chunk_avx512(op, prm, src, dst, b, e);
        return;
    }
    f64_chunk_avx2(op, prm, src, dst, b, e);
    return;
#endif
    // Elu/Softplus/LeakyRelu/Celu keep double(float(x)) rounding semantics
    // that only the AVX2 kernels reproduce; everything else goes 512-bit.
    if (avx512_available() && !f64_rounding_sensitive(op)) {
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
