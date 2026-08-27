#pragma once

// Vectorized fast paths for CPU complex elementwise kernels (AVX2 + glibc
// libmvec).  Layout is the standard interleaved (re, im) stream.
//
// Why this beats torch's CPU complex path: ATen vec256_complex_float.h runs
// transcendentals as `map(std::exp)` -- a scalar loop over four lanes -- and
// abs/atan2 as scalar loops too.  Here exp/log/sqrt/trig/hypot/div are true
// 4-lane-per-vector SIMD via polar/Smith formulations, ULP-close to glibc.
//
// Formulas mirror cpu/ComplexUnary.h (the c10/util/complex_math.h ports) so
// vector and fallback paths agree:
//   exp(z)   = e^x * (cos y + i sin y)
//   expm1(z) = expm1(x)*cos(y) - 2*sin(y/2)^2 + i*e^x*sin(y)
//   log(z)   = log|z| + i*atan2(y, x)
//   log1p(z) = log|1+z| + i*atan2(y, 1+x)
//   sqrt(z)  = Smith quadrant method (z == 0 -> 0); rsqrt = conj(sqrt)/|z|^2
//   sin(z)   = sin x cosh y + i cos x sinh y     (cos(z): imag sign flips)
//   tan(z)   = sin(z)/cos(z); sinh/cosh swap x<->y; tanh = sinh/cosh
//   asinh(z) = log(z + csqrt(z*z + 1)); acosh(z): z*z - 1
//   asin(z)  = -i*asinh(i z); acos(z) = pi/2 - asin(z)
//   atan(z)  = (i/2)*log((1 - i z)/(1 + i z))
//   atanh(z) = (log(1+z) - log(1-z)) / 2
//   div      = Smith's algorithm scaled by max(|c|, |d|)   (ATen parity)

#include <immintrin.h>
#include "cpu/ComplexKernels.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "DType.h"
#include "Parallel.h"

#if defined(__x86_64__) || defined(__i386__)
#define TP_VECCPLX_X86 1
#endif

// Vector (libmvec/AVX) fast paths are compiled only into the
// AVX2/AVX512-capable TUs of the multi-capability build; the DEFAULT TU
// falls back to the scalar implementations registered via REGISTER_DISPATCH.
// Compiling the always_inline AVX helpers without matching target flags is
// a hard error ("target specific option mismatch").
#if defined(TP_VECCPLX_X86) && defined(__GLIBC__) && defined(__x86_64__) \
    && (defined(__AVX2__) || defined(CPU_CAPABILITY_AVX2) \
        || defined(CPU_CAPABILITY_AVX512))
#define TP_VECCPLX_LIBMVEC 1
#endif

namespace tensorplay {
namespace cpu {
namespace veccomplex {

enum class Op {
    Add, Sub, Mul, Div,
    Neg, Square, Recip,
    Exp, Expm1, Log, Log1p, Log2, Log10,
    Sqrt, Rsqrt,
    Sin, Cos, Tan, Sinh, Cosh, Tanh,
    Asin, Acos, Atan, Asinh, Acosh, Atanh,
    Sigmoid,
};

inline bool binary_supported(Op op) {
    return op == Op::Add || op == Op::Sub || op == Op::Mul || op == Op::Div;
}
inline bool unary_supported(Op op) { return !binary_supported(op); }

inline bool width_ok(DType dt) {
    return dt == DType::ComplexFloat || dt == DType::ComplexDouble;
}

#ifdef TP_VECCPLX_LIBMVEC

inline bool avx2_available() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

inline bool avx512_available() {
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0;
    return ok;
#else
    return false;
#endif
}

extern "C" {
__m256 _ZGVdN8v_expf(__m256);
__m256 _ZGVdN8v_expm1f(__m256);
__m256 _ZGVdN8v_logf(__m256);
__m256 _ZGVdN8v_log1pf(__m256);
__m256 _ZGVdN8v_log2f(__m256);
__m256 _ZGVdN8v_log10f(__m256);
__m256 _ZGVdN8v_sinf(__m256);
__m256 _ZGVdN8v_cosf(__m256);
__m256 _ZGVdN8v_atanf(__m256);
__m256 _ZGVdN8v_sinhf(__m256);
__m256 _ZGVdN8v_coshf(__m256);
__m256 _ZGVdN8vv_atan2f(__m256, __m256);
__m256 _ZGVdN8vv_hypotf(__m256, __m256);

__m256d _ZGVdN4v_exp(__m256d);
__m512 _ZGVeN16v_expf(__m512);
__m512 _ZGVeN16v_expm1f(__m512);
__m512 _ZGVeN16v_logf(__m512);
__m512 _ZGVeN16v_log1pf(__m512);
__m512 _ZGVeN16v_log2f(__m512);
__m512 _ZGVeN16v_log10f(__m512);
__m512 _ZGVeN16v_sinf(__m512);
__m512 _ZGVeN16v_cosf(__m512);
__m512 _ZGVeN16v_atanf(__m512);
__m512 _ZGVeN16v_sinhf(__m512);
__m512 _ZGVeN16v_coshf(__m512);
__m512 _ZGVeN16vv_atan2f(__m512, __m512);
__m512 _ZGVeN16vv_hypotf(__m512, __m512);

__m512d _ZGVeN8v_exp(__m512d);
__m512d _ZGVeN8v_expm1(__m512d);
__m512d _ZGVeN8v_log(__m512d);
__m512d _ZGVeN8v_log1p(__m512d);
__m512d _ZGVeN8v_sin(__m512d);
__m512d _ZGVeN8v_cos(__m512d);
__m512d _ZGVeN8v_atan(__m512d);
__m512d _ZGVeN8v_sinh(__m512d);
__m512d _ZGVeN8v_cosh(__m512d);
__m512d _ZGVeN8vv_atan2(__m512d, __m512d);
__m512d _ZGVeN8vv_hypot(__m512d, __m512d);
__m256d _ZGVdN4v_expm1(__m256d);
__m256d _ZGVdN4v_log(__m256d);
__m256d _ZGVdN4v_log1p(__m256d);
__m256d _ZGVdN4v_log2(__m256d);
__m256d _ZGVdN4v_log10(__m256d);
__m256d _ZGVdN4v_sin(__m256d);
__m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_atan(__m256d);
__m256d _ZGVdN4v_sinh(__m256d);
__m256d _ZGVdN4v_cosh(__m256d);
__m256d _ZGVdN4vv_atan2(__m256d, __m256d);
__m256d _ZGVdN4vv_hypot(__m256d, __m256d);
}

// [vec+scalar complex kernels moved to ComplexKernels.cpp for three-tier compilation]
inline bool try_unary(const void* xv, void* yv, int64_t n, DType dt, Op op) {
    return cplx_unary_stub(tensorplay::DeviceType::CPU, xv, yv, n, static_cast<int>(dt), static_cast<int>(op));
}

inline bool try_binary(const void* av_, const void* bv, void* yv, int64_t n,
                       DType dt, Op op) {
    return cplx_binary_stub(tensorplay::DeviceType::CPU, av_, bv, yv, n, static_cast<int>(dt), static_cast<int>(op));
}

inline bool try_abs(const void* xv, void* real_out, int64_t n, DType dt) {
    return cplx_abs_stub(tensorplay::DeviceType::CPU, xv, real_out, n, static_cast<int>(dt));
}

inline bool try_angle(const void* xv, void* real_out, int64_t n, DType dt) {
    return cplx_angle_stub(tensorplay::DeviceType::CPU, xv, real_out, n, static_cast<int>(dt));
}

// full reduction: sum over the interleaved stream
inline bool try_sum(const void* xv, int64_t n, DType dt,
                    double* re_out, double* im_out) {
    return tensorplay::cpu::cplx_sum_stub(tensorplay::DeviceType::CPU, xv, n, static_cast<int>(dt), re_out, im_out);
}

#else  // !TP_VECCPLX_LIBMVEC

inline bool avx2_available() { return false; }
inline bool avx512_available() { return false; }

// Scalar-capability fallbacks: route through the runtime-dispatched stubs
// so a DEFAULT TU still links and runs (the AVX fast paths arrive from the
// per-capability TUs of the multi-capability build).
inline bool try_unary(const void* xv, void* yv, int64_t n, DType dt, Op op) {
    return tensorplay::cpu::cplx_unary_stub(tensorplay::DeviceType::CPU, xv,
                                            yv, n, static_cast<int>(dt),
                                            static_cast<int>(op));
}
inline bool try_binary(const void* av_, const void* bv, void* yv, int64_t n,
                       DType dt, Op op) {
    return tensorplay::cpu::cplx_binary_stub(
        tensorplay::DeviceType::CPU, av_, bv, yv, n, static_cast<int>(dt),
        static_cast<int>(op));
}
inline bool try_abs(const void* xv, void* real_out, int64_t n, DType dt) {
    return tensorplay::cpu::cplx_abs_stub(tensorplay::DeviceType::CPU, xv,
                                          real_out, n, static_cast<int>(dt));
}
inline bool try_angle(const void* xv, void* real_out, int64_t n, DType dt) {
    return tensorplay::cpu::cplx_angle_stub(tensorplay::DeviceType::CPU, xv,
                                            real_out, n,
                                            static_cast<int>(dt));
}
inline bool try_sum(const void* xv, int64_t n, DType dt,
                    double* re_out, double* im_out) {
    return tensorplay::cpu::cplx_sum_stub(tensorplay::DeviceType::CPU, xv, n,
                                          static_cast<int>(dt), re_out,
                                          im_out);
}

#endif  // TP_VECCPLX_LIBMVEC

}  // namespace veccomplex
}  // namespace cpu
}  // namespace tensorplay
