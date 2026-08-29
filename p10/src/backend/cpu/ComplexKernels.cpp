#if defined(__AVX512F__)
#define TP_VECCPLX_AVX512 1
#endif

#include "cpu/ComplexKernels.h"
#include "cpu/VecComplex.h"
#include "Parallel.h"

#include <vector>

namespace cplx_base = ::tensorplay::cpu::veccomplex;

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

namespace cplxk {

using cplx_base::Op;
using cplx_base::width_ok;
using cplx_base::binary_supported;
using cplx_base::avx2_available;
using cplx_base::avx512_available;

constexpr int64_t kGrain = 4096;

// The DEFAULT copy is deliberately scalar-only.  Keeping the ISA-specific
// definitions behind this guard is important: merely mentioning an AVX
// intrinsic in a non-targeted TU makes GCC emit a target-option mismatch even
// when the template is never instantiated.
#ifdef TP_VECCPLX_LIBMVEC

extern "C" {
__m256 _ZGVdN8v_expf(__m256); __m256 _ZGVdN8v_expm1f(__m256);
__m256 _ZGVdN8v_logf(__m256); __m256 _ZGVdN8v_log1pf(__m256);
__m256 _ZGVdN8v_log2f(__m256); __m256 _ZGVdN8v_log10f(__m256);
__m256 _ZGVdN8v_sinf(__m256); __m256 _ZGVdN8v_cosf(__m256);
__m256 _ZGVdN8v_atanf(__m256); __m256 _ZGVdN8v_sinhf(__m256);
__m256 _ZGVdN8v_coshf(__m256); __m256 _ZGVdN8vv_atan2f(__m256, __m256);
__m256 _ZGVdN8vv_hypotf(__m256, __m256);
__m512 _ZGVeN16v_expf(__m512); __m512 _ZGVeN16v_expm1f(__m512);
__m512 _ZGVeN16v_logf(__m512); __m512 _ZGVeN16v_log1pf(__m512);
__m512 _ZGVeN16v_sinf(__m512); __m512 _ZGVeN16v_cosf(__m512);
__m512 _ZGVeN16v_atanf(__m512); __m512 _ZGVeN16v_sinhf(__m512);
__m512 _ZGVeN16v_coshf(__m512);
__m512 _ZGVeN16vv_atan2f(__m512, __m512);
__m512 _ZGVeN16vv_hypotf(__m512, __m512);
__m256d _ZGVdN4v_exp(__m256d); __m256d _ZGVdN4v_expm1(__m256d);
__m256d _ZGVdN4v_log(__m256d); __m256d _ZGVdN4v_log1p(__m256d);
__m256d _ZGVdN4v_sin(__m256d); __m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_atan(__m256d); __m256d _ZGVdN4v_sinh(__m256d);
__m256d _ZGVdN4v_cosh(__m256d); __m256d _ZGVdN4vv_atan2(__m256d, __m256d);
__m256d _ZGVdN4vv_hypot(__m256d, __m256d);
__m512d _ZGVeN8v_exp(__m512d); __m512d _ZGVeN8v_expm1(__m512d);
__m512d _ZGVeN8v_log(__m512d); __m512d _ZGVeN8v_log1p(__m512d);
__m512d _ZGVeN8v_sin(__m512d); __m512d _ZGVeN8v_cos(__m512d);
__m512d _ZGVeN8v_atan(__m512d); __m512d _ZGVeN8v_sinh(__m512d);
__m512d _ZGVeN8v_cosh(__m512d);
__m512d _ZGVeN8vv_atan2(__m512d, __m512d);
__m512d _ZGVeN8vv_hypot(__m512d, __m512d);
}

template <typename V> struct Math;
template <> struct Math<__m256> {
    using scalar = float;
    static constexpr int W = 4;  // complex elements per vector
    static __m256 exp(__m256 v) { return _ZGVdN8v_expf(v); }
    static __m256 expm1(__m256 v) { return _ZGVdN8v_expm1f(v); }
    static __m256 log(__m256 v) { return _ZGVdN8v_logf(v); }
    static __m256 log1p(__m256 v) { return _ZGVdN8v_log1pf(v); }
    static __m256 sin(__m256 v) { return _ZGVdN8v_sinf(v); }
    static __m256 cos(__m256 v) { return _ZGVdN8v_cosf(v); }
    static __m256 atan(__m256 v) { return _ZGVdN8v_atanf(v); }
    static __m256 atan2(__m256 y, __m256 x) { return _ZGVdN8vv_atan2f(y, x); }
    static __m256 sinh(__m256 v) { return _ZGVdN8v_sinhf(v); }
    static __m256 cosh(__m256 v) { return _ZGVdN8v_coshf(v); }
    static __m256 hypot(__m256 y, __m256 x) { return _ZGVdN8vv_hypotf(y, x); }
    static __m256 ln2() { return _mm256_set1_ps(0.69314718055994530942f); }
    static __m256 ln10() { return _mm256_set1_ps(2.30258509299404568402f); }
};
template <> struct Math<__m256d> {
    using scalar = double;
    static constexpr int W = 2;
    static __m256d exp(__m256d v) { return _ZGVdN4v_exp(v); }
    static __m256d expm1(__m256d v) { return _ZGVdN4v_expm1(v); }
    static __m256d log(__m256d v) { return _ZGVdN4v_log(v); }
    static __m256d log1p(__m256d v) { return _ZGVdN4v_log1p(v); }
    static __m256d sin(__m256d v) { return _ZGVdN4v_sin(v); }
    static __m256d cos(__m256d v) { return _ZGVdN4v_cos(v); }
    static __m256d atan(__m256d v) { return _ZGVdN4v_atan(v); }
    static __m256d atan2(__m256d y, __m256d x) { return _ZGVdN4vv_atan2(y, x); }
    static __m256d sinh(__m256d v) { return _ZGVdN4v_sinh(v); }
    static __m256d cosh(__m256d v) { return _ZGVdN4v_cosh(v); }
    static __m256d hypot(__m256d y, __m256d x) { return _ZGVdN4vv_hypot(y, x); }
    static __m256d ln2() { return _mm256_set1_pd(0.69314718055994530942); }
    static __m256d ln10() { return _mm256_set1_pd(2.30258509299404568402); }
};

#if defined(TP_VECCPLX_AVX512)
template <> struct Math<__m512> {
    using scalar = float;
    static constexpr int W = 8;
    static __m512 exp(__m512 v) { return _ZGVeN16v_expf(v); }
    static __m512 expm1(__m512 v) { return _ZGVeN16v_expm1f(v); }
    static __m512 log(__m512 v) { return _ZGVeN16v_logf(v); }
    static __m512 log1p(__m512 v) { return _ZGVeN16v_log1pf(v); }
    static __m512 sin(__m512 v) { return _ZGVeN16v_sinf(v); }
    static __m512 cos(__m512 v) { return _ZGVeN16v_cosf(v); }
    static __m512 atan(__m512 v) { return _ZGVeN16v_atanf(v); }
    static __m512 atan2(__m512 y, __m512 x) { return _ZGVeN16vv_atan2f(y, x); }
    static __m512 sinh(__m512 v) { return _ZGVeN16v_sinhf(v); }
    static __m512 cosh(__m512 v) { return _ZGVeN16v_coshf(v); }
    static __m512 hypot(__m512 y, __m512 x) { return _ZGVeN16vv_hypotf(y, x); }
    static __m512 ln2() { return _mm512_set1_ps(0.69314718055994530942f); }
    static __m512 ln10() { return _mm512_set1_ps(2.30258509299404568402f); }
};
template <> struct Math<__m512d> {
    using scalar = double;
    static constexpr int W = 4;
    static __m512d exp(__m512d v) { return _ZGVeN8v_exp(v); }
    static __m512d expm1(__m512d v) { return _ZGVeN8v_expm1(v); }
    static __m512d log(__m512d v) { return _ZGVeN8v_log(v); }
    static __m512d log1p(__m512d v) { return _ZGVeN8v_log1p(v); }
    static __m512d sin(__m512d v) { return _ZGVeN8v_sin(v); }
    static __m512d cos(__m512d v) { return _ZGVeN8v_cos(v); }
    static __m512d atan(__m512d v) { return _ZGVeN8v_atan(v); }
    static __m512d atan2(__m512d y, __m512d x) { return _ZGVeN8vv_atan2(y, x); }
    static __m512d sinh(__m512d v) { return _ZGVeN8v_sinh(v); }
    static __m512d cosh(__m512d v) { return _ZGVeN8v_cosh(v); }
    static __m512d hypot(__m512d y, __m512d x) { return _ZGVeN8vv_hypot(y, x); }
    static __m512d ln2() { return _mm512_set1_pd(0.69314718055994530942); }
    static __m512d ln10() { return _mm512_set1_pd(2.30258509299404568402); }
};
#endif // TP_VECCPLX_AVX512


// Primitive table: every intrinsic is wrapped here so the kernel bodies stay
// type-generic (no raw _mm256_*ps leaking into double instantiations).
template <typename V> struct Ops;
template <> struct Ops<__m256> {
    static void store(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
    static __m256 set1(float v) { return _mm256_set1_ps(v); }
    static __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
    static __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    static __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    static __m256 div(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }
    static __m256 fma(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
    static __m256 max(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
    static __m256 sqrt(__m256 a) { return _mm256_sqrt_ps(a); }
    static __m256 zero() { return _mm256_setzero_ps(); }
    static __m256 one() { return _mm256_set1_ps(1.0f); }
    static __m256 half() { return _mm256_set1_ps(0.5f); }
    static __m256 sgnmask() { return _mm256_set1_ps(-0.0f); }
    static __m256 blend(__m256 a, __m256 b, __m256 m) { return _mm256_blendv_ps(a, b, m); }
    static __m256 cmp_lt(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
    static __m256 cmp_eq(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static __m256 bit_and(__m256 a, __m256 b) { return _mm256_and_ps(a, b); }
    static __m256 bit_andnot(__m256 m, __m256 a) { return _mm256_andnot_ps(m, a); }
    static __m256 bit_xor(__m256 a, __m256 b) { return _mm256_xor_ps(a, b); }
};
template <> struct Ops<__m256d> {
    static void store(double* p, __m256d v) { _mm256_storeu_pd(p, v); }
    static __m256d set1(double v) { return _mm256_set1_pd(v); }
    static __m256d add(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
    static __m256d sub(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
    static __m256d mul(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
    static __m256d div(__m256d a, __m256d b) { return _mm256_div_pd(a, b); }
    static __m256d fma(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
    static __m256d max(__m256d a, __m256d b) { return _mm256_max_pd(a, b); }
    static __m256d sqrt(__m256d a) { return _mm256_sqrt_pd(a); }
    static __m256d zero() { return _mm256_setzero_pd(); }
    static __m256d one() { return _mm256_set1_pd(1.0); }
    static __m256d half() { return _mm256_set1_pd(0.5); }
    static __m256d sgnmask() { return _mm256_set1_pd(-0.0); }
    static __m256d blend(__m256d a, __m256d b, __m256d m) { return _mm256_blendv_pd(a, b, m); }
    static __m256d cmp_lt(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
    static __m256d cmp_eq(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static __m256d bit_and(__m256d a, __m256d b) { return _mm256_and_pd(a, b); }
    static __m256d bit_andnot(__m256d m, __m256d a) { return _mm256_andnot_pd(m, a); }
    static __m256d bit_xor(__m256d a, __m256d b) { return _mm256_xor_pd(a, b); }
};

#if defined(TP_VECCPLX_AVX512)
template <> struct Ops<__m512> {
    static __m512 add(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
    static __m512 sub(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
    static __m512 mul(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }
    static __m512 div(__m512 a, __m512 b) { return _mm512_div_ps(a, b); }
    static __m512 fma(__m512 a, __m512 b, __m512 c) { return _mm512_fmadd_ps(a, b, c); }
    static __m512 max(__m512 a, __m512 b) { return _mm512_max_ps(a, b); }
    static __m512 sqrt(__m512 a) { return _mm512_sqrt_ps(a); }
    static __m512 zero() { return _mm512_setzero_ps(); }
    static __m512 one() { return _mm512_set1_ps(1.0f); }
    static __m512 half() { return _mm512_set1_ps(0.5f); }
    static __m512 sgnmask() { return _mm512_set1_ps(-0.0f); }
    static __m512 blend(__m512 a, __m512 b, __m512 m) { return _mm512_mask_blend_ps(_mm512_movepi32_mask(_mm512_castps_si512(m)), a, b); }
    static __m512 cmp_lt(__m512 a, __m512 b) {
        return _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ), _mm512_set1_epi32(-1)));
    }
    static __m512 cmp_eq(__m512 a, __m512 b) {
        return _mm512_castsi512_ps(_mm512_maskz_mov_epi32(
            _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ), _mm512_set1_epi32(-1)));
    }
    static __m512 bit_and(__m512 a, __m512 b) { return _mm512_and_ps(a, b); }
    static __m512 bit_andnot(__m512 m, __m512 a) { return _mm512_andnot_ps(m, a); }
    static __m512 bit_xor(__m512 a, __m512 b) { return _mm512_xor_ps(a, b); }
    static void store(float* p, __m512 v) { _mm512_storeu_ps(p, v); }
    static __m512 set1(float v) { return _mm512_set1_ps(v); }
};

template <> struct Ops<__m512d> {
    static __m512d add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
    static __m512d sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
    static __m512d mul(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }
    static __m512d div(__m512d a, __m512d b) { return _mm512_div_pd(a, b); }
    static __m512d fma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
    static __m512d max(__m512d a, __m512d b) { return _mm512_max_pd(a, b); }
    static __m512d sqrt(__m512d a) { return _mm512_sqrt_pd(a); }
    static __m512d zero() { return _mm512_setzero_pd(); }
    static __m512d one() { return _mm512_set1_pd(1.0); }
    static __m512d half() { return _mm512_set1_pd(0.5); }
    static __m512d sgnmask() { return _mm512_set1_pd(-0.0); }
    static __m512d blend(__m512d a, __m512d b, __m512d m) { return _mm512_mask_blend_pd(_mm512_movepi64_mask(_mm512_castpd_si512(m)), a, b); }
    static __m512d cmp_lt(__m512d a, __m512d b) {
        return _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ), _mm512_set1_epi64(-1)));
    }
    static __m512d cmp_eq(__m512d a, __m512d b) {
        return _mm512_castsi512_pd(_mm512_maskz_mov_epi64(
            _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ), _mm512_set1_epi64(-1)));
    }
    static __m512d bit_and(__m512d a, __m512d b) { return _mm512_and_pd(a, b); }
    static __m512d bit_andnot(__m512d m, __m512d a) { return _mm512_andnot_pd(m, a); }
    static __m512d bit_xor(__m512d a, __m512d b) { return _mm512_xor_pd(a, b); }
    static void store(double* p, __m512d v) { _mm512_storeu_pd(p, v); }
    static __m512d set1(double v) { return _mm512_set1_pd(v); }
};
#endif // TP_VECCPLX_AVX512

// ---------------------------------------------------------------------------
// lane plumbing (always_inline: folded into the attributed kernels below)
// ---------------------------------------------------------------------------

template <typename V>
inline V
loadu(const typename Math<V>::scalar* p) {
    if constexpr (std::is_same_v<V, __m256>) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
    } else if constexpr (std::is_same_v<V, __m512>) {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(p));
    } else if constexpr (std::is_same_v<V, __m512d>) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(p));
    } else {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    }
}

// Deinterleave the packed (re,im) stream; both halves land in the LOW 128
// bits (high zeroed) so lane-wise math pairs re_j with im_j.
// Zero padding keeps libmvec away from denormal slow paths on dead lanes.
//
//   float:  [r0 i0 r1 i1 | r2 i2 r3 i3] -> re=[r0 r1 r2 r3|0], im=[i0 i1 i2 i3|0]
//   double: [r0 i0 | r1 i1]             -> re=[r0 r1|0],          im=[i0 i1|0]
template <typename V>
inline void
split(V v, V& re, V& im) {
    if constexpr (std::is_same_v<V, __m256>) {
        const __m256 a = _mm256_permutevar8x32_ps(
            v, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
        re = _mm256_insertf128_ps(_mm256_setzero_ps(),
                                  _mm256_castps256_ps128(a), 0);
        im = _mm256_insertf128_ps(_mm256_setzero_ps(),
                                  _mm256_extractf128_ps(a, 1), 0);
    } else {
        // VPERMPD (imm8): [r0 i0 r1 i1] -> [r0 r1 i0 i1]
        const __m256d a = _mm256_permute4x64_pd(v, 0xD8);
        re = _mm256_insertf128_pd(_mm256_setzero_pd(),
                                  _mm256_castpd256_pd128(a), 0);
        im = _mm256_insertf128_pd(_mm256_setzero_pd(),
                                  _mm256_extractf128_pd(a, 1), 0);
    }
}

// Inverse of split(): stitch the two low-128 results back into the
// interleaved stream of W complexes.
template <typename V>
inline void
combine_store(V re, V im, typename Math<V>::scalar* dst) {
    if constexpr (std::is_same_v<V, __m256>) {
        const __m256 t = _mm256_permute2f128_ps(re, im, 0x20);  // [re|im]
        _mm256_storeu_ps(reinterpret_cast<float*>(dst),
                         _mm256_permutevar8x32_ps(
                             t, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    } else {
        const __m256d t = _mm256_permute2f128_pd(re, im, 0x20);  // [re|im]
        _mm256_storeu_pd(reinterpret_cast<double*>(dst),
                         _mm256_permute4x64_pd(t, 0xD8));
    }
}


#if defined(TP_VECCPLX_AVX512)
// ---- 512-bit overloads (AVX512 tier) ----
inline void
split(__m512 v, __m512& re, __m512& im) {
    const __m512i even = _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i odd = _mm512_setr_epi32(
        1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    re = _mm512_maskz_permutexvar_ps(0x00FF, even, v);
    im = _mm512_maskz_permutexvar_ps(0x00FF, odd, v);
}
inline void
split(__m512d v, __m512d& re, __m512d& im) {
    const __m512i even = _mm512_setr_epi64(0, 2, 4, 6, 0, 0, 0, 0);
    const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 0, 0, 0, 0);
    re = _mm512_maskz_permutexvar_pd(0x0F, even, v);
    im = _mm512_maskz_permutexvar_pd(0x0F, odd, v);
}

inline void
combine_store(__m512 re, __m512 im, float* dst) {
    const __m512i idx = _mm512_setr_epi32(
        0, 16, 1, 17, 2, 18, 3, 19,
        4, 20, 5, 21, 6, 22, 7, 23);
    _mm512_storeu_ps(dst, _mm512_permutex2var_ps(re, idx, im));
}
inline void
combine_store(__m512d re, __m512d im, double* dst) {
    const __m512i idx = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    _mm512_storeu_pd(dst, _mm512_permutex2var_pd(re, idx, im));
}

inline __m512
vabs(__m512 v) { return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v); }
inline __m512d
vabs(__m512d v) { return _mm512_andnot_pd(_mm512_set1_pd(-0.0), v); }
#endif // TP_VECCPLX_AVX512

template <typename V>
inline V vabs(V v) {
    return Ops<V>::bit_andnot(Ops<V>::sgnmask(), v);
}

template <typename V>
inline void store_low(V v, typename Math<V>::scalar* dst) {
    using S = typename Math<V>::scalar;
    if constexpr (std::is_same_v<V, __m256>) {
        _mm_storeu_ps(dst, _mm256_castps256_ps128(v));
    } else if constexpr (std::is_same_v<V, __m256d>) {
        _mm_storeu_pd(dst, _mm256_castpd256_pd128(v));
#if defined(TP_VECCPLX_AVX512)
    } else if constexpr (std::is_same_v<V, __m512>) {
        _mm512_mask_storeu_ps(dst, 0x00FF, v);
    } else if constexpr (std::is_same_v<V, __m512d>) {
        _mm512_mask_storeu_pd(dst, 0x0F, v);
#endif
    } else {
        static_assert(std::is_same_v<S, void>, "unsupported complex SIMD width");
    }
}

// (sr, si) = csqrt(xr + i*xi), Smith quadrant method; z == 0 -> (0, 0).
template <typename V>
inline void
cx_sqrt(V xr, V xi, V& sr, V& si) {
    using O = Ops<V>;
    const V ax = vabs(xr);
    const V m = O::sqrt(O::fma(xr, xr, O::mul(xi, xi)));       // |z|
    const V t = O::sqrt(O::mul(O::add(m, ax), O::half()));     // sqrt((|z|+|x|)/2)
    const V t2 = O::add(t, t);
    const V neg = O::cmp_lt(xr, O::zero());
    // x >= 0: (t, y/2t)          x < 0: (|y|/2t, copysign(t, y))
    const V re_pos = t, im_pos = O::div(xi, t2);
    const V re_neg = O::div(vabs(xi), t2);
    const V im_neg = O::bit_xor(t, O::bit_and(O::sgnmask(), xi));
    // blend(a, b, m): m set -> b  =>  neg lanes take the *_neg formulas
    sr = O::blend(re_pos, re_neg, neg);
    si = O::blend(im_pos, im_neg, neg);
    const V zm = O::cmp_eq(m, O::zero());
    sr = O::blend(sr, O::zero(), zm);
    si = O::blend(si, O::zero(), zm);
}

// ---------------------------------------------------------------------------
// scalar twins: identical formulas for the ragged tail
// ---------------------------------------------------------------------------

template <typename S>
inline void cx_sqrt_scalar(S xr, S xi, S& sr, S& si) {
    const S m = std::sqrt(xr * xr + xi * xi);
    if (m == S(0)) { sr = S(0); si = S(0); return; }
    const S t = std::sqrt((m + std::fabs(xr)) * S(0.5));
    if (xr >= S(0)) {
        sr = t; si = xi / (t + t);
    } else {
        sr = std::fabs(xi) / (t + t);
        si = xi < S(0) ? -t : t;
    }
}

// asinh body shared by asinh/asin: (ar, ai) = asinh(ir + i*ii)
template <typename S>
inline void cx_asinh_parts(S ir, S ii, S& ar, S& ai) {
    using std::log, std::sqrt, std::atan2;
    const S z2r = ir * ir - ii * ii;
    const S z2i = (ir + ir) * ii;
    const S wr = z2r + S(1);
    S wrr, wii;
    cx_sqrt_scalar(wr, z2i, wrr, wii);
    const S ur = ir + wrr, ui = ii + wii;
    const S mm = sqrt(ur * ur + ui * ui);
    ar = log(mm);
    ai = atan2(ui, ur);
}

template <typename S>
inline void scalar_unary(Op op, const S* xp, S* yp) {
    using std::acos, std::acosh, std::asin, std::atan, std::atanh, std::cos,
          std::cosh, std::exp, std::expm1, std::fabs, std::log, std::sin,
          std::sinh, std::sqrt, std::tan, std::tanh;
    const S xr = xp[0], xi = xp[1];
    S yr = S(0), yi = S(0);
    switch (op) {
        case Op::Neg:
            yr = -xr; yi = -xi; break;
        case Op::Square:
            yr = xr * xr - xi * xi; yi = xr * xi + xr * xi; break;
        case Op::Recip: {
            const S den = xr * xr + xi * xi;
            yr = xr / den; yi = -xi / den; break;
        }
        case Op::Exp: {
            const S e = exp(xr);
            yr = e * cos(xi); yi = e * sin(xi); break;
        }
        case Op::Expm1: {
            // cx_expm1: expm1(x)*cos(y) - 2*sin(y/2)^2 + i*e^x*sin(y)
            const S a = sin(xi / S(2));
            yr = expm1(xr) * cos(xi) - S(2) * a * a;
            yi = exp(xr) * sin(xi); break;
        }
        case Op::Log: {
            yr = log(sqrt(xr * xr + xi * xi)); yi = std::atan2(xi, xr); break;
        }
        case Op::Log1p: {
            const S mr = xr + S(1);
            yr = log(sqrt(mr * mr + xi * xi)); yi = std::atan2(xi, mr); break;
        }
        case Op::Log2: {
            yr = log(sqrt(xr * xr + xi * xi)) / S(0.69314718055994530942);
            yi = std::atan2(xi, xr) / S(0.69314718055994530942); break;
        }
        case Op::Log10: {
            yr = log(sqrt(xr * xr + xi * xi)) / S(2.30258509299404568402);
            yi = std::atan2(xi, xr) / S(2.30258509299404568402); break;
        }
        case Op::Sqrt:
            cx_sqrt_scalar(xr, xi, yr, yi); break;
        case Op::Rsqrt: {
            S sr, si;
            cx_sqrt_scalar(xr, xi, sr, si);
            const S den = sr * sr + si * si;
            yr = sr / den; yi = -si / den; break;
        }
        case Op::Sin: {
            yr = sin(xr) * cosh(xi); yi = cos(xr) * sinh(xi); break;
        }
        case Op::Cos: {
            yr = cos(xr) * cosh(xi); yi = -(sin(xr) * sinh(xi)); break;
        }
        case Op::Tan: {
            const S sx = sin(xr), cx2 = cos(xr);
            const S shy = sinh(xi), cyh = cosh(xi);
            const S den = cx2 * cx2 + shy * shy;
            yr = (sx * cyh) / den; yi = (shy * cyh) / den; break;
        }
        case Op::Sinh: {
            yr = sinh(xr) * cos(xi); yi = cosh(xr) * sin(xi); break;
        }
        case Op::Cosh: {
            yr = cosh(xr) * cos(xi); yi = sinh(xr) * sin(xi); break;
        }
        case Op::Tanh: {
            // tanh(z): num = sinh(z)*conj(cosh(z)) -> (sinh x cosh x,
            // sin y cos y); den = |cosh z|^2 = cosh^2 x - sin^2 y
            const S sxh = sinh(xr), cxh = cosh(xr);
            const S sy = sin(xi), cy = cos(xi);
            const S den = cxh * cxh - sy * sy;
            yr = (sxh * cxh) / den; yi = (sy * cy) / den; break;
        }
        case Op::Asin: {
            // asin(z) = -i*asinh(i*z),  i*z = (-y, x)
            S ar, ai;
            cx_asinh_parts(-xi, xr, ar, ai);
            yr = ai; yi = -ar; break;
        }
        case Op::Acos: {
            S ar, ai;
            cx_asinh_parts(-xi, xr, ar, ai);
            constexpr S hp = 1.57079632679489661923;
            yr = hp - ai; yi = ar; break;
        }
        case Op::Atan: {
            // atan = (i/2)*(log(1-i z) - log(1+i z))
            const S pr = xi + S(1), pi_ = -xr;
            const S qr = S(1) - xi, qi = xr;
            const S lpr = log(sqrt(pr * pr + pi_ * pi_));
            const S lpi = std::atan2(pi_, pr);
            const S lqr = log(sqrt(qr * qr + qi * qi));
            const S lqi = std::atan2(qi, qr);
            const S dr = lpr - lqr, di = lpi - lqi;
            yr = -di / S(2); yi = dr / S(2); break;
        }
        case Op::Asinh:
            cx_asinh_parts(xr, xi, yr, yi); break;
        case Op::Acosh: {
            // acosh(z) = log(z + csqrt(z*z - 1)); Re(z) < 0 takes the
            // conjugate sheet (negate the final log, see the SIMD twin)
            const S z2r = xr * xr - xi * xi;
            const S z2i = (xr + xr) * xi;
            S wrr, wii;
            cx_sqrt_scalar(z2r - S(1), z2i, wrr, wii);
            const S ur = xr + wrr, ui = xi + wii;
            yr = log(sqrt(ur * ur + ui * ui));
            yi = std::atan2(ui, ur);
            if (xr < S(0)) { yr = -yr; yi = -yi; }
            break;
        }
        case Op::Atanh: {
            // (log(1+z) - log(1-z)) / 2, polar form per part
            const S pr = xr + S(1), qr = S(1) - xr;
            const S dr = log(sqrt(pr * pr + xi * xi)) -
                         log(sqrt(qr * qr + xi * xi));
            const S di = std::atan2(xi, pr) - std::atan2(-xi, qr);
            yr = dr / S(2); yi = di / S(2); break;
        }
        case Op::Sigmoid: {
            const S nr = -xr, ni = -xi;
            const S e = exp(nr);
            const S er = e * cos(ni), ei = e * sin(ni);
            const S orr = er + S(1);
            const S den = orr * orr + ei * ei;
            yr = orr / den; yi = -ei / den; break;
        }
        default: break;
    }
    yp[0] = yr; yp[1] = yi;
}

// ---------------------------------------------------------------------------
// cores: process n complex elements from contiguous streams
// ---------------------------------------------------------------------------

template <typename V>
void binary_core(
        const typename Math<V>::scalar* a, const typename Math<V>::scalar* b,
        typename Math<V>::scalar* y, int64_t n, Op op) {
    using S = typename Math<V>::scalar;
    using O = Ops<V>;
    constexpr int64_t W = Math<V>::W;
    const int64_t vec_end = (n / W) * W;

    for (int64_t i = 0; i < vec_end; i += W) {
        const S* ap = a + 2 * i;
        const S* bp = b + 2 * i;
        // add/sub are lane-local over the interleaved stream: no deinterleave
        if (op == Op::Add || op == Op::Sub) {
            V av = loadu<V>(ap), bv = loadu<V>(bp);
            V yv = op == Op::Add ? O::add(av, bv) : O::sub(av, bv);
            O::store(y + 2 * i, yv);
            continue;
        }
        V av = loadu<V>(ap), bv = loadu<V>(bp);
        V ar, ai, br, bi;
        split(av, ar, ai);
        split(bv, br, bi);
        V yr = O::zero(), yi = O::zero();
        switch (op) {
            case Op::Mul:
                yr = O::sub(O::mul(ar, br), O::mul(ai, bi));
                yi = O::add(O::mul(ar, bi), O::mul(ai, br));
                break;
            case Op::Div: {
                V m = O::max(vabs(br), vabs(bi));
                V inv = O::div(O::one(), m);
                V b2r = O::mul(br, inv);
                V b2i = O::mul(bi, inv);
                // denominator carries the scale factor: (a+bi)/(m*(c'+di'))
                V den = O::mul(O::add(O::mul(b2r, b2r), O::mul(b2i, b2i)), m);
                yr = O::div(O::add(O::mul(ar, b2r), O::mul(ai, b2i)), den);
                yi = O::div(O::sub(O::mul(ai, b2r), O::mul(ar, b2i)), den);
                break;
            }
            default: break;
        }
        combine_store(yr, yi, y + 2 * i);
    }
    for (int64_t i = vec_end; i < n; ++i) {
        const S xr = a[2 * i], xi = a[2 * i + 1];
        const S cr = b[2 * i], ci = b[2 * i + 1];
        S yr = S(0), yi = S(0);
        switch (op) {
            case Op::Add:
                yr = xr + cr; yi = xi + ci; break;
            case Op::Sub:
                yr = xr - cr; yi = xi - ci; break;
            case Op::Mul:
                yr = xr * cr - xi * ci;
                yi = xr * ci + xi * cr;
                break;
            case Op::Div: {
                const S m = std::max(std::fabs(cr), std::fabs(ci));
                const S b2r = cr / m, b2i = ci / m;
                const S den = (b2r * b2r + b2i * b2i) * m;
                yr = (xr * b2r + xi * b2i) / den;
                yi = (xi * b2r - xr * b2i) / den;
                break;
            }
            default: break;
        }
        y[2 * i] = yr; y[2 * i + 1] = yi;
    }
}

template <typename V>
void unary_core(
        const typename Math<V>::scalar* x, typename Math<V>::scalar* y,
        int64_t n, Op op) {
    using S = typename Math<V>::scalar;
    using M = Math<V>;
    using O = Ops<V>;
    constexpr int64_t W = Math<V>::W;
    const int64_t vec_end = (n / W) * W;

    for (int64_t i = 0; i < vec_end; i += W) {
        const S* xp = x + 2 * i;
        V xv = loadu<V>(xp);
        V xr, xi;
        split(xv, xr, xi);
        V yr = O::zero(), yi = O::zero();
        switch (op) {
            case Op::Neg:
                yr = O::bit_xor(O::sgnmask(), xr);
                yi = O::bit_xor(O::sgnmask(), xi);
                break;
            case Op::Square:
                yr = O::sub(O::mul(xr, xr), O::mul(xi, xi));
                yi = O::add(O::mul(xr, xi), O::mul(xr, xi));
                break;
            case Op::Recip: {
                V m = O::add(O::mul(xr, xr), O::mul(xi, xi));
                yr = O::div(xr, m);
                yi = O::div(O::sub(O::zero(), xi), m);
                break;
            }
            case Op::Exp: {
                V e = M::exp(xr);
                yr = O::mul(e, M::cos(xi));
                yi = O::mul(e, M::sin(xi));
                break;
            }
            case Op::Expm1: {
                V syh = M::sin(O::mul(xi, O::half()));
                yr = O::sub(O::mul(M::expm1(xr), M::cos(xi)),
                            O::mul(O::mul(syh, syh), O::add(O::one(), O::one())));
                yi = O::mul(M::exp(xr), M::sin(xi));
                break;
            }
            case Op::Log: {
                V m = O::sqrt(O::add(O::mul(xr, xr), O::mul(xi, xi)));
                yr = M::log(m);
                yi = M::atan2(xi, xr);
                break;
            }
            case Op::Log1p: {
                V mr = O::add(xr, O::one());
                V m = O::sqrt(O::add(O::mul(mr, mr), O::mul(xi, xi)));
                yr = M::log(m);
                yi = M::atan2(xi, mr);
                break;
            }
            case Op::Log2:
            case Op::Log10: {
                V m = O::sqrt(O::add(O::mul(xr, xr), O::mul(xi, xi)));
                // divide by ln2/ln10: matches cx_log2/cx_log10 (log(z)/log(k))
                V k = op == Op::Log2 ? M::ln2() : M::ln10();
                yr = O::div(M::log(m), k);
                yi = O::div(M::atan2(xi, xr), k);
                break;
            }
            case Op::Sqrt:
                cx_sqrt(xr, xi, yr, yi);
                break;
            case Op::Rsqrt: {
                // rsqrt(z) = conj(csqrt(z)) / |csqrt(z)|^2
                V sr, si;
                cx_sqrt(xr, xi, sr, si);
                V den = O::add(O::mul(sr, sr), O::mul(si, si));
                yr = O::div(sr, den);
                yi = O::div(O::sub(O::zero(), si), den);
                break;
            }
            case Op::Sin: {
                yr = O::mul(M::sin(xr), M::cosh(xi));
                yi = O::mul(M::cos(xr), M::sinh(xi));
                break;
            }
            case Op::Cos: {
                yr = O::mul(M::cos(xr), M::cosh(xi));
                yi = O::sub(O::zero(), O::mul(M::sin(xr), M::sinh(xi)));
                break;
            }
            case Op::Tan: {
                // tan(z): num = sin(z)*conj(cos(z)) -> (sin x cos x,
                // cosh y sinh y); den = |cos z|^2 = cos^2 x + sinh^2 y
                V sx = M::sin(xr), cx = M::cos(xr);
                V shy = M::sinh(xi), cyh = M::cosh(xi);
                V den = O::add(O::mul(cx, cx), O::mul(shy, shy));
                yr = O::div(O::mul(sx, cx), den);
                yi = O::div(O::mul(shy, cyh), den);
                break;
            }
            case Op::Sinh: {
                yr = O::mul(M::sinh(xr), M::cos(xi));
                yi = O::mul(M::cosh(xr), M::sin(xi));
                break;
            }
            case Op::Cosh: {
                yr = O::mul(M::cosh(xr), M::cos(xi));
                yi = O::mul(M::sinh(xr), M::sin(xi));
                break;
            }
            case Op::Tanh: {
                // tanh(z): (sinh x cosh x, sin y cos y) / (cosh^2 x - sin^2 y)
                V sxh = M::sinh(xr), cxh = M::cosh(xr);
                V sy = M::sin(xi), cy = M::cos(xi);
                V den = O::sub(O::mul(cxh, cxh), O::mul(sy, sy));
                yr = O::div(O::mul(sxh, cxh), den);
                yi = O::div(O::mul(sy, cy), den);
                break;
            }
            case Op::Asin:
            case Op::Acos: {
                // -i*asinh(i*z),  i*z = (-y, x); acos adds pi/2 - on top
                V ir = O::sub(O::zero(), xi), ii = xr;
                V z2r = O::sub(O::mul(ir, ir), O::mul(ii, ii));
                V z2i = O::mul(O::add(ir, ir), ii);
                V wr = O::add(z2r, O::one());
                V wr1, wr2;
                cx_sqrt(wr, z2i, wr1, wr2);
                V ur = O::add(ir, wr1), ui = O::add(ii, wr2);
                V mm = O::sqrt(O::add(O::mul(ur, ur), O::mul(ui, ui)));
                V ar = M::log(mm);
                V ai = M::atan2(ui, ur);
                if (op == Op::Asin) {
                    // -i*(ar + i*ai) = ai - i*ar
                    yr = ai;
                    yi = O::sub(O::zero(), ar);
                } else {
                    // acos(z) = pi/2 - asin(z)
                    constexpr S hp = 1.57079632679489661923;
                    V hvec;
                    hvec = O::set1(static_cast<typename Math<V>::scalar>(hp));
                    yr = O::sub(hvec, ai);
                    yi = ar;
                }
                break;
            }
            case Op::Atan: {
                // atan = (i/2)*(log(1-i z) - log(1+i z))
                // 1 - i z = (1 + y, -x);  1 + i z = (1 - y, x)
                V pr = O::add(xi, O::one()), pi_ = O::sub(O::zero(), xr);
                V qr = O::sub(O::one(), xi), qi = xr;
                V lpr = M::log(O::sqrt(O::add(O::mul(pr, pr), O::mul(pi_, pi_))));
                V lpi = M::atan2(pi_, pr);
                V lqr = M::log(O::sqrt(O::add(O::mul(qr, qr), O::mul(qi, qi))));
                V lqi = M::atan2(qi, qr);
                V dr = O::sub(lpr, lqr);
                V di = O::sub(lpi, lqi);
                yr = O::mul(O::sub(O::zero(), di), O::half());
                yi = O::mul(dr, O::half());
                break;
            }
            case Op::Asinh:
            case Op::Acosh: {
                // log(z + csqrt(z*z +/- 1)); for acosh with Re(z) < 0 the
                // principal branch takes the conjugate sheet: z - csqrt,
                // which equals negating both parts of the final log.
                V z2r = O::sub(O::mul(xr, xr), O::mul(xi, xi));
                V z2i = O::mul(O::add(xr, xr), xi);
                V wr = op == Op::Asinh ? O::add(z2r, O::one())
                                       : O::sub(z2r, O::one());
                V w1r, w1i;
                cx_sqrt(wr, z2i, w1r, w1i);
                V ur = O::add(xr, w1r), ui = O::add(xi, w1i);
                V mm = O::sqrt(O::add(O::mul(ur, ur), O::mul(ui, ui)));
                yr = M::log(mm);
                yi = M::atan2(ui, ur);
                if (op == Op::Acosh) {
                    V flip = O::bit_and(O::sgnmask(),
                                        O::cmp_lt(xr, O::zero()));
                    yr = O::bit_xor(yr, flip);
                    yi = O::bit_xor(yi, flip);
                }
                break;
            }
            case Op::Atanh: {
                // (log(1+z) - log(1-z)) / 2, polar form per part
                V pr = O::add(xr, O::one());
                V qr = O::sub(O::one(), xr);
                V mp = O::sqrt(O::fma(pr, pr, O::mul(xi, xi)));
                V mq = O::sqrt(O::fma(qr, qr, O::mul(xi, xi)));
                V dr = O::sub(M::log(mp), M::log(mq));
                V di = O::sub(M::atan2(xi, pr),
                              M::atan2(O::sub(O::zero(), xi), qr));
                yr = O::mul(dr, O::half());
                yi = O::mul(di, O::half());
                break;
            }
            case Op::Sigmoid: {
                // 1/(1+exp(-z))
                V nr = O::bit_xor(O::sgnmask(), xr);
                V ni = O::bit_xor(O::sgnmask(), xi);
                V e = M::exp(nr);
                V er = O::mul(e, M::cos(ni));
                V ei = O::mul(e, M::sin(ni));
                V orr = O::add(er, O::one());
                V den = O::add(O::mul(orr, orr), O::mul(ei, ei));
                yr = O::div(orr, den);
                yi = O::div(O::sub(O::zero(), ei), den);
                break;
            }
            default: break;
        }
        combine_store(yr, yi, y + 2 * i);
    }
    for (int64_t i = vec_end; i < n; ++i)
        scalar_unary<S>(op, x + 2 * i, y + 2 * i);
}

template <typename V>
void abs_core(
        const typename Math<V>::scalar* x, typename Math<V>::scalar* out,
        int64_t n) {
    using M = Math<V>;
    using O = Ops<V>;
    constexpr int64_t W = M::W;
    const int64_t vec_end = (n / W) * W;
    for (int64_t i = 0; i < vec_end; i += W) {
        V xv = loadu<V>(x + 2 * i);
        V xr, xi;
        split(xv, xr, xi);
        // |z| = m * sqrt((x/m)^2 + (y/m)), m = max(|x|,|y|): overflow-safe
        // like std::hypot but branch-free and faster than libmvec's entry
        V m = O::max(vabs(xr), vabs(xi));
        V s = O::div(O::one(), m);
        V rx = O::mul(xr, s), ry = O::mul(xi, s);
        V mag = O::mul(m, O::sqrt(O::fma(rx, rx, O::mul(ry, ry))));
        // m == 0 lanes produce 0*NaN -> clamp back to zero
        mag = O::blend(mag, O::zero(), O::cmp_eq(m, O::zero()));
        store_low<V>(mag, out + i);
    }
    for (int64_t i = vec_end; i < n; ++i)
        out[i] = std::hypot(x[2 * i], x[2 * i + 1]);
}

template <typename V>
void angle_core(
        const typename Math<V>::scalar* x, typename Math<V>::scalar* out,
        int64_t n) {
    using M = Math<V>;
    constexpr int64_t W = M::W;
    const int64_t vec_end = (n / W) * W;
    for (int64_t i = 0; i < vec_end; i += W) {
        V xv = loadu<V>(x + 2 * i);
        V xr, xi;
        split(xv, xr, xi);
        V a = M::atan2(xi, xr);
        // split() keeps valid complex lanes in the low W lanes.
        store_low<V>(a, out + i);
    }
    for (int64_t i = vec_end; i < n; ++i)
        out[i] = std::atan2(x[2 * i + 1], x[2 * i]);
}

template <typename V>
void sum_core(
        const typename Math<V>::scalar* x, int64_t n,
        typename Math<V>::scalar* re_out, typename Math<V>::scalar* im_out) {
    using S = typename Math<V>::scalar;
    using O = Ops<V>;
    constexpr int64_t W = Math<V>::W;
    V acc_r = O::zero(), acc_i = O::zero();
    const int64_t vec_end = (n / W) * W;
    for (int64_t i = 0; i < vec_end; i += W) {
        V xv = loadu<V>(x + 2 * i);
        V xr, xi;
        split(xv, xr, xi);
        acc_r = O::add(acc_r, xr);
        acc_i = O::add(acc_i, xi);
    }
    S rr = S(0), ri = S(0);
    alignas(64) S buf[W];
    store_low<V>(acc_r, buf);
    for (int lane = 0; lane < W; ++lane) rr += buf[lane];
    store_low<V>(acc_i, buf);
    for (int lane = 0; lane < W; ++lane) ri += buf[lane];
    for (int64_t i = vec_end; i < n; ++i) {
        rr += x[2 * i];
        ri += x[2 * i + 1];
    }
    *re_out = rr;
    *im_out = ri;
}

#endif // TP_VECCPLX_LIBMVEC

// ---------------------------------------------------------------------------
// public entry points (parallel over chunks; return false -> caller falls
// back to the scalar drivers)
// ---------------------------------------------------------------------------

 // complex elements per parallel chunk

bool try_unary_impl(const void* xv, void* yv, int64_t n, int dt_i, int op_id) {
    const DType dt = static_cast<DType>(dt_i);
    const veccomplex::Op op = static_cast<veccomplex::Op>(op_id);

#ifdef TP_VECCPLX_LIBMVEC
    if (!width_ok(dt) || n <= 0) return false;
#ifdef TP_VECCPLX_AVX512
    if (avx512_available()) {
        if (dt == DType::ComplexFloat) {
            const float* x = static_cast<const float*>(xv);
            float* y = static_cast<float*>(yv);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    unary_core<__m512>(x + 2 * b, y + 2 * b, e - b, op);
                });
            return true;
        }
        const double* x = static_cast<const double*>(xv);
        double* y = static_cast<double*>(yv);
        tensorplay::parallel::parallel_for(0, n, kGrain,
            [&](int64_t b, int64_t e) {
                unary_core<__m512d>(x + 2 * b, y + 2 * b, e - b, op);
            });
        return true;
    }
#endif
    if (!avx2_available()) return false;
    if (dt == DType::ComplexFloat) {
        const float* x = static_cast<const float*>(xv);
        float* y = static_cast<float*>(yv);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            unary_core<__m256>(x + 2 * b, y + 2 * b, e - b, op);
        });
        return true;
    }
    const double* x = static_cast<const double*>(xv);
    double* y = static_cast<double*>(yv);
    tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
        unary_core<__m256d>(x + 2 * b, y + 2 * b, e - b, op);
    });
    return true;
#else
    (void)xv; (void)yv; (void)n; (void)dt; (void)op;
    return false;
#endif
}

bool try_binary_impl(const void* av_, const void* bv, void* yv, int64_t n, int dt_i, int op_id) {
    const DType dt = static_cast<DType>(dt_i);
    const veccomplex::Op op = static_cast<veccomplex::Op>(op_id);

#ifdef TP_VECCPLX_LIBMVEC
    if (!width_ok(dt) || n <= 0 || !binary_supported(op))
        return false;
#ifdef TP_VECCPLX_AVX512
    if (avx512_available()) {
        if (dt == DType::ComplexFloat) {
            const float* a = static_cast<const float*>(av_);
            const float* b = static_cast<const float*>(bv);
            float* y = static_cast<float*>(yv);
            tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t bgn, int64_t e) {
                binary_core<__m512>(a + 2 * bgn, b + 2 * bgn, y + 2 * bgn, e - bgn, op);
            });
            return true;
        }
        const double* a = static_cast<const double*>(av_);
        const double* b = static_cast<const double*>(bv);
        double* y = static_cast<double*>(yv);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t bgn, int64_t e) {
            binary_core<__m512d>(a + 2 * bgn, b + 2 * bgn, y + 2 * bgn, e - bgn, op);
        });
        return true;
    }
#endif
    if (!avx2_available())
        return false;
    if (dt == DType::ComplexFloat) {
        const float* a = static_cast<const float*>(av_);
        const float* b = static_cast<const float*>(bv);
        float* y = static_cast<float*>(yv);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t bgn, int64_t e) {
            binary_core<__m256>(a + 2 * bgn, b + 2 * bgn, y + 2 * bgn, e - bgn, op);
        });
        return true;
    }
    const double* a = static_cast<const double*>(av_);
    const double* b = static_cast<const double*>(bv);
    double* y = static_cast<double*>(yv);
    tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t bgn, int64_t e) {
        binary_core<__m256d>(a + 2 * bgn, b + 2 * bgn, y + 2 * bgn, e - bgn, op);
    });
    return true;
#else
    (void)av_; (void)bv; (void)yv; (void)n; (void)dt; (void)op;
    return false;
#endif
}

bool try_abs_impl(const void* xv, void* real_out, int64_t n, int dt_i) {
    const DType dt = static_cast<DType>(dt_i);

#ifdef TP_VECCPLX_LIBMVEC
    if (n <= 0) return false;
#ifdef TP_VECCPLX_AVX512
    if (avx512_available()) {
        if (dt == DType::ComplexFloat) {
            const float* x = static_cast<const float*>(xv);
            float* o = static_cast<float*>(real_out);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    abs_core<__m512>(x + 2 * b, o + b, e - b);
                });
            return true;
        }
        if (dt == DType::ComplexDouble) {
            const double* x = static_cast<const double*>(xv);
            double* o = static_cast<double*>(real_out);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    abs_core<__m512d>(x + 2 * b, o + b, e - b);
                });
            return true;
        }
        return false;
    }
#endif
    if (!avx2_available()) return false;
    if (dt == DType::ComplexFloat) {
        const float* x = static_cast<const float*>(xv);
        float* o = static_cast<float*>(real_out);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            abs_core<__m256>(x + 2 * b, o + b, e - b);
        });
        return true;
    }
    if (dt == DType::ComplexDouble) {
        const double* x = static_cast<const double*>(xv);
        double* o = static_cast<double*>(real_out);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            abs_core<__m256d>(x + 2 * b, o + b, e - b);
        });
        return true;
    }
    return false;
#else
    (void)xv; (void)real_out; (void)n; (void)dt;
    return false;
#endif
}

bool try_angle_impl(const void* xv, void* real_out, int64_t n, int dt_i) {
    const DType dt = static_cast<DType>(dt_i);

#ifdef TP_VECCPLX_LIBMVEC
    if (n <= 0) return false;
#ifdef TP_VECCPLX_AVX512
    if (avx512_available()) {
        if (dt == DType::ComplexFloat) {
            const float* x = static_cast<const float*>(xv);
            float* o = static_cast<float*>(real_out);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    angle_core<__m512>(x + 2 * b, o + b, e - b);
                });
            return true;
        }
        if (dt == DType::ComplexDouble) {
            const double* x = static_cast<const double*>(xv);
            double* o = static_cast<double*>(real_out);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    angle_core<__m512d>(x + 2 * b, o + b, e - b);
                });
            return true;
        }
        return false;
    }
#endif
    if (!avx2_available()) return false;
    if (dt == DType::ComplexFloat) {
        const float* x = static_cast<const float*>(xv);
        float* o = static_cast<float*>(real_out);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            angle_core<__m256>(x + 2 * b, o + b, e - b);
        });
        return true;
    }
    if (dt == DType::ComplexDouble) {
        const double* x = static_cast<const double*>(xv);
        double* o = static_cast<double*>(real_out);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            angle_core<__m256d>(x + 2 * b, o + b, e - b);
        });
        return true;
    }
    return false;
#else
    (void)xv; (void)real_out; (void)n; (void)dt;
    return false;
#endif
}

bool try_sum_impl(const void* xv, int64_t n, int dt_i, double* re_out, double* im_out) {
    const DType dt = static_cast<DType>(dt_i);
#ifdef TP_VECCPLX_LIBMVEC
    if (n <= 0) return false;
    const int64_t nslots = (n + kGrain - 1) / kGrain;
#ifdef TP_VECCPLX_AVX512
    if (avx512_available()) {
        if (dt == DType::ComplexFloat) {
            const float* x = static_cast<const float*>(xv);
            std::vector<float> pr(nslots, 0.f), pi(nslots, 0.f);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    float r, i;
                    sum_core<__m512>(x + 2 * b, e - b, &r, &i);
                    pr[b / kGrain] = r;
                    pi[b / kGrain] = i;
                });
            float rr = 0.f, ri = 0.f;
            for (int64_t slot = 0; slot < nslots; ++slot) {
                rr += pr[slot];
                ri += pi[slot];
            }
            *re_out = rr;
            *im_out = ri;
            return true;
        }
        if (dt == DType::ComplexDouble) {
            const double* x = static_cast<const double*>(xv);
            std::vector<double> pr(nslots, 0.), pi(nslots, 0.);
            tensorplay::parallel::parallel_for(0, n, kGrain,
                [&](int64_t b, int64_t e) {
                    double r, i;
                    sum_core<__m512d>(x + 2 * b, e - b, &r, &i);
                    pr[b / kGrain] = r;
                    pi[b / kGrain] = i;
                });
            double rr = 0., ri = 0.;
            for (int64_t slot = 0; slot < nslots; ++slot) {
                rr += pr[slot];
                ri += pi[slot];
            }
            *re_out = rr;
            *im_out = ri;
            return true;
        }
        return false;
    }
#endif
    if (!avx2_available()) return false;
    if (dt == DType::ComplexFloat) {
        const float* x = static_cast<const float*>(xv);
        std::vector<float> pr(nslots, 0.f), pi(nslots, 0.f);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            float r, i;
            sum_core<__m256>(x + 2 * b, e - b, &r, &i);
            pr[b / kGrain] = r; pi[b / kGrain] = i;
        });
        float rr = 0.f, ri = 0.f;
        for (int64_t sI = 0; sI < nslots; ++sI) { rr += pr[sI]; ri += pi[sI]; }
        *re_out = rr; *im_out = ri;
        return true;
    }
    if (dt == DType::ComplexDouble) {
        const double* x = static_cast<const double*>(xv);
        std::vector<double> pr(nslots, 0.), pi(nslots, 0.);
        tensorplay::parallel::parallel_for(0, n, kGrain, [&](int64_t b, int64_t e) {
            double r, i;
            sum_core<__m256d>(x + 2 * b, e - b, &r, &i);
            pr[b / kGrain] = r; pi[b / kGrain] = i;
        });
        double rr = 0., ri = 0.;
        for (int64_t sI = 0; sI < nslots; ++sI) { rr += pr[sI]; ri += pi[sI]; }
        *re_out = rr; *im_out = ri;
        return true;
    }
    return false;
#else
    (void)xv;
    (void)n;
    (void)dt;
    (void)re_out;
    (void)im_out;
    return false;
#endif
}

} // namespace cplxk

} // inline namespace CPU_CAPABILITY

using cplxk::kGrain;
using cplxk::try_unary_impl;
using cplxk::try_binary_impl;
using cplxk::try_abs_impl;
using cplxk::try_angle_impl;
using cplxk::try_sum_impl;

#ifndef TP_COMPLEX_KERNELS_NO_DISPATCH_DEFINITION
DEFINE_DISPATCH(cplx_unary_stub);
DEFINE_DISPATCH(cplx_binary_stub);
DEFINE_DISPATCH(cplx_abs_stub);
DEFINE_DISPATCH(cplx_angle_stub);
DEFINE_DISPATCH(cplx_sum_stub);
#endif

// The implementation is compiled once per CPU_CAPABILITY.  The AVX512
// dispatch macro needs the explicit ALSO_ form because REGISTER_DISPATCH
// intentionally leaves that slot null in the other tiered kernels.
#if defined(CPU_CAPABILITY)
#if defined(CPU_CAPABILITY_AVX512)
#define REGISTER_CPLX_DISPATCH(name, fn) ALSO_REGISTER_AVX512_DISPATCH(name, fn)
#else
#define REGISTER_CPLX_DISPATCH(name, fn) REGISTER_DISPATCH(name, fn)
#endif
REGISTER_CPLX_DISPATCH(cplx_unary_stub, &try_unary_impl);
REGISTER_CPLX_DISPATCH(cplx_binary_stub, &try_binary_impl);
REGISTER_CPLX_DISPATCH(cplx_abs_stub, &try_abs_impl);
REGISTER_CPLX_DISPATCH(cplx_angle_stub, &try_angle_impl);
REGISTER_CPLX_DISPATCH(cplx_sum_stub, &try_sum_impl);
#undef REGISTER_CPLX_DISPATCH
#else
// Source-only probes and non-tiered builds retain a usable registration.
REGISTER_ALL_CPU_DISPATCH(cplx_unary_stub, &try_unary_impl);
REGISTER_ALL_CPU_DISPATCH(cplx_binary_stub, &try_binary_impl);
REGISTER_ALL_CPU_DISPATCH(cplx_abs_stub, &try_abs_impl);
REGISTER_ALL_CPU_DISPATCH(cplx_angle_stub, &try_angle_impl);
REGISTER_ALL_CPU_DISPATCH(cplx_sum_stub, &try_sum_impl);
#endif

} // namespace cpu
} // namespace tensorplay
