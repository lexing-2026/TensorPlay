#pragma once

// 512-bit layer: float/double/int32/int64 plus the packed 16-bit float
// (bfloat16/half) registers, the cross-dtype conversion table and the
// mask wrapper.  The 256-bit types remain visible for code that holds
// both widths (e.g. bf16<->fp32 register pairs).

#include "cpu/vec/vec_base.h"

#include "cpu/vec/vec512/vec512_float.h"
#include "cpu/vec/vec512/vec512_double.h"
#include "cpu/vec/vec512/vec512_int.h"
#include "cpu/vec/vec512/vec512_bfloat16.h"
#include "cpu/vec/vec512/vec512_qint.h"
#include "cpu/vec/vec512/vec512_float8.h"
#include "cpu/vec/vec_n.h"
#include "cpu/vec/vec_convert.h"
#include "cpu/vec/vec_mask.h"
#include "cpu/vec/vec512/vec512_convert.h"
#include "cpu/vec/vec512/vec512_mask.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tensorplay::vec {

// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `tensorplay::vec`.
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX512

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX512) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm512_castpd_ps(src);
}

template <>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm512_castps_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<double>> inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  // GCC 11 lacks _mm512_i64gather_pd; split into two 256-bit gathers.
  __m512i vi = static_cast<__m512i>(vindex);
  __m256i lo = _mm512_castsi512_si256(vi);
  __m256i hi = _mm512_extracti64x4_epi64(vi, 1);
  __m256d rlo = _mm256_i64gather_pd(base_addr, lo, scale);
  __m256d rhi = _mm256_i64gather_pd(base_addr, hi, scale);
  return _mm512_insertf64x4(_mm512_castpd256_pd512(rlo), rhi, 1);
}

template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<float>> inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm512_i32gather_ps(vindex, base_addr, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>> inline mask_gather(
    const Vectorized<double>& src,
    const double* base_addr,
    const Vectorized<int64_t>& vindex,
    Vectorized<double>& mask) {
  __mmask8 k = _mm512_movepi64_mask(_mm512_castpd_si512(mask));
  __m512i vi = static_cast<__m512i>(vindex);
  __m256i lo = _mm512_castsi512_si256(vi);
  __m256i hi = _mm512_extracti64x4_epi64(vi, 1);
  __m256d slo = _mm512_castpd512_pd256(src);
  __m256d shi = _mm512_extractf64x4_pd(src, 1);
  __m256d rlo = _mm256_mask_i64gather_pd(slo, base_addr, lo, slo, scale);
  __m256d rhi = _mm256_mask_i64gather_pd(shi, base_addr, hi, shi, scale);
  return _mm512_insertf64x4(_mm512_castpd256_pd512(rlo), rhi, 1);
}

template <int64_t scale = 1>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>> inline mask_gather(
        const Vectorized<float>& src,
        const float* base_addr,
        const Vectorized<int32_t>& vindex,
        Vectorized<float>& mask) {
  return _mm512_mask_i32gather_ps(
      src,
      _mm512_movepi32_mask(_mm512_castps_si512(mask)),
      vindex,
      base_addr,
      scale);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  const __m512i idx_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
  const __m512i idx_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
  return std::make_pair(
      _mm512_permutex2var_pd(a, idx_lo, b),
      _mm512_permutex2var_pd(a, idx_hi, b));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  const __m512i idx_lo = _mm512_setr_epi32(
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i idx_hi = _mm512_setr_epi32(
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  return std::make_pair(
      _mm512_permutex2var_ps(a, idx_lo, b),
      _mm512_permutex2var_ps(a, idx_hi, b));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  const __m512i idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
  const __m512i idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
  return std::make_pair(
      _mm512_permutex2var_pd(a, idx_even, b),
      _mm512_permutex2var_pd(a, idx_odd, b));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  const __m512i idx_even = _mm512_setr_epi32(
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  const __m512i idx_odd = _mm512_setr_epi32(
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  return std::make_pair(
      _mm512_permutex2var_ps(a, idx_even, b),
      _mm512_permutex2var_ps(a, idx_odd, b));
}

#endif // CPU_CAPABILITY_AVX512

} // namespace tensorplay::vec::inline CPU_CAPABILITY

} // namespace tensorplay::vec
