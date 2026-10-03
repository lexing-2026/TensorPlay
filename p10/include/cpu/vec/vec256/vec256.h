#pragma once

// float/double/int32/int64 specializations plus the reduced-precision
// (bfloat16/half) layers, the cross-dtype conversion table and the mask
// wrapper. AVX512 inherits the 256-bit types through vec512.h.

#include "cpu/vec/vec_base.h"

#include "cpu/vec/vec256/vec256_float.h"
#include "cpu/vec/vec256/vec256_double.h"
#include "cpu/vec/vec256/vec256_int.h"
#include "cpu/vec/vec256/vec256_bfloat16.h"
#include "cpu/vec/vec256/vec256_half.h"
#include "cpu/vec/vec256/vec256_qint.h"
#include "cpu/vec/vec_n.h"
#include "cpu/vec/vec_convert.h"
#include "cpu/vec/vec256/vec256_convert.h"
#include "cpu/vec/vec_mask.h"
#include "cpu/vec/vec256/vec256_mask.h"

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
inline namespace CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX2

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm256_castpd_ps(src);
}

template <>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm256_castps_pd(src);
}

template <>
inline Vectorized<float> cast<float, int32_t>(const Vectorized<int32_t>& src) {
  return _mm256_castsi256_ps(src);
}

template <>
inline Vectorized<double> cast<double, int64_t>(
    const Vectorized<int64_t>& src) {
  return _mm256_castsi256_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<
        double>> inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm256_i64gather_pd(base_addr, vindex, scale);
}

template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<
        float>> inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm256_i32gather_ps(base_addr, vindex, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <int64_t scale = 1>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>> inline mask_gather(
        const Vectorized<double>& src,
        const double* base_addr,
        const Vectorized<int64_t>& vindex,
        Vectorized<double>& mask) {
  return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
}

template <int64_t scale = 1>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>> inline mask_gather(
        const Vectorized<float>& src,
        const float* base_addr,
        const Vectorized<int32_t>& vindex,
        Vectorized<float>& mask) {
  return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Between a floating type and the integer type of the same width, lane for
// lane.  The 256-bit tier has the 32-bit conversions as instructions; the
// 64-bit ones are done with the exponent trick below.

// Exact for inputs in [-2^51, 2^51]: adding 1.5 * 2^52 puts the integer in
// the low mantissa bits, and subtracting the same constant's bit pattern
// leaves the integer.
template <>
Vectorized<int64_t> inline convert_to_int_of_same_size<double>(
    const Vectorized<double>& src) {
  auto x = _mm256_add_pd(src, _mm256_set1_pd(0x0018000000000000));
  return _mm256_sub_epi64(
      _mm256_castpd_si256(x),
      _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
}

template <>
Vectorized<int32_t> inline convert_to_int_of_same_size<float>(
    const Vectorized<float>& src) {
  return _mm256_cvttps_epi32(src);
}

// Each 64-bit integer is split into its two 32-bit halves, each half is
// placed in the mantissa of a double with a known exponent, and the two
// doubles are combined with the exponents' contribution taken back out.
template <>
Vectorized<double> inline convert_to_fp_of_same_size<double>(
    const Vectorized<int64_t>& src) {
  __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000); /* 2^52 */
  __m256i magic_i_hi32 =
      _mm256_set1_epi64x(0x4530000080000000); /* 2^84 + 2^63 */
  __m256i magic_i_all =
      _mm256_set1_epi64x(0x4530000080100000); /* 2^84 + 2^63 + 2^52 */
  __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);

  __m256i v_lo = _mm256_blend_epi32(
      magic_i_lo, src, 0b01010101); /* low32 + 2^52 */
  __m256i v_hi = _mm256_srli_epi64(src, 32);
  v_hi = _mm256_xor_si256(v_hi, magic_i_hi32); /* high32 * 2^32 + 2^84 + 2^63 */
  /* value = low32 + high32 * 2^32 = v_hi + v_lo - 2^52 - 2^63 - 2^84 */
  __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
  return _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
}

template <>
Vectorized<float> inline convert_to_fp_of_same_size<float>(
    const Vectorized<int32_t>& src) {
  return _mm256_cvtepi32_ps(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  auto a_swapped =
      _mm256_permute2f128_pd(a, b, 0b0100000); // 0, 2.   4 bits apart
  auto b_swapped =
      _mm256_permute2f128_pd(a, b, 0b0110001); // 1, 3.   4 bits apart
  return std::make_pair(
      _mm256_permute4x64_pd(a_swapped, 0b11011000), // 0, 2, 1, 3
      _mm256_permute4x64_pd(b_swapped, 0b11011000)); // 0, 2, 1, 3
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  auto a_swapped =
      _mm256_permute2f128_ps(a, b, 0b0100000); // 0, 2.   4 bits apart
  auto b_swapped =
      _mm256_permute2f128_ps(a, b, 0b0110001); // 1, 3.   4 bits apart
  const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  return std::make_pair(
      _mm256_permutevar8x32_ps(a_swapped, group_ctrl),
      _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  auto a_grouped = _mm256_permute4x64_pd(a, 0b11011000); // 0, 2, 1, 3
  auto b_grouped = _mm256_permute4x64_pd(b, 0b11011000); // 0, 2, 1, 3
  return std::make_pair(
      _mm256_permute2f128_pd(
          a_grouped, b_grouped, 0b0100000), // 0, 2.   4 bits apart
      _mm256_permute2f128_pd(
          a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  const __m256i group_ctrl = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  auto a_grouped = _mm256_permutevar8x32_ps(a, group_ctrl);
  auto b_grouped = _mm256_permutevar8x32_ps(b, group_ctrl);
  return std::make_pair(
      _mm256_permute2f128_ps(
          a_grouped, b_grouped, 0b0100000), // 0, 2.   4 bits apart
      _mm256_permute2f128_ps(
          a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

#endif // CPU_CAPABILITY_AVX2

} // namespace tensorplay::vec::inline CPU_CAPABILITY

} // namespace tensorplay::vec
