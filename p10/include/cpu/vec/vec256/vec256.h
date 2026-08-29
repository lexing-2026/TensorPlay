#pragma once

// float/double/int32/int64 specializations. AVX512 and reduced-precision
// types can be added later following the same pattern.

#include "cpu/vec/vec_base.h"

#include "cpu/vec/vec256/vec256_float.h"
#include "cpu/vec/vec256/vec256_double.h"
#include "cpu/vec/vec256/vec256_int.h"

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
