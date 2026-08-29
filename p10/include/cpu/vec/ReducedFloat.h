#pragma once

#include "BFloat16.h"
#include "Half.h"
#include "cpu/vec/vec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

template <typename Scalar>
inline std::pair<Vectorized<float>, Vectorized<float>>
load_reduced_float_pair(const Scalar* data) {
  alignas(64) std::array<float, 2 * Vectorized<float>::size()> values{};
  for (int64_t i = 0; i < 2 * Vectorized<float>::size(); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(data[i]);
  }
  return {
      Vectorized<float>::loadu(values.data()),
      Vectorized<float>::loadu(values.data() + Vectorized<float>::size())};
}

template <typename Scalar>
inline Vectorized<float> load_reduced_float(const Scalar* data) {
  alignas(64) std::array<float, Vectorized<float>::size()> values{};
  for (int64_t i = 0; i < Vectorized<float>::size(); ++i) {
    values[static_cast<std::size_t>(i)] = static_cast<float>(data[i]);
  }
  return Vectorized<float>::loadu(values.data());
}

#if defined(CPU_CAPABILITY_AVX2)

inline std::pair<Vectorized<float>, Vectorized<float>>
load_reduced_float_pair(const BFloat16* data) {
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
  const __m128i low = _mm256_castsi256_si128(packed);
  const __m128i high = _mm256_extractf128_si256(packed, 1);
  const __m256 low_fp32 = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(low), 16));
  const __m256 high_fp32 = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(high), 16));
  return {Vectorized<float>(low_fp32), Vectorized<float>(high_fp32)};
}

inline Vectorized<float> load_reduced_float(const BFloat16* data) {
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
  const __m256 result = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
  return Vectorized<float>(result);
}

inline std::pair<Vectorized<float>, Vectorized<float>>
load_reduced_float_pair(const Half* data) {
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
  const __m128i low = _mm256_castsi256_si128(packed);
  const __m128i high = _mm256_extractf128_si256(packed, 1);
  return {
      Vectorized<float>(_mm256_cvtph_ps(low)),
      Vectorized<float>(_mm256_cvtph_ps(high))};
}

inline Vectorized<float> load_reduced_float(const Half* data) {
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
  return Vectorized<float>(_mm256_cvtph_ps(packed));
}

#elif defined(CPU_CAPABILITY_AVX512)

inline std::pair<Vectorized<float>, Vectorized<float>>
load_reduced_float_pair(const BFloat16* data) {
  const __m512i packed =
      _mm512_loadu_si512(reinterpret_cast<const void*>(data));
  const __m256i low = _mm512_castsi512_si256(packed);
  const __m256i high = _mm512_extracti64x4_epi64(packed, 1);
  const __m512 low_fp32 = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(low), 16));
  const __m512 high_fp32 = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(high), 16));
  return {Vectorized<float>(low_fp32), Vectorized<float>(high_fp32)};
}

inline Vectorized<float> load_reduced_float(const BFloat16* data) {
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
  const __m512 result = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
  return Vectorized<float>(result);
}

inline std::pair<Vectorized<float>, Vectorized<float>>
load_reduced_float_pair(const Half* data) {
  const __m512i packed =
      _mm512_loadu_si512(reinterpret_cast<const void*>(data));
  const __m256i low = _mm512_castsi512_si256(packed);
  const __m256i high = _mm512_extracti64x4_epi64(packed, 1);
  return {
      Vectorized<float>(_mm512_cvtph_ps(low)),
      Vectorized<float>(_mm512_cvtph_ps(high))};
}

inline Vectorized<float> load_reduced_float(const Half* data) {
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
  return Vectorized<float>(_mm512_cvtph_ps(packed));
}

#endif

} // namespace CPU_CAPABILITY
} // namespace tensorplay::vec
