#pragma once

// AVX-512 lane conversions over the VecConvert customization points:
// packed 16-bit float <-> float, float <-> int64, int64 <-> int32, and
// 8-bit integer <-> reduced-precision float paths.

#include "cpu/vec/intrinsics.h"
#include "cpu/vec/vec_base.h"
#include "cpu/vec/vec_convert.h"
#include "cpu/vec/vec512/vec512_bfloat16.h"

#include <cmath>
#include <tuple>

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

// float -> 8-bit integer lane conversion.  Rounding variant scales to
// nearest before the truncating store; saturating semantics keep the
// result inside [-128, 127] / [0, 255].
template <typename dst_t, bool round = false>
inline Vectorized<dst_t> tp512_convert_float_to_int8(const Vectorized<float>& src) {
  __at_align__ float tmp[16];
  __at_align__ dst_t res[16];
  src.store(tmp);
  if constexpr (round) {
    for (int64_t i = 0; i < 16; i++) {
      res[i] = static_cast<dst_t>(std::nearbyint(tmp[i]));
    }
  } else {
    for (int64_t i = 0; i < 16; i++) {
      res[i] = static_cast<dst_t>(tmp[i]);
    }
  }
  return _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(res)));
}

// 8-bit integer -> float lane conversion.
template <typename src_t>
inline Vectorized<float> tp512_convert_int8_to_float(const Vectorized<src_t>& src) {
  __at_align__ src_t tmp[16];
  __at_align__ float res[16];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), _mm512_castsi512_si256(src));
  for (int64_t i = 0; i < 16; i++) {
    res[i] = static_cast<float>(tmp[i]);
  }
  return _mm512_loadu_ps(res);
}

template <>
struct VecConvert<float, 1, tensorplay::BFloat16, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<tensorplay::BFloat16, 1>& src) {
    VectorizedN<float, 1> result;
    __m512 value;
    cvtbf16_fp32(_mm512_castsi512_si256(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
struct VecConvert<float, 1, tensorplay::Half, 1> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<tensorplay::Half, 1>& src) {
    VectorizedN<float, 1> result;
    __m512 value;
    cvtfp16_fp32(_mm512_castsi512_si256(src[0]), value);
    result[0] = value;
    return result;
  }
};

template <>
struct VecConvert<tensorplay::BFloat16, 1, float, 1> {
  static inline VectorizedN<tensorplay::BFloat16, 1> apply(
      const VectorizedN<float, 1>& src) {
    VectorizedN<tensorplay::BFloat16, 1> result;
    result[0] = _mm512_castsi256_si512(cvtfp32_bf16(src[0]));
    return result;
  }
};

template <>
struct VecConvert<tensorplay::BFloat16, 1, float, 2> {
  static inline VectorizedN<tensorplay::BFloat16, 1> apply(
      const VectorizedN<float, 2>& src) {
    VectorizedN<tensorplay::BFloat16, 1> result;
    result[0] = convert_float_bfloat16(src[0], src[1]);
    return result;
  }
};

template <>
struct VecConvert<float, 2, tensorplay::BFloat16, 1> {
  static inline VectorizedN<float, 2> apply(
      const VectorizedN<tensorplay::BFloat16, 1>& src) {
    VectorizedN<float, 2> result;
    std::tie(result[0], result[1]) = convert_bfloat16_float(src[0]);
    return result;
  }
};

template <>
struct VecConvert<tensorplay::Half, 1, float, 1> {
  static inline VectorizedN<tensorplay::Half, 1> apply(const VectorizedN<float, 1>& src) {
    VectorizedN<tensorplay::Half, 1> result;
    result[0] = _mm512_castsi256_si512(cvtfp32_fp16(src[0]));
    return result;
  }
};

template <>
struct VecConvert<tensorplay::Half, 1, float, 2> {
  static inline VectorizedN<tensorplay::Half, 1> apply(const VectorizedN<float, 2>& src) {
    VectorizedN<tensorplay::Half, 1> result;
    result[0] = convert_float_half(src[0], src[1]);
    return result;
  }
};

template <>
struct VecConvert<float, 2, tensorplay::Half, 1> {
  static inline VectorizedN<float, 2> apply(const VectorizedN<tensorplay::Half, 1>& src) {
    VectorizedN<float, 2> result;
    std::tie(result[0], result[1]) = convert_half_float(src[0]);
    return result;
  }
};

template <>
struct VecConvert<float, 1, int64_t, 2> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low = _mm512_cvtepi64_ps(src[0]);
    auto high = _mm512_cvtepi64_ps(src[1]);
    return Vectorized<float>(
        _mm512_insertf32x8(_mm512_castps256_ps512(low), high, 1));
  }
};

template <>
struct VecConvert<int64_t, 2, float, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<float, 1>& src) {
    tensorplay::vec::VectorizedN<int64_t, 2> result;
    result[0] = _mm512_cvt_roundps_epi64(
        _mm512_castps512_ps256(src[0]), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    result[1] = _mm512_cvt_roundps_epi64(
        _mm512_extractf32x8_ps(src[0], 1),
        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    return result;
  }
};

template <>
struct VecConvert<int32_t, 1, int64_t, 2> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    auto low = _mm512_cvtepi64_epi32(src[0]);
    auto high = _mm512_cvtepi64_epi32(src[1]);
    return Vectorized<int32_t>(
        _mm512_inserti32x8(_mm512_castsi256_si512(low), high, 1));
  }
};

template <>
struct VecConvert<int64_t, 2, int32_t, 1> {
  static inline VectorizedN<int64_t, 2> apply(
      const VectorizedN<int32_t, 1>& src) {
    tensorplay::vec::VectorizedN<int64_t, 2> result;
    result[0] = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(src[0]));
    result[1] = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(src[0], 1));
    return result;
  }
};

template <>
struct VecConvert<int32_t, 1, int8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<int8_t, 1>& src) {
    auto src128 = _mm512_castsi512_si128(src[0]);
    return Vectorized<int32_t>(_mm512_cvtepi8_epi32(src128));
  }
};

template <>
struct VecConvert<int32_t, 1, uint8_t, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    auto src128 = _mm512_castsi512_si128(src[0]);
    return Vectorized<int32_t>(_mm512_cvtepu8_epi32(src128));
  }
};

template <>
struct VecConvert<int32_t, 1, float, 1> {
  static inline VectorizedN<int32_t, 1> apply(
      const VectorizedN<float, 1>& src) {
    return Vectorized<int32_t>(_mm512_cvttps_epi32(src[0]));
  }
};

template <>
struct VecConvert<float, 1, int32_t, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<int32_t, 1>& src) {
    return Vectorized<float>(_mm512_cvtepi32_ps(src[0]));
  }
};

template <>
struct VecConvert<int16_t, 1, uint8_t, 1> {
  static inline VectorizedN<int16_t, 1> apply(
      const VectorizedN<uint8_t, 1>& src) {
    auto src256 = _mm512_castsi512_si256(src[0]);
    return Vectorized<int16_t>(_mm512_cvtepu8_epi16(src256));
  }
};

template <>
struct VecConvert<int8_t, 1, int32_t, 1> {
  static inline VectorizedN<int8_t, 1> apply(
      const VectorizedN<int32_t, 1>& src) {
    auto src128 = _mm512_cvtepi32_epi8(src[0]);
    return Vectorized<int8_t>(_mm512_castsi128_si512(src128));
  }
};

template <>
struct VecConvert<int8_t, 1, int16_t, 1> {
  static inline VectorizedN<int8_t, 1> apply(
      const VectorizedN<int16_t, 1>& src) {
    auto src256 = _mm512_cvtepi16_epi8(src[0]);
    return Vectorized<int8_t>(_mm512_castsi256_si512(src256));
  }
};

template <typename dst_t, typename src_t>
struct VecConvert<
    dst_t,
    1,
    src_t,
    1,
    typename std::enable_if_t<
        (is_reduced_floating_point_v<dst_t> && is_8bit_integer_v<src_t>) ||
            (is_reduced_floating_point_v<src_t> && is_8bit_integer_v<dst_t>),
        void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<src_t, 1>& src) {
    VectorizedN<float, 2> tmp_fp32 = VecConvert<float, 2, src_t, 1>::apply(src);
    return VecConvert<dst_t, 1, float, 2>::apply(tmp_fp32);
  }
};

template <typename dst_t>
struct VecRoundConvert<
    dst_t,
    1,
    float,
    2,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 2>& src) {
    tensorplay::vec::Vectorized<dst_t> vec1 =
        tp512_convert_float_to_int8<dst_t, /*round=*/true>(src[0]);
    tensorplay::vec::Vectorized<dst_t> vec2 =
        tp512_convert_float_to_int8<dst_t, /*round=*/true>(src[1]);
    __m128i vec2_lo = _mm512_castsi512_si128(vec2);
    __m512i out = _mm512_inserti32x4(vec1, vec2_lo, 1);
    return VectorizedN<dst_t, 1>(tensorplay::vec::Vectorized<dst_t>(out));
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    2,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 2>& src) {
    tensorplay::vec::Vectorized<dst_t> vec1 = tp512_convert_float_to_int8<dst_t>(src[0]);
    tensorplay::vec::Vectorized<dst_t> vec2 = tp512_convert_float_to_int8<dst_t>(src[1]);
    __m128 lane2 = _mm512_castps512_ps128(_mm512_castsi512_ps(vec2));
    __m512 result = _mm512_insertf32x4(
        _mm512_castsi512_ps(vec1),
        lane2,
        1); // Insert lane2 into the second 128-bit lane
    return tensorplay::vec::Vectorized<dst_t>(_mm512_castps_si512(result));
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    return tp512_convert_float_to_int8<dst_t>(src[0]);
  }
};

template <typename dst_t>
struct VecRoundConvert<
    dst_t,
    1,
    float,
    1,
    typename std::enable_if_t<is_8bit_integer_v<dst_t>, void>> {
  static inline VectorizedN<dst_t, 1> apply(const VectorizedN<float, 1>& src) {
    return tp512_convert_float_to_int8<dst_t, /*round=*/true>(src[0]);
  }
};

template <typename src_t>
struct VecConvert<
    float,
    2,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>, void>> {
  static inline VectorizedN<float, 2> apply(const VectorizedN<src_t, 1>& src) {
    __m512i src2 =
        _mm512_castsi128_si512(_mm_castps_si128(_mm512_extractf32x4_ps(
            _mm512_castsi512_ps(src[0]), 1) // Extract the second 128-bit lane
                                                ));
    return VectorizedN<float, 2>(
        tp512_convert_int8_to_float<src_t>(src[0]),
        tp512_convert_int8_to_float<src_t>(src2));
  }
};

template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>, void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    return tp512_convert_int8_to_float<src_t>(src[0]);
  }
};

template <typename dst_t>
struct VecConvert<
    dst_t,
    1,
    int64_t,
    2,
    std::enable_if_t<
        std::is_same_v<dst_t, int8_t> || std::is_same_v<dst_t, uint8_t>>> {
  static inline VectorizedN<dst_t, 1> apply(
      const VectorizedN<int64_t, 2>& src) {
    return VecConvert<dst_t, 1, int32_t, 1>::apply(
        VecConvert<int32_t, 1, int64_t, 2>::apply(src));
  }
};

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

} // namespace CPU_CAPABILITY
} // namespace tensorplay::vec
