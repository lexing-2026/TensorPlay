#pragma once

// half (float16) 256-bit vector layer over the shared 16-bit float
// machinery; lane conversion goes through F16C.

#include "cpu/vec/vec256/vec256_16bit_float.h"
#include "irange.h"

#include <cmath>

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <>
struct is_vec_specialized_for<tensorplay::Half> : std::bool_constant<true> {};

template <>
class Vectorized<tensorplay::Half> : public Vectorized16<tensorplay::Half> {
 public:
  using Vectorized16::Vectorized16;

  using value_type = tensorplay::Half;

  Vectorized<tensorplay::Half> frac() const;

  Vectorized<tensorplay::Half> eq(const Vectorized<tensorplay::Half>& other) const;
  Vectorized<tensorplay::Half> ne(const Vectorized<tensorplay::Half>& other) const;
  Vectorized<tensorplay::Half> gt(const Vectorized<tensorplay::Half>& other) const;
  Vectorized<tensorplay::Half> ge(const Vectorized<tensorplay::Half>& other) const;
  Vectorized<tensorplay::Half> lt(const Vectorized<tensorplay::Half>& other) const;
  Vectorized<tensorplay::Half> le(const Vectorized<tensorplay::Half>& other) const;
};

Vectorized<tensorplay::Half> inline operator+(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_add_ps(x, y);
  });
}
Vectorized<tensorplay::Half> inline operator-(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_sub_ps(x, y);
  });
}
Vectorized<tensorplay::Half> inline operator*(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_mul_ps(x, y);
  });
}
Vectorized<tensorplay::Half> inline operator/(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return binary_op_as_fp32(a, b, [](const __m256& x, const __m256& y) {
    return _mm256_div_ps(x, y);
  });
}
Vectorized<tensorplay::Half> inline operator&(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return _mm256_and_si256(a, b);
}
Vectorized<tensorplay::Half> inline operator|(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return _mm256_or_si256(a, b);
}
Vectorized<tensorplay::Half> inline operator^(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  return _mm256_xor_si256(a, b);
}

inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::eq(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this == other) & Vectorized<tensorplay::Half>(1.0f);
}
inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::ne(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this != other) & Vectorized<tensorplay::Half>(1.0f);
}
inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::gt(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this > other) & Vectorized<tensorplay::Half>(1.0f);
}
inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::ge(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this >= other) & Vectorized<tensorplay::Half>(1.0f);
}
inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::lt(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this < other) & Vectorized<tensorplay::Half>(1.0f);
}
inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::le(
    const Vectorized<tensorplay::Half>& other) const {
  return (*this <= other) & Vectorized<tensorplay::Half>(1.0f);
}

inline Vectorized<tensorplay::Half> Vectorized<tensorplay::Half>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<tensorplay::Half> inline maximum(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  auto max_lo = _mm256_max_ps(a_lo, b_lo);
  auto max_hi = _mm256_max_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(max_lo, nan_lo);
  auto o2 = _mm256_or_ps(max_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<tensorplay::Half> inline minimum(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  auto min_lo = _mm256_min_ps(a_lo, b_lo);
  auto min_hi = _mm256_min_ps(a_hi, b_hi);
  auto nan_lo = _mm256_cmp_ps(a_lo, b_lo, _CMP_UNORD_Q);
  auto nan_hi = _mm256_cmp_ps(a_hi, b_hi, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  auto o1 = _mm256_or_ps(min_lo, nan_lo);
  auto o2 = _mm256_or_ps(min_hi, nan_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<tensorplay::Half> inline clamp(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& min,
    const Vectorized<tensorplay::Half>& max) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  __m256 max_lo, max_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, _mm256_max_ps(min_lo, a_lo));
  auto o2 = _mm256_min_ps(max_hi, _mm256_max_ps(min_hi, a_hi));
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<tensorplay::Half> inline clamp_max(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& max) {
  __m256 a_lo, a_hi;
  __m256 max_lo, max_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(max), max_lo, max_hi);
  auto o1 = _mm256_min_ps(max_lo, a_lo);
  auto o2 = _mm256_min_ps(max_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
Vectorized<tensorplay::Half> inline clamp_min(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& min) {
  __m256 a_lo, a_hi;
  __m256 min_lo, min_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(min), min_lo, min_hi);
  auto o1 = _mm256_max_ps(min_lo, a_lo);
  auto o2 = _mm256_max_ps(min_hi, a_hi);
  return cvtfp32_fp16(o1, o2);
}

template <>
inline void convert(const tensorplay::Half* src, tensorplay::Half* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<tensorplay::Half>::size());
       i += Vectorized<tensorplay::Half>::size()) {
    auto vsrc =
        _mm256_loadu_si256(reinterpret_cast<__m256i*>((void*)(src + i)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>((void*)(dst + i)), vsrc);
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
inline void convert(const float* src, tensorplay::Half* dst, int64_t n) {
  int64_t i;
  for (i = 0; i + Vectorized<tensorplay::Half>::size() <= n;
       i += Vectorized<tensorplay::Half>::size()) {
    __m256 a = _mm256_loadu_ps(&src[i]);
    __m256 b = _mm256_loadu_ps(&src[i + 8]);

    __m256i bf = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = tensorplay::detail::scalar_cast<tensorplay::Half>(src[i]);
  }
}

template <>
inline void convert(const double* src, tensorplay::Half* dst, int64_t n) {
  auto load_float = [](const double* src) -> __m256 {
    // Load one float vector from an array of doubles
    __m128 a = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
    __m128 b = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
  };

  int64_t i;
  for (i = 0; i + Vectorized<tensorplay::Half>::size() <= n;
       i += Vectorized<tensorplay::Half>::size()) {
    __m256 a = load_float(&src[i]);
    __m256 b = load_float(&src[i + 8]);

    __m256i bf = cvtfp32_fp16(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[i]), bf);
  }
  for (; i < n; i++) {
    dst[i] = tensorplay::detail::scalar_cast<tensorplay::Half>(src[i]);
  }
}

template <>
Vectorized<tensorplay::Half> inline fmadd(
    const Vectorized<tensorplay::Half>& a,
    const Vectorized<tensorplay::Half>& b,
    const Vectorized<tensorplay::Half>& c) {
  __m256 a_lo, a_hi;
  __m256 b_lo, b_hi;
  __m256 c_lo, c_hi;
  cvtfp16_fp32(__m256i(a), a_lo, a_hi);
  cvtfp16_fp32(__m256i(b), b_lo, b_hi);
  cvtfp16_fp32(__m256i(c), c_lo, c_hi);
  auto o1 = _mm256_fmadd_ps(a_lo, b_lo, c_lo);
  auto o2 = _mm256_fmadd_ps(a_hi, b_hi, c_hi);
  return cvtfp32_fp16(o1, o2);
}

TP_CONVERT_VECTORIZED_INIT(tensorplay::Half, half)
TP_LOAD_FP32_VECTORIZED_INIT(tensorplay::Half, fp16)

#else // defined(CPU_CAPABILITY_AVX2)

TP_LOAD_FP32_NON_VECTORIZED_INIT_FALLBACK(tensorplay::Half, fp16)

#endif // defined(CPU_CAPABILITY_AVX2)
} // namespace CPU_CAPABILITY
} // namespace tensorplay::vec
