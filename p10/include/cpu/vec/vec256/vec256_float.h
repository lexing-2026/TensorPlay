#pragma once

// Port of ATen/cpu/vec/vec256/vec256_float.h with the TensorPlay vec layer.
// Interface matches PyTorch's Vectorized<float>; math functions that
// depend on Sleef fall back to a scalar map (auto-vectorized by the
// compiler at -O3 -mavx2), everything else uses AVX2 intrinsics.

#include <immintrin.h>
#include "cpu/vec/vec_base.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(CPU_CAPABILITY_AVX2) && defined(__GLIBC__)
// PyTorch's AVX2 Vectorized<float> uses Sleef for transcendental functions.
// On glibc systems libmvec exposes the same vector ABI, which gives Stax's
// fused CPU codegen a real vector math implementation instead of falling
// back to eight scalar calls through map().
extern "C" {
__m256 _ZGVdN8v_sinf(__m256);
__m256 _ZGVdN8v_cosf(__m256);
__m256 _ZGVdN8v_expf(__m256);
__m256 _ZGVdN8v_logf(__m256);
__m256 _ZGVdN8v_tanhf(__m256);
}
#endif

#if defined(CPU_CAPABILITY_AVX2)

namespace tensorplay::vec::inline CPU_CAPABILITY {

template <>
struct Vectorized<float> {
 private:
  __m256 values;

 public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(__m256 v) : values(v) {}
  Vectorized(float val) {
    values = _mm256_set1_ps(val);
  }
  Vectorized(
      float val1,
      float val2,
      float val3,
      float val4,
      float val5,
      float val6,
      float val7,
      float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  Vectorized(const float (&arr)[8])
      : Vectorized(
            arr[0],
            arr[1],
            arr[2],
            arr[3],
            arr[4],
            arr[5],
            arr[6],
            arr[7]) {}
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }
  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    return _mm256_blendv_ps(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }
  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    // Masked load: lanes [0, count) are read, the rest are zero.
    const __m256i mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(static_cast<int32_t>(count)),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    return _mm256_maskload_ps(reinterpret_cast<const float*>(ptr), mask);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      // Masked store: only lanes [0, count) are written.
      const __m256i mask = _mm256_cmpgt_epi32(
          _mm256_set1_epi32(static_cast<int32_t>(count)),
          _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
      _mm256_maskstore_ps(reinterpret_cast<float*>(ptr), mask, values);
    }
  }
  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    __m256 cmp = _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
    return _mm256_movemask_ps(cmp);
  }
  Vectorized<float> isnan() const {
    return _mm256_cmp_ps(values, _mm256_set1_ps(0.0f), _CMP_UNORD_Q);
  }

  bool has_inf_nan() const {
    __m256 self_sub = _mm256_sub_ps(values, values);
    return (_mm256_movemask_epi8(_mm256_castps_si256(self_sub)) & 0x77777777) !=
        0;
  }

  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    return _mm256_andnot_ps(
        _mm256_set1_ps(-0.0f), values); // clear sign bit
  }
  Vectorized<float> angle() const {
    __m256 zero = _mm256_set1_ps(0.0f);
    return _mm256_cmp_ps(
        values, zero, _CMP_LT_OQ); // zero for NaN and positive values
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return _mm256_set1_ps(0);
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> acos() const {
    return map(std::acos);
  }
  Vectorized<float> acosh() const {
    return map(std::acosh);
  }
  Vectorized<float> asin() const {
    return map(std::asin);
  }
  Vectorized<float> asinh() const {
    return map(std::asinh);
  }
  Vectorized<float> atan() const {
    return map(std::atan);
  }
  Vectorized<float> atanh() const {
    return map(std::atanh);
  }
  Vectorized<float> atan2(const Vectorized<float>& exp) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    exp.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = std::atan2(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> copysign(const Vectorized<float>& sign) const {
    // clear sign bit of a, and merge with sign bit of b
    return _mm256_or_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values),
        _mm256_and_ps(_mm256_set1_ps(-0.0f), sign));
  }
  Vectorized<float> erf() const {
    return map(std::erf);
  }
  Vectorized<float> erfc() const {
    return map(std::erfc);
  }
  Vectorized<float> exp() const {
#if defined(__GLIBC__)
    return _ZGVdN8v_expf(values);
#else
    return map(std::exp);
#endif
  }
  Vectorized<float> exp2() const {
    return map(std::exp2);
  }
  Vectorized<float> expm1() const {
    return map(std::expm1);
  }
  Vectorized<float> exp_u20() const {
    return map(std::exp);
  }
  Vectorized<float> fexp_u20() const {
    return map(std::exp);
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_q[size()];
    store(tmp);
    q.store(tmp_q);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = std::fmod(tmp[i], tmp_q[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> log() const {
#if defined(__GLIBC__)
    return _ZGVdN8v_logf(values);
#else
    return map(std::log);
#endif
  }
  Vectorized<float> log2() const {
    return map(std::log2);
  }
  Vectorized<float> log10() const {
    return map(std::log10);
  }
  Vectorized<float> log1p() const {
    return map(std::log1p);
  }
  Vectorized<float> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vectorized<float> cos() const {
#if defined(__GLIBC__)
    return _ZGVdN8v_cosf(values);
#else
    return map(std::cos);
#endif
  }
  Vectorized<float> cosh() const {
    return map(std::cosh);
  }
  Vectorized<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    __at_align__ float tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::hypot(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<float> neg() const {
    return _mm256_xor_ps(values, _mm256_set1_ps(-0.0f));
  }
  Vectorized<float> nextafter(const Vectorized<float>& b) const {
    __at_align__ float tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::nextafter(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<float> round() const {
    return _mm256_round_ps(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> sin() const {
#if defined(__GLIBC__)
    return _ZGVdN8v_sinf(values);
#else
    return map(std::sin);
#endif
  }
  Vectorized<float> sinh() const {
    return map(std::sinh);
  }
  Vectorized<float> tan() const {
    return map(std::tan);
  }
  Vectorized<float> tanh() const {
#if defined(__GLIBC__)
    return _ZGVdN8v_tanhf(values);
#else
    return map(std::tanh);
#endif
  }
  Vectorized<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    __at_align__ float tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::pow(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  float reduce_add() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = _mm256_add_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = _mm256_add_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = _mm256_add_ps(v, v1);
    return _mm256_cvtss_f32(v);
  }
  float reduce_max() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = _mm256_max_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = _mm256_max_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = _mm256_max_ps(v, v1);
    return _mm256_cvtss_f32(v);
  }
  float reduce_min() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
    v = _mm256_min_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0x4E);
    v = _mm256_min_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_ps(v, v, 0xB1);
    v = _mm256_min_ps(v, v1);
    return _mm256_cvtss_f32(v);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
  }

  Vectorized<float> frac() const;
  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  Vectorized<float> max = _mm256_max_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  Vectorized<float> min = _mm256_min_ps(a, b);
  Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  return _mm256_min_ps(max, _mm256_max_ps(min, a));
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return _mm256_min_ps(max, a);
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return _mm256_max_ps(min, a);
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm256_xor_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::eq(
    const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(
    const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(
    const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(
    const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(
    const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(
    const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
Vectorized<float> inline fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm256_fnmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm256_fmsub_ps(a, b, c);
}

template <>
Vectorized<float> inline fnmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm256_fnmsub_ps(a, b, c);
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY

#endif // CPU_CAPABILITY_AVX2
