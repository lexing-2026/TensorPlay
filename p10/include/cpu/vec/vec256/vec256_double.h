#pragma once

// Port of ATen/cpu/vec/vec256/vec256_double.h with the TensorPlay vec layer.
// Interface matches PyTorch's Vectorized<double>; math functions that
// depend on Sleef fall back to a scalar map, everything else uses AVX2.

#include <immintrin.h>
#include "cpu/vec/vec_base.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(CPU_CAPABILITY_AVX2)

namespace tensorplay::vec::inline CPU_CAPABILITY {

template <>
struct Vectorized<double> {
 private:
  __m256d values;

 public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(__m256d v) : values(v) {}
  Vectorized(double val) {
    values = _mm256_set1_pd(val);
  }
  Vectorized(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }
  Vectorized(const double (&arr)[4])
      : Vectorized(arr[0], arr[1], arr[2], arr[3]) {}
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<double> arange(
      double base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(
        base, base + step, base + 2 * step, base + 3 * step);
  }
  static Vectorized<double> set(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
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
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));
    // Masked load: lanes [0, count) are read, the rest are zero.
    const __m256i mask = _mm256_cmpgt_epi64(
        _mm256_set1_epi64x(count), _mm256_setr_epi64x(0, 1, 2, 3));
    return _mm256_maskload_pd(reinterpret_cast<const double*>(ptr), mask);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      // Masked store: only lanes [0, count) are written.
      const __m256i mask = _mm256_cmpgt_epi64(
          _mm256_set1_epi64x(count), _mm256_setr_epi64x(0, 1, 2, 3));
      _mm256_maskstore_pd(reinterpret_cast<double*>(ptr), mask, values);
    }
  }
  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    __m256d cmp = _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
    return _mm256_movemask_pd(cmp);
  }
  Vectorized<double> isnan() const {
    return _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_UNORD_Q);
  }

  bool has_inf_nan() const {
    __m256d self_sub = _mm256_sub_pd(values, values);
    return (_mm256_movemask_epi8(_mm256_castpd_si256(self_sub)) & 0x88888888) !=
        0;
  }

  Vectorized<double> map(double (*const f)(double)) const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> abs() const {
    return _mm256_andnot_pd(
        _mm256_set1_pd(-0.0), values); // clear sign bit
  }
  Vectorized<double> angle() const {
    __m256d zero = _mm256_set1_pd(0.0);
    return _mm256_cmp_pd(
        values, zero, _CMP_LT_OQ); // zero for NaN and positive values
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return _mm256_set1_pd(0);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> acos() const {
    return map(std::acos);
  }
  Vectorized<double> acosh() const {
    return map(std::acosh);
  }
  Vectorized<double> asin() const {
    return map(std::asin);
  }
  Vectorized<double> asinh() const {
    return map(std::asinh);
  }
  Vectorized<double> atan() const {
    return map(std::atan);
  }
  Vectorized<double> atanh() const {
    return map(std::atanh);
  }
  Vectorized<double> atan2(const Vectorized<double>& exp) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    exp.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = std::atan2(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> copysign(const Vectorized<double>& sign) const {
    // clear sign bit of a, and merge with sign bit of b
    return _mm256_or_pd(
        _mm256_andnot_pd(_mm256_set1_pd(-0.0), values),
        _mm256_and_pd(_mm256_set1_pd(-0.0), sign));
  }
  Vectorized<double> erf() const {
    return map(std::erf);
  }
  Vectorized<double> erfc() const {
    return map(std::erfc);
  }
  Vectorized<double> exp() const {
    return map(std::exp);
  }
  Vectorized<double> exp2() const {
    return map(std::exp2);
  }
  Vectorized<double> expm1() const {
    return map(std::expm1);
  }
  Vectorized<double> exp_u20() const {
    return map(std::exp);
  }
  Vectorized<double> fexp_u20() const {
    return map(std::exp);
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_q[size()];
    store(tmp);
    q.store(tmp_q);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = std::fmod(tmp[i], tmp_q[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> log() const {
    return map(std::log);
  }
  Vectorized<double> log2() const {
    return map(std::log2);
  }
  Vectorized<double> log10() const {
    return map(std::log10);
  }
  Vectorized<double> log1p() const {
    return map(std::log1p);
  }
  Vectorized<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vectorized<double> cos() const {
    return map(std::cos);
  }
  Vectorized<double> cosh() const {
    return map(std::cosh);
  }
  Vectorized<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    __at_align__ double tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::hypot(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<double> neg() const {
    return _mm256_xor_pd(values, _mm256_set1_pd(-0.0));
  }
  Vectorized<double> nextafter(const Vectorized<double>& b) const {
    __at_align__ double tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::nextafter(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<double> round() const {
    return _mm256_round_pd(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> sin() const {
    return map(std::sin);
  }
  Vectorized<double> sinh() const {
    return map(std::sinh);
  }
  Vectorized<double> tan() const {
    return map(std::tan);
  }
  Vectorized<double> tanh() const {
    return map(std::tanh);
  }
  Vectorized<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
  Vectorized<double> reciprocal() const {
    return _mm256_div_pd(_mm256_set1_pd(1), values);
  }
  Vectorized<double> rsqrt() const {
    return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    __at_align__ double tmp[kSize], tmp_y[kSize], tmp_result[kSize];
    store(tmp);
    b.store(tmp_y);
    for (int64_t i = 0; i < kSize; i++) {
      tmp_result[i] = std::pow(tmp[i], tmp_y[i]);
    }
    return loadu(tmp_result);
  }
  double reduce_add() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_pd(v, v, 0x1);
    v = _mm256_add_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_pd(v, v, 0x5);
    v = _mm256_add_pd(v, v1);
    return _mm256_cvtsd_f64(v);
  }
  double reduce_max() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_pd(v, v, 0x1);
    v = _mm256_max_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_pd(v, v, 0x5);
    v = _mm256_max_pd(v, v1);
    return _mm256_cvtsd_f64(v);
  }
  double reduce_min() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_pd(v, v, 0x1);
    v = _mm256_min_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_pd(v, v, 0x5);
    v = _mm256_min_pd(v, v1);
    return _mm256_cvtsd_f64(v);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
  }

  Vectorized<double> frac() const;
  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_add_pd(a, b);
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  Vectorized<double> max = _mm256_max_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  Vectorized<double> min = _mm256_min_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(min, isnan);
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  return _mm256_min_pd(max, _mm256_max_pd(min, a));
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return _mm256_min_pd(max, a);
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return _mm256_max_pd(min, a);
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm256_xor_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::eq(
    const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ne(
    const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(
    const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(
    const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(
    const Vectorized<double>& other) const {
  return (*this < other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::le(
    const Vectorized<double>& other) const {
  return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

template <>
Vectorized<double> inline fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm256_fnmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm256_fmsub_pd(a, b, c);
}

template <>
Vectorized<double> inline fnmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm256_fnmsub_pd(a, b, c);
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY

#endif // CPU_CAPABILITY_AVX2
