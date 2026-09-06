#pragma once

// AVX2 double vector layer.  Transcendental methods dispatch to the
// vendored SLEEF vector math (see cpu/vec/SleefShims.h); the remaining
// primitives use AVX2 intrinsics directly.

// x86-64 intrinsics only: the AVX specializations below are guarded by
// CPU_CAPABILITY_AVX2/AVX512, and other architectures fall back to the
// generic Vectorized template in vec_base.h.
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
#include <immintrin.h>
#endif
#include "cpu/vec/vec_base.h"
#include "cpu/vec/SleefShims.h"


#include "cpu/SpecialMath.h"

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
    // inf/NaN self-subtract to NaN, whose top exponent byte differs from
    // zero; the sign byte stays masked out so a quiet NaN (positive sign)
    // is still detected.
    return (_mm256_movemask_epi8(_mm256_castpd_si256(self_sub)) & 0x77777777) !=
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
    const auto zero_vec = _mm256_set1_pd(0.0);
    const auto nan_vec = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    const auto not_nan_mask = _mm256_cmp_pd(values, values, _CMP_EQ_OQ);
    const auto nan_mask = _mm256_cmp_pd(not_nan_mask, zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm256_set1_pd(3.141592653589793238463);
    const auto neg_mask = _mm256_cmp_pd(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm256_blendv_pd(zero_vec, pi, neg_mask);
    angle = _mm256_blendv_pd(angle, nan_vec, nan_mask);
    return angle;
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
    return tensorplay::tpsleef::acos(values);
  }
  Vectorized<double> acosh() const {
    return tensorplay::tpsleef::acosh(values);
  }
  Vectorized<double> asin() const {
    return tensorplay::tpsleef::asin(values);
  }
  Vectorized<double> asinh() const {
    return tensorplay::tpsleef::asinh(values);
  }
  Vectorized<double> atan() const {
    return tensorplay::tpsleef::atan(values);
  }
  Vectorized<double> atanh() const {
    return tensorplay::tpsleef::atanh(values);
  }
  Vectorized<double> atan2(const Vectorized<double>& exp) const {
    return tensorplay::tpsleef::atan2(values, exp.values);
  }
  Vectorized<double> copysign(const Vectorized<double>& sign) const {
    // clear sign bit of a, and merge with sign bit of b
    return _mm256_or_pd(
        _mm256_andnot_pd(_mm256_set1_pd(-0.0), values),
        _mm256_and_pd(_mm256_set1_pd(-0.0), sign));
  }
  Vectorized<double> erf() const {
    return tensorplay::tpsleef::erf(values);
  }
  Vectorized<double> erfc() const {
    return tensorplay::tpsleef::erfc(values);
  }
  Vectorized<double> exp() const {
    return tensorplay::tpsleef::exp(values);
  }
  Vectorized<double> exp2() const {
    return tensorplay::tpsleef::exp2(values);
  }
  Vectorized<double> expm1() const {
    return tensorplay::tpsleef::expm1(values);
  }
  Vectorized<double> exp_u20() const {
    return tensorplay::tpsleef::exp(values);
  }
  Vectorized<double> fexp_u20() const {
    return tensorplay::tpsleef::exp(values);
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
    return tensorplay::tpsleef::log(values);
  }
  Vectorized<double> log2() const {
    return tensorplay::tpsleef::log2(values);
  }
  Vectorized<double> log10() const {
    return tensorplay::tpsleef::log10(values);
  }
  Vectorized<double> log1p() const {
    return tensorplay::tpsleef::log1p(values);
  }
  Vectorized<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vectorized<double> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<double> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
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
    return tensorplay::tpsleef::sin(values);
  }
  Vectorized<double> sinh() const {
    return tensorplay::tpsleef::sinh(values);
  }
  Vectorized<double> tan() const {
    return tensorplay::tpsleef::tan(values);
  }
  Vectorized<double> tanh() const {
    return tensorplay::tpsleef::tanh(values);
  }
  Vectorized<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<double> digamma() const {
    return map([](double v) { return calc_digamma(v); });
  }
  Vectorized<double> erfinv() const {
    return map([](double v) { return calc_erfinv(v); });
  }
  Vectorized<double> igamma(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp); x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) tmp[i] = calc_igamma(static_cast<double>(tmp[i]), static_cast<double>(tmp_x[i]));
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp); x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) tmp[i] = calc_igammac(static_cast<double>(tmp[i]), static_cast<double>(tmp_x[i]));
    return loadu(tmp);
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
    return tensorplay::tpsleef::pow(values, b.values);
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
