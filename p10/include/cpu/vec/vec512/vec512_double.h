#pragma once

// 512-bit double vector layer (same interface as the 256-bit one).
// Transcendental methods dispatch to the vendored SLEEF vector math
// (see cpu/vec/SleefShims.h); the remaining primitives use AVX-512
// intrinsics directly.

#include <immintrin.h>
#include "cpu/vec/vec_base.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>


#if defined(CPU_CAPABILITY_AVX512)

namespace tensorplay::vec::inline CPU_CAPABILITY {

// Mask->vector widening for the comparison operator set (see vec512_float).
inline __m512d widen(__mmask8 k) {
  return _mm512_castsi512_pd(
      _mm512_maskz_mov_epi64(k, _mm512_set1_epi64(-1)));
}

template <>
struct Vectorized<double> {
 private:
  __m512d values;

 public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(__m512d v) : values(v) {}
  Vectorized(double val) {
    values = _mm512_set1_pd(val);
  }
  Vectorized(
      double val1,
      double val2,
      double val3,
      double val4,
      double val5,
      double val6,
      double val7,
      double val8) {
    values =
        _mm512_setr_pd(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  Vectorized(const double (&arr)[8])
      : Vectorized(
            arr[0],
            arr[1],
            arr[2],
            arr[3],
            arr[4],
            arr[5],
            arr[6],
            arr[7]) {}
  operator __m512d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    return _mm512_mask_blend_pd(
        static_cast<__mmask8>(mask), a.values, b.values);
  }
  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    return _mm512_mask_blend_pd(
        _mm512_movepi64_mask(_mm512_castpd_si512(mask.values)),
        a.values,
        b.values);
  }
  template <typename step_t>
  static Vectorized<double> arange(
      double base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
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
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size())
      return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));
    if (count <= 0)
      return _mm512_setzero_pd();
    // Masked load: lanes [0, count) are read, the rest are zero.
    const __mmask8 mask =
        static_cast<__mmask8>((static_cast<uint8_t>(1) << count) - 1);
    return _mm512_maskz_loadu_pd(mask, reinterpret_cast<const double*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _mm512_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      // Masked store: only lanes [0, count) are written.
      const __mmask8 mask =
          static_cast<__mmask8>((static_cast<uint8_t>(1) << count) - 1);
      _mm512_mask_storeu_pd(reinterpret_cast<double*>(ptr), mask, values);
    }
  }
  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    return static_cast<int>(
        _mm512_cmp_pd_mask(values, _mm512_set1_pd(0.0), _CMP_EQ_OQ));
  }
  Vectorized<double> isnan() const {
    return widen(_mm512_cmp_pd_mask(values, _mm512_set1_pd(0.0), _CMP_UNORD_Q));
  }

  bool has_inf_nan() const {
    // inf - inf == NaN and NaN - NaN == NaN, so an unordered self-subtraction
    // identifies exactly the lanes holding inf or NaN.
    __m512d self_sub = _mm512_sub_pd(values, values);
    return _mm512_cmp_pd_mask(self_sub, self_sub, _CMP_UNORD_Q) != 0;
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
    return _mm512_andnot_pd(
        _mm512_set1_pd(-0.0), values); // clear sign bit
  }
  Vectorized<double> angle() const {
    const auto zero_vec = _mm512_set1_pd(0.0);
    const auto nan_vec = _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN());
    const auto nan_mask = _mm512_cmp_pd_mask(values, values, _CMP_UNORD_Q);
    const auto pi = _mm512_set1_pd(3.141592653589793238463);
    const auto neg_mask = _mm512_cmp_pd_mask(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm512_mask_blend_pd(neg_mask, zero_vec, pi);
    angle = _mm512_mask_blend_pd(nan_mask, angle, nan_vec);
    return angle;
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return _mm512_set1_pd(0);
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
    return _mm512_or_pd(
        _mm512_andnot_pd(_mm512_set1_pd(-0.0), values),
        _mm512_and_pd(_mm512_set1_pd(-0.0), sign));
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
    return _mm512_roundscale_pd(
        values, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<double> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<double> floor() const {
    return _mm512_roundscale_pd(
        values, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<double> neg() const {
    return _mm512_xor_pd(values, _mm512_set1_pd(-0.0));
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
    return _mm512_roundscale_pd(
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
    return _mm512_roundscale_pd(
        values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
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
    return _mm512_sqrt_pd(values);
  }
  Vectorized<double> reciprocal() const {
    return _mm512_div_pd(_mm512_set1_pd(1), values);
  }
  Vectorized<double> rsqrt() const {
    return _mm512_div_pd(_mm512_set1_pd(1), _mm512_sqrt_pd(values));
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  double reduce_add() const {
    auto v = values;
    // 256-bit shuffle: sum the two 256-bit halves
    auto v1 = _mm512_shuffle_f64x2(v, v, 0x4E);
    v = _mm512_add_pd(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f64x2(v, v, 0xB1);
    v = _mm512_add_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_pd(v, v, 0x55);
    v = _mm512_add_pd(v, v1);
    return _mm256_cvtsd_f64(_mm512_castpd512_pd256(v));
  }
  double reduce_max() const {
    auto v = values;
    // 256-bit shuffle: max the two 256-bit halves
    auto v1 = _mm512_shuffle_f64x2(v, v, 0x4E);
    v = _mm512_max_pd(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f64x2(v, v, 0xB1);
    v = _mm512_max_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_pd(v, v, 0x55);
    v = _mm512_max_pd(v, v1);
    return _mm256_cvtsd_f64(_mm512_castpd512_pd256(v));
  }
  double reduce_min() const {
    auto v = values;
    // 256-bit shuffle: min the two 256-bit halves
    auto v1 = _mm512_shuffle_f64x2(v, v, 0x4E);
    v = _mm512_min_pd(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f64x2(v, v, 0xB1);
    v = _mm512_min_pd(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_pd(v, v, 0x55);
    v = _mm512_min_pd(v, v1);
    return _mm256_cvtsd_f64(_mm512_castpd512_pd256(v));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_EQ_OQ));
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_NEQ_UQ));
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_LT_OQ));
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_LE_OQ));
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_GT_OQ));
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return widen(_mm512_cmp_pd_mask(values, other.values, _CMP_GE_OQ));
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
  return _mm512_add_pd(a, b);
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_sub_pd(a, b);
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_mul_pd(a, b);
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_div_pd(a, b);
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
  Vectorized<double> max = _mm512_max_pd(a, b);
  Vectorized<double> nan_lanes =
      widen(_mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_pd(max, nan_lanes);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  Vectorized<double> min = _mm512_min_pd(a, b);
  Vectorized<double> nan_lanes =
      widen(_mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_pd(min, nan_lanes);
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  return _mm512_min_pd(max, _mm512_max_pd(min, a));
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return _mm512_min_pd(max, a);
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return _mm512_max_pd(min, a);
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_and_pd(a, b);
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_or_pd(a, b);
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return _mm512_xor_pd(a, b);
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
    _mm512_storeu_pd(dst + i, _mm512_loadu_pd(src + i));
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
  return _mm512_fmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm512_fnmadd_pd(a, b, c);
}

template <>
Vectorized<double> inline fmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm512_fmsub_pd(a, b, c);
}

template <>
Vectorized<double> inline fnmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return _mm512_fnmsub_pd(a, b, c);
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY

#endif // CPU_CAPABILITY_AVX512
