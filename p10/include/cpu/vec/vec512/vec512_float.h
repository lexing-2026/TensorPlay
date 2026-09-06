#pragma once

// 512-bit float vector layer (same interface as the 256-bit one).
// Transcendental methods dispatch to the vendored SLEEF vector math
// (runtime-dispatched entry points, see cpu/vec/SleefShims.h); the
// remaining primitives use AVX-512 intrinsics directly.

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


#if defined(CPU_CAPABILITY_AVX512)

namespace tensorplay::vec::inline CPU_CAPABILITY {

// GCC's AVX512 compare intrinsics expose only the __mmask16 forms; widen a
// lane mask into the all-ones/zero vector shape the operator set returns.
inline __m512 widen(__mmask16 k) {
  return _mm512_castsi512_ps(
      _mm512_maskz_mov_epi32(k, _mm512_set1_epi32(-1)));
}

template <>
struct Vectorized<float> {
 private:
  __m512 values;

 public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type kSize = 16;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(__m512 v) : values(v) {}
  Vectorized(float val) {
    values = _mm512_set1_ps(val);
  }
  Vectorized(
      float val1,
      float val2,
      float val3,
      float val4,
      float val5,
      float val6,
      float val7,
      float val8,
      float val9,
      float val10,
      float val11,
      float val12,
      float val13,
      float val14,
      float val15,
      float val16) {
    values = _mm512_setr_ps(
        val1,
        val2,
        val3,
        val4,
        val5,
        val6,
        val7,
        val8,
        val9,
        val10,
        val11,
        val12,
        val13,
        val14,
        val15,
        val16);
  }
  Vectorized(const float (&arr)[16])
      : Vectorized(
            arr[0],
            arr[1],
            arr[2],
            arr[3],
            arr[4],
            arr[5],
            arr[6],
            arr[7],
            arr[8],
            arr[9],
            arr[10],
            arr[11],
            arr[12],
            arr[13],
            arr[14],
            arr[15]) {}
  operator __m512() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    return _mm512_mask_blend_ps(
        static_cast<__mmask16>(mask), a.values, b.values);
  }
  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    return _mm512_mask_blend_ps(
        _mm512_movepi32_mask(_mm512_castps_si512(mask.values)),
        a.values,
        b.values);
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
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
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
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size())
      return _mm512_loadu_ps(reinterpret_cast<const float*>(ptr));
    if (count <= 0)
      return _mm512_setzero_ps();
    // Masked load: lanes [0, count) are read, the rest are zero.
    const __mmask16 mask =
        static_cast<__mmask16>((static_cast<uint32_t>(1) << count) - 1);
    return _mm512_maskz_loadu_ps(mask, reinterpret_cast<const float*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _mm512_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      // Masked store: only lanes [0, count) are written.
      const __mmask16 mask =
          static_cast<__mmask16>((static_cast<uint32_t>(1) << count) - 1);
      _mm512_mask_storeu_ps(reinterpret_cast<float*>(ptr), mask, values);
    }
  }
  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    return static_cast<int>(
        _mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0f), _CMP_EQ_OQ));
  }
  Vectorized<float> isnan() const {
    return widen(_mm512_cmp_ps_mask(values, _mm512_set1_ps(0.0f), _CMP_UNORD_Q));
  }

  bool has_inf_nan() const {
    // inf - inf == NaN and NaN - NaN == NaN, so an unordered self-subtraction
    // identifies exactly the lanes holding inf or NaN.
    __m512 self_sub = _mm512_sub_ps(values, values);
    return _mm512_cmp_ps_mask(self_sub, self_sub, _CMP_UNORD_Q) != 0;
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
    return _mm512_andnot_ps(
        _mm512_set1_ps(-0.0f), values); // clear sign bit
  }
  Vectorized<float> angle() const {
    const auto zero_vec = _mm512_set1_ps(0.f);
    const auto nan_vec = _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN());
    const auto nan_mask = _mm512_cmp_ps_mask(values, values, _CMP_UNORD_Q);
    const auto pi = _mm512_set1_ps(3.141592653589793238463f);
    const auto neg_mask = _mm512_cmp_ps_mask(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm512_mask_blend_ps(neg_mask, zero_vec, pi);
    angle = _mm512_mask_blend_ps(nan_mask, angle, nan_vec);
    return angle;
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return _mm512_set1_ps(0);
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> acos() const {
    return tensorplay::tpsleef::acos(values);
  }
  Vectorized<float> acosh() const {
    return tensorplay::tpsleef::acosh(values);
  }
  Vectorized<float> asin() const {
    return tensorplay::tpsleef::asin(values);
  }
  Vectorized<float> asinh() const {
    return tensorplay::tpsleef::asinh(values);
  }
  Vectorized<float> atan() const {
    return tensorplay::tpsleef::atan(values);
  }
  Vectorized<float> atanh() const {
    return tensorplay::tpsleef::atanh(values);
  }
  Vectorized<float> atan2(const Vectorized<float>& exp) const {
    return tensorplay::tpsleef::atan2(values, exp.values);
  }
  Vectorized<float> copysign(const Vectorized<float>& sign) const {
    // clear sign bit of a, and merge with sign bit of b
    return _mm512_or_ps(
        _mm512_andnot_ps(_mm512_set1_ps(-0.0f), values),
        _mm512_and_ps(_mm512_set1_ps(-0.0f), sign));
  }
  Vectorized<float> erf() const {
    // Two ranges, both evaluated and then selected between, so no lane
    // takes a branch.  Near zero the Maclaurin series is used directly:
    // the rational tail below ends in ``1 - r``, and for small arguments
    // that subtraction cancels away the result's low bits.  Away from zero
    // the series would need many more terms than the tail form, which
    // reaches float precision with five coefficients and one exponential.
    // Worst case over the whole line is under three float ulp.
    const __m512 sign_bit = _mm512_set1_ps(-0.0f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 abs_x = _mm512_andnot_ps(sign_bit, values);
    const __m512 sq = _mm512_mul_ps(values, values);

    // erf(x) = x * P(x^2) for |x| < 0.7
    __m512 near = _mm512_set1_ps(1.2055332981789664e-04f);
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(-8.5440360144887751e-04f));
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(5.2239776254421878e-03f));
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(-2.6866170645131252e-02f));
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(1.1283791670955126e-01f));
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(-3.7612638903183752e-01f));
    near = _mm512_fmadd_ps(near, sq, _mm512_set1_ps(1.1283791670955126e+00f));
    near = _mm512_mul_ps(values, near);

    // erf(|x|) = 1 - Q(t) * t * exp(-x^2), t = 1 / (1 + 0.3275911 * |x|)
    const __m512 t = _mm512_div_ps(
        one, _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), abs_x, one));
    __m512 tail = _mm512_set1_ps(1.061405429f);
    tail = _mm512_fmadd_ps(tail, t, _mm512_set1_ps(-1.453152027f));
    tail = _mm512_fmadd_ps(tail, t, _mm512_set1_ps(1.421413741f));
    tail = _mm512_fmadd_ps(tail, t, _mm512_set1_ps(-0.284496736f));
    tail = _mm512_fmadd_ps(tail, t, _mm512_set1_ps(0.254829592f));
    tail = _mm512_mul_ps(tail, t);
    tail = _mm512_sub_ps(
        one,
        _mm512_mul_ps(
            tail, tensorplay::tpsleef::exp(_mm512_xor_ps(sign_bit, sq))));
    // the tail was evaluated on |x|; erf is odd
    tail = _mm512_or_ps(_mm512_and_ps(sign_bit, values), tail);

    return _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(abs_x, _mm512_set1_ps(0.7f), _CMP_LT_OQ),
        tail,
        near);
  }
  Vectorized<float> erfc() const {
    return tensorplay::tpsleef::erfc(values);
  }
  Vectorized<float> exp() const {
    return tensorplay::tpsleef::exp(values);
  }
  Vectorized<float> exp2() const {
    return tensorplay::tpsleef::exp2(values);
  }
  Vectorized<float> expm1() const {
    return tensorplay::tpsleef::expm1(values);
  }
  Vectorized<float> exp_u20() const {
    return tensorplay::tpsleef::exp(values);
  }
  Vectorized<float> fexp_u20() const {
    return tensorplay::tpsleef::exp(values);
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
    return tensorplay::tpsleef::log(values);
  }
  Vectorized<float> log2() const {
    return tensorplay::tpsleef::log2(values);
  }
  Vectorized<float> log10() const {
    return tensorplay::tpsleef::log10(values);
  }
  Vectorized<float> log1p() const {
    return tensorplay::tpsleef::log1p(values);
  }
  Vectorized<float> ceil() const {
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<float> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<float> floor() const {
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<float> neg() const {
    return _mm512_xor_ps(values, _mm512_set1_ps(-0.0f));
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
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> sin() const {
    return tensorplay::tpsleef::sin(values);
  }
  Vectorized<float> sinh() const {
    return tensorplay::tpsleef::sinh(values);
  }
  Vectorized<float> tan() const {
    return tensorplay::tpsleef::tan(values);
  }
  Vectorized<float> tanh() const {
    return tensorplay::tpsleef::tanh(values);
  }
  Vectorized<float> trunc() const {
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<float> digamma() const {
    return map([](float v) { return calc_digamma(v); });
  }
  Vectorized<float> erfinv() const {
    return map([](float v) { return calc_erfinv(v); });
  }
  Vectorized<float> igamma(const Vectorized<float>& x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp); x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) tmp[i] = calc_igamma(static_cast<float>(tmp[i]), static_cast<float>(tmp_x[i]));
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float>& x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp); x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) tmp[i] = calc_igammac(static_cast<float>(tmp[i]), static_cast<float>(tmp_x[i]));
    return loadu(tmp);
  }
  Vectorized<float> sqrt() const {
    return _mm512_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm512_div_ps(_mm512_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  float reduce_add() const {
    auto v = values;
    // 256-bit shuffle: sum the two 256-bit halves
    auto v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = _mm512_add_ps(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = _mm512_add_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = _mm512_add_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = _mm512_add_ps(v, v1);
    return _mm256_cvtss_f32(_mm512_castps512_ps256(v));
  }
  float reduce_max() const {
    auto v = values;
    // 256-bit shuffle: max the two 256-bit halves
    auto v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = _mm512_max_ps(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = _mm512_max_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = _mm512_max_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = _mm512_max_ps(v, v1);
    return _mm256_cvtss_f32(_mm512_castps512_ps256(v));
  }
  float reduce_min() const {
    auto v = values;
    // 256-bit shuffle: min the two 256-bit halves
    auto v1 = _mm512_shuffle_f32x4(v, v, 0x4E);
    v = _mm512_min_ps(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_f32x4(v, v, 0xB1);
    v = _mm512_min_ps(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0x4E);
    v = _mm512_min_ps(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_ps(v, v, 0xB1);
    v = _mm512_min_ps(v, v1);
    return _mm256_cvtss_f32(_mm512_castps512_ps256(v));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_EQ_OQ));
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_NEQ_UQ));
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_LT_OQ));
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_LE_OQ));
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_GT_OQ));
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return widen(_mm512_cmp_ps_mask(values, other.values, _CMP_GE_OQ));
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
  return _mm512_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_div_ps(a, b);
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
  Vectorized<float> max = _mm512_max_ps(a, b);
  Vectorized<float> nan_lanes =
      widen(_mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(max, nan_lanes);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  Vectorized<float> min = _mm512_min_ps(a, b);
  Vectorized<float> nan_lanes =
      widen(_mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q));
  // Exploit the fact that all-ones is a NaN.
  return _mm512_or_ps(min, nan_lanes);
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  return _mm512_min_ps(max, _mm512_max_ps(min, a));
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return _mm512_min_ps(max, a);
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return _mm512_max_ps(min, a);
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_and_ps(a, b);
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_or_ps(a, b);
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return _mm512_xor_ps(a, b);
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
    _mm512_storeu_ps(dst + i, _mm512_loadu_ps(src + i));
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
  return _mm512_fmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm512_fnmadd_ps(a, b, c);
}

template <>
Vectorized<float> inline fmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm512_fmsub_ps(a, b, c);
}

template <>
Vectorized<float> inline fnmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return _mm512_fnmsub_ps(a, b, c);
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY

#endif // CPU_CAPABILITY_AVX512
