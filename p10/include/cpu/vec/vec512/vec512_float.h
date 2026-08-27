#pragma once

// Port of ATen/cpu/vec/vec512/vec512_float.h with the TensorPlay vec layer,
// derived from the vec256 port (same interface, 512-bit width).  Math
// functions that depend on Sleef resolve to glibc libmvec's 512-bit vector
// ABI when available and fall back to a scalar map otherwise; everything
// else uses native AVX512 intrinsics.

#include <immintrin.h>
#include "cpu/vec/vec_base.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(CPU_CAPABILITY_AVX512) && defined(__GLIBC__)
// PyTorch's AVX512 Vectorized<float> uses Sleef for transcendental functions.
// On glibc systems libmvec exposes the same vector ABI at full 512-bit width
// (_ZGVeN16v_* = sixteen floats per call).
extern "C" {
__m512 _ZGVeN16v_sinf(__m512);
__m512 _ZGVeN16v_cosf(__m512);
__m512 _ZGVeN16v_expf(__m512);
__m512 _ZGVeN16v_logf(__m512);
__m512 _ZGVeN16v_tanhf(__m512);
__m512 _ZGVeN16v__acosf(__m512);
__m512 _ZGVeN16v__acoshf(__m512);
__m512 _ZGVeN16v__asinf(__m512);
__m512 _ZGVeN16v__asinhf(__m512);
__m512 _ZGVeN16v__atanf(__m512);
__m512 _ZGVeN16v__atanhf(__m512);
__m512 _ZGVeN16v__cosf(__m512);
__m512 _ZGVeN16v__coshf(__m512);
__m512 _ZGVeN16v__erff(__m512);
__m512 _ZGVeN16v__erfcf(__m512);
__m512 _ZGVeN16v__exp2f(__m512);
__m512 _ZGVeN16v__expm1f(__m512);
__m512 _ZGVeN16v__log2f(__m512);
__m512 _ZGVeN16v__log10f(__m512);
__m512 _ZGVeN16v__log1pf(__m512);
__m512 _ZGVeN16v__sinf(__m512);
__m512 _ZGVeN16v__sinhf(__m512);
__m512 _ZGVeN16v__tanf(__m512);
}
#endif

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
    // ATen semantics: NaN -> NaN, negative -> pi, otherwise -> 0.
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
    #if defined(__GLIBC__)
    return _ZGVeN16v__acosf(values);
#else
    return map(std::acos);
#endif
  }
  Vectorized<float> acosh() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__acoshf(values);
#else
    return map(std::acosh);
#endif
  }
  Vectorized<float> asin() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__asinf(values);
#else
    return map(std::asin);
#endif
  }
  Vectorized<float> asinh() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__asinhf(values);
#else
    return map(std::asinh);
#endif
  }
  Vectorized<float> atan() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__atanf(values);
#else
    return map(std::atan);
#endif
  }
  Vectorized<float> atanh() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__atanhf(values);
#else
    return map(std::atanh);
#endif
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
    return _mm512_or_ps(
        _mm512_andnot_ps(_mm512_set1_ps(-0.0f), values),
        _mm512_and_ps(_mm512_set1_ps(-0.0f), sign));
  }
  Vectorized<float> erf() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__erff(values);
#else
    return map(std::erf);
#endif
  }
  Vectorized<float> erfc() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__erfcf(values);
#else
    return map(std::erfc);
#endif
  }
  Vectorized<float> exp() const {
#if defined(__GLIBC__)
    return _ZGVeN16v_expf(values);
#else
    return map(std::exp);
#endif
  }
  Vectorized<float> exp2() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__exp2f(values);
#else
    return map(std::exp2);
#endif
  }
  Vectorized<float> expm1() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__expm1f(values);
#else
    return map(std::expm1);
#endif
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
    return _ZGVeN16v_logf(values);
#else
    return map(std::log);
#endif
  }
  Vectorized<float> log2() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__log2f(values);
#else
    return map(std::log2);
#endif
  }
  Vectorized<float> log10() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__log10f(values);
#else
    return map(std::log10);
#endif
  }
  Vectorized<float> log1p() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__log1pf(values);
#else
    return map(std::log1p);
#endif
  }
  Vectorized<float> ceil() const {
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }
  Vectorized<float> cos() const {
#if defined(__GLIBC__)
    return _ZGVeN16v_cosf(values);
#else
    #if defined(__GLIBC__)
    return _ZGVeN16v__cosf(values);
#else
    return map(std::cos);
#endif
#endif
  }
  Vectorized<float> cosh() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__coshf(values);
#else
    return map(std::cosh);
#endif
  }
  Vectorized<float> floor() const {
    return _mm512_roundscale_ps(
        values, (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
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
#if defined(__GLIBC__)
    return _ZGVeN16v_sinf(values);
#else
    #if defined(__GLIBC__)
    return _ZGVeN16v__sinf(values);
#else
    return map(std::sin);
#endif
#endif
  }
  Vectorized<float> sinh() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__sinhf(values);
#else
    return map(std::sinh);
#endif
  }
  Vectorized<float> tan() const {
    #if defined(__GLIBC__)
    return _ZGVeN16v__tanf(values);
#else
    return map(std::tan);
#endif
  }
  Vectorized<float> tanh() const {
#if defined(__GLIBC__)
    return _ZGVeN16v_tanhf(values);
#else
    return map(std::tanh);
#endif
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
