#pragma once

// Transcendental methods dispatch to the vendored SLEEF vector math
// (runtime-dispatched entry points, see cpu/vec/SleefShims.h); the
// remaining primitives use AVX2 intrinsics directly.

// x86-64 intrinsics only: the AVX specializations below are guarded by
// CPU_CAPABILITY_AVX2/AVX512, and other architectures fall back to the
// generic Vectorized template in vec_base.h.
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
#include <immintrin.h>
#endif
#include "cpu/vec/vec_base.h"
#include "cpu/vec/ErfPoly.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>


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
    const auto zero_vec = _mm256_set1_ps(0.f);
    const auto nan_vec = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
    const auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
    const auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
    const auto pi = _mm256_set1_ps(3.141592653589793238463f);
    const auto neg_mask = _mm256_cmp_ps(values, zero_vec, _CMP_LT_OQ);
    auto angle = _mm256_blendv_ps(zero_vec, pi, neg_mask);
    angle = _mm256_blendv_ps(angle, nan_vec, nan_mask);
    return angle;
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
    return _mm256_or_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values),
        _mm256_and_ps(_mm256_set1_ps(-0.0f), sign));
  }
  Vectorized<float> erf() const {
    // Two forms, both evaluated and then selected between, so no lane takes
    // a branch; see cpu/vec/ErfPoly.h for why the split is there.
    namespace poly = tensorplay::vecmath;
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 abs_x = _mm256_andnot_ps(sign_bit, values);
    const __m256 sq = _mm256_mul_ps(values, values);

    __m256 near = _mm256_set1_ps(poly::kErfSeries[6]);
    for (int i = 5; i >= 0; --i)
      near = _mm256_fmadd_ps(near, sq, _mm256_set1_ps(poly::kErfSeries[i]));
    near = _mm256_mul_ps(values, near);

    const __m256 t = _mm256_div_ps(
        one, _mm256_fmadd_ps(_mm256_set1_ps(poly::kErfTailScale), abs_x, one));
    __m256 tail = _mm256_set1_ps(poly::kErfTail[4]);
    for (int i = 3; i >= 0; --i)
      tail = _mm256_fmadd_ps(tail, t, _mm256_set1_ps(poly::kErfTail[i]));
    tail = _mm256_mul_ps(tail, t);
    tail = _mm256_sub_ps(
        one,
        _mm256_mul_ps(tail, tensorplay::tpsleef::exp(_mm256_xor_ps(sign_bit, sq))));
    // the tail was evaluated on |x|, and erf is odd
    tail = _mm256_or_ps(_mm256_and_ps(sign_bit, values), tail);

    return _mm256_blendv_ps(
        tail,
        near,
        _mm256_cmp_ps(abs_x, _mm256_set1_ps(poly::kErfSplit), _CMP_LT_OQ));
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
    return _mm256_ceil_ps(values);
  }
  Vectorized<float> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<float> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
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
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
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
    return _mm256_sqrt_ps(values);
  }
  Vectorized<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  Vectorized<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
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
