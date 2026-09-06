#pragma once

// Vectorized<float> for the aarch64 NEON tier: one 128-bit vector (4
// lanes). This is the desktop-default backend for aarch64 builds outside
// the SVE tiers (plain Linux distributions, macOS arm64); Android keeps
// the scalar <cmath> transcendental fallback.

#include "cpu/vec/vec128/neon_helpers.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

template <>
struct is_vec_specialized_for<float> : std::bool_constant<true> {};

template <>
class Vectorized<float> {
 private:
  float32x4_t values;

 public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() : values(vdupq_n_f32(0)) {}
  Vectorized(float32x4_t v) : values(v) {}
  Vectorized(float scalar) : values(vdupq_n_f32(scalar)) {}
  Vectorized(float s0, float s1, float s2, float s3)
      : values{s0, s1, s2, s3} {}

  operator float32x4_t() const {
    return values;
  }

  // Lane i of the result comes from b when bit i of mask is set; each
  // mask bit is expanded to an all-ones/all-zeros lane selector.
  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    __at_align__ uint32_t bits[4];
    for (int i = 0; i < 4; ++i) {
      bits[i] = ((mask >> i) & 1) ? 0xffffffffu : 0u;
    }
    uint32x4_t sel = vld1q_u32(bits);
    return Vectorized<float>(vbslq_f32(sel, b.values, a.values));
  }

  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    // Mask lanes are all-ones/all-zeros from the comparison operators.
    return Vectorized<float>(vbslq_f32(
        vreinterpretq_u32_f32(mask.values), b.values, a.values));
  }

  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
        base,
        static_cast<float>(base + step),
        static_cast<float>(base + 2 * step),
        static_cast<float>(base + 3 * step));
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
    }
    return b;
  }

  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vld1q_f32(reinterpret_cast<const float*>(ptr));
    }
    __at_align__ float tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(float));
    return vld1q_f32(tmp_values);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vst1q_f32(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      __at_align__ float tmp_values[size()];
      vst1q_f32(tmp_values, values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(float));
    }
  }

  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;

  int zero_mask() const {
    uint32x4_t cmp = vceqq_f32(values, vdupq_n_f32(0.0f));
    __at_align__ uint32_t tmp[4];
    vst1q_u32(tmp, cmp);
    int mask = 0;
    for (int i = 0; i < 4; ++i) {
      if (tmp[i]) mask |= (1 << i);
    }
    return mask;
  }

  Vectorized<float> isnan() const {
    // NaN is the only value that compares unequal to itself; the result is
    // an all-ones/all-zeros mask.
    return Vectorized<float>(vreinterpretq_f32_u32(
        veorq_u32(_tp_all_ones_u32(), vceqq_f32(values, values))));
  }

  bool has_inf_nan() const {
    // inf/NaN lanes self-subtract to NaN (non-zero bits); finite lanes
    // cancel to +0.
    float32x4_t sub = vsubq_f32(values, values);
    uint32x4_t bits = vreinterpretq_u32_f32(sub);
    static const uint32x4_t exp_mask = vdupq_n_u32(0x77800000u);
    uint32x4_t bad = vandq_u32(bits, exp_mask);
    __at_align__ uint32_t tmp[4];
    vst1q_u32(tmp, bad);
    for (int i = 0; i < 4; ++i) {
      if (tmp[i]) return true;
    }
    return false;
  }

  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  template <typename F>
  Vectorized<float> map2(const Vectorized<float>& other, F f) const {
    __at_align__ float tmp[size()], tmp_o[size()];
    store(tmp);
    other.store(tmp_o);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i], tmp_o[i]);
    }
    return loadu(tmp);
  }

  Vectorized<float> abs() const {
    return vabsq_f32(values);
  }

#if defined(TP_NEON_SLEEF)
  Vectorized<float> acos() const {
    return tensorplay::tpsleef::acos(values);
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
  Vectorized<float> atan2(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::atan2(values, b.values);
  }
  Vectorized<float> copysign(const Vectorized<float>& sign) const {
    // |a| with the sign bit of b.
    uint32x4_t mag = vandq_u32(
        vreinterpretq_u32_f32(values), vdupq_n_u32(0x7fffffffu));
    uint32x4_t sgn = vandq_u32(
        vreinterpretq_u32_f32(sign.values), vdupq_n_u32(0x80000000u));
    return vreinterpretq_f32_u32(vorrq_u32(mag, sgn));
  }
  Vectorized<float> erf() const {
    return tensorplay::tpsleef::erf(values);
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
  Vectorized<float> sin() const {
    return tensorplay::tpsleef::sin(values);
  }
  Vectorized<float> tan() const {
    return tensorplay::tpsleef::tan(values);
  }
  Vectorized<float> tanh() const {
    return tensorplay::tpsleef::tanh(values);
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return tensorplay::tpsleef::fmod(values, q.values);
  }
  Vectorized<float> nextafter(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::nextafter(values, b.values);
  }
#else
  Vectorized<float> acos() const {
    return map(std::acos);
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
  Vectorized<float> atan2(const Vectorized<float>& b) const {
    return map2(b, [](float x, float y) { return std::atan2(x, y); });
  }
  Vectorized<float> copysign(const Vectorized<float>& sign) const {
    return map2(sign, [](float x, float y) { return std::copysign(x, y); });
  }
  Vectorized<float> erf() const {
    return map(std::erf);
  }
  Vectorized<float> erfc() const {
    return map(std::erfc);
  }
  Vectorized<float> exp() const {
    return map(std::exp);
  }
  Vectorized<float> exp2() const {
    return map(std::exp2);
  }
  Vectorized<float> expm1() const {
    return map(std::expm1);
  }
  Vectorized<float> log() const {
    return map(std::log);
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
  Vectorized<float> sin() const {
    return map(std::sin);
  }
  Vectorized<float> tan() const {
    return map(std::tan);
  }
  Vectorized<float> tanh() const {
    return map(std::tanh);
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return map2(b, std::pow);
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return map2(b, std::hypot);
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    return map2(q, std::fmod);
  }
  Vectorized<float> nextafter(const Vectorized<float>& b) const {
    return map2(b, [](float x, float y) {
      return std::nextafter(x, y);
    });
  }
#endif

  // acosh/asin: SLEEF's float-range intermediates overflow where the
  // scalar C library (double intermediates) stays finite; keep the scalar
  // reference semantics.
  Vectorized<float> acosh() const {
    return map(std::acosh);
  }
  Vectorized<float> angle() const {
    auto tmp = blendv(
        Vectorized<float>(0),
        Vectorized<float>(3.141592653589793238463f),
        *this < Vectorized<float>(0));
    return blendv(tmp, *this, isnan());
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return Vectorized<float>{0};
  }
  Vectorized<float> conj() const {
    return *this;
  }
  Vectorized<float> exp_u20() const {
    return exp();
  }
  Vectorized<float> fexp_u20() const {
    return exp();
  }
  Vectorized<float> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<float> ceil() const {
    return vrndpq_f32(values);
  }
  Vectorized<float> cos() const {
#if defined(TP_NEON_SLEEF)
    return tensorplay::tpsleef::cos(values);
#else
    return map(std::cos);
#endif
  }
  // SLEEF's float-range sinh/cosh overflow for large inputs where the
  // scalar C library stays finite; keep the scalar reference semantics.
  Vectorized<float> sinh() const {
    return map(std::sinh);
  }
  Vectorized<float> cosh() const {
    return map(std::cosh);
  }
  Vectorized<float> floor() const {
    return vrndmq_f32(values);
  }
  Vectorized<float> neg() const {
    return vnegq_f32(values);
  }
  Vectorized<float> round() const {
    // Round halfway cases to even (nearbyint semantics), not away from
    // zero.
    return vrndnq_f32(values);
  }
  Vectorized<float> trunc() const {
    return vrndq_f32(values);
  }
  Vectorized<float> frac() const {
    return vsubq_f32(values, trunc());
  }
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> igamma(const Vectorized<float>& x) const {
    return map2(x, [](float a, float x) { return calc_igamma(a, x); });
  }
  Vectorized<float> igammac(const Vectorized<float>& x) const {
    return map2(x, [](float a, float x) { return calc_igammac(a, x); });
  }
  Vectorized<float> sqrt() const {
    return vsqrtq_f32(values);
  }
  Vectorized<float> reciprocal() const {
    return vdivq_f32(vdupq_n_f32(1.0f), values);
  }
  Vectorized<float> rsqrt() const {
    return vdivq_f32(vdupq_n_f32(1.0f), vsqrtq_f32(values));
  }
  float reduce_add() const {
    float32x2_t s = vpadd_f32(vget_low_f32(values), vget_high_f32(values));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
  }
  float reduce_max() const {
    float32x2_t s = vpmax_f32(vget_low_f32(values), vget_high_f32(values));
    s = vpmax_f32(s, s);
    return vget_lane_f32(s, 0);
  }
  float reduce_min() const {
    float32x2_t s = vpmin_f32(vget_low_f32(values), vget_high_f32(values));
    s = vpmin_f32(s, s);
    return vget_lane_f32(s, 0);
  }

  // Ordered comparisons: false when either side is NaN.
  TP_NEON_DEFINE_MEMBER_CMP(operator==, float, f32, vceqq_f32, u32)
  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(
        veorq_u32(_tp_all_ones_u32(), vceqq_f32(values, other.values))));
  }
  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vcltq_f32(values, other.values)));
  }
  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vcleq_f32(values, other.values)));
  }
  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vcgtq_f32(values, other.values)));
  }
  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vcgeq_f32(values, other.values)));
  }
  // 0/1-valued comparison results: mask bits ANDed with the 1.0f bit
  // pattern.
  Vectorized<float> eq(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this == other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> ne(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this != other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> gt(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this > other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> ge(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this >= other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> lt(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this < other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> le(const Vectorized<float>& other) const {
    return Vectorized<float>(vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(*this <= other),
        vdupq_n_u32(0x3f800000u))));
  }
  Vectorized<float> maximum(const Vectorized<float>& other) const {
    Vectorized<float> max(vmaxnmq_f32(values, other.values));
    // vmaxnm returns the non-NaN operand; force NaN propagation to match
    // the IEEE maximum operation.
    uint32x4_t nan_a = veorq_u32(_tp_all_ones_u32(), vceqq_f32(values, values));
    uint32x4_t nan_b = veorq_u32(_tp_all_ones_u32(), vceqq_f32(other.values, other.values));
    uint32x4_t nan = vorrq_u32(nan_a, nan_b);
    return Vectorized<float>(vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(max), nan)));
  }
  Vectorized<float> minimum(const Vectorized<float>& other) const {
    Vectorized<float> min(vminnmq_f32(values, other.values));
    uint32x4_t nan_a = veorq_u32(_tp_all_ones_u32(), vceqq_f32(values, values));
    uint32x4_t nan_b = veorq_u32(_tp_all_ones_u32(), vceqq_f32(other.values, other.values));
    uint32x4_t nan = vorrq_u32(nan_a, nan_b);
    return Vectorized<float>(vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(min), nan)));
  }
};

template <>
Vectorized<float> operator+(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator-(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator*(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator/(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator&(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator|(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> operator^(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> maximum(const Vectorized<float>&, const Vectorized<float>&);
template <>
Vectorized<float> minimum(const Vectorized<float>&, const Vectorized<float>&);

template <>
Vectorized<float> inline maximum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {{
  return a.maximum(b);
}}

template <>
Vectorized<float> inline minimum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {{
  return a.minimum(b);
}}

template <>
Vectorized<float> inline operator+(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vaddq_f32(a, b);
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vsubq_f32(a, b);
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vmulq_f32(a, b);
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vdivq_f32(a, b);
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

template <>
Vectorized<float> inline fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return vfmaq_f32(c, a, b);
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  return vminnmq_f32(max, vmaxnmq_f32(min, a));
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return vmaxnmq_f32(min, a);
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return vminnmq_f32(max, a);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    vst1q_f32(dst + i, vld1q_f32(src + i));
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
