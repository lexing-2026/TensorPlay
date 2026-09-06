#pragma once

// Vectorized<double> for the aarch64 NEON tier: one 128-bit vector (2
// lanes).

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
struct is_vec_specialized_for<double> : std::bool_constant<true> {};

template <>
class Vectorized<double> {
 private:
  float64x2_t values;

 public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type kSize = 2;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() : values(vdupq_n_f64(0.0)) {}
  Vectorized(float64x2_t v) : values(v) {}
  Vectorized(double scalar) : values(vdupq_n_f64(scalar)) {}
  Vectorized(double s0, double s1) : values{s0, s1} {}

  operator float64x2_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    __at_align__ uint64_t bits[2];
    for (int i = 0; i < 2; ++i) {
      bits[i] = ((mask >> i) & 1) ? 0xffffffffffffffffull : 0ull;
    }
    uint64x2_t sel = vld1q_u64(bits);
    return Vectorized<double>(vbslq_f64(sel, b.values, a.values));
  }

  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    return Vectorized<double>(vbslq_f64(
        vreinterpretq_u64_f64(mask.values), b.values, a.values));
  }

  template <typename step_t>
  static Vectorized<double> arange(
      double base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(
        base, static_cast<double>(base + step));
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
    }
    return b;
  }

  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vld1q_f64(reinterpret_cast<const double*>(ptr));
    }
    __at_align__ double tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(double));
    return vld1q_f64(tmp_values);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vst1q_f64(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      __at_align__ double tmp_values[size()];
      vst1q_f64(tmp_values, values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(double));
    }
  }

  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;

  int zero_mask() const {
    uint64x2_t cmp = vceqq_f64(values, vdupq_n_f64(0.0));
    __at_align__ uint64_t tmp[2];
    vst1q_u64(tmp, cmp);
    int mask = 0;
    for (int i = 0; i < 2; ++i) {
      if (tmp[i]) mask |= (1 << i);
    }
    return mask;
  }

  Vectorized<double> isnan() const {
    return Vectorized<double>(vreinterpretq_f64_u64(
        veorq_u64(_tp_all_ones_u64(), vceqq_f64(values, values))));
  }

  bool has_inf_nan() const {
    float64x2_t sub = vsubq_f64(values, values);
    uint64x2_t bits = vreinterpretq_u64_f64(sub);
    static const uint64x2_t exp_mask = vdupq_n_u64(0x77f0000000000000ull);
    uint64x2_t bad = vandq_u64(bits, exp_mask);
    __at_align__ uint64_t tmp[2];
    vst1q_u64(tmp, bad);
    for (int i = 0; i < 2; ++i) {
      if (tmp[i]) return true;
    }
    return false;
  }

  Vectorized<double> map(double (*const f)(double)) const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  template <typename F>
  Vectorized<double> map2(const Vectorized<double>& other, F f) const {
    __at_align__ double tmp[size()], tmp_o[size()];
    store(tmp);
    other.store(tmp_o);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i], tmp_o[i]);
    }
    return loadu(tmp);
  }

  Vectorized<double> abs() const {
    return vabsq_f64(values);
  }

#if defined(TP_NEON_SLEEF)
  Vectorized<double> acos() const {
    return tensorplay::tpsleef::acos(values);
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
  Vectorized<double> atan2(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::atan2(values, b.values);
  }
  Vectorized<double> copysign(const Vectorized<double>& sign) const {
    uint64x2_t mag = vandq_u64(
        vreinterpretq_u64_f64(values), vdupq_n_u64(0x7fffffffffffffffull));
    uint64x2_t sgn = vandq_u64(
        vreinterpretq_u64_f64(sign.values), vdupq_n_u64(0x8000000000000000ull));
    return vreinterpretq_f64_u64(vorrq_u64(mag, sgn));
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
  Vectorized<double> sin() const {
    return tensorplay::tpsleef::sin(values);
  }
  Vectorized<double> tan() const {
    return tensorplay::tpsleef::tan(values);
  }
  Vectorized<double> tanh() const {
    return tensorplay::tpsleef::tanh(values);
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return tensorplay::tpsleef::fmod(values, q.values);
  }
  Vectorized<double> nextafter(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::nextafter(values, b.values);
  }
#else
  Vectorized<double> acos() const {
    return map(std::acos);
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
  Vectorized<double> atan2(const Vectorized<double>& b) const {
    return map2(b, [](double x, double y) { return std::atan2(x, y); });
  }
  Vectorized<double> copysign(const Vectorized<double>& sign) const {
    return map2(sign, [](double x, double y) { return std::copysign(x, y); });
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
  Vectorized<double> sin() const {
    return map(std::sin);
  }
  Vectorized<double> tan() const {
    return map(std::tan);
  }
  Vectorized<double> tanh() const {
    return map(std::tanh);
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    return map2(b, std::pow);
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return map2(b, std::hypot);
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    return map2(q, std::fmod);
  }
  Vectorized<double> nextafter(const Vectorized<double>& b) const {
    return map2(b, [](double x, double y) {
      return std::nextafter(x, y);
    });
  }
#endif

  Vectorized<double> acosh() const {
    return map(std::acosh);
  }
  Vectorized<double> angle() const {
    auto tmp = blendv(
        Vectorized<double>(0),
        Vectorized<double>(3.141592653589793238463),
        *this < Vectorized<double>(0));
    return blendv(tmp, *this, isnan());
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return Vectorized<double>{0};
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> exp_u20() const {
    return exp();
  }
  Vectorized<double> fexp_u20() const {
    return exp();
  }
  Vectorized<double> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<double> ceil() const {
    return vrndpq_f64(values);
  }
  Vectorized<double> cos() const {
#if defined(TP_NEON_SLEEF)
    return tensorplay::tpsleef::cos(values);
#else
    return map(std::cos);
#endif
  }
  Vectorized<double> sinh() const {
    return map(std::sinh);
  }
  Vectorized<double> cosh() const {
    return map(std::cosh);
  }
  Vectorized<double> floor() const {
    return vrndmq_f64(values);
  }
  Vectorized<double> neg() const {
    return vnegq_f64(values);
  }
  Vectorized<double> round() const {
    return vrndnq_f64(values);
  }
  Vectorized<double> trunc() const {
    return vrndq_f64(values);
  }
  Vectorized<double> frac() const {
    return vsubq_f64(values, trunc());
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> igamma(const Vectorized<double>& x) const {
    return map2(x, [](double a, double x) { return calc_igamma(a, x); });
  }
  Vectorized<double> igammac(const Vectorized<double>& x) const {
    return map2(x, [](double a, double x) { return calc_igammac(a, x); });
  }
  Vectorized<double> sqrt() const {
    return vsqrtq_f64(values);
  }
  Vectorized<double> reciprocal() const {
    return vdivq_f64(vdupq_n_f64(1.0), values);
  }
  Vectorized<double> rsqrt() const {
    return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(values));
  }
  double reduce_add() const {
    return vaddvq_f64(values);
  }
  double reduce_max() const {
    return vmaxvq_f64(values);
  }
  double reduce_min() const {
    return vminvq_f64(values);
  }

  TP_NEON_DEFINE_MEMBER_CMP(operator==, double, f64, vceqq_f64, u64)
  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(
        veorq_u64(_tp_all_ones_u64(), vceqq_f64(values, other.values))));
  }
  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vcltq_f64(values, other.values)));
  }
  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vcleq_f64(values, other.values)));
  }
  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vcgtq_f64(values, other.values)));
  }
  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vcgeq_f64(values, other.values)));
  }
  Vectorized<double> eq(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this == other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> ne(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this != other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> gt(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this > other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> ge(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this >= other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> lt(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this < other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> le(const Vectorized<double>& other) const {
    return Vectorized<double>(vreinterpretq_f64_u64(vandq_u64(
        vreinterpretq_u64_f64(*this <= other),
        vdupq_n_u64(0x3ff0000000000000ull))));
  }
  Vectorized<double> maximum(const Vectorized<double>& other) const {
    Vectorized<double> max(vmaxnmq_f64(values, other.values));
    uint64x2_t nan_a = veorq_u64(_tp_all_ones_u64(), vceqq_f64(values, values));
    uint64x2_t nan_b = veorq_u64(_tp_all_ones_u64(), vceqq_f64(other.values, other.values));
    uint64x2_t nan = vorrq_u64(nan_a, nan_b);
    return Vectorized<double>(vreinterpretq_f64_u64(
        vorrq_u64(vreinterpretq_u64_f64(max), nan)));
  }
  Vectorized<double> minimum(const Vectorized<double>& other) const {
    Vectorized<double> min(vminnmq_f64(values, other.values));
    uint64x2_t nan_a = veorq_u64(_tp_all_ones_u64(), vceqq_f64(values, values));
    uint64x2_t nan_b = veorq_u64(_tp_all_ones_u64(), vceqq_f64(other.values, other.values));
    uint64x2_t nan = vorrq_u64(nan_a, nan_b);
    return Vectorized<double>(vreinterpretq_f64_u64(
        vorrq_u64(vreinterpretq_u64_f64(min), nan)));
  }
};

template <>
Vectorized<double> operator+(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator-(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator*(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator/(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator&(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator|(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> operator^(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> maximum(const Vectorized<double>&, const Vectorized<double>&);
template <>
Vectorized<double> minimum(const Vectorized<double>&, const Vectorized<double>&);

template <>
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {{
  return a.maximum(b);
}}

template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {{
  return a.minimum(b);
}}

template <>
Vectorized<double> inline operator+(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vaddq_f64(a, b);
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vsubq_f64(a, b);
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vmulq_f64(a, b);
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vdivq_f64(a, b);
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(vandq_u64(
      vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(vorrq_u64(
      vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(veorq_u64(
      vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
Vectorized<double> inline fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return vfmaq_f64(c, a, b);
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  return vminnmq_f64(max, vmaxnmq_f64(min, a));
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return vmaxnmq_f64(min, a);
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return vminnmq_f64(max, a);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    vst1q_f64(dst + i, vld1q_f64(src + i));
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
