#pragma once

// Vectorized<float> for the SVE tiers: the value is a single fixed-length
// SVE vector (8 lanes at VL=256, 4 lanes at VL=128). Transcendental
// functions call the SLEEF vector-length-agnostic SVE entry points; the
// runtime tier selection guarantees the hardware VL matches the
// compile-time vector length.

#include "cpu/vec/sve/sve_helpers.h"
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
  vls_float32_t values;

 public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type kSize = VECTOR_WIDTH / sizeof(float);
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(svfloat32_t v) : values(v) {}
  Vectorized(float scalar) : values(svdup_n_f32(scalar)) {}
  Vectorized(float s0, float s1, float s2, float s3,
             float s4, float s5, float s6, float s7) {
    float buf[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    static_assert(sizeof(buf) >= sizeof(float) * size(),
                  "SVE lane count exceeds the constructor's fixed width");
    values = svld1_f32(sve_first_f32(size()), buf);
  }

  operator svfloat32_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    svbool_t m = sve_lane_pred_f32<mask, size()>();
    return svsel_f32(m, b.values, a.values);
  }

  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    return svsel_f32(sve_bits_to_pred_f32(mask.values), b.values, a.values);
  }

  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    // base + i*step: integer index vector converted to floating point.
    svfloat32_t lanes = svcvt_f32_s32_x(
        svptrue_b32(), svindex_s32(0, 1));
    svfloat32_t scaled = svmul_n_f32_x(
        svptrue_b32(), lanes, static_cast<float>(step));
    return svadd_f32_x(svptrue_b32(), svdup_n_f32(base), scaled);
  }

  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      int64_t count = size()) {
    return svsel_f32(sve_first_f32(count), b.values, a.values);
  }

  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return svld1_f32(sve_first_f32(size()),
                       reinterpret_cast<const float*>(ptr));
    }
    return svld1_f32(sve_first_f32(count),
                     reinterpret_cast<const float*>(ptr));
  }

  void store(void* ptr, int64_t count = size()) const {
    svst1_f32(sve_first_f32(count), reinterpret_cast<float*>(ptr), values);
  }

  const float& operator[](int idx) const = delete;
  float& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ float tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0.0f) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<float> isnan() const {
    // NaN is the only value that compares unequal to itself.
    return svsel_f32(svcmpne_f32(svptrue_b32(), values, values),
                     sve_ones_f32(), sve_zeros_f32());
  }

  bool has_inf_nan() const {
    svfloat32_t sub = svsub_f32_x(svptrue_b32(), values, values);
    svbool_t bad = svcmpne_n_u32(
        svptrue_b32(), svreinterpret_u32_f32(sub), 0u);
    return svptest_any(svptrue_b32(), bad);
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
    return svabs_f32_x(svptrue_b32(), values);
  }

  Vectorized<float> acos() const {
    return tensorplay::tpsleef::acos(values);
  }
  Vectorized<float> acosh() const {
    return map(std::acosh);
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
    // |a| with the sign bit of b (the ACLE has no copysign intrinsic).
    svbool_t pg = svptrue_b32();
    svuint32_t mag = svand_u32_x(
        pg, svreinterpret_u32_f32(values), svdup_n_u32(0x7fffffffu));
    svuint32_t sgn = svand_u32_x(
        pg, svreinterpret_u32_f32(sign.values), svdup_n_u32(0x80000000u));
    return svreinterpret_f32_u32(svorr_u32_x(pg, mag, sgn));
  }
  Vectorized<float> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<float> erf() const {
    return tensorplay::tpsleef::erf(values);
  }
  Vectorized<float> erfc() const {
    return tensorplay::tpsleef::erfc(values);
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> angle() const {
    const svbool_t pg = svptrue_b32();
    const svbool_t neg = svcmplt_f32(pg, values, svdup_n_f32(0));
    const svbool_t nan = svcmpne_f32(pg, values, values);
    Vectorized<float> tmp = blendv(
        Vectorized<float>(0),
        Vectorized<float>(3.141592653589793238463f),
        sve_cmp_f32(neg));
    return blendv(tmp, *this, sve_cmp_f32(nan));
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
    return exp();
  }
  Vectorized<float> fexp_u20() const {
    return exp();
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
    return svrintp_f32_x(svptrue_b32(), values);
  }
  Vectorized<float> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<float> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<float> floor() const {
    return svrintm_f32_x(svptrue_b32(), values);
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<float> neg() const {
    return svneg_f32_x(svptrue_b32(), values);
  }
  Vectorized<float> nextafter(const Vectorized<float>& b) const {
    __at_align__ float tmp[size()], tmp_b[size()], tmp_result[size()];
    store(tmp);
    b.store(tmp_b);
    for (int i = 0; i < size(); i++) {
      tmp_result[i] = std::nextafter(tmp[i], tmp_b[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<float> round() const {
    return svrintn_f32_x(svptrue_b32(), values);
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
    return svrintz_f32_x(svptrue_b32(), values);
  }
  Vectorized<float> frac() const {
    return svsub_f32_x(svptrue_b32(), values, trunc());
  }
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> igamma(const Vectorized<float>& x) const {
    __at_align__ float tmp[size()], tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float>& x) const {
    __at_align__ float tmp[size()], tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> sqrt() const {
    return svsqrt_f32_x(svptrue_b32(), values);
  }
  Vectorized<float> reciprocal() const {
    return svdivr_n_f32_x(svptrue_b32(), values, 1.0f);
  }
  Vectorized<float> rsqrt() const {
    return svdivr_n_f32_x(svptrue_b32(), sqrt(), 1.0f);
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  float reduce_add() const {
    return svaddv_f32(svptrue_b32(), values);
  }
  float reduce_max() const {
    return svmaxv_f32(svptrue_b32(), values);
  }
  float reduce_min() const {
    return svminv_f32(svptrue_b32(), values);
  }

  // Ordered comparisons (false on NaN); result lanes are all-ones/all-zeros.
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmpeq_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmpne_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmplt_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmple_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmpgt_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return sve_cmp_f32(svcmpge_f32(svptrue_b32(), values, other.values));
  }
  // 0/1-valued comparison results.
  Vectorized<float> eq(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmpeq_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> ne(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmpne_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> gt(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmpgt_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> ge(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmpge_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> lt(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmplt_f32(svptrue_b32(), values, other.values));
  }
  Vectorized<float> le(const Vectorized<float>& other) const {
    return sve_cmp01_f32(svcmple_f32(svptrue_b32(), values, other.values));
  }

  Vectorized<float> maximum(const Vectorized<float>& other) const {
    return sve_max_nan_f32(values, other.values);
  }
  Vectorized<float> minimum(const Vectorized<float>& other) const {
    return sve_min_nan_f32(values, other.values);
  }
};

template <>
Vectorized<float> inline maximum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return a.maximum(b);
}

template <>
Vectorized<float> inline minimum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return a.minimum(b);
}

template <>
Vectorized<float> inline operator+(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svadd_f32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svsub_f32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svmul_f32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svdiv_f32_z(svptrue_b32(), a, b);
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svreinterpret_f32_u32(svand_u32_x(
      svptrue_b32(),
      svreinterpret_u32_f32(a),
      svreinterpret_u32_f32(b)));
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svreinterpret_f32_u32(svorr_u32_x(
      svptrue_b32(),
      svreinterpret_u32_f32(a),
      svreinterpret_u32_f32(b)));
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svreinterpret_f32_u32(sveor_u32_x(
      svptrue_b32(),
      svreinterpret_u32_f32(a),
      svreinterpret_u32_f32(b)));
}

template <>
Vectorized<float> inline fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return svmla_f32_x(svptrue_b32(), c, a, b);
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  svbool_t pg = svptrue_b32();
  return svmin_f32_x(pg, max, svmax_f32_x(pg, min, a));
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return svmax_f32_x(svptrue_b32(), min, a);
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return svmin_f32_x(svptrue_b32(), max, a);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    svst1_f32(sve_first_f32(Vectorized<float>::size()), dst + i,
              svld1_f32(sve_first_f32(Vectorized<float>::size()), src + i));
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
