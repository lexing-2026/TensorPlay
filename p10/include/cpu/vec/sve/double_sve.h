#pragma once

// Vectorized<double> for the SVE tiers: a single fixed-length SVE vector
// (4 lanes at VL=256, 2 lanes at VL=128). Transcendental functions call the
// SLEEF vector-length-agnostic SVE entry points.

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
struct is_vec_specialized_for<double> : std::bool_constant<true> {};

template <>
class Vectorized<double> {
 private:
  vls_float64_t values;

 public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type kSize = VECTOR_WIDTH / sizeof(double);
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(svfloat64_t v) : values(v) {}
  Vectorized(double scalar) : values(svdup_n_f64(scalar)) {}
  Vectorized(double s0, double s1, double s2, double s3) {
    double buf[4] = {s0, s1, s2, s3};
    static_assert(sizeof(buf) >= sizeof(double) * size(),
                  "SVE lane count exceeds the constructor's fixed width");
    values = svld1_f64(sve_first_f64(size()), buf);
  }

  operator svfloat64_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    svbool_t m = sve_lane_pred_f64<mask, size()>();
    return svsel_f64(m, b.values, a.values);
  }

  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    return svsel_f64(sve_bits_to_pred_f64(mask.values), b.values, a.values);
  }

  template <typename step_t>
  static Vectorized<double> arange(
      double base = 0.,
      step_t step = static_cast<step_t>(1)) {
    // base + i*step: integer index vector converted to floating point.
    svfloat64_t lanes = svcvt_f64_s64_x(
        svptrue_b64(), svindex_s64(0, 1));
    svfloat64_t scaled = svmul_n_f64_x(
        svptrue_b64(), lanes, static_cast<double>(step));
    return svadd_f64_x(svptrue_b64(), svdup_n_f64(base), scaled);
  }

  static Vectorized<double> set(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      int64_t count = size()) {
    return svsel_f64(sve_first_f64(count), b.values, a.values);
  }

  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return svld1_f64(sve_first_f64(size()),
                       reinterpret_cast<const double*>(ptr));
    }
    return svld1_f64(sve_first_f64(count),
                     reinterpret_cast<const double*>(ptr));
  }

  void store(void* ptr, int64_t count = size()) const {
    svst1_f64(sve_first_f64(count), reinterpret_cast<double*>(ptr), values);
  }

  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ double tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0.0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<double> isnan() const {
    return svsel_f64(svcmpne_f64(svptrue_b64(), values, values),
                     sve_ones_f64(), sve_zeros_f64());
  }

  bool has_inf_nan() const {
    svfloat64_t sub = svsub_f64_x(svptrue_b64(), values, values);
    svbool_t bad = svcmpne_n_u64(
        svptrue_b64(), svreinterpret_u64_f64(sub), 0ull);
    return svptest_any(svptrue_b64(), bad);
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
    return svabs_f64_x(svptrue_b64(), values);
  }

  Vectorized<double> acos() const {
    return tensorplay::tpsleef::acos(values);
  }
  Vectorized<double> acosh() const {
    return map(std::acosh);
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
    svbool_t pg = svptrue_b64();
    svuint64_t mag = svand_u64_x(
        pg, svreinterpret_u64_f64(values), svdup_n_u64(0x7fffffffffffffffull));
    svuint64_t sgn = svand_u64_x(
        pg, svreinterpret_u64_f64(sign.values), svdup_n_u64(0x8000000000000000ull));
    return svreinterpret_f64_u64(svorr_u64_x(pg, mag, sgn));
  }
  Vectorized<double> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<double> erf() const {
    return tensorplay::tpsleef::erf(values);
  }
  Vectorized<double> erfc() const {
    return tensorplay::tpsleef::erfc(values);
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> angle() const {
    const svbool_t pg = svptrue_b64();
    const svbool_t neg = svcmplt_f64(pg, values, svdup_n_f64(0));
    const svbool_t nan = svcmpne_f64(pg, values, values);
    Vectorized<double> tmp = blendv(
        Vectorized<double>(0),
        Vectorized<double>(3.141592653589793238463),
        sve_cmp_f64(neg));
    return blendv(tmp, *this, sve_cmp_f64(nan));
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
    return exp();
  }
  Vectorized<double> fexp_u20() const {
    return exp();
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
    return svrintp_f64_x(svptrue_b64(), values);
  }
  Vectorized<double> cos() const {
    return tensorplay::tpsleef::cos(values);
  }
  Vectorized<double> cosh() const {
    return tensorplay::tpsleef::cosh(values);
  }
  Vectorized<double> floor() const {
    return svrintm_f64_x(svptrue_b64(), values);
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::hypot(values, b.values);
  }
  Vectorized<double> neg() const {
    return svneg_f64_x(svptrue_b64(), values);
  }
  Vectorized<double> nextafter(const Vectorized<double>& b) const {
    __at_align__ double tmp[size()], tmp_b[size()], tmp_result[size()];
    store(tmp);
    b.store(tmp_b);
    for (int i = 0; i < size(); i++) {
      tmp_result[i] = std::nextafter(tmp[i], tmp_b[i]);
    }
    return loadu(tmp_result);
  }
  Vectorized<double> round() const {
    return svrintn_f64_x(svptrue_b64(), values);
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
    return svrintz_f64_x(svptrue_b64(), values);
  }
  Vectorized<double> frac() const {
    return svsub_f64_x(svptrue_b64(), values, trunc());
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()], tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()], tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> sqrt() const {
    return svsqrt_f64_x(svptrue_b64(), values);
  }
  Vectorized<double> reciprocal() const {
    return svdivr_n_f64_x(svptrue_b64(), values, 1.0);
  }
  Vectorized<double> rsqrt() const {
    return svdivr_n_f64_x(svptrue_b64(), sqrt(), 1.0);
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    return tensorplay::tpsleef::pow(values, b.values);
  }
  double reduce_add() const {
    return svaddv_f64(svptrue_b64(), values);
  }
  double reduce_max() const {
    return svmaxv_f64(svptrue_b64(), values);
  }
  double reduce_min() const {
    return svminv_f64(svptrue_b64(), values);
  }

  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmpeq_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmpne_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmplt_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmple_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmpgt_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return sve_cmp_f64(svcmpge_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> eq(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmpeq_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> ne(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmpne_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> gt(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmpgt_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> ge(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmpge_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> lt(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmplt_f64(svptrue_b64(), values, other.values));
  }
  Vectorized<double> le(const Vectorized<double>& other) const {
    return sve_cmp01_f64(svcmple_f64(svptrue_b64(), values, other.values));
  }

  Vectorized<double> maximum(const Vectorized<double>& other) const {
    return sve_max_nan_f64(values, other.values);
  }
  Vectorized<double> minimum(const Vectorized<double>& other) const {
    return sve_min_nan_f64(values, other.values);
  }
};

template <>
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.maximum(b);
}

template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return a.minimum(b);
}

template <>
Vectorized<double> inline operator+(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svadd_f64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svsub_f64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svmul_f64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svdiv_f64_z(svptrue_b64(), a, b);
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svreinterpret_f64_u64(svand_u64_x(
      svptrue_b64(),
      svreinterpret_u64_f64(a),
      svreinterpret_u64_f64(b)));
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svreinterpret_f64_u64(svorr_u64_x(
      svptrue_b64(),
      svreinterpret_u64_f64(a),
      svreinterpret_u64_f64(b)));
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return svreinterpret_f64_u64(sveor_u64_x(
      svptrue_b64(),
      svreinterpret_u64_f64(a),
      svreinterpret_u64_f64(b)));
}

template <>
Vectorized<double> inline fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return svmla_f64_x(svptrue_b64(), c, a, b);
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  svbool_t pg = svptrue_b64();
  return svmin_f64_x(pg, max, svmax_f64_x(pg, min, a));
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return svmax_f64_x(svptrue_b64(), min, a);
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return svmin_f64_x(svptrue_b64(), max, a);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    svst1_f64(sve_first_f64(Vectorized<double>::size()), dst + i,
              svld1_f64(sve_first_f64(Vectorized<double>::size()), src + i));
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
