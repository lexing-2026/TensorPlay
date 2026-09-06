#pragma once

// Vectorized<double> for the VSX tier: 256-bit emulation over two 128-bit
// vector registers (4 lanes total). Transcendental functions call the SLEEF
// 2-double entry points once per half.

#include "cpu/vec/vec256/vsx/vsx_helpers.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

inline vfloat64 vec_max_nan_d(const vfloat64& a, const vfloat64& b);
inline vfloat64 vec_min_nan_d(const vfloat64& a, const vfloat64& b);

template <>
struct is_vec_specialized_for<double> : std::bool_constant<true> {};

template <>
class Vectorized<double> {
 private:
  vfloat64 _vec0;
  vfloat64 _vec1;

 public:
  using value_type = double;
  using vec_internal_type = vfloat64;
  using vec_internal_mask_type = vbool64;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}
  Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(double scalar) : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  Vectorized(double s0, double s1, double s2, double s3)
      : _vec0{vfloat64{s0, s1}}, _vec1{vfloat64{s2, s3}} {}

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xF;
    return Vectorized<double>{
        (vfloat64)vec_sel(a._vec0, b._vec0, vsx_dbl_mask1(m)),
        (vfloat64)vec_sel(a._vec1, b._vec1, vsx_dbl_mask2(m))};
  }

  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask) {
    return Vectorized<double>{
        (vfloat64)vec_sel(a._vec0, b._vec0, (vbool64)mask._vec0),
        (vfloat64)vec_sel(a._vec1, b._vec1, (vbool64)mask._vec1)};
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
    if (count >= size()) {
      return {
          vsx_ld_d(0, reinterpret_cast<const double*>(ptr)),
          vsx_ld_d(16, reinterpret_cast<const double*>(ptr))};
    }
    __at_align__ double tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(double));
    return {vsx_ld_d(0, tmp_values), vsx_ld_d(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vsx_st_d(_vec0, 0, reinterpret_cast<double*>(ptr));
      vsx_st_d(_vec1, 16, reinterpret_cast<double*>(ptr));
    } else if (count > 0) {
      __at_align__ double tmp_values[size()];
      vsx_st_d(_vec0, 0, tmp_values);
      vsx_st_d(_vec1, 16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(double));
    }
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
    return _nor();
  }

  bool has_inf_nan() const {
    vuint64 bits0 = (vuint64)(_vec0 - _vec0);
    vuint64 bits1 = (vuint64)(_vec1 - _vec1);
    vuint64 zero = {0ull, 0ull};
    return vec_any_ne(bits0, zero) || vec_any_ne(bits1, zero);
  }

  Vectorized<double> map(double (*const f)(double)) const {
    Vectorized<double> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  Vectorized<double> _nor() const {
    return {vec_nor(_vec0, _vec0), vec_nor(_vec1, _vec1)};
  }

  Vectorized<double> abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<double> acos() const {
    return {tensorplay::tpsleef::acos(_vec0), tensorplay::tpsleef::acos(_vec1)};
  }
  Vectorized<double> acosh() const {
    return map(std::acosh);
  }
  Vectorized<double> asin() const {
    return {tensorplay::tpsleef::asin(_vec0), tensorplay::tpsleef::asin(_vec1)};
  }
  Vectorized<double> asinh() const {
    return {tensorplay::tpsleef::asinh(_vec0), tensorplay::tpsleef::asinh(_vec1)};
  }
  Vectorized<double> atan() const {
    return {tensorplay::tpsleef::atan(_vec0), tensorplay::tpsleef::atan(_vec1)};
  }
  Vectorized<double> atanh() const {
    return {tensorplay::tpsleef::atanh(_vec0), tensorplay::tpsleef::atanh(_vec1)};
  }
  Vectorized<double> atan2(const Vectorized<double>& b) const {
    return {
        tensorplay::tpsleef::atan2(_vec0, b._vec0),
        tensorplay::tpsleef::atan2(_vec1, b._vec1)};
  }
  Vectorized<double> copysign(const Vectorized<double>& sign) const {
    return {
        (vfloat64)vec_or(
            (vuint64)vec_abs(_vec0),
            (vuint64)vec_and((vfloat64)sign._vec0, (vfloat64)vec_splats(-0.0))),
        (vfloat64)vec_or(
            (vuint64)vec_abs(_vec1),
            (vuint64)vec_and((vfloat64)sign._vec1, (vfloat64)vec_splats(-0.0)))};
  }
  Vectorized<double> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<double> erf() const {
    return {tensorplay::tpsleef::erf(_vec0), tensorplay::tpsleef::erf(_vec1)};
  }
  Vectorized<double> erfc() const {
    return {tensorplay::tpsleef::erfc(_vec0), tensorplay::tpsleef::erfc(_vec1)};
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> angle() const {
    const vbool64 neg0 = vec_cmplt(_vec0, vec_splats((double)0));
    const vbool64 neg1 = vec_cmplt(_vec1, vec_splats((double)0));
    const vbool64 nan0 = vec_cmpne(_vec0, _vec0);
    const vbool64 nan1 = vec_cmpne(_vec1, _vec1);
    Vectorized<double> tmp = blendv(
        Vectorized<double>(0),
        Vectorized<double>(3.141592653589793238463),
        Vectorized<double>((vfloat64)neg0, (vfloat64)neg1));
    return blendv(tmp, *this, Vectorized<double>((Vectorized<double>::vec_internal_type)nan0, (Vectorized<double>::vec_internal_type)nan1));
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
    return {tensorplay::tpsleef::exp(_vec0), tensorplay::tpsleef::exp(_vec1)};
  }
  Vectorized<double> exp2() const {
    return {tensorplay::tpsleef::exp2(_vec0), tensorplay::tpsleef::exp2(_vec1)};
  }
  Vectorized<double> expm1() const {
    return {tensorplay::tpsleef::expm1(_vec0), tensorplay::tpsleef::expm1(_vec1)};
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
    return {tensorplay::tpsleef::log(_vec0), tensorplay::tpsleef::log(_vec1)};
  }
  Vectorized<double> log2() const {
    return {tensorplay::tpsleef::log2(_vec0), tensorplay::tpsleef::log2(_vec1)};
  }
  Vectorized<double> log10() const {
    return {tensorplay::tpsleef::log10(_vec0), tensorplay::tpsleef::log10(_vec1)};
  }
  Vectorized<double> log1p() const {
    return {tensorplay::tpsleef::log1p(_vec0), tensorplay::tpsleef::log1p(_vec1)};
  }
  Vectorized<double> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<double> cos() const {
    return {tensorplay::tpsleef::cos(_vec0), tensorplay::tpsleef::cos(_vec1)};
  }
  Vectorized<double> cosh() const {
    return {tensorplay::tpsleef::cosh(_vec0), tensorplay::tpsleef::cosh(_vec1)};
  }
  Vectorized<double> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<double> hypot(const Vectorized<double>& b) const {
    return {
        tensorplay::tpsleef::hypot(_vec0, b._vec0),
        tensorplay::tpsleef::hypot(_vec1, b._vec1)};
  }
  Vectorized<double> neg() const {
    return {(vfloat64)vec_xor((vuint64)_vec0, vec_splats(0x8000000000000000ull)),
            (vfloat64)vec_xor((vuint64)_vec1, vec_splats(0x8000000000000000ull))};
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
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }
  Vectorized<double> sin() const {
    return {tensorplay::tpsleef::sin(_vec0), tensorplay::tpsleef::sin(_vec1)};
  }
  Vectorized<double> sinh() const {
    return {tensorplay::tpsleef::sinh(_vec0), tensorplay::tpsleef::sinh(_vec1)};
  }
  Vectorized<double> tan() const {
    return {tensorplay::tpsleef::tan(_vec0), tensorplay::tpsleef::tan(_vec1)};
  }
  Vectorized<double> tanh() const {
    return {tensorplay::tpsleef::tanh(_vec0), tensorplay::tpsleef::tanh(_vec1)};
  }
  Vectorized<double> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }
  Vectorized<double> frac() const {
    Vectorized<double> t = trunc();
    return {vec_sub(_vec0, t._vec0), vec_sub(_vec1, t._vec1)};
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
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<double> reciprocal() const {
    return {vec_div(vec_splats(1.0), _vec0), vec_div(vec_splats(1.0), _vec1)};
  }
  Vectorized<double> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {
    return {
        tensorplay::tpsleef::pow(_vec0, b._vec0),
        tensorplay::tpsleef::pow(_vec1, b._vec1)};
  }
  double reduce_add() const {
    vfloat64 s = vec_add(_vec0, _vec1);
    return s[0] + s[1];
  }
  double reduce_max() const {
    vfloat64 s = vec_max(_vec0, _vec1);
    double r = s[0];
    r = std::max(r, s[1]);
    return r;
  }
  double reduce_min() const {
    vfloat64 s = vec_min(_vec0, _vec1);
    double r = s[0];
    r = std::min(r, s[1]);
    return r;
  }

  TP_VSX_DEFINE_MEMBER_CMP(operator==, double, vec_cmpeq)
  TP_VSX_DEFINE_MEMBER_CMP(operator!=, double, vec_cmpne)
  TP_VSX_DEFINE_MEMBER_CMP(operator<, double, vec_cmplt)
  TP_VSX_DEFINE_MEMBER_CMP(operator<=, double, vec_cmple)
  TP_VSX_DEFINE_MEMBER_CMP(operator>, double, vec_cmpgt)
  TP_VSX_DEFINE_MEMBER_CMP(operator>=, double, vec_cmpge)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(eq, double, vec_cmpeq)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(ne, double, vec_cmpne)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(lt, double, vec_cmplt)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(le, double, vec_cmple)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(gt, double, vec_cmpgt)
  TP_VSX_DEFINE_MEMBER_OP_AND_ONE(ge, double, vec_cmpge)

  TP_VSX_DEFINE_MEMBER_OP(maximum, double, vec_max_nan_d)
  TP_VSX_DEFINE_MEMBER_OP(minimum, double, vec_min_nan_d)
};

inline vfloat64 vec_max_nan_d(const vfloat64& a, const vfloat64& b) {
  vfloat64 m = vec_max(a, b);
  vbool64 nan_a = vec_cmpne(a, a);
  vbool64 nan_b = vec_cmpne(b, b);
  m = (vfloat64)vec_sel(m, a, nan_a);
  return (vfloat64)vec_sel(m, b, nan_b);
}
inline vfloat64 vec_min_nan_d(const vfloat64& a, const vfloat64& b) {
  vfloat64 m = vec_min(a, b);
  vbool64 nan_a = vec_cmpne(a, a);
  vbool64 nan_b = vec_cmpne(b, b);
  m = (vfloat64)vec_sel(m, a, nan_a);
  return (vfloat64)vec_sel(m, b, nan_b);
}

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
  return {vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_mul(a.vec0(), b.vec0()), vec_mul(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_div(a.vec0(), b.vec0()), vec_div(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

template <>
Vectorized<double> inline fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return {vec_madd(a.vec0(), b.vec0(), c.vec0()),
          vec_madd(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  return {vec_min_nan_d(vec_max_nan_d(a.vec0(), min.vec0()), max.vec0()),
          vec_min_nan_d(vec_max_nan_d(a.vec1(), min.vec1()), max.vec1())};
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return {vec_max_nan_d(a.vec0(), min.vec0()), vec_max_nan_d(a.vec1(), min.vec1())};
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return {vec_min_nan_d(a.vec0(), max.vec0()), vec_min_nan_d(a.vec1(), max.vec1())};
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<double>::size());
       i += Vectorized<double>::size()) {
    vsx_st_d(vsx_ld_d(0, src + i), 0, dst + i);
    vsx_st_d(vsx_ld_d(16, src + i), 16, dst + i);
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
