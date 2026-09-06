#pragma once

// Vectorized<float> for the ZVECTOR tier: 256-bit emulation over two
// 128-bit z/Architecture vector registers. Transcendental functions call
// the SLEEF 4-float entry points once per half.

#include "cpu/vec/vec256/zarch/zarch_helpers.h"
#include "cpu/SpecialMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

inline zf32 z_max_nan(const zf32& a, const zf32& b);
inline zf32 z_min_nan(const zf32& a, const zf32& b);
inline zf32 z_cmpne_f(const zf32& a, const zf32& b) {
  return (zf32)~vec_cmpeq(a, b);
}

template <>
struct is_vec_specialized_for<float> : std::bool_constant<true> {};

template <>
class Vectorized<float> {
 private:
  zf32 _vec0;
  zf32 _vec1;

 public:
  using value_type = float;
  using vec_internal_type = zf32;
  using vec_internal_mask_type = zb32;
  using size_type = int;
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(zf32 v) : _vec0{v}, _vec1{v} {}
  Vectorized(zf32 v1, zf32 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(float scalar) : _vec0{vec_splats(scalar)}, _vec1{vec_splats(scalar)} {}
  Vectorized(
      float s0, float s1, float s2, float s3,
      float s4, float s5, float s6, float s7)
      : _vec0{zf32{s0, s1, s2, s3}}, _vec1{zf32{s4, s5, s6, s7}} {}

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xFF;
    return Vectorized<float>{
        (zf32)vec_sel(a._vec0, b._vec0, z_mask1(m)),
        (zf32)vec_sel(a._vec1, b._vec1, z_mask2(m))};
  }

  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    return Vectorized<float>{
        (zf32)vec_sel(a._vec0, b._vec0, (zb32)mask._vec0),
        (zf32)vec_sel(a._vec1, b._vec1, (zb32)mask._vec1)};
  }

  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<float>(
        base, base + step, base + 2 * step, base + 3 * step,
        base + 4 * step, base + 5 * step, base + 6 * step,
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
    if (count >= size()) {
      return {
          z_ld(0, reinterpret_cast<const float*>(ptr)),
          z_ld(16, reinterpret_cast<const float*>(ptr))};
    }
    __at_align__ float tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(float));
    return {z_ld(0, tmp_values), z_ld(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      z_st(_vec0, 0, reinterpret_cast<float*>(ptr));
      z_st(_vec1, 16, reinterpret_cast<float*>(ptr));
    } else if (count > 0) {
      __at_align__ float tmp_values[size()];
      z_st(_vec0, 0, tmp_values);
      z_st(_vec1, 16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(float));
    }
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
    return _not();
  }

  bool has_inf_nan() const {
    // Finite lanes cancel to +0 (all-zero bits); inf/NaN lanes produce NaN
    // bit patterns after self-subtraction.
    __at_align__ float tmp[size()];
    Vectorized<float> sub{_vec0 - _vec0, _vec1 - _vec1};
    sub.store(tmp);
    for (int i = 0; i < size(); ++i) {
      uint32_t bits;
      std::memcpy(&bits, &tmp[i], sizeof(bits));
      if (bits != 0) {
        return true;
      }
    }
    return false;
  }

  Vectorized<float> map(float (*const f)(float)) const {
    Vectorized<float> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  // Bitwise negation of the mask pattern produced by a comparison.
  Vectorized<float> _not() const {
    zu32 ones = vec_splats(0xffffffffu);
    return {(zf32)vec_xor((zu32)_vec0, ones), (zf32)vec_xor((zu32)_vec1, ones)};
  }

  Vectorized<float> abs() const {
    return {vec_abs(_vec0), vec_abs(_vec1)};
  }

  Vectorized<float> acos() const {
    return {tensorplay::tpsleef::acos(_vec0), tensorplay::tpsleef::acos(_vec1)};
  }
  Vectorized<float> acosh() const {
    return map(std::acosh);
  }
  Vectorized<float> asin() const {
    return {tensorplay::tpsleef::asin(_vec0), tensorplay::tpsleef::asin(_vec1)};
  }
  Vectorized<float> asinh() const {
    return {tensorplay::tpsleef::asinh(_vec0), tensorplay::tpsleef::asinh(_vec1)};
  }
  Vectorized<float> atan() const {
    return {tensorplay::tpsleef::atan(_vec0), tensorplay::tpsleef::atan(_vec1)};
  }
  Vectorized<float> atanh() const {
    return {tensorplay::tpsleef::atanh(_vec0), tensorplay::tpsleef::atanh(_vec1)};
  }
  Vectorized<float> atan2(const Vectorized<float>& b) const {
    return {
        tensorplay::tpsleef::atan2(_vec0, b._vec0),
        tensorplay::tpsleef::atan2(_vec1, b._vec1)};
  }
  Vectorized<float> copysign(const Vectorized<float>& sign) const {
    return {
        (zf32)vec_or(
            (zu32)vec_abs(_vec0),
            (zu32)vec_and((zf32)sign._vec0, (zf32)vec_splats(-0.0f))),
        (zf32)vec_or(
            (zu32)vec_abs(_vec1),
            (zu32)vec_and((zf32)sign._vec1, (zf32)vec_splats(-0.0f)))};
  }
  Vectorized<float> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<float> erf() const {
    return {tensorplay::tpsleef::erf(_vec0), tensorplay::tpsleef::erf(_vec1)};
  }
  Vectorized<float> erfc() const {
    return {tensorplay::tpsleef::erfc(_vec0), tensorplay::tpsleef::erfc(_vec1)};
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> angle() const {
    const zb32 neg0 = vec_cmplt(_vec0, vec_splats((float)0));
    const zb32 neg1 = vec_cmplt(_vec1, vec_splats((float)0));
    const zb32 nan0 = ~vec_cmpeq(_vec0, _vec0);
    const zb32 nan1 = ~vec_cmpeq(_vec1, _vec1);
    Vectorized<float> tmp = blendv(
        Vectorized<float>(0),
        Vectorized<float>(3.141592653589793238463f),
        Vectorized<float>((zb32)neg0, (zb32)neg1));
    return blendv(tmp, *this, Vectorized<float>((Vectorized<float>::vec_internal_type)nan0, (Vectorized<float>::vec_internal_type)nan1));
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
    return {tensorplay::tpsleef::exp(_vec0), tensorplay::tpsleef::exp(_vec1)};
  }
  Vectorized<float> exp2() const {
    return {tensorplay::tpsleef::exp2(_vec0), tensorplay::tpsleef::exp2(_vec1)};
  }
  Vectorized<float> expm1() const {
    return {tensorplay::tpsleef::expm1(_vec0), tensorplay::tpsleef::expm1(_vec1)};
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
    return {tensorplay::tpsleef::log(_vec0), tensorplay::tpsleef::log(_vec1)};
  }
  Vectorized<float> log2() const {
    return {tensorplay::tpsleef::log2(_vec0), tensorplay::tpsleef::log2(_vec1)};
  }
  Vectorized<float> log10() const {
    return {tensorplay::tpsleef::log10(_vec0), tensorplay::tpsleef::log10(_vec1)};
  }
  Vectorized<float> log1p() const {
    return {tensorplay::tpsleef::log1p(_vec0), tensorplay::tpsleef::log1p(_vec1)};
  }
  Vectorized<float> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<float> cos() const {
    return {tensorplay::tpsleef::cos(_vec0), tensorplay::tpsleef::cos(_vec1)};
  }
  Vectorized<float> cosh() const {
    return {tensorplay::tpsleef::cosh(_vec0), tensorplay::tpsleef::cosh(_vec1)};
  }
  Vectorized<float> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<float> hypot(const Vectorized<float>& b) const {
    return {
        tensorplay::tpsleef::hypot(_vec0, b._vec0),
        tensorplay::tpsleef::hypot(_vec1, b._vec1)};
  }
  Vectorized<float> neg() const {
    return {(zf32)vec_xor((zu32)_vec0, vec_splats(0x80000000u)),
            (zf32)vec_xor((zu32)_vec1, vec_splats(0x80000000u))};
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
    return {vec_roundc(_vec0), vec_roundc(_vec1)};
  }
  Vectorized<float> sin() const {
    return {tensorplay::tpsleef::sin(_vec0), tensorplay::tpsleef::sin(_vec1)};
  }
  Vectorized<float> sinh() const {
    return {tensorplay::tpsleef::sinh(_vec0), tensorplay::tpsleef::sinh(_vec1)};
  }
  Vectorized<float> tan() const {
    return {tensorplay::tpsleef::tan(_vec0), tensorplay::tpsleef::tan(_vec1)};
  }
  Vectorized<float> tanh() const {
    return {tensorplay::tpsleef::tanh(_vec0), tensorplay::tpsleef::tanh(_vec1)};
  }
  Vectorized<float> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }
  Vectorized<float> frac() const {
    Vectorized<float> t = trunc();
    return {_vec0 - t._vec0, _vec1 - t._vec1};
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
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<float> reciprocal() const {
    return {vec_splats(1.0f) / _vec0, vec_splats(1.0f) / _vec1};
  }
  Vectorized<float> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<float> pow(const Vectorized<float>& b) const {
    return {
        tensorplay::tpsleef::pow(_vec0, b._vec0),
        tensorplay::tpsleef::pow(_vec1, b._vec1)};
  }
  float reduce_add() const {
    zf32 s = _vec0 + _vec1;
    return s[0] + s[1] + s[2] + s[3];
  }
  float reduce_max() const {
    zf32 s = vec_max(_vec0, _vec1);
    float r = s[0];
    for (int i = 1; i < 4; ++i) {
      r = std::max(r, s[i]);
    }
    return r;
  }
  float reduce_min() const {
    zf32 s = vec_min(_vec0, _vec1);
    float r = s[0];
    for (int i = 1; i < 4; ++i) {
      r = std::min(r, s[i]);
    }
    return r;
  }

  TP_ZV_DEFINE_MEMBER_CMP(operator==, float, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_CMP(operator!=, float, z_cmpne_f)
  TP_ZV_DEFINE_MEMBER_CMP(operator<, float, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_CMP(operator<=, float, vec_cmple)
  TP_ZV_DEFINE_MEMBER_CMP(operator>, float, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_CMP(operator>=, float, vec_cmpge)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(eq, float, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ne, float, z_cmpne_f)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(lt, float, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(le, float, vec_cmple)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(gt, float, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ge, float, vec_cmpge)

  TP_ZV_DEFINE_MEMBER_OP(maximum, float, z_max_nan)
  TP_ZV_DEFINE_MEMBER_OP(minimum, float, z_min_nan)
};

// NaN-propagating min/max consistent with std::max/std::min on the scalar
// reference: a NaN on either side wins.
inline zf32 z_max_nan(const zf32& a, const zf32& b) {
  zf32 m = vec_max(a, b);
  zb32 nan_a = ~vec_cmpeq(a, a);
  zb32 nan_b = ~vec_cmpeq(b, b);
  m = (zf32)vec_sel(m, a, nan_a);
  return (zf32)vec_sel(m, b, nan_b);
}
inline zf32 z_min_nan(const zf32& a, const zf32& b) {
  zf32 m = vec_min(a, b);
  zb32 nan_a = ~vec_cmpeq(a, a);
  zb32 nan_b = ~vec_cmpeq(b, b);
  m = (zf32)vec_sel(m, a, nan_a);
  return (zf32)vec_sel(m, b, nan_b);
}

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
  return {a.vec0() + b.vec0(), a.vec1() + b.vec1()};
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {a.vec0() - b.vec0(), a.vec1() - b.vec1()};
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {a.vec0() * b.vec0(), a.vec1() * b.vec1()};
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {a.vec0() / b.vec0(), a.vec1() / b.vec1()};
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

template <>
Vectorized<float> inline fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return {vec_madd(a.vec0(), b.vec0(), c.vec0()),
          vec_madd(a.vec1(), b.vec1(), c.vec1())};
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  return {z_min_nan(z_max_nan(a.vec0(), min.vec0()), max.vec0()),
          z_min_nan(z_max_nan(a.vec1(), min.vec1()), max.vec1())};
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return {z_max_nan(a.vec0(), min.vec0()), z_max_nan(a.vec1(), min.vec1())};
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return {z_min_nan(a.vec0(), max.vec0()), z_min_nan(a.vec1(), max.vec1())};
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    z_st(z_ld(0, src + i), 0, dst + i);
    z_st(z_ld(16, src + i), 16, dst + i);
  }
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
