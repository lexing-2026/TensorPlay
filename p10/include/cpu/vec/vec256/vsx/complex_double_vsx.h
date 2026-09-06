#pragma once

// Vectorized<complex<double>> for the VSX tier: the interleaved (re, im)
// stream packed into two 128-bit registers (2 complex lanes).

#include "Complex.h"
#include "cpu/vec/vec256/vsx/vsx_helpers.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

const vbool64 zvd_real_mask = vbool64{0xFFFFFFFFFFFFFFFFull, 0x0ull};
const vbool64 zvd_imag_mask = vbool64{0x0ull, 0xFFFFFFFFFFFFFFFFull};
const vbool64 zvd_isign_mask = vbool64{0x0ull, 0x8000000000000000ull};
const vbool64 zvd_rsign_mask = vbool64{0x8000000000000000ull, 0x0ull};
const vfloat64 zvd_imag_one = vfloat64{0., 1.};
const vfloat64 zvd_imag_half = vfloat64{0., 0.5};
const vfloat64 zvd_pi_2 = vfloat64{3.141592653589793238463 / 2., 0.};

constexpr uint32_t zvd_complex_lane_mask(uint32_t mask) {
  uint32_t expanded = 0;
  if (mask & 1) expanded |= 0x3;
  if (mask & 2) expanded |= (0x3 << 2);
  return expanded;
}
constexpr vbool64 zvd_complex_mask1(uint32_t mask) {
  return vsx_dbl_mask1(zvd_complex_lane_mask(mask));
}
constexpr vbool64 zvd_complex_mask2(uint32_t mask) {
  return vsx_dbl_mask1(zvd_complex_lane_mask(mask) >> 2);
}

template <>
struct is_vec_specialized_for<complex<double>>
    : std::bool_constant<true> {};

template <>
class Vectorized<complex<double>> {
 private:
  vfloat64 _vec0;
  vfloat64 _vec1;

 public:
  using value_type = complex<double>;
  using vec_internal_type = vfloat64;
  using size_type = int;
  static constexpr size_type kSize = 2;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(vfloat64 v) : _vec0{v}, _vec1{v} {}
  Vectorized(vfloat64 v1, vfloat64 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(value_type val) {
    double re = val.real();
    double im = val.imag();
    _vec0 = vfloat64{re, im};
    _vec1 = vfloat64{re, im};
  }
  Vectorized(value_type c0, value_type c1) {
    _vec0 = vfloat64{c0.real(), c0.imag()};
    _vec1 = vfloat64{c1.real(), c1.imag()};
  }

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static Vectorized<value_type> blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = zvd_complex_lane_mask(
        static_cast<uint32_t>(mask) & 0x3);
    return Vectorized<value_type>{
        (vfloat64)vec_sel(a._vec0, b._vec0, vsx_dbl_mask1(m)),
        (vfloat64)vec_sel(a._vec1, b._vec1, vsx_dbl_mask2(m))};
  }

  static Vectorized<value_type> blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return Vectorized<value_type>{
        (vfloat64)vec_sel(a._vec0, b._vec0, (vbool64)mask._vec0),
        (vfloat64)vec_sel(a._vec1, b._vec1, (vbool64)mask._vec1)};
  }

  static Vectorized<value_type> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return {
          vsx_ld_d(0, reinterpret_cast<const double*>(ptr)),
          vsx_ld_d(16, reinterpret_cast<const double*>(ptr))};
    }
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(value_type));
    return {vsx_ld_d(0, tmp_values), vsx_ld_d(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vsx_st_d(_vec0, 0, reinterpret_cast<double*>(ptr));
      vsx_st_d(_vec1, 16, reinterpret_cast<double*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vsx_st_d(_vec0, 0, reinterpret_cast<double*>(tmp_values));
      vsx_st_d(_vec1, 16, reinterpret_cast<double*>(tmp_values));
      std::memcpy(
          ptr, tmp_values,
          std::min<int64_t>(count, size()) * sizeof(value_type));
    }
  }

  const value_type& operator[](int idx) const = delete;
  value_type& operator[](int idx) = delete;

  Vectorized<value_type> map(value_type (*const f)(value_type)) const {
    __at_align__ value_type tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<value_type> map(value_type (*const f)(const value_type&)) const {
    __at_align__ value_type tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  Vectorized<value_type> el_swapped() const {
    // Swap the two double slots of each register: with one complex lane
    // per register this swaps (re, im).
    return {_vec1, _vec0};
  }
  static Vectorized<value_type> el_mergee(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    return {
        (vfloat64)vec_mergee((vbool64)first._vec0, (vbool64)second._vec0),
        (vfloat64)vec_mergee((vbool64)first._vec1, (vbool64)second._vec1)};
  }
  static Vectorized<value_type> el_mergeo(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    return {
        (vfloat64)vec_mergeo((vbool64)first._vec0, (vbool64)second._vec0),
        (vfloat64)vec_mergeo((vbool64)first._vec1, (vbool64)second._vec1)};
  }
  Vectorized<value_type> el_mergee() const {
    return {(vfloat64)vec_mergee((vbool64)_vec0, (vbool64)_vec0),
            (vfloat64)vec_mergee((vbool64)_vec1, (vbool64)_vec1)};
  }
  Vectorized<value_type> el_mergeo() const {
    return {(vfloat64)vec_mergeo((vbool64)_vec0, (vbool64)_vec0),
            (vfloat64)vec_mergeo((vbool64)_vec1, (vbool64)_vec1)};
  }
  Vectorized<value_type> elwise_mult(const Vectorized<value_type>& b) const {
    return {vec_mul(_vec0, b._vec0), vec_mul(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_div(const Vectorized<value_type>& b) const {
    return {vec_div(_vec0, b._vec0), vec_div(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_lt_mask(const Vectorized<value_type>& b) const {
    return {(vfloat64)vec_cmplt(_vec0, b._vec0),
            (vfloat64)vec_cmplt(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_gt_mask(const Vectorized<value_type>& b) const {
    return {(vfloat64)vec_cmpgt(_vec0, b._vec0),
            (vfloat64)vec_cmpgt(_vec1, b._vec1)};
  }
  template <int64_t mask>
  static Vectorized<value_type> el_blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xF;
    return Vectorized<value_type>{
        (vfloat64)vec_sel(a._vec0, b._vec0, vsx_dbl_mask1(m)),
        (vfloat64)vec_sel(a._vec1, b._vec1, vsx_dbl_mask2(m))};
  }
  static Vectorized<value_type> elwise_blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return blendv(a, b, mask);
  }

  Vectorized<value_type> abs_2_() const {
    auto a = elwise_mult(*this);
    auto permuted = a.el_swapped();
    a = a + permuted;
    return a.el_mergee();
  }
  Vectorized<value_type> abs_() const {
    auto vi = el_mergeo();
    auto vr = el_mergee();
    return {vec_sqrt(vec_add(vec_mul(vr._vec0, vr._vec0),
                             vec_mul(vi._vec0, vi._vec0))),
            vec_sqrt(vec_add(vec_mul(vr._vec1, vr._vec1),
                             vec_mul(vi._vec1, vi._vec1)))};
  }
  Vectorized<value_type> abs() const {
    return abs_() & real();
  }
  Vectorized<value_type> real_() const {
    return *this & real();
  }
  Vectorized<value_type> real() const {
    return Vectorized<value_type>{
        (vfloat64)vec_and((vuint64)_vec0, (vuint64)zvd_real_mask),
        (vfloat64)vec_and((vuint64)_vec1, (vuint64)zvd_real_mask)};
  }
  Vectorized<value_type> imag_() const {
    return *this & imag();
  }
  Vectorized<value_type> imag() const {
    auto ret = Vectorized<value_type>{
        (vfloat64)vec_and((vuint64)_vec0, (vuint64)zvd_imag_mask),
        (vfloat64)vec_and((vuint64)_vec1, (vuint64)zvd_imag_mask)};
    // With one complex lane per register, the imag slot is lane 1; swap
    // registers to present it as the real part of a fresh lane.
    return Vectorized<value_type>{ret._vec1, ret._vec0};
  }
  Vectorized<value_type> conj_() const {
    return *this ^ conj();
  }
  Vectorized<value_type> conj() const {
    return Vectorized<value_type>{
        (vfloat64)vec_xor((vuint64)_vec0, (vuint64)zvd_isign_mask),
        (vfloat64)vec_xor((vuint64)_vec1, (vuint64)zvd_isign_mask)};
  }

  Vectorized<value_type> log() const {
    return map(tensorplay::log);
  }
  Vectorized<value_type> log2() const {
    auto ret = log();
    return ret.elwise_mult(Vectorized<value_type>(
        vfloat64{1.4426950408889634, 0.}));
  }
  Vectorized<value_type> log10() const {
    auto ret = log();
    return ret.elwise_mult(Vectorized<value_type>(
        vfloat64{0.43429448190325176, 0.}));
  }

  Vectorized<value_type> angle_() const {
    Vectorized<value_type> ret;
    for (int i = 0; i < 2; i += 2) {
      ret._vec0[i] = std::atan2(_vec0[i + 1], _vec0[i]);
      ret._vec1[i] = std::atan2(_vec1[i + 1], _vec1[i]);
    }
    return ret;
  }
  Vectorized<value_type> angle() const {
    return angle_() & real();
  }

  Vectorized<value_type> sin() const {
    return map(tensorplay::sin);
  }
  Vectorized<value_type> sinh() const {
    return map(tensorplay::sinh);
  }
  Vectorized<value_type> cos() const {
    return map(tensorplay::cos);
  }
  Vectorized<value_type> cosh() const {
    return map(tensorplay::cosh);
  }
  Vectorized<value_type> ceil() const {
    return {vec_ceil(_vec0), vec_ceil(_vec1)};
  }
  Vectorized<value_type> floor() const {
    return {vec_floor(_vec0), vec_floor(_vec1)};
  }
  Vectorized<value_type> neg() const {
    auto z = Vectorized<value_type>(value_type(0., 0.));
    return z - *this;
  }
  Vectorized<value_type> round() const {
    return {vec_rint(_vec0), vec_rint(_vec1)};
  }
  Vectorized<value_type> tan() const {
    return map(tensorplay::tan);
  }
  Vectorized<value_type> tanh() const {
    return map(tensorplay::tanh);
  }
  Vectorized<value_type> trunc() const {
    return {vec_trunc(_vec0), vec_trunc(_vec1)};
  }
  Vectorized<value_type> elwise_sqrt() const {
    return {vec_sqrt(_vec0), vec_sqrt(_vec1)};
  }
  Vectorized<value_type> sqrt() const {
    return map(tensorplay::sqrt);
  }
  Vectorized<value_type> reciprocal() const {
    auto c_d = *this ^ conj();
    auto abs = abs_2_();
    return c_d.elwise_div(abs);
  }
  Vectorized<value_type> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<value_type> pow(const Vectorized<value_type>& exp) const {
    __at_align__ value_type x_tmp[size()];
    __at_align__ value_type y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (const auto i : tensorplay::irange(size())) {
      x_tmp[i] = tensorplay::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  Vectorized<value_type> atan() const {
    auto ione = Vectorized<value_type>(zvd_imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log();
    return ln * Vectorized<value_type>(zvd_imag_half);
  }
  Vectorized<value_type> atanh() const {
    return map(tensorplay::atanh);
  }
  Vectorized<value_type> acos() const {
    return Vectorized<value_type>(zvd_pi_2) - asin();
  }
  Vectorized<value_type> asin() const {
    auto conj_v = conj();
    auto b_a = conj_v.el_swapped();
    auto ab = conj_v.elwise_mult(b_a);
    auto im = ab + ab;
    auto val_2 = elwise_mult(*this);
    auto val_2_swapped = val_2.el_swapped();
    auto re = horizontal_sub_perm(val_2, val_2_swapped);
    re = Vectorized<value_type>(
             vfloat64{1., 1.}) -
        re;
    auto root = el_blend<0xAA>(re, im).elwise_sqrt();
    auto ln = (b_a + root).log();
    return ln.el_swapped().conj();
  }
  Vectorized<value_type> exp() const {
    return map(tensorplay::exp);
  }
  Vectorized<value_type> expm1() const {
    return map(tensorplay::expm1);
  }

  Vectorized<value_type> eq(const Vectorized<value_type>& other) const {
    auto eq = (*this == other);
    auto collapsed = eq & eq.el_swapped();
    return collapsed & Vectorized<value_type>(
        vfloat64{1., 1.});
  }
  Vectorized<value_type> ne(const Vectorized<value_type>& other) const {
    auto ne = (*this != other);
    auto collapsed = ne | ne.el_swapped();
    return collapsed & Vectorized<value_type>(
        vfloat64{1., 1.});
  }

  static Vectorized<value_type> horizontal_add_perm(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    auto first_perm = first.el_swapped();
    auto second_perm = second.el_swapped();
    auto first_ret = first + first_perm;
    auto second_ret = second + second_perm;
    return el_mergee(first_ret, second_ret);
  }
  static Vectorized<value_type> horizontal_sub_perm(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    auto first_perm = first.el_swapped();
    auto second_perm = second.el_swapped();
    auto first_ret = first - first_perm;
    auto second_ret = second - second_perm;
    return el_mergee(first_ret, second_ret);
  }

  Vectorized<value_type> operator*(const Vectorized<value_type>& b) const {
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    vi = Vectorized<value_type>(
        (vfloat64)vec_xor((vuint64)vi._vec0, (vuint64)zvd_rsign_mask),
        (vfloat64)vec_xor((vuint64)vi._vec1, (vuint64)zvd_rsign_mask));
    auto ret = elwise_mult(vr);
    auto vx_swapped = el_swapped();
    ret = vx_swapped.elwise_mult(vi) + ret;
    return ret;
  }

  Vectorized<value_type> operator/(const Vectorized<value_type>& b) const {
    __at_align__ value_type tmp1[size()];
    __at_align__ value_type tmp2[size()];
    __at_align__ value_type out[size()];
    store(tmp1);
    b.store(tmp2);
    for (const auto i : tensorplay::irange(size())) {
      out[i] = tmp1[i] / tmp2[i];
    }
    return loadu(out);
  }

  Vectorized<value_type> operator==(const Vectorized<value_type>& other) const {
    return {(vfloat64)vec_cmpeq(_vec0, other._vec0), (vfloat64)vec_cmpeq(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator!=(const Vectorized<value_type>& other) const {
    return {(vfloat64)vec_cmpne(_vec0, other._vec0), (vfloat64)vec_cmpne(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator+(const Vectorized<value_type>& other) const {
    return {vec_add(_vec0, other._vec0), vec_add(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator-(const Vectorized<value_type>& other) const {
    return {vec_sub(_vec0, other._vec0), vec_sub(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator&(const Vectorized<value_type>& other) const {
    return {vec_and(_vec0, other._vec0), vec_and(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator|(const Vectorized<value_type>& other) const {
    return {vec_or(_vec0, other._vec0), vec_or(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator^(const Vectorized<value_type>& other) const {
    return {vec_xor(_vec0, other._vec0), vec_xor(_vec1, other._vec1)};
  }

  Vectorized<value_type> operator<(const Vectorized<value_type>&) const {
    throw std::runtime_error("comparison not supported for complex numbers");
  }
  Vectorized<value_type> operator<=(const Vectorized<value_type>&) const {
    throw std::runtime_error("comparison not supported for complex numbers");
  }
  Vectorized<value_type> operator>(const Vectorized<value_type>&) const {
    throw std::runtime_error("comparison not supported for complex numbers");
  }
  Vectorized<value_type> operator>=(const Vectorized<value_type>&) const {
    throw std::runtime_error("comparison not supported for complex numbers");
  }
};

template <>
Vectorized<complex<double>> inline maximum(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = abs_a.elwise_lt_mask(abs_b);
  return Vectorized<complex<double>>::elwise_blendv(a, b, mask);
}

template <>
Vectorized<complex<double>> inline minimum(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = abs_a.elwise_gt_mask(abs_b);
  return Vectorized<complex<double>>::elwise_blendv(a, b, mask);
}

template <>
Vectorized<complex<double>> inline operator+(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  return {vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<double>> inline operator-(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  return {vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<double>> inline operator&(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<double>> inline operator|(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<double>> inline operator^(
    const Vectorized<complex<double>>& a,
    const Vectorized<complex<double>>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
