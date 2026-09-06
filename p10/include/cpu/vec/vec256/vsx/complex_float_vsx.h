#pragma once

// Vectorized<complex<float>> for the VSX tier: the interleaved (re, im)
// stream packed into two 128-bit registers (4 complex lanes). Complex
// multiply/divide, conjugation, magnitude and element shuffles run on VSX
// intrinsics; transcendental functions use the native scalar complex layer.

#include "Complex.h"
#include "cpu/vec/vec256/vsx/vsx_helpers.h"

#include <cmath>
#include <stdexcept>
#include <cstring>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

// Complex lane masks and constants (float): each complex lane is a (re,
// im) float pair.
const vbool32 zvs_real_mask = vbool32{0xFFFFFFFFu, 0x0u, 0xFFFFFFFFu, 0x0u};
const vbool32 zvs_imag_mask = vbool32{0x0u, 0xFFFFFFFFu, 0x0u, 0xFFFFFFFFu};
const vbool32 zvs_sign_mask = (vbool32)vec_splats((int)0x80000000);
const vbool32 zvs_isign_mask = vbool32{0x0u, 0x80000000u, 0x0u, 0x80000000u};
const vbool32 zvs_rsign_mask = vbool32{0x80000000u, 0x0u, 0x80000000u, 0x0u};
const vfloat32 zvs_imag_one = vfloat32{0.f, 1.f, 0.f, 1.f};
const vfloat32 zvs_imag_half = vfloat32{0.f, 0.5f, 0.f, 0.5f};
const vfloat32 zvs_pi_2 = vfloat32{3.141592653589793238463f / 2.f, 0.f,
                                    3.141592653589793238463f / 2.f, 0.f};
// Byte swap of each (re, im) pair within a 128-bit register.
using zvs_u8_vec = __vector unsigned char;
const zvs_u8_vec zvs_swap_mask = zvs_u8_vec{4, 5, 6, 7, 0, 1, 2, 3,
                                            12, 13, 14, 15, 8, 9, 10, 11};

// Complex blend masks reuse the float-lane classifier; a complex-lane bit
// expands to two float-lane bits.
constexpr uint32_t zvs_complex_lane_mask(uint32_t mask) {
  uint32_t expanded = 0;
  if (mask & 1) expanded |= 0x3;
  if (mask & 2) expanded |= (0x3 << 2);
  if (mask & 4) expanded |= (0x3 << 4);
  if (mask & 8) expanded |= (0x3 << 6);
  return expanded;
}

constexpr vbool32 zvs_complex_mask1(uint32_t mask) {
  return vsx_mask1(zvs_complex_lane_mask(mask));
}
constexpr vbool32 zvs_complex_mask2(uint32_t mask) {
  return vsx_mask1(zvs_complex_lane_mask(mask) >> 4);
}

template <>
struct is_vec_specialized_for<complex<float>>
    : std::bool_constant<true> {};

template <>
class Vectorized<complex<float>> {
 private:
  vfloat32 _vec0;
  vfloat32 _vec1;

 public:
  using value_type = complex<float>;
  using vec_internal_type = vfloat32;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(vfloat32 v) : _vec0{v}, _vec1{v} {}
  Vectorized(vfloat32 v1, vfloat32 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(value_type val) {
    float re = val.real();
    float im = val.imag();
    _vec0 = vfloat32{re, im, re, im};
    _vec1 = vfloat32{re, im, re, im};
  }
  Vectorized(value_type c0, value_type c1, value_type c2, value_type c3) {
    _vec0 = vfloat32{c0.real(), c0.imag(), c1.real(), c1.imag()};
    _vec1 = vfloat32{c2.real(), c2.imag(), c3.real(), c3.imag()};
  }

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  // Each bit of mask selects one complex lane from b over a.
  template <int64_t mask>
  static Vectorized<value_type> blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = zvs_complex_lane_mask(
        static_cast<uint32_t>(mask) & 0xF);
    return Vectorized<value_type>{
        (vfloat32)vec_sel(a._vec0, b._vec0, vsx_mask1(m)),
        (vfloat32)vec_sel(a._vec1, b._vec1, vsx_mask2(m))};
  }

  static Vectorized<value_type> blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return Vectorized<value_type>{
        (vfloat32)vec_sel(a._vec0, b._vec0, (vbool32)mask._vec0),
        (vfloat32)vec_sel(a._vec1, b._vec1, (vbool32)mask._vec1)};
  }

  static Vectorized<value_type> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return {
          vsx_ld(0, reinterpret_cast<const float*>(ptr)),
          vsx_ld(16, reinterpret_cast<const float*>(ptr))};
    }
    __at_align__ value_type tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(value_type));
    return {vsx_ld(0, tmp_values), vsx_ld(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vsx_st(_vec0, 0, reinterpret_cast<float*>(ptr));
      vsx_st(_vec1, 16, reinterpret_cast<float*>(ptr));
    } else if (count > 0) {
      __at_align__ value_type tmp_values[size()];
      vsx_st(_vec0, 0, reinterpret_cast<float*>(tmp_values));
      vsx_st(_vec1, 16, reinterpret_cast<float*>(tmp_values));
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

  // Swap (re, im) within each complex lane.
  Vectorized<value_type> el_swapped() const {
    vfloat32 v0 = vec_perm(_vec0, _vec0, zvs_swap_mask);
    vfloat32 v1 = vec_perm(_vec1, _vec1, zvs_swap_mask);
    return {v0, v1};
  }
  // Even elements of each register pair merged into one register.
  static Vectorized<value_type> el_mergee(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    return {
        (vfloat32)vec_mergee((vbool32)first._vec0, (vbool32)second._vec0),
        (vfloat32)vec_mergee((vbool32)first._vec1, (vbool32)second._vec1)};
  }
  // Odd elements of each register pair merged into one register.
  static Vectorized<value_type> el_mergeo(
      const Vectorized<value_type>& first,
      const Vectorized<value_type>& second) {
    return {
        (vfloat32)vec_mergeo((vbool32)first._vec0, (vbool32)second._vec0),
        (vfloat32)vec_mergeo((vbool32)first._vec1, (vbool32)second._vec1)};
  }
  Vectorized<value_type> el_mergee() const {
    return {(vfloat32)vec_mergee((vbool32)_vec0, (vbool32)_vec0),
            (vfloat32)vec_mergee((vbool32)_vec1, (vbool32)_vec1)};
  }
  Vectorized<value_type> el_mergeo() const {
    return {(vfloat32)vec_mergeo((vbool32)_vec0, (vbool32)_vec0),
            (vfloat32)vec_mergeo((vbool32)_vec1, (vbool32)_vec1)};
  }
  // Elementwise comparisons producing all-ones/all-zeros complex lanes.
  Vectorized<value_type> elwise_lt_mask(const Vectorized<value_type>& b) const {
    return {(vfloat32)vec_cmplt(_vec0, b._vec0),
            (vfloat32)vec_cmplt(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_gt_mask(const Vectorized<value_type>& b) const {
    return {(vfloat32)vec_cmpgt(_vec0, b._vec0),
            (vfloat32)vec_cmpgt(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_mult(const Vectorized<value_type>& b) const {
    return {vec_mul(_vec0, b._vec0), vec_mul(_vec1, b._vec1)};
  }
  Vectorized<value_type> elwise_div(const Vectorized<value_type>& b) const {
    return {vec_div(_vec0, b._vec0), vec_div(_vec1, b._vec1)};
  }
  template <int64_t mask>
  static Vectorized<value_type> el_blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xFF;
    return Vectorized<value_type>{
        (vfloat32)vec_sel(a._vec0, b._vec0, vsx_mask1(m)),
        (vfloat32)vec_sel(a._vec1, b._vec1, vsx_mask2(m))};
  }
  static Vectorized<value_type> elwise_blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return blendv(a, b, mask);
  }

  // Square modulus in every (re, im) slot.
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
        (vfloat32)vec_and((vuint32)_vec0, (vuint32)zvs_real_mask),
        (vfloat32)vec_and((vuint32)_vec1, (vuint32)zvs_real_mask)};
  }
  Vectorized<value_type> imag_() const {
    return *this & imag();
  }
  Vectorized<value_type> imag() const {
    // Keep the imaginary parts, then slide each into the real slot.
    auto ret = Vectorized<value_type>{
        (vfloat32)vec_and((vuint32)_vec0, (vuint32)zvs_imag_mask),
        (vfloat32)vec_and((vuint32)_vec1, (vuint32)zvs_imag_mask)};
    return Vectorized<value_type>{
        (vfloat32)vec_sldw(ret._vec0, ret._vec0, 3),
        (vfloat32)vec_sldw(ret._vec1, ret._vec1, 3)};
  }
  Vectorized<value_type> conj_() const {
    return *this ^ conj();
  }
  Vectorized<value_type> conj() const {
    // Flip the sign of the imaginary lanes only.
    return Vectorized<value_type>{
        (vfloat32)vec_xor((vuint32)_vec0, (vuint32)zvs_isign_mask),
        (vfloat32)vec_xor((vuint32)_vec1, (vuint32)zvs_isign_mask)};
  }

  Vectorized<value_type> log() const {
    return map(tensorplay::log);
  }
  Vectorized<value_type> log2() const {
    auto ret = log();
    return ret.elwise_mult(Vectorized<value_type>(
        vfloat32{1.4426950408889634f, 0.f, 1.4426950408889634f, 0.f}));
  }
  Vectorized<value_type> log10() const {
    auto ret = log();
    return ret.elwise_mult(Vectorized<value_type>(
        vfloat32{0.43429448190325176f, 0.f, 0.43429448190325176f, 0.f}));
  }

  Vectorized<value_type> angle_() const {
    Vectorized<value_type> ret;
    for (int i = 0; i < 4; i += 2) {
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
    auto z = Vectorized<value_type>(value_type(0.f, 0.f));
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
    // 1/(a + bi) = (a - bi) / |a + bi|^2
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
    // atan(z) = i/2 * ln((i + z)/(i - z))
    auto ione = Vectorized<value_type>(zvs_imag_one);
    auto sum = ione + *this;
    auto sub = ione - *this;
    auto ln = (sum / sub).log();
    return ln * Vectorized<value_type>(zvs_imag_half);
  }
  Vectorized<value_type> atanh() const {
    return map(tensorplay::atanh);
  }
  Vectorized<value_type> acos() const {
    // acos(z) = pi/2 - asin(z)
    return Vectorized<value_type>(zvs_pi_2) - asin();
  }
  Vectorized<value_type> asin() const {
    // asin(z) = -i * ln(iz + sqrt(1 - z^2))
    auto conj_v = conj();
    auto b_a = conj_v.el_swapped();
    auto ab = conj_v.elwise_mult(b_a);
    auto im = ab + ab;
    auto val_2 = elwise_mult(*this);
    auto val_2_swapped = val_2.el_swapped();
    auto re = horizontal_sub_perm(val_2, val_2_swapped);
    re = Vectorized<value_type>(
             vfloat32{1.f, 1.f, 1.f, 1.f}) -
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
    // Both parts must compare equal: AND the two lane positions of the
    // elementwise comparison so each complex lane collapses to 0/1 in its
    // real slot and its imag slot keeps the same value.
    auto eq = (*this == other);
    auto collapsed = eq & eq.el_swapped();
    return collapsed & Vectorized<value_type>(
        vfloat32{1.f, 1.f, 1.f, 1.f});
  }
  Vectorized<value_type> ne(const Vectorized<value_type>& other) const {
    auto ne = (*this != other);
    auto collapsed = ne | ne.el_swapped();
    return collapsed & Vectorized<value_type>(
        vfloat32{1.f, 1.f, 1.f, 1.f});
  }

  // (ac - bd) + (ad + bc)i for each adjacent (a,b)/(c,d) pair.
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
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i; the merge/swap form keeps
    // it to register shuffles plus multiplies.
    auto vi = b.el_mergeo();
    auto vr = b.el_mergee();
    vi = Vectorized<value_type>(
        (vfloat32)vec_xor((vuint32)vi._vec0, (vuint32)zvs_rsign_mask),
        (vfloat32)vec_xor((vuint32)vi._vec1, (vuint32)zvs_rsign_mask));
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
    return {(vfloat32)vec_cmpeq(_vec0, other._vec0), (vfloat32)vec_cmpeq(_vec1, other._vec1)};
  }
  Vectorized<value_type> operator!=(const Vectorized<value_type>& other) const {
    return {(vfloat32)vec_cmpne(_vec0, other._vec0), (vfloat32)vec_cmpne(_vec1, other._vec1)};
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
    // Complex numbers have no total order.
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
Vectorized<complex<float>> inline maximum(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  // Ordered by square modulus, matching the complex max convention.
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = abs_a.elwise_lt_mask(abs_b);
  return Vectorized<complex<float>>::elwise_blendv(a, b, mask);
}

template <>
Vectorized<complex<float>> inline minimum(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = abs_a.elwise_gt_mask(abs_b);
  return Vectorized<complex<float>>::elwise_blendv(a, b, mask);
}

template <>
Vectorized<complex<float>> inline operator+(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  return {vec_add(a.vec0(), b.vec0()), vec_add(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<float>> inline operator-(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  return {vec_sub(a.vec0(), b.vec0()), vec_sub(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<float>> inline operator&(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<float>> inline operator|(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<complex<float>> inline operator^(
    const Vectorized<complex<float>>& a,
    const Vectorized<complex<float>>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
