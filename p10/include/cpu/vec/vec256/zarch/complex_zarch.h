#pragma once

// Vectorized<std::complex<float>/<double>> for the ZVECTOR tier: a pair of
// the inner real vectors, mirroring how the reference layer wraps the real
// vector type. The interleaved (re, im) layout matches std::complex
// storage, so load/store pass straight through. Complex multiply, conj and
// abs run on vector arithmetic; the transcendental set stays on the scalar
// <cmath> complex reference.

#include "cpu/vec/vec256/zarch/zarch_helpers.h"
#include "cpu/vec/vec256/zarch/float_zarch.h"
#include "cpu/vec/vec256/zarch/double_zarch.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

// 128-bit unsigned bitcast targets shared with the complex classes.
using zcomplex_u8 = __vector unsigned char;
using zcomplex_u32 = __vector unsigned int;
using zcomplex_u64 = __vector unsigned long long;

namespace zcomplex {

// Swap the (re, im) pair inside each 128-bit register.
inline zf32 zswap(zf32 v) {
  return (zf32)vec_perm(
      (zcomplex_u8)v, (zcomplex_u8)v,
      (zcomplex_u8){4, 5, 6, 7, 0, 1, 2, 3,
                                 12, 13, 14, 15, 8, 9, 10, 11});
}
inline zf64 zswap(zf64 v) {
  return (zf64)vec_perm(
      (zcomplex_u8)v, (zcomplex_u8)v,
      (zcomplex_u8){8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7});
}
// Even float elements of two registers merged (double: vec_mergeh).
inline zf32 zmergee(zf32 x, zf32 y) {
  return (zf32)vec_perm(
      (zcomplex_u8)x, (zcomplex_u8)y,
      (zcomplex_u8){0, 1, 2, 3, 16, 17, 18, 19,
                                 8, 9, 10, 11, 24, 25, 26, 27});
}
inline zf32 zmergeo(zf32 x, zf32 y) {
  return (zf32)vec_perm(
      (zcomplex_u8)x, (zcomplex_u8)y,
      (zcomplex_u8){4, 5, 6, 7, 20, 21, 22, 23,
                                 12, 13, 14, 15, 28, 29, 30, 31});
}
inline zf64 zmergee(zf64 x, zf64 y) {
  return vec_mergeh(x, y);
}
inline zf64 zmergeo(zf64 x, zf64 y) {
  return vec_mergel(x, y);
}

// Bit masks over the inner float lanes.
inline zf32 zreal_mask_f32() {
  return (zf32)(zcomplex_u32){0xffffffffu, 0u, 0xffffffffu, 0u};
}
inline zf64 zreal_mask_f64() {
  return (zf64)(zcomplex_u64){0xffffffffffffffffull, 0ull};
}
inline zf32 zimag_mask_f32() {
  return (zf32)(zcomplex_u32){0u, 0xffffffffu, 0u, 0xffffffffu};
}
inline zf64 zimag_mask_f64() {
  return (zf64)(zcomplex_u64){0ull, 0xffffffffffffffffull};
}
inline zf32 zisign_mask_f32() {
  return (zf32)(zcomplex_u32){0u, 0x80000000u, 0u, 0x80000000u};
}
inline zf64 zisign_mask_f64() {
  return (zf64)(zcomplex_u64){0ull, 0x8000000000000000ull};
}
inline zf32 zrsign_mask_f32() {
  return (zf32)(zcomplex_u32){0x80000000u, 0u, 0x80000000u, 0u};
}
inline zf64 zrsign_mask_f64() {
  return (zf64)(zcomplex_u64){0x8000000000000000ull, 0ull};
}

} // namespace zcomplex

template <>
class Vectorized<std::complex<float>> {
 private:
  Vectorized<float> _vec;

 public:
  using value_type = std::complex<float>;
  using vinner_type = Vectorized<float>;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(vinner_type v) : _vec(v) {}
  Vectorized(value_type val)
      : _vec(val.real(), val.imag(), val.real(), val.imag(),
             val.real(), val.imag(), val.real(), val.imag()) {}
  Vectorized(value_type s1, value_type s2, value_type s3, value_type s4)
      : _vec(s1.real(), s1.imag(), s2.real(), s2.imag(),
             s3.real(), s3.imag(), s4.real(), s4.imag()) {}
  explicit Vectorized(vinner_type::vec_internal_type v0,
                      vinner_type::vec_internal_type v1)
      : _vec(v0, v1) {}

  const vinner_type& vec() const {
    return _vec;
  }

  template <int64_t mask>
  static Vectorized<value_type> blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xF;
    // Expand each complex-lane bit into its two scalar lanes.
    uint32_t bits[8];
    for (int i = 0; i < 4; ++i) {
      bits[2 * i] = bits[2 * i + 1] = ((m >> i) & 1) ? 0xffffffffu : 0u;
    }
    zcomplex_u32 sel;
    __builtin_memcpy(&sel, bits, sizeof(sel));
    auto v0 = (zf32)vec_sel(
        (zcomplex_u32)a._vec.vec0(),
        (zcomplex_u32)b._vec.vec0(), sel);
    auto v1 = (zf32)vec_sel(
        (zcomplex_u32)a._vec.vec1(),
        (zcomplex_u32)b._vec.vec1(), sel);
    return Vectorized<value_type>{vinner_type(v0, v1)};
  }

  static Vectorized<value_type> elwise_blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return blendv(a, b, mask);
  }

  static Vectorized<value_type> blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    auto v0 = (zf32)vec_sel(
        (zcomplex_u32)a._vec.vec0(),
        (zcomplex_u32)b._vec.vec0(),
        (zcomplex_u32)mask._vec.vec0());
    auto v1 = (zf32)vec_sel(
        (zcomplex_u32)a._vec.vec1(),
        (zcomplex_u32)b._vec.vec1(),
        (zcomplex_u32)mask._vec.vec1());
    return Vectorized<value_type>{vinner_type(v0, v1)};
  }

  static Vectorized<value_type> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vinner_type::loadu(ptr);
    }
    __at_align__ value_type tmp[size()] = {};
    std::memcpy(
        tmp, ptr, std::min<int64_t>(count, size()) * sizeof(value_type));
    return vinner_type::loadu(tmp);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _vec.store(ptr);
    } else if (count > 0) {
      __at_align__ value_type tmp[size()];
      _vec.store(tmp);
      std::memcpy(
          ptr, tmp, std::min<int64_t>(count, size()) * sizeof(value_type));
    }
  }

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

  Vectorized<value_type> abs_2_() const {
    auto a = _vec * _vec;
    auto permuted = vinner_type(zcomplex::zswap(a.vec0()), zcomplex::zswap(a.vec1()));
    auto sum = a + permuted;
    // Keep the even float lane of each 128-bit register in both slots.
    return Vectorized<value_type>{
        zcomplex::zmergee(sum.vec0(), sum.vec0()),
        zcomplex::zmergee(sum.vec1(), sum.vec1())};
  }
  Vectorized<value_type> abs_() const {
    // |z|^2 per complex lane, replicated into both scalar slots of the
    // merged register.
    zf32 vr = zcomplex::zmergee(_vec.vec0(), _vec.vec0());
    zf32 vi = zcomplex::zmergeo(_vec.vec0(), _vec.vec0());
    zf32 mag0 = vec_madd(vr, vr, vi * vi);
    return Vectorized<value_type>{vinner_type(mag0, mag0)};
  }
  Vectorized<value_type> abs() const {
    auto a = abs_();
    return Vectorized<value_type>{
        vec_sqrt(a._vec.vec0()), vec_sqrt(a._vec.vec1())};
  }
  Vectorized<value_type> real() const {
    return Vectorized<value_type>{
        (zf32)vec_and((zcomplex_u32)_vec.vec0(),
                      (zcomplex_u32)zcomplex::zreal_mask_f32()),
        (zf32)vec_and((zcomplex_u32)_vec.vec1(),
                      (zcomplex_u32)zcomplex::zreal_mask_f32())};
  }
  Vectorized<value_type> conj() const {
    return Vectorized<value_type>{
        (zf32)vec_xor((zcomplex_u32)_vec.vec0(),
                      (zcomplex_u32)zcomplex::zisign_mask_f32()),
        (zf32)vec_xor((zcomplex_u32)_vec.vec1(),
                      (zcomplex_u32)zcomplex::zisign_mask_f32())};
  }

  Vectorized<value_type> log() const {
    return map(std::log);
  }
  Vectorized<value_type> angle() const {
    Vectorized<value_type> ret;
    __at_align__ value_type tmp[size()], src[size()];
    store(src);
    for (int i = 0; i < 4; i += 2) {
      tmp[i] = value_type(std::atan2(src[i].imag(), src[i].real()), 0.f);
      tmp[i + 1] = value_type(0.f, 0.f);
    }
    return loadu(tmp);
  }

  Vectorized<value_type> sin() const {
    return map(std::sin);
  }
  Vectorized<value_type> sinh() const {
    return map(std::sinh);
  }
  Vectorized<value_type> cos() const {
    return map(std::cos);
  }
  Vectorized<value_type> cosh() const {
    return map(std::cosh);
  }
  Vectorized<value_type> ceil() const {
    return Vectorized<value_type>(vinner_type(
        vec_ceil(_vec.vec0()), vec_ceil(_vec.vec1())));
  }
  Vectorized<value_type> floor() const {
    return Vectorized<value_type>(vinner_type(
        vec_floor(_vec.vec0()), vec_floor(_vec.vec1())));
  }
  Vectorized<value_type> neg() const {
    return Vectorized<value_type>(value_type(0.f, 0.f)) - *this;
  }
  Vectorized<value_type> round() const {
    return Vectorized<value_type>(vinner_type(
        vec_roundc(_vec.vec0()), vec_roundc(_vec.vec1())));
  }
  Vectorized<value_type> tan() const {
    return map(std::tan);
  }
  Vectorized<value_type> tanh() const {
    return map(std::tanh);
  }
  Vectorized<value_type> trunc() const {
    return Vectorized<value_type>(vinner_type(
        vec_trunc(_vec.vec0()), vec_trunc(_vec.vec1())));
  }
  Vectorized<value_type> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<value_type> reciprocal() const {
    // 1/(a + bi) = (a - bi) / |a + bi|^2
    vinner_type c_d = Vectorized<value_type>(*this).conj()._vec;
    vinner_type abs = abs_2_()._vec;
    return Vectorized<value_type>{c_d / abs};
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
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  Vectorized<value_type> atan() const {
    // atan(z) = i/2 * ln((i + z)/(i - z))
    static const vinner_type ione{0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f};
    static const vinner_type ihalf{0.f, 0.5f, 0.f, 0.5f, 0.f, 0.5f, 0.f, 0.5f};
    auto sum = Vectorized<value_type>(ione) + *this;
    auto sub = Vectorized<value_type>(ione) - *this;
    auto ln = (sum / sub).log();
    return ln * Vectorized<value_type>(ihalf);
  }
  Vectorized<value_type> atanh() const {
    return map(std::atanh);
  }
  Vectorized<value_type> acos() const {
    static const vinner_type pi2{
        3.141592653589793238463f / 2.f, 0.f,
        3.141592653589793238463f / 2.f, 0.f,
        3.141592653589793238463f / 2.f, 0.f,
        3.141592653589793238463f / 2.f, 0.f};
    return Vectorized<value_type>(pi2) - asin();
  }
  Vectorized<value_type> asin() const {
    // asin(z) = -i * ln(iz + sqrt(1 - z^2))
    auto conj_v = conj();
    vinner_type b_a = vinner_type(
        zcomplex::zswap(conj_v._vec.vec0()),
        zcomplex::zswap(conj_v._vec.vec1()));
    auto ab = conj_v._vec * b_a;
    auto im = ab + ab;
    auto val_2 = _vec * _vec;
    auto val_2_swapped = vinner_type(
        zcomplex::zswap(val_2.vec0()), zcomplex::zswap(val_2.vec1()));
    auto re = Vectorized<value_type>(val_2) - Vectorized<value_type>(val_2_swapped);
    static const vinner_type ones{
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    auto re1 = Vectorized<value_type>(ones) - re;
    // elementwise sqrt of (re1 | im interleaved)
    vinner_type root_in = vinner_type(
        vec_sel(re1._vec.vec0(), im.vec0(), (zcomplex_u32)im.vec0()),
        vec_sel(re1._vec.vec1(), im.vec1(), (zcomplex_u32)im.vec1()));
    auto root = Vectorized<value_type>(vec_sqrt(root_in.vec0()),
                                       vec_sqrt(root_in.vec1()));
    auto ln = (Vectorized<value_type>(b_a) + root).log();
    return Vectorized<value_type>(
        vinner_type(zcomplex::zswap(ln._vec.vec0()),
                    zcomplex::zswap(ln._vec.vec1()))).conj();
  }
  Vectorized<value_type> exp() const {
    return map(std::exp);
  }
  Vectorized<value_type> expm1() const {
    return map([](const value_type& z) {
      return std::exp(z) - value_type(1, 0);
    });
  }

  Vectorized<value_type> eq(const Vectorized<value_type>& other) const {
    auto eq = (*this == other);
    // Collapse each complex lane: both slots must match.
    auto collapsed = eq & Vectorized<value_type>(
        vinner_type(zcomplex::zswap(eq._vec.vec0()),
                    zcomplex::zswap(eq._vec.vec1())));
    return collapsed & Vectorized<value_type>(vinner_type(
        vec_splats(1.f), vec_splats(1.f)));
  }
  Vectorized<value_type> ne(const Vectorized<value_type>& other) const {
    auto ne = (*this != other);
    auto collapsed = ne | Vectorized<value_type>(
        vinner_type(zcomplex::zswap(ne._vec.vec0()),
                    zcomplex::zswap(ne._vec.vec1())));
    return collapsed & Vectorized<value_type>(vinner_type(
        vec_splats(1.f), vec_splats(1.f)));
  }

  Vectorized<value_type> elwise_lt_mask(const Vectorized<value_type>& b) const {
    return Vectorized<value_type>{
        (zf32)vec_cmplt(_vec.vec0(), b._vec.vec0()),
        (zf32)vec_cmplt(_vec.vec1(), b._vec.vec1())};
  }
  Vectorized<value_type> elwise_gt_mask(const Vectorized<value_type>& b) const {
    return Vectorized<value_type>{
        (zf32)vec_cmpgt(_vec.vec0(), b._vec.vec0()),
        (zf32)vec_cmpgt(_vec.vec1(), b._vec.vec1())};
  }
  Vectorized<value_type> operator*(const Vectorized<value_type>& b) const {
    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i via lane shuffles.
    vinner_type bv = b._vec;
    vinner_type vi(
        zcomplex::zmergeo(bv.vec0(), bv.vec0()),
        zcomplex::zmergeo(bv.vec1(), bv.vec1()));
    vinner_type vr(
        zcomplex::zmergee(bv.vec0(), bv.vec0()),
        zcomplex::zmergee(bv.vec1(), bv.vec1()));
    vi = vinner_type(
        (zf32)vec_xor((zcomplex_u32)vi.vec0(),
                      (zcomplex_u32)zcomplex::zrsign_mask_f32()),
        (zf32)vec_xor((zcomplex_u32)vi.vec1(),
                      (zcomplex_u32)zcomplex::zrsign_mask_f32()));
    vinner_type ret = _vec * vr;
    vinner_type vx_swapped(
        zcomplex::zswap(_vec.vec0()), zcomplex::zswap(_vec.vec1()));
    ret = ret + vx_swapped * vi;
    return Vectorized<value_type>(ret);
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
    return Vectorized<value_type>(vinner_type(
        vec_cmpeq(_vec.vec0(), other._vec.vec0()),
        vec_cmpeq(_vec.vec1(), other._vec.vec1())));
  }
  Vectorized<value_type> operator!=(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(vinner_type(
        (zf32)vec_xor((zcomplex_u32)vec_cmpeq(_vec.vec0(), other._vec.vec0()),
                      (zcomplex_u32)vec_splats(0xffffffffu)),
        (zf32)vec_xor((zcomplex_u32)vec_cmpeq(_vec.vec1(), other._vec.vec1()),
                      (zcomplex_u32)vec_splats(0xffffffffu))));
  }
  Vectorized<value_type> operator+(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec + other._vec);
  }
  Vectorized<value_type> operator-(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec - other._vec);
  }
  Vectorized<value_type> operator&(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec & other._vec);
  }
  Vectorized<value_type> operator|(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec | other._vec);
  }
  Vectorized<value_type> operator^(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec ^ other._vec);
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
Vectorized<std::complex<float>> inline maximum(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  return Vectorized<std::complex<float>>::elwise_blendv(
      a, b, abs_a.elwise_lt_mask(abs_b));
}

template <>
Vectorized<std::complex<float>> inline minimum(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  return Vectorized<std::complex<float>>::elwise_blendv(
      a, b, abs_a.elwise_gt_mask(abs_b));
}

template <>
Vectorized<std::complex<float>> inline operator+(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  return Vectorized<std::complex<float>>(a.vec() + b.vec());
}

template <>
Vectorized<std::complex<float>> inline operator-(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  return Vectorized<std::complex<float>>(a.vec() - b.vec());
}

template <>
Vectorized<std::complex<float>> inline operator&(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  return Vectorized<std::complex<float>>(a.vec() & b.vec());
}

template <>
Vectorized<std::complex<float>> inline operator|(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  return Vectorized<std::complex<float>>(a.vec() | b.vec());
}

template <>
Vectorized<std::complex<float>> inline operator^(
    const Vectorized<std::complex<float>>& a,
    const Vectorized<std::complex<float>>& b) {
  return Vectorized<std::complex<float>>(a.vec() ^ b.vec());
}

template <>
class Vectorized<std::complex<double>> {
 private:
  Vectorized<double> _vec;

 public:
  using value_type = std::complex<double>;
  using vinner_type = Vectorized<double>;
  using size_type = int;
  static constexpr size_type kSize = 2;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(vinner_type v) : _vec(v) {}
  Vectorized(value_type val)
      : _vec(val.real(), val.imag(), val.real(), val.imag()) {}
  Vectorized(zf64 v0, zf64 v1) : _vec(v0, v1) {}
  Vectorized(value_type s1, value_type s2)
      : _vec(s1.real(), s1.imag(), s2.real(), s2.imag()) {}

  const vinner_type& vec() const {
    return _vec;
  }

  template <int64_t mask>
  static Vectorized<value_type> blend(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0x3;
    uint64_t bits[4];
    for (int i = 0; i < 2; ++i) {
      bits[2 * i] = bits[2 * i + 1] = ((m >> i) & 1) ? 0xffffffffffffffffull : 0ull;
    }
    zcomplex_u64 sel;
    __builtin_memcpy(&sel, bits, sizeof(sel));
    auto v0 = (zf64)vec_sel(
        (zcomplex_u64)a._vec.vec0(),
        (zcomplex_u64)b._vec.vec0(), sel);
    auto v1 = (zf64)vec_sel(
        (zcomplex_u64)a._vec.vec1(),
        (zcomplex_u64)b._vec.vec1(), sel);
    return Vectorized<value_type>{vinner_type(v0, v1)};
  }

  static Vectorized<value_type> elwise_blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    return blendv(a, b, mask);
  }

  static Vectorized<value_type> blendv(
      const Vectorized<value_type>& a,
      const Vectorized<value_type>& b,
      const Vectorized<value_type>& mask) {
    auto v0 = (zf64)vec_sel(
        (zcomplex_u64)a._vec.vec0(),
        (zcomplex_u64)b._vec.vec0(),
        (zcomplex_u64)mask._vec.vec0());
    auto v1 = (zf64)vec_sel(
        (zcomplex_u64)a._vec.vec1(),
        (zcomplex_u64)b._vec.vec1(),
        (zcomplex_u64)mask._vec.vec1());
    return Vectorized<value_type>{vinner_type(v0, v1)};
  }

  static Vectorized<value_type> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vinner_type::loadu(ptr);
    }
    __at_align__ value_type tmp[size()] = {};
    std::memcpy(
        tmp, ptr, std::min<int64_t>(count, size()) * sizeof(value_type));
    return vinner_type::loadu(tmp);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      _vec.store(ptr);
    } else if (count > 0) {
      __at_align__ value_type tmp[size()];
      _vec.store(tmp);
      std::memcpy(
          ptr, tmp, std::min<int64_t>(count, size()) * sizeof(value_type));
    }
  }

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

  Vectorized<value_type> abs_2_() const {
    auto a = _vec * _vec;
    auto permuted = vinner_type(
        zcomplex::zswap(a.vec0()), zcomplex::zswap(a.vec1()));
    auto sum = a + permuted;
    zf64 mag = zcomplex::zmergee(sum.vec0(), sum.vec0());
    return Vectorized<value_type>{vinner_type(mag, mag)};
  }
  Vectorized<value_type> abs() const {
    zf64 vr = zcomplex::zmergee(_vec.vec0(), _vec.vec0());
    zf64 vi = zcomplex::zmergeo(_vec.vec0(), _vec.vec0());
    zf64 mag = vec_sqrt(vec_madd(vr, vr, vi * vi));
    return Vectorized<value_type>{vinner_type(mag, mag)};
  }
  Vectorized<value_type> elwise_lt_mask(const Vectorized<value_type>& b) const {
    return Vectorized<value_type>{vinner_type(
        (zf64)vec_cmplt(_vec.vec0(), b._vec.vec0()),
        (zf64)vec_cmplt(_vec.vec1(), b._vec.vec1()))};
  }
  Vectorized<value_type> elwise_gt_mask(const Vectorized<value_type>& b) const {
    return Vectorized<value_type>{vinner_type(
        (zf64)vec_cmpgt(_vec.vec0(), b._vec.vec0()),
        (zf64)vec_cmpgt(_vec.vec1(), b._vec.vec1()))};
  }
  Vectorized<value_type> real() const {
    return Vectorized<value_type>{
        (zf64)vec_and((zcomplex_u64)_vec.vec0(),
                      (zcomplex_u64)zcomplex::zreal_mask_f64()),
        (zf64)vec_and((zcomplex_u64)_vec.vec1(),
                      (zcomplex_u64)zcomplex::zreal_mask_f64())};
  }
  Vectorized<value_type> conj() const {
    return Vectorized<value_type>{
        (zf64)vec_xor((zcomplex_u64)_vec.vec0(),
                      (zcomplex_u64)zcomplex::zisign_mask_f64()),
        (zf64)vec_xor((zcomplex_u64)_vec.vec1(),
                      (zcomplex_u64)zcomplex::zisign_mask_f64())};
  }

  Vectorized<value_type> log() const {
    return map(std::log);
  }
  Vectorized<value_type> angle() const {
    Vectorized<value_type> ret;
    __at_align__ value_type tmp[size()], src[size()];
    store(src);
    for (int i = 0; i < 2; i += 2) {
      tmp[i] = value_type(std::atan2(src[i].imag(), src[i].real()), 0.);
      tmp[i + 1] = value_type(0., 0.);
    }
    return loadu(tmp);
  }

  Vectorized<value_type> sin() const {
    return map(std::sin);
  }
  Vectorized<value_type> sinh() const {
    return map(std::sinh);
  }
  Vectorized<value_type> cos() const {
    return map(std::cos);
  }
  Vectorized<value_type> cosh() const {
    return map(std::cosh);
  }
  Vectorized<value_type> ceil() const {
    return Vectorized<value_type>(vinner_type(
        vec_ceil(_vec.vec0()), vec_ceil(_vec.vec1())));
  }
  Vectorized<value_type> floor() const {
    return Vectorized<value_type>(vinner_type(
        vec_floor(_vec.vec0()), vec_floor(_vec.vec1())));
  }
  Vectorized<value_type> neg() const {
    return Vectorized<value_type>(value_type(0., 0.)) - *this;
  }
  Vectorized<value_type> round() const {
    return Vectorized<value_type>(vinner_type(
        vec_roundc(_vec.vec0()), vec_roundc(_vec.vec1())));
  }
  Vectorized<value_type> tan() const {
    return map(std::tan);
  }
  Vectorized<value_type> tanh() const {
    return map(std::tanh);
  }
  Vectorized<value_type> trunc() const {
    return Vectorized<value_type>(vinner_type(
        vec_trunc(_vec.vec0()), vec_trunc(_vec.vec1())));
  }
  Vectorized<value_type> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<value_type> reciprocal() const {
    vinner_type c_d = conj()._vec;
    vinner_type abs = abs_2_()._vec;
    return Vectorized<value_type>{c_d / abs};
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
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  Vectorized<value_type> atan() const {
    static const vinner_type ione{0., 1., 0., 1.};
    static const vinner_type ihalf{0., 0.5, 0., 0.5};
    auto sum = Vectorized<value_type>(ione) + *this;
    auto sub = Vectorized<value_type>(ione) - *this;
    auto ln = (sum / sub).log();
    return ln * Vectorized<value_type>(ihalf);
  }
  Vectorized<value_type> atanh() const {
    return map(std::atanh);
  }
  Vectorized<value_type> acos() const {
    static const vinner_type pi2{
        3.141592653589793238463 / 2., 0.,
        3.141592653589793238463 / 2., 0.};
    return Vectorized<value_type>(pi2) - asin();
  }
  Vectorized<value_type> asin() const {
    auto conj_v = conj();
    vinner_type b_a = vinner_type(
        zcomplex::zswap(conj_v._vec.vec0()),
        zcomplex::zswap(conj_v._vec.vec1()));
    auto ab = conj_v._vec * b_a;
    auto im = ab + ab;
    auto val_2 = _vec * _vec;
    auto val_2_swapped = vinner_type(
        zcomplex::zswap(val_2.vec0()), zcomplex::zswap(val_2.vec1()));
    auto re = Vectorized<value_type>(val_2) - Vectorized<value_type>(val_2_swapped);
    static const vinner_type ones{1., 1., 1., 1.};
    auto re1 = Vectorized<value_type>(ones) - re;
    vinner_type root_in = vinner_type(
        vec_sel(re1._vec.vec0(), im.vec0(), (zcomplex_u64)im.vec0()),
        vec_sel(re1._vec.vec1(), im.vec1(), (zcomplex_u64)im.vec1()));
    auto root = Vectorized<value_type>(vec_sqrt(root_in.vec0()),
                                       vec_sqrt(root_in.vec1()));
    auto ln = (Vectorized<value_type>(b_a) + root).log();
    return Vectorized<value_type>(
        vinner_type(zcomplex::zswap(ln._vec.vec0()),
                    zcomplex::zswap(ln._vec.vec1()))).conj();
  }
  Vectorized<value_type> exp() const {
    return map(std::exp);
  }
  Vectorized<value_type> expm1() const {
    return map([](const value_type& z) {
      return std::exp(z) - value_type(1, 0);
    });
  }

  Vectorized<value_type> eq(const Vectorized<value_type>& other) const {
    auto eq = (*this == other);
    auto collapsed = eq & Vectorized<value_type>(
        vinner_type(zcomplex::zswap(eq._vec.vec0()),
                    zcomplex::zswap(eq._vec.vec1())));
    return collapsed & Vectorized<value_type>(vinner_type(
        vec_splats(1.), vec_splats(1.)));
  }
  Vectorized<value_type> ne(const Vectorized<value_type>& other) const {
    auto ne = (*this != other);
    auto collapsed = ne | Vectorized<value_type>(
        vinner_type(zcomplex::zswap(ne._vec.vec0()),
                    zcomplex::zswap(ne._vec.vec1())));
    return collapsed & Vectorized<value_type>(vinner_type(
        vec_splats(1.), vec_splats(1.)));
  }

  Vectorized<value_type> operator*(const Vectorized<value_type>& b) const {
    vinner_type bv = b._vec;
    vinner_type vi(
        zcomplex::zmergeo(bv.vec0(), bv.vec0()),
        zcomplex::zmergeo(bv.vec1(), bv.vec1()));
    vinner_type vr(
        zcomplex::zmergee(bv.vec0(), bv.vec0()),
        zcomplex::zmergee(bv.vec1(), bv.vec1()));
    vi = vinner_type(
        (zf64)vec_xor((zcomplex_u64)vi.vec0(),
                      (zcomplex_u64)zcomplex::zrsign_mask_f64()),
        (zf64)vec_xor((zcomplex_u64)vi.vec1(),
                      (zcomplex_u64)zcomplex::zrsign_mask_f64()));
    vinner_type ret = _vec * vr;
    vinner_type vx_swapped(
        zcomplex::zswap(_vec.vec0()), zcomplex::zswap(_vec.vec1()));
    ret = ret + vx_swapped * vi;
    return Vectorized<value_type>(ret);
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
    return Vectorized<value_type>(vinner_type(
        vec_cmpeq(_vec.vec0(), other._vec.vec0()),
        vec_cmpeq(_vec.vec1(), other._vec.vec1())));
  }
  Vectorized<value_type> operator!=(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(vinner_type(
        (zf64)vec_xor((zcomplex_u64)vec_cmpeq(_vec.vec0(), other._vec.vec0()),
                      (zcomplex_u64)vec_splats(0xffffffffffffffffull)),
        (zf64)vec_xor((zcomplex_u64)vec_cmpeq(_vec.vec1(), other._vec.vec1()),
                      (zcomplex_u64)vec_splats(0xffffffffffffffffull))));
  }
  Vectorized<value_type> operator+(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec + other._vec);
  }
  Vectorized<value_type> operator-(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec - other._vec);
  }
  Vectorized<value_type> operator&(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec & other._vec);
  }
  Vectorized<value_type> operator|(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec | other._vec);
  }
  Vectorized<value_type> operator^(const Vectorized<value_type>& other) const {
    return Vectorized<value_type>(_vec ^ other._vec);
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
Vectorized<std::complex<double>> inline maximum(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  return Vectorized<std::complex<double>>::elwise_blendv(
      a, b, abs_a.elwise_lt_mask(abs_b));
}

template <>
Vectorized<std::complex<double>> inline minimum(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  return Vectorized<std::complex<double>>::elwise_blendv(
      a, b, abs_a.elwise_gt_mask(abs_b));
}

template <>
Vectorized<std::complex<double>> inline operator+(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  return Vectorized<std::complex<double>>(a.vec() + b.vec());
}

template <>
Vectorized<std::complex<double>> inline operator-(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  return Vectorized<std::complex<double>>(a.vec() - b.vec());
}

template <>
Vectorized<std::complex<double>> inline operator&(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  return Vectorized<std::complex<double>>(a.vec() & b.vec());
}

template <>
Vectorized<std::complex<double>> inline operator|(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  return Vectorized<std::complex<double>>(a.vec() | b.vec());
}

template <>
Vectorized<std::complex<double>> inline operator^(
    const Vectorized<std::complex<double>>& a,
    const Vectorized<std::complex<double>>& b) {
  return Vectorized<std::complex<double>>(a.vec() ^ b.vec());
}


} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
