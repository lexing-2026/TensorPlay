#pragma once

// Vectorized<int64_t>/Vectorized<int32_t> for the ZVECTOR tier: 256-bit
// emulation over two 128-bit z/Architecture vector registers. Division has
// no vector integer form on the ISA, so operator/ keeps the scalar lane
// loop from the generic layer.

#include "cpu/vec/vec256/zarch/zarch_helpers.h"

#include <algorithm>
#include <climits>
#include <cstring>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

inline zi64 z_cmpne_i64(const zi64& a, const zi64& b) {
  return (zi64)~vec_cmpeq(a, b);
}
inline zi32 z_cmpne_i32(const zi32& a, const zi32& b) {
  return (zi32)~vec_cmpeq(a, b);
}

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

template <>
class Vectorized<int64_t> {
 private:
  zi64 _vec0;
  zi64 _vec1;

 public:
  using value_type = int64_t;
  using vec_internal_type = zi64;
  using vec_internal_mask_type = zb64;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(zi64 v) : _vec0{v}, _vec1{v} {}
  Vectorized(zi64 v1, zi64 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(int64_t scalar) : _vec0{vec_splats((long long)scalar)}, _vec1{vec_splats((long long)scalar)} {}
  Vectorized(int64_t s0, int64_t s1, int64_t s2, int64_t s3)
      : _vec0{zi64{s0, s1}}, _vec1{zi64{s2, s3}} {}

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static Vectorized<int64_t> blend(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xF;
    return Vectorized<int64_t>{
        (zi64)vec_sel(a._vec0, b._vec0, z_dbl_mask1(m)),
        (zi64)vec_sel(a._vec1, b._vec1, z_dbl_mask2(m))};
  }

  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    return Vectorized<int64_t>{
        (zi64)vec_sel(a._vec0, b._vec0, (zb64)mask._vec0),
        (zi64)vec_sel(a._vec1, b._vec1, (zb64)mask._vec1)};
  }

  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base, base + step, base + 2 * step, base + 3 * step);
  }

  static Vectorized<int64_t> set(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
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

  static Vectorized<int64_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return {
          z_ld_l(0, reinterpret_cast<const int64_t*>(ptr)),
          z_ld_l(16, reinterpret_cast<const int64_t*>(ptr))};
    }
    __at_align__ int64_t tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(int64_t));
    return {z_ld_l(0, tmp_values), z_ld_l(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      z_st_l(_vec0, 0, reinterpret_cast<int64_t*>(ptr));
      z_st_l(_vec1, 16, reinterpret_cast<int64_t*>(ptr));
    } else if (count > 0) {
      __at_align__ int64_t tmp_values[size()];
      z_st_l(_vec0, 0, tmp_values);
      z_st_l(_vec1, 16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(int64_t));
    }
  }

  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ int64_t tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<int64_t> isnan() const {
    return Vectorized<int64_t>(0);
  }

  bool has_inf_nan() const {
    return false;
  }

  Vectorized<int64_t> map(int64_t (*const f)(int64_t)) const {
    Vectorized<int64_t> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  int64_t reduce_add() const {
    zi64 s = _vec0 + _vec1;
    return s[0] + s[1];
  }
  int64_t reduce_max() const {
    zi64 s = vec_max(_vec0, _vec1);
    int64_t r = s[0];
    r = r >= s[1] ? r : s[1];
    return r;
  }
  int64_t reduce_min() const {
    zi64 s = vec_min(_vec0, _vec1);
    int64_t r = s[0];
    r = r <= s[1] ? r : s[1];
    return r;
  }

  TP_ZV_DEFINE_MEMBER_CMP(operator==, int64_t, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_CMP(operator!=, int64_t, z_cmpne_i64)
  TP_ZV_DEFINE_MEMBER_CMP(operator<, int64_t, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_CMP(operator<=, int64_t, vec_cmple)
  TP_ZV_DEFINE_MEMBER_CMP(operator>, int64_t, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_CMP(operator>=, int64_t, vec_cmpge)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(eq, int64_t, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ne, int64_t, z_cmpne_i64)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(lt, int64_t, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(le, int64_t, vec_cmple)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(gt, int64_t, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ge, int64_t, vec_cmpge)
  TP_ZV_DEFINE_MEMBER_OP(maximum, int64_t, vec_max)
  TP_ZV_DEFINE_MEMBER_OP(minimum, int64_t, vec_min)
};

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return a.maximum(b);
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return a.minimum(b);
}

template <>
Vectorized<int64_t> inline operator+(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {a.vec0() + b.vec0(), a.vec1() + b.vec1()};
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {a.vec0() - b.vec0(), a.vec1() - b.vec1()};
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {a.vec0() * b.vec0(), a.vec1() * b.vec1()};
}

template <>
Vectorized<int64_t> inline operator&(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> inline operator|(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<int64_t> inline operator^(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

template <>
struct is_vec_specialized_for<int32_t> : std::bool_constant<true> {};

template <>
class Vectorized<int32_t> {
 private:
  zi32 _vec0;
  zi32 _vec1;

 public:
  using value_type = int32_t;
  using vec_internal_type = zi32;
  using vec_internal_mask_type = zb32;
  using size_type = int;
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(zi32 v) : _vec0{v}, _vec1{v} {}
  Vectorized(zi32 v1, zi32 v2) : _vec0{v1}, _vec1{v2} {}
  Vectorized(int32_t scalar) : _vec0{vec_splats((int)scalar)}, _vec1{vec_splats((int)scalar)} {}
  Vectorized(
      int32_t s0, int32_t s1, int32_t s2, int32_t s3,
      int32_t s4, int32_t s5, int32_t s6, int32_t s7)
      : _vec0{zi32{s0, s1, s2, s3}}, _vec1{zi32{s4, s5, s6, s7}} {}

  const vec_internal_type& vec0() const {
    return _vec0;
  }
  const vec_internal_type& vec1() const {
    return _vec1;
  }

  template <int64_t mask>
  static Vectorized<int32_t> blend(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b) {
    constexpr uint32_t m = static_cast<uint32_t>(mask) & 0xFF;
    return Vectorized<int32_t>{
        (zi32)vec_sel(a._vec0, b._vec0, z_mask1(m)),
        (zi32)vec_sel(a._vec1, b._vec1, z_mask2(m))};
  }

  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    return Vectorized<int32_t>{
        (zi32)vec_sel(a._vec0, b._vec0, (zb32)mask._vec0),
        (zi32)vec_sel(a._vec1, b._vec1, (zb32)mask._vec1)};
  }

  template <typename step_t>
  static Vectorized<int32_t> arange(
      int32_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
        base, base + step, base + 2 * step, base + 3 * step,
        base + 4 * step, base + 5 * step, base + 6 * step,
        base + 7 * step);
  }

  static Vectorized<int32_t> set(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
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

  static Vectorized<int32_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return {
          z_ld_i(0, reinterpret_cast<const int32_t*>(ptr)),
          z_ld_i(16, reinterpret_cast<const int32_t*>(ptr))};
    }
    __at_align__ int32_t tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(int32_t));
    return {z_ld_i(0, tmp_values), z_ld_i(16, tmp_values)};
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      z_st_i(_vec0, 0, reinterpret_cast<int32_t*>(ptr));
      z_st_i(_vec1, 16, reinterpret_cast<int32_t*>(ptr));
    } else if (count > 0) {
      __at_align__ int32_t tmp_values[size()];
      z_st_i(_vec0, 0, tmp_values);
      z_st_i(_vec1, 16, tmp_values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(int32_t));
    }
  }

  const int32_t& operator[](int idx) const = delete;
  int32_t& operator[](int idx) = delete;

  int zero_mask() const {
    __at_align__ int32_t tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<int32_t> isnan() const {
    return Vectorized<int32_t>(0);
  }

  bool has_inf_nan() const {
    return false;
  }

  Vectorized<int32_t> map(int32_t (*const f)(int32_t)) const {
    Vectorized<int32_t> ret;
    for (int i = 0; i < size() / 2; i++) {
      ret._vec0[i] = f(_vec0[i]);
      ret._vec1[i] = f(_vec1[i]);
    }
    return ret;
  }

  int32_t reduce_add() const {
    zi32 s = _vec0 + _vec1;
    return s[0] + s[1] + s[2] + s[3];
  }
  int32_t reduce_max() const {
    zi32 s = vec_max(_vec0, _vec1);
    int32_t r = s[0];
    for (int i = 1; i < 4; ++i) {
      r = r >= s[i] ? r : s[i];
    }
    return r;
  }
  int32_t reduce_min() const {
    zi32 s = vec_min(_vec0, _vec1);
    int32_t r = s[0];
    for (int i = 1; i < 4; ++i) {
      r = r <= s[i] ? r : s[i];
    }
    return r;
  }

  TP_ZV_DEFINE_MEMBER_CMP(operator==, int32_t, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_CMP(operator!=, int32_t, z_cmpne_i32)
  TP_ZV_DEFINE_MEMBER_CMP(operator<, int32_t, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_CMP(operator<=, int32_t, vec_cmple)
  TP_ZV_DEFINE_MEMBER_CMP(operator>, int32_t, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_CMP(operator>=, int32_t, vec_cmpge)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(eq, int32_t, vec_cmpeq)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ne, int32_t, z_cmpne_i32)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(lt, int32_t, vec_cmplt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(le, int32_t, vec_cmple)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(gt, int32_t, vec_cmpgt)
  TP_ZV_DEFINE_MEMBER_OP_AND_ONE(ge, int32_t, vec_cmpge)
  TP_ZV_DEFINE_MEMBER_OP(maximum, int32_t, vec_max)
  TP_ZV_DEFINE_MEMBER_OP(minimum, int32_t, vec_min)
};

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return a.maximum(b);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return a.minimum(b);
}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {a.vec0() + b.vec0(), a.vec1() + b.vec1()};
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {a.vec0() - b.vec0(), a.vec1() - b.vec1()};
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {a.vec0() * b.vec0(), a.vec1() * b.vec1()};
}

template <>
Vectorized<int32_t> inline operator&(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {vec_and(a.vec0(), b.vec0()), vec_and(a.vec1(), b.vec1())};
}

template <>
Vectorized<int32_t> inline operator|(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {vec_or(a.vec0(), b.vec0()), vec_or(a.vec1(), b.vec1())};
}

template <>
Vectorized<int32_t> inline operator^(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return {vec_xor(a.vec0(), b.vec0()), vec_xor(a.vec1(), b.vec1())};
}

inline Vectorized<int64_t> operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  constexpr int64_t max_shift = sizeof(int64_t) * CHAR_BIT - 1;
  const Vectorized<int64_t> clamped = Vectorized<int64_t>::blendv(
      b, Vectorized<int64_t>(max_shift),
      (b < Vectorized<int64_t>(0)) | (b >= Vectorized<int64_t>(max_shift)));
  return {a.vec0() >> clamped.vec0(), a.vec1() >> clamped.vec1()};
}

inline Vectorized<int64_t> operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  constexpr int64_t max_shift = sizeof(int64_t) * CHAR_BIT;
  const Vectorized<int64_t> clamped = Vectorized<int64_t>::blendv(
      b, Vectorized<int64_t>(0),
      (b < Vectorized<int64_t>(0)) | (b >= Vectorized<int64_t>(max_shift)));
  return {a.vec0() << clamped.vec0(), a.vec1() << clamped.vec1()};
}

inline Vectorized<int32_t> operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr int32_t max_shift = sizeof(int32_t) * CHAR_BIT - 1;
  const Vectorized<int32_t> clamped = Vectorized<int32_t>::blendv(
      b, Vectorized<int32_t>(max_shift),
      (b < Vectorized<int32_t>(0)) | (b >= Vectorized<int32_t>(max_shift)));
  return {a.vec0() >> clamped.vec0(), a.vec1() >> clamped.vec1()};
}

inline Vectorized<int32_t> operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr int32_t max_shift = sizeof(int32_t) * CHAR_BIT;
  const Vectorized<int32_t> clamped = Vectorized<int32_t>::blendv(
      b, Vectorized<int32_t>(0),
      (b < Vectorized<int32_t>(0)) | (b >= Vectorized<int32_t>(max_shift)));
  return {a.vec0() << clamped.vec0(), a.vec1() << clamped.vec1()};
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
