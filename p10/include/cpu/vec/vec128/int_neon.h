#pragma once

// Vectorized<int64_t>/Vectorized<int32_t> for the aarch64 NEON tier.
// Integer division has no NEON instruction, so operator/ keeps the scalar
// lane loop from the generic layer.

#include "cpu/vec/vec128/neon_helpers.h"

#include <algorithm>
#include <climits>
#include <cstring>

namespace tensorplay {
namespace vec {
inline namespace CPU_CAPABILITY {

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

template <>
class Vectorized<int64_t> {
 private:
  int64x2_t values;

 public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type kSize = 2;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() : values(vdupq_n_s64(0)) {}
  Vectorized(int64x2_t v) : values(v) {}
  Vectorized(int64_t scalar) : values(vdupq_n_s64(scalar)) {}
  Vectorized(int64_t s0, int64_t s1) : values{s0, s1} {}

  operator int64x2_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<int64_t> blend(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b) {
    __at_align__ uint64_t bits[2];
    for (int i = 0; i < 2; ++i) {
      bits[i] = ((mask >> i) & 1) ? 0xffffffffffffffffull : 0ull;
    }
    uint64x2_t sel = vld1q_u64(bits);
    return Vectorized<int64_t>(vbslq_s64(sel, b.values, a.values));
  }

  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(mask.values), b.values, a.values));
  }

  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base, static_cast<int64_t>(base + step));
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
    }
    return b;
  }

  static Vectorized<int64_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vld1q_s64(reinterpret_cast<const int64_t*>(ptr));
    }
    __at_align__ int64_t tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(int64_t));
    return vld1q_s64(tmp_values);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vst1q_s64(reinterpret_cast<int64_t*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int64_t tmp_values[size()];
      vst1q_s64(tmp_values, values);
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
      if (tmp[i] == 0) mask |= (1 << i);
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
    __at_align__ int64_t tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  int64_t reduce_add() const {
    return vaddvq_s64(values);
  }
  int64_t reduce_max() const {
    // No integer horizontal-reduce spelling: pairwise lanes suffice at
    // width 2.
    int64_t lo = vgetq_lane_s64(values, 0);
    int64_t hi = vgetq_lane_s64(values, 1);
    return lo >= hi ? lo : hi;
  }
  int64_t reduce_min() const {
    int64_t lo = vgetq_lane_s64(values, 0);
    int64_t hi = vgetq_lane_s64(values, 1);
    return lo <= hi ? lo : hi;
  }

  TP_NEON_DEFINE_MEMBER_CMP(operator==, int64_t, s64, vceqq_s64, u64)
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vreinterpretq_s64_u64(
        veorq_u64(_tp_all_ones_u64(), vceqq_s64(values, other.values))));
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vreinterpretq_s64_u64(vcltq_s64(values, other.values)));
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vreinterpretq_s64_u64(vcleq_s64(values, other.values)));
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vreinterpretq_s64_u64(vcgtq_s64(values, other.values)));
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vreinterpretq_s64_u64(vcgeq_s64(values, other.values)));
  }
  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this == other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this != other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this > other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this >= other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this < other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vreinterpretq_u64_s64(*this <= other), vdupq_n_s64(1), vdupq_n_s64(0)));
  }
  Vectorized<int64_t> maximum(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vcgtq_s64(values, other.values), values, other.values));
  }
  Vectorized<int64_t> minimum(const Vectorized<int64_t>& other) const {
    return Vectorized<int64_t>(vbslq_s64(
        vcltq_s64(values, other.values), values, other.values));
  }
};

template <>
Vectorized<int64_t> operator+(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator-(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator*(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator/(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator&(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator|(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> operator^(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> maximum(const Vectorized<int64_t>&, const Vectorized<int64_t>&);
template <>
Vectorized<int64_t> minimum(const Vectorized<int64_t>&, const Vectorized<int64_t>&);

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {{
  return a.maximum(b);
}}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {{
  return a.minimum(b);
}}

template <>
Vectorized<int64_t> inline operator+(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return vaddq_s64(a, b);
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return vsubq_s64(a, b);
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  int64x2_t r;
  r = vsetq_lane_s64(vgetq_lane_s64(a, 0) * vgetq_lane_s64(b, 0), r, 0);
  r = vsetq_lane_s64(vgetq_lane_s64(a, 1) * vgetq_lane_s64(b, 1), r, 1);
  return r;
}

inline Vectorized<int64_t> operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // NEON shift counts are per-lane; negative or oversized counts are
  // clamped so the result stays well-defined. No integer min/max
  // intrinsic at 64-bit width, so clamp per lane.
  constexpr int64_t kMaxShift = sizeof(int64_t) * CHAR_BIT - 1;
  int64_t c0 = vgetq_lane_s64(b, 0);
  int64_t c1 = vgetq_lane_s64(b, 1);
  c0 = c0 < 0 ? 0 : (c0 > kMaxShift ? kMaxShift : c0);
  c1 = c1 < 0 ? 0 : (c1 > kMaxShift ? kMaxShift : c1);
  // vshl shifts left by a positive count; a right shift is a left shift by
  // the negated count.
  int64x2_t clamped = vdupq_n_s64(-c0);
  clamped = vsetq_lane_s64(-c1, clamped, 1);
  return vshlq_s64(a, clamped);
}

inline Vectorized<int64_t> operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  constexpr int64_t kMaxShift = sizeof(int64_t) * CHAR_BIT - 1;
  int64_t c0 = vgetq_lane_s64(b, 0);
  int64_t c1 = vgetq_lane_s64(b, 1);
  c0 = c0 < 0 ? 0 : (c0 > kMaxShift ? kMaxShift : c0);
  c1 = c1 < 0 ? 0 : (c1 > kMaxShift ? kMaxShift : c1);
  int64x2_t clamped = vsetq_lane_s64(c1, vdupq_n_s64(c0), 1);
  return vshlq_s64(a, clamped);
}

template <>
Vectorized<int64_t> inline operator&(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return vandq_s64(a, b);
}

template <>
Vectorized<int64_t> inline operator|(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return vorrq_s64(a, b);
}

template <>
Vectorized<int64_t> inline operator^(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return veorq_s64(a, b);
}

template <>
struct is_vec_specialized_for<int32_t> : std::bool_constant<true> {};

template <>
class Vectorized<int32_t> {
 private:
  int32x4_t values;

 public:
  using value_type = int32_t;
  using size_type = int;
  static constexpr size_type kSize = 4;
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() : values(vdupq_n_s32(0)) {}
  Vectorized(int32x4_t v) : values(v) {}
  Vectorized(int32_t scalar) : values(vdupq_n_s32(scalar)) {}
  Vectorized(int32_t s0, int32_t s1, int32_t s2, int32_t s3)
      : values{s0, s1, s2, s3} {}

  operator int32x4_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<int32_t> blend(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b) {
    __at_align__ uint32_t bits[4];
    for (int i = 0; i < 4; ++i) {
      bits[i] = ((mask >> i) & 1) ? 0xffffffffu : 0u;
    }
    uint32x4_t sel = vld1q_u32(bits);
    return Vectorized<int32_t>(vbslq_s32(sel, b.values, a.values));
  }

  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(mask.values), b.values, a.values));
  }

  template <typename step_t>
  static Vectorized<int32_t> arange(
      int32_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
        base,
        static_cast<int32_t>(base + step),
        static_cast<int32_t>(base + 2 * step),
        static_cast<int32_t>(base + 3 * step));
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
    }
    return b;
  }

  static Vectorized<int32_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
    }
    __at_align__ int32_t tmp_values[size()] = {};
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(int32_t));
    return vld1q_s32(tmp_values);
  }

  void store(void* ptr, int64_t count = size()) const {
    if (count >= size()) {
      vst1q_s32(reinterpret_cast<int32_t*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int32_t tmp_values[size()];
      vst1q_s32(tmp_values, values);
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
      if (tmp[i] == 0) mask |= (1 << i);
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
    __at_align__ int32_t tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  int32_t reduce_add() const {
    return vaddvq_s32(values);
  }
  int32_t reduce_max() const {
    return vmaxvq_s32(values);
  }
  int32_t reduce_min() const {
    return vminvq_s32(values);
  }

  TP_NEON_DEFINE_MEMBER_CMP(operator==, int32_t, s32, vceqq_s32, u32)
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vreinterpretq_s32_u32(
        veorq_u32(_tp_all_ones_u32(), vceqq_s32(values, other.values))));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vreinterpretq_s32_u32(vcltq_s32(values, other.values)));
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vreinterpretq_s32_u32(vcleq_s32(values, other.values)));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vreinterpretq_s32_u32(vcgtq_s32(values, other.values)));
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vreinterpretq_s32_u32(vcgeq_s32(values, other.values)));
  }
  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this == other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this != other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this > other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this >= other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this < other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const {
    return Vectorized<int32_t>(vbslq_s32(
        vreinterpretq_u32_s32(*this <= other), vdupq_n_s32(1), vdupq_n_s32(0)));
  }
  Vectorized<int32_t> maximum(const Vectorized<int32_t>& other) const {
    return vmaxq_s32(values, other.values);
  }
  Vectorized<int32_t> minimum(const Vectorized<int32_t>& other) const {
    return vminq_s32(values, other.values);
  }
};

template <>
Vectorized<int32_t> operator+(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator-(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator*(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator/(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator&(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator|(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> operator^(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> maximum(const Vectorized<int32_t>&, const Vectorized<int32_t>&);
template <>
Vectorized<int32_t> minimum(const Vectorized<int32_t>&, const Vectorized<int32_t>&);

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {{
  return a.maximum(b);
}}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {{
  return a.minimum(b);
}}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vaddq_s32(a, b);
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vsubq_s32(a, b);
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vmulq_s32(a, b);
}

inline Vectorized<int32_t> operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr int32_t kMaxShift = sizeof(int32_t) * CHAR_BIT - 1;
  int32x4_t clamped = vminq_s32(
      vmaxq_s32(b, vdupq_n_s32(0)), vdupq_n_s32(kMaxShift));
  return vshlq_s32(a, vnegq_s32(clamped));
}

inline Vectorized<int32_t> operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr int32_t kMaxShift = sizeof(int32_t) * CHAR_BIT - 1;
  int32x4_t clamped = vminq_s32(
      vmaxq_s32(b, vdupq_n_s32(0)), vdupq_n_s32(kMaxShift));
  return vshlq_s32(a, clamped);
}

template <>
Vectorized<int32_t> inline operator&(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vandq_s32(a, b);
}

template <>
Vectorized<int32_t> inline operator|(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vorrq_s32(a, b);
}

template <>
Vectorized<int32_t> inline operator^(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return veorq_s32(a, b);
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
