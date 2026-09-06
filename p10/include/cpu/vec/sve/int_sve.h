#pragma once

// Vectorized<int64_t>/Vectorized<int32_t> for the SVE tiers. Integer
// division has no SVE instruction, so operator/ keeps the scalar lane loop
// from the generic layer.

#include "cpu/vec/sve/sve_helpers.h"

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
  vls_int64_t values;

 public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type kSize = VECTOR_WIDTH / sizeof(int64_t);
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(svint64_t v) : values(v) {}
  Vectorized(int64_t scalar) : values(svdup_n_s64(scalar)) {}
  Vectorized(int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    int64_t buf[4] = {s0, s1, s2, s3};
    static_assert(sizeof(buf) >= sizeof(int64_t) * size(),
                  "SVE lane count exceeds the constructor's fixed width");
    values = svld1_s64(sve_first_f64(size()), buf);
  }

  operator svint64_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<int64_t> blend(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b) {
    svbool_t m = sve_lane_pred_f64<mask, size()>();
    return svsel_s64(m, b.values, a.values);
  }

  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    // Mask lanes are all-ones/all-zeros from the comparison operators.
    svbool_t m = svcmpne_n_s64(svptrue_b64(), mask.values, 0);
    return svsel_s64(m, b.values, a.values);
  }

  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    svint64_t scaled = svmul_n_s64_x(
        svptrue_b64(), svindex_s64(0, 1), static_cast<int64_t>(step));
    return svadd_s64_x(svptrue_b64(), svdup_n_s64(base), scaled);
  }

  static Vectorized<int64_t> set(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      int64_t count = size()) {
    return svsel_s64(sve_first_f64(count), b.values, a.values);
  }

  static Vectorized<int64_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return svld1_s64(sve_first_f64(size()),
                       reinterpret_cast<const int64_t*>(ptr));
    }
    return svld1_s64(sve_first_f64(count),
                     reinterpret_cast<const int64_t*>(ptr));
  }

  void store(void* ptr, int64_t count = size()) const {
    svst1_s64(sve_first_f64(count), reinterpret_cast<int64_t*>(ptr), values);
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
    __at_align__ int64_t tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  int64_t reduce_add() const {
    return svaddv_s64(svptrue_b64(), values);
  }
  int64_t reduce_max() const {
    return svmaxv_s64(svptrue_b64(), values);
  }
  int64_t reduce_min() const {
    return svminv_s64(svptrue_b64(), values);
  }

  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmpeq_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmpne_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmplt_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmple_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmpgt_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return sve_cmp_s64(svcmpge_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmpeq_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmpne_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmpgt_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmpge_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmplt_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const {
    return sve_cmp01_s64(svcmple_s64(svptrue_b64(), values, other.values));
  }
  Vectorized<int64_t> maximum(const Vectorized<int64_t>& other) const {
    return svmax_s64_x(svptrue_b64(), values, other.values);
  }
  Vectorized<int64_t> minimum(const Vectorized<int64_t>& other) const {
    return svmin_s64_x(svptrue_b64(), values, other.values);
  }

 private:
  static svint64_t sve_cmp_s64(svbool_t m) {
    return svsel_s64(
        m,
        svreinterpret_s64_u64(svdup_n_u64(0xffffffffffffffffull)),
        svdup_n_s64(0));
  }
  static svint64_t sve_cmp01_s64(svbool_t m) {
    return svsel_s64(m, svdup_n_s64(1), svdup_n_s64(0));
  }
};

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svmax_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svmin_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline operator+(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svadd_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svsub_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svmul_s64_x(svptrue_b64(), a, b);
}

inline Vectorized<int64_t> operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // SVE shift amounts are taken per lane from an unsigned vector; clamp
  // negative and out-of-range counts so the result stays well-defined.
  constexpr uint64_t max_shift = sizeof(int64_t) * CHAR_BIT - 1;
  svbool_t pg = svptrue_b64();
  svuint64_t bu = svreinterpret_u64_s64(b);
  svbool_t neg = svcmplt_n_s64(pg, b, 0);
  svbool_t big = svcmpgt_n_u64(pg, bu, max_shift);
  svuint64_t clamped = svsel_u64(big, svdup_n_u64(max_shift), bu);
  clamped = svsel_u64(neg, svdup_n_u64(max_shift), clamped);
  return svasr_s64_x(pg, a, clamped);
}

inline Vectorized<int64_t> operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  constexpr uint64_t max_shift = sizeof(int64_t) * CHAR_BIT;
  svbool_t pg = svptrue_b64();
  svuint64_t bu = svreinterpret_u64_s64(b);
  svbool_t neg = svcmplt_n_s64(pg, b, 0);
  svbool_t big = svcmpgt_n_u64(pg, bu, max_shift);
  svuint64_t clamped = svsel_u64(big, svdup_n_u64(0), bu);
  clamped = svsel_u64(neg, svdup_n_u64(0), clamped);
  return svlsl_s64_x(pg, a, clamped);
}

template <>
Vectorized<int64_t> inline operator&(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svand_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline operator|(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svorr_s64_x(svptrue_b64(), a, b);
}

template <>
Vectorized<int64_t> inline operator^(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return sveor_s64_x(svptrue_b64(), a, b);
}

template <>
struct is_vec_specialized_for<int32_t> : std::bool_constant<true> {};

template <>
class Vectorized<int32_t> {
 private:
  vls_int32_t values;

 public:
  using value_type = int32_t;
  using size_type = int;
  static constexpr size_type kSize = VECTOR_WIDTH / sizeof(int32_t);
  static constexpr size_type size() {
    return kSize;
  }

  Vectorized() = default;
  Vectorized(svint32_t v) : values(v) {}
  Vectorized(int32_t scalar) : values(svdup_n_s32(scalar)) {}
  Vectorized(int32_t s0, int32_t s1, int32_t s2, int32_t s3,
             int32_t s4, int32_t s5, int32_t s6, int32_t s7) {
    int32_t buf[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    static_assert(sizeof(buf) >= sizeof(int32_t) * size(),
                  "SVE lane count exceeds the constructor's fixed width");
    values = svld1_s32(sve_first_f32(size()), buf);
  }

  operator svint32_t() const {
    return values;
  }

  template <int64_t mask>
  static Vectorized<int32_t> blend(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b) {
    svbool_t m = sve_lane_pred_f32<mask, size()>();
    return svsel_s32(m, b.values, a.values);
  }

  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    svbool_t m = svcmpne_n_s32(svptrue_b32(), mask.values, 0);
    return svsel_s32(m, b.values, a.values);
  }

  template <typename step_t>
  static Vectorized<int32_t> arange(
      int32_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    svint32_t scaled = svmul_n_s32_x(
        svptrue_b32(), svindex_s32(0, 1), static_cast<int32_t>(step));
    return svadd_s32_x(svptrue_b32(), svdup_n_s32(base), scaled);
  }

  static Vectorized<int32_t> set(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      int64_t count = size()) {
    return svsel_s32(sve_first_f32(count), b.values, a.values);
  }

  static Vectorized<int32_t> loadu(const void* ptr, int64_t count = size()) {
    if (count >= size()) {
      return svld1_s32(sve_first_f32(size()),
                       reinterpret_cast<const int32_t*>(ptr));
    }
    return svld1_s32(sve_first_f32(count),
                     reinterpret_cast<const int32_t*>(ptr));
  }

  void store(void* ptr, int64_t count = size()) const {
    svst1_s32(sve_first_f32(count), reinterpret_cast<int32_t*>(ptr), values);
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
    __at_align__ int32_t tmp[size()];
    store(tmp);
    for (const auto i : tensorplay::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  int32_t reduce_add() const {
    return svaddv_s32(svptrue_b32(), values);
  }
  int32_t reduce_max() const {
    return svmaxv_s32(svptrue_b32(), values);
  }
  int32_t reduce_min() const {
    return svminv_s32(svptrue_b32(), values);
  }

  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmpeq_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmpne_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmplt_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmple_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmpgt_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    return sve_cmp_s32(svcmpge_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmpeq_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmpne_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmpgt_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmpge_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmplt_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const {
    return sve_cmp01_s32(svcmple_s32(svptrue_b32(), values, other.values));
  }
  Vectorized<int32_t> maximum(const Vectorized<int32_t>& other) const {
    return svmax_s32_x(svptrue_b32(), values, other.values);
  }
  Vectorized<int32_t> minimum(const Vectorized<int32_t>& other) const {
    return svmin_s32_x(svptrue_b32(), values, other.values);
  }

 private:
  static svint32_t sve_cmp_s32(svbool_t m) {
    return svsel_s32(
        m,
        svreinterpret_s32_u32(svdup_n_u32(0xffffffffu)),
        svdup_n_s32(0));
  }
  static svint32_t sve_cmp01_s32(svbool_t m) {
    return svsel_s32(m, svdup_n_s32(1), svdup_n_s32(0));
  }
};

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svmax_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svmin_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svadd_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svsub_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svmul_s32_x(svptrue_b32(), a, b);
}

inline Vectorized<int32_t> operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr uint32_t max_shift = sizeof(int32_t) * CHAR_BIT - 1;
  svbool_t pg = svptrue_b32();
  svuint32_t bu = svreinterpret_u32_s32(b);
  svbool_t neg = svcmplt_n_s32(pg, b, 0);
  svbool_t big = svcmpgt_n_u32(pg, bu, max_shift);
  svuint32_t clamped = svsel_u32(big, svdup_n_u32(max_shift), bu);
  clamped = svsel_u32(neg, svdup_n_u32(max_shift), clamped);
  return svasr_s32_x(pg, a, clamped);
}

inline Vectorized<int32_t> operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  constexpr uint32_t max_shift = sizeof(int32_t) * CHAR_BIT;
  svbool_t pg = svptrue_b32();
  svuint32_t bu = svreinterpret_u32_s32(b);
  svbool_t neg = svcmplt_n_s32(pg, b, 0);
  svbool_t big = svcmpgt_n_u32(pg, bu, max_shift);
  svuint32_t clamped = svsel_u32(big, svdup_n_u32(0), bu);
  clamped = svsel_u32(neg, svdup_n_u32(0), clamped);
  return svlsl_s32_x(pg, a, clamped);
}

template <>
Vectorized<int32_t> inline operator&(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svand_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline operator|(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svorr_s32_x(svptrue_b32(), a, b);
}

template <>
Vectorized<int32_t> inline operator^(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return sveor_s32_x(svptrue_b32(), a, b);
}

} // inline namespace CPU_CAPABILITY
} // namespace vec
} // namespace tensorplay
