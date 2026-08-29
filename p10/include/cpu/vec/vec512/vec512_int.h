#pragma once

// derived from the vec256 port (same interface, 512-bit width).
// Only int32/int64 specializations are provided (int8/16/uint8 can be added
// when kernels need them).

#include <immintrin.h>
#include "cpu/vec/vec_base.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace tensorplay::vec::inline CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX512

struct Vectorizedi {
 protected:
  __m512i values;

  static inline __m512i invert(const __m512i& v) {
    const auto ones = _mm512_set1_epi64(-1);
    return _mm512_xor_si512(ones, v);
  }

 public:
  Vectorizedi() {
    values = _mm512_setzero_si512();
  }
  Vectorizedi(__m512i v) : values(v) {}
  operator __m512i() const {
    return values;
  }
};

#else

struct Vectorizedi {}; // dummy definition to make Vectorizedi always defined

#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

// Lane-mask -> all-ones/zero vector widening (GCC exposes only the __mmask
// forms for 512-bit integer compares).
inline __m512i widen_epi64(__mmask8 k) {
  return _mm512_maskz_mov_epi64(k, _mm512_set1_epi64(-1));
}
inline __m512i widen_epi32(__mmask16 k) {
  return _mm512_maskz_mov_epi32(k, _mm512_set1_epi32(-1));
}

template <>
class Vectorized<int64_t> : public Vectorizedi {
 public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }
  int64_t reduce_add() const {
    __at_align__ int64_t tmp[kSize];
    store(tmp);
    int64_t sum = 0;
    for (size_type i = 0; i < kSize; i++) {
      sum += tmp[i];
    }
    return sum;
  }
  int64_t reduce_max() const {
    __at_align__ int64_t tmp[kSize];
    store(tmp);
    int64_t max = tmp[0];
    for (size_type i = 1; i < kSize; i++) {
      max = std::max(max, tmp[i]);
    }
    return max;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {
    values = _mm512_setzero_si512();
  }
  Vectorized(int64_t v) {
    values = _mm512_set1_epi64(v);
  }
  Vectorized(
      int64_t val1,
      int64_t val2,
      int64_t val3,
      int64_t val4,
      int64_t val5,
      int64_t val6,
      int64_t val7,
      int64_t val8) {
    values = _mm512_setr_epi64(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(
      Vectorized<int64_t> a,
      Vectorized<int64_t> b) {
    return _mm512_mask_blend_epi64(static_cast<__mmask8>(mask), a, b);
  }
  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    return _mm512_mask_blend_epi64(
        _mm512_movepi64_mask(mask.values), a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step);
  }
  static Vectorized<int64_t> set(
      Vectorized<int64_t> a,
      Vectorized<int64_t> b,
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
  static Vectorized<int64_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int64_t> loadu(const void* ptr, int64_t count) {
    __at_align__ int64_t tmp_values[size()];
    // Fill tail with 1; loop for GCC 11 auto-vec.
    for (const auto i : tensorplay::irange(size())) {
      tmp_values[i] = 1;
    }
    std::memcpy(
        tmp_values, ptr, std::min<int64_t>(count, size()) * sizeof(int64_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int64_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(int64_t));
    }
  }
  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;
  Vectorized<int64_t> abs() const {
    auto zero = _mm512_set1_epi64(0);
    auto is_larger = widen_epi64(_mm512_cmpgt_epi64_mask(zero, values));
    auto inverse = _mm512_xor_si512(values, is_larger);
    return _mm512_sub_epi64(inverse, is_larger);
  }
  Vectorized<int64_t> real() const {
    return *this;
  }
  Vectorized<int64_t> imag() const {
    return _mm512_set1_epi64(0);
  }
  Vectorized<int64_t> conj() const {
    return *this;
  }
  Vectorized<int64_t> neg() const {
    return _mm512_sub_epi64(_mm512_setzero_si512(), values);
  }
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    return widen_epi64(_mm512_cmpeq_epi64_mask(values, other.values));
  }
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return invert(widen_epi64(_mm512_cmpeq_epi64_mask(values, other.values)));
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return widen_epi64(_mm512_cmpgt_epi64_mask(other.values, values));
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return invert(widen_epi64(_mm512_cmpgt_epi64_mask(values, other.values)));
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return widen_epi64(_mm512_cmpgt_epi64_mask(values, other.values));
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return invert(widen_epi64(_mm512_cmpgt_epi64_mask(other.values, values)));
  }

  Vectorized<int64_t> eq(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> ne(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> gt(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> ge(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> lt(const Vectorized<int64_t>& other) const;
  Vectorized<int64_t> le(const Vectorized<int64_t>& other) const;
};

template <>
struct is_vec_specialized_for<int32_t> : std::bool_constant<true> {};

template <>
class Vectorized<int32_t> : public Vectorizedi {
 public:
  using value_type = int32_t;
  using size_type = int;
  static constexpr size_type kSize = 16;
  static constexpr size_type size() {
    return kSize;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {
    values = _mm512_setzero_si512();
  }
  Vectorized(int32_t v) {
    values = _mm512_set1_epi32(v);
  }
  Vectorized(
      int32_t val1,
      int32_t val2,
      int32_t val3,
      int32_t val4,
      int32_t val5,
      int32_t val6,
      int32_t val7,
      int32_t val8,
      int32_t val9,
      int32_t val10,
      int32_t val11,
      int32_t val12,
      int32_t val13,
      int32_t val14,
      int32_t val15,
      int32_t val16) {
    values = _mm512_setr_epi32(
        val1,
        val2,
        val3,
        val4,
        val5,
        val6,
        val7,
        val8,
        val9,
        val10,
        val11,
        val12,
        val13,
        val14,
        val15,
        val16);
  }
  template <int64_t mask>
  static Vectorized<int32_t> blend(
      Vectorized<int32_t> a,
      Vectorized<int32_t> b) {
    return _mm512_mask_blend_epi32(static_cast<__mmask16>(mask), a, b);
  }
  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    return _mm512_mask_blend_epi32(
        _mm512_movepi32_mask(mask.values), a.values, b.values);
  }
  template <typename step_t>
  static Vectorized<int32_t> arange(
      int32_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int32_t>(
        base,
        base + step,
        base + 2 * step,
        base + 3 * step,
        base + 4 * step,
        base + 5 * step,
        base + 6 * step,
        base + 7 * step,
        base + 8 * step,
        base + 9 * step,
        base + 10 * step,
        base + 11 * step,
        base + 12 * step,
        base + 13 * step,
        base + 14 * step,
        base + 15 * step);
  }
  static Vectorized<int32_t> set(
      Vectorized<int32_t> a,
      Vectorized<int32_t> b,
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
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
    }
    return b;
  }
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
  }
  static Vectorized<int32_t> loadu(const void* ptr, int32_t count) {
    __at_align__ int32_t tmp_values[size()];
    // Fill tail with 1; loop for GCC 11 auto-vec.
    for (const auto i : tensorplay::irange(size())) {
      tmp_values[i] = 1;
    }
    std::memcpy(
        tmp_values, ptr, std::min<int32_t>(count, size()) * sizeof(int32_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int32_t tmp_values[size()];
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), values);
      std::memcpy(
          ptr, tmp_values, std::min<int32_t>(count, size()) * sizeof(int32_t));
    }
  }
  const int32_t& operator[](int idx) const = delete;
  int32_t& operator[](int idx) = delete;
  Vectorized<int32_t> abs() const {
    return _mm512_abs_epi32(values);
  }
  Vectorized<int32_t> real() const {
    return *this;
  }
  Vectorized<int32_t> imag() const {
    return _mm512_set1_epi32(0);
  }
  Vectorized<int32_t> conj() const {
    return *this;
  }
  Vectorized<int32_t> neg() const {
    return _mm512_sub_epi32(_mm512_setzero_si512(), values);
  }
  int32_t reduce_add() const {
    auto v = values;
    // 256-bit shuffle
    auto v1 = _mm512_shuffle_i32x4(v, v, 0x4E);
    v = _mm512_add_epi32(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_i32x4(v, v, 0xB1);
    v = _mm512_add_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E));
    v = _mm512_add_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0xB1));
    v = _mm512_add_epi32(v, v1);
    __m128i lo = _mm512_castsi512_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  int32_t reduce_max() const {
    auto v = values;
    // 256-bit shuffle
    auto v1 = _mm512_shuffle_i32x4(v, v, 0x4E);
    v = _mm512_max_epi32(v, v1);
    // 128-bit shuffle
    v1 = _mm512_shuffle_i32x4(v, v, 0xB1);
    v = _mm512_max_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x4E));
    v = _mm512_max_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0xB1));
    v = _mm512_max_epi32(v, v1);
    __m128i lo = _mm512_castsi512_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    return widen_epi32(_mm512_cmpeq_epi32_mask(values, other.values));
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    return invert(widen_epi32(_mm512_cmpeq_epi32_mask(values, other.values)));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    return widen_epi32(_mm512_cmpgt_epi32_mask(other.values, values));
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    return invert(widen_epi32(_mm512_cmpgt_epi32_mask(values, other.values)));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    return widen_epi32(_mm512_cmpgt_epi32_mask(values, other.values));
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    return invert(widen_epi32(_mm512_cmpgt_epi32_mask(other.values, values)));
  }

  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const;
};

#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512
template <typename T, typename Op>
inline Vectorized<T> int_elementwise_binary(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  T values_a[Vectorized<T>::size()];
  T values_b[Vectorized<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    values_a[i] = op(values_a[i], values_b[i]);
  }
  return Vectorized<T>::loadu(values_a);
}

template <>
Vectorized<int64_t> inline operator+(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_add_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_add_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_sub_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_sub_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // No 512-bit 64-bit multiply; emulate with the compiler-vectorized loop.
  return int_elementwise_binary(a, b, std::multiplies<int64_t>());
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_mullo_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return int_elementwise_binary(a, b, std::divides<int64_t>());
}

template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return int_elementwise_binary(a, b, std::divides<int32_t>());
}

template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_and_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_or_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm512_xor_si512(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  __mmask8 cmp = _mm512_cmpgt_epi64_mask(a, b);
  return _mm512_mask_blend_epi64(cmp, b, a);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_min_epi32(a, b);
}

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  __mmask8 cmp = _mm512_cmpgt_epi64_mask(a, b);
  return _mm512_mask_blend_epi64(cmp, b, a);
}

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_max_epi32(a, b);
}

template <>
Vectorized<int64_t> inline clamp(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min,
    const Vectorized<int64_t>& max) {
  return minimum(maximum(a, min), max);
}

template <>
Vectorized<int32_t> inline clamp(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min,
    const Vectorized<int32_t>& max) {
  return _mm512_min_epi32(_mm512_max_epi32(a, min), max);
}

template <>
Vectorized<int64_t> inline clamp_max(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& max) {
  return minimum(a, max);
}

template <>
Vectorized<int32_t> inline clamp_max(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& max) {
  return _mm512_min_epi32(a, max);
}

template <>
Vectorized<int64_t> inline clamp_min(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min) {
  return maximum(a, min);
}

template <>
Vectorized<int32_t> inline clamp_min(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min) {
  return _mm512_max_epi32(a, min);
}
#endif // CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512
inline Vectorized<int64_t> Vectorized<int64_t>::eq(
    const Vectorized<int64_t>& other) const {
  return (*this == other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ne(
    const Vectorized<int64_t>& other) const {
  return (*this != other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::gt(
    const Vectorized<int64_t>& other) const {
  return (*this > other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ge(
    const Vectorized<int64_t>& other) const {
  return (*this >= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::lt(
    const Vectorized<int64_t>& other) const {
  return (*this < other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::le(
    const Vectorized<int64_t>& other) const {
  return (*this <= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::eq(
    const Vectorized<int32_t>& other) const {
  return (*this == other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ne(
    const Vectorized<int32_t>& other) const {
  return (*this != other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::gt(
    const Vectorized<int32_t>& other) const {
  return (*this > other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ge(
    const Vectorized<int32_t>& other) const {
  return (*this >= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::lt(
    const Vectorized<int32_t>& other) const {
  return (*this < other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::le(
    const Vectorized<int32_t>& other) const {
  return (*this <= other) & Vectorized<int32_t>(1);
}

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm512_sllv_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_sllv_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // No vector instruction for right arithmetic shifting int64_t, so emulating
  // it instead.

  // Clamp the shift values such that shift values < 0 and > 64 are changed to
  // 64 which results in -1 for negative input and 0 for non-negative input.
  __m512i zero = _mm512_set1_epi64(0);
  __m512i max_shift = _mm512_set1_epi64(64);
  __mmask8 out_of_range =
      static_cast<__mmask8>(_mm512_cmpgt_epi64_mask(zero, b) |
                            _mm512_cmpgt_epi64_mask(b, max_shift));
  __m512i shift = _mm512_mask_blend_epi64(out_of_range, b, max_shift);
  // Shift the number logically to the right, thus filling the most
  // significant bits with 0s.  Then, replace these bits with the sign
  // bit.
  __m512i sign_bits = widen_epi64(_mm512_cmpgt_epi64_mask(zero, a));
  __m512i sign_shift = _mm512_sub_epi64(max_shift, shift);
  __m512i sign_ext = _mm512_sllv_epi64(sign_bits, sign_shift);
  __m512i c = _mm512_srlv_epi64(a, shift);
  c = _mm512_or_si512(c, sign_ext);

  return c;
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm512_srav_epi32(a, b);
}

#endif // CPU_CAPABILITY_AVX512

} // namespace tensorplay::vec::inline CPU_CAPABILITY
