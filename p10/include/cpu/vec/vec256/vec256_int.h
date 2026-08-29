#pragma once

// only int32/int64 specializations are currently provided (int8/16/uint8 can
// be added when reduction kernels need them).

#include <immintrin.h>
#include "cpu/vec/vec_base.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace tensorplay::vec::inline CPU_CAPABILITY {

#ifdef CPU_CAPABILITY_AVX2

struct Vectorizedi {
 protected:
  __m256i values;

  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }

 public:
  Vectorizedi() {
    values = _mm256_setzero_si256();
  }
  Vectorizedi(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
};

#else

struct Vectorizedi {}; // dummy definition to make Vectorizedi always defined

#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2

template <>
struct is_vec_specialized_for<int64_t> : std::bool_constant<true> {};

template <>
class Vectorized<int64_t> : public Vectorizedi {
 public:
  using value_type = int64_t;
  using size_type = int;
  static constexpr size_type kSize = 4;
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
    values = _mm256_setzero_si256();
  }
  Vectorized(int64_t v) {
    values = _mm256_set1_epi64x(v);
  }
  Vectorized(int64_t val1, int64_t val2, int64_t val3, int64_t val4) {
    values = _mm256_setr_epi64x(val1, val2, val3, val4);
  }
  template <int64_t mask>
  static Vectorized<int64_t> blend(
      Vectorized<int64_t> a,
      Vectorized<int64_t> b) {
    __at_align__ int64_t tmp_values[size()];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi64(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi64(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi64(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi64(b.values, 3);
    return loadu(tmp_values);
  }
  static Vectorized<int64_t> blendv(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b,
      const Vectorized<int64_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  template <typename step_t>
  static Vectorized<int64_t> arange(
      int64_t base = 0,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<int64_t>(
        base, base + step, base + 2 * step, base + 3 * step);
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
    }
    return b;
  }
  static Vectorized<int64_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
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
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int64_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(
          ptr, tmp_values, std::min<int64_t>(count, size()) * sizeof(int64_t));
    }
  }
  const int64_t& operator[](int idx) const = delete;
  int64_t& operator[](int idx) = delete;
  Vectorized<int64_t> abs() const {
    auto zero = _mm256_set1_epi64x(0);
    auto is_larger = _mm256_cmpgt_epi64(zero, values);
    auto inverse = _mm256_xor_si256(values, is_larger);
    return _mm256_sub_epi64(inverse, is_larger);
  }
  Vectorized<int64_t> real() const {
    return *this;
  }
  Vectorized<int64_t> imag() const {
    return _mm256_set1_epi64x(0);
  }
  Vectorized<int64_t> conj() const {
    return *this;
  }
  Vectorized<int64_t> neg() const {
    return _mm256_sub_epi64(_mm256_setzero_si256(), values);
  }
  Vectorized<int64_t> operator==(const Vectorized<int64_t>& other) const {
    return _mm256_cmpeq_epi64(values, other.values);
  }
  Vectorized<int64_t> operator!=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpeq_epi64(values, other.values));
  }
  Vectorized<int64_t> operator<(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(other.values, values);
  }
  Vectorized<int64_t> operator<=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(values, other.values));
  }
  Vectorized<int64_t> operator>(const Vectorized<int64_t>& other) const {
    return _mm256_cmpgt_epi64(values, other.values);
  }
  Vectorized<int64_t> operator>=(const Vectorized<int64_t>& other) const {
    return invert(_mm256_cmpgt_epi64(other.values, values));
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
  static constexpr size_type kSize = 8;
  static constexpr size_type size() {
    return kSize;
  }
  using Vectorizedi::Vectorizedi;
  Vectorized() {
    values = _mm256_setzero_si256();
  }
  Vectorized(int32_t v) {
    values = _mm256_set1_epi32(v);
  }
  Vectorized(int32_t val1, int32_t val2, int32_t val3, int32_t val4,
             int32_t val5, int32_t val6, int32_t val7, int32_t val8) {
    values = _mm256_setr_epi32(
        val1, val2, val3, val4, val5, val6, val7, val8);
  }
  template <int64_t mask>
  static Vectorized<int32_t> blend(
      Vectorized<int32_t> a,
      Vectorized<int32_t> b) {
    return _mm256_blend_epi32(a.values, b.values, mask);
  }
  static Vectorized<int32_t> blendv(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b,
      const Vectorized<int32_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
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
        base + 7 * step);
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
    }
    return b;
  }
  static Vectorized<int32_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
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
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align__ int32_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(
          ptr, tmp_values, std::min<int32_t>(count, size()) * sizeof(int32_t));
    }
  }
  const int32_t& operator[](int idx) const = delete;
  int32_t& operator[](int idx) = delete;
  Vectorized<int32_t> abs() const {
    return _mm256_abs_epi32(values);
  }
  Vectorized<int32_t> real() const {
    return *this;
  }
  Vectorized<int32_t> imag() const {
    return _mm256_set1_epi32(0);
  }
  Vectorized<int32_t> conj() const {
    return *this;
  }
  Vectorized<int32_t> neg() const {
    return _mm256_sub_epi32(_mm256_setzero_si256(), values);
  }
  int32_t reduce_add() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_si256(v, v, 0x1);
    v = _mm256_add_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0x4E);
    v = _mm256_add_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0xB1);
    v = _mm256_add_epi32(v, v1);
    __m128i lo = _mm256_castsi256_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  int32_t reduce_max() const {
    auto v = values;
    // 128-bit shuffle
    auto v1 = _mm256_permute2f128_si256(v, v, 0x1);
    v = _mm256_max_epi32(v, v1);
    // 64-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0x4E);
    v = _mm256_max_epi32(v, v1);
    // 32-bit shuffle
    v1 = _mm256_shuffle_epi32(v, 0xB1);
    v = _mm256_max_epi32(v, v1);
    __m128i lo = _mm256_castsi256_si128(v);
    return _mm_cvtsi128_si32(lo);
  }
  Vectorized<int32_t> operator==(const Vectorized<int32_t>& other) const {
    return _mm256_cmpeq_epi32(values, other.values);
  }
  Vectorized<int32_t> operator!=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpeq_epi32(values, other.values));
  }
  Vectorized<int32_t> operator<(const Vectorized<int32_t>& other) const {
    return _mm256_cmpgt_epi32(other.values, values);
  }
  Vectorized<int32_t> operator<=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(values, other.values));
  }
  Vectorized<int32_t> operator>(const Vectorized<int32_t>& other) const {
    return _mm256_cmpgt_epi32(values, other.values);
  }
  Vectorized<int32_t> operator>=(const Vectorized<int32_t>& other) const {
    return invert(_mm256_cmpgt_epi32(other.values, values));
  }

  Vectorized<int32_t> eq(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ne(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> gt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> ge(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> lt(const Vectorized<int32_t>& other) const;
  Vectorized<int32_t> le(const Vectorized<int32_t>& other) const;
};

#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2
template <typename T, typename Op>
inline Vectorized<T> int_elementwise_binary_256(
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
  return _mm256_add_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator+(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_add_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator-(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return _mm256_sub_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator-(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_sub_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator*(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // No AVX2 64-bit multiply; emulate with the compiler-vectorized loop.
  return int_elementwise_binary_256(a, b, std::multiplies<int64_t>());
}

template <>
Vectorized<int32_t> inline operator*(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_mullo_epi32(a, b);
}

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int64_t>());
}

template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return int_elementwise_binary_256(a, b, std::divides<int32_t>());
}

template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_and_si256(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_or_si256(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return _mm256_xor_si256(a, b);
}
template <
    class T,
    typename std::enable_if_t<
        std::is_base_of<Vectorizedi, Vectorized<T>>::value,
        int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
}

template <>
Vectorized<int64_t> inline minimum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(a, b, cmp);
}

template <>
Vectorized<int32_t> inline minimum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_min_epi32(a, b);
}

template <>
Vectorized<int64_t> inline maximum(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  __m256i cmp = _mm256_cmpgt_epi64(a, b);
  return _mm256_blendv_epi8(b, a, cmp);
}

template <>
Vectorized<int32_t> inline maximum(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_max_epi32(a, b);
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
  return _mm256_min_epi32(_mm256_max_epi32(a, min), max);
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
  return _mm256_min_epi32(a, max);
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
  return _mm256_max_epi32(a, min);
}
#endif // CPU_CAPABILITY_AVX2

#ifdef CPU_CAPABILITY_AVX2
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
  return _mm256_sllv_epi64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_sllv_epi32(a, b);
}






template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  // No vector instruction for right arithmetic shifting int64_t, so emulating
  // it instead.

  // Clamp the shift values such that shift values < 0 and > 64 are changed to
  // 64 which results in -1 for negative input and 0 for non-negative input.
  __m256i zero = _mm256_set1_epi64x(0);
  __m256i max_shift = _mm256_set1_epi64x(64);
  __m256i mask = _mm256_or_si256(
      _mm256_cmpgt_epi64(zero, b), _mm256_cmpgt_epi64(b, max_shift));
  __m256i shift = _mm256_blendv_epi8(b, max_shift, mask);
  // Shift the number logically to the right, thus filling the most
  // significant bits with 0s.  Then, replace these bits with the sign
  // bit.
  __m256i sign_bits = _mm256_cmpgt_epi64(zero, a);
  __m256i sign_shift = _mm256_sub_epi64(max_shift, shift);
  __m256i sign_ext = _mm256_sllv_epi64(sign_bits, sign_shift);
  __m256i c = _mm256_srlv_epi64(a, shift);
  c = _mm256_or_si256(c, sign_ext);

  return c;
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return _mm256_srav_epi32(a, b);
}






#endif // CPU_CAPABILITY_AVX2

} // namespace tensorplay::vec::inline CPU_CAPABILITY
