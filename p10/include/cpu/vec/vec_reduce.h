#pragma once

// Row-reduction primitives shared by every CPU kernel tier copy.  The code is
// written against the Vectorized<T> abstraction, so each tier compilation
// picks up its own lane width; the x86 branches below only exist where the
// abstraction has no primitive yet (bit-pattern blends across element types
// and horizontal NaN probes).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "cpu/vec/vec.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

// Bitmask with one bit per lane marking the NaN positions of a value vector;
// the offset of the first NaN lane is ctz(mask).  Zero means the block is
// clean.
#if defined(CPU_CAPABILITY_AVX512)
inline uint32_t nan_lane_mask(const Vectorized<float>& v) {
  // An unordered compare of a vector against itself is true exactly in the
  // lanes holding a NaN, and the width-16 compare returns the lane mask
  // directly.
  return static_cast<uint32_t>(_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q));
}
inline uint32_t nan_lane_mask(const Vectorized<double>& v) {
  return static_cast<uint32_t>(_mm512_cmp_pd_mask(v, v, _CMP_UNORD_Q));
}
#elif defined(CPU_CAPABILITY_AVX2)
inline uint32_t nan_lane_mask(const Vectorized<float>& v) {
  return static_cast<uint32_t>(_mm256_movemask_ps(
      _mm256_cmp_ps(v, v, _CMP_UNORD_Q)));
}
inline uint32_t nan_lane_mask(const Vectorized<double>& v) {
  return static_cast<uint32_t>(_mm256_movemask_pd(
      _mm256_cmp_pd(v, v, _CMP_UNORD_Q)));
}
#endif

// Generic fallback: store the block and probe each lane.
template <typename T>
inline uint32_t nan_lane_mask(const Vectorized<T>& v) {
  constexpr int n = Vectorized<T>::size();
  __at_align__ T buffer[n];
  v.store(buffer);
  uint32_t mask = 0;
  for (int k = 0; k < n; ++k) {
    if (std::isnan(buffer[k])) mask |= (uint32_t)1 << k;
  }
  return mask;
}

// Lane-wise select driven by a float/double mask: lanes whose mask lane has
// the sign bit set take \`alt\`, the rest keep \`acc\`.  Ordered-compare results
// (all-bits-one / all-bits-zero) make this an ordinary vector select.  The
// generic fallback reinterprets the mask bit pattern through a buffer; the
// low-bit test of the element-wise blend agrees with the sign-bit test on
// compare results.
#if defined(CPU_CAPABILITY_AVX512)
inline Vectorized<int32_t> blend_lanes(
    const Vectorized<int32_t>& acc,
    const Vectorized<int32_t>& alt,
    const Vectorized<float>& mask) {
  return Vectorized<int32_t>(_mm512_mask_blend_epi32(
      _mm512_movepi32_mask(_mm512_castps_si512(mask)),
      acc, alt));
}
inline Vectorized<int64_t> blend_lanes(
    const Vectorized<int64_t>& acc,
    const Vectorized<int64_t>& alt,
    const Vectorized<double>& mask) {
  return Vectorized<int64_t>(_mm512_mask_blend_epi64(
      _mm512_movepi64_mask(_mm512_castpd_si512(mask)),
      acc, alt));
}
#elif defined(CPU_CAPABILITY_AVX2)
inline Vectorized<int32_t> blend_lanes(
    const Vectorized<int32_t>& acc,
    const Vectorized<int32_t>& alt,
    const Vectorized<float>& mask) {
  return Vectorized<int32_t>(_mm256_blendv_epi8(
      acc, alt, _mm256_castps_si256(mask)));
}
inline Vectorized<int64_t> blend_lanes(
    const Vectorized<int64_t>& acc,
    const Vectorized<int64_t>& alt,
    const Vectorized<double>& mask) {
  return Vectorized<int64_t>(_mm256_blendv_epi8(
      acc, alt, _mm256_castpd_si256(mask)));
}
#else
inline Vectorized<int32_t> blend_lanes(
    const Vectorized<int32_t>& acc,
    const Vectorized<int32_t>& alt,
    const Vectorized<float>& mask) {
  constexpr int n = Vectorized<float>::size();
  __at_align__ float mask_bits[n];
  __at_align__ int32_t index_bits[n];
  mask.store(mask_bits);
  std::memcpy(index_bits, mask_bits, sizeof(index_bits));
  return Vectorized<int32_t>::blendv(
      acc, alt, Vectorized<int32_t>::loadu(index_bits));
}
inline Vectorized<int64_t> blend_lanes(
    const Vectorized<int64_t>& acc,
    const Vectorized<int64_t>& alt,
    const Vectorized<double>& mask) {
  constexpr int n = Vectorized<double>::size();
  __at_align__ double mask_bits[n];
  __at_align__ int64_t index_bits[n];
  mask.store(mask_bits);
  std::memcpy(index_bits, mask_bits, sizeof(index_bits));
  return Vectorized<int64_t>::blendv(
      acc, alt, Vectorized<int64_t>::loadu(index_bits));
}
#endif

// Streams the sum of squares over x[begin, end) through four independent
// accumulators and one horizontal fold; the caller decides where the square
// root belongs.
template <typename T>
inline T sum_of_squares(const T* x, int64_t begin, int64_t end) {
  using Vec = Vectorized<T>;
  constexpr int64_t W = static_cast<int64_t>(Vec::size());
  const int64_t n = end - begin;
  if (n < 4 * W) {
    T s = T(0);
    for (int64_t i = begin; i < end; ++i) {
      const T v = x[i];
      s += v * v;
    }
    return s;
  }
  Vec a0(T(0)), a1(T(0)), a2(T(0)), a3(T(0));
  int64_t i = begin;
  for (; i + 4 * W <= end; i += 4 * W) {
    const Vec v0 = Vec::loadu(x + i);
    const Vec v1 = Vec::loadu(x + i + W);
    const Vec v2 = Vec::loadu(x + i + 2 * W);
    const Vec v3 = Vec::loadu(x + i + 3 * W);
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    a0 = fmadd(v0, v0, a0);
    a1 = fmadd(v1, v1, a1);
    a2 = fmadd(v2, v2, a2);
    a3 = fmadd(v3, v3, a3);
#else
    a0 = a0 + v0 * v0;
    a1 = a1 + v1 * v1;
    a2 = a2 + v2 * v2;
    a3 = a3 + v3 * v3;
#endif
  }
  Vec tail = (a0 + a1) + (a2 + a3);
  for (; i + W <= end; i += W) {
    const Vec v = Vec::loadu(x + i);
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
    tail = fmadd(v, v, tail);
#else
    tail = tail + v * v;
#endif
  }
  T s = tail.reduce_add();
  for (; i < end; ++i) {
    const T v = x[i];
    s += v * v;
  }
  return s;
}

// Folds one contiguous row down to (extremum, first index carrying it).  A
// NaN wins immediately and reports its own position; equal candidates keep
// the earliest position.  Float rows carry 32-bit lane indices and double
// rows 64-bit ones, matching the value vector's lane count at every width.
template <bool IsMax, typename T>
inline std::pair<T, int64_t> row_extremum(const T* input, int64_t n) {
  using Vec = Vectorized<T>;
  using IVec = std::conditional_t<std::is_same_v<T, float>,
                                  Vectorized<int32_t>, Vectorized<int64_t>>;
  using idx_t = std::conditional_t<std::is_same_v<T, float>, int32_t, int64_t>;
  constexpr int64_t W = static_cast<int64_t>(Vec::size());
  const T nan_value = std::numeric_limits<T>::quiet_NaN();

  if (n < W) {
    T best = input[0];
    int64_t best_index = 0;
    if (std::isnan(best)) return {nan_value, 0};
    for (int64_t i = 1; i < n; ++i) {
      const T v = input[i];
      if (std::isnan(v)) return {nan_value, i};
      if (IsMax ? v > best : v < best) {
        best = v;
        best_index = i;
      }
    }
    return {best, best_index};
  }

  const Vec first = Vec::loadu(input);
  const uint32_t first_nan = nan_lane_mask(first);
  if (first_nan != 0) {
    return {nan_value, static_cast<int64_t>(__builtin_ctz(first_nan))};
  }

  const IVec lane_indices = IVec::arange(static_cast<idx_t>(0),
                                         static_cast<idx_t>(1));
  Vec best_values = first;
  IVec best_indices = lane_indices;
  int64_t i = W;
  for (; i + W <= n; i += W) {
    const Vec values = Vec::loadu(input + i);
    const uint32_t nan_mask = nan_lane_mask(values);
    if (nan_mask != 0) {
      return {nan_value, i + static_cast<int64_t>(__builtin_ctz(nan_mask))};
    }
    const Vec better =
        IsMax ? (values > best_values) : (values < best_values);
    best_values = Vec::blendv(best_values, values, better);
    best_indices = blend_lanes(
        best_indices, lane_indices + IVec(static_cast<idx_t>(i)), better);
  }

  __at_align__ T vals[Vec::size()];
  __at_align__ idx_t ids[IVec::size()];
  best_values.store(vals);
  best_indices.store(ids);
  T best = vals[0];
  int64_t best_index = ids[0];
  for (int lane = 1; lane < static_cast<int>(Vec::size()); ++lane) {
    const bool improves = IsMax ? vals[lane] > best : vals[lane] < best;
    if (improves || (vals[lane] == best && ids[lane] < best_index)) {
      best = vals[lane];
      best_index = ids[lane];
    }
  }
  for (; i < n; ++i) {
    const T v = input[i];
    if (std::isnan(v)) return {nan_value, i};
    if (IsMax ? v > best : v < best) {
      best = v;
      best_index = i;
    }
  }
  return {best, best_index};
}

}  // inline namespace CPU_CAPABILITY
}  // namespace tensorplay::vec
