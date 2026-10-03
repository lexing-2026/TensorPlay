#pragma once

// Whole-array helpers over the vector layer, for kernels written at run time:
// a vector function applied across an array, and the packing and transposes a
// blocked matrix product reads its operands through.
//
// Each helper is the plain loop it replaces, done one register at a time with a
// partial register for the tail, so an array of any length is covered and no
// element past its end is read or written.

#include "cpu/vec/vec.h"
#include "cpu/vec/vec_n.h"
#include "Exception.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace tensorplay {

// Calls f(0), f(1), ..., f(n - 1), each index passed as a compile-time
// constant, so a body written once is laid down n times with its index folded
// in.  A register block is addressed this way: which register an index names
// has to be known when the code is compiled, not when it runs.
template <int n>
struct ForcedUnroll {
  template <typename Func, typename... Args>
  inline __attribute__((always_inline)) void operator()(
      const Func& f, Args... args) const {
    ForcedUnroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct ForcedUnroll<1> {
  template <typename Func, typename... Args>
  inline __attribute__((always_inline)) void operator()(
      const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

} // namespace tensorplay

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

// Writes a register to `dst` as dst's own type.  A float register written to a
// sixteen-bit float array is narrowed on the way, so a kernel computing in
// float can hand its result to whichever array it was given; `count` is how
// many leading lanes are written.
template <typename T>
inline void store(
    T* dst, const Vectorized<T>& src, int64_t count = Vectorized<T>::size()) {
  src.store(dst, static_cast<int>(count));
}

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline void store(
    T* dst,
    const Vectorized<float>& src,
    int64_t count = Vectorized<float>::size()) {
  convert<T>(src).store(dst, static_cast<int>(count));
}

// output[i] = vec_fun(input[i]) over `size` elements.  A sixteen-bit float
// array is computed in float -- two float registers per register of the
// narrow type -- because that is the precision its arithmetic is defined in.
template <typename scalar_t, typename Op>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  if constexpr (is_reduced_floating_point_v<scalar_t>) {
    using bVec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      auto [f0, f1] = convert_to_float<scalar_t>(bVec::loadu(input_data + d));
      convert_from_float<scalar_t>(vec_fun(f0), vec_fun(f1))
          .store(output_data + d);
    }
    if (size - d > 0) {
      auto [f0, f1] =
          convert_to_float<scalar_t>(bVec::loadu(input_data + d, size - d));
      convert_from_float<scalar_t>(vec_fun(f0), vec_fun(f1))
          .store(output_data + d, size - d);
    }
  } else {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      vec_fun(Vec::loadu(input_data + d)).store(output_data + d);
    }
    if (size - d > 0) {
      vec_fun(Vec::loadu(input_data + d, size - d))
          .store(output_data + d, size - d);
    }
  }
}

// The same, reading float and writing a sixteen-bit float: the last step of a
// computation carried out in float and handed back in the narrow type.
template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const float* input_data,
    int64_t size) {
  using bVec = Vectorized<scalar_t>;
  using fVec = Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    fVec f0 = fVec::loadu(input_data + d);
    fVec f1 = fVec::loadu(input_data + d + fVec::size());
    convert_from_float<scalar_t>(vec_fun(f0), vec_fun(f1))
        .store(output_data + d);
  }
  if (size - d > 0) {
    fVec f0(0.0f), f1(0.0f);
    if (size - d > fVec::size()) {
      f0 = fVec::loadu(input_data + d);
      f1 = fVec::loadu(input_data + d + fVec::size(), size - d - fVec::size());
    } else {
      f0 = fVec::loadu(input_data + d, size - d);
    }
    convert_from_float<scalar_t>(vec_fun(f0), vec_fun(f1))
        .store(output_data + d, size - d);
  }
}

// output[i] = vec_fun(input[i], input2[i]) over `size` elements.
template <typename scalar_t, typename Op>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  if constexpr (is_reduced_floating_point_v<scalar_t>) {
    using bVec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      auto [a0, a1] = convert_to_float<scalar_t>(bVec::loadu(input_data + d));
      auto [b0, b1] = convert_to_float<scalar_t>(bVec::loadu(input_data2 + d));
      convert_from_float<scalar_t>(vec_fun(a0, b0), vec_fun(a1, b1))
          .store(output_data + d);
    }
    if (size - d > 0) {
      auto [a0, a1] =
          convert_to_float<scalar_t>(bVec::loadu(input_data + d, size - d));
      auto [b0, b1] =
          convert_to_float<scalar_t>(bVec::loadu(input_data2 + d, size - d));
      convert_from_float<scalar_t>(vec_fun(a0, b0), vec_fun(a1, b1))
          .store(output_data + d, size - d);
    }
  } else {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      vec_fun(Vec::loadu(input_data + d), Vec::loadu(input_data2 + d))
          .store(output_data + d);
    }
    if (size - d > 0) {
      vec_fun(
          Vec::loadu(input_data + d, size - d),
          Vec::loadu(input_data2 + d, size - d))
          .store(output_data + d, size - d);
    }
  }
}

// Reorders a [K, N] block of two-byte elements as [K / 2, N, 2]: each pair of
// rows is interleaved column by column, which is the order a matrix unit that
// multiplies element pairs reads its second operand in.  An odd K is padded
// with a zero row, so the last pair is still a pair.
template <typename scalar_t, typename = std::enable_if_t<sizeof(scalar_t) == 2>>
inline void pack_vnni2(
    const scalar_t* src,
    scalar_t* dst,
    int64_t ld_src,
    int64_t K,
    int64_t N) {
  for (int64_t k = 0; k < K; k += 2) {
    const scalar_t* row0 = src + k * ld_src;
    const scalar_t* row1 = k + 1 < K ? src + (k + 1) * ld_src : nullptr;
    scalar_t* out = dst + k * N;
    for (int64_t n = 0; n < N; ++n) {
      out[2 * n] = row0[n];
      out[2 * n + 1] = row1 != nullptr ? row1[n] : scalar_t(0);
    }
  }
}

// Transposes a square block held in registers: register i holds row i before
// and column i after.  The block is as wide as one register, which is the
// shape a product reading its second operand transposed walks it in.
#if defined(CPU_CAPABILITY_AVX512)
inline void transpose_block(
    VectorizedN<float, 16>& input, int M = 16, int N = 16) {
  TP_CHECK(M <= 16 && N <= 16, "transpose_block expects M, N <= 16.");
  // Interleave 32-bit lanes of row pairs, then 64-bit lanes of the pairs, then
  // 128-bit quarters, then the halves: four rounds of shuffles move every
  // element to its transposed place.
  __m512 temp[16];
  int i;
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm512_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm512_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 16; ++i) {
    temp[i] = _mm512_setzero_ps();
  }
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm512_castpd_ps(_mm512_unpacklo_pd(
        _mm512_castps_pd(temp[4 * i]), _mm512_castps_pd(temp[4 * i + 2])));
    input[4 * i + 1] = _mm512_castpd_ps(_mm512_unpackhi_pd(
        _mm512_castps_pd(temp[4 * i]), _mm512_castps_pd(temp[4 * i + 2])));
    input[4 * i + 2] = _mm512_castpd_ps(_mm512_unpacklo_pd(
        _mm512_castps_pd(temp[4 * i + 1]), _mm512_castps_pd(temp[4 * i + 3])));
    input[4 * i + 3] = _mm512_castpd_ps(_mm512_unpackhi_pd(
        _mm512_castps_pd(temp[4 * i + 1]), _mm512_castps_pd(temp[4 * i + 3])));
  }
  for (i = 0; i < (M + 7) / 8; ++i) {
    temp[8 * i] = _mm512_shuffle_f32x4(input[8 * i], input[8 * i + 4], 0x88);
    temp[8 * i + 1] =
        _mm512_shuffle_f32x4(input[8 * i + 1], input[8 * i + 5], 0x88);
    temp[8 * i + 2] =
        _mm512_shuffle_f32x4(input[8 * i + 2], input[8 * i + 6], 0x88);
    temp[8 * i + 3] =
        _mm512_shuffle_f32x4(input[8 * i + 3], input[8 * i + 7], 0x88);
    temp[8 * i + 4] =
        _mm512_shuffle_f32x4(input[8 * i], input[8 * i + 4], 0xdd);
    temp[8 * i + 5] =
        _mm512_shuffle_f32x4(input[8 * i + 1], input[8 * i + 5], 0xdd);
    temp[8 * i + 6] =
        _mm512_shuffle_f32x4(input[8 * i + 2], input[8 * i + 6], 0xdd);
    temp[8 * i + 7] =
        _mm512_shuffle_f32x4(input[8 * i + 3], input[8 * i + 7], 0xdd);
  }
  for (i = 0; i < N; ++i) {
    if (i < 8) {
      input[i] = _mm512_shuffle_f32x4(temp[i], temp[8 + i], 0x88);
    } else {
      input[i] = _mm512_shuffle_f32x4(temp[i - 8], temp[i], 0xdd);
    }
  }
}
#elif defined(CPU_CAPABILITY_AVX2)
inline void transpose_block(VectorizedN<float, 8>& input) {
  // Interleave 32-bit lanes of row pairs, then 64-bit lanes of the pairs, then
  // swap 128-bit halves between rows four apart.
  __m256 temp0[8];
  for (int i = 0; i < 4; ++i) {
    temp0[2 * i] = _mm256_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp0[2 * i + 1] = _mm256_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  __m256 temp1[8];
  for (int h = 0; h < 2; ++h) {
    const int b = 4 * h;
    temp1[b] = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(temp0[b]), _mm256_castps_pd(temp0[b + 2])));
    temp1[b + 1] = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(temp0[b]), _mm256_castps_pd(temp0[b + 2])));
    temp1[b + 2] = _mm256_castpd_ps(_mm256_unpacklo_pd(
        _mm256_castps_pd(temp0[b + 1]), _mm256_castps_pd(temp0[b + 3])));
    temp1[b + 3] = _mm256_castpd_ps(_mm256_unpackhi_pd(
        _mm256_castps_pd(temp0[b + 1]), _mm256_castps_pd(temp0[b + 3])));
  }
  for (int i = 0; i < 4; ++i) {
    input[i] = _mm256_permute2f128_ps(temp1[i], temp1[i + 4], 0x20);
    input[i + 4] = _mm256_permute2f128_ps(temp1[i], temp1[i + 4], 0x31);
  }
}
#else
// Without shuffles to do it in registers the block is written out, read back
// transposed and loaded again.
template <typename T, int N>
inline void transpose_block(VectorizedN<T, N>& input) {
  constexpr int L = Vectorized<T>::size();
  static_assert(L == N, "transpose_block transposes a square block");
  T rows[N * L];
  T cols[N * L];
  for (int i = 0; i < N; ++i) input[i].store(rows + i * L);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < L; ++j) cols[j * L + i] = rows[i * L + j];
  }
  for (int i = 0; i < N; ++i) input[i] = Vectorized<T>::loadu(cols + i * L);
}
#endif

} // namespace CPU_CAPABILITY

// A sixteen-bit float register widened to the two float registers holding its
// lanes, and two float registers narrowed back into one -- the conversions a
// whole-array helper computing a narrow type in float goes through.
template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<
    tensorplay::BFloat16>(const Vectorized<tensorplay::BFloat16>& a) {
  return convert_bfloat16_float(a);
}

template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<
    tensorplay::Half>(const Vectorized<tensorplay::Half>& a) {
  return convert_half_float(a);
}

template <>
inline Vectorized<tensorplay::BFloat16> convert_from_float<tensorplay::BFloat16>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return convert_float_bfloat16(a, b);
}

template <>
inline Vectorized<tensorplay::Half> convert_from_float<tensorplay::Half>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return convert_float_half(a, b);
}

} // namespace tensorplay::vec

namespace tensorplay::utils {

// dst[j, i] = src[i, j] for an M x N block, each side with its own leading
// dimension.
template <typename T>
inline void transpose(
    int64_t M, int64_t N, const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

} // namespace tensorplay::utils
