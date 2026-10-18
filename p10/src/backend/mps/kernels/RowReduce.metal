// Row reductions over the innermost axis of a contiguous operand.
//
// The operand is a block of M rows of N consecutive elements and the result is
// one value per row, so one SIMD group takes one row: its lanes walk the row in
// steps of the group width, which is one wide access per instruction, and the
// row's answer is folded between those lanes without leaving the group. Rows
// share nothing, so a call of this shape needs neither threadgroup memory nor a
// second dispatch.

#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "common.h"

using namespace metal;
using namespace p10::metal;

// Accumulators per lane while a lane walks its share of the row. One
// accumulator would make the lane wait on every load before it could issue the
// next; several keep a whole group of loads in flight.
P10_METAL_CONSTEXPR unsigned kSumChains = 8;

// The row count and the length of one row, which the host passes as one pair.
using RowSizes = metal::array<uint32_t, 2>;

template <typename T>
kernel void row_sum_inner(device T* out [[buffer(0)]],
                          constant T* src [[buffer(1)]],
                          constant RowSizes& sizes [[buffer(2)]],
                          constant float& divisor [[buffer(3)]],
                          uint tptg [[threads_per_threadgroup]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint group [[simdgroup_index_in_threadgroup]]) {
  const uint row = tgid * (tptg / simdgroup_size) + group;
  if (row >= sizes[0]) {
    return;
  }
  const uint n = sizes[1];
  constant T* values = src + (uint64_t)row * (uint64_t)n;

  metal::array<T, kSumChains> acc;
#pragma unroll
  for (uint j = 0; j < kSumChains; ++j) {
    acc[j] = T(0);
  }
  // A lane covers kSumChains consecutive elements at a time, so a full block of
  // the row spans the whole group once per chain.
  const uint stride = simdgroup_size * kSumChains;
  const uint whole = (n / stride) * stride;
  for (uint base = lane * kSumChains; base < whole; base += stride) {
#pragma unroll
    for (uint j = 0; j < kSumChains; ++j) {
      acc[j] += values[base + j];
    }
  }
  // The rest of the row, one element per lane per step.
  for (uint i = whole + lane; i < n; i += simdgroup_size) {
    acc[0] += values[i];
  }
  T total = acc[0];
#pragma unroll
  for (uint j = 1; j < kSumChains; ++j) {
    total += acc[j];
  }
  total = simd_sum(total);
  if (lane == 0) {
    // A divisor of zero or less asks for the sum as it stands; the average
    // passes the length of the row.
    if (divisor > 0) {
      total /= static_cast<T>(divisor);
    }
    out[row] = total;
  }
}

// Whether a candidate extremum displaces the one held. A NaN candidate displaces
// anything and a NaN held displaces anything else, so a row holding a NaN
// answers with the first of them rather than with the largest finite value.
template <typename T>
struct arg_max {
  // Minus infinity, not the least finite value: an operand that is exactly the
  // least finite value has to be reported as the extremum of its row.
  static inline constexpr T identity() { return T(-INFINITY); }
  static inline bool replace(T candidate, T held) {
    if (::metal::isnan(held)) {
      return false;
    }
    return ::metal::isnan(candidate) || candidate > held;
  }
  static inline T reduce(T value) { return simd_max(value); }
};

template <typename T>
struct arg_min {
  static inline constexpr T identity() { return T(INFINITY); }
  static inline bool replace(T candidate, T held) {
    if (::metal::isnan(held)) {
      return false;
    }
    return ::metal::isnan(candidate) || candidate < held;
  }
  static inline T reduce(T value) { return simd_min(value); }
};

// The position of the extremum among the lanes. Reducing the values alone would
// answer with whichever lane happened to hold the extremum last, and a lane can
// have scanned several positions; so the winning value is found first, then the
// lowest position among the lanes holding it -- which is the position the scan
// reached first in the row. A lane holding a NaN counts as holding the winner,
// whichever value the reduction of a row containing a NaN came back with.
template <typename Op, typename T>
inline uint32_t arg_position(T value, uint32_t index) {
  const T winner = Op::reduce(value);
  const bool holds_winner = ::metal::isnan(value) || value == winner;
  return simd_min(holds_winner ? index : 0xFFFFFFFFu);
}

template <typename Op>
kernel void row_arg_inner(device long* out [[buffer(0)]],
                          constant float* src [[buffer(1)]],
                          constant RowSizes& sizes [[buffer(2)]],
                          uint tptg [[threads_per_threadgroup]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint group [[simdgroup_index_in_threadgroup]]) {
  const uint row = tgid * (tptg / simdgroup_size) + group;
  if (row >= sizes[0]) {
    return;
  }
  const uint n = sizes[1];
  constant float* values = src + (uint64_t)row * (uint64_t)n;

  float best = Op::identity();
  uint32_t best_index = 0;
  // Only a strictly better element displaces the one held, so a tie keeps the
  // position already stored.
  for (uint i = lane; i < n; i += simdgroup_size) {
    const float candidate = values[i];
    if (Op::replace(candidate, best)) {
      best = candidate;
      best_index = i;
    }
  }
  const uint32_t position = arg_position<Op>(best, best_index);
  if (lane == 0) {
    out[row] = static_cast<long>(position);
  }
}

// Float is the only storage type these serve: an operand of another type
// reduces in a type of its own, which the graph path builds around the
// reduction. A call with such a type reaches the graph by leaving the host
// unable to resolve the kernel name.
#define REGISTER_ROW_SUM(DTYPE)                                         \
  template [[host_name("rowsum_" #DTYPE)]] kernel void                  \
  row_sum_inner<DTYPE>(device DTYPE* out, constant DTYPE* src,          \
                       constant RowSizes& sizes, constant float& divisor,  \
                       uint tptg, uint tgid, uint lane, uint group);

#define REGISTER_ARG_REDUCTION(HOST_NAME, OP)                           \
  template [[host_name(#HOST_NAME "_float")]] kernel void               \
  row_arg_inner<OP<float>>(device long* out, constant float* src,       \
                           constant RowSizes& sizes,                       \
                           uint tptg, uint tgid, uint lane, uint group);

REGISTER_ROW_SUM(float)
REGISTER_ARG_REDUCTION(argmax, arg_max)
REGISTER_ARG_REDUCTION(argmin, arg_min)
