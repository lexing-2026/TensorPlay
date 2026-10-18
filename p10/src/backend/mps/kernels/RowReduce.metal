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
};

// The extremum of a row and where the row holds it. A lane can have walked
// several positions, so a pair travels together: the value on its own would
// answer with whichever lane held the extremum last.
template <typename T>
struct Extremum {
  T value;
  uint32_t index;
};

// One lane's share of a row: the best of the positions this lane walks, with
// the position it was found at. Only a strictly better element displaces the one
// held, so a tie keeps the position already stored and the first of a row's
// equal elements is the one that answers.
template <typename Op, typename T>
inline void scan_row(constant T* values, uint n, uint lane, T& held,
                     uint32_t& held_index) {
  held = Op::identity();
  held_index = 0;
  for (uint i = lane; i < n; i += simdgroup_size) {
    const T candidate = values[i];
    if (Op::replace(candidate, held)) {
      held = candidate;
      held_index = i;
    }
  }
}

// The group's answer, folded between its lanes with the rule the scan used. Two
// NaNs are the same value, so the lower position among them is kept; otherwise a
// better value displaces the pair held. The order that puts a NaN above every
// finite value and the lowest position above a tie is total, so however the
// lanes are paired the pair every lane ends with is the one the row answers
// with, and it is the pair the scan would have produced over the row in order.
template <typename Op, typename T>
inline void fold_pair(T& held, uint32_t& held_index, uint rival_mask) {
  const T rival = simd_shuffle_xor(held, rival_mask);
  const uint32_t rival_index = simd_shuffle_xor(held_index, rival_mask);
  const bool tied =
      rival == held || (::metal::isnan(rival) && ::metal::isnan(held));
  if (Op::replace(rival, held) || (tied && rival_index < held_index)) {
    held = rival;
    held_index = rival_index;
  }
}

template <typename Op, typename T>
inline Extremum<T> fold_row(T held, uint32_t held_index) {
  // A butterfly over the group's thirty-two lanes: each step leaves every pair
  // of lanes holding one of its two pairs, so five steps answer for the group.
  fold_pair<Op>(held, held_index, 1);
  fold_pair<Op>(held, held_index, 2);
  fold_pair<Op>(held, held_index, 4);
  fold_pair<Op>(held, held_index, 8);
  fold_pair<Op>(held, held_index, 16);
  return {held, held_index};
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

  float held;
  uint32_t held_index;
  scan_row<Op>(values, n, lane, held, held_index);
  const Extremum<float> answer = fold_row<Op>(held, held_index);
  if (lane == 0) {
    out[row] = static_cast<long>(answer.index);
  }
}

// The extremum alone, and the extremum together with the position it came from.
// A paired reduction answers both from one pass, so the value an index names and
// the index a value came from cannot disagree.
template <typename Op>
kernel void row_value_inner(device float* out [[buffer(0)]],
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

  float held;
  uint32_t held_index;
  scan_row<Op>(values, n, lane, held, held_index);
  if (lane == 0) {
    out[row] = fold_row<Op>(held, held_index).value;
  }
}

template <typename Op>
kernel void row_value_arg_inner(device float* out_value [[buffer(0)]],
                                device long* out_index [[buffer(1)]],
                                constant float* src [[buffer(2)]],
                                constant RowSizes& sizes [[buffer(3)]],
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

  float held;
  uint32_t held_index;
  scan_row<Op>(values, n, lane, held, held_index);
  const Extremum<float> answer = fold_row<Op>(held, held_index);
  if (lane == 0) {
    out_value[row] = answer.value;
    out_index[row] = static_cast<long>(answer.index);
  }
}

// Whether every element of a row is nonzero, or at least one of them is. The
// test is against zero of the operand's own type, so a NaN answers set and a
// negative zero does not. One byte per element of the row is folded into the one
// byte a row answers with, so nothing of the operand's type is left to
// reconcile afterwards.
template <typename T>
inline uchar element_is_set(T value) {
  return value != static_cast<T>(0) ? uchar(1) : uchar(0);
}

inline uchar element_is_set(bool value) { return value ? uchar(1) : uchar(0); }

template <typename T>
struct all_nonzero {
  static inline constexpr uchar identity() { return uchar(1); }
  static inline uchar test(T value) { return element_is_set(value); }
  static inline uchar combine(uchar held, uchar seen) {
    return seen < held ? seen : held;
  }
  static inline uchar reduce(uchar held) { return simd_min(held); }
};

template <typename T>
struct any_nonzero {
  static inline constexpr uchar identity() { return uchar(0); }
  static inline uchar test(T value) { return element_is_set(value); }
  static inline uchar combine(uchar held, uchar seen) {
    return seen > held ? seen : held;
  }
  static inline uchar reduce(uchar held) { return simd_max(held); }
};

template <typename Op, typename T>
kernel void row_pred_inner(device bool* out [[buffer(0)]],
                           constant T* src [[buffer(1)]],
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
  constant T* values = src + (uint64_t)row * (uint64_t)n;

  uchar held = Op::identity();
  for (uint i = lane; i < n; i += simdgroup_size) {
    held = Op::combine(held, Op::test(values[i]));
  }
  if (lane == 0) {
    out[row] = Op::reduce(held) != 0;
  }
}

// Float is the only storage type the extrema read: an operand of another type
// reduces in a type of its own, which the graph path builds around the
// reduction. The predicates answer for float and for bool, the two the suite
// drives them with. A call over any other type reaches the graph by leaving the
// host unable to resolve the kernel name.
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

#define REGISTER_VALUE_REDUCTION(HOST_NAME, OP)                         \
  template [[host_name(#HOST_NAME "_float")]] kernel void               \
  row_value_inner<OP<float>>(device float* out, constant float* src,     \
                             constant RowSizes& sizes,                    \
                             uint tptg, uint tgid, uint lane, uint group);

#define REGISTER_VALUE_ARG_REDUCTION(HOST_NAME, OP)                     \
  template [[host_name(#HOST_NAME "_float")]] kernel void               \
  row_value_arg_inner<OP<float>>(device float* out_value,                \
                                 device long* out_index,                 \
                                 constant float* src,                    \
                                 constant RowSizes& sizes,                \
                                 uint tptg, uint tgid, uint lane, uint group);

#define REGISTER_PREDICATE_REDUCTION(HOST_NAME, OP, DTYPE)              \
  template [[host_name(#HOST_NAME "_" #DTYPE)]] kernel void             \
  row_pred_inner<OP<DTYPE>, DTYPE>(device bool* out,                    \
                                   constant DTYPE* src,                 \
                                   constant RowSizes& sizes,            \
                                   uint tptg, uint tgid, uint lane,     \
                                   uint group);

REGISTER_ROW_SUM(float)
REGISTER_ARG_REDUCTION(argmax, arg_max)
REGISTER_ARG_REDUCTION(argmin, arg_min)
REGISTER_VALUE_REDUCTION(rowmax, arg_max)
REGISTER_VALUE_REDUCTION(rowmin, arg_min)
REGISTER_VALUE_ARG_REDUCTION(rowmaxidx, arg_max)
REGISTER_VALUE_ARG_REDUCTION(rowminidx, arg_min)
REGISTER_PREDICATE_REDUCTION(rowall, all_nonzero, float)
REGISTER_PREDICATE_REDUCTION(rowany, any_nonzero, float)
REGISTER_PREDICATE_REDUCTION(rowall, all_nonzero, bool)
REGISTER_PREDICATE_REDUCTION(rowany, any_nonzero, bool)
