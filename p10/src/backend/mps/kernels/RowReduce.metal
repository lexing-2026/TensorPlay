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
  // One element per chain per step, the chains one group-wide stride apart, so
  // every load instruction covers one contiguous stretch of the row while the
  // chains keep several loads in flight.
  const uint stride = simdgroup_size * kSumChains;
  const uint whole = (n / stride) * stride;
  for (uint base = 0; base < whole; base += stride) {
#pragma unroll
    for (uint j = 0; j < kSumChains; ++j) {
      acc[j] += values[base + j * simdgroup_size + lane];
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
inline void scan_row(constant T* values, uint n, uint lane, thread T& held,
                     thread uint32_t& held_index) {
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
inline void fold_pair(thread T& held, thread uint32_t& held_index,
                      uint rival_mask) {
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
  // The fold exchanges values between the whole group, so every lane takes
  // part in it; only the write is one lane's business.
  const Extremum<float> answer = fold_row<Op>(held, held_index);
  if (lane == 0) {
    out[row] = answer.value;
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
// negative zero does not. Each element answers with one bit held in a lane's
// counter, the counters fold by the AND or the OR of the question, and the row
// leaves the group as the one byte that holds its answer.
template <typename T>
inline uint element_is_set(T value) {
  return value != static_cast<T>(0) ? 1u : 0u;
}

inline uint element_is_set(bool value) { return value ? 1u : 0u; }

template <typename T>
struct all_nonzero {
  static inline constexpr uint identity() { return 1u; }
  static inline uint test(T value) { return element_is_set(value); }
  static inline uint combine(uint held, uint seen) {
    return seen < held ? seen : held;
  }
  static inline uint reduce(uint held) { return simd_min(held); }
};

template <typename T>
struct any_nonzero {
  static inline constexpr uint identity() { return 0u; }
  static inline uint test(T value) { return element_is_set(value); }
  static inline uint combine(uint held, uint seen) {
    return seen > held ? seen : held;
  }
  static inline uint reduce(uint held) { return simd_max(held); }
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

  metal::array<uint, kSumChains> held;
#pragma unroll
  for (uint j = 0; j < kSumChains; ++j) {
    held[j] = Op::identity();
  }
  // Several independent chains per lane, like the sum kernel: one accumulator
  // would serialize the walk on a comparison per load, while the chains keep a
  // whole group of loads in flight and each load instruction covers one
  // contiguous stretch of the row.
  const uint stride = simdgroup_size * kSumChains;
  const uint whole = (n / stride) * stride;
  for (uint base = 0; base < whole; base += stride) {
#pragma unroll
    for (uint j = 0; j < kSumChains; ++j) {
      held[j] = Op::combine(held[j],
                            Op::test(values[base + j * simdgroup_size + lane]));
    }
  }
  for (uint i = whole + lane; i < n; i += simdgroup_size) {
    held[0] = Op::combine(held[0], Op::test(values[i]));
  }
  uint acc = held[0];
#pragma unroll
  for (uint j = 1; j < kSumChains; ++j) {
    acc = Op::combine(acc, held[j]);
  }
  // As for the extrema, the step that joins the lanes asks all of them, so it
  // happens before any lane is left behind.
  const uint folded = Op::reduce(acc);
  if (lane == 0) {
    out[row] = folded != 0;
  }
}

// log(sum(exp(x - max))) + max over one row. One SIMD group walks the row
// twice: the first pass keeps the maximum, the second accumulates the shifted
// exponentials. A row whose maximum is infinite holds no finite value to shift
// towards -- exp(inf - inf) would be a NaN -- so that row answers with its
// maximum directly, an infinity of the same sign.
template <typename T>
kernel void logsumexp_row(device T* out [[buffer(0)]],
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

  T maximum = T(-INFINITY);
  for (uint i = lane; i < n; i += simdgroup_size) {
    maximum = max(maximum, values[i]);
  }
  maximum = simd_max(maximum);

  float total = 0.0f;
  for (uint i = lane; i < n; i += simdgroup_size) {
    total += exp(static_cast<float>(values[i]) - static_cast<float>(maximum));
  }
  total = simd_sum(total);

  if (lane == 0) {
    out[row] = (maximum == T(-INFINITY) || maximum == T(INFINITY))
        ? maximum
        : static_cast<T>(log(total) + static_cast<float>(maximum));
  }
}

// A row far longer than one group can walk would leave every group but one
// idle in the per-row kernels above. The chunk kernels split such a row into
// fixed-size chunks, one group per chunk, and write one partial per chunk; the
// host folds the partials with a per-row kernel in a second pass. The
// accumulation inside a chunk is the one row_sum_inner uses, so a chunk costs
// the same per-element work a whole row does -- there are just enough groups
// to fill the device.
template <typename T>
kernel void sum_chunk(device T* out [[buffer(0)]],
                      constant T* src [[buffer(1)]],
                      constant uint& numel [[buffer(2)]],
                      constant uint& chunk [[buffer(3)]],
                      uint tptg [[threads_per_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint group [[simdgroup_index_in_threadgroup]]) {
  const uint gid = tgid * (tptg / simdgroup_size) + group;
  const uint first = gid * chunk;
  if (first >= numel) {
    return;
  }
  const uint last = min(first + chunk, numel);
  const uint count = last - first;

  metal::array<T, kSumChains> acc;
#pragma unroll
  for (uint j = 0; j < kSumChains; ++j) {
    acc[j] = T(0);
  }
  const uint stride = simdgroup_size * kSumChains;
  const uint whole = (count / stride) * stride;
  for (uint base = first; base < first + whole; base += stride) {
#pragma unroll
    for (uint j = 0; j < kSumChains; ++j) {
      acc[j] += src[base + j * simdgroup_size + lane];
    }
  }
  for (uint i = first + whole + lane; i < last; i += simdgroup_size) {
    acc[0] += src[i];
  }
  T total = acc[0];
#pragma unroll
  for (uint j = 1; j < kSumChains; ++j) {
    total += acc[j];
  }
  total = simd_sum(total);
  if (lane == 0) {
    out[gid] = total;
  }
}

// The extremum of one chunk, folded between the lanes with the same rule the
// row scan uses, so a NaN anywhere in the row survives to the partial and the
// later fold keeps it.
template <typename Op>
kernel void extremum_chunk(device float* out [[buffer(0)]],
                           constant float* src [[buffer(1)]],
                           constant uint& numel [[buffer(2)]],
                           constant uint& chunk [[buffer(3)]],
                           uint tptg [[threads_per_threadgroup]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint group [[simdgroup_index_in_threadgroup]]) {
  const uint gid = tgid * (tptg / simdgroup_size) + group;
  const uint first = gid * chunk;
  if (first >= numel) {
    return;
  }
  const uint last = min(first + chunk, numel);
  const uint count = last - first;

  // Several independent chains per lane, like the sum kernel, so the walk is
  // not serialized on one comparison and each load instruction covers one
  // contiguous stretch of the chunk. Folding the chains at the end keeps the
  // replacement rule of a single walk.
  metal::array<float, kSumChains> held;
#pragma unroll
  for (uint j = 0; j < kSumChains; ++j) {
    held[j] = Op::identity();
  }
  const uint stride = simdgroup_size * kSumChains;
  const uint whole = (count / stride) * stride;
  for (uint base = first; base < first + whole; base += stride) {
#pragma unroll
    for (uint j = 0; j < kSumChains; ++j) {
      const float candidate = src[base + j * simdgroup_size + lane];
      if (Op::replace(candidate, held[j])) {
        held[j] = candidate;
      }
    }
  }
  for (uint i = first + whole + lane; i < last; i += simdgroup_size) {
    const float candidate = src[i];
    if (Op::replace(candidate, held[0])) {
      held[0] = candidate;
    }
  }
  float best = held[0];
#pragma unroll
  for (uint j = 1; j < kSumChains; ++j) {
    if (Op::replace(held[j], best)) {
      best = held[j];
    }
  }
#pragma unroll
  for (uint mask = 1; mask < simdgroup_size; mask <<= 1) {
    const float rival = simd_shuffle_xor(best, mask);
    if (Op::replace(rival, best)) {
      best = rival;
    }
  }
  if (lane == 0) {
    out[gid] = best;
  }
}

// softmax over one row: exp(x - max) / sum(exp(x - max)). The row is cached in
// threadgroup memory as it is read, so the maximum, the shifted exponentials
// and the final scaling cost one global read and one global write. Each SIMD
// group owns the cache region at its own offset, and every lane reads back only
// the positions it wrote, so no barrier stands between the passes.
[[host_name("softmax_row_float")]]
kernel void softmax_row(device float* out [[buffer(0)]],
                        constant float* src [[buffer(1)]],
                        constant RowSizes& sizes [[buffer(2)]],
                        threadgroup float* cache [[threadgroup(0)]],
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
  device float* out_row = out + (uint64_t)row * (uint64_t)n;
  threadgroup float* cache_row = cache + (uint64_t)group * (uint64_t)n;

  float maximum = -INFINITY;
  for (uint i = lane; i < n; i += simdgroup_size) {
    const float value = values[i];
    cache_row[i] = value;
    maximum = max(maximum, value);
  }
  maximum = simd_max(maximum);

  float total = 0.0f;
  for (uint i = lane; i < n; i += simdgroup_size) {
    const float e = exp(cache_row[i] - maximum);
    cache_row[i] = e;
    total += e;
  }
  total = simd_sum(total);

  const float scale = 1.0f / total;
  for (uint i = lane; i < n; i += simdgroup_size) {
    out_row[i] = cache_row[i] * scale;
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

#define REGISTER_SUM_CHUNK(DTYPE)                                       \
  template [[host_name("sumchunk_" #DTYPE)]] kernel void                \
  sum_chunk<DTYPE>(device DTYPE* out, constant DTYPE* src,              \
                   constant uint& numel, constant uint& chunk,          \
                   uint tptg, uint tgid, uint lane, uint group);

#define REGISTER_EXTREMUM_CHUNK(HOST_NAME, OP)                          \
  template [[host_name(#HOST_NAME "_float")]] kernel void               \
  extremum_chunk<OP>(device float* out, constant float* src,            \
                     constant uint& numel, constant uint& chunk,        \
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
REGISTER_SUM_CHUNK(float)
REGISTER_EXTREMUM_CHUNK(maxchunk, arg_max<float>)
REGISTER_EXTREMUM_CHUNK(minchunk, arg_min<float>)
template [[host_name("logsumexp_row_float")]] kernel void
    logsumexp_row<float>(device float* out, constant float* src,
                         constant RowSizes& sizes, uint tptg, uint tgid,
                         uint lane, uint group);
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
