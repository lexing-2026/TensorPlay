// Normalization shaders: group normalization and evaluation-mode batch
// normalization.
//
// Group normalization assigns one threadgroup to each (batch, group) pair. The
// group is read twice, once to accumulate the sum and sum of squares and once
// to apply the normalization, and the per-group statistics are reduced inside
// the threadgroup. Batch normalization in evaluation mode only rescales each
// element by the per-channel factor the host derived from the running
// statistics, so one thread per element suffices.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "Normalization.h"
#include "common.h"
#include "utils.h"

using namespace metal;
using p10::metal::simdgroup_size;

// Reduces the group this threadgroup owns and writes its normalized elements.
// Elements are visited in ascending order in blocks of kReads, each thread
// taking the block its index selects, which keeps neighbouring threads on
// neighbouring addresses. Affine selects whether the per-channel scale and
// bias are read at all.
template <typename T, bool Affine>
inline void group_norm_body(device T* y, constant T* x, constant T* gamma,
                            constant T* beta, constant GroupNormParams& p,
                            float eps, uint tgid, uint tid, uint tptg,
                            uint lane, uint sgroup) {
  constexpr int kReads = 4;
  const uint base = tgid * p.elements_per_group;
  x += base;
  y += base;

  float sum = 0.0f;
  float sumsq = 0.0f;
  for (uint r = 0; r < p.elements_per_group; r += tptg * kReads) {
    const uint at = r + tid * kReads;
    for (int i = 0; i < kReads; ++i) {
      if (at + static_cast<uint>(i) < p.elements_per_group) {
        const float v = static_cast<float>(x[at + static_cast<uint>(i)]);
        sum += v;
        sumsq = fma(v, v, sumsq);
      }
    }
  }

  threadgroup float partial_sum[simdgroup_size];
  threadgroup float partial_sumsq[simdgroup_size];
  threadgroup float statistics[2];

  sum = simd_sum(sum);
  sumsq = simd_sum(sumsq);
  if (sgroup == 0) {
    partial_sum[lane] = 0.0f;
    partial_sumsq[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    partial_sum[sgroup] = sum;
    partial_sumsq[sgroup] = sumsq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgroup == 0) {
    sum = simd_sum(partial_sum[lane]);
    sumsq = simd_sum(partial_sumsq[lane]);
    if (lane == 0) {
      const float m = static_cast<float>(p.elements_per_group);
      const float mean = sum / m;
      // The one-pass form sumsq / m - mean^2 can dip below zero through
      // rounding; the variance is mathematically non-negative.
      const float var = max(sumsq / m - mean * mean, 0.0f);
      statistics[0] = mean;
      statistics[1] = metal::precise::rsqrt(var + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = statistics[0];
  const float rstd = statistics[1];

  const uint channel_base =
      (tgid % p.num_groups) * p.channels_per_group;
  for (uint r = 0; r < p.elements_per_group; r += tptg * kReads) {
    const uint at = r + tid * kReads;
    for (int i = 0; i < kReads; ++i) {
      const uint e = at + static_cast<uint>(i);
      if (e < p.elements_per_group) {
        float v = (static_cast<float>(x[e]) - mean) * rstd;
        if (Affine) {
          const uint c = channel_base + e / p.hxw;
          v = v * static_cast<float>(gamma[c]) + static_cast<float>(beta[c]);
        }
        y[e] = static_cast<T>(v);
      }
    }
  }
}

template <typename T>
kernel void group_norm_plain(
    device T* out [[buffer(0)]],
    constant T* x [[buffer(1)]],
    constant GroupNormParams& params [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgroup [[simdgroup_index_in_threadgroup]]) {
  group_norm_body<T, false>(out, x, nullptr, nullptr, params, eps, tgid, tid,
                            tptg, lane, sgroup);
}

template <typename T>
kernel void group_norm_affine(
    device T* out [[buffer(0)]],
    constant T* x [[buffer(1)]],
    constant T* gamma [[buffer(2)]],
    constant T* beta [[buffer(3)]],
    constant GroupNormParams& params [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgroup [[simdgroup_index_in_threadgroup]]) {
  group_norm_body<T, true>(out, x, gamma, beta, params, eps, tgid, tid, tptg,
                           lane, sgroup);
}

// Evaluation-mode batch normalization: out = x * scale[c] + shift[c]. The host
// folds the running statistics, epsilon and the optional weight and bias into
// the two per-channel arrays, so the kernel is a pure elementwise pass.
template <typename T>
kernel void batch_norm_eval(
    device T* out [[buffer(0)]],
    constant T* x [[buffer(1)]],
    constant float* scale [[buffer(2)]],
    constant float* shift [[buffer(3)]],
    constant uint& spatial [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const uint c = tid.y;
  const uint offset = (tid.z * channels + c) * spatial + tid.x;
  out[offset] =
      static_cast<T>(static_cast<float>(x[offset]) * scale[c] + shift[c]);
}

#define REGISTER_GROUP_NORM_PLAIN(DTYPE)                              \
  template [[host_name("group_norm_" #DTYPE)]] kernel void            \
  group_norm_plain<DTYPE>(                                            \
      device DTYPE * out [[buffer(0)]],                               \
      constant DTYPE * x [[buffer(1)]],                               \
      constant GroupNormParams & params [[buffer(2)]],                \
      constant float & eps [[buffer(3)]],                             \
      uint tgid [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint tptg [[threads_per_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]],                        \
      uint sgroup [[simdgroup_index_in_threadgroup]]);

#define REGISTER_GROUP_NORM_AFFINE(DTYPE)                             \
  template [[host_name("group_norm_" #DTYPE "_affine")]] kernel void  \
  group_norm_affine<DTYPE>(                                           \
      device DTYPE * out [[buffer(0)]],                               \
      constant DTYPE * x [[buffer(1)]],                               \
      constant DTYPE * gamma [[buffer(2)]],                           \
      constant DTYPE * beta [[buffer(3)]],                            \
      constant GroupNormParams & params [[buffer(4)]],                \
      constant float & eps [[buffer(5)]],                             \
      uint tgid [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                    \
      uint tptg [[threads_per_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]],                        \
      uint sgroup [[simdgroup_index_in_threadgroup]]);

#define REGISTER_NORMALIZATION_DTYPE(DTYPE) \
  REGISTER_GROUP_NORM_PLAIN(DTYPE)          \
  REGISTER_GROUP_NORM_AFFINE(DTYPE)         \
  template [[host_name("batch_norm_eval_" #DTYPE)]] kernel void         \
  batch_norm_eval<DTYPE>(                                               \
      device DTYPE * out [[buffer(0)]],                                 \
      constant DTYPE * x [[buffer(1)]],                                 \
      constant float * scale [[buffer(2)]],                             \
      constant float * shift [[buffer(3)]],                             \
      constant uint & spatial [[buffer(4)]],                            \
      constant uint & channels [[buffer(5)]],                           \
      uint3 tid [[thread_position_in_grid]]);

REGISTER_NORMALIZATION_DTYPE(float)
REGISTER_NORMALIZATION_DTYPE(half)
REGISTER_NORMALIZATION_DTYPE(bfloat)
