#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Allocator.h"
#include "CudaGemm.h"
#include "GradMode.h"
#include "../composite/AttentionComposite.h"
#include <cuda_runtime.h>
// Tensor-core primitive API: <mma.h> on the CUDA toolchain; on HIP the
// RDNA3 WMMA instruction backs a compatible subset (WmmaRocmCompat.cuh).
#if defined(USE_ROCM)
#include "WmmaRocmCompat.cuh"
#else
#include <mma.h>
#endif
#include <optional>
#include <vector>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <type_traits>

// Native aligned flash reference.  This is the standalone CUDA/CUTE kernel
// source used for the schedule comparison; FLASHATTENTION_DISABLE_DROPOUT
#if !defined(USE_ROCM) && \
    __has_include("../../../../third_party/pytorch/third_party/flash-attention/csrc/flash_attn/src/flash.h") && \
    __has_include("../../../../third_party/pytorch/third_party/cutlass/include/cute/tensor.hpp")
#define TP_HAS_NATIVE_CUTE_FLASH 1
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASH_NAMESPACE tensorplay_native_flash
#include "../../../../third_party/pytorch/third_party/flash-attention/csrc/flash_attn/src/flash.h"
#include "../../../../third_party/pytorch/third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h"
#undef FLASH_NAMESPACE
#undef FLASHATTENTION_DISABLE_SOFTCAP
#undef FLASHATTENTION_DISABLE_LOCAL
#undef FLASHATTENTION_DISABLE_ALIBI
#undef FLASHATTENTION_DISABLE_DROPOUT
#endif

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)


// Scaled dot-product attention forward.
// References:
//   - impl 0 (naive): textbook O(T^2 * D) math attention, scores in smem.
//   - impl 1 (flash): flash-attention-v1 style tiling with online softmax
//     (rescaling), no O(T^2) memory; kv tiled in blocks of Br.
//     cuda/attention.cu (FlashAttentionForwardKernel) — simplified to a
//     non-cutlass reference implementation.

namespace tensorplay {
namespace cuda {

namespace {

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

template <typename T>
__device__ inline T warpReduceMax(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = max(val, __shfl_down_sync(0xffffffffffffffffull, val, offset));
  return val;
}

template <typename T>
__device__ inline T warpReduceSum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffffffffffull, val, offset);
  return val;
}

template <typename T>
__device__ inline T blockReduceMax(T val, T* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : static_cast<T>(-INFINITY);
  if (wid == 0) val = warpReduceMax(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

template <typename T>
__device__ inline T blockReduceSum(T val, T* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : static_cast<T>(0);
  if (wid == 0) val = warpReduceSum(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// impl 0: naive math attention. One block per (b, h, t). scores row in smem.
// Requires T <= 4096 (scores fit in shared memory).
// ---------------------------------------------------------------------------

__global__ void sdpa_naive_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale, bool is_causal) {
  extern __shared__ float smem[];  // scores[T] + blockDim floats
  float* scores = smem;
  float* red = smem + T;

  int64_t bh = blockIdx.x;
  int64_t t = blockIdx.y;
  int64_t b = bh / H, h = bh % H;
  int64_t bhT = (b * H + h) * T;
  const float* q_row = q + (bhT + t) * D;
  const float* k_base = k + bhT * D;
  const float* v_base = v + bhT * D;
  float* out_row = out + (bhT + t) * D;

  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x) {
    if (is_causal && kk > t) {
      scores[kk] = -INFINITY;
      continue;
    }
    const float* k_row = k_base + kk * D;
    float s = 0.f;
    for (int64_t d = 0; d < D; ++d)
      s += q_row[d] * k_row[d];
    scores[kk] = s * scale;
  }
  __syncthreads();

  float mx = -INFINITY;
  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x)
    mx = max(mx, scores[kk]);
  mx = blockReduceMax(mx, red);

  float sum = 0.f;
  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x) {
    float e = expf(scores[kk] - mx);
    scores[kk] = e;
    sum += e;
  }
  sum = blockReduceSum(sum, red);
  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x)
    scores[kk] /= sum;
  __syncthreads();  // output phase reads all of scores; must wait for the division

  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int64_t kk = 0; kk < T; ++kk)
      acc += scores[kk] * v_base[kk * D + d];
    out_row[d] = acc;
  }
}

// ---------------------------------------------------------------------------
// impl 1: flash-attention-v1 style tiling + online softmax.
// One block per (b, h, q-tile of Bq rows); kv processed in tiles of Br.
// Supports T of any size; D <= 128. fp32/fp16/bf16 inputs.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return (float)v;
}

template <>
__device__ __forceinline__ float to_float<tensorplay::Half>(tensorplay::Half v) {
  return (float)v;
}

template <>
__device__ __forceinline__ float to_float<tensorplay::BFloat16>(tensorplay::BFloat16 v) {
  return (float)v;
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  return (T)v;
}

template <>
__device__ __forceinline__ tensorplay::Half from_float<tensorplay::Half>(float v) {
  return tensorplay::Half(v);
}

template <>
__device__ __forceinline__ tensorplay::BFloat16 from_float<tensorplay::BFloat16>(float v) {
  return tensorplay::BFloat16(v);
}

template <typename DT>
__global__ void sdpa_flash_kernel(
    const DT* __restrict__ q, const DT* __restrict__ k, const DT* __restrict__ v,
    DT* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale, bool is_causal) {
  constexpr int Bq = 16, Br = 16, kThreads = 128;
  // smem layout: q_s[Bq*128] | s_s[Bq*Br] | m_s[Bq] | l_s[Bq] | mnew_s[Bq] |
  //              alpha_s[Bq] | acc_s[Bq*128] | red_s[Bq*8]
  extern __shared__ float smem[];
  float* q_s = smem;
  float* s_s = q_s + Bq * 128;
  float* m_s = s_s + Bq * Br;
  float* l_s = m_s + Bq;
  float* mnew_s = l_s + Bq;
  float* alpha_s = mnew_s + Bq;
  float* acc_s = alpha_s + Bq;
  float* red_s = acc_s + Bq * 128;

  int64_t bh = blockIdx.x;
  int64_t q_tile = blockIdx.y;
  int64_t b = bh / H, h = bh % H;
  int64_t bhT = (b * H + h) * T;
  const DT* q_base = q + bhT * D;
  const DT* k_base = k + bhT * D;
  const DT* v_base = v + bhT * D;
  DT* out_base = out + bhT * D;

  int qk_lane = threadIdx.x & 15;
  int lane8 = threadIdx.x >> 4;  // 0..7

  // q tile stored as [Bq][128] (stride matches the score loop's q_row access);
  // entries beyond D are zeroed.
  for (int d = threadIdx.x; d < Bq * 128; d += kThreads) {
    int qk = d / 128, dd = d % 128;
    int64_t qg = q_tile * Bq + qk;
    q_s[d] = (qg < T && dd < D) ? to_float(q_base[qg * D + dd]) : 0.f;
  }
  if (threadIdx.x < Bq) {
    m_s[threadIdx.x] = -INFINITY;
    l_s[threadIdx.x] = 0.f;
  }
  for (int d = threadIdx.x; d < Bq * 128; d += kThreads)
    acc_s[d] = 0.f;
  __syncthreads();

  for (int64_t kk0 = 0; kk0 < T; kk0 += Br) {
    // 1. scores[Bq][Br]: each thread computes 2 entries (idx = qk*Br + kk)
    for (int half = 0; half < 2; ++half) {
      int idx = threadIdx.x + half * kThreads;
      int qk = idx >> 4, kk = idx & 15;
      int64_t kk_g = kk0 + kk;
      int64_t qg = q_tile * Bq + qk;
      float s = -INFINITY;
      if (kk_g < T && (!is_causal || qg >= kk_g)) {
        const DT* k_row = k_base + kk_g * D;
        const float* q_row = q_s + qk * 128;
        float dot = 0.f;
        for (int64_t d = 0; d < D; ++d)
          dot += q_row[d] * to_float(k_row[d]);
        s = dot * scale;
      }
      s_s[idx] = s;
    }
    __syncthreads();

    // 2. online softmax: per q-row, 8 lanes each max over 2 scores
    float mnew = -INFINITY;
    for (int i = lane8; i < Br; i += 8)
      mnew = max(mnew, s_s[qk_lane * Br + i]);
    red_s[qk_lane * 8 + lane8] = mnew;
    __syncthreads();
    if (lane8 == 0) {
      float m = -INFINITY;
      for (int i = 0; i < 8; ++i) m = max(m, red_s[qk_lane * 8 + i]);
      m = max(m, m_s[qk_lane]);
      float alpha = expf(m_s[qk_lane] - m);
      float rowsum = 0.f;
      for (int i = 0; i < Br; ++i)
        rowsum += expf(s_s[qk_lane * Br + i] - m);
      l_s[qk_lane] = l_s[qk_lane] * alpha + rowsum;
      m_s[qk_lane] = m;
      mnew_s[qk_lane] = m;
      alpha_s[qk_lane] = alpha;
    }
    __syncthreads();

    // 3. accumulate: thread (qk_lane, lane8) owns d in {lane8, lane8+8, ...}
    for (int d = lane8; d < D; d += 8) {
      float a = acc_s[qk_lane * 128 + d] * alpha_s[qk_lane];
      for (int kk = 0; kk < Br; ++kk) {
        float p = expf(s_s[qk_lane * Br + kk] - mnew_s[qk_lane]);
        if (p > 0.f) {
          int64_t kk_g = kk0 + kk;
          a += p * to_float(v_base[kk_g * D + d]);
        }
      }
      acc_s[qk_lane * 128 + d] = a;
    }
    __syncthreads();
  }

  // 4. normalize and write out
  int64_t qg = q_tile * Bq + qk_lane;
  if (qg < T && l_s[qk_lane] > 0.f) {
    for (int d = lane8; d < D; d += 8)
      out_base[qg * D + d] = from_float<DT>(acc_s[qk_lane * 128 + d] / l_s[qk_lane]);
  }
}

// Warp-tiled online-softmax attention.  The original impl=1 kernel assigns
// one thread to each score and serializes all D dot-product terms in that
// thread.  That is functional but leaves tensor-core capable GPUs mostly
// idle for Llama's D=128 heads.  This variant assigns one warp to one query
// row: lanes cooperate on the D reduction and then accumulate V in parallel.
// It keeps the O(T) workspace and exact causal semantics of impl=1 while
// avoiding the O(T^2) score materialization used by impl=2.
template <typename DT>
__global__ void sdpa_warp_flash_kernel(
    const DT* __restrict__ q, const DT* __restrict__ k,
    const DT* __restrict__ v, DT* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale,
    bool is_causal) {
  constexpr int q_rows_per_block = 4;
  constexpr int k_tile = 64;
  constexpr int warp_threads = 32;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;

  const int lane = threadIdx.x & (warp_threads - 1);
  const int warp = threadIdx.x / warp_threads;
  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t qg = static_cast<int64_t>(blockIdx.y) * q_rows_per_block + warp;
  if (qg >= T) return;

  const int64_t bh_base = bh * T * D;
  const DT* q_row = q + bh_base + qg * D;
  const DT* k_base = k + bh_base;
  const DT* v_base = v + bh_base;
  DT* out_row = out + bh_base + qg * D;

  // Lane 0 writes one score/probability row; the rest of the warp consumes
  // it after the warp-local barriers.  Four independent rows share one
  // block, so the shared footprint is only 1 KiB.
  __shared__ float probabilities[q_rows_per_block][k_tile];

  float accumulator[4] = {0.f, 0.f, 0.f, 0.f};
  float row_max = -INFINITY;
  float row_sum = 0.f;

  for (int64_t k0 = 0; k0 < T; k0 += k_tile) {
    for (int kk = 0; kk < k_tile; ++kk) {
      const int64_t kg = k0 + kk;
      float dot = 0.f;
      if (kg < T && (!is_causal || kg <= qg)) {
        const DT* k_row = k_base + kg * D;
        for (int64_t d = lane; d < D; d += warp_threads)
          dot += to_float(q_row[d]) * to_float(k_row[d]);
        for (int offset = 16; offset > 0; offset >>= 1)
          dot += __shfl_down_sync(full_mask, dot, offset);
        if (lane == 0) probabilities[warp][kk] = dot * scale;
      } else if (lane == 0) {
        probabilities[warp][kk] = -INFINITY;
      }
    }
    __syncwarp(full_mask);

    float alpha_lane = 0.f;
    if (lane == 0) {
      float tile_max = -INFINITY;
      for (int kk = 0; kk < k_tile; ++kk)
        tile_max = max(tile_max, probabilities[warp][kk]);
      const float new_max = max(row_max, tile_max);
      alpha_lane = isfinite(row_max) ? expf(row_max - new_max) : 0.f;
      float tile_sum = 0.f;
      for (int kk = 0; kk < k_tile; ++kk) {
        const float score = probabilities[warp][kk];
        const float p = isfinite(score) ? expf(score - new_max) : 0.f;
        probabilities[warp][kk] = p;
        tile_sum += p;
      }
      row_sum = row_sum * alpha_lane + tile_sum;
      row_max = new_max;
    }
    __syncwarp(full_mask);

    const float alpha = __shfl_sync(full_mask, alpha_lane, 0);
    for (int j = 0; j < 4; ++j) {
      const int64_t d = lane + static_cast<int64_t>(j) * warp_threads;
      if (d >= D) continue;
      float value = accumulator[j] * alpha;
      for (int kk = 0; kk < k_tile; ++kk) {
        const int64_t kg = k0 + kk;
        if (kg < T) {
          value += probabilities[warp][kk] * to_float(v_base[kg * D + d]);
        }
      }
      accumulator[j] = value;
    }
    __syncwarp(full_mask);
  }

  const float normalizer = __shfl_sync(full_mask, row_sum, 0);
  if (normalizer > 0.f) {
    for (int j = 0; j < 4; ++j) {
      const int64_t d = lane + static_cast<int64_t>(j) * warp_threads;
      if (d < D)
        out_row[d] = from_float<DT>(accumulator[j] / normalizer);
    }
  }
}

// GEMM-backed attention path for FP32.  The reference flash kernel above is
// deliberately self-contained, but its score tile computes every dot product
// serially in one thread and therefore does not use the 4090's GEMM engines.
// For the real Llama-shaped FP32 benchmark, two strided-batched cuBLAS GEMMs
// plus one fused causal softmax are materially faster at medium/long context.
// The transpose is a bandwidth-only pass; it lets cuBLAS consume a regular
// row-major [D,T] operand without creating a per-head dispatcher call.

template <typename DT>
__global__ void sdpa_transpose_k_kernel(
    const DT* __restrict__ input, DT* __restrict__ output,
    int64_t tokens, int64_t head_dim) {
  constexpr int tile = 32;
  __shared__ DT smem[tile][tile + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t head = static_cast<int64_t>(blockIdx.z);
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * tile;
  const int64_t d0 = static_cast<int64_t>(blockIdx.y) * tile;
  const int64_t input_head = head * tokens * head_dim;
  const int64_t output_head = head * head_dim * tokens;

  // A 32x8 block loads four rows.  Loads are contiguous in the original
  // [T,D] layout; the shared tile makes the writes contiguous in the
  // transposed [D,T] layout as well, avoiding the old per-element 64-bit
  // div/mod address arithmetic.
  for (int i = 0; i < 4; ++i) {
    const int t = ty + i * 8;
    const int64_t tg = t0 + t;
    const int64_t dg = d0 + tx;
    smem[t][tx] = (tg < tokens && dg < head_dim)
        ? input[input_head + tg * head_dim + dg]
        : from_float<DT>(0.f);
  }
  __syncthreads();
  for (int i = 0; i < 4; ++i) {
    const int d = ty + i * 8;
    const int64_t dg = d0 + d;
    const int64_t tg = t0 + tx;
    if (dg < head_dim && tg < tokens)
      output[output_head + dg * tokens + tg] = smem[tx][d];
  }
}

template <typename DT>
__global__ void sdpa_softmax_kernel(
    DT* __restrict__ scores, int64_t rows, int64_t tokens,
    bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows * tokens) return;
  const int64_t token = row % tokens;
  DT* values = scores + row * tokens;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;

  // One warp is enough for a row and avoids the three block-wide barriers in
  // the old 256-thread reducer.  Each lane still walks a coalesced strip of
  // the score row, while the causal mask is applied before the max reduction.
  float maximum = -INFINITY;
  for (int64_t j = threadIdx.x; j < tokens; j += 32) {
    float value = to_float(values[j]);
    if (is_causal && j > token) value = -INFINITY;
    values[j] = from_float<DT>(value);
    maximum = max(maximum, value);
  }
  maximum = warpReduceMax(maximum);
  maximum = __shfl_sync(full_mask, maximum, 0);

  float total = 0.f;
  for (int64_t j = threadIdx.x; j < tokens; j += 32) {
    const float value = to_float(values[j]);
    const float probability = isfinite(value) ? expf(value - maximum) : 0.f;
    values[j] = from_float<DT>(probability);
    total += probability;
  }
  total = warpReduceSum(total);
  total = __shfl_sync(full_mask, total, 0);
  const float inverse = total > 0.f ? 1.f / total : 0.f;
  for (int64_t j = threadIdx.x; j < tokens; j += 32)
    values[j] = from_float<DT>(to_float(values[j]) * inverse);
}

// A compact FP16 Tensor Core attention path for the Llama head shape
// (D=128).  It follows the same fused QK -> online softmax -> PV structure
// TensorPlay tensor ABI and dispatcher independent.  One block owns 16 query
// rows of one head; four warps compute QK tiles and eight warps compute the
// PV output tiles.  The 16 remaining warps each own one online-softmax row.
__device__ inline __half tp_half_to_cuda(tensorplay::Half value) {
  return *reinterpret_cast<const __half*>(&value);
}

__device__ inline tensorplay::Half tp_half_from_cuda(__half value) {
  tensorplay::Half result;
  result.x = __half_as_ushort(value);
  return result;
}

__global__ void sdpa_wmma_flash_half_kernel(
    const tensorplay::Half* __restrict__ q,
    const tensorplay::Half* __restrict__ k,
    const tensorplay::Half* __restrict__ v,
    tensorplay::Half* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale,
    bool is_causal) {
  using namespace nvcuda;
  constexpr int q_tile = 16;
  constexpr int k_tile = 64;
  constexpr int tile_d = 128;
  constexpr int threads = 512;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;

  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t q0 = static_cast<int64_t>(blockIdx.y) * q_tile;
  const int64_t bh_base = bh * T * D;

  __shared__ __half k_s[k_tile][tile_d];
  __shared__ __half v_s[k_tile][tile_d];
  __shared__ float score_s[q_tile][k_tile];
  __shared__ __half p_s[q_tile][k_tile];
  __shared__ float acc_s[q_tile][tile_d];
  __shared__ float row_max[q_tile];
  __shared__ float row_sum[q_tile];
  __shared__ float row_alpha[q_tile];

  if (thread < q_tile) {
    row_max[thread] = -INFINITY;
    row_sum[thread] = 0.f;
    row_alpha[thread] = 0.f;
  }
  for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
    const int qr = idx / tile_d;
    const int d = idx % tile_d;
    acc_s[qr][d] = 0.f;
  }
  __syncthreads();

  // Causal attention never needs key tiles past the end of the query tile
  // (rows q0..q0+15 only attend to keys 0..q0+15); the non-causal walk must
  // cover the whole key axis.  The final tile may still contain masked
  // columns either way.
  const int64_t last_k =
      is_causal ? min(T, q0 + q_tile) : T;
  for (int64_t k0 = 0; k0 < last_k; k0 += k_tile) {
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      const bool valid = kg < T && d < D;
      k_s[kr][d] = valid
          ? tp_half_to_cuda(k[bh_base + kg * D + d])
          : __float2half(0.f);
      v_s[kr][d] = valid
          ? tp_half_to_cuda(v[bh_base + kg * D + d])
          : __float2half(0.f);
    }
    __syncthreads();

    // QK^T: one warp owns each 16x16 score tile.  The column-major view of
    // K reads the row-major shared tile as K^T without another transpose.
    if (warp < q_tile / 16 * k_tile / 16) {
      const int n0 = (warp % (k_tile / 16)) * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                     wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                     wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.f);
      for (int d0 = 0; d0 < tile_d; d0 += 16) {
        const int64_t q_offset = bh_base + q0 * D + d0;
        const __half* q_ptr = reinterpret_cast<const __half*>(q + q_offset);
        wmma::load_matrix_sync(a_frag, q_ptr, D);
        wmma::load_matrix_sync(b_frag, &k_s[n0][d0], tile_d);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(&score_s[0][n0], c_frag, k_tile,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // Online softmax: one warp per query row, with lanes walking the 64-key
    // tile.  The score tile is rewritten in place as half probabilities for
    // the following Tensor Core PV multiply.
    if (warp < q_tile) {
      const int qr = warp;
      const int64_t qg = q0 + qr;
      float maximum = -INFINITY;
      for (int kk = lane; kk < k_tile; kk += 32) {
        const int64_t kg = k0 + kk;
        float score = score_s[qr][kk] * scale;
        if (is_causal && kg > qg) score = -INFINITY;
        if (kg >= T || qg >= T) score = -INFINITY;
        score_s[qr][kk] = score;
        maximum = max(maximum, score);
      }
      maximum = warpReduceMax(maximum);
      maximum = __shfl_sync(full_mask, maximum, 0);

      float alpha = 0.f;
      if (lane == 0) {
        const float old_max = row_max[qr];
        const float new_max = max(old_max, maximum);
        alpha = isfinite(old_max) ? expf(old_max - new_max) : 0.f;
        float tile_sum = 0.f;
        for (int kk = 0; kk < k_tile; ++kk) {
          const float score = score_s[qr][kk];
          const float probability = isfinite(score)
              ? expf(score - new_max) : 0.f;
          p_s[qr][kk] = __float2half(probability);
          tile_sum += probability;
        }
        row_max[qr] = new_max;
        row_sum[qr] = row_sum[qr] * alpha + tile_sum;
        row_alpha[qr] = alpha;
      }
    }
    __syncthreads();

    // Rescale the previous numerator before adding this tile's P@V.
    for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
      const int qr = idx / tile_d;
      acc_s[qr][idx % tile_d] *= row_alpha[qr];
    }
    __syncthreads();

    // PV: eight warps cover the eight 16-column tiles of the D=128 output.
    if (warp < tile_d / 16) {
      const int d0 = warp * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                     wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                     wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::load_matrix_sync(c_frag, &acc_s[0][d0], tile_d,
                              wmma::mem_row_major);
      for (int k1 = 0; k1 < k_tile; k1 += 16) {
        wmma::load_matrix_sync(a_frag, &p_s[0][k1], k_tile);
        wmma::load_matrix_sync(b_frag, &v_s[k1][d0], tile_d);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(&acc_s[0][d0], c_frag, tile_d,
                              wmma::mem_row_major);
    }
    __syncthreads();
  }

  for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
    const int qr = idx / tile_d;
    const int d = idx % tile_d;
    const int64_t qg = q0 + qr;
    if (qg < T && d < D && row_sum[qr] > 0.f) {
      out[bh_base + qg * D + d] = tp_half_from_cuda(
          __float2half(acc_s[qr][d] / row_sum[qr]));
    }
  }
}

// Q/K/V tiles on chip while the online softmax advances through the keys.
// The first WMMA prototype above used 512 threads for a 16-row tile; most of
// those warps were idle in each phase.  This variant follows the useful part
// 64x64 tile, Q is loaded once, and each warp carries two 16-column output
// fragments.  The Q/accumulator buffers are overlaid because Q is dead after
// the last PV iteration.
#if 1
struct TpWmmaFlashShared {
  __half q[64][128];
  __half k[64][128];
  __half v[64][128];
  float score[64][64];
  __half probability[64][64];
  float row_max[64];
  float row_sum[64];
  float row_alpha[64];
};

__global__ void sdpa_wmma_flash_half_4warp_kernel(
    const tensorplay::Half* __restrict__ q,
    const tensorplay::Half* __restrict__ k,
    const tensorplay::Half* __restrict__ v,
    tensorplay::Half* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale,
    bool is_causal) {
  using namespace nvcuda;
  constexpr int q_tile = 64;
  constexpr int k_tile = 64;
  constexpr int tile_d = 128;
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;
  constexpr float log2e = 1.4426950408889634f;

  extern __shared__ unsigned char smem_raw[];
  TpWmmaFlashShared& smem = *reinterpret_cast<TpWmmaFlashShared*>(smem_raw);
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t q0 = static_cast<int64_t>(blockIdx.y) * q_tile;
  const int64_t bh_base = bh * T * D;

  // Load Q once per query block.  The half representation is ABI-compatible
  // with CUDA's __half, so no conversion kernel or temporary tensor is
// needed on the hot path.
  for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
    const int qr = idx / tile_d;
    const int d = idx % tile_d;
    const int64_t qg = q0 + qr;
    smem.q[qr][d] =
        (qg < T && d < D)
        ? *reinterpret_cast<const __half*>(q + bh_base + qg * D + d)
        : __float2half(0.f);
  }
  if (thread < q_tile) {
    smem.row_max[thread] = -INFINITY;
    smem.row_sum[thread] = 0.f;
    smem.row_alpha[thread] = 0.f;
  }
  __syncthreads();

  // The largest query in this block is q0 + 63.  Causal attention therefore
  // never needs a key tile beginning after that row; the non-causal walk
  // covers the whole key axis.
  const int64_t last_k = is_causal ? min(T, q0 + q_tile) : T;

  using AccFragment =
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  // One warp owns a 16-column output strip and all four 16-row strips.  The
  // eight fragments fit in registers and are reused across key tiles.
  AccFragment accum[4][2];
  if (warp < warps) {
#pragma unroll
    for (int qr_tile = 0; qr_tile < 4; ++qr_tile) {
#pragma unroll
      for (int d_tile = 0; d_tile < 2; ++d_tile)
        wmma::fill_fragment(accum[qr_tile][d_tile], 0.f);
    }
  }
  bool first_tile = true;

  for (int64_t k0 = 0; k0 < last_k; k0 += k_tile) {
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      const bool valid = kg < T && d < D;
      smem.k[kr][d] = valid
          ? *reinterpret_cast<const __half*>(k + bh_base + kg * D + d)
          : __float2half(0.f);
      smem.v[kr][d] = valid
          ? *reinterpret_cast<const __half*>(v + bh_base + kg * D + d)
          : __float2half(0.f);
    }
    __syncthreads();

    // QK^T: warp w owns one 16-column key strip and walks the four query
    // strips.  This is the same 16x16 Tensor Core decomposition used by the
    if (warp < warps) {
      const int n0 = warp * 16;
#pragma unroll
      for (int m0 = 0; m0 < q_tile; m0 += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                       wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                       wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.f);
#pragma unroll
        for (int d0 = 0; d0 < tile_d; d0 += 16) {
          wmma::load_matrix_sync(a_frag, &smem.q[m0][d0], tile_d);
          wmma::load_matrix_sync(b_frag, &smem.k[n0][d0], tile_d);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(&smem.score[m0][n0], c_frag, k_tile,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Online softmax: one warp owns 16 rows and all lanes participate in the
    // max/sum reductions.  Unlike the prototype, exponentials and stores are
    // distributed across the warp rather than serialized in lane zero.
    if (warp < warps) {
      const int row_begin = warp * 16;
#pragma unroll 1
      for (int qr = row_begin; qr < row_begin + 16; ++qr) {
        const int64_t qg = q0 + qr;
        float maximum = -INFINITY;
        for (int kk = lane; kk < k_tile; kk += 32) {
          const int64_t kg = k0 + kk;
          float score = smem.score[qr][kk] * scale;
          if ((is_causal && kg > qg) || kg >= T || qg >= T)
            score = -INFINITY;
          maximum = max(maximum, score);
        }
        maximum = warpReduceMax(maximum);
        maximum = __shfl_sync(full_mask, maximum, 0);

        const float old_max = smem.row_max[qr];
        const float new_max = max(old_max, maximum);
        const float alpha = isfinite(old_max)
            ? exp2f((old_max - new_max) * log2e)
            : 0.f;
        float partial_sum = 0.f;
        for (int kk = lane; kk < k_tile; kk += 32) {
          const int64_t kg = k0 + kk;
          const float score = smem.score[qr][kk] * scale;
          const float p = ((is_causal && kg > qg) || kg >= T || qg >= T)
              ? 0.f
              : exp2f((score - new_max) * log2e);
          smem.probability[qr][kk] = __float2half(p);
          partial_sum += p;
        }
        partial_sum = warpReduceSum(partial_sum);
        partial_sum = __shfl_sync(full_mask, partial_sum, 0);
        if (lane == 0) {
          smem.row_max[qr] = new_max;
          smem.row_sum[qr] = smem.row_sum[qr] * alpha + partial_sum;
          smem.row_alpha[qr] = alpha;
        }
      }
    }
    __syncthreads();

    if (!first_tile && warp < warps) {
#pragma unroll
      for (int qr_tile = 0; qr_tile < 4; ++qr_tile) {
#pragma unroll
        for (int d_tile = 0; d_tile < 2; ++d_tile) {
          auto& c = accum[qr_tile][d_tile];
#pragma unroll
          for (int i = 0; i < c.num_elements; ++i) {
#if defined(USE_ROCM)
            const int row = (i & 7) + 8 * (lane >= 16);
#else
            const int row = (lane >> 2) + ((i & 2) ? 8 : 0);
#endif
            c.x[i] *= smem.row_alpha[qr_tile * 16 + row];
          }
        }
      }
    }

    // P@V.  Each warp covers two adjacent 16-column output tiles and all
    // four 16-row strips.  Rescaling happens in registers before the current
    // probability tile is accumulated.
    if (warp < warps) {
      const int d0 = warp * 32;
#pragma unroll
      for (int qr_tile = 0; qr_tile < 4; ++qr_tile) {
        const int qr0 = qr_tile * 16;
        const float alpha = smem.row_alpha[qr0];
        const float alpha1 = smem.row_alpha[qr0 + 1];
        const float alpha2 = smem.row_alpha[qr0 + 2];
        const float alpha3 = smem.row_alpha[qr0 + 3];
        // The four 16-row fragments each own one row group.  All rows in a
        // WMMA fragment do not share a single scalar, so rescale from shared
        // state after the matrix multiply instead of attempting a fragment-
        // wide multiply here.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                       wmma::row_major> p_frag[4][2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                       wmma::row_major> v_frag[4][2];
        (void)alpha;
        (void)alpha1;
        (void)alpha2;
        (void)alpha3;
        // Keep the accumulator fragments in the same warp-local mapping and
        // use the row scalars during the epilogue; WMMA's fragment layout is
        // deliberately opaque, so this avoids an invalid lane mapping.
#pragma unroll
        for (int d_tile = 0; d_tile < 2; ++d_tile) {
          wmma::fragment<wmma::accumulator, 16, 16, 16, float>& c =
              accum[qr_tile][d_tile];
#pragma unroll
          for (int k1 = 0; k1 < k_tile; k1 += 16) {
            wmma::load_matrix_sync(
                p_frag[qr_tile][d_tile], &smem.probability[qr0][k1], k_tile);
            wmma::load_matrix_sync(
                v_frag[qr_tile][d_tile], &smem.v[k1][d0 + d_tile * 16],
                tile_d);
            wmma::mma_sync(c, p_frag[qr_tile][d_tile],
                           v_frag[qr_tile][d_tile], c);
          }
        }
      }
    }
    __syncthreads();
    first_tile = false;
  }

  // Normalize and convert directly from each register fragment.
  if (warp < warps) {
    const int d0 = warp * 32;
#pragma unroll
    for (int qr_tile = 0; qr_tile < 4; ++qr_tile) {
      const int qr0 = qr_tile * 16;
#pragma unroll
      for (int d_tile = 0; d_tile < 2; ++d_tile) {
        auto& c = accum[qr_tile][d_tile];
#pragma unroll
        for (int i = 0; i < c.num_elements; ++i) {
#if defined(USE_ROCM)
          // RDNA3 WMMA accumulator order: element i of lane l covers local
          // row (i & 7) + 8 * (l >= 16) and local column l & 15.
          const int local_row = (i & 7) + 8 * (lane >= 16);
          const int local_col = lane & 15;
#else
          const int local_row = (lane >> 2) + ((i & 2) ? 8 : 0);
          const int local_col = ((lane & 3) * 2) + (i & 1) +
                                ((i >= 4) ? 8 : 0);
#endif
          const int qr = qr0 + local_row;
          const int d = d0 + d_tile * 16 + local_col;
          const float denom = smem.row_sum[qr];
          out[bh_base + (q0 + qr) * D + d] = tp_half_from_cuda(
              __float2half(denom > 0.f ? c.x[i] / denom : 0.f));
        }
      }
    }
  }
}
#endif

// Aligned native flash path for the benchmark's Llama head shape.  This is
// symbols are part of the dependency graph.
struct TpWmmaFlashAlignedShared {
  // Keep the aligned Q/V tiles on chip for the whole Q block.  Q is reused for
  // every K tile; K remains a direct aligned read because the compiler can use
  // the read-only cache without paying a shared-memory round trip.
  __half q[64][128];
  __half v[64][128];
  // score/P have matching row strides and safely share the other 16-KiB slot.
  union {
    float score[64][64];
    // Keep P's physical row stride equal to the float score row stride.  Only
    // the first 64 columns are used, but the padding prevents a half write in
    // one row from aliasing the score of an adjacent row during conversion.
    __half probability[64][128];
  } transient;
  float row_max[64];
  float row_sum[64];
  float row_alpha[64];
};

// Epilogue helper for the aligned kernel: normalizes one 16x16 accumulator
// fragment in registers and writes the FP16 result straight to the output.
// The fragment element -> (local row, local column) mapping differs between
// the two toolchains; both mappings are noted inline.
template <typename FragT>
__device__ inline void tp_wmma_store_norm(
    FragT& c, int qr0, int d0, int lane, const float* row_sum,
    __half* out_half, int64_t bh_base, int q0, int64_t D) {
  for (int i = 0; i < c.num_elements; ++i) {
#if defined(USE_ROCM)
    // RDNA3 WMMA accumulator order: element i of lane l covers local row
    // (i & 7) + 8 * (l >= 16) and local column l & 15.
    const int local_row = (i & 7) + 8 * (lane >= 16);
    const int local_col = lane & 15;
#else
    const int local_row = (lane >> 2) + ((i & 2) ? 8 : 0);
    const int local_col = ((lane & 3) * 2) + (i & 1) + ((i >= 4) ? 8 : 0);
#endif
    const int qr = qr0 + local_row;
    const int d = d0 + local_col;
    const float denom = row_sum[qr];
    out_half[bh_base + (q0 + qr) * D + d] =
        __float2half(denom > 0.f ? c.x[i] / denom : 0.f);
  }
}

__global__ __launch_bounds__(256, 2) void sdpa_wmma_flash_half_aligned_kernel(
    const tensorplay::Half* __restrict__ q,
    const tensorplay::Half* __restrict__ k,
    const tensorplay::Half* __restrict__ v,
    tensorplay::Half* __restrict__ out,
    int64_t B, int64_t H, int64_t T, int64_t D, float scale,
    bool is_causal) {
  using namespace nvcuda;
  constexpr int q_tile = 64;
  constexpr int k_tile = 64;
  constexpr int tile_d = 128;
  constexpr int warps = 8;
  constexpr int threads = warps * 32;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;
  constexpr float log2e = 1.4426950408889634f;

  extern __shared__ unsigned char smem_raw[];
  auto& smem = *reinterpret_cast<TpWmmaFlashAlignedShared*>(smem_raw);
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t q0 = static_cast<int64_t>(blockIdx.y) * q_tile;
  const int64_t bh_base = bh * T * D;
  __half* out_half = reinterpret_cast<__half*>(out);

  // Q is invariant across the online-softmax loop, so load it once.  The
  // benchmark uses T%64==0 here; keep the bounds checks for the public impl.
  for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
    const int qr = idx / tile_d;
    const int d = idx % tile_d;
    const int64_t qg = q0 + qr;
    smem.q[qr][d] =
        (qg < T && d < D)
        ? *reinterpret_cast<const __half*>(q + bh_base + qg * D + d)
        : __float2half(0.f);
  }
  if (thread < q_tile) {
    smem.row_max[thread] = -INFINITY;
    smem.row_sum[thread] = 0.f;
    smem.row_alpha[thread] = 0.f;
  }
  __syncthreads();

  using AccFragment =
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  // Keep each accumulator as a named fragment.  WMMA fragments are compiler
  // managed register objects; on SM89, indexing an array of them through a
  // loop can make the compiler assign an incomplete register tuple to some
  // rows.  The four explicit objects also make the epilogue's ownership
  // visible to the compiler without changing the tile schedule.
  AccFragment accum0;
  AccFragment accum1;
  AccFragment accum2;
  AccFragment accum3;
  bool first_tile = true;

  const int64_t last_k = is_causal ? min(T, q0 + q_tile) : T;
  for (int64_t k0 = 0; k0 < last_k; k0 += k_tile) {
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      smem.v[kr][d] = *reinterpret_cast<const __half*>(
          v + bh_base + kg * D + d);
    }
    __syncthreads();

    // QK^T.  A warp owns one 16-column key strip and two 16-row strips.  Both
    // operands are aligned row-major tensors; WMMA's column-major B view
    // consumes the K rows as the transposed operand.
    if (warp < warps) {
      const int n0 = (warp >> 1) * 16;
      const int m_begin = (warp & 1) * 16;
#pragma unroll
      for (int m0 = m_begin; m0 < q_tile; m0 += 32) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                       wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                       wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
#pragma unroll
        for (int d0 = 0; d0 < tile_d; d0 += 16) {
        wmma::load_matrix_sync(
              a, &smem.q[m0][d0], tile_d);
          wmma::load_matrix_sync(
              b,
              reinterpret_cast<const __half*>(
                  k + bh_base + (k0 + n0) * D + d0),
              D);
          wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(&smem.transient.score[m0][n0], c, k_tile,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Online softmax over this key tile.  All lanes participate in both
    // reductions; lane zero only publishes the three scalar row states.
    if (warp < warps) {
      const int row_begin = warp * 8;
#pragma unroll 1
      for (int qr = row_begin; qr < row_begin + 8; ++qr) {
        const int64_t qg = q0 + qr;
        // Score and probability share storage with different element sizes.
        // Read both values owned by this lane before any lane writes the half
        // probability tile; this makes the aliasing safe and removes a second
        // shared-memory score pass.
        const int kk0 = lane;
        const int kk1 = lane + 32;
        float score0 = smem.transient.score[qr][kk0] * scale;
        float score1 = smem.transient.score[qr][kk1] * scale;
        if (is_causal && k0 + kk0 > qg) score0 = -INFINITY;
        if (is_causal && k0 + kk1 > qg) score1 = -INFINITY;
        float maximum = max(score0, score1);
        maximum = warpReduceMax(maximum);
        maximum = __shfl_sync(full_mask, maximum, 0);

        const float old_max = smem.row_max[qr];
        const float new_max = max(old_max, maximum);
        const float alpha = isfinite(old_max)
            ? exp2f((old_max - new_max) * log2e)
            : 0.f;
        const float p0 = isfinite(score0)
            ? exp2f((score0 - new_max) * log2e) : 0.f;
        const float p1 = isfinite(score1)
            ? exp2f((score1 - new_max) * log2e) : 0.f;
        float partial_sum = p0 + p1;
        partial_sum = warpReduceSum(partial_sum);
        partial_sum = __shfl_sync(full_mask, partial_sum, 0);
        if (lane == 0) {
          smem.row_max[qr] = new_max;
          smem.row_sum[qr] = smem.row_sum[qr] * alpha + partial_sum;
          smem.row_alpha[qr] = alpha;
        }
        smem.transient.probability[qr][kk0] = __float2half(p0);
        smem.transient.probability[qr][kk1] = __float2half(p1);
      }
    }
    __syncthreads();

    // Bring the previous numerator into the new max coordinate before the
    // Tensor Core PV update.  Each lane's accumulator fragment elements map
    // to local rows per the layout note at the epilogue below.  Keeping the
    // rescale in registers saves a shared-memory round trip per key tile.
    if (!first_tile) {
      if (warp < warps) {
#pragma unroll
        for (int i = 0; i < accum0.num_elements; ++i) {
#if defined(USE_ROCM)
          const int row = (i & 7) + 8 * (lane >= 16);
#else
          const int row = (lane >> 2) + ((i & 2) ? 8 : 0);
#endif
          accum0.x[i] *= smem.row_alpha[row];
          accum1.x[i] *= smem.row_alpha[16 + row];
          accum2.x[i] *= smem.row_alpha[32 + row];
          accum3.x[i] *= smem.row_alpha[48 + row];
        }
      }
    } else if (warp < warps) {
      wmma::fill_fragment(accum0, 0.f);
      wmma::fill_fragment(accum1, 0.f);
      wmma::fill_fragment(accum2, 0.f);
      wmma::fill_fragment(accum3, 0.f);
    }
    __syncthreads();

    // PV: one warp owns a 32-column strip (two 16-column WMMA tiles) and
    // walks the four query strips.  This exactly covers [64, 128] output.
    if (warp < warps) {
      const int d0 = warp * 16;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                     wmma::row_major> p_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                     wmma::row_major> v_frag;
#pragma unroll
      for (int k1 = 0; k1 < k_tile; k1 += 16) {
        // V is shared by all four query fragments in this warp.  Load it
        // once per K sub-tile instead of issuing the same shared-memory
        // matrix load once for every 16-row fragment.
        wmma::load_matrix_sync(v_frag, &smem.v[k1][d0], tile_d);
        wmma::load_matrix_sync(
            p_frag, &smem.transient.probability[0][k1], tile_d);
        wmma::mma_sync(accum0, p_frag, v_frag, accum0);
        wmma::load_matrix_sync(
            p_frag, &smem.transient.probability[16][k1], tile_d);
        wmma::mma_sync(accum1, p_frag, v_frag, accum1);
        wmma::load_matrix_sync(
            p_frag, &smem.transient.probability[32][k1], tile_d);
        wmma::mma_sync(accum2, p_frag, v_frag, accum2);
        wmma::load_matrix_sync(
            p_frag, &smem.transient.probability[48][k1], tile_d);
        wmma::mma_sync(accum3, p_frag, v_frag, accum3);
      }
    }
    __syncthreads();
    first_tile = false;
  }

    // The accumulator fragment's element order is stable for this fixed
    // 16x16x16 WMMA shape (per-toolchain mapping noted inline).  Normalize
    // in registers and write one FP16 value per lane; this removes the 32KB
    // float output staging tile and leaves the block below the shared-memory
    // occupancy cliff.
  if (warp < warps) {
    const int d0 = warp * 16;
    tp_wmma_store_norm(accum0, 0, d0, lane, smem.row_sum, out_half,
                       bh_base, q0, D);
    tp_wmma_store_norm(accum1, 16, d0, lane, smem.row_sum, out_half,
                       bh_base, q0, D);
    tp_wmma_store_norm(accum2, 32, d0, lane, smem.row_sum, out_half,
                       bh_base, q0, D);
    tp_wmma_store_norm(accum3, 48, d0, lane, smem.row_sum, out_half,
                       bh_base, q0, D);
  }
}

#if defined(TP_HAS_NATIVE_CUTE_FLASH)
// This is the exact native 64x64/4-warp schedule used by the aligned CUDA
// path.  The wrapper only translates TensorPlay's [B,H,T,D] strides into the
// boundary.
template <bool IsCausal, bool IsEvenMN>
__global__ void tp_native_flash_hdim128_fp16_kernel(
    const ::tensorplay_native_flash::Flash_fwd_params params) {
  // Flash_fwd_kernel_traits is intentionally a global layout type in the
  // standalone source; only the executable helpers live in FLASH_NAMESPACE.
  using Traits = Flash_fwd_kernel_traits<
      128, 64, 64, 4, false, false, cutlass::half_t>;
  ::tensorplay_native_flash::compute_attn<
      Traits, false, IsCausal, false, false, IsEvenMN, true, false, false>(
      params);
}

template <bool IsCausal>
Tensor sdpa_native_cute_flash(
    const Tensor& q, const Tensor& k, const Tensor& v,
    int64_t B, int64_t H, int64_t T, int64_t D) {
  using Traits = Flash_fwd_kernel_traits<
      128, 64, 64, 4, false, false, cutlass::half_t>;
  using Params = ::tensorplay_native_flash::Flash_fwd_params;

  Tensor out = Tensor::empty({B, H, T, D}, q.dtype(), q.device());
  // The standalone flash epilogue writes LSE even when the caller does not
  // expose it.  Keep it as a regular TensorPlay temporary rather than
  // changing the kernel's proven epilogue or adding a synchronization.
  Tensor lse = Tensor::empty({B, H, T}, DType::Float32, q.device());

  Params params{};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  // FlashAttention's logical coordinates are [batch, token, head, dim],
  // while TensorPlay exposes [batch, head, token, dim].  Use the tensor's
  // real strides for the first three coordinates; this lets the native
  // kernel consume a transposed V view without a staging copy.
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.q_row_stride = q.stride(2);
  params.k_row_stride = k.stride(2);
  params.v_row_stride = v.stride(2);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.h = static_cast<int>(H);
  params.h_k = static_cast<int>(H);
  params.h_h_k_ratio = 1;
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = out.stride(2);
  params.o_head_stride = out.stride(1);
  params.p_ptr = nullptr;
  params.softmax_lse_ptr = lse.data_ptr<float>();
  params.b = static_cast<int>(B);
  params.seqlen_q = static_cast<int>(T);
  params.seqlen_k = static_cast<int>(T);
  params.seqlen_knew = 0;
  params.d = static_cast<int>(D);
  params.seqlen_q_rounded = static_cast<int>((T + 127) / 128 * 128);
  params.seqlen_k_rounded = static_cast<int>((T + 127) / 128 * 128);
  params.d_rounded = static_cast<int>(D);
  params.rotary_dim = 0;
  params.total_q = static_cast<int>(T * B);
  params.scale_softmax = 1.f / sqrtf(static_cast<float>(D));
  params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
  // FlashAttention stores the keep probability (not the drop probability).
  // Keep these fields identical to its no-dropout launch contract even though
  // this translation unit instantiates Is_dropout=false.
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  // The causal mask implementation derives its upper bound from this field;
  // -1 means an all-masked first row, while 0 is the original causal contract.
  params.window_size_right = IsCausal ? 0 : -1;
  params.softcap = 0.f;
  params.rng_state = nullptr;
  params.is_bf16 = false;
  params.is_causal = IsCausal;
  params.is_seqlens_k_cumulative = false;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.alibi_slopes_ptr = nullptr;
  params.alibi_slopes_batch_stride = 0;
  params.unpadded_lse = false;
  params.seqlenq_ngroups_swapped = false;

  static bool shared_memory_configured = false;
  if (!shared_memory_configured) {
    TP_CUDA_CHECK(cudaFuncSetAttribute(
        tp_native_flash_hdim128_fp16_kernel<IsCausal, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    TP_CUDA_CHECK(cudaFuncSetAttribute(
        tp_native_flash_hdim128_fp16_kernel<IsCausal, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    shared_memory_configured = true;
  }

  dim3 grid((static_cast<unsigned>(T) + 63u) / 64u,
            static_cast<unsigned>(B), static_cast<unsigned>(H));
  if ((T & 63) == 0) {
    tp_native_flash_hdim128_fp16_kernel<IsCausal, true><<<
        grid, Traits::kNThreads, Traits::kSmemSize,
        getCurrentCUDAStream().stream()>>>(params);
  } else {
    tp_native_flash_hdim128_fp16_kernel<IsCausal, false><<<
        grid, Traits::kNThreads, Traits::kSmemSize,
        getCurrentCUDAStream().stream()>>>(params);
  }
  TP_CUDA_CHECK(cudaGetLastError());
  return out;
}
#endif

template <typename DT>
Tensor sdpa_gemm_native(
    const Tensor& q, const Tensor& k, const Tensor& v,
    int64_t B, int64_t H, int64_t T, int64_t D, bool is_causal) {
  const DType dtype = q.dtype();
  Tensor kt = Tensor::empty({B, H, D, T}, dtype, q.device());
  dim3 transpose_grid(
      static_cast<unsigned>((T + 31) / 32),
      static_cast<unsigned>((D + 31) / 32),
      static_cast<unsigned>(B * H));
  dim3 transpose_block(32, 8);
  sdpa_transpose_k_kernel<DT><<<
      transpose_grid, transpose_block, 0, getCurrentCUDAStream().stream()>>>(
      k.data_ptr<DT>(), kt.data_ptr<DT>(), T, D);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor scores = Tensor::empty({B, H, T, T}, dtype, q.device());
  Tensor q3 = q.reshape({B * H, T, D});
  Tensor kt3 = kt.reshape({B * H, D, T});
  Tensor scores3 = scores.reshape({B * H, T, T});
  const long long q_stride = T * D;
  const long long kt_stride = D * T;
  const long long score_stride = T * T;
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  gemm_strided_batched_3d(
      q3, kt3, scores3, B * H, T, T, D, q_stride, kt_stride, scale, 0.0);

  const int64_t softmax_rows = B * H * T;
  const unsigned softmax_blocks = static_cast<unsigned>(softmax_rows);
  sdpa_softmax_kernel<DT><<<
      softmax_blocks, 32, 0, getCurrentCUDAStream().stream()>>>(
      scores.data_ptr<DT>(), B * H, T, is_causal);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
  Tensor scores3_again = scores.reshape({B * H, T, T});
  Tensor v3 = v.reshape({B * H, T, D});
  Tensor out3 = out.reshape({B * H, T, D});
  gemm_strided_batched_3d(
      scores3_again, v3, out3, B * H, T, D, T, score_stride, q_stride,
      1.0, 0.0);
  return out;
}

// ---------------------------------------------------------------------------
// Reference backward for the flash/naive forward implementations.
//
// The forward kernel intentionally does not retain an O(T^2) probability
// matrix.  Backward reconstructs the probabilities once and then uses three
// streaming matrix-vector kernels.  This keeps the autograd path correct for
// training while bounding workspace to three fp32 [B,H,T,T] buffers rather
// than retaining every forward intermediate.
// ---------------------------------------------------------------------------

template <typename DT>
__global__ void sdpa_backward_probs_kernel(
    const DT* __restrict__ q, const DT* __restrict__ k,
    float* __restrict__ probs, int64_t rows, int64_t T, int64_t D,
    float scale, bool is_causal) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T;
  if (row >= total) return;
  const int64_t bh = row / T;
  const int64_t t = row % T;
  const DT* q_row = q + (bh * T + t) * D;
  const DT* k_base = k + bh * T * D;
  float max_score = -INFINITY;
  for (int64_t kk = 0; kk < T; ++kk) {
    float score = -INFINITY;
    if (!is_causal || kk <= t) {
      score = 0.f;
      const DT* k_row = k_base + kk * D;
      for (int64_t d = 0; d < D; ++d) score += to_float(q_row[d]) * to_float(k_row[d]);
      score *= scale;
    }
    probs[(bh * T + t) * T + kk] = score;
    max_score = max(max_score, score);
  }
  float total_exp = 0.f;
  for (int64_t kk = 0; kk < T; ++kk) {
    float score = probs[(bh * T + t) * T + kk];
    float p = isfinite(score) ? expf(score - max_score) : 0.f;
    probs[(bh * T + t) * T + kk] = p;
    total_exp += p;
  }
  for (int64_t kk = 0; kk < T; ++kk)
    probs[(bh * T + t) * T + kk] /= total_exp;
}

template <typename DT>
__global__ void sdpa_backward_dprob_kernel(
    const DT* __restrict__ grad, const DT* __restrict__ value,
    float* __restrict__ dprob, int64_t rows, int64_t T, int64_t D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * T;
  if (idx >= total) return;
  int64_t tmp = idx;
  const int64_t kk = tmp % T; tmp /= T;
  const int64_t t = tmp % T; tmp /= T;
  const int64_t bh = tmp;
  const DT* g_row = grad + (bh * T + t) * D;
  const DT* v_row = value + (bh * T + kk) * D;
  float dot = 0.f;
  for (int64_t d = 0; d < D; ++d) dot += to_float(g_row[d]) * to_float(v_row[d]);
  dprob[idx] = dot;
}

__global__ void sdpa_backward_dscore_kernel(
    const float* __restrict__ probs, const float* __restrict__ dprob,
    float* __restrict__ dscore, int64_t rows, int64_t T, float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * T;
  if (idx >= total) return;
  int64_t tmp = idx;
  const int64_t kk = tmp % T; tmp /= T;
  const int64_t t = tmp % T; tmp /= T;
  const int64_t bh = tmp;
  const float* p_row = probs + (bh * T + t) * T;
  const float* dp_row = dprob + (bh * T + t) * T;
  float row_dot = 0.f;
  for (int64_t j = 0; j < T; ++j) row_dot += p_row[j] * dp_row[j];
  dscore[idx] = p_row[kk] * (dp_row[kk] - row_dot) * scale;
}

template <typename DT>
__global__ void sdpa_backward_dq_kernel(
    const float* __restrict__ dscore, const DT* __restrict__ key,
    DT* __restrict__ grad_q, int64_t rows, int64_t T, int64_t D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * D;
  if (idx >= total) return;
  int64_t tmp = idx;
  const int64_t d = tmp % D; tmp /= D;
  const int64_t t = tmp % T; tmp /= T;
  const int64_t bh = tmp;
  float acc = 0.f;
  const DT* k_base = key + bh * T * D;
  for (int64_t kk = 0; kk < T; ++kk)
    acc += dscore[(bh * T + t) * T + kk] * to_float(k_base[kk * D + d]);
  grad_q[idx] = from_float<DT>(acc);
}

template <typename DT>
__global__ void sdpa_backward_dk_kernel(
    const float* __restrict__ dscore, const DT* __restrict__ query,
    DT* __restrict__ grad_k, int64_t rows, int64_t T, int64_t D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * D;
  if (idx >= total) return;
  int64_t tmp = idx;
  const int64_t d = tmp % D; tmp /= D;
  const int64_t kk = tmp % T; tmp /= T;
  const int64_t bh = tmp;
  float acc = 0.f;
  const DT* q_base = query + bh * T * D;
  for (int64_t t = 0; t < T; ++t)
    acc += dscore[(bh * T + t) * T + kk] * to_float(q_base[t * D + d]);
  grad_k[idx] = from_float<DT>(acc);
}

template <typename DT>
__global__ void sdpa_backward_dv_kernel(
    const float* __restrict__ probs, const DT* __restrict__ grad,
    DT* __restrict__ grad_v, int64_t rows, int64_t T, int64_t D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * D;
  if (idx >= total) return;
  int64_t tmp = idx;
  const int64_t d = tmp % D; tmp /= D;
  const int64_t kk = tmp % T; tmp /= T;
  const int64_t bh = tmp;
  float acc = 0.f;
  const DT* g_base = grad + bh * T * D;
  for (int64_t t = 0; t < T; ++t)
    acc += probs[(bh * T + t) * T + kk] * to_float(g_base[t * D + d]);
  grad_v[idx] = from_float<DT>(acc);
}

template <typename DT>
std::tuple<Tensor, Tensor, Tensor> sdpa_backward_impl(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, int64_t impl) {
  (void)impl;
  Tensor q = query.contiguous();
  Tensor k = key.contiguous();
  Tensor v = value.contiguous();
  Tensor go = grad_output.contiguous();
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  const int64_t rows = B * H;
  Tensor probs = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  Tensor dprob = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  Tensor dscore = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  Tensor d_q = Tensor::empty({B, H, T, D}, q.dtype(), q.device());
  Tensor d_k = Tensor::empty({B, H, T, D}, q.dtype(), q.device());
  Tensor d_v = Tensor::empty({B, H, T, D}, q.dtype(), q.device());
  constexpr int threads = 256;
  auto blocks = [&](int64_t n) { return static_cast<unsigned>((n + threads - 1) / threads); };
  cudaStream_t stream = getCurrentCUDAStream().stream();
  sdpa_backward_probs_kernel<DT><<<blocks(rows * T), threads, 0, stream>>>(
      q.data_ptr<DT>(), k.data_ptr<DT>(), probs.data_ptr<float>(), rows, T, D,
      1.f / sqrtf(static_cast<float>(D)), is_causal);
  sdpa_backward_dprob_kernel<DT><<<blocks(rows * T * T), threads, 0, stream>>>(
      go.data_ptr<DT>(), v.data_ptr<DT>(), dprob.data_ptr<float>(), rows, T, D);
  sdpa_backward_dscore_kernel<<<blocks(rows * T * T), threads, 0, stream>>>(
      probs.data_ptr<float>(), dprob.data_ptr<float>(), dscore.data_ptr<float>(),
      rows, T, 1.f / sqrtf(static_cast<float>(D)));
  sdpa_backward_dq_kernel<DT><<<blocks(rows * T * D), threads, 0, stream>>>(
      dscore.data_ptr<float>(), k.data_ptr<DT>(), d_q.data_ptr<DT>(), rows, T, D);
  sdpa_backward_dk_kernel<DT><<<blocks(rows * T * D), threads, 0, stream>>>(
      dscore.data_ptr<float>(), q.data_ptr<DT>(), d_k.data_ptr<DT>(), rows, T, D);
  sdpa_backward_dv_kernel<DT><<<blocks(rows * T * D), threads, 0, stream>>>(
      probs.data_ptr<float>(), go.data_ptr<DT>(), d_v.data_ptr<DT>(), rows, T, D);
  TP_CUDA_CHECK(cudaGetLastError());
  return {d_q, d_k, d_v};
}

std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cuda(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, int64_t impl) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 || grad_output.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output must be 4D");
  }
  if (query.size(0) != key.size(0) || query.size(1) != key.size(1) ||
      query.size(2) != key.size(2) || query.size(3) != key.size(3) ||
      static_cast<std::vector<int64_t>>(key.shape()) != static_cast<std::vector<int64_t>>(value.shape()) ||
      static_cast<std::vector<int64_t>>(query.shape()) != static_cast<std::vector<int64_t>>(grad_output.shape())) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output shapes must match");
  }
  if (query.dtype() != key.dtype() || query.dtype() != value.dtype() ||
      query.dtype() != grad_output.dtype()) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output dtypes must match");
  }
  if (query.dtype() == DType::Float32) return sdpa_backward_impl<float>(grad_output, query, key, value, is_causal, impl);
  if (query.dtype() == DType::Float16) return sdpa_backward_impl<tensorplay::Half>(grad_output, query, key, value, is_causal, impl);
  if (query.dtype() == DType::BFloat16) return sdpa_backward_impl<tensorplay::BFloat16>(grad_output, query, key, value, is_causal, impl);
  TP_THROW(NotImplementedError, "sdpa backward: only float32/float16/bfloat16 supported");
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

Tensor sdpa_kernel_cuda(const Tensor& query, const Tensor& key, const Tensor& value, bool is_causal, int64_t impl) {
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
  const bool native_flash = impl == 5 || impl == 6 || impl == 7;
#else
  constexpr bool native_flash = false;
#endif
  // The standalone CUTE/CUTLASS kernel supports arbitrary strides in its
  // batch/head/token coordinates, but its innermost dimension is vectorized.
  // Preserve a compatible view and only materialize inputs whose D dimension
  // is not contiguous.  GEMM and the legacy kernels retain their canonical
  // contiguous-input contract below.
  auto flash_input = [](const Tensor& input) {
    return input.dim() == 4 && input.stride(3) == 1
        ? input : input.contiguous();
  };
  Tensor q = native_flash ? flash_input(query) : query.contiguous();
  Tensor k = native_flash ? flash_input(key) : key.contiguous();
  Tensor v = native_flash ? flash_input(value) : value.contiguous();
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa: query/key/value must be 4D [B, H, T, D]");
  }
  int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  if (k.size(0) != B || k.size(1) != H || v.size(0) != B || v.size(1) != H) {
    TP_THROW(RuntimeError, "sdpa: batch/head dims must match across q/k/v");
  }
  if (k.size(2) != v.size(2) || k.size(3) != D || v.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa: key/value shapes must match [B, H, T, D]");
  }
  DType dtype = q.dtype();
  if (dtype != k.dtype() || dtype != v.dtype()) {
    TP_THROW(RuntimeError, "sdpa: q/k/v dtypes must match");
  }
  if (dtype != DType::Float32 && dtype != DType::Float16 && dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError, "sdpa: only float32/float16/bfloat16 supported");
  }
  float scale = 1.f / sqrtf((float)D);

  constexpr int kThreads = 256;

  // Default (impl=0) routing: the warp-per-row flash kernel covers every
  // supported dtype at head_dim <= 128 and avoids the naive kernel's
  // float32 upcast entirely; the naive row-per-block kernel stays as the
  // fallback for wider heads.
  if (impl == 0 && D <= 128) {
    impl = 3;
  }

  if (impl == 0) {
    Tensor out;
    if (dtype != DType::Float32) {
      q = q.to(DType::Float32);
      k = k.to(DType::Float32);
      v = v.to(DType::Float32);
      out = Tensor::empty({B, H, T, D}, DType::Float32, q.device());
    }
    if (T > 4096) {
      TP_THROW(NotImplementedError, "sdpa impl=0 (naive) supports T <= 4096; use impl=1");
    }
    if (dtype == DType::Float32) {
      out = Tensor::empty({B, H, T, D}, dtype, q.device());
    }
    size_t smem = (T + kThreads) * sizeof(float);
    dim3 grid((unsigned)(B * H), (unsigned)T);
    sdpa_naive_kernel<<<grid, kThreads, smem, getCurrentCUDAStream().stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
        B, H, T, D, scale, is_causal);
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  } else if (impl == 1) {
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    if (D > 128) {
      TP_THROW(NotImplementedError, "sdpa impl=1 (flash) supports D <= 128; use impl=0");
    }
    // smem: Bq*128 + Bq*Br + 4*Bq + Bq*128 + Bq*8 floats
    size_t smem = (16 * 128 + 16 * 16 + 4 * 16 + 16 * 128 + 16 * 8) * sizeof(float);
    dim3 grid((unsigned)(B * H), (unsigned)((T + 15) / 16));
    if (dtype == DType::Float32) {
      sdpa_flash_kernel<float><<<grid, 128, smem, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
          B, H, T, D, scale, is_causal);
    } else if (dtype == DType::Float16) {
      sdpa_flash_kernel<tensorplay::Half><<<grid, 128, smem, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
          v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
          B, H, T, D, scale, is_causal);
    } else {
      sdpa_flash_kernel<tensorplay::BFloat16><<<grid, 128, smem, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::BFloat16>(), k.data_ptr<tensorplay::BFloat16>(),
          v.data_ptr<tensorplay::BFloat16>(), out.data_ptr<tensorplay::BFloat16>(),
          B, H, T, D, scale, is_causal);
    }
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  } else if (impl == 3) {
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    if (D > 128) {
      TP_THROW(NotImplementedError, "sdpa impl=3 (warp flash) supports D <= 128");
    }
    constexpr int q_rows_per_block = 4;
    dim3 grid((unsigned)(B * H),
              (unsigned)((T + q_rows_per_block - 1) / q_rows_per_block));
    if (dtype == DType::Float32) {
      sdpa_warp_flash_kernel<float><<<grid, 128, 0, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
          out.data_ptr<float>(), B, H, T, D, scale, is_causal);
    } else if (dtype == DType::Float16) {
      sdpa_warp_flash_kernel<tensorplay::Half><<<grid, 128, 0, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
          v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
          B, H, T, D, scale, is_causal);
    } else {
      sdpa_warp_flash_kernel<tensorplay::BFloat16><<<grid, 128, 0, getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::BFloat16>(), k.data_ptr<tensorplay::BFloat16>(),
          v.data_ptr<tensorplay::BFloat16>(), out.data_ptr<tensorplay::BFloat16>(),
          B, H, T, D, scale, is_causal);
    }
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  } else if (impl == 8) {
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    if (dtype != DType::Float16 || D != 128) {
      TP_THROW(NotImplementedError,
               "sdpa impl=8 (4-warp FP16 WMMA flash) requires dtype=float16 and D=128");
    }
    if ((T & 63) != 0) {
      constexpr int q_tile = 16;
      dim3 grid((unsigned)(B * H),
                (unsigned)((T + q_tile - 1) / q_tile));
      sdpa_wmma_flash_half_kernel<<<grid, 512, 0,
                                    getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
          v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
          B, H, T, D, scale, is_causal);
      TP_CUDA_CHECK(cudaGetLastError());
      return out;
    }
    static bool shared_memory_configured = false;
    if (!shared_memory_configured) {
      TP_CUDA_CHECK(cudaFuncSetAttribute(
#if defined(USE_ROCM)
          reinterpret_cast<const void*>(&sdpa_wmma_flash_half_4warp_kernel),
#else
          sdpa_wmma_flash_half_4warp_kernel,
#endif
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(TpWmmaFlashShared))));
      shared_memory_configured = true;
    }
    constexpr int q_tile = 64;
    dim3 grid((unsigned)(B * H),
              (unsigned)((T + q_tile - 1) / q_tile));
    sdpa_wmma_flash_half_4warp_kernel<<<
        grid, 128, sizeof(TpWmmaFlashShared),
        getCurrentCUDAStream().stream()>>>(
        q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
        v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
        B, H, T, D, scale, is_causal);
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  } else if (impl == 5 || impl == 6 || impl == 7) {
    if (dtype != DType::Float16 || D != 128) {
      TP_THROW(NotImplementedError,
               "sdpa impl=5 (aligned FP16 WMMA flash) requires dtype=float16 and D=128");
    }
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
    // Use the native CUTE/CUTLASS aligned path for both full and tail tiles;
    // its internal predicate schedule is also needed by autoregressive decode
    // once T grows past a multiple of 64.
    if (is_causal) {
      return sdpa_native_cute_flash<true>(q, k, v, B, H, T, D);
    }
    return sdpa_native_cute_flash<false>(q, k, v, B, H, T, D);
#else
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    // The native 64x64 schedule deliberately requires the original aligned
    // Llama shape.  Keep the older tail-safe kernel for arbitrary lengths.
    if ((T & 63) != 0) {
      constexpr int q_tile = 16;
      dim3 grid((unsigned)(B * H),
                (unsigned)((T + q_tile - 1) / q_tile));
      sdpa_wmma_flash_half_kernel<<<grid, 512, 0,
                                    getCurrentCUDAStream().stream()>>>(
          q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
          v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
          B, H, T, D, scale, is_causal);
      TP_CUDA_CHECK(cudaGetLastError());
      return out;
    }
    static bool shared_memory_configured = false;
    if (!shared_memory_configured) {
      TP_CUDA_CHECK(cudaFuncSetAttribute(
#if defined(USE_ROCM)
          reinterpret_cast<const void*>(&sdpa_wmma_flash_half_aligned_kernel),
#else
          sdpa_wmma_flash_half_aligned_kernel,
#endif
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(TpWmmaFlashAlignedShared))));
      shared_memory_configured = true;
    }
    constexpr int q_tile = 64;
    dim3 grid((unsigned)(B * H),
              (unsigned)((T + q_tile - 1) / q_tile));
    sdpa_wmma_flash_half_aligned_kernel<<<
        grid, 256, sizeof(TpWmmaFlashAlignedShared),
        getCurrentCUDAStream().stream()>>>(
        q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
        v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
        B, H, T, D, scale, is_causal);
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
#endif  // TP_HAS_NATIVE_CUTE_FLASH
  } else if (impl == 4) {
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    if (dtype != DType::Float16 || D != 128) {
      TP_THROW(NotImplementedError,
               "sdpa impl=4 (FP16 WMMA flash) requires dtype=float16 and D=128");
    }
    constexpr int q_tile = 16;
    dim3 grid((unsigned)(B * H),
              (unsigned)((T + q_tile - 1) / q_tile));
    sdpa_wmma_flash_half_kernel<<<grid, 512, 0,
                                  getCurrentCUDAStream().stream()>>>(
        q.data_ptr<tensorplay::Half>(), k.data_ptr<tensorplay::Half>(),
        v.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
        B, H, T, D, scale, is_causal);
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  } else if (impl == 2) {
    if (dtype == DType::Float32) {
      return sdpa_gemm_native<float>(q, k, v, B, H, T, D, is_causal);
    } else if (dtype == DType::Float16) {
      return sdpa_gemm_native<tensorplay::Half>(
          q, k, v, B, H, T, D, is_causal);
    } else {
      return sdpa_gemm_native<tensorplay::BFloat16>(
          q, k, v, B, H, T, D, is_causal);
    }
  } else {
    TP_THROW(RuntimeError, "sdpa: unknown impl " + std::to_string(impl));
  }
}

// Dispatcher-level primitives for the MoE grouped-GEMM composite (defined in
// TPXOpsGenerated.cpp; declared locally because tpx headers are not visible
// below the p10 layer -- same pattern as Einsum.cpp).
}  // namespace (anonymous kernels end here)

// Reopen at global scope so the declarations land in the REAL
// tensorplay::tpx::ops (defined in TPXOpsGenerated.cpp); declaring them
// under tensorplay::cuda::tensorplay would shadow the namespace and break
// both lookup and linkage.
}  // namespace cuda
}  // namespace tensorplay

// Dispatcher-level entry points (tensorplay::tpx::ops) are visible through
// AttentionComposite.h -> TPXOpsGenerated.h; no local re-declarations.

namespace tensorplay {
namespace cuda {

// nvcc mis-resolves the unqualified `tensorplay` in TP_THROW after the
// namespace reopen above (it synthesizes a phantom
// tensorplay::cuda::tensorplay member), so pin the throw target explicitly.
#if defined(TP_THROW)
#undef TP_THROW
#endif
#define TP_THROW(err_type, ...)                                               \
    throw ::tensorplay::err_type(                                             \
        {__FILE__, __func__, __LINE__},                                       \
        ::tensorplay::detail::format_msg(__VA_ARGS__))

// flash-attention's rotary Q/K handling), so keep these kernels beside SDPA
// and grouped attention rather than under an LLM-specific translation unit.
namespace {

inline bool rope_float_dtype(DType dtype) {
  return dtype == DType::Float32 || dtype == DType::Float64 ||
         dtype == DType::Float16 || dtype == DType::BFloat16;
}

inline int64_t rope_tokens(const Tensor& input, const char* op) {
  if (input.dim() < 2) {
    TP_THROW(RuntimeError, op, ": input must have at least 2 dimensions");
  }
  if ((input.size(-1) & 1) != 0) {
    TP_THROW(RuntimeError, op, ": the last dimension must be even");
  }
  if (!rope_float_dtype(input.dtype())) {
    TP_THROW(NotImplementedError, op,
             ": only float32/float64/float16/bfloat16 are supported");
  }
  return input.size(-2);
}

struct RopeTable {
  int64_t rows;
  int64_t half_dim;
};

RopeTable check_rope_table(const Tensor& cos, const Tensor& sin,
                           int64_t half_dim, int64_t tokens,
                           int64_t position_offset, const Device& device,
                           const char* op) {
  if (position_offset < 0) {
    TP_THROW(RuntimeError, op, ": position_offset must be non-negative");
  }
  if (cos.device() != device || sin.device() != device ||
      cos.device() != sin.device()) {
    TP_THROW(DeviceMismatchError, op,
             ": input and cos/sin must be on the same device");
  }
  if (cos.dtype() != sin.dtype() || !rope_float_dtype(cos.dtype())) {
    TP_THROW(RuntimeError, op,
             ": cos and sin must have the same floating dtype");
  }

  int64_t rows = 0;
  if (cos.dim() == 1 && sin.dim() == 1) {
    if (cos.size(0) != half_dim || sin.size(0) != half_dim) {
      TP_THROW(RuntimeError, op,
               ": 1D cos/sin tables must have head_dim/2 entries");
    }
    rows = 1;
  } else if (cos.dim() == 2 && sin.dim() == 2) {
    if (cos.size(1) != half_dim || sin.size(1) != half_dim ||
        cos.size(0) != sin.size(0)) {
      TP_THROW(RuntimeError, op,
               ": cos/sin tables must be [positions, head_dim/2]");
    }
    rows = cos.size(0);
  } else {
    TP_THROW(RuntimeError, op,
             ": cos and sin must both be 1D or both be 2D");
  }
  if (rows != 1 &&
      (position_offset > rows || tokens > rows - position_offset)) {
    TP_THROW(RuntimeError, op,
             ": cos/sin table is shorter than the requested positions");
  }
  if (rows == 1 && position_offset != 0) {
    TP_THROW(RuntimeError, op,
             ": position_offset must be zero for a one-row table");
  }
  return {rows, half_dim};
}

template <typename T, typename C, typename Acc>
__device__ inline void rotate_pair(const T* input, T* output, const C* cos,
                                   const C* sin, int64_t input_offset,
                                   int64_t table_offset) {
  const Acc x0 = static_cast<Acc>(input[input_offset]);
  const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
  const Acc c = static_cast<Acc>(cos[table_offset]);
  const Acc s = static_cast<Acc>(sin[table_offset]);
  output[input_offset] = static_cast<T>(x0 * c - x1 * s);
  output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
}

template <typename T, typename C, typename Acc>
__global__ void rotary_embedding_kernel(
    const T* input, T* output, const C* cos, const C* sin, int64_t pairs,
    int64_t tokens, int64_t half_dim, int64_t table_rows,
    int64_t position_offset) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t pair = first; pair < pairs; pair += stride) {
    const int64_t row = pair / half_dim;
    const int64_t pair_in_row = pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    rotate_pair<T, C, Acc>(
        input, output, cos, sin, row * (2 * half_dim) + 2 * pair_in_row,
        table_row * half_dim + pair_in_row);
  }
}

template <typename T, typename C, typename Acc>
__global__ void fused_rope_kernel(
    const T* query, T* query_out, const T* key, T* key_out,
    const C* cos, const C* sin, int64_t query_pairs, int64_t key_pairs,
    int64_t query_tokens, int64_t key_tokens, int64_t half_dim,
    int64_t table_rows, int64_t position_offset) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t total_pairs = query_pairs + key_pairs;
  for (int64_t pair = first; pair < total_pairs; pair += stride) {
    const T* input = query;
    T* output = query_out;
    int64_t local_pair = pair;
    int64_t tokens = query_tokens;
    if (pair >= query_pairs) {
      local_pair -= query_pairs;
      input = key;
      output = key_out;
      tokens = key_tokens;
    }
    const int64_t row = local_pair / half_dim;
    const int64_t pair_in_row = local_pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    rotate_pair<T, C, Acc>(
        input, output, cos, sin, row * (2 * half_dim) + 2 * pair_in_row,
        table_row * half_dim + pair_in_row);
  }
}

// The transformer path presents Q/K as a [B,H,T,D] view over a contiguous
// [B,T,H,D] linear result.  Its innermost D dimension is still contiguous, so
// RoPE can read that view directly instead of first materializing Q and K.
// Keep the generic flat kernel above for other ranks/layouts.
template <typename T, typename C, typename Acc>
__global__ void fused_rope_4d_strided_kernel(
    const T* query, T* query_out, const T* key, T* key_out,
    const C* cos, const C* sin, int64_t query_pairs, int64_t key_pairs,
    int64_t query_tokens, int64_t key_tokens, int64_t half_dim,
    int64_t table_rows, int64_t position_offset,
    int64_t query_batch_stride, int64_t query_head_stride,
    int64_t query_token_stride, int64_t query_out_batch_stride,
    int64_t query_out_head_stride, int64_t query_out_token_stride,
    int64_t query_heads, int64_t key_batch_stride, int64_t key_head_stride,
    int64_t key_token_stride, int64_t key_out_batch_stride,
    int64_t key_out_head_stride, int64_t key_out_token_stride,
    int64_t key_heads) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t total_pairs = query_pairs + key_pairs;
  for (int64_t pair = first; pair < total_pairs; pair += stride) {
    const T* input = query;
    T* output = query_out;
    int64_t local_pair = pair;
    int64_t tokens = query_tokens;
    int64_t batch_stride = query_batch_stride;
    int64_t head_stride = query_head_stride;
    int64_t token_stride = query_token_stride;
    int64_t out_batch_stride = query_out_batch_stride;
    int64_t out_head_stride = query_out_head_stride;
    int64_t out_token_stride = query_out_token_stride;
    int64_t heads = query_heads;
    if (pair >= query_pairs) {
      local_pair -= query_pairs;
      input = key;
      output = key_out;
      tokens = key_tokens;
      batch_stride = key_batch_stride;
      head_stride = key_head_stride;
      token_stride = key_token_stride;
      out_batch_stride = key_out_batch_stride;
      out_head_stride = key_out_head_stride;
      out_token_stride = key_out_token_stride;
      heads = key_heads;
    }
    const int64_t row = local_pair / half_dim;
    const int64_t pair_in_row = local_pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t outer = row / tokens;
    const int64_t batch = outer / heads;
    const int64_t head = outer - batch * heads;
    const int64_t input_offset = batch * batch_stride +
                                 head * head_stride +
                                 token * token_stride +
                                 2 * pair_in_row;
    const int64_t output_offset = batch * out_batch_stride +
                                  head * out_head_stride +
                                  token * out_token_stride +
                                  2 * pair_in_row;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    const Acc x0 = static_cast<Acc>(input[input_offset]);
    const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
    const Acc c = static_cast<Acc>(cos[table_row * half_dim + pair_in_row]);
    const Acc s = static_cast<Acc>(sin[table_row * half_dim + pair_in_row]);
    output[output_offset] = static_cast<T>(x0 * c - x1 * s);
    output[output_offset + 1] = static_cast<T>(x0 * s + x1 * c);
  }
}

inline unsigned rope_blocks(int64_t n) {
  constexpr int64_t kThreads = 256;
  return static_cast<unsigned>(std::min<int64_t>(
      (n + kThreads - 1) / kThreads, 65535));
}

template <typename T, typename C>
Tensor rope_single_typed(const Tensor& input, const Tensor& cos,
                         const Tensor& sin, const RopeTable& table,
                         int64_t tokens, int64_t position_offset) {
  Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor output = Tensor::empty(
      static_cast<std::vector<int64_t>>(input_c.shape()), input_c.dtype(),
      input_c.device());
  if (output.numel() == 0) return output;
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  rotary_embedding_kernel<T, C, Acc><<<
      rope_blocks(input_c.numel() / 2), 256, 0,
      getCurrentCUDAStream().stream()>>>(
      input_c.data_ptr<T>(), output.data_ptr<T>(), cos_c.data_ptr<C>(),
      sin_c.data_ptr<C>(), input_c.numel() / 2, tokens, table.half_dim,
      table.rows, position_offset);
  TP_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename T, typename C>
std::tuple<Tensor, Tensor> rope_pair_typed(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  const bool strided_4d = query.dim() == 4 && key.dim() == 4 &&
                          query.stride(3) == 1 && key.stride(3) == 1;
  Tensor query_c = strided_4d || query.is_contiguous()
      ? query : query.contiguous();
  Tensor key_c = strided_4d || key.is_contiguous()
      ? key : key.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor query_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(query_c.shape()), query_c.dtype(),
      query_c.device());
  Tensor key_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(key_c.shape()), key_c.dtype(),
      key_c.device());
  const int64_t query_pairs = query_c.numel() / 2;
  const int64_t key_pairs = key_c.numel() / 2;
  if (query_pairs + key_pairs == 0) {
    return std::make_tuple(query_out, key_out);
  }
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  if (strided_4d) {
    fused_rope_4d_strided_kernel<T, C, Acc><<<
        rope_blocks(query_pairs + key_pairs), 256, 0,
        getCurrentCUDAStream().stream()>>>(
        query_c.data_ptr<T>(), query_out.data_ptr<T>(), key_c.data_ptr<T>(),
        key_out.data_ptr<T>(), cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
        query_pairs, key_pairs, query_tokens, key_tokens, table.half_dim,
        table.rows, position_offset,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        query_out.stride(0), query_out.stride(1), query_out.stride(2),
        query_c.size(1), key_c.stride(0), key_c.stride(1), key_c.stride(2),
        key_out.stride(0), key_out.stride(1), key_out.stride(2),
        key_c.size(1));
  } else {
    fused_rope_kernel<T, C, Acc><<<
        rope_blocks(query_pairs + key_pairs), 256, 0,
        getCurrentCUDAStream().stream()>>>(
        query_c.data_ptr<T>(), query_out.data_ptr<T>(), key_c.data_ptr<T>(),
        key_out.data_ptr<T>(), cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
        query_pairs, key_pairs, query_tokens, key_tokens, table.half_dim,
        table.rows, position_offset);
  }
  TP_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(query_out, key_out);
}

template <typename T>
Tensor rope_single_dispatch(const Tensor& input, const Tensor& cos,
                            const Tensor& sin, const RopeTable& table,
                            int64_t tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_single_typed<T, float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_typed<T, double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_typed<T, tensorplay::Half>(
          input, cos, sin, table, tokens, position_offset);
    case DType::BFloat16:
      return rope_single_typed<T, tensorplay::BFloat16>(
          input, cos, sin, table, tokens, position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported table dtype");
  }
}

template <typename T>
std::tuple<Tensor, Tensor> rope_pair_dispatch(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_pair_typed<T, float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_typed<T, double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_typed<T, tensorplay::Half>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    case DType::BFloat16:
      return rope_pair_typed<T, tensorplay::BFloat16>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported table dtype");
  }
}

} // namespace

Tensor rotary_embedding_cuda(const Tensor& input, const Tensor& cos,
                             const Tensor& sin, int64_t position_offset) {
  const int64_t tokens = rope_tokens(input, "rotary_embedding");
  const RopeTable table = check_rope_table(
      cos, sin, input.size(-1) / 2, tokens, position_offset, input.device(),
      "rotary_embedding");
  switch (input.dtype()) {
    case DType::Float32:
      return rope_single_dispatch<float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_dispatch<double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_dispatch<tensorplay::Half>(
          input, cos, sin, table, tokens, position_offset);
    case DType::BFloat16:
      return rope_single_dispatch<tensorplay::BFloat16>(
          input, cos, sin, table, tokens, position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported input dtype");
  }
}

std::tuple<Tensor, Tensor> fused_rope_cuda(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, int64_t position_offset) {
  const int64_t query_tokens = rope_tokens(query, "fused_rope");
  const int64_t key_tokens = rope_tokens(key, "fused_rope");
  if (query.device() != key.device()) {
    TP_THROW(DeviceMismatchError,
             "fused_rope: query and key must be on the same device");
  }
  if (query.dtype() != key.dtype()) {
    TP_THROW(RuntimeError, "fused_rope: query and key must have the same dtype");
  }
  if (query.dim() != key.dim() || query.size(-1) != key.size(-1) ||
      query_tokens != key_tokens) {
    TP_THROW(RuntimeError,
             "fused_rope: query/key must have the same rank, token length, and head dimension");
  }
  const RopeTable table = check_rope_table(
      cos, sin, query.size(-1) / 2, query_tokens, position_offset,
      query.device(), "fused_rope");
  switch (query.dtype()) {
    case DType::Float32:
      return rope_pair_dispatch<float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_dispatch<double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_dispatch<tensorplay::Half>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    case DType::BFloat16:
      return rope_pair_dispatch<tensorplay::BFloat16>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported input dtype");
  }
}

// MoE expert compute on CUDA: ragged grouped GEMM.
//   - No-grad fast path: one C++ pass over groups calling the plan-cached
//     cublasLt entry (gemm_impl) on zero-copy slice views, writing straight
//     into preallocated output -- no per-group dispatcher round-trips, no
//     cat copy pass.  Zero-fill allocation covers uncovered tail rows,
//   - GradMode path: differentiable composite (narrow/mm/cat through the
//     dispatcher) so CIA records inner nodes automatically.
Tensor grouped_mm_cuda(const Tensor& self, const Tensor& mat2,
                       const Tensor& offs) {
  if (self.dim() != 2 || mat2.dim() != 3) {
    TP_THROW(RuntimeError,
             "grouped_mm(): expected 2D self and 3D mat2, got ", self.dim(),
             "D and ", mat2.dim(), "D");
  }
  const int64_t M = self.size(0), K = self.size(1);
  const int64_t G = mat2.size(0);
  const int64_t N = mat2.size(2);
  if (mat2.size(1) != K) {
    TP_THROW(RuntimeError,
             "grouped_mm(): self.size(1) must match mat2.size(1): ", K,
             " vs ", mat2.size(1));
  }
  if (offs.dim() != 1 || offs.numel() != G) {
    TP_THROW(RuntimeError,
             "grouped_mm(): offs must be 1D of length mat2.size(0)=", G,
             ", got ", offs.dim(), "D/", offs.numel(), " elements");
  }
  if (offs.dtype() != DType::Int32 && offs.dtype() != DType::Int64) {
    TP_THROW(TypeError, "grouped_mm(): offs must be int32 or int64");
  }

  // Loop bounds live on the host: read directly when offs is CPU-resident
  // (the common construction site), D2H copy when it lives on the device.
  const int64_t nbytes = G * (offs.dtype() == DType::Int32 ? 4 : 8);
  std::vector<unsigned char> hoff(static_cast<size_t>(nbytes));
  if (offs.device().type() == DeviceType::CUDA) {
    cudaMemcpy(hoff.data(), offs.data_ptr(), nbytes, cudaMemcpyDeviceToHost);
  } else {
    std::memcpy(hoff.data(), offs.data_ptr(), static_cast<size_t>(nbytes));
  }
  auto read_off = [&](int64_t i) -> int64_t {
    return offs.dtype() == DType::Int32
               ? static_cast<int64_t>(
                     reinterpret_cast<const int32_t*>(hoff.data())[i])
               : reinterpret_cast<const int64_t*>(hoff.data())[i];
  };
  int64_t covered = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    if (end < covered || end > M) {
      TP_THROW(RuntimeError,
               "grouped_mm(): offs must be non-decreasing in [0, M_total=", M,
               "], got offs[", g, "]=", end);
    }
    covered = end;
  }

  const bool needs_grad =
      GradMode::is_enabled() && (self.requires_grad() || mat2.requires_grad());
  if (needs_grad) {
    // Differentiable composite: dispatcher primitives record inner nodes.
    std::vector<Tensor> parts;
    parts.reserve(G + 1);
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int64_t len = end - start;
      if (len > 0) {
        Tensor wg = tpx::ops::narrow(mat2, 0, g, 1);
        wg = tpx::ops::reshape(wg, {K, N});
        parts.push_back(
            tpx::ops::mm(tpx::ops::narrow(self, 0, start, len), wg));
      }
      start = end;
    }
    if (start < M) {
      parts.push_back(tpx::ops::zeros({M - start, N}, self.dtype(),
                                      self.device(), false, false));
    }
    if (parts.empty()) {
      return tpx::ops::zeros({M, N}, self.dtype(), self.device(), false,
                             false);
    }
    return tpx::ops::cat(parts, 0);
  }

  // Fast path: zero-filled output + per-group cublasLt into slice views.
  Tensor out = Tensor::empty({M, N}, self.dtype(), self.device());
  zero_matmul_output_cuda(out);
  int64_t start = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    const int64_t len = end - start;
    if (len > 0) {
      Tensor aslice = self.slice(0, start, end);
      Tensor bg = mat2.select(0, g);          // [K, N], zero-copy
      Tensor oslice = out.slice(0, start, end);
      gemm_impl(aslice, bg, oslice, 1.0, 0.0, nullptr);
    }
    start = end;
  }
  return out;
}

std::tuple<Tensor, Tensor> sdpa_math_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale, bool enable_gqa) {
  return tensorplay::composite::sdpa_math_composite(
      query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale,
      enable_gqa);
}

std::tuple<Tensor, Tensor> native_multi_head_attention_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight,
    const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias,
    const std::optional<Tensor>& mask, bool need_weights,
    bool average_attn_weights, std::optional<int64_t> mask_type) {
  return tensorplay::composite::native_mha_composite(
      query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
      proj_weight, proj_bias, mask, need_weights, average_attn_weights,
      mask_type);
}

int64_t fused_sdp_choice_cuda(const Tensor& query, const Tensor& key,
                              const Tensor& value,
                              const std::optional<Tensor>& attn_mask,
                              double dropout_p, bool is_causal,
                              std::optional<double> scale, bool enable_gqa) {
  (void)key; (void)value; (void)is_causal; (void)scale;
  return tensorplay::composite::fused_sdp_choice_common(query, attn_mask,
                                                        dropout_p, enable_gqa);
}


TENSORPLAY_LIBRARY_IMPL(CUDA, AttentionKernels) {
  m.impl("scaled_dot_product_attention", sdpa_kernel_cuda);
  m.impl("scaled_dot_product_attention_backward", sdpa_backward_kernel_cuda);
  m.impl("_scaled_dot_product_attention_math", sdpa_math_cuda);
  m.impl("_native_multi_head_attention", native_multi_head_attention_cuda);
  m.impl("_fused_sdp_choice", fused_sdp_choice_cuda);
  m.impl("grouped_mm", grouped_mm_cuda);
  m.impl("rotary_embedding", rotary_embedding_cuda);
  m.impl("fused_rope", fused_rope_cuda);
}

} // namespace cuda
} // namespace tensorplay
