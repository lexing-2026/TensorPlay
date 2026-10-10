#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CudaGemm.h"
#include "Exception.h"
#include "Allocator.h"
#include "GradMode.h"
#include "AttentionUtils.cuh"
#include "../composite/AttentionComposite.h"
#include "../composite/AttentionPrivate.h"
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
//
// The kernel headers are the copy this tree keeps beside its launcher, not
// the ones the registered submodule carries.  That is the single definition
// of the launch parameters: the leaves under src/backend/cuda/transformers
// compile against this copy through FlashFwdLauncher.h, so a second copy
// reached from here would be a second definition of the same structure in the
// same namespace, and the two would agree only for as long as someone kept
// them edited together.
#if !defined(USE_ROCM) && \
    __has_include("flash/flash_fwd_kernel.h") && \
    __has_include("../../../../third_party/cutlass/include/cute/tensor.hpp")
#define TP_HAS_NATIVE_CUTE_FLASH 1
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASH_NAMESPACE tensorplay_native_flash
#include "flash/flash.h"
#include "flash/flash_fwd_kernel.h"
// The forward launcher picks the kernel by element type, head width and causal
// choice and is defined in a translation unit of its own, one leaf per file, so
// that no file holds every instantiation at once.  It is declared here, inside
// the namespace, because that is where the leaves it calls are declared.
namespace tensorplay_native_flash {
void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream,
                 bool force_split_kernel = false);
}  // namespace tensorplay_native_flash
#include "flash/flash_bwd_preprocess_kernel.h"
#include "flash/flash_bwd_kernel.h"
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

template <typename DT>
Tensor sdpa_gemm_native(
    const Tensor& q, const Tensor& k, const Tensor& v,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Tq, int64_t Tkv, int64_t D,
    bool is_causal);

// The fused wide-precision schedule, defined in a translation unit of its own
// because its tile shape and its register budget belong to it alone.  A wide
// precision otherwise reaches a schedule that writes its score matrix out and
// reads it back twice, which for a wide context costs more than the operands
// themselves.  The schedule is built from instructions the AMD toolchain does
// not have, so on that toolchain it is absent and both arms below fall back to
// the composite.
#if !defined(USE_ROCM)
Tensor sdpa_wide_tiled_cuda(const Tensor& query, const Tensor& key,
                            const Tensor& value, bool is_causal);
// The same schedule with the row log-sum-exp written out, for the fused entry
// that returns it alongside the attention.
Tensor sdpa_wide_tiled_cuda_with_lse(const Tensor& query, const Tensor& key,
                                   const Tensor& value, bool is_causal,
                                   Tensor& lse);
#endif

namespace {

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

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
// Fused short-sequence attention.
//
// One block owns one (batch, head) and keeps the whole score matrix in shared
// memory, so the probabilities are never written out: the query tile, the
// key tile and the value tile are staged, the scores are formed in place, the
// row reduction runs on that buffer, and the product with the value tile
// produces the output.  No part of a row is owned by a single lane, which is
// what lets a short sequence cost a few microseconds rather than a few hundred.
//
// Two choices keep the shared reads cheap.  A row of each staged tile is
// padded by one float, so a walk down the feature axis of a fixed row lands on
// consecutive banks instead of one bank stride D times over.  Each thread then
// holds a 4x4 patch of a product in registers, so an inner step costs eight
// shared reads per sixteen multiply-adds rather than one read per
// multiply-add.
//
// The host side admits a shape here only when the feature count is a multiple
// of four, which is what lets the staged moves and the stores travel four
// floats at a time, and only when the storage is a plain contiguous one, so the
// base address is the allocator's and therefore aligned well past what a
// four-float access needs.  Everything the gate declines keeps the kernel it
// had.
// ---------------------------------------------------------------------------

constexpr int kSdpaFusedMaxTokens = 64;
constexpr int kSdpaFusedMaxDim = 32;
constexpr int kSdpaFusedThreads = 128;
constexpr int kSdpaFusedTile = 4;
// The pad after each staged row.  It is one tile, not one float, so a row of a
// staged tile still begins on a sixteen-byte boundary -- which is what the
// four-float staged moves require -- while still moving the bank a walk down
// the feature axis of a fixed row off a single bank.  The host derives its
// shared-memory request from this same value; if the two ever disagree the
// kernel writes past the block's shared allocation, so there is exactly one
// definition of the pad and both sides name it.
constexpr int kSdpaFusedRowPad = kSdpaFusedTile;

template <typename DT>
__global__ void sdpa_fused_short_kernel(
    const DT* __restrict__ q, const DT* __restrict__ k,
    const DT* __restrict__ v, DT* __restrict__ out,
    float* __restrict__ lse_out, int64_t B, int64_t H, int64_t T, int64_t D,
    float scale, bool is_causal) {
  // The staged tiles are padded by one float along the feature axis and the
  // score buffer by one float along the key axis, for the same reason.  The
  // tile pad is four floats rather than one so a row of the tile still starts
  // on a sixteen-byte boundary, which is what the four-float staged moves need.
  const int dstride = static_cast<int>(D) + kSdpaFusedRowPad;
  const int tstride = static_cast<int>(T) + 1;
  extern __shared__ float smem[];
  float* s_q = smem;
  float* s_k = s_q + T * dstride;
  float* s_v = s_k + T * dstride;
  float* s_s = s_v + T * dstride;

  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t base = bh * T * D;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  // Stage the three input tiles.  Their element count is a multiple of four
  // because the feature count is, so a four-float move never straddles a row.
  const int64_t count = T * D;
  for (int64_t i = tid; i < count / 4; i += nthreads) {
    const int64_t row = i / (D / 4);
    const int64_t col = (i % (D / 4)) * 4;
    const float4 qv = *reinterpret_cast<const float4*>(q + base + i * 4);
    const float4 kv = *reinterpret_cast<const float4*>(k + base + i * 4);
    const float4 vv = *reinterpret_cast<const float4*>(v + base + i * 4);
    *reinterpret_cast<float4*>(s_q + row * dstride + col) = qv;
    *reinterpret_cast<float4*>(s_k + row * dstride + col) = kv;
    *reinterpret_cast<float4*>(s_v + row * dstride + col) = vv;
  }
  // The one-float pad after each row is never read, but leaving shared memory
  // undefined is a trap for whoever reads the tile next, so clear it.
  const int64_t padded = T * dstride;
  for (int64_t i = count + tid; i < padded; i += nthreads) {
    if (i % dstride >= D) {
      s_q[i] = 0.f;
      s_k[i] = 0.f;
      s_v[i] = 0.f;
    }
  }
  __syncthreads();
  // Form the scores.  A thread owns a 4x4 patch, and the patch coordinates
  // come from the flat patch index so every thread does the same work.
  const int tiles = (static_cast<int>(T) + kSdpaFusedTile - 1) / kSdpaFusedTile;
  const int ntiles = tiles * tiles;
  for (int tile = tid; tile < ntiles; tile += nthreads) {
    const int ti = tile / tiles;
    const int tj = tile - ti * tiles;
    const int i0 = ti * kSdpaFusedTile;
    const int j0 = tj * kSdpaFusedTile;
    float acc[kSdpaFusedTile][kSdpaFusedTile];
#pragma unroll
    for (int u = 0; u < kSdpaFusedTile; ++u)
#pragma unroll
      for (int w = 0; w < kSdpaFusedTile; ++w) acc[u][w] = 0.f;
    for (int64_t d = 0; d < D; ++d) {
      float a[kSdpaFusedTile];
      float b[kSdpaFusedTile];
#pragma unroll
      for (int u = 0; u < kSdpaFusedTile; ++u)
        a[u] = (i0 + u < T) ? s_q[(i0 + u) * dstride + d] : 0.f;
#pragma unroll
      for (int w = 0; w < kSdpaFusedTile; ++w)
        b[w] = (j0 + w < T) ? s_k[(j0 + w) * dstride + d] : 0.f;
#pragma unroll
      for (int u = 0; u < kSdpaFusedTile; ++u)
#pragma unroll
        for (int w = 0; w < kSdpaFusedTile; ++w) acc[u][w] += a[u] * b[w];
    }
#pragma unroll
    for (int u = 0; u < kSdpaFusedTile; ++u) {
      if (i0 + u >= T) continue;
      for (int w = 0; w < kSdpaFusedTile; ++w) {
        const int j = j0 + w;
        if (j >= T) continue;
        // A causal call has no score past the query's own position.  Row zero
        // always keeps its own key, so a row is never left without one and the
        // normaliser is never taken from an empty row.
        s_s[(i0 + u) * tstride + j] =
            (is_causal && j > i0 + u) ? -INFINITY : acc[u][w] * scale;
      }
    }
  }
  __syncthreads();

  // Row reduction over the staged scores.  One warp per row, so a row's
  // maximum and its total are both tree reductions across the warp.  The
  // shuffle reduction leaves the total in the first lane only, so it is passed
  // back out to the whole warp before anyone uses it; a block-wide reduction
  // cannot be used here, because warps on different rows reach it a different
  // number of times.
  const int warps = nthreads >> 5;
  for (int64_t row = tid >> 5; row < T; row += warps) {
    const int lane = tid & 31;
    float mx = -INFINITY;
    for (int j = lane; j < T; j += 32) mx = max(mx, s_s[row * tstride + j]);
    mx = __shfl_sync(0xffffffffffffffffull, warpReduceMax(mx), 0);
    float sum = 0.f;
    for (int j = lane; j < T; j += 32) {
      const float score = s_s[row * tstride + j];
      const float p = isfinite(score) ? expf(score - mx) : 0.f;
      s_s[row * tstride + j] = p;
      sum += p;
    }
    sum = __shfl_sync(0xffffffffffffffffull, warpReduceSum(sum), 0);
    const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (int j = lane; j < T; j += 32) s_s[row * tstride + j] *= inv;
    if (lane == 0 && lse_out != nullptr) lse_out[bh * T + row] = mx + logf(sum);
  }
  __syncthreads();

  // Product with the value tile, again a 4x4 patch per thread.
  const int dtile_n = (static_cast<int>(D) + kSdpaFusedTile - 1) / kSdpaFusedTile;
  const int dtiles = tiles * dtile_n;
  for (int tile = tid; tile < dtiles; tile += nthreads) {
    const int ti = tile / dtile_n;
    const int tj = tile - ti * dtile_n;
    const int i0 = ti * kSdpaFusedTile;
    const int d0 = tj * kSdpaFusedTile;
    float acc[kSdpaFusedTile][kSdpaFusedTile];
#pragma unroll
    for (int u = 0; u < kSdpaFusedTile; ++u)
#pragma unroll
      for (int w = 0; w < kSdpaFusedTile; ++w) acc[u][w] = 0.f;
    for (int64_t j = 0; j < T; ++j) {
      float p[kSdpaFusedTile];
      float w[kSdpaFusedTile];
#pragma unroll
      for (int u = 0; u < kSdpaFusedTile; ++u)
        p[u] = (i0 + u < T) ? s_s[(i0 + u) * tstride + j] : 0.f;
#pragma unroll
      for (int e = 0; e < kSdpaFusedTile; ++e)
        w[e] = (d0 + e < D) ? s_v[j * dstride + d0 + e] : 0.f;
#pragma unroll
      for (int u = 0; u < kSdpaFusedTile; ++u)
#pragma unroll
        for (int e = 0; e < kSdpaFusedTile; ++e) acc[u][e] += p[u] * w[e];
    }
#pragma unroll
    for (int u = 0; u < kSdpaFusedTile; ++u) {
      if (i0 + u >= T) continue;
#pragma unroll
      for (int e = 0; e < kSdpaFusedTile; ++e) {
        if (d0 + e >= D) continue;
        out[base + (i0 + u) * D + d0 + e] = from_float<DT>(acc[u][e]);
      }
    }
  }
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

// A compact FP16 Tensor Core attention path for the Llama head shape
// (D=128).  It follows the same fused QK -> online softmax -> PV structure
// TensorPlay tensor ABI and dispatcher independent.  One block owns 16 query
// rows of one head; four warps compute QK tiles and eight warps compute the
// PV output tiles.  The 16 remaining warps each own one online-softmax row.
//
// WMMA primitives exist only for compute capability 7.0 and newer CUDA
// targets; the toolkit omits the namespace from older device passes, so the
// kernels below are compiled only for the targets that expose them.  The HIP
// compatibility header defines the namespace for every target it serves.
#if defined(USE_ROCM) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
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
#endif  // WMMA kernels require compute capability 7.0 or newer.

#if defined(TP_HAS_NATIVE_CUTE_FLASH)
// What one launch needs beyond the three tensors: the two window bounds, the
// sequence table when the batch is packed, and whether the caller wants the
// normalizing constant handed back.
//
// The mask is not a field, because the reference does not carry one either: it
// is what the two bounds say.  Causal is the window whose right bound is zero
// with the left unbounded, and a window is anything else with a finite bound,
// and the launcher below reads it off the bounds the way the reference does.
struct SdpaFusedLaunch {
  int64_t window_left = -1;
  int64_t window_right = -1;
  // A non-null table means the sequences sit back to back with nothing marking
  // the joins, and the kernel finds each one's own extent from it.
  const int32_t* cu_seqlens_q = nullptr;
  const int32_t* cu_seqlens_k = nullptr;
  int64_t num_seqs = 1;
  int64_t max_seqlen_q = 0;
  int64_t max_seqlen_k = 0;
  // How many key slices to cut the call into, or zero to let the split choice
  // make it from the block count.  That is the same meaning the count carries
  // on the call this schedule's entry point takes.
  int num_splits = 0;
  Tensor* lse_out = nullptr;
};

// Fill the reference's forward parameters from this tree's tensors and launch.
//
// The split schedule: a block covers a slice of the key axis rather than all of
// it, which is what gives a short query against a long context enough blocks to
// fill the device.  It needs a second pass, because each slice's softmax saw
// only its own keys and so carries its own normalizer.
//
// The two pieces below are the reference's own split setup, transcribed because
// the reference keeps them beside its tensor entry points rather than in a
// header this layer reads: the choice of how many slices, and the buffers the
// slices write for the reduction to read.

// How many key slices to cut a call into.  Cutting costs a second pass over the
// slices and extra writes, so a call that already fills the device is cut once.
// Among the rest, the smallest cut that reaches 85% of the best occupancy the
// cut sizes allow: a cut that divides the key blocks unevenly leaves a slice
// short, and those sizes are skipped rather than scored.
inline int sdpa_num_splits(int batch_nheads_mblocks, int num_sms,
                           int num_n_blocks, int max_splits) {
  if (batch_nheads_mblocks >= 0.8f * num_sms) return 1;
  max_splits = std::min({max_splits, num_sms, num_n_blocks});
  float max_efficiency = 0.f;
  // The split count is bounded by 128, so the efficiency table rides the
  // stack; a heap allocation on every call is a cost a decode loop pays for
  // a table that fits in a few cache lines.
  float efficiency[128];
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int n) {
    return n == 1 || ceildiv(num_n_blocks, n) != ceildiv(num_n_blocks, n - 1);
  };
  for (int n = 1; n <= max_splits; n++) {
    if (!is_split_eligible(n)) {
      efficiency[n - 1] = 0.f;
      continue;
    }
    const float n_waves = float(batch_nheads_mblocks * n) / num_sms;
    const float eff = n_waves / std::ceil(n_waves);
    if (eff > max_efficiency) max_efficiency = eff;
    efficiency[n - 1] = eff;
  }
  for (int n = 1; n <= max_splits; n++) {
    if (!is_split_eligible(n)) continue;
    if (efficiency[n - 1] >= 0.85f * max_efficiency) return n;
  }
  return 1;
}

// The reference's split setup: the key block width here has to agree with the
// one the split dispatch tiles by, and the two names it is written under are the
// head width and the query's own length.  A caller that names no split count
// gets the chosen one.  Returns the buffers the slices write, which the
// reduction reads; they are handed back rather than kept local so that they
// outlive the launches.
struct SdpaSplitBuffers {
  Tensor lse_accum;
  Tensor out_accum;
};

SdpaSplitBuffers sdpa_set_params_splitkv(
    ::tensorplay_native_flash::Flash_fwd_params& params, int64_t batch_size,
    int64_t num_heads, int64_t head_size, int64_t max_seqlen_k,
    int64_t max_seqlen_q, int64_t head_size_rounded, int num_splits,
    const Device& device) {
  // The key tile the split dispatch uses, spelled the same way so the two
  // counts below are the same counts.  The wide precision halves it because
  // its operands take twice the shared memory of the reduced precisions'.
  const int block_n = params.is_fp32
      ? (head_size <= 64 ? 128 : 64)
      : (head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64));
  const int num_n_blocks = static_cast<int>((max_seqlen_k + block_n - 1) / block_n);
  const int num_m_blocks = static_cast<int>((max_seqlen_q + 64 - 1) / 64);
  params.num_splits = num_splits;
  SdpaSplitBuffers buffers;
  if (num_splits < 1) {
    // The processor count is a property of the device, not of the call, so it
    // is read once and kept; a hot decode loop would otherwise pay a runtime
    // query on every step for a number that never changes.
    static const int num_sms = []() {
      int count = 0;
      TP_CUDA_CHECK(cudaDeviceGetAttribute(
          &count, cudaDevAttrMultiProcessorCount,
          getCurrentCUDAStream().device_index()));
      return count;
    }();
    // Twice the processor count is the occupancy budget, because a block here is
    // 128 threads and two of them fit a processor.
    params.num_splits = sdpa_num_splits(
        static_cast<int>(batch_size * num_heads) * num_m_blocks, num_sms * 2,
        num_n_blocks, 128);
  }
  if (params.num_splits > 1) {
    buffers.lse_accum = Tensor::empty(
        {params.num_splits, batch_size, num_heads, max_seqlen_q},
        DType::Float32, device);
    buffers.out_accum = Tensor::empty(
        {params.num_splits, batch_size, num_heads, max_seqlen_q,
         head_size_rounded},
        DType::Float32, device);
    params.softmax_lseaccum_ptr = buffers.lse_accum.data_ptr<float>();
    params.oaccum_ptr = buffers.out_accum.data_ptr<float>();
  }
  if (params.num_splits > 128) {
    TP_THROW(RuntimeError, "num_splits > 128 not supported");
  }
  return buffers;
}

// Every field is set here under the reference's own name, so what is set and
// what the kernel reads are the same thing spelled once.  The three strides are
// in the reference's logical order -- batch, then token, then head -- which is
// not always the caller's own order: a packed tensor puts the token and head
// axes the other way round from a batched one, so they are taken from the
// layout at hand rather than assumed.
//
// A packed batch is not a separate call.  The kernels read the sequence table
// out of these parameters, so the same call serves both layouts and the grid
// is the one the reference computes: one block per query tile of the longest
// sequence, per sequence, per query head.
std::tuple<Tensor, Tensor> sdpa_fused_forward_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const SdpaFusedLaunch& launch, std::optional<double> scale, bool enable_gqa) {
  const bool packed = launch.cu_seqlens_q != nullptr;
  const int64_t rank = query.dim();
  TP_CHECK(packed ? rank == 3 : rank == 4, "sdpa fused: expected ",
           packed ? "a packed (total, heads, dim) query" : "a 4-axis query",
           ", got ", rank, " axes");
  // The schedule vectorises its innermost dimension, so a head axis that is
  // not the contiguous one is materialized; every other axis is addressed by
  // the caller's real strides and is consumed as given.
  auto head_contiguous = [](const Tensor& t) {
    return t.stride(-1) == 1 ? t : t.contiguous();
  };
  const Tensor q = head_contiguous(query);
  const Tensor k = head_contiguous(key);
  const Tensor v = head_contiguous(value);
  // A packed tensor is (total, heads, dim) and only gains a leading axis to be
  // read, so its heads sit one axis later than a batched tensor's.
  const Tensor q4 = packed ? q.unsqueeze(0) : q;
  const Tensor k4 = packed ? k.unsqueeze(0) : k;
  const Tensor v4 = packed ? v.unsqueeze(0) : v;
  const int64_t B = packed ? 1 : q4.size(0);
  const int64_t head_axis = packed ? 2 : 1;
  const int64_t row_axis = packed ? 1 : 2;
  const int64_t Hq = q4.size(head_axis);
  const int64_t Hkv = enable_gqa ? k4.size(head_axis) : Hq;
  const int64_t Tq = q4.size(row_axis);
  const int64_t Tkv = k4.size(row_axis);
  const int64_t D = q4.size(3);

  // A decode step is one query row against a long context, and a block per
  // query tile is then a block for the whole device.  Reading the group as the
  // row axis puts several query heads in one tile, which is what the reference
  // does for the same shape and under the same conditions: one query row, more
  // query heads than key heads, and no bound tying row i to key i.
  if (!packed && launch.window_left < 0 && launch.window_right < 0 &&
      enable_gqa && Tq == 1 && Hq > Hkv) {
    const int64_t group = Hq / Hkv;
    SdpaFusedLaunch flat = launch;
    Tensor q_grouped = tpx::ops::reshape(q4, {B, Hkv, group, D});
    Tensor grouped_lse;
    flat.lse_out = &grouped_lse;
    auto [out, lse] = sdpa_fused_forward_cuda(q_grouped, k4, v4, flat, scale,
                                              /*enable_gqa=*/false);
    // The result already lands on query head hk * group + s, so putting the
    // group back is a view.  The constant is indexed the same way, (hk, s) for
    // the one query row, so it reads back as (batch, query heads, 1).
    Tensor head_lse = tpx::ops::reshape(lse, {B, Hq, Tq});
    if (launch.lse_out) *launch.lse_out = head_lse;
    return {tpx::ops::reshape(out, {B, Hq, Tq, D}), head_lse};
  }

  // The output of a packed call is the whole run of tokens, not one such run
  // per sequence: the kernel places each sequence's tokens at its own offset
  // inside it, so an allocation sized per sequence would be as many times too
  // large and would leave the tail untouched.
  const int64_t grid_b = packed ? launch.num_seqs : B;
  // The output and the logsumexp are laid out the way the inputs are.  A
  // packed call keeps the tokens on the row axis and the heads on the head
  // axis; a batched call is the other way round, because the transpose above
  // put the heads second for it.  Writing either result in the batched order
  // would leave the kernel reading a row at one stride and writing it at
  // another, which fills the buffer with plausible numbers in the wrong slots.
  Tensor out = Tensor::empty(
      {packed ? 1 : B, packed ? Tq : Hq, packed ? Hq : Tq, D}, q.dtype(),
      q.device());
  Tensor lse = packed ? Tensor::empty({Hq, Tq}, DType::Float32, q.device())
                      : Tensor::empty({B, Hq, Tq}, DType::Float32, q.device());

  ::tensorplay_native_flash::Flash_fwd_params params{};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  params.softmax_lse_ptr = lse.data_ptr<float>();
  params.q_batch_stride = q4.stride(0);
  params.k_batch_stride = k4.stride(0);
  params.v_batch_stride = v4.stride(0);
  params.q_row_stride = q4.stride(row_axis);
  params.k_row_stride = k4.stride(row_axis);
  params.v_row_stride = v4.stride(row_axis);
  params.q_head_stride = q4.stride(head_axis);
  params.k_head_stride = k4.stride(head_axis);
  params.v_head_stride = v4.stride(head_axis);
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = out.stride(row_axis);
  params.o_head_stride = out.stride(head_axis);
  // Grouped-query heads: the kernel reads the key head as bidh / ratio, so one
  // key head serves a contiguous block of ratio query heads.
  params.h = static_cast<int>(Hq);
  params.h_k = static_cast<int>(Hkv);
  params.h_h_k_ratio = static_cast<int>(Hq / Hkv);
  params.b = static_cast<int>(grid_b);
  // A packed call reports the longest sequence here; each block reads its own
  // sequence's extent from the table, and these values only bound the grid.
  const int64_t seqlen_q = packed ? launch.max_seqlen_q : Tq;
  const int64_t seqlen_k = packed ? launch.max_seqlen_k : Tkv;
  params.seqlen_q = static_cast<int>(seqlen_q);
  params.seqlen_k = static_cast<int>(seqlen_k);
  params.seqlen_q_rounded = static_cast<int>((seqlen_q + 127) / 128 * 128);
  params.seqlen_k_rounded = static_cast<int>((seqlen_k + 127) / 128 * 128);
  params.seqlen_knew = 0;
  params.d = static_cast<int>(D);
  params.d_rounded = static_cast<int>(D);
  params.rotary_dim = 0;
  // A packed call arrives as one buffer of every token in the batch, so its
  // row extent is already the total and the sequence count does not multiply
  // into it again.  A batched call has one extent per batch entry, and there
  // the total is the product of the two.
  params.total_q = static_cast<int>(packed ? Tq : Tq * grid_b);
  params.cu_seqlens_q = const_cast<int32_t*>(launch.cu_seqlens_q);
  params.cu_seqlens_k = const_cast<int32_t*>(
      launch.cu_seqlens_k != nullptr ? launch.cu_seqlens_k : launch.cu_seqlens_q);
  // The tables hold running totals, so a block reads its own start and its own
  // extent by differencing two entries.
  params.is_seqlens_k_cumulative = packed;
  // Over a packed batch the constant has no batch axis to sit on: the heads and
  // the total token count are what index it, which is the layout the packed
  // callers declare.
  params.unpadded_lse = packed;
  params.seqlenq_ngroups_swapped = false;

  // Both window bounds are measured from the diagonal running from the top
  // left corner to the bottom right one, which the kernel applies itself from
  // the two lengths; what arrives here is how far either side reaches.  A
  // negative bound is unbounded, and the kernel is given a reach rather than a
  // sentinel: the reference rewrites such a bound to the key length, and this
  // is that rewrite.  Causal is the case the reference names by the bounds
  // alone -- right bound zero, left not -- and it is read that way here too.
  int64_t window_left = launch.window_left;
  int64_t window_right = launch.window_right;
  params.is_causal = window_left < 0 && window_right == 0;
  if (window_left < 0 && window_right >= 0) window_left = seqlen_k;
  if (window_left >= 0 && window_right < 0) window_right = seqlen_k;
  params.window_size_left = static_cast<int>(window_left);
  params.window_size_right = static_cast<int>(window_right);

  // The score normaliser is the caller's scale when one was given, and the
  // head width's reciprocal square root otherwise.
  const double s =
      scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(D));
  params.scale_softmax = static_cast<float>(s);
  params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
  // These carry the no-dropout launch contract; the schedule drops nothing.
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.softcap = 0.f;
  params.rng_state = nullptr;
  params.is_bf16 = q.dtype() == DType::BFloat16;
  params.is_fp32 = q.dtype() == DType::Float32;
  params.is_rotary_interleaved = false;
  params.alibi_slopes_ptr = nullptr;
  params.alibi_slopes_batch_stride = 0;
  params.leftpad_k = nullptr;
  params.seqused_k = nullptr;
  params.knew_ptr = nullptr;
  params.vnew_ptr = nullptr;
  params.cache_batch_idx = nullptr;
  params.block_table = nullptr;
  params.page_block_size = 0;
   // The block count is what decides the cut, not the shape: a call that already
   // fills the device is cut once, however long its context is.  A packed batch
   // is left alone because its extents come from a table rather than a length.
   // A causal call bounds the walk of the rows a split serves by the diagonal:
   // the rows a block owns admit no key past their own index, so the context
   // tiles past the last query row of the whole call are dead weight -- the
   // split count is capped by the blocks the diagonal actually covers, which
   // keeps a one-row decode step from paying for slices that read keys no row
   // can reach and a combine pass over slices that hold nothing.
   const int64_t split_seqlen_k =
       params.is_causal && !packed ? std::min(Tkv, Tq) : Tkv;
   const SdpaSplitBuffers split = packed
       ? SdpaSplitBuffers{}
       : sdpa_set_params_splitkv(params, grid_b, Hq, D, split_seqlen_k, Tq, D,
                                 launch.num_splits, q.device());

  ::tensorplay_native_flash::run_mha_fwd(
      params, getCurrentCUDAStream().stream(),
      /*force_split_kernel=*/params.num_splits > 1);
  TP_CUDA_CHECK(cudaGetLastError());
  if (launch.lse_out) *launch.lse_out = lse;
  // The schedule answers with the head on the second axis; each contract names
  // its own order and restates it.  A packed result is already the whole token
  // run, which is the order a packed contract names.
  return {packed ? out.reshape({Tq, Hq, D}) : out, lse};
}
#endif

// ---------------------------------------------------------------------------
// Reference backward for the flash/naive forward implementations.
//
// The forward kernel intentionally does not retain an O(T^2) probability
// matrix.  Backward reconstructs the probabilities once and then uses three
// streaming matrix-vector kernels.  This keeps the autograd path correct for
// training while bounding workspace to three fp32 [B,H,T,T] buffers rather
// than retaining every forward intermediate.
// ---------------------------------------------------------------------------

// The scores arrive scaled and unmasked, one query row per block; masking and
// normalizing them in place leaves the probability rows the gradient products
// read.
__global__ void sdpa_backward_softmax_rows_kernel(
    float* __restrict__ probs, int64_t query_rows, int64_t T, bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= query_rows) return;
  const int64_t t = row % T;
  float* p_row = probs + row * T;
  __shared__ float reduce_smem[32];

  float local_max = -INFINITY;
  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x) {
    float score = p_row[kk];
    if (is_causal && kk > t) {
      score = -INFINITY;
      p_row[kk] = score;
    }
    local_max = max(local_max, score);
  }
  const float row_max = blockReduceMax(local_max, reduce_smem);

  float local_sum = 0.f;
  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x) {
    const float score = p_row[kk];
    const float p = isfinite(score) ? expf(score - row_max) : 0.f;
    p_row[kk] = p;
    local_sum += p;
  }
  const float total_exp = blockReduceSum(local_sum, reduce_smem);
  const float inverse = total_exp > 0.f ? 1.f / total_exp : 0.f;

  for (int64_t kk = threadIdx.x; kk < T; kk += blockDim.x)
    p_row[kk] = p_row[kk] * inverse;
}

// The row statistic that the score gradient subtracts.  Written once per query
// row -- one value for every (batch, head, query) triple -- so the score
// gradient is a single pass that reads it, rather than a pass that re-reduces
// the whole row at every element of it.  A row is owned by one block, which is
// what lets the reduction over it be a block reduction.
__global__ void sdpa_backward_delta_kernel(
    const float* __restrict__ probs, const float* __restrict__ dprob,
    float* __restrict__ delta, int64_t num_query_rows, int64_t T) {
  const int64_t r = static_cast<int64_t>(blockIdx.x);
  if (r >= num_query_rows) return;
  const float* p_row = probs + r * T;
  const float* dp_row = dprob + r * T;
  float local = 0.f;
  for (int64_t j = threadIdx.x; j < T; j += blockDim.x)
    local += p_row[j] * dp_row[j];
  __shared__ float reduce_smem[32];
  delta[r] = blockReduceSum(local, reduce_smem);
}

// dS[t, j] = P[t, j] * (dP[t, j] - delta[t]), written over dP: each element is
// read and then written by the one thread that owns it, so the two pointers
// may name the same memory and neither is declared unaliased.
__global__ void sdpa_backward_dscore_kernel(
    const float* __restrict__ probs, const float* dprob,
    const float* __restrict__ delta, float* dscore, int64_t rows,
    int64_t T, float scale) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * T;
  if (idx >= total) return;
  const int64_t query_row = idx / T;
  dscore[idx] = probs[idx] * (dprob[idx] - delta[query_row]) * scale;
}

// The row statistic the score gradient subtracts, read off the forward's
// output: sum_j P[t, j] dP[t, j] = sum_j P[t, j] (dO[t] . V[j]) = dO[t] . O[t],
// a dot product over the head width instead of a pass over the score row.
// One warp owns one query row.
__global__ void sdpa_backward_delta_from_output_kernel(
    const float* __restrict__ grad_out, const float* __restrict__ out,
    float* __restrict__ delta, int64_t num_query_rows, int64_t D) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) +
                    threadIdx.x / 32;
  if (r >= num_query_rows) return;
  const int lane = threadIdx.x % 32;
  const float* go_row = grad_out + r * D;
  const float* o_row = out + r * D;
  float local = 0.f;
  for (int64_t d = lane; d < D; d += 32) local += go_row[d] * o_row[d];
  local = warpReduceSum(local);
  if (lane == 0) delta[r] = local;
}

// Probabilities and score gradient in one pass, from the scaled scores and
// the forward's log-sum-exp: P = exp(S - lse) and dS = P (dP - delta) scale.
// Each element is read and written by the one thread that owns it, so the
// results overwrite the scores and the probability gradient in place.  Keys
// past the query under the causal mask have no probability and no gradient.
__global__ void sdpa_backward_scores_from_lse_kernel(
    float* scores_to_probs, float* dprob_to_dscore,
    const float* __restrict__ lse, const float* __restrict__ delta,
    int64_t rows, int64_t T, float scale, bool is_causal) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * T * T;
  if (idx >= total) return;
  const int64_t query_row = idx / T;
  const int64_t key = idx - query_row * T;
  if (is_causal && key > query_row % T) {
    scores_to_probs[idx] = 0.f;
    dprob_to_dscore[idx] = 0.f;
    return;
  }
  const float p = expf(scores_to_probs[idx] - lse[query_row]);
  scores_to_probs[idx] = p;
  dprob_to_dscore[idx] = p * (dprob_to_dscore[idx] - delta[query_row]) * scale;
}

// Every product the backward needs is a batched matrix product over the
// (batch, head) pairs -- the scores Q K^T, the probability gradient dO V^T,
// and the three operand gradients P^T dO, dS K and dS^T Q -- so each is one
// strided-batched library call; only the row softmax, the row statistic and
// the score gradient are kernels of their own.  The two score-sized
// intermediates are materialized, as the forward of the same shapes does.  A
// reduced-precision operand is widened once on the way in, so the scores and
// every accumulation stay in float whatever the call's element type.
std::tuple<Tensor, Tensor, Tensor> sdpa_backward_impl(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal) {
  const DType dtype = query.dtype();
  const int64_t B = query.size(0), H = query.size(1);
  const int64_t T = query.size(2), D = query.size(3);
  const int64_t batch = B * H;
  const int64_t query_rows = batch * T;
  if (query_rows == 0 || D == 0) {
    return {Tensor::zeros({B, H, T, D}, dtype, query.device()),
            Tensor::zeros({B, H, T, D}, dtype, query.device()),
            Tensor::zeros({B, H, T, D}, dtype, query.device())};
  }
  auto packed = [&](const Tensor& t) {
    Tensor widened = t.dtype() == DType::Float32 ? t : t.to(DType::Float32);
    return widened.contiguous().reshape({batch, T, D});
  };
  const Tensor q = packed(query);
  const Tensor k = packed(key);
  const Tensor v = packed(value);
  const Tensor go = packed(grad_output);
  Tensor probs = Tensor::empty({batch, T, T}, DType::Float32, query.device());
  Tensor dprob = Tensor::empty({batch, T, T}, DType::Float32, query.device());
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  const long long td = static_cast<long long>(T) * D;
  const long long tt = static_cast<long long>(T) * T;
  cudaStream_t stream = getCurrentCUDAStream().stream();

  gemm_strided_batched_3d_op(q, k, probs, batch, T, T, D, td, td,
                             /*transpose_b=*/true, scale, 0.0);
  sdpa_backward_softmax_rows_kernel<<<
      static_cast<unsigned>(query_rows), 128, 0, stream>>>(
      probs.data_ptr<float>(), query_rows, T, is_causal);
  gemm_strided_batched_3d_op(go, v, dprob, batch, T, T, D, td, td,
                             /*transpose_b=*/true, 1.0, 0.0);
  Tensor delta = Tensor::empty({query_rows}, DType::Float32, query.device());
  sdpa_backward_delta_kernel<<<
      static_cast<unsigned>(query_rows), 128, 0, stream>>>(
      probs.data_ptr<float>(), dprob.data_ptr<float>(),
      delta.data_ptr<float>(), query_rows, T);
  constexpr int threads = 256;
  const int64_t score_elems = batch * tt;
  sdpa_backward_dscore_kernel<<<
      static_cast<unsigned>((score_elems + threads - 1) / threads), threads, 0,
      stream>>>(
      probs.data_ptr<float>(), dprob.data_ptr<float>(), delta.data_ptr<float>(),
      dprob.data_ptr<float>(), batch, T, scale);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor d_q = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  Tensor d_k = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  Tensor d_v = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  gemm_strided_batched_3d_ops(probs, go, d_v, batch, T, D, T, tt, td,
                              /*transpose_a=*/true, /*transpose_b=*/false,
                              1.0, 0.0);
  gemm_strided_batched_3d(dprob, k, d_q, batch, T, D, T, tt, td, 1.0, 0.0);
  gemm_strided_batched_3d_ops(dprob, q, d_k, batch, T, D, T, tt, td,
                              /*transpose_a=*/true, /*transpose_b=*/false,
                              1.0, 0.0);
  auto restored = [&](const Tensor& t) {
    Tensor shaped = t.reshape({B, H, T, D});
    return dtype == DType::Float32 ? shaped : shaped.to(dtype);
  };
  return {restored(d_q), restored(d_k), restored(d_v)};
}

// The same products when the forward kept its output and each row's
// log-sum-exp: the probabilities are one exponential per score and the row
// statistic is read off the output, so the score rows are never normalized
// again -- one pass over the two score-sized buffers instead of three.
std::tuple<Tensor, Tensor, Tensor> sdpa_backward_from_lse(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    bool is_causal) {
  const DType dtype = query.dtype();
  const int64_t B = query.size(0), H = query.size(1);
  const int64_t T = query.size(2), D = query.size(3);
  const int64_t batch = B * H;
  const int64_t query_rows = batch * T;
  auto packed = [&](const Tensor& t) {
    Tensor widened = t.dtype() == DType::Float32 ? t : t.to(DType::Float32);
    return widened.contiguous().reshape({batch, T, D});
  };
  const Tensor q = packed(query);
  const Tensor k = packed(key);
  const Tensor v = packed(value);
  const Tensor go = packed(grad_output);
  const Tensor o = packed(output);
  Tensor probs = Tensor::empty({batch, T, T}, DType::Float32, query.device());
  Tensor dprob = Tensor::empty({batch, T, T}, DType::Float32, query.device());
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  const long long td = static_cast<long long>(T) * D;
  const long long tt = static_cast<long long>(T) * T;
  cudaStream_t stream = getCurrentCUDAStream().stream();

  Tensor delta = Tensor::empty({query_rows}, DType::Float32, query.device());
  constexpr int rows_per_block = 4;
  sdpa_backward_delta_from_output_kernel<<<
      static_cast<unsigned>((query_rows + rows_per_block - 1) / rows_per_block),
      32 * rows_per_block, 0, stream>>>(
      go.data_ptr<float>(), o.data_ptr<float>(), delta.data_ptr<float>(),
      query_rows, D);
  gemm_strided_batched_3d_op(q, k, probs, batch, T, T, D, td, td,
                             /*transpose_b=*/true, scale, 0.0);
  gemm_strided_batched_3d_op(go, v, dprob, batch, T, T, D, td, td,
                             /*transpose_b=*/true, 1.0, 0.0);
  constexpr int threads = 256;
  const int64_t score_elems = batch * tt;
  sdpa_backward_scores_from_lse_kernel<<<
      static_cast<unsigned>((score_elems + threads - 1) / threads), threads, 0,
      stream>>>(
      probs.data_ptr<float>(), dprob.data_ptr<float>(),
      logsumexp.data_ptr<float>(), delta.data_ptr<float>(), batch, T, scale,
      is_causal);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor d_q = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  Tensor d_k = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  Tensor d_v = Tensor::empty({batch, T, D}, DType::Float32, query.device());
  gemm_strided_batched_3d_ops(probs, go, d_v, batch, T, D, T, tt, td,
                              /*transpose_a=*/true, /*transpose_b=*/false,
                              1.0, 0.0);
  gemm_strided_batched_3d(dprob, k, d_q, batch, T, D, T, tt, td, 1.0, 0.0);
  gemm_strided_batched_3d_ops(dprob, q, d_k, batch, T, D, T, tt, td,
                              /*transpose_a=*/true, /*transpose_b=*/false,
                              1.0, 0.0);
  auto restored = [&](const Tensor& t) {
    Tensor shaped = t.reshape({B, H, T, D});
    return dtype == DType::Float32 ? shaped : shaped.to(dtype);
  };
  return {restored(d_q), restored(d_k), restored(d_v)};
}

std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cuda(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale, bool enable_gqa) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 || grad_output.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output must be 4D");
  }
  if (attn_mask.has_value() || dropout_p != 0.0 || scale.has_value() ||
      enable_gqa) {
    return composite::sdpa_math_backward_composite(
        grad_output, query, key, value, attn_mask, dropout_p, is_causal, scale,
        enable_gqa);
  }
  if (static_cast<std::vector<int64_t>>(key.shape()) != static_cast<std::vector<int64_t>>(value.shape()) ||
      static_cast<std::vector<int64_t>>(query.shape()) != static_cast<std::vector<int64_t>>(grad_output.shape())) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output shapes must match");
  }
  if (query.size(0) != key.size(0) || query.size(1) != key.size(1) ||
      query.size(3) != key.size(3)) {
    TP_THROW(RuntimeError,
             "sdpa backward: without enable_gqa the query and key/value must "
             "agree on batch, head count and head dimension");
  }
  // The specialised backward below reads a key/value pair as long as the
  // query, so it serves exactly the self-attention shape.  A query shorter or
  // longer than its context is a different shape, not an impossible one, and
  // the composed reference covers it; refusing it instead would take
  // cross-attention and every one-row decode step out of reach.
  if (query.size(2) != key.size(2)) {
    return composite::sdpa_math_backward_composite(
        grad_output, query, key, value, attn_mask, dropout_p, is_causal, scale,
        enable_gqa);
  }
  Tensor query_c = query;
  Tensor key_c = key;
  Tensor value_c = value;
  Tensor grad_output_c = grad_output;
  const DType compute_dtype = query.dtype();
  if (key_c.dtype() != compute_dtype) key_c = key_c.to(compute_dtype);
  if (value_c.dtype() != compute_dtype) value_c = value_c.to(compute_dtype);
  if (grad_output_c.dtype() != compute_dtype) {
    grad_output_c = grad_output_c.to(compute_dtype);
  }
  if (compute_dtype == DType::Float32 || compute_dtype == DType::Float16 ||
      compute_dtype == DType::BFloat16) {
    return sdpa_backward_impl(grad_output_c, query_c, key_c, value_c, is_causal);
  }
  TP_THROW(NotImplementedError, "sdpa backward: only float32/float16/bfloat16 supported");
}

#if defined(TP_HAS_NATIVE_CUTE_FLASH)
template <typename KernelTraits>
__global__ void tp_flash_bwd_dot_do_o_kernel(
    const ::tensorplay_native_flash::Flash_bwd_params params) {
  ::tensorplay_native_flash::compute_dot_do_o<true, KernelTraits>(params);
}

template <typename KernelTraits, bool IsCausal, bool IsEvenMN>
__global__ void tp_flash_bwd_dq_dk_dv_kernel(
    const ::tensorplay_native_flash::Flash_bwd_params params) {
  ::tensorplay_native_flash::compute_dq_dk_dv_seqk_parallel<
      KernelTraits, false, IsCausal, false, false, IsEvenMN, true, false>(params);
}

template <typename KernelTraits>
__global__ void tp_flash_bwd_convert_dq_kernel(
    const ::tensorplay_native_flash::Flash_bwd_params params, int nsplits) {
  ::tensorplay_native_flash::convert_dQ<KernelTraits>(params, nsplits);
}

// The schedule each head width runs on: the block sizes and atom layouts the
// launch template picks when the shared-memory budget is tight, which keeps
// every entry launchable on the smallest sm80-and-newer part.  The widest
// heads give up the Q double buffer to fit.
template <int D, typename ElementT> struct BwdKernelTraitsFor;
template <typename T> struct BwdKernelTraitsFor<32, T> {
  using Traits = Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4, 4, true, false, T>;
};
template <typename T> struct BwdKernelTraitsFor<64, T> {
  using Traits = Flash_bwd_kernel_traits<64, 64, 128, 8, 2, 4, 4, true, false, T>;
};
template <typename T> struct BwdKernelTraitsFor<96, T> {
  using Traits = Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false, T>;
};
template <typename T> struct BwdKernelTraitsFor<128, T> {
  using Traits = Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, true, false, T>;
};
template <typename T> struct BwdKernelTraitsFor<192, T> {
  using Traits = Flash_bwd_kernel_traits<192, 64, 64, 8, 4, 2, 2, true, true, T>;
};
template <typename T> struct BwdKernelTraitsFor<256, T> {
  using Traits = Flash_bwd_kernel_traits<256, 64, 32, 8, 4, 1, 2, true, true, T>;
};

// The shared memory one launch of the schedule for ``D`` asks for.  The sizes
// scale with the element width, and the two half-precision element types are
// the same width, so one of them stands for both.
template <int D>
constexpr int bwd_launch_smem() {
  using Traits = typename BwdKernelTraitsFor<D, cutlass::half_t>::Traits;
  return std::max(Traits::kSmemSize1colblock, Traits::kSmemdQSize);
}

// The head widths a fused backward schedule exists for, and whether the
// device offers the shared memory that schedule's launch asks for.
bool flash_bwd_schedule_serves(int64_t d) {
  int wanted = 0;
  switch (d) {
    case 32: wanted = bwd_launch_smem<32>(); break;
    case 64: wanted = bwd_launch_smem<64>(); break;
    case 96: wanted = bwd_launch_smem<96>(); break;
    case 128: wanted = bwd_launch_smem<128>(); break;
    case 192: wanted = bwd_launch_smem<192>(); break;
    case 256: wanted = bwd_launch_smem<256>(); break;
    default: return false;
  }
  int max_smem = 0;
  TP_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
      getCurrentCUDAStream().device_index()));
  return wanted <= max_smem;
}

template <typename ElementT, int D, bool IsCausal, bool IsEvenMN>
void launch_tp_flash_bwd_dq_dk_dv(
    const ::tensorplay_native_flash::Flash_bwd_params& params,
    dim3 grid, cudaStream_t stream) {
  using KernelTraits = typename BwdKernelTraitsFor<D, ElementT>::Traits;
  constexpr int smem_size = KernelTraits::kSmemSize1colblock;
  auto kernel = &tp_flash_bwd_dq_dk_dv_kernel<KernelTraits, IsCausal, IsEvenMN>;
  if (smem_size >= 48 * 1024) {
    TP_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid, KernelTraits::kNThreads, smem_size, stream>>>(params);
  TP_CUDA_CHECK(cudaGetLastError());
}

template <typename ElementT, int D, bool IsCausal>
void launch_tp_flash_bwd(
    ::tensorplay_native_flash::Flash_bwd_params& params,
    int64_t B, int64_t H, int64_t lq, int64_t lk, cudaStream_t stream) {
  using KernelTraits = typename BwdKernelTraitsFor<D, ElementT>::Traits;
  const int m_blocks = static_cast<int>(
      (lq + KernelTraits::kBlockM - 1) / KernelTraits::kBlockM);
  const int n_blocks = static_cast<int>(
      (lk + KernelTraits::kBlockN - 1) / KernelTraits::kBlockN);
  const dim3 grid_m(m_blocks, static_cast<unsigned>(B), static_cast<unsigned>(H));
  const dim3 grid_n(n_blocks, static_cast<unsigned>(B), static_cast<unsigned>(H));

  tp_flash_bwd_dot_do_o_kernel<KernelTraits><<<
      grid_m, KernelTraits::kNThreads, 0, stream>>>(params);
  TP_CUDA_CHECK(cudaGetLastError());

  // A row block runs past the end of its tensor unless the length divides
  // the block height, and the query and key lengths each divide their own.
  if (lq % KernelTraits::kBlockM == 0 && lk % KernelTraits::kBlockN == 0) {
    launch_tp_flash_bwd_dq_dk_dv<ElementT, D, IsCausal, true>(
        params, grid_n, stream);
  } else {
    launch_tp_flash_bwd_dq_dk_dv<ElementT, D, IsCausal, false>(
        params, grid_n, stream);
  }

  auto convert_kernel = &tp_flash_bwd_convert_dq_kernel<KernelTraits>;
  if (KernelTraits::kSmemdQSize >= 48 * 1024) {
    TP_CUDA_CHECK(cudaFuncSetAttribute(
        convert_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits::kSmemdQSize));
  }
  convert_kernel<<<grid_m, KernelTraits::kNThreads,
                   KernelTraits::kSmemdQSize, stream>>>(params, 1);
  TP_CUDA_CHECK(cudaGetLastError());
}

template <typename ElementT, int D, bool IsCausal>
std::tuple<Tensor, Tensor, Tensor> sdpa_flash_backward_cute_impl(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    double scale) {
  const int64_t B = query.size(0);
  const int64_t H = query.size(1);
  const int64_t lq = query.size(2);
  const int64_t lk = key.size(2);
  // The accumulators the kernels walk over a 128-row pitch, whatever row
  // block height the schedule for this head width runs.
  const int64_t lq_rounded = (lq + 127) / 128 * 128;
  const int64_t lk_rounded = (lk + 127) / 128 * 128;

  auto feature_contiguous = [](const Tensor& tensor) {
    return tensor.stride(3) == 1 ? tensor : tensor.contiguous();
  };
  Tensor q = feature_contiguous(query);
  Tensor k = feature_contiguous(key);
  Tensor v = feature_contiguous(value);
  Tensor go = feature_contiguous(grad_output);
  Tensor out = feature_contiguous(output);
  Tensor d_q = Tensor::empty({B, H, lq, D}, query.dtype(), query.device());
  Tensor d_k = Tensor::empty({B, H, lk, D}, query.dtype(), query.device());
  Tensor d_v = Tensor::empty({B, H, lk, D}, query.dtype(), query.device());
  Tensor dq_accum = Tensor::empty(
      {B, lq_rounded, H, D}, DType::Float32, query.device());
  Tensor dsoftmax_sum = Tensor::empty(
      {B, H, lq_rounded}, DType::Float32, query.device());

  ::tensorplay_native_flash::Flash_bwd_params params{};
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  params.softmax_lse_ptr = logsumexp.data_ptr<float>();
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.q_row_stride = q.stride(2);
  params.k_row_stride = k.stride(2);
  params.v_row_stride = v.stride(2);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = out.stride(2);
  params.o_head_stride = out.stride(1);
  params.h = static_cast<int>(H);
  params.h_k = static_cast<int>(H);
  params.h_h_k_ratio = 1;
  params.b = static_cast<int>(B);
  params.seqlen_q = static_cast<int>(lq);
  params.seqlen_k = static_cast<int>(lk);
  params.seqlen_q_rounded = static_cast<int>(lq_rounded);
  params.seqlen_k_rounded = static_cast<int>(lk_rounded);
  params.total_q = static_cast<int>(B * lq);
  params.d = static_cast<int>(D);
  params.d_rounded = static_cast<int>(D);
  params.scale_softmax = static_cast<float>(scale);
  params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634f;
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.f;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = IsCausal ? 0 : -1;
  params.softcap = 0.f;
  params.is_bf16 = std::is_same<ElementT, cutlass::bfloat16_t>::value;
  params.is_causal = IsCausal;
  params.is_seqlens_k_cumulative = false;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.unpadded_lse = false;
  params.seqlenq_ngroups_swapped = false;
  params.do_ptr = go.data_ptr();
  params.dq_ptr = d_q.data_ptr();
  params.dk_ptr = d_k.data_ptr();
  params.dv_ptr = d_v.data_ptr();
  params.do_batch_stride = go.stride(0);
  params.do_row_stride = go.stride(2);
  params.do_head_stride = go.stride(1);
  params.dq_batch_stride = d_q.stride(0);
  params.dk_batch_stride = d_k.stride(0);
  params.dv_batch_stride = d_v.stride(0);
  params.dq_row_stride = d_q.stride(2);
  params.dk_row_stride = d_k.stride(2);
  params.dv_row_stride = d_v.stride(2);
  params.dq_head_stride = d_q.stride(1);
  params.dk_head_stride = d_k.stride(1);
  params.dv_head_stride = d_v.stride(1);
  params.dq_accum_ptr = dq_accum.data_ptr<float>();
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;
  params.dsoftmax_sum = dsoftmax_sum.data_ptr<float>();
  params.deterministic = false;
  params.dq_accum_split_stride = 0;

  const cudaStream_t stream = getCurrentCUDAStream().stream();
  launch_tp_flash_bwd<ElementT, D, IsCausal>(params, B, H, lq, lk, stream);
  return {d_q, d_k, d_v};
}

// The head width a call asks for picks the schedule its backward runs on.
template <typename ElementT, bool IsCausal>
std::tuple<Tensor, Tensor, Tensor> sdpa_flash_backward_cute_dispatch(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    int64_t d, double scale) {
  switch (d) {
    case 32:
      return sdpa_flash_backward_cute_impl<ElementT, 32, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    case 64:
      return sdpa_flash_backward_cute_impl<ElementT, 64, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    case 96:
      return sdpa_flash_backward_cute_impl<ElementT, 96, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    case 128:
      return sdpa_flash_backward_cute_impl<ElementT, 128, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    case 192:
      return sdpa_flash_backward_cute_impl<ElementT, 192, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    case 256:
      return sdpa_flash_backward_cute_impl<ElementT, 256, IsCausal>(
          grad_output, query, key, value, output, logsumexp, scale);
    default:
      TP_THROW(NotImplementedError,
               "no fused attention backward for head width ", d);
  }
}
#endif

std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cuda_with_lse(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    bool is_causal, int64_t impl) {
  if (query.numel() == 0 || key.numel() == 0) {
    // Without query rows or keys every product is empty and every gradient
    // is zero.
    auto zeros_shaped = [](const Tensor& like) {
      return Tensor::zeros(static_cast<std::vector<int64_t>>(like.shape()),
                           like.dtype(), like.device());
    };
    return std::make_tuple(zeros_shaped(query), zeros_shaped(key),
                           zeros_shaped(value));
  }
  Tensor query_c = query;
  Tensor key_c = key;
  Tensor value_c = value;
  Tensor output_c = output;
  Tensor grad_output_c = grad_output;
  const DType compute_dtype = query.dtype();
  if (key_c.dtype() != compute_dtype) key_c = key_c.to(compute_dtype);
  if (value_c.dtype() != compute_dtype) value_c = value_c.to(compute_dtype);
  if (output_c.dtype() != compute_dtype) output_c = output_c.to(compute_dtype);
  if (grad_output_c.dtype() != compute_dtype) {
    grad_output_c = grad_output_c.to(compute_dtype);
  }
  if (impl == 0 && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
      output.dim() == 4 && grad_output.dim() == 4 &&
      query_c.size(3) == key_c.size(3) && key_c.size(3) == value_c.size(3) &&
      query_c.dtype() == grad_output_c.dtype() &&
      query_c.dtype() == key_c.dtype() && query_c.dtype() == value_c.dtype() &&
      logsumexp.dtype() == DType::Float32 && logsumexp.is_contiguous() &&
      logsumexp.numel() == query_c.size(0) * query_c.size(1) * query_c.size(2) &&
      (query_c.dtype() == DType::Float16 || query_c.dtype() == DType::BFloat16)) {
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
    if (flash_bwd_schedule_serves(query_c.size(3))) {
      int major = 0;
      TP_CUDA_CHECK(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor,
          getCurrentCUDAStream().device_index()));
      if (major >= 8 && query_c.size(2) > 0 && key_c.size(2) > 0 &&
          query_c.size(0) == key_c.size(0) && query_c.size(1) == key_c.size(1) &&
          key_c.shape() == value_c.shape() && query_c.shape() == output_c.shape() &&
          query_c.shape() == grad_output_c.shape()) {
        const double s = composite::attention::resolve_scale(
            std::nullopt, query_c.size(3));
        if (is_causal) {
          if (query_c.dtype() == DType::Float16) {
            return sdpa_flash_backward_cute_dispatch<cutlass::half_t, true>(
                grad_output_c, query_c, key_c, value_c, output_c, logsumexp,
                query_c.size(3), s);
          }
          return sdpa_flash_backward_cute_dispatch<cutlass::bfloat16_t, true>(
              grad_output_c, query_c, key_c, value_c, output_c, logsumexp,
              query_c.size(3), s);
        }
        if (query_c.dtype() == DType::Float16) {
          return sdpa_flash_backward_cute_dispatch<cutlass::half_t, false>(
              grad_output_c, query_c, key_c, value_c, output_c, logsumexp,
              query_c.size(3), s);
        }
        return sdpa_flash_backward_cute_dispatch<cutlass::bfloat16_t, false>(
            grad_output_c, query_c, key_c, value_c, output_c, logsumexp,
            query_c.size(3), s);
      }
    }
#endif
  }
  if (impl == 0 && query_c.dim() == 4 && output_c.dim() == 4 &&
      query_c.shape() == key_c.shape() && key_c.shape() == value_c.shape() &&
      query_c.shape() == output_c.shape() &&
      query_c.shape() == grad_output_c.shape() && query_c.size(3) > 0 &&
      logsumexp.dtype() == DType::Float32 && logsumexp.is_contiguous() &&
      logsumexp.numel() == query_c.size(0) * query_c.size(1) * query_c.size(2) &&
      (compute_dtype == DType::Float32 || compute_dtype == DType::Float16 ||
       compute_dtype == DType::BFloat16)) {
    return sdpa_backward_from_lse(grad_output_c, query_c, key_c, value_c,
                                  output_c, logsumexp, is_causal);
  }
  return sdpa_backward_kernel_cuda(
      grad_output_c, query_c, key_c, value_c, /*attn_mask=*/std::nullopt,
      /*dropout_p=*/0.0, is_causal, /*scale=*/std::nullopt,
      /*enable_gqa=*/false);
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

// The kernel family below is what a plain call is computed by: no mask, no
// drop, no grouping, and the scale left at the one the head size implies.
Tensor sdpa_kernel_cuda_plain(const Tensor& query, const Tensor& key,
                                 const Tensor& value, bool is_causal,
                                 int64_t impl) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa: query/key/value must be 4D [B, H, T, D]");
  }
  int64_t B = query.size(0), H = query.size(1), T = query.size(2), D = query.size(3);
  if (key.size(0) != B || key.size(1) != H || value.size(0) != B || value.size(1) != H) {
    TP_THROW(RuntimeError, "sdpa: batch/head dims must match across q/k/v");
  }
  if (key.size(2) != value.size(2) || key.size(3) != D || value.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa: key/value shapes must match [B, H, T, D]");
  }
  if (key.size(2) != T) {
    // Every kernel behind this entry reads one token count for query, key and
    // value alike, so a context of a different length than the query is not a
    // shape it can express -- and taking the query's count for all three would
    // quietly answer a different question.  Callers that can have such a shape
    // ask the composed reference instead.
    TP_THROW(RuntimeError,
             "sdpa: query and key/value token counts must match; this entry "
             "point serves shapes where all three are the same length");
  }
  DType dtype = query.dtype();
  if (dtype != key.dtype() || dtype != value.dtype()) {
    TP_THROW(RuntimeError, "sdpa: q/k/v dtypes must match");
  }
  if (dtype != DType::Float32 && dtype != DType::Float16 && dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError, "sdpa: only float32/float16/bfloat16 supported");
  }
  float scale = 1.f / sqrtf((float)D);

  constexpr int kThreads = 256;

  // Default (impl=0) routing: fp16/bf16 with head_dim 32/64/96/128 takes the
  // tensor-core flash path, which beats the warp-per-row kernel on compact
  // GPUs; every other supported dtype at head_dim <= 128 keeps the
  // warp-per-row flash kernel, avoiding the naive kernel's float32 upcast.
  // The naive row-per-block kernel stays as the fallback for wider heads.
  auto route = [&]() {
    const bool flash_tensor_core_dtype =
        (dtype == DType::Float16 || dtype == DType::BFloat16) &&
        (D == 64 || D == 96 || D == 128
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
         || D == 32
#endif
        )
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
        // The wide precision's native flash leaves serve heads 64 and 128.
        || (dtype == DType::Float32 && (D == 64 || D == 128))
#endif
        ;
    bool tensor_cores_available = true;
#if !defined(USE_ROCM)
    if (impl == 0 && flash_tensor_core_dtype) {
      int major = 0;
      TP_CUDA_CHECK(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor,
          getCurrentCUDAStream().device_index()));
      tensor_cores_available = major >= (D == 32 ? 8 : 7);
    }
#endif
    if (impl == 0 && flash_tensor_core_dtype && tensor_cores_available) {
      impl = 5;
    } else if (impl == 0 && D <= 128) {
      // Scalar-precision attention outgrows the warp-per-row flash quickly:
      // the GEMM-native route runs tuned BLAS kernels and wins from short
      // sequences onward (crossover measured near T=64; at T=1024 it is an
      // order of magnitude ahead).  Short sequences keep the flash kernel's
      // single-launch simplicity.
      impl = (dtype == DType::Float32 && T >= 64) ? 2 : 3;
    }
  };

  // A short sequence with a narrow head is answered by the fused kernel: the
  // whole score matrix fits in one block's shared memory, so it never reaches
  // global memory and the row reduction is a tree instead of a single lane's
  // serial pass.  The gate is the one a four-float access needs anyway: a
  // feature count that is a multiple of four, a storage offset that keeps the
  // base address on a sixteen-byte boundary, and contiguous storage so the base
  // is the allocator's own.  A shape outside the gate is not slower for having
  // been turned away; it keeps the kernel it had.
  const bool fused_short_shape =
      impl == 0 && dtype == DType::Float32 && T > 0 && T <= kSdpaFusedMaxTokens &&
      D > 0 && D <= kSdpaFusedMaxDim && (D % 4) == 0;
  // Which schedule answers the call decides what it needs of its operands,
  // so the route is settled before anything is copied.
  if (!fused_short_shape) route();
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
  const bool native_flash = impl == 5 || impl == 6 || impl == 7;
#else
  constexpr bool native_flash = false;
#endif
  // The standalone CUTE/CUTLASS kernel supports arbitrary strides in its
  // batch/head/token coordinates, but its innermost dimension is vectorized.
  // Preserve a compatible view and only materialize inputs whose D dimension
  // is not contiguous -- the heads read straight out of a packed projection.
  // GEMM and the legacy kernels retain their canonical contiguous-input
  // contract.
  auto flash_input = [](const Tensor& input) {
    return input.stride(3) == 1 ? input : input.contiguous();
  };
  Tensor q = native_flash ? flash_input(query) : query.contiguous();
  Tensor k = native_flash ? flash_input(key) : key.contiguous();
  Tensor v = native_flash ? flash_input(value) : value.contiguous();

  const auto offset_aligned = [](const Tensor& t) {
    return (t.storage_offset() * static_cast<int64_t>(t.itemsize())) % 16 == 0;
  };
  if (fused_short_shape && q.is_contiguous() && k.is_contiguous() &&
      v.is_contiguous() && offset_aligned(q) && offset_aligned(k) &&
      offset_aligned(v)) {
    const int dstride = static_cast<int>(D) + kSdpaFusedRowPad;
    const int tstride = static_cast<int>(T) + 1;
    const size_t smem = static_cast<size_t>(3 * T * dstride + T * tstride) *
                        sizeof(float);
    Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
    sdpa_fused_short_kernel<float>
        <<<static_cast<unsigned>(B * H), kSdpaFusedThreads, smem,
           getCurrentCUDAStream().stream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            out.data_ptr<float>(), nullptr, B, H, T, D, scale, is_causal);
    TP_CUDA_CHECK(cudaGetLastError());
    return out;
  }
  // A short shape the fused kernel turned away is routed like any other; its
  // operands are already dense, which every route reads.
  if (fused_short_shape) route();

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
#if defined(USE_ROCM) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
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
#if defined(USE_ROCM)
      const void* flash_4warp_kernel =
          reinterpret_cast<const void*>(&sdpa_wmma_flash_half_4warp_kernel);
#else
      const void* flash_4warp_kernel =
          reinterpret_cast<const void*>(sdpa_wmma_flash_half_4warp_kernel);
#endif
      TP_CUDA_CHECK(cudaFuncSetAttribute(
          flash_4warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
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
#else
    TP_THROW(NotImplementedError,
             "sdpa impl=8 (4-warp FP16 WMMA flash) requires compute capability 7.0 or newer");
#endif
  } else if (impl == 5 || impl == 6 || impl == 7) {
    bool supported_head_dim = D == 64 || D == 96 || D == 128;
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
    supported_head_dim = supported_head_dim || D == 32;
#endif
    const bool wide_flash = dtype == DType::Float32 && (D == 64 || D == 128);
    if ((!wide_flash && dtype != DType::Float16 && dtype != DType::BFloat16) ||
        !supported_head_dim) {
      TP_THROW(NotImplementedError,
               "sdpa impl=5 (aligned WMMA flash) requires dtype=float16/bfloat16 and a supported head dimension");
    }
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
    // Use the native CUTE/CUTLASS aligned path for both full and tail tiles;
    // its internal predicate schedule is also needed by autoregressive decode
    // once T grows past a multiple of 64.
    SdpaFusedLaunch launch;
    // Causal is the window whose right bound is zero with the left unbounded;
    // the launcher reads the mask off the bounds the same way.
    launch.window_right = is_causal ? 0 : -1;
    return std::get<0>(
        sdpa_fused_forward_cuda(q, k, v, launch, scale, /*enable_gqa=*/false));
#elif defined(USE_ROCM) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
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
#if defined(USE_ROCM)
      const void* flash_aligned_kernel =
          reinterpret_cast<const void*>(&sdpa_wmma_flash_half_aligned_kernel);
#else
      const void* flash_aligned_kernel =
          reinterpret_cast<const void*>(sdpa_wmma_flash_half_aligned_kernel);
#endif
      TP_CUDA_CHECK(cudaFuncSetAttribute(
          flash_aligned_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
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
#else
    TP_THROW(NotImplementedError,
             "sdpa impl=5 (aligned FP16 WMMA flash) requires compute capability 7.0 or newer");
#endif  // TP_HAS_NATIVE_CUTE_FLASH
  } else if (impl == 4) {
#if defined(USE_ROCM) || !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
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
#else
    TP_THROW(NotImplementedError,
             "sdpa impl=4 (FP16 WMMA flash) requires compute capability 7.0 or newer");
#endif
  } else if (impl == 2) {
    if (dtype == DType::Float32) {
      return sdpa_gemm_native<float>(q, k, v, B, H, H, T, T, D, is_causal);
    } else if (dtype == DType::Float16) {
      return sdpa_gemm_native<tensorplay::Half>(
          q, k, v, B, H, H, T, T, D, is_causal);
    } else {
      return sdpa_gemm_native<tensorplay::BFloat16>(
          q, k, v, B, H, H, T, T, D, is_causal);
    }
  } else {
    TP_THROW(RuntimeError, "sdpa: unknown impl " + std::to_string(impl));
  }
}

// The GEMM-backed schedule over a context whose length differs from the
// query's, or over grouped heads -- the two shapes the square entry point
// cannot state.  It materializes the score matrix instead of tiling it, so it is
// what answers a precision the tiled schedule has no tiles for; within a
// precision the tiled schedule is preferred wherever it applies, because it
// never writes the scores out at all.
Tensor sdpa_gemm_cross_cuda(const Tensor& query, const Tensor& key,
                            const Tensor& value, bool is_causal) {
  const Tensor q = query.contiguous();
  const Tensor k = key.contiguous();
  const Tensor v = value.contiguous();
  const int64_t B = q.size(0);
  const int64_t Hq = q.size(1);
  const int64_t Hkv = k.size(1);
  const int64_t Tq = q.size(2);
  const int64_t Tkv = k.size(2);
  const int64_t D = q.size(3);
  if (q.dtype() == DType::Float32) {
    return sdpa_gemm_native<float>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  if (q.dtype() == DType::Float16) {
    return sdpa_gemm_native<tensorplay::Half>(
        q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  return sdpa_gemm_native<tensorplay::BFloat16>(
      q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
}

#if defined(TP_HAS_NATIVE_CUTE_FLASH)
// The fused schedule for a call that carries arguments the square self-
// attention entry point does not take: an explicit score normaliser, grouped
// heads, and a context whose length differs from the query's.  The kernel reads
// all three as launch parameters, so nothing here has to widen a tensor or
// materialise a score matrix; what is left to the caller is only the head
// dimension and the innermost stride, both of which the predicate has already
// The fused entry point for a call carrying what the square self-attention one
// does not: an explicit normaliser, grouped heads, a context whose length
// differs from the query's, a sliding window, or sequences packed end to end.
//
// The schedule vectorises its innermost dimension, so a head axis that is not
// the contiguous one is materialized; every other axis is addressed by the
// caller's real strides and is consumed as given.
//
// A packed batch arrives as (total, heads, dim) with no batch axis at all, so
// it is read as a batch of one whose single sequence is the whole packed run;
Tensor sdpa_fused_extended_cuda(const Tensor& query, const Tensor& key,
                                const Tensor& value, bool is_causal,
                                std::optional<double> scale, bool enable_gqa) {
  SdpaFusedLaunch launch;
  launch.window_right = is_causal ? 0 : -1;
  return std::get<0>(
      sdpa_fused_forward_cuda(query, key, value, launch, scale, enable_gqa));
}
#endif  // TP_HAS_NATIVE_CUTE_FLASH

// The public attention call names a mask, a drop, a scale and grouped heads, and
// the composite is what says what each of them means, for every call alike: a
// caller that named none of them is asking for the same attention, and
// answering it by a different route would make two spellings of one computation
// disagree.  The exception is a call a device schedule can state outright, which
// the selector names, so that a caller arriving here directly is not left paying
// for a score matrix the tiled schedule never materialises.  The predicate is
// the selector's own, so the two cannot drift apart.  The self-attention shape
// with the default normaliser goes to the square entry point, which is also the
// one the matching fused backward hangs off, so a training call keeps the fast
// backward.
Tensor sdpa_kernel_cuda(const Tensor& query, const Tensor& key,
                        const Tensor& value,
                        const std::optional<Tensor>& attn_mask, double dropout_p,
                        bool is_causal, std::optional<double> scale,
                        bool enable_gqa) {
  switch (composite::fused_sdpa_schedule(query, key, value, attn_mask, dropout_p,
                                        scale, enable_gqa, is_causal)) {
    case composite::FusedSdpaSchedule::kSquare:
      return sdpa_kernel_cuda_plain(query, key, value, is_causal, /*impl=*/0);
    case composite::FusedSdpaSchedule::kTiled:
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
      return sdpa_fused_extended_cuda(query, key, value, is_causal, scale,
                                      enable_gqa);
#else
      break;
#endif
    case composite::FusedSdpaSchedule::kWideTiled:
#if !defined(USE_ROCM)
      return sdpa_wide_tiled_cuda(query, key, value, is_causal);
#else
      break;
#endif
    case composite::FusedSdpaSchedule::kCrossGemm:
      // A context of another length in a precision the tiled schedule has no
      // tiles for.  The two reduced precisions have tiles, so they went to
      // kTiled instead and never reach this arm.
      return sdpa_gemm_cross_cuda(query, key, value, is_causal);
    case composite::FusedSdpaSchedule::kNone:
      break;
  }
  // The composite answers with the attention and the normalizer it built along
  // the way; only the attention is what was asked for.
  auto [output, logsumexp] = composite::sdpa_math_composite(
      query, key, value, attn_mask, dropout_p, is_causal,
      /*dropout_mask=*/std::nullopt, scale, enable_gqa);
  (void)logsumexp;
  return output;
}

std::tuple<Tensor, Tensor> sdpa_kernel_cuda_with_lse(
    const Tensor& query, const Tensor& key, const Tensor& value,
    bool is_causal, int64_t impl) {
  if (query.dim() == 4 && value.dim() == 4 && query.numel() == 0) {
    // No query rows: nothing to attend from, and no kernel to launch over
    // an empty grid.
    return std::make_tuple(
        Tensor::empty({query.size(0), query.size(1), query.size(2), value.size(3)},
                      query.dtype(), query.device()),
        Tensor::empty({query.size(0), query.size(1), query.size(2)},
                      DType::Float32, query.device()));
  }
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
  if (impl == 0 && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
      query.size(3) == 32 &&
      (query.dtype() == DType::Float16 || query.dtype() == DType::BFloat16) &&
      query.dtype() == key.dtype() && query.dtype() == value.dtype() &&
      query.size(0) == key.size(0) && query.size(1) == key.size(1) &&
      query.size(2) == key.size(2) && query.size(3) == key.size(3) &&
      static_cast<std::vector<int64_t>>(key.shape()) ==
          static_cast<std::vector<int64_t>>(value.shape())) {
    int major = 0;
    TP_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor,
        getCurrentCUDAStream().device_index()));
    if (major >= 8) {
      auto feature_contiguous = [](const Tensor& tensor) {
        return tensor.stride(3) == 1 ? tensor : tensor.contiguous();
      };
      Tensor q = feature_contiguous(query);
      Tensor k = feature_contiguous(key);
      Tensor v = feature_contiguous(value);
      const int64_t B = q.size(0);
      const int64_t H = q.size(1);
      const int64_t T = q.size(2);
      const int64_t D = q.size(3);
      // One head count and one token count, and the head width's own
      // normaliser: this entry point names none of the three.
      const double s = 1.0 / std::sqrt(static_cast<double>(D));
      Tensor lse;
      SdpaFusedLaunch launch;
      launch.lse_out = &lse;
      launch.window_right = is_causal ? 0 : -1;
      return sdpa_fused_forward_cuda(q, k, v, launch, s,
                                     /*enable_gqa=*/false);
    }
  }
#endif
  // A context of a different length than the query is served here too, by
  // the same schedule the public entry uses: the reduced precisions' tile path
  // carries its own log-sum-exp, and the wide precision writes the same row
  // statistics out of its kernel.
  if (impl == 0 && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
      query.size(2) != key.size(2) && query.size(0) == key.size(0) &&
      query.size(1) == key.size(1) && key.size(1) == value.size(1) &&
      query.size(3) == key.size(3) && key.size(3) == value.size(3) &&
      key.size(2) == value.size(2) &&
      query.dtype() == key.dtype() && key.dtype() == value.dtype()) {
    const DType dtype = query.dtype();
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
      int major = 0;
      TP_CUDA_CHECK(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor,
          getCurrentCUDAStream().device_index()));
      if (major >= 8) {
        auto feature_contiguous = [](const Tensor& tensor) {
          return tensor.stride(3) == 1 ? tensor : tensor.contiguous();
        };
        Tensor q = feature_contiguous(query);
        Tensor k = feature_contiguous(key);
        Tensor v = feature_contiguous(value);
        const double s = 1.0 / std::sqrt(static_cast<double>(q.size(3)));
        Tensor lse;
        SdpaFusedLaunch launch;
        launch.lse_out = &lse;
        launch.window_right = is_causal ? 0 : -1;
        return sdpa_fused_forward_cuda(q, k, v, launch, s,
                                       /*enable_gqa=*/false);
      }
#endif
    } else if (dtype == DType::Float32 && query.size(3) == 128) {
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
      int major = 0;
      TP_CUDA_CHECK(cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor,
          getCurrentCUDAStream().device_index()));
      if (major >= 8) {
        auto feature_contiguous = [](const Tensor& tensor) {
          return tensor.stride(3) == 1 ? tensor : tensor.contiguous();
        };
        Tensor q = feature_contiguous(query);
        Tensor k = feature_contiguous(key);
        Tensor v = feature_contiguous(value);
        const double s = 1.0 / std::sqrt(static_cast<double>(q.size(3)));
        Tensor lse;
        SdpaFusedLaunch launch;
        launch.lse_out = &lse;
        launch.window_right = is_causal ? 0 : -1;
        return sdpa_fused_forward_cuda(q, k, v, launch, s,
                                       /*enable_gqa=*/false);
      }
#endif
#if !defined(USE_ROCM)
      Tensor lse;
      Tensor output =
          sdpa_wide_tiled_cuda_with_lse(query, key, value, is_causal, lse);
      return {output, lse};
#endif
    }
  }
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
  // The fused schedule writes each row's log-sum-exp as it goes.  A
  // same-length call it serves hands that back, so the backward reads the
  // probabilities off it instead of normalizing every score row again.
  const bool flash_head =
      query.dim() == 4 &&
      ((query.dtype() == DType::Float32 &&
        (query.size(3) == 64 || query.size(3) == 128)) ||
       ((query.dtype() == DType::Float16 || query.dtype() == DType::BFloat16) &&
        (query.size(3) == 64 || query.size(3) == 96 || query.size(3) == 128)));
  if (impl == 0 && flash_head && key.dim() == 4 && value.dim() == 4 &&
      query.dtype() == key.dtype() && key.dtype() == value.dtype() &&
      query.shape() == key.shape() && key.shape() == value.shape()) {
    int major = 0;
    TP_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor,
        getCurrentCUDAStream().device_index()));
    if (major >= 8) {
      auto feature_contiguous = [](const Tensor& tensor) {
        return tensor.stride(3) == 1 ? tensor : tensor.contiguous();
      };
      Tensor lse;
      SdpaFusedLaunch launch;
      launch.lse_out = &lse;
      launch.window_right = is_causal ? 0 : -1;
      return sdpa_fused_forward_cuda(
          feature_contiguous(query), feature_contiguous(key),
          feature_contiguous(value), launch,
          1.0 / std::sqrt(static_cast<double>(query.size(3))),
          /*enable_gqa=*/false);
    }
  }
#endif
  Tensor output = sdpa_kernel_cuda_plain(query, key, value, is_causal, impl);
  Tensor lse = Tensor::empty({0}, DType::Float32, query.device());
  return {output, lse};
}

#if defined(TP_HAS_NATIVE_CUTE_FLASH)
// The three contracts below differ in which axis the sequence is on, in what
// the mask is named, and in what else the result carries, but they are the same
// computation and the same schedule.  These helpers say what they share so the
// three bodies stay about their own spelling.

// A negative bound is unbounded, and the kernel decides which mask it is by
// the same rule: unbounded on the left and zero on the right is causal, any
// other pair of finite bounds is a window, and both unbounded is no mask.

// The heads of a batched tensor are the second axis for the kernel and the
// third for two of the three contracts, so a sequence-major caller is read
// through a transpose.  That is a view: the schedule addresses every axis by
// the tensor's own strides, so nothing is copied and only the last axis, which
// the schedule vectorises over, has to be contiguous.
void heads_second(const Tensor& t, bool sequence_major, Tensor* out) {
  *out = sequence_major ? t.transpose(1, 2) : t;
}

// Whether the fused schedule can answer this call at all.  Everything it
// cannot answer is answered by the composite instead, in the same body, so a
// caller never has to know which of the two ran.
bool fused_schedule_serves(const Tensor& q, const Tensor& k, const Tensor& v,
                           bool packed, int64_t rank) {
  if (packed) {
    if (rank != 3) return false;
  } else if (rank != 4) {
    return false;
  }
  const DType dt = q.dtype();
  if (dt != DType::Float16 && dt != DType::BFloat16) return false;
  if (k.dtype() != dt || v.dtype() != dt) return false;
  const int64_t head_dim = q.size(-1);
  if (head_dim != 32 && head_dim != 64 && head_dim != 96 && head_dim != 128) {
    return false;
  }
  if (k.size(-1) != head_dim || v.size(-1) != head_dim) return false;
  if (v.size(-2) != k.size(-2)) return false;  // value and key share an extent
  if (q.size(-2) == 0 || k.size(-2) == 0) return false;
  // Grouped heads are read as one key head serving a contiguous run of query
  // heads, so the query head count has to be a whole number of runs.  On a
  // packed run the head axis sits one place in from where a batched tensor
  // carries it, and the outer axis is the token count, which has no reason to
  // divide evenly and must not be read as a head count here.
  const int64_t heads_q = packed ? q.size(1) : q.size(-3);
  const int64_t heads_k = packed ? k.size(1) : k.size(-3);
  if (heads_q == 0 || heads_k == 0) return false;
  if (heads_q % heads_k != 0) return false;
  return true;
}

// Whether the keys and values carry fewer heads than the query.  None of these
// three contracts names grouped heads, but the head counts say so plainly, and
// reading it here keeps the schedule from being handed a shape it cannot use.
bool heads_are_grouped(const Tensor& q, const Tensor& k) {
  return q.size(-3) != k.size(-3);
}

Tensor flash_rng_state(const Tensor& like) {
  return Tensor::zeros({2}, DType::UInt64, like.device());
}

Tensor flash_empty_scalar(const Tensor& like) {
  return Tensor::zeros({}, DType::UInt64, like.device());
}

Tensor flash_debug_mask(const Tensor& like) {
  return Tensor::empty({0}, like.dtype(), like.device());
}

// The schedule answers with the heads on the second axis, and a packed batch's
// output written straight through the token axis as the table describes it.
// Its constant is already in the layout every contract names -- (batch,
// heads, queries), or (heads, total) when packed -- so only the output moves.
Tensor restore_output(const Tensor& out, bool packed, bool sequence_major,
                      int64_t total, int64_t heads, int64_t dim) {
  if (packed) return out.reshape({total, heads, dim});
  return sequence_major ? out.transpose(1, 2) : out;
}

// The window a top-left causal mask is on the schedule's own terms: the
// schedule measures bounds from the lower-right diagonal, so the upper-left one
// is a right bound of lq - lk, which is a bound only when it is not negative.
std::optional<int64_t> top_left_causal_right(int64_t lq, int64_t lk) {
  if (lq < lk) return std::nullopt;
  return lq - lk;
}

// `_flash_attention_forward`: batched inputs are (batch, sequence, heads,
// dim); a cumulative-length table means they are packed as (total, heads, dim)
// instead.  The causal flag is the window whose right bound is zero.  What the
// fused schedule cannot express is answered by the composite.
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> flash_attention_forward_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& cum_seq_q, const std::optional<Tensor>& cum_seq_k,
    int64_t max_q, int64_t max_k, double dropout_p, bool is_causal,
    bool return_debug_mask, std::optional<double> scale,
    std::optional<int64_t> window_size_left, std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& seqused_k, const std::optional<Tensor>& alibi_slopes,
    const std::optional<Tensor>& block_table, std::optional<int64_t> num_splits) {
  const bool packed = cum_seq_q.has_value();
  const int64_t left = window_size_left.value_or(-1);
  const int64_t right = is_causal ? 0 : window_size_right.value_or(-1);
  const bool servable =
      dropout_p == 0.0 && !seqused_k.has_value() && !alibi_slopes.has_value() &&
      !block_table.has_value() && (cum_seq_q.has_value() == cum_seq_k.has_value()) &&
      fused_schedule_serves(query, key, value, packed, query.dim());
  if (!servable) {
    return composite::attention::flash_forward(
        query, key, value, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
        is_causal, return_debug_mask, scale, window_size_left, window_size_right,
        seqused_k, alibi_slopes, block_table, num_splits);
  }
  Tensor q, k, v;
  heads_second(query, !packed, &q);
  heads_second(key, !packed, &k);
  heads_second(value, !packed, &v);
  SdpaFusedLaunch launch;
  launch.window_left = left;
  launch.window_right = right;
  if (packed) {
    launch.cu_seqlens_q = cum_seq_q->data_ptr<int32_t>();
    launch.cu_seqlens_k = cum_seq_k->data_ptr<int32_t>();
    launch.num_seqs = cum_seq_q->size(0) - 1;
    launch.max_seqlen_q = max_q;
    launch.max_seqlen_k = max_k;
  }
  Tensor lse;
  launch.lse_out = &lse;
  // The caller's own count, or none for the split choice to make from the
  // block count.  A packed batch is answered without a cut, so the count is
  // only read on the batched path.
  launch.num_splits = static_cast<int>(num_splits.value_or(0));
  Tensor out = std::get<0>(
      sdpa_fused_forward_cuda(q, k, v, launch, scale, heads_are_grouped(q, k)));
  out = restore_output(out, packed, /*sequence_major=*/true, query.size(0),
                       query.size(1), value.size(-1));
  return {out, lse, flash_rng_state(query), flash_empty_scalar(query),
          flash_debug_mask(query)};
}

// `_cudnn_attention_forward`: batched inputs are head-major and the causal
// flag keeps the keys at or before the query index; a packed call hands its
// tables back unchanged.
std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt, Tensor, Tensor,
           Tensor>
cudnn_attention_forward_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_bias, const std::optional<Tensor>& cum_seq_q,
    const std::optional<Tensor>& cum_seq_k, int64_t max_q, int64_t max_k,
    bool compute_logsumexp, double dropout_p, bool is_causal,
    bool return_debug_mask, std::optional<double> scale,
    const std::optional<Tensor>& seqused_k, const std::optional<Tensor>& block_table) {
  const bool packed = cum_seq_q.has_value();
  // Over a packed batch each sequence aligns its own diagonal, and the two
  // corners agree for every sequence only when the query and key tables are
  // one table.
  std::optional<int64_t> causal_right;
  if (is_causal) {
    if (packed) {
      if (cum_seq_k.has_value() &&
          cum_seq_k->unsafeGetTensorImpl() == cum_seq_q->unsafeGetTensorImpl()) {
        causal_right = 0;
      }
    } else if (query.dim() == 4 && key.dim() == 4) {
      causal_right = top_left_causal_right(query.size(2), key.size(2));
    }
  }
  const bool servable =
      dropout_p == 0.0 && !attn_bias.has_value() && !seqused_k.has_value() &&
      !block_table.has_value() && cum_seq_k.has_value() == packed &&
      (!is_causal || causal_right.has_value()) &&
      fused_schedule_serves(query, key, value, packed, query.dim());
  if (!servable) {
    return composite::attention::cudnn_forward(
        query, key, value, attn_bias, cum_seq_q, cum_seq_k, max_q, max_k,
        compute_logsumexp, dropout_p, is_causal, return_debug_mask, scale,
        seqused_k, block_table);
  }
  SdpaFusedLaunch launch;
  launch.window_left = -1;
  launch.window_right = causal_right.value_or(-1);
  if (packed) {
    launch.cu_seqlens_q = cum_seq_q->data_ptr<int32_t>();
    launch.cu_seqlens_k = cum_seq_k->data_ptr<int32_t>();
    launch.num_seqs = cum_seq_q->size(0) - 1;
    launch.max_seqlen_q = max_q;
    launch.max_seqlen_k = max_k;
  }
  Tensor lse;
  if (compute_logsumexp) launch.lse_out = &lse;
  Tensor out = std::get<0>(sdpa_fused_forward_cuda(
      query, key, value, launch, scale, heads_are_grouped(query, key)));
  out = restore_output(out, packed, /*sequence_major=*/false, query.size(0),
                       query.size(1), value.size(-1));
  if (compute_logsumexp && !packed) lse = lse.unsqueeze(-1);
  return {out,
          lse,
          packed ? *cum_seq_q : Tensor(),
          packed ? *cum_seq_k : Tensor(),
          SymInt(packed ? max_q : query.size(2)),
          SymInt(packed ? max_k : key.size(2)),
          Tensor::zeros({}, DType::Int64, query.device()),
          Tensor::zeros({}, DType::Int64, query.device()),
          Tensor()};
}

// `_efficient_attention_forward`: batched inputs are (batch, sequence, heads,
// dim) like the flash-shaped one, and the mask alignment arrives as a code.
// Its constant is padded to a multiple of 32 queries with positive infinity.
std::tuple<Tensor, Tensor, Tensor, Tensor, SymInt, SymInt>
efficient_attention_forward_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& bias, const std::optional<Tensor>& cu_seqlens_q,
    const std::optional<Tensor>& cu_seqlens_k,
    std::optional<int64_t> max_seqlen_q, std::optional<int64_t> max_seqlen_k,
    double dropout_p, int64_t custom_mask_type, bool compute_log_sumexp,
    std::optional<double> scale, const std::optional<Tensor>& seqlen_k,
    std::optional<int64_t> window_size) {
  const bool packed = cu_seqlens_q.has_value();
  // The packed constant is laid out per sequence, which needs the table on the
  // host; only a packed call that does not ask for it is left to the schedule.
  std::optional<int64_t> right;
  if (!packed && query.dim() == 4 && key.dim() == 4) {
    if (custom_mask_type == composite::attention::kCausalFromTopLeft) {
      right = top_left_causal_right(query.size(1), key.size(1));
    } else if (custom_mask_type == composite::attention::kCausalFromBottomRight) {
      right = 0;
    } else if (custom_mask_type == composite::attention::kNoCustomMask) {
      right = -1;
    }
  } else if (packed && !compute_log_sumexp &&
             custom_mask_type == composite::attention::kNoCustomMask) {
    right = -1;
  }
  const bool packed_rank_ok = !packed || query.dim() == 3;
  const bool servable =
      dropout_p == 0.0 && !bias.has_value() && !window_size.has_value() &&
      !seqlen_k.has_value() && right.has_value() && packed_rank_ok &&
      cu_seqlens_q.has_value() == cu_seqlens_k.has_value() &&
      fused_schedule_serves(query, key, value, packed, query.dim());
  if (!servable) {
    return composite::attention::efficient_forward(
        query, key, value, bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        max_seqlen_k, dropout_p, custom_mask_type, compute_log_sumexp, scale,
        seqlen_k, window_size);
  }
  const int64_t length_q = packed ? query.size(0) : query.size(1);
  const int64_t length_k = packed ? key.size(0) : key.size(1);
  SdpaFusedLaunch launch;
  launch.window_left = -1;
  launch.window_right = *right;
  if (packed) {
    launch.cu_seqlens_q = cu_seqlens_q->data_ptr<int32_t>();
    launch.cu_seqlens_k = cu_seqlens_k->data_ptr<int32_t>();
    launch.num_seqs = cu_seqlens_q->size(0) - 1;
    launch.max_seqlen_q = max_seqlen_q.value_or(length_q);
    launch.max_seqlen_k = max_seqlen_k.value_or(length_k);
  }
  Tensor q, k, v;
  heads_second(query, !packed, &q);
  heads_second(key, !packed, &k);
  heads_second(value, !packed, &v);
  Tensor lse;
  if (compute_log_sumexp) launch.lse_out = &lse;
  Tensor out = std::get<0>(
      sdpa_fused_forward_cuda(q, k, v, launch, scale, heads_are_grouped(q, k)));
  out = restore_output(out, packed, /*sequence_major=*/true, query.size(0),
                       query.size(1), value.size(-1));
  const int64_t batch = packed ? cu_seqlens_q->size(0) - 1 : query.size(0);
  const int64_t heads = packed ? query.size(1) : query.size(2);
  if (compute_log_sumexp) {
    Tensor padded = composite::attention::infinite(
        {batch, heads, composite::attention::round_up_32(length_q)},
        DType::Float32, query.device());
    composite::attention::write_rows(padded, 2, 0, lse);
    lse = padded;
  } else {
    lse = Tensor::empty({batch, heads, 0}, DType::Float32, query.device());
  }
  return {out, lse, Tensor::zeros({}, DType::Int64, query.device()),
          Tensor::zeros({}, DType::Int64, query.device()),
          SymInt(packed ? launch.max_seqlen_q : length_q),
          SymInt(packed ? launch.max_seqlen_k : length_k)};
}

// `_flash_attention_backward`: the fused backward serves the half-precision
// self-attention shape -- any head width a schedule exists for, no window but
// the causal one, no dropout, and either the scale the head width implies or
// one the caller passes; every other call is the composite's.
std::tuple<Tensor, Tensor, Tensor> flash_attention_backward_cuda(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    const Tensor& cum_seq_q, const Tensor& cum_seq_k, int64_t max_q,
    int64_t max_k, double dropout_p, bool is_causal, const Tensor& rng_state,
    const Tensor& unused, std::optional<double> scale,
    std::optional<int64_t> window_size_left, std::optional<int64_t> window_size_right) {
  const int64_t left = window_size_left.value_or(-1);
  const int64_t right = is_causal ? 0 : window_size_right.value_or(-1);
  const bool plain_window = left < 0 && (right < 0 || right == 0);
  const DType dt = query.dtype();
  const bool fused =
      grad_out.defined() && !cum_seq_q.defined() && dropout_p == 0.0 &&
      plain_window && query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
      (dt == DType::Float16 || dt == DType::BFloat16) && key.dtype() == dt &&
      value.dtype() == dt && grad_out.dtype() == dt && out.dtype() == dt &&
      query.size(3) == key.size(3) && query.size(3) == value.size(3) &&
      query.size(2) == key.size(2) &&
      key.shape() == value.shape() && query.shape() == out.shape() &&
      query.shape() == grad_out.shape() && query.size(2) > 0 &&
      query.size(1) > 0 && key.size(1) > 0 &&
      logsumexp.dtype() == DType::Float32 && logsumexp.is_contiguous() &&
      logsumexp.numel() == query.size(0) * query.size(1) * query.size(2);
  if (fused) {
    int major = 0;
    TP_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor,
        getCurrentCUDAStream().device_index()));
    if (major >= 8 && flash_bwd_schedule_serves(query.size(3))) {
      const bool causal = right == 0;
      const double s = composite::attention::resolve_scale(scale, query.size(3));
      Tensor go = grad_out.transpose(1, 2), q = query.transpose(1, 2);
      Tensor k = key.transpose(1, 2), v = value.transpose(1, 2);
      Tensor o = out.transpose(1, 2);
      std::tuple<Tensor, Tensor, Tensor> grads;
      if (dt == DType::Float16) {
        grads = causal ? sdpa_flash_backward_cute_dispatch<cutlass::half_t, true>(
                             go, q, k, v, o, logsumexp, query.size(3), s)
                       : sdpa_flash_backward_cute_dispatch<cutlass::half_t, false>(
                             go, q, k, v, o, logsumexp, query.size(3), s);
      } else {
        grads = causal ? sdpa_flash_backward_cute_dispatch<cutlass::bfloat16_t, true>(
                             go, q, k, v, o, logsumexp, query.size(3), s)
                       : sdpa_flash_backward_cute_dispatch<cutlass::bfloat16_t, false>(
                             go, q, k, v, o, logsumexp, query.size(3), s);
      }
      return {std::get<0>(grads).transpose(1, 2), std::get<1>(grads).transpose(1, 2),
              std::get<2>(grads).transpose(1, 2)};
    }
  }
  return composite::attention::flash_backward(
      grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q,
      max_k, dropout_p, is_causal, rng_state, unused, scale, window_size_left,
      window_size_right);
}
#endif  // TP_HAS_NATIVE_CUTE_FLASH

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

namespace tensorplay {
namespace cuda {

TENSORPLAY_LIBRARY_IMPL(CUDA, AttentionKernels) {
  m.impl("scaled_dot_product_attention", sdpa_kernel_cuda);
  m.impl("scaled_dot_product_attention_backward", sdpa_backward_kernel_cuda);
  m.impl("_scaled_dot_product_attention_with_lse", sdpa_kernel_cuda_with_lse);
  m.impl("_scaled_dot_product_attention_backward_with_lse",
         sdpa_backward_kernel_cuda_with_lse);
#if defined(TP_HAS_NATIVE_CUTE_FLASH)
  // These three carry a mask, a window, grouped heads or a packed batch that
  // the plain entry point does not take.  Each answers from the fused schedule
  // when that schedule can express the call and from the composite otherwise,
  // so a caller gets one answer either way and never has to know which ran.
  m.impl("_flash_attention_forward", flash_attention_forward_cuda);
  m.impl("_cudnn_attention_forward", cudnn_attention_forward_cuda);
  m.impl("_efficient_attention_forward", efficient_attention_forward_cuda);
  m.impl("_flash_attention_backward", flash_attention_backward_cuda);
#endif
}

} // namespace cuda
} // namespace tensorplay
