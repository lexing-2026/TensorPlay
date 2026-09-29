#include "Tensor.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "AttentionUtils.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <tuple>

#define TP_WIDE_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, \
               std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace tensorplay {
namespace cuda {

// The wide precision has no tiles of its own in the schedule that covers the
// two reduced ones, so it is served by a schedule that materialises the score
// matrix and walks it three times: once to write it, once to normalise it, once
// to consume it.  A 32-batch 32-head 4096-token call spends 20 ms on those
// three trips while the same attention over the same operands takes 0.36 ms
// when the scores are never written out at all.
//
// The schedule below keeps the scores on the chip.  A query tile stays resident
// while the key axis is walked in tiles; each tile's scores are folded into the
// running row maximum and row sum and accumulated into the output before the
// next tile is loaded, so the only things that cross the bus are the operands
// and the result.  That is the whole of the difference.
//
// The tile shape is set by what a block may hold on its chip.  A 32-row query
// tile against a 32-key tile and a 128-wide head is a query tile, one operand
// tile and a score tile, which is 36 KB and fits the default budget; a wider
// tile would need the opt-in the block would then have to ask for on every
// launch, and the operand tiles are what makes it grow.
constexpr int kWideQueryTile = 32;
constexpr int kWideKeyTile = 32;
constexpr int kWideTileD = 128;
// Two warps, each owning half the query tile.  The output tile lives in the
// owning warp's registers, so the warp count is what sets the register budget
// for the output: more warps means fewer rows each, and the rows are spread
// over the head width by lane.
constexpr int kWideWarps = 2;
constexpr int kWideRowsPerWarp = kWideQueryTile / kWideWarps;
constexpr int kWideColsPerLane = kWideTileD / 32;

// The key and the value are read through one buffer at different times: the
// score product needs the key and is finished with it before the value product
// needs the value, so the buffer holds both and the shared allocation is a
// tile of query, a tile of key and a score tile rather than two operand tiles.
// The score tile is rewritten in place with the probability once the row
// statistics are known, so the probability costs nothing.
struct TpWideShared {
  float q[kWideQueryTile][kWideTileD];
  float kv[kWideKeyTile][kWideTileD];
  float score[kWideQueryTile][kWideKeyTile];
  float row_max[kWideQueryTile];
  float row_sum[kWideQueryTile];
  float row_alpha[kWideQueryTile];
};

__global__ void sdpa_wide_flash_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Tq, int64_t Tkv, int64_t D,
    float scale, bool is_causal) {
  constexpr int q_tile = kWideQueryTile;
  constexpr int k_tile = kWideKeyTile;
  constexpr int tile_d = kWideTileD;
  constexpr int warps = kWideWarps;
  constexpr int threads = warps * 32;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;
  constexpr float log2e = 1.4426950408889634f;

  extern __shared__ unsigned char smem_raw[];
  TpWideShared& smem = *reinterpret_cast<TpWideShared*>(smem_raw);
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t bh = static_cast<int64_t>(blockIdx.x);
  const int64_t q0 = static_cast<int64_t>(blockIdx.y) * q_tile;
  // A group of query heads reads one key head, so the head pair a block serves
  // is named once here and every offset below follows from it.
  const int64_t hq = bh % Hq;
  const int64_t hk = hq / (Hq / Hkv);
  const int64_t b = bh / Hq;
  const int64_t q_base = (b * Hq + hq) * Tq * D;
  const int64_t kv_base = (b * Hkv + hk) * Tkv * D;

  for (int idx = thread; idx < q_tile * tile_d; idx += threads) {
    const int qr = idx / tile_d;
    const int d = idx % tile_d;
    const int64_t qg = q0 + qr;
    smem.q[qr][d] = (qg < Tq && d < D) ? q[q_base + qg * D + d] : 0.f;
  }
  for (int idx = thread; idx < q_tile; idx += threads) {
    smem.row_max[idx] = -INFINITY;
    smem.row_sum[idx] = 0.f;
    smem.row_alpha[idx] = 0.f;
  }
  __syncthreads();

  // The largest query this block owns is q0 + q_tile - 1, and a causal row t
  // admits keys 0..t, so no key beyond that is ever needed.  A non-causal walk
  // covers the whole context.
  const int64_t last_k = is_causal ? min(Tkv, q0 + q_tile) : Tkv;

  // The output is carried in registers as a plain float tile rather than a
  // tensor-core fragment: the wide precision has no fragment form here, and a
  // tile this size is cheaper in registers than a round trip through the chip.
  // The block covers the whole query tile, so each thread owns a slice of it.
  float acc[kWideRowsPerWarp][kWideColsPerLane];
#pragma unroll
  for (int r = 0; r < kWideRowsPerWarp; ++r)
#pragma unroll
    for (int c = 0; c < kWideColsPerLane; ++c) acc[r][c] = 0.f;

  for (int64_t k0 = 0; k0 < last_k; k0 += k_tile) {
    // The key lands in the shared buffer and is consumed by the score product;
    // the value replaces it afterwards and is consumed by the second product.
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      smem.kv[kr][d] = (kg < Tkv && d < D) ? k[kv_base + kg * D + d] : 0.f;
    }
    __syncthreads();

    // QK^T.  A warp owns a strip of the query tile and a lane owns a strip of
    // the key tile, so every (row, key) score is produced by exactly one lane
    // and no two lanes write the same slot.  The product is a register dot
    // product, which is what a wide precision without a fragment form comes
    // down to.
    if (warp < warps) {
      const int qr0 = warp * kWideRowsPerWarp;
      for (int qr = qr0; qr < qr0 + kWideRowsPerWarp; ++qr) {
        for (int n = lane; n < k_tile; n += 32) {
          float total = 0.f;
          for (int d0 = 0; d0 < tile_d; ++d0)
            total += smem.q[qr][d0] * smem.kv[n][d0];
          smem.score[qr][n] = total;
        }
      }
    }
    __syncthreads();

    // The value is fetched once the key is dead, so it takes the same buffer
    // and the shared allocation never holds both.
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      smem.kv[kr][d] = (kg < Tkv && d < D) ? v[kv_base + kg * D + d] : 0.f;
    }
    __syncthreads();

    // Online softmax over this key tile, folded into the running row state.
    // The score tile is rewritten in place with the probability, so the value
    // product below reads a probability it does not have to be given twice.
    if (warp < warps) {
      const int qr0 = warp * kWideRowsPerWarp;
      for (int qr = qr0; qr < qr0 + kWideRowsPerWarp; ++qr) {
        const int64_t qg = q0 + qr;
        float maximum = -INFINITY;
        for (int kk = lane; kk < k_tile; kk += 32) {
          const int64_t kg = k0 + kk;
          float score = smem.score[qr][kk] * scale;
          if ((is_causal && kg > qg) || kg >= Tkv || qg >= Tq) score = -INFINITY;
          maximum = max(maximum, score);
        }
        maximum = warpReduceMax(maximum);
        maximum = __shfl_sync(full_mask, maximum, 0);
        if (maximum == -INFINITY) {
          // A row with no admissible key in this tile contributes nothing, and
          // its probability slots are zeroed so the value product adds zero.
          for (int kk = lane; kk < k_tile; kk += 32) smem.score[qr][kk] = 0.f;
          continue;
        }

        const float old_max = smem.row_max[qr];
        const float new_max = max(old_max, maximum);
        const float alpha =
            isfinite(old_max) ? exp2f((old_max - new_max) * log2e) : 0.f;
        float partial_sum = 0.f;
        for (int kk = lane; kk < k_tile; kk += 32) {
          const int64_t kg = k0 + kk;
          const float score = smem.score[qr][kk] * scale;
          const float p = ((is_causal && kg > qg) || kg >= Tkv || qg >= Tq)
              ? 0.f
              : exp2f((score - new_max) * log2e);
          smem.score[qr][kk] = p;
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

    // Rescale the carried output to the new maximum, then accumulate P@V.
    // The value product stays inside the warp that owns the row, because the
    // output tile it accumulates into lives in that warp's registers.  A lane
    // owns a slice of the head width for every key, so no column is touched by
    // two lanes and no cross-lane reduction is needed.
    if (warp < warps) {
      const int qr0 = warp * kWideRowsPerWarp;
#pragma unroll
      for (int r = 0; r < kWideRowsPerWarp; ++r) {
        const float alpha = smem.row_alpha[qr0 + r];
#pragma unroll
        for (int c = 0; c < kWideColsPerLane; ++c) acc[r][c] *= alpha;
      }
      for (int r = 0; r < kWideRowsPerWarp; ++r) {
        const int qr = qr0 + r;
        const int d0 = lane * kWideColsPerLane;
#pragma unroll
        for (int c = 0; c < kWideColsPerLane; ++c) {
          float total = 0.f;
          for (int n = 0; n < k_tile; ++n)
            total += smem.score[qr][n] * smem.kv[n][d0 + c];
          acc[r][c] += total;
        }
      }
    }
    __syncthreads();
  }

  // Normalize from the register tile straight into the result.  A row whose
  // keys were all masked away has no total and is left at zero.
  if (warp < warps) {
    const int qr0 = warp * kWideRowsPerWarp;
    for (int r = 0; r < kWideRowsPerWarp; ++r) {
      const int qr = qr0 + r;
      const int64_t qg = q0 + qr;
      if (qg >= Tq) continue;
      const float total = smem.row_sum[qr];
      const float inverse = total > 0.f ? 1.f / total : 0.f;
      const int d0 = lane * kWideColsPerLane;
#pragma unroll
      for (int c = 0; c < kWideColsPerLane; ++c) {
        if (d0 + c < D) out[q_base + qg * D + (d0 + c)] = acc[r][c] * inverse;
      }
    }
  }
}

// The fused wide-precision schedule.  A query tile stays on the chip while the
// key axis is walked, so the operands and the result are the only things that
// cross the bus.
//
// It covers the two token counts being independent, grouped heads, and the
// causal bound being the query row itself, which is what makes a context longer
// than the query a shape here rather than a special case.  The head width is
// the tile width, so a call of any other width is not one this answers.
Tensor sdpa_wide_tiled_cuda(const Tensor& query, const Tensor& key,
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
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
    TP_THROW(RuntimeError,
             "the fused wide-precision schedule takes 4-dimensional inputs");
  }
  if (D != kWideTileD || k.size(3) != D || v.size(3) != D) {
    TP_THROW(RuntimeError,
             "the fused wide-precision schedule takes one head width");
  }
  if (k.size(0) != B || v.size(0) != B) {
    TP_THROW(RuntimeError,
             "query, key and value must agree on the batch size");
  }
  if (k.size(2) != Tkv || v.size(2) != Tkv) {
    TP_THROW(RuntimeError, "key and value must agree on the context length");
  }
  if (Hkv == 0 || Hq % Hkv != 0) {
    TP_THROW(RuntimeError,
             "the context head count must divide the query head count");
  }
  if (B == 0 || Hq == 0 || Tq == 0 || D == 0) {
    return Tensor::empty({B, Hq, Tq, D}, DType::Float32, q.device());
  }

  Tensor out = Tensor::empty({B, Hq, Tq, D}, DType::Float32, q.device());
  const int64_t q_blocks = (Tq + kWideQueryTile - 1) / kWideQueryTile;
  const size_t smem_bytes = sizeof(TpWideShared);
  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  sdpa_wide_flash_kernel<<<
      dim3(static_cast<unsigned>(B * Hq), static_cast<unsigned>(q_blocks)),
      kWideWarps * 32, smem_bytes, getCurrentCUDAStream().stream()>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), B, Hq, Hkv, Tq, Tkv, D, scale, is_causal);
  TP_WIDE_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace cuda
}  // namespace tensorplay
