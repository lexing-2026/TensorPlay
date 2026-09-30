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
// The tile is a template parameter because a causal call's work is set by its
// query length, not by its context length: row t admits keys 0..t, so a call of
// one query row over a four-thousand-token context has one key to read.  A
// fixed tile would read a whole tile of keys for that row and then mask all but
// one of them away, which is thirty-two times the memory traffic for a result
// that is a copy of one value.  The wide tile is for a call that fills it, and
// the one-row tile is for a call that does not.
constexpr int kWideTileD = 128;
// The two tile axes are named separately: the query tile and the key tile do
// not have to be square, and for a causal call they should not be.  A query
// tile of a hundred and twenty-eight rows against a key tile of thirty-two
// keeps the key axis short -- the score tile and the key tile stay small -- and
// lets one block own a long stretch of the query axis, which is what cuts the
// number of times the context's early tiles are re-read.
constexpr int kWideQTileLarge = 128;
constexpr int kWideKTileLarge = 32;
// A tile of at least sixteen rows is walked by several warps so that a warp
// scheduler always has another warp ready while one waits on the chip: the
// walk is a chain of chip loads, and a block of one or two warps leaves the
// scheduler idle behind every load.  A narrower tile has fewer rows to give
// each warp, so it takes fewer warps; a one-row tile takes one.
template <int kQTile>
constexpr int kWideWarps = kQTile >= 32 ? 8 : kQTile >= 16 ? 4 : kQTile >= 4 ? 2 : 1;
template <int kQTile>
constexpr int kWideRowsPerWarp = kQTile / kWideWarps<kQTile>;
constexpr int kWideColsPerLane = kWideTileD / 32;

// The key and the value are read through one buffer at different times: the
// score product needs the key and is finished with it before the value product
// needs the value, so the buffer holds both and the shared allocation is a
// tile of query, a tile of key and a score tile rather than two operand tiles.
// The score tile is rewritten in place with the probability once the row
// statistics are known, so the probability costs nothing.
//
// The key rows are stored through a column permutation that folds the row
// index into the low five bits of the column.  A key sits in its own row and
// a row is 128 floats wide, which is four bank widths, so under the plain
// layout a sweep that reads one column of every key row finds every lane in
// one bank and pays the full conflict on every load; under the permutation the
// same sweep lands on as many banks as it has lanes.  The value rows keep the
// plain layout: their product reads one row per lane step, and giving each
// lane columns 32 floats apart serves that sweep without a permutation.
template <int kQTile, int kKTile>
struct TpWideShared {
  float q[kQTile][kWideTileD];
  float kv[kKTile][kWideTileD];
  float score[kQTile][kKTile];
  float row_max[kQTile];
  float row_sum[kQTile];
  float row_alpha[kQTile];
};

// The low five bits of the column move with the row: storing column c of row
// n at c ^ (n & 31) puts a fixed-column sweep across rows on distinct banks.
__device__ __forceinline__ int wide_kv_col(int col, int row) {
  return col ^ (row & 31);
}

template <int kQTile, int kKTile>
__global__ void sdpa_wide_flash_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Tq, int64_t Tkv, int64_t D,
    float scale, bool is_causal) {
  constexpr int q_tile = kQTile;
  constexpr int k_tile = kKTile;
  constexpr int tile_d = kWideTileD;
  constexpr int warps = kWideWarps<kQTile>;
  constexpr int rows_per_warp = kWideRowsPerWarp<kQTile>;
  constexpr int threads = warps * 32;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;
  constexpr float log2e = 1.4426950408889634f;

  extern __shared__ unsigned char smem_raw[];
  TpWideShared<kQTile, kKTile>& smem =
      *reinterpret_cast<TpWideShared<kQTile, kKTile>*>(smem_raw);
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

  // A causal row t admits keys 0..t, so the key tiles entirely past its bound
  // contribute nothing to it.  A warp whose rows all sit past a tile's bound
  // skips the whole tile for those rows: the score row is zeroed so the value
  // product adds nothing, and the carried row state survives untouched.  This
  // is what keeps a tall query tile from re-paying the early context for rows
  // that cannot see it; without it a block owning rows far above the diagonal
  // walks the tiles below for them as well, and the walk's cost grows with the
  // rectangle rather than the triangle.
  // The output is carried in registers as a plain float tile rather than a
  // tensor-core fragment: the wide precision has no fragment form here, and a
  // tile this size is cheaper in registers than a round trip through the chip.
  // The block covers the whole query tile, so each thread owns a slice of it.
  float acc[rows_per_warp][kWideColsPerLane];
#pragma unroll
  for (int r = 0; r < rows_per_warp; ++r)
#pragma unroll
    for (int c = 0; c < kWideColsPerLane; ++c) acc[r][c] = 0.f;

  for (int64_t k0 = 0; k0 < last_k; k0 += k_tile) {
    // The key lands in the shared buffer and is consumed by the score product;
    // the value replaces it afterwards and is consumed by the second product.
    for (int idx = thread; idx < k_tile * tile_d; idx += threads) {
      const int kr = idx / tile_d;
      const int d = idx % tile_d;
      const int64_t kg = k0 + kr;
      smem.kv[kr][wide_kv_col(d, kr)] = (kg < Tkv && d < D) ? k[kv_base + kg * D + d] : 0.f;
    }
    __syncthreads();

    // QK^T.  A warp owns a strip of the query tile and a lane owns a strip of
    // the key tile, so every (row, key) score is produced by exactly one lane
    // and no two lanes write the same slot.  The product is a register dot
    // product, which is what a wide precision without a fragment form comes
    // down to.  The key read goes through the permutation the key was stored
    // under, so a lane's sweep over the key rows reads across the banks
    // instead of marching down one.
    if (warp < warps) {
      const int qr0 = warp * rows_per_warp;
      for (int qr = qr0; qr < qr0 + rows_per_warp; ++qr) {
        const int64_t qg = q0 + qr;
        // A row whose bound is below this tile's first key has no score to
        // compute and no state to carry: zero its slots and step past it.
        if (is_causal && qg < k0) {
          for (int n = lane; n < k_tile; n += 32) smem.score[qr][n] = 0.f;
          continue;
        }
        for (int n = lane; n < k_tile; n += 32) {
          float total = 0.f;
          for (int d0 = 0; d0 < tile_d; ++d0)
            total += smem.q[qr][d0] * smem.kv[n][wide_kv_col(d0, n)];
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
      const int qr0 = warp * rows_per_warp;
      for (int qr = qr0; qr < qr0 + rows_per_warp; ++qr) {
        const int64_t qg = q0 + qr;
        // A row the score pass zeroed has nothing to fold: its probability
        // slots are already zero and its carried state stays as it was.
        if (is_causal && qg < k0) continue;
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
      const int qr0 = warp * rows_per_warp;
#pragma unroll
      for (int r = 0; r < rows_per_warp; ++r) {
        const int qr = qr0 + r;
        // A row the walk skips carries no new maximum, so its carried output
        // is rescaled by one rather than by whatever the last walked tile
        // left in its alpha slot -- that stale value would wipe the output.
        const float alpha =
            (is_causal && q0 + qr < k0) ? 1.f : smem.row_alpha[qr];
#pragma unroll
        for (int c = 0; c < kWideColsPerLane; ++c) acc[r][c] *= alpha;
      }
      for (int r = 0; r < rows_per_warp; ++r) {
        const int qr = qr0 + r;
        // A row past its bound carries no new probability, so its product
        // would add zero; the rescale above was all it needed.
        if (is_causal && q0 + qr < k0) continue;
        // A lane takes columns a bank apart rather than a run of them: the
        // value row a lane reads at each key step is then one float wide, and
        // a warp's sweep over the key rows sweeps the banks instead of four
        // lanes sharing one.  The column offset also coalesces the final
        // store, which writes the same columns.
#pragma unroll
        for (int c = 0; c < kWideColsPerLane; ++c) {
          float total = 0.f;
          for (int n = 0; n < k_tile; ++n)
            total += smem.score[qr][n] * smem.kv[n][lane + 32 * c];
          acc[r][c] += total;
        }
      }
    }
    __syncthreads();
  }

  // Normalize from the register tile straight into the result.  A row whose
  // keys were all masked away has no total and is left at zero.
  if (warp < warps) {
    const int qr0 = warp * rows_per_warp;
    for (int r = 0; r < rows_per_warp; ++r) {
      const int qr = qr0 + r;
      const int64_t qg = q0 + qr;
      if (qg >= Tq) continue;
      const float total = smem.row_sum[qr];
      const float inverse = total > 0.f ? 1.f / total : 0.f;
#pragma unroll
      for (int c = 0; c < kWideColsPerLane; ++c) {
        if (lane + 32 * c < D)
          out[q_base + qg * D + (lane + 32 * c)] = acc[r][c] * inverse;
      }
    }
  }
}


namespace {

// One row, causal: the row admits key zero alone, so the answer is that key's
// value row.  A thread per output element copies it, and nothing else is read:
// not the query, not the context, not the key.  The reference spends a full
// attention pass to arrive at the same row, which is why a call of this shape is
// the one where staying on the chip stops mattering and not reading at all
// starts to.
__global__ void sdpa_one_row_first_value_kernel(
    const float* __restrict__ v,
    float* __restrict__ out,
    int64_t rows, int64_t D, int64_t kv_stride, int64_t out_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * D) return;
  const int64_t r = idx / D;
  const int64_t d = idx % D;
  out[out_stride * r + d] = v[kv_stride * r + d];
}

// The same copy when query heads are grouped: query head hq reads key head
// hq / (Hq / Hkv), so the row is fetched per head rather than as one run.
__global__ void sdpa_one_row_grouped_value_kernel(
    const float* __restrict__ v,
    float* __restrict__ out,
    int64_t rows, int64_t D, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t Tkv) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * D) return;
  const int64_t r = idx / D;
  const int64_t d = idx % D;
  const int64_t b = r / Hq;
  const int64_t hq = r % Hq;
  const int64_t hk = hq / (Hq / Hkv);
  out[r * D + d] = v[((b * Hkv + hk) * Tkv) * D + d];
}

}  // namespace

Tensor sdpa_one_row_first_value_cuda(const Tensor& q, const Tensor& k,
                                     const Tensor& v, int64_t B, int64_t Hq,
                                     int64_t Hkv, int64_t Tkv, int64_t D) {
  (void)q;
  (void)k;
  (void)Tkv;
  Tensor out = Tensor::empty({B, Hq, 1, D}, DType::Float32, v.device());
  // A group's query heads read one key head, so the row being copied for query
  // head hq is the first row of key head hq / (Hq / Hkv).
  const int64_t group = Hq / Hkv;
  const int64_t rows = B * Hq;
  // The source rows are not contiguous in the query-head order when heads are
  // grouped, so the copy is driven per query head rather than as one run.
  auto stream = getCurrentCUDAStream().stream();
  const int threads = 128;
  const int64_t blocks = (rows * D + threads - 1) / threads;
  if (group == 1) {
    sdpa_one_row_first_value_kernel<<<blocks, threads, 0, stream>>>(
        v.data_ptr<float>(), out.data_ptr<float>(), rows, D, Tkv * D, D);
  } else {
    sdpa_one_row_grouped_value_kernel<<<blocks, threads, 0, stream>>>(
        v.data_ptr<float>(), out.data_ptr<float>(), rows, D, B, Hq, Hkv, Tkv);
  }
  TP_WIDE_CUDA_CHECK(cudaGetLastError());
  return out;
}

namespace {

// One launch per tile shape.  A call is served by the tile that fits its query
// length, which is what keeps a one-row call from reading a whole tile of keys
// and masking all but one of them away.  A long query takes the rectangular
// tile: the score tile and the key tile stay at the key length, while the
// query axis is stretched, so the context's early key tiles are re-read once
// per query tile instead of once per handful of rows.
template <int kQTile, int kKTile>
Tensor sdpa_wide_tiled_launch(const Tensor& q, const Tensor& k,
                              const Tensor& v, int64_t B, int64_t Hq,
                              int64_t Hkv, int64_t Tq, int64_t Tkv, int64_t D,
                              bool is_causal) {
  Tensor out = Tensor::empty({B, Hq, Tq, D}, DType::Float32, q.device());
  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  const int64_t q_blocks = (Tq + kQTile - 1) / kQTile;
  const size_t smem_bytes = sizeof(TpWideShared<kQTile, kKTile>);
  const dim3 grid(static_cast<unsigned>(B * Hq),
                  static_cast<unsigned>(q_blocks));
  const int threads = kWideWarps<kQTile> * 32;
  cudaStream_t stream = getCurrentCUDAStream().stream();
  // The rectangular tile's shared footprint clears the default per-block
  // budget, so the launch opts into the larger one the device allows.
  if (smem_bytes > 48 * 1024) {
    TP_WIDE_CUDA_CHECK(cudaFuncSetAttribute(
        sdpa_wide_flash_kernel<kQTile, kKTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  sdpa_wide_flash_kernel<kQTile, kKTile><<<grid, threads, smem_bytes, stream>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), B, Hq, Hkv, Tq, Tkv, D, scale, is_causal);
  TP_WIDE_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace

// A one-row causal call, answered without reading the context.  Declared apart
// from the schedule below because it is not a schedule: there is no product to
// perform, only a row to copy.
Tensor sdpa_one_row_first_value_cuda(const Tensor& q, const Tensor& k,
                                     const Tensor& v, int64_t B, int64_t Hq,
                                     int64_t Hkv, int64_t Tkv, int64_t D);

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
  // A causal call's work is set by its query length, so the tile is chosen to
  // fit it.  A call too long for the narrow tile is served by the wide one,
  // which walks the query axis in as many passes as it needs.
  // The tile is chosen so that a call's blocks fill the device rather than so
  // that they cover its query length.  One row over a long context is a single
  // row of work no matter how long the context is, and a tile of thirty-two
  // would read thirty-two rows of keys for it; a longer call has enough blocks
  // of its own that the tile only has to be large enough to keep each warp
  // busy, and past that a wider tile costs shared memory without buying
  // occupancy.
  // A one-row causal call has one key admissible: row zero admits key zero, so
  // its softmax is over a single value and its answer is that key's value,
  // whatever the context length.  Reading the context to find that out is the
  // whole cost of the call and there is nothing else in it, so the call is
  // answered by copying the first value row.  A non-causal one-row call has no
  // such bound and is answered by the schedule below.
  if (Tq <= 1 && is_causal) {
    return sdpa_one_row_first_value_cuda(q, k, v, B, Hq, Hkv, Tkv, D);
  }
  if (Tq <= 1) {
    return sdpa_wide_tiled_launch<1, 1>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  if (Tq <= 4) {
    return sdpa_wide_tiled_launch<4, 4>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  if (Tq <= 8) {
    return sdpa_wide_tiled_launch<8, 8>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  if (Tq <= 16) {
    return sdpa_wide_tiled_launch<16, 16>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  // A long query takes the rectangular tile.  The walk's cost for a causal
  // call is the context re-read once per query tile, so the query tile is
  // stretched until the re-reads are paid for by the shorter key walk each
  // carries; past that the score tile grows the shared footprint without
  // shortening anything.
  if (Tq <= 32) {
    return sdpa_wide_tiled_launch<32, 32>(q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
  }
  return sdpa_wide_tiled_launch<kWideQTileLarge, kWideKTileLarge>(
      q, k, v, B, Hq, Hkv, Tq, Tkv, D, is_causal);
}

}  // namespace cuda
}  // namespace tensorplay
