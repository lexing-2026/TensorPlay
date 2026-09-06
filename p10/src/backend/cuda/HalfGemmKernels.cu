// Half-precision GEMV for memory-bound skinny products: one output axis of
// size 1 (matrix @ vector) or a small activation batch (up to 8 rows) against
// a large output axis.  The classic BLAS GEMM entry routes these shapes to
// tile kernels sized for square workloads and measures two orders of
// magnitude below the copy bandwidth on integrated GPUs, so these shapes run
// through dedicated kernels here:
//
//   rows kernel  - weight rows contiguous (row-major (rows, K) storage):
//                  out[b, n] = alpha * dot(w[n, :], x[b, :]) + beta * out[b, n]
//                  One block per output row; lanes stride over 16-byte
//                  packets (8 storage values) of weights and activations,
//                  accumulate in fp32, and the block is reduced through wave
//                  shuffles plus a shared-memory pass.  Reduction lengths
//                  that do not pack into whole packets run through a
//                  scalar-load variant of the same schedule.
//
//   cols kernel  - weight columns strided (row-major (K, N) storage, the
//                  reduction walks the non-contiguous axis):
//                  out[b, n] = alpha * dot(x[b, :], mat[:, n]) + beta * out[b, n]
//                  One thread per output column; at each reduction step the
//                  warp reads N-consecutive values (coalesced), and four
//                  independent fp32 accumulators per column break the
//                  dependent fma chain.
//
// The fp32-accumulate contract matches the BLAS entry's compute type for
// reduced-precision inputs (alpha/beta applied in fp32, result stored once).

#include "CudaGemm.h"
#include "CUDARuntime.h"
#include "DType.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kHalfGemvThreads = 256;
constexpr int kHalfGemvMaxBatch = 8;

// Wave width is probed on the host once and passed as a kernel argument.
inline int half_gemv_wave_size() {
    static int wave = []() {
        int dev = 0, lanes = 32;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&lanes, cudaDevAttrWarpSize, dev);
        return lanes > 0 ? lanes : 32;
    }();
    return wave;
}

// Epilogue store.  When beta is zero the prior contents of the output are
// ignored by contract (they may be uninitialized arbitrary bits, including
// NaN patterns, which would poison the result through 0 * NaN), so the old
// value is folded in only when beta is non-zero.
template <typename storage_t>
__device__ __forceinline__ void half_gemv_store(storage_t* o, float total,
                                                float alpha, float beta) {
    const float base = (beta == 0.f) ? 0.f : static_cast<float>(*o);
    *o = storage_t(alpha * total + beta * base);
}

// Block-wide reduction of up to kHalfGemvMaxBatch accumulators at once.
// Wave partials for every batch item land in shared memory behind a single
// barrier, the first wave folds them, and one trailing barrier publishes the
// results in scratch[0 .. batch-1] -- two barriers total instead of two per
// batch item.  The caller must not touch acc[] afterwards.
template <int HW_WAVE>
__device__ __forceinline__ void half_gemv_block_sum_batch(
    float* acc, int batch, float* scratch) {
    constexpr int kNumWaves = kHalfGemvThreads / HW_WAVE;
    const int lane = static_cast<int>(threadIdx.x) & (HW_WAVE - 1);
    const int wid = static_cast<int>(threadIdx.x) / HW_WAVE;
    constexpr unsigned long long kMask =
        HW_WAVE == 64 ? 0xffffffffffffffffull : 0xffffffffull;
#pragma unroll
    for (int offset = HW_WAVE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int b = 0; b < kHalfGemvMaxBatch; ++b) {
            if (b < batch) acc[b] += __shfl_xor_sync(kMask, acc[b], offset, HW_WAVE);
        }
    }
    if (lane == 0) {
        for (int b = 0; b < batch; ++b) {
            scratch[wid * kHalfGemvMaxBatch + b] = acc[b];
        }
    }
    __syncthreads();
    // Second pass: fold the per-wave partials.  Slot [0*MAXB + b] is
    // overwritten with the total for item b; all reads for item b happen
    // before the write through the first shuffle of the reduction chain.
    __syncthreads();
    if (wid == 0) {
        for (int b = 0; b < batch; ++b) {
            float value = (lane < kNumWaves)
                              ? scratch[lane * kHalfGemvMaxBatch + b]
                              : 0.f;
#pragma unroll
            for (int offset = HW_WAVE / 2; offset > 0; offset /= 2) {
                value += __shfl_xor_sync(kMask, value, offset, HW_WAVE);
            }
            if (lane == 0) scratch[b] = value;
        }
    }
    __syncthreads();
}

#define TP_HALF_GEMV_BLOCK_SUM(acc, batch, scratch, wave_size)           \
    (wave_size == 64                                                     \
         ? half_gemv_block_sum_batch<64>(acc, batch, scratch)            \
         : half_gemv_block_sum_batch<32>(acc, batch, scratch))

// Rows kernel: one block per weight row; K must be a multiple of 8 so every
// row base stays 16-byte aligned (gated by the host launcher).
template <typename storage_t>
__global__ void __launch_bounds__(kHalfGemvThreads)
half_gemv_rows_kernel(storage_t* __restrict__ out,
                      const storage_t* __restrict__ w,
                      const storage_t* __restrict__ x,
                      int64_t K, int64_t rows, int batch,
                      float alpha, float beta, int wave_size) {
    __shared__ float scratch[kHalfGemvThreads / 32 * kHalfGemvMaxBatch];
    const int64_t n = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    const uint4* w_row = reinterpret_cast<const uint4*>(w + n * K);
    const int64_t npackets = K / 8;

    float acc[kHalfGemvMaxBatch];
#pragma unroll
    for (int b = 0; b < kHalfGemvMaxBatch; ++b) acc[b] = 0.f;

    if (batch == 1) {
        const uint4* x_row = reinterpret_cast<const uint4*>(x);
        // Two packets in flight per lane keep an independent fma chain
        // running while the loads for the next stride are outstanding.
        int64_t p = tid;
        for (; p + nthreads < npackets; p += 2 * nthreads) {
            const uint4 wp0 = w_row[p];
            const uint4 xp0 = x_row[p];
            const uint4 wp1 = w_row[p + nthreads];
            const uint4 xp1 = x_row[p + nthreads];
            const storage_t* wv0 = reinterpret_cast<const storage_t*>(&wp0);
            const storage_t* xv0 = reinterpret_cast<const storage_t*>(&xp0);
            const storage_t* wv1 = reinterpret_cast<const storage_t*>(&wp1);
            const storage_t* xv1 = reinterpret_cast<const storage_t*>(&xp1);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[0] = fmaf(static_cast<float>(wv0[e]),
                              static_cast<float>(xv0[e]), acc[0]);
                acc[1] = fmaf(static_cast<float>(wv1[e]),
                              static_cast<float>(xv1[e]), acc[1]);
            }
        }
        for (; p < npackets; p += nthreads) {
            const uint4 wp = w_row[p];
            const uint4 xp = x_row[p];
            const storage_t* wv = reinterpret_cast<const storage_t*>(&wp);
            const storage_t* xv = reinterpret_cast<const storage_t*>(&xp);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[0] = fmaf(static_cast<float>(wv[e]),
                              static_cast<float>(xv[e]), acc[0]);
            }
        }
        acc[0] += acc[1];
        TP_HALF_GEMV_BLOCK_SUM(acc, 1, scratch, wave_size);
        if (tid == 0) {
            half_gemv_store<storage_t>(out + n, scratch[0], alpha, beta);
        }
        return;
    }

    for (int64_t p = tid; p < npackets; p += nthreads) {
        const uint4 wp = w_row[p];
        const storage_t* wv = reinterpret_cast<const storage_t*>(&wp);
        for (int b = 0; b < batch; ++b) {
            const uint4 xp =
                reinterpret_cast<const uint4*>(x + static_cast<int64_t>(b) * K)[p];
            const storage_t* xv = reinterpret_cast<const storage_t*>(&xp);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[b] = fmaf(static_cast<float>(wv[e]),
                              static_cast<float>(xv[e]), acc[b]);
            }
        }
    }
    TP_HALF_GEMV_BLOCK_SUM(acc, batch, scratch, wave_size);
    if (tid == 0) {
        for (int b = 0; b < batch; ++b) {
            half_gemv_store<storage_t>(
                out + static_cast<int64_t>(b) * rows + n, scratch[b], alpha,
                beta);
        }
    }
}

// Scalar-load variant of the rows schedule for reduction lengths that do not
// pack into whole 16-byte packets (K < 8 or K not a multiple of 8): identical
// block-per-row decomposition and fp32 accumulate, unit-stride element walk,
// no alignment requirement.
template <typename storage_t>
__global__ void __launch_bounds__(kHalfGemvThreads)
half_gemv_rows_generic_kernel(storage_t* __restrict__ out,
                              const storage_t* __restrict__ w,
                              const storage_t* __restrict__ x,
                              int64_t K, int64_t rows, int batch,
                              float alpha, float beta, int wave_size) {
    __shared__ float scratch[kHalfGemvThreads / 32 * kHalfGemvMaxBatch];
    const int64_t n = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    const storage_t* w_row = w + n * K;

    float acc[kHalfGemvMaxBatch];
#pragma unroll
    for (int b = 0; b < kHalfGemvMaxBatch; ++b) acc[b] = 0.f;

    if (batch == 1) {
        int64_t k = tid;
        for (; k + nthreads < K; k += 2 * nthreads) {
            acc[0] = fmaf(static_cast<float>(w_row[k]),
                          static_cast<float>(x[k]), acc[0]);
            acc[1] = fmaf(static_cast<float>(w_row[k + nthreads]),
                          static_cast<float>(x[k + nthreads]), acc[1]);
        }
        for (; k < K; k += nthreads) {
            acc[0] = fmaf(static_cast<float>(w_row[k]),
                          static_cast<float>(x[k]), acc[0]);
        }
        acc[0] += acc[1];
        TP_HALF_GEMV_BLOCK_SUM(acc, 1, scratch, wave_size);
        if (tid == 0) {
            half_gemv_store<storage_t>(out + n, scratch[0], alpha, beta);
        }
        return;
    }

    for (int64_t k = tid; k < K; k += nthreads) {
        const float wv = static_cast<float>(w_row[k]);
        for (int b = 0; b < batch; ++b) {
            acc[b] = fmaf(wv,
                          static_cast<float>(
                              x[static_cast<int64_t>(b) * K + k]),
                          acc[b]);
        }
    }
    TP_HALF_GEMV_BLOCK_SUM(acc, batch, scratch, wave_size);
    if (tid == 0) {
        for (int b = 0; b < batch; ++b) {
            half_gemv_store<storage_t>(
                out + static_cast<int64_t>(b) * rows + n, scratch[b], alpha,
                beta);
        }
    }
}

// Cols kernel: one thread per output column; four partial accumulators per
// activation row keep the fma chains independent across the reduction walk.
template <typename storage_t>
__global__ void __launch_bounds__(kHalfGemvThreads)
half_gemv_cols_kernel(storage_t* __restrict__ out,
                      const storage_t* __restrict__ mat,
                      const storage_t* __restrict__ x,
                      int64_t K, int64_t N, int batch,
                      float alpha, float beta) {
    const int64_t col =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float acc[kHalfGemvMaxBatch][4];
    for (int b = 0; b < batch; ++b) {
        acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
    }
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        const float m0 = static_cast<float>(mat[(k + 0) * N + col]);
        const float m1 = static_cast<float>(mat[(k + 1) * N + col]);
        const float m2 = static_cast<float>(mat[(k + 2) * N + col]);
        const float m3 = static_cast<float>(mat[(k + 3) * N + col]);
        for (int b = 0; b < batch; ++b) {
            const storage_t* xr = x + static_cast<int64_t>(b) * K + k;
            acc[b][0] = fmaf(static_cast<float>(xr[0]), m0, acc[b][0]);
            acc[b][1] = fmaf(static_cast<float>(xr[1]), m1, acc[b][1]);
            acc[b][2] = fmaf(static_cast<float>(xr[2]), m2, acc[b][2]);
            acc[b][3] = fmaf(static_cast<float>(xr[3]), m3, acc[b][3]);
        }
    }
    for (; k < K; ++k) {
        const float m = static_cast<float>(mat[k * N + col]);
        for (int b = 0; b < batch; ++b) {
            acc[b][0] = fmaf(static_cast<float>(x[b * K + k]), m, acc[b][0]);
        }
    }
    for (int b = 0; b < batch; ++b) {
        const float total = (acc[b][0] + acc[b][1]) + (acc[b][2] + acc[b][3]);
        half_gemv_store<storage_t>(out + static_cast<int64_t>(b) * N + col,
                                   total, alpha, beta);
    }
}

inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename storage_t>
void launch_half_gemv_rows(const Tensor& w, const Tensor& x, Tensor& out,
                           int64_t rows, int batch, double alpha,
                           double beta, cudaStream_t stream) {
    const int wave = half_gemv_wave_size();
    half_gemv_rows_kernel<storage_t>
        <<<static_cast<unsigned>(rows), kHalfGemvThreads, 0, stream>>>(
            out.data_ptr<storage_t>(), w.data_ptr<storage_t>(),
            x.data_ptr<storage_t>(), w.size(1), rows, batch,
            static_cast<float>(alpha), static_cast<float>(beta), wave);
}

template <typename storage_t>
void launch_half_gemv_rows_generic(const Tensor& w, const Tensor& x,
                                   Tensor& out, int64_t rows, int batch,
                                   double alpha, double beta,
                                   cudaStream_t stream) {
    const int wave = half_gemv_wave_size();
    half_gemv_rows_generic_kernel<storage_t>
        <<<static_cast<unsigned>(rows), kHalfGemvThreads, 0, stream>>>(
            out.data_ptr<storage_t>(), w.data_ptr<storage_t>(),
            x.data_ptr<storage_t>(), w.size(1), rows, batch,
            static_cast<float>(alpha), static_cast<float>(beta), wave);
}

template <typename storage_t>
void launch_half_gemv_cols(const Tensor& mat, const Tensor& x, Tensor& out,
                           int64_t N, int batch, double alpha, double beta,
                           cudaStream_t stream) {
    const unsigned blocks =
        static_cast<unsigned>((N + kHalfGemvThreads - 1) / kHalfGemvThreads);
    half_gemv_cols_kernel<storage_t>
        <<<blocks, kHalfGemvThreads, 0, stream>>>(
            out.data_ptr<storage_t>(), mat.data_ptr<storage_t>(),
            x.data_ptr<storage_t>(), mat.size(0), N, batch,
            static_cast<float>(alpha), static_cast<float>(beta));
}

}  // namespace

// Shape dispatch, expressed in the logical GEMM frame result(M, N) =
// alpha * self(M, K) @ other(K, N) + beta * result:
//
//   N == 1 (matrix @ vector): the weight is `self`; rows kernel with
//       rows = M, batch = 1 when self is row-major contiguous.
//
//   other stored as its (N, K) transpose (the linear-layer pattern
//   x @ W.t()): the weight rows are `other`'s contiguous rows; rows kernel
//       with rows = N, batch = M (M capped at kHalfGemvMaxBatch).
//
//   other row-major (K, N) with a small activation batch: cols kernel with
//       N columns, batch = M.
//
// Row kernels pick the packet schedule when K packs into whole 16-byte
// packets with aligned bases, and the scalar schedule otherwise.  Anything
// else (larger batches, exotic strides) stays on the caller's GEMM path;
// returns false in that case.
bool try_half_gemv(const Tensor& self, const Tensor& other, Tensor& result,
                   double alpha, double beta, bool other_transposed) {
    const DType dt = self.dtype();
    if (dt != DType::Float16 && dt != DType::BFloat16) return false;
    if (other.dtype() != dt || result.dtype() != dt) return false;
    if (self.dim() != 2 || other.dim() != 2 || result.dim() != 2) return false;

    const int64_t M = self.size(0), K = self.size(1);
    const int64_t N = other.size(1);
    if (other.size(0) != K || result.size(0) != M || result.size(1) != N) {
        return false;
    }
    if (M == 0 || N == 0 || K == 0) return false;
    if (!result.is_contiguous()) return false;
    if (M > kHalfGemvMaxBatch && N != 1) return false;

    cudaStream_t stream = getCurrentCUDAStream().stream();

    // One packet covers 8 storage values; per-row alignment follows from the
    // row length once the base pointers are 16-byte aligned.
    const bool packet_ok = K >= 8 && (K & 7) == 0;

    if (N == 1) {
        if (!self.is_contiguous() || !other.is_contiguous()) return false;
        const bool aligned =
            aligned16(self.data_ptr()) && aligned16(other.data_ptr());
        if (dt == DType::Float16) {
            if (packet_ok && aligned) {
                launch_half_gemv_rows<Half>(self, other, result, M, 1, alpha,
                                            beta, stream);
            } else {
                launch_half_gemv_rows_generic<Half>(self, other, result, M, 1,
                                                    alpha, beta, stream);
            }
        } else {
            if (packet_ok && aligned) {
                launch_half_gemv_rows<BFloat16>(self, other, result, M, 1,
                                                alpha, beta, stream);
            } else {
                launch_half_gemv_rows_generic<BFloat16>(self, other, result,
                                                        M, 1, alpha, beta,
                                                        stream);
            }
        }
        return true;
    }

    if (!self.is_contiguous()) return false;

    if (other_transposed) {
        // `other` is a live (K, N) view of contiguous (N, K) row-major
        // storage: its rows are the weight rows the rows kernel walks.
        if (other.stride(0) != 1 || other.stride(1) != K ||
            other.size(0) != N) {
            return false;
        }
        const bool aligned =
            aligned16(other.data_ptr()) && aligned16(self.data_ptr());
        if (dt == DType::Float16) {
            if (packet_ok && aligned) {
                launch_half_gemv_rows<Half>(other, self, result, N,
                                            static_cast<int>(M), alpha, beta,
                                            stream);
            } else {
                launch_half_gemv_rows_generic<Half>(other, self, result, N,
                                                    static_cast<int>(M), alpha,
                                                    beta, stream);
            }
        } else {
            if (packet_ok && aligned) {
                launch_half_gemv_rows<BFloat16>(other, self, result, N,
                                                static_cast<int>(M), alpha,
                                                beta, stream);
            } else {
                launch_half_gemv_rows_generic<BFloat16>(other, self, result, N,
                                                        static_cast<int>(M),
                                                        alpha, beta, stream);
            }
        }
        return true;
    }

    if (other.is_contiguous()) {
        if (dt == DType::Float16) {
            launch_half_gemv_cols<Half>(other, self, result, N,
                                        static_cast<int>(M), alpha, beta,
                                        stream);
        } else {
            launch_half_gemv_cols<BFloat16>(other, self, result, N,
                                            static_cast<int>(M), alpha, beta,
                                            stream);
        }
        return true;
    }
    return false;
}

}  // namespace cuda
}  // namespace tensorplay
