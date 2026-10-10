#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

// ============================================================================
// Layer Normalization (custom kernels, no cuDNN dependency)
//
// Per-row Welford moments combined through warp shuffles and shared memory,
// fp32 accumulation for Half/BFloat16 inputs, fused stats+apply forward pass
// with vectorized loads when N % 4 == 0.
// ============================================================================

namespace layer_norm {

constexpr int kLNThreads = 256;

// Adaptive launch width: small normalized widths waste 3/4 of a 256-thread
// block on strided loads.  Fewer threads per row lets more rows resident.
inline unsigned ln_threads_for(int64_t N) {
    if (N >= 2048) return kLNThreads;
    if (N >= 512) return 128;
    return 64;
}

// Backward needs more warps in flight than the forward: each block stalls on
// three block reductions per row, and wider blocks keep the memory pipe busy
// across those stalls.
inline unsigned ln_bwd_threads_for(int64_t N) {
    if (N >= 512) return kLNThreads;
    if (N >= 128) return 128;
    return 64;
}

// Store with an evict-first hint: the grad-input output is not re-read inside
// the backward pass, so keeping it out of L2 preserves row data for the
// column-gradient kernel that follows.  The hint is advisory: a toolchain
// without the cache-hint intrinsic stores plainly, with the same result.
template <typename V>
__device__ inline void ln_stream_store(V* dst, const V& v) {
#if defined(USE_ROCM)
    *dst = v;
#else
    if constexpr (sizeof(V) == 16) {
        __stcs(reinterpret_cast<uint4*>(dst),
               *reinterpret_cast<const uint4*>(&v));
    } else if constexpr (sizeof(V) == 8) {
        __stcs(reinterpret_cast<uint2*>(dst),
               *reinterpret_cast<const uint2*>(&v));
    } else {
        *dst = v;
    }
#endif
}

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) LNAlignedVec {
    T val[VecSize];
};

template <typename ACC>
struct LNWelford {
    ACC mean;
    ACC m2;
    ACC count;
};

__device__ inline float ln_rsqrt(float v) { return rsqrtf(v); }
__device__ inline double ln_rsqrt(double v) { return 1.0 / ::sqrt(v); }

template <typename ACC>
__device__ inline LNWelford<ACC> ln_welford_online(ACC val, const LNWelford<ACC>& curr) {
    ACC delta = val - curr.mean;
    ACC new_count = curr.count + ACC(1);
    ACC new_mean = curr.mean + delta / new_count;
    return {new_mean, curr.m2 + delta * (val - new_mean), new_count};
}

template <typename ACC>
__device__ inline LNWelford<ACC> ln_welford_combine(const LNWelford<ACC>& a, const LNWelford<ACC>& b) {
    if (a.count == ACC(0)) return b;
    if (b.count == ACC(0)) return a;
    ACC count = a.count + b.count;
    ACC na = a.count / count;
    ACC nb = b.count / count;
    ACC delta = b.mean - a.mean;
    return {a.mean * na + b.mean * nb,
            a.m2 + b.m2 + delta * delta * a.count * nb,
            count};
}

template <typename ACC>
__device__ inline LNWelford<ACC> ln_warp_reduce(LNWelford<ACC> val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        LNWelford<ACC> other;
        other.mean = __shfl_down_sync(0xffffffffffffffffull, val.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffffffffffffull, val.m2, offset);
        other.count = __shfl_down_sync(0xffffffffffffffffull, val.count, offset);
        val = ln_welford_combine(val, other);
    }
    return val;
}

// Two-stage reduction (warp shuffles, then shared memory across warps).
// On return smem[0] holds the combined value and every thread is in sync.
template <typename ACC>
__device__ inline LNWelford<ACC> ln_block_reduce(LNWelford<ACC> val, LNWelford<ACC>* smem) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
    val = ln_warp_reduce(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (lane < static_cast<int>(blockDim.x >> 5))
        ? smem[lane]
        : LNWelford<ACC>{ACC(0), ACC(0), ACC(0)};
    if (wid == 0) val = ln_warp_reduce(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// Two-sum block reduction.  Warp partials land in smem0/smem1; after a
// single barrier every warp folds all partials redundantly, so each caller
// controls barrier placement.  A caller that reduces again afterwards must
// hand this a buffer pair whose previous contents are no longer being read
// (rotate buffers across consecutive reductions).
template <typename ACC>
__device__ inline void ln_block_reduce2(ACC& v0, ACC& v1, ACC* smem0, ACC* smem1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    if (lane == 0) { smem0[wid] = v0; smem1[wid] = v1; }
    __syncthreads();
    const int nw = static_cast<int>(blockDim.x >> 5);
    v0 = (lane < nw) ? smem0[lane] : ACC(0);
    v1 = (lane < nw) ? smem1[lane] : ACC(0);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    v0 = __shfl_sync(0xffffffffffffffffull, v0, 0);
    v1 = __shfl_sync(0xffffffffffffffffull, v1, 0);
}

// Fused forward: one block per row, Welford stats + normalize in one launch.
// The per-row mean and reciprocal std land in the caller's buffers so the
// backward pass can consume them instead of recomputing.  VEC > 1 requires
// N % VEC == 0 and 16B-aligned row pointers.
template <typename T, typename ACC, int VEC>
__global__ void layer_norm_forward_kernel(
    int64_t N, ACC eps,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ Y,
    ACC* __restrict__ mean_out,
    ACC* __restrict__ rstd_out) {
    __shared__ LNWelford<ACC> smem[kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const T* x_row = X + row * N;
    T* y_row = Y + row * N;

    LNWelford<ACC> wd{ACC(0), ACC(0), ACC(0)};
    if (VEC > 1) {
        using vec_t = LNAlignedVec<T, VEC>;
        const int64_t nvec = N / VEC;
        const vec_t* rowv = reinterpret_cast<const vec_t*>(x_row);
        for (int64_t j = threadIdx.x; j < nvec; j += blockDim.x) {
            vec_t pack = rowv[j];
#pragma unroll
            for (int k = 0; k < VEC; ++k)
                wd = ln_welford_online(static_cast<ACC>(pack.val[k]), wd);
        }
    } else {
        for (int64_t j = threadIdx.x; j < N; j += blockDim.x)
            wd = ln_welford_online(static_cast<ACC>(x_row[j]), wd);
    }
    wd = ln_block_reduce(wd, smem);
    const ACC mean = smem[0].mean;
    const ACC rstd = ln_rsqrt(smem[0].m2 / static_cast<ACC>(N) + eps);
    if (mean_out != nullptr && threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    if (VEC > 1) {
        using vec_t = LNAlignedVec<T, VEC>;
        const int64_t nvec = N / VEC;
        const vec_t* xv = reinterpret_cast<const vec_t*>(x_row);
        const vec_t* gv = gamma ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
        const vec_t* bv = beta ? reinterpret_cast<const vec_t*>(beta) : nullptr;
        vec_t* yv = reinterpret_cast<vec_t*>(y_row);
        for (int64_t j = threadIdx.x; j < nvec; j += blockDim.x) {
            vec_t xp = xv[j];
            vec_t out;
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                ACC g = gv ? static_cast<ACC>(gv[j].val[k]) : ACC(1);
                ACC b = bv ? static_cast<ACC>(bv[j].val[k]) : ACC(0);
                out.val[k] = static_cast<T>(
                    (static_cast<ACC>(xp.val[k]) - mean) * rstd * g + b);
            }
            yv[j] = out;
        }
    } else {
        for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
            ACC g = gamma ? static_cast<ACC>(gamma[j]) : ACC(1);
            ACC b = beta ? static_cast<ACC>(beta[j]) : ACC(0);
            y_row[j] = static_cast<T>(
                (static_cast<ACC>(x_row[j]) - mean) * rstd * g + b);
        }
    }
}

// Row-wise moments into global buffers; shared by the backward pass.
template <typename T, typename ACC, int VEC>
__global__ void layer_norm_moments_kernel(
    int64_t N, ACC eps, const T* __restrict__ X,
    ACC* __restrict__ mean_out, ACC* __restrict__ rstd_out) {
    __shared__ LNWelford<ACC> smem[kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const T* x_row = X + row * N;

    LNWelford<ACC> wd{ACC(0), ACC(0), ACC(0)};
    if (VEC > 1) {
        using vec_t = LNAlignedVec<T, VEC>;
        const int64_t nvec = N / VEC;
        const vec_t* rowv = reinterpret_cast<const vec_t*>(x_row);
        for (int64_t j = threadIdx.x; j < nvec; j += blockDim.x) {
            vec_t pack = rowv[j];
#pragma unroll
            for (int k = 0; k < VEC; ++k)
                wd = ln_welford_online(static_cast<ACC>(pack.val[k]), wd);
        }
    } else {
        for (int64_t j = threadIdx.x; j < N; j += blockDim.x)
            wd = ln_welford_online(static_cast<ACC>(x_row[j]), wd);
    }
    wd = ln_block_reduce(wd, smem);
    if (threadIdx.x == 0) {
        mean_out[row] = wd.mean;
        rstd_out[row] = ln_rsqrt(wd.m2 / static_cast<ACC>(N) + eps);
    }
}

// grad_input: one block per row.  Matches the CPU backward formula
// dx = rstd/N * (N * dy * gamma - sum(dy * gamma) - x_hat * sum(dy * gamma * x_hat)).
template <typename T, typename ACC>
__global__ void layer_norm_grad_input_kernel(
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const ACC* __restrict__ mean,
    const ACC* __restrict__ rstd,
    const T* __restrict__ gamma,
    T* __restrict__ dX) {
    __shared__ ACC smem0[kLNThreads / 32];
    __shared__ ACC smem1[kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t off = row * N;
    const T* dy_row = dY + off;
    const T* x_row = X + off;
    T* dx_row = dX + off;
    const ACC mean_v = mean[row];
    const ACC rstd_v = rstd[row];

    ACC s_dy = 0, s_dy_xhat = 0;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const ACC g = gamma ? static_cast<ACC>(gamma[j]) : ACC(1);
        const ACC dyv = static_cast<ACC>(dy_row[j]);
        s_dy += dyv * g;
        s_dy_xhat += dyv * g *
            (static_cast<ACC>(x_row[j]) - mean_v) * rstd_v;
    }
    ln_block_reduce2(s_dy, s_dy_xhat, smem0, smem1);

    const ACC fH = static_cast<ACC>(N);
    const ACC term1 = rstd_v / fH;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const ACC g = gamma ? static_cast<ACC>(gamma[j]) : ACC(1);
        const ACC dyv = static_cast<ACC>(dy_row[j]);
        const ACC xh = (static_cast<ACC>(x_row[j]) - mean_v) * rstd_v;
        dx_row[j] = static_cast<T>(
            term1 * (fH * dyv * g - s_dy - xh * s_dy_xhat));
    }
}

// Single-value block reduction.  Same single-barrier scheme as the two-sum
// variant: warp partials land in `smem`, one barrier, then every warp folds
// them redundantly.  Rotate buffers across consecutive reductions.
template <typename ACC>
__device__ inline void ln_block_reduce1(ACC& v, ACC* smem) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    const int nw = static_cast<int>(blockDim.x >> 5);
    v = (lane < nw) ? smem[lane] : ACC(0);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
    v = __shfl_sync(0xffffffffffffffffull, v, 0);
}

// grad_input fast path: one block per row and the whole row lives in
// registers between the stats pass and the write pass, so dY and X are each
// read exactly once.  Moments are recomputed here (two accumulated passes
// over the registers keep the variance free of cancellation) and, when a
// column-gradient kernel follows, are also written out for it to reuse.
//   dx = rstd/N * (N * dy * gamma - sum(dy * gamma) - x_hat * sum(dy * gamma * x_hat))
template <typename T, typename ACC, int UNROLL>
__global__ void layer_norm_grad_input_reg_kernel(
    int64_t N, ACC eps,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    ACC* __restrict__ mean_out,
    ACC* __restrict__ rstd_out,
    T* __restrict__ dX) {
    constexpr int VEC = 4;
    constexpr int SLOTS = UNROLL * VEC;
    using vec_t = LNAlignedVec<T, VEC>;
    __shared__ ACC smem[kLNThreads / 32];
    __shared__ ACC smem0[kLNThreads / 32];
    __shared__ ACC smem1[kLNThreads / 32];
    __shared__ ACC smem2[kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t off = row * N;
    const int64_t nvec = N / VEC;
    const vec_t* dyv = reinterpret_cast<const vec_t*>(dY + off);
    const vec_t* xv = reinterpret_cast<const vec_t*>(X + off);
    const vec_t* gv = gamma ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    vec_t* dxv = reinterpret_cast<vec_t*>(dX + off);
    const ACC inv_N = ACC(1) / static_cast<ACC>(N);

    vec_t dy_r[UNROLL], x_r[UNROLL];
    bool has[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        has[k] = j < nvec;
        if (has[k]) {
            dy_r[k] = dyv[j];
            x_r[k] = xv[j];
        }
    }

    // Step 1: sum(x) and sum(dy*gamma); the latter needs no moments.
    // dy*gamma is stashed so the write step never re-loads gamma.
    ACC sx = ACC(0), s_dy = ACC(0);
    ACC wg_r[SLOTS];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        if (has[k]) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const ACC g = gv ? static_cast<ACC>(gv[j].val[e]) : ACC(1);
                const ACC wg = static_cast<ACC>(dy_r[k].val[e]) * g;
                sx += static_cast<ACC>(x_r[k].val[e]);
                wg_r[k * VEC + e] = wg;
                s_dy += wg;
            }
        }
    }
    ln_block_reduce2(sx, s_dy, smem, smem0);
    const ACC mean = sx * inv_N;

    // Step 2: sum((x-mean)^2) and the un-scaled sum(dy*gamma*(x-mean)).
    ACC sxx = ACC(0), sdyx = ACC(0);
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        if (has[k]) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const ACC d = static_cast<ACC>(x_r[k].val[e]) - mean;
                sxx += d * d;
                sdyx += wg_r[k * VEC + e] * d;
            }
        }
    }
    ln_block_reduce2(sxx, sdyx, smem1, smem2);
    const ACC rstd = ln_rsqrt(sxx * inv_N + eps);
    const ACC s_dy_xhat = sdyx * rstd;
    if (mean_out != nullptr && threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    const ACC fH = static_cast<ACC>(N);
    const ACC term1 = rstd * inv_N;
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        if (has[k]) {
            vec_t out;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const ACC dy_s = static_cast<ACC>(dy_r[k].val[e]);
                const ACC xh =
                    (static_cast<ACC>(x_r[k].val[e]) - mean) * rstd;
                out.val[e] = static_cast<T>(
                    term1 * (fH * wg_r[k * VEC + e] - s_dy - xh * s_dy_xhat));
            }
            ln_stream_store(&dxv[j], out);
        }
    }
}

// Fused backward for wide rows: each block owns a run of rows and keeps one
// row in registers at a time, so dY and X are read exactly once for all
// three gradients.  The per-row work folds into two barrier-separated steps:
// the first reduces sum(x) together with sum(dy*gamma) (the latter does not
// depend on the moments, and dy*gamma is stashed for the write step), the
// second reduces sum((x-mean)^2) together with the un-scaled
// sum(dy*gamma*(x-mean)), scaled by rstd once per row afterwards.  Buffer
// pairs alternate between the two reduces, so no spacing barrier is needed
// before the next row.  dX is written straight from the registers; the
// dgamma/dbeta contributions accumulate in each thread's registers (every
// thread owns fixed vec4 columns across the whole row run) and land in one
// partials row that a small follow-up kernel folds.  Traversal order is
// fixed and no atomics are used, so results are deterministic.
//   dgamma_c = sum_rows dy * x_hat     dbeta_c = sum_rows dy
//   dx = rstd/N * (N * dy * gamma - sum(dy * gamma) - x_hat * sum(dy * gamma * x_hat))
template <typename T, typename ACC, int UNROLL>
__global__ void layer_norm_bwd_fused_kernel(
    int64_t M, int64_t N, ACC eps, int64_t rows_per_block,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    T* __restrict__ dX,
    ACC* __restrict__ part_dg,
    ACC* __restrict__ part_db) {
    constexpr int VEC = 4;
    constexpr int SLOTS = UNROLL * VEC;
    using vec_t = LNAlignedVec<T, VEC>;
    using acc_vec_t = LNAlignedVec<ACC, VEC>;
    __shared__ ACC smem[kLNThreads / 32];
    __shared__ ACC smem0[kLNThreads / 32];
    __shared__ ACC smem1[kLNThreads / 32];
    __shared__ ACC smem2[kLNThreads / 32];

    const int64_t r0 = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    int64_t r1 = r0 + rows_per_block;
    if (r1 > M) r1 = M;
    const int64_t nvec = N / VEC;
    const vec_t* dyv = reinterpret_cast<const vec_t*>(dY);
    const vec_t* xv = reinterpret_cast<const vec_t*>(X);
    const vec_t* gv = gamma ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    vec_t* dxv = reinterpret_cast<vec_t*>(dX);
    const ACC inv_N = ACC(1) / static_cast<ACC>(N);

    ACC dg_acc[SLOTS], db_acc[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        dg_acc[s] = ACC(0);
        db_acc[s] = ACC(0);
    }

    for (int64_t r = r0; r < r1; ++r) {
        const int64_t off = r * nvec;
        vec_t dy_r[UNROLL], x_r[UNROLL];
        bool has[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            has[k] = j < nvec;
            if (has[k]) {
                dy_r[k] = dyv[off + j];
                x_r[k] = xv[off + j];
            }
        }

        // Step 1: sum(x) and sum(dy*gamma); the latter needs no moments.
        // dy*gamma is stashed so the write step never re-loads gamma.
        ACC sx = ACC(0), s_dy = ACC(0);
        ACC wg_r[SLOTS];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (has[k]) {
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const ACC g =
                        gv ? static_cast<ACC>(gv[j].val[e]) : ACC(1);
                    const ACC wg =
                        static_cast<ACC>(dy_r[k].val[e]) * g;
                    sx += static_cast<ACC>(x_r[k].val[e]);
                    wg_r[k * VEC + e] = wg;
                    s_dy += wg;
                }
            }
        }
        ln_block_reduce2(sx, s_dy, smem, smem0);
        const ACC mean = sx * inv_N;

        // Step 2: sum((x-mean)^2) and the un-scaled sum(dy*gamma*(x-mean)).
        ACC sxx = ACC(0), sdyx = ACC(0);
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            if (has[k]) {
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const ACC d =
                        static_cast<ACC>(x_r[k].val[e]) - mean;
                    sxx += d * d;
                    sdyx += wg_r[k * VEC + e] * d;
                }
            }
        }
        ln_block_reduce2(sxx, sdyx, smem1, smem2);
        const ACC rstd = ln_rsqrt(sxx * inv_N + eps);
        const ACC s_dy_xhat = sdyx * rstd;

        const ACC term1 = rstd * inv_N;
        const ACC fH = static_cast<ACC>(N);
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (has[k]) {
                vec_t out;
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const ACC dy_s = static_cast<ACC>(dy_r[k].val[e]);
                    const ACC xh =
                        (static_cast<ACC>(x_r[k].val[e]) - mean) * rstd;
                    dg_acc[k * VEC + e] += dy_s * xh;
                    db_acc[k * VEC + e] += dy_s;
                    out.val[e] = static_cast<T>(
                        term1 * (fH * wg_r[k * VEC + e] - s_dy - xh * s_dy_xhat));
                }
                ln_stream_store(&dxv[off + j], out);
            }
        }
    }

    if (part_dg != nullptr) {
        acc_vec_t* dgv = reinterpret_cast<acc_vec_t*>(part_dg) +
                         static_cast<int64_t>(blockIdx.x) * nvec;
        acc_vec_t* dbv = reinterpret_cast<acc_vec_t*>(part_db) +
                         static_cast<int64_t>(blockIdx.x) * nvec;
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (j < nvec) {
                acc_vec_t pd, pb;
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    pd.val[e] = dg_acc[k * VEC + e];
                    pb.val[e] = db_acc[k * VEC + e];
                }
                dgv[j] = pd;
                dbv[j] = pb;
            }
        }
    }
}

// Column gradients (dgamma = sum_rows dy * x_hat, dbeta = sum_rows dy).
// A 2D tile strides the rows: each thread streams a fixed column and holds
// no more than one row's worth of live values, warp lanes share the per-row
// moments through shuffles, and one shared-memory transpose reduces the tile
// once at the end.  Fixed traversal order and no atomics, so results are
// deterministic.  When gridDim.y > 1 (very tall, narrow inputs) each y slice
// lands in a partials row and a follow-up kernel folds them.
template <typename T, typename ACC, int BDY>
__global__ void layer_norm_grad_cols_kernel(
    int64_t M, int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const ACC* __restrict__ mean,
    const ACC* __restrict__ rstd,
    T* __restrict__ dGamma,
    T* __restrict__ dBeta,
    ACC* __restrict__ part_dg,
    ACC* __restrict__ part_db) {
    constexpr int ROWS_PER_THREAD = 8;
    constexpr int RPB = BDY * ROWS_PER_THREAD;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int64_t col = static_cast<int64_t>(blockIdx.x) * 32 + tx;
    const bool col_ok = col < N;

    ACC dg_sum = ACC(0), db_sum = ACC(0);
    for (int64_t m0 = static_cast<int64_t>(blockIdx.y) * RPB; m0 < M;
         m0 += static_cast<int64_t>(RPB) * gridDim.y) {
        // Lanes 0..7 of the warp fetch this row group's moments; the shuffles
        // below hand each row's values to every lane.  A null mean pointer
        // means no gamma gradient was requested, so the moments stay zero.
        ACC warp_mean = ACC(0), warp_rstd = ACC(0);
        const int64_t stat_row = m0 + ty * ROWS_PER_THREAD + tx;
        if (tx < ROWS_PER_THREAD && mean != nullptr && stat_row < M) {
            warp_mean = mean[stat_row];
            warp_rstd = rstd[stat_row];
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            const ACC mn = __shfl_sync(0xffffffffffffffffull, warp_mean, i);
            const ACC rd = __shfl_sync(0xffffffffffffffffull, warp_rstd, i);
            const int64_t r = m0 + ty * ROWS_PER_THREAD + i;
            if (r < M && col_ok) {
                const ACC dy = static_cast<ACC>(dY[r * N + col]);
                const ACC xv = static_cast<ACC>(X[r * N + col]);
                db_sum += dy;
                dg_sum += dy * (xv - mn) * rd;
            }
        }
    }

    if (BDY > 1) {
        __shared__ ACC s_dg[BDY][33];
        __shared__ ACC s_db[BDY][33];
        s_dg[ty][tx] = dg_sum;
        s_db[ty][tx] = db_sum;
        __syncthreads();
        // Transposed read: warp ty folds column ty (and its stride) over the
        // y dimension held in the lanes.
        for (int i = ty; i < 32; i += BDY) {
            ACC rdg = ACC(0), rdb = ACC(0);
            if (tx < BDY) {
                rdg = s_dg[tx][i];
                rdb = s_db[tx][i];
            }
#pragma unroll
            for (int delta = BDY / 2; delta > 0; delta >>= 1) {
                rdg += __shfl_xor_sync(0xffffffffffffffffull, rdg, delta);
                rdb += __shfl_xor_sync(0xffffffffffffffffull, rdb, delta);
            }
            const int64_t out_col =
                static_cast<int64_t>(blockIdx.x) * 32 + i;
            if (tx == 0 && out_col < N) {
                if (part_dg != nullptr)
                    part_dg[static_cast<int64_t>(blockIdx.y) * N + out_col] = rdg;
                else if (dGamma != nullptr)
                    dGamma[out_col] = static_cast<T>(rdg);
                if (part_db != nullptr)
                    part_db[static_cast<int64_t>(blockIdx.y) * N + out_col] = rdb;
                else if (dBeta != nullptr)
                    dBeta[out_col] = static_cast<T>(rdb);
            }
        }
    } else if (col_ok) {
        if (part_dg != nullptr)
            part_dg[static_cast<int64_t>(blockIdx.y) * N + col] = dg_sum;
        else if (dGamma != nullptr)
            dGamma[col] = static_cast<T>(dg_sum);
        if (part_db != nullptr)
            part_db[static_cast<int64_t>(blockIdx.y) * N + col] = db_sum;
        else if (dBeta != nullptr)
            dBeta[col] = static_cast<T>(db_sum);
    }
}

// Fold the per-block partial rows produced by the fused backward kernel and
// the tall-skinny launch of the column kernel.  A block tiles 32 columns by
// 8 row groups: loads stay warp-coalesced along the columns while the eight
// group accumulators cover a partial stack eight rows deep, then shared
// memory folds the groups with a fixed tree.  Deterministic.
template <typename T, typename ACC>
__global__ void layer_norm_grad_cols_finalize_kernel(
    int64_t N, int64_t gy,
    const ACC* __restrict__ part,
    T* __restrict__ dGamma, T* __restrict__ dBeta) {
    __shared__ ACC s_dg[8][33];
    __shared__ ACC s_db[8][33];
    const int cx = static_cast<int>(threadIdx.x) & 31;
    const int cy = static_cast<int>(threadIdx.x) >> 5;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * 32 + cx;
    const bool col_ok = col < N;
    ACC sd = ACC(0), sb = ACC(0);
    if (col_ok) {
        if (dGamma != nullptr) {
            for (int64_t g = cy; g < gy; g += 8) sd += part[g * N + col];
        }
        if (dBeta != nullptr) {
            for (int64_t g = cy; g < gy; g += 8)
                sb += part[(gy + g) * N + col];
        }
    }
    s_dg[cy][cx] = sd;
    s_db[cy][cx] = sb;
    __syncthreads();
    if (cy == 0 && col_ok) {
        if (dGamma != nullptr) {
            const ACC a0 = s_dg[0][cx], a1 = s_dg[1][cx];
            const ACC a2 = s_dg[2][cx], a3 = s_dg[3][cx];
            const ACC a4 = s_dg[4][cx], a5 = s_dg[5][cx];
            const ACC a6 = s_dg[6][cx], a7 = s_dg[7][cx];
            dGamma[col] =
                static_cast<T>(((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)));
        }
        if (dBeta != nullptr) {
            const ACC b0 = s_db[0][cx], b1 = s_db[1][cx];
            const ACC b2 = s_db[2][cx], b3 = s_db[3][cx];
            const ACC b4 = s_db[4][cx], b5 = s_db[5][cx];
            const ACC b6 = s_db[6][cx], b7 = s_db[7][cx];
            dBeta[col] =
                static_cast<T>(((b0 + b1) + (b2 + b3)) + ((b4 + b5) + (b6 + b7)));
        }
    }
}

inline bool ln_ptr_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <typename T, typename ACC>
void launch_layer_norm_forward(
    int64_t M, int64_t N, double eps,
    const T* X, const T* gamma, const T* beta, T* Y,
    ACC* mean_out, ACC* rstd_out) {
    const bool vec_ok = (N % 4 == 0) && ln_ptr_aligned(X) && ln_ptr_aligned(Y) &&
        (!gamma || ln_ptr_aligned(gamma)) && (!beta || ln_ptr_aligned(beta));
    const auto stream = getCurrentCUDAStream().stream();
    if (vec_ok) {
        layer_norm_forward_kernel<T, ACC, 4><<<static_cast<unsigned>(M), ln_threads_for(N), 0, stream>>>(
            N, static_cast<ACC>(eps), X, gamma, beta, Y, mean_out, rstd_out);
    } else {
        layer_norm_forward_kernel<T, ACC, 1><<<static_cast<unsigned>(M), ln_threads_for(N), 0, stream>>>(
            N, static_cast<ACC>(eps), X, gamma, beta, Y, mean_out, rstd_out);
    }
}

template <typename T, typename ACC>
void launch_layer_norm_backward(
    int64_t M, int64_t N, double eps,
    const T* dY, const T* X, const T* gamma, T* dX,
    T* dGamma, T* dBeta) {
    const auto stream = getCurrentCUDAStream().stream();
    const ACC eps_acc = static_cast<ACC>(eps);

    Tensor stats;
    ACC* mean_p = nullptr;
    ACC* rstd_p = nullptr;
    auto ensure_stats = [&]() {
        if (!stats.defined()) {
            // One contiguous [2, M] buffer: mean at offset 0, rstd at offset M.
            stats = Tensor::empty(
                std::vector<int64_t>{2 * M},
                (std::is_same<ACC, double>::value ? DType::Float64
                                                  : DType::Float32),
                Device(DeviceType::CUDA));
            mean_p = stats.data_ptr<ACC>();
            rstd_p = mean_p + M;
        }
    };

    // Fast paths (16-byte aligned rows, width a multiple of 4, up to
    // threads * 4 packets * 4 registers deep).  Wide rows with enough batch
    // take the fused kernel, which reads dY and X once for all three
    // gradients; anything else splits into the row-register grad-input
    // kernel plus the column kernel.
    bool fast = false;
    bool fused = false;
    const bool vec_ok = (N % 4 == 0) && ln_ptr_aligned(dY) && ln_ptr_aligned(X) &&
        ln_ptr_aligned(dX) && (!gamma || ln_ptr_aligned(gamma));
    if (vec_ok) {
        const unsigned threads = ln_bwd_threads_for(N);
        const int64_t slots =
            (N / 4 + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
        if (slots <= 4) {
            fast = true;
            if (N >= 1024 && M >= 512) {
                fused = true;
                // Bound the partials volume (~8 MB per direction) while
                // keeping at least four rows per block for load depth.
                int64_t rpb = (M + 1023) / 1024;
                const int64_t by_volume =
                    (M * N + ((1 << 21) - 1)) / (1 << 21);
                if (by_volume > rpb) rpb = by_volume;
                if (rpb < 4) rpb = 4;
                const int64_t grid = (M + rpb - 1) / rpb;
                ACC* part = nullptr;
                Tensor partials;
                if (dGamma != nullptr || dBeta != nullptr) {
                    partials = Tensor::empty(
                        std::vector<int64_t>{2 * grid * N},
                        (std::is_same<ACC, double>::value ? DType::Float64
                                                          : DType::Float32),
                        Device(DeviceType::CUDA));
                    part = partials.data_ptr<ACC>();
                }
#define LN_BWD_FUSED_LAUNCH(U)                                                 \
                layer_norm_bwd_fused_kernel<T, ACC, U>                         \
                    <<<static_cast<unsigned>(grid), threads, 0, stream>>>(     \
                        M, N, eps_acc, rpb, dY, X, gamma, dX, part,            \
                        part != nullptr ? part + grid * N : nullptr)
                if (slots <= 1) {
                    LN_BWD_FUSED_LAUNCH(1);
                } else if (slots <= 2) {
                    LN_BWD_FUSED_LAUNCH(2);
                } else {
                    LN_BWD_FUSED_LAUNCH(4);
                }
#undef LN_BWD_FUSED_LAUNCH
                if (part != nullptr) {
                    layer_norm_grad_cols_finalize_kernel<T, ACC>
                        <<<static_cast<unsigned>((N + 31) / 32), 256, 0,
                           stream>>>(N, grid, part, dGamma, dBeta);
                }
            } else {
                if (dGamma != nullptr) ensure_stats();
#define LN_BWD_REG_LAUNCH(U)                                                   \
            layer_norm_grad_input_reg_kernel<T, ACC, U>                        \
                <<<static_cast<unsigned>(M), threads, 0, stream>>>(            \
                    N, eps_acc, dY, X, gamma, mean_p, rstd_p, dX)
                if (slots <= 1) {
                    LN_BWD_REG_LAUNCH(1);
                } else if (slots <= 2) {
                    LN_BWD_REG_LAUNCH(2);
                } else {
                    LN_BWD_REG_LAUNCH(4);
                }
#undef LN_BWD_REG_LAUNCH
            }
        }
    }
    if (!fast) {
        // Generic path: a separate moments pass feeds the streaming
        // grad-input kernel (which re-reads dY and X for the write pass).
        ensure_stats();
        const bool mom_vec = (N % 4 == 0) && ln_ptr_aligned(X);
        if (mom_vec) {
            layer_norm_moments_kernel<T, ACC, 4>
                <<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
                    N, eps_acc, X, mean_p, rstd_p);
        } else {
            layer_norm_moments_kernel<T, ACC, 1>
                <<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
                    N, eps_acc, X, mean_p, rstd_p);
        }
        layer_norm_grad_input_kernel<T, ACC>
            <<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
                N, dY, X, mean_p, rstd_p, gamma, dX);
    }

    if ((dGamma != nullptr || dBeta != nullptr) && !fused) {
        const unsigned bx = static_cast<unsigned>((N + 31) / 32);
        int dev = 0;
        cudaGetDevice(&dev);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        // Spread rows over enough y slices to fill the device when the
        // column grid alone cannot (narrow N), keeping every slice >= one
        // 256-row tile so the tall-slice kernel stays fully tiled.
        int64_t gy = 1;
        if (M >= 256) {
            const int64_t want =
                (static_cast<int64_t>(sm_count) * 2 + bx - 1) / bx;
            gy = std::min<int64_t>(
                std::min<int64_t>((M + 255) / 256, 2048),
                std::max<int64_t>(1, want));
        }
        if (dGamma != nullptr) ensure_stats();
        if (gy > 1) {
            Tensor partials = Tensor::empty(
                std::vector<int64_t>{2 * gy * N},
                (std::is_same<ACC, double>::value ? DType::Float64
                                                  : DType::Float32),
                Device(DeviceType::CUDA));
            ACC* part = partials.data_ptr<ACC>();
            layer_norm_grad_cols_kernel<T, ACC, 32>
                <<<dim3(bx, static_cast<unsigned>(gy)), dim3(32, 32), 0, stream>>>(
                    M, N, dY, X, mean_p, rstd_p, nullptr, nullptr,
                    part, part + gy * N);
            layer_norm_grad_cols_finalize_kernel<T, ACC>
                <<<static_cast<unsigned>((N + 31) / 32), 256, 0, stream>>>(
                    N, gy, part, dGamma, dBeta);
        } else {
            const int bdy = M < 64 ? 1 : (M < 128 ? 8 : 32);
            dim3 threads(32, static_cast<unsigned>(bdy));
#define LN_BWD_COLS_LAUNCH(B)                                                  \
            layer_norm_grad_cols_kernel<T, ACC, B>                             \
                <<<bx, threads, 0, stream>>>(                                  \
                    M, N, dY, X, mean_p, rstd_p, dGamma, dBeta, nullptr, nullptr)
            if (bdy == 1) {
                LN_BWD_COLS_LAUNCH(1);
            } else if (bdy == 8) {
                LN_BWD_COLS_LAUNCH(8);
            } else {
                LN_BWD_COLS_LAUNCH(32);
            }
#undef LN_BWD_COLS_LAUNCH
        }
    }
}


// Stats-dtype helper shared by the native entry points.
inline DType ln_stats_dtype(DType t) {
    return t == DType::Float64 ? DType::Float64 : DType::Float32;
}

// Fused backward over saved moments: the forward's mean/rstd remove both
// statistics passes, so each row needs a single dual reduction of
// sum(dy*gamma) and the un-scaled sum(dy*gamma*(x-mean)), scaled by rstd
// once afterwards.  The row stays in registers and dX is written straight
// from them; column gradients accumulate in per-thread registers and land
// in one partials row for the follow-up fold.  Buffer pairs alternate by
// alternating rows, which also spaces consecutive uses of the same shared memory.
// Traversal order is fixed and no atomics are used, so results are
// deterministic.
template <typename T, typename ACC, int UNROLL>
__global__ void layer_norm_bwd_fused_stats_kernel(
    int64_t M, int64_t N, int64_t rows_per_block,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    const ACC* __restrict__ mean,
    const ACC* __restrict__ rstd,
    T* __restrict__ dX,
    ACC* __restrict__ part_dg,
    ACC* __restrict__ part_db) {
    constexpr int VEC = 4;
    constexpr int SLOTS = UNROLL * VEC;
    using vec_t = LNAlignedVec<T, VEC>;
    using acc_vec_t = LNAlignedVec<ACC, VEC>;
    __shared__ ACC smem[kLNThreads / 32];
    __shared__ ACC smem0[kLNThreads / 32];
    __shared__ ACC smem1[kLNThreads / 32];
    __shared__ ACC smem2[kLNThreads / 32];

    const int64_t r0 = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    int64_t r1 = r0 + rows_per_block;
    if (r1 > M) r1 = M;
    const int64_t nvec = N / VEC;
    const vec_t* dyv = reinterpret_cast<const vec_t*>(dY);
    const vec_t* xv = reinterpret_cast<const vec_t*>(X);
    const vec_t* gv = gamma ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    vec_t* dxv = reinterpret_cast<vec_t*>(dX);
    const ACC inv_N = ACC(1) / static_cast<ACC>(N);

    ACC dg_acc[SLOTS], db_acc[SLOTS];
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        dg_acc[s] = ACC(0);
        db_acc[s] = ACC(0);
    }

    for (int64_t r = r0; r < r1; ++r) {
        const int64_t off = r * nvec;
        const ACC m = mean[r];
        const ACC rd = rstd[r];
        vec_t dy_r[UNROLL], x_r[UNROLL];
        bool has[UNROLL];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            has[k] = j < nvec;
            if (has[k]) {
                dy_r[k] = dyv[off + j];
                x_r[k] = xv[off + j];
            }
        }

        ACC s_dy = ACC(0), sdyx = ACC(0);
        ACC wg_r[SLOTS];
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (has[k]) {
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const ACC g =
                        gv ? static_cast<ACC>(gv[j].val[e]) : ACC(1);
                    const ACC wg =
                        static_cast<ACC>(dy_r[k].val[e]) * g;
                    wg_r[k * VEC + e] = wg;
                    s_dy += wg;
                    sdyx += wg * (static_cast<ACC>(x_r[k].val[e]) - m);
                }
            }
        }
        ACC* buf_a = (r & 1) ? smem1 : smem;
        ACC* buf_b = (r & 1) ? smem2 : smem0;
        ln_block_reduce2(s_dy, sdyx, buf_a, buf_b);
        const ACC s_dy_xhat = sdyx * rd;

        const ACC term1 = rd * inv_N;
        const ACC fH = static_cast<ACC>(N);
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (has[k]) {
                vec_t out;
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    const ACC dy_s = static_cast<ACC>(dy_r[k].val[e]);
                    const ACC xh =
                        (static_cast<ACC>(x_r[k].val[e]) - m) * rd;
                    dg_acc[k * VEC + e] += dy_s * xh;
                    db_acc[k * VEC + e] += dy_s;
                    out.val[e] = static_cast<T>(
                        term1 * (fH * wg_r[k * VEC + e] - s_dy - xh * s_dy_xhat));
                }
                ln_stream_store(&dxv[off + j], out);
            }
        }
    }

    if (part_dg != nullptr) {
        acc_vec_t* dgv = reinterpret_cast<acc_vec_t*>(part_dg) +
                         static_cast<int64_t>(blockIdx.x) * nvec;
        acc_vec_t* dbv = reinterpret_cast<acc_vec_t*>(part_db) +
                         static_cast<int64_t>(blockIdx.x) * nvec;
#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
            if (j < nvec) {
                acc_vec_t pd, pb;
#pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    pd.val[e] = dg_acc[k * VEC + e];
                    pb.val[e] = db_acc[k * VEC + e];
                }
                dgv[j] = pd;
                dbv[j] = pb;
            }
        }
    }
}

// grad_input over saved moments: one block per row, row in registers, one
// dual reduction for sum(dy*gamma) and sum(dy*gamma*(x-mean)).  The same
// mean/rstd feed the column kernel when it follows.
template <typename T, typename ACC, int UNROLL>
__global__ void layer_norm_grad_input_reg_stats_kernel(
    int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    const ACC* __restrict__ mean,
    const ACC* __restrict__ rstd,
    T* __restrict__ dX) {
    constexpr int VEC = 4;
    constexpr int SLOTS = UNROLL * VEC;
    using vec_t = LNAlignedVec<T, VEC>;
    __shared__ ACC smem[kLNThreads / 32];
    __shared__ ACC smem0[kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t off = row * N;
    const int64_t nvec = N / VEC;
    const vec_t* dyv = reinterpret_cast<const vec_t*>(dY + off);
    const vec_t* xv = reinterpret_cast<const vec_t*>(X + off);
    const vec_t* gv = gamma ? reinterpret_cast<const vec_t*>(gamma) : nullptr;
    vec_t* dxv = reinterpret_cast<vec_t*>(dX + off);
    const ACC m = mean[row];
    const ACC rd = rstd[row];

    vec_t dy_r[UNROLL], x_r[UNROLL];
    bool has[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        has[k] = j < nvec;
        if (has[k]) {
            dy_r[k] = dyv[j];
            x_r[k] = xv[j];
        }
    }

    ACC s_dy = ACC(0), sdyx = ACC(0);
    ACC wg_r[SLOTS];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        if (has[k]) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const ACC g = gv ? static_cast<ACC>(gv[j].val[e]) : ACC(1);
                const ACC wg = static_cast<ACC>(dy_r[k].val[e]) * g;
                wg_r[k * VEC + e] = wg;
                s_dy += wg;
                sdyx += wg * (static_cast<ACC>(x_r[k].val[e]) - m);
            }
        }
    }
    ln_block_reduce2(s_dy, sdyx, smem, smem0);
    const ACC s_dy_xhat = sdyx * rd;

    const ACC inv_N = ACC(1) / static_cast<ACC>(N);
    const ACC term1 = rd * inv_N;
    const ACC fH = static_cast<ACC>(N);
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        const int64_t j = threadIdx.x + static_cast<int64_t>(k) * blockDim.x;
        if (has[k]) {
            vec_t out;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const ACC dy_s = static_cast<ACC>(dy_r[k].val[e]);
                const ACC xh =
                    (static_cast<ACC>(x_r[k].val[e]) - m) * rd;
                out.val[e] = static_cast<T>(
                    term1 * (fH * wg_r[k * VEC + e] - s_dy - xh * s_dy_xhat));
            }
            ln_stream_store(&dxv[j], out);
        }
    }
}

// Backward driven by the forward's saved moments.  dX runs through the same
// fused / row-register / generic ladder as the recomputing launcher minus
// the statistics passes; column gradients reuse the saved mean/rstd so the
// column kernel never recomputes them either.
template <typename T, typename ACC>
void launch_layer_norm_backward_stats(
    int64_t M, int64_t N,
    const T* dY, const T* X, const T* gamma,
    const ACC* mean_p, const ACC* rstd_p,
    T* dX, T* dGamma, T* dBeta) {
    const auto stream = getCurrentCUDAStream().stream();

    bool fused = false;
    bool dx_done = false;
    const bool vec_ok = (N % 4 == 0) && ln_ptr_aligned(dY) && ln_ptr_aligned(X) &&
        (!dX || ln_ptr_aligned(dX)) && (!gamma || ln_ptr_aligned(gamma));
    if (dX != nullptr && vec_ok) {
        const unsigned threads = ln_bwd_threads_for(N);
        const int64_t slots =
            (N / 4 + static_cast<int64_t>(threads) - 1) / static_cast<int64_t>(threads);
        if (slots <= 4) {
            dx_done = true;
            if (N >= 1024 && M >= 512) {
                fused = true;
                int64_t rpb = (M + 1023) / 1024;
                const int64_t by_volume =
                    (M * N + ((1 << 21) - 1)) / (1 << 21);
                if (by_volume > rpb) rpb = by_volume;
                if (rpb < 4) rpb = 4;
                const int64_t grid = (M + rpb - 1) / rpb;
                ACC* part = nullptr;
                Tensor partials;
                if (dGamma != nullptr || dBeta != nullptr) {
                    partials = Tensor::empty(
                        std::vector<int64_t>{2 * grid * N},
                        (std::is_same<ACC, double>::value ? DType::Float64
                                                          : DType::Float32),
                        Device(DeviceType::CUDA));
                    part = partials.data_ptr<ACC>();
                }
#define LN_BWD_SFUSED_LAUNCH(U)                                                \
                layer_norm_bwd_fused_stats_kernel<T, ACC, U>                   \
                    <<<static_cast<unsigned>(grid), threads, 0, stream>>>(     \
                        M, N, rpb, dY, X, gamma, mean_p, rstd_p, dX, part,     \
                        part != nullptr ? part + grid * N : nullptr)
                if (slots <= 1) {
                    LN_BWD_SFUSED_LAUNCH(1);
                } else if (slots <= 2) {
                    LN_BWD_SFUSED_LAUNCH(2);
                } else {
                    LN_BWD_SFUSED_LAUNCH(4);
                }
#undef LN_BWD_SFUSED_LAUNCH
                if (part != nullptr) {
                    layer_norm_grad_cols_finalize_kernel<T, ACC>
                        <<<static_cast<unsigned>((N + 31) / 32), 256, 0,
                           stream>>>(N, grid, part, dGamma, dBeta);
                }
            } else {
#define LN_BWD_SREG_LAUNCH(U)                                                  \
            layer_norm_grad_input_reg_stats_kernel<T, ACC, U>                  \
                <<<static_cast<unsigned>(M), threads, 0, stream>>>(            \
                    N, dY, X, gamma, mean_p, rstd_p, dX)
                if (slots <= 1) {
                    LN_BWD_SREG_LAUNCH(1);
                } else if (slots <= 2) {
                    LN_BWD_SREG_LAUNCH(2);
                } else {
                    LN_BWD_SREG_LAUNCH(4);
                }
#undef LN_BWD_SREG_LAUNCH
            }
        }
    }
    if (dX != nullptr && !dx_done) {
        // Generic path: the streaming grad-input kernel over saved moments.
        layer_norm_grad_input_kernel<T, ACC>
            <<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
                N, dY, X, mean_p, rstd_p, gamma, dX);
    }

    if ((dGamma != nullptr || dBeta != nullptr) && !fused) {
        const unsigned bx = static_cast<unsigned>((N + 31) / 32);
        int dev = 0;
        cudaGetDevice(&dev);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        int64_t gy = 1;
        if (M >= 256) {
            const int64_t want =
                (static_cast<int64_t>(sm_count) * 2 + bx - 1) / bx;
            gy = std::min<int64_t>(
                std::min<int64_t>((M + 255) / 256, 2048),
                std::max<int64_t>(1, want));
        }
        if (gy > 1) {
            Tensor partials = Tensor::empty(
                std::vector<int64_t>{2 * gy * N},
                (std::is_same<ACC, double>::value ? DType::Float64
                                                  : DType::Float32),
                Device(DeviceType::CUDA));
            ACC* part = partials.data_ptr<ACC>();
            layer_norm_grad_cols_kernel<T, ACC, 32>
                <<<dim3(bx, static_cast<unsigned>(gy)), dim3(32, 32), 0, stream>>>(
                    M, N, dY, X, mean_p, rstd_p, nullptr, nullptr,
                    part, part + gy * N);
            layer_norm_grad_cols_finalize_kernel<T, ACC>
                <<<static_cast<unsigned>((N + 31) / 32), 256, 0, stream>>>(
                    N, gy, part, dGamma, dBeta);
        } else {
            const int bdy = M < 64 ? 1 : (M < 128 ? 8 : 32);
            dim3 threads(32, static_cast<unsigned>(bdy));
#define LN_BWD_SCOLS_LAUNCH(B)                                                 \
            layer_norm_grad_cols_kernel<T, ACC, B>                             \
                <<<bx, threads, 0, stream>>>(                                  \
                    M, N, dY, X, mean_p, rstd_p, dGamma, dBeta, nullptr, nullptr)
            if (bdy == 1) {
                LN_BWD_SCOLS_LAUNCH(1);
            } else if (bdy == 8) {
                LN_BWD_SCOLS_LAUNCH(8);
            } else {
                LN_BWD_SCOLS_LAUNCH(32);
            }
#undef LN_BWD_SCOLS_LAUNCH
        }
    }
}

} // namespace layer_norm

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_cuda(
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt,
        double eps) {
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim)
        TP_THROW(RuntimeError, "layer_norm: normalized_shape dim larger than input dim");

    const int64_t outer_dims = input_ndim - norm_ndim;
    int64_t N = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[i])
            TP_THROW(RuntimeError, "layer_norm: Input shape mismatch with normalized_shape");
        N *= normalized_shape[i];
    }
    const int64_t M = input.numel() / (N == 0 ? 1 : N);

    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    // The affine terms are read one per normalized element: a shape of any
    // other extent would be read past its end.
    if (has_weight && static_cast<std::vector<int64_t>>(weight_opt->shape()) != normalized_shape)
        TP_THROW(RuntimeError, "layer_norm: weight shape mismatch with normalized_shape");
    if (has_bias && static_cast<std::vector<int64_t>>(bias_opt->shape()) != normalized_shape)
        TP_THROW(RuntimeError, "layer_norm: bias shape mismatch with normalized_shape");

    Tensor in_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor bias = has_bias ? bias_opt->contiguous() : Tensor();

    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(in_contig.shape()),
                               in_contig.dtype(), in_contig.device());
    const DType stats_dt = layer_norm::ln_stats_dtype(in_contig.dtype());
    Tensor mean = Tensor::empty(std::vector<int64_t>{M}, stats_dt, in_contig.device());
    Tensor rstd = Tensor::empty(std::vector<int64_t>{M}, stats_dt, in_contig.device());
    if (in_contig.numel() == 0 || M == 0 || N == 0) {
        return std::make_tuple(out, mean, rstd);
    }

    switch (in_contig.dtype()) {
#define LN_FORWARD_CASE(ctype, name, acc_t)                                   \
        case DType::name:                                                     \
            layer_norm::launch_layer_norm_forward<ctype, acc_t>(              \
                M, N, eps,                                                    \
                in_contig.data_ptr<ctype>(),                                  \
                has_weight ? weight.data_ptr<ctype>() : nullptr,              \
                has_bias ? bias.data_ptr<ctype>() : nullptr,                  \
                out.data_ptr<ctype>(),                                        \
                mean.data_ptr<acc_t>(), rstd.data_ptr<acc_t>());              \
            break;
        LN_FORWARD_CASE(float, Float32, float)
        LN_FORWARD_CASE(double, Float64, double)
        LN_FORWARD_CASE(Half, Float16, float)
        LN_FORWARD_CASE(BFloat16, BFloat16, float)
#undef LN_FORWARD_CASE
        default:
            TP_THROW(NotImplementedError,
                     "layer_norm CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    {
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError, std::string("native_layer_norm: ") + cudaGetErrorString(error));
        }
    }
    return std::make_tuple(out, mean, rstd);
}

// The public spelling is a thin composition over the stats-producing native
// op so the autograd node saves the per-row moments for the backward pass.
Tensor layer_norm_cuda(const Tensor& input,
                       const std::vector<int64_t>& normalized_shape,
                       const std::optional<Tensor>& weight_opt,
                       const std::optional<Tensor>& bias_opt,
                       double eps) {
    return std::get<0>(tensorplay::tpx::ops::native_layer_norm(
        input, normalized_shape, weight_opt, bias_opt, eps));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim)
        TP_THROW(RuntimeError, "layer_norm_backward: normalized_shape dim larger than input dim");
    if (grad_output.dtype() != input.dtype())
        TP_THROW(RuntimeError, "layer_norm_backward: grad_output dtype must match input dtype");

    const int64_t outer_dims = input_ndim - norm_ndim;
    int64_t N = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[i])
            TP_THROW(RuntimeError, "layer_norm_backward: Input shape mismatch with normalized_shape");
        N *= normalized_shape[i];
    }
    const int64_t M = input.numel() / (N == 0 ? 1 : N);

    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();

    Tensor grad_out_contig = grad_output.contiguous();
    Tensor in_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();

    Tensor grad_input = Tensor::empty(static_cast<std::vector<int64_t>>(in_contig.shape()),
                                      in_contig.dtype(), in_contig.device());
    Tensor grad_weight = Tensor();
    Tensor grad_bias = Tensor();
    if (has_weight) {
        grad_weight = Tensor::empty(static_cast<std::vector<int64_t>>(weight.shape()),
                                    weight.dtype(), weight.device());
    }
    if (has_bias) {
        const Tensor& like = has_weight ? weight : *bias_opt;
        grad_bias = Tensor::empty(static_cast<std::vector<int64_t>>(like.shape()),
                                  like.dtype(), like.device());
    }
    if (in_contig.numel() == 0 || M == 0 || N == 0)
        return std::make_tuple(grad_input, grad_weight, grad_bias);

    switch (in_contig.dtype()) {
#define LN_BACKWARD_CASE(ctype, name, acc_t)                                  \
        case DType::name:                                                     \
            layer_norm::launch_layer_norm_backward<ctype, acc_t>(                         \
                M, N, eps,                                                    \
                grad_out_contig.data_ptr<ctype>(),                            \
                in_contig.data_ptr<ctype>(),                                  \
                has_weight ? weight.data_ptr<ctype>() : nullptr,              \
                grad_input.data_ptr<ctype>(),                                 \
                has_weight ? grad_weight.data_ptr<ctype>() : nullptr,         \
                has_bias ? grad_bias.data_ptr<ctype>() : nullptr);            \
            break;
        LN_BACKWARD_CASE(float, Float32, float)
        LN_BACKWARD_CASE(double, Float64, double)
        LN_BACKWARD_CASE(Half, Float16, float)
        LN_BACKWARD_CASE(BFloat16, BFloat16, float)
#undef LN_BACKWARD_CASE
        default:
            TP_THROW(NotImplementedError,
                     "layer_norm_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    {
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError, std::string("layer_norm_backward_cuda: ") + cudaGetErrorString(error));
        }
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Backward consuming the forward's saved per-row moments; output_mask picks
// which gradients are produced (undefined tensors for the skipped ones).
std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_cuda(
        const Tensor& grad_output,
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const Tensor& mean,
        const Tensor& rstd,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt,
        const std::vector<bool>& output_mask) {
    const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim)
        TP_THROW(RuntimeError, "native_layer_norm_backward: normalized_shape dim larger than input dim");
    if (grad_output.dtype() != input.dtype())
        TP_THROW(RuntimeError, "native_layer_norm_backward: grad_output dtype must match input dtype");

    const int64_t outer_dims = input_ndim - norm_ndim;
    int64_t N = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[i])
            TP_THROW(RuntimeError, "native_layer_norm_backward: Input shape mismatch with normalized_shape");
        N *= normalized_shape[i];
    }
    const int64_t M = input.numel() / (N == 0 ? 1 : N);
    if (mean.numel() != M || rstd.numel() != M)
        TP_THROW(RuntimeError, "native_layer_norm_backward: saved moments do not match the row count");

    const bool need_dx = output_mask.empty() || output_mask[0];
    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    const bool need_dw = output_mask.size() > 1 && output_mask[1] && has_weight;
    const bool need_db = output_mask.size() > 2 && output_mask[2] && has_bias;

    Tensor grad_out_contig = grad_output.contiguous();
    Tensor in_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor mean_c = mean.contiguous();
    Tensor rstd_c = rstd.contiguous();

    Tensor grad_input = Tensor();
    if (need_dx) {
        grad_input = Tensor::empty(static_cast<std::vector<int64_t>>(in_contig.shape()),
                                   in_contig.dtype(), in_contig.device());
    }
    Tensor grad_weight = Tensor();
    if (need_dw) {
        grad_weight = Tensor::empty(static_cast<std::vector<int64_t>>(weight.shape()),
                                    weight.dtype(), weight.device());
    }
    Tensor grad_bias = Tensor();
    if (need_db) {
        const Tensor& like = has_weight ? weight : *bias_opt;
        grad_bias = Tensor::empty(static_cast<std::vector<int64_t>>(like.shape()),
                                  like.dtype(), like.device());
    }
    if (in_contig.numel() == 0 || M == 0 || N == 0)
        return std::make_tuple(grad_input, grad_weight, grad_bias);

    switch (in_contig.dtype()) {
#define LN_BACKWARD_STATS_CASE(ctype, name, acc_t)                            \
        case DType::name:                                                     \
            layer_norm::launch_layer_norm_backward_stats<ctype, acc_t>(       \
                M, N,                                                         \
                grad_out_contig.data_ptr<ctype>(),                            \
                in_contig.data_ptr<ctype>(),                                  \
                has_weight ? weight.data_ptr<ctype>() : nullptr,              \
                mean_c.data_ptr<acc_t>(), rstd_c.data_ptr<acc_t>(),           \
                need_dx ? grad_input.data_ptr<ctype>() : nullptr,             \
                need_dw ? grad_weight.data_ptr<ctype>() : nullptr,            \
                need_db ? grad_bias.data_ptr<ctype>() : nullptr);             \
            break;
        LN_BACKWARD_STATS_CASE(float, Float32, float)
        LN_BACKWARD_STATS_CASE(double, Float64, double)
        LN_BACKWARD_STATS_CASE(Half, Float16, float)
        LN_BACKWARD_STATS_CASE(BFloat16, BFloat16, float)
#undef LN_BACKWARD_STATS_CASE
        default:
            TP_THROW(NotImplementedError,
                     "native_layer_norm_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    {
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError, std::string("native_layer_norm_backward: ") + cudaGetErrorString(error));
        }
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LayerNormKernels) {
    m.impl("layer_norm", layer_norm_cuda);
    m.impl("native_layer_norm", native_layer_norm_cuda);
    m.impl("native_layer_norm_backward", native_layer_norm_backward_cuda);
    m.impl("layer_norm_backward", layer_norm_backward_cuda);
}

}
}
