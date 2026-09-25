#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "NormalizationLayerNormUtils.cuh"

namespace tensorplay {
namespace cuda {

static void check_normalization_cuda_status(cudaError_t error,
                                           const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(operation) + ": " +
                                   cudaGetErrorString(error));
    }
}

static void check_normalization_cuda_launch(const char* operation) {
    check_normalization_cuda_status(cudaGetLastError(), operation);
}

// ===========================================================================
// GroupNorm / InstanceNorm.
// Layout: input (N, C, ...) contiguous; one row == one (n, g) group with
// inner = (C/G) * spatial elements.  Reuses the layer-norm block reducers.
// ===========================================================================

namespace {

using layer_norm::kLNThreads;
using layer_norm::ln_block_reduce2;
using layer_norm::ln_rsqrt;

inline unsigned group_norm_threads_for(int64_t work) {
    return work < kLNThreads ? 32u : static_cast<unsigned>(kLNThreads);
}

template <typename T, typename ACC>
__global__ void group_norm_forward_impl(int64_t inner, int64_t spatial,
                                        int64_t cpg, int64_t num_groups,
                                        ACC eps, const T* __restrict__ X,
                                        const T* __restrict__ gamma,
                                        const T* __restrict__ beta,
                                        T* __restrict__ Y,
                                        ACC* __restrict__ mean_out,
                                        ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;              // n * num_groups + g
    const int64_t g = row % num_groups;
    const T* x = X + row * inner;
    T* y = Y + row * inner;

    ACC s = ACC(0), sq = ACC(0);
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC v = static_cast<ACC>(x[j]);
        s += v;
        sq += v * v;
    }
    layer_norm::ln_block_reduce2(s, sq, smem0, smem1);
    const ACC mean = s / static_cast<ACC>(inner);
    const ACC var = sq / static_cast<ACC>(inner) - mean * mean;
    const ACC rstd = layer_norm::ln_rsqrt(var + eps);
    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const int64_t c_local = g * cpg + j / spatial;   // global channel
        const ACC w = gamma ? static_cast<ACC>(gamma[c_local]) : ACC(1);
        const ACC b = beta ? static_cast<ACC>(beta[c_local]) : ACC(0);
        y[j] = static_cast<T>((static_cast<ACC>(x[j]) - mean) * rstd * w + b);
    }
}

template <typename T, int V>
struct alignas(sizeof(T) * V) GNVec {
    T v[V];
};

template <typename T, typename ACC, int V, bool ChannelPacked>
__global__ void group_norm_forward_mixed_vec_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    ACC eps, const T* __restrict__ X, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ Y,
    ACC* __restrict__ mean_out, ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    const int64_t nv = inner / V;
    ACC sum = ACC(0), sum_sq = ACC(0);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> pack = xv[j];
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const ACC value = static_cast<ACC>(pack.v[k]);
            sum += value;
            sum_sq += value * value;
        }
    }
    layer_norm::ln_block_reduce2(sum, sum_sq, smem0, smem1);
    const ACC mean = sum / static_cast<ACC>(inner);
    const ACC variance = sum_sq / static_cast<ACC>(inner) - mean * mean;
    const ACC rstd = layer_norm::ln_rsqrt(variance + eps);
    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    GNVec<float, V>* yv = reinterpret_cast<GNVec<float, V>*>(Y + row * inner);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> pack = xv[j];
        GNVec<float, V> out;
        if constexpr (ChannelPacked) {
            const int64_t channel = g * cpg + j / (spatial / V);
            const ACC weight = gamma ? static_cast<ACC>(gamma[channel]) : ACC(1);
            const ACC bias = beta ? static_cast<ACC>(beta[channel]) : ACC(0);
#pragma unroll
            for (int k = 0; k < V; ++k) {
                out.v[k] = static_cast<float>(
                    (static_cast<ACC>(pack.v[k]) - mean) * rstd * weight + bias);
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) {
                const int64_t idx = j * V + k;
                const int64_t channel = g * cpg + idx / spatial;
                const ACC weight = gamma ? static_cast<ACC>(gamma[channel]) : ACC(1);
                const ACC bias = beta ? static_cast<ACC>(beta[channel]) : ACC(0);
                out.v[k] = static_cast<float>(
                    (static_cast<ACC>(pack.v[k]) - mean) * rstd * weight + bias);
            }
        }
        yv[j] = out;
    }
}

template <typename T, typename ACC>
__global__ void group_norm_forward_mixed_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    ACC eps, const T* __restrict__ X, const float* __restrict__ gamma,
    const float* __restrict__ beta, float* __restrict__ Y,
    ACC* __restrict__ mean_out, ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    const T* x = X + row * inner;
    float* y = Y + row * inner;
    ACC sum = ACC(0), sum_sq = ACC(0);
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC value = static_cast<ACC>(x[j]);
        sum += value;
        sum_sq += value * value;
    }
    layer_norm::ln_block_reduce2(sum, sum_sq, smem0, smem1);
    const ACC mean = sum / static_cast<ACC>(inner);
    const ACC variance = sum_sq / static_cast<ACC>(inner) - mean * mean;
    const ACC rstd = layer_norm::ln_rsqrt(variance + eps);
    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const int64_t channel = g * cpg + j / spatial;
        const ACC weight = gamma ? static_cast<ACC>(gamma[channel]) : ACC(1);
        const ACC bias = beta ? static_cast<ACC>(beta[channel]) : ACC(0);
        y[j] = static_cast<float>(
            (static_cast<ACC>(x[j]) - mean) * rstd * weight + bias);
    }
}

// grad_input: dx = rstd/inner * (inner*dy - sum(dy) - xhat*sum(dy*xhat)).
template <typename T, typename ACC>
__global__ void group_norm_grad_input_impl(int64_t inner, int64_t spatial,
                                           int64_t num_groups,
                                           const T* __restrict__ dY,
                                           const T* __restrict__ X,
                                           const ACC* __restrict__ mean,
                                           const ACC* __restrict__ rstd,
                                           T* __restrict__ dX) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t off = row * inner;
    const ACC m = mean[row];
    const ACC r = rstd[row];

    ACC s_dy = ACC(0), s_dy_xhat = ACC(0);
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC d = static_cast<ACC>(dY[off + j]);
        s_dy += d;
        s_dy_xhat += d * (static_cast<ACC>(X[off + j]) - m) * r;
    }
    layer_norm::ln_block_reduce2(s_dy, s_dy_xhat, smem0, smem1);
    const ACC k = s_dy;
    const ACC kx = s_dy_xhat;

    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC d = static_cast<ACC>(dY[off + j]);
        const ACC xhat = (static_cast<ACC>(X[off + j]) - m) * r;
        dX[off + j] = static_cast<T>(
            r / static_cast<ACC>(inner) *
            (static_cast<ACC>(inner) * d - k - xhat * kx));
    }
}

template <typename T, typename ACC>
__global__ void group_norm_dx_apply_impl(int64_t inner, int64_t spatial,
                                          int64_t cpg, int64_t num_groups,
                                          const T* __restrict__ dY,
                                          const T* __restrict__ X,
                                          const ACC* __restrict__ rstd,
                                          const T* __restrict__ gamma,
                                          const ACC* __restrict__ c2,
                                          const ACC* __restrict__ c3,
                                          T* __restrict__ dX) {
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    const ACC r = rstd[row];
    const ACC a2 = c2[row];
    const ACC a3 = c3[row];
    const T* dy = dY + row * inner;
    const T* x = X + row * inner;
    T* dx = dX + row * inner;
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const int64_t c_local = g * cpg + j / spatial;
        const ACC c1 = r * (gamma ? static_cast<ACC>(gamma[c_local]) : ACC(1));
        dx[j] = static_cast<T>(
            c1 * static_cast<ACC>(dy[j]) + a2 * static_cast<ACC>(x[j]) + a3);
    }
}

// dgamma/dbeta: one block per channel; reduce over N samples and spatial.
template <typename T, typename ACC>
__global__ void group_norm_gamma_beta_impl(int64_t N, int64_t C, int64_t spatial,
                                           int64_t num_groups, int64_t cpg,
                                           const T* __restrict__ dY,
                                           const T* __restrict__ X,
                                           const ACC* __restrict__ mean,
                                           const ACC* __restrict__ rstd,
                                           T* __restrict__ dgamma,
                                           T* __restrict__ dbeta) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t c = blockIdx.x;
    const int64_t g = c / cpg;
    ACC sg = ACC(0), sb = ACC(0);
    for (int64_t n = 0; n < N; ++n) {
        const int64_t row = n * num_groups + g;
        const ACC m = mean[row];
        const ACC r = rstd[row];
        const int64_t base = (n * C + c) * spatial;
        for (int64_t s = threadIdx.x; s < spatial; s += blockDim.x) {
            const ACC d = static_cast<ACC>(dY[base + s]);
            const ACC xhat = (static_cast<ACC>(X[base + s]) - m) * r;
            sg += d * xhat;
            sb += d;
        }
    }
    layer_norm::ln_block_reduce2(sg, sb, smem0, smem1);
    if (threadIdx.x == 0) {
        if (dgamma) dgamma[c] = static_cast<T>(sg);
        if (dbeta) dbeta[c] = static_cast<T>(sb);
    }
}

// ===========================================================================
// GroupNorm vectorized paths.  Every element loop moves 16 bytes per thread
// per iteration (float4 / double2), so the kernels run at memory bandwidth
// instead of being issue-bound on scalar loads.  The vector variants require
// the per-row element count to be a multiple of the vector width (every row
// base then stays naturally aligned); other shapes use the scalar kernels
// above.  Backward decomposes as: per-channel internal gradient sums, a tiny
// per-group fused-parameters pass, then an elementwise apply whose work per
// element is two multiply-adds on values already in registers.
// ===========================================================================

// Moments only (no normalized output write): shared by the forward pass and
// the backward pass's stats recomputation.
template <typename T, typename ACC, int V>
__global__ void group_norm_moments_vec_impl(int64_t inner, ACC eps,
                                            const T* __restrict__ X,
                                            ACC* __restrict__ mean_out,
                                            ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    const int64_t nv = inner / V;
    ACC s = ACC(0), sq = ACC(0);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> pack = xv[j];
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const ACC v = static_cast<ACC>(pack.v[k]);
            s += v;
            sq += v * v;
        }
    }
    layer_norm::ln_block_reduce2(s, sq, smem0, smem1);
    if (threadIdx.x == 0) {
        const ACC mean = s / static_cast<ACC>(inner);
        const ACC var = sq / static_cast<ACC>(inner) - mean * mean;
        mean_out[row] = mean;
        rstd_out[row] = layer_norm::ln_rsqrt(var + eps);
    }
}

// Scalar twin of the moments kernel for shapes the vector path cannot take.
template <typename T, typename ACC>
__global__ void group_norm_moments_impl(int64_t inner, ACC eps,
                                        const T* __restrict__ X,
                                        ACC* __restrict__ mean_out,
                                        ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const T* x = X + row * inner;
    ACC s = ACC(0), sq = ACC(0);
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const ACC v = static_cast<ACC>(x[j]);
        s += v;
        sq += v * v;
    }
    layer_norm::ln_block_reduce2(s, sq, smem0, smem1);
    if (threadIdx.x == 0) {
        const ACC mean = s / static_cast<ACC>(inner);
        const ACC var = sq / static_cast<ACC>(inner) - mean * mean;
        mean_out[row] = mean;
        rstd_out[row] = layer_norm::ln_rsqrt(var + eps);
    }
}

template <typename T, typename ACC, int V, bool ChannelPacked>
__global__ void group_norm_forward_vec_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    ACC eps, const T* __restrict__ X, const T* __restrict__ gamma,
    const T* __restrict__ beta, T* __restrict__ Y,
    ACC* __restrict__ mean_out, ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    GNVec<T, V>* yv = reinterpret_cast<GNVec<T, V>*>(Y + row * inner);
    const int64_t nv = inner / V;
    ACC sum = ACC(0), sum_sq = ACC(0);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> pack = xv[j];
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const ACC value = static_cast<ACC>(pack.v[k]);
            sum += value;
            sum_sq += value * value;
        }
    }
    layer_norm::ln_block_reduce2(sum, sum_sq, smem0, smem1);
    const ACC mean = sum / static_cast<ACC>(inner);
    const ACC variance = sum_sq / static_cast<ACC>(inner) - mean * mean;
    const ACC rstd = layer_norm::ln_rsqrt(variance + eps);
    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        GNVec<T, V> pack = xv[j];
        if constexpr (ChannelPacked) {
            const int64_t channel = g * cpg + j / (spatial / V);
            const ACC weight = gamma ? static_cast<ACC>(gamma[channel]) : ACC(1);
            const ACC bias = beta ? static_cast<ACC>(beta[channel]) : ACC(0);
#pragma unroll
            for (int k = 0; k < V; ++k) {
                pack.v[k] = static_cast<T>(
                    (static_cast<ACC>(pack.v[k]) - mean) * rstd * weight + bias);
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) {
                const int64_t idx = j * V + k;
                const int64_t channel = g * cpg + idx / spatial;
                const ACC weight = gamma ? static_cast<ACC>(gamma[channel]) : ACC(1);
                const ACC bias = beta ? static_cast<ACC>(beta[channel]) : ACC(0);
                pack.v[k] = static_cast<T>(
                    (static_cast<ACC>(pack.v[k]) - mean) * rstd * weight + bias);
            }
        }
        yv[j] = pack;
    }
}

// Per-channel internal gradient sums: ds = sum(dy*x), db = sum(dy).  One
// block per (n, c) row of `spatial` elements.
template <typename T, typename ACC, int V>
__global__ void group_norm_internal_grads_vec_impl(int64_t spatial,
                                                   const T* __restrict__ dY,
                                                   const T* __restrict__ X,
                                                   ACC* __restrict__ ds_out,
                                                   ACC* __restrict__ db_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t nc = blockIdx.x;
    const GNVec<T, V>* dyv = reinterpret_cast<const GNVec<T, V>*>(dY + nc * spatial);
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + nc * spatial);
    const int64_t nv = spatial / V;
    ACC s1 = ACC(0), s2 = ACC(0);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> d = dyv[j];
        const GNVec<T, V> x = xv[j];
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const ACC dy = static_cast<ACC>(d.v[k]);
            s1 += dy * static_cast<ACC>(x.v[k]);
            s2 += dy;
        }
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        ds_out[nc] = s1;
        db_out[nc] = s2;
    }
}

template <typename T, typename ACC, int V>
__global__ void group_norm_internal_grads_tail_impl(int64_t spatial,
                                                     const T* __restrict__ dY,
                                                     const T* __restrict__ X,
                                                     ACC* __restrict__ ds_out,
                                                     ACC* __restrict__ db_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t nc = blockIdx.x;
    const T* dy = dY + nc * spatial;
    const T* x = X + nc * spatial;
    ACC s1 = ACC(0), s2 = ACC(0);
    const int64_t step = static_cast<int64_t>(blockDim.x) * V;
    for (int64_t base = static_cast<int64_t>(threadIdx.x) * V;
         base < spatial; base += step) {
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const int64_t j = base + k;
            if (j < spatial) {
                const ACC d = static_cast<ACC>(dy[j]);
                s1 += d * static_cast<ACC>(x[j]);
                s2 += d;
            }
        }
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        ds_out[nc] = s1;
        db_out[nc] = s2;
    }
}

template <typename T, typename ACC>
__global__ void group_norm_internal_grads_impl(int64_t spatial,
                                                const T* __restrict__ dY,
                                                const T* __restrict__ X,
                                                ACC* __restrict__ ds_out,
                                                ACC* __restrict__ db_out) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t nc = blockIdx.x;
    const T* dy = dY + nc * spatial;
    const T* x = X + nc * spatial;
    ACC s1 = ACC(0), s2 = ACC(0);
    for (int64_t j = threadIdx.x; j < spatial; j += blockDim.x) {
        const ACC d = static_cast<ACC>(dy[j]);
        s1 += d * static_cast<ACC>(x[j]);
        s2 += d;
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        ds_out[nc] = s1;
        db_out[nc] = s2;
    }
}

template <typename T, int V>
__global__ void group_norm_internal_grads_mixed_vec_impl(
    int64_t spatial, const float* __restrict__ dY, const T* __restrict__ X,
    float* __restrict__ ds_out, float* __restrict__ db_out) {
    __shared__ float smem0[layer_norm::kLNThreads / 32];
    __shared__ float smem1[layer_norm::kLNThreads / 32];
    const int64_t nc = blockIdx.x;
    const GNVec<float, V>* dyv =
        reinterpret_cast<const GNVec<float, V>*>(dY + nc * spatial);
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + nc * spatial);
    const int64_t nv = spatial / V;
    float s1 = 0.0f, s2 = 0.0f;
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<float, V> d = dyv[j];
        const GNVec<T, V> x = xv[j];
#pragma unroll
        for (int k = 0; k < V; ++k) {
            s1 += d.v[k] * static_cast<float>(x.v[k]);
            s2 += d.v[k];
        }
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        ds_out[nc] = s1;
        db_out[nc] = s2;
    }
}

template <typename T>
__global__ void group_norm_internal_grads_mixed_impl(
    int64_t spatial, const float* __restrict__ dY, const T* __restrict__ X,
    float* __restrict__ ds_out, float* __restrict__ db_out) {
    __shared__ float smem0[layer_norm::kLNThreads / 32];
    __shared__ float smem1[layer_norm::kLNThreads / 32];
    const int64_t nc = blockIdx.x;
    const float* dy = dY + nc * spatial;
    const T* x = X + nc * spatial;
    float s1 = 0.0f, s2 = 0.0f;
    const int64_t block_size = blockDim.x;
    for (int64_t base = 0; base < spatial; base += 2 * block_size) {
        const int64_t j0 = base + threadIdx.x;
        const int64_t j1 = j0 + block_size;
        float d0 = 0.0f, d1 = 0.0f;
        float x0 = 0.0f, x1 = 0.0f;
        if (j0 < spatial) {
            d0 = dy[j0];
            x0 = static_cast<float>(x[j0]);
        }
        if (j1 < spatial) {
            d1 = dy[j1];
            x1 = static_cast<float>(x[j1]);
        }
        if (j0 < spatial) {
            s1 += d0 * x0;
            s2 += d0;
        }
        if (j1 < spatial) {
            s1 += d1 * x1;
            s2 += d1;
        }
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        ds_out[nc] = s1;
        db_out[nc] = s2;
    }
}

// Per-group fused backward coefficients from the channel sums.  With
// sum1 = sum_c ds[c]*gamma[c], sum2 = sum_c db[c]*gamma[c] and s = 1/inner:
//   c2 = (sum2*mean - sum1) * rstd^3 * s
//   c3 = -c2*mean - sum2 * rstd * s
// so the elementwise pass is dx = (rstd*gamma)*dy + c2*x + c3.
template <typename T, typename ACC>
__global__ void group_norm_bwd_fused_params_impl(int64_t cpg, int64_t inner,
                                                 int64_t num_groups,
                                                 const ACC* __restrict__ mean,
                                                 const ACC* __restrict__ rstd,
                                                 const T* __restrict__ gamma,
                                                 const ACC* __restrict__ ds,
                                                 const ACC* __restrict__ db,
                                                 ACC* __restrict__ c2,
                                                 ACC* __restrict__ c3) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;             // n * num_groups + g
    const int64_t g = row % num_groups;
    ACC s1 = ACC(0), s2 = ACC(0);
    for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
        const int64_t c = g * cpg + i;
        const ACC gv = gamma ? static_cast<ACC>(gamma[c]) : ACC(1);
        s1 += ds[row * cpg + i] * gv;
        s2 += db[row * cpg + i] * gv;
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    if (threadIdx.x == 0) {
        const ACC s1v = s1;
        const ACC s2v = s2;
        const ACC mm = mean[row];
        const ACC r = rstd[row];
        const ACC sc = ACC(1) / static_cast<ACC>(inner);
        const ACC x = (s2v * mm - s1v) * r * r * r * sc;
        c2[row] = x;
        c3[row] = -x * mm - s2v * r * sc;
    }
}

template <typename T, typename ACC, int V>
__global__ void group_norm_dx_apply_vec_impl(int64_t inner, int64_t spatial,
                                             int64_t cpg, int64_t num_groups,
                                             const T* __restrict__ dY,
                                             const T* __restrict__ X,
                                             const ACC* __restrict__ rstd,
                                             const T* __restrict__ gamma,
                                             const ACC* __restrict__ c2,
                                             const ACC* __restrict__ c3,
                                             T* __restrict__ dX) {
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    const ACC r = rstd[row];
    const ACC a2 = c2[row];
    const ACC a3 = c3[row];
    const GNVec<T, V>* dyv = reinterpret_cast<const GNVec<T, V>*>(dY + row * inner);
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    GNVec<T, V>* dxv = reinterpret_cast<GNVec<T, V>*>(dX + row * inner);
    const int64_t nv = inner / V;
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> d = dyv[j];
        const GNVec<T, V> x = xv[j];
        GNVec<T, V> o;
#pragma unroll
        for (int k = 0; k < V; ++k) {
            const int64_t idx = j * V + k;
            const int64_t c_local = g * cpg + idx / spatial;
            const ACC c1 = r * (gamma ? static_cast<ACC>(gamma[c_local]) : ACC(1));
            o.v[k] = static_cast<T>(
                c1 * static_cast<ACC>(d.v[k]) + a2 * static_cast<ACC>(x.v[k]) +
                a3);
        }
        dxv[j] = o;
    }
}

template <typename T, typename ACC, int V, bool ChannelPacked>
__global__ void group_norm_dx_fused_vec_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    const T* __restrict__ dY, const T* __restrict__ X,
    const ACC* __restrict__ mean, const ACC* __restrict__ rstd,
    const T* __restrict__ gamma, const ACC* __restrict__ ds,
    const ACC* __restrict__ db, T* __restrict__ dX) {
    __shared__ ACC smem0[layer_norm::kLNThreads / 32];
    __shared__ ACC smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    ACC s1 = ACC(0), s2 = ACC(0);
    for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
        const int64_t c = g * cpg + i;
        const ACC gv = gamma ? static_cast<ACC>(gamma[c]) : ACC(1);
        s1 += ds[row * cpg + i] * gv;
        s2 += db[row * cpg + i] * gv;
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    const ACC mm = mean[row];
    const ACC r = rstd[row];
    const ACC scale = ACC(1) / static_cast<ACC>(inner);
    const ACC a2 = (s2 * mm - s1) * r * r * r * scale;
    const ACC a3 = -a2 * mm - s2 * r * scale;

    const GNVec<T, V>* dyv = reinterpret_cast<const GNVec<T, V>*>(dY + row * inner);
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    GNVec<T, V>* dxv = reinterpret_cast<GNVec<T, V>*>(dX + row * inner);
    const int64_t nv = inner / V;
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<T, V> d = dyv[j];
        const GNVec<T, V> x = xv[j];
        GNVec<T, V> o;
        if constexpr (ChannelPacked) {
            const int64_t c_local = g * cpg + j / (spatial / V);
            const ACC c1 = r * (gamma ? static_cast<ACC>(gamma[c_local]) : ACC(1));
#pragma unroll
            for (int k = 0; k < V; ++k) {
                o.v[k] = static_cast<T>(
                    c1 * static_cast<ACC>(d.v[k]) + a2 * static_cast<ACC>(x.v[k]) +
                    a3);
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) {
                const int64_t idx = j * V + k;
                const int64_t c_local = g * cpg + idx / spatial;
                const ACC c1 = r * (gamma ? static_cast<ACC>(gamma[c_local]) : ACC(1));
                o.v[k] = static_cast<T>(
                    c1 * static_cast<ACC>(d.v[k]) + a2 * static_cast<ACC>(x.v[k]) +
                    a3);
            }
        }
        dxv[j] = o;
    }
}

template <typename T, int V, bool ChannelPacked>
__global__ void group_norm_dx_fused_mixed_vec_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    const float* __restrict__ dY, const T* __restrict__ X,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ ds,
    const float* __restrict__ db, T* __restrict__ dX) {
    __shared__ float smem0[layer_norm::kLNThreads / 32];
    __shared__ float smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    float s1 = 0.0f, s2 = 0.0f;
    for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
        const int64_t c = g * cpg + i;
        const float gv = gamma ? gamma[c] : 1.0f;
        s1 += ds[row * cpg + i] * gv;
        s2 += db[row * cpg + i] * gv;
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    const float m = mean[row];
    const float r = rstd[row];
    const float scale = 1.0f / static_cast<float>(inner);
    const float a2 = (s2 * m - s1) * r * r * r * scale;
    const float a3 = -a2 * m - s2 * r * scale;
    const GNVec<float, V>* dyv =
        reinterpret_cast<const GNVec<float, V>*>(dY + row * inner);
    const GNVec<T, V>* xv = reinterpret_cast<const GNVec<T, V>*>(X + row * inner);
    GNVec<T, V>* dxv = reinterpret_cast<GNVec<T, V>*>(dX + row * inner);
    const int64_t nv = inner / V;
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) {
        const GNVec<float, V> d = dyv[j];
        const GNVec<T, V> x = xv[j];
        GNVec<T, V> out;
        if constexpr (ChannelPacked) {
            const int64_t channel = g * cpg + j / (spatial / V);
            const float c1 = r * (gamma ? gamma[channel] : 1.0f);
#pragma unroll
            for (int k = 0; k < V; ++k) {
                out.v[k] = static_cast<T>(
                    c1 * d.v[k] + a2 * static_cast<float>(x.v[k]) + a3);
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) {
                const int64_t idx = j * V + k;
                const int64_t channel = g * cpg + idx / spatial;
                const float c1 = r * (gamma ? gamma[channel] : 1.0f);
                out.v[k] = static_cast<T>(
                    c1 * d.v[k] + a2 * static_cast<float>(x.v[k]) + a3);
            }
        }
        dxv[j] = out;
    }
}

template <typename T>
__global__ void group_norm_dx_fused_mixed_impl(
    int64_t inner, int64_t spatial, int64_t cpg, int64_t num_groups,
    const float* __restrict__ dY, const T* __restrict__ X,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    const float* __restrict__ gamma, const float* __restrict__ ds,
    const float* __restrict__ db, T* __restrict__ dX) {
    __shared__ float smem0[layer_norm::kLNThreads / 32];
    __shared__ float smem1[layer_norm::kLNThreads / 32];
    const int64_t row = blockIdx.x;
    const int64_t g = row % num_groups;
    float s1 = 0.0f, s2 = 0.0f;
    for (int64_t i = threadIdx.x; i < cpg; i += blockDim.x) {
        const int64_t c = g * cpg + i;
        const float gv = gamma ? gamma[c] : 1.0f;
        s1 += ds[row * cpg + i] * gv;
        s2 += db[row * cpg + i] * gv;
    }
    layer_norm::ln_block_reduce2(s1, s2, smem0, smem1);
    const float m = mean[row];
    const float r = rstd[row];
    const float scale = 1.0f / static_cast<float>(inner);
    const float a2 = (s2 * m - s1) * r * r * r * scale;
    const float a3 = -a2 * m - s2 * r * scale;
    const T* x = X + row * inner;
    T* dx = dX + row * inner;
    for (int64_t j = threadIdx.x; j < inner; j += blockDim.x) {
        const int64_t channel = g * cpg + j / spatial;
        const float c1 = r * (gamma ? gamma[channel] : 1.0f);
        dx[j] = static_cast<T>(
            c1 * dY[row * inner + j] + a2 * static_cast<float>(x[j]) + a3);
    }
}

// dgamma/dbeta from the per-channel sums; dY and X are not re-read.
template <typename ACC>
__global__ void group_norm_gamma_beta_sums_impl(int64_t N, int64_t C,
                                                int64_t num_groups, int64_t cpg,
                                                const ACC* __restrict__ mean,
                                                const ACC* __restrict__ rstd,
                                                const ACC* __restrict__ ds,
                                                const ACC* __restrict__ db,
                                                ACC* __restrict__ dgamma,
                                                ACC* __restrict__ dbeta) {
    constexpr int kThreadsX = 32;
    constexpr int kThreadsY = 16;
    __shared__ ACC gamma_sums[2 * kThreadsY][kThreadsX + 1];
    __shared__ ACC beta_sums[2 * kThreadsY][kThreadsX + 1];
    const int64_t c = static_cast<int64_t>(blockIdx.x) * kThreadsX + threadIdx.x;
    ACC gamma0 = ACC(0), gamma1 = ACC(0);
    ACC beta0 = ACC(0), beta1 = ACC(0);
    if (c < C) {
        const int64_t g = c / cpg;
        for (int64_t n = threadIdx.y; n < N; n += 2 * kThreadsY) {
            const int64_t nc = n * C + c;
            const int64_t ng = n * num_groups + g;
            gamma0 += (ds[nc] - db[nc] * mean[ng]) * rstd[ng];
            beta0 += db[nc];
            const int64_t n1 = n + kThreadsY;
            if (n1 < N) {
                const int64_t nc1 = n1 * C + c;
                const int64_t ng1 = n1 * num_groups + g;
                gamma1 += (ds[nc1] - db[nc1] * mean[ng1]) * rstd[ng1];
                beta1 += db[nc1];
            }
        }
    }
    gamma_sums[threadIdx.y][threadIdx.x] = gamma0;
    gamma_sums[kThreadsY + threadIdx.y][threadIdx.x] = gamma1;
    beta_sums[threadIdx.y][threadIdx.x] = beta0;
    beta_sums[kThreadsY + threadIdx.y][threadIdx.x] = beta1;
    __syncthreads();
    if (threadIdx.y == 0 && c < C) {
        ACC gamma_total = ACC(0), beta_total = ACC(0);
#pragma unroll
        for (int y = 0; y < 2 * kThreadsY; ++y) {
            gamma_total += gamma_sums[y][threadIdx.x];
            beta_total += beta_sums[y][threadIdx.x];
        }
        if (dgamma) dgamma[c] = gamma_total;
        if (dbeta) dbeta[c] = beta_total;
    }
}

namespace {

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_forward_dispatch(
    const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps) {
    using ACC = typename std::conditional<std::is_same<T, float>::value,
                                          float, double>::type;
    const DType acc_dt =
        std::is_same<ACC, float>::value ? DType::Float32 : DType::Float64;

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t cpg = C / num_groups;
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    const int64_t inner = cpg * spatial;

    Tensor out = Tensor::empty_like(input);
    Tensor mean = Tensor::empty({N, num_groups}, acc_dt, input.device());
    Tensor rstd = Tensor::empty({N, num_groups}, acc_dt, input.device());
    ACC* mean_p = mean.data_ptr<ACC>();
    ACC* rstd_p = rstd.data_ptr<ACC>();

    if (N == 0) return {out, mean, rstd};

    const T* w = nullptr;
    const T* b = nullptr;
    if (weight_opt.has_value() && weight_opt->defined()) w = weight_opt->data_ptr<T>();
    if (bias_opt.has_value() && bias_opt->defined()) b = bias_opt->data_ptr<T>();

    const auto stream = getCurrentCUDAStream().stream();
    constexpr int kVec = sizeof(T) <= 4 ? 4 : 2;
    const unsigned threads = group_norm_threads_for(inner);
    const T* x_p = input.data_ptr<T>();
    T* y_p = out.data_ptr<T>();
    const bool vec_ok =
        inner % kVec == 0 &&
        reinterpret_cast<uintptr_t>(x_p) % (sizeof(T) * kVec) == 0 &&
        reinterpret_cast<uintptr_t>(y_p) % (sizeof(T) * kVec) == 0;
    if (vec_ok) {
        if (spatial % kVec == 0) {
            group_norm_forward_vec_impl<T, ACC, kVec, true>
                <<<N * num_groups, threads, 0, stream>>>(
                    inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
                    x_p, w, b, y_p, mean_p, rstd_p);
        } else {
            group_norm_forward_vec_impl<T, ACC, kVec, false>
                <<<N * num_groups, threads, 0, stream>>>(
                    inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
                    x_p, w, b, y_p, mean_p, rstd_p);
        }
        check_normalization_cuda_launch("group_norm forward");
    } else {
        group_norm_forward_impl<T, ACC>
            <<<N * num_groups, threads, 0, stream>>>(
            inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
            x_p, w, b, y_p, mean_p, rstd_p);
        check_normalization_cuda_launch("group_norm forward");
    }
    return {out, mean, rstd};
}

template <typename T>
std::tuple<Tensor, Tensor, Tensor> group_norm_forward_mixed_dispatch(
    const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps) {
    using ACC = float;
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t cpg = C / num_groups;
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    const int64_t inner = cpg * spatial;
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()),
                               DType::Float32, input.device());
    Tensor mean = Tensor::empty({N, num_groups}, DType::Float32, input.device());
    Tensor rstd = Tensor::empty({N, num_groups}, DType::Float32, input.device());
    if (N == 0) return {out, mean, rstd};

    const float* gamma = nullptr;
    const float* beta = nullptr;
    if (weight_opt.has_value() && weight_opt->defined()) {
        if (weight_opt->dtype() != DType::Float32) {
            TP_THROW(RuntimeError, "mixed group_norm weight must be Float32");
        }
        gamma = weight_opt->data_ptr<float>();
    }
    if (bias_opt.has_value() && bias_opt->defined()) {
        if (bias_opt->dtype() != DType::Float32) {
            TP_THROW(RuntimeError, "mixed group_norm bias must be Float32");
        }
        beta = bias_opt->data_ptr<float>();
    }
    const auto stream = getCurrentCUDAStream().stream();
    constexpr int kVec = sizeof(T) <= 4 ? 4 : 2;
    const unsigned threads = group_norm_threads_for(inner);
    const T* x_p = input.data_ptr<T>();
    float* y_p = out.data_ptr<float>();
    const bool vec_ok =
        inner % kVec == 0 &&
        reinterpret_cast<uintptr_t>(x_p) % (sizeof(T) * kVec) == 0 &&
        reinterpret_cast<uintptr_t>(y_p) % (sizeof(float) * kVec) == 0;
    if (vec_ok) {
        if (spatial % kVec == 0) {
            group_norm_forward_mixed_vec_impl<T, ACC, kVec, true>
                <<<N * num_groups, threads, 0, stream>>>(
                    inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
                    x_p, gamma, beta, y_p, mean.data_ptr<ACC>(),
                    rstd.data_ptr<ACC>());
        } else {
            group_norm_forward_mixed_vec_impl<T, ACC, kVec, false>
                <<<N * num_groups, threads, 0, stream>>>(
                    inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
                    x_p, gamma, beta, y_p, mean.data_ptr<ACC>(),
                    rstd.data_ptr<ACC>());
        }
    } else {
        group_norm_forward_mixed_impl<T, ACC>
            <<<N * num_groups, threads, 0, stream>>>(
                inner, spatial, cpg, num_groups, static_cast<ACC>(eps),
                x_p, gamma, beta, y_p, mean.data_ptr<ACC>(), rstd.data_ptr<ACC>());
    }
    check_normalization_cuda_launch("group_norm mixed forward");
    return {out, mean, rstd};
}

} // namespace
} // namespace

Tensor group_norm_cuda(const Tensor& input, int64_t num_groups,
                       const std::optional<Tensor>& weight_opt,
                       const std::optional<Tensor>& bias_opt, double eps) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "group_norm requires at least 2 dims");
    }
    const int64_t C = input.size(1);
    if (C <= 0 || num_groups <= 0 || C % num_groups != 0) {
        TP_THROW(RuntimeError,
                 "group_norm: num_channels must be divisible by num_groups");
    }
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        if (input.size(dim) <= 0) {
            TP_THROW(RuntimeError,
                     "group_norm requires non-empty spatial dimensions");
        }
    }
    Tensor x = input.contiguous();

    std::optional<Tensor> mixed_weight = weight_opt;
    std::optional<Tensor> mixed_bias = bias_opt;
    if (mixed_weight.has_value() && mixed_weight->defined() &&
        !mixed_weight->is_contiguous()) {
        mixed_weight = mixed_weight->contiguous();
    }
    if (mixed_bias.has_value() && mixed_bias->defined() &&
        !mixed_bias->is_contiguous()) {
        mixed_bias = mixed_bias->contiguous();
    }

    if (x.dtype() == DType::Float16) {
        return std::get<0>(group_norm_forward_mixed_dispatch<Half>(
            x, num_groups, mixed_weight, mixed_bias, eps));
    }
    if (x.dtype() == DType::BFloat16) {
        return std::get<0>(group_norm_forward_mixed_dispatch<BFloat16>(
            x, num_groups, mixed_weight, mixed_bias, eps));
    }

    if (x.dtype() == DType::Float32) {
        return std::get<0>(group_norm_forward_dispatch<float>(
            x, num_groups, weight_opt, bias_opt, eps));
    }
    if (x.dtype() == DType::Float64) {
        return std::get<0>(group_norm_forward_dispatch<double>(
            x, num_groups, weight_opt, bias_opt, eps));
    }
    TP_THROW(NotImplementedError, "group_norm only supports Float32/Float64 on CUDA");
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_cuda(
    const Tensor& input, const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, int64_t N, int64_t C,
    int64_t HxW, int64_t num_groups, double eps) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "native_group_norm requires at least 2 dims");
    }
    int64_t actual_hxw = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        actual_hxw *= input.size(dim);
    }
    if (N != input.size(0) || C != input.size(1) || HxW != actual_hxw) {
        TP_THROW(RuntimeError, "native_group_norm dimensions do not match input");
    }
    if (C <= 0 || HxW <= 0 || num_groups <= 0 || C % num_groups != 0) {
        TP_THROW(RuntimeError, "native_group_norm has invalid group dimensions");
    }
    auto check_parameter = [C](const std::optional<Tensor>& param,
                               const char* name) {
        if (!param.has_value() || !param->defined()) return;
        if (param->dim() != 1 || param->numel() != C) {
            TP_THROW(RuntimeError, std::string("native_group_norm ") + name +
                                     " must have one value per channel");
        }
    };
    check_parameter(weight_opt, "weight");
    check_parameter(bias_opt, "bias");

    Tensor x = input.contiguous();
    std::optional<Tensor> weight = weight_opt;
    std::optional<Tensor> bias = bias_opt;
    if (weight.has_value() && weight->defined() && !weight->is_contiguous()) {
        weight = weight->contiguous();
    }
    if (bias.has_value() && bias->defined() && !bias->is_contiguous()) {
        bias = bias->contiguous();
    }
    if (x.dtype() == DType::Float16) {
        return group_norm_forward_mixed_dispatch<Half>(
            x, num_groups, weight, bias, eps);
    }
    if (x.dtype() == DType::BFloat16) {
        return group_norm_forward_mixed_dispatch<BFloat16>(
            x, num_groups, weight, bias, eps);
    }
    if (x.dtype() == DType::Float32) {
        return group_norm_forward_dispatch<float>(x, num_groups, weight, bias,
                                                  eps);
    }
    if (x.dtype() == DType::Float64) {
        return group_norm_forward_dispatch<double>(x, num_groups, weight, bias,
                                                   eps);
    }
    TP_THROW(NotImplementedError,
             "native_group_norm only supports Float32/Float64 on CUDA");
}

template <typename T>
static std::tuple<Tensor, Tensor, Tensor> group_norm_backward_mixed_cuda_impl(
    const Tensor& grad_output, const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps,
    const Tensor* mean_saved, const Tensor* rstd_saved,
    const std::vector<bool>* output_mask) {
    if (grad_output.dtype() != DType::Float32) {
        TP_THROW(NotImplementedError,
                 "mixed group_norm_backward requires a Float32 gradient");
    }
    if (input.dim() < 2 || input.size(1) <= 0 || num_groups <= 0 ||
        input.size(1) % num_groups != 0) {
        TP_THROW(RuntimeError, "group_norm_backward has invalid group dimensions");
    }
    if (grad_output.dim() != input.dim()) {
        TP_THROW(RuntimeError,
                 "group_norm_backward gradient shape does not match input");
    }
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
        if (grad_output.size(dim) != input.size(dim)) {
            TP_THROW(RuntimeError,
                     "group_norm_backward gradient shape does not match input");
        }
    }
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t cpg = C / num_groups;
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    if (spatial <= 0) {
        TP_THROW(RuntimeError, "group_norm_backward requires non-empty spatial dimensions");
    }
    const int64_t inner = cpg * spatial;
    const bool need_dx = !output_mask ||
        (!output_mask->empty() && (*output_mask)[0]);
    const bool need_dw = output_mask
        ? (output_mask->size() > 1 && (*output_mask)[1])
        : (weight_opt.has_value() && weight_opt->defined());
    const bool need_db = output_mask
        ? (output_mask->size() > 2 && (*output_mask)[2])
        : (bias_opt.has_value() && bias_opt->defined());

    Tensor grad_output_contig = grad_output.is_contiguous()
        ? grad_output : grad_output.contiguous();
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_input = need_dx ? Tensor::empty_like(input_contig) : Tensor();
    Tensor grad_weight = need_dw
        ? Tensor::empty({C}, DType::Float32, input.device()) : Tensor();
    Tensor grad_bias = need_db
        ? Tensor::empty({C}, DType::Float32, input.device()) : Tensor();
    std::optional<Tensor> weight_contig = weight_opt;
    if (weight_contig.has_value() && weight_contig->defined()) {
        if (weight_contig->dtype() != DType::Float32) {
            TP_THROW(RuntimeError, "mixed group_norm weight must be Float32");
        }
        if (!weight_contig->is_contiguous()) {
            weight_contig = weight_contig->contiguous();
        }
    }
    if (bias_opt.has_value() && bias_opt->defined() &&
        bias_opt->dtype() != DType::Float32) {
        TP_THROW(RuntimeError, "mixed group_norm bias must be Float32");
    }

    const auto stream = getCurrentCUDAStream().stream();
    if (N == 0) {
        if (grad_weight.defined()) {
            check_normalization_cuda_status(
                cudaMemsetAsync(grad_weight.data_ptr(), 0,
                                C * sizeof(float), stream),
                "group_norm mixed backward weight initialization");
        }
        if (grad_bias.defined()) {
            check_normalization_cuda_status(
                cudaMemsetAsync(grad_bias.data_ptr(), 0,
                                C * sizeof(float), stream),
                "group_norm mixed backward bias initialization");
        }
        return {grad_input, grad_weight, grad_bias};
    }

    Tensor mean_saved_contig;
    Tensor rstd_saved_contig;
    if (mean_saved && mean_saved->defined()) {
        if (mean_saved->dtype() != DType::Float32) {
            TP_THROW(RuntimeError, "mixed group_norm mean must be Float32");
        }
        mean_saved_contig = mean_saved->is_contiguous()
            ? *mean_saved : mean_saved->contiguous();
    }
    if (rstd_saved && rstd_saved->defined()) {
        if (rstd_saved->dtype() != DType::Float32) {
            TP_THROW(RuntimeError, "mixed group_norm rstd must be Float32");
        }
        rstd_saved_contig = rstd_saved->is_contiguous()
            ? *rstd_saved : rstd_saved->contiguous();
    }
    if (mean_saved_contig.defined() != rstd_saved_contig.defined()) {
        TP_THROW(RuntimeError, "group_norm_backward requires both saved statistics");
    }

    Tensor computed_stats;
    float* mean_p = nullptr;
    float* rstd_p = nullptr;
    if (mean_saved_contig.defined()) {
        mean_p = mean_saved_contig.data_ptr<float>();
        rstd_p = rstd_saved_contig.data_ptr<float>();
    } else {
        computed_stats = Tensor::empty({N * num_groups * 2}, DType::Float32,
                                        input.device());
        mean_p = computed_stats.data_ptr<float>();
        rstd_p = mean_p + N * num_groups;
        group_norm_moments_impl<T, float>
            <<<N * num_groups, group_norm_threads_for(inner), 0, stream>>>(
                inner, static_cast<float>(eps), input_contig.data_ptr<T>(),
                mean_p, rstd_p);
        check_normalization_cuda_launch("group_norm mixed backward moments");
    }

    Tensor sums = Tensor::empty({N * C * 2}, DType::Float32, input.device());
    float* ds_p = sums.data_ptr<float>();
    float* db_p = ds_p + N * C;
    constexpr int kVec = sizeof(T) <= 4 ? 4 : 2;
    const T* x_p = input_contig.data_ptr<T>();
    const float* dy_p = grad_output_contig.data_ptr<float>();
    const bool vec_ok =
        spatial % kVec == 0 &&
        reinterpret_cast<uintptr_t>(x_p) % (sizeof(T) * kVec) == 0 &&
        reinterpret_cast<uintptr_t>(dy_p) % (sizeof(float) * kVec) == 0;
    if (vec_ok) {
        group_norm_internal_grads_mixed_vec_impl<T, kVec>
            <<<N * C, group_norm_threads_for(spatial), 0, stream>>>(
                spatial, dy_p, x_p, ds_p, db_p);
    } else {
        group_norm_internal_grads_mixed_impl<T>
            <<<N * C, group_norm_threads_for(spatial), 0, stream>>>(
                spatial, dy_p, x_p, ds_p, db_p);
    }
    check_normalization_cuda_launch("group_norm mixed backward sums");

    const float* gamma = weight_contig.has_value() && weight_contig->defined()
        ? weight_contig->data_ptr<float>() : nullptr;
    if (need_dx) {
        T* dx_p = grad_input.data_ptr<T>();
        const bool dx_vec_ok = inner % kVec == 0 &&
            reinterpret_cast<uintptr_t>(x_p) % (sizeof(T) * kVec) == 0 &&
            reinterpret_cast<uintptr_t>(dy_p) % (sizeof(float) * kVec) == 0 &&
            reinterpret_cast<uintptr_t>(dx_p) % (sizeof(T) * kVec) == 0;
        if (dx_vec_ok) {
            if (spatial % kVec == 0) {
                group_norm_dx_fused_mixed_vec_impl<T, kVec, true>
                    <<<N * num_groups, group_norm_threads_for(inner), 0, stream>>>(
                        inner, spatial, cpg, num_groups, dy_p, x_p, mean_p, rstd_p,
                        gamma, ds_p, db_p, dx_p);
            } else {
                group_norm_dx_fused_mixed_vec_impl<T, kVec, false>
                    <<<N * num_groups, group_norm_threads_for(inner), 0, stream>>>(
                        inner, spatial, cpg, num_groups, dy_p, x_p, mean_p, rstd_p,
                        gamma, ds_p, db_p, dx_p);
            }
        } else {
            group_norm_dx_fused_mixed_impl<T>
                <<<N * num_groups, group_norm_threads_for(inner), 0, stream>>>(
                    inner, spatial, cpg, num_groups, dy_p, x_p, mean_p, rstd_p,
                    gamma, ds_p, db_p, dx_p);
        }
        check_normalization_cuda_launch("group_norm mixed backward input");
    }
    if (need_dw || need_db) {
        group_norm_gamma_beta_sums_impl<float>
            <<<(C + 31) / 32, dim3(32, 16), 0, stream>>>(
                N, C, num_groups, cpg, mean_p, rstd_p, ds_p, db_p,
                need_dw ? grad_weight.data_ptr<float>() : nullptr,
                need_db ? grad_bias.data_ptr<float>() : nullptr);
        check_normalization_cuda_launch("group_norm mixed backward parameters");
    }
    return {grad_input, grad_weight, grad_bias};
}

static std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cuda_impl(
    const Tensor& grad_output, const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps,
    const Tensor* mean_saved, const Tensor* rstd_saved,
    const std::vector<bool>* output_mask) {
    if ((input.dtype() == DType::Float16 ||
         input.dtype() == DType::BFloat16) &&
        grad_output.dtype() == DType::Float32) {
        if (input.dtype() == DType::Float16) {
            return group_norm_backward_mixed_cuda_impl<Half>(
                grad_output, input, num_groups, weight_opt, bias_opt, eps,
                mean_saved, rstd_saved, output_mask);
        }
        return group_norm_backward_mixed_cuda_impl<BFloat16>(
            grad_output, input, num_groups, weight_opt, bias_opt, eps,
            mean_saved, rstd_saved, output_mask);
    }
    if (input.dim() < 2 || input.size(1) <= 0 || num_groups <= 0 ||
        input.size(1) % num_groups != 0) {
        TP_THROW(RuntimeError, "group_norm_backward has invalid group dimensions");
    }
    if (input.dtype() != grad_output.dtype() ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64)) {
        TP_THROW(NotImplementedError,
                 "group_norm_backward supports matching Float32/Float64 or mixed Float16/BFloat16 input with Float32 gradient");
    }
    if (grad_output.dim() != input.dim()) {
        TP_THROW(RuntimeError,
                 "group_norm_backward gradient shape does not match input");
    }
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
        if (grad_output.size(dim) != input.size(dim)) {
            TP_THROW(RuntimeError,
                     "group_norm_backward gradient shape does not match input");
        }
    }
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t cpg = C / num_groups;
    int64_t spatial = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial *= input.size(dim);
    }
    if (spatial <= 0) {
        TP_THROW(RuntimeError, "group_norm_backward requires non-empty spatial dimensions");
    }
    const int64_t inner = cpg * spatial;
    const bool need_dx = !output_mask ||
        (!output_mask->empty() && (*output_mask)[0]);
    const bool need_dw = output_mask
        ? (output_mask->size() > 1 && (*output_mask)[1])
        : (weight_opt.has_value() && weight_opt->defined());
    const bool need_db = output_mask
        ? (output_mask->size() > 2 && (*output_mask)[2])
        : (bias_opt.has_value() && bias_opt->defined());

    Tensor grad_output_contig = grad_output.is_contiguous()
        ? grad_output : grad_output.contiguous();
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_input = need_dx ? Tensor::empty_like(input_contig) : Tensor();
    Tensor grad_weight, grad_bias;
    if (need_dw) {
        grad_weight = Tensor::empty({C}, input.dtype(), input.device());
    }
    if (need_db) {
        grad_bias = Tensor::empty({C}, input.dtype(), input.device());
    }
    std::optional<Tensor> weight_contig = weight_opt;
    if (weight_contig.has_value() && weight_contig->defined() &&
        !weight_contig->is_contiguous()) {
        weight_contig = weight_contig->contiguous();
    }

    const auto stream = getCurrentCUDAStream().stream();
    if (N == 0) {
        const size_t element_bytes = input.dtype() == DType::Float64 ? 8 : 4;
        if (grad_weight.defined()) {
            check_normalization_cuda_status(
                cudaMemsetAsync(grad_weight.data_ptr(), 0,
                                C * element_bytes, stream),
                "group_norm backward weight initialization");
        }
        if (grad_bias.defined()) {
            check_normalization_cuda_status(
                cudaMemsetAsync(grad_bias.data_ptr(), 0,
                                C * element_bytes, stream),
                "group_norm backward bias initialization");
        }
        return {grad_input, grad_weight, grad_bias};
    }

    Tensor mean_saved_contig;
    Tensor rstd_saved_contig;
    if (mean_saved && mean_saved->defined()) {
        mean_saved_contig = mean_saved->is_contiguous()
            ? *mean_saved : mean_saved->contiguous();
    }
    if (rstd_saved && rstd_saved->defined()) {
        rstd_saved_contig = rstd_saved->is_contiguous()
            ? *rstd_saved : rstd_saved->contiguous();
    }
    if (mean_saved_contig.defined() != rstd_saved_contig.defined()) {
        TP_THROW(RuntimeError, "group_norm_backward requires both saved statistics");
    }

    #define GN_BACKWARD_CASE(ctype, acc_t, acc_name)                            \
    {                                                                           \
        constexpr int kVec = sizeof(ctype) <= 4 ? 4 : 2;                        \
        const ctype* dy_p = grad_output_contig.data_ptr<ctype>();               \
        const ctype* x_p = input_contig.data_ptr<ctype>();                     \
        ctype* dx_p = grad_input.defined()                                      \
            ? grad_input.data_ptr<ctype>() : nullptr;                           \
        const bool vec_ok =                                                     \
            inner % kVec == 0 &&                                                \
            reinterpret_cast<uintptr_t>(dy_p) % (sizeof(ctype) * kVec) == 0 &&  \
            reinterpret_cast<uintptr_t>(x_p) % (sizeof(ctype) * kVec) == 0 &&   \
            (!need_dx || reinterpret_cast<uintptr_t>(dx_p) %                   \
                (sizeof(ctype) * kVec) == 0);                                   \
        const unsigned moments_threads = group_norm_threads_for(inner);         \
        const unsigned internal_threads = group_norm_threads_for(spatial);      \
        const unsigned fused_threads = group_norm_threads_for(cpg);            \
        Tensor computed_stats;                                                 \
        acc_t* mean_p = nullptr;                                                \
        acc_t* rstd_p = nullptr;                                                \
        if (mean_saved_contig.defined()) {                                      \
            mean_p = mean_saved_contig.data_ptr<acc_t>();                      \
            rstd_p = rstd_saved_contig.data_ptr<acc_t>();                      \
        } else {                                                                \
            computed_stats = Tensor::empty({N * num_groups * 2},                \
                                            DType::acc_name, input.device());   \
            mean_p = computed_stats.data_ptr<acc_t>();                         \
            rstd_p = mean_p + N * num_groups;                                  \
        }                                                                       \
        Tensor sums = Tensor::empty({N * C * 2}, DType::acc_name,               \
                                    input.device());                            \
        acc_t* ds_p = sums.data_ptr<acc_t>();                                   \
        acc_t* db_p = ds_p + N * C;                                             \
        if (vec_ok) {                                                           \
            if (!mean_saved_contig.defined()) {                                 \
                group_norm_moments_vec_impl<ctype, acc_t, kVec>                 \
                    <<<N * num_groups, moments_threads, 0, stream>>>(           \
                        inner, static_cast<acc_t>(eps), x_p, mean_p, rstd_p);   \
                check_normalization_cuda_launch("group_norm backward moments"); \
            }                                                                   \
            if (spatial % kVec == 0) {                                           \
                group_norm_internal_grads_vec_impl<ctype, acc_t, kVec>          \
                    <<<N * C, internal_threads, 0, stream>>>(                    \
                        spatial, dy_p, x_p, ds_p, db_p);                        \
            } else {                                                             \
                group_norm_internal_grads_tail_impl<ctype, acc_t, kVec>         \
                    <<<N * C, internal_threads, 0, stream>>>(                    \
                        spatial, dy_p, x_p, ds_p, db_p);                        \
            }                                                                    \
            check_normalization_cuda_launch("group_norm backward sums");      \
            if (need_dx) {                                                      \
                const ctype* gw_p = weight_contig.has_value() &&               \
                    weight_contig->defined()                                   \
                    ? weight_contig->data_ptr<ctype>() : nullptr;              \
                if (spatial % kVec == 0) {                                     \
                    group_norm_dx_fused_vec_impl<ctype, acc_t, kVec, true>       \
                        <<<N * num_groups, moments_threads, 0, stream>>>(       \
                            inner, spatial, cpg, num_groups, dy_p, x_p, mean_p,  \
                            rstd_p, gw_p, ds_p, db_p, dx_p);                    \
                } else {                                                       \
                    group_norm_dx_fused_vec_impl<ctype, acc_t, kVec, false>      \
                        <<<N * num_groups, moments_threads, 0, stream>>>(       \
                            inner, spatial, cpg, num_groups, dy_p, x_p, mean_p,  \
                            rstd_p, gw_p, ds_p, db_p, dx_p);                    \
                }                                                              \
                check_normalization_cuda_launch("group_norm backward input"); \
            }                                                                   \
            if (need_dw || need_db) {                                           \
                acc_t* gwv_p = need_dw ? grad_weight.data_ptr<acc_t>()         \
                                       : nullptr;                               \
                acc_t* gbv_p = need_db ? grad_bias.data_ptr<acc_t>()           \
                                       : nullptr;                               \
                group_norm_gamma_beta_sums_impl<acc_t>                          \
                    <<<(C + 31) / 32, dim3(32, 16), 0, stream>>>(                       \
                        N, C, num_groups, cpg, mean_p, rstd_p,                 \
                        ds_p, db_p, gwv_p, gbv_p);                              \
                check_normalization_cuda_launch("group_norm backward parameters"); \
            }                                                                   \
        } else {                                                                \
            if (!mean_saved_contig.defined()) {                                 \
                group_norm_moments_impl<ctype, acc_t>                           \
                    <<<N * num_groups, moments_threads, 0, stream>>>(           \
                        inner, static_cast<acc_t>(eps), x_p, mean_p, rstd_p);   \
                check_normalization_cuda_launch("group_norm backward moments"); \
            }                                                                   \
            if (need_dx || need_dw || need_db) {                                \
                group_norm_internal_grads_impl<ctype, acc_t>                     \
                    <<<N * C, internal_threads, 0, stream>>>(                    \
                        spatial, dy_p, x_p, ds_p, db_p);                         \
                check_normalization_cuda_launch("group_norm backward sums");    \
            }                                                                   \
            if (need_dx) {                                                       \
                Tensor fused = Tensor::empty({N * num_groups * 2},              \
                                              DType::acc_name, input.device()); \
                acc_t* c2_p = fused.data_ptr<acc_t>();                          \
                acc_t* c3_p = c2_p + N * num_groups;                             \
                const ctype* gw_p = weight_contig.has_value() &&               \
                    weight_contig->defined()                                   \
                    ? weight_contig->data_ptr<ctype>() : nullptr;               \
                group_norm_bwd_fused_params_impl<ctype, acc_t>                  \
                    <<<N * num_groups, fused_threads, 0, stream>>>(             \
                        cpg, inner, num_groups, mean_p, rstd_p, gw_p,           \
                        ds_p, db_p, c2_p, c3_p);                                \
                check_normalization_cuda_launch("group_norm backward coefficients"); \
                group_norm_dx_apply_impl<ctype, acc_t>                          \
                    <<<N * num_groups, moments_threads, 0, stream>>>(            \
                        inner, spatial, cpg, num_groups, dy_p, x_p, rstd_p,      \
                        gw_p, c2_p, c3_p, dx_p);                                 \
                check_normalization_cuda_launch("group_norm backward input");   \
            }                                                                   \
            if (need_dw || need_db) {                                           \
                group_norm_gamma_beta_sums_impl<acc_t>                           \
                    <<<(C + 31) / 32, dim3(32, 16), 0, stream>>>(                       \
                        N, C, num_groups, cpg, mean_p, rstd_p,                  \
                        ds_p, db_p,                                             \
                        need_dw ? grad_weight.data_ptr<acc_t>() : nullptr,      \
                        need_db ? grad_bias.data_ptr<acc_t>() : nullptr);       \
                check_normalization_cuda_launch("group_norm backward parameters"); \
            }                                                                   \
        }                                                                       \
    }

    if (input.dtype() == DType::Float32 && grad_output.dtype() == DType::Float32) {
        GN_BACKWARD_CASE(float, float, Float32)
    } else if (input.dtype() == DType::Float64 &&
               grad_output.dtype() == DType::Float64) {
        GN_BACKWARD_CASE(double, double, Float64)
    } else {
        TP_THROW(NotImplementedError,
                 "group_norm_backward supports Float32, Float64, and mixed Float16/BFloat16 on CUDA");
    }
    #undef GN_BACKWARD_CASE

    return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cuda(
    const Tensor& grad_output, const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps) {
    return group_norm_backward_cuda_impl(
        grad_output, input, num_groups, weight_opt, bias_opt, eps, nullptr,
        nullptr, nullptr);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& mean,
    const Tensor& rstd, const std::optional<Tensor>& weight_opt, int64_t N,
    int64_t C, int64_t HxW, int64_t num_groups,
    const std::vector<bool>& output_mask) {
    if (input.dim() < 2 || input.size(0) != N || input.size(1) != C) {
        TP_THROW(RuntimeError,
                 "native_group_norm_backward dimensions do not match input");
    }
    if (output_mask.size() != 3) {
        TP_THROW(RuntimeError,
                 "native_group_norm_backward output mask must have 3 values");
    }
    int64_t actual_hxw = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        actual_hxw *= input.size(dim);
    }
    if (HxW != actual_hxw || mean.numel() != N * num_groups ||
        rstd.numel() != N * num_groups) {
        TP_THROW(RuntimeError,
                 "native_group_norm_backward statistics do not match input");
    }
    return group_norm_backward_cuda_impl(
        grad_output, input, num_groups, weight_opt, std::nullopt, 0.0, &mean,
        &rstd, &output_mask);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, GroupNormKernels) {
    m.impl("group_norm", group_norm_cuda);
    m.impl("group_norm_backward", group_norm_backward_cuda);
    m.impl("native_group_norm", native_group_norm_cuda);
    m.impl("native_group_norm_backward", native_group_norm_backward_cuda);
}

}
}
