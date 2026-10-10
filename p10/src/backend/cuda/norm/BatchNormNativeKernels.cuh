#pragma once

// Device-side building blocks for the batch normalization family: per-channel
// statistics (Welford), the affine transform, the running-statistic fold and
// the two-stage backward (reduce + elementwise).
//
// Every kernel here works on a 3-D view (batch, channel, feature) built by
// merging the trailing dimensions of the operand, so the plane index selects
// the channel and the reduction runs over batch x feature.  Accessors carry
// explicit sizes/strides, which lets the same code serve the dense layouts and
// the strided fallbacks.

#include "Tensor.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace batch_norm {

constexpr int kBnWarpSize = 32;
constexpr int kBnMaxStridedDims = 8;
constexpr int kBnMaxBlockSize = 512;
constexpr unsigned kBnMaxGridSize = 65535u;

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float bn_sqrt(float v) { return ::sqrtf(v); }
__device__ __forceinline__ double bn_sqrt(double v) { return ::sqrt(v); }

__device__ __forceinline__ float bn_warp_shfl_down(float v, int offset) {
    return __shfl_down_sync(0xffffffffffffffffull, v, offset);
}
__device__ __forceinline__ double bn_warp_shfl_down(double v, int offset) {
    return __shfl_down_sync(0xffffffffffffffffull, v, offset);
}
__device__ __forceinline__ float bn_warp_shfl_xor(float v, int offset) {
    return __shfl_xor_sync(0xffffffffffffffffull, v, offset);
}
__device__ __forceinline__ double bn_warp_shfl_xor(double v, int offset) {
    return __shfl_xor_sync(0xffffffffffffffffull, v, offset);
}
__device__ __forceinline__ int bn_warp_shfl_xor(int v, int offset) {
    return __shfl_xor_sync(0xffffffffffffffffull, v, offset);
}

// Index of the most significant set bit.
__device__ __forceinline__ int bn_msb(int val) { return 31 - __clz(val); }

inline int bn_get_num_threads(int64_t n_elem) {
    const int sizes[5] = {32, 64, 128, 256, kBnMaxBlockSize};
    for (int i = 0; i != 5; ++i) {
        if (n_elem <= sizes[i]) return sizes[i];
    }
    return kBnMaxBlockSize;
}

inline int bn_last_pow2(int64_t n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    return static_cast<int>(n - (n >> 1));
}

inline int64_t bn_ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// 32-bit index math is enough while the logical element count fits in int32.
inline bool bn_can_use_32bit_index_math(const Tensor& t) {
    return t.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

// ---------------------------------------------------------------------------
// Strided accessors (1-D / 2-D / 3-D)
// ---------------------------------------------------------------------------

template <typename T, typename index_t>
struct BnAcc1 {
    T* p;
    index_t sz;
    index_t st;
    __host__ __device__ __forceinline__ index_t size(int) const { return sz; }
    __host__ __device__ __forceinline__ T* data() const { return p; }
    __host__ __device__ __forceinline__ T& operator[](index_t i) const { return p[i * st]; }
};

template <typename T, typename index_t>
struct BnAcc2 {
    T* p;
    index_t s0, s1, t0, t1;
    __host__ __device__ __forceinline__ index_t size(int d) const { return d == 0 ? s0 : s1; }
    __host__ __device__ __forceinline__ T* data() const { return p; }
    __host__ __device__ __forceinline__ BnAcc1<T, index_t> operator[](index_t i) const {
        return BnAcc1<T, index_t>{p + i * t0, s1, t1};
    }
};

template <typename T, typename index_t>
struct BnAcc3 {
    T* p;
    index_t s0, s1, s2, t0, t1, t2;
    __host__ __device__ __forceinline__ index_t size(int d) const {
        return d == 0 ? s0 : (d == 1 ? s1 : s2);
    }
    __host__ __device__ __forceinline__ T* data() const { return p; }
    __host__ __device__ __forceinline__ BnAcc2<T, index_t> operator[](index_t i) const {
        return BnAcc2<T, index_t>{p + i * t0, s1, s2, t1, t2};
    }
};

template <typename T, typename index_t>
inline BnAcc1<T, index_t> bn_acc1(const Tensor& t) {
    return BnAcc1<T, index_t>{t.data_ptr<T>(), static_cast<index_t>(t.size(0)),
                              static_cast<index_t>(t.strides()[0])};
}

template <typename T, typename index_t>
inline BnAcc1<T, index_t> bn_acc1(const std::optional<Tensor>& t) {
    if (t.has_value() && t->defined() && t->numel() > 0) {
        return bn_acc1<T, index_t>(*t);
    }
    return BnAcc1<T, index_t>{nullptr, 0, 0};
}

// An undefined operand becomes an empty view: kernels gate every access on
// size(0) > 0, so the null base pointer is never dereferenced.
template <typename T, typename index_t>
inline BnAcc1<T, index_t> bn_acc_or_dummy(const std::optional<Tensor>& t) {
    if (t.has_value() && t->defined() && t->numel() > 0) {
        return BnAcc1<T, index_t>{t->data_ptr<T>(), static_cast<index_t>(t->size(0)),
                                  static_cast<index_t>(t->strides()[0])};
    }
    return BnAcc1<T, index_t>{nullptr, 0, 0};
}

template <typename T, typename index_t>
inline BnAcc2<T, index_t> bn_acc2(const Tensor& t) {
    const std::vector<int64_t> s = t.strides();
    return BnAcc2<T, index_t>{t.data_ptr<T>(), static_cast<index_t>(t.size(0)),
                              static_cast<index_t>(t.size(1)),
                              static_cast<index_t>(s[0]),
                              static_cast<index_t>(s[1])};
}

template <typename T, typename index_t>
inline BnAcc3<T, index_t> bn_acc3(const Tensor& t) {
    const std::vector<int64_t> s = t.strides();
    return BnAcc3<T, index_t>{t.data_ptr<T>(), static_cast<index_t>(t.size(0)),
                              static_cast<index_t>(t.size(1)),
                              static_cast<index_t>(t.size(2)),
                              static_cast<index_t>(s[0]),
                              static_cast<index_t>(s[1]),
                              static_cast<index_t>(s[2])};
}

// ---------------------------------------------------------------------------
// Two-value accumulator and block reduction
// ---------------------------------------------------------------------------

template <typename scalar_t, typename accscalar_t>
struct Float2 {
    accscalar_t v1, v2;
    __device__ Float2() = default;
    __device__ Float2(scalar_t a, scalar_t b)
        : v1(static_cast<accscalar_t>(a)), v2(static_cast<accscalar_t>(b)) {}
    __device__ Float2(int v)
        : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
    __device__ Float2& operator+=(const Float2& a) {
        v1 += a.v1;
        v2 += a.v2;
        return *this;
    }
    __device__ friend Float2 operator+(Float2 a, const Float2& b) {
        a += b;
        return a;
    }
};

template <typename acc_t>
struct SumReduceOp {
    __device__ __forceinline__ acc_t combine(acc_t a, acc_t b) const { return a + b; }
    __device__ __forceinline__ acc_t warp_shfl_down(acc_t data, int offset) const {
        return bn_warp_shfl_down(data, offset);
    }
};

template <typename scalar_t, typename accscalar_t>
struct SumReduceOp<Float2<scalar_t, accscalar_t>> {
    using acc_t = Float2<scalar_t, accscalar_t>;
    __device__ __forceinline__ acc_t combine(acc_t a, acc_t b) const { return a + b; }
    __device__ __forceinline__ acc_t warp_shfl_down(acc_t data, int offset) const {
        acc_t out;
        out.v1 = bn_warp_shfl_down(data.v1, offset);
        out.v2 = bn_warp_shfl_down(data.v2, offset);
        return out;
    }
};

struct BnBlock2D {
    static __device__ __forceinline__ int tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }
    static __device__ __forceinline__ int warps() {
        return blockDim.x * blockDim.y / kBnWarpSize;
    }
};

template <typename T, class ReduceOp>
__device__ __forceinline__ T bn_warp_reduce(T val, const ReduceOp& op) {
#pragma unroll
    for (int offset = (kBnWarpSize >> 1); offset > 0; offset >>= 1) {
        val = op.combine(val, op.warp_shfl_down(val, offset));
    }
    return val;
}

// Sums `val` across the whole block; only thread 0 sees the final value.
template <typename T, class ReduceOp>
__device__ __forceinline__ T bn_block_reduce(T val, const ReduceOp& op, const T& identity,
                                              T* shared) {
    const int tid = BnBlock2D::tid();
    const int lid = tid % kBnWarpSize;
    const int wid = tid / kBnWarpSize;
    val = bn_warp_reduce(val, op);
    __syncthreads();
    if (lid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (tid < BnBlock2D::warps()) ? shared[lid] : identity;
    if (wid == 0) {
        val = bn_warp_reduce(val, op);
    }
    return val;
}

// Produces (sum(dY), sum(dY * (X - mean))) for one plane, broadcast to every
// thread of the block.
template <typename scalar_t, typename Op, typename PTA>
__device__ scalar_t bn_reduce(Op op, PTA tensor, int plane) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
        for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
            sum += op(batch, plane, x);
        }
    }
    __shared__ scalar_t shared[kBnWarpSize];
    SumReduceOp<scalar_t> reduce_op;
    sum = bn_block_reduce<scalar_t, SumReduceOp<scalar_t>>(sum, reduce_op,
                                                          static_cast<scalar_t>(0),
                                                          shared);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared[0] = sum;
    }
    __syncthreads();
    return shared[0];
}

template <typename input_scalar_t, typename stat_accscalar_t, typename PTA>
struct GradOp {
    __device__ GradOp(stat_accscalar_t m, const PTA& i, const PTA& g)
        : mean(m), input(i), grad_output(g) {}
    __device__ __forceinline__ Float2<input_scalar_t, stat_accscalar_t> operator()(
            int batch, int plane, int n) {
        stat_accscalar_t g = grad_output[batch][plane][n];
        stat_accscalar_t c =
            static_cast<stat_accscalar_t>(input[batch][plane][n]) - mean;
        return Float2<input_scalar_t, stat_accscalar_t>(g, g * c);
    }
    const stat_accscalar_t mean;
    const PTA& input;
    const PTA& grad_output;
};

// ---------------------------------------------------------------------------
// Variance post-transforms
// ---------------------------------------------------------------------------

struct BnInvStd {
    template <typename T>
    __device__ __forceinline__ T operator()(T var, double epsilon) const {
        T invstd = 0;
        if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
            invstd = static_cast<T>(1) / bn_sqrt(var + static_cast<T>(epsilon));
        }
        return invstd;
    }
};

struct BnVar {
    template <typename T>
    __device__ __forceinline__ T operator()(T var, double) const {
        return var;
    }
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Per-channel Welford pass over (batch, feature).  The thread-local
// accumulators merge pairwise with the parallel Welford recurrence, first
// inside each warp and then across the warps through shared memory.
template <typename VarTransform, typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void bn_collect_statistics_kernel(
        BnAcc3<const input_scalar_t, index_t> input,
        const stat_accscalar_t epsilon,
        const stat_accscalar_t momentum,
        BnAcc1<stat_accscalar_t, index_t> save_mean,
        BnAcc1<stat_accscalar_t, index_t> save_transformed_var) {
    __shared__ int shared_n[2 * 2 * kBnWarpSize + kBnWarpSize];
    stat_accscalar_t* shared_avg_var =
        reinterpret_cast<stat_accscalar_t*>(&shared_n[kBnWarpSize]);

    const int plane = blockIdx.x;
    const int n_total = input.size(0) * input.size(2);
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    stat_accscalar_t avg = 0;
    stat_accscalar_t var_n = 0;
    int n = 0;
    for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
        for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
            stat_accscalar_t v = input[batch][plane][x];
            stat_accscalar_t d1 = v - avg;
            n++;
            avg += d1 / n;
            var_n += d1 * (v - avg);
        }
    }

    for (int i = 0; i < bn_msb(kBnWarpSize); ++i) {
        stat_accscalar_t o_avg = bn_warp_shfl_xor(avg, 1 << i);
        int o_n = bn_warp_shfl_xor(n, 1 << i);
        stat_accscalar_t factor =
            static_cast<stat_accscalar_t>(1.0) / fmaxf(1.0f, n + o_n);
        var_n += bn_warp_shfl_xor(var_n, 1 << i) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    __syncthreads();
    if (tid % kBnWarpSize == 0) {
        shared_n[tid / kBnWarpSize] = n;
        shared_avg_var[tid / kBnWarpSize * 2] = avg;
        shared_avg_var[tid / kBnWarpSize * 2 + 1] = var_n;
    }
    __syncthreads();

    if (tid < kBnWarpSize) {
        const bool live = tid < blockDim.x * blockDim.y / kBnWarpSize;
        n = live ? shared_n[tid] : 0;
        avg = live ? shared_avg_var[2 * tid] : static_cast<stat_accscalar_t>(0);
        var_n = live ? shared_avg_var[2 * tid + 1] : static_cast<stat_accscalar_t>(0);
    }
    for (int i = 0; i < bn_msb(kBnWarpSize); ++i) {
        stat_accscalar_t o_avg = bn_warp_shfl_xor(avg, 1 << i);
        int o_n = bn_warp_shfl_xor(n, 1 << i);
        stat_accscalar_t factor =
            static_cast<stat_accscalar_t>(1.0) / fmaxf(1.0f, n + o_n);
        var_n += bn_warp_shfl_xor(var_n, 1 << i) +
                 (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }

    if (tid == 0) {
        if (save_mean.size(0) > 0) {
            save_mean[plane] = avg;
        }
        if (save_transformed_var.size(0) > 0) {
            save_transformed_var[plane] =
                VarTransform{}(var_n / n_total, epsilon);
        }
    }
}

// out = gamma * (x - mean) * invstd + beta, one block column per channel.
template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, bool train, typename index_t>
__global__ void bn_transform_input_kernel(
        BnAcc3<const input_scalar_t, index_t> input,
        BnAcc3<input_scalar_t, index_t> output,
        BnAcc1<typename std::conditional<train, stat_accscalar_t,
                                        stat_scalar_t>::type,
               index_t> mean_,
        BnAcc1<typename std::conditional<train, stat_accscalar_t,
                                        stat_scalar_t>::type,
               index_t> var_or_invstd,
        BnAcc1<stat_scalar_t, index_t> weight,
        BnAcc1<stat_scalar_t, index_t> bias,
        stat_accscalar_t epsilon) {
    const index_t plane = blockIdx.x;
    if (plane >= input.size(1)) {
        return;
    }

    stat_accscalar_t gamma = weight.size(0) > 0
        ? static_cast<stat_accscalar_t>(weight[plane])
        : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta = bias.size(0) > 0
        ? static_cast<stat_accscalar_t>(bias[plane])
        : static_cast<stat_accscalar_t>(0);
    stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
    stat_accscalar_t invstd;
    if (train) {
        invstd = var_or_invstd[plane];
    } else {
        invstd = static_cast<stat_accscalar_t>(1) /
                 bn_sqrt(static_cast<stat_accscalar_t>(var_or_invstd[plane]) +
                         epsilon);
    }

    const index_t bs = input.size(0);
    const index_t fs = input.size(2);
    const index_t bstep = blockDim.y * gridDim.y;
    for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs;
         batch += bstep) {
        auto o = output[batch][plane];
        auto i = input[batch][plane];
        for (index_t feature = threadIdx.x; feature < fs;
             feature += blockDim.x) {
            o[feature] = static_cast<input_scalar_t>(
                gamma * (i[feature] - mean) * invstd + beta);
        }
    }
}

// Folds a stack of per-rank (mean, invstd) pairs plus their element counts into
// the global mean/invstd and folds the result into the running buffers.
template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void bn_reduce_statistics_kernel(
        BnAcc2<accscalar_t, index_t> vec_mean,
        BnAcc2<accscalar_t, index_t> vec_invstd,
        BnAcc1<accscalar_t, index_t> mean,
        BnAcc1<accscalar_t, index_t> invstd,
        BnAcc1<scalar_t, index_t> running_mean,
        BnAcc1<scalar_t, index_t> running_var,
        const accscalar_t epsilon,
        const accscalar_t momentum,
        BnAcc1<scalar_t, index_t> counts) {
    const int feature_size = vec_mean.size(1);
    const int world_size = vec_mean.size(0);
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    for (int i = bid * blockDim.x + tid; i < feature_size;
         i += gridDim.x * blockDim.x) {
        accscalar_t avg = 0;
        accscalar_t var_n = 0;
        index_t n = 0;
        for (int j = 0; j < world_size; j++) {
            const accscalar_t count = static_cast<accscalar_t>(counts[j]);
            const accscalar_t n_acc = static_cast<accscalar_t>(n);
            accscalar_t m = vec_mean[j][i];
            accscalar_t v = accscalar_t(1.0) / (vec_invstd[j][i]);
            v = (v * v - epsilon) * count;
            accscalar_t factor = accscalar_t(1.0) / (n_acc + count);
            var_n += v + (avg - m) * (avg - m) * n_acc * count * factor;
            avg = n_acc * factor * avg + count * factor * m;
            n = static_cast<index_t>(n + static_cast<index_t>(count));
        }
        mean[i] = avg;
        invstd[i] = static_cast<accscalar_t>(1) /
                    bn_sqrt(var_n / n + epsilon);
        if (running_mean.size(0) > 0) {
            running_mean[i] = static_cast<scalar_t>(
                (1 - momentum) * running_mean[i] + momentum * avg);
        }
        accscalar_t unbiased_var = var_n / (n - 1);
        if (running_var.size(0) > 0) {
            running_var[i] = static_cast<scalar_t>(
                (1 - momentum) * running_var[i] + momentum * unbiased_var);
        }
    }
}

// reduce stage of the backward: per-channel (sum dY, sum dY * (X - mean)) plus
// the two parameter gradients.
template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void bn_backward_reduce_kernel(
        BnAcc3<input_scalar_t, index_t> input,
        BnAcc3<input_scalar_t, index_t> grad_output,
        BnAcc1<stat_accscalar_t, index_t> mean,
        BnAcc1<stat_accscalar_t, index_t> invstd,
        BnAcc1<stat_accscalar_t, index_t> sum_dy,
        BnAcc1<stat_accscalar_t, index_t> sum_dy_xmu,
        BnAcc1<stat_scalar_t, index_t> grad_weight,
        BnAcc1<stat_scalar_t, index_t> grad_bias) {
    const index_t plane = blockIdx.x;
    const stat_accscalar_t r_mean = mean[plane];
    const stat_accscalar_t factor = invstd[plane];

    GradOp<input_scalar_t, stat_accscalar_t, BnAcc3<input_scalar_t, index_t>> g(
        r_mean, input, grad_output);
    auto res = bn_reduce<Float2<input_scalar_t, stat_accscalar_t>>(g, grad_output,
                                                                 plane);
    if (threadIdx.x == 0) {
        if (grad_weight.size(0) > 0) {
            grad_weight[plane] = static_cast<stat_scalar_t>(res.v2 * factor);
        }
        if (grad_bias.size(0) > 0) {
            grad_bias[plane] = static_cast<stat_scalar_t>(res.v1);
        }
        if (sum_dy.size(0) > 0) {
            sum_dy[plane] = static_cast<stat_accscalar_t>(res.v1);
        }
        if (sum_dy_xmu.size(0) > 0) {
            sum_dy_xmu[plane] = static_cast<stat_accscalar_t>(res.v2);
        }
    }
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__device__ __forceinline__ void bn_backward_elemt_kernel_impl(
        BnAcc3<input_scalar_t, index_t> input,
        BnAcc3<input_scalar_t, index_t> grad_output,
        BnAcc1<stat_accscalar_t, index_t> mean,
        BnAcc1<stat_accscalar_t, index_t> invstd,
        BnAcc1<stat_scalar_t, index_t> weight,
        BnAcc1<stat_accscalar_t, index_t> sum_dy,
        BnAcc1<stat_accscalar_t, index_t> sum_dy_xmu,
        BnAcc3<input_scalar_t, index_t> grad_input,
        const stat_accscalar_t norm_fct) {
    const index_t plane = blockIdx.x;
    if (plane >= input.size(1)) {
        return;
    }

    stat_accscalar_t m_c = mean[plane];
    stat_accscalar_t m_dy_c = sum_dy[plane] * norm_fct;
    stat_accscalar_t factor_1_c = invstd[plane];
    stat_accscalar_t factor_2_c =
        weight.size(0) > 0 ? static_cast<stat_accscalar_t>(weight[plane])
                           : static_cast<stat_accscalar_t>(1);
    factor_2_c *= factor_1_c;
    factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[plane] * norm_fct;

    const index_t bs = input.size(0);
    const index_t fs = input.size(2);
    const index_t bstep = blockDim.y * gridDim.y;
    for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs;
         batch += bstep) {
        auto g_i = grad_input[batch][plane];
        auto g_o = grad_output[batch][plane];
        auto i = input[batch][plane];
        for (index_t feature = threadIdx.x; feature < fs;
             feature += blockDim.x) {
            g_i[feature] = static_cast<input_scalar_t>(
                (g_o[feature] - m_dy_c - (i[feature] - m_c) * factor_1_c) *
                factor_2_c);
        }
    }
}

// The normalization factor is derived from the element tally carried by the
// count operand, so a reduction split across ranks keeps its own scale.
template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void bn_backward_elemt_kernel(
        BnAcc3<input_scalar_t, index_t> input,
        BnAcc3<input_scalar_t, index_t> grad_output,
        BnAcc1<stat_accscalar_t, index_t> mean,
        BnAcc1<stat_accscalar_t, index_t> invstd,
        BnAcc1<stat_scalar_t, index_t> weight,
        BnAcc1<stat_accscalar_t, index_t> sum_dy,
        BnAcc1<stat_accscalar_t, index_t> sum_dy_xmu,
        BnAcc3<input_scalar_t, index_t> grad_input,
        const int* __restrict__ numel, const int world_size) {
    int64_t total_numel = 0;
    for (int i = 0; i < world_size; i++) {
        total_numel += numel[i];
    }
    const stat_accscalar_t norm_fct =
        static_cast<stat_accscalar_t>(1) /
        static_cast<stat_accscalar_t>(total_numel);
    bn_backward_elemt_kernel_impl(input, grad_output, mean, invstd, weight,
                                  sum_dy, sum_dy_xmu, grad_input, norm_fct);
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void bn_backward_elemt_kernel(
        BnAcc3<input_scalar_t, index_t> input,
        BnAcc3<input_scalar_t, index_t> grad_output,
        BnAcc1<stat_accscalar_t, index_t> mean,
        BnAcc1<stat_accscalar_t, index_t> invstd,
        BnAcc1<stat_scalar_t, index_t> weight,
        BnAcc1<stat_accscalar_t, index_t> sum_dy,
        BnAcc1<stat_accscalar_t, index_t> sum_dy_xmu,
        BnAcc3<input_scalar_t, index_t> grad_input,
        const stat_accscalar_t norm_fct) {
    bn_backward_elemt_kernel_impl(input, grad_output, mean, invstd, weight,
                                  sum_dy, sum_dy_xmu, grad_input, norm_fct);
}

// Folds the batch statistics into the running buffers:
//   running_mean = m * momentum + (1 - momentum) * running_mean
//   running_var  = unbiased_var * momentum + (1 - momentum) * running_var
template <typename input_scalar_t, typename scalar_t, typename accscalar_t>
__global__ void bn_update_stats_kernel(
        int64_t n,
        const accscalar_t* __restrict__ save_mean,
        const accscalar_t* __restrict__ save_var,
        scalar_t* __restrict__ running_mean,
        scalar_t* __restrict__ running_var,
        const accscalar_t bessel,
        const accscalar_t momentum) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    accscalar_t mean = save_mean[idx];
    accscalar_t var = save_var[idx];
    accscalar_t unbiased_var = var * bessel;
    if (running_mean != nullptr) {
        running_mean[idx] = static_cast<scalar_t>(
            mean * momentum + (1 - momentum) *
                                 static_cast<accscalar_t>(running_mean[idx]));
    }
    if (running_var != nullptr) {
        running_var[idx] = static_cast<scalar_t>(
            unbiased_var * momentum +
            (1 - momentum) * static_cast<accscalar_t>(running_var[idx]));
    }
}

// ---------------------------------------------------------------------------
// Channels-last kernels
//
// The channel axis is the fastest-moving one, so each thread owns one channel
// and walks the reduction axis with a strided loop.  When the reduction is
// spread over several blocks in that axis, every block stages its partial
// (mean, m2, count) triple and the last block to finish a channel column folds
// them together, which keeps a single kernel launch.
// ---------------------------------------------------------------------------

constexpr int kBnElementsPerIter = 4;
constexpr int kBnOptimalTileW = 32;
constexpr int kBnMaxHBlock = 128;

template <typename T, typename C>
__device__ __forceinline__ void bn_welford_merge_element(C& count, T& mean,
                                                         T& m2n,
                                                         const C& count_new,
                                                         const T& mean_new,
                                                         const T& m2n_new) {
    T factor = T(1.0) / ::max(1, (count + count_new));
    T delta0 = mean - mean_new;
    mean = (mean_new * count_new + mean * count) * factor;
    m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
    count += count_new;
}

template <typename T, typename C>
__device__ __forceinline__ void bn_welford_merge_block_vertical(
        C& count, T& mean, T& m2n, C* shmem_count, T* shmem_mean,
        T* shmem_m2n) {
    auto address_base = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y < offset * 2) {
            shmem_mean[address_base] = mean;
            shmem_m2n[address_base] = m2n;
            shmem_count[address_base] = count;
        }
        __syncthreads();
        if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
            auto address = address_base + offset * blockDim.x;
            auto count_new = shmem_count[address];
            auto mean_new = shmem_mean[address];
            auto m2n_new = shmem_m2n[address];
            bn_welford_merge_element(count, mean, m2n, count_new, mean_new,
                                     m2n_new);
        }
    }
}

template <typename T>
__device__ __forceinline__ void bn_merge_block_vertical_backward(
        T& sum_dy, T& sum_dy_xmu, T* shmem_sum_dy, T* shmem_sum_dy_xmu) {
    auto address_base = threadIdx.x + threadIdx.y * blockDim.x;
#pragma unroll
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y < offset * 2) {
            shmem_sum_dy[address_base] = sum_dy;
            shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
        }
        __syncthreads();
        if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
            auto address = address_base + offset * blockDim.x;
            sum_dy += shmem_sum_dy[address];
            sum_dy_xmu += shmem_sum_dy_xmu[address];
        }
    }
}

template <typename VarTransform, typename scalar_t, typename accscalar_t,
          int parallel_loads>
__global__ void bn_collect_statistics_channels_last_kernel(
        const scalar_t* __restrict__ input, accscalar_t* __restrict__ out_mean,
        accscalar_t* __restrict__ out_invstd,
        volatile accscalar_t* staging_data, int* semaphores,
        const int reduction_size, const int stride, accscalar_t epsilon) {
    accscalar_t x_mean[parallel_loads];
    accscalar_t m_2_n[parallel_loads];
    int count[parallel_loads];

#pragma unroll
    for (int i = 0; i < parallel_loads; i++) {
        x_mean[i] = accscalar_t(0);
        m_2_n[i] = accscalar_t(0);
        count[i] = 0;
    }

    const int inner_loop_stride = blockDim.y * gridDim.y;
    int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

    const int loop_count =
        1 + (reduction_size - 1) / (inner_loop_stride * parallel_loads);
    int address_base = m_offset * stride + c_offset;
    const int address_increment = inner_loop_stride * stride;

    for (int i = 0; i < loop_count; i++) {
        accscalar_t x_math[parallel_loads];
        accscalar_t x_count_inv[parallel_loads];
        accscalar_t is_valid[parallel_loads];

#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            if (c_offset < stride && m_offset < reduction_size) {
                x_math[j] = input[address_base];
                count[j]++;
                x_count_inv[j] = accscalar_t(1) / count[j];
                is_valid[j] = accscalar_t(1);
            } else {
                x_math[j] = accscalar_t(0);
                x_count_inv[j] = accscalar_t(0);
                is_valid[j] = accscalar_t(0);
            }
            m_offset += inner_loop_stride;
            address_base += address_increment;
        }

#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            accscalar_t delta0 = x_math[j] - x_mean[j];
            x_mean[j] += delta0 * x_count_inv[j];
            accscalar_t delta1 = x_math[j] - x_mean[j];
            m_2_n[j] += delta0 * delta1 * is_valid[j];
        }
    }

#pragma unroll
    for (int j = 1; j < parallel_loads; j++) {
        bn_welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j],
                                 x_mean[j], m_2_n[j]);
    }

    auto mean_th = x_mean[0];
    auto m2_th = m_2_n[0];
    auto count_th = count[0];

    __shared__ accscalar_t shmem_mean[kBnMaxBlockSize];
    __shared__ accscalar_t shmem_m2n[kBnMaxBlockSize];
    __shared__ int shmem_count[kBnMaxBlockSize];

    bn_welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count,
                                    shmem_mean, shmem_m2n);

    if (gridDim.y > 1) {
        volatile accscalar_t* staging_mean = staging_data;
        volatile accscalar_t* staging_m2n = &staging_data[stride * gridDim.y];
        volatile int* staging_count =
            reinterpret_cast<volatile int*>(&staging_m2n[stride * gridDim.y]);

        address_base = c_offset + blockIdx.y * stride;
        if (threadIdx.y == 0 && c_offset < stride) {
            staging_mean[address_base] = mean_th;
            staging_m2n[address_base] = m2_th;
            staging_count[address_base] = count_th;
        }

        __threadfence();
        __syncthreads();

        __shared__ bool is_last_block_done;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int old = atomicAdd(&semaphores[blockIdx.x], 1);
            is_last_block_done = (old == (gridDim.y - 1));
        }
        __syncthreads();

        if (is_last_block_done) {
            count_th = 0;
            mean_th = accscalar_t(0.0);
            m2_th = accscalar_t(0.0);

            for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
                address_base = c_offset + y * stride;
                int count_new = c_offset < stride ? staging_count[address_base] : 0;
                accscalar_t mean_new =
                    c_offset < stride ? staging_mean[address_base]
                                      : accscalar_t(0.0);
                accscalar_t m2n_new =
                    c_offset < stride ? staging_m2n[address_base]
                                      : accscalar_t(0.0);
                bn_welford_merge_element(count_th, mean_th, m2_th, count_new,
                                         mean_new, m2n_new);
            }

            bn_welford_merge_block_vertical(count_th, mean_th, m2_th,
                                            shmem_count, shmem_mean, shmem_m2n);
            if (threadIdx.y == 0 && c_offset < stride) {
                out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
                out_invstd[c_offset] =
                    VarTransform{}(m2_th / count_th, epsilon);
            }
        }
    } else {
        if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
            out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
            out_invstd[c_offset] = VarTransform{}(m2_th / count_th, epsilon);
        }
    }
}

template <typename scalar_t, typename accscalar_t, typename layerscalar_t,
          int parallel_loads>
__global__ void bn_transform_input_channels_last_kernel(
        const scalar_t* __restrict__ input, const accscalar_t* __restrict__ mean,
        const accscalar_t* __restrict__ inv_std,
        const layerscalar_t* __restrict__ weight,
        const layerscalar_t* __restrict__ shift, scalar_t* __restrict__ out,
        const int reduction_size, const int stride) {
    const int inner_loop_stride = blockDim.y * gridDim.y;
    int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_offset >= stride || m_offset >= reduction_size) {
        return;
    }

    auto m_c = mean[c_offset];
    auto inv_std_c = static_cast<accscalar_t>(inv_std[c_offset]);
    auto w_c = weight == nullptr ? accscalar_t(1.0)
                                 : static_cast<accscalar_t>(weight[c_offset]);
    auto s_c = shift == nullptr ? accscalar_t(0.0)
                                : static_cast<accscalar_t>(shift[c_offset]);

    const int loop_count =
        1 + (reduction_size - 1) / (inner_loop_stride * parallel_loads);
    int address_base = m_offset * stride + c_offset;
    const int address_increment = inner_loop_stride * stride;

    for (int i = 0; i < loop_count; i++) {
#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            if (c_offset < stride && m_offset < reduction_size) {
                auto tmp = w_c *
                               (static_cast<accscalar_t>(input[address_base]) - m_c) *
                               inv_std_c +
                           s_c;
                out[address_base] = static_cast<scalar_t>(tmp);
            }
            m_offset += inner_loop_stride;
            address_base += address_increment;
        }
    }
}

template <int parallel_loads, typename scalar_t, typename accscalar_t,
          typename layerscalar_t>
__global__ void bn_backward_reduce_channels_last_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ grad_output,
        const accscalar_t* __restrict__ mean,
        const accscalar_t* __restrict__ inv_std,
        accscalar_t* __restrict__ sum_dy_o,
        accscalar_t* __restrict__ sum_dy_xmu_o,
        layerscalar_t* __restrict__ grad_weight,
        layerscalar_t* __restrict__ grad_bias,
        volatile accscalar_t* staging_data, int* semaphores,
        const int reduction_size, const int stride) {
    accscalar_t sum_dy[parallel_loads];
    accscalar_t sum_dy_xmu[parallel_loads];

#pragma unroll
    for (int i = 0; i < parallel_loads; i++) {
        sum_dy[i] = accscalar_t(0);
        sum_dy_xmu[i] = accscalar_t(0);
    }

    const int inner_loop_stride = blockDim.y * gridDim.y;
    int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_offset >= stride || m_offset >= reduction_size) {
        return;
    }

    const int loop_count =
        1 + (reduction_size - 1) / (inner_loop_stride * parallel_loads);
    int address_base = m_offset * stride + c_offset;
    const int address_increment = inner_loop_stride * stride;

    auto r_mean = mean[c_offset];
    auto factor = inv_std[c_offset];

    for (int i = 0; i < loop_count; i++) {
        accscalar_t x_input[parallel_loads];
        accscalar_t x_grad_output[parallel_loads];

#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            if (c_offset < stride && m_offset < reduction_size) {
                x_input[j] = input[address_base];
                x_grad_output[j] = grad_output[address_base];
            } else {
                x_input[j] = accscalar_t(0);
                x_grad_output[j] = accscalar_t(0);
            }
            m_offset += inner_loop_stride;
            address_base += address_increment;
        }

#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            sum_dy[j] += x_grad_output[j];
            sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
        }
    }

#pragma unroll
    for (int j = 1; j < parallel_loads; j++) {
        sum_dy[0] += sum_dy[j];
        sum_dy_xmu[0] += sum_dy_xmu[j];
    }

    auto sum_dy_th = sum_dy[0];
    auto sum_dy_xmu_th = sum_dy_xmu[0];

    __shared__ accscalar_t shmem_sum_dy[kBnMaxBlockSize];
    __shared__ accscalar_t shmem_sum_dy_xmu[kBnMaxBlockSize];

    bn_merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy,
                                     shmem_sum_dy_xmu);

    if (gridDim.y > 1) {
        volatile accscalar_t* staging_sum_dy = staging_data;
        volatile accscalar_t* staging_sum_dy_xmu =
            &staging_data[stride * gridDim.y];

        address_base = c_offset + blockIdx.y * stride;
        if (threadIdx.y == 0 && c_offset < stride) {
            staging_sum_dy[address_base] = sum_dy_th;
            staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
        }

        __threadfence();
        __syncthreads();

        __shared__ bool is_last_block_done;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int old = atomicAdd(&semaphores[blockIdx.x], 1);
            is_last_block_done = (old == (gridDim.y - 1));
        }
        __syncthreads();

        if (is_last_block_done) {
            sum_dy_th = accscalar_t(0.0);
            sum_dy_xmu_th = accscalar_t(0.0);

            for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
                address_base = c_offset + y * stride;
                sum_dy_th += (c_offset < stride ? staging_sum_dy[address_base]
                                                : accscalar_t(0.0));
                sum_dy_xmu_th +=
                    (c_offset < stride ? staging_sum_dy_xmu[address_base]
                                       : accscalar_t(0.0));
            }

            bn_merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th,
                                             shmem_sum_dy, shmem_sum_dy_xmu);
            if (threadIdx.y == 0 && c_offset < stride) {
                if (grad_bias != nullptr) {
                    grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
                }
                if (grad_weight != nullptr) {
                    grad_weight[c_offset] =
                        static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
                }
                sum_dy_o[c_offset] = sum_dy_th;
                sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
            }
        }
    } else {
        if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
            if (grad_bias != nullptr) {
                grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
            }
            if (grad_weight != nullptr) {
                grad_weight[c_offset] =
                    static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
            }
            sum_dy_o[c_offset] = sum_dy_th;
            sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
        }
    }
}

template <int parallel_loads, typename scalar_t, typename accscalar_t,
          typename layerscalar_t>
__device__ __forceinline__ void bn_backward_elemt_channels_last_kernel_impl(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const accscalar_t* __restrict__ mean,
        const accscalar_t* __restrict__ inv_std,
        const layerscalar_t* __restrict__ weight,
        const accscalar_t* __restrict__ sum_dy,
        const accscalar_t* __restrict__ sum_dy_xmu,
        scalar_t* __restrict__ grad_input, const accscalar_t norm_fct,
        const int reduction_size, const int stride) {
    const int inner_loop_stride = blockDim.y * gridDim.y;
    int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_offset >= stride || m_offset >= reduction_size) {
        return;
    }

    auto m_c = mean[c_offset];
    auto m_dy_c = sum_dy[c_offset] * norm_fct;
    auto factor_1_c = inv_std[c_offset];
    auto factor_2_c = (weight == nullptr ? accscalar_t(1.0)
                                         : static_cast<accscalar_t>(weight[c_offset])) *
                      factor_1_c;
    factor_1_c = factor_1_c * factor_1_c * sum_dy_xmu[c_offset] * norm_fct;

    const int loop_count =
        1 + (reduction_size - 1) / (inner_loop_stride * parallel_loads);
    int address_base = m_offset * stride + c_offset;
    const int address_increment = inner_loop_stride * stride;

    for (int i = 0; i < loop_count; i++) {
#pragma unroll
        for (int j = 0; j < parallel_loads; j++) {
            if (c_offset < stride && m_offset < reduction_size) {
                grad_input[address_base] = static_cast<scalar_t>(
                    (static_cast<accscalar_t>(grad_output[address_base]) -
                     m_dy_c - (static_cast<accscalar_t>(input[address_base]) -
                               m_c) *
                                  factor_1_c) *
                    factor_2_c);
            }
            m_offset += inner_loop_stride;
            address_base += address_increment;
        }
    }
}

template <int parallel_loads, typename scalar_t, typename accscalar_t,
          typename layerscalar_t>
__global__ void bn_backward_elemt_channels_last_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const accscalar_t* __restrict__ mean,
        const accscalar_t* __restrict__ inv_std,
        const layerscalar_t* __restrict__ weight,
        const accscalar_t* __restrict__ sum_dy,
        const accscalar_t* __restrict__ sum_dy_xmu,
        const int* __restrict__ numel, scalar_t* __restrict__ grad_input,
        const int64_t world_size, const int reduction_size, const int stride) {
    int64_t total_numel = 0;
    for (int i = 0; i < world_size; i++) {
        total_numel += numel[i];
    }
    const accscalar_t norm_fct =
        static_cast<accscalar_t>(1) / static_cast<accscalar_t>(total_numel);
    bn_backward_elemt_channels_last_kernel_impl<parallel_loads>(
        grad_output, input, mean, inv_std, weight, sum_dy, sum_dy_xmu,
        grad_input, norm_fct, reduction_size, stride);
}

template <int parallel_loads, typename scalar_t, typename accscalar_t,
          typename layerscalar_t>
__global__ void bn_backward_elemt_channels_last_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const accscalar_t* __restrict__ mean,
        const accscalar_t* __restrict__ inv_std,
        const layerscalar_t* __restrict__ weight,
        const accscalar_t* __restrict__ sum_dy,
        const accscalar_t* __restrict__ sum_dy_xmu,
        scalar_t* __restrict__ grad_input, const accscalar_t norm_fct,
        const int reduction_size, const int stride) {
    bn_backward_elemt_channels_last_kernel_impl<parallel_loads>(
        grad_output, input, mean, inv_std, weight, sum_dy, sum_dy_xmu,
        grad_input, norm_fct, reduction_size, stride);
}

// ---------------------------------------------------------------------------
// Strided fallbacks
// ---------------------------------------------------------------------------

// Fully strided affine transform: every operand is addressed through its own
// strides, the per-channel parameters broadcast along the channel axis.
struct BnStridedLayout {
    int ndim = 0;
    int64_t numel = 0;
    int64_t sizes[kBnMaxStridedDims] = {};
    int64_t in_stride[kBnMaxStridedDims] = {};
    int64_t out_stride[kBnMaxStridedDims] = {};
    int64_t channel_div = 1;  // row-major product of dims 2..n-1
    int64_t channel_size = 1;
};

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
__global__ void bn_elementwise_strided_kernel(
        const BnStridedLayout layout,
        const input_scalar_t* __restrict__ input,
        input_scalar_t* __restrict__ output,
        const stat_scalar_t* __restrict__ weight,
        const stat_scalar_t* __restrict__ bias,
        const stat_accscalar_t* __restrict__ mean,
        const stat_accscalar_t* __restrict__ invstd) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= layout.numel) return;
    const int64_t plane = (idx / layout.channel_div) % layout.channel_size;

    int64_t in_off = 0;
    int64_t out_off = 0;
    int64_t rest = idx;
    for (int d = layout.ndim - 1; d >= 0; --d) {
        const int64_t coord = rest % layout.sizes[d];
        rest /= layout.sizes[d];
        in_off += coord * layout.in_stride[d];
        out_off += coord * layout.out_stride[d];
    }

    stat_accscalar_t gamma =
        weight != nullptr ? static_cast<stat_accscalar_t>(weight[plane])
                          : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta =
        bias != nullptr ? static_cast<stat_accscalar_t>(bias[plane])
                        : static_cast<stat_accscalar_t>(0);
    output[out_off] = static_cast<input_scalar_t>(
        gamma * (static_cast<stat_accscalar_t>(input[in_off]) -
                 static_cast<stat_accscalar_t>(mean[plane])) *
            static_cast<stat_accscalar_t>(invstd[plane]) +
        beta);
}

// Fully strided backward transform:
//   dx = (dY - sum_dy * norm - (X - mean) * invstd^2 * sum_dy_xmu * norm)
//        * (weight * invstd)
template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t>
__global__ void bn_backward_elemt_strided_kernel(
        const BnStridedLayout layout,
        const input_scalar_t* __restrict__ grad_out,
        const input_scalar_t* __restrict__ input,
        input_scalar_t* __restrict__ grad_input,
        const stat_scalar_t* __restrict__ weight,
        const stat_accscalar_t* __restrict__ mean,
        const stat_accscalar_t* __restrict__ invstd,
        const stat_accscalar_t* __restrict__ sum_dy,
        const stat_accscalar_t* __restrict__ sum_dy_xmu,
        const int* __restrict__ numel, const int world_size) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= layout.numel) return;
    const int64_t plane = (idx / layout.channel_div) % layout.channel_size;

    int64_t total_numel = 0;
    for (int i = 0; i < world_size; i++) {
        total_numel += numel[i];
    }
    const stat_accscalar_t norm_fct =
        static_cast<stat_accscalar_t>(1) /
        static_cast<stat_accscalar_t>(total_numel);

    int64_t in_off = 0;
    int64_t go_off = 0;
    int64_t gi_off = 0;
    int64_t rest = idx;
    for (int d = layout.ndim - 1; d >= 0; --d) {
        const int64_t coord = rest % layout.sizes[d];
        rest /= layout.sizes[d];
        in_off += coord * layout.in_stride[d];
        go_off += coord * layout.out_stride[d];
        gi_off += coord * layout.out_stride[d];
    }

    const stat_accscalar_t m_c = mean[plane];
    const stat_accscalar_t m_dy_c = sum_dy[plane] * norm_fct;
    const stat_accscalar_t invstd_c = invstd[plane];
    stat_accscalar_t factor_2_c =
        weight != nullptr ? static_cast<stat_accscalar_t>(weight[plane])
                          : static_cast<stat_accscalar_t>(1);
    factor_2_c *= invstd_c;
    const stat_accscalar_t factor_1_c =
        invstd_c * invstd_c * sum_dy_xmu[plane] * norm_fct;

    grad_input[gi_off] = static_cast<input_scalar_t>(
        (static_cast<stat_accscalar_t>(grad_out[go_off]) - m_dy_c -
         (static_cast<stat_accscalar_t>(input[in_off]) - m_c) * factor_1_c) *
        factor_2_c);
}

}
}
}
