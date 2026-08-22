#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

namespace {

bool defined(const std::optional<Tensor>& value) {
    return value.has_value() && value->defined();
}

// cuDNN's spatial BN descriptor is 1xCx1x1.  A plain [C] descriptor would
// put C in the innermost dimension and is not the channel parameter layout.
cudnnTensorDescriptor_t derive_bn_descriptor(cudnnTensorDescriptor_t x_desc) {
    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc, x_desc, CUDNN_BATCHNORM_SPATIAL));
    return bn_desc;
}

Tensor batch_norm_view(const Tensor& input) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
    while (shape.size() < 4) shape.push_back(1);
    if (shape.size() > 5) {
        TP_THROW(RuntimeError, "batch_norm CUDA supports 2D through 5D inputs");
    }
    if (shape.size() == static_cast<size_t>(input.dim())) return input;
    return input.reshape(shape);
}

void check_batch_norm_input(const Tensor& input) {
    if (input.dim() < 2 || input.dim() > 5) {
        TP_THROW(RuntimeError, "batch_norm CUDA supports 2D through 5D inputs");
    }
    if (input.dtype() != DType::Float32) {
        TP_THROW(NotImplementedError, "batch_norm CUDA currently supports Float32 only");
    }
}

__global__ void inverse_variance_kernel(
    int64_t channels, const float* variance, float* inverse_variance, float eps) {
    int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel < channels) {
        inverse_variance[channel] = 1.0f / sqrtf(variance[channel] + eps);
    }
}

void check_cuda_launch(const char* name) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(name) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

Tensor batch_norm_cuda(
    const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> bias_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    bool training,
    double momentum,
    double eps) {
    check_batch_norm_input(input);

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor input_bn = batch_norm_view(input_contig);
    Tensor output = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
    Tensor output_bn = batch_norm_view(output);

    int64_t channels = input.size(1);
    Tensor scale = defined(weight_opt)
        ? *weight_opt
        : Tensor::ones({channels}, DType::Float32, input.device());
    Tensor bias = defined(bias_opt)
        ? *bias_opt
        : Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor running_mean = defined(running_mean_opt)
        ? *running_mean_opt
        : Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor running_var = defined(running_var_opt)
        ? *running_var_opt
        : Tensor::ones({channels}, DType::Float32, input.device());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
    cudnnTensorDescriptor_t y_desc = createTensorDescriptor(output_bn);
    cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);

    float alpha = 1.0f;
    float beta = 0.0f;
    Tensor saved_mean;
    Tensor saved_inverse_variance;

    if (training) {
        saved_mean = Tensor::empty({channels}, DType::Float32, input.device());
        saved_inverse_variance = Tensor::empty({channels}, DType::Float32, input.device());
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            bias.data_ptr(),
            momentum,
            running_mean.data_ptr(),
            running_var.data_ptr(),
            eps,
            saved_mean.data_ptr(),
            saved_inverse_variance.data_ptr()));
    } else {
        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            bias.data_ptr(),
            running_mean.data_ptr(),
            running_var.data_ptr(),
            eps));
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    return output;
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    bool training,
    double eps) {
    check_batch_norm_input(input);
    if (grad_output.dtype() != DType::Float32 || grad_output.numel() != input.numel()) {
        TP_THROW(RuntimeError, "batch_norm_backward CUDA expects a Float32 gradient matching input");
    }

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_bn = batch_norm_view(input_contig);
    Tensor grad_output_bn = batch_norm_view(grad_output_contig);
    Tensor grad_input = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
    Tensor grad_input_bn = batch_norm_view(grad_input);

    int64_t channels = input.size(1);
    bool has_weight = defined(weight_opt);
    Tensor scale = has_weight
        ? *weight_opt
        : Tensor::ones({channels}, DType::Float32, input.device());
    Tensor grad_scale = Tensor::zeros({channels}, DType::Float32, input.device());
    Tensor grad_bias = Tensor::zeros({channels}, DType::Float32, input.device());

    // cuDNN's backward API consumes the batch mean and inverse variance saved
    // by its training forward.  Recompute those values from the saved input;
    // the public TensorPlay BN node intentionally stores only the input and
    // running-state handles, matching the CPU implementation.
    Tensor saved_mean = Tensor::empty({channels}, DType::Float32, input.device());
    Tensor saved_inverse_variance = Tensor::empty({channels}, DType::Float32, input.device());
    if (training) {
        Tensor forward_output = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
        Tensor forward_output_bn = batch_norm_view(forward_output);
        Tensor scratch_running_mean = Tensor::zeros({channels}, DType::Float32, input.device());
        Tensor scratch_running_var = Tensor::ones({channels}, DType::Float32, input.device());
        Tensor zero_bias = Tensor::zeros({channels}, DType::Float32, input.device());

        cudnnHandle_t handle = CUDAContext::getCudnnHandle();
        cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
        cudnnTensorDescriptor_t y_desc = createTensorDescriptor(forward_output_bn);
        cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            x_desc,
            input_bn.data_ptr(),
            y_desc,
            forward_output_bn.data_ptr(),
            bn_desc,
            scale.data_ptr(),
            zero_bias.data_ptr(),
            1.0,
            scratch_running_mean.data_ptr(),
            scratch_running_var.data_ptr(),
            eps,
            saved_mean.data_ptr(),
            saved_inverse_variance.data_ptr()));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    } else {
        if (!defined(running_mean_opt) || !defined(running_var_opt)) {
            TP_THROW(RuntimeError, "batch_norm_backward CUDA eval mode requires running statistics");
        }
        Tensor running_mean = *running_mean_opt;
        Tensor running_var = *running_var_opt;
        // The kernel only materializes the C-element inverse variance used by
        // cudnnBatchNormalizationBackward; running statistics stay untouched.
        int threads = 256;
        int blocks = static_cast<int>((channels + threads - 1) / threads);
        inverse_variance_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            channels,
            running_var.data_ptr<float>(),
            saved_inverse_variance.data_ptr<float>(),
            static_cast<float>(eps));
        cudaMemcpyAsync(
            saved_mean.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            static_cast<size_t>(channels) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            getCurrentCUDAStream().stream());
        check_cuda_launch("batch_norm_backward eval statistics");
    }

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnTensorDescriptor_t x_desc = createTensorDescriptor(input_bn);
    cudnnTensorDescriptor_t dy_desc = createTensorDescriptor(grad_output_bn);
    cudnnTensorDescriptor_t dx_desc = createTensorDescriptor(grad_input_bn);
    cudnnTensorDescriptor_t bn_desc = derive_bn_descriptor(x_desc);
    float alpha_data = 1.0f;
    float beta_data = 0.0f;
    float alpha_param = 1.0f;
    float beta_param = 0.0f;
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_data,
        &beta_data,
        &alpha_param,
        &beta_param,
        x_desc,
        input_bn.data_ptr(),
        dy_desc,
        grad_output_bn.data_ptr(),
        dx_desc,
        grad_input_bn.data_ptr(),
        bn_desc,
        scale.data_ptr(),
        grad_scale.data_ptr(),
        grad_bias.data_ptr(),
        eps,
        saved_mean.data_ptr(),
        saved_inverse_variance.data_ptr()));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));

    if (!has_weight) {
        grad_scale = Tensor();
        grad_bias = Tensor();
    }
    return std::make_tuple(grad_input, grad_scale, grad_bias);
}

#else

Tensor batch_norm_cuda(
    const Tensor&, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, std::optional<Tensor>, bool, double, double) {
    TP_THROW(NotImplementedError, "batch_norm CUDA requires cuDNN");
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor&, const Tensor&, std::optional<Tensor>, std::optional<Tensor>,
    std::optional<Tensor>, bool, double) {
    TP_THROW(NotImplementedError, "batch_norm_backward CUDA requires cuDNN");
}

#endif // USE_CUDNN

// ============================================================================
// Layer Normalization (custom kernels, no cuDNN dependency)
//
// Algorithm mirrors third_party/pytorch/aten/src/ATen/native/cuda/
// layer_norm_kernel.cu: per-row Welford moments combined through warp
// shuffles and shared memory, fp32 accumulation for Half/BFloat16 inputs,
// fused stats+apply forward pass with vectorized loads when N % 4 == 0.
// ============================================================================

namespace layer_norm {

constexpr int kLNThreads = 256;

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
__device__ inline double ln_rsqrt(double v) { return 1.0 / sqrt(v); }

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
        other.mean = __shfl_down_sync(0xffffffff, val.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, val.count, offset);
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

// Two-sum block reduction (warp shuffles, then shared memory across warps).
// On return smem0[0]/smem1[0] hold the sums and every thread is in sync.
template <typename ACC>
__device__ inline void ln_block_reduce2(ACC& v0, ACC& v1, ACC* smem0, ACC* smem1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffff, v0, offset);
        v1 += __shfl_down_sync(0xffffffff, v1, offset);
    }
    if (lane == 0) { smem0[wid] = v0; smem1[wid] = v1; }
    __syncthreads();
    v0 = (lane < static_cast<int>(blockDim.x >> 5)) ? smem0[lane] : ACC(0);
    v1 = (lane < static_cast<int>(blockDim.x >> 5)) ? smem1[lane] : ACC(0);
    if (wid == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, offset);
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
        }
    }
    if (threadIdx.x == 0) { smem0[0] = v0; smem1[0] = v1; }
    __syncthreads();
    v0 = smem0[0];
    v1 = smem1[0];
}

// Fused forward: one block per row, Welford stats + normalize in one launch.
// VEC > 1 requires N % VEC == 0 and 16B-aligned row pointers.
template <typename T, typename ACC, int VEC>
__global__ void layer_norm_forward_kernel(
    int64_t N, ACC eps,
    const T* __restrict__ X,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ Y) {
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

// grad_weight/grad_bias: column-parallel deterministic reduction (no atomics).
template <typename T, typename ACC>
__global__ void layer_norm_gamma_beta_kernel(
    int64_t M, int64_t N,
    const T* __restrict__ dY,
    const T* __restrict__ X,
    const ACC* __restrict__ mean,
    const ACC* __restrict__ rstd,
    T* __restrict__ dGamma,
    T* __restrict__ dBeta) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= N) return;
    ACC gw = 0, gb = 0;
    for (int64_t i = 0; i < M; ++i) {
        const int64_t idx = i * N + c;
        const ACC dy = static_cast<ACC>(dY[idx]);
        gb += dy;
        gw += dy * ((static_cast<ACC>(X[idx]) - mean[i]) * rstd[i]);
    }
    if (dGamma) dGamma[c] = static_cast<T>(gw);
    if (dBeta) dBeta[c] = static_cast<T>(gb);
}

inline bool ln_ptr_aligned(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <typename T, typename ACC>
void launch_layer_norm_forward(
    int64_t M, int64_t N, double eps,
    const T* X, const T* gamma, const T* beta, T* Y) {
    const bool vec_ok = (N % 4 == 0) && ln_ptr_aligned(X) && ln_ptr_aligned(Y) &&
        (!gamma || ln_ptr_aligned(gamma)) && (!beta || ln_ptr_aligned(beta));
    const auto stream = getCurrentCUDAStream().stream();
    if (vec_ok) {
        layer_norm_forward_kernel<T, ACC, 4><<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
            N, static_cast<ACC>(eps), X, gamma, beta, Y);
    } else {
        layer_norm_forward_kernel<T, ACC, 1><<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
            N, static_cast<ACC>(eps), X, gamma, beta, Y);
    }
}

template <typename T, typename ACC>
void launch_layer_norm_backward(
    int64_t M, int64_t N, double eps,
    const T* dY, const T* X, const T* gamma, T* dX,
    T* dGamma, T* dBeta) {
    const auto stream = getCurrentCUDAStream().stream();
    // One contiguous [2, M] buffer: mean at offset 0, rstd at offset M.
    Tensor stats = Tensor::empty(
        std::vector<int64_t>{2 * M},
        (std::is_same<ACC, double>::value ? DType::Float64 : DType::Float32),
        Device(DeviceType::CUDA));
    ACC* mean = stats.data_ptr<ACC>();
    ACC* rstd = stats.data_ptr<ACC>() + M;

    const bool vec_ok = (N % 4 == 0) && ln_ptr_aligned(X);
    if (vec_ok) {
        layer_norm_moments_kernel<T, ACC, 4><<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
            N, static_cast<ACC>(eps), X, mean, rstd);
    } else {
        layer_norm_moments_kernel<T, ACC, 1><<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
            N, static_cast<ACC>(eps), X, mean, rstd);
    }

    layer_norm_grad_input_kernel<T, ACC><<<static_cast<unsigned>(M), kLNThreads, 0, stream>>>(
        N, dY, X, mean, rstd, gamma, dX);

    const unsigned gthreads = 256;
    const unsigned gblocks = static_cast<unsigned>((N + gthreads - 1) / gthreads);
    layer_norm_gamma_beta_kernel<T, ACC><<<gblocks, gthreads, 0, stream>>>(
        M, N, dY, X, mean, rstd, dGamma, dBeta);
}

} // namespace layer_norm

Tensor layer_norm_cuda(const Tensor& input,
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

    Tensor in_contig = input.contiguous();
    Tensor weight = has_weight ? weight_opt->contiguous() : Tensor();
    Tensor bias = has_bias ? bias_opt->contiguous() : Tensor();

    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(in_contig.shape()),
                               in_contig.dtype(), in_contig.device());
    if (in_contig.numel() == 0 || M == 0 || N == 0) return out;

    switch (in_contig.dtype()) {
#define LN_FORWARD_CASE(ctype, name, acc_t)                                   \
        case DType::name:                                                     \
            layer_norm::launch_layer_norm_forward<ctype, acc_t>(              \
                M, N, eps,                                                    \
                in_contig.data_ptr<ctype>(),                                  \
                has_weight ? weight.data_ptr<ctype>() : nullptr,              \
                has_bias ? bias.data_ptr<ctype>() : nullptr,                  \
                out.data_ptr<ctype>());                                       \
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
            TP_THROW(RuntimeError, std::string("layer_norm_cuda: ") + cudaGetErrorString(error));
        }
    }
    return out;
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

TENSORPLAY_LIBRARY_IMPL(CUDA, NormalizationKernels) {
    m.impl("batch_norm", batch_norm_cuda);
    m.impl("batch_norm_backward", batch_norm_backward_cuda);
    m.impl("layer_norm", layer_norm_cuda);
    m.impl("layer_norm_backward", layer_norm_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
