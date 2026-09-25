#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor batch_norm_cuda(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps);

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double eps);

Tensor group_norm_cuda(
    const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt, double eps);

std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps);

namespace {

constexpr int kInstanceThreads = 256;

void check_instance_norm_cuda_status(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(operation) + ": " +
                                   cudaGetErrorString(error));
    }
}

void check_instance_norm_cuda_launch(const char* operation) {
    check_instance_norm_cuda_status(cudaGetLastError(), operation);
}

template <typename ACC>
__device__ inline ACC instance_rsqrt(ACC value) {
    if constexpr (std::is_same<ACC, float>::value) {
        return rsqrtf(value);
    } else {
        return ACC(1) / ::sqrt(value);
    }
}

template <typename ACC>
__device__ inline void instance_block_reduce2(ACC& v0, ACC& v1,
                                              ACC* smem0, ACC* smem1) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    if (lane == 0) {
        smem0[wid] = v0;
        smem1[wid] = v1;
    }
    __syncthreads();
    const int nw = static_cast<int>(blockDim.x >> 5);
    v0 = lane < nw ? smem0[lane] : ACC(0);
    v1 = lane < nw ? smem1[lane] : ACC(0);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        v0 += __shfl_down_sync(0xffffffffffffffffull, v0, offset);
        v1 += __shfl_down_sync(0xffffffffffffffffull, v1, offset);
    }
    v0 = __shfl_sync(0xffffffffffffffffull, v0, 0);
    v1 = __shfl_sync(0xffffffffffffffffull, v1, 0);
}

template <typename T, typename ACC>
__global__ void instance_moments_impl(int64_t spatial, ACC eps,
                                      const T* __restrict__ input,
                                      ACC* __restrict__ mean_out,
                                      ACC* __restrict__ rstd_out) {
    __shared__ ACC smem0[kInstanceThreads / 32];
    __shared__ ACC smem1[kInstanceThreads / 32];
    const int64_t row = blockIdx.x;
    const T* x = input + row * spatial;
    ACC sum = ACC(0);
    ACC sum_sq = ACC(0);
    for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
        const ACC value = static_cast<ACC>(x[i]);
        sum += value;
        sum_sq += value * value;
    }
    instance_block_reduce2(sum, sum_sq, smem0, smem1);
    if (threadIdx.x == 0) {
        const ACC mean = sum / static_cast<ACC>(spatial);
        const ACC variance = sum_sq / static_cast<ACC>(spatial) - mean * mean;
        mean_out[row] = mean;
        rstd_out[row] = instance_rsqrt(variance + eps);
    }
}

template <typename ACC>
__global__ void instance_running_stats_impl(
    int64_t N, int64_t C, int64_t spatial, double momentum, double eps,
    const ACC* __restrict__ mean, const ACC* __restrict__ rstd,
    ACC* __restrict__ running_mean, ACC* __restrict__ running_var) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const ACC m = static_cast<ACC>(momentum);
    ACC batch_mean = ACC(0);
    ACC batch_var = ACC(0);
    for (int64_t n = 0; n < N; ++n) {
        batch_mean += mean[n * C + c];
        const ACC r = rstd[n * C + c];
        batch_var += ACC(1) / (r * r) - static_cast<ACC>(eps);
    }
    batch_mean /= static_cast<ACC>(N);
    batch_var /= static_cast<ACC>(N);
    if (spatial > 1) {
        batch_var = batch_var * static_cast<ACC>(spatial) /
                    static_cast<ACC>(spatial - 1);
    }
    running_mean[c] = (ACC(1) - m) * running_mean[c] + m * batch_mean;
    running_var[c] = (ACC(1) - m) * running_var[c] + m * batch_var;
}

}

Tensor instance_norm_cuda(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool use_input_stats, double momentum, double eps) {
    if (!use_input_stats) {
        return batch_norm_cuda(input, weight_opt, bias_opt, running_mean_opt,
                               running_var_opt, false, momentum, eps);
    }

    const int64_t C = input.size(1);
    Tensor x = input.contiguous();
    Tensor out = group_norm_cuda(x, C, weight_opt, bias_opt, eps);

    if (running_mean_opt.has_value() && running_mean_opt->defined() &&
        running_var_opt.has_value() && running_var_opt->defined()) {
        const int64_t N = x.size(0);
        const int64_t spatial = x.numel() / (N * C);
        const auto stream = getCurrentCUDAStream().stream();
        if (x.dtype() == DType::Float32) {
            Tensor stats = Tensor::empty({N * C * 2}, DType::Float32, x.device());
            float* mean = stats.data_ptr<float>();
            float* rstd = mean + N * C;
            instance_moments_impl<float, float>
                <<<N * C, kInstanceThreads, 0, stream>>>(
                    spatial, static_cast<float>(eps), x.data_ptr<float>(),
                    mean, rstd);
            check_instance_norm_cuda_launch("instance_norm moments");
            instance_running_stats_impl<float>
                <<<(C + 255) / 256, 256, 0, stream>>>(
                    N, C, spatial, momentum, eps, mean, rstd,
                    running_mean_opt->data_ptr<float>(),
                    running_var_opt->data_ptr<float>());
            check_instance_norm_cuda_launch("instance_norm running statistics");
        } else if (x.dtype() == DType::Float64) {
            Tensor stats = Tensor::empty({N * C * 2}, DType::Float64, x.device());
            double* mean = stats.data_ptr<double>();
            double* rstd = mean + N * C;
            instance_moments_impl<double, double>
                <<<N * C, kInstanceThreads, 0, stream>>>(
                    spatial, static_cast<double>(eps), x.data_ptr<double>(),
                    mean, rstd);
            check_instance_norm_cuda_launch("instance_norm moments");
            instance_running_stats_impl<double>
                <<<(C + 255) / 256, 256, 0, stream>>>(
                    N, C, spatial, momentum, eps, mean, rstd,
                    running_mean_opt->data_ptr<double>(),
                    running_var_opt->data_ptr<double>());
            check_instance_norm_cuda_launch("instance_norm running statistics");
        }
    }
    return out;
}

std::tuple<Tensor, Tensor, Tensor> instance_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool use_input_stats, double eps) {
    if (use_input_stats) {
        const int64_t C = input.size(1);
        return group_norm_backward_cuda(grad_output, input, C, weight_opt,
                                        bias_opt, eps);
    }
    return batch_norm_backward_cuda(grad_output, input, weight_opt,
                                    running_mean_opt, running_var_opt, false,
                                    eps);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, InstanceNormKernels) {
    m.impl("instance_norm", instance_norm_cuda);
    m.impl("instance_norm_backward", instance_norm_backward_cuda);
}

}
}
