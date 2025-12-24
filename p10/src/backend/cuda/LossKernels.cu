#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDAContext.h"
#include <cuda_runtime.h>
#include <optional>
#include <tuple>

namespace tensorplay {
namespace cuda {

template <typename T, typename TargetT>
__global__ void nll_loss_forward_kernel(
    int64_t n, int64_t C,
    const T* input,
    const TargetT* target,
    const T* weight,
    T* output,
    int64_t ignore_index) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        TargetT t = target[i];
        if (t == ignore_index) {
            output[i] = 0;
            return;
        }
        if (t >= 0 && t < C) {
            T val = input[i * C + t];
            T w = (weight != nullptr) ? weight[t] : static_cast<T>(1);
            output[i] = -val * w;
        } else {
            output[i] = 0; 
        }
    }
}

template <typename T>
__global__ void nll_loss_atomic_kernel(
    int64_t n, int64_t C,
    const T* input,
    const int64_t* target,
    const T* weight,
    T* output_loss,
    T* output_weight,
    int64_t ignore_index) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t t = target[i];
        if (t != ignore_index && t >= 0 && t < C) {
            T val = input[i * C + t];
            T w = (weight != nullptr) ? weight[t] : static_cast<T>(1);
            
            atomicAdd(output_loss, -val * w);
            if (output_weight) {
                atomicAdd(output_weight, w);
            }
        }
    }
}

template <typename T>
__global__ void div_ptrs_kernel(T* a, const T* b) {
    if (b[0] != 0) a[0] /= b[0];
}

std::tuple<Tensor, Tensor> nll_loss_cuda(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    
    Tensor weight;
    if (weight_opt.has_value() && weight_opt->defined()) weight = *weight_opt;
    
    if (reduction == 0) { // None
        Tensor losses = Tensor::empty({N}, input.dtype(), input.device());
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        if (input.dtype() == DType::Float32) {
            nll_loss_forward_kernel<float, int64_t><<<blocks, threads>>>(
                N, C, input.data_ptr<float>(), target.data_ptr<int64_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                losses.data_ptr<float>(), ignore_index);
        } else {
             TP_THROW(NotImplementedError, "nll_loss CUDA: only float32 supported");
        }
        return std::make_tuple(losses, Tensor());
    } else {
        Tensor result = Tensor::zeros({}, input.dtype(), input.device());
        Tensor total_weight = Tensor::zeros({}, input.dtype(), input.device());
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        if (input.dtype() == DType::Float32) {
            nll_loss_atomic_kernel<float><<<blocks, threads>>>(
                N, C, input.data_ptr<float>(), target.data_ptr<int64_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                result.data_ptr<float>(),
                total_weight.data_ptr<float>(),
                ignore_index);
        } else {
             TP_THROW(NotImplementedError, "nll_loss CUDA: only float32 supported");
        }
        
        if (reduction == 1) { // Mean
            if (input.dtype() == DType::Float32) {
                div_ptrs_kernel<float><<<1, 1>>>(result.data_ptr<float>(), total_weight.data_ptr<float>());
            }
        }
        
        return std::make_tuple(result, total_weight);
    }
}

template <typename T, typename TargetT>
__global__ void nll_loss_backward_kernel(
    int64_t n, int64_t C,
    const T* grad_output,
    const TargetT* target,
    const T* weight,
    const T* total_weight,
    T* grad_input,
    int64_t ignore_index,
    int reduction) {
    
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        TargetT t = target[i];
        if (t == ignore_index || t < 0 || t >= C) return;
        
        T w = (weight != nullptr) ? weight[t] : static_cast<T>(1);
        T g = (reduction == 0) ? grad_output[i] : grad_output[0];
        
        if (reduction == 1 && total_weight) { // Mean
             g /= total_weight[0];
        }
        
        grad_input[i * C + t] = -g * w;
    }
}

Tensor nll_loss_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    Tensor grad_input = Tensor::zeros_like(input);
    
    Tensor weight;
    if (weight_opt.has_value() && weight_opt->defined()) weight = *weight_opt;
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    if (input.dtype() == DType::Float32) {
        nll_loss_backward_kernel<float, int64_t><<<blocks, threads>>>(
            N, C,
            grad_output.data_ptr<float>(),
            target.data_ptr<int64_t>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            total_weight.defined() ? total_weight.data_ptr<float>() : nullptr,
            grad_input.data_ptr<float>(),
            ignore_index,
            (int)reduction
        );
    } else {
        TP_THROW(NotImplementedError, "nll_loss_backward CUDA: only float32 supported");
    }
    
    return grad_input;
}

Tensor mse_loss_cuda(const Tensor& input, const Tensor& target, int64_t reduction) {
    Tensor diff = input - target;
    Tensor sq_diff = diff * diff;
    if (reduction == 0) return sq_diff;
    if (reduction == 1) return sq_diff.mean();
    if (reduction == 2) return sq_diff.sum();
    TP_THROW(ValueError, "Invalid reduction mode");
}

template <typename T>
__global__ void mse_loss_backward_kernel_cuda_impl(int64_t n, const T* grad_output, const T* input, const T* target, T* grad_input, int64_t reduction, T scale) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T diff = input[i] - target[i];
        T g = (reduction == 0) ? grad_output[i] : grad_output[0];
        grad_input[i] = scale * diff * g;
    }
}

Tensor mse_loss_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
    int64_t n = input.numel();
    Tensor grad_input = Tensor::empty_like(input);
    
    double scale = 2.0;
    if (reduction == 1) { // Mean
        scale /= (double)n;
    }
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    if (input.dtype() == DType::Float32) {
        mse_loss_backward_kernel_cuda_impl<float><<<blocks, threads>>>(
            n,
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            target.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            reduction,
            (float)scale
        );
    } else {
        TP_THROW(NotImplementedError, "mse_loss_backward CUDA: only float32 supported");
    }
    
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LossKernels) {
    m.impl("nll_loss", nll_loss_cuda);
    m.impl("nll_loss_backward", nll_loss_backward_cuda);
    m.impl("mse_loss", mse_loss_cuda);
    m.impl("mse_loss_backward", mse_loss_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
