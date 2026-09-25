#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

std::pair<int64_t, int64_t> get_pair(const std::vector<int64_t>& value) {
    if (value.size() == 1) return {value[0], value[0]};
    if (value.size() == 2) return {value[0], value[1]};
    TP_THROW(ValueError, "adaptive_max_pool2d: output_size must have one or two elements");
}

bool is_adaptive_pool_cuda_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Float16 || dtype == DType::BFloat16;
}

Tensor& write_pooling_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = std::move(value);
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

template <typename T, typename M>
__global__ void adaptive_max_pool2d_forward_kernel(
    const T* input,
    T* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out) {
    int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t output_elements = N * C * H_out * W_out;
    if (output_index >= output_elements) return;

    int64_t w = output_index % W_out;
    int64_t h = (output_index / W_out) % H_out;
    int64_t c = (output_index / (W_out * H_out)) % C;
    int64_t n = output_index / (W_out * H_out * C);

    int64_t h_start = (h * H_in) / H_out;
    int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;
    int64_t w_start = (w * W_in) / W_out;
    int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

    const T* plane = input + (n * C + c) * H_in * W_in;
    M max_val = -std::numeric_limits<M>::infinity();
    for (int64_t ih = h_start; ih < h_end; ++ih) {
        for (int64_t iw = w_start; iw < w_end; ++iw) {
            M val = static_cast<M>(plane[ih * W_in + iw]);
            if ((val > max_val) || isnan(val)) max_val = val;
        }
    }
    output[output_index] = static_cast<T>(max_val);
}

template <typename T, typename M>
__global__ void adaptive_max_pool2d_backward_kernel(
    const T* grad_output,
    const T* input,
    T* grad_input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out) {
    int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t output_elements = N * C * H_out * W_out;
    if (output_index >= output_elements) return;

    int64_t w = output_index % W_out;
    int64_t h = (output_index / W_out) % H_out;
    int64_t c = (output_index / (W_out * H_out)) % C;
    int64_t n = output_index / (W_out * H_out * C);

    int64_t h_start = (h * H_in) / H_out;
    int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;
    int64_t w_start = (w * W_in) / W_out;
    int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

    const T* plane = input + (n * C + c) * H_in * W_in;
    M max_val = -std::numeric_limits<M>::infinity();
    int64_t max_idx = h_start * W_in + w_start;
    for (int64_t ih = h_start; ih < h_end; ++ih) {
        for (int64_t iw = w_start; iw < w_end; ++iw) {
            int64_t idx = ih * W_in + iw;
            M val = static_cast<M>(plane[idx]);
            if ((val > max_val) || isnan(val)) {
                max_val = val;
                max_idx = idx;
            }
        }
    }
    gpuAtomicAdd(grad_input + (n * C + c) * H_in * W_in + max_idx,
                 grad_output[output_index]);
}

template <typename T, typename M>
__global__ void adaptive_max_pool2d_with_indices_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out) {
    int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t output_elements = N * C * H_out * W_out;
    if (output_index >= output_elements) return;

    int64_t w = output_index % W_out;
    int64_t h = (output_index / W_out) % H_out;
    int64_t c = (output_index / (W_out * H_out)) % C;
    int64_t n = output_index / (W_out * H_out * C);

    int64_t h_start = (h * H_in) / H_out;
    int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;
    int64_t w_start = (w * W_in) / W_out;
    int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

    const T* plane = input + (n * C + c) * H_in * W_in;
    M max_val = -std::numeric_limits<M>::infinity();
    int64_t max_idx = h_start * W_in + w_start;
    for (int64_t ih = h_start; ih < h_end; ++ih) {
        for (int64_t iw = w_start; iw < w_end; ++iw) {
            int64_t idx = ih * W_in + iw;
            M val = static_cast<M>(plane[idx]);
            if ((val > max_val) || isnan(val)) {
                max_val = val;
                max_idx = idx;
            }
        }
    }
    output[output_index] = static_cast<T>(max_val);
    indices[output_index] = max_idx;
}

template <typename T>
__global__ void adaptive_max_pool2d_with_indices_backward_kernel(
    const T* grad_output,
    const int64_t* indices,
    T* grad_input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out) {
    int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t output_elements = N * C * H_out * W_out;
    if (output_index >= output_elements) return;

    int64_t c = (output_index / (W_out * H_out)) % C;
    int64_t n = output_index / (W_out * H_out * C);
    const int64_t max_idx = indices[output_index];
    if (max_idx < 0) return;
    gpuAtomicAdd(grad_input + (n * C + c) * H_in * W_in + max_idx,
                 grad_output[output_index]);
}

}

Tensor adaptive_max_pool2d_cuda(const Tensor& input,
                                const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_max_pool2d_cuda(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_max_pool2d: Expected 4D input");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "adaptive_max_pool2d: Invalid output size");
    }

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty(
        {input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    int64_t elements = output.numel();
    if (elements == 0) return output;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        case DType::Float64:
            adaptive_max_pool2d_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        case DType::Float16:
            adaptive_max_pool2d_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(),
                    output.data_ptr<tensorplay::Half>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_forward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::BFloat16>(),
                    output.data_ptr<tensorplay::BFloat16>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_max_pool2d CUDA: ") + cudaGetErrorString(error));
    }
    return output;
}

Tensor adaptive_max_pool2d_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input) {
    if (input.dim() == 3 && grad_output.dim() == 3) {
        return adaptive_max_pool2d_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) ||
        input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous()
        ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros_like(input_contig);
    int64_t elements = grad_output_contig.numel();
    if (elements == 0 || input_contig.numel() == 0) return grad_input;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_backward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<float>(), input_contig.data_ptr<float>(),
                    grad_input.data_ptr<float>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float64:
            adaptive_max_pool2d_backward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<double>(), input_contig.data_ptr<double>(),
                    grad_input.data_ptr<double>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float16:
            adaptive_max_pool2d_backward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::Half>(),
                    input_contig.data_ptr<tensorplay::Half>(),
                    grad_input.data_ptr<tensorplay::Half>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_backward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::BFloat16>(),
                    input_contig.data_ptr<tensorplay::BFloat16>(),
                    grad_input.data_ptr<tensorplay::BFloat16>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_max_pool2d_backward CUDA: ") +
                     cudaGetErrorString(error));
    }
    return grad_input;
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        auto result = adaptive_max_pool2d_with_indices_cuda(
            input.unsqueeze(0), output_size);
        return std::make_tuple(std::get<0>(result).squeeze(0),
                               std::get<1>(result).squeeze(0));
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_with_indices: Expected 4D input");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_with_indices: Invalid output size");
    }

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty(
        {input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty(
        {input.size(0), input.size(1), H_out, W_out}, DType::Int64, input.device());
    int64_t elements = output.numel();
    if (elements == 0) return std::make_tuple(output, indices);
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_with_indices_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    indices.data_ptr<int64_t>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float64:
            adaptive_max_pool2d_with_indices_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    indices.data_ptr<int64_t>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float16:
            adaptive_max_pool2d_with_indices_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(),
                    output.data_ptr<tensorplay::Half>(), indices.data_ptr<int64_t>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_with_indices_forward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::BFloat16>(),
                    output.data_ptr<tensorplay::BFloat16>(), indices.data_ptr<int64_t>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_max_pool2d_with_indices CUDA: ") +
                     cudaGetErrorString(error));
    }
    return std::make_tuple(output, indices);
}

Tensor adaptive_max_pool2d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& output_size, const Tensor& indices) {
    (void)output_size;
    if (input.dim() == 3 && grad_output.dim() == 3 && indices.dim() == 3) {
        return adaptive_max_pool2d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), output_size,
                   indices.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) ||
        input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor grad_output_contig = grad_output.is_contiguous()
        ? grad_output : grad_output.contiguous();
    Tensor indices_contig = indices.is_contiguous() ? indices : indices.contiguous();
    Tensor grad_input = Tensor::zeros_like(input);
    int64_t elements = grad_output_contig.numel();
    if (elements == 0 || input.numel() == 0) return grad_input;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_with_indices_backward_kernel<float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<float>(), indices_contig.data_ptr<int64_t>(),
                    grad_input.data_ptr<float>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float64:
            adaptive_max_pool2d_with_indices_backward_kernel<double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<double>(), indices_contig.data_ptr<int64_t>(),
                    grad_input.data_ptr<double>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float16:
            adaptive_max_pool2d_with_indices_backward_kernel<tensorplay::Half>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::Half>(),
                    indices_contig.data_ptr<int64_t>(),
                    grad_input.data_ptr<tensorplay::Half>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_with_indices_backward_kernel<tensorplay::BFloat16>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::BFloat16>(),
                    indices_contig.data_ptr<int64_t>(),
                    grad_input.data_ptr<tensorplay::BFloat16>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_max_pool2d_with_indices_backward CUDA: ") +
                     cudaGetErrorString(error));
    }
    return grad_input;
}

std::tuple<Tensor, Tensor> interop_adaptive_max_pool2d_out_cuda(
    const Tensor& self, const std::vector<int64_t>& output_size,
    Tensor& out, Tensor& indices) {
    auto result = adaptive_max_pool2d_with_indices_cuda(self, output_size);
    write_pooling_out("adaptive_max_pool2d", std::get<0>(result), out);
    write_pooling_out("adaptive_max_pool2d", std::get<1>(result), indices);
    return {out, indices};
}

Tensor& interop_adaptive_max_pool2d_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& indices,
    Tensor& grad_input) {
    return write_pooling_out(
        "adaptive_max_pool2d_backward",
        adaptive_max_pool2d_with_indices_backward_cuda(
            grad_output, input, {}, indices),
        grad_input);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, AdaptiveMaxPool2dKernels) {
    m.impl("adaptive_max_pool2d", adaptive_max_pool2d_cuda);
    m.impl("adaptive_max_pool2d_backward", adaptive_max_pool2d_backward_cuda);
    m.impl("adaptive_max_pool2d_with_indices", adaptive_max_pool2d_with_indices_cuda);
    m.impl("adaptive_max_pool2d_with_indices_backward",
           adaptive_max_pool2d_with_indices_backward_cuda);
    m.impl("adaptive_max_pool2d.out", interop_adaptive_max_pool2d_out_cuda);
    m.impl("adaptive_max_pool2d_backward.grad_input",
           interop_adaptive_max_pool2d_backward_grad_input_cuda);
}

}
}
