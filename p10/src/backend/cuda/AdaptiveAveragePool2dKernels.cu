#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
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
    TP_THROW(ValueError, "adaptive_avg_pool2d: output_size must have one or two elements");
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

}

template <typename T, typename M>
__global__ void adaptive_avg_pool2d_forward_kernel(
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
    int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;
    int64_t w_start = (w * W_in) / W_out;
    int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;

    M sum = M(0);
    for (int64_t ih = h_start; ih < h_end; ++ih) {
        for (int64_t iw = w_start; iw < w_end; ++iw) {
            int64_t input_index = ((n * C + c) * H_in + ih) * W_in + iw;
            sum += static_cast<M>(input[input_index]);
        }
    }
    output[output_index] = static_cast<T>(
        sum / static_cast<M>((h_end - h_start) * (w_end - w_start)));
}

template <typename T, typename M>
__global__ void adaptive_avg_pool2d_backward_kernel(
    const T* grad_output,
    T* grad_input,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out) {
    int64_t input_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t input_elements = N * C * H_in * W_in;
    if (input_index >= input_elements) return;

    int64_t iw = input_index % W_in;
    int64_t ih = (input_index / W_in) % H_in;
    int64_t c = (input_index / (W_in * H_in)) % C;
    int64_t n = input_index / (W_in * H_in * C);

    M value = M(0);
    for (int64_t h = 0; h < H_out; ++h) {
        int64_t h_start = (h * H_in) / H_out;
        int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;
        if (ih < h_start || ih >= h_end) continue;
        for (int64_t w = 0; w < W_out; ++w) {
            int64_t w_start = (w * W_in) / W_out;
            int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;
            if (iw < w_start || iw >= w_end) continue;
            int64_t output_index = ((n * C + c) * H_out + h) * W_out + w;
            M area = static_cast<M>((h_end - h_start) * (w_end - w_start));
            value += static_cast<M>(grad_output[output_index]) / area;
        }
    }
    grad_input[input_index] = static_cast<T>(value);
}

Tensor adaptive_avg_pool2d_cuda(const Tensor& input,
                                 const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_avg_pool2d_cuda(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_avg_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");
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
            adaptive_avg_pool2d_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        case DType::Float64:
            adaptive_avg_pool2d_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    H_out, W_out);
            break;
        case DType::Float16:
            adaptive_avg_pool2d_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(),
                    output.data_ptr<tensorplay::Half>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::BFloat16:
            adaptive_avg_pool2d_forward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::BFloat16>(),
                    output.data_ptr<tensorplay::BFloat16>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), H_out, W_out);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_avg_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_avg_pool2d CUDA: ") + cudaGetErrorString(error));
    }
    return output;
}

Tensor adaptive_avg_pool2d_backward_cuda(const Tensor& grad_output,
                                          const Tensor& input) {
    if (input.dim() == 3 && grad_output.dim() == 3) {
        return adaptive_avg_pool2d_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError,
                 "adaptive_avg_pool2d_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) ||
        input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_avg_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous()
        ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty_like(
        input_contig, DType::Undefined, input_contig.device());
    int64_t elements = grad_input.numel();
    if (elements == 0) return grad_input;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_avg_pool2d_backward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<float>(), grad_input.data_ptr<float>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float64:
            adaptive_avg_pool2d_backward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<double>(), grad_input.data_ptr<double>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float16:
            adaptive_avg_pool2d_backward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::Half>(),
                    grad_input.data_ptr<tensorplay::Half>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::BFloat16:
            adaptive_avg_pool2d_backward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::BFloat16>(),
                    grad_input.data_ptr<tensorplay::BFloat16>(), input.size(0), input.size(1),
                    input.size(2), input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_avg_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("adaptive_avg_pool2d_backward CUDA: ") +
                     cudaGetErrorString(error));
    }
    return grad_input;
}

Tensor& interop_adaptive_avg_pool2d_out_cuda(
    const Tensor& self, const std::vector<int64_t>& output_size, Tensor& out) {
    return write_pooling_out(
        "adaptive_avg_pool2d", adaptive_avg_pool2d_cuda(self, output_size), out);
}

Tensor& interop_adaptive_avg_pool2d_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
    return write_pooling_out(
        "adaptive_avg_pool2d_backward",
        adaptive_avg_pool2d_backward_cuda(grad_output, input), grad_input);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, AdaptiveAveragePool2dKernels) {
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cuda);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cuda);
    m.impl("adaptive_avg_pool2d.out", interop_adaptive_avg_pool2d_out_cuda);
    m.impl("adaptive_avg_pool2d_backward.grad_input",
           interop_adaptive_avg_pool2d_backward_grad_input_cuda);
}

}
}
