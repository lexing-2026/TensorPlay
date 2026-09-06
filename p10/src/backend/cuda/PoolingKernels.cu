#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include "Atomic.cuh"
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <limits>
#include <optional>
#include <tuple>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

template <typename T> struct PoolMath { using type = T; };
template <> struct PoolMath<tensorplay::Half> { using type = float; };
template <> struct PoolMath<tensorplay::BFloat16> { using type = float; };

namespace {
    std::vector<int64_t> expand_param_if_needed(const std::vector<int64_t>& list, int64_t n, int64_t default_val) {
        if (list.empty()) return std::vector<int64_t>(n, default_val);
        if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
        if (list.size() != n) TP_THROW(ValueError, "Parameter size mismatch");
        return list;
    }

    std::pair<int64_t, int64_t> get_pair(const std::vector<int64_t>& value) {
        if (value.size() == 1) return {value[0], value[0]};
        if (value.size() == 2) return {value[0], value[1]};
        TP_THROW(ValueError, "adaptive_avg_pool2d: output_size must have one or two elements");
    }

    bool is_channels_last_4d(const Tensor& tensor) {
        if (tensor.dim() != 4) return false;
        const int64_t c = tensor.size(1);
        const int64_t h = tensor.size(2);
        const int64_t w = tensor.size(3);
        return tensor.stride(0) == c * h * w &&
               tensor.stride(1) == 1 &&
               tensor.stride(2) == w * c &&
               tensor.stride(3) == c;
    }

    Tensor empty_pool_output(
        int64_t n,
        int64_t c,
        int64_t h,
        int64_t w,
        DType dtype,
        const Device& device,
        bool channels_last) {
        const std::vector<int64_t> shape{n, c, h, w};
        Tensor result = Tensor::empty(shape, dtype, device);
        if (!channels_last) return result;
        return result.as_strided(
            shape, {c * h * w, 1, w * c, c});
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

Tensor adaptive_avg_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_avg_pool2d_cuda(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_avg_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    int64_t elements = output.numel();
    if (elements == 0) return output;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_avg_pool2d_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float64:
            adaptive_avg_pool2d_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float16:
            adaptive_avg_pool2d_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(), output.data_ptr<tensorplay::Half>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
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
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_avg_pool2d CUDA: ") + cudaGetErrorString(error));
    return output;
}

Tensor adaptive_avg_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input) {
    if (input.dim() == 3 && grad_output.dim() == 3) {
        return adaptive_avg_pool2d_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) || input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_avg_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
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
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_avg_pool2d_backward CUDA: ") + cudaGetErrorString(error));
    return grad_input;
}

// thread per output element in tp's pooling-kernel style. Window bounds come
// from AdaptivePooling.h start_index/end_index (floor start, ceil end); NaN
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

// window argmax (the dispatcher op returns values only) and scatter
// grad_output atomically, since windows overlap.
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

// with_indices variants: the forward also records the plane-linear argmax so
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

Tensor adaptive_max_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_max_pool2d_cuda(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d: Expected 4D input");
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_max_pool2d: Invalid output size");

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    int64_t elements = output.numel();
    if (elements == 0) return output;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float64:
            adaptive_max_pool2d_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
            break;
        case DType::Float16:
            adaptive_max_pool2d_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(), output.data_ptr<tensorplay::Half>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
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
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_max_pool2d CUDA: ") + cudaGetErrorString(error));
    return output;
}

Tensor adaptive_max_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input) {
    if (input.dim() == 3 && grad_output.dim() == 3) {
        return adaptive_max_pool2d_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_max_pool2d_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) || input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
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
                    input_contig.data_ptr<tensorplay::Half>(), grad_input.data_ptr<tensorplay::Half>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    grad_output.size(2), grad_output.size(3));
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
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_max_pool2d_backward CUDA: ") + cudaGetErrorString(error));
    return grad_input;
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices_cuda(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        auto result = adaptive_max_pool2d_with_indices_cuda(input.unsqueeze(0), output_size);
        return std::make_tuple(std::get<0>(result).squeeze(0), std::get<1>(result).squeeze(0));
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices: Expected 4D input");
    if (!is_adaptive_pool_cuda_dtype(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices: Invalid output size");

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({input.size(0), input.size(1), H_out, W_out}, DType::Int64, input.device());
    int64_t elements = output.numel();
    if (elements == 0) return std::make_tuple(output, indices);
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            adaptive_max_pool2d_with_indices_forward_kernel<float, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<float>(), output.data_ptr<float>(),
                    indices.data_ptr<int64_t>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), H_out, W_out);
            break;
        case DType::Float64:
            adaptive_max_pool2d_with_indices_forward_kernel<double, double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<double>(), output.data_ptr<double>(),
                    indices.data_ptr<int64_t>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), H_out, W_out);
            break;
        case DType::Float16:
            adaptive_max_pool2d_with_indices_forward_kernel<tensorplay::Half, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::Half>(), output.data_ptr<tensorplay::Half>(),
                    indices.data_ptr<int64_t>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), H_out, W_out);
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_with_indices_forward_kernel<tensorplay::BFloat16, float>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    input_contig.data_ptr<tensorplay::BFloat16>(),
                    output.data_ptr<tensorplay::BFloat16>(), indices.data_ptr<int64_t>(),
                    input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_max_pool2d_with_indices CUDA: ") + cudaGetErrorString(error));
    return std::make_tuple(output, indices);
}

Tensor adaptive_max_pool2d_with_indices_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                                      const std::vector<int64_t>& output_size, const Tensor& indices) {
    (void)output_size;
    if (input.dim() == 3 && grad_output.dim() == 3 && indices.dim() == 3) {
        return adaptive_max_pool2d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), output_size,
                   indices.unsqueeze(0)).squeeze(0);
    }
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) || input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
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
                    grad_input.data_ptr<float>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float64:
            adaptive_max_pool2d_with_indices_backward_kernel<double>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<double>(), indices_contig.data_ptr<int64_t>(),
                    grad_input.data_ptr<double>(), input.size(0), input.size(1), input.size(2),
                    input.size(3), grad_output.size(2), grad_output.size(3));
            break;
        case DType::Float16:
            adaptive_max_pool2d_with_indices_backward_kernel<tensorplay::Half>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::Half>(),
                    indices_contig.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::Half>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    grad_output.size(2), grad_output.size(3));
            break;
        case DType::BFloat16:
            adaptive_max_pool2d_with_indices_backward_kernel<tensorplay::BFloat16>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    grad_output_contig.data_ptr<tensorplay::BFloat16>(),
                    indices_contig.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::BFloat16>(),
                    input.size(0), input.size(1), input.size(2), input.size(3),
                    grad_output.size(2), grad_output.size(3));
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_max_pool2d_with_indices_backward CUDA: ") + cudaGetErrorString(error));
    return grad_input;
}

#ifdef USE_CUDNN
struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    TensorDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }

    void set(const Tensor& t) {
        cudnnDataType_t dtype;
        if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
        else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");

        int n = static_cast<int>(t.size(0));
        int c = static_cast<int>(t.size(1));
        int h = static_cast<int>(t.size(2));
        int w = static_cast<int>(t.size(3));

        // cuDNN's Ex descriptor preserves the actual logical tensor strides,
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            desc, dtype, n, c, h, w,
            static_cast<int>(t.stride(0)),
            static_cast<int>(t.stride(1)),
            static_cast<int>(t.stride(2)),
            static_cast<int>(t.stride(3))));
    }
};

struct PoolingDesc {
    cudnnPoolingDescriptor_t desc;
    PoolingDesc() { CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc)); }
    ~PoolingDesc() { cudnnDestroyPoolingDescriptor(desc); }
    operator cudnnPoolingDescriptor_t() const { return desc; }
    
    void set(cudnnPoolingMode_t mode, int h, int w, int pad_h, int pad_w, int str_h, int str_w) {
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc, mode, CUDNN_NOT_PROPAGATE_NAN, h, w, pad_h, pad_w, str_h, str_w));
    }
};
#endif

Tensor max_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, bool ceil_mode) {
#ifdef USE_CUDNN
    auto kernel_size = expand_param_if_needed(kernel_size_arg, 2, 0);
    auto stride = stride_arg;
    if (stride.empty()) stride = kernel_size;
    else stride = expand_param_if_needed(stride_arg, 2, 0);
    
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    
    if (dilation[0] != 1 || dilation[1] != 1) {
        TP_THROW(NotImplementedError, "max_pool2d_cuda: dilation not supported by cuDNN");
    }
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input);
    
    PoolingDesc pool_desc;
    pool_desc.set(CUDNN_POOLING_MAX, (int)kernel_size[0], (int)kernel_size[1], (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1]);
    
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pool_desc, x_desc, &n, &c, &h, &w));
    
    Tensor out = empty_pool_output(
        n, c, h, w, input.dtype(), input.device(), is_channels_last_4d(input));
    TensorDesc y_desc; y_desc.set(out);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, alpha_p, x_desc, input.data_ptr(), beta_p, y_desc, out.data_ptr()));
    
    return out;
#else
    TP_THROW(NotImplementedError, "max_pool2d_cuda requires cuDNN");
#endif
}

Tensor avg_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, bool ceil_mode, bool count_include_pad) {
#ifdef USE_CUDNN
    auto kernel_size = expand_param_if_needed(kernel_size_arg, 2, 0);
    auto stride = stride_arg;
    if (stride.empty()) stride = kernel_size;
    else stride = expand_param_if_needed(stride_arg, 2, 0);
    
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input);
    
    PoolingDesc pool_desc;
    cudnnPoolingMode_t mode = count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    pool_desc.set(mode, (int)kernel_size[0], (int)kernel_size[1], (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1]);
    
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pool_desc, x_desc, &n, &c, &h, &w));
    
    Tensor out = empty_pool_output(
        n, c, h, w, input.dtype(), input.device(), is_channels_last_4d(input));
    TensorDesc y_desc; y_desc.set(out);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, alpha_p, x_desc, input.data_ptr(), beta_p, y_desc, out.data_ptr()));
    
    return out;
#else
    TP_THROW(NotImplementedError, "avg_pool2d_cuda requires cuDNN");
#endif
}

Tensor max_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, bool ceil_mode) {
#ifdef USE_CUDNN
    auto kernel_size = expand_param_if_needed(kernel_size_arg, 2, 0);
    auto stride = stride_arg;
    if (stride.empty()) stride = kernel_size;
    else stride = expand_param_if_needed(stride_arg, 2, 0);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);

    if (dilation[0] != 1 || dilation[1] != 1) {
        TP_THROW(NotImplementedError, "max_pool2d_backward_cuda: dilation not supported by cuDNN");
    }
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input);
    TensorDesc dy_desc; dy_desc.set(grad_output);
    
    // Recompute output as required by cudnnPoolingBackward
    Tensor output = max_pool2d_cuda(input, kernel_size_arg, stride_arg, padding_arg, dilation_arg, ceil_mode);
    TensorDesc y_desc; y_desc.set(output);
    
    PoolingDesc pool_desc;
    pool_desc.set(CUDNN_POOLING_MAX, (int)kernel_size[0], (int)kernel_size[1], (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1]);
    
    Tensor grad_input = Tensor::empty_like(input, DType::Undefined, input.device());
    TensorDesc dx_desc; dx_desc.set(grad_input);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnPoolingBackward(handle, pool_desc, alpha_p, y_desc, output.data_ptr(), dy_desc, grad_output.data_ptr(), x_desc, input.data_ptr(), beta_p, dx_desc, grad_input.data_ptr()));
    
    return grad_input;
#else
    TP_THROW(NotImplementedError, "max_pool2d_backward_cuda requires cuDNN");
#endif
}

Tensor avg_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, bool ceil_mode, bool count_include_pad) {
#ifdef USE_CUDNN
    auto kernel_size = expand_param_if_needed(kernel_size_arg, 2, 0);
    auto stride = stride_arg;
    if (stride.empty()) stride = kernel_size;
    else stride = expand_param_if_needed(stride_arg, 2, 0);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input);
    TensorDesc dy_desc; dy_desc.set(grad_output);
    
    // Recompute output as required by cudnnPoolingBackward
    Tensor output = avg_pool2d_cuda(input, kernel_size_arg, stride_arg, padding_arg, ceil_mode, count_include_pad);
    TensorDesc y_desc; y_desc.set(output);
    
    PoolingDesc pool_desc;
    cudnnPoolingMode_t mode = count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    pool_desc.set(mode, (int)kernel_size[0], (int)kernel_size[1], (int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1]);
    
    Tensor grad_input = Tensor::empty_like(input, DType::Undefined, input.device());
    TensorDesc dx_desc; dx_desc.set(grad_input);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnPoolingBackward(handle, pool_desc, alpha_p, y_desc, output.data_ptr(), dy_desc, grad_output.data_ptr(), x_desc, input.data_ptr(), beta_p, dx_desc, grad_input.data_ptr()));
    
    return grad_input;
#else
    TP_THROW(NotImplementedError, "avg_pool2d_backward_cuda requires cuDNN");
#endif
}

// ---------------------------------------------------------------------------
// DilatedMaxPool2d.cpp / DilatedMaxPool3d.cpp / AdaptiveMaxPooling3d.cpp).
// cuDNN pooling exposes neither dilation nor argmax indices, so these run as
// plain CUDA kernels: one thread per output element, grid-stride, atomicAdd
// scatter in the backwards.  Half/BFloat16 compute in float (opmath).
// ---------------------------------------------------------------------------

inline int64_t pool_grid_blocks(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

static inline int64_t pool_div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

static int64_t pool_output_shape(int64_t in, int64_t k, int64_t pad,
                                 int64_t stride, int64_t dilation, bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    int64_t out = pool_div_rtn(in + 2 * pad - dilation * (k - 1) - 1 +
                                   (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

static std::vector<int64_t> pool_expand_param(const std::vector<int64_t>& list,
                                              const char* name, int64_t n,
                                              int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if ((int64_t)list.size() != n)
        TP_THROW(ValueError, std::string(name) + ": expected " + std::to_string(n) + " values");
    return list;
}

template <typename T, typename M>
__global__ void max_pool2d_wi_fwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW,
    int64_t pH, int64_t pW, int64_t dH, int64_t dW,
    const T* __restrict__ input, T* __restrict__ output,
    int64_t* __restrict__ indices) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        const T* plane = input + nc * H_in * W_in;
        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t hi = h * sH - pH + kh * dH;
            if (hi < 0 || hi >= H_in) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t wi = w * sW - pW + kw * dW;
                if (wi < 0 || wi >= W_in) continue;
                const int64_t idx = hi * W_in + wi;
                const M val = static_cast<M>(plane[idx]);
                if ((val > max_val) || std::isnan(val)) {
                    max_val = val;
                    max_idx = idx;
                }
            }
        }
        output[i] = static_cast<T>(max_val);
        indices[i] = max_idx;
    }
}

template <typename T>
__global__ void max_pool_wi_bwd_kernel(
    int64_t total, int64_t out_spatial, int64_t in_plane,
    const T* __restrict__ grad_output, const int64_t* __restrict__ indices,
    T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        const int64_t idx = indices[i];
        if (idx < 0) continue;
        const int64_t nc = i / out_spatial;
        gpuAtomicAdd(grad_input + nc * in_plane + idx, grad_output[i]);
    }
}

// Average pooling shares the grid-stride structure of the max variants.
// Padded positions contribute zero to the numerator either way; the divisor
// is the window size or the count of valid positions depending on
// count_include_pad.  A window that lands entirely in padding divides by
// zero, which yields zero here as well.
template <typename T, typename M>
__global__ void avg_pool2d_fwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW,
    bool count_include_pad, const T* __restrict__ input,
    T* __restrict__ output) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        const T* plane = input + nc * H_in * W_in;
        M acc = M(0);
        int64_t cnt = 0;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t hi = h * sH - pH + kh;
            if (hi < 0 || hi >= H_in) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t wi = w * sW - pW + kw;
                if (wi < 0 || wi >= W_in) continue;
                acc += static_cast<M>(plane[hi * W_in + wi]);
                ++cnt;
            }
        }
        const M divisor = count_include_pad ? static_cast<M>(kH * kW)
                                            : static_cast<M>(cnt);
        output[i] = divisor > M(0) ? static_cast<T>(acc / divisor) : T(0);
    }
}

template <typename T, typename M>
__global__ void avg_pool2d_bwd_kernel(
    int64_t total, int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out,
    int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t pH, int64_t pW,
    bool count_include_pad, const T* __restrict__ grad_output,
    T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = H_out * W_out;
    const int64_t in_plane = H_in * W_in;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t nc = i / out_spatial;
        int64_t cnt = 0;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t hi = h * sH - pH + kh;
            if (hi < 0 || hi >= H_in) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t wi = w * sW - pW + kw;
                if (wi < 0 || wi >= W_in) continue;
                ++cnt;
            }
        }
        const M divisor = count_include_pad ? static_cast<M>(kH * kW)
                                            : static_cast<M>(cnt);
        if (divisor <= M(0)) continue;
        const M g = static_cast<M>(grad_output[i]) / divisor;
        T* plane = grad_input + nc * in_plane;
        for (int64_t kh = 0; kh < kH; ++kh) {
            const int64_t hi = h * sH - pH + kh;
            if (hi < 0 || hi >= H_in) continue;
            for (int64_t kw = 0; kw < kW; ++kw) {
                const int64_t wi = w * sW - pW + kw;
                if (wi < 0 || wi >= W_in) continue;
                gpuAtomicAdd(plane + hi * W_in + wi, static_cast<T>(g));
            }
        }
    }
}

template <typename T, typename M>
__global__ void max_pool3d_wi_fwd_kernel(
    int64_t total, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW, int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW,
    const T* __restrict__ input, T* __restrict__ output,
    int64_t* __restrict__ indices) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = D_out * H_out * W_out;
    for (; i < total; i += stride) {
        const int64_t w = i % W_out;
        const int64_t h = (i / W_out) % H_out;
        const int64_t d = (i / (W_out * H_out)) % D_out;
        const int64_t nc = i / out_spatial;
        const T* vol = input + nc * D_in * H_in * W_in;
        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (int64_t kd = 0; kd < kD; ++kd) {
            const int64_t di = d * sD - pD + kd * dD;
            if (di < 0 || di >= D_in) continue;
            for (int64_t kh = 0; kh < kH; ++kh) {
                const int64_t hi = h * sH - pH + kh * dH;
                if (hi < 0 || hi >= H_in) continue;
                for (int64_t kw = 0; kw < kW; ++kw) {
                    const int64_t wi = w * sW - pW + kw * dW;
                    if (wi < 0 || wi >= W_in) continue;
                    const int64_t idx = (di * H_in + hi) * W_in + wi;
                    const M val = static_cast<M>(vol[idx]);
                    if ((val > max_val) || std::isnan(val)) {
                        max_val = val;
                        max_idx = idx;
                    }
                }
            }
        }
        output[i] = static_cast<T>(max_val);
        indices[i] = max_idx;
    }
}

template <typename T, typename M>
__global__ void adaptive_max_pool3d_fwd_kernel(
    int64_t total, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    const T* __restrict__ input, T* __restrict__ output) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = oD * oH * oW;
    for (; i < total; i += stride) {
        const int64_t w = i % oW;
        const int64_t h = (i / oW) % oH;
        const int64_t d = (i / (oW * oH)) % oD;
        const int64_t nc = i / out_spatial;
        const T* vol = input + nc * D * H * W;
        const int64_t ds = d * D / oD, de = 1 + (((d + 1) * D) - 1) / oD;
        const int64_t hs = h * H / oH, he = 1 + (((h + 1) * H) - 1) / oH;
        const int64_t ws = w * W / oW, we = 1 + (((w + 1) * W) - 1) / oW;
        M max_val = -std::numeric_limits<M>::infinity();
        for (int64_t z = ds; z < de; ++z)
        for (int64_t y = hs; y < he; ++y)
        for (int64_t x = ws; x < we; ++x) {
            const M val = static_cast<M>(vol[(z * H + y) * W + x]);
            if ((val > max_val) || std::isnan(val)) max_val = val;
        }
        output[i] = static_cast<T>(max_val);
    }
}

template <typename T, typename M>
__global__ void adaptive_max_pool3d_bwd_kernel(
    int64_t total, int64_t D, int64_t H, int64_t W,
    int64_t oD, int64_t oH, int64_t oW,
    const T* __restrict__ input, const T* __restrict__ grad_output,
    T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t out_spatial = oD * oH * oW;
    for (; i < total; i += stride) {
        const int64_t w = i % oW;
        const int64_t h = (i / oW) % oH;
        const int64_t d = (i / (oW * oH)) % oD;
        const int64_t nc = i / out_spatial;
        const T* vol = input + nc * D * H * W;
        const int64_t ds = d * D / oD, de = 1 + (((d + 1) * D) - 1) / oD;
        const int64_t hs = h * H / oH, he = 1 + (((h + 1) * H) - 1) / oH;
        const int64_t ws = w * W / oW, we = 1 + (((w + 1) * W) - 1) / oW;
        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (int64_t z = ds; z < de; ++z)
        for (int64_t y = hs; y < he; ++y)
        for (int64_t x = ws; x < we; ++x) {
            const int64_t idx = (z * H + y) * W + x;
            const M val = static_cast<M>(vol[idx]);
            if ((val > max_val) || std::isnan(val)) {
                max_val = val;
                max_idx = idx;
            }
        }
        if (max_idx >= 0)
            gpuAtomicAdd(grad_input + nc * D * H * W + max_idx, grad_output[i]);
    }
}

#define POOL_CUDA_DISPATCH(ctype, name, ...)                                \
    case DType::name: {                                                     \
        using M = typename PoolMath<ctype>::type;                           \
        __VA_ARGS__;                                                        \
        break;                                                              \
    }

std::tuple<Tensor, Tensor> max_pool2d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        auto r = max_pool2d_with_indices_cuda(input.unsqueeze(0), kernel_size,
                                              stride, padding, dilation, ceil_mode);
        return std::make_tuple(std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0));
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d_with_indices: Expected 4D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(kernel_size, "max_pool2d_with_indices kernel_size", 2, 1);
    const auto st = pool_expand_param(stride.empty() ? ks : stride, "max_pool2d_with_indices stride", 2, ks[0]);
    const auto pd = pool_expand_param(padding, "max_pool2d_with_indices padding", 2, 0);
    const auto dl = pool_expand_param(dilation, "max_pool2d_with_indices dilation", 2, 1);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1], dH = dl[0], dW = dl[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, dW, ceil_mode);
    if (H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "max_pool2d_with_indices: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({N, C, H_out, W_out}, DType::Int64, input.device());
    const int64_t total = N * C * H_out * W_out;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();

    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool2d_wi_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW,
                input_c.data_ptr<float>(), out.data_ptr<float>(), indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool2d_wi_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW,
                input_c.data_ptr<double>(), out.data_ptr<double>(), indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool2d_wi_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW,
                input_c.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
                indices.data_ptr<int64_t>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool2d_wi_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW,
                input_c.data_ptr<tensorplay::BFloat16>(), out.data_ptr<tensorplay::BFloat16>(),
                indices.data_ptr<int64_t>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return std::make_tuple(out, indices);
}

Tensor max_pool2d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size; (void)stride; (void)padding; (void)dilation; (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined())
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: indices is required");
    if (input.dim() == 3) {
        // Unbatched (C,H,W): pool as a batch of one, matching the forward.
        return max_pool2d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode, indices_opt->unsqueeze(0))
            .squeeze(0);
    }
    if (grad_output.dim() != 4 || input.dim() != 4)
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    const Tensor& idx_shape_ref = *indices_opt;
    if (idx_shape_ref.dim() != 4 || idx_shape_ref.size(0) != grad_output.size(0) ||
        idx_shape_ref.size(1) != grad_output.size(1) ||
        idx_shape_ref.size(2) != grad_output.size(2) ||
        idx_shape_ref.size(3) != grad_output.size(3)) {
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: expected grad_output with shape [",
                 grad_output.size(0), ", ", grad_output.size(1), ", ", grad_output.size(2), ", ",
                 grad_output.size(3), "] to match indices shape [",
                 idx_shape_ref.size(0), ", ", idx_shape_ref.size(1), ", ", idx_shape_ref.size(2),
                 ", ", idx_shape_ref.size(3), "]");
    }
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices_opt->contiguous();
    Tensor grad_input = Tensor::zeros_like(input);
    const int64_t total = go.numel();
    const int64_t out_spatial = go.size(2) * go.size(3);
    const int64_t in_plane = input.size(2) * input.size(3);
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool_wi_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<float>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool_wi_bwd_kernel<double><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<double>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool_wi_bwd_kernel<tensorplay::Half><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::Half>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool_wi_bwd_kernel<tensorplay::BFloat16><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::BFloat16>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        auto r = max_pool3d_with_indices_cuda(input.unsqueeze(0), kernel_size,
                                              stride, padding, dilation, ceil_mode);
        return std::make_tuple(std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0));
    }
    if (input.dim() != 5) TP_THROW(RuntimeError, "max_pool3d_with_indices: Expected 5D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D_in = input_c.size(2), H_in = input_c.size(3), W_in = input_c.size(4);

    const auto ks = pool_expand_param(kernel_size, "max_pool3d kernel_size", 3, 1);
    const auto st = pool_expand_param(stride.empty() ? ks : stride, "max_pool3d stride", 3, ks[0]);
    const auto pd = pool_expand_param(padding, "max_pool3d padding", 3, 0);
    const auto dl = pool_expand_param(dilation, "max_pool3d dilation", 3, 1);
    const int64_t kD = ks[0], kH = ks[1], kW = ks[2];
    const int64_t sD = st[0], sH = st[1], sW = st[2];
    const int64_t pD = pd[0], pH = pd[1], pW = pd[2];
    const int64_t dD = dl[0], dH = dl[1], dW = dl[2];

    const int64_t D_out = pool_output_shape(D_in, kD, pD, sD, dD, ceil_mode);
    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, dW, ceil_mode);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "max_pool3d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({N, C, D_out, H_out, W_out}, DType::Int64, input.device());
    const int64_t total = N * C * D_out * H_out * W_out;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();

    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool3d_wi_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, D_in, H_in, W_in, D_out, H_out, W_out,
                kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                input_c.data_ptr<float>(), out.data_ptr<float>(), indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool3d_wi_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, D_in, H_in, W_in, D_out, H_out, W_out,
                kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                input_c.data_ptr<double>(), out.data_ptr<double>(), indices.data_ptr<int64_t>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool3d_wi_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, D_in, H_in, W_in, D_out, H_out, W_out,
                kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                input_c.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>(),
                indices.data_ptr<int64_t>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool3d_wi_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, D_in, H_in, W_in, D_out, H_out, W_out,
                kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                input_c.data_ptr<tensorplay::BFloat16>(), out.data_ptr<tensorplay::BFloat16>(),
                indices.data_ptr<int64_t>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d_with_indices CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return std::make_tuple(out, indices);
}

Tensor max_pool3d_cuda(const Tensor& input, const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        return max_pool3d_cuda(input.unsqueeze(0), kernel_size, stride, padding,
                               dilation, ceil_mode).squeeze(0);
    }
    return std::get<0>(max_pool3d_with_indices_cuda(input, kernel_size, stride,
                                                    padding, dilation, ceil_mode));
}

Tensor max_pool3d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt);

Tensor max_pool3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& kernel_size,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation, bool ceil_mode) {
    // Reuse the indices path: recompute argmax via the forward kernel, then
    if (grad_output.dim() == 4 && input.dim() == 4) {
        return max_pool3d_backward_cuda(grad_output.unsqueeze(0), input.unsqueeze(0),
                                        kernel_size, stride, padding, dilation,
                                        ceil_mode).squeeze(0);
    }
    auto fw = max_pool3d_with_indices_cuda(input, kernel_size, stride, padding,
                                           dilation, ceil_mode);
    return max_pool3d_with_indices_backward_cuda(grad_output, input, kernel_size,
                                                 stride, padding, dilation, ceil_mode,
                                                 std::get<1>(fw));
}

Tensor max_pool3d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size; (void)stride; (void)padding; (void)dilation; (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined())
        TP_THROW(RuntimeError, "max_pool3d_with_indices_backward: indices is required");
    if (input.dim() == 4) {
        // Unbatched (C,D,H,W): pool as a batch of one, matching the forward.
        return max_pool3d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode, indices_opt->unsqueeze(0))
            .squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "max_pool3d_with_indices_backward: Expected 5D input and grad_output");
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices_opt->contiguous();
    Tensor grad_input = Tensor::zeros_like(input);
    const int64_t total = go.numel();
    const int64_t out_spatial = go.size(2) * go.size(3) * go.size(4);
    const int64_t in_plane = input.size(2) * input.size(3) * input.size(4);
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool_wi_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<float>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool_wi_bwd_kernel<double><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<double>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool_wi_bwd_kernel<tensorplay::Half><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::Half>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool_wi_bwd_kernel<tensorplay::BFloat16><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::BFloat16>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

Tensor adaptive_max_pool3d_cuda(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 4)
        return adaptive_max_pool3d_cuda(input.unsqueeze(0), output_size).squeeze(0);
    if (input.dim() != 5) TP_THROW(RuntimeError, "adaptive_max_pool3d: Expected 5D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D = input_c.size(2), H = input_c.size(3), W = input_c.size(4);
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    if (oD <= 0 || oH <= 0 || oW <= 0)
        TP_THROW(RuntimeError, "adaptive_max_pool3d: Invalid output size");
    Tensor out = Tensor::empty({N, C, oD, oH, oW}, input.dtype(), input.device());
    const int64_t total = N * C * oD * oH * oW;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            adaptive_max_pool3d_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<float>(), out.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            adaptive_max_pool3d_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<double>(), out.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (adaptive_max_pool3d_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<tensorplay::Half>(), out.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (adaptive_max_pool3d_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<tensorplay::BFloat16>(), out.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool3d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return out;
}

Tensor adaptive_max_pool3d_backward_cuda(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() == 4 && input.dim() == 4)
        return adaptive_max_pool3d_backward_cuda(grad_output.unsqueeze(0),
                                                 input.unsqueeze(0)).squeeze(0);
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "adaptive_max_pool3d_backward: Expected 5D input and grad_output");
    const Tensor input_c = input.contiguous();
    const Tensor go = grad_output.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D = input_c.size(2), H = input_c.size(3), W = input_c.size(4);
    const int64_t oD = go.size(2), oH = go.size(3), oW = go.size(4);
    Tensor grad_input = Tensor::zeros({N, C, D, H, W}, input.dtype(), input.device());
    const int64_t total = go.numel();
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            adaptive_max_pool3d_bwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<float>(), go.data_ptr<float>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            adaptive_max_pool3d_bwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<double>(), go.data_ptr<double>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (adaptive_max_pool3d_bwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<tensorplay::Half>(), go.data_ptr<tensorplay::Half>(),
                grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (adaptive_max_pool3d_bwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, D, H, W, oD, oH, oW,
                input_c.data_ptr<tensorplay::BFloat16>(), go.data_ptr<tensorplay::BFloat16>(),
                grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool3d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

// Native average-pooling entry points.  They exist so the AMD build can route
// avg_pool2d off the DNN library the same way max_pool2d already runs on its
// own kernels; on the CUDA build the cuDNN-backed entry points above stay in
// charge.
Tensor avg_pool2d_native_cuda(const Tensor& input,
                              const std::vector<int64_t>& kernel_size_arg,
                              const std::vector<int64_t>& stride_arg,
                              const std::vector<int64_t>& padding_arg,
                              bool ceil_mode, bool count_include_pad) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d: Expected 4D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(kernel_size_arg, "avg_pool2d kernel_size", 2, 1);
    const auto st = pool_expand_param(stride_arg.empty() ? kernel_size_arg : stride_arg,
                                      "avg_pool2d stride", 2, ks[0]);
    const auto pd = pool_expand_param(padding_arg, "avg_pool2d padding", 2, 0);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, 1, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, 1, ceil_mode);
    if (H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "avg_pool2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    const int64_t total = N * C * H_out * W_out;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            avg_pool2d_fwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, input_c.data_ptr<float>(), out.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            avg_pool2d_fwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, input_c.data_ptr<double>(), out.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (avg_pool2d_fwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, input_c.data_ptr<tensorplay::Half>(),
                out.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (avg_pool2d_fwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, input_c.data_ptr<tensorplay::BFloat16>(),
                out.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool2d CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return out;
}

Tensor avg_pool2d_backward_native_cuda(const Tensor& grad_output,
                                       const Tensor& input,
                                       const std::vector<int64_t>& kernel_size_arg,
                                       const std::vector<int64_t>& stride_arg,
                                       const std::vector<int64_t>& padding_arg,
                                       bool ceil_mode, bool count_include_pad) {
    if (input.dim() != 4)
        TP_THROW(RuntimeError, "avg_pool2d_backward: Expected 4D input");
    const Tensor input_c = input.contiguous();
    const Tensor go = grad_output.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = pool_expand_param(kernel_size_arg, "avg_pool2d_backward kernel_size", 2, 1);
    const auto st = pool_expand_param(stride_arg.empty() ? kernel_size_arg : stride_arg,
                                      "avg_pool2d_backward stride", 2, ks[0]);
    const auto pd = pool_expand_param(padding_arg, "avg_pool2d_backward padding", 2, 0);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1];

    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, 1, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, 1, ceil_mode);
    if (go.size(0) != N || go.size(1) != C || go.size(2) != H_out ||
        go.size(3) != W_out)
        TP_THROW(RuntimeError, "avg_pool2d_backward: grad_output shape mismatch");

    Tensor grad_input = Tensor::zeros({N, C, H_in, W_in}, input.dtype(), input.device());
    const int64_t total = go.numel();
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            avg_pool2d_bwd_kernel<float, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, go.data_ptr<float>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            avg_pool2d_bwd_kernel<double, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, go.data_ptr<double>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (avg_pool2d_bwd_kernel<tensorplay::Half, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, go.data_ptr<tensorplay::Half>(),
                grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (avg_pool2d_bwd_kernel<tensorplay::BFloat16, M><<<blocks, threads, 0, stream>>>(
                total, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW,
                count_include_pad, go.data_ptr<tensorplay::BFloat16>(),
                grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool2d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

// Direct max_pool2d_backward without indices: recompute them through the
// with_indices forward, then reuse its scatter kernel.
Tensor max_pool2d_backward_native_cuda(const Tensor& grad_output,
                                       const Tensor& input,
                                       const std::vector<int64_t>& kernel_size_arg,
                                       const std::vector<int64_t>& stride_arg,
                                       const std::vector<int64_t>& padding_arg,
                                       const std::vector<int64_t>& dilation_arg,
                                       bool ceil_mode) {
    Tensor indices = std::get<1>(max_pool2d_with_indices_cuda(
        input, kernel_size_arg, stride_arg, padding_arg, dilation_arg,
        ceil_mode));
    return max_pool2d_with_indices_backward_cuda(
        grad_output, input, kernel_size_arg, stride_arg, padding_arg,
        dilation_arg, ceil_mode, indices);
}

#undef POOL_CUDA_DISPATCH

Tensor& interop_avg_pool2d_out_cuda(const Tensor& self, const std::vector<int64_t>& kernel_size,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding, bool ceil_mode,
                                bool count_include_pad,
                                std::optional<int64_t> divisor_override, Tensor& out) {
        (void)divisor_override;
        out = avg_pool2d_native_cuda(self, kernel_size, stride, padding,
                                     ceil_mode, count_include_pad);
        return out;
    
}

Tensor& interop_avg_pool2d_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input,
              const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
              bool ceil_mode, bool count_include_pad,
              std::optional<int64_t> divisor_override, Tensor& grad_input) {
        (void)divisor_override;
        grad_input = avg_pool2d_backward_native_cuda(
            grad_output, input, kernel_size, stride, padding, ceil_mode,
            count_include_pad);
        return grad_input;
    
}

Tensor& interop_adaptive_avg_pool2d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size, Tensor& out) {
        out = adaptive_avg_pool2d_cuda(self, output_size);
        return out;
    
}

Tensor& interop_adaptive_avg_pool2d_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
        grad_input = adaptive_avg_pool2d_backward_cuda(grad_output, input);
        return grad_input;
    
}

Tensor& interop_adaptive_max_pool2d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
              Tensor& out, Tensor& indices) {
        out = adaptive_max_pool2d_cuda(self, output_size);
        indices = Tensor();
        return out;
    
}

Tensor& interop_adaptive_max_pool2d_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& indices,
              Tensor& grad_input) {
        (void)indices;
        return write_pooling_out("adaptive_max_pool2d_backward",
                                 adaptive_max_pool2d_backward_cuda(grad_output, input),
                                 grad_input);
    
}

Tensor& interop_adaptive_max_pool3d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
              Tensor& out, Tensor& indices) {
        out = adaptive_max_pool3d_cuda(self, output_size);
        indices = Tensor();
        return out;
    
}

Tensor& interop_adaptive_max_pool3d_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& indices,
              Tensor& grad_input) {
        (void)indices;
        return write_pooling_out("adaptive_max_pool3d_backward",
                                 adaptive_max_pool3d_backward_cuda(grad_output, input),
                                 grad_input);
    
}

Tensor& interop_max_pool2d_with_indices_out_cuda(const Tensor& self, const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
              const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out,
              Tensor& indices) {
        std::tie(out, indices) = max_pool2d_with_indices_cuda(
            self, kernel_size, stride, padding, dilation, ceil_mode);
        return out;
    
}

Tensor& interop_max_pool2d_with_indices_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& self,
              const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
              const std::vector<int64_t>& dilation, bool ceil_mode,
              const Tensor& indices, Tensor& grad_input) {
        grad_input = max_pool2d_with_indices_backward_cuda(
            grad_output, self, kernel_size, stride, padding, dilation,
            ceil_mode, indices);
        return grad_input;
    
}

Tensor& interop_max_pool3d_with_indices_out_cuda(const Tensor& self, const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
              const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out,
              Tensor& indices) {
        std::tie(out, indices) = max_pool3d_with_indices_cuda(
            self, kernel_size, stride, padding, dilation, ceil_mode);
        return out;
    
}

Tensor& interop_max_pool3d_with_indices_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& self,
              const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
              const std::vector<int64_t>& dilation, bool ceil_mode,
              const Tensor& indices, Tensor& grad_input) {
        grad_input = max_pool3d_with_indices_backward_cuda(
            grad_output, self, kernel_size, stride, padding, dilation,
            ceil_mode, indices);
        return grad_input;
    
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PoolingKernels) {
    // max_pool2d / max_pool3d / adaptive_max_pool2d are Composite over their
    // *_with_indices variants (see the schema); only the with_indices
    // kernels are registered here so the dispatcher falls back to the Composite
    // forward, which records the indices-scatter autograd node.
#ifdef USE_ROCM
    // The AMD build routes pooling to native kernels: the DNN library's
    // 2-D pooling produces wrong output on the supported GPU (see the
    // compatibility header), and the with_indices kernels above already
    // cover the surface.
    m.impl("avg_pool2d", avg_pool2d_native_cuda);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_native_cuda);
    m.impl("max_pool2d_backward", max_pool2d_backward_native_cuda);
#else
    m.impl("avg_pool2d", avg_pool2d_cuda);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_cuda);
    m.impl("max_pool2d_backward", max_pool2d_backward_cuda);
#endif
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cuda);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cuda);
    m.impl("adaptive_max_pool2d_backward", adaptive_max_pool2d_backward_cuda);
    m.impl("adaptive_max_pool2d_with_indices", adaptive_max_pool2d_with_indices_cuda);
    m.impl("adaptive_max_pool2d_with_indices_backward", adaptive_max_pool2d_with_indices_backward_cuda);
    m.impl("max_pool2d_with_indices", max_pool2d_with_indices_cuda);
    m.impl("max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_cuda);
    m.impl("max_pool3d_backward", max_pool3d_backward_cuda);
    m.impl("max_pool3d_with_indices", max_pool3d_with_indices_cuda);
    m.impl("max_pool3d_with_indices_backward", max_pool3d_with_indices_backward_cuda);
    m.impl("adaptive_max_pool3d", adaptive_max_pool3d_cuda);
    m.impl("adaptive_max_pool3d_backward", adaptive_max_pool3d_backward_cuda);

    // out-variants: run the value kernel, then transfer into the caller's
    // buffer (grad_input for backward spellings).
    m.impl("avg_pool2d.out", interop_avg_pool2d_out_cuda);
    m.impl("avg_pool2d_backward.grad_input", interop_avg_pool2d_backward_grad_input_cuda);
    m.impl("adaptive_avg_pool2d.out", interop_adaptive_avg_pool2d_out_cuda);
    m.impl("adaptive_avg_pool2d_backward.grad_input", interop_adaptive_avg_pool2d_backward_grad_input_cuda);
    m.impl("adaptive_max_pool2d.out", interop_adaptive_max_pool2d_out_cuda);
    m.impl("adaptive_max_pool2d_backward.grad_input", interop_adaptive_max_pool2d_backward_grad_input_cuda);
    m.impl("adaptive_max_pool3d.out", interop_adaptive_max_pool3d_out_cuda);
    m.impl("adaptive_max_pool3d_backward.grad_input", interop_adaptive_max_pool3d_backward_grad_input_cuda);
    m.impl("max_pool2d_with_indices.out", interop_max_pool2d_with_indices_out_cuda);
    m.impl("max_pool2d_with_indices_backward.grad_input", interop_max_pool2d_with_indices_backward_grad_input_cuda);
    m.impl("max_pool3d_with_indices.out", interop_max_pool3d_with_indices_out_cuda);
    m.impl("max_pool3d_with_indices_backward.grad_input", interop_max_pool3d_with_indices_backward_grad_input_cuda);
}
} // namespace cuda
} // namespace tensorplay
