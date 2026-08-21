#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

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
}

__global__ void adaptive_avg_pool2d_forward_kernel(
    const float* input,
    float* output,
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

    float sum = 0.0f;
    for (int64_t ih = h_start; ih < h_end; ++ih) {
        for (int64_t iw = w_start; iw < w_end; ++iw) {
            int64_t input_index = ((n * C + c) * H_in + ih) * W_in + iw;
            sum += input[input_index];
        }
    }
    output[output_index] = sum / static_cast<float>((h_end - h_start) * (w_end - w_start));
}

__global__ void adaptive_avg_pool2d_backward_kernel(
    const float* grad_output,
    float* grad_input,
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

    float value = 0.0f;
    for (int64_t h = 0; h < H_out; ++h) {
        int64_t h_start = (h * H_in) / H_out;
        int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;
        if (ih < h_start || ih >= h_end) continue;
        for (int64_t w = 0; w < W_out; ++w) {
            int64_t w_start = (w * W_in) / W_out;
            int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;
            if (iw < w_start || iw >= w_end) continue;
            int64_t output_index = ((n * C + c) * H_out + h) * W_out + w;
            float area = static_cast<float>((h_end - h_start) * (w_end - w_start));
            value += grad_output[output_index] / area;
        }
    }
    grad_input[input_index] = value;
}

Tensor adaptive_avg_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    if (input.dtype() != DType::Float32) {
        TP_THROW(NotImplementedError, "adaptive_avg_pool2d CUDA only supports Float32");
    }
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");

    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor output = Tensor::empty({input.size(0), input.size(1), H_out, W_out}, input.dtype(), input.device());
    int64_t elements = output.numel();
    if (elements == 0) return output;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    adaptive_avg_pool2d_forward_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        input_contig.data_ptr<float>(), output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3), H_out, W_out);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_avg_pool2d CUDA: ") + cudaGetErrorString(error));
    return output;
}

Tensor adaptive_avg_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input) {
    if (input.dim() != 4 || grad_output.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d_backward: Expected 4D input and grad_output");
    }
    if (input.dtype() != DType::Float32 || grad_output.dtype() != DType::Float32) {
        TP_THROW(NotImplementedError, "adaptive_avg_pool2d_backward CUDA only supports Float32");
    }
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty_like(input_contig, DType::Undefined, input_contig.device());
    int64_t elements = grad_input.numel();
    if (elements == 0) return grad_input;
    int threads = 256;
    int blocks = static_cast<int>((elements + threads - 1) / threads);
    adaptive_avg_pool2d_backward_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
        grad_output_contig.data_ptr<float>(), grad_input.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        grad_output.size(2), grad_output.size(3));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) TP_THROW(RuntimeError, std::string("adaptive_avg_pool2d_backward CUDA: ") + cudaGetErrorString(error));
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
        // matching the descriptor construction in PyTorch's cuDNN v8 path.
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

TENSORPLAY_LIBRARY_IMPL(CUDA, PoolingKernels) {
    m.impl("max_pool2d", max_pool2d_cuda);
    m.impl("max_pool2d_backward", max_pool2d_backward_cuda);
    m.impl("avg_pool2d", avg_pool2d_cuda);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_cuda);
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cuda);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
