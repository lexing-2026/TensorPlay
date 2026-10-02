#include "Tensor.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <optional>

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

Tensor avg_pool2d_native_cuda(const Tensor& input,
                              const std::vector<int64_t>& kernel_size_arg,
                              const std::vector<int64_t>& stride_arg,
                              const std::vector<int64_t>& padding_arg,
                              bool ceil_mode, bool count_include_pad,
                              std::optional<int64_t> divisor_override);
Tensor avg_pool2d_backward_native_cuda(const Tensor& grad_output,
                                       const Tensor& input,
                                       const std::vector<int64_t>& kernel_size_arg,
                                       const std::vector<int64_t>& stride_arg,
                                       const std::vector<int64_t>& padding_arg,
                                       bool ceil_mode, bool count_include_pad,
                                       std::optional<int64_t> divisor_override);

// The DNN library takes float and double only, counts its windows rounding
// down, and takes the full window as the divisor, so an explicit divisor, a
// window count rounded up by ceil_mode, and any narrower element type all go
// through the native kernel instead.
static inline bool avg_pool2d_prefers_native(const Tensor& input,
                                             bool ceil_mode,
                                             bool /*count_include_pad*/,
                                             const std::optional<int64_t>& divisor_override) {
    if (divisor_override.has_value() || ceil_mode) {
        return true;
    }
    return !(input.dtype() == DType::Float32 || input.dtype() == DType::Float64);
}

Tensor avg_pool2d_cuda(const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (avg_pool2d_prefers_native(input, ceil_mode, count_include_pad, divisor_override)) {
        return avg_pool2d_native_cuda(input, kernel_size_arg, stride_arg, padding_arg,
                                      ceil_mode, count_include_pad, divisor_override);
    }
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

Tensor avg_pool2d_backward_cuda(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size_arg, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (avg_pool2d_prefers_native(input, ceil_mode, count_include_pad, divisor_override)) {
        return avg_pool2d_backward_native_cuda(grad_output, input, kernel_size_arg, stride_arg,
                                               padding_arg, ceil_mode, count_include_pad,
                                               divisor_override);
    }
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
    Tensor output = avg_pool2d_cuda(input, kernel_size_arg, stride_arg, padding_arg, ceil_mode, count_include_pad, std::nullopt);
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



} // namespace cuda
} // namespace tensorplay
