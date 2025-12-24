#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDNNUtils.h"
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
}

#ifdef USE_CUDNN

// RAII Wrappers for cuDNN descriptors
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
        
        // TensorPlay is NCHW by default
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dtype, n, c, h, w));
    }
};

struct FilterDesc {
    cudnnFilterDescriptor_t desc;
    FilterDesc() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }
    
    void set(const Tensor& t) {
        cudnnDataType_t dtype;
        if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
        else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
        
        int k = static_cast<int>(t.size(0));
        int c = static_cast<int>(t.size(1));
        int h = static_cast<int>(t.size(2));
        int w = static_cast<int>(t.size(3));
        
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc, dtype, CUDNN_TENSOR_NCHW, k, c, h, w));
    }
};

struct ConvDesc {
    cudnnConvolutionDescriptor_t desc;
    ConvDesc() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }
    
    void set(int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w, int groups, DType dtype) {
        cudnnDataType_t computeType;
        if (dtype == DType::Float32) computeType = CUDNN_DATA_FLOAT;
        else if (dtype == DType::Float64) computeType = CUDNN_DATA_DOUBLE;
        else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
        
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CROSS_CORRELATION, computeType));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    }
};

#endif

Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    TensorDesc x_desc; x_desc.set(input);
    FilterDesc w_desc; w_desc.set(weight);
    
    ConvDesc conv_desc;
    conv_desc.set((int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1], (int)dilation[0], (int)dilation[1], (int)groups, input.dtype());
    
    // Get output shape
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &n, &c, &h, &w));
    
    Tensor out = Tensor::empty({n, c, h, w}, input.dtype(), input.device());
    TensorDesc y_desc; y_desc.set(out);
    
    // Find algo
    cudnnConvolutionFwdAlgo_t algo;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, x_desc, w_desc, conv_desc, y_desc,
        1, &returnedAlgoCount, &perfResults));
    algo = perfResults.algo;
    
    // Workspace
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionForward(handle, alpha_p, x_desc, input.data_ptr(), w_desc, weight.data_ptr(), conv_desc, algo, workspace, workspace_size, beta_p, y_desc, out.data_ptr()));
    
    if (workspace) cudaFree(workspace);
    
    if (bias.defined()) {
        TensorDesc b_desc;
        Tensor b_reshaped = bias.reshape({1, bias.size(0), 1, 1});
        b_desc.set(b_reshaped);
        
        CUDNN_CHECK(cudnnAddTensor(handle, alpha_p, b_desc, bias.data_ptr(), alpha_p, y_desc, out.data_ptr()));
    }
    
    return out;
#else
    TP_THROW(NotImplementedError, "conv2d_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    TensorDesc dx_desc; dx_desc.set(input); // gradient of input has same shape as input
    FilterDesc w_desc; w_desc.set(weight);
    TensorDesc dy_desc; dy_desc.set(grad_output);
    
    ConvDesc conv_desc;
    conv_desc.set((int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1], (int)dilation[0], (int)dilation[1], (int)groups, input.dtype());
    
    Tensor grad_input = Tensor::empty_like(input, DType::Undefined, input.device());
    
    // Find algo
    cudnnConvolutionBwdDataAlgo_t algo;
    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;
    
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, w_desc, dy_desc, conv_desc, dx_desc,
        1, &returnedAlgoCount, &perfResults));
    algo = perfResults.algo;
    
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, alpha_p, w_desc, weight.data_ptr(), dy_desc, grad_output.data_ptr(), conv_desc, algo, workspace, workspace_size, beta_p, dx_desc, grad_input.data_ptr()));
    
    if (workspace) cudaFree(workspace);
    
    return grad_input;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_input_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    TensorDesc x_desc; x_desc.set(input);
    TensorDesc dy_desc; dy_desc.set(grad_output);
    FilterDesc dw_desc; dw_desc.set(weight); // grad_weight has same shape as weight
    
    ConvDesc conv_desc;
    conv_desc.set((int)padding[0], (int)padding[1], (int)stride[0], (int)stride[1], (int)dilation[0], (int)dilation[1], (int)groups, input.dtype());
    
    Tensor grad_weight = Tensor::empty_like(weight, DType::Undefined, weight.device());
    
    cudnnConvolutionBwdFilterAlgo_t algo;
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, x_desc, dy_desc, conv_desc, dw_desc,
        1, &returnedAlgoCount, &perfResults));
    algo = perfResults.algo;
    
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, dy_desc, conv_desc, dw_desc, algo, &workspace_size));
    
    void* workspace = nullptr;
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, alpha_p, x_desc, input.data_ptr(), dy_desc, grad_output.data_ptr(), conv_desc, algo, workspace, workspace_size, beta_p, dw_desc, grad_weight.data_ptr()));
    
    if (workspace) cudaFree(workspace);
    
    return grad_weight;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_weight_cuda requires cuDNN");
#endif
}

Tensor conv2d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
#ifdef USE_CUDNN
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    
    TensorDesc dy_desc; dy_desc.set(grad_output);
    
    Tensor grad_bias = Tensor::empty({grad_output.size(1)}, grad_output.dtype(), grad_output.device());
    
    TensorDesc db_desc; 
    Tensor grad_bias_reshaped = grad_bias.reshape({1, grad_bias.size(0), 1, 1});
    db_desc.set(grad_bias_reshaped);
    
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (grad_output.dtype() == DType::Float64) {
        alpha_p = &alpha_d; beta_p = &beta_d;
    }
    
    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, alpha_p, dy_desc, grad_output.data_ptr(), beta_p, db_desc, grad_bias.data_ptr()));
    
    return grad_bias;
#else
    TP_THROW(NotImplementedError, "conv2d_grad_bias_cuda requires cuDNN");
#endif
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ConvKernels) {
    m.impl("conv2d", conv2d_cuda);
    m.impl("conv2d_grad_input", conv2d_grad_input_cuda);
    m.impl("conv2d_grad_weight", conv2d_grad_weight_cuda);
    m.impl("conv2d_grad_bias", conv2d_grad_bias_cuda);
}

} // namespace cuda
} // namespace tensorplay
