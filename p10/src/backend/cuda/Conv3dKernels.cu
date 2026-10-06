#include "Tensor.h"
#include "Dispatcher.h"
#include "Convolution.h"
#include "Context.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "CUDNNUtils.h"
#include "Allocator.h"

#include <cuda_runtime.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#ifdef USE_CUDNN

inline cudnnDataType_t to_cudnn_data_type(DType d) {
    if (d == DType::Float32) return CUDNN_DATA_FLOAT;
    if (d == DType::Float64) return CUDNN_DATA_DOUBLE;
    if (d == DType::Float16) return CUDNN_DATA_HALF;
    if (d == DType::BFloat16) return CUDNN_DATA_BFLOAT16;
    TP_THROW(NotImplementedError, "cuDNN: only float/double/half/bfloat16 supported");
}

inline cudnnDataType_t to_cudnn_compute_type(DType d) {
    return d == DType::Float64 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
}

inline std::vector<int64_t> expand_param_if_needed(const std::vector<int64_t>& list,
                                                   int64_t n, int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (list.size() != n) TP_THROW(ValueError, "Parameter size mismatch");
    return list;
}

struct ConvBwdAlgo {
    int algorithm;
    size_t workspace_size;
};

static std::unordered_map<std::string, ConvBwdAlgo> g_conv3d_algo_cache;
static std::mutex g_conv3d_cache_mutex;
static constexpr size_t kConvAutotuneWorkspaceCap = 512ULL * 1024 * 1024;

inline bool conv_autotune_enabled() {
    return tensorplay::globalContext().cudnnBenchmark() &&
           !tensorplay::globalContext().deterministicAlgorithms();
}

inline void key_append(std::string& key, const std::vector<int64_t>& v) {
    for (int64_t d : v) key += ":" + std::to_string(d);
}

template <class XDesc, class WDesc, class CDesc, class YDesc>
ConvBwdAlgo autotune_conv_fwd(
    cudnnHandle_t handle,
    const XDesc& x_desc, const void* x_ptr,
    const WDesc& w_desc, const void* w_ptr,
    const CDesc& conv_desc,
    const YDesc& y_desc, void* y_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionFwdAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle, x_desc, x_ptr, w_desc, w_ptr, conv_desc, y_desc, y_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no forward convolution algorithm");
}

template <class WDesc, class DYDesc, class CDesc, class DXDesc>
ConvBwdAlgo autotune_conv_bwd_data(
    cudnnHandle_t handle,
    const WDesc& w_desc, const void* w_ptr,
    const DYDesc& dy_desc, const void* dy_ptr,
    const CDesc& conv_desc,
    const DXDesc& dx_desc, void* dx_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionBwdDataAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle, w_desc, w_ptr, dy_desc, dy_ptr, conv_desc, dx_desc, dx_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no backward-data convolution algorithm");
}

template <class XDesc, class DYDesc, class CDesc, class DWDesc>
ConvBwdAlgo autotune_conv_bwd_filter(
    cudnnHandle_t handle,
    const XDesc& x_desc, const void* x_ptr,
    const DYDesc& dy_desc, const void* dy_ptr,
    const CDesc& conv_desc,
    const DWDesc& dw_desc, void* dw_ptr,
    const Device& device) {
    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        kConvAutotuneWorkspaceCap, device);
    cudnnConvolutionBwdFilterAlgoPerf_t perfs[16];
    int returned = 0;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle, x_desc, x_ptr, dy_desc, dy_ptr, conv_desc, dw_desc, dw_ptr,
        16, &returned, perfs, workspace.get(), kConvAutotuneWorkspaceCap));
    for (int i = 0; i < returned; ++i) {
        if (perfs[i].status == CUDNN_STATUS_SUCCESS) {
            return ConvBwdAlgo{static_cast<int>(perfs[i].algo), perfs[i].memory};
        }
    }
    TP_THROW(RuntimeError, "cuDNN: autotune found no backward-filter convolution algorithm");
}

template <class Select>
ConvBwdAlgo cached_conv_algo(const std::string& key, Select&& select) {
    {
        std::lock_guard<std::mutex> lock(g_conv3d_cache_mutex);
        auto it = g_conv3d_algo_cache.find(key);
        if (it != g_conv3d_algo_cache.end()) return it->second;
    }
    ConvBwdAlgo entry = select();
    {
        std::lock_guard<std::mutex> lock(g_conv3d_cache_mutex);
        g_conv3d_algo_cache.emplace(key, entry);
    }
    return entry;
}

struct TensorDescNd {
    cudnnTensorDescriptor_t desc;
    TensorDescNd() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDescNd() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }

    void set(const std::vector<int64_t>& sizes, DType dtype) {
        int nbDims = static_cast<int>(sizes.size());
        int dims[8], strides[8];
        int64_t stride = 1;
        for (int i = nbDims - 1; i >= 0; --i) {
            dims[i] = static_cast<int>(sizes[i]);
            strides[i] = static_cast<int>(stride);
            stride *= sizes[i];
        }
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, to_cudnn_data_type(dtype), nbDims,
                                               dims, strides));
    }
    void set(const Tensor& t) { set(t.shape(), t.dtype()); }
};

struct FilterDescNd {
    cudnnFilterDescriptor_t desc;
    FilterDescNd() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDescNd() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }

    void set(const Tensor& t) {
        int nbDims = static_cast<int>(t.dim());
        int dims[8];
        for (int i = 0; i < nbDims; ++i) dims[i] = static_cast<int>(t.size(i));
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, to_cudnn_data_type(t.dtype()),
                                               CUDNN_TENSOR_NCHW, nbDims, dims));
    }
};

struct ConvDescNd {
    cudnnConvolutionDescriptor_t desc;
    ConvDescNd() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvDescNd() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }

    void set(const std::vector<int64_t>& pads, const std::vector<int64_t>& strides,
             const std::vector<int64_t>& dilations, int64_t groups, DType dtype) {
        int nbDims = static_cast<int>(pads.size());
        int p[3], s[3], d[3];
        for (int i = 0; i < nbDims; ++i) {
            p[i] = static_cast<int>(pads[i]);
            s[i] = static_cast<int>(strides[i]);
            d[i] = static_cast<int>(dilations[i]);
        }
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(desc, nbDims, p, s, d,
                                                    CUDNN_CROSS_CORRELATION,
                                                    to_cudnn_compute_type(dtype)));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, static_cast<int>(groups)));
        cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
        if (dtype == DType::Float16 ||
            (dtype == DType::Float32 && tensorplay::globalContext().allowTF32CuDNN())) {
            math_type = CUDNN_TENSOR_OP_MATH;
        }
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc, math_type));
    }
};

inline void conv3d_add_bias(cudnnHandle_t handle, const Tensor& out, const Tensor& bias) {
    if (!bias.defined() || bias.numel() == 0) return;
    TensorDescNd b_desc;
    b_desc.set(std::vector<int64_t>{1, bias.size(0), 1, 1, 1}, bias.dtype());
    TensorDescNd y_desc;
    y_desc.set(out.shape(), out.dtype());
    float alpha = 1.0f, beta = 1.0f;
    double alpha_d = 1.0, beta_d = 1.0;
    void* alpha_p = &alpha;
    void* beta_p = &beta;
    if (out.dtype() == DType::Float64) {
        alpha_p = &alpha_d;
        beta_p = &beta_d;
    }
    CUDNN_CHECK(cudnnAddTensor(handle, alpha_p, b_desc, bias.data_ptr(),
                               beta_p, y_desc, out.data_ptr()));
}

inline void* conv_alpha_ptr(DType dtype, float& alpha, double& alpha_d) {
    if (dtype == DType::Float64) return &alpha_d;
    return &alpha;
}

#endif

}

Tensor conv3d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg,
                   const std::vector<int64_t>& dilation_arg, int64_t groups) {
    convolution::check_conv_shapes(input, weight, bias, groups, false);
    convolution::check_conv_geometry(input, weight, stride_arg, padding_arg,
                                     dilation_arg, "conv3d");
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t D_in = input_c.size(2), H_in = input_c.size(3), W_in = input_c.size(4);
    const int64_t kD = weight_c.size(2), kH = weight_c.size(3), kW = weight_c.size(4);
    const int64_t D_out = (D_in + 2 * padding[0] - dilation[0] * (kD - 1) - 1) / stride[0] + 1;
    const int64_t H_out = (H_in + 2 * padding[1] - dilation[1] * (kH - 1) - 1) / stride[1] + 1;
    const int64_t W_out = (W_in + 2 * padding[2] - dilation[2] * (kW - 1) - 1) / stride[2] + 1;
    if (D_out <= 0 || H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "conv3d: Calculated output size is too small");

    Tensor out = Tensor::empty({input_c.size(0), weight_c.size(0), D_out, H_out, W_out},
                               input_c.dtype(), input_c.device());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDescNd x_desc; x_desc.set(input_c);
    FilterDescNd w_desc; w_desc.set(weight_c);
    TensorDescNd y_desc; y_desc.set(out);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    std::string algo_key = "c3fwd:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_fwd(handle, x_desc, input_c.data_ptr(),
                                     w_desc, weight_c.data_ptr(), conv_desc,
                                     y_desc, out.data_ptr(), input_c.device());
        }
        cudnnConvolutionFwdAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, x_desc, w_desc, conv_desc, y_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no forward convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, x_desc, w_desc, conv_desc, y_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionForward(handle, alpha_p, x_desc, input_c.data_ptr(),
                                        w_desc, weight_c.data_ptr(), conv_desc,
                                        static_cast<cudnnConvolutionFwdAlgo_t>(algo_entry.algorithm),
                                        workspace.get(), algo_entry.workspace_size, beta_p,
                                        y_desc, out.data_ptr()));
    conv3d_add_bias(handle, out, bias);
    return out;
#else
    TP_THROW(NotImplementedError, "conv3d_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const std::vector<int64_t>& stride_arg,
                              const std::vector<int64_t>& padding_arg,
                              const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDescNd dy_desc; dy_desc.set(grad_output_c);
    FilterDescNd w_desc; w_desc.set(weight_c);
    TensorDescNd dx_desc; dx_desc.set(input_c);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    Tensor grad_input = Tensor::empty_like(input_c, DType::Undefined, input_c.device());

    std::string algo_key = "c3gi:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, grad_output_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_bwd_data(handle, w_desc, weight_c.data_ptr(),
                                           dy_desc, grad_output_c.data_ptr(), conv_desc,
                                           dx_desc, grad_input.data_ptr(), input_c.device());
        }
        cudnnConvolutionBwdDataAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle, w_desc, dy_desc, conv_desc, dx_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no backward-data convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, w_desc, dy_desc, conv_desc, dx_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, alpha_p, w_desc, weight_c.data_ptr(),
                                             dy_desc, grad_output_c.data_ptr(), conv_desc,
                                             static_cast<cudnnConvolutionBwdDataAlgo_t>(algo_entry.algorithm),
                                             workspace.get(), algo_entry.workspace_size,
                                             beta_p, dx_desc, grad_input.data_ptr()));
    return grad_input;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_input_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                               const std::vector<int64_t>& stride_arg,
                               const std::vector<int64_t>& padding_arg,
                               const std::vector<int64_t>& dilation_arg, int64_t groups) {
#ifdef USE_CUDNN
    auto stride = expand_param_if_needed(stride_arg, 3, 1);
    auto padding = expand_param_if_needed(padding_arg, 3, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 3, 1);

    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDescNd x_desc; x_desc.set(input_c);
    TensorDescNd dy_desc; dy_desc.set(grad_output_c);
    FilterDescNd dw_desc; dw_desc.set(weight_c);
    ConvDescNd conv_desc;
    conv_desc.set(padding, stride, dilation, groups, input_c.dtype());

    Tensor grad_weight = Tensor::empty_like(weight_c, DType::Undefined, weight_c.device());

    std::string algo_key = "c3gw:" + std::to_string(static_cast<int>(input_c.dtype())) +
                           ":" + std::to_string(static_cast<int>(input_c.device().index()));
    key_append(algo_key, input_c.shape());
    key_append(algo_key, weight_c.shape());
    key_append(algo_key, grad_output_c.shape());
    key_append(algo_key, padding);
    key_append(algo_key, stride);
    key_append(algo_key, dilation);
    algo_key += ":" + std::to_string(groups);

    ConvBwdAlgo algo_entry = cached_conv_algo(algo_key, [&]() -> ConvBwdAlgo {
        if (conv_autotune_enabled()) {
            return autotune_conv_bwd_filter(handle, x_desc, input_c.data_ptr(),
                                            dy_desc, grad_output_c.data_ptr(), conv_desc,
                                            dw_desc, grad_weight.data_ptr(), input_c.device());
        }
        cudnnConvolutionBwdFilterAlgoPerf_t perf;
        int returned = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle, x_desc, dy_desc, conv_desc, dw_desc, 1, &returned, &perf));
        if (returned == 0) TP_THROW(RuntimeError, "cuDNN: no backward-filter convolution algorithm");
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, x_desc, dy_desc, conv_desc, dw_desc, perf.algo, &workspace_size));
        return ConvBwdAlgo{static_cast<int>(perf.algo), workspace_size};
    });

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        algo_entry.workspace_size ? algo_entry.workspace_size : 1, input_c.device());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(input_c.dtype(), alpha, alpha_d);
    void* beta_p = input_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                     : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, alpha_p, x_desc, input_c.data_ptr(),
                                               dy_desc, grad_output_c.data_ptr(), conv_desc,
                                               static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo_entry.algorithm),
                                               workspace.get(), algo_entry.workspace_size,
                                               beta_p, dw_desc, grad_weight.data_ptr()));
    return grad_weight;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_weight_cuda requires cuDNN");
#endif
}

Tensor conv3d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
#ifdef USE_CUDNN
    Tensor grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    Tensor grad_bias = Tensor::empty({grad_output_c.size(1)}, grad_output_c.dtype(),
                                     grad_output_c.device());

    TensorDescNd dy_desc;
    dy_desc.set(grad_output_c);
    TensorDescNd db_desc;
    db_desc.set(std::vector<int64_t>{1, grad_bias.size(0), 1, 1, 1}, grad_bias.dtype());

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = conv_alpha_ptr(grad_output_c.dtype(), alpha, alpha_d);
    void* beta_p = grad_output_c.dtype() == DType::Float64 ? static_cast<void*>(&beta_d)
                                                           : static_cast<void*>(&beta);
    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, alpha_p, dy_desc,
                                             grad_output_c.data_ptr(), beta_p, db_desc,
                                             grad_bias.data_ptr()));
    return grad_bias;
#else
    TP_THROW(NotImplementedError, "conv3d_grad_bias_cuda requires cuDNN");
#endif
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Conv3dKernels) {
    m.impl("conv3d", conv3d_cuda);
    m.impl("conv3d_grad_input", conv3d_grad_input_cuda);
    m.impl("conv3d_grad_weight", conv3d_grad_weight_cuda);
    m.impl("conv3d_grad_bias", conv3d_grad_bias_cuda);
}

}
}
