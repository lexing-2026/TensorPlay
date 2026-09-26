#include "cudnn/ConvShared.h"
#include "Tensor.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Context.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "CUDNNUtils.h"
#include "CudaGemm.h"
#include "Allocator.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <cstdint>
#include <optional>

namespace tensorplay {
namespace cuda {

// ---------------------------------------------------------------------------
// Legacy cuDNN descriptor wrappers and forward algorithm cache
// ---------------------------------------------------------------------------
namespace {

// RAII Wrappers for cuDNN descriptors
struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    TensorDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }

    void set(const Tensor& t) {
        cudnnDataType_t dtype = to_cudnn_data_type(t.dtype());

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
        cudnnDataType_t dtype = to_cudnn_data_type(t.dtype());

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
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CROSS_CORRELATION, to_cudnn_compute_type(dtype)));
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
        // Half convolutions map to tensor-op kernels regardless of any
        // global TF32 switch; float32 follows the context's TF32 switch.
        cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
        if (dtype == DType::Float16 ||
            (dtype == DType::Float32 && tensorplay::globalContext().allowTF32CuDNN())) {
            math_type = CUDNN_TENSOR_OP_MATH;
        }
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc, math_type));
    }
};

struct ActivationDesc {
    cudnnActivationDescriptor_t desc;
    ActivationDesc() { CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc)); }
    ~ActivationDesc() { cudnnDestroyActivationDescriptor(desc); }
    operator cudnnActivationDescriptor_t() const { return desc; }

    void set_relu() {
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    }
};

struct ConvFwdAlgo {
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_size;
};

// ---------------------------------------------------------------------------
// Cached legacy cuDNN descriptors
//
// The backward convolution paths configure the same 4-D NCHW tensor/filter
// and 2-D convolution descriptors on every call; creating and setting them
// costs a driver round-trip each time while the shapes are stable across
// training iterations.  Each cache entry is keyed by the exact tuple the
// setters read, and entries are never mutated after construction, so cached
// handles can be shared across calls without synchronization beyond the
// lookup lock.
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, ConvFwdAlgo> g_conv_fwd_algo_cache;
static std::mutex g_conv_fwd_cache_mutex;

std::string make_conv_fwd_cache_key(
    const Tensor& input,
    const Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    bool fused_relu) {
    std::string key = fused_relu ? "relu:" : "conv:";
    key += std::to_string(static_cast<int>(input.dtype()));
    key += ":" + std::to_string(input.device().index());
    for (int64_t dim : {input.size(0), input.size(1), input.size(2), input.size(3),
                        weight.size(0), weight.size(1), weight.size(2), weight.size(3),
                        stride[0], stride[1], padding[0], padding[1],
                        dilation[0], dilation[1], groups}) {
        key += ":" + std::to_string(dim);
    }
    return key;
}

// Backward convolution algorithm selection is shape- and dtype-dependent,
// decision; caching it here removes a cuDNN v7 heuristic query from every
}  // namespace

template <class XDesc, class WDesc, class CDesc, class YDesc>
static ConvBwdAlgo autotune_conv_fwd(
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

#ifdef USE_CUDNN
Tensor conv2d_relu_cudnn(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg,
    const std::vector<int64_t>& dilation_arg,
    int64_t groups) {
    if (!bias.defined() ||
        (input.dtype() != DType::Float32 && input.dtype() != DType::Float64)) {
        return Tensor();
    }

    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();

    const int64_t n = input_c.size(0);
    const int64_t k = weight_c.size(0);
    const int64_t h = input_c.size(2);
    const int64_t w = input_c.size(3);
    const int64_t r = weight_c.size(2);
    const int64_t s = weight_c.size(3);
    const int64_t oh = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1;
    const int64_t ow = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1;

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input_c);
    FilterDesc w_desc; w_desc.set(weight_c);
    Tensor out = Tensor::empty({n, k, oh, ow}, input_c.dtype(), input_c.device());
    TensorDesc y_desc; y_desc.set(out);
    Tensor bias_4d = bias_c.reshape({1, k, 1, 1});
    TensorDesc bias_desc; bias_desc.set(bias_4d);
    ConvDesc conv_desc;
    conv_desc.set(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]),
        static_cast<int>(groups), input_c.dtype());
    ActivationDesc activation_desc;
    activation_desc.set_relu();

    std::string cache_key = make_conv_fwd_cache_key(
        input_c, weight_c, stride, padding, dilation, groups, true);
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_size;
    {
        std::lock_guard<std::mutex> lock(g_conv_fwd_cache_mutex);
        auto it = g_conv_fwd_algo_cache.find(cache_key);
        if (it == g_conv_fwd_algo_cache.end()) {
            cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            int returned_algo_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, x_desc, w_desc, conv_desc, y_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count,
                perf_results));
            // The heuristic order is the preference order; entries without
            // a successful status are not executable on this hardware, so
            // skip them instead of failing at run time.
            int chosen = -1;
            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS) {
                    chosen = i;
                    break;
                }
            }
            if (chosen < 0) {
                TP_THROW(RuntimeError, "cuDNN: no fused forward convolution algorithm");
            }
            algorithm = perf_results[chosen].algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, x_desc, w_desc, conv_desc, y_desc,
                algorithm, &workspace_size));
            g_conv_fwd_algo_cache.emplace(
                cache_key, ConvFwdAlgo{algorithm, workspace_size});
        } else {
            algorithm = it->second.algorithm;
            workspace_size = it->second.workspace_size;
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input_c.device());
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = &alpha;
    void* beta_p = &beta;
    if (input_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d;
        beta_p = &beta_d;
    }

    // alpha2 is zero, so z is not read; using y as its descriptor/pointer
    // keeps the legacy cuDNN API valid across cuDNN 8 and 9.
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        handle, alpha_p, x_desc, input_c.data_ptr(), w_desc, weight_c.data_ptr(),
        conv_desc, algorithm, workspace.get(), workspace_size, beta_p,
        y_desc, out.data_ptr(), bias_desc, bias_4d.data_ptr(), activation_desc,
        y_desc, out.data_ptr()));
    return out;
}
#endif

// The graph API is header-only and is not present in every cuDNN runtime
// image.  Keep the legacy cuDNN forward path available in that case so a
// missing optional frontend header does not prevent unrelated targets (in
// particular optimizer kernels) from being rebuilt.
#ifdef USE_CUDNN
Tensor conv2d_cudnn_legacy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const std::vector<int64_t>& stride_arg,
    const std::vector<int64_t>& padding_arg,
    const std::vector<int64_t>& dilation_arg,
    int64_t groups,
    bool fused_relu) {
    auto stride = expand_param_if_needed(stride_arg, 2, 1);
    auto padding = expand_param_if_needed(padding_arg, 2, 0);
    auto dilation = expand_param_if_needed(dilation_arg, 2, 1);
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    const int64_t n = input_c.size(0);
    const int64_t k = weight_c.size(0);
    const int64_t h = input_c.size(2);
    const int64_t w = input_c.size(3);
    const int64_t r = weight_c.size(2);
    const int64_t s = weight_c.size(3);
    const int64_t oh = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) /
        stride[0] + 1;
    const int64_t ow = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) /
        stride[1] + 1;
    if (oh <= 0 || ow <= 0) {
        TP_THROW(RuntimeError, "conv2d: calculated output size is too small");
    }

    if (fused_relu && bias.defined() &&
        (input_c.dtype() == DType::Float32 ||
         input_c.dtype() == DType::Float64)) {
        return conv2d_relu_cudnn(
            input_c, weight_c, bias, stride, padding, dilation, groups);
    }

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    TensorDesc x_desc; x_desc.set(input_c);
    FilterDesc w_desc; w_desc.set(weight_c);
    Tensor out = Tensor::empty({n, k, oh, ow}, input_c.dtype(), input_c.device());
    TensorDesc y_desc; y_desc.set(out);
    ConvDesc conv_desc;
    conv_desc.set(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]),
        static_cast<int>(groups), input_c.dtype());

    const std::string cache_key = make_conv_fwd_cache_key(
        input_c, weight_c, stride, padding, dilation, groups, false);
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_size;
    {
        std::lock_guard<std::mutex> lock(g_conv_fwd_cache_mutex);
        auto it = g_conv_fwd_algo_cache.find(cache_key);
        if (it == g_conv_fwd_algo_cache.end()) {
            cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            int returned_algo_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, x_desc, w_desc, conv_desc, y_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count,
                perf_results));
            // Same skip-unexecutable rule as the fused path.
            int chosen = -1;
            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS) {
                    chosen = i;
                    break;
                }
            }
            if (chosen < 0) {
                TP_THROW(RuntimeError, "cuDNN: no forward convolution algorithm");
            }
            algorithm = perf_results[chosen].algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, x_desc, w_desc, conv_desc, y_desc,
                algorithm, &workspace_size));
            g_conv_fwd_algo_cache.emplace(
                cache_key, ConvFwdAlgo{algorithm, workspace_size});
        } else {
            algorithm = it->second.algorithm;
            workspace_size = it->second.workspace_size;
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input_c.device());
    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_p = &alpha;
    void* beta_p = &beta;
    if (input_c.dtype() == DType::Float64) {
        alpha_p = &alpha_d;
        beta_p = &beta_d;
    }
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, alpha_p, x_desc, input_c.data_ptr(), w_desc,
        weight_c.data_ptr(), conv_desc, algorithm, workspace.get(),
        workspace_size, beta_p, y_desc, out.data_ptr()));

    if (bias.defined() && bias.numel() != 0) {
        Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();
        Tensor bias_4d = bias_c.reshape({1, k, 1, 1});
        TensorDesc bias_desc; bias_desc.set(bias_4d);
        float beta_one = 1.0f;
        double beta_one_d = 1.0;
        void* beta_one_p = &beta_one;
        if (input_c.dtype() == DType::Float64) beta_one_p = &beta_one_d;
        CUDNN_CHECK(cudnnAddTensor(
            handle, alpha_p, bias_desc, bias_4d.data_ptr(), beta_one_p,
            y_desc, out.data_ptr()));
    }
    if (fused_relu) relu_inplace_kernel_cudnn(out);
    return out;
}
#endif
}  // namespace cuda
}  // namespace tensorplay
