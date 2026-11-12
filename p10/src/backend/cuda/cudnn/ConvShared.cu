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
#include "GraphRuntimeScope.h"
#include <vector>
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <cstdint>
#include <optional>
#include <utility>

namespace tensorplay {
namespace cuda {

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

    std::array<int64_t, 4> channels_last_strides(
        int64_t c,
        int64_t h,
        int64_t w) {
        return {c * h * w, 1, w * c, c};
    }

    Tensor empty_conv_output(
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
        const auto strides = channels_last_strides(c, h, w);
        return result.as_strided(
            shape, std::vector<int64_t>(strides.begin(), strides.end()));
    }

    bool conv_channel_major_enabled() {
        // Reduced-precision convolution plans come in two orders: the
        // channel-major engines read their operands as they lie but ask the
        // caller to repack row-major buffers first, while the row-major
        // engines repack internally.  Repacking here also hands the result
        // back channel-major, a layout the caller did not ask for.  A lowered
        // graph reads every result with the strides its plan recorded for the
        // operands it passes, so inside one the operands are taken as they
        // lie: the plan itself passes channel-major operands where that
        // order pays.  Eager calls normalize back to row-major between
        // operators, where the internal repack wins; TP_CONV_CHANNEL_MAJOR=1
        // repacks them anyway.
        if (impl::in_lowered_graph()) return false;
        static const bool forced = [] {
            const char* env = std::getenv("TP_CONV_CHANNEL_MAJOR");
            return env != nullptr && std::string(env) == "1";
        }();
        return forced;
    }

    bool conv_operand_repackable(const Tensor& t, bool is_weight) {
        // Reduced-precision 4-D operands in the row-major order make the
        // engine selection settle on plans that repack the operands around
        // the convolution itself; the channel-major order is read as it
        // lies.  An operand already in that order needs no work, and a
        // single channel or a single spatial position gives both orders the
        // same layout.
        if (!conv_channel_major_enabled()) return false;
        if (t.dim() != 4) return false;
        if (t.dtype() != DType::Float16 && t.dtype() != DType::BFloat16) return false;
        if (is_channels_last_4d(t) || !t.is_contiguous()) return false;
        if (t.size(1) <= 1) return false;
        if (!is_weight && t.size(2) * t.size(3) <= 1) return false;
        return true;
    }

    Tensor conv_to_channel_major(const Tensor& t) {
        // Same shape, channel-major strides: one repacking kernel writes the
        // values in the order the channel-major engines read, and the
        // trailing permutation is a view.
        return t.permute({0, 2, 3, 1}).contiguous().permute({0, 3, 1, 2});
    }

#ifdef USE_CUDNN

// Shared dtype mapping for the descriptor helpers below.  Half/BFloat16 run
// float scalars stay valid for them.




static std::mutex g_conv_desc_cache_mutex;
static std::unordered_map<std::string, std::shared_ptr<CachedTensorDesc>>
    g_tensor_desc_cache;
static std::unordered_map<std::string, std::shared_ptr<CachedFilterDesc>>
    g_filter_desc_cache;
static std::unordered_map<std::string, std::shared_ptr<CachedConvDesc>>
    g_conv_desc_cache;

std::shared_ptr<CachedTensorDesc> get_cached_tensor_desc(const Tensor& t) {
    std::string key = std::to_string(static_cast<int>(t.dtype()));
    for (int64_t dim : {t.size(0), t.size(1), t.size(2), t.size(3)}) {
        key += ":" + std::to_string(dim);
    }
    for (int64_t s : {t.stride(0), t.stride(1), t.stride(2), t.stride(3)}) {
        key += ":" + std::to_string(s);
    }
    std::lock_guard<std::mutex> lock(g_conv_desc_cache_mutex);
    auto it = g_tensor_desc_cache.find(key);
    if (it != g_tensor_desc_cache.end()) return it->second;
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc, to_cudnn_data_type(t.dtype()),
        static_cast<int>(t.size(0)), static_cast<int>(t.size(1)),
        static_cast<int>(t.size(2)), static_cast<int>(t.size(3)),
        static_cast<int>(t.stride(0)), static_cast<int>(t.stride(1)),
        static_cast<int>(t.stride(2)), static_cast<int>(t.stride(3))));
    auto holder = std::make_shared<CachedTensorDesc>();
    holder->desc = desc;
    g_tensor_desc_cache.emplace(key, holder);
    return holder;
}

std::shared_ptr<CachedFilterDesc> get_cached_filter_desc(const Tensor& t) {
    std::string key = std::to_string(static_cast<int>(t.dtype()));
    for (int64_t dim : {t.size(0), t.size(1), t.size(2), t.size(3)}) {
        key += ":" + std::to_string(dim);
    }
    for (int64_t s : {t.stride(0), t.stride(1), t.stride(2), t.stride(3)}) {
        key += ":" + std::to_string(s);
    }
    std::lock_guard<std::mutex> lock(g_conv_desc_cache_mutex);
    auto it = g_filter_desc_cache.find(key);
    if (it != g_filter_desc_cache.end()) return it->second;
    cudnnFilterDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc, to_cudnn_data_type(t.dtype()), CUDNN_TENSOR_NCHW,
        static_cast<int>(t.size(0)), static_cast<int>(t.size(1)),
        static_cast<int>(t.size(2)), static_cast<int>(t.size(3))));
    auto holder = std::make_shared<CachedFilterDesc>();
    holder->desc = desc;
    g_filter_desc_cache.emplace(key, holder);
    return holder;
}

std::shared_ptr<CachedConvDesc> get_cached_conv_desc(
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    int groups, DType dtype) {
    std::string key = std::to_string(static_cast<int>(dtype)) + ":" +
        std::to_string(pad_h) + ":" + std::to_string(pad_w) + ":" +
        std::to_string(str_h) + ":" + std::to_string(str_w) + ":" +
        std::to_string(dil_h) + ":" + std::to_string(dil_w) + ":" +
        std::to_string(groups);
    std::lock_guard<std::mutex> lock(g_conv_desc_cache_mutex);
    auto it = g_conv_desc_cache.find(key);
    if (it != g_conv_desc_cache.end()) return it->second;
    cudnnConvolutionDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CROSS_CORRELATION,
        to_cudnn_compute_type(dtype)));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    // Same math-type policy as the per-call descriptor above.
    cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;
    if (dtype == DType::Float16 ||
        (dtype == DType::Float32 && tensorplay::globalContext().allowTF32CuDNN())) {
        math_type = CUDNN_TENSOR_OP_MATH;
    }
    CUDNN_CHECK(cudnnSetConvolutionMathType(desc, math_type));
    auto holder = std::make_shared<CachedConvDesc>();
    holder->desc = desc;
    g_conv_desc_cache.emplace(key, holder);
    return holder;
}

std::shared_ptr<CachedFilterDesc> get_cached_filter_desc_channel_last(const Tensor& t) {
    std::string key = "cl:";
    key += std::to_string(static_cast<int>(t.dtype()));
    for (int64_t dim : {t.size(0), t.size(1), t.size(2), t.size(3)}) {
        key += ":" + std::to_string(dim);
    }
    for (int64_t s : {t.stride(0), t.stride(1), t.stride(2), t.stride(3)}) {
        key += ":" + std::to_string(s);
    }
    std::lock_guard<std::mutex> lock(g_conv_desc_cache_mutex);
    auto it = g_filter_desc_cache.find(key);
    if (it != g_filter_desc_cache.end()) return it->second;
    cudnnFilterDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc));
    // Channel-last filter: the channel dimension is the fastest-varying one, so
    // a filter laid out K-R-S-C is named by the NHWC form of the same extents.
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc, to_cudnn_data_type(t.dtype()), CUDNN_TENSOR_NHWC,
        static_cast<int>(t.size(0)), static_cast<int>(t.size(1)),
        static_cast<int>(t.size(2)), static_cast<int>(t.size(3))));
    auto holder = std::make_shared<CachedFilterDesc>();
    holder->desc = desc;
    g_filter_desc_cache.emplace(key, holder);
    return holder;
}

bool conv_channel_last_eligible(const Tensor& input, const Tensor& weight,
                                int64_t groups) {
    // The layout of the result follows the operands, as the two engines read
    // different memory: a convolution whose operands arrive row-major keeps
    // the row-major route, while one that arrives channel-last is served by
    // the channel-last engines and gives its result back in that order.
    //
    // The backend name reaches the host translation units as a define; the
    // device ones see only the compatibility headers, which mark themselves.
#if !defined(USE_METAX) && !defined(ENABLE_CUDA_TO_MACA_ADAPTOR)
    (void)input; (void)weight; (void)groups;
    return false;
#else
    if (groups != 1) return false;
    if (input.dim() != 4 || weight.dim() != 4) return false;
    const DType dtype = input.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float16 &&
        dtype != DType::BFloat16) {
        return false;
    }
    if (weight.dtype() != dtype) return false;
    return is_channels_last_4d(input) || is_channels_last_4d(weight);
#endif
}

namespace {

std::mutex g_cl_algo_mutex;
std::unordered_map<std::string, std::pair<int, size_t>> g_cl_algo_cache;

std::string cl_algo_key(const char* kind, const Tensor& a, const Tensor& b,
                        const Tensor& out, const std::vector<int64_t>& stride,
                        const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation) {
    std::string key = kind;
    key += ":" + std::to_string(static_cast<int>(a.dtype()));
    key += ":" + std::to_string(a.device().index());
    for (const Tensor* t : {&a, &b, &out}) {
        for (int64_t d : {t->size(0), t->size(1), t->size(2), t->size(3)}) {
            key += ":" + std::to_string(d);
        }
    }
    for (int64_t v : {stride[0], stride[1], padding[0], padding[1],
                      dilation[0], dilation[1]}) {
        key += ":" + std::to_string(v);
    }
    return key;
}

}  // namespace

Tensor conv_forward_channel_last(const Tensor& input, const Tensor& weight,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation,
                                 bool fused_relu) {
    const int64_t n = input.size(0);
    const int64_t k = weight.size(0);
    const int64_t h = input.size(2);
    const int64_t w = input.size(3);
    const int64_t r = weight.size(2);
    const int64_t s = weight.size(3);
    const int64_t oh = (h + 2 * padding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1;
    const int64_t ow = (w + 2 * padding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1;

    // An operand that already carries the channel-last order is read as it
    // lies; the others are repacked once.  The result keeps that order, so a
    // channel-last model pays for no conversion between layers.
    Tensor x = is_channels_last_4d(input) ? input : conv_to_channel_major(input);
    Tensor flt = is_channels_last_4d(weight) ? weight : conv_to_channel_major(weight);
    Tensor out = empty_conv_output(n, k, oh, ow, input.dtype(), input.device(), true);

    auto x_desc = get_cached_tensor_desc(x);
    auto w_desc = get_cached_filter_desc_channel_last(flt);
    auto y_desc = get_cached_tensor_desc(out);
    auto conv_desc = get_cached_conv_desc(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), 1, input.dtype());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_size = 0;
    {
        const std::string key = cl_algo_key("fwd", input, weight, out, stride, padding, dilation);
        std::lock_guard<std::mutex> lock(g_cl_algo_mutex);
        auto it = g_cl_algo_cache.find(key);
        if (it != g_cl_algo_cache.end()) {
            algo = static_cast<cudnnConvolutionFwdAlgo_t>(it->second.first);
            workspace_size = it->second.second;
        } else {
            cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            int returned = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, *x_desc, *w_desc, *conv_desc, *y_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf_results));
            int chosen = -1;
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { chosen = i; break; }
            }
            if (chosen < 0) {
                TP_THROW(RuntimeError, "cuDNN: no channel-last forward convolution algorithm");
            }
            algo = perf_results[chosen].algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, *x_desc, *w_desc, *conv_desc, *y_desc, algo, &workspace_size));
            g_cl_algo_cache.emplace(key, std::make_pair(static_cast<int>(algo), workspace_size));
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input.device());
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha, *x_desc, x.data_ptr(), *w_desc, flt.data_ptr(), *conv_desc,
        algo, workspace.get(), workspace_size, &beta, *y_desc, out.data_ptr()));

    if (fused_relu) relu_inplace_kernel_cudnn(out);
    return out;
}

Tensor conv_grad_input_channel_last(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& weight,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation) {
    const int64_t n = input.size(0);
    const int64_t c = input.size(1);
    const int64_t h = input.size(2);
    const int64_t w = input.size(3);

    Tensor dy = is_channels_last_4d(grad_output) ? grad_output
                                                  : conv_to_channel_major(grad_output);
    Tensor flt = is_channels_last_4d(weight) ? weight : conv_to_channel_major(weight);
    Tensor grad_input = empty_conv_output(n, c, h, w, input.dtype(), input.device(), true);

    auto w_desc = get_cached_filter_desc_channel_last(flt);
    auto dy_desc = get_cached_tensor_desc(dy);
    auto dx_desc = get_cached_tensor_desc(grad_input);
    auto conv_desc = get_cached_conv_desc(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), 1, input.dtype());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t workspace_size = 0;
    {
        const std::string key = cl_algo_key("dgrad", grad_output, weight, grad_input,
                                            stride, padding, dilation);
        std::lock_guard<std::mutex> lock(g_cl_algo_mutex);
        auto it = g_cl_algo_cache.find(key);
        if (it != g_cl_algo_cache.end()) {
            algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(it->second.first);
            workspace_size = it->second.second;
        } else {
            cudnnConvolutionBwdDataAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
            int returned = 0;
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, *w_desc, *dy_desc, *conv_desc, *dx_desc,
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returned, perf_results));
            int chosen = -1;
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { chosen = i; break; }
            }
            if (chosen < 0) {
                TP_THROW(RuntimeError, "cuDNN: no channel-last backward-data algorithm");
            }
            algo = perf_results[chosen].algo;
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle, *w_desc, *dy_desc, *conv_desc, *dx_desc, algo, &workspace_size));
            g_cl_algo_cache.emplace(key, std::make_pair(static_cast<int>(algo), workspace_size));
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, input.device());
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle, &alpha, *w_desc, flt.data_ptr(), *dy_desc, dy.data_ptr(), *conv_desc,
        algo, workspace.get(), workspace_size, &beta, *dx_desc, grad_input.data_ptr()));
    return grad_input;
}

Tensor conv_grad_weight_channel_last(const Tensor& grad_output, const Tensor& input,
                                     const Tensor& weight,
                                     const std::vector<int64_t>& stride,
                                     const std::vector<int64_t>& padding,
                                     const std::vector<int64_t>& dilation) {
    const int64_t k = weight.size(0);
    const int64_t c = weight.size(1);
    const int64_t r = weight.size(2);
    const int64_t s = weight.size(3);

    Tensor dy = is_channels_last_4d(grad_output) ? grad_output
                                                  : conv_to_channel_major(grad_output);
    Tensor x = is_channels_last_4d(input) ? input : conv_to_channel_major(input);
    Tensor grad_weight = empty_conv_output(k, c, r, s, weight.dtype(), weight.device(), true);

    auto x_desc = get_cached_tensor_desc(x);
    auto dy_desc = get_cached_tensor_desc(dy);
    auto dw_desc = get_cached_filter_desc_channel_last(grad_weight);
    auto conv_desc = get_cached_conv_desc(
        static_cast<int>(padding[0]), static_cast<int>(padding[1]),
        static_cast<int>(stride[0]), static_cast<int>(stride[1]),
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), 1, weight.dtype());

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();
    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    size_t workspace_size = 0;
    {
        const std::string key = cl_algo_key("wgrad", grad_output, input, grad_weight,
                                            stride, padding, dilation);
        std::lock_guard<std::mutex> lock(g_cl_algo_mutex);
        auto it = g_cl_algo_cache.find(key);
        if (it != g_cl_algo_cache.end()) {
            algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(it->second.first);
            workspace_size = it->second.second;
        } else {
            cudnnConvolutionBwdFilterAlgoPerf_t perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
            int returned = 0;
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle, *x_desc, *dy_desc, *conv_desc, *dw_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &returned, perf_results));
            int chosen = -1;
            for (int i = 0; i < returned; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS) { chosen = i; break; }
            }
            if (chosen < 0) {
                TP_THROW(RuntimeError, "cuDNN: no channel-last backward-filter algorithm");
            }
            algo = perf_results[chosen].algo;
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, *x_desc, *dy_desc, *conv_desc, *dw_desc, algo, &workspace_size));
            g_cl_algo_cache.emplace(key, std::make_pair(static_cast<int>(algo), workspace_size));
        }
    }

    auto workspace = getAllocator(DeviceType::CUDA)->allocate(
        workspace_size ? workspace_size : 1, weight.device());
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        handle, &alpha, *x_desc, x.data_ptr(), *dy_desc, dy.data_ptr(), *conv_desc,
        algo, workspace.get(), workspace_size, &beta, *dw_desc, grad_weight.data_ptr()));
    // The gradient keeps the order the engine wrote it in, which is the order
    // the parameter it belongs to already carries.
    return grad_weight;
}



#ifndef TP_CONV_CUDA_CHECK
#define TP_CONV_CUDA_CHECK(condition) \
    do { \
        cudaError_t tp_conv_err__ = (condition); \
        if (tp_conv_err__ != cudaSuccess) { \
            TP_THROW(RuntimeError, std::string("CUDA error: ") + \
                                   cudaGetErrorString(tp_conv_err__)); \
        } \
    } while (0)
#endif


bool conv_autotune_enabled() {
    return tensorplay::globalContext().cudnnBenchmark() &&
           !tensorplay::globalContext().deterministicAlgorithms();
}



#endif
}  // namespace cuda
}  // namespace tensorplay
