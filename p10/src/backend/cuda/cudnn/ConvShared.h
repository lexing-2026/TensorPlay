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
#include <string>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <cstdint>
#include <array>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

bool add_channel_broadcast_inplace_cuda(Tensor& self, const Tensor& other);

bool is_channels_last_4d(const Tensor& tensor);

// Whether a reduced-precision 4-D operand gains from being repacked into the
// channel-major order before a convolution reads it.  ``is_weight`` lifts the
// single-spatial-position exclusion, which is about activations only.
bool conv_operand_repackable(const Tensor& t, bool is_weight);

// Same shape and values with channel-major strides, by way of one repacking
// kernel.
Tensor conv_to_channel_major(const Tensor& t);

std::array<int64_t, 4> channels_last_strides(int64_t c, int64_t h, int64_t w);

Tensor empty_conv_output(int64_t n, int64_t c, int64_t h, int64_t w, DType dtype,
                         const Device& device, bool channels_last);

std::vector<int64_t> expand_param_if_needed(const std::vector<int64_t>& list,
                                            int64_t n, int64_t default_val);

#ifdef USE_CUDNN
Tensor& relu_inplace_kernel_cudnn(Tensor& self);

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

static constexpr size_t kConvAutotuneWorkspaceCap = 512ULL * 1024 * 1024;

struct CachedTensorDesc {
    cudnnTensorDescriptor_t desc = nullptr;
    ~CachedTensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }
};

struct CachedFilterDesc {
    cudnnFilterDescriptor_t desc = nullptr;
    ~CachedFilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }
};

struct CachedConvDesc {
    cudnnConvolutionDescriptor_t desc = nullptr;
    ~CachedConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }
};


struct ConvBwdKey {
    int kind;  // 0 = grad-input, 1 = grad-weight
    int dtype;
    int device;
    std::array<int64_t, 4> input_shape;
    std::array<int64_t, 4> weight_shape;
    std::array<int64_t, 4> grad_shape;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> padding;
    std::array<int64_t, 2> dilation;
    int64_t groups;

    bool operator==(const ConvBwdKey& other) const {
        return kind == other.kind && dtype == other.dtype && device == other.device &&
               input_shape == other.input_shape && weight_shape == other.weight_shape &&
               grad_shape == other.grad_shape && stride == other.stride &&
               padding == other.padding && dilation == other.dilation &&
               groups == other.groups;
    }
};

struct ConvBwdKeyHash {
    size_t operator()(const ConvBwdKey& key) const {
        size_t hash = 0;
        auto combine = [&hash](int64_t value) {
            hash = hash * 1000003U ^ std::hash<int64_t>{}(value);
        };
        combine(key.kind);
        combine(key.dtype);
        combine(key.device);
        for (auto value : key.input_shape) combine(value);
        for (auto value : key.weight_shape) combine(value);
        for (auto value : key.grad_shape) combine(value);
        for (auto value : key.stride) combine(value);
        for (auto value : key.padding) combine(value);
        for (auto value : key.dilation) combine(value);
        combine(key.groups);
        return hash;
    }
};

struct ConvBwdAlgo {
    int algorithm;
    size_t workspace_size;
};

// Backward algorithm cache shared by both descriptor styles.
inline std::unordered_map<ConvBwdKey, ConvBwdAlgo, ConvBwdKeyHash> g_conv_bwd_algo_cache;
inline std::mutex g_conv_bwd_cache_mutex;

inline ConvBwdKey make_conv_bwd_key(
    int kind,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& grad_output,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
    return ConvBwdKey{
        kind,
        static_cast<int>(input.dtype()),
        static_cast<int>(input.device().index()),
        {input.size(0), input.size(1), input.size(2), input.size(3)},
        {weight.size(0), weight.size(1), weight.size(2), weight.size(3)},
        {grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3)},
        {stride[0], stride[1]},
        {padding[0], padding[1]},
        {dilation[0], dilation[1]},
        groups};
}

template <class WDesc, class DYDesc, class CDesc, class DXDesc>
inline ConvBwdAlgo autotune_conv_bwd_data(
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
inline ConvBwdAlgo autotune_conv_bwd_filter(
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


std::shared_ptr<CachedTensorDesc> get_cached_tensor_desc(const Tensor& t);
std::shared_ptr<CachedFilterDesc> get_cached_filter_desc(const Tensor& t);
std::shared_ptr<CachedConvDesc> get_cached_conv_desc(int pad_h, int pad_w, int str_h,
                                                      int str_w, int dil_h, int dil_w,
                                                      int groups, DType dtype);

bool conv_autotune_enabled();

Tensor conv2d_relu_cudnn(const Tensor& input, const Tensor& weight,
                         const Tensor& bias, const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& dilation, int64_t groups);

// Undefined tensor when the case is not representable on the graph path.
Tensor conv_transpose2d_cudnn_v8(const Tensor& input, const Tensor& weight,
                                 const Tensor& bias,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& output_padding,
                                 int64_t groups,
                                 const std::vector<int64_t>& dilation);

// Backward-data convolution on the graph path.  Undefined tensor when the
// case is not representable there (dtype, grouping, or a version that cannot
// build the operation), in which case the caller falls back to the legacy
// descriptor path.
Tensor conv2d_grad_input_cudnn_v8(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& weight,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  int64_t groups,
                                  const std::vector<int64_t>& dilation);

// Backward-filter convolution on the graph path.  Undefined tensor when the
// case is not representable there, in which case the caller falls back to the
// legacy descriptor path.
Tensor conv2d_grad_weight_cudnn_v8(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight,
                                   const std::vector<int64_t>& stride,
                                   const std::vector<int64_t>& padding,
                                   int64_t groups,
                                   const std::vector<int64_t>& dilation);

Tensor conv2d_cudnn_legacy(const Tensor& input, const Tensor& weight,
                            const Tensor& bias, const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, int64_t groups,
                            bool fused_relu);
#endif

Tensor conv2d_slow_fp64(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_slow_fp64_grad_input(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight, const std::vector<int64_t>& stride,
                                   const std::vector<int64_t>& padding,
                                   const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_slow_fp64_grad_weight(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& filter, const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_slow_fp64_grad_bias(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& weight, const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, int64_t groups);

Tensor conv3d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv3d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                              const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv3d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                               const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv3d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                             const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups);

}  // namespace cuda
}  // namespace tensorplay
