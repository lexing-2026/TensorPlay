#pragma once

#include "Tensor.h"
#include "Exception.h"
#include "MemoryFormat.h"
#include <vector>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

#define CUDNN_CHECK(condition) \
  do { \
    const cudnnStatus_t _tp_cudnn_status = (condition); \
    if (_tp_cudnn_status != CUDNN_STATUS_SUCCESS) { \
      std::string err_msg = std::string("cuDNN Error: ") + cudnnGetErrorString(_tp_cudnn_status) + " at " + __FILE__ + ":" + std::to_string(__LINE__) + " in " + #condition; \
      std::cout << "CUDNN FAILURE: " << err_msg << std::endl; \
      TP_THROW(RuntimeError, err_msg); \
    } \
  } while (0)

inline cudnnTensorDescriptor_t createTensorDescriptor(const Tensor& t, bool pad_to_4d = true) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    // For 1D/generic tensors, we can treat them as 4D NCHW with some dims as 1
    // cudnnSetTensorNdDescriptor requires nbDims >= 4 in many versions (or >=3)
    // We pad to 4 dims to be safe.

    int dim = t.dim();
    std::vector<int> dimA;
    std::vector<int> strideA;

    // Copy existing dims
    for (int i = 0; i < dim; ++i) {
        dimA.push_back(static_cast<int>(t.shape()[i]));
        strideA.push_back(static_cast<int>(t.strides()[i]));
    }

    // Pad to 4 dims (prepend 1s)
    if (pad_to_4d) {
        while (dimA.size() < 4) {
            int next_stride = 1;
            if (!strideA.empty()) {
                // Use dim[0] * stride[0] to mimic contiguous layout
                next_stride = dimA[0] * strideA[0];
            }
            dimA.insert(dimA.begin(), 1);
            strideA.insert(strideA.begin(), next_stride);
        }
    }

    cudnnDataType_t dtype;
    if (t.dtype() == DType::Float16) dtype = CUDNN_DATA_HALF;
    else if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
    else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, dtype, static_cast<int>(dimA.size()), dimA.data(), strideA.data()));
    return desc;
}

// A contiguous 4-D tensor whose channel dimension is innermost.
inline bool is_channel_last_4d(const Tensor& t) {
    return t.dim() == 4 && t.is_contiguous(MemoryFormat::ChannelsLast);
}

inline cudnnDataType_t tensor_descriptor_dtype(const Tensor& t) {
    if (t.dtype() == DType::Float16) return CUDNN_DATA_HALF;
    if (t.dtype() == DType::Float32) return CUDNN_DATA_FLOAT;
    if (t.dtype() == DType::Float64) return CUDNN_DATA_DOUBLE;
    TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
}

// Descriptor that reads the operand where it lies: a channel-last operand is
// named through the extended form, which carries the order rather than a
// stride list, so the routine addresses it without a repack.
inline cudnnTensorDescriptor_t createTensorDescriptorPreservingFormat(
    const Tensor& t, bool pad_to_4d = true) {
    if (!is_channel_last_4d(t)) return createTensorDescriptor(t, pad_to_4d);
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    std::vector<int> dims;
    dims.reserve(static_cast<size_t>(t.dim()));
    for (int i = 0; i < t.dim(); ++i) {
        dims.push_back(static_cast<int>(t.size(i)));
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptorEx(
        desc, CUDNN_TENSOR_NHWC, tensor_descriptor_dtype(t), t.dim(),
        dims.data()));
    return desc;
}

// Uninitialized tensor carrying the same order as `t`.
inline Tensor empty_like_preserving_format(const Tensor& t) {
    const std::vector<int64_t> shape =
        static_cast<std::vector<int64_t>>(t.shape());
    Tensor out = Tensor::empty(shape, t.dtype(), t.device());
    if (is_channel_last_4d(t)) {
        out = out.as_strided(
            shape, get_strides_for(shape, MemoryFormat::ChannelsLast));
    }
    return out;
}

#endif

} // namespace cuda
} // namespace tensorplay
