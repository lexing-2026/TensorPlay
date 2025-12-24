#pragma once

#include "Tensor.h"
#include "Exception.h"
#include <cudnn.h>
#include <vector>

namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      std::string err_msg = std::string("cuDNN Error: ") + cudnnGetErrorString(status) + " at " + __FILE__ + ":" + std::to_string(__LINE__) + " in " + #condition; \
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
    if (t.dtype() == DType::Float32) dtype = CUDNN_DATA_FLOAT;
    else if (t.dtype() == DType::Float64) dtype = CUDNN_DATA_DOUBLE;
    else TP_THROW(NotImplementedError, "cuDNN: only float/double supported");
    
    // DEBUG PRINT
    
    std::cout << "createTensorDescriptor: shape=(";
    for(auto d : dimA) std::cout << d << ",";
    std::cout << ") stride=(";
    for(auto s : strideA) std::cout << s << ",";
    std::cout << ")" << std::endl;
    
    
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, dtype, static_cast<int>(dimA.size()), dimA.data(), strideA.data()));
    return desc;
}

#endif

} // namespace cuda
} // namespace tensorplay
