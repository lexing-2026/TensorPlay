#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "Exception.h"
#include <cuda_runtime.h>
#include <curand.h>

namespace tensorplay {
namespace cuda {

// --- Kernels ---

Tensor rand_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();
    
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        curandGenerator_t gen = CUDAContext::getCurandGenerator();
        
        // curandGenerateUniform generates uniformly distributed floats in (0.0, 1.0]
        // Note: It returns (0, 1], while C++ std::uniform_real_distribution returns [0, 1).
        // This slight difference is usually acceptable.
        curandStatus_t status = curandGenerateUniform(gen, data, n);
        
        if (status != CURAND_STATUS_SUCCESS) {
             TP_THROW(RuntimeError, "curandGenerateUniform failed");
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        curandGenerator_t gen = CUDAContext::getCurandGenerator();
        curandStatus_t status = curandGenerateUniformDouble(gen, data, n);
         if (status != CURAND_STATUS_SUCCESS) {
             TP_THROW(RuntimeError, "curandGenerateUniformDouble failed");
        }
    } else {
         TP_THROW(NotImplementedError, "rand() only supports Float32/Float64 on CUDA for now");
    }
    return t;
}

Tensor randn_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();
    
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        curandGenerator_t gen = CUDAContext::getCurandGenerator();
        
        // curandGenerateNormal generates normally distributed floats
        // mean=0.0, stddev=1.0
        curandStatus_t status = curandGenerateNormal(gen, data, n, 0.0f, 1.0f);
        
        if (status != CURAND_STATUS_SUCCESS) {
             TP_THROW(RuntimeError, "curandGenerateNormal failed");
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        curandGenerator_t gen = CUDAContext::getCurandGenerator();
        curandStatus_t status = curandGenerateNormalDouble(gen, data, n, 0.0, 1.0);
         if (status != CURAND_STATUS_SUCCESS) {
             TP_THROW(RuntimeError, "curandGenerateNormalDouble failed");
        }
    } else {
         TP_THROW(NotImplementedError, "randn() only supports Float32/Float64 on CUDA for now");
    }
    return t;
}

Tensor rand_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    
    Device target_device = device.has_value() ? *device : self.device();
    
    return rand_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

Tensor randn_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device target_device = device.has_value() ? *device : self.device();
    return randn_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RandomKernels) {
    m.impl("rand", rand_kernel_cuda);
    m.impl("randn", randn_kernel_cuda);
    m.impl("rand_like", rand_like_kernel_cuda);
    m.impl("randn_like", randn_like_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
