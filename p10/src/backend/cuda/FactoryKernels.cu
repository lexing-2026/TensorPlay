#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include <cuda_runtime.h>
#include <vector>

namespace tensorplay {
namespace cuda {

template <typename T>
__global__ void fill_kernel_cuda_impl(int n, T* data, T value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}

Tensor& fill_kernel(Tensor& self, Scalar value) {
    int64_t n = self.numel();
    if (n == 0) return self;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype val = value.to<ctype>(); \
        fill_kernel_cuda_impl<ctype><<<blocks, threads>>>(n, self.data_ptr<ctype>(), val); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "fill_ not implemented for this dtype on CUDA");
    }
    #undef OP_CASE
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         TP_THROW(RuntimeError, std::string("CUDA fill_ Error: ") + cudaGetErrorString(err));
    }
    
    return self;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    // Tensor constructor allocates memory (via empty)
    Tensor t(size, dtype, device);
    fill_kernel(t, 0);
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    fill_kernel(t, 1);
    return t;
}

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    // pin_memory ignored for now (or TODO)
    return Tensor(size, dtype, device);
}

Tensor rand_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    // For now we don't have rand_kernel exposed here, but we can implement it or leave it
    // Wait, RandomKernels.cu should implement rand/randn.
    // Let's just implement zeros_like/ones_like/empty_like which rely on kernels in this file.
    TP_THROW(NotImplementedError, "rand_like not fully implemented in FactoryKernels.cu");
}

Tensor zeros_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return zeros_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return ones_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
}

Tensor full_like_kernel(const Tensor& self, Scalar fill_value, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    Tensor t = empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
    return fill_kernel(t, fill_value);
}

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device, bool requires_grad) {
    if (dtype == DType::Undefined) {
        if (fill_value.isFloatingPoint()) dtype = DType::Float32;
        else if (fill_value.isIntegral()) dtype = DType::Int64;
        else if (fill_value.isBoolean()) dtype = DType::Bool;
        else dtype = DType::Float32;
    }
    Tensor t(size, dtype, device);
    fill_kernel(t, fill_value);
    return t;
}

template <typename T>
__global__ void eye_kernel_cuda_impl(int64_t n, int64_t m, T* data) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m) return;
    
    int64_t r = idx / m;
    int64_t c = idx % m;
    
    if (r == c) data[idx] = 1;
    else data[idx] = 0;
}

Tensor eye_kernel(int64_t n, int64_t m, DType dtype, Device device, bool requires_grad) {
    if (m == -1) m = n;
    Tensor t({n, m}, dtype, device);
    
    int64_t numel = n * m;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    if (dtype == DType::Float32) {
        eye_kernel_cuda_impl<float><<<blocks, threads>>>(n, m, t.data_ptr<float>());
    } else {
        TP_THROW(NotImplementedError, "CUDA eye: only float32 supported");
    }
    
    return t;
}

Tensor& zero_inplace_kernel(Tensor& self) {
    return fill_kernel(self, 0);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, FactoryKernels) {
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_inplace_kernel);
    m.impl("zeros", zeros_kernel);
    m.impl("ones", ones_kernel);
    m.impl("empty", empty_kernel);
    m.impl("zeros_like", zeros_like_kernel);
    m.impl("ones_like", ones_like_kernel);
    m.impl("empty_like", empty_like_kernel);
    m.impl("full_like", full_like_kernel);
    m.impl("full", full_kernel);
    m.impl("eye", eye_kernel);
}

} // namespace cuda
} // namespace tensorplay
