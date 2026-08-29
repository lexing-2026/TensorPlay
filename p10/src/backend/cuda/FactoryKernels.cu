#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Context.h"
#include <cuda_runtime.h>
#include <cmath>
#include <complex>
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
        fill_kernel_cuda_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(n, self.data_ptr<ctype>(), val); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        // Complex storage dtypes are not part of the real-type macro.
        case DType::ComplexFloat: {
            std::complex<float> val = value.to<std::complex<float>>();
            fill_kernel_cuda_impl<std::complex<float>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, self.data_ptr<std::complex<float>>(), val);
            break;
        }
        case DType::ComplexDouble: {
            std::complex<double> val = value.to<std::complex<double>>();
            fill_kernel_cuda_impl<std::complex<double>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, self.data_ptr<std::complex<double>>(), val);
            break;
        }
        default: TP_THROW(NotImplementedError, "fill_ not implemented for this dtype on CUDA");
    }
    #undef OP_CASE
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         TP_THROW(RuntimeError, std::string("CUDA fill_ Error: ") + cudaGetErrorString(err));
    }
    
    return self;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    if (pin_memory) TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    // Tensor constructor allocates memory (via empty)
    Tensor t(size, dtype, device);
    fill_kernel(t, 0);
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    if (pin_memory) TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    Tensor t(size, dtype, device);
    fill_kernel(t, 1);
    return t;
}

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    if (pin_memory) TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
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
    return zeros_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
}

Tensor ones_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return ones_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
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

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device, bool pin_memory) {
    if (pin_memory) TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    if (dtype == DType::Undefined) {
        if (fill_value.isFloatingPoint()) dtype = globalContext().defaultDType();
        else if (fill_value.isIntegral()) dtype = DType::Int64;
        else if (fill_value.isBoolean()) dtype = DType::Bool;
        else dtype = globalContext().defaultDType();
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
        eye_kernel_cuda_impl<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(n, m, t.data_ptr<float>());
    } else if (dtype == DType::Float64) {
        eye_kernel_cuda_impl<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(n, m, t.data_ptr<double>());
    } else if (dtype == DType::Int64) {
        eye_kernel_cuda_impl<int64_t><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(n, m, t.data_ptr<int64_t>());
    } else if (dtype == DType::Int32) {
        eye_kernel_cuda_impl<int32_t><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(n, m, t.data_ptr<int32_t>());
    } else {
        TP_THROW(NotImplementedError, "CUDA eye: only float32/float64/int64/int32 supported");
    }
    
    return t;
}

Tensor& zero_inplace_kernel(Tensor& self) {
    return fill_kernel(self, 0);
}


template <typename T>
__global__ void arange_fill_impl(int64_t n, double start, double step, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<T>(start + static_cast<double>(i) * step);
    }
}

template <typename T>
__global__ void linspace_fill_impl(int64_t n, double start, double step, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        // steps == 1 collapses to `start` on the host; here step is pre-divided.
        out[i] = static_cast<T>(start + static_cast<double>(i) * step);
    }
}

template <typename T>
__global__ void logspace_fill_impl(int64_t n, double start, double step,
                                   double base, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        double val = start + static_cast<double>(i) * step;
        out[i] = static_cast<T>(std::pow(base, val));
    }
}

static int64_t arange_length(Scalar start, Scalar end, Scalar step) {
    double s_d = start.toDouble();
    double e_d = end.toDouble();
    double st_d = step.toDouble();
    int64_t len;
    if (start.isIntegral() && end.isIntegral() && step.isIntegral()) {
        int64_t s = start.to<int64_t>();
        int64_t e = end.to<int64_t>();
        int64_t st = step.to<int64_t>();
        if (st == 0) TP_THROW(RuntimeError, "step must be nonzero");
        if ((st > 0 && s > e) || (st < 0 && s < e)) {
            len = 0;
        } else {
            len = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
        }
    } else {
        if (st_d == 0) TP_THROW(RuntimeError, "step must be nonzero");
        len = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
    }
    if (len < 0) len = 0;
    return len;
}

static Device resolve_factory_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : Device(globalContext().defaultDevice());
}

// Signatures mirror the CPU backend's registration stubs exactly:
// std::optional<Device>, no requires_grad (dispatcher-owned).
Tensor arange_start_step_cuda(Scalar start, Scalar end, Scalar step,
                              DType dtype, std::optional<Device> device) {
    Device dev = resolve_factory_device(device);
    int64_t len = arange_length(start, end, step);
    if (dtype == DType::Undefined) {
        if (start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()) {
            dtype = globalContext().defaultDType();
        } else {
            dtype = DType::Int64;
        }
    }

    // (upstream dispatches over AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16)).
    const bool arange_dtype_supported =
        dtype == DType::Float32 || dtype == DType::Float64 ||
        dtype == DType::Int64 || dtype == DType::Int32 || dtype == DType::Int16 ||
        dtype == DType::Int8 || dtype == DType::UInt8 ||
        dtype == DType::Float16 || dtype == DType::BFloat16;
    if (!arange_dtype_supported) {
        TP_THROW(NotImplementedError,
                 "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
    }

    Tensor t({len}, dtype, dev);
    if (len == 0) return t;

    double s = start.toDouble();
    double st = step.toDouble();
    int threads = 256;
    int blocks = static_cast<int>((len + threads - 1) / threads);

    #define ARANGE_CASE(ctype, name) \
    case DType::name: \
        arange_fill_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            len, s, st, t.data_ptr<ctype>()); \
        break;
    switch (dtype) {
        ARANGE_CASE(float, Float32)
        ARANGE_CASE(double, Float64)
        ARANGE_CASE(int64_t, Int64)
        ARANGE_CASE(int32_t, Int32)
        ARANGE_CASE(int16_t, Int16)
        ARANGE_CASE(int8_t, Int8)
        ARANGE_CASE(uint8_t, UInt8)
        ARANGE_CASE(tensorplay::Half, Float16)
        ARANGE_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
    }
    #undef ARANGE_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA arange Error: ") + cudaGetErrorString(err));
    }
    return t;
}

Tensor arange_end_cuda(Scalar end, DType dtype, std::optional<Device> device) {
    return arange_start_step_cuda(Scalar(0), end, Scalar(1), dtype, device);
}

Tensor linspace_cuda(Scalar start, Scalar end, int64_t steps,
                     DType dtype, std::optional<Device> device) {
    Device dev = resolve_factory_device(device);
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");

    Tensor t({steps}, dtype, dev);
    if (steps == 0) return t;

    double s = start.toDouble();
    double e = end.toDouble();
    double step = (steps == 1) ? 0.0 : (e - s) / (steps - 1);

    int threads = 256;
    int blocks = static_cast<int>((steps + threads - 1) / threads);

    #define LINSPACE_CASE(ctype, name) \
    case DType::name: \
        linspace_fill_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            steps, s, step, t.data_ptr<ctype>()); \
        break;
    switch (dtype) {
        LINSPACE_CASE(float, Float32)
        LINSPACE_CASE(double, Float64)
        default:
            TP_THROW(NotImplementedError, "linspace: only Float32/Float64 supported on CUDA");
    }
    #undef LINSPACE_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA linspace Error: ") + cudaGetErrorString(err));
    }
    return t;
}

Tensor logspace_cuda(Scalar start, Scalar end, int64_t steps, double base,
                     DType dtype, std::optional<Device> device) {
    Device dev = resolve_factory_device(device);
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");

    Tensor t({steps}, dtype, dev);
    if (steps == 0) return t;

    double s = start.toDouble();
    double e = end.toDouble();
    double step = (steps == 1) ? 0.0 : (e - s) / (steps - 1);

    int threads = 256;
    int blocks = static_cast<int>((steps + threads - 1) / threads);

    #define LOGSPACE_CASE(ctype, name) \
    case DType::name: \
        logspace_fill_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            steps, s, step, base, t.data_ptr<ctype>()); \
        break;
    switch (dtype) {
        LOGSPACE_CASE(float, Float32)
        LOGSPACE_CASE(double, Float64)
        default:
            TP_THROW(NotImplementedError, "logspace: only Float32/Float64 supported on CUDA");
    }
    #undef LOGSPACE_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA logspace Error: ") + cudaGetErrorString(err));
    }
    return t;
}


// --- Stub-ABI adapters ------------------------------------------------------
// The dispatcher invokes kernels with schema-level argument types
// (std::optional<DType> / std::optional<Device>, no requires_grad), while the
// raw kernels above keep concrete parameters for internal reuse.
namespace {

DType resolve_factory_dtype(const std::optional<DType>& dtype) {
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

Tensor zeros_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return zeros_kernel(size, resolve_factory_dtype(dtype),
                        device.value_or(Device(DeviceType::CUDA)), pin_memory);
}

Tensor ones_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device, bool pin_memory) {
    return ones_kernel(size, resolve_factory_dtype(dtype),
                       device.value_or(Device(DeviceType::CUDA)), pin_memory);
}

Tensor empty_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return empty_kernel(size, resolve_factory_dtype(dtype),
                        device.value_or(Device(DeviceType::CUDA)), pin_memory);
}

Tensor full_stub(const std::vector<int64_t>& size, Scalar fill_value,
                 DType dtype, std::optional<Device> device, bool pin_memory) {
    return full_kernel(size, fill_value, dtype,
                       device.value_or(Device(DeviceType::CUDA)), pin_memory);
}

Tensor eye_stub(int64_t n, int64_t m, DType dtype, std::optional<Device> device) {
    return eye_kernel(n, m, dtype, device.value_or(Device(DeviceType::CUDA)), false);
}

Tensor arange_start_step_stub(Scalar start, Scalar end, Scalar step, DType dtype,
                              std::optional<Device> device) {
    return arange_start_step_cuda(start, end, step, dtype, device);
}

Tensor arange_end_stub(Scalar end, DType dtype, std::optional<Device> device) {
    return arange_end_cuda(end, dtype, device);
}

Tensor linspace_stub(Scalar start, Scalar end, int64_t steps, DType dtype,
                     std::optional<Device> device) {
    return linspace_cuda(start, end, steps, dtype, device);
}

Tensor logspace_stub(Scalar start, Scalar end, int64_t steps, double base,
                     DType dtype, std::optional<Device> device) {
    return logspace_cuda(start, end, steps, base, dtype, device);
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, FactoryKernels) {
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_inplace_kernel);
    m.impl("zeros", zeros_stub);
    m.impl("ones", ones_stub);
    m.impl("empty", empty_stub);
    m.impl("zeros_like", zeros_like_kernel);
    m.impl("ones_like", ones_like_kernel);
    m.impl("empty_like", empty_like_kernel);
    m.impl("full_like", full_like_kernel);
    m.impl("full", full_stub);
    m.impl("eye", eye_stub);
    m.impl("arange", arange_start_step_stub);
    m.impl("arange.end", arange_end_stub);
    m.impl("linspace", linspace_stub);
    m.impl("logspace", logspace_stub);
}

} // namespace cuda
} // namespace tensorplay
