#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Context.h"
#include "Complex.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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

template <typename T>
struct FillFunctor {
    T value;

    __device__ T operator()() const { return value; }
};

template <typename T>
inline void fill_iter(TensorIteratorBase& iter, T value) {
    gpu_kernel(iter, FillFunctor<T>{value});
}

Tensor& fill_kernel(Tensor& self, Scalar value) {
    int64_t n = self.numel();
    if (n == 0) return self;
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(self)
        .resize_outputs(false)
        .build();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        fill_iter<ctype>(iter, value.to<ctype>()); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        // Complex storage dtypes are not part of the real-type macro.
        case DType::ComplexFloat: {
            tensorplay::complex<float> val = value.to<tensorplay::complex<float>>();
            fill_kernel_cuda_impl<tensorplay::complex<float>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, static_cast<tensorplay::complex<float>*>(self.data_ptr()), val);
            break;
        }
        case DType::ComplexDouble: {
            tensorplay::complex<double> val = value.to<tensorplay::complex<double>>();
            fill_kernel_cuda_impl<tensorplay::complex<double>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, static_cast<tensorplay::complex<double>*>(self.data_ptr()), val);
            break;
        }
        case DType::ComplexHalf: {
            fill_kernel_cuda_impl<tensorplay::complex<Half>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, static_cast<tensorplay::complex<Half>*>(self.data_ptr()),
                value.to<tensorplay::complex<Half>>());
            break;
        }
        case DType::BComplex32: {
            fill_kernel_cuda_impl<tensorplay::complex<BFloat16>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                n, static_cast<tensorplay::complex<BFloat16>*>(self.data_ptr()),
                value.to<tensorplay::complex<BFloat16>>());
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

template <typename T>
__global__ void fill_diagonal_strided_cuda_impl(
        int64_t count, T* data, int64_t base, int64_t diag_stride, T value) {
    // Diagonal positions sit count * diag_stride apart in the storage, so
    // each thread writes through the stride pattern instead of a flat range.
    int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) {
        data[base + k * diag_stride] = value;
    }
}

Tensor& fill_diagonal__kernel(Tensor& self, Scalar fill_value, bool wrap) {
    const int64_t n_dims = self.dim();
    if (n_dims < 2) {
        TP_THROW(ValueError, "fill_diagonal_ expects a tensor with at least 2 dimensions");
    }
    const int64_t height = self.size(0);
    const int64_t width = self.size(1);
    if (n_dims > 2) {
        for (int64_t i = 1; i < n_dims; ++i) {
            if (self.size(i) != height) {
                TP_THROW(ValueError, "all dimensions of input must be of equal length");
            }
        }
    }
    if (self.numel() == 0) return self;

    const auto strides = self.strides();
    int64_t diag_stride = 0;
    for (int64_t i = 0; i < n_dims; ++i) {
        diag_stride += strides[i];
    }
    const int64_t base = static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset());
    const int64_t diag_size = std::min(height, width);
    int64_t wrap_count = 0;
    int64_t wrap_base = 0;
    if (wrap && n_dims == 2 && height > width + 1) {
        const int64_t step = width + 1;
        wrap_count = (self.numel() + step - 1) / step - diag_size;
        wrap_base = base + self.stride(0) * step;
    }

    const int threads = 256;
    const int64_t max_count = diag_size > wrap_count ? diag_size : wrap_count;
    const int blocks = static_cast<int>((max_count + threads - 1) / threads);

    #define TP_FILL_DIAG_CUDA_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        ctype val = fill_value.to<ctype>(); \
        fill_diagonal_strided_cuda_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            diag_size, data, base, diag_stride, val); \
        if (wrap_count > 0) { \
            fill_diagonal_strided_cuda_impl<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
                wrap_count, data, wrap_base, diag_stride, val); \
        } \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_FILL_DIAG_CUDA_CASE)
        case DType::ComplexFloat: {
            tensorplay::complex<float>* data = static_cast<tensorplay::complex<float>*>(self.data_ptr());
            tensorplay::complex<float> val = fill_value.to<tensorplay::complex<float>>();
            fill_diagonal_strided_cuda_impl<tensorplay::complex<float>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                diag_size, data, base, diag_stride, val);
            if (wrap_count > 0) {
                fill_diagonal_strided_cuda_impl<tensorplay::complex<float>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    wrap_count, data, wrap_base, diag_stride, val);
            }
            break;
        }
        case DType::ComplexDouble: {
            tensorplay::complex<double>* data = static_cast<tensorplay::complex<double>*>(self.data_ptr());
            tensorplay::complex<double> val = fill_value.to<tensorplay::complex<double>>();
            fill_diagonal_strided_cuda_impl<tensorplay::complex<double>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                diag_size, data, base, diag_stride, val);
            if (wrap_count > 0) {
                fill_diagonal_strided_cuda_impl<tensorplay::complex<double>><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    wrap_count, data, wrap_base, diag_stride, val);
            }
            break;
        }
        default: TP_THROW(NotImplementedError, "fill_diagonal_ not implemented for this dtype on CUDA");
    }
    #undef TP_FILL_DIAG_CUDA_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA fill_diagonal_ Error: ") + cudaGetErrorString(err));
    }
    return self;
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

namespace {

DType infer_full_dtype(const Scalar& fill_value) {
    if (fill_value.isBoolean()) return DType::Bool;
    if (fill_value.isIntegral(false)) return DType::Int64;
    if (fill_value.isComplex()) {
        return globalContext().defaultDType() == DType::Float64
                   ? DType::ComplexDouble
                   : DType::ComplexFloat;
    }
    return globalContext().defaultDType();
}

} // namespace

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device, bool pin_memory) {
    if (pin_memory) TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
    if (dtype == DType::Undefined) {
        dtype = infer_full_dtype(fill_value);
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
    
    data[idx] = r == c ? T(1) : T(0);
}

Tensor eye_kernel(int64_t n, int64_t m, DType dtype, Device device, bool requires_grad) {
    if (n < 0) TP_THROW(RuntimeError, "n must be greater or equal to 0, got ", n);
    if (m < 0 && m != -1) {
        TP_THROW(RuntimeError, "m must be greater or equal to 0, got ", m);
    }
    if (m == -1) m = n;
    Tensor t({n, m}, dtype, device);

    int64_t numel = n * m;
    if (numel == 0) return t;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

#define EYE_CUDA_CASE(ctype, name)                                          \
    case DType::name:                                                        \
        eye_kernel_cuda_impl<ctype><<<blocks, threads, 0,                       \
                                      getCurrentCUDAStream().stream()>>>(n, m, \
                                                                          t.data_ptr<ctype>()); \
        break;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(EYE_CUDA_CASE)
        case DType::ComplexFloat:
            eye_kernel_cuda_impl<tensorplay::complex<float>>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    n, m, static_cast<tensorplay::complex<float>*>(t.data_ptr()));
            break;
        case DType::ComplexDouble:
            eye_kernel_cuda_impl<tensorplay::complex<double>>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    n, m, static_cast<tensorplay::complex<double>*>(t.data_ptr()));
            break;
        case DType::ComplexHalf:
            eye_kernel_cuda_impl<tensorplay::complex<Half>>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    n, m, static_cast<tensorplay::complex<Half>*>(t.data_ptr()));
            break;
        case DType::BComplex32:
            eye_kernel_cuda_impl<tensorplay::complex<BFloat16>>
                <<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
                    n, m, static_cast<tensorplay::complex<BFloat16>*>(t.data_ptr()));
            break;
        default:
            TP_THROW(NotImplementedError,
                     "CUDA eye does not support dtype '" + std::string(toString(dtype)) + "'");
    }
#undef EYE_CUDA_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA eye Error: ") + cudaGetErrorString(err));
    }

    return t;
}

Tensor& zero_inplace_kernel(Tensor& self) {
    return fill_kernel(self, 0);
}


template <typename compute_t, typename store_t, typename host_t>
void launch_linspace_complex(
    int64_t steps, const Scalar& start, const Scalar& end, Tensor& output) {
    using value_t = typename compute_t::value_type;
    const host_t start_host = start.to<host_t>();
    const host_t end_host = end.to<host_t>();
    const compute_t start_value(
        static_cast<value_t>(start_host.real()),
        static_cast<value_t>(start_host.imag()));
    const compute_t end_value(
        static_cast<value_t>(end_host.real()),
        static_cast<value_t>(end_host.imag()));
    const compute_t step = steps == 1
        ? compute_t(value_t(0), value_t(0))
        : (end_value - start_value) / static_cast<value_t>(steps - 1);
    const int64_t halfway = steps / 2;
    gpu_kernel_with_index(output,
        [start_value, end_value, step, steps, halfway] GPU_LAMBDA(
            int64_t index) -> store_t {
            if (steps == 1) {
                return store_t(start_value);
            }
            const int64_t distance = index < halfway ? index : steps - index - 1;
            const compute_t value = index < halfway
                ? start_value + step * static_cast<value_t>(distance)
                : end_value - step * static_cast<value_t>(distance);
            return store_t(value);
        });
}

template <typename compute_t, typename store_t, typename host_t>
void launch_logspace_complex(
    int64_t steps, const Scalar& start, const Scalar& end, double base,
    Tensor& output) {
    using value_t = typename compute_t::value_type;
    const host_t start_host = start.to<host_t>();
    const host_t end_host = end.to<host_t>();
    const compute_t start_value(
        static_cast<value_t>(start_host.real()),
        static_cast<value_t>(start_host.imag()));
    const compute_t end_value(
        static_cast<value_t>(end_host.real()),
        static_cast<value_t>(end_host.imag()));
    const compute_t step = steps == 1
        ? compute_t(value_t(0), value_t(0))
        : (end_value - start_value) / static_cast<value_t>(steps - 1);
    const compute_t base_value(
        static_cast<value_t>(base), static_cast<value_t>(0));
    const int64_t halfway = steps / 2;
    gpu_kernel_with_index(output,
        [start_value, end_value, step, base_value, steps, halfway] GPU_LAMBDA(
            int64_t index) -> store_t {
            if (steps == 1) {
                return store_t(tensorplay::pow(base_value, start_value));
            }
            const int64_t distance = index < halfway ? index : steps - index - 1;
            const compute_t exponent = index < halfway
                ? start_value + step * static_cast<value_t>(distance)
                : end_value - step * static_cast<value_t>(distance);
            return store_t(tensorplay::pow(base_value, exponent));
        });
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

// Signatures match the CPU backend's registration stubs:
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

    // (the dispatcher selects all supported types, including Half and
    // BFloat16).
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

    const double s = start.toDouble();
    const double st = step.toDouble();

    #define ARANGE_CASE(ctype, name) \
    case DType::name: \
        gpu_kernel_with_index(t, [s, st] GPU_LAMBDA(int64_t index) -> ctype { \
            return static_cast<ctype>(s + static_cast<double>(index) * st); \
        }); \
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
    return t;
}

Tensor arange_end_cuda(Scalar end, DType dtype, std::optional<Device> device) {
    return arange_start_step_cuda(Scalar(0), end, Scalar(1), dtype, device);
}

static bool is_sequence_factory_dtype(DType dtype) {
    return isIntegralType(dtype) || dtype == DType::Float16 ||
           dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::BFloat16 || isComplexType(dtype);
}

Tensor linspace_cuda(Scalar start, Scalar end, int64_t steps,
                     DType dtype, std::optional<Device> device) {
    Device dev = resolve_factory_device(device);
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    if (!is_sequence_factory_dtype(dtype)) {
        TP_THROW(NotImplementedError,
                 "linspace CUDA does not support dtype '" + std::string(toString(dtype)) + "'");
    }

    Tensor t({steps}, dtype, dev);
    if (steps == 0) return t;

    if (isComplexType(dtype)) {
        switch (dtype) {
            case DType::ComplexHalf:
                launch_linspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<Half>,
                    tensorplay::complex<float>>(steps, start, end, t);
                break;
            case DType::ComplexFloat:
                launch_linspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<float>,
                    tensorplay::complex<float>>(steps, start, end, t);
                break;
            case DType::ComplexDouble:
                launch_linspace_complex<
                    tensorplay::complex<double>, tensorplay::complex<double>,
                    tensorplay::complex<double>>(steps, start, end, t);
                break;
            case DType::BComplex32:
                launch_linspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<BFloat16>,
                    tensorplay::complex<float>>(steps, start, end, t);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "linspace CUDA does not support dtype '" +
                         std::string(toString(dtype)) + "'");
        }
    } else {
        const double s = start.toDouble();
        const double e = end.toDouble();
        const double step = (steps == 1) ? 0.0 : (e - s) / (steps - 1);
        const int64_t halfway = steps / 2;

#define LINSPACE_CASE(ctype, name)                                           \
    case DType::name:                                                         \
        gpu_kernel_with_index(t, [s, e, step, steps, halfway]                 \
            GPU_LAMBDA(int64_t index) -> ctype {                              \
                if (steps == 1) return static_cast<ctype>(s);                \
                const double value = index < halfway                          \
                    ? s + static_cast<double>(index) * step                   \
                    : e - static_cast<double>(steps - index - 1) * step;      \
                return static_cast<ctype>(value);                             \
            });                                                               \
        break;
        switch (dtype) {
            LINSPACE_CASE(uint8_t, UInt8)
            LINSPACE_CASE(int8_t, Int8)
            LINSPACE_CASE(int16_t, Int16)
            LINSPACE_CASE(int32_t, Int32)
            LINSPACE_CASE(int64_t, Int64)
            LINSPACE_CASE(uint16_t, UInt16)
            LINSPACE_CASE(uint32_t, UInt32)
            LINSPACE_CASE(uint64_t, UInt64)
            LINSPACE_CASE(float, Float32)
            LINSPACE_CASE(double, Float64)
            LINSPACE_CASE(tensorplay::Half, Float16)
            LINSPACE_CASE(tensorplay::BFloat16, BFloat16)
            default:
                TP_THROW(NotImplementedError,
                         "linspace CUDA does not support dtype '" +
                         std::string(toString(dtype)) + "'");
        }
#undef LINSPACE_CASE
    }
    return t;
}

Tensor logspace_cuda(Scalar start, Scalar end, int64_t steps, double base,
                     DType dtype, std::optional<Device> device) {
    Device dev = resolve_factory_device(device);
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    if (!is_sequence_factory_dtype(dtype)) {
        TP_THROW(NotImplementedError,
                 "logspace CUDA does not support dtype '" + std::string(toString(dtype)) + "'");
    }

    Tensor t({steps}, dtype, dev);
    if (steps == 0) return t;

    if (isComplexType(dtype)) {
        switch (dtype) {
            case DType::ComplexHalf:
                launch_logspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<Half>,
                    tensorplay::complex<float>>(steps, start, end, base, t);
                break;
            case DType::ComplexFloat:
                launch_logspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<float>,
                    tensorplay::complex<float>>(steps, start, end, base, t);
                break;
            case DType::ComplexDouble:
                launch_logspace_complex<
                    tensorplay::complex<double>, tensorplay::complex<double>,
                    tensorplay::complex<double>>(steps, start, end, base, t);
                break;
            case DType::BComplex32:
                launch_logspace_complex<
                    tensorplay::complex<float>, tensorplay::complex<BFloat16>,
                    tensorplay::complex<float>>(steps, start, end, base, t);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "logspace CUDA does not support dtype '" +
                         std::string(toString(dtype)) + "'");
        }
    } else {
        const double s = start.toDouble();
        const double e = end.toDouble();
        const double step = (steps == 1) ? 0.0 : (e - s) / (steps - 1);
        const int64_t halfway = steps / 2;

#define LOGSPACE_CASE(ctype, name)                                           \
    case DType::name:                                                         \
        gpu_kernel_with_index(t, [s, e, step, base, steps, halfway]            \
            GPU_LAMBDA(int64_t index) -> ctype {                              \
                if (steps == 1) {                                             \
                    return static_cast<ctype>(std::pow(base, s));              \
                }                                                               \
                const double exponent = index < halfway                       \
                    ? s + static_cast<double>(index) * step                   \
                    : e - static_cast<double>(steps - index - 1) * step;      \
                return static_cast<ctype>(std::pow(base, exponent));           \
            });                                                               \
        break;
        switch (dtype) {
            LOGSPACE_CASE(uint8_t, UInt8)
            LOGSPACE_CASE(int8_t, Int8)
            LOGSPACE_CASE(int16_t, Int16)
            LOGSPACE_CASE(int32_t, Int32)
            LOGSPACE_CASE(int64_t, Int64)
            LOGSPACE_CASE(uint16_t, UInt16)
            LOGSPACE_CASE(uint32_t, UInt32)
            LOGSPACE_CASE(uint64_t, UInt64)
            LOGSPACE_CASE(float, Float32)
            LOGSPACE_CASE(double, Float64)
            LOGSPACE_CASE(tensorplay::Half, Float16)
            LOGSPACE_CASE(tensorplay::BFloat16, BFloat16)
            default:
                TP_THROW(NotImplementedError,
                         "logspace CUDA does not support dtype '" +
                         std::string(toString(dtype)) + "'");
        }
#undef LOGSPACE_CASE
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

Tensor empty_memory_format_stub(const std::vector<int64_t>& size,
                                std::optional<DType> dtype,
                                std::optional<int64_t> layout,
                                std::optional<Device> device,
                                std::optional<bool> pin_memory,
                                std::optional<int64_t> memory_format) {
    (void)memory_format;
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError,
                 "empty is only implemented for strided (dense) layout tensors");
    }
    return empty_kernel(size, resolve_factory_dtype(dtype),
                        device.value_or(Device(DeviceType::CUDA)),
                        pin_memory.value_or(false));
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
    m.impl("fill_diagonal_", fill_diagonal__kernel);
    m.impl("zero_", zero_inplace_kernel);
    m.impl("zeros", zeros_stub);
    m.impl("ones", ones_stub);
    m.impl("empty", empty_stub);
    m.impl("empty.memory_format", empty_memory_format_stub);
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
