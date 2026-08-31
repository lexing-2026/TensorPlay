#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "DistributionDispatch.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace tensorplay {
namespace cpu {

Tensor& fill_kernel(Tensor& self, Scalar value);

Tensor allocate_cpu_tensor(const std::vector<int64_t>& size, DType dtype, bool pin_memory) {
#ifdef USE_CUDA
    if (pin_memory) {
        int64_t numel = 1;
        for (int64_t value : size) numel *= value;
        const size_t nbytes = static_cast<size_t>(numel) * elementSize(dtype);
        Storage storage(nbytes, getPinnedMemoryAllocator(), Device(DeviceType::CPU));
        return Tensor(storage, size, dtype);
    }
#else
    if (pin_memory) {
        TP_THROW(RuntimeError, "pin_memory requires a CUDA-enabled TensorPlay build");
    }
#endif
    return Tensor(size, dtype, Device(DeviceType::CPU));
}

Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    int64_t n = t.numel();
    auto& gen = default_generator();

    switch (dtype) {
        case DType::Float32: {
            float* data = t.data_ptr<float>();
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            const float to_scalar = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                float value = static_cast<float>(dist(&gen));
                data[i] = value == to_scalar ? 0.0f : value;
            }
            break;
        }
        case DType::Float64: {
            double* data = t.data_ptr<double>();
            uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = dist(&gen);
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // uniform_kernel_cpu): Half/BFloat16 sample in opmath_t (float,
            // 24-bit mantissa mask) and cast to the storage dtype, clamping a
            // cast that rounded up to the upper bound back to 'from'.
            using math_t = float;
            const math_t lo = static_cast<math_t>(0.0);
            const math_t hi = static_cast<math_t>(1.0);
            const math_t to_scalar = hi;
            const math_t from_scalar = lo;
            if (dtype == DType::Float16) {
                Half* data = t.data_ptr<Half>();
                uniform_real_distribution<math_t> dist(lo, hi);
                for (int64_t i = 0; i < n; ++i) {
                    Half value = static_cast<Half>(dist(&gen));
                    data[i] = static_cast<math_t>(value) == to_scalar
                                  ? static_cast<Half>(from_scalar) : value;
                }
            } else {
                BFloat16* data = t.data_ptr<BFloat16>();
                uniform_real_distribution<math_t> dist(lo, hi);
                for (int64_t i = 0; i < n; ++i) {
                    BFloat16 value = static_cast<BFloat16>(dist(&gen));
                    data[i] = static_cast<math_t>(value) == to_scalar
                                  ? static_cast<BFloat16>(from_scalar) : value;
                }
            }
            break;
        }
        case DType::ComplexFloat: {
            std::complex<float>* data = t.data_ptr<std::complex<float>>();
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            const float to_scalar = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                float re = static_cast<float>(dist(&gen));
                float im = static_cast<float>(dist(&gen));
                data[i] = std::complex<float>(
                    re == to_scalar ? 0.0f : re,
                    im == to_scalar ? 0.0f : im);
            }
            break;
        }
        case DType::ComplexDouble: {
            std::complex<double>* data = t.data_ptr<std::complex<double>>();
            uniform_real_distribution<double> dist(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = std::complex<double>(dist(&gen), dist(&gen));
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "rand() only supports floating dtypes for now");
    }
    return t;
}

namespace {
// E|z|^2 == 1; complex rand draws components uniform on [0, 1).
constexpr double kComplexComponentStd = 0.7071067811865476;  // 1/sqrt(2)
}  // namespace

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    Tensor t = allocate_cpu_tensor(size, dtype, pin_memory);
    size_t nbytes = t.numel() * t.itemsize();
    if (t.data_ptr()) {
        std::memset(t.data_ptr(), 0, nbytes);
    }
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    Tensor t = allocate_cpu_tensor(size, dtype, pin_memory);
    fill_kernel(t, 1);
    return t;
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
    DType inferred_dtype = dtype;
    if (inferred_dtype == DType::Undefined) {
        inferred_dtype = infer_full_dtype(fill_value);
    }
    Tensor t = allocate_cpu_tensor(size, inferred_dtype, pin_memory);
    fill_kernel(t, fill_value);
    return t;
}

Tensor& fill_kernel(Tensor& self, Scalar value) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        int64_t n = self.numel(); \
        ctype val = value.to<ctype>(); \
        std::fill(data, data + n, val); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(OP_CASE)
        default: 
            std::cerr << "fill_kernel error: dtype=" << (int)self.dtype() << std::endl;
            TP_THROW(NotImplementedError, "fill_ not implemented for this dtype");
    }
    #undef OP_CASE
    return self;
}

#include <iostream>

Tensor arange_start_step_kernel(Scalar start, Scalar end, Scalar step, DType dtype, Device device) {
    // Better length calculation to avoid precision issues with large integers
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
             // ceil((end-start)/step)
             double tmp = std::ceil((e_d - s_d) / st_d);
             len = static_cast<int64_t>(tmp);
         }
    } else {
         len = static_cast<int64_t>(std::ceil((e_d - s_d) / st_d));
    }

    if (len < 0) len = 0;
    
    // Type inference if Undefined
    if (dtype == DType::Undefined) {
        if (start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()) {
            dtype = globalContext().defaultDType();
        } else {
            dtype = DType::Int64;
        }
    }
    
    Tensor t({len}, dtype, device);

    // AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16) -- Bool, UInt16/32/64 and
    // complex are not implemented.  The sub-32-bit types supported below
    // were previously left uninitialized.
    switch (dtype) {
        case DType::Bool:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
#define ARANGE_CASE(ctype, name) \
        case DType::name: { \
            ctype* data = t.data_ptr<ctype>(); \
            for (int64_t i = 0; i < len; ++i) data[i] = static_cast<ctype>(s_d + i * st_d); \
            break; \
        }
        ARANGE_CASE(uint8_t, UInt8)
        ARANGE_CASE(int8_t, Int8)
        ARANGE_CASE(int16_t, Int16)
        ARANGE_CASE(int32_t, Int32)
        ARANGE_CASE(int64_t, Int64)
        ARANGE_CASE(float, Float32)
        ARANGE_CASE(double, Float64)
        ARANGE_CASE(tensorplay::Half, Float16)
        ARANGE_CASE(tensorplay::BFloat16, BFloat16)
#undef ARANGE_CASE
        default:
            TP_THROW(NotImplementedError,
                     "\"arange\" not implemented for '" + std::string(toString(dtype)) + "'");
    }
    return t;
}

Tensor arange_kernel(Scalar end, DType dtype, Device device) {
    return arange_start_step_kernel(Scalar(0), end, Scalar(1), dtype, device);
}

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype, Device device, bool pin_memory) {
    return allocate_cpu_tensor(size, dtype, pin_memory);
}

Tensor eye_kernel(int64_t n, int64_t m, DType dtype, Device device) {
    if (m < 0) m = n;
    Tensor t = Tensor::zeros({n, m}, dtype, device);
    int64_t min_dim = std::min(n, m);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        for(int64_t i=0; i<min_dim; ++i) {
            data[i*m + i] = 1.0f;
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        for(int64_t i=0; i<min_dim; ++i) {
            data[i*m + i] = 1.0;
        }
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        for(int64_t i=0; i<min_dim; ++i) {
            data[i*m + i] = 1;
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = t.data_ptr<int32_t>();
        for(int64_t i=0; i<min_dim; ++i) {
            data[i*m + i] = 1;
        }
    }
    return t;
}

Tensor linspace_kernel(Scalar start, Scalar end, int64_t steps, DType dtype, Device device) {
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    Tensor t({steps}, dtype, device);
    if (steps == 0) return t;
    
    double s = start.toDouble();
    double e = end.toDouble();
    
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        if (steps == 1) {
            data[0] = static_cast<float>(s);
        } else {
            double step = (e - s) / (steps - 1);
            for(int64_t i=0; i<steps; ++i) {
                data[i] = static_cast<float>(s + i * step);
            }
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        if (steps == 1) {
            data[0] = s;
        } else {
            double step = (e - s) / (steps - 1);
            for(int64_t i=0; i<steps; ++i) {
                data[i] = s + i * step;
            }
        }
    } else {
         TP_THROW(NotImplementedError, "linspace only supports Float32/Float64");
    }
    return t;
}

Tensor logspace_kernel(Scalar start, Scalar end, int64_t steps, double base, DType dtype, Device device) {
    if (steps < 0) TP_THROW(RuntimeError, "number of steps must be non-negative");
    Tensor t({steps}, dtype, device);
    if (steps == 0) return t;
    
    double s = start.toDouble();
    double e = end.toDouble();
    
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        if (steps == 1) {
            data[0] = static_cast<float>(std::pow(base, s));
        } else {
            double step = (e - s) / (steps - 1);
            for(int64_t i=0; i<steps; ++i) {
                double val = s + i * step;
                data[i] = static_cast<float>(std::pow(base, val));
            }
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        if (steps == 1) {
            data[0] = std::pow(base, s);
        } else {
            double step = (e - s) / (steps - 1);
            for(int64_t i=0; i<steps; ++i) {
                double val = s + i * step;
                data[i] = std::pow(base, val);
            }
        }
    } else {
         TP_THROW(NotImplementedError, "logspace only supports Float32/Float64");
    }
    return t;
}

// --- Random Factory Kernels ---

Tensor randn_kernel(const std::vector<int64_t>& size, DType dtype, Device device,
                    Generator* generator = nullptr, bool pin_memory = false) {
    Tensor t = device.type() == DeviceType::CPU
        ? allocate_cpu_tensor(size, dtype, pin_memory)
        : Tensor(size, dtype, device);
    int64_t n = t.numel();
    Generator* gen = generator != nullptr ? generator : &default_generator();

    switch (dtype) {
        case DType::Float32: {
            float* data = t.data_ptr<float>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<float>(data, n, 0.0f, 1.0f, gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = static_cast<float>(dist(gen));
                }
            }
            break;
        }
        case DType::Float64: {
            double* data = t.data_ptr<double>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<double>(data, n, 0.0, 1.0, gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // normal_fill<scalar_t>): Half/BFloat16 draw uniforms in opmath
            // (float, 24-bit mantissa) through a 16-element stack buffer,
            // Box-Muller in float, then cast down to the storage dtype.
            if (n >= 16 && t.is_contiguous()) {
                if (dtype == DType::Float16) {
                    normal_fill_cast<Half>(t.data_ptr<Half>(), n, 0.0, 1.0, gen);
                } else {
                    normal_fill_cast<BFloat16>(t.data_ptr<BFloat16>(), n, 0.0, 1.0, gen);
                }
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                if (dtype == DType::Float16) {
                    Half* data = t.data_ptr<Half>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<Half>(dist(gen));
                    }
                } else {
                    BFloat16* data = t.data_ptr<BFloat16>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<BFloat16>(dist(gen));
                    }
                }
            }
            break;
        }
        case DType::ComplexFloat:
        case DType::ComplexDouble: {
            // normal samples view_as_real(self) with std/sqrt(2); with the
            // standard-normal factory that is N(0, 1/sqrt(2)) per component.
            const double comp_std = kComplexComponentStd;
            if (dtype == DType::ComplexFloat) {
                std::complex<float>* data = t.data_ptr<std::complex<float>>();
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::complex<float>(
                        static_cast<float>(dist(gen) * comp_std),
                        static_cast<float>(dist(gen) * comp_std));
                }
            } else {
                std::complex<double>* data = t.data_ptr<std::complex<double>>();
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::complex<double>(dist(gen) * comp_std,
                                                   dist(gen) * comp_std);
                }
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "randn() only supports floating dtypes for now");
    }
    return t;
}

Tensor randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& size, DType dtype, Device device) {
    TP_THROW_IF(high <= low, RuntimeError,
                "randint expects 'from' to be less than 'to', but got from=",
                low, " >= to=", high);

    Tensor t(size, dtype, device);
    auto& gen = default_generator();
    const uint64_t range = static_cast<uint64_t>(high - low);
    const int64_t base = low;

    distribution::check_random_from_to_bounds(low, high, dtype);

    int64_t n = t.numel();
    distribution::dispatch_dtype(dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = t.data_ptr<scalar_t>();
        uniform_int_from_to_distribution<scalar_t> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    });
    return t;
}

Tensor randperm_kernel(int64_t n, DType dtype, Device device) {
    Tensor t({n}, dtype, device);
    if (dtype == DType::Int64 || dtype == DType::Int32) {
        // one 32-bit draw modulo the remaining tail per position.
        auto& gen = default_generator();
        if (dtype == DType::Int64) {
            int64_t* data = t.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; ++i) data[i] = i;
            for (int64_t i = 0; i < n - 1; i++) {
                int64_t z = static_cast<int64_t>(gen.random() % static_cast<uint32_t>(n - i));
                int64_t sav = data[i];
                data[i] = data[z + i];
                data[z + i] = sav;
            }
        } else {
            int32_t* data = t.data_ptr<int32_t>();
            for (int64_t i = 0; i < n; ++i) data[i] = static_cast<int32_t>(i);
            for (int64_t i = 0; i < n - 1; i++) {
                int64_t z = static_cast<int64_t>(gen.random() % static_cast<uint32_t>(n - i));
                int32_t sav = data[i];
                data[i] = data[z + i];
                data[z + i] = sav;
            }
        }
    } else {
        TP_THROW(NotImplementedError, "randperm() only supports Int64/Int32");
    }
    return t;
}

Tensor rand_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return rand_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor randint_like_kernel(const Tensor& self, int64_t low, int64_t high, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return randint_kernel(low, high, static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor randn_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return randn_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev, false);
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

Tensor full_like_kernel(const Tensor& self, Scalar fill_value, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return full_kernel(static_cast<std::vector<int64_t>>(self.shape()), fill_value, dtype, dev, false);
}


Tensor& zero_kernel(Tensor& self) {
    return fill_kernel(self, 0);
}

// --- Stub-ABI adapters ------------------------------------------------------
// The dispatcher invokes kernels with schema-level argument types
// (std::optional<DType> / std::optional<Device>), while the raw kernels above
// keep concrete parameters for internal reuse.  These thin wrappers bridge the
namespace {

Device resolve_factory_device(const std::optional<Device>& device) {
    return device.has_value() ? *device : Device(globalContext().defaultDevice());
}

DType resolve_factory_dtype(const std::optional<DType>& dtype) {
    return (dtype.has_value() && *dtype != DType::Undefined)
               ? *dtype
               : globalContext().defaultDType();
}

Tensor rand_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device) {
    return rand_kernel(size, resolve_factory_dtype(dtype),
                       resolve_factory_device(device));
}

Tensor randn_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device) {
    return randn_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device));
}

Tensor randn_generator_stub(const std::vector<int64_t>& size,
                            std::optional<Generator> generator,
                            std::optional<DType> dtype,
                            std::optional<int64_t> layout,
                            std::optional<Device> device,
                            std::optional<bool> pin_memory) {
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError,
                 "randn is only implemented for strided (dense) layout tensors");
    }
    Generator* generator_ptr = generator.has_value() ? &*generator : nullptr;
    return randn_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), generator_ptr,
                        pin_memory.value_or(false));
}

Tensor randint_stub(int64_t low, int64_t high, const std::vector<int64_t>& size,
                    DType dtype, std::optional<Device> device) {
    return randint_kernel(low, high, size, dtype, resolve_factory_device(device));
}

Tensor randperm_stub(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_kernel(n, dtype, resolve_factory_device(device));
}

Tensor eye_stub(int64_t n, int64_t m, DType dtype, std::optional<Device> device) {
    return eye_kernel(n, m, dtype, resolve_factory_device(device));
}

Tensor arange_start_step_stub(Scalar start, Scalar end, Scalar step, DType dtype,
                              std::optional<Device> device) {
    return arange_start_step_kernel(start, end, step, dtype,
                                    resolve_factory_device(device));
}

Tensor arange_end_stub(Scalar end, DType dtype, std::optional<Device> device) {
    return arange_kernel(end, dtype, resolve_factory_device(device));
}

Tensor linspace_stub(Scalar start, Scalar end, int64_t steps, DType dtype,
                     std::optional<Device> device) {
    return linspace_kernel(start, end, steps, dtype, resolve_factory_device(device));
}

Tensor logspace_stub(Scalar start, Scalar end, int64_t steps, double base,
                     DType dtype, std::optional<Device> device) {
    return logspace_kernel(start, end, steps, base, dtype,
                           resolve_factory_device(device));
}

Tensor empty_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return empty_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), pin_memory);
}

Tensor zeros_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                  std::optional<Device> device, bool pin_memory) {
    return zeros_kernel(size, resolve_factory_dtype(dtype),
                        resolve_factory_device(device), pin_memory);
}

Tensor ones_stub(const std::vector<int64_t>& size, std::optional<DType> dtype,
                 std::optional<Device> device, bool pin_memory) {
    return ones_kernel(size, resolve_factory_dtype(dtype),
                       resolve_factory_device(device), pin_memory);
}

Tensor full_stub(const std::vector<int64_t>& size, Scalar fill_value,
                 DType dtype, std::optional<Device> device, bool pin_memory) {
    return full_kernel(size, fill_value, dtype, resolve_factory_device(device),
                       pin_memory);
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CPU, FactoryKernels) {
    m.impl("rand", rand_stub);
    m.impl("zeros", zeros_stub);
    m.impl("ones", ones_stub);
    m.impl("full", full_stub);
    m.impl("arange", arange_start_step_stub);
    m.impl("arange.end", arange_end_stub);
    m.impl("empty", empty_stub);
    m.impl("eye", eye_stub);
    m.impl("linspace", linspace_stub);
    m.impl("logspace", logspace_stub);
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_kernel);
    m.impl("randn", randn_stub);
    m.impl("randn.generator", randn_generator_stub);
    m.impl("randint", randint_stub);
    m.impl("randperm", randperm_stub);
    m.impl("rand_like", rand_like_kernel);
    m.impl("randint_like", randint_like_kernel);
    m.impl("randn_like", randn_like_kernel);
    m.impl("empty_like", empty_like_kernel);
    m.impl("zeros_like", zeros_like_kernel);
    m.impl("ones_like", ones_like_kernel);
    m.impl("full_like", full_like_kernel);
}

} // namespace cpu
} // namespace tensorplay
