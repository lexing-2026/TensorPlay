#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
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
            // Sample in float precision and cast down, like torch's
            // opmath-based uniform_ path (with the round-up clamp).
            if (dtype == DType::Float16) {
                Half* data = t.data_ptr<Half>();
                uniform_real_distribution<float> dist(0.0f, 1.0f);
                const Half to_scalar = static_cast<Half>(1.0f);
                for (int64_t i = 0; i < n; ++i) {
                    float value = static_cast<float>(dist(&gen));
                    Half casted = static_cast<Half>(value);
                    data[i] = casted == to_scalar ? static_cast<Half>(0.0f) : casted;
                }
            } else {
                BFloat16* data = t.data_ptr<BFloat16>();
                uniform_real_distribution<float> dist(0.0f, 1.0f);
                const BFloat16 to_scalar = static_cast<BFloat16>(1.0f);
                for (int64_t i = 0; i < n; ++i) {
                    float value = static_cast<float>(dist(&gen));
                    BFloat16 casted = static_cast<BFloat16>(value);
                    data[i] = casted == to_scalar ? static_cast<BFloat16>(0.0f) : casted;
                }
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "rand() only supports floating dtypes for now");
    }
    return t;
}

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

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device, bool pin_memory) {
    DType inferred_dtype = dtype;
    if (inferred_dtype == DType::Undefined) {
        inferred_dtype = fill_value.dtype();
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
    
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<float>(s_d + i * st_d);
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        // Use double accumulation to avoid overflow if possible, or int accumulation?
        // PyTorch uses accumulation in result type.
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<int64_t>(s_d + i * st_d);
    } else if (dtype == DType::Int32) {
        int32_t* data = t.data_ptr<int32_t>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<int32_t>(s_d + i * st_d);
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
    } else {
         TP_THROW(NotImplementedError, "linspace only supports Float32");
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
    } else {
         TP_THROW(NotImplementedError, "logspace only supports Float32");
    }
    return t;
}

// --- Random Factory Kernels ---

Tensor randn_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    int64_t n = t.numel();
    auto& gen = default_generator();

    switch (dtype) {
        case DType::Float32: {
            float* data = t.data_ptr<float>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<float>(data, n, 0.0f, 1.0f, &gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = static_cast<float>(dist(&gen));
                }
            }
            break;
        }
        case DType::Float64: {
            double* data = t.data_ptr<double>();
            if (n >= 16 && t.is_contiguous()) {
                normal_fill<double>(data, n, 0.0, 1.0, &gen);
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = dist(&gen);
                }
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // Sample in float precision through a stack buffer, then cast.
            if (n >= 16 && t.is_contiguous()) {
                if (dtype == DType::Float16) {
                    normal_fill_cast<Half>(t.data_ptr<Half>(), n, 0.0, 1.0, &gen);
                } else {
                    normal_fill_cast<BFloat16>(t.data_ptr<BFloat16>(), n, 0.0, 1.0, &gen);
                }
            } else {
                normal_distribution<double> dist(0.0, 1.0);
                if (dtype == DType::Float16) {
                    Half* data = t.data_ptr<Half>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<Half>(dist(&gen));
                    }
                } else {
                    BFloat16* data = t.data_ptr<BFloat16>();
                    for (int64_t i = 0; i < n; ++i) {
                        data[i] = static_cast<BFloat16>(dist(&gen));
                    }
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
    Tensor t(size, dtype, device);
    int64_t n = t.numel();
    auto& gen = default_generator();
    const uint64_t range = static_cast<uint64_t>(high - low);
    const int64_t base = low;

    if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        uniform_int_from_to_distribution<int64_t> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = t.data_ptr<int32_t>();
        uniform_int_from_to_distribution<int32_t> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    } else if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        uniform_int_from_to_distribution<float> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        uniform_int_from_to_distribution<double> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    } else {
         TP_THROW(NotImplementedError, "randint() only supports Int64/Int32/Float32/Float64");
    }
    return t;
}

Tensor randperm_kernel(int64_t n, DType dtype, Device device) {
    Tensor t({n}, dtype, device);
    if (dtype == DType::Int64 || dtype == DType::Int32) {
        // Fisher-Yates with the same draw pattern as torch's randperm_cpu:
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

TENSORPLAY_LIBRARY_IMPL(CPU, FactoryKernels) {
    m.impl("rand", rand_kernel);
    m.impl("zeros", zeros_kernel);
    m.impl("ones", ones_kernel);
    m.impl("full", full_kernel);
    m.impl("arange", arange_start_step_kernel);
    m.impl("arange.end", arange_kernel);
    m.impl("empty", empty_kernel);
    m.impl("eye", eye_kernel);
    m.impl("linspace", linspace_kernel);
    m.impl("logspace", logspace_kernel);
    m.impl("fill_.Scalar", fill_kernel);
    m.impl("zero_", zero_kernel);
    m.impl("randn", randn_kernel);
    m.impl("randint", randint_kernel);
    m.impl("randperm", randperm_kernel);
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
