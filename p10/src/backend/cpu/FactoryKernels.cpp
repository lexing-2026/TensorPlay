#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "Scalar.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>

namespace tensorplay {
namespace cpu {

Tensor& fill_kernel(Tensor& self, Scalar value);

Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
         TP_THROW(NotImplementedError, "rand() only supports Float32 for now");
    }
    return t;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    size_t nbytes = t.numel() * t.itemsize();
    if (t.data_ptr()) {
        std::memset(t.data_ptr(), 0, nbytes);
    }
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    fill_kernel(t, 1);
    return t;
}

Tensor full_kernel(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device) {
    DType inferred_dtype = dtype;
    if (inferred_dtype == DType::Undefined) {
        inferred_dtype = fill_value.dtype();
    }
    Tensor t(size, inferred_dtype, device);
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
            dtype = DType::Float32;
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

Tensor empty_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    return t;
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
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        std::normal_distribution<float> dist(0.0f, 1.0f);
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
         TP_THROW(NotImplementedError, "randn() only supports Float32 for now");
    }
    return t;
}

Tensor randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    int64_t n = t.numel();
    auto& gen = default_generator().engine();
    
    if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        std::uniform_int_distribution<int64_t> dist(low, high - 1); 
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = t.data_ptr<int32_t>();
        std::uniform_int_distribution<int32_t> dist((int32_t)low, (int32_t)high - 1);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<float>(dist(gen));
        }
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        std::uniform_int_distribution<int64_t> dist(low, high - 1);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<double>(dist(gen));
        }
    } else {
         TP_THROW(NotImplementedError, "randint() only supports Int64/Int32/Float32/Float64");
    }
    return t;
}

Tensor randperm_kernel(int64_t n, DType dtype, Device device) {
    Tensor t({n}, dtype, device);
    if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) data[i] = i;
        std::shuffle(data, data + n, default_generator().engine());
    } else {
        TP_THROW(NotImplementedError, "randperm() only supports Int64");
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
    return empty_kernel(static_cast<std::vector<int64_t>>(self.shape()), dtype, dev);
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

Tensor full_like_kernel(const Tensor& self, Scalar fill_value, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device dev = device.has_value() ? *device : self.device();
    return full_kernel(static_cast<std::vector<int64_t>>(self.shape()), fill_value, dtype, dev);
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
