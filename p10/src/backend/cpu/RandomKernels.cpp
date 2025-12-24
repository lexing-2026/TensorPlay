#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "Exception.h"
#include <random>

namespace tensorplay {
namespace cpu {

Tensor bernoulli_kernel(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    
    if (self.dtype() == DType::Float32) {
        const float* inp = self.data_ptr<float>();
        float* res = out.data_ptr<float>();
        int64_t n = self.numel();
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            std::bernoulli_distribution dist(inp[i]);
            res[i] = dist(gen) ? 1.0f : 0.0f;
        }
    } else {
        TP_THROW(NotImplementedError, "bernoulli only supports Float32 inputs");
    }
    return out;
}

Tensor normal_kernel(const Tensor& mean, const Tensor& std) {
    if (mean.shape() != std.shape()) {
        TP_THROW(RuntimeError, "normal: mean and std must have same size (broadcasting not implemented yet)");
    }
    Tensor out(static_cast<std::vector<int64_t>>(mean.shape()), mean.dtype(), mean.device());
    
    if (mean.dtype() == DType::Float32 && std.dtype() == DType::Float32) {
        const float* m_data = mean.data_ptr<float>();
        const float* s_data = std.data_ptr<float>();
        float* out_data = out.data_ptr<float>();
        int64_t n = mean.numel();
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            std::normal_distribution<float> dist(m_data[i], s_data[i]);
            out_data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "normal only supports Float32");
    }
    return out;
}

Tensor poisson_kernel(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    
    if (self.dtype() == DType::Float32) {
        const float* inp = self.data_ptr<float>();
        float* res = out.data_ptr<float>();
        int64_t n = self.numel();
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            std::poisson_distribution<int> dist(inp[i]); 
            res[i] = static_cast<float>(dist(gen));
        }
    } else {
        TP_THROW(NotImplementedError, "poisson only supports Float32 inputs");
    }
    return out;
}

// In-place kernels
// Note: Must take Tensor& and return Tensor& to match DispatchStub signature for Tensor(a!)

Tensor& bernoulli_inplace_kernel(Tensor& self) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            std::bernoulli_distribution dist(data[i]);
            data[i] = dist(gen) ? 1.0f : 0.0f;
        }
    } else {
        TP_THROW(NotImplementedError, "bernoulli_ only supports Float32 inputs (as probabilities)");
    }
    return self;
}

Tensor& cauchy_kernel(Tensor& self, double median, double sigma) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::cauchy_distribution<float> dist(static_cast<float>(median), static_cast<float>(sigma));
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "cauchy_ only supports Float32");
    }
    return self;
}

Tensor& exponential_kernel(Tensor& self, double lambd) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::exponential_distribution<float> dist(static_cast<float>(lambd));
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "exponential_ only supports Float32");
    }
    return self;
}

Tensor& geometric_kernel(Tensor& self, double p) {
    // PyTorch geometric returns number of trials to get first success (1, 2, ...).
    // std::geometric_distribution returns number of failures before first success (0, 1, ...).
    // So we add 1.
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::geometric_distribution<int> dist(p);
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<float>(dist(gen) + 1);
        }
    } else if (self.dtype() == DType::Int64) {
        int64_t* data = self.data_ptr<int64_t>();
        int64_t n = self.numel();
        std::geometric_distribution<int64_t> dist(p);
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen) + 1;
        }
    } else {
        TP_THROW(NotImplementedError, "geometric_ only supports Float32/Int64");
    }
    return self;
}

Tensor& log_normal_kernel(Tensor& self, double mean, double std) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::lognormal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std));
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "log_normal_ only supports Float32");
    }
    return self;
}

Tensor& normal_inplace_kernel(Tensor& self, double mean, double std) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::normal_distribution<float> dist(static_cast<float>(mean), static_cast<float>(std));
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "normal_ only supports Float32");
    }
    return self;
}

Tensor& random_kernel(Tensor& self, int64_t low, int64_t high) {
    if (high <= low) {
        TP_THROW(RuntimeError, "random_: high must be greater than low");
    }
    
    int64_t max_val = high - 1;
    
    auto& gen = default_generator().engine();
    
    if (self.dtype() == DType::Int64) {
        int64_t* data = self.data_ptr<int64_t>();
        int64_t n = self.numel();
        std::uniform_int_distribution<int64_t> dist(low, max_val);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else if (self.dtype() == DType::Int32) {
        int32_t* data = self.data_ptr<int32_t>();
        int64_t n = self.numel();
        std::uniform_int_distribution<int32_t> dist((int32_t)low, (int32_t)max_val);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::uniform_int_distribution<int64_t> dist(low, max_val);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<float>(dist(gen));
        }
    } else {
        TP_THROW(NotImplementedError, "random_ only supports Int64/Int32/Float32");
    }
    return self;
}

Tensor& uniform_kernel(Tensor& self, double from, double to) {
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        int64_t n = self.numel();
        std::uniform_real_distribution<float> dist(static_cast<float>(from), static_cast<float>(to));
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
        TP_THROW(NotImplementedError, "uniform_ only supports Float32");
    }
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, RandomKernels) {
    m.impl("bernoulli", bernoulli_kernel);
    m.impl("normal", normal_kernel);
    m.impl("poisson", poisson_kernel);
    m.impl("bernoulli_", bernoulli_inplace_kernel);
    m.impl("cauchy_", cauchy_kernel);
    m.impl("exponential_", exponential_kernel);
    m.impl("geometric_", geometric_kernel);
    m.impl("log_normal_", log_normal_kernel);
    m.impl("normal_", normal_inplace_kernel);
    m.impl("random_", random_kernel);
    m.impl("uniform_", uniform_kernel);
}

} // namespace cpu
} // namespace tensorplay
