#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "Exception.h"
#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

// Port of torch's sample_poisson (adapted from Numpy's distributions.c):
// transformed rejection for lambda >= 10, multiplication (Knuth) otherwise.
int64_t sample_poisson(double lambda, Generator* generator) {
    uniform_real_distribution<double> standard_uniform(0.0, 1.0);
    if (lambda >= 10) {
        // transformed rejection method, (Hoermann, 1993)
        double slam = std::sqrt(lambda);
        double loglam = std::log(lambda);
        double b = 0.931 + 2.53 * slam;
        double a = -0.059 + 0.02483 * b;
        double invalpha = 1.1239 + 1.1328 / (b - 3.4);
        double vr = 0.9277 - 3.6224 / (b - 2);

        while (true) {
            double U = standard_uniform(generator) - 0.5;
            double V = standard_uniform(generator);
            double us = 0.5 - std::fabs(U);
            auto k = std::floor((2 * a / us + b) * U + lambda + 0.43);
            if ((us >= 0.07) && (V <= vr)) {
                return static_cast<int64_t>(k);
            }
            if ((k < 0) || ((us < 0.013) && (V > us))) {
                continue;
            }
            if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
                (-lambda + k * loglam - std::lgamma(k + 1))) {
                return static_cast<int64_t>(k);
            }
        }
    } else if (lambda == 0) {
        return 0;
    } else {
        auto enlam = std::exp(-lambda);
        int64_t X = 0;
        auto prod = 1.0;
        while (true) {
            auto U = standard_uniform(generator);
            prod *= U;
            if (prod > enlam) {
                X += 1;
            } else {
                return X;
            }
        }
    }
}

// Dispatch over floating dtypes torch's distribution kernels support
// (AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16)).
template <typename Func>
void dispatch_floating(DType dtype, Func&& fn) {
    switch (dtype) {
        case DType::Float32: fn(float{}); break;
        case DType::Float64: fn(double{}); break;
        case DType::Float16: fn(Half{}); break;
        case DType::BFloat16: fn(BFloat16{}); break;
        default:
            TP_THROW(NotImplementedError, "distribution only supports floating dtypes");
    }
}

// Dispatch over all dtypes torch's random_/geometric_ kernels support
// (AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16) plus Bool).
template <typename Func>
void dispatch_all(DType dtype, Func&& fn) {
    switch (dtype) {
        case DType::UInt8: fn(uint8_t{}); break;
        case DType::Int8: fn(int8_t{}); break;
        case DType::Int16: fn(int16_t{}); break;
        case DType::Int32: fn(int32_t{}); break;
        case DType::Int64: fn(int64_t{}); break;
        case DType::UInt16: fn(uint16_t{}); break;
        case DType::UInt32: fn(uint32_t{}); break;
        case DType::UInt64: fn(uint64_t{}); break;
        case DType::Float32: fn(float{}); break;
        case DType::Float64: fn(double{}); break;
        case DType::Float16: fn(Half{}); break;
        case DType::BFloat16: fn(BFloat16{}); break;
        case DType::Bool: fn(bool{}); break;
        default:
            TP_THROW(NotImplementedError, "distribution does not support this dtype");
    }
}

} // namespace

Tensor bernoulli_kernel(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    auto& gen = default_generator();

    if (self.dtype() == DType::Float32) {
        const float* inp = self.data_ptr<float>();
        float* res = out.data_ptr<float>();
        uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int64_t i = 0; i < n; ++i) {
            res[i] = uniform(&gen) < static_cast<double>(inp[i]) ? 1.0f : 0.0f;
        }
    } else if (self.dtype() == DType::Float64) {
        const double* inp = self.data_ptr<double>();
        double* res = out.data_ptr<double>();
        uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int64_t i = 0; i < n; ++i) {
            res[i] = uniform(&gen) < inp[i] ? 1.0 : 0.0;
        }
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        // Probabilities are read in float precision; output keeps self dtype.
        if (self.dtype() == DType::Float16) {
            const Half* inp = self.data_ptr<Half>();
            Half* res = out.data_ptr<Half>();
            uniform_real_distribution<double> uniform(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                res[i] = static_cast<Half>(uniform(&gen) < static_cast<double>(inp[i]) ? 1.0f : 0.0f);
            }
        } else {
            const BFloat16* inp = self.data_ptr<BFloat16>();
            BFloat16* res = out.data_ptr<BFloat16>();
            uniform_real_distribution<double> uniform(0.0, 1.0);
            for (int64_t i = 0; i < n; ++i) {
                res[i] = static_cast<BFloat16>(uniform(&gen) < static_cast<double>(inp[i]) ? 1.0f : 0.0f);
            }
        }
    } else {
        TP_THROW(NotImplementedError, "bernoulli only supports floating dtype inputs");
    }
    return out;
}

Tensor normal_kernel(const Tensor& mean, const Tensor& std) {
    if (mean.shape() != std.shape()) {
        TP_THROW(RuntimeError, "normal: mean and std must have same size (broadcasting not implemented yet)");
    }
    Tensor out(static_cast<std::vector<int64_t>>(mean.shape()), mean.dtype(), mean.device());
    int64_t n = mean.numel();
    auto& gen = default_generator();
    if (mean.dtype() != std.dtype()) {
        TP_THROW(RuntimeError, "normal: mean and std must have the same dtype");
    }

    dispatch_floating(mean.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* m_data = mean.data_ptr<scalar_t>();
        const scalar_t* s_data = std.data_ptr<scalar_t>();
        scalar_t* out_data = out.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            normal_distribution<double> dist(static_cast<double>(m_data[i]),
                                             static_cast<double>(s_data[i]));
            out_data[i] = static_cast<scalar_t>(dist(&gen));
        }
    });
    return out;
}

Tensor poisson_kernel(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    auto& gen = default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* inp = self.data_ptr<scalar_t>();
        scalar_t* res = out.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            res[i] = static_cast<scalar_t>(sample_poisson(static_cast<double>(inp[i]), &gen));
        }
    });
    return out;
}

// In-place kernels
// Note: Must take Tensor& and return Tensor& to match DispatchStub signature for Tensor(a!)

Tensor& bernoulli_inplace_kernel(Tensor& self) {
    int64_t n = self.numel();
    auto& gen = default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(
                uniform(&gen) < static_cast<double>(data[i]) ? 1.0 : 0.0);
        }
    });
    return self;
}

Tensor& cauchy_kernel(Tensor& self, double median, double sigma) {
    int64_t n = self.numel();
    auto& gen = default_generator();
    cauchy_distribution<double> dist(median, sigma);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(dist(&gen));
        }
    });
    return self;
}

Tensor& exponential_kernel(Tensor& self, double lambd) {
    int64_t n = self.numel();
    auto& gen = default_generator();
    exponential_distribution<double> dist(lambd);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(dist(&gen));
        }
    });
    return self;
}

Tensor& geometric_kernel(Tensor& self, double p) {
    int64_t n = self.numel();
    auto& gen = default_generator();
    geometric_distribution<double> dist(p);

    dispatch_all(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(dist(&gen));
        }
    });
    return self;
}

Tensor& log_normal_kernel(Tensor& self, double mean, double std) {
    int64_t n = self.numel();
    auto& gen = default_generator();
    lognormal_distribution<double> dist(mean, std);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<scalar_t>(dist(&gen));
        }
    });
    return self;
}

Tensor& normal_inplace_kernel(Tensor& self, double mean, double std) {
    int64_t size = self.numel();
    auto& gen = default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        using math_t = opmath_t<scalar_t>;
        scalar_t* data = self.data_ptr<scalar_t>();
        if constexpr (std::is_same_v<scalar_t, math_t>) {
            if (size >= 16 && self.is_contiguous()) {
                normal_fill<math_t>(data, size, static_cast<math_t>(mean),
                                    static_cast<math_t>(std), &gen);
            } else {
                normal_distribution<double> dist(mean, std);
                for (int64_t i = 0; i < size; ++i) {
                    data[i] = static_cast<scalar_t>(dist(&gen));
                }
            }
        } else {
            // Half/BFloat16: sample in float precision through a stack buffer.
            if (size >= 16 && self.is_contiguous()) {
                normal_fill_cast<scalar_t>(data, size, mean, std, &gen);
            } else {
                normal_distribution<double> dist(mean, std);
                for (int64_t i = 0; i < size; ++i) {
                    data[i] = static_cast<scalar_t>(dist(&gen));
                }
            }
        }
    });
    return self;
}

Tensor& random_kernel(Tensor& self, int64_t low, int64_t high) {
    if (high <= low) {
        TP_THROW(RuntimeError, "random_: high must be greater than low");
    }
    const uint64_t range = static_cast<uint64_t>(high - low);
    const int64_t base = low;
    int64_t n = self.numel();
    auto& gen = default_generator();

    dispatch_all(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* data = self.data_ptr<scalar_t>();
        uniform_int_from_to_distribution<scalar_t> dist(range, base);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(&gen);
        }
    });
    return self;
}

Tensor& uniform_kernel(Tensor& self, double from, double to) {
    int64_t n = self.numel();
    auto& gen = default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        using math_t = opmath_t<scalar_t>;
        scalar_t* data = self.data_ptr<scalar_t>();
        const math_t lo = static_cast<math_t>(from);
        const math_t hi = static_cast<math_t>(to);
        const scalar_t to_scalar = static_cast<scalar_t>(to);
        const scalar_t from_scalar = static_cast<scalar_t>(from);
        uniform_real_distribution<math_t> dist(lo, hi);
        for (int64_t i = 0; i < n; ++i) {
            math_t value = static_cast<math_t>(dist(&gen));
            // Clamp if the cast rounded up to the upper bound.
            data[i] = static_cast<scalar_t>(value) == to_scalar ? from_scalar
                                                                : static_cast<scalar_t>(value);
        }
    });
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
