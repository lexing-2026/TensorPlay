#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "DistributionDispatch.h"
#include "TensorIterator.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <utility>

namespace tensorplay {
namespace cpu {

namespace {

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

// In-place sampling requires each destination element to be independently
// addressable: a stride-0 dimension with more than one element aliases the
// whole dimension and would make the draw order observable.
void check_writable_inplace(const Tensor& t) {
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    const auto strides = t.strides();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (strides[i] == 0 && sizes[i] > 1) {
            TP_THROW(RuntimeError,
                     "unsupported operation: more than one element of the written-to tensor "
                     "refers to a single memory location. Please clone() the tensor before "
                     "performing the operation.");
        }
    }
}

// Row-major traversal (last dimension fastest) over an arbitrary strided
// layout, invoking fn with a reference to each element.  Contiguous tensors
// take a flat fast path; strided tensors recurse dimension by dimension so
// expanded/transposed views stay in bounds.  The pointer's constness decides
// whether fn receives mutable references.
template <typename scalar_t, typename Func>
void strided_for_each(scalar_t* data, const std::vector<int64_t>& sizes,
                      const std::vector<int64_t>& strides, int64_t dim, Func&& fn) {
    if (dim == static_cast<int64_t>(sizes.size())) {
        fn(*data);
        return;
    }
    const int64_t count = sizes[dim];
    const int64_t stride = strides[dim];
    for (int64_t i = 0; i < count; ++i) {
        strided_for_each(data + i * stride, sizes, strides, dim + 1, fn);
    }
}

template <typename scalar_t, typename Func>
void for_each_element(Tensor& t, Func&& fn) {
    scalar_t* data = t.data_ptr<scalar_t>();
    if (t.is_contiguous()) {
        const int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) {
            fn(data[i]);
        }
        return;
    }
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    const auto strides = t.strides();
    strided_for_each(data, sizes, strides, 0, fn);
}

template <typename scalar_t, typename Func>
void for_each_element_const(const Tensor& t, Func&& fn) {
    const scalar_t* data = t.data_ptr<scalar_t>();
    if (t.is_contiguous()) {
        const int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) {
            fn(data[i]);
        }
        return;
    }
    const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
    const auto strides = t.strides();
    strided_for_each(data, sizes, strides, 0, fn);
}

// Lockstep row-major traversal of two same-shape tensors, invoking fn with
// one element of each.
template <typename scalar_t, typename Func>
void strided_for_each_pair(const scalar_t* a, const std::vector<int64_t>& strides_a,
                           const scalar_t* b, const std::vector<int64_t>& strides_b,
                           const std::vector<int64_t>& sizes, int64_t dim, Func&& fn) {
    if (dim == static_cast<int64_t>(sizes.size())) {
        fn(*a, *b);
        return;
    }
    for (int64_t i = 0; i < sizes[dim]; ++i) {
        strided_for_each_pair(a + i * strides_a[dim], strides_a,
                              b + i * strides_b[dim], strides_b, sizes, dim + 1, fn);
    }
}

template <typename scalar_t, typename Func>
void for_each_element_pair(const Tensor& a, const Tensor& b, Func&& fn) {
    const scalar_t* pa = a.data_ptr<scalar_t>();
    const scalar_t* pb = b.data_ptr<scalar_t>();
    if (a.is_contiguous() && b.is_contiguous()) {
        const int64_t n = a.numel();
        for (int64_t i = 0; i < n; ++i) {
            fn(pa[i], pb[i]);
        }
        return;
    }
    const auto sizes = static_cast<std::vector<int64_t>>(a.shape());
    strided_for_each_pair(pa, a.strides(), pb, b.strides(), sizes, 0, fn);
}

} // namespace

template <typename prob_t>
void validate_bernoulli_probabilities(const Tensor& probabilities) {
    for_each_element_const<prob_t>(probabilities, [](const prob_t& value) {
        const double p = static_cast<double>(value);
        if (!(p >= 0.0 && p <= 1.0)) {
            TP_THROW(ValueError,
                     "bernoulli_ expects probability values in [0, 1]");
        }
    });
}

Tensor bernoulli_kernel(const Tensor& self, std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    Generator& gen = generator.has_value() ? *generator : default_generator();

    if (self.dtype() == DType::Float32) {
        validate_bernoulli_probabilities<float>(self);
        // rand() (24-bit mantissa mask) and compares strictly against p.
        float* res = out.data_ptr<float>();
        for_each_element_const<float>(self, [&](const float& p) {
            const uint32_t r = gen.random();
            const double u = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
            *res++ = u < static_cast<double>(p) ? 1.0f : 0.0f;
        });
    } else if (self.dtype() == DType::Float64) {
        validate_bernoulli_probabilities<double>(self);
        double* res = out.data_ptr<double>();
        uniform_real_distribution<double> uniform(0.0, 1.0);
        for_each_element_const<double>(self, [&](const double& p) {
            *res++ = uniform(&gen) < p ? 1.0 : 0.0;
        });
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        // Probabilities are read in float precision; output keeps self dtype.
        if (self.dtype() == DType::Float16) {
            validate_bernoulli_probabilities<Half>(self);
            Half* res = out.data_ptr<Half>();
            uniform_real_distribution<double> uniform(0.0, 1.0);
            for_each_element_const<Half>(self, [&](const Half& p) {
                *res++ = static_cast<Half>(uniform(&gen) < static_cast<double>(p) ? 1.0f : 0.0f);
            });
        } else {
            validate_bernoulli_probabilities<BFloat16>(self);
            BFloat16* res = out.data_ptr<BFloat16>();
            uniform_real_distribution<double> uniform(0.0, 1.0);
            for_each_element_const<BFloat16>(self, [&](const BFloat16& p) {
                *res++ = static_cast<BFloat16>(uniform(&gen) < static_cast<double>(p) ? 1.0f : 0.0f);
            });
        }
    } else {
        TP_THROW(NotImplementedError, "bernoulli only supports floating dtype inputs");
    }
    return out;
}

template <typename output_t, typename prob_t>
void bernoulli_tensor_loop(TensorIterator& iter, Generator* generator) {
    uniform_real_distribution<double> uniform(0.0, 1.0);
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
        auto* output = reinterpret_cast<output_t*>(data[0]);
        const auto* probabilities = reinterpret_cast<const prob_t*>(data[1]);
        for (int64_t i = 0; i < n; ++i) {
            const auto* probability = reinterpret_cast<const prob_t*>(
                reinterpret_cast<const char*>(probabilities) + i * strides[1]);
            auto* value = reinterpret_cast<output_t*>(
                reinterpret_cast<char*>(output) + i * strides[0]);
            *value = static_cast<output_t>(
                uniform(generator) < static_cast<double>(*probability));
        }
    });
}

template <typename output_t>
void bernoulli_tensor_loop_for_probability_dtype(
        TensorIterator& iter, const Tensor& probabilities, Generator* generator) {
    switch (probabilities.dtype()) {
        case DType::Float32:
            bernoulli_tensor_loop<output_t, float>(iter, generator);
            return;
        case DType::Float64:
            bernoulli_tensor_loop<output_t, double>(iter, generator);
            return;
        case DType::Float16:
            bernoulli_tensor_loop<output_t, Half>(iter, generator);
            return;
        case DType::BFloat16:
            bernoulli_tensor_loop<output_t, BFloat16>(iter, generator);
            return;
        default:
            TP_THROW(TypeError,
                     "bernoulli_ probability tensor must have a floating dtype");
    }
}

Tensor& bernoulli_tensor_inplace_kernel(
        Tensor& self, const Tensor& probabilities,
        std::optional<Generator> generator) {
    if (self.device() != probabilities.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli_: probability tensor must be on the same device");
    }
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    switch (probabilities.dtype()) {
        case DType::Float32:
            validate_bernoulli_probabilities<float>(probabilities);
            break;
        case DType::Float64:
            validate_bernoulli_probabilities<double>(probabilities);
            break;
        case DType::Float16:
            validate_bernoulli_probabilities<Half>(probabilities);
            break;
        case DType::BFloat16:
            validate_bernoulli_probabilities<BFloat16>(probabilities);
            break;
        default:
            TP_THROW(TypeError,
                     "bernoulli_ probability tensor must have a floating dtype");
    }

    TensorIterator iter = TensorIteratorConfig()
        .add_output(self)
        .add_const_input(probabilities)
        .check_all_same_dtype(false)
        .build();
    Generator& gen = generator.has_value() ? *generator : default_generator();
#define TP_BERNOULLI_OUTPUT_CASE(ctype, name) \
    case DType::name: \
        bernoulli_tensor_loop_for_probability_dtype<ctype>(iter, probabilities, &gen); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_OUTPUT_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_OUTPUT_CASE
    return self;
}

Tensor& bernoulli_out_kernel(const Tensor& self,
                             std::optional<Generator> generator,
                             Tensor& out) {
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli: output must be on the same device as input");
    }
    tpx::ops::resize_(out, static_cast<std::vector<int64_t>>(self.shape()));
    return bernoulli_tensor_inplace_kernel(out, self, std::move(generator));
}

Tensor& bernoulli_scalar_inplace_kernel(
        Tensor& self, double p, std::optional<Generator> generator) {
    if (!(p >= 0.0 && p <= 1.0)) {
        TP_THROW(ValueError, "bernoulli_ expects p to be in [0, 1]");
    }
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    Generator& gen = generator.has_value() ? *generator : default_generator();
    uniform_real_distribution<double> uniform(0.0, 1.0);
#define TP_BERNOULLI_SCALAR_CASE(ctype, name) \
    case DType::name: \
        for_each_element<ctype>(self, [&](ctype& value) { \
            value = static_cast<ctype>(uniform(&gen) < p); \
        }); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_SCALAR_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_SCALAR_CASE
    return self;
}

Tensor bernoulli_p_kernel(const Tensor& self, double p,
                          std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()),
               self.dtype(), self.device());
    bernoulli_scalar_inplace_kernel(out, p, std::move(generator));
    return out;
}

Tensor normal_kernel(const Tensor& mean, const Tensor& std) {
    if (mean.shape() != std.shape()) {
        TP_THROW(RuntimeError, "normal: mean and std must have same size (broadcasting not implemented yet)");
    }
    Tensor out(static_cast<std::vector<int64_t>>(mean.shape()), mean.dtype(), mean.device());
    auto& gen = default_generator();
    if (mean.dtype() != std.dtype()) {
        TP_THROW(RuntimeError, "normal: mean and std must have the same dtype");
    }

    dispatch_floating(mean.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* out_data = out.data_ptr<scalar_t>();
        int64_t k = 0;
        for_each_element_pair<scalar_t>(mean, std, [&](const scalar_t& m, const scalar_t& s) {
            normal_distribution<double> dist(static_cast<double>(m),
                                             static_cast<double>(s));
            out_data[k++] = static_cast<scalar_t>(dist(&gen));
        });
    });
    return out;
}

Tensor poisson_kernel(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    auto& gen = default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* res = out.data_ptr<scalar_t>();
        int64_t k = 0;
        for_each_element_const<scalar_t>(self, [&](const scalar_t& p) {
            res[k++] = static_cast<scalar_t>(sample_poisson(static_cast<double>(p), &gen));
        });
    });
    return out;
}

// In-place kernels
// Note: Must take Tensor& and return Tensor& to match DispatchStub signature for Tensor(a!)

Tensor& bernoulli_inplace_kernel(Tensor& self) {
    auto& gen = default_generator();
    if (self.numel() == 0) return self;
    check_writable_inplace(self);

    if (self.dtype() == DType::Float32) {
        // rand() (24-bit mantissa mask) and compares strictly against p.
        for_each_element<float>(self, [&](float& v) {
            const uint32_t r = gen.random();
            const double u = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
            v = u < static_cast<double>(v) ? 1.0f : 0.0f;
        });
        return self;
    }

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        uniform_real_distribution<double> uniform(0.0, 1.0);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = static_cast<scalar_t>(uniform(&gen) < static_cast<double>(v) ? 1.0 : 0.0);
        });
    });
    return self;
}

Tensor& cauchy_kernel(Tensor& self, double median, double sigma) {
    auto& gen = default_generator();
    TP_THROW_IF(sigma <= 0.0, RuntimeError, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    cauchy_distribution<double> dist(median, sigma);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = static_cast<scalar_t>(dist(&gen));
        });
    });
    return self;
}

Tensor& exponential_kernel(Tensor& self, double lambd) {
    auto& gen = default_generator();
    TP_THROW_IF(lambd <= 0.0, RuntimeError, "exponential_ expects lambda > 0.0, but found lambda=", lambd);
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    exponential_distribution<double> dist(lambd);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = static_cast<scalar_t>(dist(&gen));
        });
    });
    return self;
}

Tensor& geometric_kernel(Tensor& self, double p) {
    auto& gen = default_generator();
    TP_THROW_IF(!(0.0 < p && p < 1.0), RuntimeError, "geometric_ expects p to be in (0, 1), but got p=", p);
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    geometric_distribution<double> dist(p);

    dispatch_all(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = static_cast<scalar_t>(dist(&gen));
        });
    });
    return self;
}

Tensor& log_normal_kernel(Tensor& self, double mean, double std) {
    auto& gen = default_generator();
    TP_THROW_IF(std <= 0.0, RuntimeError, "log_normal_ expects std > 0.0, but found std=", std);
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    lognormal_distribution<double> dist(mean, std);

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = static_cast<scalar_t>(dist(&gen));
        });
    });
    return self;
}

Tensor& normal_inplace_kernel(Tensor& self, double mean, double std,
                              std::optional<Generator> generator) {
    Generator& gen = generator.has_value() ? *generator : default_generator();
    TP_THROW_IF(std < 0.0, RuntimeError, "normal expects std >= 0.0, but found std ", std);
    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    const int64_t size = self.numel();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        using math_t = opmath_t<scalar_t>;
        if constexpr (std::is_same_v<scalar_t, math_t>) {
            if (size >= 16 && self.is_contiguous()) {
                scalar_t* data = self.data_ptr<scalar_t>();
                normal_fill<math_t>(data, size, static_cast<math_t>(mean),
                                    static_cast<math_t>(std), &gen);
            } else {
                normal_distribution<double> dist(mean, std);
                for_each_element<scalar_t>(self, [&](scalar_t& v) {
                    v = static_cast<scalar_t>(dist(&gen));
                });
            }
        } else {
            // Half/BFloat16 draw uniforms in opmath (float) through a
            // 16-element stack buffer, Box-Muller in float, then cast down
            // to the storage dtype -- including the inplace entrypoint.
            if (size >= 16 && self.is_contiguous()) {
                scalar_t* data = self.data_ptr<scalar_t>();
                normal_fill_cast<scalar_t>(data, size, mean, std, &gen);
            } else {
                normal_distribution<double> dist(mean, std);
                for_each_element<scalar_t>(self, [&](scalar_t& v) {
                    v = static_cast<scalar_t>(dist(&gen));
                });
            }
        }
    });
    return self;
}

Tensor& random_kernel(Tensor& self, int64_t low, int64_t high) {
    TP_THROW_IF(high <= low, RuntimeError,
                "random_ expects 'from' to be less than 'to', but got from=",
                low, " >= to=", high);
    auto& gen = default_generator();

    distribution::check_random_from_to_bounds(low, high, self.dtype());

    if (self.numel() == 0) return self;
    check_writable_inplace(self);

    const uint64_t range = static_cast<uint64_t>(high - low);
    const int64_t base = low;
    dispatch_all(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        uniform_int_from_to_distribution<scalar_t> dist(range, base);
        for_each_element<scalar_t>(self, [&](scalar_t& v) {
            v = dist(&gen);
        });
    });
    return self;
}

Tensor& uniform_kernel(Tensor& self, double from, double to,
                       std::optional<Generator> generator) {
    // Complex tensors fill both components: recurse over the interleaved
    // real view, which keeps the [from, to) contract per component.
    if (isComplexType(self.dtype())) {
        Tensor real_view = tpx::ops::view_as_real(self);
        return uniform_kernel(real_view, from, to, std::move(generator));
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(NotImplementedError, "\"check_uniform_bounds\" not implemented for '",
                 toString(self.dtype()), "'");
    }

    const char* dtype_name = distribution::bounds_dtype_name(self.dtype());
    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const double min = distribution::fp_dtype_lowest<scalar_t>();
        const double max = distribution::fp_dtype_max<scalar_t>();
        TP_THROW_IF(!(from >= min && from <= max), RuntimeError,
                    "from is out of bounds for ", dtype_name);
        TP_THROW_IF(!(to >= min && to <= max), RuntimeError,
                    "to is out of bounds for ", dtype_name);
        TP_THROW_IF(from > to, RuntimeError,
                    "uniform_ expects to return a [from, to) range, but found from=",
                    from, " > to=", to);
        TP_THROW_IF((to - from) > distribution::fp_dtype_max<scalar_t>(), RuntimeError,
                    "uniform_ expects to-from <= std::numeric_limits<", dtype_name,
                    ">::max(), but found to=", to, " and from=", from,
                    " which result in to-from to exceed the limit");
        from = std::clamp(from, min, max);
        to = std::clamp(to, min, max);
    });

    if (self.numel() == 0) return self;
    check_writable_inplace(self);
    auto& gen = generator.has_value() ? *generator : default_generator();

    dispatch_floating(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        if constexpr (std::is_same_v<scalar_t, float>) {
            // 24-bit mantissa draw scaled to [from, to); the product
            // accumulates in double before storing back.
            for_each_element<scalar_t>(self, [&](scalar_t& v) {
                const uint32_t r = gen.random();
                const double x = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
                v = static_cast<scalar_t>(x * (to - from) + from);
            });
        } else {
            // Half/BFloat16 sample in opmath_t (float, 24-bit mantissa mask)
            // and cast to the storage dtype, clamping a cast that rounded up
            // to the upper bound back to 'from'.
            using math_t = opmath_t<scalar_t>;
            uniform_real_distribution<math_t> dist(
                static_cast<math_t>(from), static_cast<math_t>(to));
            const scalar_t to_scalar = static_cast<scalar_t>(to);
            const scalar_t from_scalar = static_cast<scalar_t>(from);
            for_each_element<scalar_t>(self, [&](scalar_t& v) {
                scalar_t value = static_cast<scalar_t>(dist(&gen));
                v = value == to_scalar ? from_scalar : value;
            });
        }
    });
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, RandomKernels) {
    m.impl("bernoulli", bernoulli_kernel);
    m.impl("bernoulli.out", bernoulli_out_kernel);
    m.impl("bernoulli.p", bernoulli_p_kernel);
    m.impl("normal", normal_kernel);
    m.impl("poisson", poisson_kernel);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel);
    m.impl("bernoulli_.float", bernoulli_scalar_inplace_kernel);
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
