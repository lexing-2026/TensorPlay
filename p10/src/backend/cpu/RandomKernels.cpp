#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "DistributionDispatch.h"
#include "TensorIterator.h"
#include "Exception.h"
#include "Utils.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <SpecialMath.h>
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
    if (!(lambda >= 0.0)) {
        TP_THROW(RuntimeError, "invalid Poisson rate, expected rate to be non-negative");
    }
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
    if (mean.device() != std.device()) {
        TP_THROW(DeviceMismatchError, "normal: mean and std must be on the same device");
    }
    if (mean.dtype() != std.dtype()) {
        TP_THROW(RuntimeError, "normal: mean and std must have the same dtype");
    }

    const std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(mean.shape()),
        static_cast<std::vector<int64_t>>(std.shape()));
    Tensor out(out_shape, mean.dtype(), mean.device());
    if (out.numel() == 0) {
        return out;
    }

    Tensor mean_broadcast = mean.expand(out_shape).contiguous();
    Tensor std_broadcast = std.expand(out_shape).contiguous();

    dispatch_floating(mean.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* out_data = out.data_ptr<scalar_t>();
        const scalar_t* mean_data = mean_broadcast.data_ptr<scalar_t>();
        const scalar_t* std_data = std_broadcast.data_ptr<scalar_t>();
        const int64_t n = out.numel();

        for (int64_t i = 0; i < n; ++i) {
            const double std_value = static_cast<double>(std_data[i]);
            if (!(std_value >= 0.0)) {
                TP_THROW(RuntimeError, "normal: standard deviation must be non-negative");
            }
        }

        auto& gen = default_generator();
        for (int64_t i = 0; i < n; ++i) {
            normal_distribution<double> dist(static_cast<double>(mean_data[i]),
                                             static_cast<double>(std_data[i]));
            out_data[i] = static_cast<scalar_t>(dist(&gen));
        }
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

// Truncated Stirling series for log(n!) - 0.5*log(2*pi*n) - n + n*log(n),
// tabulated for the first few integers and evaluated from the asymptotic
// expansion afterwards.
double stirling_approx_tail(double k) {
    static const double kTailValues[10] = {
        0.0810614667953272, 0.0413406959554092, 0.0276779256849983,
        0.02079067210376509, 0.0166446911898211, 0.0138761288230707,
        0.0118967099458917, 0.0104112652619720, 0.00925546218271273,
        0.00833056343336287};
    if (k < 10.0) {
        return kTailValues[static_cast<size_t>(k)];
    }
    const double kp1sq = (k + 1) * (k + 1);
    return (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1);
}

// Exact small-parameter binomial draw: count how many geometric variates
// with success probability `prob` fit below `count`.
double binomial_inversion(double count, double prob, Generator* gen) {
    uniform_real_distribution<double> standard_uniform(0.0, 1.0);
    const double log1mprob = std::log1p(-prob);
    double geom_sum = 0.0;
    double num_geom = 0.0;
    while (true) {
        const double u = standard_uniform(gen);
        const double geom = std::ceil(std::log(u) / log1mprob);
        geom_sum += geom;
        if (geom_sum > count) {
            break;
        }
        num_geom += 1.0;
    }
    return num_geom;
}

// Transformed rejection for the binomial law when count * prob is large;
// most draws are accepted after the squeeze test without evaluating logs.
double binomial_btrs(double count, double prob, Generator* gen) {
    uniform_real_distribution<double> standard_uniform(0.0, 1.0);
    const double stddev = std::sqrt(count * prob * (1.0 - prob));
    const double b = 1.15 + 2.53 * stddev;
    const double a = -0.0873 + 0.0248 * b + 0.01 * prob;
    const double c = count * prob + 0.5;
    const double v_r = 0.92 - 4.2 / b;
    const double r = prob / (1.0 - prob);
    const double alpha = (2.83 + 5.1 / b) * stddev;
    const double m = std::floor((count + 1.0) * prob);
    while (true) {
        const double u0 = standard_uniform(gen);
        const double v = standard_uniform(gen);
        const double u = u0 - 0.5;
        const double us = 0.5 - std::fabs(u);
        const double k = std::floor((2.0 * a / us + b) * u + c);
        if (k < 0.0 || k > count) {
            continue;
        }
        if (us >= 0.07 && v <= v_r) {
            return k;
        }
        const double vlog = std::log(v * alpha / (a / (us * us) + b));
        const double upperbound =
            (m + 0.5) * std::log((m + 1.0) / (r * (count - m + 1.0))) +
            (count + 1.0) * std::log((count - m + 1.0) / (count - k + 1.0)) +
            (k + 0.5) * std::log(r * (count - k + 1.0) / (k + 1.0)) +
            stirling_approx_tail(m) + stirling_approx_tail(count - m) -
            stirling_approx_tail(k) - stirling_approx_tail(count - k);
        if (vlog <= upperbound) {
            return k;
        }
    }
}

// Binomial draw for one (count, prob) pair.  prob > 0.5 samples the
// complementary failure count so the acceptance rate stays controlled; a
// NaN prob propagates as NaN.  Callers must guarantee count is not NaN
// (a NaN count would never terminate the geometric sum below).
double sample_binomial(double count, double prob, Generator* gen) {
    if (count != count) {
        TP_THROW(ValueError, "binomial: count must not be NaN");
    }
    if (count <= 0.0 || prob <= 0.0) {
        return 0.0;
    }
    if (prob >= 1.0) {
        return count;
    }
    if (prob != prob) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (prob <= 0.5) {
        if (count * prob >= 10.0) {
            return binomial_btrs(count, prob, gen);
        }
        return binomial_inversion(count, prob, gen);
    }
    const double qprob = 1.0 - prob;
    if (count * qprob >= 10.0) {
        return count - binomial_btrs(count, qprob, gen);
    }
    return count - binomial_inversion(count, qprob, gen);
}

// Gamma draw with shape `alpha` and unit scale: boosting handles the
// alpha < 1 regime, then a squeeze-based acceptance test on the cubic of a
// normal variate; all arithmetic stays in double regardless of storage
// dtype.  `alpha` must be finite and >= 0 (alpha == 0 yields 0).
template <typename scalar_t>
scalar_t sample_gamma(scalar_t alpha_in, Generator* gen) {
    uniform_real_distribution<double> standard_uniform(0.0, 1.0);
    normal_distribution<double> standard_normal(0.0, 1.0);
    double alpha = static_cast<double>(alpha_in);
    double scale = 1.0;
    if (alpha < 1.0) {
        if (alpha == 0.0) {
            return static_cast<scalar_t>(0.0);
        }
        scale *= std::pow(1.0 - standard_uniform(gen), 1.0 / alpha);
        alpha += 1.0;
    }
    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, y;
        do {
            x = standard_normal(gen);
            y = 1.0 + c * x;
        } while (y <= 0.0);
        const double v = y * y * y;
        const double u = 1.0 - standard_uniform(gen);
        const double xx = x * x;
        if (u < 1.0 - 0.0331 * xx * xx) {
            return static_cast<scalar_t>(scale * d * v);
        }
        if (std::log(u) < 0.5 * xx + d * (1.0 - v + std::log(v))) {
            return static_cast<scalar_t>(scale * d * v);
        }
    }
}

double standard_gamma_grad_one_cpu(double alpha, double x) {
    if (x < 0.8) {
        double numer = 1.0;
        double denom = alpha;
        double series1 = numer / denom;
        double series2 = numer / (denom * denom);
        for (int i = 1; i <= 5; ++i) {
            numer *= -x / static_cast<double>(i);
            denom += 1.0;
            series1 += numer / denom;
            series2 += numer / (denom * denom);
        }
        const double pow_x_alpha = std::pow(x, alpha);
        const double gamma_pdf = std::pow(x, alpha - 1.0) * std::exp(-x);
        const double gamma_cdf = pow_x_alpha * series1;
        const double gamma_cdf_alpha =
            (std::log(x) - tensorplay::special_math::calc_digamma(alpha)) *
                gamma_cdf -
            pow_x_alpha * series2;
        const double result = -gamma_cdf_alpha / gamma_pdf;
        return std::isnan(result) ? 0.0 : result;
    }
    if (alpha > 8.0) {
        if (0.9 * alpha <= x && x <= 1.1 * alpha) {
            const double numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
            const double numer_2 =
                1440 * (alpha * alpha) + 6 * x * (53 - 120 * x) -
                65 * x * x / alpha + alpha * (107 + 3600 * x);
            const double denom = 1244160 * (alpha * alpha) * (alpha * alpha);
            return numer_1 * numer_2 / denom;
        }
        const double denom = std::sqrt(8 * alpha);
        const double term2 = denom / (alpha - x);
        const double term3 = std::pow(
            x - alpha - alpha * std::log(x / alpha), -1.5);
        const double term23 = (x < alpha) ? term2 - term3 : term2 + term3;
        const double term1 = std::log(x / alpha) * term23 -
            std::sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
        const double stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
        const double numer = x * term1;
        return -stirling * numer / denom;
    }

    const double u = std::log(x / alpha);
    const double v = std::log(alpha);
    static const double coefficients[3][8] = {
        {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
         1, 0.32668115, 0.10406089, 0.0014179084},
        {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
         0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
        {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
         0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
    };
    double coefficient_v[8];
    for (int i = 0; i < 8; ++i) {
        coefficient_v[i] = coefficients[0][i] +
            u * (coefficients[1][i] + u * coefficients[2][i]);
    }
    const double p = coefficient_v[0] +
        v * (coefficient_v[1] + v * (coefficient_v[2] + v * coefficient_v[3]));
    const double q = coefficient_v[4] +
        v * (coefficient_v[5] + v * (coefficient_v[6] + v * coefficient_v[7]));
    return std::exp(p / q);
}

double dirichlet_grad_alpha_small_cpu(double x, double alpha, double beta) {
    const double factor = tensorplay::special_math::calc_digamma(alpha) -
        tensorplay::special_math::calc_digamma(alpha + beta) - std::log(x);
    double numer = 1.0;
    double series = numer / alpha * (factor + 1 / alpha);
    for (int i = 1; i <= 10; ++i) {
        const double casted_i = static_cast<double>(i);
        numer *= (casted_i - beta) * x / casted_i;
        const double denom = alpha + casted_i;
        series += numer / denom * (factor + 1 / denom);
    }
    const double result = -std::pow(1 - x, 1 - beta) * series;
    return std::isnan(result) ? 0.0 : result;
}

double dirichlet_grad_beta_small_cpu(double x, double alpha, double beta) {
    const double factor = tensorplay::special_math::calc_digamma(alpha + beta) -
        tensorplay::special_math::calc_digamma(beta);
    double numer = 1.0;
    double betas = 1.0;
    double dbetas = 0.0;
    double series = factor / alpha;
    for (int i = 1; i <= 8; ++i) {
        const double casted_i = static_cast<double>(i);
        numer *= -x / casted_i;
        dbetas = dbetas * (beta - casted_i) + betas;
        betas *= beta - casted_i;
        series += numer / (alpha + casted_i) * (dbetas + factor * betas);
    }
    const double result = -std::pow(1 - x, 1 - beta) * series;
    return std::isnan(result) ? 0.0 : result;
}

double dirichlet_grad_alpha_mid_cpu(double x, double alpha, double beta) {
    const double total = alpha + beta;
    const double mean = alpha / total;
    const double deviation = std::sqrt(alpha * beta / (total + 1)) / total;
    if (mean - 0.1 * deviation <= x && x <= mean + 0.1 * deviation) {
        const double polynomial =
            47 * x * (beta * beta) * (beta * beta) + alpha * (
                (43 + 20 * (16 + 27 * beta) * x) * (beta * beta) * beta + alpha * (
                    3 * (59 + 180 * beta - 90 * x) * (beta * beta) + alpha * (
                        (453 + 1620 * beta * (1 - x) - 455 * x) * beta + alpha * (
                            8 * (1 - x) * (135 * beta - 11)))));
        const double prefactor_num =
            (1 + 12 * alpha) * (1 + 12 * beta) / (total * total);
        const double prefactor_den =
            12960 * alpha * alpha * alpha * beta * beta * (1 + 12 * total);
        return prefactor_num / (1 - x) * polynomial / prefactor_den;
    }
    const double prefactor = -x / std::sqrt(2 * alpha * beta / total);
    const double stirling =
        (1 + 1 / (12 * alpha) + 1 / (288 * alpha * alpha)) *
        (1 + 1 / (12 * beta) + 1 / (288 * beta * beta)) /
        (1 + 1 / (12 * total) + 1 / (288 * total * total));
    const double term1_num =
        2 * (alpha * alpha) * (x - 1) + alpha * beta * (x - 1) -
        x * (beta * beta);
    const double axbx = alpha * (x - 1) + beta * x;
    const double term1_den =
        std::sqrt(2 * alpha / beta) * std::pow(total, 1.5) * axbx * axbx;
    const double term1 = term1_num / term1_den;
    const double term2 = 0.5 * std::log(alpha / (total * x));
    const double term3_num = std::sqrt(8 * alpha * beta / total);
    const double term3_den = beta * x + alpha * (x - 1);
    const double term3 = term3_num / term3_den;
    const double term4_base =
        beta * std::log(beta / (total * (1 - x))) +
        alpha * std::log(alpha / (total * x));
    const double term4 = std::pow(term4_base, -1.5);
    const double term1234 = term1 + term2 * (term3 +
        (x < mean ? term4 : -term4));
    return stirling * prefactor * term1234;
}

double dirichlet_grad_rational_cpu(double x, double alpha, double total) {
    const double u = std::log(x);
    const double a = std::log(alpha) - u;
    const double b = std::log(total) - a;
    const double pow_u[3] = {1, u, u * u};
    const double pow_a[3] = {1, a, a * a};
    static const double coefficients[2][3][3][4] = {
        {{{1.003668233, -0.01061107488, -0.0657888334, 0.01201642863},
          {0.6336835991, -0.3557432599, 0.05486251648, -0.001465281033},
          {-0.03276231906, 0.004474107445, 0.002429354597, -0.0001557569013}},
         {{0.221950385, -0.3187676331, 0.01799915743, 0.01074823814},
          {-0.2951249643, 0.06219954479, 0.01535556598, 0.001550077057},
          {0.02155310298, 0.004170831599, 0.001292462449, 6.976601077e-05}},
         {{-0.05980841433, 0.008441916499, 0.01085618172, 0.002319392565},
          {0.02911413504, 0.01400243777, -0.002721828457, 0.000751041181},
          {0.005900514878, -0.001936558688, -9.495446725e-06, 5.385558597e-05}}},
        {{{1, -0.02924021934, -0.04438342661, 0.007285809825},
          {0.6357567472, -0.3473456711, 0.05454656494, -0.002407477521},
          {-0.03301322327, 0.004845219414, 0.00231480583, -0.0002307248149}},
         {{0.5925320577, -0.1757678135, 0.01505928619, 0.000564515273},
          {0.1014815858, -0.06589186703, 0.01272886114, -0.0007316646956},
          {-0.007258481865, 0.001096195486, 0.0003934994223, -4.12701925e-05}},
         {{0.06469649321, -0.0236701437, 0.002902096474, -5.896963079e-05},
          {0.001925008108, -0.002869809258, 0.0008000589141, -6.063713228e-05},
          {-0.0003477407336, 6.959756487e-05, 1.097287507e-05, -1.650964693e-06}}},
    };
    double p = 0.0;
    double q = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double ua = pow_u[i] * pow_a[j];
            p += ua * (coefficients[0][i][j][0] +
                       b * (coefficients[0][i][j][1] +
                            b * (coefficients[0][i][j][2] +
                                 b * coefficients[0][i][j][3])));
            q += ua * (coefficients[1][i][j][0] +
                       b * (coefficients[1][i][j][1] +
                            b * (coefficients[1][i][j][2] +
                                 b * coefficients[1][i][j][3])));
        }
    }
    const double approximation =
        x * (tensorplay::special_math::calc_digamma(total) -
             tensorplay::special_math::calc_digamma(alpha)) /
        (total - alpha);
    return p / q * approximation;
}

double dirichlet_grad_one_cpu(double x, double alpha, double total) {
    const double beta = total - alpha;
    const double boundary = total * x * (1 - x);
    if (x <= 0.5 && boundary < 2.5) {
        return dirichlet_grad_alpha_small_cpu(x, alpha, beta);
    }
    if (x >= 0.5 && boundary < 0.75) {
        return -dirichlet_grad_beta_small_cpu(1 - x, beta, alpha);
    }
    if (alpha > 6 && beta > 6) {
        return dirichlet_grad_alpha_mid_cpu(x, alpha, beta);
    }
    return dirichlet_grad_rational_cpu(x, alpha, total);
}

Tensor binomial_kernel(const Tensor& count, const Tensor& prob,
                       std::optional<Generator> generator) {
    if (!isFloatingType(count.dtype())) {
        TP_THROW(ValueError, "binomial only supports floating-point dtypes for count, got: ",
                  toString(count.dtype()));
    }
    if (!isFloatingType(prob.dtype())) {
        TP_THROW(ValueError, "binomial only supports floating-point dtypes for prob, got: ",
                  toString(prob.dtype()));
    }
    if (prob.dtype() != count.dtype()) {
        TP_THROW(RuntimeError, "Found dtype ", toString(prob.dtype()),
                  " but expected ", toString(count.dtype()));
    }
    Generator& gen = generator.has_value() ? *generator : default_generator();
    const std::vector<int64_t> bshape =
        broadcast_shapes(static_cast<std::vector<int64_t>>(count.shape()),
                         static_cast<std::vector<int64_t>>(prob.shape()));
    Tensor out(bshape, count.dtype(), count.device());
    if (out.numel() == 0) {
        return out;
    }
    Tensor count_b = count.expand(bshape).contiguous();
    Tensor prob_b = prob.expand(bshape).contiguous();

    if (count.dtype() == DType::Float32) {
        const float* cp = count_b.data_ptr<float>();
        const float* pp = prob_b.data_ptr<float>();
        float* res = out.data_ptr<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            res[i] = static_cast<float>(
                sample_binomial(static_cast<double>(cp[i]), static_cast<double>(pp[i]), &gen));
        }
    } else if (count.dtype() == DType::Float64) {
        const double* cp = count_b.data_ptr<double>();
        const double* pp = prob_b.data_ptr<double>();
        double* res = out.data_ptr<double>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            res[i] = sample_binomial(cp[i], pp[i], &gen);
        }
    } else {
        TP_THROW(NotImplementedError, "\"binomial_cpu\" not implemented for '",
                  toString(count.dtype()), "'");
    }
    return out;
}

Tensor standard_gamma_kernel(const Tensor& self, std::optional<Generator> generator) {
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, "\"standard_gamma_cpu\" not implemented for '",
                  toString(self.dtype()), "'");
    }
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) {
        return out;
    }
    Generator& gen = generator.has_value() ? *generator : default_generator();

    Tensor sc = self.contiguous();

    // The rejection loop makes no progress for NaN or negative shapes, so
    // screen the concentrations up front.
    if (sc.dtype() == DType::Float32) {
        const float* vp = sc.data_ptr<float>();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            if (!(vp[i] >= 0.0f)) {
                TP_THROW(ValueError, "standard_gamma: concentration values must be non-negative");
            }
        }
    } else {
        const double* vp = sc.data_ptr<double>();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            if (!(vp[i] >= 0.0)) {
                TP_THROW(ValueError, "standard_gamma: concentration values must be non-negative");
            }
        }
    }

    if (sc.dtype() == DType::Float32) {
        const float* sp = sc.data_ptr<float>();
        float* res = out.data_ptr<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            const float sample = sample_gamma(sp[i], &gen);
            res[i] = std::max(std::numeric_limits<float>::min(), sample);
        }
    } else {
        const double* sp = sc.data_ptr<double>();
        double* res = out.data_ptr<double>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            const double sample = sample_gamma(sp[i], &gen);
            res[i] = std::max(std::numeric_limits<double>::min(), sample);
        }
    }
    return out;
}

Tensor sample_dirichlet_kernel(const Tensor& self, std::optional<Generator> generator) {
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, "\"_sample_dirichlet_cpu\" not implemented for '",
                  toString(self.dtype()), "'");
    }
    TP_CHECK(self.dim() >= 1, "dirichlet: expects a tensor with at least one dimension");
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) {
        return out;
    }
    Generator& gen = generator.has_value() ? *generator : default_generator();

    Tensor sc = self.contiguous();
    const int64_t k = sc.size(-1);
    const int64_t rows = sc.numel() / k;

    // Draw the gamma variates in double so tiny concentrations survive the
    // normalization, clamped away from zero.
    std::vector<double> gamma_vals(static_cast<size_t>(sc.numel()));
    if (sc.dtype() == DType::Float32) {
        const float* sp = sc.data_ptr<float>();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            const double a = static_cast<double>(sp[i]);
            if (!(a >= 0.0)) {
                TP_THROW(ValueError, "dirichlet: concentration values must be non-negative");
            }
            gamma_vals[static_cast<size_t>(i)] =
                std::max(std::numeric_limits<double>::min(),
                         static_cast<double>(sample_gamma(a, &gen)));
        }
    } else {
        const double* sp = sc.data_ptr<double>();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            if (!(sp[i] >= 0.0)) {
                TP_THROW(ValueError, "dirichlet: concentration values must be non-negative");
            }
            gamma_vals[static_cast<size_t>(i)] =
                std::max(std::numeric_limits<double>::min(),
                         static_cast<double>(sample_gamma(sp[i], &gen)));
        }
    }

    // Normalize each last-dimension group and clamp into the representable
    // range of the storage dtype.
    auto write = [&](auto tag) {
        using scalar_t = decltype(tag);
        scalar_t* res = out.data_ptr<scalar_t>();
        const scalar_t min_val = std::numeric_limits<scalar_t>::min();
        const scalar_t max_val =
            static_cast<scalar_t>(std::nexttoward(static_cast<scalar_t>(1.0f), 0.0L));
        for (int64_t r = 0; r < rows; ++r) {
            double rowsum = 0.0;
            for (int64_t j = 0; j < k; ++j) {
                rowsum += gamma_vals[static_cast<size_t>(r * k + j)];
            }
            for (int64_t j = 0; j < k; ++j) {
                const double ratio =
                    gamma_vals[static_cast<size_t>(r * k + j)] / rowsum;
                const scalar_t v = static_cast<scalar_t>(ratio);
                res[r * k + j] = std::min(max_val, std::max(min_val, v));
            }
        }
    };
    if (out.dtype() == DType::Float32) {
        write(float{});
    } else {
        write(double{});
    }
    return out;
}

Tensor standard_gamma_grad_kernel(const Tensor& self, const Tensor& output) {
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(TypeError, "standard_gamma_grad expects Float32 or Float64 input");
    }
    if (output.dtype() != self.dtype() || output.shape() != self.shape()) {
        TP_THROW(RuntimeError,
                 "standard_gamma_grad: input and sample must have matching dtype and shape");
    }

    Tensor alpha = self.is_contiguous() ? self : self.contiguous();
    Tensor sample = output.is_contiguous() ? output : output.contiguous();
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (out.numel() == 0) return out;

    if (self.dtype() == DType::Float32) {
        const float* alpha_data = alpha.data_ptr<float>();
        const float* sample_data = sample.data_ptr<float>();
        float* output_data = out.data_ptr<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            output_data[i] = static_cast<float>(standard_gamma_grad_one_cpu(
                static_cast<double>(alpha_data[i]),
                static_cast<double>(sample_data[i])));
        }
    } else {
        const double* alpha_data = alpha.data_ptr<double>();
        const double* sample_data = sample.data_ptr<double>();
        double* output_data = out.data_ptr<double>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            output_data[i] = standard_gamma_grad_one_cpu(alpha_data[i], sample_data[i]);
        }
    }
    return out;
}

Tensor dirichlet_grad_kernel(const Tensor& x, const Tensor& alpha,
                             const Tensor& total) {
    if (x.dtype() != DType::Float32 && x.dtype() != DType::Float64) {
        TP_THROW(TypeError, "dirichlet_grad expects Float32 or Float64 input");
    }
    if (alpha.dtype() != x.dtype() || total.dtype() != x.dtype() ||
        alpha.shape() != x.shape() || total.shape() != x.shape()) {
        TP_THROW(RuntimeError,
                 "dirichlet_grad: inputs must have matching dtype and shape");
    }

    Tensor x_contiguous = x.is_contiguous() ? x : x.contiguous();
    Tensor alpha_contiguous = alpha.is_contiguous() ? alpha : alpha.contiguous();
    Tensor total_contiguous = total.is_contiguous() ? total : total.contiguous();
    Tensor out(static_cast<std::vector<int64_t>>(x.shape()), x.dtype(), x.device());
    if (out.numel() == 0) return out;

    if (x.dtype() == DType::Float32) {
        const float* x_data = x_contiguous.data_ptr<float>();
        const float* alpha_data = alpha_contiguous.data_ptr<float>();
        const float* total_data = total_contiguous.data_ptr<float>();
        float* output_data = out.data_ptr<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            output_data[i] = static_cast<float>(dirichlet_grad_one_cpu(
                static_cast<double>(x_data[i]),
                static_cast<double>(alpha_data[i]),
                static_cast<double>(total_data[i])));
        }
    } else {
        const double* x_data = x_contiguous.data_ptr<double>();
        const double* alpha_data = alpha_contiguous.data_ptr<double>();
        const double* total_data = total_contiguous.data_ptr<double>();
        double* output_data = out.data_ptr<double>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            output_data[i] = dirichlet_grad_one_cpu(
                x_data[i], alpha_data[i], total_data[i]);
        }
    }
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
    auto& gen = default_generator();
    const bool full_range = (low == 0 && high == 0);
    if (full_range) {
        if (self.numel() == 0) return self;
        check_writable_inplace(self);
        dispatch_all(self.dtype(), [&](auto tag) {
            using scalar_t = decltype(tag);
            if constexpr (std::is_same_v<scalar_t, uint64_t>) {
                for_each_element<scalar_t>(self, [&](scalar_t& value) {
                    value = static_cast<scalar_t>(gen.random64());
                });
            } else {
                uint64_t range;
                if constexpr (std::is_same_v<scalar_t, int64_t>) {
                    range = uint64_t{1} << 63;
                } else if constexpr (std::is_same_v<scalar_t, double>) {
                    range = uint64_t{1} << 53;
                } else if constexpr (std::is_same_v<scalar_t, float>) {
                    range = uint64_t{1} << 24;
                } else if constexpr (std::is_same_v<scalar_t, Half>) {
                    range = uint64_t{1} << 11;
                } else if constexpr (std::is_same_v<scalar_t, BFloat16>) {
                    range = uint64_t{1} << 8;
                } else if constexpr (std::is_same_v<scalar_t, bool>) {
                    range = 2;
                } else {
                    range = static_cast<uint64_t>(
                        std::numeric_limits<scalar_t>::max()) + 1;
                }
                uniform_int_from_to_distribution<scalar_t> dist(range, 0);
                for_each_element<scalar_t>(self, [&](scalar_t& value) {
                    value = dist(&gen);
                });
            }
        });
        return self;
    }
    TP_THROW_IF(high <= low, RuntimeError,
                "random_ expects 'from' to be less than 'to', but got from=",
                low, " >= to=", high);

    distribution::check_random_from_to_bounds(low, high, self.dtype());

    if (self.numel() == 0) return self;
    check_writable_inplace(self);

    const uint64_t range = static_cast<uint64_t>(high) -
        static_cast<uint64_t>(low);
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
    m.impl("binomial", binomial_kernel);
    m.impl("_standard_gamma", standard_gamma_kernel);
    m.impl("_standard_gamma_grad", standard_gamma_grad_kernel);
    m.impl("_sample_dirichlet", sample_dirichlet_kernel);
    m.impl("_dirichlet_grad", dirichlet_grad_kernel);
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
