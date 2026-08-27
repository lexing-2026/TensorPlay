#pragma once

// Port of ATen's distribution layer (ATen/core/TransformationHelper.h and
// ATen/core/DistributionsHelper.h). The transformation formulas and the number
// of engine draws per sample must match PyTorch exactly so that a given seed
// reproduces torch's random tensors; std:: distributions cannot be used
// because their algorithms are implementation-defined.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "BFloat16.h"
#include "Half.h"

namespace tensorplay {

constexpr double pi_d = 3.14159265358979323846;

// Accumulation type for distributions: Half/BFloat16 use float, float stays
// float, and double stays double (matches at::dist_acctype).
template <typename T>
struct DistAccumType {};
template <> struct DistAccumType<Half> { using type = float; };
template <> struct DistAccumType<BFloat16> { using type = float; };
template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

// std::numeric_limits is not specialized for TensorPlay's storage wrappers.
// Keep the distribution's mantissa width explicit so Half/BFloat16 consume
// the same masked raw words as ATen's scalar-dtype CPU kernels.
template <typename T>
struct DistMantissaBits {
    static constexpr int value = std::numeric_limits<T>::digits;
};
template <> struct DistMantissaBits<Half> { static constexpr int value = 11; };
template <> struct DistMantissaBits<BFloat16> { static constexpr int value = 8; };

// Computation precision for a storage dtype: Half/BFloat16 sample in float
// (mirrors at::opmath_type).
template <typename T>
struct OpMathType { using type = T; };
template <> struct OpMathType<Half> { using type = float; };
template <> struct OpMathType<BFloat16> { using type = float; };

template <typename T>
using opmath_t = typename OpMathType<T>::type;

// ATen includes c10's reduced-floating-point math overloads for its CPU
// distribution kernels.  Those overloads return Half/BFloat16 after each
// elementary function (std::log(Half), std::sqrt(Half), ...), rather than
// leaving the result in float.  Keep the same storage-dtype boundary without
// adding non-standard overloads to namespace std for TensorPlay's wrappers.
template <typename T>
inline T distribution_log(T value) {
    if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(std::log(static_cast<float>(value)));
    } else {
        return std::log(value);
    }
}

template <typename T>
inline T distribution_sqrt(T value) {
    if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(std::sqrt(static_cast<float>(value)));
    } else {
        return std::sqrt(value);
    }
}

template <typename T>
inline T distribution_cos(T value) {
    if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(std::cos(static_cast<float>(value)));
    } else {
        return std::cos(value);
    }
}

template <typename T>
inline T distribution_sin(T value) {
    if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(std::sin(static_cast<float>(value)));
    } else {
        return std::sin(value);
    }
}

namespace transformation {

template <typename T, typename V>
inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
    return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

template <typename T, typename V>
inline T uniform_int_full_range(V val) {
    return static_cast<T>(static_cast<int64_t>(val));
}

template <typename T, typename V>
inline std::enable_if_t<!std::is_floating_point_v<T>, T> uniform_int(V val) {
    if constexpr (std::is_same_v<T, bool>) {
        return static_cast<bool>(val & 1);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else {
        return 0;
    }
}

// Transforms a raw unsigned integer draw into [from, to) with a mask of the
// mantissa bits of T. float consumes one 32-bit draw, double one 64-bit draw.
template <typename T, typename V>
inline dist_acctype<T> uniform_real(V val, T from, T to) {
    constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << DistMantissaBits<T>::value) - 1);
    constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) /
                             (static_cast<uint64_t>(1) << DistMantissaBits<T>::value);
    dist_acctype<T> x = (val & MASK) * DIVISOR;
    // c10::Half/BFloat16 round the storage-dtype subtraction first, but
    // their mixed float/double operators keep the subsequent arithmetic in
    // the accumulator type.  Spell that boundary out so custom storage
    // wrappers cannot introduce an extra product/add round-trip.
    const dist_acctype<T> range = static_cast<dist_acctype<T>>(to - from);
    const dist_acctype<T> base = static_cast<dist_acctype<T>>(from);
    return x * range + base;
}

template <typename T>
inline T normal(T val, T mean, T std) {
    return val * std + mean;
}

template <typename T>
inline T cauchy(T val, T median, T sigma) {
    // https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function
    // tan overflows and returns inf/-inf when (val > 1 - eps) or (val < eps),
    // thus we clip those values.
    constexpr T eps = std::numeric_limits<T>::epsilon();
    constexpr T one_minus_eps = 1 - eps;
    constexpr T zero_plus_eps = 0 + eps;
    val = (val > one_minus_eps ? one_minus_eps : val);
    val = (val < zero_plus_eps ? zero_plus_eps : val);
    return median + sigma * std::tan(pi_d * (val - static_cast<T>(0.5)));
}

template <>
inline double cauchy(double val, double median, double sigma) {
    return median + sigma * std::tan(pi_d * (val - 0.5));
}

template <typename T>
inline T exponential(T val, T lambda) {
    // CPU variant: log1p keeps precision for small val.
    return static_cast<T>(-1.0) / lambda * std::log1p(-val);
}

template <typename T>
inline T geometric(T val, T p) {
    // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
    return static_cast<T>(std::ceil(std::log(val) / std::log1p(-p)));
}

template <typename T>
inline T log_normal(T val) {
    return std::exp(val);
}

template <typename T>
inline T bernoulli(T val, T p) {
    return val < p;
}

} // namespace transformation

// Transforms uniformly distributed [0, 1) values into [from, to). For double,
// consumes a 64-bit draw; otherwise a single 32-bit draw.
template <typename T>
struct uniform_real_distribution {
    inline uniform_real_distribution(T from, T to) : from_(from), to_(to) {}

    template <typename RNG>
    inline dist_acctype<T> operator()(RNG* generator) const {
        if constexpr (std::is_same_v<T, double>) {
            return transformation::uniform_real<T>(generator->random64(), from_, to_);
        } else {
            return transformation::uniform_real<T>(generator->random(), from_, to_);
        }
    }

private:
    T from_;
    T to_;
};

// Box-Muller; returns two samples at a time so the second is cached in the
// generator and survives across calls (and get_state/set_state).
template <typename RNG>
inline bool maybe_get_next_normal_sample(RNG* generator, double* ret) {
    const auto sample = generator->next_double_normal_sample();
    if (!sample.has_value()) return false;
    *ret = sample.value();
    generator->set_next_double_normal_sample(std::nullopt);
    return true;
}

template <typename RNG>
inline bool maybe_get_next_normal_sample(RNG* generator, float* ret) {
    const auto sample = generator->next_float_normal_sample();
    if (!sample.has_value()) return false;
    *ret = sample.value();
    generator->set_next_float_normal_sample(std::nullopt);
    return true;
}

template <typename RNG>
inline void maybe_set_next_normal_sample(RNG* generator, const double* cache) {
    generator->set_next_double_normal_sample(*cache);
}

template <typename RNG>
inline void maybe_set_next_normal_sample(RNG* generator, const float* cache) {
    generator->set_next_float_normal_sample(*cache);
}

template <typename T>
struct normal_distribution {
    inline normal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {}

    template <typename RNG>
    inline dist_acctype<T> operator()(RNG* generator) const {
        dist_acctype<T> ret;
        // return cached values if available
        if (maybe_get_next_normal_sample(generator, &ret)) {
            return transformation::normal(ret, mean, stdv);
        }

        // otherwise generate new normal values
        uniform_real_distribution<T> uniform(0.0, 1.0);
        const dist_acctype<T> u1 = uniform(generator);
        const dist_acctype<T> u2 = uniform(generator);
        const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log1p(-u2));
        const dist_acctype<T> theta = static_cast<T>(2.0) * pi_d * u1;
        const dist_acctype<T> sample = r * ::sin(theta);
        maybe_set_next_normal_sample(generator, &sample);

        ret = r * ::cos(theta);
        return transformation::normal(ret, mean, stdv);
    }

private:
    T mean;
    T stdv;
};

// (val % range) + base over the raw engine draw; a range >= 2^28 switches to
// 64-bit draws, mirroring torch's uniform_int_from_to_distribution.
template <typename T>
struct uniform_int_from_to_distribution {
    uniform_int_from_to_distribution(uint64_t range, int64_t base) : range_(range), base_(base) {}

    template <typename RNG>
    T operator()(RNG* generator) const {
        if (range_ >= 1ULL << 28) {
            return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
        } else {
            return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
        }
    }

private:
    uint64_t range_;
    int64_t base_;
};

template <typename T>
struct bernoulli_distribution {
    inline bernoulli_distribution(T p_in) : p(p_in) {}

    template <typename RNG>
    inline T operator()(RNG* generator) const {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::bernoulli<T>(uniform(generator), p);
    }

private:
    T p;
};

template <typename T>
struct geometric_distribution {
    inline geometric_distribution(T p_in) : p(p_in) {}

    template <typename RNG>
    inline T operator()(RNG* generator) const {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::geometric<T>(uniform(generator), p);
    }

private:
    T p;
};

template <typename T>
struct exponential_distribution {
    inline exponential_distribution(T lambda_in) : lambda(lambda_in) {}

    template <typename RNG>
    inline T operator()(RNG* generator) const {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::exponential<T>(uniform(generator), lambda);
    }

private:
    T lambda;
};

template <typename T>
struct cauchy_distribution {
    inline cauchy_distribution(T median_in, T sigma_in) : median(median_in), sigma(sigma_in) {}

    template <typename RNG>
    inline T operator()(RNG* generator) const {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::cauchy<T>(uniform(generator), median, sigma);
    }

private:
    T median;
    T sigma;
};

template <typename T>
struct lognormal_distribution {
    inline lognormal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {}

    template <typename RNG>
    inline T operator()(RNG* generator) const {
        normal_distribution<T> normal(mean, stdv);
        return transformation::log_normal<T>(normal(generator));
    }

private:
    T mean;
    T stdv;
};

// Box-Muller applied in-place to blocks of 16 uniforms; scalar variant of
// torch's NormalFill16.
template <typename T>
struct NormalFill16 {
    T mean_;
    T std_;

    NormalFill16(T mean, T std) : mean_(mean), std_(std) {}

    void operator()(T* data) const {
        for (int j = 0; j < 8; ++j) {
            const T u1 = T(1) - data[j]; // [0, 1) -> (0, 1] for log.
            const T u2 = data[j + 8];
            const T radius = distribution_sqrt<T>(T(-2) * distribution_log<T>(u1));
            const T theta = 2.0 * pi_d * u2;
            // Keep the scalar expression in the same order as ATen's
            // normal_fill_16.  In particular, do not fuse the multiply-add:
            // Half/BFloat16 must round at the storage-dtype operation points.
            data[j] = radius * distribution_cos<T>(theta) * std_ + mean_;
            data[j + 8] = radius * distribution_sin<T>(theta) * std_ + mean_;
        }
    }
};

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define TENSORPLAY_X86_AVX2_DISPATCH 1
#endif

#ifdef TENSORPLAY_X86_AVX2_DISPATCH

#include "avx_mathfun.h"

namespace detail {

inline bool cpu_supports_avx2() {
    static const bool cached = __builtin_cpu_supports("avx2") != 0;
    return cached;
}

// AVX2 variant of torch's NormalFill16<float, true> (avx_mathfun polynomial
// approximations). Bit-identical to torch's AVX2-dispatched kernel; the
// scalar fallback above only runs on pre-AVX2 machines.
struct NormalFill16AVX2 {
    float mean_;
    float std_;

    NormalFill16AVX2(float mean, float std) : mean_(mean), std_(std) {}

    __attribute__((target("avx2,fma")))
    void operator()(float* data) const {
        const __m256 v_mean = _mm256_set1_ps(mean_);
        const __m256 v_std = _mm256_set1_ps(std_);
        const __m256 two_pi_ = _mm256_set1_ps(2.0f * pi_d);
        const __m256 one_ = _mm256_set1_ps(1.0f);
        const __m256 minus_two_ = _mm256_set1_ps(-2.0f);
        const __m256 u1 = _mm256_sub_ps(one_, _mm256_loadu_ps(data));
        const __m256 u2 = _mm256_loadu_ps(data + 8);
        const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two_, log256_ps(u1)));
        const __m256 theta = _mm256_mul_ps(two_pi_, u2);
        __m256 sintheta, costheta;
        sincos256_ps(theta, &sintheta, &costheta);
        const __m256 n1 = _mm256_mul_ps(radius, costheta);
        const __m256 n2 = _mm256_mul_ps(radius, sintheta);
        _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, v_std, v_mean));
        _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, v_std, v_mean));
    }
};

} // namespace detail

#endif // TENSORPLAY_X86_AVX2_DISPATCH

// Fast path for contiguous tensors with >= 16 elements: fill with uniforms,
// then convert to normals in-place block by block. Mirrors torch's
// normal_fill consumption pattern element for element.
template <typename T, typename RNG>
void normal_fill(T* data, int64_t size, T mean, T std, RNG* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
#ifdef TENSORPLAY_X86_AVX2_DISPATCH
    if constexpr (std::is_same_v<T, float>) {
        if (detail::cpu_supports_avx2()) {
            detail::NormalFill16AVX2 normal_fill_16(mean, std);
            for (int64_t i = 0; i < size; ++i) {
                data[i] = uniform(generator);
            }
            for (int64_t i = 0; i < size - 15; i += 16) {
                normal_fill_16(data + i);
            }
            // Recompute the last 16 values.
            if (size % 16 != 0) {
                data = data + size - 16;
                for (int i = 0; i < 16; ++i) {
                    data[i] = uniform(generator);
                }
                normal_fill_16(data);
            }
            return;
        }
    }
#endif
    NormalFill16<T> normal_fill_16(mean, std);
    for (int64_t i = 0; i < size; ++i) {
        data[i] = uniform(generator);
    }
    for (int64_t i = 0; i < size - 15; i += 16) {
        normal_fill_16(data + i);
    }
    // Recompute the last 16 values.
    if (size % 16 != 0) {
        data = data + size - 16;
        for (int i = 0; i < 16; ++i) {
            data[i] = uniform(generator);
        }
        normal_fill_16(data);
    }
}

// Half/BFloat16 variant of torch's normal_fill: sample in float precision
// through a 16-element stack buffer, Box-Muller in-place, then cast down.
template <typename scalar_t, typename RNG>
void normal_fill_cast(scalar_t* data, int64_t size, double mean, double std, RNG* generator) {
    using math_t = opmath_t<scalar_t>;
    uniform_real_distribution<math_t> uniform(0.0, 1.0);
    auto fill_block = [&](math_t* buf) {
#ifdef TENSORPLAY_X86_AVX2_DISPATCH
        if (detail::cpu_supports_avx2()) {
            detail::NormalFill16AVX2 fill16(static_cast<float>(mean), static_cast<float>(std));
            fill16(buf);
            return;
        }
#endif
        NormalFill16<math_t> fill16(static_cast<math_t>(mean), static_cast<math_t>(std));
        fill16(buf);
    };
    math_t buf[16];
    for (int64_t i = 0; i < size - 15; i += 16) {
        for (int j = 0; j < 16; ++j) buf[j] = uniform(generator);
        fill_block(buf);
        for (int j = 0; j < 16; ++j) data[i + j] = static_cast<scalar_t>(buf[j]);
    }
    // Recompute the last 16 values.
    if (size % 16 != 0) {
        int64_t offset = size - 16;
        for (int j = 0; j < 16; ++j) buf[j] = uniform(generator);
        fill_block(buf);
        for (int j = 0; j < 16; ++j) data[offset + j] = static_cast<scalar_t>(buf[j]);
    }
}

} // namespace tensorplay
