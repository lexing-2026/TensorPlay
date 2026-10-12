#pragma once

#include "Tensor.h"
#include "Quantizer.h"
#include <cmath>
#include <limits>
#include <cstdint>

#if defined(__CUDACC__)
#define TP_QE_HD __host__ __device__
#else
#define TP_QE_HD
#endif

namespace tensorplay::quant {

struct Activation {
    int op;
    float slope = 0, alpha = 1, coefficient = 1, input_coefficient = 1;
    TP_QE_HD float operator()(float x) const {
        switch (op) {
            case 0: return x > 0 ? x : x * slope;
            case 1: return x >= 0 ? x * coefficient
                : ::expm1f(x * input_coefficient) * alpha * coefficient;
            case 2: return x * ::fminf(::fmaxf(x + 3.f, 0.f), 6.f) / 6.f;
            case 3: return ::fminf(::fmaxf(x + 3.f, 0.f), 6.f) / 6.f;
            case 4: return 1.f / (1.f + ::expf(-x));
            case 5: return ::tanhf(x);
            default: return x;
        }
    }
    bool operator==(const Activation& b) const {
        return op == b.op && slope == b.slope && alpha == b.alpha &&
            coefficient == b.coefficient && input_coefficient == b.input_coefficient;
    }
};

// Round the real value before adding the integer zero point, and saturate
// before converting to storage so infinities cannot overflow an integer cast.
template<class T>
TP_QE_HD T encode(float value, float inverse_scale, int64_t zero_point) {
    const double code = static_cast<double>(::nearbyintf(value * inverse_scale)) + zero_point;
    constexpr double lo = std::numeric_limits<T>::lowest();
    constexpr double hi = std::numeric_limits<T>::max();
    return static_cast<T>(::fmin(hi, ::fmax(lo, code)));
}

Tensor activation_cpu(const Tensor& input, double scale, int64_t zero_point,
                      Activation activation);
Tensor relu6_cpu(const Tensor& input);
Tensor relu_cpu(const Tensor& input);
}

#undef TP_QE_HD
