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

inline MemoryFormat activation_format(const Tensor& input) {
    if (input.dim() != 4 && input.dim() != 5) return MemoryFormat::Contiguous;
    if (input.stride(1) == 0) return MemoryFormat::Contiguous;
    int64_t minimum = 0;
    const int order4[4]{1, 3, 2, 0}, order5[5]{1, 4, 3, 2, 0};
    const int* order = input.dim() == 4 ? order4 : order5;
    for (int i = 0; i < input.dim(); ++i) {
        const int d = order[i];
        if (input.size(d) == 0 || input.stride(d) < minimum ||
            (d == 0 && minimum == input.stride(1))) return MemoryFormat::Contiguous;
        minimum = input.stride(d);
        if (input.size(d) > 1) minimum *= input.size(d);
    }
    return input.dim() == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
}

inline Tensor activation_output(const Tensor& input, MemoryFormat format) {
    Tensor out = Tensor::empty(input.sizes().vec(), input.dtype(), input.device());
    if (format != MemoryFormat::Contiguous)
        out = Tensor(out.impl()->storage(), input.sizes().vec(), get_channels_last_strides(input.sizes().vec()), input.dtype(), 0);
    return out;
}

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
            case 6: return x > slope ? x : alpha;
            case 7: return ::fminf(::fmaxf(x, slope), alpha);
            case 8: return x * .5f * (1.f + ::erff(x * .7071067811865475f));
            case 9: return x * .5f * (1.f + ::tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
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
#ifdef USE_CUDA
Tensor activation_cuda(const Tensor& input, double scale, int64_t zero_point,
                       Activation activation);
#endif
Tensor relu6_cpu(const Tensor& input);
Tensor relu_cpu(const Tensor& input);
}

#undef TP_QE_HD
