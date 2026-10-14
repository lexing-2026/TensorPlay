#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace tensorplay {
namespace cpu {

namespace {

// Per-row quantization for 2-bit / 1-bit signed codes.
// scale = max(|row|) / scale_divisor; codes = clamp(round(x / scale),
// quant_min, quant_max).
template <typename scalar_t>
void lowbit_quantize_cpu_impl(const Tensor& x, Tensor& codes, Tensor& scales,
                              float eps, float scale_divisor, int quant_min,
                              int quant_max, bool sign_mode) {
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const scalar_t* x_ptr = x.data_ptr<scalar_t>();
    int8_t* codes_ptr = codes.data_ptr<int8_t>();
    float* scales_ptr = scales.data_ptr<float>();
    tensorplay::parallel::parallel_for(0, M, 1, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float absmax = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                absmax =
                    std::max(absmax, std::fabs(static_cast<float>(x_ptr[r * K + k])));
            }
            const float scale = std::max(absmax, eps) / scale_divisor;
            scales_ptr[r] = scale;
            for (int64_t k = 0; k < K; ++k) {
                int c;
                if (sign_mode) {
                    c = (static_cast<float>(x_ptr[r * K + k]) >= 0.0f)
                            ? quant_max
                            : quant_min;
                } else {
                    c = static_cast<int>(std::nearbyint(
                        static_cast<float>(x_ptr[r * K + k]) / scale));
                    c = c < quant_min ? quant_min : (c > quant_max ? quant_max : c);
                }
                codes_ptr[r * K + k] = static_cast<int8_t>(c);
            }
        }
    });
}

void launch_lowbit_quantize_cpu(const Tensor& x, Tensor& codes, Tensor& scales,
                                float eps, float scale_divisor, int quant_min,
                                int quant_max, bool sign_mode) {
    switch (x.dtype()) {
        case DType::Float16:
            lowbit_quantize_cpu_impl<tensorplay::Half>(
                x, codes, scales, eps, scale_divisor, quant_min, quant_max,
                sign_mode);
            break;
        case DType::BFloat16:
            lowbit_quantize_cpu_impl<tensorplay::BFloat16>(
                x, codes, scales, eps, scale_divisor, quant_min, quant_max,
                sign_mode);
            break;
        case DType::Float32:
            lowbit_quantize_cpu_impl<float>(
                x, codes, scales, eps, scale_divisor, quant_min, quant_max,
                sign_mode);
            break;
        default:
            TP_THROW(TypeError, "lowbit quantize: unsupported input dtype");
    }
}

std::tuple<Tensor, Tensor> quantize_lowbit_rowwise(const Tensor& self,
                                                     double eps,
                                                     float scale_divisor,
                                                     int quant_min,
                                                     int quant_max,
                                                     bool sign_mode,
                                                     const char* op_name) {
    TP_CHECK(self.dim() == 2, std::string(op_name) + "(): expected a 2-D tensor");
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_lowbit_quantize_cpu(x, codes, scales, static_cast<float>(eps),
                               scale_divisor, quant_min, quant_max, sign_mode);
    return {codes, scales};
}

}  // namespace

// Row-wise int2 quantization: signed codes in {-2, -1, 0, 1} and
// scale = max(|row|) / 2.
std::tuple<Tensor, Tensor> _quantize_int2_rowwise_cpu(const Tensor& self,
                                                       double eps) {
    return quantize_lowbit_rowwise(self, eps, 2.0f, -2, 1, false,
                                   "_quantize_int2_rowwise");
}

// Row-wise 1-bit quantization: strict sign codes in {-1, +1} and
// scale = max(|row|).
std::tuple<Tensor, Tensor> _quantize_binary_rowwise_cpu(const Tensor& self,
                                                        double eps) {
    return quantize_lowbit_rowwise(self, eps, 1.0f, -1, 1, true,
                                   "_quantize_binary_rowwise");
}

TENSORPLAY_LIBRARY_IMPL(CPU, LowBitQuantNative) {
    m.impl("_quantize_int2_rowwise", _quantize_int2_rowwise_cpu);
    m.impl("_quantize_binary_rowwise", _quantize_binary_rowwise_cpu);
}

}  // namespace cpu
}  // namespace tensorplay