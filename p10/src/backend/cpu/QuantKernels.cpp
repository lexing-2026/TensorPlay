#include "QuantKernels.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"

#include <cmath>
#include <vector>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// Quantization is defined over real (floating) values only.  Narrow the
// dtype space up front: everything lands on Float32/Float64 compute and an
// Int8 storage, so the kernels below need exactly two scalar instantiations.
Tensor promote_to_compute_dtype(const Tensor& self) {
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        return self.is_contiguous() ? self : self.contiguous();
    }
    if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        return self.to(DType::Float32).contiguous();
    }
    TP_THROW(TypeError,
             std::string("quantize(): expected a floating point tensor, got ") +
                 toString(self.dtype()));
}

void check_qparams(double scale, int64_t zero_point, int64_t quant_min,
                   int64_t quant_max) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "quantize(): scale must be positive");
    }
    if (quant_min >= quant_max) {
        TP_THROW(ValueError, "quantize(): quant_min must be < quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError, "quantize(): zero_point out of the quantized range");
    }
}

template <typename T>
void quantize_kernel(const T* input, int8_t* output, int64_t numel,
                     double scale, int64_t zero_point, int64_t quant_min,
                     int64_t quant_max) {
    // nearbyint matches ATen's round-to-even quantization semantics.
    for (int64_t i = 0; i < numel; ++i) {
        const double rounded =
            std::nearbyint(static_cast<double>(input[i]) / scale) +
            static_cast<double>(zero_point);
        const int64_t q =
            std::min<int64_t>(quant_max,
                              std::max<int64_t>(quant_min,
                                                static_cast<int64_t>(rounded)));
        output[i] = static_cast<int8_t>(q);
    }
}

template <typename T>
void dequantize_kernel(const int8_t* input, T* output, int64_t numel,
                       double scale, int64_t zero_point) {
    for (int64_t i = 0; i < numel; ++i) {
        output[i] = static_cast<T>((static_cast<double>(input[i]) -
                                    static_cast<double>(zero_point)) * scale);
    }
}

// Channel index of a contiguous element under `axis`-major layout.
inline int64_t channel_of(int64_t linear, int64_t stride_on_axis) {
    return stride_on_axis == 0 ? 0 : (linear / stride_on_axis);
}

} // namespace

Tensor quantize_per_tensor_cpu(const Tensor& self, double scale,
                               int64_t zero_point, int64_t quant_min,
                               int64_t quant_max) {
    check_qparams(scale, zero_point, quant_min, quant_max);
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::Int8, self.device());
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float64) {
        quantize_kernel<double>(input.data_ptr<double>(), out.data_ptr<int8_t>(),
                                numel, scale, zero_point, quant_min, quant_max);
    } else {
        quantize_kernel<float>(input.data_ptr<float>(), out.data_ptr<int8_t>(),
                               numel, scale, zero_point, quant_min, quant_max);
    }
    return out;
}

Tensor dequantize_per_tensor_cpu(const Tensor& self, double scale,
                                 int64_t zero_point) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "dequantize(): expected an Int8 tensor");
    }
    if (!(scale > 0.0)) {
        TP_THROW(ValueError, "dequantize(): scale must be positive");
    }
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    dequantize_kernel<float>(input.data_ptr<int8_t>(), out.data_ptr<float>(),
                             input.numel(), scale, zero_point);
    return out;
}

Tensor quantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                const Tensor& zero_points, int64_t axis) {
    if (scales.dim() != 1 || zero_points.shape() != scales.shape()) {
        TP_THROW(ValueError,
                 "quantize(): scales/zero_points must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "quantize(): axis out of range");
    }
    if (scales.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "quantize(): scales size must match the quantized dimension");
    }
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::Int8, self.device());

    // Contiguous strides let each flat index derive its channel id.
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    Tensor zp = zero_points.to(DType::Int64).contiguous();
    Tensor sc = scales.to(DType::Float64).contiguous();
    const double* sc_ptr = sc.data_ptr<double>();
    const int64_t* zp_ptr = zp.data_ptr<int64_t>();
    const int64_t channels = scales.size(0);
    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float64) {
        const double* in = input.data_ptr<double>();
        int8_t* outp = out.data_ptr<int8_t>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t c = channel_of(i, stride_on_axis) % channels;
            const double q = std::nearbyint(in[i] / sc_ptr[c]) +
                             static_cast<double>(zp_ptr[c]);
            outp[i] = static_cast<int8_t>(
                std::min<int64_t>(127, std::max<int64_t>(-128, static_cast<int64_t>(q))));
        }
    } else {
        const float* in = input.data_ptr<float>();
        int8_t* outp = out.data_ptr<int8_t>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t c = channel_of(i, stride_on_axis) % channels;
            const double q = std::nearbyint(static_cast<double>(in[i]) / sc_ptr[c]) +
                             static_cast<double>(zp_ptr[c]);
            outp[i] = static_cast<int8_t>(
                std::min<int64_t>(127, std::max<int64_t>(-128, static_cast<int64_t>(q))));
        }
    }
    return out;
}

Tensor dequantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                  const Tensor& zero_points, int64_t axis) {
    if (self.dtype() != DType::Int8) {
        TP_THROW(TypeError, "dequantize(): expected an Int8 tensor");
    }
    if (scales.dim() != 1 || zero_points.shape() != scales.shape()) {
        TP_THROW(ValueError,
                 "dequantize(): scales/zero_points must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "dequantize(): axis out of range");
    }
    if (scales.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "dequantize(): scales size must match the quantized dimension");
    }
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    Tensor zp = zero_points.to(DType::Int64).contiguous();
    Tensor sc = scales.to(DType::Float32).contiguous();
    const float* sc_ptr = sc.data_ptr<float>();
    const int64_t* zp_ptr = zp.data_ptr<int64_t>();
    const int64_t channels = scales.size(0);
    const int64_t numel = input.numel();
    const int8_t* in = input.data_ptr<int8_t>();
    float* outp = out.data_ptr<float>();
    for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = channel_of(i, stride_on_axis) % channels;
        outp[i] = (static_cast<float>(in[i]) - static_cast<float>(zp_ptr[c])) *
                  sc_ptr[c];
    }
    return out;
}

Tensor quantized_linear_cpu(const Tensor& input, const Tensor& weight,
                            double input_scale, int64_t input_zero_point,
                            const Tensor& weight_scales,
                            const Tensor& weight_zero_points,
                            std::optional<Tensor> bias) {
    // Fused Int8 GEMM with per-channel weight requantization (the dynamic
    // quantized linear output stage): out[m, n] = input_scale *
    // weight_scales[n] * Σ_k (x[m,k] - x_zp) * (w[n,k] - w_zp[n]) + bias[n].
    if (input.dtype() != DType::Int8 || weight.dtype() != DType::Int8) {
        TP_THROW(TypeError,
                 "quantized_linear(): activations and weights must be Int8");
    }
    if (input.dim() != 2 || weight.dim() != 2) {
        TP_THROW(ValueError,
                 "quantized_linear(): expected 2-D [M,K] activations and "
                 "[N,K] weights");
    }
    if (!(input_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_linear(): scale must be positive");
    }
    if (input.size(1) != weight.size(1)) {
        TP_THROW(ValueError,
                 "quantized_linear(): incompatible K dimensions (" +
                     std::to_string(input.size(1)) + " vs " +
                     std::to_string(weight.size(1)) + ")");
    }
    const int64_t out_features = weight.size(0);
    if (weight_scales.dim() != 1 || weight_scales.size(0) != out_features ||
        weight_zero_points.shape() != weight_scales.shape()) {
        TP_THROW(ValueError,
                 "quantized_linear(): weight scales/zero_points must be 1-D "
                 "of length out_features");
    }
    Tensor x = input.is_contiguous() ? input : input.contiguous();
    Tensor w = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor sc = weight_scales.to(DType::Float32).contiguous();
    Tensor zp = weight_zero_points.to(DType::Int64).contiguous();
    const int64_t* zp_ptr = zp.data_ptr<int64_t>();
    for (int64_t n = 0; n < out_features; ++n) {
        if (zp_ptr[n] < -128 || zp_ptr[n] > 127) {
            TP_THROW(ValueError,
                     "quantized_linear(): zero_point out of the Int8 range");
        }
    }

    Tensor bias_f;
    if (bias.has_value()) {
        if (!isFloatingType(bias->dtype()) || bias->dim() != 1 ||
            bias->size(0) != out_features) {
            TP_THROW(ValueError,
                     "quantized_linear(): bias must be a 1-D floating tensor "
                     "of length out_features");
        }
        bias_f = bias->to(DType::Float32).contiguous();
    } else {
        bias_f = Tensor::zeros({out_features}, DType::Float32, x.device());
    }

    const int64_t m_size = x.size(0);
    const int64_t k_size = x.size(1);
    Tensor out = Tensor::empty({m_size, out_features}, DType::Float32,
                               x.device());
    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = w.data_ptr<int8_t>();
    const float* sc_ptr = sc.data_ptr<float>();
    const float* b_ptr = bias_f.data_ptr<float>();

    parallel::parallel_for(
        0, m_size, /*grain_size=*/4,
        [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const int8_t* x_row = x_ptr + m * k_size;
            float* out_row = out.data_ptr<float>() + m * out_features;
            for (int64_t n = 0; n < out_features; ++n) {
                const int64_t w_zp = zp_ptr[n];
                const int8_t* w_row = w_ptr + n * k_size;
                int64_t acc = 0;
                for (int64_t k = 0; k < k_size; ++k) {
                    acc += static_cast<int64_t>(x_row[k] - input_zero_point) *
                           static_cast<int64_t>(w_row[k] - w_zp);
                }
                out_row[n] = static_cast<float>(input_scale) * sc_ptr[n] *
                                 static_cast<float>(acc) +
                             b_ptr[n];
            }
        }
    });
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cpu);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cpu);
    m.impl("quantize_per_channel", quantize_per_channel_cpu);
    m.impl("dequantize_per_channel", dequantize_per_channel_cpu);
    m.impl("quantized_linear", quantized_linear_cpu);
}

} // namespace cpu
} // namespace tensorplay
