#include "QuantKernels.h"
#include "Exception.h"
#include "Utils.h"

#include <cmath>
#include <vector>

namespace tensorplay {
namespace cpu {
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

TENSORPLAY_LIBRARY_IMPL(CPU, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cpu);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cpu);
    m.impl("quantize_per_channel", quantize_per_channel_cpu);
    m.impl("dequantize_per_channel", dequantize_per_channel_cpu);
}

} // namespace cpu
} // namespace tensorplay
