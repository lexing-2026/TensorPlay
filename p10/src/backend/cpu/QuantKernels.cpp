#include "QuantKernels.h"
#include "Convolution.h"
#include "Exception.h"
#include "Parallel.h"
#include "QuantConvPacking.h"
#include "Quantizer.h"
#include "Scalar.h"
#include "SizesAndStrides.h"
#include "Utils.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

// Defined in PoolingKernels.cpp; the quantized window maximum shares the
// float kernel's window logic order-preservingly on Int8 storage.
Tensor max_pool2d_cpu(const Tensor& input,
                      const std::vector<int64_t>& kernel_size,
                      const std::vector<int64_t>& stride,
                      const std::vector<int64_t>& padding,
                      const std::vector<int64_t>& dilation, bool ceil_mode);

namespace {

// Quantization is defined over real (floating) values only.  Narrow the
// dtype space up front: everything lands on Float32/Float64 compute and an
// Int8 storage, so the kernels below need exactly two scalar instantiations.
Tensor promote_to_compute_dtype(const Tensor& self) {
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        return self.is_contiguous() ? self : self.contiguous();
    }
    if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
        isFloat8Type(self.dtype())) {
        return self.to(DType::Float32).contiguous();
    }
    TP_THROW(TypeError,
             std::string("quantize(): expected a floating point tensor, got ") +
                 toString(self.dtype()));
}

void check_qparams(double scale, int64_t zero_point, int64_t quant_min,
                   int64_t quant_max) {
    if (!std::isfinite(scale) || !(scale > 0.0)) {
        TP_THROW(ValueError, "quantize(): scale must be positive");
    }
    if (quant_min >= quant_max) {
        TP_THROW(ValueError, "quantize(): quant_min must be < quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError, "quantize(): zero_point out of the quantized range");
    }
}

void check_storage_range(int64_t quant_min, int64_t quant_max,
                         int64_t storage_min, int64_t storage_max) {
    if (quant_min < storage_min || quant_max > storage_max) {
        TP_THROW(ValueError,
                 "quantize(): quantization range does not fit the output storage");
    }
}

// Affine quantization arithmetic.  The grid position is computed in single
// precision as value * (1 / scale): computing the reciprocal once and
// multiplying is the code path that every backend (including the SIMD
// paths) shares, and dividing in extended precision would resolve
// near-halfway products differently.  Rounding is round-half-even.
inline float quantize_multiplier(double scale) {
    return 1.0f / static_cast<float>(scale);
}

inline int64_t grid_position(float value, float inv_scale) {
    return static_cast<int64_t>(std::nearbyint(value * inv_scale));
}

template <typename T>
void quantize_kernel(const T* input, int8_t* output, int64_t numel,
                     double scale, int64_t zero_point, int64_t quant_min,
                     int64_t quant_max) {
    const float inv_scale = quantize_multiplier(scale);
    for (int64_t i = 0; i < numel; ++i) {
        const int64_t rounded =
            static_cast<int64_t>(zero_point) +
            grid_position(static_cast<float>(input[i]), inv_scale);
        const int64_t q =
            std::min<int64_t>(quant_max,
                              std::max<int64_t>(quant_min, rounded));
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

template <typename T>
void quantize_generic_kernel(const T* input, int64_t* output, int64_t numel,
                             double scale, int64_t zero_point,
                             int64_t quant_min, int64_t quant_max) {
    const float inv_scale = quantize_multiplier(scale);
    for (int64_t i = 0; i < numel; ++i) {
        const int64_t rounded =
            static_cast<int64_t>(zero_point) +
            grid_position(static_cast<float>(input[i]), inv_scale);
        output[i] =
            std::min<int64_t>(quant_max,
                              std::max<int64_t>(quant_min, rounded));
    }
}

} // namespace

Tensor quantize_per_tensor_quint8_cpu(const Tensor& self, double scale,
                                       int64_t zero_point, int64_t quant_min,
                                       int64_t quant_max) {
    check_qparams(scale, zero_point, quant_min, quant_max);
    check_storage_range(quant_min, quant_max, 0, 255);
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::QUInt8, self.device());
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    if (input.dtype() == DType::Float64) {
        quantize_generic_kernel<double>(
            input.data_ptr<double>(), codes.data_ptr<int64_t>(),
            input.numel(), scale, zero_point, quant_min, quant_max);
    } else {
        quantize_generic_kernel<float>(
            input.data_ptr<float>(), codes.data_ptr<int64_t>(),
            input.numel(), scale, zero_point, quant_min, quant_max);
    }
    const int64_t* pc = codes.data_ptr<int64_t>();
    uint8_t* po = out.data_ptr<uint8_t>();
    for (int64_t i = 0; i < codes.numel(); ++i) {
        po[i] = static_cast<uint8_t>(pc[i]);
    }
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8));
    return out;
}

Tensor dequantize_per_tensor_quint8_cpu(const Tensor& self, double scale,
                                         int64_t zero_point) {
    if (self.dtype() != DType::QUInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QUInt8 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8);
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const uint8_t* pi = input.data_ptr<uint8_t>();
    float* po = out.data_ptr<float>();
    const int64_t numel = input.numel();
    for (int64_t i = 0; i < numel; ++i) {
        po[i] = static_cast<float>(
            (static_cast<double>(pi[i]) - static_cast<double>(zero_point)) *
            scale);
    }
    return out;
}

Tensor quantize_per_tensor_qint32_cpu(const Tensor& self, double scale,
                                       int64_t zero_point) {
    check_qparams(scale, zero_point, std::numeric_limits<int32_t>::min(),
                  std::numeric_limits<int32_t>::max());
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::QInt32, self.device());
    Tensor codes = Tensor::empty(self.shape(), DType::Int64, self.device());
    if (input.dtype() == DType::Float64) {
        quantize_generic_kernel<double>(
            input.data_ptr<double>(), codes.data_ptr<int64_t>(),
            input.numel(), scale, zero_point, -2147483647LL - 1,
            2147483647LL);
    } else {
        quantize_generic_kernel<float>(
            input.data_ptr<float>(), codes.data_ptr<int64_t>(),
            input.numel(), scale, zero_point, -2147483647LL - 1,
            2147483647LL);
    }
    const int64_t* pc = codes.data_ptr<int64_t>();
    int32_t* po = out.data_ptr<int32_t>();
    const int64_t numel = codes.numel();
    for (int64_t i = 0; i < numel; ++i) {
        po[i] = static_cast<int32_t>(pc[i]);
    }
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32));
    return out;
}

Tensor dequantize_per_tensor_qint32_cpu(const Tensor& self, double scale,
                                         int64_t zero_point) {
    if (self.dtype() != DType::QInt32) {
        TP_THROW(TypeError, "dequantize(): expected a QInt32 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32);
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());
    const int32_t* pi = input.data_ptr<int32_t>();
    float* po = out.data_ptr<float>();
    const int64_t numel = input.numel();
    for (int64_t i = 0; i < numel; ++i) {
        po[i] = static_cast<float>(
            (static_cast<double>(pi[i]) - static_cast<double>(zero_point)) *
            scale);
    }
    return out;
}

Tensor quantize_per_tensor_cpu(const Tensor& self, double scale,
                               int64_t zero_point, int64_t quant_min,
                               int64_t quant_max) {
    check_qparams(scale, zero_point, quant_min, quant_max);
    check_storage_range(quant_min, quant_max, -128, 127);
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float64) {
        quantize_kernel<double>(input.data_ptr<double>(), out.data_ptr<int8_t>(),
                                numel, scale, zero_point, quant_min, quant_max);
    } else {
        quantize_kernel<float>(input.data_ptr<float>(), out.data_ptr<int8_t>(),
                               numel, scale, zero_point, quant_min, quant_max);
    }
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8));
    return out;
}

Tensor dequantize_per_tensor_cpu(const Tensor& self, double scale,
                                 int64_t zero_point) {
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QInt8 tensor");
    }
    make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8);
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
    if (scales.device() != self.device() ||
        zero_points.device() != self.device()) {
        TP_THROW(RuntimeError,
                 "quantize(): scales and zero_points must be on the input device");
    }
    QuantizerPtr quantizer = make_per_channel_affine_quantizer(
        scales, zero_points, axis, DType::QInt8);
    Tensor input = promote_to_compute_dtype(self);
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());

    // Contiguous strides let each flat index derive its channel id.
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    Tensor sc = scales.to(DType::Float64).contiguous();
    const double* sc_ptr = sc.data_ptr<double>();
    const int64_t channels = scales.size(0);
    const int64_t numel = input.numel();
    int8_t* outp = out.data_ptr<int8_t>();
    if (isFloatingType(zero_points.dtype())) {
        Tensor zp = zero_points.to(DType::Float32).contiguous();
        const float* zp_ptr = zp.data_ptr<float>();
        if (input.dtype() == DType::Float64) {
            const double* in = input.data_ptr<double>();
            for (int64_t i = 0; i < numel; ++i) {
                const int64_t c = channel_of(i, stride_on_axis) % channels;
                const float q = std::nearbyint(
                    static_cast<float>(in[i]) * quantize_multiplier(sc_ptr[c]) +
                    zp_ptr[c]);
                outp[i] = static_cast<int8_t>(std::min(127.0f,
                                                        std::max(-128.0f, q)));
            }
        } else {
            const float* in = input.data_ptr<float>();
            for (int64_t i = 0; i < numel; ++i) {
                const int64_t c = channel_of(i, stride_on_axis) % channels;
                const float q = std::nearbyint(
                    in[i] * quantize_multiplier(sc_ptr[c]) + zp_ptr[c]);
                outp[i] = static_cast<int8_t>(std::min(127.0f,
                                                        std::max(-128.0f, q)));
            }
        }
    } else {
        Tensor zp = zero_points.to(DType::Int64).contiguous();
        const int64_t* zp_ptr = zp.data_ptr<int64_t>();
        if (input.dtype() == DType::Float64) {
            const double* in = input.data_ptr<double>();
            for (int64_t i = 0; i < numel; ++i) {
                const int64_t c = channel_of(i, stride_on_axis) % channels;
                const int64_t q = zp_ptr[c] +
                                  grid_position(static_cast<float>(in[i]),
                                                quantize_multiplier(sc_ptr[c]));
                outp[i] = static_cast<int8_t>(
                    std::min<int64_t>(127, std::max<int64_t>(-128, q)));
            }
        } else {
            const float* in = input.data_ptr<float>();
            for (int64_t i = 0; i < numel; ++i) {
                const int64_t c = channel_of(i, stride_on_axis) % channels;
                const int64_t q = zp_ptr[c] +
                                  grid_position(in[i],
                                                quantize_multiplier(sc_ptr[c]));
                outp[i] = static_cast<int8_t>(
                    std::min<int64_t>(127, std::max<int64_t>(-128, q)));
            }
        }
    }
    out.impl()->set_quantizer(std::move(quantizer));
    return out;
}

Tensor dequantize_per_channel_cpu(const Tensor& self, const Tensor& scales,
                                  const Tensor& zero_points, int64_t axis) {
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "dequantize(): expected a QInt8 tensor");
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
    if (scales.device() != self.device() ||
        zero_points.device() != self.device()) {
        TP_THROW(RuntimeError,
                 "dequantize(): scales and zero_points must be on the input device");
    }
    QuantizerPtr quantizer = make_per_channel_affine_quantizer(
        scales, zero_points, axis, DType::QInt8);
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::Float32, self.device());

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) stride_on_axis *= input.size(d);

    Tensor sc = scales.to(DType::Float32).contiguous();
    const float* sc_ptr = sc.data_ptr<float>();
    const int64_t channels = scales.size(0);
    const int64_t numel = input.numel();
    const int8_t* in = input.data_ptr<int8_t>();
    float* outp = out.data_ptr<float>();
    if (isFloatingType(zero_points.dtype())) {
        Tensor zp = zero_points.to(DType::Float32).contiguous();
        const float* zp_ptr = zp.data_ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t c = channel_of(i, stride_on_axis) % channels;
            outp[i] = (static_cast<float>(in[i]) - zp_ptr[c]) * sc_ptr[c];
        }
    } else {
        Tensor zp = zero_points.to(DType::Int64).contiguous();
        const int64_t* zp_ptr = zp.data_ptr<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t c = channel_of(i, stride_on_axis) % channels;
            outp[i] = (static_cast<float>(in[i]) -
                       static_cast<float>(zp_ptr[c])) * sc_ptr[c];
        }
    }
    (void)quantizer;
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
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_linear(): activations and weights must be QInt8");
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

// ---------------------------------------------------------------------------
// Quantized elementwise arithmetic over Int8 storage with explicit qparams.
// Each operand is dequantized as (q - zero_point) * scale, the float
// operation is applied in double precision, and the result is requantized
// into [-128, 127]: q = clamp(nearbyint(y / out_scale) + out_zero_point),
// the same rounding convention as quantize_per_tensor above.  Division by
// zero follows IEEE float rules in the dequantized domain.
// ---------------------------------------------------------------------------

namespace {

inline int8_t requantize_value(double y, double inv_out_scale,
                               int64_t out_zero_point) {
    const double rounded =
        std::nearbyint(y * inv_out_scale) + static_cast<double>(out_zero_point);
    return static_cast<int8_t>(
        std::min<int64_t>(127, std::max<int64_t>(-128,
                                                 static_cast<int64_t>(rounded))));
}

void check_quantized_binary(
    const Tensor& a, const Tensor& b, double out_scale) {
    if (a.dtype() != DType::QInt8 || b.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized op(): operands must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized op(): out_scale must be positive");
    }
}

Tensor quantized_binary_cpu(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point,
    int op) {
    check_quantized_binary(a, b, out_scale);
    const auto out_sizes =
        broadcast_shapes(static_cast<std::vector<int64_t>>(a.shape()),
                         static_cast<std::vector<int64_t>>(b.shape()));
    Tensor out = Tensor::empty(out_sizes, DType::QInt8, a.device());

    // Broadcast strides pin singleton dimensions with a zero step, so one
    // flat index decomposition addresses both operands without materializing
    // an expanded copy.
    const std::vector<int64_t> sa = broadcast_strides(a, out_sizes);
    const std::vector<int64_t> sb = broadcast_strides(b, out_sizes);
    const std::vector<int64_t> so =
        SizesAndStrides::compute_contiguous_strides(out_sizes);

    const Tensor ac = a.is_contiguous() ? a : a.contiguous();
    const Tensor bc = b.is_contiguous() ? b : b.contiguous();
    const int8_t* pa = ac.data_ptr<int8_t>();
    const int8_t* pb = bc.data_ptr<int8_t>();
    int8_t* po = out.data_ptr<int8_t>();
    const int64_t numel = out.numel();
    const double inv_out = 1.0 / out_scale;

    for (int64_t flat = 0; flat < numel; ++flat) {
        int64_t rem = flat;
        int64_t ia = 0, ib = 0;
        for (size_t d = 0; d < so.size(); ++d) {
            const int64_t coord = rem / so[d];
            rem -= coord * so[d];
            ia += coord * sa[d];
            ib += coord * sb[d];
        }
        const double xa = (static_cast<double>(pa[ia]) -
                           static_cast<double>(a_zero_point)) * a_scale;
        const double xb = (static_cast<double>(pb[ib]) -
                           static_cast<double>(b_zero_point)) * b_scale;
        double y;
        switch (op) {
            case 0: y = xa + xb; break;
            case 1: y = xa - xb; break;
            case 2: y = xa * xb; break;
            default: y = xa / xb; break;
        }
        po[flat] = requantize_value(y, inv_out, out_zero_point);
    }
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

} // namespace

Tensor quantized_add_cpu(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cpu(a, b, a_scale, a_zero_point, b_scale,
                                b_zero_point, out_scale, out_zero_point, 0);
}

Tensor quantized_sub_cpu(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cpu(a, b, a_scale, a_zero_point, b_scale,
                                b_zero_point, out_scale, out_zero_point, 1);
}

Tensor quantized_mul_cpu(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cpu(a, b, a_scale, a_zero_point, b_scale,
                                b_zero_point, out_scale, out_zero_point, 2);
}

Tensor quantized_div_cpu(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
    return quantized_binary_cpu(a, b, a_scale, a_zero_point, b_scale,
                                b_zero_point, out_scale, out_zero_point, 3);
}

Tensor quantized_clamp_cpu(
    const Tensor& self, double self_scale, int64_t self_zero_point,
    double out_scale, int64_t out_zero_point,
    std::optional<Scalar> min, std::optional<Scalar> max) {
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized_clamp(): expected a QInt8 tensor");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_clamp(): out_scale must be positive");
    }
    const Tensor sc = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), DType::QInt8, self.device());

    const bool has_min = min.has_value();
    const bool has_max = max.has_value();
    const double lo = has_min ? min->toDouble() : 0.0;
    const double hi = has_max ? max->toDouble() : 0.0;

    const int8_t* in = sc.data_ptr<int8_t>();
    int8_t* outp = out.data_ptr<int8_t>();
    const int64_t numel = self.numel();
    const double inv_out = 1.0 / out_scale;

    for (int64_t i = 0; i < numel; ++i) {
        double y = (static_cast<double>(in[i]) -
                    static_cast<double>(self_zero_point)) * self_scale;
        if (has_min && y < lo) y = lo;
        if (has_max && y > hi) y = hi;
        outp[i] = requantize_value(y, inv_out, out_zero_point);
    }
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

Tensor quantized_max_pool2d_cpu(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    // The window maximum is order-preserving in the quantized domain, so the
    // pooling runs on an Int8 view of the code storage and the output is
    // re-wrapped with the input quantizer untouched.
    if (self.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "quantized_max_pool2d(): expected a QInt8 tensor");
    }
    Tensor codes = quantized::strip_quantizer(self);
    Tensor out_codes =
        max_pool2d_cpu(codes, kernel_size, stride, padding, dilation,
                       ceil_mode);
    return quantized::make_qtensor(out_codes, self.impl()->quantizer(),
                                   DType::QInt8);
}

Tensor quantized_conv2d_cpu(
    const Tensor& input, const Tensor& weight, std::optional<Tensor> bias,
    double input_scale, int64_t input_zero_point, double weight_scale,
    int64_t weight_zero_point, double out_scale, int64_t out_zero_point,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dtype() != DType::QInt8 || weight.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "quantized_conv2d(): activations and weights must be QInt8");
    }
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_conv2d(): out_scale must be positive");
    }
    // Dequantize both operands, run the float convolution (the bias already
    // lives in the float domain), then requantize into the output qparams.
    Tensor x = dequantize_per_tensor_cpu(
        input, input_scale, input_zero_point);
    Tensor w = dequantize_per_tensor_cpu(
        weight, weight_scale, weight_zero_point);
    Tensor acc = conv2d_cpu(
        x, w,
        bias.has_value() ? bias->to(DType::Float32).contiguous()
                         : Tensor(),
        stride, padding, dilation, groups);

    Tensor out = Tensor::empty(
        static_cast<std::vector<int64_t>>(acc.shape()), DType::QInt8,
        input.device());
    const float* pa = acc.data_ptr<float>();
    int8_t* po = out.data_ptr<int8_t>();
    const int64_t numel = acc.numel();
    const double inv_out = 1.0 / out_scale;
    for (int64_t i = 0; i < numel; ++i) {
        po[i] = requantize_value(pa[i], inv_out, out_zero_point);
    }
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zero_point, DType::QInt8));
    return out;
}

// ---------------------------------------------------------------------------
// Fake quantization: map real values through the affine Int8 grid and back,
// with a cached in-range mask for the backward pass.  Rounding is
// round-half-even; the raw (pre-clamp) grid position decides the mask.
// ---------------------------------------------------------------------------

namespace {

constexpr double kSmallScaleThreshold = 6.1e-5;

// Derives affine qparams from a real range [min, max] over the grid
// [qmin, qmax]: the interval is widened to contain 0, scale =
// (max - min) / (qmax - qmin) with degenerate/too-small ranges repaired,
// and the zero point is nudged into the grid with round-half-even.
// preserve_sparsity forces a symmetric range and a centered zero point.
struct QParams {
    double scale;
    int64_t zero_point;
};

QParams choose_qparams_cpu(double min, double max, int64_t qmin,
                           int64_t qmax, bool preserve_sparsity) {
    TP_CHECK(min <= max, "choose qparams: min must be <= max");
    if (min < 0 && max > 0 && preserve_sparsity) {
        const int64_t symmetric_qmin = -((qmax - qmin) / 2 + 1);
        const int64_t symmetric_qmax = (qmax - qmin) / 2;
        const double max_scale = std::max(
            std::fabs(min / static_cast<double>(symmetric_qmin)),
            std::fabs(max / static_cast<double>(symmetric_qmax)));
        min = max_scale * static_cast<double>(symmetric_qmin);
        max = max_scale * static_cast<double>(symmetric_qmax);
    }

    // Widen so that 0 stays exactly representable on the grid.
    min = std::min(min, 0.0);
    max = std::max(max, 0.0);
    TP_CHECK(qmin < qmax, "choose qparams: qmin must be < qmax");

    double scale = (max - min) / static_cast<double>(qmax - qmin);
    if (static_cast<float>(scale) == 0.0f ||
        std::isinf(1.0f / static_cast<float>(scale))) {
        scale = 0.1;
    }
    // Avoid subnormal-range scales; widen the range to compensate.
    if (scale < kSmallScaleThreshold) {
        const double org_scale = scale;
        scale = kSmallScaleThreshold;
        if (min == 0.0) {
            max = kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else if (max == 0.0) {
            min = -kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else {
            const double amplifier = kSmallScaleThreshold / org_scale;
            min *= amplifier;
            max *= amplifier;
        }
    }

    // Pick the zero-point anchor with the smaller accumulated rounding error.
    const double zero_point_from_min =
        static_cast<double>(qmin) - min / scale;
    const double zero_point_from_max =
        static_cast<double>(qmax) - max / scale;
    const double zero_point_from_min_error =
        std::abs(static_cast<double>(qmin)) - std::abs(min / scale);
    const double zero_point_from_max_error =
        std::abs(static_cast<double>(qmax)) - std::abs(max / scale);
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;
    if (min < 0 && max > 0 && preserve_sparsity) {
        initial_zero_point =
            static_cast<double>(qmin + qmax) / 2.0;
    }

    int64_t nudged_zero_point = 0;
    if (initial_zero_point < static_cast<double>(qmin)) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > static_cast<double>(qmax)) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
            static_cast<int64_t>(std::nearbyint(initial_zero_point));
    }
    return {scale, nudged_zero_point};
}

void check_fake_quant_range(int64_t zero_point, int64_t quant_min,
                            int64_t quant_max) {
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): zero_point must be between quant_min and "
                 "quant_max");
    }
}

void check_real_dtype(const Tensor& self, const char* op) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a floating point tensor, got " +
                     toString(self.dtype()));
    }
}

// One element of the fake-quant round trip.  Returns the fake-quantized
// value and whether the raw grid position landed inside [quant_min,
// quant_max].  ACum is the compute type (float or double).
template <typename T, typename ACum>
inline std::pair<T, bool> fake_quant_element(T x, ACum inv_scale,
                                             int64_t zero_point,
                                             int64_t quant_min,
                                             int64_t quant_max) {
    const ACum raw = std::nearbyint(static_cast<ACum>(x) * inv_scale) +
                     static_cast<ACum>(zero_point);
    const int64_t q = std::min<int64_t>(
        quant_max, std::max<int64_t>(quant_min,
                                     static_cast<int64_t>(raw)));
    return {static_cast<T>((static_cast<ACum>(q) -
                            static_cast<ACum>(zero_point)) *
                           (static_cast<ACum>(1.0) / inv_scale)),
            raw >= static_cast<ACum>(quant_min) &&
                raw <= static_cast<ACum>(quant_max)};
}

} // namespace

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max);

std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask_cpu(
    const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    check_fake_quant_range(zero_point, quant_min, quant_max);
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const int64_t numel = input.numel();
    const double inv_scale = 1.0 / scale;
    switch (self.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            float* op = out.data_ptr<float>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<float, double>(
                    in[i], inv_scale, zero_point, quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            double* op = out.data_ptr<double>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<double, double>(
                    in[i], inv_scale, zero_point, quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float16: {
            const Half* in = input.data_ptr<Half>();
            Half* op = out.data_ptr<Half>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<Half, float>(
                    in[i], static_cast<float>(inv_scale), zero_point,
                    quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::BFloat16: {
            const BFloat16* in = input.data_ptr<BFloat16>();
            BFloat16* op = out.data_ptr<BFloat16>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<BFloat16, float>(
                    in[i], static_cast<float>(inv_scale), zero_point,
                    quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_tensor_affine_cpu(const Tensor& self, double scale,
                                           int64_t zero_point,
                                           int64_t quant_min,
                                           int64_t quant_max) {
    return std::get<0>(fake_quantize_per_tensor_affine_cachemask_cpu(
        self, scale, zero_point, quant_min, quant_max));
}

// Tensor-qparams overload: the enable flag can suspend fake quantization,
// leaving the input untouched with a fully open mask.
Tensor fake_quantize_per_tensor_affine_tensor_qparams_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max) {
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return std::get<0>(
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu(
            self, scale, zero_point, enabled, quant_min, quant_max));
}

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1 &&
                 fake_quant_enabled.numel() == 1,
             "fake_quantize(): scale, zero_point and the fake-quant flag "
             "must be 1-element tensors");
    const double scale_val = scale.item().toDouble();
    const int64_t zero_point_val =
        static_cast<int64_t>(std::nearbyint(zero_point.item().toDouble()));
    const bool fake_on = fake_quant_enabled.item().to<int64_t>() != 0;

    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    const int64_t numel = input.numel();
    bool* mp = mask.data_ptr<bool>();

    if (!fake_on) {
        // Suspended fake quant: identity output, gradient unmasked.
        switch (self.dtype()) {
            case DType::Float32:
                std::memcpy(out.data_ptr<float>(),
                            input.data_ptr<float>(),
                            static_cast<size_t>(numel) * sizeof(float));
                break;
            case DType::Float64:
                std::memcpy(out.data_ptr<double>(),
                            input.data_ptr<double>(),
                            static_cast<size_t>(numel) * sizeof(double));
                break;
            case DType::Float16:
                std::memcpy(out.data_ptr<Half>(), input.data_ptr<Half>(),
                            static_cast<size_t>(numel) * sizeof(Half));
                break;
            case DType::BFloat16:
                std::memcpy(out.data_ptr<BFloat16>(),
                            input.data_ptr<BFloat16>(),
                            static_cast<size_t>(numel) * sizeof(BFloat16));
                break;
            default:
                TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                    "unsupported input dtype");
        }
        for (int64_t i = 0; i < numel; ++i) mp[i] = true;
        return {std::move(out), std::move(mask)};
    }
    check_fake_quant_range(zero_point_val, quant_min, quant_max);
    switch (self.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            float* op = out.data_ptr<float>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<float, double>(
                    in[i], 1.0 / scale_val, zero_point_val, quant_min,
                    quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            double* op = out.data_ptr<double>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<double, double>(
                    in[i], 1.0 / scale_val, zero_point_val, quant_min,
                    quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float16: {
            const Half* in = input.data_ptr<Half>();
            Half* op = out.data_ptr<Half>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<Half, float>(
                    in[i], static_cast<float>(1.0 / scale_val),
                    zero_point_val, quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::BFloat16: {
            const BFloat16* in = input.data_ptr<BFloat16>();
            BFloat16* op = out.data_ptr<BFloat16>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = fake_quant_element<BFloat16, float>(
                    in[i], static_cast<float>(1.0 / scale_val),
                    zero_point_val, quant_min, quant_max);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    return {std::move(out), std::move(mask)};
}

namespace {

// grad * mask, evaluated elementwise; the mask blocks gradients that the
// saturating forward produced.
template <typename T>
Tensor masked_grad_cpu(const Tensor& grad, const Tensor& mask) {
    Tensor out = Tensor::empty(grad.shape(), grad.dtype(), grad.device());
    const int64_t numel = grad.numel();
    const T* gp = grad.data_ptr<T>();
    const bool* mp = mask.data_ptr<bool>();
    T* op = out.data_ptr<T>();
    for (int64_t i = 0; i < numel; ++i) {
        op[i] = mp[i] ? gp[i] : static_cast<T>(0);
    }
    return out;
}

} // namespace

Tensor fake_quantize_per_tensor_affine_cachemask_backward_cpu(
    const Tensor& grad, const Tensor& mask) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "fake_quantize backward: mask must be Bool");
    }
    if (mask.numel() != grad.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: mask and grad must have the same "
                 "number of elements");
    }
    if (grad.numel() == 0) return grad;
    const Tensor gc = grad.is_contiguous() ? grad : grad.contiguous();
    const Tensor mc = mask.is_contiguous() ? mask : mask.contiguous();
    switch (gc.dtype()) {
        case DType::Float32: return masked_grad_cpu<float>(gc, mc);
        case DType::Float64: return masked_grad_cpu<double>(gc, mc);
        case DType::Float16: return masked_grad_cpu<Half>(gc, mc);
        case DType::BFloat16: return masked_grad_cpu<BFloat16>(gc, mc);
        default:
            TP_THROW(TypeError, "fake_quantize backward: expected a "
                                "floating point gradient");
    }
}

namespace {

// Learnable-qparams backward: dX is a straight-through inside the
// representable range and zero outside; dScale and dZeroPoint accumulate
// one contribution per element:
//   dScale   = (qmin - z) if x_q < qmin; (qmax - z) if x_q > qmax;
//              else (x_fq - x) / scale
//   dZeroPoint = -scale if x_q saturates, else 0
// all scaled by grad_factor.
template <typename T>
void learnable_backward_kernel_cpu(
    const T* x, const T* dy, int64_t numel, double scale, double inv_scale,
    int64_t zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor, T* dx, T* dscale, T* dzp) {
    const float dscale_small =
        static_cast<float>(quant_min - zero_point);
    const float dscale_big = static_cast<float>(quant_max - zero_point);
    for (int64_t i = 0; i < numel; ++i) {
        const float xf = static_cast<float>(x[i]);
        const float dyf = static_cast<float>(dy[i]);
        const int64_t xq = static_cast<int64_t>(std::nearbyint(
                                xf * static_cast<float>(inv_scale))) +
                           zero_point;
        dx[i] = static_cast<T>(dyf * (xq >= quant_min && xq <= quant_max));
        const float xfq = static_cast<float>(
            (std::min<int64_t>(std::max<int64_t>(xq, quant_min), quant_max) -
             zero_point) * scale);
        if (xq < quant_min || xq > quant_max) {
            dscale[i] = static_cast<T>(
                (dyf * ((xq < quant_min) ? dscale_small : dscale_big)) *
                grad_factor);
            dzp[i] = static_cast<T>(dyf * (-1.0f) *
                                    static_cast<float>(scale) * grad_factor);
        } else {
            dscale[i] = static_cast<T>(
                dyf * (xfq - xf) * static_cast<float>(inv_scale) *
                grad_factor);
            dzp[i] = static_cast<T>(0);
        }
    }
}

std::tuple<Tensor, Tensor, Tensor> learnable_backward_cpu(
    const Tensor& grad, const Tensor& x, double scale_val,
    int64_t zero_point_val, int64_t quant_min, int64_t quant_max,
    double grad_factor, DType out_dtype) {
    const int64_t numel = x.numel();
    Tensor dx = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    if (out_dtype == DType::Float64) {
        learnable_backward_kernel_cpu<double>(
            x.data_ptr<double>(), grad.data_ptr<double>(), numel, scale_val,
            1.0 / scale_val, zero_point_val, quant_min, quant_max,
            grad_factor, dx.data_ptr<double>(), dscale_vec.data_ptr<double>(),
            dzp_vec.data_ptr<double>());
    } else {
        learnable_backward_kernel_cpu<float>(
            x.data_ptr<float>(), grad.data_ptr<float>(), numel, scale_val,
            1.0 / scale_val, zero_point_val, quant_min, quant_max,
            grad_factor, dx.data_ptr<float>(), dscale_vec.data_ptr<float>(),
            dzp_vec.data_ptr<float>());
    }
    return {std::move(dx), std::move(dscale_vec.sum().reshape({1})),
            std::move(dzp_vec.sum().reshape({1}))};
}

// Promotion for the learnable backward: Float32/Float64 pass through, the
// reduced formats land on Float32.
std::tuple<Tensor, Tensor, DType> promote_learnable_pair(const Tensor& grad,
                                                         const Tensor& x) {
    check_real_dtype(x, "fake_quantize backward");
    if (!isFloatingType(grad.dtype())) {
        TP_THROW(TypeError,
                 "fake_quantize backward: expected a floating point grad");
    }
    DType compute = x.dtype();
    if (compute == DType::Float16 || compute == DType::BFloat16) {
        compute = DType::Float32;
    }
    Tensor xc = (x.dtype() == compute) ? x : x.to(compute);
    Tensor gc = (grad.dtype() == compute) ? grad : grad.to(compute);
    xc = xc.is_contiguous() ? xc : xc.contiguous();
    gc = gc.is_contiguous() ? gc : gc.contiguous();
    if (xc.numel() != gc.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: X and dY must have the same number "
                 "of elements");
    }
    return {gc, xc, compute};
}

} // namespace

Tensor _fake_quantize_learnable_per_tensor_affine_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize(): scale and zero_point must be 1-element "
             "tensors");
    const double scale_val = scale.item().toDouble();
    // Forward: the zero point is rounded to the grid and clamped.
    double zp_fp = zero_point.item().toDouble();
    zp_fp = std::nearbyint(zp_fp);
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    return fake_quantize_per_tensor_affine_cpu(
        self, scale_val, static_cast<int64_t>(zp_fp), quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward_cpu(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor) {
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize backward: scale and zero_point must be "
             "1-element tensors");
    // Backward nudges the zero point half a grid step before clamping.
    double zp_fp = zero_point.item().toDouble() + 0.5;
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    const int64_t zero_point_val = static_cast<int64_t>(zp_fp);
    if (zero_point_val < quant_min || zero_point_val > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize backward: zero_point out of the quantized "
                 "range");
    }
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }
    auto promoted = promote_learnable_pair(grad, self);
    return learnable_backward_cpu(
        std::get<0>(promoted), std::get<1>(promoted),
        scale.item().toDouble(), zero_point_val, quant_min, quant_max,
        grad_factor, std::get<2>(promoted));
}

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_channel_affine");
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1,
             "fake_quantize(): scale and zero_point must be 1-D tensors");
    TP_CHECK(scale.numel() == zero_point.numel(),
             "fake_quantize(): scale and zero_point must have the same "
             "size");
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize(): axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize(): scale size must match the quantized "
             "dimension");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    const bool zp_is_float = !isIntegralType(zero_point.dtype());
    if (!zp_is_float) {
        Tensor zpc = zero_point.to(DType::Int64).contiguous();
        const int64_t* zp = zpc.data_ptr<int64_t>();
        for (int64_t i = 0; i < zpc.numel(); ++i) {
            if (zp[i] < quant_min || zp[i] > quant_max) {
                TP_THROW(ValueError,
                         "fake_quantize(): zero_point out of the quantized "
                         "range");
            }
        }
    }

    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor sc = scale.to(DType::Float64).contiguous();
    Tensor zpi = zp_is_float ? zero_point.to(DType::Float64).contiguous()
                             : zero_point.to(DType::Int64).contiguous();
    const double* sc_ptr = sc.data_ptr<double>();
    const double* zpf_ptr = zp_is_float ? zpi.data_ptr<double>() : nullptr;
    const int64_t* zp_ptr = zp_is_float ? nullptr : zpi.data_ptr<int64_t>();
    const bool zp_float_flag = zp_is_float;

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) {
        stride_on_axis *= input.size(d);
    }
    const int64_t channels = scale.numel();
    const int64_t numel = input.numel();

    // Integral zero points round the grid position first; floating zero
    // points fold the shift into the rounding itself.
    auto element = [&](int64_t i, auto raw_in) -> std::pair<double, bool> {
        const int64_t c = (stride_on_axis == 0)
                              ? 0
                              : (i / stride_on_axis) % channels;
        const double inv_scale = 1.0 / sc_ptr[c];
        const double raw =
            zp_float_flag
                ? std::lrint(static_cast<double>(raw_in) * inv_scale +
                             zpf_ptr[c])
                : std::nearbyint(static_cast<double>(raw_in) * inv_scale) +
                      static_cast<double>(zp_ptr[c]);
        const int64_t q = std::min<int64_t>(
            quant_max,
            std::max<int64_t>(quant_min, static_cast<int64_t>(raw)));
        const double zpv = zp_float_flag
                               ? zpf_ptr[c]
                               : static_cast<double>(zp_ptr[c]);
        const double y =
            (static_cast<double>(q) - zpv) * sc_ptr[c];
        return {y, raw >= static_cast<double>(quant_min) &&
                       raw <= static_cast<double>(quant_max)};
    };

    switch (self.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            float* op = out.data_ptr<float>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = element(i, in[i]);
                op[i] = static_cast<float>(r.first);
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            double* op = out.data_ptr<double>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = element(i, in[i]);
                op[i] = r.first;
                mp[i] = r.second;
            }
            break;
        }
        case DType::Float16: {
            const Half* in = input.data_ptr<Half>();
            Half* op = out.data_ptr<Half>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = element(i, static_cast<float>(in[i]));
                op[i] = static_cast<Half>(static_cast<float>(r.first));
                mp[i] = r.second;
            }
            break;
        }
        case DType::BFloat16: {
            const BFloat16* in = input.data_ptr<BFloat16>();
            BFloat16* op = out.data_ptr<BFloat16>();
            bool* mp = mask.data_ptr<bool>();
            for (int64_t i = 0; i < numel; ++i) {
                auto r = element(i, static_cast<float>(in[i]));
                op[i] = static_cast<BFloat16>(static_cast<float>(r.first));
                mp[i] = r.second;
            }
            break;
        }
        default:
            TP_THROW(TypeError, "fake_quantize_per_channel_affine(): "
                                "unsupported input dtype");
    }
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_channel_affine_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    return std::get<0>(fake_quantize_per_channel_affine_cachemask_cpu(
        self, scale, zero_point, axis, quant_min, quant_max));
}

Tensor fake_quantize_per_channel_affine_cachemask_backward_cpu(
    const Tensor& grad, const Tensor& mask) {
    return fake_quantize_per_tensor_affine_cachemask_backward_cpu(grad,
                                                                  mask);
}

Tensor _fake_quantize_learnable_per_channel_affine_cpu(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    // Forward: round the zero points to the grid and clamp them.
    Tensor zp = zero_point.to(DType::Float32)
                    .round()
                    .clamp(Scalar(static_cast<int64_t>(quant_min)),
                           Scalar(static_cast<int64_t>(quant_max)))
                    .to(DType::Int64);
    return fake_quantize_per_channel_affine_cpu(
        self, scale.to(DType::Float32), zp, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward_cpu(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t axis, int64_t quant_min,
    int64_t quant_max, double grad_factor) {
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1 &&
                 scale.numel() == zero_point.numel(),
             "fake_quantize backward: scale and zero_point must be 1-D "
             "tensors of the same size");
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize backward: axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize backward: scale size must match the quantized "
             "dimension");
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }

    auto promoted = promote_learnable_pair(grad, self);
    const Tensor& x = std::get<1>(promoted);
    const DType compute = std::get<2>(promoted);

    // The zero points stay floating here: each element rounds its grid
    // position against its channel's shifted zero point.
    Tensor zp_f = zero_point.to(DType::Float32)
                      .round()
                      .clamp(Scalar(static_cast<int64_t>(quant_min)),
                             Scalar(static_cast<int64_t>(quant_max)))
                      .contiguous();
    Tensor sc = scale.to(DType::Float32).contiguous();
    const float* sc_ptr = sc.data_ptr<float>();
    const float* zp_ptr = zp_f.data_ptr<float>();
    const int64_t channels = scale.numel();

    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < x.dim(); ++d) stride_on_axis *= x.size(d);

    Tensor dx = Tensor::empty(x.shape(), compute, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), compute, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), compute, x.device());
    const int64_t numel = x.numel();
    const float dscale_small_base = static_cast<float>(quant_min);
    const float dscale_big_base = static_cast<float>(quant_max);

    auto kernel = [&](auto xk) {
        using T = decltype(xk);
        const T* xp = x.data_ptr<T>();
        const T* gp = std::get<0>(promoted).data_ptr<T>();
        T* dxp = dx.data_ptr<T>();
        T* dsp = dscale_vec.data_ptr<T>();
        T* dzp = dzp_vec.data_ptr<T>();
        for (int64_t i = 0; i < numel; ++i) {
            const int64_t c =
                (stride_on_axis == 0) ? 0 : (i / stride_on_axis) % channels;
            const float inv_scale = 1.0f / sc_ptr[c];
            const float zpf = zp_ptr[c];
            const float xf = static_cast<float>(xp[i]);
            const float dyf = static_cast<float>(gp[i]);
            const int64_t xq = static_cast<int64_t>(std::nearbyint(
                                    xf * inv_scale)) +
                               static_cast<int64_t>(zpf);
            dxp[i] = static_cast<T>(dyf *
                                    (xq >= quant_min && xq <= quant_max));
            const float xfq = static_cast<float>(
                (std::min<int64_t>(std::max<int64_t>(xq, quant_min),
                                   quant_max) -
                 static_cast<int64_t>(zpf)) * sc_ptr[c]);
            if (xq < quant_min || xq > quant_max) {
                dsp[i] = static_cast<T>(
                    dyf * ((xq < quant_min) ? (dscale_small_base - zpf)
                                            : (dscale_big_base - zpf)) *
                    grad_factor);
                dzp[i] = static_cast<T>(dyf * (-1.0f) * sc_ptr[c] *
                                        grad_factor);
            } else {
                dsp[i] = static_cast<T>(
                    dyf * (xfq - xf) * inv_scale * grad_factor);
                dzp[i] = static_cast<T>(0);
            }
        }
    };

    if (compute == DType::Float64) {
        kernel(0.0);
    } else {
        kernel(0.0f);
    }

    // Reduce the per-element contributions over every axis but the
    // quantized one, leaving one value per channel.
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < x.dim(); ++d) {
        if (d != axis) reduce_dims.push_back(d);
    }
    Tensor dscale = dscale_vec.sum(reduce_dims, false);
    Tensor dzp = dzp_vec.sum(reduce_dims, false);
    return {std::move(dx), std::move(dscale), std::move(dzp)};
}

// ---------------------------------------------------------------------------
// Dynamic quantization: derive qparams from the tensor's own min/max, then
// quantize to Int8/UInt8 storage.  reduce_range halves the quantized grid
// for faster low-precision accumulation.
// ---------------------------------------------------------------------------

namespace {

QParams dynamic_qparams_cpu(double x_min, double x_max, bool reduce_range,
                            int64_t qmin, int64_t qmax) {
    if (reduce_range) {
        qmin /= 2;
        qmax /= 2;
    }
    return choose_qparams_cpu(x_min, x_max, qmin, qmax,
                              /*preserve_sparsity=*/false);
}

template <typename T>
void dynamic_quantize_kernel_cpu(const float* input, T* output,
                                 int64_t numel, double scale,
                                 int64_t zero_point) {
    const float inv_scale = quantize_multiplier(scale);
    for (int64_t i = 0; i < numel; ++i) {
        const int64_t rounded =
            static_cast<int64_t>(zero_point) +
            grid_position(input[i], inv_scale);
        const int64_t q =
            std::min<int64_t>(127, std::max<int64_t>(-128, rounded));
        output[i] = static_cast<T>(q);
    }
}

template <typename T>
void dynamic_quantize_unsigned_kernel_cpu(const float* input, T* output,
                                          int64_t numel, double scale,
                                          int64_t zero_point) {
    const float inv_scale = quantize_multiplier(scale);
    for (int64_t i = 0; i < numel; ++i) {
        const int64_t rounded =
            static_cast<int64_t>(zero_point) +
            grid_position(input[i], inv_scale);
        const int64_t q =
            std::min<int64_t>(255, std::max<int64_t>(0, rounded));
        output[i] = static_cast<T>(q);
    }
}

} // namespace

Tensor quantize_per_tensor_dynamic_cpu(const Tensor& self, DType dtype,
                                       bool reduce_range) {
    if (dtype != DType::QInt8 && dtype != DType::QUInt8 &&
        dtype != DType::Float16) {
        TP_THROW(TypeError,
                 "quantize_per_tensor_dynamic(): only QInt8, QUInt8 and "
                 "Float16 outputs are supported");
    }
    check_real_dtype(self, "quantize_per_tensor_dynamic");
    if (dtype == DType::Float16) {
        return self.contiguous().to(DType::Float16);
    }
    const Tensor input = self.contiguous();
    auto mm = Tensor::aminmax(input.reshape({input.numel()}), {}, false);
    const double x_min = std::get<0>(mm).item().toDouble();
    const double x_max = std::get<1>(mm).item().toDouble();
    const int64_t qmin = (dtype == DType::QInt8) ? -128 : 0;
    const int64_t qmax = (dtype == DType::QInt8) ? 127 : 255;
    const QParams qp =
        dynamic_qparams_cpu(x_min, x_max, reduce_range, qmin, qmax);

    Tensor out = Tensor::empty(self.shape(), dtype, self.device());
    const Tensor input_f =
        (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);
    const float* in = input_f.data_ptr<float>();
    const int64_t numel = input.numel();
    if (dtype == DType::QInt8) {
        dynamic_quantize_kernel_cpu<int8_t>(in, out.data_ptr<int8_t>(),
                                            numel, qp.scale, qp.zero_point);
    } else {
        dynamic_quantize_unsigned_kernel_cpu<uint8_t>(
            in, out.data_ptr<uint8_t>(), numel, qp.scale, qp.zero_point);
    }
    // Dynamic quantization derives its parameters from the data, so the
    // result is a first-class quantized tensor carrying them.
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(qp.scale, qp.zero_point, dtype));
    return out;
}

std::tuple<double, int64_t> _choose_qparams_per_tensor_cpu(
    const Tensor& self, bool reduce_range) {
    check_real_dtype(self, "_choose_qparams_per_tensor");
    const Tensor input = self.contiguous();
    auto mm = Tensor::aminmax(input.reshape({input.numel()}), {}, false);
    const double x_min = std::get<0>(mm).item().toDouble();
    const double x_max = std::get<1>(mm).item().toDouble();
    const QParams qp = dynamic_qparams_cpu(x_min, x_max, reduce_range,
                                           /*qmin=*/0, /*qmax=*/255);
    return {qp.scale, qp.zero_point};
}

// ---------------------------------------------------------------------------
// Fused moving-average observer + fake quant: updates the running min/max
// state under the observer flag, derives qparams from the running range,
// and returns the fake-quantized output with the backward mask.
// ---------------------------------------------------------------------------

namespace {

void moving_average_update_cpu(const Tensor& x_min_in, const Tensor& x_max_in,
                               Tensor& running_min, Tensor& running_max,
                               float averaging_const) {
    const Tensor x_min = x_min_in.to(DType::Float32).contiguous();
    const Tensor x_max = x_max_in.to(DType::Float32).contiguous();
    float* rmin = running_min.data_ptr<float>();
    float* rmax = running_max.data_ptr<float>();
    const float* cmin = x_min.data_ptr<float>();
    const float* cmax = x_max.data_ptr<float>();
    for (int64_t i = 0; i < x_min.numel(); ++i) {
        rmin[i] = std::isinf(rmin[i])
                      ? cmin[i]
                      : rmin[i] + averaging_const * (cmin[i] - rmin[i]);
        rmax[i] = std::isinf(rmax[i])
                      ? cmax[i]
                      : rmax[i] + averaging_const * (cmax[i] - rmax[i]);
    }
}

void fill_or_resize_f32(Tensor& t, int64_t size, float value,
                        const Device& device) {
    if (t.numel() == 0) {
        t = Tensor::full({size}, Scalar(value), DType::Float32, device);
        return;
    }
    if (t.numel() != size) {
        t.resize_({size});
        t.fill_(Scalar(value));
        return;
    }
    t.fill_(Scalar(value));
}

} // namespace

std::tuple<Tensor, Tensor> _fused_moving_avg_obs_fq_helper_cpu(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (ch_axis >= self.dim()) {
        TP_THROW(ValueError,
                 "fused_moving_avg_obs_fake_quant(): ch_axis must be < "
                 "self.dim()");
    }
    check_real_dtype(self, "fused_moving_avg_obs_fake_quant");
    const bool observe = observer_on.item().to<int64_t>() != 0;
    const bool fake_on = fake_quant_on.item().to<int64_t>() != 0;

    if (per_row_fake_quant) {
        Tensor y = self;
        if (self.dim() != 2) {
            std::vector<int64_t> dims(self.dim());
            for (int64_t d = 0; d < self.dim(); ++d) dims[d] = d;
            dims[ch_axis] = 0;
            dims[0] = ch_axis;
            y = self.permute(dims).flatten(1);
        }
        const int64_t size = self.size(ch_axis);
        if (running_min.numel() == 0) {
            fill_or_resize_f32(running_min, size,
                               std::numeric_limits<float>::infinity(),
                               self.device());
            fill_or_resize_f32(running_max, size,
                               -std::numeric_limits<float>::infinity(),
                               self.device());
            scale.resize_({size});
            zero_point.resize_({size});
        }
        if (observe) {
            auto mm = Tensor::aminmax(y.contiguous(), {1}, false);
            moving_average_update_cpu(std::get<0>(mm), std::get<1>(mm),
                                      running_min, running_max,
                                      static_cast<float>(averaging_const));
        }
        if (!fake_on) {
            Tensor mask = Tensor::full_like(self, 1, DType::Bool);
            return {self.clone(), std::move(mask)};
        }
        // Derive per-channel qparams from the running range.
        Tensor rmin = running_min.to(DType::Float32).contiguous();
        Tensor rmax = running_max.to(DType::Float32).contiguous();
        const float* mn = rmin.data_ptr<float>();
        const float* mx = rmax.data_ptr<float>();
        Tensor sc = Tensor::empty({size}, DType::Float32, self.device());
        Tensor zp = Tensor::empty({size}, DType::Int64, self.device());
        float* sp = sc.data_ptr<float>();
        int64_t* zpp = zp.data_ptr<int64_t>();
        for (int64_t i = 0; i < size; ++i) {
            const QParams qp = choose_qparams_cpu(
                static_cast<double>(mn[i]), static_cast<double>(mx[i]),
                quant_min, quant_max, symmetric_quant);
            sp[i] = static_cast<float>(qp.scale);
            zpp[i] = qp.zero_point;
        }
        scale.copy_(sc);
        zero_point.copy_(zp);
        return fake_quantize_per_channel_affine_cachemask_cpu(
            self, sc, zp, ch_axis, quant_min, quant_max);
    }

    if (observe) {
        auto mm = Tensor::aminmax(self.reshape({self.numel()}), {}, false);
        moving_average_update_cpu(std::get<0>(mm), std::get<1>(mm),
                                  running_min, running_max,
                                  static_cast<float>(averaging_const));
    }
    if (!fake_on) {
        Tensor mask = Tensor::full_like(self, 1, DType::Bool);
        return {self.clone(), std::move(mask)};
    }
    const double mn = running_min.item().toDouble();
    const double mx = running_max.item().toDouble();
    const QParams qp = choose_qparams_cpu(mn, mx, quant_min, quant_max,
                                          symmetric_quant);
    Tensor sc = Tensor::full({1}, Scalar(static_cast<float>(qp.scale)),
                             DType::Float32, self.device());
    Tensor zp = Tensor::full({1}, Scalar(qp.zero_point), DType::Int64,
                             self.device());
    scale.copy_(sc);
    zero_point.copy_(zp);
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu(
        self, sc, zp, enabled, quant_min, quant_max);
}

Tensor fused_moving_avg_obs_fake_quant_cpu(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (self.numel() == 0) {
        return self.clone();
    }
    return std::get<0>(_fused_moving_avg_obs_fq_helper_cpu(
        self, observer_on, fake_quant_on, running_min, running_max, scale,
        zero_point, averaging_const, quant_min, quant_max, ch_axis,
        per_row_fake_quant, symmetric_quant));
}

// ---------------------------------------------------------------------------
// Tensor-level quantization metadata.  Quantized tensors carry an immutable
// quantizer (scheme + affine parameters); these kernels read it, strip it
// (int_repr), or attach one to raw integer code storage (_make_per_*).
// ---------------------------------------------------------------------------

bool is_quantized_cpu(const Tensor& self) {
    return quantized::is_quantized(self);
}

int64_t qscheme_cpu(const Tensor& self) {
    quantized::require_quantized(self, "qscheme");
    return static_cast<int64_t>(
        quantized::quantizer_of(self)->qscheme());
}

double q_scale_cpu(const Tensor& self) {
    return quantized::q_scale(self);
}

int64_t q_zero_point_cpu(const Tensor& self) {
    return quantized::q_zero_point(self);
}

Tensor q_per_channel_scales_cpu(const Tensor& self) {
    return quantized::q_per_channel_scales(self);
}

Tensor q_per_channel_zero_points_cpu(const Tensor& self) {
    return quantized::q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis_cpu(const Tensor& self) {
    return quantized::q_per_channel_axis(self);
}

Tensor int_repr_cpu(const Tensor& self) {
    quantized::require_quantized(self, "int_repr");
    return quantized::strip_quantizer(self).clone();
}

Tensor dequantize_self_cpu(const Tensor& self) {
    if (!quantized::is_quantized(self)) {
        return self;
    }
    return quantized::quantizer_of(self)->dequantize(self);
}

namespace {

// Quantized dtype that backs a raw integer code tensor.
DType quantized_dtype_of_codes(const Tensor& codes, const char* op) {
    switch (codes.dtype()) {
        case DType::Int8:
            return DType::QInt8;
        case DType::UInt8:
            return DType::QUInt8;
        case DType::Int32:
            return DType::QInt32;
        default:
            TP_THROW(TypeError,
                     std::string(op) +
                         ": expected an Int8/UInt8/Int32 code tensor, got " +
                         toString(codes.dtype()));
    }
}

} // namespace

Tensor _make_per_tensor_quantized_tensor_cpu(const Tensor& self, double scale,
                                             int64_t zero_point) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError,
                 "_make_per_tensor_quantized_tensor(): scale must be positive");
    }
    const DType qdt = quantized_dtype_of_codes(
        self, "_make_per_tensor_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(), make_per_tensor_affine_quantizer(scale, zero_point, qdt),
        qdt);
}

Tensor _make_per_channel_quantized_tensor_cpu(const Tensor& self,
                                              const Tensor& scale,
                                              const Tensor& zero_point,
                                              int64_t axis) {
    if (scale.dim() != 1 || zero_point.shape() != scale.shape()) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): scales/zero_points "
                 "must be 1-D with equal sizes");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): axis out of range");
    }
    if (scale.size(0) != self.size(axis)) {
        TP_THROW(ValueError,
                 "_make_per_channel_quantized_tensor(): scales size must "
                 "match the quantized dimension");
    }
    Tensor sc = scale.to(DType::Float64).contiguous();
    const double* sp = sc.data_ptr<double>();
    for (int64_t i = 0; i < sc.numel(); ++i) {
        if (!(sp[i] > 0.0)) {
            TP_THROW(ValueError,
                     "_make_per_channel_quantized_tensor(): scales must be "
                     "positive");
        }
    }
    const DType qdt = quantized_dtype_of_codes(
        self, "_make_per_channel_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(),
        make_per_channel_affine_quantizer(scale, zero_point, axis, qdt), qdt);
}

TENSORPLAY_LIBRARY_IMPL(CPU, QuantKernels) {
    m.impl("quantize_per_tensor", quantize_per_tensor_cpu);
    m.impl("dequantize_per_tensor", dequantize_per_tensor_cpu);
    m.impl("quantize_per_channel", quantize_per_channel_cpu);
    m.impl("dequantize_per_channel", dequantize_per_channel_cpu);
    m.impl("quantized_linear", quantized_linear_cpu);
    m.impl("quantized_add", quantized_add_cpu);
    m.impl("quantized_sub", quantized_sub_cpu);
    m.impl("quantized_mul", quantized_mul_cpu);
    m.impl("quantized_div", quantized_div_cpu);
    m.impl("quantized_clamp", quantized_clamp_cpu);
    m.impl("quantized_max_pool2d", quantized_max_pool2d_cpu);
    m.impl("quantized_conv2d", quantized_conv2d_cpu);
    m.impl("quantized_conv2d_prepack", quantized_conv2d_prepack_cpu);
    m.impl("quantized_conv2d_unpack", quantized_conv2d_unpack_cpu);
    m.impl("quantize_per_tensor_quint8", quantize_per_tensor_quint8_cpu);
    m.impl("dequantize_per_tensor_quint8", dequantize_per_tensor_quint8_cpu);
    m.impl("quantize_per_tensor_qint32", quantize_per_tensor_qint32_cpu);
    m.impl("dequantize_per_tensor_qint32", dequantize_per_tensor_qint32_cpu);
    m.impl("fake_quantize_per_tensor_affine",
           fake_quantize_per_tensor_affine_cpu);
    m.impl("fake_quantize_per_tensor_affine.tensor_qparams",
           fake_quantize_per_tensor_affine_tensor_qparams_cpu);
    m.impl("fake_quantize_per_tensor_affine_cachemask",
           fake_quantize_per_tensor_affine_cachemask_cpu);
    m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
           _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cpu);
    m.impl("fake_quantize_per_tensor_affine_cachemask_backward",
           fake_quantize_per_tensor_affine_cachemask_backward_cpu);
    m.impl("_fake_quantize_learnable_per_tensor_affine",
           _fake_quantize_learnable_per_tensor_affine_cpu);
    m.impl("_fake_quantize_learnable_per_tensor_affine_backward",
           _fake_quantize_learnable_per_tensor_affine_backward_cpu);
    m.impl("fake_quantize_per_channel_affine",
           fake_quantize_per_channel_affine_cpu);
    m.impl("fake_quantize_per_channel_affine_cachemask",
           fake_quantize_per_channel_affine_cachemask_cpu);
    m.impl("fake_quantize_per_channel_affine_cachemask_backward",
           fake_quantize_per_channel_affine_cachemask_backward_cpu);
    m.impl("_fake_quantize_learnable_per_channel_affine",
           _fake_quantize_learnable_per_channel_affine_cpu);
    m.impl("_fake_quantize_learnable_per_channel_affine_backward",
           _fake_quantize_learnable_per_channel_affine_backward_cpu);
    m.impl("quantize_per_tensor_dynamic", quantize_per_tensor_dynamic_cpu);
    m.impl("_choose_qparams_per_tensor", _choose_qparams_per_tensor_cpu);
    m.impl("fused_moving_avg_obs_fake_quant",
           fused_moving_avg_obs_fake_quant_cpu);
    m.impl("_fused_moving_avg_obs_fq_helper",
           _fused_moving_avg_obs_fq_helper_cpu);
    m.impl("is_quantized", is_quantized_cpu);
    m.impl("qscheme", qscheme_cpu);
    m.impl("q_scale", q_scale_cpu);
    m.impl("q_zero_point", q_zero_point_cpu);
    m.impl("q_per_channel_scales", q_per_channel_scales_cpu);
    m.impl("q_per_channel_zero_points", q_per_channel_zero_points_cpu);
    m.impl("q_per_channel_axis", q_per_channel_axis_cpu);
    m.impl("int_repr", int_repr_cpu);
    m.impl("dequantize.self", dequantize_self_cpu);
    m.impl("_make_per_tensor_quantized_tensor",
           _make_per_tensor_quantized_tensor_cpu);
    m.impl("_make_per_channel_quantized_tensor",
           _make_per_channel_quantized_tensor_cpu);
}

} // namespace cpu
} // namespace tensorplay
