#include "QuantKernels.h"
#include "Exception.h"
#include "Quantizer.h"
#include "Utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tensorplay {
namespace cuda {

// ---------------------------------------------------------------------------
// Tensor-level quantization metadata on CUDA.  The quantizer lives in host
// memory on the TensorImpl; only int_repr and _make_per_* touch codes.
// ---------------------------------------------------------------------------

bool is_quantized_cuda(const Tensor& self) {
    return quantized::is_quantized(self);
}

int64_t qscheme_cuda(const Tensor& self) {
    quantized::require_quantized(self, "qscheme");
    return static_cast<int64_t>(
        quantized::quantizer_of(self)->qscheme());
}

double q_scale_cuda(const Tensor& self) {
    return quantized::q_scale(self);
}

int64_t q_zero_point_cuda(const Tensor& self) {
    return quantized::q_zero_point(self);
}

Tensor q_per_channel_scales_cuda(const Tensor& self) {
    return quantized::q_per_channel_scales(self);
}

Tensor q_per_channel_zero_points_cuda(const Tensor& self) {
    return quantized::q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis_cuda(const Tensor& self) {
    return quantized::q_per_channel_axis(self);
}

Tensor int_repr_cuda(const Tensor& self) {
    quantized::require_quantized(self, "int_repr");
    return quantized::strip_quantizer(self).clone();
}

Tensor dequantize_self_cuda(const Tensor& self) {
    if (!quantized::is_quantized(self)) {
        return self.to(DType::Float32);
    }
    return quantized::quantizer_of(self)->dequantize(self);
}

namespace {

DType quantized_dtype_of_codes_cuda(const Tensor& codes, const char* op) {
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

Tensor _make_per_tensor_quantized_tensor_cuda(const Tensor& self, double scale,
                                              int64_t zero_point) {
    if (!(scale > 0.0)) {
        TP_THROW(ValueError,
                 "_make_per_tensor_quantized_tensor(): scale must be positive");
    }
    const DType qdt = quantized_dtype_of_codes_cuda(
        self, "_make_per_tensor_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(), make_per_tensor_affine_quantizer(scale, zero_point, qdt),
        qdt);
}

Tensor _make_per_channel_quantized_tensor_cuda(const Tensor& self,
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
    const DType qdt = quantized_dtype_of_codes_cuda(
        self, "_make_per_channel_quantized_tensor");
    return quantized::make_qtensor(
        self.clone(),
        make_per_channel_affine_quantizer(scale, zero_point, axis, qdt), qdt);
}


TENSORPLAY_LIBRARY_IMPL(CUDA, QuantizedTensorKernels) {
    m.impl("is_quantized", is_quantized_cuda);
    m.impl("qscheme", qscheme_cuda);
    m.impl("q_scale", q_scale_cuda);
    m.impl("q_zero_point", q_zero_point_cuda);
    m.impl("q_per_channel_scales", q_per_channel_scales_cuda);
    m.impl("q_per_channel_zero_points", q_per_channel_zero_points_cuda);
    m.impl("q_per_channel_axis", q_per_channel_axis_cuda);
    m.impl("int_repr", int_repr_cuda);
    m.impl("dequantize.self", dequantize_self_cuda);
    m.impl("_make_per_tensor_quantized_tensor",
           _make_per_tensor_quantized_tensor_cuda);
    m.impl("_make_per_channel_quantized_tensor",
           _make_per_channel_quantized_tensor_cuda);
}

}
}
