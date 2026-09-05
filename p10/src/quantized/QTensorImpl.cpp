#include "Quantizer.h"
#include "Exception.h"
#include "QuantKernels.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace tensorplay {
namespace {

using tensorplay::DType;
using tensorplay::Device;
using tensorplay::DeviceType;
using tensorplay::Tensor;

std::pair<int64_t, int64_t> quantized_range(DType dtype) {
    switch (dtype) {
        case DType::QInt8:
            return {-128, 127};
        case DType::QUInt8:
            return {0, 255};
        case DType::QInt32:
            return {std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max()};
        default:
            TP_THROW(TypeError,
                     "quantized qparams require a quantized dtype");
    }
}

void validate_per_tensor_qparams(double scale, int64_t zero_point,
                                DType scalar_type, const char* op) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        TP_THROW(ValueError, op, ": scale must be positive");
    }
    const auto range = quantized_range(scalar_type);
    if (zero_point < range.first || zero_point > range.second) {
        TP_THROW(ValueError, op,
                 ": zero_point is outside the quantized range");
    }
}

void validate_per_channel_qparams(const Tensor& scales,
                                  const Tensor& zero_points,
                                  DType scalar_type, const char* op) {
    if (scales.dim() != 1 || zero_points.dim() != 1 ||
        scales.numel() != zero_points.numel()) {
        TP_THROW(
            ValueError, op,
            ": scales and zero_points must be 1-D tensors with equal sizes");
    }
    if (!tensorplay::isFloatingType(scales.dtype())) {
        TP_THROW(TypeError, op, ": scales must be floating point");
    }

    const Tensor scales_cpu = scales.to(DType::Float64)
                                  .to(Device(DeviceType::CPU))
                                  .contiguous();
    const double* scale_data = scales_cpu.data_ptr<double>();
    for (int64_t i = 0; i < scales_cpu.numel(); ++i) {
        if (!(scale_data[i] > 0.0) || !std::isfinite(scale_data[i])) {
            TP_THROW(ValueError, op, ": scales must be positive");
        }
    }

    const auto range = quantized_range(scalar_type);
    if (tensorplay::isFloatingType(zero_points.dtype())) {
        const Tensor zero_points_cpu = zero_points.to(DType::Float32)
                                           .to(Device(DeviceType::CPU))
                                           .contiguous();
        const float* zero_point_data = zero_points_cpu.data_ptr<float>();
        for (int64_t i = 0; i < zero_points_cpu.numel(); ++i) {
            if (!std::isfinite(zero_point_data[i]) ||
                zero_point_data[i] < static_cast<float>(range.first) ||
                zero_point_data[i] > static_cast<float>(range.second)) {
                TP_THROW(ValueError, op,
                         ": zero_points are outside the quantized range");
            }
        }
    } else {
        const Tensor zero_points_cpu = zero_points.to(DType::Int64)
                                           .to(Device(DeviceType::CPU))
                                           .contiguous();
        const int64_t* zero_point_data = zero_points_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < zero_points_cpu.numel(); ++i) {
            if (zero_point_data[i] < range.first ||
                zero_point_data[i] > range.second) {
                TP_THROW(ValueError, op,
                         ": zero_points are outside the quantized range");
            }
        }
    }
}

bool same_tensor(const Tensor& lhs, const Tensor& rhs) {
    if (!lhs.defined() || !rhs.defined()) {
        return lhs.defined() == rhs.defined();
    }
    if (lhs.dtype() != rhs.dtype() || lhs.shape() != rhs.shape() ||
        lhs.device() != rhs.device()) {
        return false;
    }
    if (lhs.numel() == 0) return true;
    const Tensor a = lhs.device().is_cpu()
                         ? (lhs.is_contiguous() ? lhs : lhs.contiguous())
                         : lhs.to(Device(DeviceType::CPU)).contiguous();
    const Tensor b = rhs.device().is_cpu()
                         ? (rhs.is_contiguous() ? rhs : rhs.contiguous())
                         : rhs.to(Device(DeviceType::CPU)).contiguous();
    return std::memcmp(a.data_ptr(), b.data_ptr(),
                       static_cast<size_t>(a.numel()) * a.itemsize()) == 0;
}

void check_float_input(const Tensor& tensor) {
    if (tensor.dtype() != DType::Float32) {
        TP_THROW(TypeError, "quantize(): expected a Float32 tensor, got ",
                 toString(tensor.dtype()));
    }
}

void prepare_dequantize_out(Tensor& out, const Tensor& tensor) {
    out.resize_(static_cast<std::vector<int64_t>>(tensor.shape()));
    if (out.device() != tensor.device()) {
        TP_THROW(DeviceMismatchError,
                 "dequantize_out(): output and input must be on the same device");
    }
    if (out.dtype() != DType::Float32 || !out.is_contiguous()) {
        TP_THROW(TypeError,
                 "dequantize_out(): output must be a contiguous Float32 tensor");
    }
}

template <typename Dequantize>
Tensor& dequantize_out_from(Tensor& out, const Tensor& tensor,
                            Dequantize&& dequantize) {
    prepare_dequantize_out(out, tensor);
    out.copy_(dequantize());
    return out;
}

Tensor quantize_per_tensor_with_dtype(const Tensor& tensor, double scale,
                                      int64_t zero_point, DType dtype) {
    if (tensor.device().is_cpu()) {
        return cpu::quantize_per_tensor_dtype_cpu(
            tensor, scale, zero_point, dtype);
    }
#ifdef USE_CUDA
    if (tensor.device().is_cuda()) {
        return cuda::quantize_per_tensor_dtype_cuda(
            tensor, scale, zero_point, dtype);
    }
#endif
#ifdef USE_VULKAN
    if (tensor.device().is_vulkan()) {
        return vulkan::ops::quantize_per_tensor_kernel(
            tensor, scale, zero_point, dtype);
    }
#endif
    TP_THROW(NotImplementedError,
             "per-tensor quantization is not implemented for this device");
}

Tensor dequantize_per_tensor_with_dtype(const Tensor& tensor, double scale,
                                        int64_t zero_point, DType dtype) {
    if (tensor.device().is_cpu()) {
        return cpu::dequantize_per_tensor_dtype_cpu(
            tensor, scale, zero_point, dtype);
    }
#ifdef USE_CUDA
    if (tensor.device().is_cuda()) {
        return cuda::dequantize_per_tensor_dtype_cuda(
            tensor, scale, zero_point, dtype);
    }
#endif
#ifdef USE_VULKAN
    if (tensor.device().is_vulkan()) {
        return vulkan::ops::dequantize_per_tensor_kernel(
            tensor, scale, zero_point, dtype);
    }
#endif
    TP_THROW(NotImplementedError,
             "per-tensor dequantization is not implemented for this device");
}

Tensor quantize_per_channel_with_dtype(const Tensor& tensor,
                                       const Tensor& scales,
                                       const Tensor& zero_points, int64_t axis,
                                       DType dtype) {
    if (tensor.device().is_cpu()) {
        return cpu::quantize_per_channel_dtype_cpu(
            tensor, scales, zero_points, axis, dtype);
    }
#ifdef USE_CUDA
    if (tensor.device().is_cuda()) {
        return cuda::quantize_per_channel_dtype_cuda(
            tensor, scales, zero_points, axis, dtype);
    }
#endif
    TP_THROW(NotImplementedError,
             "per-channel quantization is not implemented for this device");
}

Tensor dequantize_per_channel_with_dtype(const Tensor& tensor,
                                         const Tensor& scales,
                                         const Tensor& zero_points,
                                         int64_t axis, DType dtype) {
    if (tensor.device().is_cpu()) {
        return cpu::dequantize_per_channel_dtype_cpu(
            tensor, scales, zero_points, axis, dtype);
    }
#ifdef USE_CUDA
    if (tensor.device().is_cuda()) {
        return cuda::dequantize_per_channel_dtype_cuda(
            tensor, scales, zero_points, axis, dtype);
    }
#endif
    TP_THROW(NotImplementedError,
             "per-channel dequantization is not implemented for this device");
}

} // namespace

Tensor UnknownQuantizer::quantize(const Tensor&) {
    TP_THROW(RuntimeError, "cannot quantize with UnknownQuantizer");
}

Tensor UnknownQuantizer::dequantize(const Tensor&) {
    TP_THROW(RuntimeError, "cannot dequantize with UnknownQuantizer");
}

Tensor& UnknownQuantizer::dequantize_out(Tensor&, const Tensor&) {
    TP_THROW(RuntimeError, "cannot dequantize_out with UnknownQuantizer");
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor& tensor) {
    check_float_input(tensor);
    return quantize_per_tensor_with_dtype(
        tensor, scale_, zero_point_, scalar_type());
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor& tensor) {
    return dequantize_per_tensor_with_dtype(
        tensor, scale_, zero_point_, scalar_type());
}

Tensor& PerTensorAffineQuantizer::dequantize_out(Tensor& out,
                                                  const Tensor& tensor) {
    return dequantize_out_from(out, tensor, [&] { return dequantize(tensor); });
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor& tensor) {
    check_float_input(tensor);
    return quantize_per_channel_with_dtype(
        tensor, scales_, zero_points_, axis_, scalar_type());
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor& tensor) {
    return dequantize_per_channel_with_dtype(
        tensor, scales_, zero_points_, axis_, scalar_type());
}

Tensor& PerChannelAffineQuantizer::dequantize_out(Tensor& out,
                                                   const Tensor& tensor) {
    return dequantize_out_from(out, tensor, [&] { return dequantize(tensor); });
}

Tensor PerChannelAffineFloatQParamsQuantizer::quantize(const Tensor& tensor) {
    check_float_input(tensor);
    return quantize_per_channel_with_dtype(
        tensor, scales_, zero_points_, axis_, scalar_type());
}

Tensor PerChannelAffineFloatQParamsQuantizer::dequantize(
    const Tensor& tensor) {
    return dequantize_per_channel_with_dtype(
        tensor, scales_, zero_points_, axis_, scalar_type());
}

Tensor& PerChannelAffineFloatQParamsQuantizer::dequantize_out(
    Tensor& out, const Tensor& tensor) {
    return dequantize_out_from(out, tensor, [&] { return dequantize(tensor); });
}

bool PerTensorAffineQuantizer::equalTo(const QuantizerPtr& other) const {
    if (!other || other->qscheme() != kPerTensorAffine) return false;
    const auto* rhs = dynamic_cast<const PerTensorAffineQuantizer*>(other.get());
    return rhs != nullptr && scalar_type() == rhs->scalar_type() &&
           scale() == rhs->scale() && zero_point() == rhs->zero_point();
}

bool PerChannelAffineQuantizer::equalTo(const QuantizerPtr& other) const {
    if (!other || other->qscheme() != kPerChannelAffine) return false;
    const auto* rhs = dynamic_cast<const PerChannelAffineQuantizer*>(other.get());
    return rhs != nullptr && scalar_type() == rhs->scalar_type() &&
           axis() == rhs->axis() && same_tensor(scales(), rhs->scales()) &&
           same_tensor(zero_points(), rhs->zero_points());
}

bool PerChannelAffineFloatQParamsQuantizer::equalTo(
    const QuantizerPtr& other) const {
    if (!other || other->qscheme() != kPerChannelAffineFloatQParams) {
        return false;
    }
    const auto* rhs =
        dynamic_cast<const PerChannelAffineFloatQParamsQuantizer*>(other.get());
    return rhs != nullptr && scalar_type() == rhs->scalar_type() &&
           axis() == rhs->axis() && same_tensor(scales(), rhs->scales()) &&
           same_tensor(zero_points(), rhs->zero_points());
}

QuantizerPtr make_per_tensor_affine_quantizer(double scale,
                                               int64_t zero_point,
                                               DType scalar_type) {
    validate_per_tensor_qparams(scale, zero_point, scalar_type,
                                "make_per_tensor_affine_quantizer()");
    return std::make_shared<PerTensorAffineQuantizer>(scalar_type, scale,
                                                       zero_point);
}

QuantizerPtr make_per_channel_affine_quantizer(const Tensor& scales,
                                                const Tensor& zero_points,
                                                int64_t axis,
                                                DType scalar_type) {
    validate_per_channel_qparams(
        scales, zero_points, scalar_type, "make_per_channel_affine_quantizer()");
    if (isFloatingType(zero_points.dtype())) {
        return std::make_shared<PerChannelAffineFloatQParamsQuantizer>(
            scalar_type, scales.to(DType::Float32).contiguous(),
            zero_points.to(DType::Float32).contiguous(), axis);
    }
    return std::make_shared<PerChannelAffineQuantizer>(
        scalar_type, scales.to(DType::Float64).contiguous(),
        zero_points.to(DType::Int64).contiguous(), axis);
}

QuantizerPtr make_unknown_quantizer(DType scalar_type) {
    TP_CHECK(isQuantizedType(scalar_type),
             "make_unknown_quantizer(): scalar_type must be quantized");
    return std::make_shared<UnknownQuantizer>(scalar_type);
}

ScalarType underlying_storage_type(DType dtype) {
    switch (dtype) {
        case DType::QInt8:
            return DType::Int8;
        case DType::QUInt8:
            return DType::UInt8;
        case DType::QInt32:
            return DType::Int32;
        default:
            TP_CHECK(isQuantizedType(dtype),
                     "underlying_storage_type(): not a quantized dtype");
            return DType::Undefined;
    }
}

DType quantized_dtype_for_scheme(QScheme scheme) {
    switch (scheme) {
        case kPerTensorAffine:
        case kPerChannelAffine:
        case kPerTensorSymmetric:
        case kPerChannelSymmetric:
        case kPerChannelAffineFloatQParams:
            return DType::QInt8;
        case QScheme::COMPILE_TIME_NUM_QSCHEMES:
            break;
    }
    TP_THROW(ValueError, "quantized_dtype_for_scheme(): unknown scheme");
}

namespace quantized {

bool is_quantized(const Tensor& t) {
    return t.defined() && t.impl() && t.impl()->has_quantizer();
}

QuantizerPtr quantizer_of(const Tensor& t) {
    if (!t.defined() || !t.impl()) return nullptr;
    return t.impl()->quantizer();
}

void require_quantized(const Tensor& t, const char* op) {
    if (!is_quantized(t)) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a quantized tensor, got " +
                     toString(t.dtype()));
    }
}

Tensor make_qtensor(const Tensor& codes, QuantizerPtr quantizer, DType dtype) {
    TP_CHECK(quantizer != nullptr,
             "make_qtensor(): a quantizer is required");
    TP_CHECK(isQuantizedType(dtype),
             "make_qtensor(): dtype must be a quantized dtype");
    TP_CHECK(quantizer->scalar_type() == dtype,
             "make_qtensor(): quantizer scalar type must match dtype");
    TP_CHECK(codes.dtype() == underlying_storage_type(dtype),
             "make_qtensor(): code storage dtype must match the quantized "
             "dtype's underlying integer type");
    const auto* per_channel_quantizer =
        dynamic_cast<const PerChannelAffineQuantizer*>(quantizer.get());
    if (per_channel_quantizer != nullptr &&
        (per_channel_quantizer->scales().device() != codes.device() ||
         per_channel_quantizer->zero_points().device() != codes.device())) {
        TP_CHECK(false,
                 "make_qtensor(): per-channel qparams must be on the code tensor device");
    }
    Tensor out(codes.impl()->storage(),
               static_cast<std::vector<int64_t>>(codes.shape()),
               codes.strides(), dtype,
               static_cast<size_t>(codes.impl()->storage_offset()));
    out.impl()->set_quantizer(std::move(quantizer));
    return out;
}

Tensor strip_quantizer(const Tensor& t) {
    require_quantized(t, "strip_quantizer");
    return Tensor(t.impl()->storage(),
                  static_cast<std::vector<int64_t>>(t.shape()),
                  static_cast<std::vector<int64_t>>(t.strides()),
                  underlying_storage_type(t.dtype()), t.storage_offset());
}

double q_scale(const Tensor& t) {
    require_quantized(t, "q_scale");
    const auto q = quantizer_of(t);
    if (!isPerTensorQScheme(q->qscheme())) {
        TP_THROW(RuntimeError,
                 "q_scale(): per-channel quantized tensors do not carry a "
                 "single scale; use q_per_channel_scales()");
    }
    return q->scale();
}

int64_t q_zero_point(const Tensor& t) {
    require_quantized(t, "q_zero_point");
    const auto q = quantizer_of(t);
    if (!isPerTensorQScheme(q->qscheme())) {
        TP_THROW(RuntimeError,
                 "q_zero_point(): per-channel quantized tensors do not carry "
                 "a single zero point; use q_per_channel_zero_points()");
    }
    return q->zero_point();
}

Tensor q_per_channel_scales(const Tensor& t) {
    require_quantized(t, "q_per_channel_scales");
    const auto q = quantizer_of(t);
    if (!isPerChannelQScheme(q->qscheme())) {
        TP_THROW(RuntimeError,
                 "q_per_channel_scales(): per-tensor quantized tensors do "
                 "not carry per-channel scales; use q_scale()");
    }
    return q->scales();
}

Tensor q_per_channel_zero_points(const Tensor& t) {
    require_quantized(t, "q_per_channel_zero_points");
    const auto q = quantizer_of(t);
    if (!isPerChannelQScheme(q->qscheme())) {
        TP_THROW(RuntimeError,
                 "q_per_channel_zero_points(): per-tensor quantized tensors "
                 "do not carry per-channel zero points; use q_zero_point()");
    }
    return q->zero_points();
}

int64_t q_per_channel_axis(const Tensor& t) {
    require_quantized(t, "q_per_channel_axis");
    const auto q = quantizer_of(t);
    if (!isPerChannelQScheme(q->qscheme())) {
        TP_THROW(RuntimeError,
                 "q_per_channel_axis(): per-tensor quantized tensors do not "
                 "have a quantized axis");
    }
    return q->axis();
}

} // namespace quantized
} // namespace tensorplay
