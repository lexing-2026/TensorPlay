#include "Quantizer.h"
#include "Exception.h"

namespace tensorplay {

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
        case QScheme::PerTensorAffine:
            return DType::QInt8;
        case QScheme::PerChannelAffine:
            return DType::QInt8;
    }
    TP_THROW(ValueError, "quantized_dtype_for_scheme(): unknown scheme");
}

namespace quantized {

bool is_quantized(const Tensor& t) {
    return t.defined() && t.impl() && t.impl()->has_quantizer();
}

std::shared_ptr<Quantizer> quantizer_of(const Tensor& t) {
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

Tensor make_qtensor(const Tensor& codes, std::shared_ptr<Quantizer> quantizer,
                    DType dtype) {
    TP_CHECK(quantizer != nullptr,
             "make_qtensor(): a quantizer is required");
    TP_CHECK(isQuantizedType(dtype),
             "make_qtensor(): dtype must be a quantized dtype");
    TP_CHECK(codes.dtype() == underlying_storage_type(dtype),
             "make_qtensor(): code storage dtype must match the quantized "
             "dtype's underlying integer type");
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
    if (q->qscheme() != QScheme::PerTensorAffine) {
        TP_THROW(RuntimeError,
                 "q_scale(): per-channel quantized tensors do not carry a "
                 "single scale; use q_per_channel_scales()");
    }
    return q->scale();
}

int64_t q_zero_point(const Tensor& t) {
    require_quantized(t, "q_zero_point");
    const auto q = quantizer_of(t);
    if (q->qscheme() != QScheme::PerTensorAffine) {
        TP_THROW(RuntimeError,
                 "q_zero_point(): per-channel quantized tensors do not carry "
                 "a single zero point; use q_per_channel_zero_points()");
    }
    return q->zero_point();
}

Tensor q_per_channel_scales(const Tensor& t) {
    require_quantized(t, "q_per_channel_scales");
    const auto q = quantizer_of(t);
    if (q->qscheme() != QScheme::PerChannelAffine) {
        TP_THROW(RuntimeError,
                 "q_per_channel_scales(): per-tensor quantized tensors do "
                 "not carry per-channel scales; use q_scale()");
    }
    return q->scales();
}

Tensor q_per_channel_zero_points(const Tensor& t) {
    require_quantized(t, "q_per_channel_zero_points");
    const auto q = quantizer_of(t);
    if (q->qscheme() != QScheme::PerChannelAffine) {
        TP_THROW(RuntimeError,
                 "q_per_channel_zero_points(): per-tensor quantized tensors "
                 "do not carry per-channel zero points; use q_zero_point()");
    }
    return q->zero_points();
}

int64_t q_per_channel_axis(const Tensor& t) {
    require_quantized(t, "q_per_channel_axis");
    const auto q = quantizer_of(t);
    if (q->qscheme() != QScheme::PerChannelAffine) {
        TP_THROW(RuntimeError,
                 "q_per_channel_axis(): per-tensor quantized tensors do not "
                 "have a quantized axis");
    }
    return q->axis();
}

} // namespace quantized
} // namespace tensorplay
