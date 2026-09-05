#pragma once

#include "Exception.h"
#include "Macros.h"
#include "Tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace tensorplay {

enum class QScheme : uint8_t {
    PER_TENSOR_AFFINE = 0,
    PER_CHANNEL_AFFINE = 1,
    PER_TENSOR_SYMMETRIC = 2,
    PER_CHANNEL_SYMMETRIC = 3,
    PER_CHANNEL_AFFINE_FLOAT_QPARAMS = 4,
    COMPILE_TIME_NUM_QSCHEMES = 5,
};

constexpr QScheme kPerTensorAffine = QScheme::PER_TENSOR_AFFINE;
constexpr QScheme kPerChannelAffine = QScheme::PER_CHANNEL_AFFINE;
constexpr QScheme kPerTensorSymmetric = QScheme::PER_TENSOR_SYMMETRIC;
constexpr QScheme kPerChannelSymmetric = QScheme::PER_CHANNEL_SYMMETRIC;
constexpr QScheme kPerChannelAffineFloatQParams =
    QScheme::PER_CHANNEL_AFFINE_FLOAT_QPARAMS;
constexpr int COMPILE_TIME_NUM_QSCHEMES =
    static_cast<int>(QScheme::COMPILE_TIME_NUM_QSCHEMES);

inline std::string toString(QScheme scheme) {
    switch (scheme) {
        case kPerTensorAffine:
            return "per_tensor_affine";
        case kPerChannelAffine:
            return "per_channel_affine";
        case kPerTensorSymmetric:
            return "per_tensor_symmetric";
        case kPerChannelSymmetric:
            return "per_channel_symmetric";
        case kPerChannelAffineFloatQParams:
            return "per_channel_affine_float_qparams";
        default:
            TP_THROW(ValueError, "unrecognized quantization scheme: ",
                     static_cast<int>(scheme));
    }
}

inline bool isPerTensorQScheme(QScheme scheme) {
    return scheme == kPerTensorAffine || scheme == kPerTensorSymmetric;
}

inline bool isPerChannelQScheme(QScheme scheme) {
    return scheme == kPerChannelAffine ||
           scheme == kPerChannelSymmetric ||
           scheme == kPerChannelAffineFloatQParams;
}

class Quantizer;
using QuantizerPtr = std::shared_ptr<Quantizer>;
using ConstQuantizerPtr = std::shared_ptr<const Quantizer>;

class P10_API Quantizer {
public:
    explicit Quantizer(DType scalar_type) : scalar_type_(scalar_type) {}
    virtual ~Quantizer() = default;

    virtual QScheme qscheme() const = 0;
    DType scalar_type() const { return scalar_type_; }
    virtual Tensor quantize(const Tensor& tensor) = 0;
    virtual Tensor dequantize(const Tensor& tensor) = 0;
    virtual Tensor& dequantize_out(Tensor& out, const Tensor& tensor) = 0;
    virtual bool equalTo(const QuantizerPtr& other) const = 0;

    virtual double scale() const { return 1.0; }
    virtual int64_t zero_point() const { return 0; }
    virtual Tensor scales() const { return Tensor(); }
    virtual Tensor zero_points() const { return Tensor(); }
    virtual int64_t axis() const { return -1; }

private:
    const DType scalar_type_;
};

class P10_API UnknownQuantizer final : public Quantizer {
public:
    explicit UnknownQuantizer(DType scalar_type) : Quantizer(scalar_type) {}

    Tensor quantize(const Tensor& tensor) override;
    Tensor dequantize(const Tensor& tensor) override;
    Tensor& dequantize_out(Tensor& out, const Tensor& tensor) override;
    QScheme qscheme() const override {
        TP_THROW(RuntimeError, "an unknown quantizer has no quantization scheme");
    }

    bool equalTo(const QuantizerPtr&) const override {
        TP_THROW(RuntimeError, "an unknown quantizer cannot be compared");
    }
};

class P10_API UniformQuantizer : public Quantizer {
public:
    explicit UniformQuantizer(DType scalar_type) : Quantizer(scalar_type) {}
};

class P10_API NonUniformQuantizer : public Quantizer {
public:
    explicit NonUniformQuantizer(DType scalar_type) : Quantizer(scalar_type) {}
};

class P10_API AffineQuantizer : public UniformQuantizer {
public:
    explicit AffineQuantizer(DType scalar_type)
        : UniformQuantizer(scalar_type) {}
};

class P10_API PerTensorAffineQuantizer final : public AffineQuantizer {
public:
    PerTensorAffineQuantizer(DType scalar_type, double scale,
                             int64_t zero_point)
        : AffineQuantizer(scalar_type),
          scale_(scale),
          zero_point_(zero_point) {}

    Tensor quantize(const Tensor& tensor) override;
    Tensor dequantize(const Tensor& tensor) override;
    Tensor& dequantize_out(Tensor& out, const Tensor& tensor) override;
    QScheme qscheme() const override { return kPerTensorAffine; }
    double scale() const override { return scale_; }
    int64_t zero_point() const override { return zero_point_; }
    bool equalTo(const QuantizerPtr& other) const override;

private:
    const double scale_;
    const int64_t zero_point_;
};

class P10_API PerChannelAffineQuantizer : public AffineQuantizer {
public:
    PerChannelAffineQuantizer(DType scalar_type, Tensor scales,
                              Tensor zero_points, int64_t axis)
        : AffineQuantizer(scalar_type),
          scales_(std::move(scales)),
          zero_points_(std::move(zero_points)),
          axis_(axis) {}

    Tensor quantize(const Tensor& tensor) override;
    Tensor dequantize(const Tensor& tensor) override;
    Tensor& dequantize_out(Tensor& out, const Tensor& tensor) override;
    QScheme qscheme() const override { return kPerChannelAffine; }
    Tensor scales() const override { return scales_; }
    Tensor zero_points() const override { return zero_points_; }
    int64_t axis() const override { return axis_; }
    bool equalTo(const QuantizerPtr& other) const override;

protected:
    Tensor scales_;
    Tensor zero_points_;
    const int64_t axis_;
};

class P10_API PerChannelAffineFloatQParamsQuantizer final
    : public PerChannelAffineQuantizer {
public:
    PerChannelAffineFloatQParamsQuantizer(DType scalar_type, Tensor scales,
                                           Tensor zero_points, int64_t axis)
        : PerChannelAffineQuantizer(scalar_type, std::move(scales),
                                    std::move(zero_points), axis) {}

    Tensor quantize(const Tensor& tensor) override;
    Tensor dequantize(const Tensor& tensor) override;
    Tensor& dequantize_out(Tensor& out, const Tensor& tensor) override;
    QScheme qscheme() const override { return kPerChannelAffineFloatQParams; }
    bool equalTo(const QuantizerPtr& other) const override;
};

P10_API QuantizerPtr make_per_tensor_affine_quantizer(
    double scale, int64_t zero_point, DType scalar_type);
P10_API QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales, const Tensor& zero_points, int64_t axis,
    DType scalar_type);
P10_API QuantizerPtr make_unknown_quantizer(DType scalar_type);

P10_API ScalarType underlying_storage_type(DType dtype);
P10_API DType quantized_dtype_for_scheme(QScheme scheme);

namespace quantized {

P10_API bool is_quantized(const Tensor& t);
P10_API QuantizerPtr quantizer_of(const Tensor& t);
P10_API void require_quantized(const Tensor& t, const char* op);

P10_API Tensor make_qtensor(const Tensor& codes, QuantizerPtr quantizer,
                            DType dtype);
P10_API Tensor strip_quantizer(const Tensor& t);

P10_API double q_scale(const Tensor& t);
P10_API int64_t q_zero_point(const Tensor& t);
P10_API Tensor q_per_channel_scales(const Tensor& t);
P10_API Tensor q_per_channel_zero_points(const Tensor& t);
P10_API int64_t q_per_channel_axis(const Tensor& t);

} // namespace quantized
} // namespace tensorplay
