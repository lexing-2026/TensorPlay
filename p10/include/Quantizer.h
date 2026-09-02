#pragma once

#include "Macros.h"
#include "Tensor.h"

#include <memory>
#include <vector>

namespace tensorplay {

// Quantization schemes carried by a quantized tensor's quantizer.
// Per-tensor affine stores one (scale, zero_point) pair; per-channel affine
// stores a 1-D scale/zero-point vector indexed along `axis`.
enum class QScheme : int8_t {
    PerTensorAffine = 0,
    PerChannelAffine = 1,
};

inline const char* toString(QScheme scheme) {
    switch (scheme) {
        case QScheme::PerTensorAffine:
            return "per_tensor_affine";
        case QScheme::PerChannelAffine:
            return "per_channel_affine";
    }
    return "unknown_scheme";
}

// Quantizer: the affine parameter set attached to a quantized tensor.  The
// real-domain mapping is real = scale * (code - zero_point); per-channel
// variants index the mapping along one tensor axis.  Quantizers are
// immutable once attached and shared (not copied) between tensor views.
class P10_API Quantizer {
public:
    explicit Quantizer(QScheme scheme) : scheme_(scheme) {}
    virtual ~Quantizer() = default;

    QScheme qscheme() const { return scheme_; }

    // Per-tensor parameters; per-channel quantizers aggregate to a single
    // pair only when their scales/zero_points are uniform, so the defaults
    // below are only meaningful for the per-tensor scheme.
    virtual double scale() const { return 1.0; }
    virtual int64_t zero_point() const { return 0; }

    // Per-channel parameters; empty tensors for the per-tensor scheme.
    virtual Tensor scales() const { return Tensor(); }
    virtual Tensor zero_points() const { return Tensor(); }
    virtual int64_t axis() const { return -1; }

private:
    QScheme scheme_;
};

class P10_API PerTensorAffineQuantizer final : public Quantizer {
public:
    PerTensorAffineQuantizer(double scale, int64_t zero_point)
        : Quantizer(QScheme::PerTensorAffine),
          scale_(scale),
          zero_point_(zero_point) {}

    double scale() const override { return scale_; }
    int64_t zero_point() const override { return zero_point_; }

private:
    double scale_;
    int64_t zero_point_;
};

class P10_API PerChannelAffineQuantizer final : public Quantizer {
public:
    PerChannelAffineQuantizer(Tensor scales, Tensor zero_points, int64_t axis)
        : Quantizer(QScheme::PerChannelAffine),
          scales_(std::move(scales)),
          zero_points_(std::move(zero_points)),
          axis_(axis) {}

    Tensor scales() const override { return scales_; }
    Tensor zero_points() const override { return zero_points_; }
    int64_t axis() const override { return axis_; }

private:
    Tensor scales_;
    Tensor zero_points_;
    int64_t axis_;
};

// Storage code type a quantized dtype maps onto.
ScalarType underlying_storage_type(DType dtype);

// DType used for a quantized tensor under `scheme`: per-channel tensors are
// Int8 storage, matching how the quantize kernels emit codes.
DType quantized_dtype_for_scheme(QScheme scheme);

namespace quantized {

// True when the tensor carries a quantizer.
P10_API bool is_quantized(const Tensor& t);

// Returns the attached quantizer or nullptr.
P10_API std::shared_ptr<Quantizer> quantizer_of(const Tensor& t);

// Requires a quantized tensor carrying a quantizer; throws otherwise.
P10_API void require_quantized(const Tensor& t, const char* op);

// Wraps `codes` (an Int8/UInt8/Int32 tensor) as a quantized tensor of
// dtype `dtype` carrying `quantizer`.  The storage is shared, not copied:
// codes are reinterpreted as the quantized dtype's storage.
P10_API Tensor make_qtensor(const Tensor& codes,
                            std::shared_ptr<Quantizer> quantizer,
                            DType dtype);

// Detaches the quantizer, returning a plain integer tensor over the same
// storage and shape (the "integer representation" view).
P10_API Tensor strip_quantizer(const Tensor& t);

// Per-tensor qparams; throws for tensors without a per-tensor quantizer.
P10_API double q_scale(const Tensor& t);
P10_API int64_t q_zero_point(const Tensor& t);

// Per-channel qparams; throws for tensors without a per-channel quantizer.
P10_API Tensor q_per_channel_scales(const Tensor& t);
P10_API Tensor q_per_channel_zero_points(const Tensor& t);
P10_API int64_t q_per_channel_axis(const Tensor& t);

} // namespace quantized
} // namespace tensorplay
