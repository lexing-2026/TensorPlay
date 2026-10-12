#pragma once

#include "Quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay::quant {

struct NormalizationGeometry {
    int64_t groups;
    int64_t elements;
    int64_t channels;
    int64_t channel_groups;
    bool affine_per_channel;
    bool channels_last;
};

inline void check_normalization_input(const Tensor& input) {
    if (input.dtype() != DType::QInt8 && input.dtype() != DType::QUInt8 &&
        input.dtype() != DType::QInt32) {
        TP_THROW(TypeError, "quantized normalization requires a quantized tensor");
    }
    // A single scale is required to recover group moments from integer codes.
    quantized::q_scale(input);
}

inline void check_normalization_affine(const Tensor& input,
                                       const std::optional<Tensor>& parameter,
                                       int64_t elements) {
    if (!parameter.has_value()) return;
    if (parameter->numel() != elements) {
        TP_THROW(ValueError, "quantized normalization: incorrect affine parameter size");
    }
    if (parameter->dtype() != DType::Float32) {
        TP_THROW(TypeError, "quantized normalization: affine parameters must be Float32");
    }
    if (parameter->device() != input.device()) {
        TP_THROW(ValueError, "quantized normalization: affine parameters must share the input device");
    }
}

inline NormalizationGeometry layer_norm_geometry(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias) {
    check_normalization_input(input);
    const int64_t rank = static_cast<int64_t>(normalized_shape.size());
    if (rank == 0 || input.dim() < rank) {
        TP_THROW(ValueError, "quantized_layer_norm(): invalid normalized_shape rank");
    }
    const int64_t axis = input.dim() - rank;
    int64_t M = 1, N = 1;
    for (int64_t d = 0; d < axis; ++d) M *= input.size(d);
    for (int64_t d = 0; d < rank; ++d) {
        if (input.size(axis + d) != normalized_shape[d]) {
            TP_THROW(ValueError, "quantized_layer_norm(): normalized_shape must match the input trailing dimensions");
        }
        N *= normalized_shape[d];
    }
    for (const auto* parameter : {&weight, &bias}) {
        if (parameter->has_value() && (*parameter)->shape() != normalized_shape) {
            TP_THROW(ValueError, "quantized_layer_norm(): affine parameter shape must equal normalized_shape");
        }
        check_normalization_affine(input, *parameter, N);
    }
    return {M, N, 1, 1, false, false};
}

inline NormalizationGeometry group_norm_geometry(
    const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight, const std::optional<Tensor>& bias) {
    check_normalization_input(input);
    if (input.dim() < 3) {
        TP_THROW(ValueError, "quantized_group_norm(): expected at least a 3d input");
    }
    const int64_t channels = input.size(1);
    if (num_groups <= 0 || channels % num_groups != 0) {
        TP_THROW(ValueError, "quantized_group_norm(): channels must be divisible by positive num_groups");
    }
    check_normalization_affine(input, weight, channels);
    check_normalization_affine(input, bias, channels);
    int64_t elements = channels / num_groups;
    for (int64_t d = 2; d < input.dim(); ++d) elements *= input.size(d);
    return {input.size(0) * num_groups, elements, channels, num_groups, true,
            input.dim() == 4 && input.is_contiguous(MemoryFormat::ChannelsLast)};
}

inline NormalizationGeometry instance_norm_geometry(
    const Tensor& input, const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias) {
    if (input.dim() < 3 || input.size(1) <= 0) {
        TP_THROW(ValueError, "quantized_instance_norm(): expected at least a 3d input with positive channels");
    }
    return group_norm_geometry(input, input.size(1), weight, bias);
}

inline Tensor normalization_output(const Tensor& input,
                                   const NormalizationGeometry& geometry,
                                   double scale, int64_t zero_point) {
    auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, input.dtype());
    Tensor out = geometry.channels_last
        ? Tensor::empty_strided(input.shape(), input.strides(), input.dtype(), input.device())
        : Tensor::empty(input.shape(), input.dtype(), input.device());
    out.impl()->set_quantizer(std::move(quantizer));
    return out;
}

#if defined(__CUDACC__)
#define TP_NORM_HD __host__ __device__ inline
#else
#define TP_NORM_HD inline
#endif

// Translate a group's logical channel-major index into its physical storage.
TP_NORM_HD int64_t normalization_offset(
    int64_t group, int64_t element, const NormalizationGeometry& geometry) {
    if (!geometry.channels_last) return group * geometry.elements + element;
    const int64_t channels_per_group = geometry.channels / geometry.channel_groups;
    const int64_t spatial = geometry.elements / channels_per_group;
    const int64_t channel = (group % geometry.channel_groups) * channels_per_group
                            + element / spatial;
    return (group / geometry.channel_groups) * geometry.elements * geometry.channel_groups
           + (element % spatial) * geometry.channels + channel;
}

template <typename Storage>
TP_NORM_HD Storage normalization_requantize(float value, float inv_scale,
                                           int64_t zero_point) {
    // Clamp before converting to storage, including the positive Int32 endpoint
    // which cannot be represented exactly in Float32.
    const double rounded = static_cast<double>(::nearbyintf(value * inv_scale))
                           + static_cast<double>(zero_point);
    constexpr double lower = static_cast<double>(std::numeric_limits<Storage>::min());
    constexpr double upper = static_cast<double>(std::numeric_limits<Storage>::max());
    return static_cast<Storage>(rounded < lower ? lower : rounded > upper ? upper : rounded);
}

#undef TP_NORM_HD

} // namespace tensorplay::quant
