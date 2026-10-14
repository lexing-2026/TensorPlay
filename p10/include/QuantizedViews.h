#pragma once

#include "Quantizer.h"
#include "MemoryFormat.h"
#include <algorithm>
#include <optional>
#include <vector>

namespace tensorplay::quantized {

inline Tensor squeeze_view(const Tensor& self,
                           const std::vector<int64_t>& sizes,
                           const std::vector<int64_t>& strides,
                           const std::optional<std::vector<int64_t>>& dims = std::nullopt) {
    Tensor out = self.as_strided(sizes, strides);
    if (!self.impl()->has_quantizer()) return out;
    const auto q = self.impl()->quantizer();
    if (q->qscheme() != kPerChannelAffine) return out;
    int64_t shift = 0;
    const int64_t axis = q->axis();
    for (int64_t d = 0; d < self.dim(); ++d) {
        const bool selected = !dims || std::find(dims->begin(), dims->end(), d) != dims->end();
        if (!selected || self.size(d) != 1) continue;
        TP_CHECK(d != axis, "cannot squeeze the quantized channel axis");
        if (d < axis) ++shift;
    }
    out.impl()->set_quantizer(make_per_channel_affine_quantizer(
        q->scales(), q->zero_points(), axis - shift, self.dtype()));
    return out;
}

inline Tensor unsqueeze_view(const Tensor& self,
                             const std::vector<int64_t>& sizes,
                             const std::vector<int64_t>& strides, int64_t dim) {
    Tensor out = self.as_strided(sizes, strides);
    if (!self.impl()->has_quantizer()) return out;
    const auto q = self.impl()->quantizer();
    if (q->qscheme() == kPerChannelAffine) {
        const int64_t axis = q->axis() + (q->axis() >= dim);
        out.impl()->set_quantizer(make_per_channel_affine_quantizer(
            q->scales(), q->zero_points(), axis, self.dtype()));
    }
    return out;
}

inline void check_quantized_strides(const Tensor& self) {
    if (self.impl()->has_quantizer()) {
        TP_CHECK(self.impl()->quantizer()->qscheme() == kPerTensorAffine,
                 "setting strides requires uniform affine quantization");
    }
}

inline Tensor empty_like(const Tensor& self, DType dtype, Device device) {
    require_quantized(self, "empty_like");
    TP_CHECK(dtype == self.dtype(), "empty_like: dtype must match the quantized input");
    const auto q = quantizer_of(self);
    TP_CHECK(q->qscheme() == kPerTensorAffine || q->qscheme() == kPerChannelAffine,
             "empty_like: unsupported quantization scheme");
    const auto sizes = self.sizes().vec();
    Tensor out = Tensor::empty(sizes, dtype, device);
    if (!self.is_contiguous() &&
        ((self.dim() == 4 && self.is_contiguous(MemoryFormat::ChannelsLast)) ||
         (self.dim() == 5 && self.is_contiguous(MemoryFormat::ChannelsLast3d)))) {
        out = out.as_strided(sizes, get_channels_last_strides(sizes));
    }
    if (q->qscheme() == kPerTensorAffine) {
        out.impl()->set_quantizer(make_per_tensor_affine_quantizer(q->scale(), q->zero_point(), dtype));
    } else {
        Tensor scales = q->scales().clone().to(device);
        Tensor zeros = q->zero_points().clone().to(device);
        out.impl()->set_quantizer(make_per_channel_affine_quantizer(scales, zeros, q->axis(), dtype));
    }
    return out;
}

} // namespace tensorplay::quantized
