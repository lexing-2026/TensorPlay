#pragma once

#include "Macros.h"
#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay {

// Tensor memory layouts. A tensor carries at most one format tag: Contiguous
// (row-major) or one of the channels-last
// layouts; Preserve is a request-level sentinel (factories/clones) and is
// never stored on a TensorImpl.
enum class MemoryFormat : int8_t {
    Contiguous = 0,
    Preserve = 1,
    ChannelsLast = 2,   // NHWC (4-D)
    ChannelsLast3d = 3, // NDHWC (5-D)
};

P10_API const char* toString(MemoryFormat format);

// Channel-second strides ("dim 1 moves to the end"): for sizes
// [N,C,H,W] -> [C*H*W, 1, C*W, C]; [N,C,D,H,W] -> [C*D*H*W, 1, C*H*W, C*W, C].
// fall back to row-major (the layout is not representable there).
inline std::vector<int64_t> get_channels_last_strides(
    const std::vector<int64_t>& sizes) {
    const int64_t ndim = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> strides(ndim);
    if (ndim < 3) {
        int64_t prod = 1;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            strides[i] = prod;
            prod *= sizes[i];
        }
        return strides;
    }
    strides[1] = 1;
    strides[ndim - 1] = sizes[1];
    for (int64_t i = ndim - 2; i >= 2; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    strides[0] = strides[2] * sizes[2];
    return strides;
}

// Strides a freshly materialized tensor should adopt for `format`.
// Preserve must be resolved by the caller before calling this.
inline std::vector<int64_t> get_strides_for(
    const std::vector<int64_t>& sizes, MemoryFormat format) {
    switch (format) {
        case MemoryFormat::Contiguous: {
            std::vector<int64_t> strides(sizes.size());
            int64_t prod = 1;
            for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
                strides[i] = prod;
                prod *= sizes[i];
            }
            return strides;
        }
        case MemoryFormat::ChannelsLast:
        case MemoryFormat::ChannelsLast3d:
            return get_channels_last_strides(sizes);
        default:
            return {};
    }
}

} // namespace tensorplay
