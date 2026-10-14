#pragma once

// Shared requantization primitives for the quantized kernels.  The affine
// grid contract is: q = clamp(nearbyint(y / out_scale) + out_zero_point),
// with round-half-even semantics and saturation at the Int8 range bounds.

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define TP_QUANT_HD __device__ __host__ inline
#else
#define TP_QUANT_HD inline
#endif

namespace tensorplay {
namespace quant {

TP_QUANT_HD int8_t requantize_value(double y, double inv_out_scale,
                                    int64_t out_zero_point) {
#if defined(__CUDACC__)
    const double rounded =
        ::rint(y * inv_out_scale) + static_cast<double>(out_zero_point);
    long long clamped = static_cast<long long>(rounded);
    if (clamped < -128LL) clamped = -128LL;
    if (clamped > 127LL) clamped = 127LL;
    return static_cast<int8_t>(clamped);
#else
    const double rounded =
        std::nearbyint(y * inv_out_scale) + static_cast<double>(out_zero_point);
    return static_cast<int8_t>(
        std::min<int64_t>(127, std::max<int64_t>(
                                   -128, static_cast<int64_t>(rounded))));
#endif
}

} // namespace quant
} // namespace tensorplay

#undef TP_QUANT_HD