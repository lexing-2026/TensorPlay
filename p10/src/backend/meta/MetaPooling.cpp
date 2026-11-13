// Shape-only kernels for pooling on the meta device.
//
// Pooling rearranges (max) or combines (average) existing elements, so only
// the output extents are observable here.  Max pooling reports the winning
// linear offsets alongside the pooled values, in an Int64 tensor of the same
// shape.  The output geometry is validated with the same formulas the
// computing backends use, so a call they would reject is rejected here too.

#include "MetaCommon.h"

#include "DType.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// Floor division that rounds toward negative infinity, matching the extent
// formula for strided windows with padding.
int64_t div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t pooling_output_shape(int64_t in, int64_t k, int64_t pad,
                             int64_t stride, int64_t dilation,
                             bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    int64_t out = div_rtn(in + 2 * pad - dilation * (k - 1) - 1 +
                              (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

std::vector<int64_t> expand_pool_param(const std::vector<int64_t>& list,
                                       const char* name, int64_t n,
                                       int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (static_cast<int64_t>(list.size()) != n) {
        TP_THROW(ValueError, std::string(name) + ": expected " +
                                 std::to_string(n) + " values");
    }
    return list;
}

std::pair<int64_t, int64_t> pool_pair(const std::vector<int64_t>& list,
                                      int64_t default_val) {
    if (list.empty()) return {default_val, default_val};
    if (list.size() == 1) return {list[0], list[0]};
    return {list[0], list[1]};
}

// ---------------------------------------------------------------------------
// Max pooling.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> max_pool2d_with_indices_meta(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        auto r = max_pool2d_with_indices_meta(
            input.unsqueeze(0), kernel_size, stride, padding, dilation, ceil_mode);
        return {std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0)};
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "max_pool2d_with_indices: Expected 4D input");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H_in = input.size(2), W_in = input.size(3);

    const auto ks = expand_pool_param(kernel_size,
                                      "max_pool2d_with_indices kernel_size", 2, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride,
                                      "max_pool2d_with_indices stride", 2, ks[0]);
    const auto pd = expand_pool_param(padding,
                                      "max_pool2d_with_indices padding", 2, 0);
    const auto dl = expand_pool_param(dilation,
                                      "max_pool2d_with_indices dilation", 2, 1);
    const int64_t H_out = pooling_output_shape(H_in, ks[0], pd[0], st[0], dl[0],
                                               ceil_mode);
    const int64_t W_out = pooling_output_shape(W_in, ks[1], pd[1], st[1], dl[1],
                                               ceil_mode);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError,
                 "max_pool2d_with_indices: Calculated output size is too small");
    }
    const std::vector<int64_t> out_shape{N, C, H_out, W_out};
    return {empty_meta(out_shape, input.dtype(), input.device()),
            empty_meta(out_shape, DType::Int64, input.device())};
}

Tensor max_pool2d_meta(const Tensor& input,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    return std::get<0>(max_pool2d_with_indices_meta(
        input, kernel_size, stride, padding, dilation, ceil_mode));
}

// ---------------------------------------------------------------------------
// Average pooling.
// ---------------------------------------------------------------------------

Tensor avg_pool2d_meta(const Tensor& input,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding, bool ceil_mode,
                       bool /*count_include_pad*/,
                       std::optional<int64_t> divisor_override) {
    if (divisor_override.has_value() && *divisor_override == 0) {
        TP_THROW(RuntimeError, "divisor must be not zero");
    }
    if (input.dim() == 3) {
        return avg_pool2d_meta(input.unsqueeze(0), kernel_size, stride, padding,
                               ceil_mode, /*count_include_pad=*/true,
                               divisor_override)
            .squeeze(0);
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "avg_pool2d: Expected 4D input");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t H_in = input.size(2), W_in = input.size(3);

    const auto [kH, kW] = pool_pair(kernel_size, 0);
    const auto [sH, sW] = pool_pair(stride.empty() ? kernel_size : stride, 0);
    const auto [pH, pW] = pool_pair(padding, 0);

    int64_t H_out, W_out;
    if (ceil_mode) {
        H_out = static_cast<int64_t>(
                    std::ceil((float)(H_in + 2 * pH - kH) / sH)) + 1;
        W_out = static_cast<int64_t>(
                    std::ceil((float)(W_in + 2 * pW - kW) / sW)) + 1;
        if (H_out > 1 && (H_out - 1) * sH >= H_in + pH) --H_out;
        if (W_out > 1 && (W_out - 1) * sW >= W_in + pW) --W_out;
    } else {
        H_out = (H_in + 2 * pH - kH) / sH + 1;
        W_out = (W_in + 2 * pW - kW) / sW + 1;
    }
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "avg_pool2d: Calculated output size is too small");
    }
    return empty_meta({N, C, H_out, W_out}, input.dtype(), input.device());
}

// ---------------------------------------------------------------------------
// Adaptive pooling.
// ---------------------------------------------------------------------------

Tensor adaptive_avg_pool2d_meta(const Tensor& input,
                                const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_avg_pool2d_meta(input.unsqueeze(0), output_size)
            .squeeze(0);
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    }
    const auto [H_out, W_out] = pool_pair(output_size, 0);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");
    }
    return empty_meta({input.size(0), input.size(1), H_out, W_out},
                      input.dtype(), input.device());
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices_meta(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        auto r = adaptive_max_pool2d_with_indices_meta(input.unsqueeze(0),
                                                       output_size);
        return {std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0)};
    }
    if (input.dim() != 4) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_with_indices: Expected 4D input");
    }
    const auto [H_out, W_out] = pool_pair(output_size, 0);
    if (H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool2d_with_indices: Invalid output size");
    }
    const std::vector<int64_t> out_shape{input.size(0), input.size(1), H_out,
                                         W_out};
    return {empty_meta(out_shape, input.dtype(), input.device()),
            empty_meta(out_shape, DType::Int64, input.device())};
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaPooling) {
    m.impl("max_pool2d_with_indices", max_pool2d_with_indices_meta);
    m.impl("max_pool2d", max_pool2d_meta);
    m.impl("avg_pool2d", avg_pool2d_meta);
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_meta);
    m.impl("adaptive_max_pool2d_with_indices",
           adaptive_max_pool2d_with_indices_meta);
}

} // namespace meta
} // namespace tensorplay