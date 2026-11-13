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
    if (sH == 0 || sW == 0) {
        TP_THROW(RuntimeError, "stride should not be zero");
    }

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

Tensor adaptive_max_pool2d_meta(const Tensor& input,
                                const std::vector<int64_t>& output_size) {
    return std::get<0>(
        adaptive_max_pool2d_with_indices_meta(input, output_size));
}

// ---------------------------------------------------------------------------
// One-dimensional pooling.
//
// A 1-D window is a 2-D window whose height axis holds one element, so the
// wrappers below fold the parameters onto the width axis and delegate.  The
// parameter lists must hold exactly one value; the dims must be batch-first
// or unbatched.
// ---------------------------------------------------------------------------

void check_pool1d_dims(const Tensor& self, const char* function) {
    if (self.dim() < 2 || self.dim() > 3) {
        TP_THROW(RuntimeError, function, ": Expected 2D or 3D input, but got ",
                 self.dim(), "D");
    }
}

int64_t check_pool1d_arg(const std::vector<int64_t>& value,
                         const char* function, const char* argument,
                         bool allow_empty, int64_t empty_value) {
    if (allow_empty && value.empty()) return empty_value;
    if (value.size() != 1) {
        TP_THROW(RuntimeError, function, "() argument '", argument,
                 "' should contain one int (got ", value.size(), ")");
    }
    return value[0];
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices_meta(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    check_pool1d_dims(self, "max_pool1d");
    const int64_t kernel = check_pool1d_arg(kernel_size, "max_pool1d",
                                            "kernel_size", false, 0);
    const int64_t step = check_pool1d_arg(stride, "max_pool1d", "stride", true,
                                          kernel);
    const int64_t pad = check_pool1d_arg(padding, "max_pool1d", "padding", true,
                                         0);
    const int64_t dil = check_pool1d_arg(dilation, "max_pool1d", "dilation",
                                         true, 1);
    auto pooled = max_pool2d_with_indices_meta(self.unsqueeze(-2), {1, kernel},
                                               {1, step}, {0, pad}, {1, dil},
                                               ceil_mode);
    return {std::get<0>(pooled).squeeze(-2),
            std::get<1>(pooled).squeeze(-2)};
}

Tensor max_pool1d_meta(const Tensor& self,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (self.dim() != 2 && self.dim() != 3) {
        TP_THROW(RuntimeError, "max_pool1d() expected 2D or 3D input tensor");
    }
    if (kernel_size.size() != 1) {
        TP_THROW(RuntimeError, "max_pool1d() kernel_size must contain one int");
    }
    if (!stride.empty() && stride.size() != 1) {
        TP_THROW(RuntimeError, "max_pool1d() stride must contain one int");
    }
    if (padding.size() != 1 || dilation.size() != 1) {
        TP_THROW(RuntimeError,
                 "max_pool1d() padding and dilation must contain one int");
    }
    const int64_t step = stride.empty() ? kernel_size[0] : stride[0];
    if ((self.dim() == 2 && (self.size(0) == 0 || self.size(1) == 0)) ||
        (self.dim() == 3 && (self.size(1) == 0 || self.size(2) == 0))) {
        TP_THROW(RuntimeError,
                 "max_pool1d(): input dimensions must be non-zero");
    }
    if (kernel_size[0] <= 0) {
        TP_THROW(RuntimeError,
                 "max_pool1d() kernel_size must be greater than zero");
    }
    if (step <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() stride must be greater than zero");
    }
    if (padding[0] < 0) {
        TP_THROW(RuntimeError, "max_pool1d() padding must be non-negative");
    }
    if (padding[0] > kernel_size[0] / 2) {
        TP_THROW(RuntimeError,
                 "max_pool1d() padding should be at most half of kernel size");
    }
    if (dilation[0] <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() dilation must be greater than zero");
    }
    const int64_t nb = self.dim() == 3 ? self.size(-3) : 1;
    const int64_t nc = self.size(-2);
    const int64_t iw = self.size(-1);
    const int64_t ow = pooling_output_shape(iw, kernel_size[0], padding[0], step,
                                            dilation[0], ceil_mode);
    if (ow <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() Invalid computed output size");
    }
    const Tensor out = empty_meta({nb, nc, ow}, self.dtype(), self.device());
    return self.dim() == 2 ? out.squeeze(0) : out;
}

Tensor avg_pool1d_meta(const Tensor& self,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding, bool ceil_mode,
                       bool count_include_pad) {
    check_pool1d_dims(self, "avg_pool1d");
    const int64_t kernel = check_pool1d_arg(kernel_size, "avg_pool1d",
                                            "kernel_size", false, 0);
    const int64_t step = check_pool1d_arg(stride, "avg_pool1d", "stride", true,
                                          kernel);
    const int64_t pad = check_pool1d_arg(padding, "avg_pool1d", "padding", true,
                                         0);
    return avg_pool2d_meta(self.unsqueeze(-2), {1, kernel}, {1, step}, {0, pad},
                           ceil_mode, count_include_pad, std::nullopt)
        .squeeze(-2);
}

Tensor adaptive_avg_pool1d_meta(const Tensor& self,
                                const std::vector<int64_t>& output_size) {
    check_pool1d_dims(self, "adaptive_avg_pool1d");
    const int64_t output = check_pool1d_arg(output_size, "adaptive_avg_pool1d",
                                            "output_size", false, 0);
    return adaptive_avg_pool2d_meta(self.unsqueeze(-2), {1, output})
        .squeeze(-2);
}

std::tuple<Tensor, Tensor> adaptive_max_pool1d_meta(
    const Tensor& self, const std::vector<int64_t>& output_size) {
    check_pool1d_dims(self, "adaptive_max_pool1d");
    const int64_t output = check_pool1d_arg(output_size, "adaptive_max_pool1d",
                                            "output_size", false, 0);
    for (int64_t dim = 1; dim < self.dim(); ++dim) {
        if (self.size(dim) == 0) {
            TP_THROW(RuntimeError,
                     "adaptive_max_pool1d(): Expected input to have non-zero "
                     "size for non-batch dimensions");
        }
    }
    auto pooled = adaptive_max_pool2d_with_indices_meta(self.unsqueeze(-2),
                                                        {1, output});
    return {std::get<0>(pooled).squeeze(-2),
            std::get<1>(pooled).squeeze(-2)};
}

// ---------------------------------------------------------------------------
// Three-dimensional pooling.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> max_pool3d_with_indices_meta(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        auto r = max_pool3d_with_indices_meta(input.unsqueeze(0), kernel_size,
                                              stride, padding, dilation,
                                              ceil_mode);
        return {std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0)};
    }
    if (input.dim() != 5) {
        TP_THROW(RuntimeError, "max_pool3d_with_indices: Expected 5D input");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D_in = input.size(2), H_in = input.size(3),
                  W_in = input.size(4);

    const auto ks = expand_pool_param(kernel_size, "max_pool3d kernel_size", 3, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride,
                                      "max_pool3d stride", 3, ks[0]);
    const auto pd = expand_pool_param(padding, "max_pool3d padding", 3, 0);
    const auto dl = expand_pool_param(dilation, "max_pool3d dilation", 3, 1);
    const int64_t D_out = pooling_output_shape(D_in, ks[0], pd[0], st[0], dl[0],
                                               ceil_mode);
    const int64_t H_out = pooling_output_shape(H_in, ks[1], pd[1], st[1], dl[1],
                                               ceil_mode);
    const int64_t W_out = pooling_output_shape(W_in, ks[2], pd[2], st[2], dl[2],
                                               ceil_mode);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError,
                 "max_pool3d: Calculated output size is too small");
    }
    const std::vector<int64_t> out_shape{N, C, D_out, H_out, W_out};
    return {empty_meta(out_shape, input.dtype(), input.device()),
            empty_meta(out_shape, DType::Int64, input.device())};
}

Tensor max_pool3d_meta(const Tensor& input,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        return max_pool3d_meta(input.unsqueeze(0), kernel_size, stride, padding,
                               dilation, ceil_mode)
            .squeeze(0);
    }
    return std::get<0>(max_pool3d_with_indices_meta(
        input, kernel_size, stride, padding, dilation, ceil_mode));
}

Tensor avg_pool3d_meta(const Tensor& input,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding, bool ceil_mode,
                       bool /*count_include_pad*/,
                       std::optional<int64_t> divisor_override) {
    if (divisor_override.has_value() && *divisor_override == 0) {
        TP_THROW(RuntimeError, "divisor must be not zero");
    }
    if (input.dim() == 4) {
        return avg_pool3d_meta(input.unsqueeze(0), kernel_size, stride, padding,
                               ceil_mode, /*count_include_pad=*/true,
                               divisor_override)
            .squeeze(0);
    }
    if (input.dim() != 5) {
        TP_THROW(RuntimeError, "avg_pool3d: Expected 5D input");
    }
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const auto ks = expand_pool_param(kernel_size, "avg_pool3d kernel_size", 3, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride,
                                      "avg_pool3d stride", 3, ks[0]);
    const auto pd = expand_pool_param(padding, "avg_pool3d padding", 3, 0);
    if (st[0] == 0 || st[1] == 0 || st[2] == 0) {
        TP_THROW(RuntimeError, "stride should not be zero");
    }
    auto out_size = [&](int64_t in, int64_t k, int64_t s, int64_t p) {
        int64_t num = in + 2 * p - k + (ceil_mode ? s - 1 : 0);
        int64_t out = num / s;
        if (num % s != 0 && (num < 0) != (s < 0)) --out;
        out += 1;
        if (ceil_mode && (out - 1) * s >= in + p) --out;
        return out;
    };
    const int64_t D_out = out_size(D, ks[0], st[0], pd[0]);
    const int64_t H_out = out_size(H, ks[1], st[1], pd[1]);
    const int64_t W_out = out_size(W, ks[2], st[2], pd[2]);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "avg_pool3d: Calculated output size is too small");
    }
    return empty_meta({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
}

Tensor adaptive_avg_pool3d_meta(const Tensor& input,
                                const std::vector<int64_t>& output_size) {
    if (input.dim() == 4) {
        return adaptive_avg_pool3d_meta(input.unsqueeze(0), output_size)
            .squeeze(0);
    }
    if (input.dim() != 5) {
        TP_THROW(RuntimeError, "adaptive_avg_pool3d: Expected 5D input");
    }
    if (output_size.size() != 3) {
        TP_THROW(RuntimeError, "adaptive_avg_pool3d: output_size must be 3");
    }
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    if (oD < 0 || oH < 0 || oW < 0) {
        TP_THROW(RuntimeError,
                 "adaptive_avg_pool3d: elements of output_size must be greater "
                 "than or equal to 0 but received {", oD, ", ", oH, ", ", oW,
                 "}");
    }
    return empty_meta({input.size(0), input.size(1), oD, oH, oW},
                      input.dtype(), input.device());
}

std::tuple<Tensor, Tensor> adaptive_max_pool3d_meta(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 4) {
        auto r = adaptive_max_pool3d_meta(input.unsqueeze(0), output_size);
        return {std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0)};
    }
    if (input.dim() != 5) {
        TP_THROW(RuntimeError, "adaptive_max_pool3d: Expected 5D input");
    }
    if (output_size.size() != 3) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool3d(): internal error: output_size.size() "
                 "must be 3");
    }
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    if (oD <= 0 || oH <= 0 || oW <= 0) {
        TP_THROW(RuntimeError, "adaptive_max_pool3d: Invalid output size");
    }
    const std::vector<int64_t> out_shape{input.size(0), input.size(1), oD, oH,
                                         oW};
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
    m.impl("adaptive_max_pool2d", adaptive_max_pool2d_meta);
    m.impl("max_pool1d_with_indices", max_pool1d_with_indices_meta);
    m.impl("max_pool1d", max_pool1d_meta);
    m.impl("avg_pool1d", avg_pool1d_meta);
    m.impl("adaptive_avg_pool1d", adaptive_avg_pool1d_meta);
    m.impl("adaptive_max_pool1d", adaptive_max_pool1d_meta);
    m.impl("max_pool3d_with_indices", max_pool3d_with_indices_meta);
    m.impl("max_pool3d", max_pool3d_meta);
    m.impl("avg_pool3d", avg_pool3d_meta);
    m.impl("adaptive_avg_pool3d", adaptive_avg_pool3d_meta);
    m.impl("adaptive_max_pool3d", adaptive_max_pool3d_meta);
}

} // namespace meta
} // namespace tensorplay