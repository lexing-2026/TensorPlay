#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <string>

// Dispatcher-level *_with_indices entry points for the pooling composites
// (defined in TPXOpsGenerated.cpp; declared locally -- same pattern as
// LinearAlgebraKernels.cpp).  Declared at global scope before
// `namespace tensorplay` below so the names land in the real
// tensorplay::tpx::ops.
namespace tensorplay {
namespace tpx {
namespace ops {
TENSORPLAY_API std::tuple<Tensor, Tensor> max_pool2d_with_indices(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode);
TENSORPLAY_API std::tuple<Tensor, Tensor> max_pool3d_with_indices(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode);
TENSORPLAY_API std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(const Tensor& input, const std::vector<int64_t>& output_size);
} // namespace ops
} // namespace tpx
} // namespace tensorplay

using namespace tensorplay::parallel;

namespace tensorplay {
namespace cpu {

// Helper to handle optional arguments or defaults
static std::pair<int64_t, int64_t> get_pair(const std::vector<int64_t>& list, int64_t default_val = 0) {
    if (list.empty()) return {default_val, default_val};
    if (list.size() == 1) return {list[0], list[0]};
    return {list[0], list[1]};
}

static std::pair<int64_t, int64_t> get_pair_from_kernel(const std::vector<int64_t>& list, const std::vector<int64_t>& kernel) {
    if (list.empty()) return get_pair(kernel);
    return get_pair(list);
}

// invoked lambda, scalar_t hint inside, Double before Float, and the exact
// '"kernel" not implemented for '<Type>'' wording on the default branch.
#define TP_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                        \
    [&]() {                                                                \
        const auto& the_type = TYPE;                                       \
        (void)the_type;                                                    \
        switch (the_type) {                                                \
            case DType::Float64: {                                         \
                using scalar_t [[maybe_unused]] = double;                  \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Float32: {                                         \
                using scalar_t [[maybe_unused]] = float;                   \
                return __VA_ARGS__();                                      \
            }                                                              \
            default:                                                       \
                TP_THROW(NotImplementedError,                              \
                    std::string("\"") + NAME +                             \
                    "\" not implemented for '" +                           \
                    tensorplay::toString(the_type) + "'");                 \
        }                                                                  \
    }()

// Numeric dispatch covers byte, char, short, int, long, float, and double.
#define TP_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                             \
    [&]() {                                                                \
        const auto& the_type = TYPE;                                       \
        (void)the_type;                                                    \
        switch (the_type) {                                                \
            case DType::Float64: {                                         \
                using scalar_t [[maybe_unused]] = double;                  \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Float32: {                                         \
                using scalar_t [[maybe_unused]] = float;                   \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Int64: {                                           \
                using scalar_t [[maybe_unused]] = int64_t;                 \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Int32: {                                           \
                using scalar_t [[maybe_unused]] = int32_t;                 \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Int16: {                                           \
                using scalar_t [[maybe_unused]] = int16_t;                 \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Int8: {                                            \
                using scalar_t [[maybe_unused]] = int8_t;                  \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::UInt8: {                                           \
                using scalar_t [[maybe_unused]] = uint8_t;                 \
                return __VA_ARGS__();                                      \
            }                                                              \
            default:                                                       \
                TP_THROW(NotImplementedError,                              \
                    std::string("\"") + NAME +                             \
                    "\" not implemented for '" +                           \
                    tensorplay::toString(the_type) + "'");                 \
        }                                                                  \
    }()

// AT_DISPATCH_FLOATING_TYPES_AND3(kLong, kBFloat16, kHalf) minus the half
// types (p10 has no compute path for them yet): floats + Long.
#define TP_DISPATCH_FLOATING_TYPES_AND_LONG(TYPE, NAME, ...)               \
    [&]() {                                                                \
        const auto& the_type = TYPE;                                       \
        (void)the_type;                                                    \
        switch (the_type) {                                                \
            case DType::Float64: {                                         \
                using scalar_t [[maybe_unused]] = double;                  \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Float32: {                                         \
                using scalar_t [[maybe_unused]] = float;                   \
                return __VA_ARGS__();                                      \
            }                                                              \
            case DType::Int64: {                                           \
                using scalar_t [[maybe_unused]] = int64_t;                 \
                return __VA_ARGS__();                                      \
            }                                                              \
            default:                                                       \
                TP_THROW(NotImplementedError,                              \
                    std::string("\"") + NAME +                             \
                    "\" not implemented for '" +                           \
                    tensorplay::toString(the_type) + "'");                 \
        }                                                                  \
    }()

Tensor avg_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size,
                      const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                      bool ceil_mode, bool count_include_pad,
                      std::optional<int64_t> divisor_override) {
    if (input.dim() == 4) {
        return avg_pool3d_cpu(input.unsqueeze(0), kernel_size, stride, padding,
                              ceil_mode, count_include_pad,
                              divisor_override).squeeze(0);
    }
    if (input.dim() != 5) TP_THROW(RuntimeError, "avg_pool3d: Expected 5D input");
    const Tensor input_c = input.contiguous();
    int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D = input_c.size(2), H = input_c.size(3), W = input_c.size(4);
    const int64_t kd = kernel_size[0], kh = kernel_size[1], kw = kernel_size[2];
    const int64_t sd = stride[0], sh = stride[1], sw = stride[2];
    const int64_t pd_ = padding[0], ph = padding[1], pw = padding[2];
    auto out_size = [&](int64_t in, int64_t k, int64_t s, int64_t p) {
        return ceil_mode ? (in + 2 * p - k + s - 1) / s + 1
                         : (in + 2 * p - k) / s + 1;
    };
    const int64_t oD = out_size(D, kd, sd, pd_);
    const int64_t oH = out_size(H, kh, sh, ph);
    const int64_t oW = out_size(W, kw, sw, pw);
    Tensor out = Tensor::empty({N, C, oD, oH, oW}, input.dtype(), input.device());
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "avg_pool3d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();
        parallel_for(0, N * C * oD * oH, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t oh = idx % oH;
                const int64_t od = (idx / oH) % oD;
                const int64_t nc = idx / (oH * oD);
                const scalar_t* in_base = in_ptr + nc * D * H * W;
                scalar_t* out_row = out_ptr + idx * oW;

                const int64_t d0 = od * sd - pd_, h0 = oh * sh - ph;
                const int64_t dlo = std::max(d0, int64_t(0)), d1 = std::min(d0 + kd, D);
                const int64_t hlo = std::max(h0, int64_t(0)), h1 = std::min(h0 + kh, H);

                for (int64_t ow = 0; ow < oW; ++ow) {
                    const int64_t w0 = ow * sw - pw;
                    const int64_t wlo = std::max(w0, int64_t(0)), w1 = std::min(w0 + kw, W);
                    scalar_t sum = scalar_t(0);
                    for (int64_t d = dlo; d < d1; ++d)
                    for (int64_t h = hlo; h < h1; ++h) {
                        const scalar_t* row = in_base + (d * H + h) * W;
                        for (int64_t w = wlo; w < w1; ++w) sum += row[w];
                    }
                    const int64_t cnt = (d1 - dlo) * (h1 - hlo) * (w1 - wlo);
                    int64_t div = divisor_override.has_value()
                                      ? *divisor_override
                                      : (count_include_pad ? kd * kh * kw : cnt);
                    out_row[ow] = div > 0 ? sum / static_cast<scalar_t>(div) : scalar_t(0);
                }
            }
        });
    });
    return out;
}

Tensor max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        return max_pool2d_cpu(input.unsqueeze(0), kernel_size, stride, padding,
                              dilation, ceil_mode).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d: Expected 4D input");
    // The kernel indexes raw NCHW pointers; normalize views (no-op when
    const Tensor input_c = input.contiguous();

    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t H_in = input_c.size(2);
    int64_t W_in = input_c.size(3);
    
    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    auto [dH, dW] = get_pair(dilation, 1);
    
    int64_t H_out, W_out;
    if (ceil_mode) {
        H_out = (int64_t)(std::ceil((float)(H_in + 2 * pH - dH * (kH - 1) - 1) / sH)) + 1;
        W_out = (int64_t)(std::ceil((float)(W_in + 2 * pW - dW * (kW - 1) - 1) / sW)) + 1;
    } else {
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    }

    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "max_pool2d: Calculated output size is too small");
    
    // Ensure padding doesn't make us start reading out of bounds if ceil_mode used?
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool2d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();

        // Parallel over (n, c, h_out); per-row window bounds are hoisted so
        // the innermost loop is a contiguous, branch-free scan (vectorizes)
        // when dilation == 1.
        parallel_for(0, N * C * H_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % H_out;
                const int64_t nc = idx / H_out;
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* out_row = out_ptr + idx * W_out;

                const int64_t h_start = h * sH - pH;
                int64_t kh0 = 0, kh1 = kH;
                if (dH == 1) {
                    if (h_start < 0) kh0 = -h_start;
                    if (h_start + kH > H_in) kh1 = H_in - h_start;
                } else {
                    while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                    while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H_in) --kh1;
                }

                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_start = w * sW - pW;
                    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                    if (dW == 1) {
                        const int64_t kw0 = w_start < 0 ? -w_start : 0;
                        const int64_t kw1 = w_start + kW > W_in ? W_in - w_start : kW;
                        for (int64_t kh = kh0; kh < kh1; ++kh) {
                            const scalar_t* row = in_base + (h_start + kh) * W_in + w_start;
                            for (int64_t kw = kw0; kw < kw1; ++kw) {
                                const scalar_t val = row[kw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    } else {
                        for (int64_t kh = kh0; kh < kh1; ++kh) {
                            const scalar_t* row = in_base + (h_start + kh * dH) * W_in;
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                const int64_t w_in = w_start + kw * dW;
                                if (w_in >= 0 && w_in < W_in) {
                                    const scalar_t val = row[w_in];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    out_row[w] = max_val;
                }
            }
        });
    });
    
    return out;
}

Tensor avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (input.dim() == 3) {
        return avg_pool2d_cpu(input.unsqueeze(0), kernel_size, stride, padding,
                              ceil_mode, count_include_pad, divisor_override).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d: Expected 4D input");
    
    const Tensor input_c = input.contiguous();
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    
    int64_t H_out, W_out;
    if (ceil_mode) {
        H_out = (int64_t)(std::ceil((float)(H_in + 2 * pH - kH) / sH)) + 1;
        W_out = (int64_t)(std::ceil((float)(W_in + 2 * pW - kW) / sW)) + 1;
        if (H_out > 1 && (H_out - 1) * sH >= H_in + pH) --H_out;
        if (W_out > 1 && (W_out - 1) * sW >= W_in + pW) --W_out;
    } else {
        H_out = (H_in + 2 * pH - kH) / sH + 1;
        W_out = (W_in + 2 * pW - kW) / sW + 1;
    }

    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "avg_pool2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "avg_pool2d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input.data_ptr<scalar_t>();

        parallel_for(0, N * C * H_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % H_out;
                const int64_t nc = idx / H_out;
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* out_row = out_ptr + idx * W_out;

                const int64_t h_start = h * sH - pH;
                // Rows clipped to the input (padding rows contribute nothing).
                const int64_t ih0 = h_start < 0 ? 0 : h_start;
                const int64_t ih1 = h_start + kH > H_in ? H_in : h_start + kH;
                // Window extent over input+padding for count_include_pad.
                const int64_t clip_h = std::min(h_start + kH, H_in + pH) - h_start;

                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_start = w * sW - pW;
                    const int64_t iw0 = w_start < 0 ? 0 : w_start;
                    const int64_t iw1 = w_start + kW > W_in ? W_in : w_start + kW;
                    const int64_t clip_w = std::min(w_start + kW, W_in + pW) - w_start;

                    scalar_t sum = 0;
                    for (int64_t ih = ih0; ih < ih1; ++ih) {
                        const scalar_t* row = in_base + ih * W_in;
                        for (int64_t iw = iw0; iw < iw1; ++iw) sum += row[iw];
                    }

                    scalar_t divisor;
                    if (divisor_override.has_value()) {
                        divisor = (scalar_t)divisor_override.value();
                    } else if (count_include_pad) {
                        divisor = (scalar_t)(clip_h * clip_w);
                    } else {
                        divisor = (scalar_t)((ih1 - ih0) * (iw1 - iw0));
                    }
                    out_row[w] = sum / divisor;
                }
            }
        });
    });
    
    return out;
}

Tensor adaptive_avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_avg_pool2d_cpu(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    
    const Tensor input_c = input.contiguous();
    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t H_in = input_c.size(2);
    int64_t W_in = input_c.size(3);
    
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "adaptive_avg_pool2d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input.data_ptr<scalar_t>();

        parallel_for(0, N * C * H_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % H_out;
                const int64_t nc = idx / H_out;
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* out_row = out_ptr + idx * W_out;

                const int64_t h_start = (h * H_in) / H_out;
                const int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;

                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_start = (w * W_in) / W_out;
                    const int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;

                    scalar_t sum = 0;
                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        const scalar_t* row = in_base + ih * W_in;
                        for (int64_t iw = w_start; iw < w_end; ++iw) sum += row[iw];
                    }
                    out_row[w] = sum / ((h_end - h_start) * (w_end - w_start));
                }
            }
        });
    });
    
    return out;
}

Tensor adaptive_max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        return adaptive_max_pool2d_cpu(input.unsqueeze(0), output_size).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d: Expected 4D input");
    
    const Tensor input_c = input.contiguous();
    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t H_in = input_c.size(2);
    int64_t W_in = input_c.size(3);
    
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_max_pool2d: Invalid output size");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());

    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool2d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();

        parallel_for(0, N * C * H_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % H_out;
                const int64_t nc = idx / H_out;
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* out_row = out_ptr + idx * W_out;

                // AdaptivePooling.h start_index/end_index: floor start,
                // ceil end -- the same bins as adaptive avg pooling.
                const int64_t h_start = (h * H_in) / H_out;
                const int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;

                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_start = (w * W_in) / W_out;
                    const int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

                    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        const scalar_t* row = in_base + ih * W_in;
                        for (int64_t iw = w_start; iw < w_end; ++iw) {
                            const scalar_t val = row[iw];
                            if ((val > max_val) || std::isnan(val)) max_val = val;
                        }
                    }
                    out_row[w] = max_val;
                }
            }
        });
    });

    return out;
}

Tensor max_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d_backward: Expected 4D input and grad_output");
    const Tensor input_c = input.contiguous();

    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t H_in = input_c.size(2);
    int64_t W_in = input_c.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    auto [dH, dW] = get_pair(dilation, 1);

    Tensor grad_input = Tensor::zeros_like(input);
    
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool2d_backward", [&]() {
        scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr = grad_output.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();

        // Scatter into grad_input: parallel over (n, c) planes (each plane is
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* gi_base = grad_in_ptr + nc * H_in * W_in;
                const scalar_t* go_base = grad_out_ptr + nc * H_out * W_out;

                for (int64_t h = 0; h < H_out; ++h) {
                    const int64_t h_start = h * sH - pH;
                    int64_t kh0 = 0, kh1 = kH;
                    if (dH == 1) {
                        if (h_start < 0) kh0 = -h_start;
                        if (h_start + kH > H_in) kh1 = H_in - h_start;
                    } else {
                        while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                        while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H_in) --kh1;
                    }
                    const scalar_t* go_row = go_base + h * W_out;

                    for (int64_t w = 0; w < W_out; ++w) {
                        const int64_t w_start = w * sW - pW;
                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        int64_t max_off = -1;
                        if (dW == 1) {
                            const int64_t kw0 = w_start < 0 ? -w_start : 0;
                            const int64_t kw1 = w_start + kW > W_in ? W_in - w_start : kW;
                            for (int64_t kh = kh0; kh < kh1; ++kh) {
                                const int64_t row_off = (h_start + kh) * W_in + w_start;
                                const scalar_t* row = in_base + row_off;
                                for (int64_t kw = kw0; kw < kw1; ++kw) {
                                    if (row[kw] > max_val) {
                                        max_val = row[kw];
                                        max_off = row_off + kw;
                                    }
                                }
                            }
                        } else {
                            for (int64_t kh = kh0; kh < kh1; ++kh) {
                                const int64_t row_off = (h_start + kh * dH) * W_in;
                                const scalar_t* row = in_base + row_off;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t w_in = w_start + kw * dW;
                                    if (w_in >= 0 && w_in < W_in && row[w_in] > max_val) {
                                        max_val = row[w_in];
                                        max_off = row_off + w_in;
                                    }
                                }
                            }
                        }
                        if (max_off != -1) gi_base[max_off] += go_row[w];
                    }
                }
            }
        });
    });
    return grad_input;
}

Tensor avg_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d_backward: Expected 4D input and grad_output");
    const Tensor input_c = input.contiguous();

    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t H_in = input_c.size(2);
    int64_t W_in = input_c.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);

    Tensor grad_input = Tensor::zeros_like(input);

    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "avg_pool2d_backward", [&]() {
        scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr = grad_output.data_ptr<scalar_t>();

        // Scatter into grad_input: parallel over (n, c) planes (independent,
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = grad_in_ptr + nc * H_in * W_in;
                const scalar_t* go_base = grad_out_ptr + nc * H_out * W_out;

                for (int64_t h = 0; h < H_out; ++h) {
                    const int64_t h_start = h * sH - pH;
                    const int64_t ih0 = h_start < 0 ? 0 : h_start;
                    const int64_t ih1 = h_start + kH > H_in ? H_in : h_start + kH;
                    const int64_t clip_h = std::min(h_start + kH, H_in + pH) - h_start;

                    for (int64_t w = 0; w < W_out; ++w) {
                        const int64_t w_start = w * sW - pW;
                        const int64_t iw0 = w_start < 0 ? 0 : w_start;
                        const int64_t iw1 = w_start + kW > W_in ? W_in : w_start + kW;
                        const int64_t clip_w = std::min(w_start + kW, W_in + pW) - w_start;

                        scalar_t divisor;
                        if (divisor_override.has_value()) {
                            divisor = (scalar_t)divisor_override.value();
                        } else if (count_include_pad) {
                            divisor = (scalar_t)(clip_h * clip_w);
                        } else {
                            divisor = (scalar_t)((ih1 - ih0) * (iw1 - iw0));
                        }

                        const scalar_t grad_val = go_base[h * W_out + w] / divisor;
                        for (int64_t ih = ih0; ih < ih1; ++ih) {
                            scalar_t* row = gi_base + ih * W_in;
                            for (int64_t iw = iw0; iw < iw1; ++iw) row[iw] += grad_val;
                        }
                    }
                }
            }
        });
    });
    return grad_input;
}

Tensor adaptive_avg_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    Tensor grad_input = Tensor::zeros_like(input);

    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "adaptive_avg_pool2d_backward", [&]() {
        scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr = grad_output.data_ptr<scalar_t>();

        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = grad_in_ptr + nc * H_in * W_in;
                const scalar_t* go_base = grad_out_ptr + nc * H_out * W_out;

                for (int64_t h = 0; h < H_out; ++h) {
                    const int64_t h_start = (h * H_in) / H_out;
                    const int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;

                    for (int64_t w = 0; w < W_out; ++w) {
                        const int64_t w_start = (w * W_in) / W_out;
                        const int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;

                        const scalar_t grad_val =
                            go_base[h * W_out + w] / ((h_end - h_start) * (w_end - w_start));
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            scalar_t* row = gi_base + ih * W_in;
                            for (int64_t iw = w_start; iw < w_end; ++iw) row[iw] += grad_val;
                        }
                    }
                }
            }
        });
    });
    return grad_input;
}

Tensor adaptive_max_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    Tensor grad_input = Tensor::zeros_like(input);

    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool2d_backward", [&]() {
        scalar_t* grad_in_ptr = grad_input.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr = grad_output.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input.data_ptr<scalar_t>();

        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* gi_base = grad_in_ptr + nc * H_in * W_in;
                const scalar_t* go_base = grad_out_ptr + nc * H_out * W_out;

                for (int64_t h = 0; h < H_out; ++h) {
                    // AdaptivePooling.h start_index/end_index (floor/ceil).
                    const int64_t h_start = (h * H_in) / H_out;
                    const int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;

                    for (int64_t w = 0; w < W_out; ++w) {
                        const int64_t w_start = (w * W_in) / W_out;
                        const int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        int64_t max_off = -1;
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            const scalar_t* row = in_base + ih * W_in;
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                const scalar_t val = row[iw];
                                if ((val > max_val) || std::isnan(val)) {
                                    max_val = val;
                                    max_off = ih * W_in + iw;
                                }
                            }
                        }
                        if (max_off != -1) gi_base[max_off] += go_base[h * W_out + w];
                    }
                }
            }
        });
    });
    return grad_input;
}

// backward scatter instead of recomputing the argmax.  Indices are plane
std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        auto r = adaptive_max_pool2d_with_indices_cpu(input.unsqueeze(0), output_size);
        return std::make_tuple(std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0));
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices: Expected 4D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices: Invalid output size");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({N, C, H_out, W_out}, DType::Int64, input.device());

    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool2d_with_indices", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        int64_t* idx_ptr = indices.data_ptr<int64_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();
        parallel_for(0, N * C * H_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % H_out;
                const int64_t nc = idx / H_out;
                const scalar_t* in_base = in_ptr + nc * H_in * W_in;
                scalar_t* out_row = out_ptr + idx * W_out;
                int64_t* idx_row = idx_ptr + idx * W_out;
                const int64_t h_start = (h * H_in) / H_out;
                const int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;
                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_start = (w * W_in) / W_out;
                    const int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;
                    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                    int64_t max_off = -1;
                    for (int64_t ih = h_start; ih < h_end; ++ih) {
                        const scalar_t* row = in_base + ih * W_in;
                        for (int64_t iw = w_start; iw < w_end; ++iw) {
                            const scalar_t val = row[iw];
                            if ((val > max_val) || std::isnan(val)) { max_val = val; max_off = ih * W_in + iw; }
                        }
                    }
                    out_row[w] = max_val;
                    idx_row[w] = max_off;
                }
            }
        });
    });
    return std::make_tuple(out, indices);
}

Tensor adaptive_max_pool2d_with_indices_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                                     const std::vector<int64_t>& output_size, const Tensor& indices) {
    (void)output_size;
    if (input.dim() == 3) {
        // Unbatched (C,H,W): pool as a batch of one, matching the forward.
        return adaptive_max_pool2d_with_indices_backward_cpu(
                   grad_output.unsqueeze(0), input.unsqueeze(0), output_size,
                   indices.unsqueeze(0))
            .squeeze(0);
    }
    if (grad_output.dim() != 4 || input.dim() != 4)
        TP_THROW(RuntimeError, "adaptive_max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    Tensor grad_input = Tensor::zeros_like(input);
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices.contiguous();
    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool2d_with_indices_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* gop = go.data_ptr<scalar_t>();
        const int64_t* idxp = idx.data_ptr<int64_t>();
        const int64_t plane = input.size(2) * input.size(3);
        const int64_t out_plane = go.size(2) * go.size(3);
        const int64_t NC = go.size(0) * go.size(1);
        // Scatter via indices: parallel over (n, c) planes (race free).
        parallel_for(0, NC, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = gi + nc * plane;
                const scalar_t* go_base = gop + nc * out_plane;
                const int64_t* idx_base = idxp + nc * out_plane;
                for (int64_t i = 0; i < out_plane; ++i) {
                    const int64_t max_idx = idx_base[i];
                    if (max_idx < 0) continue;
                    gi_base[max_idx] += go_base[i];
                }
            }
        });
    });
    return grad_input;
}

Tensor avg_pool3d_backward_cpu(const Tensor& grad_output, const Tensor& input,
                               const std::vector<int64_t>& kernel_size,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               bool ceil_mode, bool count_include_pad,
                               std::optional<int64_t> divisor_override) {
    if (grad_output.dim() == 4 && input.dim() == 4) {
        return avg_pool3d_backward_cpu(grad_output.unsqueeze(0), input.unsqueeze(0),
                                       kernel_size, stride, padding, ceil_mode,
                                       count_include_pad,
                                       divisor_override).squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "avg_pool3d_backward: Expected 5D input and grad_output");
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t oD = grad_output.size(2), oH = grad_output.size(3), oW = grad_output.size(4);
    const int64_t kd = kernel_size[0], kh = kernel_size[1], kw = kernel_size[2];
    const int64_t sd = stride[0], sh = stride[1], sw = stride[2];
    const int64_t pd_ = padding[0], ph = padding[1], pw = padding[2];
    Tensor grad_input = Tensor::zeros({N, C, D, H, W}, input.dtype(), input.device());
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "avg_pool3d_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* go = grad_output.data_ptr<scalar_t>();
        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = gi + nc * D * H * W;
                const scalar_t* go_base = go + nc * oD * oH * oW;
                for (int64_t od = 0; od < oD; ++od)
                for (int64_t oh = 0; oh < oH; ++oh)
                for (int64_t ow = 0; ow < oW; ++ow) {
                    // window bounds in padded coordinates first (pool_size includes
                    // padding when count_include_pad), then clip to the input.
                    int64_t d0 = od * sd - pd_, h0 = oh * sh - ph, w0 = ow * sw - pw;
                    int64_t d1 = std::min(d0 + kd, D + pd_), h1 = std::min(h0 + kh, H + ph), w1 = std::min(w0 + kw, W + pw);
                    int64_t pool_size = (d1 - d0) * (h1 - h0) * (w1 - w0);
                    int64_t cd0 = std::max(d0, int64_t(0)), ch0 = std::max(h0, int64_t(0)), cw0 = std::max(w0, int64_t(0));
                    int64_t cd1 = std::min(d1, D), ch1 = std::min(h1, H), cw1 = std::min(w1, W);
                    int64_t div = divisor_override.has_value() ? *divisor_override
                                  : count_include_pad ? pool_size
                                  : (cd1 - cd0) * (ch1 - ch0) * (cw1 - cw0);
                    if (div <= 0) continue;
                    const scalar_t g = go_base[(od * oH + oh) * oW + ow] / static_cast<scalar_t>(div);
                    for (int64_t d = cd0; d < cd1; ++d)
                    for (int64_t h = ch0; h < ch1; ++h) {
                        scalar_t* row = gi_base + (d * H + h) * W;
                        for (int64_t w = cw0; w < cw1; ++w) row[w] += g;
                    }
                }
            }
        });
    });
    return grad_input;
}

// start = floor(i * in / out), end = ceil((i+1) * in / out).
Tensor adaptive_avg_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 4)
        return adaptive_avg_pool3d_cpu(input.unsqueeze(0), output_size).squeeze(0);
    if (input.dim() != 5) TP_THROW(RuntimeError, "adaptive_avg_pool3d: Expected 5D input");
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    Tensor out = Tensor::empty({N, C, oD, oH, oW}, input.dtype(), input.device());
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "adaptive_avg_pool3d", [&]() {
        scalar_t* op = out.data_ptr<scalar_t>();
        const scalar_t* ip = input.data_ptr<scalar_t>();
        parallel_for(0, N * C * oD * oH, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t h = idx % oH;
                const int64_t d = (idx / oH) % oD;
                const int64_t nc = idx / (oH * oD);
                const scalar_t* in_base = ip + nc * D * H * W;
                scalar_t* out_row = op + idx * oW;

                const int64_t ds = d * D / oD;
                const int64_t de = static_cast<int64_t>(std::ceil((d + 1) * D / static_cast<double>(oD)));
                const int64_t hs = h * H / oH;
                const int64_t he = static_cast<int64_t>(std::ceil((h + 1) * H / static_cast<double>(oH)));

                for (int64_t w = 0; w < oW; ++w) {
                    const int64_t ws = w * W / oW;
                    const int64_t we = static_cast<int64_t>(std::ceil((w + 1) * W / static_cast<double>(oW)));
                    scalar_t sum = scalar_t(0);
                    for (int64_t z = ds; z < de; ++z)
                    for (int64_t y = hs; y < he; ++y) {
                        const scalar_t* row = in_base + (z * H + y) * W;
                        for (int64_t x = ws; x < we; ++x) sum += row[x];
                    }
                    out_row[w] = sum / static_cast<scalar_t>((de - ds) * (he - hs) * (we - ws));
                }
            }
        });
    });
    return out;
}

Tensor adaptive_avg_pool3d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() == 4 && input.dim() == 4)
        return adaptive_avg_pool3d_backward_cpu(grad_output.unsqueeze(0),
                                                input.unsqueeze(0)).squeeze(0);
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "adaptive_avg_pool3d_backward: Expected 5D input and grad_output");
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t oD = grad_output.size(2), oH = grad_output.size(3), oW = grad_output.size(4);
    Tensor grad_input = Tensor::zeros({N, C, D, H, W}, input.dtype(), input.device());
    TP_DISPATCH_FLOATING_TYPES_AND_LONG(input.dtype(), "adaptive_avg_pool3d_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* go = grad_output.data_ptr<scalar_t>();
        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = gi + nc * D * H * W;
                const scalar_t* go_base = go + nc * oD * oH * oW;
                for (int64_t d = 0; d < oD; ++d) {
                    const int64_t ds = d * D / oD;
                    const int64_t de = static_cast<int64_t>(std::ceil((d + 1) * D / static_cast<double>(oD)));
                    for (int64_t h = 0; h < oH; ++h) {
                        const int64_t hs = h * H / oH;
                        const int64_t he = static_cast<int64_t>(std::ceil((h + 1) * H / static_cast<double>(oH)));
                        for (int64_t w = 0; w < oW; ++w) {
                            const int64_t ws = w * W / oW;
                            const int64_t we = static_cast<int64_t>(std::ceil((w + 1) * W / static_cast<double>(oW)));
                            const scalar_t g =
                                go_base[(d * oH + h) * oW + w] /
                                static_cast<scalar_t>((de - ds) * (he - hs) * (we - ws));
                            for (int64_t z = ds; z < de; ++z)
                            for (int64_t y = hs; y < he; ++y) {
                                scalar_t* row = gi_base + (z * H + y) * W;
                                for (int64_t x = ws; x < we; ++x) row[x] += g;
                            }
                        }
                    }
                }
            }
        });
    });
    return grad_input;
}

// ---------------------------------------------------------------------------
// Pool.h pooling_output_shape_pad_lr; indices are linear offsets into the
// per-(n, c) input plane (H*W for 2d, D*H*W for 3d); NaN wins the window and
// ---------------------------------------------------------------------------

static inline int64_t div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

static int64_t pooling_output_shape_aten(int64_t in, int64_t k, int64_t pad,
                                         int64_t stride, int64_t dilation,
                                         bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    int64_t out = div_rtn(in + 2 * pad - dilation * (k - 1) - 1 +
                              (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

static std::vector<int64_t> expand_pool_param(const std::vector<int64_t>& list,
                                              const char* name, int64_t n,
                                              int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if ((int64_t)list.size() != n)
        TP_THROW(ValueError, std::string(name) + ": expected " + std::to_string(n) + " values");
    return list;
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices_cpu(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        auto r = max_pool2d_with_indices_cpu(input.unsqueeze(0), kernel_size,
                                             stride, padding, dilation, ceil_mode);
        return std::make_tuple(std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0));
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d_with_indices: Expected 4D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t H_in = input_c.size(2), W_in = input_c.size(3);

    const auto ks = expand_pool_param(kernel_size, "max_pool2d_with_indices kernel_size", 2, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride, "max_pool2d_with_indices stride", 2, ks[0]);
    const auto pd = expand_pool_param(padding, "max_pool2d_with_indices padding", 2, 0);
    const auto dl = expand_pool_param(dilation, "max_pool2d_with_indices dilation", 2, 1);
    const int64_t kH = ks[0], kW = ks[1], sH = st[0], sW = st[1];
    const int64_t pH = pd[0], pW = pd[1], dH = dl[0], dW = dl[1];

    const int64_t H_out = pooling_output_shape_aten(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pooling_output_shape_aten(W_in, kW, pW, sW, dW, ceil_mode);
    if (H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "max_pool2d_with_indices: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({N, C, H_out, W_out}, DType::Int64, input.device());

    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool2d_with_indices", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        int64_t* idx_ptr = indices.data_ptr<int64_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* plane = in_ptr + nc * H_in * W_in;
                scalar_t* out_base = out_ptr + nc * H_out * W_out;
                int64_t* idx_base = idx_ptr + nc * H_out * W_out;
                for (int64_t h = 0; h < H_out; ++h) {
                    const int64_t h_start = h * sH - pH;
                    int64_t kh0 = 0, kh1 = kH;
                    if (dH == 1) {
                        if (h_start < 0) kh0 = -h_start;
                        if (h_start + kH > H_in) kh1 = H_in - h_start;
                    } else {
                        while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                        while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H_in) --kh1;
                    }
                    for (int64_t w = 0; w < W_out; ++w) {
                        const int64_t w_start = w * sW - pW;
                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        int64_t max_idx = -1;
                        if (dW == 1) {
                            const int64_t kw0 = w_start < 0 ? -w_start : 0;
                            const int64_t kw1 = w_start + kW > W_in ? W_in - w_start : kW;
                            for (int64_t kh = kh0; kh < kh1; ++kh) {
                                const int64_t row_off = (h_start + kh) * W_in + w_start;
                                const scalar_t* row = plane + row_off;
                                for (int64_t kw = kw0; kw < kw1; ++kw) {
                                    const scalar_t val = row[kw];
                                    if ((val > max_val) || std::isnan(val)) {
                                        max_val = val;
                                        max_idx = row_off + kw;
                                    }
                                }
                            }
                        } else {
                            for (int64_t kh = kh0; kh < kh1; ++kh) {
                                const int64_t row_off = (h_start + kh * dH) * W_in;
                                const scalar_t* row = plane + row_off;
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    const int64_t wi = w_start + kw * dW;
                                    if (wi >= 0 && wi < W_in) {
                                        const scalar_t val = row[wi];
                                        if ((val > max_val) || std::isnan(val)) {
                                            max_val = val;
                                            max_idx = row_off + wi;
                                        }
                                    }
                                }
                            }
                        }
                        const int64_t o = h * W_out + w;
                        out_base[o] = max_val;
                        idx_base[o] = max_idx;
                    }
                }
            }
        });
    });
    return std::make_tuple(out, indices);
}

Tensor max_pool2d_with_indices_backward_cpu(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size; (void)stride; (void)padding; (void)dilation; (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined())
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: indices is required");
    const Tensor& indices = *indices_opt;
    if (input.dim() == 3) {
        // Unbatched (C,H,W): pool as a batch of one, matching the forward.
        return max_pool2d_with_indices_backward_cpu(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode, indices.unsqueeze(0))
            .squeeze(0);
    }
    if (grad_output.dim() != 4 || input.dim() != 4)
        TP_THROW(RuntimeError, "max_pool2d_with_indices_backward: Expected 4D input and grad_output");
    Tensor grad_input = Tensor::zeros_like(input);
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices.contiguous();
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool2d_with_indices_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* gop = go.data_ptr<scalar_t>();
        const int64_t* idxp = idx.data_ptr<int64_t>();
        const int64_t plane = input.size(2) * input.size(3);
        const int64_t out_plane = go.size(2) * go.size(3);
        const int64_t NC = go.size(0) * go.size(1);
        // Scatter via indices: parallel over (n, c) planes (race free).
        parallel_for(0, NC, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = gi + nc * plane;
                const scalar_t* go_base = gop + nc * out_plane;
                const int64_t* idx_base = idxp + nc * out_plane;
                for (int64_t i = 0; i < out_plane; ++i) {
                    const int64_t max_idx = idx_base[i];
                    if (max_idx < 0) continue;
                    gi_base[max_idx] += go_base[i];
                }
            }
        });
    });
    return grad_input;
}

Tensor max_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size,
                      const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                      const std::vector<int64_t>& dilation, bool ceil_mode);

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cpu(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode);

Tensor max_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size,
                      const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                      const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        return max_pool3d_cpu(input.unsqueeze(0), kernel_size, stride, padding,
                              dilation, ceil_mode).squeeze(0);
    }
    return std::get<0>(max_pool3d_with_indices_cpu(input, kernel_size, stride,
                                                   padding, dilation, ceil_mode));
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cpu(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        auto r = max_pool3d_with_indices_cpu(input.unsqueeze(0), kernel_size,
                                             stride, padding, dilation, ceil_mode);
        return std::make_tuple(std::get<0>(r).squeeze(0), std::get<1>(r).squeeze(0));
    }
    if (input.dim() != 5) TP_THROW(RuntimeError, "max_pool3d_with_indices: Expected 5D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D_in = input_c.size(2), H_in = input_c.size(3), W_in = input_c.size(4);

    const auto ks = expand_pool_param(kernel_size, "max_pool3d kernel_size", 3, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride, "max_pool3d stride", 3, ks[0]);
    const auto pd = expand_pool_param(padding, "max_pool3d padding", 3, 0);
    const auto dl = expand_pool_param(dilation, "max_pool3d dilation", 3, 1);
    const int64_t kD = ks[0], kH = ks[1], kW = ks[2];
    const int64_t sD = st[0], sH = st[1], sW = st[2];
    const int64_t pD = pd[0], pH = pd[1], pW = pd[2];
    const int64_t dD = dl[0], dH = dl[1], dW = dl[2];

    const int64_t D_out = pooling_output_shape_aten(D_in, kD, pD, sD, dD, ceil_mode);
    const int64_t H_out = pooling_output_shape_aten(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pooling_output_shape_aten(W_in, kW, pW, sW, dW, ceil_mode);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0)
        TP_THROW(RuntimeError, "max_pool3d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices = Tensor::empty({N, C, D_out, H_out, W_out}, DType::Int64, input.device());

    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool3d_with_indices", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        int64_t* idx_ptr = indices.data_ptr<int64_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();
        const int64_t in_plane = D_in * H_in * W_in;
        const int64_t out_plane = D_out * H_out * W_out;
        parallel_for(0, N * C * D_out, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx0 = begin; idx0 < end; ++idx0) {
                const int64_t d = idx0 % D_out;
                const int64_t nc = idx0 / D_out;
                const scalar_t* vol = in_ptr + nc * in_plane;
                scalar_t* out_base = out_ptr + nc * out_plane;
                int64_t* idx_base = idx_ptr + nc * out_plane;
                {
                    const int64_t d_start = d * sD - pD;
                    int64_t kd0 = 0, kd1 = kD;
                    if (dD == 1) {
                        if (d_start < 0) kd0 = -d_start;
                        if (d_start + kD > D_in) kd1 = D_in - d_start;
                    } else {
                        while (kd0 < kD && d_start + kd0 * dD < 0) ++kd0;
                        while (kd1 > kd0 && d_start + (kd1 - 1) * dD >= D_in) --kd1;
                    }
                    for (int64_t h = 0; h < H_out; ++h) {
                        const int64_t h_start = h * sH - pH;
                        int64_t kh0 = 0, kh1 = kH;
                        if (dH == 1) {
                            if (h_start < 0) kh0 = -h_start;
                            if (h_start + kH > H_in) kh1 = H_in - h_start;
                        } else {
                            while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                            while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H_in) --kh1;
                        }
                        for (int64_t w = 0; w < W_out; ++w) {
                            const int64_t w_start = w * sW - pW;
                            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                            int64_t max_idx = -1;
                            const bool unit_w = (dW == 1);
                            const int64_t kw0 = unit_w && w_start < 0 ? -w_start : 0;
                            const int64_t kw1 = unit_w && w_start + kW > W_in ? W_in - w_start : kW;
                            for (int64_t kd = kd0; kd < kd1; ++kd) {
                                const int64_t di = d_start + kd * dD;
                                const int64_t slice_off = di * H_in * W_in;
                                const scalar_t* slice = vol + slice_off;
                                for (int64_t kh = kh0; kh < kh1; ++kh) {
                                    const int64_t row_off = (h_start + kh * dH) * W_in;
                                    const scalar_t* row = slice + row_off;
                                    if (unit_w) {
                                        for (int64_t kw = kw0; kw < kw1; ++kw) {
                                            const scalar_t val = row[w_start + kw];
                                            if ((val > max_val) || std::isnan(val)) {
                                                max_val = val;
                                                max_idx = slice_off + row_off + w_start + kw;
                                            }
                                        }
                                    } else {
                                        for (int64_t kw = 0; kw < kW; ++kw) {
                                            const int64_t wi = w_start + kw * dW;
                                            if (wi >= 0 && wi < W_in) {
                                                const scalar_t val = row[wi];
                                                if ((val > max_val) || std::isnan(val)) {
                                                    max_val = val;
                                                    max_idx = slice_off + row_off + wi;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            const int64_t o = (d * H_out + h) * W_out + w;
                            out_base[o] = max_val;
                            idx_base[o] = max_idx;
                        }
                    }
                }
            }
        });
    });
    return std::make_tuple(out, indices);
}

Tensor max_pool3d_backward_cpu(const Tensor& grad_output, const Tensor& input,
                               const std::vector<int64_t>& kernel_size,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, bool ceil_mode) {
    // scatter the gradient onto each window's argmax (first-max wins ties).
    if (grad_output.dim() == 4 && input.dim() == 4) {
        return max_pool3d_backward_cpu(grad_output.unsqueeze(0), input.unsqueeze(0),
                                       kernel_size, stride, padding, dilation,
                                       ceil_mode).squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "max_pool3d_backward: Expected 5D input and grad_output");
    const Tensor input_c = input.contiguous();
    const Tensor go = grad_output.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D_in = input_c.size(2), H_in = input_c.size(3), W_in = input_c.size(4);
    const int64_t D_out = go.size(2), H_out = go.size(3), W_out = go.size(4);

    const auto ks = expand_pool_param(kernel_size, "max_pool3d_backward kernel_size", 3, 1);
    const auto st = expand_pool_param(stride.empty() ? ks : stride, "max_pool3d_backward stride", 3, ks[0]);
    const auto pd = expand_pool_param(padding, "max_pool3d_backward padding", 3, 0);
    const auto dl = expand_pool_param(dilation, "max_pool3d_backward dilation", 3, 1);
    const int64_t kD = ks[0], kH = ks[1], kW = ks[2];
    const int64_t sD = st[0], sH = st[1], sW = st[2];
    const int64_t pD = pd[0], pH = pd[1], pW = pd[2];
    const int64_t dD = dl[0], dH = dl[1], dW = dl[2];

    Tensor grad_input = Tensor::zeros_like(input);
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool3d_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* gop = go.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();
        const int64_t in_plane = D_in * H_in * W_in;
        const int64_t out_plane = D_out * H_out * W_out;
        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* vol = in_ptr + nc * in_plane;
                scalar_t* gvol = gi + nc * in_plane;
                const scalar_t* go_base = gop + nc * out_plane;
                for (int64_t d = 0; d < D_out; ++d) {
                    const int64_t d_start = d * sD - pD;
                    int64_t kd0 = 0, kd1 = kD;
                    if (dD == 1) {
                        if (d_start < 0) kd0 = -d_start;
                        if (d_start + kD > D_in) kd1 = D_in - d_start;
                    } else {
                        while (kd0 < kD && d_start + kd0 * dD < 0) ++kd0;
                        while (kd1 > kd0 && d_start + (kd1 - 1) * dD >= D_in) --kd1;
                    }
                    for (int64_t h = 0; h < H_out; ++h) {
                        const int64_t h_start = h * sH - pH;
                        int64_t kh0 = 0, kh1 = kH;
                        if (dH == 1) {
                            if (h_start < 0) kh0 = -h_start;
                            if (h_start + kH > H_in) kh1 = H_in - h_start;
                        } else {
                            while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                            while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H_in) --kh1;
                        }
                        for (int64_t w = 0; w < W_out; ++w) {
                            const int64_t w_start = w * sW - pW;
                            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                            int64_t max_idx = -1;
                            const bool unit_w = (dW == 1);
                            const int64_t kw0 = unit_w && w_start < 0 ? -w_start : 0;
                            const int64_t kw1 = unit_w && w_start + kW > W_in ? W_in - w_start : kW;
                            for (int64_t kd = kd0; kd < kd1; ++kd) {
                                const int64_t di = d_start + kd * dD;
                                const int64_t slice_off = di * H_in * W_in;
                                const scalar_t* slice = vol + slice_off;
                                for (int64_t kh = kh0; kh < kh1; ++kh) {
                                    const int64_t row_off = (h_start + kh * dH) * W_in;
                                    const scalar_t* row = slice + row_off;
                                    if (unit_w) {
                                        for (int64_t kw = kw0; kw < kw1; ++kw) {
                                            const scalar_t val = row[w_start + kw];
                                            if ((val > max_val) || std::isnan(val)) {
                                                max_val = val;
                                                max_idx = slice_off + row_off + w_start + kw;
                                            }
                                        }
                                    } else {
                                        for (int64_t kw = 0; kw < kW; ++kw) {
                                            const int64_t wi = w_start + kw * dW;
                                            if (wi >= 0 && wi < W_in) {
                                                const scalar_t val = row[wi];
                                                if ((val > max_val) || std::isnan(val)) {
                                                    max_val = val;
                                                    max_idx = slice_off + row_off + wi;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (max_idx != -1)
                                gvol[max_idx] += go_base[(d * H_out + h) * W_out + w];
                        }
                    }
                }
            }
        });
    });
    return grad_input;
}

Tensor max_pool3d_with_indices_backward_cpu(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size; (void)stride; (void)padding; (void)dilation; (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined())
        TP_THROW(RuntimeError, "max_pool3d_with_indices_backward: indices is required");
    const Tensor& indices = *indices_opt;
    if (input.dim() == 4) {
        // Unbatched (C,D,H,W): pool as a batch of one, matching the forward.
        return max_pool3d_with_indices_backward_cpu(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode, indices.unsqueeze(0))
            .squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "max_pool3d_with_indices_backward: Expected 5D input and grad_output");
    Tensor grad_input = Tensor::zeros_like(input);
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices.contiguous();
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool3d_with_indices_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* gop = go.data_ptr<scalar_t>();
        const int64_t* idxp = idx.data_ptr<int64_t>();
        const int64_t in_plane = input.size(2) * input.size(3) * input.size(4);
        const int64_t out_spatial = go.size(2) * go.size(3) * go.size(4);
        const int64_t NC = go.size(0) * go.size(1);
        // Scatter via indices: parallel over (n, c) planes (race free).
        parallel_for(0, NC, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                scalar_t* gi_base = gi + nc * in_plane;
                const scalar_t* go_base = gop + nc * out_spatial;
                const int64_t* idx_base = idxp + nc * out_spatial;
                for (int64_t i = 0; i < out_spatial; ++i) {
                    const int64_t max_idx = idx_base[i];
                    if (max_idx < 0) continue;
                    gi_base[max_idx] += go_base[i];
                }
            }
        });
    });
    return grad_input;
}

// average pool; indices are linear offsets into the (D, H, W) volume.
Tensor adaptive_max_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 4)
        return adaptive_max_pool3d_cpu(input.unsqueeze(0), output_size).squeeze(0);
    if (input.dim() != 5) TP_THROW(RuntimeError, "adaptive_max_pool3d: Expected 5D input");
    const Tensor input_c = input.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D = input_c.size(2), H = input_c.size(3), W = input_c.size(4);
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    if (oD <= 0 || oH <= 0 || oW <= 0)
        TP_THROW(RuntimeError, "adaptive_max_pool3d: Invalid output size");
    Tensor out = Tensor::empty({N, C, oD, oH, oW}, input.dtype(), input.device());
    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool3d", [&]() {
        scalar_t* op = out.data_ptr<scalar_t>();
        const scalar_t* ip = input_c.data_ptr<scalar_t>();
        const int64_t in_plane = D * H * W;
        const int64_t out_plane = oD * oH * oW;
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* vol = ip + nc * in_plane;
                scalar_t* out_base = op + nc * out_plane;
                for (int64_t d = 0; d < oD; ++d) {
                    const int64_t ds = d * D / oD;
                    const int64_t de = 1 + (((d + 1) * D) - 1) / oD;
                    for (int64_t h = 0; h < oH; ++h) {
                        const int64_t hs = h * H / oH;
                        const int64_t he = 1 + (((h + 1) * H) - 1) / oH;
                        for (int64_t w = 0; w < oW; ++w) {
                            const int64_t ws = w * W / oW;
                            const int64_t we = 1 + (((w + 1) * W) - 1) / oW;
                            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                            for (int64_t z = ds; z < de; ++z)
                            for (int64_t y = hs; y < he; ++y) {
                                const scalar_t* row = vol + (z * H + y) * W;
                                for (int64_t x = ws; x < we; ++x) {
                                    const scalar_t val = row[x];
                                    if ((val > max_val) || std::isnan(val)) max_val = val;
                                }
                            }
                            out_base[(d * oH + h) * oW + w] = max_val;
                        }
                    }
                }
            }
        });
    });
    return out;
}

Tensor adaptive_max_pool3d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() == 4 && input.dim() == 4)
        return adaptive_max_pool3d_backward_cpu(grad_output.unsqueeze(0),
                                                input.unsqueeze(0)).squeeze(0);
    if (grad_output.dim() != 5 || input.dim() != 5)
        TP_THROW(RuntimeError, "adaptive_max_pool3d_backward: Expected 5D input and grad_output");
    const Tensor input_c = input.contiguous();
    const Tensor go = grad_output.contiguous();
    const int64_t N = input_c.size(0), C = input_c.size(1);
    const int64_t D = input_c.size(2), H = input_c.size(3), W = input_c.size(4);
    const int64_t oD = go.size(2), oH = go.size(3), oW = go.size(4);
    Tensor grad_input = Tensor::zeros({N, C, D, H, W}, input.dtype(), input.device());
    TP_DISPATCH_ALL_TYPES(input.dtype(), "adaptive_max_pool3d_backward", [&]() {
        scalar_t* gi = grad_input.data_ptr<scalar_t>();
        const scalar_t* gop = go.data_ptr<scalar_t>();
        const scalar_t* ip = input_c.data_ptr<scalar_t>();
        const int64_t in_plane = D * H * W;
        const int64_t out_plane = oD * oH * oW;
        // Scatter: parallel over (n, c) planes (race free).
        parallel_for(0, N * C, 1, [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const scalar_t* vol = ip + nc * in_plane;
                scalar_t* gvol = gi + nc * in_plane;
                const scalar_t* go_base = gop + nc * out_plane;
                for (int64_t d = 0; d < oD; ++d) {
                    const int64_t ds = d * D / oD;
                    const int64_t de = 1 + (((d + 1) * D) - 1) / oD;
                    for (int64_t h = 0; h < oH; ++h) {
                        const int64_t hs = h * H / oH;
                        const int64_t he = 1 + (((h + 1) * H) - 1) / oH;
                        for (int64_t w = 0; w < oW; ++w) {
                            const int64_t ws = w * W / oW;
                            const int64_t we = 1 + (((w + 1) * W) - 1) / oW;
                            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                            int64_t max_idx = -1;
                            for (int64_t z = ds; z < de; ++z)
                            for (int64_t y = hs; y < he; ++y) {
                                const int64_t row_off = (z * H + y) * W;
                                const scalar_t* row = vol + row_off;
                                for (int64_t x = ws; x < we; ++x) {
                                    const scalar_t val = row[x];
                                    if ((val > max_val) || std::isnan(val)) {
                                        max_val = val;
                                        max_idx = row_off + x;
                                    }
                                }
                            }
                            if (max_idx != -1)
                                gvol[max_idx] += go_base[(d * oH + h) * oW + w];
                        }
                    }
                }
            }
        });
    });
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CPU, PoolingKernels) {
    m.impl("avg_pool2d", avg_pool2d_cpu);
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cpu);
    m.impl("max_pool2d_backward", max_pool2d_backward_cpu);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_cpu);
    m.impl("avg_pool3d", avg_pool3d_cpu);
    m.impl("avg_pool3d_backward", avg_pool3d_backward_cpu);
    m.impl("adaptive_avg_pool3d", adaptive_avg_pool3d_cpu);
    m.impl("adaptive_avg_pool3d_backward", adaptive_avg_pool3d_backward_cpu);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cpu);
    m.impl("adaptive_max_pool2d_backward", adaptive_max_pool2d_backward_cpu);
    m.impl("adaptive_max_pool2d_with_indices", adaptive_max_pool2d_with_indices_cpu);
    m.impl("adaptive_max_pool2d_with_indices_backward", adaptive_max_pool2d_with_indices_backward_cpu);
    m.impl("max_pool2d_with_indices", max_pool2d_with_indices_cpu);
    m.impl("max_pool2d_with_indices_backward", max_pool2d_with_indices_backward_cpu);
    m.impl("max_pool3d_backward", max_pool3d_backward_cpu);
    m.impl("max_pool3d_with_indices", max_pool3d_with_indices_cpu);
    m.impl("max_pool3d_with_indices_backward", max_pool3d_with_indices_backward_cpu);
    m.impl("adaptive_max_pool3d", adaptive_max_pool3d_cpu);
    m.impl("adaptive_max_pool3d_backward", adaptive_max_pool3d_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay

// CompositeImplicitAutograd over their *_with_indices variants: routing the
// forward through with_indices lets autograd save the indices and scatter in
// max_poolNd backward is the indices scatter, not a re-scan).
namespace tensorplay {
namespace cpu {

Tensor max_pool2d_composite(const Tensor& input, const std::vector<int64_t>& kernel_size,
                            const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, bool ceil_mode) {
    return std::get<0>(tpx::ops::max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode));
}

Tensor max_pool3d_composite(const Tensor& input, const std::vector<int64_t>& kernel_size,
                            const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, bool ceil_mode) {
    return std::get<0>(tpx::ops::max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode));
}

Tensor adaptive_max_pool2d_composite(const Tensor& input, const std::vector<int64_t>& output_size) {
    return std::get<0>(tpx::ops::adaptive_max_pool2d_with_indices(input, output_size));
}

TENSORPLAY_LIBRARY_IMPL(Composite, PoolingComposite) {
    m.impl("max_pool2d", max_pool2d_composite);
    m.impl("max_pool3d", max_pool3d_composite);
    m.impl("adaptive_max_pool2d", adaptive_max_pool2d_composite);
}

} // namespace cpu
} // namespace tensorplay
