#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

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

// Upstream ATen/Dispatch.h AT_DISPATCH_FLOATING_TYPES parity: immediately
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

// AT_DISPATCH_ALL_TYPES parity: Byte, Char, Short, Int, Long, Float, Double.
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

// Upstream aten/src/ATen/native/AveragePool3d.cpp parity.
Tensor avg_pool3d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size,
                      const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                      bool ceil_mode, bool count_include_pad,
                      std::optional<int64_t> divisor_override) {
    if (input.dim() == 4) {
        // torch accepts unbatched (C,D,H,W): pool as a batch of one.
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
        for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
        for (int64_t od = 0; od < oD; ++od)
        for (int64_t oh = 0; oh < oH; ++oh)
        for (int64_t ow = 0; ow < oW; ++ow) {
            const int64_t d0 = od * sd - pd_, h0 = oh * sh - ph, w0 = ow * sw - pw;
            const int64_t d1 = std::min(d0 + kd, D), h1 = std::min(h0 + kh, H), w1 = std::min(w0 + kw, W);
            scalar_t sum = scalar_t(0);
            int64_t cnt = 0;
            for (int64_t d = std::max(d0, int64_t(0)); d < d1; ++d)
            for (int64_t h = std::max(h0, int64_t(0)); h < h1; ++h)
            for (int64_t w = std::max(w0, int64_t(0)); w < w1; ++w) {
                sum += in_ptr[((n * C + c) * D + d) * H * W + h * W + w];
                ++cnt;
            }
            int64_t div = divisor_override.has_value()
                              ? *divisor_override
                              : (count_include_pad ? kd * kh * kw : cnt);
            out_ptr[((n * C + c) * oD + od) * oH * oW + oh * oW + ow] =
                div > 0 ? sum / static_cast<scalar_t>(div) : scalar_t(0);
        }
    });
    return out;
}

Tensor max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 3) {
        // torch accepts unbatched (C,H,W): pool as a batch of one.
        return max_pool2d_cpu(input.unsqueeze(0), kernel_size, stride, padding,
                              dilation, ceil_mode).squeeze(0);
    }
    if (input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d: Expected 4D input");
    // The kernel indexes raw NCHW pointers; normalize views (no-op when
    // already contiguous) so non-contiguous inputs match torch's results.
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
    // Usually PyTorch clamps the window end.
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    TP_DISPATCH_ALL_TYPES(input.dtype(), "max_pool2d", [&]() {
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        const scalar_t* in_ptr = input_c.data_ptr<scalar_t>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {

                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;

                        // Valid window range
                        // We iterate kernel
                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh * dH;
                                int64_t w_in_idx = w_start + kw * dW;

                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    scalar_t val = in_ptr[idx];
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = max_val;
                    }
                }
            }
        }
    });
    
    return out;
}

Tensor avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (input.dim() == 3) {
        // torch accepts unbatched (C,H,W): pool as a batch of one.
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
        // ATen alignment: last window must start strictly inside (input + padding)
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
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        int64_t h_end = std::min(h_start + kH, H_in + pH);
                        int64_t w_end = std::min(w_start + kW, W_in + pW);
                        
                        int64_t pool_size = (h_end - h_start) * (w_end - w_start); // This calculation is slightly wrong if we consider padding logic
                        // Let's iterate explicitly
                        
                        scalar_t sum = 0.0f;
                        int64_t count = 0;
                        
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh;
                                int64_t w_in_idx = w_start + kw;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    sum += in_ptr[idx];
                                    count++;
                                }
                            }
                        }
                        
                        scalar_t divisor;
                        if (divisor_override.has_value()) {
                            divisor = (scalar_t)divisor_override.value();
                        } else if (count_include_pad) {
                            // ATen alignment: window area clipped to (input + padding)
                            int64_t clip_h = std::min(h_start + kH, H_in + pH) - h_start;
                            int64_t clip_w = std::min(w_start + kW, W_in + pW) - w_start;
                            divisor = (scalar_t)(clip_h * clip_w);
                        } else {
                             divisor = (scalar_t)count;
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = sum / divisor;
                    }
                }
            }
        }
    });
    
    return out;
}

Tensor adaptive_avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        // torch accepts unbatched (C,H,W): pool as a batch of one.
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
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    // Match PyTorch's adaptive pooling bins: floor(start), ceil(end).
                    int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;
                    
                    int64_t kH = h_end - h_start;
                    
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;
                        
                        int64_t kW = w_end - w_start;
                        
                        scalar_t sum = 0.0f;
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                sum += in_ptr[idx];
                            }
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = sum / (kH * kW);
                    }
                }
            }
        }
    });
    
    return out;
}

Tensor adaptive_max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() == 3) {
        // torch accepts unbatched (C,H,W): pool as a batch of one.
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

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    // AdaptivePooling.h start_index/end_index: floor start,
                    // ceil end -- the same bins as adaptive avg pooling.
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;

                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                scalar_t val = in_ptr[idx];
                                if ((val > max_val) || std::isnan(val)) max_val = val;
                            }
                        }

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = max_val;
                    }
                }
            }
        }
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

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        
                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        int64_t max_idx = -1;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh * dH;
                                int64_t w_in_idx = w_start + kw * dW;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    scalar_t val = in_ptr[idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = idx;
                                    }
                                }
                            }
                        }
                        
                        if (max_idx != -1) {
                            int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                            grad_in_ptr[max_idx] += grad_out_ptr[out_idx];
                        }
                    }
                }
            }
        }
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

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        int64_t h_end = std::min(h_start + kH, H_in + pH);
                        int64_t w_end = std::min(w_start + kW, W_in + pW);
                        
                        scalar_t divisor;
                         // Recalculate divisor logic from forward
                        if (divisor_override.has_value()) {
                            divisor = (scalar_t)divisor_override.value();
                        } else if (count_include_pad) {
                            // ATen alignment: window area clipped to (input + padding)
                            int64_t clip_h = std::min(h_start + kH, H_in + pH) - h_start;
                            int64_t clip_w = std::min(w_start + kW, W_in + pW) - w_start;
                            divisor = (scalar_t)(clip_h * clip_w);
                        } else {
                            // Calculate count excluding pad
                            int64_t count = 0;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t h_in_idx = h_start + kh;
                                    int64_t w_in_idx = w_start + kw;
                                    if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                        count++;
                                    }
                                }
                            }
                            divisor = (scalar_t)count;
                        }

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        scalar_t grad_val = grad_out_ptr[out_idx] / divisor;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh;
                                int64_t w_in_idx = w_start + kw;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    grad_in_ptr[idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
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

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out;
                    int64_t kH = h_end - h_start;

                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in + W_out - 1) / W_out;
                        int64_t kW = w_end - w_start;

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        scalar_t grad_val = grad_out_ptr[out_idx] / (kH * kW);

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                grad_in_ptr[idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }
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

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    // AdaptivePooling.h start_index/end_index (floor/ceil).
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = 1 + (((h + 1) * H_in) - 1) / H_out;

                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = 1 + (((w + 1) * W_in) - 1) / W_out;

                        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                        int64_t max_idx = -1;

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                scalar_t val = in_ptr[idx];
                                if ((val > max_val) || std::isnan(val)) {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        if (max_idx != -1) {
                            int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                            grad_in_ptr[max_idx] += grad_out_ptr[out_idx];
                        }
                    }
                }
            }
        }
    });
    return grad_input;
}

// Upstream aten/src/ATen/native/AveragePool3d.cpp avg_pool3d_backward_out_frame.
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
        for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
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
            const scalar_t g = go[((n * C + c) * oD + od) * oH * oW + oh * oW + ow] / static_cast<scalar_t>(div);
            for (int64_t d = cd0; d < cd1; ++d)
            for (int64_t h = ch0; h < ch1; ++h)
            for (int64_t w = cw0; w < cw1; ++w)
                gi[((n * C + c) * D + d) * H * W + h * W + w] += g;
        }
    });
    return grad_input;
}

// Upstream ATen/native/AdaptiveAveragePooling3d.cpp: window bounds are
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
        for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
        for (int64_t d = 0; d < oD; ++d) {
            const int64_t ds = d * D / oD, de = (d + 1) * D / oD + ((d + 1) * D % oD > 0 ? 0 : 0);
            const int64_t de_ = static_cast<int64_t>(std::ceil((d + 1) * D / static_cast<double>(oD)));
            (void)de;
            for (int64_t h = 0; h < oH; ++h) {
                const int64_t hs = h * H / oH;
                const int64_t he = static_cast<int64_t>(std::ceil((h + 1) * H / static_cast<double>(oH)));
                for (int64_t w = 0; w < oW; ++w) {
                    const int64_t ws = w * W / oW;
                    const int64_t we = static_cast<int64_t>(std::ceil((w + 1) * W / static_cast<double>(oW)));
                    scalar_t sum = scalar_t(0);
                    for (int64_t z = ds; z < de_; ++z)
                    for (int64_t y = hs; y < he; ++y)
                    for (int64_t x = ws; x < we; ++x)
                        sum += ip[((n * C + c) * D + z) * H * W + y * W + x];
                    op[((n * C + c) * oD + d) * oH * oW + h * oW + w] =
                        sum / static_cast<scalar_t>((de_ - ds) * (he - hs) * (we - ws));
                }
            }
        }
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
        for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
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
                        go[((n * C + c) * oD + d) * oH * oW + h * oW + w] /
                        static_cast<scalar_t>((de - ds) * (he - hs) * (we - ws));
                    for (int64_t z = ds; z < de; ++z)
                    for (int64_t y = hs; y < he; ++y)
                    for (int64_t x = ws; x < we; ++x)
                        gi[((n * C + c) * D + z) * H * W + y * W + x] += g;
                }
            }
        }
    });
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CPU, PoolingKernels) {
    m.impl("max_pool2d", max_pool2d_cpu);
    m.impl("avg_pool2d", avg_pool2d_cpu);
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cpu);
    m.impl("adaptive_max_pool2d", adaptive_max_pool2d_cpu);
    m.impl("max_pool2d_backward", max_pool2d_backward_cpu);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_cpu);
    m.impl("avg_pool3d", avg_pool3d_cpu);
    m.impl("avg_pool3d_backward", avg_pool3d_backward_cpu);
    m.impl("adaptive_avg_pool3d", adaptive_avg_pool3d_cpu);
    m.impl("adaptive_avg_pool3d_backward", adaptive_avg_pool3d_backward_cpu);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cpu);
    m.impl("adaptive_max_pool2d_backward", adaptive_max_pool2d_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
