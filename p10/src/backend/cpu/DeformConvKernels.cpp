// deform_conv2d / deform_conv2d_backward CPU kernels.
//
// Deformable convolution with per-tap sampling offsets. offset is
// (N, 2*offset_groups*kh*kw, out_h, out_w): within one offset group the
// channels are (dh, dw) pairs per kernel tap in row-major tap order, so
// channel 2*k gives the y offset of tap k = i*kw+j and 2*k+1 the x offset.
// The sampled value is the bilinear interpolation of the input at
//   y = out_y*stride_h - pad_h + i*dilation_h + offset_h
// with out-of-range neighbours contributing zero (no edge snapping).
// mask (N, offset_groups*kh*kw, out_h, out_w) optionally scales each tap
// (DCNv2); backward propagates into input, weight, offset, mask and bias.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

template <typename T, typename CT>
T deform_bilinear_sample(const T* data, int64_t height, int64_t width,
                         CT y, CT x, int64_t& y_low, int64_t& x_low,
                         CT& ly, CT& lx, CT& hy, CT& hx,
                         CT* v1, CT* v2, CT* v3, CT* v4) {
    y_low = static_cast<int64_t>(std::floor(y));
    x_low = static_cast<int64_t>(std::floor(x));
    *v1 = *v2 = *v3 = *v4 = CT(0);
    ly = lx = hy = hx = CT(0);
    if (y <= CT(-1) || CT(height) <= y || x <= CT(-1) || CT(width) <= x) {
        return T(0);
    }
    ly = y - y_low;
    lx = x - x_low;
    hy = CT(1) - ly;
    hx = CT(1) - lx;
    *v1 = (y_low >= 0 && x_low >= 0) ? static_cast<CT>(data[y_low * width + x_low]) : CT(0);
    *v2 = (y_low >= 0 && x_low + 1 <= width - 1) ? static_cast<CT>(data[y_low * width + x_low + 1]) : CT(0);
    *v3 = (y_low + 1 <= height - 1 && x_low >= 0) ? static_cast<CT>(data[(y_low + 1) * width + x_low]) : CT(0);
    *v4 = (y_low + 1 <= height - 1 && x_low + 1 <= width - 1) ? static_cast<CT>(data[(y_low + 1) * width + x_low + 1]) : CT(0);
    return T(hy * hx * (*v1) + hy * lx * (*v2) + ly * hx * (*v3) + ly * lx * (*v4));
}

template <typename storage_t, typename acc_t>
static Tensor deform_conv2d_cpu_impl(
        const Tensor& input, const Tensor& weight, const Tensor& offset,
        const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t OC = weight.size(0);
    const int64_t kh = weight.size(2);
    const int64_t kw = weight.size(3);
    const int64_t stride_h = stride[0], stride_w = stride[1];
    const int64_t pad_h = padding[0], pad_w = padding[1];
    const int64_t dil_h = dilation[0], dil_w = dilation[1];
    const int64_t ker_h = dil_h * (kh - 1) + 1;
    const int64_t ker_w = dil_w * (kw - 1) + 1;
    const int64_t out_h = (H + 2 * pad_h - ker_h) / stride_h + 1;
    const int64_t out_w = (W + 2 * pad_w - ker_w) / stride_w + 1;
    const int64_t in_per_group = C / groups;
    const int64_t out_per_group = OC / groups;
    const int64_t c_per_off = C / offset_groups;

    Tensor output = Tensor::empty({N, OC, out_h, out_w}, input.dtype(), input.device());
    if (output.numel() == 0) return output;

    const storage_t* in_ptr = input.data_ptr<storage_t>();
    const storage_t* w_ptr = weight.data_ptr<storage_t>();
    const storage_t* off_ptr = offset.data_ptr<storage_t>();
    const storage_t* mask_ptr = use_mask ? mask.data_ptr<storage_t>() : nullptr;
    const storage_t* bias_ptr = bias.has_value() ? bias->data_ptr<storage_t>() : nullptr;
    storage_t* out_ptr = output.data_ptr<storage_t>();
    const int64_t noff = offset_groups * kh * kw;

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* in_n = in_ptr + n * C * H * W;
            const storage_t* off_n = off_ptr + n * 2 * noff * out_h * out_w;
            const storage_t* mask_n = use_mask ? mask_ptr + n * noff * out_h * out_w : nullptr;
            storage_t* out_n = out_ptr + n * OC * out_h * out_w;
            for (int64_t oc = 0; oc < OC; ++oc) {
                const int64_t g_w = oc / out_per_group;
                const storage_t* w_oc = w_ptr + oc * in_per_group * kh * kw;
                const acc_t bias_val = bias_ptr ? static_cast<acc_t>(bias_ptr[oc]) : acc_t(0);
                storage_t* out_c = out_n + oc * out_h * out_w;
                for (int64_t oy = 0; oy < out_h; ++oy) {
                    for (int64_t ox = 0; ox < out_w; ++ox) {
                        acc_t val = 0;
                        for (int64_t i = 0; i < kh; ++i) {
                            for (int64_t j = 0; j < kw; ++j) {
                                const int64_t k = i * kw + j;
                                for (int64_t ic = 0; ic < in_per_group; ++ic) {
                                    const int64_t in_c = g_w * in_per_group + ic;
                                    const int64_t g_o = in_c / c_per_off;
                                    const storage_t* off_h_ch = off_n + (2 * g_o * kh * kw + 2 * k) * out_h * out_w;
                                    const storage_t* off_w_ch = off_n + (2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w;
                                    const acc_t off_h = static_cast<acc_t>(off_h_ch[oy * out_w + ox]);
                                    const acc_t off_w = static_cast<acc_t>(off_w_ch[oy * out_w + ox]);
                                    const acc_t y = static_cast<acc_t>(oy) * stride_h - pad_h + i * dil_h + off_h;
                                    const acc_t x = static_cast<acc_t>(ox) * stride_w - pad_w + j * dil_w + off_w;
                                    int64_t y_low, x_low;
                                    acc_t ly, lx, hy, hx;
                                    acc_t v1, v2, v3, v4;
                                    const acc_t sample = deform_bilinear_sample(
                                        in_n + in_c * H * W, H, W, y, x,
                                        y_low, x_low, ly, lx, hy, hx, &v1, &v2, &v3, &v4);
                                    acc_t s = sample;
                                    if (use_mask) {
                                        const storage_t* mask_ch = mask_n + (g_o * kh * kw + k) * out_h * out_w;
                                        s *= static_cast<acc_t>(mask_ch[oy * out_w + ox]);
                                    }
                                    val += static_cast<acc_t>(w_oc[(ic * kh + i) * kw + j]) * s;
                                }
                            }
                        }
                        out_c[oy * out_w + ox] = static_cast<storage_t>(val + bias_val);
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename acc_t>
static std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> deform_conv2d_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const Tensor& offset, const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask, const std::vector<bool>& output_mask) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t OC = weight.size(0);
    const int64_t IC = weight.size(1);
    const int64_t kh = weight.size(2);
    const int64_t kw = weight.size(3);
    const int64_t stride_h = stride[0], stride_w = stride[1];
    const int64_t pad_h = padding[0], pad_w = padding[1];
    const int64_t dil_h = dilation[0], dil_w = dilation[1];
    const int64_t ker_h = dil_h * (kh - 1) + 1;
    const int64_t ker_w = dil_w * (kw - 1) + 1;
    const int64_t out_h = (H + 2 * pad_h - ker_h) / stride_h + 1;
    const int64_t out_w = (W + 2 * pad_w - ker_w) / stride_w + 1;
    const int64_t in_per_group = C / groups;
    const int64_t out_per_group = OC / groups;
    const int64_t c_per_off = C / offset_groups;
    const int64_t noff = offset_groups * kh * kw;

    const bool need_input = output_mask[0];
    const bool need_weight = output_mask[1];
    const bool need_offset = output_mask[2];
    const bool need_mask = output_mask[3];
    const bool need_bias = output_mask[4];

    Tensor grad_input = need_input
        ? Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    Tensor grad_weight = need_weight
        ? Tensor::zeros({OC, IC, kh, kw}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({OC, IC, kh, kw}, grad_output.dtype(), grad_output.device());
    Tensor grad_offset = need_offset
        ? Tensor::zeros({N, 2 * noff, out_h, out_w}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({N, 2 * noff, out_h, out_w}, grad_output.dtype(), grad_output.device());
    Tensor grad_mask = need_mask
        ? Tensor::zeros({N, noff, out_h, out_w}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({N, noff, out_h, out_w}, grad_output.dtype(), grad_output.device());
    Tensor grad_bias = need_bias
        ? Tensor::zeros({OC}, grad_output.dtype(), grad_output.device())
        : Tensor::zeros({OC}, grad_output.dtype(), grad_output.device());
    if (grad_output.numel() == 0) {
        return {grad_input, grad_weight, grad_offset, grad_mask, grad_bias};
    }

    const storage_t* grad_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* in_ptr = input.data_ptr<storage_t>();
    const storage_t* w_ptr = weight.data_ptr<storage_t>();
    const storage_t* off_ptr = offset.data_ptr<storage_t>();
    const storage_t* mask_ptr = use_mask ? mask.data_ptr<storage_t>() : nullptr;
    storage_t* g_in_ptr = grad_input.data_ptr<storage_t>();
    storage_t* g_w_ptr = grad_weight.data_ptr<storage_t>();
    storage_t* g_off_ptr = grad_offset.data_ptr<storage_t>();
    storage_t* g_mask_ptr = grad_mask.data_ptr<storage_t>();
    storage_t* g_bias_ptr = grad_bias.data_ptr<storage_t>();

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* in_n = in_ptr + n * C * H * W;
            const storage_t* off_n = off_ptr + n * 2 * noff * out_h * out_w;
            const storage_t* mask_n = use_mask ? mask_ptr + n * noff * out_h * out_w : nullptr;
            const storage_t* grad_n = grad_ptr + n * OC * out_h * out_w;
            storage_t* g_in_n = need_input ? g_in_ptr + n * C * H * W : nullptr;
            storage_t* g_off_n = need_offset ? g_off_ptr + n * 2 * noff * out_h * out_w : nullptr;
            storage_t* g_mask_n = need_mask ? g_mask_ptr + n * noff * out_h * out_w : nullptr;
            for (int64_t oc = 0; oc < OC; ++oc) {
                const int64_t g_w = oc / out_per_group;
                const storage_t* w_oc = w_ptr + oc * in_per_group * kh * kw;
                const storage_t* grad_c = grad_n + oc * out_h * out_w;
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t k = i * kw + j;
                        for (int64_t ic = 0; ic < in_per_group; ++ic) {
                            const int64_t in_c = g_w * in_per_group + ic;
                            const int64_t g_o = in_c / c_per_off;
                            const storage_t* off_h_ch = off_n + (2 * g_o * kh * kw + 2 * k) * out_h * out_w;
                            const storage_t* off_w_ch = off_n + (2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w;
                            const storage_t* mask_ch = use_mask
                                ? mask_n + (g_o * kh * kw + k) * out_h * out_w : nullptr;
                            const storage_t* in_cp = in_n + in_c * H * W;
                            const storage_t* w_icp = w_oc + ic * kh * kw;
                            for (int64_t oy = 0; oy < out_h; ++oy) {
                                for (int64_t ox = 0; ox < out_w; ++ox) {
                                    const acc_t g = static_cast<acc_t>(grad_c[oy * out_w + ox]);
                                    if (g == acc_t(0)) continue;
                                    const acc_t off_h = static_cast<acc_t>(off_h_ch[oy * out_w + ox]);
                                    const acc_t off_w = static_cast<acc_t>(off_w_ch[oy * out_w + ox]);
                                    const acc_t y = static_cast<acc_t>(oy) * stride_h - pad_h + i * dil_h + off_h;
                                    const acc_t x = static_cast<acc_t>(ox) * stride_w - pad_w + j * dil_w + off_w;
                                    int64_t y_low, x_low;
                                    acc_t ly, lx, hy, hx;
                                    acc_t v1, v2, v3, v4;
                                    const acc_t sample = deform_bilinear_sample(
                                        in_cp, H, W, y, x, y_low, x_low, ly, lx, hy, hx,
                                        &v1, &v2, &v3, &v4);
                                    acc_t mval = acc_t(1);
                                    if (use_mask) {
                                        mval = static_cast<acc_t>(mask_ch[oy * out_w + ox]);
                                    }
                                    const acc_t w_val = static_cast<acc_t>(w_icp[k]);
                                    const acc_t contrib = g * w_val * mval;
                                    if (need_input && y_low >= 0 && x_low >= 0) {
                                        g_in_n[in_c * H * W + y_low * W + x_low] +=
                                            static_cast<storage_t>(contrib * hy * hx);
                                    }
                                    if (need_input && y_low >= 0 && x_low + 1 <= W - 1) {
                                        g_in_n[in_c * H * W + y_low * W + x_low + 1] +=
                                            static_cast<storage_t>(contrib * hy * lx);
                                    }
                                    if (need_input && y_low + 1 <= H - 1 && x_low >= 0) {
                                        g_in_n[in_c * H * W + (y_low + 1) * W + x_low] +=
                                            static_cast<storage_t>(contrib * ly * hx);
                                    }
                                    if (need_input && y_low + 1 <= H - 1 && x_low + 1 <= W - 1) {
                                        g_in_n[in_c * H * W + (y_low + 1) * W + x_low + 1] +=
                                            static_cast<storage_t>(contrib * ly * lx);
                                    }
                                    if (need_offset) {
                                        const acc_t dval_dy = -hx * v1 - lx * v2 + hx * v3 + lx * v4;
                                        const acc_t dval_dx = -hy * v1 + hy * v2 - ly * v3 + ly * v4;
                                        g_off_n[(2 * g_o * kh * kw + 2 * k) * out_h * out_w + oy * out_w + ox] +=
                                            static_cast<storage_t>(contrib * dval_dy);
                                        g_off_n[(2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w + oy * out_w + ox] +=
                                            static_cast<storage_t>(contrib * dval_dx);
                                    }
                                    if (need_mask && use_mask) {
                                        g_mask_n[(g_o * kh * kw + k) * out_h * out_w + oy * out_w + ox] +=
                                            static_cast<storage_t>(g * w_val * sample);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    // Weight and bias gradients are shared across batch threads, so they get
    // a second pass with per-output-channel ownership. This re-runs the
    // sampling decomposition, trading a redundant forward pass for a race-free
    // accumulation.
    if (need_weight || need_bias) {
        parallel_for(0, OC, 1, [&](int64_t begin, int64_t end) {
            for (int64_t oc = begin; oc < end; ++oc) {
                const int64_t g_w = oc / out_per_group;
                const storage_t* w_oc = w_ptr + oc * in_per_group * kh * kw;
                for (int64_t n = 0; n < N; ++n) {
                    const storage_t* in_n = in_ptr + n * C * H * W;
                    const storage_t* off_n = off_ptr + n * 2 * noff * out_h * out_w;
                    const storage_t* mask_n = use_mask ? mask_ptr + n * noff * out_h * out_w : nullptr;
                    const storage_t* grad_c = grad_ptr + (n * OC + oc) * out_h * out_w;
                    for (int64_t i = 0; i < kh; ++i) {
                        for (int64_t j = 0; j < kw; ++j) {
                            const int64_t k = i * kw + j;
                            for (int64_t ic = 0; ic < in_per_group; ++ic) {
                                const int64_t in_c = g_w * in_per_group + ic;
                                const int64_t g_o = in_c / c_per_off;
                                const storage_t* off_h_ch = off_n + (2 * g_o * kh * kw + 2 * k) * out_h * out_w;
                                const storage_t* off_w_ch = off_n + (2 * g_o * kh * kw + 2 * k + 1) * out_h * out_w;
                                const storage_t* mask_ch = use_mask
                                    ? mask_n + (g_o * kh * kw + k) * out_h * out_w : nullptr;
                                const storage_t* in_cp = in_n + in_c * H * W;
                                const storage_t* w_icp = w_oc + ic * kh * kw;
                                for (int64_t oy = 0; oy < out_h; ++oy) {
                                    for (int64_t ox = 0; ox < out_w; ++ox) {
                                        const acc_t g = static_cast<acc_t>(grad_c[oy * out_w + ox]);
                                        if (g == acc_t(0)) continue;
                                        const acc_t off_h = static_cast<acc_t>(off_h_ch[oy * out_w + ox]);
                                        const acc_t off_w = static_cast<acc_t>(off_w_ch[oy * out_w + ox]);
                                        const acc_t y = static_cast<acc_t>(oy) * stride_h - pad_h + i * dil_h + off_h;
                                        const acc_t x = static_cast<acc_t>(ox) * stride_w - pad_w + j * dil_w + off_w;
                                        int64_t y_low, x_low;
                                        acc_t ly, lx, hy, hx;
                                        acc_t v1, v2, v3, v4;
                                        const acc_t sample = deform_bilinear_sample(
                                            in_cp, H, W, y, x, y_low, x_low, ly, lx, hy, hx,
                                            &v1, &v2, &v3, &v4);
                                        acc_t mval = acc_t(1);
                                        if (use_mask) {
                                            mval = static_cast<acc_t>(mask_ch[oy * out_w + ox]);
                                        }
                                        if (need_weight) {
                                            g_w_ptr[(oc * in_per_group + ic) * kh * kw + k] +=
                                                static_cast<storage_t>(g * sample * mval);
                                        }
                                        if (need_bias && i == 0 && j == 0 && ic == 0) {
                                            g_bias_ptr[oc] += static_cast<storage_t>(g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    return {grad_input, grad_weight, grad_offset, grad_mask, grad_bias};
}

} // namespace

Tensor deform_conv2d_cpu(const Tensor& input, const Tensor& weight, const Tensor& offset,
                         const Tensor& mask, const std::optional<Tensor>& bias,
                         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& dilation, int64_t groups,
                         int64_t offset_groups, bool use_mask) {
    if (input.dim() != 4 || weight.dim() != 4 || offset.dim() != 4)
        TP_THROW(RuntimeError, "deform_conv2d: expected 4-D input, weight and offset");
    if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2)
        TP_THROW(RuntimeError, "deform_conv2d: stride, padding and dilation must have 2 entries");
    if (stride[0] <= 0 || stride[1] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 ||
        padding[0] < 0 || padding[1] < 0)
        TP_THROW(RuntimeError, "deform_conv2d: stride/dilation must be positive and padding non-negative");
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t kh = weight.size(2);
    const int64_t kw = weight.size(3);
    if (weight.size(1) * groups != C)
        TP_THROW(RuntimeError, "deform_conv2d: weight channels times groups must match input channels");
    if (weight.size(0) % groups != 0)
        TP_THROW(RuntimeError, "deform_conv2d: output channels must be divisible by groups");
    if (offset.size(1) != offset_groups * 2 * kh * kw)
        TP_THROW(RuntimeError, "deform_conv2d: offset channel count mismatch");
    if (use_mask && mask.size(1) != offset_groups * kh * kw)
        TP_THROW(RuntimeError, "deform_conv2d: mask channel count mismatch");
    if (C % offset_groups != 0)
        TP_THROW(RuntimeError, "deform_conv2d: input channels must be divisible by offset_groups");
    const int64_t ker_h = dilation[0] * (kh - 1) + 1;
    const int64_t ker_w = dilation[1] * (kw - 1) + 1;
    const int64_t out_h = (input.size(2) + 2 * padding[0] - ker_h) / stride[0] + 1;
    const int64_t out_w = (input.size(3) + 2 * padding[1] - ker_w) / stride[1] + 1;
    if (out_h <= 0 || out_w <= 0)
        TP_THROW(RuntimeError, "deform_conv2d: computed output size must be positive");
    if (offset.size(2) != out_h || offset.size(3) != out_w)
        TP_THROW(RuntimeError, "deform_conv2d: offset spatial size must match the computed output size");
    if (bias.has_value() && bias->size(0) != weight.size(0))
        TP_THROW(RuntimeError, "deform_conv2d: bias size must match output channels");
    if (input.dtype() != weight.dtype() || input.dtype() != offset.dtype() ||
        (use_mask && input.dtype() != mask.dtype()))
        TP_THROW(RuntimeError, "deform_conv2d: input, weight, offset and mask must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor wc = weight.contiguous();
    const Tensor oc = offset.contiguous();
    std::optional<Tensor> bc = std::nullopt;
    if (bias.has_value()) bc = bias->contiguous();
    const Tensor mc = use_mask ? mask.contiguous() : mask;
    const std::vector<int64_t> st{stride[0], stride[1]};
    const std::vector<int64_t> pd{padding[0], padding[1]};
    const std::vector<int64_t> dl{dilation[0], dilation[1]};
    switch (ic.dtype()) {
        case DType::Float32:
            return deform_conv2d_cpu_impl<float, float>(ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask);
        case DType::Float64:
            return deform_conv2d_cpu_impl<double, double>(ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask);
        case DType::Float16:
            return deform_conv2d_cpu_impl<Half, float>(ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask);
        default: TP_THROW(TypeError, "deform_conv2d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> deform_conv2d_backward_cpu(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const Tensor& offset, const Tensor& mask, const std::optional<Tensor>& bias,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, int64_t groups,
        int64_t offset_groups, bool use_mask, const std::vector<bool>& output_mask) {
    if (output_mask.size() != 5)
        TP_THROW(RuntimeError, "deform_conv2d_backward: output_mask must have 5 entries");
    const Tensor gc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor wc = weight.contiguous();
    const Tensor oc = offset.contiguous();
    std::optional<Tensor> bc = std::nullopt;
    if (bias.has_value()) bc = bias->contiguous();
    const Tensor mc = use_mask ? mask.contiguous() : mask;
    const std::vector<int64_t> st{stride[0], stride[1]};
    const std::vector<int64_t> pd{padding[0], padding[1]};
    const std::vector<int64_t> dl{dilation[0], dilation[1]};
    switch (ic.dtype()) {
        case DType::Float32:
            return deform_conv2d_backward_cpu_impl<float, float>(
                gc, ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask, output_mask);
        case DType::Float64:
            return deform_conv2d_backward_cpu_impl<double, double>(
                gc, ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask, output_mask);
        case DType::Float16:
            return deform_conv2d_backward_cpu_impl<Half, float>(
                gc, ic, wc, oc, mc, bc, st, pd, dl, groups, offset_groups, use_mask, output_mask);
        default: TP_THROW(TypeError, "deform_conv2d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, DeformConvKernels) {
    m.impl("deform_conv2d", deform_conv2d_cpu);
    m.impl("deform_conv2d_backward", deform_conv2d_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay