// ps_roi_align / ps_roi_align_backward CPU kernels.
//
// Position-sensitive ROI align: the input channel axis is laid out as
// (out_channels, pooled_height, pooled_width) groups, and the pooled output
// bin (ph, pw) of output channel c reads input channel
//   c_in = c * (pooled_height * pooled_width) + ph * pooled_width + pw.
// Sampling math (bilinear grid averaging) matches roi_align; there is no
// half-pixel offset mode.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

template <typename T, typename CT>
T bilinear_sample_ps(const T* data, int64_t height, int64_t width, CT y, CT x) {
    if (y < CT(-1) || y > CT(height) || x < CT(-1) || x > CT(width)) {
        return T(0);
    }
    if (y <= CT(0)) y = CT(0);
    if (x <= CT(0)) x = CT(0);
    int64_t y_low = static_cast<int64_t>(y);
    int64_t x_low = static_cast<int64_t>(x);
    int64_t y_high = y_low + 1;
    int64_t x_high = x_low + 1;
    if (y_low >= height - 1) {
        y_high = y_low;
        y = CT(y_low);
    }
    if (x_low >= width - 1) {
        x_high = x_low;
        x = CT(x_low);
    }
    const CT ly = y - y_low, lx = x - x_low;
    const CT hy = CT(1) - ly, hx = CT(1) - lx;
    return T(hy * hx * static_cast<CT>(data[y_low * width + x_low]) +
             hy * lx * static_cast<CT>(data[y_low * width + x_high]) +
             ly * hx * static_cast<CT>(data[y_high * width + x_low]) +
             ly * lx * static_cast<CT>(data[y_high * width + x_high]));
}

template <typename storage_t, typename acc_t>
static Tensor ps_roi_align_cpu_impl(const Tensor& input, const Tensor& rois,
                                    acc_t spatial_scale, int64_t pooled_height,
                                    int64_t pooled_width, int64_t sampling_ratio) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t R = rois.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor output = Tensor::empty({R, channels, pooled_height, pooled_width},
                                  input.dtype(), input.device());
    if (R == 0 || output.numel() == 0) return output;

    const storage_t* in_ptr = input.data_ptr<storage_t>();
    const storage_t* rois_ptr = rois.data_ptr<storage_t>();
    storage_t* out_ptr = output.data_ptr<storage_t>();

    parallel_for(0, R, 1, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
            const storage_t* rv = rois_ptr + r * 5;
            const int64_t batch = static_cast<int64_t>(rv[0]);
            if (batch < 0 || batch >= N) continue;
            // Position-sensitive align is always half-pixel offset.
            const acc_t roi_start_w = rv[1] * spatial_scale - acc_t(0.5);
            const acc_t roi_start_h = rv[2] * spatial_scale - acc_t(0.5);
            const acc_t roi_end_w = rv[3] * spatial_scale - acc_t(0.5);
            const acc_t roi_end_h = rv[4] * spatial_scale - acc_t(0.5);
            const acc_t roi_width = roi_end_w - roi_start_w;
            const acc_t roi_height = roi_end_h - roi_start_h;
            const acc_t bin_size_h = roi_height / pooled_height;
            const acc_t bin_size_w = roi_width / pooled_width;

            int64_t grid_h, grid_w;
            if (sampling_ratio > 0) {
                grid_h = grid_w = sampling_ratio;
            } else {
                grid_h = static_cast<int64_t>(std::ceil(roi_height / pooled_height));
                grid_w = static_cast<int64_t>(std::ceil(roi_width / pooled_width));
            }
            const acc_t count = static_cast<acc_t>(grid_h * grid_w);

            const storage_t* in_b = in_ptr + batch * C * H * W;
            storage_t* out_r = out_ptr + r * channels * pooled_height * pooled_width;
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t ph = 0; ph < pooled_height; ++ph) {
                    for (int64_t pw = 0; pw < pooled_width; ++pw) {
                        const int64_t c_in = c * (pooled_height * pooled_width) +
                                             ph * pooled_width + pw;
                        const storage_t* in_c = in_b + c_in * H * W;
                        acc_t val = 0;
                        for (int64_t iy = 0; iy < grid_h; ++iy) {
                            const acc_t y = roi_start_h + ph * bin_size_h +
                                (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid_h;
                            for (int64_t ix = 0; ix < grid_w; ++ix) {
                                const acc_t x = roi_start_w + pw * bin_size_w +
                                    (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid_w;
                                val += bilinear_sample_ps(in_c, H, W, y, x);
                            }
                        }
                        out_r[(c * pooled_height + ph) * pooled_width + pw] =
                            static_cast<storage_t>(val / count);
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename acc_t>
static Tensor ps_roi_align_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& rois, acc_t spatial_scale,
        int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio,
        const std::vector<int64_t>& input_size) {
    const int64_t N = input_size[0];
    const int64_t C = input_size[1];
    const int64_t H = input_size[2];
    const int64_t W = input_size[3];
    const int64_t R = rois.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    if (R == 0 || grad_input.numel() == 0) return grad_input;

    const storage_t* grad_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* rois_ptr = rois.data_ptr<storage_t>();
    storage_t* g_in = grad_input.data_ptr<storage_t>();

    // Batch-parallel: ROIs sharing a batch can scatter into the same pixel.
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            storage_t* g_b = g_in + batch * C * H * W;
            for (int64_t r = 0; r < R; ++r) {
                const storage_t* rv = rois_ptr + r * 5;
                if (static_cast<int64_t>(rv[0]) != batch) continue;
                const acc_t roi_start_w = rv[1] * spatial_scale - acc_t(0.5);
                const acc_t roi_start_h = rv[2] * spatial_scale - acc_t(0.5);
                const acc_t roi_end_w = rv[3] * spatial_scale - acc_t(0.5);
                const acc_t roi_end_h = rv[4] * spatial_scale - acc_t(0.5);
                const acc_t roi_width = roi_end_w - roi_start_w;
                const acc_t roi_height = roi_end_h - roi_start_h;
                const acc_t bin_size_h = roi_height / pooled_height;
                const acc_t bin_size_w = roi_width / pooled_width;

                int64_t grid_h, grid_w;
                if (sampling_ratio > 0) {
                    grid_h = grid_w = sampling_ratio;
                } else {
                    grid_h = static_cast<int64_t>(std::ceil(roi_height / pooled_height));
                    grid_w = static_cast<int64_t>(std::ceil(roi_width / pooled_width));
                }
                const acc_t count = static_cast<acc_t>(grid_h * grid_w);

                const storage_t* grad_r = grad_ptr + r * channels * pooled_height * pooled_width;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t ph = 0; ph < pooled_height; ++ph) {
                        for (int64_t pw = 0; pw < pooled_width; ++pw) {
                            const int64_t c_in = c * (pooled_height * pooled_width) +
                                                 ph * pooled_width + pw;
                            storage_t* g_c = g_b + c_in * H * W;
                            const acc_t grad_val = static_cast<acc_t>(
                                grad_r[(c * pooled_height + ph) * pooled_width + pw]) / count;
                            for (int64_t iy = 0; iy < grid_h; ++iy) {
                                const acc_t y = roi_start_h + ph * bin_size_h +
                                    (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid_h;
                                for (int64_t ix = 0; ix < grid_w; ++ix) {
                                    const acc_t x = roi_start_w + pw * bin_size_w +
                                        (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid_w;
                                    if (y < acc_t(-1) || y > acc_t(H) ||
                                        x < acc_t(-1) || x > acc_t(W)) {
                                        continue;
                                    }
                                    acc_t yc = y, xc = x;
                                    if (yc <= acc_t(0)) yc = acc_t(0);
                                    if (xc <= acc_t(0)) xc = acc_t(0);
                                    int64_t y_low = static_cast<int64_t>(yc);
                                    int64_t x_low = static_cast<int64_t>(xc);
                                    int64_t y_high = y_low + 1;
                                    int64_t x_high = x_low + 1;
                                    if (y_low >= H - 1) {
                                        y_high = y_low;
                                        yc = acc_t(y_low);
                                    }
                                    if (x_low >= W - 1) {
                                        x_high = x_low;
                                        xc = acc_t(x_low);
                                    }
                                    const acc_t ly = yc - y_low, lx = xc - x_low;
                                    const acc_t hy = acc_t(1) - ly, hx = acc_t(1) - lx;
                                    g_c[y_low * W + x_low] += static_cast<storage_t>(hy * hx * grad_val);
                                    g_c[y_low * W + x_high] += static_cast<storage_t>(hy * lx * grad_val);
                                    g_c[y_high * W + x_low] += static_cast<storage_t>(ly * hx * grad_val);
                                    g_c[y_high * W + x_high] += static_cast<storage_t>(ly * lx * grad_val);
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

} // namespace

Tensor ps_roi_align_cpu(const Tensor& input, const Tensor& rois, double spatial_scale,
                        int64_t pooled_height, int64_t pooled_width,
                        int64_t sampling_ratio) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "ps_roi_align: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_align: rois must be of shape (R, 5)");
    if (input.size(1) % (pooled_height * pooled_width) != 0)
        TP_THROW(RuntimeError, "ps_roi_align: channels must be divisible by pooled_height * pooled_width");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "ps_roi_align: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32:
            return ps_roi_align_cpu_impl<float, float>(ic, rc, scale, pooled_height, pooled_width, sampling_ratio);
        case DType::Float64:
            return ps_roi_align_cpu_impl<double, double>(ic, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio);
        case DType::Float16:
            return ps_roi_align_cpu_impl<Half, float>(ic, rc, scale, pooled_height, pooled_width, sampling_ratio);
        default: TP_THROW(TypeError, "ps_roi_align: unsupported dtype");
    }
}

Tensor ps_roi_align_backward_cpu(const Tensor& grad_output, const Tensor& rois,
                                 double spatial_scale, int64_t pooled_height,
                                 int64_t pooled_width, int64_t sampling_ratio,
                                 const std::vector<int64_t>& input_size) {
    if (input_size.size() != 4)
        TP_THROW(RuntimeError, "ps_roi_align_backward: input_size must have 4 entries");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_align_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (gc.dtype()) {
        case DType::Float32:
            return ps_roi_align_backward_cpu_impl<float, float>(gc, rc, scale, pooled_height, pooled_width, sampling_ratio, input_size);
        case DType::Float64:
            return ps_roi_align_backward_cpu_impl<double, double>(gc, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio, input_size);
        case DType::Float16:
            return ps_roi_align_backward_cpu_impl<Half, float>(gc, rc, scale, pooled_height, pooled_width, sampling_ratio, input_size);
        default: TP_THROW(TypeError, "ps_roi_align_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, PsRoiAlignKernels) {
    m.impl("ps_roi_align", ps_roi_align_cpu);
    m.impl("ps_roi_align_backward", ps_roi_align_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay