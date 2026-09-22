// roi_align / roi_align_backward CPU kernels.
//
// ROIs are (R, 5) float rows [batch_index, x1, y1, x2, y2]. Each output bin
// averages a grid of bilinearly sampled points; the grid density is either
// explicit (sampling_ratio > 0) or adaptive, one point per unit of ROI size.
// With aligned=true the ROI corners are shifted by half a pixel so pixel
// centers, not edges, carry the coordinates.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

template <typename T, typename CT>
T bilinear_sample(const T* data, int64_t height, int64_t width, CT y, CT x) {
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

struct RoiBinGrid {
    int64_t grid_h;
    int64_t grid_w;
    double count;
};

inline RoiBinGrid compute_bin_grid(double roi_height, double roi_width,
                                   int64_t pooled_height, int64_t pooled_width,
                                   int64_t sampling_ratio) {
    RoiBinGrid g;
    if (sampling_ratio > 0) {
        g.grid_h = g.grid_w = sampling_ratio;
    } else {
        // Adaptive density: one sample per unit of ROI size per bin. The
        // count may collapse to zero for degenerate (negative-size) ROIs;
        // forward guards the division, backward deliberately does not.
        g.grid_h = static_cast<int64_t>(std::ceil(roi_height / pooled_height));
        g.grid_w = static_cast<int64_t>(std::ceil(roi_width / pooled_width));
    }
    g.count = static_cast<double>(g.grid_h * g.grid_w);
    return g;
}

template <typename storage_t, typename acc_t>
static Tensor roi_align_cpu_impl(const Tensor& input, const Tensor& rois,
                                 acc_t spatial_scale, int64_t pooled_height,
                                 int64_t pooled_width, int64_t sampling_ratio,
                                 bool aligned) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t R = rois.size(0);
    Tensor output = Tensor::empty({R, C, pooled_height, pooled_width},
                                  input.dtype(), input.device());
    if (R == 0 || output.numel() == 0) return output;

    const storage_t* in_ptr = input.data_ptr<storage_t>();
    const storage_t* rois_ptr = rois.data_ptr<storage_t>();
    storage_t* out_ptr = output.data_ptr<storage_t>();
    const acc_t half = acc_t(0.5);

    parallel_for(0, R, 1, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
            const storage_t* rv = rois_ptr + r * 5;
            const int64_t batch = static_cast<int64_t>(rv[0]);
            if (batch < 0 || batch >= N) {
                continue; // out-of-range batch index yields zero output
            }
            const acc_t roi_start_w = rv[1] * spatial_scale - (aligned ? half : acc_t(0));
            const acc_t roi_start_h = rv[2] * spatial_scale - (aligned ? half : acc_t(0));
            const acc_t roi_end_w = rv[3] * spatial_scale - (aligned ? half : acc_t(0));
            const acc_t roi_end_h = rv[4] * spatial_scale - (aligned ? half : acc_t(0));
            acc_t roi_width = roi_end_w - roi_start_w;
            acc_t roi_height = roi_end_h - roi_start_h;
            if (!aligned) {
                if (roi_width < acc_t(1)) roi_width = acc_t(1);
                if (roi_height < acc_t(1)) roi_height = acc_t(1);
            }
            const acc_t bin_size_h = roi_height / pooled_height;
            const acc_t bin_size_w = roi_width / pooled_width;
            const RoiBinGrid grid = compute_bin_grid(
                static_cast<double>(roi_height), static_cast<double>(roi_width),
                pooled_height, pooled_width, sampling_ratio);
            const acc_t count = std::max(acc_t(1), static_cast<acc_t>(grid.count));

            const storage_t* in_b = in_ptr + batch * C * H * W;
            storage_t* out_r = out_ptr + r * C * pooled_height * pooled_width;
            for (int64_t c = 0; c < C; ++c) {
                const storage_t* in_c = in_b + c * H * W;
                storage_t* out_c = out_r + c * pooled_height * pooled_width;
                for (int64_t ph = 0; ph < pooled_height; ++ph) {
                    for (int64_t pw = 0; pw < pooled_width; ++pw) {
                        acc_t val = 0;
                        for (int64_t iy = 0; iy < grid.grid_h; ++iy) {
                            const acc_t y = roi_start_h + ph * bin_size_h +
                                (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid.grid_h;
                            for (int64_t ix = 0; ix < grid.grid_w; ++ix) {
                                const acc_t x = roi_start_w + pw * bin_size_w +
                                    (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid.grid_w;
                                val += bilinear_sample(in_c, H, W, y, x);
                            }
                        }
                        out_c[ph * pooled_width + pw] = static_cast<storage_t>(val / count);
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename acc_t>
static Tensor roi_align_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& rois, acc_t spatial_scale,
        int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio,
        bool aligned, const std::vector<int64_t>& input_size) {
    const int64_t N = input_size[0];
    const int64_t C = input_size[1];
    const int64_t H = input_size[2];
    const int64_t W = input_size[3];
    const int64_t R = rois.size(0);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    if (R == 0 || grad_input.numel() == 0) return grad_input;

    const storage_t* grad_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* rois_ptr = rois.data_ptr<storage_t>();
    storage_t* g_in = grad_input.data_ptr<storage_t>();
    const acc_t half = acc_t(0.5);

    // Parallelize over batch, not over ROIs: ROIs of one batch can scatter
    // into the same input pixel, so their contributions must accumulate
    // sequentially inside one thread.
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            storage_t* g_b = g_in + batch * C * H * W;
            for (int64_t r = 0; r < R; ++r) {
                const storage_t* rv = rois_ptr + r * 5;
                if (static_cast<int64_t>(rv[0]) != batch) continue;
                const acc_t roi_start_w = rv[1] * spatial_scale - (aligned ? half : acc_t(0));
                const acc_t roi_start_h = rv[2] * spatial_scale - (aligned ? half : acc_t(0));
                const acc_t roi_end_w = rv[3] * spatial_scale - (aligned ? half : acc_t(0));
                const acc_t roi_end_h = rv[4] * spatial_scale - (aligned ? half : acc_t(0));
                acc_t roi_width = roi_end_w - roi_start_w;
                acc_t roi_height = roi_end_h - roi_start_h;
                if (!aligned) {
                    if (roi_width < acc_t(1)) roi_width = acc_t(1);
                    if (roi_height < acc_t(1)) roi_height = acc_t(1);
                }
                const acc_t bin_size_h = roi_height / pooled_height;
                const acc_t bin_size_w = roi_width / pooled_width;
                const RoiBinGrid grid = compute_bin_grid(
                    static_cast<double>(roi_height), static_cast<double>(roi_width),
                    pooled_height, pooled_width, sampling_ratio);
                const acc_t count = static_cast<acc_t>(grid.count);

                const storage_t* grad_r = grad_ptr + r * C * pooled_height * pooled_width;
                for (int64_t c = 0; c < C; ++c) {
                    const storage_t* grad_c = grad_r + c * pooled_height * pooled_width;
                    storage_t* g_c = g_b + c * H * W;
                    for (int64_t ph = 0; ph < pooled_height; ++ph) {
                        for (int64_t pw = 0; pw < pooled_width; ++pw) {
                            const acc_t grad_val = static_cast<acc_t>(
                                grad_c[ph * pooled_width + pw]) / count;
                            for (int64_t iy = 0; iy < grid.grid_h; ++iy) {
                                const acc_t y = roi_start_h + ph * bin_size_h +
                                    (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid.grid_h;
                                for (int64_t ix = 0; ix < grid.grid_w; ++ix) {
                                    const acc_t x = roi_start_w + pw * bin_size_w +
                                        (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid.grid_w;
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

Tensor roi_align_cpu(const Tensor& input, const Tensor& rois, double spatial_scale,
                     int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio,
                     bool aligned) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "roi_align: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_align: rois must be of shape (R, 5)");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "roi_align: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32: return roi_align_cpu_impl<float, float>(
            ic, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned);
        case DType::Float64: return roi_align_cpu_impl<double, double>(
            ic, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
        case DType::Float16: return roi_align_cpu_impl<Half, float>(
            ic, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned);
        default: TP_THROW(TypeError, "roi_align: unsupported dtype");
    }
}

Tensor roi_align_backward_cpu(const Tensor& grad_output, const Tensor& rois,
                              double spatial_scale, int64_t pooled_height,
                              int64_t pooled_width, int64_t sampling_ratio,
                              bool aligned, const std::vector<int64_t>& input_size) {
    if (input_size.size() != 4)
        TP_THROW(RuntimeError, "roi_align_backward: input_size must have 4 entries");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_align_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (gc.dtype()) {
        case DType::Float32: return roi_align_backward_cpu_impl<float, float>(
            gc, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        case DType::Float64: return roi_align_backward_cpu_impl<double, double>(
            gc, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        case DType::Float16: return roi_align_backward_cpu_impl<Half, float>(
            gc, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        default: TP_THROW(TypeError, "roi_align_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, RoiAlignKernels) {
    m.impl("roi_align", roi_align_cpu);
    m.impl("roi_align_backward", roi_align_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay