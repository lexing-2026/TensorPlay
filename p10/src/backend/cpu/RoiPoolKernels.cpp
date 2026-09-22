// roi_pool / ps_roi_pool CPU kernels.
//
// roi_pool is max pooling over quantized (integer-rounded) ROI bins; the
// backward re-runs the forward max selection to locate the argmax and scatters
// the gradient there.
//
// ps_roi_pool is average pooling over the same bins, with the input channel
// axis laid out as (out_channels, pooled_height, pooled_width) groups:
//   c_in = (c_out * pooled_height + ph) * pooled_width + pw.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

struct PoolBin {
    int64_t hstart;
    int64_t hend;
    int64_t wstart;
    int64_t wend;
};

inline int64_t clamp_to(int64_t v, int64_t limit) {
    return std::min(std::max(v, int64_t(0)), limit);
}

// Bins for roi_pool: floor/ceil of the scaled bin positions, shifted by the
// rounded ROI start and clamped to [0, size] (inclusive).
inline void fill_roi_pool_bin(PoolBin& bin, int64_t ph, int64_t pw,
                              int64_t roi_start_h, int64_t roi_start_w,
                              int64_t height, int64_t width,
                              double bin_size_h, double bin_size_w) {
    bin.hstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(ph) * bin_size_h)) + roi_start_h, height);
    bin.hend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(ph + 1) * bin_size_h)) + roi_start_h, height);
    bin.wstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(pw) * bin_size_w)) + roi_start_w, width);
    bin.wend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(pw + 1) * bin_size_w)) + roi_start_w, width);
}

// Bins for ps_roi_pool: same floor/ceil geometry, but clamped to [0, size-1]
// so the pooling window never extends past the last pixel.
inline void fill_ps_pool_bin(PoolBin& bin, int64_t ph, int64_t pw,
                             int64_t roi_start_h, int64_t roi_start_w,
                             int64_t height, int64_t width,
                             double bin_size_h, double bin_size_w) {
    bin.hstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(ph) * bin_size_h)) + roi_start_h, height - 1);
    bin.hend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(ph + 1) * bin_size_h)) + roi_start_h, height - 1);
    bin.wstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(pw) * bin_size_w)) + roi_start_w, width - 1);
    bin.wend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(pw + 1) * bin_size_w)) + roi_start_w, width - 1);
}

struct RoiBox {
    int64_t batch;
    int64_t start_h;
    int64_t start_w;
    int64_t width;
    int64_t height;
    double bin_size_h;
    double bin_size_w;
};

template <typename storage_t>
RoiBox scaled_roi_box(const storage_t* rv, double spatial_scale,
                      int64_t pooled_height, int64_t pooled_width,
                      bool first_start) {
    RoiBox box;
    box.batch = static_cast<int64_t>(rv[0]);
    box.start_w = static_cast<int64_t>(std::round(rv[1] * spatial_scale));
    box.start_h = static_cast<int64_t>(std::round(rv[2] * spatial_scale));
    const int64_t end_w = static_cast<int64_t>(std::round(rv[3] * spatial_scale));
    const int64_t end_h = static_cast<int64_t>(std::round(rv[4] * spatial_scale));
    if (first_start) {
        box.width = std::max(end_w - box.start_w + 1, int64_t(1));
        box.height = std::max(end_h - box.start_h + 1, int64_t(1));
    } else {
        box.width = std::max(end_w - box.start_w, int64_t(1));
        box.height = std::max(end_h - box.start_h, int64_t(1));
    }
    box.bin_size_h = static_cast<double>(box.height) / pooled_height;
    box.bin_size_w = static_cast<double>(box.width) / pooled_width;
    return box;
}

template <typename storage_t, typename acc_t>
static Tensor roi_pool_cpu_impl(const Tensor& input, const Tensor& rois,
                                acc_t spatial_scale, int64_t pooled_height,
                                int64_t pooled_width) {
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

    parallel_for(0, R, 1, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; ++r) {
            const storage_t* rv = rois_ptr + r * 5;
            const RoiBox box = scaled_roi_box(rv, spatial_scale, pooled_height, pooled_width, true);
            if (box.batch < 0 || box.batch >= N) continue;

            const storage_t* in_b = in_ptr + box.batch * C * H * W;
            storage_t* out_r = out_ptr + r * C * pooled_height * pooled_width;
            for (int64_t c = 0; c < C; ++c) {
                const storage_t* in_c = in_b + c * H * W;
                storage_t* out_c = out_r + c * pooled_height * pooled_width;
                for (int64_t ph = 0; ph < pooled_height; ++ph) {
                    for (int64_t pw = 0; pw < pooled_width; ++pw) {
                        PoolBin bin;
                        fill_roi_pool_bin(bin, ph, pw, box.start_h, box.start_w, H, W,
                                          box.bin_size_h, box.bin_size_w);
                        acc_t maxval = -std::numeric_limits<acc_t>::infinity();
                        for (int64_t h = bin.hstart; h < bin.hend; ++h) {
                            for (int64_t w = bin.wstart; w < bin.wend; ++w) {
                                maxval = std::max(maxval, static_cast<acc_t>(in_c[h * W + w]));
                            }
                        }
                        out_c[ph * pooled_width + pw] =
                            (bin.hend > bin.hstart && bin.wend > bin.wstart)
                            ? static_cast<storage_t>(maxval) : storage_t(0);
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename acc_t>
static Tensor roi_pool_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& input, const Tensor& rois,
        acc_t spatial_scale, int64_t pooled_height, int64_t pooled_width) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t R = rois.size(0);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    if (R == 0 || grad_input.numel() == 0) return grad_input;

    const storage_t* grad_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* in_ptr = input.data_ptr<storage_t>();
    const storage_t* rois_ptr = rois.data_ptr<storage_t>();
    storage_t* g_in = grad_input.data_ptr<storage_t>();

    // Batch-parallel: ROIs sharing a batch can scatter into the same pixel.
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            storage_t* g_b = g_in + batch * C * H * W;
            const storage_t* in_b = in_ptr + batch * C * H * W;
            for (int64_t r = 0; r < R; ++r) {
                const storage_t* rv = rois_ptr + r * 5;
                if (static_cast<int64_t>(rv[0]) != batch) continue;
                const RoiBox box = scaled_roi_box(rv, spatial_scale, pooled_height, pooled_width, true);
                const storage_t* grad_r = grad_ptr + r * C * pooled_height * pooled_width;
                for (int64_t c = 0; c < C; ++c) {
                    const storage_t* in_c = in_b + c * H * W;
                    storage_t* g_c = g_b + c * H * W;
                    const storage_t* grad_c = grad_r + c * pooled_height * pooled_width;
                    for (int64_t ph = 0; ph < pooled_height; ++ph) {
                        for (int64_t pw = 0; pw < pooled_width; ++pw) {
                            PoolBin bin;
                            fill_roi_pool_bin(bin, ph, pw, box.start_h, box.start_w, H, W,
                                              box.bin_size_h, box.bin_size_w);
                            if (bin.hend <= bin.hstart || bin.wend <= bin.wstart) continue;
                            acc_t maxval = -std::numeric_limits<acc_t>::infinity();
                            int64_t max_h = bin.hstart, max_w = bin.wstart;
                            for (int64_t h = bin.hstart; h < bin.hend; ++h) {
                                for (int64_t w = bin.wstart; w < bin.wend; ++w) {
                                    const acc_t v = static_cast<acc_t>(in_c[h * W + w]);
                                    if (v > maxval) {
                                        maxval = v;
                                        max_h = h;
                                        max_w = w;
                                    }
                                }
                            }
                            g_c[max_h * W + max_w] += grad_c[ph * pooled_width + pw];
                        }
                    }
                }
            }
        }
    });
    return grad_input;
}

template <typename storage_t, typename acc_t>
static Tensor ps_roi_pool_cpu_impl(const Tensor& input, const Tensor& rois,
                                   acc_t spatial_scale, int64_t pooled_height,
                                   int64_t pooled_width) {
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
            const RoiBox box = scaled_roi_box(rv, spatial_scale, pooled_height, pooled_width, false);
            if (box.batch < 0 || box.batch >= N) continue;

            const storage_t* in_b = in_ptr + box.batch * C * H * W;
            storage_t* out_r = out_ptr + r * channels * pooled_height * pooled_width;
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t ph = 0; ph < pooled_height; ++ph) {
                    for (int64_t pw = 0; pw < pooled_width; ++pw) {
                        PoolBin bin;
                        fill_ps_pool_bin(bin, ph, pw, box.start_h, box.start_w, H, W,
                                         box.bin_size_h, box.bin_size_w);
                        const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
                        const storage_t* in_c = in_b + c_in * H * W;
                        if (bin.hend <= bin.hstart || bin.wend <= bin.wstart) {
                            out_r[(c * pooled_height + ph) * pooled_width + pw] = storage_t(0);
                            continue;
                        }
                        acc_t sum = 0;
                        for (int64_t h = bin.hstart; h < bin.hend; ++h) {
                            for (int64_t w = bin.wstart; w < bin.wend; ++w) {
                                sum += static_cast<acc_t>(in_c[h * W + w]);
                            }
                        }
                        const acc_t area = static_cast<acc_t>((bin.hend - bin.hstart) * (bin.wend - bin.wstart));
                        out_r[(c * pooled_height + ph) * pooled_width + pw] =
                            static_cast<storage_t>(sum / area);
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename acc_t>
static Tensor ps_roi_pool_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& rois, acc_t spatial_scale,
        int64_t pooled_height, int64_t pooled_width,
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

    // Batch-parallel: ROI gradient contributions into one batch's planes
    // accumulate sequentially inside a single thread.
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            storage_t* g_b = g_in + batch * C * H * W;
            for (int64_t r = 0; r < R; ++r) {
                const storage_t* rv = rois_ptr + r * 5;
                if (static_cast<int64_t>(rv[0]) != batch) continue;
                const RoiBox box = scaled_roi_box(rv, spatial_scale, pooled_height, pooled_width, false);
                const storage_t* grad_r = grad_ptr + r * channels * pooled_height * pooled_width;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t ph = 0; ph < pooled_height; ++ph) {
                        for (int64_t pw = 0; pw < pooled_width; ++pw) {
                            // Backward clamps to [0, size] like the forward bin
                            // arithmetic of the max variant.
                            PoolBin bin;
                            bin.hstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(ph) * box.bin_size_h)) + box.start_h, H);
                            bin.hend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(ph + 1) * box.bin_size_h)) + box.start_h, H);
                            bin.wstart = clamp_to(static_cast<int64_t>(std::floor(static_cast<double>(pw) * box.bin_size_w)) + box.start_w, W);
                            bin.wend = clamp_to(static_cast<int64_t>(std::ceil(static_cast<double>(pw + 1) * box.bin_size_w)) + box.start_w, W);
                            if (bin.hend <= bin.hstart || bin.wend <= bin.wstart) continue;
                            const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
                            storage_t* g_c = g_b + c_in * H * W;
                            const acc_t area = static_cast<acc_t>((bin.hend - bin.hstart) * (bin.wend - bin.wstart));
                            const acc_t grad_val = static_cast<acc_t>(grad_r[(c * pooled_height + ph) * pooled_width + pw]) / area;
                            for (int64_t h = bin.hstart; h < bin.hend; ++h) {
                                for (int64_t w = bin.wstart; w < bin.wend; ++w) {
                                    g_c[h * W + w] += static_cast<storage_t>(grad_val);
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

Tensor roi_pool_cpu(const Tensor& input, const Tensor& rois, double spatial_scale,
                    int64_t pooled_height, int64_t pooled_width) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "roi_pool: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_pool: rois must be of shape (R, 5)");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "roi_pool: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32:
            return roi_pool_cpu_impl<float, float>(ic, rc, scale, pooled_height, pooled_width);
        case DType::Float64:
            return roi_pool_cpu_impl<double, double>(ic, rc, spatial_scale, pooled_height, pooled_width);
        case DType::Float16:
            return roi_pool_cpu_impl<Half, float>(ic, rc, scale, pooled_height, pooled_width);
        default: TP_THROW(TypeError, "roi_pool: unsupported dtype");
    }
}

Tensor roi_pool_backward_cpu(const Tensor& grad_output, const Tensor& input,
                             const Tensor& rois, double spatial_scale,
                             int64_t pooled_height, int64_t pooled_width) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "roi_pool_backward: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_pool_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32:
            return roi_pool_backward_cpu_impl<float, float>(gc, ic, rc, scale, pooled_height, pooled_width);
        case DType::Float64:
            return roi_pool_backward_cpu_impl<double, double>(gc, ic, rc, spatial_scale, pooled_height, pooled_width);
        case DType::Float16:
            return roi_pool_backward_cpu_impl<Half, float>(gc, ic, rc, scale, pooled_height, pooled_width);
        default: TP_THROW(TypeError, "roi_pool_backward: unsupported dtype");
    }
}

Tensor ps_roi_pool_cpu(const Tensor& input, const Tensor& rois, double spatial_scale,
                       int64_t pooled_height, int64_t pooled_width) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "ps_roi_pool: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_pool: rois must be of shape (R, 5)");
    if (input.size(1) % (pooled_height * pooled_width) != 0)
        TP_THROW(RuntimeError, "ps_roi_pool: channels must be divisible by pooled_height * pooled_width");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "ps_roi_pool: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32:
            return ps_roi_pool_cpu_impl<float, float>(ic, rc, scale, pooled_height, pooled_width);
        case DType::Float64:
            return ps_roi_pool_cpu_impl<double, double>(ic, rc, spatial_scale, pooled_height, pooled_width);
        case DType::Float16:
            return ps_roi_pool_cpu_impl<Half, float>(ic, rc, scale, pooled_height, pooled_width);
        default: TP_THROW(TypeError, "ps_roi_pool: unsupported dtype");
    }
}

Tensor ps_roi_pool_backward_cpu(const Tensor& grad_output, const Tensor& rois,
                                double spatial_scale, int64_t pooled_height,
                                int64_t pooled_width,
                                const std::vector<int64_t>& input_size) {
    if (input_size.size() != 4)
        TP_THROW(RuntimeError, "ps_roi_pool_backward: input_size must have 4 entries");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_pool_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    switch (gc.dtype()) {
        case DType::Float32:
            return ps_roi_pool_backward_cpu_impl<float, float>(gc, rc, scale, pooled_height, pooled_width, input_size);
        case DType::Float64:
            return ps_roi_pool_backward_cpu_impl<double, double>(gc, rc, spatial_scale, pooled_height, pooled_width, input_size);
        case DType::Float16:
            return ps_roi_pool_backward_cpu_impl<Half, float>(gc, rc, scale, pooled_height, pooled_width, input_size);
        default: TP_THROW(TypeError, "ps_roi_pool_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, RoiPoolKernels) {
    m.impl("roi_pool", roi_pool_cpu);
    m.impl("roi_pool_backward", roi_pool_backward_cpu);
    m.impl("ps_roi_pool", ps_roi_pool_cpu);
    m.impl("ps_roi_pool_backward", ps_roi_pool_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay