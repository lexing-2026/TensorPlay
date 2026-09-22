// ps_roi_align / ps_roi_align_backward CUDA kernels.
//
// Position-sensitive ROI align: input channel axis is (out_channels,
// pooled_height, pooled_width) groups; output bin (c, ph, pw) averages
// bilinear samples of input channel (c * pooled_height + ph) * pooled_width +
// pw. ROI coordinates always carry the half-pixel offset. Backward scatters
// through atomics.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Atomic.cuh"
#include "Half.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace tensorplay {
namespace cuda {
namespace {

constexpr int kThreads = 256;

inline int64_t ceil_div_blocks(int64_t n, int64_t threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

template <typename T, typename CT>
__device__ inline T bilinear_sample_ps_cuda(const T* data, int64_t height, int64_t width, CT y, CT x) {
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
__global__ void ps_roi_align_forward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ output,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const acc_t spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t sampling_ratio, const int64_t channels) {
    const int64_t out_per_roi = channels * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % channels;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const int64_t batch = static_cast<int64_t>(rv[0]);
        if (batch < 0 || batch >= N) {
            output[index] = storage_t(0);
            continue;
        }
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
            grid_h = static_cast<int64_t>(ceil(static_cast<double>(roi_height) / pooled_height));
            grid_w = static_cast<int64_t>(ceil(static_cast<double>(roi_width) / pooled_width));
        }
        const acc_t count = static_cast<acc_t>(grid_h * grid_w);
        const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
        const storage_t* in_c = input + (batch * C + c_in) * H * W;
        acc_t val = 0;
        for (int64_t iy = 0; iy < grid_h; ++iy) {
            const acc_t y = roi_start_h + ph * bin_size_h +
                (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid_h;
            for (int64_t ix = 0; ix < grid_w; ++ix) {
                const acc_t x = roi_start_w + pw * bin_size_w +
                    (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid_w;
                val += static_cast<acc_t>(bilinear_sample_ps_cuda(in_c, H, W, y, x));
            }
        }
        output[index] = static_cast<storage_t>(val / count);
    }
}

template <typename storage_t, typename acc_t>
__global__ void ps_roi_align_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ grad_input,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const acc_t spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t sampling_ratio, const int64_t channels) {
    const int64_t out_per_roi = channels * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % channels;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const int64_t batch = static_cast<int64_t>(rv[0]);
        if (batch < 0 || batch >= N) continue;
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
            grid_h = static_cast<int64_t>(ceil(static_cast<double>(roi_height) / pooled_height));
            grid_w = static_cast<int64_t>(ceil(static_cast<double>(roi_width) / pooled_width));
        }
        const acc_t count = static_cast<acc_t>(grid_h * grid_w);
        const acc_t grad_val = static_cast<acc_t>(grad_output[index]) / count;
        const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
        storage_t* g_c = grad_input + (batch * C + c_in) * H * W;
        for (int64_t iy = 0; iy < grid_h; ++iy) {
            const acc_t y = roi_start_h + ph * bin_size_h +
                (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid_h;
            for (int64_t ix = 0; ix < grid_w; ++ix) {
                const acc_t x = roi_start_w + pw * bin_size_w +
                    (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid_w;
                if (y < acc_t(-1) || y > acc_t(H) || x < acc_t(-1) || x > acc_t(W)) {
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
                gpuAtomicAdd(&g_c[y_low * W + x_low], static_cast<storage_t>(hy * hx * grad_val));
                gpuAtomicAdd(&g_c[y_low * W + x_high], static_cast<storage_t>(hy * lx * grad_val));
                gpuAtomicAdd(&g_c[y_high * W + x_low], static_cast<storage_t>(ly * hx * grad_val));
                gpuAtomicAdd(&g_c[y_high * W + x_high], static_cast<storage_t>(ly * lx * grad_val));
            }
        }
    }
}

} // namespace

Tensor ps_roi_align_cuda(const Tensor& input, const Tensor& rois, double spatial_scale,
                         int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "ps_roi_align: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_align: rois must be of shape (R, 5)");
    if (input.size(1) % (pooled_height * pooled_width) != 0)
        TP_THROW(RuntimeError, "ps_roi_align: channels must be divisible by pooled_height * pooled_width");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "ps_roi_align: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const int64_t N = ic.size(0), C = ic.size(1), H = ic.size(2), W = ic.size(3);
    const int64_t R = rc.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor output = Tensor::empty({R, channels, pooled_height, pooled_width}, ic.dtype(), ic.device());
    const int64_t nthreads = R * channels * pooled_height * pooled_width;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    const float scale = static_cast<float>(spatial_scale);
    switch (ic.dtype()) {
        case DType::Float32:
            ps_roi_align_forward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<float>(), rc.data_ptr<float>(), output.data_ptr<float>(),
                N, C, H, W, scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        case DType::Float64:
            ps_roi_align_forward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<double>(), rc.data_ptr<double>(), output.data_ptr<double>(),
                N, C, H, W, spatial_scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        case DType::Float16:
            ps_roi_align_forward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<Half>(), rc.data_ptr<Half>(), output.data_ptr<Half>(),
                N, C, H, W, scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        default: TP_THROW(TypeError, "ps_roi_align: unsupported dtype");
    }
    return output;
}

Tensor ps_roi_align_backward_cuda(const Tensor& grad_output, const Tensor& rois,
                                  double spatial_scale, int64_t pooled_height,
                                  int64_t pooled_width, int64_t sampling_ratio,
                                  const std::vector<int64_t>& input_size) {
    if (input_size.size() != 4)
        TP_THROW(RuntimeError, "ps_roi_align_backward: input_size must have 4 entries");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_align_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor rc = rois.contiguous();
    const int64_t N = input_size[0], C = input_size[1], H = input_size[2], W = input_size[3];
    const int64_t R = rc.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, gc.dtype(), gc.device());
    const int64_t nthreads = R * channels * pooled_height * pooled_width;
    if (nthreads == 0) return grad_input;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    const float scale = static_cast<float>(spatial_scale);
    switch (gc.dtype()) {
        case DType::Float32:
            ps_roi_align_backward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<float>(), rc.data_ptr<float>(), grad_input.data_ptr<float>(),
                N, C, H, W, scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        case DType::Float64:
            ps_roi_align_backward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<double>(), rc.data_ptr<double>(), grad_input.data_ptr<double>(),
                N, C, H, W, spatial_scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        case DType::Float16:
            ps_roi_align_backward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<Half>(), rc.data_ptr<Half>(), grad_input.data_ptr<Half>(),
                N, C, H, W, scale, pooled_height, pooled_width, sampling_ratio, channels);
            break;
        default: TP_THROW(TypeError, "ps_roi_align_backward: unsupported dtype");
    }
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PsRoiAlignKernels) {
    m.impl("ps_roi_align", ps_roi_align_cuda);
    m.impl("ps_roi_align_backward", ps_roi_align_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay