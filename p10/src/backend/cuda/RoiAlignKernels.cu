// roi_align / roi_align_backward CUDA kernels.
//
// Same sampling contract as the CPU kernels: bilinear samples averaged over a
// per-bin grid, half-pixel offset with aligned=true, out-of-range samples
// contribute zero. The backward scatters through atomics because many bins
// can land on the same input pixel.
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
__device__ inline T bilinear_sample_cuda(const T* data, int64_t height, int64_t width, CT y, CT x) {
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
__global__ void roi_align_forward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ output,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const acc_t spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t sampling_ratio, const bool aligned) {
    const int64_t out_per_roi = C * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % C;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const int64_t batch = static_cast<int64_t>(rv[0]);
        if (batch < 0 || batch >= N) {
            output[index] = storage_t(0);
            continue;
        }
        const acc_t half = acc_t(0.5);
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
        int64_t grid_h, grid_w;
        if (sampling_ratio > 0) {
            grid_h = grid_w = sampling_ratio;
        } else {
            grid_h = static_cast<int64_t>(ceil(static_cast<double>(roi_height) / pooled_height));
            grid_w = static_cast<int64_t>(ceil(static_cast<double>(roi_width) / pooled_width));
        }
        const acc_t count = static_cast<acc_t>(
            grid_h * grid_w > 0 ? grid_h * grid_w : 1);

        const storage_t* in_c = input + (batch * C + c) * H * W;
        acc_t val = 0;
        for (int64_t iy = 0; iy < grid_h; ++iy) {
            const acc_t y = roi_start_h + ph * bin_size_h +
                (static_cast<acc_t>(iy) + acc_t(0.5)) * bin_size_h / grid_h;
            for (int64_t ix = 0; ix < grid_w; ++ix) {
                const acc_t x = roi_start_w + pw * bin_size_w +
                    (static_cast<acc_t>(ix) + acc_t(0.5)) * bin_size_w / grid_w;
                val += static_cast<acc_t>(bilinear_sample_cuda(in_c, H, W, y, x));
            }
        }
        output[index] = static_cast<storage_t>(val / count);
    }
}

template <typename storage_t, typename acc_t>
__global__ void roi_align_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ grad_input,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const acc_t spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t sampling_ratio, const bool aligned) {
    const int64_t out_per_roi = C * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % C;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const int64_t batch = static_cast<int64_t>(rv[0]);
        if (batch < 0 || batch >= N) continue;
        const acc_t half = acc_t(0.5);
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
        int64_t grid_h, grid_w;
        if (sampling_ratio > 0) {
            grid_h = grid_w = sampling_ratio;
        } else {
            grid_h = static_cast<int64_t>(ceil(static_cast<double>(roi_height) / pooled_height));
            grid_w = static_cast<int64_t>(ceil(static_cast<double>(roi_width) / pooled_width));
        }
        const acc_t count = static_cast<acc_t>(grid_h * grid_w);
        const acc_t grad_val = static_cast<acc_t>(grad_output[index]) / count;
        storage_t* g_c = grad_input + (batch * C + c) * H * W;
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

template <typename storage_t, typename acc_t>
static Tensor roi_align_cuda_impl(const Tensor& input, const Tensor& rois,
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
    const int64_t nthreads = R * C * pooled_height * pooled_width;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    roi_align_forward_kernel<storage_t, acc_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, input.data_ptr<storage_t>(), rois.data_ptr<storage_t>(),
        output.data_ptr<storage_t>(), N, C, H, W, spatial_scale,
        pooled_height, pooled_width, sampling_ratio, aligned);
    return output;
}

template <typename storage_t, typename acc_t>
static Tensor roi_align_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& rois, acc_t spatial_scale,
        int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio,
        bool aligned, const std::vector<int64_t>& input_size) {
    const int64_t N = input_size[0];
    const int64_t C = input_size[1];
    const int64_t H = input_size[2];
    const int64_t W = input_size[3];
    const int64_t R = rois.size(0);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, grad_output.dtype(), grad_output.device());
    const int64_t nthreads = R * C * pooled_height * pooled_width;
    if (nthreads == 0) return grad_input;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    roi_align_backward_kernel<storage_t, acc_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, grad_output.data_ptr<storage_t>(), rois.data_ptr<storage_t>(),
        grad_input.data_ptr<storage_t>(), N, C, H, W, spatial_scale,
        pooled_height, pooled_width, sampling_ratio, aligned);
    return grad_input;
}

Tensor roi_align_cuda(const Tensor& input, const Tensor& rois, double spatial_scale,
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
        case DType::Float32:
            return roi_align_cuda_impl<float, float>(ic, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned);
        case DType::Float64:
            return roi_align_cuda_impl<double, double>(ic, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
        case DType::Float16:
            return roi_align_cuda_impl<Half, float>(ic, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned);
        default: TP_THROW(TypeError, "roi_align: unsupported dtype");
    }
}

Tensor roi_align_backward_cuda(const Tensor& grad_output, const Tensor& rois,
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
        case DType::Float32:
            return roi_align_backward_cuda_impl<float, float>(gc, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        case DType::Float64:
            return roi_align_backward_cuda_impl<double, double>(gc, rc, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        case DType::Float16:
            return roi_align_backward_cuda_impl<Half, float>(gc, rc, scale, pooled_height, pooled_width, sampling_ratio, aligned, input_size);
        default: TP_THROW(TypeError, "roi_align_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RoiAlignKernels) {
    m.impl("roi_align", roi_align_cuda);
    m.impl("roi_align_backward", roi_align_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay