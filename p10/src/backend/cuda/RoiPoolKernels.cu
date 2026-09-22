// roi_pool / ps_roi_pool CUDA kernels.
//
// roi_pool: max pooling over quantized ROI bins, backward re-runs the forward
// max selection and scatters through atomics.
// ps_roi_pool: average pooling with the position-sensitive input channel map
//   c_in = (c_out * pooled_height + ph) * pooled_width + pw,
// backward spreads the gradient evenly over the bin region.
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

struct RoiBoxCuda {
    int64_t batch;
    int64_t start_h;
    int64_t start_w;
    int64_t width;
    int64_t height;
    double bin_size_h;
    double bin_size_w;
};

template <typename T>
__device__ inline RoiBoxCuda scaled_roi_box_cuda(const T* rv, float spatial_scale,
                                                 int64_t pooled_height, int64_t pooled_width,
                                                 bool first_start) {
    RoiBoxCuda box;
    box.batch = static_cast<int64_t>(rv[0]);
    box.start_w = static_cast<int64_t>(round(rv[1] * spatial_scale));
    box.start_h = static_cast<int64_t>(round(rv[2] * spatial_scale));
    const int64_t end_w = static_cast<int64_t>(round(rv[3] * spatial_scale));
    const int64_t end_h = static_cast<int64_t>(round(rv[4] * spatial_scale));
    if (first_start) {
        box.width = max(end_w - box.start_w + 1, static_cast<int64_t>(1));
        box.height = max(end_h - box.start_h + 1, static_cast<int64_t>(1));
    } else {
        box.width = max(end_w - box.start_w, static_cast<int64_t>(1));
        box.height = max(end_h - box.start_h, static_cast<int64_t>(1));
    }
    box.bin_size_h = static_cast<double>(box.height) / pooled_height;
    box.bin_size_w = static_cast<double>(box.width) / pooled_width;
    return box;
}

__device__ inline int64_t clamp_idx(int64_t v, int64_t limit) {
    return min(max(v, static_cast<int64_t>(0)), limit);
}

template <typename storage_t, typename acc_t>
__global__ void roi_pool_forward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ output,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const float spatial_scale, const int64_t pooled_height, const int64_t pooled_width) {
    const int64_t out_per_roi = C * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % C;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const RoiBoxCuda box = scaled_roi_box_cuda(rv, spatial_scale, pooled_height, pooled_width, true);
        output[index] = storage_t(0);
        if (box.batch < 0 || box.batch >= N) continue;
        const int64_t hstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(ph) * box.bin_size_h)) + box.start_h, H);
        const int64_t hend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(ph + 1) * box.bin_size_h)) + box.start_h, H);
        const int64_t wstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(pw) * box.bin_size_w)) + box.start_w, W);
        const int64_t wend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(pw + 1) * box.bin_size_w)) + box.start_w, W);
        if (hend <= hstart || wend <= wstart) continue;
        const storage_t* in_c = input + (box.batch * C + c) * H * W;
        acc_t maxval = -1.0e30;
        for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
                maxval = max(maxval, static_cast<acc_t>(in_c[h * W + w]));
            }
        }
        output[index] = static_cast<storage_t>(maxval);
    }
}

template <typename storage_t, typename acc_t>
__global__ void roi_pool_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ grad_input,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const float spatial_scale, const int64_t pooled_height, const int64_t pooled_width) {
    const int64_t out_per_roi = C * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % C;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const RoiBoxCuda box = scaled_roi_box_cuda(rv, spatial_scale, pooled_height, pooled_width, true);
        if (box.batch < 0 || box.batch >= N) continue;
        const int64_t hstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(ph) * box.bin_size_h)) + box.start_h, H);
        const int64_t hend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(ph + 1) * box.bin_size_h)) + box.start_h, H);
        const int64_t wstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(pw) * box.bin_size_w)) + box.start_w, W);
        const int64_t wend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(pw + 1) * box.bin_size_w)) + box.start_w, W);
        if (hend <= hstart || wend <= wstart) continue;
        const storage_t* in_c = input + (box.batch * C + c) * H * W;
        acc_t maxval = -1.0e30;
        int64_t max_h = hstart, max_w = wstart;
        for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
                const acc_t v = static_cast<acc_t>(in_c[h * W + w]);
                if (v > maxval) {
                    maxval = v;
                    max_h = h;
                    max_w = w;
                }
            }
        }
        gpuAtomicAdd(&grad_input[(box.batch * C + c) * H * W + max_h * W + max_w],
                     grad_output[index]);
    }
}

template <typename storage_t, typename acc_t>
__global__ void ps_roi_pool_forward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ output,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const float spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t channels) {
    const int64_t out_per_roi = channels * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % channels;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const RoiBoxCuda box = scaled_roi_box_cuda(rv, spatial_scale, pooled_height, pooled_width, false);
        output[index] = storage_t(0);
        if (box.batch < 0 || box.batch >= N) continue;
        const int64_t hstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(ph) * box.bin_size_h)) + box.start_h, H - 1);
        const int64_t hend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(ph + 1) * box.bin_size_h)) + box.start_h, H - 1);
        const int64_t wstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(pw) * box.bin_size_w)) + box.start_w, W - 1);
        const int64_t wend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(pw + 1) * box.bin_size_w)) + box.start_w, W - 1);
        if (hend <= hstart || wend <= wstart) continue;
        const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
        const storage_t* in_c = input + (box.batch * C + c_in) * H * W;
        acc_t sum = 0;
        for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
                sum += static_cast<acc_t>(in_c[h * W + w]);
            }
        }
        const acc_t area = static_cast<acc_t>((hend - hstart) * (wend - wstart));
        output[index] = static_cast<storage_t>(sum / area);
    }
}

template <typename storage_t, typename acc_t>
__global__ void ps_roi_pool_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ rois,
        storage_t* __restrict__ grad_input,
        const int64_t N, const int64_t C, const int64_t H, const int64_t W,
        const float spatial_scale, const int64_t pooled_height, const int64_t pooled_width,
        const int64_t channels) {
    const int64_t out_per_roi = channels * pooled_height * pooled_width;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t pw = index % pooled_width;
        const int64_t ph = (index / pooled_width) % pooled_height;
        const int64_t c = (index / (pooled_width * pooled_height)) % channels;
        const int64_t r = index / out_per_roi;
        const storage_t* rv = rois + r * 5;
        const RoiBoxCuda box = scaled_roi_box_cuda(rv, spatial_scale, pooled_height, pooled_width, false);
        if (box.batch < 0 || box.batch >= N) continue;
        const int64_t hstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(ph) * box.bin_size_h)) + box.start_h, H);
        const int64_t hend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(ph + 1) * box.bin_size_h)) + box.start_h, H);
        const int64_t wstart = clamp_idx(
            static_cast<int64_t>(floor(static_cast<double>(pw) * box.bin_size_w)) + box.start_w, W);
        const int64_t wend = clamp_idx(
            static_cast<int64_t>(ceil(static_cast<double>(pw + 1) * box.bin_size_w)) + box.start_w, W);
        if (hend <= hstart || wend <= wstart) continue;
        const int64_t c_in = (c * pooled_height + ph) * pooled_width + pw;
        storage_t* g_c = grad_input + (box.batch * C + c_in) * H * W;
        const acc_t area = static_cast<acc_t>((hend - hstart) * (wend - wstart));
        const acc_t grad_val = static_cast<acc_t>(grad_output[index]) / area;
        for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
                gpuAtomicAdd(&g_c[h * W + w], static_cast<storage_t>(grad_val));
            }
        }
    }
}

} // namespace

Tensor roi_pool_cuda(const Tensor& input, const Tensor& rois, double spatial_scale,
                     int64_t pooled_height, int64_t pooled_width) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "roi_pool: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_pool: rois must be of shape (R, 5)");
    if (input.dtype() != rois.dtype())
        TP_THROW(RuntimeError, "roi_pool: input and rois must have the same dtype");
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    const int64_t N = ic.size(0), C = ic.size(1), H = ic.size(2), W = ic.size(3);
    const int64_t R = rc.size(0);
    Tensor output = Tensor::empty({R, C, pooled_height, pooled_width}, ic.dtype(), ic.device());
    const int64_t nthreads = R * C * pooled_height * pooled_width;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    switch (ic.dtype()) {
        case DType::Float32:
            roi_pool_forward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<float>(), rc.data_ptr<float>(), output.data_ptr<float>(),
                N, C, H, W, scale, pooled_height, pooled_width);
            break;
        case DType::Float64:
            roi_pool_forward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<double>(), rc.data_ptr<double>(), output.data_ptr<double>(),
                N, C, H, W, scale, pooled_height, pooled_width);
            break;
        case DType::Float16:
            roi_pool_forward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<Half>(), rc.data_ptr<Half>(), output.data_ptr<Half>(),
                N, C, H, W, scale, pooled_height, pooled_width);
            break;
        default: TP_THROW(TypeError, "roi_pool: unsupported dtype");
    }
    return output;
}

Tensor roi_pool_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& rois, double spatial_scale,
                              int64_t pooled_height, int64_t pooled_width) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "roi_pool_backward: expected 4-D input");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "roi_pool_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    const int64_t N = ic.size(0), C = ic.size(1), H = ic.size(2), W = ic.size(3);
    const int64_t R = rc.size(0);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, ic.dtype(), ic.device());
    const int64_t nthreads = R * C * pooled_height * pooled_width;
    if (nthreads == 0) return grad_input;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    switch (ic.dtype()) {
        case DType::Float32:
            roi_pool_backward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<float>(), ic.data_ptr<float>(), rc.data_ptr<float>(),
                grad_input.data_ptr<float>(), N, C, H, W, scale, pooled_height, pooled_width);
            break;
        case DType::Float64:
            roi_pool_backward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<double>(), ic.data_ptr<double>(), rc.data_ptr<double>(),
                grad_input.data_ptr<double>(), N, C, H, W, scale, pooled_height, pooled_width);
            break;
        case DType::Float16:
            roi_pool_backward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<Half>(), ic.data_ptr<Half>(), rc.data_ptr<Half>(),
                grad_input.data_ptr<Half>(), N, C, H, W, scale, pooled_height, pooled_width);
            break;
        default: TP_THROW(TypeError, "roi_pool_backward: unsupported dtype");
    }
    return grad_input;
}

Tensor ps_roi_pool_cuda(const Tensor& input, const Tensor& rois, double spatial_scale,
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
    const int64_t N = ic.size(0), C = ic.size(1), H = ic.size(2), W = ic.size(3);
    const int64_t R = rc.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor output = Tensor::empty({R, channels, pooled_height, pooled_width}, ic.dtype(), ic.device());
    const int64_t nthreads = R * channels * pooled_height * pooled_width;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    switch (ic.dtype()) {
        case DType::Float32:
            ps_roi_pool_forward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<float>(), rc.data_ptr<float>(), output.data_ptr<float>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        case DType::Float64:
            ps_roi_pool_forward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<double>(), rc.data_ptr<double>(), output.data_ptr<double>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        case DType::Float16:
            ps_roi_pool_forward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, ic.data_ptr<Half>(), rc.data_ptr<Half>(), output.data_ptr<Half>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        default: TP_THROW(TypeError, "ps_roi_pool: unsupported dtype");
    }
    return output;
}

Tensor ps_roi_pool_backward_cuda(const Tensor& grad_output, const Tensor& rois,
                                 double spatial_scale, int64_t pooled_height,
                                 int64_t pooled_width, const std::vector<int64_t>& input_size) {
    if (input_size.size() != 4)
        TP_THROW(RuntimeError, "ps_roi_pool_backward: input_size must have 4 entries");
    if (rois.dim() != 2 || rois.size(1) != 5)
        TP_THROW(RuntimeError, "ps_roi_pool_backward: rois must be of shape (R, 5)");
    const Tensor gc = grad_output.contiguous();
    const Tensor rc = rois.contiguous();
    const float scale = static_cast<float>(spatial_scale);
    const int64_t N = input_size[0], C = input_size[1], H = input_size[2], W = input_size[3];
    const int64_t R = rc.size(0);
    const int64_t channels = C / (pooled_height * pooled_width);
    Tensor grad_input = Tensor::zeros({N, C, H, W}, gc.dtype(), gc.device());
    const int64_t nthreads = R * channels * pooled_height * pooled_width;
    if (nthreads == 0) return grad_input;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(nthreads, kThreads));
    switch (gc.dtype()) {
        case DType::Float32:
            ps_roi_pool_backward_kernel<float, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<float>(), rc.data_ptr<float>(), grad_input.data_ptr<float>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        case DType::Float64:
            ps_roi_pool_backward_kernel<double, double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<double>(), rc.data_ptr<double>(), grad_input.data_ptr<double>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        case DType::Float16:
            ps_roi_pool_backward_kernel<Half, float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                nthreads, gc.data_ptr<Half>(), rc.data_ptr<Half>(), grad_input.data_ptr<Half>(),
                N, C, H, W, scale, pooled_height, pooled_width, channels);
            break;
        default: TP_THROW(TypeError, "ps_roi_pool_backward: unsupported dtype");
    }
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RoiPoolKernels) {
    m.impl("roi_pool", roi_pool_cuda);
    m.impl("roi_pool_backward", roi_pool_backward_cuda);
    m.impl("ps_roi_pool", ps_roi_pool_cuda);
    m.impl("ps_roi_pool_backward", ps_roi_pool_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay