// Upsampling CUDA kernels: trilinear interpolation of 3-D volumes.
//
// Kernels operate on contiguous NCT(D)HW tensors.  The linear/bicubic
// backwards accumulate with atomicAdd, so their reduction order (and
// therefore bit-exact results) is nondeterministic across runs.

#include "UpsampleCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Utils.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Atomic.cuh"
#include "OutWrite.h"
#include <cmath>

namespace tensorplay {
namespace cuda {
namespace {

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_trilinear3d_out_frame(
    const int64_t num_kernels,
    const accscalar_t rdepth, const accscalar_t rheight, const accscalar_t rwidth,
    const bool align_corners,
    const scalar_t* idata, scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;

    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t t2 = index / (height2 * width2);

    const accscalar_t t1r = area_pixel_compute_source_index(rdepth, t2, align_corners, false);
    const int64_t t1 = static_cast<int64_t>(t1r);
    const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - static_cast<accscalar_t>(t1);
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;

    const accscalar_t h1r = area_pixel_compute_source_index(rheight, h2, align_corners, false);
    const int64_t h1 = static_cast<int64_t>(h1r);
    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, false);
    const int64_t w1 = static_cast<int64_t>(w1r);
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    for (int64_t n = 0; n < batchsize; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const scalar_t* iptr = idata + (n * channels + c) * depth1 * height1 * width1;
            const accscalar_t val = t0lambda *
                    (h0lambda *
                         (w0lambda * iptr[(t1 * height1 + h1) * width1 + w1] +
                          w1lambda * iptr[(t1 * height1 + h1) * width1 + w1 + w1p]) +
                     h1lambda *
                         (w0lambda * iptr[(t1 * height1 + h1 + h1p) * width1 + w1] +
                          w1lambda * iptr[(t1 * height1 + h1 + h1p) * width1 + w1 + w1p])) +
                t1lambda *
                    (h0lambda *
                         (w0lambda * iptr[((t1 + t1p) * height1 + h1) * width1 + w1] +
                          w1lambda * iptr[((t1 + t1p) * height1 + h1) * width1 + w1 + w1p]) +
                     h1lambda *
                         (w0lambda * iptr[((t1 + t1p) * height1 + h1 + h1p) * width1 + w1] +
                          w1lambda * iptr[((t1 + t1p) * height1 + h1 + h1p) * width1 + w1 + w1p]));
            odata[((n * channels + c) * depth2 + t2) * (height2 * width2) + h2 * width2 + w2] =
                static_cast<scalar_t>(val);
        }
    }
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_trilinear3d_backward_out_frame(
    const int64_t o_numel,
    const accscalar_t rdepth, const accscalar_t rheight, const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2) {
    // eight-corner scatter with atomicAdd.
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= o_numel) return;

    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t t2 = index / (height2 * width2);
    const int64_t n_c = index / (depth2 * height2 * width2);

    const accscalar_t t1r = area_pixel_compute_source_index(rdepth, t2, align_corners, false);
    const int64_t t1 = static_cast<int64_t>(t1r);
    const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - static_cast<accscalar_t>(t1);
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;

    const accscalar_t h1r = area_pixel_compute_source_index(rheight, h2, align_corners, false);
    const int64_t h1 = static_cast<int64_t>(h1r);
    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, false);
    const int64_t w1 = static_cast<int64_t>(w1r);
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    const accscalar_t val = odata[((n_c)*depth2 + t2) * (height2 * width2) + h2 * width2 + w2];
    scalar_t* base = idata + n_c * depth1 * height1 * width1;
    for (int dk = 0; dk < 2; ++dk) {
        const accscalar_t dt = dk == 0 ? t0lambda : t1lambda;
        const int64_t tt = t1 + dk * t1p;
        for (int hk = 0; hk < 2; ++hk) {
            const accscalar_t dh = hk == 0 ? h0lambda : h1lambda;
            const int64_t hh = h1 + hk * h1p;
            for (int wk = 0; wk < 2; ++wk) {
                const accscalar_t dw = wk == 0 ? w0lambda : w1lambda;
                const int64_t ww = w1 + wk * w1p;
                atomicAdd(base + (tt * height1 + hh) * width1 + ww,
                          static_cast<scalar_t>(dt * dh * dw * val));
            }
        }
    }
}

Tensor upsample_trilinear3d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, bool align_corners, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t depth1 = in.size(2), height1 = in.size(3), width1 = in.size(4);
    const int64_t depth2 = output_size[0], height2 = output_size[1], width2 = output_size[2];
    if (in.numel() == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(depth2 * height2 * width2, block, grid);
        upsample_trilinear3d_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            depth2 * height2 * width2,
            area_pixel_compute_scale_h<accscalar_t>(depth1, depth2, align_corners, scales_d),
            area_pixel_compute_scale_h<accscalar_t>(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(width1, width2, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, depth1, height1, width1, depth2, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_trilinear3d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, bool align_corners, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_trilinear3d_backward_cuda");
    // trilinear scatter with atomicAdd; implemented as two chained bilinear
    // scatters per depth slice pair (identical weight decomposition).
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t depth1 = input_size[2], height1 = input_size[3], width1 = input_size[4];
    const int64_t depth2 = output_size[0], height2 = output_size[1], width2 = output_size[2];
    if (go.numel() == 0 || depth1 * height1 * width1 == 0 || depth2 * height2 * width2 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(depth2 * height2 * width2, block, grid);
        upsample_trilinear3d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            depth2 * height2 * width2,
            area_pixel_compute_scale_h<accscalar_t>(depth1, depth2, align_corners, scales_d),
            area_pixel_compute_scale_h<accscalar_t>(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(width1, width2, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, depth1, height1, width1, depth2, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor& upsample_trilinear3d_out_cuda(const Tensor& self,
                                      const std::vector<int64_t>& output_size,
                                      bool align_corners,
                                      std::optional<double> scales_d,
                                      std::optional<double> scales_h,
                                      std::optional<double> scales_w,
                                      Tensor& out) {
    write_out(out, upsample_trilinear3d_cuda(self, std::move(output_size), align_corners,
                                    scales_d, scales_h, scales_w));
    return out;
}

Tensor& upsample_trilinear3d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, bool align_corners,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w, Tensor& grad_input) {
    write_out(grad_input, upsample_trilinear3d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size),
        align_corners, scales_d, scales_h, scales_w));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleTrilinear3dKernels) {
    m.impl("upsample_trilinear3d", upsample_trilinear3d_cuda);
    m.impl("upsample_trilinear3d_backward", upsample_trilinear3d_backward_cuda);
    m.impl("upsample_trilinear3d.out", upsample_trilinear3d_out_cuda);
    m.impl("upsample_trilinear3d_backward.grad_input", upsample_trilinear3d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
