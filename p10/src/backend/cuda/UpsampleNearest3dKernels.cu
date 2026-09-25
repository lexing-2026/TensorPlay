// Upsampling CUDA kernels: nearest-neighbour sampling of 3-D volumes, together with the
// exact (half-pixel) variant of the same family.
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

template <typename scalar_t>
__global__ void upsample_nearest3d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2,
    float depth_scale, float height_scale, float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * depth2 * height2 * width2) return;

    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t d2 = (index / (height2 * width2)) % depth2;
    const int64_t n_c = index / (depth2 * height2 * width2);

    const int d1 = depth1 == depth2 ? static_cast<int>(d2)
                                    : nearest_neighbor_compute_source_index(depth_scale, static_cast<int>(d2), static_cast<int>(depth1));
    const int h1 = height1 == height2 ? static_cast<int>(h2)
                                      : nearest_neighbor_compute_source_index(height_scale, static_cast<int>(h2), static_cast<int>(height1));
    const int w1 = width1 == width2 ? static_cast<int>(w2)
                                    : nearest_neighbor_compute_source_index(width_scale, static_cast<int>(w2), static_cast<int>(width1));

    odata[index] = idata[((n_c * depth1 + d1) * height1 + h1) * width1 + w1];
}

// ===========================================================================
// (gather formulation over input pixels; no atomics needed)
// ===========================================================================

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest3d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_d, const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_d, const int64_t dst_dim_h, const int64_t dst_dim_w,
    scalar_t* grad_i, float depth_scale, float height_scale, float width_scale) {
    const int64_t dst_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w) return;

    const int64_t dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
    const int64_t c = (dst_idx / dst_c_stride) % dim_c;
    const int dst_t = static_cast<int>((dst_idx / (dst_dim_h * dst_dim_w)) % dst_dim_d);
    const int dst_y = static_cast<int>((dst_idx / dst_dim_w) % dst_dim_h);
    const int dst_x = static_cast<int>(dst_idx % dst_dim_w);

    const int src_t = nearest_neighbor_bw_compute_source_index(depth_scale, dst_t, static_cast<int>(src_dim_d));
    const int src_t_up = nearest_neighbor_bw_compute_source_index(depth_scale, dst_t + 1, static_cast<int>(src_dim_d));
    const int src_y = nearest_neighbor_bw_compute_source_index(height_scale, dst_y, static_cast<int>(src_dim_h));
    const int src_y_up = nearest_neighbor_bw_compute_source_index(height_scale, dst_y + 1, static_cast<int>(src_dim_h));
    const int src_x = nearest_neighbor_bw_compute_source_index(width_scale, dst_x, static_cast<int>(src_dim_w));
    const int src_x_up = nearest_neighbor_bw_compute_source_index(width_scale, dst_x + 1, static_cast<int>(src_dim_w));

    for (int64_t b = 0; b < dim_b; ++b) {
        accscalar_t grad = 0;
        for (int t = src_t; t < src_t_up; ++t) {
            for (int y = src_y; y < src_y_up; ++y) {
                for (int x = src_x; x < src_x_up; ++x) {
                    grad += grad_o[((b * dim_c + c) * src_dim_d + t) * (src_dim_h * src_dim_w) + y * src_dim_w + x];
                }
            }
        }
        grad_i[dst_idx + b * dim_c * dst_c_stride] = static_cast<scalar_t>(grad);
    }
}

// ===========================================================================
// UpSampleTrilinear3d.cu *_out_frame
// ===========================================================================

Tensor upsample_nearest3d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t D1 = in.size(2), H1 = in.size(3), W1 = in.size(4);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    if (in.numel() == 0) return result;
    UP_NEAREST_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * D2 * H2 * W2, block, grid);
        upsample_nearest3d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, D1, H1, W1, D2, H2, W2,
            compute_scales_value_h<accscalar_t>(scales_d, D1, D2), compute_scales_value_h<accscalar_t>(scales_h, H1, H2),
            compute_scales_value_h<accscalar_t>(scales_w, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest3d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    const int64_t D1 = input_size[2], H1 = input_size[3], W1 = input_size[4];
    if (go.numel() == 0) return grad_input;
    UP_NEAREST_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(dim_c * D1 * H1 * W1, block, grid);
        upsample_nearest3d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, D2, H2, W2, D1, H1, W1,
            grad_input.data_ptr<scalar_t>(),
            compute_scales_value_backwards_h<accscalar_t>(scales_d, D2, D1), compute_scales_value_backwards_h<accscalar_t>(scales_h, H2, H1),
            compute_scales_value_backwards_h<accscalar_t>(scales_w, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

template <typename scalar_t>
__global__ void upsample_nearest_exact3d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2,
    const float depth_scale, const float height_scale, const float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * depth2 * height2 * width2) return;
    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t d2 = (index / (width2 * height2)) % depth2;
    const int64_t c = index / (width2 * height2 * depth2);
    const int64_t d1 = nearest_exact_source_index_h(depth_scale, d2, depth1);
    const int64_t h1 = nearest_exact_source_index_h(height_scale, h2, height1);
    const int64_t w1 = nearest_exact_source_index_h(width_scale, w2, width1);
    odata[index] = idata[((c * depth1 + d1) * height1 + h1) * width1 + w1];
}

template <typename scalar_t>
__global__ void upsample_nearest_exact3d_backward_out_frame(
    const int64_t o_numel, const float depth_scale, const float height_scale, const float width_scale,
    scalar_t* idata, const scalar_t* odata,
    const int64_t nc, const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * depth2 * height2 * width2) return;
    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t d2 = (index / (width2 * height2)) % depth2;
    const int64_t c = index / (width2 * height2 * depth2);
    const int64_t d1 = nearest_exact_source_index_h(depth_scale, d2, depth1);
    const int64_t h1 = nearest_exact_source_index_h(height_scale, h2, height1);
    const int64_t w1 = nearest_exact_source_index_h(width_scale, w2, width1);
    atomicAdd(idata + ((c * depth1 + d1) * height1 + h1) * width1 + w1,
              static_cast<scalar_t>(odata[index]));
}

// ===========================================================================
// Antialiased 2-D upsampling.  With antialiasing the filter support on the
// source grid stretches with the downscale factor, so each output pixel is a
// normalized filter sum over its source window.  Per-axis (window start,
// window size, normalized weights) tables are built on the host (double
// precision, then cast to the tensor compute type) and uploaded; the forward
// applies them as separable horizontal/vertical passes through a scratch
// plane, the backward scatters with the same weights via atomicAdd
// (nondeterministic order, like the other backward frames here).
// ===========================================================================

Tensor _upsample_nearest_exact3d_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                      std::optional<double> scales_d, std::optional<double> scales_h,
                                      std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t D1 = in.size(2), H1 = in.size(3), W1 = in.size(4);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    if (in.numel() == 0 || D2 == 0 || H2 == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * D2 * H2 * W2, block, grid);
        upsample_nearest_exact3d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, D1, H1, W1, D2, H2, W2,
            nearest_exact_scale_h(D1, D2, scales_d), nearest_exact_scale_h(H1, H2, scales_h),
            nearest_exact_scale_h(W1, W2, scales_w));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor _upsample_nearest_exact3d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                               const std::vector<int64_t>& input_size, std::optional<double> scales_d,
                                               std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t N = go.size(0), C = go.size(1);
    const int64_t D1 = input_size[2], H1 = input_size[3], W1 = input_size[4];
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    if (go.numel() == 0 || D1 * H1 * W1 == 0 || D2 * H2 * W2 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(N * C * D2 * H2 * W2, block, grid);
        upsample_nearest_exact3d_backward_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * D2 * H2 * W2,
            nearest_exact_scale_h(D1, D2, scales_d), nearest_exact_scale_h(H1, H2, scales_h),
            nearest_exact_scale_h(W1, W2, scales_w),
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(), N * C, D1, H1, W1, D2, H2, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor& upsample_nearest_exact3d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                          std::optional<double> scales_d, std::optional<double> scales_h,
                                          std::optional<double> scales_w, Tensor& out) {
    write_out(out, _upsample_nearest_exact3d_cuda(self, std::move(output_size), scales_d, scales_h, scales_w));
    return out;
}

Tensor& upsample_nearest_exact3d_backward_grad_input_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                                          const std::vector<int64_t>& input_size, std::optional<double> scales_d,
                                                          std::optional<double> scales_h, std::optional<double> scales_w,
                                                          Tensor& grad_input) {
    write_out(grad_input, _upsample_nearest_exact3d_backward_cuda(grad_output, std::move(output_size),
                                                         std::move(input_size), scales_d, scales_h, scales_w));
    return grad_input;
}

Tensor& upsample_nearest3d_out_cuda(const Tensor& self,
                                    const std::vector<int64_t>& output_size,
                                    std::optional<double> scales_d,
                                    std::optional<double> scales_h,
                                    std::optional<double> scales_w,
                                    Tensor& out) {
    write_out(out, upsample_nearest3d_cuda(self, std::move(output_size), scales_d,
                                  scales_h, scales_w));
    return out;
}

Tensor& upsample_nearest3d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, std::optional<double> scales_d,
    std::optional<double> scales_h, std::optional<double> scales_w,
    Tensor& grad_input) {
    write_out(grad_input, upsample_nearest3d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size), scales_d,
        scales_h, scales_w));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleNearest3dKernels) {
    m.impl("upsample_nearest3d", upsample_nearest3d_cuda);
    m.impl("upsample_nearest3d_backward", upsample_nearest3d_backward_cuda);
    m.impl("_upsample_nearest_exact3d", _upsample_nearest_exact3d_cuda);
    m.impl("_upsample_nearest_exact3d.out", upsample_nearest_exact3d_out_cuda);
    m.impl("_upsample_nearest_exact3d_backward", _upsample_nearest_exact3d_backward_cuda);
    m.impl("_upsample_nearest_exact3d_backward.grad_input", upsample_nearest_exact3d_backward_grad_input_cuda);
    m.impl("upsample_nearest3d.out", upsample_nearest3d_out_cuda);
    m.impl("upsample_nearest3d_backward.grad_input", upsample_nearest3d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
