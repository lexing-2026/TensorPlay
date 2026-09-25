// Upsampling CUDA kernels: nearest-neighbour sampling of 2-D planes, including the
// integer-scale, rational-scale and exact (half-pixel) variants.
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

template <typename scalar_t, int ScaleH, int ScaleW>
__global__ void upsample_nearest2d_integer_scale_kernel(
    const scalar_t* __restrict__ idata, scalar_t* __restrict__ odata,
    const int64_t nc, const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2) {
    const int64_t w2 = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * ScaleW;
    const int64_t h2 = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (w2 >= width2 || h2 >= height2) return;
    const int64_t nc_start = static_cast<int64_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    const int64_t nc_stride = static_cast<int64_t>(blockDim.z) * gridDim.z;
    const int64_t h1 = h2 / ScaleH;
    const int64_t w1 = w2 / ScaleW;
    for (int64_t n_c = nc_start; n_c < nc; n_c += nc_stride) {
        const scalar_t value = idata[(n_c * height1 + h1) * width1 + w1];
#pragma unroll
        for (int k = 0; k < ScaleW; ++k) {
            odata[(n_c * height2 + h2) * width2 + w2 + k] = value;
        }
    }
}

template <typename accscalar_t, typename scalar_t, int ScaleH, int ScaleW>
__global__ void upsample_nearest2d_integer_scale_backward_out_frame(
    const scalar_t* __restrict__ grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_h, const int64_t dst_dim_w,
    scalar_t* __restrict__ grad_i) {
    const int64_t spatial = src_dim_h * src_dim_w;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dim_b * dim_c * spatial) return;
    const int64_t nc = index / spatial;
    const int64_t rem = index % spatial;
    const int64_t src_h = rem / src_dim_w;
    const int64_t src_w = rem % src_dim_w;
    accscalar_t grad = 0;
    for (int64_t dh = 0; dh < ScaleH; ++dh) {
        for (int64_t dw = 0; dw < ScaleW; ++dw) {
            const int64_t dst_h = src_h * ScaleH + dh;
            const int64_t dst_w = src_w * ScaleW + dw;
            grad += grad_o[(nc * dst_dim_h + dst_h) * dst_dim_w + dst_w];
        }
    }
    grad_i[index] = static_cast<scalar_t>(grad);
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest2d_rational_backward_out_frame(
    const scalar_t* __restrict__ grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_h, const int64_t dst_dim_w,
    scalar_t* __restrict__ grad_i) {
    const int64_t spatial = src_dim_h * src_dim_w;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dim_b * dim_c * spatial) return;
    const int64_t nc = index / spatial;
    const int64_t rem = index % spatial;
    const int64_t src_h = rem / src_dim_w;
    const int64_t src_w = rem % src_dim_w;
    const int64_t dst_h_begin = (src_h * dst_dim_h + src_dim_h - 1) / src_dim_h;
    const int64_t dst_h_end = std::min(
        dst_dim_h, ((src_h + 1) * dst_dim_h + src_dim_h - 1) / src_dim_h);
    const int64_t dst_w_begin = (src_w * dst_dim_w + src_dim_w - 1) / src_dim_w;
    const int64_t dst_w_end = std::min(
        dst_dim_w, ((src_w + 1) * dst_dim_w + src_dim_w - 1) / src_dim_w);
    accscalar_t grad = 0;
    for (int64_t dst_h = dst_h_begin; dst_h < dst_h_end; ++dst_h) {
        for (int64_t dst_w = dst_w_begin; dst_w < dst_w_end; ++dst_w) {
            grad += grad_o[(nc * dst_dim_h + dst_h) * dst_dim_w + dst_w];
        }
    }
    grad_i[index] = static_cast<scalar_t>(grad);
}

template <typename scalar_t>
__global__ void upsample_nearest2d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2,
    float height_scale, float width_scale) {
    const int64_t w2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t h2 = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (w2 >= width2 || h2 >= height2) return;

    const int h1 = height1 == height2
        ? static_cast<int>(h2)
        : nearest_neighbor_compute_source_index(
              height_scale, static_cast<int>(h2), static_cast<int>(height1));
    const int w1 = width1 == width2
        ? static_cast<int>(w2)
        : nearest_neighbor_compute_source_index(
              width_scale, static_cast<int>(w2), static_cast<int>(width1));
    const int64_t nc_start =
        static_cast<int64_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    const int64_t nc_stride = static_cast<int64_t>(blockDim.z) * gridDim.z;
    for (int64_t n_c = nc_start; n_c < nc; n_c += nc_stride) {
        odata[(n_c * height2 + h2) * width2 + w2] =
            idata[(n_c * height1 + h1) * width1 + w1];
    }
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest2d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_h, const int64_t dst_dim_w,
    scalar_t* grad_i, float height_scale, float width_scale) {
    const int64_t dst_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dst_idx >= dim_c * dst_dim_h * dst_dim_w) return;

    const int64_t dst_c_stride = dst_dim_h * dst_dim_w;
    const int64_t src_c_stride = src_dim_h * src_dim_w;
    const int64_t c = (dst_idx / dst_c_stride) % dim_c;
    const int dst_y = static_cast<int>((dst_idx / dst_dim_w) % dst_dim_h);
    // note that we do not want to clamp src_y to src_dim_y, since we might
    // intentionally want to skip in case of scale_factor < 1.0
    const int src_y = nearest_neighbor_bw_compute_source_index(height_scale, dst_y, static_cast<int>(src_dim_h));
    const int src_y_up = nearest_neighbor_bw_compute_source_index(height_scale, dst_y + 1, static_cast<int>(src_dim_h));
    const int dst_x = static_cast<int>(dst_idx % dst_dim_w);
    const int src_x = nearest_neighbor_bw_compute_source_index(width_scale, dst_x, static_cast<int>(src_dim_w));
    const int src_x_up = nearest_neighbor_bw_compute_source_index(width_scale, dst_x + 1, static_cast<int>(src_dim_w));

    for (int64_t b = 0; b < dim_b; ++b) {
        accscalar_t grad = 0;
        for (int y = src_y; y < src_y_up; ++y) {
            for (int x = src_x; x < src_x_up; ++x) {
                grad += grad_o[b * dim_c * src_c_stride + c * src_c_stride + y * src_dim_w + x];
            }
        }
        grad_i[dst_idx + b * dim_c * dst_c_stride] = static_cast<scalar_t>(grad);
    }
}

Tensor upsample_nearest2d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H1 = in.size(2), W1 = in.size(3);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (in.numel() == 0 || H2 == 0 || W2 == 0) return result;
    if (H1 == H2 && W1 == W2) {
        result.copy_(in);
        return result;
    }
    UP_NEAREST_DISPATCH(in, {
        const int64_t scale_h = H2 / H1;
        const int64_t scale_w = W2 / W1;
        if (H2 % H1 == 0 && W2 % W1 == 0 &&
            integer_scale_matches(scales_h, H1, H2) &&
            integer_scale_matches(scales_w, W1, W2) &&
            scale_h >= 2 && scale_h <= 4 && scale_w >= 2 && scale_w <= 4) {
            auto launch_integer = [&]<int ScaleH, int ScaleW>() {
                const unsigned block_x = static_cast<unsigned>(std::min<int64_t>(
                    32, std::max<int64_t>(1, W2 / ScaleW)));
                const unsigned block_y = static_cast<unsigned>(std::min<int64_t>(
                    8, std::max<int64_t>(1, H2)));
                const unsigned grid_x = static_cast<unsigned>(
                    (W2 / ScaleW + block_x - 1) / block_x);
                const unsigned grid_y = static_cast<unsigned>(
                    (H2 + block_y - 1) / block_y);
                const unsigned grid_z = static_cast<unsigned>(std::min<int64_t>(
                    65535, N * C));
                dim3 block(block_x, block_y, 1);
                dim3 grid(grid_x, grid_y, grid_z);
                upsample_nearest2d_integer_scale_kernel<scalar_t, ScaleH, ScaleW>
                    <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                        in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
                        N * C, H1, W1, H2, W2);
            };
            bool launched = true;
            switch (scale_h * 10 + scale_w) {
                case 22: launch_integer.template operator()<2, 2>(); break;
                case 23: launch_integer.template operator()<2, 3>(); break;
                case 24: launch_integer.template operator()<2, 4>(); break;
                case 32: launch_integer.template operator()<3, 2>(); break;
                case 33: launch_integer.template operator()<3, 3>(); break;
                case 34: launch_integer.template operator()<3, 4>(); break;
                case 42: launch_integer.template operator()<4, 2>(); break;
                case 43: launch_integer.template operator()<4, 3>(); break;
                case 44: launch_integer.template operator()<4, 4>(); break;
                default: launched = false; break;
            }
            if (launched) {
                CUDA_CHECK(cudaGetLastError());
                return result;
            }
        }
        const unsigned block_x = static_cast<unsigned>(
            std::min<int64_t>(32, std::max<int64_t>(1, W2)));
        const unsigned block_y = static_cast<unsigned>(
            std::min<int64_t>(8, std::max<int64_t>(1, H2)));
        const unsigned grid_x = static_cast<unsigned>((W2 + block_x - 1) / block_x);
        const unsigned grid_y = static_cast<unsigned>((H2 + block_y - 1) / block_y);
        const unsigned grid_z = static_cast<unsigned>(std::min<int64_t>(
            65535, N * C));
        dim3 block(block_x, block_y, 1);
        dim3 grid(grid_x, grid_y, grid_z);
        upsample_nearest2d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, H1, W1, H2, W2,
            compute_scales_value_h<accscalar_t>(scales_h, H1, H2), compute_scales_value_h<accscalar_t>(scales_w, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest2d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    const int64_t H1 = input_size[2], W1 = input_size[3];
    if (go.numel() == 0 || H2 * W2 == 0 || H1 * W1 == 0) return grad_input;
    UP_NEAREST_DISPATCH(go, {
        const int64_t scale_h = H2 / H1;
        const int64_t scale_w = W2 / W1;
        if (H2 % H1 == 0 && W2 % W1 == 0 &&
            integer_scale_matches(scales_h, H1, H2) &&
            integer_scale_matches(scales_w, W1, W2) &&
            scale_h >= 2 && scale_h <= 4 && scale_w >= 2 && scale_w <= 4) {
            auto launch_integer = [&]<int ScaleH, int ScaleW>() {
                dim3 block, grid;
                launch_dims(dim_b * dim_c * H1 * W1, block, grid);
                upsample_nearest2d_integer_scale_backward_out_frame<accscalar_t, scalar_t, ScaleH, ScaleW>
                    <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                        go.data_ptr<scalar_t>(), dim_b, dim_c, H1, W1, H2, W2,
                        grad_input.data_ptr<scalar_t>());
            };
            bool launched = true;
            switch (scale_h * 10 + scale_w) {
                case 22: launch_integer.template operator()<2, 2>(); break;
                case 23: launch_integer.template operator()<2, 3>(); break;
                case 24: launch_integer.template operator()<2, 4>(); break;
                case 32: launch_integer.template operator()<3, 2>(); break;
                case 33: launch_integer.template operator()<3, 3>(); break;
                case 34: launch_integer.template operator()<3, 4>(); break;
                case 42: launch_integer.template operator()<4, 2>(); break;
                case 43: launch_integer.template operator()<4, 3>(); break;
                case 44: launch_integer.template operator()<4, 4>(); break;
                default: launched = false; break;
            }
            if (launched) {
                CUDA_CHECK(cudaGetLastError());
                return grad_input;
            }
        }
        if (H2 >= H1 && W2 >= W1 &&
            integer_scale_matches(scales_h, H1, H2) &&
            integer_scale_matches(scales_w, W1, W2) &&
            (H2 > H1 || W2 > W1)) {
            dim3 block, grid;
            launch_dims(dim_b * dim_c * H1 * W1, block, grid);
            upsample_nearest2d_rational_backward_out_frame<accscalar_t, scalar_t>
                <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                    go.data_ptr<scalar_t>(), dim_b, dim_c, H1, W1, H2, W2,
                    grad_input.data_ptr<scalar_t>());
            CUDA_CHECK(cudaGetLastError());
            return grad_input;
        }
        dim3 block, grid;
        launch_dims(dim_c * H1 * W1, block, grid);
        upsample_nearest2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, H2, W2, H1, W1,
            grad_input.data_ptr<scalar_t>(),
            compute_scales_value_backwards_h<accscalar_t>(scales_h, H2, H1), compute_scales_value_backwards_h<accscalar_t>(scales_w, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

template <typename scalar_t>
__global__ void upsample_nearest_exact2d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2,
    const float height_scale, const float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * height2 * width2) return;
    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t c = index / (width2 * height2);
    const int64_t h1 = nearest_exact_source_index_h(height_scale, h2, height1);
    const int64_t w1 = nearest_exact_source_index_h(width_scale, w2, width1);
    odata[index] = idata[(c * height1 + h1) * width1 + w1];
}

template <typename scalar_t>
__global__ void upsample_nearest_exact2d_backward_out_frame(
    const int64_t o_numel, const float height_scale, const float width_scale,
    scalar_t* idata, const scalar_t* odata,
    const int64_t nc, const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * height2 * width2) return;
    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t c = index / (width2 * height2);
    const int64_t h1 = nearest_exact_source_index_h(height_scale, h2, height1);
    const int64_t w1 = nearest_exact_source_index_h(width_scale, w2, width1);
    atomicAdd(idata + (c * height1 + h1) * width1 + w1, static_cast<scalar_t>(odata[index]));
}

Tensor _upsample_nearest_exact2d_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                      std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H1 = in.size(2), W1 = in.size(3);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (in.numel() == 0 || H2 == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * H2 * W2, block, grid);
        upsample_nearest_exact2d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, H1, W1, H2, W2,
            nearest_exact_scale_h(H1, H2, scales_h), nearest_exact_scale_h(W1, W2, scales_w));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor _upsample_nearest_exact2d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                               const std::vector<int64_t>& input_size, std::optional<double> scales_h,
                                               std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t N = go.size(0), C = go.size(1);
    const int64_t H1 = input_size[2], W1 = input_size[3];
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (go.numel() == 0 || H1 * W1 == 0 || H2 * W2 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(N * C * H2 * W2, block, grid);
        upsample_nearest_exact2d_backward_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * H2 * W2, nearest_exact_scale_h(H1, H2, scales_h), nearest_exact_scale_h(W1, W2, scales_w),
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(), N * C, H1, W1, H2, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor& upsample_nearest_exact2d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                          std::optional<double> scales_h, std::optional<double> scales_w,
                                          Tensor& out) {
    write_out(out, _upsample_nearest_exact2d_cuda(self, std::move(output_size), scales_h, scales_w));
    return out;
}

Tensor& upsample_nearest_exact2d_backward_grad_input_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                                          const std::vector<int64_t>& input_size, std::optional<double> scales_h,
                                                          std::optional<double> scales_w, Tensor& grad_input) {
    write_out(grad_input, _upsample_nearest_exact2d_backward_cuda(grad_output, std::move(output_size),
                                                         std::move(input_size), scales_h, scales_w));
    return grad_input;
}

Tensor& upsample_nearest2d_out_cuda(const Tensor& self,
                                    const std::vector<int64_t>& output_size,
                                    std::optional<double> scales_h,
                                    std::optional<double> scales_w,
                                    Tensor& out) {
    write_out(out, upsample_nearest2d_cuda(self, std::move(output_size), scales_h,
                                  scales_w));
    return out;
}

Tensor& upsample_nearest2d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, std::optional<double> scales_h,
    std::optional<double> scales_w, Tensor& grad_input) {
    write_out(grad_input, upsample_nearest2d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size), scales_h,
        scales_w));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleNearest2dKernels) {
    m.impl("upsample_nearest2d", upsample_nearest2d_cuda);
    m.impl("upsample_nearest2d_backward", upsample_nearest2d_backward_cuda);
    m.impl("_upsample_nearest_exact2d", _upsample_nearest_exact2d_cuda);
    m.impl("_upsample_nearest_exact2d.out", upsample_nearest_exact2d_out_cuda);
    m.impl("_upsample_nearest_exact2d_backward", _upsample_nearest_exact2d_backward_cuda);
    m.impl("_upsample_nearest_exact2d_backward.grad_input", upsample_nearest_exact2d_backward_grad_input_cuda);
    m.impl("upsample_nearest2d.out", upsample_nearest2d_out_cuda);
    m.impl("upsample_nearest2d_backward.grad_input", upsample_nearest2d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
