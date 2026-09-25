// Upsampling CUDA kernels: nearest-neighbour sampling along one axis, together with the
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
__global__ void upsample_nearest1d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t width1, const int64_t width2, float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * width2) return;
    const int64_t w2 = index % width2;
    const int64_t n_c = index / width2;
    const int w1 = width1 == width2 ? static_cast<int>(w2)
                                    : nearest_neighbor_compute_source_index(width_scale, static_cast<int>(w2), static_cast<int>(width1));
    odata[index] = idata[n_c * width1 + w1];
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest1d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_w, const int64_t dst_dim_w,
    scalar_t* grad_i, float width_scale) {
    const int64_t dst_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dst_idx >= dim_c * dst_dim_w) return;
    const int64_t c = dst_idx / dst_dim_w;
    const int dst_x = static_cast<int>(dst_idx % dst_dim_w);
    // note that we do not want to clamp src_x to src_dim_w, since we might
    // intentionally want to skip in case of scale_factor < 1.0
    const int src_x = nearest_neighbor_bw_compute_source_index(width_scale, dst_x, static_cast<int>(src_dim_w));
    const int src_x_up = nearest_neighbor_bw_compute_source_index(width_scale, dst_x + 1, static_cast<int>(src_dim_w));
    for (int64_t b = 0; b < dim_b; ++b) {
        accscalar_t grad = 0;
        for (int x = src_x; x < src_x_up; ++x) {
            grad += grad_o[b * dim_c * src_dim_w + c * src_dim_w + x];
        }
        grad_i[b * dim_c * dst_dim_w + dst_idx] = static_cast<scalar_t>(grad);
    }
}

Tensor upsample_nearest1d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;
    UP_NEAREST_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * W2, block, grid);
        upsample_nearest1d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, W1, W2,
            compute_scales_value_h<accscalar_t>(scales, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest1d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, std::optional<double> scales) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::empty(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t W2 = output_size[0], W1 = input_size[2];
    if (go.numel() == 0 || W2 == 0 || W1 == 0) return grad_input;
    UP_NEAREST_DISPATCH(go, {
        // zeroed buffer is already scalar_t so accumulate via an accscalar
        // staging is unnecessary for f32/f64 (accscalar_t == scalar_t).
        dim3 block, grid;
        launch_dims(dim_c * W1, block, grid);
        upsample_nearest1d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, W2, W1,
            grad_input.data_ptr<scalar_t>(),
            compute_scales_value_backwards_h<accscalar_t>(scales, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

template <typename scalar_t>
__global__ void upsample_nearest_exact1d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t width1, const int64_t width2, const float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * width2) return;
    const int64_t w2 = index % width2;
    const int64_t c = index / width2;
    odata[index] = idata[c * width1 + nearest_exact_source_index_h(width_scale, w2, width1)];
}

template <typename scalar_t>
__global__ void upsample_nearest_exact1d_backward_out_frame(
    const int64_t o_numel, const float width_scale,
    scalar_t* idata, const scalar_t* odata,
    const int64_t nc, const int64_t width1, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * width2) return;
    const int64_t w2 = index % width2;
    const int64_t c = index / width2;
    atomicAdd(idata + c * width1 + nearest_exact_source_index_h(width_scale, w2, width1),
              static_cast<scalar_t>(odata[index]));
}

Tensor _upsample_nearest_exact1d_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                      std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * W2, block, grid);
        upsample_nearest_exact1d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, W1, W2,
            nearest_exact_scale_h(W1, W2, scales));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor _upsample_nearest_exact1d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                               const std::vector<int64_t>& input_size, std::optional<double> scales) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t N = go.size(0), C = go.size(1);
    const int64_t W1 = input_size[2], W2 = output_size[0];
    if (go.numel() == 0 || W1 == 0 || W2 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(N * C * W2, block, grid);
        upsample_nearest_exact1d_backward_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * W2, nearest_exact_scale_h(W1, W2, scales),
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(), N * C, W1, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor& upsample_nearest_exact1d_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                          std::optional<double> scales, Tensor& out) {
    write_out(out, _upsample_nearest_exact1d_cuda(self, std::move(output_size), scales));
    return out;
}

Tensor& upsample_nearest_exact1d_backward_grad_input_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                                          const std::vector<int64_t>& input_size, std::optional<double> scales,
                                                          Tensor& grad_input) {
    write_out(grad_input, _upsample_nearest_exact1d_backward_cuda(grad_output, std::move(output_size),
                                                         std::move(input_size), scales));
    return grad_input;
}

Tensor& upsample_nearest1d_out_cuda(const Tensor& self,
                                    const std::vector<int64_t>& output_size,
                                    std::optional<double> scales,
                                    Tensor& out) {
    write_out(out, upsample_nearest1d_cuda(self, std::move(output_size), scales));
    return out;
}

Tensor& upsample_nearest1d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, std::optional<double> scales,
    Tensor& grad_input) {
    write_out(grad_input, upsample_nearest1d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size), scales));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleNearest1dKernels) {
    m.impl("upsample_nearest1d", upsample_nearest1d_cuda);
    m.impl("upsample_nearest1d_backward", upsample_nearest1d_backward_cuda);
    m.impl("_upsample_nearest_exact1d", _upsample_nearest_exact1d_cuda);
    m.impl("_upsample_nearest_exact1d.out", upsample_nearest_exact1d_out_cuda);
    m.impl("_upsample_nearest_exact1d_backward", _upsample_nearest_exact1d_backward_cuda);
    m.impl("_upsample_nearest_exact1d_backward.grad_input", upsample_nearest_exact1d_backward_grad_input_cuda);
    m.impl("upsample_nearest1d.out", upsample_nearest1d_out_cuda);
    m.impl("upsample_nearest1d_backward.grad_input", upsample_nearest1d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
