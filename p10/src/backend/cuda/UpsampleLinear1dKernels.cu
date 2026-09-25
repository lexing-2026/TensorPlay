// Upsampling CUDA kernels: linear interpolation along one axis.
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
__global__ void upsample_linear1d_out_frame(
    const int64_t num_kernels,
    const accscalar_t rwidth, const bool align_corners,
    const scalar_t* idata, scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t width1, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;
    const int64_t w2 = index;
    const accscalar_t w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, false);
    const int64_t w1 = static_cast<int64_t>(w1r);
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    for (int64_t n = 0; n < batchsize; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const scalar_t* iptr = idata + (n * channels + c) * width1;
            const accscalar_t val =
                w0lambda * iptr[w1] + w1lambda * iptr[w1 + w1p];
            odata[(n * channels + c) * width2 + w2] = static_cast<scalar_t>(val);
        }
    }
}

// ===========================================================================
// ===========================================================================

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_linear1d_backward_out_frame(
    const int64_t o_numel,
    const accscalar_t rwidth, const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t width1, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= o_numel) return;
    const int64_t w2 = index % width2;
    const int64_t n_c = index / width2;

    const accscalar_t w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, false);
    const int64_t w1 = static_cast<int64_t>(w1r);
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    const accscalar_t val = odata[index];
    atomicAdd(&idata[(n_c)*width1 + w1], static_cast<accscalar_t>(w0lambda * val));
    atomicAdd(&idata[(n_c)*width1 + w1 + w1p], static_cast<accscalar_t>(w1lambda * val));
}

Tensor upsample_linear1d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, bool align_corners, std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(W2, block, grid);
        upsample_linear1d_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            W2, area_pixel_compute_scale_h<accscalar_t>(W1, W2, align_corners, scales), align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N, C, W1, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_linear1d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, bool align_corners, std::optional<double> scales) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_linear1d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t W2 = output_size[0], W1 = input_size[2];
    if (go.numel() == 0 || W2 == 0 || W1 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(batchsize * channels * W2, block, grid);
        upsample_linear1d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            batchsize * channels * W2,
            area_pixel_compute_scale_h<accscalar_t>(W1, W2, align_corners, scales), align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, W1, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor& upsample_linear1d_out_cuda(const Tensor& self,
                                   const std::vector<int64_t>& output_size,
                                   bool align_corners,
                                   std::optional<double> scales, Tensor& out) {
    write_out(out, upsample_linear1d_cuda(self, std::move(output_size), align_corners,
                                 scales));
    return out;
}

Tensor& upsample_linear1d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, bool align_corners,
    std::optional<double> scales, Tensor& grad_input) {
    write_out(grad_input, upsample_linear1d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size),
        align_corners, scales));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleLinear1dKernels) {
    m.impl("upsample_linear1d", upsample_linear1d_cuda);
    m.impl("upsample_linear1d_backward", upsample_linear1d_backward_cuda);
    m.impl("upsample_linear1d.out", upsample_linear1d_out_cuda);
    m.impl("upsample_linear1d_backward.grad_input", upsample_linear1d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
