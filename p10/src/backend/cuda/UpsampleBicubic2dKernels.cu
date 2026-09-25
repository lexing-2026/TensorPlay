// Upsampling CUDA kernels: bicubic (Keys) interpolation of 2-D planes.
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
__global__ void upsample_bicubic2d_out_frame(
    const int64_t num_elements,
    const accscalar_t height_scale, const accscalar_t width_scale,
    const bool align_corners,
    const scalar_t* idata, scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t input_height, const int64_t input_width,
    const int64_t output_height, const int64_t output_width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    const int64_t output_x = index % output_width;
    const int64_t output_y = index / output_width;

    if (input_height == output_height && input_width == output_width) {
        for (int64_t n = 0; n < batchsize; n++) {
            for (int64_t c = 0; c < channels; c++) {
                odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x] =
                    idata[(n * channels + c) * input_height * input_width + output_y * input_width + output_x];
            }
        }
        return;
    }

    const accscalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
    const int64_t in_x = static_cast<int64_t>(floorf(real_x));
    const accscalar_t t_x = real_x - static_cast<accscalar_t>(in_x);

    const accscalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
    const int64_t in_y = static_cast<int64_t>(floorf(real_y));
    const accscalar_t t_y = real_y - static_cast<accscalar_t>(in_y);

    auto get_value_bounded = [&](int64_t n, int64_t c, int64_t y, int64_t x) -> scalar_t {
        const int64_t access_y = max(min(y, input_height - 1), static_cast<int64_t>(0));
        const int64_t access_x = max(min(x, input_width - 1), static_cast<int64_t>(0));
        return idata[(n * channels + c) * input_height * input_width + access_y * input_width + access_x];
    };

    for (int64_t n = 0; n < batchsize; n++) {
        for (int64_t c = 0; c < channels; c++) {
            accscalar_t coefficients[4];
            for (int k = 0; k < 4; ++k) {
                coefficients[k] = cubic_interp1d(
                    get_value_bounded(n, c, in_y - 1 + k, in_x - 1),
                    get_value_bounded(n, c, in_y - 1 + k, in_x + 0),
                    get_value_bounded(n, c, in_y - 1 + k, in_x + 1),
                    get_value_bounded(n, c, in_y - 1 + k, in_x + 2),
                    t_x);
            }
            odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x] =
                static_cast<scalar_t>(cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y));
        }
    }
}

// ===========================================================================
// Backward frames: atomicAdd accumulation; reduction order is
// nondeterministic across runs.
// ===========================================================================

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_bicubic2d_backward_out_frame(
    const int64_t num_elements,
    const accscalar_t height_scale, const accscalar_t width_scale,
    const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t input_height, const int64_t input_width,
    const int64_t output_height, const int64_t output_width) {
    // scatter each output gradient into the bounded 4x4 input window.
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    const int64_t output_x = index % output_width;
    const int64_t output_y = index / output_width;

    if (input_height == output_height && input_width == output_width) {
        for (int64_t n = 0; n < batchsize; n++) {
            for (int64_t c = 0; c < channels; ++c) {
                idata[(n * channels + c) * input_height * input_width + output_y * input_width + output_x] +=
                    odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x];
            }
        }
        return;
    }

    const accscalar_t real_x = area_pixel_compute_source_index(width_scale, output_x, align_corners, /*cubic=*/true);
    const int64_t input_x = static_cast<int64_t>(floorf(real_x));
    const accscalar_t t_x = real_x - static_cast<accscalar_t>(input_x);

    const accscalar_t real_y = area_pixel_compute_source_index(height_scale, output_y, align_corners, /*cubic=*/true);
    const int64_t input_y = static_cast<int64_t>(floorf(real_y));
    const accscalar_t t_y = real_y - static_cast<accscalar_t>(input_y);

    accscalar_t x_coeffs[4];
    accscalar_t y_coeffs[4];
    get_cubic_upsample_coefficients(x_coeffs, t_x);
    get_cubic_upsample_coefficients(y_coeffs, t_y);

    auto increment_value_bounded = [&](int64_t n, int64_t c, int64_t y, int64_t x, accscalar_t value) {
        const int64_t access_y = max(min(y, input_height - 1), static_cast<int64_t>(0));
        const int64_t access_x = max(min(x, input_width - 1), static_cast<int64_t>(0));
        atomicAdd(idata + (n * channels + c) * input_height * input_width + access_y * input_width + access_x,
                  static_cast<scalar_t>(value));
    };

    for (int64_t n = 0; n < batchsize; n++) {
        for (int64_t c = 0; c < channels; ++c) {
            const scalar_t out_value =
                odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    increment_value_bounded(n, c, input_y - 1 + i, input_x - 1 + j,
                                            static_cast<accscalar_t>(out_value) * y_coeffs[i] * x_coeffs[j]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

Tensor upsample_bicubic2d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t input_height = in.size(2), input_width = in.size(3);
    const int64_t output_height = output_size[0], output_width = output_size[1];
    if (in.numel() == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(output_height * output_width, block, grid);
        upsample_bicubic2d_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output_height * output_width,
            area_pixel_compute_scale_h<accscalar_t>(input_height, output_height, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(input_width, output_width, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, input_height, input_width, output_height, output_width);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_bicubic2d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_bicubic2d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t input_height = input_size[2], input_width = input_size[3];
    const int64_t output_height = output_size[0], output_width = output_size[1];
    if (go.numel() == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(output_height * output_width, block, grid);
        upsample_bicubic2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output_height * output_width,
            area_pixel_compute_scale_h<accscalar_t>(input_height, output_height, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(input_width, output_width, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, input_height, input_width, output_height, output_width);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

// ===========================================================================
// nearest-exact upsampling (Pillow / Scikit-Image convention).  Forward
// gathers input[floor(scale*(i+0.5))]; backward scatters each output gradient
// to its owning source pixel with atomicAdd (nondeterministic order).
// ===========================================================================

Tensor& upsample_bicubic2d_out_cuda(const Tensor& self,
                                    const std::vector<int64_t>& output_size,
                                    bool align_corners,
                                    std::optional<double> scales_h,
                                    std::optional<double> scales_w,
                                    Tensor& out) {
    write_out(out, upsample_bicubic2d_cuda(self, std::move(output_size), align_corners,
                                  scales_h, scales_w));
    return out;
}

Tensor& upsample_bicubic2d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    Tensor& grad_input) {
    write_out(grad_input, upsample_bicubic2d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size),
        align_corners, scales_h, scales_w));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleBicubic2dKernels) {
    m.impl("upsample_bicubic2d", upsample_bicubic2d_cuda);
    m.impl("upsample_bicubic2d_backward", upsample_bicubic2d_backward_cuda);
    m.impl("upsample_bicubic2d.out", upsample_bicubic2d_out_cuda);
    m.impl("upsample_bicubic2d_backward.grad_input", upsample_bicubic2d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
