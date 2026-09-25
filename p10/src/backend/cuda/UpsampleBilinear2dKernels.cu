// Upsampling CUDA kernels: bilinear interpolation of 2-D planes together with the
// antialiased 2-D path, whose separable filter tables serve both the
// bilinear and the bicubic entry points.
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
__global__ void upsample_bilinear2d_out_frame(
    const int64_t num_kernels,
    const accscalar_t rheight, const accscalar_t rwidth,
    const bool align_corners,
    const scalar_t* idata, scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;

    const int64_t w2 = index % width2;
    const int64_t h2 = index / width2;

    const accscalar_t h1r = area_pixel_compute_source_index(rheight, h2, align_corners, /*cubic=*/false);
    const int64_t h1 = static_cast<int64_t>(h1r);
    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, /*cubic=*/false);
    const int64_t w1 = static_cast<int64_t>(w1r);
    const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    using accscalar_pack_t = accscalar_t;
    for (int64_t n = 0; n < batchsize; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const scalar_t* iptr = idata + (n * channels + c) * height1 * width1;
            const accscalar_pack_t val = h0lambda *
                    (w0lambda * iptr[h1 * width1 + w1] +
                     w1lambda * iptr[h1 * width1 + w1 + w1p]) +
                h1lambda *
                    (w0lambda * iptr[(h1 + h1p) * width1 + w1] +
                     w1lambda * iptr[(h1 + h1p) * width1 + w1 + w1p]);
            odata[(n * channels + c) * height2 * width2 + h2 * width2 + w2] = static_cast<scalar_t>(val);
        }
    }
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_bilinear2d_backward_out_frame(
    const int64_t o_numel,
    const accscalar_t rheight, const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= o_numel) return;

    // (non-ROCm branch).
    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t n_c = index / (height2 * width2);

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

    const accscalar_t val = odata[n_c * height2 * width2 + h2 * width2 + w2];
    scalar_t* base = idata + n_c * height1 * width1;
    atomicAdd(base + h1 * width1 + w1, static_cast<scalar_t>(h0lambda * w0lambda * val));
    atomicAdd(base + h1 * width1 + w1 + w1p, static_cast<scalar_t>(h0lambda * w1lambda * val));
    atomicAdd(base + (h1 + h1p) * width1 + w1, static_cast<scalar_t>(h1lambda * w0lambda * val));
    atomicAdd(base + (h1 + h1p) * width1 + w1 + w1p, static_cast<scalar_t>(h1lambda * w1lambda * val));
}

Tensor upsample_bilinear2d_cuda(const Tensor& self, const std::vector<int64_t>& output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t height1 = in.size(2), width1 = in.size(3);
    const int64_t height2 = output_size[0], width2 = output_size[1];
    if (in.numel() == 0 || height2 == 0 || width2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(height2 * width2, block, grid);
        upsample_bilinear2d_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            height2 * width2,
            area_pixel_compute_scale_h<accscalar_t>(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(width1, width2, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, height1, width1, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_bilinear2d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size, const std::vector<int64_t>& input_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_bilinear2d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t height1 = input_size[2], width1 = input_size[3];
    const int64_t height2 = output_size[0], width2 = output_size[1];
    if (go.numel() == 0 || height2 * width2 == 0 || height1 * width1 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(batchsize * channels * height2 * width2, block, grid);
        upsample_bilinear2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            batchsize * channels * height2 * width2,
            area_pixel_compute_scale_h<accscalar_t>(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h<accscalar_t>(width1, width2, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, height1, width1, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

inline double aa_triangle_h(double x) {
    x = std::abs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

// Keys cubic convolution (a = -0.5); the antialias path uses the Keys kernel
// for PIL compatibility, unlike the non-antialias bicubic path (a = -0.75).

inline double aa_cubic_h(double x) {
    constexpr double A = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((A + 2) * x - (A + 3)) * x * x + 1;
    if (x < 2.0) return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    return 0.0;
}

using aa_filter_h = double (*)(double);

inline double aa_axis_scale_h(int64_t in_size, int64_t out_size, bool align_corners,
                              const std::optional<double>& scale) {
    if (align_corners) {
        return out_size > 1
            ? static_cast<double>(in_size - 1) / static_cast<double>(out_size - 1)
            : 0.0;
    }
    return (scale.has_value() && scale.value() > 0.)
        ? 1.0 / scale.value()
        : static_cast<double>(in_size) / static_cast<double>(out_size);
}

struct AaTablesDevice {
    int64_t max_taps = 0;
    Tensor begins;   // Int64 [out_size]
    Tensor taps;     // Int64 [out_size]
    Tensor weights;  // accscalar dtype [out_size * max_taps]
};

template <typename accscalar_t>
AaTablesDevice aa_make_tables(const Tensor& proto, int64_t in_size, int64_t out_size,
                              double scale, int taps_half, aa_filter_h filter) {
    const double support = (scale >= 1.0) ? static_cast<double>(taps_half) * scale
                                          : static_cast<double>(taps_half);
    const int64_t max_taps = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
    const double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
    std::vector<int64_t> hb(static_cast<size_t>(out_size));
    std::vector<int64_t> ht(static_cast<size_t>(out_size));
    std::vector<accscalar_t> hw(static_cast<size_t>(out_size) * static_cast<size_t>(max_taps),
                                static_cast<accscalar_t>(0));
    for (int64_t i = 0; i < out_size; ++i) {
        const double center = scale * (static_cast<double>(i) + 0.5);
        const int64_t lo = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
        int64_t n = std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), in_size) - lo;
        n = std::clamp<int64_t>(n, 0, max_taps);
        double total = 0.0;
        for (int64_t j = 0; j < n; ++j)
            total += filter((static_cast<double>(j + lo) - center + 0.5) * invscale);
        for (int64_t j = 0; j < n; ++j) {
            const double w = filter((static_cast<double>(j + lo) - center + 0.5) * invscale);
            hw[static_cast<size_t>(i) * max_taps + j] =
                static_cast<accscalar_t>(total != 0.0 ? w / total : 0.0);
        }
        hb[static_cast<size_t>(i)] = lo;
        ht[static_cast<size_t>(i)] = n;
    }
    AaTablesDevice t;
    t.max_taps = max_taps;
    t.begins = Tensor::empty({out_size}, DType::Int64, proto.device());
    t.taps = Tensor::empty({out_size}, DType::Int64, proto.device());
    t.weights = Tensor::empty({out_size * max_taps},
                              std::is_same<accscalar_t, float>::value ? DType::Float32 : DType::Float64,
                              proto.device());
    CUDA_CHECK(cudaMemcpy(t.begins.data_ptr<int64_t>(), hb.data(),
                          hb.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t.taps.data_ptr<int64_t>(), ht.data(),
                          ht.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t.weights.data_ptr<accscalar_t>(), hw.data(),
                          hw.size() * sizeof(accscalar_t), cudaMemcpyHostToDevice));
    return t;
}

template <typename accscalar_t, typename scalar_t>
__global__ void aa_2d_horizontal_frame(
    const int64_t total, const int64_t W1, const int64_t W2, const int64_t max_taps,
    const int64_t* begins, const int64_t* taps, const accscalar_t* weights,
    const scalar_t* in, scalar_t* scratch) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t w2 = index % W2;
    const int64_t h1nc = index / W2;  // h1 plus the (n, c) plane offset
    const int64_t lo = begins[w2];
    const int64_t n = taps[w2];
    const accscalar_t* w = weights + static_cast<size_t>(w2) * max_taps;
    const scalar_t* src = in + h1nc * W1;
    accscalar_t acc = 0;
    for (int64_t j = 0; j < n; ++j) acc += w[j] * src[lo + j];
    scratch[index] = static_cast<scalar_t>(acc);
}

template <typename accscalar_t, typename scalar_t>
__global__ void aa_2d_vertical_frame(
    const int64_t total, const int64_t H1, const int64_t H2, const int64_t W2,
    const int64_t max_taps,
    const int64_t* begins, const int64_t* taps, const accscalar_t* weights,
    const scalar_t* scratch, scalar_t* out) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t w2 = index % W2;
    const int64_t h2 = (index / W2) % H2;
    const int64_t nc = index / (W2 * H2);
    const int64_t lo = begins[h2];
    const int64_t n = taps[h2];
    const accscalar_t* w = weights + static_cast<size_t>(h2) * max_taps;
    accscalar_t acc = 0;
    for (int64_t k = 0; k < n; ++k)
        acc += w[k] * scratch[(nc * H1 + lo + k) * W2 + w2];
    out[index] = static_cast<scalar_t>(acc);
}

template <typename accscalar_t, typename scalar_t>
__global__ void aa_2d_backward_frame(
    const int64_t total, const int64_t H1, const int64_t W1,
    const int64_t H2, const int64_t W2,
    const int64_t max_taps_h, const int64_t max_taps_w,
    const int64_t* begins_h, const int64_t* taps_h, const accscalar_t* weights_h,
    const int64_t* begins_w, const int64_t* taps_w, const accscalar_t* weights_w,
    const scalar_t* go, scalar_t* gi) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t w2 = index % W2;
    const int64_t h2 = (index / W2) % H2;
    const int64_t nc = index / (W2 * H2);
    const scalar_t g = go[index];
    if (g == static_cast<scalar_t>(0)) return;
    const int64_t ylo = begins_h[h2];
    const int64_t yn = taps_h[h2];
    const int64_t xlo = begins_w[w2];
    const int64_t xn = taps_w[w2];
    const accscalar_t* wy = weights_h + static_cast<size_t>(h2) * max_taps_h;
    const accscalar_t* wx = weights_w + static_cast<size_t>(w2) * max_taps_w;
    scalar_t* base = gi + nc * H1 * W1;
    for (int64_t k = 0; k < yn; ++k) {
        const scalar_t gy = static_cast<scalar_t>(wy[k]) * g;
        scalar_t* row = base + (ylo + k) * W1;
        for (int64_t j = 0; j < xn; ++j)
            atomicAdd(row + xlo + j, static_cast<scalar_t>(wx[j]) * gy);
    }
}

Tensor aa_2d_forward_cuda(const Tensor& in, const std::vector<int64_t>& output_size,
                          bool align_corners, const std::optional<double>& scales_h,
                          const std::optional<double>& scales_w,
                          int taps_half, aa_filter_h filter) {
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H1 = in.size(2), W1 = in.size(3);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (in.numel() == 0 || H2 == 0 || W2 == 0) return result;
    Tensor scratch = Tensor::empty({N * C * H1 * W2}, in.dtype(), in.device());
    const double sh = aa_axis_scale_h(H1, H2, align_corners, scales_h);
    const double sw = aa_axis_scale_h(W1, W2, align_corners, scales_w);
    UP_DISPATCH(in, {
        const AaTablesDevice th = aa_make_tables<accscalar_t>(in, H1, H2, sh, taps_half, filter);
        const AaTablesDevice tw = aa_make_tables<accscalar_t>(in, W1, W2, sw, taps_half, filter);
        dim3 block, grid;
        launch_dims(N * C * H1 * W2, block, grid);
        aa_2d_horizontal_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * H1 * W2, W1, W2, tw.max_taps,
            tw.begins.data_ptr<int64_t>(), tw.taps.data_ptr<int64_t>(),
            tw.weights.data_ptr<accscalar_t>(), in.data_ptr<scalar_t>(),
            scratch.data_ptr<scalar_t>());
        launch_dims(N * C * H2 * W2, block, grid);
        aa_2d_vertical_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * H2 * W2, H1, H2, W2, th.max_taps,
            th.begins.data_ptr<int64_t>(), th.taps.data_ptr<int64_t>(),
            th.weights.data_ptr<accscalar_t>(), scratch.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>());
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor aa_2d_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                           const std::vector<int64_t>& input_size, bool align_corners,
                           const std::optional<double>& scales_h, const std::optional<double>& scales_w,
                           int taps_half, aa_filter_h filter) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(grad_input_shape(go, input_size), go.dtype(), go.device());
    const int64_t N = go.size(0), C = go.size(1);
    const int64_t H1 = input_size[2], W1 = input_size[3];
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (go.numel() == 0 || H1 == 0 || W1 == 0 || H2 == 0 || W2 == 0) return grad_input;
    const double sh = aa_axis_scale_h(H1, H2, align_corners, scales_h);
    const double sw = aa_axis_scale_h(W1, W2, align_corners, scales_w);
    UP_DISPATCH(go, {
        const AaTablesDevice th = aa_make_tables<accscalar_t>(go, H1, H2, sh, taps_half, filter);
        const AaTablesDevice tw = aa_make_tables<accscalar_t>(go, W1, W2, sw, taps_half, filter);
        dim3 block, grid;
        launch_dims(N * C * H2 * W2, block, grid);
        aa_2d_backward_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            N * C * H2 * W2, H1, W1, H2, W2, th.max_taps, tw.max_taps,
            th.begins.data_ptr<int64_t>(), th.taps.data_ptr<int64_t>(),
            th.weights.data_ptr<accscalar_t>(),
            tw.begins.data_ptr<int64_t>(), tw.taps.data_ptr<int64_t>(),
            tw.weights.data_ptr<accscalar_t>(),
            go.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>());
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_bilinear2d_aa_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                   bool align_corners, std::optional<double> scales_h,
                                   std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    return aa_2d_forward_cuda(in, output_size, align_corners, scales_h, scales_w,
                              /*taps_half=*/1, aa_triangle_h);
}

Tensor upsample_bicubic2d_aa_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                  bool align_corners, std::optional<double> scales_h,
                                  std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    return aa_2d_forward_cuda(in, output_size, align_corners, scales_h, scales_w,
                              /*taps_half=*/2, aa_cubic_h);
}

Tensor upsample_bilinear2d_aa_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                            const std::vector<int64_t>& input_size, bool align_corners,
                                            std::optional<double> scales_h, std::optional<double> scales_w) {
    return aa_2d_backward_cuda(grad_output, output_size, input_size, align_corners, scales_h, scales_w,
                               /*taps_half=*/1, aa_triangle_h);
}

Tensor upsample_bicubic2d_aa_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                           const std::vector<int64_t>& input_size, bool align_corners,
                                           std::optional<double> scales_h, std::optional<double> scales_w) {
    return aa_2d_backward_cuda(grad_output, output_size, input_size, align_corners, scales_h, scales_w,
                               /*taps_half=*/2, aa_cubic_h);
}

Tensor& upsample_bilinear2d_aa_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                        bool align_corners, std::optional<double> scales_h,
                                        std::optional<double> scales_w, Tensor& out) {
    write_out(out, upsample_bilinear2d_aa_cuda(self, std::move(output_size), align_corners, scales_h, scales_w));
    return out;
}

Tensor& upsample_bicubic2d_aa_out_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                                       bool align_corners, std::optional<double> scales_h,
                                       std::optional<double> scales_w, Tensor& out) {
    write_out(out, upsample_bicubic2d_aa_cuda(self, std::move(output_size), align_corners, scales_h, scales_w));
    return out;
}

Tensor& upsample_bilinear2d_aa_backward_grad_input_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                                        const std::vector<int64_t>& input_size, bool align_corners,
                                                        std::optional<double> scales_h,
                                                        std::optional<double> scales_w, Tensor& grad_input) {
    write_out(grad_input, upsample_bilinear2d_aa_backward_cuda(grad_output, std::move(output_size),
                                                      std::move(input_size), align_corners, scales_h, scales_w));
    return grad_input;
}

Tensor& upsample_bicubic2d_aa_backward_grad_input_cuda(const Tensor& grad_output, const std::vector<int64_t>& output_size,
                                                       const std::vector<int64_t>& input_size, bool align_corners,
                                                       std::optional<double> scales_h,
                                                       std::optional<double> scales_w, Tensor& grad_input) {
    write_out(grad_input, upsample_bicubic2d_aa_backward_cuda(grad_output, std::move(output_size),
                                                     std::move(input_size), align_corners, scales_h, scales_w));
    return grad_input;
}

std::vector<int64_t> aa_vec_output_size_h(const Tensor& input,
                                          const std::optional<std::vector<int64_t>>& output_size,
                                          const std::optional<std::vector<double>>& scale_factors) {
    if (output_size.has_value()) {
        if (output_size.value().size() != 2)
            TP_THROW(RuntimeError, "_upsample_aa: vec output_size must have 2 entries");
        return output_size.value();
    }
    if (!scale_factors.has_value())
        TP_THROW(RuntimeError, "_upsample_aa: vec form needs output_size or scale_factors");
    const auto& sf = scale_factors.value();
    if (sf.size() != 2)
        TP_THROW(RuntimeError, "_upsample_aa: vec scale_factors must have 2 entries");
    return {static_cast<int64_t>(std::floor(static_cast<double>(input.size(2)) * sf[0])),
            static_cast<int64_t>(std::floor(static_cast<double>(input.size(3)) * sf[1]))};
}

Tensor _upsample_bilinear2d_aa_vec_cuda(const Tensor& input,
                                        const std::optional<std::vector<int64_t>>& output_size,
                                        bool align_corners,
                                        const std::optional<std::vector<double>>& scale_factors) {
    return tpx::ops::_upsample_bilinear2d_aa(
        input, aa_vec_output_size_h(input, output_size, scale_factors), align_corners);
}

Tensor _upsample_bicubic2d_aa_vec_cuda(const Tensor& input,
                                       const std::optional<std::vector<int64_t>>& output_size,
                                       bool align_corners,
                                       const std::optional<std::vector<double>>& scale_factors) {
    return tpx::ops::_upsample_bicubic2d_aa(
        input, aa_vec_output_size_h(input, output_size, scale_factors), align_corners);
}

Tensor& upsample_bilinear2d_out_cuda(const Tensor& self,
                                     const std::vector<int64_t>& output_size,
                                     bool align_corners,
                                     std::optional<double> scales_h,
                                     std::optional<double> scales_w,
                                     Tensor& out) {
    write_out(out, upsample_bilinear2d_cuda(self, std::move(output_size), align_corners,
                                   scales_h, scales_w));
    return out;
}

Tensor& upsample_bilinear2d_backward_grad_input_cuda(
    const Tensor& grad_output, const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w,
    Tensor& grad_input) {
    write_out(grad_input, upsample_bilinear2d_backward_cuda(
        grad_output, std::move(output_size), std::move(input_size),
        align_corners, scales_h, scales_w));
    return grad_input;
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleBilinear2dKernels) {
    m.impl("upsample_bilinear2d", upsample_bilinear2d_cuda);
    m.impl("upsample_bilinear2d_backward", upsample_bilinear2d_backward_cuda);
    m.impl("_upsample_bilinear2d_aa", upsample_bilinear2d_aa_cuda);
    m.impl("_upsample_bilinear2d_aa.out", upsample_bilinear2d_aa_out_cuda);
    m.impl("_upsample_bilinear2d_aa.vec", _upsample_bilinear2d_aa_vec_cuda);
    m.impl("_upsample_bilinear2d_aa_backward", upsample_bilinear2d_aa_backward_cuda);
    m.impl("_upsample_bilinear2d_aa_backward.grad_input", upsample_bilinear2d_aa_backward_grad_input_cuda);
    m.impl("_upsample_bicubic2d_aa", upsample_bicubic2d_aa_cuda);
    m.impl("_upsample_bicubic2d_aa.out", upsample_bicubic2d_aa_out_cuda);
    m.impl("_upsample_bicubic2d_aa.vec", _upsample_bicubic2d_aa_vec_cuda);
    m.impl("_upsample_bicubic2d_aa_backward", upsample_bicubic2d_aa_backward_cuda);
    m.impl("_upsample_bicubic2d_aa_backward.grad_input", upsample_bicubic2d_aa_backward_grad_input_cuda);
    m.impl("upsample_bilinear2d.out", upsample_bilinear2d_out_cuda);
    m.impl("upsample_bilinear2d_backward.grad_input", upsample_bilinear2d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
