// Upsampling CUDA kernels.
//
// Direct port of the ATen frame kernels; each site cites its source:
//   aten/src/ATen/native/cuda/UpSampleNearest1d.cu
//   aten/src/ATen/native/cuda/UpSampleNearest2d.cu
//     upsample_nearest2d_out_frame / upsample_nearest2d_backward_out_frame
//   aten/src/ATen/native/cuda/UpSampleNearest3d.cu
//   aten/src/ATen/native/cuda/UpSampleLinear1d.cu
//   aten/src/ATen/native/cuda/UpSampleBilinear2d.cu
//     upsample_bilinear2d_out_frame / upsample_bilinear2d_backward_out_frame
//   aten/src/ATen/native/cuda/UpSampleBicubic2d.cu
//     upsample_bicubic2d_out_frame / upsample_bicubic2d_backward_out_frame
//   aten/src/ATen/native/cuda/UpSampleTrilinear3d.cu
//   and UpSample.h / UpSample.cuh for the shared index helpers (the same
//   helpers are ported in backend/cpu/UpsampleKernels.cpp).
//
// Kernels operate on contiguous NCT(D)HW tensors and support Float32/Float64,
// matching the CPU dispatch.  Scatter backwards use atomicAdd like ATen
// ("Nondeterministic because of atomicAdd usage", UpSampleBicubic2d.cu).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Utils.h"

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t error = (expr); \
        if (error != cudaSuccess) { \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        } \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// UpSample.h helpers (mirrors backend/cpu/UpsampleKernels.cpp)
// ---------------------------------------------------------------------------

inline float compute_scales_value_h(const std::optional<double>& scale, int64_t input_size, int64_t output_size) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<float>(1.0 / scale.value())
        : static_cast<float>(static_cast<double>(input_size) / output_size);
}

// ATen UpSample.cuh compute_scales_value_backwards: nearest-backward index
// math wants the output/input ratio, and an explicit scale_factor is used
// as-is (not inverted like in the forward).
inline float compute_scales_value_backwards_h(const std::optional<double>& scale,
                                              int64_t src_size, int64_t dst_size) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<float>(scale.value())
        : static_cast<float>(static_cast<double>(src_size) / dst_size);
}

inline float area_pixel_compute_scale_h(int64_t input_size, int64_t output_size,
                                        bool align_corners, const std::optional<double>& scale) {
    if (align_corners) {
        if (output_size > 1) return static_cast<float>(static_cast<double>(input_size - 1) / (output_size - 1));
        return 0.f;
    }
    return compute_scales_value_h(scale, input_size, output_size);
}

__host__ __device__ inline float area_pixel_compute_source_index(float scale, int64_t dst_index,
                                                                 bool align_corners, bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    }
    float src_idx = scale * (dst_index + 0.5f) - 0.5f;
    // [Note] Follow Opencv resize logic; linear modes bound negatives to zero.
    return (!cubic && src_idx < 0.f) ? 0.f : src_idx;
}

// UpSample.h nearest_neighbor_compute_source_index (OpenCV INTER_NEAREST BC).
__host__ __device__ inline int nearest_neighbor_compute_source_index(float scale, int dst_index,
                                                                     int input_size) {
    int src_index = static_cast<int>(fminf(floorf(static_cast<float>(dst_index) * scale),
                                           static_cast<float>(input_size - 1)));
    return src_index < 0 ? 0 : src_index;
}

// UpSample.cuh nearest_neighbor_bw_compute_source_index.
__host__ __device__ inline int nearest_neighbor_bw_compute_source_index(float scale, int dst_index,
                                                                        int output_size) {
    int src_index = static_cast<int>(fminf(ceilf(static_cast<float>(dst_index) * scale),
                                           static_cast<float>(output_size)));
    return src_index;
}

// UpSample.h cubic machinery (A = -0.75).
template <typename scalar_t>
__host__ __device__ inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}
template <typename scalar_t>
__host__ __device__ inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}
template <typename scalar_t>
__host__ __device__ inline void get_cubic_upsample_coefficients(scalar_t coeffs[4], scalar_t t) {
    scalar_t A = -0.75;
    scalar_t x1 = t;
    coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1<scalar_t>(x1, A);
    scalar_t x2 = 1.0 - t;
    coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
    coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}
template <typename scalar_t>
__host__ __device__ inline scalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t t) {
    scalar_t coeffs[4];
    get_cubic_upsample_coefficients<scalar_t>(coeffs, t);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

inline std::vector<int64_t> out_shape(const Tensor& self, const std::vector<int64_t>& out_sizes) {
    std::vector<int64_t> s{self.size(0), self.size(1)};
    s.insert(s.end(), out_sizes.begin(), out_sizes.end());
    return s;
}

inline void launch_dims(int64_t total, dim3& block, dim3& grid) {
    block = dim3(256);
    grid = dim3(static_cast<unsigned>((total + 255) / 256));
}

#define UP_DISPATCH(t, ...) \
    switch ((t).dtype()) { \
        case DType::Float32: { using scalar_t = float; using accscalar_t = float; __VA_ARGS__; break; } \
        case DType::Float64: { using scalar_t = double; using accscalar_t = double; __VA_ARGS__; break; } \
        default: TP_THROW(NotImplementedError, "cuda upsample only supports Float32/Float64"); \
    }

} // anonymous namespace

// ===========================================================================
// Nearest forwards — ATen UpSampleNearest{1d,2d,3d}.cu *_out_frame
// ===========================================================================

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

template <typename scalar_t>
__global__ void upsample_nearest2d_out_frame(
    const scalar_t* idata, scalar_t* odata,
    const int64_t nc, const int64_t height1, const int64_t width1,
    const int64_t height2, const int64_t width2,
    float height_scale, float width_scale) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= nc * height2 * width2) return;

    const int64_t w2 = index % width2;
    const int64_t h2 = (index / width2) % height2;
    const int64_t n_c = index / (height2 * width2);

    // ATen: size equality fast path, else nn index fn
    const int h1 = height1 == height2 ? static_cast<int>(h2)
                                      : nearest_neighbor_compute_source_index(height_scale, static_cast<int>(h2), static_cast<int>(height1));
    const int w1 = width1 == width2 ? static_cast<int>(w2)
                                    : nearest_neighbor_compute_source_index(width_scale, static_cast<int>(w2), static_cast<int>(width1));

    odata[index] = idata[(n_c * height1 + h1) * width1 + w1];
}

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
// Nearest backwards — ATen UpSampleNearest{1d,2d,3d}.cu *_backward_out_frame
// (gather formulation over input pixels; no atomics needed)
// ===========================================================================

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest1d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_w, const int64_t dst_dim_w,
    accscalar_t* grad_i, float width_scale) {
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
        grad_i[b * dim_c * dst_dim_w + dst_idx] = grad;
    }
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest2d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_h, const int64_t dst_dim_w,
    accscalar_t* grad_i, float height_scale, float width_scale) {
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
        grad_i[dst_idx + b * dim_c * dst_c_stride] = grad;
    }
}

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_nearest3d_backward_out_frame(
    const scalar_t* grad_o, const int64_t dim_b, const int64_t dim_c,
    const int64_t src_dim_d, const int64_t src_dim_h, const int64_t src_dim_w,
    const int64_t dst_dim_d, const int64_t dst_dim_h, const int64_t dst_dim_w,
    accscalar_t* grad_i, float depth_scale, float height_scale, float width_scale) {
    const int64_t dst_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w) return;

    const int64_t dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
    const int64_t src_c_stride = src_dim_d * src_dim_h * src_dim_w;
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
        grad_i[dst_idx + b * dim_c * dst_c_stride] = grad;
    }
}

// ===========================================================================
// Linear forwards — ATen UpSampleLinear1d.cu / UpSampleBilinear2d.cu /
// UpSampleTrilinear3d.cu *_out_frame
// ===========================================================================

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

    // ATen UpSampleBilinear2d.cu upsample_bilinear2d_out_frame
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

    // ATen UpSampleTrilinear3d.cu upsample_trilinear3d_out_frame
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
__global__ void upsample_linear1d_out_frame(
    const int64_t num_kernels,
    const accscalar_t rwidth, const bool align_corners,
    const scalar_t* idata, scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t width1, const int64_t width2) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= num_kernels) return;
    // ATen UpSampleLinear1d.cu upsample_linear1d_out_frame
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
// Bicubic forward — ATen UpSampleBicubic2d.cu upsample_bicubic2d_out_frame
// ===========================================================================

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
        // ATen UpSample.cuh upsample_get_value_bounded
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
// Scatter backwards (linear/bicubic/trilinear) — ATen frame kernels with
#// atomicAdd accumulation ("Nondeterministic because of atomicAdd usage").
// ===========================================================================

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

    // ATen UpSampleBilinear2d.cu upsample_bilinear2d_backward_out_frame
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
    // ATen uses gpuAtomicAddNoReturn; atomicAdd overloads cover float/double.
    scalar_t* base = idata + n_c * height1 * width1;
    atomicAdd(base + h1 * width1 + w1, static_cast<scalar_t>(h0lambda * w0lambda * val));
    atomicAdd(base + h1 * width1 + w1 + w1p, static_cast<scalar_t>(h0lambda * w1lambda * val));
    atomicAdd(base + (h1 + h1p) * width1 + w1, static_cast<scalar_t>(h1lambda * w0lambda * val));
    atomicAdd(base + (h1 + h1p) * width1 + w1 + w1p, static_cast<scalar_t>(h1lambda * w1lambda * val));
}

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

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_trilinear3d_backward_out_frame(
    const int64_t o_numel,
    const accscalar_t rdepth, const accscalar_t rheight, const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t depth1, const int64_t height1, const int64_t width1,
    const int64_t depth2, const int64_t height2, const int64_t width2) {
    // ATen UpSampleTrilinear3d.cu upsample_trilinear3d_backward_out_frame:
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

template <typename accscalar_t, typename scalar_t>
__global__ void upsample_bicubic2d_backward_out_frame(
    const int64_t num_elements,
    const accscalar_t height_scale, const accscalar_t width_scale,
    const bool align_corners,
    scalar_t* idata, const scalar_t* odata,
    const int64_t batchsize, const int64_t channels,
    const int64_t input_height, const int64_t input_width,
    const int64_t output_height, const int64_t output_width) {
    // ATen UpSampleBicubic2d.cu upsample_bicubic2d_backward_out_frame:
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

Tensor upsample_nearest1d_cuda(const Tensor& self, std::vector<int64_t> output_size, std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * W2, block, grid);
        upsample_nearest1d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, W1, W2,
            compute_scales_value_h(scales, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest2d_cuda(const Tensor& self, std::vector<int64_t> output_size, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H1 = in.size(2), W1 = in.size(3);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (in.numel() == 0 || H2 == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * H2 * W2, block, grid);
        upsample_nearest2d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, H1, W1, H2, W2,
            compute_scales_value_h(scales_h, H1, H2), compute_scales_value_h(scales_w, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest3d_cuda(const Tensor& self, std::vector<int64_t> output_size, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t D1 = in.size(2), H1 = in.size(3), W1 = in.size(4);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    if (in.numel() == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(N * C * D2 * H2 * W2, block, grid);
        upsample_nearest3d_out_frame<scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N * C, D1, H1, W1, D2, H2, W2,
            compute_scales_value_h(scales_d, D1, D2), compute_scales_value_h(scales_h, H1, H2),
            compute_scales_value_h(scales_w, W1, W2));
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_nearest1d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, std::optional<double> scales) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t W2 = output_size[0], W1 = input_size[0];
    if (go.numel() == 0 || W2 == 0 || W1 == 0) return grad_input;
    UP_DISPATCH(go, {
        // ATen computes backward in opmath precision then casts down; here the
        // zeroed buffer is already scalar_t so accumulate via an accscalar
        // staging is unnecessary for f32/f64 (accscalar_t == scalar_t).
        dim3 block, grid;
        launch_dims(dim_c * W1, block, grid);
        upsample_nearest1d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, W2, W1,
            reinterpret_cast<accscalar_t*>(grad_input.data_ptr<scalar_t>()),
            // ATen compute_scales_value_backwards: output/input ratio.
            compute_scales_value_backwards_h(scales, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_nearest2d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    const int64_t H1 = input_size[0], W1 = input_size[1];
    if (go.numel() == 0 || H2 * W2 == 0 || H1 * W1 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(dim_c * H1 * W1, block, grid);
        upsample_nearest2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, H2, W2, H1, W1,
            reinterpret_cast<accscalar_t*>(grad_input.data_ptr<scalar_t>()),
            compute_scales_value_backwards_h(scales_h, H2, H1), compute_scales_value_backwards_h(scales_w, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_nearest3d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    const int64_t D1 = input_size[0], H1 = input_size[1], W1 = input_size[2];
    if (go.numel() == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(dim_c * D1 * H1 * W1, block, grid);
        upsample_nearest3d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            go.data_ptr<scalar_t>(), dim_b, dim_c, D2, H2, W2, D1, H1, W1,
            reinterpret_cast<accscalar_t*>(grad_input.data_ptr<scalar_t>()),
            compute_scales_value_backwards_h(scales_d, D2, D1), compute_scales_value_backwards_h(scales_h, H2, H1),
            compute_scales_value_backwards_h(scales_w, W2, W1));
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_linear1d_cuda(const Tensor& self, std::vector<int64_t> output_size, bool align_corners, std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;
    UP_DISPATCH(in, {
        dim3 block, grid;
        launch_dims(W2, block, grid);
        upsample_linear1d_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            W2, area_pixel_compute_scale_h(W1, W2, align_corners, scales), align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), N, C, W1, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_bilinear2d_cuda(const Tensor& self, std::vector<int64_t> output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
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
            area_pixel_compute_scale_h(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h(width1, width2, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, height1, width1, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_trilinear3d_cuda(const Tensor& self, std::vector<int64_t> output_size, bool align_corners, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
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
            area_pixel_compute_scale_h(depth1, depth2, align_corners, scales_d),
            area_pixel_compute_scale_h(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h(width1, width2, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, depth1, height1, width1, depth2, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_linear1d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, std::optional<double> scales) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_linear1d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t W2 = output_size[0], W1 = input_size[0];
    if (go.numel() == 0 || W2 == 0 || W1 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(batchsize * channels * W2, block, grid);
        upsample_linear1d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            batchsize * channels * W2,
            area_pixel_compute_scale_h(W1, W2, align_corners, scales), align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, W1, W2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_bilinear2d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_bilinear2d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t height1 = input_size[0], width1 = input_size[1];
    const int64_t height2 = output_size[0], width2 = output_size[1];
    if (go.numel() == 0 || height2 * width2 == 0 || height1 * width1 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(batchsize * channels * height2 * width2, block, grid);
        upsample_bilinear2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            batchsize * channels * height2 * width2,
            area_pixel_compute_scale_h(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h(width1, width2, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, height1, width1, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_trilinear3d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_trilinear3d_backward_cuda");
    // ATen UpSampleTrilinear3d.cu upsample_trilinear3d_backward_out_frame:
    // trilinear scatter with atomicAdd; implemented as two chained bilinear
    // scatters per depth slice pair (identical weight decomposition).
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t depth1 = input_size[0], height1 = input_size[1], width1 = input_size[2];
    const int64_t depth2 = output_size[0], height2 = output_size[1], width2 = output_size[2];
    if (go.numel() == 0 || depth1 * height1 * width1 == 0 || depth2 * height2 * width2 == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(depth2 * height2 * width2, block, grid);
        upsample_trilinear3d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            depth2 * height2 * width2,
            area_pixel_compute_scale_h(depth1, depth2, align_corners, scales_d),
            area_pixel_compute_scale_h(height1, height2, align_corners, scales_h),
            area_pixel_compute_scale_h(width1, width2, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, depth1, height1, width1, depth2, height2, width2);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

Tensor upsample_bicubic2d_cuda(const Tensor& self, std::vector<int64_t> output_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
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
            area_pixel_compute_scale_h(input_height, output_height, align_corners, scales_h),
            area_pixel_compute_scale_h(input_width, output_width, align_corners, scales_w),
            align_corners,
            in.data_ptr<scalar_t>(), result.data_ptr<scalar_t>(),
            batchsize, channels, input_height, input_width, output_height, output_width);
    });
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor upsample_bicubic2d_backward_cuda(const Tensor& grad_output, std::vector<int64_t> output_size, std::vector<int64_t> input_size, bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("upsample_bicubic2d_backward_cuda");
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t input_height = input_size[0], input_width = input_size[1];
    const int64_t output_height = output_size[0], output_width = output_size[1];
    if (go.numel() == 0) return grad_input;
    UP_DISPATCH(go, {
        dim3 block, grid;
        launch_dims(output_height * output_width, block, grid);
        upsample_bicubic2d_backward_out_frame<accscalar_t, scalar_t><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output_height * output_width,
            area_pixel_compute_scale_h(input_height, output_height, align_corners, scales_h),
            area_pixel_compute_scale_h(input_width, output_width, align_corners, scales_w),
            align_corners,
            grad_input.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
            batchsize, channels, input_height, input_width, output_height, output_width);
    });
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

#undef UP_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CUDA, UpsampleKernels) {
    m.impl("upsample_nearest1d", upsample_nearest1d_cuda);
    m.impl("upsample_nearest2d", upsample_nearest2d_cuda);
    m.impl("upsample_nearest3d", upsample_nearest3d_cuda);
    m.impl("upsample_linear1d", upsample_linear1d_cuda);
    m.impl("upsample_bilinear2d", upsample_bilinear2d_cuda);
    m.impl("upsample_bicubic2d", upsample_bicubic2d_cuda);
    m.impl("upsample_trilinear3d", upsample_trilinear3d_cuda);
    m.impl("upsample_nearest1d_backward", upsample_nearest1d_backward_cuda);
    m.impl("upsample_nearest2d_backward", upsample_nearest2d_backward_cuda);
    m.impl("upsample_nearest3d_backward", upsample_nearest3d_backward_cuda);
    m.impl("upsample_linear1d_backward", upsample_linear1d_backward_cuda);
    m.impl("upsample_bilinear2d_backward", upsample_bilinear2d_backward_cuda);
    m.impl("upsample_bicubic2d_backward", upsample_bicubic2d_backward_cuda);
    m.impl("upsample_trilinear3d_backward", upsample_trilinear3d_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
