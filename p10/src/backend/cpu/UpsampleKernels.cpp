// Upsampling kernels.
//
// Direct port of the ATen upsample kernels.  Each function mirrors its
// counterpart frame kernel; citations at each site:
//   third_party/pytorch/aten/src/ATen/native/UpSample.h
//     compute_scales_value / area_pixel_compute_scale /
//     area_pixel_compute_source_index / nearest_neighbor_compute_source_index
//   third_party/pytorch/aten/src/ATen/native/cuda/UpSample.cuh
//     nearest_neighbor_bw_compute_source_index
//   third_party/pytorch/aten/src/ATen/native/cuda/UpSampleNearest2d.cu
//     upsample_nearest2d_out_frame / upsample_nearest2d_backward_out_frame
//   third_party/pytorch/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu
//     upsample_bilinear2d_out_frame / upsample_bilinear2d_backward_out_frame
//   third_party/pytorch/aten/src/ATen/native/cuda/UpSampleBicubic2d.cu
//     upsample_bicubic2d_out_frame / upsample_bicubic2d_backward_out_frame
//   third_party/pytorch/aten/src/ATen/native/cuda/UpSampleTrilinear3d.cu
//     upsample_trilinear3d_out_frame / upsample_trilinear3d_backward_out_frame
//
// Tensors must be contiguous NCW / NCHW / NCDHW.  The linear/bicubic backwards
// distribute output gradients to input pixels serially on CPU; this matches
// the atomicAdd semantics of the CUDA kernels.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "Parallel.h"
#include <cmath>
#include <algorithm>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

// ---------------------------------------------------------------------------
// UpSample.h index/weight helpers (float scale path; double tensors compute
// scales in float exactly like the ATen CUDA kernels' accscalar_t=float path).
// ---------------------------------------------------------------------------

inline float compute_scales_value_f(const std::optional<double>& scale,
                                    int64_t input_size, int64_t output_size) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<float>(1.0 / scale.value())
        : static_cast<float>(static_cast<double>(input_size) / output_size);
}

inline float area_pixel_compute_scale_f(int64_t input_size, int64_t output_size,
                                        bool align_corners,
                                        const std::optional<double>& scale) {
    if (align_corners) {
        if (output_size > 1) {
            return static_cast<float>(static_cast<double>(input_size - 1) / (output_size - 1));
        }
        return 0.f;
    }
    return compute_scales_value_f(scale, input_size, output_size);
}

inline float area_pixel_compute_source_index_f(float scale, int64_t dst_index,
                                               bool align_corners, bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    }
    float src_idx = scale * (dst_index + 0.5f) - 0.5f;
    // [Note] Follow Opencv resize logic; linear modes bound negatives to zero.
    return (!cubic && src_idx < 0.f) ? 0.f : src_idx;
}

// OpenCV INTER_NEAREST semantics, kept for BC (UpSample.h).
inline int64_t nearest_neighbor_compute_source_index(float scale, int64_t dst_index,
                                                     int64_t input_size) {
    return std::min(static_cast<int64_t>(std::floor(static_cast<float>(dst_index) * scale)),
                    input_size - 1);
}

// UpSample.cuh nearest_neighbor_bw_compute_source_index.
inline int nearest_neighbor_bw_compute_source_index(float scale, int dst_index,
                                                    int output_size) {
    int src_index = std::min(static_cast<int>(std::ceil(static_cast<float>(dst_index) * scale)),
                             output_size);
    return src_index;
}

// UpSample.h cubic machinery (A = -0.75).
template <typename scalar_t>
inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}
template <typename scalar_t>
inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}
template <typename scalar_t>
inline void get_cubic_upsample_coefficients(scalar_t coeffs[4], scalar_t t) {
    constexpr scalar_t A = -0.75;
    scalar_t x1 = t;
    coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1<scalar_t>(x1, A);
    scalar_t x2 = 1.0 - t;
    coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
    coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0, A);
}
template <typename scalar_t>
inline scalar_t cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t t) {
    scalar_t coeffs[4];
    get_cubic_upsample_coefficients<scalar_t>(coeffs, t);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

inline std::vector<int64_t> out_shape(const Tensor& self, const std::vector<int64_t>& out_sizes) {
    std::vector<int64_t> s{self.size(0), self.size(1)};
    s.insert(s.end(), out_sizes.begin(), out_sizes.end());
    return s;
}

#define UP_DISPATCH(t, ...) \
    switch ((t).dtype()) { \
        case DType::Float32: { using scalar_t = float; using accscalar_t = float; __VA_ARGS__; break; } \
        case DType::Float64: { using scalar_t = double; using accscalar_t = double; __VA_ARGS__; break; } \
        default: TP_THROW(NotImplementedError, "upsample only supports Float32/Float64"); \
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Nearest forwards — ATen UpSampleNearest{1d,2d,3d}.cu *_out_frame
// ---------------------------------------------------------------------------

Tensor upsample_nearest1d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                              std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const float width_scale = compute_scales_value_f(scales, W1, W2);
        parallel_for(0, N * C * W2, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t it = begin; it < end; ++it) {
                const int64_t w2 = it % W2;
                const int64_t nc = it / W2;
                const int64_t w1 = (W1 == W2) ? w2 : nearest_neighbor_compute_source_index(width_scale, w2, W1);
                odata[it] = idata[nc * W1 + w1];
            }
        });
    });
    return result;
}

Tensor upsample_nearest2d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                              std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t H1 = in.size(2), W1 = in.size(3);
    const int64_t H2 = output_size[0], W2 = output_size[1];
    if (in.numel() == 0 || H2 == 0 || W2 == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const float height_scale = compute_scales_value_f(scales_h, H1, H2);
        const float width_scale = compute_scales_value_f(scales_w, W1, W2);
        parallel_for(0, N * C * H2 * W2, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t it = begin; it < end; ++it) {
                const int64_t w2 = it % W2;
                const int64_t h2 = (it / W2) % H2;
                const int64_t nc = it / (H2 * W2);
                // ATen UpSampleNearest2d.cu upsample_nearest2d_out_frame
                const int64_t h1 = (H1 == H2) ? h2 : nearest_neighbor_compute_source_index(height_scale, h2, H1);
                const int64_t w1 = (W1 == W2) ? w2 : nearest_neighbor_compute_source_index(width_scale, w2, W1);
                odata[it] = idata[(nc * H1 + h1) * W1 + w1];
            }
        });
    });
    return result;
}

Tensor upsample_nearest3d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                              std::optional<double> scales_d, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t D1 = in.size(2), H1 = in.size(3), W1 = in.size(4);
    const int64_t D2 = output_size[0], H2 = output_size[1], W2 = output_size[2];
    if (in.numel() == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const float depth_scale = compute_scales_value_f(scales_d, D1, D2);
        const float height_scale = compute_scales_value_f(scales_h, H1, H2);
        const float width_scale = compute_scales_value_f(scales_w, W1, W2);
        parallel_for(0, N * C * D2 * H2 * W2, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t it = begin; it < end; ++it) {
                const int64_t w2 = it % W2;
                const int64_t h2 = (it / W2) % H2;
                const int64_t d2 = (it / (W2 * H2)) % D2;
                const int64_t nc = it / (W2 * H2 * D2);
                const int64_t d1 = (D1 == D2) ? d2 : nearest_neighbor_compute_source_index(depth_scale, d2, D1);
                const int64_t h1 = (H1 == H2) ? h2 : nearest_neighbor_compute_source_index(height_scale, h2, H1);
                const int64_t w1 = (W1 == W2) ? w2 : nearest_neighbor_compute_source_index(width_scale, w2, W1);
                odata[it] = idata[((nc * D1 + d1) * H1 + h1) * W1 + w1];
            }
        });
    });
    return result;
}

// ---------------------------------------------------------------------------
// Linear/bilinear/trilinear forwards — ATen UpSampleLinear1d.cu /
// UpSampleBilinear2d.cu / UpSampleTrilinear3d.cu *_out_frame
// ---------------------------------------------------------------------------

Tensor upsample_linear1d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                             bool align_corners, std::optional<double> scales) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t N = in.size(0), C = in.size(1);
    const int64_t W1 = in.size(2), W2 = output_size[0];
    if (in.numel() == 0 || W2 == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const accscalar_t rwidth = area_pixel_compute_scale_f(W1, W2, align_corners, scales);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t w2 = 0; w2 < W2; ++w2) {
                    // ATen UpSampleLinear1d.cu upsample_linear1d_out_frame
                    const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, /*cubic=*/false);
                    const int64_t w1 = static_cast<int64_t>(w1r);
                    const int64_t w1p = (w1 < W1 - 1) ? 1 : 0;
                    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
                    const accscalar_t val =
                        w0lambda * idata[(n * C + c) * W1 + w1] +
                        w1lambda * idata[(n * C + c) * W1 + w1 + w1p];
                    odata[(n * C + c) * W2 + w2] = static_cast<scalar_t>(val);
                }
            }
        }
    });
    return result;
}

Tensor upsample_bilinear2d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                               bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t height1 = in.size(2), width1 = in.size(3);
    const int64_t height2 = output_size[0], width2 = output_size[1];
    if (in.numel() == 0 || height2 == 0 || width2 == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const accscalar_t rheight = area_pixel_compute_scale_f(height1, height2, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale_f(width1, width2, align_corners, scales_w);
        const int64_t num_kernels = height2 * width2;
        parallel_for(0, num_kernels, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t index = begin; index < end; ++index) {
                // ATen UpSampleBilinear2d.cu upsample_bilinear2d_out_frame
                const int64_t w2 = index % width2;
                const int64_t h2 = index / width2;

                const accscalar_t h1r = area_pixel_compute_source_index_f(rheight, h2, align_corners, /*cubic=*/false);
                const int64_t h1 = static_cast<int64_t>(h1r);
                const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
                const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
                const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

                const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, /*cubic=*/false);
                const int64_t w1 = static_cast<int64_t>(w1r);
                const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
                const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

                for (int64_t n = 0; n < batchsize; ++n) {
                    for (int64_t c = 0; c < channels; ++c) {
                        const scalar_t* iptr = idata + (n * channels + c) * height1 * width1;
                        const accscalar_t val = h0lambda *
                                (w0lambda * iptr[h1 * width1 + w1] +
                                 w1lambda * iptr[h1 * width1 + w1 + w1p]) +
                            h1lambda *
                                (w0lambda * iptr[(h1 + h1p) * width1 + w1] +
                                 w1lambda * iptr[(h1 + h1p) * width1 + w1 + w1p]);
                        odata[(n * channels + c) * height2 * width2 + h2 * width2 + w2] =
                            static_cast<scalar_t>(val);
                    }
                }
            }
        });
    });
    return result;
}

Tensor upsample_trilinear3d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                                bool align_corners, std::optional<double> scales_d,
                                std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t depth1 = in.size(2), height1 = in.size(3), width1 = in.size(4);
    const int64_t depth2 = output_size[0], height2 = output_size[1], width2 = output_size[2];
    if (in.numel() == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const accscalar_t rdepth = area_pixel_compute_scale_f(depth1, depth2, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale_f(height1, height2, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale_f(width1, width2, align_corners, scales_w);
        for (int64_t n = 0; n < batchsize; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                const scalar_t* iptr = idata + (n * channels + c) * depth1 * height1 * width1;
                scalar_t* optr = odata + (n * channels + c) * depth2 * height2 * width2;
                for (int64_t t2 = 0; t2 < depth2; ++t2) {
                    for (int64_t h2 = 0; h2 < height2; ++h2) {
                        for (int64_t w2 = 0; w2 < width2; ++w2) {
                            // ATen UpSampleTrilinear3d.cu upsample_trilinear3d_out_frame
                            const accscalar_t t1r = area_pixel_compute_source_index_f(rdepth, t2, align_corners, false);
                            const int64_t t1 = static_cast<int64_t>(t1r);
                            const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
                            const accscalar_t t1lambda = t1r - static_cast<accscalar_t>(t1);
                            const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;

                            const accscalar_t h1r = area_pixel_compute_source_index_f(rheight, h2, align_corners, false);
                            const int64_t h1 = static_cast<int64_t>(h1r);
                            const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
                            const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
                            const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

                            const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, false);
                            const int64_t w1 = static_cast<int64_t>(w1r);
                            const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
                            const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                            const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

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
                            optr[(t2 * height2 + h2) * width2 + w2] = static_cast<scalar_t>(val);
                        }
                    }
                }
            }
        }
    });
    return result;
}

// ---------------------------------------------------------------------------
// Bicubic forward — ATen UpSampleBicubic2d.cu upsample_bicubic2d_out_frame
// ---------------------------------------------------------------------------

Tensor upsample_bicubic2d_cpu(const Tensor& self, std::vector<int64_t> output_size,
                              bool align_corners, std::optional<double> scales_h, std::optional<double> scales_w) {
    Tensor in = self.is_contiguous() ? self : self.contiguous();
    Tensor result = Tensor::empty(out_shape(in, output_size), in.dtype(), in.device());
    const int64_t batchsize = in.size(0), channels = in.size(1);
    const int64_t input_height = in.size(2), input_width = in.size(3);
    const int64_t output_height = output_size[0], output_width = output_size[1];
    if (in.numel() == 0) return result;

    UP_DISPATCH(in, {
        const scalar_t* idata = in.data_ptr<scalar_t>();
        scalar_t* odata = result.data_ptr<scalar_t>();
        const accscalar_t height_scale = area_pixel_compute_scale_f(input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = area_pixel_compute_scale_f(input_width, output_width, align_corners, scales_w);

        auto get_value_bounded = [&](int64_t n, int64_t c, int64_t y, int64_t x) -> scalar_t {
            // ATen UpSample.cuh upsample_get_value_bounded
            const int64_t access_y = std::clamp(y, static_cast<int64_t>(0), input_height - 1);
            const int64_t access_x = std::clamp(x, static_cast<int64_t>(0), input_width - 1);
            return idata[(n * channels + c) * input_height * input_width + access_y * input_width + access_x];
        };

        const int64_t num_kernels = output_height * output_width;
        parallel_for(0, num_kernels, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t index = begin; index < end; ++index) {
                const int64_t output_x = index % output_width;
                const int64_t output_y = index / output_width;
                if (input_height == output_height && input_width == output_width) {
                    for (int64_t n = 0; n < batchsize; ++n) {
                        for (int64_t c = 0; c < channels; ++c) {
                            odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x] =
                                idata[(n * channels + c) * input_height * input_width + output_y * input_width + output_x];
                        }
                    }
                    continue;
                }

                const accscalar_t real_x = area_pixel_compute_source_index_f(width_scale, output_x, align_corners, /*cubic=*/true);
                const int64_t in_x = static_cast<int64_t>(std::floor(real_x));
                const accscalar_t t_x = real_x - static_cast<accscalar_t>(in_x);

                const accscalar_t real_y = area_pixel_compute_source_index_f(height_scale, output_y, align_corners, /*cubic=*/true);
                const int64_t in_y = static_cast<int64_t>(std::floor(real_y));
                const accscalar_t t_y = real_y - static_cast<accscalar_t>(in_y);

                for (int64_t n = 0; n < batchsize; ++n) {
                    for (int64_t c = 0; c < channels; ++c) {
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
                            static_cast<scalar_t>(cubic_interp1d(
                                coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y));
                    }
                }
            }
        });
    });
    return result;
}

// ---------------------------------------------------------------------------
// Nearest backwards — ATen UpSampleNearest{1d,2d,3d}.cu
// *_backward_out_frame (gather formulation, race free)
// ---------------------------------------------------------------------------

Tensor upsample_nearest1d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                       std::vector<int64_t> input_size, std::optional<double> scales) {
    // ATen UpSampleNearest1d.cu upsample_nearest1d_backward_out_frame.
    // "src" = output pixels, "dst" = input pixels; every input pixel gathers
    // the outputs that map onto it (race free, plain assignment).
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t src_dim_w = output_size[0];  // W2
    const int64_t dst_dim_w = input_size[0];   // W1
    if (go.numel() == 0 || src_dim_w == 0 || dst_dim_w == 0) return grad_input;
    const int64_t src_c_stride = src_dim_w;

    UP_DISPATCH(go, {
        const scalar_t* grad_o = go.data_ptr<scalar_t>();
        scalar_t* grad_i = grad_input.data_ptr<scalar_t>();
        const float width_scale = compute_scales_value_f(scales, dst_dim_w, src_dim_w);
        parallel_for(0, dim_c * dst_dim_w, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t dst_idx = begin; dst_idx < end; ++dst_idx) {
                const int64_t c = dst_idx / dst_dim_w;
                const int64_t dst_x = dst_idx % dst_dim_w;
                // note that we do not want to clamp src_x to src_dim_w,
                // since we might intentionally want to skip in case of
                // scale_factor < 1.0
                const int src_x = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x), static_cast<int>(src_dim_w));
                const int src_x_up = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x + 1)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x + 1), static_cast<int>(src_dim_w));

                for (int64_t b = 0; b < dim_b; ++b) {
                    accscalar_t grad = 0;
                    for (int x = src_x; x < src_x_up; ++x) {
                        grad += grad_o[b * dim_c * src_c_stride + c * src_c_stride + x];
                    }
                    grad_i[dst_idx + b * dim_c * dst_dim_w] = static_cast<scalar_t>(grad);
                }
            }
        });
    });
    return grad_input;
}

Tensor upsample_nearest2d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                       std::vector<int64_t> input_size, std::optional<double> scales_h, std::optional<double> scales_w) {
    // ATen UpSampleNearest2d.cu upsample_nearest2d_backward_out_frame.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t src_dim_h = output_size[0], src_dim_w = output_size[1];   // H2, W2
    const int64_t dst_dim_h = input_size[0], dst_dim_w = input_size[1];     // H1, W1
    if (go.numel() == 0 || src_dim_h * src_dim_w == 0 || dst_dim_h * dst_dim_w == 0) return grad_input;
    const int64_t dst_c_stride = dst_dim_h * dst_dim_w;
    const int64_t src_c_stride = src_dim_h * src_dim_w;

    UP_DISPATCH(go, {
        const scalar_t* grad_o = go.data_ptr<scalar_t>();
        scalar_t* grad_i = grad_input.data_ptr<scalar_t>();
        const float height_scale = compute_scales_value_f(scales_h, dst_dim_h, src_dim_h);
        const float width_scale = compute_scales_value_f(scales_w, dst_dim_w, src_dim_w);
        parallel_for(0, dim_c * dst_c_stride, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t dst_idx = begin; dst_idx < end; ++dst_idx) {
                const int64_t c = (dst_idx / dst_c_stride) % dim_c;
                const int64_t dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
                // note that we do not want to clamp src_y to src_dim_y,
                // since we might intentionally want to skip in case of
                // scale_factor < 1.0
                const int src_y = (src_dim_h == dst_dim_h)
                    ? static_cast<int>(dst_y)
                    : nearest_neighbor_bw_compute_source_index(height_scale, static_cast<int>(dst_y), static_cast<int>(src_dim_h));
                const int src_y_up = (src_dim_h == dst_dim_h)
                    ? static_cast<int>(dst_y + 1)
                    : nearest_neighbor_bw_compute_source_index(height_scale, static_cast<int>(dst_y + 1), static_cast<int>(src_dim_h));
                const int64_t dst_x = dst_idx % dst_dim_w;
                const int src_x = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x), static_cast<int>(src_dim_w));
                const int src_x_up = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x + 1)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x + 1), static_cast<int>(src_dim_w));

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
        });
    });
    return grad_input;
}

Tensor upsample_nearest3d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                       std::vector<int64_t> input_size, std::optional<double> scales_d,
                                       std::optional<double> scales_h, std::optional<double> scales_w) {
    // ATen UpSampleNearest3d.cu upsample_nearest3d_backward_out_frame.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t dim_b = go.size(0), dim_c = go.size(1);
    const int64_t src_dim_d = output_size[0], src_dim_h = output_size[1], src_dim_w = output_size[2];
    const int64_t dst_dim_d = input_size[0], dst_dim_h = input_size[1], dst_dim_w = input_size[2];
    if (go.numel() == 0) return grad_input;
    if (src_dim_d * src_dim_h * src_dim_w == 0 || dst_dim_d * dst_dim_h * dst_dim_w == 0) return grad_input;
    const int64_t dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
    const int64_t src_c_stride = src_dim_d * src_dim_h * src_dim_w;

    UP_DISPATCH(go, {
        const scalar_t* grad_o = go.data_ptr<scalar_t>();
        scalar_t* grad_i = grad_input.data_ptr<scalar_t>();
        const float depth_scale = compute_scales_value_f(scales_d, dst_dim_d, src_dim_d);
        const float height_scale = compute_scales_value_f(scales_h, dst_dim_h, src_dim_h);
        const float width_scale = compute_scales_value_f(scales_w, dst_dim_w, src_dim_w);
        parallel_for(0, dim_c * dst_c_stride, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t dst_idx = begin; dst_idx < end; ++dst_idx) {
                const int64_t c = (dst_idx / dst_c_stride) % dim_c;
                const int64_t dst_t = (dst_idx / (dst_dim_h * dst_dim_w)) % dst_dim_d;
                const int64_t dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
                const int64_t dst_x = dst_idx % dst_dim_w;
                const int src_t = (src_dim_d == dst_dim_d)
                    ? static_cast<int>(dst_t)
                    : nearest_neighbor_bw_compute_source_index(depth_scale, static_cast<int>(dst_t), static_cast<int>(src_dim_d));
                const int src_t_up = (src_dim_d == dst_dim_d)
                    ? static_cast<int>(dst_t + 1)
                    : nearest_neighbor_bw_compute_source_index(depth_scale, static_cast<int>(dst_t + 1), static_cast<int>(src_dim_d));
                const int src_y = (src_dim_h == dst_dim_h)
                    ? static_cast<int>(dst_y)
                    : nearest_neighbor_bw_compute_source_index(height_scale, static_cast<int>(dst_y), static_cast<int>(src_dim_h));
                const int src_y_up = (src_dim_h == dst_dim_h)
                    ? static_cast<int>(dst_y + 1)
                    : nearest_neighbor_bw_compute_source_index(height_scale, static_cast<int>(dst_y + 1), static_cast<int>(src_dim_h));
                const int src_x = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x), static_cast<int>(src_dim_w));
                const int src_x_up = (src_dim_w == dst_dim_w)
                    ? static_cast<int>(dst_x + 1)
                    : nearest_neighbor_bw_compute_source_index(width_scale, static_cast<int>(dst_x + 1), static_cast<int>(src_dim_w));

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
        });
    });
    return grad_input;
}

// ---------------------------------------------------------------------------
// Linear backwards — ATen UpSampleLinear1d.cu / UpSampleBilinear2d.cu /
// UpSampleTrilinear3d.cu *_backward_out_frame (scatter formulation with
// atomicAdd on CUDA; serial accumulation here preserves the same semantics).
// ---------------------------------------------------------------------------

Tensor upsample_linear1d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                      std::vector<int64_t> input_size, bool align_corners,
                                      std::optional<double> scales) {
    // ATen UpSampleLinear1d.cu upsample_linear1d_backward_out_frame.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t N = go.size(0), C = go.size(1);
    const int64_t W2 = output_size[0], W1 = input_size[0];
    if (go.numel() == 0 || W2 == 0 || W1 == 0) return grad_input;

    UP_DISPATCH(go, {
        const scalar_t* odata = go.data_ptr<scalar_t>();
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        const accscalar_t rwidth = area_pixel_compute_scale_f(W1, W2, align_corners, scales);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t w2 = 0; w2 < W2; ++w2) {
                    const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, /*cubic=*/false);
                    const int64_t w1 = static_cast<int64_t>(w1r);
                    const int64_t w1p = (w1 < W1 - 1) ? 1 : 0;
                    const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
                    const accscalar_t val = odata[(n * C + c) * W2 + w2];
                    idata[(n * C + c) * W1 + w1] += static_cast<scalar_t>(w0lambda * val);
                    idata[(n * C + c) * W1 + w1 + w1p] += static_cast<scalar_t>(w1lambda * val);
                }
            }
        }
    });
    return grad_input;
}

Tensor upsample_bilinear2d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                        std::vector<int64_t> input_size, bool align_corners,
                                        std::optional<double> scales_h, std::optional<double> scales_w) {
    // ATen UpSampleBilinear2d.cu upsample_bilinear2d_backward_out_frame
    // (non-ROCm path): iterate output pixels, distribute to the four corners.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t height1 = input_size[0], width1 = input_size[1];
    const int64_t height2 = output_size[0], width2 = output_size[1];
    if (go.numel() == 0 || height2 * width2 == 0 || height1 * width1 == 0) return grad_input;

    UP_DISPATCH(go, {
        const scalar_t* odata = go.data_ptr<scalar_t>();
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        const accscalar_t rheight = area_pixel_compute_scale_f(height1, height2, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale_f(width1, width2, align_corners, scales_w);
        for (int64_t n = 0; n < batchsize; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                const scalar_t* optr = odata + (n * channels + c) * height2 * width2;
                scalar_t* iptr = idata + (n * channels + c) * height1 * width1;
                for (int64_t h2 = 0; h2 < height2; ++h2) {
                    const accscalar_t h1r = area_pixel_compute_source_index_f(rheight, h2, align_corners, false);
                    const int64_t h1 = static_cast<int64_t>(h1r);
                    const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
                    const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
                    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
                    for (int64_t w2 = 0; w2 < width2; ++w2) {
                        const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, false);
                        const int64_t w1 = static_cast<int64_t>(w1r);
                        const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
                        const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
                        const accscalar_t val = optr[h2 * width2 + w2];
                        iptr[h1 * width1 + w1] += static_cast<scalar_t>(h0lambda * w0lambda * val);
                        iptr[h1 * width1 + w1 + w1p] += static_cast<scalar_t>(h0lambda * w1lambda * val);
                        iptr[(h1 + h1p) * width1 + w1] += static_cast<scalar_t>(h1lambda * w0lambda * val);
                        iptr[(h1 + h1p) * width1 + w1 + w1p] += static_cast<scalar_t>(h1lambda * w1lambda * val);
                    }
                }
            }
        }
    });
    return grad_input;
}

Tensor upsample_trilinear3d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                         std::vector<int64_t> input_size, bool align_corners,
                                         std::optional<double> scales_d, std::optional<double> scales_h,
                                         std::optional<double> scales_w) {
    // ATen UpSampleTrilinear3d.cu upsample_trilinear3d_backward_out_frame.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t depth1 = input_size[0], height1 = input_size[1], width1 = input_size[2];
    const int64_t depth2 = output_size[0], height2 = output_size[1], width2 = output_size[2];
    if (go.numel() == 0) return grad_input;

    UP_DISPATCH(go, {
        const scalar_t* odata = go.data_ptr<scalar_t>();
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        const accscalar_t rdepth = area_pixel_compute_scale_f(depth1, depth2, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale_f(height1, height2, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale_f(width1, width2, align_corners, scales_w);
        for (int64_t n = 0; n < batchsize; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                const scalar_t* optr = odata + (n * channels + c) * depth2 * height2 * width2;
                scalar_t* iptr = idata + (n * channels + c) * depth1 * height1 * width1;
                for (int64_t t2 = 0; t2 < depth2; ++t2) {
                    const accscalar_t t1r = area_pixel_compute_source_index_f(rdepth, t2, align_corners, false);
                    const int64_t t1 = static_cast<int64_t>(t1r);
                    const int64_t t1p = (t1 < depth1 - 1) ? 1 : 0;
                    const accscalar_t t1lambda = t1r - static_cast<accscalar_t>(t1);
                    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
                    for (int64_t h2 = 0; h2 < height2; ++h2) {
                        const accscalar_t h1r = area_pixel_compute_source_index_f(rheight, h2, align_corners, false);
                        const int64_t h1 = static_cast<int64_t>(h1r);
                        const int64_t h1p = (h1 < height1 - 1) ? 1 : 0;
                        const accscalar_t h1lambda = h1r - static_cast<accscalar_t>(h1);
                        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
                        for (int64_t w2 = 0; w2 < width2; ++w2) {
                            const accscalar_t w1r = area_pixel_compute_source_index_f(rwidth, w2, align_corners, false);
                            const int64_t w1 = static_cast<int64_t>(w1r);
                            const int64_t w1p = (w1 < width1 - 1) ? 1 : 0;
                            const accscalar_t w1lambda = w1r - static_cast<accscalar_t>(w1);
                            const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
                            const accscalar_t val = optr[(t2 * height2 + h2) * width2 + w2];
                            iptr[(t1 * height1 + h1) * width1 + w1] += static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * val);
                            iptr[(t1 * height1 + h1) * width1 + w1 + w1p] += static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * val);
                            iptr[(t1 * height1 + h1 + h1p) * width1 + w1] += static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * val);
                            iptr[(t1 * height1 + h1 + h1p) * width1 + w1 + w1p] += static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * val);
                        }
                    }
                }
            }
        }
    });
    return grad_input;
}

Tensor upsample_bicubic2d_backward_cpu(const Tensor& grad_output, std::vector<int64_t> output_size,
                                       std::vector<int64_t> input_size, bool align_corners,
                                       std::optional<double> scales_h, std::optional<double> scales_w) {
    // ATen UpSampleBicubic2d.cu upsample_bicubic2d_backward_out_frame:
    // scatter each output gradient into the bounded 4x4 input window.
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor grad_input = Tensor::zeros(out_shape(go, input_size), go.dtype(), go.device());
    const int64_t batchsize = go.size(0), channels = go.size(1);
    const int64_t input_height = input_size[0], input_width = input_size[1];
    const int64_t output_height = output_size[0], output_width = output_size[1];
    if (go.numel() == 0) return grad_input;

    UP_DISPATCH(go, {
        const scalar_t* odata = go.data_ptr<scalar_t>();
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        const accscalar_t height_scale = area_pixel_compute_scale_f(input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = area_pixel_compute_scale_f(input_width, output_width, align_corners, scales_w);

        auto increment_value_bounded = [&](int64_t n, int64_t c, int64_t y, int64_t x, accscalar_t value) {
            // ATen UpSample.cuh upsample_increment_value_bounded
            const int64_t access_y = std::clamp(y, static_cast<int64_t>(0), input_height - 1);
            const int64_t access_x = std::clamp(x, static_cast<int64_t>(0), input_width - 1);
            idata[(n * channels + c) * input_height * input_width + access_y * input_width + access_x] +=
                static_cast<scalar_t>(value);
        };

        for (int64_t n = 0; n < batchsize; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t output_y = 0; output_y < output_height; ++output_y) {
                    const accscalar_t real_y = area_pixel_compute_source_index_f(height_scale, output_y, align_corners, /*cubic=*/true);
                    const int64_t input_y = static_cast<int64_t>(std::floor(real_y));
                    const accscalar_t t_y = real_y - static_cast<accscalar_t>(input_y);
                    for (int64_t output_x = 0; output_x < output_width; ++output_x) {
                        const accscalar_t real_x = area_pixel_compute_source_index_f(width_scale, output_x, align_corners, /*cubic=*/true);
                        const int64_t input_x = static_cast<int64_t>(std::floor(real_x));
                        const accscalar_t t_x = real_x - static_cast<accscalar_t>(input_x);

                        accscalar_t x_coeffs[4];
                        accscalar_t y_coeffs[4];
                        get_cubic_upsample_coefficients(x_coeffs, t_x);
                        get_cubic_upsample_coefficients(y_coeffs, t_y);

                        const scalar_t out_value = odata[(n * channels + c) * output_height * output_width + output_y * output_width + output_x];
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                increment_value_bounded(n, c, input_y - 1 + i, input_x - 1 + j,
                                                        static_cast<accscalar_t>(out_value) * y_coeffs[i] * x_coeffs[j]);
                            }
                        }
                    }
                }
            }
        }
    });
    return grad_input;
}

#undef UP_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CPU, UpsampleKernels) {
    m.impl("upsample_nearest1d", upsample_nearest1d_cpu);
    m.impl("upsample_nearest2d", upsample_nearest2d_cpu);
    m.impl("upsample_nearest3d", upsample_nearest3d_cpu);
    m.impl("upsample_linear1d", upsample_linear1d_cpu);
    m.impl("upsample_bilinear2d", upsample_bilinear2d_cpu);
    m.impl("upsample_bicubic2d", upsample_bicubic2d_cpu);
    m.impl("upsample_trilinear3d", upsample_trilinear3d_cpu);
    m.impl("upsample_nearest1d_backward", upsample_nearest1d_backward_cpu);
    m.impl("upsample_nearest2d_backward", upsample_nearest2d_backward_cpu);
    m.impl("upsample_nearest3d_backward", upsample_nearest3d_backward_cpu);
    m.impl("upsample_linear1d_backward", upsample_linear1d_backward_cpu);
    m.impl("upsample_bilinear2d_backward", upsample_bilinear2d_backward_cpu);
    m.impl("upsample_bicubic2d_backward", upsample_bicubic2d_backward_cpu);
    m.impl("upsample_trilinear3d_backward", upsample_trilinear3d_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
