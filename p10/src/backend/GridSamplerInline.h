#pragma once
// Shared host/device coordinate math for the grid_sampler kernels.  Port of
// third_party/pytorch/aten/src/ATen/native/GridSampler.h (coordinate mapping)
// plus the cubic-convolution helpers of aten/src/ATen/native/UpSample.h.

#ifdef __CUDACC__
#define TP_GS_INLINE __host__ __device__ inline
#else
#define TP_GS_INLINE inline
#endif

#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace gridsampler {

// interpolation_mode values (ATen GridSamplerUtils.h GridSamplerInterpolation)
enum Interp : int { Bilinear = 0, Nearest = 1, Bicubic = 2 };
// padding_mode values (ATen GridSamplerUtils.h GridSamplerPadding)
enum Pad : int { Zeros = 0, Border = 1, Reflection = 2 };

// Unnormalizes a coordinate from the -1..+1 scale to its pixel index value,
// where each pixel is viewed as the area between (idx - 0.5) and (idx + 0.5).
template <typename scalar_t>
TP_GS_INLINE scalar_t unnormalize(scalar_t coord, int64_t size, bool align_corners) {
    if (align_corners) {
        // [-1, 1] -> [0, size - 1]
        return ((coord + 1) / 2) * (size - 1);
    }
    // [-1, 1] -> [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
}

// Same as unnormalize, additionally returning d output / d input in *grad_in.
template <typename scalar_t>
TP_GS_INLINE scalar_t unnormalize_set_grad(scalar_t coord, int64_t size,
                                           bool align_corners, scalar_t* grad_in) {
    if (align_corners) {
        *grad_in = static_cast<scalar_t>(size - 1) / 2;
        return ((coord + 1) / 2) * (size - 1);
    }
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1) * size - 1) / 2;
}

template <typename scalar_t>
TP_GS_INLINE scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
    scalar_t lo = static_cast<scalar_t>(0);
    scalar_t hi = static_cast<scalar_t>(clip_limit - 1);
    return in < lo ? lo : (in > hi ? hi : in);
}

// Border clip with gradient: the borders themselves count as out of bounds,
// so the gradient is zero unless strictly inside (0, clip_limit - 1).
template <typename scalar_t>
TP_GS_INLINE scalar_t clip_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                scalar_t* grad_in) {
    if (in <= static_cast<scalar_t>(0)) {
        *grad_in = static_cast<scalar_t>(0);
        return static_cast<scalar_t>(0);
    }
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
        *grad_in = static_cast<scalar_t>(0);
        return max;
    }
    *grad_in = static_cast<scalar_t>(1);
    return in;
}

// Reflects coordinates until they fall between low and high (inclusive).
// Bounds are passed as twice their value so half-integers stay integral.
template <typename scalar_t>
TP_GS_INLINE scalar_t reflect_coordinates(scalar_t in, int64_t twice_low,
                                          int64_t twice_high) {
    if (twice_low == twice_high) {
        return static_cast<scalar_t>(0);
    }
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = std::fabs(in - min);
    scalar_t extra = std::fmod(in, span);
    int flips = static_cast<int>(std::floor(in / span));
    if (flips % 2 == 0) {
        return extra + min;
    }
    return span - extra + min;
}

template <typename scalar_t>
TP_GS_INLINE scalar_t reflect_coordinates_set_grad(scalar_t in, int64_t twice_low,
                                                   int64_t twice_high, scalar_t* grad_in) {
    if (twice_low == twice_high) {
        *grad_in = static_cast<scalar_t>(0);
        return static_cast<scalar_t>(0);
    }
    int grad_in_mult_;
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = in - min;
    if (in < static_cast<scalar_t>(0)) {
        grad_in_mult_ = -1;
        in = -in;
    } else {
        grad_in_mult_ = 1;
    }
    scalar_t extra = std::fmod(in, span);
    int flips = static_cast<int>(std::floor(in / span));
    if (flips % 2 == 0) {
        *grad_in = static_cast<scalar_t>(grad_in_mult_);
        return extra + min;
    }
    *grad_in = static_cast<scalar_t>(-grad_in_mult_);
    return span - extra + min;
}

// Maps out-of-boundary coordinates back into bounds; only affects
// padding_mode == Border or Reflection.
template <typename scalar_t>
TP_GS_INLINE scalar_t compute_coordinates(scalar_t coord, int64_t size,
                                          int padding_mode, bool align_corners) {
    if (padding_mode == Pad::Border) {
        coord = clip_coordinates(coord, size);
    } else if (padding_mode == Pad::Reflection) {
        if (align_corners) {
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        } else {
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        coord = clip_coordinates(coord, size);
    }
    return coord;
}

// Normalized grid coordinate -> source pixel index.
template <typename scalar_t>
TP_GS_INLINE scalar_t compute_source_index(scalar_t coord, int64_t size,
                                           int padding_mode, bool align_corners) {
    coord = unnormalize(coord, size, align_corners);
    return compute_coordinates(coord, size, padding_mode, align_corners);
}

// Same, additionally returning d source / d grid-coordinate in *grad_in.
template <typename scalar_t>
TP_GS_INLINE scalar_t compute_source_index_set_grad(scalar_t coord, int64_t size,
                                                    int padding_mode, bool align_corners,
                                                    scalar_t* grad_in) {
    scalar_t grad_clip, grad_refl;
    coord = unnormalize_set_grad(coord, size, align_corners, grad_in);
    if (padding_mode == Pad::Border) {
        coord = clip_coordinates_set_grad(coord, size, &grad_clip);
        *grad_in = (*grad_in) * grad_clip;
    } else if (padding_mode == Pad::Reflection) {
        if (align_corners) {
            coord = reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
        } else {
            coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
        }
        coord = clip_coordinates_set_grad(coord, size, &grad_clip);
        *grad_in = (*grad_in) * grad_refl * grad_clip;
    }
    return coord;
}

TP_GS_INLINE bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
}

TP_GS_INLINE bool within_bounds_3d(int64_t d, int64_t h, int64_t w,
                                   int64_t D, int64_t H, int64_t W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

// Reads a pixel applying the padding mode per tap (used by bicubic, whose
// taps are individually padding-adjusted rather than the base coordinate).
// scalar_t is the compute type; storage_t the (possibly reduced) storage type.
template <typename scalar_t, typename storage_t>
TP_GS_INLINE scalar_t get_value_bounded(const storage_t* data, scalar_t x, scalar_t y,
                                        int64_t W, int64_t H, int64_t sW, int64_t sH,
                                        int padding_mode, bool align_corners) {
    x = compute_coordinates(x, W, padding_mode, align_corners);
    y = compute_coordinates(y, H, padding_mode, align_corners);
    int64_t ix = static_cast<int64_t>(x);
    int64_t iy = static_cast<int64_t>(y);
    if (within_bounds_2d(iy, ix, H, W)) {
        return static_cast<scalar_t>(data[iy * sH + ix * sW]);
    }
    return static_cast<scalar_t>(0);
}

// --- cubic convolution (alpha = -0.75), ATen UpSample.h -------------------
template <typename scalar_t>
TP_GS_INLINE scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
TP_GS_INLINE scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
TP_GS_INLINE void get_cubic_upsampling_coefficients(scalar_t coeffs[4], scalar_t t) {
    scalar_t A = -0.75;
    coeffs[0] = cubic_convolution2<scalar_t>(t + 1, A);
    coeffs[1] = cubic_convolution1<scalar_t>(t, A);
    coeffs[2] = cubic_convolution1<scalar_t>(1 - t, A);
    coeffs[3] = cubic_convolution2<scalar_t>(2 - t, A);
}

// d coeff / d t, matching get_cubic_upsampling_coefficients (ATen GridSampler.h
// get_cubic_coefficients_grad).
template <typename scalar_t>
TP_GS_INLINE void get_cubic_coefficients_grad(scalar_t coeffs[4], scalar_t t) {
    scalar_t A = -0.75;
    scalar_t x;
    x = -1 - t;  // 1 < |x| < 2
    coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
    x = -t;      // |x| <= 1
    coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
    x = 1 - t;   // |x| <= 1
    coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
    x = 2 - t;   // 1 < |x| < 2
    coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

template <typename scalar_t>
TP_GS_INLINE scalar_t cubic_interp1d(scalar_t p0, scalar_t p1, scalar_t p2,
                                     scalar_t p3, scalar_t t) {
    scalar_t coeffs[4];
    get_cubic_upsampling_coefficients<scalar_t>(coeffs, t);
    return coeffs[0] * p0 + coeffs[1] * p1 + coeffs[2] * p2 + coeffs[3] * p3;
}

}  // namespace gridsampler
}  // namespace tensorplay

#undef TP_GS_INLINE
