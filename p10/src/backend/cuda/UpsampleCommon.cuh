#pragma once

// Scale, source-index and interpolation helpers shared by the upsampling
// families.  Each family lives in its own translation unit and includes
// this header, so the helpers are inline and free of file-local state.

#include "Tensor.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

// Launch status check shared by the families below.
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t error = (expr); \
        if (error != cudaSuccess) { \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        } \
    } while (0)


// Scaling, index and interpolation helpers.
//
// The scale and the source index are carried in the operand's accumulate type
// so that a double-precision operand keeps double precision through the index
// arithmetic; a float operand pays nothing for the wider type.
template <typename accscalar_t>
inline accscalar_t compute_scales_value_h(const std::optional<double>& scale,
                                          int64_t input_size, int64_t output_size) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<accscalar_t>(1.0 / scale.value())
        : static_cast<accscalar_t>(static_cast<double>(input_size) / output_size);
}

// math wants the output/input ratio, and an explicit scale_factor is used
// as-is (not inverted like in the forward).

template <typename accscalar_t>
inline accscalar_t compute_scales_value_backwards_h(const std::optional<double>& scale,
                                                    int64_t src_size, int64_t dst_size) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<accscalar_t>(scale.value())
        : static_cast<accscalar_t>(static_cast<double>(src_size) / dst_size);
}


inline bool integer_scale_matches(const std::optional<double>& scale,
                                  int64_t input_size, int64_t output_size) {
    if (!scale.has_value() || output_size == 0) return true;
    const double expected = static_cast<double>(input_size) / output_size;
    return std::abs(scale.value() - expected) <=
        1e-6 * std::max(1.0, std::abs(expected));
}


template <typename accscalar_t>
inline accscalar_t area_pixel_compute_scale_h(int64_t input_size, int64_t output_size,
                                              bool align_corners,
                                              const std::optional<double>& scale) {
    if (align_corners) {
        if (output_size > 1) {
            return static_cast<accscalar_t>(input_size - 1) / (output_size - 1);
        }
        return static_cast<accscalar_t>(0);
    }
    return compute_scales_value_h<accscalar_t>(scale, input_size, output_size);
}


template <typename accscalar_t>
__host__ __device__ inline accscalar_t area_pixel_compute_source_index(
        accscalar_t scale, int64_t dst_index, bool align_corners, bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    }
    accscalar_t src_idx =
        scale * (static_cast<accscalar_t>(dst_index) +
                 static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    // The linear modes bound a negative source coordinate to zero.
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
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

// Backward kernels take the full input shape (batch, channels, spatial...).

inline std::vector<int64_t> grad_input_shape(const Tensor& grad_output,
                                             const std::vector<int64_t>& input_size) {
    TP_CHECK(static_cast<int64_t>(input_size.size()) == grad_output.dim(),
             "It is expected input_size equals to ", grad_output.dim(),
             ", but got size ", input_size.size());
    return input_size;
}


inline void launch_dims(int64_t total, dim3& block, dim3& grid) {
    block = dim3(256);
    grid = dim3(static_cast<unsigned>((total + 255) / 256));
}


inline float nearest_exact_scale_h(int64_t in_size, int64_t out_size,
                                   const std::optional<double>& scale) {
    return (scale.has_value() && scale.value() > 0.)
        ? static_cast<float>(1.0 / scale.value())
        : static_cast<float>(static_cast<double>(in_size) / out_size);
}


__host__ __device__ inline int64_t nearest_exact_source_index_h(float scale, int64_t dst_index,
                                                                int64_t input_size) {
    return std::min(static_cast<int64_t>(floorf(scale * (static_cast<float>(dst_index) + 0.5f))),
                    input_size - 1);
}


// Per-dtype dispatch for the interpolating families.
#define UP_DISPATCH(t, ...) \
    switch ((t).dtype()) { \
        case DType::Float32: { using scalar_t = float; using accscalar_t = float; __VA_ARGS__; break; } \
        case DType::Float64: { using scalar_t = double; using accscalar_t = double; __VA_ARGS__; break; } \
        default: TP_THROW(NotImplementedError, "cuda upsample only supports Float32/Float64"); \
    }

// Per-dtype dispatch for the nearest-neighbour families.
#define UP_NEAREST_DISPATCH(t, ...) \
    switch ((t).dtype()) { \
        case DType::Float32: { using scalar_t = float; using accscalar_t = float; __VA_ARGS__; break; } \
        case DType::Float64: { using scalar_t = double; using accscalar_t = double; __VA_ARGS__; break; } \
        case DType::Float16: { using scalar_t = Half; using accscalar_t = float; __VA_ARGS__; break; } \
        case DType::BFloat16: { using scalar_t = BFloat16; using accscalar_t = float; __VA_ARGS__; break; } \
        default: TP_THROW(NotImplementedError, "cuda nearest upsample does not support this dtype"); \
    }

}  // namespace cuda
}  // namespace tensorplay
