// grid_sampler_2d / grid_sampler_3d CUDA kernels.
//
// and grid_sampler_{2d,3d}_backward_kernel) over contiguous tensors:
//   interpolation_mode: 0=Bilinear 1=Nearest 2=Bicubic(2d only)
//   padding_mode:       0=Zeros 1=Border 2=Reflection
// grad_input scatter uses gpuAtomicAdd (Atomic.cuh).
#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include "Atomic.cuh"
#include "../GridSamplerInline.h"
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <cmath>

namespace tensorplay {
namespace cuda {

using namespace tensorplay::gridsampler;

namespace {

constexpr int kThreads = 256;

inline int64_t gs_grid_blocks(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

template <typename storage_t, typename compute_t>
__device__ inline void gs_safe_add(storage_t* data, int64_t off, compute_t delta) {
    gpuAtomicAdd(data + off, static_cast<storage_t>(delta));
}

template <typename storage_t, typename compute_t>
__global__ void grid_sampler_2d_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ grid,
        storage_t* __restrict__ output,
        const int64_t C, const int64_t inp_H, const int64_t inp_W,
        const int64_t out_H, const int64_t out_W,
        const int interpolation_mode, const int padding_mode,
        const bool align_corners) {
    const int64_t out_vol = out_H * out_W;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t w = index % out_W;
        const int64_t h = (index / out_W) % out_H;
        const int64_t n = index / out_vol;
        const storage_t* grid_ptr_NHW = grid + index * 2;

        compute_t ix = compute_source_index<compute_t>(
            static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, padding_mode, align_corners);
        compute_t iy = compute_source_index<compute_t>(
            static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, padding_mode, align_corners);

        const storage_t* inp_ptr_N = input + n * C * inp_H * inp_W;
        storage_t* out_ptr_NHW = output + n * C * out_vol + h * out_W + w;

        if (interpolation_mode == Interp::Bilinear) {
            int64_t ix_nw = static_cast<int64_t>(::floor(ix));
            int64_t iy_nw = static_cast<int64_t>(::floor(iy));
            int64_t ix_ne = ix_nw + 1, iy_ne = iy_nw;
            int64_t ix_sw = ix_nw,     iy_sw = iy_nw + 1;
            int64_t ix_se = ix_nw + 1, iy_se = iy_nw + 1;

            compute_t nw = (ix_se - ix) * (iy_se - iy);
            compute_t ne = (ix - ix_sw) * (iy_sw - iy);
            compute_t sw = (ix_ne - ix) * (iy - iy_ne);
            compute_t se = (ix - ix_nw) * (iy - iy_nw);

            for (int64_t c = 0; c < C; ++c) {
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                compute_t acc = 0;
                if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W))
                    acc += static_cast<compute_t>(inp_ptr_NC[iy_nw * inp_W + ix_nw]) * nw;
                if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W))
                    acc += static_cast<compute_t>(inp_ptr_NC[iy_ne * inp_W + ix_ne]) * ne;
                if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W))
                    acc += static_cast<compute_t>(inp_ptr_NC[iy_sw * inp_W + ix_sw]) * sw;
                if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W))
                    acc += static_cast<compute_t>(inp_ptr_NC[iy_se * inp_W + ix_se]) * se;
                out_ptr_NHW[c * out_vol] = static_cast<storage_t>(acc);
            }
        } else if (interpolation_mode == Interp::Nearest) {
            int64_t ix_near = static_cast<int64_t>(::nearbyint(ix));
            int64_t iy_near = static_cast<int64_t>(::nearbyint(iy));
            for (int64_t c = 0; c < C; ++c) {
                storage_t v = static_cast<storage_t>(0);
                if (within_bounds_2d(iy_near, ix_near, inp_H, inp_W))
                    v = inp_ptr_N[c * inp_H * inp_W + iy_near * inp_W + ix_near];
                out_ptr_NHW[c * out_vol] = v;
            }
        } else {  // Bicubic
            compute_t x = unnormalize<compute_t>(
                static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, align_corners);
            compute_t y = unnormalize<compute_t>(
                static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, align_corners);
            compute_t ix_nw = ::floor(x);
            compute_t iy_nw = ::floor(y);
            compute_t tx = x - ix_nw;
            compute_t ty = y - iy_nw;
            for (int64_t c = 0; c < C; ++c) {
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                compute_t coeffs[4];
                #pragma unroll 4
                for (int64_t i = 0; i < 4; ++i) {
                    coeffs[i] = cubic_interp1d<compute_t>(
                        get_value_bounded<compute_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                        get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                        get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                        get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                        tx);
                }
                out_ptr_NHW[c * out_vol] = static_cast<storage_t>(
                    cubic_interp1d<compute_t>(coeffs[0], coeffs[1], coeffs[2], coeffs[3], ty));
            }
        }
    }
}

template <typename storage_t, typename compute_t>
__global__ void grid_sampler_2d_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ grid,
        storage_t* __restrict__ grad_input,   // zeros-initialized or null
        storage_t* __restrict__ grad_grid,    // or null
        const int64_t C, const int64_t inp_H, const int64_t inp_W,
        const int64_t out_H, const int64_t out_W,
        const int interpolation_mode, const int padding_mode,
        const bool align_corners, const bool need_gi, const bool need_gg) {
    const int64_t out_vol = out_H * out_W;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t w = index % out_W;
        const int64_t h = (index / out_W) % out_H;
        const int64_t n = index / out_vol;
        const storage_t* grid_ptr_NHW = grid + index * 2;

        compute_t gix_mult, giy_mult;
        compute_t ix = compute_source_index_set_grad<compute_t>(
            static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, padding_mode, align_corners, &gix_mult);
        compute_t iy = compute_source_index_set_grad<compute_t>(
            static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, padding_mode, align_corners, &giy_mult);

        const storage_t* inp_ptr_N = input + n * C * inp_H * inp_W;

        if (interpolation_mode == Interp::Bilinear) {
            int64_t ix_nw = static_cast<int64_t>(::floor(ix));
            int64_t iy_nw = static_cast<int64_t>(::floor(iy));
            int64_t ix_ne = ix_nw + 1, iy_ne = iy_nw;
            int64_t ix_sw = ix_nw,     iy_sw = iy_nw + 1;
            int64_t ix_se = ix_nw + 1, iy_se = iy_nw + 1;

            compute_t nw = (ix_se - ix) * (iy_se - iy);
            compute_t ne = (ix - ix_sw) * (iy_sw - iy);
            compute_t sw = (ix_ne - ix) * (iy - iy_ne);
            compute_t se = (ix - ix_nw) * (iy - iy_nw);

            compute_t gix = 0, giy = 0;
            for (int64_t c = 0; c < C; ++c) {
                const compute_t gOut = static_cast<compute_t>(
                    grad_output[(n * C + c) * out_vol + h * out_W + w]);
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                storage_t* gInp_ptr_NC = need_gi ? grad_input + (n * C + c) * inp_H * inp_W : nullptr;

                if (need_gi) {
                    if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W))
                        gs_safe_add(gInp_ptr_NC, iy_nw * inp_W + ix_nw, nw * gOut);
                    if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W))
                        gs_safe_add(gInp_ptr_NC, iy_ne * inp_W + ix_ne, ne * gOut);
                    if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W))
                        gs_safe_add(gInp_ptr_NC, iy_sw * inp_W + ix_sw, sw * gOut);
                    if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W))
                        gs_safe_add(gInp_ptr_NC, iy_se * inp_W + ix_se, se * gOut);
                }

                if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                    compute_t v = static_cast<compute_t>(inp_ptr_NC[iy_nw * inp_W + ix_nw]);
                    gix -= v * (iy_se - iy) * gOut;
                    giy -= v * (ix_se - ix) * gOut;
                }
                if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                    compute_t v = static_cast<compute_t>(inp_ptr_NC[iy_ne * inp_W + ix_ne]);
                    gix += v * (iy_sw - iy) * gOut;
                    giy -= v * (ix - ix_sw) * gOut;
                }
                if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                    compute_t v = static_cast<compute_t>(inp_ptr_NC[iy_sw * inp_W + ix_sw]);
                    gix -= v * (iy - iy_ne) * gOut;
                    giy += v * (ix_ne - ix) * gOut;
                }
                if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                    compute_t v = static_cast<compute_t>(inp_ptr_NC[iy_se * inp_W + ix_se]);
                    gix += v * (iy - iy_nw) * gOut;
                    giy += v * (ix - ix_nw) * gOut;
                }
            }
            if (need_gg) {
                storage_t* gGrid_ptr_NHW = grad_grid + index * 2;
                gGrid_ptr_NHW[0] = static_cast<storage_t>(gix_mult * gix);
                gGrid_ptr_NHW[1] = static_cast<storage_t>(giy_mult * giy);
            }
        } else if (interpolation_mode == Interp::Nearest) {
            if (need_gi) {
                int64_t ix_near = static_cast<int64_t>(::nearbyint(ix));
                int64_t iy_near = static_cast<int64_t>(::nearbyint(iy));
                if (within_bounds_2d(iy_near, ix_near, inp_H, inp_W)) {
                    for (int64_t c = 0; c < C; ++c) {
                        const compute_t gOut = static_cast<compute_t>(
                            grad_output[(n * C + c) * out_vol + h * out_W + w]);
                        gs_safe_add(grad_input + (n * C + c) * inp_H * inp_W,
                                    iy_near * inp_W + ix_near, gOut);
                    }
                }
            }
            if (need_gg) {
                storage_t* gGrid_ptr_NHW = grad_grid + index * 2;
                gGrid_ptr_NHW[0] = static_cast<storage_t>(0);
                gGrid_ptr_NHW[1] = static_cast<storage_t>(0);
            }
        } else {  // Bicubic
            compute_t x = unnormalize_set_grad<compute_t>(
                static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, align_corners, &gix_mult);
            compute_t y = unnormalize_set_grad<compute_t>(
                static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, align_corners, &giy_mult);
            compute_t ix_nw = ::floor(x);
            compute_t iy_nw = ::floor(y);
            compute_t tx = x - ix_nw;
            compute_t ty = y - iy_nw;

            compute_t x_coeffs[4], y_coeffs[4], x_coeffs_grad[4], y_coeffs_grad[4];
            get_cubic_upsampling_coefficients<compute_t>(x_coeffs, tx);
            get_cubic_upsampling_coefficients<compute_t>(y_coeffs, ty);
            get_cubic_coefficients_grad<compute_t>(x_coeffs_grad, tx);
            get_cubic_coefficients_grad<compute_t>(y_coeffs_grad, ty);

            compute_t gix = 0, giy = 0;
            for (int64_t c = 0; c < C; ++c) {
                const compute_t gOut = static_cast<compute_t>(
                    grad_output[(n * C + c) * out_vol + h * out_W + w]);
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                storage_t* gInp_ptr_NC = need_gi ? grad_input + (n * C + c) * inp_H * inp_W : nullptr;
                #pragma unroll 4
                for (int64_t i = 0; i < 4; ++i) {
                    #pragma unroll 4
                    for (int64_t j = 0; j < 4; ++j) {
                        const compute_t tap_x = ix_nw - 1 + i;
                        const compute_t tap_y = iy_nw - 1 + j;
                        if (need_gi) {
                            compute_t ax = compute_coordinates(tap_x, inp_W, padding_mode, align_corners);
                            compute_t ay = compute_coordinates(tap_y, inp_H, padding_mode, align_corners);
                            int64_t iax = static_cast<int64_t>(ax);
                            int64_t iay = static_cast<int64_t>(ay);
                            if (within_bounds_2d(iay, iax, inp_H, inp_W))
                                gs_safe_add(gInp_ptr_NC, iay * inp_W + iax,
                                            gOut * x_coeffs[i] * y_coeffs[j]);
                        }
                        compute_t val = get_value_bounded<compute_t>(
                            inp_ptr_NC, tap_x, tap_y, inp_W, inp_H, 1, inp_W,
                            padding_mode, align_corners);
                        gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
                        giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
                    }
                }
            }
            if (need_gg) {
                storage_t* gGrid_ptr_NHW = grad_grid + index * 2;
                gGrid_ptr_NHW[0] = static_cast<storage_t>(gix_mult * gix);
                gGrid_ptr_NHW[1] = static_cast<storage_t>(giy_mult * giy);
            }
        }
    }
}

template <typename storage_t, typename compute_t>
__global__ void grid_sampler_3d_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ grid,
        storage_t* __restrict__ output,
        const int64_t C, const int64_t inp_D, const int64_t inp_H, const int64_t inp_W,
        const int64_t out_D, const int64_t out_H, const int64_t out_W,
        const int interpolation_mode, const int padding_mode,
        const bool align_corners) {
    const int64_t out_vol = out_D * out_H * out_W;
    const int64_t inp_vol = inp_D * inp_H * inp_W;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t w = index % out_W;
        const int64_t h = (index / out_W) % out_H;
        const int64_t d = (index / (out_H * out_W)) % out_D;
        const int64_t n = index / out_vol;
        const storage_t* grid_ptr_NDHW = grid + index * 3;

        compute_t ix = compute_source_index<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[0]), inp_W, padding_mode, align_corners);
        compute_t iy = compute_source_index<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[1]), inp_H, padding_mode, align_corners);
        compute_t iz = compute_source_index<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[2]), inp_D, padding_mode, align_corners);

        const storage_t* inp_ptr_N = input + n * C * inp_vol;
        storage_t* out_ptr_NDHW = output + n * C * out_vol + d * out_H * out_W + h * out_W + w;

        if (interpolation_mode == Interp::Bilinear) {
            int64_t ix0 = static_cast<int64_t>(::floor(ix));
            int64_t iy0 = static_cast<int64_t>(::floor(iy));
            int64_t iz0 = static_cast<int64_t>(::floor(iz));
            const int64_t cix[8] = {ix0, ix0 + 1, ix0, ix0 + 1, ix0, ix0 + 1, ix0, ix0 + 1};
            const int64_t ciy[8] = {iy0, iy0, iy0 + 1, iy0 + 1, iy0, iy0, iy0 + 1, iy0 + 1};
            const int64_t ciz[8] = {iz0, iz0, iz0, iz0, iz0 + 1, iz0 + 1, iz0 + 1, iz0 + 1};
            compute_t wgt[8];
            wgt[0] = (ix0 + 1 - ix) * (iy0 + 1 - iy) * (iz0 + 1 - iz);
            wgt[1] = (ix - ix0)     * (iy0 + 1 - iy) * (iz0 + 1 - iz);
            wgt[2] = (ix0 + 1 - ix) * (iy - iy0)     * (iz0 + 1 - iz);
            wgt[3] = (ix - ix0)     * (iy - iy0)     * (iz0 + 1 - iz);
            wgt[4] = (ix0 + 1 - ix) * (iy0 + 1 - iy) * (iz - iz0);
            wgt[5] = (ix - ix0)     * (iy0 + 1 - iy) * (iz - iz0);
            wgt[6] = (ix0 + 1 - ix) * (iy - iy0)     * (iz - iz0);
            wgt[7] = (ix - ix0)     * (iy - iy0)     * (iz - iz0);
            for (int64_t c = 0; c < C; ++c) {
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_vol;
                compute_t acc = 0;
                #pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                    if (within_bounds_3d(ciz[k], ciy[k], cix[k], inp_D, inp_H, inp_W))
                        acc += static_cast<compute_t>(
                            inp_ptr_NC[(ciz[k] * inp_H + ciy[k]) * inp_W + cix[k]]) * wgt[k];
                }
                out_ptr_NDHW[c * out_vol] = static_cast<storage_t>(acc);
            }
        } else {  // Nearest
            int64_t ix_near = static_cast<int64_t>(::nearbyint(ix));
            int64_t iy_near = static_cast<int64_t>(::nearbyint(iy));
            int64_t iz_near = static_cast<int64_t>(::nearbyint(iz));
            for (int64_t c = 0; c < C; ++c) {
                storage_t v = static_cast<storage_t>(0);
                if (within_bounds_3d(iz_near, iy_near, ix_near, inp_D, inp_H, inp_W))
                    v = inp_ptr_N[c * inp_vol + (iz_near * inp_H + iy_near) * inp_W + ix_near];
                out_ptr_NDHW[c * out_vol] = v;
            }
        }
    }
}

template <typename storage_t, typename compute_t>
__global__ void grid_sampler_3d_backward_kernel(
        const int64_t nthreads,
        const storage_t* __restrict__ grad_output,
        const storage_t* __restrict__ input,
        const storage_t* __restrict__ grid,
        storage_t* __restrict__ grad_input,   // zeros-initialized or null
        storage_t* __restrict__ grad_grid,    // or null
        const int64_t C, const int64_t inp_D, const int64_t inp_H, const int64_t inp_W,
        const int64_t out_D, const int64_t out_H, const int64_t out_W,
        const int interpolation_mode, const int padding_mode,
        const bool align_corners, const bool need_gi, const bool need_gg) {
    const int64_t out_vol = out_D * out_H * out_W;
    const int64_t inp_vol = inp_D * inp_H * inp_W;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        const int64_t w = index % out_W;
        const int64_t h = (index / out_W) % out_H;
        const int64_t d = (index / (out_H * out_W)) % out_D;
        const int64_t n = index / out_vol;
        const storage_t* grid_ptr_NDHW = grid + index * 3;

        compute_t gix_mult, giy_mult, giz_mult;
        compute_t ix = compute_source_index_set_grad<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[0]), inp_W, padding_mode, align_corners, &gix_mult);
        compute_t iy = compute_source_index_set_grad<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[1]), inp_H, padding_mode, align_corners, &giy_mult);
        compute_t iz = compute_source_index_set_grad<compute_t>(
            static_cast<compute_t>(grid_ptr_NDHW[2]), inp_D, padding_mode, align_corners, &giz_mult);

        const storage_t* inp_ptr_N = input + n * C * inp_vol;

        if (interpolation_mode == Interp::Bilinear) {
            int64_t ix0 = static_cast<int64_t>(::floor(ix));
            int64_t iy0 = static_cast<int64_t>(::floor(iy));
            int64_t iz0 = static_cast<int64_t>(::floor(iz));
            const int64_t cix[8] = {ix0, ix0 + 1, ix0, ix0 + 1, ix0, ix0 + 1, ix0, ix0 + 1};
            const int64_t ciy[8] = {iy0, iy0, iy0 + 1, iy0 + 1, iy0, iy0, iy0 + 1, iy0 + 1};
            const int64_t ciz[8] = {iz0, iz0, iz0, iz0, iz0 + 1, iz0 + 1, iz0 + 1, iz0 + 1};
            compute_t wgt[8];
            wgt[0] = (ix0 + 1 - ix) * (iy0 + 1 - iy) * (iz0 + 1 - iz);
            wgt[1] = (ix - ix0)     * (iy0 + 1 - iy) * (iz0 + 1 - iz);
            wgt[2] = (ix0 + 1 - ix) * (iy - iy0)     * (iz0 + 1 - iz);
            wgt[3] = (ix - ix0)     * (iy - iy0)     * (iz0 + 1 - iz);
            wgt[4] = (ix0 + 1 - ix) * (iy0 + 1 - iy) * (iz - iz0);
            wgt[5] = (ix - ix0)     * (iy0 + 1 - iy) * (iz - iz0);
            wgt[6] = (ix0 + 1 - ix) * (iy - iy0)     * (iz - iz0);
            wgt[7] = (ix - ix0)     * (iy - iy0)     * (iz - iz0);
            // grid_sampler_3d_backward_kernel sign pattern).
            const compute_t dwx[8] = {-(iy0 + 1 - iy) * (iz0 + 1 - iz),
                                       (iy0 + 1 - iy) * (iz0 + 1 - iz),
                                       -(iy - iy0)     * (iz0 + 1 - iz),
                                       (iy - iy0)     * (iz0 + 1 - iz),
                                       -(iy0 + 1 - iy) * (iz - iz0),
                                       (iy0 + 1 - iy) * (iz - iz0),
                                       -(iy - iy0)     * (iz - iz0),
                                       (iy - iy0)     * (iz - iz0)};
            const compute_t dwy[8] = {-(ix0 + 1 - ix) * (iz0 + 1 - iz),
                                       -(ix - ix0)     * (iz0 + 1 - iz),
                                       (ix0 + 1 - ix) * (iz0 + 1 - iz),
                                       (ix - ix0)     * (iz0 + 1 - iz),
                                       -(ix0 + 1 - ix) * (iz - iz0),
                                       -(ix - ix0)     * (iz - iz0),
                                       (ix0 + 1 - ix) * (iz - iz0),
                                       (ix - ix0)     * (iz - iz0)};
            const compute_t dwz[8] = {-(ix0 + 1 - ix) * (iy0 + 1 - iy),
                                       -(ix - ix0)     * (iy0 + 1 - iy),
                                       -(ix0 + 1 - ix) * (iy - iy0),
                                       -(ix - ix0)     * (iy - iy0),
                                       (ix0 + 1 - ix) * (iy0 + 1 - iy),
                                       (ix - ix0)     * (iy0 + 1 - iy),
                                       (ix0 + 1 - ix) * (iy - iy0),
                                       (ix - ix0)     * (iy - iy0)};

            compute_t gix = 0, giy = 0, giz = 0;
            for (int64_t c = 0; c < C; ++c) {
                const compute_t gOut = static_cast<compute_t>(
                    grad_output[(n * C + c) * out_vol + d * out_H * out_W + h * out_W + w]);
                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_vol;
                storage_t* gInp_ptr_NC = need_gi ? grad_input + (n * C + c) * inp_vol : nullptr;
                #pragma unroll 8
                for (int k = 0; k < 8; ++k) {
                    if (!within_bounds_3d(ciz[k], ciy[k], cix[k], inp_D, inp_H, inp_W))
                        continue;
                    const int64_t off = (ciz[k] * inp_H + ciy[k]) * inp_W + cix[k];
                    if (need_gi)
                        gs_safe_add(gInp_ptr_NC, off, wgt[k] * gOut);
                    const compute_t v = static_cast<compute_t>(inp_ptr_NC[off]);
                    gix += v * dwx[k] * gOut;
                    giy += v * dwy[k] * gOut;
                    giz += v * dwz[k] * gOut;
                }
            }
            if (need_gg) {
                storage_t* gGrid_ptr_NDHW = grad_grid + index * 3;
                gGrid_ptr_NDHW[0] = static_cast<storage_t>(gix_mult * gix);
                gGrid_ptr_NDHW[1] = static_cast<storage_t>(giy_mult * giy);
                gGrid_ptr_NDHW[2] = static_cast<storage_t>(giz_mult * giz);
            }
        } else {  // Nearest
            if (need_gi) {
                int64_t ix_near = static_cast<int64_t>(::nearbyint(ix));
                int64_t iy_near = static_cast<int64_t>(::nearbyint(iy));
                int64_t iz_near = static_cast<int64_t>(::nearbyint(iz));
                if (within_bounds_3d(iz_near, iy_near, ix_near, inp_D, inp_H, inp_W)) {
                    const int64_t off = (iz_near * inp_H + iy_near) * inp_W + ix_near;
                    for (int64_t c = 0; c < C; ++c) {
                        const compute_t gOut = static_cast<compute_t>(
                            grad_output[(n * C + c) * out_vol + d * out_H * out_W + h * out_W + w]);
                        gs_safe_add(grad_input + (n * C + c) * inp_vol, off, gOut);
                    }
                }
            }
            if (need_gg) {
                storage_t* gGrid_ptr_NDHW = grad_grid + index * 3;
                gGrid_ptr_NDHW[0] = static_cast<storage_t>(0);
                gGrid_ptr_NDHW[1] = static_cast<storage_t>(0);
                gGrid_ptr_NDHW[2] = static_cast<storage_t>(0);
            }
        }
    }
}

}  // namespace

template <typename storage_t, typename compute_t>
static Tensor grid_sampler_2d_cuda_impl(const Tensor& input, const Tensor& grid,
                                        int interpolation_mode, int padding_mode,
                                        bool align_corners) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_H = input.size(2);
    const int64_t inp_W = input.size(3);
    const int64_t out_H = grid.size(1);
    const int64_t out_W = grid.size(2);
    Tensor output = Tensor::empty({N, C, out_H, out_W}, input.dtype(), input.device());
    const int64_t nthreads = N * out_H * out_W;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(gs_grid_blocks(nthreads, kThreads));
    grid_sampler_2d_kernel<storage_t, compute_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, input.data_ptr<storage_t>(), grid.data_ptr<storage_t>(),
        output.data_ptr<storage_t>(), C, inp_H, inp_W, out_H, out_W,
        interpolation_mode, padding_mode, align_corners);
    return output;
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> grid_sampler_2d_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& input, const Tensor& grid,
        int interpolation_mode, int padding_mode, bool align_corners,
        const std::vector<bool>& output_mask) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_H = input.size(2);
    const int64_t inp_W = input.size(3);
    const int64_t out_H = grid.size(1);
    const int64_t out_W = grid.size(2);
    const bool need_gi = output_mask.size() > 0 && output_mask[0];
    const bool need_gg = output_mask.size() > 1 && output_mask[1];

    Tensor grad_input = need_gi
        ? Tensor::zeros({N, C, inp_H, inp_W}, input.dtype(), input.device())
        : Tensor::empty({N, C, inp_H, inp_W}, input.dtype(), input.device());
    Tensor grad_grid = need_gg
        ? Tensor::empty({N, out_H, out_W, 2}, grid.dtype(), grid.device())
        : Tensor::zeros({N, out_H, out_W, 2}, grid.dtype(), grid.device());
    const int64_t nthreads = N * out_H * out_W;
    if (nthreads == 0) return {grad_input, grad_grid};
    dim3 block(kThreads);
    dim3 grid_dim(gs_grid_blocks(nthreads, kThreads));
    grid_sampler_2d_backward_kernel<storage_t, compute_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, grad_output.data_ptr<storage_t>(), input.data_ptr<storage_t>(),
        grid.data_ptr<storage_t>(),
        need_gi ? grad_input.data_ptr<storage_t>() : nullptr,
        need_gg ? grad_grid.data_ptr<storage_t>() : nullptr,
        C, inp_H, inp_W, out_H, out_W,
        interpolation_mode, padding_mode, align_corners, need_gi, need_gg);
    return {grad_input, grad_grid};
}

template <typename storage_t, typename compute_t>
static Tensor grid_sampler_3d_cuda_impl(const Tensor& input, const Tensor& grid,
                                        int interpolation_mode, int padding_mode,
                                        bool align_corners) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_D = input.size(2);
    const int64_t inp_H = input.size(3);
    const int64_t inp_W = input.size(4);
    const int64_t out_D = grid.size(1);
    const int64_t out_H = grid.size(2);
    const int64_t out_W = grid.size(3);
    Tensor output = Tensor::empty({N, C, out_D, out_H, out_W}, input.dtype(), input.device());
    const int64_t nthreads = N * out_D * out_H * out_W;
    if (nthreads == 0) return output;
    dim3 block(kThreads);
    dim3 grid_dim(gs_grid_blocks(nthreads, kThreads));
    grid_sampler_3d_kernel<storage_t, compute_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, input.data_ptr<storage_t>(), grid.data_ptr<storage_t>(),
        output.data_ptr<storage_t>(), C, inp_D, inp_H, inp_W, out_D, out_H, out_W,
        interpolation_mode, padding_mode, align_corners);
    return output;
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> grid_sampler_3d_backward_cuda_impl(
        const Tensor& grad_output, const Tensor& input, const Tensor& grid,
        int interpolation_mode, int padding_mode, bool align_corners,
        const std::vector<bool>& output_mask) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_D = input.size(2);
    const int64_t inp_H = input.size(3);
    const int64_t inp_W = input.size(4);
    const int64_t out_D = grid.size(1);
    const int64_t out_H = grid.size(2);
    const int64_t out_W = grid.size(3);
    const bool need_gi = output_mask.size() > 0 && output_mask[0];
    const bool need_gg = output_mask.size() > 1 && output_mask[1];

    Tensor grad_input = need_gi
        ? Tensor::zeros({N, C, inp_D, inp_H, inp_W}, input.dtype(), input.device())
        : Tensor::empty({N, C, inp_D, inp_H, inp_W}, input.dtype(), input.device());
    Tensor grad_grid = need_gg
        ? Tensor::empty({N, out_D, out_H, out_W, 3}, grid.dtype(), grid.device())
        : Tensor::zeros({N, out_D, out_H, out_W, 3}, grid.dtype(), grid.device());
    const int64_t nthreads = N * out_D * out_H * out_W;
    if (nthreads == 0) return {grad_input, grad_grid};
    dim3 block(kThreads);
    dim3 grid_dim(gs_grid_blocks(nthreads, kThreads));
    grid_sampler_3d_backward_kernel<storage_t, compute_t><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
        nthreads, grad_output.data_ptr<storage_t>(), input.data_ptr<storage_t>(),
        grid.data_ptr<storage_t>(),
        need_gi ? grad_input.data_ptr<storage_t>() : nullptr,
        need_gg ? grad_grid.data_ptr<storage_t>() : nullptr,
        C, inp_D, inp_H, inp_W, out_D, out_H, out_W,
        interpolation_mode, padding_mode, align_corners, need_gi, need_gg);
    return {grad_input, grad_grid};
}

Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
    if (input.dim() != 4 || grid.dim() != 4)
        TP_THROW(RuntimeError, "grid_sampler_2d: expected 4D input and grid");
    if (input.size(0) != grid.size(0) || grid.size(3) != 2)
        TP_THROW(RuntimeError, "grid_sampler_2d: grid must be (N, H_out, W_out, 2) with matching N");
    const Tensor ic = input.contiguous();
    const Tensor gc = grid.contiguous();
    const int im = static_cast<int>(interpolation_mode);
    const int pm = static_cast<int>(padding_mode);
    switch (input.dtype()) {
        case DType::Float32: return grid_sampler_2d_cuda_impl<float, float>(ic, gc, im, pm, align_corners);
        case DType::Float64: return grid_sampler_2d_cuda_impl<double, double>(ic, gc, im, pm, align_corners);
        case DType::Float16: return grid_sampler_2d_cuda_impl<Half, float>(ic, gc, im, pm, align_corners);
        case DType::BFloat16: return grid_sampler_2d_cuda_impl<BFloat16, float>(ic, gc, im, pm, align_corners);
        default: TP_THROW(TypeError, "grid_sampler_2d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& grid,
        int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
        const std::vector<bool>& output_mask) {
    if (input.dim() != 4 || grid.dim() != 4)
        TP_THROW(RuntimeError, "grid_sampler_2d_backward: expected 4D input and grid");
    const Tensor goc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor gc = grid.contiguous();
    const int im = static_cast<int>(interpolation_mode);
    const int pm = static_cast<int>(padding_mode);
    switch (input.dtype()) {
        case DType::Float32: return grid_sampler_2d_backward_cuda_impl<float, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float64: return grid_sampler_2d_backward_cuda_impl<double, double>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float16: return grid_sampler_2d_backward_cuda_impl<Half, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::BFloat16: return grid_sampler_2d_backward_cuda_impl<BFloat16, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        default: TP_THROW(TypeError, "grid_sampler_2d_backward: unsupported dtype");
    }
}

Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
    if (input.dim() != 5 || grid.dim() != 5)
        TP_THROW(RuntimeError, "grid_sampler_3d: expected 5D input and grid");
    if (input.size(0) != grid.size(0) || grid.size(4) != 3)
        TP_THROW(RuntimeError, "grid_sampler_3d: grid must be (N, D_out, H_out, W_out, 3) with matching N");
    if (interpolation_mode == Interp::Bicubic)
        TP_THROW(RuntimeError, "grid_sampler_3d: bicubic only supports 4D input");
    const Tensor ic = input.contiguous();
    const Tensor gc = grid.contiguous();
    const int im = static_cast<int>(interpolation_mode);
    const int pm = static_cast<int>(padding_mode);
    switch (input.dtype()) {
        case DType::Float32: return grid_sampler_3d_cuda_impl<float, float>(ic, gc, im, pm, align_corners);
        case DType::Float64: return grid_sampler_3d_cuda_impl<double, double>(ic, gc, im, pm, align_corners);
        case DType::Float16: return grid_sampler_3d_cuda_impl<Half, float>(ic, gc, im, pm, align_corners);
        case DType::BFloat16: return grid_sampler_3d_cuda_impl<BFloat16, float>(ic, gc, im, pm, align_corners);
        default: TP_THROW(TypeError, "grid_sampler_3d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& grid,
        int64_t interpolation_mode, int64_t padding_mode, bool align_corners,
        const std::vector<bool>& output_mask) {
    if (input.dim() != 5 || grid.dim() != 5)
        TP_THROW(RuntimeError, "grid_sampler_3d_backward: expected 5D input and grid");
    if (interpolation_mode == Interp::Bicubic)
        TP_THROW(RuntimeError, "grid_sampler_3d_backward: bicubic only supports 4D input");
    const Tensor goc = grad_output.contiguous();
    const Tensor ic = input.contiguous();
    const Tensor gc = grid.contiguous();
    const int im = static_cast<int>(interpolation_mode);
    const int pm = static_cast<int>(padding_mode);
    switch (input.dtype()) {
        case DType::Float32: return grid_sampler_3d_backward_cuda_impl<float, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float64: return grid_sampler_3d_backward_cuda_impl<double, double>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float16: return grid_sampler_3d_backward_cuda_impl<Half, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::BFloat16: return grid_sampler_3d_backward_cuda_impl<BFloat16, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        default: TP_THROW(TypeError, "grid_sampler_3d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, GridSamplerKernels) {
    m.impl("grid_sampler_2d", grid_sampler_2d_cuda);
    m.impl("grid_sampler_2d_backward", grid_sampler_2d_backward_cuda);
    m.impl("grid_sampler_3d", grid_sampler_3d_cuda);
    m.impl("grid_sampler_3d_backward", grid_sampler_3d_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
