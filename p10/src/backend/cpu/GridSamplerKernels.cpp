// grid_sampler_2d / grid_sampler_3d CPU kernels.
//
// Port of aten/src/ATen/native/cpu/GridSamplerKernel.cpp (which mirrors the
// per-output-element formulation of aten/src/ATen/native/cuda/GridSampler.cu):
//   interpolation_mode: 0=Bilinear 1=Nearest 2=Bicubic(2d only)
//   padding_mode:       0=Zeros 1=Border 2=Reflection
// Reduced-precision inputs (f16/bf16) compute in float, matching ATen's
// opmath_t usage.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include "../GridSamplerInline.h"
#include <vector>
#include <tuple>
#include <cmath>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::gridsampler;
using namespace tensorplay::parallel;

template <typename storage_t, typename compute_t>
static Tensor grid_sampler_2d_cpu_impl(const Tensor& input, const Tensor& grid,
                                       int interpolation_mode, int padding_mode,
                                       bool align_corners) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t inp_H = input.size(2);
    const int64_t inp_W = input.size(3);
    const int64_t out_H = grid.size(1);
    const int64_t out_W = grid.size(2);
    Tensor output = Tensor::empty({N, C, out_H, out_W}, input.dtype(), input.device());
    if (output.numel() == 0) return output;

    const storage_t* inp_ptr = input.data_ptr<storage_t>();
    const storage_t* grid_ptr = grid.data_ptr<storage_t>();
    storage_t* out_ptr = output.data_ptr<storage_t>();

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* grid_ptr_N = grid_ptr + n * out_H * out_W * 2;
            const storage_t* inp_ptr_N = inp_ptr + n * C * inp_H * inp_W;
            storage_t* out_ptr_N = out_ptr + n * C * out_H * out_W;
            for (int64_t h = 0; h < out_H; ++h) {
                for (int64_t w = 0; w < out_W; ++w) {
                    const storage_t* grid_ptr_NHW = grid_ptr_N + (h * out_W + w) * 2;
                    compute_t ix = compute_source_index<compute_t>(
                        static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, padding_mode, align_corners);
                    compute_t iy = compute_source_index<compute_t>(
                        static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, padding_mode, align_corners);

                    if (interpolation_mode == Interp::Bilinear) {
                        int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
                        int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
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
                            out_ptr_N[(c * out_H + h) * out_W + w] = static_cast<storage_t>(acc);
                        }
                    } else if (interpolation_mode == Interp::Nearest) {
                        int64_t ix_near = static_cast<int64_t>(std::nearbyint(ix));
                        int64_t iy_near = static_cast<int64_t>(std::nearbyint(iy));
                        for (int64_t c = 0; c < C; ++c) {
                            storage_t v = static_cast<storage_t>(0);
                            if (within_bounds_2d(iy_near, ix_near, inp_H, inp_W))
                                v = inp_ptr_N[c * inp_H * inp_W + iy_near * inp_W + ix_near];
                            out_ptr_N[(c * out_H + h) * out_W + w] = v;
                        }
                    } else {  // Bicubic
                        // ATen bicubic skips coordinate-level padding: each
                        // cubic tap is padding-adjusted individually.
                        compute_t x = unnormalize<compute_t>(
                            static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, align_corners);
                        compute_t y = unnormalize<compute_t>(
                            static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, align_corners);
                        compute_t ix_nw = std::floor(x);
                        compute_t iy_nw = std::floor(y);
                        compute_t tx = x - ix_nw;
                        compute_t ty = y - iy_nw;
                        for (int64_t c = 0; c < C; ++c) {
                            const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                            compute_t coeffs[4];
                            for (int64_t i = 0; i < 4; ++i) {
                                coeffs[i] = cubic_interp1d<compute_t>(
                                    get_value_bounded<compute_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                                    get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                                    get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                                    get_value_bounded<compute_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, 1, inp_W, padding_mode, align_corners),
                                    tx);
                            }
                            out_ptr_N[(c * out_H + h) * out_W + w] =
                                static_cast<storage_t>(cubic_interp1d<compute_t>(coeffs[0], coeffs[1], coeffs[2], coeffs[3], ty));
                        }
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> grid_sampler_2d_backward_cpu_impl(
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
    if (grad_output.numel() == 0) return {grad_input, grad_grid};

    const storage_t* gOut_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* inp_ptr = input.data_ptr<storage_t>();
    const storage_t* grid_ptr = grid.data_ptr<storage_t>();
    storage_t* gInp_ptr = need_gi ? grad_input.data_ptr<storage_t>() : nullptr;
    storage_t* gGrid_ptr = need_gg ? grad_grid.data_ptr<storage_t>() : nullptr;

    // Parallel over the batch: grad_input[n] is only written by threads
    // handling batch element n, so the scatter adds are race-free.
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* grid_ptr_N = grid_ptr + n * out_H * out_W * 2;
            const storage_t* inp_ptr_N = inp_ptr + n * C * inp_H * inp_W;
            storage_t* gInp_ptr_N = need_gi ? gInp_ptr + n * C * inp_H * inp_W : nullptr;
            storage_t* gGrid_ptr_N = need_gg ? gGrid_ptr + n * out_H * out_W * 2 : nullptr;
            for (int64_t h = 0; h < out_H; ++h) {
                for (int64_t w = 0; w < out_W; ++w) {
                    const storage_t* grid_ptr_NHW = grid_ptr_N + (h * out_W + w) * 2;
                    compute_t gix_mult, giy_mult;
                    compute_t ix = compute_source_index_set_grad<compute_t>(
                        static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, padding_mode, align_corners, &gix_mult);
                    compute_t iy = compute_source_index_set_grad<compute_t>(
                        static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, padding_mode, align_corners, &giy_mult);

                    if (interpolation_mode == Interp::Bilinear) {
                        int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
                        int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
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
                                gOut_ptr[(n * C + c) * out_H * out_W + h * out_W + w]);
                            const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                            storage_t* gInp_ptr_NC = need_gi ? gInp_ptr_N + c * inp_H * inp_W : nullptr;

                            if (need_gi) {
                                if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W))
                                    gInp_ptr_NC[iy_nw * inp_W + ix_nw] += static_cast<storage_t>(nw * gOut);
                                if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W))
                                    gInp_ptr_NC[iy_ne * inp_W + ix_ne] += static_cast<storage_t>(ne * gOut);
                                if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W))
                                    gInp_ptr_NC[iy_sw * inp_W + ix_sw] += static_cast<storage_t>(sw * gOut);
                                if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W))
                                    gInp_ptr_NC[iy_se * inp_W + ix_se] += static_cast<storage_t>(se * gOut);
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
                            storage_t* gGrid_ptr_NHW = gGrid_ptr_N + (h * out_W + w) * 2;
                            gGrid_ptr_NHW[0] = static_cast<storage_t>(gix_mult * gix);
                            gGrid_ptr_NHW[1] = static_cast<storage_t>(giy_mult * giy);
                        }
                    } else if (interpolation_mode == Interp::Nearest) {
                        if (need_gi) {
                            int64_t ix_near = static_cast<int64_t>(std::nearbyint(ix));
                            int64_t iy_near = static_cast<int64_t>(std::nearbyint(iy));
                            if (within_bounds_2d(iy_near, ix_near, inp_H, inp_W)) {
                                for (int64_t c = 0; c < C; ++c) {
                                    const compute_t gOut = static_cast<compute_t>(
                                        gOut_ptr[(n * C + c) * out_H * out_W + h * out_W + w]);
                                    gInp_ptr_N[c * inp_H * inp_W + iy_near * inp_W + ix_near] +=
                                        static_cast<storage_t>(gOut);
                                }
                            }
                        }
                        if (need_gg) {
                            storage_t* gGrid_ptr_NHW = gGrid_ptr_N + (h * out_W + w) * 2;
                            gGrid_ptr_NHW[0] = static_cast<storage_t>(0);
                            gGrid_ptr_NHW[1] = static_cast<storage_t>(0);
                        }
                    } else {  // Bicubic
                        compute_t x = unnormalize_set_grad<compute_t>(
                            static_cast<compute_t>(grid_ptr_NHW[0]), inp_W, align_corners, &gix_mult);
                        compute_t y = unnormalize_set_grad<compute_t>(
                            static_cast<compute_t>(grid_ptr_NHW[1]), inp_H, align_corners, &giy_mult);
                        compute_t ix_nw = std::floor(x);
                        compute_t iy_nw = std::floor(y);
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
                                gOut_ptr[(n * C + c) * out_H * out_W + h * out_W + w]);
                            const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_H * inp_W;
                            storage_t* gInp_ptr_NC = need_gi ? gInp_ptr_N + c * inp_H * inp_W : nullptr;
                            for (int64_t i = 0; i < 4; ++i) {
                                for (int64_t j = 0; j < 4; ++j) {
                                    const compute_t tap_x = ix_nw - 1 + i;
                                    const compute_t tap_y = iy_nw - 1 + j;
                                    if (need_gi) {
                                        compute_t ax = compute_coordinates(tap_x, inp_W, padding_mode, align_corners);
                                        compute_t ay = compute_coordinates(tap_y, inp_H, padding_mode, align_corners);
                                        int64_t iax = static_cast<int64_t>(ax);
                                        int64_t iay = static_cast<int64_t>(ay);
                                        if (within_bounds_2d(iay, iax, inp_H, inp_W))
                                            gInp_ptr_NC[iay * inp_W + iax] +=
                                                static_cast<storage_t>(gOut * x_coeffs[i] * y_coeffs[j]);
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
                            storage_t* gGrid_ptr_NHW = gGrid_ptr_N + (h * out_W + w) * 2;
                            gGrid_ptr_NHW[0] = static_cast<storage_t>(gix_mult * gix);
                            gGrid_ptr_NHW[1] = static_cast<storage_t>(giy_mult * giy);
                        }
                    }
                }
            }
        }
    });
    return {grad_input, grad_grid};
}

template <typename storage_t, typename compute_t>
static Tensor grid_sampler_3d_cpu_impl(const Tensor& input, const Tensor& grid,
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
    if (output.numel() == 0) return output;

    const storage_t* inp_ptr = input.data_ptr<storage_t>();
    const storage_t* grid_ptr = grid.data_ptr<storage_t>();
    storage_t* out_ptr = output.data_ptr<storage_t>();
    const int64_t inp_vol = inp_D * inp_H * inp_W;
    const int64_t out_vol = out_D * out_H * out_W;

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* grid_ptr_N = grid_ptr + n * out_vol * 3;
            const storage_t* inp_ptr_N = inp_ptr + n * C * inp_vol;
            storage_t* out_ptr_N = out_ptr + n * C * out_vol;
            for (int64_t d = 0; d < out_D; ++d) {
                for (int64_t h = 0; h < out_H; ++h) {
                    for (int64_t w = 0; w < out_W; ++w) {
                        const storage_t* grid_ptr_NDHW = grid_ptr_N + ((d * out_H + h) * out_W + w) * 3;
                        compute_t ix = compute_source_index<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[0]), inp_W, padding_mode, align_corners);
                        compute_t iy = compute_source_index<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[1]), inp_H, padding_mode, align_corners);
                        compute_t iz = compute_source_index<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[2]), inp_D, padding_mode, align_corners);

                        if (interpolation_mode == Interp::Bilinear) {
                            int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
                            int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
                            int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));
                            const int64_t cix[8] = {ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1,
                                                    ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1};
                            const int64_t ciy[8] = {iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1,
                                                    iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1};
                            const int64_t ciz[8] = {iz_tnw, iz_tnw, iz_tnw, iz_tnw,
                                                    iz_tnw + 1, iz_tnw + 1, iz_tnw + 1, iz_tnw + 1};
                            compute_t wgt[8];
                            wgt[0] = (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy) * (iz_tnw + 1 - iz);  // tnw
                            wgt[1] = (ix - ix_tnw)     * (iy_tnw + 1 - iy) * (iz_tnw + 1 - iz);  // tne
                            wgt[2] = (ix_tnw + 1 - ix) * (iy - iy_tnw)     * (iz_tnw + 1 - iz);  // tsw
                            wgt[3] = (ix - ix_tnw)     * (iy - iy_tnw)     * (iz_tnw + 1 - iz);  // tse
                            wgt[4] = (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy) * (iz - iz_tnw);      // bnw
                            wgt[5] = (ix - ix_tnw)     * (iy_tnw + 1 - iy) * (iz - iz_tnw);      // bne
                            wgt[6] = (ix_tnw + 1 - ix) * (iy - iy_tnw)     * (iz - iz_tnw);      // bsw
                            wgt[7] = (ix - ix_tnw)     * (iy - iy_tnw)     * (iz - iz_tnw);      // bse
                            for (int64_t c = 0; c < C; ++c) {
                                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_vol;
                                compute_t acc = 0;
                                for (int k = 0; k < 8; ++k) {
                                    if (within_bounds_3d(ciz[k], ciy[k], cix[k], inp_D, inp_H, inp_W))
                                        acc += static_cast<compute_t>(
                                            inp_ptr_NC[(ciz[k] * inp_H + ciy[k]) * inp_W + cix[k]]) * wgt[k];
                                }
                                out_ptr_N[(c * out_D + d) * out_H * out_W + h * out_W + w] =
                                    static_cast<storage_t>(acc);
                            }
                        } else {  // Nearest
                            int64_t ix_near = static_cast<int64_t>(std::nearbyint(ix));
                            int64_t iy_near = static_cast<int64_t>(std::nearbyint(iy));
                            int64_t iz_near = static_cast<int64_t>(std::nearbyint(iz));
                            for (int64_t c = 0; c < C; ++c) {
                                storage_t v = static_cast<storage_t>(0);
                                if (within_bounds_3d(iz_near, iy_near, ix_near, inp_D, inp_H, inp_W))
                                    v = inp_ptr_N[c * inp_vol + (iz_near * inp_H + iy_near) * inp_W + ix_near];
                                out_ptr_N[(c * out_D + d) * out_H * out_W + h * out_W + w] = v;
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

template <typename storage_t, typename compute_t>
static std::tuple<Tensor, Tensor> grid_sampler_3d_backward_cpu_impl(
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
    if (grad_output.numel() == 0) return {grad_input, grad_grid};

    const storage_t* gOut_ptr = grad_output.data_ptr<storage_t>();
    const storage_t* inp_ptr = input.data_ptr<storage_t>();
    const storage_t* grid_ptr = grid.data_ptr<storage_t>();
    storage_t* gInp_ptr = need_gi ? grad_input.data_ptr<storage_t>() : nullptr;
    storage_t* gGrid_ptr = need_gg ? grad_grid.data_ptr<storage_t>() : nullptr;
    const int64_t inp_vol = inp_D * inp_H * inp_W;
    const int64_t out_vol = out_D * out_H * out_W;

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const storage_t* grid_ptr_N = grid_ptr + n * out_vol * 3;
            const storage_t* inp_ptr_N = inp_ptr + n * C * inp_vol;
            storage_t* gInp_ptr_N = need_gi ? gInp_ptr + n * C * inp_vol : nullptr;
            storage_t* gGrid_ptr_N = need_gg ? gGrid_ptr + n * out_vol * 3 : nullptr;
            for (int64_t d = 0; d < out_D; ++d) {
                for (int64_t h = 0; h < out_H; ++h) {
                    for (int64_t w = 0; w < out_W; ++w) {
                        const storage_t* grid_ptr_NDHW = grid_ptr_N + ((d * out_H + h) * out_W + w) * 3;
                        compute_t gix_mult, giy_mult, giz_mult;
                        compute_t ix = compute_source_index_set_grad<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[0]), inp_W, padding_mode, align_corners, &gix_mult);
                        compute_t iy = compute_source_index_set_grad<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[1]), inp_H, padding_mode, align_corners, &giy_mult);
                        compute_t iz = compute_source_index_set_grad<compute_t>(
                            static_cast<compute_t>(grid_ptr_NDHW[2]), inp_D, padding_mode, align_corners, &giz_mult);

                        if (interpolation_mode == Interp::Bilinear) {
                            int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
                            int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
                            int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));
                            const int64_t cix[8] = {ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1,
                                                    ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1};
                            const int64_t ciy[8] = {iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1,
                                                    iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1};
                            const int64_t ciz[8] = {iz_tnw, iz_tnw, iz_tnw, iz_tnw,
                                                    iz_tnw + 1, iz_tnw + 1, iz_tnw + 1, iz_tnw + 1};
                            compute_t wgt[8];
                            wgt[0] = (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy) * (iz_tnw + 1 - iz);
                            wgt[1] = (ix - ix_tnw)     * (iy_tnw + 1 - iy) * (iz_tnw + 1 - iz);
                            wgt[2] = (ix_tnw + 1 - ix) * (iy - iy_tnw)     * (iz_tnw + 1 - iz);
                            wgt[3] = (ix - ix_tnw)     * (iy - iy_tnw)     * (iz_tnw + 1 - iz);
                            wgt[4] = (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy) * (iz - iz_tnw);
                            wgt[5] = (ix - ix_tnw)     * (iy_tnw + 1 - iy) * (iz - iz_tnw);
                            wgt[6] = (ix_tnw + 1 - ix) * (iy - iy_tnw)     * (iz - iz_tnw);
                            wgt[7] = (ix - ix_tnw)     * (iy - iy_tnw)     * (iz - iz_tnw);
                            // d weight / d (ix, iy, iz) per corner; sign pattern
                            // follows ATen grid_sampler_3d_backward_kernel.
                            const compute_t dwx[8] = {-(iy_tnw + 1 - iy) * (iz_tnw + 1 - iz),
                                                       (iy_tnw + 1 - iy) * (iz_tnw + 1 - iz),
                                                       -(iy - iy_tnw)     * (iz_tnw + 1 - iz),
                                                       (iy - iy_tnw)     * (iz_tnw + 1 - iz),
                                                       -(iy_tnw + 1 - iy) * (iz - iz_tnw),
                                                       (iy_tnw + 1 - iy) * (iz - iz_tnw),
                                                       -(iy - iy_tnw)     * (iz - iz_tnw),
                                                       (iy - iy_tnw)     * (iz - iz_tnw)};
                            const compute_t dwy[8] = {-(ix_tnw + 1 - ix) * (iz_tnw + 1 - iz),
                                                       -(ix - ix_tnw)     * (iz_tnw + 1 - iz),
                                                       (ix_tnw + 1 - ix) * (iz_tnw + 1 - iz),
                                                       (ix - ix_tnw)     * (iz_tnw + 1 - iz),
                                                       -(ix_tnw + 1 - ix) * (iz - iz_tnw),
                                                       -(ix - ix_tnw)     * (iz - iz_tnw),
                                                       (ix_tnw + 1 - ix) * (iz - iz_tnw),
                                                       (ix - ix_tnw)     * (iz - iz_tnw)};
                            const compute_t dwz[8] = {-(ix_tnw + 1 - ix) * (iy_tnw + 1 - iy),
                                                       -(ix - ix_tnw)     * (iy_tnw + 1 - iy),
                                                       -(ix_tnw + 1 - ix) * (iy - iy_tnw),
                                                       -(ix - ix_tnw)     * (iy - iy_tnw),
                                                       (ix_tnw + 1 - ix) * (iy_tnw + 1 - iy),
                                                       (ix - ix_tnw)     * (iy_tnw + 1 - iy),
                                                       (ix_tnw + 1 - ix) * (iy - iy_tnw),
                                                       (ix - ix_tnw)     * (iy - iy_tnw)};

                            compute_t gix = 0, giy = 0, giz = 0;
                            for (int64_t c = 0; c < C; ++c) {
                                const compute_t gOut = static_cast<compute_t>(
                                    gOut_ptr[(n * C + c) * out_vol + (d * out_H + h) * out_W + w]);
                                const storage_t* inp_ptr_NC = inp_ptr_N + c * inp_vol;
                                storage_t* gInp_ptr_NC = need_gi ? gInp_ptr_N + c * inp_vol : nullptr;
                                for (int k = 0; k < 8; ++k) {
                                    if (!within_bounds_3d(ciz[k], ciy[k], cix[k], inp_D, inp_H, inp_W))
                                        continue;
                                    const int64_t off = (ciz[k] * inp_H + ciy[k]) * inp_W + cix[k];
                                    if (need_gi)
                                        gInp_ptr_NC[off] += static_cast<storage_t>(wgt[k] * gOut);
                                    const compute_t v = static_cast<compute_t>(inp_ptr_NC[off]);
                                    gix += v * dwx[k] * gOut;
                                    giy += v * dwy[k] * gOut;
                                    giz += v * dwz[k] * gOut;
                                }
                            }
                            if (need_gg) {
                                storage_t* gGrid_ptr_NDHW = gGrid_ptr_N + ((d * out_H + h) * out_W + w) * 3;
                                gGrid_ptr_NDHW[0] = static_cast<storage_t>(gix_mult * gix);
                                gGrid_ptr_NDHW[1] = static_cast<storage_t>(giy_mult * giy);
                                gGrid_ptr_NDHW[2] = static_cast<storage_t>(giz_mult * giz);
                            }
                        } else {  // Nearest
                            if (need_gi) {
                                int64_t ix_near = static_cast<int64_t>(std::nearbyint(ix));
                                int64_t iy_near = static_cast<int64_t>(std::nearbyint(iy));
                                int64_t iz_near = static_cast<int64_t>(std::nearbyint(iz));
                                if (within_bounds_3d(iz_near, iy_near, ix_near, inp_D, inp_H, inp_W)) {
                                    const int64_t off = (iz_near * inp_H + iy_near) * inp_W + ix_near;
                                    for (int64_t c = 0; c < C; ++c) {
                                        const compute_t gOut = static_cast<compute_t>(
                                            gOut_ptr[(n * C + c) * out_vol + (d * out_H + h) * out_W + w]);
                                        gInp_ptr_N[c * inp_vol + off] += static_cast<storage_t>(gOut);
                                    }
                                }
                            }
                            if (need_gg) {
                                storage_t* gGrid_ptr_NDHW = gGrid_ptr_N + ((d * out_H + h) * out_W + w) * 3;
                                gGrid_ptr_NDHW[0] = static_cast<storage_t>(0);
                                gGrid_ptr_NDHW[1] = static_cast<storage_t>(0);
                                gGrid_ptr_NDHW[2] = static_cast<storage_t>(0);
                            }
                        }
                    }
                }
            }
        }
    });
    return {grad_input, grad_grid};
}

Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
    if (input.dim() != 4 || grid.dim() != 4)
        TP_THROW(RuntimeError, "grid_sampler_2d: expected 4D input and grid");
    if (input.size(0) != grid.size(0) || grid.size(3) != 2)
        TP_THROW(RuntimeError, "grid_sampler_2d: grid must be (N, H_out, W_out, 2) with matching N");
    if (interpolation_mode == Interp::Bicubic && input.size(2) * input.size(3) == 0)
        TP_THROW(RuntimeError, "grid_sampler_2d: bicubic requires non-empty input spatial dims");
    const Tensor ic = input.contiguous();
    const Tensor gc = grid.contiguous();
    const int im = static_cast<int>(interpolation_mode);
    const int pm = static_cast<int>(padding_mode);
    switch (input.dtype()) {
        case DType::Float32: return grid_sampler_2d_cpu_impl<float, float>(ic, gc, im, pm, align_corners);
        case DType::Float64: return grid_sampler_2d_cpu_impl<double, double>(ic, gc, im, pm, align_corners);
        case DType::Float16: return grid_sampler_2d_cpu_impl<Half, float>(ic, gc, im, pm, align_corners);
        case DType::BFloat16: return grid_sampler_2d_cpu_impl<BFloat16, float>(ic, gc, im, pm, align_corners);
        default: TP_THROW(TypeError, "grid_sampler_2d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_cpu(
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
        case DType::Float32: return grid_sampler_2d_backward_cpu_impl<float, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float64: return grid_sampler_2d_backward_cpu_impl<double, double>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float16: return grid_sampler_2d_backward_cpu_impl<Half, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::BFloat16: return grid_sampler_2d_backward_cpu_impl<BFloat16, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        default: TP_THROW(TypeError, "grid_sampler_2d_backward: unsupported dtype");
    }
}

Tensor grid_sampler_3d_cpu(const Tensor& input, const Tensor& grid,
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
        case DType::Float32: return grid_sampler_3d_cpu_impl<float, float>(ic, gc, im, pm, align_corners);
        case DType::Float64: return grid_sampler_3d_cpu_impl<double, double>(ic, gc, im, pm, align_corners);
        case DType::Float16: return grid_sampler_3d_cpu_impl<Half, float>(ic, gc, im, pm, align_corners);
        case DType::BFloat16: return grid_sampler_3d_cpu_impl<BFloat16, float>(ic, gc, im, pm, align_corners);
        default: TP_THROW(TypeError, "grid_sampler_3d: unsupported dtype");
    }
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_cpu(
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
        case DType::Float32: return grid_sampler_3d_backward_cpu_impl<float, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float64: return grid_sampler_3d_backward_cpu_impl<double, double>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::Float16: return grid_sampler_3d_backward_cpu_impl<Half, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        case DType::BFloat16: return grid_sampler_3d_backward_cpu_impl<BFloat16, float>(goc, ic, gc, im, pm, align_corners, output_mask);
        default: TP_THROW(TypeError, "grid_sampler_3d_backward: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, GridSamplerKernels) {
    m.impl("grid_sampler_2d", grid_sampler_2d_cpu);
    m.impl("grid_sampler_2d_backward", grid_sampler_2d_backward_cpu);
    m.impl("grid_sampler_3d", grid_sampler_3d_cpu);
    m.impl("grid_sampler_3d_backward", grid_sampler_3d_backward_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
