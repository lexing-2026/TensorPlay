// Tier-compiled channels-last kernels for grid_sampler_2d (bilinear and
// nearest; bicubic keeps the scalar frame). In channels-last the channel axis
// is the contiguous one, so it is also the vector axis: for every output
// pixel the tap pixels are each loaded for a block of channels with one
// vector load, scaled by the tap weight and accumulated. The per-pixel
// coordinate/weight section is the scalar frame's own arithmetic (same order
// and rounding), so the two frames are bit-identical by construction.

#include "cpu/GridSamplerKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"
#include "../GridSamplerInline.h"
#include "cpu/vec/vec.h"

namespace tensorplay {
namespace cpu {
namespace {

using tensorplay::vec::Vectorized;
using tensorplay::parallel::parallel_for;
using gridsampler::Interp;
using gridsampler::Pad;

template <typename T>
using Vec = Vectorized<T>;

// ---------------------------------------------------------------------------
// Bilinear
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
__attribute__((optimize("-ffp-contract=off")))
void grid_sample2d_bilinear_cl(const T* inp, const T* grid, T* out,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               int64_t oH, int64_t oW, bool align_corners) {
    // channels-last strides: sN = C*H*W, sC = 1
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N * oH, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t n = task / oH;
            const int64_t h = task - n * oH;
            const T* inp_n = inp + n * C * H * W;
            const T* grid_row = grid + (n * oH + h) * oW * 2;
            T* out_n = out + n * C * out_plane;
            const int64_t row_out = h * oW;
            for (int64_t w = 0; w < oW; ++w) {
                // Coordinate/weight section verbatim from the scalar frame:
                // the pixel centers sit on integers, so floor(x) + 1 - x is
                // exact and the weights come out identical lane for lane.
                const T x = gridsampler::compute_source_index<T>(
                    grid_row[2 * w], W, PADDING, align_corners);
                const T y = gridsampler::compute_source_index<T>(
                    grid_row[2 * w + 1], H, PADDING, align_corners);
                const int64_t ix_nw = static_cast<int64_t>(std::floor(x));
                const int64_t iy_nw = static_cast<int64_t>(std::floor(y));
                const T nw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(iy_nw + 1 - y);
                const T ne = static_cast<T>(x - ix_nw) * static_cast<T>(iy_nw + 1 - y);
                const T sw = static_cast<T>(ix_nw + 1 - x) * static_cast<T>(y - iy_nw);
                const T se = static_cast<T>(x - ix_nw) * static_cast<T>(y - iy_nw);
                // Per-tap bounds test in every padding mode: an unclipped NaN
                // coordinate casts to an out-of-range index and must
                // contribute nothing, exactly like the scalar frame.
                const bool ok_nw = gridsampler::within_bounds_2d(iy_nw, ix_nw, H, W);
                const bool ok_ne = gridsampler::within_bounds_2d(iy_nw, ix_nw + 1, H, W);
                const bool ok_sw = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw, H, W);
                const bool ok_se = gridsampler::within_bounds_2d(iy_nw + 1, ix_nw + 1, H, W);
                // Accumulate in nw, ne, sw, se order from +0 like the scalar
                // frame; skipped taps contribute nothing at all.
                T* out_px = out_n + (row_out + w) * C;
                int64_t c = 0;
                for (; c + step <= C; c += step) {
                    Vec<T> acc = Vec<T>(T(0));
                    if (ok_nw)
                        acc = acc + Vec<T>::loadu(inp_n + iy_nw * inp_sH + ix_nw * inp_sW + c) * Vec<T>(nw);
                    if (ok_ne)
                        acc = acc + Vec<T>::loadu(inp_n + iy_nw * inp_sH + (ix_nw + 1) * inp_sW + c) * Vec<T>(ne);
                    if (ok_sw)
                        acc = acc + Vec<T>::loadu(inp_n + (iy_nw + 1) * inp_sH + ix_nw * inp_sW + c) * Vec<T>(sw);
                    if (ok_se)
                        acc = acc + Vec<T>::loadu(inp_n + (iy_nw + 1) * inp_sH + (ix_nw + 1) * inp_sW + c) * Vec<T>(se);
                    acc.store(out_px + c);
                }
                for (; c < C; ++c) {
                    T acc = T(0);
                    if (ok_nw) acc += inp_n[iy_nw * inp_sH + ix_nw * inp_sW + c] * nw;
                    if (ok_ne) acc += inp_n[iy_nw * inp_sH + (ix_nw + 1) * inp_sW + c] * ne;
                    if (ok_sw) acc += inp_n[(iy_nw + 1) * inp_sH + ix_nw * inp_sW + c] * sw;
                    if (ok_se) acc += inp_n[(iy_nw + 1) * inp_sH + (ix_nw + 1) * inp_sW + c] * se;
                    out_px[c] = acc;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Nearest
// ---------------------------------------------------------------------------
template <typename T, int PADDING>
__attribute__((optimize("-ffp-contract=off")))
void grid_sample2d_nearest_cl(const T* inp, const T* grid, T* out,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t oH, int64_t oW, bool align_corners) {
    const int64_t inp_sH = W * C;
    const int64_t inp_sW = C;
    const int64_t out_plane = oH * oW;
    constexpr int64_t step = Vec<T>::size();

    parallel_for(0, N * oH, 1, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const int64_t n = task / oH;
            const int64_t h = task - n * oH;
            const T* inp_n = inp + n * C * H * W;
            const T* grid_row = grid + (n * oH + h) * oW * 2;
            T* out_n = out + n * C * out_plane;
            const int64_t row_out = h * oW;
            for (int64_t w = 0; w < oW; ++w) {
                const T x = gridsampler::compute_source_index<T>(
                    grid_row[2 * w], W, PADDING, align_corners);
                const T y = gridsampler::compute_source_index<T>(
                    grid_row[2 * w + 1], H, PADDING, align_corners);
                // round-to-nearest-even, matching the scalar nearbyint.
                const int64_t ix_near = static_cast<int64_t>(std::nearbyint(x));
                const int64_t iy_near = static_cast<int64_t>(std::nearbyint(y));
                T* out_px = out_n + (row_out + w) * C;
                if (gridsampler::within_bounds_2d(iy_near, ix_near, H, W)) {
                    const T* src_px = inp_n + iy_near * inp_sH + ix_near * inp_sW;
                    int64_t c = 0;
                    for (; c + step <= C; c += step)
                        Vec<T>::loadu(src_px + c).store(out_px + c);
                    for (; c < C; ++c)
                        out_px[c] = src_px[c];
                } else {
                    int64_t c = 0;
                    for (; c + step <= C; c += step)
                        Vec<T>(T(0)).store(out_px + c);
                    for (; c < C; ++c)
                        out_px[c] = T(0);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Entry: switch interpolation/padding at run time inside each tier binary.
// ---------------------------------------------------------------------------
template <typename T>
void grid_sample2d_cl_typed(const void* in_, const void* grid_, void* out_,
                            int64_t N, int64_t C, int64_t H, int64_t W,
                            int64_t oH, int64_t oW, int interpolation_mode,
                            int padding_mode, bool align_corners) {
    const T* in = static_cast<const T*>(in_);
    const T* grid = static_cast<const T*>(grid_);
    T* out = static_cast<T*>(out_);
    if (interpolation_mode == Interp::Bilinear) {
        switch (padding_mode) {
            case Pad::Zeros:
                grid_sample2d_bilinear_cl<T, Pad::Zeros>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            case Pad::Border:
                grid_sample2d_bilinear_cl<T, Pad::Border>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            default:
                grid_sample2d_bilinear_cl<T, Pad::Reflection>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
        }
    }
    if (interpolation_mode == Interp::Nearest) {
        switch (padding_mode) {
            case Pad::Zeros:
                grid_sample2d_nearest_cl<T, Pad::Zeros>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            case Pad::Border:
                grid_sample2d_nearest_cl<T, Pad::Border>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
            default:
                grid_sample2d_nearest_cl<T, Pad::Reflection>(in, grid, out, N, C, H, W, oH, oW, align_corners);
                return;
        }
    }
    TP_THROW(NotImplementedError, "grid_sampler_2d channels-last: bicubic stays on the scalar frame");
}

void grid_sample2d_cl_impl(const void* in, const void* grid, void* out,
                           int64_t N, int64_t C, int64_t H, int64_t W,
                           int64_t oH, int64_t oW, int interpolation_mode,
                           int padding_mode, int align_corners, int dtype) {
    const bool ac = align_corners != 0;
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            grid_sample2d_cl_typed<float>(in, grid, out, N, C, H, W, oH, oW,
                                          interpolation_mode, padding_mode, ac);
            return;
        case DType::Float64:
            grid_sample2d_cl_typed<double>(in, grid, out, N, C, H, W, oH, oW,
                                           interpolation_mode, padding_mode, ac);
            return;
        default:
            TP_THROW(NotImplementedError, "grid_sampler_2d channels-last: unsupported dtype");
    }
}

}  // namespace

// DEFAULT/AVX2 tier copies take the REGISTER_DISPATCH slots; the AVX512 tier
// copy uses the opt-in AVX512 slot instead (same as the pooling kernels).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(grid_sample2d_cl_stub, &grid_sample2d_cl_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(grid_sample2d_cl_stub, &grid_sample2d_cl_impl);
#endif

}  // namespace cpu
}  // namespace tensorplay
