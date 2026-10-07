// Channels-last upsampling cores, compiled once per CPU capability tier (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Each copy lands in the
// CPU_CAPABILITY inline namespace and registers its own slot on the stubs
// declared in cpu/UpsampleKernels.h; DispatchStub picks the best tier at
// runtime.  Geometry, scale computation and dispatcher registration stay in
// the base tier (UpsampleKernels.cpp).

#include "cpu/UpsampleKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// Source index for the non-cubic area-pixel interpolation: align_corners
// places samples on the corners, otherwise the half-pixel-centered mapping
// clamps negatives to zero. The index is evaluated in float for every dtype
// (matching the scalar frames' rounding) — the lambdas and blends then
// continue in the element type.
template <typename T>
inline T area_pixel_source_index(T scale, int64_t dst_index, bool align_corners) {
    if (align_corners) {
        return scale * static_cast<T>(dst_index);
    }
    const T src_idx = scale * (static_cast<T>(dst_index) + T(0.5)) - T(0.5);
    return src_idx < T(0) ? T(0) : src_idx;
}

// The element-type weight for one source index: the index arithmetic runs in
// float (single-sourced with the scalar frames), the remaining lambda math
// runs in the element type.
template <typename T>
inline void bilinear_axis_weights(float scale, int64_t dst_index,
                                  bool align_corners, int64_t src_size,
                                  int64_t* src, int64_t* src_p,
                                  T* lambda0, T* lambda1) {
    const T r = static_cast<T>(area_pixel_source_index<float>(
        scale, dst_index, align_corners));
    const int64_t i = static_cast<int64_t>(r);
    *src = i;
    *src_p = (i < src_size - 1) ? 1 : 0;
    *lambda1 = r - static_cast<T>(i);
    *lambda0 = static_cast<T>(1) - *lambda1;
}

// Nearest neighbor: each output pixel copies its source pixel's channel
// span, the vectorized dimension.
template <typename T>
void upsample_nearest2d_cl_typed(const T* in, T* out,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t oH, int64_t oW,
                                 double sh, double sw) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    // Index arithmetic runs in float, matching the scalar frames for every
    // dtype (they compute the source index through the float helper).
    const float hscale = static_cast<float>(sh);
    const float wscale = static_cast<float>(sw);
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
            const int64_t ow = item % oW;
            const int64_t oh = (item / oW) % oH;
            const int64_t n = item / (oW * oH);
            const int64_t ih = (H == oH)
                ? oh
                : std::min(static_cast<int64_t>(std::floor(static_cast<float>(oh) * hscale)), H - 1);
            const int64_t iw = (W == oW)
                ? ow
                : std::min(static_cast<int64_t>(std::floor(static_cast<float>(ow) * wscale)), W - 1);
            const T* src = in + ((n * H + ih) * W + iw) * C;
            T* dst = out + item * C;
            int64_t c = 0;
            for (; c < C - (C % V); c += V) {
                Vec::loadu(src + c).store(dst + c);
            }
            for (; c < C; ++c) {
                dst[c] = src[c];
            }
        }
    });
}

// Bilinear: one worker owns a whole output row, so the vertical indices and
// lambdas are computed once; the horizontal weights come from a task-local
// table built with the same element-type arithmetic as the NCHW frame.  The
// blend evaluates h0*(w0*a + w1*b) + h1*(w0*c + w1*d) in the same
// association order as the scalar frame.  Contraction stays off so the tier
// copies (which compile with FMA) keep the same per-operation rounding as
// the base-tier scalar frame, which has no FMA available.
template <typename T>
TP_NO_FP_CONTRACT
void upsample_bilinear2d_cl_typed(const T* in, T* out,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t oH, int64_t oW,
                                  bool align_corners, double rh, double rw) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    // The scales arrive widened to double from the base tier; the scalar
    // frames narrow them back to float for the index arithmetic (both dtypes)
    // and continue the lambda/blend math in the element type.
    const float rheight_f = static_cast<float>(rh);
    const float rwidth_f = static_cast<float>(rw);
    const int64_t len = C - (C % V);
    const int64_t rows = N * oH;
    parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        std::vector<T> w0l_tab(oW), w1l_tab(oW);
        std::vector<int64_t> w1_tab(oW), w1p_tab(oW);
        for (int64_t ow = 0; ow < oW; ++ow) {
            bilinear_axis_weights<T>(rwidth_f, ow, align_corners, W,
                                     &w1_tab[ow], &w1p_tab[ow],
                                     &w0l_tab[ow], &w1l_tab[ow]);
        }
        for (int64_t r = begin; r < end; ++r) {
            const int64_t oh = r % oH;
            const int64_t n = r / oH;
            int64_t h1, h1p;
            T h1lambda, h0lambda;
            bilinear_axis_weights<T>(rheight_f, oh, align_corners, H,
                                     &h1, &h1p, &h0lambda, &h1lambda);
            const Vec h0v(h0lambda), h1v(h1lambda);
            const T* row0 = in + (n * H + h1) * W * C;
            const T* row1 = row0 + h1p * W * C;
            T* out_row = out + r * oW * C;
            for (int64_t ow = 0; ow < oW; ++ow) {
                const T* i00 = row0 + w1_tab[ow] * C;
                const T* i01 = i00 + w1p_tab[ow] * C;
                const T* i10 = row1 + w1_tab[ow] * C;
                const T* i11 = i10 + w1p_tab[ow] * C;
                const Vec w0v(w0l_tab[ow]), w1v(w1l_tab[ow]);
                T* dst = out_row + ow * C;
                int64_t c = 0;
                for (; c < len; c += V) {
                    const Vec v00 = Vec::loadu(i00 + c);
                    const Vec v01 = Vec::loadu(i01 + c);
                    const Vec v10 = Vec::loadu(i10 + c);
                    const Vec v11 = Vec::loadu(i11 + c);
                    const Vec t0 = v00 * w0v + v01 * w1v;
                    const Vec t1 = v10 * w0v + v11 * w1v;
                    (t0 * h0v + t1 * h1v).store(dst + c);
                }
                for (; c < C; ++c) {
                    const T t0 = i00[c] * w0l_tab[ow] + i01[c] * w1l_tab[ow];
                    const T t1 = i10[c] * w0l_tab[ow] + i11[c] * w1l_tab[ow];
                    dst[c] = t0 * h0lambda + t1 * h1lambda;
                }
            }
        }
    });
}

void upsample_nearest2d_cl_impl(const void* in, void* out,
                                int64_t N, int64_t C, int64_t H, int64_t W,
                                int64_t oH, int64_t oW,
                                double sh, double sw, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            upsample_nearest2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out),
                N, C, H, W, oH, oW, sh, sw);
            break;
        case DType::Float64:
            upsample_nearest2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out),
                N, C, H, W, oH, oW, sh, sw);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "upsample_nearest2d: channels-last kernel supports only "
                     "float and double");
    }
}

void upsample_bilinear2d_cl_impl(const void* in, void* out,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t oH, int64_t oW,
                                 int align_corners, double rh, double rw,
                                 int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            upsample_bilinear2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out),
                N, C, H, W, oH, oW, align_corners != 0, rh, rw);
            break;
        case DType::Float64:
            upsample_bilinear2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out),
                N, C, H, W, oH, oW, align_corners != 0, rh, rw);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "upsample_bilinear2d: channels-last kernel supports only "
                     "float and double");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(upsample_nearest2d_cl_stub, &upsample_nearest2d_cl_impl);
REGISTER_DISPATCH(upsample_bilinear2d_cl_stub, &upsample_bilinear2d_cl_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(upsample_nearest2d_cl_stub,
                              &upsample_nearest2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(upsample_bilinear2d_cl_stub,
                              &upsample_bilinear2d_cl_impl);
#endif

} // namespace cpu
} // namespace tensorplay
