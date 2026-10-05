// Channels-last pooling cores, compiled once per CPU capability tier (see
// TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Each copy lands in the
// CPU_CAPABILITY inline namespace and registers its own slot on the stubs
// declared in cpu/PoolingKernels.h; DispatchStub picks the best tier at
// runtime.  Geometry, divisor rules and dispatcher registration stay in the
// base tier (PoolingKernels.cpp).

#include "cpu/PoolingKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <algorithm>
#include <array>
#include <limits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// Output pixels are independent, so workers own whole pixels and the channel
// span of one pixel is the vectorized dimension (long enough to amortize the
// per-pixel setup, unlike the short pooling-window scans of the NCHW frames).

template <typename T>
void avg_pool2d_cl_typed(const T* in, T* out,
                         int64_t N, int64_t C, int64_t H, int64_t W,
                         int64_t oH, int64_t oW,
                         int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                         int64_t pH, int64_t pW,
                         bool count_include_pad, int64_t divisor_override) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
            const int64_t ow = item % oW;
            const int64_t oh = (item / oW) % oH;
            const int64_t n = item / (oW * oH);
            T* out_lane = out + item * C;
            const int64_t h_start = oh * sH - pH;
            const int64_t w_start = ow * sW - pW;
            const int64_t ih0 = std::max(h_start, int64_t(0));
            const int64_t iw0 = std::max(w_start, int64_t(0));
            const int64_t ih1 = std::min(h_start + kH, H);
            const int64_t iw1 = std::min(w_start + kW, W);
            // Window extent over input+padding, the divisor when padded
            // positions count; an all-padding window divides zero the same
            // way the NCHW frame does.
            const int64_t clip_h = std::min(h_start + kH, H + pH) - h_start;
            const int64_t clip_w = std::min(w_start + kW, W + pW) - w_start;
            const int64_t divisor = divisor_override != 0
                ? divisor_override
                : (count_include_pad ? clip_h * clip_w
                                     : (ih1 - ih0) * (iw1 - iw0));

            // Pass I: zero the accumulator lane.
            int64_t c = 0;
            for (; c < len; c += V) {
                Vec(T(0)).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = T(0);
            }
            // Pass II: sum the window.  Every channel accumulates the window
            // positions in the same order as the NCHW scalar frame, so both
            // memory formats round identically.
            for (int64_t ih = ih0; ih < ih1; ++ih) {
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t iw = iw0; iw < iw1; ++iw) {
                    const T* in_lane = row + iw * C;
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        (Vec::loadu(out_lane + c2) + Vec::loadu(in_lane + c2))
                            .store(out_lane + c2);
                    }
                    for (; c2 < C; ++c2) {
                        out_lane[c2] += in_lane[c2];
                    }
                }
            }
            // Pass III: divide.
            const T dv = static_cast<T>(divisor);
            c = 0;
            for (; c < len; c += V) {
                (Vec::loadu(out_lane + c) / Vec(dv)).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = out_lane[c] / dv;
            }
        }
    });
}

// Max-reduction update for one window position.  A NaN takes the maximum
// (carrying its own bit pattern, so the last NaN in scan order wins), the
// same contract the scalar max-pool frames follow.  `mask` carries all-bits
// lanes where the value updates; the index pass replicates it per lane.
template <typename T>
inline void max_update_lane(T* out_p, const T* in_p, int64_t* ind_p,
                            int64_t plane_off) {
    const bool take = (in_p[0] != in_p[0]) || (in_p[0] > out_p[0]);
    if (take) {
        out_p[0] = in_p[0];
        ind_p[0] = plane_off;
    }
}

template <typename T>
void max_pool2d_cl_typed(const T* in, T* out, int64_t* ind,
                         int64_t N, int64_t C, int64_t H, int64_t W,
                         int64_t oH, int64_t oW,
                         int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                         int64_t pH, int64_t pW, int64_t dH, int64_t dW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    using vec::int_same_size_t;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    // Index element of channel c for one output pixel; the indices tensor is
    // dense NCHW, so consecutive channels sit one output plane apart.
    const int64_t plane_out = oH * oW;
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        std::array<int_same_size_t<T>, V> mbits;
        for (int64_t item = begin; item < end; ++item) {
            const int64_t ow = item % oW;
            const int64_t oh = (item / oW) % oH;
            const int64_t n = item / (oW * oH);
            T* out_lane = out + item * C;
            int64_t* ind_pix = ind + n * C * plane_out + oh * oW + ow;
            const int64_t h_start = oh * sH - pH;
            const int64_t w_start = ow * sW - pW;
            // Dilated windows clip to the input extent; the strides-1 case
            // keeps the branch-free closed form of the NCHW frame.
            int64_t kh0 = 0, kh1 = kH;
            if (dH == 1) {
                if (h_start < 0) kh0 = -h_start;
                if (h_start + kH > H) kh1 = H - h_start;
            } else {
                while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H) --kh1;
            }
            int64_t kw0 = 0, kw1 = kW;
            if (dW == 1) {
                if (w_start < 0) kw0 = -w_start;
                if (w_start + kW > W) kw1 = W - w_start;
            } else {
                while (kw0 < kW && w_start + kw0 * dW < 0) ++kw0;
                while (kw1 > kw0 && w_start + (kw1 - 1) * dW >= W) --kw1;
            }
            // Init the lane to the max identity and the index to the "no
            // winner" sentinel; a window clipped away entirely keeps both,
            // matching the NCHW frame.
            int64_t c = 0;
            for (; c < len; c += V) {
                Vec(lo).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = lo;
            }
            for (c = 0; c < C; ++c) {
                ind_pix[c * plane_out] = -1;
            }
            for (int64_t kh = kh0; kh < kh1; ++kh) {
                const int64_t ih = h_start + kh * dH;
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t kw = kw0; kw < kw1; ++kw) {
                    const int64_t iw = w_start + kw * dW;
                    const T* in_lane = row + iw * C;
                    const int64_t plane_off = ih * W + iw;
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        const Vec maxv = Vec::loadu(out_lane + c2);
                        const Vec val = Vec::loadu(in_lane + c2);
                        const Vec mask = (val != val) | (val > maxv);
                        const Vec next = Vec::blendv(maxv, val, mask);
                        next.store(out_lane + c2);
                        // Horizontal change test first; only a lane whose max
                        // moved pays for the scalar index walk.
                        if (next.ne(maxv).reduce_add() > T(0)) {
                            mask.store(reinterpret_cast<T*>(mbits.data()));
                            const int64_t lane_max =
                                std::min(c2 + V, C) - c2;
                            for (int64_t l = 0; l < lane_max; ++l) {
                                if (mbits[l] & 0x01) {
                                    ind_pix[(c2 + l) * plane_out] = plane_off;
                                }
                            }
                        }
                    }
                    for (; c2 < C; ++c2) {
                        max_update_lane<T>(out_lane + c2, in_lane + c2,
                                           ind_pix + c2 * plane_out, plane_off);
                    }
                }
            }
        }
    });
}

// Adaptive variants: each output pixel's window is derived from the
// input/output extents with the same start/end arithmetic as the NCHW frame,
// so no window parameters travel through the stub.
template <typename T>
void adaptive_avg_pool2d_cl_typed(const T* in, T* out,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        for (int64_t item = begin; item < end; ++item) {
            const int64_t ow = item % oW;
            const int64_t oh = (item / oW) % oH;
            const int64_t n = item / (oW * oH);
            const int64_t ih0 = (oh * H) / oH;
            const int64_t ih1 = ((oh + 1) * H + oH - 1) / oH;
            const int64_t iw0 = (ow * W) / oW;
            const int64_t iw1 = ((ow + 1) * W + oW - 1) / oW;
            T* out_lane = out + item * C;

            int64_t c = 0;
            for (; c < len; c += V) {
                Vec(T(0)).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = T(0);
            }
            for (int64_t ih = ih0; ih < ih1; ++ih) {
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t iw = iw0; iw < iw1; ++iw) {
                    const T* in_lane = row + iw * C;
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        (Vec::loadu(out_lane + c2) + Vec::loadu(in_lane + c2))
                            .store(out_lane + c2);
                    }
                    for (; c2 < C; ++c2) {
                        out_lane[c2] += in_lane[c2];
                    }
                }
            }
            const T dv = static_cast<T>((ih1 - ih0) * (iw1 - iw0));
            c = 0;
            for (; c < len; c += V) {
                (Vec::loadu(out_lane + c) / Vec(dv)).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = out_lane[c] / dv;
            }
        }
    });
}

template <typename T>
void adaptive_max_pool2d_cl_typed(const T* in, T* out, int64_t* ind,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    using vec::int_same_size_t;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    const int64_t plane_out = oH * oW;
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        std::array<int_same_size_t<T>, V> mbits;
        for (int64_t item = begin; item < end; ++item) {
            const int64_t ow = item % oW;
            const int64_t oh = (item / oW) % oH;
            const int64_t n = item / (oW * oH);
            const int64_t ih0 = (oh * H) / oH;
            const int64_t ih1 = ((oh + 1) * H + oH - 1) / oH;
            const int64_t iw0 = (ow * W) / oW;
            const int64_t iw1 = ((ow + 1) * W + oW - 1) / oW;
            T* out_lane = out + item * C;
            int64_t* ind_pix = ind + n * C * plane_out + oh * oW + ow;

            int64_t c = 0;
            for (; c < len; c += V) {
                Vec(lo).store(out_lane + c);
            }
            for (; c < C; ++c) {
                out_lane[c] = lo;
            }
            for (c = 0; c < C; ++c) {
                ind_pix[c * plane_out] = -1;
            }
            for (int64_t ih = ih0; ih < ih1; ++ih) {
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t iw = iw0; iw < iw1; ++iw) {
                    const T* in_lane = row + iw * C;
                    const int64_t plane_off = ih * W + iw;
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        const Vec maxv = Vec::loadu(out_lane + c2);
                        const Vec val = Vec::loadu(in_lane + c2);
                        // A NaN takes the maximum, like the NCHW frame.
                        const Vec mask = (val != val) | (val > maxv);
                        const Vec next = Vec::blendv(maxv, val, mask);
                        next.store(out_lane + c2);
                        if (next.ne(maxv).reduce_add() > T(0)) {
                            mask.store(reinterpret_cast<T*>(mbits.data()));
                            const int64_t lane_max =
                                std::min(c2 + V, C) - c2;
                            for (int64_t l = 0; l < lane_max; ++l) {
                                if (mbits[l] & 0x01) {
                                    ind_pix[(c2 + l) * plane_out] = plane_off;
                                }
                            }
                        }
                    }
                    for (; c2 < C; ++c2) {
                        max_update_lane<T>(out_lane + c2, in_lane + c2,
                                           ind_pix + c2 * plane_out, plane_off);
                    }
                }
            }
        }
    });
}

void avg_pool2d_cl_impl(const void* in, void* out,
                        int64_t N, int64_t C, int64_t H, int64_t W,
                        int64_t oH, int64_t oW,
                        int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                        int64_t pH, int64_t pW,
                        bool count_include_pad, int64_t divisor_override,
                        int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            avg_pool2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out),
                N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW,
                count_include_pad, divisor_override);
            break;
        case DType::Float64:
            avg_pool2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out),
                N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW,
                count_include_pad, divisor_override);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "avg_pool2d: channels-last kernel supports only float "
                     "and double");
    }
}

void max_pool2d_cl_impl(const void* in, void* out, int64_t* ind,
                        int64_t N, int64_t C, int64_t H, int64_t W,
                        int64_t oH, int64_t oW,
                        int64_t kH, int64_t kW, int64_t sH, int64_t sW,
                        int64_t pH, int64_t pW, int64_t dH, int64_t dW,
                        int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            max_pool2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out), ind,
                N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW);
            break;
        case DType::Float64:
            max_pool2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out), ind,
                N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d: channels-last kernel supports only float "
                     "and double");
    }
}

void adaptive_avg_pool2d_cl_impl(const void* in, void* out,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            adaptive_avg_pool2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out),
                N, C, H, W, oH, oW);
            break;
        case DType::Float64:
            adaptive_avg_pool2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out),
                N, C, H, W, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_avg_pool2d: channels-last kernel supports "
                     "only float and double");
    }
}

void adaptive_max_pool2d_cl_impl(const void* in, void* out, int64_t* ind,
                                 int64_t N, int64_t C, int64_t H, int64_t W,
                                 int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            adaptive_max_pool2d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out), ind,
                N, C, H, W, oH, oW);
            break;
        case DType::Float64:
            adaptive_max_pool2d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out), ind,
                N, C, H, W, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d: channels-last kernel supports "
                     "only float and double");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(avg_pool2d_cl_stub, &avg_pool2d_cl_impl);
REGISTER_DISPATCH(max_pool2d_cl_stub, &max_pool2d_cl_impl);
REGISTER_DISPATCH(adaptive_avg_pool2d_cl_stub, &adaptive_avg_pool2d_cl_impl);
REGISTER_DISPATCH(adaptive_max_pool2d_cl_stub, &adaptive_max_pool2d_cl_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(avg_pool2d_cl_stub, &avg_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(max_pool2d_cl_stub, &max_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_avg_pool2d_cl_stub,
                              &adaptive_avg_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_max_pool2d_cl_stub,
                              &adaptive_max_pool2d_cl_impl);
#endif

} // namespace cpu
} // namespace tensorplay
