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
#include <memory>

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
    using I = vec::int_same_size_t<T>;
    using iVec = tensorplay::vec::Vectorized<I>;
    constexpr int64_t V = Vec::size();
    static_assert(iVec::size() == V, "index lanes must follow value lanes");
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        // Scratch: this pixel's current winner index per channel, kept as one
        // lane of integer vectors so the sweep updates values and indices
        // with the same blend and no scalar walk runs per window position.
        std::unique_ptr<I[]> idxstate(new I[C > 0 ? C : 1]);
        int64_t rest = begin;
        int64_t ow = rest % oW; rest /= oW;
        int64_t oh = rest % oH; rest /= oH;
        int64_t n = rest;
        for (int64_t item = begin; item < end; ++item) {
            T* out_lane = out + item * C;
            int64_t* ind_lane = ind + item * C;
            const int64_t h_start = oh * sH - pH;
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
            const int64_t w_start = ow * sW - pW;
            int64_t kw0 = 0, kw1 = kW;
            if (dW == 1) {
                if (w_start < 0) kw0 = -w_start;
                if (w_start + kW > W) kw1 = W - w_start;
            } else {
                while (kw0 < kW && w_start + kw0 * dW < 0) ++kw0;
                while (kw1 > kw0 && w_start + (kw1 - 1) * dW >= W) --kw1;
            }
            // Start from the max identity and the "no winner" sentinel; a
            // window clipped away entirely keeps both, matching the NCHW
            // frame.
            const iVec minus1(I(-1));
            {
                int64_t c = 0;
                for (; c < len; c += V) {
                    Vec(lo).store(out_lane + c);
                    minus1.store(idxstate.get() + c);
                }
                for (; c < C; ++c) {
                    out_lane[c] = lo;
                    idxstate[c] = I(-1);
                }
            }
            for (int64_t kh = kh0; kh < kh1; ++kh) {
                const int64_t ih = h_start + kh * dH;
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t kw = kw0; kw < kw1; ++kw) {
                    const int64_t iw = w_start + kw * dW;
                    const T* in_lane = row + iw * C;
                    const I off = I(ih * W + iw);
                    const iVec off_vec(off);
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        const Vec maxv = Vec::loadu(out_lane + c2);
                        const Vec val = Vec::loadu(in_lane + c2);
                        // A NaN takes the maximum, like the NCHW frame; the
                        // same mask drives the value and index blend.
                        const Vec mask = (val != val) | (val > maxv);
                        Vec::blendv(maxv, val, mask).store(out_lane + c2);
                        // All-ones float lanes are all-ones integer bits, so
                        // the mask crosses to the index blend unchanged.
                        const iVec imask = vec::cast<I>(mask);
                        const iVec cur = iVec::loadu(idxstate.get() + c2);
                        iVec::blendv(cur, off_vec, imask)
                            .store(idxstate.get() + c2);
                    }
                    for (; c2 < C; ++c2) {
                        const T v = in_lane[c2];
                        if ((v != v) || (v > out_lane[c2])) {
                            out_lane[c2] = v;
                            idxstate[c2] = off;
                        }
                    }
                }
            }
            for (int64_t c = 0; c < C; ++c) {
                ind_lane[c] = int64_t(idxstate[c]);
            }
            ++ow;
            if (ow == oW) {
                ow = 0;
                if (++oh == oH) {
                    oh = 0;
                    ++n;
                }
            }
        }
    });
}

// Scatter twin of the scalar with-indices backward: each output position's
// grad_output channel block reads as one vector, and every lane lands on its
// own argmax position of the NHWC grad_input.  Output positions run in the
// scalar frame's (H, W) lexicographic order, so overlapping windows add
// their contributions to each input element in the same sequence and the
// strided grad_input matches the NCHW frame bit for bit.
template <typename T>
void max_pool2d_backward_cl_typed(const T* gout, const int64_t* ind, T* gin,
                                  int64_t N, int64_t C,
                                  int64_t H, int64_t W,
                                  int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const int64_t in_plane = H * W;
    const int64_t out_plane = oH * oW;
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        T gbuf[V];
        for (int64_t n = begin; n < end; ++n) {
            const T* go_n = gout + n * out_plane * C;
            const int64_t* ind_n = ind + n * out_plane * C;
            T* gi_n = gin + n * in_plane * C;
            for (int64_t o = 0; o < out_plane; ++o) {
                const T* go_lane = go_n + o * C;
                const int64_t* ind_o = ind_n + o * C;
                int64_t c = 0;
                for (; c < len; c += V) {
                    Vec::loadu(go_lane + c).store(gbuf);
                    for (int64_t l = 0; l < V; ++l) {
                        const int64_t max_idx = ind_o[c + l];
                        if (max_idx >= 0) {
                            gi_n[max_idx * C + c + l] += gbuf[l];
                        }
                    }
                }
                for (; c < C; ++c) {
                    const int64_t max_idx = ind_o[c];
                    if (max_idx >= 0) {
                        gi_n[max_idx * C + c] += go_lane[c];
                    }
                }
            }
        }
    });
}

// Channels-last-3d twin of max_pool2d_cl_typed.  Workers own single output
// pixels, so the value and index lanes a worker blends are one channel block
// each and stay cache-resident regardless of the channel width or the output
// row length; window positions arrive in the scalar frame's (kd, kh, kw)
// order, and the per-lane update predicate is the scalar frame's own
// "strictly greater, or NaN", so values, first-max/last-NaN selection and
// the winning positions all match bit for bit.  Index lanes live in
// same-width integer vectors blended off the same mask as the values, and
// the winners publish densely into the channels-last-3d index tensor.
template <typename T>
void max_pool3d_cl_typed(const T* in, T* out, int64_t* ind,
                         int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                         int64_t oD, int64_t oH, int64_t oW,
                         int64_t kD, int64_t kH, int64_t kW,
                         int64_t sD, int64_t sH, int64_t sW,
                         int64_t pD, int64_t pH, int64_t pW,
                         int64_t dD, int64_t dH, int64_t dW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    using I = vec::int_same_size_t<T>;
    using iVec = tensorplay::vec::Vectorized<I>;
    constexpr int64_t V = Vec::size();
    static_assert(iVec::size() == V, "index lanes must follow value lanes");
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    // One output pixel per task: the value and index lanes a task blends are
    // a single channel block each, so the read-modify-write set stays
    // cache-resident no matter how wide the channel axis or the output row.
    const int64_t tasks = N * oD * oH * oW;
    parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
        // Scratch: this pixel's current winner index per channel, kept as one
        // lane of integer vectors so the sweep updates values and indices
        // with the same blend and no scalar walk runs per window position.
        std::unique_ptr<I[]> idxstate(new I[C > 0 ? C : 1]);
        int64_t rest = begin;
        int64_t ow = rest % oW; rest /= oW;
        int64_t oh = rest % oH; rest /= oH;
        int64_t od = rest % oD; rest /= oD;
        int64_t n = rest;
        for (int64_t task = begin; task < end; ++task) {
            T* out_lane = out + task * C;
            int64_t* ind_lane = ind + task * C;
            const int64_t d_start = od * sD - pD;
            int64_t kd0 = 0, kd1 = kD;
            if (dD == 1) {
                if (d_start < 0) kd0 = -d_start;
                if (d_start + kD > D) kd1 = D - d_start;
            } else {
                while (kd0 < kD && d_start + kd0 * dD < 0) ++kd0;
                while (kd1 > kd0 && d_start + (kd1 - 1) * dD >= D) --kd1;
            }
            const int64_t h_start = oh * sH - pH;
            int64_t kh0 = 0, kh1 = kH;
            if (dH == 1) {
                if (h_start < 0) kh0 = -h_start;
                if (h_start + kH > H) kh1 = H - h_start;
            } else {
                while (kh0 < kH && h_start + kh0 * dH < 0) ++kh0;
                while (kh1 > kh0 && h_start + (kh1 - 1) * dH >= H) --kh1;
            }
            // Start from the max identity and the "no winner" sentinel; a
            // window clipped away entirely keeps both, matching the NCHW
            // frame.
            const iVec minus1(I(-1));
            {
                int64_t c = 0;
                for (; c < len; c += V) {
                    Vec(lo).store(out_lane + c);
                    minus1.store(idxstate.get() + c);
                }
                for (; c < C; ++c) {
                    out_lane[c] = lo;
                    idxstate[c] = I(-1);
                }
            }
            for (int64_t kd = kd0; kd < kd1; ++kd) {
                const int64_t di = d_start + kd * dD;
                const int64_t slice_off = di * H * W;
                const T* slice = in + (n * D + di) * H * W * C;
                for (int64_t kh = kh0; kh < kh1; ++kh) {
                    const int64_t hi = h_start + kh * dH;
                    const int64_t row_off = hi * W;
                    const T* row = slice + hi * W * C;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        const int64_t wi = ow * sW - pW + kw * dW;
                        if (wi < 0 || wi >= W) continue;
                        const T* in_lane = row + wi * C;
                        const I off = I(slice_off + row_off + wi);
                        const iVec off_vec(off);
                        int64_t c2 = 0;
                        for (; c2 < len; c2 += V) {
                            const Vec maxv = Vec::loadu(out_lane + c2);
                            const Vec val = Vec::loadu(in_lane + c2);
                            // A NaN takes the maximum, like the NCHW
                            // frame; the same mask drives the value and
                            // index blend.
                            const Vec mask = (val != val) | (val > maxv);
                            Vec::blendv(maxv, val, mask).store(out_lane + c2);
                            // All-ones float lanes are all-ones integer
                            // bits, so the mask crosses to the index
                            // blend unchanged.
                            const iVec imask = vec::cast<I>(mask);
                            const iVec cur = iVec::loadu(idxstate.get() + c2);
                            iVec::blendv(cur, off_vec, imask)
                                .store(idxstate.get() + c2);
                        }
                        for (; c2 < C; ++c2) {
                            const T v = in_lane[c2];
                            if ((v != v) || (v > out_lane[c2])) {
                                out_lane[c2] = v;
                                idxstate[c2] = off;
                            }
                        }
                    }
                }
            }
            for (int64_t c = 0; c < C; ++c) {
                ind_lane[c] = int64_t(idxstate[c]);
            }
            ++ow;
            if (ow == oW) {
                ow = 0;
                if (++oh == oH) {
                    oh = 0;
                    if (++od == oD) {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    });
}

// Scatter twin of the scalar with-indices backward: each output position's
// grad_output channel block reads as one vector, and every lane lands on its
// own argmax position of the NDHWC grad_input.  Output positions run in the
// scalar frame's (D, H, W) lexicographic order, so overlapping windows add
// their contributions to each input element in the same sequence and the
// strided grad_input matches the NCDHW frame bit for bit.
template <typename T>
void max_pool3d_backward_cl_typed(const T* gout, const int64_t* ind, T* gin,
                                  int64_t N, int64_t C,
                                  int64_t D, int64_t H, int64_t W,
                                  int64_t oD, int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t V = Vec::size();
    const int64_t len = C - (C % V);
    const int64_t in_vol = D * H * W;
    const int64_t out_vol = oD * oH * oW;
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        T gbuf[V];
        for (int64_t n = begin; n < end; ++n) {
            const T* go_n = gout + n * out_vol * C;
            const int64_t* ind_n = ind + n * out_vol * C;
            T* gi_n = gin + n * in_vol * C;
            for (int64_t od = 0; od < oD; ++od) {
                for (int64_t oh = 0; oh < oH; ++oh) {
                    for (int64_t ow = 0; ow < oW; ++ow) {
                        const int64_t o = (od * oH + oh) * oW + ow;
                        const T* go_lane = go_n + o * C;
                        const int64_t* ind_o = ind_n + o * C;
                        int64_t c = 0;
                        for (; c < len; c += V) {
                            Vec::loadu(go_lane + c).store(gbuf);
                            for (int64_t l = 0; l < V; ++l) {
                                const int64_t max_idx = ind_o[c + l];
                                if (max_idx >= 0) {
                                    gi_n[max_idx * C + c + l] += gbuf[l];
                                }
                            }
                        }
                        for (; c < C; ++c) {
                            const int64_t max_idx = ind_o[c];
                            if (max_idx >= 0) {
                                gi_n[max_idx * C + c] += go_lane[c];
                            }
                        }
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
    using I = vec::int_same_size_t<T>;
    using iVec = tensorplay::vec::Vectorized<I>;
    constexpr int64_t V = Vec::size();
    static_assert(iVec::size() == V, "index lanes must follow value lanes");
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    const int64_t items = N * oH * oW;
    parallel_for(0, items, 1, [&](int64_t begin, int64_t end) {
        // Scratch: this pixel's current winner index per channel, kept as one
        // lane of integer vectors so the sweep updates values and indices
        // with the same blend and no scalar walk runs per window position.
        std::unique_ptr<I[]> idxstate(new I[C > 0 ? C : 1]);
        int64_t rest = begin;
        int64_t ow = rest % oW; rest /= oW;
        int64_t oh = rest % oH; rest /= oH;
        int64_t n = rest;
        for (int64_t item = begin; item < end; ++item) {
            // Adaptive windows are derived from the extents with the same
            // start/end arithmetic as the NCHW frame and always contain at
            // least one position.
            T* out_lane = out + item * C;
            int64_t* ind_lane = ind + item * C;
            const int64_t ih0 = (oh * H) / oH;
            const int64_t ih1 = ((oh + 1) * H + oH - 1) / oH;
            const int64_t iw0 = (ow * W) / oW;
            const int64_t iw1 = ((ow + 1) * W + oW - 1) / oW;
            const iVec minus1(I(-1));
            {
                int64_t c = 0;
                for (; c < len; c += V) {
                    Vec(lo).store(out_lane + c);
                    minus1.store(idxstate.get() + c);
                }
                for (; c < C; ++c) {
                    out_lane[c] = lo;
                    idxstate[c] = I(-1);
                }
            }
            for (int64_t ih = ih0; ih < ih1; ++ih) {
                const T* row = in + (n * H + ih) * W * C;
                for (int64_t iw = iw0; iw < iw1; ++iw) {
                    const T* in_lane = row + iw * C;
                    const I off = I(ih * W + iw);
                    const iVec off_vec(off);
                    int64_t c2 = 0;
                    for (; c2 < len; c2 += V) {
                        const Vec maxv = Vec::loadu(out_lane + c2);
                        const Vec val = Vec::loadu(in_lane + c2);
                        // A NaN takes the maximum, like the NCHW frame; the
                        // same mask drives the value and index blend.
                        const Vec mask = (val != val) | (val > maxv);
                        Vec::blendv(maxv, val, mask).store(out_lane + c2);
                        // All-ones float lanes are all-ones integer bits, so
                        // the mask crosses to the index blend unchanged.
                        const iVec imask = vec::cast<I>(mask);
                        const iVec cur = iVec::loadu(idxstate.get() + c2);
                        iVec::blendv(cur, off_vec, imask)
                            .store(idxstate.get() + c2);
                    }
                    for (; c2 < C; ++c2) {
                        const T v = in_lane[c2];
                        if ((v != v) || (v > out_lane[c2])) {
                            out_lane[c2] = v;
                            idxstate[c2] = off;
                        }
                    }
                }
            }
            for (int64_t c = 0; c < C; ++c) {
                ind_lane[c] = int64_t(idxstate[c]);
            }
            ++ow;
            if (ow == oW) {
                ow = 0;
                if (++oh == oH) {
                    oh = 0;
                    ++n;
                }
            }
        }
    });
}

// Adaptive channels-last-3d max pool.  Workers own single output pixels and
// each pixel's window is derived from the input/output extents with the same
// start/end arithmetic as the NCDHW frame, always covering at least one
// position; the sweep, the NaN/tie predicate and the dense channels-last-3d
// index publish match the fixed-shape frame bit for bit.
template <typename T>
void adaptive_max_pool3d_cl_typed(const T* in, T* out, int64_t* ind,
                                  int64_t N, int64_t C,
                                  int64_t D, int64_t H, int64_t W,
                                  int64_t oD, int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    using I = vec::int_same_size_t<T>;
    using iVec = tensorplay::vec::Vectorized<I>;
    constexpr int64_t V = Vec::size();
    static_assert(iVec::size() == V, "index lanes must follow value lanes");
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    const int64_t tasks = N * oD * oH * oW;
    parallel_for(0, tasks, 1, [&](int64_t begin, int64_t end) {
        std::unique_ptr<I[]> idxstate(new I[C > 0 ? C : 1]);
        int64_t rest = begin;
        int64_t ow = rest % oW; rest /= oW;
        int64_t oh = rest % oH; rest /= oH;
        int64_t od = rest % oD; rest /= oD;
        int64_t n = rest;
        for (int64_t task = begin; task < end; ++task) {
            T* out_lane = out + task * C;
            int64_t* ind_lane = ind + task * C;
            const int64_t ds = od * D / oD;
            const int64_t de = 1 + (((od + 1) * D) - 1) / oD;
            const int64_t hs = oh * H / oH;
            const int64_t he = 1 + (((oh + 1) * H) - 1) / oH;
            const int64_t ws = ow * W / oW;
            const int64_t we = 1 + (((ow + 1) * W) - 1) / oW;
            const iVec minus1(I(-1));
            {
                int64_t c = 0;
                for (; c < len; c += V) {
                    Vec(lo).store(out_lane + c);
                    minus1.store(idxstate.get() + c);
                }
                for (; c < C; ++c) {
                    out_lane[c] = lo;
                    idxstate[c] = I(-1);
                }
            }
            for (int64_t kd = ds; kd < de; ++kd) {
                const T* slice = in + (n * D + kd) * H * W * C;
                for (int64_t kh = hs; kh < he; ++kh) {
                    const T* row = slice + kh * W * C;
                    for (int64_t kw = ws; kw < we; ++kw) {
                        const T* in_lane = row + kw * C;
                        const I off = I((kd * H + kh) * W + kw);
                        const iVec off_vec(off);
                        int64_t c2 = 0;
                        for (; c2 < len; c2 += V) {
                            const Vec maxv = Vec::loadu(out_lane + c2);
                            const Vec val = Vec::loadu(in_lane + c2);
                            // A NaN takes the maximum, like the NCDHW frame;
                            // the same mask drives the value and index blend.
                            const Vec mask = (val != val) | (val > maxv);
                            Vec::blendv(maxv, val, mask).store(out_lane + c2);
                            // All-ones float lanes are all-ones integer bits,
                            // so the mask crosses to the index blend
                            // unchanged.
                            const iVec imask = vec::cast<I>(mask);
                            const iVec cur = iVec::loadu(idxstate.get() + c2);
                            iVec::blendv(cur, off_vec, imask)
                                .store(idxstate.get() + c2);
                        }
                        for (; c2 < C; ++c2) {
                            const T v = in_lane[c2];
                            if ((v != v) || (v > out_lane[c2])) {
                                out_lane[c2] = v;
                                idxstate[c2] = off;
                            }
                        }
                    }
                }
            }
            for (int64_t c = 0; c < C; ++c) {
                ind_lane[c] = int64_t(idxstate[c]);
            }
            ++ow;
            if (ow == oW) {
                ow = 0;
                if (++oh == oH) {
                    oh = 0;
                    if (++od == oD) {
                        od = 0;
                        ++n;
                    }
                }
            }
        }
    });
}

// Scatter twin of the NCDHW adaptive backward: each output pixel re-finds
// its per-lane argmax with the forward sweep and predicate, then adds its
// grad_output lane onto that position of the NDHWC grad_input.  Output
// pixels run in the scalar frame's (D, H, W) lexicographic order, so
// overlapping windows accumulate in the same sequence and gradients match
// the NCDHW frame bit for bit.
template <typename T>
void adaptive_max_pool3d_backward_cl_typed(const T* gout, const T* in, T* gin,
                                           int64_t N, int64_t C,
                                           int64_t D, int64_t H, int64_t W,
                                           int64_t oD, int64_t oH, int64_t oW) {
    using Vec = tensorplay::vec::Vectorized<T>;
    using I = vec::int_same_size_t<T>;
    using iVec = tensorplay::vec::Vectorized<I>;
    constexpr int64_t V = Vec::size();
    static_assert(iVec::size() == V, "index lanes must follow value lanes");
    const int64_t len = C - (C % V);
    const T lo = std::numeric_limits<T>::is_iec559
        ? -std::numeric_limits<T>::infinity()
        : std::numeric_limits<T>::lowest();
    const int64_t in_vol = D * H * W;
    const int64_t out_vol = oD * oH * oW;
    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        T gbuf[V];
        std::unique_ptr<T[]> maxstate(new T[C > 0 ? C : 1]);
        std::unique_ptr<I[]> idxstate(new I[C > 0 ? C : 1]);
        for (int64_t n = begin; n < end; ++n) {
            const T* in_n = in + n * in_vol * C;
            const T* go_n = gout + n * out_vol * C;
            T* gi_n = gin + n * in_vol * C;
            for (int64_t od = 0; od < oD; ++od) {
                const int64_t ds = od * D / oD;
                const int64_t de = 1 + (((od + 1) * D) - 1) / oD;
                for (int64_t oh = 0; oh < oH; ++oh) {
                    const int64_t hs = oh * H / oH;
                    const int64_t he = 1 + (((oh + 1) * H) - 1) / oH;
                    for (int64_t ow = 0; ow < oW; ++ow) {
                        const int64_t ws = ow * W / oW;
                        const int64_t we = 1 + (((ow + 1) * W) - 1) / oW;
                        const int64_t o = (od * oH + oh) * oW + ow;
                        const T* go_lane = go_n + o * C;
                        const iVec minus1(I(-1));
                        {
                            int64_t c = 0;
                            for (; c < len; c += V) {
                                Vec(lo).store(maxstate.get() + c);
                                minus1.store(idxstate.get() + c);
                            }
                            for (; c < C; ++c) {
                                maxstate[c] = lo;
                                idxstate[c] = I(-1);
                            }
                        }
                        for (int64_t kd = ds; kd < de; ++kd) {
                            const T* slice = in_n + kd * H * W * C;
                            for (int64_t kh = hs; kh < he; ++kh) {
                                const T* row = slice + kh * W * C;
                                for (int64_t kw = ws; kw < we; ++kw) {
                                    const T* in_lane = row + kw * C;
                                    const I off = I((kd * H + kh) * W + kw);
                                    const iVec off_vec(off);
                                    int64_t c2 = 0;
                                    for (; c2 < len; c2 += V) {
                                        const Vec maxv =
                                            Vec::loadu(maxstate.get() + c2);
                                        const Vec val =
                                            Vec::loadu(in_lane + c2);
                                        const Vec mask =
                                            (val != val) | (val > maxv);
                                        Vec::blendv(maxv, val, mask)
                                            .store(maxstate.get() + c2);
                                        const iVec imask = vec::cast<I>(mask);
                                        const iVec cur =
                                            iVec::loadu(idxstate.get() + c2);
                                        iVec::blendv(cur, off_vec, imask)
                                            .store(idxstate.get() + c2);
                                    }
                                    for (; c2 < C; ++c2) {
                                        const T v = in_lane[c2];
                                        if ((v != v) || (v > maxstate[c2])) {
                                            maxstate[c2] = v;
                                            idxstate[c2] = off;
                                        }
                                    }
                                }
                            }
                        }
                        int64_t c2 = 0;
                        for (; c2 < len; c2 += V) {
                            Vec::loadu(go_lane + c2).store(gbuf);
                            for (int64_t l = 0; l < V; ++l) {
                                const int64_t max_idx =
                                    int64_t(idxstate[c2 + l]);
                                if (max_idx >= 0) {
                                    gi_n[max_idx * C + c2 + l] += gbuf[l];
                                }
                            }
                        }
                        for (; c2 < C; ++c2) {
                            const int64_t max_idx = int64_t(idxstate[c2]);
                            if (max_idx >= 0) {
                                gi_n[max_idx * C + c2] += go_lane[c2];
                            }
                        }
                    }
                }
            }
        }
    });
}

void adaptive_max_pool3d_cl_impl(const void* in, void* out, int64_t* ind,
                                 int64_t N, int64_t C,
                                 int64_t D, int64_t H, int64_t W,
                                 int64_t oD, int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            adaptive_max_pool3d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out), ind,
                N, C, D, H, W, oD, oH, oW);
            break;
        case DType::Float64:
            adaptive_max_pool3d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out), ind,
                N, C, D, H, W, oD, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool3d: channels-last kernel supports only "
                     "float and double");
    }
}

void adaptive_max_pool3d_backward_cl_impl(const void* gout, const void* in,
                                          void* gin,
                                          int64_t N, int64_t C,
                                          int64_t D, int64_t H, int64_t W,
                                          int64_t oD, int64_t oH, int64_t oW,
                                          int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            adaptive_max_pool3d_backward_cl_typed<float>(
                static_cast<const float*>(gout), static_cast<const float*>(in),
                static_cast<float*>(gin), N, C, D, H, W, oD, oH, oW);
            break;
        case DType::Float64:
            adaptive_max_pool3d_backward_cl_typed<double>(
                static_cast<const double*>(gout),
                static_cast<const double*>(in), static_cast<double*>(gin),
                N, C, D, H, W, oD, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool3d_backward: channels-last kernel "
                     "supports only float and double");
    }
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

void max_pool2d_backward_cl_impl(const void* gout, const int64_t* ind, void* gin,
                                 int64_t N, int64_t C,
                                 int64_t H, int64_t W,
                                 int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            max_pool2d_backward_cl_typed<float>(
                static_cast<const float*>(gout), ind, static_cast<float*>(gin),
                N, C, H, W, oH, oW);
            break;
        case DType::Float64:
            max_pool2d_backward_cl_typed<double>(
                static_cast<const double*>(gout), ind,
                static_cast<double*>(gin), N, C, H, W, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "max_pool2d_backward: channels-last kernel supports only "
                     "float and double");
    }
}

void max_pool3d_cl_impl(const void* in, void* out, int64_t* ind,
                        int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                        int64_t oD, int64_t oH, int64_t oW,
                        int64_t kD, int64_t kH, int64_t kW,
                        int64_t sD, int64_t sH, int64_t sW,
                        int64_t pD, int64_t pH, int64_t pW,
                        int64_t dD, int64_t dH, int64_t dW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            max_pool3d_cl_typed<float>(
                static_cast<const float*>(in), static_cast<float*>(out), ind,
                N, C, D, H, W, oD, oH, oW, kD, kH, kW, sD, sH, sW,
                pD, pH, pW, dD, dH, dW);
            break;
        case DType::Float64:
            max_pool3d_cl_typed<double>(
                static_cast<const double*>(in), static_cast<double*>(out), ind,
                N, C, D, H, W, oD, oH, oW, kD, kH, kW, sD, sH, sW,
                pD, pH, pW, dD, dH, dW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d: channels-last kernel supports only float "
                     "and double");
    }
}

void max_pool3d_backward_cl_impl(const void* gout, const int64_t* ind, void* gin,
                                 int64_t N, int64_t C,
                                 int64_t D, int64_t H, int64_t W,
                                 int64_t oD, int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            max_pool3d_backward_cl_typed<float>(
                static_cast<const float*>(gout), ind, static_cast<float*>(gin),
                N, C, D, H, W, oD, oH, oW);
            break;
        case DType::Float64:
            max_pool3d_backward_cl_typed<double>(
                static_cast<const double*>(gout), ind, static_cast<double*>(gin),
                N, C, D, H, W, oD, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d_backward: channels-last kernel supports only "
                     "float and double");
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

void adaptive_max_pool2d_backward_cl_impl(const void* gout, const int64_t* ind,
                                          void* gin,
                                          int64_t N, int64_t C,
                                          int64_t H, int64_t W,
                                          int64_t oH, int64_t oW, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            max_pool2d_backward_cl_typed<float>(
                static_cast<const float*>(gout), ind, static_cast<float*>(gin),
                N, C, H, W, oH, oW);
            break;
        case DType::Float64:
            max_pool2d_backward_cl_typed<double>(
                static_cast<const double*>(gout), ind,
                static_cast<double*>(gin), N, C, H, W, oH, oW);
            break;
        default:
            TP_THROW(NotImplementedError,
                     "adaptive_max_pool2d_backward: channels-last kernel "
                     "supports only float and double");
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
REGISTER_DISPATCH(max_pool2d_backward_cl_stub, &max_pool2d_backward_cl_impl);
REGISTER_DISPATCH(max_pool3d_cl_stub, &max_pool3d_cl_impl);
REGISTER_DISPATCH(max_pool3d_backward_cl_stub, &max_pool3d_backward_cl_impl);
REGISTER_DISPATCH(adaptive_avg_pool2d_cl_stub, &adaptive_avg_pool2d_cl_impl);
REGISTER_DISPATCH(adaptive_max_pool2d_cl_stub, &adaptive_max_pool2d_cl_impl);
REGISTER_DISPATCH(adaptive_max_pool2d_backward_cl_stub,
                  &adaptive_max_pool2d_backward_cl_impl);
REGISTER_DISPATCH(adaptive_max_pool3d_cl_stub, &adaptive_max_pool3d_cl_impl);
REGISTER_DISPATCH(adaptive_max_pool3d_backward_cl_stub,
                  &adaptive_max_pool3d_backward_cl_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(avg_pool2d_cl_stub, &avg_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(max_pool2d_cl_stub, &max_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(max_pool2d_backward_cl_stub,
                              &max_pool2d_backward_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(max_pool3d_cl_stub, &max_pool3d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(max_pool3d_backward_cl_stub,
                              &max_pool3d_backward_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_avg_pool2d_cl_stub,
                              &adaptive_avg_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_max_pool2d_cl_stub,
                              &adaptive_max_pool2d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_max_pool2d_backward_cl_stub,
                              &adaptive_max_pool2d_backward_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_max_pool3d_cl_stub,
                              &adaptive_max_pool3d_cl_impl);
ALSO_REGISTER_AVX512_DISPATCH(adaptive_max_pool3d_backward_cl_stub,
                              &adaptive_max_pool3d_backward_cl_impl);
#endif

} // namespace cpu
} // namespace tensorplay
