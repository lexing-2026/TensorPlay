// Shared SIMD row helpers for the normalization kernels (layer/group/instance
// norm and batch norm).  Every helper is hand-coded with a per-function target
// attribute so the translation unit can compile for the baseline ISA while the
// helper body uses AVX-512; call sites gate on avx512_ok() at runtime and fall
// back to compiler-auto-vectorized scalar loops.
#pragma once

#include <cstdint>
#include <cmath>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
namespace norm_row {

#if defined(__x86_64__)

inline bool avx512_ok() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

// ---------------------------------------------------------------------------
// Layer/group-norm primitives: per-row mean+rstd stats and the affine
// normalize pass, with optional per-element weight/bias.
// ---------------------------------------------------------------------------

template <bool HW, bool HB>
__attribute__((target("avx512f")))
inline void apply_f32_512(const float* in, float* out, int64_t n,
                          float mean, float rstd, const float* w, const float* b) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        v = _mm512_mul_ps(_mm512_sub_ps(v, vm), vr);
        if constexpr (HW) v = _mm512_mul_ps(v, _mm512_loadu_ps(w + i));
        if constexpr (HB) v = _mm512_add_ps(v, _mm512_loadu_ps(b + i));
        _mm512_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        float t = (in[i] - mean) * rstd;
        if constexpr (HW) t *= w[i];
        if constexpr (HB) t += b[i];
        out[i] = t;
    }
}

template <bool HW, bool HB>
__attribute__((target("avx512f")))
inline void apply_group_f32_512(const float* in, float* out,
                                int64_t channels, int64_t spatial,
                                float mean, float rstd,
                                const float* w, const float* b) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    for (int64_t c = 0; c < channels; ++c) {
        const float wc = HW ? w[c] : 1.0f;
        const float bc = HB ? b[c] : 0.0f;
        const __m512 vw = _mm512_set1_ps(wc);
        const __m512 vb = _mm512_set1_ps(bc);
        const float* ip = in + c * spatial;
        float* op = out + c * spatial;
        int64_t s = 0;
        for (; s + 16 <= spatial; s += 16) {
            __m512 v = _mm512_loadu_ps(ip + s);
            v = _mm512_mul_ps(_mm512_sub_ps(v, vm), vr);
            if constexpr (HW) v = _mm512_mul_ps(v, vw);
            if constexpr (HB) v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(op + s, v);
        }
        for (; s < spatial; ++s) {
            float v = (ip[s] - mean) * rstd;
            if constexpr (HW) v *= wc;
            if constexpr (HB) v += bc;
            op[s] = v;
        }
    }
}

__attribute__((target("avx512f")))
inline void stats_f32_512(const float* x, int64_t n, float eps,
                          float* mean_out, float* rstd_out) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 q0 = _mm512_setzero_ps(), q1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 v0 = _mm512_loadu_ps(x + i);
        __m512 v1 = _mm512_loadu_ps(x + i + 16);
        s0 = _mm512_add_ps(s0, v0);
        s1 = _mm512_add_ps(s1, v1);
        q0 = _mm512_add_ps(q0, _mm512_mul_ps(v0, v0));
        q1 = _mm512_add_ps(q1, _mm512_mul_ps(v1, v1));
    }
    __m512 s = _mm512_add_ps(s0, s1);
    __m512 q = _mm512_add_ps(q0, q1);
    alignas(64) float sb[16], qb[16];
    _mm512_storeu_ps(sb, s);
    _mm512_storeu_ps(qb, q);
    float sum = 0.f, sq = 0.f;
    for (int64_t k = 0; k < 16; ++k) { sum += sb[k]; sq += qb[k]; }
    for (; i < n; ++i) { float v = x[i]; sum += v; sq += v * v; }
    float mean = sum / static_cast<float>(n);
    float var = sq / static_cast<float>(n) - mean * mean;
    *mean_out = mean;
    *rstd_out = 1.0f / std::sqrt(var + eps);
}

template <bool HW, bool HB>
__attribute__((target("avx512f")))
inline void apply_f64_512(const double* in, double* out, int64_t n,
                          double mean, double rstd, const double* w, const double* b) {
    const __m512d vm = _mm512_set1_pd(mean);
    const __m512d vr = _mm512_set1_pd(rstd);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(in + i);
        v = _mm512_mul_pd(_mm512_sub_pd(v, vm), vr);
        if constexpr (HW) v = _mm512_mul_pd(v, _mm512_loadu_pd(w + i));
        if constexpr (HB) v = _mm512_add_pd(v, _mm512_loadu_pd(b + i));
        _mm512_storeu_pd(out + i, v);
    }
    for (; i < n; ++i) {
        double t = (in[i] - mean) * rstd;
        if constexpr (HW) t *= w[i];
        if constexpr (HB) t += b[i];
        out[i] = t;
    }
}

__attribute__((target("avx512f")))
inline void stats_f64_512(const double* x, int64_t n, double eps,
                          double* mean_out, double* rstd_out) {
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    __m512d q0 = _mm512_setzero_pd(), q1 = _mm512_setzero_pd();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512d v0 = _mm512_loadu_pd(x + i);
        __m512d v1 = _mm512_loadu_pd(x + i + 8);
        s0 = _mm512_add_pd(s0, v0);
        s1 = _mm512_add_pd(s1, v1);
        q0 = _mm512_add_pd(q0, _mm512_mul_pd(v0, v0));
        q1 = _mm512_add_pd(q1, _mm512_mul_pd(v1, v1));
    }
    __m512d s = _mm512_add_pd(s0, s1);
    __m512d q = _mm512_add_pd(q0, q1);
    alignas(64) double sb[8], qb[8];
    _mm512_storeu_pd(sb, s);
    _mm512_storeu_pd(qb, q);
    double sum = 0.0, sq = 0.0;
    for (int64_t k = 0; k < 8; ++k) { sum += sb[k]; sq += qb[k]; }
    for (; i < n; ++i) { double v = x[i]; sum += v; sq += v * v; }
    double mean = sum / static_cast<double>(n);
    double var = sq / static_cast<double>(n) - mean * mean;
    *mean_out = mean;
    *rstd_out = 1.0 / std::sqrt(var + eps);
}

// ---------------------------------------------------------------------------
// Batch-norm primitives.  The per-plane output is an affine map of the input
// with plane-constant scale/bias: y = x * alpha + beta.
// ---------------------------------------------------------------------------

// Float32 in / Float32 out affine plane.
__attribute__((target("avx512f")))
inline void plane_affine_f32_512(const float* in, float* out, int64_t n, float a, float b) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(v, va, vb));
    }
    for (; i < n; ++i) out[i] = in[i] * a + b;
}

// Float32 data with double-precision accumulators: sum and sum of squares.
__attribute__((target("avx512f")))
inline void acc_stats_f64_512(const float* p, int64_t n, double& s, double& q) {
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    __m512d q0 = _mm512_setzero_pd(), q1 = _mm512_setzero_pd();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512d v0 = _mm512_cvtps_pd(_mm256_loadu_ps(p + i));
        __m512d v1 = _mm512_cvtps_pd(_mm256_loadu_ps(p + i + 8));
        s0 = _mm512_add_pd(s0, v0);
        q0 = _mm512_fmadd_pd(v0, v0, q0);
        s1 = _mm512_add_pd(s1, v1);
        q1 = _mm512_fmadd_pd(v1, v1, q1);
    }
    s0 = _mm512_add_pd(s0, s1);
    q0 = _mm512_add_pd(q0, q1);
    alignas(64) double sb[8], qb[8];
    _mm512_storeu_pd(sb, s0);
    _mm512_storeu_pd(qb, q0);
    for (int64_t k = 0; k < 8; ++k) { s += sb[k]; q += qb[k]; }
    for (; i < n; ++i) { double v = static_cast<double>(p[i]); s += v; q += v * v; }
}

// Float32 data with double accumulators: sum(dy) and sum(dy * x) in one pass.
__attribute__((target("avx512f")))
inline void acc_dot2_f64_512(const float* dy, const float* x, int64_t n,
                             double& s, double& d) {
    __m512d s0 = _mm512_setzero_pd(), d0 = _mm512_setzero_pd();
    __m512d s1 = _mm512_setzero_pd(), d1 = _mm512_setzero_pd();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512d y0 = _mm512_cvtps_pd(_mm256_loadu_ps(dy + i));
        __m512d y1 = _mm512_cvtps_pd(_mm256_loadu_ps(dy + i + 8));
        __m512d x0 = _mm512_cvtps_pd(_mm256_loadu_ps(x + i));
        __m512d x1 = _mm512_cvtps_pd(_mm256_loadu_ps(x + i + 8));
        s0 = _mm512_add_pd(s0, y0);
        d0 = _mm512_fmadd_pd(y0, x0, d0);
        s1 = _mm512_add_pd(s1, y1);
        d1 = _mm512_fmadd_pd(y1, x1, d1);
    }
    s0 = _mm512_add_pd(s0, s1);
    d0 = _mm512_add_pd(d0, d1);
    alignas(64) double sb[8], db[8];
    _mm512_storeu_pd(sb, s0);
    _mm512_storeu_pd(db, d0);
    for (int64_t k = 0; k < 8; ++k) { s += sb[k]; d += db[k]; }
    for (; i < n; ++i) {
        double y = static_cast<double>(dy[i]);
        s += y;
        d += y * static_cast<double>(x[i]);
    }
}

// Training-mode grad_input for one plane:
//   dx = (dy - grad_mean - (x - mean) * k) * rstd * w
// where k = dotp * rstd^2 / M, grad_mean = sum(dy) / M (both plane-constant).
__attribute__((target("avx512f")))
inline void plane_bn_dx_f32_512(const float* x, const float* dy, float* out, int64_t n,
                                float mean, float k, float grad_mean, float rstd, float w) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vk = _mm512_set1_ps(k);
    const __m512 vg = _mm512_set1_ps(grad_mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    const __m512 vw = _mm512_set1_ps(w);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vdy = _mm512_loadu_ps(dy + i);
        __m512 t = _mm512_mul_ps(_mm512_sub_ps(vx, vm), vk);
        __m512 u = _mm512_sub_ps(_mm512_sub_ps(vdy, vg), t);
        __m512 r = _mm512_mul_ps(u, vr);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(r, vw));
    }
    for (; i < n; ++i) {
        out[i] = (dy[i] - grad_mean - (x[i] - mean) * k) * rstd * w;
    }
}

// Eval-mode grad_input for one plane: dx = dy * scale.
__attribute__((target("avx512f")))
inline void plane_scale_f32_512(const float* dy, float* out, int64_t n, float scale) {
    const __m512 vs = _mm512_set1_ps(scale);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(dy + i), vs));
    }
    for (; i < n; ++i) out[i] = dy[i] * scale;
}

// ---------------------------------------------------------------------------
// Layer-norm backward primitives (float).  Per row:
//   s_dy      = sum(dy * w)
//   s_dy_xhat = sum(dy * w * x_hat),  x_hat = (x - mean) * rstd
// and
//   grad = term1 * (M * dy * w - s_dy - x_hat * s_dy_xhat).
// ---------------------------------------------------------------------------

template <bool HW>
__attribute__((target("avx512f")))
inline void ln_bwd_stats_f32_512(const float* dy, const float* x, const float* w,
                                 int64_t n, float mean, float rstd,
                                 float* s_dy, float* s_dy_xhat) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 y0 = _mm512_loadu_ps(dy + i);
        __m512 y1 = _mm512_loadu_ps(dy + i + 16);
        if constexpr (HW) {
            y0 = _mm512_mul_ps(y0, _mm512_loadu_ps(w + i));
            y1 = _mm512_mul_ps(y1, _mm512_loadu_ps(w + i + 16));
        }
        __m512 x0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i), vm), vr);
        __m512 x1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i + 16), vm), vr);
        a0 = _mm512_add_ps(a0, y0);
        a1 = _mm512_add_ps(a1, y1);
        b0 = _mm512_fmadd_ps(y0, x0, b0);
        b1 = _mm512_fmadd_ps(y1, x1, b1);
    }
    __m512 a = _mm512_add_ps(a0, a1);
    __m512 b = _mm512_add_ps(b0, b1);
    alignas(64) float ab[16], bb[16];
    _mm512_storeu_ps(ab, a);
    _mm512_storeu_ps(bb, b);
    float sa = 0.f, sb = 0.f;
    for (int64_t k = 0; k < 16; ++k) { sa += ab[k]; sb += bb[k]; }
    for (; i < n; ++i) {
        float y = dy[i];
        if constexpr (HW) y *= w[i];
        float xh = (x[i] - mean) * rstd;
        sa += y;
        sb += y * xh;
    }
    *s_dy = sa;
    *s_dy_xhat = sb;
}

template <bool HW>
__attribute__((target("avx512f")))
inline void ln_bwd_apply_f32_512(const float* dy, const float* x, const float* w,
                                 float* out, int64_t n, float mean, float rstd,
                                 float term1, float M, float s_dy, float s_dy_xhat) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    const __m512 vt1 = _mm512_set1_ps(term1);
    const __m512 vmM = _mm512_set1_ps(M);
    const __m512 vsd = _mm512_set1_ps(s_dy);
    const __m512 vsx = _mm512_set1_ps(s_dy_xhat);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vdy = _mm512_loadu_ps(dy + i);
        if constexpr (HW) vdy = _mm512_mul_ps(vdy, _mm512_loadu_ps(w + i));
        __m512 xh = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i), vm), vr);
        __m512 t = _mm512_mul_ps(vdy, vmM);
        t = _mm512_sub_ps(t, vsd);
        t = _mm512_fnmadd_ps(xh, vsx, t);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(t, vt1));
    }
    for (; i < n; ++i) {
        float y = dy[i];
        if constexpr (HW) y *= w[i];
        float xh = (x[i] - mean) * rstd;
        out[i] = term1 * (M * y - s_dy - xh * s_dy_xhat);
    }
}

// Group-norm backward: one channel plane of a group row.
//   grad = term1 * (M * dy * wc - s_dy - x_hat * s_dy_xhat)
template <bool UseW>
__attribute__((target("avx512f")))
inline void gn_bwd_plane_f32_512(const float* x, const float* dy, float* out,
                                 int64_t spatial, float mean, float rstd, float wc,
                                 float term1, float M, float s_dy, float s_dy_xhat) {
    const __m512 vm = _mm512_set1_ps(mean);
    const __m512 vr = _mm512_set1_ps(rstd);
    const __m512 vw = _mm512_set1_ps(wc);
    const __m512 vt1 = _mm512_set1_ps(term1);
    const __m512 vmM = _mm512_set1_ps(M);
    const __m512 vsd = _mm512_set1_ps(s_dy);
    const __m512 vsx = _mm512_set1_ps(s_dy_xhat);
    int64_t s = 0;
    for (; s + 16 <= spatial; s += 16) {
        __m512 vdy = _mm512_loadu_ps(dy + s);
        if constexpr (UseW) vdy = _mm512_mul_ps(vdy, vw);
        __m512 xh = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + s), vm), vr);
        __m512 t = _mm512_mul_ps(vdy, vmM);
        t = _mm512_sub_ps(t, vsd);
        t = _mm512_fnmadd_ps(xh, vsx, t);
        _mm512_storeu_ps(out + s, _mm512_mul_ps(t, vt1));
    }
    for (; s < spatial; ++s) {
        float y = dy[s];
        if constexpr (UseW) y *= wc;
        float xh = (x[s] - mean) * rstd;
        out[s] = term1 * (M * y - s_dy - xh * s_dy_xhat);
    }
}

#endif  // __x86_64__

}  // namespace norm_row
}  // namespace cpu
}  // namespace tensorplay
