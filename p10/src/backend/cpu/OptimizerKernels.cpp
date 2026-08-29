#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TP_OPT_X86_SIMD 1
#endif

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

// ------------------------------------------------------------------
// Horizontal batch scheduling + runtime-dispatched SIMD inner loops.
//
// The foreach/fused optimizer entry points hand us an entire parameter
// group at once, so every tensor is flattened into one chunked work
// list and that list is scheduled exactly once.  Scheduling per tensor
// (a per-tensor parallel_for) leaves many-small-parameter workloads --
// transformer-like groups of 100+ (128,128) tensors -- serialized
// behind a single worker with one barrier per tensor.  Inner chunk
// loops are compiled in AVX2/AVX512 `#pragma GCC target` shims and
// selected at runtime via __builtin_cpu_supports, so no build-flag
// changes are required and non-x86 builds fall back to plain scalar.
// ------------------------------------------------------------------

struct OptChunk {
    int64_t list_index;
    int64_t begin;
    int64_t end;
};

std::vector<OptChunk> build_opt_work_list(const int64_t* numels, size_t count) {
    std::vector<OptChunk> work;
    work.reserve(count * 2);
    for (size_t i = 0; i < count; ++i) {
        for (int64_t b = 0; b < numels[i]; b += GRAIN_SIZE) {
            work.push_back({static_cast<int64_t>(i), b,
                            std::min<int64_t>(b + GRAIN_SIZE, numels[i])});
        }
    }
    return work;
}

#ifdef TP_OPT_X86_SIMD
bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") &&
                           __builtin_cpu_supports("fma");
    return ok;
}
bool have_avx512f() {
    static const bool ok = __builtin_cpu_supports("avx512f");
    return ok;
}
#endif

// float and write the result back to the tensor dtype after every op.  The
// native optimizer loops below are intentionally scalar for Half/BFloat16,
// so keep that observable cast point in one helper.  For float/double this is
// a no-op and leaves the existing vector fast paths unchanged.
template <typename scalar_t, typename math_t>
inline math_t optimizer_round(math_t value) {
    if constexpr (std::is_same_v<scalar_t, math_t>) {
        return value;
    } else {
        return static_cast<math_t>(static_cast<scalar_t>(value));
    }
}

template <typename math_t>
inline math_t optimizer_sqrt(math_t value) {
    return static_cast<math_t>(std::sqrt(value));
}

template <typename scalar_t, typename math_t>
inline math_t optimizer_lerp(math_t self, math_t end, math_t weight) {
    const math_t result = std::abs(weight) < math_t(0.5)
        ? self + weight * (end - self)
        : end - (end - self) * (math_t(1) - weight);
    return optimizer_round<scalar_t, math_t>(result);
}

template <typename scalar_t, typename math_t>
inline math_t optimizer_addcmul(
        math_t input, math_t tensor1, math_t tensor2, math_t value) {
    return optimizer_round<scalar_t, math_t>(
        input + value * tensor1 * tensor2);
}

template <typename scalar_t, typename math_t>
inline math_t optimizer_addcdiv(
        math_t input, math_t tensor1, math_t tensor2, math_t value) {
    return optimizer_round<scalar_t, math_t>(
        input + value * tensor1 / tensor2);
}

template <typename scalar_t, typename math_t>
inline math_t optimizer_add(math_t input, math_t value, math_t alpha) {
    return optimizer_round<scalar_t, math_t>(input + alpha * value);
}

// Portable scalar fallbacks; also used for vector-loop tails.
template <typename T>
inline void adam_range_scalar(const T* grad, T* param, T* m, T* v, T* maxv,
                              bool amsgrad, bool wd, const T sc[8],
                              int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        T g = grad[i];
        if (wd) g += sc[7] * param[i];
        T mv = sc[0] * m[i] + sc[1] * g;
        m[i] = mv;
        T vv = sc[2] * v[i] + sc[3] * g * g;
        v[i] = vv;
        T s = vv;
        if (amsgrad) {
            s = maxv[i] < vv ? vv : maxv[i];
            maxv[i] = s;
        }
        const T denom = std::sqrt(s) / sc[5] + sc[6];
        param[i] -= sc[4] * mv / denom;
    }
}

template <typename T>
inline void sgd_range_scalar(const T* grad, T* param, T* buf, bool has_buf,
                             bool first_step, bool nesterov, bool wd,
                             const T sc[4], int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        T g = grad[i];
        if (wd) g += sc[3] * param[i];
        if (has_buf) {
            T b = first_step ? g : sc[1] * buf[i] + sc[2] * g;
            buf[i] = b;
            g = nesterov ? g + sc[1] * b : b;
        }
        param[i] -= sc[0] * g;
    }
}

// Fused-SGD scalar range.  sc holds {lr, momentum, 1-dampening,
template <typename T>
inline void fused_sgd_range_scalar(T* grad, T* param, T* buf, bool has_buf,
                                   bool first_step, bool nesterov,
                                   bool has_scale, bool maximize, bool wd,
                                   const T sc[5], int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        T g = grad[i];
        if (has_scale) { g = g / sc[4]; grad[i] = g; }
        if (maximize) g = -g;
        T p = param[i];
        if (wd) g += sc[3] * p;
        if (has_buf) {
            T b = first_step ? g : sc[1] * buf[i] + sc[2] * g;
            buf[i] = b;
            g = nesterov ? g + sc[1] * b : b;
        }
        param[i] = p - sc[0] * g;
    }
}

// Fused-Adam scalar range.  sc holds {beta1, lerp_weight(1-beta1), beta2,
// 1-beta2, step_size, correction2_sqrt, eps, decay, wd_factor, grad_scale};
template <typename T>
inline void fused_adam_range_scalar(T* grad, T* param, T* m, T* v, T* maxv,
                                    bool amsgrad, bool adamw, bool coupled_wd,
                                    bool has_scale, bool maximize,
                                    const T sc[10], int64_t begin,
                                    int64_t end) {
    const bool small_lerp = std::abs(static_cast<double>(sc[1])) < 0.5;
    for (int64_t i = begin; i < end; ++i) {
        T g = grad[i];
        if (has_scale) { g = g / sc[9]; grad[i] = g; }
        if (maximize) g = -g;
        T p = param[i];
        if (adamw) {
            p = p * sc[8];
        } else if (coupled_wd) {
            g = g + sc[7] * p;
        }
        const T old_m = m[i];
        T mv;
        if (small_lerp) {
            mv = old_m + sc[1] * (g - old_m);
        } else {
            mv = g - (g - old_m) * (static_cast<T>(1) - sc[1]);
        }
        m[i] = mv;
        const T vv = sc[2] * v[i] + sc[3] * g * g;
        v[i] = vv;
        T s = vv;
        if (amsgrad) {
            s = maxv[i] < vv ? vv : maxv[i];
            maxv[i] = s;
        }
        const T denom = std::sqrt(s) / sc[5] + sc[6];
        param[i] = p - sc[4] * mv / denom;
    }
}

// Fused-Adagrad scalar range.  sc holds {corrected_lr, eps, weight_decay,
// keeps the scalar helper for tails and non-x86 dtypes.
template <typename T, typename M = T>
inline void fused_adagrad_range_scalar(T* grad, T* param, T* state_sum,
                                       bool has_scale, bool maximize, bool wd,
                                       const M sc[4], int64_t begin,
                                       int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        M g = static_cast<M>(grad[i]);
        if (has_scale) {
            g = optimizer_round<T, M>(g / sc[3]);
            grad[i] = static_cast<T>(g);
        }
        if (maximize) g = optimizer_round<T, M>(-g);
        M p = static_cast<M>(param[i]);
        if (wd) g = optimizer_round<T, M>(g + sc[2] * p);
        const M sum = optimizer_round<T, M>(
            static_cast<M>(state_sum[i]) + g * g);
        state_sum[i] = static_cast<T>(sum);
        M denom = optimizer_round<T, M>(optimizer_sqrt(sum));
        denom = optimizer_round<T, M>(denom + sc[1]);
        const M numerator = optimizer_round<T, M>(-sc[0] * g);
        param[i] = static_cast<T>(optimizer_round<T, M>(
            p + numerator / denom));
    }
}

template <typename scalar_t, typename math_t>
inline void fused_nadam_range_scalar(
        const scalar_t* grad, scalar_t* param, scalar_t* exp_avg,
        scalar_t* exp_avg_sq, bool wd, bool decoupled_wd, bool maximize,
        const math_t sc[10], int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        math_t g = static_cast<math_t>(grad[i]);
        if (maximize) g = optimizer_round<scalar_t, math_t>(-g);
        math_t p = static_cast<math_t>(param[i]);
        if (wd) {
            if (decoupled_wd) {
                p = optimizer_round<scalar_t, math_t>(
                    p * (math_t(1) - sc[0] * sc[4]));
            } else {
                g = optimizer_round<scalar_t, math_t>(g + sc[4] * p);
            }
        }

        const math_t old_m = static_cast<math_t>(exp_avg[i]);
        const math_t m = optimizer_lerp<scalar_t, math_t>(
            old_m, g, sc[5]);
        exp_avg[i] = static_cast<scalar_t>(m);
        math_t v = optimizer_round<scalar_t, math_t>(
            sc[2] * static_cast<math_t>(exp_avg_sq[i]));
        v = optimizer_addcmul<scalar_t, math_t>(v, g, g, sc[6]);
        exp_avg_sq[i] = static_cast<scalar_t>(v);
        math_t denom = optimizer_round<scalar_t, math_t>(
            optimizer_sqrt(v));
        denom = optimizer_round<scalar_t, math_t>(denom / sc[9]);
        denom = optimizer_round<scalar_t, math_t>(denom + sc[3]);
        p = optimizer_addcdiv<scalar_t, math_t>(p, g, denom, sc[7]);
        p = optimizer_addcdiv<scalar_t, math_t>(p, m, denom, sc[8]);
        param[i] = static_cast<scalar_t>(p);
    }
}

template <typename scalar_t, typename math_t>
inline void fused_rmsprop_range_scalar(
        const scalar_t* grad, scalar_t* param, scalar_t* square_avg,
        scalar_t* grad_avg, scalar_t* momentum_buffer, bool centered,
        bool has_momentum, bool wd, bool maximize, const math_t sc[6],
        int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        math_t g = static_cast<math_t>(grad[i]);
        if (maximize) g = optimizer_round<scalar_t, math_t>(-g);
        math_t p = static_cast<math_t>(param[i]);
        if (wd) g = optimizer_round<scalar_t, math_t>(g + sc[4] * p);

        math_t square = optimizer_round<scalar_t, math_t>(
            sc[0] * static_cast<math_t>(square_avg[i]));
        square = optimizer_addcmul<scalar_t, math_t>(square, g, g, sc[1]);
        square_avg[i] = static_cast<scalar_t>(square);
        math_t avg = square;
        if (centered) {
            math_t mean = optimizer_lerp<scalar_t, math_t>(
                static_cast<math_t>(grad_avg[i]), g, sc[1]);
            grad_avg[i] = static_cast<scalar_t>(mean);
            avg = optimizer_addcmul<scalar_t, math_t>(
                square, mean, mean, math_t(-1));
        }
        avg = optimizer_round<scalar_t, math_t>(optimizer_sqrt(avg));
        avg = optimizer_round<scalar_t, math_t>(avg + sc[3]);
        if (has_momentum) {
            math_t buffer = optimizer_round<scalar_t, math_t>(
                sc[5] * static_cast<math_t>(momentum_buffer[i]));
            buffer = optimizer_addcdiv<scalar_t, math_t>(
                buffer, g, avg, math_t(1));
            momentum_buffer[i] = static_cast<scalar_t>(buffer);
            p = optimizer_add<scalar_t, math_t>(p, buffer, -sc[2]);
        } else {
            p = optimizer_addcdiv<scalar_t, math_t>(
                p, g, avg, -sc[2]);
        }
        param[i] = static_cast<scalar_t>(p);
    }
}

template <typename scalar_t, typename math_t>
inline void fused_adadelta_range_scalar(
        const scalar_t* grad, scalar_t* param, scalar_t* square_avg,
        scalar_t* acc_delta, bool wd, bool maximize, const math_t sc[5],
        int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        math_t g = static_cast<math_t>(grad[i]);
        if (maximize) g = optimizer_round<scalar_t, math_t>(-g);
        math_t p = static_cast<math_t>(param[i]);
        if (wd) g = optimizer_round<scalar_t, math_t>(g + sc[4] * p);

        math_t square = optimizer_round<scalar_t, math_t>(
            sc[0] * static_cast<math_t>(square_avg[i]));
        square = optimizer_addcmul<scalar_t, math_t>(square, g, g, sc[1]);
        square_avg[i] = static_cast<scalar_t>(square);
        math_t std = optimizer_round<scalar_t, math_t>(square + sc[3]);
        std = optimizer_round<scalar_t, math_t>(optimizer_sqrt(std));
        math_t delta = optimizer_round<scalar_t, math_t>(
            static_cast<math_t>(acc_delta[i]) + sc[3]);
        delta = optimizer_round<scalar_t, math_t>(optimizer_sqrt(delta));
        delta = optimizer_round<scalar_t, math_t>(delta / std);
        delta = optimizer_round<scalar_t, math_t>(delta * g);
        math_t next_acc = optimizer_round<scalar_t, math_t>(
            sc[0] * static_cast<math_t>(acc_delta[i]));
        next_acc = optimizer_addcmul<scalar_t, math_t>(
            next_acc, delta, delta, sc[1]);
        acc_delta[i] = static_cast<scalar_t>(next_acc);
        param[i] = static_cast<scalar_t>(optimizer_add<scalar_t, math_t>(
            p, delta, -sc[2]));
    }
}

#ifdef TP_OPT_X86_SIMD

// Vector chunk kernels are generated by macro into `#pragma GCC target`
// regions so every function touching vector registers is compiled with its
// ISA flags -- no cross-region template instantiation, which GCC resolves
// at end-of-TU with default flags and corrupts the ABI.

#define TP_OPT_ADAM_VEC(FN, T, VTYPE, W, SUF, VECW)                                 \
static void FN(const T* grad, T* param, T* m, T* v, T* maxv,                 \
               bool amsgrad, bool wd, const T sc[8],                         \
               int64_t begin, int64_t end) {                                 \
    const VTYPE vb1 = _mm##W##_set1_##SUF(sc[0]);                           \
    const VTYPE vomb1 = _mm##W##_set1_##SUF(sc[1]);                         \
    const VTYPE vb2 = _mm##W##_set1_##SUF(sc[2]);                           \
    const VTYPE vomb2 = _mm##W##_set1_##SUF(sc[3]);                         \
    const VTYPE vstep = _mm##W##_set1_##SUF(sc[4]);                         \
    const VTYPE vc2s = _mm##W##_set1_##SUF(sc[5]);                          \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[6]);                          \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[7]);                        \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                     \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        VTYPE mv = _mm##W##_loadu_##SUF(m + i);                             \
        mv = _mm##W##_fmadd_##SUF(vb1, mv, _mm##W##_mul_##SUF(vomb1, g));    \
        _mm##W##_storeu_##SUF(m + i, mv);                                    \
        VTYPE vv = _mm##W##_loadu_##SUF(v + i);                             \
        vv = _mm##W##_fmadd_##SUF(vb2, vv,                                   \
                                  _mm##W##_mul_##SUF(vomb2,                  \
                                                     _mm##W##_mul_##SUF(g, g))); \
        _mm##W##_storeu_##SUF(v + i, vv);                                    \
        VTYPE s = vv;                                                       \
        if (amsgrad) {                                                       \
            VTYPE mx = _mm##W##_max_##SUF(_mm##W##_loadu_##SUF(maxv + i),   \
                                           vv);                              \
            _mm##W##_storeu_##SUF(maxv + i, mx);                             \
            s = mx;                                                          \
        }                                                                    \
        const VTYPE denom = _mm##W##_add_##SUF(                             \
            _mm##W##_div_##SUF(_mm##W##_sqrt_##SUF(s), vc2s), veps);         \
        const VTYPE upd =                                                   \
            _mm##W##_div_##SUF(_mm##W##_mul_##SUF(vstep, mv), denom);        \
        _mm##W##_storeu_##SUF(param + i, _mm##W##_sub_##SUF(p, upd));        \
    }                                                                        \
    adam_range_scalar(grad, param, m, v, maxv, amsgrad, wd, sc, i, end);     \
}

#define TP_OPT_SGD_VEC(FN, T, VTYPE, W, SUF, VECW)                                  \
static void FN(const T* grad, T* param, T* buf, bool has_buf,                \
               bool first_step, bool nesterov, bool wd, const T sc[4],       \
               int64_t begin, int64_t end) {                                 \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[0]);                           \
    const VTYPE vmom = _mm##W##_set1_##SUF(sc[1]);                          \
    const VTYPE vdamp = _mm##W##_set1_##SUF(sc[2]);                         \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[3]);                        \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                     \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        if (has_buf) {                                                       \
            VTYPE b = _mm##W##_loadu_##SUF(buf + i);                        \
            b = first_step ? g                                               \
                           : _mm##W##_fmadd_##SUF(vmom, b,                   \
                                                  _mm##W##_mul_##SUF(vdamp,  \
                                                                     g));    \
            _mm##W##_storeu_##SUF(buf + i, b);                               \
            g = nesterov ? _mm##W##_fmadd_##SUF(vmom, b, g) : b;             \
        }                                                                    \
        _mm##W##_storeu_##SUF(param + i,                                     \
                              _mm##W##_sub_##SUF(p,                          \
                                                 _mm##W##_mul_##SUF(vlr, g)));\
    }                                                                        \
    sgd_range_scalar(grad, param, buf, has_buf, first_step, nesterov, wd,    \
                     sc, i, end);                                            \
}

#define TP_OPT_FSGD_VEC(FN, T, VTYPE, W, SUF, VECW)                                 \
static void FN(T* grad, T* param, T* buf, bool has_buf, bool first_step,     \
               bool nesterov, bool has_scale, bool maximize, bool wd,        \
               const T sc[5], int64_t begin, int64_t end) {                  \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                           \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[0]);                           \
    const VTYPE vmom = _mm##W##_set1_##SUF(sc[1]);                          \
    const VTYPE vdamp = _mm##W##_set1_##SUF(sc[2]);                         \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[3]);                        \
    const VTYPE vscale = _mm##W##_set1_##SUF(sc[4]);                        \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                     \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        if (has_scale) {                                                     \
            g = _mm##W##_div_##SUF(g, vscale);                               \
            _mm##W##_storeu_##SUF(grad + i, g);                              \
        }                                                                    \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        if (has_buf) {                                                       \
            VTYPE b = _mm##W##_loadu_##SUF(buf + i);                        \
            b = first_step ? g                                               \
                           : _mm##W##_fmadd_##SUF(vmom, b,                   \
                                                  _mm##W##_mul_##SUF(vdamp,  \
                                                                     g));    \
            _mm##W##_storeu_##SUF(buf + i, b);                               \
            g = nesterov ? _mm##W##_fmadd_##SUF(vmom, b, g) : b;             \
        }                                                                    \
        _mm##W##_storeu_##SUF(param + i,                                     \
                              _mm##W##_sub_##SUF(p,                          \
                                                 _mm##W##_mul_##SUF(vlr, g)));\
    }                                                                        \
    fused_sgd_range_scalar(grad, param, buf, has_buf, first_step, nesterov,  \
                           has_scale, maximize, wd, sc, i, end);             \
}

#define TP_OPT_FADAM_VEC(FN, T, VTYPE, W, SUF, VECW)                                \
static void FN(T* grad, T* param, T* m, T* v, T* maxv, bool amsgrad,         \
               bool adamw, bool coupled_wd, bool has_scale, bool maximize,   \
               const T sc[10], int64_t begin, int64_t end) {                 \
    const bool small_lerp = std::abs(static_cast<double>(sc[1])) < 0.5;      \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                           \
    const VTYPE vlw = _mm##W##_set1_##SUF(sc[1]);                           \
    const VTYPE voneminuslw =                                               \
        _mm##W##_set1_##SUF(static_cast<T>(1) - sc[1]);                      \
    const VTYPE vb2 = _mm##W##_set1_##SUF(sc[2]);                           \
    const VTYPE vomb2 = _mm##W##_set1_##SUF(sc[3]);                         \
    const VTYPE vstep = _mm##W##_set1_##SUF(sc[4]);                         \
    const VTYPE vc2s = _mm##W##_set1_##SUF(sc[5]);                          \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[6]);                          \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[7]);                        \
    const VTYPE vwdfac = _mm##W##_set1_##SUF(sc[8]);                        \
    const VTYPE vscale = _mm##W##_set1_##SUF(sc[9]);                        \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                     \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        if (has_scale) {                                                     \
            g = _mm##W##_div_##SUF(g, vscale);                               \
            _mm##W##_storeu_##SUF(grad + i, g);                              \
        }                                                                    \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (adamw) {                                                         \
            p = _mm##W##_mul_##SUF(p, vwdfac);                               \
        } else if (coupled_wd) {                                             \
            g = _mm##W##_fmadd_##SUF(vdecay, p, g);                          \
        }                                                                    \
        VTYPE mv = _mm##W##_loadu_##SUF(m + i);                             \
        if (small_lerp) {                                                    \
            mv = _mm##W##_fmadd_##SUF(vlw, _mm##W##_sub_##SUF(g, mv), mv);   \
        } else {                                                             \
            mv = _mm##W##_fnmadd_##SUF(                                      \
                _mm##W##_sub_##SUF(g, mv), voneminuslw, g);                  \
        }                                                                    \
        _mm##W##_storeu_##SUF(m + i, mv);                                    \
        VTYPE vv = _mm##W##_loadu_##SUF(v + i);                             \
        vv = _mm##W##_fmadd_##SUF(vb2, vv,                                   \
                                  _mm##W##_mul_##SUF(vomb2,                  \
                                                     _mm##W##_mul_##SUF(g, g))); \
        _mm##W##_storeu_##SUF(v + i, vv);                                    \
        VTYPE s = vv;                                                       \
        if (amsgrad) {                                                       \
            VTYPE mx = _mm##W##_max_##SUF(_mm##W##_loadu_##SUF(maxv + i),   \
                                           vv);                              \
            _mm##W##_storeu_##SUF(maxv + i, mx);                             \
            s = mx;                                                          \
        }                                                                    \
        const VTYPE denom = _mm##W##_add_##SUF(                             \
            _mm##W##_div_##SUF(_mm##W##_sqrt_##SUF(s), vc2s), veps);         \
        const VTYPE upd =                                                   \
            _mm##W##_div_##SUF(_mm##W##_mul_##SUF(vstep, mv), denom);        \
        _mm##W##_storeu_##SUF(param + i, _mm##W##_sub_##SUF(p, upd));        \
    }                                                                        \
    fused_adam_range_scalar(grad, param, m, v, maxv, amsgrad, adamw,         \
                            coupled_wd, has_scale, maximize, sc, i, end);    \
}

#define TP_OPT_FADAGRAD_VEC(FN, T, VTYPE, W, SUF, VECW)                              \
static void FN(T* grad, T* param, T* state_sum, bool has_scale, bool maximize, \
               bool wd, const T sc[4], int64_t begin, int64_t end) {        \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                            \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[0]);                            \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[1]);                           \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[2]);                         \
    const VTYPE vscale = _mm##W##_set1_##SUF(sc[3]);                         \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                     \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                            \
        if (has_scale) {                                                     \
            g = _mm##W##_div_##SUF(g, vscale);                               \
            _mm##W##_storeu_##SUF(grad + i, g);                              \
        }                                                                    \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                           \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        VTYPE sum = _mm##W##_add_##SUF(                                      \
            _mm##W##_loadu_##SUF(state_sum + i),                             \
            _mm##W##_mul_##SUF(g, g));                                       \
        _mm##W##_storeu_##SUF(state_sum + i, sum);                           \
        VTYPE denom = _mm##W##_add_##SUF(                                    \
            _mm##W##_sqrt_##SUF(sum), veps);                                 \
        VTYPE update = _mm##W##_div_##SUF(g, denom);                          \
        _mm##W##_storeu_##SUF(param + i, _mm##W##_sub_##SUF(                 \
            p, _mm##W##_mul_##SUF(vlr, update)));                             \
    }                                                                        \
    fused_adagrad_range_scalar(grad, param, state_sum, has_scale, maximize,  \
                               wd, sc, i, end);                              \
}

#define TP_OPT_FRMS_VEC(FN, T, VTYPE, W, SUF, VECW)                                  \
static void FN(const T* grad, T* param, T* square_avg, T* grad_avg,         \
               T* momentum_buffer, bool centered, bool has_momentum,        \
               bool wd, bool maximize, const T sc[6],                      \
               int64_t begin, int64_t end) {                                \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                           \
    const VTYPE valpha = _mm##W##_set1_##SUF(sc[0]);                         \
    const VTYPE vomalpha = _mm##W##_set1_##SUF(sc[1]);                       \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[2]);                            \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[3]);                           \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[4]);                         \
    const VTYPE vmomentum = _mm##W##_set1_##SUF(sc[5]);                      \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                    \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        VTYPE square = _mm##W##_fmadd_##SUF(                                \
            valpha, _mm##W##_loadu_##SUF(square_avg + i),                   \
            _mm##W##_mul_##SUF(vomalpha, _mm##W##_mul_##SUF(g, g)));         \
        _mm##W##_storeu_##SUF(square_avg + i, square);                      \
        VTYPE avg = square;                                                  \
        if (centered) {                                                      \
            VTYPE mean = _mm##W##_fmadd_##SUF(                               \
                valpha, _mm##W##_loadu_##SUF(grad_avg + i),                  \
                _mm##W##_mul_##SUF(vomalpha, g));                            \
            _mm##W##_storeu_##SUF(grad_avg + i, mean);                       \
            avg = _mm##W##_sub_##SUF(                                         \
                square, _mm##W##_mul_##SUF(mean, mean));                     \
        }                                                                    \
        avg = _mm##W##_add_##SUF(                                             \
            _mm##W##_sqrt_##SUF(avg), veps);                                 \
        VTYPE update = _mm##W##_div_##SUF(g, avg);                            \
        if (has_momentum) {                                                  \
            VTYPE buffer = _mm##W##_add_##SUF(                               \
                _mm##W##_mul_##SUF(vmomentum,                               \
                                   _mm##W##_loadu_##SUF(momentum_buffer + i)), \
                update);                                                      \
            _mm##W##_storeu_##SUF(momentum_buffer + i, buffer);              \
            update = buffer;                                                 \
        }                                                                    \
        _mm##W##_storeu_##SUF(param + i, _mm##W##_sub_##SUF(                 \
            p, _mm##W##_mul_##SUF(vlr, update)));                             \
    }                                                                        \
    fused_rmsprop_range_scalar<T, T>(                                        \
        grad, param, square_avg, grad_avg, momentum_buffer, centered,        \
        has_momentum, wd, maximize, sc, i, end);                              \
}

#define TP_OPT_FADDELTA_VEC(FN, T, VTYPE, W, SUF, VECW)                              \
static void FN(const T* grad, T* param, T* square_avg, T* acc_delta,         \
               bool wd, bool maximize, const T sc[5],                      \
               int64_t begin, int64_t end) {                                \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                           \
    const VTYPE vone = _mm##W##_set1_##SUF(static_cast<T>(1));               \
    const VTYPE vrho = _mm##W##_set1_##SUF(sc[0]);                           \
    const VTYPE vomrho = _mm##W##_set1_##SUF(sc[1]);                         \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[2]);                            \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[3]);                           \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[4]);                         \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                    \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        VTYPE square = _mm##W##_fmadd_##SUF(                                \
            vrho, _mm##W##_loadu_##SUF(square_avg + i),                     \
            _mm##W##_mul_##SUF(vomrho, _mm##W##_mul_##SUF(g, g)));           \
        _mm##W##_storeu_##SUF(square_avg + i, square);                      \
        VTYPE std = _mm##W##_sqrt_##SUF(                                     \
            _mm##W##_add_##SUF(square, veps));                              \
        VTYPE delta = _mm##W##_sqrt_##SUF(                                   \
            _mm##W##_add_##SUF(_mm##W##_loadu_##SUF(acc_delta + i), veps));  \
        delta = _mm##W##_mul_##SUF(                                          \
            _mm##W##_div_##SUF(delta, std), g);                              \
        VTYPE next_acc = _mm##W##_fmadd_##SUF(                               \
            vrho, _mm##W##_loadu_##SUF(acc_delta + i),                       \
            _mm##W##_mul_##SUF(vomrho, _mm##W##_mul_##SUF(delta, delta)));   \
        _mm##W##_storeu_##SUF(acc_delta + i, next_acc);                      \
        _mm##W##_storeu_##SUF(param + i, _mm##W##_sub_##SUF(                 \
            p, _mm##W##_mul_##SUF(vlr, delta)));                              \
    }                                                                        \
    fused_adadelta_range_scalar<T, T>(                                       \
        grad, param, square_avg, acc_delta, wd, maximize, sc, i, end);       \
}

#define TP_OPT_FNADAM_VEC(FN, T, VTYPE, W, SUF, VECW)                                \
static void FN(const T* grad, T* param, T* exp_avg, T* exp_avg_sq,          \
               bool wd, bool decoupled_wd, bool maximize, const T sc[10],   \
               int64_t begin, int64_t end) {                                \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                           \
    const VTYPE vone = _mm##W##_set1_##SUF(static_cast<T>(1));               \
    const VTYPE vlr = _mm##W##_set1_##SUF(sc[0]);                            \
    const VTYPE vomb1 = _mm##W##_set1_##SUF(sc[5]);                          \
    const VTYPE vb2 = _mm##W##_set1_##SUF(sc[2]);                            \
    const VTYPE vomb2 = _mm##W##_set1_##SUF(sc[6]);                          \
    const VTYPE vgrad_coeff = _mm##W##_set1_##SUF(sc[7]);                    \
    const VTYPE vexpavg_coeff = _mm##W##_set1_##SUF(sc[8]);                  \
    const VTYPE veps = _mm##W##_set1_##SUF(sc[3]);                           \
    const VTYPE vdecay = _mm##W##_set1_##SUF(sc[4]);                         \
    const VTYPE vc2s = _mm##W##_set1_##SUF(sc[9]);                          \
    int64_t i = begin;                                                       \
    for (; i + VECW <= end; i += VECW) {                                    \
        VTYPE g = _mm##W##_loadu_##SUF(grad + i);                           \
        if (maximize) g = _mm##W##_sub_##SUF(vzero, g);                      \
        VTYPE p = _mm##W##_loadu_##SUF(param + i);                          \
        if (wd) {                                                            \
            if (decoupled_wd)                                                \
                p = _mm##W##_mul_##SUF(p, _mm##W##_sub_##SUF(               \
                    vone, _mm##W##_mul_##SUF(vlr, vdecay)));                \
            else                                                             \
                g = _mm##W##_fmadd_##SUF(vdecay, p, g);                      \
        }                                                                    \
        VTYPE m = _mm##W##_loadu_##SUF(exp_avg + i);                         \
        m = _mm##W##_fmadd_##SUF(vomb1, _mm##W##_sub_##SUF(g, m), m);        \
        _mm##W##_storeu_##SUF(exp_avg + i, m);                              \
        VTYPE v = _mm##W##_loadu_##SUF(exp_avg_sq + i);                      \
        v = _mm##W##_fmadd_##SUF(vb2, v,                                    \
            _mm##W##_mul_##SUF(vomb2, _mm##W##_mul_##SUF(g, g)));            \
        _mm##W##_storeu_##SUF(exp_avg_sq + i, v);                            \
        const VTYPE denom = _mm##W##_add_##SUF(                             \
            _mm##W##_div_##SUF(_mm##W##_sqrt_##SUF(v), vc2s), veps);         \
        p = _mm##W##_add_##SUF(p, _mm##W##_div_##SUF(                        \
            _mm##W##_mul_##SUF(vgrad_coeff, g), denom));                     \
        p = _mm##W##_add_##SUF(p, _mm##W##_div_##SUF(                        \
            _mm##W##_mul_##SUF(vexpavg_coeff, m), denom));                   \
        _mm##W##_storeu_##SUF(param + i, p);                                \
    }                                                                        \
    fused_nadam_range_scalar<T, T>(grad, param, exp_avg, exp_avg_sq, wd,     \
                                   decoupled_wd, maximize, sc, i, end);      \
}

#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace avx2_target {
TP_OPT_ADAM_VEC(adam_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_ADAM_VEC(adam_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_SGD_VEC(sgd_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_SGD_VEC(sgd_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FSGD_VEC(fused_sgd_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FSGD_VEC(fused_sgd_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FADAM_VEC(fused_adam_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FADAM_VEC(fused_adam_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FADAGRAD_VEC(fused_adagrad_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FADAGRAD_VEC(fused_adagrad_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FRMS_VEC(fused_rmsprop_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FRMS_VEC(fused_rmsprop_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FADDELTA_VEC(fused_adadelta_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FADDELTA_VEC(fused_adadelta_f64_chunk, double, __m256d, 256, pd, 4)
TP_OPT_FNADAM_VEC(fused_nadam_f32_chunk, float, __m256, 256, ps, 8)
TP_OPT_FNADAM_VEC(fused_nadam_f64_chunk, double, __m256d, 256, pd, 4)
}  // namespace avx2_target
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx512f")
namespace avx512_target {
TP_OPT_ADAM_VEC(adam_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_ADAM_VEC(adam_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_SGD_VEC(sgd_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_SGD_VEC(sgd_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FSGD_VEC(fused_sgd_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FSGD_VEC(fused_sgd_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FADAM_VEC(fused_adam_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FADAM_VEC(fused_adam_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FADAGRAD_VEC(fused_adagrad_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FADAGRAD_VEC(fused_adagrad_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FRMS_VEC(fused_rmsprop_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FRMS_VEC(fused_rmsprop_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FADDELTA_VEC(fused_adadelta_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FADDELTA_VEC(fused_adadelta_f64_chunk, double, __m512d, 512, pd, 8)
TP_OPT_FNADAM_VEC(fused_nadam_f32_chunk, float, __m512, 512, ps, 16)
TP_OPT_FNADAM_VEC(fused_nadam_f64_chunk, double, __m512d, 512, pd, 8)
}  // namespace avx512_target
#pragma GCC pop_options

#undef TP_OPT_ADAM_VEC
#undef TP_OPT_SGD_VEC
#undef TP_OPT_FSGD_VEC
#undef TP_OPT_FADAM_VEC
#undef TP_OPT_FADAGRAD_VEC
#undef TP_OPT_FRMS_VEC
#undef TP_OPT_FADDELTA_VEC
#undef TP_OPT_FNADAM_VEC

template <typename T>
void adam_chunk_dispatch(const T* grad, T* param, T* m, T* v, T* maxv,
                         bool amsgrad, bool wd, const T sc[8],
                         int64_t begin, int64_t end) {
    if (have_avx512f()) {
        if constexpr (std::is_same_v<T, float>) {
            avx512_target::adam_f32_chunk(grad, param, m, v, maxv, amsgrad,
                                          wd, sc, begin, end);
        } else {
            avx512_target::adam_f64_chunk(grad, param, m, v, maxv, amsgrad,
                                          wd, sc, begin, end);
        }
    } else if (have_avx2()) {
        if constexpr (std::is_same_v<T, float>) {
            avx2_target::adam_f32_chunk(grad, param, m, v, maxv, amsgrad, wd,
                                        sc, begin, end);
        } else {
            avx2_target::adam_f64_chunk(grad, param, m, v, maxv, amsgrad, wd,
                                        sc, begin, end);
        }
    } else {
        adam_range_scalar(grad, param, m, v, maxv, amsgrad, wd, sc, begin,
                          end);
    }
}

template <typename T>
void sgd_chunk_dispatch(const T* grad, T* param, T* buf, bool has_buf,
                        bool first_step, bool nesterov, bool wd,
                        const T sc[4], int64_t begin, int64_t end) {
    if (have_avx512f()) {
        if constexpr (std::is_same_v<T, float>) {
            avx512_target::sgd_f32_chunk(grad, param, buf, has_buf,
                                         first_step, nesterov, wd, sc, begin,
                                         end);
        } else {
            avx512_target::sgd_f64_chunk(grad, param, buf, has_buf,
                                         first_step, nesterov, wd, sc, begin,
                                         end);
        }
    } else if (have_avx2()) {
        if constexpr (std::is_same_v<T, float>) {
            avx2_target::sgd_f32_chunk(grad, param, buf, has_buf, first_step,
                                       nesterov, wd, sc, begin, end);
        } else {
            avx2_target::sgd_f64_chunk(grad, param, buf, has_buf, first_step,
                                       nesterov, wd, sc, begin, end);
        }
    } else {
        sgd_range_scalar(grad, param, buf, has_buf, first_step, nesterov, wd,
                         sc, begin, end);
    }
}

template <typename T>
void fused_sgd_chunk_dispatch(T* grad, T* param, T* buf, bool has_buf,
                              bool first_step, bool nesterov, bool has_scale,
                              bool maximize, bool wd, const T sc[5],
                              int64_t begin, int64_t end) {
    if (have_avx512f()) {
        if constexpr (std::is_same_v<T, float>) {
            avx512_target::fused_sgd_f32_chunk(grad, param, buf, has_buf,
                                               first_step, nesterov,
                                               has_scale, maximize, wd, sc,
                                               begin, end);
        } else {
            avx512_target::fused_sgd_f64_chunk(grad, param, buf, has_buf,
                                               first_step, nesterov,
                                               has_scale, maximize, wd, sc,
                                               begin, end);
        }
    } else if (have_avx2()) {
        if constexpr (std::is_same_v<T, float>) {
            avx2_target::fused_sgd_f32_chunk(grad, param, buf, has_buf,
                                             first_step, nesterov, has_scale,
                                             maximize, wd, sc, begin, end);
        } else {
            avx2_target::fused_sgd_f64_chunk(grad, param, buf, has_buf,
                                             first_step, nesterov, has_scale,
                                             maximize, wd, sc, begin, end);
        }
    } else {
        fused_sgd_range_scalar(grad, param, buf, has_buf, first_step,
                               nesterov, has_scale, maximize, wd, sc, begin,
                               end);
    }
}

template <typename T>
void fused_adam_chunk_dispatch(T* grad, T* param, T* m, T* v, T* maxv,
                               bool amsgrad, bool adamw, bool coupled_wd,
                               bool has_scale, bool maximize, const T sc[10],
                               int64_t begin, int64_t end) {
    if (have_avx512f()) {
        if constexpr (std::is_same_v<T, float>) {
            avx512_target::fused_adam_f32_chunk(grad, param, m, v, maxv,
                                                amsgrad, adamw, coupled_wd,
                                                has_scale, maximize, sc,
                                                begin, end);
        } else {
            avx512_target::fused_adam_f64_chunk(grad, param, m, v, maxv,
                                                amsgrad, adamw, coupled_wd,
                                                has_scale, maximize, sc,
                                                begin, end);
        }
    } else if (have_avx2()) {
        if constexpr (std::is_same_v<T, float>) {
            avx2_target::fused_adam_f32_chunk(grad, param, m, v, maxv,
                                              amsgrad, adamw, coupled_wd,
                                              has_scale, maximize, sc, begin,
                                              end);
        } else {
            avx2_target::fused_adam_f64_chunk(grad, param, m, v, maxv,
                                              amsgrad, adamw, coupled_wd,
                                              has_scale, maximize, sc, begin,
                                              end);
        }
    } else {
        fused_adam_range_scalar(grad, param, m, v, maxv, amsgrad, adamw,
                                coupled_wd, has_scale, maximize, sc, begin,
                                end);
    }
}

template <typename T, typename M>
void fused_adagrad_chunk_dispatch(T* grad, T* param, T* state_sum,
                                  bool has_scale, bool maximize, bool wd,
                                  const M sc[4], int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<T, M> &&
                  (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        if (have_avx512f()) {
            if constexpr (std::is_same_v<T, float>) {
                avx512_target::fused_adagrad_f32_chunk(
                    grad, param, state_sum, has_scale, maximize, wd, sc,
                    begin, end);
            } else {
                avx512_target::fused_adagrad_f64_chunk(
                    grad, param, state_sum, has_scale, maximize, wd, sc,
                    begin, end);
            }
        } else if (have_avx2()) {
            if constexpr (std::is_same_v<T, float>) {
                avx2_target::fused_adagrad_f32_chunk(
                    grad, param, state_sum, has_scale, maximize, wd, sc,
                    begin, end);
            } else {
                avx2_target::fused_adagrad_f64_chunk(
                    grad, param, state_sum, has_scale, maximize, wd, sc,
                    begin, end);
            }
        } else {
            fused_adagrad_range_scalar<T, M>(
                grad, param, state_sum, has_scale, maximize, wd, sc, begin,
                end);
        }
    } else {
        fused_adagrad_range_scalar<T, M>(
            grad, param, state_sum, has_scale, maximize, wd, sc, begin, end);
    }
}

template <typename T, typename M>
void fused_nadam_chunk_dispatch(
        const T* grad, T* param, T* exp_avg, T* exp_avg_sq, bool wd,
        bool decoupled_wd, bool maximize, const M sc[10], int64_t begin,
        int64_t end) {
    if constexpr (std::is_same_v<T, M> &&
                  (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        if (have_avx512f()) {
            if constexpr (std::is_same_v<T, float>) {
                avx512_target::fused_nadam_f32_chunk(
                    grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd,
                    maximize, sc,
                    begin, end);
            } else {
                avx512_target::fused_nadam_f64_chunk(
                    grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd,
                    maximize, sc,
                    begin, end);
            }
        } else if (have_avx2()) {
            if constexpr (std::is_same_v<T, float>) {
                avx2_target::fused_nadam_f32_chunk(
                    grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd,
                    maximize, sc,
                    begin, end);
            } else {
                avx2_target::fused_nadam_f64_chunk(
                    grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd,
                    maximize, sc,
                    begin, end);
            }
        } else {
            fused_nadam_range_scalar<T, M>(
                grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd, maximize,
                sc, begin, end);
        }
    } else {
        fused_nadam_range_scalar<T, M>(
            grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd, maximize, sc,
            begin, end);
    }
}

template <typename T, typename M>
void fused_rmsprop_chunk_dispatch(
        const T* grad, T* param, T* square_avg, T* grad_avg,
        T* momentum_buffer, bool centered, bool has_momentum, bool wd,
        bool maximize, const M sc[6], int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<T, M> &&
                  (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        if (have_avx512f()) {
            if constexpr (std::is_same_v<T, float>) {
                avx512_target::fused_rmsprop_f32_chunk(
                    grad, param, square_avg, grad_avg, momentum_buffer,
                    centered, has_momentum, wd, maximize, sc, begin, end);
            } else {
                avx512_target::fused_rmsprop_f64_chunk(
                    grad, param, square_avg, grad_avg, momentum_buffer,
                    centered, has_momentum, wd, maximize, sc, begin, end);
            }
        } else if (have_avx2()) {
            if constexpr (std::is_same_v<T, float>) {
                avx2_target::fused_rmsprop_f32_chunk(
                    grad, param, square_avg, grad_avg, momentum_buffer,
                    centered, has_momentum, wd, maximize, sc, begin, end);
            } else {
                avx2_target::fused_rmsprop_f64_chunk(
                    grad, param, square_avg, grad_avg, momentum_buffer,
                    centered, has_momentum, wd, maximize, sc, begin, end);
            }
        } else {
            fused_rmsprop_range_scalar<T, M>(
                grad, param, square_avg, grad_avg, momentum_buffer,
                centered, has_momentum, wd, maximize, sc, begin, end);
        }
    } else {
        fused_rmsprop_range_scalar<T, M>(
            grad, param, square_avg, grad_avg, momentum_buffer, centered,
            has_momentum, wd, maximize, sc, begin, end);
    }
}

template <typename T, typename M>
void fused_adadelta_chunk_dispatch(
        const T* grad, T* param, T* square_avg, T* acc_delta, bool wd,
        bool maximize, const M sc[5], int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<T, M> &&
                  (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        if (have_avx512f()) {
            if constexpr (std::is_same_v<T, float>) {
                avx512_target::fused_adadelta_f32_chunk(
                    grad, param, square_avg, acc_delta, wd, maximize, sc,
                    begin, end);
            } else {
                avx512_target::fused_adadelta_f64_chunk(
                    grad, param, square_avg, acc_delta, wd, maximize, sc,
                    begin, end);
            }
        } else if (have_avx2()) {
            if constexpr (std::is_same_v<T, float>) {
                avx2_target::fused_adadelta_f32_chunk(
                    grad, param, square_avg, acc_delta, wd, maximize, sc,
                    begin, end);
            } else {
                avx2_target::fused_adadelta_f64_chunk(
                    grad, param, square_avg, acc_delta, wd, maximize, sc,
                    begin, end);
            }
        } else {
            fused_adadelta_range_scalar<T, M>(
                grad, param, square_avg, acc_delta, wd, maximize, sc, begin,
                end);
        }
    } else {
        fused_adadelta_range_scalar<T, M>(
            grad, param, square_avg, acc_delta, wd, maximize, sc, begin, end);
    }
}
#else
template <typename T>
void adam_chunk_dispatch(const T* grad, T* param, T* m, T* v, T* maxv,
                         bool amsgrad, bool wd, const T sc[8],
                         int64_t begin, int64_t end) {
    adam_range_scalar(grad, param, m, v, maxv, amsgrad, wd, sc, begin, end);
}

template <typename T>
void sgd_chunk_dispatch(const T* grad, T* param, T* buf, bool has_buf,
                        bool first_step, bool nesterov, bool wd,
                        const T sc[4], int64_t begin, int64_t end) {
    sgd_range_scalar(grad, param, buf, has_buf, first_step, nesterov, wd, sc,
                     begin, end);
}

template <typename T>
void fused_sgd_chunk_dispatch(T* grad, T* param, T* buf, bool has_buf,
                              bool first_step, bool nesterov, bool has_scale,
                              bool maximize, bool wd, const T sc[5],
                              int64_t begin, int64_t end) {
    fused_sgd_range_scalar(grad, param, buf, has_buf, first_step, nesterov,
                           has_scale, maximize, wd, sc, begin, end);
}

template <typename T>
void fused_adam_chunk_dispatch(T* grad, T* param, T* m, T* v, T* maxv,
                               bool amsgrad, bool adamw, bool coupled_wd,
                               bool has_scale, bool maximize, const T sc[10],
                               int64_t begin, int64_t end) {
    fused_adam_range_scalar(grad, param, m, v, maxv, amsgrad, adamw,
                            coupled_wd, has_scale, maximize, sc, begin, end);
}

template <typename T, typename M>
void fused_adagrad_chunk_dispatch(T* grad, T* param, T* state_sum,
                                  bool has_scale, bool maximize, bool wd,
                                  const M sc[4], int64_t begin, int64_t end) {
    fused_adagrad_range_scalar<T, M>(
        grad, param, state_sum, has_scale, maximize, wd, sc, begin, end);
}

template <typename T, typename M>
void fused_nadam_chunk_dispatch(
        const T* grad, T* param, T* exp_avg, T* exp_avg_sq, bool wd,
        bool decoupled_wd, bool maximize, const M sc[10], int64_t begin,
        int64_t end) {
    fused_nadam_range_scalar<T, M>(
        grad, param, exp_avg, exp_avg_sq, wd, decoupled_wd, maximize, sc,
        begin, end);
}

template <typename T, typename M>
void fused_rmsprop_chunk_dispatch(
        const T* grad, T* param, T* square_avg, T* grad_avg,
        T* momentum_buffer, bool centered, bool has_momentum, bool wd,
        bool maximize, const M sc[6], int64_t begin, int64_t end) {
    fused_rmsprop_range_scalar<T, M>(
        grad, param, square_avg, grad_avg, momentum_buffer, centered,
        has_momentum, wd, maximize, sc, begin, end);
}

template <typename T, typename M>
void fused_adadelta_chunk_dispatch(
        const T* grad, T* param, T* square_avg, T* acc_delta, bool wd,
        bool maximize, const M sc[5], int64_t begin, int64_t end) {
    fused_adadelta_range_scalar<T, M>(
        grad, param, square_avg, acc_delta, wd, maximize, sc, begin, end);
}
#endif  // TP_OPT_X86_SIMD

void validate_lists(const std::vector<Tensor>& params,
                   const std::vector<Tensor>& grads,
                   const std::vector<Tensor>& first_state,
                   const std::vector<Tensor>& second_state,
                   const std::vector<Tensor>& third_state,
                   const std::vector<int64_t>& steps,
                   bool require_first_state,
                   bool require_second_state,
                   bool require_third_state,
                   const char* op_name) {
    const auto count = params.size();
    // An optional state list may be entirely absent (e.g. _foreach_sgd with
    // momentum == 0 receives no momentum buffers); when present it must still
    // cover every parameter.
    if (grads.size() != count ||
        ((require_first_state || !first_state.empty()) &&
         first_state.size() != count) ||
        ((require_second_state || !second_state.empty()) &&
         second_state.size() != count) ||
        (!third_state.empty() && third_state.size() != count)) {
        TP_THROW(ValueError, std::string(op_name) +
            ": tensor list sizes must match");
    }
    if (!steps.empty() && steps.size() != count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": step list size must match parameter list");
    }

    const DType dtype = count ? params[0].dtype() : DType::Undefined;
    const Device device = count ? params[0].device() : Device(DeviceType::CPU);
    for (size_t i = 0; i < count; ++i) {
        const Tensor& param = params[i];
        const Tensor& grad = grads[i];
        if (!param.defined() || !grad.defined()) {
            TP_THROW(ValueError, std::string(op_name) +
                ": parameters and gradients must be defined");
        }
        if (!param.is_contiguous() || !grad.is_contiguous() ||
            param.shape() != grad.shape() || param.dtype() != grad.dtype() ||
            param.dtype() != dtype || param.device() != device) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": requires contiguous parameter/gradient pairs with matching shape and dtype");
        }
        if (param.device() != grad.device()) {
            TP_THROW(DeviceMismatchError, std::string(op_name) +
                ": parameter and gradient must be on the same device");
        }

        if (require_first_state && !first_state.empty()) {
            const Tensor& state = first_state[i];
            if (!state.defined()) {
                TP_THROW(ValueError, std::string(op_name) +
                    ": required optimizer state is undefined");
            }
            if (!state.is_contiguous() || state.shape() != param.shape() ||
                state.dtype() != param.dtype() || state.device() != param.device()) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": optimizer state must match its parameter layout");
            }
        }
        if (require_second_state && !second_state.empty()) {
            const Tensor& state = second_state[i];
            if (!state.defined()) {
                TP_THROW(ValueError, std::string(op_name) +
                    ": required optimizer state is undefined");
            }
            if (!state.is_contiguous() || state.shape() != param.shape() ||
                state.dtype() != param.dtype() || state.device() != param.device()) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": optimizer state must match its parameter layout");
            }
        }
        if (require_third_state) {
            if (third_state.empty()) {
                TP_THROW(ValueError, std::string(op_name) +
                    ": required optimizer state is undefined");
            }
            const Tensor& state = third_state[i];
            if (!state.defined() || !state.is_contiguous() ||
                state.shape() != param.shape() || state.dtype() != param.dtype() ||
                state.device() != param.device()) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": AMSGrad state must match its parameter layout");
            }
        }
    }
}

template <typename scalar_t>
void sgd_impl(const std::vector<Tensor>& params,
              const std::vector<Tensor>& grads,
              const std::vector<Tensor>& momentum_buffers,
              double lr,
              double momentum,
              double dampening,
              double weight_decay,
              bool nesterov,
              bool first_momentum_step) {
    const bool has_momentum = momentum != 0.0;
    const bool has_wd = weight_decay != 0.0;
    const scalar_t sc[4] = {
        static_cast<scalar_t>(lr),
        static_cast<scalar_t>(momentum),
        static_cast<scalar_t>(1.0 - dampening),
        static_cast<scalar_t>(weight_decay),
    };

    // into one horizontally fused work list, then schedule that list once.
    // Scheduling each parameter independently leaves large ResNet tensors
    // serialized behind a single worker and pays one barrier per tensor.
    std::vector<int64_t> numels(params.size());
    for (size_t i = 0; i < params.size(); ++i) numels[i] = params[i].numel();
    const auto work = build_opt_work_list(numels.data(), numels.size());
    if (work.empty()) return;

    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end; ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            scalar_t* param = params[item.list_index].template data_ptr<scalar_t>();
            const scalar_t* grad = grads[item.list_index].template data_ptr<scalar_t>();
            scalar_t* buffer = has_momentum && momentum_buffers[item.list_index].defined()
                ? momentum_buffers[item.list_index].template data_ptr<scalar_t>() : nullptr;

            sgd_chunk_dispatch<scalar_t>(
                grad, param, buffer, buffer != nullptr, first_momentum_step,
                nesterov, has_wd, sc, item.begin, item.end);
        }
    });
}

template <typename scalar_t>
void adam_impl(const std::vector<Tensor>& params,
               const std::vector<Tensor>& grads,
               const std::vector<Tensor>& exp_avgs,
               const std::vector<Tensor>& exp_avg_sqs,
               const std::vector<Tensor>& max_exp_avg_sqs,
               const std::vector<int64_t>& steps,
               double lr,
               double beta1,
               double beta2,
               double eps,
               double weight_decay,
               bool amsgrad) {
    const bool has_wd = weight_decay != 0.0;
    const size_t count = params.size();

    std::vector<scalar_t*> param_p(count);
    std::vector<const scalar_t*> grad_p(count);
    std::vector<scalar_t*> m_p(count);
    std::vector<scalar_t*> v_p(count);
    std::vector<scalar_t*> maxv_p(count);
    // Per-tensor bias-corrected constants, hoisted out of the worker loop.
    // sc layout: {beta1, 1-beta1, beta2, 1-beta2, step_size, corr2_sqrt, eps, decay}
    std::vector<std::array<scalar_t, 8>> sc(count);
    for (size_t i = 0; i < count; ++i) {
        param_p[i] = params[i].data_ptr<scalar_t>();
        grad_p[i] = grads[i].data_ptr<scalar_t>();
        m_p[i] = exp_avgs[i].data_ptr<scalar_t>();
        v_p[i] = exp_avg_sqs[i].data_ptr<scalar_t>();
        maxv_p[i] = amsgrad ? max_exp_avg_sqs[i].data_ptr<scalar_t>() : nullptr;
        const int64_t step = steps[i];
        const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
        const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));
        auto& s = sc[i];
        s[0] = static_cast<scalar_t>(beta1);
        s[1] = static_cast<scalar_t>(1.0 - beta1);
        s[2] = static_cast<scalar_t>(beta2);
        s[3] = static_cast<scalar_t>(1.0 - beta2);
        s[4] = static_cast<scalar_t>(lr / bc1);
        s[5] = static_cast<scalar_t>(std::sqrt(bc2));
        s[6] = static_cast<scalar_t>(eps);
        s[7] = static_cast<scalar_t>(weight_decay);
    }

    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) numels[i] = params[i].numel();
    const auto work = build_opt_work_list(numels.data(), count);
    if (work.empty()) return;

    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end; ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            adam_chunk_dispatch<scalar_t>(
                grad_p[li], param_p[li], m_p[li], v_p[li], maxv_p[li],
                amsgrad, has_wd, sc[li].data(), item.begin, item.end);
        }
    });
}

// Fused optimizers deliberately live in the native backend, just like
// and selects the overload; it must not rebuild these algorithms from
// pointwise Python calls.  `math_t` is the accumulation type used by the
bool fused_found_inf(const std::optional<Tensor>& found_inf) {
    return found_inf.has_value() && found_inf->defined() &&
        found_inf->numel() == 1 && found_inf->item().toDouble() == 1.0;
}

double fused_grad_scale(const std::optional<Tensor>& grad_scale) {
    if (!grad_scale.has_value() || !grad_scale->defined()) return 1.0;
    if (grad_scale->numel() != 1) {
        TP_THROW(ValueError, "fused optimizer grad_scale must be a singleton tensor");
    }
    return grad_scale->item().toDouble();
}

void validate_fused_pairs(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& grads,
                          const char* op_name) {
    if (params.size() != grads.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": parameter and gradient lists must have the same length");
    }
    if (params.empty()) return;
    const DType dtype = params[0].dtype();
    if (dtype != DType::Float16 && dtype != DType::BFloat16 &&
        dtype != DType::Float32 && dtype != DType::Float64) {
        TP_THROW(NotImplementedError, std::string(op_name) +
            ": fused kernels support float16, bfloat16, float32, and float64");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& param = params[i];
        const Tensor& grad = grads[i];
        if (!param.defined() || !grad.defined()) {
            TP_THROW(ValueError, std::string(op_name) +
                ": parameters and gradients must be defined");
        }
        if (param.is_sparse() || grad.is_sparse()) {
            TP_THROW(RuntimeError, std::string(op_name) +
                ": fused optimizers do not support sparse tensors");
        }
        if (isComplexType(param.dtype()) || !param.is_contiguous() ||
            !grad.is_contiguous() || param.shape() != grad.shape() ||
            param.dtype() != grad.dtype() || param.dtype() != dtype ||
            param.device() != Device(DeviceType::CPU) ||
            grad.device() != Device(DeviceType::CPU)) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": requires contiguous CPU tensors with matching floating dtype and shape");
        }
    }
}

void validate_fused_state(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& state,
                          bool required,
                          const char* op_name) {
    if (!required && state.empty()) return;
    if (state.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": optimizer state list must match parameter list");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        if (!state[i].defined() || !state[i].is_contiguous() ||
            state[i].shape() != params[i].shape() ||
            state[i].dtype() != params[i].dtype() ||
            state[i].device() != params[i].device()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": optimizer state must match its parameter layout");
        }
    }
}

void validate_fused_steps(const std::vector<Tensor>& params,
                          const std::vector<Tensor>& state_steps,
                          const char* op_name) {
    if (state_steps.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": state_steps must match parameter list");
    }
    for (const Tensor& step : state_steps) {
        if (!step.defined() || !step.is_contiguous() || step.numel() != 1 ||
            step.device() != Device(DeviceType::CPU) ||
            !isFloatingType(step.dtype())) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": state_steps must be singleton CPU floating tensors");
        }
    }
}

// Fused-SGD inner range.  sc holds {lr, momentum, 1-dampening,
// weight_decay, grad_scale}; implements the per-element update math.
template <typename T, typename Ops>
inline void fused_sgd_vec_range(T* grad, T* param, T* buf, bool has_buf,
                                bool first_step, bool nesterov,
                                bool has_scale, bool maximize, bool wd,
                                const T sc[5], int64_t begin, int64_t end) {
    const auto vlr = Ops::set1(sc[0]);
    const auto vmom = Ops::set1(sc[1]);
    const auto vdamp = Ops::set1(sc[2]);
    const auto vdecay = Ops::set1(sc[3]);
    const auto vscale = Ops::set1(sc[4]);
    constexpr int W = Ops::kW;
    int64_t i = begin;
    for (; i + W <= end; i += W) {
        auto g = Ops::load(grad + i);
        if (has_scale) { g = Ops::div(g, vscale); Ops::store(grad + i, g); }
        if (maximize) g = Ops::sub(Ops::set1(static_cast<T>(0)), g);
        auto p = Ops::load(param + i);
        if (wd) g = Ops::fmadd(vdecay, p, g);
        if (has_buf) {
            auto b = Ops::load(buf + i);
            b = first_step ? g : Ops::fmadd(vmom, b, Ops::mul(vdamp, g));
            Ops::store(buf + i, b);
            g = nesterov ? Ops::fmadd(vmom, b, g) : b;
        }
        Ops::store(param + i, Ops::sub(p, Ops::mul(vlr, g)));
    }
    for (; i < end; ++i) {
        T g = static_cast<T>(grad[i]);
        T p = static_cast<T>(param[i]);
        if (has_scale) { g = g / sc[4]; grad[i] = g; }
        if (maximize) g = -g;
        if (wd) g += sc[3] * p;
        if (has_buf) {
            T b = first_step ? g : sc[1] * buf[i] + sc[2] * g;
            buf[i] = b;
            g = nesterov ? g + sc[1] * b : b;
        }
        param[i] = static_cast<T>(p - sc[0] * g);
    }
}

template <typename scalar_t, typename math_t>
void fused_sgd_math(const std::vector<Tensor>& params,
                    const std::vector<Tensor>& grads,
                    const std::vector<Tensor>& momentum_buffers,
                    double lr,
                    double momentum,
                    double dampening,
                    double weight_decay,
                    bool nesterov,
                    bool maximize,
                    bool is_first_step,
                    double grad_scale) {
    const bool has_momentum = momentum != 0.0;
    const bool has_scale = grad_scale != 1.0;
    const bool has_wd = weight_decay != 0.0;
    const size_t count = params.size();
    if constexpr (std::is_same_v<scalar_t, math_t>) {
        std::vector<scalar_t*> param_p(count);
        std::vector<scalar_t*> grad_p(count);
        std::vector<scalar_t*> buf_p(count);
        for (size_t i = 0; i < count; ++i) {
            param_p[i] = params[i].data_ptr<scalar_t>();
            grad_p[i] = grads[i].data_ptr<scalar_t>();
            buf_p[i] = has_momentum
                ? momentum_buffers[i].data_ptr<scalar_t>() : nullptr;
        }
        const scalar_t sc[5] = {
            static_cast<scalar_t>(lr),
            static_cast<scalar_t>(momentum),
            static_cast<scalar_t>(1.0 - dampening),
            static_cast<scalar_t>(weight_decay),
            static_cast<scalar_t>(grad_scale),
        };
        std::vector<int64_t> numels(count);
        for (size_t i = 0; i < count; ++i) numels[i] = params[i].numel();
        const auto work = build_opt_work_list(numels.data(), count);
        parallel_for(0, static_cast<int64_t>(work.size()), 1,
                     [&](int64_t wb, int64_t we) {
            for (int64_t k = wb; k < we; ++k) {
                const auto& item = work[static_cast<size_t>(k)];
                const size_t li = static_cast<size_t>(item.list_index);
                fused_sgd_chunk_dispatch<scalar_t>(
                    grad_p[li], param_p[li], buf_p[li], has_momentum,
                    is_first_step, nesterov, has_scale, maximize, has_wd,
                    &sc[0], item.begin, item.end);
            }
        });
        return;
    }
    // Half/BFloat16: scalar math in float, but still scheduled through the
    // single horizontal chunk list.
    for (size_t list_index = 0; list_index < params.size(); ++list_index) {
        scalar_t* param = params[list_index].data_ptr<scalar_t>();
        scalar_t* grad = grads[list_index].data_ptr<scalar_t>();
        scalar_t* buffer = has_momentum
            ? momentum_buffers[list_index].data_ptr<scalar_t>() : nullptr;
        const int64_t n = params[list_index].numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                math_t p = static_cast<math_t>(param[i]);
                if (has_scale) {
                    g /= static_cast<math_t>(grad_scale);
                    grad[i] = static_cast<scalar_t>(g);
                }
                if (maximize) g = -g;
                if (weight_decay != 0.0) {
                    g += static_cast<math_t>(weight_decay) * p;
                }
                if (has_momentum) {
                    math_t buf = static_cast<math_t>(buffer[i]);
                    if (is_first_step) {
                        buf = g;
                    } else {
                        buf = static_cast<math_t>(momentum) * buf +
                            static_cast<math_t>(1.0 - dampening) * g;
                    }
                    buffer[i] = static_cast<scalar_t>(buf);
                    g = nesterov
                        ? g + static_cast<math_t>(momentum) * buf : buf;
                }
                param[i] = static_cast<scalar_t>(
                    p - static_cast<math_t>(lr) * g);
            }
        });
    }
}

// Fused-Adam inner range.  sc holds {beta1, lerp_weight(1-beta1), beta2,
// 1-beta2, step_size, correction2_sqrt, eps, decay, wd_factor,
// grad_scale}; implements the per-element math (lerp branch
template <typename T, typename Ops>
inline void fused_adam_vec_range(T* grad, T* param, T* m, T* v, T* maxv,
                                 bool amsgrad, bool adamw, bool coupled_wd,
                                 bool has_scale, bool maximize,
                                 const T sc[10], int64_t begin, int64_t end) {
    const auto vb2 = Ops::set1(sc[2]);
    const auto vomb2 = Ops::set1(sc[3]);
    const auto vstep = Ops::set1(sc[4]);
    const auto vc2s = Ops::set1(sc[5]);
    const auto veps = Ops::set1(sc[6]);
    const auto vscale = Ops::set1(sc[9]);
    // The lerp weight |1-beta1| < 0.5 for any practical beta1 > 0.5.
    const bool small_lerp = std::abs(static_cast<double>(sc[1])) < 0.5;
    const auto vlw = Ops::set1(sc[1]);
    const auto voneminuslw = Ops::set1(static_cast<T>(1) - sc[1]);
    constexpr int W = Ops::kW;
    int64_t i = begin;
    for (; i + W <= end; i += W) {
        auto g = Ops::load(grad + i);
        if (has_scale) { g = Ops::div(g, vscale); Ops::store(grad + i, g); }
        if (maximize) g = Ops::sub(Ops::set1(static_cast<T>(0)), g);
        auto p = Ops::load(param + i);
        if (adamw) {
            p = Ops::mul(p, Ops::set1(sc[8]));
        } else if (coupled_wd) {
            g = Ops::fmadd(Ops::set1(sc[7]), p, g);
        }
        auto mv = Ops::load(m + i);
        if (small_lerp) {
            mv = Ops::fmadd(vlw, Ops::sub(g, mv), mv);
        } else {
            mv = Ops::fnmadd(Ops::sub(g, mv), voneminuslw, g);
        }
        Ops::store(m + i, mv);
        auto vv = Ops::load(v + i);
        vv = Ops::fmadd(vb2, vv, Ops::mul(vomb2, Ops::mul(g, g)));
        Ops::store(v + i, vv);
        auto s = vv;
        if (amsgrad) {
            auto mx = Ops::vmax(Ops::load(maxv + i), vv);
            Ops::store(maxv + i, mx);
            s = mx;
        }
        const auto denom = Ops::add(Ops::div(Ops::sqrt(s), vc2s), veps);
        const auto upd = Ops::div(Ops::mul(vstep, mv), denom);
        Ops::store(param + i, Ops::sub(p, upd));
    }
    for (; i < end; ++i) {
        T g = static_cast<T>(grad[i]);
        if (has_scale) { g = g / sc[9]; grad[i] = static_cast<T>(g); }
        if (maximize) g = -g;
        T p = static_cast<T>(param[i]);
        if (adamw) {
            p = p * sc[8];
        } else if (coupled_wd) {
            g = g + sc[7] * p;
        }
        T mv;
        T old_m = static_cast<T>(m[i]);
        if (small_lerp) {
            mv = old_m + sc[1] * (g - old_m);
        } else {
            mv = g - (g - old_m) * (static_cast<T>(1) - sc[1]);
        }
        m[i] = static_cast<T>(mv);
        T vv = sc[2] * static_cast<T>(v[i]) + sc[3] * g * g;
        v[i] = static_cast<T>(vv);
        T s = vv;
        if (amsgrad) {
            T mx = static_cast<T>(maxv[i]);
            mx = mx < vv ? vv : mx;
            maxv[i] = static_cast<T>(mx);
            s = mx;
        }
        const T denom = static_cast<T>(std::sqrt(static_cast<double>(s))) /
            sc[5] + static_cast<T>(sc[6]);
        param[i] = static_cast<T>(p - sc[4] * mv / denom);
    }
}

template <typename scalar_t, typename math_t, bool adamw>
void fused_adam_math(const std::vector<Tensor>& params,
                     const std::vector<Tensor>& grads,
                     const std::vector<Tensor>& exp_avgs,
                     const std::vector<Tensor>& exp_avg_sqs,
                     const std::vector<Tensor>& max_exp_avg_sqs,
                     const std::vector<Tensor>& state_steps,
                     double lr,
                     double beta1,
                     double beta2,
                     double weight_decay,
                     double eps,
                     bool amsgrad,
                     bool maximize,
                     double grad_scale) {
    const bool has_scale = grad_scale != 1.0;
    const size_t count = params.size();
    if constexpr (std::is_same_v<scalar_t, math_t>) {
        using T = scalar_t;
        std::vector<T*> param_p(count);
        std::vector<T*> grad_p(count);
        std::vector<T*> m_p(count);
        std::vector<T*> v_p(count);
        std::vector<T*> maxv_p(count);
        // Per-tensor constants; layout documented on fused_adam_vec_range.
        std::vector<std::array<T, 10>> sc(count);
        for (size_t i = 0; i < count; ++i) {
            param_p[i] = params[i].data_ptr<T>();
            grad_p[i] = grads[i].data_ptr<T>();
            m_p[i] = exp_avgs[i].data_ptr<T>();
            v_p[i] = exp_avg_sqs[i].data_ptr<T>();
            maxv_p[i] = amsgrad ? max_exp_avg_sqs[i].data_ptr<T>() : nullptr;
            const double step = state_steps[i].item().toDouble();
            const double correction1 = 1.0 - std::pow(beta1, step);
            const double correction2 = 1.0 - std::pow(beta2, step);
            auto& s = sc[i];
            s[0] = static_cast<T>(beta1);
            s[1] = static_cast<T>(1.0 - beta1);
            s[2] = static_cast<T>(beta2);
            s[3] = static_cast<T>(1.0 - beta2);
            s[4] = static_cast<T>(lr / correction1);
            s[5] = static_cast<T>(std::sqrt(correction2));
            s[6] = static_cast<T>(eps);
            s[7] = static_cast<T>(weight_decay);
            s[8] = static_cast<T>(1.0 - lr * weight_decay);
            s[9] = static_cast<T>(grad_scale);
        }
        std::vector<int64_t> numels(count);
        for (size_t i = 0; i < count; ++i) numels[i] = params[i].numel();
        const auto work = build_opt_work_list(numels.data(), count);
        parallel_for(0, static_cast<int64_t>(work.size()), 1,
                     [&](int64_t wb, int64_t we) {
            for (int64_t k = wb; k < we; ++k) {
                const auto& item = work[static_cast<size_t>(k)];
                const size_t li = static_cast<size_t>(item.list_index);
                fused_adam_chunk_dispatch<T>(
                    grad_p[li], param_p[li], m_p[li], v_p[li], maxv_p[li],
                    amsgrad, adamw, !adamw && weight_decay != 0.0,
                    has_scale, maximize, sc[li].data(), item.begin,
                    item.end);
            }
        });
        return;
    }
    const math_t one_minus_beta1 = static_cast<math_t>(1.0 - beta1);
    const math_t beta1_value = static_cast<math_t>(beta1);
    const math_t beta2_value = static_cast<math_t>(beta2);
    const math_t one_minus_beta2 = static_cast<math_t>(1.0 - beta2);
    for (size_t list_index = 0; list_index < params.size(); ++list_index) {
        const double step = state_steps[list_index].item().toDouble();
        const double correction1 = 1.0 - std::pow(beta1, step);
        const double correction2 = 1.0 - std::pow(beta2, step);
        const math_t step_size = static_cast<math_t>(lr / correction1);
        const math_t correction2_sqrt = static_cast<math_t>(std::sqrt(correction2));
        scalar_t* param = params[list_index].data_ptr<scalar_t>();
        scalar_t* grad = grads[list_index].data_ptr<scalar_t>();
        scalar_t* exp_avg = exp_avgs[list_index].data_ptr<scalar_t>();
        scalar_t* exp_avg_sq = exp_avg_sqs[list_index].data_ptr<scalar_t>();
        scalar_t* max_exp_avg_sq = amsgrad
            ? max_exp_avg_sqs[list_index].data_ptr<scalar_t>() : nullptr;
        const int64_t n = params[list_index].numel();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                math_t p = static_cast<math_t>(param[i]);
                if (has_scale) {
                    g /= static_cast<math_t>(grad_scale);
                    grad[i] = static_cast<scalar_t>(g);
                }
                if (maximize) g = -g;
                if constexpr (adamw) {
                    p *= static_cast<math_t>(1.0 - lr * weight_decay);
                } else if (weight_decay != 0.0) {
                    g += static_cast<math_t>(weight_decay) * p;
                }

                math_t old_exp_avg = static_cast<math_t>(exp_avg[i]);
                const math_t lerp_weight = one_minus_beta1;
                if (std::abs(lerp_weight) < static_cast<math_t>(0.5)) {
                    old_exp_avg += lerp_weight * (g - old_exp_avg);
                } else {
                    old_exp_avg = g - (g - old_exp_avg) *
                        (static_cast<math_t>(1.0) - lerp_weight);
                }
                const math_t old_exp_avg_sq = static_cast<math_t>(exp_avg_sq[i]);
                const math_t new_exp_avg_sq = beta2_value * old_exp_avg_sq +
                    one_minus_beta2 * g * g;
                exp_avg[i] = static_cast<scalar_t>(old_exp_avg);
                exp_avg_sq[i] = static_cast<scalar_t>(new_exp_avg_sq);

                math_t second_moment = new_exp_avg_sq;
                if (amsgrad) {
                    math_t max_value = static_cast<math_t>(max_exp_avg_sq[i]);
                    max_value = std::max(max_value, second_moment);
                    max_exp_avg_sq[i] = static_cast<scalar_t>(max_value);
                    second_moment = max_value;
                }
                const math_t denom = static_cast<math_t>(std::sqrt(
                    static_cast<double>(second_moment))) / correction2_sqrt +
                    static_cast<math_t>(eps);
                param[i] = static_cast<scalar_t>(
                    p - step_size * old_exp_avg / denom);
            }
        });
    }
}

template <typename scalar_t, typename math_t>
void fused_adagrad_math(const std::vector<Tensor>& params,
                        const std::vector<Tensor>& grads,
                        const std::vector<Tensor>& state_sums,
                        const std::vector<Tensor>& state_steps,
                        double lr,
                        double lr_decay,
                        double weight_decay,
                        double eps,
                        bool maximize,
                        double grad_scale) {
    const bool has_scale = grad_scale != 1.0;
    const size_t count = params.size();
    std::vector<scalar_t*> param_p(count);
    std::vector<scalar_t*> grad_p(count);
    std::vector<scalar_t*> sum_p(count);
    std::vector<math_t> clr(count);
    for (size_t i = 0; i < count; ++i) {
        param_p[i] = params[i].data_ptr<scalar_t>();
        grad_p[i] = grads[i].data_ptr<scalar_t>();
        sum_p[i] = state_sums[i].data_ptr<scalar_t>();
        const double step = state_steps[i].item().toDouble();
        clr[i] = static_cast<math_t>(lr / (1.0 + (step - 1.0) * lr_decay));
    }
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) numels[i] = params[i].numel();
    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t wb, int64_t we) {
        for (int64_t k = wb; k < we; ++k) {
            const auto& item = work[static_cast<size_t>(k)];
            const size_t li = static_cast<size_t>(item.list_index);
            scalar_t* param = param_p[li];
            scalar_t* grad = grad_p[li];
            scalar_t* state_sum = sum_p[li];
            const math_t step_clr = clr[li];
            const math_t sc[4] = {
                static_cast<math_t>(step_clr),
                static_cast<math_t>(eps),
                static_cast<math_t>(weight_decay),
                static_cast<math_t>(grad_scale),
            };
            fused_adagrad_chunk_dispatch<scalar_t, math_t>(
                grad, param, state_sum, has_scale, maximize,
                weight_decay != 0.0, sc, item.begin, item.end);
        }
    });
}

template <typename F>
void dispatch_fused_dtype(const std::vector<Tensor>& params,
                          const char* op_name,
                          F&& fn) {
    if (params.empty()) return;
    switch (params[0].dtype()) {
        case DType::Float16: fn.template operator()<Half, float>(); break;
        case DType::BFloat16: fn.template operator()<BFloat16, float>(); break;
        case DType::Float32: fn.template operator()<float, float>(); break;
        case DType::Float64: fn.template operator()<double, double>(); break;
        default:
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": unsupported fused optimizer dtype");
    }
}

void fused_sgd_cpu_impl(std::vector<Tensor> params,
                        const std::vector<Tensor>& grads,
                        const std::vector<Tensor>& momentum_buffers,
                        double lr,
                        double momentum,
                        double dampening,
                        double weight_decay,
                        bool nesterov,
                        bool maximize,
                        bool is_first_step,
                        const std::optional<Tensor>& grad_scale,
                        const std::optional<Tensor>& found_inf) {
    validate_fused_pairs(params, grads, "_fused_sgd_");
    if (fused_found_inf(found_inf)) return;
    if (momentum == 0.0) {
        if (!momentum_buffers.empty()) {
            TP_THROW(ValueError, "_fused_sgd_: momentum buffer list must be empty when momentum is zero");
        }
    } else {
        validate_fused_state(params, momentum_buffers, true, "_fused_sgd_");
    }
    const double scale = fused_grad_scale(grad_scale);
    dispatch_fused_dtype(params, "_fused_sgd_", [&]<typename scalar_t, typename math_t>() {
        fused_sgd_math<scalar_t, math_t>(params, grads, momentum_buffers, lr,
            momentum, dampening, weight_decay, nesterov, maximize,
            is_first_step, scale);
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cpu_impl(std::vector<Tensor> params,
                         const std::vector<Tensor>& grads,
                         const std::vector<Tensor>& exp_avgs,
                         const std::vector<Tensor>& exp_avg_sqs,
                         const std::vector<Tensor>& max_exp_avg_sqs,
                         const std::vector<Tensor>& state_steps,
                         double lr,
                         double beta1,
                         double beta2,
                         double weight_decay,
                         double eps,
                         bool amsgrad,
                         bool maximize,
                         bool adamw,
                         const std::optional<Tensor>& grad_scale,
                         const std::optional<Tensor>& found_inf) {
    validate_fused_pairs(params, grads, adamw ? "_fused_adamw_" : "_fused_adam_");
    if (fused_found_inf(found_inf)) return;
    const char* op_name = adamw ? "_fused_adamw_" : "_fused_adam_";
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_avg_sqs, true, op_name);
    validate_fused_state(params, max_exp_avg_sqs, amsgrad, op_name);
    if (!amsgrad && !max_exp_avg_sqs.empty()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": max_exp_avg_sqs must be empty when amsgrad is false");
    }
    validate_fused_steps(params, state_steps, op_name);
    const double scale = fused_grad_scale(grad_scale);
    dispatch_fused_dtype(params, op_name, [&]<typename scalar_t, typename math_t>() {
        if (adamw) {
            fused_adam_math<scalar_t, math_t, true>(params, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                weight_decay, eps, amsgrad, maximize, scale);
        } else {
            fused_adam_math<scalar_t, math_t, false>(params, grads, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                weight_decay, eps, amsgrad, maximize, scale);
        }
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adagrad_cpu_impl(std::vector<Tensor> params,
                            const std::vector<Tensor>& grads,
                            const std::vector<Tensor>& state_sums,
                            const std::vector<Tensor>& state_steps,
                            double lr,
                            double lr_decay,
                            double weight_decay,
                            double eps,
                            bool maximize,
                            const std::optional<Tensor>& grad_scale,
                            const std::optional<Tensor>& found_inf) {
    validate_fused_pairs(params, grads, "_fused_adagrad_");
    if (fused_found_inf(found_inf)) return;
    validate_fused_state(params, state_sums, true, "_fused_adagrad_");
    validate_fused_steps(params, state_steps, "_fused_adagrad_");
    const double scale = fused_grad_scale(grad_scale);
    dispatch_fused_dtype(params, "_fused_adagrad_", [&]<typename scalar_t, typename math_t>() {
        fused_adagrad_math<scalar_t, math_t>(params, grads, state_sums,
            state_steps, lr, lr_decay, weight_decay, eps, maximize, scale);
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cpu(std::vector<Tensor> params,
                    std::vector<Tensor> grads,
                    std::vector<Tensor> exp_avgs,
                    std::vector<Tensor> exp_avg_sqs,
                    std::vector<Tensor> max_exp_avg_sqs,
                    const std::vector<Tensor>& state_steps,
                    double lr, double beta1, double beta2, double weight_decay,
                    double eps, bool amsgrad, bool maximize,
                    const std::optional<Tensor>& grad_scale,
                    const std::optional<Tensor>& found_inf,
                    bool exact) {
    (void)exact;
    fused_adam_cpu_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, false, grad_scale, found_inf);
}

void fused_adam_tensor_lr_cpu(std::vector<Tensor> params,
                              std::vector<Tensor> grads,
                              std::vector<Tensor> exp_avgs,
                              std::vector<Tensor> exp_avg_sqs,
                              std::vector<Tensor> max_exp_avg_sqs,
                              const std::vector<Tensor>& state_steps,
                              const Tensor& lr, double beta1, double beta2,
                              double weight_decay, double eps, bool amsgrad,
                              bool maximize,
                              const std::optional<Tensor>& grad_scale,
                              const std::optional<Tensor>& found_inf,
                              bool exact) {
    (void)exact;
    fused_adam_cpu_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr.item().toDouble(), beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, false, grad_scale, found_inf);
}

void fused_adamw_cpu(std::vector<Tensor> params,
                     std::vector<Tensor> grads,
                     std::vector<Tensor> exp_avgs,
                     std::vector<Tensor> exp_avg_sqs,
                     std::vector<Tensor> max_exp_avg_sqs,
                     const std::vector<Tensor>& state_steps,
                     double lr, double beta1, double beta2, double weight_decay,
                     double eps, bool amsgrad, bool maximize,
                     const std::optional<Tensor>& grad_scale,
                     const std::optional<Tensor>& found_inf,
                     bool exact) {
    (void)exact;
    fused_adam_cpu_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, true, grad_scale, found_inf);
}

void fused_adamw_tensor_lr_cpu(std::vector<Tensor> params,
                               std::vector<Tensor> grads,
                               std::vector<Tensor> exp_avgs,
                               std::vector<Tensor> exp_avg_sqs,
                               std::vector<Tensor> max_exp_avg_sqs,
                               const std::vector<Tensor>& state_steps,
                               const Tensor& lr, double beta1, double beta2,
                               double weight_decay, double eps, bool amsgrad,
                               bool maximize,
                               const std::optional<Tensor>& grad_scale,
                               const std::optional<Tensor>& found_inf,
                               bool exact) {
    (void)exact;
    fused_adam_cpu_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr.item().toDouble(), beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, true, grad_scale, found_inf);
}

void fused_sgd_cpu(std::vector<Tensor> params,
                   std::vector<Tensor> grads,
                   std::vector<Tensor> momentum_buffers,
                   double weight_decay, double momentum, double lr,
                   double dampening, bool nesterov, bool maximize,
                   bool is_first_step, const std::optional<Tensor>& grad_scale,
                   const std::optional<Tensor>& found_inf) {
    fused_sgd_cpu_impl(std::move(params), grads, momentum_buffers,
        lr, momentum, dampening,
        weight_decay, nesterov, maximize, is_first_step,
        grad_scale, found_inf);
}

void fused_sgd_tensor_lr_cpu(std::vector<Tensor> params,
                             std::vector<Tensor> grads,
                             std::vector<Tensor> momentum_buffers,
                             double weight_decay, double momentum,
                             const Tensor& lr, double dampening, bool nesterov,
                             bool maximize, bool is_first_step,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf) {
    fused_sgd_cpu_impl(std::move(params), grads, momentum_buffers,
        lr.item().toDouble(), momentum, dampening,
        weight_decay, nesterov, maximize, is_first_step,
        grad_scale, found_inf);
}

void fused_adagrad_cpu(std::vector<Tensor> params,
                       std::vector<Tensor> grads,
                       std::vector<Tensor> state_sums,
                       std::vector<Tensor> state_steps,
                       double lr, double lr_decay, double weight_decay,
                       double eps, bool maximize,
                       const std::optional<Tensor>& grad_scale,
                       const std::optional<Tensor>& found_inf) {
    fused_adagrad_cpu_impl(std::move(params), grads, state_sums, state_steps,
        lr, lr_decay, weight_decay,
        eps, maximize, grad_scale, found_inf);
}

void fused_adagrad_tensor_lr_cpu(std::vector<Tensor> params,
                                 std::vector<Tensor> grads,
                                 std::vector<Tensor> state_sums,
                                 std::vector<Tensor> state_steps,
                                 const Tensor& lr, double lr_decay,
                                 double weight_decay, double eps, bool maximize,
                                 const std::optional<Tensor>& grad_scale,
                                 const std::optional<Tensor>& found_inf) {
    fused_adagrad_cpu_impl(std::move(params), grads, state_sums, state_steps,
        lr.item().toDouble(), lr_decay, weight_decay,
        eps, maximize, grad_scale, found_inf);
}

// RMSprop is exposed as a single native CPU pass for the non-complex,
// from five-to-eight foreach calls; keeping the state-step increment and the
// element update in the same horizontal schedule removes those intermediate
// tensor lists and barriers while preserving the scalar operation order.
double increment_cpu_step(Tensor& step, const char* op_name) {
    if (step.dtype() == DType::Float32) {
        float* value = step.data_ptr<float>();
        *value += 1.0f;
        return static_cast<double>(*value);
    }
    if (step.dtype() == DType::Float64) {
        double* value = step.data_ptr<double>();
        *value += 1.0;
        return *value;
    }
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": state_steps must be float32 or float64 on CPU");
}

template <typename scalar_t, typename math_t>
void fused_rmsprop_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& grad_avgs,
        const std::vector<Tensor>& momentum_buffers,
        std::vector<Tensor>& state_steps,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool centered, bool maximize) {
    const size_t count = params.size();
    const bool has_weight_decay = weight_decay != 0.0;
    const bool has_momentum = momentum != 0.0;

    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> square_avg_ptrs(count);
    std::vector<scalar_t*> grad_avg_ptrs(count, nullptr);
    std::vector<scalar_t*> momentum_ptrs(count, nullptr);
    std::vector<std::array<math_t, 6>> constants(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        square_avg_ptrs[i] = square_avgs[i].data_ptr<scalar_t>();
        if (centered) grad_avg_ptrs[i] = grad_avgs[i].data_ptr<scalar_t>();
        if (has_momentum) {
            momentum_ptrs[i] = momentum_buffers[i].data_ptr<scalar_t>();
        }
        increment_cpu_step(state_steps[i], "_fused_rmsprop_");
        constants[i] = {
            static_cast<math_t>(alpha),
            static_cast<math_t>(1.0 - alpha),
            static_cast<math_t>(lr),
            static_cast<math_t>(eps),
            static_cast<math_t>(weight_decay),
            static_cast<math_t>(momentum),
        };
        numels[i] = params[i].numel();
    }

    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const auto& sc = constants[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* square_avg = square_avg_ptrs[li];
            scalar_t* grad_avg = grad_avg_ptrs[li];
            scalar_t* momentum_buffer = momentum_ptrs[li];
            fused_rmsprop_chunk_dispatch<scalar_t, math_t>(
                grad, param, square_avg, grad_avg, momentum_buffer,
                centered, has_momentum, has_weight_decay, maximize, sc.data(),
                item.begin, item.end);
        }
    });
}

void fused_rmsprop_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> square_avgs, std::vector<Tensor> grad_avgs,
        std::vector<Tensor> momentum_buffers, std::vector<Tensor> state_steps,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool centered, bool maximize) {
    validate_fused_pairs(params, grads, "_fused_rmsprop_");
    if (params.empty()) return;
    validate_fused_state(params, square_avgs, true, "_fused_rmsprop_");
    validate_fused_state(params, grad_avgs, centered, "_fused_rmsprop_");
    validate_fused_state(params, momentum_buffers, momentum != 0.0,
                         "_fused_rmsprop_");
    validate_fused_steps(params, state_steps, "_fused_rmsprop_");
    if (params[0].dtype() == DType::Float16) {
        fused_rmsprop_math<Half, float>(params, grads, square_avgs, grad_avgs,
            momentum_buffers, state_steps, lr, alpha, eps, weight_decay,
            momentum, centered, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_rmsprop_math<BFloat16, float>(params, grads, square_avgs,
            grad_avgs, momentum_buffers, state_steps, lr, alpha, eps,
            weight_decay, momentum, centered, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_rmsprop_math<float, float>(params, grads, square_avgs, grad_avgs,
            momentum_buffers, state_steps, lr, alpha, eps, weight_decay,
            momentum, centered, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_rmsprop_math<double, double>(params, grads, square_avgs,
            grad_avgs, momentum_buffers, state_steps, lr, alpha, eps,
            weight_decay, momentum, centered, maximize);
    } else {
        TP_THROW(NotImplementedError,
                 "_fused_rmsprop_: unsupported floating dtype");
    }
    for (const Tensor& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
}

template <typename math_t>
math_t read_optimizer_scalar(const Tensor& value, const char* op_name) {
    if (value.dtype() == DType::Float32) {
        return static_cast<math_t>(*value.data_ptr<float>());
    }
    if (value.dtype() == DType::Float64) {
        return static_cast<math_t>(*value.data_ptr<double>());
    }
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": scalar optimizer state must be float32 or float64");
}

void validate_optimizer_scalar_states(const std::vector<Tensor>& params,
                                      const std::vector<Tensor>& states,
                                      const char* op_name) {
    if (states.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": scalar optimizer state list must match parameter list");
    }
    for (const Tensor& state : states) {
        if (!state.defined() || !state.is_contiguous() || state.numel() != 1 ||
            state.device() != Device(DeviceType::CPU) ||
            (state.dtype() != DType::Float32 &&
             state.dtype() != DType::Float64)) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": scalar optimizer states must be singleton CPU float32/float64 tensors");
        }
    }
}

template <typename math_t>
void write_optimizer_scalar(Tensor& value, math_t result, const char* op_name) {
    if (value.dtype() == DType::Float32) {
        *value.data_ptr<float>() = static_cast<float>(result);
        return;
    }
    if (value.dtype() == DType::Float64) {
        *value.data_ptr<double>() = static_cast<double>(result);
        return;
    }
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": scalar optimizer state must be float32 or float64");
}

template <typename scalar_t, typename math_t>
void fused_adadelta_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& acc_deltas,
        std::vector<Tensor>& state_steps,
        double lr, double rho, double eps, double weight_decay,
        bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> square_avg_ptrs(count);
    std::vector<scalar_t*> acc_delta_ptrs(count);
    std::vector<std::array<math_t, 5>> constants(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        square_avg_ptrs[i] = square_avgs[i].data_ptr<scalar_t>();
        acc_delta_ptrs[i] = acc_deltas[i].data_ptr<scalar_t>();
        increment_cpu_step(state_steps[i], "_fused_adadelta_");
        constants[i] = {
            static_cast<math_t>(rho), static_cast<math_t>(1.0 - rho),
            static_cast<math_t>(lr), static_cast<math_t>(eps),
            static_cast<math_t>(weight_decay),
        };
        numels[i] = params[i].numel();
    }

    const bool has_weight_decay = weight_decay != 0.0;
    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const auto& sc = constants[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* square_avg = square_avg_ptrs[li];
            scalar_t* acc_delta = acc_delta_ptrs[li];
            fused_adadelta_chunk_dispatch<scalar_t, math_t>(
                grad, param, square_avg, acc_delta, has_weight_decay,
                maximize, sc.data(), item.begin, item.end);
        }
    });
}

void fused_adadelta_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> square_avgs, std::vector<Tensor> acc_deltas,
        std::vector<Tensor> state_steps, double lr, double rho, double eps,
        double weight_decay, bool maximize) {
    validate_fused_pairs(params, grads, "_fused_adadelta_");
    if (params.empty()) return;
    validate_fused_state(params, square_avgs, true, "_fused_adadelta_");
    validate_fused_state(params, acc_deltas, true, "_fused_adadelta_");
    validate_fused_steps(params, state_steps, "_fused_adadelta_");
    if (params[0].dtype() == DType::Float16) {
        fused_adadelta_math<Half, float>(params, grads, square_avgs,
            acc_deltas, state_steps, lr, rho, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_adadelta_math<BFloat16, float>(params, grads, square_avgs,
            acc_deltas, state_steps, lr, rho, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_adadelta_math<float, float>(params, grads, square_avgs,
            acc_deltas, state_steps, lr, rho, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_adadelta_math<double, double>(params, grads, square_avgs,
            acc_deltas, state_steps, lr, rho, eps, weight_decay, maximize);
    } else {
        TP_THROW(NotImplementedError,
                 "_fused_adadelta_: unsupported floating dtype");
    }
    for (const Tensor& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
}

template <typename scalar_t, typename math_t>
void fused_adamax_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_infs,
        std::vector<Tensor>& state_steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> exp_avg_ptrs(count);
    std::vector<scalar_t*> exp_inf_ptrs(count);
    std::vector<std::array<math_t, 6>> constants(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        exp_avg_ptrs[i] = exp_avgs[i].data_ptr<scalar_t>();
        exp_inf_ptrs[i] = exp_infs[i].data_ptr<scalar_t>();
        const double step = increment_cpu_step(state_steps[i], "_fused_adamax_");
        const double correction = 1.0 - std::pow(beta1, step);
        constants[i] = {
            static_cast<math_t>(beta1), static_cast<math_t>(1.0 - beta1),
            static_cast<math_t>(beta2), static_cast<math_t>(eps),
            static_cast<math_t>(weight_decay),
            static_cast<math_t>(-lr / correction),
        };
        numels[i] = params[i].numel();
    }

    const bool has_weight_decay = weight_decay != 0.0;
    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const auto& sc = constants[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* exp_avg = exp_avg_ptrs[li];
            scalar_t* exp_inf = exp_inf_ptrs[li];
            for (int64_t i = item.begin; i < item.end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                if (maximize) g = optimizer_round<scalar_t, math_t>(-g);
                math_t p = static_cast<math_t>(param[i]);
                if (has_weight_decay) {
                    g = optimizer_round<scalar_t, math_t>(g + sc[4] * p);
                }

                const math_t old_avg = static_cast<math_t>(exp_avg[i]);
                const math_t avg = optimizer_lerp<scalar_t, math_t>(
                    old_avg, g, sc[1]);
                exp_avg[i] = static_cast<scalar_t>(avg);
                math_t inf = optimizer_round<scalar_t, math_t>(
                    sc[2] * static_cast<math_t>(exp_inf[i]));
                math_t candidate = optimizer_round<scalar_t, math_t>(
                    static_cast<math_t>(std::abs(g)));
                candidate = optimizer_round<scalar_t, math_t>(
                    candidate + sc[3]);
                if (inf < candidate) inf = candidate;
                exp_inf[i] = static_cast<scalar_t>(inf);
                param[i] = static_cast<scalar_t>(optimizer_addcdiv<
                    scalar_t, math_t>(p, avg, inf, sc[5]));
            }
        }
    });
}

void fused_adamax_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_infs,
        std::vector<Tensor> state_steps, double lr, double beta1,
        double beta2, double eps, double weight_decay, bool maximize) {
    validate_fused_pairs(params, grads, "_fused_adamax_");
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, "_fused_adamax_");
    validate_fused_state(params, exp_infs, true, "_fused_adamax_");
    validate_fused_steps(params, state_steps, "_fused_adamax_");
    if (params[0].dtype() == DType::Float16) {
        fused_adamax_math<Half, float>(params, grads, exp_avgs, exp_infs,
            state_steps, lr, beta1, beta2, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_adamax_math<BFloat16, float>(params, grads, exp_avgs, exp_infs,
            state_steps, lr, beta1, beta2, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_adamax_math<float, float>(params, grads, exp_avgs, exp_infs,
            state_steps, lr, beta1, beta2, eps, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_adamax_math<double, double>(params, grads, exp_avgs, exp_infs,
            state_steps, lr, beta1, beta2, eps, weight_decay, maximize);
    } else {
        TP_THROW(NotImplementedError,
                 "_fused_adamax_: unsupported floating dtype");
    }
    for (const Tensor& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
}

template <typename scalar_t, typename math_t>
void fused_asgd_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& axs,
        std::vector<Tensor>& mus,
        std::vector<Tensor>& etas,
        std::vector<Tensor>& state_steps,
        double lr, double lambd, double t0, double alpha,
        double weight_decay, bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> ax_ptrs(count);
    std::vector<math_t> eta_values(count);
    std::vector<math_t> mu_values(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        ax_ptrs[i] = axs[i].data_ptr<scalar_t>();
        const double step = increment_cpu_step(state_steps[i], "_fused_asgd_");
        const math_t eta = read_optimizer_scalar<math_t>(etas[i],
                                                         "_fused_asgd_");
        const math_t mu = read_optimizer_scalar<math_t>(mus[i],
                                                        "_fused_asgd_");
        eta_values[i] = eta;
        mu_values[i] = mu;
        const math_t new_eta = static_cast<math_t>(lr / std::pow(
            1.0 + lambd * lr * step, alpha));
        const math_t new_mu = static_cast<math_t>(1.0 / std::max(
            1.0, step - t0));
        write_optimizer_scalar(etas[i], new_eta, "_fused_asgd_");
        write_optimizer_scalar(mus[i], new_mu, "_fused_asgd_");
        numels[i] = params[i].numel();
    }

    const bool has_weight_decay = weight_decay != 0.0;
    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const math_t eta = eta_values[li];
            const math_t mu = mu_values[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* ax = ax_ptrs[li];
            for (int64_t i = item.begin; i < item.end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                if (maximize) g = -g;
                math_t p = static_cast<math_t>(param[i]);
                if (has_weight_decay) g += static_cast<math_t>(weight_decay) * p;
                p = p * (static_cast<math_t>(1) -
                         static_cast<math_t>(lambd) * eta) - eta * g;
                param[i] = static_cast<scalar_t>(p);

                math_t average = static_cast<math_t>(ax[i]);
                if (mu == static_cast<math_t>(1)) {
                    average = p;
                } else {
                    average += (p - average) * mu;
                }
                ax[i] = static_cast<scalar_t>(average);
            }
        }
    });
}

void fused_asgd_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> axs, std::vector<Tensor> mus,
        std::vector<Tensor> etas, std::vector<Tensor> state_steps,
        double lr, double lambd, double t0, double alpha,
        double weight_decay, bool maximize) {
    validate_fused_pairs(params, grads, "_fused_asgd_");
    if (params.empty()) return;
    validate_fused_state(params, axs, true, "_fused_asgd_");
    validate_optimizer_scalar_states(params, mus, "_fused_asgd_");
    validate_optimizer_scalar_states(params, etas, "_fused_asgd_");
    validate_fused_steps(params, state_steps, "_fused_asgd_");
    if (params[0].dtype() == DType::Float16) {
        fused_asgd_math<Half, float>(params, grads, axs, mus, etas,
            state_steps, lr, lambd, t0, alpha, weight_decay, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_asgd_math<BFloat16, float>(params, grads, axs, mus, etas,
            state_steps, lr, lambd, t0, alpha, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_asgd_math<float, float>(params, grads, axs, mus, etas,
            state_steps, lr, lambd, t0, alpha, weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_asgd_math<double, double>(params, grads, axs, mus, etas,
            state_steps, lr, lambd, t0, alpha, weight_decay, maximize);
    } else {
        TP_THROW(NotImplementedError,
                 "_fused_asgd_: unsupported floating dtype");
    }
    for (const Tensor& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
}

template <typename scalar_t, typename math_t>
void fused_rprop_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& prevs,
        const std::vector<Tensor>& step_sizes,
        std::vector<Tensor>& state_steps,
        double step_size_min, double step_size_max, double etaminus,
        double etaplus, bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> prev_ptrs(count);
    std::vector<scalar_t*> step_size_ptrs(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        prev_ptrs[i] = prevs[i].data_ptr<scalar_t>();
        step_size_ptrs[i] = step_sizes[i].data_ptr<scalar_t>();
        increment_cpu_step(state_steps[i], "_fused_rprop_");
        numels[i] = params[i].numel();
    }

    const math_t vmin = static_cast<math_t>(step_size_min);
    const math_t vmax = static_cast<math_t>(step_size_max);
    const math_t vminus = static_cast<math_t>(etaminus);
    const math_t vplus = static_cast<math_t>(etaplus);
    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* prev = prev_ptrs[li];
            scalar_t* step_size = step_size_ptrs[li];
            for (int64_t i = item.begin; i < item.end; ++i) {
                const math_t raw_grad = static_cast<math_t>(grad[i]);
                math_t product = optimizer_round<scalar_t, math_t>(
                    raw_grad * static_cast<math_t>(prev[i]));
                if (maximize) {
                    product = optimizer_round<scalar_t, math_t>(-product);
                }
                math_t sign = static_cast<math_t>(1);
                if (product > static_cast<math_t>(0)) sign = vplus;
                else if (product < static_cast<math_t>(0)) sign = vminus;

                // The foreach sign tensor has the parameter dtype, so both
                // eta assignments and the subsequent step-size multiply are
                // rounded at their individual operation boundaries.
                sign = sign == vplus
                    ? optimizer_round<scalar_t, math_t>(vplus)
                    : sign == vminus
                        ? optimizer_round<scalar_t, math_t>(vminus)
                        : optimizer_round<scalar_t, math_t>(math_t(1));
                math_t next_step = optimizer_round<scalar_t, math_t>(
                    static_cast<math_t>(step_size[i]) * sign);
                next_step = std::min(vmax, std::max(vmin, next_step));
                next_step = optimizer_round<scalar_t, math_t>(next_step);
                step_size[i] = static_cast<scalar_t>(next_step);
                math_t stored_grad = maximize
                    ? optimizer_round<scalar_t, math_t>(-raw_grad)
                    : raw_grad;
                const math_t masked_grad = sign ==
                        optimizer_round<scalar_t, math_t>(vminus)
                    ? math_t(0) : stored_grad;
                const math_t grad_sign =
                    masked_grad > static_cast<math_t>(0) ? static_cast<math_t>(1) :
                    masked_grad < static_cast<math_t>(0) ? static_cast<math_t>(-1) :
                    static_cast<math_t>(0);
                // a direction reversal therefore becomes zero for the next
                // iteration.  Keeping the raw gradient here changes the
                // following sign product and diverges after one step.
                prev[i] = static_cast<scalar_t>(masked_grad);
                param[i] = static_cast<scalar_t>(optimizer_addcmul<
                    scalar_t, math_t>(static_cast<math_t>(param[i]),
                                      grad_sign, next_step, math_t(-1)));
            }
        }
    });
}

void fused_rprop_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> prevs, std::vector<Tensor> step_sizes,
        std::vector<Tensor> state_steps, double step_size_min,
        double step_size_max, double etaminus, double etaplus,
        bool maximize) {
    validate_fused_pairs(params, grads, "_fused_rprop_");
    if (params.empty()) return;
    validate_fused_state(params, prevs, true, "_fused_rprop_");
    validate_fused_state(params, step_sizes, true, "_fused_rprop_");
    validate_fused_steps(params, state_steps, "_fused_rprop_");
    if (params[0].dtype() == DType::Float16) {
        fused_rprop_math<Half, float>(params, grads, prevs, step_sizes,
            state_steps, step_size_min, step_size_max, etaminus, etaplus,
            maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_rprop_math<BFloat16, float>(params, grads, prevs, step_sizes,
            state_steps, step_size_min, step_size_max, etaminus, etaplus,
            maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_rprop_math<float, float>(params, grads, prevs, step_sizes,
            state_steps, step_size_min, step_size_max, etaminus, etaplus,
            maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_rprop_math<double, double>(params, grads, prevs, step_sizes,
            state_steps, step_size_min, step_size_max, etaminus, etaplus,
            maximize);
    } else {
        TP_THROW(NotImplementedError,
                 "_fused_rprop_: unsupported floating dtype");
    }
    for (const Tensor& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
}

template <typename scalar_t, typename math_t>
void fused_nadam_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        std::vector<Tensor>& mu_products,
        std::vector<Tensor>& state_steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, double momentum_decay,
        bool decoupled_weight_decay, bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> exp_avg_ptrs(count);
    std::vector<scalar_t*> exp_avg_sq_ptrs(count);
    std::vector<std::array<math_t, 10>> constants(count);
    std::vector<int64_t> numels(count);

    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        exp_avg_ptrs[i] = exp_avgs[i].data_ptr<scalar_t>();
        exp_avg_sq_ptrs[i] = exp_avg_sqs[i].data_ptr<scalar_t>();
        const double step = increment_cpu_step(state_steps[i], "_fused_nadam_");
        const math_t mu = static_cast<math_t>(beta1 *
            (1.0 - 0.5 * std::pow(0.96, step * momentum_decay)));
        const math_t mu_next = static_cast<math_t>(beta1 *
            (1.0 - 0.5 * std::pow(0.96, (step + 1.0) * momentum_decay)));
        const math_t mu_product = static_cast<math_t>(
            read_optimizer_scalar<math_t>(mu_products[i], "_fused_nadam_"));
        const math_t next_mu_product = mu_product * mu;
        write_optimizer_scalar(mu_products[i], next_mu_product, "_fused_nadam_");
        const math_t one_minus_beta1 = static_cast<math_t>(1.0 - beta1);
        const math_t one_minus_beta2 = static_cast<math_t>(1.0 - beta2);
        constants[i] = {
            static_cast<math_t>(lr), static_cast<math_t>(beta1),
            static_cast<math_t>(beta2), static_cast<math_t>(eps),
            static_cast<math_t>(weight_decay), one_minus_beta1,
            one_minus_beta2, static_cast<math_t>(momentum_decay),
        };
        // Store the two per-tensor coefficients in the last two slots after
        // computing them from the updated mu product.  The first six slots
        // above are retained to keep the operation readable in the loop.
        constants[i][7] = -static_cast<math_t>(lr) *
            (math_t(1) - mu) / (math_t(1) - next_mu_product);
        constants[i][8] = -static_cast<math_t>(lr) * mu_next /
            (math_t(1) - next_mu_product * mu_next);
        constants[i][9] = static_cast<math_t>(std::sqrt(
            1.0 - std::pow(static_cast<double>(beta2), step)));
        numels[i] = params[i].numel();
    }

    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const auto& sc = constants[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* exp_avg = exp_avg_ptrs[li];
            scalar_t* exp_avg_sq = exp_avg_sq_ptrs[li];
            const math_t weight_decay_value = sc[4];
            fused_nadam_chunk_dispatch<scalar_t, math_t>(
                grad, param, exp_avg, exp_avg_sq,
                weight_decay_value != math_t(0), decoupled_weight_decay,
                maximize, sc.data(), item.begin, item.end);
        }
    });
}

void fused_nadam_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs,
        std::vector<Tensor> mu_products, std::vector<Tensor> state_steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, double momentum_decay,
        bool decoupled_weight_decay, bool maximize) {
    const char* op_name = "_fused_nadam_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_avg_sqs, true, op_name);
    validate_optimizer_scalar_states(params, mu_products, op_name);
    validate_fused_steps(params, state_steps, op_name);
    if (params[0].dtype() == DType::Float16) {
        fused_nadam_math<Half, float>(params, grads, exp_avgs, exp_avg_sqs,
            mu_products, state_steps, lr, beta1, beta2, eps, weight_decay,
            momentum_decay, decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_nadam_math<BFloat16, float>(params, grads, exp_avgs,
            exp_avg_sqs, mu_products, state_steps, lr, beta1, beta2, eps,
            weight_decay, momentum_decay, decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_nadam_math<float, float>(params, grads, exp_avgs, exp_avg_sqs,
            mu_products, state_steps, lr, beta1, beta2, eps, weight_decay,
            momentum_decay, decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_nadam_math<double, double>(params, grads, exp_avgs,
            exp_avg_sqs, mu_products, state_steps, lr, beta1, beta2, eps,
            weight_decay, momentum_decay, decoupled_weight_decay, maximize);
    } else {
        TP_THROW(NotImplementedError, std::string(op_name) +
            ": unsupported floating dtype");
    }
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

template <typename scalar_t, typename math_t>
void fused_radam_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        std::vector<Tensor>& state_steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool decoupled_weight_decay, bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> exp_avg_ptrs(count);
    std::vector<scalar_t*> exp_avg_sq_ptrs(count);
    std::vector<std::array<math_t, 8>> constants(count);
    std::vector<int64_t> numels(count);

    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        exp_avg_ptrs[i] = exp_avgs[i].data_ptr<scalar_t>();
        exp_avg_sq_ptrs[i] = exp_avg_sqs[i].data_ptr<scalar_t>();
        const double step = increment_cpu_step(state_steps[i], "_fused_radam_");
        const double bc1 = 1.0 - std::pow(beta1, step);
        const double bc2 = 1.0 - std::pow(beta2, step);
        const double rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        const double rho_t = rho_inf - 2.0 * step * std::pow(beta2, step) / bc2;
        const math_t unrectified = static_cast<math_t>(-lr / bc1);
        const math_t rectified = rho_t > 5.0
            ? static_cast<math_t>(-lr * std::sqrt(bc2) * std::sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)) / bc1)
            : math_t(0);
        constants[i] = {
            static_cast<math_t>(beta1), static_cast<math_t>(beta2),
            static_cast<math_t>(eps), static_cast<math_t>(weight_decay),
            static_cast<math_t>(lr), static_cast<math_t>(1.0 - beta1),
            unrectified, rectified,
        };
        numels[i] = params[i].numel();
    }

    const auto work = build_opt_work_list(numels.data(), count);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const auto& sc = constants[li];
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            scalar_t* exp_avg = exp_avg_ptrs[li];
            scalar_t* exp_avg_sq = exp_avg_sq_ptrs[li];
            const math_t beta1_value = sc[0];
            const math_t beta2_value = sc[1];
            const math_t eps_value = sc[2];
            const math_t weight_decay_value = sc[3];
            const math_t lr_value = sc[4];
            const math_t one_minus_beta1 = sc[5];
            for (int64_t i = item.begin; i < item.end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                if (maximize) g = optimizer_round<scalar_t, math_t>(-g);
                math_t p = static_cast<math_t>(param[i]);
                if (weight_decay_value != math_t(0)) {
                    if (decoupled_weight_decay) {
                        p = optimizer_round<scalar_t, math_t>(
                            p * (math_t(1) - lr_value * weight_decay_value));
                    } else {
                        g = optimizer_round<scalar_t, math_t>(
                            g + weight_decay_value * p);
                    }
                }
                const math_t old_m = static_cast<math_t>(exp_avg[i]);
                const math_t m = optimizer_lerp<scalar_t, math_t>(
                    old_m, g, one_minus_beta1);
                exp_avg[i] = static_cast<scalar_t>(m);
                math_t v = optimizer_round<scalar_t, math_t>(
                    beta2_value * static_cast<math_t>(exp_avg_sq[i]));
                v = optimizer_addcmul<scalar_t, math_t>(
                    v, g, g, math_t(1) - beta2_value);
                exp_avg_sq[i] = static_cast<scalar_t>(v);

                math_t buffer = optimizer_round<scalar_t, math_t>(
                    optimizer_sqrt(v));
                buffer = optimizer_round<scalar_t, math_t>(
                    buffer + eps_value);
                if (sc[7] != math_t(0)) {
                    buffer = optimizer_round<scalar_t, math_t>(
                        buffer / sc[7]);
                    buffer = optimizer_round<scalar_t, math_t>(
                        math_t(1) / buffer);
                } else {
                    // value through divide-by-zero and reciprocal.  Avoid
                    // manufacturing an infinity here; the surviving term is
                    // the unrectified step size.
                    buffer = optimizer_round<scalar_t, math_t>(sc[6]);
                }
                p = optimizer_addcmul<scalar_t, math_t>(
                    p, m, buffer, math_t(1));
                param[i] = static_cast<scalar_t>(p);
            }
        }
    });
}

void fused_radam_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_avg_sqs,
        std::vector<Tensor> state_steps, double lr, double beta1,
        double beta2, double eps, double weight_decay,
        bool decoupled_weight_decay, bool maximize) {
    const char* op_name = "_fused_radam_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_avg_sqs, true, op_name);
    validate_fused_steps(params, state_steps, op_name);
    if (params[0].dtype() == DType::Float16) {
        fused_radam_math<Half, float>(params, grads, exp_avgs, exp_avg_sqs,
            state_steps, lr, beta1, beta2, eps, weight_decay,
            decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::BFloat16) {
        fused_radam_math<BFloat16, float>(params, grads, exp_avgs,
            exp_avg_sqs, state_steps, lr, beta1, beta2, eps, weight_decay,
            decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float32) {
        fused_radam_math<float, float>(params, grads, exp_avgs, exp_avg_sqs,
            state_steps, lr, beta1, beta2, eps, weight_decay,
            decoupled_weight_decay, maximize);
    } else if (params[0].dtype() == DType::Float64) {
        fused_radam_math<double, double>(params, grads, exp_avgs,
            exp_avg_sqs, state_steps, lr, beta1, beta2, eps, weight_decay,
            decoupled_weight_decay, maximize);
    } else {
        TP_THROW(NotImplementedError, std::string(op_name) +
            ": unsupported floating dtype");
    }
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

template <typename math_t>
struct AdafactorPartial {
    math_t param_sum = math_t(0);
    math_t update_sum = math_t(0);
};

#ifdef TP_OPT_X86_SIMD

// The factored path is naturally contiguous by row.  Keep the row/column
// state update scalar (there are only O(rows + cols) values), but vectorize
// the two O(rows * cols) passes that dominate the optimizer step.  This is
// deliberately a separate target-dispatched shim: the main translation unit
// must remain runnable on machines without AVX2/AVX512.
#define TP_ADAFACTOR_STATS_VEC(FN, T, VTYPE, W, SUF, VECW)                     \
static void FN(const T* param, const T* grad, const T* row_var,              \
               const T* col_var, int64_t row_begin, int64_t row_end,         \
               int64_t cols, T denominator, T eps1_sq, bool maximize,         \
               T* param_sum, T* update_sum) {                                 \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                             \
    const VTYPE vdenominator = _mm##W##_set1_##SUF(denominator);              \
    const VTYPE veps1_sq = _mm##W##_set1_##SUF(eps1_sq);                      \
    VTYPE vparam_sum = vzero;                                                 \
    VTYPE vupdate_sum = vzero;                                                \
    for (int64_t row = row_begin; row < row_end; ++row) {                     \
        const T* row_param = param + row * cols;                               \
        const T* row_grad = grad + row * cols;                                 \
        const VTYPE vrow = _mm##W##_set1_##SUF(row_var[row]);                 \
        int64_t col = 0;                                                       \
        for (; col + VECW <= cols; col += VECW) {                             \
            VTYPE vg = _mm##W##_loadu_##SUF(row_grad + col);                  \
            if (maximize) vg = _mm##W##_sub_##SUF(vzero, vg);                 \
            VTYPE vv = _mm##W##_mul_##SUF(                                   \
                vrow, _mm##W##_loadu_##SUF(col_var + col));                  \
            vv = _mm##W##_div_##SUF(vv, vdenominator);                       \
            vv = _mm##W##_max_##SUF(vv, veps1_sq);                            \
            const VTYPE vu = _mm##W##_div_##SUF(                              \
                vg, _mm##W##_sqrt_##SUF(vv));                                \
            const VTYPE vp = _mm##W##_loadu_##SUF(row_param + col);           \
            vparam_sum = _mm##W##_fmadd_##SUF(vp, vp, vparam_sum);            \
            vupdate_sum = _mm##W##_fmadd_##SUF(vu, vu, vupdate_sum);          \
        }                                                                      \
        for (; col < cols; ++col) {                                            \
            T g = row_grad[col];                                               \
            if (maximize) g = -g;                                              \
            const T v = std::max(row_var[row] * col_var[col] / denominator,   \
                                 eps1_sq);                                     \
            const T u = g / std::sqrt(v);                                      \
            *param_sum += row_param[col] * row_param[col];                    \
            *update_sum += u * u;                                              \
        }                                                                      \
    }                                                                          \
    alignas(VECW * sizeof(T)) T param_lanes[VECW];                            \
    alignas(VECW * sizeof(T)) T update_lanes[VECW];                           \
    _mm##W##_storeu_##SUF(param_lanes, vparam_sum);                            \
    _mm##W##_storeu_##SUF(update_lanes, vupdate_sum);                         \
    for (int lane = 0; lane < VECW; ++lane) {                                  \
        *param_sum += param_lanes[lane];                                      \
        *update_sum += update_lanes[lane];                                    \
    }                                                                          \
}

#define TP_ADAFACTOR_APPLY_VEC(FN, T, VTYPE, W, SUF, VECW)                    \
static void FN(T* param, const T* grad, const T* row_var, const T* col_var,  \
               int64_t row_begin, int64_t row_end, int64_t cols,              \
               T denominator, T eps1_sq, T scale, T decay_scale,               \
               bool has_decay, bool maximize) {                                \
    const VTYPE vzero = _mm##W##_setzero_##SUF();                             \
    const VTYPE vdenominator = _mm##W##_set1_##SUF(denominator);              \
    const VTYPE veps1_sq = _mm##W##_set1_##SUF(eps1_sq);                      \
    const VTYPE vscale = _mm##W##_set1_##SUF(scale);                          \
    const VTYPE vdecay = _mm##W##_set1_##SUF(decay_scale);                    \
    for (int64_t row = row_begin; row < row_end; ++row) {                     \
        T* row_param = param + row * cols;                                     \
        const T* row_grad = grad + row * cols;                                 \
        const VTYPE vrow = _mm##W##_set1_##SUF(row_var[row]);                 \
        int64_t col = 0;                                                       \
        for (; col + VECW <= cols; col += VECW) {                             \
            VTYPE vg = _mm##W##_loadu_##SUF(row_grad + col);                  \
            if (maximize) vg = _mm##W##_sub_##SUF(vzero, vg);                 \
            VTYPE vv = _mm##W##_mul_##SUF(                                   \
                vrow, _mm##W##_loadu_##SUF(col_var + col));                  \
            vv = _mm##W##_div_##SUF(vv, vdenominator);                       \
            vv = _mm##W##_max_##SUF(vv, veps1_sq);                            \
            const VTYPE vu = _mm##W##_div_##SUF(                              \
                vg, _mm##W##_sqrt_##SUF(vv));                                \
            VTYPE vp = _mm##W##_loadu_##SUF(row_param + col);                 \
            if (has_decay) vp = _mm##W##_mul_##SUF(vp, vdecay);               \
            vp = _mm##W##_sub_##SUF(                                          \
                vp, _mm##W##_mul_##SUF(vscale, vu));                          \
            _mm##W##_storeu_##SUF(row_param + col, vp);                      \
        }                                                                      \
        for (; col < cols; ++col) {                                            \
            T g = row_grad[col];                                               \
            if (maximize) g = -g;                                              \
            const T v = std::max(row_var[row] * col_var[col] / denominator,   \
                                 eps1_sq);                                     \
            const T u = g / std::sqrt(v);                                      \
            T p = row_param[col];                                              \
            if (has_decay) p *= decay_scale;                                   \
            row_param[col] = p - scale * u;                                    \
        }                                                                      \
    }                                                                          \
}

#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace adafactor_avx2_target {
TP_ADAFACTOR_STATS_VEC(factored_stats_f32, float, __m256, 256, ps, 8)
TP_ADAFACTOR_STATS_VEC(factored_stats_f64, double, __m256d, 256, pd, 4)
TP_ADAFACTOR_APPLY_VEC(factored_apply_f32, float, __m256, 256, ps, 8)
TP_ADAFACTOR_APPLY_VEC(factored_apply_f64, double, __m256d, 256, pd, 4)
}  // namespace adafactor_avx2_target
#pragma GCC pop_options

#pragma GCC push_options
#pragma GCC target("avx512f")
namespace adafactor_avx512_target {
TP_ADAFACTOR_STATS_VEC(factored_stats_f32, float, __m512, 512, ps, 16)
TP_ADAFACTOR_STATS_VEC(factored_stats_f64, double, __m512d, 512, pd, 8)
TP_ADAFACTOR_APPLY_VEC(factored_apply_f32, float, __m512, 512, ps, 16)
TP_ADAFACTOR_APPLY_VEC(factored_apply_f64, double, __m512d, 512, pd, 8)
}  // namespace adafactor_avx512_target
#pragma GCC pop_options

#undef TP_ADAFACTOR_STATS_VEC
#undef TP_ADAFACTOR_APPLY_VEC

// The row/column variance updates are reductions over the gradient.  Keep
// contiguous data, while columns reduce a block of adjacent columns at once
// so the strided dimension is still read as contiguous cache lines.
#pragma GCC push_options
#pragma GCC target("avx512f")
namespace adafactor_avx512_target {
void state_stats_f32(const float* grad, float* row_var, float* col_var,
                     int64_t rows, int64_t cols, float beta) {
    for (int64_t row = 0; row < rows; ++row) {
        const float* row_grad = grad + row * cols;
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        int64_t col = 0;
        for (; col + 64 <= cols; col += 64) {
            __m512 x0 = _mm512_loadu_ps(row_grad + col);
            __m512 x1 = _mm512_loadu_ps(row_grad + col + 16);
            __m512 x2 = _mm512_loadu_ps(row_grad + col + 32);
            __m512 x3 = _mm512_loadu_ps(row_grad + col + 48);
            s0 = _mm512_add_ps(s0, _mm512_mul_ps(x0, x0));
            s1 = _mm512_add_ps(s1, _mm512_mul_ps(x1, x1));
            s2 = _mm512_add_ps(s2, _mm512_mul_ps(x2, x2));
            s3 = _mm512_add_ps(s3, _mm512_mul_ps(x3, x3));
        }
        for (; col + 16 <= cols; col += 16) {
            __m512 x = _mm512_loadu_ps(row_grad + col);
            s0 = _mm512_add_ps(s0, _mm512_mul_ps(x, x));
        }
        float sum = _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
        for (; col < cols; ++col) {
            const float x = row_grad[col];
            sum += x * x;
        }
        const float mean = sum / static_cast<float>(cols);
        row_var[row] += beta * (mean - row_var[row]);
    }

    int64_t col = 0;
    for (; col + 16 <= cols; col += 16) {
        __m512 s0 = _mm512_setzero_ps();
        __m512 s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps();
        __m512 s3 = _mm512_setzero_ps();
        int64_t row = 0;
        for (; row + 4 <= rows; row += 4) {
            __m512 x0 = _mm512_loadu_ps(grad + row * cols + col);
            __m512 x1 = _mm512_loadu_ps(grad + (row + 1) * cols + col);
            __m512 x2 = _mm512_loadu_ps(grad + (row + 2) * cols + col);
            __m512 x3 = _mm512_loadu_ps(grad + (row + 3) * cols + col);
            s0 = _mm512_add_ps(s0, _mm512_mul_ps(x0, x0));
            s1 = _mm512_add_ps(s1, _mm512_mul_ps(x1, x1));
            s2 = _mm512_add_ps(s2, _mm512_mul_ps(x2, x2));
            s3 = _mm512_add_ps(s3, _mm512_mul_ps(x3, x3));
        }
        s0 = _mm512_add_ps(_mm512_add_ps(s0, s1),
                           _mm512_add_ps(s2, s3));
        alignas(64) float sums[16];
        _mm512_store_ps(sums, s0);
        const int64_t tail_begin = row;
        for (int lane = 0; lane < 16; ++lane) {
            for (int64_t r = tail_begin; r < rows; ++r) {
                const float x = grad[r * cols + col + lane];
                sums[lane] += x * x;
            }
            col_var[col + lane] += beta *
                (sums[lane] / static_cast<float>(rows) - col_var[col + lane]);
        }
    }
    for (; col < cols; ++col) {
        float sum = 0.0f;
        for (int64_t row = 0; row < rows; ++row) {
            const float x = grad[row * cols + col];
            sum += x * x;
        }
        col_var[col] += beta *
            (sum / static_cast<float>(rows) - col_var[col]);
    }
}

void state_stats_f64(const double* grad, double* row_var, double* col_var,
                     int64_t rows, int64_t cols, double beta) {
    for (int64_t row = 0; row < rows; ++row) {
        const double* row_grad = grad + row * cols;
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        int64_t col = 0;
        for (; col + 32 <= cols; col += 32) {
            __m512d x0 = _mm512_loadu_pd(row_grad + col);
            __m512d x1 = _mm512_loadu_pd(row_grad + col + 8);
            __m512d x2 = _mm512_loadu_pd(row_grad + col + 16);
            __m512d x3 = _mm512_loadu_pd(row_grad + col + 24);
            s0 = _mm512_add_pd(s0, _mm512_mul_pd(x0, x0));
            s1 = _mm512_add_pd(s1, _mm512_mul_pd(x1, x1));
            s2 = _mm512_add_pd(s2, _mm512_mul_pd(x2, x2));
            s3 = _mm512_add_pd(s3, _mm512_mul_pd(x3, x3));
        }
        for (; col + 8 <= cols; col += 8) {
            __m512d x = _mm512_loadu_pd(row_grad + col);
            s0 = _mm512_add_pd(s0, _mm512_mul_pd(x, x));
        }
        double sum = _mm512_reduce_add_pd(_mm512_add_pd(
            _mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3)));
        for (; col < cols; ++col) {
            const double x = row_grad[col];
            sum += x * x;
        }
        const double mean = sum / static_cast<double>(cols);
        row_var[row] += beta * (mean - row_var[row]);
    }

    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        int64_t row = 0;
        for (; row + 4 <= rows; row += 4) {
            __m512d x0 = _mm512_loadu_pd(grad + row * cols + col);
            __m512d x1 = _mm512_loadu_pd(grad + (row + 1) * cols + col);
            __m512d x2 = _mm512_loadu_pd(grad + (row + 2) * cols + col);
            __m512d x3 = _mm512_loadu_pd(grad + (row + 3) * cols + col);
            s0 = _mm512_add_pd(s0, _mm512_mul_pd(x0, x0));
            s1 = _mm512_add_pd(s1, _mm512_mul_pd(x1, x1));
            s2 = _mm512_add_pd(s2, _mm512_mul_pd(x2, x2));
            s3 = _mm512_add_pd(s3, _mm512_mul_pd(x3, x3));
        }
        s0 = _mm512_add_pd(_mm512_add_pd(s0, s1),
                           _mm512_add_pd(s2, s3));
        alignas(64) double sums[8];
        _mm512_store_pd(sums, s0);
        for (int lane = 0; lane < 8; ++lane) {
            for (int64_t r = row; r < rows; ++r) {
                const double x = grad[r * cols + col + lane];
                sums[lane] += x * x;
            }
            col_var[col + lane] += beta *
                (sums[lane] / static_cast<double>(rows) - col_var[col + lane]);
        }
    }
    for (; col < cols; ++col) {
        double sum = 0.0;
        for (int64_t row = 0; row < rows; ++row) {
            const double x = grad[row * cols + col];
            sum += x * x;
        }
        col_var[col] += beta *
            (sum / static_cast<double>(rows) - col_var[col]);
    }
}
}  // namespace adafactor_avx512_target
#pragma GCC pop_options

#endif  // TP_OPT_X86_SIMD

template <typename scalar_t, typename math_t>
void adafactor_factored_state_stats_scalar(
        const scalar_t* grad, scalar_t* row_var, scalar_t* col_var,
        int64_t rows, int64_t cols, math_t beta) {
    for (int64_t row = 0; row < rows; ++row) {
        math_t sum = math_t(0);
        for (int64_t col = 0; col < cols; ++col) {
            const math_t x = static_cast<math_t>(grad[row * cols + col]);
            sum += x * x;
        }
        const math_t mean = sum / static_cast<math_t>(cols);
        const math_t old = static_cast<math_t>(row_var[row]);
        row_var[row] = static_cast<scalar_t>(old + beta * (mean - old));
    }
    for (int64_t col = 0; col < cols; ++col) {
        math_t sum = math_t(0);
        for (int64_t row = 0; row < rows; ++row) {
            const math_t x = static_cast<math_t>(grad[row * cols + col]);
            sum += x * x;
        }
        const math_t mean = sum / static_cast<math_t>(rows);
        const math_t old = static_cast<math_t>(col_var[col]);
        col_var[col] = static_cast<scalar_t>(old + beta * (mean - old));
    }
}

template <typename scalar_t, typename math_t>
bool adafactor_factored_state_stats_dispatch(
        const scalar_t* grad, scalar_t* row_var, scalar_t* col_var,
        int64_t rows, int64_t cols, math_t beta) {
#ifdef TP_OPT_X86_SIMD
    if constexpr (std::is_same_v<scalar_t, float> &&
                  std::is_same_v<math_t, float>) {
        if (have_avx512f()) {
            adafactor_avx512_target::state_stats_f32(
                grad, row_var, col_var, rows, cols, beta);
            return true;
        }
    } else if constexpr (std::is_same_v<scalar_t, double> &&
                         std::is_same_v<math_t, double>) {
        if (have_avx512f()) {
            adafactor_avx512_target::state_stats_f64(
                grad, row_var, col_var, rows, cols, beta);
            return true;
        }
    }
#endif
    return false;
}

template <typename scalar_t, typename math_t>
void adafactor_factored_stats_dispatch(
        const scalar_t* param, const scalar_t* grad,
        const scalar_t* row_var, const scalar_t* col_var,
        int64_t row_begin, int64_t row_end, int64_t cols,
        math_t denominator, math_t eps1_sq, bool maximize,
        math_t* param_sum, math_t* update_sum) {
#ifdef TP_OPT_X86_SIMD
    if constexpr (std::is_same_v<scalar_t, float> &&
                  std::is_same_v<math_t, float>) {
        if (have_avx512f()) {
            adafactor_avx512_target::factored_stats_f32(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, maximize, param_sum, update_sum);
            return;
        }
        if (have_avx2()) {
            adafactor_avx2_target::factored_stats_f32(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, maximize, param_sum, update_sum);
            return;
        }
    } else if constexpr (std::is_same_v<scalar_t, double> &&
                         std::is_same_v<math_t, double>) {
        if (have_avx512f()) {
            adafactor_avx512_target::factored_stats_f64(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, maximize, param_sum, update_sum);
            return;
        }
        if (have_avx2()) {
            adafactor_avx2_target::factored_stats_f64(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, maximize, param_sum, update_sum);
            return;
        }
    }
#endif
    math_t p_sum = math_t(0);
    math_t u_sum = math_t(0);
    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t row_offset = row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            math_t g = static_cast<math_t>(grad[row_offset + col]);
            if (maximize) g = -g;
            const math_t v = std::max(
                static_cast<math_t>(row_var[row]) *
                    static_cast<math_t>(col_var[col]) / denominator,
                eps1_sq);
            const math_t update = g / std::sqrt(v);
            const math_t p = static_cast<math_t>(param[row_offset + col]);
            p_sum += p * p;
            u_sum += update * update;
        }
    }
    *param_sum = p_sum;
    *update_sum = u_sum;
}

template <typename scalar_t, typename math_t>
void adafactor_factored_apply_dispatch(
        scalar_t* param, const scalar_t* grad, const scalar_t* row_var,
        const scalar_t* col_var, int64_t row_begin, int64_t row_end,
        int64_t cols, math_t denominator, math_t eps1_sq, math_t scale,
        math_t decay_scale, bool has_decay, bool maximize) {
#ifdef TP_OPT_X86_SIMD
    if constexpr (std::is_same_v<scalar_t, float> &&
                  std::is_same_v<math_t, float>) {
        if (have_avx512f()) {
            adafactor_avx512_target::factored_apply_f32(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, scale, decay_scale, has_decay, maximize);
            return;
        }
        if (have_avx2()) {
            adafactor_avx2_target::factored_apply_f32(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, scale, decay_scale, has_decay, maximize);
            return;
        }
    } else if constexpr (std::is_same_v<scalar_t, double> &&
                         std::is_same_v<math_t, double>) {
        if (have_avx512f()) {
            adafactor_avx512_target::factored_apply_f64(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, scale, decay_scale, has_decay, maximize);
            return;
        }
        if (have_avx2()) {
            adafactor_avx2_target::factored_apply_f64(
                param, grad, row_var, col_var, row_begin, row_end, cols,
                denominator, eps1_sq, scale, decay_scale, has_decay, maximize);
            return;
        }
    }
#endif
    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t row_offset = row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            math_t g = static_cast<math_t>(grad[row_offset + col]);
            if (maximize) g = -g;
            const math_t v = std::max(
                static_cast<math_t>(row_var[row]) *
                    static_cast<math_t>(col_var[col]) / denominator,
                eps1_sq);
            const math_t update = g / std::sqrt(v);
            math_t p = static_cast<math_t>(param[row_offset + col]);
            if (has_decay) p *= decay_scale;
            param[row_offset + col] = static_cast<scalar_t>(
                p - scale * update);
        }
    }
}

template <typename scalar_t, typename math_t>
void fused_adafactor_vector_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& variances,
        std::vector<Tensor>& state_steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> variance_ptrs(count);
    std::vector<math_t> steps(count), beta2_weights(count), rhos(count);
    std::vector<int64_t> numels(count);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        variance_ptrs[i] = variances[i].data_ptr<scalar_t>();
        steps[i] = static_cast<math_t>(increment_cpu_step(
            state_steps[i], "_fused_adafactor_"));
        beta2_weights[i] = static_cast<math_t>(std::pow(
            static_cast<double>(steps[i]), beta2_decay));
        rhos[i] = std::min(static_cast<math_t>(lr),
            math_t(1) / std::sqrt(steps[i]));
        numels[i] = params[i].numel();
    }

    const auto work = build_opt_work_list(numels.data(), count);
    std::vector<AdafactorPartial<math_t>> partials(work.size());
    const math_t eps1_value = static_cast<math_t>(eps1);
    const math_t eps1_sq = eps1_value * eps1_value;
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            scalar_t* variance = variance_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            math_t param_sum = math_t(0);
            math_t update_sum = math_t(0);
            for (int64_t i = item.begin; i < item.end; ++i) {
                const math_t p = static_cast<math_t>(param_ptrs[li][i]);
                math_t g = static_cast<math_t>(grad[i]);
                if (maximize) g = -g;
                const math_t old = static_cast<math_t>(variance[i]);
                const math_t next = old + beta2_weights[li] * (g * g - old);
                variance[i] = static_cast<scalar_t>(next);
                const math_t update = g / std::sqrt(
                    std::max(next, eps1_sq));
                param_sum += p * p;
                update_sum += update * update;
            }
            partials[static_cast<size_t>(work_index)] = {
                param_sum, update_sum};
        }
    });

    std::vector<math_t> param_sums(count, math_t(0));
    std::vector<math_t> update_sums(count, math_t(0));
    for (size_t i = 0; i < work.size(); ++i) {
        const size_t li = static_cast<size_t>(work[i].list_index);
        param_sums[li] += partials[i].param_sum;
        update_sums[li] += partials[i].update_sum;
    }

    const math_t lr_value = static_cast<math_t>(lr);
    const math_t eps2_value = static_cast<math_t>(eps2);
    const math_t d_value = static_cast<math_t>(d);
    const math_t decay_scale = math_t(1) - lr_value *
        static_cast<math_t>(weight_decay);
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const math_t rms_param = std::sqrt(param_sums[li] /
                static_cast<math_t>(numels[li]));
            const math_t alpha = std::max(eps2_value, rms_param) * rhos[li];
            const math_t rms_update = std::sqrt(update_sums[li] /
                static_cast<math_t>(numels[li]));
            const math_t clip = std::max(math_t(1), rms_update / d_value);
            const math_t scale = alpha / clip;
            scalar_t* param = param_ptrs[li];
            const scalar_t* grad = grad_ptrs[li];
            const scalar_t* variance = variance_ptrs[li];
            for (int64_t i = item.begin; i < item.end; ++i) {
                math_t g = static_cast<math_t>(grad[i]);
                if (maximize) g = -g;
                const math_t update = g / std::sqrt(std::max(
                    static_cast<math_t>(variance[i]), eps1_sq));
                math_t p = static_cast<math_t>(param[i]);
                if (weight_decay != 0.0) p *= decay_scale;
                param[i] = static_cast<scalar_t>(p - scale * update);
            }
        }
    });
}

struct AdafactorMatrixWork {
    size_t list_index;
    int64_t batch;
};

struct AdafactorMatrixChunk {
    size_t list_index;
    int64_t batch;
    int64_t row_begin;
    int64_t row_end;
};

template <typename scalar_t, typename math_t>
void fused_adafactor_factored_math(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& row_vars,
        const std::vector<Tensor>& col_vars,
        std::vector<Tensor>& state_steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize) {
    const size_t count = params.size();
    std::vector<scalar_t*> param_ptrs(count);
    std::vector<const scalar_t*> grad_ptrs(count);
    std::vector<scalar_t*> row_ptrs(count), col_ptrs(count);
    std::vector<math_t> steps(count), beta2_weights(count), rhos(count);
    std::vector<int64_t> numels(count), rows(count), cols(count), matrix_sizes(count);
    std::vector<size_t> batch_offsets(count + 1, 0);
    for (size_t i = 0; i < count; ++i) {
        param_ptrs[i] = params[i].data_ptr<scalar_t>();
        grad_ptrs[i] = grads[i].data_ptr<scalar_t>();
        row_ptrs[i] = row_vars[i].data_ptr<scalar_t>();
        col_ptrs[i] = col_vars[i].data_ptr<scalar_t>();
        steps[i] = static_cast<math_t>(increment_cpu_step(
            state_steps[i], "_fused_adafactor_factored_"));
        beta2_weights[i] = static_cast<math_t>(std::pow(
            static_cast<double>(steps[i]), beta2_decay));
        rhos[i] = std::min(static_cast<math_t>(lr),
            math_t(1) / std::sqrt(steps[i]));
        numels[i] = params[i].numel();
        rows[i] = params[i].size(-2);
        cols[i] = params[i].size(-1);
        matrix_sizes[i] = rows[i] * cols[i];
        const int64_t outer = matrix_sizes[i] == 0
            ? 0 : numels[i] / matrix_sizes[i];
        batch_offsets[i + 1] = batch_offsets[i] +
            static_cast<size_t>(outer);
    }

    std::vector<AdafactorMatrixWork> batch_work;
    std::vector<AdafactorMatrixChunk> matrix_work;
    for (size_t i = 0; i < count; ++i) {
        const int64_t outer = matrix_sizes[i] == 0
            ? 0 : numels[i] / matrix_sizes[i];
        for (int64_t b = 0; b < outer; ++b) batch_work.push_back({i, b});
        for (int64_t b = 0; b < outer; ++b) {
            const int64_t rows_per_chunk = std::max<int64_t>(
                1, GRAIN_SIZE / std::max<int64_t>(cols[i], 1));
            for (int64_t r = 0; r < rows[i]; r += rows_per_chunk) {
                matrix_work.push_back({i, b, r,
                    std::min<int64_t>(r + rows_per_chunk, rows[i])});
            }
        }
    }

    parallel_for(0, static_cast<int64_t>(batch_work.size()), 1,
                 [&](int64_t begin, int64_t end) {
        for (int64_t wi = begin; wi < end; ++wi) {
            const auto& item = batch_work[static_cast<size_t>(wi)];
            const size_t li = item.list_index;
            const int64_t batch = item.batch;
            const int64_t base = batch * matrix_sizes[li];
            scalar_t* row_var = row_ptrs[li] + batch * rows[li];
            scalar_t* col_var = col_ptrs[li] + batch * cols[li];
            const bool vectorized = adafactor_factored_state_stats_dispatch(
                grad_ptrs[li] + base, row_var, col_var, rows[li], cols[li],
                beta2_weights[li]);
            if (!vectorized) {
                adafactor_factored_state_stats_scalar(
                    grad_ptrs[li] + base, row_var, col_var, rows[li],
                    cols[li], beta2_weights[li]);
            }
        }
    });

    std::vector<math_t> row_means(batch_work.size(), math_t(0));
    parallel_for(0, static_cast<int64_t>(batch_work.size()), 1,
                 [&](int64_t begin, int64_t end) {
        for (int64_t wi = begin; wi < end; ++wi) {
            const auto& item = batch_work[static_cast<size_t>(wi)];
            const size_t li = item.list_index;
            const int64_t batch = item.batch;
            math_t sum = math_t(0);
            for (int64_t r = 0; r < rows[li]; ++r) {
                sum += static_cast<math_t>(row_ptrs[li][
                    batch * rows[li] + r]);
            }
            row_means[static_cast<size_t>(wi)] = sum /
                static_cast<math_t>(rows[li]);
        }
    });

    std::vector<AdafactorPartial<math_t>> partials(matrix_work.size());
    const math_t eps1_value = static_cast<math_t>(eps1);
    const math_t eps1_sq = eps1_value * eps1_value;
    parallel_for(0, static_cast<int64_t>(matrix_work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = matrix_work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const int64_t batch = item.batch;
            const int64_t matrix = matrix_sizes[li];
            const int64_t cols_value = cols[li];
            const int64_t batch_base = batch * matrix;
            const int64_t row_state_base = batch * rows[li];
            const int64_t col_state_base = batch * cols_value;
            const math_t row_mean = row_means[batch_offsets[li] +
                static_cast<size_t>(batch)];
            math_t param_sum = math_t(0);
            math_t update_sum = math_t(0);
            const math_t denominator = std::max(row_mean, eps1_value);
            adafactor_factored_stats_dispatch(
                param_ptrs[li] + batch_base, grad_ptrs[li] + batch_base,
                row_ptrs[li] + row_state_base,
                col_ptrs[li] + col_state_base, item.row_begin, item.row_end,
                cols_value, denominator, eps1_sq, maximize, &param_sum,
                &update_sum);
            partials[static_cast<size_t>(work_index)] = {
                param_sum, update_sum};
        }
    });

    std::vector<math_t> param_sums(count, math_t(0));
    std::vector<math_t> update_sums(count, math_t(0));
    for (size_t i = 0; i < matrix_work.size(); ++i) {
        const size_t li = static_cast<size_t>(matrix_work[i].list_index);
        param_sums[li] += partials[i].param_sum;
        update_sums[li] += partials[i].update_sum;
    }
    const math_t lr_value = static_cast<math_t>(lr);
    const math_t eps2_value = static_cast<math_t>(eps2);
    const math_t d_value = static_cast<math_t>(d);
    std::vector<math_t> scales(count, math_t(0));
    for (size_t i = 0; i < count; ++i) {
        const math_t alpha = std::max(eps2_value,
            std::sqrt(param_sums[i] / static_cast<math_t>(numels[i]))) *
            rhos[i];
        const math_t clip = std::max(math_t(1),
            std::sqrt(update_sums[i] / static_cast<math_t>(numels[i])) /
            d_value);
        scales[i] = alpha / clip;
    }
    const math_t decay_scale = math_t(1) - lr_value *
        static_cast<math_t>(weight_decay);
    parallel_for(0, static_cast<int64_t>(matrix_work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end;
             ++work_index) {
            const auto& item = matrix_work[static_cast<size_t>(work_index)];
            const size_t li = static_cast<size_t>(item.list_index);
            const math_t scale = scales[li];
            const int64_t cols_value = cols[li];
            const int64_t matrix = matrix_sizes[li];
            const int64_t batch_base = item.batch * matrix;
            const int64_t row_state_base = item.batch * rows[li];
            const int64_t col_state_base = item.batch * cols_value;
            const math_t row_mean = row_means[batch_offsets[li] +
                static_cast<size_t>(item.batch)];
            const math_t denominator = std::max(row_mean, eps1_value);
            adafactor_factored_apply_dispatch(
                param_ptrs[li] + batch_base, grad_ptrs[li] + batch_base,
                row_ptrs[li] + row_state_base,
                col_ptrs[li] + col_state_base, item.row_begin, item.row_end,
                cols_value, denominator, eps1_sq, scale, decay_scale,
                weight_decay != 0.0, maximize);
        }
    });
}

void validate_adafactor_factored_state_cpu(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& row_vars,
        const std::vector<Tensor>& col_vars,
        const char* op_name) {
    if (row_vars.size() != params.size() || col_vars.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": factored state lists must match parameter list");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = params[i];
        std::vector<int64_t> row_shape = p.shape();
        std::vector<int64_t> col_shape = p.shape();
        if (p.dim() < 2) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": factored state requires tensors with at least two dimensions");
        }
        row_shape.back() = 1;
        col_shape[col_shape.size() - 2] = 1;
        const Tensor& row = row_vars[i];
        const Tensor& col = col_vars[i];
        if (!row.defined() || !col.defined() || !row.is_contiguous() ||
            !col.is_contiguous() || row.shape() != Size(row_shape) ||
            col.shape() != Size(col_shape) || row.dtype() != p.dtype() ||
            col.dtype() != p.dtype() || row.device() != p.device() ||
            col.device() != p.device()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": factored states must match the parameter layout");
        }
    }
}

void fused_adafactor_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> variances, std::vector<Tensor> state_steps,
        double lr, double beta2_decay, double eps1, double eps2, double d,
        double weight_decay, bool maximize) {
    const char* op_name = "_fused_adafactor_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, variances, true, op_name);
    validate_fused_steps(params, state_steps, op_name);
    dispatch_fused_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            fused_adafactor_vector_math<scalar_t, math_t>(
                params, grads, variances, state_steps, lr, beta2_decay,
                eps1, eps2, d, weight_decay, maximize);
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adafactor_factored_cpu(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> row_vars, std::vector<Tensor> col_vars,
        std::vector<Tensor> state_steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize) {
    const char* op_name = "_fused_adafactor_factored_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_adafactor_factored_state_cpu(
        params, row_vars, col_vars, op_name);
    validate_fused_steps(params, state_steps, op_name);
    dispatch_fused_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            fused_adafactor_factored_math<scalar_t, math_t>(
                params, grads, row_vars, col_vars, state_steps, lr,
                beta2_decay, eps1, eps2, d, weight_decay, maximize);
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

} // namespace

std::vector<Tensor> foreach_sgd_cpu(const std::vector<Tensor>& params,
                                     const std::vector<Tensor>& grads,
                                     const std::vector<Tensor>& momentum_buffers,
                                     double lr,
                                     double momentum,
                                     double dampening,
                                     double weight_decay,
                                     bool nesterov,
                                     bool first_momentum_step) {
    std::vector<Tensor> empty_states(params.size());
    std::vector<int64_t> no_steps;
    validate_lists(params, grads, momentum_buffers, empty_states, empty_states,
                   no_steps, momentum != 0.0, false, false, "_foreach_sgd");

    if (params.empty()) return params;
    if (params[0].dtype() == DType::Float32) {
        sgd_impl<float>(params, grads, momentum_buffers, lr, momentum,
                        dampening, weight_decay, nesterov, first_momentum_step);
    } else if (params[0].dtype() == DType::Float64) {
        sgd_impl<double>(params, grads, momentum_buffers, lr, momentum,
                         dampening, weight_decay, nesterov, first_momentum_step);
    } else {
        TP_THROW(NotImplementedError,
                 "_foreach_sgd supports float32 and float64 tensors");
    }
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}

std::vector<Tensor> foreach_adam_cpu(const std::vector<Tensor>& params,
                                      const std::vector<Tensor>& grads,
                                      const std::vector<Tensor>& exp_avgs,
                                      const std::vector<Tensor>& exp_avg_sqs,
                                      const std::vector<Tensor>& max_exp_avg_sqs,
                                      const std::vector<int64_t>& steps,
                                      double lr,
                                      double beta1,
                                      double beta2,
                                      double eps,
                                      double weight_decay,
                                      bool amsgrad) {
    if (steps.size() != params.size()) {
        TP_THROW(ValueError, "_foreach_adam: step list size must match parameter list");
    }
    std::vector<Tensor> empty_states(params.size());
    validate_lists(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                   steps, true, true, amsgrad, "_foreach_adam");

    if (params.empty()) return params;
    if (params[0].dtype() == DType::Float32) {
        adam_impl<float>(params, grads, exp_avgs, exp_avg_sqs,
                         max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                         weight_decay, amsgrad);
    } else if (params[0].dtype() == DType::Float64) {
        adam_impl<double>(params, grads, exp_avgs, exp_avg_sqs,
                          max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                          weight_decay, amsgrad);
    } else {
        TP_THROW(NotImplementedError,
                 "_foreach_adam supports float32 and float64 tensors");
    }
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}

TENSORPLAY_LIBRARY_IMPL(CPU, OptimizerKernels) {
    m.impl("_foreach_sgd", foreach_sgd_cpu);
    m.impl("_foreach_adam", foreach_adam_cpu);
    m.impl("_fused_adam_", fused_adam_cpu);
    m.impl("_fused_adam_.tensor_lr", fused_adam_tensor_lr_cpu);
    m.impl("_fused_adamw_", fused_adamw_cpu);
    m.impl("_fused_adamw_.tensor_lr", fused_adamw_tensor_lr_cpu);
    m.impl("_fused_sgd_", fused_sgd_cpu);
    m.impl("_fused_sgd_.tensor_lr", fused_sgd_tensor_lr_cpu);
    m.impl("_fused_adagrad_", fused_adagrad_cpu);
    m.impl("_fused_adagrad_.tensor_lr", fused_adagrad_tensor_lr_cpu);
    m.impl("_fused_rmsprop_", fused_rmsprop_cpu);
    m.impl("_fused_adadelta_", fused_adadelta_cpu);
    m.impl("_fused_adamax_", fused_adamax_cpu);
    m.impl("_fused_asgd_", fused_asgd_cpu);
    m.impl("_fused_rprop_", fused_rprop_cpu);
    m.impl("_fused_nadam_", fused_nadam_cpu);
    m.impl("_fused_radam_", fused_radam_cpu);
    m.impl("_fused_adafactor_", fused_adafactor_cpu);
    m.impl("_fused_adafactor_factored_", fused_adafactor_factored_cpu);
}

} // namespace cpu
} // namespace tensorplay
