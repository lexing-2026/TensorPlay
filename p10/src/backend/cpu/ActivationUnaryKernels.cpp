#include "cpu/ActivationUnaryKernels.h"
#include "cpu/VecUnary.h"
#include "cpu/vec/vec.h"
#include "Parallel.h"

// Tier-compiled activation kernels (see TP_CPU_KERNEL_SRCS in
// p10/CMakeLists.txt).  Each copy lands in the CPU_CAPABILITY inline
// namespace; DispatchStub picks the best registered tier at runtime.
// NB: REGISTER_DISPATCH under CPU_CAPABILITY_AVX512 intentionally registers
// REGISTER_AVX512_DISPATCH below, otherwise these ops silently fall back to
// the AVX2 copy.

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

namespace {
// parallel_for / GRAIN_SIZE live under tensorplay::parallel.
using namespace tensorplay::parallel;

// sigmoid(x) = 1 / (1 + exp(-x)), silu(x) = x * sigmoid(x).
//
// `vecunary::run_f32` is the GCC build's body: on a compiler without
// per-function ISA attributes (MSVC) its vector form does not exist and every
// element would fall into the scalar tail even though this TU is compiled at a
// full tier width.  Writing the two activations against the vector layer here
// keeps both toolchains on the same loop.
template <typename T>
void sigmoid_silu_walk(const T* src, T* dst, int64_t n, bool silu) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();
    int64_t i = 0;
    for (; i + W <= n; i += W) {
        const V x = V::loadu(src + i);
        const V den = V(T(1)) + x.neg().exp();
        (silu ? x / den : V(T(1)) / den).store(dst + i);
    }
    for (; i < n; ++i) {
        const T den = T(1) + std::exp(-src[i]);
        dst[i] = silu ? static_cast<T>(src[i] / den)
                      : static_cast<T>(T(1) / den);
    }
}

void sigmoid_f32_impl(const float* src, float* dst, int64_t n) {
    sigmoid_silu_walk(src, dst, n, false);
}
void silu_f32_impl(const float* src, float* dst, int64_t n) {
    sigmoid_silu_walk(src, dst, n, true);
}
void sigmoid_f64_impl(const double* src, double* dst, int64_t n) {
    sigmoid_silu_walk(src, dst, n, false);
}
void silu_f64_impl(const double* src, double* dst, int64_t n) {
    sigmoid_silu_walk(src, dst, n, true);
}

// The backward of a softmax over the last dim.  Every row answers
//   r = o * (g - sum(g * o))
// and the logarithmic form answers
//   r = g - exp(o) * sum(g).
// Both the sum and the answer run over the vector layer, so every
// CPU_CAPABILITY tier brings its own width.  A row that does not fill a
// vector is finished on its own.
template <typename T, typename V>
void softmax_backward_lastdim_walk(const T* grad, const T* out, T* grad_in,
                                   int64_t rows, int64_t cols, bool log_mode) {
    const int64_t lanes = V::size();
    const int64_t vec_cols = (cols / lanes) * lanes;
    parallel_for(0, rows, 1, [&](int64_t ub, int64_t ue) {
        for (int64_t r = ub; r < ue; ++r) {
            const T* g = grad + r * cols;
            const T* o = out + r * cols;
            T* ri = grad_in + r * cols;
            V acc(T(0));
            for (int64_t k = 0; k < vec_cols; k += lanes) {
                const V gv = V::loadu(g + k);
                acc = log_mode ? acc + gv : acc + gv * V::loadu(o + k);
            }
            T sum = acc.reduce_add();
            for (int64_t k = vec_cols; k < cols; ++k) {
                sum = log_mode ? sum + g[k] : sum + g[k] * o[k];
            }
            const V s(sum);
            if (log_mode) {
                for (int64_t k = 0; k < vec_cols; k += lanes) {
                    (V::loadu(g + k) - V::loadu(o + k).exp() * s).store(ri + k);
                }
                for (int64_t k = vec_cols; k < cols; ++k) {
                    ri[k] = g[k] - std::exp(o[k]) * sum;
                }
            } else {
                for (int64_t k = 0; k < vec_cols; k += lanes) {
                    (V::loadu(o + k) * (V::loadu(g + k) - s)).store(ri + k);
                }
                for (int64_t k = vec_cols; k < cols; ++k) {
                    ri[k] = o[k] * (g[k] - sum);
                }
            }
        }
    });
}

void softmax_backward_lastdim_f32_impl(const float* grad, const float* out,
                                       float* grad_in, int64_t rows, int64_t cols,
                                       bool log_mode) {
    softmax_backward_lastdim_walk<float, vec::Vectorized<float>>(
        grad, out, grad_in, rows, cols, log_mode);
}

void softmax_backward_lastdim_f64_impl(const double* grad, const double* out,
                                       double* grad_in, int64_t rows,
                                       int64_t cols, bool log_mode) {
    softmax_backward_lastdim_walk<double, vec::Vectorized<double>>(
        grad, out, grad_in, rows, cols, log_mode);
}
} // anonymous namespace
} // inline namespace CPU_CAPABILITY

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(sigmoid_f32_stub, &sigmoid_f32_impl);
REGISTER_DISPATCH(silu_f32_stub, &silu_f32_impl);
REGISTER_DISPATCH(sigmoid_f64_stub, &sigmoid_f64_impl);
REGISTER_DISPATCH(silu_f64_stub, &silu_f64_impl);
REGISTER_DISPATCH(softmax_backward_lastdim_f32_stub,
                   &softmax_backward_lastdim_f32_impl);
REGISTER_DISPATCH(softmax_backward_lastdim_f64_stub,
                   &softmax_backward_lastdim_f64_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_f32_stub, &sigmoid_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(silu_f32_stub, &silu_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_f64_stub, &sigmoid_f64_impl);
ALSO_REGISTER_AVX512_DISPATCH(silu_f64_stub, &silu_f64_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmax_backward_lastdim_f32_stub,
                              &softmax_backward_lastdim_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(softmax_backward_lastdim_f64_stub,
                              &softmax_backward_lastdim_f64_impl);
#endif

} // namespace cpu
} // namespace tensorplay
