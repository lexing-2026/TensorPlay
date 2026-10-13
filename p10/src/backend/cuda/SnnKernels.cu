// Spiking neural network (SNN) primitives - CUDA kernels.
//
// Fused single-step leaky integrate-and-fire update (see the CPU twin in
// SnnKernels.cpp for the math).  Each neuron element is updated by one
// grid-stride thread; the forward stores the spike and the updated voltage,
// and the backward computes all four input gradients in a single pass using
// the selected surrogate gradient.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <optional>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

namespace {

template <typename T>
using snn_compute_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T>
__host__ __device__ inline T snn_exp(T x) {
    return static_cast<T>(::exp(static_cast<double>(x)));
}

// Surrogate gradient evaluated at x = h - threshold.
template <typename T>
__host__ __device__ inline T snn_surrogate_grad_value(T x, int64_t surrogate,
                                                     double alpha) {
    switch (surrogate) {
        case 0:  // straight-through estimator
            return T(1);
        case 1: {  // sigmoid surrogate
            const T ax = static_cast<T>(alpha) * x;
            const T s = T(1) / (T(1) + snn_exp<T>(-ax));
            return static_cast<T>(alpha) * s * (T(1) - s);
        }
        case 2: {  // fast sigmoid surrogate
            const T ax = static_cast<T>(alpha) * x;
            const T denom = T(1) + (ax < T(0) ? -ax : ax);
            return static_cast<T>(alpha) / (denom * denom);
        }
        case 3:  // piecewise-linear (triangular) surrogate
            return (x > -static_cast<T>(alpha) && x < static_cast<T>(alpha))
                       ? T(1)
                       : T(0);
        default:
            return T(1);
    }
}

template <typename T, typename ComputeT>
__global__ void lif_step_kernel(const T* __restrict__ input_p,
                                const T* __restrict__ v_p,
                                const T* __restrict__ decay_p,
                                int64_t decay_n,
                                const T* __restrict__ threshold_p,
                                int64_t threshold_n, int64_t n,
                                bool has_reset, double reset_val,
                                T* __restrict__ spike_p,
                                T* __restrict__ v_new_p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const ComputeT x = static_cast<ComputeT>(input_p[i]);
        const ComputeT vv = static_cast<ComputeT>(v_p[i]);
        const ComputeT dc =
            static_cast<ComputeT>(decay_p[decay_n == 1 ? 0 : i]);
        const ComputeT th =
            static_cast<ComputeT>(threshold_p[threshold_n == 1 ? 0 : i]);
        const ComputeT h = dc * vv + x;
        const ComputeT diff = h - th;
        const ComputeT s = diff >= ComputeT(0) ? ComputeT(1) : ComputeT(0);
        const ComputeT vn = has_reset
                                ? h * (ComputeT(1) - s) +
                                      static_cast<ComputeT>(reset_val) * s
                                : h - th * s;
        spike_p[i] = static_cast<T>(s);
        v_new_p[i] = static_cast<T>(vn);
    }
}

template <typename T, typename ComputeT>
__global__ void lif_step_backward_kernel(
    const T* __restrict__ grad_spike_p, const T* __restrict__ grad_v_p,
    const T* __restrict__ input_p, const T* __restrict__ v_p,
    const T* __restrict__ decay_p, int64_t decay_n,
    const T* __restrict__ threshold_p, int64_t threshold_n,
    const T* __restrict__ spike_p, int64_t n, bool has_reset,
    double reset_val, int64_t surrogate, double alpha, bool detach_reset,
    T* __restrict__ grad_input_p, T* __restrict__ grad_v_init_p,
    T* __restrict__ grad_decay_p, T* __restrict__ grad_threshold_p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const ComputeT x = static_cast<ComputeT>(input_p[i]);
        const ComputeT vv = static_cast<ComputeT>(v_p[i]);
        const ComputeT dc =
            static_cast<ComputeT>(decay_p[decay_n == 1 ? 0 : i]);
        const ComputeT th =
            static_cast<ComputeT>(threshold_p[threshold_n == 1 ? 0 : i]);
        const ComputeT s = static_cast<ComputeT>(spike_p[i]);
        const ComputeT h = dc * vv + x;
        const ComputeT sg =
            snn_surrogate_grad_value<ComputeT>(h - th, surrogate, alpha);
        const ComputeT gs = static_cast<ComputeT>(grad_spike_p[i]);
        const ComputeT gv = static_cast<ComputeT>(grad_v_p[i]);

        ComputeT grad_h;
        ComputeT grad_th;
        if (detach_reset) {
            // The spike is a constant in the voltage-update path.
            grad_h = (has_reset ? (ComputeT(1) - s) : ComputeT(1)) * gv;
            if (has_reset) {
                grad_th = -sg * gs;
            } else {
                grad_th = -sg * gs - s * gv;
            }
        } else {
            if (has_reset) {
                grad_h = ((ComputeT(1) - s) +
                          (static_cast<ComputeT>(reset_val) - h) * sg) *
                             gv +
                         sg * gs;
                grad_th =
                    -sg * (gs + (static_cast<ComputeT>(reset_val) - h) * gv);
            } else {
                grad_h = (ComputeT(1) - th * sg) * gv + sg * gs;
                grad_th = -sg * gs + (-s + th * sg) * gv;
            }
        }

        grad_input_p[i] = static_cast<T>(grad_h);
        grad_v_init_p[i] = static_cast<T>(grad_h * dc);
        grad_decay_p[i] = static_cast<T>(grad_h * vv);
        grad_threshold_p[i] = static_cast<T>(grad_th);
    }
}

template <typename T, typename ComputeT>
__global__ void surrogate_grad_kernel(const T* __restrict__ x_p, int64_t n,
                                     int64_t surrogate, double alpha,
                                     T* __restrict__ out_p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out_p[i] = static_cast<T>(snn_surrogate_grad_value<ComputeT>(
            static_cast<ComputeT>(x_p[i]), surrogate, alpha));
    }
}

template <typename T, typename ComputeT>
void launch_lif_step(const Tensor& input_c, const Tensor& v_c,
                     const Tensor& decay_c, int64_t decay_n,
                     const Tensor& threshold_c, int64_t threshold_n,
                     bool has_reset, double reset_val, Tensor& spike,
                     Tensor& v_new) {
    const int64_t n = input_c.numel();
    if (n == 0) return;
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    lif_step_kernel<T, ComputeT><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        input_c.data_ptr<T>(), v_c.data_ptr<T>(), decay_c.data_ptr<T>(),
        decay_n, threshold_c.data_ptr<T>(), threshold_n, n, has_reset,
        reset_val, spike.data_ptr<T>(), v_new.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename ComputeT>
void launch_lif_step_backward(const Tensor& grad_spike_c,
                              const Tensor& grad_v_c, const Tensor& input_c,
                              const Tensor& v_c, const Tensor& decay_c,
                              int64_t decay_n, const Tensor& threshold_c,
                              int64_t threshold_n, const Tensor& spike_c,
                              bool has_reset, double reset_val,
                              int64_t surrogate, double alpha,
                              bool detach_reset, Tensor& grad_input,
                              Tensor& grad_v_init, Tensor& grad_decay_full,
                              Tensor& grad_threshold_full) {
    const int64_t n = input_c.numel();
    if (n == 0) return;
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    lif_step_backward_kernel<T, ComputeT>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            grad_spike_c.data_ptr<T>(), grad_v_c.data_ptr<T>(),
            input_c.data_ptr<T>(), v_c.data_ptr<T>(), decay_c.data_ptr<T>(),
            decay_n, threshold_c.data_ptr<T>(), threshold_n,
            spike_c.data_ptr<T>(), n, has_reset, reset_val, surrogate, alpha,
            detach_reset, grad_input.data_ptr<T>(),
            grad_v_init.data_ptr<T>(), grad_decay_full.data_ptr<T>(),
            grad_threshold_full.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename ComputeT>
void launch_surrogate_grad(const Tensor& x_c, int64_t surrogate, double alpha,
                           Tensor& out) {
    const int64_t n = x_c.numel();
    if (n == 0) return;
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    surrogate_grad_kernel<T, ComputeT>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            x_c.data_ptr<T>(), n, surrogate, alpha, out.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

// Reduce a full-shape gradient to the shape of the (possibly broadcast)
// parameter it belongs to; see the CPU twin for the semantics.
Tensor snn_sum_to_shape(const Tensor& grad_full, const Tensor& target) {
    auto gs = static_cast<std::vector<int64_t>>(grad_full.shape());
    auto ts = static_cast<std::vector<int64_t>>(target.shape());
    if (gs == ts) return grad_full;
    if (target.numel() == 1) return ops::sum(grad_full);
    const int64_t offset = static_cast<int64_t>(gs.size()) -
                           static_cast<int64_t>(ts.size());
    std::vector<int64_t> dims;
    for (int64_t i = 0; i < static_cast<int64_t>(gs.size()); ++i) {
        const int64_t t = (i < offset) ? 1 : ts[i - offset];
        if (t == 1 && gs[i] != 1) dims.push_back(i);
    }
    if (dims.empty()) return grad_full;
    return ops::view(ops::sum(grad_full, dims, /*keepdim=*/true), ts);
}

std::tuple<Tensor, Tensor> snn_lif_step_cuda(const Tensor& input,
                                             const Tensor& v,
                                             const Tensor& decay,
                                             const Tensor& threshold,
                                             const std::optional<Tensor>& v_reset,
                                             int64_t surrogate, double alpha,
                                             bool detach_reset) {
    TP_CHECK(input.device() == v.device(),
             "snn_lif_step: input and v must be on the same device");
    TP_CHECK(decay.device() == input.device(),
             "snn_lif_step: decay must be on the same device as input");
    TP_CHECK(threshold.device() == input.device(),
             "snn_lif_step: threshold must be on the same device as input");
    TP_CHECK(input.dtype() == v.dtype(),
             "snn_lif_step: input and v must have the same dtype");
    TP_CHECK(input.sizes() == v.sizes(),
             "snn_lif_step: input and v must have the same shape");
    const DType dt = input.dtype();
    TP_CHECK(isFloatingType(dt),
             "snn_lif_step: input must be a floating point tensor");
    TP_CHECK(surrogate >= 0 && surrogate <= 3,
             "snn_lif_step: surrogate must be in [0, 3]");

    const bool has_reset = v_reset.has_value();
    double reset_val = 0.0;
    if (has_reset) {
        TP_CHECK(v_reset->numel() == 1,
                 "snn_lif_step: v_reset must be a scalar tensor");
        reset_val = v_reset->item().toDouble();
    }

    const auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    Tensor input_c = input.contiguous();
    Tensor v_c = v.contiguous();
    const Tensor decay_c = decay.numel() == 1
                               ? decay.to(dt)
                               : decay.to(dt).expand(sizes).contiguous();
    const Tensor threshold_c = threshold.numel() == 1
                                   ? threshold.to(dt)
                                   : threshold.to(dt).expand(sizes).contiguous();
    Tensor spike = Tensor::empty(sizes, dt, input.device());
    Tensor v_new = Tensor::empty(sizes, dt, input.device());

    const int64_t decay_n = decay_c.numel();
    const int64_t threshold_n = threshold_c.numel();
#define TP_SNN_CUDA_DISPATCH(ctype, name_)                                   \
    case DType::name_:                                                       \
        launch_lif_step<ctype, snn_compute_t<ctype>>(                        \
            input_c, v_c, decay_c, decay_n, threshold_c, threshold_n,        \
            has_reset, reset_val, spike, v_new);                             \
        break;
    switch (dt) {
        TP_SNN_CUDA_DISPATCH(float, Float32)
        TP_SNN_CUDA_DISPATCH(double, Float64)
        TP_SNN_CUDA_DISPATCH(tensorplay::Half, Float16)
        TP_SNN_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(TypeError, "snn_lif_step: unsupported dtype");
    }
#undef TP_SNN_CUDA_DISPATCH
    return {spike, v_new};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> snn_lif_step_backward_cuda(
    const Tensor& grad_spike, const Tensor& grad_v, const Tensor& input,
    const Tensor& v, const Tensor& decay, const Tensor& threshold,
    const Tensor& spike, const Tensor& v_new,
    const std::optional<Tensor>& v_reset, int64_t surrogate, double alpha,
    bool detach_reset) {
    (void)v_new;  // the voltage is recomputed from (input, v, decay)
    TP_CHECK(input.device() == v.device(),
             "snn_lif_step_backward: input and v must be on the same device");
    TP_CHECK(decay.device() == input.device(),
             "snn_lif_step_backward: decay must be on the same device as input");
    TP_CHECK(threshold.device() == input.device(),
             "snn_lif_step_backward: threshold must be on the same device as input");
    TP_CHECK(grad_spike.device() == input.device(),
             "snn_lif_step_backward: grad_spike must be on the same device as input");
    TP_CHECK(grad_v.device() == input.device(),
             "snn_lif_step_backward: grad_v must be on the same device as input");
    TP_CHECK(input.dtype() == v.dtype(),
             "snn_lif_step_backward: input and v must have the same dtype");
    TP_CHECK(input.sizes() == v.sizes(),
             "snn_lif_step_backward: input and v must have the same shape");
    const DType dt = input.dtype();
    TP_CHECK(isFloatingType(dt),
             "snn_lif_step_backward: input must be a floating point tensor");
    TP_CHECK(surrogate >= 0 && surrogate <= 3,
             "snn_lif_step_backward: surrogate must be in [0, 3]");

    const bool has_reset = v_reset.has_value();
    double reset_val = 0.0;
    if (has_reset) {
        TP_CHECK(v_reset->numel() == 1,
                 "snn_lif_step_backward: v_reset must be a scalar tensor");
        reset_val = v_reset->item().toDouble();
    }

    const auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    Tensor grad_spike_c = grad_spike.to(dt).contiguous();
    Tensor grad_v_c = grad_v.to(dt).contiguous();
    Tensor input_c = input.contiguous();
    Tensor v_c = v.contiguous();
    Tensor spike_c = spike.to(dt).contiguous();
    const Tensor decay_c = decay.numel() == 1
                               ? decay.to(dt)
                               : decay.to(dt).expand(sizes).contiguous();
    const Tensor threshold_c = threshold.numel() == 1
                                   ? threshold.to(dt)
                                   : threshold.to(dt).expand(sizes).contiguous();

    Tensor grad_input = Tensor::empty(sizes, dt, input.device());
    Tensor grad_v_init = Tensor::empty(sizes, dt, input.device());
    Tensor grad_decay_full = Tensor::empty(sizes, dt, input.device());
    Tensor grad_threshold_full = Tensor::empty(sizes, dt, input.device());

    const int64_t decay_n = decay_c.numel();
    const int64_t threshold_n = threshold_c.numel();
#define TP_SNN_CUDA_BWD_DISPATCH(ctype, name_)                               \
    case DType::name_:                                                       \
        launch_lif_step_backward<ctype, snn_compute_t<ctype>>(               \
            grad_spike_c, grad_v_c, input_c, v_c, decay_c, decay_n,          \
            threshold_c, threshold_n, spike_c, has_reset, reset_val,         \
            surrogate, alpha, detach_reset, grad_input, grad_v_init,         \
            grad_decay_full, grad_threshold_full);                           \
        break;
    switch (dt) {
        TP_SNN_CUDA_BWD_DISPATCH(float, Float32)
        TP_SNN_CUDA_BWD_DISPATCH(double, Float64)
        TP_SNN_CUDA_BWD_DISPATCH(tensorplay::Half, Float16)
        TP_SNN_CUDA_BWD_DISPATCH(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(TypeError, "snn_lif_step_backward: unsupported dtype");
    }
#undef TP_SNN_CUDA_BWD_DISPATCH

    const Tensor grad_decay = snn_sum_to_shape(grad_decay_full, decay_c);
    const Tensor grad_threshold =
        snn_sum_to_shape(grad_threshold_full, threshold_c);
    return {grad_input, grad_v_init, grad_decay, grad_threshold};
}

Tensor snn_surrogate_grad_cuda(const Tensor& x, int64_t surrogate,
                               double alpha) {
    const DType dt = x.dtype();
    TP_CHECK(isFloatingType(dt),
             "snn_surrogate_grad: x must be a floating point tensor");
    TP_CHECK(surrogate >= 0 && surrogate <= 3,
             "snn_surrogate_grad: surrogate must be in [0, 3]");
    Tensor x_c = x.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(x.shape()),
                               dt, x.device());
#define TP_SNN_CUDA_SG_DISPATCH(ctype, name_)                                \
    case DType::name_:                                                       \
        launch_surrogate_grad<ctype, snn_compute_t<ctype>>(x_c, surrogate,   \
                                                           alpha, out);      \
        break;
    switch (dt) {
        TP_SNN_CUDA_SG_DISPATCH(float, Float32)
        TP_SNN_CUDA_SG_DISPATCH(double, Float64)
        TP_SNN_CUDA_SG_DISPATCH(tensorplay::Half, Float16)
        TP_SNN_CUDA_SG_DISPATCH(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(TypeError, "snn_surrogate_grad: unsupported dtype");
    }
#undef TP_SNN_CUDA_SG_DISPATCH
    return out;
}

template <typename T, typename ComputeT>
__global__ void lif_seq_kernel(const T* __restrict__ input_p,
                               const T* __restrict__ v_init_p,
                               const T* __restrict__ decay_p, int64_t decay_n,
                               const T* __restrict__ threshold_p,
                               int64_t threshold_n, int64_t T_steps, int64_t M,
                               bool has_reset, double reset_val,
                               T* __restrict__ spike_p,
                               T* __restrict__ v_final_p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t m = blockIdx.x * blockDim.x + threadIdx.x; m < M;
         m += stride) {
        const ComputeT dc = static_cast<ComputeT>(decay_p[decay_n == 1 ? 0 : m]);
        const ComputeT th =
            static_cast<ComputeT>(threshold_p[threshold_n == 1 ? 0 : m]);
        ComputeT vv = static_cast<ComputeT>(v_init_p[m]);
        for (int64_t t = 0; t < T_steps; ++t) {
            const int64_t idx = t * M + m;
            const ComputeT x = static_cast<ComputeT>(input_p[idx]);
            const ComputeT h = dc * vv + x;
            const ComputeT diff = h - th;
            const ComputeT s = diff >= ComputeT(0) ? ComputeT(1) : ComputeT(0);
            vv = has_reset
                     ? h * (ComputeT(1) - s) +
                           static_cast<ComputeT>(reset_val) * s
                     : h - th * s;
            spike_p[idx] = static_cast<T>(s);
        }
        v_final_p[m] = static_cast<T>(vv);
    }
}

template <typename T, typename ComputeT>
__global__ void lif_seq_backward_kernel(
    const T* __restrict__ grad_spike_p, const T* __restrict__ grad_v_final_p,
    const T* __restrict__ input_p, const T* __restrict__ v_init_p,
    const T* __restrict__ decay_p, int64_t decay_n,
    const T* __restrict__ threshold_p, int64_t threshold_n,
    const T* __restrict__ spike_p, int64_t T_steps, int64_t M, bool has_reset,
    double reset_val, int64_t surrogate, double alpha, bool detach_reset,
    ComputeT* h_buf, ComputeT* v_buf, T* __restrict__ grad_input_p,
    T* __restrict__ grad_v_init_p, T* __restrict__ grad_decay_p,
    T* __restrict__ grad_threshold_p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t m = blockIdx.x * blockDim.x + threadIdx.x; m < M;
         m += stride) {
        const ComputeT dc = static_cast<ComputeT>(decay_p[decay_n == 1 ? 0 : m]);
        const ComputeT th =
            static_cast<ComputeT>(threshold_p[threshold_n == 1 ? 0 : m]);

        // Recompute the forward trajectory from the stored spikes so the
        // backward sees exactly the voltages the forward used.
        ComputeT vv = static_cast<ComputeT>(v_init_p[m]);
        for (int64_t t = 0; t < T_steps; ++t) {
            const int64_t idx = t * M + m;
            const ComputeT x = static_cast<ComputeT>(input_p[idx]);
            const ComputeT s = static_cast<ComputeT>(spike_p[idx]);
            const ComputeT h = dc * vv + x;
            h_buf[idx] = h;
            v_buf[idx] = vv;
            vv = has_reset
                     ? h * (ComputeT(1) - s) +
                           static_cast<ComputeT>(reset_val) * s
                     : h - th * s;
        }

        // Back-propagate the voltage gradient across time (BPTT).
        ComputeT gv = static_cast<ComputeT>(grad_v_final_p[m]);
        ComputeT grad_decay = ComputeT(0);
        ComputeT grad_threshold = ComputeT(0);
        for (int64_t t = T_steps - 1; t >= 0; --t) {
            const int64_t idx = t * M + m;
            const ComputeT s = static_cast<ComputeT>(spike_p[idx]);
            const ComputeT h = h_buf[idx];
            const ComputeT v_prev = v_buf[idx];
            const ComputeT sg = snn_surrogate_grad_value<ComputeT>(
                h - th, surrogate, alpha);
            const ComputeT gs = static_cast<ComputeT>(grad_spike_p[idx]);

            ComputeT grad_h;
            ComputeT grad_th;
            if (detach_reset) {
                grad_h = (has_reset ? (ComputeT(1) - s) : ComputeT(1)) * gv;
                if (has_reset) {
                    grad_th = -sg * gs;
                } else {
                    grad_th = -sg * gs - s * gv;
                }
            } else {
                if (has_reset) {
                    grad_h = ((ComputeT(1) - s) +
                              (static_cast<ComputeT>(reset_val) - h) * sg) *
                                 gv +
                             sg * gs;
                    grad_th = -sg * (gs +
                                     (static_cast<ComputeT>(reset_val) - h) * gv);
                } else {
                    grad_h = (ComputeT(1) - th * sg) * gv + sg * gs;
                    grad_th = -sg * gs + (-s + th * sg) * gv;
                }
            }

            grad_input_p[idx] = static_cast<T>(grad_h);
            grad_decay += grad_h * v_prev;
            grad_threshold += grad_th;
            gv = grad_h * dc;
        }
        grad_v_init_p[m] = static_cast<T>(gv);
        grad_decay_p[m] = static_cast<T>(grad_decay);
        grad_threshold_p[m] = static_cast<T>(grad_threshold);
    }
}

template <typename T, typename ComputeT>
void launch_lif_seq(const Tensor& input_c, const Tensor& v_init_c,
                    const Tensor& decay_c, int64_t decay_n,
                    const Tensor& threshold_c, int64_t threshold_n,
                    int64_t T_steps, int64_t M, bool has_reset, double reset_val,
                    Tensor& spikes, Tensor& v_final) {
    if (M == 0) return;
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((M + kThreads - 1) / kThreads));
    lif_seq_kernel<T, ComputeT>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            input_c.data_ptr<T>(), v_init_c.data_ptr<T>(),
            decay_c.data_ptr<T>(), decay_n, threshold_c.data_ptr<T>(),
            threshold_n, T_steps, M, has_reset, reset_val, spikes.data_ptr<T>(),
            v_final.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename ComputeT>
void launch_lif_seq_backward(const Tensor& grad_spikes_c,
                             const Tensor& grad_v_final_c,
                             const Tensor& input_c, const Tensor& v_init_c,
                             const Tensor& decay_c, int64_t decay_n,
                             const Tensor& threshold_c, int64_t threshold_n,
                             const Tensor& spike_c, int64_t T_steps, int64_t M,
                             bool has_reset, double reset_val,
                             int64_t surrogate, double alpha,
                             bool detach_reset, Tensor& h_buf, Tensor& v_buf,
                             Tensor& grad_input, Tensor& grad_v_init,
                             Tensor& grad_decay_full,
                             Tensor& grad_threshold_full) {
    if (M == 0) return;
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((M + kThreads - 1) / kThreads));
    lif_seq_backward_kernel<T, ComputeT>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            grad_spikes_c.data_ptr<T>(), grad_v_final_c.data_ptr<T>(),
            input_c.data_ptr<T>(), v_init_c.data_ptr<T>(),
            decay_c.data_ptr<T>(), decay_n, threshold_c.data_ptr<T>(),
            threshold_n, spike_c.data_ptr<T>(), T_steps, M, has_reset,
            reset_val, surrogate, alpha, detach_reset, h_buf.data_ptr<ComputeT>(),
            v_buf.data_ptr<ComputeT>(), grad_input.data_ptr<T>(),
            grad_v_init.data_ptr<T>(), grad_decay_full.data_ptr<T>(),
            grad_threshold_full.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

std::tuple<Tensor, Tensor> snn_lif_seq_cuda(
    const Tensor& input, const Tensor& v_init, const Tensor& decay,
    const Tensor& threshold, const std::optional<Tensor>& v_reset,
    int64_t surrogate, double alpha, bool detach_reset) {
    TP_CHECK(input.dim() >= 2,
             "snn_lif_seq: input must have at least 2 dims [T, B, ...]");
    const auto input_sizes = static_cast<std::vector<int64_t>>(input.shape());
    const int64_t T = input_sizes[0];
    std::vector<int64_t> v_target(input_sizes.begin() + 1, input_sizes.end());
    TP_CHECK(v_init.dim() <= static_cast<int64_t>(v_target.size()),
             "snn_lif_seq: v_init has more dims than input[1:]");
    TP_CHECK(decay.dim() <= static_cast<int64_t>(v_target.size()),
             "snn_lif_seq: decay has more dims than input[1:]");
    TP_CHECK(threshold.dim() <= static_cast<int64_t>(v_target.size()),
             "snn_lif_seq: threshold has more dims than input[1:]");
    TP_CHECK(input.device() == v_init.device(),
             "snn_lif_seq: input and v_init must be on the same device");
    TP_CHECK(decay.device() == input.device(),
             "snn_lif_seq: decay must be on the same device as input");
    TP_CHECK(threshold.device() == input.device(),
             "snn_lif_seq: threshold must be on the same device as input");
    const DType dt = input.dtype();
    TP_CHECK(isFloatingType(dt),
             "snn_lif_seq: input must be a floating point tensor");
    TP_CHECK(surrogate >= 0 && surrogate <= 3,
             "snn_lif_seq: surrogate must be in [0, 3]");

    const bool has_reset = v_reset.has_value();
    double reset_val = 0.0;
    if (has_reset) {
        TP_CHECK(v_reset->numel() == 1,
                 "snn_lif_seq: v_reset must be a scalar tensor");
        reset_val = v_reset->item().toDouble();
    }

    Tensor input_c = input.contiguous();
    const Tensor v_init_c = v_init.to(dt).expand(v_target).contiguous();
    const Tensor decay_c = decay.numel() == 1
                               ? decay.to(dt)
                               : decay.to(dt).expand(v_target).contiguous();
    const Tensor threshold_c = threshold.numel() == 1
                                    ? threshold.to(dt)
                                    : threshold.to(dt).expand(v_target).contiguous();
    Tensor spikes = Tensor::empty(input_sizes, dt, input.device());
    Tensor v_final = Tensor::empty(v_target, dt, input.device());
    const int64_t M = v_final.numel();

    const int64_t decay_n = decay_c.numel();
    const int64_t threshold_n = threshold_c.numel();
#define TP_SNN_CUDA_SEQ_DISPATCH(ctype, name_)                               \
    case DType::name_:                                                       \
        launch_lif_seq<ctype, snn_compute_t<ctype>>(                         \
            input_c, v_init_c, decay_c, decay_n, threshold_c, threshold_n,   \
            T, M, has_reset, reset_val, spikes, v_final);                    \
        break;
    switch (dt) {
        TP_SNN_CUDA_SEQ_DISPATCH(float, Float32)
        TP_SNN_CUDA_SEQ_DISPATCH(double, Float64)
        TP_SNN_CUDA_SEQ_DISPATCH(tensorplay::Half, Float16)
        TP_SNN_CUDA_SEQ_DISPATCH(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(TypeError, "snn_lif_seq: unsupported dtype");
    }
#undef TP_SNN_CUDA_SEQ_DISPATCH
    return {spikes, v_final};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> snn_lif_seq_backward_cuda(
    const Tensor& grad_spikes, const Tensor& grad_v_final,
    const Tensor& input, const Tensor& v_init, const Tensor& decay,
    const Tensor& threshold, const Tensor& spikes, const Tensor& v_final,
    const std::optional<Tensor>& v_reset, int64_t surrogate, double alpha,
    bool detach_reset) {
    (void)v_final;  // the trajectory is recomputed from (input, v_init, spikes)
    TP_CHECK(input.dim() >= 2,
             "snn_lif_seq_backward: input must have at least 2 dims");
    const auto input_sizes = static_cast<std::vector<int64_t>>(input.shape());
    const int64_t T = input_sizes[0];
    std::vector<int64_t> v_target(input_sizes.begin() + 1, input_sizes.end());
    TP_CHECK(grad_spikes.device() == input.device(),
             "snn_lif_seq_backward: grad_spikes must be on the same device");
    TP_CHECK(grad_v_final.device() == input.device(),
             "snn_lif_seq_backward: grad_v_final must be on the same device");
    const DType dt = input.dtype();
    TP_CHECK(isFloatingType(dt),
             "snn_lif_seq_backward: input must be a floating point tensor");
    TP_CHECK(surrogate >= 0 && surrogate <= 3,
             "snn_lif_seq_backward: surrogate must be in [0, 3]");

    const bool has_reset = v_reset.has_value();
    double reset_val = 0.0;
    if (has_reset) {
        TP_CHECK(v_reset->numel() == 1,
                 "snn_lif_seq_backward: v_reset must be a scalar tensor");
        reset_val = v_reset->item().toDouble();
    }

    Tensor input_c = input.contiguous();
    Tensor grad_spikes_c = grad_spikes.to(dt).contiguous();
    const Tensor grad_v_final_c = grad_v_final.to(dt).expand(v_target).contiguous();
    const Tensor v_init_c = v_init.to(dt).expand(v_target).contiguous();
    const Tensor spike_c = spikes.to(dt).contiguous();
    const Tensor decay_c = decay.numel() == 1
                               ? decay.to(dt)
                               : decay.to(dt).expand(v_target).contiguous();
    const Tensor threshold_c = threshold.numel() == 1
                                    ? threshold.to(dt)
                                    : threshold.to(dt).expand(v_target).contiguous();

    Tensor grad_input = Tensor::empty(input_sizes, dt, input.device());
    Tensor grad_v_init = Tensor::empty(v_target, dt, input.device());
    Tensor grad_decay_full = Tensor::empty(v_target, dt, input.device());
    Tensor grad_threshold_full = Tensor::empty(v_target, dt, input.device());
    const DType buf_dt =
        (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor h_buf = Tensor::empty(input_sizes, buf_dt, input.device());
    Tensor v_buf = Tensor::empty(input_sizes, buf_dt, input.device());
    const int64_t M = grad_v_init.numel();

    const int64_t decay_n = decay_c.numel();
    const int64_t threshold_n = threshold_c.numel();
#define TP_SNN_CUDA_SEQ_BWD_DISPATCH(ctype, name_)                           \
    case DType::name_:                                                       \
        launch_lif_seq_backward<ctype, snn_compute_t<ctype>>(                \
            grad_spikes_c, grad_v_final_c, input_c, v_init_c, decay_c,        \
            decay_n, threshold_c, threshold_n, spike_c, T, M, has_reset,     \
            reset_val, surrogate, alpha, detach_reset, h_buf, v_buf,         \
            grad_input, grad_v_init, grad_decay_full, grad_threshold_full);  \
        break;
    switch (dt) {
        TP_SNN_CUDA_SEQ_BWD_DISPATCH(float, Float32)
        TP_SNN_CUDA_SEQ_BWD_DISPATCH(double, Float64)
        TP_SNN_CUDA_SEQ_BWD_DISPATCH(tensorplay::Half, Float16)
        TP_SNN_CUDA_SEQ_BWD_DISPATCH(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(TypeError, "snn_lif_seq_backward: unsupported dtype");
    }
#undef TP_SNN_CUDA_SEQ_BWD_DISPATCH

    const Tensor grad_decay = snn_sum_to_shape(grad_decay_full, decay_c);
    const Tensor grad_threshold =
        snn_sum_to_shape(grad_threshold_full, threshold_c);
    return {grad_input, grad_v_init, grad_decay, grad_threshold};
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SnnKernels) {
    m.impl("snn_lif_step", snn_lif_step_cuda);
    m.impl("snn_lif_step_backward", snn_lif_step_backward_cuda);
    m.impl("snn_lif_seq", snn_lif_seq_cuda);
    m.impl("snn_lif_seq_backward", snn_lif_seq_backward_cuda);
    m.impl("snn_surrogate_grad", snn_surrogate_grad_cuda);
}

}  // namespace cuda
}  // namespace tensorplay