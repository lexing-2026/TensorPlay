#include "Tensor.h"
#include "Generator.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "CUDAContext.h"
#include <cuda_runtime.h>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cmath>

namespace tensorplay {
namespace cuda {

// ATen alignment: reduced floating types (Half/BFloat16) compute in float32.
template <typename T> struct LossMath { using type = T; };
template <> struct LossMath<tensorplay::Half> { using type = float; };
template <> struct LossMath<tensorplay::BFloat16> { using type = float; };

// DType of the accumulator matching LossMath<T>::type.
template <typename M> struct LossAccDType;
template <> struct LossAccDType<float> { static constexpr DType value = DType::Float32; };
template <> struct LossAccDType<double> { static constexpr DType value = DType::Float64; };

template <typename T, typename TargetT>
__global__ void nll_loss_forward_kernel(
    int64_t n, int64_t C,
    const T* input,
    const TargetT* target,
    const T* weight,
    T* output,
    int64_t ignore_index) {
    using M = typename LossMath<T>::type;

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        TargetT t = target[i];
        if (t == ignore_index) {
            output[i] = static_cast<T>(0);
            return;
        }
        if (t >= 0 && t < C) {
            M val = static_cast<M>(input[i * C + t]);
            M w = (weight != nullptr) ? static_cast<M>(weight[t]) : static_cast<M>(1);
            output[i] = static_cast<T>(-val * w);
        } else {
            output[i] = static_cast<T>(0);
        }
    }
}

// Reduced-precision outputs accumulate in fp32 scalars: BF16 atomicAdd is
// sm_90+ only, and fp32 accumulation matches ATen's acc_type semantics.
template <typename T, typename ACC>
__global__ void nll_loss_atomic_kernel(
    int64_t n, int64_t C,
    const T* input,
    const int64_t* target,
    const T* weight,
    ACC* output_loss,
    ACC* output_weight,
    int64_t ignore_index) {
    using M = typename LossMath<T>::type;
    static_assert(std::is_same<ACC, M>::value,
                  "nll_loss accumulation type must match the math type");

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t t = target[i];
        if (t != ignore_index && t >= 0 && t < C) {
            M val = static_cast<M>(input[i * C + t]);
            M w = (weight != nullptr) ? static_cast<M>(weight[t]) : static_cast<M>(1);

            atomicAdd(output_loss, static_cast<ACC>(-val * w));
            if (output_weight) {
                atomicAdd(output_weight, static_cast<ACC>(w));
            }
        }
    }
}

// Applies the mean reduction and converts the fp32 accumulators to T.
template <typename T, typename ACC>
__global__ void nll_loss_finalize_kernel(
    const ACC* loss_acc,
    const ACC* total_weight_acc,
    bool divide_by_weight,
    T* output_loss,
    T* output_weight) {
    ACC loss = loss_acc[0];
    if (divide_by_weight && total_weight_acc[0] != ACC(0)) loss /= total_weight_acc[0];
    output_loss[0] = static_cast<T>(loss);
    if (output_weight) output_weight[0] = static_cast<T>(total_weight_acc[0]);
}

std::tuple<Tensor, Tensor> nll_loss_cuda(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);

    Tensor weight;
    if (weight_opt.has_value() && weight_opt->defined()) weight = *weight_opt;

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    const auto stream = getCurrentCUDAStream().stream();

#define NLL_LOSS_CASE(ctype, name)                                          \
    case DType::name: {                                                     \
        using acc_t = typename LossMath<ctype>::type;                       \
        /* None */ if (reduction == 0) {                                     \
            Tensor losses = Tensor::empty({N}, input.dtype(), input.device()); \
            nll_loss_forward_kernel<ctype, int64_t><<<blocks, threads, 0, stream>>>( \
                N, C, input.data_ptr<ctype>(), target.data_ptr<int64_t>(),  \
                weight.defined() ? weight.data_ptr<ctype>() : nullptr,      \
                losses.data_ptr<ctype>(), ignore_index);                    \
            return std::make_tuple(losses, Tensor());                       \
        }                                                                   \
        Tensor result = Tensor::zeros({}, input.dtype(), input.device());   \
        Tensor total_weight = Tensor::zeros({}, input.dtype(), input.device()); \
        constexpr DType acc_dt = LossAccDType<acc_t>::value;                 \
        Tensor loss_acc = Tensor::zeros({}, acc_dt, input.device());        \
        Tensor weight_acc = Tensor::zeros({}, acc_dt, input.device());      \
        nll_loss_atomic_kernel<ctype, acc_t><<<blocks, threads, 0, stream>>>( \
            N, C, input.data_ptr<ctype>(), target.data_ptr<int64_t>(),      \
            weight.defined() ? weight.data_ptr<ctype>() : nullptr,          \
            loss_acc.data_ptr<acc_t>(),                                     \
            weight_acc.data_ptr<acc_t>(),                                   \
            ignore_index);                                                  \
        nll_loss_finalize_kernel<ctype, acc_t><<<1, 1, 0, stream>>>(        \
            loss_acc.data_ptr<acc_t>(), weight_acc.data_ptr<acc_t>(),       \
            reduction == 1,                                                 \
            result.data_ptr<ctype>(),                                       \
            total_weight.data_ptr<ctype>());                                \
        return std::make_tuple(result, total_weight);                       \
    }

    switch (input.dtype()) {
        NLL_LOSS_CASE(float, Float32)
        NLL_LOSS_CASE(double, Float64)
        NLL_LOSS_CASE(tensorplay::Half, Float16)
        NLL_LOSS_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "nll_loss CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
#undef NLL_LOSS_CASE
}

template <typename T, typename TargetT>
__global__ void nll_loss_backward_kernel(
    int64_t n, int64_t C,
    const T* grad_output,
    const TargetT* target,
    const T* weight,
    const T* total_weight,
    T* grad_input,
    int64_t ignore_index,
    int reduction) {
    using M = typename LossMath<T>::type;

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        TargetT t = target[i];
        if (t == ignore_index || t < 0 || t >= C) return;

        M w = (weight != nullptr) ? static_cast<M>(weight[t]) : static_cast<M>(1);
        M g = (reduction == 0) ? static_cast<M>(grad_output[i]) : static_cast<M>(grad_output[0]);

        if (reduction == 1 && total_weight) { // Mean
             g /= static_cast<M>(total_weight[0]);
        }

        grad_input[i * C + t] = static_cast<T>(-g * w);
    }
}

Tensor nll_loss_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("nll_loss_backward_cuda");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    Tensor grad_input = Tensor::zeros_like(input);

    Tensor weight;
    if (weight_opt.has_value() && weight_opt->defined()) weight = *weight_opt;

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    const auto stream = getCurrentCUDAStream().stream();

#define NLL_BACKWARD_CASE(ctype, name)                                      \
    case DType::name:                                                       \
        nll_loss_backward_kernel<ctype, int64_t><<<blocks, threads, 0, stream>>>( \
            N, C,                                                           \
            grad_output.data_ptr<ctype>(),                                  \
            target.data_ptr<int64_t>(),                                     \
            weight.defined() ? weight.data_ptr<ctype>() : nullptr,          \
            total_weight.defined() ? total_weight.data_ptr<ctype>() : nullptr, \
            grad_input.data_ptr<ctype>(),                                   \
            ignore_index,                                                   \
            (int)reduction);                                                \
        break;

    switch (input.dtype()) {
        NLL_BACKWARD_CASE(float, Float32)
        NLL_BACKWARD_CASE(double, Float64)
        NLL_BACKWARD_CASE(tensorplay::Half, Float16)
        NLL_BACKWARD_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "nll_loss_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
#undef NLL_BACKWARD_CASE

    return grad_input;
}

Tensor mse_loss_cuda(const Tensor& input, const Tensor& target, int64_t reduction) {
    Tensor diff = input - target;
    Tensor sq_diff = diff * diff;
    if (reduction == 0) return sq_diff;
    if (reduction == 1) return sq_diff.mean();
    if (reduction == 2) return sq_diff.sum();
    TP_THROW(ValueError, "Invalid reduction mode");
}

template <typename T>
__global__ void mse_loss_backward_kernel_cuda_impl(int64_t n, const T* grad_output, const T* input, const T* target, T* grad_input, int64_t reduction, typename LossMath<T>::type scale) {
    using M = typename LossMath<T>::type;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        M diff = static_cast<M>(input[i]) - static_cast<M>(target[i]);
        M g = (reduction == 0) ? static_cast<M>(grad_output[i]) : static_cast<M>(grad_output[0]);
        grad_input[i] = static_cast<T>(scale * diff * g);
    }
}

Tensor mse_loss_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
    int64_t n = input.numel();
    Tensor grad_input = Tensor::empty_like(input);

    double scale = 2.0;
    if (reduction == 1) { // Mean
        scale /= (double)n;
    }

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    const auto stream = getCurrentCUDAStream().stream();

#define MSE_BACKWARD_CASE(ctype, name)                                      \
    case DType::name: {                                                     \
        using math_t = typename LossMath<ctype>::type;                      \
        mse_loss_backward_kernel_cuda_impl<ctype><<<blocks, threads, 0, stream>>>( \
            n,                                                              \
            grad_output.data_ptr<ctype>(),                                  \
            input.data_ptr<ctype>(),                                        \
            target.data_ptr<ctype>(),                                       \
            grad_input.data_ptr<ctype>(),                                   \
            reduction,                                                      \
            static_cast<math_t>(scale));                                    \
        break;                                                              \
    }

    switch (input.dtype()) {
        MSE_BACKWARD_CASE(float, Float32)
        MSE_BACKWARD_CASE(double, Float64)
        MSE_BACKWARD_CASE(tensorplay::Half, Float16)
        MSE_BACKWARD_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "mse_loss_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
#undef MSE_BACKWARD_CASE

    return grad_input;
}

// Torch-aligned loss family: composition-style CUDA kernels. All elementwise
// primitives (where/log/exp/sigmoid/clamp/sum/mean) dispatch to their CUDA
// implementations, so these mirror the CPU compositions exactly.
// -----------------------------------------------------------------------------

namespace {

Tensor cuda_loss_reduce(const Tensor& x, int64_t reduction) {
    if (reduction == 0) return x;
    if (reduction == 1) return x.mean();
    if (reduction == 2) return x.sum();
    TP_THROW(ValueError, "Invalid reduction mode");
}

Tensor cuda_scale_grad(const Tensor& g, int64_t reduction, int64_t numel) {
    if (reduction == 1) return g / static_cast<double>(numel);
    return g;
}

} // anonymous namespace

Tensor l1_loss_cuda2(const Tensor& input, const Tensor& target, int64_t reduction) {
    return cuda_loss_reduce((input - target).abs(), reduction);
}

Tensor l1_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target, int64_t reduction) {
    Tensor g = (input - target).sign() * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor smooth_l1_loss_cuda2(const Tensor& input, const Tensor& target,
                            int64_t reduction, double beta) {
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(beta)), diff * diff * (0.5 / beta),
                                absd - 0.5 * beta);
    return cuda_loss_reduce(loss, reduction);
}

Tensor smooth_l1_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                     const Tensor& target, int64_t reduction, double beta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(beta)), diff / beta, diff.sign()) * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor huber_loss_cuda2(const Tensor& input, const Tensor& target,
                        int64_t reduction, double delta) {
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(delta)), diff * diff * 0.5,
                                delta * (absd - 0.5 * delta));
    return cuda_loss_reduce(loss, reduction);
}

Tensor huber_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                 const Tensor& target, int64_t reduction, double delta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(delta)), diff, delta * diff.sign()) * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor kl_div_cuda2(const Tensor& input, const Tensor& target,
                    int64_t reduction, bool log_target) {
    Tensor nz = (target.ne(0)).to(input.dtype());
    Tensor loss;
    if (!log_target) {
        Tensor t_safe = target + (1.0 - nz);
        loss = (t_safe.log() * target - target * input) * nz;
    } else {
        Tensor t = target.exp();
        loss = (t * (target - input)) * nz;
    }
    return cuda_loss_reduce(loss, reduction);
}

Tensor kl_div_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                             const Tensor& target, int64_t reduction, bool log_target) {
    Tensor nz = (target.ne(0)).to(input.dtype());
    Tensor g;
    if (!log_target) {
        g = (-target * nz) * grad_output;
    } else {
        g = -(target.exp()) * grad_output;
    }
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor binary_cross_entropy_cuda2(const Tensor& input, const Tensor& target,
                                  const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor x = input.clamp(0.0, 1.0);
    Tensor loss = -(x.log() * target + (-x + 1.0).log() * (-target + 1.0));
    if (weight.has_value() && weight->defined()) loss = loss * weight.value();
    return cuda_loss_reduce(loss, reduction);
}

Tensor binary_cross_entropy_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& target,
                                           const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor x = input.clamp(0.0, 1.0);
    Tensor eps = Tensor::full_like(x, 1e-12);
    Tensor g = (x - target) / Tensor::maximum(x * (-x + 1.0), eps);
    if (weight.has_value() && weight->defined()) g = g * weight.value();
    g = g * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor margin_ranking_loss_cuda2(const Tensor& input1, const Tensor& input2,
                                 const Tensor& target, double margin, int64_t reduction) {
    Tensor loss = (-(input1 - input2) * target + margin).clamp(Scalar(0.0), std::nullopt);
    return cuda_loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> margin_ranking_loss_backward_cuda2(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    Tensor active = ((-(input1 - input2) * target + margin).gt(Scalar(0.0))).to(input1.dtype());
    Tensor g = -active * target * grad_output;
    g = cuda_scale_grad(g, reduction, input1.numel());
    return std::make_tuple(g, -g);
}

Tensor hinge_embedding_loss_cuda2(const Tensor& input, const Tensor& target,
                                  double margin, int64_t reduction) {
    Tensor z = Tensor::zeros_like(input);
    Tensor loss = Tensor::where(target.eq(1), input,
                  Tensor::where(target.eq(-1), (margin - input).clamp(Scalar(0.0), std::nullopt), z));
    return cuda_loss_reduce(loss, reduction);
}

Tensor hinge_embedding_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& target, double margin, int64_t reduction) {
    Tensor z = Tensor::zeros_like(input);
    Tensor g = Tensor::where(target.eq(1), Tensor::ones_like(input),
               Tensor::where(target.eq(-1), ((margin - input).gt(Scalar(0.0))).to(input.dtype()), z));
    g = g * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor soft_margin_loss_cuda2(const Tensor& input, const Tensor& target, int64_t reduction) {
    Tensor loss = ((input * target) * -1.0).exp().add(1.0).log();
    return cuda_loss_reduce(loss, reduction);
}

Tensor soft_margin_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& target, int64_t reduction) {
    Tensor g = -target * ((input * target) * -1.0).exp().sigmoid() * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

Tensor cosine_embedding_loss_cuda2(const Tensor& x1, const Tensor& x2,
                                   const Tensor& target, double margin, int64_t reduction) {
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = (n1 * n2).sqrt();
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;
    Tensor zero = Tensor::zeros_like(cos);
    Tensor loss = Tensor::where(target.eq(1), zero + (1.0 - cos),
                  Tensor::where(target.eq(-1), (cos - margin).clamp(Scalar(margin), std::nullopt),
                                zero + (1.0 - cos - margin).clamp(Scalar(0.0), std::nullopt)));
    return cuda_loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> cosine_embedding_loss_backward_cuda2(
        const Tensor& grad_output, const Tensor& x1, const Tensor& x2,
        const Tensor& target, double margin, int64_t reduction) {
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = ((n1 * n2).sqrt()).clamp(Scalar(1e-12), std::nullopt);
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;

    Tensor ones_row = Tensor::ones({x1.size(0)}, x1.dtype(), x1.device());
    Tensor dl_dcos = Tensor::where(target.eq(1), -1.0 * ones_row,
                     Tensor::where(target.eq(-1),
                         ((cos - margin).gt(Scalar(0.0))).to(x1.dtype()),
                         ((1.0 - cos - margin).gt(Scalar(0.0))).to(x1.dtype()) * -1.0));

    if (reduction == 1) dl_dcos = dl_dcos / static_cast<double>(x1.size(0));

    Tensor c = cos.unsqueeze(1);
    Tensor g1 = (x2 / d.unsqueeze(1)) - c * (x1 / n1.unsqueeze(1));
    Tensor g2 = (x1 / d.unsqueeze(1)) - c * (x2 / n2.unsqueeze(1));
    g1 = g1 * (dl_dcos * grad_output).unsqueeze(1);
    g2 = g2 * (dl_dcos * grad_output).unsqueeze(1);
    return std::make_tuple(g1, g2);
}

Tensor poisson_nll_loss_cuda2(const Tensor& input, const Tensor& target,
                              bool log_input, bool full, double eps, int64_t reduction) {
    Tensor loss;
    if (log_input) {
        loss = input.exp() - target * input;
    } else {
        loss = input - target * (input + eps).log();
    }
    if (full) {
        Tensor pos = target.gt(0).to(input.dtype());
        Tensor t_safe = target + (1.0 - pos);
        Tensor stirling = ((t_safe.log() * target) - t_safe +
                           (t_safe * (2.0 * M_PI)).log() * 0.5) * pos;
        loss = loss + stirling;
    }
    return cuda_loss_reduce(loss, reduction);
}

Tensor poisson_nll_loss_backward_cuda2(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& target, bool log_input, bool full,
                                       double eps, int64_t reduction) {
    Tensor g;
    if (log_input) {
        g = input.exp() - target;
    } else {
        g = 1.0 - target / (input + eps);
    }
    g = g * grad_output;
    return cuda_scale_grad(g, reduction, input.numel());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LossKernels) {
    m.impl("nll_loss", nll_loss_cuda);
    m.impl("nll_loss_backward", nll_loss_backward_cuda);
    m.impl("mse_loss", mse_loss_cuda);
    m.impl("mse_loss_backward", mse_loss_backward_cuda);
    m.impl("tp_l1_loss", l1_loss_cuda2);
    m.impl("tp_l1_loss_backward", l1_loss_backward_cuda2);
    m.impl("tp_smooth_l1_loss", smooth_l1_loss_cuda2);
    m.impl("tp_smooth_l1_loss_backward", smooth_l1_loss_backward_cuda2);
    m.impl("tp_huber_loss", huber_loss_cuda2);
    m.impl("tp_huber_loss_backward", huber_loss_backward_cuda2);
    m.impl("tp_kl_div", kl_div_cuda2);
    m.impl("tp_kl_div_backward", kl_div_backward_cuda2);
    m.impl("tp_binary_cross_entropy", binary_cross_entropy_cuda2);
    m.impl("tp_binary_cross_entropy_backward", binary_cross_entropy_backward_cuda2);
    m.impl("tp_margin_ranking_loss", margin_ranking_loss_cuda2);
    m.impl("tp_margin_ranking_loss_backward", margin_ranking_loss_backward_cuda2);
    m.impl("tp_hinge_embedding_loss", hinge_embedding_loss_cuda2);
    m.impl("tp_hinge_embedding_loss_backward", hinge_embedding_loss_backward_cuda2);
    m.impl("tp_soft_margin_loss", soft_margin_loss_cuda2);
    m.impl("tp_soft_margin_loss_backward", soft_margin_loss_backward_cuda2);
    m.impl("tp_cosine_embedding_loss", cosine_embedding_loss_cuda2);
    m.impl("tp_cosine_embedding_loss_backward", cosine_embedding_loss_backward_cuda2);
    m.impl("tp_poisson_nll_loss", poisson_nll_loss_cuda2);
    m.impl("tp_poisson_nll_loss_backward", poisson_nll_loss_backward_cuda2);
}

} // namespace cuda
} // namespace tensorplay

// -----------------------------------------------------------------------------