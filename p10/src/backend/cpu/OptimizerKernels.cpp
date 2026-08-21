#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

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
    if (grads.size() != count || first_state.size() != count ||
        second_state.size() != count || third_state.size() != count) {
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

        const Tensor* states[] = {&first_state[i], &second_state[i]};
        const bool required[] = {require_first_state, require_second_state};
        for (size_t state_index = 0; state_index < 2; ++state_index) {
            const Tensor* state = states[state_index];
            if (!required[state_index]) continue;
            if (!state->defined()) {
                TP_THROW(ValueError, std::string(op_name) +
                    ": required optimizer state is undefined");
            }
            if (!state->is_contiguous() || state->shape() != param.shape() ||
                state->dtype() != param.dtype() || state->device() != param.device()) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": optimizer state must match its parameter layout");
            }
        }
        if (require_third_state) {
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
    const scalar_t alpha = static_cast<scalar_t>(lr);
    const scalar_t momentum_value = static_cast<scalar_t>(momentum);
    const scalar_t dampening_value = static_cast<scalar_t>(dampening);
    const scalar_t decay_value = static_cast<scalar_t>(weight_decay);

    struct WorkItem {
        size_t list_index;
        int64_t begin;
        int64_t end;
    };
    std::vector<WorkItem> work;
    work.reserve(1 + params.size());
    for (size_t list_index = 0; list_index < params.size(); ++list_index) {
        const int64_t n = params[list_index].numel();
        for (int64_t begin = 0; begin < n; begin += GRAIN_SIZE) {
            work.push_back({
                list_index,
                begin,
                std::min<int64_t>(begin + GRAIN_SIZE, n),
            });
        }
    }

    // Match Torch's foreach optimizer contract: split all parameter tensors
    // into one horizontally fused work list, then schedule that list once.
    // Scheduling each parameter independently leaves large ResNet tensors
    // serialized behind a single worker and pays one barrier per tensor.
    parallel_for(0, static_cast<int64_t>(work.size()), 1,
                 [&](int64_t work_begin, int64_t work_end) {
        for (int64_t work_index = work_begin; work_index < work_end; ++work_index) {
            const auto& item = work[static_cast<size_t>(work_index)];
            scalar_t* param = params[item.list_index].template data_ptr<scalar_t>();
            const scalar_t* grad = grads[item.list_index].template data_ptr<scalar_t>();
            scalar_t* buffer = momentum_buffers[item.list_index].defined()
                ? momentum_buffers[item.list_index].template data_ptr<scalar_t>() : nullptr;

            for (int64_t i = item.begin; i < item.end; ++i) {
                scalar_t update = grad[i];
                if (weight_decay != 0.0) {
                    update += decay_value * param[i];
                }

                if (momentum != 0.0) {
                    if (first_momentum_step) {
                        buffer[i] = update;
                        if (nesterov) {
                            update += momentum_value * buffer[i];
                        } else {
                            update = buffer[i];
                        }
                    } else if (nesterov) {
                        buffer[i] = momentum_value * buffer[i] +
                            (scalar_t(1) - dampening_value) * update;
                        update += momentum_value * buffer[i];
                    } else {
                        buffer[i] = momentum_value * buffer[i] +
                            (scalar_t(1) - dampening_value) * update;
                        update = buffer[i];
                    }
                }
                param[i] -= alpha * update;
            }
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
    const scalar_t beta1_value = static_cast<scalar_t>(beta1);
    const scalar_t beta2_value = static_cast<scalar_t>(beta2);
    const scalar_t one_minus_beta1 = static_cast<scalar_t>(1.0 - beta1);
    const scalar_t one_minus_beta2 = static_cast<scalar_t>(1.0 - beta2);
    const scalar_t decay_value = static_cast<scalar_t>(weight_decay);
    const scalar_t eps_value = static_cast<scalar_t>(eps);

    for (size_t list_index = 0; list_index < params.size(); ++list_index) {
        scalar_t* param = params[list_index].data_ptr<scalar_t>();
        const scalar_t* grad = grads[list_index].data_ptr<scalar_t>();
        scalar_t* exp_avg = exp_avgs[list_index].data_ptr<scalar_t>();
        scalar_t* exp_avg_sq = exp_avg_sqs[list_index].data_ptr<scalar_t>();
        scalar_t* max_exp_avg_sq = amsgrad
            ? max_exp_avg_sqs[list_index].data_ptr<scalar_t>() : nullptr;
        const int64_t n = params[list_index].numel();
        const int64_t step = steps[list_index];
        const scalar_t bias_correction1 = static_cast<scalar_t>(
            1.0 - std::pow(beta1, static_cast<double>(step)));
        const scalar_t bias_correction2 = static_cast<scalar_t>(
            1.0 - std::pow(beta2, static_cast<double>(step)));
        const scalar_t step_size = static_cast<scalar_t>(lr) / bias_correction1;
        const scalar_t correction2_sqrt = static_cast<scalar_t>(std::sqrt(
            static_cast<double>(bias_correction2)));

        for (int64_t i = 0; i < n; ++i) {
            scalar_t g = grad[i];
            if (weight_decay != 0.0) {
                g += decay_value * param[i];
            }
            exp_avg[i] = beta1_value * exp_avg[i] + one_minus_beta1 * g;
            exp_avg_sq[i] = beta2_value * exp_avg_sq[i] +
                one_minus_beta2 * g * g;

            scalar_t second_moment = exp_avg_sq[i];
            if (amsgrad) {
                if (max_exp_avg_sq[i] < second_moment) {
                    max_exp_avg_sq[i] = second_moment;
                }
                second_moment = max_exp_avg_sq[i];
            }
            const scalar_t denom = static_cast<scalar_t>(std::sqrt(
                static_cast<double>(second_moment)) /
                static_cast<double>(correction2_sqrt)) + eps_value;
            param[i] -= step_size * exp_avg[i] / denom;
        }
    }
}

// Fused optimizers deliberately live in the native backend, just like
// ATen's Fused{SGD,Adam,Adagrad}.  The Python optimizer only groups tensors
// and selects the overload; it must not rebuild these algorithms from
// pointwise Python calls.  `math_t` is the accumulation type used by the
// half/bfloat16 paths, matching the opmath type used by Torch's CPU kernels.
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
    const math_t beta1_value = static_cast<math_t>(beta1);
    const math_t beta2_value = static_cast<math_t>(beta2);
    const math_t one_minus_beta1 = static_cast<math_t>(1.0 - beta1);
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
    for (size_t list_index = 0; list_index < params.size(); ++list_index) {
        const double step = state_steps[list_index].item().toDouble();
        const math_t clr = static_cast<math_t>(
            lr / (1.0 + (step - 1.0) * lr_decay));
        scalar_t* param = params[list_index].data_ptr<scalar_t>();
        scalar_t* grad = grads[list_index].data_ptr<scalar_t>();
        scalar_t* state_sum = state_sums[list_index].data_ptr<scalar_t>();
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
                math_t sum = static_cast<math_t>(state_sum[i]) + g * g;
                state_sum[i] = static_cast<scalar_t>(sum);
                param[i] = static_cast<scalar_t>(
                    p - clr * g / (static_cast<math_t>(std::sqrt(
                        static_cast<double>(sum))) + static_cast<math_t>(eps)));
            }
        });
    }
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
                    const std::optional<Tensor>& found_inf) {
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
                              const std::optional<Tensor>& found_inf) {
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
                     const std::optional<Tensor>& found_inf) {
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
                               const std::optional<Tensor>& found_inf) {
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
}

} // namespace cpu
} // namespace tensorplay
