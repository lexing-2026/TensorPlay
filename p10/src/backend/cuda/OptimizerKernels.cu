#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "ForeachKernels.h"
#include "ForeachMultiTensor.cuh"
#include "OptimizerMTA.cuh"
#include "Exception.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

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
    if (count > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        TP_THROW(ValueError, std::string(op_name) +
            ": too many tensors for one CUDA grid");
    }

    const DType dtype = count ? params[0].dtype() : DType::Undefined;
    const Device device = count ? params[0].device() : Device(DeviceType::CUDA);
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
                ": requires contiguous same-device parameter/gradient pairs with one dtype");
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
                state.shape() != param.shape() || state.dtype() != dtype ||
                state.device() != device) {
                TP_THROW(NotImplementedError, std::string(op_name) +
                    ": AMSGrad state must match its parameter layout");
            }
        }
    }
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
        if (param.is_sparse() || grad.is_sparse() || isComplexType(param.dtype()) ||
            !param.is_contiguous() || !grad.is_contiguous() ||
            param.shape() != grad.shape() || param.dtype() != grad.dtype() ||
            param.dtype() != dtype || !param.device().is_cuda() ||
            !grad.device().is_cuda()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": requires contiguous CUDA tensors with matching floating dtype and shape");
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
            step.dtype() != DType::Float32 ||
            !step.device().is_cuda()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": state_steps must be singleton CUDA float32 tensors");
        }
    }
}

// Non-capturable optimizer state keeps the scalar step counters on CPU, just
// incrementing the counters here avoids a separate device foreach launch and
// leaves graph-capture/device-step cases on their existing fallback path.
void validate_host_steps(const std::vector<Tensor>& params,
                         const std::vector<Tensor>& state_steps,
                         const char* op_name) {
    if (state_steps.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": state_steps must match parameter list");
    }
    for (const Tensor& step : state_steps) {
        if (!step.defined() || !step.is_contiguous() || step.numel() != 1 ||
            step.device() != Device(DeviceType::CPU) ||
            (step.dtype() != DType::Float32 &&
             step.dtype() != DType::Float64)) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": native CUDA path requires singleton CPU float32/float64 state_steps");
        }
    }
}

double increment_host_step(const Tensor& step, const char* op_name) {
    Tensor& mutable_step = const_cast<Tensor&>(step);
    if (step.dtype() == DType::Float32) {
        float* value = mutable_step.data_ptr<float>();
        *value += 1.0f;
        return static_cast<double>(*value);
    }
    if (step.dtype() == DType::Float64) {
        double* value = mutable_step.data_ptr<double>();
        *value += 1.0;
        return *value;
    }
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": state_steps must be float32 or float64");
}

std::vector<int64_t> increment_host_step_indices(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& state_steps, const char* op_name) {
    validate_host_steps(params, state_steps, op_name);
    std::vector<int64_t> steps(state_steps.size());
    for (size_t i = 0; i < state_steps.size(); ++i) {
        // Keep the host-step update and the value consumed by bias correction
        // in one C++ pass.  The Python foreach increment followed by
        // Tensor.item() used to add one dispatch and one scalar extraction per
        // parameter tensor on the CUDA foreach path.
        steps[i] = static_cast<int64_t>(
            increment_host_step(state_steps[i], op_name));
    }
    return steps;
}

std::vector<double> increment_host_steps(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& state_steps, const char* op_name) {
    validate_host_steps(params, state_steps, op_name);
    std::vector<double> steps(state_steps.size());
    for (size_t i = 0; i < state_steps.size(); ++i) {
        steps[i] = increment_host_step(state_steps[i], op_name);
    }
    return steps;
}

void increment_host_steps_inplace(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& state_steps, const char* op_name) {
    validate_host_steps(params, state_steps, op_name);
    for (const Tensor& step : state_steps) {
        (void)increment_host_step(step, op_name);
    }
}

void validate_cuda_scalar_states(const std::vector<Tensor>& params,
                                 const std::vector<Tensor>& states,
                                 const char* name, const char* op_name) {
    if (states.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) + ": " + name +
            " list must match parameter list");
    }
    for (const Tensor& state : states) {
        if (!state.defined() || !state.is_contiguous() || state.numel() != 1 ||
            !state.device().is_cuda() || state.dtype() != DType::Float32) {
            TP_THROW(NotImplementedError, std::string(op_name) + ": " + name +
                " must be singleton CUDA float32 tensors");
        }
    }
}

void validate_host_scalar_states(const std::vector<Tensor>& params,
                                 const std::vector<Tensor>& states,
                                 const char* name, const char* op_name) {
    if (states.size() != params.size()) {
        TP_THROW(ValueError, std::string(op_name) + ": " + name +
            " list must match parameter list");
    }
    for (const Tensor& state : states) {
        if (!state.defined() || !state.is_contiguous() || state.numel() != 1 ||
            state.device() != Device(DeviceType::CPU) ||
            (state.dtype() != DType::Float32 &&
             state.dtype() != DType::Float64)) {
            TP_THROW(NotImplementedError, std::string(op_name) + ": " + name +
                " must be singleton CPU float32/float64 tensors");
        }
    }
}

double read_host_scalar(const Tensor& state, const char* op_name) {
    if (state.dtype() == DType::Float32) return *state.data_ptr<float>();
    if (state.dtype() == DType::Float64) return *state.data_ptr<double>();
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": scalar optimizer state must be float32 or float64");
}

void write_host_scalar(const Tensor& state, double value, const char* op_name) {
    Tensor& mutable_state = const_cast<Tensor&>(state);
    if (state.dtype() == DType::Float32) {
        *mutable_state.data_ptr<float>() = static_cast<float>(value);
        return;
    }
    if (state.dtype() == DType::Float64) {
        *mutable_state.data_ptr<double>() = value;
        return;
    }
    TP_THROW(NotImplementedError, std::string(op_name) +
        ": scalar optimizer state must be float32 or float64");
}

const float* optional_fused_float_ptr(const std::optional<Tensor>& value,
                                      const char* name) {
    if (!value.has_value() || !value->defined()) return nullptr;
    if (value->numel() != 1 || value->dtype() != DType::Float32 ||
        !value->device().is_cuda()) {
        TP_THROW(NotImplementedError, std::string(name) +
            " must be a singleton CUDA float32 tensor");
    }
    return value->data_ptr<float>();
}


template <typename F>
void dispatch_fused_cuda_dtype(const std::vector<Tensor>& params,
                               const char* op_name,
                               F&& fn) {
    if (params.empty()) return;
    if (!foreach_mta::dispatch_dtype(params[0].dtype(), fn)) {
        TP_THROW(NotImplementedError, std::string(op_name) +
            ": fused optimizer supports float16, bfloat16, float32, and float64");
    }
}

bool uses_foreach_exact_lowp(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16;
}

template <typename F>
void dispatch_fused_cuda_lr(const Tensor* lr, F&& fn) {
    if (!lr) {
        fn.template operator()<double>();
    } else if (lr->dtype() == DType::Float32) {
        fn.template operator()<float>();
    } else if (lr->dtype() == DType::Float64) {
        fn.template operator()<double>();
    } else {
        TP_THROW(NotImplementedError, "fused optimizer Tensor lr must be float32 or float64");
    }
}


void fused_sgd_cuda_impl(std::vector<Tensor> params,
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
                         const std::optional<Tensor>& found_inf,
                         const Tensor* tensor_lr) {
    validate_fused_pairs(params, grads, "_fused_sgd_");
    if (params.empty()) return;
    if (momentum == 0.0) {
        if (!momentum_buffers.empty()) {
            TP_THROW(ValueError, "_fused_sgd_: momentum buffer list must be empty when momentum is zero");
        }
    } else {
        validate_fused_state(params, momentum_buffers, true, "_fused_sgd_");
    }
    const float* scale_ptr = optional_fused_float_ptr(grad_scale, "grad_scale");
    const float* found_ptr = optional_fused_float_ptr(found_inf, "found_inf");
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, "_fused_sgd_", [&]<typename scalar_t, typename math_t>() {
            if (momentum == 0.0) {
                optimizer_mta::launch_sgd<scalar_t, math_t, lr_t, false>(
                    params, grads, momentum_buffers, lr, momentum, dampening,
                    weight_decay, nesterov, is_first_step, maximize,
                    scale_ptr, found_ptr, tensor_lr, "_fused_sgd_");
            } else {
                optimizer_mta::launch_sgd<scalar_t, math_t, lr_t, true>(
                    params, grads, momentum_buffers, lr, momentum, dampening,
                    weight_decay, nesterov, is_first_step, maximize,
                    scale_ptr, found_ptr, tensor_lr, "_fused_sgd_");
            }
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cuda_impl(std::vector<Tensor> params,
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
                          const std::optional<Tensor>& found_inf,
                          const Tensor* tensor_lr,
                          bool exact) {
    const char* op_name = adamw ? "_fused_adamw_" : "_fused_adam_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_avg_sqs, true, op_name);
    validate_fused_state(params, max_exp_avg_sqs, amsgrad, op_name);
    if (!amsgrad && !max_exp_avg_sqs.empty()) {
        TP_THROW(ValueError, std::string(op_name) +
            ": max_exp_avg_sqs must be empty when amsgrad is false");
    }
    const float* scale_ptr = optional_fused_float_ptr(grad_scale, "grad_scale");
    const float* found_ptr = optional_fused_float_ptr(found_inf, "found_inf");
    // CPU.  Update them and materialize the host snapshot in one pass, then
    // use the same native MTA kernel as the device-step fused path.  AMP
    // metadata is intentionally excluded here because the host-step kernel
    // has no device-side scale/found-inf arguments; callers with AMP continue
    // through the existing route.
    const bool host_steps = !state_steps.empty() &&
        state_steps[0].device() == Device(DeviceType::CPU);
    if (host_steps) {
        validate_host_steps(params, state_steps, op_name);
        if (grad_scale.has_value() || found_inf.has_value()) {
            TP_THROW(NotImplementedError,
                std::string(op_name) +
                ": host state-step CUDA path does not support AMP metadata");
        }
        const std::vector<int64_t> steps = increment_host_step_indices(
            params, state_steps, op_name);
        dispatch_fused_cuda_dtype(params, op_name,
            [&]<typename scalar_t, typename math_t>() {
                // Standard foreach Adam on Half/BFloat16 rounds at every
                // intermediate foreach boundary.  Keep the explicit fused
                // kernel's all-opmath behavior for its public API, while the
                // private exact route used by _multi_tensor_adam_ gets the
                const bool use_exact = exact &&
                    (params[0].dtype() == DType::Float16 ||
                     params[0].dtype() == DType::BFloat16);
                if (use_exact) {
                    if (adamw) {
                        if (amsgrad) {
                            optimizer_mta::launch_adam_host_exact<
                                scalar_t, math_t, true, true>(
                                params, grads, exp_avgs, exp_avg_sqs,
                                max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                                weight_decay, maximize, op_name);
                        } else {
                            optimizer_mta::launch_adam_host_exact<
                                scalar_t, math_t, true, false>(
                                params, grads, exp_avgs, exp_avg_sqs,
                                max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                                weight_decay, maximize, op_name);
                        }
                    } else if (amsgrad) {
                        optimizer_mta::launch_adam_host_exact<
                            scalar_t, math_t, false, true>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    } else {
                        optimizer_mta::launch_adam_host_exact<
                            scalar_t, math_t, false, false>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    }
                    return;
                }
                if (adamw) {
                    if (amsgrad) {
                        optimizer_mta::launch_adam_host<
                            scalar_t, math_t, double, true, true>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    } else {
                        optimizer_mta::launch_adam_host<
                            scalar_t, math_t, double, true, false>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    }
                } else {
                    if (amsgrad) {
                        optimizer_mta::launch_adam_host<
                            scalar_t, math_t, double, false, true>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    } else {
                        optimizer_mta::launch_adam_host<
                            scalar_t, math_t, double, false, false>(
                            params, grads, exp_avgs, exp_avg_sqs,
                            max_exp_avg_sqs, steps, lr, beta1, beta2, eps,
                            weight_decay, maximize, op_name);
                    }
                }
            });
        for (const Tensor& param : params) {
            param.unsafeGetTensorImpl()->bump_version();
        }
        return;
    }
    validate_fused_steps(params, state_steps, op_name);
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, op_name, [&]<typename scalar_t, typename math_t>() {
            if (adamw) {
                if (amsgrad) {
                    optimizer_mta::launch_adam_fused<
                        scalar_t, math_t, lr_t, true, true>(
                        params, grads, exp_avgs, exp_avg_sqs,
                        max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                        eps, weight_decay, maximize, scale_ptr, found_ptr,
                        tensor_lr, op_name);
                } else {
                    optimizer_mta::launch_adam_fused<
                        scalar_t, math_t, lr_t, true, false>(
                        params, grads, exp_avgs, exp_avg_sqs,
                        max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                        eps, weight_decay, maximize, scale_ptr, found_ptr,
                        tensor_lr, op_name);
                }
            } else if (amsgrad) {
                optimizer_mta::launch_adam_fused<
                    scalar_t, math_t, lr_t, false, true>(
                    params, grads, exp_avgs, exp_avg_sqs,
                    max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                    eps, weight_decay, maximize, scale_ptr, found_ptr,
                    tensor_lr, op_name);
            } else {
                optimizer_mta::launch_adam_fused<
                    scalar_t, math_t, lr_t, false, false>(
                    params, grads, exp_avgs, exp_avg_sqs,
                    max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                    eps, weight_decay, maximize, scale_ptr, found_ptr,
                    tensor_lr, op_name);
            }
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adagrad_cuda_impl(std::vector<Tensor> params,
                             const std::vector<Tensor>& grads,
                             const std::vector<Tensor>& state_sums,
                             const std::vector<Tensor>& state_steps,
                             double lr,
                             double lr_decay,
                             double weight_decay,
                             double eps,
                             bool maximize,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf,
                             const Tensor* tensor_lr) {
    validate_fused_pairs(params, grads, "_fused_adagrad_");
    if (params.empty()) return;
    validate_fused_state(params, state_sums, true, "_fused_adagrad_");
    const bool host_steps = !state_steps.empty() &&
        state_steps[0].device() == Device(DeviceType::CPU);
    for (const Tensor& step : state_steps) {
        if ((step.device() == Device(DeviceType::CPU)) != host_steps) {
            TP_THROW(NotImplementedError,
                "_fused_adagrad_: state_steps must be all CPU or all CUDA");
        }
    }
    const float* scale_ptr = optional_fused_float_ptr(grad_scale, "grad_scale");
    const float* found_ptr = optional_fused_float_ptr(found_inf, "found_inf");
    if (host_steps) {
        if (grad_scale.has_value() || found_inf.has_value()) {
            TP_THROW(NotImplementedError,
                "_fused_adagrad_: host state-step CUDA path does not support AMP metadata");
        }
        const std::vector<double> steps = increment_host_steps(
            params, state_steps, "_fused_adagrad_");
        const double scalar_lr = tensor_lr == nullptr
            ? lr : tensor_lr->item().toDouble();
        const bool exact = uses_foreach_exact_lowp(params[0].dtype());
        dispatch_fused_cuda_dtype(params, "_fused_adagrad_",
            [&]<typename scalar_t, typename math_t>() {
                if (exact) {
                    std::vector<double> corrected_lrs(steps.size());
                    for (size_t i = 0; i < steps.size(); ++i) {
                        // The foreach path materializes minus_clr before
                        // addcdiv_; keep the negative scalar in metadata so
                        corrected_lrs[i] = -scalar_lr /
                            (1.0 + (steps[i] - 1.0) * lr_decay);
                    }
                    optimizer_mta::launch_adagrad_host_exact<scalar_t, math_t>(
                        params, grads, state_sums, corrected_lrs, eps,
                        weight_decay, maximize,
                        "_fused_adagrad_");
                } else {
                    optimizer_mta::launch_adagrad_host<scalar_t, math_t>(
                        params, grads, state_sums, steps, scalar_lr, lr_decay,
                        weight_decay, eps, maximize, "_fused_adagrad_");
                }
            });
        for (const Tensor& param : params) {
            param.unsafeGetTensorImpl()->bump_version();
        }
        return;
    }
    validate_fused_steps(params, state_steps, "_fused_adagrad_");
    dispatch_fused_cuda_lr(tensor_lr, [&]<typename lr_t>() {
        dispatch_fused_cuda_dtype(params, "_fused_adagrad_", [&]<typename scalar_t, typename math_t>() {
            optimizer_mta::launch_adagrad_fused<scalar_t, math_t, lr_t>(
                params, grads, state_sums, state_steps, lr, lr_decay,
                weight_decay, eps, maximize, scale_ptr, found_ptr,
                tensor_lr, "_fused_adagrad_");
        });
    });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_rmsprop_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> square_avgs, std::vector<Tensor> grad_avgs,
        std::vector<Tensor> momentum_buffers, std::vector<Tensor> state_steps,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool centered, bool maximize) {
    const char* op_name = "_fused_rmsprop_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, square_avgs, true, op_name);
    validate_fused_state(params, grad_avgs, centered, op_name);
    validate_fused_state(params, momentum_buffers, momentum != 0.0, op_name);
    increment_host_steps_inplace(params, state_steps, op_name);
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_rmsprop_exact<scalar_t, math_t>(
                    params, grads, square_avgs, grad_avgs, momentum_buffers,
                    lr, alpha, eps, weight_decay, momentum, centered,
                    maximize, op_name);
            } else {
                optimizer_mta::launch_rmsprop<scalar_t, math_t>(
                    params, grads, square_avgs, grad_avgs, momentum_buffers,
                    lr, alpha, eps, weight_decay, momentum, centered,
                    maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adadelta_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> square_avgs, std::vector<Tensor> acc_deltas,
        std::vector<Tensor> state_steps, double lr, double rho, double eps,
        double weight_decay, bool maximize) {
    const char* op_name = "_fused_adadelta_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, square_avgs, true, op_name);
    validate_fused_state(params, acc_deltas, true, op_name);
    increment_host_steps_inplace(params, state_steps, op_name);
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_adadelta_exact<scalar_t, math_t>(
                    params, grads, square_avgs, acc_deltas, lr, rho, eps,
                    weight_decay, maximize, op_name);
            } else {
                optimizer_mta::launch_adadelta<scalar_t, math_t>(
                    params, grads, square_avgs, acc_deltas, lr, rho, eps,
                    weight_decay, maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adamax_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> exp_avgs, std::vector<Tensor> exp_infs,
        std::vector<Tensor> state_steps, double lr, double beta1,
        double beta2, double eps, double weight_decay, bool maximize) {
    const char* op_name = "_fused_adamax_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, exp_avgs, true, op_name);
    validate_fused_state(params, exp_infs, true, op_name);
    const std::vector<double> steps = increment_host_steps(
        params, state_steps, op_name);
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_adamax_exact<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_infs, steps, lr, beta1,
                    beta2, eps, weight_decay, maximize, op_name);
            } else {
                optimizer_mta::launch_adamax<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_infs, steps, lr, beta1,
                    beta2, eps, weight_decay, maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_asgd_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> axs, std::vector<Tensor> mus,
        std::vector<Tensor> etas, std::vector<Tensor> state_steps,
        double lr, double lambd, double t0, double alpha,
        double weight_decay, bool maximize) {
    const char* op_name = "_fused_asgd_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, axs, true, op_name);
    if (params[0].dtype() != DType::Float32) {
        TP_THROW(NotImplementedError,
            "_fused_asgd_: native CUDA path currently requires float32 parameters");
    }
    validate_cuda_scalar_states(params, mus, "mus", op_name);
    validate_cuda_scalar_states(params, etas, "etas", op_name);
    const bool device_steps = !state_steps.empty() &&
        state_steps[0].device().is_cuda();
    for (const Tensor& step : state_steps) {
        if ((step.device().is_cuda()) != device_steps) {
            TP_THROW(NotImplementedError,
                "_fused_asgd_: state_steps must be all CPU or all CUDA");
        }
    }
    std::vector<double> steps;
    if (device_steps) {
        validate_fused_steps(params, state_steps, op_name);
    } else {
        steps = increment_host_steps(params, state_steps, op_name);
    }
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            optimizer_mta::launch_asgd<scalar_t, math_t>(
                params, grads, axs, mus, etas,
                device_steps ? nullptr : &steps,
                device_steps ? &state_steps : nullptr,
                lr, lambd, t0, alpha, weight_decay, maximize, op_name);
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_rprop_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> prevs, std::vector<Tensor> step_sizes,
        std::vector<Tensor> state_steps, double step_size_min,
        double step_size_max, double etaminus, double etaplus,
        bool maximize) {
    const char* op_name = "_fused_rprop_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, prevs, true, op_name);
    validate_fused_state(params, step_sizes, true, op_name);
    (void)increment_host_steps(params, state_steps, op_name);
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_rprop_exact<scalar_t, math_t>(
                    params, grads, prevs, step_sizes, step_size_min,
                    step_size_max, etaminus, etaplus, maximize, op_name);
            } else {
                optimizer_mta::launch_rprop<scalar_t, math_t>(
                    params, grads, prevs, step_sizes, step_size_min,
                    step_size_max, etaminus, etaplus, maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_nadam_cuda(
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
    validate_host_scalar_states(params, mu_products, "mu_products", op_name);
    const std::vector<double> steps = increment_host_steps(
        params, state_steps, op_name);
    std::vector<double> next_mu_products(mu_products.size());
    for (size_t i = 0; i < mu_products.size(); ++i) {
        const double mu = beta1 * (1.0 - 0.5 * std::pow(
            0.96, steps[i] * momentum_decay));
        next_mu_products[i] = read_host_scalar(mu_products[i], op_name) * mu;
        write_host_scalar(mu_products[i], next_mu_products[i], op_name);
        // foreach launch observes it.  Feed the rounded value to the kernel,
        // not the pre-store double temporary.
        next_mu_products[i] = read_host_scalar(mu_products[i], op_name);
    }
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_nadam_exact<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_avg_sqs, steps,
                    next_mu_products, lr, beta1, beta2, eps, momentum_decay,
                    weight_decay, decoupled_weight_decay, maximize, op_name);
            } else {
                optimizer_mta::launch_nadam<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_avg_sqs, steps,
                    next_mu_products, lr, beta1, beta2, eps, momentum_decay,
                    weight_decay, decoupled_weight_decay, maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_radam_cuda(
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
    const std::vector<double> steps = increment_host_steps(
        params, state_steps, op_name);
    const bool exact = uses_foreach_exact_lowp(params[0].dtype());
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            if (exact) {
                optimizer_mta::launch_radam_exact<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_avg_sqs, steps, lr, beta1,
                    beta2, eps, weight_decay, decoupled_weight_decay,
                    maximize, op_name);
            } else {
                optimizer_mta::launch_radam<scalar_t, math_t>(
                    params, grads, exp_avgs, exp_avg_sqs, steps, lr, beta1,
                    beta2, eps, weight_decay, decoupled_weight_decay,
                    maximize, op_name);
            }
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void validate_adafactor_factored_state(
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
        const Tensor& row = row_vars[i];
        const Tensor& col = col_vars[i];
        if (p.dim() != 2 || !row.defined() || !col.defined() ||
            !row.is_contiguous() || !col.is_contiguous() ||
            row.shape() != Size({p.size(0), 1}) ||
            col.shape() != Size({1, p.size(1)}) ||
            row.dtype() != p.dtype() || col.dtype() != p.dtype() ||
            row.device() != p.device() || col.device() != p.device()) {
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": factored states require contiguous 2-D matching tensors");
        }
    }
}

void fused_adafactor_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> variances, std::vector<Tensor> state_steps,
        double lr, double beta2_decay, double eps1, double eps2, double d,
        double weight_decay, bool maximize) {
    const char* op_name = "_fused_adafactor_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_fused_state(params, variances, true, op_name);
    const std::vector<double> steps = increment_host_steps(
        params, state_steps, op_name);
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            optimizer_mta::launch_adafactor_vector<scalar_t, math_t>(
                params, grads, variances, steps, lr, beta2_decay, eps1, eps2,
                d, weight_decay, maximize, op_name);
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adafactor_factored_cuda(
        std::vector<Tensor> params, const std::vector<Tensor>& grads,
        std::vector<Tensor> row_vars, std::vector<Tensor> col_vars,
        std::vector<Tensor> state_steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize) {
    const char* op_name = "_fused_adafactor_factored_";
    validate_fused_pairs(params, grads, op_name);
    if (params.empty()) return;
    validate_adafactor_factored_state(params, row_vars, col_vars, op_name);
    const std::vector<double> steps = increment_host_steps(
        params, state_steps, op_name);
    dispatch_fused_cuda_dtype(params, op_name,
        [&]<typename scalar_t, typename math_t>() {
            optimizer_mta::launch_adafactor_factored<scalar_t, math_t>(
                params, grads, row_vars, col_vars, steps, lr, beta2_decay,
                eps1, eps2, d, weight_decay, maximize, op_name);
        });
    for (const Tensor& param : params) param.unsafeGetTensorImpl()->bump_version();
}

void fused_adam_cuda(std::vector<Tensor> params,
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
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, false, grad_scale, found_inf, nullptr, exact);
}

void fused_adam_tensor_lr_cuda(std::vector<Tensor> params,
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
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, 0.0, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, false,
        grad_scale, found_inf, &lr, exact);
}

void fused_adamw_cuda(std::vector<Tensor> params,
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
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, lr, beta1,
        beta2, weight_decay, eps, amsgrad,
        maximize, true, grad_scale, found_inf, nullptr, exact);
}

void fused_adamw_tensor_lr_cuda(std::vector<Tensor> params,
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
    fused_adam_cuda_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
        max_exp_avg_sqs, state_steps, 0.0, beta1, beta2,
        weight_decay, eps, amsgrad, maximize, true,
        grad_scale, found_inf, &lr, exact);
}

void fused_sgd_cuda(std::vector<Tensor> params,
                    std::vector<Tensor> grads,
                    std::vector<Tensor> momentum_buffers,
                    double weight_decay, double momentum, double lr,
                    double dampening, bool nesterov, bool maximize,
                    bool is_first_step, const std::optional<Tensor>& grad_scale,
                    const std::optional<Tensor>& found_inf) {
    fused_sgd_cuda_impl(std::move(params), grads, momentum_buffers,
        lr, momentum, dampening,
        weight_decay, nesterov, maximize, is_first_step,
        grad_scale, found_inf, nullptr);
}

void fused_sgd_tensor_lr_cuda(std::vector<Tensor> params,
                              std::vector<Tensor> grads,
                              std::vector<Tensor> momentum_buffers,
                              double weight_decay, double momentum,
                              const Tensor& lr, double dampening, bool nesterov,
                              bool maximize, bool is_first_step,
                              const std::optional<Tensor>& grad_scale,
                              const std::optional<Tensor>& found_inf) {
    fused_sgd_cuda_impl(std::move(params), grads, momentum_buffers, 0.0,
        momentum, dampening, weight_decay,
        nesterov, maximize, is_first_step, grad_scale, found_inf, &lr);
}

void fused_adagrad_cuda(std::vector<Tensor> params,
                        std::vector<Tensor> grads,
                        std::vector<Tensor> state_sums,
                        std::vector<Tensor> state_steps,
                        double lr, double lr_decay, double weight_decay,
                        double eps, bool maximize,
                        const std::optional<Tensor>& grad_scale,
                        const std::optional<Tensor>& found_inf) {
    fused_adagrad_cuda_impl(std::move(params), grads, state_sums, state_steps,
        lr, lr_decay, weight_decay,
        eps, maximize, grad_scale, found_inf, nullptr);
}

void fused_adagrad_tensor_lr_cuda(std::vector<Tensor> params,
                                  std::vector<Tensor> grads,
                                  std::vector<Tensor> state_sums,
                                  std::vector<Tensor> state_steps,
                                  const Tensor& lr, double lr_decay,
                                  double weight_decay, double eps, bool maximize,
                                  const std::optional<Tensor>& grad_scale,
                                  const std::optional<Tensor>& found_inf) {
    fused_adagrad_cuda_impl(std::move(params), grads, state_sums, state_steps,
        0.0, lr_decay, weight_decay, eps,
        maximize, grad_scale, found_inf, &lr);
}

} // namespace

std::vector<Tensor> foreach_sgd_cuda(const std::vector<Tensor>& params,
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
    const bool launched = foreach_mta::dispatch_dtype(
        params[0].dtype(), [&]<typename scalar_t, typename math_t>() {
            if (momentum == 0.0) {
                optimizer_mta::launch_sgd<scalar_t, math_t, double, false>(
                    params, grads, momentum_buffers, lr, momentum, dampening,
                    weight_decay, nesterov, first_momentum_step, false,
                    nullptr, nullptr, nullptr, "_foreach_sgd");
            } else {
                optimizer_mta::launch_sgd<scalar_t, math_t, double, true>(
                    params, grads, momentum_buffers, lr, momentum, dampening,
                    weight_decay, nesterov, first_momentum_step, false,
                    nullptr, nullptr, nullptr, "_foreach_sgd");
            }
        });
    if (!launched) {
        TP_THROW(NotImplementedError,
                 "_foreach_sgd supports floating CUDA tensors");
    }
    // changes immediately after the queued update, even though the CUDA
    // kernel itself executes asynchronously.
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}

std::vector<Tensor> foreach_adam_cuda(const std::vector<Tensor>& params,
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
    const bool launched = foreach_mta::dispatch_dtype(
        params[0].dtype(), [&]<typename scalar_t, typename math_t>() {
            if (amsgrad) {
                optimizer_mta::launch_adam_host<
                    scalar_t, math_t, double, false, true>(
                    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                    steps, lr, beta1, beta2, eps, weight_decay,
                    false, "_foreach_adam");
            } else {
                optimizer_mta::launch_adam_host<
                    scalar_t, math_t, double, false, false>(
                    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                    steps, lr, beta1, beta2, eps, weight_decay,
                    false, "_foreach_adam");
            }
        });
    if (!launched) {
        TP_THROW(NotImplementedError,
                 "_foreach_adam supports floating CUDA tensors");
    }
    for (const auto& param : params) {
        param.unsafeGetTensorImpl()->bump_version();
    }
    return params;
}






TENSORPLAY_LIBRARY_IMPL(CUDA, OptimizerKernels) {
    m.impl("_foreach_sgd", foreach_sgd_cuda);
    m.impl("_foreach_adam", foreach_adam_cuda);
    m.impl("_fused_adam_", fused_adam_cuda);
    m.impl("_fused_adam_.tensor_lr", fused_adam_tensor_lr_cuda);
    m.impl("_fused_adamw_", fused_adamw_cuda);
    m.impl("_fused_adamw_.tensor_lr", fused_adamw_tensor_lr_cuda);
    m.impl("_fused_sgd_", fused_sgd_cuda);
    m.impl("_fused_sgd_.tensor_lr", fused_sgd_tensor_lr_cuda);
    m.impl("_fused_adagrad_", fused_adagrad_cuda);
    m.impl("_fused_adagrad_.tensor_lr", fused_adagrad_tensor_lr_cuda);
    m.impl("_fused_rmsprop_", fused_rmsprop_cuda);
    m.impl("_fused_adadelta_", fused_adadelta_cuda);
    m.impl("_fused_adamax_", fused_adamax_cuda);
    m.impl("_fused_asgd_", fused_asgd_cuda);
    m.impl("_fused_rprop_", fused_rprop_cuda);
    m.impl("_fused_nadam_", fused_nadam_cuda);
    m.impl("_fused_radam_", fused_radam_cuda);
    m.impl("_fused_adafactor_", fused_adafactor_cuda);
    m.impl("_fused_adafactor_factored_", fused_adafactor_factored_cuda);
    m.impl("_foreach_add.Scalar", foreach_add_scalar_mta_ret_cuda);
    m.impl("_foreach_add.List", foreach_add_list_mta_ret_cuda);
    m.impl("_foreach_add.ScalarList", foreach_add_scalar_list_mta_ret_cuda);
    m.impl("_foreach_add.Tensor", foreach_add_tensor_cuda);
    m.impl("_foreach_add_.Scalar", foreach_add_scalar_mta_inplace_cuda);
    m.impl("_foreach_add_.List", foreach_add_list_mta_inplace_cuda);
    m.impl("_foreach_add_.ScalarList", foreach_add_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_add_.Tensor", foreach_add_tensor_inplace_cuda);

#define REGISTER_FOREACH_BINARY(NAME) \
    m.impl("_foreach_" #NAME ".Scalar", foreach_##NAME##_scalar_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".List", foreach_##NAME##_list_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".ScalarList", foreach_##NAME##_scalar_list_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".Tensor", foreach_##NAME##_tensor_cuda); \
    m.impl("_foreach_" #NAME "_.Scalar", foreach_##NAME##_scalar_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.List", foreach_##NAME##_list_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.ScalarList", foreach_##NAME##_scalar_list_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.Tensor", foreach_##NAME##_tensor_inplace_cuda);

#define REGISTER_FOREACH_BINARY_ALPHA(NAME) \
    m.impl("_foreach_" #NAME ".Scalar", foreach_##NAME##_scalar_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".List", foreach_##NAME##_list_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".ScalarList", foreach_##NAME##_scalar_list_mta_ret_cuda); \
    m.impl("_foreach_" #NAME ".Tensor", foreach_##NAME##_tensor_cuda); \
    m.impl("_foreach_" #NAME "_.Scalar", foreach_##NAME##_scalar_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.List", foreach_##NAME##_list_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.ScalarList", foreach_##NAME##_scalar_list_mta_inplace_cuda); \
    m.impl("_foreach_" #NAME "_.Tensor", foreach_##NAME##_tensor_inplace_cuda);
    REGISTER_FOREACH_BINARY_ALPHA(sub)
    REGISTER_FOREACH_BINARY(mul)
    REGISTER_FOREACH_BINARY(div)
#undef REGISTER_FOREACH_BINARY_ALPHA
#undef REGISTER_FOREACH_BINARY

#define REGISTER_FOREACH_UNARY(NAME) \
    m.impl("_foreach_" #NAME, foreach_##NAME##_cuda); \
    m.impl("_foreach_" #NAME "_", foreach_##NAME##_inplace_cuda);
    m.impl("_foreach_sqrt", foreach_sqrt_mta_ret_cuda); \
    m.impl("_foreach_sqrt_", foreach_sqrt_mta_inplace_cuda); \
    m.impl("_foreach_sqrt.out", foreach_sqrt_out_cuda);
    m.impl("_foreach_rsqrt", foreach_rsqrt_mta_ret_cuda); \
    m.impl("_foreach_rsqrt_", foreach_rsqrt_mta_inplace_cuda); \
    m.impl("_foreach_rsqrt.out", foreach_rsqrt_out_cuda);
    m.impl("_foreach_neg", foreach_neg_mta_ret_cuda); \
    m.impl("_foreach_neg_", foreach_neg_mta_inplace_cuda); \
    m.impl("_foreach_neg.out", foreach_neg_out_cuda);
    m.impl("_foreach_abs", foreach_abs_mta_ret_cuda); \
    m.impl("_foreach_abs_", foreach_abs_mta_inplace_cuda); \
    m.impl("_foreach_abs.out", foreach_abs_out_cuda);
    m.impl("_foreach_reciprocal", foreach_reciprocal_mta_ret_cuda); \
    m.impl("_foreach_reciprocal_", foreach_reciprocal_mta_inplace_cuda); \
    m.impl("_foreach_reciprocal.out", foreach_reciprocal_out_cuda);
    m.impl("_foreach_sign", foreach_sign_mta_ret_cuda); \
    m.impl("_foreach_sign_", foreach_sign_mta_inplace_cuda); \
    m.impl("_foreach_sign.out", foreach_sign_out_cuda);
#undef REGISTER_FOREACH_UNARY

#define REGISTER_FOREACH_UNARY_OUT(NAME) \
    m.impl("_foreach_" #NAME ".out", foreach_##NAME##_out_cuda);
    REGISTER_FOREACH_UNARY_OUT(sqrt)
    REGISTER_FOREACH_UNARY_OUT(rsqrt)
    REGISTER_FOREACH_UNARY_OUT(neg)
    REGISTER_FOREACH_UNARY_OUT(abs)
    REGISTER_FOREACH_UNARY_OUT(sign)
    REGISTER_FOREACH_UNARY_OUT(reciprocal)
    REGISTER_FOREACH_UNARY_OUT(acos)
    REGISTER_FOREACH_UNARY_OUT(asin)
    REGISTER_FOREACH_UNARY_OUT(atan)
    REGISTER_FOREACH_UNARY_OUT(ceil)
    REGISTER_FOREACH_UNARY_OUT(cos)
    REGISTER_FOREACH_UNARY_OUT(cosh)
    REGISTER_FOREACH_UNARY_OUT(erf)
    REGISTER_FOREACH_UNARY_OUT(erfc)
    REGISTER_FOREACH_UNARY_OUT(exp)
    REGISTER_FOREACH_UNARY_OUT(expm1)
    REGISTER_FOREACH_UNARY_OUT(floor)
    REGISTER_FOREACH_UNARY_OUT(frac)
    REGISTER_FOREACH_UNARY_OUT(lgamma)
    REGISTER_FOREACH_UNARY_OUT(log)
    REGISTER_FOREACH_UNARY_OUT(log10)
    REGISTER_FOREACH_UNARY_OUT(log1p)
    REGISTER_FOREACH_UNARY_OUT(log2)
    REGISTER_FOREACH_UNARY_OUT(round)
    REGISTER_FOREACH_UNARY_OUT(sigmoid)
    REGISTER_FOREACH_UNARY_OUT(sin)
    REGISTER_FOREACH_UNARY_OUT(sinh)
    REGISTER_FOREACH_UNARY_OUT(tan)
    REGISTER_FOREACH_UNARY_OUT(tanh)
    REGISTER_FOREACH_UNARY_OUT(trunc)
#undef REGISTER_FOREACH_UNARY_OUT

#define REGISTER_FOREACH_EXTRA_UNARY(NAME) \
    m.impl("_foreach_" #NAME, foreach_##NAME##_mta_ret_cuda); \
    m.impl("_foreach_" #NAME "_", foreach_##NAME##_mta_inplace_cuda);
    REGISTER_FOREACH_EXTRA_UNARY(acos)
    REGISTER_FOREACH_EXTRA_UNARY(asin)
    REGISTER_FOREACH_EXTRA_UNARY(atan)
    REGISTER_FOREACH_EXTRA_UNARY(ceil)
    REGISTER_FOREACH_EXTRA_UNARY(cos)
    REGISTER_FOREACH_EXTRA_UNARY(cosh)
    REGISTER_FOREACH_EXTRA_UNARY(erf)
    REGISTER_FOREACH_EXTRA_UNARY(erfc)
    REGISTER_FOREACH_EXTRA_UNARY(exp)
    REGISTER_FOREACH_EXTRA_UNARY(expm1)
    REGISTER_FOREACH_EXTRA_UNARY(floor)
    REGISTER_FOREACH_EXTRA_UNARY(frac)
    REGISTER_FOREACH_EXTRA_UNARY(lgamma)
    REGISTER_FOREACH_EXTRA_UNARY(log)
    REGISTER_FOREACH_EXTRA_UNARY(log10)
    REGISTER_FOREACH_EXTRA_UNARY(log1p)
    REGISTER_FOREACH_EXTRA_UNARY(log2)
    REGISTER_FOREACH_EXTRA_UNARY(round)
    REGISTER_FOREACH_EXTRA_UNARY(sigmoid)
    REGISTER_FOREACH_EXTRA_UNARY(sin)
    REGISTER_FOREACH_EXTRA_UNARY(sinh)
    REGISTER_FOREACH_EXTRA_UNARY(tan)
    REGISTER_FOREACH_EXTRA_UNARY(tanh)
    REGISTER_FOREACH_EXTRA_UNARY(trunc)
#undef REGISTER_FOREACH_EXTRA_UNARY

    m.impl("_foreach_max", foreach_max_cuda);
    m.impl("_foreach_max.out", foreach_max_out_cuda);
    m.impl("_foreach_zero", foreach_zero_mta_ret_cuda);
    m.impl("_foreach_zero.out", foreach_zero_out_cuda);
    m.impl("_foreach_clone", foreach_clone_cuda);
    m.impl("_foreach_clone.out", foreach_clone_out_cuda);
    m.impl("_foreach_copy", foreach_copy_cuda);
    m.impl("_foreach_copy.out", foreach_copy_out_cuda);
    m.impl("_foreach_mm", foreach_mm_cuda);
    m.impl("_foreach_norm.Scalar", foreach_norm_cuda);
    m.impl("_foreach_norm.Scalar_out", foreach_norm_out_cuda);
    m.impl("_foreach_powsum.Scalar", foreach_powsum_cuda);
    m.impl("_foreach_powsum.Scalar_out", foreach_powsum_out_cuda);

    m.impl("_foreach_add.List_out", foreach_add_list_out_cuda);
    m.impl("_foreach_add.ScalarList_out", foreach_add_scalar_list_out_cuda);
    m.impl("_foreach_add.Scalar_out", foreach_add_scalar_out_cuda);
    m.impl("_foreach_add.Tensor_out", foreach_add_tensor_out_cuda);
    m.impl("_foreach_sub_.Scalar", foreach_sub_scalar_mta_inplace_cuda);
    m.impl("_foreach_sub_.List", foreach_sub_list_mta_inplace_cuda);
    m.impl("_foreach_sub.List_out", foreach_sub_list_out_cuda);
    m.impl("_foreach_sub.ScalarList_out", foreach_sub_scalar_list_out_cuda);
    m.impl("_foreach_sub.Scalar_out", foreach_sub_scalar_out_cuda);
    m.impl("_foreach_sub.Tensor_out", foreach_sub_tensor_out_cuda);
    m.impl("_foreach_mul.List_out", foreach_mul_list_out_cuda);
    m.impl("_foreach_mul.ScalarList_out", foreach_mul_scalar_list_out_cuda);
    m.impl("_foreach_mul.Scalar_out", foreach_mul_scalar_out_cuda);
    m.impl("_foreach_mul.Tensor_out", foreach_mul_tensor_out_cuda);
    m.impl("_foreach_div.List_out", foreach_div_list_out_cuda);
    m.impl("_foreach_div.ScalarList_out", foreach_div_scalar_list_out_cuda);
    m.impl("_foreach_div.Scalar_out", foreach_div_scalar_out_cuda);
    m.impl("_foreach_div.Tensor_out", foreach_div_tensor_out_cuda);

    m.impl("_foreach_clamp_max.List_out", foreach_clamp_max_list_out_cuda);
    m.impl("_foreach_clamp_max.ScalarList_out", foreach_clamp_max_scalar_list_out_cuda);
    m.impl("_foreach_clamp_max.Scalar_out", foreach_clamp_max_scalar_out_cuda);
    m.impl("_foreach_clamp_min.List_out", foreach_clamp_min_list_out_cuda);
    m.impl("_foreach_clamp_min.ScalarList_out", foreach_clamp_min_scalar_list_out_cuda);
    m.impl("_foreach_clamp_min.Scalar_out", foreach_clamp_min_scalar_out_cuda);
    m.impl("_foreach_maximum.List_out", foreach_maximum_list_out_cuda);
    m.impl("_foreach_maximum.ScalarList_out", foreach_maximum_scalar_list_out_cuda);
    m.impl("_foreach_maximum.Scalar_out", foreach_maximum_scalar_out_cuda);
    m.impl("_foreach_minimum.List_out", foreach_minimum_list_out_cuda);
    m.impl("_foreach_minimum.ScalarList_out", foreach_minimum_scalar_list_out_cuda);
    m.impl("_foreach_minimum.Scalar_out", foreach_minimum_scalar_out_cuda);

    m.impl("_foreach_lerp.Scalar_out", foreach_lerp_scalar_out_cuda);
    m.impl("_foreach_lerp.List_out", foreach_lerp_list_out_cuda);
    m.impl("_foreach_lerp.ScalarList_out", foreach_lerp_scalar_list_out_cuda);

    m.impl("_foreach_pow.Scalar_out", foreach_pow_scalar_out_cuda);
    m.impl("_foreach_pow.List_out", foreach_pow_list_out_cuda);
    m.impl("_foreach_pow.ScalarList_out", foreach_pow_scalar_list_out_cuda);

    m.impl("_foreach_addcmul.Scalar_out", foreach_addcmul_scalar_out_cuda);
    m.impl("_foreach_addcmul.ScalarList_out", foreach_addcmul_scalar_list_out_cuda);
    m.impl("_foreach_addcmul.Tensor_out", foreach_addcmul_tensor_out_cuda);
    m.impl("_foreach_addcdiv.Scalar_out", foreach_addcdiv_scalar_out_cuda);
    m.impl("_foreach_addcdiv.ScalarList_out", foreach_addcdiv_scalar_list_out_cuda);
    m.impl("_foreach_addcdiv.Tensor_out", foreach_addcdiv_tensor_out_cuda);
#undef REGISTER_FOREACH_UNARY

    m.impl("_foreach_addcmul.Scalar", foreach_addcmul_scalar_cuda);
    m.impl("_foreach_addcmul_.Scalar", foreach_addcmul_scalar_mta_inplace_cuda);
    m.impl("_foreach_addcmul.ScalarList", foreach_addcmul_scalar_list_cuda);
    m.impl("_foreach_addcmul_.ScalarList", foreach_addcmul_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_addcmul.Tensor", foreach_addcmul_tensor_cuda);
    m.impl("_foreach_addcmul_.Tensor", foreach_addcmul_tensor_inplace_cuda);
    m.impl("_foreach_addcdiv.Scalar", foreach_addcdiv_scalar_cuda);
    m.impl("_foreach_addcdiv_.Scalar", foreach_addcdiv_scalar_mta_inplace_cuda);
    m.impl("_foreach_addcdiv.ScalarList", foreach_addcdiv_scalar_list_cuda);
    m.impl("_foreach_addcdiv_.ScalarList", foreach_addcdiv_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_addcdiv.Tensor", foreach_addcdiv_tensor_cuda);
    m.impl("_foreach_addcdiv_.Tensor", foreach_addcdiv_tensor_inplace_cuda);
    m.impl("_foreach_lerp.Scalar", foreach_lerp_scalar_cuda);
    m.impl("_foreach_lerp.List", foreach_lerp_list_cuda);
    m.impl("_foreach_lerp_.Scalar", foreach_lerp_scalar_mta_inplace_cuda);
    m.impl("_foreach_lerp_.List", foreach_lerp_list_inplace_cuda);
    m.impl("_foreach_lerp.ScalarList", foreach_lerp_scalar_list_cuda);
    m.impl("_foreach_lerp_.ScalarList", foreach_lerp_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_pow.Scalar", foreach_pow_scalar_mta_ret_cuda);
    m.impl("_foreach_pow.ScalarAndTensor", foreach_pow_scalar_tensor_cuda);
    m.impl("_foreach_pow.TensorAndTensor", foreach_pow_tensor_tensor_cuda);
    m.impl("_foreach_pow.List", foreach_pow_list_mta_ret_cuda);
    m.impl("_foreach_pow_.Scalar", foreach_pow_scalar_mta_inplace_cuda);
    m.impl("_foreach_pow_.List", foreach_pow_list_mta_inplace_cuda);
    m.impl("_foreach_pow.ScalarList", foreach_pow_scalar_list_mta_ret_cuda);
    m.impl("_foreach_pow_.ScalarList", foreach_pow_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_clamp_min.Scalar", foreach_clamp_min_scalar_mta_ret_cuda);
    m.impl("_foreach_clamp_max.Scalar", foreach_clamp_max_scalar_mta_ret_cuda);
    m.impl("_foreach_clamp_min_.Scalar", foreach_clamp_min_scalar_mta_inplace_cuda);
    m.impl("_foreach_clamp_max_.Scalar", foreach_clamp_max_scalar_mta_inplace_cuda);
    m.impl("_foreach_clamp_min.List", foreach_clamp_min_list_mta_ret_cuda);
    m.impl("_foreach_clamp_min_.List", foreach_clamp_min_list_mta_inplace_cuda);
    m.impl("_foreach_clamp_min.ScalarList", foreach_clamp_min_scalar_list_mta_ret_cuda);
    m.impl("_foreach_clamp_min_.ScalarList", foreach_clamp_min_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_clamp_max.List", foreach_clamp_max_list_mta_ret_cuda);
    m.impl("_foreach_clamp_max_.List", foreach_clamp_max_list_mta_inplace_cuda);
    m.impl("_foreach_clamp_max.ScalarList", foreach_clamp_max_scalar_list_mta_ret_cuda);
    m.impl("_foreach_clamp_max_.ScalarList", foreach_clamp_max_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_maximum.Scalar", foreach_maximum_scalar_mta_ret_cuda);
    m.impl("_foreach_minimum.Scalar", foreach_minimum_scalar_mta_ret_cuda);
    m.impl("_foreach_maximum_.Scalar", foreach_maximum_scalar_mta_inplace_cuda);
    m.impl("_foreach_minimum_.Scalar", foreach_minimum_scalar_mta_inplace_cuda);
    m.impl("_foreach_maximum.List", foreach_maximum_list_mta_ret_cuda);
    m.impl("_foreach_maximum_.List", foreach_maximum_list_mta_inplace_cuda);
    m.impl("_foreach_maximum.ScalarList", foreach_maximum_scalar_list_mta_ret_cuda);
    m.impl("_foreach_maximum_.ScalarList", foreach_maximum_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_minimum.List", foreach_minimum_list_mta_ret_cuda);
    m.impl("_foreach_minimum_.List", foreach_minimum_list_mta_inplace_cuda);
    m.impl("_foreach_minimum.ScalarList", foreach_minimum_scalar_list_mta_ret_cuda);
    m.impl("_foreach_minimum_.ScalarList", foreach_minimum_scalar_list_mta_inplace_cuda);
    m.impl("_foreach_copy_", foreach_copy_inplace_cuda);
    m.impl("_foreach_zero_", foreach_zero_mta_inplace_cuda);
}

} // namespace cuda
} // namespace tensorplay
