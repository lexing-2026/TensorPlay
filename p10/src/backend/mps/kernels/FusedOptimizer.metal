// Fused Adam / AdamW / AdamW-amsgrad optimizer kernels.  Each kernel handles
// one parameter tensor (flattened over all elements). Bias correction uses
// the state_step value supplied by the caller.

#include <metal_stdlib>

using metal::max;
bfloat max(bfloat a, bfloat b) {
  return a > b ? a : b;
}

enum ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
inline void adam_math_amsgrad(
    device T& param,
    device T& grad,
    device T& exp_avg,
    device T& exp_avg_sq,
    device T& max_exp_avg_sq,
    device state_steps_t& state_steps,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const uint8_t maximize,
    device const float* grad_scale) {
  T grad_ = grad;
  if (grad_scale) {
    grad_ = T(static_cast<float>(grad_) / *grad_scale);
    grad = grad_;
  }

  if (maximize) {
    grad = -grad;
  }

  // Update param, grad, 1st and 2nd order momentum.
  if (weight_decay != 0) {
    switch (adam_mode) {
      case ADAM_MODE::ORIGINAL:
        grad += T(param * weight_decay);
        break;
      case ADAM_MODE::ADAMW:
        param -= T(lr * weight_decay * param);
        break;
    }
  }

  exp_avg = T(beta1 * exp_avg + (1 - beta1) * grad);
  exp_avg_sq = T(beta2 * exp_avg_sq + (1 - beta2) * grad * grad);
  const float casted_state_steps = static_cast<float>(state_steps);
  const auto bias_correction1 =
      1 - metal::precise::pow(beta1, casted_state_steps);
  const auto step_size = lr / bias_correction1;
  const auto bias_correction2 =
      1 - metal::precise::pow(beta2, casted_state_steps);
  const auto bias_correction2_sqrt = metal::precise::sqrt(bias_correction2);
  max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq);

  const auto denom =
      (metal::precise::sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + eps;
  param -= T(step_size * exp_avg / denom);
  grad = grad_;
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
inline void adam_math(
    device T& param,
    device T& grad,
    device T& exp_avg,
    device T& exp_avg_sq,
    device state_steps_t& state_steps,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const uint8_t maximize,
    device const float* grad_scale) {
  T grad_ = grad;
  if (grad_scale) {
    grad_ = T(static_cast<float>(grad_) / *grad_scale);
    grad = grad_;
  }

  if (maximize) {
    grad = -grad;
  }

  // Update param, grad, 1st and 2nd order momentum.
  if (weight_decay != 0) {
    switch (adam_mode) {
      case ADAM_MODE::ORIGINAL:
        grad += T(param * weight_decay);
        break;
      case ADAM_MODE::ADAMW:
        param -= T(lr * weight_decay * param);
        break;
    }
  }

  exp_avg = T(beta1 * exp_avg + (1 - beta1) * grad);
  exp_avg_sq = T(beta2 * exp_avg_sq + (1 - beta2) * grad * grad);
  const float casted_state_steps = static_cast<float>(state_steps);
  const auto bias_correction1 =
      1 - metal::precise::pow(beta1, casted_state_steps);
  const auto step_size = lr / bias_correction1;
  const auto bias_correction2 =
      1 - metal::precise::pow(beta2, casted_state_steps);
  const auto bias_correction2_sqrt = metal::precise::sqrt(bias_correction2);
  const auto denom =
      (metal::precise::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
  param -= T(step_size * exp_avg / denom);
  grad = grad_;
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_amsgrad_single(
    device T* param [[buffer(0)]],
    device T* grad [[buffer(1)]],
    device T* exp_avg [[buffer(2)]],
    device T* exp_avg_sq [[buffer(3)]],
    device T* max_exp_avg_sq [[buffer(4)]],
    device state_steps_t* state_steps [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant float& beta1 [[buffer(7)]],
    constant float& beta2 [[buffer(8)]],
    constant float& weight_decay [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant uint8_t& maximize [[buffer(11)]],
    device const float* grad_scale [[buffer(12)]],
    device const float* found_inf [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  adam_math_amsgrad<T, state_steps_t, adam_mode>(
      param[tid],
      grad[tid],
      exp_avg[tid],
      exp_avg_sq[tid],
      max_exp_avg_sq[tid],
      state_steps[0],
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale);
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_single(
    device T* param [[buffer(0)]],
    device T* grad [[buffer(1)]],
    device T* exp_avg [[buffer(2)]],
    device T* exp_avg_sq [[buffer(3)]],
    device state_steps_t* state_steps [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint8_t& maximize [[buffer(10)]],
    device const float* grad_scale [[buffer(11)]],
    device const float* found_inf [[buffer(12)]],
    uint tid [[thread_position_in_grid]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  adam_math<T, state_steps_t, adam_mode>(
      param[tid],
      grad[tid],
      exp_avg[tid],
      exp_avg_sq[tid],
      state_steps[0],
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale);
}

#define REGISTER_FUSED_SINGLE(                                              \
    DTYPE, ADAM_MODE_DTYPE, HOST_NAME, KERNEL_NAME, AMSGRAD)                \
  template [[host_name(#HOST_NAME "_" #DTYPE "_float")]] kernel void        \
  KERNEL_NAME<DTYPE, float, ADAM_MODE_DTYPE>(                               \
      device DTYPE* param [[buffer(0)]],                                    \
      device DTYPE* grad [[buffer(1)]],                                     \
      device DTYPE* exp_avg [[buffer(2)]],                                   \
      device DTYPE* exp_avg_sq [[buffer(3)]],                                \
      device DTYPE* max_exp_avg_sq [[buffer(4)]],                           \
      device float* state_steps [[buffer(5)]],                                \
      constant float& lr [[buffer(6)]],                                       \
      constant float& beta1 [[buffer(7)]],                                   \
      constant float& beta2 [[buffer(8)]],                                   \
      constant float& weight_decay [[buffer(9)]],                            \
      constant float& eps [[buffer(10)]],                                    \
      constant uint8_t& maximize [[buffer(11)]],                            \
      device const float* grad_scale [[buffer(12)]],                       \
      device const float* found_inf [[buffer(13)]],                       \
      uint tid [[thread_position_in_grid]])

#define REGISTER_FUSED_ADAM_OP(D1)                       \
  template [[host_name("fused_adam_" #D1 "_float")]]     \
  kernel void fused_adam_single<D1, float, ADAM_MODE::ORIGINAL>(            \
      device D1* param [[buffer(0)]],                                       \
      device D1* grad [[buffer(1)]],                                        \
      device D1* exp_avg [[buffer(2)]],                                    \
      device D1* exp_avg_sq [[buffer(3)]],                                \
      device float* state_steps [[buffer(4)]],                               \
      constant float& lr [[buffer(5)]],                                      \
      constant float& beta1 [[buffer(6)]],                                    \
      constant float& beta2 [[buffer(7)]],                                   \
      constant float& weight_decay [[buffer(8)]],                            \
      constant float& eps [[buffer(9)]],                                     \
      constant uint8_t& maximize [[buffer(10)]],                            \
      device const float* grad_scale [[buffer(11)]],                       \
      device const float* found_inf [[buffer(12)]],                       \
      uint tid [[thread_position_in_grid]])

// clang-format on
#define REGISTER_FUSED_ADAM_AMSGRAD_OP(D1) \
  template [[host_name("fused_adam_amsgrad_" #D1 "_float")]] \
  kernel void fused_adam_amsgrad_single<D1, float, ADAM_MODE::ORIGINAL>( \
      device D1* param [[buffer(0)]], device D1* grad [[buffer(1)]], \
      device D1* exp_avg [[buffer(2)]], device D1* exp_avg_sq [[buffer(3)]], \
      device D1* max_exp_avg_sq [[buffer(4)]], device float* state_steps [[buffer(5)]], \
      constant float& lr [[buffer(6)]], constant float& beta1 [[buffer(7)]], \
      constant float& beta2 [[buffer(8)]], constant float& weight_decay [[buffer(9)]], \
      constant float& eps [[buffer(10)]], constant uint8_t& maximize [[buffer(11)]], \
      device const float* grad_scale [[buffer(12)]], device const float* found_inf [[buffer(13)]], \
      uint tid [[thread_position_in_grid]])

#define REGISTER_FUSED_ADAMW_AMSGRAD_OP(D1) \
  template [[host_name("fused_adamw_amsgrad_" #D1 "_float")]] \
  kernel void fused_adam_amsgrad_single<D1, float, ADAM_MODE::ADAMW>( \
      device D1* param [[buffer(0)]], device D1* grad [[buffer(1)]], \
      device D1* exp_avg [[buffer(2)]], device D1* exp_avg_sq [[buffer(3)]], \
      device D1* max_exp_avg_sq [[buffer(4)]], device float* state_steps [[buffer(5)]], \
      constant float& lr [[buffer(6)]], constant float& beta1 [[buffer(7)]], \
      constant float& beta2 [[buffer(8)]], constant float& weight_decay [[buffer(9)]], \
      constant float& eps [[buffer(10)]], constant uint8_t& maximize [[buffer(11)]], \
      device const float* grad_scale [[buffer(12)]], device const float* found_inf [[buffer(13)]], \
      uint tid [[thread_position_in_grid]])

#define REGISTER_FUSED_ADAMW_OP(D1) \
  template [[host_name("fused_adamw_" #D1 "_float")]] \
  kernel void fused_adam_single<D1, float, ADAM_MODE::ADAMW>( \
      device D1* param [[buffer(0)]], device D1* grad [[buffer(1)]], \
      device D1* exp_avg [[buffer(2)]], device D1* exp_avg_sq [[buffer(3)]], \
      device float* state_steps [[buffer(4)]], \
      constant float& lr [[buffer(5)]], constant float& beta1 [[buffer(6)]], \
      constant float& beta2 [[buffer(7)]], constant float& weight_decay [[buffer(8)]], \
      constant float& eps [[buffer(9)]], constant uint8_t& maximize [[buffer(10)]], \
      device const float* grad_scale [[buffer(11)]], device const float* found_inf [[buffer(12)]], \
      uint tid [[thread_position_in_grid]])

REGISTER_FUSED_ADAM_OP(float);
REGISTER_FUSED_ADAMW_OP(float);
REGISTER_FUSED_ADAM_AMSGRAD_OP(float);
REGISTER_FUSED_ADAMW_AMSGRAD_OP(float);
REGISTER_FUSED_ADAM_OP(half);
REGISTER_FUSED_ADAMW_OP(half);
REGISTER_FUSED_ADAM_AMSGRAD_OP(half);
REGISTER_FUSED_ADAMW_AMSGRAD_OP(half);

// ---------------------------------------------------------------------------
// Multi-tensor (foreach) kernels: one dispatch updates up to 32 parameters.
// The argument struct binds arrays of device pointers via an argument
// encoder; MetadataArguments maps each threadgroup to a (tensor, chunk) pair.
// ---------------------------------------------------------------------------

#include <metal_array>

constexpr constant uint kmaxThreadGroups = 32;
constexpr constant uint kmaxTensors = 32;
constexpr constant uint kChunkSize = 65536;

constexpr constant uint kParamIdx = 0;
constexpr constant uint kGradIdx = kParamIdx + kmaxTensors;
constexpr constant uint kExpAvgIdx = kGradIdx + kmaxTensors;
constexpr constant uint kExpAvgSqIdx = kExpAvgIdx + kmaxTensors;
constexpr constant uint kMaxExpAvgSqIdx = kExpAvgSqIdx + kmaxTensors;
constexpr constant uint kStateStepsIdx = kExpAvgSqIdx + kmaxTensors;
constexpr constant uint kStateStepsIdxForAmsgrad =
    kMaxExpAvgSqIdx + kmaxTensors;

template <typename T, typename state_steps_t>
struct AdamArguments {
  metal::array<device T*, kmaxTensors> params [[id(kParamIdx)]];
  metal::array<device T*, kmaxTensors> grads [[id(kGradIdx)]];
  metal::array<device T*, kmaxTensors> exp_avgs [[id(kExpAvgIdx)]];
  metal::array<device T*, kmaxTensors> exp_avg_sqs [[id(kExpAvgSqIdx)]];
  metal::array<device state_steps_t*, kmaxTensors> state_steps
      [[id(kStateStepsIdx)]];
};

template <typename T, typename state_steps_t>
struct AdamAmsgradArguments {
  metal::array<device T*, kmaxTensors> params [[id(kParamIdx)]];
  metal::array<device T*, kmaxTensors> grads [[id(kGradIdx)]];
  metal::array<device T*, kmaxTensors> exp_avgs [[id(kExpAvgIdx)]];
  metal::array<device T*, kmaxTensors> exp_avg_sqs [[id(kExpAvgSqIdx)]];
  metal::array<device T*, kmaxTensors> max_exp_avg_sqs [[id(kMaxExpAvgSqIdx)]];
  metal::array<device state_steps_t*, kmaxTensors> state_steps
      [[id(kStateStepsIdxForAmsgrad)]];
};

struct MetadataArguments {
  ulong numels[kmaxTensors];
  ulong threadgroup_to_tensor[kmaxThreadGroups];
  ulong threadgroup_to_chunk[kmaxThreadGroups];
};

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_multi(
    device AdamArguments<T, state_steps_t>& args [[buffer(0)]],
    constant MetadataArguments& metadata_args [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    constant float& beta1 [[buffer(3)]],
    constant float& beta2 [[buffer(4)]],
    constant float& weight_decay [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint8_t& maximize [[buffer(7)]],
    device const float* grad_scale [[buffer(8)]],
    device const float* found_inf [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
  const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
  const uint32_t chunk_offset = chunk_idx * kChunkSize;
  const uint32_t numel =
      static_cast<uint32_t>(metadata_args.numels[tensor_loc] - chunk_offset);
  const auto step_count = args.state_steps[tensor_loc];

  auto param = args.params[tensor_loc] + chunk_offset;
  auto grad = args.grads[tensor_loc] + chunk_offset;
  auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
  auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;

  for (uint32_t i_start = tid; i_start < numel && i_start < kChunkSize;
       i_start += tptg) {
    adam_math<T, state_steps_t, adam_mode>(
        *(param + i_start),
        *(grad + i_start),
        *(exp_avg + i_start),
        *(exp_avg_sq + i_start),
        *step_count,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale);
  }
}

template <typename T, typename state_steps_t, ADAM_MODE adam_mode>
kernel void fused_adam_amsgrad_multi(
    device AdamAmsgradArguments<T, state_steps_t>& args [[buffer(0)]],
    constant MetadataArguments& metadata_args [[buffer(1)]],
    constant float& lr [[buffer(2)]],
    constant float& beta1 [[buffer(3)]],
    constant float& beta2 [[buffer(4)]],
    constant float& weight_decay [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint8_t& maximize [[buffer(7)]],
    device const float* grad_scale [[buffer(8)]],
    device const float* found_inf [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
  const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
  const uint32_t chunk_offset = chunk_idx * kChunkSize;
  const uint32_t numel =
      static_cast<uint32_t>(metadata_args.numels[tensor_loc] - chunk_offset);
  const auto step_count = args.state_steps[tensor_loc];

  auto param = args.params[tensor_loc] + chunk_offset;
  auto grad = args.grads[tensor_loc] + chunk_offset;
  auto exp_avg = args.exp_avgs[tensor_loc] + chunk_offset;
  auto exp_avg_sq = args.exp_avg_sqs[tensor_loc] + chunk_offset;
  auto max_exp_avg_sq = args.max_exp_avg_sqs[tensor_loc] + chunk_offset;

  for (uint32_t i_start = tid; i_start < numel && i_start < kChunkSize;
       i_start += tptg) {
    adam_math_amsgrad<T, state_steps_t, adam_mode>(
        *(param + i_start),
        *(grad + i_start),
        *(exp_avg + i_start),
        *(exp_avg_sq + i_start),
        *(max_exp_avg_sq + i_start),
        *step_count,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale);
  }
}

#define REGISTER_FUSED_MULTI(DTYPE, KERNEL_NAME, HOST_NAME, ARGS_TYPE,      \
                              ADAM_MODE_DTYPE)                               \
  template [[host_name(#HOST_NAME "_" #DTYPE "_float")]] kernel void        \
  KERNEL_NAME<DTYPE, float, ADAM_MODE_DTYPE>(                               \
      device ARGS_TYPE<DTYPE, float>& args [[buffer(0)]],                  \
      constant MetadataArguments& metadata_args [[buffer(1)]],              \
      constant float& lr [[buffer(2)]],                                    \
      constant float& beta1 [[buffer(3)]],                                \
      constant float& beta2 [[buffer(4)]],                                \
      constant float& weight_decay [[buffer(5)]],                          \
      constant float& eps [[buffer(6)]],                                    \
      constant uint8_t& maximize [[buffer(7)]],                           \
      device const float* grad_scale [[buffer(8)]],                       \
      device const float* found_inf [[buffer(9)]],                        \
      uint tid [[thread_position_in_threadgroup]],                         \
      uint tgid [[threadgroup_position_in_grid]],                          \
      uint tptg [[threads_per_threadgroup]])

#define REGISTER_FUSED_MULTI_ADAM(D1) \
  REGISTER_FUSED_MULTI(D1, fused_adam_multi, fused_adam_multi, AdamArguments, \
                       ADAM_MODE::ORIGINAL)

#define REGISTER_FUSED_MULTI_ADAMW(D1) \
  REGISTER_FUSED_MULTI(D1, fused_adam_multi, fused_adamw_multi, AdamArguments, \
                      ADAM_MODE::ADAMW)

#define REGISTER_FUSED_MULTI_ADAM_AMSGRAD(D1) \
  REGISTER_FUSED_MULTI(D1, fused_adam_amsgrad_multi, fused_adam_amsgrad_multi, \
                       AdamAmsgradArguments, ADAM_MODE::ORIGINAL)

#define REGISTER_FUSED_MULTI_ADAMW_AMSGRAD(D1) \
  REGISTER_FUSED_MULTI(D1, fused_adam_amsgrad_multi, fused_adamw_amsgrad_multi, \
                       AdamAmsgradArguments, ADAM_MODE::ADAMW)

REGISTER_FUSED_MULTI_ADAM(float);
REGISTER_FUSED_MULTI_ADAMW(float);
REGISTER_FUSED_MULTI_ADAM_AMSGRAD(float);
REGISTER_FUSED_MULTI_ADAMW_AMSGRAD(float);
REGISTER_FUSED_MULTI_ADAM(half);
REGISTER_FUSED_MULTI_ADAMW(half);
REGISTER_FUSED_MULTI_ADAM_AMSGRAD(half);
REGISTER_FUSED_MULTI_ADAMW_AMSGRAD(half);

constexpr constant uint kMomentumBufferListIdx = kGradIdx + kmaxTensors;

template <typename T>
struct SgdArguments {
  metal::array<device T*, kmaxTensors> params [[id(kParamIdx)]];
  metal::array<device T*, kmaxTensors> grads [[id(kGradIdx)]];
};

template <typename T>
struct SgdMomentumArguments {
  metal::array<device T*, kmaxTensors> params [[id(kParamIdx)]];
  metal::array<device T*, kmaxTensors> grads [[id(kGradIdx)]];
  metal::array<device T*, kmaxTensors> momentum_buffer_list
      [[id(kMomentumBufferListIdx)]];
};

template <typename T>
inline void sgd_momentum_math(
    device T& param,
    device T& grad,
    device T& momentum_buffer,
    const float weight_decay,
    const float momentum,
    const float lr,
    const float dampening,
    const uint8_t nesterov,
    const uint8_t maximize,
    const uint8_t is_first_step,
    device const float* grad_scale) {
  auto grad_ = grad;
  if (grad_scale) {
    grad_ = T(static_cast<float>(grad_) / *grad_scale);
    grad = grad_;
  }
  if (maximize) {
    grad_ *= T(-1.0);
  }
  if (weight_decay != 0) {
    grad_ += T(weight_decay * param);
  }

  momentum_buffer = is_first_step
      ? grad_
      : T(momentum * momentum_buffer + (1 - dampening) * grad_);
  if (nesterov) {
    grad_ += T(momentum * momentum_buffer);
  } else {
    grad_ = momentum_buffer;
  }

  param -= T(lr * grad_);
}

template <typename T>
inline void sgd_math(device T& param,
                     device T& grad,
                     const float weight_decay,
                     const float lr,
                     const uint8_t maximize,
                     device const float* grad_scale) {
  auto grad_ = grad;
  if (grad_scale) {
    grad_ = T(static_cast<float>(grad_) / *grad_scale);
    grad = grad_;
  }
  if (maximize) {
    grad_ *= T(-1.0);
  }
  if (weight_decay != 0) {
    grad_ += T(weight_decay * param);
  }

  param -= T(lr * grad_);
}

template <typename T>
kernel void fused_sgd(
    device SgdArguments<T>& args [[buffer(0)]],
    constant MetadataArguments& metadata_args [[buffer(1)]],
    constant float& weight_decay [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant uint8_t& maximize [[buffer(4)]],
    device const float* grad_scale [[buffer(5)]],
    device const float* found_inf [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
  const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
  const uint32_t chunk_offset = chunk_idx * kChunkSize;
  const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

  auto param = args.params[tensor_loc] + chunk_offset;
  auto grad = args.grads[tensor_loc] + chunk_offset;

  for (uint32_t i_start = tid; i_start < numel && i_start < kChunkSize;
       i_start += tptg) {
    sgd_math<T>(*(param + i_start), *(grad + i_start), weight_decay, lr,
                maximize, grad_scale);
  }
}

template <typename T>
kernel void fused_sgd(
    device SgdMomentumArguments<T>& args [[buffer(0)]],
    constant MetadataArguments& metadata_args [[buffer(1)]],
    constant float& weight_decay [[buffer(2)]],
    constant float& momentum [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& dampening [[buffer(5)]],
    constant uint8_t& nesterov [[buffer(6)]],
    constant uint8_t& maximize [[buffer(7)]],
    constant uint8_t& is_first_step [[buffer(8)]],
    device const float* grad_scale [[buffer(9)]],
    device const float* found_inf [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  const uint32_t tensor_loc = metadata_args.threadgroup_to_tensor[tgid];
  const uint32_t chunk_idx = metadata_args.threadgroup_to_chunk[tgid];
  const uint32_t chunk_offset = chunk_idx * kChunkSize;
  const uint32_t numel = metadata_args.numels[tensor_loc] - chunk_offset;

  auto param = args.params[tensor_loc] + chunk_offset;
  auto grad = args.grads[tensor_loc] + chunk_offset;
  auto momentum_buffer_list =
      args.momentum_buffer_list[tensor_loc] + chunk_offset;

  for (uint32_t i_start = tid; i_start < numel && i_start < kChunkSize;
       i_start += tptg) {
    sgd_momentum_math<T>(*(param + i_start), *(grad + i_start),
                         *(momentum_buffer_list + i_start), weight_decay,
                         momentum, lr, dampening, nesterov, maximize,
                         is_first_step, grad_scale);
  }
}

// One parameter per launch, bound directly rather than through an argument
// struct.  These need no argument encoder, so they also run on a GPU image that
// has none.
template <typename T>
kernel void fused_sgd_single(
    device T* param [[buffer(0)]],
    device T* grad [[buffer(1)]],
    constant float& weight_decay [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    constant uint8_t& maximize [[buffer(4)]],
    device const float* grad_scale [[buffer(5)]],
    device const float* found_inf [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  sgd_math<T>(param[tid], grad[tid], weight_decay, lr, maximize, grad_scale);
}

template <typename T>
kernel void fused_sgd_momentum_single(
    device T* param [[buffer(0)]],
    device T* grad [[buffer(1)]],
    device T* momentum_buffer [[buffer(2)]],
    constant float& weight_decay [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant float& dampening [[buffer(6)]],
    constant uint8_t& nesterov [[buffer(7)]],
    constant uint8_t& maximize [[buffer(8)]],
    constant uint8_t& is_first_step [[buffer(9)]],
    device const float* grad_scale [[buffer(10)]],
    device const float* found_inf [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  if (found_inf && *found_inf == 1) {
    return;
  }
  sgd_momentum_math<T>(param[tid], grad[tid], momentum_buffer[tid], weight_decay,
                       momentum, lr, dampening, nesterov, maximize,
                       is_first_step, grad_scale);
}

#define REGISTER_FUSED_SGD_SINGLE_OP(DTYPE)                                 \
  template [[host_name("fused_sgd_single_" #DTYPE)]] kernel void            \
  fused_sgd_single<DTYPE>(                                                  \
      device DTYPE* param [[buffer(0)]],                                    \
      device DTYPE* grad [[buffer(1)]],                                     \
      constant float& weight_decay [[buffer(2)]],                           \
      constant float& lr [[buffer(3)]],                                     \
      constant uint8_t& maximize [[buffer(4)]],                             \
      device const float* grad_scale [[buffer(5)]],                         \
      device const float* found_inf [[buffer(6)]],                          \
      uint tid [[thread_position_in_grid]])

#define REGISTER_FUSED_SGD_MOMENTUM_SINGLE_OP(DTYPE)                        \
  template [[host_name("fused_sgd_momentum_single_" #DTYPE)]] kernel void    \
  fused_sgd_momentum_single<DTYPE>(                                         \
      device DTYPE* param [[buffer(0)]],                                    \
      device DTYPE* grad [[buffer(1)]],                                     \
      device DTYPE* momentum_buffer [[buffer(2)]],                          \
      constant float& weight_decay [[buffer(3)]],                           \
      constant float& momentum [[buffer(4)]],                               \
      constant float& lr [[buffer(5)]],                                     \
      constant float& dampening [[buffer(6)]],                              \
      constant uint8_t& nesterov [[buffer(7)]],                             \
      constant uint8_t& maximize [[buffer(8)]],                             \
      constant uint8_t& is_first_step [[buffer(9)]],                        \
      device const float* grad_scale [[buffer(10)]],                        \
      device const float* found_inf [[buffer(11)]],                         \
      uint tid [[thread_position_in_grid]])

#define REGISTER_FUSED_SGD_OP(DTYPE)                                        \
  template [[host_name("fused_sgd_" #DTYPE)]] kernel void fused_sgd<DTYPE>(  \
      device SgdArguments<DTYPE>& args [[buffer(0)]],                       \
      constant MetadataArguments& metadata_args [[buffer(1)]],             \
      constant float& weight_decay [[buffer(2)]],                           \
      constant float& lr [[buffer(3)]],                                      \
      constant uint8_t& maximize [[buffer(4)]],                           \
      device const float* grad_scale [[buffer(5)]],                         \
      device const float* found_inf [[buffer(6)]],                          \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint tgid [[threadgroup_position_in_grid]],                           \
      uint tptg [[threads_per_threadgroup]])

#define REGISTER_FUSED_SGD_MOMENTUM_OP(DTYPE)                       \
  template [[host_name("fused_sgd_momentum_" #DTYPE)]] kernel    \
  void fused_sgd<DTYPE>(                                         \
      device SgdMomentumArguments<DTYPE>& args [[buffer(0)]],  \
      constant MetadataArguments& metadata_args [[buffer(1)]],   \
      constant float& weight_decay [[buffer(2)]],                 \
      constant float& momentum [[buffer(3)]],                     \
      constant float& lr [[buffer(4)]],                           \
      constant float& dampening [[buffer(5)]],                    \
      constant uint8_t& nesterov [[buffer(6)]],                  \
      constant uint8_t& maximize [[buffer(7)]],                 \
      constant uint8_t& is_first_step [[buffer(8)]],             \
      device const float* grad_scale [[buffer(9)]],              \
      device const float* found_inf [[buffer(10)]],               \
      uint tid [[thread_position_in_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],                \
      uint tptg [[threads_per_threadgroup]])

REGISTER_FUSED_SGD_OP(float);
REGISTER_FUSED_SGD_OP(half);
REGISTER_FUSED_SGD_OP(bfloat);
REGISTER_FUSED_SGD_MOMENTUM_OP(float);
REGISTER_FUSED_SGD_MOMENTUM_OP(half);
REGISTER_FUSED_SGD_MOMENTUM_OP(bfloat);
REGISTER_FUSED_SGD_SINGLE_OP(float);
REGISTER_FUSED_SGD_SINGLE_OP(half);
REGISTER_FUSED_SGD_SINGLE_OP(bfloat);
REGISTER_FUSED_SGD_MOMENTUM_SINGLE_OP(float);
REGISTER_FUSED_SGD_MOMENTUM_SINGLE_OP(half);
REGISTER_FUSED_SGD_MOMENTUM_SINGLE_OP(bfloat);
