// Fused Adam / AdamW / AdamW-amsgrad optimizer kernels.  Each kernel handles
// one parameter tensor (flattened over all elements); the caller increments
// the 0-d state_step tensor inside the kernel so the bias correction uses
// the new step count.

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
      [[id(kStateStepsIdx)]];
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