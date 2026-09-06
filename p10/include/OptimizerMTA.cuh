#pragma once

// Native CUDA multi-tensor optimizer kernels.
//
// This is intentionally separate from the generic foreach header: optimizer
// updates write more than one tensor (parameter plus optimizer state), and
// fused optimizers also carry a device-side step list.  The layout follows
// chunks from all tensors are packed into one launch, and each thread handles
// four contiguous scalar values in the aligned path.

#include "CUDARuntime.h"
#include "Exception.h"
#include "Tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace optimizer_mta {

constexpr int kILP = 4;
constexpr int64_t kChunkSize = 65536;
constexpr int kBlockSize = 512;

// optimizer kernels within the conservative 4 KiB kernel-argument limit.
template <int Depth>
constexpr int kMaxTensorsForDepth =
    Depth == 1 ? 77 :
    (Depth == 2 ? 62 :
     (Depth == 3 ? 51 :
      (Depth == 4 ? 44 :
       (Depth == 5 ? 38 : 0))));
// Adafactor carries original tensor indices and trailing dimensions for its
// reduction/apply split, so its metadata has a smaller, extended capacity.
template <int Depth>
constexpr int kMaxExtendedTensorsForDepth =
    Depth == 1 ? 47 :
    (Depth == 2 ? 41 :
     (Depth == 3 ? 36 :
      (Depth == 4 ? 32 :
       (Depth == 5 ? 29 : 0))));
// Plain MTA does not need step values or Adafactor's tensor bookkeeping.
// Keep a larger per-launch capacity than the extended metadata while staying
// below CUDA 12's 4 KiB kernel-argument limit.  This matters for the common
// optimizer shapes: a depth-3 SGD group of 104 tensors fits in two launches,
// and a depth-5 centered RMSprop group fits in three.
template <int Depth>
constexpr int kMaxPlainTensorsForDepth =
    Depth == 1 ? 128 :
    (Depth == 2 ? 96 :
     (Depth == 3 ? 64 :
      (Depth == 4 ? 48 :
       (Depth == 5 ? 40 : 0))));
constexpr int kMaxBlocks = 320;

template <typename T>
struct alignas(kILP * sizeof(T)) AlignedVec {
    T values[kILP];
};

template <typename T>
__device__ __forceinline__ bool is_aligned(const T* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) &
            ((kILP * sizeof(T)) - 1)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(
        T* dst, const T* src, int64_t dst_index, int64_t src_index) {
    using Vec = AlignedVec<T>;
    reinterpret_cast<Vec*>(dst)[dst_index] =
        reinterpret_cast<const Vec*>(src)[src_index];
}

template <int Depth>
struct TensorMetadata {
    static constexpr int kMaxTensors = kMaxExtendedTensorsForDepth<Depth>;
    struct HostSteps {
        double step_sizes[kMaxTensors]{};
        double correction2_sqrts[kMaxTensors]{};
    };
    union StepStorage {
        const void* state_steps[kMaxTensors];
        HostSteps host;
    };
    const void* addresses[Depth][kMaxTensors]{};
    int64_t numel_for_tensor[kMaxTensors]{};
    int32_t tensor_indices[kMaxTensors]{};
    int64_t dim_minus2[kMaxTensors]{};
    int64_t dim_minus1[kMaxTensors]{};
    StepStorage step_metadata{};
    uint8_t block_to_tensor[kMaxBlocks]{};
    int32_t block_to_chunk[kMaxBlocks]{};
};

template <int Depth>
struct SimpleTensorMetadata {
    static constexpr int kMaxTensors = kMaxTensorsForDepth<Depth>;
    struct HostSteps {
        double step_sizes[kMaxTensors]{};
        double correction2_sqrts[kMaxTensors]{};
    };
    union StepStorage {
        const void* state_steps[kMaxTensors];
        HostSteps host;
    };
    const void* addresses[Depth][kMaxTensors]{};
    int64_t numel_for_tensor[kMaxTensors]{};
    StepStorage step_metadata{};
    uint8_t block_to_tensor[kMaxBlocks]{};
    int32_t block_to_chunk[kMaxBlocks]{};
};

template <int Depth>
struct PlainTensorMetadata {
    static constexpr int kMaxTensors = kMaxPlainTensorsForDepth<Depth>;
    const void* addresses[Depth][kMaxTensors]{};
    int64_t numel_for_tensor[kMaxTensors]{};
    uint8_t block_to_tensor[kMaxBlocks]{};
    int32_t block_to_chunk[kMaxBlocks]{};
};

template <int Depth, typename Metadata>
inline void set_extended_metadata(
        Metadata& metadata, int32_t slot, size_t tensor_index,
        const std::array<const std::vector<Tensor>*, Depth>& lists) {
    // Keep this as one dependent template instead of an overload set. NVCC
    // otherwise fails to resolve the metadata overloads from
    // launch_batches_impl while Metadata is still dependent.
    if constexpr (std::is_same_v<Metadata, TensorMetadata<Depth>>) {
        metadata.tensor_indices[slot] = static_cast<int32_t>(tensor_index);
        metadata.dim_minus2[slot] = (*lists[0])[tensor_index].dim() >= 2
            ? (*lists[0])[tensor_index].size(-2) : 0;
        metadata.dim_minus1[slot] = (*lists[0])[tensor_index].dim() >= 1
            ? (*lists[0])[tensor_index].size(-1) : 0;
    }
}

template <typename Metadata>
inline void set_step_metadata(
        Metadata&, int32_t, size_t, const std::vector<Tensor>*,
        const std::vector<double>*, const std::vector<double>*) {}

template <int Depth>
inline void set_step_metadata(
        TensorMetadata<Depth>& metadata, int32_t slot, size_t tensor_index,
        const std::vector<Tensor>* state_steps,
        const std::vector<double>* step_sizes,
        const std::vector<double>* correction2_sqrts) {
    if (state_steps != nullptr) {
        metadata.step_metadata.state_steps[slot] =
            (*state_steps)[tensor_index].data_ptr<float>();
    }
    if (step_sizes != nullptr) {
        metadata.step_metadata.host.step_sizes[slot] =
            (*step_sizes)[tensor_index];
    }
    if (correction2_sqrts != nullptr) {
        metadata.step_metadata.host.correction2_sqrts[slot] =
            (*correction2_sqrts)[tensor_index];
    }
}

template <int Depth>
inline void set_step_metadata(
        SimpleTensorMetadata<Depth>& metadata, int32_t slot,
        size_t tensor_index, const std::vector<Tensor>* state_steps,
        const std::vector<double>* step_sizes,
        const std::vector<double>* correction2_sqrts) {
    if (state_steps != nullptr) {
        metadata.step_metadata.state_steps[slot] =
            (*state_steps)[tensor_index].data_ptr<float>();
    }
    if (step_sizes != nullptr) {
        metadata.step_metadata.host.step_sizes[slot] =
            (*step_sizes)[tensor_index];
    }
    if (correction2_sqrts != nullptr) {
        metadata.step_metadata.host.correction2_sqrts[slot] =
            (*correction2_sqrts)[tensor_index];
    }
}

template <int Depth, typename Metadata, typename LaunchFn>
void launch_batches_impl(
        const std::array<const std::vector<Tensor>*, Depth>& lists,
        const std::vector<Tensor>* state_steps,
        const std::vector<double>* step_sizes,
        const std::vector<double>* correction2_sqrts,
        LaunchFn&& launch, const char* op_name) {
    constexpr int max_tensors = Metadata::kMaxTensors;
    const size_t tensor_count = lists[0]->size();
    for (int depth = 1; depth < Depth; ++depth) {
        if (lists[depth]->size() != tensor_count) {
            TP_THROW(ValueError, std::string(op_name) +
                ": tensor list arguments must have the same length");
        }
    }
    if (state_steps != nullptr && state_steps->size() != tensor_count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": state_steps must have the same length as the tensor list");
    }
    if (step_sizes != nullptr && step_sizes->size() != tensor_count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": step-size metadata must have the same length as the tensor list");
    }
    if (correction2_sqrts != nullptr &&
        correction2_sqrts->size() != tensor_count) {
        TP_THROW(ValueError, std::string(op_name) +
            ": bias-correction metadata must have the same length as the tensor list");
    }
    if (tensor_count == 0) return;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    size_t tensor_index = 0;
    int32_t next_chunk = 0;
    while (tensor_index < tensor_count) {
        Metadata metadata{};
        int32_t tensor_slots = 0;
        int32_t block_count = 0;

        while (tensor_index < tensor_count) {
            const int64_t numel = (*lists[0])[tensor_index].numel();
            if (numel == 0) {
                ++tensor_index;
                next_chunk = 0;
                continue;
            }
            const int32_t chunks = static_cast<int32_t>(
                (numel + kChunkSize - 1) / kChunkSize);

            if (tensor_slots == 0 || next_chunk == 0) {
                if (tensor_slots == max_tensors ||
                    block_count == kMaxBlocks) {
                    break;
                }
                for (int depth = 0; depth < Depth; ++depth) {
                    metadata.addresses[depth][tensor_slots] =
                        (*lists[depth])[tensor_index].data_ptr();
                }
                metadata.numel_for_tensor[tensor_slots] = numel;
                set_extended_metadata<Depth, Metadata>(
                    metadata, tensor_slots, tensor_index, lists);
                set_step_metadata(metadata, tensor_slots, tensor_index,
                                  state_steps, step_sizes, correction2_sqrts);
                ++tensor_slots;
            }

            const int32_t available = kMaxBlocks - block_count;
            const int32_t remaining = chunks - next_chunk;
            const int32_t take = std::min(available, remaining);
            for (int32_t chunk = 0; chunk < take; ++chunk) {
                metadata.block_to_tensor[block_count + chunk] =
                    static_cast<uint8_t>(tensor_slots - 1);
                metadata.block_to_chunk[block_count + chunk] =
                    next_chunk + chunk;
            }
            block_count += take;
            next_chunk += take;
            if (next_chunk == chunks) {
                ++tensor_index;
                next_chunk = 0;
            } else {
                break;
            }
        }

        if (block_count == 0) {
            TP_THROW(ValueError, std::string(op_name) +
                ": unable to construct a CUDA optimizer launch");
        }
        launch(metadata, block_count, stream);
    }
}

template <int Depth, typename LaunchFn>
void launch_batches(
        const std::array<const std::vector<Tensor>*, Depth>& lists,
        const std::vector<Tensor>* state_steps,
        const std::vector<double>* step_sizes,
        const std::vector<double>* correction2_sqrts,
        LaunchFn&& launch, const char* op_name) {
    launch_batches_impl<Depth, TensorMetadata<Depth>>(
        lists, state_steps, step_sizes, correction2_sqrts,
        std::forward<LaunchFn>(launch), op_name);
}

template <int Depth, typename LaunchFn>
void launch_simple_batches(
        const std::array<const std::vector<Tensor>*, Depth>& lists,
        const std::vector<Tensor>* state_steps,
        const std::vector<double>* step_sizes,
        const std::vector<double>* correction2_sqrts,
        LaunchFn&& launch, const char* op_name) {
    launch_batches_impl<Depth, SimpleTensorMetadata<Depth>>(
        lists, state_steps, step_sizes, correction2_sqrts,
        std::forward<LaunchFn>(launch), op_name);
}

template <int Depth, typename LaunchFn>
void launch_plain_batches(
        const std::array<const std::vector<Tensor>*, Depth>& lists,
        LaunchFn&& launch, const char* op_name) {
    launch_batches_impl<Depth, PlainTensorMetadata<Depth>>(
        lists, nullptr, nullptr, nullptr,
        std::forward<LaunchFn>(launch), op_name);
}

template <int Depth, typename T, typename M>
__device__ __forceinline__ void load_args(
        M values[][kILP], const T* const* args,
        int64_t i_start, int64_t chunk_size, int64_t n) {
#pragma unroll
    for (int lane = 0; lane < kILP; ++lane) {
        const int64_t i = i_start + threadIdx.x +
                          static_cast<int64_t>(lane) * blockDim.x;
#pragma unroll
        for (int depth = 0; depth < Depth; ++depth) {
            values[depth][lane] = M(0);
            if (i < n && i < chunk_size) {
                values[depth][lane] = static_cast<M>(args[depth][i]);
            }
        }
    }
}

template <int Depth, typename T, typename M>
__device__ __forceinline__ void store_args(
        T* const* args, const M values[][kILP],
        int64_t i_start, int64_t chunk_size, int64_t n,
        bool store_grad) {
#pragma unroll
    for (int lane = 0; lane < kILP; ++lane) {
        const int64_t i = i_start + threadIdx.x +
                          static_cast<int64_t>(lane) * blockDim.x;
        if (i < n && i < chunk_size) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                if (depth == 1 && !store_grad) continue;
                args[depth][i] = static_cast<T>(values[depth][lane]);
            }
        }
    }
}

// Keep the low-precision foreach bodies in their own fragment.  It is
// included inside optimizer_mta so it can reuse the common metadata and
// pointwise kernel without making the launch/dispatch header a monolith.
#include "OptimizerMTAExact.cuh"

template <typename scalar_t, typename math_t, bool HasMomentum>
__device__ __forceinline__ void sgd_math(
        scalar_t r_args[][kILP], math_t lr, math_t momentum,
        math_t dampening, math_t weight_decay, bool nesterov,
        bool first_momentum_step, bool maximize, const float* grad_scale) {
#pragma unroll
    for (int lane = 0; lane < kILP; ++lane) {
        math_t param = static_cast<math_t>(r_args[0][lane]);
        math_t grad = static_cast<math_t>(r_args[1][lane]);
        if (grad_scale != nullptr) {
            grad /= static_cast<math_t>(*grad_scale);
            r_args[1][lane] = static_cast<scalar_t>(grad);
        }
        if (maximize) grad *= math_t(-1);
        if (weight_decay != math_t(0)) grad += weight_decay * param;
        math_t update = grad;
        if constexpr (HasMomentum) {
            const math_t buffer = first_momentum_step
                ? grad
                : momentum * static_cast<math_t>(r_args[2][lane]) +
                    (math_t(1) - dampening) * grad;
            update = nesterov ? grad + momentum * buffer : buffer;
            // Keep the persistent velocity write separate from the
            // Nesterov update.  This ordering is important for the custom
            // half/bfloat16 scalar types: storing the final update through
            // the same temporary used for the parameter otherwise aliases
            // the momentum value in nvcc's low-precision lowering.
            r_args[2][lane] = static_cast<scalar_t>(buffer);
        }
        r_args[0][lane] = static_cast<scalar_t>(param - lr * update);
    }
}

template <typename lr_t, typename scalar_t, typename math_t,
          bool HasMomentum>
__global__ __launch_bounds__(kBlockSize) void sgd_kernel(
        PlainTensorMetadata<HasMomentum ? 3 : 2> metadata,
        const lr_t* tensor_lr, double scalar_lr,
        double momentum_value, double dampening_value,
        double weight_decay_value, int nesterov, int first_momentum_step,
        bool maximize, const float* grad_scale, const float* found_inf) {
    constexpr int Depth = HasMomentum ? 3 : 2;
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    if (found_inf != nullptr && *found_inf == 1.0f) return;
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    const scalar_t* args[Depth];
    scalar_t* outputs[Depth];
    args[0] = static_cast<const scalar_t*>(
        metadata.addresses[0][tensor_index]) + begin;
    args[1] = static_cast<const scalar_t*>(
        metadata.addresses[1][tensor_index]) + begin;
    outputs[0] = const_cast<scalar_t*>(args[0]);
    outputs[1] = const_cast<scalar_t*>(args[1]);
    if constexpr (HasMomentum) {
        args[2] = static_cast<const scalar_t*>(
            metadata.addresses[2][tensor_index]) + begin;
        outputs[2] = const_cast<scalar_t*>(args[2]);
    }

    const math_t lr = tensor_lr != nullptr
        ? static_cast<math_t>(*tensor_lr)
        : static_cast<math_t>(scalar_lr);
    const math_t momentum = static_cast<math_t>(momentum_value);
    const math_t dampening = static_cast<math_t>(dampening_value);
    const math_t weight_decay = static_cast<math_t>(weight_decay_value);
    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t r_args[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(r_args[depth], args[depth], 0, vector_index);
            }
            sgd_math<scalar_t, math_t, HasMomentum>(
                r_args, lr, momentum, dampening, weight_decay,
                nesterov != 0, first_momentum_step != 0, maximize,
                grad_scale);
            load_store(outputs[0], r_args[0], vector_index, 0);
            if (grad_scale != nullptr) {
                load_store(outputs[1], r_args[1], vector_index, 0);
            }
            if constexpr (HasMomentum) {
                load_store(outputs[2], r_args[2], vector_index, 0);
            }
        }
    } else {
        scalar_t r_args[Depth][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
            load_args<Depth>(r_args, args, i_start, kChunkSize, n);
            sgd_math<scalar_t, math_t, HasMomentum>(
                r_args, lr, momentum, dampening, weight_decay,
                nesterov != 0, first_momentum_step != 0, maximize,
                grad_scale);
            store_args<Depth>(outputs, r_args, i_start, kChunkSize, n,
                              grad_scale != nullptr);
        }
    }
}

template <typename M, bool AdamW, bool AMSGrad>
__device__ __forceinline__ void adam_math(
        M& param, M& grad, M& exp_avg, M& exp_avg_sq, M* max_exp_avg_sq,
        M lr, M step_size, M beta1, M beta2, M correction2_sqrt, M eps,
        M weight_decay, bool maximize, const float* grad_scale) {
    if (grad_scale != nullptr) {
        grad = grad / static_cast<M>(*grad_scale);
    }
    M update_grad = maximize ? -grad : grad;
    if constexpr (AdamW) {
        param *= M(1) - lr * weight_decay;
    } else if (weight_decay != M(0)) {
        update_grad += weight_decay * param;
    }
    const M old_exp_avg = exp_avg;
    const M lerp_weight = M(1) - beta1;
    if (std::abs(lerp_weight) < M(0.5)) {
        exp_avg = old_exp_avg + lerp_weight *
            (update_grad - old_exp_avg);
    } else {
        exp_avg = update_grad - (update_grad - old_exp_avg) *
            (M(1) - lerp_weight);
    }
    exp_avg_sq = beta2 * exp_avg_sq +
        (M(1) - beta2) * update_grad * update_grad;
    M second_moment = exp_avg_sq;
    if constexpr (AMSGrad) {
        second_moment = *max_exp_avg_sq < second_moment
            ? second_moment : *max_exp_avg_sq;
        *max_exp_avg_sq = second_moment;
    }
    const M denom = std::sqrt(second_moment) / correction2_sqrt + eps;
    param -= step_size * exp_avg / denom;
}

// the same operation as its explicit fused kernel: every foreach call writes
// a Half/BFloat16 result before the next call consumes it.  Keep those write
// boundaries in one launch so the Python/C++ composition is removed without
// changing its rounding behavior.
template <typename scalar_t, typename math_t, bool AdamW, bool AMSGrad>
__device__ __forceinline__ void adam_exact_math(
        scalar_t& param, const scalar_t& grad_input, scalar_t& exp_avg,
        scalar_t& exp_avg_sq, scalar_t* max_exp_avg_sq,
        math_t step_size, math_t beta2, math_t lerp_weight,
        math_t beta2_weight, math_t correction2_sqrt, math_t eps,
        math_t weight_decay, math_t adamw_factor, bool maximize) {
    math_t param_value = static_cast<math_t>(param);
    scalar_t update_grad_t = grad_input;
    if (maximize) {
        update_grad_t = static_cast<scalar_t>(
            -static_cast<math_t>(grad_input));
    }
    math_t update_grad = static_cast<math_t>(update_grad_t);

    if constexpr (AdamW) {
        // _foreach_mul_(params, 1 - lr * weight_decay): opmath multiply,
        // then a scalar_t writeback before the moments are updated.
        param = static_cast<scalar_t>(param_value * adamw_factor);
        param_value = static_cast<math_t>(param);
    } else if (weight_decay != math_t(0)) {
        // _foreach_add(device_grads, params, alpha=weight_decay) is an
        // out-of-place opmath add followed by a scalar_t writeback.
        update_grad_t = static_cast<scalar_t>(
            update_grad + weight_decay * param_value);
        update_grad = static_cast<math_t>(update_grad_t);
    }

    const math_t old_exp_avg = static_cast<math_t>(exp_avg);
    if (std::abs(lerp_weight) < math_t(0.5)) {
        exp_avg = static_cast<scalar_t>(
            old_exp_avg + lerp_weight * (update_grad - old_exp_avg));
    } else {
        exp_avg = static_cast<scalar_t>(
            update_grad - (update_grad - old_exp_avg) *
                (math_t(1) - lerp_weight));
    }

    // _foreach_mul_ followed by _foreach_addcmul_: two scalar_t writebacks.
    exp_avg_sq = static_cast<scalar_t>(
        static_cast<math_t>(exp_avg_sq) * beta2);
    const math_t old_second = static_cast<math_t>(exp_avg_sq);
    const math_t second = beta2_weight == math_t(1)
        ? std::fma(update_grad, update_grad, old_second)
        : std::fma(beta2_weight, update_grad * update_grad, old_second);
    exp_avg_sq = static_cast<scalar_t>(second);

    scalar_t second_t = exp_avg_sq;
    if constexpr (AMSGrad) {
        const math_t old_max = static_cast<math_t>(*max_exp_avg_sq);
        const math_t new_second = static_cast<math_t>(exp_avg_sq);
        second_t = old_max < new_second ? exp_avg_sq : *max_exp_avg_sq;
        *max_exp_avg_sq = second_t;
    }

    // The remaining unary/binary foreach calls each materialize a low-
    // precision temporary before the final addcdiv.
    const scalar_t sqrt_t = static_cast<scalar_t>(
        std::sqrt(static_cast<math_t>(second_t)));
    const scalar_t divided_t = static_cast<scalar_t>(
        static_cast<math_t>(sqrt_t) / correction2_sqrt);
    const scalar_t denom_t = static_cast<scalar_t>(
        static_cast<math_t>(divided_t) + eps);
    const math_t quotient = static_cast<math_t>(exp_avg) /
        static_cast<math_t>(denom_t);
    const math_t updated_param = step_size == math_t(1)
        ? param_value + quotient
        : std::fma(step_size, quotient, param_value);
    param = static_cast<scalar_t>(updated_param);
}

template <typename lr_t, typename scalar_t, typename math_t, int Depth,
          bool DeviceSteps, bool AdamW, bool AMSGrad>
__global__ __launch_bounds__(kBlockSize) void adam_kernel(
        SimpleTensorMetadata<Depth> metadata,
        const lr_t* tensor_lr, double scalar_lr,
        double beta1_value, double beta2_value, double eps_value,
        double weight_decay_value, bool maximize,
        const float* grad_scale, const float* found_inf) {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    if (found_inf != nullptr && *found_inf == 1.0f) return;
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    const math_t lr = tensor_lr != nullptr
        ? static_cast<math_t>(*tensor_lr)
        : static_cast<math_t>(scalar_lr);
    const math_t beta1 = static_cast<math_t>(beta1_value);
    const math_t beta2 = static_cast<math_t>(beta2_value);
    const math_t eps = static_cast<math_t>(eps_value);
    const math_t weight_decay = static_cast<math_t>(weight_decay_value);
    math_t step_size;
    math_t correction2_sqrt;
    if constexpr (DeviceSteps) {
        const auto* step_ptr = static_cast<const float*>(
            metadata.step_metadata.state_steps[tensor_index]);
        const math_t step = static_cast<math_t>(*step_ptr);
        const math_t correction1 = math_t(1) - ::pow(beta1, step);
        correction2_sqrt = std::sqrt(math_t(1) - ::pow(beta2, step));
        step_size = lr / correction1;
    } else {
        step_size = static_cast<math_t>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        correction2_sqrt = static_cast<math_t>(
            metadata.step_metadata.host.correction2_sqrts[tensor_index]);
    }

    const scalar_t* args[Depth];
    scalar_t* outputs[Depth];
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }
    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t packed[Depth][kILP];
        math_t values[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(packed[depth], args[depth], 0, vector_index);
#pragma unroll
                for (int lane = 0; lane < kILP; ++lane) {
                    values[depth][lane] =
                        static_cast<math_t>(packed[depth][lane]);
                }
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                if constexpr (AMSGrad) {
                    adam_math<math_t, AdamW, true>(
                        values[0][lane], values[1][lane],
                        values[2][lane], values[3][lane],
                        &values[4][lane], lr, step_size, beta1, beta2,
                        correction2_sqrt, eps, weight_decay,
                        maximize, grad_scale);
                } else {
                    adam_math<math_t, AdamW, false>(
                        values[0][lane], values[1][lane],
                        values[2][lane], values[3][lane], nullptr,
                        lr, step_size, beta1, beta2, correction2_sqrt,
                        eps, weight_decay, maximize, grad_scale);
                }
            }
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                if (depth == 1 && grad_scale == nullptr) continue;
#pragma unroll
                for (int lane = 0; lane < kILP; ++lane) {
                    packed[depth][lane] =
                        static_cast<scalar_t>(values[depth][lane]);
                }
                load_store(outputs[depth], packed[depth], vector_index, 0);
            }
        }
    } else {
        math_t values[Depth][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
            load_args<Depth>(values, args, i_start, kChunkSize, n);
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                if constexpr (AMSGrad) {
                    adam_math<math_t, AdamW, true>(
                        values[0][lane], values[1][lane],
                        values[2][lane], values[3][lane],
                        &values[4][lane], lr, step_size, beta1, beta2,
                        correction2_sqrt, eps, weight_decay,
                        maximize, grad_scale);
                } else {
                    adam_math<math_t, AdamW, false>(
                        values[0][lane], values[1][lane],
                        values[2][lane], values[3][lane], nullptr,
                        lr, step_size, beta1, beta2, correction2_sqrt,
                        eps, weight_decay, maximize, grad_scale);
                }
            }
            store_args<Depth>(outputs, values, i_start, kChunkSize, n,
                              grad_scale != nullptr);
        }
    }
}

template <typename scalar_t, typename math_t, int Depth,
          bool AdamW, bool AMSGrad>
__global__ __launch_bounds__(kBlockSize) void adam_exact_kernel(
        SimpleTensorMetadata<Depth> metadata,
        double beta2_value, double lerp_weight_value,
        double beta2_weight_value, double eps_value,
        double weight_decay_value, double adamw_factor_value,
        bool maximize) {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    const math_t beta2 = static_cast<math_t>(beta2_value);
    const math_t lerp_weight = static_cast<math_t>(lerp_weight_value);
    const math_t beta2_weight = static_cast<math_t>(beta2_weight_value);
    const math_t eps = static_cast<math_t>(eps_value);
    const math_t weight_decay = static_cast<math_t>(weight_decay_value);
    const math_t adamw_factor = static_cast<math_t>(adamw_factor_value);
    const math_t step_size = static_cast<math_t>(
        metadata.step_metadata.host.step_sizes[tensor_index]);
    const math_t correction2_sqrt = static_cast<math_t>(
        metadata.step_metadata.host.correction2_sqrts[tensor_index]);

    const scalar_t* args[Depth];
    scalar_t* outputs[Depth];
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }
    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t packed[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(packed[depth], args[depth], 0, vector_index);
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                if constexpr (AMSGrad) {
                    adam_exact_math<scalar_t, math_t, AdamW, true>(
                        packed[0][lane], packed[1][lane], packed[2][lane],
                        packed[3][lane], &packed[4][lane], step_size,
                        beta2, lerp_weight, beta2_weight, correction2_sqrt,
                        eps, weight_decay, adamw_factor, maximize);
                } else {
                    adam_exact_math<scalar_t, math_t, AdamW, false>(
                        packed[0][lane], packed[1][lane], packed[2][lane],
                        packed[3][lane], nullptr, step_size, beta2,
                        lerp_weight, beta2_weight, correction2_sqrt, eps,
                        weight_decay, adamw_factor, maximize);
                }
            }
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                if (depth == 1) continue;
                load_store(outputs[depth], packed[depth], vector_index, 0);
            }
        }
    } else {
        scalar_t values[Depth][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
            load_args<Depth>(values, args, i_start, kChunkSize, n);
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                if constexpr (AMSGrad) {
                    adam_exact_math<scalar_t, math_t, AdamW, true>(
                        values[0][lane], values[1][lane], values[2][lane],
                        values[3][lane], &values[4][lane], step_size, beta2,
                        lerp_weight, beta2_weight, correction2_sqrt, eps,
                        weight_decay, adamw_factor, maximize);
                } else {
                    adam_exact_math<scalar_t, math_t, AdamW, false>(
                        values[0][lane], values[1][lane], values[2][lane],
                        values[3][lane], nullptr, step_size, beta2,
                        lerp_weight, beta2_weight, correction2_sqrt, eps,
                        weight_decay, adamw_factor, maximize);
                }
            }
            store_args<Depth>(outputs, values, i_start, kChunkSize, n, false);
        }
    }
}

template <typename lr_t, typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adagrad_kernel(
        SimpleTensorMetadata<3> metadata,
        const lr_t* tensor_lr, double scalar_lr,
        double lr_decay_value, double weight_decay_value, double eps_value,
        bool maximize, const float* grad_scale, const float* found_inf) {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    if (found_inf != nullptr && *found_inf == 1.0f) return;
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;
    const math_t lr = tensor_lr != nullptr
        ? static_cast<math_t>(*tensor_lr)
        : static_cast<math_t>(scalar_lr);
    const math_t lr_decay = static_cast<math_t>(lr_decay_value);
    const math_t weight_decay = static_cast<math_t>(weight_decay_value);
    const math_t eps = static_cast<math_t>(eps_value);
    const math_t step = static_cast<math_t>(*static_cast<const float*>(
        metadata.step_metadata.state_steps[tensor_index]));
    const math_t clr = lr / (math_t(1) +
        (step - math_t(1)) * lr_decay);

    const scalar_t* args[3];
    scalar_t* outputs[3];
    for (int depth = 0; depth < 3; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }
    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
    for (int depth = 0; depth < 3; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t packed[3][kILP];
        math_t values[3][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
            for (int depth = 0; depth < 3; ++depth) {
                load_store(packed[depth], args[depth], 0, vector_index);
                for (int lane = 0; lane < kILP; ++lane) {
                    values[depth][lane] =
                        static_cast<math_t>(packed[depth][lane]);
                }
            }
            for (int lane = 0; lane < kILP; ++lane) {
                math_t grad = values[1][lane];
                if (grad_scale != nullptr) {
                    grad = grad / static_cast<math_t>(*grad_scale);
                    values[1][lane] = grad;
                }
                if (maximize) grad = -grad;
                if (weight_decay != math_t(0)) {
                    grad += weight_decay * values[0][lane];
                }
                values[2][lane] += grad * grad;
                values[0][lane] -= clr * grad /
                    (std::sqrt(values[2][lane]) + eps);
            }
            for (int depth = 0; depth < 3; ++depth) {
                if (depth == 1 && grad_scale == nullptr) continue;
                for (int lane = 0; lane < kILP; ++lane) {
                    packed[depth][lane] =
                        static_cast<scalar_t>(values[depth][lane]);
                }
                load_store(outputs[depth], packed[depth], vector_index, 0);
            }
        }
    } else {
        math_t values[3][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
            load_args<3>(values, args, i_start, kChunkSize, n);
            for (int lane = 0; lane < kILP; ++lane) {
                math_t grad = values[1][lane];
                if (grad_scale != nullptr) {
                    grad = grad / static_cast<math_t>(*grad_scale);
                    values[1][lane] = grad;
                }
                if (maximize) grad = -grad;
                if (weight_decay != math_t(0)) {
                    grad += weight_decay * values[0][lane];
                }
                values[2][lane] += grad * grad;
                values[0][lane] -= clr * grad /
                    (std::sqrt(values[2][lane]) + eps);
            }
            store_args<3>(outputs, values, i_start, kChunkSize, n,
                          grad_scale != nullptr);
        }
    }
}

// RMSprop is a common CUDA path without a fused kernel.
// Keep the metadata depth exact for each option combination instead of
// carrying two dummy lists and branching on runtime flags for every element.
// the generic Body/should_load indirection in the hot loop.
template <typename scalar_t, typename math_t, int Depth,
          bool Centered, bool HasMomentum>
__device__ __forceinline__ void rmsprop_math(
        scalar_t r_args[Depth][kILP], math_t lr, math_t alpha, math_t eps,
        math_t weight_decay, math_t momentum, bool maximize) {
#pragma unroll
    for (int lane = 0; lane < kILP; ++lane) {
        math_t g = static_cast<math_t>(r_args[1][lane]);
        if (maximize) g = -g;
        const math_t p = static_cast<math_t>(r_args[0][lane]);
        if (weight_decay != math_t(0)) g += weight_decay * p;

        const math_t square_avg = alpha *
                static_cast<math_t>(r_args[2][lane]) +
            (math_t(1) - alpha) * g * g;
        r_args[2][lane] = static_cast<scalar_t>(square_avg);
        math_t average = square_avg;
        if constexpr (Centered) {
            constexpr int kGradAvgIndex = 3;
            const math_t mean = alpha *
                    static_cast<math_t>(r_args[kGradAvgIndex][lane]) +
                (math_t(1) - alpha) * g;
            r_args[kGradAvgIndex][lane] = static_cast<scalar_t>(mean);
            average = square_avg - mean * mean;
        }

        math_t update = g / (std::sqrt(average) + eps);
        if constexpr (HasMomentum) {
            constexpr int kMomentumIndex = Centered ? 4 : 3;
            math_t buffer = momentum *
                    static_cast<math_t>(r_args[kMomentumIndex][lane]) +
                update;
            r_args[kMomentumIndex][lane] = static_cast<scalar_t>(buffer);
            update = buffer;
        }
        r_args[0][lane] = static_cast<scalar_t>(p - lr * update);
    }
}

template <typename scalar_t, typename math_t, int Depth,
          bool Centered, bool HasMomentum>
__global__ __launch_bounds__(kBlockSize) void rmsprop_kernel(
        PlainTensorMetadata<Depth> metadata,
        math_t lr, math_t alpha, math_t eps, math_t weight_decay,
        math_t momentum, bool maximize) {
    constexpr int kGradAvgIndex = Centered ? 3 : -1;
    constexpr int kMomentumIndex = HasMomentum ? (Centered ? 4 : 3) : -1;
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    const scalar_t* args[Depth];
    scalar_t* outputs[Depth];
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }

    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t r_args[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                load_store(r_args[depth], args[depth], 0, vector_index);
            }
            rmsprop_math<scalar_t, math_t, Depth, Centered, HasMomentum>(
                r_args, lr, alpha, eps, weight_decay, momentum, maximize);
            load_store(outputs[0], r_args[0], vector_index, 0);
            load_store(outputs[2], r_args[2], vector_index, 0);
            if constexpr (Centered) {
                load_store(outputs[kGradAvgIndex], r_args[kGradAvgIndex],
                           vector_index, 0);
            }
            if constexpr (HasMomentum) {
                load_store(outputs[kMomentumIndex], r_args[kMomentumIndex],
                           vector_index, 0);
            }
        }
    } else {
        scalar_t r_args[Depth][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t i = i_start + threadIdx.x +
                    static_cast<int64_t>(lane) * blockDim.x;
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    r_args[depth][lane] = (i < n && i < kChunkSize)
                        ? args[depth][i] : scalar_t(0);
                }
            }
            rmsprop_math<scalar_t, math_t, Depth, Centered, HasMomentum>(
                r_args, lr, alpha, eps, weight_decay, momentum, maximize);
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t i = i_start + threadIdx.x +
                    static_cast<int64_t>(lane) * blockDim.x;
                if (i >= n || i >= kChunkSize) continue;
                outputs[0][i] = r_args[0][lane];
                outputs[2][i] = r_args[2][lane];
                if constexpr (Centered) {
                    outputs[kGradAvgIndex][i] = r_args[kGradAvgIndex][lane];
                }
                if constexpr (HasMomentum) {
                    outputs[kMomentumIndex][i] = r_args[kMomentumIndex][lane];
                }
            }
        }
    }
}

template <typename M>
struct AdadeltaBody {
    M lr, rho, eps, weight_decay;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata&, int) {}
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) g += weight_decay * p;
        const M one_minus_rho = M(1) - rho;
        M square = rho * values[2][lane] + one_minus_rho * g * g;
        values[2][lane] = square;
        const M std = std::sqrt(square + eps);
        M delta = std::sqrt(values[3][lane] + eps) / std * g;
        values[3][lane] = rho * values[3][lane] +
            one_minus_rho * delta * delta;
        values[0][lane] = p - lr * delta;
    }
};

template <typename M>
struct AdagradHostBody {
    M lr, lr_decay, eps, weight_decay;
    bool maximize;
    M corrected_lr;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        const M step = static_cast<M>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        corrected_lr = lr / (M(1) + (step - M(1)) * lr_decay);
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) g += weight_decay * p;
        values[2][lane] += g * g;
        values[0][lane] = p - corrected_lr * g /
            (std::sqrt(values[2][lane]) + eps);
    }
};

// generic pointwise kernel is useful for the less regular optimizers, but its
// per-element should_load/should_store dispatch leaves avoidable branches in
// this three-list hot path.  Keep the host step correction in metadata and
// specialize the load/update/store loop instead.
template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adagrad_host_kernel(
        SimpleTensorMetadata<3> metadata, double lr_value,
        double lr_decay_value, double weight_decay_value, double eps_value,
        bool maximize) {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    const math_t lr = static_cast<math_t>(lr_value);
    const math_t lr_decay = static_cast<math_t>(lr_decay_value);
    const math_t weight_decay = static_cast<math_t>(weight_decay_value);
    const math_t eps = static_cast<math_t>(eps_value);
    const math_t step = static_cast<math_t>(
        metadata.step_metadata.host.step_sizes[tensor_index]);
    const math_t corrected_lr = lr / (math_t(1) +
        (step - math_t(1)) * lr_decay);

    const scalar_t* args[3];
    scalar_t* outputs[3];
#pragma unroll
    for (int depth = 0; depth < 3; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }

    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < 3; ++depth) {
        aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t packed[3][kILP];
        math_t values[3][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n &&
                 vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < 3; ++depth) {
                load_store(packed[depth], args[depth], 0, vector_index);
#pragma unroll
                for (int lane = 0; lane < kILP; ++lane) {
                    values[depth][lane] =
                        static_cast<math_t>(packed[depth][lane]);
                }
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                math_t grad = values[1][lane];
                if (maximize) grad = -grad;
                if (weight_decay != math_t(0)) {
                    grad += weight_decay * values[0][lane];
                }
                values[2][lane] += grad * grad;
                values[0][lane] -= corrected_lr * grad /
                    (std::sqrt(values[2][lane]) + eps);
            }
            for (int depth = 0; depth < 3; depth += 2) {
#pragma unroll
                for (int lane = 0; lane < kILP; ++lane) {
                    packed[depth][lane] =
                        static_cast<scalar_t>(values[depth][lane]);
                }
                load_store(outputs[depth], packed[depth], vector_index, 0);
            }
        }
    } else {
        math_t values[3][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
            load_args<3>(values, args, i_start, kChunkSize, n);
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                math_t grad = values[1][lane];
                if (maximize) grad = -grad;
                if (weight_decay != math_t(0)) {
                    grad += weight_decay * values[0][lane];
                }
                values[2][lane] += grad * grad;
                values[0][lane] -= corrected_lr * grad /
                    (std::sqrt(values[2][lane]) + eps);
            }
            store_args<3>(outputs, values, i_start, kChunkSize, n, false);
        }
    }
}

template <typename M>
struct AdamaxBody {
    M lr, beta1, beta2, eps, weight_decay;
    bool maximize;
    M step, step_size;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        step = static_cast<M>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        step_size = -lr / (M(1) - ::pow(beta1, step));
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) g += weight_decay * p;
        values[2][lane] = values[2][lane] +
            (M(1) - beta1) * (g - values[2][lane]);
        values[3][lane] = ::fmax(beta2 * values[3][lane],
                                std::abs(g) + eps);
        values[0][lane] = p + step_size * values[2][lane] /
            values[3][lane];
    }
};

template <typename M, typename S>
struct AsgdBody {
    M lr, lambd, weight_decay;
    M eta, mu;
    bool maximize;

    template <int Depth, typename IgnoredScalar, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        eta = static_cast<M>(*static_cast<const S*>(
            metadata.addresses[4][tensor_index]));
        mu = static_cast<M>(*static_cast<const S*>(
            metadata.addresses[3][tensor_index]));
    }
    __device__ __forceinline__ bool should_load(int depth) const {
        return depth < 3;
    }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth == 0 || depth == 2;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) g += weight_decay * p;
        p = p * (M(1) - lambd * eta) - eta * g;
        values[0][lane] = p;
        M average = values[2][lane];
        if (mu == M(1)) average = p;
        else average += (p - average) * mu;
        values[2][lane] = average;
    }
};

template <typename M>
struct RpropBody {
    M step_size_min, step_size_max, etaminus, etaplus;
    bool maximize;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata&, int) {}
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        const M product = g * values[2][lane];
        M sign = product > M(0) ? etaplus :
            (product < M(0) ? etaminus : M(1));
        M step_size = values[3][lane] * sign;
        step_size = ::fmin(step_size_max, ::fmax(step_size_min, step_size));
        values[3][lane] = step_size;
        if (sign == etaminus) g = M(0);
        const M grad_sign = g > M(0) ? M(1) :
            (g < M(0) ? M(-1) : M(0));
        values[0][lane] -= grad_sign * step_size;
        values[2][lane] = g;
    }
};

template <typename M>
struct NadamBody {
    M lr, beta1, beta2, eps, momentum_decay, weight_decay;
    bool maximize, decoupled_weight_decay;
    M step, mu_product, mu, mu_next, grad_coefficient, expavg_coefficient,
        correction2_sqrt;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        step = static_cast<M>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        mu_product = static_cast<M>(
            metadata.step_metadata.host.correction2_sqrts[tensor_index]);
        mu = beta1 * (M(1) - M(0.5) * ::pow(M(0.96),
                                          step * momentum_decay));
        mu_next = beta1 * (M(1) - M(0.5) * ::pow(M(0.96),
                                               (step + M(1)) * momentum_decay));
        grad_coefficient = -lr * (M(1) - mu) /
            (M(1) - mu_product);
        expavg_coefficient = -lr * mu_next /
            (M(1) - mu_product * mu_next);
        correction2_sqrt = std::sqrt(M(1) - ::pow(beta2, step));
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) {
            if (decoupled_weight_decay) p *= M(1) - lr * weight_decay;
            else g += weight_decay * p;
        }
        values[2][lane] = values[2][lane] +
            (M(1) - beta1) * (g - values[2][lane]);
        values[3][lane] = beta2 * values[3][lane] +
            (M(1) - beta2) * g * g;
        const M denom = std::sqrt(values[3][lane]) / correction2_sqrt + eps;
        values[0][lane] = p +
            (grad_coefficient * g + expavg_coefficient * values[2][lane]) /
            denom;
    }
};

template <typename M>
struct RadamBody {
    M lr, beta1, beta2, eps, weight_decay;
    bool maximize, decoupled_weight_decay;
    M step, unrectified_coefficient, rectified_coefficient;

    template <int Depth, typename S, typename Metadata>
    __device__ __forceinline__ void prepare(
            const Metadata& metadata, int tensor_index) {
        step = static_cast<M>(
            metadata.step_metadata.host.step_sizes[tensor_index]);
        const M bc1 = M(1) - ::pow(beta1, step);
        const M bc2 = M(1) - ::pow(beta2, step);
        const M rho_inf = M(2) / (M(1) - beta2) - M(1);
        const M rho_t = rho_inf - M(2) * step * ::pow(beta2, step) / bc2;
        unrectified_coefficient = -lr / bc1;
        rectified_coefficient = M(0);
        if (rho_t > M(5)) {
            const M rect = std::sqrt((rho_t - M(4)) * (rho_t - M(2)) * rho_inf /
                ((rho_inf - M(4)) * (rho_inf - M(2)) * rho_t));
            rectified_coefficient = -lr * std::sqrt(bc2) * rect / bc1;
        }
    }
    __device__ __forceinline__ bool should_load(int) const { return true; }
    __device__ __forceinline__ bool should_store(int depth) const {
        return depth != 1;
    }
    __device__ __forceinline__ void operator()(
            M values[][kILP], int lane) const {
        M g = maximize ? -values[1][lane] : values[1][lane];
        M p = values[0][lane];
        if (weight_decay != M(0)) {
            if (decoupled_weight_decay) p *= M(1) - lr * weight_decay;
            else g += weight_decay * p;
        }
        values[2][lane] = values[2][lane] +
            (M(1) - beta1) * (g - values[2][lane]);
        values[3][lane] = beta2 * values[3][lane] +
            (M(1) - beta2) * g * g;
        if (rectified_coefficient != M(0)) {
            values[0][lane] = p + rectified_coefficient * values[2][lane] /
                (std::sqrt(values[3][lane]) + eps);
        } else {
            // RAdam is deliberately unrectified during the short warm-up:
            // this branch uses the bias-corrected first moment directly and
            // must not divide by the second-moment estimate.
            values[0][lane] = p + unrectified_coefficient * values[2][lane];
        }
    }
};

template <int Depth, typename scalar_t, typename math_t, typename Body,
          typename Metadata>
__global__ __launch_bounds__(kBlockSize) void pointwise_optimizer_kernel(
        Metadata metadata, Body body) {
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = metadata.numel_for_tensor[tensor_index] - begin;
    if (n <= 0) return;

    Body op = body;
    op.template prepare<Depth, scalar_t>(metadata, tensor_index);
    const scalar_t* args[Depth];
    scalar_t* outputs[Depth];
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        args[depth] = static_cast<const scalar_t*>(
            metadata.addresses[depth][tensor_index]) + begin;
        outputs[depth] = const_cast<scalar_t*>(args[depth]);
    }

    bool aligned = (n % kILP == 0) && (kChunkSize % kILP == 0);
#pragma unroll
    for (int depth = 0; depth < Depth; ++depth) {
        if (op.should_load(depth)) aligned = aligned && is_aligned(args[depth]);
    }

    if (aligned) {
        alignas(kILP * sizeof(scalar_t)) scalar_t packed[Depth][kILP];
        math_t values[Depth][kILP];
        for (int64_t vector_index = static_cast<int64_t>(threadIdx.x);
             vector_index * kILP < n && vector_index * kILP < kChunkSize;
             vector_index += static_cast<int64_t>(blockDim.x)) {
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                if (op.should_load(depth)) {
                    load_store(packed[depth], args[depth], 0, vector_index);
#pragma unroll
                    for (int lane = 0; lane < kILP; ++lane) {
                        values[depth][lane] =
                            static_cast<math_t>(packed[depth][lane]);
                    }
                } else {
#pragma unroll
                    for (int lane = 0; lane < kILP; ++lane) {
                        values[depth][lane] = math_t(0);
                    }
                }
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) op(values, lane);
#pragma unroll
            for (int depth = 0; depth < Depth; ++depth) {
                if (!op.should_store(depth)) continue;
#pragma unroll
                for (int lane = 0; lane < kILP; ++lane) {
                    packed[depth][lane] =
                        static_cast<scalar_t>(values[depth][lane]);
                }
                load_store(outputs[depth], packed[depth], vector_index, 0);
            }
        }
    } else {
        math_t values[Depth][kILP];
        for (int64_t i_start = 0; i_start < n && i_start < kChunkSize;
             i_start += static_cast<int64_t>(blockDim.x) * kILP) {
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t i = i_start + threadIdx.x +
                    static_cast<int64_t>(lane) * blockDim.x;
#pragma unroll
                for (int depth = 0; depth < Depth; ++depth) {
                    values[depth][lane] = math_t(0);
                    if (op.should_load(depth) && i < n && i < kChunkSize) {
                        values[depth][lane] =
                            static_cast<math_t>(args[depth][i]);
                    }
                }
            }
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) op(values, lane);
#pragma unroll
            for (int lane = 0; lane < kILP; ++lane) {
                const int64_t i = i_start + threadIdx.x +
                    static_cast<int64_t>(lane) * blockDim.x;
                if (i < n && i < kChunkSize) {
#pragma unroll
                    for (int depth = 0; depth < Depth; ++depth) {
                        if (op.should_store(depth)) {
                            outputs[depth][i] =
                                static_cast<scalar_t>(values[depth][lane]);
                        }
                    }
                }
            }
        }
    }
}

// Adafactor needs two per-tensor reductions (parameter RMS and update RMS)
// in addition to the elementwise variance update.  The vector form keeps the
// reductions in one MTA launch and applies the final clipped update in a
// second pass.  This removes the host-side .item() synchronizations from the
// Python reference implementation while retaining one metadata list for all
// tensors in the group.
template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_vector_stats_kernel(
        TensorMetadata<3> metadata, double beta2_decay_value,
        double eps1_value, bool maximize, math_t* stats) {
    const int slot = metadata.block_to_tensor[blockIdx.x];
    const int global_index = metadata.tensor_indices[slot];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t total_n = metadata.numel_for_tensor[slot];
    const int64_t remaining = total_n - begin;
    const int64_t n = remaining < kChunkSize ? remaining : kChunkSize;
    if (n <= 0) return;

    const math_t beta2_weight = ::pow(
        static_cast<math_t>(metadata.step_metadata.host.step_sizes[slot]),
        static_cast<math_t>(beta2_decay_value));
    const math_t old_weight = math_t(1) - beta2_weight;
    const math_t eps1 = static_cast<math_t>(eps1_value);
    const math_t eps1_sq = eps1 * eps1;
    const scalar_t* param = static_cast<const scalar_t*>(
        metadata.addresses[0][slot]) + begin;
    const scalar_t* grad = static_cast<const scalar_t*>(
        metadata.addresses[1][slot]) + begin;
    scalar_t* variance = const_cast<scalar_t*>(static_cast<const scalar_t*>(
        metadata.addresses[2][slot])) + begin;

    math_t param_sum = math_t(0);
    math_t update_sum = math_t(0);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const math_t p = static_cast<math_t>(param[i]);
        math_t g = static_cast<math_t>(grad[i]);
        if (maximize) g = -g;
        const math_t old_variance = static_cast<math_t>(variance[i]);
        const math_t next_variance = old_weight * old_variance +
            beta2_weight * g * g;
        variance[i] = static_cast<scalar_t>(next_variance);
        const math_t update = g / std::sqrt(::fmax(next_variance, eps1_sq));
        param_sum += p * p;
        update_sum += update * update;
    }

    __shared__ math_t param_sums[kBlockSize];
    __shared__ math_t update_sums[kBlockSize];
    param_sums[threadIdx.x] = param_sum;
    update_sums[threadIdx.x] = update_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            param_sums[threadIdx.x] += param_sums[threadIdx.x + stride];
            update_sums[threadIdx.x] += update_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(stats + 2 * global_index, param_sums[0]);
        atomicAdd(stats + 2 * global_index + 1, update_sums[0]);
    }
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_vector_apply_kernel(
        TensorMetadata<3> metadata, double lr_value,
        double beta2_decay_value, double eps1_value, double eps2_value,
        double d_value, double weight_decay_value, bool maximize,
        const math_t* stats) {
    const int slot = metadata.block_to_tensor[blockIdx.x];
    const int global_index = metadata.tensor_indices[slot];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t total_n = metadata.numel_for_tensor[slot];
    const int64_t remaining = total_n - begin;
    const int64_t n = remaining < kChunkSize ? remaining : kChunkSize;
    if (n <= 0) return;

    const math_t lr = static_cast<math_t>(lr_value);
    const math_t beta2_weight = ::pow(
        static_cast<math_t>(metadata.step_metadata.host.step_sizes[slot]),
        static_cast<math_t>(beta2_decay_value));
    const math_t eps1 = static_cast<math_t>(eps1_value);
    const math_t eps1_sq = eps1 * eps1;
    const math_t rho = ::fmin(lr, math_t(1) / std::sqrt(
        static_cast<math_t>(metadata.step_metadata.host.step_sizes[slot])));
    const math_t rms_param = std::sqrt(stats[2 * global_index] /
        static_cast<math_t>(total_n));
    const math_t alpha = ::fmax(static_cast<math_t>(eps2_value), rms_param) * rho;
    const math_t rms_update = std::sqrt(stats[2 * global_index + 1] /
        static_cast<math_t>(total_n));
    const math_t clip = ::fmax(math_t(1), rms_update /
        static_cast<math_t>(d_value));
    const math_t update_scale = alpha / clip;
    const math_t param_scale = math_t(1) - lr *
        static_cast<math_t>(weight_decay_value);
    const scalar_t* grad = static_cast<const scalar_t*>(
        metadata.addresses[1][slot]) + begin;
    scalar_t* param = const_cast<scalar_t*>(static_cast<const scalar_t*>(
        metadata.addresses[0][slot])) + begin;
    const scalar_t* variance = static_cast<const scalar_t*>(
        metadata.addresses[2][slot]) + begin;

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        math_t g = static_cast<math_t>(grad[i]);
        if (maximize) g = -g;
        const math_t v = static_cast<math_t>(variance[i]);
        const math_t update = g / std::sqrt(::fmax(v, eps1_sq));
        math_t p = static_cast<math_t>(param[i]);
        if (weight_decay_value != 0.0) p *= param_scale;
        param[i] = static_cast<scalar_t>(p - update_scale * update);
    }
}

// The factored form is restricted by the Python route to ordinary 2-D
// tensors.  Each row/column reduction is therefore independent and can be
// launched without materializing row_mean/col_mean intermediates.  The row
// reduction also accumulates the row-state mean used by the normalized
// Kronecker estimate.
template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_factored_row_kernel(
        const scalar_t* grad, scalar_t* row_var, int64_t rows,
        int64_t cols, double beta2_weight_value, math_t* stats,
        int global_index) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || cols <= 0) return;
    math_t sum = math_t(0);
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const math_t g = static_cast<math_t>(grad[row * cols + col]);
        sum += g * g;
    }
    __shared__ math_t sums[kBlockSize];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const math_t weight = static_cast<math_t>(beta2_weight_value);
        const math_t old = static_cast<math_t>(row_var[row]);
        const math_t mean = sums[0] / static_cast<math_t>(cols);
        const math_t next = old + weight * (mean - old);
        row_var[row] = static_cast<scalar_t>(next);
        atomicAdd(stats + 3 * global_index + 2, next);
    }
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_factored_col_kernel(
        const scalar_t* grad, scalar_t* col_var, int64_t rows,
        int64_t cols, double beta2_decay_value) {
    const int64_t col = static_cast<int64_t>(blockIdx.x);
    if (col >= cols || rows <= 0) return;
    math_t sum = math_t(0);
    for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        const math_t g = static_cast<math_t>(grad[row * cols + col]);
        sum += g * g;
    }
    __shared__ math_t sums[kBlockSize];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const math_t weight = static_cast<math_t>(beta2_decay_value);
        const math_t old = static_cast<math_t>(col_var[col]);
        const math_t mean = sums[0] / static_cast<math_t>(rows);
        col_var[col] = static_cast<scalar_t>(old + weight * (mean - old));
    }
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_factored_stats_kernel(
        TensorMetadata<4> metadata, double eps1_value, bool maximize,
        math_t* stats) {
    const int slot = metadata.block_to_tensor[blockIdx.x];
    const int global_index = metadata.tensor_indices[slot];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t total_n = metadata.numel_for_tensor[slot];
    const int64_t remaining = total_n - begin;
    const int64_t n = remaining < kChunkSize ? remaining : kChunkSize;
    if (n <= 0) return;
    const int64_t rows = metadata.dim_minus2[slot];
    const int64_t cols = metadata.dim_minus1[slot];
    if (rows <= 0 || cols <= 0) return;
    const int64_t matrix_n = rows * cols;
    const math_t eps1 = static_cast<math_t>(eps1_value);
    const math_t eps1_sq = eps1 * eps1;
    const math_t row_mean = stats[3 * global_index + 2] /
        static_cast<math_t>(rows);
    const scalar_t* param = static_cast<const scalar_t*>(
        metadata.addresses[0][slot]) + begin;
    const scalar_t* grad = static_cast<const scalar_t*>(
        metadata.addresses[1][slot]) + begin;
    const scalar_t* row_var = static_cast<const scalar_t*>(
        metadata.addresses[2][slot]);
    const scalar_t* col_var = static_cast<const scalar_t*>(
        metadata.addresses[3][slot]);
    math_t param_sum = math_t(0);
    math_t update_sum = math_t(0);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t absolute = begin + i;
        const int64_t row = (absolute % matrix_n) / cols;
        const int64_t col = absolute % cols;
        const math_t g0 = static_cast<math_t>(grad[i]);
        const math_t g = maximize ? -g0 : g0;
        const math_t v = static_cast<math_t>(row_var[row]) *
            static_cast<math_t>(col_var[col]) / ::fmax(row_mean, eps1);
        const math_t update = g / std::sqrt(::fmax(v, eps1_sq));
        const math_t p = static_cast<math_t>(param[i]);
        param_sum += p * p;
        update_sum += update * update;
    }
    __shared__ math_t param_sums[kBlockSize];
    __shared__ math_t update_sums[kBlockSize];
    param_sums[threadIdx.x] = param_sum;
    update_sums[threadIdx.x] = update_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            param_sums[threadIdx.x] += param_sums[threadIdx.x + stride];
            update_sums[threadIdx.x] += update_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(stats + 3 * global_index, param_sums[0]);
        atomicAdd(stats + 3 * global_index + 1, update_sums[0]);
    }
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void adafactor_factored_apply_kernel(
        TensorMetadata<4> metadata, double lr_value, double eps1_value,
        double eps2_value, double d_value, double weight_decay_value,
        bool maximize, const math_t* stats) {
    const int slot = metadata.block_to_tensor[blockIdx.x];
    const int global_index = metadata.tensor_indices[slot];
    const int64_t begin = static_cast<int64_t>(
        metadata.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t total_n = metadata.numel_for_tensor[slot];
    const int64_t remaining = total_n - begin;
    const int64_t n = remaining < kChunkSize ? remaining : kChunkSize;
    if (n <= 0) return;
    const int64_t rows = metadata.dim_minus2[slot];
    const int64_t cols = metadata.dim_minus1[slot];
    const int64_t matrix_n = rows * cols;
    const math_t lr = static_cast<math_t>(lr_value);
    const math_t step = static_cast<math_t>(
        metadata.step_metadata.host.step_sizes[slot]);
    const math_t rho = ::fmin(lr, math_t(1) / std::sqrt(step));
    const math_t rms_param = std::sqrt(stats[3 * global_index] /
        static_cast<math_t>(total_n));
    const math_t alpha = ::fmax(static_cast<math_t>(eps2_value), rms_param) * rho;
    const math_t rms_update = std::sqrt(stats[3 * global_index + 1] /
        static_cast<math_t>(total_n));
    const math_t clip = ::fmax(math_t(1), rms_update /
        static_cast<math_t>(d_value));
    const math_t scale = alpha / clip;
    const math_t eps1 = static_cast<math_t>(eps1_value);
    const math_t eps1_sq = eps1 * eps1;
    const math_t param_scale = math_t(1) - lr *
        static_cast<math_t>(weight_decay_value);
    const scalar_t* grad = static_cast<const scalar_t*>(
        metadata.addresses[1][slot]) + begin;
    scalar_t* param = const_cast<scalar_t*>(static_cast<const scalar_t*>(
        metadata.addresses[0][slot])) + begin;
    const scalar_t* row_var = static_cast<const scalar_t*>(
        metadata.addresses[2][slot]);
    const scalar_t* col_var = static_cast<const scalar_t*>(
        metadata.addresses[3][slot]);
    const math_t row_mean = stats[3 * global_index + 2] /
        static_cast<math_t>(rows);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t absolute = begin + i;
        const int64_t row = (absolute % matrix_n) / cols;
        const int64_t col = absolute % cols;
        const math_t g0 = static_cast<math_t>(grad[i]);
        const math_t g = maximize ? -g0 : g0;
        const math_t v = static_cast<math_t>(row_var[row]) *
            static_cast<math_t>(col_var[col]) / ::fmax(row_mean, eps1);
        const math_t update = g / std::sqrt(::fmax(v, eps1_sq));
        math_t p = static_cast<math_t>(param[i]);
        if (weight_decay_value != 0.0) p *= param_scale;
        param[i] = static_cast<scalar_t>(p - scale * update);
    }
}

template <typename scalar_t, typename math_t>
void launch_adafactor_vector(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& variances,
        const std::vector<double>& steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize, const char* op_name) {
    const DType stats_dtype = std::is_same<math_t, double>::value
        ? DType::Float64 : DType::Float32;
    Tensor stats = Tensor::zeros(
        {static_cast<int64_t>(params.size() * 2)}, stats_dtype,
        params[0].device());
    std::array<const std::vector<Tensor>*, 3> lists{
        &params, &grads, &variances};
    launch_batches<3>(lists, nullptr, &steps, nullptr,
        [&](const TensorMetadata<3>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adafactor_vector_stats_kernel<scalar_t, math_t><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, beta2_decay, eps1, maximize,
                stats.data_ptr<math_t>());
            checkCuda(cudaGetLastError(), op_name);
            adafactor_vector_apply_kernel<scalar_t, math_t><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, lr, beta2_decay, eps1, eps2, d, weight_decay,
                maximize, stats.data_ptr<math_t>());
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adafactor_factored(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& row_vars,
        const std::vector<Tensor>& col_vars,
        const std::vector<double>& steps, double lr, double beta2_decay,
        double eps1, double eps2, double d, double weight_decay,
        bool maximize, const char* op_name) {
    const DType stats_dtype = std::is_same<math_t, double>::value
        ? DType::Float64 : DType::Float32;
    Tensor stats = Tensor::zeros(
        {static_cast<int64_t>(params.size() * 3)}, stats_dtype,
        params[0].device());
    for (size_t i = 0; i < params.size(); ++i) {
        const int64_t rows = params[i].size(-2);
        const int64_t cols = params[i].size(-1);
        adafactor_factored_row_kernel<scalar_t, math_t><<<
            static_cast<unsigned int>(rows), kBlockSize, 0,
            getCurrentCUDAStream().stream()>>>(
            grads[i].data_ptr<scalar_t>(), row_vars[i].data_ptr<scalar_t>(),
            rows, cols, std::pow(steps[i], beta2_decay),
            stats.data_ptr<math_t>(), static_cast<int>(i));
        checkCuda(cudaGetLastError(), op_name);
        adafactor_factored_col_kernel<scalar_t, math_t><<<
            static_cast<unsigned int>(cols), kBlockSize, 0,
            getCurrentCUDAStream().stream()>>>(
            grads[i].data_ptr<scalar_t>(), col_vars[i].data_ptr<scalar_t>(),
            rows, cols, std::pow(steps[i], beta2_decay));
        checkCuda(cudaGetLastError(), op_name);
    }
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &row_vars, &col_vars};
    launch_batches<4>(lists, nullptr, &steps, nullptr,
        [&](const TensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adafactor_factored_stats_kernel<scalar_t, math_t><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, eps1, maximize, stats.data_ptr<math_t>());
            checkCuda(cudaGetLastError(), op_name);
            adafactor_factored_apply_kernel<scalar_t, math_t><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, lr, eps1, eps2, d, weight_decay, maximize,
                stats.data_ptr<math_t>());
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void asgd_increment_device_steps_kernel(
        SimpleTensorMetadata<5> metadata) {
    if (threadIdx.x != 0 || metadata.block_to_chunk[blockIdx.x] != 0) return;
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    float* step = static_cast<float*>(const_cast<void*>(
        metadata.step_metadata.state_steps[tensor_index]));
    *step += 1.0f;
}

template <typename scalar_t, typename math_t>
__global__ __launch_bounds__(kBlockSize) void asgd_update_scalars_kernel(
        SimpleTensorMetadata<5> metadata, double lr_value, double lambd_value,
        double t0_value, double alpha_value, bool device_steps) {
    if (threadIdx.x != 0 || metadata.block_to_chunk[blockIdx.x] != 0) return;
    const int tensor_index = metadata.block_to_tensor[blockIdx.x];
    const double step = device_steps
        ? static_cast<double>(*static_cast<const float*>(
              metadata.step_metadata.state_steps[tensor_index]))
        : metadata.step_metadata.host.step_sizes[tensor_index];
    const math_t lr = static_cast<math_t>(lr_value);
    const math_t lambd = static_cast<math_t>(lambd_value);
    const math_t t0 = static_cast<math_t>(t0_value);
    const math_t alpha = static_cast<math_t>(alpha_value);
    const math_t eta = lr / ::pow(math_t(1) + lambd * lr *
        static_cast<math_t>(step), alpha);
    const math_t mu = math_t(1) / ::fmax(math_t(1),
        static_cast<math_t>(step) - t0);
    *static_cast<scalar_t*>(const_cast<void*>(
        metadata.addresses[4][tensor_index])) = static_cast<scalar_t>(eta);
    *static_cast<scalar_t*>(const_cast<void*>(
        metadata.addresses[3][tensor_index])) = static_cast<scalar_t>(mu);
}

template <typename scalar_t, typename math_t, bool Centered, bool HasMomentum>
void launch_rmsprop_variant(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& grad_avgs,
        const std::vector<Tensor>& momentum_buffers,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool maximize, const char* op_name) {
    constexpr int Depth = 3 + (Centered ? 1 : 0) + (HasMomentum ? 1 : 0);
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    lists[2] = &square_avgs;
    int slot = 3;
    if constexpr (Centered) lists[slot++] = &grad_avgs;
    if constexpr (HasMomentum) lists[slot++] = &momentum_buffers;
    launch_plain_batches<Depth>(lists,
        [&](const PlainTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            rmsprop_kernel<scalar_t, math_t, Depth, Centered, HasMomentum>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, static_cast<math_t>(lr), static_cast<math_t>(alpha),
                static_cast<math_t>(eps), static_cast<math_t>(weight_decay),
                static_cast<math_t>(momentum), maximize);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_rmsprop(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& grad_avgs,
        const std::vector<Tensor>& momentum_buffers,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool centered, bool maximize, const char* op_name) {
    if (centered) {
        if (momentum != 0.0) {
            launch_rmsprop_variant<scalar_t, math_t, true, true>(
                params, grads, square_avgs, grad_avgs, momentum_buffers,
                lr, alpha, eps, weight_decay, momentum, maximize,
                op_name);
        } else {
            launch_rmsprop_variant<scalar_t, math_t, true, false>(
                params, grads, square_avgs, grad_avgs, momentum_buffers,
                lr, alpha, eps, weight_decay, momentum, maximize,
                op_name);
        }
    } else if (momentum != 0.0) {
        launch_rmsprop_variant<scalar_t, math_t, false, true>(
            params, grads, square_avgs, grad_avgs, momentum_buffers,
            lr, alpha, eps, weight_decay, momentum, maximize,
            op_name);
    } else {
        launch_rmsprop_variant<scalar_t, math_t, false, false>(
            params, grads, square_avgs, grad_avgs, momentum_buffers,
            lr, alpha, eps, weight_decay, momentum, maximize,
            op_name);
    }
}

template <typename scalar_t, typename math_t, bool Centered, bool HasMomentum>
void launch_rmsprop_exact_variant(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& grad_avgs,
        const std::vector<Tensor>& momentum_buffers,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool maximize, const char* op_name) {
    constexpr int Depth = 3 + (Centered ? 1 : 0) + (HasMomentum ? 1 : 0);
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    lists[2] = &square_avgs;
    int slot = 3;
    if constexpr (Centered) lists[slot++] = &grad_avgs;
    if constexpr (HasMomentum) lists[slot++] = &momentum_buffers;
    RmspropExactBody<scalar_t, math_t, Centered, HasMomentum> body{
        static_cast<math_t>(lr), static_cast<math_t>(alpha),
        static_cast<math_t>(1.0 - alpha), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), static_cast<math_t>(momentum),
        maximize};
    launch_plain_batches<Depth>(lists,
        [&](const PlainTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                Depth, scalar_t, math_t,
                RmspropExactBody<scalar_t, math_t, Centered, HasMomentum>,
                PlainTensorMetadata<Depth>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_rmsprop_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& grad_avgs,
        const std::vector<Tensor>& momentum_buffers,
        double lr, double alpha, double eps, double weight_decay,
        double momentum, bool centered, bool maximize, const char* op_name) {
    if (centered) {
        if (momentum != 0.0) {
            launch_rmsprop_exact_variant<scalar_t, math_t, true, true>(
                params, grads, square_avgs, grad_avgs, momentum_buffers,
                lr, alpha, eps, weight_decay, momentum, maximize, op_name);
        } else {
            launch_rmsprop_exact_variant<scalar_t, math_t, true, false>(
                params, grads, square_avgs, grad_avgs, momentum_buffers,
                lr, alpha, eps, weight_decay, momentum, maximize, op_name);
        }
    } else if (momentum != 0.0) {
        launch_rmsprop_exact_variant<scalar_t, math_t, false, true>(
            params, grads, square_avgs, grad_avgs, momentum_buffers,
            lr, alpha, eps, weight_decay, momentum, maximize, op_name);
    } else {
        launch_rmsprop_exact_variant<scalar_t, math_t, false, false>(
            params, grads, square_avgs, grad_avgs, momentum_buffers,
            lr, alpha, eps, weight_decay, momentum, maximize, op_name);
    }
}

template <typename scalar_t, typename math_t>
void launch_adadelta(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& acc_deltas,
        double lr, double rho, double eps, double weight_decay,
        bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &square_avgs, &acc_deltas};
    AdadeltaBody<math_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(rho),
        static_cast<math_t>(eps), static_cast<math_t>(weight_decay), maximize};
    launch_plain_batches<4>(lists,
        [&](const PlainTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<4, scalar_t, math_t,
                                       AdadeltaBody<math_t>,
                                       PlainTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adadelta_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& square_avgs,
        const std::vector<Tensor>& acc_deltas,
        double lr, double rho, double eps, double weight_decay,
        bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &square_avgs, &acc_deltas};
    AdadeltaExactBody<scalar_t, math_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(rho),
        static_cast<math_t>(1.0 - rho), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), maximize};
    launch_plain_batches<4>(lists,
        [&](const PlainTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                4, scalar_t, math_t, AdadeltaExactBody<scalar_t, math_t>,
                PlainTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adamax(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_infs,
        const std::vector<double>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_infs};
    AdamaxBody<math_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(beta1),
        static_cast<math_t>(beta2), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), maximize, math_t(0), math_t(0)};
    launch_simple_batches<4>(lists, nullptr, &steps, nullptr,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<4, scalar_t, math_t,
                                       AdamaxBody<math_t>,
                                       SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adamax_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_infs,
        const std::vector<double>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_infs};
    std::vector<double> step_sizes(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        step_sizes[i] = -lr /
            (1.0 - std::pow(beta1, static_cast<double>(steps[i])));
    }
    AdamaxExactBody<scalar_t, math_t> body{
        static_cast<math_t>(beta1), static_cast<math_t>(beta2),
        static_cast<math_t>(1.0 - beta1), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), maximize, math_t(0)};
    launch_simple_batches<4>(lists, nullptr, &step_sizes, nullptr,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                4, scalar_t, math_t, AdamaxExactBody<scalar_t, math_t>,
                SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_asgd(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& axs,
        const std::vector<Tensor>& mus,
        const std::vector<Tensor>& etas,
        const std::vector<double>* steps,
        const std::vector<Tensor>* device_steps,
        double lr, double lambd, double t0, double alpha,
        double weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 5> lists{
        &params, &grads, &axs, &mus, &etas};
    AsgdBody<math_t, scalar_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(lambd),
        static_cast<math_t>(weight_decay), math_t(0), math_t(0), maximize};
    launch_simple_batches<5>(lists, device_steps, steps, nullptr,
        [&](const SimpleTensorMetadata<5>& metadata, int32_t blocks,
            cudaStream_t stream) {
            if (device_steps != nullptr) {
                asgd_increment_device_steps_kernel<scalar_t, math_t>
                    <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                        metadata);
                checkCuda(cudaGetLastError(), op_name);
            }
            pointwise_optimizer_kernel<5, scalar_t, math_t,
                                       AsgdBody<math_t, scalar_t>,
                                       SimpleTensorMetadata<5>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
            asgd_update_scalars_kernel<scalar_t, math_t>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, lr, lambd, t0, alpha, device_steps != nullptr);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_rprop(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& prevs,
        const std::vector<Tensor>& step_sizes,
        double step_size_min, double step_size_max, double etaminus,
        double etaplus, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &prevs, &step_sizes};
    RpropBody<math_t> body{
        static_cast<math_t>(step_size_min), static_cast<math_t>(step_size_max),
        static_cast<math_t>(etaminus), static_cast<math_t>(etaplus), maximize};
    launch_plain_batches<4>(lists,
        [&](const PlainTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<4, scalar_t, math_t,
                                       RpropBody<math_t>,
                                       PlainTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_rprop_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& prevs,
        const std::vector<Tensor>& step_sizes,
        double step_size_min, double step_size_max, double etaminus,
        double etaplus, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &prevs, &step_sizes};
    RpropExactBody<scalar_t, math_t> body{
        static_cast<math_t>(step_size_min), static_cast<math_t>(step_size_max),
        static_cast<math_t>(etaminus), static_cast<math_t>(etaplus), maximize};
    launch_plain_batches<4>(lists,
        [&](const PlainTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                4, scalar_t, math_t, RpropExactBody<scalar_t, math_t>,
                PlainTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_nadam(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<double>& steps,
        const std::vector<double>& mu_products,
        double lr, double beta1, double beta2, double eps,
        double momentum_decay, double weight_decay,
        bool decoupled_weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_avg_sqs};
    NadamBody<math_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(beta1),
        static_cast<math_t>(beta2), static_cast<math_t>(eps),
        static_cast<math_t>(momentum_decay), static_cast<math_t>(weight_decay),
        maximize, decoupled_weight_decay, math_t(0), math_t(0), math_t(0),
        math_t(0), math_t(0), math_t(0), math_t(0)};
    launch_simple_batches<4>(lists, nullptr, &steps, &mu_products,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<4, scalar_t, math_t,
                                       NadamBody<math_t>,
                                       SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_nadam_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<double>& steps,
        const std::vector<double>& mu_products,
        double lr, double beta1, double beta2, double eps,
        double momentum_decay, double weight_decay,
        bool decoupled_weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_avg_sqs};
    NadamExactBody<scalar_t, math_t> body{
        static_cast<math_t>(beta1), static_cast<math_t>(beta2),
        static_cast<math_t>(1.0 - beta1), static_cast<math_t>(1.0 - beta2),
        static_cast<math_t>(eps), static_cast<math_t>(weight_decay),
        static_cast<math_t>(lr), static_cast<math_t>(1.0 - lr * weight_decay),
        maximize, decoupled_weight_decay, lr, beta1, beta2, momentum_decay,
        math_t(0), math_t(0), math_t(0)};
    // step_sizes carries the host step; correction2_sqrts is reused for the
    // already-rounded CPU mu_product.  NadamExactBody expands both into the
    // three opmath values needed by the elementwise update.
    launch_simple_batches<4>(lists, nullptr, &steps, &mu_products,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                4, scalar_t, math_t, NadamExactBody<scalar_t, math_t>,
                SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_radam(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<double>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool decoupled_weight_decay, bool maximize,
        const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_avg_sqs};
    RadamBody<math_t> body{
        static_cast<math_t>(lr), static_cast<math_t>(beta1),
        static_cast<math_t>(beta2), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), maximize,
        decoupled_weight_decay, math_t(0), math_t(0), math_t(0)};
    launch_simple_batches<4>(lists, nullptr, &steps, nullptr,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<4, scalar_t, math_t,
                                       RadamBody<math_t>,
                                       SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_radam_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<double>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool decoupled_weight_decay, bool maximize,
        const char* op_name) {
    std::array<const std::vector<Tensor>*, 4> lists{
        &params, &grads, &exp_avgs, &exp_avg_sqs};
    std::vector<double> unrectified(params.size());
    std::vector<double> rectified(params.size());
    const double rho_inf = 2.0 / (1.0 - beta2) - 1.0;
    for (size_t i = 0; i < params.size(); ++i) {
        const double step = steps[i];
        const double bc1 = 1.0 - std::pow(beta1, step);
        const double bc2 = 1.0 - std::pow(beta2, step);
        const double rho_t = rho_inf -
            2.0 * step * std::pow(beta2, step) / bc2;
        if (rho_t > 5.0) {
            const double rect = std::sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t));
            unrectified[i] = 0.0;
            rectified[i] = -lr * std::sqrt(bc2) * rect / bc1;
        } else {
            unrectified[i] = -lr / bc1;
            rectified[i] = 0.0;
        }
    }
    RadamExactBody<scalar_t, math_t> body{
        static_cast<math_t>(beta1), static_cast<math_t>(beta2),
        static_cast<math_t>(1.0 - beta1), static_cast<math_t>(1.0 - beta2),
        static_cast<math_t>(eps), static_cast<math_t>(weight_decay),
        static_cast<math_t>(lr), static_cast<math_t>(1.0 - lr * weight_decay),
        maximize, decoupled_weight_decay, math_t(0), math_t(0)};
    // The two arrays are the Python scalar-list coefficients used by the
    // final foreach buffer construction, so the device body never has to
    // recompute rho_t in reduced precision.
    launch_simple_batches<4>(lists, nullptr, &unrectified, &rectified,
        [&](const SimpleTensorMetadata<4>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                4, scalar_t, math_t, RadamExactBody<scalar_t, math_t>,
                SimpleTensorMetadata<4>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t, typename lr_t,
          bool HasMomentum>
void launch_sgd(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& momentum_buffers,
        double lr, double momentum, double dampening, double weight_decay,
        bool nesterov, bool first_momentum_step, bool maximize,
        const float* grad_scale, const float* found_inf,
        const Tensor* tensor_lr, const char* op_name) {
    constexpr int Depth = HasMomentum ? 3 : 2;
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    if constexpr (HasMomentum) lists[2] = &momentum_buffers;
    const lr_t* lr_ptr = tensor_lr == nullptr
        ? nullptr : tensor_lr->data_ptr<lr_t>();
    launch_plain_batches<Depth>(
        lists,
        [&](const PlainTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            sgd_kernel<lr_t, scalar_t, math_t, HasMomentum><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, lr_ptr, lr, momentum, dampening, weight_decay,
                nesterov ? 1 : 0, first_momentum_step ? 1 : 0,
                maximize, grad_scale, found_inf);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t, typename lr_t,
          bool AdamW, bool AMSGrad>
void launch_adam_host(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<Tensor>& max_exp_avg_sqs,
        const std::vector<int64_t>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize, const char* op_name) {
    constexpr int Depth = AMSGrad ? 5 : 4;
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    lists[2] = &exp_avgs;
    lists[3] = &exp_avg_sqs;
    if constexpr (AMSGrad) lists[4] = &max_exp_avg_sqs;
    std::vector<double> step_sizes(params.size());
    std::vector<double> correction2_sqrts(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(steps[i]));
        const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(steps[i]));
        step_sizes[i] = lr / bc1;
        correction2_sqrts[i] = std::sqrt(bc2);
    }
    launch_simple_batches<Depth>(
        lists, nullptr, &step_sizes, &correction2_sqrts,
        [&](const SimpleTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adam_kernel<lr_t, scalar_t, math_t, Depth, false, AdamW, AMSGrad>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, nullptr, lr, beta1, beta2, eps, weight_decay,
                    maximize, nullptr, nullptr);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t, bool AdamW, bool AMSGrad>
void launch_adam_host_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<Tensor>& max_exp_avg_sqs,
        const std::vector<int64_t>& steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize, const char* op_name) {
    constexpr int Depth = AMSGrad ? 5 : 4;
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    lists[2] = &exp_avgs;
    lists[3] = &exp_avg_sqs;
    if constexpr (AMSGrad) lists[4] = &max_exp_avg_sqs;

    std::vector<double> step_sizes(params.size());
    std::vector<double> correction2_sqrts(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(steps[i]));
        const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(steps[i]));
        // The composed foreach path uses addcdiv(..., value=-step_size).
        // Keep the negative scalar in metadata so the final fma has the same
        step_sizes[i] = -(lr / bc1);
        correction2_sqrts[i] = std::sqrt(bc2);
    }
    const double lerp_weight = 1.0 - beta1;
    const double beta2_weight = 1.0 - beta2;
    const double adamw_factor = 1.0 - lr * weight_decay;
    launch_simple_batches<Depth>(
        lists, nullptr, &step_sizes, &correction2_sqrts,
        [&](const SimpleTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adam_exact_kernel<scalar_t, math_t, Depth, AdamW, AMSGrad>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, beta2, lerp_weight, beta2_weight, eps,
                    weight_decay, adamw_factor, maximize);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t, typename lr_t,
          bool AdamW, bool AMSGrad>
void launch_adam_fused(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& exp_avgs,
        const std::vector<Tensor>& exp_avg_sqs,
        const std::vector<Tensor>& max_exp_avg_sqs,
        const std::vector<Tensor>& state_steps,
        double lr, double beta1, double beta2, double eps,
        double weight_decay, bool maximize,
        const float* grad_scale, const float* found_inf,
        const Tensor* tensor_lr, const char* op_name) {
    constexpr int Depth = AMSGrad ? 5 : 4;
    std::array<const std::vector<Tensor>*, Depth> lists{};
    lists[0] = &params;
    lists[1] = &grads;
    lists[2] = &exp_avgs;
    lists[3] = &exp_avg_sqs;
    if constexpr (AMSGrad) lists[4] = &max_exp_avg_sqs;
    const lr_t* lr_ptr = tensor_lr == nullptr
        ? nullptr : tensor_lr->data_ptr<lr_t>();
    launch_simple_batches<Depth>(
        lists, &state_steps, nullptr, nullptr,
        [&](const SimpleTensorMetadata<Depth>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adam_kernel<lr_t, scalar_t, math_t, Depth, true, AdamW, AMSGrad>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, lr_ptr, lr, beta1, beta2, eps, weight_decay,
                    maximize, grad_scale, found_inf);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t, typename lr_t>
void launch_adagrad_fused(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& state_sums,
        const std::vector<Tensor>& state_steps,
        double lr, double lr_decay, double weight_decay, double eps,
        bool maximize, const float* grad_scale, const float* found_inf,
        const Tensor* tensor_lr, const char* op_name) {
    std::array<const std::vector<Tensor>*, 3> lists{
        &params, &grads, &state_sums};
    const lr_t* lr_ptr = tensor_lr == nullptr
        ? nullptr : tensor_lr->data_ptr<lr_t>();
    launch_simple_batches<3>(
        lists, &state_steps, nullptr, nullptr,
        [&](const SimpleTensorMetadata<3>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adagrad_kernel<lr_t, scalar_t, math_t><<<
                static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                metadata, lr_ptr, lr, lr_decay, weight_decay, eps,
                maximize, grad_scale, found_inf);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adagrad_host(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& state_sums,
        const std::vector<double>& steps,
        double lr, double lr_decay, double weight_decay, double eps,
        bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 3> lists{
        &params, &grads, &state_sums};
    launch_simple_batches<3>(lists, nullptr, &steps, nullptr,
        [&](const SimpleTensorMetadata<3>& metadata, int32_t blocks,
            cudaStream_t stream) {
            adagrad_host_kernel<scalar_t, math_t>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, lr, lr_decay, weight_decay, eps, maximize);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

template <typename scalar_t, typename math_t>
void launch_adagrad_host_exact(
        const std::vector<Tensor>& params,
        const std::vector<Tensor>& grads,
        const std::vector<Tensor>& state_sums,
        const std::vector<double>& corrected_lrs,
        double eps, double weight_decay, bool maximize, const char* op_name) {
    std::array<const std::vector<Tensor>*, 3> lists{
        &params, &grads, &state_sums};
    AdagradExactBody<scalar_t, math_t> body{
        math_t(0), static_cast<math_t>(eps),
        static_cast<math_t>(weight_decay), maximize};
    launch_simple_batches<3>(lists, nullptr, &corrected_lrs, nullptr,
        [&](const SimpleTensorMetadata<3>& metadata, int32_t blocks,
            cudaStream_t stream) {
            pointwise_optimizer_kernel<
                3, scalar_t, math_t, AdagradExactBody<scalar_t, math_t>,
                SimpleTensorMetadata<3>>
                <<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
                    metadata, body);
            checkCuda(cudaGetLastError(), op_name);
        }, op_name);
}

} // namespace optimizer_mta
} // namespace cuda
} // namespace tensorplay
