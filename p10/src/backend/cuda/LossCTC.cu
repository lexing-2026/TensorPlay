// Native CUDA connectionist-temporal-classification implementation. The
// alpha/beta recurrences use log-space arithmetic. Generated schemas store
// padded targets as a (batch, max_target_length) int64 tensor, so this path
// keeps that layout while using state-parallel kernels.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "../composite/AttentionComposite.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

template <typename target_t>
__device__ inline int64_t get_target_prime(const target_t* targets,
                                           int64_t batch_offset,
                                           int64_t target_stride,
                                           int64_t index, int64_t blank) {
    return (index & 1) == 0
        ? blank : targets[batch_offset + (index / 2) * target_stride];
}

template <typename scalar_t, typename target_t>
__global__ void ctc_loss_log_alpha_gpu_kernel(
        scalar_t* __restrict__ log_alpha,
        const scalar_t* __restrict__ log_probs,
        const int64_t* __restrict__ input_lengths,
        const target_t* __restrict__ targets,
        const int64_t* __restrict__ target_offsets,
        const int64_t* __restrict__ target_lengths,
        scalar_t* __restrict__ neg_log_likelihood,
        int64_t max_input_length, int64_t num_labels,
        int64_t target_stride,
        int64_t max_target_length,
        int64_t batch_size, int64_t blank) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

    const int64_t b = static_cast<int64_t>(threadIdx.y) +
                      static_cast<int64_t>(blockIdx.y) * blockDim.y;
    const bool valid_batch = b < batch_size;
    const int64_t input_length = valid_batch ? input_lengths[b] : 0;
    const int64_t target_length = valid_batch ? target_lengths[b] : 0;
    const int64_t states = 2 * max_target_length + 1;
    const int64_t state_stride = states;
    const int64_t batch_alpha_offset = b * max_input_length * state_stride;

    // Every thread in the block must reach every barrier.  Invalid y-lanes
    // in the last batch block therefore execute the same recurrence with an
    // empty sequence instead of returning early.
    for (int64_t block_s = 0; block_s < states; block_s += blockDim.x) {
        const int64_t s = block_s + threadIdx.x;
        scalar_t value = neginf;
        if (valid_batch && input_length > 0 && s == 0) {
            value = log_probs[b * num_labels + blank];
        } else if (valid_batch && input_length > 0 && s == 1 && target_length > 0) {
            const int64_t target = get_target_prime(
                    targets, target_offsets[b], target_stride, 1, blank);
            value = log_probs[b * num_labels + target];
        }
        if (valid_batch && s < states) {
            log_alpha[batch_alpha_offset + s] = value;
        }
        __syncthreads();
    }

    for (int64_t block_s = 0; block_s < states; block_s += blockDim.x) {
        const int64_t s = block_s + threadIdx.x;
        int64_t current_char = blank;
        bool have_three = false;
        if (valid_batch && target_length > 0 && s < 2 * target_length + 1) {
            current_char = get_target_prime(
                    targets, target_offsets[b], target_stride, s, blank);
            have_three = s > 1 &&
                get_target_prime(targets, target_offsets[b], target_stride,
                                 s - 2, blank) != current_char;
        }

        for (int64_t t = 1; t < max_input_length; ++t) {
            __syncthreads();
            if (valid_batch && input_length > t && s < 2 * target_length + 1) {
                const scalar_t la1 = log_alpha[
                    batch_alpha_offset + (t - 1) * state_stride + s];
                scalar_t lamax = la1;
                scalar_t la2 = neginf;
                scalar_t la3 = neginf;
                if (s > 0) {
                    la2 = log_alpha[
                        batch_alpha_offset + (t - 1) * state_stride + s - 1];
                    if (la2 > lamax) lamax = la2;
                }
                if (have_three) {
                    la3 = log_alpha[
                        batch_alpha_offset + (t - 1) * state_stride + s - 2];
                    if (la3 > lamax) lamax = la3;
                }
                if (lamax == neginf) lamax = static_cast<scalar_t>(0);
                const scalar_t sum = std::exp(la1 - lamax) +
                                     std::exp(la2 - lamax) +
                                     std::exp(la3 - lamax);
                log_alpha[batch_alpha_offset + t * state_stride + s] =
                    std::log(sum) + lamax +
                    log_probs[(t * batch_size + b) * num_labels + current_char];
            } else if (valid_batch && s < states) {
                log_alpha[batch_alpha_offset + t * state_stride + s] = neginf;
            }
        }
        __syncthreads();
    }

    if (valid_batch && threadIdx.x == 0) {
        if (input_length == 0) {
            neg_log_likelihood[b] = target_length == 0
                ? static_cast<scalar_t>(0) : -neginf;
            return;
        }
        const int64_t last = input_length - 1;
        const scalar_t l1 = log_alpha[
            batch_alpha_offset + last * state_stride + 2 * target_length];
        const scalar_t l2 = target_length > 0
            ? log_alpha[batch_alpha_offset + last * state_stride +
                        2 * target_length - 1]
            : neginf;
        scalar_t m = l1 > l2 ? l1 : l2;
        if (m == neginf) m = static_cast<scalar_t>(0);
        const scalar_t log_likelihood =
            std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
        neg_log_likelihood[b] = -log_likelihood;
    }
}

template <typename scalar_t, typename target_t>
__global__ void ctc_loss_backward_log_beta_gpu_kernel(
        scalar_t* __restrict__ log_beta,
        const scalar_t* __restrict__ log_probs,
        const int64_t* __restrict__ input_lengths,
        const target_t* __restrict__ targets,
        const int64_t* __restrict__ target_offsets,
        const int64_t* __restrict__ target_lengths,
        int64_t max_input_length, int64_t num_labels,
        int64_t target_stride,
        int64_t max_target_length,
        int64_t batch_size, int64_t blank) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

    const int64_t b = static_cast<int64_t>(threadIdx.y) +
                      static_cast<int64_t>(blockIdx.y) * blockDim.y;
    const bool valid_batch = b < batch_size;
    const int64_t input_length = valid_batch ? input_lengths[b] : 0;
    const int64_t target_length = valid_batch ? target_lengths[b] : 0;
    const int64_t states = 2 * max_target_length + 1;
    const int64_t batch_beta_offset = b * max_input_length * states;
    const int64_t last_block = states - 1 - (states - 1) % blockDim.x;
    for (int64_t block_s = last_block; block_s >= 0; block_s -= blockDim.x) {
        const int64_t s = block_s + threadIdx.x;
        scalar_t value = neginf;
        if (valid_batch && input_length > 0 && s == 2 * target_length) {
            value = log_probs[((input_length - 1) * batch_size + b) * num_labels + blank];
        } else if (valid_batch && input_length > 0 &&
                   s == 2 * target_length - 1) {
            const int64_t target = get_target_prime(
                targets, target_offsets[b], target_stride, s, blank);
            value = log_probs[((input_length - 1) * batch_size + b) * num_labels + target];
        }
        if (valid_batch && s < states) {
            log_beta[batch_beta_offset +
                     (input_length > 0 ? input_length - 1 : 0) * states + s] = value;
        }
        __syncthreads();

        int64_t current_char = blank;
        bool have_three = false;
        if (valid_batch && target_length > 0 && s < 2 * target_length + 1) {
            current_char = get_target_prime(
                targets, target_offsets[b], target_stride, s, blank);
            have_three = s < 2 * target_length - 1 &&
                get_target_prime(targets, target_offsets[b], target_stride,
                                 s + 2, blank) !=
                current_char;
        }

        for (int64_t t = max_input_length - 2; t >= 0; --t) {
            __syncthreads();
            if (valid_batch && t < input_length - 1 &&
                s < 2 * target_length + 1) {
                const scalar_t lb1 = log_beta[
                    batch_beta_offset + (t + 1) * states + s];
                scalar_t lbmax = lb1;
                scalar_t lb2 = neginf;
                scalar_t lb3 = neginf;
                if (s < 2 * target_length) {
                    lb2 = log_beta[
                        batch_beta_offset + (t + 1) * states + s + 1];
                    if (lb2 > lbmax) lbmax = lb2;
                }
                if (have_three) {
                    lb3 = log_beta[
                        batch_beta_offset + (t + 1) * states + s + 2];
                    if (lb3 > lbmax) lbmax = lb3;
                }
                if (lbmax == neginf) lbmax = static_cast<scalar_t>(0);
                log_beta[batch_beta_offset + t * states + s] =
                    std::log(std::exp(lb1 - lbmax) +
                             std::exp(lb2 - lbmax) +
                             std::exp(lb3 - lbmax)) + lbmax +
                    log_probs[(t * batch_size + b) * num_labels + current_char];
            } else if (valid_batch && s < states) {
                log_beta[batch_beta_offset + t * states + s] = neginf;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename target_t>
__global__ void ctc_loss_backward_collect_gpu_kernel(
        scalar_t* __restrict__ gradient,
        const scalar_t* __restrict__ grad_out,
        const scalar_t* __restrict__ log_alpha,
        const scalar_t* __restrict__ log_beta,
        const scalar_t* __restrict__ log_probs,
        const int64_t* __restrict__ input_lengths,
        const target_t* __restrict__ targets,
        const int64_t* __restrict__ target_offsets,
        const int64_t* __restrict__ target_lengths,
        const scalar_t* __restrict__ neg_log_likelihood,
        int64_t max_input_length, int64_t batch_size, int64_t num_labels,
        int64_t target_stride,
        int64_t max_target_length, int64_t blank,
        bool zero_infinity) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    const int64_t t = static_cast<int64_t>(threadIdx.x) +
                      static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t b = static_cast<int64_t>(threadIdx.y) +
                      static_cast<int64_t>(blockIdx.y) * blockDim.y;
    if (t >= max_input_length || b >= batch_size) return;

    const int64_t input_length = input_lengths[b];
    const int64_t target_length = target_lengths[b];
    const int64_t states = 2 * max_target_length + 1;
    const int64_t lp_offset = (t * batch_size + b) * num_labels;
    const int64_t alpha_offset = (b * max_input_length + t) * states;
    const int64_t beta_offset = alpha_offset;
    const int64_t grad_offset = lp_offset;
    // Collect log(alpha * beta) for all augmented states sharing a label.
    // One thread owns one (batch, timestep) row, so the label collection is
    // deterministic and needs no atomic adds.
    for (int64_t s = 0; s < 2 * target_length + 1; ++s) {
        const int64_t label = get_target_prime(
            targets, target_offsets[b], target_stride, s, blank);
        const scalar_t log_alpha_beta =
            log_alpha[alpha_offset + s] + log_beta[beta_offset + s];
        scalar_t& collected = gradient[grad_offset + label];
        if (collected == neginf) {
            collected = log_alpha_beta;
        } else {
            const scalar_t m = collected > log_alpha_beta
                ? collected : log_alpha_beta;
            collected = std::log(std::exp(collected - m) +
                                 std::exp(log_alpha_beta - m)) + m;
        }
    }

    const scalar_t nll = neg_log_likelihood[b];
    const scalar_t grad_scale = grad_out[b];
    for (int64_t c = 0; c < num_labels; ++c) {
        scalar_t& result = gradient[grad_offset + c];
        if (t < input_length &&
            (!zero_infinity ||
             nll != std::numeric_limits<scalar_t>::infinity())) {
            const scalar_t lp = log_probs[lp_offset + c];
            result = (std::exp(lp) -
                      std::exp(result + nll - lp)) * grad_scale;
        } else {
            result = static_cast<scalar_t>(0);
        }
    }
}

struct CtcPrepared {
    Tensor log_probs;
    Tensor targets;
    Tensor target_offsets;
    Tensor input_lengths;
    Tensor target_lengths;
    std::vector<int64_t> input_lengths_host;
    std::vector<int64_t> target_lengths_host;
    int64_t max_target_length = 0;
};

int64_t ctc_target_stride(const CtcPrepared& prepared) {
    return prepared.targets.dim() == 1
        ? prepared.targets.stride(0) : prepared.targets.stride(1);
}

std::vector<int64_t> read_lengths(const Tensor& lengths, int64_t batch_size,
                                  const char* name) {
    if (!lengths.defined() || lengths.numel() != batch_size) {
        TP_THROW(ValueError, name, " must have one value per batch item");
    }
    Tensor host = lengths.to(Device(DeviceType::CPU), DType::Int64).contiguous();
    const auto* data = host.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + host.numel());
}

CtcPrepared prepare_ctc(const Tensor& log_probs, const Tensor& targets,
                        const Tensor& input_lengths, const Tensor& target_lengths,
                        int64_t blank) {
    if (log_probs.dim() != 3) {
        TP_THROW(ValueError, "_ctc_loss: log_probs must be 3-dimensional");
    }
    if (log_probs.numel() == 0) {
        TP_THROW(ValueError, "_ctc_loss: log_probs tensor must not be empty");
    }
    if (targets.dim() != 1 && targets.dim() != 2) {
        TP_THROW(ValueError,
                 "_ctc_loss: targets must be a 1-dimensional concatenated "
                 "tensor or a 2-dimensional padded tensor");
    }
    if (targets.dim() == 2 && targets.size(0) != log_probs.size(1)) {
        TP_THROW(ValueError, "_ctc_loss: targets batch dimension must match log_probs");
    }
    if (targets.dtype() != DType::Int64 && targets.dtype() != DType::Int32) {
        TP_THROW(TypeError, "_ctc_loss: targets must have Int32 or Int64 dtype");
    }
    if (blank < 0 || blank >= log_probs.size(2)) {
        TP_THROW(ValueError, "_ctc_loss: blank must be in the label range");
    }
    if (log_probs.dtype() != DType::Float32 && log_probs.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError,
                 "_ctc_loss CUDA supports Float32 and Float64 only");
    }

    const int64_t batch_size = log_probs.size(1);
    auto input_lengths_host = read_lengths(input_lengths, batch_size, "input_lengths");
    auto target_lengths_host = read_lengths(target_lengths, batch_size, "target_lengths");
    int64_t max_target_length = 0;
    for (int64_t b = 0; b < batch_size; ++b) {
        if (input_lengths_host[b] < 0 || input_lengths_host[b] > log_probs.size(0)) {
            TP_THROW(ValueError, "_ctc_loss: input_lengths out of range");
        }
        const int64_t target_capacity = targets.dim() == 1
            ? targets.numel() : targets.size(1);
        if (target_lengths_host[b] < 0 || target_lengths_host[b] > target_capacity) {
            TP_THROW(ValueError, "_ctc_loss: target_lengths out of range");
        }
        max_target_length = std::max(max_target_length, target_lengths_host[b]);
    }

    const Device device = log_probs.device();
    Tensor target_device = targets.to(device).contiguous();
    Tensor target_offsets_cpu = Tensor::empty(
        {batch_size}, DType::Int64, Device(DeviceType::CPU));
    int64_t* target_offsets_data = target_offsets_cpu.data_ptr<int64_t>();
    const int64_t target_stride = target_device.dim() == 1
        ? target_device.stride(0) : target_device.stride(1);
    if (target_device.dim() == 1) {
        int64_t offset = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            target_offsets_data[b] = offset;
            offset += target_lengths_host[b];
        }
        if (offset != target_device.numel()) {
            TP_THROW(ValueError,
                     "_ctc_loss: concatenated targets size must equal the "
                     "sum of target_lengths");
        }
    } else {
        for (int64_t b = 0; b < batch_size; ++b) {
            target_offsets_data[b] = b * target_device.stride(0);
        }
    }
    CtcPrepared result{
        log_probs.is_contiguous() ? log_probs : log_probs.contiguous(),
        std::move(target_device),
        target_offsets_cpu.to(device).contiguous(),
        input_lengths.to(device, DType::Int64).contiguous(),
        target_lengths.to(device, DType::Int64).contiguous(),
        std::move(input_lengths_host), std::move(target_lengths_host),
        max_target_length};
    return result;
}

struct CtcStateLaunch {
    int threads_state;
    int threads_batch;
};

CtcStateLaunch state_launch(int64_t states, int64_t batch_size) {
    int threads_state = 1;
    while (threads_state < states && threads_state < 256) threads_state <<= 1;
    threads_state = std::min(threads_state, 256);
    const int threads_batch = std::max(
        1, std::min(256 / threads_state, static_cast<int>(batch_size)));
    return {threads_state, threads_batch};
}

void check_ctc_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, op, ": ", cudaGetErrorString(error));
    }
}

template <typename scalar_t, typename target_t>
std::tuple<Tensor, Tensor> ctc_loss_cuda_impl(const CtcPrepared& prepared,
                                              int64_t blank) {
    const int64_t max_input_length = prepared.log_probs.size(0);
    const int64_t batch_size = prepared.log_probs.size(1);
    const int64_t num_labels = prepared.log_probs.size(2);
    const int64_t states = 2 * prepared.max_target_length + 1;
    Tensor neg_log_likelihood = Tensor::empty(
        {batch_size}, prepared.log_probs.dtype(), prepared.log_probs.device());
    Tensor log_alpha = Tensor::empty(
        {batch_size, max_input_length, states},
        prepared.log_probs.dtype(), prepared.log_probs.device());

    const CtcStateLaunch launch = state_launch(states, batch_size);
    const dim3 block(launch.threads_state, launch.threads_batch);
    const dim3 grid(1, (batch_size + launch.threads_batch - 1) /
                           launch.threads_batch);
    ctc_loss_log_alpha_gpu_kernel<scalar_t, target_t><<<
        grid, block, 0, getCurrentCUDAStream().stream()>>>(
        log_alpha.data_ptr<scalar_t>(),
        prepared.log_probs.data_ptr<scalar_t>(),
        prepared.input_lengths.data_ptr<int64_t>(),
        prepared.targets.data_ptr<target_t>(),
        prepared.target_offsets.data_ptr<int64_t>(),
        prepared.target_lengths.data_ptr<int64_t>(),
        neg_log_likelihood.data_ptr<scalar_t>(),
        max_input_length, num_labels, ctc_target_stride(prepared),
        prepared.max_target_length, batch_size, blank);
    check_ctc_launch("_ctc_loss CUDA alpha kernel");
    return {neg_log_likelihood, log_alpha};
}

template <typename scalar_t, typename target_t>
Tensor ctc_loss_backward_cuda_impl(const Tensor& grad,
                                   const CtcPrepared& prepared,
                                   const Tensor& neg_log_likelihood,
                                   const Tensor& log_alpha, int64_t blank,
                                   bool zero_infinity) {
    const Tensor grad_c = grad.contiguous();
    const Tensor nll_c = neg_log_likelihood.contiguous();
    const Tensor alpha_c = log_alpha.contiguous();
    const int64_t max_input_length = prepared.log_probs.size(0);
    const int64_t batch_size = prepared.log_probs.size(1);
    const int64_t num_labels = prepared.log_probs.size(2);
    const int64_t states = 2 * prepared.max_target_length + 1;
    if (alpha_c.dim() != 3 || alpha_c.size(0) != batch_size ||
        alpha_c.size(1) != max_input_length || alpha_c.size(2) != states) {
        TP_THROW(ValueError, "_ctc_loss_backward: invalid log_alpha shape");
    }
    if (nll_c.numel() != batch_size || grad_c.numel() != batch_size) {
        TP_THROW(ValueError, "_ctc_loss_backward: grad and nll must match the batch size");
    }

    const scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    Tensor log_beta = Tensor::full_like(
        alpha_c, Scalar(static_cast<double>(neginf)));
    Tensor gradient = Tensor::full_like(
        prepared.log_probs, Scalar(static_cast<double>(neginf)));

    const CtcStateLaunch state_cfg = state_launch(states, batch_size);
    const dim3 state_block(state_cfg.threads_state, state_cfg.threads_batch);
    const dim3 state_grid(1, (batch_size + state_cfg.threads_batch - 1) /
                              state_cfg.threads_batch);
    ctc_loss_backward_log_beta_gpu_kernel<scalar_t, target_t><<<
        state_grid, state_block, 0, getCurrentCUDAStream().stream()>>>(
        log_beta.data_ptr<scalar_t>(),
        prepared.log_probs.data_ptr<scalar_t>(),
        prepared.input_lengths.data_ptr<int64_t>(),
        prepared.targets.data_ptr<target_t>(),
        prepared.target_offsets.data_ptr<int64_t>(),
        prepared.target_lengths.data_ptr<int64_t>(),
        max_input_length, num_labels, ctc_target_stride(prepared),
        prepared.max_target_length, batch_size, blank);
    check_ctc_launch("_ctc_loss_backward CUDA beta kernel");

    const int threads_time = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(256, max_input_length)));
    const int threads_batch = std::max(
        1, std::min(1024 / threads_time, static_cast<int>(batch_size)));
    const dim3 collect_block(threads_time, threads_batch);
    const dim3 collect_grid(
        (max_input_length + threads_time - 1) / threads_time,
        (batch_size + threads_batch - 1) / threads_batch);
    ctc_loss_backward_collect_gpu_kernel<scalar_t, target_t><<<
        collect_grid, collect_block, 0, getCurrentCUDAStream().stream()>>>(
        gradient.data_ptr<scalar_t>(), grad_c.data_ptr<scalar_t>(),
        alpha_c.data_ptr<scalar_t>(), log_beta.data_ptr<scalar_t>(),
        prepared.log_probs.data_ptr<scalar_t>(),
        prepared.input_lengths.data_ptr<int64_t>(),
        prepared.targets.data_ptr<target_t>(),
        prepared.target_offsets.data_ptr<int64_t>(),
        prepared.target_lengths.data_ptr<int64_t>(),
        nll_c.data_ptr<scalar_t>(), max_input_length, batch_size, num_labels,
        ctc_target_stride(prepared),
        prepared.max_target_length, blank,
        zero_infinity);
    check_ctc_launch("_ctc_loss_backward CUDA collect kernel");
    return gradient;
}

}  // namespace

std::tuple<Tensor, Tensor> _ctc_loss_cuda(const Tensor& log_probs,
                                          const Tensor& targets,
                                          const Tensor& input_lengths,
                                          const Tensor& target_lengths,
                                          int64_t blank, bool zero_infinity) {
    (void)zero_infinity;
    CtcPrepared prepared = prepare_ctc(
        log_probs, targets, input_lengths, target_lengths, blank);
    if (prepared.log_probs.dtype() == DType::Float32) {
        if (prepared.targets.dtype() == DType::Int64) {
            return ctc_loss_cuda_impl<float, int64_t>(prepared, blank);
        }
        return ctc_loss_cuda_impl<float, int32_t>(prepared, blank);
    }
    if (prepared.targets.dtype() == DType::Int64) {
        return ctc_loss_cuda_impl<double, int64_t>(prepared, blank);
    }
    return ctc_loss_cuda_impl<double, int32_t>(prepared, blank);
}

Tensor _ctc_loss_backward_cuda(const Tensor& grad, const Tensor& log_probs,
                               const Tensor& targets,
                               const Tensor& input_lengths,
                               const Tensor& target_lengths,
                               const Tensor& neg_log_likelihood,
                               const Tensor& log_alpha, int64_t blank,
                               bool zero_infinity) {
    CtcPrepared prepared = prepare_ctc(
        log_probs, targets, input_lengths, target_lengths, blank);
    if (neg_log_likelihood.device() != prepared.log_probs.device() ||
        log_alpha.device() != prepared.log_probs.device() ||
        grad.device() != prepared.log_probs.device()) {
        TP_THROW(DeviceMismatchError,
                 "_ctc_loss_backward: all CUDA tensors must be on the same device");
    }
    if (prepared.log_probs.dtype() == DType::Float32) {
        if (prepared.targets.dtype() == DType::Int64) {
            return ctc_loss_backward_cuda_impl<float, int64_t>(
                grad, prepared, neg_log_likelihood, log_alpha, blank, zero_infinity);
        }
        return ctc_loss_backward_cuda_impl<float, int32_t>(
            grad, prepared, neg_log_likelihood, log_alpha, blank, zero_infinity);
    }
    if (prepared.targets.dtype() == DType::Int64) {
        return ctc_loss_backward_cuda_impl<double, int64_t>(
            grad, prepared, neg_log_likelihood, log_alpha, blank, zero_infinity);
    }
    return ctc_loss_backward_cuda_impl<double, int32_t>(
        grad, prepared, neg_log_likelihood, log_alpha, blank, zero_infinity);
}


// Public-op composites shared with the CPU registration; the inner `_ctc_loss`
// dispatches to the CUDA kernel above.
Tensor ctc_loss_intlist_cuda(const Tensor& log_probs, const Tensor& targets,
                             const std::vector<int64_t>& input_lengths,
                             const std::vector<int64_t>& target_lengths,
                             int64_t blank, int64_t reduction,
                             bool zero_infinity) {
  Tensor il = Tensor::tensor(input_lengths, DType::Int64, log_probs.device());
  Tensor tl = Tensor::tensor(target_lengths, DType::Int64, log_probs.device());
  return tensorplay::composite::ctc_loss_compose(log_probs, targets, il, tl,
                                                 blank, reduction,
                                                 zero_infinity);
}

Tensor ctc_loss_tensor_cuda(const Tensor& log_probs, const Tensor& targets,
                            const Tensor& input_lengths,
                            const Tensor& target_lengths, int64_t blank,
                            int64_t reduction, bool zero_infinity) {
  return tensorplay::composite::ctc_loss_compose(log_probs, targets,
                                                 input_lengths, target_lengths,
                                                 blank, reduction,
                                                 zero_infinity);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeCTCLoss) {
    m.impl("_ctc_loss", _ctc_loss_cuda);
    m.impl("_ctc_loss_backward", _ctc_loss_backward_cuda);
    m.impl("ctc_loss.IntList", ctc_loss_intlist_cuda);
    m.impl("ctc_loss.Tensor", ctc_loss_tensor_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
