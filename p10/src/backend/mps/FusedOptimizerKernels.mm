#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

namespace {

bool is_mps_tensor(const Tensor& t, const char* op) {
  if (!t.defined() || !t.device().is_mps()) {
    TP_THROW(RuntimeError, op, ": all tensors must be on the MPS device");
  }
  return true;
}

// Returns a contiguous, offset-0 tensor safe for direct buffer binding.
Tensor shader_tensor(const Tensor& t) {
  return (t.is_contiguous() && t.storage_offset() == 0) ? t
                                                        : t.contiguous();
}

id<MTLBuffer> mtl_buffer_of(const Tensor& t) {
  void* buf = getMTLBufferFromPointer(t.data_ptr());
  if (!buf) {
    TP_THROW(RuntimeError,
             "MPS tensor storage is not managed by the MPS allocator");
  }
  return (__bridge id<MTLBuffer>)buf;
}

// Binds an optional 0-d float tensor (grad_scale / found_inf) or nil.
void set_optional_float_buffer(id<MTLComputeCommandEncoder> encoder,
                              const std::optional<Tensor>& t, unsigned idx) {
  if (t.has_value() && t->defined()) {
    Tensor c = shader_tensor(*t);
    [encoder setBuffer:mtl_buffer_of(c) offset:0 atIndex:idx];
  } else {
    [encoder setBuffer:nil offset:0 atIndex:idx];
  }
}

double read_scalar_lr(const Tensor& lr) {
  if (lr.numel() != 1) {
    TP_THROW(RuntimeError, "_fused_adamw_: tensor lr must be a scalar");
  }
  Tensor lr_cpu = lr.to(Device(DeviceType::CPU));
  if (lr_cpu.dtype() == DType::Float32) {
    return static_cast<double>(*static_cast<const float*>(lr_cpu.data_ptr()));
  }
  if (lr_cpu.dtype() == DType::Float64) {
    return *static_cast<const double*>(lr_cpu.data_ptr());
  }
  TP_THROW(RuntimeError, "_fused_adamw_: unsupported tensor lr dtype");
}

struct MPSMetadataArguments {
  uint64_t numels[32];
  uint64_t threadgroup_to_tensor[32];
  uint64_t threadgroup_to_chunk[32];
};

// One parameter per launch, with every tensor bound directly.  A GPU image
// without argument encoders can only reach tensors this way, so a foreach call
// made there runs one launch per parameter.
void dispatch_fused_adam_single(
    id<MTLComputeCommandEncoder> computeEncoder,
    const std::string& dtype,
    const Tensor& param,
    const Tensor& grad,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    const Tensor& max_exp_avg_sq,
    const Tensor& state_steps,
    bool amsgrad,
    bool adamw,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool maximize,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf) {
  if (param.numel() == 0) {
    return;
  }
  const std::string kernel =
      (amsgrad ? (adamw ? "fused_adamw_amsgrad_" : "fused_adam_amsgrad_")
               : (adamw ? "fused_adamw_" : "fused_adam_")) +
      dtype + "_float";
  id<MTLComputePipelineState> PSO =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(kernel);
  [computeEncoder setComputePipelineState:PSO];
  unsigned idx = 0;
  [computeEncoder setBuffer:mtl_buffer_of(param) offset:0 atIndex:idx++];
  [computeEncoder setBuffer:mtl_buffer_of(grad) offset:0 atIndex:idx++];
  [computeEncoder setBuffer:mtl_buffer_of(exp_avg) offset:0 atIndex:idx++];
  [computeEncoder setBuffer:mtl_buffer_of(exp_avg_sq) offset:0 atIndex:idx++];
  if (amsgrad) {
    [computeEncoder setBuffer:mtl_buffer_of(max_exp_avg_sq)
                       offset:0
                      atIndex:idx++];
  }
  [computeEncoder setBuffer:mtl_buffer_of(state_steps)
                     offset:0
                    atIndex:idx++];
  const float lr_f = static_cast<float>(lr);
  const float beta1_f = static_cast<float>(beta1);
  const float beta2_f = static_cast<float>(beta2);
  const float wd_f = static_cast<float>(weight_decay);
  const float eps_f = static_cast<float>(eps);
  const uint8_t max_f = maximize ? 1 : 0;
  [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:idx++];
  [computeEncoder setBytes:&beta1_f length:sizeof(float) atIndex:idx++];
  [computeEncoder setBytes:&beta2_f length:sizeof(float) atIndex:idx++];
  [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:idx++];
  [computeEncoder setBytes:&eps_f length:sizeof(float) atIndex:idx++];
  [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
  set_optional_float_buffer(computeEncoder, grad_scale, idx++);
  set_optional_float_buffer(computeEncoder, found_inf, idx++);
  mtl_dispatch1DJob(computeEncoder, PSO,
                    static_cast<NSUInteger>(param.numel()));
}

// Multi-tensor (foreach) fused optimizer dispatch.  A single kernel launch
// updates up to 32 parameters; each parameter is processed in 64K-element
// chunks mapped to threadgroups.  Buffer 0 (the argument struct with arrays
// of device pointers) is bound with ForeachArguments.
void dispatch_fused_adam_multi(
    id<MTLComputeCommandEncoder> computeEncoder,
    const std::vector<Tensor>& params,
    const std::vector<Tensor>& grads,
    const std::vector<Tensor>& exp_avgs,
    const std::vector<Tensor>& exp_avg_sqs,
    const std::vector<Tensor>& max_exp_avg_sqs,
    const std::vector<Tensor>& state_steps,
    bool amsgrad,
    bool adamw,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps,
    bool maximize,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf) {
  constexpr int64_t kChunkSize = 65536;
  constexpr int64_t kmaxThreadGroups = 32;
  constexpr int64_t kmaxTensors = 32;
  constexpr int64_t kThreadGroupSize = 512;
  const int64_t depth = amsgrad ? 5 : 4;
  const size_t num_tensors = params.size();
  if (num_tensors == 0) {
    return;
  }

  const std::string dtype = scalarToMetalTypeString(params[0].dtype());
  const std::string kernel =
      (amsgrad
           ? (adamw ? "fused_adamw_amsgrad_multi_"
                    : "fused_adam_amsgrad_multi_")
           : (adamw ? "fused_adamw_multi_" : "fused_adam_multi_")) +
      dtype + "_float";
  MTLFunction_t func =
      MetalShaderLibrary::getBundledLibrary().getFunction(kernel);
  if (!argumentEncodersAvailable(func)) {
    [func release];
    for (size_t i = 0; i < num_tensors; ++i) {
      dispatch_fused_adam_single(
          computeEncoder, dtype, params[i], grads[i], exp_avgs[i],
          exp_avg_sqs[i], amsgrad ? max_exp_avg_sqs[i] : Tensor(),
          state_steps[i], amsgrad, adamw, lr, beta1, beta2, weight_decay, eps,
          maximize, grad_scale, found_inf);
    }
    return;
  }
  id<MTLComputePipelineState> PSO =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(kernel);
  [computeEncoder setComputePipelineState:PSO];
  ForeachArguments arguments(computeEncoder, func,
                             MPSDevice::getInstance()->device());
  [func release];

  int64_t tensor_loc = 0;
  int64_t threadgroup_loc = 0;
  MPSMetadataArguments metadata_arguments;
  memset(&metadata_arguments, 0, sizeof(metadata_arguments));
  id<MTLBuffer> tensorArgumentBuffer = nil;

  for (size_t tensor_index = 0; tensor_index < num_tensors; ++tensor_index) {
    const int64_t numel = params[tensor_index].numel();
    if (numel == 0) {
      continue;
    }
    if (tensorArgumentBuffer == nil) {
      tensorArgumentBuffer = arguments.buffer();
    }
    for (int64_t d = 0; d < depth; ++d) {
      const Tensor* t = nullptr;
      if (d == 0) t = &params[tensor_index];
      else if (d == 1) t = &grads[tensor_index];
      else if (d == 2) t = &exp_avgs[tensor_index];
      else if (d == 3) t = &exp_avg_sqs[tensor_index];
      else t = &max_exp_avg_sqs[tensor_index];
      arguments.bind(static_cast<NSUInteger>(d * kmaxTensors + tensor_loc),
                     mtl_buffer_of(*t));
    }
    arguments.bind(static_cast<NSUInteger>(depth * kmaxTensors + tensor_loc),
                   mtl_buffer_of(state_steps[tensor_index]));
    metadata_arguments.numels[tensor_loc] = static_cast<uint64_t>(numel);
    tensor_loc++;

    const int64_t chunks =
        numel / kChunkSize + (numel % kChunkSize != 0);
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      metadata_arguments.threadgroup_to_tensor[threadgroup_loc] =
          static_cast<uint64_t>(tensor_loc - 1);
      metadata_arguments.threadgroup_to_chunk[threadgroup_loc] =
          static_cast<uint64_t>(chunk);
      threadgroup_loc++;

      const bool tensor_full =
          tensor_loc == kmaxTensors && chunk == chunks - 1;
      const bool blocks_full = threadgroup_loc == kmaxThreadGroups;

      if (tensor_full || blocks_full) {
        [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
        [computeEncoder setBytes:&metadata_arguments
                          length:sizeof(MPSMetadataArguments)
                           atIndex:1];
        const float lr_f = static_cast<float>(lr);
        const float beta1_f = static_cast<float>(beta1);
        const float beta2_f = static_cast<float>(beta2);
        const float wd_f = static_cast<float>(weight_decay);
        const float eps_f = static_cast<float>(eps);
        const uint8_t max_f = maximize ? 1 : 0;
        [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:2];
        [computeEncoder setBytes:&beta1_f length:sizeof(float) atIndex:3];
        [computeEncoder setBytes:&beta2_f length:sizeof(float) atIndex:4];
        [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:5];
        [computeEncoder setBytes:&eps_f length:sizeof(float) atIndex:6];
        [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:7];
        set_optional_float_buffer(computeEncoder, grad_scale, 8);
        set_optional_float_buffer(computeEncoder, found_inf, 9);

        MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
        const uint32_t maxThreadsPerGroup =
            [PSO maxTotalThreadsPerThreadgroup];
        MTLSize threadGroupSize = MTLSizeMake(
            std::min<uint32_t>(maxThreadsPerGroup, kThreadGroupSize), 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadGroupSize];

        threadgroup_loc = 0;
        if (chunk == chunks - 1) {
          tensor_loc = 0;
          tensorArgumentBuffer = arguments.buffer();
        } else {
          // Reuse the current tensor for the remaining chunks.
          metadata_arguments.numels[0] =
              metadata_arguments.numels[tensor_loc - 1];
          tensorArgumentBuffer = arguments.buffer();
          for (int64_t d = 0; d < depth; ++d) {
            const Tensor* t = nullptr;
            if (d == 0) t = &params[tensor_index];
            else if (d == 1) t = &grads[tensor_index];
            else if (d == 2) t = &exp_avgs[tensor_index];
            else if (d == 3) t = &exp_avg_sqs[tensor_index];
            else t = &max_exp_avg_sqs[tensor_index];
            arguments.bind(static_cast<NSUInteger>(d * kmaxTensors),
                           mtl_buffer_of(*t));
          }
          arguments.bind(static_cast<NSUInteger>(depth * kmaxTensors),
                         mtl_buffer_of(state_steps[tensor_index]));
          tensor_loc = 1;
        }
      }
    }
  }

  if (threadgroup_loc != 0) {
    [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
    [computeEncoder setBytes:&metadata_arguments
                      length:sizeof(MPSMetadataArguments)
                       atIndex:1];
    const float lr_f = static_cast<float>(lr);
    const float beta1_f = static_cast<float>(beta1);
    const float beta2_f = static_cast<float>(beta2);
    const float wd_f = static_cast<float>(weight_decay);
    const float eps_f = static_cast<float>(eps);
    const uint8_t max_f = maximize ? 1 : 0;
    [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:2];
    [computeEncoder setBytes:&beta1_f length:sizeof(float) atIndex:3];
    [computeEncoder setBytes:&beta2_f length:sizeof(float) atIndex:4];
    [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:5];
    [computeEncoder setBytes:&eps_f length:sizeof(float) atIndex:6];
    [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:7];
    set_optional_float_buffer(computeEncoder, grad_scale, 8);
    set_optional_float_buffer(computeEncoder, found_inf, 9);
    MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
    const uint32_t maxThreadsPerGroup = [PSO maxTotalThreadsPerThreadgroup];
    MTLSize threadGroupSize =
        MTLSizeMake(std::min<uint32_t>(maxThreadsPerGroup, kThreadGroupSize),
                    1, 1);
    [computeEncoder dispatchThreadgroups:gridSize
                    threadsPerThreadgroup:threadGroupSize];
  }
}

// One parameter per launch, with every tensor bound directly.  SGD has no
// foreach-free kernel of its own, so this form is also what a GPU image
// without argument encoders uses for every parameter of a foreach call.
void dispatch_fused_sgd_single(
    id<MTLComputeCommandEncoder> computeEncoder,
    const std::string& dtype,
    const Tensor& param,
    const Tensor& grad,
    const Tensor& momentum_buffer,
    bool has_momentum,
    double weight_decay,
    double lr,
    const std::optional<Tensor>& lr_tensor,
    double momentum_coeff,
    double dampening,
    bool nesterov,
    bool maximize,
    bool is_first_step,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf) {
  if (param.numel() == 0) {
    return;
  }
  const std::string kernel =
      (has_momentum ? "fused_sgd_momentum_single_" : "fused_sgd_single_") +
      dtype;
  id<MTLComputePipelineState> PSO =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(kernel);
  [computeEncoder setComputePipelineState:PSO];
  unsigned idx = 0;
  [computeEncoder setBuffer:mtl_buffer_of(param) offset:0 atIndex:idx++];
  [computeEncoder setBuffer:mtl_buffer_of(grad) offset:0 atIndex:idx++];
  if (has_momentum) {
    [computeEncoder setBuffer:mtl_buffer_of(momentum_buffer)
                       offset:0
                      atIndex:idx++];
  }
  const float wd_f = static_cast<float>(weight_decay);
  [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:idx++];
  if (has_momentum) {
    const float mom_f = static_cast<float>(momentum_coeff);
    [computeEncoder setBytes:&mom_f length:sizeof(float) atIndex:idx++];
  }
  if (lr_tensor.has_value() && lr_tensor->defined()) {
    Tensor lr_c = shader_tensor(*lr_tensor);
    [computeEncoder setBuffer:mtl_buffer_of(lr_c) offset:0 atIndex:idx++];
  } else {
    const float lr_f = static_cast<float>(lr);
    [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:idx++];
  }
  if (has_momentum) {
    const float damp_f = static_cast<float>(dampening);
    const uint8_t nest_f = nesterov ? 1 : 0;
    const uint8_t max_f = maximize ? 1 : 0;
    const uint8_t first_f = is_first_step ? 1 : 0;
    [computeEncoder setBytes:&damp_f length:sizeof(float) atIndex:idx++];
    [computeEncoder setBytes:&nest_f length:sizeof(uint8_t) atIndex:idx++];
    [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
    [computeEncoder setBytes:&first_f length:sizeof(uint8_t) atIndex:idx++];
  } else {
    const uint8_t max_f = maximize ? 1 : 0;
    [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
  }
  set_optional_float_buffer(computeEncoder, grad_scale, idx++);
  set_optional_float_buffer(computeEncoder, found_inf, idx++);
  mtl_dispatch1DJob(computeEncoder, PSO,
                    static_cast<NSUInteger>(param.numel()));
}

// Multi-tensor (foreach) fused SGD dispatch.  Uses the same chunked
// threadgroup scheme as the Adam path: up to 32 parameters per launch, each
// split into 64K-element chunks.  With momentum the depth is 3 (params,
// grads, momentum buffers), otherwise 2.
void dispatch_fused_sgd_multi(
    id<MTLComputeCommandEncoder> computeEncoder,
    const std::vector<Tensor>& params,
    const std::vector<Tensor>& grads,
    const std::vector<Tensor>& momentum_buffers,
    bool momentum,
    double weight_decay,
    double lr,
    const std::optional<Tensor>& lr_tensor,
    double momentum_coeff,
    double dampening,
    bool nesterov,
    bool maximize,
    bool is_first_step,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf) {
  constexpr int64_t kChunkSize = 65536;
  constexpr int64_t kmaxThreadGroups = 32;
  constexpr int64_t kmaxTensors = 32;
  constexpr int64_t kThreadGroupSize = 512;
  const int64_t depth = momentum ? 3 : 2;
  const size_t num_tensors = params.size();
  if (num_tensors == 0) {
    return;
  }

  const std::string dtype = scalarToMetalTypeString(params[0].dtype());
  const std::string kernel =
      (momentum ? "fused_sgd_momentum_" : "fused_sgd_") + dtype;
  MTLFunction_t func =
      MetalShaderLibrary::getBundledLibrary().getFunction(kernel);
  if (!argumentEncodersAvailable(func)) {
    [func release];
    for (size_t i = 0; i < num_tensors; ++i) {
      dispatch_fused_sgd_single(
          computeEncoder, dtype, params[i], grads[i],
          momentum ? momentum_buffers[i] : Tensor(), momentum, weight_decay, lr,
          lr_tensor, momentum_coeff, dampening, nesterov, maximize,
          is_first_step, grad_scale, found_inf);
    }
    return;
  }
  id<MTLComputePipelineState> PSO =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(kernel);
  [computeEncoder setComputePipelineState:PSO];
  ForeachArguments arguments(computeEncoder, func,
                             MPSDevice::getInstance()->device());
  [func release];

  int64_t tensor_loc = 0;
  int64_t threadgroup_loc = 0;
  MPSMetadataArguments metadata_arguments;
  memset(&metadata_arguments, 0, sizeof(metadata_arguments));
  id<MTLBuffer> tensorArgumentBuffer = nil;

  for (size_t tensor_index = 0; tensor_index < num_tensors; ++tensor_index) {
    const int64_t numel = params[tensor_index].numel();
    if (numel == 0) {
      continue;
    }
    if (tensorArgumentBuffer == nil) {
      tensorArgumentBuffer = arguments.buffer();
    }
    for (int64_t d = 0; d < depth; ++d) {
      const Tensor* t = (d == 0)   ? &params[tensor_index]
                        : (d == 1) ? &grads[tensor_index]
                                   : &momentum_buffers[tensor_index];
      arguments.bind(static_cast<NSUInteger>(d * kmaxTensors + tensor_loc),
                     mtl_buffer_of(*t));
    }
    metadata_arguments.numels[tensor_loc] = static_cast<uint64_t>(numel);
    tensor_loc++;

    const int64_t chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      metadata_arguments.threadgroup_to_tensor[threadgroup_loc] =
          static_cast<uint64_t>(tensor_loc - 1);
      metadata_arguments.threadgroup_to_chunk[threadgroup_loc] =
          static_cast<uint64_t>(chunk);
      threadgroup_loc++;

      const bool tensor_full =
          tensor_loc == kmaxTensors && chunk == chunks - 1;
      const bool blocks_full = threadgroup_loc == kmaxThreadGroups;

      if (tensor_full || blocks_full) {
        [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
        [computeEncoder setBytes:&metadata_arguments
                          length:sizeof(MPSMetadataArguments)
                           atIndex:1];
        const float wd_f = static_cast<float>(weight_decay);
        [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:2];
        unsigned idx = 3;
        if (momentum) {
          const float mom_f = static_cast<float>(momentum_coeff);
          [computeEncoder setBytes:&mom_f length:sizeof(float) atIndex:idx++];
        }
        if (lr_tensor.has_value() && lr_tensor->defined()) {
          Tensor lr_c = shader_tensor(*lr_tensor);
          [computeEncoder setBuffer:mtl_buffer_of(lr_c) offset:0 atIndex:idx++];
        } else {
          const float lr_f = static_cast<float>(lr);
          [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:idx++];
        }
        if (momentum) {
          const float damp_f = static_cast<float>(dampening);
          const uint8_t nest_f = nesterov ? 1 : 0;
          const uint8_t max_f = maximize ? 1 : 0;
          const uint8_t first_f = is_first_step ? 1 : 0;
          [computeEncoder setBytes:&damp_f length:sizeof(float) atIndex:idx++];
          [computeEncoder setBytes:&nest_f length:sizeof(uint8_t) atIndex:idx++];
          [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
          [computeEncoder setBytes:&first_f length:sizeof(uint8_t) atIndex:idx++];
        } else {
          const uint8_t max_f = maximize ? 1 : 0;
          [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
        }
        set_optional_float_buffer(computeEncoder, grad_scale, idx++);
        set_optional_float_buffer(computeEncoder, found_inf, idx++);

        MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
        const uint32_t maxThreadsPerGroup =
            [PSO maxTotalThreadsPerThreadgroup];
        MTLSize threadGroupSize = MTLSizeMake(
            std::min<uint32_t>(maxThreadsPerGroup, kThreadGroupSize), 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadGroupSize];

        threadgroup_loc = 0;
        if (chunk == chunks - 1) {
          tensor_loc = 0;
          tensorArgumentBuffer = arguments.buffer();
        } else {
          // Reuse the current tensor for the remaining chunks.
          metadata_arguments.numels[0] =
              metadata_arguments.numels[tensor_loc - 1];
          tensorArgumentBuffer = arguments.buffer();
          for (int64_t d = 0; d < depth; ++d) {
            const Tensor* t = (d == 0)   ? &params[tensor_index]
                              : (d == 1) ? &grads[tensor_index]
                                         : &momentum_buffers[tensor_index];
            arguments.bind(static_cast<NSUInteger>(d * kmaxTensors),
                           mtl_buffer_of(*t));
          }
          tensor_loc = 1;
        }
      }
    }
  }

  if (threadgroup_loc != 0) {
    [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
    [computeEncoder setBytes:&metadata_arguments
                      length:sizeof(MPSMetadataArguments)
                       atIndex:1];
    const float wd_f = static_cast<float>(weight_decay);
    [computeEncoder setBytes:&wd_f length:sizeof(float) atIndex:2];
    unsigned idx = 3;
    if (momentum) {
      const float mom_f = static_cast<float>(momentum_coeff);
      [computeEncoder setBytes:&mom_f length:sizeof(float) atIndex:idx++];
    }
    if (lr_tensor.has_value() && lr_tensor->defined()) {
      Tensor lr_c = shader_tensor(*lr_tensor);
      [computeEncoder setBuffer:mtl_buffer_of(lr_c) offset:0 atIndex:idx++];
    } else {
      const float lr_f = static_cast<float>(lr);
      [computeEncoder setBytes:&lr_f length:sizeof(float) atIndex:idx++];
    }
    if (momentum) {
      const float damp_f = static_cast<float>(dampening);
      const uint8_t nest_f = nesterov ? 1 : 0;
      const uint8_t max_f = maximize ? 1 : 0;
      const uint8_t first_f = is_first_step ? 1 : 0;
      [computeEncoder setBytes:&damp_f length:sizeof(float) atIndex:idx++];
      [computeEncoder setBytes:&nest_f length:sizeof(uint8_t) atIndex:idx++];
      [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
      [computeEncoder setBytes:&first_f length:sizeof(uint8_t) atIndex:idx++];
    } else {
      const uint8_t max_f = maximize ? 1 : 0;
      [computeEncoder setBytes:&max_f length:sizeof(uint8_t) atIndex:idx++];
    }
    set_optional_float_buffer(computeEncoder, grad_scale, idx++);
    set_optional_float_buffer(computeEncoder, found_inf, idx++);

    MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
    const uint32_t maxThreadsPerGroup = [PSO maxTotalThreadsPerThreadgroup];
    MTLSize threadGroupSize =
        MTLSizeMake(std::min<uint32_t>(maxThreadsPerGroup, kThreadGroupSize),
                    1, 1);
    [computeEncoder dispatchThreadgroups:gridSize
                    threadsPerThreadgroup:threadGroupSize];
  }
}

void fused_adam_impl(std::vector<Tensor> params,
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
  const char* op = adamw ? "_fused_adamw_" : "_fused_adam_";
  const size_t n = params.size();
  if (n == 0) {
    return;
  }
  if (grads.size() != n || exp_avgs.size() != n || exp_avg_sqs.size() != n ||
      state_steps.size() != n || (amsgrad && max_exp_avg_sqs.size() != n)) {
    TP_THROW(ValueError, op, ": parameter list sizes mismatch");
  }
  for (size_t i = 0; i < n; ++i) {
    is_mps_tensor(params[i], op);
    is_mps_tensor(grads[i], op);
    is_mps_tensor(exp_avgs[i], op);
    is_mps_tensor(exp_avg_sqs[i], op);
    if (amsgrad) {
      is_mps_tensor(max_exp_avg_sqs[i], op);
    }
    is_mps_tensor(state_steps[i], op);
  }

  // Collect non-empty, shader-ready tensors.
  std::vector<Tensor> p_nz, g_nz, ea_nz, es_nz, me_nz, st_nz;
  for (size_t i = 0; i < n; ++i) {
    if (params[i].numel() == 0) {
      continue;
    }
    p_nz.push_back(shader_tensor(params[i]));
    g_nz.push_back(shader_tensor(grads[i]));
    ea_nz.push_back(shader_tensor(exp_avgs[i]));
    es_nz.push_back(shader_tensor(exp_avg_sqs[i]));
    if (amsgrad) {
      me_nz.push_back(shader_tensor(max_exp_avg_sqs[i]));
    }
    st_nz.push_back(shader_tensor(state_steps[i]));
  }
  if (p_nz.empty()) {
    return;
  }
  const std::string dtype = scalarToMetalTypeString(p_nz[0].dtype());
  if (dtype != "float" && dtype != "half") {
    TP_THROW(RuntimeError,
             op, ": MPS fused optimizer supports float32/float16 only");
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  // The caller advances the state steps before invoking this update.
  run_serialized(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      if (p_nz.size() > 1) {
        dispatch_fused_adam_multi(
            computeEncoder, p_nz, g_nz, ea_nz, es_nz, me_nz, st_nz, amsgrad,
            adamw, lr, beta1, beta2, weight_decay, eps, maximize, grad_scale,
            found_inf);
      } else {
        dispatch_fused_adam_single(computeEncoder, dtype, p_nz[0], g_nz[0],
                                   ea_nz[0], es_nz[0],
                                   amsgrad ? me_nz[0] : Tensor(), st_nz[0],
                                   amsgrad, adamw, lr, beta1, beta2,
                                   weight_decay, eps, maximize, grad_scale,
                                   found_inf);
      }
      mpsStream->endKernelCoalescing();
    }
  });
}
void fused_adam_mps(std::vector<Tensor> params,
                    std::vector<Tensor> grads,
                    std::vector<Tensor> exp_avgs,
                    std::vector<Tensor> exp_avg_sqs,
                    std::vector<Tensor> max_exp_avg_sqs,
                    const std::vector<Tensor>& state_steps,
                    double lr,
                    double beta1,
                    double beta2,
                    double weight_decay,
                    double eps,
                    bool amsgrad,
                    bool maximize,
                    const std::optional<Tensor>& grad_scale,
                    const std::optional<Tensor>& found_inf,
                    bool exact) {
  (void)exact;
  fused_adam_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
                  max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                  weight_decay, eps, amsgrad, maximize, false, grad_scale,
                  found_inf);
}

void fused_adamw_mps(std::vector<Tensor> params,
                     std::vector<Tensor> grads,
                     std::vector<Tensor> exp_avgs,
                     std::vector<Tensor> exp_avg_sqs,
                     std::vector<Tensor> max_exp_avg_sqs,
                     const std::vector<Tensor>& state_steps,
                     double lr,
                     double beta1,
                     double beta2,
                     double weight_decay,
                     double eps,
                     bool amsgrad,
                     bool maximize,
                     const std::optional<Tensor>& grad_scale,
                     const std::optional<Tensor>& found_inf,
                     bool exact) {
  (void)exact;
  fused_adam_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
                  max_exp_avg_sqs, state_steps, lr, beta1, beta2,
                  weight_decay, eps, amsgrad, maximize, true, grad_scale,
                  found_inf);
}

void fused_adam_tensor_lr_mps(std::vector<Tensor> params,
                              std::vector<Tensor> grads,
                              std::vector<Tensor> exp_avgs,
                              std::vector<Tensor> exp_avg_sqs,
                              std::vector<Tensor> max_exp_avg_sqs,
                              const std::vector<Tensor>& state_steps,
                              const Tensor& lr,
                              double beta1,
                              double beta2,
                              double weight_decay,
                              double eps,
                              bool amsgrad,
                              bool maximize,
                              const std::optional<Tensor>& grad_scale,
                              const std::optional<Tensor>& found_inf,
                              bool exact) {
  (void)exact;
  fused_adam_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
                  max_exp_avg_sqs, state_steps, read_scalar_lr(lr), beta1,
                  beta2, weight_decay, eps, amsgrad, maximize, false,
                  grad_scale, found_inf);
}

void fused_adamw_tensor_lr_mps(std::vector<Tensor> params,
                               std::vector<Tensor> grads,
                               std::vector<Tensor> exp_avgs,
                               std::vector<Tensor> exp_avg_sqs,
                               std::vector<Tensor> max_exp_avg_sqs,
                               const std::vector<Tensor>& state_steps,
                               const Tensor& lr,
                               double beta1,
                               double beta2,
                               double weight_decay,
                               double eps,
                               bool amsgrad,
                               bool maximize,
                               const std::optional<Tensor>& grad_scale,
                               const std::optional<Tensor>& found_inf,
                               bool exact) {
  (void)exact;
  fused_adam_impl(std::move(params), grads, exp_avgs, exp_avg_sqs,
                  max_exp_avg_sqs, state_steps, read_scalar_lr(lr), beta1,
                  beta2, weight_decay, eps, amsgrad, maximize, true,
                  grad_scale, found_inf);
}

void fused_sgd_impl(std::vector<Tensor> params,
                    const std::vector<Tensor>& grads,
                    const std::vector<Tensor>& momentum_buffers,
                    double weight_decay,
                    double momentum,
                    double lr,
                    const std::optional<Tensor>& lr_tensor,
                    double dampening,
                    bool nesterov,
                    bool maximize,
                    bool is_first_step,
                    const std::optional<Tensor>& grad_scale,
                    const std::optional<Tensor>& found_inf) {
  const char* op = "_fused_sgd_";
  const size_t n = params.size();
  if (n == 0) {
    return;
  }
  if (grads.size() != n) {
    TP_THROW(ValueError, op, ": parameter and gradient list sizes mismatch");
  }
  if (!momentum_buffers.empty() && momentum_buffers.size() != n) {
    TP_THROW(ValueError, op, ": momentum buffer list size mismatch");
  }
  for (size_t i = 0; i < n; ++i) {
    is_mps_tensor(params[i], op);
    is_mps_tensor(grads[i], op);
  }
  for (size_t i = 0; i < momentum_buffers.size(); ++i) {
    is_mps_tensor(momentum_buffers[i], op);
  }

  const bool use_momentum = !momentum_buffers.empty();
  if (use_momentum && momentum <= 0) {
    TP_THROW(RuntimeError,
             op, ": momentum must be > 0 when momentum buffers are provided");
  }
  if (!use_momentum && momentum != 0) {
    TP_THROW(RuntimeError,
             op, ": momentum must be 0 when momentum buffers are empty");
  }

  std::vector<Tensor> p_nz, g_nz, m_nz;
  for (size_t i = 0; i < n; ++i) {
    if (params[i].numel() == 0) {
      continue;
    }
    p_nz.push_back(shader_tensor(params[i]));
    g_nz.push_back(shader_tensor(grads[i]));
    if (use_momentum) {
      m_nz.push_back(shader_tensor(momentum_buffers[i]));
    }
  }
  if (p_nz.empty()) {
    return;
  }
  const std::string dtype = scalarToMetalTypeString(p_nz[0].dtype());
  if (dtype != "float" && dtype != "half" && dtype != "bfloat") {
    TP_THROW(RuntimeError,
             op, ": MPS fused SGD supports float32/float16/bfloat16");
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  run_serialized(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      dispatch_fused_sgd_multi(computeEncoder, p_nz, g_nz, m_nz, use_momentum,
                               weight_decay, lr, lr_tensor, momentum,
                               dampening, nesterov, maximize, is_first_step,
                               grad_scale, found_inf);
      mpsStream->endKernelCoalescing();
    }
  });
}

void fused_sgd_mps(std::vector<Tensor> params,
                   std::vector<Tensor> grads,
                   std::vector<Tensor> momentum_buffers,
                   double weight_decay,
                   double momentum,
                   double lr,
                   double dampening,
                   bool nesterov,
                   bool maximize,
                   bool is_first_step,
                   const std::optional<Tensor>& grad_scale,
                   const std::optional<Tensor>& found_inf) {
  fused_sgd_impl(std::move(params), std::move(grads),
                 std::move(momentum_buffers), weight_decay, momentum, lr,
                 std::nullopt, dampening, nesterov, maximize, is_first_step,
                 grad_scale, found_inf);
}

void fused_sgd_tensor_lr_mps(std::vector<Tensor> params,
                             std::vector<Tensor> grads,
                             std::vector<Tensor> momentum_buffers,
                             double weight_decay,
                             double momentum,
                             const Tensor& lr,
                             double dampening,
                             bool nesterov,
                             bool maximize,
                             bool is_first_step,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf) {
  if (lr.device().is_cpu()) {
    fused_sgd_impl(std::move(params), std::move(grads),
                   std::move(momentum_buffers), weight_decay, momentum,
                   read_scalar_lr(lr), std::nullopt, dampening, nesterov,
                   maximize, is_first_step, grad_scale, found_inf);
    return;
  }
  if (lr.numel() != 1) {
    TP_THROW(RuntimeError,
             "_fused_sgd_.tensor_lr: lr must be a scalar tensor");
  }
  fused_sgd_impl(std::move(params), std::move(grads),
                 std::move(momentum_buffers), weight_decay, momentum, 0.0, lr,
                 dampening, nesterov, maximize, is_first_step, grad_scale,
                 found_inf);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, FusedOptimizerKernels) {
  m.impl("_fused_adam_", fused_adam_mps);
  m.impl("_fused_adamw_", fused_adamw_mps);
  m.impl("_fused_adam_.tensor_lr", fused_adam_tensor_lr_mps);
  m.impl("_fused_adamw_.tensor_lr", fused_adamw_tensor_lr_mps);
  m.impl("_fused_sgd_", fused_sgd_mps);
  m.impl("_fused_sgd_.tensor_lr", fused_sgd_tensor_lr_mps);
}

} // namespace mps
} // namespace tensorplay
