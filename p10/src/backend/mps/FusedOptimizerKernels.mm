#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

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

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      for (size_t i = 0; i < n; ++i) {
        Tensor p = shader_tensor(params[i]);
        Tensor g = shader_tensor(grads[i]);
        Tensor ea = shader_tensor(exp_avgs[i]);
        Tensor es = shader_tensor(exp_avg_sqs[i]);
        Tensor me = amsgrad ? shader_tensor(max_exp_avg_sqs[i]) : Tensor();
        Tensor st = shader_tensor(state_steps[i]);

        const std::string dtype = scalarToMetalTypeString(p.dtype());
        if (dtype != "float" && dtype != "half") {
          TP_THROW(RuntimeError,
                   op, ": MPS fused optimizer supports float32/float16 only");
        }
        const std::string kernel =
            (amsgrad ? (adamw ? "fused_adamw_amsgrad_" : "fused_adam_amsgrad_")
                     : (adamw ? "fused_adamw_" : "fused_adam_")) +
            dtype + "_float";
        id<MTLComputePipelineState> PSO =
            MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
                kernel);
        [computeEncoder setComputePipelineState:PSO];

        unsigned idx = 0;
        [computeEncoder setBuffer:mtl_buffer_of(p) offset:0 atIndex:idx++];
        [computeEncoder setBuffer:mtl_buffer_of(g) offset:0 atIndex:idx++];
        [computeEncoder setBuffer:mtl_buffer_of(ea) offset:0 atIndex:idx++];
        [computeEncoder setBuffer:mtl_buffer_of(es) offset:0 atIndex:idx++];
        if (amsgrad) {
          [computeEncoder setBuffer:mtl_buffer_of(me) offset:0 atIndex:idx++];
        }
        [computeEncoder setBuffer:mtl_buffer_of(st) offset:0 atIndex:idx++];
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
                          static_cast<NSUInteger>(p.numel()));
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

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, FusedOptimizerKernels) {
  m.impl("_fused_adam_", fused_adam_mps);
  m.impl("_fused_adamw_", fused_adamw_mps);
  m.impl("_fused_adam_.tensor_lr", fused_adam_tensor_lr_mps);
  m.impl("_fused_adamw_.tensor_lr", fused_adamw_tensor_lr_mps);
}

} // namespace mps
} // namespace tensorplay