#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

namespace {

// Returns the number of trailing dimensions covered by `normalized_shape`
// and validates the shape against the input.
std::pair<int64_t, int64_t> split_normalized_shape(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape) {
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  const int64_t input_ndim = input.dim();
  if (norm_ndim > input_ndim) {
    TP_THROW(RuntimeError,
             "rms_norm: normalized_shape is larger than input dims");
  }
  int64_t inner = 1;
  for (int64_t i = 0; i < norm_ndim; ++i) {
    const int64_t dim = input_ndim - norm_ndim + i;
    if (input.size(dim) != normalized_shape[i]) {
      TP_THROW(RuntimeError,
               "rms_norm: input shape mismatch with normalized_shape at dim ",
               dim);
    }
    inner *= normalized_shape[i];
  }
  const int64_t outer = input.numel() / inner;
  return {outer, inner};
}

// Common forward used by both `rms_norm` and `_fused_rms_norm`.  When no
// weight is given an all-ones buffer is substituted so the shader always has
// a valid weight pointer.
std::tuple<Tensor, Tensor> rms_norm_forward(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    std::optional<double> eps_opt) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError, "rms_norm: expected input on MPS device");
  }
  const DType dt = input.dtype();
  if (dt != DType::Float32 && dt != DType::Float16 &&
      dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "rms_norm: MPS supports float32, float16 and bfloat16 only");
  }
  const double eps = eps_opt.has_value()
      ? *eps_opt
      : static_cast<double>(std::numeric_limits<float>::epsilon());
  const auto [M, N] = split_normalized_shape(input, normalized_shape);

  Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  Tensor output = Tensor::empty(
      static_cast<std::vector<int64_t>>(input.shape()), dt, input.device());

  // Reciprocal standard deviation, one value per outer row (broadcastable
  // to the normalized axes).
  const int64_t input_ndim = input.dim();
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  const int64_t axis = input_ndim - norm_ndim;
  std::vector<int64_t> stat_shape;
  for (int64_t i = 0; i < axis; ++i) {
    stat_shape.push_back(input.size(i));
  }
  for (int64_t i = axis; i < input_ndim; ++i) {
    stat_shape.push_back(1);
  }
  Tensor rstd = Tensor::empty(stat_shape, DType::Float32, input.device());

  const bool has_w = weight_opt.has_value() && weight_opt->defined();
  Tensor weight;
  if (has_w) {
    weight = weight_opt->is_contiguous() ? *weight_opt : weight_opt->contiguous();
    if (weight.dtype() != dt) {
      TP_THROW(RuntimeError,
               "rms_norm: weight dtype must match input dtype");
    }
  } else {
    // All-ones weight so `x * inv * w` degenerates to `x * inv`.
    weight = Tensor::full(normalized_shape, 1.0, dt, input.device());
  }

  if (input_c.numel() == 0) {
    return {output, rstd};
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  const float eps_f = static_cast<float>(eps);
  const uint32_t axis_size = static_cast<uint32_t>(N);
  const uint32_t w_stride = 1;
  constexpr uint32_t kLoopedLimit = 4096;
  const bool looped = N > kLoopedLimit;

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = (looped ? "rms_norm_looped_" : "rms_norm_") +
          scalarToMetalTypeString(dt);
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, input_c, weight, output, rstd, eps_f,
                  axis_size, w_stride);

      const auto maxThreadsPerGroup =
          static_cast<size_t>([PSO maxTotalThreadsPerThreadgroup]);
      size_t threadgroup_size = maxThreadsPerGroup;
      if (!looped) {
        constexpr size_t kReads = 4;
        constexpr size_t kSimdSize = 32;
        const size_t tg_needed = (static_cast<size_t>(N) + kReads - 1) / kReads;
        const size_t simds = (tg_needed + kSimdSize - 1) / kSimdSize;
        threadgroup_size = kSimdSize * simds;
        if (threadgroup_size > maxThreadsPerGroup) {
          threadgroup_size = maxThreadsPerGroup;
        }
      }
      const size_t n_threads =
          static_cast<size_t>(M) * threadgroup_size;
      [computeEncoder dispatchThreads:MTLSizeMake(n_threads, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      mpsStream->endKernelCoalescing();
    }
  });

  return {output, rstd};
}

Tensor rms_norm_kernel(const Tensor& input,
                       const std::vector<int64_t>& normalized_shape,
                       const std::optional<Tensor>& weight,
                       std::optional<double> eps) {
  return std::get<0>(rms_norm_forward(input, normalized_shape, weight, eps));
}

std::tuple<Tensor, Tensor> fused_rms_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight,
    std::optional<double> eps) {
  return rms_norm_forward(input, normalized_shape, weight, eps);
}

// Backward for fused RMSNorm, composed from GPU primitives in float32
// accumulation.  Formulas (with x_hat = x * rstd):
//   d_input  = (d_y * w - x_hat * mean(d_y * w * x_hat)) * rstd
//   d_weight = sum(d_y * x_hat) over the outer (non-normalized) dims
std::tuple<Tensor, Tensor> fused_rms_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& rstd,
    const std::optional<Tensor>& weight,
    const std::vector<bool>& output_mask) {
  const int64_t input_ndim = input.dim();
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  const int64_t axis = input_ndim - norm_ndim;
  std::vector<int64_t> inner_dims;
  std::vector<int64_t> outer_dims;
  for (int64_t i = 0; i < input_ndim; ++i) {
    if (i >= axis) {
      inner_dims.push_back(i);
    } else {
      outer_dims.push_back(i);
    }
  }
  int64_t inner_size = 1;
  for (int64_t s : normalized_shape) {
    inner_size *= s;
  }

  const DType dt = input.dtype();
  Tensor compute = input.is_contiguous() ? input : input.contiguous();
  if (compute.dtype() != DType::Float32) {
    compute = compute.to(DType::Float32);
  }
  Tensor g_o = grad_out.is_contiguous() ? grad_out : grad_out.contiguous();
  if (g_o.dtype() != DType::Float32) {
    g_o = g_o.to(DType::Float32);
  }
  Tensor rstd_f = rstd.dtype() == DType::Float32 ? rstd : rstd.to(DType::Float32);
  const bool has_w = weight.has_value() && weight->defined();
  Tensor w_f;
  if (has_w) {
    w_f = weight->is_contiguous() ? *weight : weight->contiguous();
    if (w_f.dtype() != DType::Float32) {
      w_f = w_f.to(DType::Float32);
    }
  }

  Tensor x_hat = ops::mul(compute, rstd_f);
  Tensor grad_x_hat = has_w ? ops::mul(g_o, w_f) : g_o;
  Tensor grad_input;
  if (output_mask[0]) {
    Tensor total = ops::sum(ops::mul(x_hat, grad_x_hat), inner_dims,
                            /*keepdim=*/true);
    Tensor mean_total = ops::div(total, Scalar(static_cast<double>(inner_size)));
    Tensor grad_input_f32 =
        ops::mul(ops::sub(grad_x_hat, ops::mul(x_hat, mean_total)), rstd_f);
    grad_input = grad_input_f32.to(dt);
  }
  Tensor grad_weight;
  if (output_mask[1] && has_w) {
    Tensor grad_weight_f32 =
        ops::sum(ops::mul(g_o, x_hat), outer_dims, /*keepdim=*/false);
    grad_weight = grad_weight_f32.to(w_f.dtype());
  }
  return {grad_input, grad_weight};
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, RMSNormKernels) {
  m.impl("rms_norm", rms_norm_kernel);
  m.impl("_fused_rms_norm", fused_rms_norm_kernel);
  m.impl("_fused_rms_norm_backward", fused_rms_norm_backward_kernel);
}

} // namespace mps
} // namespace tensorplay