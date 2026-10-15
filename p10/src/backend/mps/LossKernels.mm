#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

// Defined in ReductionKernels.mm.
Tensor sum_kernel(const Tensor& self, DType dtype);
Tensor mean_kernel(const Tensor& self, DType dtype);

namespace {

Tensor scalarToTensor(const Scalar& s, DType dtype, const Device& device) {
  Tensor t = Tensor({1}, dtype, device);
  fill_kernel(t, s);
  return t;
}

MPSBinaryOpBlock addBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock subBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock mulBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock divBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock maximumBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                MPSGraphTensor* b) {
  return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock leBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                            MPSGraphTensor* b) {
  return [g lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSUnaryOpBlock absBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g absoluteWithTensor:x name:nil];
};

MPSUnaryOpBlock squareBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g squareWithTensor:x name:nil];
};

MPSUnaryOpBlock logBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g logarithmWithTensor:x name:nil];
};

MPSUnaryOpBlock negBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g negativeWithTensor:x name:nil];
};

MPSUnaryOpBlock signBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g signWithTensor:x name:nil];
};

// reduction: 0=none, 1=mean, 2=sum.
Tensor loss_reduce(const Tensor& x, int64_t reduction) {
  if (reduction == 0) return x;
  if (reduction == 1) return mean_kernel(x, DType::Undefined);
  if (reduction == 2) return sum_kernel(x, DType::Undefined);
  TP_THROW(ValueError, "Invalid reduction mode");
}

// Scales a gradient by 1/numel for mean reduction (reduction == 1).
Tensor scale_grad(const Tensor& g, int64_t reduction, int64_t numel) {
  if (reduction == 1) {
    return binaryOp(g, scalarToTensor(1.0 / static_cast<double>(numel),
                                      g.dtype(), g.device()),
                    mulBlock);
  }
  return g;
}

// ---------------------------------------------------------------------------
// MSE loss: L = (x - y)^2
// ---------------------------------------------------------------------------

Tensor mse_loss_kernel(const Tensor& input, const Tensor& target,
                       int64_t reduction) {
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor sq = unaryOp(diff, squareBlock);
  return loss_reduce(sq, reduction);
}

Tensor mse_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                const Tensor& target, int64_t reduction) {
  Tensor diff = binaryOp(input, target, subBlock);
  // dL/dx = 2 * (x - y) * grad_output (mean: /numel)
  const double norm = (reduction == 1)
      ? 2.0 / static_cast<double>(input.numel())
      : 2.0;
  Tensor scaled = binaryOp(diff, scalarToTensor(norm, diff.dtype(),
                                                 diff.device()), mulBlock);
  return binaryOp(scaled, grad_output, mulBlock);
}

// ---------------------------------------------------------------------------
// L1 loss: mean(|x - y|)
// ---------------------------------------------------------------------------

Tensor l1_loss_kernel(const Tensor& input, const Tensor& target) {
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor absd = unaryOp(diff, absBlock);
  return mean_kernel(absd, DType::Undefined);
}

// ---------------------------------------------------------------------------
// Smooth L1 loss: z = |x-y|; z < beta ? 0.5 z^2 / beta : z - 0.5 beta
// ---------------------------------------------------------------------------

Tensor smooth_l1_loss_kernel(const Tensor& input, const Tensor& target,
                             int64_t reduction, double beta) {
  if (beta < 0) {
    TP_THROW(ValueError,
             "smooth_l1_loss does not support negative values for beta.");
  }
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor absd = unaryOp(diff, absBlock);
  Tensor beta_t = scalarToTensor(beta, diff.dtype(), diff.device());
  Tensor half_over_beta =
      scalarToTensor(0.5 / beta, diff.dtype(), diff.device());
  Tensor half_beta = scalarToTensor(0.5 * beta, diff.dtype(), diff.device());
  Tensor quad = binaryOp(binaryOp(diff, diff, mulBlock), half_over_beta,
                        mulBlock);
  Tensor lin = binaryOp(absd, half_beta, subBlock);
  Tensor cond = binaryOp(absd, beta_t, leBlock);
  return loss_reduce(whereOp(cond, quad, lin), reduction);
}

Tensor smooth_l1_loss_backward_kernel(const Tensor& grad_output,
                                      const Tensor& input,
                                      const Tensor& target, int64_t reduction,
                                      double beta) {
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor absd = unaryOp(diff, absBlock);
  Tensor beta_t = scalarToTensor(beta, diff.dtype(), diff.device());
  Tensor cond = binaryOp(absd, beta_t, leBlock);
  Tensor g_inner =
      whereOp(cond, binaryOp(diff, beta_t, divBlock), unaryOp(diff, signBlock));
  Tensor g = binaryOp(g_inner, grad_output, mulBlock);
  return scale_grad(g, reduction, input.numel());
}

// ---------------------------------------------------------------------------
// Huber loss: z = |x-y|; z < delta ? 0.5 z^2 : delta (z - 0.5 delta)
// ---------------------------------------------------------------------------

Tensor huber_loss_kernel(const Tensor& input, const Tensor& target,
                         int64_t reduction, double delta) {
  if (delta <= 0) {
    TP_THROW(ValueError,
             "huber_loss does not support non-positive values for delta.");
  }
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor absd = unaryOp(diff, absBlock);
  Tensor delta_t = scalarToTensor(delta, diff.dtype(), diff.device());
  Tensor half_delta = scalarToTensor(0.5 * delta, diff.dtype(), diff.device());
  Tensor quad = binaryOp(binaryOp(diff, diff, mulBlock),
                         scalarToTensor(0.5, diff.dtype(), diff.device()),
                         mulBlock);
  Tensor lin = binaryOp(delta_t, binaryOp(absd, half_delta, subBlock), mulBlock);
  Tensor cond = binaryOp(absd, delta_t, leBlock);
  return loss_reduce(whereOp(cond, quad, lin), reduction);
}

Tensor huber_loss_backward_kernel(const Tensor& grad_output,
                                  const Tensor& input, const Tensor& target,
                                  int64_t reduction, double delta) {
  Tensor diff = binaryOp(input, target, subBlock);
  Tensor absd = unaryOp(diff, absBlock);
  Tensor delta_t = scalarToTensor(delta, diff.dtype(), diff.device());
  Tensor cond = binaryOp(absd, delta_t, leBlock);
  Tensor scaled_sign = binaryOp(delta_t, unaryOp(diff, signBlock), mulBlock);
  Tensor g_inner = whereOp(cond, diff, scaled_sign);
  Tensor g = binaryOp(g_inner, grad_output, mulBlock);
  return scale_grad(g, reduction, input.numel());
}

// ---------------------------------------------------------------------------
// Binary cross entropy: L = -w (y ln(x) + (1-y) ln(1-x))
// ---------------------------------------------------------------------------

Tensor binary_cross_entropy_kernel(const Tensor& input, const Tensor& target,
                                   const std::optional<Tensor>& weight,
                                   int64_t reduction) {
  Tensor one = scalarToTensor(1.0, input.dtype(), input.device());
  // -100 is the hard clamp limit used by the framework-wide BCE spec.
  Tensor neg100 = scalarToTensor(-100.0, input.dtype(), input.device());

  Tensor logInput = unaryOp(input, logBlock);
  Tensor clampedLogInput = binaryOp(logInput, neg100, maximumBlock);
  Tensor oneMinusInput = binaryOp(one, input, subBlock);
  Tensor log1Minus = unaryOp(oneMinusInput, logBlock);
  Tensor clampedLog1 = binaryOp(log1Minus, neg100, maximumBlock);

  Tensor yLog = binaryOp(target, clampedLogInput, mulBlock);
  Tensor oneMinusTarget = binaryOp(one, target, subBlock);
  Tensor y1Log1 = binaryOp(oneMinusTarget, clampedLog1, mulBlock);
  Tensor sum = binaryOp(y1Log1, yLog, addBlock);
  Tensor loss = unaryOp(sum, negBlock);

  if (weight.has_value() && weight->defined()) {
    loss = binaryOp(loss, *weight, mulBlock);
  }
  return loss_reduce(loss, reduction);
}

Tensor binary_cross_entropy_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction) {
  // dL/dx = w * (x - y) / max(x * (1-x), eps)
  Tensor one = scalarToTensor(1.0, input.dtype(), input.device());
  Tensor epsilon = scalarToTensor(1e-12, input.dtype(), input.device());
  Tensor oneMinusInput = binaryOp(one, input, subBlock);
  Tensor xTimes1x = binaryOp(input, oneMinusInput, mulBlock);
  Tensor denom = binaryOp(xTimes1x, epsilon, maximumBlock);
  Tensor xMinusY = binaryOp(input, target, subBlock);
  Tensor ratio = binaryOp(xMinusY, denom, divBlock);
  Tensor g = binaryOp(ratio, grad_output, mulBlock);
  if (weight.has_value() && weight->defined()) {
    g = binaryOp(g, *weight, mulBlock);
  }
  return scale_grad(g, reduction, input.numel());
}

// ---------------------------------------------------------------------------
// NLL loss (host-side through the shared-memory mapping, mirroring the CPU
// kernel semantics: input (N, C) / (C), target (N) / 0-d, optional class
// weights, ignore_index and mean/sum reduction with total_weight).
// ---------------------------------------------------------------------------

template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss_impl(const Tensor& input_arg,
                                         const Tensor& target_arg,
                                         const std::optional<Tensor>& weight_arg,
                                         int64_t reduction,
                                         int64_t ignore_index) {
  const bool no_batch = input_arg.dim() == 1;
  const Tensor input = (no_batch ? input_arg.unsqueeze(0) : input_arg).contiguous();
  const Tensor target = (no_batch ? target_arg.reshape({1}) : target_arg)
                            .to(DType::Int64).contiguous();
  const int64_t n_batch = input.size(0);
  const int64_t n_classes = input.size(1);
  if (target.numel() != n_batch) {
    TP_THROW(ValueError, "nll_loss: expected target of ", n_batch,
             " elements, got ", target.numel());
  }
  Tensor weight;
  if (weight_arg.has_value() && weight_arg->defined()) {
    if (weight_arg->numel() != n_classes) {
      TP_THROW(RuntimeError, "weight tensor should be defined either for all ",
               n_classes, " classes or no classes");
    }
    weight = weight_arg->to(input.dtype()).contiguous();
  }

  const scalar_t* input_data = input.data_ptr<scalar_t>();
  const int64_t* target_data = target.data_ptr<int64_t>();
  const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;

  std::vector<scalar_t> output_data(static_cast<size_t>(n_batch));
  double total_weight = 0;
  for (int64_t i = 0; i < n_batch; ++i) {
    const int64_t t = target_data[i];
    if (t == ignore_index) {
      output_data[static_cast<size_t>(i)] = 0;
      continue;
    }
    if (t < 0 || t >= n_classes) {
      TP_THROW(IndexError, "Target ", t, " is out of bounds.");
    }
    const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
    output_data[static_cast<size_t>(i)] =
        static_cast<scalar_t>(-static_cast<double>(input_data[i * n_classes + t]) * w);
    total_weight += w;
  }

  const DType dt = input.dtype();
  if (reduction == 0) {
    Tensor per_sample = Tensor({n_batch}, dt, input.device());
    scalar_t* out = per_sample.data_ptr<scalar_t>();
    for (int64_t i = 0; i < n_batch; ++i) out[i] = output_data[i];
    if (no_batch) per_sample = per_sample.reshape({});
    // Unreduced batched losses carry no normalizer.
    const double reported = no_batch ? total_weight : 0.0;
    Tensor tw = Tensor({}, dt, input.device());
    fill_kernel(tw, reported);
    return std::make_tuple(per_sample, tw);
  }
  double sum = 0;
  for (scalar_t x : output_data) sum += static_cast<double>(x);
  if (reduction == 1) {
    sum /= total_weight;  // every target ignored: 0 / 0 is NaN
  } else if (reduction != 2) {
    TP_THROW(ValueError, "Invalid reduction mode");
  }
  Tensor out = Tensor({}, dt, input.device());
  fill_kernel(out, sum);
  Tensor tw = Tensor({}, dt, input.device());
  fill_kernel(tw, total_weight);
  return std::make_tuple(out, tw);
}

std::tuple<Tensor, Tensor> nll_loss_kernel(
    const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  if (input.dtype() == DType::Float32) {
    return nll_loss_impl<float>(input, target, weight, reduction, ignore_index);
  }
  if (input.dtype() == DType::Float64) {
    return nll_loss_impl<double>(input, target, weight, reduction, ignore_index);
  }
  TP_THROW(NotImplementedError, "nll_loss only supports Float32/Float64");
}

template <typename scalar_t>
Tensor nll_loss_backward_impl(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target,
                              const std::optional<Tensor>& weight,
                              int64_t reduction, int64_t ignore_index,
                              const Tensor& total_weight) {
  const int64_t n_batch = input.size(0);
  const int64_t n_classes = input.size(1);

  Tensor grad_input = Tensor::zeros({n_batch, n_classes}, input.dtype(),
                                    input.device());
  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();
  const Tensor target_c = target.contiguous();
  const int64_t* target_data = target_c.data_ptr<int64_t>();
  const scalar_t* weight_data =
      (weight.has_value() && weight->defined())
      ? weight->to(input.dtype()).contiguous().data_ptr<scalar_t>()
      : nullptr;

  double tw = 0;
  if (total_weight.defined()) {
    tw = total_weight.item().to<double>();
  } else if (reduction == 1) {
    for (int64_t i = 0; i < n_batch; ++i) {
      const int64_t t = target_data[i];
      if (t != ignore_index) {
        tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
      }
    }
  }

  if (reduction == 0) {
    const Tensor go = grad_output.contiguous();
    const scalar_t* grad_out_data = go.data_ptr<scalar_t>();
    for (int64_t i = 0; i < n_batch; ++i) {
      const int64_t t = target_data[i];
      if (t == ignore_index) continue;
      const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
      grad_input_data[i * n_classes + t] =
          static_cast<scalar_t>(-w * static_cast<double>(grad_out_data[i]));
    }
  } else {
    double grad_val = grad_output.item().to<double>();
    if (reduction == 1 && tw > 0) grad_val /= tw;
    for (int64_t i = 0; i < n_batch; ++i) {
      const int64_t t = target_data[i];
      if (t == ignore_index) continue;
      const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
      grad_input_data[i * n_classes + t] = static_cast<scalar_t>(-w * grad_val);
    }
  }
  return grad_input;
}

Tensor nll_loss_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const Tensor& total_weight) {
  if (input.dtype() == DType::Float32) {
    return nll_loss_backward_impl<float>(grad_output, input, target, weight,
                                         reduction, ignore_index, total_weight);
  }
  if (input.dtype() == DType::Float64) {
    return nll_loss_backward_impl<double>(grad_output, input, target, weight,
                                          reduction, ignore_index, total_weight);
  }
  TP_THROW(NotImplementedError,
           "nll_loss backward only supports Float32/Float64");
}

// (N, C, H, W) input, (N, H, W) target; every spatial position is an
// independent batch row.
template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss2d_impl(const Tensor& input,
                                           const Tensor& target,
                                           const std::optional<Tensor>& weight,
                                           int64_t reduction,
                                           int64_t ignore_index) {
  if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d: Expected 4D input");
  if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d: Expected 3D target");
  const int64_t N = input.size(0), C = input.size(1);
  const int64_t H = input.size(2), W = input.size(3);
  if (target.size(0) != N || target.size(1) != H || target.size(2) != W) {
    TP_THROW(RuntimeError, "nll_loss2d: target shape must match input spatial dims");
  }
  const int64_t rows = N * H * W;

  const Tensor input_c = input.contiguous();
  const Tensor target_c = target.contiguous();
  const scalar_t* input_data = input_c.data_ptr<scalar_t>();
  const int64_t* target_data = target_c.data_ptr<int64_t>();
  const scalar_t* weight_data =
      (weight.has_value() && weight->defined())
      ? weight->to(input.dtype()).contiguous().data_ptr<scalar_t>()
      : nullptr;

  std::vector<scalar_t> output_data(reduction == 0 ? rows : 1);
  double total_weight = 0;
  double sum = 0;

  for (int64_t i = 0; i < rows; ++i) {
    const int64_t t = target_data[i];
    if (t == ignore_index) {
      if (reduction == 0) output_data[i] = 0;
      continue;
    }
    if (t < 0 || t >= C) TP_THROW(RuntimeError, "Target out of bounds");
    const int64_t n = i / (H * W);
    const int64_t pos = i % (H * W);
    const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
    const double loss = -static_cast<double>(input_data[(n * C + t) * H * W + pos]) * w;
    total_weight += w;
    sum += loss;
    if (reduction == 0) output_data[i] = static_cast<scalar_t>(loss);
  }

  const DType dt = input.dtype();
  if (reduction == 0) {
    Tensor per_sample = Tensor({N, H, W}, dt, input.device());
    scalar_t* out = per_sample.data_ptr<scalar_t>();
    for (int64_t i = 0; i < rows; ++i) out[i] = output_data[i];
    Tensor tw = Tensor({}, dt, input.device());
    fill_kernel(tw, 0.0);
    return std::make_tuple(per_sample, tw);
  }
  if (reduction != 1 && reduction != 2) {
    TP_THROW(ValueError, "Invalid reduction mode");
  }
  if (reduction == 1) sum /= total_weight;
  Tensor out = Tensor({}, dt, input.device());
  fill_kernel(out, sum);
  Tensor tw = Tensor({}, dt, input.device());
  fill_kernel(tw, total_weight);
  return std::make_tuple(out, tw);
}

std::tuple<Tensor, Tensor> nll_loss2d_kernel(
    const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  if (input.dtype() == DType::Float32) {
    return nll_loss2d_impl<float>(input, target, weight, reduction, ignore_index);
  }
  if (input.dtype() == DType::Float64) {
    return nll_loss2d_impl<double>(input, target, weight, reduction, ignore_index);
  }
  TP_THROW(NotImplementedError, "nll_loss2d only supports Float32/Float64");
}

template <typename scalar_t>
Tensor nll_loss2d_backward_impl(const Tensor& grad_output, const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight,
                                int64_t reduction, int64_t ignore_index,
                                const Tensor& total_weight) {
  if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 4D input");
  if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 3D target");
  const int64_t N = input.size(0), C = input.size(1);
  const int64_t H = input.size(2), W = input.size(3);
  const int64_t rows = N * H * W;

  Tensor grad_input = Tensor::zeros({N, C, H, W}, input.dtype(), input.device());
  scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();
  const Tensor target_c = target.contiguous();
  const int64_t* target_data = target_c.data_ptr<int64_t>();
  const scalar_t* weight_data =
      (weight.has_value() && weight->defined())
      ? weight->to(input.dtype()).contiguous().data_ptr<scalar_t>()
      : nullptr;

  double tw = 0;
  if (total_weight.defined()) {
    tw = total_weight.item().to<double>();
  } else if (reduction == 1) {
    for (int64_t i = 0; i < rows; ++i) {
      const int64_t t = target_data[i];
      if (t != ignore_index) {
        tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
      }
    }
  }

  const Tensor go = reduction == 0 ? grad_output.contiguous() : Tensor();
  const scalar_t* grad_out_data = reduction == 0 ? go.data_ptr<scalar_t>() : nullptr;
  const double grad_scalar = reduction == 0 ? 0.0 : grad_output.item().to<double>();

  for (int64_t i = 0; i < rows; ++i) {
    const int64_t t = target_data[i];
    if (t == ignore_index) continue;
    if (t < 0 || t >= C) TP_THROW(RuntimeError, "Target out of bounds");
    const int64_t n = i / (H * W);
    const int64_t pos = i % (H * W);
    const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
    double g = reduction == 0 ? static_cast<double>(grad_out_data[i]) : grad_scalar;
    if (reduction == 1 && tw > 0) g /= tw;
    grad_input_data[(n * C + t) * H * W + pos] = static_cast<scalar_t>(-w * g);
  }
  return grad_input;
}

Tensor nll_loss2d_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight, int64_t reduction,
    int64_t ignore_index, const Tensor& total_weight) {
  if (input.dtype() == DType::Float32) {
    return nll_loss2d_backward_impl<float>(grad_output, input, target, weight,
                                           reduction, ignore_index, total_weight);
  }
  if (input.dtype() == DType::Float64) {
    return nll_loss2d_backward_impl<double>(grad_output, input, target, weight,
                                            reduction, ignore_index, total_weight);
  }
  TP_THROW(NotImplementedError,
           "nll_loss2d backward only supports Float32/Float64");
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(MPS, LossKernels) {
  m.impl("mse_loss", mse_loss_kernel);
  m.impl("mse_loss_backward", mse_loss_backward_kernel);
  m.impl("l1_loss", l1_loss_kernel);
  m.impl("smooth_l1_loss", smooth_l1_loss_kernel);
  m.impl("smooth_l1_loss_backward", smooth_l1_loss_backward_kernel);
  m.impl("huber_loss", huber_loss_kernel);
  m.impl("huber_loss_backward", huber_loss_backward_kernel);
  m.impl("binary_cross_entropy", binary_cross_entropy_kernel);
  m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_kernel);
  m.impl("nll_loss", nll_loss_kernel);
  m.impl("nll_loss_backward", nll_loss_backward_kernel);
  m.impl("nll_loss2d", nll_loss2d_kernel);
  m.impl("nll_loss2d_backward", nll_loss2d_backward_kernel);
}

} // namespace mps
} // namespace tensorplay