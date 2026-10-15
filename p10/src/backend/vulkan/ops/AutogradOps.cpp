#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"

#include <string>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Elementwise kernels defined in sibling translation units.
Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha);
Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha);
Tensor mul_kernel(const Tensor& self, const Tensor& other);
Tensor div_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha);
Tensor rsub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha);
Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor exp_kernel(const Tensor& self);
Tensor expm1_kernel(const Tensor& self);
Tensor log1p_kernel(const Tensor& self);
Tensor erf_kernel(const Tensor& self);
Tensor sigmoid_kernel(const Tensor& self);
Tensor tanh_kernel(const Tensor& self);
Tensor neg_kernel(const Tensor& self);
Tensor gt_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor ge_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor lt_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor le_scalar_kernel(const Tensor& self, const Scalar& other);
Tensor where_kernel(const Tensor& condition, const Tensor& self, const Tensor& other);
Tensor where_scalar_other_kernel(
    const Tensor& condition,
    const Tensor& self,
    const Scalar& other);
Tensor sum_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    DType dtype);
Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape);
Tensor expand_kernel(const Tensor& self, const std::vector<int64_t>& size, bool implicit);
std::vector<Tensor> chunk_kernel(const Tensor& self, int64_t chunks, int64_t dim);
Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim);

namespace {

void validate_float(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16,
      std::string("Vulkan ") + name + " supports Float32/Float16 tensors only");
}

// mask ? a : b, where mask is a Float32 0/1 tensor from a comparison.
Tensor select_by_float_mask(const Tensor& mask, const Tensor& a, const Tensor& b) {
  // where(gt(mask, 0.5), a, b)
  return where_kernel(gt_scalar_kernel(mask, Scalar(0.5)), a, b);
}

} // namespace

// Gradient of thresholding: grad * (self > threshold).
Tensor threshold_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  validate_float(grad_output, "threshold_backward");
  return where_scalar_other_kernel(
      gt_scalar_kernel(self, threshold), grad_output, Scalar(0.0));
}

// Sigmoid backward in terms of the forward output: grad * y * (1 - y).
Tensor sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output) {
  validate_float(grad_output, "sigmoid_backward");
  Tensor one_minus = rsub_scalar_kernel(output, Scalar(1.0), Scalar(1.0));
  return mul_kernel(grad_output, mul_kernel(output, one_minus));
}

// Tanh backward in terms of the forward output: grad * (1 - y^2).
Tensor tanh_backward_kernel(const Tensor& grad_output, const Tensor& output) {
  validate_float(grad_output, "tanh_backward");
  Tensor one_minus_sq =
      rsub_scalar_kernel(mul_kernel(output, output), Scalar(1.0), Scalar(1.0));
  return mul_kernel(grad_output, one_minus_sq);
}

// GELU backward for the erf and tanh approximations.
Tensor gelu_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const std::string& approximate) {
  validate_float(grad_output, "gelu_backward");
  if (approximate == "tanh") {
    // d/dx gelu_tanh(x) = 0.5*(1+th) + 0.5*x*(1-th^2)*dinner,
    // with inner = beta*(x + kappa*x^3), dinner = beta*(1 + 3*kappa*x^2).
    constexpr double kBeta = 0.7978845608028654; // sqrt(2/pi)
    constexpr double kKappa = 0.044715;
    Tensor x_sq = mul_kernel(self, self);
    Tensor inner = mul_scalar_kernel(
        add_kernel(
            self, mul_scalar_kernel(mul_kernel(self, x_sq), kKappa), Scalar(1.0)),
        kBeta);
    Tensor th = tanh_kernel(inner);
    Tensor dth =
        rsub_scalar_kernel(mul_kernel(th, th), Scalar(1.0), Scalar(1.0));
    Tensor dinner = mul_scalar_kernel(
        add_scalar_kernel(
            mul_scalar_kernel(x_sq, 3.0 * kKappa), Scalar(1.0), Scalar(1.0)),
        kBeta);
    Tensor dgelu = add_kernel(
        mul_scalar_kernel(add_scalar_kernel(th, Scalar(1.0), Scalar(1.0)), 0.5),
        mul_scalar_kernel(mul_kernel(mul_kernel(self, dth), dinner), 0.5),
        Scalar(1.0));
    return mul_kernel(grad_output, dgelu);
  }
  // erf form: 0.5*(1+erf(x/sqrt2)) + x * exp(-x^2/2)/sqrt(2*pi)
  constexpr double kSqrt1_2 = 0.7071067811865476; // 1/sqrt(2)
  constexpr double kPdfCoeff = 0.3989422804014327; // 1/sqrt(2*pi)
  Tensor cdf = mul_scalar_kernel(
      add_scalar_kernel(
          erf_kernel(mul_scalar_kernel(self, kSqrt1_2)),
          Scalar(1.0),
          Scalar(1.0)),
      0.5);
  Tensor pdf = mul_scalar_kernel(
      exp_kernel(mul_scalar_kernel(mul_kernel(self, self), -0.5)),
      kPdfCoeff);
  Tensor dgelu = add_kernel(cdf, mul_kernel(self, pdf), Scalar(1.0));
  return mul_kernel(grad_output, dgelu);
}

// Leaky ReLU backward: grad where x > 0, else grad * negative_slope.
Tensor leaky_relu_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& negative_slope,
    bool self_is_result) {
  (void)self_is_result;
  validate_float(grad_output, "leaky_relu_backward");
  Tensor scaled = mul_scalar_kernel(grad_output, negative_slope.to<double>());
  return where_kernel(gt_scalar_kernel(self, Scalar(0.0)), grad_output, scaled);
}

// Hardtanh backward: grad inside [min, max], zero outside.
Tensor hardtanh_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& min_val,
    const Scalar& max_val) {
  validate_float(grad_output, "hardtanh_backward");
  Tensor inside_min = where_scalar_other_kernel(
      ge_scalar_kernel(self, min_val), grad_output, Scalar(0.0));
  return where_scalar_other_kernel(
      le_scalar_kernel(self, max_val), inside_min, Scalar(0.0));
}

// Hardsigmoid backward: grad/6 on (-3, 3), zero elsewhere.
Tensor hardsigmoid_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self) {
  validate_float(grad_output, "hardsigmoid_backward");
  Tensor above_neg3 = where_scalar_other_kernel(
      gt_scalar_kernel(self, Scalar(-3.0)), grad_output, Scalar(0.0));
  Tensor inside = where_scalar_other_kernel(
      lt_scalar_kernel(self, Scalar(3.0)), above_neg3, Scalar(0.0));
  return mul_scalar_kernel(inside, 1.0 / 6.0);
}

// Hardswish backward: 0 below -3, grad*(x/3+0.5) on (-3, 3), grad above.
Tensor hardswish_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self) {
  validate_float(grad_output, "hardswish_backward");
  Tensor slope = add_scalar_kernel(
      div_scalar_kernel(self, Scalar(3.0)), Scalar(0.5), Scalar(1.0));
  Tensor scaled = mul_kernel(grad_output, slope);
  Tensor mid = where_kernel(lt_scalar_kernel(self, Scalar(3.0)), scaled, grad_output);
  return where_scalar_other_kernel(
      gt_scalar_kernel(self, Scalar(-3.0)), mid, Scalar(0.0));
}

// SiLU backward: grad * sigmoid(x) * (1 + x*(1 - sigmoid(x))).
Tensor silu_backward_kernel(const Tensor& grad_output, const Tensor& self) {
  validate_float(grad_output, "silu_backward");
  Tensor sig = sigmoid_kernel(self);
  Tensor one_minus_sig = rsub_scalar_kernel(sig, Scalar(1.0), Scalar(1.0));
  Tensor bracket = add_scalar_kernel(
      mul_kernel(self, one_minus_sig), Scalar(1.0), Scalar(1.0));
  return mul_kernel(grad_output, mul_kernel(sig, bracket));
}

// Mish backward:
// dy * (tanh(softplus(x)) + x*sigmoid(x)*(1 - tanh(softplus(x))^2)).
Tensor mish_backward_kernel(const Tensor& grad_output, const Tensor& self) {
  validate_float(grad_output, "mish_backward");
  Tensor sp = log1p_kernel(exp_kernel(self)); // softplus(x)
  Tensor th = tanh_kernel(sp);
  Tensor one_minus_tsq =
      rsub_scalar_kernel(mul_kernel(th, th), Scalar(1.0), Scalar(1.0));
  Tensor sig = sigmoid_kernel(self);
  Tensor deriv =
      add_kernel(th, mul_kernel(mul_kernel(self, sig), one_minus_tsq), Scalar(1.0));
  return mul_kernel(grad_output, deriv);
}

// GLU backward: split along dim, a = first half, b = second half.
// grad_a = grad * sigmoid(b); grad_b = grad * a * sigmoid(b) * (1-sigmoid(b)).
Tensor glu_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim) {
  validate_float(grad_output, "glu_backward");
  const std::vector<Tensor> halves = chunk_kernel(self, 2, dim);
  const Tensor& a = halves[0];
  const Tensor& b = halves[1];
  Tensor sig_b = sigmoid_kernel(b);
  Tensor one_minus = rsub_scalar_kernel(sig_b, Scalar(1.0), Scalar(1.0));
  Tensor grad_a = mul_kernel(grad_output, sig_b);
  Tensor grad_b =
      mul_kernel(mul_kernel(mul_kernel(grad_output, a), sig_b), one_minus);
  return cat_kernel({grad_a, grad_b}, dim);
}

// ELU forward: x < 0 ? expm1(x*input_scale)*alpha*scale : x*scale.
Tensor elu_kernel(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  const double a = alpha.to<double>();
  const double s = scale.to<double>();
  const double is = input_scale.to<double>();
  Tensor neg = mul_scalar_kernel(expm1_kernel(mul_scalar_kernel(self, is)), a * s);
  Tensor pos = mul_scalar_kernel(self, s);
  return where_kernel(lt_scalar_kernel(self, Scalar(0.0)), neg, pos);
}

// ELU backward. With is_result the last operand is the forward output y and
// the negative branch uses grad*input_scale*(y + alpha*scale); otherwise it
// is the forward input x and the branch uses grad*input_scale*alpha*scale*
// exp(x*input_scale).
Tensor elu_backward_kernel(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result) {
  const double a = alpha.to<double>();
  const double s = scale.to<double>();
  const double is = input_scale.to<double>();
  Tensor neg;
  if (is_result) {
    neg = mul_scalar_kernel(
        mul_kernel(
            grad_output,
            add_scalar_kernel(self_or_result, Scalar(a * s), Scalar(1.0))),
        is);
  } else {
    neg = mul_scalar_kernel(
        mul_kernel(
            grad_output, exp_kernel(mul_scalar_kernel(self_or_result, is))),
        a * s * is);
  }
  Tensor pos = mul_scalar_kernel(grad_output, s);
  return where_kernel(le_scalar_kernel(self_or_result, Scalar(0.0)), neg, pos);
}

// SELU forward: elu with the fixed SELU alpha and scale.
Tensor selu_kernel(const Tensor& self) {
  constexpr double kAlpha = 1.6732632423543772848170429916717;
  constexpr double kScale = 1.0507009873554804934193349852946;
  return elu_kernel(self, Scalar(kAlpha), Scalar(kScale), Scalar(1.0));
}

// Layer-norm backward. The mean/rstd tensors are per-row [M] vectors; rows
// span the outer (batch) dims, columns span the normalized axes. The
// per-row reduction follows the same algebra as the forward shader.
std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::vector<bool>& output_mask) {
  (void)bias_opt;
  TP_CHECK(
      grad_out.dtype() == DType::Float32 && input.dtype() == DType::Float32,
      "Vulkan native_layer_norm_backward supports Float32 tensors only");

  const int64_t ndim = input.dim();
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  const int64_t axis = ndim - norm_ndim;

  std::vector<int64_t> batch_dims;
  std::vector<int64_t> norm_dims;
  std::vector<int64_t> broadcast_shape;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i < axis) {
      batch_dims.push_back(i);
      broadcast_shape.push_back(input.size(i));
    } else {
      norm_dims.push_back(i);
      broadcast_shape.push_back(1);
    }
  }

  int64_t N = 1;
  for (const int64_t s : normalized_shape) {
    N *= s;
  }
  const double inv_n = 1.0 / static_cast<double>(N);

  const std::vector<int64_t> full_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  // Materialize broadcasts with expand so every elementwise multiply below
  // operates on equal-shaped tensors.
  const Tensor mean_b = expand_kernel(
      reshape_kernel(mean, broadcast_shape), full_shape, false);
  const Tensor rstd_b = expand_kernel(
      reshape_kernel(rstd, broadcast_shape), full_shape, false);
  const Tensor dy = grad_out;

  Tensor weight_b;
  if (weight_opt.has_value()) {
    std::vector<int64_t> wshape(axis, 1);
    wshape.insert(wshape.end(), normalized_shape.begin(), normalized_shape.end());
    weight_b = reshape_kernel(*weight_opt, wshape);
  }
  // Expand weight to the full input shape so every multiply below is
  // equal-shaped (texture storage does not support mid-axis broadcasting).
  const Tensor dy_gamma = weight_opt.has_value()
      ? mul_kernel(dy, expand_kernel(weight_b, full_shape, false))
      : dy;

  // Row-wise sums over the normalized axes (kept) for the dx correction
  // terms; ds = sum(dy*gamma*x), db = sum(dy*gamma).
  const Tensor ds = expand_kernel(
      sum_dim_kernel(
          mul_kernel(dy_gamma, input), norm_dims, true, DType::Float32),
      full_shape,
      false);
  const Tensor db = expand_kernel(
      sum_dim_kernel(dy_gamma, norm_dims, true, DType::Float32),
      full_shape,
      false);

  const Tensor rstd_cubed = mul_kernel(mul_kernel(rstd_b, rstd_b), rstd_b);
  const Tensor b_coef = mul_scalar_kernel(
      mul_kernel(sub_kernel(mul_kernel(db, mean_b), ds, Scalar(1.0)), rstd_cubed),
      inv_n);
  const Tensor c_coef = sub_kernel(
      neg_kernel(mul_kernel(b_coef, mean_b)),
      mul_scalar_kernel(mul_kernel(db, rstd_b), inv_n),
      Scalar(1.0));

  Tensor dx;
  if (output_mask[0]) {
    dx = add_kernel(
        add_kernel(
            mul_kernel(rstd_b, dy_gamma),
            mul_kernel(b_coef, input),
            Scalar(1.0)),
        c_coef,
        Scalar(1.0));
  }

  Tensor dgamma;
  if (output_mask[1] && weight_opt.has_value()) {
    const Tensor x_hat =
        mul_kernel(sub_kernel(input, mean_b, Scalar(1.0)), rstd_b);
    if (batch_dims.empty()) {
      // Whole tensor is one row: the per-element weight gradient is just
      // dy * x_hat with no cross-row reduction.
      dgamma = mul_kernel(dy, x_hat);
    } else {
      dgamma = sum_dim_kernel(
          mul_kernel(dy, x_hat), batch_dims, false, DType::Float32);
    }
  }

  Tensor dbeta;
  if (output_mask[2]) {
    if (batch_dims.empty()) {
      dbeta = dy;
    } else {
      dbeta =
          sum_dim_kernel(dy, batch_dims, false, DType::Float32);
    }
  }

  return std::make_tuple(dx, dgamma, dbeta);
}

// Softplus forward: x*beta > threshold ? x : log1p(exp(x*beta))/beta.
Tensor softplus_kernel(
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  const double b = beta.to<double>();
  Tensor xb = mul_scalar_kernel(self, b);
  Tensor safe = div_scalar_kernel(log1p_kernel(exp_kernel(xb)), Scalar(b));
  return where_kernel(gt_scalar_kernel(xb, threshold), self, safe);
}

// Softplus backward: grad * sigmoid(x*beta) unless x*beta > threshold.
Tensor softplus_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& beta,
    const Scalar& threshold) {
  const double b = beta.to<double>();
  Tensor xb = mul_scalar_kernel(self, b);
  Tensor sig = sigmoid_kernel(xb);
  Tensor scaled = mul_kernel(grad_output, sig);
  return where_kernel(gt_scalar_kernel(xb, threshold), grad_output, scaled);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, AutogradKernels) {
  m.impl("threshold_backward",
         &tensorplay::vulkan::ops::threshold_backward_kernel);
  m.impl("sigmoid_backward",
         &tensorplay::vulkan::ops::sigmoid_backward_kernel);
  m.impl("tanh_backward", &tensorplay::vulkan::ops::tanh_backward_kernel);
  m.impl("gelu_backward", &tensorplay::vulkan::ops::gelu_backward_kernel);
  m.impl("leaky_relu_backward",
         &tensorplay::vulkan::ops::leaky_relu_backward_kernel);
  m.impl("hardtanh_backward",
         &tensorplay::vulkan::ops::hardtanh_backward_kernel);
  m.impl("hardsigmoid_backward",
         &tensorplay::vulkan::ops::hardsigmoid_backward_kernel);
  m.impl("hardswish_backward",
         &tensorplay::vulkan::ops::hardswish_backward_kernel);
  m.impl("silu_backward", &tensorplay::vulkan::ops::silu_backward_kernel);
  m.impl("mish_backward", &tensorplay::vulkan::ops::mish_backward_kernel);
  m.impl("glu_backward", &tensorplay::vulkan::ops::glu_backward_kernel);
  m.impl("elu", &tensorplay::vulkan::ops::elu_kernel);
  m.impl("elu_backward", &tensorplay::vulkan::ops::elu_backward_kernel);
  m.impl("selu", &tensorplay::vulkan::ops::selu_kernel);
  m.impl("softplus", &tensorplay::vulkan::ops::softplus_kernel);
  m.impl("softplus_backward",
         &tensorplay::vulkan::ops::softplus_backward_kernel);
  m.impl("native_layer_norm_backward",
         &tensorplay::vulkan::ops::native_layer_norm_backward_kernel);
}

#endif /* USE_VULKAN */