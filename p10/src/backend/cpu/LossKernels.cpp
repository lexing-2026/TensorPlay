#include "Tensor.h"
#include "Generator.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include <tuple>
#include <optional>
#include <cmath>
#include <vector>

#include "../composite/AttentionComposite.h"

namespace tensorplay {
namespace cpu {

namespace ops = tensorplay::tpx::ops;

template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss_impl(const Tensor& input, const Tensor& target,
                                         const std::optional<Tensor>& weight,
                                         int64_t reduction, int64_t ignore_index) {
    // reduction: 0=none, 1=mean, 2=sum
    int64_t n_batch = input.size(0);
    int64_t n_classes = input.size(1);

    const scalar_t* input_data = input.data_ptr<scalar_t>();
    const int64_t* target_data = target.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<scalar_t>() : nullptr;

    std::vector<scalar_t> output_data(n_batch);
    double total_weight = 0;

    for (int64_t i = 0; i < n_batch; ++i) {
        int64_t t = target_data[i];
        if (t == ignore_index) {
            output_data[i] = 0;
            continue;
        }
        if (t < 0 || t >= n_classes) TP_THROW(RuntimeError, "Target out of bounds");

        double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
        output_data[i] = static_cast<scalar_t>(-static_cast<double>(input_data[i * n_classes + t]) * w);
        total_weight += w;
    }

    DType dt = input.dtype();
    Tensor total_weight_tensor = Tensor::tensor({total_weight}, DType::Float64, input.device()).to(dt == DType::Float64 ? DType::Float64 : DType::Float32);

    if (reduction == 0) { // None
        return std::make_tuple(Tensor::tensor(output_data, dt, input.device()), total_weight_tensor);
    } else if (reduction == 1) { // Mean
        double sum = 0;
        for (scalar_t x : output_data) sum += x;
        if (total_weight > 0) sum /= total_weight;
        return std::make_tuple(Tensor::tensor({sum}, DType::Float64, input.device()).to(dt).reshape({}), total_weight_tensor);
    } else if (reduction == 2) { // Sum
        double sum = 0;
        for (scalar_t x : output_data) sum += x;
        return std::make_tuple(Tensor::tensor({sum}, DType::Float64, input.device()).to(dt).reshape({}), total_weight_tensor);
    }
    TP_THROW(ValueError, "Invalid reduction mode");
}

std::tuple<Tensor, Tensor> nll_loss_kernel(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    if (input.dtype() == DType::Float32) return nll_loss_impl<float>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Float64) return nll_loss_impl<double>(input, target, weight, reduction, ignore_index);
    TP_THROW(NotImplementedError, "nll_loss only supports Float32/Float64");
}

template <typename scalar_t>
Tensor nll_loss_backward_impl(const Tensor& grad_output, const Tensor& input, const Tensor& target,
                              const std::optional<Tensor>& weight, int64_t reduction,
                              int64_t ignore_index, const Tensor& total_weight) {
    int64_t n_batch = input.size(0);
    int64_t n_classes = input.size(1);

    Tensor grad_input = Tensor::zeros({n_batch, n_classes}, input.dtype(), input.device());
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

    const int64_t* target_data = target.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<scalar_t>() : nullptr;

    double tw = 0;
    if (total_weight.defined()) {
        tw = total_weight.item().to<double>();
    } else if (reduction == 1) {
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t != ignore_index) {
                tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            }
        }
    }

    if (reduction == 0) { // None
        const scalar_t* grad_out_data = grad_output.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            grad_input_data[i * n_classes + t] = static_cast<scalar_t>(-w * static_cast<double>(grad_out_data[i]));
        }
    } else {
        double grad_val = grad_output.item().to<double>();
        if (reduction == 1) { // Mean
             if (tw > 0) grad_val /= tw;
        }

        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            grad_input_data[i * n_classes + t] = static_cast<scalar_t>(-w * grad_val);
        }
    }

    return grad_input;
}

Tensor nll_loss_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    if (input.dtype() == DType::Float32) return nll_loss_backward_impl<float>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Float64) return nll_loss_backward_impl<double>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    TP_THROW(NotImplementedError, "nll_loss backward only supports Float32/Float64");
}

// (N, C, H, W), target (N, H, W); every spatial position is an independent
// batch row, so the row layout is input[(n * C + t) * H * W + pos].
template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss2d_impl(const Tensor& input, const Tensor& target,
                                           const std::optional<Tensor>& weight,
                                           int64_t reduction, int64_t ignore_index) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d: Expected 3D target");
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    if (target.size(0) != N || target.size(1) != H || target.size(2) != W)
        TP_THROW(RuntimeError, "nll_loss2d: target shape must match input spatial dims");
    const int64_t rows = N * H * W;

    const Tensor input_c = input.contiguous();
    const Tensor target_c = target.contiguous();
    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<scalar_t>() : nullptr;

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

    DType dt = input.dtype();
    DType tw_dt = dt == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor total_weight_tensor = Tensor::tensor({total_weight}, DType::Float64, input.device()).to(tw_dt);

    if (reduction == 0) {
        return std::make_tuple(Tensor::tensor(output_data, dt, input.device()).reshape({N, H, W}), total_weight_tensor);
    }
    if (reduction == 1 && total_weight > 0) sum /= total_weight;
    if (reduction != 1 && reduction != 2) TP_THROW(ValueError, "Invalid reduction mode");
    return std::make_tuple(Tensor::tensor({sum}, DType::Float64, input.device()).to(dt).reshape({}), total_weight_tensor);
}

std::tuple<Tensor, Tensor> nll_loss2d_kernel(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    if (input.dtype() == DType::Float32) return nll_loss2d_impl<float>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Float64) return nll_loss2d_impl<double>(input, target, weight, reduction, ignore_index);
    TP_THROW(NotImplementedError, "nll_loss2d only supports Float32/Float64");
}

template <typename scalar_t>
Tensor nll_loss2d_backward_impl(const Tensor& grad_output, const Tensor& input, const Tensor& target,
                                const std::optional<Tensor>& weight, int64_t reduction,
                                int64_t ignore_index, const Tensor& total_weight) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 3D target");
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t rows = N * H * W;

    const Tensor target_c = target.contiguous();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<scalar_t>() : nullptr;

    Tensor grad_input = Tensor::zeros({N, C, H, W}, input.dtype(), input.device());
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

    double tw = 0;
    if (total_weight.defined()) {
        tw = total_weight.item().to<double>();
    } else if (reduction == 1) {
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t t = target_data[i];
            if (t != ignore_index) tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
        }
    }

    const Tensor grad_out_c = reduction == 0 ? grad_output.contiguous() : Tensor();
    const scalar_t* grad_out_data = reduction == 0 ? grad_out_c.data_ptr<scalar_t>() : nullptr;
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

Tensor nll_loss2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    if (input.dtype() == DType::Float32) return nll_loss2d_backward_impl<float>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Float64) return nll_loss2d_backward_impl<double>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    TP_THROW(NotImplementedError, "nll_loss2d backward only supports Float32/Float64");
}

// MSE Loss
Tensor mse_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    // reduction: 0=none, 1=mean, 2=sum
    Tensor diff = input - target;
    Tensor sq_diff = diff * diff;
    
    if (reduction == 0) { // None
        return sq_diff;
    } else if (reduction == 1) { // Mean
        return sq_diff.mean();
    } else if (reduction == 2) { // Sum
        return sq_diff.sum();
    }
    TP_THROW(ValueError, "Invalid reduction mode");
}

Tensor mse_loss_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
    Tensor diff = input - target;
    Tensor grad_input;
    
    if (reduction == 0) { // None
        // grad_output shape matches input/target shape (broadcasted)
        // L = (x-y)^2
        // dL/dx = 2(x-y) * dL_out
        grad_input = 2.0 * diff * grad_output;
    } else {
        // Scalar output
        // Mean: L = mean((x-y)^2) = 1/N * sum((x-y)^2)
        // dL/dx = 2/N * (x-y) * grad_output
        // Sum: L = sum((x-y)^2)
        // dL/dx = 2 * (x-y) * grad_output
        
        double scale = 2.0;
        if (reduction == 1) {
            scale /= (double)input.numel();
        }
        
        grad_input = (scale * diff) * grad_output;
    }
    
    return grad_input;
}



// following the mse_loss_kernel house style.
// reduction: 0=none, 1=mean, 2=sum
// -----------------------------------------------------------------------------

namespace {

Tensor loss_reduce(const Tensor& x, int64_t reduction) {
    if (reduction == 0) return x;
    if (reduction == 1) return x.mean();
    if (reduction == 2) return x.sum();
    TP_THROW(ValueError, "Invalid reduction mode");
}

Tensor scale_grad(const Tensor& g, int64_t reduction, int64_t numel) {
    if (reduction == 1) return g / static_cast<double>(numel);
    return g;
}

// Local stand-ins until clamp_min/xlogy/zeros_like land as real Tensor ops.
Tensor zeros_like_shim(const Tensor& t) { return t * 0; }

Tensor clamp_min_shim(const Tensor& x, Scalar s) {
    return Tensor::where(x.lt(s), s, x);
}

Tensor xlogy_shim(const Tensor& x, const Tensor& y) {
    return Tensor::where(x.eq(Scalar(0)), x * 0, x * y.log());
}

} // anonymous namespace

Tensor l1_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    return loss_reduce((input - target).abs(), reduction);
}

Tensor l1_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction) {
    Tensor g = (input - target).sign() * grad_output;
    return scale_grad(g, reduction, input.numel());
}

// z = |x - t|; z < beta ? 0.5 z^2 / beta : z - 0.5 beta (equal at z == beta).
Tensor smooth_l1_loss_cpu(const Tensor& input, const Tensor& target,
                          int64_t reduction, double beta) {
    if (beta < 0)
        TP_THROW(ValueError, "smooth_l1_loss does not support negative values for beta.");
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(beta)), diff * diff * (0.5 / beta),
                                absd - 0.5 * beta);
    return loss_reduce(loss, reduction);
}

// x = input - target; |x| <= beta ? norm * x / beta : norm * sign(x),
// with norm = 1/numel for mean reduction (inclusive/exclusive boundary
// forms agree since both give norm * grad at |x| == beta).
Tensor smooth_l1_loss_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& target, int64_t reduction, double beta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(beta)), diff / beta, diff.sign()) * grad_output;
    return scale_grad(g, reduction, input.numel());
}

// z = |x - t|; z < delta ? 0.5 z^2 : delta (z - 0.5 delta).
Tensor huber_loss_cpu(const Tensor& input, const Tensor& target,
                      int64_t reduction, double delta) {
    if (delta <= 0)
        TP_THROW(ValueError, "huber_loss does not support non-positive values for delta.");
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(delta)), diff * diff * 0.5,
                                delta * (absd - 0.5 * delta));
    return loss_reduce(loss, reduction);
}

// x = input - target; |x| <= delta ? norm * x : norm * delta * sign(x).
Tensor huber_loss_backward_cpu(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction, double delta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(delta)), diff, delta * diff.sign()) * grad_output;
    return scale_grad(g, reduction, input.numel());
}

Tensor kl_div_kernel(const Tensor& input, const Tensor& target,
                     int64_t reduction, bool log_target) {
    Tensor nz = target.ne(0).to(input.dtype());
    Tensor loss;
    if (!log_target) {
        loss = (xlogy_shim(target, target) - target * input) * nz;
    } else {
        Tensor t = target.exp();
        loss = (t * (target - input)) * nz;
    }
    return loss_reduce(loss, reduction);
}

Tensor kl_div_backward_kernel(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target, int64_t reduction, bool log_target) {
    Tensor nz = target.ne(0).to(input.dtype());
    Tensor g;
    if (!log_target) {
        g = (-target * nz) * grad_output;   // d/dinput of -target*input
    } else {
        g = -(target.exp()) * grad_output;
    }
    return scale_grad(g, reduction, input.numel());
}

// per-element loss is (t-1)*max(log(1-x),-100) - t*max(log(x),-100), then
// optional weight multiply and reduction.
void bce_check_01(const Tensor& t, const char* what) {
    if (t.numel() == 0) return;
    double bad = (t.lt(0.0).to(DType::Float64) + t.gt(1.0).to(DType::Float64))
                     .sum().item().to<double>();
    if (bad > 0)
        TP_THROW(RuntimeError, std::string("all elements of ") + what +
                 " should be between 0 and 1");
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target,
                                const std::optional<Tensor>& weight, int64_t reduction) {
    bce_check_01(input, "input");
    bce_check_01(target, "target");
    Tensor log_x = clamp_min_shim(input.log(), Scalar(-100.0));
    Tensor log_1mx = clamp_min_shim((-input + 1.0).log(), Scalar(-100.0));
    Tensor loss = (target - 1.0) * log_1mx - target * log_x;
    if (weight.has_value() && weight->defined()) loss = loss * weight.value();
    return loss_reduce(loss, reduction);
}

// grad * (x - t) / max((1 - x) * x, 1e-12), then weight multiply, and
// division by input.numel() for mean reduction.
Tensor binary_cross_entropy_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor denom = clamp_min_shim(input * (-input + 1.0), Scalar(1e-12));
    Tensor g = grad_output * (input - target) / denom;
    if (weight.has_value() && weight->defined()) g = g * weight.value();
    return scale_grad(g, reduction, input.numel());
}

Tensor margin_ranking_loss_kernel(const Tensor& input1, const Tensor& input2,
                                  const Tensor& target, double margin, int64_t reduction) {
    Tensor loss = clamp_min_shim(-(input1 - input2) * target + margin, Scalar(0.0));
    return loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> margin_ranking_loss_backward_kernel(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    Tensor active = ((-(input1 - input2) * target + margin).gt(0.0)).to(input1.dtype());
    Tensor g = -active * target * grad_output;
    g = scale_grad(g, reduction, input1.numel());
    return std::make_tuple(g, -g);
}

Tensor hinge_embedding_loss_kernel(const Tensor& input, const Tensor& target,
                                   double margin, int64_t reduction) {
    //                        + where(t != -1, x, 0)
    Tensor z = zeros_like_shim(input);
    Tensor margin_part = Tensor::where(target.ne(1),
                                       clamp_min_shim(margin - input, Scalar(0.0)), z);
    Tensor self_part = Tensor::where(target.ne(-1), input, z);
    return loss_reduce(margin_part + self_part, reduction);
}

Tensor hinge_embedding_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target, double margin, int64_t reduction) {
    //   y == 1          -> 1
    //   y == -1         -> (margin - x > 0) ? -1 : 0
    //   otherwise       -> 1 + ((margin - x > 0) ? -1 : 0)
    Tensor ones = Tensor::ones_like(input);
    Tensor active = ((margin - input).gt(Scalar(0.0))).to(input.dtype());
    Tensor g = Tensor::where(target.eq(1), ones,
               Tensor::where(target.eq(-1), -active, ones - active));
    g = g * grad_output;
    return scale_grad(g, reduction, input.numel());
}

Tensor soft_margin_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    // log(1 + exp(-target*input))
    Tensor loss = ((input * target) * -1.0).exp().add(1.0).log();
    return loss_reduce(loss, reduction);
}

Tensor soft_margin_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& target, int64_t reduction) {
    Tensor g = -target * ((input * target) * -1.0).exp().sigmoid() * grad_output;
    return scale_grad(g, reduction, input.numel());
}

Tensor cosine_embedding_loss_kernel(const Tensor& x1, const Tensor& x2,
                                    const Tensor& target, double margin, int64_t reduction) {
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = (n1 * n2).sqrt();
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;
    Tensor zero = zeros_like_shim(cos);
    //   t ==  1 -> 1 - cos
    //   t == -1 -> max(0, cos - margin)
    //   else    -> 0
    Tensor loss = Tensor::where(target.eq(1), zero + (1.0 - cos),
                   Tensor::where(target.eq(-1), clamp_min_shim(cos - margin, Scalar(0.0)),
                                 zero));
    return loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> cosine_embedding_loss_backward_kernel(
        const Tensor& grad_output, const Tensor& x1, const Tensor& x2,
        const Tensor& target, double margin, int64_t reduction) {
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = clamp_min_shim((n1 * n2).sqrt(), Scalar(1e-12));
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;

    Tensor ones_row = Tensor::ones({x1.size(0)}, x1.dtype(), x1.device());
    Tensor dl_dcos = Tensor::where(target.eq(1), -1.0 * ones_row,
                     Tensor::where(target.eq(-1),
                         ((cos - margin).gt(0.0)).to(x1.dtype()),
                         ((1.0 - cos - margin).gt(0.0)).to(x1.dtype()) * -1.0));

    if (reduction == 1) dl_dcos = dl_dcos / static_cast<double>(x1.size(0));

    Tensor c = cos.unsqueeze(1);
    Tensor g1 = (x2 / d.unsqueeze(1)) - c * (x1 / n1.unsqueeze(1));
    Tensor g2 = (x1 / d.unsqueeze(1)) - c * (x2 / n2.unsqueeze(1));
    g1 = g1 * (dl_dcos * grad_output).unsqueeze(1);
    g2 = g2 * (dl_dcos * grad_output).unsqueeze(1);
    return std::make_tuple(g1, g2);
}

Tensor poisson_nll_loss_kernel(const Tensor& input, const Tensor& target,
                               bool log_input, bool full, double eps, int64_t reduction) {
    Tensor loss;
    if (log_input) {
        loss = input.exp() - target * input;
    } else {
        loss = input - target * (input + eps).log();
    }
    if (full) {
        // Stirling approximation term: t*log(t) - t + 0.5*log(2*pi*t), for t>0
        Tensor pos = target.gt(0).to(input.dtype());
        Tensor t_safe = target + (1.0 - pos);   // avoid log(0)
        Tensor stirling = (xlogy_shim(target, t_safe) - t_safe +
                           (t_safe * (2.0 * M_PI)).log() * 0.5) * pos;
        loss = loss + stirling;
    }
    return loss_reduce(loss, reduction);
}

Tensor poisson_nll_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& target, bool log_input, bool full,
                                        double eps, int64_t reduction) {
    Tensor g;
    if (log_input) {
        g = input.exp() - target;
    } else {
        g = 1.0 - target / (input + eps);
    }
    g = g * grad_output;
    return scale_grad(g, reduction, input.numel());
}

// ---------------------------------------------------------------------------
//
// `_ctc_loss` returns the RAW per-sequence negative log likelihood (entries
// stay +inf for impossible alignments; `zero_infinity` is honored only by
// with the log-alpha table (N, T, 2S+1) consumed by the backward.
// Targets use the batch-first zero-padded layout (N, S); lengths are (N,).

namespace {

inline int64_t ctc_target_prime(const int64_t* targets, int64_t stride,
                                int64_t idx, int64_t blank) {
    if (idx % 2 == 0) return blank;
    return targets[(idx / 2) * stride];
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> ctc_loss_impl(const Tensor& log_probs,
                                         const Tensor& targets,
                                         const std::vector<int64_t>& input_lengths,
                                         const std::vector<int64_t>& target_lengths,
                                         int64_t blank) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    int64_t T = log_probs.size(0);
    int64_t N = log_probs.size(1);
    int64_t C = log_probs.size(2);
    int64_t S = targets.size(1);
    int64_t M = 2 * S + 1;

    Tensor neg_log_likelihood = Tensor::empty({N}, log_probs.dtype(), log_probs.device());
    Tensor log_alpha = Tensor::empty({N, T, M}, log_probs.dtype(), log_probs.device());
    const scalar_t* lpp = log_probs.data_ptr<scalar_t>();
    scalar_t* lap = log_alpha.data_ptr<scalar_t>();
    scalar_t* nllp = neg_log_likelihood.data_ptr<scalar_t>();
    const int64_t* tgt = targets.data_ptr<int64_t>();

    for (int64_t b = 0; b < N; ++b) {
        int64_t il = input_lengths[b];
        int64_t tl = target_lengths[b];
        if (il < 0 || il > T)
            TP_THROW(RuntimeError, "_ctc_loss: input_lengths out of range");
        if (tl < 0 || tl > S)
            TP_THROW(RuntimeError, "_ctc_loss: target_lengths out of range");
        const int64_t* tb = tgt + b * S;
        scalar_t* lab = lap + b * T * M;

        if (il == 0) {
            scalar_t log_likelihood = (tl == 0) ? static_cast<scalar_t>(0) : neginf;
            nllp[b] = static_cast<scalar_t>(-log_likelihood);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t s = 0; s < M; ++s) lab[t * M + s] = neginf;
            continue;
        }

        // first row of eq (6): everything starts from -inf
        for (int64_t s = 0; s < M; ++s) lab[s] = neginf;
        const scalar_t* lp0 = lpp + (0 * N + b) * C;
        lab[0] = lp0[blank];
        if (tl > 0)
            lab[1] = lp0[ctc_target_prime(tb, 1, 1, blank)];

        // alpha recursion, eqs (6)/(7)
        for (int64_t t = 1; t < il; ++t) {
            const scalar_t* lpt = lpp + (t * N + b) * C;
            const scalar_t* prev = lab + (t - 1) * M;
            scalar_t* cur = lab + t * M;
            for (int64_t s = 0; s < M; ++s) cur[s] = neginf;
            for (int64_t s = 0; s <= 2 * tl && s < M; ++s) {
                const int64_t current_target_prime =
                    ctc_target_prime(tb, 1, s, blank);
                scalar_t la1 = prev[s];
                scalar_t lamax = la1;
                scalar_t la2 = neginf, la3 = neginf;
                if (s > 0) {
                    la2 = prev[s - 1];
                    if (la2 > lamax) lamax = la2;
                }
                if (s > 1 && ctc_target_prime(tb, 1, s - 2, blank) !=
                                 ctc_target_prime(tb, 1, s, blank)) {
                    la3 = prev[s - 2];
                    if (la3 > lamax) lamax = la3;
                }
                if (lamax == neginf) lamax = 0;  // cannot do neginf - neginf
                cur[s] = std::log(std::exp(la1 - lamax) + std::exp(la2 - lamax) +
                                  std::exp(la3 - lamax)) +
                         lamax + lpt[current_target_prime];
            }
        }
        // deterministic -inf for frames beyond each sample's input length
        for (int64_t t = il; t < T; ++t)
            for (int64_t s = 0; s < M; ++s) lab[t * M + s] = neginf;

        // likelihood: sum of the last two alphas, eq (8)
        if (tl == 0) {
            nllp[b] = -lab[(il - 1) * M];
        } else {
            scalar_t l1 = lab[(il - 1) * M + tl * 2];
            scalar_t l2 = lab[(il - 1) * M + tl * 2 - 1];
            scalar_t m = std::max(l1, l2);
            m = (m == neginf) ? static_cast<scalar_t>(0) : m;
            scalar_t log_likelihood =
                std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
            nllp[b] = -log_likelihood;
        }
    }
    return std::make_tuple(neg_log_likelihood, log_alpha);
}

template <typename scalar_t>
Tensor ctc_loss_backward_impl(const Tensor& grad_out, const Tensor& log_probs,
                              const Tensor& targets,
                              const std::vector<int64_t>& input_lengths,
                              const std::vector<int64_t>& target_lengths,
                              const Tensor& neg_log_likelihood,
                              const Tensor& log_alpha, int64_t blank,
                              bool zero_infinity) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    int64_t T = log_probs.size(0);
    int64_t N = log_probs.size(1);
    int64_t C = log_probs.size(2);
    int64_t S = targets.size(1);
    int64_t M = 2 * S + 1;

    // eq (16) collection detects untouched entries via that sentinel.
    Tensor grad = Tensor::empty_like(log_probs);
    scalar_t* gp_init = grad.data_ptr<scalar_t>();
    std::fill_n(gp_init, static_cast<size_t>(T * N * C), neginf);
    const scalar_t* lpp = log_probs.data_ptr<scalar_t>();
    const scalar_t* lap = log_alpha.data_ptr<scalar_t>();
    const scalar_t* nllp = neg_log_likelihood.data_ptr<scalar_t>();
    const int64_t* tgt = targets.data_ptr<int64_t>();
    const scalar_t* gop = grad_out.data_ptr<scalar_t>();
    scalar_t* gp = grad.data_ptr<scalar_t>();
    std::vector<scalar_t> beta(M);

    for (int64_t b = 0; b < N; ++b) {
        if (zero_infinity && nllp[b] == std::numeric_limits<scalar_t>::infinity()) {
            // zeroed loss: zero this batch item's (strided) grad column
            for (int64_t t = 0; t < T; ++t)
                for (int64_t c = 0; c < C; ++c) gp[(t * N + b) * C + c] = static_cast<scalar_t>(0);
            continue;
        }
        int64_t il = input_lengths[b];
        int64_t tl = target_lengths[b];
        const int64_t* tb = tgt + b * S;
        const scalar_t* lpb = lpp + b * C;          // frame stride N*C
        const scalar_t* lab = lap + b * T * M;
        scalar_t* gb = gp + b * C;

        if (il > 0) {
            // beta initialization at t = input_length - 1
            for (int64_t s = 0; s < M; ++s) beta[s] = neginf;
            const scalar_t* lp_last = lpb + ((il - 1) * N) * C;
            scalar_t* g_last = gb + (il - 1) * N * C;
            const scalar_t* la_last = lab + (il - 1) * M;
            beta[2 * tl] = lp_last[blank];
            g_last[blank] = la_last[2 * tl] + beta[2 * tl];
            if (tl > 0) {
                int64_t prime = ctc_target_prime(tb, 1, 2 * tl - 1, blank);
                beta[2 * tl - 1] = lp_last[prime];
                // first two states are a blank and a non-blank: no log+ needed
                g_last[prime] = la_last[2 * tl - 1] + beta[2 * tl - 1];
            }

            // eq (10)/(11) recursion plus eq (16) collection.
            // beta is a rolling single-row buffer: ascending s keeps
            // beta[s+1]/beta[s+2] holding the previous-row (t+1) values
            // until after they are consumed at step s.
            for (int64_t t = il - 2; t >= 0; --t) {
                const scalar_t* lpt = lpb + (t * N) * C;
                const scalar_t* lan = lab + ((t + 1) * M);
                const scalar_t* lat = lab + (t * M);
                scalar_t* gt = gb + (t * N) * C;
                for (int64_t s = 0; s <= 2 * tl; ++s) {
                    const int64_t current_target_prime =
                        ctc_target_prime(tb, 1, s, blank);
                    scalar_t lb1 = beta[s];
                    scalar_t lbmax = lb1;
                    scalar_t lb2 = neginf, lb3 = neginf;
                    if (s < 2 * tl) {
                        lb2 = beta[s + 1];
                        if (lb2 > lbmax) lbmax = lb2;
                    }
                    if (s < 2 * tl - 1 &&
                        ctc_target_prime(tb, 1, s + 2, blank) != current_target_prime) {
                        lb3 = beta[s + 2];
                        if (lb3 > lbmax) lbmax = lb3;
                    }
                    if (lbmax == neginf) lbmax = 0;
                    beta[s] = std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) +
                                       std::exp(lb3 - lbmax)) +
                              lbmax + lpt[current_target_prime];
                    // collected[b, t, target'[s]] "log+=" alpha + beta
                    scalar_t log_alpha_beta = lat[s] + beta[s];
                    scalar_t& lcab = gt[current_target_prime];
                    if (lcab == neginf) {
                        lcab = log_alpha_beta;
                    } else {
                        scalar_t mx = std::max(lcab, log_alpha_beta);
                        lcab = std::log(std::exp(lcab - mx) +
                                        std::exp(log_alpha_beta - mx)) + mx;
                    }
                }
            }
        }

        // wrap up with the remaining items of eq (16); note the dense softmax
        // training-time log_softmax, so unused labels receive exp(lp) * gr.
        scalar_t gr = gop[b];
        for (int64_t t = 0; t < il; ++t) {
            const scalar_t* lpt = lpb + (t * N) * C;
            scalar_t* gt = gb + (t * N) * C;
            for (int64_t c = 0; c < C; ++c) {
                scalar_t res = gt[c];
                scalar_t lpv = lpt[c];
                gt[c] = (std::exp(lpv) - std::exp(res + nllp[b] - lpv)) * gr;
            }
        }
        // zero the remainder (frames beyond input_length)
        for (int64_t t = il; t < T; ++t)
            for (int64_t c = 0; c < C; ++c) gb[(t * N) * C + c] = static_cast<scalar_t>(0);
    }
    return grad;
}

}  // namespace

Tensor _ctc_loss_backward_cpu(const Tensor& grad, const Tensor& log_probs,
                              const Tensor& targets, const Tensor& input_lengths,
                              const Tensor& target_lengths,
                              const Tensor& neg_log_likelihood,
                              const Tensor& log_alpha, int64_t blank,
                              bool zero_infinity) {
    auto in_l = input_lengths.contiguous();
    auto tg_l = target_lengths.contiguous();
    std::vector<int64_t> il(in_l.data_ptr<int64_t>(),
                            in_l.data_ptr<int64_t>() + in_l.numel());
    std::vector<int64_t> tl(tg_l.data_ptr<int64_t>(),
                            tg_l.data_ptr<int64_t>() + tg_l.numel());
    if (log_probs.dtype() == DType::Float64)
        return ctc_loss_backward_impl<double>(
            grad.contiguous(), log_probs.contiguous(), targets.contiguous(),
            il, tl, neg_log_likelihood.contiguous(), log_alpha.contiguous(),
            blank, zero_infinity);
    if (log_probs.dtype() == DType::Float32)
        return ctc_loss_backward_impl<float>(
            grad.contiguous(), log_probs.contiguous(), targets.contiguous(),
            il, tl, neg_log_likelihood.contiguous(), log_alpha.contiguous(),
            blank, zero_infinity);
    TP_THROW(NotImplementedError, "_ctc_loss_backward only supports Float32/Float64");
}

std::tuple<Tensor, Tensor> _ctc_loss_cpu(const Tensor& log_probs,
                                         const Tensor& targets,
                                         const Tensor& input_lengths,
                                         const Tensor& target_lengths,
                                         int64_t blank, bool zero_infinity) {
    (void)zero_infinity;
    auto in_l = input_lengths.contiguous();
    auto tg_l = target_lengths.contiguous();
    std::vector<int64_t> il(in_l.data_ptr<int64_t>(),
                            in_l.data_ptr<int64_t>() + in_l.numel());
    std::vector<int64_t> tl(tg_l.data_ptr<int64_t>(),
                            tg_l.data_ptr<int64_t>() + tg_l.numel());
    if (log_probs.dtype() == DType::Float64)
        return ctc_loss_impl<double>(log_probs.contiguous(),
                                     targets.contiguous(), il, tl, blank);
    if (log_probs.dtype() == DType::Float32)
        return ctc_loss_impl<float>(log_probs.contiguous(),
                                    targets.contiguous(), il, tl, blank);
    TP_THROW(NotImplementedError, "_ctc_loss only supports Float32/Float64");
}


Tensor ctc_loss_intlist_cpu(const Tensor& log_probs, const Tensor& targets,
                            const std::vector<int64_t>& input_lengths,
                            const std::vector<int64_t>& target_lengths,
                            int64_t blank, int64_t reduction,
                            bool zero_infinity) {
    Tensor il = Tensor::tensor(input_lengths, DType::Int64, log_probs.device());
    Tensor tl = Tensor::tensor(target_lengths, DType::Int64, log_probs.device());
    return composite::ctc_loss_compose(log_probs, targets, il, tl, blank, reduction,
                            zero_infinity);
}

Tensor ctc_loss_tensor_cpu(const Tensor& log_probs, const Tensor& targets,
                           const Tensor& input_lengths,
                           const Tensor& target_lengths, int64_t blank,
                           int64_t reduction, bool zero_infinity) {
    if (input_lengths.dtype() != DType::Int64 &&
        input_lengths.dtype() != DType::Int32) {
        TP_THROW(TypeError, "ctc_loss: input_lengths must be integral");
    }
    if (target_lengths.dtype() != DType::Int64 &&
        target_lengths.dtype() != DType::Int32) {
        TP_THROW(TypeError, "ctc_loss: target_lengths must be integral");
    }
    return composite::ctc_loss_compose(log_probs, targets, input_lengths, target_lengths,
                            blank, reduction, zero_infinity);
}

TENSORPLAY_LIBRARY_IMPL(CPU, LossKernels) {
    m.impl("nll_loss", nll_loss_kernel);
    m.impl("nll_loss_backward", nll_loss_backward_kernel);
    m.impl("nll_loss2d", nll_loss2d_kernel);
    m.impl("nll_loss2d_backward", nll_loss2d_backward_kernel);
    m.impl("_ctc_loss", _ctc_loss_cpu);
    m.impl("_ctc_loss_backward", _ctc_loss_backward_cpu);
    m.impl("ctc_loss.IntList", ctc_loss_intlist_cpu);
    m.impl("ctc_loss.Tensor", ctc_loss_tensor_cpu);

    m.impl("mse_loss", mse_loss_kernel);
    m.impl("mse_loss_backward", mse_loss_backward_kernel);

    m.impl("tp_l1_loss", l1_loss_kernel);
    m.impl("tp_l1_loss_backward", l1_loss_backward_kernel);
    m.impl("smooth_l1_loss", smooth_l1_loss_cpu);
    m.impl("smooth_l1_loss_backward", smooth_l1_loss_backward_cpu);
    m.impl("huber_loss", huber_loss_cpu);
    m.impl("huber_loss_backward", huber_loss_backward_cpu);
    m.impl("tp_kl_div", kl_div_kernel);
    m.impl("tp_kl_div_backward", kl_div_backward_kernel);
    m.impl("binary_cross_entropy", binary_cross_entropy_cpu);
    m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_cpu);
    m.impl("tp_margin_ranking_loss", margin_ranking_loss_kernel);
    m.impl("tp_margin_ranking_loss_backward", margin_ranking_loss_backward_kernel);
    m.impl("tp_hinge_embedding_loss", hinge_embedding_loss_kernel);
    m.impl("tp_hinge_embedding_loss_backward", hinge_embedding_loss_backward_kernel);
    m.impl("tp_soft_margin_loss", soft_margin_loss_kernel);
    m.impl("tp_soft_margin_loss_backward", soft_margin_loss_backward_kernel);
    m.impl("tp_cosine_embedding_loss", cosine_embedding_loss_kernel);
    m.impl("tp_cosine_embedding_loss_backward", cosine_embedding_loss_backward_kernel);
    m.impl("tp_poisson_nll_loss", poisson_nll_loss_kernel);
    m.impl("tp_poisson_nll_loss_backward", poisson_nll_loss_backward_kernel);
}

} // namespace cpu
} // namespace tensorplay


// -----------------------------------------------------------------------------