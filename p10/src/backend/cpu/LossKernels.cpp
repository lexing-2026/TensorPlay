#include "Tensor.h"
#include "Generator.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include <tuple>
#include <optional>
#include <cmath>
#include <vector>

namespace tensorplay {
namespace cpu {

// NLL Loss
std::tuple<Tensor, Tensor> nll_loss_kernel(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    // reduction: 0=none, 1=mean, 2=sum
    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "nll_loss only supports Float32");
    if (target.dtype() != DType::Int64) TP_THROW(NotImplementedError, "nll_loss target must be Int64");
    
    int64_t n_batch = input.size(0);
    int64_t n_classes = input.size(1);
    
    const float* input_data = input.data_ptr<float>();
    const int64_t* target_data = target.data_ptr<int64_t>();
    const float* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<float>() : nullptr;
    
    std::vector<float> output_data(n_batch);
    float total_weight = 0;
    
    for (int64_t i = 0; i < n_batch; ++i) {
        int64_t t = target_data[i];
        if (t == ignore_index) {
            output_data[i] = 0;
            continue;
        }
        if (t < 0 || t >= n_classes) TP_THROW(RuntimeError, "Target out of bounds");
        
        float w = weight_data ? weight_data[t] : 1.0f;
        output_data[i] = -input_data[i * n_classes + t] * w;
        total_weight += w;
    }
    
    Tensor total_weight_tensor = Tensor::tensor({total_weight}, DType::Float32, input.device());

    if (reduction == 0) { // None
        return std::make_tuple(Tensor::tensor(output_data, DType::Float32, input.device()), total_weight_tensor);
    } else if (reduction == 1) { // Mean
        double sum = 0;
        for (float x : output_data) sum += x;
        if (total_weight > 0) sum /= total_weight;
        return std::make_tuple(Tensor::tensor({(float)sum}, DType::Float32, input.device()).reshape({}), total_weight_tensor);
    } else if (reduction == 2) { // Sum
        double sum = 0;
        for (float x : output_data) sum += x;
        return std::make_tuple(Tensor::tensor({(float)sum}, DType::Float32, input.device()).reshape({}), total_weight_tensor);
    }
    TP_THROW(ValueError, "Invalid reduction mode");
}

Tensor nll_loss_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    int64_t n_batch = input.size(0);
    int64_t n_classes = input.size(1);
    
    Tensor grad_input = Tensor::zeros({n_batch, n_classes}, input.dtype(), input.device());
    float* grad_input_data = grad_input.data_ptr<float>();
    
    const int64_t* target_data = target.data_ptr<int64_t>();
    const float* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<float>() : nullptr;
    
    double tw = 0;
    if (total_weight.defined()) {
        tw = total_weight.item<float>();
    } else if (reduction == 1) {
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t != ignore_index) {
                tw += weight_data ? weight_data[t] : 1.0f;
            }
        }
    }

    if (reduction == 0) { // None
        const float* grad_out_data = grad_output.data_ptr<float>();
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            float w = weight_data ? weight_data[t] : 1.0f;
            grad_input_data[i * n_classes + t] = -w * grad_out_data[i];
        }
    } else {
        float grad_val = grad_output.item<float>();
        if (reduction == 1) { // Mean
             if (tw > 0) grad_val /= tw;
        }
        
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            float w = weight_data ? weight_data[t] : 1.0f;
            grad_input_data[i * n_classes + t] = -w * grad_val;
        }
    }
    
    return grad_input;
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



// Torch-aligned loss family (reduction-aware forward + explicit backward),
// ported from aten/src/ATen/native/Loss.cpp. Composed from dispatcher ops
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

Tensor smooth_l1_loss_kernel(const Tensor& input, const Tensor& target,
                             int64_t reduction, double beta) {
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    // |x| <= beta ? 0.5*x^2/beta : |x| - 0.5*beta
    Tensor loss = Tensor::where(absd.le(Scalar(beta)), diff * diff * (0.5 / beta),
                                absd - 0.5 * beta);
    return loss_reduce(loss, reduction);
}

Tensor smooth_l1_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& target, int64_t reduction, double beta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(beta)), diff / beta, diff.sign()) * grad_output;
    return scale_grad(g, reduction, input.numel());
}

Tensor huber_loss_kernel(const Tensor& input, const Tensor& target,
                         int64_t reduction, double delta) {
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(delta)), diff * diff * 0.5,
                                delta * (absd - 0.5 * delta));
    return loss_reduce(loss, reduction);
}

Tensor huber_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
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

Tensor binary_cross_entropy_kernel(const Tensor& input, const Tensor& target,
                                   const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor x = input.clamp(0.0, 1.0);
    Tensor loss = -(x.log() * target + (-x + 1.0).log() * (-target + 1.0));
    if (weight.has_value() && weight->defined()) loss = loss * weight.value();
    return loss_reduce(loss, reduction);
}

Tensor binary_cross_entropy_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target,
                                            const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor x = input.clamp(0.0, 1.0);
    Tensor eps = Tensor::full_like(x, 1e-12);
    Tensor g = (x - target) / Tensor::maximum(x * (-x + 1.0), eps);
    if (weight.has_value() && weight->defined()) g = g * weight.value();
    g = g * grad_output;
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
    Tensor z = zeros_like_shim(input);
    Tensor loss = Tensor::where(target.eq(1), input,
                  Tensor::where(target.eq(-1), clamp_min_shim(margin - input, Scalar(0.0)), z));
    return loss_reduce(loss, reduction);
}

Tensor hinge_embedding_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target, double margin, int64_t reduction) {
    Tensor z = zeros_like_shim(input);
    Tensor g = Tensor::where(target.eq(1), Tensor::ones_like(input),
               Tensor::where(target.eq(-1), ((margin - input).gt(Scalar(0.0))).to(input.dtype()), z));
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
    Tensor loss = Tensor::where(target.eq(1), zero + (1.0 - cos),
                  Tensor::where(target.eq(-1), clamp_min_shim(cos - margin, Scalar(margin)),
                                zero + clamp_min_shim(1.0 - cos - margin, Scalar(0.0))));
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

TENSORPLAY_LIBRARY_IMPL(CPU, LossKernels) {
    m.impl("nll_loss", nll_loss_kernel);
    m.impl("nll_loss_backward", nll_loss_backward_kernel);
    m.impl("mse_loss", mse_loss_kernel);
    m.impl("mse_loss_backward", mse_loss_backward_kernel);

    m.impl("tp_l1_loss", l1_loss_kernel);
    m.impl("tp_l1_loss_backward", l1_loss_backward_kernel);
    m.impl("tp_smooth_l1_loss", smooth_l1_loss_kernel);
    m.impl("tp_smooth_l1_loss_backward", smooth_l1_loss_backward_kernel);
    m.impl("tp_huber_loss", huber_loss_kernel);
    m.impl("tp_huber_loss_backward", huber_loss_backward_kernel);
    m.impl("tp_kl_div", kl_div_kernel);
    m.impl("tp_kl_div_backward", kl_div_backward_kernel);
    m.impl("tp_binary_cross_entropy", binary_cross_entropy_kernel);
    m.impl("tp_binary_cross_entropy_backward", binary_cross_entropy_backward_kernel);
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