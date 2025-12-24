#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include <tuple>
#include <optional>

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


TENSORPLAY_LIBRARY_IMPL(CPU, LossKernels) {
    m.impl("nll_loss", nll_loss_kernel);
    m.impl("nll_loss_backward", nll_loss_backward_kernel);
    m.impl("mse_loss", mse_loss_kernel);
    m.impl("mse_loss_backward", mse_loss_backward_kernel);
}

} // namespace cpu
} // namespace tensorplay
