#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tensorplay {
namespace cpu {

// Helper to check input validity
static void check_dims(const Tensor& input, int64_t expected_dim, const char* name) {
    if (input.dim() != expected_dim) {
        TP_THROW(RuntimeError, std::string(name) + ": Expected " + std::to_string(expected_dim) + "D input");
    }
}

// Forward declarations
Tensor batch_norm_cpu(const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                      std::optional<Tensor>& running_mean_opt, std::optional<Tensor>& running_var_opt, 
                      bool training, double momentum, double eps);

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(const Tensor& grad_output, const Tensor& input, 
                               const std::optional<Tensor>& weight_opt, 
                               const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, 
                               bool training, double eps);

// Backward for GroupNorm
std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cpu(const Tensor& grad_output, const Tensor& input, 
                              int64_t num_groups, 
                              const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                              double eps) {
    // Reusing LayerNorm backward logic or implementing similar logic
    // GroupNorm(N, C, ...) -> Reshape to (N, G, C/G, ...) -> LayerNorm over (C/G, ...)
    
    // For simplicity, implementing directly.
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    
    if (C % num_groups != 0) TP_THROW(RuntimeError, "group_norm_backward: C not divisible by num_groups");
    
    int64_t G = num_groups;
    int64_t D = C / G;
    
    int64_t numel = input.numel();
    int64_t spatial_size = numel / (N * C);
    int64_t group_size = D * spatial_size; // Normalization size
    
    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "group_norm_backward only supports Float32");

    Tensor grad_input = Tensor::empty_like(input);
    Tensor grad_weight;
    Tensor grad_bias;
    
    if (weight_opt.has_value() && weight_opt->defined()) grad_weight = Tensor::empty_like(*weight_opt);
    if (bias_opt.has_value() && bias_opt->defined()) grad_bias = Tensor::empty_like(*bias_opt);
    
    float* grad_in_ptr = grad_input.data_ptr<float>();
    const float* grad_out_ptr = grad_output.data_ptr<float>();
    const float* in_ptr = input.data_ptr<float>();
    
    float* gw_ptr = (grad_weight.defined()) ? grad_weight.data_ptr<float>() : nullptr;
    float* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<float>() : nullptr;
    const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;

    // Initialize grad_weight/grad_bias to 0
    if (gw_ptr) std::fill(gw_ptr, gw_ptr + C, 0.0f);
    if (gb_ptr) std::fill(gb_ptr, gb_ptr + C, 0.0f);

    // Iterate over (N, G)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < G; ++g) {
             int64_t group_offset = n * C * spatial_size + g * D * spatial_size;
             
             // 1. Recompute statistics for this group
             float sum = 0.0f;
             float sq_sum = 0.0f;
             
             // Iterate over D channels in this group, and all spatial pixels
             for (int64_t d = 0; d < D; ++d) {
                 int64_t c = g * D + d; // absolute channel index
                 int64_t c_offset = group_offset + d * spatial_size;
                 
                 for (int64_t s = 0; s < spatial_size; ++s) {
                     float val = in_ptr[c_offset + s];
                     sum += val;
                     sq_sum += val * val;
                 }
             }
             
             float mean = sum / group_size;
             float var = (sq_sum / group_size) - (mean * mean);
             float inv_std = 1.0f / std::sqrt(var + (float)eps);
             
             // 2. Compute local gradients
             float s_dy = 0.0f;
             float s_dy_x_hat = 0.0f;
             
             for (int64_t d = 0; d < D; ++d) {
                 int64_t c = g * D + d;
                 int64_t c_offset = group_offset + d * spatial_size;
                 float w = (w_ptr) ? w_ptr[c] : 1.0f;
                 
                 for (int64_t s = 0; s < spatial_size; ++s) {
                     float dy = grad_out_ptr[c_offset + s];
                     float x = in_ptr[c_offset + s];
                     float x_hat = (x - mean) * inv_std;
                     
                     if (gw_ptr) gw_ptr[c] += dy * x_hat;
                     if (gb_ptr) gb_ptr[c] += dy;
                     
                     float dy_eff = dy * w;
                     s_dy += dy_eff;
                     s_dy_x_hat += dy_eff * x_hat;
                 }
             }
             
             // 3. Compute grad_input
             float term1 = inv_std / group_size;
             float M = (float)group_size;
             
             for (int64_t d = 0; d < D; ++d) {
                 int64_t c = g * D + d;
                 int64_t c_offset = group_offset + d * spatial_size;
                 float w = (w_ptr) ? w_ptr[c] : 1.0f;
                 
                 for (int64_t s = 0; s < spatial_size; ++s) {
                     float dy = grad_out_ptr[c_offset + s];
                     float x = in_ptr[c_offset + s];
                     float x_hat = (x - mean) * inv_std;
                     float dy_eff = dy * w;
                     
                     grad_in_ptr[c_offset + s] = term1 * (M * dy_eff - s_dy - x_hat * s_dy_x_hat);
                 }
             }
        }
    }
    
    if (!weight_opt.has_value() || !weight_opt->defined()) grad_weight = Tensor();
    if (!bias_opt.has_value() || !bias_opt->defined()) grad_bias = Tensor();

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Backward for InstanceNorm
std::tuple<Tensor, Tensor, Tensor> instance_norm_backward_cpu(const Tensor& grad_output, const Tensor& input, 
                              const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
                              const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, 
                              bool use_input_stats, double eps) {
    
    if (use_input_stats) {
        // Use GroupNorm backward with G=C
        int64_t C = input.size(1);
        return group_norm_backward_cpu(grad_output, input, C, weight_opt, bias_opt, eps);
    } else {
        // Use BatchNorm backward (eval mode)
        // training=false
        return batch_norm_backward_cpu(grad_output, input, weight_opt, running_mean_opt, running_var_opt, false, eps);
    }
}



// ============================================================================
// Batch Normalization
// ============================================================================

// Forward
// input: (N, C, H, W) or (N, C, L) or (N, C)
// weight: (C)
// bias: (C)
// running_mean: (C)
// running_var: (C)
Tensor batch_norm_cpu(const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                      std::optional<Tensor>& running_mean_opt, std::optional<Tensor>& running_var_opt, 
                      bool training, double momentum, double eps) {
    
    // Support 2D (N, C), 3D (N, C, L), 4D (N, C, H, W)
    if (input.dim() < 2 || input.dim() > 5) TP_THROW(RuntimeError, "batch_norm: Input must be between 2D and 5D");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t numel = input.numel();
    int64_t spatial_size = numel / (N * C);

    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "batch_norm only supports Float32");

    Tensor out = Tensor::empty_like(input);
    float* out_ptr = out.data_ptr<float>();
    const float* in_ptr = input.data_ptr<float>();

    const float* weight_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
    const float* bias_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
    
    // We need mean and var
    std::vector<float> mean(C, 0.0f);
    std::vector<float> var(C, 0.0f);
    std::vector<float> inv_std(C, 0.0f);

    if (training) {
        // Calculate mean and var per channel
        // Iterate over N and spatial dims for each C
        // Optimization: Use separate loops for better cache locality if possible?
        // Standard layout is (N, C, ...) so C is not contiguous.
        
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            float sq_sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = n * C * spatial_size + c * spatial_size + s;
                    float val = in_ptr[idx];
                    sum += val;
                    sq_sum += val * val;
                }
            }
            float count = (float)(N * spatial_size);
            float mu = sum / count;
            float sigma2 = (sq_sum / count) - (mu * mu);
            
            mean[c] = mu;
            var[c] = sigma2;
            inv_std[c] = 1.0f / std::sqrt(sigma2 + (float)eps);
        }

        // Update running stats
        if (running_mean_opt.has_value() && running_mean_opt->defined() && 
            running_var_opt.has_value() && running_var_opt->defined()) {
            
            float* rm_ptr = running_mean_opt->data_ptr<float>();
            float* rv_ptr = running_var_opt->data_ptr<float>();
            
            // Unbiased var for running stats? PyTorch uses unbiased=False for batch stats but unbiased for running var updates?
            // Actually PyTorch doc says: running_mean = (1 - m) * running_mean + m * batch_mean
            // running_var = (1 - m) * running_var + m * batch_var (unbiased?)
            // Usually batch_var is biased in calculation, but running_var update might use unbiased.
            // PyTorch default momentum is 0.1.
            // Let's assume simple update for now.
            
            float m = (float)momentum;
            float count = (float)(N * spatial_size);
            float unbiased_scale = (count > 1.0f) ? (count / (count - 1.0f)) : 1.0f;

            for (int64_t c = 0; c < C; ++c) {
                rm_ptr[c] = (1.0f - m) * rm_ptr[c] + m * mean[c];
                rv_ptr[c] = (1.0f - m) * rv_ptr[c] + m * var[c] * unbiased_scale;
            }
        }
        
    } else {
        // Inference: use running stats
        if (running_mean_opt.has_value() && running_mean_opt->defined() && 
            running_var_opt.has_value() && running_var_opt->defined()) {
            
            const float* rm_ptr = running_mean_opt->data_ptr<float>();
            const float* rv_ptr = running_var_opt->data_ptr<float>();
            
            for (int64_t c = 0; c < C; ++c) {
                mean[c] = rm_ptr[c];
                var[c] = rv_ptr[c]; // This is variance
                inv_std[c] = 1.0f / std::sqrt(var[c] + (float)eps);
            }
        } else {
            // No running stats? behave like instance norm or error?
            // Fallback to batch stats if no running stats provided?
            // For now, assume provided or zero.
             for (int64_t c = 0; c < C; ++c) {
                inv_std[c] = 1.0f / std::sqrt((float)eps); // mean=0, var=0
            }
        }
    }

    // Apply normalization
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float mu = mean[c];
            float scale = inv_std[c];
            float w = (weight_ptr) ? weight_ptr[c] : 1.0f;
            float b = (bias_ptr) ? bias_ptr[c] : 0.0f;
            
            // Combined scale and bias
            // y = (x - mu) * scale * w + b
            //   = x * (scale * w) + (b - mu * scale * w)
            float effective_scale = scale * w;
            float effective_bias = b - mu * effective_scale;

            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = n * C * spatial_size + c * spatial_size + s;
                out_ptr[idx] = in_ptr[idx] * effective_scale + effective_bias;
            }
        }
    }

    return out;
}

// Backward
// Simplified: Recompute mean/var if needed
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(const Tensor& grad_output, const Tensor& input, 
                               const std::optional<Tensor>& weight_opt, 
                               const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, 
                               bool training, double eps) {
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t numel = input.numel();
    int64_t spatial_size = numel / (N * C);

    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "batch_norm_backward only supports Float32");

    Tensor grad_input = Tensor::empty_like(input);
    Tensor grad_weight;
    Tensor grad_bias;
    
    if (weight_opt.has_value() && weight_opt->defined()) {
        grad_weight = Tensor::empty_like(*weight_opt);
    }
    if (weight_opt.has_value() && weight_opt->defined()) { // Bias grad shape is same as weight
         grad_bias = Tensor::empty_like(*weight_opt); // Assuming weight and bias have same shape (C)
    }

    const float* grad_out_ptr = grad_output.data_ptr<float>();
    const float* in_ptr = input.data_ptr<float>();
    float* grad_in_ptr = grad_input.data_ptr<float>();
    
    float* gw_ptr = (grad_weight.defined()) ? grad_weight.data_ptr<float>() : nullptr;
    float* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<float>() : nullptr;
    
    const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
    
    // We need mean and inv_std again
    std::vector<float> mean(C, 0.0f);
    std::vector<float> inv_std(C, 0.0f);
    
    if (training) {
        // Recompute mean/inv_std from input
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0.0f;
            float sq_sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = n * C * spatial_size + c * spatial_size + s;
                    float val = in_ptr[idx];
                    sum += val;
                    sq_sum += val * val;
                }
            }
            float count = (float)(N * spatial_size);
            float mu = sum / count;
            float var = (sq_sum / count) - (mu * mu);
            mean[c] = mu;
            inv_std[c] = 1.0f / std::sqrt(var + (float)eps);
        }
    } else {
        // Use running stats
         if (running_mean_opt.has_value() && running_mean_opt->defined() && 
            running_var_opt.has_value() && running_var_opt->defined()) {
            const float* rm_ptr = running_mean_opt->data_ptr<float>();
            const float* rv_ptr = running_var_opt->data_ptr<float>();
            for (int64_t c = 0; c < C; ++c) {
                mean[c] = rm_ptr[c];
                inv_std[c] = 1.0f / std::sqrt(rv_ptr[c] + (float)eps);
            }
        } else {
             for (int64_t c = 0; c < C; ++c) {
                mean[c] = 0.0f;
                inv_std[c] = 1.0f / std::sqrt((float)eps);
            }
        }
    }
    
    // 1. Compute grad_weight and grad_bias (always needed if defined)
    // dL/dw = sum(dL/dy * x_hat)
    // dL/db = sum(dL/dy)
    // x_hat = (x - mean) * inv_std
    
    // Intermediate storage for dL/dy sum and dL/dy * x sum per channel
    std::vector<float> sum_dy(C, 0.0f);
    std::vector<float> sum_dy_x(C, 0.0f); // sum(dy * x)  -> needed for training mode gradient derivation
    
    // Actually for grad_weight: sum(dL/dy * x_hat)
    // sum_dy_x_hat = sum(dL/dy * (x - mean) * inv_std) = inv_std * (sum(dL/dy * x) - mean * sum(dL/dy))
    
    for (int64_t c = 0; c < C; ++c) {
        float s_dy = 0.0f;
        float s_dy_x = 0.0f;
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = n * C * spatial_size + c * spatial_size + s;
                float dy = grad_out_ptr[idx];
                float x = in_ptr[idx];
                s_dy += dy;
                s_dy_x += dy * x;
            }
        }
        sum_dy[c] = s_dy;
        sum_dy_x[c] = s_dy_x;
        
        if (gb_ptr) gb_ptr[c] = s_dy;
        if (gw_ptr) {
             // sum(dy * (x - mu) * inv_std)
             gw_ptr[c] = (s_dy_x - mean[c] * s_dy) * inv_std[c];
        }
    }
    
    // 2. Compute grad_input
    // If training:
    // dL/dx = (1 / (N*inv_std)) * (N*dL/dx_hat - sum(dL/dx_hat) - x_hat*sum(dL/dx_hat * x_hat))  <-- standard BN backward formula?
    // Let's use the explicit formula:
    // dL/dx_hat = dL/dy * gamma
    // dL/dvar = sum(dL/dx_hat * (x - mu) * (-0.5 * (var+eps)^-1.5))
    // dL/dmu = sum(dL/dx_hat * (-inv_std)) + dL/dvar * (-2/N * sum(x-mu)) --> sum term is 0
    // Actually let's use the simpler form for BN backward:
    // dL/dx = (gamma * inv_std / M) * (M * dy - sum(dy) - x_hat * sum(dy * x_hat))
    // where M = N * spatial_size
    // dy here is grad_output
    
    float M = (float)(N * spatial_size);
    
    for (int64_t c = 0; c < C; ++c) {
        float gamma = (w_ptr) ? w_ptr[c] : 1.0f;
        float inv_s = inv_std[c];
        float mu = mean[c];
        
        if (training) {
            float s_dy = sum_dy[c];
            float s_dy_x_hat = (gw_ptr) ? gw_ptr[c] : (sum_dy_x[c] - mu * s_dy) * inv_s; 
            // Note: if gw_ptr calculated above, it is exactly sum(dy * x_hat)
            // wait, if w_ptr is present, gw_ptr is calculated.
            // If w_ptr is NOT present (gamma=1), we still need sum(dy * x_hat).
            
            float term1 = gamma * inv_s / M;
            
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = n * C * spatial_size + c * spatial_size + s;
                    float dy = grad_out_ptr[idx];
                    float x = in_ptr[idx];
                    float x_hat = (x - mu) * inv_s;
                    
                    // dL/dx = term1 * (M * dy - s_dy - x_hat * s_dy_x_hat)
                    grad_in_ptr[idx] = term1 * (M * dy - s_dy - x_hat * s_dy_x_hat);
                }
            }
        } else {
            // Eval mode: simple scale and shift derivative
            // y = (x - mu) * inv_std * gamma + beta
            // dL/dx = dL/dy * gamma * inv_std
            float scale = gamma * inv_s;
             for (int64_t n = 0; n < N; ++n) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = n * C * spatial_size + c * spatial_size + s;
                    grad_in_ptr[idx] = grad_out_ptr[idx] * scale;
                }
            }
        }
    }
    
    if (!weight_opt.has_value() || !weight_opt->defined()) grad_weight = Tensor();
    // For batch_norm, if weight is present, we assume bias is present (affine=True) or we just return grad_bias anyway.
    // Since we don't pass bias in arguments, we can't check it. 
    // Usually if weight is defined, we compute grad_bias.
    if (!weight_opt.has_value() || !weight_opt->defined()) grad_bias = Tensor();

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// Layer Normalization
// ============================================================================

Tensor layer_norm_cpu(const Tensor& input, const std::vector<int64_t>& normalized_shape, 
                      const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                      double eps) {
    
    // normalized_shape defines the last D dimensions to normalize over.
    // e.g. input (N, C, H, W), normalized_shape (C, H, W) -> normalize over C,H,W (per N)
    // e.g. input (N, L, D), normalized_shape (D) -> normalize over D (per N, L)
    
    int64_t norm_ndim = normalized_shape.size();
    int64_t input_ndim = input.dim();
    
    if (norm_ndim > input_ndim) TP_THROW(RuntimeError, "layer_norm: normalized_shape dim larger than input dim");
    
    // Check shapes match last dims
    int64_t outer_dims = input_ndim - norm_ndim;
    int64_t inner_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[i]) {
            TP_THROW(RuntimeError, "layer_norm: Input shape mismatch with normalized_shape");
        }
        inner_size *= normalized_shape[i];
    }
    
    int64_t outer_size = input.numel() / inner_size;
    
    Tensor out = Tensor::empty_like(input);
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
        const float* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
        
        for (int64_t i = 0; i < outer_size; ++i) {
            // Compute mean/var for this block
            float sum = 0.0f;
            float sq_sum = 0.0f;
            int64_t offset = i * inner_size;
            
            for (int64_t j = 0; j < inner_size; ++j) {
                float val = in_ptr[offset + j];
                sum += val;
                sq_sum += val * val;
            }
            
            float mean = sum / inner_size;
            float var = (sq_sum / inner_size) - (mean * mean);
            float inv_std = 1.0f / std::sqrt(var + (float)eps);
            
            for (int64_t j = 0; j < inner_size; ++j) {
                float val = in_ptr[offset + j];
                float normalized = (val - mean) * inv_std;
                
                if (w_ptr) normalized *= w_ptr[j];
                if (b_ptr) normalized += b_ptr[j];
                
                out_ptr[offset + j] = normalized;
            }
        }
    } else {
        TP_THROW(NotImplementedError, "layer_norm only supports Float32");
    }
    
    return out;
}

// ============================================================================
// Group Normalization
// ============================================================================

Tensor group_norm_cpu(const Tensor& input, int64_t num_groups, 
                      const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                      double eps) {
    
    // input: (N, C, *)
    if (input.dim() < 2) TP_THROW(RuntimeError, "group_norm requires at least 2 dims");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    
    if (C % num_groups != 0) TP_THROW(RuntimeError, "group_norm: num_channels must be divisible by num_groups");
    
    int64_t channels_per_group = C / num_groups;
    int64_t spatial_size = input.numel() / (N * C);
    
    // Effectively we reshape (N, G, C/G, *) and normalize over (C/G, *)
    // inner_size = (C/G) * spatial_size
    int64_t inner_size = channels_per_group * spatial_size;
    
    Tensor out = Tensor::empty_like(input);
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
        const float* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < num_groups; ++g) {
                // Compute mean/var for this group
                float sum = 0.0f;
                float sq_sum = 0.0f;
                
                // Group g covers channels [g*channels_per_group, (g+1)*channels_per_group)
                int64_t c_start = g * channels_per_group;
                
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t current_c = c_start + c;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = n * C * spatial_size + current_c * spatial_size + s;
                        float val = in_ptr[idx];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
                
                float mean = sum / inner_size;
                float var = (sq_sum / inner_size) - (mean * mean);
                float inv_std = 1.0f / std::sqrt(var + (float)eps);
                
                // Apply
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    int64_t current_c = c_start + c;
                    
                    float w = (w_ptr) ? w_ptr[current_c] : 1.0f;
                    float b = (b_ptr) ? b_ptr[current_c] : 0.0f;
                    
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = n * C * spatial_size + current_c * spatial_size + s;
                        float val = in_ptr[idx];
                        float normalized = (val - mean) * inv_std;
                        out_ptr[idx] = normalized * w + b;
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "group_norm only supports Float32");
    }
    
    return out;
}

// ============================================================================
// Instance Normalization
// ============================================================================

Tensor instance_norm_cpu(const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
                         std::optional<Tensor>& running_mean_opt, std::optional<Tensor>& running_var_opt,
                         bool use_input_stats, double momentum, double eps) {
    
    // Instance Norm is Group Norm with num_groups = C
    // But it also has optional running stats tracking (mostly for tracking, not used in inference usually unless track_running_stats=True and training=False?)
    // Actually PyTorch InstanceNorm:
    // "InstanceNorm is applied per channel of each sample."
    // "If track_running_stats is set to True, during training this layer keeps running estimates of its computed mean and variance, which are then used for evaluation."
    // "If track_running_stats is set to False, this layer does not keep running estimates, and batch statistics are always used during evaluation."
    
    // So if use_input_stats=True (Training or track_running_stats=False), we compute stats.
    // If use_input_stats=False (Eval with track_running_stats=True), we use running stats.
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    
    if (use_input_stats) {
        // Equivalent to GroupNorm(num_groups=C)
        // But we might need to update running stats
        // Running stats for InstanceNorm are usually averaged over N as well?
        // PyTorch docs: "The running mean and variance are computed using the momentum strategy... based on the values of the current mini-batch."
        
        // Let's reuse GroupNorm logic for calculation, but handle running stats manually?
        // GroupNorm doesn't take running stats.
        
        // Optimization: Just call group_norm_cpu with G=C?
        // But we need to update running stats.
        // Also group_norm doesn't support running_stats arguments.
        
        // Let's implement directly.
        int64_t spatial_size = input.numel() / (N * C);
        Tensor out = Tensor::empty_like(input);
        
        if (input.dtype() == DType::Float32) {
             float* out_ptr = out.data_ptr<float>();
             const float* in_ptr = input.data_ptr<float>();
             const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
             const float* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
             
             // Temp storage for batch mean/var (averaged over N) to update running stats
             std::vector<float> batch_mean(C, 0.0f);
             std::vector<float> batch_var(C, 0.0f);
             
             for (int64_t n = 0; n < N; ++n) {
                 for (int64_t c = 0; c < C; ++c) {
                     float sum = 0.0f;
                     float sq_sum = 0.0f;
                     int64_t offset = n * C * spatial_size + c * spatial_size;
                     
                     for (int64_t s = 0; s < spatial_size; ++s) {
                         float val = in_ptr[offset + s];
                         sum += val;
                         sq_sum += val * val;
                     }
                     
                     float mean = sum / spatial_size;
                     float var = (sq_sum / spatial_size) - (mean * mean);
                     float inv_std = 1.0f / std::sqrt(var + (float)eps);
                     
                     // Accumulate for running stats
                     batch_mean[c] += mean;
                     batch_var[c] += var;
                     
                     float w = (w_ptr) ? w_ptr[c] : 1.0f;
                     float b = (b_ptr) ? b_ptr[c] : 0.0f;
                     
                     for (int64_t s = 0; s < spatial_size; ++s) {
                         float val = in_ptr[offset + s];
                         out_ptr[offset + s] = (val - mean) * inv_std * w + b;
                     }
                 }
             }
             
             // Update running stats
             if (running_mean_opt.has_value() && running_mean_opt->defined()) {
                 float* rm = running_mean_opt->data_ptr<float>();
                 float* rv = running_var_opt->data_ptr<float>();
                 float m = (float)momentum;
                 
                 for (int64_t c = 0; c < C; ++c) {
                     float bm = batch_mean[c] / N;
                     float bv = batch_var[c] / N; // Average variance across batch? Or variance of combined? PyTorch does average.
                     // Note: Unbiased?
                     float unbiased_scale = (spatial_size > 1) ? ((float)spatial_size / (spatial_size - 1)) : 1.0f;
                     
                     rm[c] = (1.0f - m) * rm[c] + m * bm;
                     rv[c] = (1.0f - m) * rv[c] + m * bv * unbiased_scale;
                 }
             }
             
        } else {
            TP_THROW(NotImplementedError, "instance_norm only supports Float32");
        }
        return out;
        
    } else {
        // Use running stats
        // This is exactly like BatchNorm eval mode!
        // Except weight/bias are per channel.
        return batch_norm_cpu(input, weight_opt, bias_opt, running_mean_opt, running_var_opt, false, momentum, eps);
    }
}


// Backward for LayerNorm
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(const Tensor& grad_output, const Tensor& input, 
                              const std::vector<int64_t>& normalized_shape, 
                              const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, 
                              double eps) {
    
    int64_t norm_ndim = normalized_shape.size();
    int64_t input_ndim = input.dim();
    int64_t outer_dims = input_ndim - norm_ndim;
    int64_t inner_size = 1;
    for (auto s : normalized_shape) inner_size *= s;
    int64_t outer_size = input.numel() / inner_size;
    
    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "layer_norm_backward only supports Float32");
    
    Tensor grad_input = Tensor::empty_like(input);
    Tensor grad_weight;
    Tensor grad_bias;
    
    if (weight_opt.has_value() && weight_opt->defined()) grad_weight = Tensor::empty_like(*weight_opt);
    if (bias_opt.has_value() && bias_opt->defined()) grad_bias = Tensor::empty_like(*bias_opt);
    
    float* grad_in_ptr = grad_input.data_ptr<float>();
    const float* grad_out_ptr = grad_output.data_ptr<float>();
    const float* in_ptr = input.data_ptr<float>();
    
    float* gw_ptr = (grad_weight.defined()) ? grad_weight.data_ptr<float>() : nullptr;
    float* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<float>() : nullptr;
    const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
    
    // Initialize grad_weight/grad_bias to 0
    if (gw_ptr) std::fill(gw_ptr, gw_ptr + inner_size, 0.0f);
    if (gb_ptr) std::fill(gb_ptr, gb_ptr + inner_size, 0.0f);
    
    // We iterate over outer_size (batches/sequences), and for each, compute gradients.
    // Unlike BN, LN statistics are computed per-element in outer loop.
    
    for (int64_t i = 0; i < outer_size; ++i) {
        int64_t offset = i * inner_size;
        
        // 1. Recompute statistics
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int64_t j = 0; j < inner_size; ++j) {
            float val = in_ptr[offset + j];
            sum += val;
            sq_sum += val * val;
        }
        float mean = sum / inner_size;
        float var = (sq_sum / inner_size) - (mean * mean);
        float inv_std = 1.0f / std::sqrt(var + (float)eps);
        
        // 2. Compute local gradients
        float s_dy = 0.0f;
        float s_dy_x_hat = 0.0f;
        
        for (int64_t j = 0; j < inner_size; ++j) {
            float dy = grad_out_ptr[offset + j];
            float x = in_ptr[offset + j];
            float x_hat = (x - mean) * inv_std;
            
            // Accumulate grad_weight / grad_bias
            if (gw_ptr) gw_ptr[j] += dy * x_hat;
            if (gb_ptr) gb_ptr[j] += dy;
            
            // For grad_input calculation:
            // dL/dx depends on dL/dy * gamma.
            // Let dy_eff = dL/dy * gamma
            float gamma = (w_ptr) ? w_ptr[j] : 1.0f;
            float dy_eff = dy * gamma;
            
            s_dy += dy_eff;
            s_dy_x_hat += dy_eff * x_hat;
        }
        
        // 3. Compute grad_input for this block
        float term1 = inv_std / inner_size;
        float M = (float)inner_size;
        
        for (int64_t j = 0; j < inner_size; ++j) {
            float dy = grad_out_ptr[offset + j];
            float x = in_ptr[offset + j];
            float x_hat = (x - mean) * inv_std;
            float gamma = (w_ptr) ? w_ptr[j] : 1.0f;
            float dy_eff = dy * gamma;
            
            grad_in_ptr[offset + j] = term1 * (M * dy_eff - s_dy - x_hat * s_dy_x_hat);
        }
    }
    
    if (!weight_opt.has_value() || !weight_opt->defined()) grad_weight = Tensor();
    if (!bias_opt.has_value() || !bias_opt->defined()) grad_bias = Tensor();

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Registration
TENSORPLAY_LIBRARY_IMPL(CPU, NormalizationKernels) {
    m.impl("batch_norm", batch_norm_cpu);
    m.impl("layer_norm", layer_norm_cpu);
    m.impl("group_norm", group_norm_cpu);
    m.impl("instance_norm", instance_norm_cpu);
    
    m.impl("batch_norm_backward", batch_norm_backward_cpu);
    m.impl("layer_norm_backward", layer_norm_backward_cpu);
    m.impl("group_norm_backward", group_norm_backward_cpu);
    m.impl("instance_norm_backward", instance_norm_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
