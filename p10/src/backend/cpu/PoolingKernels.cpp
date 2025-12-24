#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace tensorplay {
namespace cpu {

// Helper to handle optional arguments or defaults
static std::pair<int64_t, int64_t> get_pair(const std::vector<int64_t>& list, int64_t default_val = 0) {
    if (list.empty()) return {default_val, default_val};
    if (list.size() == 1) return {list[0], list[0]};
    return {list[0], list[1]};
}

static std::pair<int64_t, int64_t> get_pair_from_kernel(const std::vector<int64_t>& list, const std::vector<int64_t>& kernel) {
    if (list.empty()) return get_pair(kernel);
    return get_pair(list);
}

Tensor max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d: Expected 4D input");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    auto [dH, dW] = get_pair(dilation, 1);
    
    int64_t H_out, W_out;
    if (ceil_mode) {
        H_out = (int64_t)(std::ceil((float)(H_in + 2 * pH - dH * (kH - 1) - 1) / sH)) + 1;
        W_out = (int64_t)(std::ceil((float)(W_in + 2 * pW - dW * (kW - 1) - 1) / sW)) + 1;
    } else {
        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    }

    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "max_pool2d: Calculated output size is too small");
    
    // Ensure padding doesn't make us start reading out of bounds if ceil_mode used?
    // Usually PyTorch clamps the window end.
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        int64_t h_end = h_start + (kH - 1) * dH + 1;
                        int64_t w_end = w_start + (kW - 1) * dW + 1;
                        
                        // Valid window range
                        // We iterate kernel
                        float max_val = -std::numeric_limits<float>::infinity();
                        
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh * dH;
                                int64_t w_in_idx = w_start + kw * dW;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    float val = in_ptr[idx];
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = max_val;
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "max_pool2d only supports Float32");
    }
    
    return out;
}

Tensor avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d: Expected 4D input");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    
    int64_t H_out, W_out;
    if (ceil_mode) {
        H_out = (int64_t)(std::ceil((float)(H_in + 2 * pH - kH) / sH)) + 1;
        W_out = (int64_t)(std::ceil((float)(W_in + 2 * pW - kW) / sW)) + 1;
    } else {
        H_out = (H_in + 2 * pH - kH) / sH + 1;
        W_out = (W_in + 2 * pW - kW) / sW + 1;
    }

    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "avg_pool2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        int64_t h_end = std::min(h_start + kH, H_in + pH);
                        int64_t w_end = std::min(w_start + kW, W_in + pW);
                        
                        int64_t pool_size = (h_end - h_start) * (w_end - w_start); // This calculation is slightly wrong if we consider padding logic
                        // Let's iterate explicitly
                        
                        float sum = 0.0f;
                        int64_t count = 0;
                        
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh;
                                int64_t w_in_idx = w_start + kw;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    sum += in_ptr[idx];
                                    count++;
                                }
                            }
                        }
                        
                        float divisor;
                        if (divisor_override.has_value()) {
                            divisor = (float)divisor_override.value();
                        } else if (count_include_pad) {
                             divisor = (float)(kH * kW);
                        } else {
                             divisor = (float)count;
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = sum / divisor;
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "avg_pool2d only supports Float32");
    }
    
    return out;
}

Tensor adaptive_avg_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Expected 4D input");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_avg_pool2d: Invalid output size");
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = ((h + 1) * H_in + H_out - 1) / H_out; // Ceil division
                    // Actually usually: floor( (h+1) * H_in / H_out )?
                    // PyTorch: start = floor(i * in / out), end = ceil((i+1) * in / out)
                    // Wait, standard is floor for start and ceil for end.
                    h_end = ((h + 1) * H_in) / H_out; 
                    if (h_end == h_start) h_end += 1; // Ensure at least 1 pixel
                    
                    int64_t kH = h_end - h_start;
                    
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in) / W_out;
                        if (w_end == w_start) w_end += 1;
                        
                        int64_t kW = w_end - w_start;
                        
                        float sum = 0.0f;
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                sum += in_ptr[idx];
                            }
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = sum / (kH * kW);
                    }
                }
            }
        }
    } else {
         TP_THROW(NotImplementedError, "adaptive_avg_pool2d only supports Float32");
    }
    
    return out;
}

Tensor adaptive_max_pool2d_cpu(const Tensor& input, const std::vector<int64_t>& output_size) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d: Expected 4D input");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    auto [H_out, W_out] = get_pair(output_size);
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "adaptive_max_pool2d: Invalid output size");
    
    Tensor out = Tensor::empty({N, C, H_out, W_out}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = ((h + 1) * H_in) / H_out;
                    if (h_end == h_start) h_end += 1;
                    
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in) / W_out;
                        if (w_end == w_start) w_end += 1;
                        
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                float val = in_ptr[idx];
                                if (val > max_val) max_val = val;
                            }
                        }
                        
                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        out_ptr[out_idx] = max_val;
                    }
                }
            }
        }
    } else {
         TP_THROW(NotImplementedError, "adaptive_max_pool2d only supports Float32");
    }
    
    return out;
}

Tensor max_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "max_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);
    auto [dH, dW] = get_pair(dilation, 1);

    Tensor grad_input = Tensor::zeros_like(input);
    
    if (input.dtype() == DType::Float32) {
        float* grad_in_ptr = grad_input.data_ptr<float>();
        const float* grad_out_ptr = grad_output.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        
                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_idx = -1;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh * dH;
                                int64_t w_in_idx = w_start + kw * dW;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    float val = in_ptr[idx];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = idx;
                                    }
                                }
                            }
                        }
                        
                        if (max_idx != -1) {
                            int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                            grad_in_ptr[max_idx] += grad_out_ptr[out_idx];
                        }
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "max_pool2d_backward only supports Float32");
    }
    return grad_input;
}

Tensor avg_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "avg_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    auto [kH, kW] = get_pair(kernel_size);
    auto [sH, sW] = get_pair_from_kernel(stride, kernel_size);
    auto [pH, pW] = get_pair(padding, 0);

    Tensor grad_input = Tensor::zeros_like(input);

    if (input.dtype() == DType::Float32) {
        float* grad_in_ptr = grad_input.data_ptr<float>();
        const float* grad_out_ptr = grad_output.data_ptr<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t h_start = h * sH - pH;
                        int64_t w_start = w * sW - pW;
                        int64_t h_end = std::min(h_start + kH, H_in + pH);
                        int64_t w_end = std::min(w_start + kW, W_in + pW);
                        
                        float divisor;
                         // Recalculate divisor logic from forward
                        if (divisor_override.has_value()) {
                            divisor = (float)divisor_override.value();
                        } else if (count_include_pad) {
                             divisor = (float)(kH * kW);
                        } else {
                            // Calculate count excluding pad
                            int64_t count = 0;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t h_in_idx = h_start + kh;
                                    int64_t w_in_idx = w_start + kw;
                                    if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                        count++;
                                    }
                                }
                            }
                            divisor = (float)count;
                        }

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        float grad_val = grad_out_ptr[out_idx] / divisor;

                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t h_in_idx = h_start + kh;
                                int64_t w_in_idx = w_start + kw;
                                
                                if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                    int64_t idx = ((n * C + c) * H_in + h_in_idx) * W_in + w_in_idx;
                                    grad_in_ptr[idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "avg_pool2d_backward only supports Float32");
    }
    return grad_input;
}

Tensor adaptive_avg_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "adaptive_avg_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    Tensor grad_input = Tensor::zeros_like(input);

    if (input.dtype() == DType::Float32) {
        float* grad_in_ptr = grad_input.data_ptr<float>();
        const float* grad_out_ptr = grad_output.data_ptr<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = ((h + 1) * H_in) / H_out;
                    if (h_end == h_start) h_end += 1;
                    int64_t kH = h_end - h_start;

                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in) / W_out;
                        if (w_end == w_start) w_end += 1;
                        int64_t kW = w_end - w_start;

                        int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                        float grad_val = grad_out_ptr[out_idx] / (kH * kW);

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                grad_in_ptr[idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "adaptive_avg_pool2d_backward only supports Float32");
    }
    return grad_input;
}

Tensor adaptive_max_pool2d_backward_cpu(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dim() != 4 || input.dim() != 4) TP_THROW(RuntimeError, "adaptive_max_pool2d_backward: Expected 4D input and grad_output");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    Tensor grad_input = Tensor::zeros_like(input);

    if (input.dtype() == DType::Float32) {
        float* grad_in_ptr = grad_input.data_ptr<float>();
        const float* grad_out_ptr = grad_output.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H_out; ++h) {
                    int64_t h_start = (h * H_in) / H_out;
                    int64_t h_end = ((h + 1) * H_in) / H_out;
                    if (h_end == h_start) h_end += 1;
                    
                    for (int64_t w = 0; w < W_out; ++w) {
                        int64_t w_start = (w * W_in) / W_out;
                        int64_t w_end = ((w + 1) * W_in) / W_out;
                        if (w_end == w_start) w_end += 1;
                        
                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_idx = -1;

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                int64_t idx = ((n * C + c) * H_in + ih) * W_in + iw;
                                float val = in_ptr[idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        if (max_idx != -1) {
                            int64_t out_idx = ((n * C + c) * H_out + h) * W_out + w;
                            grad_in_ptr[max_idx] += grad_out_ptr[out_idx];
                        }
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "adaptive_max_pool2d_backward only supports Float32");
    }
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CPU, PoolingKernels) {
    m.impl("max_pool2d", max_pool2d_cpu);
    m.impl("avg_pool2d", avg_pool2d_cpu);
    m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_cpu);
    m.impl("adaptive_max_pool2d", adaptive_max_pool2d_cpu);
    m.impl("max_pool2d_backward", max_pool2d_backward_cpu);
    m.impl("avg_pool2d_backward", avg_pool2d_backward_cpu);
    m.impl("adaptive_avg_pool2d_backward", adaptive_avg_pool2d_backward_cpu);
    m.impl("adaptive_max_pool2d_backward", adaptive_max_pool2d_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay
