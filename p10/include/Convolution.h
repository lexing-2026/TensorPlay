#pragma once

// Rank-generic convolution layer.
//
// `convolution` is the single operator that every spatial rank and both the
// direct and transposed forms funnel through.  It owns no math of its own: it
// validates the shape family and forwards to the rank-specialized kernel that
// already selects the best available path (oneDNN on CPU, cuDNN on CUDA).
// The per-rank kernels it forwards to are declared here so the routing layer
// lives in its own translation unit instead of growing the kernel files.

#include "Tensor.h"

#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {

namespace cpu {

Tensor conv1d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                  const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                  const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv3d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                  const std::vector<int64_t>& dilation, int64_t groups);

Tensor conv_transpose1d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& output_padding, int64_t groups,
                            const std::vector<int64_t>& dilation);
Tensor conv_transpose2d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& output_padding, int64_t groups,
                            const std::vector<int64_t>& dilation);
Tensor conv_transpose3d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                            const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& output_padding, int64_t groups,
                            const std::vector<int64_t>& dilation);

#define TP_DECLARE_CONV_GRADS(rank)                                                     \
    Tensor conv##rank##d_grad_input_cpu(                                                \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);                          \
    Tensor conv##rank##d_grad_weight_cpu(                                               \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);                          \
    Tensor conv##rank##d_grad_bias_cpu(                                                 \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);

TP_DECLARE_CONV_GRADS(1)
TP_DECLARE_CONV_GRADS(2)
TP_DECLARE_CONV_GRADS(3)
#undef TP_DECLARE_CONV_GRADS

#define TP_DECLARE_CONV_TRANSPOSE_GRADS(rank)                                           \
    Tensor conv_transpose##rank##d_grad_input_cpu(                                      \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);                                          \
    Tensor conv_transpose##rank##d_grad_weight_cpu(                                     \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);                                          \
    Tensor conv_transpose##rank##d_grad_bias_cpu(                                       \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);

TP_DECLARE_CONV_TRANSPOSE_GRADS(1)
TP_DECLARE_CONV_TRANSPOSE_GRADS(2)
TP_DECLARE_CONV_TRANSPOSE_GRADS(3)
#undef TP_DECLARE_CONV_TRANSPOSE_GRADS

} // namespace cpu

namespace cuda {

Tensor conv1d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv3d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);

Tensor conv_transpose1d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& output_padding, int64_t groups,
                             const std::vector<int64_t>& dilation);
Tensor conv_transpose2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& output_padding, int64_t groups,
                             const std::vector<int64_t>& dilation);
Tensor conv_transpose3d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& output_padding, int64_t groups,
                             const std::vector<int64_t>& dilation);

#define TP_DECLARE_CONV_GRADS(rank)                                                     \
    Tensor conv##rank##d_grad_input_cuda(                                               \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);                          \
    Tensor conv##rank##d_grad_weight_cuda(                                              \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);                          \
    Tensor conv##rank##d_grad_bias_cuda(                                                \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& dilation, int64_t groups);

TP_DECLARE_CONV_GRADS(1)
TP_DECLARE_CONV_GRADS(2)
TP_DECLARE_CONV_GRADS(3)
#undef TP_DECLARE_CONV_GRADS

#define TP_DECLARE_CONV_TRANSPOSE_GRADS(rank)                                           \
    Tensor conv_transpose##rank##d_grad_input_cuda(                                     \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);                                          \
    Tensor conv_transpose##rank##d_grad_weight_cuda(                                    \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);                                          \
    Tensor conv_transpose##rank##d_grad_bias_cuda(                                      \
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,           \
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,        \
        const std::vector<int64_t>& output_padding, int64_t groups,                     \
        const std::vector<int64_t>& dilation);

TP_DECLARE_CONV_TRANSPOSE_GRADS(1)
TP_DECLARE_CONV_TRANSPOSE_GRADS(2)
TP_DECLARE_CONV_TRANSPOSE_GRADS(3)
#undef TP_DECLARE_CONV_TRANSPOSE_GRADS

} // namespace cuda

namespace convolution {

// Shared shape validation: returns the number of spatial dimensions (1, 2 or
// 3) that both operands agree on.
P10_API int64_t spatial_dims(const Tensor& input, const Tensor& weight, const char* name);

// Decodes the three-slot backward request mask, tolerating a short vector.
struct GradRequest {
    bool input;
    bool weight;
    bool bias;
};

inline GradRequest decode_mask(const std::vector<bool>& output_mask) {
    return GradRequest{output_mask.size() > 0 && output_mask[0],
                       output_mask.size() > 1 && output_mask[1],
                       output_mask.size() > 2 && output_mask[2]};
}

// The bias gradient is the reduction of grad_output over every axis but the
// channel one, so a recorded bias length is only a consistency check.
P10_API void check_bias_sizes(const std::optional<std::vector<int64_t>>& bias_sizes,
                              int64_t out_channels);

} // namespace convolution
} // namespace tensorplay
