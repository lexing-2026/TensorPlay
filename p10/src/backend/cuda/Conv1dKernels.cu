#include "Tensor.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor conv2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const std::vector<int64_t>& stride,
                              const std::vector<int64_t>& padding,
                              const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv2d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups);
Tensor conv_transpose2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& output_padding, int64_t groups,
                             const std::vector<int64_t>& dilation);
Tensor conv_transpose2d_grad_input_cuda(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& weight,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& padding,
                                        const std::vector<int64_t>& output_padding,
                                        int64_t groups,
                                        const std::vector<int64_t>& dilation);
Tensor conv_transpose2d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& weight,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& output_padding,
                                         int64_t groups,
                                         const std::vector<int64_t>& dilation);
Tensor conv_transpose2d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& weight,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& output_padding,
                                       int64_t groups,
                                       const std::vector<int64_t>& dilation);

Tensor conv1d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 3) TP_THROW(RuntimeError, "conv1d: Expected 3D input (N, C, L)");
    convolution::check_conv_shapes(input, weight, bias, groups, false);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv2d_cuda(in2, w2, bias, s2, p2, d2, groups).squeeze(2);
}

Tensor conv1d_grad_input_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                              const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv2d_grad_input_cuda(go2, in2, w2, s2, p2, d2, groups).squeeze(2);
}

Tensor conv1d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                               const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                               const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv2d_grad_weight_cuda(go2, in2, w2, s2, p2, d2, groups).squeeze(2);
}

Tensor conv1d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                             const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv2d_grad_bias_cuda(go2, in2, w2, s2, p2, d2, groups);
}

Tensor conv_transpose1d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias,
                             const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& output_padding, int64_t groups,
                             const std::vector<int64_t>& dilation) {
    convolution::check_conv_shapes(input, weight, bias, groups, true);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> op2 = {0, output_padding.empty() ? 0 : output_padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv_transpose2d_cuda(in2, w2, bias, s2, p2, op2, groups, d2).squeeze(2);
}

Tensor conv_transpose1d_grad_input_cuda(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& weight,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& padding,
                                        const std::vector<int64_t>& output_padding,
                                        int64_t groups,
                                        const std::vector<int64_t>& dilation) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> op2 = {0, output_padding.empty() ? 0 : output_padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv_transpose2d_grad_input_cuda(go2, in2, w2, s2, p2, op2, groups, d2).squeeze(2);
}

Tensor conv_transpose1d_grad_weight_cuda(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& weight,
                                         const std::vector<int64_t>& stride,
                                         const std::vector<int64_t>& padding,
                                         const std::vector<int64_t>& output_padding,
                                         int64_t groups,
                                         const std::vector<int64_t>& dilation) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> op2 = {0, output_padding.empty() ? 0 : output_padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv_transpose2d_grad_weight_cuda(go2, in2, w2, s2, p2, op2, groups, d2).squeeze(2);
}

Tensor conv_transpose1d_grad_bias_cuda(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& weight,
                                       const std::vector<int64_t>& stride,
                                       const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& output_padding,
                                       int64_t groups,
                                       const std::vector<int64_t>& dilation) {
    Tensor go2 = grad_output.unsqueeze(2);
    Tensor in2 = input.unsqueeze(2);
    Tensor w2 = weight.unsqueeze(2);
    std::vector<int64_t> s2 = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p2 = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> op2 = {0, output_padding.empty() ? 0 : output_padding[0]};
    std::vector<int64_t> d2 = {1, dilation.empty() ? 1 : dilation[0]};
    return conv_transpose2d_grad_bias_cuda(go2, in2, w2, s2, p2, op2, groups, d2);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Conv1dKernels) {
    m.impl("conv1d", conv1d_cuda);
    m.impl("conv1d_grad_input", conv1d_grad_input_cuda);
    m.impl("conv1d_grad_weight", conv1d_grad_weight_cuda);
    m.impl("conv1d_grad_bias", conv1d_grad_bias_cuda);
    m.impl("conv_transpose1d", conv_transpose1d_cuda);
    m.impl("conv_transpose1d_grad_input", conv_transpose1d_grad_input_cuda);
    m.impl("conv_transpose1d_grad_weight", conv_transpose1d_grad_weight_cuda);
    m.impl("conv_transpose1d_grad_bias", conv_transpose1d_grad_bias_cuda);
}

}
}
