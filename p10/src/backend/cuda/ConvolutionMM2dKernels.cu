#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop__slow_conv2d_forward_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    return ops::conv2d(self, weight, bias, stride, padding, dilation, 1);
}


Tensor& interop__slow_conv2d_forward_output_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, Tensor& output) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    write_out(output, ops::conv2d(self, weight, bias, stride, padding,
                                  dilation, 1));
    return output;
}


std::tuple<Tensor, Tensor, Tensor> interop__slow_conv2d_backward_grad_input_cuda(
        const Tensor& grad_output, const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding, Tensor& grad_input,
        Tensor& grad_weight, Tensor& grad_bias) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    write_out(grad_input, ops::conv2d_grad_input(grad_output, self, weight, stride, padding, dilation, 1));
    write_out(grad_weight, ops::conv2d_grad_weight(grad_output, self, weight, stride, padding, dilation, 1));
    write_out(grad_bias, ops::conv2d_grad_bias(grad_output, self, weight, stride, padding, dilation, 1));
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}


std::tuple<Tensor, Tensor, Tensor>
interop__slow_conv2d_backward_output_mask_cuda(
        const Tensor& grad_output, const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<bool>& output_mask) {
    (void)kernel_size;
    const std::vector<int64_t> dilation{1, 1};
    const bool want_i = output_mask.size() > 0 && output_mask[0];
    const bool want_w = output_mask.size() > 1 && output_mask[1];
    const bool want_b = output_mask.size() > 2 && output_mask[2];
    Tensor gi = want_i ? ops::conv2d_grad_input(grad_output, self, weight, stride, padding, dilation, 1)
                       : Tensor();
    Tensor gw = want_w ? ops::conv2d_grad_weight(grad_output, self, weight, stride, padding, dilation, 1)
                       : Tensor();
    Tensor gb = want_b ? ops::conv2d_grad_bias(grad_output, self, weight, stride, padding, dilation, 1)
                       : Tensor();
    return std::make_tuple(gi, gw, gb);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, ConvolutionMM2dKernels) {
    m.impl("_slow_conv2d_forward", interop__slow_conv2d_forward_cuda);
    m.impl("_slow_conv2d_forward.output", interop__slow_conv2d_forward_output_cuda);
    m.impl("_slow_conv2d_backward.grad_input", interop__slow_conv2d_backward_grad_input_cuda);
    m.impl("_slow_conv2d_backward.output_mask", interop__slow_conv2d_backward_output_mask_cuda);
}

} // namespace cuda
} // namespace tensorplay
