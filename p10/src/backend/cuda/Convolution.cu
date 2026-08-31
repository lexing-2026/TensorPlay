#include "Convolution.h"

#include "Dispatcher.h"
#include "Exception.h"

#include <string>

namespace tensorplay {

namespace cuda {

Tensor convolution_cuda(const Tensor& input, const Tensor& weight,
                       const std::optional<Tensor>& bias,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation,
                       bool transposed,
                       const std::vector<int64_t>& output_padding,
                       int64_t groups) {
    const Tensor bias_t = bias.has_value() ? *bias : Tensor();
    const int64_t k = convolution::spatial_dims(input, weight, "convolution");
    if (!transposed) {
        switch (k) {
            case 1: return conv1d_cuda(input, weight, bias_t, stride, padding, dilation, groups);
            case 2: return conv2d_cuda(input, weight, bias_t, stride, padding, dilation, groups);
            default: return conv3d_cuda(input, weight, bias_t, stride, padding, dilation, groups);
        }
    }
    switch (k) {
        case 1: return conv_transpose1d_cuda(input, weight, bias_t, stride, padding,
                                            output_padding, groups, dilation);
        case 2: return conv_transpose2d_cuda(input, weight, bias_t, stride, padding,
                                            output_padding, groups, dilation);
        default: return conv_transpose3d_cuda(input, weight, bias_t, stride, padding,
                                             output_padding, groups, dilation);
    }
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        std::optional<std::vector<int64_t>> bias_sizes,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool transposed,
        const std::vector<int64_t>& output_padding, int64_t groups,
        const std::vector<bool>& output_mask) {
    const int64_t k = convolution::spatial_dims(input, weight, "convolution_backward");
    const convolution::GradRequest want = convolution::decode_mask(output_mask);
    if (want.bias) convolution::check_bias_sizes(bias_sizes, grad_output.size(1));

    // Slots the caller did not ask for stay undefined; that is the signal the
    // autograd engine reads for "no gradient flows to this input".
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;

#define TP_CONV_BACKWARD_CASE(fn_prefix, ...)                                              \
    do {                                                                                   \
        if (want.input) grad_input = fn_prefix##_grad_input_cuda(__VA_ARGS__);               \
        if (want.weight) grad_weight = fn_prefix##_grad_weight_cuda(__VA_ARGS__);            \
        if (want.bias) grad_bias = fn_prefix##_grad_bias_cuda(__VA_ARGS__);                  \
    } while (0)

    if (!transposed) {
        switch (k) {
            case 1: TP_CONV_BACKWARD_CASE(conv1d, grad_output, input, weight, stride,
                                          padding, dilation, groups); break;
            case 2: TP_CONV_BACKWARD_CASE(conv2d, grad_output, input, weight, stride,
                                          padding, dilation, groups); break;
            default: TP_CONV_BACKWARD_CASE(conv3d, grad_output, input, weight, stride,
                                           padding, dilation, groups); break;
        }
    } else {
        switch (k) {
            case 1: TP_CONV_BACKWARD_CASE(conv_transpose1d, grad_output, input, weight,
                                          stride, padding, output_padding, groups,
                                          dilation); break;
            case 2: TP_CONV_BACKWARD_CASE(conv_transpose2d, grad_output, input, weight,
                                          stride, padding, output_padding, groups,
                                          dilation); break;
            default: TP_CONV_BACKWARD_CASE(conv_transpose3d, grad_output, input, weight,
                                           stride, padding, output_padding, groups,
                                           dilation); break;
        }
    }
#undef TP_CONV_BACKWARD_CASE

    return {grad_input, grad_weight, grad_bias};
}

// Extension point for backends registered outside this tree.  CPU and CUDA
// answer it with the same routing as `convolution` so the two spellings never
// disagree on a device this build already supports.
Tensor convolution_overrideable_cuda(const Tensor& input, const Tensor& weight,
                                    const std::optional<Tensor>& bias,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation,
                                    bool transposed,
                                    const std::vector<int64_t>& output_padding,
                                    int64_t groups) {
    return convolution_cuda(input, weight, bias, stride, padding, dilation,
                           transposed, output_padding, groups);
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable_cuda(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, bool transposed,
        const std::vector<int64_t>& output_padding, int64_t groups,
        const std::vector<bool>& output_mask) {
    return convolution_backward_cuda(grad_output, input, weight, std::nullopt, stride,
                                    padding, dilation, transposed, output_padding,
                                    groups, output_mask);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Convolution) {
    m.impl("convolution", convolution_cuda);
    m.impl("convolution_backward", convolution_backward_cuda);
    m.impl("convolution_overrideable", convolution_overrideable_cuda);
    m.impl("convolution_backward_overrideable", convolution_backward_overrideable_cuda);
}

} // namespace cuda
} // namespace tensorplay
