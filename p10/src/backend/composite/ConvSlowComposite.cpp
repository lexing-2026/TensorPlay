// Rank-specific "slow" convolution spellings.
//
// These entries are the explicit single-group, unit-dilation layer entry
// points; the math routes through the rank-generic convolution kernels that
// the dispatcher already owns, so every backend those kernels support is
// available and no per-operator math is duplicated.  The explicit kernel
// size arguments are validated against the weight shape as a consistency
// check.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_kernel_size(const std::vector<int64_t>& kernel_size,
                       const Tensor& weight, const char* name) {
    TP_CHECK(static_cast<int64_t>(kernel_size.size()) == weight.dim() - 2,
             name, ": kernel_size must have ", weight.dim() - 2,
             " elements");
    for (int64_t d = 0; d < weight.dim() - 2; ++d) {
        TP_CHECK(kernel_size[static_cast<size_t>(d)] == weight.size(d + 2),
                 name, ": kernel_size does not match the weight shape");
    }
}

std::vector<int64_t> fill_stride_or_padding(const std::vector<int64_t>& value,
                                            int64_t rank, const char* name) {
    if (value.size() == 1) {
        return std::vector<int64_t>(static_cast<size_t>(rank), value[0]);
    }
    TP_CHECK(static_cast<int64_t>(value.size()) == rank, name,
             ": expected ", rank, " values, got ", value.size());
    return value;
}

// Output mask for the three-slot backward: forward which slots were asked
// for; every call requests exactly one gradient through the generated
// per-slot formulas.
std::vector<bool> mask_with(bool i, bool w, bool b) {
    return {i, w, b};
}

}  // namespace

Tensor thnn_conv2d_native(const Tensor& self, const Tensor& weight,
                          const std::vector<int64_t>& kernel_size,
                          const std::optional<Tensor>& bias,
                          const std::vector<int64_t>& stride,
                          const std::vector<int64_t>& padding) {
    check_kernel_size(kernel_size, weight, "thnn_conv2d");
    return ops::conv2d(self, weight, bias,
                       fill_stride_or_padding(stride, 2, "thnn_conv2d"),
                       fill_stride_or_padding(padding, 2, "thnn_conv2d"),
                       {1, 1}, 1);
}

Tensor& thnn_conv2d_out_native(const Tensor& self, const Tensor& weight,
                               const std::vector<int64_t>& kernel_size,
                               const std::optional<Tensor>& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               Tensor& out) {
    Tensor result = thnn_conv2d_native(self, weight, kernel_size, bias,
                                       stride, padding);
    out.copy_(result);
    return out;
}

Tensor _slow_conv2d_forward_native(const Tensor& self, const Tensor& weight,
                                   const std::vector<int64_t>& kernel_size,
                                   const std::optional<Tensor>& bias,
                                   const std::vector<int64_t>& stride,
                                   const std::vector<int64_t>& padding) {
    return thnn_conv2d_native(self, weight, kernel_size, bias, stride, padding);
}

Tensor& _slow_conv2d_forward_output_native(
    const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    Tensor& output) {
    return thnn_conv2d_out_native(self, weight, kernel_size, bias, stride,
                                  padding, output);
}

std::tuple<Tensor, Tensor, Tensor> _slow_conv2d_backward_native(
    const Tensor& grad_output, const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<bool>& output_mask) {
    (void)kernel_size;
    TP_CHECK(output_mask.size() >= 3,
             "_slow_conv2d_backward: output_mask must have 3 elements");
    return ops::convolution_backward(
        grad_output, self, weight, std::nullopt,
        fill_stride_or_padding(stride, 2, "_slow_conv2d_backward"),
        fill_stride_or_padding(padding, 2, "_slow_conv2d_backward"),
        {1, 1}, false, {0, 0}, 1, output_mask);
}

std::tuple<Tensor, Tensor, Tensor> _slow_conv2d_backward_grad_input_native(
    const Tensor& grad_output, const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, Tensor& grad_input,
    Tensor& grad_weight, Tensor& grad_bias) {
    auto grads = _slow_conv2d_backward_native(
        grad_output, self, weight, kernel_size, stride, padding,
        mask_with(true, true, true));
    grad_input.copy_(std::get<0>(grads));
    grad_weight.copy_(std::get<1>(grads));
    grad_bias.copy_(std::get<2>(grads));
    return {grad_input, grad_weight, grad_bias};
}

Tensor slow_conv3d_native(const Tensor& self, const Tensor& weight,
                          const std::vector<int64_t>& kernel_size,
                          const std::optional<Tensor>& bias,
                          const std::vector<int64_t>& stride,
                          const std::vector<int64_t>& padding) {
    check_kernel_size(kernel_size, weight, "slow_conv3d");
    return ops::conv3d(self, weight, bias,
                       fill_stride_or_padding(stride, 3, "slow_conv3d"),
                       fill_stride_or_padding(padding, 3, "slow_conv3d"),
                       {1, 1, 1}, 1);
}

Tensor& slow_conv3d_out_native(const Tensor& self, const Tensor& weight,
                               const std::vector<int64_t>& kernel_size,
                               const std::optional<Tensor>& bias,
                               const std::vector<int64_t>& stride,
                               const std::vector<int64_t>& padding,
                               Tensor& out) {
    Tensor result = slow_conv3d_native(self, weight, kernel_size, bias, stride,
                                       padding);
    out.copy_(result);
    return out;
}

Tensor slow_conv3d_forward_native(const Tensor& self, const Tensor& weight,
                                  const std::vector<int64_t>& kernel_size,
                                  const std::optional<Tensor>& bias,
                                  const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding) {
    return slow_conv3d_native(self, weight, kernel_size, bias, stride, padding);
}

Tensor& slow_conv3d_forward_output_native(
    const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    Tensor& output) {
    return slow_conv3d_out_native(self, weight, kernel_size, bias, stride,
                                  padding, output);
}

Tensor slow_conv_transpose2d_native(
    const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation) {
    check_kernel_size(kernel_size, weight, "slow_conv_transpose2d");
    return ops::conv_transpose2d(
        self, weight, bias,
        fill_stride_or_padding(stride, 2, "slow_conv_transpose2d"),
        fill_stride_or_padding(padding, 2, "slow_conv_transpose2d"),
        fill_stride_or_padding(output_padding, 2, "slow_conv_transpose2d"), 1,
        fill_stride_or_padding(dilation, 2, "slow_conv_transpose2d"));
}

Tensor& slow_conv_transpose2d_out_native(
    const Tensor& self, const Tensor& weight,
    const std::vector<int64_t>& kernel_size, const std::optional<Tensor>& bias,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation, Tensor& out) {
    Tensor result = slow_conv_transpose2d_native(
        self, weight, kernel_size, bias, stride, padding, output_padding,
        dilation);
    out.copy_(result);
    return out;
}

}  // namespace composite
}  // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Composite, ConvSlowComposite) {
    using namespace tensorplay::composite;
    m.impl("thnn_conv2d", thnn_conv2d_native);
    m.impl("thnn_conv2d.out", thnn_conv2d_out_native);
    m.impl("_slow_conv2d_forward", _slow_conv2d_forward_native);
    m.impl("_slow_conv2d_forward.output", _slow_conv2d_forward_output_native);
    m.impl("_slow_conv2d_backward.output_mask", _slow_conv2d_backward_native);
    m.impl("_slow_conv2d_backward.grad_input",
           _slow_conv2d_backward_grad_input_native);
    m.impl("slow_conv3d", slow_conv3d_native);
    m.impl("slow_conv3d.out", slow_conv3d_out_native);
    m.impl("slow_conv3d_forward", slow_conv3d_forward_native);
    m.impl("slow_conv3d_forward.output", slow_conv3d_forward_output_native);
    m.impl("slow_conv_transpose2d", slow_conv_transpose2d_native);
    m.impl("slow_conv_transpose2d.out", slow_conv_transpose2d_out_native);
}
