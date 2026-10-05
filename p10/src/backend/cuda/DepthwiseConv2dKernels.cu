#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// Depthwise / slow convolution spellings.  tp's conv2d/conv3d kernels take
// groups natively; a depthwise convolution has one group per input channel,
// each widened to weight.size(0) / channels outputs.
// ---------------------------------------------------------------------------

Tensor interop__conv_depthwise2d_cuda(const Tensor& self, const Tensor& weight,
                                      const std::vector<int64_t>& kernel_size,
                                      const std::optional<Tensor>& bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv2d(self, weight, bias, stride, padding, dilation,
                       self.size(-3));
}


Tensor& interop__conv_depthwise2d_out_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    write_out(out, ops::conv2d(self, weight, bias, stride, padding, dilation,
                               self.size(-3)));
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, DepthwiseConv2dKernels) {
    // depthwise / slow convolutions
    m.impl("_conv_depthwise2d", interop__conv_depthwise2d_cuda);
    m.impl("_conv_depthwise2d.out", interop__conv_depthwise2d_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
