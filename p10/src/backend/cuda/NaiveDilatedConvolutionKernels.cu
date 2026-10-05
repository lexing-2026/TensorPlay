#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop_slow_conv_dilated2d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv2d(self, weight, bias, stride, padding, dilation, 1);
}


Tensor interop_slow_conv_dilated3d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv3d(self, weight, bias, stride, padding, dilation, 1);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NaiveDilatedConvolutionKernels) {
    m.impl("slow_conv_dilated2d", interop_slow_conv_dilated2d_cuda);
    m.impl("slow_conv_dilated3d", interop_slow_conv_dilated3d_cuda);
}

} // namespace cuda
} // namespace tensorplay
