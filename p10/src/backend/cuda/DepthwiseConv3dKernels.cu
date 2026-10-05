#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop_conv_depthwise3d_cuda(
        const Tensor& self, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv3d(self, weight, bias, stride, padding, dilation,
                       self.size(-4));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, DepthwiseConv3dKernels) {
    m.impl("conv_depthwise3d", interop_conv_depthwise3d_cuda);
}

} // namespace cuda
} // namespace tensorplay
