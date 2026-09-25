#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop__adaptive_avg_pool3d_cuda(
        const Tensor& self, const std::vector<int64_t>& output_size) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool3d", self, output_size);
}


Tensor interop__adaptive_avg_pool3d_backward_cuda(const Tensor& grad_output,
                                                  const Tensor& input) {
    return dispatch_cuda<Tensor>("adaptive_avg_pool3d_backward", grad_output,
                                 input);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, AdaptiveAveragePooling3dKernels) {
    m.impl("_adaptive_avg_pool3d", interop__adaptive_avg_pool3d_cuda);
    m.impl("_adaptive_avg_pool3d_backward", interop__adaptive_avg_pool3d_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
