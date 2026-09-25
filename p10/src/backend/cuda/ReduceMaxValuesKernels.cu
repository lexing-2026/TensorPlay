#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {



Tensor amax_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amax()");
    return amax_dim_kernel(self, dim, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMaxValuesKernels) {
    m.impl("amax", amax_cuda2);
}

} // namespace cuda
} // namespace tensorplay
