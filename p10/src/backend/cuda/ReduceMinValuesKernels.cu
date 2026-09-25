#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {


Tensor amin_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amin()");
    return amin_dim_kernel(self, dim, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMinValuesKernels) {
    m.impl("amin", amin_cuda2);
}

} // namespace cuda
} // namespace tensorplay
