#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {



Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    return nansum_dim_kernel(self, dim_in, keepdim, DType::Undefined);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceSumProdKernels) {
    m.impl("nansum", nansum_cuda2);
}

} // namespace cuda
} // namespace tensorplay
