#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {

namespace {

std::tuple<Tensor, Tensor> aminmax_cuda(const Tensor& self, const std::vector<int64_t>& dim,
                                        bool keepdim) {
    if (self.numel() == 0) {
        if (dim.empty()) {
            TP_THROW(RuntimeError, "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim, "aminmax");
    }
    return {amin_cuda2(self, dim, keepdim), amax_cuda2(self, dim, keepdim)};
}


std::tuple<Tensor, Tensor> aminmax_all_cuda(const Tensor& self) {
    return aminmax_cuda(self, {}, false);
}


std::tuple<Tensor, Tensor> aminmax_dim_cuda(const Tensor& self, int64_t dim,
                                            bool keepdim) {
    return aminmax_cuda(self, {dim}, keepdim);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceAMinMaxKernels) {
    m.impl("aminmax", aminmax_cuda);
    m.impl("_aminmax", aminmax_all_cuda);
    m.impl("_aminmax.dim", aminmax_dim_cuda);
}

} // namespace cuda
} // namespace tensorplay
