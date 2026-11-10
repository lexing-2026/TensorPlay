// Ragged grouped GEMM, MACA lane.
//
// The kernels in GroupedGemmKernels.cu ride on the tensor-core grouped
// schedule, which this toolchain does not ship: the file cannot compile here.
// Its single shared entry point stays referenced by the attention reduction,
// so this translation unit exports it and reports every configuration as
// outside its envelope -- the caller then keeps the per-group math-library
// path instead of the persistent grouped kernel.
//
// The signature must match the declaration in CudaGemm.h.

#include "CudaGemm.h"

namespace tensorplay {
namespace cuda {

bool try_grouped_gemm_tensor_op(const Tensor& self, const Tensor& mat2,
                                const Tensor& offs, Tensor& out,
                                const int64_t* ends_host,
                                int64_t max_group_rows) {
    (void)self;
    (void)mat2;
    (void)offs;
    (void)out;
    (void)ends_host;
    (void)max_group_rows;
    return false;
}

}  // namespace cuda
}  // namespace tensorplay
