#include "SparseKernels.h"
#include "Dispatcher.h"

namespace tensorplay {
namespace cuda {

// Every sparse op this backend implements is registered here; the bodies are
// split across the family units next door (factories / tensor math / csr math /
// binary-op intersection / matmul, plus spdiags in SpdiagsKernels.cu).  Keeping
// the wiring in one file means the op-to-unit map can be read at a glance.
//
// The two keys carry different subsets on purpose: CUDA is the fallback the
// dense dispatcher reaches for, SparseCUDA is the backend-tagged key.

TENSORPLAY_LIBRARY_IMPL(CUDA, SparseConversionOps) {
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cuda);
    m.impl("coalesce", coalesce_sparse_cuda);
    m.impl("_coalesce", coalesce_sparse_cuda);
    m.impl("sparse_mask", sparse_mask_cuda);
    m.impl("to_dense", to_dense_sparse_cuda);
    m.impl("to_sparse", to_sparse_coo_cuda);
    m.impl("to_sparse_csr", to_sparse_csr_cuda);
    m.impl("_nnz", sparse_nnz_cuda);
    m.impl("sparse_mm", sparse_mm_cuda);
    m.impl("sparse_sum", sparse_sum_cuda);
    m.impl("sparse_add", sparse_add_cuda);
    m.impl("sparse_mul", sparse_mul_cuda);
}

TENSORPLAY_LIBRARY_IMPL(SparseCUDA, CopySparseKernels) {
    m.impl("coalesce", coalesce_sparse_cuda);
    m.impl("_coalesce", coalesce_sparse_cuda);
    m.impl("sparse_mask", sparse_mask_cuda);
    m.impl("to_dense", to_dense_sparse_cuda);
    m.impl("to_sparse", to_sparse_coo_cuda);
    m.impl("to_sparse_csr", to_sparse_csr_cuda);
    m.impl("_nnz", sparse_nnz_cuda);
    m.impl("sparse_mm", sparse_mm_cuda);
    m.impl("sparse_sum", sparse_sum_cuda);
    m.impl("sparse_add", sparse_add_cuda);
    m.impl("sparse_mul", sparse_mul_cuda);
}

} // namespace cuda
} // namespace tensorplay
