#pragma once

// Sparse BLAS kernels that need per-ISA compilation (DEFAULT/AVX2/...).  The
// implementations live in SparseBlasKernelsImpl.cpp, listed in
// TP_CPU_KERNEL_SRCS; the entry points in SparseBlasKernels.cpp stay in the
// base tier and hand prepared, contiguous buffers to the stub.

#include "DispatchStub.h"
#include "Scalar.h"
#include "Tensor.h"

namespace tensorplay {
namespace cpu {

// In-place sampled SpMM over the stored entries of a sparse CSR result:
// result.values = alpha * (mat1 @ mat2)[pattern] + beta * result.values.
// `mat1` is contiguous (B, M, K); `mat2_t` is contiguous (B, N, K), the
// transpose of the user's right operand.  `result` is CSR with crow/col/
// values shaped (B, M+1)/(B, nnz)/(B, nnz).
using sampled_addmm_fn = void (*)(const Tensor& mat1, const Tensor& mat2_t,
                                  const Scalar& beta, const Scalar& alpha,
                                  const Tensor& result);
DECLARE_DISPATCH(sampled_addmm_fn, sampled_addmm_stub)

} // namespace cpu
} // namespace tensorplay
