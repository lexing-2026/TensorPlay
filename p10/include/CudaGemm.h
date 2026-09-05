#pragma once

#include "DType.h"
#include "Tensor.h"

namespace tensorplay {
namespace cuda {

// Throws NotImplementedError when `t` has no cuBLAS GEMM path (integers,
void check_cublas_gemm_dtype(DType t);

// Row-major single GEMM: result(M,N) = alpha * self(M,K) @ other(K,N) + beta * result.
// `bias` (optional) enables the cuBLASLt bias epilogue and must be length-N.
void gemm_impl(const Tensor& self, const Tensor& other, Tensor& result,
               double alpha, double beta, const Tensor* bias);

// One strided-batched GEMM over (batch_size, M, K) x (batch_size, K, N)
// operand stacks into result (batch_size, M, N).  `stride_a`/`stride_b` are
// the per-batch strides of self/other (0 reuses one matrix across the batch).
void gemm_strided_batched_3d(const Tensor& self_3d, const Tensor& other_3d,
                             Tensor& result_3d, int64_t batch_size,
                             int64_t M, int64_t N, int64_t K,
                             long long stride_a, long long stride_b,
                             double alpha, double beta);

// Zero-fill used for empty-K GEMM outputs.
Tensor& zero_matmul_output_cuda(Tensor& output);

// Half-precision GEMV fast path for the memory-bound shapes: N == 1
// (matrix @ vector) or a small activation batch (M <= 8) against a large
// output axis.  `other_transposed` reports that `other` is a live (K, N)
// view of contiguous (N, K) row-major storage (the x @ W.t() pattern), whose
// rows the kernel walks directly.  Returns false when the shape falls
// outside the custom kernels' envelope and the caller must use the classic
// GEMM entry.
bool try_half_gemv(const Tensor& self, const Tensor& other, Tensor& result,
                   double alpha, double beta, bool other_transposed);

} // namespace cuda
} // namespace tensorplay
