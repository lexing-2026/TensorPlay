#pragma once

#include "DType.h"
#include "Tensor.h"

namespace tensorplay {
namespace cuda {

// Throws NotImplementedError when `t` has no cuBLAS GEMM path (integers,
// bool), mirroring torch's "addmm_cuda" not implemented wording upstream.
void check_cublas_gemm_dtype(DType t);

// Row-major single GEMM: result(M,N) = alpha * self(M,K) @ other(K,N) + beta * result.
// `bias` (optional) enables the cuBLASLt bias epilogue and must be length-N.
void gemm_impl(const Tensor& self, const Tensor& other, const Tensor& result,
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

} // namespace cuda
} // namespace tensorplay
