#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "Scalar.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {
// Helper for cuBLAS error checking
// (Assume checkCublasError is not exposed in header, so redefine or expose it. 
//  CUDAContext.cpp has it in anon namespace. Let's define a macro locally for now or make it public later.)
#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      TP_THROW(RuntimeError, "cuBLAS Error"); \
    } \
  } while (0)

Tensor mm_kernel_cuda(const Tensor& self, const Tensor& other) {
    // self: (M, K)
    // other: (K, N)
    // result: (M, N)
    
    if (self.dim() != 2 || other.dim() != 2) {
        TP_THROW(RuntimeError, "mm: tensors must be 2D");
    }
    if (self.shape()[1] != other.shape()[0]) {
        TP_THROW(RuntimeError, "mm: shape mismatch");
    }
    
    // Ensure inputs are contiguous for cuBLAS
    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
    Tensor other_contig = other.is_contiguous() ? other : other.contiguous();
    
    int64_t M = self_contig.shape()[0];
    int64_t K = self_contig.shape()[1];
    int64_t N = other_contig.shape()[1];
    
    Tensor result = Tensor::empty({M, N}, self.dtype(), self.device());
    
    cublasHandle_t handle = CUDAContext::getCublasHandle();

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
        // Enable TF32 for Tensor Cores on Ampere+
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    #endif
    
    // Set stream? For now default stream.
    // TODO: Get current stream from CUDAContext if we support streams.
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Row-Major trick: C = A * B
    // cublas computes C' = B' * A' (where ' denotes interpretation as col-major, which effectively transposes row-major data)
    // We pass B as first matrix (A in gemm), A as second matrix (B in gemm).
    // Dimensions:
    // B (passed as A): (K, N) -> interpreted as (N, K)
    // A (passed as B): (M, K) -> interpreted as (K, M)
    // Result C: (N, M) -> interpreted as (M, N) which is what we want in Row Major.
    
    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // We want C (MxN). 
    // In cuBLAS column major terms: C is (M rows, N cols).
    // But since we use the transpose trick:
    // We are computing C^T = B^T * A^T
    // C^T is (N, M). 
    // So we tell cublas m=N, n=M, k=K.
    
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32) {
        // Use cublasGemmEx to enable Tensor Cores (TF32 on Ampere+)
        // and allow algorithm heuristics.
        
        // Force TF32 compute type (requires CUDA 11+)
        cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
        // cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

        CUBLAS_CHECK(cublasGemmEx(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            N, M, K, 
            &alpha, 
            other_contig.data_ptr<float>(), CUDA_R_32F, N, 
            self_contig.data_ptr<float>(), CUDA_R_32F, K, 
            &beta, 
            result.data_ptr<float>(), CUDA_R_32F, N,
            computeType,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else if (self.dtype() == DType::Float64 && other.dtype() == DType::Float64) {
        double alpha_d = 1.0;
        double beta_d = 0.0;
        CUBLAS_CHECK(cublasDgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            N, M, K, 
            &alpha_d, 
            other_contig.data_ptr<double>(), N, 
            self_contig.data_ptr<double>(), K, 
            &beta_d, 
            result.data_ptr<double>(), N
        ));
    } else {
        TP_THROW(NotImplementedError, "mm: only float32/float64 supported on CUDA");
    }
    
    return result;
}

Tensor matmul_kernel_cuda(const Tensor& self, const Tensor& other) {
    // Basic matmul support (currently just aliases mm for 2D)
    // TODO: Support broadcasting and >2D tensors
    if (self.dim() == 2 && other.dim() == 2) {
        return mm_kernel_cuda(self, other);
    }
    TP_THROW(NotImplementedError, "matmul: only 2D supported on CUDA for now");
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LinearAlgebraKernels) {
    m.impl("mm", mm_kernel_cuda);
    m.impl("matmul", matmul_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
