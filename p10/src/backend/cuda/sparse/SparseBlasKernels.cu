// Sparse BLAS on CUDA.  sparse_sampled_addmm evaluates the dense product
// mat1 @ mat2 only on the stored entries of a sparse CSR result:
//   result.values = alpha * (mat1 @ mat2)[pattern] + beta * result.values

#include "SparseKernels.cuh"
#include "SparseKernels.h"
#include "CUDARuntime.h"
#include "Complex.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace tensorplay {
namespace cuda {

namespace {

template <typename scalar_t>
__global__ void sampled_addmm_kernel(
    int64_t total,
    int64_t nnz_per_batch,
    int64_t K,
    int64_t M,
    int64_t N,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* mat1,
    const scalar_t* mat2_t,
    scalar_t alpha,
    scalar_t beta,
    scalar_t* values) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t b = e / nnz_per_batch;
    const int64_t le = e % nnz_per_batch;
    // Locate the row of stored element `le` inside batch b's crow slice.
    const int64_t* crow_b = crow + b * (M + 1);
    int64_t lo = 0;
    int64_t hi = M;
    while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (crow_b[mid + 1] <= le) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int64_t m = lo;
    const int64_t n = col[b * nnz_per_batch + le];
    const scalar_t* a = mat1 + b * M * K + m * K;
    const scalar_t* bvec = mat2_t + b * N * K + n * K;
    scalar_t dot = scalar_t(0);
    for (int64_t k = 0; k < K; ++k) {
        dot += a[k] * bvec[k];
    }
    values[b * nnz_per_batch + le] = alpha * dot + beta * values[b * nnz_per_batch + le];
}

template <typename real_t>
__global__ void sampled_addmm_complex_kernel(
    int64_t total,
    int64_t nnz_per_batch,
    int64_t K,
    int64_t M,
    int64_t N,
    const int64_t* crow,
    const int64_t* col,
    const CudaComplexPair<real_t>* mat1,
    const CudaComplexPair<real_t>* mat2_t,
    CudaComplexPair<real_t> alpha,
    CudaComplexPair<real_t> beta,
    CudaComplexPair<real_t>* values) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t b = e / nnz_per_batch;
    const int64_t le = e % nnz_per_batch;
    const int64_t* crow_b = crow + b * (M + 1);
    int64_t lo = 0;
    int64_t hi = M;
    while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (crow_b[mid + 1] <= le) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int64_t m = lo;
    const int64_t n = col[b * nnz_per_batch + le];
    using compute_t = std::conditional_t<std::is_same_v<real_t, double>, double, float>;
    const CudaComplexPair<real_t>* a = mat1 + b * M * K + m * K;
    const CudaComplexPair<real_t>* bvec = mat2_t + b * N * K + n * K;
    compute_t dot_real = 0;
    compute_t dot_imag = 0;
    for (int64_t k = 0; k < K; ++k) {
        const compute_t ar = static_cast<compute_t>(a[k].real);
        const compute_t ai = static_cast<compute_t>(a[k].imag);
        const compute_t br = static_cast<compute_t>(bvec[k].real);
        const compute_t bi = static_cast<compute_t>(bvec[k].imag);
        dot_real += ar * br - ai * bi;
        dot_imag += ar * bi + ai * br;
    }
    const compute_t alpha_r = static_cast<compute_t>(alpha.real);
    const compute_t alpha_i = static_cast<compute_t>(alpha.imag);
    const compute_t beta_r = static_cast<compute_t>(beta.real);
    const compute_t beta_i = static_cast<compute_t>(beta.imag);
    const int64_t slot = b * nnz_per_batch + le;
    const compute_t old_r = static_cast<compute_t>(values[slot].real);
    const compute_t old_i = static_cast<compute_t>(values[slot].imag);
    values[slot].real = static_cast<real_t>(
        alpha_r * dot_real - alpha_i * dot_imag + beta_r * old_r - beta_i * old_i);
    values[slot].imag = static_cast<real_t>(
        alpha_r * dot_imag + alpha_i * dot_real + beta_r * old_i + beta_i * old_r);
}

// Flatten batch axes above the 2-D matrix into a single CSR instance so the
// kernels above see one crow pointer per batch slice.  Returns the flattened
// crow/col/values and the per-batch offsets used to index them.
struct FlatCsr {
    Tensor crow;    // (B, M+1) flattened to (B*(M+1),)
    Tensor col;     // (B, nnz) flattened
    Tensor values;  // (B, nnz) flattened
    int64_t batch;
    int64_t M;
    int64_t nnz;
};

FlatCsr flatten_csr(const Tensor& csr) {
    const int64_t dim = csr.dim();
    const int64_t batch = dim == 2 ? 1 : csr.size(0);
    const int64_t M = csr.size(dim - 2);
    // Stored elements per batch slice: values are (*batch, nnz, ...).
    const int64_t nnz = csr._values().size(-1);
    Tensor crow = csr._crow_indices().reshape({-1});
    Tensor col = csr._col_indices().reshape({-1});
    if (crow.dtype() != DType::Int64) crow = crow.to(DType::Int64);
    if (col.dtype() != DType::Int64) col = col.to(DType::Int64);
    return FlatCsr{
        crow.contiguous(), col.contiguous(),
        csr._values().reshape({-1}).contiguous(),
        batch, M, nnz};
}

} // namespace

Tensor& sparse_sampled_addmm_out_sparse_csr_cuda(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha, Tensor& result) {
    if (!self.is_sparse_csr() || self.dense_dim() != 0 ||
        result.dense_dim() != 0) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected non-hybrid sparse CSR "
                 "self/result");
    }
    if (mat1.is_sparse() || mat2.is_sparse() ||
        !result.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: layout mismatch among operands");
    }
    if (mat1.dtype() != mat2.dtype() || mat1.dtype() != self.dtype() ||
        result.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_sampled_addmm: mat1, mat2, self and result must "
                 "share one dtype");
    }
    if (mat1.dim() < 2 || mat2.dim() < 2 || result.dim() < 2) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: mat1, mat2 and result must be "
                 "matrices");
    }
    if (mat1.size(mat1.dim() - 1) != mat2.size(mat2.dim() - 2) ||
        self.size(self.dim() - 2) != mat1.size(mat1.dim() - 2) ||
        self.size(self.dim() - 1) != mat2.size(mat2.dim() - 1)) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: operand shapes are incompatible");
    }
    if (self.device() != mat1.device() || self.device() != mat2.device() ||
        self.device() != result.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_sampled_addmm: all operands must be on the same "
                 "device");
    }
    const DType dtype = self.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::ComplexFloat && dtype != DType::ComplexDouble) {
        TP_THROW(RuntimeError,
                 "sparse_sampled_addmm: expected self to be a "
                 "floating-point or complex tensor, but got ",
                 toString(dtype));
    }

    if (&result != &self || !result._values().is_contiguous()) {
        Tensor crow = self._crow_indices().contiguous().clone();
        Tensor col = self._col_indices().contiguous().clone();
        Tensor values = self._values().contiguous().clone();
        result.unsafeGetTensorImpl()->set_sparse_compressed_state(
            crow.unsafeGetTensorImpl(), col.unsafeGetTensorImpl(),
            values.unsafeGetTensorImpl(),
            static_cast<std::vector<int64_t>>(self.shape()),
            TensorImpl::kSparseCSRLayout, {0, 0});
    }

    const int64_t nnz = self._nnz();
    if (mat1.numel() == 0 || mat2.numel() == 0 || nnz == 0) {
        result._values().mul_(beta);
        return result;
    }

    const int64_t K = mat1.size(mat1.dim() - 1);
    const int64_t N = mat2.size(mat2.dim() - 1);
    Tensor mat1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    Tensor mat2_t = mat2.transpose(-1, -2).contiguous();
    FlatCsr flat = flatten_csr(result);

    constexpr int threads = 128;
    const int blocks = static_cast<int>((flat.batch * flat.nnz + threads - 1) / threads);
    if (blocks > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        dispatch_coalesce_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            if constexpr (is_complex_type_v<scalar_t>) {
                using real_t = typename is_complex_type<scalar_t>::value_type;
                const auto beta_c = beta.to<complex<real_t>>();
                const auto alpha_c = alpha.to<complex<real_t>>();
                const CudaComplexPair<real_t> beta_pair{beta_c.real(),
                                                        beta_c.imag()};
                const CudaComplexPair<real_t> alpha_pair{alpha_c.real(),
                                                         alpha_c.imag()};
                sampled_addmm_complex_kernel<real_t>
                    <<<blocks, threads, 0, stream>>>(
                        flat.batch * flat.nnz, flat.nnz, K, flat.M, N,
                        flat.crow.data_ptr<int64_t>(),
                        flat.col.data_ptr<int64_t>(),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            mat1_c.data_ptr()),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            mat2_t.data_ptr()),
                        alpha_pair, beta_pair,
                        reinterpret_cast<CudaComplexPair<real_t>*>(
                            flat.values.data_ptr()));
            } else {
                sampled_addmm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    flat.batch * flat.nnz, flat.nnz, K, flat.M, N,
                    flat.crow.data_ptr<int64_t>(),
                    flat.col.data_ptr<int64_t>(),
                    mat1_c.data_ptr<scalar_t>(),
                    mat2_t.data_ptr<scalar_t>(),
                    alpha.to<scalar_t>(), beta.to<scalar_t>(),
                    flat.values.data_ptr<scalar_t>());
            }
        });
        checkCuda(cudaGetLastError(), "CUDA sparse_sampled_addmm kernel");
    }
    return result;
}

Tensor sparse_sampled_addmm_sparse_csr_cuda(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta, const Scalar& alpha) {
    Tensor result = Tensor::empty({0}, self.dtype(), self.device());
    sparse_sampled_addmm_out_sparse_csr_cuda(self, mat1, mat2, beta, alpha,
                                             result);
    return result;
}

} // namespace cuda
} // namespace tensorplay
