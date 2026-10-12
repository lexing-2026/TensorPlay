#include "SparseKernels.cuh"
#include "SparseKernels.h"
#include "CUDARuntime.h"
#include "Complex.h"

#include "GPUPrimitives.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <climits>
#include <type_traits>
#include <utility>
#include "Atomic.cuh"
#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {

namespace {


template <typename scalar_t>
__global__ void sparse_coo_mm_kernel(
    int64_t total,
    int64_t cols,
    const int64_t* row_indices,
    const int64_t* col_indices,
    const scalar_t* values,
    const scalar_t* dense,
    scalar_t* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t n = linear / cols;
    const int64_t j = linear % cols;
    // Distinct coordinates may still share a row, so several threads can
    // target the same output cell; accumulate atomically.
    atomicAdd(&out[row_indices[n] * cols + j],
              values[n] * dense[col_indices[n] * cols + j]);
}


template <typename scalar_t>
__global__ void sparse_csr_mm_kernel(
    int64_t total,
    int64_t cols,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* dense,
    scalar_t* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t i = linear / cols;
    const int64_t j = linear % cols;
    scalar_t accumulator = scalar_t(0);
    for (int64_t t = crow[i]; t < crow[i + 1]; ++t) {
        accumulator += values[t] * dense[col[t] * cols + j];
    }
    out[linear] = accumulator;
}


template <typename real_t>
__global__ void sparse_csr_mm_complex_kernel(
    int64_t total,
    int64_t cols,
    const int64_t* crow,
    const int64_t* col,
    const CudaComplexPair<real_t>* values,
    const CudaComplexPair<real_t>* dense,
    CudaComplexPair<real_t>* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t row = linear / cols;
    const int64_t column = linear % cols;
    using compute_t = std::conditional_t<std::is_same_v<real_t, double>, double, float>;
    compute_t real = 0;
    compute_t imag = 0;
    for (int64_t index = crow[row]; index < crow[row + 1]; ++index) {
        const CudaComplexPair<real_t> left = values[index];
        const CudaComplexPair<real_t> right = dense[col[index] * cols + column];
        const compute_t left_real = static_cast<compute_t>(left.real);
        const compute_t left_imag = static_cast<compute_t>(left.imag);
        const compute_t right_real = static_cast<compute_t>(right.real);
        const compute_t right_imag = static_cast<compute_t>(right.imag);
        real += left_real * right_real - left_imag * right_imag;
        imag += left_real * right_imag + left_imag * right_real;
    }
    out[linear].real = static_cast<real_t>(real);
    out[linear].imag = static_cast<real_t>(imag);
}

} // namespace


Tensor sparse_mm_cuda(const Tensor& self, const Tensor& dense) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_mm(): expected a sparse COO/CSR first argument");
    }
    if (self.dim() != 2 || dense.dim() != 2) {
        TP_THROW(RuntimeError, "sparse_mm(): both operands must be 2-D");
    }
    if (dense.size(0) != self.size(1)) {
        TP_THROW(RuntimeError,
                 "sparse_mm(): operand shapes are incompatible for matmul");
    }
    if (dense.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm(): operands must share the sparse tensor's dtype");
    }
    if (self.device() != dense.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mm(): operands must be on the same device");
    }
    constexpr int threads = 128;
    Tensor source = self;
    if (self.is_sparse_csr()) {
        if (self.sparse_dim() != 2 || self._values().dim() != 1) {
            TP_THROW(RuntimeError,
                     "sparse_mm(): hybrid CSR tensors are not supported");
        }
    } else {
        if (self.sparse_dim() != 2) {
            TP_THROW(RuntimeError,
                     "sparse_mm(): hybrid COO tensors are not supported");
        }
        Tensor canonical = self.is_coalesced() ? self : self.coalesce();
        if (canonical._values().dim() != 1) {
            TP_THROW(RuntimeError,
                     "sparse_mm(): hybrid COO tensors are not supported");
        }
        source = coo_to_csr_native(canonical);
    }

    Tensor crow = source._crow_indices().contiguous();
    Tensor col = source._col_indices().contiguous();
    Tensor values = source._values().contiguous();
    Tensor dense_contiguous = dense.is_contiguous() ? dense : dense.contiguous();
    Tensor out = Tensor::zeros({self.size(0), dense.size(1)}, self.dtype(),
                               self.device());
    const int64_t cols = dense.size(1);
    const int64_t total = self.size(0) * cols;
    if (total > 0) {
        const cudaStream_t mm_stream = getCurrentCUDAStream().stream();
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        dispatch_coalesce_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            if constexpr (is_complex_type_v<scalar_t>) {
                using real_t = typename is_complex_type<scalar_t>::value_type;
                sparse_csr_mm_complex_kernel<real_t><<<blocks, threads, 0,
                                                       mm_stream>>>(
                    total, cols, crow.data_ptr<int64_t>(),
                    col.data_ptr<int64_t>(),
                    reinterpret_cast<const CudaComplexPair<real_t>*>(
                        values.data_ptr()),
                    reinterpret_cast<const CudaComplexPair<real_t>*>(
                        dense_contiguous.data_ptr()),
                    reinterpret_cast<CudaComplexPair<real_t>*>(out.data_ptr()));
            } else {
                sparse_csr_mm_kernel<scalar_t><<<blocks, threads, 0, mm_stream>>>(
                    total, cols, crow.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
                    values.data_ptr<scalar_t>(),
                    dense_contiguous.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>());
            }
        });
        checkCuda(cudaGetLastError(), "CUDA sparse_mm kernel");
    }
    return out;
}

namespace {

// Fused dense-seeded SpMM: one thread computes one output cell as
// beta * self[i,j] + alpha * sum_t values[t] * dense[col[t], j].
template <typename scalar_t>
__global__ void sparse_csr_addmm_kernel(
    int64_t total,
    int64_t cols,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* dense,
    scalar_t beta,
    scalar_t alpha,
    const scalar_t* self_data,
    int64_t self_stride0,
    int64_t self_stride1,
    scalar_t* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t i = linear / cols;
    const int64_t j = linear % cols;
    scalar_t acc = beta * self_data[i * self_stride0 + j * self_stride1];
    for (int64_t t = crow[i]; t < crow[i + 1]; ++t) {
        acc += alpha * values[t] * dense[col[t] * cols + j];
    }
    out[linear] = acc;
}

template <typename real_t>
__global__ void sparse_csr_addmm_complex_kernel(
    int64_t total,
    int64_t cols,
    const int64_t* crow,
    const int64_t* col,
    const CudaComplexPair<real_t>* values,
    const CudaComplexPair<real_t>* dense,
    CudaComplexPair<real_t> beta,
    CudaComplexPair<real_t> alpha,
    const CudaComplexPair<real_t>* self_data,
    int64_t self_stride0,
    int64_t self_stride1,
    CudaComplexPair<real_t>* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t i = linear / cols;
    const int64_t j = linear % cols;
    using compute_t = std::conditional_t<std::is_same_v<real_t, double>, double, float>;
    const CudaComplexPair<real_t> seed = self_data[i * self_stride0 + j * self_stride1];
    compute_t acc_real = static_cast<compute_t>(beta.real) * static_cast<compute_t>(seed.real) -
                         static_cast<compute_t>(beta.imag) * static_cast<compute_t>(seed.imag);
    compute_t acc_imag = static_cast<compute_t>(beta.real) * static_cast<compute_t>(seed.imag) +
                         static_cast<compute_t>(beta.imag) * static_cast<compute_t>(seed.real);
    for (int64_t t = crow[i]; t < crow[i + 1]; ++t) {
        const CudaComplexPair<real_t> left = values[t];
        const CudaComplexPair<real_t> right = dense[col[t] * cols + j];
        const compute_t left_real = static_cast<compute_t>(left.real);
        const compute_t left_imag = static_cast<compute_t>(left.imag);
        const compute_t right_real = static_cast<compute_t>(right.real);
        const compute_t right_imag = static_cast<compute_t>(right.imag);
        const compute_t prod_real = left_real * right_real - left_imag * right_imag;
        const compute_t prod_imag = left_real * right_imag + left_imag * right_real;
        acc_real += static_cast<compute_t>(alpha.real) * prod_real -
                    static_cast<compute_t>(alpha.imag) * prod_imag;
        acc_imag += static_cast<compute_t>(alpha.real) * prod_imag +
                    static_cast<compute_t>(alpha.imag) * prod_real;
    }
    out[linear].real = static_cast<real_t>(acc_real);
    out[linear].imag = static_cast<real_t>(acc_imag);
}

// addmm-style broadcast of the dense seed onto (rows, cols): trailing dims
// align, singleton axes get stride 0.
Tensor broadcast_addmm_seed_cuda(const Tensor& input, int64_t rows,
                                 int64_t cols) {
    if (input.dim() == 2 && input.size(0) == rows && input.size(1) == cols) {
        return input;
    }
    if (input.dim() > 2) {
        TP_THROW(RuntimeError, "_sparse_addmm: self must be at most 2-D, got ",
                 input.dim(), "-D");
    }
    std::vector<int64_t> sizes = {rows, cols};
    std::vector<int64_t> strides = {0, 0};
    if (input.dim() == 2) {
        if (input.size(0) != 1 && input.size(0) != rows) {
            TP_THROW(RuntimeError,
                     "_sparse_addmm: self is not broadcastable to ",
                     rows, "x", cols);
        }
        if (input.size(1) != 1 && input.size(1) != cols) {
            TP_THROW(RuntimeError,
                     "_sparse_addmm: self is not broadcastable to ",
                     rows, "x", cols);
        }
        strides[0] = input.size(0) == 1 ? 0 : input.stride(0);
        strides[1] = input.size(1) == 1 ? 0 : input.stride(1);
    } else if (input.dim() == 1) {
        if (input.size(0) != 1 && input.size(0) != cols) {
            TP_THROW(RuntimeError,
                     "_sparse_addmm: self is not broadcastable to ",
                     rows, "x", cols);
        }
        strides[1] = input.size(0) == 1 ? 0 : input.stride(0);
    }
    return input.as_strided(sizes, strides);
}

} // namespace

Tensor _sparse_addmm_cuda(const Tensor& self, const Tensor& mat1,
                       const Tensor& mat2, const Scalar& beta,
                       const Scalar& alpha) {
    if (!mat1.is_sparse()) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): expected 'mat1' to be a sparse tensor");
    }
    if (mat1.dim() != 2) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): matrices expected, got ", mat1.dim(),
                 "-D tensor");
    }
    if (mat1.sparse_dim() != 2 || mat1._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): hybrid sparse tensors are not supported");
    }
    if (mat2.dim() != 2) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): matrices expected, got ", mat2.dim(),
                 "-D tensor");
    }
    if (mat1.device() != mat2.device() || mat1.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "_sparse_addmm(): all operands must be on the same device");
    }
    if (mat2.dtype() != mat1.dtype() || self.dtype() != mat1.dtype()) {
        TP_THROW(TypeError,
                 "_sparse_addmm(): all operands must share one dtype");
    }
    const int64_t rows = mat1.size(0);
    const int64_t inner = mat1.size(1);
    const int64_t cols = mat2.size(1);
    if (mat2.size(0) != inner) {
        TP_THROW(RuntimeError,
                 "_sparse_addmm(): mat1 and mat2 shapes cannot be multiplied (",
                 rows, "x", inner, " and ", mat2.size(0), "x", cols, ")");
    }

    Tensor self_b = broadcast_addmm_seed_cuda(self, rows, cols);
    Tensor source = mat1;
    if (!mat1.is_sparse_csr()) {
        Tensor canonical = mat1.is_coalesced() ? mat1 : mat1.coalesce();
        source = coo_to_csr_native(canonical);
    }
    Tensor crow = source._crow_indices().contiguous();
    Tensor col = source._col_indices().contiguous();
    Tensor values = source._values().contiguous();
    Tensor dense_contiguous = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    Tensor out = Tensor::empty({rows, cols}, mat1.dtype(), mat1.device());

    const int64_t total = rows * cols;
    if (total > 0) {
        constexpr int threads = 128;
        const cudaStream_t mm_stream = getCurrentCUDAStream().stream();
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        dispatch_coalesce_dtype(mat1.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            if constexpr (is_complex_type_v<scalar_t>) {
                using real_t = typename is_complex_type<scalar_t>::value_type;
                const auto beta_c = beta.to<complex<real_t>>();
                const auto alpha_c = alpha.to<complex<real_t>>();
                const auto beta_pair = CudaComplexPair<real_t>{
                    beta_c.real(), beta_c.imag()};
                const auto alpha_pair = CudaComplexPair<real_t>{
                    alpha_c.real(), alpha_c.imag()};
                sparse_csr_addmm_complex_kernel<real_t>
                    <<<blocks, threads, 0, mm_stream>>>(
                        total, cols, crow.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            values.data_ptr()),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            dense_contiguous.data_ptr()),
                        beta_pair, alpha_pair,
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            self_b.data_ptr()),
                        self_b.stride(0), self_b.stride(1),
                        reinterpret_cast<CudaComplexPair<real_t>*>(
                            out.data_ptr()));
            } else {
                sparse_csr_addmm_kernel<scalar_t><<<blocks, threads, 0,
                                                     mm_stream>>>(
                    total, cols, crow.data_ptr<int64_t>(),
                    col.data_ptr<int64_t>(), values.data_ptr<scalar_t>(),
                    dense_contiguous.data_ptr<scalar_t>(),
                    beta.to<scalar_t>(), alpha.to<scalar_t>(),
                    self_b.data_ptr<scalar_t>(), self_b.stride(0),
                    self_b.stride(1), out.data_ptr<scalar_t>());
            }
        });
        checkCuda(cudaGetLastError(), "CUDA _sparse_addmm kernel");
    }
    return out;
}

namespace {


// ---------------------------------------------------------------------------
// sspaddmm: sparse-only in the reference; tp has no sparse CUDA backend, so
// the contract is an explicit rejection rather than a dense reinterpretation.
// ---------------------------------------------------------------------------

Tensor interop_sspaddmm_cuda(const Tensor& /*self*/, const Tensor& mat1,
                             const Tensor& /*mat2*/, const Scalar& /*beta*/,
                             const Scalar& /*alpha*/) {
    TP_THROW(NotImplementedError,
             "sspaddmm requires a sparse CUDA backend; mat1 is dense with dtype ",
             toString(mat1.dtype()));
}


Tensor& interop_sspaddmm_out_cuda(const Tensor& self, const Tensor& mat1,
                                  const Tensor& mat2, const Scalar& beta, const Scalar& alpha,
                                  Tensor& out) {
    write_out(out, interop_sspaddmm_cuda(self, mat1, mat2, beta, alpha));
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SparseMatMulInterop) {
    m.impl("sspaddmm", interop_sspaddmm_cuda);
    m.impl("sspaddmm.out", interop_sspaddmm_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
