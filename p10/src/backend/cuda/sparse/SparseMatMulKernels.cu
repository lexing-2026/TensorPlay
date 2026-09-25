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

} // namespace cuda
} // namespace tensorplay
