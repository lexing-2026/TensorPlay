// CUDA SpMM-reduction kernels for _sparse_mm_reduce_impl (sum/mean/amax/amin)
// and its backward. The forward assigns one thread per output cell;
// the backward scatters with gpuAtomicAdd.

#include "SparseKernels.cuh"
#include "SparseKernels.h"
#include "CUDARuntime.h"
#include "Atomic.cuh"
#include "Complex.h"

#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cuda {

namespace {

template <typename F>
void dispatch_reduce_backward_dtype(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Float32: f(CoalesceTypeTag<float>{}); return;
        case DType::Float64: f(CoalesceTypeTag<double>{}); return;
        case DType::Float16: f(CoalesceTypeTag<Half>{}); return;
        case DType::BFloat16: f(CoalesceTypeTag<BFloat16>{}); return;
        default: TP_THROW(TypeError, "sparse reduction backward requires floating values");
    }
}

template <typename scalar_t>
__global__ void spmm_reduce_sum_mean_kernel(
    int64_t total,
    int64_t K,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* other,
    scalar_t* out,
    int reduce) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t m = linear / K;
    const int64_t k = linear % K;
    const int64_t row_start = crow[m];
    const int64_t row_end = crow[m + 1];
    scalar_t acc = scalar_t(0);
    for (int64_t e = row_start; e < row_end; ++e) {
        acc += values[e] * other[col[e] * K + k];
    }
    if (reduce == 1 && row_end != row_start) {
        acc /= static_cast<scalar_t>(row_end - row_start);
    }
    out[linear] = acc;
}

template <typename real_t>
__global__ void spmm_reduce_sum_mean_complex_kernel(
    int64_t total,
    int64_t K,
    const int64_t* crow,
    const int64_t* col,
    const CudaComplexPair<real_t>* values,
    const CudaComplexPair<real_t>* other,
    CudaComplexPair<real_t>* out,
    int reduce) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t m = linear / K;
    const int64_t k = linear % K;
    using compute_t = std::conditional_t<std::is_same_v<real_t, double>, double, float>;
    compute_t acc_real = 0;
    compute_t acc_imag = 0;
    const int64_t row_start = crow[m];
    const int64_t row_end = crow[m + 1];
    for (int64_t e = row_start; e < row_end; ++e) {
        const compute_t vr = static_cast<compute_t>(values[e].real);
        const compute_t vi = static_cast<compute_t>(values[e].imag);
        const compute_t or_ = static_cast<compute_t>(other[col[e] * K + k].real);
        const compute_t oi = static_cast<compute_t>(other[col[e] * K + k].imag);
        acc_real += vr * or_ - vi * oi;
        acc_imag += vr * oi + vi * or_;
    }
    if (reduce == 1 && row_end != row_start) {
        const compute_t inv = static_cast<compute_t>(1) / static_cast<compute_t>(row_end - row_start);
        acc_real *= inv;
        acc_imag *= inv;
    }
    out[linear].real = static_cast<real_t>(acc_real);
    out[linear].imag = static_cast<real_t>(acc_imag);
}

template <typename scalar_t>
__global__ void spmm_reduce_max_min_kernel(
    int64_t total,
    int64_t K,
    int64_t nnz,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* other,
    scalar_t* out,
    int64_t* arg_out,
    int reduce) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t m = linear / K;
    const int64_t k = linear % K;
    const int64_t row_start = crow[m];
    const int64_t row_end = crow[m + 1];
    const bool is_max = reduce == 2;
    scalar_t best = is_max
        ? -std::numeric_limits<scalar_t>::infinity()
        : std::numeric_limits<scalar_t>::infinity();
    int64_t best_e = nnz;
    for (int64_t e = row_start; e < row_end; ++e) {
        const scalar_t prod = values[e] * other[col[e] * K + k];
        if (is_max) {
            if (prod > best || !(prod == prod)) { best = prod; best_e = e; }
        } else {
            if (prod < best || !(prod == prod)) { best = prod; best_e = e; }
        }
    }
    out[linear] = best;
    arg_out[linear] = best_e;
}

// Backward for sum/mean: one thread per stored element computes the dot
// product grad_out[row[e]] . other[col[e]] (scaled for mean).
template <typename scalar_t>
__global__ void spmm_reduce_backward_input_kernel(
    int64_t nnz,
    int64_t K,
    const int64_t* crow,
    const int64_t* col,
    const scalar_t* other,
    const scalar_t* grad_out,
    scalar_t* grad_values,
    const int64_t* row,
    int reduce) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const int64_t r = row[e];
    const int64_t c = col[e];
    const scalar_t* other_row = other + c * K;
    const scalar_t* grad_row = grad_out + r * K;
    scalar_t acc = scalar_t(0);
    for (int64_t k = 0; k < K; ++k) {
        acc += other_row[k] * grad_row[k];
    }
    if (reduce == 1) {
        const int64_t count = crow[r + 1] - crow[r];
        acc /= static_cast<scalar_t>(count);
    }
    grad_values[e] = acc;
}

// Backward for amax/amin: scatter grad_out-weighted other rows into the
// winning stored elements.
template <typename scalar_t>
__global__ void spmm_reduce_backward_input_arg_kernel(
    int64_t total,
    int64_t K,
    int64_t nnz,
    const int64_t* col,
    const scalar_t* other,
    const scalar_t* grad_out,
    const int64_t* arg_out,
    scalar_t* grad_values) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t m = linear / K;
    const int64_t k = linear % K;
    const int64_t arg = arg_out[m * K + k];
    if (arg == nnz) return;
    const int64_t c = col[arg];
    gpuAtomicAdd(&grad_values[arg], other[c * K + k] * grad_out[m * K + k]);
}

// Backward for sum/mean grad_other: scatter each stored element's outer
// product values[e] * grad_out[row[e], :] into grad_other[col[e], :].
template <typename scalar_t>
__global__ void spmm_reduce_backward_other_kernel(
    int64_t nnz,
    int64_t K,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* grad_out,
    const int64_t* row,
    scalar_t* grad_other) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const int64_t r = row[e];
    const int64_t c = col[e];
    const scalar_t v = values[e];
    const scalar_t* grad_row = grad_out + r * K;
    scalar_t* dst = grad_other + c * K;
    for (int64_t k = 0; k < K; ++k) {
        gpuAtomicAdd(&dst[k], v * grad_row[k]);
    }
}

// Backward for amax/amin grad_other: scatter values[arg] * grad_out[m, :].
template <typename scalar_t>
__global__ void spmm_reduce_backward_other_arg_kernel(
    int64_t total,
    int64_t K,
    int64_t nnz,
    const int64_t* col,
    const scalar_t* values,
    const scalar_t* grad_out,
    const int64_t* arg_out,
    scalar_t* grad_other) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t m = linear / K;
    const int64_t k = linear % K;
    const int64_t arg = arg_out[m * K + k];
    if (arg == nnz) return;
    const int64_t c = col[arg];
    gpuAtomicAdd(&grad_other[c * K + k], values[arg] * grad_out[m * K + k]);
}

} // namespace

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_sparse_csr_cuda(
    const Tensor& self, const Tensor& other, const std::string& reduce) {
    // The fused evaluation works on CSR; other 2-D compressed layouts are
    // expanded to CSR first (blocked layouts unrolled to scalar entries).
    Tensor self_csr = self;
    if (!self.is_sparse_csr() && self.is_sparse_compressed() &&
        self.dim() == 2) {
        self_csr = compressed_to_csr_cuda(self);
    }
    if (!self_csr.is_sparse_csr() || self_csr.dense_dim() != 0 ||
        self_csr.dim() != 2) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: expected a non-hybrid 2-D sparse CSR "
                 "self tensor");
    }
    if (other.is_sparse() || other.dtype() != self_csr.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm_reduce: other must be a strided tensor sharing "
                 "self's dtype");
    }
    if (other.dim() != 2 || other.size(0) != self_csr.size(1)) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce: other shape mismatch");
    }
    if (self_csr.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mm_reduce: self and other must be on the same "
                 "device");
    }
    int op = -1;
    if (reduce == "sum") op = 0;
    else if (reduce == "mean") op = 1;
    else if (reduce == "amax") op = 2;
    else if (reduce == "amin") op = 3;
    else {
        TP_THROW(RuntimeError, "sparse_mm_reduce: unsupported reduce type \"",
                 reduce, "\"");
    }

    Tensor crow = self_csr._crow_indices().contiguous();
    Tensor col = self_csr._col_indices().contiguous();
    Tensor val = self_csr._values().contiguous();
    if (crow.dtype() != DType::Int64) crow = crow.to(DType::Int64);
    if (col.dtype() != DType::Int64) col = col.to(DType::Int64);
    const int64_t M = self_csr.size(0);
    const int64_t K = other.size(1);
    Tensor out = Tensor::zeros({M, K}, other.dtype(), other.device());
    Tensor arg_out = Tensor::empty({0}, col.dtype(), col.device());

    const int64_t nnz = self_csr._nnz();
    if (nnz == 0 || M == 0 || K == 0) {
        return std::make_tuple(std::move(out), std::move(arg_out));
    }

    const bool need_arg = GradMode::is_enabled() &&
                          (self.requires_grad() || other.requires_grad()) &&
                          (op == 2 || op == 3);
    const int64_t total = M * K;
    constexpr int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const cudaStream_t stream = getCurrentCUDAStream().stream();

    if (op == 2 || op == 3) {
        arg_out.resize_({M, K});
        arg_out.fill_(nnz);
        Tensor arg_i64 = arg_out.to(DType::Int64);
        dispatch_coalesce_dtype(self_csr.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            if constexpr (is_complex_type_v<scalar_t>) {
                // Complex amax/amin is not supported; treat as unsupported.
                TP_THROW(NotImplementedError,
                         "sparse_mm_reduce: amax/amin with complex dtype is "
                         "not supported on CUDA");
            } else {
                spmm_reduce_max_min_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        total, K, nnz, crow.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                        other.const_data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(),
                        arg_i64.data_ptr<int64_t>(), op);
            }
        });
        checkCuda(cudaGetLastError(), "CUDA spmm_reduce max/min kernel");
        if (arg_out.dtype() != DType::Int64) {
            arg_out.copy_(arg_i64);
        }
    } else {
        dispatch_coalesce_dtype(self_csr.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            if constexpr (is_complex_type_v<scalar_t>) {
                using real_t = typename is_complex_type<scalar_t>::value_type;
                spmm_reduce_sum_mean_complex_kernel<real_t>
                    <<<blocks, threads, 0, stream>>>(
                        total, K, crow.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            val.data_ptr()),
                        reinterpret_cast<const CudaComplexPair<real_t>*>(
                            other.const_data_ptr()),
                        reinterpret_cast<CudaComplexPair<real_t>*>(
                            out.data_ptr()),
                        op);
            } else {
                spmm_reduce_sum_mean_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        total, K, crow.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(), val.data_ptr<scalar_t>(),
                        other.const_data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(), op);
            }
        });
        checkCuda(cudaGetLastError(), "CUDA spmm_reduce sum/mean kernel");
    }
    return std::make_tuple(std::move(out), std::move(arg_out));
}

std::tuple<Tensor, Tensor> _sparse_mm_reduce_impl_backward_sparse_csr_cuda(
    const Tensor& self, const Tensor& grad_out, const Tensor& other,
    const std::string& reduce, const Tensor& arg_out,
    const std::vector<bool>& output_mask) {
    if (!self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse_mm_reduce_backward: expected a sparse CSR self");
    }
    if (grad_out.is_sparse() ||
        grad_out.dtype() != self.dtype() ||
        other.is_sparse() || other.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm_reduce_backward: dtype/layout mismatch");
    }
    int op = -1;
    if (reduce == "sum") op = 0;
    else if (reduce == "mean") op = 1;
    else if (reduce == "amax") op = 2;
    else if (reduce == "amin") op = 3;
    else {
        TP_THROW(RuntimeError, "sparse_mm_reduce_backward: unsupported reduce");
    }
    // The mask names the backward's own two inputs (self, other).  A longer
    // mask is the full forward-input mask of a multi-input op (e.g. sparse
    // addmm), whose matmul operands sit at positions 1 and 2.
    const std::array<bool, 2> mask = {
        output_mask.size() > 2 ? output_mask[1] : output_mask[0],
        output_mask.size() > 2 ? output_mask[2] : output_mask[1]};

    Tensor crow = self._crow_indices().contiguous();
    Tensor col = self._col_indices().contiguous();
    Tensor val = self._values().contiguous();
    if (crow.dtype() != DType::Int64) crow = crow.to(DType::Int64);
    if (col.dtype() != DType::Int64) col = col.to(DType::Int64);
    Tensor grad_out_c = grad_out.is_contiguous() ? grad_out : grad_out.contiguous();
    Tensor other_c = other.is_contiguous() ? other : other.contiguous();
    const int64_t nnz = self._nnz();
    const int64_t M = self.size(0);
    const int64_t K = other.size(1);
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    constexpr int threads = 128;

    Tensor grad_self;
    if (mask[0]) {
        grad_self = Tensor::make_sparse_csr_tensor(
            self._crow_indices().clone(), self._col_indices().clone(),
            Tensor::zeros_like(self._values()),
            static_cast<std::vector<int64_t>>(self.shape()));
        if (op == 2 || op == 3) {
            Tensor arg_i64 = arg_out.to(DType::Int64);
            const int64_t total = M * K;
            const int blocks = static_cast<int>((total + threads - 1) / threads);
            dispatch_reduce_backward_dtype(self.dtype(), [&](auto tag) {
                using scalar_t = typename decltype(tag)::type;
                spmm_reduce_backward_input_arg_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        total, K, nnz, col.data_ptr<int64_t>(),
                        other_c.const_data_ptr<scalar_t>(),
                        grad_out_c.const_data_ptr<scalar_t>(),
                        arg_i64.data_ptr<int64_t>(),
                        grad_self._values().data_ptr<scalar_t>());
            });
            checkCuda(cudaGetLastError(),
                      "CUDA spmm_reduce backward input arg kernel");
        } else {
            // row indices of each stored element.
            Tensor row = Tensor::_convert_indices_from_csr_to_coo(
                crow, col, /*out_int32=*/false, /*transpose=*/false)
                             .select(0, 0)
                             .contiguous();
            const int blocks = static_cast<int>((nnz + threads - 1) / threads);
            dispatch_reduce_backward_dtype(self.dtype(), [&](auto tag) {
                using scalar_t = typename decltype(tag)::type;
                spmm_reduce_backward_input_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        nnz, K, crow.data_ptr<int64_t>(),
                        col.data_ptr<int64_t>(),
                        other_c.const_data_ptr<scalar_t>(),
                        grad_out_c.const_data_ptr<scalar_t>(),
                        grad_self._values().data_ptr<scalar_t>(),
                        row.data_ptr<int64_t>(), op);
            });
            checkCuda(cudaGetLastError(),
                      "CUDA spmm_reduce backward input kernel");
        }
    }

    Tensor grad_other;
    if (mask[1]) {
        grad_other = Tensor::zeros({other.size(0), other.size(1)},
                                   other.dtype(), other.device());
        if (op == 2 || op == 3) {
            Tensor arg_i64 = arg_out.to(DType::Int64);
            const int64_t total = M * K;
            const int blocks = static_cast<int>((total + threads - 1) / threads);
            dispatch_reduce_backward_dtype(self.dtype(), [&](auto tag) {
                using scalar_t = typename decltype(tag)::type;
                spmm_reduce_backward_other_arg_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        total, K, nnz, col.data_ptr<int64_t>(),
                        val.data_ptr<scalar_t>(),
                        grad_out_c.const_data_ptr<scalar_t>(),
                        arg_i64.data_ptr<int64_t>(),
                        grad_other.data_ptr<scalar_t>());
            });
            checkCuda(cudaGetLastError(),
                      "CUDA spmm_reduce backward other arg kernel");
        } else {
            Tensor row = Tensor::_convert_indices_from_csr_to_coo(
                crow, col, /*out_int32=*/false, /*transpose=*/false)
                             .select(0, 0)
                             .contiguous();
            const int blocks = static_cast<int>((nnz + threads - 1) / threads);
            dispatch_reduce_backward_dtype(self.dtype(), [&](auto tag) {
                using scalar_t = typename decltype(tag)::type;
                spmm_reduce_backward_other_kernel<scalar_t>
                    <<<blocks, threads, 0, stream>>>(
                        nnz, K, col.data_ptr<int64_t>(),
                        val.data_ptr<scalar_t>(),
                        grad_out_c.const_data_ptr<scalar_t>(),
                        row.data_ptr<int64_t>(),
                        grad_other.data_ptr<scalar_t>());
            });
            checkCuda(cudaGetLastError(),
                      "CUDA spmm_reduce backward other kernel");
        }
    }
    return std::make_tuple(std::move(grad_self), std::move(grad_other));
}

} // namespace cuda
} // namespace tensorplay
