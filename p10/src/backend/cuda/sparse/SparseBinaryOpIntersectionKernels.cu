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
__global__ void sparse_add_kernel(
    int64_t update_numel,
    const int64_t* indices,
    int64_t nnz,
    int64_t dense_numel,
    scalar_t* dense,
    const scalar_t* values,
    scalar_t alpha,
    SparseGatherInfo info) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= update_numel) return;
    const int64_t entry = linear / dense_numel;
    const int64_t inner = linear % dense_numel;

    int64_t destination_offset = 0;
    for (int d = 0; d < info.sparse_dim; ++d) {
        destination_offset += indices[d * nnz + entry] * info.strides[d];
    }
    int64_t remainder = inner;
    for (int d = info.dense_dim - 1; d >= 0; --d) {
        const int64_t dim_size = info.shape[info.sparse_dim + d];
        const int64_t coordinate = dim_size == 0 ? 0 : remainder % dim_size;
        remainder = dim_size == 0 ? 0 : remainder / dim_size;
        destination_offset += coordinate * info.strides[info.sparse_dim + d];
    }
    dense[destination_offset] += alpha * values[linear];
}


template <typename real_t>
__global__ void sparse_add_complex_kernel(
    int64_t update_numel,
    const int64_t* indices,
    int64_t nnz,
    int64_t dense_numel,
    CudaComplexPair<real_t>* dense,
    const CudaComplexPair<real_t>* values,
    CudaComplexPair<real_t> alpha,
    SparseGatherInfo info) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= update_numel) return;
    const int64_t entry = linear / dense_numel;
    const int64_t inner = linear % dense_numel;

    int64_t destination_offset = 0;
    for (int d = 0; d < info.sparse_dim; ++d) {
        destination_offset += indices[d * nnz + entry] * info.strides[d];
    }
    int64_t remainder = inner;
    for (int d = info.dense_dim - 1; d >= 0; --d) {
        const int64_t dim_size = info.shape[info.sparse_dim + d];
        const int64_t coordinate = dim_size == 0 ? 0 : remainder % dim_size;
        remainder = dim_size == 0 ? 0 : remainder / dim_size;
        destination_offset += coordinate * info.strides[info.sparse_dim + d];
    }

    using compute_t = std::conditional_t<std::is_same_v<real_t, double>, double, float>;
    const CudaComplexPair<real_t> value = values[linear];
    CudaComplexPair<real_t>& destination = dense[destination_offset];
    const compute_t value_real = static_cast<compute_t>(value.real);
    const compute_t value_imag = static_cast<compute_t>(value.imag);
    const compute_t alpha_real = static_cast<compute_t>(alpha.real);
    const compute_t alpha_imag = static_cast<compute_t>(alpha.imag);
    destination.real = static_cast<real_t>(
        static_cast<compute_t>(destination.real) +
        alpha_real * value_real - alpha_imag * value_imag);
    destination.imag = static_cast<real_t>(
        static_cast<compute_t>(destination.imag) +
        alpha_real * value_imag + alpha_imag * value_real);
}


__device__ bool coord_less(const int64_t* idx, int64_t nnz, int64_t j,
                           const int64_t* idx_a, int64_t nnz_a, int64_t i,
                           int64_t sparse_dim) {
    // Column-major storage: coordinate d of entry n lives at idx[d*nnz+n].
    for (int64_t d = 0; d < sparse_dim; ++d) {
        const int64_t a = idx[d * nnz + j];
        const int64_t b = idx_a[d * nnz_a + i];
        if (a < b) return true;
        if (a > b) return false;
    }
    return false;
}


__device__ bool coord_equal(const int64_t* idx, int64_t nnz, int64_t j,
                            const int64_t* idx_a, int64_t nnz_a, int64_t i,
                            int64_t sparse_dim) {
    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (idx[d * nnz + j] != idx_a[d * nnz_a + i]) return false;
    }
    return true;
}


// Sorted-merge intersection: each coalesced A entry binary-searches its
// coordinate in the sorted B array and, when matched, records the product.
template <typename scalar_t>
__global__ void coo_intersect_mul_kernel(
    int64_t nnz_a, int64_t nnz_b, int64_t sparse_dim,
    const int64_t* idx_a, const int64_t* idx_b,
    const scalar_t* val_a, const scalar_t* val_b,
    bool* flags, scalar_t* products) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz_a) return;
    int64_t lo = 0, hi = nnz_b;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (coord_less(idx_b, nnz_b, mid, idx_a, nnz_a, i, sparse_dim)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const bool found =
        lo < nnz_b && coord_equal(idx_b, nnz_b, lo, idx_a, nnz_a, i, sparse_dim);
    flags[i] = found;
    if (found) products[i] = val_a[i] * val_b[lo];
}

} // namespace


Tensor& add_sparse_to_dense_cuda(Tensor& dense, const Tensor& sparse, Scalar alpha) {
    if (dense.is_sparse() || !sparse.is_sparse()) {
        TP_THROW(RuntimeError, "add_: expected a dense self and sparse COO other");
    }
    if (dense.shape() != sparse.shape()) {
        TP_THROW(RuntimeError, "add_: sparse COO operands must have identical sizes");
    }
    Tensor canonical = sparse.is_coalesced() ? sparse : sparse.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values();
    if (values.dtype() != dense.dtype()) {
        values = Tensor::make_sparse_coo_tensor(
            indices, values.to(dense.dtype()),
            static_cast<std::vector<int64_t>>(sparse.shape()), true)._values();
    }
    values = values.contiguous();
    const int64_t nnz = indices.size(1);
    int64_t dense_numel = 1;
    for (int64_t d = canonical.sparse_dim(); d < canonical.dim(); ++d) {
        dense_numel *= canonical.size(d);
    }
    const int64_t update_numel = nnz * dense_numel;
    if (update_numel == 0) return dense;

    SparseGatherInfo info = make_gather_info(dense, canonical);
    const int threads = 256;
    const int blocks = static_cast<int>((update_numel + threads - 1) / threads);
    const cudaStream_t add_stream = getCurrentCUDAStream().stream();
    if (dense.dtype() == DType::ComplexHalf ||
        dense.dtype() == DType::ComplexFloat ||
        dense.dtype() == DType::ComplexDouble ||
        dense.dtype() == DType::BComplex32) {
        const auto alpha_value = alpha.to<tensorplay::complex<double>>();
        if (dense.dtype() == DType::ComplexHalf) {
            const CudaComplexPair<tensorplay::Half> alpha_pair{
                tensorplay::Half(static_cast<float>(alpha_value.real())),
                tensorplay::Half(static_cast<float>(alpha_value.imag()))};
            sparse_add_complex_kernel<tensorplay::Half><<<
                blocks, threads, 0, add_stream>>>(
                update_numel, indices.data_ptr<int64_t>(), nnz, dense_numel,
                reinterpret_cast<CudaComplexPair<tensorplay::Half>*>(
                    dense.data_ptr()),
                reinterpret_cast<const CudaComplexPair<tensorplay::Half>*>(
                    values.data_ptr()),
                alpha_pair, info);
        } else if (dense.dtype() == DType::ComplexFloat) {
            const CudaComplexPair<float> alpha_pair{
                static_cast<float>(alpha_value.real()),
                static_cast<float>(alpha_value.imag())};
            sparse_add_complex_kernel<float><<<blocks, threads, 0, add_stream>>>(
                update_numel, indices.data_ptr<int64_t>(), nnz, dense_numel,
                reinterpret_cast<CudaComplexPair<float>*>(dense.data_ptr()),
                reinterpret_cast<const CudaComplexPair<float>*>(values.data_ptr()),
                alpha_pair, info);
        } else if (dense.dtype() == DType::ComplexDouble) {
            const CudaComplexPair<double> alpha_pair{
                alpha_value.real(), alpha_value.imag()};
            sparse_add_complex_kernel<double><<<blocks, threads, 0, add_stream>>>(
                update_numel, indices.data_ptr<int64_t>(), nnz, dense_numel,
                reinterpret_cast<CudaComplexPair<double>*>(dense.data_ptr()),
                reinterpret_cast<const CudaComplexPair<double>*>(values.data_ptr()),
                alpha_pair, info);
        } else {
            const CudaComplexPair<tensorplay::BFloat16> alpha_pair{
                tensorplay::BFloat16(static_cast<float>(alpha_value.real())),
                tensorplay::BFloat16(static_cast<float>(alpha_value.imag()))};
            sparse_add_complex_kernel<tensorplay::BFloat16><<<
                blocks, threads, 0, add_stream>>>(
                update_numel, indices.data_ptr<int64_t>(), nnz, dense_numel,
                reinterpret_cast<CudaComplexPair<tensorplay::BFloat16>*>(
                    dense.data_ptr()),
                reinterpret_cast<const CudaComplexPair<tensorplay::BFloat16>*>(
                    values.data_ptr()),
                alpha_pair, info);
        }
        checkCuda(cudaGetLastError(), "CUDA sparse complex add kernel");
        dense.unsafeGetTensorImpl()->bump_version();
        return dense;
    }
#define TP_SPARSE_ADD_CASE(ctype, name) \
    case DType::name: \
        sparse_add_kernel<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            update_numel, indices.data_ptr<int64_t>(), nnz, dense_numel, \
            dense.data_ptr<ctype>(), values.data_ptr<ctype>(), alpha.to<ctype>(), info); \
        break;
    switch (dense.dtype()) {
        TP_SPARSE_ADD_CASE(uint8_t, UInt8)
        TP_SPARSE_ADD_CASE(int8_t, Int8)
        TP_SPARSE_ADD_CASE(int16_t, Int16)
        TP_SPARSE_ADD_CASE(int32_t, Int32)
        TP_SPARSE_ADD_CASE(int64_t, Int64)
        TP_SPARSE_ADD_CASE(uint16_t, UInt16)
        TP_SPARSE_ADD_CASE(uint32_t, UInt32)
        TP_SPARSE_ADD_CASE(uint64_t, UInt64)
        TP_SPARSE_ADD_CASE(float, Float32)
        TP_SPARSE_ADD_CASE(double, Float64)
        TP_SPARSE_ADD_CASE(tensorplay::Half, Float16)
        TP_SPARSE_ADD_CASE(tensorplay::BFloat16, BFloat16)
        TP_SPARSE_ADD_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "CUDA sparse add: unsupported dtype");
    }
#undef TP_SPARSE_ADD_CASE
    checkCuda(cudaGetLastError(), "CUDA sparse add kernel");
    dense.unsafeGetTensorImpl()->bump_version();
    return dense;
}


// Coordinate-union addition: concatenate both COO component sets on-device
// alpha=1).
Tensor sparse_add_cuda(const Tensor& self, const Tensor& other) {
    if (!self.is_sparse() || self.is_sparse_csr() ||
        !other.is_sparse() || other.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): expected two sparse COO tensors");
    }
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): operands must have identical sizes");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(TypeError, "sparse.add(): operands must share one dtype");
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse.add(): operands must share one device");
    }
    Tensor a = self.is_coalesced() ? self : self.coalesce();
    Tensor b = other.is_coalesced() ? other : other.coalesce();
    if (a.sparse_dim() != b.sparse_dim() || a._values().dim() != b._values().dim()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): sparse dimensions and value shapes must match");
    }
    for (int64_t d = 1; d < a._values().dim(); ++d) {
        TP_CHECK(a._values().size(d) == b._values().size(d),
                 "sparse.add(): sparse dimensions and value shapes must match");
    }
    Tensor cat_indices = Tensor::cat({a._indices(), b._indices()}, 1);
    Tensor cat_values = Tensor::cat({a._values(), b._values()}, 0);
    return Tensor::make_sparse_coo_tensor(
        cat_indices, cat_values,
        static_cast<std::vector<int64_t>>(a.shape()),
        /*is_coalesced=*/false).coalesce();
}


Tensor sparse_mul_cuda(const Tensor& self, const Tensor& other) {
    if (!self.is_sparse() || self.is_sparse_csr() ||
        !other.is_sparse() || other.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): expected two sparse COO tensors");
    }
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): operands must have identical sizes");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(TypeError, "sparse.mul(): operands must share one dtype");
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse.mul(): operands must share one device");
    }
    Tensor a = self.coalesce();
    Tensor b = other.coalesce();
    if (a._values().dim() != 1 || b._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): hybrid COO tensors are not supported");
    }
    Tensor ia = a._indices().contiguous();
    Tensor va = a._values().contiguous();
    Tensor ib = b._indices().contiguous();
    Tensor vb = b._values().contiguous();
    const int64_t nnz_a = va.size(0);
    const int64_t nnz_b = vb.size(0);
    const int64_t sparse_dim = ia.size(0);
    const cudaStream_t stream = getCurrentCUDAStream().stream();

    Tensor flags = Tensor::empty({nnz_a}, DType::Bool, self.device());
    Tensor products = Tensor::empty({nnz_a}, self.dtype(), self.device());
    dispatch_coalesce_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        coo_intersect_mul_kernel<scalar_t><<<coalesce_blocks(nnz_a),
                                             kCoalesceThreads, 0, stream>>>(
            nnz_a, nnz_b, sparse_dim, ia.data_ptr<int64_t>(),
            ib.data_ptr<int64_t>(), va.data_ptr<scalar_t>(),
            vb.data_ptr<scalar_t>(), flags.data_ptr<bool>(),
            products.data_ptr<scalar_t>());
    });
    checkCuda(cudaGetLastError(), "CUDA sparse_mul intersect kernel");

    // Compaction of matched entries through an exclusive sum over flags.
    Tensor flags_i64 = flags.to(DType::Int64);
    Tensor slots = Tensor::zeros({nnz_a}, DType::Int64, self.device());
    size_t scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  nullptr, scan_bytes, flags_i64.data_ptr<int64_t>(),
                  slots.data_ptr<int64_t>(), static_cast<int>(nnz_a), stream),
              "CUB sparse_mul exclusive-sum size query");
    Tensor scan_temporary = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  scan_temporary.data_ptr(), scan_bytes,
                  flags_i64.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
                  static_cast<int>(nnz_a), stream),
              "CUB sparse_mul exclusive sum");

    // Read back the matched count: slots[nnz_a-1] + flags[nnz_a-1].
    int64_t slot_tail = 0;
    int64_t flag_tail = 0;
    checkCuda(cudaMemcpyAsync(&slot_tail,
                              nnz_a > 0
                                  ? slots.data_ptr<int64_t>() + (nnz_a - 1)
                                  : slots.data_ptr<int64_t>(),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse_mul slot readback");
    checkCuda(cudaMemcpyAsync(&flag_tail,
                              nnz_a > 0
                                  ? flags_i64.data_ptr<int64_t>() + (nnz_a - 1)
                                  : flags_i64.data_ptr<int64_t>(),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse_mul tail-flag readback");
    checkCuda(cudaStreamSynchronize(stream), "CUDA sparse_mul nnz sync");
    const int64_t out_nnz =
        (nnz_a > 0 ? slot_tail + flag_tail : 0);

    Tensor out_indices = Tensor::empty({sparse_dim, out_nnz}, DType::Int64,
                                       self.device());
    Tensor out_values = Tensor::empty({out_nnz}, self.dtype(), self.device());
    if (out_nnz > 0) {
        // Compaction of matched entries through CUB Flagged: coordinates
        // per dimension, then the product values.  The count sink is a
        // throwaway device scalar (the real count was derived above).
        Tensor count_sink = Tensor::zeros({1}, DType::Int64, self.device());
        size_t select_bytes = 0;
        checkCuda(cub::DeviceSelect::Flagged(
                      nullptr, select_bytes, ia.data_ptr<int64_t>(),
                      flags.data_ptr<bool>(), out_indices.data_ptr<int64_t>(),
                      count_sink.data_ptr<int64_t>(),
                      static_cast<int>(nnz_a), stream),
                  "CUB sparse_mul select size query");
        Tensor select_temporary = Tensor::empty(
            {static_cast<int64_t>(select_bytes == 0 ? 1 : select_bytes)},
            DType::UInt8, self.device());
        for (int64_t d = 0; d < sparse_dim; ++d) {
            checkCuda(cub::DeviceSelect::Flagged(
                          select_temporary.data_ptr(), select_bytes,
                          ia.data_ptr<int64_t>() + d * nnz_a,
                          flags.data_ptr<bool>(),
                          out_indices.data_ptr<int64_t>() + d * out_nnz,
                          count_sink.data_ptr<int64_t>(),
                          static_cast<int>(nnz_a), stream),
                      "CUB sparse_mul coordinate select");
        }
        dispatch_coalesce_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            checkCuda(cub::DeviceSelect::Flagged(
                          select_temporary.data_ptr(), select_bytes,
                          products.data_ptr<scalar_t>(), flags.data_ptr<bool>(),
                          out_values.data_ptr<scalar_t>(),
                          count_sink.data_ptr<int64_t>(),
                          static_cast<int>(nnz_a), stream),
                      "CUB sparse_mul value select");
        });
        checkCuda(cudaGetLastError(), "CUDA sparse_mul compaction");
    }
    return Tensor::make_sparse_coo_tensor(
        out_indices, out_values,
        static_cast<std::vector<int64_t>>(a.shape()), true);
}

} // namespace cuda
} // namespace tensorplay
