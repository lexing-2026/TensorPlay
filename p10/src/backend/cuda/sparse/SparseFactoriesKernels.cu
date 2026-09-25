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


// Per-dimension max of the coordinate rows (size inference for
// sparse_coo_tensor with size=None).
__global__ void coord_max_kernel(int64_t nnz, int64_t sparse_dim,
                                 const int64_t* coords, int64_t* maxima) {
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= nnz) return;
    for (int64_t d = 0; d < sparse_dim; ++d) {
        // CUDA provides no atomicMax(long*) overload on this arch set; the
        // ULL intrinsic is bit-identical for two's-complement int64 max.
        atomicMax(reinterpret_cast<unsigned long long*>(maxima + d),
                  static_cast<unsigned long long>(coords[d * nnz + n]));
    }
}


// ---- dense -> sparse (native nonzero extraction) ---------------------------
//
// Zero detection at the byte level: every numeric encoding in use (IEEE
// floats incl. +/-0, two's-complement ints, bool, complex pairs) is all-zero
// bytes exactly when the value equals zero, and NaN carries nonzero bytes

__global__ void nonzero_mask_bytes_kernel(int64_t n, int64_t elem_size,
                                          const unsigned char* data,
                                          bool* mask) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* p = data + i * elem_size;
    unsigned char acc = 0;
    for (int64_t b = 0; b < elem_size; ++b) acc |= p[b];
    mask[i] = acc != 0;
}


// Gathers selected flat positions into COO components (row-major coords).
__global__ void coo_from_positions_kernel(int64_t nnz, int64_t ncols,
                                          int64_t elem_size,
                                          const int64_t* positions,
                                          const unsigned char* dense_data,
                                          int64_t* rows, int64_t* cols,
                                          unsigned char* values) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    const int64_t p = positions[i];
    rows[i] = p / ncols;
    cols[i] = p % ncols;
    const unsigned char* src = dense_data + p * elem_size;
    unsigned char* dst = values + i * elem_size;
    for (int64_t b = 0; b < elem_size; ++b) dst[b] = src[b];
}


__global__ void sparse_block_mask_bytes_kernel(
    int64_t blocks, int64_t block_bytes, const unsigned char* data, bool* mask) {
    const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block >= blocks) return;
    if (block_bytes == 0) {
        mask[block] = false;
        return;
    }
    const unsigned char* source = data + block * block_bytes;
    unsigned char accumulator = 0;
    for (int64_t byte = 0; byte < block_bytes; ++byte) {
        accumulator |= source[byte];
    }
    mask[block] = accumulator != 0;
}


__global__ void sparse_blocks_from_positions_kernel(
    int64_t nnz, int64_t sparse_dim, int64_t block_bytes,
    SparseBlockLayoutInfo layout, const int64_t* positions,
    const unsigned char* dense_data, int64_t* indices,
    unsigned char* values) {
    const int64_t output =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output >= nnz) return;

    const int64_t block = positions[output];
    int64_t remainder = block;
    for (int64_t d = sparse_dim - 1; d >= 0; --d) {
        const int64_t dim_size = layout.shape[d];
        indices[d * nnz + output] = remainder % dim_size;
        remainder /= dim_size;
    }

    const unsigned char* source = dense_data + block * block_bytes;
    unsigned char* destination = values + output * block_bytes;
    for (int64_t byte = 0; byte < block_bytes; ++byte) {
        destination[byte] = source[byte];
    }
}

} // namespace


Tensor to_sparse_coo_native(const Tensor& self) {
    Tensor contiguous_self = self.contiguous();
    const int64_t total = contiguous_self.numel();
    const int64_t ncols = contiguous_self.size(-1);
    Tensor flat = contiguous_self.reshape({total});
    const cudaStream_t stream = getCurrentCUDAStream().stream();

    Tensor mask = Tensor::empty({total}, DType::Bool, self.device());
    if (total > 0) {
        nonzero_mask_bytes_kernel<<<coalesce_blocks(total), kCoalesceThreads,
                                    0, stream>>>(
            total, static_cast<int64_t>(flat.itemsize()),
            reinterpret_cast<const unsigned char*>(flat.data_ptr()),
            mask.data_ptr<bool>());
        checkCuda(cudaGetLastError(), "CUDA to_sparse mask kernel");
    }

    thrust::counting_iterator<int64_t> counting(0);
    Tensor positions = Tensor::empty({total}, DType::Int64, self.device());
    Tensor count_dev = Tensor::zeros({1}, DType::Int64, self.device());
    size_t temporary_bytes = 0;
    checkCuda(cub::DeviceSelect::Flagged(
                  nullptr, temporary_bytes, counting, mask.data_ptr<bool>(),
                  positions.data_ptr<int64_t>(), count_dev.data_ptr<int64_t>(),
                  static_cast<int>(total), stream),
              "CUB to_sparse select size query");
    Tensor temporary = Tensor::empty(
        {static_cast<int64_t>(temporary_bytes == 0 ? 1 : temporary_bytes)},
        DType::UInt8, self.device());
    checkCuda(cub::DeviceSelect::Flagged(
                  temporary.data_ptr(), temporary_bytes, counting,
                  mask.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                  count_dev.data_ptr<int64_t>(), static_cast<int>(total),
                  stream),
              "CUB to_sparse select");

    int64_t nnz = 0;
    checkCuda(cudaMemcpyAsync(&nnz, count_dev.data_ptr<int64_t>(),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA to_sparse nnz readback");
    checkCuda(cudaStreamSynchronize(stream), "CUDA to_sparse nnz sync");

    Tensor indices = Tensor::empty({2, nnz}, DType::Int64, self.device());
    Tensor values = Tensor::empty(
        {nnz}, contiguous_self.dtype(), self.device());
    if (nnz > 0) {
        coo_from_positions_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                    stream>>>(
            nnz, ncols, static_cast<int64_t>(flat.itemsize()),
            positions.data_ptr<int64_t>(),
            reinterpret_cast<const unsigned char*>(flat.data_ptr()),
            indices.data_ptr<int64_t>(),
            indices.data_ptr<int64_t>() + nnz,
            reinterpret_cast<unsigned char*>(values.data_ptr()));
        checkCuda(cudaGetLastError(), "CUDA to_sparse gather kernel");
    }
    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(self.shape()), true);
}


Tensor to_sparse_coo_native_sparse_dim(const Tensor& self, int64_t sparse_dim) {
    const int64_t ndim = self.dim();
    if (sparse_dim < 0 || sparse_dim > ndim) {
        TP_THROW(ValueError,
                 "to_sparse(): sparse_dim must be in [0," +
                     std::to_string(ndim) + "]");
    }
    if (ndim > 0 && sparse_dim == 0) {
        TP_THROW(ValueError,
                 "to_sparse(): sparse_dim must be greater than zero for a non-scalar tensor");
    }
    if (ndim > kMaxSparseDims) {
        TP_THROW(RuntimeError, "to_sparse(): tensor rank exceeds CUDA sparse limit");
    }

    Tensor contiguous_self = self.contiguous();
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(contiguous_self.shape());
    const int64_t outer_numel = product_of(
        std::vector<int64_t>(sizes.begin(), sizes.begin() + sparse_dim));
    const int64_t block_numel = product_of(
        std::vector<int64_t>(sizes.begin() + sparse_dim, sizes.end()));
    const int64_t nnz_capacity = outer_numel;
    const int64_t block_bytes =
        block_numel * static_cast<int64_t>(contiguous_self.itemsize());
    SparseBlockLayoutInfo layout{};
    for (int64_t d = 0; d < ndim; ++d) {
        layout.shape[d] = sizes[static_cast<size_t>(d)];
    }
    std::vector<int64_t> values_shape{0};
    values_shape.insert(values_shape.end(), sizes.begin() + sparse_dim, sizes.end());

    if (outer_numel == 0) {
        Tensor indices = Tensor::empty({sparse_dim, 0}, DType::Int64, self.device());
        Tensor values = Tensor::empty(values_shape, contiguous_self.dtype(), self.device());
        return Tensor::make_sparse_coo_tensor(indices, values, sizes, true);
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    Tensor mask = Tensor::empty({outer_numel}, DType::Bool, self.device());
    sparse_block_mask_bytes_kernel<<<coalesce_blocks(outer_numel), kCoalesceThreads,
                                    0, stream>>>(
        outer_numel, block_bytes,
        reinterpret_cast<const unsigned char*>(contiguous_self.data_ptr()),
        mask.data_ptr<bool>());
    checkCuda(cudaGetLastError(), "CUDA sparse block mask kernel");

    thrust::counting_iterator<int64_t> counting(0);
    Tensor positions = Tensor::empty({nnz_capacity}, DType::Int64, self.device());
    Tensor count_dev = Tensor::zeros({1}, DType::Int64, self.device());
    size_t temporary_bytes = 0;
    checkCuda(cub::DeviceSelect::Flagged(
                  nullptr, temporary_bytes, counting, mask.data_ptr<bool>(),
                  positions.data_ptr<int64_t>(), count_dev.data_ptr<int64_t>(),
                  static_cast<int>(outer_numel), stream),
              "CUB sparse block select size query");
    Tensor temporary = Tensor::empty(
        {static_cast<int64_t>(temporary_bytes == 0 ? 1 : temporary_bytes)},
        DType::UInt8, self.device());
    checkCuda(cub::DeviceSelect::Flagged(
                  temporary.data_ptr(), temporary_bytes, counting,
                  mask.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                  count_dev.data_ptr<int64_t>(), static_cast<int>(outer_numel),
                  stream),
              "CUB sparse block select");

    int64_t nnz = 0;
    checkCuda(cudaMemcpyAsync(&nnz, count_dev.data_ptr<int64_t>(),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse block nnz readback");
    checkCuda(cudaStreamSynchronize(stream), "CUDA sparse block nnz sync");

    Tensor indices = Tensor::empty({sparse_dim, nnz}, DType::Int64, self.device());
    values_shape[0] = nnz;
    Tensor values = Tensor::empty(values_shape, contiguous_self.dtype(), self.device());
    if (nnz > 0) {
        sparse_blocks_from_positions_kernel<<<coalesce_blocks(nnz), kCoalesceThreads,
                                              0, stream>>>(
            nnz, sparse_dim, block_bytes, layout,
            positions.data_ptr<int64_t>(),
            reinterpret_cast<const unsigned char*>(contiguous_self.data_ptr()),
            indices.data_ptr<int64_t>(),
            reinterpret_cast<unsigned char*>(values.data_ptr()));
        checkCuda(cudaGetLastError(), "CUDA sparse block gather kernel");
    }
    return Tensor::make_sparse_coo_tensor(indices, values, sizes, true);
}


Tensor sparse_coo_tensor_cuda(const Tensor& indices, const Tensor& values,
                              const std::optional<std::vector<int64_t>>& size,
                              bool is_coalesced) {
    if (size.has_value()) {
        return Tensor::make_sparse_coo_tensor(indices, values, *size, is_coalesced);
    }
    // max(coord)+1; trailing dense dims come from the values' shape.
    Tensor canonical_indices = indices.dtype() == DType::Int64
        ? indices
        : indices.to(DType::Int64);
    if (!canonical_indices.is_contiguous()) {
        TP_THROW(RuntimeError,
                 "sparse_coo_tensor(): indices must be contiguous");
    }
    if (values.dim() == 0) {
        TP_THROW(ValueError,
                 "sparse_coo_tensor(): values must have an nnz dimension");
    }
    const int64_t sparse_dim = canonical_indices.size(0);
    const int64_t nnz = canonical_indices.size(1);
    Tensor maxima = Tensor::zeros(
        {sparse_dim}, DType::Int64, canonical_indices.device());
    if (nnz > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        coord_max_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                           stream>>>(
            nnz, sparse_dim, canonical_indices.data_ptr<int64_t>(),
            maxima.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA sparse_coo_tensor coord max kernel");
    }
    std::vector<int64_t> max_host(static_cast<size_t>(sparse_dim), 0);
    checkCuda(cudaMemcpy(max_host.data(), maxima.data_ptr<int64_t>(),
                         sparse_dim * sizeof(int64_t), cudaMemcpyDeviceToHost),
              "CUDA sparse_coo_tensor maxima readback");
    std::vector<int64_t> inferred;
    for (int64_t d = 0; d < sparse_dim; ++d) {
        inferred.push_back(max_host[static_cast<size_t>(d)] + 1);
    }
    auto values_shape = static_cast<std::vector<int64_t>>(values.shape());
    inferred.insert(inferred.end(), values_shape.begin() + 1,
                    values_shape.end());
    return Tensor::make_sparse_coo_tensor(canonical_indices, values, inferred,
                                          is_coalesced);
}

} // namespace cuda
} // namespace tensorplay
