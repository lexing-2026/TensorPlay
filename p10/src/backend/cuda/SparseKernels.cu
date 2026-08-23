#include "SparseKernels.h"
#include "CUDARuntime.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <climits>

namespace tensorplay {
namespace cuda {
namespace {

constexpr int kMaxSparseDims = 64;

struct SparseGatherInfo {
    int sparse_dim;
    int dense_dim;
    int64_t shape[kMaxSparseDims];
    int64_t strides[kMaxSparseDims];
};

template <typename index_t>
__global__ void sparse_embedding_keep_kernel(
    int64_t num_indices,
    const index_t* indices,
    int64_t padding_idx,
    bool* keep) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < num_indices) {
        // This intentionally does not normalize or range-check indices.  The
        // ATen sparse backward helper filters only the exact padding index;
        // embedding's forward/backward wrapper owns the normal validation.
        keep[index] = static_cast<int64_t>(indices[index]) != padding_idx;
    }
}

template <typename index_t>
__global__ void sparse_embedding_pack_indices_kernel(
    int64_t selected_count,
    const index_t* indices,
    const int64_t* selected_positions,
    int64_t* output_indices) {
    const int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output < selected_count) {
        output_indices[output] = static_cast<int64_t>(
            indices[selected_positions[output]]);
    }
}

__global__ void sparse_embedding_pack_values_kernel(
    int64_t output_numel,
    int64_t row_size,
    int64_t itemsize,
    const int64_t* selected_positions,
    const uint8_t* grad,
    uint8_t* output) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= output_numel) return;
    const int64_t row = row_size == 0 ? 0 : linear / row_size;
    const int64_t column = row_size == 0 ? 0 : linear % row_size;
    const int64_t source_row = selected_positions[row];
    const int64_t source_offset = (source_row * row_size + column) * itemsize;
    const int64_t output_offset = linear * itemsize;
    for (int64_t byte = 0; byte < itemsize; ++byte) {
        output[output_offset + byte] = grad[source_offset + byte];
    }
}

template <typename scalar_t>
__global__ void sparse_mask_gather_kernel(
    int64_t output_numel,
    const int64_t* indices,
    int64_t nnz,
    int64_t dense_numel,
    const scalar_t* dense,
    scalar_t* values,
    SparseGatherInfo info) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= output_numel) return;

    const int64_t entry = linear / dense_numel;
    const int64_t inner = linear % dense_numel;

    int64_t source_offset = 0;
    for (int d = 0; d < info.sparse_dim; ++d) {
        source_offset += indices[d * nnz + entry] * info.strides[d];
    }
    int64_t remainder = inner;
    for (int d = info.dense_dim - 1; d >= 0; --d) {
        const int64_t dim_size = info.shape[info.sparse_dim + d];
        const int64_t coordinate = dim_size == 0 ? 0 : remainder % dim_size;
        remainder = dim_size == 0 ? 0 : remainder / dim_size;
        source_offset += coordinate * info.strides[info.sparse_dim + d];
    }
    values[linear] = dense[source_offset];
}

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

SparseGatherInfo make_gather_info(const Tensor& dense, const Tensor& mask) {
    SparseGatherInfo info{};
    info.sparse_dim = static_cast<int>(mask.sparse_dim());
    info.dense_dim = static_cast<int>(mask.dense_dim());
    if (dense.dim() > kMaxSparseDims) {
        TP_THROW(RuntimeError, "sparse_mask(): tensor rank exceeds CUDA sparse limit");
    }
    for (int64_t d = 0; d < dense.dim(); ++d) {
        info.shape[d] = dense.size(d);
        info.strides[d] = dense.stride(d);
    }
    return info;
}

} // namespace

Tensor sparse_coo_tensor_cuda(const Tensor& indices, const Tensor& values,
                              std::optional<std::vector<int64_t>> size,
                              bool is_coalesced) {
    if (size.has_value()) {
        return Tensor::make_sparse_coo_tensor(indices, values, *size, is_coalesced);
    }
    // Size inference reads the coordinate rows; stage them through the CPU
    // and reuse the CPU inference logic.
    Tensor host_indices = indices.to(Device(DeviceType::CPU));
    Tensor host_values = values.to(Device(DeviceType::CPU));
    Tensor staged = cpu::sparse_coo_tensor_cpu(
        host_indices, host_values, std::nullopt, is_coalesced);
    return staged.to(values.device());
}

Tensor coalesce_sparse_cuda(const Tensor& self) {
    // COO coalescing is a sorting/reduction operation.  The dense CUDA
    // kernels above are intentionally kept asynchronous, but this fallback
    // uses the same CPU algorithm as ATen's canonical COO implementation for
    // now; the component transfers are ordered by Tensor::to().
    Tensor host = self.to(Device(DeviceType::CPU));
    Tensor result = host.coalesce();
    return result.to(self.device());
}

Tensor sparse_mask_cuda(const Tensor& dense, const Tensor& mask) {
    if (!mask.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse COO");
    }
    if (dense.device() != mask.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mask(): dense and mask must be on the same device");
    }
    if (dense.shape() != mask.shape()) {
        TP_THROW(RuntimeError,
                 "sparse_mask(): operands have incompatible sizes; self and mask must have the same shape");
    }
    // Preserve the mask's COO ordering and duplicate entries.  This is the
    // same projection semantics as ATen::sparse_mask; coalescing belongs to
    // callers that explicitly request it.
    Tensor canonical_mask = mask;
    Tensor dense_contiguous = dense.is_contiguous() ? dense : dense.contiguous();
    Tensor indices = canonical_mask._indices().contiguous();
    const int64_t nnz = indices.size(1);
    int64_t dense_numel = 1;
    for (int64_t d = canonical_mask.sparse_dim(); d < canonical_mask.dim(); ++d) {
        dense_numel *= canonical_mask.size(d);
    }

    std::vector<int64_t> values_shape = {nnz};
    for (int64_t d = canonical_mask.sparse_dim(); d < canonical_mask.dim(); ++d) {
        values_shape.push_back(canonical_mask.size(d));
    }
    Tensor values = Tensor::empty(values_shape, dense.dtype(), dense.device());
    const int64_t output_numel = values.numel();
    if (output_numel == 0) {
        return Tensor::make_sparse_coo_tensor(
            indices, values, static_cast<std::vector<int64_t>>(mask.shape()),
            mask.is_coalesced());
    }

    SparseGatherInfo info = make_gather_info(dense_contiguous, canonical_mask);
    const int threads = 256;
    const int blocks = static_cast<int>((output_numel + threads - 1) / threads);
#define TP_SPARSE_GATHER_CASE(ctype, name) \
    case DType::name: \
        sparse_mask_gather_kernel<ctype><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>( \
            output_numel, indices.data_ptr<int64_t>(), nnz, dense_numel, \
            dense_contiguous.data_ptr<ctype>(), values.data_ptr<ctype>(), info); \
        break;
    switch (dense.dtype()) {
        TP_SPARSE_GATHER_CASE(uint8_t, UInt8)
        TP_SPARSE_GATHER_CASE(int8_t, Int8)
        TP_SPARSE_GATHER_CASE(int16_t, Int16)
        TP_SPARSE_GATHER_CASE(int32_t, Int32)
        TP_SPARSE_GATHER_CASE(int64_t, Int64)
        TP_SPARSE_GATHER_CASE(uint16_t, UInt16)
        TP_SPARSE_GATHER_CASE(uint32_t, UInt32)
        TP_SPARSE_GATHER_CASE(uint64_t, UInt64)
        TP_SPARSE_GATHER_CASE(float, Float32)
        TP_SPARSE_GATHER_CASE(double, Float64)
        TP_SPARSE_GATHER_CASE(tensorplay::Half, Float16)
        TP_SPARSE_GATHER_CASE(tensorplay::BFloat16, BFloat16)
        TP_SPARSE_GATHER_CASE(bool, Bool)
        default: {
            // std::complex is a host-only project type in CUDA translation
            // units.  Preserve exact values through the ordinary copy path.
            Tensor host = dense.to(Device(DeviceType::CPU));
            Tensor host_mask = canonical_mask.to(Device(DeviceType::CPU));
            return cpu::sparse_mask_cpu(host, host_mask).to(dense.device());
        }
    }
#undef TP_SPARSE_GATHER_CASE
    checkCuda(cudaGetLastError(), "CUDA sparse_mask gather kernel");
    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(mask.shape()),
        mask.is_coalesced());
}

Tensor& add_sparse_to_dense_cuda(Tensor& dense, const Tensor& sparse, Scalar alpha) {
    if (dense.is_sparse() || !sparse.is_sparse()) {
        TP_THROW(RuntimeError, "add_: expected a dense self and sparse COO other");
    }
    if (dense.shape() != sparse.shape()) {
        TP_THROW(RuntimeError, "add_: sparse COO operands must have identical sizes");
    }
    if (dense.dtype() == DType::ComplexHalf || dense.dtype() == DType::ComplexFloat ||
        dense.dtype() == DType::ComplexDouble || dense.dtype() == DType::BComplex32) {
        Tensor host_dense = dense.to(Device(DeviceType::CPU));
        Tensor host_sparse = sparse.to(Device(DeviceType::CPU));
        cpu::add_sparse_to_dense_cpu(host_dense, host_sparse, alpha);
        dense.copy_(host_dense);
        return dense;
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

Tensor embedding_sparse_backward_cuda(const Tensor& grad,
                                      const Tensor& indices,
                                      int64_t num_weights,
                                      int64_t padding_idx,
                                      bool scale_grad_by_freq) {
    if (scale_grad_by_freq) {
        TP_THROW(RuntimeError,
                 "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
    }
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_sparse_backward: indices must be Int64 or Int32");
    }
    if (grad.dim() == 0) {
        TP_THROW(RuntimeError,
                 "embedding_sparse_backward: grad must have a feature dimension");
    }
    if (indices.device() != grad.device()) {
        TP_THROW(DeviceMismatchError,
                 "embedding_backward: grad and indices must be on the same CUDA device");
    }

    const int64_t num_indices = indices.numel();
    const int64_t row_size = grad.size(grad.dim() - 1);
    if (grad.numel() != num_indices * row_size) {
        TP_THROW(RuntimeError,
                 "embedding_sparse_backward: incompatible grad and indices shapes");
    }

    Tensor grad_contiguous = grad.contiguous();
    Tensor indices_contiguous = indices.contiguous();
    Tensor index_flat = indices_contiguous.view({num_indices});
    Tensor output_indices;
    Tensor output_values;

    // With no padding filtering, the ATen helper is just a pair of views plus
    // the canonical int64 index conversion.  This avoids both a launch and a
    // device-to-host synchronization for the common embedding case.
    if (padding_idx == -1) {
        output_indices = index_flat.view({1, num_indices});
        if (output_indices.dtype() != DType::Int64) {
            output_indices = output_indices.to(DType::Int64);
        }
        output_values = grad_contiguous.view({num_indices, row_size});
        return Tensor::make_sparse_coo_tensor(
            output_indices, output_values, {num_weights, row_size}, false);
    }

    if (num_indices == 0) {
        output_indices = Tensor::empty({1, 0}, DType::Int64, grad.device());
        output_values = Tensor::empty({0, row_size}, grad.dtype(), grad.device());
        return Tensor::make_sparse_coo_tensor(
            output_indices, output_values, {num_weights, row_size}, true);
    }
    if (num_indices > static_cast<int64_t>(INT_MAX)) {
        TP_THROW(ValueError,
                 "embedding_sparse_backward: CUDA index list exceeds CUB's item limit");
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((num_indices + threads - 1) / threads);
    Tensor keep = Tensor::empty({num_indices}, DType::Bool, grad.device());
    if (index_flat.dtype() == DType::Int64) {
        sparse_embedding_keep_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            num_indices, index_flat.data_ptr<int64_t>(), padding_idx,
            keep.data_ptr<bool>());
    } else {
        sparse_embedding_keep_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            num_indices, index_flat.data_ptr<int32_t>(), padding_idx,
            keep.data_ptr<bool>());
    }
    checkCuda(cudaGetLastError(), "CUDA sparse embedding padding filter");

    Tensor selected_positions = Tensor::empty(
        {num_indices}, DType::Int64, grad.device());
    Tensor selected_count = Tensor::zeros({1}, DType::Int64, grad.device());
    // CUDA 13 / CCCL 3 removed both cub::CountingInputIterator and the
    // experimental <cuda/iterator>; thrust::counting_iterator ships in the
    // same CCCL package and satisfies DeviceSelect::Flagged.
    thrust::counting_iterator<int64_t> counting(0);
    size_t temporary_bytes = 0;
    checkCuda(cub::DeviceSelect::Flagged(
        nullptr, temporary_bytes, counting, keep.data_ptr<bool>(),
        selected_positions.data_ptr<int64_t>(), selected_count.data_ptr<int64_t>(),
        static_cast<int>(num_indices), stream),
        "CUB sparse embedding select size");
    Tensor temporary = Tensor::empty(
        {static_cast<int64_t>(temporary_bytes == 0 ? 1 : temporary_bytes)},
        DType::UInt8, grad.device());
    checkCuda(cub::DeviceSelect::Flagged(
        temporary.data_ptr(), temporary_bytes, counting, keep.data_ptr<bool>(),
        selected_positions.data_ptr<int64_t>(), selected_count.data_ptr<int64_t>(),
        static_cast<int>(num_indices), stream),
        "CUB sparse embedding select");

    int64_t selected = 0;
    checkCuda(cudaMemcpyAsync(&selected, selected_count.data_ptr<int64_t>(),
                              sizeof(selected), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse embedding selected-count copy");
    checkCuda(cudaStreamSynchronize(stream),
              "CUDA sparse embedding selected-count synchronization");

    output_indices = Tensor::empty({1, selected}, DType::Int64, grad.device());
    output_values = Tensor::empty({selected, row_size}, grad.dtype(), grad.device());
    if (selected == 0) {
        return Tensor::make_sparse_coo_tensor(
            output_indices, output_values, {num_weights, row_size}, true);
    }

    const int selected_blocks = static_cast<int>((selected + threads - 1) / threads);
    if (index_flat.dtype() == DType::Int64) {
        sparse_embedding_pack_indices_kernel<int64_t><<<
            selected_blocks, threads, 0, stream>>>(
            selected, index_flat.data_ptr<int64_t>(),
            selected_positions.data_ptr<int64_t>(),
            output_indices.data_ptr<int64_t>());
    } else {
        sparse_embedding_pack_indices_kernel<int32_t><<<
            selected_blocks, threads, 0, stream>>>(
            selected, index_flat.data_ptr<int32_t>(),
            selected_positions.data_ptr<int64_t>(),
            output_indices.data_ptr<int64_t>());
    }

    const int64_t output_numel = selected * row_size;
    if (output_numel > 0) {
        const int value_blocks = static_cast<int>((output_numel + threads - 1) / threads);
        sparse_embedding_pack_values_kernel<<<value_blocks, threads, 0, stream>>>(
            output_numel, row_size, static_cast<int64_t>(grad.itemsize()),
            selected_positions.data_ptr<int64_t>(),
            static_cast<const uint8_t*>(grad_contiguous.data_ptr()),
            static_cast<uint8_t*>(output_values.data_ptr()));
    }
    checkCuda(cudaGetLastError(), "CUDA sparse embedding pack");
    return Tensor::make_sparse_coo_tensor(
        output_indices, output_values, {num_weights, row_size}, selected <= 1);
}

namespace {

// Layout of a freshly allocated contiguous dense output, passed by value so
// kernels read it from parameter space (mirrors SparseGatherInfo).
struct DenseLayoutInfo {
    int64_t ndim;
    int64_t shape[kMaxSparseDims];
    int64_t strides[kMaxSparseDims];
};

DenseLayoutInfo make_layout_info(const std::vector<int64_t>& sizes) {
    TP_CHECK(static_cast<int64_t>(sizes.size()) <= kMaxSparseDims,
             "to_dense(): tensor rank exceeds CUDA sparse limit");
    DenseLayoutInfo info{};
    info.ndim = static_cast<int64_t>(sizes.size());
    int64_t stride = 1;
    for (int64_t d = info.ndim - 1; d >= 0; --d) {
        info.shape[d] = sizes[static_cast<size_t>(d)];
        info.strides[d] = stride;
        stride *= sizes[static_cast<size_t>(d)];
    }
    return info;
}

// One thread per (stored element, inner column).  Byte-wise copies keep the
// scatter dtype-agnostic (same trick as sparse_embedding_pack_values_kernel).
__global__ void sparse_coo_to_dense_kernel(
    int64_t total,
    int64_t nnz,
    int64_t dense_numel,
    int64_t itemsize,
    int64_t sparse_dim,
    DenseLayoutInfo layout,
    const int64_t* indices,
    const uint8_t* values,
    uint8_t* out) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int64_t n = linear / dense_numel;
    const int64_t j = linear % dense_numel;
    int64_t destination = 0;
    for (int64_t d = 0; d < sparse_dim; ++d) {
        destination += indices[d * nnz + n] * layout.strides[d];
    }
    int64_t remainder = j;
    for (int64_t d = sparse_dim; d < layout.ndim; ++d) {
        const int64_t coordinate = (remainder / layout.strides[d]) %
                                   layout.shape[d];
        destination += coordinate * layout.strides[d];
        remainder -= (remainder / layout.strides[d]) * layout.strides[d];
    }
    uint8_t* destination_bytes = out + destination * itemsize;
    const uint8_t* source_bytes = values + linear * itemsize;
    for (int64_t byte = 0; byte < itemsize; ++byte) {
        destination_bytes[byte] = source_bytes[byte];
    }
}

__global__ void sparse_csr_to_dense_kernel(
    int64_t rows,
    int64_t cols,
    const int64_t* crow,
    const int64_t* col,
    const uint8_t* values,
    uint8_t* out,
    int64_t itemsize) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    for (int64_t t = crow[i]; t < crow[i + 1]; ++t) {
        uint8_t* destination = out + (i * cols + col[t]) * itemsize;
        const uint8_t* source = values + t * itemsize;
        for (int64_t byte = 0; byte < itemsize; ++byte) {
            destination[byte] = source[byte];
        }
    }
}

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
    // Post-coalesce coordinates are unique, so two threads only collide on
    // the same output cell when their coordinates match exactly.
    out[row_indices[n] * cols + j] +=
        values[n] * dense[col_indices[n] * cols + j];
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

template <typename scalar_t>
__global__ void sparse_sum_reduce_kernel(
    int64_t numel,
    const scalar_t* data,
    scalar_t* out) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = index; i < numel; i += grid_stride) {
        atomicAdd(out, data[i]);
    }
}

int64_t product_of(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (int64_t dim : dims) result *= dim;
    return result;
}

} // namespace

Tensor to_dense_sparse_cuda(const Tensor& self) {
    if (!self.is_sparse()) return self;

    if (self.is_sparse_csr()) {
        if (self.dim() != 2) {
            TP_THROW(RuntimeError, "to_dense(): CSR tensors must be 2-D");
        }
        Tensor crow = self._crow_indices().contiguous();
        Tensor col = self._col_indices().contiguous();
        Tensor values = self._values().contiguous();
        if (values.dim() != 1) {
            TP_THROW(RuntimeError,
                     "to_dense(): hybrid CSR tensors are not supported");
        }
        Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());
        const int64_t rows = self.size(0);
        const int64_t cols = self.size(1);
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        const int threads = 128;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);
        sparse_csr_to_dense_kernel<<<blocks, threads, 0, stream>>>(
            rows, cols,
            crow.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
            reinterpret_cast<const uint8_t*>(values.data_ptr()),
            reinterpret_cast<uint8_t*>(out.data_ptr()),
            static_cast<int64_t>(values.itemsize()));
        checkCuda(cudaGetLastError(), "CUDA CSR to_dense kernel");
        return out;
    }

    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());

    const int64_t sparse_dim = canonical.sparse_dim();
    std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(canonical.shape());
    DenseLayoutInfo layout = make_layout_info(sizes);
    int64_t dense_numel = product_of(std::vector<int64_t>(
        sizes.begin() + sparse_dim, sizes.end()));
    const int64_t nnz = indices.size(1);
    const int64_t total = nnz * dense_numel;
    if (total == 0) return out;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    sparse_coo_to_dense_kernel<<<blocks, threads, 0, stream>>>(
        total, nnz, dense_numel, static_cast<int64_t>(values.itemsize()),
        sparse_dim, layout,
        indices.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t*>(values.data_ptr()),
        reinterpret_cast<uint8_t*>(out.data_ptr()));
    checkCuda(cudaGetLastError(), "CUDA COO to_dense kernel");
    return out;
}

int64_t sparse_nnz_cuda(const Tensor& self) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "_nnz(): expected a sparse tensor");
    }
    return self._values().size(0);
}

Tensor to_sparse_coo_cuda(const Tensor& self) {
    if (self.is_sparse()) return self.coalesce();
    // Nonzero extraction needs a device-wide compaction; stage through the
    // CPU like coalesce_cuda does (conversions, not hot-loop ops).
    Tensor host = self.to(Device(DeviceType::CPU));
    Tensor sparse_host = cpu::to_sparse_coo_cpu(host);
    return sparse_host.to(self.device());
}

Tensor to_sparse_csr_cuda(const Tensor& self) {
    Tensor host = self.to(Device(DeviceType::CPU));
    Tensor sparse_host = cpu::to_sparse_csr_cpu(host);
    return sparse_host.to(self.device());
}

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

#define TP_SPARSE_MM_CASE(ctype, name)                                        \
    case DType::name: {                                                       \
        using scalar_t = ctype;                                               \
        Tensor dense_contiguous =                                             \
            dense.is_contiguous() ? dense : dense.contiguous();               \
        Tensor out =                                                          \
            Tensor::zeros({self.size(0), dense.size(1)}, self.dtype(),        \
                          self.device());                                     \
        const int64_t cols = dense.size(1);                                   \
        const cudaStream_t mm_stream = getCurrentCUDAStream().stream();       \
        if (self.is_sparse_csr()) {                                           \
            Tensor crow = self._crow_indices().contiguous();                  \
            Tensor col = self._col_indices().contiguous();                    \
            Tensor values = self._values().contiguous();                      \
            const int64_t total = self.size(0) * cols;                        \
            if (total > 0) {                                                  \
                const int blocks =                                            \
                    static_cast<int>((total + threads - 1) / threads);        \
                sparse_csr_mm_kernel<scalar_t><<<blocks, threads, 0,          \
                                                 mm_stream>>>(                \
                    total, cols, crow.data_ptr<int64_t>(),                    \
                    col.data_ptr<int64_t>(),                                  \
                    values.data_ptr<scalar_t>(),                              \
                    dense_contiguous.data_ptr<scalar_t>(),                    \
                    out.data_ptr<scalar_t>());                                \
            }                                                                 \
        } else {                                                              \
            Tensor canonical =                                                \
                self.is_coalesced() ? self : self.coalesce();                 \
            Tensor indices = canonical._indices().contiguous();               \
            Tensor values = canonical._values().contiguous();                 \
            if (values.dim() != 1) {                                          \
                TP_THROW(RuntimeError,                                        \
                         "sparse_mm(): hybrid COO tensors are not supported");\
            }                                                                 \
            const int64_t nnz = indices.size(1);                              \
            const int64_t total = nnz * cols;                                 \
            if (total > 0) {                                                  \
                const int blocks =                                            \
                    static_cast<int>((total + threads - 1) / threads);        \
                sparse_coo_mm_kernel<scalar_t><<<blocks, threads, 0,          \
                                                 mm_stream>>>(                \
                    total, cols, indices.data_ptr<int64_t>(),                 \
                    indices.data_ptr<int64_t>() + nnz,                        \
                    values.data_ptr<scalar_t>(),                              \
                    dense_contiguous.data_ptr<scalar_t>(),                    \
                    out.data_ptr<scalar_t>());                                \
            }                                                                 \
        }                                                                     \
        checkCuda(cudaGetLastError(), "CUDA sparse_mm kernel");               \
        return out;                                                           \
    }

    constexpr int threads = 128;
    switch (self.dtype()) {
        TP_SPARSE_MM_CASE(float, Float32)
        TP_SPARSE_MM_CASE(double, Float64)
        default:
            break;
    }
#undef TP_SPARSE_MM_CASE

    // Non-float dtypes fall back to CPU staging.
    Tensor host_self = self.to(Device(DeviceType::CPU));
    Tensor host_dense = dense.to(Device(DeviceType::CPU));
    Tensor result_host = cpu::sparse_mm_cpu(host_self, host_dense);
    return result_host.to(self.device());
}

Tensor sparse_sum_cuda(const Tensor& self) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_sum(): expected a sparse tensor");
    }
    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor values = canonical._values().contiguous();
    const int64_t numel = values.numel();

#define TP_SPARSE_SUM_CASE(ctype, name)                                       \
    case DType::name: {                                                       \
        Tensor out = Tensor::zeros({}, self.dtype(), self.device());          \
        if (numel > 0) {                                                      \
            const cudaStream_t sum_stream = getCurrentCUDAStream().stream();  \
            const int blocks = static_cast<int>(                              \
                (numel + kSumThreads - 1) / kSumThreads);                     \
            sparse_sum_reduce_kernel<ctype><<<blocks, kSumThreads, 0,         \
                                              sum_stream>>>(                  \
                numel, values.data_ptr<ctype>(), out.data_ptr<ctype>());      \
            checkCuda(cudaGetLastError(), "CUDA sparse_sum kernel");          \
        }                                                                     \
        return out;                                                           \
    }
    constexpr int kSumThreads = 128;
    switch (self.dtype()) {
        TP_SPARSE_SUM_CASE(float, Float32)
        TP_SPARSE_SUM_CASE(double, Float64)
        default:
            break;
    }
#undef TP_SPARSE_SUM_CASE

    Tensor host = self.to(Device(DeviceType::CPU));
    Tensor result_host = cpu::sparse_sum_cpu(host);
    return result_host.to(self.device());
}

} // namespace cuda
} // namespace tensorplay
