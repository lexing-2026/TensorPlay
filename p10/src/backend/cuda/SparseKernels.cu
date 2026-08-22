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
                              const std::vector<int64_t>& size, bool is_coalesced) {
    return Tensor::make_sparse_coo_tensor(indices, values, size, is_coalesced);
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

} // namespace cuda
} // namespace tensorplay
