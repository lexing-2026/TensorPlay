#include "sparse/SparseKernels.cuh"
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


template <typename index_t>
__global__ void sparse_embedding_keep_kernel(
    int64_t num_indices,
    const index_t* indices,
    int64_t padding_idx,
    bool* keep) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < num_indices) {
        // This intentionally does not normalize or range-check indices.  The
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

} // namespace


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
