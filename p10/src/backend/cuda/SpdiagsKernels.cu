#include "SparseKernels.h"
#include "CUDARuntime.h"
#include "GPUPrimitives.cuh"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor coo_to_csr_native_cuda(const Tensor& coalesced);

namespace {

constexpr int kCoalesceThreads = 128;

inline int coalesce_blocks(int64_t n) {
    return static_cast<int>((n + kCoalesceThreads - 1) / kCoalesceThreads);
}

} // namespace

__global__ void spdiags_fill_kernel(
    const unsigned char* diagonals,   // byte-typed base pointer
    int64_t length,
    const int64_t* offsets,
    const int64_t* starts,
    const int64_t* counts,
    int64_t elem_size,
    unsigned char* rows,              // byte views of the int64 outputs
    unsigned char* cols,
    unsigned char* values) {
    const int64_t j = blockIdx.x;
    const int64_t d = offsets[j];
    const int64_t count = counts[j];
    const int64_t slot = starts[j];
    const int64_t first_col = d > 0 ? d : 0;
    const int64_t first_row = first_col - d;
    const unsigned char* read =
        diagonals + j * length * elem_size + first_col * elem_size;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        *reinterpret_cast<int64_t*>(rows + (slot + i) * sizeof(int64_t)) =
            first_row + i;
        *reinterpret_cast<int64_t*>(cols + (slot + i) * sizeof(int64_t)) =
            first_col + i;
        for (int64_t b = 0; b < elem_size; ++b) {
            values[(slot + i) * elem_size + b] = read[i * elem_size + b];
        }
    }
}

__global__ void spdiags_count_kernel(int64_t n_diag, int64_t rows,
                                     int64_t cols, int64_t length,
                                     const int64_t* offsets, int64_t* counts) {
    const int64_t diagonal =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (diagonal >= n_diag) return;
    const int64_t offset = offsets[diagonal];
    const int64_t available = offset <= 0
        ? ((offset + rows) < length ? offset + rows : length)
        : ((cols < length ? cols : length) - offset);
    counts[diagonal] = available > 0 ? available : 0;
}

__global__ void spdiags_starts_kernel(int64_t n_diag, const int64_t* counts,
                                      const int64_t* cumulative,
                                      int64_t* starts) {
    const int64_t diagonal =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (diagonal < n_diag) {
        starts[diagonal] = cumulative[diagonal] - counts[diagonal];
    }
}

__global__ void spdiags_duplicate_kernel(int64_t n_offsets,
                                         const int64_t* sorted_offsets,
                                         int32_t* duplicate) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index > 0 && index < n_offsets &&
        sorted_offsets[index] == sorted_offsets[index - 1]) {
        atomicExch(duplicate, 1);
    }
}

Tensor spdiags_cuda(const Tensor& diagonals, const Tensor& offsets,
                    const std::vector<int64_t>& shape,
                    std::optional<int64_t> layout) {
    if (layout.has_value() && *layout != 0 && *layout != 1) {
        TP_THROW(ValueError,
                 "spdiags(): only sparse_coo (0) and sparse_csr (1) output "
                 "layouts are supported");
    }
    if (shape.size() != 2) {
        TP_THROW(ValueError, "spdiags(): output shape must be 2-dimensional");
    }
    Tensor diags2d = diagonals.dim() == 1 ? diagonals.unsqueeze(0) : diagonals;
    if (diags2d.dim() != 2) {
        TP_THROW(ValueError, "spdiags(): diagonals must be a vector or matrix");
    }
    if (diags2d.device() != offsets.device()) {
        TP_THROW(DeviceMismatchError,
                 "spdiags(): diagonals and offsets must share one device");
    }
    Tensor offs = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;
    if (offs.dim() != 1 || offs.dtype() != DType::Int64) {
        TP_THROW(TypeError, "spdiags(): offset tensor must be 1-D int64");
    }
    const int64_t n_diag = offs.size(0);
    if (diags2d.size(0) != n_diag) {
        TP_THROW(ValueError,
                 "spdiags(): number of diagonals (" +
                     std::to_string(diags2d.size(0)) +
                     ") does not match the number of offsets (" +
                     std::to_string(n_diag) + ")");
    }

    const int64_t m_size = shape[0];
    const int64_t n_size = shape[1];
    const int64_t length = diags2d.size(1);

    Tensor offs_c = offs.contiguous();
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    if (n_diag > 1) {
        Tensor sorted_offsets = Tensor::empty(
            {n_diag}, DType::Int64, offsets.device());
        size_t sort_bytes = 0;
        checkCuda(cub::DeviceRadixSort::SortKeys(
                      nullptr, sort_bytes, offs_c.data_ptr<int64_t>(),
                      sorted_offsets.data_ptr<int64_t>(),
                      static_cast<int>(n_diag), 0, sizeof(int64_t) * 8,
                      stream),
                  "CUB spdiags offset sort size query");
        Tensor sort_temporary = Tensor::empty(
            {static_cast<int64_t>(sort_bytes == 0 ? 1 : sort_bytes)},
            DType::UInt8, offsets.device());
        checkCuda(cub::DeviceRadixSort::SortKeys(
                      sort_temporary.data_ptr(), sort_bytes,
                      offs_c.data_ptr<int64_t>(), sorted_offsets.data_ptr<int64_t>(),
                      static_cast<int>(n_diag), 0, sizeof(int64_t) * 8,
                      stream),
                  "CUB spdiags offset sort");
        Tensor duplicate = Tensor::zeros({1}, DType::Int32, offsets.device());
        spdiags_duplicate_kernel<<<coalesce_blocks(n_diag), kCoalesceThreads,
                                   0, stream>>>(
            n_diag, sorted_offsets.data_ptr<int64_t>(),
            duplicate.data_ptr<int32_t>());
        checkCuda(cudaGetLastError(), "CUDA spdiags duplicate check");
        int32_t duplicate_host = 0;
        checkCuda(cudaMemcpyAsync(&duplicate_host, duplicate.data_ptr<int32_t>(),
                                  sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  stream),
                  "CUDA spdiags duplicate readback");
        checkCuda(cudaStreamSynchronize(stream), "CUDA spdiags metadata sync");
        if (duplicate_host != 0) {
            TP_THROW(ValueError, "spdiags(): offset tensor contains duplicate values");
        }
    }

    Tensor counts_d = Tensor::empty({n_diag}, DType::Int64, offsets.device());
    Tensor starts_d = Tensor::empty({n_diag}, DType::Int64, offsets.device());
    Tensor cumulative_d = Tensor::empty(
        {n_diag}, DType::Int64, offsets.device());
    int64_t total_nnz = 0;
    if (n_diag > 0) {
        spdiags_count_kernel<<<coalesce_blocks(n_diag), kCoalesceThreads, 0,
                               stream>>>(
            n_diag, m_size, n_size, length, offs_c.data_ptr<int64_t>(),
            counts_d.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA spdiags count kernel");
        size_t scan_bytes = 0;
        checkCuda(cub::DeviceScan::InclusiveSum(
                      nullptr, scan_bytes, counts_d.data_ptr<int64_t>(),
                      cumulative_d.data_ptr<int64_t>(),
                      static_cast<int>(n_diag), stream),
                  "CUB spdiags count scan size query");
        Tensor scan_temporary = Tensor::empty(
            {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
            DType::UInt8, offsets.device());
        checkCuda(cub::DeviceScan::InclusiveSum(
                      scan_temporary.data_ptr(), scan_bytes,
                      counts_d.data_ptr<int64_t>(),
                      cumulative_d.data_ptr<int64_t>(),
                      static_cast<int>(n_diag), stream),
                  "CUB spdiags count scan");
        spdiags_starts_kernel<<<coalesce_blocks(n_diag), kCoalesceThreads, 0,
                                stream>>>(
            n_diag, counts_d.data_ptr<int64_t>(),
            cumulative_d.data_ptr<int64_t>(), starts_d.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA spdiags starts kernel");
        checkCuda(cudaMemcpyAsync(
                      &total_nnz,
                      cumulative_d.data_ptr<int64_t>() + (n_diag - 1),
                      sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
                  "CUDA spdiags nnz readback");
        checkCuda(cudaStreamSynchronize(stream), "CUDA spdiags metadata sync");
    }

    Tensor diags_c = diags2d.contiguous();
    Tensor indices = Tensor::empty({2, total_nnz}, DType::Int64,
                                   offsets.device());
    Tensor values = Tensor::empty({total_nnz}, diags_c.dtype(),
                                  diags_c.device());
    if (total_nnz > 0 && n_diag > 0) {
        const size_t elem = diags_c.itemsize();
        const cudaStream_t fill_stream = getCurrentCUDAStream().stream();
        spdiags_fill_kernel<<<n_diag, 128, 0, fill_stream>>>(
            reinterpret_cast<const unsigned char*>(diags_c.data_ptr()),
            length, offs_c.data_ptr<int64_t>(), starts_d.data_ptr<int64_t>(),
            counts_d.data_ptr<int64_t>(), static_cast<int64_t>(elem),
            reinterpret_cast<unsigned char*>(indices.data_ptr<int64_t>()),
            reinterpret_cast<unsigned char*>(indices.data_ptr<int64_t>()) +
                total_nnz * sizeof(int64_t),
            reinterpret_cast<unsigned char*>(values.data_ptr()));
        checkCuda(cudaGetLastError(), "CUDA spdiags fill kernel");
    }

    auto result = Tensor::make_sparse_coo_tensor(indices, values, shape,
                                                 /*is_coalesced=*/false);
    if (layout.has_value() && *layout == 1) {
        return coo_to_csr_native_cuda(result.coalesce());
    }
    return result;
}


} // namespace cuda
} // namespace tensorplay
