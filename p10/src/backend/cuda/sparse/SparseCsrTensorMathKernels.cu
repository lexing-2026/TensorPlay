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


__global__ void csr_count_rows_kernel(int64_t nnz, const int64_t* row_coords,
                                      int64_t* counts) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nnz) atomicAdd(reinterpret_cast<unsigned long long*>(
                               counts + row_coords[i]),
                           1ull);
}


// Shared native dense->COO extraction: byte-level nonzero mask + CUB
// Flagged compaction over a counting iterator + coordinate gather.
__global__ void csr_rows_to_coo_kernel(int64_t rows, const int64_t* crow,
                                       int64_t* row_indices) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (int64_t entry = crow[row]; entry < crow[row + 1]; ++entry) {
        row_indices[entry] = row;
    }
}

} // namespace


// Native COO (coalesced, 2-D) -> canonical CSR.  The coalesced coordinate
// order is row-major, so columns stay ascending within each row.
Tensor coo_to_csr_native(const Tensor& coalesced) {
    Tensor indices = coalesced._indices().contiguous();
    Tensor values = coalesced._values().contiguous();
    const auto shape =
        static_cast<std::vector<int64_t>>(coalesced.shape());
    const int64_t rows = shape[0];
    const int64_t nnz = indices.size(1);
    const cudaStream_t stream = getCurrentCUDAStream().stream();

    Tensor counts = Tensor::zeros({rows}, DType::Int64, coalesced.device());
    if (nnz > 0) {
        csr_count_rows_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                stream>>>(
            nnz, indices.data_ptr<int64_t>(), counts.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA CSR row-count kernel");
    }
    Tensor crow = Tensor::zeros({rows + 1}, DType::Int64, coalesced.device());
    if (rows > 0) {
        size_t scan_bytes = 0;
        checkCuda(cub::DeviceScan::InclusiveSum(
                      nullptr, scan_bytes, counts.data_ptr<int64_t>(),
                      crow.data_ptr<int64_t>() + 1, static_cast<int>(rows),
                      stream),
                  "CUB CSR inclusive-sum size query");
        Tensor scan_temporary = Tensor::empty(
            {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
            DType::UInt8, coalesced.device());
        checkCuda(cub::DeviceScan::InclusiveSum(
                      scan_temporary.data_ptr(), scan_bytes,
                      counts.data_ptr<int64_t>(), crow.data_ptr<int64_t>() + 1,
                      static_cast<int>(rows), stream),
                  "CUB CSR inclusive sum");
    }
    return Tensor::make_sparse_csr_tensor(
        crow, indices.select(0, 1), values, shape);
}


Tensor coo_to_csr_native_cuda(const Tensor& coalesced) {
    return coo_to_csr_native(coalesced);
}


Tensor csr_to_coo_cuda(const Tensor& self) {
    if (!self.is_sparse_csr() || self.dim() != 2) {
        TP_THROW(RuntimeError, "CSR to COO conversion requires a 2-D CSR tensor");
    }
    Tensor crow = self._crow_indices().contiguous();
    Tensor col = self._col_indices().contiguous();
    Tensor values = self._values().contiguous();
    if (values.dim() != 1) {
        TP_THROW(RuntimeError,
                 "CSR to COO conversion does not support hybrid values");
    }
    const int64_t rows = self.size(0);
    const int64_t nnz = values.size(0);
    if (crow.size(0) != rows + 1 || col.size(0) != nnz) {
        TP_THROW(RuntimeError, "CSR index buffers do not match the tensor shape");
    }

    Tensor indices = Tensor::empty({2, nnz}, DType::Int64, self.device());
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    if (rows > 0) {
        csr_rows_to_coo_kernel<<<coalesce_blocks(rows), kCoalesceThreads, 0,
                                 stream>>>(
            rows, crow.data_ptr<int64_t>(), indices.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA CSR to COO row expansion kernel");
    }
    if (nnz > 0) {
        checkCuda(cudaMemcpyAsync(
                      indices.data_ptr<int64_t>() + nnz,
                      col.data_ptr<int64_t>(), nnz * sizeof(int64_t),
                      cudaMemcpyDeviceToDevice, stream),
                  "CUDA CSR to COO column copy");
    }
    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(self.shape()), false);
}


Tensor to_sparse_csr_cuda(const Tensor& self) {
    if (self.is_sparse()) {
        if (self.is_sparse_csr()) return self;
        return coo_to_csr_native(self.coalesce());
    }
    if (self.dim() != 2) {
        TP_THROW(RuntimeError,
                 "to_sparse_csr(): only 2-D input is supported, got " +
                     std::to_string(self.dim()) + "-D");
    }
    return coo_to_csr_native(to_sparse_coo_native(self));
}

} // namespace cuda
} // namespace tensorplay
