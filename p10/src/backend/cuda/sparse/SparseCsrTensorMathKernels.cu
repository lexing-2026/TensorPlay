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

constexpr int kSparseIndexThreads = 256;
constexpr int kSparseIndexMaxBatchDims = 64;

struct SparseBatchShape {
    int64_t ndim;
    int64_t dims[kSparseIndexMaxBatchDims];
};
namespace {


inline void sparse_index_cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("CUDA Error: ") + cudaGetErrorString(status));
    }
}


void clear_sparse_index_output(Tensor& result) {
    if (result.numel() == 0) return;
    sparse_index_cuda_check(cudaMemsetAsync(
        result.data_ptr(), 0,
        static_cast<size_t>(result.numel()) * result.itemsize(),
        getCurrentCUDAStream().stream()));
}


template <typename input_t, typename output_t>
__global__ void coo_to_csr_kernel(output_t* data_out, const input_t* data_in,
                                  int64_t size, int64_t numel) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t work = tid; work <= numel; work += stride) {
        if (work == 0) {
            const int64_t first = static_cast<int64_t>(data_in[0]);
            for (int64_t i = 0; i <= first; ++i) {
                data_out[i] = static_cast<output_t>(0);
            }
        } else if (work < numel) {
            const int64_t begin = static_cast<int64_t>(data_in[work - 1]);
            const int64_t end = static_cast<int64_t>(data_in[work]);
            for (int64_t i = begin; i < end; ++i) {
                data_out[i + 1] = static_cast<output_t>(work);
            }
        } else {
            const int64_t last = static_cast<int64_t>(data_in[numel - 1]);
            for (int64_t i = last + 1; i < size + 1; ++i) {
                data_out[i] = static_cast<output_t>(numel);
            }
        }
    }
}


template <typename input_t, typename output_t>
void fill_coo_to_csr_cuda(Tensor& result, const Tensor& input,
                          int64_t size) {
    const Tensor input_c = input.contiguous();
    const int64_t numel = input_c.numel();
    if (numel == 0) {
        clear_sparse_index_output(result);
        return;
    }
    const int64_t blocks =
        (numel + 1 + kSparseIndexThreads - 1) / kSparseIndexThreads;
    coo_to_csr_kernel<input_t, output_t>
        <<<static_cast<unsigned int>(blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            result.data_ptr<output_t>(), input_c.data_ptr<input_t>(), size,
            numel);
    sparse_index_cuda_check(cudaGetLastError());
}


template <typename output_t>
void dispatch_coo_to_csr_input_cuda(Tensor& result, const Tensor& input,
                                    int64_t size) {
#define TP_CUDA_COO_TO_CSR_CASE(ctype, name)                                  \
    case DType::name:                                                          \
        fill_coo_to_csr_cuda<ctype, output_t>(result, input, size);            \
        return;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_COO_TO_CSR_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_coo_to_csr: input must be integral");
    }
#undef TP_CUDA_COO_TO_CSR_CASE
}


void check_coo_to_csr_cuda(const Tensor& input, int64_t size) {
    TP_CHECK(input.dim() <= 1,
             "_convert_indices_from_coo_to_csr: input must be a vector, got ",
             input.dim(), " dimensions");
    TP_CHECK(size >= 0,
             "_convert_indices_from_coo_to_csr: size must be non-negative, got ",
             size);
    TP_CHECK(isIntegralType(input.dtype(), false),
             "_convert_indices_from_coo_to_csr: input must be integral");
}


template <typename input_t, typename output_t>
__global__ void csr_to_coo_rows_kernel(output_t* row_data,
                                       const input_t* crow_data,
                                       int64_t nrows, int64_t nnz,
                                       int64_t nbatches) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t row_count = nrows * nbatches;
    for (int64_t linear_row = tid; linear_row < row_count;
         linear_row += stride) {
        const int64_t batch = linear_row / nrows;
        const int64_t row = linear_row % nrows;
        const int64_t base = batch * (nrows + 1);
        const int64_t begin = static_cast<int64_t>(crow_data[base + row]);
        const int64_t end = static_cast<int64_t>(crow_data[base + row + 1]);
        for (int64_t index = begin; index < end; ++index) {
            row_data[batch * nnz + index] = static_cast<output_t>(row);
        }
    }
}


template <typename col_t, typename output_t>
__global__ void csr_to_coo_columns_kernel(
    output_t* result_data, output_t* row0, output_t* row1,
    const col_t* col_data, int64_t total_nnz, int64_t nnz,
    SparseBatchShape batch_shape, bool transpose) {
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = tid; index < total_nnz; index += stride) {
        const int64_t batch = index / nnz;
        int64_t remainder = batch;
        for (int64_t dim = batch_shape.ndim - 1; dim >= 0; --dim) {
            const int64_t extent = batch_shape.dims[dim];
            result_data[dim * total_nnz + index] =
                static_cast<output_t>(remainder % extent);
            remainder /= extent;
        }
        if (transpose) {
            row0[index] = static_cast<output_t>(col_data[index]);
        } else {
            row1[index] = static_cast<output_t>(col_data[index]);
        }
    }
}


template <typename crow_t, typename col_t, typename output_t>
void fill_csr_to_coo_cuda(Tensor& result, const Tensor& crow_indices,
                          const Tensor& col_indices, bool transpose,
                          const std::vector<int64_t>& batch_shape) {
    const Tensor crow_c = crow_indices.contiguous();
    const Tensor col_c = col_indices.contiguous();
    const int64_t nrows = crow_c.size(-1) - 1;
    const int64_t nnz = col_c.size(-1);
    const int64_t total_nnz = col_c.numel();
    const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
    const int64_t batch_count =
        batch_ndim == 0 ? 1 : (nnz == 0 ? 0 : total_nnz / nnz);
    if (nrows == 0 || nnz == 0 || total_nnz == 0) {
        clear_sparse_index_output(result);
        return;
    }

    SparseBatchShape shape{};
    shape.ndim = batch_ndim;
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        shape.dims[dim] = batch_shape[static_cast<size_t>(dim)];
    }
    output_t* result_data = result.data_ptr<output_t>();
    output_t* row0 = result.select(0, transpose ? batch_ndim + 1 : batch_ndim)
                          .data_ptr<output_t>();
    output_t* row1 = result.select(0, transpose ? batch_ndim : batch_ndim + 1)
                          .data_ptr<output_t>();
    const int64_t row_count = batch_count * nrows;
    const int row_blocks =
        static_cast<int>((row_count + kSparseIndexThreads - 1) /
                         kSparseIndexThreads);
    csr_to_coo_rows_kernel<crow_t, output_t>
        <<<static_cast<unsigned int>(row_blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            transpose ? row1 : row0, crow_c.data_ptr<crow_t>(), nrows, nnz,
            batch_count);
    sparse_index_cuda_check(cudaGetLastError());

    const int64_t column_blocks =
        (total_nnz + kSparseIndexThreads - 1) / kSparseIndexThreads;
    csr_to_coo_columns_kernel<col_t, output_t>
        <<<static_cast<unsigned int>(column_blocks), kSparseIndexThreads, 0,
           getCurrentCUDAStream().stream()>>>(
            result_data, row0, row1, col_c.data_ptr<col_t>(), total_nnz, nnz,
            shape, transpose);
    sparse_index_cuda_check(cudaGetLastError());
}


template <typename crow_t, typename output_t>
void dispatch_csr_to_coo_columns_cuda(
    Tensor& result, const Tensor& crow_indices, const Tensor& col_indices,
    bool transpose, const std::vector<int64_t>& batch_shape) {
#define TP_CUDA_CSR_TO_COO_CASE(ctype, name)                                  \
    case DType::name:                                                          \
        fill_csr_to_coo_cuda<crow_t, ctype, output_t>(                         \
            result, crow_indices, col_indices, transpose, batch_shape);        \
        return;
    switch (col_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_CSR_TO_COO_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: columns must be integral");
    }
#undef TP_CUDA_CSR_TO_COO_CASE
}


template <typename output_t>
void dispatch_csr_to_coo_rows_cuda(
    Tensor& result, const Tensor& crow_indices, const Tensor& col_indices,
    bool transpose, const std::vector<int64_t>& batch_shape) {
#define TP_CUDA_CSR_TO_COO_ROW_CASE(ctype, name)                              \
    case DType::name:                                                          \
        dispatch_csr_to_coo_columns_cuda<ctype, output_t>(                     \
            result, crow_indices, col_indices, transpose, batch_shape);        \
        return;
    switch (crow_indices.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_CUDA_CSR_TO_COO_ROW_CASE)
        default:
            TP_THROW(TypeError,
                     "_convert_indices_from_csr_to_coo: row pointers must be integral");
    }
#undef TP_CUDA_CSR_TO_COO_ROW_CASE
}


std::vector<int64_t> check_csr_to_coo_cuda(const Tensor& crow_indices,
                                           const Tensor& col_indices) {
    TP_CHECK(crow_indices.dim() >= 1 && col_indices.dim() >= 1,
             "_convert_indices_from_csr_to_coo: inputs must have at least one dimension");
    TP_CHECK(crow_indices.dim() == col_indices.dim(),
             "_convert_indices_from_csr_to_coo: inputs must have the same dimensionality");
    TP_CHECK(crow_indices.size(-1) >= 1,
             "_convert_indices_from_csr_to_coo: row pointer dimension must be non-empty");
    TP_CHECK(crow_indices.dim() - 1 <= kSparseIndexMaxBatchDims,
             "_convert_indices_from_csr_to_coo: too many batch dimensions");
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        TP_CHECK(crow_indices.size(dim) == col_indices.size(dim),
                 "_convert_indices_from_csr_to_coo: batch dimensions must match");
    }
    TP_CHECK(isIntegralType(crow_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: row pointers must be integral");
    TP_CHECK(isIntegralType(col_indices.dtype(), false),
             "_convert_indices_from_csr_to_coo: columns must be integral");

    std::vector<int64_t> batch_shape;
    batch_shape.reserve(static_cast<size_t>(crow_indices.dim() - 1));
    int64_t batch_count = 1;
    for (int64_t dim = 0; dim < crow_indices.dim() - 1; ++dim) {
        const int64_t extent = crow_indices.size(dim);
        batch_shape.push_back(extent);
        batch_count *= extent;
    }
    const int64_t nrows = crow_indices.size(-1) - 1;
    const int64_t nnz = col_indices.size(-1);
    TP_CHECK(col_indices.numel() == batch_count * nnz,
             "_convert_indices_from_csr_to_coo: invalid batch layout");
    TP_CHECK(crow_indices.numel() == batch_count * (nrows + 1),
             "_convert_indices_from_csr_to_coo: invalid row pointer layout");
    return batch_shape;
}


Tensor convert_indices_from_coo_to_csr_cuda(const Tensor& input,
                                            int64_t size, bool out_int32) {
    check_coo_to_csr_cuda(input, size);
    Tensor result = ops::empty(
        {size + 1}, out_int32 ? DType::Int32 : DType::Int64, input.device());
    if (out_int32) {
        dispatch_coo_to_csr_input_cuda<int32_t>(result, input, size);
    } else {
        dispatch_coo_to_csr_input_cuda<int64_t>(result, input, size);
    }
    return result;
}


Tensor& _convert_indices_from_coo_to_csr_structured_cuda(
    const Tensor& input, int64_t size, bool out_int32, Tensor& out) {
    check_coo_to_csr_cuda(input, size);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_coo_to_csr: output dtype is incorrect");
    TP_CHECK(out.device() == input.device(),
             "_convert_indices_from_coo_to_csr: output device must match input");
    const std::vector<int64_t> shape = {size + 1};
    out.resize_(shape);
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back ? ops::empty(shape, dtype, input.device()) : out;
    if (out_int32) {
        dispatch_coo_to_csr_input_cuda<int32_t>(target, input, size);
    } else {
        dispatch_coo_to_csr_input_cuda<int64_t>(target, input, size);
    }
    if (copy_back) out.copy_(target);
    return out;
}


Tensor convert_indices_from_csr_to_coo_cuda(
    const Tensor& crow_indices, const Tensor& col_indices, bool out_int32,
    bool transpose) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cuda(crow_indices, col_indices);
    Tensor result = ops::empty(
        {col_indices.dim() + 1, col_indices.numel()},
        out_int32 ? DType::Int32 : DType::Int64, crow_indices.device());
    if (out_int32) {
        dispatch_csr_to_coo_rows_cuda<int32_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows_cuda<int64_t>(
            result, crow_indices, col_indices, transpose, batch_shape);
    }
    return result;
}


Tensor& _convert_indices_from_csr_to_coo_structured_cuda(
    const Tensor& crow_indices, const Tensor& col_indices, bool out_int32,
    bool transpose, Tensor& out) {
    const std::vector<int64_t> batch_shape =
        check_csr_to_coo_cuda(crow_indices, col_indices);
    const DType dtype = out_int32 ? DType::Int32 : DType::Int64;
    TP_CHECK(out.defined() && out.dtype() == dtype,
             "_convert_indices_from_csr_to_coo: output dtype is incorrect");
    TP_CHECK(out.device() == crow_indices.device(),
             "_convert_indices_from_csr_to_coo: output device must match input");
    const std::vector<int64_t> shape =
        {col_indices.dim() + 1, col_indices.numel()};
    out.resize_(shape);
    const bool copy_back = !out.is_contiguous();
    Tensor target = copy_back
        ? ops::empty(shape, dtype, crow_indices.device())
        : out;
    if (out_int32) {
        dispatch_csr_to_coo_rows_cuda<int32_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    } else {
        dispatch_csr_to_coo_rows_cuda<int64_t>(
            target, crow_indices, col_indices, transpose, batch_shape);
    }
    if (copy_back) out.copy_(target);
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SparseCsrInterop) {
    m.impl("_convert_indices_from_coo_to_csr", convert_indices_from_coo_to_csr_cuda);
    m.impl("_convert_indices_from_coo_to_csr.out", _convert_indices_from_coo_to_csr_structured_cuda);
    m.impl("_convert_indices_from_csr_to_coo", convert_indices_from_csr_to_coo_cuda);
    m.impl("_convert_indices_from_csr_to_coo.out", _convert_indices_from_csr_to_coo_structured_cuda);
}

} // namespace cuda
} // namespace tensorplay
