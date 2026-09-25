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


// ------------------- native COO coalesce infrastructure --------------------
//
// Lexicographic sort of the coordinates (successive stable radix passes from
// the last sparse dim to the first, carrying an element permutation) followed
// by run detection over sorted tuples and typed atomic folding of duplicate
// values.  No CPU staging anywhere; the only host synchronization is the
// output nnz readback.

__global__ void iota_kernel(int64_t n, int64_t* out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}


__global__ void select_index_rows_kernel(int64_t total, int64_t nnz,
                                         const int64_t* src,
                                         const int64_t* kept_rows,
                                         int64_t n_kept, int64_t* dst) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int64_t k = e / nnz;
    const int64_t n = e - k * nnz;
    dst[e] = src[kept_rows[k] * nnz + n];
}


__global__ void gather_i64_by_perm_kernel(int64_t n, const int64_t* src,
                                          const int64_t* perm, int64_t* dst) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}


// Byte-span copy keeps value gathers dtype-agnostic (works for bool, ints,
// floats and complex alike); no arithmetic here so alignment is per-byte.
__global__ void gather_bytes_by_perm_kernel(int64_t n, int64_t span,
                                            const unsigned char* src,
                                            const int64_t* perm,
                                            unsigned char* dst) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n * span) return;
    const int64_t row = i / span;
    dst[i] = src[perm[row] * span + (i - row * span)];
}


__global__ void coo_run_start_flags_kernel(int64_t nnz, int64_t sparse_dim,
                                           const int64_t* coords,
                                           bool* flags) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    if (i == 0) {
        flags[0] = true;
        return;
    }
    bool same = true;
    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (coords[d * nnz + i] != coords[d * nnz + i - 1]) {
            same = false;
            break;
        }
    }
    flags[i] = !same;
}


// Writes each unique coordinate row once, at its compacted slot.
__global__ void coo_write_unique_coords_kernel(int64_t nnz, int64_t sparse_dim,
                                               const int64_t* coords,
                                               const bool* flags,
                                               const int64_t* slots,
                                               int64_t* out_coords,
                                               int64_t out_nnz) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz || !flags[i]) return;
    const int64_t slot = slots[i];
    for (int64_t d = 0; d < sparse_dim; ++d) {
        out_coords[d * out_nnz + slot] = coords[d * nnz + i];
    }
}


// Compacts the sorted positions of run-start entries: run r begins at
// sorted index run_starts[r].
__global__ void coo_run_start_positions_kernel(int64_t nnz, const bool* flags,
                                               const int64_t* slots,
                                               int64_t* run_starts) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz || !flags[i]) return;
    run_starts[slots[i]] = i;
}


// Deterministic duplicate folding: one thread per run accumulates its
// contiguous segment in order.  Works for every dtype (including complex,
// which has no CUDA atomicAdd) and keeps summation order fixed.
template <typename scalar_t>
__global__ void coo_fold_runs_kernel(int64_t num_runs, int64_t span,
                                     const int64_t* run_starts,
                                     const int64_t* run_lengths,
                                     const scalar_t* sorted_values,
                                     scalar_t* out_values) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_runs) return;
    const int64_t begin = run_starts[r];
    const int64_t length = run_lengths[r];
    for (int64_t k = 0; k < length; ++k) {
        const scalar_t* src = sorted_values + (begin + k) * span;
        scalar_t* dst = out_values + r * span;
        for (int64_t c = 0; c < span; ++c) dst[c] += src[c];
    }
}


// One cub radix pass: sorts keys and carries the int64 payload.
void radix_sort_pairs_i64(const int64_t* keys_in, int64_t* keys_out,
                          const int64_t* vals_in, int64_t* vals_out,
                          int64_t n, const cudaStream_t stream,
                          const Tensor& device_for_alloc) {
    size_t temporary_bytes = 0;
    checkCuda(cub::DeviceRadixSort::SortPairs(
                  nullptr, temporary_bytes, keys_in, keys_out, vals_in,
                  vals_out, static_cast<int>(n)),
              "CUB coalesce radix sort size query");
    Tensor temporary = Tensor::empty(
        {static_cast<int64_t>(temporary_bytes == 0 ? 1 : temporary_bytes)},
        DType::UInt8, device_for_alloc.device());
    checkCuda(cub::DeviceRadixSort::SortPairs(
                  temporary.data_ptr(), temporary_bytes, keys_in, keys_out,
                  vals_in, vals_out, static_cast<int>(n), 0,
                  sizeof(int64_t) * 8, stream),
              "CUB coalesce radix sort");
}


__global__ void coo_run_lengths_kernel_from_starts(int64_t num_runs,
                                                   int64_t nnz,
                                                   const int64_t* run_starts,
                                                   int64_t* run_lengths) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_runs) return;
    const int64_t end =
        r + 1 < num_runs ? run_starts[r + 1] : nnz;
    run_lengths[r] = end - run_starts[r];
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


__global__ void sparse_mask_gather_bytes_kernel(
    int64_t output_numel,
    int64_t dense_numel,
    int64_t itemsize,
    const int64_t* indices,
    int64_t nnz,
    const uint8_t* dense,
    uint8_t* values,
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
    const uint8_t* source = dense + source_offset * itemsize;
    uint8_t* destination = values + linear * itemsize;
    for (int64_t byte = 0; byte < itemsize; ++byte) {
        destination[byte] = source[byte];
    }
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

} // namespace


Tensor coalesce_sparse_cuda(const Tensor& self) {
    if (!self.is_sparse() || self.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    if (self.is_coalesced()) return self;

    Tensor indices = self._indices().contiguous();
    Tensor values = self._values().contiguous();
    const int64_t nnz = indices.size(1);
    const int64_t sparse_dim = indices.size(0);
    const auto shape =
        static_cast<std::vector<int64_t>>(self.shape());
    if (nnz <= 1) {
        return Tensor::make_sparse_coo_tensor(indices, values, shape, true);
    }

    const cudaStream_t stream = getCurrentCUDAStream().stream();

    // Successive stable sorts from the last coordinate dim to the first:
    // radix sort is stable, so after the final pass `perm` orders the
    // entries lexicographically by coordinate.
    Tensor perm_a = Tensor::empty({nnz}, DType::Int64, self.device());
    Tensor perm_b = Tensor::empty({nnz}, DType::Int64, self.device());
    Tensor keys_a = Tensor::empty({nnz}, DType::Int64, self.device());
    Tensor keys_b = Tensor::empty({nnz}, DType::Int64, self.device());
    iota_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0, stream>>>(
        nnz, perm_a.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA coalesce iota kernel");
    for (int64_t d = sparse_dim - 1; d >= 0; --d) {
        gather_i64_by_perm_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                    stream>>>(
            nnz, indices.data_ptr<int64_t>() + d * nnz,
            perm_a.data_ptr<int64_t>(), keys_a.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA coalesce key gather kernel");
        radix_sort_pairs_i64(keys_a.data_ptr<int64_t>(),
                             keys_b.data_ptr<int64_t>(),
                             perm_a.data_ptr<int64_t>(),
                             perm_b.data_ptr<int64_t>(), nnz, stream, self);
        std::swap(perm_a, perm_b);
    }

    // Materialize the sorted coordinates and values through the permutation.
    Tensor sorted_indices = Tensor::empty(
        static_cast<std::vector<int64_t>>(indices.shape()), DType::Int64,
        self.device());
    for (int64_t d = 0; d < sparse_dim; ++d) {
        gather_i64_by_perm_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                    stream>>>(
            nnz, indices.data_ptr<int64_t>() + d * nnz,
            perm_a.data_ptr<int64_t>(),
            sorted_indices.data_ptr<int64_t>() + d * nnz);
        checkCuda(cudaGetLastError(), "CUDA coalesce coord gather kernel");
    }
    const int64_t span = values.numel() / std::max<int64_t>(nnz, 1);
    const int64_t row_bytes = span * static_cast<int64_t>(values.itemsize());
    Tensor sorted_values = Tensor::empty(
        static_cast<std::vector<int64_t>>(values.shape()), values.dtype(),
        self.device());
    if (row_bytes > 0) {
        gather_bytes_by_perm_kernel<<<
            coalesce_blocks(nnz * row_bytes), kCoalesceThreads, 0, stream>>>(
            nnz, row_bytes,
            reinterpret_cast<const unsigned char*>(values.data_ptr()),
            perm_a.data_ptr<int64_t>(),
            reinterpret_cast<unsigned char*>(sorted_values.data_ptr()));
        checkCuda(cudaGetLastError(), "CUDA coalesce value gather kernel");
    }

    // Run detection over sorted tuples -> compaction slots via exclusive
    // sum.  ExclusiveSum cannot consume bool*; stage flags as int64.
    Tensor flags = Tensor::empty({nnz}, DType::Bool, self.device());
    coo_run_start_flags_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                 stream>>>(
        nnz, sparse_dim, sorted_indices.data_ptr<int64_t>(),
        flags.data_ptr<bool>());
    checkCuda(cudaGetLastError(), "CUDA coalesce run-start kernel");
    Tensor flags_i64 = flags.to(DType::Int64);
    Tensor flag_sums = Tensor::zeros({nnz}, DType::Int64, self.device());
    size_t scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  nullptr, scan_bytes, flags_i64.data_ptr<int64_t>(),
                  flag_sums.data_ptr<int64_t>(), static_cast<int>(nnz), stream),
              "CUB coalesce exclusive-sum size query");
    Tensor scan_temporary = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  scan_temporary.data_ptr(), scan_bytes,
                  flags_i64.data_ptr<int64_t>(), flag_sums.data_ptr<int64_t>(),
                  static_cast<int>(nnz), stream),
              "CUB coalesce exclusive sum");

    // out_nnz = number of runs = slots[nnz-1] + flag[nnz-1]; async-read the
    // two tail elements into host memory, then sync once.
    int64_t slot_tail = 0;
    int64_t flag_tail = 0;
    checkCuda(cudaMemcpyAsync(&slot_tail,
                              flag_sums.data_ptr<int64_t>() + (nnz - 1),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA coalesce slot readback");
    checkCuda(cudaMemcpyAsync(&flag_tail,
                              flags_i64.data_ptr<int64_t>() + (nnz - 1),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA coalesce tail-flag readback");
    checkCuda(cudaStreamSynchronize(stream), "CUDA coalesce nnz sync");
    const int64_t out_nnz = slot_tail + flag_tail;

    Tensor out_indices = Tensor::empty({sparse_dim, out_nnz}, DType::Int64,
                                       self.device());
    coo_write_unique_coords_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                     stream>>>(
        nnz, sparse_dim, sorted_indices.data_ptr<int64_t>(),
        flags.data_ptr<bool>(), flag_sums.data_ptr<int64_t>(),
        out_indices.data_ptr<int64_t>(), out_nnz);
    checkCuda(cudaGetLastError(), "CUDA coalesce unique coords kernel");

    std::vector<int64_t> out_values_shape =
        static_cast<std::vector<int64_t>>(values.shape());
    if (!out_values_shape.empty()) out_values_shape[0] = out_nnz;
    Tensor out_values = Tensor::zeros(out_values_shape, values.dtype(),
                                      self.device());
    if (out_nnz > 0 && span > 0) {
        // Run boundaries in sorted order (run r covers
        // [run_starts[r], run_starts[r] + run_lengths[r])).
        Tensor run_starts = Tensor::empty({out_nnz}, DType::Int64,
                                          self.device());
        coo_run_start_positions_kernel<<<coalesce_blocks(nnz), kCoalesceThreads,
                                         0, stream>>>(
            nnz, flags.data_ptr<bool>(), flag_sums.data_ptr<int64_t>(),
            run_starts.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA coalesce run positions kernel");
        Tensor run_lengths = Tensor::empty({out_nnz}, DType::Int64,
                                           self.device());
        coo_run_lengths_kernel_from_starts<<<coalesce_blocks(out_nnz),
                                             kCoalesceThreads, 0, stream>>>(
            out_nnz, nnz, run_starts.data_ptr<int64_t>(),
            run_lengths.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA coalesce run lengths kernel");
        dispatch_coalesce_dtype(values.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            coo_fold_runs_kernel<scalar_t><<<coalesce_blocks(out_nnz),
                                             kCoalesceThreads, 0, stream>>>(
                out_nnz, span, run_starts.data_ptr<int64_t>(),
                run_lengths.data_ptr<int64_t>(),
                reinterpret_cast<const scalar_t*>(sorted_values.data_ptr()),
                reinterpret_cast<scalar_t*>(out_values.data_ptr()));
        });
        checkCuda(cudaGetLastError(), "CUDA coalesce fold kernel");
    }

    return Tensor::make_sparse_coo_tensor(out_indices, out_values, shape, true);
}


Tensor sparse_mask_cuda(const Tensor& dense, const Tensor& mask) {
    if (!mask.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse");
    }
    if (dense.device() != mask.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mask(): dense and mask must be on the same device");
    }
    if (dense.shape() != mask.shape()) {
        TP_THROW(RuntimeError,
                 "sparse_mask(): operands have incompatible sizes; self and mask must have the same shape");
    }
    // Preserve a COO mask's ordering and duplicate entries.  Compressed masks
    // are expanded once so the gather kernel has one coordinate representation.
    Tensor canonical_mask = mask.is_sparse_csr() ? to_sparse_coo_cuda(mask) : mask;
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
            canonical_mask.is_coalesced());
    }

    SparseGatherInfo info = make_gather_info(dense_contiguous, canonical_mask);
    const int threads = 256;
    const int blocks = static_cast<int>((output_numel + threads - 1) / threads);
    sparse_mask_gather_bytes_kernel<<<blocks, threads, 0,
                                      getCurrentCUDAStream().stream()>>>(
        output_numel, dense_numel, static_cast<int64_t>(dense.itemsize()),
        indices.data_ptr<int64_t>(), nnz,
        reinterpret_cast<const uint8_t*>(dense_contiguous.data_ptr()),
        reinterpret_cast<uint8_t*>(values.data_ptr()), info);
    checkCuda(cudaGetLastError(), "CUDA sparse_mask gather kernel");
    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(mask.shape()),
        canonical_mask.is_coalesced());
}


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
    if (self.is_sparse_csr()) return csr_to_coo_cuda(self).coalesce();
    if (self.is_sparse()) return self.coalesce();
    if (self.dim() == 0) {
        TP_THROW(RuntimeError,
                 "to_sparse(): a 0-dim tensor cannot be made sparse");
    }
    return to_sparse_coo_native(self);
}


Tensor to_sparse_coo_cuda_sparse_dim(const Tensor& self, int64_t sparse_dim) {
    if (self.is_sparse_csr()) {
        if (sparse_dim != 2) {
            TP_THROW(ValueError,
                     "to_sparse(): compressed input requires sparse_dim=2");
        }
        return csr_to_coo_cuda(self).coalesce();
    }
    if (self.is_sparse()) {
        if (sparse_dim != self.sparse_dim()) {
            TP_THROW(ValueError,
                     "to_sparse(): sparse_dim must match the sparse input");
        }
        return self.coalesce();
    }
    return to_sparse_coo_native_sparse_dim(self, sparse_dim);
}


Tensor sparse_sum_cuda(const Tensor& self,
                       const std::optional<std::vector<int64_t>>& dim,
                       std::optional<DType> dtype) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_sum(): expected a sparse tensor");
    }
    Tensor input = self;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != self.dtype()) {
        input = self.to(*dtype);
    }
    const bool reduce_all = !dim.has_value() || dim->empty();
    Tensor canonical;
    if (input.is_sparse_csr()) {
        canonical = reduce_all ? input : csr_to_coo_cuda(input).coalesce();
    } else {
        canonical = input.is_coalesced() ? input : input.coalesce();
    }
    if (canonical._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse_sum(): hybrid sparse tensors are not supported");
    }

    // coordinate rows on-device, rebuild an uncoalesced COO over the kept
    // dims and fold duplicates through the native coalesce.
    if (!reduce_all) {
        const int64_t sparse_dim = canonical.sparse_dim();
        std::vector<bool> reduced(static_cast<size_t>(sparse_dim), false);
        for (int64_t d : *dim) {
            if (d < 0) d += canonical.dim();
            if (d < 0 || d >= sparse_dim) {
                TP_THROW(ValueError, "sparse_sum(): dim out of the sparse range");
            }
            reduced[static_cast<size_t>(d)] = true;
        }
        int64_t num_reduced = 0;
        for (bool r : reduced) num_reduced += r ? 1 : 0;
        if (num_reduced == sparse_dim) {
            return canonical._values().sum();
        }

        std::vector<int64_t> kept_dims;
        for (int64_t d = 0; d < sparse_dim; ++d) {
            if (!reduced[static_cast<size_t>(d)]) kept_dims.push_back(d);
        }
        const auto sizes =
            static_cast<std::vector<int64_t>>(canonical.shape());
        std::vector<int64_t> out_sizes;
        for (int64_t d : kept_dims) {
            out_sizes.push_back(sizes[static_cast<size_t>(d)]);
        }

        Tensor indices = canonical._indices().contiguous();
        Tensor values = canonical._values().contiguous().clone();
        const int64_t nnz = indices.size(1);
        Tensor kept_dev = Tensor::zeros(
            {static_cast<int64_t>(kept_dims.size())}, DType::Int64,
            indices.device());
        checkCuda(cudaMemcpy(kept_dev.data_ptr<int64_t>(), kept_dims.data(),
                             kept_dims.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "CUDA sparse_sum kept-dims upload");
        Tensor new_indices = Tensor::empty(
            {static_cast<int64_t>(kept_dims.size()), nnz}, DType::Int64,
            indices.device());
        const int64_t total = static_cast<int64_t>(kept_dims.size()) * nnz;
        const cudaStream_t sum_stream = getCurrentCUDAStream().stream();
        if (total > 0) {
            select_index_rows_kernel<<<coalesce_blocks(total), kCoalesceThreads,
                                       0, sum_stream>>>(
                total, nnz, indices.data_ptr<int64_t>(),
                kept_dev.data_ptr<int64_t>(),
                static_cast<int64_t>(kept_dims.size()),
                new_indices.data_ptr<int64_t>());
            checkCuda(cudaGetLastError(), "CUDA sparse_sum row select kernel");
        }
        return Tensor::make_sparse_coo_tensor(new_indices, values, out_sizes,
                                              /*is_coalesced=*/false)
            .coalesce();
    }

    Tensor values = canonical._values().contiguous();
    const int64_t numel = values.numel();

#define TP_SPARSE_SUM_CASE(ctype, name)                                       \
    case DType::name: {                                                       \
        Tensor out = Tensor::zeros({}, values.dtype(), self.device());          \
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
    switch (values.dtype()) {
        TP_SPARSE_SUM_CASE(float, Float32)
        TP_SPARSE_SUM_CASE(double, Float64)
        default:
            // Non-float dtypes go through the native dense reduction.
            return canonical._values().sum();
    }
#undef TP_SPARSE_SUM_CASE
}

} // namespace cuda
} // namespace tensorplay
