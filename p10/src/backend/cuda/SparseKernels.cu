#include "SparseKernels.h"
#include "CUDARuntime.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <climits>
#include <unordered_set>
#include <utility>

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

namespace {

constexpr int kCoalesceThreads = 128;

int coalesce_blocks(int64_t n) {
    return static_cast<int>((n + kCoalesceThreads - 1) / kCoalesceThreads);
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

template <typename T>
struct CoalesceTypeTag { using type = T; };

template <typename F>
void dispatch_coalesce_dtype(DType dtype, F&& f) {
#define TP_COALESCE_DISPATCH(ctype, name) \
    case DType::name: f(CoalesceTypeTag<ctype>{}); return;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_COALESCE_DISPATCH)
        default:
            TP_THROW(NotImplementedError, "unsupported dtype in CUDA coalesce");
    }
#undef TP_COALESCE_DISPATCH
}

// ---- dense -> sparse (native nonzero extraction) ---------------------------
//
// Zero detection at the byte level: every numeric encoding in use (IEEE
// floats incl. +/-0, two's-complement ints, bool, complex pairs) is all-zero
// bytes exactly when the value equals zero, and NaN carries nonzero bytes
// like torch's nonzero semantics require.

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

__global__ void csr_count_rows_kernel(int64_t nnz, const int64_t* row_coords,
                                      int64_t* counts) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nnz) atomicAdd(reinterpret_cast<unsigned long long*>(
                               counts + row_coords[i]),
                           1ull);
}

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

} // namespace

Tensor sparse_coo_tensor_cuda(const Tensor& indices, const Tensor& values,
                              std::optional<std::vector<int64_t>> size,
                              bool is_coalesced) {
    if (size.has_value()) {
        return Tensor::make_sparse_coo_tensor(indices, values, *size, is_coalesced);
    }
    // Size inference (ATen SparseTensorConstructor.cpp): each sparse dim is
    // max(coord)+1; trailing dense dims come from the values' shape.
    if (!indices.is_contiguous()) {
        TP_THROW(RuntimeError,
                 "sparse_coo_tensor(): indices must be contiguous");
    }
    const int64_t sparse_dim = indices.size(0);
    const int64_t nnz = indices.size(1);
    Tensor maxima = Tensor::zeros({sparse_dim}, DType::Int64, indices.device());
    if (nnz > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        coord_max_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                           stream>>>(
            nnz, sparse_dim, indices.data_ptr<int64_t>(),
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
    return Tensor::make_sparse_coo_tensor(indices, values, inferred,
                                          is_coalesced);
}

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
    // Distinct coordinates may still share a row, so several threads can
    // target the same output cell; accumulate atomically.
    atomicAdd(&out[row_indices[n] * cols + j],
              values[n] * dense[col_indices[n] * cols + j]);
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

namespace {

// Shared native dense->COO extraction: byte-level nonzero mask + CUB
// Flagged compaction over a counting iterator + coordinate gather.
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

} // namespace

Tensor to_sparse_coo_cuda(const Tensor& self) {
    if (self.is_sparse()) return self.coalesce();
    if (self.dim() == 0) {
        TP_THROW(RuntimeError,
                 "to_sparse(): a 0-dim tensor cannot be made sparse");
    }
    return to_sparse_coo_native(self);
}

Tensor to_sparse_csr_cuda(const Tensor& self) {
    if (self.is_sparse()) {
        if (self.is_sparse_csr()) return self.coalesce();
        return coo_to_csr_native(self.coalesce());
    }
    if (self.dim() != 2) {
        TP_THROW(RuntimeError,
                 "to_sparse_csr(): only 2-D input is supported, got " +
                     std::to_string(self.dim()) + "-D");
    }
    return coo_to_csr_native(to_sparse_coo_native(self));
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

Tensor sparse_sum_cuda(const Tensor& self,
                       std::optional<std::vector<int64_t>> dim,
                       std::optional<DType> dtype) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_sum(): expected a sparse tensor");
    }
    Tensor input = self;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != self.dtype()) {
        input = self.to(*dtype);
    }
    Tensor canonical = input.is_coalesced() ? input : input.coalesce();
    const bool reduce_all = !dim.has_value() || dim->empty();

    // Partial reduction (ATen _sparse_sum semantics): keep the surviving
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

// Coordinate-union addition: concatenate both COO component sets on-device
// and fold duplicates through the native coalesce (ATen add semantics for
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
    Tensor a = self.is_coalesced() ? self : self.coalesce();
    Tensor b = other.is_coalesced() ? other : other.coalesce();
    if (a._values().dim() != 1 || b._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse.add(): hybrid COO tensors are not supported");
    }
    Tensor cat_indices = Tensor::cat({a._indices(), b._indices()}, 1);
    Tensor cat_values = Tensor::cat({a._values(), b._values()}, 0);
    return Tensor::make_sparse_coo_tensor(
        cat_indices, cat_values,
        static_cast<std::vector<int64_t>>(a.shape()),
        /*is_coalesced=*/false).coalesce();
}

namespace {

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

Tensor spdiags_cuda(const Tensor& diagonals, const Tensor& offsets,
                    std::vector<int64_t> shape,
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

    // Offsets and the derived per-diagonal slots are tiny; compute them on
    // the host and upload.
    Tensor offs_host_t = offs.to(Device(DeviceType::CPU)).contiguous();
    const int64_t* off_data = offs_host_t.data_ptr<int64_t>();
    std::vector<int64_t> off_host(off_data, off_data + n_diag);
    std::unordered_set<int64_t> unique_offs(off_host.begin(), off_host.end());
    if (unique_offs.size() != static_cast<size_t>(n_diag)) {
        TP_THROW(ValueError, "spdiags(): offset tensor contains duplicate values");
    }
    const int64_t m_size = shape[0];
    const int64_t n_size = shape[1];
    const int64_t length = diags2d.size(1);
    std::vector<int64_t> counts(static_cast<size_t>(n_diag), 0);
    std::vector<int64_t> starts(static_cast<size_t>(n_diag), 0);
    int64_t total_nnz = 0;
    for (int64_t j = 0; j < n_diag; ++j) {
        const int64_t d = off_host[static_cast<size_t>(j)];
        const int64_t count = d <= 0 ? std::min(d + m_size, length)
                                     : std::min(n_size, length) - d;
        counts[static_cast<size_t>(j)] = std::max<int64_t>(count, 0);
        starts[static_cast<size_t>(j)] = total_nnz;
        total_nnz += counts[static_cast<size_t>(j)];
    }

    Tensor diags_c = diags2d.contiguous();
    Tensor indices = Tensor::empty({2, total_nnz}, DType::Int64,
                                   offsets.device());
    Tensor values = Tensor::empty({total_nnz}, diags_c.dtype(),
                                  diags_c.device());
    auto meta_upload = [&total_nnz, &offsets](
                           const std::vector<int64_t>& host,
                           Tensor& device_tensor) {
        device_tensor = Tensor::zeros(
            {static_cast<int64_t>(host.size())}, DType::Int64, offsets.device());
        checkCuda(cudaMemcpy(device_tensor.data_ptr<int64_t>(), host.data(),
                             host.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  "spdiags meta upload");
    };
    Tensor offs_d, starts_d, counts_d;
    meta_upload(off_host, offs_d);
    meta_upload(starts, starts_d);
    meta_upload(counts, counts_d);

    if (total_nnz > 0 && n_diag > 0) {
        const size_t elem = diags_c.itemsize();
        const cudaStream_t fill_stream = getCurrentCUDAStream().stream();
        spdiags_fill_kernel<<<n_diag, 128, 0, fill_stream>>>(
            reinterpret_cast<const unsigned char*>(diags_c.data_ptr()),
            length, offs_d.data_ptr<int64_t>(), starts_d.data_ptr<int64_t>(),
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
        return coo_to_csr_native(result.coalesce());
    }
    return result;
}

} // namespace cuda
} // namespace tensorplay
