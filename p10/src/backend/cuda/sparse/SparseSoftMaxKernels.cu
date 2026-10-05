// Sparse COO softmax / log_softmax and their backward data kernels on CUDA.
//
// The reduction runs over independent "pools" of stored entries: entries
// whose coordinates agree outside the softmax dim share one softmax
// computation.  Unspecified entries contribute nothing to the exponent sums
// (they behave as negative infinities), so the output keeps the input's
// coordinates and values count.  When the softmax dim lies in the dense part
// the values payload reduces with the dense kernels directly.
//
// Pool grouping happens on device: every stored entry gets a flattened pool
// key (its coordinates contracted with the row strides, the softmax dim
// collapsed to zero), a radix sort of (key, entry) pairs makes the entries
// of one pool contiguous, and run detection plus an exclusive scan turn the
// sorted keys into per-pool start positions and sizes.  The pool count is
// the only host readback.
//
// The backward routes the output gradient back per pool:
//   softmax:     gI_i = out_i * (g_i - sum_j out_j * g_j)
//   log_softmax: gI_i = g_i - exp(out_i) * sum_j g_j
// with the pool sums accumulated over matching grad coordinates (matched by
// flattened-coordinate binary search).

#include "SparseKernels.cuh"
#include "SparseKernels.h"
#include "CUDARuntime.h"

#include "GPUPrimitives.cuh"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

__global__ void iota_kernel(int64_t n, int64_t* out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

// Flattened pool key per stored entry: the entry's coordinates contracted
// with the row strides, with coordinate `dim` collapsed to zero (pass
// dim = -1 to keep every coordinate).  Entries with equal keys share one
// softmax computation.
__global__ void pool_keys_kernel(int64_t nnz, int64_t ndim,
                                 const int64_t* indices,
                                 const int64_t* strides, int64_t dim,
                                 int64_t* keys) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    int64_t acc = 0;
    for (int64_t j = 0; j < ndim; ++j) {
        if (j == dim) continue;
        acc += strides[j] * indices[j * nnz + e];
    }
    keys[e] = acc;
}

// Run-start flags over the sorted keys.
__global__ void pool_run_flags_kernel(int64_t nnz, const int64_t* sorted_keys,
                                      bool* flags) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    if (i == 0) {
        flags[0] = true;
        return;
    }
    flags[i] = sorted_keys[i] != sorted_keys[i - 1];
}

// Compacts the sorted positions of run-start entries: pool r begins at
// sorted position starts[r].
__global__ void pool_run_positions_kernel(int64_t nnz, const bool* flags,
                                          const int64_t* slots,
                                          int64_t* starts) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nnz || !flags[i]) return;
    starts[slots[i]] = i;
}

__global__ void pool_run_lengths_kernel(int64_t num_pools, int64_t nnz,
                                        const int64_t* starts,
                                        int64_t* lengths) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= num_pools) return;
    const int64_t end = r + 1 < num_pools ? starts[r + 1] : nnz;
    lengths[r] = end - starts[r];
}

// First grad entry (in the grad coordinate order) whose flattened offset is
// not smaller than the output entry's; exact match decides whether the grad
// carries data for that output entry.
__global__ void grad_lower_bound_kernel(int64_t out_nnz,
                                        const int64_t* grad_offsets,
                                        int64_t grad_nnz,
                                        const int64_t* out_offsets,
                                        int64_t* lb) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out_nnz) return;
    const int64_t target = out_offsets[i];
    int64_t lo = 0;
    int64_t hi = grad_nnz;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (grad_offsets[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lb[i] = lo;
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
              "CUB sparse softmax radix sort size query");
    Tensor temporary = Tensor::empty(
        {static_cast<int64_t>(temporary_bytes == 0 ? 1 : temporary_bytes)},
        DType::UInt8, device_for_alloc.device());
    checkCuda(cub::DeviceRadixSort::SortPairs(
                  temporary.data_ptr(), temporary_bytes, keys_in, keys_out,
                  vals_in, vals_out, static_cast<int>(n), 0,
                  sizeof(int64_t) * 8, stream),
              "CUB sparse softmax radix sort");
}

// Groups stored entries by pool on device.  Returns (entry ids sorted so
// that each pool is a contiguous run, per-pool start positions, per-pool
// sizes, pool count).  The pool count is read back to the host, which is
// the only synchronization here.
std::tuple<Tensor, Tensor, Tensor, int64_t> group_pools(
    const Tensor& indices, const std::vector<int64_t>& sizes, int64_t dim) {
    const int64_t ndim = indices.size(0);
    const int64_t nnz = indices.size(1);
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const Device device = indices.device();

    Tensor empty_i64 = Tensor::empty({0}, DType::Int64, device);
    if (nnz == 0) {
        return std::make_tuple(empty_i64, empty_i64, empty_i64,
                               static_cast<int64_t>(0));
    }

    // Row strides of the coordinate matrix with the softmax dim collapsed.
    std::vector<int64_t> host_strides(static_cast<size_t>(ndim), 1);
    for (int64_t i = ndim - 2; i >= 0; --i) {
        host_strides[static_cast<size_t>(i)] =
            host_strides[static_cast<size_t>(i + 1)] *
            (i + 1 == dim ? 1 : sizes[static_cast<size_t>(i + 1)]);
    }
    Tensor strides = Tensor::zeros({ndim}, DType::Int64, device);
    checkCuda(cudaMemcpyAsync(strides.data_ptr<int64_t>(), host_strides.data(),
                              host_strides.size() * sizeof(int64_t),
                              cudaMemcpyHostToDevice, stream),
              "CUDA sparse softmax stride upload");

    Tensor keys = Tensor::empty({nnz}, DType::Int64, device);
    Tensor keys_sorted = Tensor::empty({nnz}, DType::Int64, device);
    Tensor perm = Tensor::empty({nnz}, DType::Int64, device);
    Tensor perm_sorted = Tensor::empty({nnz}, DType::Int64, device);
    pool_keys_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0, stream>>>(
        nnz, ndim, indices.data_ptr<int64_t>(), strides.data_ptr<int64_t>(),
        dim, keys.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax pool-key kernel");
    iota_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0, stream>>>(
        nnz, perm.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax iota kernel");
    radix_sort_pairs_i64(keys.data_ptr<int64_t>(), keys_sorted.data_ptr<int64_t>(),
                         perm.data_ptr<int64_t>(), perm_sorted.data_ptr<int64_t>(),
                         nnz, stream, indices);

    Tensor flags = Tensor::empty({nnz}, DType::Bool, device);
    pool_run_flags_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0, stream>>>(
        nnz, keys_sorted.data_ptr<int64_t>(), flags.data_ptr<bool>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax run-flag kernel");

    Tensor slots = Tensor::zeros({nnz}, DType::Int64, device);
    // Named staging tensors: ExclusiveSum is stream-ordered, so the input
    // and workspace must outlive the scan (the caching allocator would
    // happily reuse a temporary).
    Tensor flags_i64 = flags.to(DType::Int64);
    size_t scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  nullptr, scan_bytes, flags_i64.data_ptr<int64_t>(),
                  slots.data_ptr<int64_t>(), static_cast<int>(nnz), stream),
              "CUB sparse softmax exclusive-sum size query");
    Tensor scan_temporary = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)}, DType::UInt8,
        device);
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  scan_temporary.data_ptr(), scan_bytes,
                  flags_i64.data_ptr<int64_t>(),
                  slots.data_ptr<int64_t>(), static_cast<int>(nnz), stream),
              "CUB sparse softmax exclusive sum");

    Tensor starts = Tensor::empty({nnz}, DType::Int64, device);
    pool_run_positions_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0,
                                stream>>>(
        nnz, flags.data_ptr<bool>(), slots.data_ptr<int64_t>(),
        starts.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax run-position kernel");

    // pool count = slots[nnz-1] + flag[nnz-1]; async readback, one sync.
    int64_t slot_tail = 0;
    int64_t flag_tail = 0;
    checkCuda(cudaMemcpyAsync(&slot_tail,
                              slots.data_ptr<int64_t>() + (nnz - 1),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse softmax slot readback");
    checkCuda(cudaMemcpyAsync(&flag_tail, flags_i64.data_ptr<int64_t>() +
                                                  (nnz - 1),
                              sizeof(int64_t), cudaMemcpyDeviceToHost, stream),
              "CUDA sparse softmax tail-flag readback");
    checkCuda(cudaStreamSynchronize(stream), "CUDA sparse softmax pool sync");
    const int64_t pool_count = slot_tail + flag_tail;

    Tensor lengths = Tensor::empty({nnz}, DType::Int64, device);
    if (pool_count > 0) {
        pool_run_lengths_kernel<<<coalesce_blocks(pool_count), kCoalesceThreads,
                                  0, stream>>>(
            pool_count, nnz, starts.data_ptr<int64_t>(),
            lengths.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA sparse softmax run-length kernel");
    }
    return std::make_tuple(perm_sorted, starts, lengths, pool_count);
}

// Flattened coordinates of the stored entries, keeping every coordinate
// (dim = -1).  Used to align grad entries with output entries.
Tensor flattened_offsets(const Tensor& indices,
                         const std::vector<int64_t>& sizes) {
    const int64_t ndim = indices.size(0);
    const int64_t nnz = indices.size(1);
    const cudaStream_t stream = getCurrentCUDAStream().stream();

    std::vector<int64_t> host_strides(static_cast<size_t>(ndim), 1);
    for (int64_t i = ndim - 2; i >= 0; --i) {
        host_strides[static_cast<size_t>(i)] =
            host_strides[static_cast<size_t>(i + 1)] *
            sizes[static_cast<size_t>(i + 1)];
    }
    Tensor strides = Tensor::zeros({ndim}, DType::Int64, indices.device());
    checkCuda(cudaMemcpyAsync(strides.data_ptr<int64_t>(), host_strides.data(),
                              host_strides.size() * sizeof(int64_t),
                              cudaMemcpyHostToDevice, stream),
              "CUDA sparse softmax stride upload");

    Tensor offsets = Tensor::zeros({nnz}, DType::Int64, indices.device());
    if (nnz > 0) {
        pool_keys_kernel<<<coalesce_blocks(nnz), kCoalesceThreads, 0, stream>>>(
            nnz, ndim, indices.data_ptr<int64_t>(),
            strides.data_ptr<int64_t>(), -1, offsets.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA sparse softmax offset kernel");
    }
    return offsets;
}

// Forward: one thread per (pool, dense-payload column).  The per-pool
// maximum stays in registers; exp sums and the normalization walk the pool
// entries in sorted order.
template <typename scalar_t, bool LogSoftMax>
__global__ void sparse_softmax_pools_kernel(int64_t total, int64_t nvalues,
                                            const int64_t* pool_entries,
                                            const int64_t* pool_starts,
                                            const int64_t* pool_lengths,
                                            const scalar_t* values,
                                            scalar_t* out) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int64_t p = t / nvalues;
    const int64_t j = t - p * nvalues;
    const int64_t begin = pool_starts[p];
    const int64_t end = begin + pool_lengths[p];

    scalar_t mx = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t s = begin; s < end; ++s) {
        const int64_t i = pool_entries[s];
        const scalar_t v = values[i * nvalues + j];
        if (v > mx) mx = v;
    }
    scalar_t sum = 0;
    for (int64_t s = begin; s < end; ++s) {
        const int64_t i = pool_entries[s];
        const scalar_t v = std::exp(values[i * nvalues + j] - mx);
        if (!LogSoftMax) out[i * nvalues + j] = v;
        sum += v;
    }
    const scalar_t log_sum = std::log(sum);
    const scalar_t inv_sum = scalar_t(1) / sum;
    for (int64_t s = begin; s < end; ++s) {
        const int64_t i = pool_entries[s];
        scalar_t* out_at = out + i * nvalues + j;
        if (LogSoftMax) {
            *out_at = values[i * nvalues + j] - mx - log_sum;
        } else {
            *out_at *= inv_sum;
        }
    }
}

// Backward: one thread per (pool, dense-payload column).  The per-column
// pool sum tmp = -sum_j out_j * g_j (softmax) or -sum_j g_j (log softmax)
// is accumulated in registers, then every entry's gradient is written from
// it, matched against the grad by flattened coordinate.
template <typename scalar_t, bool LogSoftMax>
__global__ void sparse_softmax_backward_pools_kernel(
    int64_t total, int64_t nvalues, const int64_t* pool_entries,
    const int64_t* pool_starts, const int64_t* pool_lengths,
    const int64_t* lb, const int64_t* out_offsets, int64_t grad_nnz,
    const int64_t* grad_offsets, const scalar_t* out_values,
    const scalar_t* grad_values, scalar_t* grad_in) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int64_t p = t / nvalues;
    const int64_t k = t - p * nvalues;
    const int64_t begin = pool_starts[p];
    const int64_t end = begin + pool_lengths[p];

    scalar_t tmp = 0;
    for (int64_t s = begin; s < end; ++s) {
        const int64_t i = pool_entries[s];
        const int64_t j = lb[i];
        if (j < grad_nnz && out_offsets[i] == grad_offsets[j]) {
            if (LogSoftMax) {
                tmp -= grad_values[j * nvalues + k];
            } else {
                tmp -= out_values[i * nvalues + k] * grad_values[j * nvalues + k];
            }
        }
    }
    for (int64_t s = begin; s < end; ++s) {
        const int64_t i = pool_entries[s];
        const int64_t j = lb[i];
        scalar_t* grad_at = grad_in + i * nvalues + k;
        if (j < grad_nnz && out_offsets[i] == grad_offsets[j]) {
            if (LogSoftMax) {
                *grad_at = grad_values[j * nvalues + k] +
                           std::exp(out_values[i * nvalues + k]) * tmp;
            } else {
                *grad_at = out_values[i * nvalues + k] *
                           (grad_values[j * nvalues + k] + tmp);
            }
        } else {
            if (LogSoftMax) {
                *grad_at = std::exp(out_values[i * nvalues + k]) * tmp;
            } else {
                *grad_at = out_values[i * nvalues + k] * tmp;
            }
        }
    }
}

template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_float_dtype(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Float32: f(TypeTag<float>{}); return;
        case DType::Float64: f(TypeTag<double>{}); return;
        default:
            TP_THROW(NotImplementedError,
                     "sparse softmax: unsupported dtype");
    }
}

// Shared preprocessing: coalesce, build the output shell with the same
// coordinates, wrap the dim.
std::tuple<Tensor, Tensor, int64_t> softmax_sparse_preprocessing(
    const Tensor& input_, int64_t dim_, const char* fn_name) {
    TP_CHECK(input_.is_sparse() && !input_.is_sparse_compressed(),
             fn_name, ": expected a sparse COO tensor");
    Tensor input = input_.is_coalesced() ? input_ : input_.coalesce();
    Tensor indices = input._indices().contiguous();
    Tensor output = Tensor::make_sparse_coo_tensor(
        indices,
        Tensor::empty(static_cast<std::vector<int64_t>>(input._values().shape()),
                      input.dtype(), input.device()),
        static_cast<std::vector<int64_t>>(input.shape()), true);
    const int64_t ndim = input.dim();
    TP_CHECK(dim_ >= -ndim && dim_ < ndim,
             "Dimension out of range (expected to be in range of [", -ndim,
             ", ", ndim - 1, "], but got ", dim_, ")");
    const int64_t dim = dim_ < 0 ? dim_ + ndim : dim_;
    return {input, output, dim};
}

Tensor ops_softmax(const Tensor& values, int64_t dim) {
    return tensorplay::tpx::ops::_softmax(values, dim, false);
}
Tensor ops_log_softmax(const Tensor& values, int64_t dim) {
    return tensorplay::tpx::ops::_log_softmax(values, dim, false);
}
Tensor ops_softmax_backward(const Tensor& grad, const Tensor& output,
                            int64_t dim) {
    return tensorplay::tpx::ops::_softmax_backward_data(
        grad, output, dim, output.dtype());
}
Tensor ops_log_softmax_backward(const Tensor& grad, const Tensor& output,
                                int64_t dim) {
    return tensorplay::tpx::ops::_log_softmax_backward_data(
        grad, output, dim, output.dtype());
}

template <typename scalar_t, bool LogSoftMax>
void sparse_coo_softmax(Tensor& output, const Tensor& input, int64_t dim) {
    const int64_t sparse_dim = input.sparse_dim();
    Tensor indices = input._indices().contiguous();
    Tensor values = input._values().contiguous();
    Tensor out_indices = output._indices();
    Tensor out_values = output._values();
    out_indices.copy_(indices);

    auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    const int64_t nnz = values.size(0);

    if (dim >= sparse_dim) {
        // The softmax dim is inside the dense payload: reduce the values
        // with the dense kernels along the payload-relative dim.  The dense
        // result replaces the shell's values storage outright; copying
        // elementwise would double the memory traffic of the whole op.
        const int64_t values_dim = dim - sparse_dim + 1;
        Tensor new_values = LogSoftMax
            ? ops_log_softmax(values, values_dim)
            : ops_softmax(values, values_dim);
        output = Tensor::make_sparse_coo_tensor(out_indices, new_values, sizes,
                                                true);
        return;
    }

    const int64_t nvalues = [&] {
        int64_t acc = 1;
        for (int64_t d = sparse_dim; d < static_cast<int64_t>(sizes.size()); ++d) {
            acc *= sizes[static_cast<size_t>(d)];
        }
        return acc;
    }();

    auto [pool_entries, pool_starts, pool_lengths, pool_count] =
        group_pools(indices, sizes, dim);
    const int64_t total = pool_count * nvalues;
    if (total == 0) return;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    sparse_softmax_pools_kernel<scalar_t, LogSoftMax>
        <<<coalesce_blocks(total), kCoalesceThreads, 0, stream>>>(
            total, nvalues, pool_entries.data_ptr<int64_t>(),
            pool_starts.data_ptr<int64_t>(),
            pool_lengths.data_ptr<int64_t>(), values.data_ptr<scalar_t>(),
            out_values.data_ptr<scalar_t>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax kernel");
}

template <typename scalar_t, bool LogSoftMax>
void sparse_coo_softmax_backward(Tensor& grad_input, const Tensor& grad,
                                 const Tensor& output, int64_t dim) {
    const int64_t sparse_dim = output.sparse_dim();
    auto sizes = static_cast<std::vector<int64_t>>(output.shape());
    Tensor grad_indices = grad._indices().contiguous();
    Tensor grad_values = grad._values().contiguous();
    Tensor out_indices = output._indices().contiguous();
    Tensor out_values = output._values().contiguous();
    Tensor values = grad_input._values();
    Tensor indices = grad_input._indices();
    const int64_t out_nnz = out_values.size(0);
    const int64_t grad_nnz = grad_values.size(0);

    values.resize_as_(out_values).zero_();
    indices.copy_(out_indices);

    Tensor out_offsets = flattened_offsets(out_indices, sizes);
    Tensor grad_offsets = flattened_offsets(grad_indices, sizes);

    if (dim >= sparse_dim) {
        // Dense payload case: grad and output must share coordinates.
        bool offsets_match = out_nnz == grad_nnz &&
                             tensorplay::tpx::ops::equal(out_offsets,
                                                         grad_offsets);
        if (offsets_match) {
            Tensor r = LogSoftMax
                ? ops_log_softmax_backward(grad_values, out_values,
                                           dim - sparse_dim + 1)
                : ops_softmax_backward(grad_values, out_values,
                                       dim - sparse_dim + 1);
            grad_input = Tensor::make_sparse_coo_tensor(indices, r, sizes,
                                                        true);
        } else {
            // Coordinates differ per entry: fall back to one dense backward
            // per matched output row, looked up by flattened coordinate.
            std::vector<int64_t> host_out_offsets(
                static_cast<size_t>(out_nnz), 0);
            std::vector<int64_t> host_grad_offsets(
                static_cast<size_t>(grad_nnz), 0);
            const cudaStream_t stream = getCurrentCUDAStream().stream();
            if (out_nnz > 0) {
                checkCuda(cudaMemcpyAsync(
                              host_out_offsets.data(),
                              out_offsets.data_ptr<int64_t>(),
                              static_cast<size_t>(out_nnz) * sizeof(int64_t),
                              cudaMemcpyDeviceToHost, stream),
                          "CUDA sparse softmax out-offset download");
            }
            if (grad_nnz > 0) {
                checkCuda(cudaMemcpyAsync(
                              host_grad_offsets.data(),
                              grad_offsets.data_ptr<int64_t>(),
                              static_cast<size_t>(grad_nnz) * sizeof(int64_t),
                              cudaMemcpyDeviceToHost, stream),
                          "CUDA sparse softmax grad-offset download");
            }
            checkCuda(cudaStreamSynchronize(stream),
                      "CUDA sparse softmax offset sync");
            for (int64_t i = 0; i < out_nnz; ++i) {
                auto low = std::lower_bound(host_grad_offsets.begin(),
                                            host_grad_offsets.end(),
                                            host_out_offsets[static_cast<size_t>(i)]);
                const int64_t j = low - host_grad_offsets.begin();
                if (j < grad_nnz &&
                    host_out_offsets[static_cast<size_t>(i)] ==
                        host_grad_offsets[static_cast<size_t>(j)]) {
                    Tensor r = LogSoftMax
                        ? ops_log_softmax_backward(
                            grad_values.select(0, j),
                            out_values.select(0, i), dim - sparse_dim)
                        : ops_softmax_backward(
                            grad_values.select(0, j),
                            out_values.select(0, i), dim - sparse_dim);
                    values.select(0, i).copy_(r);
                }
            }
        }
        return;
    }

    const int64_t nvalues = [&] {
        int64_t acc = 1;
        for (int64_t d = sparse_dim; d < static_cast<int64_t>(sizes.size()); ++d) {
            acc *= sizes[static_cast<size_t>(d)];
        }
        return acc;
    }();

    Tensor lb = Tensor::zeros({out_nnz}, DType::Int64, out_offsets.device());
    if (out_nnz > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        grad_lower_bound_kernel<<<coalesce_blocks(out_nnz), kCoalesceThreads,
                                  0, stream>>>(
            out_nnz, grad_offsets.data_ptr<int64_t>(), grad_nnz,
            out_offsets.data_ptr<int64_t>(), lb.data_ptr<int64_t>());
        checkCuda(cudaGetLastError(), "CUDA sparse softmax lower-bound kernel");
    }

    auto [pool_entries, pool_starts, pool_lengths, pool_count] =
        group_pools(out_indices, sizes, dim);
    const int64_t total = pool_count * nvalues;
    if (total == 0) return;

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    sparse_softmax_backward_pools_kernel<scalar_t, LogSoftMax>
        <<<coalesce_blocks(total), kCoalesceThreads, 0, stream>>>(
            total, nvalues, pool_entries.data_ptr<int64_t>(),
            pool_starts.data_ptr<int64_t>(),
            pool_lengths.data_ptr<int64_t>(), lb.data_ptr<int64_t>(),
            out_offsets.data_ptr<int64_t>(), grad_nnz,
            grad_offsets.data_ptr<int64_t>(),
            out_values.data_ptr<scalar_t>(), grad_values.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>());
    checkCuda(cudaGetLastError(), "CUDA sparse softmax backward kernel");
}

template <bool LogSoftMax>
Tensor softmax_sparse_forward(const Tensor& input_, int64_t dim_,
                              bool half_to_float, const char* fn_name) {
    TP_CHECK(!half_to_float, fn_name,
             ": with half to float conversion is not supported");
    // Explicit decomposition: the bindings feed a lambda below, and
    // capturing structured bindings is not portable across compilers.
    Tensor input, output;
    int64_t dim;
    std::tie(input, output, dim) =
        softmax_sparse_preprocessing(input_, dim_, fn_name);
    if (input.numel() == 0) {
        return output;
    }
    dispatch_float_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        sparse_coo_softmax<scalar_t, LogSoftMax>(output, input, dim);
    });
    return output;
}

template <bool LogSoftMax>
Tensor softmax_backward_sparse(const Tensor& grad_, const Tensor& output_,
                               int64_t dim_, const Tensor& input_) {
    (void)input_;
    TP_CHECK(grad_.is_sparse() && !grad_.is_sparse_compressed(),
             "_sparse_softmax_backward_data: expected a sparse COO grad");
    TP_CHECK(output_.is_sparse() && !output_.is_sparse_compressed(),
             "_sparse_softmax_backward_data: expected a sparse COO output");
    TP_CHECK(output_.shape() == grad_.shape(),
             "_sparse_softmax_backward_data: grad and output must have the "
             "same sizes");
    int64_t dim = dim_;
    const int64_t ndim = grad_.dim();
    TP_CHECK(dim >= -ndim && dim < ndim,
             "Dimension out of range (expected to be in range of [", -ndim,
             ", ", ndim - 1, "], but got ", dim, ")");
    if (dim < 0) dim += ndim;

    Tensor grad = grad_.is_coalesced() ? grad_ : grad_.coalesce();
    Tensor output = output_.is_coalesced() ? output_ : output_.coalesce();
    TP_CHECK(grad.sparse_dim() == output.sparse_dim(),
             "_sparse_softmax_backward_data: grad and output sparse dimensions "
             "must be equal");
    Tensor grad_input = Tensor::make_sparse_coo_tensor(
        output._indices().contiguous(),
        Tensor::empty(static_cast<std::vector<int64_t>>(output._values().shape()),
                      output.dtype(), output.device()),
        static_cast<std::vector<int64_t>>(output.shape()), true);
    if (output.numel() == 0) {
        return grad_input;
    }
    dispatch_float_dtype(grad.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        sparse_coo_softmax_backward<scalar_t, LogSoftMax>(
            grad_input, grad, output, dim);
    });
    return grad_input;
}

}  // namespace

Tensor _sparse_softmax_cuda(const Tensor& input, int64_t dim, bool half_to_float) {
    return softmax_sparse_forward<false>(input, dim, half_to_float,
                                         "softmax");
}

Tensor _sparse_softmax_int_cuda(const Tensor& input, int64_t dim,
                                std::optional<DType> dtype) {
    Tensor converted = dtype.has_value() && *dtype != DType::Undefined &&
                               *dtype != input.dtype()
                           ? input.to(*dtype)
                           : input;
    return _sparse_softmax_cuda(converted, dim, false);
}

Tensor _sparse_log_softmax_cuda(const Tensor& input, int64_t dim, bool half_to_float) {
    return softmax_sparse_forward<true>(input, dim, half_to_float,
                                        "log_softmax");
}

Tensor _sparse_log_softmax_int_cuda(const Tensor& input, int64_t dim,
                                    std::optional<DType> dtype) {
    Tensor converted = dtype.has_value() && *dtype != DType::Undefined &&
                               *dtype != input.dtype()
                           ? input.to(*dtype)
                           : input;
    return _sparse_log_softmax_cuda(converted, dim, false);
}

Tensor _sparse_softmax_backward_data_cuda(const Tensor& grad,
                                          const Tensor& output, int64_t dim,
                                          const Tensor& input) {
    return softmax_backward_sparse<false>(grad, output, dim, input);
}

Tensor _sparse_log_softmax_backward_data_cuda(const Tensor& grad,
                                              const Tensor& output,
                                              int64_t dim,
                                              const Tensor& input) {
    return softmax_backward_sparse<true>(grad, output, dim, input);
}

}  // namespace cuda
}  // namespace tensorplay
