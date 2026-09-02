// High-throughput indexing, masking, and scan kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Allocator.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include "GPUPrimitives.cuh"

// Narrow floating-point atomics operate on the containing 32-bit word and
// replace the selected half with an atomic compare-and-swap. The overloads
// stay at global scope so qualified scalar types resolve without ambiguity.
#include "Atomic.cuh"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace {
inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                             const std::vector<int64_t>& b) {
    // Broadcast dimensions from the trailing axis; size-one axes stretch.
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x = i < a.size() ? a[a.size() - 1 - i] : 1;
        const int64_t y = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (x != y && x != 1 && y != 1) {
            TP_THROW(RuntimeError,
                     "The size of tensor a (", x,
                     ") must match the size of tensor b (", y,
                     ") at non-singleton dimension ", rank - 1 - i);
        }
        out[rank - 1 - i] = std::max(x, y);
    }
    return out;
}
} // namespace

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

// ---------------------------------------------------------------------------
// Elementwise broadcast and mask application.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void masked_fill_kernel(int64_t n, const T* self, const bool* mask,
                                   T value, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = mask[i] ? value : self[i];
}

// Keep-predicate for lower and upper triangular masks.
template <typename T, bool Lower>
__global__ void triangular_mask_kernel(int64_t batch_rows, int64_t rows, int64_t cols,
                                       const T* in, T* out, int64_t diagonal) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < batch_rows; t += stride) {
        int64_t bi = t / rows, r = t % rows;
        const T* sp = in + bi * rows * cols + r * cols;
        T* dp = out + bi * rows * cols + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            bool keep = Lower ? (c <= r + diagonal) : (c >= r + diagonal);
            dp[c] = keep ? sp[c] : static_cast<T>(0);
        }
    }
}

// Cumulative scan, one thread per (outer, inner) slice scanned sequentially
// along the selected dimension.
template <typename T, typename Op>
__global__ void scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                            const T* in, T* out, T init_val, Op op) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        T acc = init_val;
        for (int64_t j = 0; j < d_size; ++j) {
            acc = op(acc, sp[j * inner]);
            dp[j * inner] = acc;
        }
    }
}

template <typename T, typename Op, typename IndexT>
__global__ void scan_outer_kernel(IndexT n_outer, IndexT d_size, IndexT inner,
                                  const T* __restrict__ in,
                                  T* __restrict__ out, T init_val, Op op) {
    const IndexT outer_stride = static_cast<IndexT>(gridDim.x);
    const IndexT inner_stride = static_cast<IndexT>(gridDim.y) *
        static_cast<IndexT>(blockDim.x);
    for (IndexT outer_index = static_cast<IndexT>(blockIdx.x);
         outer_index < n_outer; outer_index += outer_stride) {
        for (IndexT inner_index = static_cast<IndexT>(blockIdx.y) *
                 static_cast<IndexT>(blockDim.x) + static_cast<IndexT>(threadIdx.x);
             inner_index < inner; inner_index += inner_stride) {
            const T* sp = in + (outer_index * d_size * inner + inner_index);
            T* dp = out + (outer_index * d_size * inner + inner_index);
            T acc = init_val;
            #pragma unroll 4
            for (IndexT j = 0; j < d_size; ++j) {
                acc = op(acc, *sp);
                *dp = acc;
                sp += inner;
                dp += inner;
            }
        }
    }
}

template <typename T>
using scan_accum_t = std::conditional_t<
    (std::is_same_v<T, bool> || sizeof(T) < sizeof(int32_t)), int32_t, T>;

template <typename T, typename AccT, typename Op>
__device__ __forceinline__ AccT scan_combine(const Op& op, AccT lhs, AccT rhs) {
    return static_cast<AccT>(op(static_cast<T>(lhs), static_cast<T>(rhs)));
}

template <typename T, typename Op, int kThreadsX, int kThreadsY>
__global__ void scan_short_rows_kernel(int64_t n_rows, int64_t d_size,
                                        const T* in, T* out, T init_val, Op op) {
    static_assert(kThreadsX * kThreadsY == 512);
    using AccT = scan_accum_t<T>;
    alignas(sizeof(double)) extern __shared__ unsigned char raw[];
    AccT* shared = reinterpret_cast<AccT*>(raw);
    AccT* row_buf = shared + static_cast<int>(threadIdx.y) * (2 * kThreadsX);
    const int tid = threadIdx.x;
    const AccT identity = static_cast<AccT>(init_val);

    for (int64_t block_row = static_cast<int64_t>(blockIdx.x) * kThreadsY;
         block_row < n_rows;
         block_row += static_cast<int64_t>(gridDim.x) * kThreadsY) {
        const int64_t row = block_row + threadIdx.y;
        const bool row_exists = row < n_rows;
        const T* row_in = row_exists ? in + row * d_size : nullptr;
        T* row_out = row_exists ? out + row * d_size : nullptr;
        AccT carry = identity;

        for (int64_t tile = 0; tile < d_size;
             tile += static_cast<int64_t>(2 * kThreadsX)) {
            const int64_t pos1 = tile + tid;
            const int64_t pos2 = tile + kThreadsX + tid;
            row_buf[tid] = row_exists && pos1 < d_size
                ? static_cast<AccT>(row_in[pos1]) : identity;
            row_buf[kThreadsX + tid] = row_exists && pos2 < d_size
                ? static_cast<AccT>(row_in[pos2]) : identity;
            __syncthreads();

            if (tid == 0) {
                row_buf[0] = scan_combine<T>(op, carry, row_buf[0]);
            }
            __syncthreads();

            for (int stride = 1; stride <= kThreadsX; stride <<= 1) {
                const int base = (tid / stride) * (2 * stride) + stride;
                const int target = base + (tid % stride);
                const int source = base - 1;
                row_buf[target] = scan_combine<T>(op, row_buf[source], row_buf[target]);
                __syncthreads();
            }

            if (row_exists) {
                if (pos1 < d_size) row_out[pos1] = static_cast<T>(row_buf[tid]);
                if (pos2 < d_size) {
                    row_out[pos2] = static_cast<T>(row_buf[kThreadsX + tid]);
                }
            }
            carry = row_buf[2 * kThreadsX - 1];
            __syncthreads();
        }
    }
}

inline int scan_log_threads_x(int64_t n_rows, int64_t row_size) {
    int log_x = 0;
    int log_y = 0;
    while ((int64_t{1} << log_x) < row_size) ++log_x;
    while ((int64_t{1} << log_y) < n_rows) ++log_y;
    log_x = std::clamp((9 + log_x - log_y) / 2, 4, 9);
    return log_x;
}

template <typename T, typename Op>
void launch_short_rows_scan(int64_t n_rows, int64_t row_size,
                            const T* in, T* out, T init_val, Op op,
                            cudaStream_t stream) {
    const int log_x = scan_log_threads_x(n_rows, row_size);
    const int threads_x = 1 << log_x;
    const int threads_y = 512 / threads_x;
    const int64_t blocks = std::min<int64_t>((n_rows + threads_y - 1) / threads_y, 65535);
    const size_t shared_bytes = static_cast<size_t>(2) * threads_x * threads_y * sizeof(scan_accum_t<T>);

    switch (log_x) {
        case 4:
            scan_short_rows_kernel<T, Op, 16, 32><<<static_cast<unsigned>(blocks), dim3(16, 32), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
        case 5:
            scan_short_rows_kernel<T, Op, 32, 16><<<static_cast<unsigned>(blocks), dim3(32, 16), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
        case 6:
            scan_short_rows_kernel<T, Op, 64, 8><<<static_cast<unsigned>(blocks), dim3(64, 8), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
        case 7:
            scan_short_rows_kernel<T, Op, 128, 4><<<static_cast<unsigned>(blocks), dim3(128, 4), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
        case 8:
            scan_short_rows_kernel<T, Op, 256, 2><<<static_cast<unsigned>(blocks), dim3(256, 2), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
        default:
            scan_short_rows_kernel<T, Op, 512, 1><<<static_cast<unsigned>(blocks), dim3(512, 1), shared_bytes, stream>>>(
                n_rows, row_size, in, out, init_val, op);
            break;
    }
}

template <typename T, bool Product>
struct scan_arithmetic_op {
    __host__ __device__ T operator()(T lhs, T rhs) const {
        if constexpr (Product) {
            return static_cast<T>(lhs * rhs);
        } else {
            return static_cast<T>(lhs + rhs);
        }
    }
};

struct ScanWorkspaceKey {
    int device;
    std::uintptr_t stream;
    bool product;

    bool operator==(const ScanWorkspaceKey& other) const {
        return device == other.device && stream == other.stream && product == other.product;
    }
};

struct ScanWorkspaceKeyHash {
    size_t operator()(const ScanWorkspaceKey& key) const {
        size_t hash = std::hash<int>{}(key.device);
        hash ^= std::hash<std::uintptr_t>{}(key.stream) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
        hash ^= std::hash<bool>{}(key.product) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct ScanWorkspaceEntry {
    DataPtr storage;
    size_t capacity = 0;
    size_t required = 0;
    int64_t count = -1;
};

template <typename T>
struct ScanWorkspaceCache {
    std::mutex mutex;
    std::unordered_map<ScanWorkspaceKey, ScanWorkspaceEntry, ScanWorkspaceKeyHash> entries;
};

template <typename T>
ScanWorkspaceCache<T>& scan_workspace_cache() {
    static auto* cache = new ScanWorkspaceCache<T>();
    return *cache;
}

template <typename T, typename Launch>
bool scan_with_cached_workspace(const Tensor& input, Tensor& output,
                                bool product, Launch launch) {
    constexpr bool supported =
        std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
        std::is_same_v<T, float> || std::is_same_v<T, double>;
    if constexpr (!supported) {
        return false;
    } else {
        const int64_t count = input.numel();
        if (count > std::numeric_limits<int>::max()) return false;
        const auto stream = getCurrentCUDAStream().stream();
        ScanWorkspaceKey key{
            currentDevice(), reinterpret_cast<std::uintptr_t>(stream), product};
        auto& cache = scan_workspace_cache<T>();
        std::lock_guard<std::mutex> lock(cache.mutex);
        auto& entry = cache.entries[key];
        if (entry.count != count) {
            CUDA_CHECK(launch(nullptr, 0));
            entry.required = launch.required_bytes;
            entry.count = count;
        }
        if (entry.capacity < entry.required) {
            entry.storage = getAllocator(DeviceType::CUDA)->allocate(
                std::max<size_t>(entry.required, 1), input.device());
            entry.capacity = std::max<size_t>(entry.required, 1);
        }
        CUDA_CHECK(launch(entry.storage.get(), entry.required));
        return true;
    }
}

template <typename T, typename Op>
bool scan_flat_with_cub(const Tensor& input, Tensor& output, Op op) {
    struct Launch {
        const Tensor& input;
        Tensor& output;
        Op op;
        cudaStream_t stream;
        size_t required_bytes = 0;

        cudaError_t operator()(void* storage, size_t bytes) {
            cudaError_t error = cub::DeviceScan::InclusiveScan(
                storage, bytes, input.data_ptr<T>(), output.data_ptr<T>(), op,
                static_cast<int>(input.numel()), stream);
            required_bytes = bytes;
            return error;
        }
    } launch{input, output, op, getCurrentCUDAStream().stream()};
    return scan_with_cached_workspace<T>(input, output, true, launch);
}

template <typename T>
bool scan_flat_sum_with_cub(const Tensor& input, Tensor& output) {
    struct Launch {
        const Tensor& input;
        Tensor& output;
        cudaStream_t stream;
        size_t required_bytes = 0;

        cudaError_t operator()(void* storage, size_t bytes) {
            cudaError_t error = cub::DeviceScan::InclusiveSum(
                storage, bytes, input.data_ptr<T>(), output.data_ptr<T>(),
                static_cast<int>(input.numel()), stream);
            required_bytes = bytes;
            return error;
        }
    } launch{input, output, getCurrentCUDAStream().stream()};
    return scan_with_cached_workspace<T>(input, output, false, launch);
}

template <typename T>
bool scan_flat_product_with_cub(const Tensor& input, Tensor& output) {
    struct Launch {
        const Tensor& input;
        Tensor& output;
        cudaStream_t stream;
        size_t required_bytes = 0;

        cudaError_t operator()(void* storage, size_t bytes) {
            cudaError_t error = cub::DeviceScan::InclusiveScan(
                storage, bytes, input.data_ptr<T>(), output.data_ptr<T>(),
                std::multiplies<T>{}, static_cast<int>(input.numel()), stream);
            required_bytes = bytes;
            return error;
        }
    } launch{input, output, getCurrentCUDAStream().stream()};
    return scan_with_cached_workspace<T>(input, output, true, launch);
}

// A block owns one contiguous row and scans fixed-size tiles in order.  The
// tile carry keeps the synchronization cost bounded while allowing rows much
// longer than one block.
template <typename T, typename Op>
__global__ void scan_row_kernel(int64_t n_rows, int64_t d_size,
                                const T* in, T* out, T init_val, Op op) {
    using AccT = scan_accum_t<T>;
    constexpr int kItemsPerThread = 4;
    constexpr unsigned long long mask = 0xffffffffffffffffull;
    const unsigned lane = threadIdx.x & 31u;
    const AccT identity = static_cast<AccT>(init_val);
    const int64_t row_stride = static_cast<int64_t>(gridDim.x);

    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < n_rows;
         row += row_stride) {
        AccT carry = identity;
        for (int64_t tile = 0; tile < d_size;
             tile += static_cast<int64_t>(blockDim.x) * kItemsPerThread) {
            AccT local_prefix[kItemsPerThread];
            AccT local_total = identity;
            #pragma unroll
            for (int j = 0; j < kItemsPerThread; ++j) {
                const int64_t pos = tile +
                    static_cast<int64_t>(threadIdx.x) * kItemsPerThread + j;
                if (pos < d_size) {
                    const AccT value = static_cast<AccT>(in[row * d_size + pos]);
                    local_total = scan_combine<T>(op, local_total, value);
                    local_prefix[j] = local_total;
                } else {
                    local_prefix[j] = identity;
                }
            }

            AccT thread_prefix = local_total;
            for (unsigned offset = 1; offset < 32; offset <<= 1) {
                const AccT other = __shfl_up_sync(mask, thread_prefix, offset);
                if (lane >= offset) {
                    thread_prefix = scan_combine<T>(op, other, thread_prefix);
                }
            }
            const AccT prior_thread_prefix = __shfl_up_sync(mask, thread_prefix, 1);
            AccT before = lane == 0u ? identity : prior_thread_prefix;
            before = scan_combine<T>(op, carry, before);

            #pragma unroll
            for (int j = 0; j < kItemsPerThread; ++j) {
                const int64_t pos = tile +
                    static_cast<int64_t>(threadIdx.x) * kItemsPerThread + j;
                if (pos < d_size) {
                    out[row * d_size + pos] = static_cast<T>(
                        scan_combine<T>(op, before, local_prefix[j]));
                }
            }

            const AccT tile_total = __shfl_sync(mask, thread_prefix, 31);
            carry = scan_combine<T>(op, carry, tile_total);
        }
    }
}

template <typename T, typename Op, int kBlockThreads>
__global__ void scan_single_row_block_kernel(int64_t n_rows, int64_t d_size,
                                              const T* in, T* out,
                                              T init_val, Op op) {
    using AccT = scan_accum_t<T>;
    constexpr int kItemsPerThread = 2;
    alignas(sizeof(double)) extern __shared__ unsigned char raw[];
    AccT* buf = reinterpret_cast<AccT*>(raw);
    const int tid = threadIdx.x;
    const AccT identity = static_cast<AccT>(init_val);
    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < n_rows;
         row += static_cast<int64_t>(gridDim.x)) {
        const T* row_in = in + row * d_size;
        T* row_out = out + row * d_size;
        AccT carry = identity;

        for (int64_t tile = 0; tile < d_size;
             tile += static_cast<int64_t>(kBlockThreads) * kItemsPerThread) {
            const int64_t pos1 = tile + tid;
            const int64_t pos2 = tile + kBlockThreads + tid;
            buf[tid] = pos1 < d_size ? static_cast<AccT>(row_in[pos1]) : identity;
            buf[kBlockThreads + tid] =
                pos2 < d_size ? static_cast<AccT>(row_in[pos2]) : identity;
            __syncthreads();

            if (tid == 0) {
                buf[0] = scan_combine<T>(op, carry, buf[0]);
            }
            __syncthreads();

            for (int stride = 1; stride <= kBlockThreads; stride <<= 1) {
                const int base = (tid / stride) * (2 * stride) + stride;
                const int target = base + (tid % stride);
                const int source = base - 1;
                buf[target] = scan_combine<T>(op, buf[source], buf[target]);
                __syncthreads();
            }

            if (pos1 < d_size) row_out[pos1] = static_cast<T>(buf[tid]);
            if (pos2 < d_size) {
                row_out[pos2] = static_cast<T>(buf[kBlockThreads + tid]);
            }
            carry = buf[2 * kBlockThreads - 1];
            __syncthreads();
        }
    }
}

template <typename T, typename Op, int kBlockThreads, int kItemsPerThread>
__global__ void scan_register_block_kernel(int64_t n_rows, int64_t d_size,
                                            const T* in, T* out,
                                            T init_val, Op op) {
    using AccT = scan_accum_t<T>;
    alignas(sizeof(double)) extern __shared__ unsigned char raw[];
    AccT* totals = reinterpret_cast<AccT*>(raw);
    const int tid = threadIdx.x;
    const AccT identity = static_cast<AccT>(init_val);

    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < n_rows;
         row += static_cast<int64_t>(gridDim.x)) {
        const T* row_in = in + row * d_size;
        T* row_out = out + row * d_size;
        AccT carry = identity;

        for (int64_t tile = 0; tile < d_size;
             tile += static_cast<int64_t>(kBlockThreads) * kItemsPerThread) {
            AccT local_prefix[kItemsPerThread];
            AccT local_total = identity;
            #pragma unroll
            for (int j = 0; j < kItemsPerThread; ++j) {
                const int64_t pos = tile + static_cast<int64_t>(tid) * kItemsPerThread + j;
                if (pos < d_size) {
                    local_total = scan_combine<T>(
                        op, local_total, static_cast<AccT>(row_in[pos]));
                    local_prefix[j] = local_total;
                } else {
                    local_prefix[j] = identity;
                }
            }
            totals[tid] = local_total;
            __syncthreads();

            for (int stride = 1; stride < kBlockThreads; stride <<= 1) {
                if ((tid % (2 * stride)) >= stride) {
                    const int group = (tid / (2 * stride)) * (2 * stride);
                    totals[tid] = scan_combine<T>(
                        op, totals[group + stride - 1], totals[tid]);
                }
                __syncthreads();
            }

            AccT before = tid == 0 ? carry :
                scan_combine<T>(op, carry, totals[tid - 1]);
            #pragma unroll
            for (int j = 0; j < kItemsPerThread; ++j) {
                const int64_t pos = tile + static_cast<int64_t>(tid) * kItemsPerThread + j;
                if (pos < d_size) {
                    row_out[pos] = static_cast<T>(
                        scan_combine<T>(op, before, local_prefix[j]));
                }
            }
            carry = scan_combine<T>(op, carry, totals[kBlockThreads - 1]);
            __syncthreads();
        }
    }
}

// Stable running log-sum-exp:
// m = max(x, acc); acc = m + log1p(exp(-|x - acc|)).
template <typename T>
__global__ void logcumsumexp_scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                         const T* in, T* out) {
    using acc_t = T;
    constexpr acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        acc_t acc = neg_inf;
        for (int64_t j = 0; j < d_size; ++j) {
            acc_t x = static_cast<acc_t>(sp[j * inner]);
            acc_t m = ::max(x, acc);
            acc = (m == neg_inf) ? m : (m + ::log1p(::exp(-::fabs(x - acc))));
            dp[j * inner] = static_cast<T>(acc);
        }
    }
}

// Gather with separate result and source trailing extents.
template <typename T>
__global__ void gather_kernel(int64_t n, int64_t idx_dim_size, int64_t idx_inner,
                              int64_t self_dim_size, int64_t self_inner,
                              const T* s, const int64_t* ip, T* d) {
    // The result follows the index shape; source and index trailing extents
    // can differ on axes other than the selected one.
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; flat < n; flat += stride) {
        int64_t rem = flat;
        int64_t outer_off = rem / (idx_dim_size * idx_inner); rem -= outer_off * idx_dim_size * idx_inner;
        int64_t t = rem % idx_inner;
        int64_t idx = ip[flat];
        if (idx < 0) idx += self_dim_size;
        d[flat] = s[(outer_off * self_dim_size + idx) * self_inner + t];
    }
}

__device__ __forceinline__ void atomic_add_rel(int64_t* addr, int64_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int32_t* addr, int32_t v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(float* addr, float v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(double* addr, double v) { gpuAtomicAdd(addr, v); }
__device__ __forceinline__ void atomic_add_rel(Half* addr, Half v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(BFloat16* addr, BFloat16 v) {
    gpuAtomicAdd(addr, v);
}

__device__ __forceinline__ int64_t atomic_add_rel_return(int64_t* addr) {
    // Local overloads handle scalar widths without native atomicAdd support.
    return static_cast<int64_t>(::atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                                            static_cast<unsigned long long>(1)));
}

// Scatter and scatter-add use elementwise indexed writes. Add mode uses
// atomic accumulation and is intentionally unordered for colliding indices.
template <typename T, bool Add>
__global__ void scatter_kernel(int64_t total_idx, int64_t idx_dim_size, int64_t idx_inner,
                               int64_t self_dim_size, int64_t self_inner,
                               T* d, const int64_t* ip, const T* vp) {
    // One thread handles one indexed element. Colliding additions serialize
    // through atomics.
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; flat < total_idx; flat += stride) {
        int64_t rem = flat;
        int64_t outer_off = rem / (idx_dim_size * idx_inner);
        rem -= outer_off * idx_dim_size * idx_inner;
        int64_t t = rem % idx_inner;
        int64_t idx = ip[flat];
        if (idx < 0) idx += self_dim_size;
        int64_t dst = (outer_off * self_dim_size + idx) * self_inner + t;
        if constexpr (Add) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
                atomic_add_rel(&d[dst], vp[flat]);
            }
        }
        else d[dst] = vp[flat];
    }
}

template <typename T>
__global__ void index_add_kernel(int64_t total, int64_t inner, int64_t row,
                                 T* d, const int64_t* ip, const T* sp) {
    // One thread per (source position, inner column): adds sv into the
    // selected destination slice.
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
            int64_t k = t / inner;
            int64_t c = t % inner;
            int64_t iv = ip[k];
            if (iv < 0) iv += row;
            atomic_add_rel(&d[iv * inner + c], sp[t]);
        }
    }
}

// Index-select row gather.
template <typename T>
__global__ void index_select_kernel(int64_t total_out_elems, int64_t n_idx, int64_t inner,
                                    int64_t row, const T* s, const int64_t* ip, T* d) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < total_out_elems; i += stride) {
        int64_t t = i / inner;      // (o * n_idx + k)
        int64_t c = i % inner;
        int64_t k = t % n_idx;
        int64_t iv = ip[k];
        if (iv < 0) iv += row;
        d[i] = s[(t / n_idx * row + iv) * inner + c];
    }
}

template <typename T>
__global__ void index_select_slice_kernel(int64_t n_slices, int64_t n_idx,
                                          int64_t inner, int64_t row,
                                          const T* s, const int64_t* ip, T* d) {
    const int64_t slice_stride = static_cast<int64_t>(gridDim.x);
    for (int64_t slice = static_cast<int64_t>(blockIdx.x); slice < n_slices;
         slice += slice_stride) {
        const int64_t outer_index = slice / n_idx;
        const int64_t index_position = slice % n_idx;
        int64_t source_index = ip[index_position];
        if (source_index < 0) source_index += row;
        const T* source = s + (outer_index * row + source_index) * inner;
        T* destination = d + slice * inner;
        for (int64_t c = threadIdx.x; c < inner; c += blockDim.x) {
            destination[c] = source[c];
        }
    }
}

template <typename T>
__global__ void index_copy_kernel(int64_t n_idx_x_inner, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, const T* sp) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < n_idx_x_inner; t += stride) {
        int64_t k = t / inner, c = t % inner;
        int64_t iv = ip[k];
        if (iv < 0) iv += row;
        d[iv * inner + c] = sp[t];
    }
}

template <typename T>
__global__ void index_fill_kernel(int64_t total, int64_t inner, int64_t row,
                                  T* d, const int64_t* ip, T v) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        int64_t k = t / inner, c = t % inner;
        int64_t iv = ip[k];
        if (iv < 0) iv += row;
        d[iv * inner + c] = v;
    }
}

template <typename T, bool Accumulate>
__global__ void index_put_kernel(int64_t n, T* d, const int64_t* ip, const T* vp) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if constexpr (Accumulate) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
                atomic_add_rel(&d[ip[i]], vp[i]);
            }
        }
        else d[ip[i]] = vp[i];
    }
}

// Nonzero uses a count pass followed by a coordinate-writing pass. Matches
// claim output slots atomically during the second pass.
template <typename T>
__global__ void nonzero_count_kernel(int64_t n, const T* x, int64_t* counter) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (static_cast<bool>(x[i])) atomic_add_rel(counter, static_cast<int64_t>(1));
    }
}

template <typename T>
__global__ void nonzero_fill_kernel(int64_t n, int64_t ndim, const T* x,
                                    const int64_t* sizes, int64_t* counter, int64_t* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (!static_cast<bool>(x[i])) continue;
        int64_t slot = atomic_add_rel_return(counter);
        int64_t rem = i;
        for (int64_t d2 = ndim - 1; d2 >= 0; --d2) {
            out[slot * ndim + d2] = rem % sizes[d2];
            rem /= sizes[d2];
        }
    }
}

// Searchsorted uses a binary search: right=false selects the lower bound and
// right=true selects the upper bound.
template <typename S, typename V>
__global__ void searchsorted_kernel(int64_t n, int64_t seq_len, bool right,
                                    const S* sp, const V* vp, int64_t* rp) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        V v = vp[i];
        int64_t lo = 0, hi = seq_len;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            bool go_right = right ? !(v < static_cast<V>(sp[mid]))
                                  : (static_cast<V>(sp[mid]) < v);
            if (go_right) lo = mid + 1; else hi = mid;
        }
        rp[i] = lo;
    }
}

// max-reduce used by bincount to size the histogram on the host.
template <typename T>
__global__ void max_reduce_kernel(int64_t n, const T* x, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    T local = std::numeric_limits<T>::lowest();
    for (; i < n; i += stride) local = ::max(local, x[i]);
    // single-block style: rely on one block when launched with grid=1
    __shared__ T shm[kThreads];
    int tid = threadIdx.x;
    shm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] = ::max(shm[tid], shm[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = shm[0];
}

template <typename T>
__global__ void bincount_count_kernel(int64_t n, const T* x, int64_t* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], static_cast<int64_t>(1));
}

template <typename W>
__global__ void bincount_weighted_kernel(int64_t n, const int64_t* x, const W* wp, W* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], wp[i]);
}

// Per-slice in-place heapsort carrying original positions. Global-memory
// storage supports arbitrary slice sizes without shared-memory limits while
// preserving the stable-order contract.
template <typename T>
__global__ void sort_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                            bool descending, const T* in, T* vals, int64_t* idxs) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* vb = vals + o * d_size * inner + in2;
        int64_t* ib = idxs + o * d_size * inner + in2;
        for (int64_t j = 0; j < d_size; ++j) { vb[j * inner] = sp[j * inner]; ib[j * inner] = j; }
        // build heap on (value, index) pairs
        auto less = [&](int64_t a, int64_t b) {
            T va = vb[a * inner], vbv = vb[b * inner];
            bool lt = va < vbv, gt = va > vbv;
            // Ascending needs a MAX-heap (largest at root, extracted to the
            // tail), i.e. the *larger* element must sink on sift_down.
            return descending ? gt : lt;
        };
        auto swap_pair = [&](int64_t a, int64_t b) {
            T tv = vb[a * inner]; vb[a * inner] = vb[b * inner]; vb[b * inner] = tv;
            int64_t ti = ib[a * inner]; ib[a * inner] = ib[b * inner]; ib[b * inner] = ti;
        };
        auto sift_down = [&](int64_t start, int64_t end) {
            int64_t root = start;
            while (2 * root + 1 <= end) {
                int64_t child = 2 * root + 1;
                if (child + 1 <= end && less(child, child + 1)) child++;
                if (less(root, child)) { swap_pair(root, child); root = child; }
                else break;
            }
        };
        for (int64_t st = d_size / 2 - 1; st >= 0; --st) sift_down(st, d_size - 1);
        for (int64_t end = d_size - 1; end > 0; --end) {
            swap_pair(0, end);
            sift_down(0, end - 1);
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
// ---------------------------------------------------------------------------

Tensor masked_fill_cuda(const Tensor& self, const Tensor& mask, Scalar value) {
    // Broadcast the mask and source once, then apply the replacement in one pass.
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(mask.shape()));
    Tensor self_b = self.expand(out_shape).contiguous();
    Tensor mask_b = mask.expand(out_shape).contiguous();
    Tensor result = self_b.clone();
    int64_t n = result.numel();
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
#define TP_MF_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        masked_fill_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, self_b.data_ptr<ctype>(), mask_b.data_ptr<bool>(), v, result.data_ptr<ctype>()); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MF_CASE)
        default: TP_THROW(TypeError, "masked_fill: unsupported dtype");
    }
#undef TP_MF_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& masked_fill__cuda(Tensor& self, const Tensor& mask, Scalar value) {
    Tensor r = masked_fill_cuda(self, mask, value);
    self.copy_(r);
    return self;
}

Tensor& masked_fill_tensor__cuda(Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill__cuda(self, mask, value.item());
}

Tensor masked_fill_tensor_cuda(const Tensor& self, const Tensor& mask, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return masked_fill_cuda(self, mask, value.item());
}

// ---------------------------------------------------------------------------
// tril / triu.
// ---------------------------------------------------------------------------

Tensor tril_cuda(const Tensor& self, int64_t diagonal);
Tensor triu_cuda(const Tensor& self, int64_t diagonal);

namespace {
template <bool Lower>
Tensor triangular_mask_entry(const Tensor& self, int64_t diagonal) {
    int64_t ndim = self.dim();
    if (ndim < 2) TP_THROW(RuntimeError, "tril/triu requires tensor with at least 2 dimensions");
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t rows = self.size(ndim - 2);
    int64_t cols = self.size(ndim - 1);
    int64_t batch = self.numel() / (rows * cols);
    if (batch == 0 || rows == 0 || cols == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    int64_t work = batch * rows;
#define TP_TRI_CASE(ctype, name) \
    case DType::name: \
        triangular_mask_kernel<ctype, Lower><<<(work + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            work, rows, cols, self_c.data_ptr<ctype>(), result.data_ptr<ctype>(), diagonal); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_TRI_CASE)
        default: TP_THROW(TypeError, "tril/triu: unsupported dtype");
    }
#undef TP_TRI_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor tril_cuda(const Tensor& self, int64_t diagonal) { return triangular_mask_entry<true>(self, diagonal); }
Tensor triu_cuda(const Tensor& self, int64_t diagonal) { return triangular_mask_entry<false>(self, diagonal); }

// ---------------------------------------------------------------------------
// cumsum / cumprod / logcumsumexp.
// ---------------------------------------------------------------------------

namespace {
template <typename T, typename Op>
Tensor scan_entry(const Tensor& self, int64_t dim, T init_val, Op op) {
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), self_c.dtype(), self_c.device());
    int64_t d_size = self_c.size(dim);
    if (d_size == 0 || self_c.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    if (inner == 1 && d_size >= 16 && d_size < 512) {
        launch_short_rows_scan<T>(outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op, stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (inner == 1 && outer == 1 && d_size >= 512 && d_size <= 8192) {
        if constexpr (std::is_same_v<T, double>) {
            if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, false>>) {
                if (scan_flat_sum_with_cub<T>(self_c, result)) return result;
            } else if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, true>>) {
                if (scan_flat_product_with_cub<T>(self_c, result)) return result;
            } else if (scan_flat_with_cub<T>(self_c, result, op)) {
                return result;
            }
        }
        constexpr int kScanBlockThreads = 512;
        scan_register_block_kernel<T, Op, kScanBlockThreads, 4><<<
            1, kScanBlockThreads,
            kScanBlockThreads * sizeof(scan_accum_t<T>), stream>>>(
            1, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (inner == 1 && outer > 1 && outer <= 64 && d_size >= 8192) {
        constexpr int kScanBlockThreads = 512;
        const int64_t blocks = std::min<int64_t>(outer, 4096);
        scan_register_block_kernel<T, Op, kScanBlockThreads, 4><<<
            static_cast<unsigned>(blocks), kScanBlockThreads,
            kScanBlockThreads * sizeof(scan_accum_t<T>), stream>>>(
            outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (inner == 1 && outer == 1 && d_size >= 512) {
        if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, false>>) {
            if (scan_flat_sum_with_cub<T>(self_c, result)) return result;
        } else if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, true>>) {
            if (scan_flat_product_with_cub<T>(self_c, result)) return result;
        } else if (scan_flat_with_cub<T>(self_c, result, op)) {
            return result;
        }
    }
    if (inner == 1 && d_size >= 512) {
        const int64_t blocks = std::min<int64_t>(outer, 4096);
        constexpr int kWarpThreads = 32;
        scan_row_kernel<T, Op><<<static_cast<unsigned>(blocks), kWarpThreads, 0, stream>>>(
            outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op);
    } else if (inner > 1) {
        const int threads = static_cast<int>(std::min<int64_t>(inner, 512));
        const int64_t blocks_x = std::min<int64_t>(outer, 65535);
        const int64_t blocks_y = std::min<int64_t>(
            (inner + threads - 1) / threads, 65535);
        const dim3 grid(static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y));
        if (static_cast<uint64_t>(self_c.numel()) <=
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            scan_outer_kernel<T, Op, uint32_t><<<grid, threads, 0, stream>>>(
                static_cast<uint32_t>(outer), static_cast<uint32_t>(d_size),
                static_cast<uint32_t>(inner), self_c.data_ptr<T>(),
                result.data_ptr<T>(), init_val, op);
        } else {
            scan_outer_kernel<T, Op, int64_t><<<grid, threads, 0, stream>>>(
                outer, d_size, inner, self_c.data_ptr<T>(),
                result.data_ptr<T>(), init_val, op);
        }
    } else {
        scan_kernel<T, Op><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner,
            self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor cumsum_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "cumsum: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self : self.to(out_dtype);
#define TP_CS_CASE(ctype, name) \
    case DType::name: \
        return scan_entry<ctype>(src, dim, static_cast<ctype>(0), \
                                 scan_arithmetic_op<ctype, false>{});
    switch (out_dtype) {
        TP_CS_CASE(uint8_t, UInt8)
        TP_CS_CASE(int8_t, Int8)
        TP_CS_CASE(int16_t, Int16)
        TP_CS_CASE(int32_t, Int32)
        TP_CS_CASE(int64_t, Int64)
        TP_CS_CASE(bool, Bool)
        TP_CS_CASE(float, Float32)
        TP_CS_CASE(double, Float64)
        default: TP_THROW(TypeError, "cumsum: unsupported dtype");
    }
#undef TP_CS_CASE
    return src;
}

Tensor cumprod_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "cumprod: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self : self.to(out_dtype);
#define TP_CP_CASE(ctype, name) \
    case DType::name: \
        return scan_entry<ctype>(src, dim, static_cast<ctype>(1), \
                                 scan_arithmetic_op<ctype, true>{});
    switch (out_dtype) {
        TP_CP_CASE(uint8_t, UInt8)
        TP_CP_CASE(int8_t, Int8)
        TP_CP_CASE(int16_t, Int16)
        TP_CP_CASE(int32_t, Int32)
        TP_CP_CASE(int64_t, Int64)
        TP_CP_CASE(bool, Bool)
        TP_CP_CASE(float, Float32)
        TP_CP_CASE(double, Float64)
        default: TP_THROW(TypeError, "cumprod: unsupported dtype");
    }
#undef TP_CP_CASE
    return src;
}

Tensor logcumsumexp_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "logcumsumexp: dimension not supported for scalar tensors");
    dim = wrap_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
    int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
    int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    if (out_dtype == DType::Float32) {
        logcumsumexp_scan_kernel<float><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<float>(), result.data_ptr<float>());
    } else if (out_dtype == DType::Float64) {
        logcumsumexp_scan_kernel<double><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<double>(), result.data_ptr<double>());
    } else {
        TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// gather.
// ---------------------------------------------------------------------------

Tensor gather_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as input tensor");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim && index.size(i) > self.size(i)) {
            TP_THROW(IndexError, "Size does not match at dimension ", i,
                     " (input: ", self.size(i), ", index: ", index.size(i), ")");
        }
    }
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(idx_c.shape()), self.dtype(), self.device());
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t n = result.numel();
    int64_t self_dim_size = self.size(dim);
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
#define TP_GA_CASE(ctype, name) \
    case DType::name: \
        gather_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, idx_dim_size, idx_inner, self_dim_size, self_inner, self_c.data_ptr<ctype>(), \
            idx_c.data_ptr<int64_t>(), result.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_GA_CASE)
        default: TP_THROW(TypeError, "gather: unsupported dtype");
    }
#undef TP_GA_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// scatter / scatter_add.
// ---------------------------------------------------------------------------

namespace {
enum class ScatterMode { Assign, Add };

template <bool Add>
Tensor scatter_base_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& src) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    Tensor result = detail::contiguous_clone(self);
    int64_t idx_outer = 1;
    for (int64_t i = 0; i < dim; ++i) idx_outer *= idx_c.size(i);
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);
    if (total_idx == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
    // atomicAdd supports float/double/int32/int64(via ull); other dtypes fall
    // accelerates.
    if (Add) {
        switch (self.dtype()) {
            case DType::Float32: case DType::Float64:
            case DType::Int32: case DType::Int64:
                break;
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add on CUDA supports Float32/Float64/Int32/Int64 only");
        }
    }
#define TP_SC_CASE(ctype, name) \
    case DType::name: \
        scatter_kernel<ctype, Add><<<(total_idx * inner + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total_idx, idx_dim_size, idx_inner, self_dim_size, inner, \
            result.data_ptr<ctype>(), idx_c.data_ptr<int64_t>(), src_b.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_CASE)
        default: TP_THROW(TypeError, "scatter: unsupported dtype");
    }
#undef TP_SC_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor scatter_add_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("scatter_add_cuda");
    return scatter_base_cuda<true>(self, dim, index, src);
}
Tensor scatter_src_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_cuda<false>(self, dim, index, src);
}
Tensor scatter_value_cuda(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_cuda<false>(self, dim, index, full);
}

// scatter, written directly into self instead of a clone.  Non-contiguous self
// falls back to the out-of-place
// kernel plus a copy back so any layout works.
static Tensor& scatter_base_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index,
                                         const Tensor& src, bool add) {
    if (add) {
        // Accumulates with atomicAdd (no deterministic variant implemented).
        globalContext().alertNotDeterministic("scatter_add_");
    }
    if (!self.is_contiguous()) {
        // scatter_base_cuda<Add> already starts from a clone of self, so its
        // result is exactly what scatter_/scatter_add_ should leave in self.
        Tensor out = add ? scatter_base_cuda<true>(self, dim, index, src)
                         : scatter_base_cuda<false>(self, dim, index, src);
        self.copy_(out);
        return self;
    }
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError, "Index must have same number of dimensions as output tensor");
    }
    Tensor idx_c = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src.numel() == 1) {
        src_b = src.expand(idx_shape).contiguous();
    } else {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError, "scatter_: src shape must broadcast to the index shape");
        }
        src_b = src.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) {
        src_b = src_b.to(self.dtype());
    }
    Tensor& result = self;
    int64_t idx_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) idx_inner *= idx_c.size(i);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t idx_dim_size = idx_c.size(dim);
    int64_t total_idx = idx_c.numel();
    int64_t self_dim_size = self.size(dim);
    if (total_idx == 0) return result;
    auto stream = getCurrentCUDAStream().stream();
#define TP_SC_INPLACE_ASSIGN_CASE(ctype, name) \
    case DType::name: { \
        scatter_kernel<ctype, false><<<(total_idx * inner + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total_idx, idx_dim_size, idx_inner, self_dim_size, inner, \
            result.data_ptr<ctype>(), idx_c.data_ptr<int64_t>(), src_b.data_ptr<ctype>()); \
        break; \
    }
    if (add) {
        // atomic_add_rel covers Float32/Float64/Int32/Int64 natively; other
        // dtypes reject here instead of silently mis-accumulating.
        switch (self.dtype()) {
            case DType::Float32:
                scatter_kernel<float, true><<<(total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    total_idx, idx_dim_size, idx_inner, self_dim_size, inner, result.data_ptr<float>(),
                    idx_c.data_ptr<int64_t>(), src_b.data_ptr<float>());
                break;
            case DType::Float64:
                scatter_kernel<double, true><<<(total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    total_idx, idx_dim_size, idx_inner, self_dim_size, inner, result.data_ptr<double>(),
                    idx_c.data_ptr<int64_t>(), src_b.data_ptr<double>());
                break;
            case DType::Int32:
                scatter_kernel<int32_t, true><<<(total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    total_idx, idx_dim_size, idx_inner, self_dim_size, inner, result.data_ptr<int32_t>(),
                    idx_c.data_ptr<int64_t>(), src_b.data_ptr<int32_t>());
                break;
            case DType::Int64:
                scatter_kernel<int64_t, true><<<(total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    total_idx, idx_dim_size, idx_inner, self_dim_size, inner, result.data_ptr<int64_t>(),
                    idx_c.data_ptr<int64_t>(), src_b.data_ptr<int64_t>());
                break;
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add_ on CUDA supports Float32/Float64/Int32/Int64 only");
        }
    } else {
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_INPLACE_ASSIGN_CASE)
            default: TP_THROW(TypeError, "scatter_: unsupported dtype");
        }
    }
#undef TP_SC_INPLACE_ASSIGN_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& scatter_inplace_src_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cuda(self, dim, index, src, /*add=*/false);
}

Tensor& scatter_inplace_value_cuda(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor full = Tensor::full({}, value, self.dtype(), self.device());
    return scatter_base_inplace_cuda(self, dim, index, full, /*add=*/false);
}

Tensor& scatter_add_inplace_cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return scatter_base_inplace_cuda(self, dim, index, src, /*add=*/true);
}

// ---------------------------------------------------------------------------
// index_select.
// ---------------------------------------------------------------------------

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != 1) TP_THROW(IndexError, "index_select(): index should be a vector");
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    int64_t n_idx = idx.numel();
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape(static_cast<std::vector<int64_t>>(self.shape()));
    out_shape[dim] = n_idx;
    Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t total = result.numel();
    if (total == 0) return result;
    Tensor self_c = self.contiguous();
    auto stream = getCurrentCUDAStream().stream();
    const int slice_threads = inner >= 1024 ? 512 : kThreads;
#define TP_IS_CASE(ctype, name) \
    case DType::name: { \
        if (inner >= 64) { \
            const int64_t slices = outer * n_idx; \
            const int64_t blocks = std::min<int64_t>(slices, 4096); \
            index_select_slice_kernel<ctype><<<static_cast<unsigned>(blocks), slice_threads, 0, stream>>>( \
                slices, n_idx, inner, row, self_c.data_ptr<ctype>(), \
                idx.data_ptr<int64_t>(), result.data_ptr<ctype>()); \
        } else { \
            index_select_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, self_c.data_ptr<ctype>(), \
                idx.data_ptr<int64_t>(), result.data_ptr<ctype>()); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IS_CASE)
        default: TP_THROW(TypeError, "index_select: unsupported dtype");
    }
#undef TP_IS_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// index_add with atomic accumulation.
// ---------------------------------------------------------------------------

Tensor index_add_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor source_c = source.contiguous();
    if (source_c.dim() != nd) {
        TP_THROW(RuntimeError, "index_add: source must have same number of dims as input");
    }
    if (source_c.size(dim) != n_idx) {
        TP_THROW(RuntimeError, "index_add: source size along dim must equal index length");
    }
    int64_t total = n_idx * inner;
    auto stream = getCurrentCUDAStream().stream();
    switch (self.dtype()) {
        case DType::Float32:
            index_add_kernel<float><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, result.data_ptr<float>(), idx.data_ptr<int64_t>(),
                source_c.data_ptr<float>());
            break;
        case DType::Float64:
            index_add_kernel<double><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, result.data_ptr<double>(), idx.data_ptr<int64_t>(),
                source_c.data_ptr<double>());
            break;
        case DType::Int32:
            index_add_kernel<int32_t><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, result.data_ptr<int32_t>(), idx.data_ptr<int64_t>(),
                source_c.data_ptr<int32_t>());
            break;
        case DType::Int64:
            index_add_kernel<int64_t><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, result.data_ptr<int64_t>(), idx.data_ptr<int64_t>(),
                source_c.data_ptr<int64_t>());
            break;
        default:
            TP_THROW(NotImplementedError, "index_add on CUDA supports Float32/Float64/Int32/Int64 only");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// index_copy / index_fill.
// ---------------------------------------------------------------------------

Tensor index_copy_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t total = n_idx * inner;
    Tensor source_c = source.contiguous();
    auto stream = getCurrentCUDAStream().stream();
#define TP_IC_CASE(ctype, name) \
    case DType::name: \
        index_copy_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, inner, row, result.data_ptr<ctype>(), idx.data_ptr<int64_t>(), \
            source_c.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IC_CASE)
        default: TP_THROW(TypeError, "index_copy: unsupported dtype");
    }
#undef TP_IC_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor index_fill_scalar_cuda(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);

Tensor index_fill_tensor_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    Scalar v = value.item();
    return index_fill_scalar_cuda(self, dim, index, v);
}

Tensor index_fill_scalar_cuda(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor idx = (index.dtype() == DType::Int64) ? index.contiguous() : index.to(DType::Int64).contiguous();
    Tensor result = detail::contiguous_clone(self);
    int64_t n_idx = idx.numel();
    if (n_idx == 0) return result;
    int64_t row = self.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) inner *= self.size(i);
    int64_t total = n_idx * inner;
    auto stream = getCurrentCUDAStream().stream();
#define TP_IF_CASE(ctype, name) \
    case DType::name: { \
        ctype v = value.to<ctype>(); \
        index_fill_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total, inner, row, result.data_ptr<ctype>(), idx.data_ptr<int64_t>(), v); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IF_CASE)
        default: TP_THROW(TypeError, "index_fill: unsupported dtype");
    }
#undef TP_IF_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& index_fill_scalar__cuda(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    // Fill a clone, then copy it back through the existing in-place path.
    self.copy_(index_fill_scalar_cuda(self, dim, index, value));
    return self;
}

Tensor& index_fill_tensor__cuda(Tensor& self, int64_t dim, const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
                 value.dim(), " dimension(s).");
    }
    return index_fill_scalar__cuda(self, dim, index, value.item());
}

// ---------------------------------------------------------------------------
// index_put / index_put_.
// ---------------------------------------------------------------------------

namespace {
Tensor index_put_impl_cuda(Tensor& result, const std::vector<Tensor>& indices,
                           const Tensor& values, bool accumulate) {
    if (indices.empty()) TP_THROW(IndexError, "index_put: at least one index tensor required");
    if (accumulate) {
        // Accumulates with atomicAdd (no deterministic variant implemented).
        globalContext().alertNotDeterministic("index_put");
    }
    int64_t numel_self = result.numel();
    Tensor flat_idx = indices[0].to(DType::Int64).contiguous();
    for (size_t i = 1; i < indices.size(); ++i) {
        flat_idx = flat_idx * static_cast<int64_t>(result.size(i)) +
                   indices[i].to(DType::Int64).contiguous();
    }
    Tensor vals = values.to(result.dtype()).contiguous();
    int64_t n = flat_idx.numel();
    bool scalar_vals = vals.numel() == 1;
    if (!scalar_vals && vals.numel() != n) {
        TP_THROW(RuntimeError, "index_put: values must match number of indexed elements");
    }
    if (!scalar_vals) {
        // broadcast scalar-shaped values to n
        vals = vals.expand(std::vector<int64_t>{n}).contiguous();
    }
    auto stream = getCurrentCUDAStream().stream();
    if (accumulate) {
        switch (result.dtype()) {
            case DType::Float32:
                index_put_kernel<float, true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    n, result.data_ptr<float>(), flat_idx.data_ptr<int64_t>(), vals.data_ptr<float>());
                break;
            case DType::Float64:
                index_put_kernel<double, true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    n, result.data_ptr<double>(), flat_idx.data_ptr<int64_t>(), vals.data_ptr<double>());
                break;
            case DType::Int32:
                index_put_kernel<int32_t, true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    n, result.data_ptr<int32_t>(), flat_idx.data_ptr<int64_t>(), vals.data_ptr<int32_t>());
                break;
            case DType::Int64:
                index_put_kernel<int64_t, true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    n, result.data_ptr<int64_t>(), flat_idx.data_ptr<int64_t>(), vals.data_ptr<int64_t>());
                break;
            default:
                TP_THROW(NotImplementedError, "index_put accumulate=True on CUDA supports Float32/Float64/Int32/Int64 only");
        }
    } else {
#define TP_IP_CASE(ctype, name) \
        case DType::name: \
            index_put_kernel<ctype, false><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                n, result.data_ptr<ctype>(), flat_idx.data_ptr<int64_t>(), vals.data_ptr<ctype>()); \
            break;
        switch (result.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_IP_CASE)
            default: TP_THROW(TypeError, "index_put: unsupported dtype");
        }
#undef TP_IP_CASE
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor index_put_cuda(const Tensor& self, const std::vector<Tensor>& indices,
                      const Tensor& values, bool accumulate) {
    Tensor result = detail::contiguous_clone(self);
    return index_put_impl_cuda(result, indices, values, accumulate);
}

Tensor& index_put__cuda(Tensor& self, const std::vector<Tensor>& indices,
                        const Tensor& values, bool accumulate) {
    index_put_impl_cuda(self, indices, values, accumulate);
    return self;
}

// ---------------------------------------------------------------------------
// nonzero (count pass followed by an atomic slot-claiming pass).
// ---------------------------------------------------------------------------

Tensor nonzero_cuda(const Tensor& self) {
    Tensor self_c = self.contiguous();
    int64_t nd = self.dim();
    int64_t n = self_c.numel();
    // Empty input: no matches. Launching with a 0-block grid is a CUDA error,
    if (n == 0) {
        return Tensor::zeros({0, nd}, DType::Int64, self.device());
    }
    Tensor counter = Tensor::zeros({1}, DType::Int64, self.device());
    auto stream = getCurrentCUDAStream().stream();
#define TP_NZC_CASE(ctype, name) \
    case DType::name: \
        nonzero_count_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, self_c.data_ptr<ctype>(), counter.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZC_CASE)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZC_CASE
    CUDA_CHECK(cudaGetLastError());
    int64_t count_host = 0;
    CUDA_CHECK(cudaMemcpy(&count_host, counter.data_ptr<int64_t>(), sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    Tensor result = Tensor::zeros({count_host, nd}, DType::Int64, self.device());
    if (count_host == 0) return result;
    // The fill pass reuses counter as a slot allocator, so rewind it to zero
    // (the count pass left the match total in it).
    CUDA_CHECK(cudaMemsetAsync(counter.data_ptr<int64_t>(), 0, sizeof(int64_t),
                               stream));
    // sizes live on the host; stage them on-device for the fill kernel
    std::vector<int64_t> h_sizes(static_cast<std::vector<int64_t>>(self_c.shape()));
    Tensor sizes_d = Tensor::empty({nd}, DType::Int64, self.device());
    CUDA_CHECK(cudaMemcpy(sizes_d.data_ptr<int64_t>(), h_sizes.data(), nd * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
#define TP_NZF_CASE(ctype, name) \
    case DType::name: \
        nonzero_fill_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, nd, self_c.data_ptr<ctype>(), sizes_d.data_ptr<int64_t>(), \
            counter.data_ptr<int64_t>(), result.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZF_CASE)
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZF_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// searchsorted / bucketize.
// ---------------------------------------------------------------------------

namespace {
Tensor searchsorted_impl_cuda(const Tensor& seq_f, const Tensor& vals_f, bool out_int32, bool right) {
    Tensor seq = seq_f.contiguous();
    Tensor vals = vals_f.contiguous();
    int64_t seq_len = seq.size(-1);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                  out_int32 ? DType::Int32 : DType::Int64, vals.device());
    int64_t n = vals.numel();
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();

    auto run = [&](auto s_type_tag, auto v_type_tag) {
        using S = decltype(s_type_tag);
        using V = decltype(v_type_tag);
        if (out_int32) {
            Tensor tmp = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                       DType::Int64, vals.device());
            searchsorted_kernel<S, V><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, seq_len, right, seq.data_ptr<S>(), vals.data_ptr<V>(), tmp.data_ptr<int64_t>());
            CUDA_CHECK(cudaGetLastError());
            return tmp.to(DType::Int32);
        }
        searchsorted_kernel<S, V><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, seq_len, right, seq.data_ptr<S>(), vals.data_ptr<V>(), result.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    };

#define TP_SS_V(stype) \
    if (vals.dtype() == DType::Float32) return run(stype{}, float{}); \
    if (vals.dtype() == DType::Float64) return run(stype{}, double{}); \
    if (vals.dtype() == DType::Int64)   return run(stype{}, int64_t{}); \
    if (vals.dtype() == DType::Int32)   return run(stype{}, int32_t{});

    if (seq.dtype() == DType::Float32) { TP_SS_V(float) }
    else if (seq.dtype() == DType::Float64) { TP_SS_V(double) }
    else if (seq.dtype() == DType::Int64) { TP_SS_V(int64_t) }
    else if (seq.dtype() == DType::Int32) { TP_SS_V(int32_t) }
    else {
        Tensor seq_d = seq.to(DType::Float64);
        Tensor vals_d = vals.to(DType::Float64);
        return searchsorted_impl_cuda(seq_d, vals_d, out_int32, right);
    }
#undef TP_SS_V
    return result;
}
} // anonymous namespace

Tensor searchsorted_cuda(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
    return searchsorted_impl_cuda(sorted_sequence, self, out_int32, right);
}

Tensor bucketize_cuda(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
    // Boundaries are the sorted sequence and self supplies the query values.
    return searchsorted_impl_cuda(boundaries, self, out_int32, right);
}

// ---------------------------------------------------------------------------
// bincount (host-sized histogram followed by atomic accumulation).
// ---------------------------------------------------------------------------

Tensor bincount_cuda(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
    Tensor weights = weights_opt.value_or(Tensor());
    // Accumulates with atomicAdd (no deterministic variant implemented).
    globalContext().alertNotDeterministic("bincount_cuda");
    if (minlength < 0) TP_THROW(RuntimeError, "minlength should be >= 0");
    if (isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    Tensor inp = self.to(DType::Int64).contiguous();
    int64_t n = inp.numel();
    if (self.dim() == 1 && n == 0) {
        return Tensor::zeros({minlength}, DType::Int64, self.device());
    }
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    bool has_weights = weights.defined() && weights.numel() > 0;
    if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
        TP_THROW(RuntimeError, "weights should be 1-d and have the same length as input");
    }
    // find max via a one-block reduce (input is 1-D; sync once to read nbins)
    Tensor max_d = Tensor::zeros({1}, DType::Int64, self.device());
    auto stream = getCurrentCUDAStream().stream();
    if (n > 0) {
        max_reduce_kernel<int64_t><<<1, kThreads, 0, stream>>>(
            n, inp.data_ptr<int64_t>(), max_d.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
    }
    int64_t max_v = 0;
    CUDA_CHECK(cudaMemcpy(&max_v, max_d.data_ptr<int64_t>(), sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    if (max_v < 0) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    int64_t nbins = std::max(max_v + 1, minlength);
    if (has_weights) {
        if (weights.dtype() == DType::Float32) {
            Tensor rf = Tensor::zeros({nbins}, DType::Float32, self.device());
            Tensor w = weights.contiguous();
            bincount_weighted_kernel<float><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, inp.data_ptr<int64_t>(), w.data_ptr<float>(), rf.data_ptr<float>());
            CUDA_CHECK(cudaGetLastError());
            return rf;
        }
        Tensor rf = Tensor::zeros({nbins}, DType::Float64, self.device());
        Tensor w = weights.to(DType::Float64).contiguous();
        bincount_weighted_kernel<double><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, inp.data_ptr<int64_t>(), w.data_ptr<double>(), rf.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
        return rf;
    }
    Tensor result = Tensor::zeros({nbins}, DType::Int64, self.device());
    bincount_count_kernel<int64_t><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        n, inp.data_ptr<int64_t>(), result.data_ptr<int64_t>());
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ---------------------------------------------------------------------------
// take (reshape -> index_select -> reshape).
// ---------------------------------------------------------------------------

Tensor take_cuda(const Tensor& self, const Tensor& index) {
    Tensor flat = self.reshape({self.numel()});
    return index_select_cuda(flat, 0, index.reshape({index.numel()}))
        .reshape(static_cast<std::vector<int64_t>>(index.shape()));
}

// ---------------------------------------------------------------------------
// masked_scatter (source values are consumed in mask order).
// ---------------------------------------------------------------------------

Tensor masked_scatter_cuda(const Tensor& self, const Tensor& mask, const Tensor& source) {
    Tensor m_full = mask.to(DType::Bool).expand(
        static_cast<std::vector<int64_t>>(self.shape())).contiguous();
    Tensor src = source.contiguous().to(self.device());
    Tensor result = detail::contiguous_clone(self);
    // Deterministic source order requires a sequential walk; done on the host
    // side against a staged copy (rare op, correctness first).
    Tensor m_host = m_full.to(Device(DeviceType::CPU));
    Tensor src_host = src.to(Device(DeviceType::CPU));
    Tensor res_host = result.to(Device(DeviceType::CPU));
    int64_t n = res_host.numel();
    const bool* mp = m_host.data_ptr<bool>();
    int64_t src_n = src_host.numel();
    int64_t src_i = 0;
#define TP_MS_CASE(ctype, name) \
    case DType::name: { \
        const ctype* sp = src_host.data_ptr<ctype>(); \
        ctype* d = res_host.data_ptr<ctype>(); \
        for (int64_t i = 0; i < n && src_i < src_n; ++i) { \
            if (mp[i]) d[i] = sp[src_i++]; \
        } \
        break; \
    }
    switch (res_host.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MS_CASE)
        default: TP_THROW(TypeError, "masked_scatter: unsupported dtype");
    }
#undef TP_MS_CASE
    result.copy_(res_host);
    return result;
}

// ---------------------------------------------------------------------------
// sort / argsort (per-slice sort carrying original positions).
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> sort_cuda(const Tensor& self, int64_t dim, bool descending) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "sort: expects at least 1 dimension");
    dim = wrap_dim(dim, nd);
    Tensor self_c = self.contiguous();
    int64_t d_size = self_c.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    Tensor values = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), self_c.dtype(), self_c.device());
    Tensor indices = Tensor::empty(static_cast<std::vector<int64_t>>(self_c.shape()), DType::Int64, self_c.device());
    int64_t slices = outer * inner;
    if (slices == 0 || d_size == 0) return {values, indices};
    auto stream = getCurrentCUDAStream().stream();
#define TP_SORT_CASE(ctype, name) \
    case DType::name: \
        sort_kernel<ctype><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            slices, d_size, inner, descending, self_c.data_ptr<ctype>(), \
            values.data_ptr<ctype>(), indices.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_CASE)
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
#undef TP_SORT_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}

Tensor argsort_cuda(const Tensor& self, int64_t dim, bool descending) {
    // Indices-only variant of the per-slice sort.
    return std::get<1>(sort_cuda(self, dim, descending));
}

// ---------------------------------------------------------------------------
//   flags[i] = (i == 0) || sorted[i] != sorted[i-1]
//   group id = inclusive cumsum(flags) - 1
//   inverse[order[i]] = gid[i]; counts[g] = next boundary - boundary
// ---------------------------------------------------------------------------
namespace {

template <typename T>
__global__ void unique_flags_kernel(int64_t n, const T* __restrict__ sorted,
                                    int64_t* __restrict__ flags) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;
}

__global__ void unique_inverse_kernel(int64_t n, const int64_t* __restrict__ order,
                                      const int64_t* __restrict__ gid,
                                      int64_t* __restrict__ inverse) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) inverse[order[i]] = gid[i] - 1;
}

template <typename T>
__global__ void unique_emit_kernel(int64_t n, const T* __restrict__ sorted,
                                   const int64_t* __restrict__ flags,
                                   const int64_t* __restrict__ gid_inclusive,
                                   T* __restrict__ values,
                                   int64_t* __restrict__ starts) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i]) return;
    const int64_t g = gid_inclusive[i] - 1;  // inclusive cumsum is 1-based
    values[g] = sorted[i];
    starts[g] = i;   // segment length = next start (or n) - i, resolved on host
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> unique_cuda(const Tensor& self, bool sorted,
                                               bool return_inverse,
                                               bool return_counts) {
    TP_CHECK(self.dim() <= 1 || self.numel() == self.size(-1),
             "unique: only 1D tensors are supported");
    Tensor flat = self.contiguous().reshape({self.numel()});
    const int64_t n = flat.numel();

    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = return_inverse ? Tensor::empty({0}, DType::Int64, self.device())
                                    : Tensor();
    Tensor counts = return_counts ? Tensor::empty({0}, DType::Int64, self.device())
                                  : Tensor();
    if (n == 0) return std::make_tuple(values, inverse, counts);

    auto [sorted_vals, order] = sort_cuda(flat, 0, false);
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    Tensor flags = Tensor::zeros({n}, DType::Int64, self.device());

    #define UNIQUE_FLAGS_CASE(ctype, name)                                     \
    case DType::name:                                                          \
        unique_flags_kernel<ctype><<<blocks, threads>>>(                        \
            n, sorted_vals.data_ptr<ctype>(), flags.data_ptr<int64_t>());      \
        break;

    switch (self.dtype()) {
        UNIQUE_FLAGS_CASE(float, Float32)
        UNIQUE_FLAGS_CASE(double, Float64)
        UNIQUE_FLAGS_CASE(int64_t, Int64)
        UNIQUE_FLAGS_CASE(int32_t, Int32)
        UNIQUE_FLAGS_CASE(int16_t, Int16)
        UNIQUE_FLAGS_CASE(int8_t, Int8)
        UNIQUE_FLAGS_CASE(uint8_t, UInt8)
        case DType::Bool:
            unique_flags_kernel<bool><<<blocks, threads>>>(
                n, reinterpret_cast<const bool*>(sorted_vals.data_ptr<bool>()),
                flags.data_ptr<int64_t>());
            break;
        default:
            TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
    }
    #undef UNIQUE_FLAGS_CASE

    // gid = inclusive cumsum(flags); last element == number of groups.
    Tensor gid = flags.cumsum(0);
    const int64_t num_groups =
        gid.to(Device(DeviceType::CPU)).data_ptr<int64_t>()[n - 1];

    values = Tensor::empty({num_groups}, self.dtype(), self.device());
    if (return_inverse) {
        inverse = Tensor::empty({n}, DType::Int64, self.device());
        unique_inverse_kernel<<<blocks, threads>>>(
            n, order.data_ptr<int64_t>(), gid.data_ptr<int64_t>(),
            inverse.data_ptr<int64_t>());
    }
    if (return_counts) {
        counts = Tensor::zeros({num_groups}, DType::Int64, self.device());
        // counts[g] via a small host pass over boundaries would need a sync;
        // do it with an atomic-free scatter of segment lengths on device:
        Tensor starts = Tensor::full({num_groups}, int64_t(-1), DType::Int64,
                                     self.device());
        // emit values + record start positions
        #define UNIQUE_EMIT_CASE(ctype, name)                                  \
        case DType::name:                                                      \
            unique_emit_kernel<ctype><<<blocks, threads>>>(                     \
                n, sorted_vals.data_ptr<ctype>(), flags.data_ptr<int64_t>(),   \
                gid.data_ptr<int64_t>(), values.data_ptr<ctype>(),             \
                starts.data_ptr<int64_t>());                                    \
            break;
        switch (self.dtype()) {
            UNIQUE_EMIT_CASE(float, Float32)
            UNIQUE_EMIT_CASE(double, Float64)
            UNIQUE_EMIT_CASE(int64_t, Int64)
            UNIQUE_EMIT_CASE(int32_t, Int32)
            UNIQUE_EMIT_CASE(int16_t, Int16)
            UNIQUE_EMIT_CASE(int8_t, Int8)
            UNIQUE_EMIT_CASE(uint8_t, UInt8)
            default:
                TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
        }
        #undef UNIQUE_EMIT_CASE
        // counts[g] = next_start - start; resolved on host over the small
        // starts buffer (num_groups entries).
        std::vector<int64_t> h(num_groups);
        std::memcpy(h.data(), starts.to(Device(DeviceType::CPU)).data_ptr<int64_t>(),
                    static_cast<size_t>(num_groups) * sizeof(int64_t));
        std::vector<int64_t> hc(num_groups);
        for (int64_t g = 0; g < num_groups; ++g) {
            const int64_t e = (g + 1 < num_groups) ? h[g + 1] : n;
            hc[g] = e - h[g];
        }
        // NB: counts lives on the device -- materialize on CPU via the tensor
        // factory, then move (H2D); a raw memcpy into device memory segfaults
        // on discrete GPUs, and tensor() itself is CPU-only in this tree.
        counts = Tensor::tensor(hc, DType::Int64).to(self.device());
    }
    return std::make_tuple(values, inverse, counts);
}

// ---------------------------------------------------------------------------
// cumsum_backward: reverse scan R[i] = sum_{j>=i} g[j].
// ---------------------------------------------------------------------------

namespace {
template <typename T>
__global__ void cumsum_backward_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                       const T* in, T* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        T acc = static_cast<T>(0);
        for (int64_t j = d_size - 1; j >= 0; --j) {
            acc = static_cast<T>(acc + sp[j * inner]);
            dp[j * inner] = acc;
        }
    }
}
} // anonymous namespace

Tensor cumsum_backward_cuda(const Tensor& grad, int64_t dim) {
    int64_t nd = grad.dim();
    dim = wrap_dim(dim, nd);
    Tensor g = grad.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), g.dtype(), g.device());
    int64_t d_size = g.size(dim);
    if (d_size == 0 || g.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(g.shape()), dim, outer, inner);
    int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
#define TP_CSB_CASE(ctype, name) \
    case DType::name: \
        cumsum_backward_kernel<ctype><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            slices, d_size, inner, g.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break;
    switch (g.dtype()) {
        TP_CSB_CASE(uint8_t, UInt8)
        TP_CSB_CASE(int8_t, Int8)
        TP_CSB_CASE(int16_t, Int16)
        TP_CSB_CASE(int32_t, Int32)
        TP_CSB_CASE(int64_t, Int64)
        TP_CSB_CASE(float, Float32)
        TP_CSB_CASE(double, Float64)
        default: TP_THROW(TypeError, "cumsum_backward: unsupported dtype");
    }
#undef TP_CSB_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor scatter_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                           const Tensor& src, const std::string& reduce,
                           bool include_self);
Tensor index_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& source, const std::string& reduce,
                         bool include_self);
Tensor scatter_reduce_backward_self_cuda(const Tensor& grad,
                                         const Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Tensor& src,
                                         const std::string& reduce,
                                         bool include_self);
Tensor scatter_reduce_backward_src_cuda(const Tensor& grad,
                                        const Tensor& self, int64_t dim,
                                        const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self);
Tensor index_reduce_backward_self_cuda(const Tensor& grad,
                                       const Tensor& self, int64_t dim,
                                       const Tensor& index,
                                       const Tensor& source,
                                       const std::string& reduce,
                                       bool include_self);
Tensor index_reduce_backward_src_cuda(const Tensor& grad,
                                      const Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self);
TENSORPLAY_LIBRARY_IMPL(CUDA, IndexingKernels) {
    m.impl("masked_fill", masked_fill_cuda);
    m.impl("masked_fill_", masked_fill__cuda);
    m.impl("masked_fill.Tensor", masked_fill_tensor_cuda);
    m.impl("masked_fill_.Tensor", masked_fill_tensor__cuda);
    m.impl("tril", tril_cuda);
    m.impl("triu", triu_cuda);
    m.impl("cumsum", cumsum_cuda);
    m.impl("cumsum_backward", cumsum_backward_cuda);
    m.impl("cumprod", cumprod_cuda);
    m.impl("logcumsumexp", logcumsumexp_cuda);
    m.impl("gather", gather_cuda);
    m.impl("scatter_add", scatter_add_cuda);
    m.impl("scatter_reduce", scatter_reduce_cuda);
    m.impl("index_reduce", index_reduce_cuda);
    m.impl("_scatter_reduce_backward_self", scatter_reduce_backward_self_cuda);
    m.impl("_scatter_reduce_backward_src", scatter_reduce_backward_src_cuda);
    m.impl("_index_reduce_backward_self", index_reduce_backward_self_cuda);
    m.impl("_index_reduce_backward_src", index_reduce_backward_src_cuda);
    m.impl("scatter.src", scatter_src_cuda);
    m.impl("scatter.value", scatter_value_cuda);
    m.impl("scatter_.src", scatter_inplace_src_cuda);
    m.impl("scatter_.value", scatter_inplace_value_cuda);
    m.impl("scatter_add_", scatter_add_inplace_cuda);
    m.impl("index_select", index_select_cuda);
    m.impl("index_add", index_add_cuda);
    m.impl("index_copy", index_copy_cuda);
    m.impl("index_fill.Tensor", index_fill_tensor_cuda);
    m.impl("index_fill.Scalar", index_fill_scalar_cuda);
    m.impl("index_fill_.Tensor", index_fill_tensor__cuda);
    m.impl("index_fill_.Scalar", index_fill_scalar__cuda);
    m.impl("index_put", index_put_cuda);
    m.impl("index_put_", index_put__cuda);
    m.impl("nonzero", nonzero_cuda);
    m.impl("sort", sort_cuda);
    m.impl("argsort", argsort_cuda);
    m.impl("unique", unique_cuda);
    m.impl("searchsorted.Tensor", searchsorted_cuda);
    m.impl("bucketize.Tensor", bucketize_cuda);
    m.impl("bincount", bincount_cuda);
    m.impl("take", take_cuda);
    m.impl("masked_scatter", masked_scatter_cuda);
}

} // namespace cuda

// ---------------------------------------------------------------------------
// Reduction functors and indexed reduction helpers. With include_self=False,
// indexed slices are reset to the operation identity before accumulation;
// untouched slices retain their original values. Mean divides by full-rank
// counts, replacing zero counts with one before division.
// Floating-point dtypes only: the atomic primitives cover Float32/Float64.
// scatter_reduce_backward / index_reduce_backward.
// ---------------------------------------------------------------------------

namespace cuda {

namespace {

enum class SrReduceCuda { Sum, Prod, Mean, AMin, AMax };

SrReduceCuda parse_sr_reduce_cuda(const std::string& r) {
    if (r == "sum") return SrReduceCuda::Sum;
    if (r == "prod") return SrReduceCuda::Prod;
    if (r == "mean") return SrReduceCuda::Mean;
    if (r == "amin") return SrReduceCuda::AMin;
    if (r == "amax") return SrReduceCuda::AMax;
    TP_THROW(ValueError,
             "reduce argument must be one of 'sum', 'prod', 'mean', 'amin', "
             "'amax' but got: " + r);
}

template <typename T>
inline T sr_identity_cuda(SrReduceCuda op) {
    switch (op) {
        case SrReduceCuda::Sum:
        case SrReduceCuda::Mean: return static_cast<T>(0);
        case SrReduceCuda::Prod: return static_cast<T>(1);
        case SrReduceCuda::AMin:
            return std::numeric_limits<T>::has_infinity
                       ? std::numeric_limits<T>::infinity()
                       : std::numeric_limits<T>::max();
        case SrReduceCuda::AMax:
            return std::numeric_limits<T>::has_infinity
                       ? -std::numeric_limits<T>::infinity()
                       : std::numeric_limits<T>::lowest();
    }
    return static_cast<T>(0);  // unreachable
}

template <typename T>
__global__ void sr_exclude_init_kernel(int64_t total_idx, int64_t idx_dim_size,
                                       int64_t idx_inner, int64_t self_dim_size,
                                       int64_t self_inner, T* d,
                                       const int64_t* ip, T init_v) {
    // One thread per index element: reset that destination slot to the op
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_idx) return;
    int64_t rem = flat;
    int64_t oo = rem / (idx_dim_size * idx_inner);
    rem -= oo * idx_dim_size * idx_inner;
    int64_t j = rem % idx_inner;
    int64_t idx = ip[flat];
    if (idx < 0) idx += self_dim_size;
    d[(oo * self_dim_size + idx) * self_inner + j] = init_v;
}

template <typename T>
__global__ void sr_accum_kernel(int64_t total_idx, int64_t idx_dim_size,
                                int64_t idx_inner, int64_t self_dim_size,
                                int64_t self_inner, T* d, const int64_t* ip,
                                const T* vp, int64_t* cp, int op_int) {
    // One thread per index element: one destination element per thread, so
    // collisions go through the atomics exactly like gpu_scatter_reduce.
    int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_idx) return;
    const SrReduceCuda op = static_cast<SrReduceCuda>(op_int);
    int64_t rem = flat;
    int64_t oo = rem / (idx_dim_size * idx_inner);
    rem -= oo * idx_dim_size * idx_inner;
    int64_t j = rem % idx_inner;
    int64_t idx = ip[flat];
    if (idx < 0) idx += self_dim_size;
    const int64_t dst = (oo * self_dim_size + idx) * self_inner + j;
    const T v = vp[flat];
    switch (op) {
        case SrReduceCuda::Sum:
            gpuAtomicAdd(&d[dst], v);
            break;
        case SrReduceCuda::Prod:
            gpuAtomicMul(&d[dst], v);
            break;
        case SrReduceCuda::AMin:
            gpuAtomicMin(&d[dst], v);
            break;
        case SrReduceCuda::AMax:
            gpuAtomicMax(&d[dst], v);
            break;
        case SrReduceCuda::Mean:
            gpuAtomicAdd(&d[dst], v);
            gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
            break;
    }
}

template <typename T>
__global__ void sr_exclude_init_rows_kernel(int64_t total, int64_t K,
                                            int64_t self_dim_size,
                                            int64_t self_inner, T* d,
                                            const int64_t* ip, T init_v) {
    // index_reduce: the index is a vector, so each destination is a whole
    int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; w < total; w += stride) {
        int64_t oo = w / (K * self_inner);
        int64_t rem = w - oo * K * self_inner;
        int64_t j = rem / self_inner;
        int64_t t = rem - j * self_inner;
        d[(oo * self_dim_size + ip[j]) * self_inner + t] = init_v;
    }
}

template <typename T>
__global__ void sr_accum_rows_kernel(int64_t total, int64_t K,
                                     int64_t self_dim_size,
                                     int64_t self_inner, T* d,
                                     const int64_t* ip, const T* vp,
                                     int64_t* cp, int op_int) {
    // One thread per source element; collisions go through atomics exactly
    int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const SrReduceCuda op = static_cast<SrReduceCuda>(op_int);
    for (; w < total; w += stride) {
        int64_t oo = w / (K * self_inner);
        int64_t rem = w - oo * K * self_inner;
        int64_t j = rem / self_inner;
        int64_t t = rem - j * self_inner;
        const int64_t dst = (oo * self_dim_size + ip[j]) * self_inner + t;
        const T v = vp[w];
        switch (op) {
            case SrReduceCuda::Sum:
                gpuAtomicAdd(&d[dst], v);
                break;
            case SrReduceCuda::Prod:
                gpuAtomicMul(&d[dst], v);
                break;
            case SrReduceCuda::AMin:
                gpuAtomicMin(&d[dst], v);
                break;
            case SrReduceCuda::AMax:
                gpuAtomicMax(&d[dst], v);
                break;
            case SrReduceCuda::Mean:
                gpuAtomicAdd(&d[dst], v);
                gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
                break;
        }
    }
}

inline Tensor sr_where_cuda(const Tensor& cond, const Tensor& a,
                            const Tensor& b) {
    return Tensor::where(cond, a, b);
}

} // anonymous namespace

Tensor sr_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src_in,
                            const std::string& reduce, bool include_self);

Tensor sr_result_for_backward_cuda(const Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src,
                                   const std::string& reduce,
                                   bool include_self) {
    return sr_forward_cuda_impl(self, dim, index, src, reduce, include_self);
}

Tensor sr_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src_in,
                            const std::string& reduce, bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "index must have the same number of dimensions as self");
    }
    switch (self.dtype()) {
        case DType::Float32:
        case DType::Float64:
            break;
        default:
            TP_THROW(NotImplementedError,
                     "scatter_reduce/index_reduce on CUDA supports "
                     "floating point dtypes only");
    }
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(
        static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    {
        std::vector<int64_t> bshape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(src_in.shape()), idx_shape);
        if (bshape != idx_shape) {
            TP_THROW(RuntimeError,
                     "src/source shape must broadcast to the index shape");
        }
        src_b = src_in.expand(idx_shape).contiguous();
    }
    if (src_b.dtype() != self.dtype()) src_b = src_b.to(self.dtype());

    const int64_t idx_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= idx_c.size(i);
        return v;
    }();
    const int64_t self_inner = [&] {
        int64_t v = 1;
        for (int64_t i = dim + 1; i < nd; ++i) v *= self.size(i);
        return v;
    }();
    const int64_t idx_dim_size = idx_c.size(dim);
    const int64_t total_idx = idx_c.numel();
    const int64_t self_dim_size = self.size(dim);

    {
        const int64_t* ip0 = idx_c.data_ptr<int64_t>();
        for (int64_t i = 0; i < total_idx; ++i) {
            // ("index -1 is out of bounds for dimension D with size N").
            const int64_t v = ip0[i];
            if (v < 0 || v >= self_dim_size) {
                TP_THROW(IndexError, "index ", v,
                         " is out of bounds for dimension ", dim,
                         " with size ", self_dim_size);
            }
        }
    }

    Tensor result = detail::contiguous_clone(self);
    Tensor count;
    int64_t* cp = nullptr;
    if (op == SrReduceCuda::Mean) {
        count = Tensor::full(static_cast<std::vector<int64_t>>(self.shape()),
                             include_self ? 1 : 0, DType::Int64,
                             self.device());
        cp = count.data_ptr<int64_t>();
    }

    auto stream = getCurrentCUDAStream().stream();
    const int64_t blocks =
        total_idx > 0 ? (total_idx + kThreads - 1) / kThreads : 1;

#define TP_SR_CUDA_CASE(ctype, name)                                            \
    case DType::name: {                                                         \
        ctype* dp = result.data_ptr<ctype>();                                   \
        const int64_t* ip = idx_c.data_ptr<int64_t>();                          \
        const ctype* vp = src_b.data_ptr<ctype>();                              \
        if (!include_self && total_idx > 0 && result.numel() > 0) {              \
            sr_exclude_init_kernel<ctype>                                       \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total_idx, idx_dim_size, idx_inner, self_dim_size,           \
                    self_inner, dp, ip, sr_identity_cuda<ctype>(op));            \
        }                                                                       \
        if (total_idx > 0) {                                                     \
            sr_accum_kernel<ctype>                                              \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total_idx, idx_dim_size, idx_inner, self_dim_size,           \
                    self_inner, dp, ip, vp, cp, static_cast<int>(op));            \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TP_SR_CUDA_CASE(float, Float32)
        TP_SR_CUDA_CASE(double, Float64)
        default:
            TP_THROW(TypeError, "scatter_reduce: unsupported dtype");
    }
#undef TP_SR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

    if (op == SrReduceCuda::Mean) {
        // Avoid division by zero for slices with no selected entries.
        count = count.masked_fill(count.eq(0), 1);
        result = result.div(count.to(result.dtype()));
    }
    return result;
}

Tensor scatter_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                           const Tensor& src, const std::string& reduce,
                           bool include_self) {
    globalContext().alertNotDeterministic("scatter_reduce_cuda");
    return sr_forward_cuda_impl(self, dim, index, src, reduce, include_self);
}

// scatter_reduce this variant takes a 1-D index, a source of self's rank
// (equal sizes except dim == index.numel()), and rejects 'sum'.
Tensor ir_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& source_in,
                            const std::string& reduce, bool include_self);

Tensor index_reduce_cuda(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& source, const std::string& reduce,
                         bool include_self) {
    globalContext().alertNotDeterministic("index_reduce_cuda");
    return ir_forward_cuda_impl(self, dim, index, source, reduce,
                                include_self);
}

// scatter_reduce this variant takes a 1-D index, a source of self's rank
// (equal sizes except dim == index.numel()), and rejects 'sum'.
Tensor ir_forward_cuda_impl(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& source_in,
                            const std::string& reduce, bool include_self) {
    if (reduce != "prod" && reduce != "mean" && reduce != "amax" &&
        reduce != "amin") {
        TP_THROW(ValueError,
                 "index_reduce(): Expected reduce to be one of prod, mean, "
                 "amax or amin but got ",
                 reduce);
    }
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (nd == 0) {
        TP_THROW(RuntimeError,
                 "index_reduce(): dimension not supported for scalar tensors");
    }
    if (source_in.dim() != nd) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector");
    }
    if (index.dim() != 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got dim: ",
                 index.dim());
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) continue;
        if (source_in.size(i) != self.size(i)) {
            TP_THROW(IndexError,
                     "index_reduce(): Expected source and self to have the "
                     "same size at dimension ", i);
        }
    }
    switch (self.dtype()) {
        case DType::Float32:
        case DType::Float64:
            break;
        default:
            TP_THROW(NotImplementedError,
                     "index_reduce on CUDA supports floating point dtypes only");
    }
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    Tensor src_c = (source_in.dtype() == self.dtype())
                       ? source_in.contiguous()
                       : source_in.to(self.dtype()).contiguous();
    const int64_t K = idx_c.numel();
    if (src_c.size(dim) != K) {
        TP_THROW(IndexError,
                 "index_reduce(): Number of indices (", K,
                 ") should be equal to source.size(dim): (", src_c.size(dim),
                 "),");
    }
    const int64_t* ip = idx_c.data_ptr<int64_t>();
    const int64_t self_dim_size = self.size(dim);
    for (int64_t j = 0; j < K; ++j) {
        if (ip[j] < 0 || ip[j] >= self_dim_size) {
            TP_THROW(IndexError, "index ", ip[j],
                     " is out of bounds for dimension ", dim,
                     " with size ", self_dim_size);
        }
    }
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= self.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);

    Tensor result = detail::contiguous_clone(self);
    Tensor count;
    int64_t* cp = nullptr;
    if (reduce == "mean") {
        count = Tensor::full(static_cast<std::vector<int64_t>>(self.shape()),
                             include_self ? 1 : 0, DType::Int64,
                             self.device());
        cp = count.data_ptr<int64_t>();
    }

    auto stream = getCurrentCUDAStream().stream();
    const int64_t total = outer * K * self_inner;
    const int64_t blocks =
        total > 0 ? (total + kThreads - 1) / kThreads : 1;

#define TP_IR_CUDA_CASE(ctype, name)                                           \
    case DType::name: {                                                         \
        ctype* dp = result.data_ptr<ctype>();                                   \
        const ctype* sp = src_c.data_ptr<ctype>();                              \
        const ctype init_v = sr_identity_cuda<ctype>(op);                        \
        if (!include_self && K > 0 && total > 0) {                               \
            sr_exclude_init_rows_kernel<ctype>                                  \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total, K, self_dim_size, self_inner, dp, ip, init_v);         \
        }                                                                       \
        if (total > 0) {                                                         \
            sr_accum_rows_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(        \
                    total, K, self_dim_size, self_inner, dp, ip, sp, cp,          \
                    static_cast<int>(op));                                        \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TP_IR_CUDA_CASE(float, Float32)
        TP_IR_CUDA_CASE(double, Float64)
        default:
            TP_THROW(TypeError, "index_reduce: unsupported dtype");
    }
#undef TP_IR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

    if (reduce == "mean") {
        // Avoid division by zero for slices with no selected entries.
        count = count.masked_fill(count.eq(0), 1);
        result = result.div(count.to(result.dtype()));
    }
    return result;
}

Tensor scatter_reduce_backward_self_cuda(const Tensor& grad,
                                         const Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Tensor& src,
                                         const std::string& reduce,
                                         bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) {
        if (!include_self) return grad.scatter(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(
            dim, index, Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.scatter(dim, index, 0.0);
        return gself;
    }
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor result = sr_result_for_backward_cuda(self, dim, index, src,
                                                    reduce, include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.scatter(dim, index, Scalar(0.0));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = sr_result_for_backward_cuda(masked_self, dim, index,
                                                       src, reduce,
                                                       include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.scatter(dim, index, 0.0);
    return gself;
}

Tensor scatter_reduce_backward_src_cuda(const Tensor& grad,
                                        const Tensor& self, int64_t dim,
                                        const Tensor& index,
                                        const Tensor& src,
                                        const std::string& reduce,
                                        bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) return grad.gather(dim, index);
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = N.scatter_add(
            dim, index, Tensor::ones_like(src, src.dtype(), src.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.gather(dim, index).div(N.gather(dim, index));
    }
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor result = sr_result_for_backward_cuda(self, dim, index, src,
                                                    reduce, include_self);
        Tensor value = result.gather(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor src_is_result = src.eq(value).to(self.dtype());
        Tensor n_dist = self_is_result.scatter_add(dim, index, src_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = src_is_result.mul(distributed.gather(dim, index));
        // The source gradient is defined for every selected source entry.
        return out;
    }
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = sr_result_for_backward_cuda(
        masked_self, dim, index, src, reduce, include_self);
    Tensor src_zero = src.eq(0);
    Tensor num_zeros = Tensor::zeros_like(self, self.dtype(), self.device())
                           .scatter_add(dim, index, src_zero.to(self.dtype()))
                           .gather(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_src = src.masked_fill(single_zero, 1.0);
    Tensor masked_src_result = sr_result_for_backward_cuda(
        self, dim, index, masked_src, reduce, include_self);
    Tensor result = sr_result_for_backward_cuda(self, dim, index, src, reduce,
                                                include_self);
    Tensor gsrc = sr_where_cuda(
        single_zero,
        grad.mul(masked_src_result).gather(dim, index),
        grad.mul(result).gather(dim, index).div(src.masked_fill(src_zero, 1.0)));
    return gsrc;
}

Tensor index_reduce_backward_self_cuda(const Tensor& grad,
                                       const Tensor& self, int64_t dim,
                                       const Tensor& index,
                                       const Tensor& source,
                                       const std::string& reduce,
                                       bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    // Exclude the original values from the gradient when requested.
    if (op == SrReduceCuda::Sum) {
        if (!include_self) return grad.index_fill(dim, index, Scalar(0));
        return grad;
    }
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                              Tensor::ones_like(source, source.dtype(),
                                                source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        Tensor gself = grad.div(N);
        if (!include_self) gself = gself.index_fill(dim, index, 0.0);
        return gself;
    }
    Tensor result = index_reduce_cuda(self, dim, index, source, reduce,
                                      include_self);
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = self_is_result.mul(distributed);
        if (!include_self) out = out.index_fill(dim, index, Scalar(0.0));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_result = index_reduce_cuda(masked_self, dim, index, source,
                                             reduce, include_self);
    Tensor gself = grad.mul(masked_result).div(masked_self);
    if (!include_self) gself = gself.index_fill(dim, index, 0.0);
    return gself;
}

Tensor index_reduce_backward_src_cuda(const Tensor& grad,
                                      const Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& source,
                                      const std::string& reduce,
                                      bool include_self) {
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    if (op == SrReduceCuda::Sum) return grad.index_select(dim, index);
    if (op == SrReduceCuda::Mean) {
        Tensor N = include_self
                       ? Tensor::ones_like(grad, grad.dtype(), grad.device())
                       : Tensor::zeros_like(grad, grad.dtype(), grad.device());
        N = Tensor::index_add(N, dim, index,
                              Tensor::ones_like(source, source.dtype(),
                                                source.device()));
        N = N.masked_fill(N.eq(0), 1.0);
        return grad.index_select(dim, index).div(N.index_select(dim, index));
    }
    Tensor result = index_reduce_cuda(self, dim, index, source, reduce,
                                      include_self);
    if (op == SrReduceCuda::AMin || op == SrReduceCuda::AMax) {
        Tensor value = result.index_select(dim, index);
        Tensor self_is_result = self.eq(result).to(self.dtype());
        Tensor source_is_result = source.eq(value).to(self.dtype());
        Tensor n_dist = Tensor::index_add(self_is_result, dim, index,
                                          source_is_result);
        Tensor distributed = grad.div(n_dist);
        Tensor out = source_is_result.mul(distributed.index_select(dim, index));
        return out;
    }
    // prod
    Tensor masked_self = self.masked_fill(self.eq(0), 1.0);
    Tensor masked_self_result = index_reduce_cuda(masked_self, dim, index,
                                                  source, reduce,
                                                  include_self);
    Tensor src_zero = source.eq(0);
    Tensor num_zeros =
        Tensor::zeros_like(self, self.dtype(), self.device())
            .index_add(dim, index, src_zero.to(self.dtype()))
            .index_select(dim, index);
    Tensor single_zero = src_zero.bitwise_and(num_zeros.eq(1));
    Tensor masked_source = source.masked_fill(single_zero, 1.0);
    Tensor masked_result = index_reduce_cuda(self, dim, index, masked_source,
                                             reduce, include_self);
    Tensor gsrc = sr_where_cuda(
        single_zero,
        grad.mul(masked_result).index_select(dim, index),
        grad.mul(result).index_select(dim, index).div(
            source.masked_fill(src_zero, 1.0)));
    return gsrc;
}

} // namespace cuda

} // namespace tensorplay
