// High-throughput indexing, masking, and scan kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Bucketization.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Allocator.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include "GPUPrimitives.cuh"
#include "SortingRadixSelect.cuh"
#include "SortUtils.cuh"
#include "Complex.h"
#include "CUDALoops.cuh"

// Narrow floating-point atomics operate on the containing 32-bit word and
// replace the selected half with an atomic compare-and-swap. The overloads
// stay at global scope so qualified scalar types resolve without ambiguity.
#include "Atomic.cuh"

#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
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

inline int64_t wrap_scan_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        if (dim == -1 || dim == 0) return 0;
        TP_THROW(IndexError,
                 "Dimension out of range for a scalar tensor (expected -1 or 0, but got ",
                 dim, ")");
    }
    return wrap_dim(dim, ndim);
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
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
    (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>), float,
    std::conditional_t<
        (std::is_same_v<T, bool> || sizeof(T) < sizeof(int32_t)), int32_t, T>>;

template <typename T, typename AccT, typename Op>
__device__ __forceinline__ AccT scan_combine(const Op& op, AccT lhs, AccT rhs) {
    return static_cast<AccT>(op(lhs, rhs));
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
    template <typename AccT>
    __host__ __device__ AccT operator()(AccT lhs, AccT rhs) const {
        if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
            if constexpr (Product) {
                return static_cast<AccT>(lhs * rhs);
            } else {
                return static_cast<AccT>(lhs + rhs);
            }
        }
        const T left = static_cast<T>(lhs);
        const T right = static_cast<T>(rhs);
        if constexpr (Product) {
            return static_cast<AccT>(static_cast<T>(left * right));
        } else {
            return static_cast<AccT>(static_cast<T>(left + right));
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
template <typename T, typename AccT>
__global__ void logcumsumexp_scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                         const T* in, T* out) {
    using acc_t = AccT;
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

template <typename T>
__device__ __forceinline__ tensorplay::complex<T> logcumsumexp_complex_pair(
        const tensorplay::complex<T>& x, const tensorplay::complex<T>& y) {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    if (::isnan(x.real()) || ::isnan(x.imag()) ||
        ::isnan(y.real()) || ::isnan(y.imag())) {
        return tensorplay::complex<T>(nan, nan);
    }
    const tensorplay::complex<T> min = x.real() < y.real() ? x : y;
    const tensorplay::complex<T> max = x.real() >= y.real() ? x : y;
    const T min_real = min.real();
    const T max_real = max.real();
    if (!::isfinite(min_real) && min_real == max_real) {
        if (min_real < 0) return min;
        return tensorplay::log1p(tensorplay::exp(min) + tensorplay::exp(max) - T(1));
    }
    return tensorplay::log1p(tensorplay::exp(min - max)) + max;
}

template <typename T>
__global__ void logcumsumexp_complex_scan_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const tensorplay::complex<T>* in, tensorplay::complex<T>* out) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const tensorplay::complex<T>* sp = in + o * d_size * inner + in2;
        tensorplay::complex<T>* dp = out + o * d_size * inner + in2;
        tensorplay::complex<T> acc(neg_inf, T(0));
        for (int64_t j = 0; j < d_size; ++j) {
            acc = logcumsumexp_complex_pair(acc, sp[j * inner]);
            dp[j * inner] = acc;
        }
    }
}

template <typename T>
Tensor complex_logcumsumexp_cuda(const Tensor& src, int64_t dim) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(src.shape()),
        src.dtype(), src.device());
    const int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
    const int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    logcumsumexp_complex_scan_kernel<T><<<(slices + kThreads - 1) / kThreads,
                                           kThreads, 0, stream>>>(
        slices, d_size, inner,
        static_cast<const tensorplay::complex<T>*>(src.data_ptr()),
        static_cast<tensorplay::complex<T>*>(result.data_ptr()));
    CUDA_CHECK(cudaGetLastError());
    return result;
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
__device__ __forceinline__ void atomic_add_rel(uint8_t* addr, uint8_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int8_t* addr, int8_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(int16_t* addr, int16_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint16_t* addr, uint16_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint32_t* addr, uint32_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(uint64_t* addr, uint64_t v) {
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

// Boolean accumulation is logical OR, realized through the byte CAS loop in
// the integer-family atomic.
__device__ __forceinline__ void atomic_add_rel(bool* addr, bool v) {
    gpuAtomicAdd(addr, v);
}

template <typename T>
__device__ __forceinline__ void atomic_add_rel(tensorplay::complex<T>* addr,
                                               tensorplay::complex<T> v) {
    atomic_add_rel(reinterpret_cast<T*>(addr), v.real());
    atomic_add_rel(reinterpret_cast<T*>(addr) + 1, v.imag());
}

template <typename T>
__device__ __forceinline__ void indexed_atomic_add(T* addr, T v) {
    if constexpr (std::is_same_v<T, bool>) {
        gpuAtomicAdd(addr, v);
    } else {
        atomic_add_rel(addr, v);
    }
}

template <typename T>
inline constexpr bool scatter_add_supported_v =
    std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, Half> || std::is_same_v<T, BFloat16> ||
    std::is_same_v<T, bool> ||
    std::is_same_v<T, tensorplay::complex<float>> ||
    std::is_same_v<T, tensorplay::complex<double>> ||
    std::is_same_v<T, tensorplay::complex<Half>> ||
    std::is_same_v<T, tensorplay::complex<BFloat16>>;

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
            if constexpr (scatter_add_supported_v<T>) {
                indexed_atomic_add(&d[dst], vp[flat]);
            }
        }
        else d[dst] = vp[flat];
    }
}

template <typename T>
__global__ void index_add_kernel(int64_t total, int64_t inner, int64_t row,
                                 int64_t n_idx,
                                 T* d, const int64_t* ip, const T* sp) {
    // One thread per (source position, inner column): adds sv into the
    // selected destination slice.
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        if constexpr (scatter_add_supported_v<T> ||
                      std::is_same_v<T, tensorplay::complex<float>> ||
                      std::is_same_v<T, tensorplay::complex<double>> ||
                      std::is_same_v<T, tensorplay::complex<Half>> ||
                      std::is_same_v<T, tensorplay::complex<BFloat16>>) {
            int64_t source_slice = t / inner;
            int64_t c = t % inner;
            int64_t k = source_slice % n_idx;
            int64_t o = source_slice / n_idx;
            int64_t iv = ip[k];
            if (iv < 0) iv += row;
            indexed_atomic_add(&d[(o * row + iv) * inner + c], sp[t]);
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
            if constexpr (
                scatter_add_supported_v<T> ||
                std::is_same_v<T, tensorplay::complex<float>> ||
                std::is_same_v<T, tensorplay::complex<double>> ||
                std::is_same_v<T, tensorplay::complex<Half>> ||
                std::is_same_v<T, tensorplay::complex<BFloat16>>) {
                indexed_atomic_add(&d[ip[i]], vp[i]);
            }
        }
        else d[ip[i]] = vp[i];
    }
}

template <typename T>
inline void run_nonzero_mark_iter(TensorIteratorBase& iter) {
    gpu_kernel(iter, [] __host__ __device__(T value) -> int64_t {
        return static_cast<bool>(value != T(0)) ? int64_t(1) : int64_t(0);
    });
}

template <typename T>
__global__ void nonzero_fill_kernel(int64_t n, int64_t ndim, const T* x,
                                    const int64_t* sizes, const int64_t* positions,
                                    int64_t* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (!(x[i] != T(0))) continue;
        const int64_t slot = positions[i] - 1;
        int64_t rem = i;
        for (int64_t d2 = ndim - 1; d2 >= 0; --d2) {
            out[slot * ndim + d2] = rem % sizes[d2];
            rem /= sizes[d2];
        }
    }
}

// Searchsorted uses a binary search: right=false selects the lower bound and
// right=true selects the upper bound.  The bound comparators are written as
// `!(bd >= v)` / `!(bd > v)` so a NaN query compares greater than every
// boundary entry and lands at the end of the searched range.  Boundaries may
// be 1-D (shared table for all queries) or N-D matching the leading query
// dimensions (one table per row, shared along the innermost axis).
template <typename T>
__global__ void searchsorted_kernel(int64_t n, int64_t seq_len, bool right,
                                    int64_t idim_in, bool is_1d_boundaries,
                                    const T* sp, const T* vp, int64_t* rp) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const T v = vp[i];
        // A 1-D boundary table is shared by every query; a row-wise table
        // starts at (query row / input innermost) * table innermost.
        int64_t lo = is_1d_boundaries ? 0 : i / idim_in * seq_len;
        int64_t hi = lo + seq_len;
        const int64_t base = lo;
        while (lo < hi) {
            const int64_t mid = lo + ((hi - lo) >> 1);
            const bool go_right = right ? !(sp[mid] > v) : !(sp[mid] >= v);
            if (go_right) lo = mid + 1; else hi = mid;
        }
        rp[i] = lo - base;
    }
}

// Device range reduction used by bincount to validate inputs and size bins.
template <typename T>
__global__ void minmax_reduce_kernel(int64_t n, const T* x, T* min_out,
                                     T* max_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    for (; i < n; i += stride) {
        local_min = ::min(local_min, x[i]);
        local_max = ::max(local_max, x[i]);
    }
    __shared__ T min_shm[kThreads];
    __shared__ T max_shm[kThreads];
    int tid = threadIdx.x;
    min_shm[tid] = local_min;
    max_shm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            min_shm[tid] = ::min(min_shm[tid], min_shm[tid + s]);
            max_shm[tid] = ::max(max_shm[tid], max_shm[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_out[blockIdx.x] = min_shm[0];
        max_out[blockIdx.x] = max_shm[0];
    }
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
// preserving the stable-order contract. Kept as the fallback for shapes
// beyond the radix path's limits (slice count > 2^21 or numel > INT_MAX).
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

// Radix-sort path: each element is packed into a contiguous (sortable key,
// position) pair, one segmented radix pass orders every slice, then results
// are scattered back to the strided output layout. Radix ordering is stable,
// so equal keys keep their original relative order in both directions.
// Encodings reuse the topk bit-twiddling traits; bool gets a trivial one.
template <typename T>
struct SortRadixTraits : topk_detail::TopKRadixTraits<T> {};

// bool has no topk trait: a single-bit key suffices (false < true).
template <>
struct SortRadixTraits<bool> {
    using key_type = uint32_t;
    static constexpr int bit_count = 1;
    __device__ static inline key_type encode(bool value) { return value ? 1u : 0u; }
    __device__ static inline bool deconvert(key_type value) { return value != 0u; }
};

// Floating encodings fold negative zero onto positive zero before the sign
// flip: the two zero bit patterns compare equal (stable order), not as
// distinct magnitudes.
template <>
struct SortRadixTraits<float> : topk_detail::TopKRadixTraits<float> {
    __device__ static inline key_type encode(float value) {
        uint32_t bits = static_cast<uint32_t>(__float_as_int(value));
        if ((bits & 0x7fffffffu) == 0u) bits = 0u;
        const uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
        return value == value ? static_cast<uint32_t>(bits ^ mask) : 0xffffffffu;
    }
};

template <>
struct SortRadixTraits<double> : topk_detail::TopKRadixTraits<double> {
    __device__ static inline key_type encode(double value) {
        uint64_t bits = static_cast<uint64_t>(__double_as_longlong(value));
        if ((bits & 0x7fffffffffffffffULL) == 0ULL) bits = 0ULL;
        const uint64_t mask = (bits >> 63) ? 0xffffffffffffffffULL
                                           : 0x8000000000000000ULL;
        return value == value ? static_cast<uint64_t>(bits ^ mask)
                              : 0xffffffffffffffffULL;
    }
};

template <>
struct SortRadixTraits<Half> : topk_detail::TopKRadixTraits<Half> {
    __device__ static inline key_type encode(Half value) {
        uint16_t bits = static_cast<uint16_t>(value.x);
        if ((bits & 0x7fffu) == 0u) bits = 0u;
        const uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
        const float converted = static_cast<float>(value);
        return converted == converted ? static_cast<uint32_t>(bits ^ mask)
                                      : 0xffffu;
    }
};

template <>
struct SortRadixTraits<BFloat16> : topk_detail::TopKRadixTraits<BFloat16> {
    __device__ static inline key_type encode(BFloat16 value) {
        uint16_t bits = static_cast<uint16_t>(value.x);
        if ((bits & 0x7fffu) == 0u) bits = 0u;
        const uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
        const float converted = static_cast<float>(value);
        return converted == converted ? static_cast<uint32_t>(bits ^ mask)
                                      : 0xffffu;
    }
};

// Encoded-key comparators for the warp merge path: plain less/greater over
// the radix key, so the ordering matches the radix path exactly.
struct SortKeyLessOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a < b; }
};
struct SortKeyGreaterOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a > b; }
};

// ---------------------------------------------------------------------------
// Warp-per-slice merge sort for short slices.  One warp owns one slice and
// several warps share a block, so short rows run at wave occupancy instead
// of one full block per row; the collective load/store transpose keeps the
// global passes coalesced.  Ordering follows the encoded-key semantics of
// the radix path: keys that compare after every valid value (or NaN) land
// at the end, and stable ordering preserves the input order of ties.
// ---------------------------------------------------------------------------
template <typename T, int SortSize>
__global__ void sort_warp_merge_kernel(
    const T* __restrict__ in, T* __restrict__ vals, int64_t* __restrict__ idxs,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    constexpr int kWarpThreads = 32;
    constexpr int kItemsPerThread = SortSize / kWarpThreads;
    constexpr int kMaxBlockWarps = 16;
    using LoadValues = cub::WarpLoad<
        T, kItemsPerThread, cub::WARP_LOAD_TRANSPOSE>;
    using Sort = cub::WarpMergeSort<
        Key, kItemsPerThread, kWarpThreads, int32_t>;
    using StoreValues = cub::WarpStore<
        T, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;
    using StoreIndices = cub::WarpStore<
        int64_t, kItemsPerThread, cub::WARP_STORE_TRANSPOSE>;
    __shared__ union {
        typename LoadValues::TempStorage load_values;
        typename Sort::TempStorage sort;
        typename StoreValues::TempStorage store_values;
        typename StoreIndices::TempStorage store_indices;
    } temp_storage[kMaxBlockWarps];

    const int64_t slice = static_cast<int64_t>(blockIdx.x) * blockDim.y +
        threadIdx.y;
    if (slice >= slices) return;
    auto& warp_storage = temp_storage[threadIdx.y];
    const int64_t outer_index = slice / inner;
    const int64_t inner_index = slice - outer_index * inner;
    const int64_t base = outer_index * d_size * inner + inner_index;

    T local_values[kItemsPerThread];
    Key local_keys[kItemsPerThread];
    int32_t local_indices[kItemsPerThread];
    LoadValues(warp_storage.load_values).Load(
        topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
        local_values, static_cast<int>(d_size), static_cast<T>(0));
    __syncwarp();
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        const int position = threadIdx.x * kItemsPerThread + item;
        const bool valid = position < d_size;
        local_indices[item] = valid ? static_cast<int32_t>(position) : -1;
        local_keys[item] = valid
            ? SortRadixTraits<T>::encode(local_values[item])
            : std::numeric_limits<Key>::max();
    }
    // The oob default sorts after every valid key under the active
    // comparator: all-ones sorts last ascending, zero sorts last under the
    // descending (greater-than) order.  NaN encodes to the all-ones key,
    // which the radix ordering already places last ascending.
    const Key oob_key = descending
        ? static_cast<Key>(0)
        : std::numeric_limits<Key>::max();
    if (descending) {
        Sort(warp_storage.sort).StableSort(
            local_keys, local_indices, SortKeyGreaterOp{},
            static_cast<int>(d_size), oob_key);
    } else {
        Sort(warp_storage.sort).StableSort(
            local_keys, local_indices, SortKeyLessOp{},
            static_cast<int>(d_size), oob_key);
    }
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        local_values[item] = SortRadixTraits<T>::deconvert(local_keys[item]);
    }
    int64_t out_indices[kItemsPerThread];
    #pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
        out_indices[item] = static_cast<int64_t>(local_indices[item]);
    }
    StoreValues(warp_storage.store_values).Store(
        topk_detail::TopKStridedWriteAccessor<T>{vals + base, inner},
        local_values, static_cast<int>(d_size));
    __syncwarp();
    StoreIndices(warp_storage.store_indices).Store(
        topk_detail::TopKStridedWriteAccessor<int64_t>{idxs + base, inner},
        out_indices, static_cast<int>(d_size));
}

// ---------------------------------------------------------------------------
// Block-per-slice radix sort.  One block stages one slice through shared
// memory with cub's collective primitives, so the whole sort costs one
// coalesced read and one coalesced write with no global scatter passes.
// Slices may be strided (any sort dimension); slices shorter than the block
// capacity are padded with keys that always sort to the end, matching the
// NaN-last ordering of the encoded floating keys.
// ---------------------------------------------------------------------------
template <typename T, int BlockThreads, int ItemsPerThread>
__global__ void sort_block_radix_kernel(
    const T* __restrict__ in, T* __restrict__ vals, int64_t* __restrict__ idxs,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    using LoadValues = cub::BlockLoad<T, BlockThreads, ItemsPerThread,
                                      cub::BLOCK_LOAD_TRANSPOSE>;
    using StoreValues = cub::BlockStore<T, BlockThreads, ItemsPerThread,
                                        cub::BLOCK_STORE_TRANSPOSE>;
    using StoreIndices = cub::BlockStore<int64_t, BlockThreads, ItemsPerThread,
                                         cub::BLOCK_STORE_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<Key, BlockThreads, ItemsPerThread, int32_t>;
    __shared__ union {
        typename LoadValues::TempStorage load_values;
        typename Sort::TempStorage sort;
        typename StoreValues::TempStorage store_values;
        typename StoreIndices::TempStorage store_indices;
    } temp_storage;

    const int64_t slice = static_cast<int64_t>(blockIdx.x);
    if (slice >= slices) return;
    const int64_t outer_index = slice / inner;
    const int64_t inner_index = slice - outer_index * inner;
    const int64_t base = outer_index * d_size * inner + inner_index;

    T local_values[ItemsPerThread];
    int32_t local_indices[ItemsPerThread];
    Key local_keys[ItemsPerThread];

    // Keys that always sort to the end of the block regardless of direction.
    const Key end_key = descending ? static_cast<Key>(0)
                                   : std::numeric_limits<Key>::max();
    constexpr int capacity = BlockThreads * ItemsPerThread;
    if (d_size >= capacity) {
        LoadValues(temp_storage.load_values).Load(
            topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
            local_values);
    } else {
        LoadValues(temp_storage.load_values).Load(
            topk_detail::TopKStridedReadAccessor<T>{in + base, inner},
            local_values, static_cast<int>(d_size), static_cast<T>(0));
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int position = threadIdx.x * ItemsPerThread + item;
        const bool valid = position < d_size;
        local_indices[item] = valid ? static_cast<int32_t>(position) : -1;
        local_keys[item] = valid
            ? SortRadixTraits<T>::encode(local_values[item])
            : end_key;
    }
    if (descending) {
        Sort(temp_storage.sort).SortDescending(local_keys, local_indices);
    } else {
        Sort(temp_storage.sort).Sort(local_keys, local_indices);
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        local_values[item] = SortRadixTraits<T>::deconvert(local_keys[item]);
    }
    int64_t out_indices[ItemsPerThread];
    #pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        out_indices[item] = static_cast<int64_t>(local_indices[item]);
    }
    StoreValues(temp_storage.store_values).Store(
        topk_detail::TopKStridedWriteAccessor<T>{vals + base, inner},
        local_values, static_cast<int>(d_size));
    __syncthreads();
    StoreIndices(temp_storage.store_indices).Store(
        topk_detail::TopKStridedWriteAccessor<int64_t>{idxs + base, inner},
        out_indices, static_cast<int>(d_size));
}

// Fixed block capacity: pick the smallest power-of-two bucket covering the
// slice length so the sort performs no wasted digit passes.  Short slices
// run one warp per slice instead: the merge sort amortizes the launch over
// 16 rows per block, which dominates a block-per-row radix pass there.
template <typename T>
void launch_sort_block_radix(
    const Tensor& self_c, Tensor& values, Tensor& indices,
    int64_t slices, int64_t d_size, int64_t inner, bool descending) {
    auto stream = getCurrentCUDAStream().stream();
    if (d_size <= 128) {
        dim3 block(32, 16);
        dim3 grid(static_cast<unsigned>((slices + 15) / 16));
        sort_warp_merge_kernel<T, 128><<<grid, block, 0, stream>>>(
            self_c.data_ptr<T>(), values.data_ptr<T>(),
            indices.data_ptr<int64_t>(), slices, d_size, inner, descending);
        return;
    }
    dim3 grid(static_cast<unsigned>(slices));
    #define TP_SORT_BLOCK_CASE(CAP, IPT)                                     \
        if (d_size <= CAP) {                                                 \
            sort_block_radix_kernel<T, CAP / IPT, IPT>                       \
                <<<grid, CAP / IPT, 0, stream>>>(                            \
                    self_c.data_ptr<T>(), values.data_ptr<T>(),              \
                    indices.data_ptr<int64_t>(), slices, d_size, inner,      \
                    descending);                                             \
            return;                                                          \
        }
    TP_SORT_BLOCK_CASE(256, 4)
    TP_SORT_BLOCK_CASE(512, 8)
    TP_SORT_BLOCK_CASE(1024, 8)
    TP_SORT_BLOCK_CASE(2048, 8)
    TP_SORT_BLOCK_CASE(4096, 8)
    #undef TP_SORT_BLOCK_CASE
}

void sort_block_radix_entry(const Tensor& self_c, Tensor& values, Tensor& indices,
                            int64_t slices, int64_t d_size, int64_t inner,
                            bool descending) {
    switch (self_c.dtype()) {
        #define TP_SORT_BLOCK_TYPE(ctype, name)                              \
        case DType::name:                                                    \
            launch_sort_block_radix<ctype>(                                  \
                self_c, values, indices, slices, d_size, inner, descending); \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_BLOCK_TYPE)
        #undef TP_SORT_BLOCK_TYPE
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
}

template <typename T>
__global__ void sort_radix_pack_kernel(int64_t n, int64_t d_size, int64_t inner,
                                       const T* in,
                                       typename SortRadixTraits<T>::key_type* keys,
                                       int64_t* pos) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t slice = i / d_size;
        const int64_t j = i - slice * d_size;
        const int64_t o = slice / inner;
        const int64_t in2 = slice - o * inner;
        const int64_t src = (o * d_size + j) * inner + in2;
        keys[i] = SortRadixTraits<T>::encode(in[src]);
        pos[i] = j;
    }
}

template <typename T>
__global__ void sort_radix_unpack_kernel(int64_t n, int64_t d_size, int64_t inner,
                                         const typename SortRadixTraits<T>::key_type* keys,
                                         const int64_t* pos, T* vals, int64_t* idxs) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t slice = i / d_size;
        const int64_t j = i - slice * d_size;
        const int64_t o = slice / inner;
        const int64_t in2 = slice - o * inner;
        const int64_t dst = (o * d_size + j) * inner + in2;
        vals[dst] = SortRadixTraits<T>::deconvert(keys[i]);
        idxs[dst] = pos[i];
    }
}

// offsets[s] = s * d_size; end offsets are served by the same buffer shifted
// by one entry since every segment has identical length.
__global__ void sort_radix_fill_offsets_kernel(int n_offsets, int64_t d_size, int* offsets) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_offsets) offsets[i] = static_cast<int>(static_cast<int64_t>(i) * d_size);
}

template <typename T>
void sort_radix_impl(const Tensor& self_c, Tensor& values, Tensor& indices,
                     int64_t d_size, int64_t slices, int64_t inner, bool descending) {
    using Key = typename SortRadixTraits<T>::key_type;
    const int64_t n = self_c.numel();
    const auto device = self_c.device();
    const DType key_dtype = sizeof(Key) == 8 ? DType::UInt64 : DType::UInt32;
    Tensor keys_a = Tensor::empty({n}, key_dtype, device);
    Tensor keys_b = Tensor::empty({n}, key_dtype, device);
    Tensor pos_a = Tensor::empty({n}, DType::Int64, device);
    Tensor pos_b = Tensor::empty({n}, DType::Int64, device);
    Tensor offsets = Tensor::empty({slices + 1}, DType::Int32, device);
    auto stream = getCurrentCUDAStream().stream();
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    const int off_blocks = static_cast<int>((slices + 1 + kThreads - 1) / kThreads);
    sort_radix_pack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, static_cast<const T*>(self_c.data_ptr()),
        keys_a.data_ptr<Key>(), pos_a.data_ptr<int64_t>());
    sort_radix_fill_offsets_kernel<<<off_blocks, kThreads, 0, stream>>>(
        static_cast<int>(slices) + 1, d_size, offsets.data_ptr<int32_t>());
    cub::DoubleBuffer<Key> key_buf(keys_a.data_ptr<Key>(), keys_b.data_ptr<Key>());
    cub::DoubleBuffer<int64_t> pos_buf(pos_a.data_ptr<int64_t>(), pos_b.data_ptr<int64_t>());
    const int* begin_offsets = offsets.data_ptr<int32_t>();
    const int* end_offsets = begin_offsets + 1;
    const int n_items = static_cast<int>(n);
    const int n_segments = static_cast<int>(slices);
    const int bits = SortRadixTraits<T>::bit_count;
    size_t tmp_bytes = 0;
    cudaError_t err = descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
              nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream)
        : cub::DeviceSegmentedRadixSort::SortPairs(
              nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream);
    CUDA_CHECK(err);
    Tensor tmp = Tensor::empty({static_cast<int64_t>(std::max<size_t>(tmp_bytes, 1))},
                               DType::UInt8, device);
    err = descending
        ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
              tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream)
        : cub::DeviceSegmentedRadixSort::SortPairs(
              tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, n_segments,
              begin_offsets, end_offsets, 0, bits, stream);
    CUDA_CHECK(err);
    sort_radix_unpack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, key_buf.Current(), pos_buf.Current(),
        static_cast<T*>(values.data_ptr()), indices.data_ptr<int64_t>());
}

void radix_sort_impl(const Tensor& self_c, Tensor& values, Tensor& indices,
                     int64_t /*dim*/, int64_t /*outer*/, int64_t inner, int64_t d_size,
                     int64_t slices, bool descending) {
#define TP_RADIX_CASE(ctype, name) \
    case DType::name: \
        sort_radix_impl<ctype>(self_c, values, indices, d_size, slices, inner, descending); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RADIX_CASE)
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
#undef TP_RADIX_CASE
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
// ---------------------------------------------------------------------------

template <typename T>
inline void run_masked_fill_iter(TensorIteratorBase& iter, T value) {
    gpu_kernel(iter, [value] __host__ __device__(T self_value, bool mask_value) -> T {
        return mask_value ? value : self_value;
    });
}

inline void dispatch_masked_fill_iter(TensorIteratorBase& iter,
                                      DType dtype, const Scalar& value) {
#define TP_MF_ITER_CASE(ctype, name) \
    case DType::name: \
        run_masked_fill_iter<ctype>(iter, value.to<ctype>()); \
        break;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MF_ITER_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_MF_ITER_CASE)
        case DType::ComplexHalf:
            run_masked_fill_iter<tensorplay::complex<Half>>(
                iter, value.to<tensorplay::complex<Half>>());
            break;
        case DType::ComplexFloat:
            run_masked_fill_iter<tensorplay::complex<float>>(
                iter, value.to<tensorplay::complex<float>>());
            break;
        case DType::ComplexDouble:
            run_masked_fill_iter<tensorplay::complex<double>>(
                iter, value.to<tensorplay::complex<double>>());
            break;
        case DType::BComplex32:
            run_masked_fill_iter<tensorplay::complex<BFloat16>>(
                iter, value.to<tensorplay::complex<BFloat16>>());
            break;
        default: TP_THROW(TypeError, "masked_fill: unsupported dtype");
    }
#undef TP_MF_ITER_CASE
}

Tensor masked_fill_cuda(const Tensor& self, const Tensor& mask, Scalar value) {
    // Broadcast the mask and source once, then apply the replacement in one pass.
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(mask.shape()));
    Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
    TensorIterator iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(mask)
        .build();
    dispatch_masked_fill_iter(iter, self.dtype(), value);
    return result;
}

Tensor& masked_fill__cuda(Tensor& self, const Tensor& mask, Scalar value) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(self)
        .add_const_input(self)
        .add_const_input(mask)
        .build();
    dispatch_masked_fill_iter(iter, self.dtype(), value);
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

template <typename ComplexT, bool Product>
Tensor scan_complex_entry(const Tensor& self, int64_t dim) {
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(self_c.shape()),
        self_c.dtype(), self_c.device());
    const int64_t d_size = self_c.size(dim);
    if (d_size == 0 || self_c.numel() == 0) return result;

    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    const int64_t slices = outer * inner;
    const auto stream = getCurrentCUDAStream().stream();
    const ComplexT init_value = Product ? ComplexT(1, 0) : ComplexT(0, 0);
    using Op = scan_arithmetic_op<ComplexT, Product>;
    scan_kernel<ComplexT, Op>
        <<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner,
            static_cast<const ComplexT*>(self_c.data_ptr()),
            static_cast<ComplexT*>(result.data_ptr()), init_value, Op{});
    CUDA_CHECK(cudaGetLastError());
    return result;
}
} // anonymous namespace

Tensor cumsum_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self : self.to(out_dtype);
    if (nd == 0) {
        Tensor result = Tensor::empty({}, out_dtype, src.device());
        result.copy_(src);
        return result;
    }
    if (isComplexType(out_dtype)) {
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return scan_complex_entry<tensorplay::complex<double>, false>(compute_src, dim)
                .to(out_dtype);
        }
        return scan_complex_entry<tensorplay::complex<float>, false>(compute_src, dim)
            .to(out_dtype);
    }
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
        TP_CS_CASE(uint16_t, UInt16)
        TP_CS_CASE(uint32_t, UInt32)
        TP_CS_CASE(uint64_t, UInt64)
        TP_CS_CASE(bool, Bool)
        TP_CS_CASE(float, Float32)
        TP_CS_CASE(double, Float64)
        TP_CS_CASE(Half, Float16)
        TP_CS_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "cumsum: unsupported dtype");
    }
#undef TP_CS_CASE
    return src;
}

Tensor cumprod_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                                         : self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self : self.to(out_dtype);
    if (nd == 0) {
        Tensor result = Tensor::empty({}, out_dtype, src.device());
        result.copy_(src);
        return result;
    }
    if (isComplexType(out_dtype)) {
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return scan_complex_entry<tensorplay::complex<double>, true>(compute_src, dim)
                .to(out_dtype);
        }
        return scan_complex_entry<tensorplay::complex<float>, true>(compute_src, dim)
            .to(out_dtype);
    }
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
        TP_CP_CASE(uint16_t, UInt16)
        TP_CP_CASE(uint32_t, UInt32)
        TP_CP_CASE(uint64_t, UInt64)
        TP_CP_CASE(bool, Bool)
        TP_CP_CASE(float, Float32)
        TP_CP_CASE(double, Float64)
        TP_CP_CASE(Half, Float16)
        TP_CP_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "cumprod: unsupported dtype");
    }
#undef TP_CP_CASE
    return src;
}

Tensor logcumsumexp_cuda(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    int64_t nd = self.dim();
    dim = wrap_scan_dim(dim, nd);
    DType out_dtype = dtype.value_or(self.dtype());
    Tensor src = (self.dtype() == out_dtype) ? self.contiguous() : self.to(out_dtype).contiguous();
    if (isComplexType(out_dtype)) {
        if (nd == 0) {
            Tensor result = Tensor::empty({}, out_dtype, src.device());
            result.copy_(src);
            return result;
        }
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return complex_logcumsumexp_cuda<double>(compute_src, dim).to(out_dtype);
        }
        return complex_logcumsumexp_cuda<float>(compute_src, dim).to(out_dtype);
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(src.shape()), out_dtype, src.device());
    if (nd == 0) {
        switch (out_dtype) {
            case DType::Float32:
            case DType::Float64:
            case DType::Float16:
            case DType::BFloat16:
                result.copy_(src);
                return result;
            default:
                TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
        }
    }
    int64_t d_size = src.size(dim);
    if (d_size == 0 || src.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(src.shape()), dim, outer, inner);
    int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    if (out_dtype == DType::Float32) {
        logcumsumexp_scan_kernel<float, float><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<float>(), result.data_ptr<float>());
    } else if (out_dtype == DType::Float64) {
        logcumsumexp_scan_kernel<double, double><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<double>(), result.data_ptr<double>());
    } else if (out_dtype == DType::Float16) {
        logcumsumexp_scan_kernel<Half, float><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<Half>(), result.data_ptr<Half>());
    } else if (out_dtype == DType::BFloat16) {
        logcumsumexp_scan_kernel<BFloat16, float><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            slices, d_size, inner, src.data_ptr<BFloat16>(), result.data_ptr<BFloat16>());
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
        TENSORPLAY_FORALL_FP8_TYPES(TP_GA_CASE)
        case DType::ComplexHalf:
            gather_kernel<tensorplay::complex<Half>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<Half>*>(self_c.data_ptr()),
                idx_c.data_ptr<int64_t>(),
                static_cast<tensorplay::complex<Half>*>(result.data_ptr()));
            break;
        case DType::ComplexFloat:
            gather_kernel<tensorplay::complex<float>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<float>*>(self_c.data_ptr()),
                idx_c.data_ptr<int64_t>(),
                static_cast<tensorplay::complex<float>*>(result.data_ptr()));
            break;
        case DType::ComplexDouble:
            gather_kernel<tensorplay::complex<double>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<double>*>(self_c.data_ptr()),
                idx_c.data_ptr<int64_t>(),
                static_cast<tensorplay::complex<double>*>(result.data_ptr()));
            break;
        case DType::BComplex32:
            gather_kernel<tensorplay::complex<BFloat16>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, idx_dim_size, idx_inner, self_dim_size, self_inner,
                static_cast<const tensorplay::complex<BFloat16>*>(self_c.data_ptr()),
                idx_c.data_ptr<int64_t>(),
                static_cast<tensorplay::complex<BFloat16>*>(result.data_ptr()));
            break;
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
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
    if (Add) {
        switch (self.dtype()) {
            case DType::UInt8: case DType::Int8: case DType::Int16:
            case DType::Int32: case DType::Int64:
            case DType::UInt16: case DType::UInt32: case DType::UInt64:
            case DType::Float32: case DType::Float64:
            case DType::Float16: case DType::BFloat16: case DType::Bool:
            case DType::ComplexHalf: case DType::ComplexFloat:
            case DType::ComplexDouble: case DType::BComplex32:
                break;
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add on CUDA does not support this dtype");
        }
    }
#define TP_SC_CASE(ctype, name) \
    case DType::name: \
        scatter_kernel<ctype, Add><<<(total_idx * inner + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            total_idx, idx_dim_size, idx_inner, self_dim_size, inner, \
            static_cast<ctype*>(result.data_ptr()), idx_c.data_ptr<int64_t>(), \
            static_cast<const ctype*>(src_b.data_ptr())); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_SC_CASE)
        TP_SC_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_SC_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_SC_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_SC_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
            static_cast<ctype*>(result.data_ptr()), idx_c.data_ptr<int64_t>(), \
            static_cast<const ctype*>(src_b.data_ptr())); \
        break; \
    }
    if (add) {
#define TP_SC_INPLACE_ADD_CASE(ctype, name) \
        case DType::name: { \
            scatter_kernel<ctype, true><<<(total_idx + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total_idx, idx_dim_size, idx_inner, self_dim_size, inner, \
                static_cast<ctype*>(result.data_ptr()), idx_c.data_ptr<int64_t>(), \
                static_cast<const ctype*>(src_b.data_ptr())); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_INPLACE_ADD_CASE)
            TP_SC_INPLACE_ADD_CASE(tensorplay::complex<Half>, ComplexHalf)
            TP_SC_INPLACE_ADD_CASE(tensorplay::complex<float>, ComplexFloat)
            TP_SC_INPLACE_ADD_CASE(tensorplay::complex<double>, ComplexDouble)
            TP_SC_INPLACE_ADD_CASE(tensorplay::complex<BFloat16>, BComplex32)
            default:
                TP_THROW(NotImplementedError,
                         "scatter_add_ on CUDA does not support this dtype");
        }
#undef TP_SC_INPLACE_ADD_CASE
    } else {
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_SC_INPLACE_ASSIGN_CASE)
            TENSORPLAY_FORALL_FP8_TYPES(TP_SC_INPLACE_ASSIGN_CASE)
            TP_SC_INPLACE_ASSIGN_CASE(tensorplay::complex<Half>, ComplexHalf)
            TP_SC_INPLACE_ASSIGN_CASE(tensorplay::complex<float>, ComplexFloat)
            TP_SC_INPLACE_ASSIGN_CASE(tensorplay::complex<double>, ComplexDouble)
            TP_SC_INPLACE_ASSIGN_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
                slices, n_idx, inner, row, \
                static_cast<const ctype*>(self_c.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<ctype*>(result.data_ptr())); \
        } else { \
            index_select_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, n_idx, inner, row, \
                static_cast<const ctype*>(self_c.data_ptr()), \
                idx.data_ptr<int64_t>(), static_cast<ctype*>(result.data_ptr())); \
        } \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IS_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IS_CASE)
        TP_IS_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IS_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IS_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IS_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
    int64_t total = outer * n_idx * inner;
    auto stream = getCurrentCUDAStream().stream();
#define TP_IADD_CASE(ctype, name) \
        case DType::name: \
            index_add_kernel<ctype><<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                total, inner, row, n_idx, result.data_ptr<ctype>(), idx.data_ptr<int64_t>(), \
                source_c.data_ptr<ctype>()); \
            break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IADD_CASE)
        case DType::ComplexFloat:
            index_add_kernel<tensorplay::complex<float>><<<
                (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, n_idx,
                static_cast<tensorplay::complex<float>*>(result.data_ptr()),
                idx.data_ptr<int64_t>(),
                static_cast<const tensorplay::complex<float>*>(source_c.data_ptr()));
            break;
        case DType::ComplexDouble:
            index_add_kernel<tensorplay::complex<double>><<<
                (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, n_idx,
                static_cast<tensorplay::complex<double>*>(result.data_ptr()),
                idx.data_ptr<int64_t>(),
                static_cast<const tensorplay::complex<double>*>(source_c.data_ptr()));
            break;
        case DType::ComplexHalf:
            index_add_kernel<tensorplay::complex<Half>><<<
                (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, n_idx,
                static_cast<tensorplay::complex<Half>*>(result.data_ptr()),
                idx.data_ptr<int64_t>(),
                static_cast<const tensorplay::complex<Half>*>(source_c.data_ptr()));
            break;
        case DType::BComplex32:
            index_add_kernel<tensorplay::complex<BFloat16>><<<
                (total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                total, inner, row, n_idx,
                static_cast<tensorplay::complex<BFloat16>*>(result.data_ptr()),
                idx.data_ptr<int64_t>(),
                static_cast<const tensorplay::complex<BFloat16>*>(source_c.data_ptr()));
            break;
        default:
            TP_THROW(NotImplementedError, "index_add on CUDA does not support this dtype");
    }
#undef TP_IADD_CASE
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
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
            total, inner, row, static_cast<ctype*>(result.data_ptr()), \
            idx.data_ptr<int64_t>(), static_cast<const ctype*>(source_c.data_ptr())); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IC_CASE)
        TP_IC_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IC_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IC_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IC_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
            total, inner, row, static_cast<ctype*>(result.data_ptr()), \
            idx.data_ptr<int64_t>(), v); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IF_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_IF_CASE)
        TP_IF_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_IF_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_IF_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_IF_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
#define TP_IP_ACC_CASE(ctype, name) \
            case DType::name: \
                index_put_kernel<ctype, true><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                    n, static_cast<ctype*>(result.data_ptr()), \
                    flat_idx.data_ptr<int64_t>(), \
                    static_cast<const ctype*>(vals.data_ptr())); \
                break;
        switch (result.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_IP_ACC_CASE)
            TP_IP_ACC_CASE(tensorplay::complex<Half>, ComplexHalf)
            TP_IP_ACC_CASE(tensorplay::complex<float>, ComplexFloat)
            TP_IP_ACC_CASE(tensorplay::complex<double>, ComplexDouble)
            TP_IP_ACC_CASE(tensorplay::complex<BFloat16>, BComplex32)
#undef TP_IP_ACC_CASE
            default:
                TP_THROW(NotImplementedError, "index_put accumulate=True on CUDA does not support this dtype");
        }
    } else {
#define TP_IP_CASE(ctype, name) \
        case DType::name: \
            index_put_kernel<ctype, false><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
                n, static_cast<ctype*>(result.data_ptr()), \
                flat_idx.data_ptr<int64_t>(), \
                static_cast<const ctype*>(vals.data_ptr())); \
            break;
        switch (result.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_IP_CASE)
            TENSORPLAY_FORALL_FP8_TYPES(TP_IP_CASE)
            TP_IP_CASE(tensorplay::complex<Half>, ComplexHalf)
            TP_IP_CASE(tensorplay::complex<float>, ComplexFloat)
            TP_IP_CASE(tensorplay::complex<double>, ComplexDouble)
            TP_IP_CASE(tensorplay::complex<BFloat16>, BComplex32)
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
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    return index_put_impl_cuda(result, indices, values, accumulate);
}

Tensor& index_put__cuda(Tensor& self, const std::vector<Tensor>& indices,
                        const Tensor& values, bool accumulate) {
    index_put_impl_cuda(self, indices, values, accumulate);
    return self;
}

// ---------------------------------------------------------------------------
// nonzero (device flag/prefix pass followed by an ordered coordinate pass).
// ---------------------------------------------------------------------------

Tensor nonzero_cuda(const Tensor& self) {
    Tensor self_c = self.contiguous();
    int64_t nd = self.dim();
    int64_t n = self_c.numel();
    // Empty input: no matches. Launching with a 0-block grid is a CUDA error,
    if (n == 0) {
        return Tensor::zeros({0, nd}, DType::Int64, self.device());
    }
    TP_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
             "nonzero: input is too large for device scan");
    Tensor flags = Tensor::empty({n}, DType::Int64, self.device());
    Tensor self_flat = self_c.reshape({n});
    TensorIterator flag_iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(flags)
        .add_const_input(self_flat)
        .build();
#define TP_NZC_CASE(ctype, name) \
    case DType::name: \
        run_nonzero_mark_iter<ctype>(flag_iter); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZC_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZC_CASE)
        case DType::ComplexHalf:
            run_nonzero_mark_iter<tensorplay::complex<Half>>(flag_iter);
            break;
        case DType::ComplexFloat:
            run_nonzero_mark_iter<tensorplay::complex<float>>(flag_iter);
            break;
        case DType::ComplexDouble:
            run_nonzero_mark_iter<tensorplay::complex<double>>(flag_iter);
            break;
        case DType::BComplex32:
            run_nonzero_mark_iter<tensorplay::complex<BFloat16>>(flag_iter);
            break;
        default: TP_THROW(TypeError, "nonzero: unsupported dtype");
    }
#undef TP_NZC_CASE
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    Tensor positions = Tensor::empty({n}, DType::Int64, self.device());
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));
    Tensor scan_storage = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_storage.data_ptr(), scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));
    int64_t count_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &count_host, positions.data_ptr<int64_t>() + n - 1, sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Tensor result = Tensor::zeros({count_host, nd}, DType::Int64, self.device());
    if (count_host == 0) return result;
    // sizes live on the host; stage them on-device for the fill kernel
    std::vector<int64_t> h_sizes(static_cast<std::vector<int64_t>>(self_c.shape()));
    Tensor sizes_d = Tensor::empty({nd}, DType::Int64, self.device());
    CUDA_CHECK(cudaMemcpy(sizes_d.data_ptr<int64_t>(), h_sizes.data(), nd * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
#define TP_NZF_CASE(ctype, name) \
    case DType::name: \
        nonzero_fill_kernel<ctype><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            n, nd, self_c.data_ptr<ctype>(), sizes_d.data_ptr<int64_t>(), \
            positions.data_ptr<int64_t>(), result.data_ptr<int64_t>()); \
        break;
    switch (self_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NZF_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_NZF_CASE)
        case DType::ComplexHalf:
            nonzero_fill_kernel<tensorplay::complex<Half>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<Half>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::ComplexFloat:
            nonzero_fill_kernel<tensorplay::complex<float>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<float>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::ComplexDouble:
            nonzero_fill_kernel<tensorplay::complex<double>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<double>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
        case DType::BComplex32:
            nonzero_fill_kernel<tensorplay::complex<BFloat16>><<<
                (n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, nd, static_cast<const tensorplay::complex<BFloat16>*>(self_c.data_ptr()),
                sizes_d.data_ptr<int64_t>(), positions.data_ptr<int64_t>(),
                result.data_ptr<int64_t>());
            break;
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

// Materializes boundaries[sorter[i]] element by element: an unsorted boundary
// tensor is reindexed into ascending order once, before the search loop runs.
template <typename T>
__global__ void sorter_gather_kernel(int64_t n, const T* src,
                                     const int64_t* sorter, T* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[sorter[i]];
}

Tensor searchsorted_apply_sorter_cuda(const Tensor& boundaries,
                                      const Tensor& sorter) {
    Tensor sorted = Tensor::empty(
        static_cast<std::vector<int64_t>>(boundaries.shape()),
        boundaries.dtype(), boundaries.device());
    Tensor seq_c = boundaries.contiguous();
    Tensor sorter_c = sorter.contiguous();
    const int64_t n = seq_c.numel();
    if (n == 0) return sorted;
    auto stream = getCurrentCUDAStream().stream();
#define TP_SS_SORTER_CASE(ctype, name)                                        \
    case DType::name:                                                         \
        sorter_gather_kernel<ctype><<<(n + kThreads - 1) / kThreads,          \
                                      kThreads, 0, stream>>>(                 \
            n, seq_c.data_ptr<ctype>(), sorter_c.data_ptr<int64_t>(),         \
            sorted.data_ptr<ctype>());                                        \
        break;
    switch (seq_c.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SS_SORTER_CASE)
        default: TP_THROW(TypeError,
                          "searchsorted(): unsupported boundaries dtype ",
                          toString(seq_c.dtype()));
    }
#undef TP_SS_SORTER_CASE
    CUDA_CHECK(cudaGetLastError());
    return sorted;
}

// Lower bound (`right=false`) or upper bound (`right=true`) positions for
// every query value.  Inputs must be contiguous with a common dtype; the
// direction flag comes pre-resolved through the `side` alias.
Tensor searchsorted_impl_cuda(const Tensor& seq_f, const Tensor& vals_f, bool out_int32, bool right) {
    Tensor seq = seq_f.contiguous();
    Tensor vals = vals_f.contiguous();
    int64_t seq_len = seq.size(-1);
    const bool is_1d_boundaries = seq.dim() == 1;
    const int64_t idim_in =
        (vals.dim() == 0 && vals.numel() == 1) ? 1 : vals.size(-1);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                  out_int32 ? DType::Int32 : DType::Int64, vals.device());
    int64_t n = vals.numel();
    if (n == 0) return result;
    auto stream = getCurrentCUDAStream().stream();

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        if (out_int32) {
            Tensor tmp = Tensor::empty(static_cast<std::vector<int64_t>>(vals.shape()),
                                       DType::Int64, vals.device());
            searchsorted_kernel<T><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, seq_len, right, idim_in, is_1d_boundaries,
                seq.data_ptr<T>(), vals.data_ptr<T>(), tmp.data_ptr<int64_t>());
            CUDA_CHECK(cudaGetLastError());
            return tmp.to(DType::Int32);
        }
        searchsorted_kernel<T><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, seq_len, right, idim_in, is_1d_boundaries,
            seq.data_ptr<T>(), vals.data_ptr<T>(), result.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
        return result;
    };

#define TP_SS_CASE(ctype, name) \
    case DType::name: return run(ctype{});
    switch (vals.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SS_CASE)
        default:
            TP_THROW(TypeError, "searchsorted: unsupported dtype ",
                    toString(vals.dtype()));
    }
#undef TP_SS_CASE
    return result;
}
} // anonymous namespace

// Direction resolution, validation and operand normalization follow the same
// contract as the CPU kernels; only the contiguous search loop is device code.
Tensor& searchsorted_out_cuda_impl(const Tensor& sorted_sequence,
                                   const Tensor& self, bool out_int32,
                                   bool right,
                                   const std::optional<std::string>& side_opt,
                                   const Tensor& sorter_opt, Tensor& result) {
    bucketization::pre_check(sorted_sequence, self, result, out_int32, right,
                             side_opt, sorter_opt);
    result.resize_(static_cast<std::vector<int64_t>>(self.shape()));
    const bool is_right = side_opt.has_value() ? *side_opt == "right" : right;
    if (self.numel() == 0) return result;

    Tensor seq = sorted_sequence;
    Tensor sorter = sorter_opt;
    if (sorter.defined()) {
        // Materialize the reindexed boundaries so the device search loop needs
        // no per-comparison indirection.
        seq = searchsorted_apply_sorter_cuda(seq, sorter);
    }

    Tensor vals = self;
    Tensor trimmed_input, trimmed_boundaries;
    bucketization::maybe_trim_input_tensors(trimmed_input, trimmed_boundaries,
                                            vals, seq);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : vals;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : seq;
    Tensor computed = searchsorted_impl_cuda(final_boundaries, final_input,
                                             out_int32, is_right);
    if (&result != &computed) {
        result.copy_(computed);
    }
    return result;
}

Tensor& searchsorted_out_cuda(const Tensor& sorted_sequence, const Tensor& self,
                              bool out_int32, bool right,
                              const std::optional<std::string>& side_opt,
                              const std::optional<Tensor>& sorter_opt,
                              Tensor& result) {
    return searchsorted_out_cuda_impl(
        sorted_sequence, self, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_cuda(const Tensor& sorted_sequence, const Tensor& self,
                         bool out_int32, bool right,
                         const std::optional<std::string>& side_opt,
                         const std::optional<Tensor>& sorter_opt) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, self.device());
    searchsorted_out_cuda_impl(sorted_sequence, self, out_int32, right,
                               side_opt, sorter_opt.value_or(Tensor()),
                               result);
    return result;
}

Tensor& searchsorted_scalar_out_cuda(const Tensor& sorted_sequence,
                                     const Scalar& self, bool out_int32,
                                     bool right,
                                     const std::optional<std::string>& side_opt,
                                     const std::optional<Tensor>& sorter_opt,
                                     Tensor& result) {
    Tensor scalar_tensor =
        bucketization::scalar_tensor(self, sorted_sequence.device());
    return searchsorted_out_cuda_impl(
        sorted_sequence, scalar_tensor, out_int32, right, side_opt,
        sorter_opt.value_or(Tensor()), result);
}

Tensor searchsorted_scalar_cuda(const Tensor& sorted_sequence, const Scalar& self,
                                bool out_int32, bool right,
                                const std::optional<std::string>& side_opt,
                                const std::optional<Tensor>& sorter_opt) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, sorted_sequence.device());
    searchsorted_scalar_out_cuda(sorted_sequence, self, out_int32, right,
                                 side_opt, sorter_opt.value_or(Tensor()),
                                 result);
    return result;
}

Tensor& bucketize_out_cuda(const Tensor& self, const Tensor& boundaries,
                           bool out_int32, bool right, Tensor& result) {
    TP_CHECK(boundaries.dim() == 1,
             "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
             boundaries.dim(), ")");
    return searchsorted_out_cuda_impl(boundaries, self, out_int32, right,
                                      std::nullopt, Tensor(), result);
}

Tensor bucketize_cuda(const Tensor& self, const Tensor& boundaries,
                      bool out_int32, bool right) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, self.device());
    bucketize_out_cuda(self, boundaries, out_int32, right, result);
    return result;
}

Tensor& bucketize_scalar_out_cuda(const Scalar& self, const Tensor& boundaries,
                                  bool out_int32, bool right, Tensor& result) {
    Tensor scalar_tensor =
        bucketization::scalar_tensor(self, boundaries.device());
    return bucketize_out_cuda(scalar_tensor, boundaries, out_int32, right,
                              result);
}

Tensor bucketize_scalar_cuda(const Scalar& self, const Tensor& boundaries,
                             bool out_int32, bool right) {
    Tensor result = Tensor::empty(
        {}, out_int32 ? DType::Int32 : DType::Int64, boundaries.device());
    bucketize_scalar_out_cuda(self, boundaries, out_int32, right, result);
    return result;
}

// ---------------------------------------------------------------------------
// bincount (device range reduction followed by atomic accumulation).
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
    // Read only the reduced range metadata before allocating the output.
    Tensor bounds_d = Tensor::empty({2}, DType::Int64, self.device());
    auto stream = getCurrentCUDAStream().stream();
    minmax_reduce_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        n, inp.data_ptr<int64_t>(), bounds_d.data_ptr<int64_t>(),
        bounds_d.data_ptr<int64_t>() + 1);
    CUDA_CHECK(cudaGetLastError());
    int64_t bounds[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(bounds, bounds_d.data_ptr<int64_t>(),
                          sizeof(bounds), cudaMemcpyDeviceToHost));
    const int64_t min_v = bounds[0];
    const int64_t max_v = bounds[1];
    if (min_v < 0) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    if (max_v >= std::numeric_limits<int64_t>::max()) {
        TP_THROW(RuntimeError, "maximum value of input overflowed");
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

template <typename T>
inline void run_masked_scatter_iter(TensorIteratorBase& iter,
                                    const T* source) {
    gpu_kernel(iter, [source] __host__ __device__(
        T self_value, bool mask_value, int64_t source_offset) -> T {
        return mask_value ? source[source_offset] : self_value;
    });
}

__global__ void masked_scatter_size_check(
    const int64_t* source_offsets, const bool* mask, int64_t last,
    int64_t source_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int64_t selected =
            source_offsets[last] + (mask[last] ? int64_t(1) : int64_t(0));
        assert(selected <= source_size);
    }
}

Tensor masked_scatter_cuda(const Tensor& self, const Tensor& mask, const Tensor& source) {
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "masked_scatter: self and source must have the same dtype");
    }
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_scatter: mask must be bool");
    }
    if (mask.device() != self.device() || source.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "masked_scatter: self, mask, and source must be on the same device");
    }

    Tensor self_iter = self.dim() == 0 ? self.unsqueeze(0) : self;
    Tensor mask_temp = mask.dim() == 0 ? mask.unsqueeze(0) : mask;
    Tensor m_full = mask_temp.expand(
        static_cast<std::vector<int64_t>>(self_iter.shape())).contiguous();
    Tensor src = source.contiguous();
    Tensor result = ::tensorplay::detail::contiguous_clone(self);
    Tensor result_iter = result.dim() == 0 ? result.unsqueeze(0) : result;
    const int64_t n = result_iter.numel();
    if (n == 0) return result;

    TP_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
             "masked_scatter: input is too large for device scan");
    Tensor mask_flat = m_full.reshape({n});
    Tensor flags = Tensor::empty({n}, DType::Int64, self.device());
    Tensor source_offsets = Tensor::empty({n}, DType::Int64, self.device());
    TensorIterator flag_iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(flags)
        .add_const_input(mask_flat)
        .build();
    gpu_kernel(flag_iter, [] __host__ __device__(bool value) -> int64_t {
        return value ? int64_t(1) : int64_t(0);
    });

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, flags.data_ptr<int64_t>(),
        source_offsets.data_ptr<int64_t>(), static_cast<int>(n), stream));
    Tensor scan_storage = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_storage.data_ptr(), scan_bytes, flags.data_ptr<int64_t>(),
        source_offsets.data_ptr<int64_t>(), static_cast<int>(n), stream));

    masked_scatter_size_check<<<1, 1, 0, stream>>>(
        source_offsets.data_ptr<int64_t>(), m_full.data_ptr<bool>(), n - 1,
        src.numel());
    CUDA_CHECK(cudaGetLastError());

    Tensor source_offsets_view = source_offsets.reshape(
        static_cast<std::vector<int64_t>>(result_iter.shape()));
    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(result_iter)
        .add_input(result_iter)
        .add_const_input(m_full)
        .add_input(source_offsets_view)
        .build();

#define TP_MS_CASE(ctype, name) \
    case DType::name: \
        run_masked_scatter_iter<ctype>(iter, src.data_ptr<ctype>()); \
        break;
    switch (result.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MS_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_MS_CASE)
        case DType::ComplexHalf:
            run_masked_scatter_iter<tensorplay::complex<Half>>(
                iter, static_cast<const tensorplay::complex<Half>*>(src.data_ptr()));
            break;
        case DType::ComplexFloat:
            run_masked_scatter_iter<tensorplay::complex<float>>(
                iter, static_cast<const tensorplay::complex<float>*>(src.data_ptr()));
            break;
        case DType::ComplexDouble:
            run_masked_scatter_iter<tensorplay::complex<double>>(
                iter, static_cast<const tensorplay::complex<double>*>(src.data_ptr()));
            break;
        case DType::BComplex32:
            run_masked_scatter_iter<tensorplay::complex<BFloat16>>(
                iter, static_cast<const tensorplay::complex<BFloat16>*>(src.data_ptr()));
            break;
        default: TP_THROW(TypeError, "masked_scatter: unsupported dtype");
    }
#undef TP_MS_CASE
    CUDA_CHECK(cudaGetLastError());
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
    // Radix path: one segmented radix pass orders every slice together.
    // Complexity is O(n * bytes) versus the heapsort fallback's O(n log n)
    // serialized per-slice walk; the fallback remains for tensors beyond
    // the 32-bit size limit of the device primitives.
    // Narrow multi-slice shapes go through the block-per-slice radix sort
    // instead: the slice is staged and ordered entirely in shared memory,
    // which avoids the global scatter/gather of the segmented device pass.
    // A sort dimension that is not the innermost one keeps the same kernel
    // but routes the data through a transposed staging buffer: both the
    // staging pass and the ordered write-back walk contiguous rows, and the
    // final layout fix-up is a single strided copy.
    if (self_c.numel() <= std::numeric_limits<int>::max() &&
        d_size >= 2 && d_size <= 4096 && slices > 1) {
        // Sort the innermost staging rows.  The staging tensor is the input
        // permuted so that the sort dimension is last, giving contiguous
        // slices; for inner == 1 the input already has that shape.
        Tensor staged = self_c;
        Tensor staged_values = values;
        Tensor staged_indices = indices;
        std::vector<int64_t> order;
        if (inner != 1) {
            order.resize(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) order[static_cast<size_t>(d)] = d;
            std::swap(order[static_cast<size_t>(dim)],
                      order[static_cast<size_t>(nd - 1)]);
            staged = self_c.permute(order).contiguous();
            staged_values = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                staged.dtype(), staged.device());
            staged_indices = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                DType::Int64, staged.device());
        }
        sort_block_radix_entry(staged, staged_values, staged_indices,
                               slices, d_size, 1, descending);
        if (inner != 1) {
            // Undo the permutation with one strided copy into the results.
            std::vector<int64_t> inverse(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) {
                inverse[static_cast<size_t>(order[static_cast<size_t>(d)])] = d;
            }
            values.copy_(staged_values.permute(inverse));
            indices.copy_(staged_indices.permute(inverse));
        }
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    if (self_c.numel() <= std::numeric_limits<int>::max()) {
        radix_sort_impl(self_c, values, indices, dim, outer, inner, d_size, slices, descending);
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
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
    const int64_t g = gid_inclusive[i] - 1;
    values[g] = sorted[i];
    if (starts != nullptr) starts[g] = i;
}

__global__ void unique_counts_kernel(int64_t num_groups, int64_t n,
                                     const int64_t* __restrict__ starts,
                                     int64_t* __restrict__ counts) {
    int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= num_groups) return;
    const int64_t end = (g + 1 < num_groups) ? starts[g + 1] : n;
    counts[g] = end - starts[g];
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> unique_cuda(const Tensor& self, bool sorted,
                                               bool return_inverse,
                                               bool return_counts) {
    Tensor flat = self.contiguous().reshape({self.numel()});
    const int64_t n = flat.numel();

    Tensor values = Tensor::empty({0}, self.dtype(), self.device());
    Tensor inverse = return_inverse
                         ? Tensor::empty(
                               static_cast<std::vector<int64_t>>(self.shape()),
                               DType::Int64, self.device())
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
        UNIQUE_FLAGS_CASE(uint16_t, UInt16)
        UNIQUE_FLAGS_CASE(uint32_t, UInt32)
        UNIQUE_FLAGS_CASE(uint64_t, UInt64)
        UNIQUE_FLAGS_CASE(Half, Float16)
        UNIQUE_FLAGS_CASE(BFloat16, BFloat16)
        UNIQUE_FLAGS_CASE(bool, Bool)
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
    Tensor starts;
    int64_t* starts_ptr = nullptr;
    if (return_counts) {
        counts = Tensor::zeros({num_groups}, DType::Int64, self.device());
        starts = Tensor::full({num_groups}, int64_t(-1), DType::Int64,
                              self.device());
        starts_ptr = starts.data_ptr<int64_t>();
    }

    #define UNIQUE_EMIT_CASE(ctype, name)                                      \
        case DType::name:                                                      \
            unique_emit_kernel<ctype><<<blocks, threads>>>(                     \
                n, sorted_vals.data_ptr<ctype>(), flags.data_ptr<int64_t>(),   \
                gid.data_ptr<int64_t>(), values.data_ptr<ctype>(),             \
                starts_ptr);                                                     \
            break;
    switch (self.dtype()) {
        UNIQUE_EMIT_CASE(float, Float32)
        UNIQUE_EMIT_CASE(double, Float64)
        UNIQUE_EMIT_CASE(int64_t, Int64)
        UNIQUE_EMIT_CASE(int32_t, Int32)
        UNIQUE_EMIT_CASE(int16_t, Int16)
        UNIQUE_EMIT_CASE(int8_t, Int8)
        UNIQUE_EMIT_CASE(uint8_t, UInt8)
        UNIQUE_EMIT_CASE(uint16_t, UInt16)
        UNIQUE_EMIT_CASE(uint32_t, UInt32)
        UNIQUE_EMIT_CASE(uint64_t, UInt64)
        UNIQUE_EMIT_CASE(Half, Float16)
        UNIQUE_EMIT_CASE(BFloat16, BFloat16)
        UNIQUE_EMIT_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
    }
    #undef UNIQUE_EMIT_CASE
    CUDA_CHECK(cudaGetLastError());

    if (return_counts) {
        const int count_blocks =
            static_cast<int>((num_groups + threads - 1) / threads);
        unique_counts_kernel<<<count_blocks, threads>>>(
            num_groups, n, starts_ptr, counts.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
    }
    return std::make_tuple(values, inverse, counts);
}

// Two-output flat unique: drops the counts tensor of the three-output path.
std::tuple<Tensor, Tensor> _unique_cuda(const Tensor& self, bool sorted,
                                        bool return_inverse) {
    auto result = unique_cuda(self, sorted, return_inverse, /*return_counts=*/false);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<Tensor, Tensor, Tensor> _unique2_cuda(const Tensor& self, bool sorted,
                                                 bool return_inverse,
                                                 bool return_counts) {
    return unique_cuda(self, sorted, return_inverse, return_counts);
}

// Row equality over the flattened {n, row_len} matrix: two rows match when
// every column pair is equal (a NaN cell never matches another NaN).
template <typename T>
__global__ void unique_row_equal_kernel(int64_t n, int64_t row_len,
                                        const T* rows, int64_t* is_new) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (i == 0) { is_new[0] = 1; continue; }
        const T* cur = rows + i * row_len;
        const T* prev = rows + (i - 1) * row_len;
        int64_t same = 1;
        for (int64_t c = 0; c < row_len; ++c) {
            if (cur[c] != prev[c]) { same = 0; break; }
        }
        is_new[i] = same ? 0 : 1;
    }
}

// Gathers the kept rows into a compact buffer and writes the inverse mapping
// from original row positions to group ids (row order already applied).
template <typename T>
__global__ void unique_row_emit_kernel(int64_t n, int64_t row_len,
                                       const T* rows, const int64_t* order,
                                       const int64_t* gid, T* out,
                                       int64_t* inverse) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t row = order[i];
        const int64_t g = gid[i] - 1;  // inclusive cumsum -> 0-based group id
        const T* src = rows + i * row_len;
        T* dst = out + g * row_len;
        for (int64_t c = 0; c < row_len; ++c) dst[c] = src[c];
        if (inverse != nullptr) inverse[row] = g;
    }
}

// Dim-wise unique.  Rows (slices along `dim`) are sorted lexicographically by
// `sort_cuda` applied to the transposed matrix — sorting each row-position
// column independently is not row order, so instead the sort runs on a
// {row_len, n} layout along the last axis, which orders rows by their
// first column, and repeated stable passes over remaining columns refine the
// order (LSD across columns; each pass must be stable, guaranteed by the
// tie-breaking index in the radix sort).
std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda_impl(const Tensor& self,
                                                        int64_t dim,
                                                        bool consecutive,
                                                        bool return_inverse,
                                                        bool return_counts) {
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(self.shape());
    const int64_t zero_dims = std::count(sizes.begin(), sizes.end(), 0);
    if (self.size(dim) == 0) {
        TP_CHECK(zero_dims == 1,
                 "Number of zero sized dimensions is more than one, so unique "
                 "cannot be applied");
        Tensor values = Tensor::empty(sizes, self.dtype(), self.device());
        Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
        Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
        return std::make_tuple(values, inverse, counts);
    }
    TP_CHECK(zero_dims == 0,
             "There are 0 sized dimensions, and they aren't selected, so "
             "unique cannot be applied");

    Tensor input_flat = self.moveaxis(dim, 0).contiguous();
    std::vector<int64_t> front_sizes =
        static_cast<std::vector<int64_t>>(input_flat.shape());
    const int64_t n = front_sizes[0];
    input_flat = input_flat.reshape({n, -1});
    const int64_t row_len = input_flat.size(1);

    Tensor rows_sorted;
    Tensor order;
    if (consecutive) {
        rows_sorted = input_flat;
        order = Tensor::arange(Scalar(int64_t(0)), Scalar(n), Scalar(int64_t(1)),
                               DType::Int64, self.device());
    } else {
        // LSD refinement: stable-sort rows by each column from last to first.
        // The radix sort's tie-breaking on row index keeps each pass stable.
        rows_sorted = input_flat;
        order = Tensor::arange(Scalar(int64_t(0)), Scalar(n), Scalar(int64_t(1)),
                               DType::Int64, self.device());
        for (int64_t c = row_len - 1; c >= 0; --c) {
            // Sort the current rows by column c: gather the column, sort its
            // (key, row) pairs, and reorder the rows through the permutation.
            Tensor col = rows_sorted.slice(1, c, c + 1).reshape({n});
            Tensor col_sorted, col_order;
            std::tie(col_sorted, col_order) = sort_cuda(col, 0, false);
            // col_order indexes rows within the current rows_sorted layout.
            // Apply the same permutation to the row data and original indices.
            order = order.gather(0, col_order);
            Tensor idx = col_order.reshape({n, 1})
                             .expand(std::vector<int64_t>{n, row_len});
            rows_sorted = rows_sorted.gather(0, idx);
        }
    }

    Tensor flags = Tensor::zeros({n}, DType::Int64, self.device());
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = getCurrentCUDAStream().stream();

#define UNIQUE_ROW_CASE(ctype, name)                                          \
    case DType::name:                                                          \
        unique_row_equal_kernel<ctype><<<blocks, threads, 0, stream>>>(        \
            n, row_len, rows_sorted.data_ptr<ctype>(),                         \
            flags.data_ptr<int64_t>());                                         \
        break;
    switch (self.dtype()) {
        UNIQUE_ROW_CASE(float, Float32)
        UNIQUE_ROW_CASE(double, Float64)
        UNIQUE_ROW_CASE(int64_t, Int64)
        UNIQUE_ROW_CASE(int32_t, Int32)
        UNIQUE_ROW_CASE(int16_t, Int16)
        UNIQUE_ROW_CASE(int8_t, Int8)
        UNIQUE_ROW_CASE(uint8_t, UInt8)
        UNIQUE_ROW_CASE(uint16_t, UInt16)
        UNIQUE_ROW_CASE(uint32_t, UInt32)
        UNIQUE_ROW_CASE(uint64_t, UInt64)
        UNIQUE_ROW_CASE(Half, Float16)
        UNIQUE_ROW_CASE(BFloat16, BFloat16)
        UNIQUE_ROW_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError,
                     "unique_dim: unsupported dtype on CUDA");
    }
#undef UNIQUE_ROW_CASE

    Tensor gid = flags.cumsum(0);
    const int64_t num_groups =
        gid.to(Device(DeviceType::CPU)).data_ptr<int64_t>()[n - 1];

    Tensor kept_rows = Tensor::empty({num_groups, row_len}, self.dtype(),
                                     self.device());
    Tensor inverse = return_inverse
                         ? Tensor::empty({n}, DType::Int64, self.device())
                         : Tensor();
    int64_t* inverse_ptr = return_inverse ? inverse.data_ptr<int64_t>() : nullptr;

#define UNIQUE_ROW_EMIT_CASE(ctype, name)                                     \
    case DType::name:                                                          \
        unique_row_emit_kernel<ctype><<<blocks, threads, 0, stream>>>(         \
            n, row_len, rows_sorted.data_ptr<ctype>(),                         \
            order.data_ptr<int64_t>(), gid.data_ptr<int64_t>(),                \
            kept_rows.data_ptr<ctype>(), inverse_ptr);                           \
        break;
    switch (self.dtype()) {
        UNIQUE_ROW_EMIT_CASE(float, Float32)
        UNIQUE_ROW_EMIT_CASE(double, Float64)
        UNIQUE_ROW_EMIT_CASE(int64_t, Int64)
        UNIQUE_ROW_EMIT_CASE(int32_t, Int32)
        UNIQUE_ROW_EMIT_CASE(int16_t, Int16)
        UNIQUE_ROW_EMIT_CASE(int8_t, Int8)
        UNIQUE_ROW_EMIT_CASE(uint8_t, UInt8)
        UNIQUE_ROW_EMIT_CASE(uint16_t, UInt16)
        UNIQUE_ROW_EMIT_CASE(uint32_t, UInt32)
        UNIQUE_ROW_EMIT_CASE(uint64_t, UInt64)
        UNIQUE_ROW_EMIT_CASE(Half, Float16)
        UNIQUE_ROW_EMIT_CASE(BFloat16, BFloat16)
        UNIQUE_ROW_EMIT_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError,
                     "unique_dim: unsupported dtype on CUDA");
    }
#undef UNIQUE_ROW_EMIT_CASE

    front_sizes[0] = num_groups;
    Tensor values = kept_rows.reshape(front_sizes).moveaxis(0, dim);

    Tensor counts;
    if (return_counts) {
        // counts[g] = number of positions whose gid equals g+1; resolved via
        // a bincount over the shifted gid buffer.
        Tensor one = Tensor::ones({n}, DType::Int64, self.device());
        Tensor shifted = gid.sub(Scalar(int64_t(1)));
        counts = shifted.bincount(one, num_groups);
    }
    return std::make_tuple(values, inverse, counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_cuda(const Tensor& self,
                                                   int64_t dim, bool sorted,
                                                   bool return_inverse,
                                                   bool return_counts) {
    (void)sorted;
    return unique_dim_cuda_impl(self, dim, /*consecutive=*/false,
                                return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_cuda(
        const Tensor& self, int64_t dim, bool return_inverse,
        bool return_counts) {
    return unique_dim_cuda_impl(self, dim, /*consecutive=*/true,
                                return_inverse, return_counts);
}

// ---------------------------------------------------------------------------
// cumsum_backward: reverse scan R[i] = sum_{j>=i} g[j].
// ---------------------------------------------------------------------------

namespace {
template <typename T, typename AccT>
__global__ void cumsum_backward_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                       const T* in, T* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        AccT acc = static_cast<AccT>(0);
        for (int64_t j = d_size - 1; j >= 0; --j) {
            acc += static_cast<AccT>(sp[j * inner]);
            dp[j * inner] = static_cast<T>(acc);
        }
    }
}
} // anonymous namespace

Tensor cumsum_backward_cuda(const Tensor& grad, int64_t dim) {
    int64_t nd = grad.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor g = grad.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), g.dtype(), g.device());
    if (nd == 0) {
        result.copy_(g);
        return result;
    }
    int64_t d_size = g.size(dim);
    if (d_size == 0 || g.numel() == 0) return result;
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(g.shape()), dim, outer, inner);
    int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
#define TP_CSB_CASE(ctype, acc_type, name) \
    case DType::name: \
        cumsum_backward_kernel<ctype, acc_type><<<(slices + kThreads - 1) / kThreads, kThreads, 0, stream>>>( \
            slices, d_size, inner, g.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break;
    switch (g.dtype()) {
        TP_CSB_CASE(uint8_t, uint8_t, UInt8)
        TP_CSB_CASE(int8_t, int8_t, Int8)
        TP_CSB_CASE(int16_t, int16_t, Int16)
        TP_CSB_CASE(int32_t, int32_t, Int32)
        TP_CSB_CASE(int64_t, int64_t, Int64)
        TP_CSB_CASE(uint16_t, uint16_t, UInt16)
        TP_CSB_CASE(uint32_t, uint32_t, UInt32)
        TP_CSB_CASE(uint64_t, uint64_t, UInt64)
        TP_CSB_CASE(float, float, Float32)
        TP_CSB_CASE(double, double, Float64)
        TP_CSB_CASE(Half, float, Float16)
        TP_CSB_CASE(BFloat16, float, BFloat16)
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
Tensor& interop_tril_out_cuda(const Tensor& self, int64_t diagonal, Tensor& out) {
        out = tril_cuda(self, diagonal);
        return out;

}

Tensor& interop_triu_out_cuda(const Tensor& self, int64_t diagonal, Tensor& out) {
        out = triu_cuda(self, diagonal);
        return out;

}

Tensor& interop_index_add_out_cuda(const Tensor& self, int64_t dim, const Tensor& index,
              const Tensor& source, Scalar alpha, Tensor& out) {
        (void)alpha;
        out = index_add_cuda(self, dim, index, source);
        return out;

}

Tensor& interop_index_reduce_out_cuda(const Tensor& self, int64_t dim, const Tensor& index,
              const Tensor& source, std::string reduce, bool include_self,
              Tensor& out) {
        out = index_reduce_cuda(self, dim, index, source, reduce, include_self);
        return out;

}

Tensor& interop_masked_fill__Scalar_cuda(Tensor& self, const Tensor& mask, Scalar value) {
        return masked_fill__cuda(self, mask, value);
    
}

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
    m.impl("_unique", _unique_cuda);
    m.impl("_unique2", _unique2_cuda);
    m.impl("unique_dim", unique_dim_cuda);
    m.impl("unique_dim_consecutive", unique_dim_consecutive_cuda);
    m.impl("searchsorted.Tensor", searchsorted_cuda);
    m.impl("searchsorted.Tensor_out", searchsorted_out_cuda);
    m.impl("searchsorted.Scalar", searchsorted_scalar_cuda);
    m.impl("searchsorted.Scalar_out", searchsorted_scalar_out_cuda);
    m.impl("bucketize.Tensor", bucketize_cuda);
    m.impl("bucketize.Tensor_out", bucketize_out_cuda);
    m.impl("bucketize.Scalar", bucketize_scalar_cuda);
    m.impl("bucketize.Scalar_out", bucketize_scalar_out_cuda);
    m.impl("bincount", bincount_cuda);
    m.impl("take", take_cuda);
    m.impl("masked_scatter", masked_scatter_cuda);

    // out-variants: run the value kernel, then transfer into the caller's
    // buffer.  masked_fill_.Scalar routes through the tensor-overload kernel.
    m.impl("tril.out", interop_tril_out_cuda);
    m.impl("triu.out", interop_triu_out_cuda);
    m.impl("index_add.out", interop_index_add_out_cuda);
    m.impl("index_reduce.out", interop_index_reduce_out_cuda);
    m.impl("masked_fill_.Scalar", interop_masked_fill__Scalar_cuda);
}

} // namespace cuda

// ---------------------------------------------------------------------------
// Reduction functors and indexed reduction helpers. With include_self=False,
// indexed slices are reset to the operation identity before accumulation;
// untouched slices retain their original values. Mean divides by full-rank
// counts, replacing zero counts with one before division.
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
            if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
                return static_cast<T>(std::numeric_limits<float>::infinity());
            } else {
                return std::numeric_limits<T>::has_infinity
                           ? std::numeric_limits<T>::infinity()
                           : std::numeric_limits<T>::max();
            }
        case SrReduceCuda::AMax:
            if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
                return static_cast<T>(-std::numeric_limits<float>::infinity());
            } else {
                return std::numeric_limits<T>::has_infinity
                           ? -std::numeric_limits<T>::infinity()
                           : std::numeric_limits<T>::lowest();
            }
    }
    return static_cast<T>(0);  // unreachable
}

template <typename T>
__device__ __forceinline__ void sr_atomic_reduce(T* addr, T value,
                                                  SrReduceCuda op) {
    switch (op) {
        case SrReduceCuda::Sum:
        case SrReduceCuda::Mean:
            gpuAtomicAdd(addr, value);
            break;
        case SrReduceCuda::Prod:
            gpuAtomicMul(addr, value);
            break;
        case SrReduceCuda::AMin:
            gpuAtomicMin(addr, value);
            break;
        case SrReduceCuda::AMax:
            gpuAtomicMax(addr, value);
            break;
    }
}

template <typename T>
__device__ __forceinline__ T sr_mean_divide_value(T value, int64_t count) {
    if (count == 0 || std::is_same_v<T, bool>) return value;
    if constexpr (std::is_integral_v<T>) {
        const T divisor = static_cast<T>(count);
        T quotient = static_cast<T>(value / divisor);
        if constexpr (std::is_signed_v<T>) {
            const T remainder = static_cast<T>(value - quotient * divisor);
            if (remainder != static_cast<T>(0) && remainder < static_cast<T>(0)) {
                quotient = static_cast<T>(quotient - static_cast<T>(1));
            }
        }
        return quotient;
    } else if constexpr (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>) {
        return static_cast<T>(static_cast<float>(value) /
                              static_cast<float>(count));
    } else {
        return static_cast<T>(value / static_cast<T>(count));
    }
}

template <typename T>
__global__ void sr_mean_divide_kernel(int64_t n, T* data,
                                      const int64_t* counts) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        data[i] = sr_mean_divide_value(data[i], counts[i]);
    }
}

template <bool AllowNegative>
__global__ void sr_validate_indices_kernel(int64_t n, const int64_t* indices,
                                            int64_t size, int32_t* invalid,
                                            int64_t* bad_value) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t value = indices[i];
        const bool out = AllowNegative
            ? (value < -size || value >= size)
            : (value < 0 || value >= size);
        if (out && atomicCAS(invalid, 0, 1) == 0) {
            *bad_value = value;
        }
    }
}

template <bool AllowNegative>
std::optional<int64_t> sr_validate_indices_cuda(const Tensor& indices,
                                                int64_t size) {
    const int64_t n = indices.numel();
    if (n == 0) return std::nullopt;
    Tensor invalid = Tensor::zeros({1}, DType::Int32, indices.device());
    Tensor bad_value = Tensor::zeros({1}, DType::Int64, indices.device());
    auto stream = getCurrentCUDAStream().stream();
    sr_validate_indices_kernel<AllowNegative>
        <<<static_cast<uint32_t>((n + kThreads - 1) / kThreads), kThreads,
           0, stream>>>(n, indices.data_ptr<int64_t>(), size,
                        invalid.data_ptr<int32_t>(),
                        bad_value.data_ptr<int64_t>());
    CUDA_CHECK(cudaGetLastError());
    int32_t invalid_host = 0;
    CUDA_CHECK(cudaMemcpy(&invalid_host, invalid.data_ptr<int32_t>(),
                          sizeof(invalid_host), cudaMemcpyDeviceToHost));
    if (invalid_host == 0) return std::nullopt;
    int64_t bad_host = 0;
    CUDA_CHECK(cudaMemcpy(&bad_host, bad_value.data_ptr<int64_t>(),
                          sizeof(bad_host), cudaMemcpyDeviceToHost));
    return bad_host;
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
    sr_atomic_reduce(&d[dst], v, op);
    if (op == SrReduceCuda::Mean) {
        gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
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
        sr_atomic_reduce(&d[dst], v, op);
        if (op == SrReduceCuda::Mean) {
            gpuAtomicAdd(&cp[dst], static_cast<int64_t>(1));
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
    if (index.numel() != 0 && index.dtype() != DType::Int32 &&
        index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected dtype int32/int64 for index");
    }
    if (src_in.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected self.dtype to be equal to src.dtype");
    }
    if (index.device() != self.device() || src_in.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "scatter_reduce: self, index, and src must be on the same device");
    }
    if (index.numel() == 0) return ::tensorplay::detail::contiguous_clone(self);
    Tensor idx_c = (index.dtype() == DType::Int64)
                       ? index.contiguous()
                       : index.to(DType::Int64).contiguous();
    std::vector<int64_t> idx_shape(
        static_cast<std::vector<int64_t>>(idx_c.shape()));
    Tensor src_b;
    if (src_in.dim() == 0) {
        if (nd != 1 || idx_c.size(0) != 1) {
            TP_THROW(RuntimeError,
                     "src/source shape must match the index shape");
        }
        src_b = src_in.expand(idx_shape).contiguous();
    } else {
        if (src_in.dim() != nd) {
            TP_THROW(IndexError,
                     "src/source must have the same number of dimensions as index");
        }
        for (int64_t i = 0; i < nd; ++i) {
            if (i != dim && idx_c.size(i) > self.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed self shape outside the reduced dimension");
            }
            if (idx_c.size(i) > src_in.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed source shape");
            }
        }
        Tensor src_view = src_in;
        for (int64_t i = 0; i < nd; ++i) {
            if (src_view.size(i) > idx_shape[static_cast<size_t>(i)]) {
                src_view = src_view.narrow(
                    i, 0, idx_shape[static_cast<size_t>(i)]);
            }
        }
        src_b = src_view.contiguous();
    }

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

    if (auto bad_index = sr_validate_indices_cuda<false>(idx_c, self_dim_size)) {
        TP_THROW(IndexError, "index ", *bad_index,
                 " is out of bounds for dimension ", dim,
                 " with size ", self_dim_size);
    }

    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
    const int64_t result_numel = result.numel();
    const int64_t result_blocks =
        result_numel > 0 ? (result_numel + kThreads - 1) / kThreads : 1;

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
        if (op == SrReduceCuda::Mean && result_numel > 0) {                       \
            sr_mean_divide_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(result_blocks), kThreads, 0, stream>>>(  \
                    result_numel, dp, cp);                                       \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SR_CUDA_CASE)
        default:
            TP_THROW(TypeError, "scatter_reduce: unsupported dtype");
    }
#undef TP_SR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

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
    if (index.dim() > 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got dim: ",
                 index.dim());
    }
    if (index.dtype() != DType::Int32 && index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected dtype int32/int64 for index");
    }
    if (source_in.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected self.dtype to be equal to source.dtype");
    }
    if (index.device() != self.device() || source_in.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "index_reduce: self, index, and source must be on the same device");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) continue;
        if (source_in.size(i) != self.size(i)) {
            TP_THROW(IndexError,
                     "index_reduce(): Expected source and self to have the "
                     "same size at dimension ", i);
        }
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
    const int64_t self_dim_size = self.size(dim);
    if (auto bad_index = sr_validate_indices_cuda<false>(idx_c, self_dim_size)) {
        TP_THROW(IndexError, "index ", *bad_index,
                 " is out of bounds for dimension ", dim,
                 " with size ", self_dim_size);
    }
    const int64_t* ip = idx_c.data_ptr<int64_t>();
    const SrReduceCuda op = parse_sr_reduce_cuda(reduce);
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= self.size(i);
    int64_t self_inner = 1;
    for (int64_t i = dim + 1; i < nd; ++i) self_inner *= self.size(i);

    Tensor result = ::tensorplay::detail::contiguous_clone(self);
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
    const int64_t result_numel = result.numel();
    const int64_t result_blocks =
        result_numel > 0 ? (result_numel + kThreads - 1) / kThreads : 1;

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
        if (reduce == "mean" && result_numel > 0) {                               \
            sr_mean_divide_kernel<ctype>                                         \
                <<<static_cast<uint32_t>(result_blocks), kThreads, 0, stream>>>(  \
                    result_numel, dp, cp);                                       \
        }                                                                       \
        break;                                                                   \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_IR_CUDA_CASE)
        default:
            TP_THROW(TypeError, "index_reduce: unsupported dtype");
    }
#undef TP_IR_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());

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
