#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Allocator.h"
#include "Complex.h"
#include "GPUPrimitives.cuh"

#include <cuda_runtime.h>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor flip_cuda(const Tensor& self, const std::vector<int64_t>& dims);

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
    outer = 1;
    inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

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

template <typename T>
constexpr bool is_direct_cub_scan_v =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
constexpr bool is_widen_cub_scan_v =
    std::is_same_v<T, Half> || std::is_same_v<T, BFloat16> ||
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
    std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>;

template <typename T>
using scan_cub_compute_t = std::conditional_t<
    (std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>),
    float,
    std::conditional_t<
        (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
         std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>),
        int32_t, T>>;

template <typename T, typename ComputeT>
__global__ void scan_cast_kernel(int64_t n, const T* in, ComputeT* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = static_cast<ComputeT>(in[i]);
}

template <typename T, typename Op>
__global__ void scan_transform_first_kernel(const T* prev, const T* cur,
                                            T* first, Op op) {
    *first = op(*prev, *cur);
}

template <typename T, typename Op>
__global__ void scan_apply_carry_kernel(int64_t n, const T* carry, T* data,
                                        Op op) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) data[i] = op(*carry, data[i]);
}

// CUB's device scan only accepts int-sized item counts, so oversized flat
// scans proceed in chunks of at most 2^30 items.  Each chunk after the first
// is seeded with the carry produced by the previous chunk: the first element
// of the chunk is `op(previous_last, current_first)`, and the remaining
// elements are an inclusive scan of the rest combined with that carry.
template <typename T, typename Op>
void scan_flat_chunked_with_cub(const Tensor& input, Tensor& output, Op op,
                                cudaStream_t stream) {
    const int64_t n = input.numel();
    constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1;
    const T* in = input.data_ptr<T>();
    T* out = output.data_ptr<T>();
    size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        nullptr, tmp_bytes, in, out, op, max_cub_size, stream));
    auto storage = getAllocator(DeviceType::CUDA)->allocate(
        std::max<size_t>(tmp_bytes, 1), input.device());
    int64_t i = 0;
    while (i < n) {
        const int size_cub =
            static_cast<int>(std::min<int64_t>(n - i, max_cub_size));
        if (i == 0) {
            CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                storage.get(), tmp_bytes, in, out, op, size_cub, stream));
        } else {
            scan_transform_first_kernel<T, Op><<<1, 1, 0, stream>>>(
                out - 1, in, out, op);
            if (size_cub > 1) {
                CUDA_CHECK(cub::DeviceScan::InclusiveScan(
                    storage.get(), tmp_bytes, in + 1, out + 1, op,
                    size_cub - 1, stream));
                const int blocks = static_cast<int>(
                    (size_cub - 1 + kThreads - 1) / kThreads);
                scan_apply_carry_kernel<T, Op><<<blocks, kThreads, 0, stream>>>(
                    size_cub - 1, out, out + 1, op);
            }
        }
        i += size_cub;
        in += size_cub;
        out += size_cub;
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename ComputeT, typename Op>
bool scan_flat_widen_with_cub(const Tensor& input, Tensor& output, Op op,
                              cudaStream_t stream) {
    const int64_t n = input.numel();
    const auto device = input.device();
    const DType compute_dtype =
        std::is_same_v<ComputeT, float> ? DType::Float32 : DType::Int32;
    Tensor compute_in = Tensor::empty({n}, compute_dtype, device);
    Tensor compute_out = Tensor::empty({n}, compute_dtype, device);
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    scan_cast_kernel<T, ComputeT><<<blocks, kThreads, 0, stream>>>(
        n, input.data_ptr<T>(), compute_in.data_ptr<ComputeT>());
    if (n <= std::numeric_limits<int>::max()) {
        if (!scan_flat_with_cub<ComputeT>(compute_in, compute_out, op)) {
            return false;
        }
    } else {
        scan_flat_chunked_with_cub<ComputeT>(compute_in, compute_out, op,
                                             stream);
    }
    scan_cast_kernel<ComputeT, T><<<blocks, kThreads, 0, stream>>>(
        n, compute_out.data_ptr<ComputeT>(), output.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T, typename Op>
bool scan_flat_try_cub(const Tensor& input, Tensor& output, Op op,
                       cudaStream_t stream) {
    if constexpr (is_direct_cub_scan_v<T>) {
        if (input.numel() <= std::numeric_limits<int>::max()) {
            return scan_flat_with_cub<T>(input, output, op);
        }
        scan_flat_chunked_with_cub<T>(input, output, op, stream);
        return true;
    } else if constexpr (is_widen_cub_scan_v<T>) {
        using ComputeT = scan_cub_compute_t<T>;
        return scan_flat_widen_with_cub<T, ComputeT>(input, output, op,
                                                     stream);
    } else {
        return false;
    }
}

template <typename T, bool Product>
bool scan_flat_try_cub_arithmetic(const Tensor& input, Tensor& output,
                                  cudaStream_t stream) {
    if constexpr (is_direct_cub_scan_v<T>) {
        if (input.numel() <= std::numeric_limits<int>::max()) {
            if constexpr (Product) {
                return scan_flat_product_with_cub<T>(input, output);
            } else {
                return scan_flat_sum_with_cub<T>(input, output);
            }
        }
        if constexpr (Product) {
            scan_flat_chunked_with_cub<T>(
                input, output, scan_arithmetic_op<T, true>{}, stream);
        } else {
            scan_flat_chunked_with_cub<T>(
                input, output, scan_arithmetic_op<T, false>{}, stream);
        }
        return true;
    } else if constexpr (is_widen_cub_scan_v<T>) {
        using ComputeT = scan_cub_compute_t<T>;
        const int64_t n = input.numel();
        const auto device = input.device();
        const DType compute_dtype =
            std::is_same_v<ComputeT, float> ? DType::Float32 : DType::Int32;
        Tensor compute_in = Tensor::empty({n}, compute_dtype, device);
        Tensor compute_out = Tensor::empty({n}, compute_dtype, device);
        const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
        scan_cast_kernel<T, ComputeT><<<blocks, kThreads, 0, stream>>>(
            n, input.data_ptr<T>(), compute_in.data_ptr<ComputeT>());
        if (n <= std::numeric_limits<int>::max()) {
            if constexpr (Product) {
                if (!scan_flat_product_with_cub<ComputeT>(compute_in,
                                                          compute_out)) {
                    return false;
                }
            } else {
                if (!scan_flat_sum_with_cub<ComputeT>(compute_in,
                                                      compute_out)) {
                    return false;
                }
            }
        } else {
            if constexpr (Product) {
                scan_flat_chunked_with_cub<ComputeT>(
                    compute_in, compute_out, scan_arithmetic_op<ComputeT, true>{},
                    stream);
            } else {
                scan_flat_chunked_with_cub<ComputeT>(
                    compute_in, compute_out, scan_arithmetic_op<ComputeT, false>{},
                    stream);
            }
        }
        scan_cast_kernel<ComputeT, T><<<blocks, kThreads, 0, stream>>>(
            n, compute_out.data_ptr<ComputeT>(), output.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
        return true;
    } else {
        return false;
    }
}

template <typename T, bool nonzero>
struct scan_transform_functor {
    __host__ __device__ __forceinline__ T operator()(T value) const {
        if constexpr (!nonzero) {
            return value;
        } else {
            return value != T(0) ? T(1) : T(0);
        }
    }
};

template <int Size>
constexpr int scan_deterministic_block_threads() {
    if constexpr (Size >= 16) {
        return 128;
    } else if constexpr (Size >= 8) {
        return 256;
    } else {
        return 512;
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, bool nonzero, typename T>
__global__ void scan_calc_block_sums(const T* d_in, T* agg, int64_t nelem,
                                     int iters_per_cta) {
    using BlockLoad = cub::BlockLoad<
        T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>;
    using BlockReduce = cub::BlockReduce<T, BLOCK_THREADS>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;
    const int64_t offset = static_cast<int64_t>(BLOCK_THREADS) *
        ITEMS_PER_THREAD * iters_per_cta * blockIdx.x;
    int64_t remaining = nelem - offset;
    if (remaining <= 0) return;
    d_in += offset;
    scan_transform_functor<T, nonzero> functor;
    thrust::transform_iterator<scan_transform_functor<T, nonzero>, const T*>
        iter_in(d_in, functor);
    T data[ITEMS_PER_THREAD];
    T agg_val = T(0);
    for (int i = 0; i < iters_per_cta; ++i) {
        if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
            BlockLoad(temp_storage.load).Load(iter_in, data);
            __syncthreads();
            agg_val += BlockReduce(temp_storage.reduce).Sum(data);
        } else {
            BlockLoad(temp_storage.load).Load(iter_in, data, remaining, T(0));
            __syncthreads();
            agg_val += BlockReduce(temp_storage.reduce).Sum(data);
        }
        iter_in += BLOCK_THREADS * ITEMS_PER_THREAD;
        remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
        if (remaining <= 0) {
            if (nonzero && threadIdx.x == 0) agg[blockIdx.x] = agg_val;
            return;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) agg[blockIdx.x] = agg_val;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T>
__global__ void scan_final_kernel(const T* d_in, T* d_out, T* agg,
                                  int64_t nelem, int iters_per_cta) {
const int64_t offset = static_cast<int64_t>(BLOCK_THREADS) *
        ITEMS_PER_THREAD * iters_per_cta * blockIdx.x;
    int64_t remaining = nelem - offset;
    if (remaining <= 0) return;
    d_in += offset;
    d_out += offset;
    using BlockLoad = cub::BlockLoad<
        T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<
        T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScan = cub::BlockScan<
        T, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReduce = cub::BlockReduce<T, BLOCK_THREADS>;
    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockScan::TempStorage scan;
        typename BlockReduce::TempStorage reduce;
    } temp_storage;
    T agg_data = T(0);
    if (threadIdx.x < blockIdx.x) agg_data = agg[threadIdx.x];
    for (unsigned i = threadIdx.x + blockDim.x; i < blockIdx.x;
         i += blockDim.x) {
        agg_data += agg[i];
    }
    T aggregate = BlockReduce(temp_storage.reduce).Sum(agg_data);
    __syncthreads();
    struct scan_prefix_callback {
        T running_total;
        __device__ __forceinline__ T operator()(T block_aggregate) {
            const T old = running_total;
            running_total += block_aggregate;
            return old;
        }
    } prefix_op{aggregate};
    T data[ITEMS_PER_THREAD];
    for (int i = 0; i < iters_per_cta; ++i) {
        if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
            BlockLoad(temp_storage.load).Load(d_in, data);
        } else {
#pragma unroll
            for (int j = 0; j < ITEMS_PER_THREAD; ++j) data[j] = T(0);
            BlockLoad(temp_storage.load).Load(d_in, data, remaining);
        }
        __syncthreads();
        BlockScan(temp_storage.scan).InclusiveSum(data, data, prefix_op);
        __syncthreads();
        if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
            BlockStore(temp_storage.store).Store(d_out, data);
        } else {
            BlockStore(temp_storage.store).Store(d_out, data, remaining);
        }
        d_in += BLOCK_THREADS * ITEMS_PER_THREAD;
        d_out += BLOCK_THREADS * ITEMS_PER_THREAD;
        remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
        if (remaining <= 0) return;
        __syncthreads();
    }
}

template <typename T>
bool scan_flat_deterministic_impl(const Tensor& input, Tensor& output,
                                  cudaStream_t stream) {
    const int64_t n = input.numel();
    constexpr int BLOCK_THREADS =
        scan_deterministic_block_threads<sizeof(T)>();
    constexpr int ITEMS_PER_THREAD = 16;
    int64_t grid_size =
        (n + BLOCK_THREADS * ITEMS_PER_THREAD - 1) /
        (BLOCK_THREADS * ITEMS_PER_THREAD);
    if (grid_size == 0) return true;
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, currentDevice()));
    num_sms = std::max(num_sms, 1);
    const int64_t iters_per_cta = (grid_size + num_sms - 1) / num_sms;
    grid_size = std::min<int64_t>(num_sms, grid_size);
    Tensor agg = Tensor::empty({grid_size}, input.dtype(), input.device());
    scan_calc_block_sums<BLOCK_THREADS, ITEMS_PER_THREAD, false, T>
        <<<static_cast<unsigned>(grid_size), BLOCK_THREADS, 0, stream>>>(
            input.data_ptr<T>(), agg.data_ptr<T>(), n,
            static_cast<int>(iters_per_cta));
    CUDA_CHECK(cudaGetLastError());
    scan_final_kernel<BLOCK_THREADS, ITEMS_PER_THREAD, T>
        <<<static_cast<unsigned>(grid_size), BLOCK_THREADS, 0, stream>>>(
            input.data_ptr<T>(), output.data_ptr<T>(), agg.data_ptr<T>(), n,
            static_cast<int>(iters_per_cta));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename T>
bool scan_flat_deterministic(const Tensor& input, Tensor& output,
                             cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float> ||
                  std::is_same_v<T, double> ||
                  std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, int64_t>) {
        return scan_flat_deterministic_impl<T>(input, output, stream);
    } else if constexpr (std::is_same_v<T, Half> ||
                         std::is_same_v<T, BFloat16>) {
        using ComputeT = float;
        const int64_t n = input.numel();
        Tensor compute_in =
            Tensor::empty({n}, DType::Float32, input.device());
        Tensor compute_out =
            Tensor::empty({n}, DType::Float32, input.device());
        const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
        scan_cast_kernel<T, ComputeT><<<blocks, kThreads, 0, stream>>>(
            n, input.data_ptr<T>(), compute_in.data_ptr<ComputeT>());
        scan_flat_deterministic_impl<ComputeT>(compute_in, compute_out,
                                               stream);
        scan_cast_kernel<ComputeT, T><<<blocks, kThreads, 0, stream>>>(
            n, compute_out.data_ptr<ComputeT>(), output.data_ptr<T>());
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    return false;
}

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

template <typename T>
__host__ __device__ __forceinline__ tensorplay::complex<T> logcumsumexp_complex_pair(
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
struct logcumsumexp_scan_op {
    template <typename AccT>
    __host__ __device__ AccT operator()(AccT lhs, AccT rhs) const {
        const bool lhs_nan = ::isnan(lhs);
        const bool rhs_nan = ::isnan(rhs);
        const AccT min_value = rhs_nan ? rhs : (lhs_nan ? lhs : (lhs < rhs ? lhs : rhs));
        const AccT max_value = rhs_nan ? rhs : (lhs_nan ? lhs : (lhs > rhs ? lhs : rhs));
        if (min_value != max_value || ::isfinite(min_value)) {
            return ::log1p(::exp(min_value - max_value)) + max_value;
        }
        return lhs;
    }
};

template <typename T>
struct logcumsumexp_complex_op {
    __host__ __device__ tensorplay::complex<T> operator()(
            tensorplay::complex<T> lhs, tensorplay::complex<T> rhs) const {
        return logcumsumexp_complex_pair(lhs, rhs);
    }
};

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
    if (inner == 1 && d_size >= 2 && d_size < 512) {
        launch_short_rows_scan<T>(outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op, stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    if (inner == 1 && outer == 1 && d_size >= 512 && d_size <= 8192) {
        if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, false>>) {
            if (globalContext().deterministicAlgorithms() &&
                (std::is_same_v<T, float> ||
                 std::is_same_v<T, double> ||
                 std::is_same_v<T, Half> ||
                 std::is_same_v<T, BFloat16>)) {
                if (scan_flat_deterministic<T>(self_c, result, stream)) {
                    return result;
                }
            }
            if (scan_flat_try_cub_arithmetic<T, false>(self_c, result,
                                                       stream)) {
                return result;
            }
        } else if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, true>>) {
            if (scan_flat_try_cub_arithmetic<T, true>(self_c, result,
                                                      stream)) {
                return result;
            }
        } else if (scan_flat_try_cub<T>(self_c, result, op, stream)) {
            return result;
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
            if (globalContext().deterministicAlgorithms() &&
                (std::is_same_v<T, float> ||
                 std::is_same_v<T, double> ||
                 std::is_same_v<T, Half> ||
                 std::is_same_v<T, BFloat16>)) {
                if (scan_flat_deterministic<T>(self_c, result, stream)) {
                    return result;
                }
            }
            if (scan_flat_try_cub_arithmetic<T, false>(self_c, result,
                                                       stream)) {
                return result;
            }
        } else if constexpr (std::is_same_v<Op, scan_arithmetic_op<T, true>>) {
            if (scan_flat_try_cub_arithmetic<T, true>(self_c, result,
                                                      stream)) {
                return result;
            }
        } else if (scan_flat_try_cub<T>(self_c, result, op, stream)) {
            return result;
        }
    }
    if (inner == 1 && d_size >= 512) {
        const int64_t blocks = std::min<int64_t>(outer, 4096);
        if constexpr (std::is_same_v<T, tensorplay::complex<float>> ||
                      std::is_same_v<T, tensorplay::complex<double>>) {
            constexpr int kScanBlockThreads = 512;
            scan_single_row_block_kernel<T, Op, kScanBlockThreads><<<
                static_cast<unsigned>(blocks), kScanBlockThreads,
                static_cast<size_t>(2) * kScanBlockThreads * sizeof(scan_accum_t<T>),
                stream>>>(outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(),
                          init_val, op);
        } else {
            constexpr int kWarpThreads = 32;
            scan_row_kernel<T, Op><<<static_cast<unsigned>(blocks), kWarpThreads, 0, stream>>>(
                outer, d_size, self_c.data_ptr<T>(), result.data_ptr<T>(), init_val, op);
        }
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

}

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
            return scan_entry<tensorplay::complex<double>>(
                       compute_src, dim, tensorplay::complex<double>(0, 0),
                       scan_arithmetic_op<tensorplay::complex<double>, false>{})
                .to(out_dtype);
        }
        return scan_entry<tensorplay::complex<float>>(
                   compute_src, dim, tensorplay::complex<float>(0, 0),
                   scan_arithmetic_op<tensorplay::complex<float>, false>{})
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
            return scan_entry<tensorplay::complex<double>>(
                       compute_src, dim, tensorplay::complex<double>(1, 0),
                       scan_arithmetic_op<tensorplay::complex<double>, true>{})
                .to(out_dtype);
        }
        return scan_entry<tensorplay::complex<float>>(
                   compute_src, dim, tensorplay::complex<float>(1, 0),
                   scan_arithmetic_op<tensorplay::complex<float>, true>{})
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
    if (nd == 0) {
        Tensor result = Tensor::empty({}, out_dtype, src.device());
        switch (out_dtype) {
            case DType::Float32:
            case DType::Float64:
            case DType::Float16:
            case DType::BFloat16:
            case DType::ComplexFloat:
            case DType::ComplexDouble:
                result.copy_(src);
                return result;
            default:
                TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
        }
    }
    if (isComplexType(out_dtype)) {
        const DType compute_dtype =
            out_dtype == DType::ComplexDouble ? DType::ComplexDouble : DType::ComplexFloat;
        Tensor compute_src = src.dtype() == compute_dtype ? src : src.to(compute_dtype);
        if (compute_dtype == DType::ComplexDouble) {
            return scan_entry<tensorplay::complex<double>>(
                compute_src, dim,
                tensorplay::complex<double>(-std::numeric_limits<double>::infinity(), 0.0),
                logcumsumexp_complex_op<double>{}).to(out_dtype);
        }
        return scan_entry<tensorplay::complex<float>>(
            compute_src, dim,
            tensorplay::complex<float>(-std::numeric_limits<float>::infinity(), 0.0f),
            logcumsumexp_complex_op<float>{}).to(out_dtype);
    }
#define TP_LC_CASE(ctype, name) \
    case DType::name: \
        return scan_entry<ctype>( \
            src, dim, static_cast<ctype>(-std::numeric_limits<float>::infinity()), \
            logcumsumexp_scan_op<ctype>{});
    switch (out_dtype) {
        TP_LC_CASE(float, Float32)
        case DType::Float64:
            return scan_entry<double>(
                src, dim, -std::numeric_limits<double>::infinity(),
                logcumsumexp_scan_op<double>{});
        TP_LC_CASE(Half, Float16)
        TP_LC_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "logcumsumexp: unsupported dtype");
    }
#undef TP_LC_CASE
}

Tensor cumsum_backward_cuda(const Tensor& grad, int64_t dim) {
    int64_t nd = grad.dim();
    dim = wrap_scan_dim(dim, nd);
    Tensor g = grad.contiguous();
    if (nd == 0) {
        Tensor result = Tensor::empty({}, g.dtype(), g.device());
        result.copy_(g);
        return result;
    }
    int64_t d_size = g.size(dim);
    if (d_size == 0 || g.numel() == 0) {
        return Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()),
                             g.dtype(), g.device());
    }
    // A reverse inclusive scan along `dim`: flip, scan forward, flip back.
    // This runs the same parallel engines as cumsum instead of walking each
    // slice serially.
    Tensor flipped = flip_cuda(g, {dim});
    Tensor scanned;
#define TP_CSB_CASE(ctype, name) \
    case DType::name: \
        scanned = scan_entry<ctype>(flipped, dim, static_cast<ctype>(0), \
                                    scan_arithmetic_op<ctype, false>{}); \
        break;
    switch (g.dtype()) {
        TP_CSB_CASE(uint8_t, UInt8)
        TP_CSB_CASE(int8_t, Int8)
        TP_CSB_CASE(int16_t, Int16)
        TP_CSB_CASE(int32_t, Int32)
        TP_CSB_CASE(int64_t, Int64)
        TP_CSB_CASE(uint16_t, UInt16)
        TP_CSB_CASE(uint32_t, UInt32)
        TP_CSB_CASE(uint64_t, UInt64)
        TP_CSB_CASE(float, Float32)
        TP_CSB_CASE(double, Float64)
        TP_CSB_CASE(Half, Float16)
        TP_CSB_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "cumsum_backward: unsupported dtype");
    }
#undef TP_CSB_CASE
    return flip_cuda(scanned, {dim});
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ScanKernels) {
    m.impl("cumsum", cumsum_cuda);
    m.impl("cumsum_backward", cumsum_backward_cuda);
    m.impl("cumprod", cumprod_cuda);
    m.impl("logcumsumexp", logcumsumexp_cuda);
}

}
}
