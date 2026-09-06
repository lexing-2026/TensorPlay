#pragma once

// TensorPlay's CUDA reduction engine.
//
//   * reduced dimensions are mapped to block.x / block.y;
//   * warp shuffle handles the intra-warp tree;
//   * shared memory handles inter-warp reduction;
//   * a small number of independent accumulators hides the add/mul latency;
//   * large reductions can be split across CTAs and finalized by a second
//     kernel.

#include "TensorIterator.h"
#include "CUDARuntime.h"
#include "Complex.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>

namespace tensorplay {
namespace cuda {
namespace reduction {

constexpr int kWarpSize = 32;
constexpr int kMaxReduceDims = 64;
constexpr int kMaxReduceThreads = 512;
constexpr int kDefaultValuesPerThread = 4;
constexpr int kMaxCachedReduceDevices = 64;
// Bump when the header-only launch path changes; this also keeps generated
// CUDA objects from silently reusing an older reduction implementation.
constexpr int kReductionEngineRevision = 4;

// Per-device launch geometry, queried once via cudaDeviceGetAttribute and
// cached: cudaGetDeviceProperties costs ~1ms per call on the target GPU and
// this struct feeds the global-reduce CTA-count decision on every launch.
struct DeviceReduceProps {
    int multi_processor_count = 0;
    int max_threads_per_sm = 0;
};

inline DeviceReduceProps query_reduce_device_props(int device) {
    DeviceReduceProps props;
    if (cudaDeviceGetAttribute(&props.multi_processor_count,
                               cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess ||
        props.multi_processor_count <= 0) {
        props.multi_processor_count = 128;
    }
    if (cudaDeviceGetAttribute(&props.max_threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor,
                               device) != cudaSuccess ||
        props.max_threads_per_sm <= 0) {
        props.max_threads_per_sm = 2048;
    }
    cudaGetLastError();  // clear any attribute-query error above
    return props;
}

inline const DeviceReduceProps& reduce_device_props(int device) {
    static std::array<DeviceReduceProps, kMaxCachedReduceDevices> cache;
    static std::array<std::once_flag, kMaxCachedReduceDevices> flags;
    if (device < 0 || device >= kMaxCachedReduceDevices) {
        // Uncached fallback for out-of-range device indices; same geometry.
        static const DeviceReduceProps fallback = query_reduce_device_props(device);
        return fallback;
    }
    std::call_once(flags[device],
                   [device] { cache[device] = query_reduce_device_props(device); });
    return cache[device];
}

template <typename T>
struct is_half_like : std::false_type {};

template <>
struct is_half_like<Half> : std::true_type {};

template <>
struct is_half_like<BFloat16> : std::true_type {};

template <typename T>
inline constexpr bool is_half_like_v = is_half_like<T>::value;

template <typename T>
struct default_accumulation_type {
    using type = T;
};

template <>
struct default_accumulation_type<Half> {
    using type = float;
};

template <>
struct default_accumulation_type<BFloat16> {
    using type = float;
};

template <>
struct default_accumulation_type<bool> {
    using type = int;
};

template <>
struct default_accumulation_type<uint8_t> {
    using type = int64_t;
};

template <>
struct default_accumulation_type<int8_t> {
    using type = int64_t;
};

template <>
struct default_accumulation_type<int16_t> {
    using type = int64_t;
};

template <>
struct default_accumulation_type<uint16_t> {
    using type = int64_t;
};

template <>
struct default_accumulation_type<int32_t> {
    using type = int64_t;
};

template <>
struct default_accumulation_type<uint32_t> {
    using type = uint64_t;
};

template <typename T>
using default_accumulation_t = typename default_accumulation_type<T>::type;

template <typename T>
__device__ __forceinline__ bool reduce_isnan(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return ::isnan(value);
    } else if constexpr (std::is_same_v<T, double>) {
        return ::isnan(value);
    } else {
        (void)value;
        return false;
    }
}

template <typename T>
__device__ __forceinline__ T reduce_warp_shuffle_down(
        T value, unsigned long long mask, int offset) {
    return __shfl_down_sync(mask, value, offset);
}

// Complex values have no intrinsic __shfl_down_sync overload; shuffle their
// components independently.
__device__ __forceinline__ tensorplay::complex<float> reduce_warp_shuffle_down(
        tensorplay::complex<float> value, unsigned long long mask, int offset) {
    float re = __shfl_down_sync(mask, value.real(), offset);
    float im = __shfl_down_sync(mask, value.imag(), offset);
    return tensorplay::complex<float>(re, im);
}

__device__ __forceinline__ tensorplay::complex<double> reduce_warp_shuffle_down(
        tensorplay::complex<double> value, unsigned long long mask, int offset) {
    double re = __shfl_down_sync(mask, value.real(), offset);
    double im = __shfl_down_sync(mask, value.imag(), offset);
    return tensorplay::complex<double>(re, im);
}

template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
    T val[N];
};

template <typename T>
__host__ __device__ inline T reduction_lower_bound() {
    if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
__host__ __device__ inline T reduction_upper_bound() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T>
struct ArgPair {
    T value;
    int64_t index;
};

template <typename T>
__device__ __forceinline__ ArgPair<T> reduce_warp_shuffle_down(
        ArgPair<T> value, unsigned long long mask, int offset) {
    return {
        reduce_warp_shuffle_down(value.value, mask, offset),
        reduce_warp_shuffle_down(value.index, mask, offset)};
}

template <typename T>
struct WelfordData {
    T mean;
    T m2;
    int64_t n;
    T nf;
};

template <typename T>
__device__ __forceinline__ WelfordData<T> reduce_warp_shuffle_down(
        WelfordData<T> value, unsigned long long mask, int offset) {
    return {
        reduce_warp_shuffle_down(value.mean, mask, offset),
        reduce_warp_shuffle_down(value.m2, mask, offset),
        reduce_warp_shuffle_down(value.n, mask, offset),
        reduce_warp_shuffle_down(value.nf, mask, offset)};
}

struct ReduceConfig {
    int ndim = 0;
    int num_reduce_dims = 0;
    int64_t shape[kMaxReduceDims] = {};
    int64_t input_strides[kMaxReduceDims] = {};
    int64_t output_strides[kMaxReduceDims] = {};

    int64_t num_inputs = 1;
    int64_t num_input_units = 1;
    int64_t num_outputs = 1;
    int64_t step_input = 1;
    int64_t step_output = 1;
    int input_mult[3] = {0, 0, 0};
    int output_mult[2] = {0, 0};

    int block_width = kWarpSize;
    int block_height = 1;
    int num_threads = kWarpSize;
    int input_vec_size = 1;
    bool vectorize_input = false;
    bool global_reduce = false;
    int ctas_per_output = 1;

    __host__ int split_input(int parallelism) {
        int old_step = static_cast<int>(step_input);
        step_input *= parallelism;
        return old_step;
    }

    __host__ int split_output(int parallelism) {
        int old_step = static_cast<int>(step_output);
        step_output *= parallelism;
        return old_step;
    }

    __host__ __device__ bool should_block_x_reduce() const {
        return input_mult[0] != 0;
    }

    __host__ __device__ bool should_block_y_reduce() const {
        return input_mult[1] != 0;
    }

    __host__ __device__ int64_t input_idx() const {
        return static_cast<int64_t>(threadIdx.x) * input_mult[0] +
            static_cast<int64_t>(threadIdx.y) * input_mult[1] +
            static_cast<int64_t>(blockIdx.y) * input_mult[2];
    }

    __host__ __device__ int64_t output_idx() const {
        return static_cast<int64_t>(blockIdx.x) * step_output +
            static_cast<int64_t>(threadIdx.x) * output_mult[0] +
            static_cast<int64_t>(threadIdx.y) * output_mult[1];
    }

    __device__ __forceinline__ int64_t input_base_offset(int64_t output) const {
        int64_t offset = 0;
        int64_t remainder = output;
        // Dimension zero is the fastest-moving dimension after TensorIterator
        // reorders dimensions. Decode in that order.
        for (int dim = num_reduce_dims; dim < ndim; ++dim) {
            const int64_t extent = shape[dim];
            const int64_t coordinate = extent > 0 ? remainder % extent : 0;
            remainder = extent > 0 ? remainder / extent : 0;
            offset += coordinate * input_strides[dim];
        }
        return offset;
    }

    __device__ __forceinline__ int64_t output_offset(int64_t output) const {
        int64_t offset = 0;
        int64_t remainder = output;
        for (int dim = num_reduce_dims; dim < ndim; ++dim) {
            const int64_t extent = shape[dim];
            const int64_t coordinate = extent > 0 ? remainder % extent : 0;
            remainder = extent > 0 ? remainder / extent : 0;
            offset += coordinate * output_strides[dim];
        }
        return offset;
    }

    __device__ __forceinline__ int64_t input_offset(int64_t index) const {
        if (num_reduce_dims == 0) return 0;
        if (num_reduce_dims == 1) return index * input_strides[0];

        int64_t offset = 0;
        int64_t remainder = index;
        for (int dim = 0; dim < num_reduce_dims; ++dim) {
            const int64_t extent = shape[dim];
            const int64_t coordinate = extent > 0 ? remainder % extent : 0;
            remainder = extent > 0 ? remainder / extent : 0;
            offset += coordinate * input_strides[dim];
        }
        return offset;
    }

    __host__ __device__ bool should_store(int64_t output) const {
        return output < num_outputs &&
            (!should_block_x_reduce() || threadIdx.x == 0) &&
            (!should_block_y_reduce() || threadIdx.y == 0);
    }

    __host__ int shared_memory_size(size_t element_size) const {
        if (!should_block_y_reduce() &&
            (!should_block_x_reduce() || block_width <= kWarpSize)) {
            return 0;
        }
        return static_cast<int>(element_size * static_cast<size_t>(num_threads));
    }
};

inline int reduction_last_pow2(int64_t value) {
    if (value <= 1) return 1;
    int result = 1;
    while (result <= value / 2 && result < kMaxReduceThreads) result <<= 1;
    return result;
}

inline bool reduction_pointer_aligned(const TensorIterator& iter, size_t bytes) {
    if (bytes == 0) return false;
    const auto address = reinterpret_cast<uintptr_t>(iter.data_ptr(1));
    return address % bytes == 0;
}

template <typename InputT, typename AccT, typename OutputT>
inline ReduceConfig make_reduce_config(const TensorIterator& iter) {
    ReduceConfig config;
    config.ndim = iter.ndim();
    config.num_reduce_dims = iter.num_reduce_dims();
    if (config.ndim > kMaxReduceDims) {
        TP_THROW(NotImplementedError, "CUDA reduction supports at most 64 dimensions");
    }

    config.num_outputs = iter.num_output_elements();
    config.num_inputs = config.num_outputs == 0
        ? 0
        : iter.numel() / config.num_outputs;
    config.num_input_units = config.num_inputs;

    for (int dim = 0; dim < config.ndim; ++dim) {
        config.shape[dim] = iter.shape()[dim];
        const int64_t input_stride_bytes = iter.strides(1)[dim];
        const int64_t output_stride_bytes = iter.strides(0)[dim];
        config.input_strides[dim] = input_stride_bytes / static_cast<int64_t>(sizeof(InputT));
        config.output_strides[dim] = output_stride_bytes / static_cast<int64_t>(sizeof(OutputT));
    }

    if (config.ndim == 0) {
        config.num_inputs = 1;
        config.num_input_units = 1;
        config.num_outputs = 1;
    }

    const bool reduction_on_fastest_dimension =
        config.ndim == 0 ||
        config.num_reduce_dims == config.ndim ||
        (config.num_reduce_dims > 0 &&
         iter.strides(1)[0] < iter.strides(1)[config.num_reduce_dims]);

    int64_t dim0 = reduction_on_fastest_dimension
        ? config.num_inputs : config.num_outputs;
    int64_t dim1 = reduction_on_fastest_dimension
        ? config.num_outputs : config.num_inputs;

    if (reduction_on_fastest_dimension && config.num_reduce_dims == 1 &&
        config.input_strides[0] == 1 && config.num_inputs >= 128) {
        // vec=8 instantiations triple reduce_kernel PTX for negligible gain.
        config.input_vec_size = 4;
        const size_t vector_bytes = sizeof(InputT) * static_cast<size_t>(config.input_vec_size);
        bool aligned = reduction_pointer_aligned(iter, vector_bytes);
        for (int dim = config.num_reduce_dims; dim < config.ndim; ++dim) {
            aligned = aligned &&
                (config.input_strides[dim] % config.input_vec_size == 0);
        }
        if (aligned) {
            config.vectorize_input = true;
            config.num_input_units =
                (config.num_inputs + config.input_vec_size - 1) / config.input_vec_size;
            dim0 = config.num_input_units;
        } else {
            config.input_vec_size = 1;
        }
    }

    const int max_threads = sizeof(AccT) > 4 ? 256 : kMaxReduceThreads;
    const int max_height = std::max(1, max_threads / kWarpSize);
    // Keeping block.x at a full warp makes the CUDA warp layout independent of
    // block.y. Threads beyond a short reduction simply carry the identity.
    config.block_width = kWarpSize;
    int desired_height = static_cast<int>(reduction_last_pow2(dim1));
    // Global-reduce prediction: a single-output (dim1 == 1) reduction large
    // enough to trigger the multi-CTA branch below runs with a taller block —
    // 8 warps share one CTA's completion-counter slot and staging partial,
    // cutting same-address atomic traffic and the last-CTA fold length 8x
    // The 16384-element floor guarantees the warp-split below actually
    // engages (input_mult[1] != 0), keeping output_mult clean for the gate.
    if (reduction_on_fastest_dimension && dim1 == 1 && config.num_inputs >= 16384) {
        desired_height = std::min(8, max_height);
    }
    config.block_height = std::min(desired_height, max_height);
    config.block_height = std::max(1, config.block_height);
    config.num_threads = config.block_width * config.block_height;

    if (reduction_on_fastest_dimension || config.ndim == 0) {
        config.input_mult[0] = config.split_input(config.block_width);
    } else {
        config.output_mult[0] = config.split_output(config.block_width);
    }

    const int64_t values_per_thread =
        (config.num_input_units + config.step_input - 1) / config.step_input;
    const int64_t warp_split_threshold =
        std::min<int64_t>(static_cast<int64_t>(config.block_height) * 16, 256);
    const bool split_across_warps = config.block_height > 1 &&
        values_per_thread >= warp_split_threshold;

    if (split_across_warps) {
        config.input_mult[1] = config.split_input(config.block_height);
    } else if (config.block_height > 1) {
        config.output_mult[1] = config.split_output(config.block_height);
    }

    // The generic TensorIterator path handles the usual case. For a very long
    // reduction with too few outputs, use more CTAs per output, matching the
    // CTA count is std::clamp'd between the SM-balanced target grid and
    // values_per_thread / {min,max}_values_per_thread so the whole machine
    // stays busy while each thread still reduces a useful number of elements.
    // This branch is restricted to one output per block so the partial buffer
    // has a simple layout.
    if (reduction_on_fastest_dimension &&
        config.output_mult[0] == 0 && config.output_mult[1] == 0 &&
        config.num_outputs > 0) {
        // (= div_up(num_inputs, step_input) in elements): num_input_units
        // counts vectorized units, so a unit-based count would under-report
        // by InputVecSize for vectorized loads.
        const int64_t values_per_thread_elems =
            (config.num_inputs + config.step_input - 1) / config.step_input;
        if (values_per_thread_elems >= 256) {
            int device = -1;
            checkCuda(cudaGetDevice(&device), "cudaGetDevice");
            // Geometry comes from the per-device cache below: querying
            // cudaGetDeviceProperties here — on EVERY launch of every global
            // reduction — costs ~0.9-1.5ms on the target GPU (the same pathology
            // the Muon norm2 path hit; see ReductionKernels.cu), dwarfing a 20us
            // kernel.  cudaDeviceGetAttribute is served from the runtime's own
            // cache and the results are immutable per device.
            const auto& properties = reduce_device_props(device);
            const int blocks_per_sm = std::max(1, properties.max_threads_per_sm /
                                                    config.num_threads);
            const int target_grid = std::max(1, properties.multi_processor_count * blocks_per_sm);
            // scheduled output block), ctas2/ctas3 bound the split so each
            // thread keeps >= min_values_per_thread(16) elements but no more
            // than max_values_per_thread(256).
            const int64_t grid_x = (config.num_outputs + config.step_output - 1) /
                                   config.step_output;
            const int64_t ctas_per_output1 = (target_grid + grid_x - 1) / grid_x;
            const int64_t ctas_per_output2 = (values_per_thread_elems + 15) / 16;
            const int64_t ctas_per_output3 = (values_per_thread_elems + 255) / 256;
            int64_t ctas = ctas_per_output1;
            if (ctas < ctas_per_output3) ctas = ctas_per_output3;
            if (ctas > ctas_per_output2) ctas = ctas_per_output2;
            ctas = std::min<int64_t>(ctas, 65535);  // gridDim.y hardware limit
            if (ctas > 1) {
                config.ctas_per_output = static_cast<int>(ctas);
                config.input_mult[2] = config.split_input(config.ctas_per_output);
                config.global_reduce = true;
            }
        }
    }

    return config;
}

template <typename AccT, typename Ops>
__device__ __forceinline__ AccT block_x_reduce(
        AccT value, AccT identity, const ReduceConfig& config, Ops ops, AccT* shared) {
    const int lane = threadIdx.x;
    const int row_base = threadIdx.y * blockDim.x;
    if (config.block_width > kWarpSize) {
        shared[row_base + lane] = value;
        for (int offset = config.block_width / 2; offset >= kWarpSize; offset >>= 1) {
            __syncthreads();
            if (lane < offset && lane + offset < config.block_width) {
                value = ops.combine(value, shared[row_base + lane + offset]);
                shared[row_base + lane] = value;
            }
        }
        __syncthreads();
        value = lane < kWarpSize ? shared[row_base + lane] : identity;
    }

    // block.x is always a full warp in this implementation, so every shuffle
    // operates within one logical reduction row.
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = ops.combine(value,
            reduce_warp_shuffle_down(value, 0xffffffffu, offset));
    }
    return value;
}

template <typename AccT, typename Ops>
__device__ __forceinline__ AccT block_y_reduce(
        AccT value, const ReduceConfig& config, Ops ops, AccT* shared) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared[tid] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset) {
            value = ops.combine(value,
                shared[(threadIdx.y + offset) * blockDim.x + threadIdx.x]);
            shared[tid] = value;
        }
    }
    __syncthreads();
    return value;
}

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize>
struct ReduceOp {
    ReduceConfig config;
    const InputT* input;
    OutputT* output;
    AccT* partials;
    unsigned long long* counters;
    unsigned long long* flags;
    unsigned long long tag;
    AccT identity;
    Ops ops;

    __device__ __forceinline__ AccT reduce_unit(
            AccT value, const InputT* row, int64_t unit, int64_t logical_base) const {
        if constexpr (InputVecSize == 1) {
            if (logical_base < config.num_inputs) {
                value = ops.reduce(value, row[config.input_offset(logical_base)], logical_base);
            }
        } else if (logical_base + InputVecSize <= config.num_inputs &&
                   config.input_strides[0] == 1 && config.vectorize_input) {
            using Vec = aligned_vector<InputT, InputVecSize>;
            const Vec loaded = *reinterpret_cast<const Vec*>(row + logical_base);
            #pragma unroll
            for (int i = 0; i < InputVecSize; ++i) {
                value = ops.reduce(value, loaded.val[i], logical_base + i);
            }
        } else {
            for (int i = 0; i < InputVecSize; ++i) {
                const int64_t logical = logical_base + i;
                if (logical < config.num_inputs) {
                    value = ops.reduce(value, row[config.input_offset(logical)], logical);
                }
            }
        }
        (void)unit;
        return value;
    }

    // thread with loop-invariant state and purely affine addressing, so the
    // multi-dim div/mod offset decomposition never lands inside an unrolled
    // region. Only the rare ragged tail goes through input_offset.
    __device__ __forceinline__ AccT thread_reduce(int64_t output_index) const {
        AccT values[ValuesPerThread];
        #pragma unroll
        for (int i = 0; i < ValuesPerThread; ++i) values[i] = identity;

        const int64_t start = config.input_idx();
        const int64_t step = config.step_input;
        const int64_t end = config.num_input_units;
        const int64_t base = config.input_base_offset(output_index);
        const InputT* row = input + base;
        using Vec = aligned_vector<InputT, InputVecSize>;
        // Branchless fast path: when every unit maps to a full aligned vector
        // (num_inputs divisible by the vector width, unit strides keep vector
        // alignment — host-side config checks guarantee both), the hot loop
        // thread_reduce loop shape.
        const bool can_vec = InputVecSize > 1 &&
            config.input_strides[0] == 1 && config.vectorize_input;
        const bool can_vec_full = can_vec && config.num_inputs % InputVecSize == 0;

        if (can_vec_full) {
            int64_t unit = start;
            while (unit + static_cast<int64_t>(ValuesPerThread - 1) * step < end) {
                #pragma unroll
                for (int i = 0; i < ValuesPerThread; ++i) {
                    const int64_t logical_base =
                        (unit + static_cast<int64_t>(i) * step) * InputVecSize;
                    const Vec loaded = *reinterpret_cast<const Vec*>(row + logical_base);
                    #pragma unroll
                    for (int j = 0; j < InputVecSize; ++j) {
                        values[i] = ops.reduce(values[i], loaded.val[j], logical_base + j);
                    }
                }
                unit += step * ValuesPerThread;
            }
            while (unit < end) {
                const int64_t logical_base = unit * InputVecSize;
                const Vec loaded = *reinterpret_cast<const Vec*>(row + logical_base);
                #pragma unroll
                for (int j = 0; j < InputVecSize; ++j) {
                    values[0] = ops.reduce(values[0], loaded.val[j], logical_base + j);
                }
                unit += step;
            }
        } else {
            int64_t unit = start;
            while (unit + static_cast<int64_t>(ValuesPerThread - 1) * step < end) {
                #pragma unroll
                for (int i = 0; i < ValuesPerThread; ++i) {
                    const int64_t current = unit + static_cast<int64_t>(i) * step;
                    const int64_t logical_base = current * InputVecSize;
                    if (can_vec && logical_base + InputVecSize <= config.num_inputs) {
                        const Vec loaded = *reinterpret_cast<const Vec*>(row + logical_base);
                        #pragma unroll
                        for (int j = 0; j < InputVecSize; ++j) {
                            values[i] = ops.reduce(values[i], loaded.val[j], logical_base + j);
                        }
                    } else {
                        for (int j = 0; j < InputVecSize; ++j) {
                            const int64_t logical = logical_base + j;
                            if (logical < config.num_inputs) {
                                values[i] = ops.reduce(
                                    values[i], row[config.input_offset(logical)], logical);
                            }
                        }
                    }
                }
                unit += step * ValuesPerThread;
            }
            while (unit < end) {
                values[0] = reduce_unit(values[0], row, unit, unit * InputVecSize);
                // Threads stride by step_input; a plain ++unit makes every lane
                // walk into its neighbours' units (each element counted
                // (num_inputs - lane) times -> triangular sums).
                unit += step;
            }
        }

        #pragma unroll
        for (int i = 1; i < ValuesPerThread; ++i) {
            values[0] = ops.combine(values[0], values[i]);
        }
        return values[0];
    }

    __device__ __forceinline__ void run() {
        extern __shared__ unsigned char shared_raw[];
        AccT* shared = reinterpret_cast<AccT*>(shared_raw);
        const bool block_leader = threadIdx.x == 0 && threadIdx.y == 0;
        // Zero the per-output completion counter for THIS launch and publish
        // the unique launch tag before any work: peers later check the tag
        // (once, right before their single atomicAdd), so the counter is
        // per-launch cudaMemsetAsync (a ~1us GPU stream op per reduction)
        // with an in-kernel initialization.
        if (config.global_reduce &&
            blockIdx.y == 0 && block_leader) {
            counters[blockIdx.x] = 0;
            __threadfence();  // counter zeroed before the tag is published
            *(volatile unsigned long long*)(flags + blockIdx.x) = tag;
        }
        const int64_t output_index = config.output_idx();
        AccT value = identity;
        __shared__ bool is_last_block;

        if (output_index < config.num_outputs && config.input_idx() < config.num_input_units) {
            value = thread_reduce(output_index);
        }

        if (config.should_block_x_reduce()) {
            value = block_x_reduce(value, identity, config, ops, shared);
        }
        if (config.should_block_y_reduce()) {
            value = block_y_reduce(value, config, ops, shared);
        }

        // NB: the fold/staging paths below require the global-reduce buffers
        // (partials/counters/flags), which are only allocated when
        // config.global_reduce is set, so every branch touching them must be
        // gated on it. Without the gate, small reductions dereference null
        // staging pointers (illegal address on the first max/sum of a tiny
        // tensor).
        if (config.global_reduce) {
            const int64_t slot_base = output_index * config.ctas_per_output;
            if (block_leader && output_index < config.num_outputs) {
                partials[slot_base + static_cast<int64_t>(blockIdx.y)] = value;
                __threadfence();  // partial globally visible before the count
                // Wait until this launch's counter is initialized (CTA y==0
                // does it once, near kernel start; the unique per-launch tag
                // makes stale flag content from previous launches
                // indistinguishable-safe: it can never match). One short
                // bounded spin per CTA — no polling storm.
                volatile unsigned long long* flag = flags + blockIdx.x;
                while (*flag != tag) {}
                __threadfence();  // acquire the counter==0 establishment
                const unsigned long long prev =
                    atomicAdd(counters + blockIdx.x, 1ULL);
                is_last_block = prev == static_cast<unsigned long long>(
                                          config.ctas_per_output - 1);
            }
            __syncthreads();
            if (is_last_block && output_index < config.num_outputs) {
                __threadfence();  // acquire: peer partials are visible
                const int tid = threadIdx.x + threadIdx.y * blockDim.x;
                const int tcount = blockDim.x * blockDim.y;
                // The fold is latency-bound: one small L2 load per CTA
                // partial. Keep several independent accumulators in flight
                // per thread to hide L2 latency.
                constexpr int kFoldAcc = 8;
                AccT accs[kFoldAcc];
                #pragma unroll
                for (int k = 0; k < kFoldAcc; ++k) accs[k] = identity;
                int64_t i = tid;
                for (; i + static_cast<int64_t>(kFoldAcc - 1) * tcount <
                       config.ctas_per_output;
                     i += static_cast<int64_t>(kFoldAcc) * tcount) {
                    #pragma unroll
                    for (int k = 0; k < kFoldAcc; ++k) {
                        accs[k] = ops.combine(accs[k],
                            partials[slot_base + i +
                                     static_cast<int64_t>(k) * tcount]);
                    }
                }
                for (; i < config.ctas_per_output; i += tcount) {
                    accs[0] = ops.combine(accs[0], partials[slot_base + i]);
                }
                #pragma unroll
                for (int k = 1; k < kFoldAcc; ++k) {
                    accs[0] = ops.combine(accs[0], accs[k]);
                }
                AccT final_value = accs[0];
                final_value = block_y_reduce(final_value, config, ops, shared);
                final_value = block_x_reduce(final_value, identity, config, ops, shared);
                if (block_leader) {
                    output[config.output_offset(output_index)] = ops.project(final_value);
                }
            }
        } else if (config.should_store(output_index)) {
            output[config.output_offset(output_index)] = ops.project(value);
        }
    }
};

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize>
__global__ void reduce_kernel(ReduceOp<InputT, AccT, OutputT, Ops,
                                       ValuesPerThread, InputVecSize> op) {
    op.run();
}

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize>
inline void launch_reduce(
        TensorIterator& iter, Ops ops, AccT identity) {
    ReduceConfig config = make_reduce_config<InputT, AccT, OutputT>(iter);
    if (config.num_outputs == 0 || config.num_inputs == 0) return;
    if (!iter.can_use_32bit_indexing()) {
        TP_THROW(NotImplementedError,
                 "CUDA reduction requires 32-bit TensorIterator indexing");
    }

    const auto stream = getCurrentCUDAStream().stream();
    const dim3 block(config.block_width, config.block_height, 1);
    const dim3 grid(
        static_cast<unsigned int>((config.num_outputs + config.step_output - 1) /
                                  config.step_output),
        static_cast<unsigned int>(config.global_reduce ? config.ctas_per_output : 1),
        1);
    int shared_bytes = config.shared_memory_size(sizeof(AccT));
    if (config.global_reduce) {
        // The last-CTA fold stages one accumulator per thread in shared
        // memory (the normal path sizes this via should_block_y_reduce,
        // which the fold path cannot rely on).
        shared_bytes = std::max(shared_bytes,
            static_cast<int>(config.num_threads * sizeof(AccT)));
    }
    ReduceOp<InputT, AccT, OutputT, Ops, ValuesPerThread, InputVecSize> reduction{
        config,
        static_cast<const InputT*>(iter.data_ptr(1)),
        static_cast<OutputT*>(iter.data_ptr(0)),
        nullptr,
        nullptr,
        nullptr,
        0,
        identity,
        ops};

    DataPtr partial_buffer;
    if (config.global_reduce) {
        // Scratch layout: one completion counter + one init flag per output
        // block (u64 each), then the partials. The counter is zeroed
        // in-kernel by the (x, y==0) CTA and its readiness is published via
        // the flag holding this launch's unique tag — no cudaMemsetAsync, no
        // reliance on allocator-held state, and stale content from prior
        // launches can never match the tag.
        const size_t slots = static_cast<size_t>(config.num_outputs) *
                             config.ctas_per_output;
        const size_t head = static_cast<size_t>(grid.x) * 2 *
                            sizeof(unsigned long long);
        partial_buffer = getAllocator(DeviceType::CUDA)->allocate(
            head + slots * sizeof(AccT), iter.device());
        char* base = static_cast<char*>(partial_buffer.get());
        reduction.counters = reinterpret_cast<unsigned long long*>(base);
        reduction.flags = reinterpret_cast<unsigned long long*>(
            base + static_cast<size_t>(grid.x) * sizeof(unsigned long long));
        reduction.partials = reinterpret_cast<AccT*>(base + head);
        static std::atomic<unsigned long long> tag_counter{1};
        reduction.tag = tag_counter.fetch_add(1, std::memory_order_relaxed);
    }

    reduce_kernel<InputT, AccT, OutputT, Ops, ValuesPerThread, InputVecSize>
        <<<grid, block, shared_bytes, stream>>>(reduction);
    checkCuda(cudaGetLastError(), "CUDA reduction kernel launch");
}

// Operations -----------------------------------------------------------------

template <typename ScalarT, typename AccT, typename OutputT>
struct SumOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc + static_cast<AccT>(value);
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const { return static_cast<OutputT>(value); }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct ProdOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc * static_cast<AccT>(value);
    }
    __device__ AccT combine(AccT a, AccT b) const { return a * b; }
    __device__ OutputT project(AccT value) const { return static_cast<OutputT>(value); }
};

template <typename ScalarT, typename AccT, typename OutputT, bool MaxMode>
struct MinMaxOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return combine(acc, static_cast<AccT>(value));
    }
    __device__ AccT combine(AccT a, AccT b) const {
        if (reduce_isnan(a)) return a;
        if (reduce_isnan(b)) return b;
        if constexpr (MaxMode) return a > b ? a : b;
        else return a < b ? a : b;
    }
    __device__ OutputT project(AccT value) const { return static_cast<OutputT>(value); }
};

template <typename AccT, bool MaxMode>
struct ArgOps {
    using pair_type = ArgPair<AccT>;

    __device__ pair_type reduce(pair_type acc, AccT value, int64_t index) const {
        return better(acc, pair_type{value, index}) ? acc : pair_type{value, index};
    }
    __device__ pair_type combine(pair_type a, pair_type b) const {
        return better(a, b) ? a : b;
    }
    __device__ int64_t project(pair_type value) const { return value.index; }

    __device__ bool better(pair_type a, pair_type b) const {
        if (reduce_isnan(a.value)) {
            if (reduce_isnan(b.value)) return a.index < b.index;
            return true;
        }
        if (reduce_isnan(b.value)) return false;
        if (a.value == b.value) return a.index < b.index;
        if constexpr (MaxMode) return a.value > b.value;
        else return a.value < b.value;
    }
};

// Packed argmax (warp-shuffle form): the whole reduction state is ONE 64-bit
// word — [monotone value key (high 32) | ~index (low 32)] — selected with a
// plain integer max.  Each shuffle level moves a single u64 (native
// __shfl_down_sync overload) instead of the two shuffles plus comparator
// branches an ArgPair<float> tree performs, and no divergent NaN/tie logic
// survives in the hot loop because ordering is baked into the encoding:
//   * finite values map monotonically via the IEEE trick
//     bits ^ (sign ? 0xFFFFFFFF : 0x80000000);
//   * every NaN collapses to canonical qNaN, so any NaN outranks +inf and
//     equal NaN keys fall through to the index half (first NaN wins);
//   * -0 folds onto +0 so IEEE equality keeps the first-occurrence rule;
//   * ~index in the low half: on equal keys integer max keeps the smaller
//     index — bit-identical winners to ArgOps<float, true>.
// Row length must fit int32 (host-side guard); identities at padding lanes
// encode key 0, below every representable element key.
struct PackedArgMaxOps {
    using acc_type = unsigned long long;

    __device__ static unsigned long long pack(float value, int64_t index) {
        unsigned bits = __float_as_uint(value);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            bits = 0x7FC00000u;  // NaN family -> canonical qNaN
        } else if (bits == 0x80000000u) {
            bits = 0u;           // fold -0 onto +0
        }
        const unsigned sign = static_cast<unsigned>(static_cast<int>(bits) >> 31);
        const unsigned key = bits ^ (0x80000000u | (0x7FFFFFFFu & sign));
        return (static_cast<unsigned long long>(key) << 32) |
               static_cast<unsigned>(~static_cast<unsigned>(index));
    }

    template <typename V>
    __device__ unsigned long long reduce(
            unsigned long long acc, V value, int64_t index) const {
        const unsigned long long candidate = pack(
            static_cast<float>(value), index);
        return candidate > acc ? candidate : acc;
    }

    __device__ unsigned long long combine(unsigned long long a,
                                          unsigned long long b) const {
        return a > b ? a : b;
    }

    __device__ int64_t project(unsigned long long value) const {
        const unsigned idx =
            ~static_cast<unsigned>(value & 0xFFFFFFFFull);
        return static_cast<int64_t>(static_cast<int32_t>(idx));
    }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct AllOps {
    __device__ int reduce(int acc, ScalarT value, int64_t) const {
        return acc && static_cast<bool>(value);
    }
    __device__ int combine(int a, int b) const { return a && b; }
    __device__ OutputT project(int value) const { return static_cast<OutputT>(value != 0); }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct AnyOps {
    __device__ int reduce(int acc, ScalarT value, int64_t) const {
        return acc || static_cast<bool>(value);
    }
    __device__ int combine(int a, int b) const { return a || b; }
    __device__ OutputT project(int value) const { return static_cast<OutputT>(value != 0); }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct MeanOps {
    AccT factor;
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc + static_cast<AccT>(value);
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(value * factor);
    }
};

template <typename ScalarT, typename AccT>
__device__ __forceinline__ AccT norm_abs(ScalarT value) {
    const AccT converted = static_cast<AccT>(value);
    if constexpr (std::is_same_v<AccT, float>) {
        return ::fabsf(converted);
    } else {
        return ::fabs(converted);
    }
}

template <typename T>
__device__ __forceinline__ T norm_pow(T value, T exponent) {
    if constexpr (std::is_same_v<T, float>) {
        return ::powf(value, exponent);
    } else {
        return ::pow(value, exponent);
    }
}

template <typename ScalarT, typename AccT, typename OutputT>
struct NormZeroOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc + (value == static_cast<ScalarT>(0) ? AccT(0) : AccT(1));
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(value);
    }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct NormOneOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc + norm_abs<ScalarT, AccT>(value);
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(value);
    }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct AbsMinOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return combine(acc, norm_abs<ScalarT, AccT>(value));
    }
    __device__ AccT combine(AccT a, AccT b) const {
        if (reduce_isnan(a)) return a;
        if (reduce_isnan(b)) return b;
        return a < b ? a : b;
    }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(value);
    }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct AbsMaxOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return combine(acc, norm_abs<ScalarT, AccT>(value));
    }
    __device__ AccT combine(AccT a, AccT b) const {
        if (reduce_isnan(a)) return a;
        if (reduce_isnan(b)) return b;
        return a > b ? a : b;
    }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(value);
    }
};

template <typename ScalarT, typename AccT, typename OutputT>
struct NormOps {
    AccT norm;

    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return acc + norm_pow(norm_abs<ScalarT, AccT>(value), norm);
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const {
        return static_cast<OutputT>(norm_pow(value, AccT(1) / norm));
    }
};

template <typename AccT, typename OutputT>
struct NormTwoOps {
    __device__ AccT reduce(AccT acc, AccT value, int64_t) const {
        return acc + value * value;
    }
    template <typename ScalarT>
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        const AccT converted = static_cast<AccT>(value);
        return acc + converted * converted;
    }
    __device__ AccT combine(AccT a, AccT b) const { return a + b; }
    __device__ OutputT project(AccT value) const {
        if constexpr (std::is_same_v<AccT, float>) return static_cast<OutputT>(sqrtf(value));
        else return static_cast<OutputT>(sqrt(value));
    }
};

template <typename AccT, typename OutputT>
struct WelfordOps {
    AccT correction;
    bool take_sqrt;
    using acc_type = WelfordData<AccT>;

    __device__ acc_type reduce(acc_type acc, AccT value, int64_t) const {
        const int64_t new_n = acc.n + 1;
        const AccT new_nf = static_cast<AccT>(new_n);
        const AccT delta = value - acc.mean;
        const AccT new_mean = acc.mean + delta / new_nf;
        const AccT new_delta = value - new_mean;
        return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
    }
    template <typename ScalarT>
    __device__ acc_type reduce(acc_type acc, ScalarT value, int64_t index) const {
        return reduce(acc, static_cast<AccT>(value), index);
    }
    __device__ acc_type combine(acc_type a, acc_type b) const {
        if (a.nf == 0) return b;
        if (b.nf == 0) return a;
        const AccT delta = b.mean - a.mean;
        const AccT new_count = a.nf + b.nf;
        const AccT b_over_n = b.nf / new_count;
        return {
            a.mean + delta * b_over_n,
            a.m2 + b.m2 + delta * delta * a.nf * b_over_n,
            -1,
            new_count};
    }
    __device__ OutputT project(acc_type acc) const {
        const AccT divisor = acc.nf > correction ? acc.nf - correction : AccT(0);
        const AccT variance = acc.m2 / divisor;
        if (take_sqrt) {
            if constexpr (std::is_same_v<AccT, float>) {
                return static_cast<OutputT>(sqrtf(variance));
            } else {
                return static_cast<OutputT>(sqrt(variance));
            }
        }
        return static_cast<OutputT>(variance);
    }
};

} // namespace reduction
} // namespace cuda
} // namespace tensorplay
