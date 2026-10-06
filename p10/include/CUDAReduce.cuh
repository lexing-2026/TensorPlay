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
constexpr int kReductionEngineRevision = 13;

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
struct default_accumulation_type<tensorplay::complex<Half>> {
    using type = tensorplay::complex<float>;
};

template <>
struct default_accumulation_type<tensorplay::complex<BFloat16>> {
    using type = tensorplay::complex<float>;
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
    } else if constexpr (std::is_same_v<T, Half> ||
                         std::is_same_v<T, BFloat16>) {
        return ::isnan(static_cast<float>(value));
    } else if constexpr (
            std::is_same_v<T, tensorplay::complex<Half>> ||
            std::is_same_v<T, tensorplay::complex<float>> ||
            std::is_same_v<T, tensorplay::complex<double>> ||
            std::is_same_v<T, tensorplay::complex<BFloat16>>) {
        return ::isnan(static_cast<double>(value.real())) ||
               ::isnan(static_cast<double>(value.imag()));
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

// Pair accumulator for a fused aminmax: both extrema ride through every
// shuffle level as one struct, so a single pass serves both outputs.
template <typename AccT>
struct MinMaxPair {
    AccT min_val;
    AccT max_val;
};

template <typename AccT>
__device__ __forceinline__ MinMaxPair<AccT> reduce_warp_shuffle_down(
        MinMaxPair<AccT> value, unsigned long long mask, int offset) {
    return {
        reduce_warp_shuffle_down(value.min_val, mask, offset),
        reduce_warp_shuffle_down(value.max_val, mask, offset)};
}

// The count rides along every shuffle and every shared-memory staging step, so
// its width matters: a 32-bit count keeps the float accumulator at four words
// (mean, m2, n, nf) instead of five, which is what the shuffle tree and the
// block staging move.  Callers pick the width from the reduction extent.
template <typename T, typename IndexT = int64_t>
struct WelfordData {
    T mean;
    T m2;
    IndexT n;
    T nf;
};

template <typename T, typename IndexT>
__device__ __forceinline__ WelfordData<T, IndexT> reduce_warp_shuffle_down(
        WelfordData<T, IndexT> value, unsigned long long mask, int offset) {
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
    int input_head = 0;
    int output_vec_size = 1;
    bool vectorize_input = false;
    // Vec loads stay inside the fastest reduced chunk (no crossing of the
    // outer reduced dims' holes), so units need a coordinate decomposition.
    bool vectorize_chunked = false;
    bool index32 = false;
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
        return (static_cast<int64_t>(blockIdx.x) * step_output +
            static_cast<int64_t>(threadIdx.x) * output_mult[0] +
            static_cast<int64_t>(threadIdx.y) * output_mult[1]) * output_vec_size;
    }

    __device__ __forceinline__ int64_t input_base_offset(int64_t output) const {
        // Fastest-moving dimension last after reorder; decode in that order.
        // The dim count is fixed per launch, so dispatch on it and keep every
        // case at a fixed trip count: the compiler can then hold the extent /
        // stride loads in registers and drop the per-element div/mod chains
        // that a dynamic loop forces through local memory.
        const int out_dims = ndim - num_reduce_dims;
        if (out_dims <= 0) return 0;
        if (index32) {
            uint32_t off = 0, remainder = static_cast<uint32_t>(output);
            switch (out_dims) {
                case 1:
                    off = remainder * static_cast<uint32_t>(input_strides[num_reduce_dims]);
                    return off;
                case 2: {
                    const uint32_t e0 = static_cast<uint32_t>(shape[num_reduce_dims]);
                    off = (remainder % e0) * static_cast<uint32_t>(input_strides[num_reduce_dims]);
                    remainder /= e0;
                    off += remainder * static_cast<uint32_t>(input_strides[num_reduce_dims + 1]);
                    return off;
                }
                default: break;
            }
        }
        int64_t offset = 0;
        int64_t rest = output;
        for (int dim = num_reduce_dims; dim < ndim; ++dim) {
            const int64_t extent = shape[dim];
            const int64_t coordinate = extent > 0 ? rest % extent : 0;
            rest = extent > 0 ? rest / extent : 0;
            offset += coordinate * input_strides[dim];
        }
        return offset;
    }

    __device__ __forceinline__ int64_t output_offset(int64_t output) const {
        const int out_dims = ndim - num_reduce_dims;
        if (out_dims <= 0) return 0;
        if (index32) {
            uint32_t off = 0, remainder = static_cast<uint32_t>(output);
            switch (out_dims) {
                case 1:
                    off = remainder * static_cast<uint32_t>(output_strides[num_reduce_dims]);
                    return off;
                case 2: {
                    const uint32_t e0 = static_cast<uint32_t>(shape[num_reduce_dims]);
                    off = (remainder % e0) * static_cast<uint32_t>(output_strides[num_reduce_dims]);
                    remainder /= e0;
                    off += remainder * static_cast<uint32_t>(output_strides[num_reduce_dims + 1]);
                    return off;
                }
                default: break;
            }
        }
        int64_t offset = 0;
        int64_t rest = output;
        for (int dim = num_reduce_dims; dim < ndim; ++dim) {
            const int64_t extent = shape[dim];
            const int64_t coordinate = extent > 0 ? rest % extent : 0;
            rest = extent > 0 ? rest / extent : 0;
            offset += coordinate * output_strides[dim];
        }
        return offset;
    }

    __device__ __forceinline__ int64_t input_offset(int64_t index) const {
        if (num_reduce_dims == 0) return 0;
        if (num_reduce_dims == 1) return index * input_strides[0];
        if (index32) {
            uint32_t off, r = static_cast<uint32_t>(index);
            switch (num_reduce_dims) {
                case 2:
                    off  = (r % static_cast<uint32_t>(shape[0])) *
                           static_cast<uint32_t>(input_strides[0]);
                    r /= static_cast<uint32_t>(shape[0]);
                    off += r * static_cast<uint32_t>(input_strides[1]);
                    return off;
                case 3:
                    off  = (r % static_cast<uint32_t>(shape[0])) *
                           static_cast<uint32_t>(input_strides[0]);
                    r /= static_cast<uint32_t>(shape[0]);
                    off += (r % static_cast<uint32_t>(shape[1])) *
                           static_cast<uint32_t>(input_strides[1]);
                    r /= static_cast<uint32_t>(shape[1]);
                    off += r * static_cast<uint32_t>(input_strides[2]);
                    return off;
                case 4:
                    off  = (r % static_cast<uint32_t>(shape[0])) *
                           static_cast<uint32_t>(input_strides[0]);
                    r /= static_cast<uint32_t>(shape[0]);
                    off += (r % static_cast<uint32_t>(shape[1])) *
                           static_cast<uint32_t>(input_strides[1]);
                    r /= static_cast<uint32_t>(shape[1]);
                    off += (r % static_cast<uint32_t>(shape[2])) *
                           static_cast<uint32_t>(input_strides[2]);
                    r /= static_cast<uint32_t>(shape[2]);
                    off += r * static_cast<uint32_t>(input_strides[3]);
                    return off;
                default: break;
            }
        }
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

    // Compile-time specialized variant for the per-element hot loop: the
    // reduced-dim count is folded at the per-thread dispatch, so the offset
    // math is straight-line and the group's loads can be hoisted together.
    // A runtime branch wrapping each load (as in input_offset) blocks that
    // hoisting and roughly triples the small-chunk kernel time.
    // NRED >= 2 assumes 32-bit indices: launch_reduce rejects iterators that
    // cannot use 32-bit indexing, so the u32 decomposition below is exact.
    template <int NRED>
    __device__ __forceinline__ int64_t input_offset_nt(int64_t index) const {
        if constexpr (NRED == 0) {
            return input_offset(index);
        } else if constexpr (NRED == 1) {
            return index * input_strides[0];
        } else {
            uint32_t off = 0, r = static_cast<uint32_t>(index);
            #pragma unroll
            for (int dim = 0; dim < NRED; ++dim) {
                off += (r % static_cast<uint32_t>(shape[dim])) *
                       static_cast<uint32_t>(input_strides[dim]);
                r /= static_cast<uint32_t>(shape[dim]);
            }
            return off;
        }
    }

    // Slot of one CTA partial.  When block.x walks the outputs instead of the
    // reduced axis, every lane owns an output and therefore needs its own run
    // of ctas_per_output slots.
    __host__ __device__ int64_t staging_offset(int64_t cta) const {
        const int64_t base = cta + static_cast<int64_t>(blockIdx.x) * gridDim.y;
        if (!should_block_x_reduce()) {
            return threadIdx.x + base * blockDim.x;
        }
        return base;
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

    config.index32 = iter.can_use_32bit_indexing();

    int64_t dim0 = reduction_on_fastest_dimension
        ? config.num_inputs : config.num_outputs;
    int64_t dim1 = reduction_on_fastest_dimension
        ? config.num_outputs : config.num_inputs;

    if (reduction_on_fastest_dimension &&
        config.input_strides[0] == 1 && config.num_inputs >= 128) {
        // 16-bit types load eight elements per instruction (one 16B vector,
        // matching the 32-bit types' vec4); wider types keep vec4.  vec8 is
        // only instantiated for Half/BFloat16 so the PTX growth stays out of
        // the 32/64-bit instantiations.
        config.input_vec_size = (sizeof(InputT) == 2) ? 8 : 4;
        const size_t vector_bytes = sizeof(InputT) * static_cast<size_t>(config.input_vec_size);
        const bool aligned = reduction_pointer_aligned(iter, vector_bytes);
        const size_t address = reinterpret_cast<uintptr_t>(iter.data_ptr(1));
        const int head = aligned
            ? 0
            : static_cast<int>((vector_bytes - (address % vector_bytes)) /
                               sizeof(InputT));
        // The per-output row base is the sum over non-reduced dims of
        // coordinate * stride, so those strides must keep every row start on
        // a vector boundary, not just the storage pointer.
        bool rows_aligned = true;
        for (int dim = config.num_reduce_dims; dim < config.ndim; ++dim) {
            rows_aligned = rows_aligned &&
                (config.input_strides[dim] % config.input_vec_size == 0);
        }
        // A unit of InputVecSize logical elements must map to InputVecSize
        // consecutive physical elements. With one reduced dim the whole row
        // is one physical run, so a ragged extent is fine (bounds-checked
        // units plus a scalar tail). With several reduced dims the outer
        // chunks leave holes in the row, so every unit must stay inside the
        // fastest chunk: chunk-multiple extent and vector-multiple strides
        // everywhere else make the per-unit decomposition exact and keep the
        // vec address aligned.
        if (config.num_reduce_dims == 1) {
            if (rows_aligned) {
                config.vectorize_input = true;
                config.input_head = head;
                config.num_input_units =
                    (config.num_inputs - config.input_head + config.input_vec_size - 1) /
                    config.input_vec_size;
                dim0 = config.num_input_units;
            } else {
                config.input_vec_size = 1;
            }
        } else {
            bool chunkable = config.index32 && aligned &&
                (config.shape[0] % config.input_vec_size) == 0;
            for (int dim = 1; dim < config.num_reduce_dims && chunkable; ++dim) {
                chunkable = chunkable &&
                    (config.input_strides[dim] % config.input_vec_size == 0);
            }
            for (int dim = config.num_reduce_dims; dim < config.ndim; ++dim) {
                chunkable = chunkable &&
                    (config.input_strides[dim] % config.input_vec_size == 0);
            }
            if (chunkable) {
                config.vectorize_input = true;
                config.vectorize_chunked = true;
                config.num_input_units = config.num_inputs / config.input_vec_size;
                dim0 = config.num_input_units;
            } else {
                config.input_vec_size = 1;
            }
        }
    }

    if (!reduction_on_fastest_dimension &&
        config.input_strides[config.num_reduce_dims] == 1) {
        int width = 4;
        auto restrict_width = [&](uint64_t value) {
            while (value % width != 0) width /= 2;
        };
        restrict_width(reinterpret_cast<uintptr_t>(iter.data_ptr(1)) / sizeof(InputT));
        restrict_width(config.shape[config.num_reduce_dims]);
        for (int dim = 0; dim < config.ndim; ++dim) {
            if (dim != config.num_reduce_dims) {
                restrict_width(config.input_strides[dim]);
            }
        }
        config.output_vec_size = width;
        dim0 /= width;
    }

    // Only the 16-byte element type trades thread budget for occupancy;
    // every other accumulator keeps the full ceiling.
    const int max_threads =
        (std::is_same<InputT, tensorplay::complex<double>>::value
            ? 256
            : kMaxReduceThreads) / config.output_vec_size;
    // Block shape in both mappings: block.x is sized from the per-output
    // extent and block.y from the output count, so a block always covers whole
    // rows.  block.x targets kElemsPerLane elements per lane - one or two per
    // lane spends the block on scheduling and shuffle traffic rather than on
    // loads, while devoting a full warp to a short row wastes most of its
    // lanes.  Clamped to [2, one warp]: a row spans at least two lanes and
    // never more than a warp.  The thresholds come from a measured sweep over
    // row lengths 4..8192 (see the commit message for the numbers).
    //
    // The width lands on a power of two: both cross-lane folds walk their
    // offset from half the width down to one, and that walk only reaches every
    // lane when the width divides the warp evenly, so a width like 7 or 12
    // would drop contributions (a 56-element row summed to 8.5x its value).
    constexpr int kElemsPerLane = 8;
    const int want_width = reduction_last_pow2(std::max<int64_t>(
        2, std::min<int64_t>(kWarpSize, config.num_inputs / kElemsPerLane)));
    const int dim0_pow2 = reduction_on_fastest_dimension
        ? std::min(want_width, reduction_last_pow2(config.num_inputs))
        : std::min(kWarpSize, reduction_last_pow2(dim0));
    const int dim1_pow2 = reduction_last_pow2(dim1);
    config.block_width = std::min(dim0_pow2, kWarpSize);
    const int max_height =
        std::max(1, max_threads / std::max(1, config.block_width));
    int desired_height = std::min(dim1_pow2, max_height);
    // Global-reduce prediction: a single-output (dim1 == 1) reduction large
    // enough to trigger the multi-CTA branch below runs with a taller block —
    // 8 warps share one CTA's completion-counter slot and staging partial,
    // cutting same-address atomic traffic and the last-CTA fold length 8x
    // The 16384-element floor guarantees the warp-split below actually
    // engages (input_mult[1] != 0), keeping output_mult clean for the gate.
    if (reduction_on_fastest_dimension && dim1 == 1 && config.num_inputs >= 16384) {
        desired_height = std::min(8, max_height);
    }
    config.block_height = std::max(1, std::min(desired_height, max_height));
    // A tall block can squeeze block.x below the extent-derived width; keep
    // the row's lanes a power of two no wider than that width.
    config.block_width = std::max(
        1, std::min(dim0_pow2, max_threads / config.block_height));
    config.num_threads = config.block_width * config.block_height;

    if (reduction_on_fastest_dimension || config.ndim == 0) {
        config.input_mult[0] = config.split_input(config.block_width);
    } else {
        config.output_mult[0] = config.split_output(config.block_width);
    }

    // Parallelism thresholds are element-based: vectorized units each cover
    // InputVecSize elements, so a unit count would under-report the work per
    // thread by that factor and wrongly suppress the splits below.
    const int64_t values_per_thread =
        ((config.num_input_units + config.step_input - 1) / config.step_input) *
        config.input_vec_size;
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
    // A block owns one contiguous output range when only one of its two
    // extents carries the output split; when both do, the owned outputs are a
    // two-dimensional set and the single-range fold below cannot address them.
    const bool outputs_form_one_range =
        config.output_mult[0] == 0 || config.output_mult[1] == 0;
    if (outputs_form_one_range && config.num_outputs > 0) {
        // Elements still to be consumed per thread after the lane and warp
        // splits: num_inputs spread over step_input units of InputVecSize
        // elements each.
        const int64_t values_per_thread_elems =
            (config.num_inputs + config.step_input * config.input_vec_size - 1) /
            (config.step_input * config.input_vec_size);
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
            const int64_t output_step = config.step_output * config.output_vec_size;
            const int64_t grid_x = (config.num_outputs + output_step - 1) / output_step;
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

    // Shuffles must stay inside one reduction row: with block.x narrower than
    // a warp a row owns only the leading lanes of each group, so the walk
    // starts at the row width.  Wider blocks were folded down to a full warp
    // by the shared-memory phase above.
    const int row_width =
        config.block_width > kWarpSize ? kWarpSize : config.block_width;
    for (int offset = row_width / 2; offset > 0; offset >>= 1) {
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

// Ops that can hand back a companion value (the Welford mean rides along with
// the variance) define project_second; every other reduction leaves it out and
// the companion write is compiled away.
template <typename Ops, typename = void>
struct has_second_project : std::false_type {};
template <typename Ops>
struct has_second_project<Ops, std::void_t<decltype(
    std::declval<const Ops&>().project_second(
        std::declval<typename Ops::acc_type>()))>> : std::true_type {};

// Index-carrying ops (argmax/argmin, max/min with indices) define
// translate_idx so a 32-bit split can rebase sub-iteration indices onto the
// original reduction space before combining partials.
template <typename Ops, typename = void>
struct has_translate_idx : std::false_type {};
template <typename Ops>
struct has_translate_idx<Ops, std::void_t<decltype(
    std::declval<const Ops&>().translate_idx(
        std::declval<typename Ops::acc_type>(),
        std::declval<int64_t>()))>> : std::true_type {};

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize, typename SecondOutputT = OutputT>
struct ReduceOp {
    ReduceConfig config;
    const InputT* input;
    OutputT* output;
    SecondOutputT* output2;
    AccT* partials;
    unsigned long long* counters;
    unsigned long long* flags;
    unsigned long long tag;
    AccT identity;
    Ops ops;
    // 32-bit split support: non-32-bit iterators recurse over sub-iterators
    // that share this accumulator buffer; each sub-launch rebases its index
    // space by base_idx and only the final sub-iteration projects to output.
    AccT* acc_buf;
    int64_t base_idx;
    bool accumulate;
    bool final_output;

    template <int NRED>
    __device__ __forceinline__ AccT reduce_unit(
            AccT value, const InputT* row, int64_t unit, int64_t logical_base) const {
        if constexpr (InputVecSize == 1) {
            if (logical_base < config.num_inputs) {
                value = ops.reduce(value,
                    row[config.template input_offset_nt<NRED>(logical_base)], logical_base);
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
                    value = ops.reduce(value,
                        row[config.template input_offset_nt<NRED>(logical)], logical);
                }
            }
        }
        (void)unit;
        return value;
    }

    // Fixed-shape offset decomposition: the per-launch dim count lets every
    // unit resolve its address through a compile-time-unrolled chain, so the
    // div/mod work rides in registers alongside the loads instead of forcing
    // a dynamic loop through local memory.
    template <int NRED>
    __device__ __forceinline__ AccT thread_reduce_nt(int64_t output_index) const {
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
        const bool can_vec_full = can_vec && config.input_head == 0 &&
            config.num_inputs % InputVecSize == 0;
        // Multi-dim reduction rows contain holes between the outer chunks;
        // units are still chunk-aligned (host-side gate), so each unit is one
        // vec load whose row-relative address comes from the per-unit
        // offset decomposition.
        const bool can_vec_chunked = can_vec && config.vectorize_chunked;

        if (can_vec_chunked) {
            int64_t unit = start;
            while (unit + static_cast<int64_t>(ValuesPerThread - 1) * step < end) {
                #pragma unroll
                for (int i = 0; i < ValuesPerThread; ++i) {
                    const int64_t logical_base =
                        (unit + static_cast<int64_t>(i) * step) * InputVecSize;
                    const Vec loaded = *reinterpret_cast<const Vec*>(
                        row + config.template input_offset_nt<NRED>(logical_base));
                    #pragma unroll
                    for (int j = 0; j < InputVecSize; ++j) {
                        values[i] = ops.reduce(values[i], loaded.val[j], logical_base + j);
                    }
                }
                unit += step * ValuesPerThread;
            }
            while (unit < end) {
                const int64_t logical_base = unit * InputVecSize;
                const Vec loaded = *reinterpret_cast<const Vec*>(
                    row + config.template input_offset_nt<NRED>(logical_base));
                #pragma unroll
                for (int j = 0; j < InputVecSize; ++j) {
                    values[0] = ops.reduce(values[0], loaded.val[j], logical_base + j);
                }
                unit += step;
            }
        } else if (can_vec_full) {
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
            if constexpr (InputVecSize == 1) {
                // Scalar main loop: the while condition bounds every unit of
                // the group (unit + (ValuesPerThread-1)*step < end), so
                // per-element guards would be redundant — and measurably
                // double the runtime on small chunks by blocking load
                // hoisting.
                while (unit + static_cast<int64_t>(ValuesPerThread - 1) * step < end) {
                    #pragma unroll
                    for (int i = 0; i < ValuesPerThread; ++i) {
                        const int64_t logical = unit + static_cast<int64_t>(i) * step;
                        values[i] = ops.reduce(values[i], row[config.template input_offset_nt<NRED>(logical)], logical);
                    }
                    unit += step * ValuesPerThread;
                }
            } else {
                // Vector units with a ragged final chunk (num_inputs not
                // divisible by InputVecSize): bound the main loop by the
                // full-unit count so no vec load can overrun the row; the
                // checked tail below consumes the remainder. For
                // InputVecSize > 1 the host only enables vectorization when
                // the fastest row is contiguous, so unit*InputVecSize is the
                // exact element base of each unit.
                const int64_t full_units =
                    (config.num_inputs - config.input_head) / InputVecSize;
                while (unit + static_cast<int64_t>(ValuesPerThread - 1) * step < full_units) {
                    #pragma unroll
                    for (int i = 0; i < ValuesPerThread; ++i) {
                        const int64_t logical_base =
                            config.input_head +
                            (unit + static_cast<int64_t>(i) * step) * InputVecSize;
                        const Vec loaded = *reinterpret_cast<const Vec*>(
                            row + config.template input_offset_nt<NRED>(logical_base));
                        #pragma unroll
                        for (int j = 0; j < InputVecSize; ++j) {
                            values[i] = ops.reduce(values[i], loaded.val[j], logical_base + j);
                        }
                    }
                    unit += step * ValuesPerThread;
                }
            }
            while (unit < end) {
                values[0] = reduce_unit<NRED>(
                    values[0], row, unit,
                    config.input_head + unit * InputVecSize);
                // Threads stride by step_input; a plain ++unit makes every lane
                // walk into its neighbours' units (each element counted
                // (num_inputs - lane) times -> triangular sums).
                unit += step;
            }
        }

        if (InputVecSize > 1 && config.input_head > 0 &&
            (config.input_mult[1] == 0 || threadIdx.y == 0) &&
            blockIdx.y == 0) {
            #pragma unroll
            for (int i = 0; i < InputVecSize; ++i) {
                if (i < config.input_head && i < config.num_inputs &&
                    threadIdx.x == i) {
                    values[0] = ops.reduce(values[0], row[i], i);
                }
            }
        }

        #pragma unroll
        for (int i = 1; i < ValuesPerThread; ++i) {
            values[0] = ops.combine(values[0], values[i]);
        }
        return values[0];
    }

    __device__ __forceinline__ AccT thread_reduce(int64_t output_index) const {
        // One uniform branch per thread (not per element): the per-element
        // offset math becomes compile-time straight-line for the common dim
        // counts, which lets the unrolled loads hoist together.
        switch (config.num_reduce_dims) {
            case 1: return thread_reduce_nt<1>(output_index);
            case 2: return thread_reduce_nt<2>(output_index);
            case 3: return thread_reduce_nt<3>(output_index);
            case 4: return thread_reduce_nt<4>(output_index);
            default: return thread_reduce_nt<0>(output_index);
        }
    }

    template <int NRED, int OutputVecSize>
    __device__ __forceinline__ std::array<AccT, OutputVecSize>
    thread_reduce_outputs_nt(int64_t output_index) const {
        AccT values[ValuesPerThread][OutputVecSize];
        #pragma unroll
        for (int i = 0; i < ValuesPerThread; ++i) {
            #pragma unroll
            for (int j = 0; j < OutputVecSize; ++j) values[i][j] = identity;
        }
        using Vec = aligned_vector<InputT, OutputVecSize>;
        const InputT* row = input + config.input_base_offset(output_index);
        int64_t index = config.input_idx();
        const int64_t step = config.step_input;
        while (index + (ValuesPerThread - 1) * step < config.num_inputs) {
            Vec loaded[ValuesPerThread];
            #pragma unroll
            for (int i = 0; i < ValuesPerThread; ++i) {
                loaded[i] = *reinterpret_cast<const Vec*>(row +
                    config.template input_offset_nt<NRED>(index + i * step));
            }
            #pragma unroll
            for (int i = 0; i < ValuesPerThread; ++i) {
                #pragma unroll
                for (int j = 0; j < OutputVecSize; ++j) {
                    values[i][j] = ops.reduce(values[i][j], loaded[i].val[j], index + i * step);
                }
            }
            index += ValuesPerThread * step;
        }
        #pragma unroll
        for (int i = 0; i < ValuesPerThread; ++i) {
            if (index < config.num_inputs) {
                const Vec loaded = *reinterpret_cast<const Vec*>(row +
                    config.template input_offset_nt<NRED>(index));
                #pragma unroll
                for (int j = 0; j < OutputVecSize; ++j) {
                    values[i][j] = ops.reduce(values[i][j], loaded.val[j], index);
                }
                index += step;
            }
        }
        std::array<AccT, OutputVecSize> result;
        #pragma unroll
        for (int j = 0; j < OutputVecSize; ++j) {
            result[j] = values[0][j];
            #pragma unroll
            for (int i = 1; i < ValuesPerThread; ++i) {
                result[j] = ops.combine(result[j], values[i][j]);
            }
        }
        return result;
    }

    template <int OutputVecSize>
    __device__ __forceinline__ std::array<AccT, OutputVecSize>
    thread_reduce_outputs(int64_t output_index) const {
        if constexpr (OutputVecSize == 1) {
            return {thread_reduce(output_index)};
        } else {
            switch (config.num_reduce_dims) {
                case 1: return thread_reduce_outputs_nt<1, OutputVecSize>(output_index);
                case 2: return thread_reduce_outputs_nt<2, OutputVecSize>(output_index);
                case 3: return thread_reduce_outputs_nt<3, OutputVecSize>(output_index);
                case 4: return thread_reduce_outputs_nt<4, OutputVecSize>(output_index);
                default: return thread_reduce_outputs_nt<0, OutputVecSize>(output_index);
            }
        }
    }

    template <int OutputVecSize>
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
        std::array<AccT, OutputVecSize> values;
        #pragma unroll
        for (int j = 0; j < OutputVecSize; ++j) values[j] = identity;
        __shared__ bool is_last_block;

        if (output_index < config.num_outputs && config.input_idx() < config.num_input_units) {
            values = thread_reduce_outputs<OutputVecSize>(output_index);
        }

        #pragma unroll
        for (int j = 0; j < OutputVecSize; ++j) {
            if (config.should_block_x_reduce()) {
                values[j] = block_x_reduce(values[j], identity, config, ops, shared);
            }
            if (config.should_block_y_reduce()) {
                values[j] = block_y_reduce(values[j], config, ops, shared);
            }
        }

        // NB: the fold/staging paths below require the global-reduce buffers
        // (partials/counters/flags), which are only allocated when
        // config.global_reduce is set, so every branch touching them must be
        // gated on it. Without the gate, small reductions dereference null
        // staging pointers (illegal address on the first max/sum of a tiny
        // tensor).
        auto store_out = [&](int64_t out, const AccT& raw_value) {
            const int64_t off = config.output_offset(out);
            AccT value = raw_value;
            if (acc_buf != nullptr) {
                // The output dtype cannot hold the accumulator (Welford
                // state, arg pairs, packed words, narrow outputs): stage the
                // raw accumulator in a dedicated buffer and only project on
                // the final sub-iteration.
                if (accumulate) {
                    if constexpr (has_translate_idx<Ops>::value) {
                        value = ops.translate_idx(value, base_idx);
                    }
                    value = ops.combine(acc_buf[off], value);
                }
                if (final_output) {
                    output[off] = ops.project(value);
                    if (output2 != nullptr) {
                        if constexpr (has_second_project<Ops>::value) {
                            output2[off] = ops.project_second(value);
                        }
                    }
                } else {
                    acc_buf[off] = value;
                }
            } else {
                // The output buffer itself holds the running accumulator.
                if (accumulate) {
                    if constexpr (has_translate_idx<Ops>::value) {
                        value = ops.translate_idx(value, base_idx);
                    }
                    if constexpr (std::is_convertible_v<OutputT, AccT>) {
                        value = ops.combine(static_cast<AccT>(output[off]), value);
                    }
                }
                if (final_output) {
                    output[off] = ops.project(value);
                    if (output2 != nullptr) {
                        if constexpr (has_second_project<Ops>::value) {
                            output2[off] = ops.project_second(value);
                        }
                    }
                } else if constexpr (std::is_convertible_v<AccT, OutputT>) {
                    // Keep the raw accumulator (not the projected result) so
                    // the next sub-iteration can combine with it.
                    output[off] = static_cast<OutputT>(value);
                }
            }
        };
        if (config.global_reduce) {
            // Publish this CTA's piece, then find out whether every piece of
            // this block's outputs has landed.  Only the lane that owns an
            // output publishes (should_store), so the staging run of an output
            // receives exactly one value per CTA.
            const bool writes = config.should_store(output_index);
            if (writes) {
                #pragma unroll
                for (int j = 0; j < OutputVecSize; ++j) {
                    partials[config.staging_offset(blockIdx.y) * OutputVecSize + j] = values[j];
                }
            }
            __threadfence();  // make the writes globally visible
            __syncthreads();  // ... and complete before the arrival count moves
            if (block_leader) {
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
            if (is_last_block) {
                __threadfence();  // complete the acquire after the atomic
                // The fold may only walk the block extent that does not carry
                // the output split: with the outputs on block.x, folding across
                // x would merge different outputs into one.  So the lanes take
                // turns over the CTA partials of their own output, and the
                // cross-lane folds follow the same split.
                #pragma unroll
                for (int j = 0; j < OutputVecSize; ++j) {
                    AccT value = identity;
                    if (config.should_block_x_reduce()) {
                        const int64_t step = static_cast<int64_t>(blockDim.x) * blockDim.y;
                        for (int64_t cta = threadIdx.x + threadIdx.y * blockDim.x;
                             cta < config.ctas_per_output; cta += step) {
                            value = ops.combine(value,
                                partials[config.staging_offset(cta) * OutputVecSize + j]);
                        }
                    } else {
                        for (int64_t cta = threadIdx.y; cta < config.ctas_per_output;
                             cta += blockDim.y) {
                            value = ops.combine(value,
                                partials[config.staging_offset(cta) * OutputVecSize + j]);
                        }
                    }
                    value = block_y_reduce(value, config, ops, shared);
                    if (config.should_block_x_reduce()) {
                        value = block_x_reduce(value, identity, config, ops, shared);
                    }
                    if (writes) store_out(output_index + j, value);
                }
            }
        } else if (config.should_store(output_index)) {
            #pragma unroll
            for (int j = 0; j < OutputVecSize; ++j) store_out(output_index + j, values[j]);
        }
    }
};

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize, typename SecondOutputT = OutputT>
__global__ void __launch_bounds__(kMaxReduceThreads, 4)
reduce_kernel(ReduceOp<InputT, AccT, OutputT, Ops, ValuesPerThread,
                    InputVecSize, SecondOutputT> op) {
    if constexpr (InputVecSize == 1) {
        if (op.config.output_vec_size == 4) op.template run<4>();
        else if (op.config.output_vec_size == 2) op.template run<2>();
        else op.template run<1>();
    } else {
        op.template run<1>();
    }
}

// Flattened offset of a 32-bit sub-iterator's start inside the original
// reduction index space.  Reduced dims are reordered to the front, so the
// base is the sum over reduced dims of view_offset * product of faster
// reduced extents.
inline int64_t reduction_base_offset(
        const TensorIterator& iter, const TensorIterator& sub_iter) {
    int64_t base = 0;
    int64_t multiplier = 1;
    const int nrd = iter.num_reduce_dims();
    for (int d = 0; d < nrd; ++d) {
        base += sub_iter.view_offsets()[d] * multiplier;
        multiplier *= iter.shape()[d];
    }
    return base;
}

template <typename InputT, typename AccT, typename OutputT, typename Ops,
          int ValuesPerThread, int InputVecSize, typename SecondOutputT = OutputT>
inline void launch_reduce(
        TensorIterator& iter, Ops ops, AccT identity, SecondOutputT* output2 = nullptr,
        AccT* acc_buf = nullptr, int64_t base_idx = 0) {
    ReduceConfig config = make_reduce_config<InputT, AccT, OutputT>(iter);
    if (config.num_outputs == 0 || config.num_inputs == 0) return;

    if (!iter.can_use_32bit_indexing()) {
        // Split the iteration space into 32-bit sub-iterators and accumulate
        // the partials.  When the output dtype can hold the accumulator (same
        // width, convertible), the output buffer doubles as the accumulation
        // buffer; otherwise a dedicated AccT buffer stages the partials and
        // the final sub-iteration projects them.
        constexpr bool can_accumulate_in_output =
            (sizeof(AccT) <= sizeof(OutputT)) &&
            std::is_convertible_v<AccT, OutputT> &&
            !(std::is_same_v<InputT, Half> && std::is_same_v<OutputT, Half>) &&
            !(std::is_same_v<InputT, BFloat16> && std::is_same_v<OutputT, BFloat16>) &&
            !(std::is_same_v<InputT, tensorplay::complex<Half>> &&
              std::is_same_v<OutputT, tensorplay::complex<Half>>);
        DataPtr owned_acc;
        if (!can_accumulate_in_output && acc_buf == nullptr) {
            owned_acc = getAllocator(DeviceType::CUDA)->allocate(
                sizeof(AccT) * static_cast<size_t>(iter.num_output_elements()),
                iter.device());
            acc_buf = static_cast<AccT*>(owned_acc.get());
        }
        for (auto& sub_iter : iter.with_32bit_indexing()) {
            const int64_t sub_base = reduction_base_offset(iter, sub_iter);
            const int64_t output_offset = static_cast<OutputT*>(sub_iter.data_ptr(0)) -
                static_cast<OutputT*>(iter.data_ptr(0));
            AccT* sub_acc = acc_buf == nullptr ? nullptr : acc_buf + output_offset;
            SecondOutputT* sub_output2 = output2 == nullptr ? nullptr : output2 + output_offset;
            // A split can shrink the reduced extent below the vectorization
            // floor, so re-derive the vector width for this sub-iterator
            // instead of assuming the parent's.
            const auto sub_config = make_reduce_config<InputT, AccT, OutputT>(sub_iter);
            if (sub_config.input_vec_size == 8) {
                launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 8, SecondOutputT>(
                    sub_iter, ops, identity, sub_output2, sub_acc, sub_base);
            } else if (sub_config.input_vec_size == 4) {
                launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 4, SecondOutputT>(
                    sub_iter, ops, identity, sub_output2, sub_acc, sub_base);
            } else {
                launch_reduce<InputT, AccT, OutputT, Ops, ValuesPerThread, 1, SecondOutputT>(
                    sub_iter, ops, identity, sub_output2, sub_acc, sub_base);
            }
        }
        return;
    }

    const auto stream = getCurrentCUDAStream().stream();
    const dim3 block(config.block_width, config.block_height, 1);
    const dim3 grid(
        static_cast<unsigned int>((config.num_outputs +
                                  config.step_output * config.output_vec_size - 1) /
                                  (config.step_output * config.output_vec_size)),
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
    ReduceOp<InputT, AccT, OutputT, Ops, ValuesPerThread, InputVecSize, SecondOutputT> reduction{
        config,
        static_cast<const InputT*>(iter.data_ptr(1)),
        static_cast<OutputT*>(iter.data_ptr(0)),
        output2,
        nullptr,
        nullptr,
        nullptr,
        0,
        identity,
        ops,
        acc_buf,
        base_idx,
        iter.should_accumulate(),
        iter.is_final_output()};

    DataPtr partial_buffer;
    if (config.global_reduce) {
        // Scratch layout: one completion counter + one init flag per output
        // block (u64 each), then the partials. The counter is zeroed
        // in-kernel by the (x, y==0) CTA and its readiness is published via
        // the flag holding this launch's unique tag — no cudaMemsetAsync, no
        // reliance on allocator-held state, and stale content from prior
        // launches can never match the tag.
        size_t slots = static_cast<size_t>(grid.x) * grid.y * config.output_vec_size;
        if (!config.should_block_x_reduce()) {
            slots *= static_cast<size_t>(config.block_width);
        }
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

    reduce_kernel<InputT, AccT, OutputT, Ops, ValuesPerThread, InputVecSize, SecondOutputT>
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
struct NanSumOps {
    __device__ AccT reduce(AccT acc, ScalarT value, int64_t) const {
        return reduce_isnan(value) ? acc : acc + static_cast<AccT>(value);
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
    using acc_type = ArgPair<AccT>;

    __device__ pair_type reduce(pair_type acc, AccT value, int64_t index) const {
        return better(acc, pair_type{value, index}) ? acc : pair_type{value, index};
    }
    __device__ pair_type combine(pair_type a, pair_type b) const {
        return better(a, b) ? a : b;
    }
    __device__ int64_t project(pair_type value) const { return value.index; }
    __device__ pair_type translate_idx(pair_type value, int64_t base_idx) const {
        return {value.value, value.index + base_idx};
    }

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

// Generic single-pass extremum with indices for non-float accumulators: one
// ArgPair rides through the reduction and both the value and the first
// occurrence index come out of the same read.  This is the pair-tree sibling
// of PackedExtremumOps (float family); it carries the identical NaN/tie rules.
template <typename ScalarT, typename AccT, typename OutputT, bool MaxMode>
struct ExtremumOps {
    using pair_type = ArgPair<AccT>;
    using acc_type = ArgPair<AccT>;

    __device__ pair_type reduce(pair_type acc, ScalarT value, int64_t index) const {
        return better(acc, pair_type{static_cast<AccT>(value), index})
            ? acc : pair_type{static_cast<AccT>(value), index};
    }
    __device__ pair_type combine(pair_type a, pair_type b) const {
        return better(a, b) ? a : b;
    }
    __device__ OutputT project(pair_type value) const {
        return static_cast<OutputT>(value.value);
    }
    __device__ int64_t project_second(pair_type value) const {
        return value.index;
    }
    __device__ pair_type translate_idx(pair_type value, int64_t base_idx) const {
        return {value.value, value.index + base_idx};
    }

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

// Fused min+max for non-float aminmax: a MinMaxPair accumulator carries both
// extrema through one pass.  NaN semantics match MinMaxOps (a NaN claims its side
// and sticks).  The float family uses PackedAminMaxOps instead.
template <typename ScalarT, typename AccT, typename OutputT>
struct MinMaxPairOps {
    using pair_type = MinMaxPair<AccT>;
    using acc_type = MinMaxPair<AccT>;

    __device__ pair_type reduce(pair_type acc, ScalarT value, int64_t) const {
        const AccT converted = static_cast<AccT>(value);
        return combine(acc, pair_type{converted, converted});
    }
    __device__ pair_type combine(pair_type a, pair_type b) const {
        AccT min_val = a.min_val;
        AccT max_val = a.max_val;
        if (reduce_isnan(min_val)) {
            // keep NaN
        } else if (reduce_isnan(b.min_val)) {
            min_val = b.min_val;
        } else if (b.min_val < min_val) {
            min_val = b.min_val;
        }
        if (reduce_isnan(max_val)) {
            // keep NaN
        } else if (reduce_isnan(b.max_val)) {
            max_val = b.max_val;
        } else if (b.max_val > max_val) {
            max_val = b.max_val;
        }
        return {min_val, max_val};
    }
    __device__ OutputT project(pair_type value) const {
        return static_cast<OutputT>(value.max_val);
    }
    __device__ OutputT project_second(pair_type value) const {
        return static_cast<OutputT>(value.min_val);
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

// Monotone float -> uint32 map shared by the packed extremum forms: IEEE
// total order on the normalized bits (NaN family collapses to canonical
// qNaN, -0 folds onto +0).
__device__ __forceinline__ unsigned extremum_value_key(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        bits = 0x7FC00000u;  // NaN family -> canonical qNaN
    } else if (bits == 0x80000000u) {
        bits = 0u;           // fold -0 onto +0
    }
    const unsigned sign = static_cast<unsigned>(static_cast<int>(bits) >> 31);
    return bits ^ (0x80000000u | (0x7FFFFFFFu & sign));
}

// Inverse of extremum_value_key: top bit set means the source was
// non-negative (only the sign bit was flipped); otherwise every bit was.
__device__ __forceinline__ float extremum_key_value(unsigned key) {
    const unsigned bits = (key & 0x80000000u) ? key ^ 0x80000000u : ~key;
    return __uint_as_float(bits);
}

// Fused value+index extremum for max(dim)/min(dim): the whole reduction
// state is ONE 64-bit word — [stored value key (high 32) | ~index (low 32)]
// — selected with a plain integer compare.  One u64 shuffle per fold level
// replaces the two-shuffle pair tree an (value, int64) accumulator needs,
// and the NaN/tie rules live in the encoding instead of comparator branches:
//   * extremum_value_key is order-preserving, so integer max on the key is
//     float max; the min form stores ~key and takes integer min;
//   * qNaN outranks +inf for max and (via inversion) every finite key for
//     min — a NaN wins immediately and never moves off, with the first NaN
//     winning ties through the index half;
//   * ~index in the low half keeps the smaller index on equal keys (first
//     occurrence), for both directions;
//   * max form never sees key 0 or all-ones (both encode NaNs, which pack
//     to qNaN), so 0 / ~0ull serve as identities for max / min.
// Row length must fit int32 (host-side guard).  The value reconstructs
// exactly: float16/bfloat16 round-trip through float losslessly.
template <typename OutputT, bool kIsMax>
struct PackedExtremumOps {
    using acc_type = unsigned long long;

    __device__ static unsigned long long pack(float value, int64_t index) {
        unsigned key = extremum_value_key(value);
        if constexpr (!kIsMax) {
            // Integer-min direction: store the order-preserving key directly
            // (smaller value -> smaller key) and rebase NaN to key 0, below
            // every finite key, so a NaN wins immediately.  The low half
            // stores the plain index: on equal keys integer min keeps the
            // smaller index (first occurrence).
            if (reduce_isnan(value)) key = 0u;
            return (static_cast<unsigned long long>(key) << 32) |
                   static_cast<unsigned>(static_cast<unsigned>(index));
        } else {
            // Integer-max direction: NaN already maps to the largest key
            // (canonical qNaN beats +inf); ~index keeps the smaller index.
            return (static_cast<unsigned long long>(key) << 32) |
                   static_cast<unsigned>(~static_cast<unsigned>(index));
        }
    }

    template <typename V>
    __device__ unsigned long long reduce(
            unsigned long long acc, V value, int64_t index) const {
        const unsigned long long candidate = pack(
            static_cast<float>(value), index);
        if constexpr (kIsMax) return candidate > acc ? candidate : acc;
        else return candidate < acc ? candidate : acc;
    }

    __device__ unsigned long long combine(unsigned long long a,
                                          unsigned long long b) const {
        if constexpr (kIsMax) return a > b ? a : b;
        else return a < b ? a : b;
    }

    __device__ OutputT project(unsigned long long word) const {
        const unsigned key = static_cast<unsigned>(word >> 32);
        // The min direction stores the key un-inverted; key 0 encodes the
        // winning NaN and extremum_key_value(0) reconstructs a NaN bit
        // pattern, so no special case is needed here.
        return static_cast<OutputT>(extremum_key_value(key));
    }

    __device__ int64_t project_second(unsigned long long word) const {
        if constexpr (!kIsMax) {
            return static_cast<int64_t>(
                static_cast<unsigned>(word & 0xFFFFFFFFull));
        }
        const unsigned idx =
            ~static_cast<unsigned>(word & 0xFFFFFFFFull);
        return static_cast<int64_t>(static_cast<int32_t>(idx));
    }
};

// Packed argmin: the integer-min form of the extremum key map.  The high
// half stores the order-preserving key directly (smaller value -> smaller
// key), NaNs rebase to key 0 (below every finite key, so a NaN wins at
// once), and the low half stores the plain index so equal keys fall through
// to the first occurrence.
struct PackedArgMinOps {
    using acc_type = unsigned long long;

    template <typename V>
    __device__ unsigned long long reduce(
            unsigned long long acc, V value, int64_t index) const {
        const float converted = static_cast<float>(value);
        unsigned key = extremum_value_key(converted);
        if (reduce_isnan(converted)) key = 0u;
        const unsigned long long candidate =
            (static_cast<unsigned long long>(key) << 32) |
            static_cast<unsigned>(static_cast<unsigned>(index));
        return candidate < acc ? candidate : acc;
    }

    __device__ unsigned long long combine(unsigned long long a,
                                          unsigned long long b) const {
        return a < b ? a : b;
    }

    __device__ int64_t project(unsigned long long value) const {
        return static_cast<int64_t>(
            static_cast<unsigned>(value & 0xFFFFFFFFull));
    }
};

// Fused min+max for aminmax: ONE 64-bit word carries both extremum keys —
// [max key (high 32) | stored min key (low 32)] — so a single shuffle chain
// per fold level serves two reductions that would otherwise each re-read the
// whole input.  Each half is its own independent extremum scan:
//   * the high half keeps extremum_value_key as-is (order-preserving), so
//     integer max selects the largest value and canonical qNaN outranks +inf;
//   * the low half stores ~key (order-reversing), so the same integer max
//     selects the smallest value; NaN maps to all-ones there, outranking
//     every finite stored key, so a NaN claims both outputs;
//   * no real key lands on 0 or all-ones (the float key map never emits
//     either), so 0ull is the identity for both halves at once.
// The per-half max combine is componentwise, hence associative and
// order-independent.  Float family only (the key map is a float encoding);
// row length must fit int32 (host-side guard).
template <typename OutputT>
struct PackedAminMaxOps {
    using acc_type = unsigned long long;

    __device__ static unsigned long long pack(float value) {
        const unsigned key = extremum_value_key(value);
        const unsigned min_stored =
            (key == 0xFFC00000u) ? 0xFFFFFFFFu : ~key;
        return (static_cast<unsigned long long>(key) << 32) | min_stored;
    }

    __device__ static unsigned long long half_max(unsigned long long a,
                                                  unsigned long long b) {
        const unsigned a_hi = static_cast<unsigned>(a >> 32);
        const unsigned b_hi = static_cast<unsigned>(b >> 32);
        const unsigned hi = a_hi > b_hi ? a_hi : b_hi;
        const unsigned a_lo = static_cast<unsigned>(a);
        const unsigned b_lo = static_cast<unsigned>(b);
        const unsigned lo = a_lo > b_lo ? a_lo : b_lo;
        return (static_cast<unsigned long long>(hi) << 32) | lo;
    }

    template <typename V>
    __device__ unsigned long long reduce(
            unsigned long long acc, V value, int64_t) const {
        return half_max(acc, pack(static_cast<float>(value)));
    }

    __device__ unsigned long long combine(unsigned long long a,
                                          unsigned long long b) const {
        return half_max(a, b);
    }

    __device__ OutputT project(unsigned long long word) const {
        return static_cast<OutputT>(extremum_key_value(
            static_cast<unsigned>(word >> 32)));
    }

    __device__ OutputT project_second(unsigned long long word) const {
        const unsigned stored = static_cast<unsigned>(word);
        if (stored == 0xFFFFFFFFu) {
            // the all-ones min slot encodes the winning NaN
            return static_cast<OutputT>(extremum_key_value(0xFFC00000u));
        }
        return static_cast<OutputT>(extremum_key_value(~stored));
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

template <typename ScalarT, typename AccT, typename OutputT,
          typename FactorT = typename tensorplay::scalar_value_type<AccT>::type>
struct MeanOps {
    FactorT factor;
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
        else return static_cast<OutputT>(::sqrt(value));
    }
};

template <typename AccT, typename OutputT, typename IndexT = int64_t>
struct WelfordOps {
    AccT correction;
    bool take_sqrt;
    using acc_type = WelfordData<AccT, IndexT>;

    __device__ acc_type reduce(acc_type acc, AccT value, int64_t) const {
        const IndexT new_n = static_cast<IndexT>(acc.n + 1);
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
    // Companion output for a fused var/mean reduction: the accumulator already
    // carries the mean, so the second write costs one extra store per output.
    __device__ OutputT project_second(acc_type acc) const {
        return static_cast<OutputT>(acc.mean);
    }
    __device__ OutputT project(acc_type acc) const {
        const AccT divisor = acc.nf > correction ? acc.nf - correction : AccT(0);
        const AccT variance = acc.m2 / divisor;
        if (take_sqrt) {
            if constexpr (std::is_same_v<AccT, float>) {
                return static_cast<OutputT>(sqrtf(variance));
            } else {
                return static_cast<OutputT>(::sqrt(variance));
            }
        }
        return static_cast<OutputT>(variance);
    }
};

} // namespace reduction
} // namespace cuda
} // namespace tensorplay
