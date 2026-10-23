#include "ReduceKernels.cuh"
#include "CudaDispatchHelpers.cuh"
#include "DTypeNames.h"

namespace tensorplay {
namespace cuda {

// Cross-file entry in this backend: the values-only radix sort never
// materialises the permutation, so its passes move half the bytes.
Tensor sort_values_only_cuda(const Tensor& self, int64_t dim);

namespace {


template <typename T>
__device__ inline T reduce_empty_value() {
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return T(static_cast<float>(std::numeric_limits<float>::quiet_NaN()));
    } else {
        return std::numeric_limits<T>::lowest();
    }
}


// The selection kernels take their index type as a parameter: a radix pass
// carries counters, ballots and offsets in it, and the 64-bit form of all three
// is markedly more expensive than the 32-bit one.
template <typename T, typename index_t>
__global__ void nanmedian_select_flat_kernel(
        index_t n, const T* input, T* result) {
    __shared__ index_t radix_smem[32];
    __shared__ index_t nan_count;
    if (threadIdx.x == 0) nan_count = 0;
    __syncthreads();

    index_t local_nan_count = 0;
    for (index_t i = static_cast<index_t>(threadIdx.x); i < n;
         i += static_cast<index_t>(blockDim.x)) {
        local_nan_count += reduce_value_is_nan(input[i]) ? 1 : 0;
    }
    if (local_nan_count != 0) atomicAdd(&nan_count, local_nan_count);
    __syncthreads();

    const index_t valid = n - static_cast<index_t>(nan_count);
    if (valid == 0) {
        if (threadIdx.x == 0) result[0] = reduce_empty_value<T>();
        return;
    }
    const index_t k = (valid - 1) / 2 + 1;
    T median = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        input, k, false, n, static_cast<index_t>(1), radix_smem, &median);
    if (threadIdx.x == 0) result[0] = median;
}

// A NaN ranks as the largest value, and the radix sort's sign-magnitude order
// leaves a negatively signed NaN below every number, so NaNs occupy a prefix
// and a suffix of the ordered sequence.  Two boundary searches size the two
// runs, which is enough to aim at the middle of the values between them; no
// counting pass and no host round trip are needed.
template <typename T>
__global__ void nanmedian_pick_sorted_kernel(int64_t n, const T* sorted,
                                             T* result) {
    int64_t head = 0;
    int64_t hi = n;
    while (head < hi) {
        const int64_t mid = head + (hi - head) / 2;
        if (reduce_value_is_nan(sorted[mid])) {
            head = mid + 1;
        } else {
            hi = mid;
        }
    }
    int64_t tail = head;
    hi = n;
    while (tail < hi) {
        const int64_t mid = tail + (hi - tail) / 2;
        if (reduce_value_is_nan(sorted[mid])) {
            hi = mid;
        } else {
            tail = mid + 1;
        }
    }
    const int64_t valid = tail - head;
    result[0] = sorted[valid > 0 ? head + (valid - 1) / 2 : 0];
}


// Whether two entries are the same value for the purposes of a run or a match.
// A NaN counts as equal only to another NaN, so a stretch of NaNs is one run
// and a NaN never joins the run of the number it sits beside once ordered --
// reading this as "neither is less" would do exactly that, because every
// comparison with a NaN is false.
template <typename T>
__device__ inline bool same_ordered_value(T lhs, T rhs) {
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        return lhs == rhs || (::isnan(lhs) && ::isnan(rhs));
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return lhs == rhs ||
            (::isnan(static_cast<float>(lhs)) &&
             ::isnan(static_cast<float>(rhs)));
    } else {
        return lhs == rhs;
    }
}


// Smallest position holding the selected value.  The scan stops as soon as a
// lane reaches a position that is no better than the best found so far, so the
// shared minimum both collects the hits and prunes the rest of the walk.  A
// 32-bit minimum is a hardware instruction, and a slice never holds more than
// 2^31 values here, so the position fits in the slot that carries it.
template <typename T>
__device__ inline int64_t reduce_first_match(
        const T* slice_input, int64_t d_size, int64_t inner, T selected,
        int32_t* match_slot) {
    if (threadIdx.x == 0) {
        *match_slot = static_cast<int32_t>(d_size);
    }
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(blockDim.x);
    for (int64_t i = static_cast<int64_t>(threadIdx.x); i < d_size;
         i += stride) {
        if (i >= static_cast<int64_t>(*match_slot)) break;
        if (same_ordered_value(slice_input[i * inner], selected)) {
            atomicMin(match_slot, static_cast<int32_t>(i));
            break;
        }
    }
    __syncthreads();
    return static_cast<int64_t>(*match_slot);
}


template <typename T, typename index_t>
__global__ void median_select_dim_kernel(
        index_t n_slices, index_t d_size, index_t inner, const T* input,
        T* values, int64_t* indices, bool ignore_nan) {
    const index_t si = static_cast<index_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ index_t radix_smem[32];
    __shared__ unsigned long long nan_count;
    __shared__ int32_t match_slot;
    if (threadIdx.x == 0) {
        nan_count = 0;
    }
    __syncthreads();

    const index_t outer_index = si / inner;
    const index_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;
    index_t local_nan_count = 0;
    for (index_t i = static_cast<index_t>(threadIdx.x); i < d_size;
         i += static_cast<index_t>(blockDim.x)) {
        local_nan_count += reduce_value_is_nan(slice_input[i * inner]) ? 1 : 0;
    }
    if (local_nan_count != 0) atomicAdd(&nan_count, local_nan_count);
    __syncthreads();

    const index_t valid = d_size - static_cast<index_t>(nan_count);
    if (ignore_nan && valid == 0) {
        if (threadIdx.x == 0) {
            values[si] = reduce_empty_value<T>();
            indices[si] = 0;
        }
        return;
    }
    const index_t k = !ignore_nan && nan_count != 0
        ? d_size
        : (valid - 1) / 2 + 1;
    T median = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        slice_input, k, false, d_size, inner, radix_smem, &median);
    const int64_t first_match =
        reduce_first_match(slice_input, d_size, inner, median, &match_slot);
    if (threadIdx.x == 0) {
        values[si] = median;
        indices[si] = first_match;
    }
}


template <typename T, typename index_t>
__global__ void kthvalue_select_kernel(
        index_t n_slices, index_t d_size, index_t inner, index_t k,
        const T* input, T* values, int64_t* indices) {
    const index_t si = static_cast<index_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ index_t radix_smem[32];
    __shared__ int32_t match_slot;

    const index_t outer_index = si / inner;
    const index_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;
    T selected = static_cast<T>(0);
    topk_detail::topk_radix_select<T, index_t>(
        slice_input, k, false, d_size, inner, radix_smem, &selected);
    const int64_t first_match =
        reduce_first_match(slice_input, d_size, inner, selected, &match_slot);
    if (threadIdx.x == 0) {
        values[si] = selected;
        indices[si] = first_match;
    }
}


// NaNs rank as the largest values, while the sign-magnitude order puts a
// negatively signed NaN below every number and a positively signed one above.
// The NaNs therefore form a prefix and a suffix of the ordered sequence, and
// the two boundaries say how far each end reaches.
template <typename T>
__device__ inline void kthvalue_nan_bounds(
        const T* ordered, int64_t d_size, int64_t inner,
        int64_t& lead, int64_t& trail) {
    lead = 0;
    trail = d_size;
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value ||
                  std::is_same<T, Half>::value ||
                  std::is_same<T, BFloat16>::value) {
        int64_t lo = 0;
        int64_t hi = d_size;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (reduce_value_is_nan(ordered[mid * inner])) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lead = lo;
        hi = d_size;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (reduce_value_is_nan(ordered[mid * inner])) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        trail = lo;
    }
}

// A stable sort keeps equal values in input order, so the first position of
// the k-th value's run carries the smallest original index; one binary search
// finds where that run starts.  The leading NaN run is skipped over: those
// NaNs count as the largest values, not the smallest.
template <typename T>
__global__ void kthvalue_from_sorted_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner, int64_t k,
        const T* sorted, const int64_t* sorted_indices, T* values,
        int64_t* indices) {
    const int64_t si =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (si >= n_slices) return;
    const int64_t outer_index = si / inner;
    const int64_t base =
        outer_index * d_size * inner + (si - outer_index * inner);
    const T* source = sorted + base;
    const int64_t* source_indices = sorted_indices + base;
    int64_t lead = 0;
    int64_t trail = d_size;
    kthvalue_nan_bounds(source, d_size, inner, lead, trail);
    const int64_t valid = trail - lead;
    if (k > valid) {
        // The k-th value is a NaN; the smallest original index over all NaNs
        // is the first entry of whichever end run is non-empty.
        const int64_t trailing = trail < d_size ? trail : 0;
        int64_t position = trailing;
        if (lead > 0 && (trail == d_size || source_indices[0] <
                         source_indices[trailing * inner])) {
            position = 0;
        }
        values[si] = source[position * inner];
        indices[si] = source_indices[position * inner];
        return;
    }
    const int64_t target = lead + k - 1;
    const T value = source[target * inner];
    int64_t lo = lead;
    int64_t hi = target;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (same_ordered_value(source[mid * inner], value)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    values[si] = value;
    indices[si] = source_indices[lo * inner];
}


// The ordered slice is a chain of runs of equal values; the mode is the value
// of the longest run.  A run's start is where that value first appears, so
// bisecting the ordered values places every position inside its run without
// walking the run.  Only the block's sorted copy is searched, whose length is
// the power-of-two size, so 32-bit bounds cover every position.
template <typename T>
__device__ inline int32_t mode_run_start(
        const T* ordered, int32_t position) {
    const T value = ordered[position];
    int32_t lo = 0;
    int32_t hi = position;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (same_ordered_value(ordered[mid], value)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}


// Longest run among the per-thread candidates of one block.  Equal lengths are
// settled by position, not by the order the reduction happens to visit them: a
// shuffle butterfly moves a later lane's report into an earlier one before the
// two ever meet, so "keep unless the partner reports strictly more" would let a
// later run outrank an earlier one of the same length.  The earlier run is the
// run of the smaller value, which is what the ordered-route fold also reports.
// Threads past the last pair report the first position at length one, which is
// only ever chosen when every run is a single value.  The winner is broadcast
// through shared memory so every thread can use it.
__device__ inline int32_t mode_block_longest_run(int32_t length,
                                                 int32_t position) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int32_t other_length = __shfl_down_sync(0xffffffffu, length, offset);
        const int32_t other_position = __shfl_down_sync(0xffffffffu, position, offset);
        if (other_length > length ||
            (other_length == length && other_position < position)) {
            length = other_length;
            position = other_position;
        }
    }
    __shared__ int32_t warp_lengths[32];
    __shared__ int32_t warp_positions[32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        warp_lengths[warp] = length;
        warp_positions[warp] = position;
    }
    __syncthreads();
    const int warps = static_cast<int>(blockDim.x) >> 5;
    if (warp == 0) {
        length = lane < warps ? warp_lengths[lane] : 0;
        position = lane < warps ? warp_positions[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const int32_t other_length =
                __shfl_down_sync(0xffffffffu, length, offset);
            const int32_t other_position =
                __shfl_down_sync(0xffffffffu, position, offset);
            if (other_length > length ||
                (other_length == length && other_position < position)) {
                length = other_length;
                position = other_position;
            }
        }
        if (lane == 0) {
            warp_lengths[0] = length;
            warp_positions[0] = position;
        }
    }
    __syncthreads();
    return warp_positions[0];
}


// Folds the position each thread kept into one block-wide answer: `farthest`
// takes the largest reported position, otherwise the smallest, and -1 means
// the thread found none.  Warps reduce among themselves; thread 0 walks the
// per-warp results in lane order and leaves the winner in `result`.  Only
// thread 0 reads `result`, so no barrier follows the fold.  The index type is
// the widest position the caller can produce: a block that sorted a slice of
// at most 2048 values needs 32-bit shuffles, a whole-slice scan needs 64-bit
// ones.
template <typename IndexType>
__device__ inline void mode_block_span(IndexType own, bool farthest,
                                       IndexType* result) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const IndexType other = __shfl_down_sync(0xffffffffu, own, offset);
        if (farthest) {
            if (other > own) own = other;
        } else if (other >= 0 && (own < 0 || other < own)) {
            own = other;
        }
    }
    __shared__ IndexType warp_span[32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warp_span[warp] = own;
    __syncthreads();
    if (threadIdx.x == 0) {
        IndexType reported = warp_span[0];
        const int warps = static_cast<int>(blockDim.x) >> 5;
        for (int w = 1; w < warps; ++w) {
            const IndexType candidate = warp_span[w];
            if (farthest) {
                if (candidate > reported) reported = candidate;
            } else if (candidate >= 0 && (reported < 0 || candidate < reported)) {
                reported = candidate;
            }
        }
        *result = reported;
    }
}


__global__ void mode_bool_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner, bool keep_last,
        const bool* input, bool* values, int64_t* indices) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    __shared__ unsigned int true_count;
    __shared__ int64_t selected_index;
    if (threadIdx.x == 0) true_count = 0;
    __syncthreads();

    const int64_t outer_index = si / inner;
    const int64_t inner_index = si % inner;
    const bool* slice_input = input + outer_index * d_size * inner + inner_index;
    for (uint64_t i = static_cast<uint64_t>(threadIdx.x);
         i < static_cast<uint64_t>(d_size);
         i += static_cast<uint64_t>(blockDim.x)) {
        if (slice_input[i * inner]) atomicAdd(&true_count, 1u);
    }
    __syncthreads();

    const bool mode = true_count >
        static_cast<unsigned int>(d_size) - true_count;
    // Every thread walks the same number of steps so the fold after the loop
    // sees a converged warp; an out-of-range step reads the last element and
    // reports no match.
    const uint64_t stride = static_cast<uint64_t>(blockDim.x);
    const uint64_t length = static_cast<uint64_t>(d_size);
    const uint64_t steps = (length + stride - 1) / stride;
    int64_t best = -1;
    for (uint64_t step = 0; step < steps; ++step) {
        const uint64_t i = static_cast<uint64_t>(threadIdx.x) + step * stride;
        const uint64_t clamped = i < length ? i : length - 1;
        const bool match = i < length && slice_input[clamped * inner] == mode;
        const bool take = match && (keep_last || best < 0);
        best = take ? static_cast<int64_t>(i) : best;
    }
    mode_block_span(best, keep_last, &selected_index);
    if (threadIdx.x == 0) {
        values[si] = mode;
        indices[si] = selected_index;
    }
}


template <typename T>
__device__ __forceinline__ bool mode_value_less(T lhs, T rhs) {
    const bool lhs_nan = reduce_value_is_nan(lhs);
    const bool rhs_nan = reduce_value_is_nan(rhs);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (lhs_nan) return false;
    return lhs < rhs;
}


template <typename T>
__device__ __forceinline__ void mode_bitonic_swap(
        T& lhs, bool& lhs_valid, T& rhs, bool& rhs_valid, bool direction) {
    const bool should_swap =
        (mode_value_less(lhs, rhs) && lhs_valid) || !rhs_valid;
    if (should_swap == direction) {
        T value = lhs;
        lhs = rhs;
        rhs = value;
        const bool valid = lhs_valid;
        lhs_valid = rhs_valid;
        rhs_valid = valid;
    }
}


template <typename T, unsigned int Power2Size>
__device__ inline void mode_bitonic_sort(T* values, bool* valid) {
    for (unsigned int size = 2; size < Power2Size; size <<= 1) {
        const bool direction = (threadIdx.x & (size / 2)) != 0;
        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            const unsigned int position =
                2 * threadIdx.x - (threadIdx.x & (stride - 1));
            mode_bitonic_swap(
                values[position], valid[position],
                values[position + stride], valid[position + stride], direction);
        }
    }
    for (unsigned int stride = Power2Size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        const unsigned int position =
            2 * threadIdx.x - (threadIdx.x & (stride - 1));
        mode_bitonic_swap(
            values[position], valid[position],
            values[position + stride], valid[position + stride], false);
    }
    __syncthreads();
}


template <typename T, unsigned int Power2Size>
__global__ void mode_fused_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const T* input, T* values, int64_t* indices) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    extern __shared__ unsigned char storage[];
    T* sorted = reinterpret_cast<T*>(storage);
    bool* valid = reinterpret_cast<bool*>(sorted + Power2Size);
    const int64_t outer_index = si / inner;
    const int64_t inner_index = si % inner;
    const T* slice_input = input + outer_index * d_size * inner + inner_index;

    const unsigned int second = blockDim.x + threadIdx.x;
    if (threadIdx.x < Power2Size) {
        valid[threadIdx.x] = threadIdx.x < static_cast<unsigned int>(d_size);
        sorted[threadIdx.x] = valid[threadIdx.x]
            ? slice_input[static_cast<int64_t>(threadIdx.x) * inner]
            : static_cast<T>(0);
    }
    if (second < Power2Size) {
        valid[second] = second < static_cast<unsigned int>(d_size);
        sorted[second] = valid[second]
            ? slice_input[static_cast<int64_t>(second) * inner]
            : static_cast<T>(0);
    }
    __syncthreads();
    mode_bitonic_sort<T, Power2Size>(sorted, valid);

    // Each thread owns the adjacent ordered pair 2t and 2t+1, and a run is only
    // measured where it closes: an interior position reports the same value at a
    // shorter length, so it can never win.  Where the pair holds one value the
    // run it belongs to ends at the later slot, where they differ the earlier
    // slot closes its own run and the later one only opens the next, so one
    // search per pair covers both.  Every run end is a candidate, and equal
    // lengths settle on the earliest, which is the run of the smaller value.
    const int32_t first = static_cast<int32_t>(threadIdx.x) * 2;
    const int32_t low = first < d_size ? first : 0;
    const int32_t high = first + 1 < d_size ? first + 1 : low;
    const int32_t probe = same_ordered_value(sorted[low], sorted[high])
        ? high : low;
    // The slot after the last one of the slice is padding, so a probe sitting
    // there compares against itself and reads as an end.
    const int32_t after = probe + 1 < d_size ? probe + 1 : probe;
    const bool closed = probe == d_size - 1 ||
        !same_ordered_value(sorted[probe], sorted[after]);
    int32_t into_run = 0;
    if (closed) into_run = probe - mode_run_start(sorted, probe);
    // The search above is taken by some lanes and not others, so the warp is
    // explicitly reconverged before the fold reads across lanes.
    __syncwarp();
    const int32_t best_position = mode_block_longest_run(into_run, probe);

    // The reported position is the last occurrence of the mode: each thread
    // keeps the last match it owns and the block folds the per-thread results.
    __shared__ T mode;
    __shared__ int32_t mode_index;
    if (threadIdx.x == 0) mode = sorted[best_position];
    __syncthreads();
    // Uniform step count keeps the warp converged for the fold; an
    // out-of-range step rereads the last element and reports no match.
    const int32_t stride = static_cast<int32_t>(blockDim.x);
    const int32_t length = static_cast<int32_t>(d_size);
    const int32_t steps = (length + stride - 1) / stride;
    int32_t best = -1;
    for (int32_t step = 0; step < steps; ++step) {
        const int32_t i = static_cast<int32_t>(threadIdx.x) + step * stride;
        const int32_t clamped = i < length ? i : length - 1;
        const bool match = i < length &&
            same_ordered_value(slice_input[static_cast<int64_t>(clamped) * inner],
                               mode);
        best = match ? i : best;
    }
    mode_block_span<int32_t>(best, true, &mode_index);
    if (threadIdx.x == 0) {
        values[si] = mode;
        indices[si] = mode_index;
    }
}


// Slices up to 128 values use the 128-wide block, larger ones the 1024- or
// 2048-wide block.  Every block holds whole warps, which the reduction and its
// tie handling rely on.
template <typename T>
void launch_mode_fused(
        int64_t n_slices, int64_t d_size, int64_t inner,
        const Tensor& input, Tensor& values, Tensor& indices) {
    const int64_t power = d_size <= 128 ? 128 : d_size <= 1024 ? 1024 : 2048;
    const dim3 grid(static_cast<unsigned>(n_slices));
    auto stream = getCurrentCUDAStream().stream();
    switch (power) {
        case 128:
            mode_fused_kernel<T, 128><<<grid, 64, sizeof(T) * 128 + sizeof(bool) * 128, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
        case 1024:
            mode_fused_kernel<T, 1024><<<grid, 512, sizeof(T) * 1024 + sizeof(bool) * 1024, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
        default:
            mode_fused_kernel<T, 2048><<<grid, 1024, sizeof(T) * 2048 + sizeof(bool) * 2048, stream>>>(
                n_slices, d_size, inner, input.data_ptr<T>(),
                values.data_ptr<T>(), indices.data_ptr<int64_t>());
            break;
    }
}


// What one band of the ordered slice reports: the longest run closed inside the
// band and the runs touching its two ends, which may still join a neighbour.
// Positions name a place in the slice, so bands of a chunk and chunks of a slice
// fold with the same rule.
template <typename T>
struct ModeBandSummary {
    T first_value;
    T last_value;
    int64_t length;
    int64_t first_length;
    int64_t last_length;
    int64_t last_start;
    int64_t best_length;
    int64_t best_start;
};


// Equal lengths keep the earlier start: over an ordered slice that is the run of
// the smaller value, and a stable order puts the smallest original index at that
// run's front.
__device__ inline void mode_take_run(
        int64_t length, int64_t start, int64_t& best_length,
        int64_t& best_start) {
    if (length > 0 && (length > best_length ||
                       (length == best_length && start < best_start))) {
        best_length = length;
        best_start = start;
    }
}


// Combines two adjacent ordered bands into their union.  A run that reaches the
// left band's end keeps growing into the right band while the values match, and
// it only becomes a candidate for the longest run once it is closed on both
// sides — which is why the left band's trailing run is weighed here instead of
// being left to the caller.
template <typename T>
__device__ inline ModeBandSummary<T> mode_join(
        const ModeBandSummary<T>& left, const ModeBandSummary<T>& right) {
    if (left.length == 0) return right;
    if (right.length == 0) return left;
    ModeBandSummary<T> out{};
    out.length = left.length + right.length;
    out.first_value = left.first_value;
    out.last_value = right.last_value;
    out.best_length = left.best_length;
    out.best_start = left.best_start;
    const bool merge = same_ordered_value(left.last_value, right.first_value);
    // Only a left band given over to one run can pass its leading run on.
    out.first_length = left.first_length == left.length && merge
        ? left.length + right.first_length : left.first_length;
    // The union's trailing run is the right band's unless that band is spent on
    // its own leading run, in which case it is the growing join of both.
    const bool right_spent = right.first_length == right.length;
    out.last_length = merge && right_spent
        ? right.length + left.last_length : right.last_length;
    out.last_start = merge && right_spent ? left.last_start : right.last_start;
    mode_take_run(right.best_length, right.best_start, out.best_length,
                  out.best_start);
    // A joined run that reaches the right band's end is still open, so it
    // travels on as the union's trailing run rather than being weighed here.
    if (merge && !right_spent) {
        mode_take_run(left.last_length + right.first_length,
                      left.last_start, out.best_length, out.best_start);
    } else if (!merge) {
        mode_take_run(left.last_length, left.last_start, out.best_length,
                      out.best_start);
    }
    return out;
}


// One thread's contiguous band of an ordered slice.  Runs are weighed as they
// close inside the band; whatever reaches either end stays open for the fold.
template <typename T>
__device__ inline ModeBandSummary<T> mode_summarize_band(
        const T* source, int64_t inner, int64_t lo, int64_t hi) {
    ModeBandSummary<T> own{};
    own.length = hi - lo;
    if (own.length <= 0) return own;
    own.first_value = source[lo * inner];
    own.best_start = lo;
    T value = own.first_value;
    int64_t run_start = lo;
    int64_t run_length = 1;
    int64_t first_length = -1;
    for (int64_t j = lo + 1; j < hi; ++j) {
        const T next = source[j * inner];
        if (same_ordered_value(next, value)) {
            ++run_length;
            continue;
        }
        if (first_length < 0) first_length = run_length;
        mode_take_run(run_length, run_start, own.best_length, own.best_start);
        value = next;
        run_start = j;
        run_length = 1;
    }
    own.first_length = first_length < 0 ? own.length : first_length;
    own.last_value = value;
    own.last_length = run_length;
    own.last_start = run_start;
    return own;
}


// The folded summary of a whole slice answers it: the longest run closed inside
// it, or its trailing run, which is closed by the end of the slice itself.
template <typename T>
__device__ inline void mode_report_folded(
        const ModeBandSummary<T>& fold, const T* source,
        const int64_t* source_indices, int64_t inner, int64_t si, T* values,
        int64_t* indices) {
    int64_t best_length = fold.best_length;
    int64_t best_start = fold.best_start;
    mode_take_run(fold.last_length, fold.last_start, best_length, best_start);
    values[si] = source[best_start * inner];
    indices[si] = source_indices[best_start * inner];
}


// One block per slice and chunk, one band per thread.  Folding a chunk's bands
// in order lets a run that crosses a band boundary keep growing, and handing the
// chunk summaries to a second pass lets a long slice be measured by many blocks
// instead of by the one lane that owns it.  With a single chunk per slice there
// is nothing to hand on, so the block reports the answer itself.  The ordered
// sequence is stable, so a run's first position also holds the smallest original
// index, which is the one reported.
template <typename T>
__global__ void mode_chunk_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner, int64_t chunk,
        int64_t chunks_per_slice, const T* sorted,
        const int64_t* sorted_indices, T* values, int64_t* indices,
        ModeBandSummary<T>* summaries) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    const int64_t outer_index = si / inner;
    const int64_t base =
        outer_index * d_size * inner + (si - outer_index * inner);
    const T* source = sorted + base;
    const int64_t ci = static_cast<int64_t>(blockIdx.y);
    const int64_t lo = ci * chunk;
    const int64_t hi = lo + chunk < d_size ? lo + chunk : d_size;
    const int64_t band =
        (hi - lo + static_cast<int64_t>(blockDim.x) - 1) /
        static_cast<int64_t>(blockDim.x);
    const int64_t own_lo =
        lo + static_cast<int64_t>(threadIdx.x) * band < hi
        ? lo + static_cast<int64_t>(threadIdx.x) * band : hi;
    const int64_t own_hi = own_lo + band < hi ? own_lo + band : hi;

    __shared__ ModeBandSummary<T> bands[kThreads];
    bands[threadIdx.x] = mode_summarize_band(source, inner, own_lo, own_hi);
    __syncthreads();
    if (threadIdx.x != 0) return;

    ModeBandSummary<T> fold{};
    for (int64_t t = 0; t < static_cast<int64_t>(blockDim.x); ++t) {
        fold = mode_join(fold, bands[t]);
    }
    if (summaries != nullptr) {
        summaries[si * chunks_per_slice + ci] = fold;
        return;
    }
    mode_report_folded(fold, source, sorted_indices + base, inner, si, values,
                       indices);
}


// Joins the chunk summaries of one slice back into the slice answer.
template <typename T>
__global__ void mode_chunk_combine_kernel(
        int64_t n_slices, int64_t d_size, int64_t inner,
        int64_t chunks_per_slice, const T* sorted,
        const int64_t* sorted_indices, const ModeBandSummary<T>* summaries,
        T* values, int64_t* indices) {
    const int64_t si = static_cast<int64_t>(blockIdx.x);
    if (si >= n_slices) return;
    if (threadIdx.x != 0) return;
    const int64_t outer_index = si / inner;
    const int64_t base =
        outer_index * d_size * inner + (si - outer_index * inner);
    ModeBandSummary<T> fold{};
    const ModeBandSummary<T>* slice_summaries =
        summaries + si * chunks_per_slice;
    for (int64_t c = 0; c < chunks_per_slice; ++c) {
        fold = mode_join(fold, slice_summaries[c]);
    }
    mode_report_folded(fold, sorted + base, sorted_indices + base, inner, si,
                       values, indices);
}


Tensor nanmedian_cuda(const Tensor& self) {
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    DType work_dt = out_dt;
    if (isFloat8Type(work_dt)) work_dt = DType::Float32;
    if (self.numel() == 0) {
        Tensor result = Tensor::zeros({}, out_dt, self.device());
        if (isFloatingType(out_dt)) {
            return result.fill_(Scalar(std::numeric_limits<double>::quiet_NaN()));
        }
        return result.fill_(Scalar(std::numeric_limits<int64_t>::lowest()));
    }
    Tensor input = self.to(work_dt).contiguous().reshape({self.numel()});
    // The flat selection kernel is single-block; a full radix sort spreads
    // across the whole GPU and directly yields the median for large inputs.
    constexpr int64_t kNanMedianSortThreshold = int64_t{1} << 14;
    if (input.numel() > kNanMedianSortThreshold) {
        // Only ordered values are read back, so no permutation is carried.
        Tensor sorted = sort_values_only_cuda(input, 0);
        if (!isFloatingType(work_dt)) {
            return sorted.select(0, (input.numel() - 1) / 2);
        }
        Tensor result = Tensor::empty({}, work_dt, input.device());
        const auto stream = getCurrentCUDAStream().stream();
#define TP_NANMEDIAN_PICK_CASE(ctype, name_)                             \
    case DType::name_:                                                   \
        nanmedian_pick_sorted_kernel<ctype><<<1, 1, 0, stream>>>(        \
            input.numel(), static_cast<const ctype*>(sorted.data_ptr()), \
            static_cast<ctype*>(result.data_ptr()));                     \
        break;
        switch (work_dt) {
            TP_NANMEDIAN_PICK_CASE(float, Float32)
            TP_NANMEDIAN_PICK_CASE(double, Float64)
            TP_NANMEDIAN_PICK_CASE(Half, Float16)
            TP_NANMEDIAN_PICK_CASE(BFloat16, BFloat16)
            default: break;
        }
#undef TP_NANMEDIAN_PICK_CASE
        CUDA_CHECK(cudaGetLastError());
        return work_dt == out_dt ? result : result.to(out_dt);
    }
    Tensor result = Tensor::empty({}, work_dt, self.device());
    auto stream = getCurrentCUDAStream().stream();
    const bool flat32 =
        input.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_NANMEDIAN_FLAT_CASE(ctype, name_) \
    case DType::name_: \
        if (flat32) \
            nanmedian_select_flat_kernel<ctype, int32_t><<< \
                1, selection_threads(input.numel()), 0, stream>>>( \
                    static_cast<int32_t>(input.numel()), input.data_ptr<ctype>(), \
                    result.data_ptr<ctype>()); \
        else \
            nanmedian_select_flat_kernel<ctype, int64_t><<<1, selection_threads(input.numel()), 0, stream>>>( \
            input.numel(), input.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break;
    switch (work_dt) {
        TP_NANMEDIAN_FLAT_CASE(int64_t, Int64)
        TP_NANMEDIAN_FLAT_CASE(float, Float32)
        TP_NANMEDIAN_FLAT_CASE(double, Float64)
        TP_NANMEDIAN_FLAT_CASE(Half, Float16)
        TP_NANMEDIAN_FLAT_CASE(BFloat16, BFloat16)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype");
    }
#undef TP_NANMEDIAN_FLAT_CASE
    CUDA_CHECK(cudaGetLastError());
    return work_dt == out_dt ? result : result.to(out_dt);
}


std::tuple<Tensor, Tensor> nanmedian_dim_cuda(const Tensor& self, int64_t dim,
                                              bool keepdim) {
    const int64_t nd = self.dim();
    TP_CHECK(nd > 0,
             "nanmedian(): expects a tensor with at least one dimension");
    dim = wrap_dim(dim, nd);
    TP_CHECK(isFloatingType(self.dtype()),
             "nanmedian(): only floating point dtypes are supported");
    TP_CHECK(self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
                 self.dtype() == DType::Float32 || self.dtype() == DType::Float64,
             "nanmedian(): unsupported dtype ", toString(self.dtype()));
    Tensor input = self.contiguous();
    const int64_t d_size = input.size(dim);
    TP_CHECK(d_size > 0, "nanmedian(): Expected reduction dim ", dim,
             " to have non-zero size");
    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t slices = outer * inner;
    if (slices == 0) return {values, indices};
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_NANMEDIAN_DIM_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            median_select_dim_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), input.data_ptr<ctype>(), \
                values.data_ptr<ctype>(), indices.data_ptr<int64_t>(), true); \
        else \
            median_select_dim_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, input.data_ptr<ctype>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>(), true); \
        break;
    switch (input.dtype()) {
        TP_NANMEDIAN_DIM_CASE(Half, Float16)
        TP_NANMEDIAN_DIM_CASE(BFloat16, BFloat16)
        TP_NANMEDIAN_DIM_CASE(float, Float32)
        TP_NANMEDIAN_DIM_CASE(double, Float64)
        default: TP_THROW(TypeError, "nanmedian: unsupported dtype");
    }
#undef TP_NANMEDIAN_DIM_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


std::tuple<Tensor, Tensor> mode_cuda(const Tensor& self, int64_t dim, bool keepdim) {
    int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar mode input: ", dim);
        }
        Tensor values = Tensor::empty({}, self.dtype(), self.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, self.device());
        values.copy_(self);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    Tensor input = self.contiguous();
    int64_t d_size = input.size(dim);
    TP_CHECK(d_size > 0,
             "mode: expected reduction dimension to have non-zero size");
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t slices = outer * inner;
    if (slices == 0) return {values, indices};
    if (input.dtype() == DType::Bool) {
        auto stream = getCurrentCUDAStream().stream();
        // Short slices report the last occurrence of the mode, longer ones the
        // first; both name a position the value really occupies.
        const bool keep_last = d_size <= 2048;
        mode_bool_kernel<<<
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>(
            slices, d_size, inner, keep_last, input.data_ptr<bool>(),
            values.data_ptr<bool>(), indices.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    if (d_size >= 2 && d_size <= 2048) {
        switch (input.dtype()) {
#define TP_MODE_FUSED_CASE(ctype, name_) \
            case DType::name_: \
                launch_mode_fused<ctype>( \
                    slices, d_size, inner, input, values, indices); \
                break;
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_FUSED_CASE)
#undef TP_MODE_FUSED_CASE
            default:
                TP_THROW(TypeError, "mode: unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
        return {values, indices};
    }
    auto sorted_result = sort_cuda(input, dim, false);
    Tensor sorted = std::get<0>(sorted_result);
    Tensor sorted_indices = std::get<1>(sorted_result);
    auto stream = getCurrentCUDAStream().stream();
    // A chunk holds enough elements for a whole band per thread, up to one
    // chunk per lane; past that a slice is measured by more blocks and the
    // chunk summaries are joined by a second pass.
    constexpr int64_t kModeChunkElements = int64_t{kThreads} * 64;
    int64_t chunks = (d_size + kModeChunkElements - 1) / kModeChunkElements;
    if (chunks > kThreads) chunks = kThreads;
    const int64_t chunk = (d_size + chunks - 1) / chunks;
#define TP_MODE_DEVICE_CASE(ctype, name_) \
    case DType::name_: { \
        Tensor band_notes; \
        ModeBandSummary<ctype>* summaries = nullptr; \
        if (chunks > 1) { \
            band_notes = Tensor::empty( \
                {slices * chunks * static_cast<int64_t>( \
                    sizeof(ModeBandSummary<ctype>))}, \
                DType::UInt8, input.device()); \
            summaries = reinterpret_cast<ModeBandSummary<ctype>*>( \
                band_notes.data_ptr<uint8_t>()); \
        } \
        mode_chunk_kernel<ctype><<< \
            dim3(static_cast<unsigned>(slices), static_cast<unsigned>(chunks)), \
            kThreads, 0, stream>>>( \
            slices, d_size, inner, chunk, chunks, sorted.data_ptr<ctype>(), \
            sorted_indices.data_ptr<int64_t>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>(), summaries); \
        if (chunks > 1) { \
            mode_chunk_combine_kernel<ctype><<< \
                dim3(static_cast<unsigned>(slices)), 1, 0, stream>>>( \
                slices, d_size, inner, chunks, sorted.data_ptr<ctype>(), \
                sorted_indices.data_ptr<int64_t>(), summaries, \
                values.data_ptr<ctype>(), indices.data_ptr<int64_t>()); \
        } \
    } break;
    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_DEVICE_CASE)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODE_DEVICE_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


// The selector walks a slice once per radix digit from the single block that
// owns it, so a long slice leaves the rest of the GPU idle while its last digits
// are being read; a multi-block sort instead costs a fixed multiple of the whole
// tensor and answers the k-th value straight from the ordered run.  Both
// estimates are microseconds measured on this device for a four-byte element:
// the selector reads ~42us of one slice's 16k elements while up to ~128 slices
// are in flight, settles at ~0.035us per thousand elements once the blocks fill
// the GPU, and the sort plus its lookup starts at ~60us of launches and moves
// ~0.29us per thousand elements.  The element width shifts the two routes about
// equally, so the comparison is left in element units.
inline bool kthvalue_prefer_sort(int64_t slices, int64_t d_size) {
    const double elements =
        static_cast<double>(slices) * static_cast<double>(d_size);
    const double waves = static_cast<double>((slices + 127) / 128);
    double select = 42.0 * (static_cast<double>(d_size) / 16384.0) * waves;
    const double throughput = 0.035 * elements / 1000.0;
    if (throughput > select) select = throughput;
    const double sort = 60.0 + 0.29 * elements / 1000.0;
    return sort < select;
}


std::tuple<Tensor, Tensor> kthvalue_cuda(const Tensor& self, int64_t k, int64_t dim,
                                         bool keepdim) {
    Tensor input = self.contiguous();
    int64_t nd = input.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range for scalar kthvalue input: ", dim);
        }
        if (k != 1) {
            TP_THROW(RuntimeError,
                     "kthvalue(): selected number k out of range for dim 0");
        }
        Tensor values = Tensor::empty({}, input.dtype(), input.device());
        Tensor indices = Tensor::zeros({}, DType::Int64, input.device());
        values.copy_(input);
        return {values, indices};
    }
    dim = wrap_dim(dim, nd);
    int64_t d_size = input.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values_out = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices_out = Tensor::empty(out_shape, DType::Int64, input.device());
    if (input.numel() == 0) return {values_out, indices_out};
    if (input.dtype() == DType::Bool) {
        Tensor selected_values;
        Tensor selected_indices;
        std::tie(selected_values, selected_indices) =
            sort_cuda(input, dim, false);
        Tensor values = selected_values.select(dim, k - 1);
        Tensor indices = selected_indices.select(dim, k - 1);
        if (keepdim) {
            values = values.unsqueeze(dim);
            indices = indices.unsqueeze(dim);
        }
        return {values, indices};
    }

    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    const int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    // The per-slice selector reads its slice once per radix digit from a single
    // block; the sort route is picked only where that latency outweighs it.
    if (kthvalue_prefer_sort(slices, d_size)) {
        Tensor sorted;
        Tensor sorted_indices;
        std::tie(sorted, sorted_indices) = sort_cuda(input, dim, false);
#define TP_KTHVALUE_SORTED_CASE(ctype, name_)                              \
    case DType::name_:                                                     \
        kthvalue_from_sorted_kernel<ctype><<<make_grid(slices), kThreads,   \
                                             0, stream>>>(                 \
            slices, d_size, inner, k,                                      \
            static_cast<const ctype*>(sorted.data_ptr()),                   \
            sorted_indices.data_ptr<int64_t>(),                            \
            static_cast<ctype*>(values_out.data_ptr()),                    \
            indices_out.data_ptr<int64_t>());                              \
        break;
        switch (input.dtype()) {
            TP_KTHVALUE_SORTED_CASE(uint8_t, UInt8)
            TP_KTHVALUE_SORTED_CASE(int8_t, Int8)
            TP_KTHVALUE_SORTED_CASE(int16_t, Int16)
            TP_KTHVALUE_SORTED_CASE(int32_t, Int32)
            TP_KTHVALUE_SORTED_CASE(int64_t, Int64)
            TP_KTHVALUE_SORTED_CASE(uint16_t, UInt16)
            TP_KTHVALUE_SORTED_CASE(uint32_t, UInt32)
            TP_KTHVALUE_SORTED_CASE(uint64_t, UInt64)
            TP_KTHVALUE_SORTED_CASE(Half, Float16)
            TP_KTHVALUE_SORTED_CASE(BFloat16, BFloat16)
            TP_KTHVALUE_SORTED_CASE(float, Float32)
            TP_KTHVALUE_SORTED_CASE(double, Float64)
#undef TP_KTHVALUE_SORTED_CASE
            default:
                TP_THROW(NotImplementedError, "kthvalue: unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
        return {values_out, indices_out};
    }
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_KTHVALUE_SELECT_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            kthvalue_select_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), static_cast<int32_t>(k), \
                input.data_ptr<ctype>(), values_out.data_ptr<ctype>(), \
                indices_out.data_ptr<int64_t>()); \
        else \
            kthvalue_select_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, k, input.data_ptr<ctype>(), \
            values_out.data_ptr<ctype>(), indices_out.data_ptr<int64_t>()); \
        break;
    switch (input.dtype()) {
        TP_KTHVALUE_SELECT_CASE(uint8_t, UInt8)
        TP_KTHVALUE_SELECT_CASE(int8_t, Int8)
        TP_KTHVALUE_SELECT_CASE(int16_t, Int16)
        TP_KTHVALUE_SELECT_CASE(int32_t, Int32)
        TP_KTHVALUE_SELECT_CASE(int64_t, Int64)
        TP_KTHVALUE_SELECT_CASE(uint16_t, UInt16)
        TP_KTHVALUE_SELECT_CASE(uint32_t, UInt32)
        TP_KTHVALUE_SELECT_CASE(uint64_t, UInt64)
        TP_KTHVALUE_SELECT_CASE(Half, Float16)
        TP_KTHVALUE_SELECT_CASE(BFloat16, BFloat16)
        TP_KTHVALUE_SELECT_CASE(float, Float32)
        TP_KTHVALUE_SELECT_CASE(double, Float64)
#undef TP_KTHVALUE_SELECT_CASE
        default:
            TP_THROW(NotImplementedError, "kthvalue: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return {values_out, indices_out};
}


std::tuple<Tensor, Tensor> median_dim_cuda(const Tensor& self, int64_t dim,
                                           bool keepdim) {
    Tensor input = self.contiguous();
    const int64_t nd = input.dim();
    if (nd == 0) return kthvalue_cuda(input, 1, dim, keepdim);
    dim = wrap_dim(dim, nd);
    const int64_t d_size = input.size(dim);
    const int64_t k = (d_size + 1) / 2;
    const bool selection_supported =
        isIntegralType(input.dtype()) ||
        input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16 ||
        input.dtype() == DType::Float32 || input.dtype() == DType::Float64;
    if (!selection_supported) return kthvalue_cuda(input, k, dim, keepdim);

    std::vector<int64_t> out_shape = shape_of(input);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor values = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    if (input.numel() == 0) return {values, indices};

    int64_t outer = 1;
    int64_t inner = 1;
    outer_inner(shape_of(input), dim, outer, inner);
    const int64_t slices = outer * inner;
    auto stream = getCurrentCUDAStream().stream();
    const bool sel32 = input.numel() <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
#define TP_MEDIAN_DIM_CASE(ctype, name_) \
    case DType::name_: \
        if (sel32) \
            median_select_dim_kernel<ctype, int32_t><<< \
                dim3(static_cast<unsigned>(slices)), selection_threads(d_size), \
                0, stream>>>( \
                static_cast<int32_t>(slices), static_cast<int32_t>(d_size), \
                static_cast<int32_t>(inner), input.data_ptr<ctype>(), \
                values.data_ptr<ctype>(), \
                indices.data_ptr<int64_t>(), false); \
        else \
            median_select_dim_kernel<ctype, int64_t><<< \
            dim3(static_cast<unsigned>(slices)), selection_threads(d_size), 0, stream>>>( \
            slices, d_size, inner, input.data_ptr<ctype>(), values.data_ptr<ctype>(), \
            indices.data_ptr<int64_t>(), false); \
        break;
    switch (input.dtype()) {
        TP_MEDIAN_DIM_CASE(uint8_t, UInt8)
        TP_MEDIAN_DIM_CASE(int8_t, Int8)
        TP_MEDIAN_DIM_CASE(int16_t, Int16)
        TP_MEDIAN_DIM_CASE(int32_t, Int32)
        TP_MEDIAN_DIM_CASE(int64_t, Int64)
        TP_MEDIAN_DIM_CASE(uint16_t, UInt16)
        TP_MEDIAN_DIM_CASE(uint32_t, UInt32)
        TP_MEDIAN_DIM_CASE(uint64_t, UInt64)
        TP_MEDIAN_DIM_CASE(Half, Float16)
        TP_MEDIAN_DIM_CASE(BFloat16, BFloat16)
        TP_MEDIAN_DIM_CASE(float, Float32)
        TP_MEDIAN_DIM_CASE(double, Float64)
        default: return kthvalue_cuda(input, k, dim, keepdim);
    }
#undef TP_MEDIAN_DIM_CASE
    CUDA_CHECK(cudaGetLastError());
    return {values, indices};
}


std::tuple<Tensor, Tensor> interop_kthvalue_values_cuda(const Tensor& self, int64_t k, int64_t dim, bool keepdim,
              Tensor& values, Tensor& indices) {
        {
            auto __tp_result = kthvalue_cuda(self, k, dim, keepdim);
            write_out(values, std::get<0>(__tp_result));
            write_out(indices, std::get<1>(__tp_result));
        }
        return {values, indices};

}


std::tuple<Tensor, Tensor> interop_median_dim_values_cuda(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    {
        auto __tp_result = median_dim_cuda(self, dim, keepdim);
        write_out(values, std::get<0>(__tp_result));
        write_out(indices, std::get<1>(__tp_result));
    }
    return {values, indices};
}


std::tuple<Tensor, Tensor> interop_nanmedian_dim_values_cuda(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& values,
    Tensor& indices) {
    {
        auto __tp_result = nanmedian_dim_cuda(self, dim, keepdim);
        write_out(values, std::get<0>(__tp_result));
        write_out(indices, std::get<1>(__tp_result));
    }
    return {values, indices};
}

// Every listed dimension is reduced; an empty list reduces them all.
// Integer and boolean inputs answer in the default floating type.
Tensor logsumexp_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const bool integral = isIntegralType(self.dtype(), true);
    if (!isFloatingType(self.dtype()) && !integral)
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type, got ",
                 scalarTypeName(self.dtype()));
    const int64_t nd = self.dim();
    std::vector<int64_t> dims;
    std::vector<bool> seen(static_cast<size_t>(std::max<int64_t>(nd, 1)), false);
    for (int64_t d : dim) {
        const int64_t wrapped = wrap_dim(d, std::max<int64_t>(nd, 1));
        TP_CHECK(!seen[static_cast<size_t>(wrapped)], "logsumexp(): dim ", wrapped,
                 " appears multiple times in the list of dims");
        seen[static_cast<size_t>(wrapped)] = true;
        dims.push_back(wrapped);
    }
    if (nd == 0) {
        // A zero-dim tensor reduces along dim 0 or -1 as one value along one
        // axis, and one value is its own log-sum-exp.
        return integral ? self.to(globalContext().defaultDType()) : self.clone();
    }
    if (dims.empty()) {
        for (int64_t i = 0; i < nd; ++i) dims.push_back(i);
    }
    Tensor sc = self.contiguous();
    if (integral) sc = sc.to(globalContext().defaultDType());
    if (sc.numel() == 0) {
        return sum_dim_kernel(sc.exp(), dims, keepdim, sc.dtype()).log();
    }
    Tensor max_keep = amax_dim_kernel(sc, dims, true);
    const Scalar infinity(std::numeric_limits<double>::infinity());
    Tensor safe_max = Tensor::where(max_keep.abs().eq(infinity), Scalar(0), max_keep);
    Tensor summed = sum_dim_kernel((sc - safe_max).exp(), dims, keepdim, sc.dtype());
    Tensor result = summed.log();
    return result + (keepdim ? safe_max : safe_max.reshape(result.shape()));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SummaryOpsKernels) {
    m.impl("logsumexp", logsumexp_cuda2);
    m.impl("nanmedian", nanmedian_cuda);
    m.impl("nanmedian.dim", nanmedian_dim_cuda);
    m.impl("nanmedian.dim_values", interop_nanmedian_dim_values_cuda);
    m.impl("mode", mode_cuda);
    m.impl("kthvalue", kthvalue_cuda);
    m.impl("kthvalue.values", interop_kthvalue_values_cuda);
    m.impl("median.dim", median_dim_cuda);
    m.impl("median.dim_values", interop_median_dim_values_cuda);
}

namespace {


// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

int embedding_summary_multiprocessor_count() {
    static thread_local int cached_device = -1;
    static thread_local int cached_count = 1;
    const int device = currentDevice();
    if (device != cached_device) {
        int value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess) {
            value = 1;
        }
        cached_count = value > 0 ? value : 1;
        cached_device = device;
    }
    return cached_count;
}

// A range that collapsed to a single value is widened by one in the compute
// type the binning runs in, so the value still lands inside the histogram.
// Widening in double instead would move the edges the binning never sees.
void histc_expand_constant_range(DType dtype, double& lo, double& hi) {
    if (dtype == DType::Float64) {
        lo -= 1.0;
        hi += 1.0;
        return;
    }
    lo = static_cast<double>(static_cast<float>(lo) - 1.0f);
    hi = static_cast<double>(static_cast<float>(hi) + 1.0f);
}


// ---------------------------------------------------------------------------
// histc: bin edges are equally spaced over [min, max]; values outside the
// range are dropped, the rightmost edge is inclusive.
// ---------------------------------------------------------------------------

// Bin edges span [lo, hi] and the rightmost edge is inclusive, so a value at
// hi maps to the last bin.  NaN fails both bounds and drops.  The edge test
// runs in the input's own compute precision -- a double-precision edge test
// would send values sitting on a boundary to a different bin than the same
// input gets at single precision.
template <typename ComputeT>
__device__ __forceinline__ int histc_bin(ComputeT value, ComputeT lo,
                                        ComputeT hi, int bins) {
    if (!(value >= lo) || !(value <= hi)) return -1;
    int b = static_cast<int>((value - lo) * bins / (hi - lo));
    return b == bins ? bins - 1 : b;
}

// Staged histogram: every block keeps its own bin counters in shared memory and
// publishes them once at the end.  Straight to global memory, the atomics of a
// narrow histogram all land on the same handful of addresses and serialize --
// the shared copy turns those into per-block races that resolve in shared
// memory, and leaves one global atomic per bin per block.
template <typename InT, typename ComputeT, typename CountT>
__global__ void histc_shared_kernel(const InT* input, int64_t n, ComputeT lo,
                                    ComputeT hi, int bins,
                                    unsigned long long* counts) {
    extern __shared__ unsigned char histc_smem[];
    CountT* local = reinterpret_cast<CountT*>(histc_smem);
    for (int i = threadIdx.x; i < bins; i += blockDim.x) local[i] = CountT(0);
    __syncthreads();

    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const int b = histc_bin(static_cast<ComputeT>(input[i]), lo, hi, bins);
        if (b >= 0) atomicAdd(local + b, CountT(1));
    }
    __syncthreads();

    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        const CountT v = local[i];
        if (v != CountT(0)) atomicAdd(counts + i, static_cast<unsigned long long>(v));
    }
}

// Fallback for bin counts that do not fit in shared memory: one pass over the
// input with every vote going straight to the output.
template <typename InT, typename ComputeT>
__global__ void histc_count_kernel(const InT* input, int64_t n, ComputeT lo,
                                   ComputeT hi, int bins,
                                   unsigned long long* counts) {
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const int b = histc_bin(static_cast<ComputeT>(input[i]), lo, hi, bins);
        if (b >= 0) atomicAdd(counts + b, 1ull);
    }
}


Tensor interop_histc_cuda(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max) {
    if (bins <= 0) TP_THROW(RuntimeError, "histc(): bins must be positive");
    // Integer inputs compute in double precision and report counts in the
    // input dtype; that keeps the CUDA contract wider than the CPU one.
    const bool promote_to_f64 = !isFloatingType(self.dtype());
    if (promote_to_f64 && !isIntegralType(self.dtype(), /*includeBool=*/false)) {
        TP_THROW(TypeError, "histc(): expected a floating-point tensor, got ",
                 toString(self.dtype()));
    }
    double lo = min.toDouble();
    double hi = max.toDouble();
    if (lo == hi && self.numel() > 0) {
        auto extrema = ops::aminmax(self);
        lo = std::get<0>(extrema).item().toDouble();
        hi = std::get<1>(extrema).item().toDouble();
    }
    if (lo == hi) {
        histc_expand_constant_range(self.dtype(), lo, hi);
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        TP_THROW(RuntimeError, "histc: range of [", lo, ", ", hi,
                 "] is not finite");
    }
    if (!(lo < hi)) TP_THROW(RuntimeError, "histc: max must be larger than min");

    Tensor counts = Tensor::empty({bins}, DType::Int64, self.device());
    cudaStream_t stream = getCurrentCUDAStream().stream();
    const cudaError_t hist_memset = cudaMemsetAsync(
        counts.data_ptr(), 0, sizeof(int64_t) * static_cast<size_t>(bins),
        stream);
    if (hist_memset != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("CUDA Error: ") + cudaGetErrorString(hist_memset));
    }
    const Tensor flat = self.reshape({-1});
    const int64_t n = flat.numel();
    if (n > 0) {

        unsigned long long* counts_ptr =
            static_cast<unsigned long long*>(counts.data_ptr());
        const void* in_ptr = flat.data_ptr();
        // Staging costs one bin counter per bin per block and one global atomic
        // per bin per block at the end, so the grid is sized to keep that
        // publishing traffic a small fraction of the shared-memory votes: aim
        // for kHistcGlobalVoteRatio votes per published count, but never fewer
        // blocks than there are multiprocessors, and never so many that a block
        // runs dry before the next wave.
        constexpr int64_t kHistcGlobalVoteRatio = 64;
        constexpr int64_t kHistcMinBlocksPerSM = 4;
        const int threads = 256;
        const size_t shared_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);
        const size_t smem_limit = 32 * 1024;
        const bool staged = shared_bytes <= smem_limit;
        const int sms = embedding_summary_multiprocessor_count();
        const int64_t ceiling_blocks = (n + threads - 1) / threads;
        int64_t grid = ceiling_blocks;
        if (staged) {
            // Every block publishes one atomic per bin, so the block count is
            // what decides how much of the traffic lands in global memory:
            // hold it to a small fraction of the votes, and let the floor keep
            // enough blocks resident to cover the machine.
            const int64_t publish_bound =
                n / (kHistcGlobalVoteRatio * static_cast<int64_t>(bins));
            grid = std::max<int64_t>(publish_bound,
                                     sms * kHistcMinBlocksPerSM);
            grid = std::min<int64_t>(grid, ceiling_blocks);
        } else {
            grid = std::min<int64_t>(grid, 4096);
        }
        const int grid_i = static_cast<int>(std::max<int64_t>(grid, 1));
        // Single-precision inputs bin in single precision, doubles in double.
        const bool wide = self.dtype() == DType::Float64;
        const float lo_f = static_cast<float>(lo);
        const float hi_f = static_cast<float>(hi);
#define TP_HISTC_CASE(ctype, name)                                            \
    case DType::name:                                                         \
        if (staged) {                                                         \
            if (wide) {                                                       \
                histc_shared_kernel<ctype, double, unsigned int>               \
                    <<<grid_i, threads, shared_bytes, stream>>>(                 \
                        static_cast<const ctype*>(in_ptr), n, lo, hi, bins,    \
                        counts_ptr);                                          \
            } else {                                                          \
                histc_shared_kernel<ctype, float, unsigned int>                \
                    <<<grid_i, threads, shared_bytes, stream>>>(                 \
                        static_cast<const ctype*>(in_ptr), n, lo_f, hi_f,      \
                        static_cast<int>(bins), counts_ptr);                  \
            }                                                                  \
        } else {                                                              \
            if (wide) {                                                       \
                histc_count_kernel<ctype, double><<<grid_i, threads, 0, stream>>>( \
                    static_cast<const ctype*>(in_ptr), n, lo, hi, bins,        \
                    counts_ptr);                                              \
            } else {                                                          \
                histc_count_kernel<ctype, float><<<grid_i, threads, 0, stream>>>(  \
                    static_cast<const ctype*>(in_ptr), n, lo_f, hi_f,          \
                    static_cast<int>(bins), counts_ptr);                      \
            }                                                                  \
        }                                                                      \
        break;
        switch (self.dtype()) {
            TP_HISTC_CASE(double, Float64)
            TP_HISTC_CASE(float, Float32)
            TP_HISTC_CASE(tensorplay::Half, Float16)
            TP_HISTC_CASE(tensorplay::BFloat16, BFloat16)
            TP_HISTC_CASE(int64_t, Int64)
            TP_HISTC_CASE(int32_t, Int32)
            TP_HISTC_CASE(int16_t, Int16)
            TP_HISTC_CASE(int8_t, Int8)
            TP_HISTC_CASE(uint8_t, UInt8)
            TP_HISTC_CASE(uint16_t, UInt16)
            TP_HISTC_CASE(uint32_t, UInt32)
            TP_HISTC_CASE(uint64_t, UInt64)
            default:
                TP_THROW(NotImplementedError,
                         "histc(): unsupported dtype '" +
                             std::string(toString(self.dtype())) + "'");
        }
#undef TP_HISTC_CASE
        const cudaError_t hist_err = cudaGetLastError();
        if (hist_err != cudaSuccess) {
            TP_THROW(RuntimeError,
                     std::string("CUDA Error: ") + cudaGetErrorString(hist_err));
        }
    }
    return counts.to(self.dtype());
}


Tensor& interop_histc_out_cuda(const Tensor& self, int64_t bins, const Scalar& min,
                               const Scalar& max, Tensor& out) {
    if (out.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "histc(): out tensor must have the same dtype as the input");
    }
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "histc(): out tensor must be on the same device as the input");
    }
    Tensor r = interop_histc_cuda(self, bins, min, max);
    out.resize_(static_cast<std::vector<int64_t>>(r.shape()));
    out.copy_(r);
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SummaryOpsHistc) {
    // histogram / diagonal index builders
    m.impl("histc", interop_histc_cuda);
    m.impl("histc.out", interop_histc_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
