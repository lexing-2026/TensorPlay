#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "CUDARuntime.h"
#include "GPUPrimitives.cuh"
#include "SortingRadixSelect.cuh"
#include "SortUtils.cuh"
#include "CUDALoops.cuh"

#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {

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
        for (int64_t j = 0; j < d_size; ++j) {
            vb[j * inner] = sp[j * inner];
            ib[j * inner] = j;
        }
        auto less = [&](int64_t a, int64_t b) {
            T va = vb[a * inner], vbv = vb[b * inner];
            bool lt = va < vbv, gt = va > vbv;
            return descending ? gt : lt;
        };
        auto swap_pair = [&](int64_t a, int64_t b) {
            T tv = vb[a * inner];
            vb[a * inner] = vb[b * inner];
            vb[b * inner] = tv;
            int64_t ti = ib[a * inner];
            ib[a * inner] = ib[b * inner];
            ib[b * inner] = ti;
        };
        auto sift_down = [&](int64_t start, int64_t end) {
            int64_t root = start;
            while (2 * root + 1 <= end) {
                int64_t child = 2 * root + 1;
                if (child + 1 <= end && less(child, child + 1)) ++child;
                if (less(root, child)) {
                    swap_pair(root, child);
                    root = child;
                } else {
                    break;
                }
            }
        };
        for (int64_t st = d_size / 2 - 1; st >= 0; --st) sift_down(st, d_size - 1);
        for (int64_t end = d_size - 1; end > 0; --end) {
            swap_pair(0, end);
            sift_down(0, end - 1);
        }
    }
}

template <typename T>
struct SortRadixTraits : topk_detail::TopKRadixTraits<T> {};

template <>
struct SortRadixTraits<bool> {
    using key_type = uint32_t;
    static constexpr int bit_count = 1;
    __device__ static inline key_type encode(bool value) { return value ? 1u : 0u; }
    __device__ static inline bool deconvert(key_type value) { return value != 0u; }
};

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

struct SortKeyLessOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a < b; }
};

struct SortKeyGreaterOp {
    template <typename K>
    __device__ __forceinline__ bool operator()(K a, K b) const { return a > b; }
};

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

    const int64_t slice = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
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
#define TP_SORT_BLOCK_CASE(CAP, IPT) \
    if (d_size <= CAP) { \
        sort_block_radix_kernel<T, CAP / IPT, IPT> \
            <<<grid, CAP / IPT, 0, stream>>>( \
                self_c.data_ptr<T>(), values.data_ptr<T>(), \
                indices.data_ptr<int64_t>(), slices, d_size, inner, \
                descending); \
        return; \
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
#define TP_SORT_BLOCK_TYPE(ctype, name) \
    case DType::name: \
        launch_sort_block_radix<ctype>( \
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
        if (pos != nullptr) pos[i] = j;
    }
}

template <typename T>
__global__ void sort_radix_unpack_kernel(int64_t n, int64_t d_size, int64_t inner,
                                         typename SortRadixTraits<T>::key_type const* keys,
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
        if (idxs != nullptr) idxs[dst] = pos[i];
    }
}

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
    sort_radix_pack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, static_cast<const T*>(self_c.data_ptr()),
        keys_a.data_ptr<Key>(), pos_a.data_ptr<int64_t>());
    const int n_items = static_cast<int>(n);
    if (slices != 1) {
        // Several independent runs need their segment bounds; a single run
        // sorts globally and never reads them.
        const int off_blocks =
            static_cast<int>((slices + 1 + kThreads - 1) / kThreads);
        sort_radix_fill_offsets_kernel<<<off_blocks, kThreads, 0, stream>>>(
            static_cast<int>(slices) + 1, d_size, offsets.data_ptr<int32_t>());
    }
    cub::DoubleBuffer<Key> key_buf(keys_a.data_ptr<Key>(), keys_b.data_ptr<Key>());
    cub::DoubleBuffer<int64_t> pos_buf(pos_a.data_ptr<int64_t>(), pos_b.data_ptr<int64_t>());
    const int bits = SortRadixTraits<T>::bit_count;
    size_t tmp_bytes = 0;
    cudaError_t err;
    if (slices == 1) {
        // A single run needs no segment bookkeeping, and one global sort
        // produces the same order.  The segmented kernel carries per-segment
        // state that dominates when there is only one segment to amortize it
        // over -- a 1-D sort is exactly that case.
        err = descending
            ? cub::DeviceRadixSort::SortPairsDescending(
                  nullptr, tmp_bytes, key_buf, pos_buf, n_items, 0, bits, stream)
            : cub::DeviceRadixSort::SortPairs(
                  nullptr, tmp_bytes, key_buf, pos_buf, n_items, 0, bits, stream);
        CUDA_CHECK(err);
        Tensor tmp = Tensor::empty(
            {static_cast<int64_t>(std::max<size_t>(tmp_bytes, 1))}, DType::UInt8,
            device);
        err = descending
            ? cub::DeviceRadixSort::SortPairsDescending(
                  tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, 0, bits,
                  stream)
            : cub::DeviceRadixSort::SortPairs(
                  tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items, 0, bits,
                  stream);
        CUDA_CHECK(err);
    } else {
        const int* begin_offsets = offsets.data_ptr<int32_t>();
        const int* end_offsets = begin_offsets + 1;
        const int n_segments = static_cast<int>(slices);
        err = descending
            ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
                  nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
                  begin_offsets, end_offsets, 0, bits, stream)
            : cub::DeviceSegmentedRadixSort::SortPairs(
                  nullptr, tmp_bytes, key_buf, pos_buf, n_items, n_segments,
                  begin_offsets, end_offsets, 0, bits, stream);
        CUDA_CHECK(err);
        Tensor tmp = Tensor::empty(
            {static_cast<int64_t>(std::max<size_t>(tmp_bytes, 1))}, DType::UInt8,
            device);
        err = descending
            ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
                  tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items,
                  n_segments, begin_offsets, end_offsets, 0, bits, stream)
            : cub::DeviceSegmentedRadixSort::SortPairs(
                  tmp.data_ptr(), tmp_bytes, key_buf, pos_buf, n_items,
                  n_segments, begin_offsets, end_offsets, 0, bits, stream);
        CUDA_CHECK(err);
    }
    sort_radix_unpack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, key_buf.Current(), pos_buf.Current(),
        static_cast<T*>(values.data_ptr()), indices.data_ptr<int64_t>());
}

// Keys-only radix sort for callers that never look at the permutation:
// carrying an int64 position through every radix pass doubles the traffic of
// the sort itself, and the sorted values are all such callers want.
template <typename T>
void sort_keys_radix_impl(const Tensor& self_c, Tensor& values,
                          int64_t d_size, int64_t inner, int64_t slices) {
    using Key = typename SortRadixTraits<T>::key_type;
    const int64_t n = self_c.numel();
    const auto device = self_c.device();
    const DType key_dtype = sizeof(Key) == 8 ? DType::UInt64 : DType::UInt32;
    Tensor keys_a = Tensor::empty({n}, key_dtype, device);
    Tensor keys_b = Tensor::empty({n}, key_dtype, device);
    auto stream = getCurrentCUDAStream().stream();
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    sort_radix_pack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, static_cast<const T*>(self_c.data_ptr()),
        keys_a.data_ptr<Key>(), static_cast<int64_t*>(nullptr));
    cub::DoubleBuffer<Key> key_buf(keys_a.data_ptr<Key>(), keys_b.data_ptr<Key>());
    const int n_items = static_cast<int>(n);
    const int bits = SortRadixTraits<T>::bit_count;
    size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, tmp_bytes, key_buf, n_items, 0, bits, stream));
    Tensor tmp = Tensor::empty(
        {static_cast<int64_t>(std::max<size_t>(tmp_bytes, 1))}, DType::UInt8,
        device);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        tmp.data_ptr(), tmp_bytes, key_buf, n_items, 0, bits, stream));
    sort_radix_unpack_kernel<T><<<blocks, kThreads, 0, stream>>>(
        n, d_size, inner, key_buf.Current(), static_cast<const int64_t*>(nullptr),
        static_cast<T*>(values.data_ptr()), static_cast<int64_t*>(nullptr));
    (void)slices;
}

void radix_sort_impl(const Tensor& self_c, Tensor& values, Tensor& indices,
                     int64_t, int64_t, int64_t inner, int64_t d_size,
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

// Maps every position of the sorted order back to its run: the run of a
// position is the last one whose start is at or before it.  Locating it by
// search costs a few dependent loads, which is cheaper than materialising a
// per-position group id for the whole input just to read it back here.
__global__ void unique_inverse_kernel(int64_t n, const int64_t* __restrict__ order,
                                      const int64_t* __restrict__ run_starts,
                                      int64_t num_runs,
                                      int64_t* __restrict__ inverse) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t low = 0;
    int64_t high = num_runs - 1;
    while (low < high) {
        const int64_t mid = (low + high + 1) >> 1;
        if (run_starts[mid] <= i) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    inverse[order[i]] = low;
}

template <typename T>
__global__ void unique_row_equal_kernel(int64_t n, int64_t row_len,
                                        const T* rows, int64_t* is_new) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (i == 0) {
            is_new[0] = 1;
            continue;
        }
        const T* cur = rows + i * row_len;
        const T* prev = rows + (i - 1) * row_len;
        int64_t same = 1;
        for (int64_t c = 0; c < row_len; ++c) {
            if (cur[c] != prev[c]) {
                same = 0;
                break;
            }
        }
        is_new[i] = same ? 0 : 1;
    }
}

template <typename T>
__global__ void unique_row_emit_kernel(int64_t n, int64_t row_len,
                                       const T* rows, const int64_t* order,
                                       const int64_t* gid, T* out,
                                       int64_t* inverse) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const int64_t row = order[i];
        const int64_t g = gid[i] - 1;
        const T* src = rows + i * row_len;
        T* dst = out + g * row_len;
        for (int64_t c = 0; c < row_len; ++c) dst[c] = src[c];
        if (inverse != nullptr) inverse[row] = g;
    }
}

}

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
    if (self_c.numel() <= std::numeric_limits<int>::max() &&
        d_size >= 2 && d_size <= 4096 && slices > 1) {
        constexpr int64_t kMaxStridedInner = 16;
        if (inner > kMaxStridedInner) {
            std::vector<int64_t> order(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) order[static_cast<size_t>(d)] = d;
            std::swap(order[static_cast<size_t>(dim)],
                      order[static_cast<size_t>(nd - 1)]);
            Tensor staged = self_c.permute(order).contiguous();
            Tensor staged_values = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                staged.dtype(), staged.device());
            Tensor staged_indices = Tensor::empty(
                static_cast<std::vector<int64_t>>(staged.shape()),
                DType::Int64, staged.device());
            sort_block_radix_entry(staged, staged_values, staged_indices,
                                   slices, d_size, 1, descending);
            std::vector<int64_t> inverse(static_cast<size_t>(nd));
            for (int64_t d = 0; d < nd; ++d) {
                inverse[static_cast<size_t>(order[static_cast<size_t>(d)])] = d;
            }
            values.copy_(staged_values.permute(inverse));
            indices.copy_(staged_indices.permute(inverse));
        } else {
            sort_block_radix_entry(self_c, values, indices,
                                   slices, d_size, inner, descending);
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
    return std::get<1>(sort_cuda(self, dim, descending));
}

// Sorts along `dim` and returns only the values.  The permutation is never
// materialised, so the radix passes move half the bytes.
static Tensor sort_values_only_cuda(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor self_c = self.contiguous();
    Tensor values = Tensor::empty(
        static_cast<std::vector<int64_t>>(self_c.shape()), self_c.dtype(),
        self_c.device());
    int64_t d_size = self_c.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self_c.shape()), dim, outer, inner);
    const int64_t slices = outer * inner;
    if (slices == 0 || d_size == 0) return values;
    TP_CHECK(self_c.numel() <= std::numeric_limits<int>::max(),
             "sort: input is too large for the radix sort");
    switch (self_c.dtype()) {
#define TP_SORT_KEYS_CASE(ctype, name) \
    case DType::name: \
        sort_keys_radix_impl<ctype>(self_c, values, d_size, inner, slices); \
        break;
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SORT_KEYS_CASE)
#undef TP_SORT_KEYS_CASE
        default: TP_THROW(TypeError, "sort: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return values;
}

std::tuple<Tensor, Tensor, Tensor> unique_cuda(const Tensor& self, bool sorted,
                                               bool return_inverse,
                                               bool return_counts) {
    (void)sorted;
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

    // The permutation is only needed to place the inverse indices; without it
    // the sort carries no values and moves half the bytes per pass.
    Tensor sorted_vals;
    Tensor order;
    if (return_inverse) {
        auto sorted_pair = sort_cuda(flat, 0, false);
        sorted_vals = std::get<0>(sorted_pair);
        order = std::get<1>(sorted_pair);
    } else {
        sorted_vals = sort_values_only_cuda(flat, 0);
    }

    // One pass over the sorted values yields the distinct values themselves and
    // how many there are; the run starts come along only when the counts need
    // them.  Deriving the boundaries this way keeps a flags array and a prefix
    // sum over the whole input out of the picture -- both are as large as the
    // input, while the run table is as small as the output.
    Tensor values_buffer = Tensor::empty({n}, self.dtype(), self.device());
    // The run table serves both the counts and the inverse mapping.
    Tensor run_starts = (return_counts || return_inverse)
                            ? Tensor::empty({n}, DType::Int64, self.device())
                            : Tensor();
    Tensor num_selected = Tensor::empty({}, DType::Int64, self.device());
    const auto stream = getCurrentCUDAStream().stream();
    thrust::counting_iterator<int64_t> positions(0);
    size_t temp_bytes = 0;
    bool launched = false;
    // Without the counts there is nothing to gain from the run boundaries, and
    // the plain unique pass writes half as many outputs.
#define TP_UNIQUE_SPLIT_CASE(ctype, name)                                     \
    case DType::name: {                                                       \
        ctype* sorted_ptr = sorted_vals.data_ptr<ctype>();                    \
        size_t bytes = 0;                                                     \
        if (run_starts.defined()) {                                           \
            CUDA_CHECK(cub::DeviceSelect::UniqueByKey(                        \
                nullptr, bytes, sorted_ptr, positions,                        \
                values_buffer.data_ptr<ctype>(),                              \
                run_starts.data_ptr<int64_t>(),                               \
                num_selected.data_ptr<int64_t>(), static_cast<int>(n),        \
                stream));                                                     \
        } else {                                                              \
            CUDA_CHECK(cub::DeviceSelect::Unique(                             \
                nullptr, bytes, sorted_ptr,                                   \
                values_buffer.data_ptr<ctype>(),                              \
                num_selected.data_ptr<int64_t>(), static_cast<int>(n),        \
                stream));                                                     \
        }                                                                     \
        Tensor temp = Tensor::empty(                                          \
            {static_cast<int64_t>(std::max<size_t>(bytes, 1))},               \
            DType::UInt8, self.device());                                     \
        if (run_starts.defined()) {                                           \
            CUDA_CHECK(cub::DeviceSelect::UniqueByKey(                        \
                temp.data_ptr(), bytes, sorted_ptr, positions,                \
                values_buffer.data_ptr<ctype>(),                              \
                run_starts.data_ptr<int64_t>(),                               \
                num_selected.data_ptr<int64_t>(), static_cast<int>(n),        \
                stream));                                                     \
        } else {                                                              \
            CUDA_CHECK(cub::DeviceSelect::Unique(                             \
                temp.data_ptr(), bytes, sorted_ptr,                           \
                values_buffer.data_ptr<ctype>(),                              \
                num_selected.data_ptr<int64_t>(), static_cast<int>(n),        \
                stream));                                                     \
        }                                                                     \
        temp_bytes = bytes;                                                   \
        launched = true;                                                      \
        break;                                                                \
    }
    switch (self.dtype()) {
        TP_UNIQUE_SPLIT_CASE(float, Float32)
        TP_UNIQUE_SPLIT_CASE(double, Float64)
        TP_UNIQUE_SPLIT_CASE(int64_t, Int64)
        TP_UNIQUE_SPLIT_CASE(int32_t, Int32)
        TP_UNIQUE_SPLIT_CASE(int16_t, Int16)
        TP_UNIQUE_SPLIT_CASE(int8_t, Int8)
        TP_UNIQUE_SPLIT_CASE(uint8_t, UInt8)
        TP_UNIQUE_SPLIT_CASE(uint16_t, UInt16)
        TP_UNIQUE_SPLIT_CASE(uint32_t, UInt32)
        TP_UNIQUE_SPLIT_CASE(uint64_t, UInt64)
        TP_UNIQUE_SPLIT_CASE(Half, Float16)
        TP_UNIQUE_SPLIT_CASE(BFloat16, BFloat16)
        TP_UNIQUE_SPLIT_CASE(bool, Bool)
        default:
            TP_THROW(NotImplementedError, "unique: unsupported dtype on CUDA");
    }
#undef TP_UNIQUE_SPLIT_CASE
    (void)temp_bytes;
    (void)launched;

    // The output shape is only known on the device, so one element crosses the
    // bus; copying the whole run table back to read its last entry would cost
    // more than every kernel above put together.
    int64_t num_groups = 0;
    CUDA_CHECK(cudaMemcpyAsync(&num_groups, num_selected.data_ptr<int64_t>(),
                              sizeof(num_groups), cudaMemcpyDeviceToHost,
                              stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    values = values_buffer.slice(0, 0, num_groups);

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (return_inverse) {
        unique_inverse_kernel<<<blocks, threads, 0, stream>>>(
            n, order.data_ptr<int64_t>(), run_starts.data_ptr<int64_t>(),
            num_groups, inverse.data_ptr<int64_t>());
    }
    if (return_counts) {
        // A run's length is the gap to the next start; the last run ends at the
        // end of the input.
        counts = Tensor::empty({num_groups}, DType::Int64, self.device());
        const int64_t* starts = run_starts.data_ptr<int64_t>();
        gpu_kernel_with_index(
            counts, [=] GPU_LAMBDA(int64_t g) -> int64_t {
                const int64_t end = (g + 1 < num_groups) ? starts[g + 1] : n;
                return end - starts[g];
            });
    }
    CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(values, inverse, counts);
}

std::tuple<Tensor, Tensor> _unique_cuda(const Tensor& self, bool sorted,
                                        bool return_inverse) {
    auto result = unique_cuda(self, sorted, return_inverse, false);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<Tensor, Tensor, Tensor> _unique2_cuda(const Tensor& self, bool sorted,
                                                 bool return_inverse,
                                                 bool return_counts) {
    return unique_cuda(self, sorted, return_inverse, return_counts);
}

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
                 "Number of zero sized dimensions is more than one, so unique cannot be applied");
        Tensor values = Tensor::empty(sizes, self.dtype(), self.device());
        Tensor inverse = Tensor::empty({0}, DType::Int64, self.device());
        Tensor counts = Tensor::empty({0}, DType::Int64, self.device());
        return std::make_tuple(values, inverse, counts);
    }
    TP_CHECK(zero_dims == 0,
             "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");
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
        rows_sorted = input_flat;
        order = Tensor::arange(Scalar(int64_t(0)), Scalar(n), Scalar(int64_t(1)),
                               DType::Int64, self.device());
        for (int64_t c = row_len - 1; c >= 0; --c) {
            Tensor col = rows_sorted.slice(1, c, c + 1).reshape({n});
            Tensor col_sorted, col_order;
            std::tie(col_sorted, col_order) = sort_cuda(col, 0, false);
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
#define UNIQUE_ROW_CASE(ctype, name) \
    case DType::name: \
        unique_row_equal_kernel<ctype><<<blocks, threads, 0, stream>>>( \
            n, row_len, rows_sorted.data_ptr<ctype>(), flags.data_ptr<int64_t>()); \
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
            TP_THROW(NotImplementedError, "unique_dim: unsupported dtype on CUDA");
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
#define UNIQUE_ROW_EMIT_CASE(ctype, name) \
    case DType::name: \
        unique_row_emit_kernel<ctype><<<blocks, threads, 0, stream>>>( \
            n, row_len, rows_sorted.data_ptr<ctype>(), order.data_ptr<int64_t>(), \
            gid.data_ptr<int64_t>(), kept_rows.data_ptr<ctype>(), inverse_ptr); \
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
            TP_THROW(NotImplementedError, "unique_dim: unsupported dtype on CUDA");
    }
#undef UNIQUE_ROW_EMIT_CASE
    front_sizes[0] = num_groups;
    Tensor values = kept_rows.reshape(front_sizes).moveaxis(0, dim);
    Tensor counts;
    if (return_counts) {
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
    return unique_dim_cuda_impl(self, dim, false, return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_cuda(
        const Tensor& self, int64_t dim, bool return_inverse,
        bool return_counts) {
    return unique_dim_cuda_impl(self, dim, true, return_inverse, return_counts);
}

Tensor msort_cuda(const Tensor& self) {
    Tensor values = std::get<0>(self.sort(0, false));
    return values;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, SortingKernels) {
    m.impl("msort", msort_cuda);
    m.impl("sort", sort_cuda);
    m.impl("argsort", argsort_cuda);
    m.impl("unique", unique_cuda);
    m.impl("_unique", _unique_cuda);
    m.impl("_unique2", _unique2_cuda);
    m.impl("unique_dim", unique_dim_cuda);
    m.impl("unique_dim_consecutive", unique_dim_consecutive_cuda);
}

}
}
