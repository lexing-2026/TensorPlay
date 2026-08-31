#pragma once

#include "DType.h"
#include <cuda_runtime.h>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_store.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cuda {
namespace topk_detail {

template <typename IndexType = uint64_t>
__device__ inline IndexType topk_linear_block_id() {
  return static_cast<IndexType>(blockIdx.x) +
      static_cast<IndexType>(gridDim.x) * blockIdx.y +
      static_cast<IndexType>(gridDim.x) * gridDim.y * blockIdx.z;
}

template <typename T>
__device__ inline bool topk_is_nan_device(T value) {
  if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    return isnan(value);
  } else if constexpr (std::is_same<T, Half>::value ||
                       std::is_same<T, BFloat16>::value) {
    return isnan(static_cast<float>(value));
  } else {
    return false;
  }
}

template <typename T>
__device__ inline T topk_padding_device(bool largest) {
  if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    return largest ? static_cast<T>(-INFINITY) : static_cast<T>(NAN);
  } else if constexpr (std::is_same<T, Half>::value ||
                       std::is_same<T, BFloat16>::value) {
    return T(largest ? -INFINITY : NAN);
  } else {
    return largest ? std::numeric_limits<T>::lowest()
                   : std::numeric_limits<T>::max();
  }
}

template <typename T>
struct TopKSortValue {
  T value;
  int64_t index;
};

template <typename T>
struct TopKStridedReadAccessor {
  const T* pointer;
  int64_t stride;

  __device__ __forceinline__ const T& operator[](int64_t index) const {
    return pointer[index * stride];
  }

  __device__ __forceinline__ TopKStridedReadAccessor operator+(
      int64_t offset) const {
    return {pointer + offset * stride, stride};
  }
};

template <typename T>
struct TopKStridedWriteAccessor {
  T* pointer;
  int64_t stride;

  __device__ __forceinline__ T& operator[](int64_t index) const {
    return pointer[index * stride];
  }

  __device__ __forceinline__ TopKStridedWriteAccessor operator+(
      int64_t offset) const {
    return {pointer + offset * stride, stride};
  }
};

template <typename T, bool Largest>
struct TopKMergeSortComparator {
  __device__ __forceinline__ bool operator()(T lhs, T rhs) const {
    const bool lhs_nan = topk_is_nan_device(lhs);
    const bool rhs_nan = topk_is_nan_device(rhs);
    if (lhs_nan != rhs_nan) return Largest ? lhs_nan : rhs_nan;
    if (lhs_nan) return false;
    if (lhs < rhs) return !Largest;
    if (rhs < lhs) return Largest;
    return false;
  }
};

template <typename T, bool Largest>
struct TopKBitonicComparator {
  __device__ __forceinline__ bool operator()(
      T lhs, int64_t lhs_index, T rhs, int64_t rhs_index) const {
    const bool lhs_nan = topk_is_nan_device(lhs);
    const bool rhs_nan = topk_is_nan_device(rhs);
    if (lhs_nan != rhs_nan) return Largest ? lhs_nan : rhs_nan;
    if (lhs_nan) return lhs_index < rhs_index;
    if (lhs < rhs) return !Largest;
    if (rhs < lhs) return Largest;
    return lhs_index < rhs_index;
  }
};

template <typename Comparator, typename K, typename V>
__device__ __forceinline__ void topk_bitonic_swap(
    K& key_a, V& value_a, bool& valid_a,
    K& key_b, V& value_b, bool& valid_b,
    bool direction, const Comparator& comparator) {
  const bool should_swap =
      (comparator(key_a, value_a, key_b, value_b) && valid_a) || !valid_b;
  if (should_swap == direction) {
    K key = key_a;
    key_a = key_b;
    key_b = key;
    V value = value_a;
    value_a = value_b;
    value_b = value;
    bool valid = valid_a;
    valid_a = valid_b;
    valid_b = valid;
  }
}

template <int SortSize, int BlockThreads, typename T, typename Comparator>
__device__ __forceinline__ void topk_bitonic_sort(
    T* values, int64_t* indices, bool* valid, const Comparator& comparator) {
  for (unsigned int size = 2; size < SortSize; size *= 2) {
    const bool direction = ((threadIdx.x & (size / 2)) != 0);
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();
      const unsigned int position =
          2 * threadIdx.x - (threadIdx.x & (stride - 1));
      topk_bitonic_swap(
          values[position], indices[position], valid[position],
          values[position + stride], indices[position + stride],
          valid[position + stride], direction, comparator);
    }
  }
  for (unsigned int stride = SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();
    const unsigned int position =
        2 * threadIdx.x - (threadIdx.x & (stride - 1));
    topk_bitonic_swap(
        values[position], indices[position], valid[position],
        values[position + stride], indices[position + stride],
        valid[position + stride], false, comparator);
  }
  __syncthreads();
}

template <typename T, int MaxBlockY>
__launch_bounds__(16 * MaxBlockY)
__global__ void bitonic_sort_selected_kernel(
    T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t rows, int64_t k, int64_t inner, bool largest) {
  constexpr int sort_size = 32;
  __shared__ T shared_values[MaxBlockY][sort_size];
  __shared__ int64_t shared_indices[MaxBlockY][sort_size];
  __shared__ bool shared_valid[MaxBlockY][sort_size];

  const uint64_t block_index = topk_linear_block_id();
  if (block_index * blockDim.y >= static_cast<uint64_t>(rows)) return;
  const int64_t row = static_cast<int64_t>(block_index) * blockDim.y +
      threadIdx.y;
  const bool row_valid = row < rows;
  const int64_t outer = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t base = outer * k * inner + inner_index;

  for (int item = 0; item < 2; ++item) {
    const int position = threadIdx.x + item * 16;
    const bool valid = row_valid && position < k;
    shared_values[threadIdx.y][position] = valid
        ? values[base + static_cast<int64_t>(position) * inner]
        : static_cast<T>(0);
    shared_indices[threadIdx.y][position] = valid
        ? indices[base + static_cast<int64_t>(position) * inner]
        : static_cast<int64_t>(-1);
    shared_valid[threadIdx.y][position] = valid;
  }

  if (largest) {
    topk_bitonic_sort<sort_size, 16>(
        shared_values[threadIdx.y], shared_indices[threadIdx.y],
        shared_valid[threadIdx.y], TopKBitonicComparator<T, true>());
  } else {
    topk_bitonic_sort<sort_size, 16>(
        shared_values[threadIdx.y], shared_indices[threadIdx.y],
        shared_valid[threadIdx.y], TopKBitonicComparator<T, false>());
  }

  if (!row_valid) return;
  for (int item = 0; item < 2; ++item) {
    const int position = threadIdx.x + item * 16;
    if (position < k) {
      values[base + static_cast<int64_t>(position) * inner] =
          shared_values[threadIdx.y][position];
      indices[base + static_cast<int64_t>(position) * inner] =
          shared_indices[threadIdx.y][position];
    }
  }
}

template <typename T, int SortSize, int MaxBlockY>
__launch_bounds__(32 * MaxBlockY)
__global__ void warp_merge_sort_selected_kernel(
    T* __restrict__ values, int64_t* __restrict__ indices, int64_t rows,
    int64_t k, int64_t inner, bool largest) {
  constexpr int items_per_thread = SortSize / 32;
  using LoadValues = cub::WarpLoad<
      T, items_per_thread, cub::WARP_LOAD_TRANSPOSE>;
  using LoadIndices = cub::WarpLoad<
      int64_t, items_per_thread, cub::WARP_LOAD_TRANSPOSE>;
  using Sort = cub::WarpMergeSort<T, items_per_thread, 32, int64_t>;
  using StoreValues = cub::WarpStore<
      T, items_per_thread, cub::WARP_STORE_TRANSPOSE>;
  using StoreIndices = cub::WarpStore<
      int64_t, items_per_thread, cub::WARP_STORE_TRANSPOSE>;
  __shared__ union {
    typename LoadValues::TempStorage load_values;
    typename LoadIndices::TempStorage load_indices;
    typename Sort::TempStorage sort;
    typename StoreValues::TempStorage store_values;
    typename StoreIndices::TempStorage store_indices;
  } temp_storage[MaxBlockY];

  const int64_t row = static_cast<int64_t>(topk_linear_block_id()) * blockDim.y +
      static_cast<int64_t>(threadIdx.y);
  if (row >= rows) return;
  auto& warp_storage = temp_storage[threadIdx.y];
  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t base = outer_index * k * inner + inner_index;
  const T padding = topk_padding_device<T>(largest);
  const int valid_items = static_cast<int>(k);
  TopKStridedReadAccessor<T> value_input{values + base, inner};
  TopKStridedReadAccessor<int64_t> index_input{indices + base, inner};
  TopKStridedWriteAccessor<T> value_output{values + base, inner};
  TopKStridedWriteAccessor<int64_t> index_output{indices + base, inner};
  T local_values[items_per_thread];
  int64_t local_indices[items_per_thread];

  LoadValues(warp_storage.load_values).Load(
      value_input, local_values, valid_items, padding);
  __syncwarp();
  LoadIndices(warp_storage.load_indices).Load(
      index_input, local_indices, valid_items, int64_t{-1});
  __syncwarp();
  if (largest) {
    Sort(warp_storage.sort).StableSort(
        local_values, local_indices, TopKMergeSortComparator<T, true>(),
        valid_items, padding);
  } else {
    Sort(warp_storage.sort).StableSort(
        local_values, local_indices, TopKMergeSortComparator<T, false>(),
        valid_items, padding);
  }
  __syncwarp();
  StoreValues(warp_storage.store_values).Store(
      value_output, local_values, valid_items);
  __syncwarp();
  StoreIndices(warp_storage.store_indices).Store(
      index_output, local_indices, valid_items);
}

template <typename T, typename Key, int BlockThreads, int ItemsPerThread>
__global__ void radix_sort_selected_kernel(
    T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t rows, int64_t k, int64_t inner, bool largest) {
  using ValueSort = cub::BlockRadixSort<Key, BlockThreads, ItemsPerThread,
                                        TopKSortValue<T>>;
  __shared__ typename ValueSort::TempStorage temp_storage;
  const int64_t row = static_cast<int64_t>(topk_linear_block_id());
  if (row >= rows) return;

  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t base = outer_index * k * inner + inner_index;
  Key keys[ItemsPerThread];
  TopKSortValue<T> items[ItemsPerThread];
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int64_t position = static_cast<int64_t>(threadIdx.x) *
        ItemsPerThread + item;
    if (position < k) {
      const int64_t offset = base + position * inner;
      items[item].value = values[offset];
      items[item].index = indices[offset];
      keys[item] = TopKRadixTraits<T>::encode(items[item].value);
    } else {
      items[item].value = static_cast<T>(0);
      items[item].index = -1;
      keys[item] = largest ? static_cast<Key>(0) : ~static_cast<Key>(0);
    }
  }

  if (largest) {
    ValueSort(temp_storage).SortDescending(keys, items);
  } else {
    ValueSort(temp_storage).Sort(keys, items);
  }
  __syncthreads();
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int64_t position = static_cast<int64_t>(threadIdx.x) *
        ItemsPerThread + item;
    if (position < k) {
      const int64_t offset = base + position * inner;
      values[offset] = items[item].value;
      indices[offset] = items[item].index;
    }
  }
}

template <typename T, typename Key, int BlockThreads, int ItemsPerThread>
__launch_bounds__(BlockThreads)
__global__ void radix_sort_all_topk_kernel(
    const T* __restrict__ input, T* __restrict__ values,
    int64_t* __restrict__ indices, int64_t rows, int64_t cols, int64_t k,
    int64_t inner, bool largest) {
  using ValueSort = cub::BlockRadixSort<
      Key, BlockThreads, ItemsPerThread, TopKSortValue<T>>;
  using LoadValues = cub::BlockLoad<
      T, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>;
  using StoreValues = cub::BlockStore<
      T, BlockThreads, ItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;
  using StoreIndices = cub::BlockStore<
      int64_t, BlockThreads, ItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;
  __shared__ union {
    typename LoadValues::TempStorage load_values;
    typename ValueSort::TempStorage sort;
    typename StoreValues::TempStorage store_values;
    typename StoreIndices::TempStorage store_indices;
  } temp_storage;
  const uint64_t row_index = topk_linear_block_id();
  if (row_index >= static_cast<uint64_t>(rows)) return;

  const int64_t row = static_cast<int64_t>(row_index);
  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t input_base = outer_index * cols * inner + inner_index;
  const int64_t output_base = outer_index * k * inner + inner_index;
  TopKStridedReadAccessor<T> input_accessor{
      input + input_base, inner};
  TopKStridedWriteAccessor<T> value_accessor{
      values + output_base, inner};
  TopKStridedWriteAccessor<int64_t> index_accessor{
      indices + output_base, inner};
  T local_values[ItemsPerThread];
  int64_t local_indices[ItemsPerThread];
  Key keys[ItemsPerThread];
  TopKSortValue<T> items[ItemsPerThread];
  LoadValues(temp_storage.load_values).Load(
      input_accessor, local_values, static_cast<int>(cols), static_cast<T>(0));
  __syncthreads();
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int position = static_cast<int>(threadIdx.x) * ItemsPerThread + item;
    const bool valid = position < cols;
    if (valid) {
      items[item].value = local_values[item];
      items[item].index = position;
      local_indices[item] = position;
      keys[item] = TopKRadixTraits<T>::encode(local_values[item]);
    } else {
      items[item].value = static_cast<T>(0);
      items[item].index = -1;
      local_indices[item] = -1;
      keys[item] = largest ? static_cast<Key>(0) :
          std::numeric_limits<Key>::max();
    }
  }

  if (largest) {
    ValueSort(temp_storage.sort).SortDescending(keys, items);
  } else {
    ValueSort(temp_storage.sort).Sort(keys, items);
  }
  __syncthreads();
  for (int item = 0; item < ItemsPerThread; ++item) {
    local_values[item] = items[item].value;
    local_indices[item] = items[item].index;
  }
  StoreValues(temp_storage.store_values).Store(
      value_accessor, local_values, static_cast<int>(k));
  __syncthreads();
  StoreIndices(temp_storage.store_indices).Store(
      index_accessor, local_indices, static_cast<int>(k));
}

template <typename T, typename Key>
__global__ void topk_pack_sort_kernel(
    const T* __restrict__ values, Key* __restrict__ keys,
    int64_t* __restrict__ positions,
    int64_t rows, int64_t k, int64_t inner) {
  using Traits = TopKRadixTraits<T>;
  const int64_t total = rows * k;
  const int64_t first = static_cast<int64_t>(topk_linear_block_id()) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) *
      gridDim.x * gridDim.y * gridDim.z;
  for (int64_t position = first; position < total; position += stride) {
    const int64_t row = position / k;
    const int64_t column = position % k;
    const int64_t outer_index = row / inner;
    const int64_t inner_index = row % inner;
    const int64_t offset = outer_index * k * inner + column * inner +
        inner_index;
    keys[position] = Traits::encode(values[offset]);
    positions[position] = position;
  }
}

template <typename T, typename Key, int BlockThreads, int ItemsPerThread>
__launch_bounds__(BlockThreads)
__global__ void radix_sort_all_topk_indices_kernel(
    const T* __restrict__ input, T* __restrict__ values,
    int64_t* __restrict__ indices, int64_t rows, int64_t cols, int64_t k,
    int64_t inner, bool largest) {
  using IndexSort = cub::BlockRadixSort<
      Key, BlockThreads, ItemsPerThread, int32_t>;
  using LoadValues = cub::BlockLoad<
      T, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_TRANSPOSE>;
  __shared__ union {
    typename LoadValues::TempStorage load_values;
    typename IndexSort::TempStorage sort;
  } temp_storage;
  const uint64_t row_index = topk_linear_block_id();
  if (row_index >= static_cast<uint64_t>(rows)) return;

  const int64_t row = static_cast<int64_t>(row_index);
  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t input_base = outer_index * cols * inner + inner_index;
  const int64_t output_base = outer_index * k * inner + inner_index;
  TopKStridedReadAccessor<T> input_accessor{input + input_base, inner};
  T local_values[ItemsPerThread];
  int32_t local_indices[ItemsPerThread];
  Key keys[ItemsPerThread];
  LoadValues(temp_storage.load_values).Load(
      input_accessor, local_values, static_cast<int>(cols), static_cast<T>(0));
  __syncthreads();
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int position = static_cast<int>(threadIdx.x) * ItemsPerThread + item;
    const bool valid = position < cols;
    local_indices[item] = valid ? position : -1;
    keys[item] = valid ? TopKRadixTraits<T>::encode(local_values[item])
                       : (largest ? static_cast<Key>(0)
                                   : std::numeric_limits<Key>::max());
  }

  if (largest) {
    IndexSort(temp_storage.sort).SortDescending(keys, local_indices);
  } else {
    IndexSort(temp_storage.sort).Sort(keys, local_indices);
  }
  __syncthreads();
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int position = static_cast<int>(threadIdx.x) * ItemsPerThread + item;
    if (position < k) {
      const int32_t source = local_indices[item];
      const int64_t output_offset =
          output_base + static_cast<int64_t>(position) * inner;
      values[output_offset] = topk_load(
          input + input_base + static_cast<int64_t>(source) * inner);
      indices[output_offset] = static_cast<int64_t>(source);
    }
  }
}

__global__ void topk_fill_segment_offsets(
    uint32_t* __restrict__ offsets, uint32_t rows, uint32_t length) {
  const uint32_t index = static_cast<uint32_t>(topk_linear_block_id()) * blockDim.x +
      threadIdx.x;
  if (index <= rows) offsets[index] = index * length;
}

template <typename T>
__global__ void topk_unpack_sort_kernel(
    const T* __restrict__ values, const int64_t* __restrict__ indices,
    const int64_t* __restrict__ positions, T* __restrict__ sorted_values,
    int64_t* __restrict__ sorted_indices, int64_t rows, int64_t k,
    int64_t inner) {
  const int64_t total = rows * k;
  const int64_t first = static_cast<int64_t>(topk_linear_block_id()) * blockDim.x +
      threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) *
      gridDim.x * gridDim.y * gridDim.z;
  for (int64_t position = first; position < total; position += stride) {
    const int64_t source_position = positions[position];
    const int64_t source_row = source_position / k;
    const int64_t source_column = source_position % k;
    const int64_t source_outer = source_row / inner;
    const int64_t source_inner = source_row % inner;
    const int64_t source_offset = source_outer * k * inner +
        source_column * inner + source_inner;
    const int64_t row = position / k;
    const int64_t column = position % k;
    const int64_t outer_index = row / inner;
    const int64_t inner_index = row % inner;
    const int64_t output_offset = outer_index * k * inner +
        column * inner + inner_index;
    sorted_values[output_offset] = values[source_offset];
    sorted_indices[output_offset] = indices[source_offset];
  }
}
}
}
}
