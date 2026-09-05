#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#include "ScanUtils.cuh"
#include "SortingRadixSelect.cuh"
#include "SortUtils.cuh"
#include "GPUPrimitives.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace tensorplay {
namespace cuda {

namespace {
using namespace topk_detail;


const cudaDeviceProp& topk_device_properties() {
  static thread_local cudaDeviceProp properties{};
  static thread_local int device = -1;
  const int current = currentDevice();
  if (device != current) {
    TP_CUDA_CHECK(cudaGetDeviceProperties(&properties, current));
    device = current;
  }
  return properties;
}

dim3 topk_grid_for(uint64_t count) {
  const auto& properties = topk_device_properties();
  const uint64_t max_x = static_cast<uint64_t>(properties.maxGridSize[0]);
  const uint64_t max_y = static_cast<uint64_t>(properties.maxGridSize[1]);
  const uint64_t max_z = static_cast<uint64_t>(properties.maxGridSize[2]);
  const uint64_t x = std::max<uint64_t>(1, std::min(count, max_x));
  const uint64_t remaining = (count + x - 1) / x;
  const uint64_t y = std::max<uint64_t>(1, std::min(remaining, max_y));
  const uint64_t z = std::max<uint64_t>(1, (remaining + y - 1) / y);
  TP_CHECK(z <= max_z, "topk: grid exceeds device limits");
  return dim3(static_cast<unsigned>(x), static_cast<unsigned>(y),
              static_cast<unsigned>(z));
}

template <typename T, typename IndexType>
__device__ inline void topk_gather_topk(
    const T* input, T* values, int64_t* indices, IndexType input_base,
    IndexType output_base, IndexType slice_size, IndexType k,
    IndexType within_slice_stride,
    T top_k, bool largest, IndexType* smem) {
  using Key = typename TopKRadixTraits<T>::key_type;
  const Key top_k_converted = TopKRadixTraits<T>::encode(top_k);
  const IndexType iterations =
      ((slice_size + static_cast<IndexType>(blockDim.x) - 1) /
       static_cast<IndexType>(blockDim.x)) * static_cast<IndexType>(blockDim.x);
  IndexType write_index_start = 0;
  for (IndexType i = static_cast<IndexType>(threadIdx.x); i < iterations;
       i += static_cast<IndexType>(blockDim.x)) {
    const bool in_range = i < slice_size;
    const T value = in_range
        ? topk_load(&input[input_base + i * within_slice_stride])
        : static_cast<T>(0);
    const Key converted = TopKRadixTraits<T>::encode(value);
    const bool has_top_k = in_range &&
        (largest ? converted > top_k_converted : converted < top_k_converted);
    IndexType index = 0;
    IndexType carry = 0;
    topk_exclusive_binary_prefix_scan<IndexType, true>(
        smem, has_top_k, &index, &carry, TopKAddOp<IndexType>());
    if (has_top_k) {
      const IndexType write_index = write_index_start + index;
      values[output_base + write_index * within_slice_stride] = value;
      indices[output_base + write_index * within_slice_stride] =
          static_cast<int64_t>(i);
    }
    write_index_start += carry;
  }

  IndexType top_k_remaining = k - write_index_start;
  for (IndexType i = static_cast<IndexType>(threadIdx.x); i < iterations;
       i += static_cast<IndexType>(blockDim.x)) {
    const bool in_range = i < slice_size;
    const T value = in_range
        ? topk_load(&input[input_base + i * within_slice_stride])
        : static_cast<T>(0);
    const Key converted = TopKRadixTraits<T>::encode(value);
    const bool has_top_k = in_range && converted == top_k_converted;
    IndexType index = 0;
    IndexType carry = 0;
    topk_exclusive_binary_prefix_scan<IndexType, true>(
        smem, has_top_k, &index, &carry, TopKAddOp<IndexType>());
    if (has_top_k && index < top_k_remaining) {
      const IndexType write_index = write_index_start + index;
      values[output_base + write_index * within_slice_stride] = value;
      indices[output_base + write_index * within_slice_stride] =
          static_cast<int64_t>(i);
    }
    if (carry >= top_k_remaining) break;
    top_k_remaining -= carry;
    write_index_start += carry;
  }
}

template <typename T, typename IndexType>
__launch_bounds__(1024)
__global__ void radix_topk_kernel(
    const T* __restrict__ input,
    T* __restrict__ values,
    int64_t* __restrict__ indices,
    IndexType rows, IndexType cols, IndexType k, IndexType inner,
    bool largest) {
  __shared__ union {
    IndexType scan[32];
    T pattern[2];
  } smem_storage;
  IndexType* smem = smem_storage.scan;
  const IndexType row = topk_linear_block_id<IndexType>();
  if (row >= rows) return;

  const IndexType outer_index = row / inner;
  const IndexType inner_index = row % inner;
  const IndexType input_base = outer_index * cols * inner + inner_index;
  const IndexType output_base = outer_index * k * inner + inner_index;
  T top_k;
  topk_radix_select<T, IndexType>(
      input + input_base, static_cast<IndexType>(k), largest,
      static_cast<IndexType>(cols), static_cast<IndexType>(inner), smem,
      &top_k);
  topk_gather_topk<T, IndexType>(
      input, values, indices, input_base, output_base, cols, k, inner, top_k,
      largest, smem);
}

constexpr int topk_multiblock_threads = 256;
constexpr int topk_multiblock_radix_bits = 8;
constexpr int topk_multiblock_radix_size = 1 << topk_multiblock_radix_bits;
constexpr int topk_multiblock_radix_mask = topk_multiblock_radix_size - 1;
constexpr int topk_multiblock_min_items_per_thread = 4;
constexpr int topk_multiblock_max_items_per_thread = 64;

template <typename T>
__global__ void topk_multiblock_fill_kernel(
    T* __restrict__ output, T value, uint32_t size) {
  const uint64_t blocks = static_cast<uint64_t>(gridDim.x) * gridDim.y *
      gridDim.z;
  const uint64_t first = topk_linear_block_id() * blockDim.x + threadIdx.x;
  const uint64_t stride = blocks * blockDim.x;
  for (uint64_t i = first; i < size; i += stride) output[i] = value;
}

template <typename T, typename Key, typename IndexType>
__launch_bounds__(topk_multiblock_threads)
__global__ void topk_multiblock_digit_counts(
    const T* __restrict__ input, const uint32_t* __restrict__ ranks,
    const Key* __restrict__ desired_values, Key desired_mask,
    uint16_t* __restrict__ counts, uint32_t rows, uint32_t cols,
    IndexType inner, uint32_t blocks_per_row, int items_per_thread,
    int digit_pos) {
  using Traits = TopKRadixTraits<T>;
  __shared__ uint32_t digit_counts[topk_multiblock_radix_size];
  const uint32_t block_index = topk_linear_block_id<uint32_t>();
  const uint32_t num_blocks = rows * blocks_per_row;
  if (block_index >= num_blocks) return;
  const uint32_t row = static_cast<uint32_t>(block_index / blocks_per_row);
  const uint32_t block_in_row = static_cast<uint32_t>(block_index % blocks_per_row);
  const IndexType row_index = static_cast<IndexType>(row);
  const IndexType outer_index = row_index / inner;
  const IndexType inner_index = row_index % inner;
  const IndexType input_base = outer_index * static_cast<IndexType>(cols) * inner +
      inner_index;
  const int items_per_block = items_per_thread * topk_multiblock_threads;
  const IndexType block_base = static_cast<IndexType>(block_in_row) *
      static_cast<IndexType>(items_per_block);
  const int items = block_in_row + 1 < blocks_per_row
      ? items_per_thread
      : static_cast<int>((static_cast<int64_t>(cols) - block_base +
                          topk_multiblock_threads - 1) /
                         topk_multiblock_threads);
  const Key desired = __ldg(desired_values + row);
  if (threadIdx.x < topk_multiblock_radix_size) digit_counts[threadIdx.x] = 0;
  __syncthreads();
  for (int item = 0; item < items; ++item) {
    const IndexType column = block_base +
        static_cast<IndexType>(item) * topk_multiblock_threads + threadIdx.x;
    if (column < static_cast<IndexType>(cols)) {
      const Key key = Traits::encode(
          topk_load(&input[input_base + column * inner]));
      if ((key & desired_mask) == (desired & desired_mask)) {
        const int bucket = static_cast<int>((key >> digit_pos) &
                                            static_cast<Key>(topk_multiblock_radix_mask));
        atomicAdd(&digit_counts[bucket], 1u);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < topk_multiblock_radix_size) {
    counts[block_index * topk_multiblock_radix_size + threadIdx.x] =
        static_cast<uint16_t>(digit_counts[threadIdx.x]);
  }
  (void)ranks;
}

template <typename T, typename Key>
__launch_bounds__(topk_multiblock_radix_size)
__global__ void topk_multiblock_within_k_counts(
    const uint16_t* __restrict__ counts,
    const Key* __restrict__ desired_in, const uint32_t* __restrict__ ranks_in,
    Key* __restrict__ desired_out, uint32_t* __restrict__ ranks_out,
    uint32_t rows, uint32_t blocks_per_row, int digit_pos,
    bool largest, uint32_t* __restrict__ within_k_counts,
    T* __restrict__ kth_values) {
  using Traits = TopKRadixTraits<T>;
  using Scan = cub::BlockScan<uint32_t, topk_multiblock_radix_size>;
  __shared__ union {
    uint32_t digit_count_cumsum[topk_multiblock_radix_size];
    typename Scan::TempStorage scan_storage;
  } temp_storage;
  __shared__ Key desired;
  const uint32_t block_index = topk_linear_block_id<uint32_t>();
  const uint32_t num_blocks = rows * blocks_per_row;
  if (block_index >= num_blocks) return;
  const uint32_t row = static_cast<uint32_t>(block_index / blocks_per_row);
  const uint32_t block_in_row = static_cast<uint32_t>(block_index % blocks_per_row);
  uint32_t rank = __ldg(ranks_in + row);
  uint32_t digit_count = 0;
  if (threadIdx.x < topk_multiblock_radix_size) {
    for (uint32_t block = 0; block < blocks_per_row; ++block) {
      digit_count += __ldg(counts +
          (row * blocks_per_row + block) * topk_multiblock_radix_size +
          threadIdx.x);
    }
  }
  uint32_t inclusive = 0;
  Scan(temp_storage.scan_storage).InclusiveSum(digit_count, inclusive);
  __syncthreads();
  if (threadIdx.x < topk_multiblock_radix_size) {
    temp_storage.digit_count_cumsum[threadIdx.x] = inclusive;
  }
  __syncthreads();
  if (threadIdx.x < topk_multiblock_radix_size) {
    const uint32_t left = threadIdx.x == 0
        ? 0 : temp_storage.digit_count_cumsum[threadIdx.x - 1];
    if (left < rank && rank <= inclusive) {
      const Key digit_mask = static_cast<Key>(
          topk_multiblock_radix_mask) << digit_pos;
      desired = (__ldg(desired_in + row) & ~digit_mask) |
          (static_cast<Key>(threadIdx.x) << digit_pos);
      if (block_in_row == 0) {
        desired_out[row] = desired;
        if (digit_pos > 0) {
          ranks_out[row] = rank - left;
        } else {
          kth_values[row] = Traits::deconvert(desired);
        }
      }
    }
  }
  __syncthreads();
  const Key selected = desired;
  const Key selected_digit = (selected >> digit_pos) &
      static_cast<Key>(topk_multiblock_radix_mask);
  const int warp = threadIdx.x / 32;
  const int warp_start = warp * 32;
  const int warp_end = warp_start + 31;
  const bool warp_active = largest ? warp_end > selected_digit
                                   : warp_start < selected_digit;
  const bool active = largest ? threadIdx.x > selected_digit
                              : threadIdx.x < selected_digit;
  uint32_t count = 0;
  if (warp_active) {
    if (active) {
      count = __ldg(counts + block_index * topk_multiblock_radix_size +
                    threadIdx.x);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      count += __shfl_down_sync(0xffffffffffffffffull, count, offset);
    }
  }
  __shared__ uint32_t warp_counts[topk_multiblock_threads / 32];
  if ((threadIdx.x & 31) == 0) warp_counts[warp] = count;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t total = 0;
    for (int w = 0; w < topk_multiblock_threads / 32; ++w) total += warp_counts[w];
    within_k_counts[block_index] += total;
  }
}

template <typename Key>
__global__ void topk_multiblock_kth_counts(
    const Key* __restrict__ desired, const uint16_t* __restrict__ counts,
    uint32_t* __restrict__ kth_counts, uint32_t num_blocks,
    uint32_t blocks_per_row) {
  const uint64_t grid_blocks = static_cast<uint64_t>(gridDim.x) * gridDim.y *
      gridDim.z;
  const uint64_t first = static_cast<uint64_t>(topk_linear_block_id<uint32_t>()) *
      blockDim.x + threadIdx.x;
  const uint64_t stride = grid_blocks * blockDim.x;
  for (uint64_t block_index = first; block_index < num_blocks;
       block_index += stride) {
    const uint32_t row = static_cast<uint32_t>(block_index / blocks_per_row);
    const int digit = static_cast<int>(desired[row] &
                                       static_cast<Key>(topk_multiblock_radix_mask));
    kth_counts[block_index] = __ldg(
        counts + block_index * topk_multiblock_radix_size + digit);
  }
}

template <typename Key>
struct topk_multiblock_block_to_row {
  uint32_t blocks_per_row;
  __host__ __device__ uint32_t operator()(uint32_t block) const {
    return block / blocks_per_row;
  }
};

template <typename T, typename Key, typename IndexType>
__launch_bounds__(topk_multiblock_threads)
__global__ void topk_multiblock_gather(
    const T* __restrict__ input, T* __restrict__ values,
    int64_t* __restrict__ indices, const T* __restrict__ kth_values,
    const uint32_t* __restrict__ within_k_counts,
    const uint32_t* __restrict__ kth_counts, uint32_t rows, uint32_t cols,
    IndexType inner, uint32_t blocks_per_row, int items_per_thread,
    IndexType k, bool largest) {
  using Traits = TopKRadixTraits<T>;
  using Scan = cub::BlockScan<uint32_t, topk_multiblock_threads>;
  __shared__ typename Scan::TempStorage scan_storage;
  const uint32_t block_index = topk_linear_block_id<uint32_t>();
  const uint32_t num_blocks = rows * blocks_per_row;
  if (block_index >= num_blocks) return;
  const uint32_t row = static_cast<uint32_t>(block_index / blocks_per_row);
  const uint32_t block_in_row = static_cast<uint32_t>(block_index % blocks_per_row);
  const IndexType row_index = static_cast<IndexType>(row);
  const IndexType outer_index = row_index / inner;
  const IndexType inner_index = row_index % inner;
  const IndexType input_base = outer_index * static_cast<IndexType>(cols) * inner +
      inner_index;
  const IndexType output_base = outer_index * static_cast<IndexType>(k) * inner +
      inner_index;
  const int items_per_block = items_per_thread * topk_multiblock_threads;
  const IndexType block_base = static_cast<IndexType>(block_in_row) *
      static_cast<IndexType>(items_per_block);
  const int items = block_in_row + 1 < blocks_per_row
      ? items_per_thread
      : static_cast<int>((static_cast<uint64_t>(cols) - block_base +
                          topk_multiblock_threads - 1) /
                         topk_multiblock_threads);
  const Key kth_key = Traits::encode(topk_load(kth_values + row));
  uint32_t start_within = block_in_row == 0
      ? 0 : __ldg(within_k_counts + block_index - 1);
  const uint32_t total_within =
      __ldg(within_k_counts + row * blocks_per_row + blocks_per_row - 1);
  uint32_t start_kth = total_within + (block_in_row == 0
      ? 0 : __ldg(kth_counts + block_index - 1));
  for (int item = 0; item < items; ++item) {
    const IndexType column = block_base +
        static_cast<IndexType>(item) * topk_multiblock_threads + threadIdx.x;
    T value = static_cast<T>(0);
    uint32_t within = 0;
    uint32_t kth = 0;
    if (column < static_cast<IndexType>(cols)) {
      value = topk_load(&input[input_base + column * inner]);
      const Key key = Traits::encode(value);
      within = largest ? key > kth_key : key < kth_key;
      kth = key == kth_key;
    }
    uint32_t within_index = 0;
    uint32_t num_within = 0;
    Scan(scan_storage).ExclusiveSum(within, within_index, num_within);
    __syncthreads();
    if (within) {
      const uint32_t offset = start_within + within_index;
      values[output_base + static_cast<IndexType>(offset) * inner] = value;
      indices[output_base + static_cast<IndexType>(offset) * inner] = column;
    }
    start_within += num_within;
    if (start_kth < static_cast<uint32_t>(k)) {
      uint32_t kth_index = 0;
      uint32_t num_kth = 0;
      Scan(scan_storage).ExclusiveSum(kth, kth_index, num_kth);
      __syncthreads();
      if (kth) {
        const uint32_t offset = start_kth + kth_index;
        if (offset < static_cast<uint32_t>(k)) {
          values[output_base + static_cast<IndexType>(offset) * inner] = value;
          indices[output_base + static_cast<IndexType>(offset) * inner] = column;
        }
      }
      start_kth += num_kth;
    }
  }
}

bool topk_should_use_multiblock(int64_t rows, int64_t cols) {
  if (rows > std::numeric_limits<uint32_t>::max() ||
      cols > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  return (rows <= 20 && cols >= 20000) ||
      (rows > 20 && rows <= 40 && cols >= 10000) ||
      (rows > 40 && rows <= 80 && cols >= 8000) ||
      (rows > 80 && rows < 200 && cols >= 5000) ||
      (rows >= 200 && rows < 800 && cols >= 3000) ||
      (rows >= 800 && rows <= 4000 && cols >= 800) ||
      (rows > 4000 && cols >= 400);
}

int topk_multiblock_items_per_thread(uint32_t rows, uint32_t cols) {
  const auto& properties = topk_device_properties();
  constexpr int registers_per_thread = 40;
  constexpr int registers_per_block =
      registers_per_thread * topk_multiblock_threads;
  const int blocks_per_mp = std::min(
      properties.regsPerMultiprocessor / registers_per_block,
      properties.maxBlocksPerMultiProcessor);
  const uint64_t denominator = static_cast<uint64_t>(
      properties.multiProcessorCount) * std::max(blocks_per_mp, 1) *
      topk_multiblock_threads;
  const uint64_t total = static_cast<uint64_t>(rows) * cols;
  const uint64_t rounded = (total + denominator - 1) / denominator;
  return static_cast<int>(std::max<uint64_t>(
      topk_multiblock_min_items_per_thread,
      std::min<uint64_t>(rounded, topk_multiblock_max_items_per_thread)));
}

template <typename T>
void launch_sorted_topk(Tensor& values, Tensor& indices, int64_t rows,
                        int64_t k, int64_t inner, bool largest);

template <typename T, typename IndexType>
void launch_multiblock_topk_impl(const Tensor& input, Tensor& values,
                                 Tensor& indices, int64_t rows, int64_t cols,
                                 int64_t k, int64_t inner, bool largest,
                                 bool sorted) {
  using Key = typename TopKRadixTraits<T>::key_type;
  using Traits = TopKRadixTraits<T>;
  TP_CHECK(rows <= std::numeric_limits<uint32_t>::max() &&
               cols <= std::numeric_limits<uint32_t>::max(),
           "topk: multi-block dimensions exceed uint32 range");
  const uint32_t row_count = static_cast<uint32_t>(rows);
  const uint32_t column_count = static_cast<uint32_t>(cols);
  const int items_per_thread = topk_multiblock_items_per_thread(
      row_count, column_count);
  const int items_per_block = items_per_thread * topk_multiblock_threads;
  const uint32_t blocks_per_row = static_cast<uint32_t>(
      (column_count + items_per_block - 1) / items_per_block);
  const uint64_t block_count64 = static_cast<uint64_t>(row_count) * blocks_per_row;
  TP_CHECK(block_count64 <= std::numeric_limits<uint32_t>::max(),
           "topk: too many multi-block tiles");
  const uint32_t block_count = static_cast<uint32_t>(block_count64);
  const auto& properties = topk_device_properties();
  const auto grid_for = [&properties](uint64_t count) {
    const uint64_t max_x = static_cast<uint64_t>(properties.maxGridSize[0]);
    const uint64_t max_y = static_cast<uint64_t>(properties.maxGridSize[1]);
    const uint64_t x = std::max<uint64_t>(1, std::min(count, max_x));
    const uint64_t y = std::max<uint64_t>(1, (count + x - 1) / x);
    TP_CHECK(y <= max_y, "topk: grid exceeds device limits");
    return dim3(static_cast<unsigned>(x), static_cast<unsigned>(y), 1);
  };
  Tensor desired = Tensor::empty(
      {static_cast<int64_t>(row_count) * 2},
      sizeof(Key) == sizeof(uint32_t) ? DType::UInt32 : DType::UInt64,
      input.device());
  Tensor ranks = Tensor::empty(
      {static_cast<int64_t>(row_count) * 2}, DType::UInt32, input.device());
  Tensor counts = Tensor::empty(
      {static_cast<int64_t>(block_count) * topk_multiblock_radix_size},
      DType::UInt16, input.device());
  Tensor kth_values = Tensor::empty(
      {static_cast<int64_t>(row_count)}, input.dtype(), input.device());
  Tensor within_k_counts = Tensor::empty(
      {static_cast<int64_t>(block_count)}, DType::UInt32, input.device());
  Tensor kth_counts = Tensor::empty(
      {static_cast<int64_t>(block_count)}, DType::UInt32, input.device());
  TP_CUDA_CHECK(cudaMemsetAsync(
      within_k_counts.data_ptr<uint32_t>(), 0,
      static_cast<size_t>(block_count) * sizeof(uint32_t),
      getCurrentCUDAStream().stream()));
  const dim3 fill_grid = grid_for((static_cast<uint64_t>(row_count) + 511) / 512);
  topk_multiblock_fill_kernel<uint32_t>
      <<<fill_grid, 512, 0,
         getCurrentCUDAStream().stream()>>>(
          ranks.data_ptr<uint32_t>(),
          largest ? column_count - static_cast<uint32_t>(k) + 1
                  : static_cast<uint32_t>(k),
          row_count);
  TP_CUDA_CHECK(cudaGetLastError());
  Key* desired_in = desired.data_ptr<Key>();
  Key* desired_out = desired_in + row_count;
  uint32_t* ranks_in = ranks.data_ptr<uint32_t>();
  uint32_t* ranks_out = ranks_in + row_count;
  Key desired_mask = 0;
  const dim3 block_grid = grid_for(block_count);
  for (int digit_pos = Traits::bit_count - topk_multiblock_radix_bits;
       digit_pos >= 0; digit_pos -= topk_multiblock_radix_bits) {
    topk_multiblock_digit_counts<T, Key, IndexType>
        <<<block_grid, topk_multiblock_threads, 0,
           getCurrentCUDAStream().stream()>>>(
            input.data_ptr<T>(), ranks_in, desired_in, desired_mask,
            counts.data_ptr<uint16_t>(), row_count, column_count,
            static_cast<IndexType>(inner),
            blocks_per_row, items_per_thread, digit_pos);
    TP_CUDA_CHECK(cudaGetLastError());
    topk_multiblock_within_k_counts<T, Key>
        <<<block_grid, topk_multiblock_radix_size, 0,
           getCurrentCUDAStream().stream()>>>(
            counts.data_ptr<uint16_t>(), desired_in, ranks_in, desired_out,
            ranks_out, row_count,
            blocks_per_row, digit_pos, largest,
            within_k_counts.data_ptr<uint32_t>(), kth_values.data_ptr<T>());
    TP_CUDA_CHECK(cudaGetLastError());
    std::swap(desired_in, desired_out);
    std::swap(ranks_in, ranks_out);
    desired_mask |= static_cast<Key>(topk_multiblock_radix_mask) << digit_pos;
  }
  const dim3 kth_grid = grid_for((static_cast<uint64_t>(row_count) + 255) / 256);
  topk_multiblock_kth_counts<Key>
      <<<kth_grid, topk_multiblock_threads, 0,
         getCurrentCUDAStream().stream()>>>(
          desired_in, counts.data_ptr<uint16_t>(), kth_counts.data_ptr<uint32_t>(),
          block_count, blocks_per_row);
  TP_CUDA_CHECK(cudaGetLastError());
  TP_CHECK(block_count <= static_cast<uint32_t>(std::numeric_limits<int>::max()),
           "topk: scan range exceeds device scan limit");
  using Counter = cub::CountingInputIterator<uint32_t, uint32_t>;
  using KeyIterator = cub::TransformInputIterator<
      uint32_t, topk_multiblock_block_to_row<Key>, Counter>;
  KeyIterator key_iterator(
      Counter(0), topk_multiblock_block_to_row<Key>{blocks_per_row});
  size_t scan_bytes = 0;
  TP_CUDA_CHECK(cub::DeviceScan::InclusiveSumByKey(
      nullptr, scan_bytes, key_iterator, within_k_counts.data_ptr<uint32_t>(),
      within_k_counts.data_ptr<uint32_t>(), static_cast<int>(block_count),
      cub::Equality(), getCurrentCUDAStream().stream()));
  Tensor scan_storage = Tensor::empty(
      {static_cast<int64_t>(std::max<size_t>(scan_bytes, 1))}, DType::UInt8,
      input.device());
  TP_CUDA_CHECK(cub::DeviceScan::InclusiveSumByKey(
      scan_storage.data_ptr(), scan_bytes, key_iterator,
      within_k_counts.data_ptr<uint32_t>(), within_k_counts.data_ptr<uint32_t>(),
      static_cast<int>(block_count), cub::Equality(),
      getCurrentCUDAStream().stream()));
  TP_CUDA_CHECK(cub::DeviceScan::InclusiveSumByKey(
      scan_storage.data_ptr(), scan_bytes, key_iterator,
      kth_counts.data_ptr<uint32_t>(), kth_counts.data_ptr<uint32_t>(),
      static_cast<int>(block_count), cub::Equality(),
      getCurrentCUDAStream().stream()));
  topk_multiblock_gather<T, Key, IndexType>
      <<<block_grid, topk_multiblock_threads, 0,
         getCurrentCUDAStream().stream()>>>(
          input.data_ptr<T>(), values.data_ptr<T>(), indices.data_ptr<int64_t>(),
          kth_values.data_ptr<T>(), within_k_counts.data_ptr<uint32_t>(),
          kth_counts.data_ptr<uint32_t>(), row_count, column_count,
          static_cast<IndexType>(inner),
          blocks_per_row, items_per_thread, static_cast<IndexType>(k), largest);
  TP_CUDA_CHECK(cudaGetLastError());
  if (sorted && k > 1) {
    launch_sorted_topk<T>(values, indices, rows, k, inner, largest);
  }
}

template <typename T>
void launch_multiblock_topk(const Tensor& input, Tensor& values,
                            Tensor& indices, int64_t rows, int64_t cols,
                            int64_t k, int64_t inner, bool largest,
                            bool sorted) {
  const int64_t max_index =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  if (input.numel() <= max_index && values.numel() <= max_index &&
      indices.numel() <= max_index) {
    launch_multiblock_topk_impl<T, uint32_t>(
        input, values, indices, rows, cols, k, inner, largest, sorted);
  } else {
    launch_multiblock_topk_impl<T, uint64_t>(
        input, values, indices, rows, cols, k, inner, largest, sorted);
  }
}

template <typename T>
void launch_segmented_sorted_topk(Tensor& values, Tensor& indices, int64_t rows,
                                  int64_t k, int64_t inner, bool largest) {
  using Key = typename TopKRadixTraits<T>::key_type;
  using Traits = TopKRadixTraits<T>;
  const int64_t item_count = rows * k;
  TP_CHECK(item_count <= std::numeric_limits<int>::max(),
           "topk: sorted output exceeds the segmented sort limit");
  TP_CHECK(rows <= std::numeric_limits<uint32_t>::max() &&
               k <= std::numeric_limits<uint32_t>::max(),
           "topk: sorted output dimensions exceed the segmented sort limit");
  const auto stream = getCurrentCUDAStream().stream();
  const DType key_dtype = sizeof(Key) == sizeof(uint32_t)
      ? DType::UInt32 : DType::UInt64;
  Tensor key_input = Tensor::empty({item_count}, key_dtype, values.device());
  Tensor key_alternate = Tensor::empty({item_count}, key_dtype, values.device());
  Tensor position_input = Tensor::empty({item_count}, DType::Int64, values.device());
  Tensor position_alternate = Tensor::empty({item_count}, DType::Int64, values.device());
  Tensor offsets = Tensor::empty({rows + 1}, DType::UInt32, values.device());
  const int threads = 256;
  const int item_blocks = static_cast<int>((item_count + threads - 1) / threads);
  topk_pack_sort_kernel<T, Key><<<topk_grid_for(item_blocks), threads, 0, stream>>>(
      values.data_ptr<T>(), key_input.data_ptr<Key>(), position_input.data_ptr<int64_t>(),
      rows, k, inner);
  TP_CUDA_CHECK(cudaGetLastError());
  const int offset_blocks = static_cast<int>((rows + 1 + threads - 1) / threads);
  topk_fill_segment_offsets<<<topk_grid_for(offset_blocks), threads, 0, stream>>>(
      offsets.data_ptr<uint32_t>(), static_cast<uint32_t>(rows),
      static_cast<uint32_t>(k));
  TP_CUDA_CHECK(cudaGetLastError());

  cub::DoubleBuffer<Key> key_buffer(
      key_input.data_ptr<Key>(), key_alternate.data_ptr<Key>());
  cub::DoubleBuffer<int64_t> position_buffer(
      position_input.data_ptr<int64_t>(), position_alternate.data_ptr<int64_t>());
  const int item_count_int = static_cast<int>(item_count);
  const int row_count_int = static_cast<int>(rows);
  uint32_t* begin_offsets = offsets.data_ptr<uint32_t>();
  uint32_t* end_offsets = begin_offsets + 1;
  size_t temp_bytes = 0;
  cudaError_t status;
  if (largest) {
    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_bytes, key_buffer, position_buffer, item_count_int,
        row_count_int, begin_offsets, end_offsets, 0, Traits::bit_count,
        stream);
  } else {
    status = cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes, key_buffer, position_buffer, item_count_int,
        row_count_int, begin_offsets, end_offsets, 0, Traits::bit_count,
        stream);
  }
  TP_CUDA_CHECK(status);
  Tensor temp_storage = Tensor::empty(
      {static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))}, DType::UInt8,
      values.device());
  if (largest) {
    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.data_ptr(), temp_bytes, key_buffer, position_buffer,
        item_count_int, row_count_int, begin_offsets, end_offsets, 0,
        Traits::bit_count, stream);
  } else {
    status = cub::DeviceSegmentedRadixSort::SortPairs(
        temp_storage.data_ptr(), temp_bytes, key_buffer, position_buffer,
        item_count_int, row_count_int, begin_offsets, end_offsets, 0,
        Traits::bit_count, stream);
  }
  TP_CUDA_CHECK(status);

  Tensor sorted_values = Tensor::empty(
      static_cast<std::vector<int64_t>>(values.shape()), values.dtype(),
      values.device());
  Tensor sorted_indices = Tensor::empty(
      static_cast<std::vector<int64_t>>(indices.shape()), indices.dtype(),
      indices.device());
  const int output_blocks = item_blocks;
  topk_unpack_sort_kernel<T><<<topk_grid_for(output_blocks), threads, 0, stream>>>(
      values.data_ptr<T>(), indices.data_ptr<int64_t>(),
      position_buffer.Current(), sorted_values.data_ptr<T>(),
      sorted_indices.data_ptr<int64_t>(), rows, k, inner);
  TP_CUDA_CHECK(cudaGetLastError());
  values.copy_(sorted_values);
  indices.copy_(sorted_indices);
}

template <typename T, typename Key, int BlockThreads, int ItemsPerThread>
void launch_selected_radix_sort(Tensor& values, Tensor& indices, int64_t rows,
                                int64_t k, int64_t inner, bool largest) {
  const dim3 grid = topk_grid_for(static_cast<uint64_t>(rows));
  radix_sort_selected_kernel<T, Key, BlockThreads, ItemsPerThread>
      <<<grid, BlockThreads, 0,
         getCurrentCUDAStream().stream()>>>(
          values.data_ptr<T>(), indices.data_ptr<int64_t>(), rows, k, inner,
          largest);
  TP_CUDA_CHECK(cudaGetLastError());
}

template <typename T, int SortSize>
void launch_warp_merge_sort(Tensor& values, Tensor& indices, int64_t rows,
                            int64_t k, int64_t inner, bool largest) {
  constexpr int max_block_y = 16;
  int min_grid_size = 0;
  int suggested_block_size = 0;
  TP_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &suggested_block_size,
      warp_merge_sort_selected_kernel<T, SortSize, max_block_y>, 0,
      32 * max_block_y));
  (void)suggested_block_size;
  const int64_t occupancy_grid = std::max(min_grid_size, 1);
  const int64_t max_batch = std::max<int64_t>(1, rows / occupancy_grid);
  const int block_y = static_cast<int>(std::min<int64_t>(max_block_y, max_batch));
  const int64_t grid_count = (rows + block_y - 1) / block_y;
  const dim3 grid = topk_grid_for(static_cast<uint64_t>(grid_count));
  warp_merge_sort_selected_kernel<T, SortSize, max_block_y>
      <<<grid, dim3(32, block_y), 0,
         getCurrentCUDAStream().stream()>>>(
          values.data_ptr<T>(), indices.data_ptr<int64_t>(), rows, k, inner,
          largest);
  TP_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_sorted_topk(Tensor& values, Tensor& indices, int64_t rows,
                        int64_t k, int64_t inner, bool largest) {
  using Key = typename TopKRadixTraits<T>::key_type;
  int64_t padded = 1;
  while (padded < k) padded <<= 1;
  if (padded > 4096) {
    launch_segmented_sorted_topk<T>(values, indices, rows, k, inner, largest);
    return;
  }
  if (padded <= 32) {
    constexpr int max_block_y = 16;
    int min_grid_size = 0;
    int suggested_block_size = 0;
    TP_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &suggested_block_size,
        bitonic_sort_selected_kernel<T, max_block_y>, 0, 16 * max_block_y));
    (void)suggested_block_size;
    const int64_t occupancy_grid = std::max(min_grid_size, 1);
    const int64_t max_batch = std::max<int64_t>(1, rows / occupancy_grid);
    const int block_y = static_cast<int>(std::min<int64_t>(
        max_block_y, max_batch));
    const int64_t grid_count = (rows + block_y - 1) / block_y;
    const dim3 grid = topk_grid_for(static_cast<uint64_t>(grid_count));
    bitonic_sort_selected_kernel<T, max_block_y>
        <<<grid, dim3(16, block_y), 0,
           getCurrentCUDAStream().stream()>>>(
            values.data_ptr<T>(), indices.data_ptr<int64_t>(), rows, k, inner,
            largest);
    TP_CUDA_CHECK(cudaGetLastError());
  } else if (padded <= 128) {
    launch_warp_merge_sort<T, 128>(values, indices, rows, k, inner, largest);
  } else if (padded <= 1024) {
    launch_selected_radix_sort<T, Key, 32, 32>(values, indices, rows, k, inner,
                                                largest);
  } else if (padded <= 2048) {
    launch_selected_radix_sort<T, Key, 64, 32>(values, indices, rows, k, inner,
                                                largest);
  } else if (padded <= 4096) {
    launch_selected_radix_sort<T, Key, 128, 32>(values, indices, rows, k, inner,
                                                 largest);
  }
}

template <typename T>
void launch_topk_cuda(const Tensor& input, Tensor& values, Tensor& indices,
                      int64_t rows, int64_t cols, int64_t k, int64_t inner,
                      bool largest, bool sorted, int64_t impl) {
  if (impl != 0 && impl != 1) {
    TP_THROW(RuntimeError, "topk: unknown impl " + std::to_string(impl));
  }

  if (sorted && cols > 128 && cols <= 1024 && rows <= 2048) {
    using Key = typename TopKRadixTraits<T>::key_type;
    const dim3 grid = topk_grid_for(static_cast<uint64_t>(rows));
    if (k <= 128) {
      radix_sort_all_topk_indices_kernel<T, Key, 32, 32>
          <<<grid, 32, 0, getCurrentCUDAStream().stream()>>>(
              input.data_ptr<T>(), values.data_ptr<T>(),
              indices.data_ptr<int64_t>(), rows, cols, k, inner, largest);
    } else {
      radix_sort_all_topk_kernel<T, Key, 32, 32>
          <<<grid, 32, 0, getCurrentCUDAStream().stream()>>>(
              input.data_ptr<T>(), values.data_ptr<T>(),
              indices.data_ptr<int64_t>(), rows, cols, k, inner, largest);
    }
    TP_CUDA_CHECK(cudaGetLastError());
    return;
  }

  const bool use_multiblock = topk_should_use_multiblock(rows, cols) &&
      !(rows <= 512 && cols <= 4096);
  if (use_multiblock) {
    launch_multiblock_topk<T>(input, values, indices, rows, cols, k, inner,
                              largest, sorted);
    return;
  }

  int threads = static_cast<int>(std::min<int64_t>(
      ((cols + 31) / 32) * 32, static_cast<int64_t>(1024)));
  if (threads < 32) threads = 32;
  const int64_t max_index = static_cast<int64_t>(
      std::numeric_limits<uint32_t>::max() - 1024u);
  const bool use_32bit_index = input.numel() <= max_index &&
      values.numel() <= max_index && indices.numel() <= max_index;
  const dim3 grid = topk_grid_for(static_cast<uint64_t>(rows));
  if (use_32bit_index) {
    radix_topk_kernel<T, uint32_t>
        <<<grid, threads, 0,
           getCurrentCUDAStream().stream()>>>(
            input.data_ptr<T>(), values.data_ptr<T>(),
            indices.data_ptr<int64_t>(), static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols), static_cast<uint32_t>(k),
            static_cast<uint32_t>(inner), largest);
  } else {
    radix_topk_kernel<T, uint64_t>
        <<<grid, threads, 0,
           getCurrentCUDAStream().stream()>>>(
            input.data_ptr<T>(), values.data_ptr<T>(),
            indices.data_ptr<int64_t>(), static_cast<uint64_t>(rows),
            static_cast<uint64_t>(cols), static_cast<uint64_t>(k),
            static_cast<uint64_t>(inner), largest);
  }
  TP_CUDA_CHECK(cudaGetLastError());
  if (sorted && k > 1) {
    launch_sorted_topk<T>(values, indices, rows, k, inner, largest);
  }
}

} // namespace

std::tuple<Tensor, Tensor> topk_kernel_cuda(const Tensor& self, int64_t k, int64_t dim,
                                             bool largest, bool sorted, int64_t impl) {
  Tensor input = self.contiguous();
  const int64_t ndim = input.dim();
  if (ndim == 0) {
    TP_CHECK(dim == 0 || dim == -1, "topk: dimension out of range");
  } else {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "topk: dimension out of range");
  }
  const int64_t dim_size = ndim == 0 ? 1 : input.size(dim);
  if (k < 0 || k > dim_size) {
    TP_THROW(RuntimeError, "topk: k must be in [0, dimension size]");
  }

  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
  if (ndim != 0) shape[static_cast<size_t>(dim)] = k;
  Tensor values = Tensor::empty(shape, input.dtype(), input.device());
  Tensor indices = Tensor::empty(shape, DType::Int64, input.device());
  if (k == 0 || input.numel() == 0) return {values, indices};

  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t axis = 0; axis < (ndim == 0 ? 0 : dim); ++axis) outer *= input.size(axis);
  for (int64_t axis = ndim == 0 ? 0 : dim + 1; axis < ndim; ++axis) inner *= input.size(axis);
  const int64_t rows = outer * inner;

  switch (input.dtype()) {
    case DType::UInt8:
      launch_topk_cuda<uint8_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Int8:
      launch_topk_cuda<int8_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Int16:
      launch_topk_cuda<int16_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Int32:
      launch_topk_cuda<int32_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Int64:
      launch_topk_cuda<int64_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::UInt16:
      launch_topk_cuda<uint16_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::UInt32:
      launch_topk_cuda<uint32_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::UInt64:
      launch_topk_cuda<uint64_t>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Float16:
      launch_topk_cuda<Half>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::BFloat16:
      launch_topk_cuda<BFloat16>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Float32:
      launch_topk_cuda<float>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Float64:
      launch_topk_cuda<double>(input, values, indices, rows, dim_size, k, inner, largest, sorted, impl);
      break;
    case DType::Bool:
    case DType::ComplexFloat:
    case DType::ComplexDouble:
    case DType::ComplexHalf:
    case DType::BComplex32:
    case DType::Float8_e4m3fn:
    case DType::Float8_e5m2:
    case DType::Float8_e4m3fnuz:
    case DType::Float8_e5m2fnuz:
    case DType::Float8_e8m0fnu:
    case DType::Undefined:
    case DType::NumOptions:
      TP_THROW(NotImplementedError, "topk: unsupported dtype");
  }
  return {values, indices};
}

std::tuple<Tensor, Tensor> interop_topk_values_cuda(const Tensor& self, int64_t k, int64_t dim,
                                                    bool largest, bool sorted,
                                                    Tensor& values, Tensor& indices) {
  std::tie(values, indices) = topk_kernel_cuda(self, k, dim, largest, sorted, 0);
  return {values, indices};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, TopKKernels) {
  m.impl("topk", topk_kernel_cuda);
  // out-variant: run the value kernel, then transfer into the caller's
  // values/indices buffers.
  m.impl("topk.values", interop_topk_values_cuda);
}

} // namespace cuda
} // namespace tensorplay
