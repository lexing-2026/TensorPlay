#include "Tensor.h"
#include "SparseKernels.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include "Atomic.cuh"
#include "GPUPrimitives.cuh"
#include <thrust/iterator/counting_iterator.h>

namespace tensorplay {
namespace cuda {
namespace {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

constexpr int kThreads = 256;
constexpr int kSmallEmbeddingDim = 32;
constexpr int kEmbeddingWarpSize = 32;

inline int block_size_for(int64_t work_items) {
  if (work_items <= 32) return 32;
  if (work_items <= 64) return 64;
  if (work_items <= 128) return 128;
  return kThreads;
}

template <typename IndexT>
__global__ void embedding_validate_indices_kernel(
    int64_t n_indices,
    int64_t num_weights,
    const IndexT* __restrict__ indices) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_indices) return;
  const int64_t index = static_cast<int64_t>(indices[i]);
  assert(index >= 0 && index < num_weights);
}

// One CUDA thread owns one lookup. This is the lowest-overhead path for the
// small rows common in categorical features and recommendation models.
template <typename T, typename IndexT>
__global__ void embedding_forward_flat_kernel(
    int64_t n_indices,
    int64_t embedding_dim,
    int64_t num_weights,
    const T* __restrict__ weight,
    const IndexT* __restrict__ indices,
    T* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_indices) return;

  const int64_t index = static_cast<int64_t>(indices[row]);
  assert(index >= 0 && index < num_weights);

  const T* src = weight + index * embedding_dim;
  T* dst = output + row * embedding_dim;
  for (int64_t col = 0; col < embedding_dim; ++col) {
    dst[col] = src[col];
  }
}

template <typename IndexT>
__global__ void embedding_forward_float4_flat_kernel(
    int64_t n_indices,
    int64_t n_vectors,
    int64_t num_weights,
    const float4* __restrict__ weight,
    const IndexT* __restrict__ indices,
    float4* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_indices) return;

  const int64_t index = static_cast<int64_t>(indices[row]);
  assert(index >= 0 && index < num_weights);

  const float4* src = weight + index * n_vectors;
  float4* dst = output + row * n_vectors;
  for (int64_t col = 0; col < n_vectors; ++col) {
    dst[col] = src[col];
  }
}

template <typename IndexT>
__global__ void embedding_forward_half2_flat_kernel(
    int64_t n_indices,
    int64_t n_vectors,
    int64_t num_weights,
    const __half2* __restrict__ weight,
    const IndexT* __restrict__ indices,
    __half2* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_indices) return;

  const int64_t index = static_cast<int64_t>(indices[row]);
  assert(index >= 0 && index < num_weights);

  const __half2* src = weight + index * n_vectors;
  __half2* dst = output + row * n_vectors;
  for (int64_t col = 0; col < n_vectors; ++col) {
    dst[col] = src[col];
  }
}

// For wider rows, one block owns a lookup and all threads copy that row. The
// index is loaded once into shared memory, avoiding both repeated index loads
// and the integer divide required by a flat output-element kernel.
template <typename T, typename IndexT>
__global__ void embedding_forward_row_kernel(
    int64_t n_indices,
    int64_t embedding_dim,
    int64_t num_weights,
    const T* __restrict__ weight,
    const IndexT* __restrict__ indices,
    T* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= n_indices) return;

  __shared__ int64_t index;
  if (threadIdx.x == 0) {
    index = static_cast<int64_t>(indices[row]);
    assert(index >= 0 && index < num_weights);
  }
  __syncthreads();

  const T* src = weight + index * embedding_dim;
  T* dst = output + row * embedding_dim;
  for (int64_t col = threadIdx.x; col < embedding_dim; col += blockDim.x) {
    dst[col] = src[col];
  }
}

template <typename IndexT>
__global__ void embedding_forward_float4_row_kernel(
    int64_t n_indices,
    int64_t n_vectors,
    int64_t num_weights,
    const float4* __restrict__ weight,
    const IndexT* __restrict__ indices,
    float4* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= n_indices) return;

  __shared__ int64_t index;
  if (threadIdx.x == 0) {
    index = static_cast<int64_t>(indices[row]);
    assert(index >= 0 && index < num_weights);
  }
  __syncthreads();

  const float4* src = weight + index * n_vectors;
  float4* dst = output + row * n_vectors;
  for (int64_t col = threadIdx.x; col < n_vectors; col += blockDim.x) {
    dst[col] = src[col];
  }
}

template <typename IndexT>
__global__ void embedding_forward_half2_row_kernel(
    int64_t n_indices,
    int64_t n_vectors,
    int64_t num_weights,
    const __half2* __restrict__ weight,
    const IndexT* __restrict__ indices,
    __half2* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= n_indices) return;

  __shared__ int64_t index;
  if (threadIdx.x == 0) {
    index = static_cast<int64_t>(indices[row]);
    assert(index >= 0 && index < num_weights);
  }
  __syncthreads();

  const __half2* src = weight + index * n_vectors;
  __half2* dst = output + row * n_vectors;
  for (int64_t col = threadIdx.x; col < n_vectors; col += blockDim.x) {
    dst[col] = src[col];
  }
}

template <typename T, typename IndexT>
void launch_embedding_forward_dtype(
    int64_t n_indices,
    int64_t embedding_dim,
    int64_t num_weights,
    const T* weight,
    const IndexT* indices,
    T* output,
    cudaStream_t stream) {
  const dim3 flat_block(kThreads);

  if (embedding_dim <= kSmallEmbeddingDim) {
    const dim3 grid(static_cast<unsigned int>((n_indices + kThreads - 1) / kThreads));

    if constexpr (std::is_same_v<T, float>) {
      if (embedding_dim % 4 == 0 &&
          reinterpret_cast<uintptr_t>(weight) % alignof(float4) == 0 &&
          reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0) {
        embedding_forward_float4_flat_kernel<<<grid, flat_block, 0, stream>>>(
            n_indices, embedding_dim / 4, num_weights,
            reinterpret_cast<const float4*>(weight), indices,
            reinterpret_cast<float4*>(output));
        return;
      }
    }

    if constexpr (std::is_same_v<T, tensorplay::Half>) {
      if (embedding_dim % 2 == 0 &&
          reinterpret_cast<uintptr_t>(weight) % alignof(__half2) == 0 &&
          reinterpret_cast<uintptr_t>(output) % alignof(__half2) == 0) {
        embedding_forward_half2_flat_kernel<<<grid, flat_block, 0, stream>>>(
            n_indices, embedding_dim / 2, num_weights,
            reinterpret_cast<const __half2*>(weight), indices,
            reinterpret_cast<__half2*>(output));
        return;
      }
    }

    embedding_forward_flat_kernel<<<grid, flat_block, 0, stream>>>(
        n_indices, embedding_dim, num_weights, weight, indices, output);
    return;
  }

  int vector_work_items = embedding_dim;
  if constexpr (std::is_same_v<T, float>) {
    if (embedding_dim % 4 == 0 &&
        reinterpret_cast<uintptr_t>(weight) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0) {
      vector_work_items = embedding_dim / 4;
    }
  }
  if constexpr (std::is_same_v<T, tensorplay::Half>) {
    if (embedding_dim % 2 == 0 &&
        reinterpret_cast<uintptr_t>(weight) % alignof(__half2) == 0 &&
        reinterpret_cast<uintptr_t>(output) % alignof(__half2) == 0) {
      vector_work_items = embedding_dim / 2;
    }
  }

  const int threads = block_size_for(vector_work_items);
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(n_indices));

  if constexpr (std::is_same_v<T, float>) {
    if (embedding_dim % 4 == 0 &&
        reinterpret_cast<uintptr_t>(weight) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(output) % alignof(float4) == 0) {
      embedding_forward_float4_row_kernel<<<grid, block, 0, stream>>>(
          n_indices, embedding_dim / 4, num_weights,
          reinterpret_cast<const float4*>(weight), indices,
          reinterpret_cast<float4*>(output));
      return;
    }
  }

  if constexpr (std::is_same_v<T, tensorplay::Half>) {
    if (embedding_dim % 2 == 0 &&
        reinterpret_cast<uintptr_t>(weight) % alignof(__half2) == 0 &&
        reinterpret_cast<uintptr_t>(output) % alignof(__half2) == 0) {
      embedding_forward_half2_row_kernel<<<grid, block, 0, stream>>>(
          n_indices, embedding_dim / 2, num_weights,
          reinterpret_cast<const __half2*>(weight), indices,
          reinterpret_cast<__half2*>(output));
      return;
    }
  }

  embedding_forward_row_kernel<<<grid, block, 0, stream>>>(
      n_indices, embedding_dim, num_weights, weight, indices, output);
}

template <typename IndexT>
void launch_embedding_validate(
    int64_t n_indices,
    int64_t num_weights,
    const IndexT* indices,
    cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid(static_cast<unsigned int>((n_indices + kThreads - 1) / kThreads));
  embedding_validate_indices_kernel<<<grid, block, 0, stream>>>(n_indices, num_weights, indices);
}

template <typename IndexT>
void launch_embedding_forward(
    int64_t n_indices,
    int64_t embedding_dim,
    int64_t num_weights,
    const Tensor& weight,
    const Tensor& indices,
    Tensor& output,
    cudaStream_t stream) {
  const IndexT* index_data = indices.data_ptr<IndexT>();
  if (embedding_dim == 0) {
    launch_embedding_validate(n_indices, num_weights, index_data, stream);
    return;
  }

  switch (weight.dtype()) {
#define TP_EMBEDDING_FORWARD_CASE(ctype, dtype_name) \
    case DType::dtype_name: \
      launch_embedding_forward_dtype<ctype>( \
          n_indices, embedding_dim, num_weights, \
          weight.data_ptr<ctype>(), index_data, output.data_ptr<ctype>(), stream); \
      break;
    TENSORPLAY_FORALL_SCALAR_TYPES(TP_EMBEDDING_FORWARD_CASE)
#undef TP_EMBEDDING_FORWARD_CASE
    default:
      TP_THROW(NotImplementedError, "embedding_cuda: unsupported dtype");
  }
}

template <typename T, typename Accum, typename IndexT>
__global__ void embedding_backward_flat_kernel(
    int64_t n_indices,
    int64_t row_size,
    int64_t num_weights,
    int64_t padding_idx,
    const T* __restrict__ grad_output,
    const IndexT* __restrict__ indices,
    const int64_t* __restrict__ counts,
    Accum* __restrict__ grad_weight) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_indices * row_size;
  if (linear >= total) return;

  const int64_t index_pos = linear / row_size;
  const int64_t column = linear - index_pos * row_size;
  const int64_t index = static_cast<int64_t>(indices[index_pos]);
  assert(index >= 0 && index < num_weights);
  if (index == padding_idx) return;

  Accum value = static_cast<Accum>(grad_output[linear]);
  if (counts != nullptr) {
    value /= static_cast<Accum>(counts[index]);
  }
  atomicAdd(grad_weight + index * row_size + column, value);
}

template <typename T, typename Accum, typename IndexT>
__global__ void embedding_backward_row_kernel(
    int64_t n_indices,
    int64_t row_size,
    int64_t num_weights,
    int64_t padding_idx,
    const T* __restrict__ grad_output,
    const IndexT* __restrict__ indices,
    const int64_t* __restrict__ counts,
    Accum* __restrict__ grad_weight) {
  const int64_t index_pos = static_cast<int64_t>(blockIdx.x);
  if (index_pos >= n_indices) return;

  __shared__ int64_t index;
  if (threadIdx.x == 0) {
    index = static_cast<int64_t>(indices[index_pos]);
    assert(index >= 0 && index < num_weights);
  }
  __syncthreads();
  if (index == padding_idx) return;

  const Accum scale = counts == nullptr
      ? static_cast<Accum>(1)
      : static_cast<Accum>(1) / static_cast<Accum>(counts[index]);
  const int64_t grad_offset = index_pos * row_size;
  Accum* dst = grad_weight + index * row_size;
  for (int64_t column = threadIdx.x; column < row_size; column += blockDim.x) {
    atomicAdd(dst + column, static_cast<Accum>(grad_output[grad_offset + column]) * scale);
  }
}

template <typename T, typename Accum, typename IndexT>
__global__ void embedding_backward_single_row_kernel(
    int64_t row_size,
    int64_t num_weights,
    int64_t padding_idx,
    const T* __restrict__ grad_output,
    const IndexT* __restrict__ indices,
    Accum* __restrict__ grad_weight) {
  const int64_t index = static_cast<int64_t>(indices[0]);
  assert(index >= 0 && index < num_weights);
  if (index == padding_idx) return;

  Accum* dst = grad_weight + index * row_size;
  for (int64_t column = threadIdx.x; column < row_size; column += blockDim.x) {
    dst[column] = static_cast<Accum>(grad_output[column]);
  }
}

template <typename T, typename Accum>
__global__ void embedding_backward_cast_kernel(
    int64_t n,
    const Accum* __restrict__ src,
    T* __restrict__ dst) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<T>(src[i]);
}

template <typename IndexT>
__global__ void embedding_count_indices_kernel(
    int64_t n_indices,
    int64_t num_weights,
    int64_t padding_idx,
    const IndexT* __restrict__ indices,
    int64_t* __restrict__ counts) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_indices) return;

  const int64_t index = static_cast<int64_t>(indices[i]);
  assert(index >= 0 && index < num_weights);
  if (index != padding_idx) {
    atomicAdd(reinterpret_cast<unsigned long long*>(counts + index), 1ULL);
  }
}

template <typename IndexT>
void launch_embedding_counts(
    int64_t n_indices,
    int64_t num_weights,
    int64_t padding_idx,
    const IndexT* indices,
    int64_t* counts,
    cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid(static_cast<unsigned int>((n_indices + kThreads - 1) / kThreads));
  embedding_count_indices_kernel<<<grid, block, 0, stream>>>(
      n_indices, num_weights, padding_idx, indices, counts);
}

// ---------------------------------------------------------------------------
// Segmented gradient accumulation.
//
// The rows of an embedding table are looked up over and over, so the gradient
// of a row is a segmented sum over the lookups that hit it.  Sorting the
// lookups by row turns that into a segmented sum over contiguous runs, and
// every run is then cut into pieces of at most `rows_per_chunk` lookups:
//
//   1. every piece is summed on its own into a scratch buffer, no atomics, so
//      the result is reproducible run to run;
//   2. the piece sums belonging to one run are added together and written to
//      the output row.
//
// Cutting the runs is what keeps the machine busy: the available parallelism
// follows the number of pieces and not the number of distinct rows.  A plain
// atomic scatter instead offers one unit of work per lookup, and when a handful
// of rows are hit very often those atomics pile up on the same addresses and
// serialize.
//
// Two shapes of stage 1 are provided.  The scratch variant is the default.  The
// fused variant publishes each piece sum into the output row with an atomic
// add, which removes the serial second stage entirely; it is only worth its
// non-determinism when there are so few runs that the second stage would run on
// a handful of threads while the device idles.
//
// `rows_per_chunk` is picked so that the piece count keeps every
// multiprocessor busy: fewer pieces would starve the first stage, more pieces
// would only add scratch traffic and atomic contention without adding any
// parallelism.
// ---------------------------------------------------------------------------

// Blocks per multiprocessor the piece count aims for.  A block covers
// block_y pieces of block_x features, so this many blocks per multiprocessor is
// what the first stage needs before more pieces stop adding parallelism and
// only add scratch traffic or atomic contention.
constexpr int64_t kEmbeddingBlocksPerSM = 8;
// Beyond this many blocks per multiprocessor the first stage saturates the
// device on its own, so the serial second stage is the cheaper structure.
constexpr int64_t kEmbeddingMaxAtomicBlocksPerSM = 4;
// Thread budget of one block in the accumulation stages.  The features of a
// row are spread over as many feature tiles as it takes to stay inside it.
constexpr int kEmbeddingTileThreads = 256;

// The scatter alternative spends one atomic per gradient element, at roughly
// three times what the streamed read of the segmented path spends per element,
// and those atomics serialize when a few rows are hit over and over.  The
// segmented path instead pays a fixed handful of passes over the index list,
// measured at 50-80us for this class of hardware.  These two element counts are
// where the shapes cross over, measured across row counts from 8 to 65536: past
// the first one the segmented path wins whatever the row count is, and before
// the second one it only wins when the row count is small enough for the atomics
// to pile up.
constexpr int64_t kEmbeddingSegmentedElements = 12 << 20;
constexpr int64_t kEmbeddingContendedElements = 4 << 20;
constexpr int64_t kEmbeddingContendedRows = 64;

__host__ __device__ inline int64_t embedding_ceil_div(int64_t numerator,
                                                      int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// Number of significant bits of the largest legal row index.  Sorting only
// those bits keeps the radix pass count at the minimum the row count allows
// (12 bits for 4096 rows is two 8-bit passes, not three).
inline int embedding_sort_bits(int64_t num_weights) {
  const int64_t max_index = num_weights - 1;
  int bits = 1;
  while ((max_index >> bits) != 0) ++bits;
  return bits;
}

inline int embedding_multiprocessor_count() {
  static thread_local int cached_device = -1;
  static thread_local int cached_count = 0;
  const int device = currentDevice();
  if (device != cached_device) {
    int value = 0;
    checkCuda(cudaDeviceGetAttribute(
                  &value, cudaDevAttrMultiProcessorCount, device),
              "embedding_dense_backward: multiprocessor count");
    cached_count = value;
    cached_device = device;
  }
  return cached_count;
}

// Lookup positions and piece boundaries both index the lookup list, whose
// length the sort interface already bounds by 2^31, so they stay 32-bit even
// when the row indices are 64-bit wide.
using EmbeddingPosition = int32_t;

template <typename PositionT>
__global__ void embedding_position_kernel(int64_t n, PositionT* __restrict__ out) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = static_cast<PositionT>(i);
}

template <typename IndexT>
__global__ void embedding_chunk_count_kernel(
    IndexT* __restrict__ chunks_per_run,
    const IndexT* __restrict__ run_offsets,
    const int64_t* __restrict__ num_runs_ptr,
    int64_t numel,
    int64_t rows_per_chunk) {
  const int64_t num_runs = *num_runs_ptr;
  const int64_t id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= num_runs) return;
  const int64_t begin = run_offsets[id];
  const int64_t end = (id == num_runs - 1) ? numel : run_offsets[id + 1];
  chunks_per_run[id] =
      static_cast<IndexT>(embedding_ceil_div(end - begin, rows_per_chunk));
}

template <typename IndexT>
__global__ void embedding_num_chunks_kernel(
    const IndexT* __restrict__ chunks_per_run,
    const IndexT* __restrict__ chunk_run_base,
    const int64_t* __restrict__ num_runs_ptr,
    int64_t* __restrict__ num_chunks_ptr) {
  const int64_t num_runs = *num_runs_ptr;
  *num_chunks_ptr = static_cast<int64_t>(chunks_per_run[num_runs - 1]) +
      static_cast<int64_t>(chunk_run_base[num_runs - 1]);
}

// One thread per piece.  Locating the owning run by binary search over the
// piece bases keeps this pass linear in the piece count; expanding one run per
// thread instead would serialise on the few-thread case this path exists for.
template <typename IndexT, bool kWriteRunMap>
__global__ void embedding_chunk_expand_kernel(
    IndexT* __restrict__ chunk_offsets,
    IndexT* __restrict__ chunk_run,
    const IndexT* __restrict__ chunk_run_base,
    const IndexT* __restrict__ run_offsets,
    const int64_t* __restrict__ num_runs_ptr,
    const int64_t* __restrict__ num_chunks_ptr,
    int64_t rows_per_chunk) {
  const int64_t num_runs = *num_runs_ptr;
  const int64_t num_chunks = *num_chunks_ptr;
  const int64_t chunk =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk >= num_chunks) return;

  int64_t low = 0;
  int64_t high = num_runs - 1;
  while (low < high) {
    const int64_t mid = (low + high + 1) >> 1;
    if (chunk_run_base[mid] <= chunk) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  chunk_offsets[chunk] = static_cast<IndexT>(
      static_cast<int64_t>(run_offsets[low]) +
      (chunk - chunk_run_base[low]) * rows_per_chunk);
  if constexpr (kWriteRunMap) {
    chunk_run[chunk] = static_cast<IndexT>(low);
  }
}

// Stage 1 with a scratch buffer: threadIdx.x walks the row, threadIdx.y walks
// the pieces, so a block covers several pieces of the same features at once.
template <typename T, typename Accum, typename IndexT, typename PositionT>
__global__ void embedding_chunk_sum_kernel(
    const PositionT* __restrict__ lookup_rows,
    const T* __restrict__ grad_output,
    const IndexT* __restrict__ sorted_rows,
    const int64_t* __restrict__ counts,
    int64_t numel,
    int64_t stride,
    const IndexT* __restrict__ chunk_offsets,
    const int64_t* __restrict__ num_chunks_ptr,
    Accum* __restrict__ chunk_sums) {
  const int64_t num_chunks = *num_chunks_ptr;
  const int64_t feature =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (feature >= stride) return;

  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.y;
  for (int64_t chunk =
           static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       chunk < num_chunks; chunk += step) {
    const int64_t begin = chunk_offsets[chunk];
    const int64_t end =
        (chunk == num_chunks - 1) ? numel : chunk_offsets[chunk + 1];
    Accum sum = Accum(0);
    for (int64_t i = begin; i < end; ++i) {
      // The lookup position addresses the incoming gradient; the sorted key is
      // the row it belongs to, which is what the frequency table is indexed by.
      const int64_t position = static_cast<int64_t>(lookup_rows[i]);
      Accum value = static_cast<Accum>(grad_output[position * stride + feature]);
      if (counts != nullptr) {
        value /= static_cast<Accum>(counts[sorted_rows[i]]);
      }
      sum += value;
    }
    chunk_sums[chunk * stride + feature] = sum;
  }
}

// Stage 1 fused into the output: identical accumulation, but each piece sum
// lands in its row with an atomic add instead of a scratch slot, so no second
// pass over the pieces is needed.
template <typename T, typename Accum, typename IndexT, typename PositionT>
__global__ void embedding_chunk_atomic_kernel(
    const PositionT* __restrict__ lookup_rows,
    const T* __restrict__ grad_output,
    const int64_t* __restrict__ counts,
    int64_t numel,
    int64_t stride,
    const IndexT* __restrict__ chunk_offsets,
    const int64_t* __restrict__ num_chunks_ptr,
    const IndexT* __restrict__ sorted_rows,
    const IndexT* __restrict__ chunk_run,
    const IndexT* __restrict__ run_offsets,
    Accum* __restrict__ grad_weight,
    int64_t padding_idx) {
  const int64_t num_chunks = *num_chunks_ptr;
  const int64_t feature =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (feature >= stride) return;

  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.y;
  for (int64_t chunk =
           static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       chunk < num_chunks; chunk += step) {
    const int64_t begin = chunk_offsets[chunk];
    const int64_t end =
        (chunk == num_chunks - 1) ? numel : chunk_offsets[chunk + 1];
    Accum sum = Accum(0);
    for (int64_t i = begin; i < end; ++i) {
      // The lookup position addresses the incoming gradient; the sorted key is
      // the row it belongs to, which is what the frequency table is indexed by.
      const int64_t position = static_cast<int64_t>(lookup_rows[i]);
      Accum value = static_cast<Accum>(grad_output[position * stride + feature]);
      if (counts != nullptr) {
        value /= static_cast<Accum>(counts[sorted_rows[i]]);
      }
      sum += value;
    }
    const int64_t run = static_cast<int64_t>(chunk_run[chunk]);
    // Every lookup of a run shares its row, so the row of the run is the one
    // sitting at the run's first sorted position.
    const int64_t target = static_cast<int64_t>(sorted_rows[run_offsets[run]]);
    if (target != padding_idx) {
      atomicAdd(grad_weight + target * stride + feature, sum);
    }
  }
}

// Stage 2: add up the piece sums of one run and store them in its row.  Every
// run owns exactly one row, so plain stores are enough.
template <typename Accum, typename IndexT>
__global__ void embedding_run_write_kernel(
    const IndexT* __restrict__ sorted_rows,
    const Accum* __restrict__ chunk_sums,
    Accum* __restrict__ grad_weight,
    int64_t stride,
    const IndexT* __restrict__ run_offsets,
    const int64_t* __restrict__ num_runs_ptr,
    const IndexT* __restrict__ chunk_run_base,
    const int64_t* __restrict__ num_chunks_ptr,
    int64_t padding_idx) {
  const int64_t num_runs = *num_runs_ptr;
  const int64_t num_chunks = *num_chunks_ptr;
  const int64_t feature =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (feature >= stride) return;

  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.y;
  for (int64_t run = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
       run < num_runs; run += step) {
    const int64_t first = chunk_run_base[run];
    const int64_t last =
        (run == num_runs - 1) ? num_chunks : chunk_run_base[run + 1];
    Accum sum = Accum(0);
    for (int64_t chunk = first; chunk < last; ++chunk) {
      sum += chunk_sums[chunk * stride + feature];
    }
    const int64_t target = static_cast<int64_t>(sorted_rows[run_offsets[run]]);
    if (target != padding_idx) {
      grad_weight[target * stride + feature] = sum;
    }
  }
}

// Returns false when the segmented path does not apply and the caller must
// fall back to the atomic scatter.  used_atomic reports whether the chosen
// variant publishes through atomics.
template <typename T, typename Accum, typename IndexT>
bool embedding_backward_segmented(
    int64_t n_indices,
    int64_t stride,
    int64_t num_weights,
    int64_t padding_idx,
    const Tensor& grad_output,
    const IndexT* indices,
    const int64_t* counts,
    Accum* grad_weight,
    DType accum_dtype,
    cudaStream_t stream,
    bool* used_atomic) {
  if (n_indices <= 0 || num_weights <= 0) return true;
  if (n_indices > std::numeric_limits<int>::max()) return false;
  if (stride <= 0) return true;

  const Device device = grad_output.device();
  const T* grad_data = grad_output.data_ptr<T>();
  const DType index_dtype = sizeof(IndexT) == 8 ? DType::Int64 : DType::Int32;
  const int64_t max_runs = std::min<int64_t>(n_indices, num_weights);

  // 1. sort the lookups by row, carrying the original position of each lookup
  //    so the gradient rows can be gathered afterwards.
  Tensor sorted_rows = Tensor::empty({n_indices}, index_dtype, device);
  Tensor source_rows = Tensor::empty({n_indices}, DType::Int32, device);
  Tensor positions = Tensor::empty({n_indices}, DType::Int32, device);
  const int64_t position_blocks = embedding_ceil_div(n_indices, kThreads);
  embedding_position_kernel<<<static_cast<unsigned int>(position_blocks),
                              kThreads, 0, stream>>>(
      n_indices, positions.data_ptr<EmbeddingPosition>());
  checkCuda(cudaGetLastError(), "embedding_dense_backward: positions");
  const int sort_bits = embedding_sort_bits(num_weights);
  size_t temp_bytes = 0;
  checkCuda(cub::DeviceRadixSort::SortPairs(
                nullptr, temp_bytes, indices, sorted_rows.data_ptr<IndexT>(),
                positions.data_ptr<EmbeddingPosition>(),
                source_rows.data_ptr<EmbeddingPosition>(),
                static_cast<int>(n_indices), 0, sort_bits, stream),
            "embedding_dense_backward: sort size");
  Tensor sort_temp = Tensor::empty(
      {static_cast<int64_t>(temp_bytes == 0 ? 1 : temp_bytes)}, DType::UInt8,
      device);
  checkCuda(cub::DeviceRadixSort::SortPairs(
                sort_temp.data_ptr(), temp_bytes, indices,
                sorted_rows.data_ptr<IndexT>(),
                positions.data_ptr<EmbeddingPosition>(),
                source_rows.data_ptr<EmbeddingPosition>(),
                static_cast<int>(n_indices), 0, sort_bits, stream),
            "embedding_dense_backward: sort");

  // 2. one entry per run: where it starts in the sorted order.  The run start
  //    offsets are what the passes below consume, so the distinct row values
  //    land in a scratch buffer that is never read back.
  Tensor run_offsets = Tensor::empty({n_indices}, index_dtype, device);
  Tensor unique_rows = Tensor::empty({max_runs}, index_dtype, device);
  Tensor num_runs = Tensor::empty({}, DType::Int64, device);
  thrust::counting_iterator<EmbeddingPosition> run_starts(0);
  temp_bytes = 0;
  checkCuda(cub::DeviceSelect::UniqueByKey(
                nullptr, temp_bytes, sorted_rows.data_ptr<IndexT>(), run_starts,
                unique_rows.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
                num_runs.data_ptr<int64_t>(), static_cast<int>(n_indices),
                stream),
            "embedding_dense_backward: run split size");
  Tensor unique_temp = Tensor::empty(
      {static_cast<int64_t>(temp_bytes == 0 ? 1 : temp_bytes)}, DType::UInt8,
      device);
  checkCuda(cub::DeviceSelect::UniqueByKey(
                unique_temp.data_ptr(), temp_bytes,
                sorted_rows.data_ptr<IndexT>(), run_starts,
                unique_rows.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
                num_runs.data_ptr<int64_t>(), static_cast<int>(n_indices),
                stream),
            "embedding_dense_backward: run split");

  // 3. launch shape.  threadIdx.x walks the features, threadIdx.y walks the
  //    pieces or runs, and blockIdx.y walks the feature tiles.  Splitting the
  //    features across the y grid keeps a block at a fixed thread budget, so
  //    short rows still get blocks wide enough to saturate the device while
  //    rows longer than one block stay coalesced per tile.
  const int64_t stride_warped =
      embedding_ceil_div(stride, kEmbeddingWarpSize) * kEmbeddingWarpSize;
  const int block_x = static_cast<int>(
      std::min<int64_t>(stride_warped, kEmbeddingTileThreads));
  const int block_y = std::max(1, kEmbeddingTileThreads / block_x);
  const int64_t feature_tiles = embedding_ceil_div(stride_warped, block_x);
  const dim3 block(static_cast<unsigned int>(block_x),
                   static_cast<unsigned int>(block_y));
  const int64_t block_threads =
      static_cast<int64_t>(block_x) * static_cast<int64_t>(block_y);
  // Longest piece the first stage will sum: short enough to keep every
  // multiprocessor busy, long enough to keep the scratch traffic and the
  // atomic count down.
  const int64_t rows_per_chunk = std::max<int64_t>(
      1, embedding_ceil_div(n_indices,
                            static_cast<int64_t>(embedding_multiprocessor_count()) *
                                kEmbeddingBlocksPerSM * block_y));
  // Upper bound on the piece count: every run contributes at least one piece
  // and at most one piece per rows_per_chunk lookups.
  const int64_t max_chunks =
      embedding_ceil_div(n_indices, rows_per_chunk) + max_runs;
  const int64_t expand_blocks = embedding_ceil_div(max_chunks, kThreads);

  // 4. cut every run into pieces: how many pieces per run, the exclusive
  //    prefix over that count, and where each piece starts.
  Tensor chunks_per_run = Tensor::zeros({max_runs}, index_dtype, device);
  Tensor chunk_run_base = Tensor::empty({max_runs}, index_dtype, device);
  Tensor num_chunks = Tensor::empty({}, DType::Int64, device);
  Tensor chunk_offsets = Tensor::empty({max_chunks}, index_dtype, device);
  const int64_t* num_runs_ptr = num_runs.data_ptr<int64_t>();
  int64_t* num_chunks_ptr = num_chunks.data_ptr<int64_t>();
  const int run_blocks = static_cast<int>(
      embedding_ceil_div(max_runs, kEmbeddingWarpSize));
  const dim3 run_block(kEmbeddingWarpSize);
  embedding_chunk_count_kernel<<<run_blocks, run_block, 0, stream>>>(
      chunks_per_run.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
      num_runs_ptr, n_indices, rows_per_chunk);
  checkCuda(cudaGetLastError(), "embedding_dense_backward: piece count");
  temp_bytes = 0;
  checkCuda(cub::DeviceScan::ExclusiveSum(
                nullptr, temp_bytes, chunks_per_run.data_ptr<IndexT>(),
                chunk_run_base.data_ptr<IndexT>(),
                static_cast<int>(max_runs), stream),
            "embedding_dense_backward: piece base size");
  Tensor scan_temp = Tensor::empty(
      {static_cast<int64_t>(temp_bytes == 0 ? 1 : temp_bytes)}, DType::UInt8,
      device);
  checkCuda(cub::DeviceScan::ExclusiveSum(
                scan_temp.data_ptr(), temp_bytes,
                chunks_per_run.data_ptr<IndexT>(),
                chunk_run_base.data_ptr<IndexT>(),
                static_cast<int>(max_runs), stream),
            "embedding_dense_backward: piece base");
  embedding_num_chunks_kernel<<<1, 1, 0, stream>>>(
      chunks_per_run.data_ptr<IndexT>(), chunk_run_base.data_ptr<IndexT>(),
      num_runs_ptr, num_chunks_ptr);
  checkCuda(cudaGetLastError(), "embedding_dense_backward: piece total");

  // 5. The second stage is serial in the number of runs: when the runs alone
  // cannot fill the device but the pieces can, fusing it away is the better
  // trade.  Deterministic requests keep the scratch variant.
  const int64_t run_blocks_full =
      embedding_ceil_div(max_runs * stride_warped, block_threads);
  const bool use_atomic =
      !globalContext().deterministicAlgorithms() &&
      run_blocks_full < static_cast<int64_t>(embedding_multiprocessor_count()) *
                            kEmbeddingMaxAtomicBlocksPerSM &&
      max_chunks > run_blocks_full * 4;
  *used_atomic = use_atomic;

  if (use_atomic) {
    Tensor chunk_run = Tensor::empty({max_chunks}, index_dtype, device);
    embedding_chunk_expand_kernel<IndexT, true>
        <<<static_cast<unsigned int>(expand_blocks), kThreads, 0, stream>>>(
            chunk_offsets.data_ptr<IndexT>(), chunk_run.data_ptr<IndexT>(),
            chunk_run_base.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
            num_runs_ptr, num_chunks_ptr, rows_per_chunk);
    checkCuda(cudaGetLastError(), "embedding_dense_backward: piece map");
    const dim3 grid(static_cast<unsigned int>(
                        embedding_ceil_div(max_chunks, block_y)),
                    static_cast<unsigned int>(feature_tiles));
    embedding_chunk_atomic_kernel<T, Accum, IndexT, EmbeddingPosition>
        <<<grid, block, 0, stream>>>(
            source_rows.data_ptr<EmbeddingPosition>(), grad_data, counts,
            n_indices, stride, chunk_offsets.data_ptr<IndexT>(),
            num_chunks_ptr, sorted_rows.data_ptr<IndexT>(),
            chunk_run.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
            grad_weight, padding_idx);
    checkCuda(cudaGetLastError(), "embedding_dense_backward: fused pieces");
    return true;
  }

  embedding_chunk_expand_kernel<IndexT, false>
      <<<static_cast<unsigned int>(expand_blocks), kThreads, 0, stream>>>(
          chunk_offsets.data_ptr<IndexT>(), static_cast<IndexT*>(nullptr),
          chunk_run_base.data_ptr<IndexT>(), run_offsets.data_ptr<IndexT>(),
          num_runs_ptr, num_chunks_ptr, rows_per_chunk);
  checkCuda(cudaGetLastError(), "embedding_dense_backward: piece offsets");
  Tensor chunk_sums = Tensor::empty({max_chunks * stride}, accum_dtype, device);
  const dim3 sum_grid(static_cast<unsigned int>(
                          embedding_ceil_div(max_chunks, block_y)),
                      static_cast<unsigned int>(feature_tiles));
  embedding_chunk_sum_kernel<T, Accum, IndexT, EmbeddingPosition>
      <<<sum_grid, block, 0, stream>>>(
          source_rows.data_ptr<EmbeddingPosition>(), grad_data,
          sorted_rows.data_ptr<IndexT>(), counts, n_indices, stride,
          chunk_offsets.data_ptr<IndexT>(), num_chunks_ptr,
          chunk_sums.data_ptr<Accum>());
  checkCuda(cudaGetLastError(), "embedding_dense_backward: piece sums");
  const dim3 write_grid(static_cast<unsigned int>(
                            embedding_ceil_div(max_runs, block_y)),
                        static_cast<unsigned int>(feature_tiles));
  embedding_run_write_kernel<Accum, IndexT>
      <<<write_grid, block, 0, stream>>>(
          sorted_rows.data_ptr<IndexT>(), chunk_sums.data_ptr<Accum>(),
          grad_weight, stride, run_offsets.data_ptr<IndexT>(), num_runs_ptr,
          chunk_run_base.data_ptr<IndexT>(), num_chunks_ptr, padding_idx);
  checkCuda(cudaGetLastError(), "embedding_dense_backward: run reduce");
  return true;
}

template <typename T, typename Accum, typename IndexT>
void launch_embedding_backward_dtype(
    int64_t n_indices,
    int64_t row_size,
    int64_t num_weights,
    int64_t padding_idx,
    const Tensor& grad_output,
    const Tensor& indices,
    const int64_t* counts,
    Accum* grad_weight,
    DType accum_dtype,
    cudaStream_t stream,
    bool* used_atomic) {
  const T* grad_data = grad_output.data_ptr<T>();
  const IndexT* index_data = indices.data_ptr<IndexT>();

  *used_atomic = true;
  const int64_t elements = n_indices * row_size;
  const int64_t distinct_rows = std::min<int64_t>(num_weights, n_indices);
  if (elements >= kEmbeddingSegmentedElements ||
      (distinct_rows <= kEmbeddingContendedRows &&
       elements >= kEmbeddingContendedElements)) {
    if (embedding_backward_segmented<T, Accum, IndexT>(
            n_indices, row_size, num_weights, padding_idx, grad_output,
            index_data, counts, grad_weight, accum_dtype, stream,
            used_atomic)) {
      return;
    }
  }

  if (n_indices == 1) {
    const dim3 block(static_cast<unsigned int>(block_size_for(row_size)));
    embedding_backward_single_row_kernel<<<1, block, 0, stream>>>(
        row_size, num_weights, padding_idx, grad_data, index_data, grad_weight);
    return;
  }

  if (row_size <= kSmallEmbeddingDim) {
    const int64_t total = n_indices * row_size;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
    embedding_backward_flat_kernel<<<grid, block, 0, stream>>>(
        n_indices, row_size, num_weights, padding_idx,
        grad_data, index_data, counts, grad_weight);
    return;
  }

  const dim3 block(static_cast<unsigned int>(block_size_for(row_size)));
  const dim3 grid(static_cast<unsigned int>(n_indices));
  embedding_backward_row_kernel<<<grid, block, 0, stream>>>(
      n_indices, row_size, num_weights, padding_idx,
      grad_data, index_data, counts, grad_weight);
}

template <typename IndexT>
void launch_embedding_backward(
    int64_t n_indices,
    int64_t row_size,
    int64_t num_weights,
    int64_t padding_idx,
    const Tensor& grad_output,
    const Tensor& indices,
    const int64_t* counts,
    Tensor& grad_weight,
    cudaStream_t stream,
    bool* used_atomic) {
  const IndexT* index_data = indices.data_ptr<IndexT>();
  if (row_size == 0) {
    launch_embedding_validate(n_indices, num_weights, index_data, stream);
    return;
  }

  switch (grad_output.dtype()) {
    case DType::Float32:
      launch_embedding_backward_dtype<float, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(),
          DType::Float32, stream, used_atomic);
      break;
    case DType::Float64:
      launch_embedding_backward_dtype<double, double, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<double>(),
          DType::Float64, stream, used_atomic);
      break;
    case DType::Float16:
      launch_embedding_backward_dtype<tensorplay::Half, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(),
          DType::Float32, stream, used_atomic);
      break;
    case DType::BFloat16:
      launch_embedding_backward_dtype<tensorplay::BFloat16, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(),
          DType::Float32, stream, used_atomic);
      break;
    default:
      TP_THROW(NotImplementedError,
               "embedding_dense_backward_cuda: unsupported gradient dtype");
  }
}

} // namespace

namespace {

constexpr int kEmbeddingRenormThreads = 128;

template <typename T>
__device__ inline T embedding_warp_reduce_sum(T value) {
#pragma unroll
    for (int offset = kEmbeddingWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    return value;
}

template <typename T>
__device__ inline T embedding_block_reduce_sum(T value, T* shared) {
    const int lane = threadIdx.x % kEmbeddingWarpSize;
    const int warp = threadIdx.x / kEmbeddingWarpSize;
    const int warps = blockDim.x / kEmbeddingWarpSize;
    value = embedding_warp_reduce_sum(value);
    __syncthreads();
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < warps ? shared[lane] : T(0);
    if (warp == 0) value = embedding_warp_reduce_sum(value);
    return value;
}

__device__ inline float embedding_abs(float value) {
    return fabsf(value);
}

__device__ inline double embedding_abs(double value) {
    return fabs(value);
}

__device__ inline float embedding_pow(float value, float exponent) {
    return powf(value, exponent);
}

__device__ inline double embedding_pow(double value, double exponent) {
    return ::pow(value, exponent);
}

template <typename Scalar, typename Acc, typename Index>
__global__ void embedding_renorm_kernel(
        Scalar* weights,
        const Index* indices,
        Acc max_norm,
        Acc norm_type,
        int64_t dim,
        int64_t num_weights,
        int64_t weights_stride0,
        int64_t weights_stride1,
        const int32_t* num_unique_indices) {
    if (static_cast<int64_t>(blockIdx.x) >= *num_unique_indices) return;

    extern __shared__ unsigned char shared_bytes[];
    Acc* shared = reinterpret_cast<Acc*>(shared_bytes);
    const int64_t index = static_cast<int64_t>(indices[blockIdx.x]);
    assert(index >= 0 && index < num_weights);
    const int64_t base = index * weights_stride0;

    Acc value = Acc(0);
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const Acc x = static_cast<Acc>(weights[base + i * weights_stride1]);
        if (norm_type == Acc(1)) {
            value += embedding_abs(x);
        } else if (norm_type == Acc(2)) {
            value += x * x;
        } else {
            value += embedding_pow(x, norm_type);
        }
    }
    value = embedding_block_reduce_sum(value, shared);
    if (threadIdx.x == 0) {
        shared[0] = embedding_pow(value, Acc(1) / norm_type);
    }
    __syncthreads();
    if (shared[0] > max_norm) {
        const Acc factor = max_norm / (shared[0] + Acc(1e-7));
        for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
            weights[base + i * weights_stride1] = static_cast<Scalar>(
                static_cast<Acc>(weights[base + i * weights_stride1]) * factor);
        }
    }
}

template <typename Index>
__global__ void embedding_renorm_wrap_indices_kernel(
        const Index* indices,
        Index* wrapped_indices,
        int64_t num_indices,
        int64_t num_weights) {
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= num_indices) return;
    int64_t index = static_cast<int64_t>(indices[offset]);
    assert(index >= -num_weights && index < num_weights);
    if (index < 0) index += num_weights;
    wrapped_indices[offset] = static_cast<Index>(index);
}

template <typename Scalar, typename Acc, typename Index>
void embedding_renorm_dtype(
        Tensor& weight,
        const Tensor& indices,
        double max_norm,
        double norm_type,
        cudaStream_t stream) {
    const int64_t num_indices = indices.numel();
    if (num_indices == 0) return;
    TP_CHECK(num_indices <= std::numeric_limits<int>::max(),
             "embedding_renorm_: too many indices");
    const int64_t num_weights = weight.size(0);
    TP_CHECK(num_weights > 0, "embedding_renorm_: weight must have rows");
    const int64_t dim = weight.stride(0);

    Tensor wrapped = Tensor::empty(
        {num_indices}, indices.dtype(), weight.device());
    const int threads = kEmbeddingRenormThreads;
    const int blocks = static_cast<int>((num_indices + threads - 1) / threads);
    embedding_renorm_wrap_indices_kernel<<<blocks, threads, 0, stream>>>(
        indices.data_ptr<Index>(), wrapped.data_ptr<Index>(),
        num_indices, num_weights);
    CUDA_CHECK(cudaGetLastError());

    Tensor sorted = Tensor::empty(
        {num_indices}, indices.dtype(), weight.device());
    size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, sort_bytes, wrapped.data_ptr<Index>(), sorted.data_ptr<Index>(),
        static_cast<int>(num_indices), 0, sizeof(Index) * 8, stream));
    Tensor sort_storage = Tensor::empty(
        {static_cast<int64_t>(sort_bytes == 0 ? 1 : sort_bytes)},
        DType::UInt8, weight.device());
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        sort_storage.data_ptr(), sort_bytes,
        wrapped.data_ptr<Index>(), sorted.data_ptr<Index>(),
        static_cast<int>(num_indices), 0, sizeof(Index) * 8, stream));

    Tensor unique = Tensor::empty(
        {num_indices}, indices.dtype(), weight.device());
    Tensor unique_count = Tensor::empty({}, DType::Int32, weight.device());
    size_t unique_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(
        nullptr, unique_bytes, sorted.data_ptr<Index>(),
        unique.data_ptr<Index>(), unique_count.data_ptr<int32_t>(),
        static_cast<int>(num_indices), stream));
    Tensor unique_storage = Tensor::empty(
        {static_cast<int64_t>(unique_bytes == 0 ? 1 : unique_bytes)},
        DType::UInt8, weight.device());
    CUDA_CHECK(cub::DeviceSelect::Unique(
        unique_storage.data_ptr(), unique_bytes,
        sorted.data_ptr<Index>(), unique.data_ptr<Index>(),
        unique_count.data_ptr<int32_t>(), static_cast<int>(num_indices), stream));

    embedding_renorm_kernel<Scalar, Acc, Index>
        <<<static_cast<unsigned int>(num_indices), threads,
           threads * sizeof(Acc), stream>>>(
            weight.data_ptr<Scalar>(), unique.data_ptr<Index>(),
            static_cast<Acc>(max_norm), static_cast<Acc>(norm_type), dim,
            num_weights, weight.stride(0), weight.stride(1),
            unique_count.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());
}

}

Tensor& embedding_renorm_cuda(
        Tensor& weight, const Tensor& indices, double max_norm, double norm_type) {
    if (weight.dim() != 2) {
        TP_THROW(RuntimeError,
                 "embedding_renorm_cuda: weight must be 2-D, got dim ",
                 weight.dim(), " defined ", weight.defined());
    }
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_renorm_: indices must be Int64 or Int32");
    }
    if (weight.device() != indices.device()) {
        TP_THROW(DeviceMismatchError,
                 "embedding_renorm_: weight and indices must be on the same device");
    }
    Tensor indices_contig = indices.is_contiguous() ? indices : indices.contiguous();
    cudaStream_t stream = getCurrentCUDAStream().stream();
    if (indices.dtype() == DType::Int64) {
        switch (weight.dtype()) {
            case DType::Float32:
                embedding_renorm_dtype<float, float, int64_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::Float64:
                embedding_renorm_dtype<double, double, int64_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::Float16:
                embedding_renorm_dtype<tensorplay::Half, float, int64_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::BFloat16:
                embedding_renorm_dtype<tensorplay::BFloat16, float, int64_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "embedding_renorm_cuda: unsupported weight dtype");
        }
    } else {
        switch (weight.dtype()) {
            case DType::Float32:
                embedding_renorm_dtype<float, float, int32_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::Float64:
                embedding_renorm_dtype<double, double, int32_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::Float16:
                embedding_renorm_dtype<tensorplay::Half, float, int32_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            case DType::BFloat16:
                embedding_renorm_dtype<tensorplay::BFloat16, float, int32_t>(
                    weight, indices_contig, max_norm, norm_type, stream);
                break;
            default:
                TP_THROW(NotImplementedError,
                         "embedding_renorm_cuda: unsupported weight dtype");
        }
    }
    return weight;
}

Tensor embedding_cuda(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  if (weight.dim() != 2) TP_THROW(RuntimeError, "'weight' must be 2-D");
  if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
    TP_THROW(TypeError, "embedding: indices must be Int64 or Int32");
  }
  // scale_grad_by_freq affects only the derivative; accepting it here is
  (void)scale_grad_by_freq;
  (void)padding_idx;

  const Tensor weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
  const Tensor indices_contig = indices.is_contiguous() ? indices : indices.contiguous();
  const int64_t num_weights = weight_contig.size(0);
  const int64_t embedding_dim = weight_contig.size(1);
  const int64_t n_indices = indices_contig.numel();

  std::vector<int64_t> output_shape = static_cast<std::vector<int64_t>>(indices.shape());
  output_shape.push_back(embedding_dim);
  Tensor output = Tensor::empty(output_shape, weight.dtype(), weight.device());
  if (n_indices == 0) return output;

  const cudaStream_t stream = getCurrentCUDAStream().stream();
  if (indices_contig.dtype() == DType::Int64) {
    launch_embedding_forward<int64_t>(
        n_indices, embedding_dim, num_weights,
        weight_contig, indices_contig, output, stream);
  } else {
    launch_embedding_forward<int32_t>(
        n_indices, embedding_dim, num_weights,
        weight_contig, indices_contig, output, stream);
  }
  CUDA_CHECK(cudaGetLastError());
  return output;
}

Tensor embedding_dense_backward_cuda(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
    TP_THROW(TypeError, "embedding_dense_backward: indices must be Int64 or Int32");
  }
  if (num_weights < 0) {
    TP_THROW(ValueError, "embedding_dense_backward: num_weights must be non-negative");
  }
  if (grad_output.dim() != indices.dim() + 1) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: grad_output rank must equal indices rank + 1");
  }
  for (int64_t dim = 0; dim < indices.dim(); ++dim) {
    if (grad_output.size(dim) != indices.size(dim)) {
      TP_THROW(RuntimeError,
               "embedding_dense_backward: grad_output shape does not match indices shape");
    }
  }
  if (grad_output.dtype() == DType::Bool) {
    TP_THROW(RuntimeError, "embedding_dense_backward: grad_output cannot be Bool");
  }

  // The public Python functional wrapper normalizes negative padding indices;
  // keep the native sentinel -1 for "no padding" and accept already-normalized
  // values here as well.
  if (padding_idx < -1) {
    if (padding_idx < -num_weights) {
      TP_THROW(ValueError, "embedding_dense_backward: padding_idx out of range");
    }
    padding_idx += num_weights;
  }
  if (padding_idx >= num_weights && padding_idx != -1) {
    TP_THROW(ValueError, "embedding_dense_backward: padding_idx out of range");
  }

  const Tensor indices_contig = indices.is_contiguous() ? indices : indices.contiguous();
  const Tensor grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  const int64_t n_indices = indices_contig.numel();
  const int64_t row_size = grad_output.size(-1);
  const std::vector<int64_t> grad_shape = {num_weights, row_size};
  Tensor grad_weight = Tensor::zeros(grad_shape, grad_output.dtype(), grad_output.device());
  if (n_indices == 0) return grad_weight;

  const cudaStream_t stream = getCurrentCUDAStream().stream();
  if (row_size == 0) {
    if (indices_contig.dtype() == DType::Int64) {
      launch_embedding_validate<int64_t>(
          n_indices, num_weights, indices_contig.data_ptr<int64_t>(), stream);
    } else {
      launch_embedding_validate<int32_t>(
          n_indices, num_weights, indices_contig.data_ptr<int32_t>(), stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
  }
  const int64_t* counts_ptr = nullptr;
  Tensor counts;
  if (scale_grad_by_freq) {
    counts = Tensor::zeros({num_weights}, DType::Int64, grad_output.device());
    if (indices_contig.dtype() == DType::Int64) {
      launch_embedding_counts<int64_t>(
          n_indices, num_weights, padding_idx,
          indices_contig.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), stream);
    } else {
      launch_embedding_counts<int32_t>(
          n_indices, num_weights, padding_idx,
          indices_contig.data_ptr<int32_t>(), counts.data_ptr<int64_t>(), stream);
    }
    counts_ptr = counts.data_ptr<int64_t>();
  }

  // The segmented path is reproducible unless it picks the fused variant, so
  // the non-determinism alert follows the variant that actually ran.
  bool used_atomic = true;
  if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
    Tensor accum = Tensor::zeros(grad_shape, DType::Float32, grad_output.device());
    if (indices_contig.dtype() == DType::Int64) {
      launch_embedding_backward<int64_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, accum, stream, &used_atomic);
    } else {
      launch_embedding_backward<int32_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, accum, stream, &used_atomic);
    }
    const int64_t total = num_weights * row_size;
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((total + kThreads - 1) / kThreads));
    if (grad_output.dtype() == DType::Float16) {
      embedding_backward_cast_kernel<tensorplay::Half, float>
          <<<grid, block, 0, stream>>>(
              total, accum.data_ptr<float>(), grad_weight.data_ptr<tensorplay::Half>());
    } else {
      embedding_backward_cast_kernel<tensorplay::BFloat16, float>
          <<<grid, block, 0, stream>>>(
              total, accum.data_ptr<float>(), grad_weight.data_ptr<tensorplay::BFloat16>());
    }
  } else if (grad_output.dtype() == DType::Float32 || grad_output.dtype() == DType::Float64) {
    if (indices_contig.dtype() == DType::Int64) {
      launch_embedding_backward<int64_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, grad_weight, stream, &used_atomic);
    } else {
      launch_embedding_backward<int32_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, grad_weight, stream, &used_atomic);
    }
  } else {
    TP_THROW(NotImplementedError,
             "embedding_dense_backward_cuda: unsupported gradient dtype");
  }

  if (used_atomic) {
    globalContext().alertNotDeterministic("embedding_dense_backward_cuda");
  }
  CUDA_CHECK(cudaGetLastError());
  return grad_weight;
}

Tensor embedding_backward_cuda(const Tensor& grad_output, const Tensor& indices,
                               int64_t num_weights, int64_t padding_idx,
                               bool scale_grad_by_freq, bool sparse) {
  if (sparse) {
    return embedding_sparse_backward_cuda(grad_output, indices, num_weights,
                                          padding_idx, scale_grad_by_freq);
  }
  return embedding_dense_backward_cuda(grad_output, indices, num_weights,
                                       padding_idx, scale_grad_by_freq);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EmbeddingKernels) {
  m.impl("embedding", embedding_cuda);
  m.impl("embedding_renorm_", embedding_renorm_cuda);
  m.impl("embedding_dense_backward", embedding_dense_backward_cuda);
  m.impl("embedding_sparse_backward", embedding_sparse_backward_cuda);
  m.impl("embedding_backward", embedding_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
