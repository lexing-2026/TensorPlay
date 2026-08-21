#include "Tensor.h"
#include "Dispatcher.h"
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
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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
    cudaStream_t stream) {
  const T* grad_data = grad_output.data_ptr<T>();
  const IndexT* index_data = indices.data_ptr<IndexT>();

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
    cudaStream_t stream) {
  const IndexT* index_data = indices.data_ptr<IndexT>();
  if (row_size == 0) {
    launch_embedding_validate(n_indices, num_weights, index_data, stream);
    return;
  }

  switch (grad_output.dtype()) {
    case DType::Float32:
      launch_embedding_backward_dtype<float, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(), stream);
      break;
    case DType::Float64:
      launch_embedding_backward_dtype<double, double, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<double>(), stream);
      break;
    case DType::Float16:
      launch_embedding_backward_dtype<tensorplay::Half, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(), stream);
      break;
    case DType::BFloat16:
      launch_embedding_backward_dtype<tensorplay::BFloat16, float, IndexT>(
          n_indices, row_size, num_weights, padding_idx,
          grad_output, indices, counts, grad_weight.data_ptr<float>(), stream);
      break;
    default:
      TP_THROW(NotImplementedError,
               "embedding_dense_backward_cuda: unsupported gradient dtype");
  }
}

} // namespace

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
  if (sparse) {
    TP_THROW(NotImplementedError, "embedding_cuda: sparse gradients are not supported");
  }

  // scale_grad_by_freq affects only the derivative; accepting it here is
  // required for the forward half of the Torch API contract.
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

  if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
    Tensor accum = Tensor::zeros(grad_shape, DType::Float32, grad_output.device());
    if (indices_contig.dtype() == DType::Int64) {
      launch_embedding_backward<int64_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, accum, stream);
    } else {
      launch_embedding_backward<int32_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, accum, stream);
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
          grad_contig, indices_contig, counts_ptr, grad_weight, stream);
    } else {
      launch_embedding_backward<int32_t>(
          n_indices, row_size, num_weights, padding_idx,
          grad_contig, indices_contig, counts_ptr, grad_weight, stream);
    }
  } else {
    TP_THROW(NotImplementedError,
             "embedding_dense_backward_cuda: unsupported gradient dtype");
  }

  CUDA_CHECK(cudaGetLastError());
  return grad_weight;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, EmbeddingKernels) {
  m.impl("embedding", embedding_cuda);
  m.impl("embedding_dense_backward", embedding_dense_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
