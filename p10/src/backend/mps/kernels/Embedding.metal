#include "Embedding.h"
#include "atomic.h"
#include "utils.h"
#include <metal_stdlib>

using namespace metal;
using namespace p10::metal;

template <typename O>
static inline void resolve_outer_offsets(
    uint tid,
    constant EmbeddingDenseBackwardParams<uint32_t>& params,
    thread O& grad_offset,
    thread O& indices_offset) {
  grad_offset = 0;
  indices_offset = 0;
  uint remaining = tid;
  uint nd = params.outer_ndim;
  for (uint d = 0; d < nd; ++d) {
    uint dim_idx = nd - 1 - d;
    uint dim_size = params.outer_sizes[dim_idx];
    uint coord = remaining % dim_size;
    remaining /= dim_size;
    grad_offset += static_cast<O>(coord) *
        static_cast<O>(params.grad_outer_strides[dim_idx]);
    indices_offset +=
        static_cast<O>(coord) * static_cast<O>(params.indices_strides[dim_idx]);
  }
}

template <typename I, typename O>
kernel void embedding_dense_backward_count(
    constant I* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  O grad_offset_unused = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(tid, params, grad_offset_unused, indices_offset);
  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }
  atomic_fetch_add_explicit(counts + weight_idx, 1u, memory_order_relaxed);
}

template <typename T, typename I, typename O>
kernel void embedding_dense_backward(
    constant T* grad [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    constant uint* counts [[buffer(2)]],
    device AtomicType_t<float>* grad_weight [[buffer(3)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  uint feature_size = params.feature_size;
  uint index_pos = tid / feature_size;
  uint feature_idx = tid % feature_size;

  O grad_outer_offset = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(
      index_pos, params, grad_outer_offset, indices_offset);

  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }

  O grad_offset = grad_outer_offset +
      static_cast<O>(feature_idx) * static_cast<O>(params.grad_feature_stride);
  float grad_val = static_cast<float>(grad[grad_offset]);
  if (params.scale_grad_by_freq) {
    uint c = counts[weight_idx];
    if (c == 0) {
      return;
    }
    grad_val /= static_cast<float>(c);
  }

  AtomicType<float>::atomic_add(
      grad_weight,
      static_cast<long>(weight_idx) * feature_size + feature_idx,
      grad_val);
}

#define REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(I, O, OSUFFIX)                \
  template [[host_name("embedding_dense_backward_count_" #I "_" #OSUFFIX)]]   \
  kernel void embedding_dense_backward_count<I, O>(                           \
      constant I * indices [[buffer(0)]],                                     \
      device atomic_uint * counts [[buffer(1)]],                              \
      constant EmbeddingDenseBackwardParams<uint32_t> & params [[buffer(2)]], \
      uint tid [[thread_position_in_grid]])

#define REGISTER_EMBEDDING_DENSE_BACKWARD(T, I, O, OSUFFIX)                   \
  template [[host_name("embedding_dense_backward_" #T "_" #I "_" #OSUFFIX)]]  \
  kernel void embedding_dense_backward<T, I, O>(                              \
      constant T * grad [[buffer(0)]],                                        \
      constant I * indices [[buffer(1)]],                                     \
      constant uint * counts [[buffer(2)]],                                   \
      device AtomicType_t<float> * grad_weight [[buffer(3)]],                 \
      constant EmbeddingDenseBackwardParams<uint32_t> & params [[buffer(4)]], \
      uint tid [[thread_position_in_grid]])

REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, ulong, 64);

REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, ulong, 64);

// Embedding forward: gathers `weight[indices[i]]` rows into the output.
// One thread moves up to 16 bytes of a row, so neighboring lanes of a
// threadgroup touch consecutive addresses on both sides. The row is treated as
// raw bytes, which lets a single kernel serve every storage type.
template <typename I>
kernel void embedding_forward(
    constant uchar* weight [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    constant uint& num_embeddings [[buffer(3)]],
    constant uint& row_bytes [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint pos = tid.x * 16u;
  if (pos >= row_bytes) {
    return;
  }
  const long ix_raw = static_cast<long>(indices[tid.y]);
  const long ix = ix_raw < 0 ? ix_raw + long(num_embeddings) : ix_raw;
  const ulong src = static_cast<ulong>(ix) * row_bytes + pos;
  const ulong dst = static_cast<ulong>(tid.y) * row_bytes + pos;
  copy_bytes_aligned(out + dst, weight + src, min(16u, row_bytes - pos));
}

#define REGISTER_EMBEDDING_FORWARD(I)                                  \
  template [[host_name("embedding_forward_" #I)]] kernel void          \
  embedding_forward<I>(                                                \
      constant uchar * weight [[buffer(0)]],                           \
      constant I * indices [[buffer(1)]],                              \
      device uchar * out [[buffer(2)]],                                \
      constant uint& num_embeddings [[buffer(3)]],                     \
      constant uint& row_bytes [[buffer(4)]],                          \
      uint2 tid [[thread_position_in_grid]])

REGISTER_EMBEDDING_FORWARD(int);
REGISTER_EMBEDDING_FORWARD(long);