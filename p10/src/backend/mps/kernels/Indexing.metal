#include <metal_stdlib>
#include "common.h"
#include "atomic.h"

using namespace metal;
using namespace p10::metal;

// Dense index_select/gather/scatter/scatter_add kernels.
//
// All kernels operate on contiguous storage laid out as [outer, dim, inner],
// where `dim` is the indexed dimension. The grid carries the three coordinates
// directly -- x walks `inner`, y the indexed dimension, z the leading dims --
// so no thread has to decompose a flat index.
//
// Indices are validated and wrapped on the host before dispatch, so the
// kernels read them directly without bounds checks.

// index_select: output[o, j, i] = input[o, index[j], i]
template <typename T, typename IT>
kernel void index_select_dense(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    constant long& in_dim [[buffer(4)]],
    constant long& n_idx [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const long raw = static_cast<long>(index[tid.y]);
  const ulong j = static_cast<ulong>(raw < 0 ? raw + in_dim : raw);
  const ulong o = tid.z;
  const ulong out_off = (o * n_idx + tid.y) * inner + x;
  const ulong in_off = (o * in_dim + j) * inner + x;
  output[out_off] = input[in_off];
}

// gather: output[o, c, i] = input[o, index[o, c, i], i]
template <typename T, typename IT>
kernel void gather_dense(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    constant long& in_dim [[buffer(4)]],
    constant long& idx_dim [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const ulong pos = (ulong(tid.z) * idx_dim + tid.y) * inner + x;
  const long raw = static_cast<long>(index[pos]);
  const long idx = raw < 0 ? raw + in_dim : raw;
  const ulong in_off = (ulong(tid.z) * in_dim + idx) * inner + x;
  output[pos] = input[in_off];
}

// scatter (assign): output[o, index[o, c, i], i] = src[o, c, i]
template <typename T, typename IT>
kernel void scatter_set_dense(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    constant long& out_dim [[buffer(4)]],
    constant long& idx_dim [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const ulong pos = (ulong(tid.z) * idx_dim + tid.y) * inner + x;
  const long raw = static_cast<long>(index[pos]);
  const long idx = raw < 0 ? raw + out_dim : raw;
  const ulong out_off = (ulong(tid.z) * out_dim + idx) * inner + x;
  output[out_off] = src[pos];
}

// scatter (assign) from a scalar: output[o, index[o, c, i], i] = value
template <typename T, typename IT>
kernel void scatter_set_dense_value(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    constant long& out_dim [[buffer(4)]],
    constant long& idx_dim [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const ulong pos = (ulong(tid.z) * idx_dim + tid.y) * inner + x;
  const long raw = static_cast<long>(index[pos]);
  const long idx = raw < 0 ? raw + out_dim : raw;
  const ulong out_off = (ulong(tid.z) * out_dim + idx) * inner + x;
  output[out_off] = value;
}

// scatter_add: output[o, index[o, c, i], i] += src[o, c, i]
template <typename T, typename IT>
kernel void scatter_add_dense(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner [[buffer(3)]],
    constant long& out_dim [[buffer(4)]],
    constant long& idx_dim [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const ulong pos = (ulong(tid.z) * idx_dim + tid.y) * inner + x;
  const long raw = static_cast<long>(index[pos]);
  const long idx = raw < 0 ? raw + out_dim : raw;
  const ulong out_off = (ulong(tid.z) * out_dim + idx) * inner + x;
  AtomicType<T>::atomic_add(
      reinterpret_cast<device AtomicType_t<T>*>(output), out_off, src[pos]);
}

// masked_fill: out[i] = mask[i] ? value : input[i]. The mask has the input's
// shape, so one thread handles one element.
template <typename T>
kernel void masked_fill_scalar(
    device T* out [[buffer(0)]],
    constant bool* mask [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant T& value [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = mask[tid] ? value : input[tid];
}

// Reports an index that cannot address the indexed dimension. `out[0]` becomes
// nonzero and `out[1]` holds the offending value; the host reads them after
// the launch completes and raises before any gather/scatter kernel runs.
template <typename IT>
kernel void index_bounds_check(
    constant IT* index [[buffer(0)]],
    constant long& dim_size [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const long raw = static_cast<long>(index[tid]);
  if (raw < -dim_size || raw >= dim_size) {
    out[0] = 1;
    out[1] = raw;
  }
}

#define REGISTER_INDEX_SELECT_DTYPE(DTYPE)                                   \
  template [[host_name("index_select_dense_" #DTYPE "_long")]] kernel void   \
  index_select_dense<DTYPE, long>(                                           \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * input [[buffer(1)]],                                   \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & in_dim [[buffer(4)]],                                   \
      constant long & n_idx [[buffer(5)]],                                    \
      uint3 tid [[thread_position_in_grid]]);

#define REGISTER_GATHER_DTYPE(DTYPE)                                         \
  template [[host_name("gather_dense_" #DTYPE "_long")]] kernel void         \
  gather_dense<DTYPE, long>(                                                 \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * input [[buffer(1)]],                                   \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & in_dim [[buffer(4)]],                                   \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint3 tid [[thread_position_in_grid]]);

#define REGISTER_SCATTER_SET_DTYPE(DTYPE)                                    \
  template [[host_name("scatter_set_dense_" #DTYPE "_long")]] kernel void    \
  scatter_set_dense<DTYPE, long>(                                            \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * src [[buffer(1)]],                                     \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint3 tid [[thread_position_in_grid]]);                                  \
  template [[host_name("scatter_set_dense_value_" #DTYPE "_long")]]          \
  kernel void scatter_set_dense_value<DTYPE, long>(                           \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE & value [[buffer(1)]],                                   \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint3 tid [[thread_position_in_grid]]);

#define REGISTER_SCATTER_ADD_DTYPE(DTYPE)                                    \
  template [[host_name("scatter_add_dense_" #DTYPE "_long")]] kernel void    \
  scatter_add_dense<DTYPE, long>(                                            \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * src [[buffer(1)]],                                     \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint3 tid [[thread_position_in_grid]]);

#define REGISTER_MASKED_FILL_DTYPE(DTYPE)                                    \
  template [[host_name("masked_fill_scalar_" #DTYPE)]] kernel void           \
  masked_fill_scalar<DTYPE>(                                                 \
      device DTYPE * out [[buffer(0)]],                                      \
      constant bool * mask [[buffer(1)]],                                    \
      constant DTYPE * input [[buffer(2)]],                                  \
      constant DTYPE & value [[buffer(3)]],                                  \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_INDEXING_DTYPE(DTYPE) \
  REGISTER_INDEX_SELECT_DTYPE(DTYPE)  \
  REGISTER_GATHER_DTYPE(DTYPE)        \
  REGISTER_SCATTER_SET_DTYPE(DTYPE)   \
  REGISTER_SCATTER_ADD_DTYPE(DTYPE)   \
  REGISTER_MASKED_FILL_DTYPE(DTYPE);

#define REGISTER_INDEX_BOUNDS_CHECK(IT)                          \
  template [[host_name("index_bounds_check_" #IT)]] kernel void  \
  index_bounds_check<IT>(                                        \
      constant IT * index [[buffer(0)]],                         \
      constant long & dim_size [[buffer(1)]],                    \
      device long * out [[buffer(2)]],                           \
      uint tid [[thread_position_in_grid]]);

REGISTER_INDEX_BOUNDS_CHECK(int)
REGISTER_INDEX_BOUNDS_CHECK(long)

REGISTER_INDEXING_DTYPE(float)
REGISTER_INDEXING_DTYPE(half)
REGISTER_INDEXING_DTYPE(bfloat)
REGISTER_INDEXING_DTYPE(long)
REGISTER_INDEXING_DTYPE(int)
REGISTER_INDEXING_DTYPE(short)
REGISTER_INDEXING_DTYPE(char)
REGISTER_INDEXING_DTYPE(uchar)
REGISTER_INDEXING_DTYPE(bool)