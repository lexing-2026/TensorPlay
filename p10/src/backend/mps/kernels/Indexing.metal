#include <metal_stdlib>
#include "common.h"
#include "atomic.h"

using namespace metal;
using namespace p10::metal;

// Dense index_select/gather/scatter/scatter_add kernels.
//
// All kernels operate on contiguous storage laid out as [outer, dim, inner],
// where `dim` is the indexed dimension. `outer` and `inner` are the products
// of the leading and trailing dimensions respectively. One thread is launched
// per output/index element and recovers (inner, dim-position, outer) with two
// div/mod operations, so the hot loop is branch-free.
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
    uint tid [[thread_position_in_grid]]) {
  const long x = long(tid) % inner;
  long t = long(tid) / inner;
  const long j = t % n_idx;
  const long o = t / n_idx;
  const long out_off = (o * n_idx + j) * inner + x;
  const long in_off = (o * in_dim + index[j]) * inner + x;
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
    uint tid [[thread_position_in_grid]]) {
  const long x = long(tid) % inner;
  long t = long(tid) / inner;
  const long c = t % idx_dim;
  const long o = t / idx_dim;
  const long idx = index[tid];
  const long in_off = (o * in_dim + idx) * inner + x;
  output[tid] = input[in_off];
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
    uint tid [[thread_position_in_grid]]) {
  const long x = long(tid) % inner;
  long t = long(tid) / inner;
  const long c = t % idx_dim;
  const long o = t / idx_dim;
  const long idx = index[tid];
  const long out_off = (o * out_dim + idx) * inner + x;
  output[out_off] = src[tid];
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
    uint tid [[thread_position_in_grid]]) {
  const long x = long(tid) % inner;
  long t = long(tid) / inner;
  const long c = t % idx_dim;
  const long o = t / idx_dim;
  const long idx = index[tid];
  const long out_off = (o * out_dim + idx) * inner + x;
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
    uint tid [[thread_position_in_grid]]) {
  const long x = long(tid) % inner;
  long t = long(tid) / inner;
  const long c = t % idx_dim;
  const long o = t / idx_dim;
  const long idx = index[tid];
  const long out_off = (o * out_dim + idx) * inner + x;
  AtomicType<T>::atomic_add(
      reinterpret_cast<device AtomicType_t<T>*>(output), out_off, src[tid]);
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
      uint tid [[thread_position_in_grid]])

#define REGISTER_GATHER_DTYPE(DTYPE)                                         \
  template [[host_name("gather_dense_" #DTYPE "_long")]] kernel void         \
  gather_dense<DTYPE, long>(                                                 \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * input [[buffer(1)]],                                   \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & in_dim [[buffer(4)]],                                   \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint tid [[thread_position_in_grid]])

#define REGISTER_SCATTER_SET_DTYPE(DTYPE)                                    \
  template [[host_name("scatter_set_dense_" #DTYPE "_long")]] kernel void    \
  scatter_set_dense<DTYPE, long>(                                            \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * src [[buffer(1)]],                                     \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint tid [[thread_position_in_grid]]);                                  \
  template [[host_name("scatter_set_dense_value_" #DTYPE "_long")]]          \
  kernel void scatter_set_dense_value<DTYPE, long>(                           \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE & value [[buffer(1)]],                                   \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint tid [[thread_position_in_grid]])

#define REGISTER_SCATTER_ADD_DTYPE(DTYPE)                                    \
  template [[host_name("scatter_add_dense_" #DTYPE "_long")]] kernel void    \
  scatter_add_dense<DTYPE, long>(                                            \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * src [[buffer(1)]],                                     \
      constant long * index [[buffer(2)]],                                    \
      constant long & inner [[buffer(3)]],                                    \
      constant long & out_dim [[buffer(4)]],                                  \
      constant long & idx_dim [[buffer(5)]],                                  \
      uint tid [[thread_position_in_grid]])

#define REGISTER_INDEXING_DTYPE(DTYPE) \
  REGISTER_INDEX_SELECT_DTYPE(DTYPE)  \
  REGISTER_GATHER_DTYPE(DTYPE)        \
  REGISTER_SCATTER_SET_DTYPE(DTYPE)   \
  REGISTER_SCATTER_ADD_DTYPE(DTYPE)

REGISTER_INDEXING_DTYPE(float)
REGISTER_INDEXING_DTYPE(half)
REGISTER_INDEXING_DTYPE(bfloat)
REGISTER_INDEXING_DTYPE(long)
REGISTER_INDEXING_DTYPE(int)
REGISTER_INDEXING_DTYPE(short)
REGISTER_INDEXING_DTYPE(char)
REGISTER_INDEXING_DTYPE(uchar)
REGISTER_INDEXING_DTYPE(bool)