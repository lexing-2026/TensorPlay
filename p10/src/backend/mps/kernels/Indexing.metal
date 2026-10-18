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
// A bounds-check kernel is encoded ahead of the gather/scatter in the same
// command buffer and reports a bad entry into the stream's error buffer; the
// wait that drains the work raises it. Until it does, the access kernels clamp
// their index so an out-of-range entry cannot address outside the storage.

// Wraps a negative entry and clamps anything still outside the dimension, so a
// reported entry cannot make the access kernels read or write out of bounds.
static inline ulong wrapped_index(long raw, long dim) {
  long idx = raw < 0 ? raw + dim : raw;
  const long top = dim > 0 ? dim - 1 : 0;
  idx = idx < 0 ? 0 : (idx >= dim ? top : idx);
  return static_cast<ulong>(idx);
}

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
  const ulong j = wrapped_index(static_cast<long>(index[tid.y]), in_dim);
  const ulong o = tid.z;
  const ulong out_off = (o * n_idx + tid.y) * inner + x;
  const ulong in_off = (o * in_dim + j) * inner + x;
  output[out_off] = input[in_off];
}

// Byte-wise form of index_select: one thread moves a unit of 16, 8 or 4 bytes
// through the storage, so a row takes a fraction of the threads the element
// form needs. The host picks the widest unit that divides the row span and
// both base addresses.
template <typename IT, typename Unit>
kernel void index_select_dense_vec(
    device uchar* output [[buffer(0)]],
    const device uchar* input [[buffer(1)]],
    constant IT* index [[buffer(2)]],
    constant long& inner_units [[buffer(3)]],
    constant long& in_dim [[buffer(4)]],
    constant long& n_idx [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]) {
  const ulong x = tid.x;
  const ulong j = wrapped_index(static_cast<long>(index[tid.y]), in_dim);
  const ulong o = tid.z;
  const ulong out_off = (o * n_idx + tid.y) * inner_units + x;
  const ulong in_off = (o * in_dim + j) * inner_units + x;
  reinterpret_cast<device Unit*>(output)[out_off] =
      reinterpret_cast<const device Unit*>(input)[in_off];
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
  const ulong idx = wrapped_index(static_cast<long>(index[pos]), in_dim);
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
  const ulong idx = wrapped_index(static_cast<long>(index[pos]), out_dim);
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
  const ulong idx = wrapped_index(static_cast<long>(index[pos]), out_dim);
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
  const ulong idx = wrapped_index(static_cast<long>(index[pos]), out_dim);
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
// nonzero, `out[1]` holds the offending value and `out[2]` the dimension size;
// the report is raised by the next wait that drains the work. Any offending
// entry is representative, so the stores need no ordering between threads.
template <typename IT>
kernel void index_bounds_check(
    constant IT* index [[buffer(0)]],
    constant long& dim_size [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const long raw = static_cast<long>(index[tid]);
  if (raw < -dim_size || raw >= dim_size) {
    out[1] = raw;
    out[2] = dim_size;
    out[0] = 1;
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

#define REGISTER_INDEX_SELECT_VEC(NAME, UNIT)                                \
  template [[host_name("index_select_vec" #NAME "_long")]] kernel void       \
  index_select_dense_vec<long, UNIT>(                                        \
      device uchar * output [[buffer(0)]],                                   \
      const device uchar * input [[buffer(1)]],                              \
      constant long * index [[buffer(2)]],                                   \
      constant long & inner_units [[buffer(3)]],                             \
      constant long & in_dim [[buffer(4)]],                                  \
      constant long & n_idx [[buffer(5)]],                                   \
      uint3 tid [[thread_position_in_grid]]);

REGISTER_INDEX_SELECT_VEC(16, uint4)
REGISTER_INDEX_SELECT_VEC(8, uint2)
REGISTER_INDEX_SELECT_VEC(4, uint)

REGISTER_INDEXING_DTYPE(float)
REGISTER_INDEXING_DTYPE(half)
REGISTER_INDEXING_DTYPE(bfloat)
REGISTER_INDEXING_DTYPE(long)
REGISTER_INDEXING_DTYPE(int)
REGISTER_INDEXING_DTYPE(short)
REGISTER_INDEXING_DTYPE(char)
REGISTER_INDEXING_DTYPE(uchar)
REGISTER_INDEXING_DTYPE(bool)