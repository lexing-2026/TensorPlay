// Stride-addressed elementwise binary kernels for broadcast operands.
//
// One thread writes one output element. The output index is split into
// per-dimension coordinates, and every coordinate is multiplied by the stride
// its operand has for that dimension. A stride of 0 stands wherever an operand
// is broadcast -- a dimension of size 1, or a dimension the operand does not
// have -- so that element is re-read along the axis instead of being stored
// again. Addressing the operands through strides keeps a broadcast input in
// the storage it already has.

#include <metal_stdlib>

#include "common.h"

using namespace metal;
using p10::metal::ILP_PER_THREAD;

// `sizes` and both stride arrays are ordered fastest-varying dimension first,
// which is the order the host builds them in.
template <typename IndexType>
inline IndexType element_offset(uint linear, constant uint* sizes,
                                constant IndexType* strides, uint ndim) {
  IndexType offset = 0;
  uint remaining = linear;
  for (uint d = 0; d < ndim; ++d) {
    uint coord = remaining % sizes[d];
    remaining /= sizes[d];
    offset += static_cast<IndexType>(coord) * strides[d];
  }
  return offset;
}

struct add_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) { return lhs + rhs; }
};

struct sub_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) { return lhs - rhs; }
};

struct mul_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) { return lhs * rhs; }
};

struct div_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) { return lhs / rhs; }
};

template <typename T, typename Op>
kernel void broadcast_binary(device T* out [[buffer(0)]],
                             device const T* lhs [[buffer(1)]],
                             device const T* rhs [[buffer(2)]],
                             constant uint& ndim [[buffer(3)]],
                             constant uint* sizes [[buffer(4)]],
                             constant int* lhs_strides [[buffer(5)]],
                             constant int* rhs_strides [[buffer(6)]],
                             uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::apply(lhs[element_offset<int>(tid, sizes, lhs_strides, ndim)],
                       rhs[element_offset<int>(tid, sizes, rhs_strides, ndim)]);
}

// The same addressing with a run of ILP_PER_THREAD consecutive elements per
// thread. The host only selects it when the innermost dimension is a multiple
// of the run width, so a run never crosses a row: the coordinates of the outer
// dimensions are formed once and the innermost coordinate advances by a
// multiple of its stride. The partition of the linear index is reused, so a
// run costs one decomposition instead of one per element.
template <typename T, typename Op>
kernel void broadcast_binary_ilp(device T* out [[buffer(0)]],
                                 device const T* lhs [[buffer(1)]],
                                 device const T* rhs [[buffer(2)]],
                                 constant uint& ndim [[buffer(3)]],
                                 constant uint* sizes [[buffer(4)]],
                                 constant int* lhs_strides [[buffer(5)]],
                                 constant int* rhs_strides [[buffer(6)]],
                                 constant uint& total [[buffer(7)]],
                                 uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  const uint inner = sizes[0];
  if (total >= ILP_PER_THREAD && first <= total - ILP_PER_THREAD) {
    uint remaining = first / inner;
    int lhs_outer = 0;
    int rhs_outer = 0;
    for (uint d = 1; d < ndim; ++d) {
      const uint coord = remaining % sizes[d];
      remaining /= sizes[d];
      lhs_outer += static_cast<int>(coord) * lhs_strides[d];
      rhs_outer += static_cast<int>(coord) * rhs_strides[d];
    }
    const uint start = first % inner;
    const int lhs_first = lhs_outer + static_cast<int>(start) * lhs_strides[0];
    const int rhs_first = rhs_outer + static_cast<int>(start) * rhs_strides[0];
    array<T, ILP_PER_THREAD> left;
    array<T, ILP_PER_THREAD> right;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = lhs[lhs_first + static_cast<int>(j) * lhs_strides[0]];
      right[j] = rhs[rhs_first + static_cast<int>(j) * rhs_strides[0]];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = Op::apply(left[j], right[j]);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = left[j];
    }
  } else {
    for (uint i = first; i < total; ++i) {
      out[i] = Op::apply(
          lhs[element_offset<int>(i, sizes, lhs_strides, ndim)],
          rhs[element_offset<int>(i, sizes, rhs_strides, ndim)]);
    }
  }
}

#define REGISTER_BROADCAST_BINARY(OP, DTYPE)                   \
  template [[host_name("broadcast_" #OP "_" #DTYPE)]]          \
  kernel void broadcast_binary<DTYPE, OP##_op>(                \
      device DTYPE* out [[buffer(0)]],                         \
      device const DTYPE* lhs [[buffer(1)]],                   \
      device const DTYPE* rhs [[buffer(2)]],                   \
      constant uint& ndim [[buffer(3)]],                       \
      constant uint* sizes [[buffer(4)]],                      \
      constant int* lhs_strides [[buffer(5)]],                 \
      constant int* rhs_strides [[buffer(6)]],                 \
      uint tid [[thread_position_in_grid]]);

// Float16 and the narrow integer types are deliberately absent: their operation
// math type and overflow behaviour would have to match the graph path exactly,
// and a call with one of them takes the graph path instead (the host looks the
// kernel name up before committing to it).
#define REGISTER_BROADCAST_DTYPES(OP)   \
  REGISTER_BROADCAST_BINARY(OP, float)  \
  REGISTER_BROADCAST_BINARY(OP, long)   \
  REGISTER_BROADCAST_BINARY(OP, int)

REGISTER_BROADCAST_DTYPES(add)
REGISTER_BROADCAST_DTYPES(sub)
REGISTER_BROADCAST_DTYPES(mul)
// Division reads integral operands as the float type they promote to before
// this kernel would see them, so only the float form is instantiated.
REGISTER_BROADCAST_BINARY(div, float)

#define REGISTER_BROADCAST_BINARY_ILP(OP, DTYPE)                   \
  template [[host_name("broadcast_" #OP "_" #DTYPE "_ilp")]]       \
  kernel void broadcast_binary_ilp<DTYPE, OP##_op>(                \
      device DTYPE * out [[buffer(0)]],                            \
      device const DTYPE * lhs [[buffer(1)]],                      \
      device const DTYPE * rhs [[buffer(2)]],                      \
      constant uint & ndim [[buffer(3)]],                          \
      constant uint * sizes [[buffer(4)]],                         \
      constant int * lhs_strides [[buffer(5)]],                    \
      constant int * rhs_strides [[buffer(6)]],                    \
      constant uint & total [[buffer(7)]],                         \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_BROADCAST_ILP_DTYPES(OP)   \
  REGISTER_BROADCAST_BINARY_ILP(OP, float)  \
  REGISTER_BROADCAST_BINARY_ILP(OP, long)   \
  REGISTER_BROADCAST_BINARY_ILP(OP, int)

REGISTER_BROADCAST_ILP_DTYPES(add)
REGISTER_BROADCAST_ILP_DTYPES(sub)
REGISTER_BROADCAST_ILP_DTYPES(mul)
REGISTER_BROADCAST_BINARY_ILP(div, float)

// Dense forms: both operands and the result already share a shape and each runs
// unit-stride, so the element index is the buffer index and no stride
// arithmetic is needed. One thread handles a run of ILP_PER_THREAD consecutive
// elements, which keeps several loads in flight per thread while adjacent
// threads still cover adjacent storage.
template <typename T, typename Op>
kernel void dense_binary(device T* out [[buffer(0)]],
                         constant T* lhs [[buffer(1)]],
                         constant T* rhs [[buffer(2)]],
                         constant uint& numel [[buffer(3)]],
                         uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= numel) {
    array<T, ILP_PER_THREAD> left;
    array<T, ILP_PER_THREAD> right;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = lhs[first + j];
      right[j] = rhs[first + j];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = Op::apply(left[j], right[j]);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = left[j];
    }
  } else {
    // The last thread of the grid, and the whole grid, when the operand is
    // shorter than one run.
    for (uint i = first; i < numel; ++i) {
      out[i] = Op::apply(lhs[i], rhs[i]);
    }
  }
}

// The scalar operand travels by value in the argument buffer, so the scalar
// forms cost the same one dispatch as the tensor forms and allocate no
// single-element tensor to hold it.
template <typename T, typename Op>
kernel void dense_scalar_binary(device T* out [[buffer(0)]],
                                constant T* lhs [[buffer(1)]],
                                constant T& rhs [[buffer(2)]],
                                constant uint& numel [[buffer(3)]],
                                uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= numel) {
    array<T, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = lhs[first + j];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = Op::apply(values[j], rhs);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = values[j];
    }
  } else {
    for (uint i = first; i < numel; ++i) {
      out[i] = Op::apply(lhs[i], rhs);
    }
  }
}

// Four-wide forms of the two dense kernels. The vector type states the
// 16-byte alignment the buffers carry on this path, so the compiler emits one
// load and one store per four elements instead of four of each. The host only
// selects these when the storage is float, the span holds whole vectors and
// every base is aligned.
//
// A thread takes one vector per grid-wide stride: the lanes of a SIMD group
// then access consecutive 16-byte vectors, so each instruction covers a
// contiguous span instead of reaching across as many cache lines as a block of
// consecutive vectors per thread would.
template <typename Op>
inline float4 apply_binary4(float4 a, float4 b) {
  return float4(Op::apply(a.x, b.x), Op::apply(a.y, b.y),
                Op::apply(a.z, b.z), Op::apply(a.w, b.w));
}

template <typename Op>
inline float4 apply_binary4(float4 a, float b) {
  return float4(Op::apply(a.x, b), Op::apply(a.y, b), Op::apply(a.z, b),
                Op::apply(a.w, b));
}

template <typename Op>
kernel void dense_binary_vec(device float4* out [[buffer(0)]],
                             constant float4* lhs [[buffer(1)]],
                             constant float4* rhs [[buffer(2)]],
                             constant uint& nvec [[buffer(3)]],
                             uint tid [[thread_position_in_grid]],
                             uint grid [[threads_per_grid]]) {
  if (tid + (ILP_PER_THREAD - 1) * grid < nvec) {
    array<float4, ILP_PER_THREAD> left;
    array<float4, ILP_PER_THREAD> right;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = lhs[tid + j * grid];
      right[j] = rhs[tid + j * grid];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      left[j] = apply_binary4<Op>(left[j], right[j]);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[tid + j * grid] = left[j];
    }
  } else {
    for (uint i = tid; i < nvec; i += grid) {
      out[i] = apply_binary4<Op>(lhs[i], rhs[i]);
    }
  }
}

template <typename Op>
kernel void dense_scalar_binary_vec(device float4* out [[buffer(0)]],
                                    constant float4* lhs [[buffer(1)]],
                                    constant float& rhs [[buffer(2)]],
                                    constant uint& nvec [[buffer(3)]],
                                    uint tid [[thread_position_in_grid]],
                                    uint grid [[threads_per_grid]]) {
  if (tid + (ILP_PER_THREAD - 1) * grid < nvec) {
    array<float4, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = lhs[tid + j * grid];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = apply_binary4<Op>(values[j], rhs);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[tid + j * grid] = values[j];
    }
  } else {
    for (uint i = tid; i < nvec; i += grid) {
      out[i] = apply_binary4<Op>(lhs[i], rhs);
    }
  }
}

#define REGISTER_DENSE_BINARY(OP, DTYPE)                       \
  template [[host_name("dense_" #OP "_" #DTYPE)]]              \
  kernel void dense_binary<DTYPE, OP##_op>(                    \
      device DTYPE * out [[buffer(0)]],                        \
      constant DTYPE * lhs [[buffer(1)]],                      \
      constant DTYPE * rhs [[buffer(2)]],                      \
      constant uint & numel [[buffer(3)]],                     \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_DENSE_SCALAR_BINARY(OP, DTYPE)                \
  template [[host_name("dense_" #OP "_scalar_" #DTYPE)]]       \
  kernel void dense_scalar_binary<DTYPE, OP##_op>(             \
      device DTYPE * out [[buffer(0)]],                        \
      constant DTYPE * lhs [[buffer(1)]],                      \
      constant DTYPE & rhs [[buffer(2)]],                      \
      constant uint & numel [[buffer(3)]],                     \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_DENSE_BINARY_VEC(OP)                 \
  template [[host_name("dense_vec_" #OP "_float")]]   \
  kernel void dense_binary_vec<OP##_op>(              \
      device float4 * out [[buffer(0)]],              \
      constant float4 * lhs [[buffer(1)]],            \
      constant float4 * rhs [[buffer(2)]],            \
      constant uint & nvec [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]],           \
      uint grid [[threads_per_grid]]);

#define REGISTER_DENSE_SCALAR_BINARY_VEC(OP)             \
  template [[host_name("dense_vec_" #OP "_scalar_float")]] \
  kernel void dense_scalar_binary_vec<OP##_op>(          \
      device float4 * out [[buffer(0)]],                 \
      constant float4 * lhs [[buffer(1)]],               \
      constant float & rhs [[buffer(2)]],                \
      constant uint & nvec [[buffer(3)]],                \
      uint tid [[thread_position_in_grid]],              \
      uint grid [[threads_per_grid]]);

// Same dtype set as the stride-addressed forms: half, bfloat16 and the narrow
// integers do their operation math in another type and stay with the graph,
// which the host reaches by failing to resolve the kernel name.
#define REGISTER_DENSE_DTYPES(OP)       \
  REGISTER_DENSE_BINARY(OP, float)      \
  REGISTER_DENSE_BINARY(OP, long)       \
  REGISTER_DENSE_BINARY(OP, int)

#define REGISTER_DENSE_SCALAR_DTYPES(OP)       \
  REGISTER_DENSE_SCALAR_BINARY(OP, float)      \
  REGISTER_DENSE_SCALAR_BINARY(OP, long)       \
  REGISTER_DENSE_SCALAR_BINARY(OP, int)

REGISTER_DENSE_DTYPES(add)
REGISTER_DENSE_DTYPES(sub)
REGISTER_DENSE_DTYPES(mul)
REGISTER_DENSE_SCALAR_DTYPES(add)
REGISTER_DENSE_SCALAR_DTYPES(sub)
REGISTER_DENSE_SCALAR_DTYPES(mul)

REGISTER_DENSE_BINARY_VEC(add)
REGISTER_DENSE_BINARY_VEC(sub)
REGISTER_DENSE_BINARY_VEC(mul)
REGISTER_DENSE_SCALAR_BINARY_VEC(add)
REGISTER_DENSE_SCALAR_BINARY_VEC(sub)
REGISTER_DENSE_SCALAR_BINARY_VEC(mul)

REGISTER_DENSE_BINARY(div, float)
REGISTER_DENSE_SCALAR_BINARY(div, float)

REGISTER_DENSE_BINARY_VEC(div)
REGISTER_DENSE_SCALAR_BINARY_VEC(div)

// maximum / minimum: a NaN on either side is the answer, and a tie takes the
// second operand. Only float is instantiated; the integral forms stay with the
// graph, whose maximum/minimum are the same ordered comparison on integers.
struct maximum_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) {
    return (isnan(lhs) || lhs > rhs) ? lhs : rhs;
  }
};

struct minimum_op {
  template <typename T>
  static inline T apply(T lhs, T rhs) {
    return (isnan(lhs) || lhs < rhs) ? lhs : rhs;
  }
};

REGISTER_DENSE_BINARY(maximum, float)
REGISTER_DENSE_BINARY(minimum, float)

REGISTER_DENSE_BINARY_VEC(maximum)
REGISTER_DENSE_BINARY_VEC(minimum)

// Ternary forms: out = self + value * lhs * rhs (addcmul) or
// self + value * lhs / rhs (addcdiv). All three operands broadcast
// independently and the scalar factor is folded into the same pass, so the
// computation costs one launch instead of the chain of binary launches.
struct addcmul_op {
  template <typename T>
  static inline T apply(T self, T lhs, T rhs, T value) {
    return self + value * lhs * rhs;
  }
};

struct addcdiv_op {
  template <typename T>
  static inline T apply(T self, T lhs, T rhs, T value) {
    return self + value * lhs / rhs;
  }
};

template <typename T, typename Op>
kernel void broadcast_ternary(
    device T* out [[buffer(0)]],
    device const T* self [[buffer(1)]],
    device const T* lhs [[buffer(2)]],
    device const T* rhs [[buffer(3)]],
    constant T& value [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint* sizes [[buffer(6)]],
    constant int* self_strides [[buffer(7)]],
    constant int* lhs_strides [[buffer(8)]],
    constant int* rhs_strides [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::apply(
      self[element_offset<int>(tid, sizes, self_strides, ndim)],
      lhs[element_offset<int>(tid, sizes, lhs_strides, ndim)],
      rhs[element_offset<int>(tid, sizes, rhs_strides, ndim)],
      value);
}

#define REGISTER_BROADCAST_TERNARY(OP, DTYPE)                        \
  template [[host_name("broadcast_" #OP "_" #DTYPE)]] kernel void    \
  broadcast_ternary<DTYPE, OP##_op>(                                 \
      device DTYPE * out [[buffer(0)]],                              \
      device const DTYPE * self [[buffer(1)]],                       \
      device const DTYPE * lhs [[buffer(2)]],                        \
      device const DTYPE * rhs [[buffer(3)]],                        \
      constant DTYPE & value [[buffer(4)]],                          \
      constant uint & ndim [[buffer(5)]],                            \
      constant uint * sizes [[buffer(6)]],                           \
      constant int * self_strides [[buffer(7)]],                     \
      constant int * lhs_strides [[buffer(8)]],                      \
      constant int * rhs_strides [[buffer(9)]],                      \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_BROADCAST_TERNARY_DTYPES(OP) \
  REGISTER_BROADCAST_TERNARY(OP, float)       \
  REGISTER_BROADCAST_TERNARY(OP, long)        \
  REGISTER_BROADCAST_TERNARY(OP, int)

REGISTER_BROADCAST_TERNARY_DTYPES(addcmul)
REGISTER_BROADCAST_TERNARY_DTYPES(addcdiv)

// where: out = condition ? lhs : rhs. The condition is a byte per element and
// each operand walks its own strides, so the mask broadcasts independently.
template <typename T>
kernel void broadcast_where(device T* out [[buffer(0)]],
                            device const bool* cond [[buffer(1)]],
                            device const T* lhs [[buffer(2)]],
                            device const T* rhs [[buffer(3)]],
                            constant uint& ndim [[buffer(4)]],
                            constant uint* sizes [[buffer(5)]],
                            constant int* cond_strides [[buffer(6)]],
                            constant int* lhs_strides [[buffer(7)]],
                            constant int* rhs_strides [[buffer(8)]],
                            uint tid [[thread_position_in_grid]]) {
  out[tid] =
      cond[element_offset<int>(tid, sizes, cond_strides, ndim)]
          ? lhs[element_offset<int>(tid, sizes, lhs_strides, ndim)]
          : rhs[element_offset<int>(tid, sizes, rhs_strides, ndim)];
}

#define REGISTER_BROADCAST_WHERE(DTYPE)                             \
  template [[host_name("broadcast_where_" #DTYPE)]] kernel void     \
  broadcast_where<DTYPE>(                                           \
      device DTYPE * out [[buffer(0)]],                             \
      device const bool * cond [[buffer(1)]],                       \
      device const DTYPE * lhs [[buffer(2)]],                       \
      device const DTYPE * rhs [[buffer(3)]],                       \
      constant uint & ndim [[buffer(4)]],                           \
      constant uint * sizes [[buffer(5)]],                          \
      constant int * cond_strides [[buffer(6)]],                    \
      constant int * lhs_strides [[buffer(7)]],                     \
      constant int * rhs_strides [[buffer(8)]],                     \
      uint tid [[thread_position_in_grid]]);

// where over three operands that already share a shape and run unit-stride:
// one thread per element, with no stride arithmetic per operand.
template <typename T>
kernel void where_dense(device T* out [[buffer(0)]],
                        constant bool* cond [[buffer(1)]],
                        constant T* lhs [[buffer(2)]],
                        constant T* rhs [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
  out[tid] = cond[tid] ? lhs[tid] : rhs[tid];
}

#define REGISTER_WHERE_DENSE(DTYPE)                             \
  template [[host_name("where_dense_" #DTYPE)]] kernel void     \
  where_dense<DTYPE>(                                           \
      device DTYPE * out [[buffer(0)]],                         \
      constant bool * cond [[buffer(1)]],                       \
      constant DTYPE * lhs [[buffer(2)]],                       \
      constant DTYPE * rhs [[buffer(3)]],                       \
      uint tid [[thread_position_in_grid]]);

REGISTER_WHERE_DENSE(float)
REGISTER_WHERE_DENSE(long)
REGISTER_WHERE_DENSE(int)

REGISTER_BROADCAST_WHERE(float)
REGISTER_BROADCAST_WHERE(long)
REGISTER_BROADCAST_WHERE(int)
