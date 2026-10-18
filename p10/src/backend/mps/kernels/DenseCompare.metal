// Unit-stride comparison kernels for operands that already share a shape.
//
// The scalar form answers one element per thread: it reads both operands at its
// own index and writes the predicate into the single byte that a boolean
// element occupies, so a threadgroup is served by two wide accesses and the
// whole call is one command over three buffers rather than a graph execution
// with its feeds and results. The four-wide form below reads one 16-byte vector
// of each operand per predicate group instead.
//
// The comparisons follow the ordering of the types they read: two floats are
// equal only when neither is NaN, and every ordered comparison against a NaN is
// false.

#include <metal_stdlib>

#include "common.h"

using namespace metal;
using p10::metal::ILP_PER_THREAD;

struct eq_op {
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return lhs == rhs;
  }
};

struct ne_op {
  // The negation of the equality above rather than a comparison against a
  // different value, so that a NaN answers that it is not a NaN.
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return !(lhs == rhs);
  }
};

struct lt_op {
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return lhs < rhs;
  }
};

struct le_op {
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return lhs <= rhs;
  }
};

struct gt_op {
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return lhs > rhs;
  }
};

struct ge_op {
  template <typename T>
  static inline bool apply(T lhs, T rhs) {
    return lhs >= rhs;
  }
};

template <typename T, typename Op>
kernel void compare_dense(device bool* out [[buffer(0)]],
                          constant T* lhs [[buffer(1)]],
                          constant T* rhs [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::apply(lhs[tid], rhs[tid]);
}

// A comparison against one scalar value reads that value from the single
// element of its buffer and applies it to every operand element, which keeps
// the operand on the unit-stride path.
template <typename T, typename Op>
kernel void compare_scalar_dense(device bool* out [[buffer(0)]],
                                 constant T* lhs [[buffer(1)]],
                                 constant T* rhs [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::apply(lhs[tid], rhs[0]);
}

// Four-wide form: a thread reads four consecutive floats of each operand as one
// 16-byte access and writes the four predicate bytes as one 4-byte store. The
// host only takes this path when the operands hold whole vectors, every base is
// 16-byte aligned, and the storage type is float; anything else stays with the
// scalar forms above.
//
// Each thread takes one vector per grid-wide stride rather than a run of
// consecutive vectors, so the lanes of a SIMD group address adjacent vectors on
// every access and one instruction covers a contiguous span.
template <typename Op>
inline uchar4 apply_compare4(float4 lhs, float4 rhs) {
  return uchar4(static_cast<uchar>(Op::apply(lhs.x, rhs.x)),
                static_cast<uchar>(Op::apply(lhs.y, rhs.y)),
                static_cast<uchar>(Op::apply(lhs.z, rhs.z)),
                static_cast<uchar>(Op::apply(lhs.w, rhs.w)));
}

template <typename Op>
kernel void compare_vec(device uchar4* out [[buffer(0)]],
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
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      right[j] = rhs[tid + j * grid];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[tid + j * grid] = apply_compare4<Op>(left[j], right[j]);
    }
  } else {
    // The threads whose last stride lands past the end, and the whole grid,
    // when the operand holds fewer vectors than it has threads.
    for (uint i = tid; i < nvec; i += grid) {
      out[i] = apply_compare4<Op>(lhs[i], rhs[i]);
    }
  }
}

// The scalar operand is read as one float and broadcast to all four lanes, so
// the read stays within the single element its buffer holds.
template <typename Op>
kernel void compare_scalar_vec(device uchar4* out [[buffer(0)]],
                               constant float4* lhs [[buffer(1)]],
                               constant float* rhs [[buffer(2)]],
                               constant uint& nvec [[buffer(3)]],
                               uint tid [[thread_position_in_grid]],
                               uint grid [[threads_per_grid]]) {
  const float4 scalar = float4(rhs[0]);
  if (tid + (ILP_PER_THREAD - 1) * grid < nvec) {
    array<float4, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = lhs[tid + j * grid];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[tid + j * grid] = apply_compare4<Op>(values[j], scalar);
    }
  } else {
    for (uint i = tid; i < nvec; i += grid) {
      out[i] = apply_compare4<Op>(lhs[i], scalar);
    }
  }
}

#define REGISTER_COMPARE_DENSE(OP, DTYPE)                  \
  template [[host_name("compare_" #OP "_" #DTYPE)]]        \
  kernel void compare_dense<DTYPE, OP##_op>(               \
      device bool* out [[buffer(0)]],                      \
      constant DTYPE* lhs [[buffer(1)]],                   \
      constant DTYPE* rhs [[buffer(2)]],                   \
      uint tid [[thread_position_in_grid]]);

// Only these three storage types are instantiated. A call over half, bfloat16
// or a narrow integer is answered by the graph instead, which the host reaches
// by failing to resolve the kernel name.
#define REGISTER_COMPARE_DTYPES(OP)       \
  REGISTER_COMPARE_DENSE(OP, float)       \
  REGISTER_COMPARE_DENSE(OP, long)        \
  REGISTER_COMPARE_DENSE(OP, int)

#define REGISTER_COMPARE_SCALAR(OP, DTYPE)                 \
  template [[host_name("compare_" #OP "_scalar_" #DTYPE)]] \
  kernel void compare_scalar_dense<DTYPE, OP##_op>(        \
      device bool* out [[buffer(0)]],                      \
      constant DTYPE* lhs [[buffer(1)]],                   \
      constant DTYPE* rhs [[buffer(2)]],                   \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_COMPARE_SCALAR_DTYPES(OP) \
  REGISTER_COMPARE_SCALAR(OP, float)       \
  REGISTER_COMPARE_SCALAR(OP, long)        \
  REGISTER_COMPARE_SCALAR(OP, int)

#define REGISTER_COMPARE_VEC(OP)                      \
  template [[host_name("compare_vec_" #OP "_float")]] \
  kernel void compare_vec<OP##_op>(                   \
      device uchar4* out [[buffer(0)]],               \
      constant float4* lhs [[buffer(1)]],             \
      constant float4* rhs [[buffer(2)]],             \
      constant uint& nvec [[buffer(3)]],              \
      uint tid [[thread_position_in_grid]],           \
      uint grid [[threads_per_grid]]);

#define REGISTER_COMPARE_VEC_SCALAR(OP)                      \
  template [[host_name("compare_vec_" #OP "_scalar_float")]] \
  kernel void compare_scalar_vec<OP##_op>(                   \
      device uchar4* out [[buffer(0)]],                      \
      constant float4* lhs [[buffer(1)]],                    \
      constant float* rhs [[buffer(2)]],                     \
      constant uint& nvec [[buffer(3)]],                     \
      uint tid [[thread_position_in_grid]],                  \
      uint grid [[threads_per_grid]]);

REGISTER_COMPARE_DTYPES(eq)
REGISTER_COMPARE_DTYPES(ne)
REGISTER_COMPARE_DTYPES(lt)
REGISTER_COMPARE_DTYPES(le)
REGISTER_COMPARE_DTYPES(gt)
REGISTER_COMPARE_DTYPES(ge)

REGISTER_COMPARE_SCALAR_DTYPES(eq)
REGISTER_COMPARE_SCALAR_DTYPES(ne)
REGISTER_COMPARE_SCALAR_DTYPES(lt)
REGISTER_COMPARE_SCALAR_DTYPES(le)
REGISTER_COMPARE_SCALAR_DTYPES(gt)
REGISTER_COMPARE_SCALAR_DTYPES(ge)

REGISTER_COMPARE_VEC(eq)
REGISTER_COMPARE_VEC(ne)
REGISTER_COMPARE_VEC(lt)
REGISTER_COMPARE_VEC(le)
REGISTER_COMPARE_VEC(gt)
REGISTER_COMPARE_VEC(ge)

REGISTER_COMPARE_VEC_SCALAR(eq)
REGISTER_COMPARE_VEC_SCALAR(ne)
REGISTER_COMPARE_VEC_SCALAR(lt)
REGISTER_COMPARE_VEC_SCALAR(le)
REGISTER_COMPARE_VEC_SCALAR(gt)
REGISTER_COMPARE_VEC_SCALAR(ge)
