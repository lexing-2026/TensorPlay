// Single-dispatch elementwise kernels over a contiguous unary operand.
//
// A thread takes ILP_PER_THREAD consecutive elements: it reads the whole run,
// applies the operation to each, then writes the run back. Adjacent threads
// still touch adjacent storage, so one wide access serves a threadgroup, but a
// thread has several loads in flight at once instead of one.

#include <metal_stdlib>

#include "common.h"

using namespace metal;
using p10::metal::ILP_PER_THREAD;

struct exp_op {
  // The precise spelling, not the plain one, which a shorter approximation may
  // answer.
  template <typename T>
  static inline T apply(T x) { return ::precise::exp(x); }
};

struct sqrt_op {
  template <typename T>
  static inline T apply(T x) { return ::precise::sqrt(x); }
};

struct rsqrt_op {
  // A reciprocal of the square root rather than the rsqrt builtin, whose result
  // is implementation-defined: x == 0 must give +inf and x < 0 must give NaN,
  // the way a division by the square root does.
  template <typename T>
  static inline T apply(T x) { return T(1) / ::precise::sqrt(x); }
};

struct abs_op {
  template <typename T>
  static inline T apply(T x) { return ::abs(x); }
};

struct relu_op {
  // Clamped below by a comparison against zero: only an operand less than zero
  // becomes zero, so a NaN reads back as a NaN.
  template <typename T>
  static inline T apply(T x) { return x < T(0) ? T(0) : x; }
};

template <typename T, typename Op>
kernel void unary_elementwise(device T* out [[buffer(0)]],
                              constant T* src [[buffer(1)]],
                              constant uint& numel [[buffer(2)]],
                              uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= numel) {
    array<T, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = src[first + j];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = Op::apply(values[j]);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = values[j];
    }
  } else {
    // The last thread of the grid, and the whole grid, when the operand is
    // shorter than one run.
    for (uint i = first; i < numel; ++i) {
      out[i] = Op::apply(src[i]);
    }
  }
}

#define REGISTER_UNARY_ELEMENTWISE(OP, DTYPE)     \
  template [[host_name("unary_" #OP "_" #DTYPE)]] \
  kernel void unary_elementwise<DTYPE, OP##_op>(  \
      device DTYPE* out [[buffer(0)]],            \
      constant DTYPE* src [[buffer(1)]],          \
      constant uint& numel [[buffer(2)]],         \
      uint tid [[thread_position_in_grid]]);

// Half, bfloat16 and the integer storage types are absent: an operation with
// one of them runs its math in a wider type, which the graph path builds as a
// cast around the operation. A call with such a dtype stays with the graph,
// which the host reaches by failing to resolve the kernel name.
REGISTER_UNARY_ELEMENTWISE(exp, float)
REGISTER_UNARY_ELEMENTWISE(sqrt, float)
REGISTER_UNARY_ELEMENTWISE(rsqrt, float)
REGISTER_UNARY_ELEMENTWISE(abs, float)
REGISTER_UNARY_ELEMENTWISE(relu, float)
