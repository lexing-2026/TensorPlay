// Single-dispatch elementwise kernels over a contiguous unary operand.
//
// One thread writes one output element and reads the input at the same index,
// so the call is one command over two buffers with the same layout. Adjacent
// threads touch adjacent elements, which the hardware serves as one wide access
// per threadgroup.

#include <metal_stdlib>

using namespace metal;

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
                              device const T* src [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::apply(src[tid]);
}

#define REGISTER_UNARY_ELEMENTWISE(OP, DTYPE)     \
  template [[host_name("unary_" #OP "_" #DTYPE)]] \
  kernel void unary_elementwise<DTYPE, OP##_op>(  \
      device DTYPE* out [[buffer(0)]],            \
      device const DTYPE* src [[buffer(1)]],      \
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
