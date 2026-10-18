// Unit-stride comparison kernels for operands that already share a shape.
//
// One thread answers one element: it reads both operands at its own index and
// writes the predicate into the single byte that a boolean element occupies.
// Both reads step by one element, so a threadgroup is served by two wide
// accesses, and the whole call is one command over three buffers rather than a
// graph execution with its feeds and results.
//
// The comparisons follow the ordering of the types they read: two floats are
// equal only when neither is NaN, and every ordered comparison against a NaN is
// false.

#include <metal_stdlib>

using namespace metal;

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

REGISTER_COMPARE_DTYPES(eq)
REGISTER_COMPARE_DTYPES(ne)
REGISTER_COMPARE_DTYPES(lt)
REGISTER_COMPARE_DTYPES(le)
REGISTER_COMPARE_DTYPES(gt)
REGISTER_COMPARE_DTYPES(ge)
