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

using namespace metal;

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

REGISTER_BROADCAST_WHERE(float)
REGISTER_BROADCAST_WHERE(long)
REGISTER_BROADCAST_WHERE(int)
