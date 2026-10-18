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

struct neg_op {
  // Multiplied by minus one rather than subtracted from zero, so that a zero
  // comes back with its sign reversed the way the operation asks.
  template <typename T>
  static inline T apply(T x) { return T(-1) * x; }
};

struct square_op {
  template <typename T>
  static inline T apply(T x) { return x * x; }
};

struct reciprocal_op {
  // A division by the operand, so an operand of zero answers with an infinity
  // of that zero's sign.
  template <typename T>
  static inline T apply(T x) { return T(1) / x; }
};

struct log_op {
  template <typename T>
  static inline T apply(T x) { return ::precise::log(x); }
};

struct sin_op {
  template <typename T>
  static inline T apply(T x) { return ::precise::sin(x); }
};

struct cos_op {
  template <typename T>
  static inline T apply(T x) { return ::precise::cos(x); }
};

struct tanh_op {
  template <typename T>
  static inline T apply(T x) { return ::precise::tanh(x); }
};

struct floor_op {
  template <typename T>
  static inline T apply(T x) { return floor(x); }
};

struct ceil_op {
  template <typename T>
  static inline T apply(T x) { return ceil(x); }
};

struct round_op {
  // Halfway cases go to the even neighbour, the rounding the CPU kernel uses.
  template <typename T>
  static inline T apply(T x) { return rint(x); }
};

struct sign_op {
  // Zero keeps its sign bit and a NaN stays a NaN; only a nonzero operand is
  // replaced by a signed one.
  template <typename T>
  static inline T apply(T x) {
    return x > T(0) ? T(1) : (x < T(0) ? T(-1) : x);
  }
};

struct erf_op {
  // The error function, faithfully rounded to within about one unit in the
  // last place. Above the branch point the answer is 1 - exp(-q(|x|)) for a
  // degree-6 polynomial q; below it a degree-6 polynomial in x*x, scaled by x.
  template <typename T>
  static inline T apply(T x) {
    const float a = static_cast<float>(x);
    const float t = ::metal::abs(a);
    const float s = a * a;
    if (t > 0.927734375f) {
      float r = ::metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
      const float u = ::metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
      r = ::metal::fma(r, s, u);
      r = ::metal::fma(r, t, -1.06777877e-1f);
      r = ::metal::fma(r, t, -6.34846687e-1f);
      r = ::metal::fma(r, t, -1.28717512e-1f);
      r = ::metal::fma(r, t, -t);
      r = 1.0f - ::metal::exp(r);
      return static_cast<T>(::metal::copysign(r, a));
    }
    float r = -5.96761703e-4f;
    r = ::metal::fma(r, s, 4.99119423e-3f);
    r = ::metal::fma(r, s, -2.67681349e-2f);
    r = ::metal::fma(r, s, 1.12819925e-1f);
    r = ::metal::fma(r, s, -3.76125336e-1f);
    r = ::metal::fma(r, s, 1.28379166e-1f);
    return static_cast<T>(::metal::fma(r, a, a));
  }
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

// The same run over four-wide vectors. Without an alignment the compiler has
// to assume the base pointer may be arbitrary, so the scalar form above costs
// four separate 4-byte accesses each way even though a thread's run is
// contiguous. Stating the vector type turns each run into one 16-byte load and
// one 16-byte store while the bytes a thread keeps in flight stay the same.
// The host only takes this path when both buffers are 16-byte aligned and the
// operand holds whole vectors.
template <typename Op>
inline float4 apply_unary4(float4 v) {
  return float4(Op::apply(v.x), Op::apply(v.y), Op::apply(v.z),
                Op::apply(v.w));
}

template <typename Op>
kernel void unary_elementwise_vec(device float4* out [[buffer(0)]],
                                  constant float4* src [[buffer(1)]],
                                  constant uint& nvec [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= nvec) {
    array<float4, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = src[first + j];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = apply_unary4<Op>(values[j]);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = values[j];
    }
  } else {
    for (uint i = first; i < nvec; ++i) {
      out[i] = apply_unary4<Op>(src[i]);
    }
  }
}

#define REGISTER_UNARY_ELEMENTWISE_VEC(OP)           \
  template [[host_name("unary_vec_" #OP "_float")]]  \
  kernel void unary_elementwise_vec<OP##_op>(        \
      device float4* out [[buffer(0)]],              \
      constant float4* src [[buffer(1)]],            \
      constant uint& nvec [[buffer(2)]],             \
      uint tid [[thread_position_in_grid]]);

// Clamps a contiguous operand into [lo, hi]. The lower comparison is applied
// first and the upper one to its result, so a NaN operand stays a NaN and a
// lower bound above the upper bound answers with the upper bound, which is what
// the operation asks for. An absent bound arrives as the infinity of its sign.
template <typename T>
kernel void clamp_elementwise(device T* out [[buffer(0)]],
                              constant T* src [[buffer(1)]],
                              constant uint& numel [[buffer(2)]],
                              constant T& lo [[buffer(3)]],
                              constant T& hi [[buffer(4)]],
                              uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= numel) {
    array<T, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      T x = src[first + j];
      T y = x < lo ? lo : x;
      values[j] = y > hi ? hi : y;
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = values[j];
    }
  } else {
    for (uint i = first; i < numel; ++i) {
      T x = src[i];
      T y = x < lo ? lo : x;
      out[i] = y > hi ? hi : y;
    }
  }
}

template [[host_name("clamp_elementwise_float")]]
kernel void clamp_elementwise<float>(device float* out [[buffer(0)]],
                                     constant float* src [[buffer(1)]],
                                     constant uint& numel [[buffer(2)]],
                                     constant float& lo [[buffer(3)]],
                                     constant float& hi [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]);

// Four-wide clamp: the same lower-then-upper order on each component, with the
// run of four moved by one 16-byte access each way.
inline float4 clamp4(float4 v, float lo, float hi) {
  const float x = v.x < lo ? lo : v.x;
  const float y = v.y < lo ? lo : v.y;
  const float z = v.z < lo ? lo : v.z;
  const float w = v.w < lo ? lo : v.w;
  return float4(x > hi ? hi : x, y > hi ? hi : y, z > hi ? hi : z,
                w > hi ? hi : w);
}

kernel void clamp_elementwise_vec_float(device float4* out [[buffer(0)]],
                                        constant float4* src [[buffer(1)]],
                                        constant uint& nvec [[buffer(2)]],
                                        constant float& lo [[buffer(3)]],
                                        constant float& hi [[buffer(4)]],
                                        uint tid [[thread_position_in_grid]]) {
  const uint first = tid * ILP_PER_THREAD;
  if (first + ILP_PER_THREAD <= nvec) {
    array<float4, ILP_PER_THREAD> values;
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = src[first + j];
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      values[j] = clamp4(values[j], lo, hi);
    }
#pragma unroll
    for (uint j = 0; j < ILP_PER_THREAD; ++j) {
      out[first + j] = values[j];
    }
  } else {
    for (uint i = first; i < nvec; ++i) {
      out[i] = clamp4(src[i], lo, hi);
    }
  }
}

// Half, bfloat16 and the integer storage types are absent: an operation with
// one of them runs its math in a wider type, which the graph path builds as a
// cast around the operation. A call with such a dtype stays with the graph,
// which the host reaches by failing to resolve the kernel name.
REGISTER_UNARY_ELEMENTWISE(exp, float)
REGISTER_UNARY_ELEMENTWISE(sqrt, float)
REGISTER_UNARY_ELEMENTWISE(rsqrt, float)
REGISTER_UNARY_ELEMENTWISE(abs, float)
REGISTER_UNARY_ELEMENTWISE(relu, float)
REGISTER_UNARY_ELEMENTWISE(neg, float)
REGISTER_UNARY_ELEMENTWISE(square, float)
REGISTER_UNARY_ELEMENTWISE(reciprocal, float)
REGISTER_UNARY_ELEMENTWISE(log, float)
REGISTER_UNARY_ELEMENTWISE(sin, float)
REGISTER_UNARY_ELEMENTWISE(cos, float)
REGISTER_UNARY_ELEMENTWISE(tanh, float)
REGISTER_UNARY_ELEMENTWISE(floor, float)
REGISTER_UNARY_ELEMENTWISE(ceil, float)
REGISTER_UNARY_ELEMENTWISE(round, float)
REGISTER_UNARY_ELEMENTWISE(sign, float)
REGISTER_UNARY_ELEMENTWISE(erf, float)

REGISTER_UNARY_ELEMENTWISE_VEC(exp)
REGISTER_UNARY_ELEMENTWISE_VEC(sqrt)
REGISTER_UNARY_ELEMENTWISE_VEC(rsqrt)
REGISTER_UNARY_ELEMENTWISE_VEC(abs)
REGISTER_UNARY_ELEMENTWISE_VEC(relu)
REGISTER_UNARY_ELEMENTWISE_VEC(neg)
REGISTER_UNARY_ELEMENTWISE_VEC(square)
REGISTER_UNARY_ELEMENTWISE_VEC(reciprocal)
REGISTER_UNARY_ELEMENTWISE_VEC(log)
REGISTER_UNARY_ELEMENTWISE_VEC(sin)
REGISTER_UNARY_ELEMENTWISE_VEC(cos)
REGISTER_UNARY_ELEMENTWISE_VEC(tanh)
REGISTER_UNARY_ELEMENTWISE_VEC(floor)
REGISTER_UNARY_ELEMENTWISE_VEC(ceil)
REGISTER_UNARY_ELEMENTWISE_VEC(round)
REGISTER_UNARY_ELEMENTWISE_VEC(sign)
REGISTER_UNARY_ELEMENTWISE_VEC(erf)
