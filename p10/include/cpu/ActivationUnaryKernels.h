#pragma once

// Three-tier (DEFAULT/AVX2/AVX512) dispatch stubs for the libmvec-backed
// unary activations.  Migrated out of PointwiseKernels.cpp so these kernels
// compile per CPU_CAPABILITY like the reductions: inside an AVX512 tier copy
// vecunary's runtime capability branch is resolved at compile time and the
// 512-bit chunk kernels are selected statically.
// See p10/CMakeLists.txt Note [CPU_CAPABILITY namespace] and DispatchStub.h.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using unary_range_fn = void (*)(const float*, float*, int64_t);
DECLARE_DISPATCH(unary_range_fn, sigmoid_f32_stub)
DECLARE_DISPATCH(unary_range_fn, silu_f32_stub)

using unary_range_f64_fn = void (*)(const double*, double*, int64_t);
DECLARE_DISPATCH(unary_range_f64_fn, sigmoid_f64_stub)
DECLARE_DISPATCH(unary_range_f64_fn, silu_f64_stub)

// The backward of a softmax over the last dim: every row answers
//   r = o * (g - sum(g * o))
// and the logarithmic form answers
//   r = g - exp(o) * sum(g).
// The rows are given one after the other, each of them `cols` elements wide.
using softmax_backward_lastdim_fn = void (*)(const float*, const float*,
                                             float*, int64_t, int64_t, bool);
DECLARE_DISPATCH(softmax_backward_lastdim_fn, softmax_backward_lastdim_f32_stub)

using softmax_backward_lastdim_f64_fn = void (*)(const double*, const double*,
                                                 double*, int64_t, int64_t, bool);
DECLARE_DISPATCH(softmax_backward_lastdim_f64_fn,
                  softmax_backward_lastdim_f64_stub)

} // namespace cpu
} // namespace tensorplay
