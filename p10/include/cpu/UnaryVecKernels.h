#pragma once

// Tier-compiled chunk walk for the unary transcendental map behind
// PointwiseKernels.cpp's `unary_float_op_kernel`.
//
// On a GCC/Clang build that map is answered by `vecunary::run_f32`, whose
// bodies carry per-function ISA attributes.  MSVC has no such attribute, so
// that header's vector bodies never compile there and every element falls into
// the scalar lambda (one libm call per element: 1M exp = 0.45 ms on an 8-thread
// box against 0.18 ms for the reference runtime).
//
// The implementations live in UnaryVecKernelsImpl.cpp, listed in
// TP_CPU_KERNEL_SRCS: each CPU capability tier compiles the same body at its
// own /arch level, `vec::Vectorized<T>` therefore resolves to that tier's
// width, and DispatchStub picks the widest copy the host supports.
//
// `op` is vecunary::VOp cast to int.  A chunk returns true when it wrote
// [begin, end); false means this tier has no body for that op and the caller
// runs its scalar lambda over the chunk instead.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using unary_vec_fn = bool (*)(const void* src, void* dst,
                              int64_t begin, int64_t end, int op,
                              double p0, double p1, double p2, int dtype);

DECLARE_DISPATCH(unary_vec_fn, unary_vec_stub)

} // namespace cpu
} // namespace tensorplay
