#pragma once

// Tier-compiled chunk walk for the selection maps behind clamp_write and
// maximum_cpu / minimum_cpu.
//
// On a GCC/Clang build these three answer a packed f32/f64 pair through
// vecbinary::* chunk kernels or clamp_f32_avx*/clamp_f64_avx*, whose bodies
// carry per-function ISA attributes.  MSVC has no such attribute, so those
// bodies never compile there and the maps fall into the flat per-element loops
// (each carrying a NaN test per element): 1M maximum takes 0.85 ms on an
// 8-thread box against 0.034 ms for the reference runtime.
//
// The implementations live in SelectVecKernelsImpl.cpp, listed in
// TP_CPU_KERNEL_SRCS: each CPU capability tier compiles the same body at its
// own /arch level, `vec::Vectorized<T>` therefore resolves to that tier's
// width, and DispatchStub picks the widest copy the host supports.
//
// `op`: 0 = clamp (a, bounds p0/p1 -> dst), 1 = maximum(a, b), 2 = minimum(a, b).
// A null `b` is only legal for op 0.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using select_vec_fn = void (*)(const void* a, const void* b, void* dst,
                               int64_t n, int op, double p0, double p1,
                               int dtype);

DECLARE_DISPATCH(select_vec_fn, select_vec_stub)

} // namespace cpu
} // namespace tensorplay
