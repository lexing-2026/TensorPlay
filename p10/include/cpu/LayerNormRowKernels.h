#pragma once

// Tier-compiled row walk for layer_norm's f32 fast path.
//
// On a GCC/Clang build the row primitives live in NormRowHelpers.h behind
// `#if defined(__x86_64__)` with per-function target attributes
// (stats_f32_512 / apply_f32_512 and their AVX2 twins).  MSVC compiles none of
// them, so each row fell back to the serial float loops there: 1024x1024
// layer_norm takes 0.135 ms against 0.073 ms for the reference runtime.
//
// The implementation lives in LayerNormRowKernelsImpl.cpp, listed in
// TP_CPU_KERNEL_SRCS: every CPU capability tier compiles it at its own /arch
// level, so `vec::Vectorized<float>` resolves to that tier's width, and
// DispatchStub picks the widest copy the host supports.
//
// One call covers rows [begin, end) of a packed [outer, inner] plane.  Stats
// are written through `mean_out` / `rstd_out` when non-null, in `stats_dtype`
// (a DType value, or -1 when the caller keeps no stats).

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using lnrow_fn = void (*)(const void* src, void* dst, int64_t begin, int64_t end,
                          int64_t inner, double eps, const void* w, const void* b,
                          void* mean_out, void* rstd_out, int stats_dtype);

DECLARE_DISPATCH(lnrow_fn, lnrow_stub)

} // namespace cpu
} // namespace tensorplay
