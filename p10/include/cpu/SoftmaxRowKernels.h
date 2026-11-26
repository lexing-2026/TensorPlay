#pragma once

// Tier-compiled last-dimension softmax row walk.
//
// PointwiseKernels.cpp's innermost-softmax branch carries a vector row
// (softmax_row::row_max_f32 / row_expsum_f32 / row_scale_f32) behind
// `#if defined(__x86_64__)` with GCC target attributes, so MSVC runs the
// scalar row: one std::exp per element, which is what puts softmax at 0.24x of
// the reference runtime at 1024x1024.
//
// The implementation lives in SoftmaxRowKernelsImpl.cpp, listed in
// TP_CPU_KERNEL_SRCS, so each tier compiles the row walk at its own width.
// dtype is the DType enumerator cast to int; f32 and f64 both answer.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using softmax_row_fn = void (*)(const void* in, void* out,
                                int64_t rows, int64_t size, bool log_mode,
                                int dtype);

DECLARE_DISPATCH(softmax_row_fn, softmax_row_stub)

} // namespace cpu
} // namespace tensorplay
