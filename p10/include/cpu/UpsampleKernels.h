#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512 and the per-architecture vector tiers)
// channels-last upsampling cores.  UpsampleKernelsImpl.cpp is listed in
// TP_CPU_KERNEL_SRCS, so every CPU capability tier gets its own copy of the
// vectorized channel-span loops; the upsampling entry points in
// UpsampleKernels.cpp stay in the base tier and reach these through the
// DispatchStub mechanism.  The NCHW frames are untouched.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Channels-last nearest-neighbor upsampling over one NHWC buffer.  Each
// output pixel (n, oh, ow) copies its source pixel's whole channel span, so
// the span is the vectorized dimension.  `sh`/`sw` are the source-index
// scales (output/input ratio) the entry point already computed; the kernel
// narrows them back to the element type, keeping the index arithmetic
// single-sourced in the base tier.
//
// in/out are NHWC contiguous float or double buffers; both memory formats
// copy the same source values, so results are bit-identical to the NCHW
// frames.
using upsample_nearest2d_cl_fn = void (*)(const void* in, void* out,
                                          int64_t N, int64_t C,
                                          int64_t H, int64_t W,
                                          int64_t oH, int64_t oW,
                                          double sh, double sw, int dtype);

// Channels-last bilinear upsampling.  Each output pixel blends four source
// channel spans with weights derived from the horizontal/vertical source
// positions.  `rh`/`rw` are the area-pixel scales (widened to double) the
// entry point computed; the kernel derives the per-pixel indices, clamps and
// lambdas from them with the same element-type arithmetic as the NCHW frame
// and evaluates the blend in the same association order, so both memory
// formats round identically.
using upsample_bilinear2d_cl_fn = void (*)(const void* in, void* out,
                                           int64_t N, int64_t C,
                                           int64_t H, int64_t W,
                                           int64_t oH, int64_t oW,
                                           int align_corners,
                                           double rh, double rw, int dtype);

DECLARE_DISPATCH(upsample_nearest2d_cl_fn, upsample_nearest2d_cl_stub)
DECLARE_DISPATCH(upsample_bilinear2d_cl_fn, upsample_bilinear2d_cl_stub)

} // namespace cpu
} // namespace tensorplay
