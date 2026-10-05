#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512 and the per-architecture vector tiers)
// channels-last pooling cores.  PoolingKernelsImpl.cpp is listed in
// TP_CPU_KERNEL_SRCS, so every CPU capability tier gets its own copy of the
// vectorized channel-span loops; the pooling entry points in
// PoolingKernels.cpp stay in the base tier and reach these through the
// DispatchStub mechanism.  The NCHW frames are untouched.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Channels-last forward pooling over one NHWC buffer.  Each output pixel
// (n, oh, ow) reduces its pooling window across all C channels, and the
// channel span is the vectorized dimension: averaging runs three passes
// (zero the accumulator, sum the window, divide), maximization initializes
// the lane to the max identity and blends the window in.
//
// in/out are NHWC contiguous float or double buffers.  Window bounds,
// padding clipping and divisor selection mirror the NCHW frames in
// PoolingKernels.cpp, and each channel accumulates its window in the same
// order, so both memory formats produce identical values.
//
// The max kernels also fill `ind`, a dense NCHW int64 tensor of the winning
// position's offset within each (n, c) input plane (ih * W + iw).  A NaN in
// the window becomes the maximum and carries the offset of the last NaN in
// scan order; a window with no in-bounds position keeps the -1 sentinel, the
// same contract the scalar max-pool frames follow.
//
// divisor_override follows the entry-point convention: 0 encodes "not set"
// (a caller-supplied divisor of 0 is rejected before reaching the stub).

using avg_pool2d_cl_fn = void (*)(const void* in, void* out,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t oH, int64_t oW,
                                  int64_t kH, int64_t kW,
                                  int64_t sH, int64_t sW,
                                  int64_t pH, int64_t pW,
                                  bool count_include_pad,
                                  int64_t divisor_override, int dtype);
using max_pool2d_cl_fn = void (*)(const void* in, void* out, int64_t* ind,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  int64_t oH, int64_t oW,
                                  int64_t kH, int64_t kW,
                                  int64_t sH, int64_t sW,
                                  int64_t pH, int64_t pW,
                                  int64_t dH, int64_t dW, int dtype);

// Adaptive variants derive each output pixel's window from the input/output
// extents (start = floor(i*in/out), end = ceil((i+1)*in/out)), so only the
// shapes travel through the stub.
using adaptive_avg_pool2d_cl_fn = void (*)(const void* in, void* out,
                                           int64_t N, int64_t C,
                                           int64_t H, int64_t W,
                                           int64_t oH, int64_t oW, int dtype);
using adaptive_max_pool2d_cl_fn = void (*)(const void* in, void* out,
                                           int64_t* ind,
                                           int64_t N, int64_t C,
                                           int64_t H, int64_t W,
                                           int64_t oH, int64_t oW, int dtype);

DECLARE_DISPATCH(avg_pool2d_cl_fn, avg_pool2d_cl_stub)
DECLARE_DISPATCH(max_pool2d_cl_fn, max_pool2d_cl_stub)
DECLARE_DISPATCH(adaptive_avg_pool2d_cl_fn, adaptive_avg_pool2d_cl_stub)
DECLARE_DISPATCH(adaptive_max_pool2d_cl_fn, adaptive_max_pool2d_cl_stub)

} // namespace cpu
} // namespace tensorplay
