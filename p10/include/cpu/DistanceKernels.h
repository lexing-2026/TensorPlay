#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512 and the per-architecture vector tiers)
// p-norm distance kernels.  The implementations live in
// DistanceKernelsImpl.cpp, which is listed in TP_CPU_KERNEL_SRCS so each CPU
// capability tier gets its own copy of the parallel vectorized loops (see
// p10/CMakeLists.txt); the pdist/cdist entry points in DistanceKernels.cpp
// stay in the base tier and hand the ready-to-run buffers to these stubs.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Norm selector shared by both stubs:
//   0  p == 0     count of differing lanes (hamming-like)
//   1  p == 1     sum of absolute differences
//   2  p == 2     euclidean norm of the difference
//   3  p == inf   maximum absolute difference
//   4  otherwise  generalized p-norm of the difference
// dtype is the DType enumerator cast to int (encoding shared with the other
// raw-pointer stubs); only floating-point widths are accepted.

// pdist: `data` holds the contiguous (n, width) input in the working dtype;
// `out` receives the n*(n-1)/2 condensed upper-triangle distances.
using pdist_fn = void (*)(const void* data, void* out, int64_t n,
                          int64_t width, double p, int mode, int dtype);

// cdist: `lhs`/`rhs` hold the contiguous (batches, rows, width) inputs in the
// working dtype; `out` receives batches*rows1*rows2 distances, ordered batch
// major, then row1, then row2.
using cdist_fn = void (*)(const void* lhs, const void* rhs, void* out,
                          int64_t batches, int64_t rows1, int64_t rows2,
                          int64_t width, double p, int mode, int dtype);

DECLARE_DISPATCH(pdist_fn, pdist_stub)
DECLARE_DISPATCH(cdist_fn, cdist_stub)

} // namespace cpu
} // namespace tensorplay
