#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512 and the per-architecture vector tiers)
// arange fill kernel.  The implementation lives in RangeKernelsImpl.cpp,
// which is listed in TP_CPU_KERNEL_SRCS so each CPU capability tier gets its
// own copy of the vectorized fill (see p10/CMakeLists.txt); the arange entry
// points in FactoryKernels.cpp stay in the base tier and reach it through
// this stub.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Fills `steps` elements with `start + i * step`, vectorized per tier.
// start/step travel as double (the accumulation type) and narrow per lane;
// dtype is the DType enumerator cast to int (encoding shared with the other
// raw-pointer stubs).
using arange_fill_fn = void (*)(void* data, int64_t steps, double start,
                                double step, int dtype);

DECLARE_DISPATCH(arange_fill_fn, arange_fill_stub)

} // namespace cpu
} // namespace tensorplay
