#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512 and the per-architecture vector tiers)
// bitwise kernels.  The implementations live in BitwiseKernelsImpl.cpp, which
// is listed in TP_CPU_KERNEL_SRCS so each CPU capability tier gets its own
// copy of the vectorized cores (see p10/CMakeLists.txt); the bitwise entry
// points in BitwiseKernels.cpp stay in the base tier and reach them through
// these stubs.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Operation codes shared by the bitwise stubs.  The values match the C++
// operators applied lane-wise: and/or/xor on every supported width, left and
// right shift with per-lane shift amounts.
enum class BitwiseOp : int {
    kAnd = 0,
    kOr = 1,
    kXor = 2,
    kLshift = 3,
    kRshift = 4,
};

// Dtype is the DType enumerators cast to int, mirroring the encoding used by
// the complex kernel stubs (carrying the enum type here would pull DType.h
// into every DispatchStub consumer).

using bitwise_binary_fn = void (*)(const void* lhs, const void* rhs,
                                   void* out, int64_t n, int dtype, int op);
using bitwise_scalar_fn = void (*)(const void* lhs, int64_t scalar, void* out,
                                   int64_t n, int dtype, int op);
using bitwise_not_fn = void (*)(const void* lhs, void* out, int64_t n,
                                int dtype);

DECLARE_DISPATCH(bitwise_binary_fn, bitwise_binary_stub)
DECLARE_DISPATCH(bitwise_scalar_fn, bitwise_scalar_stub)
DECLARE_DISPATCH(bitwise_not_fn, bitwise_not_stub)

} // namespace cpu
} // namespace tensorplay
