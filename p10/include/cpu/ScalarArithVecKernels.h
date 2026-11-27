#pragma once

// Tier-compiled contiguous scalar walks for the wrapped-scalar arithmetic
// forms: add.Scalar, sub.Scalar, mul.Scalar, div.Scalar and their in-place,
// scalar-list and _out variants all funnel into these three walks.
//
// The implementations live in ScalarArithVecKernelsImpl.cpp, which is listed
// in TP_CPU_KERNEL_SRCS so each CPU capability tier compiles its own vector
// loop (see p10/CMakeLists.txt).  The kernels in ArithmeticKernels.cpp stay in
// the base tier and hand contiguous buffers to this stub.  That is what makes
// the vector path reachable from a compiler without per-function ISA
// attributes (MSVC): the tier copy's own /arch flag supplies the ISA, and the
// runtime dispatch picks the widest copy the CPU supports.
//
// code selects the walk -- 0 add, 1 mul, 2 div -- and dtype is the DType
// enumerator cast to int (Float32 or Float64 only).  The caller folds the
// wrapped scalar into `scalar` and expresses subtraction by handing the negated
// scalar to the add walk, exactly like the in-house AVX2/AVX512 helpers in
// ArithmeticKernels.cpp.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

using scalar_vec_fn = void (*)(const void* a, void* y, int64_t n,
                               double scalar, int code, int dtype);

DECLARE_DISPATCH(scalar_vec_fn, scalar_vec_stub)

// Same-shape, same-dtype, both-contiguous tensor OP tensor for the real
// arithmetic kernels (mul.Tensor, div.Tensor and the sum/difference forms).
// `a`, `b` and `y` are the three flat buffers, `n` the element count, `code`
// 0 add, 1 sub, 2 mul, 3 div, and `dtype` the common DType enumerator cast to
// int (Float32 or Float64 only).  The caller has already checked shapes,
// dtypes and contiguity, so the walk is one unit-stride pass; a case that does
// not meet those conditions stays on the generic iterator path.
using binary_vec_fn = void (*)(const void* a, const void* b, void* y,
                               int64_t n, int code, int dtype);

DECLARE_DISPATCH(binary_vec_fn, binary_vec_stub)

} // namespace cpu
} // namespace tensorplay
