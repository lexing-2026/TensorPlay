#pragma once

// Three-tier (DEFAULT/AVX2/AVX512) dispatch stubs for the complex SIMD
// kernels.  The implementations live in ComplexKernels.cpp, which is listed
// in TP_CPU_KERNEL_SRCS so each CPU capability tier gets its own copy of the
// vecunary/veccomplex cores (see p10/CMakeLists.txt).

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// Op is veccomplex::Op; encoded as int in stub signatures to avoid including
// VecComplex.h (which would drag the AVX2 kernels into every consumer).
using cplx_unary_fn = bool (*)(const void*, void*, int64_t, int, int);
DECLARE_DISPATCH(cplx_unary_fn, cplx_unary_stub)

using cplx_binary_fn = bool (*)(const void*, const void*, void*, int64_t, int, int);
DECLARE_DISPATCH(cplx_binary_fn, cplx_binary_stub)

using cplx_sum_fn = bool (*)(const void*, int64_t, int, double*, double*);
DECLARE_DISPATCH(cplx_sum_fn, cplx_sum_stub)

using cplx_abs_fn = bool (*)(const void*, void*, int64_t, int);
DECLARE_DISPATCH(cplx_abs_fn, cplx_abs_stub)

using cplx_angle_fn = bool (*)(const void*, void*, int64_t, int);
DECLARE_DISPATCH(cplx_angle_fn, cplx_angle_stub)

} // namespace cpu
} // namespace tensorplay
