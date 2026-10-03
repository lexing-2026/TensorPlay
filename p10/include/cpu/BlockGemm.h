#pragma once

// A product of two row-major blocks, for kernels written at run time.
//
// A kernel generated while a program is compiled cannot name the BLAS the
// library was built against -- it does not know which one that was, or whether
// there is one -- so the library offers the product here and the kernel calls
// it.  The answer is computed in float whatever the operands are stored as, and
// written to a float block.

#include <cstdint>

#include "BFloat16.h"
#include "DType.h"
#include "Half.h"
#include "Macros.h"

namespace tensorplay {

// C = A @ B, or C += A @ B when `add_C`, for an M x K block A and a K x N block
// B; `ld_*` is how far apart two rows of each block are.  `is_vnni` asks for B
// in the pair-interleaved order a matrix unit reads; this library has no such
// unit to hand a product to, so a caller asking for it is refused.
P10_API void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a,
                    int64_t ld_b, int64_t ld_c, bool add_C, const float* A,
                    const float* B, float* C, bool is_vnni = false);
P10_API void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a,
                    int64_t ld_b, int64_t ld_c, bool add_C,
                    const BFloat16* A, const BFloat16* B, float* C,
                    bool is_vnni = false);
P10_API void brgemm(int64_t M, int64_t N, int64_t K, int64_t ld_a,
                    int64_t ld_b, int64_t ld_c, bool add_C, const Half* A,
                    const Half* B, float* C, bool is_vnni = false);

// Whether a block of this type can be packed for a matrix unit.  Without one
// the answer is no, and a kernel keeps its operands in the order they came in.
P10_API bool could_pack(DType dtype);

// Lets go of whatever a packed product kept configured.  Nothing is kept when
// nothing was packed, so this is a call a kernel can always make.
P10_API void brgemm_release(bool is_vnni = true);

}  // namespace tensorplay
