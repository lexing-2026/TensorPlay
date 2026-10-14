#pragma once

// Per-ISA sparse SpMM-reduction kernels (sum/mean/amax/amin).  The forward
// and backward loops live in SpmmReduceKernelsImpl.cpp (listed in
// TP_CPU_KERNEL_SRCS); the CSR/CSC bookkeeping, input checks and operator
// entry points stay in the base tier (SpmmReduceKernels.cpp).

#include "DispatchStub.h"
#include "Tensor.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

// reduce selector shared by every stub:
//   0  sum
//   1  mean
//   2  amax
//   3  amin

// Forward: out(m, k) = reduce over stored entries e in row m of
// values[e] * other[col[e], k].  When `need_arg` is set (amax/amin under
// autograd), arg_out receives the winning stored-element index per cell and
// is pre-filled with `nnz` as the "no entry" marker.
using spmm_reduce_fn = void (*)(const Tensor& out, const Tensor& arg_out,
                                const Tensor& crow, const Tensor& col,
                                const Tensor& values, const Tensor& other,
                                int reduce, bool need_arg);

// grad_self (sum/mean): grad_self.values[e] =
//   dot(other[col[e]], grad_out[row[e]]) / row_nnz(row[e]) for mean.
// grad_self (amax/amin): grad_self.values[arg_out[m,k]] +=
//   other[col[arg_out[m,k]], k] * grad_out[m, k].
using spmm_reduce_backward_input_fn = void (*)(
    const Tensor& grad_self, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& other, const Tensor& arg_out,
    const Tensor& row, int reduce);

// grad_other (sum/mean): grad_other[n, k] +=
//   sum over stored e with col[e]=n of values[e] * grad_out[row[e], k].
// grad_other (amax/amin): grad_other[col[arg], k] +=
//   values[arg] * grad_out[m, k] at the arg positions.
using spmm_reduce_backward_other_fn = void (*)(
    const Tensor& grad_other, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& values, const Tensor& arg_out,
    const Tensor& row, const Tensor& ccol, const Tensor& permute, int reduce);

DECLARE_DISPATCH(spmm_reduce_fn, spmm_reduce_stub)
DECLARE_DISPATCH(spmm_reduce_backward_input_fn, spmm_reduce_backward_input_stub)
DECLARE_DISPATCH(spmm_reduce_backward_other_fn, spmm_reduce_backward_other_stub)

} // namespace cpu
} // namespace tensorplay
