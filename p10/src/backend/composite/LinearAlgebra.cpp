// Backend-neutral linear-algebra composites.
//
//   chain_matmul (alias of linalg.multi_dot).  The optimal-parenthesization
//   DP only changes evaluation order, so the sequential matmul fold is
//   numerically equivalent up to fp associativity.
//
//   The determinant family.  det/slogdet forward to their linalg spellings;
//   logdet reads the signed decomposition and reports NaN where a real
//   determinant is negative (its logarithm is not real).  The two underscore
//   entry points additionally hand back the LU factorization and its pivots
//   from one factorization pass, so a caller that needs the decomposition for
//   a subsequent derivative does not factor the matrix twice.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <limits>
#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor chain_matmul_native(const std::vector<Tensor>& matrices) {
    for (const auto& m : matrices) {
        if (m.dim() != 2) {
            TP_THROW(RuntimeError,
                     "chain_matmul(): all matrices must be 2-D, but got a ",
                     m.dim(), "-D tensor");
        }
    }
    if (matrices.empty()) {
        TP_THROW(RuntimeError,
                 "chain_matmul(): Expected one or more matrices");
    }
    if (matrices.size() == 1) return ops::clone(matrices[0], kContiguous);
    Tensor result = matrices[0];
    for (size_t i = 1; i < matrices.size(); ++i) {
        result = ops::matmul(result, matrices[i]);
    }
    return result;
}


namespace {

// Determinant of the row permutation LAPACK reports.  Pivot entries are
// 1-based; every entry that differs from its own position marks one
// transposition, so an even count leaves the sign at +1 and an odd count
// flips it.
Tensor lu_permutation_sign(const Tensor& pivots, DType dtype) {
    const int64_t k = pivots.size(-1);
    const Tensor positions =
        ops::arange(Scalar(static_cast<int64_t>(1)),
                    Scalar(static_cast<int64_t>(k + 1)),
                    Scalar(static_cast<int64_t>(1)),
                    pivots.dtype(), pivots.device());
    const Tensor swaps =
        ops::sum(ops::ne(pivots, positions), {-1}, false, DType::Int64);
    const Tensor even =
        ops::eq(ops::fmod(swaps, Scalar(static_cast<int64_t>(2))),
                Scalar(static_cast<int64_t>(0)));
    return ops::where(even, Scalar(1.0), Scalar(-1.0)).to(dtype);
}

// The factorization shared by the determinant entry points.  A contiguous
// real matrix is factored transposed: the column-major kernels then read it
// without a repacking copy, and det(A^T) == det(A).
std::tuple<Tensor, Tensor> det_lu_factor(const Tensor& A) {
    const bool transpose = A.is_contiguous() && !isComplexType(A.dtype());
    auto factored =
        ops::linalg_lu_factor_ex(transpose ? A.transpose(-2, -1) : A, true, false);
    return {std::get<0>(factored), std::get<1>(factored)};
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> _linalg_det_native(const Tensor& A) {
    auto [LU, pivots] = det_lu_factor(A);
    const Tensor diagonal = ops::diagonal(LU, 0, -2, -1);
    Tensor result = ops::mul(ops::prod(diagonal, -1, false),
                             lu_permutation_sign(pivots, A.dtype()));
    return {result, LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _linalg_slogdet_native(const Tensor& A) {
    auto [LU, pivots] = det_lu_factor(A);
    const Tensor diagonal = ops::diagonal(LU, 0, -2, -1);
    Tensor sign = ops::mul(ops::prod(ops::sgn(diagonal), -1, false),
                           lu_permutation_sign(pivots, A.dtype()));
    Tensor logabsdet = ops::sum(ops::log(ops::abs(diagonal)), {-1}, false);
    return {sign, logabsdet, LU, pivots};
}

Tensor det_native(const Tensor& self) {
    return ops::linalg_det(self);
}

std::tuple<Tensor, Tensor> slogdet_native(const Tensor& self) {
    return ops::linalg_slogdet(self);
}

// log|det| carries the determinant's sign: a negative real determinant has no
// real logarithm and reports NaN, while a complex determinant folds the phase
// in through log(sign).
Tensor logdet_native(const Tensor& self) {
    auto [sign, logabsdet] = ops::linalg_slogdet(self);
    if (isComplexType(self.dtype())) {
        return ops::add(ops::log(sign), logabsdet);
    }
    return ops::where(ops::eq(sign, Scalar(-1.0)),
                      Scalar(std::numeric_limits<double>::quiet_NaN()),
                      logabsdet);
}

// out= form of the LU unpacking: each destination keeps the caller's buffer,
// resized only when the produced factor does not already fit.
namespace {

Tensor& adopt_out(Tensor& out, const Tensor& value) {
    if (!out.defined()) {
        out = value;
        return out;
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

}  // namespace

std::tuple<Tensor&, Tensor&, Tensor&> lu_unpack_out_native(
        const Tensor& LU_data, const Tensor& LU_pivots, bool unpack_data,
        bool unpack_pivots, Tensor& P, Tensor& L, Tensor& U) {
    auto unpacked =
        ops::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
    adopt_out(P, std::get<0>(unpacked));
    adopt_out(L, std::get<1>(unpacked));
    adopt_out(U, std::get<2>(unpacked));
    return {P, L, U};
}

TENSORPLAY_LIBRARY_IMPL(Composite, LinearAlgebraComposite) {
    m.impl("chain_matmul", chain_matmul_native);
    m.impl("det", det_native);
    m.impl("slogdet", slogdet_native);
    m.impl("logdet", logdet_native);
    m.impl("_linalg_det", _linalg_det_native);
    m.impl("_linalg_slogdet", _linalg_slogdet_native);
    m.impl("lu_unpack.out", lu_unpack_out_native);
}

} // namespace composite
} // namespace tensorplay
