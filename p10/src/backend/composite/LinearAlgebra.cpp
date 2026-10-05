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
//
//   The user-facing factorizations, solvers and pseudo-inverse spellings
//   forward to the core operators that carry the derivatives, so a gradient
//   reaches the input whichever spelling the caller used.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>
#include <string>
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

std::tuple<Tensor, Tensor, Tensor> lu_unpack_out_native(
        const Tensor& LU_data, const Tensor& LU_pivots, bool unpack_data,
        bool unpack_pivots, Tensor& P, Tensor& L, Tensor& U) {
    auto unpacked =
        ops::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
    adopt_out(P, std::get<0>(unpacked));
    adopt_out(L, std::get<1>(unpacked));
    adopt_out(U, std::get<2>(unpacked));
    return {P, L, U};
}

// The user-facing factorizations and solvers.  Each answers through the core
// operator that carries the derivative and then reports what its info codes
// say; the eigenvalue and singular-value spellings compute the vectors only
// when a gradient will need them.
namespace {

bool may_need_grad(const Tensor& A) {
    return GradMode::is_enabled() && A.requires_grad();
}

}  // namespace

Tensor linalg_cholesky_native(const Tensor& A, bool upper) {
    auto [L, info] = ops::linalg_cholesky_ex(A, upper, false);
    ops::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
    return L;
}

Tensor linalg_inv_native(const Tensor& A) {
    auto [inverse, info] = ops::linalg_inv_ex(A, false);
    ops::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
    return inverse;
}

Tensor linalg_det_native(const Tensor& A) {
    return std::get<0>(ops::_linalg_det(A));
}

std::tuple<Tensor, Tensor> linalg_slogdet_native(const Tensor& A) {
    auto values = ops::_linalg_slogdet(A);
    return {std::get<0>(values), std::get<1>(values)};
}

std::tuple<Tensor, Tensor> linalg_solve_ex_native(const Tensor& A, const Tensor& B,
                                                  bool left, bool check_errors) {
    auto values = ops::_linalg_solve_ex(A, B, left, check_errors);
    return {std::get<0>(values), std::get<3>(values)};
}

Tensor linalg_solve_native(const Tensor& A, const Tensor& B, bool left) {
    auto [result, info] = ops::linalg_solve_ex(A, B, left, false);
    ops::_linalg_check_errors(info, "linalg.solve", A.dim() == 2);
    return result;
}

std::tuple<Tensor, Tensor> linalg_lu_factor_native(const Tensor& A, bool pivot) {
    auto [LU, pivots, info] = ops::linalg_lu_factor_ex(A, pivot, false);
    ops::_linalg_check_errors(info, "linalg.lu_factor", A.dim() == 2);
    return {LU, pivots};
}

std::tuple<Tensor, Tensor> linalg_eigh_native(const Tensor& A, const std::string& UPLO) {
    return ops::_linalg_eigh(A, UPLO, true);
}

Tensor linalg_eigvalsh_native(const Tensor& A, const std::string& UPLO) {
    return std::get<0>(ops::_linalg_eigh(A, UPLO, may_need_grad(A)));
}

// The eigenvalues of a general matrix are differentiated through its
// eigenvectors, which are then computed and dropped.
Tensor linalg_eigvals_native(const Tensor& A) {
    if (may_need_grad(A)) return std::get<0>(ops::linalg_eig(A));
    return ops::_linalg_eigvals(A);
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_native(
        const Tensor& A, bool full_matrices, const std::optional<std::string>& driver) {
    return ops::_linalg_svd(A, full_matrices, true, driver);
}

Tensor linalg_svdvals_native(const Tensor& A, const std::optional<std::string>& driver) {
    return std::get<1>(ops::_linalg_svd(A, false, may_need_grad(A), driver));
}

// The pseudo-inverse spellings answer through the tensor-tolerance form,
// which carries the derivative.  A plain rcond is a relative tolerance.
Tensor linalg_pinv_float_native(const Tensor& input, std::optional<double> atol,
                                std::optional<double> rtol, bool hermitian) {
    const auto as_tensor = [&](std::optional<double> value) -> std::optional<Tensor> {
        if (!value.has_value()) return std::nullopt;
        return ops::full({}, Scalar(*value), DType::Float64, input.device());
    };
    return ops::linalg_pinv(input, as_tensor(atol), as_tensor(rtol), hermitian);
}

Tensor linalg_pinv_rcond_native(const Tensor& input, double rcond, bool hermitian) {
    return ops::linalg_pinv(input, std::optional<double>(0.0), std::optional<double>(rcond),
                            hermitian);
}

Tensor linalg_pinv_rcond_tensor_native(const Tensor& input, const Tensor& rcond,
                                       bool hermitian) {
    TP_CHECK(!isComplexType(rcond.dtype()),
             "linalg.pinv: rcond tensor of complex type is not supported.");
    return ops::linalg_pinv(
        input, std::optional<Tensor>(ops::zeros({}, DType::Float64, input.device())),
        std::optional<Tensor>(rcond), hermitian);
}

TENSORPLAY_LIBRARY_IMPL(Composite, LinearAlgebraComposite) {
    m.impl("chain_matmul", chain_matmul_native);
    m.impl("linalg_cholesky", linalg_cholesky_native);
    m.impl("linalg_inv", linalg_inv_native);
    m.impl("linalg_det", linalg_det_native);
    m.impl("linalg_slogdet", linalg_slogdet_native);
    m.impl("linalg_solve_ex", linalg_solve_ex_native);
    m.impl("linalg_solve", linalg_solve_native);
    m.impl("linalg_lu_factor", linalg_lu_factor_native);
    m.impl("linalg_eigh", linalg_eigh_native);
    m.impl("linalg_eigvalsh", linalg_eigvalsh_native);
    m.impl("linalg_eigvals", linalg_eigvals_native);
    m.impl("linalg_svd", linalg_svd_native);
    m.impl("linalg_svdvals", linalg_svdvals_native);
    m.impl("linalg_pinv.atol_rtol_float", linalg_pinv_float_native);
    m.impl("linalg_pinv", linalg_pinv_rcond_native);
    m.impl("linalg_pinv.rcond_tensor", linalg_pinv_rcond_tensor_native);
    m.impl("det", det_native);
    m.impl("slogdet", slogdet_native);
    m.impl("logdet", logdet_native);
    m.impl("_linalg_det", _linalg_det_native);
    m.impl("_linalg_slogdet", _linalg_slogdet_native);
    m.impl("lu_unpack.out", lu_unpack_out_native);
}

} // namespace composite
} // namespace tensorplay
