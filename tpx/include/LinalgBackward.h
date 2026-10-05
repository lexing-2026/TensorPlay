#pragma once
// Backward of the matrix factorizations and solvers.
//
// Every formula is composed of dispatched, recordable ops, so the same code
// runs on every device and create_graph records through it.  mH is the
// conjugate transpose; real and complex inputs share the formulas.  The
// factorizations with more than one differentiable output (svd, eigh, eig,
// qr, lu, slogdet, lstsq, triangular_solve) receive one gradient per output
// and are hand-written nodes at the end of this file.

#include "Node.h"
#include "Autograd.h"
#include "SavedVariable.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace linalg_bwd_detail {

inline double epsilon_of(DType dtype) {
    return toRealValueType(dtype) == DType::Float64
               ? std::numeric_limits<double>::epsilon()
               : static_cast<double>(std::numeric_limits<float>::epsilon());
}

inline Tensor identity_like(const Tensor& A) {
    const int64_t n = A.size(-1);
    return ops::eye(n, n, A.dtype(), std::nullopt, A.device(), std::nullopt);
}

// A batch of square matrices with x on the diagonal.
inline Tensor diag_of(const Tensor& x, int64_t n) {
    return ops::diag_embed(
        ops::expand(ops::unsqueeze(x, -1),
                    [&] {
                        auto shape = static_cast<std::vector<int64_t>>(x.shape());
                        shape.push_back(n);
                        return shape;
                    }(),
                    false),
        0, -2, -1);
}

// B is a vector right-hand side when it is one, or when it has the batch
// shape of A with the last extent dropped.
inline bool is_vector_rhs(const Tensor& A, const Tensor& B) {
    if (B.dim() == 1) return true;
    if (A.dim() - 1 != B.dim()) return false;
    for (int64_t i = 0; i < B.dim(); ++i) {
        if (B.size(i) != A.size(i)) return false;
    }
    return true;
}

}  // namespace linalg_bwd_detail

// inverse: d(A^-1) = -A^-1 dA A^-1.
inline Tensor linalg_inv_backward(const Tensor& grad, const Tensor& inverse) {
    const Tensor inv_h = ops::mH(inverse);
    return -ops::matmul(inv_h, ops::matmul(grad, inv_h));
}

// det: d(det A) = det A tr(A^-1 dA), so the gradient is grad conj(det) A^-H.
// A singular matrix has no inverse; there the determinant is taken through
// the singular values, det A = det U det Vh prod S, whose product rule needs
// no division.
inline Tensor linalg_det_backward(const Tensor& grad, const Tensor& det, const Tensor& A) {
    using namespace linalg_bwd_detail;
    if (!grad.defined()) return Tensor();
    if (A.numel() == 0) return ops::zeros_like(A);
    const auto shape = static_cast<std::vector<int64_t>>(A.shape());
    const int64_t n = A.size(-1);
    if (n == 1) {
        return ops::expand(ops::unsqueeze(ops::unsqueeze(grad, -1), -1), shape, false);
    }
    const Tensor d = diag_of(grad * ops::conj(det), n);
    const auto nonsingular = [&](const Tensor& M) {
        return ops::linalg_solve(ops::mH(M), d, true);
    };
    const auto singular = [&](const Tensor& M) {
        auto [U, S, Vh] = ops::linalg_svd(M, true, std::nullopt);
        const Tensor alpha = ops::conj(ops::linalg_det(U) * ops::linalg_det(Vh)) * grad;
        const Tensor D = prod_safe_zeros_backward(ops::unsqueeze(alpha, -1), S, S.dim() - 1);
        return ops::matmul(U * ops::unsqueeze(D, -2), Vh);
    };
    const Tensor is_singular =
        ops::lt(ops::abs(det), Scalar(100.0 * epsilon_of(A.dtype())));
    if (!ops::any(is_singular).item<bool>()) return nonsingular(A);
    if (ops::all(is_singular).item<bool>()) return singular(A);
    // A batch of both kinds: each path sees a harmless stand-in for the
    // matrices it does not own.
    const Tensor mask = ops::unsqueeze(ops::unsqueeze(is_singular, -1), -1);
    const Tensor ones = ops::ones_like(ops::diagonal(A, 0, -2, -1));
    const Tensor identity = ops::diag_embed(ones, 0, -2, -1);
    const Tensor distinct = ops::diag_embed(ops::cumsum(ones, -1, std::nullopt), 0, -2, -1);
    return ops::where(mask, singular(ops::where(mask, A, distinct)),
                      nonsingular(ops::where(mask, identity, A)));
}

// log|det A| moves by tr(A^-1 dA); for complex A the sign's phase moves by
// the imaginary part of the same trace.
inline Tensor slogdet_backward(const Tensor& grad_sign, const Tensor& grad_logabsdet,
                               const Tensor& A, const Tensor& sign) {
    using namespace linalg_bwd_detail;
    const bool is_complex = isComplexType(A.dtype());
    if (!grad_logabsdet.defined() && (!is_complex || !grad_sign.defined())) {
        return Tensor();
    }
    Tensor g = grad_logabsdet;
    if (is_complex) {
        Tensor re = g.defined() ? g : ops::zeros_like(ops::real(sign));
        Tensor im = grad_sign.defined()
                        ? -ops::imag(ops::conj(grad_sign) * sign)
                        : ops::zeros_like(re);
        g = ops::complex(re, im);
    }
    return ops::linalg_solve(ops::mH(A), diag_of(g, A.size(-1)), true);
}

// pseudo-inverse: the derivative of A^+ in terms of A^+ itself.
inline Tensor pinv_backward(const Tensor& grad, const Tensor& pinvA, const Tensor& A) {
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const Tensor pinvAh = ops::mH(pinvA);
    const Tensor gradh = ops::mH(grad);
    if (m <= n) {
        const Tensor K = ops::matmul(gradh, pinvA);
        const Tensor KpinvAh = ops::matmul(K, pinvAh);
        return -ops::mH(ops::matmul(pinvA, K)) + KpinvAh -
               ops::matmul(ops::matmul(A, pinvA), KpinvAh) +
               ops::matmul(ops::matmul(pinvAh, pinvA), gradh - ops::matmul(K, A));
    }
    const Tensor K = ops::matmul(pinvA, gradh);
    const Tensor pinvAhK = ops::matmul(pinvAh, K);
    return -ops::mH(ops::matmul(K, pinvA)) +
           ops::matmul(ops::matmul(gradh - ops::matmul(A, K), pinvA), pinvAh) + pinvAhK -
           ops::matmul(ops::matmul(pinvAhK, pinvA), A);
}

// A = P L U.  The factors' gradients are pulled back through the two
// triangular solves; a wide or tall A carries the extra block of U or L
// straight through.
inline Tensor linalg_lu_backward(const Tensor& L_grad, const Tensor& U_grad,
                                 const Tensor& P, const Tensor& L, const Tensor& U,
                                 bool pivot) {
    if (!L_grad.defined() && !U_grad.defined()) return Tensor();
    const int64_t m = L.size(-2);
    const int64_t n = U.size(-1);
    const int64_t k = std::min(m, n);
    Tensor A_grad;
    if (m == n) {
        if (L_grad.defined()) A_grad = ops::tril(ops::matmul(ops::mH(L), L_grad), -1);
        if (U_grad.defined()) {
            Tensor upper = ops::triu(ops::matmul(U_grad, ops::mH(U)), 0);
            A_grad = A_grad.defined() ? A_grad + upper : upper;
        }
        A_grad = ops::linalg_solve_triangular(ops::mH(U), A_grad, false, false, false);
        A_grad = ops::linalg_solve_triangular(ops::mH(L), A_grad, true, true, true);
    } else if (m < n) {
        const Tensor U1 = ops::narrow(U, -1, 0, k);
        if (L_grad.defined()) A_grad = ops::matmul(ops::mH(L), L_grad);
        if (U_grad.defined()) {
            Tensor term = -ops::matmul(ops::triu(U_grad, 0), ops::mH(U));
            A_grad = A_grad.defined() ? A_grad + term : term;
        }
        A_grad = ops::linalg_solve_triangular(ops::mH(U1), ops::tril(A_grad, -1), false,
                                              false, false);
        if (U_grad.defined()) {
            A_grad = ops::cat({A_grad + ops::triu(ops::narrow(U_grad, -1, 0, k), 0),
                               ops::narrow(U_grad, -1, k, n - k)},
                              -1);
        }
        A_grad = ops::linalg_solve_triangular(ops::mH(L), A_grad, true, true, true);
        if (!U_grad.defined()) {
            A_grad = ops::cat({A_grad, ops::zeros_like(ops::narrow(U, -1, k, n - k))}, -1);
        }
    } else {
        const Tensor L1 = ops::narrow(L, -2, 0, k);
        if (U_grad.defined()) A_grad = ops::matmul(U_grad, ops::mH(U));
        if (L_grad.defined()) {
            Tensor term = -ops::matmul(ops::mH(L), ops::tril(L_grad, -1));
            A_grad = A_grad.defined() ? A_grad + term : term;
        }
        A_grad = ops::linalg_solve_triangular(ops::mH(L1), ops::triu(A_grad, 0), true,
                                              true, true);
        if (L_grad.defined()) {
            A_grad = ops::cat({A_grad + ops::tril(ops::narrow(L_grad, -2, 0, k), -1),
                               ops::narrow(L_grad, -2, k, m - k)},
                              -2);
        }
        A_grad = ops::linalg_solve_triangular(ops::mH(U), A_grad, false, false, false);
        if (!L_grad.defined()) {
            A_grad = ops::cat({A_grad, ops::zeros_like(ops::narrow(L, -2, k, m - k))}, -2);
        }
    }
    return pivot ? ops::matmul(P, A_grad) : A_grad;
}

// The packed factorization holds L below the diagonal and U on and above it.
inline Tensor lu_factor_ex_backward(const Tensor& grad, const Tensor& LU,
                                    const Tensor& pivots, bool pivot) {
    if (!grad.defined()) return Tensor();
    auto [P, L, U] = ops::lu_unpack(LU, pivots, true, pivot);
    const int64_t k = std::min(LU.size(-2), LU.size(-1));
    return linalg_lu_backward(ops::narrow(grad, -1, 0, k), ops::narrow(grad, -2, 0, k),
                              P, L, U, pivot);
}

// Unpacking reads L's strict lower part and U's upper part out of the packed
// matrix; the gradient is written back into the same places.
inline Tensor lu_unpack_backward(const Tensor& L_grad, const Tensor& U_grad,
                                 int64_t m, int64_t n) {
    if (!L_grad.defined() && !U_grad.defined()) return Tensor();
    const int64_t k = std::min(m, n);
    if (L_grad.defined() && U_grad.defined()) {
        if (m == n) return ops::tril(L_grad, -1) + ops::triu(U_grad, 0);
        Tensor A1 = ops::tril(ops::narrow(L_grad, -2, 0, k), -1) +
                    ops::triu(ops::narrow(U_grad, -1, 0, k), 0);
        Tensor A2 = m > n ? ops::narrow(L_grad, -2, k, m - k)
                          : ops::narrow(U_grad, -1, k, n - k);
        return ops::cat({A1, A2}, m > n ? -2 : -1);
    }
    if (L_grad.defined()) {
        if (m >= n) return ops::tril(L_grad, -1);
        auto shape = static_cast<std::vector<int64_t>>(L_grad.shape());
        shape.back() = n - m;
        return ops::cat({ops::tril(L_grad, -1),
                         ops::zeros(shape, L_grad.dtype(), L_grad.device())},
                        -1);
    }
    if (n >= m) return ops::triu(U_grad, 0);
    auto shape = static_cast<std::vector<int64_t>>(U_grad.shape());
    shape[shape.size() - 2] = m - n;
    return ops::cat({ops::triu(U_grad, 0),
                     ops::zeros(shape, U_grad.dtype(), U_grad.device())},
                    -2);
}

// X solves LU X = B (or X LU = B, or with the adjoint); the gradient with
// respect to the packed factors.
inline Tensor linalg_lu_solve_LU(const Tensor& gX, const Tensor& LU, const Tensor& pivots,
                                 const Tensor& X, bool left, bool adjoint) {
    auto [P, L, U] = ops::lu_unpack(LU, pivots, true, left == adjoint);
    if (left != adjoint) {
        const Tensor rhs = -ops::matmul(left ? gX : ops::mH(gX), left ? ops::mH(X) : X);
        const Tensor gR = ops::linalg_solve_triangular(ops::mH(U), rhs, false, true, false);
        const Tensor gL = ops::tril(
            ops::linalg_solve_triangular(ops::mH(L), ops::matmul(gR, ops::mH(U)), true,
                                         true, true),
            -1);
        return gL + ops::triu(gR, 0);
    }
    Tensor gR = -ops::matmul(
        ops::matmul(ops::matmul(ops::mT(P), left ? X : ops::mH(X)), left ? ops::mH(gX) : gX),
        P);
    gR = ops::linalg_solve_triangular(ops::mH(L), gR, true, false, true);
    const Tensor gU = ops::triu(
        ops::linalg_solve_triangular(ops::mH(U), ops::matmul(ops::mH(L), gR), false, false,
                                     false),
        0);
    return ops::tril(gR, -1) + gU;
}

// The right-hand side's gradient is the same solve with the adjoint.
inline Tensor linalg_lu_solve_B(const Tensor& gX, const Tensor& LU, const Tensor& pivots,
                                bool left, bool adjoint) {
    return ops::linalg_lu_solve(LU, pivots, gX, left, !adjoint);
}

// X solves A X = B (or X A = B): gB solves the adjoint system and gA = -gB X^H.
// The forward's factorization of A answers the adjoint system directly; a
// backward that is itself recorded solves again so the graph reaches A.
inline std::tuple<Tensor, Tensor> linalg_solve_backward(const Tensor& gX, const Tensor& X,
                                                        const Tensor& A, const Tensor& B,
                                                        const Tensor& LU, const Tensor& pivots,
                                                        bool left,
                                                        const std::vector<bool>& mask) {
    using namespace linalg_bwd_detail;
    const bool A_needed = !mask.empty() && mask[0];
    const bool B_needed = mask.size() > 1 && mask[1];
    if (!gX.defined() || (!A_needed && !B_needed)) return {};
    const bool vector_case = is_vector_rhs(A, B);
    const Tensor gX_ = vector_case ? ops::unsqueeze(gX, -1) : gX;
    const Tensor gB = GradMode::is_enabled()
                          ? ops::linalg_solve(ops::mH(A), gX_, left)
                          : ops::linalg_lu_solve(LU, pivots, gX_, left, true);
    Tensor gA;
    if (A_needed) {
        const Tensor X_ = vector_case ? ops::unsqueeze(X, -1) : X;
        gA = left ? -ops::matmul(gB, ops::mH(X_)) : -ops::matmul(ops::mH(X_), gB);
    }
    return {gA, B_needed ? (vector_case ? ops::squeeze(gB, -1) : gB) : Tensor()};
}

// X solves the triangular system; A's gradient keeps only its triangle (and
// leaves the diagonal alone when it is taken to be ones).
inline std::tuple<Tensor, Tensor> linalg_solve_triangular_backward(
        const Tensor& grad, const Tensor& A, const Tensor& X, bool upper, bool left,
        bool unitriangular, const std::vector<bool>& mask) {
    const bool A_needed = !mask.empty() && mask[0];
    const bool B_needed = mask.size() > 1 && mask[1];
    if (!grad.defined() || (!A_needed && !B_needed)) return {};
    const Tensor G_B =
        ops::linalg_solve_triangular(ops::mH(A), grad, !upper, left, unitriangular);
    if (!A_needed) return {Tensor(), G_B};
    const Tensor X_H = ops::mH(X);
    Tensor G_A = left ? -ops::matmul(G_B, X_H) : -ops::matmul(X_H, G_B);
    G_A = upper ? ops::triu(G_A, unitriangular ? 1 : 0) : ops::tril(G_A, unitriangular ? -1 : 0);
    return {G_A, B_needed ? G_B : Tensor()};
}

// triangular_solve returns the solution and a copy of A; a gradient reaching
// that copy goes straight to A.
inline std::tuple<Tensor, Tensor> triangular_solve_backward(
        const Tensor& grad_x, const Tensor& grad_m, const Tensor& b, const Tensor& a,
        const Tensor& x, bool upper, bool transpose, bool unitriangular,
        bool b_needed, bool a_needed) {
    Tensor grad_b, grad_a;
    if (!grad_x.defined() && !grad_m.defined()) return {};
    if (grad_x.defined()) {
        grad_b = std::get<0>(
            ops::triangular_solve(grad_x, ops::conj(a), upper, !transpose, unitriangular));
        if (a_needed) {
            grad_a = transpose ? -ops::matmul(ops::conj(x), ops::mT(grad_b))
                               : -ops::matmul(grad_b, ops::mH(x));
            grad_a = upper ? ops::triu(grad_a, unitriangular ? 1 : 0)
                           : ops::tril(grad_a, unitriangular ? -1 : 0);
        }
    }
    if (a_needed && grad_m.defined()) {
        grad_a = grad_a.defined() ? grad_a + grad_m : grad_m;
    }
    return {b_needed ? grad_b : Tensor(), a_needed ? grad_a : Tensor()};
}

// A = L L^H: the gradient is symmetrized and pulled back through L.
inline Tensor cholesky_backward(const Tensor& gL, bool upper, const Tensor& L) {
    if (!gL.defined()) return Tensor();
    const Tensor L_ = upper ? ops::mH(L) : L;
    const Tensor gL_ = upper ? ops::mH(gL) : gL;
    Tensor gA = ops::tril(ops::matmul(ops::mH(L_), gL_), 0);
    gA = (gA + ops::mH(ops::tril(gA, -1))) * 0.5;
    gA = ops::linalg_solve_triangular(ops::mH(L_), gA, true, true, false);
    return ops::linalg_solve_triangular(L_, gA, false, false, false);
}

// X = (L L^H)^-1 B.
inline std::tuple<Tensor, Tensor> cholesky_solve_backward(const Tensor& grad_x,
                                                          const Tensor& self,
                                                          const Tensor& input2,
                                                          const Tensor& result, bool upper,
                                                          const std::vector<bool>& mask) {
    (void)self;
    if (!grad_x.defined()) return {};
    const Tensor grad_self = ops::cholesky_solve(grad_x, input2, upper);
    Tensor grad_input2;
    if (mask.size() > 1 && mask[1]) {
        Tensor common = ops::matmul(grad_self, ops::mH(result));
        common = common + ops::mH(common);
        grad_input2 = upper ? -ops::matmul(input2, common) : -ops::matmul(common, input2);
    }
    return {grad_self, grad_input2};
}

// (L L^H)^-1 from its factor.
inline Tensor cholesky_inverse_backward(const Tensor& grad, const Tensor& L, bool upper,
                                        const Tensor& inverse) {
    if (!grad.defined()) return Tensor();
    Tensor common = grad + ops::mH(grad);
    common = ops::matmul(inverse, ops::matmul(common, inverse));
    return upper ? -ops::matmul(L, common) : -ops::matmul(common, L);
}

// Least squares: the solution is pinv(A) B; the residuals |A X - B|^2 add
// their own term.
inline std::tuple<Tensor, Tensor> linalg_lstsq_backward(const Tensor& gX_, const Tensor& gL_,
                                                        const Tensor& A, const Tensor& B_,
                                                        const Tensor& X_, bool A_needed,
                                                        bool B_needed) {
    using namespace linalg_bwd_detail;
    const bool has_gL = gL_.defined() && gL_.numel() > 0;
    if ((!gX_.defined() && !has_gL) || (!A_needed && !B_needed)) return {};
    const bool vector_case = is_vector_rhs(A, B_);
    const auto to_matrix = [&](const Tensor& t) {
        return vector_case ? ops::unsqueeze(t, -1) : t;
    };
    const auto to_vector = [&](const Tensor& t) {
        return vector_case ? ops::squeeze(t, -1) : t;
    };
    const Tensor B = to_matrix(B_);
    Tensor A_grad, B_grad;
    if (gX_.defined()) {
        const Tensor gX = to_matrix(gX_);
        const Tensor pinvA = ops::linalg_pinv(A, std::optional<Tensor>(), std::optional<Tensor>(),
                                              false);
        if (A_needed) A_grad = pinv_backward(ops::matmul(gX, ops::mH(B)), pinvA, A);
        if (B_needed) B_grad = to_vector(ops::matmul(ops::mH(pinvA), gX));
    }
    if (has_gL) {
        const Tensor X = to_matrix(X_);
        const Tensor r = ops::matmul(A, X) - B;
        const Tensor gL = ops::unsqueeze(gL_, -2);
        if (A_needed) {
            Tensor term = ops::matmul(gL * r, ops::mH(X)) * 2;
            A_grad = A_grad.defined() ? A_grad + term : term;
        }
        if (B_needed) {
            Tensor term = to_vector(gL * r * (-2));
            B_grad = B_grad.defined() ? B_grad + term : term;
        }
    }
    return {A_grad, B_grad};
}

// A = U diag(S) Vh.  The singular vectors are defined up to a common phase;
// a loss that depends on it has no derivative, which is reported for
// complex inputs.
inline Tensor svd_backward(const Tensor& gU, const Tensor& gS, const Tensor& gVh,
                           const Tensor& U, const Tensor& S, const Tensor& Vh) {
    if (!gS.defined() && !gU.defined() && !gVh.defined()) return Tensor();
    const int64_t m = U.size(-2);
    const int64_t n = Vh.size(-1);
    if (!gU.defined() && !gVh.defined()) {
        return m >= n ? ops::matmul(U, ops::unsqueeze(gS, -1) * Vh)
                      : ops::matmul(U * ops::unsqueeze(gS, -2), Vh);
    }
    const bool is_complex = isComplexType(U.dtype());
    const auto skew = [](const Tensor& X) { return X - ops::mH(X); };
    const Tensor UhgU = gU.defined() ? skew(ops::matmul(ops::mH(U), gU)) : Tensor();
    const Tensor VhgV = gVh.defined() ? skew(ops::matmul(Vh, ops::mH(gVh))) : Tensor();
    if (is_complex) {
        const Tensor a = gU.defined() ? ops::imag(ops::diagonal(UhgU, 0, -2, -1))
                                      : ops::zeros_like(S);
        const Tensor b = gVh.defined() ? ops::imag(ops::diagonal(VhgV, 0, -2, -1))
                                       : ops::zeros_like(S);
        TP_CHECK(ops::allclose(a, -b, 1e-2, 1e-2, false),
                 "svd_backward: the singular vectors of a complex matrix are defined up to "
                 "a phase, and the loss depends on that phase, so it has no derivative");
    }
    const Tensor S2 = S * S;
    // E_ij = S_j^2 - S_i^2, with ones on the diagonal.
    const Tensor E = ops::unsqueeze(S2, -2) - ops::unsqueeze(S2, -1) +
                     ops::diag_embed(ops::ones_like(S), 0, -2, -1);
    Tensor gA;
    if (gU.defined() && gVh.defined()) {
        gA = (UhgU * ops::unsqueeze(S, -2) + ops::unsqueeze(S, -1) * VhgV) / E;
    } else if (gU.defined()) {
        gA = (UhgU / E) * ops::unsqueeze(S, -2);
    } else {
        gA = ops::unsqueeze(S, -1) * (VhgV / E);
    }
    if (gS.defined()) gA = gA + ops::diag_embed(gS, 0, -2, -1);
    if (is_complex && gU.defined() && gVh.defined()) {
        gA = gA + ops::diag_embed(ops::diagonal(UhgU, 0, -2, -1) / (S * 2), 0, -2, -1);
    }
    if (m > n && gU.defined()) {
        gA = ops::matmul(U, gA);
        const Tensor gUSinv = gU / ops::unsqueeze(S, -2);
        gA = gA + gUSinv - ops::matmul(U, ops::matmul(ops::mH(U), gUSinv));
        return ops::matmul(gA, Vh);
    }
    if (m < n && gVh.defined()) {
        gA = ops::matmul(gA, Vh);
        const Tensor SinvgVh = gVh / ops::unsqueeze(S, -1);
        gA = gA + SinvgVh - ops::matmul(ops::matmul(SinvgVh, ops::mH(Vh)), Vh);
        return ops::matmul(U, gA);
    }
    return m >= n ? ops::matmul(U, ops::matmul(gA, Vh)) : ops::matmul(ops::matmul(U, gA), Vh);
}

// A V = V diag(L).  For a Hermitian A the eigenvectors are unitary and the
// gradient stays Hermitian; otherwise V is normalized to unit columns and
// the gradient is pulled back through V^-H.
inline Tensor linalg_eig_backward(const Tensor& gL, const Tensor& gV, const Tensor& L,
                                  const Tensor& V, bool is_hermitian) {
    if (!gL.defined() && !gV.defined()) return Tensor();
    const Tensor Vh = ops::mH(V);
    if (!gV.defined()) {
        if (is_hermitian) return ops::matmul(V * ops::unsqueeze(gL, -2), Vh);
        return ops::linalg_solve(Vh, ops::unsqueeze(gL, -1) * Vh, true);
    }
    Tensor VhgV = ops::matmul(Vh, gV);
    const Tensor diag_VhgV = ops::diagonal(VhgV, 0, -2, -1);
    if (isComplexType(V.dtype())) {
        const Tensor im = ops::imag(diag_VhgV);
        TP_CHECK(ops::allclose(im, ops::zeros_like(im), 1e-2, 1e-2, false),
                 is_hermitian ? "linalg_eigh_backward" : "linalg_eig_backward",
                 ": the eigenvectors of a complex matrix are defined up to a phase, and the "
                 "loss depends on that phase, so it has no derivative");
    }
    if (is_hermitian) {
        VhgV = (VhgV - ops::mH(VhgV)) * 0.5;
    } else {
        VhgV = VhgV - ops::matmul(Vh, V * ops::unsqueeze(ops::real(diag_VhgV), -2));
    }
    const Tensor Lconj = ops::conj(L);
    // E_ij = conj(L_j) - conj(L_i), with ones on the diagonal.
    const Tensor E = ops::unsqueeze(Lconj, -2) - ops::unsqueeze(Lconj, -1) +
                     ops::diag_embed(ops::ones_like(Lconj), 0, -2, -1);
    Tensor gA = VhgV / E;
    if (gL.defined()) {
        gA = gA - ops::diag_embed(ops::diagonal(gA, 0, -2, -1), 0, -2, -1) +
             ops::diag_embed(gL.dtype() == gA.dtype() ? gL : gL.to(gA.dtype()), 0, -2, -1);
    }
    if (is_hermitian) return ops::matmul(V, ops::matmul(gA, Vh));
    return ops::linalg_solve(Vh, ops::matmul(gA, Vh), true);
}

// A = Q R.  The mode must have produced Q, and a complete factorization of a
// tall matrix leaves part of Q undetermined, so neither is differentiable.
inline Tensor linalg_qr_backward(const Tensor& gQ, const Tensor& gR, const Tensor& Q,
                                 const Tensor& R, const std::string& mode) {
    TP_CHECK(mode != "r",
             "The derivative of linalg.qr depends on Q, which is not computed when "
             "mode='r'. Please use linalg.qr(A, mode='reduced') if you are going to "
             "differentiate through linalg.qr.");
    const int64_t m = Q.size(-2);
    const int64_t n = R.size(-1);
    TP_CHECK(mode == "reduced" || m <= n,
             "The QR decomposition is not differentiable when mode='complete' and "
             "nrows > ncols.");
    if (!gQ.defined() && !gR.defined()) return Tensor();
    Tensor gA;
    if (gQ.defined()) {
        gA = gR.defined() ? ops::matmul(gR, ops::mH(R)) - ops::matmul(ops::mH(Q), gQ)
                          : -ops::matmul(ops::mH(Q), gQ);
    } else {
        gA = ops::matmul(gR, ops::mH(R));
    }
    if (m >= n) {
        // X + X^H with the real part of the diagonal halved.
        Tensor sym = ops::triu(gA, 0);
        sym = sym + ops::mH(sym);
        sym = sym - ops::diag_embed(ops::real(ops::diagonal(sym, 0, -2, -1)) * 0.5, 0, -2, -1);
        gA = ops::matmul(Q, sym);
        if (gQ.defined()) gA = gA + gQ;
        return ops::linalg_solve_triangular(ops::mH(R), gA, false, false, false);
    }
    // The strictly lower part of X - X^H, with the imaginary part of the
    // diagonal halved.
    const Tensor X = -gA;
    Tensor skew = ops::tril(X - ops::mH(X), 0);
    if (isComplexType(skew.dtype())) {
        const Tensor im = ops::imag(ops::diagonal(skew, 0, -2, -1)) * 0.5;
        skew = skew - ops::diag_embed(ops::complex(ops::zeros_like(im), im), 0, -2, -1);
    }
    gA = ops::matmul(Q, skew);
    gA = ops::linalg_solve_triangular(ops::mH(ops::narrow(R, -1, 0, m)), gA, false, false,
                                      false);
    auto shape = static_cast<std::vector<int64_t>>(R.shape());
    shape.back() = n - m;
    gA = ops::cat({gA, ops::zeros(shape, gA.dtype(), gA.device())},
                  -1);
    if (gR.defined()) gA = gA + ops::matmul(Q, gR);
    return gA;
}


// ---------------------------------------------------------------------------
// Factorizations with several differentiable outputs.  The engine delivers
// one gradient per output, in output order, so each node takes as many
// inputs as its operator has outputs.
// ---------------------------------------------------------------------------

namespace linalg_bwd_detail {

inline Tensor input_at(const variable_list& inputs, size_t i) {
    return inputs.size() > i ? inputs[i] : Tensor();
}

}  // namespace linalg_bwd_detail

// _linalg_svd: A = U diag(S) Vh.  A full factorization's extra columns of U
// and rows of Vh carry no information about A and are dropped first.
struct LinalgSvdBackward : public Node {
    bool full_matrices_;
    bool compute_uv_;
    SavedVariable U_;
    SavedVariable S_;
    SavedVariable Vh_;

    LinalgSvdBackward(bool full_matrices, bool compute_uv, Tensor U, Tensor S, Tensor Vh)
        : full_matrices_(full_matrices), compute_uv_(compute_uv), U_(U, true),
          S_(S, true), Vh_(Vh, true) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        Tensor gU = input_at(inputs, 0);
        const Tensor gS = input_at(inputs, 1);
        Tensor gVh = input_at(inputs, 2);
        if (!gU.defined() && !gS.defined() && !gVh.defined()) return {Tensor()};
        TP_CHECK(compute_uv_,
                 "linalg.svd: the singular values alone are not differentiable; "
                 "compute the singular vectors as well (compute_uv=True)");
        Tensor U = U_.unpack_output(shared_from_this(), 0);
        const Tensor S = S_.unpack_output(shared_from_this(), 1);
        Tensor Vh = Vh_.unpack_output(shared_from_this(), 2);
        if (full_matrices_) {
            const int64_t k = S.size(-1);
            U = ops::narrow(U, -1, 0, k);
            Vh = ops::narrow(Vh, -2, 0, k);
            if (gU.defined()) gU = ops::narrow(gU, -1, 0, k);
            if (gVh.defined()) gVh = ops::narrow(gVh, -2, 0, k);
        }
        return {svd_backward(gU, gS, gVh, U, S, Vh)};
    }

    void release_variables() override {
        Node::release_variables();
        U_.reset_data();
        S_.reset_data();
        Vh_.reset_data();
    }
};

// _linalg_eigh: A = V diag(L) V^H.
struct LinalgEighBackward : public Node {
    bool compute_v_;
    SavedVariable eigenvalues_;
    SavedVariable eigenvectors_;

    LinalgEighBackward(bool compute_v, Tensor eigenvalues, Tensor eigenvectors)
        : compute_v_(compute_v), eigenvalues_(eigenvalues, true),
          eigenvectors_(eigenvectors, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gL = input_at(inputs, 0);
        const Tensor gV = input_at(inputs, 1);
        if (!gL.defined() && !gV.defined()) return {Tensor()};
        TP_CHECK(compute_v_,
                 "linalg.eigh: the eigenvalues alone are not differentiable; compute the "
                 "eigenvectors as well (compute_v=True)");
        const auto self = shared_from_this();
        return {linalg_eig_backward(gL, gV, eigenvalues_.unpack_output(self, 0),
                                    eigenvectors_.unpack_output(self, 1), true)};
    }

    void release_variables() override {
        Node::release_variables();
        eigenvalues_.reset_data();
        eigenvectors_.reset_data();
    }
};

// linalg_eig: A V = V diag(L), complex even for a real A, whose gradient is
// the real part.
struct LinalgEigBackward : public Node {
    DType input_dtype_;
    SavedVariable eigenvalues_;
    SavedVariable eigenvectors_;

    LinalgEigBackward(const Tensor& A, Tensor eigenvalues, Tensor eigenvectors)
        : input_dtype_(A.dtype()), eigenvalues_(eigenvalues, true),
          eigenvectors_(eigenvectors, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gL = input_at(inputs, 0);
        const Tensor gV = input_at(inputs, 1);
        if (!gL.defined() && !gV.defined()) return {Tensor()};
        Tensor gA = linalg_eig_backward(gL, gV, eigenvalues_.unpack_output(shared_from_this(), 0),
                                        eigenvectors_.unpack_output(shared_from_this(), 1), false);
        if (!isComplexType(input_dtype_) && isComplexType(gA.dtype())) gA = ops::real(gA);
        return {gA};
    }

    void release_variables() override {
        Node::release_variables();
        eigenvalues_.reset_data();
        eigenvectors_.reset_data();
    }
};

// linalg_qr: A = Q R.
struct LinalgQrBackward : public Node {
    std::string mode_;
    SavedVariable Q_;
    SavedVariable R_;

    LinalgQrBackward(std::string mode, Tensor Q, Tensor R)
        : mode_(std::move(mode)), Q_(Q, true), R_(R, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gQ = input_at(inputs, 0);
        const Tensor gR = input_at(inputs, 1);
        if (!gQ.defined() && !gR.defined()) return {Tensor()};
        const auto self = shared_from_this();
        return {linalg_qr_backward(gQ, gR, Q_.unpack_output(self, 0), R_.unpack_output(self, 1),
                                   mode_)};
    }

    void release_variables() override {
        Node::release_variables();
        Q_.reset_data();
        R_.reset_data();
    }
};

// linalg_lu: A = P L U; the permutation is not differentiable.
struct LinalgLuBackward : public Node {
    bool pivot_;
    SavedVariable P_;
    SavedVariable L_;
    SavedVariable U_;

    LinalgLuBackward(bool pivot, Tensor P, Tensor L, Tensor U)
        : pivot_(pivot), P_(P, true), L_(L, true), U_(U, true) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gL = input_at(inputs, 1);
        const Tensor gU = input_at(inputs, 2);
        if (!gL.defined() && !gU.defined()) return {Tensor()};
        const auto self = shared_from_this();
        return {linalg_lu_backward(gL, gU, P_.unpack_output(self, 0), L_.unpack_output(self, 1),
                                   U_.unpack_output(self, 2), pivot_)};
    }

    void release_variables() override {
        Node::release_variables();
        P_.reset_data();
        L_.reset_data();
        U_.reset_data();
    }
};

// lu_unpack: the factors read out of a packed factorization.
struct LuUnpackBackward : public Node {
    int64_t m_;
    int64_t n_;

    LuUnpackBackward(const Tensor& LU_data)
        : m_(LU_data.size(-2)), n_(LU_data.size(-1)) {}

    size_t num_inputs() const override { return 3; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gL = input_at(inputs, 1);
        const Tensor gU = input_at(inputs, 2);
        if (!gL.defined() && !gU.defined()) return {Tensor(), Tensor()};
        return {lu_unpack_backward(gL, gU, m_, n_), Tensor()};
    }
};

// _linalg_slogdet: the sign and log|det A|.
struct LinalgSlogdetBackward : public Node {
    SavedVariable A_;
    SavedVariable sign_;

    LinalgSlogdetBackward(Tensor A, Tensor sign) : A_(std::move(A)), sign_(sign, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor g_sign = input_at(inputs, 0);
        const Tensor g_logabsdet = input_at(inputs, 1);
        if (!g_sign.defined() && !g_logabsdet.defined()) return {Tensor()};
        return {slogdet_backward(g_sign, g_logabsdet, A_.unpack(),
                                 sign_.unpack_output(shared_from_this(), 0))};
    }

    void release_variables() override {
        Node::release_variables();
        A_.reset_data();
        sign_.reset_data();
    }
};

// linalg_lstsq: the solution and the residuals are differentiable.
struct LinalgLstsqBackward : public Node {
    SavedVariable A_;
    SavedVariable B_;
    SavedVariable solution_;

    LinalgLstsqBackward(Tensor A, Tensor B, Tensor solution)
        : A_(std::move(A)), B_(std::move(B)), solution_(solution, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor gX = input_at(inputs, 0);
        const Tensor gL = input_at(inputs, 1);
        const bool A_needed = should_compute_output(0);
        const bool B_needed = should_compute_output(1);
        auto [gA, gB] = linalg_lstsq_backward(gX, gL, A_.unpack(), B_.unpack(),
                                              solution_.unpack_output(shared_from_this(), 0),
                                              A_needed, B_needed);
        return {gA, gB};
    }

    void release_variables() override {
        Node::release_variables();
        A_.reset_data();
        B_.reset_data();
        solution_.reset_data();
    }
};

// triangular_solve: the solution and the copy of the coefficient matrix.
struct TriangularSolveBackward : public Node {
    SavedVariable self_;
    SavedVariable A_;
    bool upper_;
    bool transpose_;
    bool unitriangular_;
    SavedVariable solution_;

    TriangularSolveBackward(Tensor self, Tensor A, bool upper, bool transpose,
                            bool unitriangular, Tensor solution)
        : self_(std::move(self)), A_(std::move(A)), upper_(upper), transpose_(transpose),
          unitriangular_(unitriangular), solution_(solution, true) {}

    size_t num_inputs() const override { return 2; }

    variable_list apply(variable_list&& inputs) override {
        using linalg_bwd_detail::input_at;
        const Tensor g_solution = input_at(inputs, 0);
        const Tensor g_coefficient = input_at(inputs, 1);
        const bool b_needed = should_compute_output(0);
        const bool a_needed = should_compute_output(1);
        auto [gb, ga] = triangular_solve_backward(
            g_solution, g_coefficient, self_.unpack(), A_.unpack(),
            solution_.unpack_output(shared_from_this(), 0), upper_, transpose_, unitriangular_,
            b_needed, a_needed);
        return {gb, ga};
    }

    void release_variables() override {
        Node::release_variables();
        self_.reset_data();
        A_.reset_data();
        solution_.reset_data();
    }
};

}  // namespace tpx
}  // namespace tensorplay
