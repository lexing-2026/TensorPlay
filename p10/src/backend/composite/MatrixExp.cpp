// Backend-neutral matrix exponential.
//
// exp(A) is evaluated by scaling and squaring around a diagonal Pade
// approximant.  The [m/m] Pade rational R_m(x) = p_m(x) / p_m(-x) has a
// backward error bounded by the machine epsilon whenever ||A||_1 stays under
// the degree's threshold theta_m, so the routine picks the smallest degree
// whose threshold covers the batch and, when even the largest degree is not
// enough, divides each matrix by a power of two, evaluates there and squares
// the result back up.  The squaring count is per matrix, so a batch of mixed
// magnitudes never over-scales the small members.
//
// Every step is a matmul, an elementwise combination or one linear solve, so
// the whole routine executes through the native kernels of whichever device
// holds the input.
//
// The gradient uses the block form
//     exp([[A^H, G], [0, A^H]]) = [[exp(A^H), D], [0, exp(A^H)]],
// whose upper-right block D is exactly the adjoint of the differential of
// exp at A applied to G.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// Numerator coefficients of the diagonal Pade approximant of exp, degree by
// degree; b_k = (2m-k)! m! / ((2m)! k! (m-k)!) cleared of its denominator.
constexpr double kPade3[] = {120., 60., 12., 1.};
constexpr double kPade5[] = {30240., 15120., 3360., 420., 30., 1.};
constexpr double kPade7[] = {17297280., 8648640., 1995840., 277200.,
                             25200., 1512., 56., 1.};
constexpr double kPade9[] = {17643225600., 8821612800., 2075673600., 302702400.,
                             30270240., 2162160., 110880., 3960., 90., 1.};
constexpr double kPade13[] = {
    64764752532480000., 32382376266240000., 7771770303897600.,
    1187353796428800., 129060195264000., 10559470521600., 670442572800.,
    33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};

// Largest ||A||_1 at which each degree still meets a backward error of one
// double-precision unit roundoff.  The same bounds serve the lower-precision
// element types: they are strictly tighter than those formats need, so a
// float or half input is only ever evaluated at a degree at least as accurate
// as its own precision requires.
constexpr double kTheta3 = 1.495585217958292e-2;
constexpr double kTheta5 = 2.539398330063230e-1;
constexpr double kTheta7 = 9.504178996162932e-1;
constexpr double kTheta9 = 2.097847961257068e0;
constexpr double kTheta13 = 5.371920351148152e0;

// Operator 1-norm of every matrix in the batch: the largest absolute column
// sum, reported as one real value per batch member.
Tensor operator_one_norm(const Tensor& batch) {
    return ops::amax(ops::sum(ops::abs(batch), {-2}, false), {-1}, false);
}

// U and V halves of the Pade pair for the low degrees.  The odd-power half
// factors A out front, so both halves only ever need the even powers of A.
std::pair<Tensor, Tensor> pade_low_degree(const Tensor& A, const Tensor& I,
                                          const double* b, int degree) {
    const Tensor A2 = ops::matmul(A, A);
    Tensor odd = ops::mul(I, Scalar(b[1]));
    Tensor even = ops::mul(I, Scalar(b[0]));
    Tensor power = A2;
    for (int k = 3; k <= degree; k += 2) {
        odd = ops::add(odd, ops::mul(power, Scalar(b[k])));
        even = ops::add(even, ops::mul(power, Scalar(b[k - 1])));
        if (k + 2 <= degree) power = ops::matmul(power, A2);
    }
    return {ops::matmul(A, odd), even};
}

// Degree 13 splits each half again around A^6 so the evaluation needs six
// matrix products instead of the twelve a plain Horner walk would take.
std::pair<Tensor, Tensor> pade_degree13(const Tensor& A, const Tensor& I) {
    const double* b = kPade13;
    const Tensor A2 = ops::matmul(A, A);
    const Tensor A4 = ops::matmul(A2, A2);
    const Tensor A6 = ops::matmul(A4, A2);

    Tensor odd_high = ops::add(ops::add(ops::mul(A6, Scalar(b[13])),
                                        ops::mul(A4, Scalar(b[11]))),
                               ops::mul(A2, Scalar(b[9])));
    Tensor odd_low = ops::add(ops::add(ops::add(ops::mul(A6, Scalar(b[7])),
                                                ops::mul(A4, Scalar(b[5]))),
                                       ops::mul(A2, Scalar(b[3]))),
                              ops::mul(I, Scalar(b[1])));
    Tensor U = ops::matmul(A, ops::add(ops::matmul(A6, odd_high), odd_low));

    Tensor even_high = ops::add(ops::add(ops::mul(A6, Scalar(b[12])),
                                         ops::mul(A4, Scalar(b[10]))),
                                ops::mul(A2, Scalar(b[8])));
    Tensor even_low = ops::add(ops::add(ops::add(ops::mul(A6, Scalar(b[6])),
                                                 ops::mul(A4, Scalar(b[4]))),
                                        ops::mul(A2, Scalar(b[2]))),
                               ops::mul(I, Scalar(b[0])));
    Tensor V = ops::add(ops::matmul(A6, even_high), even_low);
    return {U, V};
}

// R_m(A) = (V - U)^{-1} (V + U): one linear solve per batch member.
Tensor pade_ratio(const Tensor& U, const Tensor& V) {
    return ops::linalg_solve(ops::sub(V, U), ops::add(V, U), true);
}

Tensor matrix_exp_batched(const Tensor& batch) {
    const int64_t n = batch.size(-1);
    const Tensor identity =
        ops::eye(n, n, batch.dtype(), batch.device(), false);
    const Tensor norm = operator_one_norm(batch);
    const double largest = ops::max(norm).item().toDouble();

    // A degree wide enough for the largest member is at least as accurate for
    // the smaller ones, so one degree serves the whole batch.
    if (largest <= kTheta3) {
        auto [U, V] = pade_low_degree(batch, identity, kPade3, 3);
        return pade_ratio(U, V);
    }
    if (largest <= kTheta5) {
        auto [U, V] = pade_low_degree(batch, identity, kPade5, 5);
        return pade_ratio(U, V);
    }
    if (largest <= kTheta7) {
        auto [U, V] = pade_low_degree(batch, identity, kPade7, 7);
        return pade_ratio(U, V);
    }
    if (largest <= kTheta9) {
        auto [U, V] = pade_low_degree(batch, identity, kPade9, 9);
        return pade_ratio(U, V);
    }

    // Scale each matrix by its own power of two, evaluate at degree 13 and
    // square back.  The squaring loop runs as many rounds as the widest
    // member needs; a member that is already done keeps its value, so nothing
    // is scaled further than its own norm requires.
    const Tensor exponent = ops::clamp(
        ops::ceil(ops::log2(ops::div(norm, Scalar(kTheta13)))),
        Scalar(0.0), std::nullopt);
    const std::vector<int64_t> broadcast{batch.size(0), 1, 1};
    const Tensor factor =
        ops::reshape(ops::reciprocal(ops::exp2(exponent)), broadcast);

    auto [U, V] = pade_degree13(ops::mul(batch, factor), identity);
    Tensor result = pade_ratio(U, V);

    const int64_t rounds = static_cast<int64_t>(ops::max(exponent).item().toDouble());
    for (int64_t round = 0; round < rounds; ++round) {
        const Tensor pending = ops::reshape(
            ops::gt(exponent, Scalar(static_cast<double>(round))), broadcast);
        result = ops::where(pending, ops::matmul(result, result), result);
    }
    return result;
}

void check_square_floating(const Tensor& self, const char* name) {
    TP_CHECK(self.dim() >= 2, name,
             ": expected a batch of square matrices, got a ", self.dim(),
             "-D tensor");
    TP_CHECK(self.size(-1) == self.size(-2), name,
             ": expected square matrices, got ", self.size(-2), " by ",
             self.size(-1));
    TP_CHECK(isFloatingType(self.dtype()) || isComplexType(self.dtype()), name,
             ": expected a floating point or complex tensor");
}

}  // namespace

Tensor linalg_matrix_exp_native(const Tensor& self) {
    check_square_floating(self, "linalg.matrix_exp");
    const int64_t n = self.size(-1);
    if (n == 0) return ops::clone(self, kContiguous);
    if (n == 1) return ops::exp(self);

    const auto shape = static_cast<std::vector<int64_t>>(self.shape());
    const Tensor batched = ops::reshape(self, {-1, n, n});
    return ops::reshape(matrix_exp_batched(batched), shape);
}

Tensor matrix_exp_native(const Tensor& self) {
    return linalg_matrix_exp_native(self);
}

// Adjoint of the differential of exp at `self` applied to `grad`, read off the
// upper-right block of the exponential of the doubled block matrix.
Tensor matrix_exp_backward_native(const Tensor& self, const Tensor& grad) {
    check_square_floating(self, "matrix_exp_backward");
    const int64_t n = self.size(-1);
    if (n == 0) return ops::clone(self, kContiguous);

    const Tensor adjoint = isComplexType(self.dtype())
                               ? ops::conj(ops::transpose(self, -2, -1))
                               : ops::transpose(self, -2, -1);
    const Tensor zero = ops::zeros_like(adjoint);
    const Tensor upper = ops::cat({adjoint, grad}, -1);
    const Tensor lower = ops::cat({zero, adjoint}, -1);
    const Tensor blocked = ops::cat({upper, lower}, -2);

    const Tensor exponential = linalg_matrix_exp_native(blocked);
    return ops::narrow(ops::narrow(exponential, -2, 0, n), -1, n, n);
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, MatrixExpComposite) {
    m.impl("linalg_matrix_exp", composite::linalg_matrix_exp_native);
    m.impl("matrix_exp", composite::matrix_exp_native);
    m.impl("matrix_exp_backward", composite::matrix_exp_backward_native);
}

}  // namespace tensorplay
