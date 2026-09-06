// Legacy-namespace batched matrix decompositions on CUDA: cholesky,
// cholesky_inverse, cholesky_solve, triangular_solve, svd.
//
// Each op composes the cusolver-backed linalg.* kernels (cholesky_ex,
// solve_triangular, svd) through the dispatcher, so the heavy factorization
// never leaves the GPU.  Half/BFloat16 inputs compute in float32.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Utils.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>
#include <tuple>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

namespace {

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
}

bool is_low_precision(DType d) {
    return d == DType::Float16 || d == DType::BFloat16;
}

// Raise the not-positive-definite condition the same way the CPU path does.
void check_cholesky_info(const Tensor& info) {
    Tensor info_host = info.to(Device(DeviceType::CPU));
    const int32_t* data = info_host.data_ptr<int32_t>();
    for (int64_t i = 0; i < info_host.numel(); ++i) {
        if (data[i] != 0) {
            TP_THROW(RuntimeError, "cholesky: matrix is not positive definite");
        }
    }
}

// Compute dtype routing for low-precision inputs.
Tensor to_compute(const Tensor& t) {
    return is_low_precision(t.dtype()) ? t.to(DType::Float32) : t;
}
Tensor from_compute(const Tensor& t, DType dt) {
    return t.dtype() == dt ? t : t.to(dt);
}

}  // namespace

Tensor cholesky_cuda(const Tensor& self, bool upper) {
    require_float(self, "cholesky");
    const DType dt = self.dtype();
    auto [L, info] = ops::linalg_cholesky_ex(to_compute(self), upper, false);
    check_cholesky_info(info);
    return from_compute(L, dt);
}

Tensor cholesky_inverse_cuda(const Tensor& self, bool upper) {
    // A = L L^T (upper: A = U^T U) -> A^{-1} via two triangular solves
    // against the identity: L X1 = I, then L^T X = X1.
    require_float(self, "cholesky_inverse");
    const DType dt = self.dtype();
    const Tensor factor = to_compute(self);
    const int64_t n = factor.size(-1);
    const std::vector<int64_t> batch(factor.shape().begin(), factor.shape().end() - 2);
    std::vector<int64_t> eye_shape = batch;
    eye_shape.push_back(n);
    eye_shape.push_back(n);
    Tensor identity = ops::eye(n, n, factor.dtype(), factor.device())
                          .expand(eye_shape)
                          .contiguous();
    Tensor inner, inverse;
    if (upper) {
        // A = U^T U: solve U Y = I (upper), then U^T X = Y (lower).
        inner = ops::linalg_solve_triangular(factor, identity, true, true, false);
        inverse = ops::linalg_solve_triangular(
            factor.transpose(-2, -1), inner, false, true, false);
    } else {
        // A = L L^T: solve L Y = I (lower), then L^T X = Y (upper).
        inner = ops::linalg_solve_triangular(factor, identity, false, true, false);
        inverse = ops::linalg_solve_triangular(
            factor.transpose(-2, -1), inner, true, true, false);
    }
    return from_compute(inverse, dt);
}

Tensor cholesky_solve_cuda(const Tensor& self, const Tensor& input2, bool upper) {
    // self = B (..., n, rhs); input2 = factor; solve (L L^T) X = B with two
    // triangular sweeps, as in the classic potrs factorization.
    require_float(self, "cholesky_solve");
    require_float(input2, "cholesky_solve");
    const DType dt = self.dtype();
    const Tensor factor = to_compute(input2);
    const Tensor rhs = to_compute(self);
    Tensor inner, solution;
    if (upper) {
        // A = U^T U: U Y = B (upper), then U^T X = Y (lower).
        inner = ops::linalg_solve_triangular(factor, rhs, true, true, false);
        solution = ops::linalg_solve_triangular(
            factor.transpose(-2, -1), inner, false, true, false);
    } else {
        // A = L L^T: L Y = B (lower), then L^T X = Y (upper).
        inner = ops::linalg_solve_triangular(factor, rhs, false, true, false);
        solution = ops::linalg_solve_triangular(
            factor.transpose(-2, -1), inner, true, true, false);
    }
    return from_compute(solution, dt);
}

std::tuple<Tensor, Tensor> triangular_solve_cuda(const Tensor& self, const Tensor& A,
                                                 bool upper, bool transpose,
                                                 bool unitriangular) {
    require_float(A, "triangular_solve");
    // transpose=True solves A^T X = B, which is a plain solve against the
    // flipped triangle.
    const Tensor factor = transpose ? A.transpose(-2, -1) : A;
    const bool factor_upper = transpose ? !upper : upper;
    Tensor X = ops::linalg_solve_triangular(to_compute(factor), to_compute(self),
                                            factor_upper, true, unitriangular);
    return {from_compute(X, self.dtype()), A.clone()};
}

std::tuple<Tensor, Tensor, Tensor> svd_cuda(const Tensor& self, bool some, bool compute_uv) {
    require_float(self, "svd");
    (void)some;
    const DType dt = self.dtype();
    if (compute_uv) {
        // Reduced factorization; the legacy contract returns V (A =
        // U diag(S) V^T), so the Vh factor is transposed into the third
        // slot, matching the CPU kernel.
        auto [U, S, Vh] = ops::linalg_svd(to_compute(self), false, std::optional<std::string>());
        return {from_compute(U, dt), from_compute(S, dt),
                from_compute(Vh, dt).transpose(-2, -1).contiguous()};
    }
    Tensor S = ops::linalg_svdvals(to_compute(self), std::optional<std::string>());
    Tensor zero = Tensor::zeros({}, dt, self.device());
    return {zero, from_compute(S, dt), zero};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, BatchLinearAlgebra) {
    m.impl("cholesky", cholesky_cuda);
    m.impl("cholesky_inverse", cholesky_inverse_cuda);
    m.impl("cholesky_solve", cholesky_solve_cuda);
    m.impl("triangular_solve", triangular_solve_cuda);
    m.impl("svd", svd_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
