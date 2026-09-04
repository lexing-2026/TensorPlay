// Legacy-namespace batched matrix decompositions: cholesky,
// cholesky_inverse, cholesky_solve, triangular_solve, svd.
//
// Fast paths bind the LAPACK routines (potrf/potri/potrs/trtrs/gesdd) in the
// same manner as the linalg.* kernels; without a LAPACK runtime the
// self-contained scalar fallbacks below keep the ops available in minimal
// builds.  Half/BFloat16 compute in float64 on both paths.

#include "Tensor.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "cpu/Lapack.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <cstring>
#include <atomic>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace ops = tensorplay::tpx::ops;

namespace {

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
}

std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

std::vector<double> batch_block_f64(const Tensor& t, int64_t b, int64_t elems) {
    Tensor tc = t.contiguous().to(DType::Float64);
    std::vector<double> out(static_cast<size_t>(elems));
    std::memcpy(out.data(), tc.data_ptr<double>() + b * elems, elems * sizeof(double));
    return out;
}

void put_batch_block(Tensor& dst, const std::vector<double>& blkHost, int64_t bidx,
                     int64_t elems) {
    Tensor host = Tensor::empty({elems}, DType::Float64, Device(DeviceType::CPU));
    std::memcpy(host.data_ptr<double>(), blkHost.data(), elems * sizeof(double));
    Tensor piece = host.to(dst.dtype()).to(dst.device());
    Tensor dc = dst.contiguous();
    std::memcpy(reinterpret_cast<char*>(dc.data_ptr()) + bidx * elems * dst.itemsize(),
                reinterpret_cast<const char*>(piece.contiguous().data_ptr()),
                elems * dst.itemsize());
    dst = dc;
}

// Solve triangular M x = xvec (M row-major n x n values, not modified).
void triangular_solve_vec(std::vector<double> M, int64_t n, std::vector<double>& xvec,
                          bool upper, bool transposed, bool unitriangular) {
    if (transposed) {
        std::vector<double> Mt(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) Mt[i * n + j] = M[j * n + i];
        M.swap(Mt);
        upper = !upper;
    }
    auto at = [&](int64_t i, int64_t j) { return M[i * n + j]; };
    if (!upper) {
        for (int64_t i = 0; i < n; ++i) {
            double s2 = xvec[i];
            for (int64_t j = 0; j < i; ++j) s2 -= at(i, j) * xvec[j];
            xvec[i] = s2 / (unitriangular ? 1.0 : at(i, i));
        }
    } else {
        for (int64_t i = n - 1; i >= 0; --i) {
            double s2 = xvec[i];
            for (int64_t j = i + 1; j < n; ++j) s2 -= at(i, j) * xvec[j];
            xvec[i] = s2 / (unitriangular ? 1.0 : at(i, i));
        }
    }
}

// ---------------------------------------------------------------------------
// LAPACK fast paths
//
// LAPACK speaks column-major; each operand is converted to a column-major
// buffer (transpose + contiguous keeps the logical shape, transposing the
// view back restores it), the routine runs with standard Fortran
// conventions, and the result is materialized back to row-major by a final
// contiguous() which reads the logical elements in row-major order.
// ---------------------------------------------------------------------------

// Column-major buffer view of `src`: same logical shape; element (i, j) of
// each trailing matrix lives at i + j * rows.
Tensor clone_batched_col_major(const Tensor& src) {
    return src.transpose(-2, -1).contiguous().transpose(-2, -1);
}

bool cholesky_lapack(const Tensor& self, Tensor& out, int64_t n, int64_t batch,
                     bool upper) {
    const char uplo = upper ? 'U' : 'L';
    const int64_t n2 = n * n;
    std::atomic<int64_t> first_err{-1};
    auto record = [&](int64_t err) {
        int64_t expected = -1;
        if (err != 0) first_err.compare_exchange_strong(expected, err);
        return err == 0;
    };
    // Zero the triangle opposite to `uplo` in column-major coordinates.
    auto zero_opposite_cm = [&](auto* a, auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                const bool strictly_opposite = upper ? (i > j) : (i < j);
                if (strictly_opposite) a[i + j * n] = T(0);
            }
        }
    };
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        Tensor work = clone_batched_col_major(out);
        parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t bi = begin; bi < end; ++bi) {
                if (self.dtype() == DType::Float32) {
                    float* block = work.data_ptr<float>() + bi * n2;
                    if (record(lapack_spotrf(uplo, n, block, n))) {
                        zero_opposite_cm(block, static_cast<float*>(nullptr));
                    }
                } else {
                    double* block = work.data_ptr<double>() + bi * n2;
                    if (record(lapack_dpotrf(uplo, n, block, n))) {
                        zero_opposite_cm(block, static_cast<double*>(nullptr));
                    }
                }
            }
        });
        out.copy_(work.contiguous());
    } else {
        // Half/BFloat16: factor in float64 and write back in storage dtype.
        Tensor work = clone_batched_col_major(out.to(DType::Float64));
        parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t bi = begin; bi < end; ++bi) {
                double* block = work.data_ptr<double>() + bi * n2;
                if (record(lapack_dpotrf(uplo, n, block, n))) {
                    zero_opposite_cm(block, static_cast<double*>(nullptr));
                }
            }
        });
        out.copy_(work.contiguous().to(out.dtype()));
    }
    if (first_err.load() >= 0) {
        TP_THROW(RuntimeError, "cholesky: matrix is not positive definite");
    }
    return true;
}

Tensor cholesky_inverse_lapack(const Tensor& self, int64_t n, int64_t batch, bool upper) {
    // A^{-1} via two triangular sweeps against the identity (potri in the
    // bundled wheel is slow for large n; trsm is the level-3 fast path).
    const int64_t order = 101;  // CblasRowMajor
    const int64_t side = 141;   // CblasLeft
    const int64_t uplo = upper ? 121 : 122;  // CblasUpper : CblasLower
    const int64_t diag = 131;                // CblasNonUnit
    Tensor fac = detail::contiguous_clone(self);
    Tensor out = Tensor::zeros(shape_of(self), self.dtype(), self.device());
    const int64_t n2 = n * n;
    auto run_f32 = [&](const Tensor& f32, Tensor& work) {
        // One identity column per rhs pass; trsm sweeps the whole batch of
        // columns at once, so seed B with I (row-major blocks) and solve.
        for (int64_t bi = 0; bi < batch; ++bi) {
            float* x = work.data_ptr<float>() + bi * n2;
            for (int64_t d = 0; d < n; ++d) x[d * n + d] = 1.0f;
            const float* f = f32.data_ptr<float>() + bi * n2;
            if (upper) {
                lapack_strsm(order, side, uplo, 112, diag, n, n, 1.0f, f, n, x, n);
                lapack_strsm(order, side, uplo, 111, diag, n, n, 1.0f, f, n, x, n);
            } else {
                lapack_strsm(order, side, uplo, 111, diag, n, n, 1.0f, f, n, x, n);
                lapack_strsm(order, side, uplo, 112, diag, n, n, 1.0f, f, n, x, n);
            }
        }
    };
    if (self.dtype() == DType::Float32) {
        run_f32(fac, out);
        return out;
    }
    if (self.dtype() == DType::Float64) {
        Tensor fac64 = fac.to(DType::Float64);
        Tensor work = Tensor::zeros(shape_of(self), DType::Float64, self.device());
        for (int64_t bi = 0; bi < batch; ++bi) {
            double* x = work.data_ptr<double>() + bi * n2;
            for (int64_t d = 0; d < n; ++d) x[d * n + d] = 1.0;
            const double* f = fac64.data_ptr<double>() + bi * n2;
            if (upper) {
                lapack_dtrsm(order, side, uplo, 112, diag, n, n, 1.0, f, n, x, n);
                lapack_dtrsm(order, side, uplo, 111, diag, n, n, 1.0, f, n, x, n);
            } else {
                lapack_dtrsm(order, side, uplo, 111, diag, n, n, 1.0, f, n, x, n);
                lapack_dtrsm(order, side, uplo, 112, diag, n, n, 1.0, f, n, x, n);
            }
        }
        return work.to(self.dtype());
    }
    // Half/BFloat16: invert in float32 and cast back.
    Tensor fac32 = fac.to(DType::Float32);
    Tensor work = Tensor::zeros(shape_of(self), DType::Float32, self.device());
    run_f32(fac32, work);
    return work.to(self.dtype());
}

Tensor cholesky_solve_lapack(const Tensor& self, const Tensor& factor,
                             int64_t n, int64_t rhs, int64_t batchB, int64_t batchL,
                             bool upper) {
    // (L L^T) X = B for a lower factor, (U^T U) X = B for an upper one, via
    // two triangular sweeps (the bundled wheel's potrs degrades super-
    // linearly with the number of right-hand sides).  trsm consumes
    // row-major operands natively.
    const int64_t order = 101;  // CblasRowMajor
    const int64_t side = 141;   // CblasLeft
    const int64_t uplo = upper ? 121 : 122;  // CblasUpper : CblasLower
    const int64_t diag = 131;                // CblasNonUnit
    Tensor fac = detail::contiguous_clone(factor);
    Tensor out = detail::contiguous_clone(self);
    const int64_t n2 = n * n;
    const int64_t nb = n * rhs;
    auto run_f32 = [&](const Tensor& f32, Tensor& work) {
        for (int64_t bi = 0; bi < batchB; ++bi) {
            const int64_t li = batchL == 1 ? 0 : bi;
            const float* f = f32.data_ptr<float>() + li * n2;
            float* x = work.data_ptr<float>() + bi * nb;
            if (upper) {
                // A = U^T U: U^T Z = B (trans), then U X = Z (notrans).
                lapack_strsm(order, side, uplo, 112, diag, n, rhs, 1.0f, f, n, x, rhs);
                lapack_strsm(order, side, uplo, 111, diag, n, rhs, 1.0f, f, n, x, rhs);
            } else {
                // A = L L^T: L Y = B (notrans), then L^T X = Y (trans).
                lapack_strsm(order, side, uplo, 111, diag, n, rhs, 1.0f, f, n, x, rhs);
                lapack_strsm(order, side, uplo, 112, diag, n, rhs, 1.0f, f, n, x, rhs);
            }
        }
    };
    if (self.dtype() == DType::Float32) {
        run_f32(fac, out);
        return out;
    }
    if (self.dtype() == DType::Float64) {
        Tensor fac64 = fac.to(DType::Float64);
        Tensor work = detail::contiguous_clone(self.to(DType::Float64));
        for (int64_t bi = 0; bi < batchB; ++bi) {
            const int64_t li = batchL == 1 ? 0 : bi;
            const double* f = fac64.data_ptr<double>() + li * n2;
            double* x = work.data_ptr<double>() + bi * nb;
            if (upper) {
                lapack_dtrsm(order, side, uplo, 112, diag, n, rhs, 1.0, f, n, x, rhs);
                lapack_dtrsm(order, side, uplo, 111, diag, n, rhs, 1.0, f, n, x, rhs);
            } else {
                lapack_dtrsm(order, side, uplo, 111, diag, n, rhs, 1.0, f, n, x, rhs);
                lapack_dtrsm(order, side, uplo, 112, diag, n, rhs, 1.0, f, n, x, rhs);
            }
        }
        return work.to(self.dtype());
    }
    // Half/BFloat16: solve in float32 and cast back.
    Tensor fac32 = fac.to(DType::Float32);
    Tensor work = detail::contiguous_clone(self.to(DType::Float32));
    run_f32(fac32, work);
    return work.to(self.dtype());
}

Tensor triangular_solve_lapack(const Tensor& self, const Tensor& A,
                               int64_t n, int64_t rhs, int64_t batchB, int64_t batchA,
                               bool upper, bool transpose, bool unitriangular) {
    // CBLAS trsm consumes row-major operands natively, so no layout
    // conversion is needed: X = op(A)^-1 B per batch block.
    // CBLAS enum values (netlib ABI).
    const int64_t order = 101;  // CblasRowMajor
    const int64_t side = 141;   // CblasLeft
    const int64_t uplo = upper ? 121 : 122;       // CblasUpper : CblasLower
    const int64_t trans = transpose ? 112 : 111;  // CblasTrans : CblasNoTrans
    const int64_t diag = unitriangular ? 132 : 131; // CblasUnit : CblasNonUnit
    Tensor a = detail::contiguous_clone(A);
    Tensor out = detail::contiguous_clone(self);
    const int64_t n2 = n * n;
    const int64_t nb = n * rhs;
    if (self.dtype() == DType::Float32 || self.dtype() == DType::Float64) {
        parallel_for(0, batchB, 1, [&](int64_t begin, int64_t end) {
            for (int64_t bi = begin; bi < end; ++bi) {
                const int64_t ai = batchA == 1 ? 0 : bi;
                if (self.dtype() == DType::Float32) {
                    lapack_strsm(order, side, uplo, trans, diag, n, rhs, 1.0f,
                                 a.data_ptr<float>() + ai * n2, n,
                                 out.data_ptr<float>() + bi * nb, rhs);
                } else {
                    lapack_dtrsm(order, side, uplo, trans, diag, n, rhs, 1.0,
                                 a.data_ptr<double>() + ai * n2, n,
                                 out.data_ptr<double>() + bi * nb, rhs);
                }
            }
        });
        return out;
    }
    // Half/BFloat16: solve in float32 and cast back.
    Tensor a32 = a.to(DType::Float32);
    Tensor work = detail::contiguous_clone(self.to(DType::Float32));
    parallel_for(0, batchB, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bi = begin; bi < end; ++bi) {
            const int64_t ai = batchA == 1 ? 0 : bi;
            lapack_strsm(order, side, uplo, trans, diag, n, rhs, 1.0f,
                         a32.data_ptr<float>() + ai * n2, n,
                         work.data_ptr<float>() + bi * nb, rhs);
        }
    });
    return work.to(self.dtype());
}

// ---------------------------------------------------------------------------
// Scalar fallbacks (no LAPACK runtime)
// ---------------------------------------------------------------------------

Tensor cholesky_scalar(const Tensor& self, int64_t n, int64_t batch, bool upper) {
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> A = batch_block_f64(self, bidx, n * n);
        // Cholesky-Banachiewicz on the lower triangle.
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s2 = A[i * n + j];
                for (int64_t kk = 0; kk < j; ++kk) s2 -= A[i * n + kk] * A[j * n + kk];
                if (i == j) {
                    if (!(s2 > 0))
                        TP_THROW(RuntimeError,
                                 "cholesky: matrix is not positive definite");
                    A[i * n + i] = std::sqrt(s2);
                } else {
                    A[i * n + j] = s2 / A[j * n + j];
                }
            }
            for (int64_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }
        if (upper) {
            std::vector<double> U(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) U[i * n + j] = A[j * n + i];
            A.swap(U);
        }
        put_batch_block(out, A, bidx, n * n);
    }
    return out;
}

Tensor cholesky_inverse_scalar(const Tensor& self, int64_t n, int64_t batch, bool upper) {
    // A = L L^T -> A^{-1} = L^{-T} L^{-1}; the input is the factor.
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> L = batch_block_f64(self, bidx, n * n);
        if (upper) {
            std::vector<double> Lt(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) Lt[i * n + j] = L[j * n + i];
            L.swap(Lt);
        }
        std::vector<double> Linv(n * n, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            std::vector<double> e(n, 0.0);
            e[j] = 1.0;
            triangular_solve_vec(L, n, e, false, false, false);
            for (int64_t i = 0; i < n; ++i) Linv[i * n + j] = e[i];
        }
        std::vector<double> Ainv(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s2 = 0;
                for (int64_t kk = 0; kk < n; ++kk)
                    s2 += Linv[kk * n + i] * Linv[kk * n + j];
                Ainv[i * n + j] = s2;
            }
        put_batch_block(out, Ainv, bidx, n * n);
    }
    return out;
}

Tensor cholesky_solve_scalar(const Tensor& self, const Tensor& input2,
                             int64_t n, int64_t rhs, int64_t batchB, int64_t batchL,
                             bool upper) {
    // self = B (..., n, rhs); input2 = factor; solve (L L^T) X = B.
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batchB; ++bidx) {
        int64_t li = batchL == 1 ? 0 : bidx;
        std::vector<double> L = batch_block_f64(input2, li, n * n);
        if (upper) {
            std::vector<double> Lt(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) Lt[i * n + j] = L[j * n + i];
            L.swap(Lt);
        }
        std::vector<double> B = batch_block_f64(self, bidx, n * rhs);
        for (int64_t c = 0; c < rhs; ++c) {
            std::vector<double> xcol(n);
            for (int64_t i = 0; i < n; ++i) xcol[i] = B[i * rhs + c];
            triangular_solve_vec(L, n, xcol, false, false, false);   // L y = b
            triangular_solve_vec(L, n, xcol, false, true, false);    // L^T x = y
            for (int64_t i = 0; i < n; ++i) B[i * rhs + c] = xcol[i];
        }
        put_batch_block(out, B, bidx, n * rhs);
    }
    return out;
}

Tensor triangular_solve_scalar(const Tensor& self, const Tensor& A,
                               int64_t n, int64_t rhs, int64_t batchB, int64_t batchA,
                               bool upper, bool transpose, bool unitriangular) {
    Tensor X = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batchB; ++bidx) {
        int64_t ai = batchA == 1 ? 0 : bidx;
        std::vector<double> M = batch_block_f64(A, ai, n * n);
        std::vector<double> B = batch_block_f64(self, bidx, n * rhs);
        for (int64_t c = 0; c < rhs; ++c) {
            std::vector<double> xcol(n);
            for (int64_t i = 0; i < n; ++i) xcol[i] = B[i * rhs + c];
            triangular_solve_vec(M, n, xcol, upper, transpose, unitriangular);
            for (int64_t i = 0; i < n; ++i) B[i * rhs + c] = xcol[i];
        }
        put_batch_block(X, B, bidx, n * rhs);
    }
    return X;
}

// One-sided Jacobi reference SVD, always "reduced" for compute_uv=true.
std::tuple<Tensor, Tensor, Tensor> svd_scalar(const Tensor& self, int64_t m, int64_t k,
                                              int64_t batch, bool compute_uv) {
    const int64_t nd = self.dim();
    int64_t r = std::min(m, k);
    bool flip = m < k;
    int64_t wm = flip ? k : m;  // working-matrix rows
    int64_t wn = flip ? m : k;  // working-matrix cols == rank

    Tensor U = Tensor::empty(flip ? [&]{auto v=shape_of(self); v[nd-2]=m; v[nd-1]=r; return v;}()
                                  : shape_of(self),
                             self.dtype(), self.device());
    Tensor S = Tensor::empty([&]{auto v=shape_of(self); v[nd - 2] = r; v.pop_back(); return v;}(),
                             self.dtype(), self.device());
    std::vector<int64_t> vh_shape = shape_of(self);
    vh_shape[nd - 2] = r;
    vh_shape[nd - 1] = k;
    Tensor Vh = Tensor::empty(vh_shape, self.dtype(), self.device());

    const double kTol = 1e-14;
    const int kMaxSweeps = 60;
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> W = batch_block_f64(self, bidx, m * k);
        if (flip) {
            // work on the transpose (wm x wn)
            std::vector<double> Wt(wm * wn);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < k; ++j) Wt[j * wm + i] = W[i * k + j];
            W.swap(Wt);
        }
        std::vector<double> V(wn * wn, 0.0);
        for (int64_t i = 0; i < wn; ++i) V[i * wn + i] = 1.0;
        for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
            double offmax = 0;
            for (int64_t pp = 0; pp < wn; ++pp) {
                for (int64_t q = pp + 1; q < wn; ++q) {
                    double al = 0, be = 0, ga = 0;
                    for (int64_t i2 = 0; i2 < wm; ++i2) {
                        double xp = W[i2 * wn + pp], xq = W[i2 * wn + q];
                        al += xp * xp;
                        be += xq * xq;
                        ga += xp * xq;
                    }
                    double rel = ga / std::sqrt(std::max(al * be, 1e-300));
                    offmax = std::max(offmax, std::fabs(rel));
                    if (ga == 0.0 || std::fabs(rel) <= kTol) continue;
                    double zeta = (be - al) / (2.0 * ga);
                    double t = (zeta >= 0 ? 1.0 : -1.0) /
                               (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double sv = c * t;
                    for (int64_t i2 = 0; i2 < wm; ++i2) {
                        double xp = W[i2 * wn + pp], xq = W[i2 * wn + q];
                        W[i2 * wn + pp] = c * xp - sv * xq;
                        W[i2 * wn + q] = sv * xp + c * xq;
                    }
                    for (int64_t i2 = 0; i2 < wn; ++i2) {
                        double vp = V[i2 * wn + pp], vq = V[i2 * wn + q];
                        V[i2 * wn + pp] = c * vp - sv * vq;
                        V[i2 * wn + q] = sv * vp + c * vq;
                    }
                }
            }
            if (offmax <= kTol) break;
        }
        std::vector<double> sv(wn, 0.0);
        for (int64_t j = 0; j < wn; ++j) {
            double s2 = 0;
            for (int64_t i2 = 0; i2 < wm; ++i2) s2 += W[i2 * wn + j] * W[i2 * wn + j];
            sv[j] = std::sqrt(s2);
        }
        std::vector<int64_t> ord(wn);
        for (int64_t j = 0; j < wn; ++j) ord[j] = j;
        std::sort(ord.begin(), ord.end(),
                  [&](int64_t a2, int64_t b2) { return sv[a2] > sv[b2]; });

        // Non-flip: A = U S V^T with U = normalized W cols (wm x r),
        // Vh = V cols transposed (r x wn).
        // Flip: worked on A^T: U_A = V cols (m x r), Vh_A = W cols/sv (r x k).
        std::vector<double> Uout(static_cast<size_t>((flip ? m : wm) * r), 0.0);
        std::vector<double> Vhout(static_cast<size_t>(r * (flip ? k : wn)), 0.0);
        for (int64_t jj = 0; jj < r; ++jj) {
            int64_t j = ord[jj];
            double denom = sv[j] > 0 ? sv[j] : 1.0;
            if (!flip) {
                for (int64_t i2 = 0; i2 < wm; ++i2)
                    Uout[i2 * r + jj] = W[i2 * wn + j] / denom;
                for (int64_t i2 = 0; i2 < wn; ++i2)
                    Vhout[jj * wn + i2] = V[i2 * wn + j];
            } else {
                for (int64_t i2 = 0; i2 < m; ++i2)
                    Uout[i2 * r + jj] = V[i2 * wn + j];
                for (int64_t i2 = 0; i2 < k; ++i2)
                    Vhout[jj * k + i2] = W[i2 * wn + j] / denom;
            }
        }
        // S must follow the same descending permutation as the U / Vh columns.
        std::vector<double> sv_out(static_cast<size_t>(r), 0.0);
        for (int64_t jj = 0; jj < r; ++jj) sv_out[static_cast<size_t>(jj)] = sv[static_cast<size_t>(ord[jj])];
        put_batch_block(S, sv_out, bidx, r);
        put_batch_block(U, Uout, bidx, (flip ? m : wm) * r);
        put_batch_block(Vh, Vhout, bidx, r * (flip ? k : wn));
    }
    if (!compute_uv) {
        Tensor zero = Tensor::zeros({}, self.dtype(), self.device());
        return {zero, S, zero};
    }
    return {U, S, Vh};
}

}  // namespace

Tensor cholesky_cpu(const Tensor& self, bool upper) {
    require_float(self, "cholesky");
    const int64_t nd = self.dim();
    if (nd < 2 || self.size(-1) != self.size(-2))
        TP_THROW(RuntimeError, "cholesky: input must be batches of square matrices");
    const int64_t n = self.size(-1);
    const int64_t batch = self.numel() / (n * n);
    if (lapack_available()) {
        Tensor out = detail::contiguous_clone(self);
        cholesky_lapack(self, out, n, batch, upper);
        return out;
    }
    return cholesky_scalar(self, n, batch, upper);
}

Tensor cholesky_inverse_cpu(const Tensor& self, bool upper) {
    // A = L L^T -> A^{-1} = L^{-T} L^{-1}. Input is the factor.
    require_float(self, "cholesky_inverse");
    const int64_t n = self.size(-1);
    const int64_t batch = self.numel() / (n * n);
    if (lapack_available()) {
        return cholesky_inverse_lapack(self, n, batch, upper);
    }
    return cholesky_inverse_scalar(self, n, batch, upper);
}

Tensor cholesky_solve_cpu(const Tensor& self, const Tensor& input2, bool upper) {
    // self = B (..., n, rhs); input2 = factor; solve (L L^T) X = B.
    require_float(self, "cholesky_solve");
    require_float(input2, "cholesky_solve");
    const int64_t n = self.size(-2), rhs = self.size(-1);
    const int64_t batchB = self.numel() / (n * rhs);
    const int64_t batchL = input2.numel() / (n * n);
    if (lapack_available()) {
        return cholesky_solve_lapack(self, input2, n, rhs, batchB, batchL, upper);
    }
    return cholesky_solve_scalar(self, input2, n, rhs, batchB, batchL, upper);
}

std::tuple<Tensor, Tensor> triangular_solve_cpu(const Tensor& self, const Tensor& A,
                                                bool upper, bool transpose,
                                                bool unitriangular) {
    require_float(A, "triangular_solve");
    const int64_t n = A.size(-1);
    const int64_t rhs = self.size(-1);
    const int64_t batchB = self.numel() / (n * rhs);
    const int64_t batchA = A.numel() / (n * n);
    Tensor X;
    if (lapack_available()) {
        X = triangular_solve_lapack(self, A, n, rhs, batchB, batchA, upper,
                                    transpose, unitriangular);
    } else {
        X = triangular_solve_scalar(self, A, n, rhs, batchB, batchA, upper,
                                    transpose, unitriangular);
    }
    return {X, A.clone()};
}

std::tuple<Tensor, Tensor, Tensor> svd_cpu(const Tensor& self, bool some, bool compute_uv) {
    require_float(self, "svd");
    const int64_t m = self.size(-2), k = self.size(-1);
    const int64_t batch = self.numel() / (m * k);
    (void)some;
    if (lapack_available()) {
        // gesdd-backed factorization via the linalg.svd kernel.  The legacy
        // contract returns V (A = U diag(S) V^T), so the Vh factor is
        // transposed before it lands in the third slot.
        if (compute_uv) {
            auto [U, S, Vh] = ops::linalg_svd(self, false, std::optional<std::string>());
            return {U, S, Vh.transpose(-2, -1).contiguous()};
        }
        Tensor S = ops::linalg_svdvals(self, std::optional<std::string>());
        Tensor zero = Tensor::zeros({}, self.dtype(), self.device());
        return {zero, S, zero};
    }
    return svd_scalar(self, m, k, batch, compute_uv);
}

TENSORPLAY_LIBRARY_IMPL(CPU, BatchLinearAlgebra) {
    m.impl("cholesky", cholesky_cpu);
    m.impl("cholesky_inverse", cholesky_inverse_cpu);
    m.impl("cholesky_solve", cholesky_solve_cpu);
    m.impl("triangular_solve", triangular_solve_cpu);
    m.impl("svd", svd_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
