// CPU _spsolve: native sparse direct solver.  A left-looking sparse LU with
// partial pivoting factorizes the CSR matrix in CSC form (P A = L U); the
// triangular factors stay sparse and only the length-n elimination vector
// is dense.  Right-side solves factorize A^T and solve A^T X^T = B^T.

#include "SparseKernels.h"
#include "Exception.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
struct SparseSolveTypeTag { using type = T; };

template <typename F>
void dispatch_dtype(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Float32: f(SparseSolveTypeTag<float>{}); return;
        case DType::Float64: f(SparseSolveTypeTag<double>{}); return;
        case DType::ComplexFloat: f(SparseSolveTypeTag<complex<float>>{}); return;
        case DType::ComplexDouble: f(SparseSolveTypeTag<complex<double>>{}); return;
        default: TP_THROW(TypeError, "sparse solve requires floating or complex floating values");
    }
}

template <typename compute_t>
struct SparseLU {
    int64_t n = 0;
    std::vector<int64_t> L_colptr;
    std::vector<int64_t> L_rowidx;
    std::vector<compute_t> L_val;
    std::vector<int64_t> U_colptr;
    std::vector<int64_t> U_rowidx;
    std::vector<compute_t> U_val;
    std::vector<int64_t> perm;  // PA[i, :] == A[perm[i], :]
};

// Left-looking sparse LU with partial pivoting.  `A` is supplied in CSC
// (column pointers, row indices, values).  Returns the L/U factors and the
// row permutation.  Throws when a zero pivot is encountered.
template <typename compute_t>
SparseLU<compute_t> sparse_lu_factor(const std::vector<int64_t>& colptr,
                                     const std::vector<int64_t>& row,
                                     const std::vector<compute_t>& val,
                                     int64_t n) {
    SparseLU<compute_t> lu;
    lu.n = n;
    lu.L_colptr.assign(n + 1, 0);
    lu.U_colptr.assign(n + 1, 0);
    lu.perm.resize(n);
    for (int64_t i = 0; i < n; ++i) lu.perm[i] = i;
    // inv_perm[original_row] is the current position of that row; it keeps
    // the column scatter consistent after row swaps.
    std::vector<int64_t> inv_perm(n);
    for (int64_t i = 0; i < n; ++i) inv_perm[i] = i;

    std::vector<compute_t> x(n, compute_t(0));
    std::vector<int64_t> nz_rows;
    std::vector<uint8_t> in_list(static_cast<size_t>(n), 0);

    for (int64_t j = 0; j < n; ++j) {
        // Gather column j of A into the elimination vector, placing each
        // original row at its current permuted position.
        for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) {
            const int64_t pos = inv_perm[row[p]];
            if (x[pos] == compute_t(0) &&
                !in_list[static_cast<size_t>(pos)]) {
                in_list[static_cast<size_t>(pos)] = 1;
                nz_rows.push_back(pos);
            }
            x[pos] = val[p];
        }

        // Sparse forward solve: update x[i] -= L[i][k] * x[k] for k < j.
        // Processing k in increasing order keeps the elimination correct as
        // new nonzeros in earlier rows propagate to later rows.
        for (int64_t k = 0; k < j; ++k) {
            const compute_t xk = x[k];
            if (xk == compute_t(0)) continue;
            for (int64_t q = lu.L_colptr[k]; q < lu.L_colptr[k + 1]; ++q) {
                const int64_t i = lu.L_rowidx[q];
                const compute_t newv = x[i] - lu.L_val[q] * xk;
                if (x[i] == compute_t(0) && newv != compute_t(0) &&
                    !in_list[static_cast<size_t>(i)]) {
                    in_list[static_cast<size_t>(i)] = 1;
                    nz_rows.push_back(i);
                }
                x[i] = newv;
            }
        }

        // Partial pivoting: largest magnitude in rows >= j.
        int64_t piv_row = -1;
        double best_abs = 0.0;
        for (int64_t i : nz_rows) {
            if (i < j) continue;
            const double absv = std::abs(x[i]);
            if (absv > best_abs) {
                best_abs = absv;
                piv_row = i;
            }
        }
        if (piv_row < 0) {
            TP_THROW(RuntimeError, "_spsolve(): matrix is singular");
        }
        if (piv_row != j) {
            std::swap(lu.perm[j], lu.perm[static_cast<size_t>(piv_row)]);
            inv_perm[lu.perm[j]] = j;
            inv_perm[lu.perm[static_cast<size_t>(piv_row)]] = piv_row;
            std::swap(x[j], x[static_cast<size_t>(piv_row)]);
            // The swapped rows carry their multipliers from earlier L
            // columns along with them, so exchange those entries too.
            for (int64_t k = 0; k < j; ++k) {
                int64_t idx_j = -1;
                int64_t idx_piv = -1;
                for (int64_t q = lu.L_colptr[k]; q < lu.L_colptr[k + 1];
                     ++q) {
                    if (lu.L_rowidx[q] == j) {
                        idx_j = q;
                    } else if (lu.L_rowidx[q] == piv_row) {
                        idx_piv = q;
                    }
                }
                if (idx_j >= 0 && idx_piv >= 0) {
                    std::swap(lu.L_val[idx_j], lu.L_val[idx_piv]);
                } else if (idx_j >= 0) {
                    lu.L_rowidx[idx_j] = piv_row;
                } else if (idx_piv >= 0) {
                    lu.L_rowidx[idx_piv] = j;
                }
            }
        }
        const compute_t pivot_val = x[j];

        // Store column j of U (entries at rows <= j).
        lu.U_colptr[j] = static_cast<int64_t>(lu.U_rowidx.size());
        for (int64_t i : nz_rows) {
            if (i > j || x[i] == compute_t(0)) continue;
            lu.U_rowidx.push_back(i);
            lu.U_val.push_back(x[i]);
        }

        // Store column j of L (strictly lower entries; unit diagonal is
        // implicit).
        lu.L_colptr[j] = static_cast<int64_t>(lu.L_rowidx.size());
        for (int64_t i : nz_rows) {
            if (i <= j || x[i] == compute_t(0)) continue;
            lu.L_rowidx.push_back(i);
            lu.L_val.push_back(x[i] / pivot_val);
        }
        // Close column j so the next forward solve can see it.
        lu.L_colptr[j + 1] = static_cast<int64_t>(lu.L_rowidx.size());

        // Reset the elimination vector.
        for (int64_t i : nz_rows) {
            x[i] = compute_t(0);
            in_list[static_cast<size_t>(i)] = 0;
        }
        nz_rows.clear();
    }
    lu.L_colptr[n] = static_cast<int64_t>(lu.L_rowidx.size());
    lu.U_colptr[n] = static_cast<int64_t>(lu.U_rowidx.size());
    return lu;
}

// Solves A X = B for one RHS column b (length n), writing into x.
template <typename compute_t>
void lu_solve_one(const SparseLU<compute_t>& lu,
                  const compute_t* b, compute_t* x) {
    const int64_t n = lu.n;
    std::vector<compute_t> y(static_cast<size_t>(n));
    // Forward substitution: L y = P b.
    for (int64_t i = 0; i < n; ++i) {
        y[static_cast<size_t>(i)] = b[lu.perm[static_cast<size_t>(i)]];
    }
    for (int64_t k = 0; k < n; ++k) {
        const compute_t yk = y[static_cast<size_t>(k)];
        for (int64_t q = lu.L_colptr[k]; q < lu.L_colptr[k + 1]; ++q) {
            y[static_cast<size_t>(lu.L_rowidx[q])] -= lu.L_val[q] * yk;
        }
    }
    // Back substitution: U x = y.  U is stored by columns, so each
    // column is applied from the bottom up: normalize the diagonal, then
    // eliminate its contribution from the rows above.
    for (int64_t i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = y[static_cast<size_t>(i)];
    }
    for (int64_t j = n - 1; j >= 0; --j) {
        compute_t diag = compute_t(0);
        for (int64_t q = lu.U_colptr[j]; q < lu.U_colptr[j + 1]; ++q) {
            if (lu.U_rowidx[q] == j) {
                diag = lu.U_val[q];
                break;
            }
        }
        x[static_cast<size_t>(j)] /= diag;
        const compute_t xj = x[static_cast<size_t>(j)];
        for (int64_t q = lu.U_colptr[j]; q < lu.U_colptr[j + 1]; ++q) {
            const int64_t i = lu.U_rowidx[q];
            if (i < j) {
                x[static_cast<size_t>(i)] -= lu.U_val[q] * xj;
            }
        }
    }
}

// Converts a CSR matrix (crow/col/values) into CSC form for column access.
template <typename compute_t>
void csr_to_csc(const std::vector<int64_t>& crow,
                const std::vector<int64_t>& col,
                const std::vector<compute_t>& val, int64_t n,
                std::vector<int64_t>& colptr, std::vector<int64_t>& row,
                std::vector<compute_t>& csc_val) {
    const int64_t nnz = static_cast<int64_t>(col.size());
    colptr.assign(n + 1, 0);
    for (int64_t e = 0; e < nnz; ++e) ++colptr[col[static_cast<size_t>(e)] + 1];
    for (int64_t i = 0; i < n; ++i) colptr[i + 1] += colptr[i];
    row.resize(static_cast<size_t>(nnz));
    csc_val.resize(static_cast<size_t>(nnz));
    std::vector<int64_t> next = colptr;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t q = crow[static_cast<size_t>(i)];
             q < crow[static_cast<size_t>(i + 1)]; ++q) {
            const int64_t c = col[static_cast<size_t>(q)];
            const int64_t pos = next[static_cast<size_t>(c)]++;
            row[static_cast<size_t>(pos)] = i;
            csc_val[static_cast<size_t>(pos)] = val[static_cast<size_t>(q)];
        }
    }
}

// CSC of the transpose: columns are the original rows, rows the original
// columns.
template <typename compute_t>
void csr_to_csc_transpose(const std::vector<int64_t>& crow,
                          const std::vector<int64_t>& col,
                          const std::vector<compute_t>& val, int64_t n,
                          std::vector<int64_t>& colptr,
                          std::vector<int64_t>& row,
                          std::vector<compute_t>& csc_val) {
    const int64_t nnz = static_cast<int64_t>(col.size());
    colptr.assign(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) {
        colptr[i + 1] = colptr[i] + (crow[static_cast<size_t>(i + 1)] -
                                     crow[static_cast<size_t>(i)]);
    }
    row.resize(static_cast<size_t>(nnz));
    csc_val.resize(static_cast<size_t>(nnz));
    std::vector<int64_t> next = colptr;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t q = crow[static_cast<size_t>(i)];
             q < crow[static_cast<size_t>(i + 1)]; ++q) {
            const int64_t pos = next[static_cast<size_t>(i)]++;
            row[static_cast<size_t>(pos)] = col[static_cast<size_t>(q)];
            csc_val[static_cast<size_t>(pos)] = val[static_cast<size_t>(q)];
        }
    }
}

} // namespace

Tensor _spsolve_cpu(const Tensor& A, const Tensor& B, bool left) {
    if (!A.is_sparse_csr() || A.dim() != 2 || A.size(0) != A.size(1)) {
        TP_THROW(RuntimeError,
                 "_spsolve(): expected a square 2-D sparse CSR matrix A");
    }
    if (B.is_sparse()) {
        TP_THROW(RuntimeError,
                 "_spsolve(): expected B to be a strided tensor");
    }
    if (B.dtype() != A.dtype()) {
        TP_THROW(TypeError, "_spsolve(): A and B must share one dtype");
    }
    if (B.device() != A.device()) {
        TP_THROW(DeviceMismatchError,
                 "_spsolve(): A and B must be on the same device");
    }
    if (B.dim() != 1 && B.dim() != 2) {
        TP_THROW(RuntimeError, "_spsolve(): B must be 1-D or 2-D");
    }
    const int64_t n = A.size(0);
    if ((left && B.size(0) != n) || (!left && B.dim() == 2 && B.size(1) != n) ||
        (!left && B.dim() == 1 && B.size(0) != n)) {
        TP_THROW(RuntimeError,
                 "_spsolve(): B has an incompatible dimension");
    }

    const int64_t nrhs = B.dim() == 1 ? 1 : B.size(left ? 1 : 0);
    Tensor B_work = B.dim() == 1 ? B.contiguous().unsqueeze(1)
                               : B.t().contiguous();
    Tensor Y = Tensor::empty({nrhs, n}, A.dtype(), A.device());

    Tensor crow_t = A._crow_indices().contiguous();
    Tensor col_t = A._col_indices().contiguous();
    Tensor val_t = A._values().contiguous();
    const int64_t* crow = crow_t.data_ptr<int64_t>();
    const int64_t* col = col_t.data_ptr<int64_t>();

    dispatch_dtype(A.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        using compute_t =
            std::conditional_t<std::is_same_v<scalar_t, float> ||
                                   std::is_same_v<scalar_t, double>,
                               double,
                               std::complex<double>>;
        const scalar_t* aval = val_t.data_ptr<scalar_t>();
        std::vector<compute_t> a_val(static_cast<size_t>(A._nnz()));
        for (int64_t e = 0; e < A._nnz(); ++e) {
            a_val[static_cast<size_t>(e)] = static_cast<compute_t>(aval[e]);
        }

        std::vector<int64_t> colptr;
        std::vector<int64_t> row;
        std::vector<compute_t> csc_val;
        if (left) {
            csr_to_csc(std::vector<int64_t>(crow, crow + n + 1),
                       std::vector<int64_t>(col, col + A._nnz()), a_val, n,
                       colptr, row, csc_val);
        } else {
            csr_to_csc_transpose(std::vector<int64_t>(crow, crow + n + 1),
                                 std::vector<int64_t>(col, col + A._nnz()),
                                 a_val, n, colptr, row, csc_val);
        }

        SparseLU<compute_t> lu = sparse_lu_factor(colptr, row, csc_val, n);

        const scalar_t* b_data = B_work.const_data_ptr<scalar_t>();
        scalar_t* y_data = Y.data_ptr<scalar_t>();
        std::vector<compute_t> b_col(static_cast<size_t>(n));
        std::vector<compute_t> x_col(static_cast<size_t>(n));
        for (int64_t j = 0; j < nrhs; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                b_col[static_cast<size_t>(i)] =
                    static_cast<compute_t>(b_data[j * n + i]);
            }
            lu_solve_one(lu, b_col.data(), x_col.data());
            for (int64_t i = 0; i < n; ++i) {
                y_data[j * n + i] = static_cast<scalar_t>(x_col[static_cast<size_t>(i)]);
            }
        }
    });

    if (B.dim() == 1) {
        return Y.squeeze(0);
    }
    if (left) {
        return Y.t();
    }
    return Y;
}

} // namespace cpu
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(SparseCPU, SparseSolveKernels) {
    using namespace tensorplay::cpu;
    m.impl("_spsolve", _spsolve_cpu);
}
