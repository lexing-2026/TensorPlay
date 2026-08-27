// CPU linalg kernels — ported from third_party/pytorch
// aten/src/ATen/native/BatchLinearAlgebraKernel.cpp (and the composite
// wrappers in BatchLinearAlgebra.cpp / LinearAlgebra.cpp).
//
// Torch calls LAPACK through at::lapack; here the same routines come from a
// runtime-resolved ILP64 LAPACK (see cpu/Lapack.h).  All matrices follow the
// Fortran (batched column-major) layout that LAPACK expects, produced with
// clone_batched_column_major / empty_column_major just like torch's
// cloneBatchedColumnMajor.  Complex inputs are rejected until the complex
// paths are ported.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"
#include "LinearAlgebraNames.h"
#include "cpu/Lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

// ---------------------------------------------------------------- helpers --

template <class Kernel>
decltype(auto) run_real(DType dt, Kernel&& k) {
    switch (dt) {
        case DType::Float32:
            return k(static_cast<float*>(nullptr));
        case DType::Float64:
            return k(static_cast<double*>(nullptr));
        default:
            TP_THROW(NotImplementedError,
                     "unsupported dtype ", pretty_dtype_name(dt),
                     " for torch.linalg on CPU (only float32/float64 are implemented)");
    }
}

inline void check_is_matrix(const Tensor& A, const char* fn, const char* arg = "A") {
    if (A.dim() < 2) {
        TP_THROW(RuntimeError, fn, ": The input tensor ", arg,
                 " must have at least 2 dimensions.");
    }
}

inline void square_check_inputs(const Tensor& A, const char* fn, const char* arg = "A") {
    check_is_matrix(A, fn, arg);
    if (A.size(-1) != A.size(-2)) {
        TP_THROW(RuntimeError, fn, ": ", arg,
                 " must be batches of square matrices, but they are ",
                 A.size(-2), " by ", A.size(-1), " matrices");
    }
}

inline void check_inputs_solver(const Tensor& A, const Tensor& B, bool left, const char* fn) {
    square_check_inputs(A, fn, "A");
    check_is_matrix(B, fn, "B");
    if (!(left ? A.size(-2) == B.size(-2) : A.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, fn, ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", A.size(-2), "x", A.size(-1), " and ",
                 B.size(-2), "x", B.size(-1), ")");
    }
}

std::vector<int64_t> batch_shape_of(const Tensor& t) {
    const Size t_shape = t.shape();
    return std::vector<int64_t>(t_shape.begin(), t_shape.end() - 2);
}

int64_t matrix_stride_of(const Tensor& t) { return t.size(-1) * t.size(-2); }

int64_t batch_count_of(const Tensor& t) {
    const int64_t plane = matrix_stride_of(t);
    return plane == 0 ? 0 : t.numel() / plane;
}

// torch::cloneBatchedColumnMajor — logical shape unchanged, Fortran-contiguous
// memory so LAPACK can work in place with lda = size(-2).
Tensor clone_batched_column_major(const Tensor& src) {
    auto result = src.transpose(-2, -1).clone(static_cast<int64_t>(MemoryFormat::Contiguous));
    return result.transpose(-2, -1);
}

Tensor empty_column_major(std::vector<int64_t> shape, DType dt, Device dev) {
    std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
    return Tensor::empty(shape, dt, dev).transpose(-2, -1);
}

std::vector<int64_t> broadcast_batch(const Tensor& a, const Tensor& b) {
    const auto as = batch_shape_of(a);
    const auto bs = batch_shape_of(b);
    const size_t rank = std::max(as.size(), bs.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - as.size() ? 1 : as[i - (rank - as.size())];
        const int64_t db = i < rank - bs.size() ? 1 : bs[i - (rank - bs.size())];
        if (da != db && da != 1 && db != 1) {
            TP_THROW(RuntimeError, "The size of tensor a (", da,
                     ") must match the size of tensor b (", db,
                     ") at non-singleton dimension ", i);
        }
        out[i] = std::max(da, db);
    }
    return out;
}

Tensor expand_to_batch(const Tensor& t, const std::vector<int64_t>& batch) {
    std::vector<int64_t> shape = batch;
    shape.push_back(t.size(-2));
    shape.push_back(t.size(-1));
    return t.expand(shape);
}

// Port of at::native::_linalg_check_errors (BatchLinearAlgebra.cpp:1610).
void linalg_check_errors(const Tensor& infos, std::string_view api_name, bool is_matrix) {
    if (!infos.any().item<bool>()) return;

    int64_t info = 0;
    std::string batch_str;
    if (is_matrix) {
        info = infos.item<int64_t>();
    } else {
        const auto* ptr = infos.data_ptr<int32_t>();
        const int64_t n = infos.numel();
        for (int64_t i = 0; i < n; ++i) {
            if (ptr[i] != 0) {
                info = ptr[i];
                batch_str = ": (Batch element " + std::to_string(i) + ")";
                break;
            }
        }
    }

    if (info < 0) {
        if (api_name.find("svd") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The algorithm failed to converge because the input matrix contained non-finite values.");
        }
        TP_THROW(RuntimeError, api_name, batch_str,
                 ": Argument ", -info, " has illegal value. Most certainly there is a bug in the implementation calling the backend library.");
    } else if (info > 0) {
        if (api_name.find("inv") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
        } else if (api_name.find("solve") != std::string_view::npos &&
                   api_name.find("lu_solve") == std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The solver failed because the input matrix is singular.");
        } else if (api_name.find("cholesky") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
        } else if (api_name.find("svd") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
        } else if (api_name.find("eig") != std::string_view::npos ||
                   api_name.find("syevd") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
        } else if (api_name.find("lstsq") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
        } else if (api_name.find("lu_factor") != std::string_view::npos) {
            TP_THROW(RuntimeError, api_name, batch_str,
                     ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
                     "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
                     "linalg.lu_factor_ex(A, pivot)");
        } else {
            TP_THROW(RuntimeError, api_name, batch_str, ": failed with error code ", info, ".");
        }
    }
}

Tensor empty_info_like(const Tensor& proto, const std::vector<int64_t>& batch) {
    return Tensor::empty(batch, DType::Int32, proto.device());
}

Tensor empty_pivots(const Tensor& proto, const std::vector<int64_t>& batch, int64_t k) {
    std::vector<int64_t> shape = batch;
    shape.push_back(k);
    return Tensor::zeros(shape, DType::Int32, proto.device());
}

// Defined further below (geev section); used by pack_complex_outputs.
template <typename scalar_t, typename cplx_t>
void make_complex_eigenvectors(const Tensor& result, const Tensor& complex_values,
                               const Tensor& real_vectors);

std::vector<int64_t> repeat_batch(std::vector<int64_t> batch, int64_t last) {
    batch.push_back(last);
    return batch;
}

std::vector<int64_t> cat_batch(const std::vector<int64_t>& batch,
                               std::vector<int64_t> tail) {
    std::vector<int64_t> out = batch;
    for (int64_t v : tail) out.push_back(v);
    return out;
}

int64_t linear_batch_size(const std::vector<int64_t>& batch) {
    return static_cast<int64_t>(std::accumulate(batch.begin(), batch.end(), int64_t{1},
                                                std::multiplies<int64_t>()));
}

// Pack GEEV's real outputs into complex tensors: eigenvalues wr + i*wi and,
// when requested, the conjugate-pair eigenvector expansion.
void pack_complex_outputs(bool is_float_input, const Tensor& wr, const Tensor& wi,
                          const Tensor& rvectors, Tensor& values, Tensor& eigvecs,
                          bool compute_eigenvectors) {
    if (is_float_input) {
        auto* v = values.data_ptr<std::complex<float>>();
        const auto* r = wr.data_ptr<float>();
        const auto* im = wi.data_ptr<float>();
        for (int64_t i = 0; i < wr.numel(); ++i) v[i] = std::complex<float>(r[i], im[i]);
        if (compute_eigenvectors)
            make_complex_eigenvectors<float, std::complex<float>>(eigvecs, values, rvectors);
    } else {
        auto* v = values.data_ptr<std::complex<double>>();
        const auto* r = wr.data_ptr<double>();
        const auto* im = wi.data_ptr<double>();
        for (int64_t i = 0; i < wr.numel(); ++i) v[i] = std::complex<double>(r[i], im[i]);
        if (compute_eigenvectors)
            make_complex_eigenvectors<double, std::complex<double>>(eigvecs, values, rvectors);
    }
}

// ------------------------------------------------------------------- getrf --

template <typename scalar_t>
void apply_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos) {
    auto* input_data = input.data_ptr<scalar_t>();
    auto* pivots_data = pivots.data_ptr<int32_t>();
    auto* infos_data = infos.data_ptr<int32_t>();
    const int64_t input_matrix_stride = matrix_stride_of(input);
    const int64_t pivots_stride = pivots.size(-1);
    const int64_t batch_size = batch_count_of(input);
    const int64_t m = input.size(-2);
    const int64_t n = input.size(-1);
    const int64_t leading_dimension = std::max<int64_t>(1, m);
    constexpr bool is_float = std::is_same_v<scalar_t, float>;

    // Parallel-grain heuristic copied from torch (PR #93037 discussion).
    const int64_t matrix_rank = std::min(m, n);
    const int64_t chunk_size_per_thread = static_cast<int64_t>(
        std::min(1.0, 3200.0 / (static_cast<double>(matrix_rank * matrix_rank * matrix_rank))));
    const int64_t grain_size = chunk_size_per_thread * static_cast<int64_t>(1);
    parallel_for(0, batch_size > 0 ? batch_size : 1, grain_size,
                 [&](int64_t begin, int64_t end) {
                     for (int64_t i = begin; i < end && i < batch_size; ++i) {
                         scalar_t* a = &input_data[i * input_matrix_stride];
                         int32_t* piv = &pivots_data[i * pivots_stride];
                         std::vector<int64_t> ipiv(pivots_stride);
                         int64_t info;
                         if constexpr (is_float) {
                             info = lapack_sgetrf(m, n, a, leading_dimension, ipiv.data());
                         } else {
                             info = lapack_dgetrf(m, n, a, leading_dimension, ipiv.data());
                         }
                         for (int64_t j = 0; j < pivots_stride; ++j) piv[j] = static_cast<int32_t>(ipiv[j]);
                         infos_data[i] = static_cast<int32_t>(info);
                     }
                 });
}

std::tuple<Tensor, Tensor, Tensor> lu_factor_ex_impl(
        const Tensor& A, bool /*pivot*/, bool check_errors, const char* api_name) {
    require_lapack(api_name);
    square_check_inputs(A, "linalg.lu_factor_ex");
    const auto batch = batch_shape_of(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    Tensor LU = clone_batched_column_major(A);
    Tensor pivots = empty_pivots(A, batch, k);
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_lu_factor<T>(LU, pivots, info);
    });
    if (check_errors) linalg_check_errors(info, api_name, A.dim() == 2);
    return {LU.contiguous(), pivots, info};
}

// ------------------------------------------------------- det / slogdet ------

// As P is a permutation matrix: det(P) = (-1)^{#swaps}.  Port of lu_det_P.
int64_t lu_perm_sign(const int32_t* pivots, int64_t k) {
    int64_t parity = 0;
    for (int64_t i = 0; i < k; ++i) {
        if (pivots[i] - 1 != static_cast<int32_t>(i)) ++parity;
    }
    return (parity % 2 == 0) ? 1 : -1;
}

Tensor linalg_det_kernel(const Tensor& A) {
    require_lapack("linalg.det");
    square_check_inputs(A, "linalg.det");
    // det(A^T) = det(A): reuse the contiguous layout as the column-major copy.
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    auto [LU, pivots, info] = lu_factor_ex_impl(src, true, false, "linalg.lu_factor");
    (void)info;
    const int64_t batch_size = batch_count_of(LU);
    const int64_t n = LU.size(-1);
    Tensor result = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* lu = LU.data_ptr<T>();
        auto* out = result.data_ptr<T>();
        const auto* piv = pivots.data_ptr<int32_t>();
        const int64_t ms = matrix_stride_of(LU);
        for (int64_t b = 0; b < batch_size; ++b) {
            T det = T(1);
            for (int64_t i = 0; i < n; ++i) det *= lu[b * ms + i * n + i];
            out[b] = det * static_cast<T>(lu_perm_sign(&piv[b * n], n));
        }
    });
    return result;
}

std::tuple<Tensor, Tensor> linalg_slogdet_kernel(const Tensor& A) {
    require_lapack("linalg.slogdet");
    square_check_inputs(A, "linalg.slogdet");
    Tensor work = A.is_contiguous() ? A.transpose(-2, -1) : A;  // det(A^T) = det(A)
    auto [LU, pivots, info] = lu_factor_ex_impl(work, true, false, "linalg.lu_factor");
    (void)info;
    const int64_t batch_size = batch_count_of(LU);
    const int64_t n = LU.size(-1);
    const auto batch = batch_shape_of(A);
    Tensor sign = Tensor::empty(batch, A.dtype(), A.device());
    Tensor logabsdet = Tensor::empty(batch, A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* lu = LU.data_ptr<T>();
        auto* s_out = sign.data_ptr<T>();
        auto* l_out = logabsdet.data_ptr<T>();
        const auto* piv = pivots.data_ptr<int32_t>();
        const int64_t ms = matrix_stride_of(LU);
        const T neg_inf = -std::numeric_limits<T>::infinity();
        for (int64_t b = 0; b < batch_size; ++b) {
            T logdet = T(0);
            T sgn_prod = T(1);
            bool singular = false;
            for (int64_t i = 0; i < n; ++i) {
                const T d = lu[b * ms + i * n + i];
                if (d == T(0)) { singular = true; break; }
                logdet += std::log(std::abs(d));
                sgn_prod *= (d < T(0)) ? T(-1) : T(1);
            }
            const int64_t perm_sign = lu_perm_sign(&piv[b * n], n);
            if (singular) {
                s_out[b] = T(0);
                l_out[b] = neg_inf;
            } else {
                s_out[b] = sgn_prod * static_cast<T>(perm_sign);
                l_out[b] = logdet;
            }
        }
    });
    return {sign, logabsdet};
}

// ------------------------------------------------------------- getrs solve --

// Core: solve op(A) X = B given LU/ipiv of A.  B is overwritten in place and
// must be column-major with ldb = B.size(-2).  Mirrors apply_lu_solve without
// broadcasting (broadcasting resolved by callers).
template <typename scalar_t>
void getrs_inplace(char trans, const Tensor& LU, const int32_t* pivots,
                   int64_t pivots_stride, Tensor& B, int64_t b_offset) {
    auto* lu = LU.data_ptr<scalar_t>();
    auto* b = B.data_ptr<scalar_t>() + b_offset;
    const int64_t n = LU.size(-2);
    const int64_t nrhs = B.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);
    const int64_t ldb = std::max<int64_t>(1, B.size(-2));
    std::vector<int64_t> ipiv(pivots_stride);
    for (int64_t j = 0; j < pivots_stride; ++j) ipiv[j] = pivots[j];
    int64_t info;
    if constexpr (std::is_same_v<scalar_t, float>) {
        info = lapack_sgetrs(trans, n, nrhs, lu, lda, ipiv.data(), b, ldb);
    } else {
        info = lapack_dgetrs(trans, n, nrhs, lu, lda, ipiv.data(), b, ldb);
    }
    (void)info;  // only reports bad arguments
}

std::tuple<Tensor, Tensor> linalg_solve_ex_kernel(
        const Tensor& A, const Tensor& B, bool left, bool check_errors) {
    const char* api = "linalg.solve";
    require_lapack(api);
    check_inputs_solver(A, B, left, api);
    if (left) {
        const auto batch = broadcast_batch(A, B);
        Tensor LU_work;
        {
            const Tensor A_exp = expand_to_batch(A, batch);
            LU_work = clone_batched_column_major(A_exp);
        }
        const int64_t n = A.size(-2);
        Tensor pivots = empty_pivots(A, batch, n);
        Tensor info = empty_info_like(A, batch);
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_lu_factor<T>(LU_work, pivots, info);
        });
        Tensor B_work = clone_batched_column_major(expand_to_batch(B, batch));
        const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(std::accumulate(
                                                  batch.begin(), batch.end(), int64_t{1},
                                                  std::multiplies<int64_t>())));
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            const int64_t lu_ms = matrix_stride_of(LU_work);
            const int64_t b_ms = matrix_stride_of(B_work);
            const int64_t piv_stride = pivots.dim() > 1 ? pivots.size(-1) : 0;
            const auto* piv = pivots.data_ptr<int32_t>();
            const auto* inf = info.data_ptr<int32_t>();
            for (int64_t i = 0; i < bs; ++i) {
                if (inf[i] != 0) continue;  // singular: skip, report below
                getrs_inplace<T>('N', LU_work, &piv[i * piv_stride], pivots.size(-1),
                                 B_work, i * b_ms);
            }
        });
        Tensor result = B_work.contiguous();
        if (check_errors) linalg_check_errors(info, api, A.dim() == 2 && B.dim() == 2);
        return {result, info};
    }
    // X A = B  <=>  A^T X^T = B^T (real dtypes only).
    auto [xt, info] = linalg_solve_ex_kernel(A.transpose(-2, -1), B.transpose(-2, -1), true, false);
    Tensor result = xt.transpose(-2, -1).contiguous();
    if (check_errors) linalg_check_errors(info, api, A.dim() == 2 && B.dim() == 2);
    return {result, info};
}

Tensor linalg_solve_kernel(const Tensor& A, const Tensor& B, bool left) {
    auto [result, info] = linalg_solve_ex_kernel(A, B, left, false);
    linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
    return result;
}

// torch::linalg_inv_ex: solve A X = I in place (torch composes this through
// linalg_solve_ex_out with result pre-filled with the identity).
std::tuple<Tensor, Tensor> linalg_inv_ex_kernel(const Tensor& A, bool check_errors) {
    // Identity RHS has the FULL shape of A (torch fills (..., n, n)); using
    // batch_shape_of here collapsed 2-D inputs to a 0-D scalar and made
    // linalg.inv reject its own RHS.
    Tensor identity = Tensor::empty(static_cast<std::vector<int64_t>>(A.shape()),
                                    A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* p = identity.data_ptr<T>();
        const int64_t n = A.size(-1);
        const int64_t ms = n * n;
        const int64_t bs = std::max<int64_t>(1, batch_count_of(A));
        for (int64_t b = 0; b < bs; ++b) {
            std::memset(p + b * ms, 0, sizeof(T) * ms);
            for (int64_t i = 0; i < n; ++i) p[b * ms + i * n + i] = T(1);
        }
    });
    auto [inv, info] = linalg_solve_ex_kernel(A, identity, /*left=*/true, false);
    if (check_errors) linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
    return {inv, info};
}

Tensor linalg_inv_kernel(const Tensor& A) {
    auto [result, info] = linalg_inv_ex_kernel(A, false);
    linalg_check_errors(info, "linalg.inv", A.dim() == 2);
    return result;
}

// ------------------------------------------------------------------ potrf --

template <typename scalar_t>
void apply_cholesky(const Tensor& input, const Tensor& info, bool upper) {
    auto* input_data = input.data_ptr<scalar_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t input_matrix_stride = matrix_stride_of(input);
    const int64_t batch_size = batch_count_of(input);
    const int64_t n = input.size(-2);
    const int64_t lda = std::max<int64_t>(1, n);
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &input_data[i * input_matrix_stride];
        const char uplo = upper ? 'U' : 'L';
        int64_t err;
        if constexpr (is_float) {
            err = lapack_spotrf(uplo, n, a, lda);
        } else {
            err = lapack_dpotrf(uplo, n, a, lda);
        }
        // torch parity: the triangle opposite `uplo` is zeroed in the result
        // (LAPACK leaves the input's untouched entries there).
        if (err == 0) {
            for (int64_t r = 0; r < n; ++r) {
                for (int64_t c = 0; c < n; ++c) {
                    const bool strictly_opposite =
                        upper ? (r > c) : (r < c);
                    if (strictly_opposite) a[c * n + r] = scalar_t(0);
                }
            }
        }
        info_data[i] = static_cast<int32_t>(err);
    }
}

std::tuple<Tensor, Tensor> linalg_cholesky_ex_kernel(const Tensor& A, bool upper, bool check_errors) {
    const char* api = check_errors ? "linalg.cholesky_ex" : "linalg.cholesky";
    require_lapack(api);
    square_check_inputs(A, api);
    Tensor L = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_cholesky<T>(L, info, upper);
    });
    if (check_errors) linalg_check_errors(info, api, A.dim() == 2);
    return {L.contiguous(), info};
}

Tensor linalg_cholesky_kernel(const Tensor& A, bool upper) {
    auto [L, info] = linalg_cholesky_ex_kernel(A, upper, false);
    linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
    return L;
}

// --------------------------------------------------------- triangular solve

Tensor linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                      bool upper, bool left, bool unitriangular) {
    const char* api = "linalg.solve_triangular";
    require_lapack(api);
    check_is_matrix(A, api, "A");
    check_is_matrix(B, api, "B");
    const auto batch = broadcast_batch(A, B);
    Tensor B_work = clone_batched_column_major(expand_to_batch(B, batch));
    Tensor A_work = clone_batched_column_major(expand_to_batch(A, batch));
    const char side = left ? 'L' : 'R';
    const char uplo = upper ? 'U' : 'L';
    const char diag = unitriangular ? 'U' : 'N';
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(std::accumulate(
                                            batch.begin(), batch.end(), int64_t{1},
                                            std::multiplies<int64_t>())));
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* a = A_work.data_ptr<T>();
        auto* b = B_work.data_ptr<T>();
        const int64_t a_ms = matrix_stride_of(A_work);
        const int64_t b_ms = matrix_stride_of(B_work);
        // This allows to pass rectangular A and B when left = True.
        const int64_t m = left ? A.size(-1) : B.size(-2);
        const int64_t n = B.size(-1);
        const int64_t lda = std::max<int64_t>(1, A.size(-2));
        const int64_t ldb = std::max<int64_t>(1, B.size(-2));
        for (int64_t i = 0; i < bs; ++i) {
            int64_t info;
            if constexpr (std::is_same_v<T, float>) {
                info = lapack_strtrs(side, uplo, 'N', diag, m, n, &a[i * a_ms], lda,
                                     &b[i * b_ms], ldb);
            } else {
                info = lapack_dtrtrs(side, uplo, 'N', diag, m, n, &a[i * a_ms], lda,
                                     &b[i * b_ms], ldb);
            }
            (void)info;
        }
    });
    return B_work.contiguous();
}

// -------------------------------------------------------------- lu_solve ---

Tensor linalg_lu_solve_kernel(const Tensor& LU, const Tensor& pivots,
                              const Tensor& B, bool left, bool adjoint) {
    const char* api = "linalg.lu_solve";
    require_lapack(api);
    square_check_inputs(LU, api, "LU");
    check_is_matrix(B, api, "B");
    // Sanity checks copied from torch (lu_solve_kernel).
    {
        const int64_t np = pivots.numel();
        const auto* pv = pivots.data_ptr<int32_t>();
        for (int64_t i = 0; i < np; ++i) {
            if (pv[i] <= 0) {
                TP_THROW(RuntimeError, "Pivots given to lu_solve must all be greater or equal to 1. "
                                       "Did you properly pass the result of lu_factor?");
            }
            if (pv[i] > LU.size(-2)) {
                TP_THROW(RuntimeError, "Pivots given to lu_solve must all be smaller or equal to LU.size(-2). "
                                       "Did you properly pass the result of lu_factor?");
            }
        }
    }
    if (!(left ? LU.size(-2) == B.size(-2) : LU.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of LU and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", LU.size(-2), "x", LU.size(-1), " and ",
                 B.size(-2), "x", B.size(-1), ")");
    }
    const auto batch = broadcast_batch(LU, B);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(LU, batch));
    std::vector<int64_t> piv_shape = batch;
    piv_shape.push_back(pivots.size(-1));
    Tensor piv_exp = pivots.expand(piv_shape).contiguous();
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(std::accumulate(
                                            batch.begin(), batch.end(), int64_t{1},
                                            std::multiplies<int64_t>())));
    // Effective getrs transposition:
    //   left,  !adj -> 'N' ; left,  adj -> 'T'
    //   !left, !adj -> X A = B <=> A^T X^T = B^T -> 'T' on the transposed RHS
    //   !left,  adj -> X A^T = B <=> A X^T = B^T -> 'N' on the transposed RHS
    const bool rhs_transposed = !left;
    Tensor work_cm;
    if (rhs_transposed) {
        // Logical (... , n, r) column-major copy of B^T.
        work_cm = clone_batched_column_major(expand_to_batch(B, batch).transpose(-2, -1));
    } else {
        work_cm = clone_batched_column_major(expand_to_batch(B, batch));
    }
    const char trans = (left == adjoint) ? 'N' : 'T';
    run_real(LU.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* piv = piv_exp.data_ptr<int32_t>();
        const int64_t piv_stride = pivots.size(-1);
        const int64_t lu_ms = matrix_stride_of(LU_work);
        const int64_t rhs_ms = matrix_stride_of(work_cm);
        for (int64_t i = 0; i < bs; ++i) {
            getrs_inplace<T>(trans, LU_work, &piv[i * piv_stride], pivots.size(-1),
                             work_cm, i * rhs_ms);
        }
    });
    if (rhs_transposed) {
        return work_cm.contiguous().transpose(-2, -1).contiguous();
    }
    return work_cm.contiguous();
}

// ------------------------------------------------------------------ syevd --

template <typename scalar_t>
void apply_syevd(const Tensor& vectors, const Tensor& values, const Tensor& infos,
                 bool upper, bool compute_eigenvectors) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    const char uplo = upper ? 'U' : 'L';
    const char jobz = compute_eigenvectors ? 'V' : 'N';
    auto* vectors_data = vectors.data_ptr<scalar_t>();
    auto* values_data = values.data_ptr<scalar_t>();
    auto* infos_data = infos.data_ptr<int32_t>();
    const int64_t vectors_stride = matrix_stride_of(vectors);
    const int64_t values_stride = values.size(-1);
    const int64_t batch_size = batch_count_of(vectors);
    const int64_t n = vectors.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);

    // Workspace query once for the whole batch (torch: apply_lapack_eigh).
    int64_t lwork = -1;
    int64_t liwork = -1;
    std::vector<scalar_t> work(1);
    std::vector<int64_t> iwork(1);
    if constexpr (is_float) {
        lapack_ssyevd(jobz, uplo, n, vectors_data, lda, values_data,
                      work.data(), lwork, iwork.data(), liwork);
    } else {
        lapack_dsyevd(jobz, uplo, n, vectors_data, lda, values_data,
                      work.data(), lwork, iwork.data(), liwork);
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    liwork = std::max<int64_t>(1, iwork[0]);
    work.resize(lwork);
    iwork.resize(liwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* v = &vectors_data[i * vectors_stride];
        scalar_t* w = &values_data[i * values_stride];
        int64_t err;
        if constexpr (is_float) {
            err = lapack_ssyevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                iwork.data(), liwork);
        } else {
            err = lapack_dsyevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                iwork.data(), liwork);
        }
        infos_data[i] = static_cast<int32_t>(err);
        // torch returns early on the first failure (BatchLinearAlgebraKernel.cpp:364).
        if (err != 0) break;
    }
}

std::tuple<Tensor, Tensor> eigh_impl(const Tensor& A, bool upper, bool compute_eigenvectors) {
    require_lapack("linalg.eigh");
    square_check_inputs(A, "linalg.eigh");
    Tensor vectors = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    Tensor values = Tensor::empty(batch.empty() ? std::vector<int64_t>{A.size(-1)} : cat_batch(batch, std::vector<int64_t>{A.size(-1)}),
                                  A.dtype(), A.device());
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_syevd<T>(vectors, values, info, upper, compute_eigenvectors);
    });
    return {values.contiguous(), vectors.contiguous()};
}

// Public entries: schema passes UPLO as a string.
std::tuple<Tensor, Tensor> linalg_eigh_kernel(const Tensor& A, std::string UPLO) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return eigh_impl(A, UPLO == "U", true);
}

Tensor linalg_eigvalsh_kernel(const Tensor& A, std::string UPLO) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigvalsh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return std::get<0>(eigh_impl(A, UPLO == "U", false));
}

// ------------------------------------------------------------------- geev --

template <typename scalar_t>
void apply_geev(const Tensor& input, const Tensor& wr, const Tensor& wi,
                const Tensor& rvectors, const Tensor& infos, bool compute_eigenvectors) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    auto* a_data = input.data_ptr<scalar_t>();
    auto* wr_data = wr.data_ptr<scalar_t>();
    auto* wi_data = wi.data_ptr<scalar_t>();
    auto* vr_data = compute_eigenvectors ? rvectors.data_ptr<scalar_t>() : nullptr;
    auto* infos_data = infos.data_ptr<int32_t>();
    const char jobvl = 'N';  // only right eigenvectors are computed
    const char jobvr = compute_eigenvectors ? 'V' : 'N';
    const int64_t n = input.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);
    const int64_t ldvr = compute_eigenvectors ? lda : 1;
    const int64_t input_matrix_stride = matrix_stride_of(input);

    // Workspace query once (apply_linalg_eig in torch).
    std::vector<scalar_t> work(1);
    int64_t info_q = 0;
    if constexpr (is_float) {
        lapack_sgeev(jobvl, jobvr, n, a_data, lda, wr_data, wi_data, nullptr, 1,
                     vr_data, ldvr, work.data(), -1);
    } else {
        lapack_dgeev(jobvl, jobvr, n, a_data, lda, wr_data, wi_data, nullptr, 1,
                     vr_data, ldvr, work.data(), -1);
    }
    int64_t lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    work.resize(lwork);

    const int64_t batch_size = batch_count_of(input);
    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &a_data[i * input_matrix_stride];
        scalar_t* w_r = &wr_data[i * n];
        scalar_t* w_i = &wi_data[i * n];
        scalar_t* vr = compute_eigenvectors ? &vr_data[i * input_matrix_stride] : nullptr;
        int64_t err;
        if constexpr (is_float) {
            err = lapack_sgeev(jobvl, jobvr, n, a, lda, w_r, w_i, nullptr, 1, vr,
                               ldvr, work.data(), lwork);
        } else {
            err = lapack_dgeev(jobvl, jobvr, n, a, lda, w_r, w_i, nullptr, 1, vr,
                               ldvr, work.data(), lwork);
        }
        infos_data[i] = static_cast<int32_t>(err);
    }
}

// Port of linalg_eig_make_complex_eigenvectors_cpu_impl
// (BatchLinearAlgebraKernel.cpp:146): GEEV packs complex conjugate pairs into
// consecutive columns of VR.
template <typename scalar_t, typename cplx_t>
void make_complex_eigenvectors(const Tensor& result, const Tensor& complex_values,
                               const Tensor& real_vectors) {
    const int64_t batch_size = batch_count_of(real_vectors);
    const int64_t n = real_vectors.size(-1);
    const int64_t matrix_stride = matrix_stride_of(real_vectors);
    auto* res = result.data_ptr<cplx_t>();
    const auto* vecs = real_vectors.data_ptr<scalar_t>();
    const auto* vals = complex_values.data_ptr<cplx_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        const scalar_t* v = &vecs[b * matrix_stride];
        cplx_t* r = &res[b * matrix_stride];
        const cplx_t* ev = &vals[b * n];
        for (int64_t j = 0; j < n; ++j) {
            if (ev[j].imag() == scalar_t(0)) {
                for (int64_t i = 0; i < n; ++i)
                    r[j * n + i] = cplx_t(v[j * n + i], scalar_t(0));
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    r[j * n + i] = cplx_t(v[j * n + i], v[(j + 1) * n + i]);
                    r[(j + 1) * n + i] = cplx_t(v[j * n + i], -v[(j + 1) * n + i]);
                }
                ++j;
            }
        }
    }
}

std::tuple<Tensor, Tensor> eig_impl(const Tensor& A, bool compute_eigenvectors) {
    require_lapack("linalg.eig");
    square_check_inputs(A, "linalg.eig");
    const bool is_float_input = A.dtype() == DType::Float32;
    // Working copy in Fortran layout (destroyed by geev), as torch does.
    Tensor input = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    Tensor wr = Tensor::empty(repeat_batch(batch, n), A.dtype(), A.device());
    Tensor wi = Tensor::empty(repeat_batch(batch, n), A.dtype(), A.device());
    Tensor rvectors;
    if (compute_eigenvectors) {
        // Logical shape (..., n, n): batch_shape_of collapses a 2-D input to
        // an empty vector, and empty_column_major indexes shape[-2] — build
        // the full shape explicitly.
        std::vector<int64_t> vshape = batch;
        vshape.push_back(n);
        vshape.push_back(n);
        rvectors = empty_column_major(vshape, A.dtype(), A.device());
    }
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geev<T>(input, wr, wi, rvectors, info, compute_eigenvectors);
    });

    // Complex outputs: cfloat/cdouble matching the input precision.
    const DType cdtype = is_float_input ? DType::ComplexFloat : DType::ComplexDouble;
    Tensor values = Tensor::empty(repeat_batch(batch, n), cdtype, A.device());
    Tensor eigvecs;
    if (compute_eigenvectors) {
        std::vector<int64_t> cshape = batch;
        cshape.push_back(n);
        cshape.push_back(n);
        eigvecs = empty_column_major(cshape, cdtype, A.device());
    }
    pack_complex_outputs(is_float_input, wr, wi, rvectors, values, eigvecs,
                         compute_eigenvectors);
    linalg_check_errors(info, "linalg.eig", A.dim() == 2);
    return {values.contiguous(), compute_eigenvectors ? eigvecs.contiguous() : eigvecs};
}

std::tuple<Tensor, Tensor> linalg_eig_kernel(const Tensor& A) {
    return eig_impl(A, true);
}

Tensor linalg_eigvals_kernel(const Tensor& A) {
    return std::get<0>(eig_impl(A, false));
}

// ------------------------------------------------------------------- gesdd --

template <typename scalar_t>
void apply_svd(const Tensor& A, bool full_matrices, bool compute_uv,
               const Tensor& U, const Tensor& S, const Tensor& Vh, const Tensor& info) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    auto* a_data = A.data_ptr<scalar_t>();
    auto* u_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
    auto* s_data = S.data_ptr<scalar_t>();
    auto* vh_data = compute_uv ? Vh.data_ptr<scalar_t>() : nullptr;
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t a_stride = matrix_stride_of(A);
    const int64_t s_stride = S.size(-1);
    const int64_t u_stride = compute_uv ? matrix_stride_of(U) : 1;
    const int64_t vh_stride = compute_uv ? matrix_stride_of(Vh) : 1;
    const int64_t batch_size = batch_count_of(A);
    const char jobz = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const int64_t lda = A.stride(-1);
    const int64_t ldu = compute_uv ? U.stride(-1) : 1;
    const int64_t ldvh = compute_uv ? Vh.stride(-1) : 1;
    std::vector<int64_t> iwork(static_cast<size_t>(8 * k));

    int64_t lwork = -1;
    std::vector<scalar_t> work(1);
    if constexpr (is_float) {
        lapack_sgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, iwork.data());
    } else {
        lapack_dgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, iwork.data());
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t err;
        if constexpr (is_float) {
            err = lapack_sgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, iwork.data());
        } else {
            err = lapack_dgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, iwork.data());
        }
        info_data[i] = static_cast<int32_t>(err);
    }
}

std::tuple<Tensor, Tensor, Tensor> svd_impl(const Tensor& A, bool full_matrices,
                                            bool compute_uv) {
    require_lapack("linalg.svd");
    check_is_matrix(A, "linalg.svd");
    // gesdd destroys its input: column-major working copy (svd_kernel in torch).
    Tensor a_copy = clone_batched_column_major(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    Tensor U, S, Vh;
    if (compute_uv) {
        U = empty_column_major(cat_batch(batch, std::vector<int64_t>{m, full_matrices ? m : k}),
                               A.dtype(), A.device());
        Vh = empty_column_major(cat_batch(batch, std::vector<int64_t>{full_matrices ? n : k, n}),
                                A.dtype(), A.device());
    } else {
        U = Tensor::empty({0}, A.dtype(), A.device());
        Vh = Tensor::empty({0}, A.dtype(), A.device());
    }
    S = Tensor::empty(cat_batch(batch, std::vector<int64_t>{k}), A.dtype(), A.device());
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_svd<T>(a_copy, full_matrices, compute_uv, U, S, Vh, info);
    });
    linalg_check_errors(info, "linalg.svd", A.dim() == 2);
    if (!compute_uv) {
        U = Tensor::empty(cat_batch(batch, std::vector<int64_t>{m, 0}), A.dtype(), A.device());
        Vh = Tensor::empty(cat_batch(batch, std::vector<int64_t>{0, n}), A.dtype(), A.device());
    }
    return {U.contiguous(), S.contiguous(), Vh.contiguous()};
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_kernel(const Tensor& A, bool full_matrices,
                                                     std::optional<std::string> driver) {
    if (driver.has_value() && driver.value() != "gesvd" && driver.value() != "gesvdj") {
        TP_THROW(RuntimeError, "linalg.svd(): driver ", driver.value(),
                 " is not supported on CPU. Consider torch.linalg.svd(A, full_matrices) instead.");
    }
    return svd_impl(A, full_matrices, true);
}

Tensor linalg_svdvals_kernel(const Tensor& A, std::optional<std::string> /*driver*/) {
    return std::get<1>(svd_impl(A, false, false));
}

// ------------------------------------------------------------- geqrf/orgqr --

template <typename scalar_t>
void apply_geqrf(const Tensor& input, const Tensor& tau) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    auto* input_data = input.data_ptr<scalar_t>();
    auto* tau_data = tau.data_ptr<scalar_t>();
    const int64_t input_matrix_stride = matrix_stride_of(input);
    const int64_t tau_stride = tau.size(-1);
    const int64_t batch_size = batch_count_of(input);
    const int64_t m = input.size(-2);
    const int64_t n = input.size(-1);
    const int64_t lda = std::max<int64_t>(1, m);

    int64_t lwork = -1;
    std::vector<scalar_t> work(1);
    if constexpr (is_float) {
        lapack_sgeqrf(m, n, input_data, lda, tau_data, work.data(), lwork);
    } else {
        lapack_dgeqrf(m, n, input_data, lda, tau_data, work.data(), lwork);
    }
    // torch clamps to at least n (MKL requirement), see apply_geqrf.
    lwork = std::max<int64_t>(n, static_cast<int64_t>(work[0]));
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &input_data[i * input_matrix_stride];
        scalar_t* t = &tau_data[i * tau_stride];
        if constexpr (is_float) {
            lapack_sgeqrf(m, n, a, lda, t, work.data(), lwork);
        } else {
            lapack_dgeqrf(m, n, a, lda, t, work.data(), lwork);
        }
    }
}

template <typename scalar_t>
void apply_orgqr(Tensor& self, const Tensor& tau) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    if (self.numel() == 0) return;
    auto* self_data = self.data_ptr<scalar_t>();
    const auto* tau_data = tau.data_ptr<scalar_t>();
    const int64_t self_matrix_stride = matrix_stride_of(self);
    const int64_t tau_stride = tau.size(-1);
    const int64_t batch_size = batch_count_of(self);
    const int64_t m = self.size(-2);
    const int64_t n = self.size(-1);
    const int64_t k = tau.size(-1);
    const int64_t lda = std::max<int64_t>(1, m);

    int64_t lwork = -1;
    std::vector<scalar_t> work(1);
    if constexpr (is_float) {
        lapack_sorgqr(m, n, k, self_data, lda, tau_data, work.data(), lwork);
    } else {
        lapack_dorgqr(m, n, k, self_data, lda, tau_data, work.data(), lwork);
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* s = &self_data[i * self_matrix_stride];
        const scalar_t* t = &tau_data[i * tau_stride];
        if constexpr (is_float) {
            lapack_sorgqr(m, n, k, s, lda, t, work.data(), lwork);
        } else {
            lapack_dorgqr(m, n, k, s, lda, t, work.data(), lwork);
        }
    }
}

// torch::linalg_qr composite: geqrf + orgqr + triangular extraction.
std::tuple<Tensor, Tensor> linalg_qr_kernel(const Tensor& A, std::string mode) {
    require_lapack("linalg.qr");
    check_is_matrix(A, "linalg.qr");
    if (mode != "reduced" && mode != "complete" && mode != "r" && mode != "R") {
        TP_THROW(RuntimeError, "linalg.qr: mode '", mode,
                 "' not recognized. Mode must be one of 'reduced', 'complete', 'r' or 'R'");
    }
    const bool reduced = (mode == "reduced" || mode == "r" || mode == "R");
    Tensor QR = clone_batched_column_major(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    Tensor tau = Tensor::empty(cat_batch(batch, std::vector<int64_t>{k}), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geqrf<T>(QR, tau);
    });

    const int64_t qcols = reduced ? k : m;
    // Pack the first qcols columns of the reflector buffer into an
    // (m x qcols) column-major buffer for orgqr.
    Tensor Q_in = empty_column_major(cat_batch(batch, std::vector<int64_t>{m, qcols}),
                                     A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* src = QR.data_ptr<T>();
        auto* dst = Q_in.data_ptr<T>();
        const int64_t bs = linear_batch_size(batch);
        for (int64_t b = 0; b < bs; ++b)
            for (int64_t col = 0; col < qcols; ++col)
                std::memcpy(dst + (b * m * qcols) + col * m,
                            src + (b * m * n) + col * m, sizeof(T) * m);
        apply_orgqr<T>(Q_in, tau);
    });

    // R: upper triangle of the geqrf buffer.
    const int64_t rrows = reduced ? k : m;
    Tensor R = empty_column_major(cat_batch(batch, std::vector<int64_t>{rrows, n}),
                                  A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* src = QR.data_ptr<T>();
        auto* dst = R.data_ptr<T>();
        const int64_t bs = linear_batch_size(batch);
        for (int64_t b = 0; b < bs; ++b)
            for (int64_t row = 0; row < rrows; ++row)
                for (int64_t col = row; col < n; ++col)
                    dst[b * rrows * n + col * rrows + row] =
                        src[b * m * n + col * m + row];
    });
    return {Q_in.contiguous(), R.contiguous()};
}

Tensor linalg_householder_product_kernel(const Tensor& input, const Tensor& tau) {
    require_lapack("linalg.householder_product");
    check_is_matrix(input, "linalg.householder_product");
    if (input.size(-2) < input.size(-1)) {
        TP_THROW(RuntimeError, "linalg.householder_product: If input has size (..., m, n), "
                               "n must be less than or equal to m, but got n = ",
                 input.size(-1), " and m = ", input.size(-2));
    }
    if (tau.dim() < 1 || tau.size(-1) != std::min(input.size(-2), input.size(-1))) {
        TP_THROW(RuntimeError, "linalg.householder_product: If tau has size (..., k), then "
                               "when input has size (..., m, n) we require k == min(m, n)");
    }
    if (tau.dtype() != input.dtype()) {
        TP_THROW(RuntimeError, "linalg.householder_product: input and tau must have the same dtype");
    }
    Tensor result = clone_batched_column_major(input);
    run_real(input.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_orgqr<T>(result, tau);
    });
    return result.contiguous();
}

// ------------------------------------------------------------------- gels --

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq_kernel(
        const Tensor& A, const Tensor& B, std::optional<double> rcond,
        std::optional<std::string> driver_opt) {
    const char* api = "linalg.lstsq";
    require_lapack(api);
    const std::string driver = driver_opt.value_or("gels");
    if (driver != "gels") {
        TP_THROW(NotImplementedError, api, ": driver '", driver,
                 "' is not supported on CPU; only 'gels' is implemented");
    }
    check_is_matrix(A, api);
    check_is_matrix(B, api);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    if (m < n) {
        TP_THROW(RuntimeError, api,
                 ": The input tensor A should have at least as many rows as columns, "
                 "but they are ", m, " by ", n);
    }
    // rcond is ignored for the gels driver (torch warns; keep silent parity).
    (void)rcond;

    const auto batch = broadcast_batch(A, B);
    const int64_t nrhs = B.size(-1);
    const int64_t ldb = std::max<int64_t>(m, n);
    const int64_t bs = linear_batch_size(batch);

    Tensor A_work = clone_batched_column_major(expand_to_batch(A, batch));  // destroyed
    Tensor B_work = empty_column_major(cat_batch(batch, std::vector<int64_t>{ldb, nrhs}),
                                        B.dtype(), B.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* a = A_work.data_ptr<T>();
        auto* b = B_work.data_ptr<T>();
        constexpr bool is_float = std::is_same_v<T, float>;

        // Zero-fill the padded buffer, then copy B into its top m rows.
        // b_rowmajor is a contiguous ROW-major (... m, nrhs) tensor: element
        // (row, col) lives at row * nrhs + col (the old col * m stride read
        // garbage for nrhs != 1).
        const Tensor b_rowmajor = expand_to_batch(B, batch).contiguous();
        const auto* bsrc = b_rowmajor.data_ptr<T>();
        const int64_t b_ms_src = matrix_stride_of(b_rowmajor);
        const int64_t b_cols = B.size(-1);
        std::memset(b, 0, sizeof(T) * static_cast<size_t>(B_work.numel()));
        for (int64_t i = 0; i < bs; ++i)
            for (int64_t row = 0; row < m; ++row)
                for (int64_t col = 0; col < nrhs; ++col)
                    b[i * ldb * nrhs + col * ldb + row] =
                        bsrc[i * b_ms_src + row * b_cols + col];

        // Workspace query once, then one gels call per batch element.
        int64_t lwork = -1;
        std::vector<T> work(1);
        if constexpr (is_float) {
            lapack_sgels('N', m, n, nrhs, a, m, b, ldb, work.data(), lwork);
        } else {
            lapack_dgels('N', m, n, nrhs, a, m, b, ldb, work.data(), lwork);
        }
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
        work.resize(lwork);
        for (int64_t i = 0; i < bs; ++i) {
            int64_t err;
            if constexpr (is_float) {
                err = lapack_sgels('N', m, n, nrhs, &a[i * m * n], m,
                                   &b[i * ldb * nrhs], ldb, work.data(), lwork);
            } else {
                err = lapack_dgels('N', m, n, nrhs, &a[i * m * n], m,
                                   &b[i * ldb * nrhs], ldb, work.data(), lwork);
            }
            if (err != 0) {
                TP_THROW(RuntimeError, api, ": (Batch element ", i,
                         ") The least squares solution could not be computed.");
            }
        }
    });

    Tensor solution = B_work.contiguous().slice(-2, 0, n).contiguous();
    Tensor residuals;
    if (m > n) {
        residuals = Tensor::empty(cat_batch(batch, std::vector<int64_t>{nrhs}), B.dtype(), B.device());
        const Tensor b_rowmajor = B_work.contiguous();
        run_real(B.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            const auto* b = b_rowmajor.data_ptr<T>();
            auto* res = residuals.data_ptr<T>();
            for (int64_t i = 0; i < bs; ++i)
                for (int64_t col = 0; col < nrhs; ++col) {
                    T acc = T(0);
                    for (int64_t row = n; row < ldb; ++row)
                        acc += b[i * ldb * nrhs + col * ldb + row] *
                               b[i * ldb * nrhs + col * ldb + row];
                    res[i * nrhs + col] = acc;
                }
        });
    } else {
        residuals = Tensor::empty(cat_batch(batch, std::vector<int64_t>{0}), B.dtype(), B.device());
    }
    Tensor rank = Tensor::full(batch, Scalar(static_cast<int64_t>(n)),
                               DType::Int64, B.device());
    return {solution, residuals, rank, solution};
}

// --------------------------------------------------------------- sytrf LDL --

template <typename scalar_t>
void apply_ldl_factor(const Tensor& LD, const Tensor& pivots, const Tensor& info, char uplo) {
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    auto* a_data = LD.data_ptr<scalar_t>();
    auto* pivots_data = pivots.data_ptr<int32_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t batch_size = batch_count_of(LD);
    const int64_t n = LD.size(-2);
    const int64_t lda = LD.stride(-1);
    const int64_t a_stride = LD.dim() > 2 ? LD.stride(-3) : 0;
    const int64_t pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

    int64_t lwork = -1;
    std::vector<scalar_t> work(1);
    std::vector<int64_t> ipiv(static_cast<size_t>(n));
    if constexpr (is_float) {
        lapack_ssytrf(uplo, n, a_data, lda, ipiv.data(), work.data(), lwork);
    } else {
        lapack_dsytrf(uplo, n, a_data, lda, ipiv.data(), work.data(), lwork);
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &a_data[i * a_stride];
        int64_t err;
        if constexpr (is_float) {
            err = lapack_ssytrf(uplo, n, a, lda, ipiv.data(), work.data(), lwork);
        } else {
            err = lapack_dsytrf(uplo, n, a, lda, ipiv.data(), work.data(), lwork);
        }
        for (int64_t j = 0; j < n; ++j)
            pivots_data[i * pivots_stride + j] = static_cast<int32_t>(ipiv[j]);
        info_data[i] = static_cast<int32_t>(err);
    }
}

std::tuple<Tensor, Tensor, Tensor> ldl_factor_impl(const Tensor& A, bool /*hermitian*/,
                                                   bool check_errors) {
    const char* api = check_errors ? "linalg.ldl_factor_ex" : "linalg.ldl_factor";
    require_lapack(api);
    square_check_inputs(A, api);
    Tensor LD = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    Tensor pivots = empty_pivots(A, batch, n);
    Tensor info = empty_info_like(A, batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_ldl_factor<T>(LD, pivots, info, 'L');
    });
    if (check_errors) linalg_check_errors(info, api, A.dim() == 2);
    return {LD.contiguous(), pivots, info};
}

Tensor ldl_solve_impl(const Tensor& LD, const Tensor& pivots, const Tensor& B,
                      bool /*hermitian*/) {
    const char* api = "linalg.ldl_solve";
    require_lapack(api);
    square_check_inputs(LD, api, "LD");
    check_is_matrix(B, api, "B");
    // Sanity checks ported from torch's ldl_solve_kernel.
    {
        Tensor pv64 = pivots.to(DType::Int64);
        const auto* pv = pv64.data_ptr<int64_t>();
        for (int64_t i = 0; i < pivots.numel(); ++i) {
            if (pv[i] < 1 || pv[i] > LD.size(-2)) {
                TP_THROW(RuntimeError, "Pivots given to ldl_solve must all satisfy |pivot| >= 1. "
                                       "Did you properly pass the result of ldl_factor?");
            }
        }
    }
    const auto batch = broadcast_batch(LD, B);
    Tensor B_work = clone_batched_column_major(expand_to_batch(B, batch));
    Tensor LD_work = clone_batched_column_major(expand_to_batch(LD, batch));
    Tensor piv32 = pivots.expand(cat_batch(batch, std::vector<int64_t>{LD.size(-2)})).to(DType::Int32);
    const int64_t n = LD.size(-2);
    const int64_t nrhs = B.size(-1);
    const int64_t bs = linear_batch_size(batch);
    run_real(LD.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* a = LD_work.data_ptr<T>();
        auto* b = B_work.data_ptr<T>();
        const auto* piv = piv32.data_ptr<int32_t>();
        constexpr bool is_float = std::is_same_v<T, float>;
        for (int64_t i = 0; i < bs; ++i) {
            std::vector<int64_t> ipiv(static_cast<size_t>(n));
            for (int64_t j = 0; j < n; ++j) ipiv[j] = piv[i * n + j];
            int64_t err;
            if constexpr (is_float) {
                err = lapack_ssytrs('L', n, nrhs, &a[i * n * n], n, ipiv.data(),
                                    &b[i * n * nrhs], n);
            } else {
                err = lapack_dsytrs('L', n, nrhs, &a[i * n * n], n, ipiv.data(),
                                    &b[i * n * nrhs], n);
            }
            (void)err;
        }
    });
    return B_work.contiguous();
}

// ------------------------------------------------------------ lu with unpack

std::tuple<Tensor, Tensor, Tensor> linalg_lu_kernel(const Tensor& A, bool pivot) {
    require_lapack("linalg.lu");
    if (!pivot) {
        TP_THROW(RuntimeError, "linalg.lu: LU without pivoting is not implemented");
    }
    square_check_inputs(A, "linalg.lu");
    auto [LU, pivots, info] = lu_factor_ex_impl(A, pivot, false,
                                                "torch.linalg.lu_factor_ex");
    (void)info;
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t kk = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const int64_t bs = linear_batch_size(batch);
    Tensor P = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{m, m}), A.dtype(), A.device());
    Tensor L = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{m, kk}), A.dtype(), A.device());
    Tensor U = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{kk, n}), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* lu_all = LU.data_ptr<T>();  // column-major (*, m, n), lda = m
        const auto* piv = pivots.data_ptr<int32_t>();
        auto* p_out = P.data_ptr<T>();
        auto* l_out = L.data_ptr<T>();
        auto* u_out = U.data_ptr<T>();
        for (int64_t b = 0; b < bs; ++b) {
            const T* lu = &lu_all[b * m * n];
            // Permutation from the ipiv swap sequence applied to identity rows.
            std::vector<int64_t> perm(static_cast<size_t>(m));
            for (int64_t i = 0; i < m; ++i) perm[i] = i;
            for (int64_t i = 0; i < kk; ++i) {
                const int64_t p = piv[b * kk + i] - 1;
                if (p != i) std::swap(perm[i], perm[p]);
            }
            T* pm = &p_out[b * m * m];
            std::memset(pm, 0, sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(m));
            for (int64_t j = 0; j < m; ++j) pm[j * m + perm[j]] = T(1);
            T* lm = &l_out[b * m * kk];
            for (int64_t col = 0; col < kk; ++col)
                for (int64_t row = 0; row < m; ++row)
                    lm[col * m + row] =
                        row < col ? T(0) : (row == col ? T(1) : lu[col * m + row]);
            T* um = &u_out[b * kk * n];
            for (int64_t col = 0; col < n; ++col)
                for (int64_t row = 0; row < kk; ++row)
                    um[col * kk + row] =
                        row <= col && col < n ? lu[col * m + row] : T(0);
        }
    });
    return {P, L, U};
}

// --------------------------------------------------------------- diagonal --

// torch::linalg.diagonal: extract diagonals along dims (dim1, dim2).
Tensor linalg_diagonal_kernel(const Tensor& A, int64_t offset, int64_t dim1, int64_t dim2) {
    const char* api = "linalg.diagonal";
    check_is_matrix(A, api);
    const int64_t ndim = A.dim();
    const auto norm_dim = [&](int64_t d) {
        const int64_t r = d < 0 ? d + ndim : d;
        if (r < 0 || r >= ndim) {
            TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                     -ndim, ", ", ndim - 1, "], but got ", d, ")");
        }
        return r;
    };
    dim1 = norm_dim(dim1);
    dim2 = norm_dim(dim2);
    if (dim1 == dim2) {
        TP_THROW(RuntimeError, "linalg.diagonal: dimension 1 and dimension 2 cannot be equal");
    }
    const int64_t d1 = A.size(dim1);
    const int64_t d2 = A.size(dim2);
    const int64_t diag_len =
        offset >= 0 ? std::max<int64_t>(std::min(d1, d2 - offset), 0)
                    : std::max<int64_t>(std::min(d1 + offset, d2), 0);

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) out_shape.push_back(A.size(i));
    }
    out_shape.push_back(diag_len);
    Tensor out = Tensor::empty(out_shape, A.dtype(), A.device());

    // Row-major strides of the output for flat iteration.
    std::vector<int64_t> ostrides(out_shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(out_shape.size()) - 2; i >= 0; --i)
        ostrides[i] = ostrides[i + 1] * out_shape[i + 1];

    // Map each output axis back to an input axis.
    std::vector<int64_t> in_strides(ndim);
    {
        std::vector<int64_t> istrides(ndim, 1);
        for (int64_t i = ndim - 2; i >= 0; --i) istrides[i] = istrides[i + 1] * A.size(i + 1);
        in_strides = istrides;
    }
    std::vector<int64_t> axis_map(out_shape.size());
    {
        size_t k = 0;
        for (int64_t i = 0; i < ndim; ++i)
            if (i != dim1 && i != dim2) axis_map[k++] = i;
    }

    const int64_t total = out.numel();
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* src = A.data_ptr<T>();
        auto* dst = out.data_ptr<T>();
        std::vector<int64_t> idx(out_shape.size(), 0);
        for (int64_t linear = 0; linear < total; ++linear) {
            // Decompose into multi-index.
            int64_t rem = linear;
            int64_t t = -1;
            int64_t src_off = 0;
            for (size_t ax = 0; ax < out_shape.size(); ++ax) {
                idx[ax] = rem / ostrides[ax];
                rem -= idx[ax] * ostrides[ax];
                if (static_cast<int64_t>(ax) == static_cast<int64_t>(out_shape.size()) - 1) {
                    t = idx[ax];
                } else {
                    src_off += idx[ax] * in_strides[axis_map[ax]];
                }
            }
            src_off += (t + (offset >= 0 ? 0 : -offset)) * in_strides[dim1];
            src_off += (t + (offset >= 0 ? offset : 0)) * in_strides[dim2];
            dst[linear] = src[src_off];
        }
    });
    return out;
}

// ------------------------------------------------------- public composites --

std::tuple<Tensor, Tensor> linalg_lu_factor_kernel(const Tensor& A, bool pivot) {
    auto [LU, pivots, info] =
        lu_factor_ex_impl(A, pivot, false, "torch.linalg.lu_factor");
    (void)info;
    return {LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor> linalg_lu_factor_ex_kernel(const Tensor& A,
                                                              bool pivot,
                                                              bool check_errors) {
    return lu_factor_ex_impl(A, pivot, check_errors, "torch.linalg.lu_factor_ex");
}

std::tuple<Tensor, Tensor> linalg_ldl_factor_kernel(const Tensor& A, bool hermitian) {
    auto [LD, pivots, info] = ldl_factor_impl(A, hermitian, false);
    linalg_check_errors(info, "linalg.ldl_factor", A.dim() == 2);
    return {LD, pivots};
}

std::tuple<Tensor, Tensor, Tensor> linalg_ldl_factor_ex_kernel(const Tensor& A,
                                                               bool hermitian,
                                                               bool check_errors) {
    return ldl_factor_impl(A, hermitian, check_errors);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, LinalgKernels) {
    m.impl("linalg_cholesky", linalg_cholesky_kernel);
    m.impl("linalg_cholesky_ex", linalg_cholesky_ex_kernel);
    m.impl("linalg_inv", linalg_inv_kernel);
    m.impl("linalg_inv_ex", linalg_inv_ex_kernel);
    m.impl("linalg_det", linalg_det_kernel);
    m.impl("linalg_slogdet", linalg_slogdet_kernel);
    m.impl("linalg_solve", linalg_solve_kernel);
    m.impl("linalg_solve_ex", linalg_solve_ex_kernel);
    m.impl("linalg_lu_factor", linalg_lu_factor_kernel);
    m.impl("linalg_lu_factor_ex", linalg_lu_factor_ex_kernel);
    m.impl("linalg_lu", linalg_lu_kernel);
    m.impl("linalg_lu_solve", linalg_lu_solve_kernel);
    m.impl("linalg_solve_triangular", linalg_solve_triangular_kernel);
    m.impl("linalg_eigh", linalg_eigh_kernel);
    m.impl("linalg_eigvalsh", linalg_eigvalsh_kernel);
    m.impl("linalg_eig", linalg_eig_kernel);
    m.impl("linalg_eigvals", linalg_eigvals_kernel);
    m.impl("linalg_svd", linalg_svd_kernel);
    m.impl("linalg_svdvals", linalg_svdvals_kernel);
    m.impl("linalg_lstsq", linalg_lstsq_kernel);
    m.impl("linalg_qr", linalg_qr_kernel);
    m.impl("linalg_householder_product", linalg_householder_product_kernel);
    m.impl("linalg_ldl_factor", linalg_ldl_factor_kernel);
    m.impl("linalg_ldl_factor_ex", linalg_ldl_factor_ex_kernel);
    m.impl("linalg_ldl_solve", ldl_solve_impl);
    m.impl("linalg_diagonal", linalg_diagonal_kernel);
}

}  // namespace cpu
}  // namespace tensorplay
