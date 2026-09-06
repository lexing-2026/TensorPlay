// runtime-resolved ILP64 LAPACK (see cpu/Lapack.h).  All matrices follow the
// Fortran (batched column-major) layout that LAPACK expects, produced with
// cloneBatchedColumnMajor.

#include "Tensor.h"
#include "Complex.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"
#include "LinearAlgebraNames.h"
#include "cpu/Lapack.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace ops = tensorplay::tpx::ops;

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
                     " for linalg on CPU (only float32/float64 are implemented)");
    }
}

template <typename T>
struct LinalgScalarTraits {
    using value_type = T;
    static constexpr bool is_complex = false;
};

template <typename T>
struct LinalgScalarTraits<complex<T>> {
    using value_type = T;
    static constexpr bool is_complex = true;
};

template <typename T>
auto linalg_abs(const T& value) {
    if constexpr (LinalgScalarTraits<T>::is_complex) {
        return tensorplay::abs(value);
    } else {
        return std::abs(value);
    }
}

template <class Kernel>
decltype(auto) run_linalg(DType dt, Kernel&& k) {
    switch (dt) {
        case DType::Float32:
            return k(static_cast<float*>(nullptr));
        case DType::Float64:
            return k(static_cast<double*>(nullptr));
        case DType::ComplexFloat:
            return k(static_cast<complex<float>*>(nullptr));
        case DType::ComplexDouble:
            return k(static_cast<complex<double>*>(nullptr));
        default:
            TP_THROW(NotImplementedError,
                     "unsupported dtype ", pretty_dtype_name(dt),
                     " for linalg on CPU");
    }
}

// Complex-only dispatch: kernels that wrap complex eigensolver routines
// routines carry no real-typed instantiation, so every switch arm must map
// onto a native complex element type for the template to type-check.
template <class Kernel>
decltype(auto) run_linalg_complex(DType dt, Kernel&& k) {
    switch (dt) {
        case DType::ComplexFloat:
            return k(static_cast<complex<float>*>(nullptr));
        case DType::ComplexDouble:
            return k(static_cast<complex<double>*>(nullptr));
        default:
            TP_THROW(NotImplementedError,
                     "unsupported dtype ", pretty_dtype_name(dt),
                     " for complex linalg on CPU");
    }
}

void write_linalg_output(const char* op, const Tensor& value, Tensor& out) {
    if (!out.defined()) {
        out = value;
        return;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    out.resize_(static_cast<std::vector<int64_t>>(value.shape()));
    out.copy_(value);
}

template <typename T>
int64_t lapack_getrf(int64_t m, int64_t n, T* a, int64_t lda, int64_t* ipiv) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgetrf(m, n, a, lda, ipiv);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgetrf(m, n, a, lda, ipiv);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgetrf(m, n, a, lda, ipiv);
    } else {
        return lapack_zgetrf(m, n, a, lda, ipiv);
    }
}

template <typename T>
int64_t lapack_getrs(char trans, int64_t n, int64_t nrhs, const T* a,
                     int64_t lda, const int64_t* ipiv, T* b, int64_t ldb) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
    } else {
        return lapack_zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
    }
}

template <typename T>
int64_t lapack_potrf(char uplo, int64_t n, T* a, int64_t lda) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_spotrf(uplo, n, a, lda);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dpotrf(uplo, n, a, lda);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cpotrf(uplo, n, a, lda);
    } else {
        return lapack_zpotrf(uplo, n, a, lda);
    }
}

template <typename T>
int64_t lapack_trtrs(char side, char uplo, char transa, char diag, int64_t n,
                     int64_t nrhs, const T* a, int64_t lda, T* b, int64_t ldb) {
    const int64_t order = 102;
    const int64_t side_code = side == 'L' ? 141 : 142;
    const int64_t uplo_code = uplo == 'U' ? 121 : 122;
    const int64_t trans_code = transa == 'N' ? 111 : (transa == 'T' ? 112 : 113);
    const int64_t diag_code = diag == 'U' ? 132 : 131;
    if constexpr (std::is_same_v<T, float>) {
        lapack_strsm(order, side_code, uplo_code, trans_code, diag_code, n, nrhs,
                     1.0f, a, lda, b, ldb);
        return 0;
    } else if constexpr (std::is_same_v<T, double>) {
        lapack_dtrsm(order, side_code, uplo_code, trans_code, diag_code, n, nrhs,
                     1.0, a, lda, b, ldb);
        return 0;
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        const complex<float> alpha(1.0f, 0.0f);
        lapack_ctrsm(order, side_code, uplo_code, trans_code, diag_code, n, nrhs,
                     &alpha, a, lda, b, ldb);
        return 0;
    } else {
        const complex<double> alpha(1.0, 0.0);
        lapack_ztrsm(order, side_code, uplo_code, trans_code, diag_code, n, nrhs,
                     &alpha, a, lda, b, ldb);
        return 0;
    }
}

template <typename T>
int64_t lapack_geqrf(int64_t m, int64_t n, T* a, int64_t lda, T* tau,
                     T* work, int64_t lwork) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgeqrf(m, n, a, lda, tau, work, lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgeqrf(m, n, a, lda, tau, work, lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgeqrf(m, n, a, lda, tau, work, lwork);
    } else {
        return lapack_zgeqrf(m, n, a, lda, tau, work, lwork);
    }
}

template <typename T>
int64_t lapack_orgqr(int64_t m, int64_t n, int64_t k, T* a, int64_t lda,
                     const T* tau, T* work, int64_t lwork) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sorgqr(m, n, k, a, lda, tau, work, lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dorgqr(m, n, k, a, lda, tau, work, lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cungqr(m, n, k, a, lda, tau, work, lwork);
    } else {
        return lapack_zungqr(m, n, k, a, lda, tau, work, lwork);
    }
}

template <typename T>
int64_t lapack_gels(char trans, int64_t m, int64_t n, int64_t nrhs, T* a,
                    int64_t lda, T* b, int64_t ldb, T* work, int64_t lwork) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
    } else {
        return lapack_zgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork);
    }
}

enum class LstsqDriver { Gels, Gelsy, Gelsd, Gelss };

template <typename T>
int64_t lapack_gelsy(int64_t m, int64_t n, int64_t nrhs, T* a, int64_t lda,
                     T* b, int64_t ldb, int64_t* jpvt,
                     typename LinalgScalarTraits<T>::value_type rcond,
                     int64_t* rank, T* work, int64_t lwork,
                     typename LinalgScalarTraits<T>::value_type* rwork) {
    using R = typename LinalgScalarTraits<T>::value_type;
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,
                             work, lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,
                             work, lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,
                             work, lwork, rwork);
    } else {
        return lapack_zgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,
                             work, lwork, rwork);
    }
}

template <typename T>
int64_t lapack_gelsd(int64_t m, int64_t n, int64_t nrhs, T* a, int64_t lda,
                     T* b, int64_t ldb,
                     typename LinalgScalarTraits<T>::value_type* s,
                     typename LinalgScalarTraits<T>::value_type rcond,
                     int64_t* rank, T* work, int64_t lwork,
                     typename LinalgScalarTraits<T>::value_type* rwork,
                     int64_t* iwork) {
    using R = typename LinalgScalarTraits<T>::value_type;
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, iwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, iwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, rwork, iwork);
    } else {
        return lapack_zgelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, rwork, iwork);
    }
}

template <typename T>
int64_t lapack_gelss(int64_t m, int64_t n, int64_t nrhs, T* a, int64_t lda,
                     T* b, int64_t ldb,
                     typename LinalgScalarTraits<T>::value_type* s,
                     typename LinalgScalarTraits<T>::value_type rcond,
                     int64_t* rank, T* work, int64_t lwork,
                     typename LinalgScalarTraits<T>::value_type* rwork) {
    using R = typename LinalgScalarTraits<T>::value_type;
    if constexpr (std::is_same_v<T, float>) {
        return lapack_sgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return lapack_cgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, rwork);
    } else {
        return lapack_zgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work,
                             lwork, rwork);
    }
}

template <typename T>
int64_t lapack_lstsq_call(
        LstsqDriver driver, int64_t m, int64_t n, int64_t nrhs, T* a,
        int64_t lda, T* b, int64_t ldb, T* work, int64_t lwork,
        int64_t* jpvt, typename LinalgScalarTraits<T>::value_type rcond,
        int64_t* rank, typename LinalgScalarTraits<T>::value_type* rwork,
        typename LinalgScalarTraits<T>::value_type* s, int64_t* iwork) {
    switch (driver) {
        case LstsqDriver::Gels:
            return lapack_gels('N', m, n, nrhs, a, lda, b, ldb, work, lwork);
        case LstsqDriver::Gelsy:
            return lapack_gelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, rank,
                                work, lwork, rwork);
        case LstsqDriver::Gelsd:
            return lapack_gelsd(m, n, nrhs, a, lda, b, ldb, s, rcond, rank,
                                work, lwork, rwork, iwork);
        case LstsqDriver::Gelss:
            return lapack_gelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank,
                                work, lwork, rwork);
    }
    return -1;
}

template <typename T>
int64_t lapack_ldl_factor(char uplo, bool hermitian, int64_t n, T* a,
                          int64_t lda, int64_t* ipiv, T* work, int64_t lwork) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_ssytrf(uplo, n, a, lda, ipiv, work, lwork);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dsytrf(uplo, n, a, lda, ipiv, work, lwork);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return hermitian ? lapack_chetrf(uplo, n, a, lda, ipiv, work, lwork)
                         : lapack_csytrf(uplo, n, a, lda, ipiv, work, lwork);
    } else {
        return hermitian ? lapack_zhetrf(uplo, n, a, lda, ipiv, work, lwork)
                         : lapack_zsytrf(uplo, n, a, lda, ipiv, work, lwork);
    }
}

template <typename T>
int64_t lapack_ldl_solve(char uplo, bool hermitian, int64_t n, int64_t nrhs,
                         const T* a, int64_t lda, const int64_t* ipiv, T* b,
                         int64_t ldb) {
    if constexpr (std::is_same_v<T, float>) {
        return lapack_ssytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
    } else if constexpr (std::is_same_v<T, double>) {
        return lapack_dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
    } else if constexpr (std::is_same_v<T, complex<float>>) {
        return hermitian ? lapack_chetrs(uplo, n, nrhs, a, lda, ipiv, b, ldb)
                         : lapack_csytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
    } else {
        return hermitian ? lapack_zhetrs(uplo, n, nrhs, a, lda, ipiv, b, ldb)
                         : lapack_zsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
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
    if (A.device() != B.device()) {
        TP_THROW(DeviceMismatchError, fn, ": A and B must be on the same device");
    }
    if (A.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, fn, ": A and B must have the same dtype, but got ",
                 pretty_dtype_name(A.dtype()), " and ", pretty_dtype_name(B.dtype()));
    }
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

void linalg_check_errors_kernel(const Tensor& infos, std::string api_name,
                                bool is_matrix) {
    linalg_check_errors(infos, api_name, is_matrix);
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

int64_t svd_real_workspace(char jobz, int64_t m, int64_t n) {
    const int64_t mn = std::min(m, n);
    const int64_t mx = std::max(m, n);
    if (jobz == 'N') return 5 * mn;
    if (mx > 10 * mn) return 5 * mn * mn + 5 * mn;
    return std::max(5 * mn * mn + 5 * mn,
                    2 * mx * mn + 2 * mn * mn + mn);
}

// Pack GEEV's real outputs into complex tensors: eigenvalues wr + i*wi and,
// when requested, the conjugate-pair eigenvector expansion.
void pack_complex_outputs(bool is_float_input, const Tensor& wr, const Tensor& wi,
                          const Tensor& rvectors, Tensor& values, Tensor& eigvecs,
                          bool compute_eigenvectors) {
    if (is_float_input) {
        auto* v = values.data_ptr<complex<float>>();
        const auto* r = wr.data_ptr<float>();
        const auto* im = wi.data_ptr<float>();
        for (int64_t i = 0; i < wr.numel(); ++i) v[i] = complex<float>(r[i], im[i]);
        if (compute_eigenvectors)
            make_complex_eigenvectors<float, complex<float>>(eigvecs, values, rvectors);
    } else {
        auto* v = values.data_ptr<complex<double>>();
        const auto* r = wr.data_ptr<double>();
        const auto* im = wi.data_ptr<double>();
        for (int64_t i = 0; i < wr.numel(); ++i) v[i] = complex<double>(r[i], im[i]);
        if (compute_eigenvectors)
            make_complex_eigenvectors<double, complex<double>>(eigvecs, values, rvectors);
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
    const int64_t matrix_rank = std::min(m, n);
    const double rank3 = static_cast<double>(matrix_rank) * matrix_rank * matrix_rank;
    const int64_t chunk_size_per_thread = static_cast<int64_t>(
        std::min(1.0, rank3 == 0.0 ? 3200.0 : 3200.0 / rank3));
    const int64_t grain_size = chunk_size_per_thread * static_cast<int64_t>(1);
    parallel_for(0, batch_size > 0 ? batch_size : 1, grain_size,
                 [&](int64_t begin, int64_t end) {
                     for (int64_t i = begin; i < end && i < batch_size; ++i) {
                         scalar_t* a = &input_data[i * input_matrix_stride];
                         int32_t* piv = &pivots_data[i * pivots_stride];
                         std::vector<int64_t> ipiv(pivots_stride);
                         int64_t info;
                         info = lapack_getrf(m, n, a, leading_dimension, ipiv.data());
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
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_lu_factor<T>(LU, pivots, info);
    });
    if (check_errors) linalg_check_errors(info, api_name, A.dim() == 2);
    return {LU.contiguous(), pivots, info};
}

// ------------------------------------------------------- det / slogdet ------

// The determinant of a permutation is determined by the number of its swaps.
int64_t lu_perm_sign(const int32_t* pivots, int64_t k) {
            int64_t sign_changes = 0;
    for (int64_t i = 0; i < k; ++i) {
                if (pivots[i] - 1 != static_cast<int32_t>(i)) ++sign_changes;
    }
            return (sign_changes % 2 == 0) ? 1 : -1;
}

std::tuple<Tensor, Tensor, Tensor> linalg_det_internal_kernel(const Tensor& A) {
    require_lapack("linalg.det");
    square_check_inputs(A, "linalg.det");
    // det(A^T) = det(A): reuse the contiguous layout as the column-major copy.
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    // Named locals: the lambda below references these, and structured
    // bindings are not capturable on every supported compiler.
    Tensor LU_tensor;
    Tensor pivots_tensor;
    std::tie(LU_tensor, pivots_tensor, std::ignore) =
        lu_factor_ex_impl(src, true, false, "linalg.lu_factor");
    const int64_t batch_size = batch_count_of(LU_tensor);
    const int64_t n = LU_tensor.size(-1);
    Tensor result = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* lu = LU_tensor.data_ptr<T>();
        auto* out = result.data_ptr<T>();
        const auto* piv = pivots_tensor.data_ptr<int32_t>();
        const int64_t ms = matrix_stride_of(LU_tensor);
        for (int64_t b = 0; b < batch_size; ++b) {
            T det = T(1);
            for (int64_t i = 0; i < n; ++i) det *= lu[b * ms + i * n + i];
            out[b] = det * static_cast<T>(lu_perm_sign(&piv[b * n], n));
        }
    });
    return {result, LU_tensor, pivots_tensor};
}

Tensor linalg_det_kernel(const Tensor& A) {
    return std::get<0>(linalg_det_internal_kernel(A));
}

std::tuple<Tensor, Tensor, Tensor> linalg_det_internal_out_kernel(
        const Tensor& A, Tensor& result, Tensor& LU, Tensor& pivots) {
    auto values = linalg_det_internal_kernel(A);
    write_linalg_output("linalg.det", std::get<0>(values), result);
    write_linalg_output("linalg.det", std::get<1>(values), LU);
    write_linalg_output("linalg.det", std::get<2>(values), pivots);
    return {result, LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_slogdet_internal_kernel(
        const Tensor& A) {
    require_lapack("linalg.slogdet");
    square_check_inputs(A, "linalg.slogdet");
    Tensor work = A.is_contiguous() ? A.transpose(-2, -1) : A;  // det(A^T) = det(A)
    Tensor LU_tensor;
    Tensor pivots_tensor;
    std::tie(LU_tensor, pivots_tensor, std::ignore) =
        lu_factor_ex_impl(work, true, false, "linalg.lu_factor");
    const int64_t batch_size = batch_count_of(LU_tensor);
    const int64_t n = LU_tensor.size(-1);
    const auto batch = batch_shape_of(A);
    Tensor sign = Tensor::empty(batch, A.dtype(), A.device());
    const DType value_dtype = isComplexType(A.dtype()) ? toRealValueType(A.dtype()) : A.dtype();
    Tensor logabsdet = Tensor::empty(batch, value_dtype, A.device());
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        using R = typename LinalgScalarTraits<T>::value_type;
        auto* lu = LU_tensor.data_ptr<T>();
        auto* s_out = sign.data_ptr<T>();
        auto* l_out = logabsdet.data_ptr<R>();
        const auto* piv = pivots_tensor.data_ptr<int32_t>();
        const int64_t ms = matrix_stride_of(LU_tensor);
        const R neg_inf = -std::numeric_limits<R>::infinity();
        for (int64_t b = 0; b < batch_size; ++b) {
            R logdet = R(0);
            T det = T(1);
            bool singular = false;
            for (int64_t i = 0; i < n; ++i) {
                const T d = lu[b * ms + i * n + i];
                if (linalg_abs(d) == R(0)) { singular = true; break; }
                logdet += std::log(linalg_abs(d));
                det *= d;
            }
            const int64_t perm_sign = lu_perm_sign(&piv[b * n], n);
            if (singular) {
                s_out[b] = T(0);
                l_out[b] = neg_inf;
            } else {
                det *= T(perm_sign);
                if constexpr (LinalgScalarTraits<T>::is_complex) {
                    s_out[b] = det / static_cast<R>(linalg_abs(det));
                } else {
                    s_out[b] = det < T(0) ? T(-1) : T(1);
                }
                l_out[b] = logdet;
            }
        }
    });
    return {sign, logabsdet, LU_tensor, pivots_tensor};
}

std::tuple<Tensor, Tensor> linalg_slogdet_kernel(const Tensor& A) {
    auto values = linalg_slogdet_internal_kernel(A);
    return {std::get<0>(values), std::get<1>(values)};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_slogdet_internal_out_kernel(
        const Tensor& A, Tensor& sign, Tensor& logabsdet, Tensor& LU,
        Tensor& pivots) {
    auto values = linalg_slogdet_internal_kernel(A);
    write_linalg_output("linalg.slogdet", std::get<0>(values), sign);
    write_linalg_output("linalg.slogdet", std::get<1>(values), logabsdet);
    write_linalg_output("linalg.slogdet", std::get<2>(values), LU);
    write_linalg_output("linalg.slogdet", std::get<3>(values), pivots);
    return {sign, logabsdet, LU, pivots};
}

// ------------------------------------------------------------- getrs solve --

// Core: solve op(A) X = B given LU/ipiv of A.  B is overwritten in place and
// must be column-major with ldb = B.size(-2).  Uses the same solve core without
// broadcasting (broadcasting resolved by callers).  Both operands are indexed
// by an explicit element offset, so a batched call must advance the
// factorization alongside the right-hand side.
template <typename scalar_t>
void getrs_inplace(char trans, const Tensor& LU, int64_t lu_offset,
                   const int32_t* pivots, int64_t pivots_stride, Tensor& B,
                   int64_t b_offset) {
    auto* lu = LU.data_ptr<scalar_t>() + lu_offset;
    auto* b = B.data_ptr<scalar_t>() + b_offset;
    const int64_t n = LU.size(-2);
    const int64_t nrhs = B.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);
    const int64_t ldb = std::max<int64_t>(1, B.size(-2));
    std::vector<int64_t> ipiv(pivots_stride);
    for (int64_t j = 0; j < pivots_stride; ++j) ipiv[j] = pivots[j];
    int64_t info;
    info = lapack_getrs(trans, n, nrhs, lu, lda, ipiv.data(), b, ldb);
    (void)info;  // only reports bad arguments
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_solve_ex_internal_kernel(
        const Tensor& A, const Tensor& B, bool left, bool check_errors) {
    const char* api = "linalg.solve_ex";
    require_lapack(api);
    bool vector_case = B.dim() == 1;
    if (!vector_case && A.dim() - 1 == B.dim()) {
        vector_case = true;
        for (int64_t i = 0; i < A.dim() - 1; ++i) {
            if (A.size(i) != B.size(i)) {
                vector_case = false;
                break;
            }
        }
    }
    Tensor B_2d = vector_case ? B.unsqueeze(-1) : B;
    check_inputs_solver(A, B_2d, left, api);
    if (!left && vector_case) {
        TP_THROW(RuntimeError,
                 "linalg.solve: Vector right-hand sides are not supported when left is false");
    }

    const auto batch = broadcast_batch(A, B_2d);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(A, batch));
    const int64_t n = A.size(-2);
    Tensor pivots = empty_pivots(A, batch, n);
    Tensor info = empty_info_like(A, batch);
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_lu_factor<T>(LU_work, pivots, info);
    });

    const bool rhs_transposed = !left;
    Tensor work_cm;
    if (rhs_transposed) {
        work_cm = clone_batched_column_major(
            expand_to_batch(B_2d, batch).conj().transpose(-2, -1));
    } else {
        work_cm = clone_batched_column_major(expand_to_batch(B_2d, batch));
    }
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(std::accumulate(
                                              batch.begin(), batch.end(), int64_t{1},
                                              std::multiplies<int64_t>())));
    const char trans = rhs_transposed
        ? (isComplexType(A.dtype()) ? 'C' : 'T')
        : 'N';
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* piv = pivots.data_ptr<int32_t>();
        const int64_t piv_stride = pivots.size(-1);
        const int64_t lu_ms = matrix_stride_of(LU_work);
        const int64_t rhs_ms = matrix_stride_of(work_cm);
        for (int64_t i = 0; i < bs; ++i) {
            if (info.data_ptr<int32_t>()[i] != 0) continue;
            getrs_inplace<T>(trans, LU_work, i * lu_ms, &piv[i * piv_stride],
                             pivots.size(-1), work_cm, i * rhs_ms);
        }
    });

    Tensor result;
    if (rhs_transposed) {
        result = work_cm.contiguous().conj().transpose(-2, -1).contiguous();
    } else {
        result = work_cm.contiguous();
    }
    if (vector_case) result = result.squeeze(-1);
    if (check_errors) linalg_check_errors(info, api, A.dim() == 2 && B.dim() == 2);
    return {result, LU_work.contiguous(), pivots.contiguous(), info.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_solve_ex_kernel(
        const Tensor& A, const Tensor& B, bool left, bool check_errors) {
    auto values = linalg_solve_ex_internal_kernel(A, B, left, check_errors);
    return {std::get<0>(values), std::get<3>(values)};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_solve_ex_internal_out_kernel(
        const Tensor& A, const Tensor& B, bool left, bool check_errors,
        Tensor& result, Tensor& LU, Tensor& pivots, Tensor& info) {
    auto values = linalg_solve_ex_internal_kernel(A, B, left, check_errors);
    write_linalg_output("linalg.solve_ex", std::get<0>(values), result);
    write_linalg_output("linalg.solve_ex", std::get<1>(values), LU);
    write_linalg_output("linalg.solve_ex", std::get<2>(values), pivots);
    write_linalg_output("linalg.solve_ex", std::get<3>(values), info);
    return {result, LU, pivots, info};
}

Tensor linalg_solve_kernel(const Tensor& A, const Tensor& B, bool left) {
    auto [result, info] = linalg_solve_ex_kernel(A, B, left, false);
    linalg_check_errors(info, "linalg.solve", A.dim() == 2);
    return result;
}

// linalg_solve_ex_out with result pre-filled with the identity).
std::tuple<Tensor, Tensor> linalg_inv_ex_kernel(const Tensor& A, bool check_errors) {
    // batch_shape_of here collapsed 2-D inputs to a 0-D scalar and made
    // linalg.inv reject its own RHS.
    Tensor identity = Tensor::empty(static_cast<std::vector<int64_t>>(A.shape()),
                                    A.dtype(), A.device());
    run_linalg(A.dtype(), [&](auto tag) {
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
    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &input_data[i * input_matrix_stride];
        const char uplo = upper ? 'U' : 'L';
        int64_t err;
        err = lapack_potrf(uplo, n, a, lda);
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
    run_linalg(A.dtype(), [&](auto tag) {
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
    if (A.device() != B.device()) {
        TP_THROW(DeviceMismatchError, api, ": A and B must be on the same device");
    }
    if (A.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, api, ": A and B must have the same dtype, but got ",
                 pretty_dtype_name(A.dtype()), " and ", pretty_dtype_name(B.dtype()));
    }
    if (!(left ? A.size(-1) == B.size(-2) : A.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B");
    }
    const auto batch = broadcast_batch(A, B);
    Tensor B_work = clone_batched_column_major(expand_to_batch(B, batch));
    Tensor A_work = clone_batched_column_major(expand_to_batch(A, batch));
    const char side = left ? 'L' : 'R';
    const char uplo = upper ? 'U' : 'L';
    const char diag = unitriangular ? 'U' : 'N';
    const int64_t bs = std::max<int64_t>(1, static_cast<int64_t>(std::accumulate(
                                            batch.begin(), batch.end(), int64_t{1},
                                            std::multiplies<int64_t>())));
    run_linalg(A.dtype(), [&](auto tag) {
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
            info = lapack_trtrs(side, uplo, 'N', diag, m, n, &a[i * a_ms], lda,
                                &b[i * b_ms], ldb);
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
    if (B.dim() < 1) {
        TP_THROW(RuntimeError, api, ": B must have at least 1 dimension");
    }
    if (LU.device() != B.device()) {
        TP_THROW(DeviceMismatchError, api, ": LU and B must be on the same device");
    }
    if (LU.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, api, ": LU and B must have the same dtype, but got ",
                 pretty_dtype_name(LU.dtype()), " and ", pretty_dtype_name(B.dtype()));
    }
    bool vector_case = B.dim() == 1;
    if (!vector_case && LU.dim() - 1 == B.dim()) {
        vector_case = true;
        for (int64_t i = 0; i < LU.dim() - 1; ++i) {
            if (LU.size(i) != B.size(i)) {
                vector_case = false;
                break;
            }
        }
    }
    Tensor B_2d = vector_case ? B.unsqueeze(-1) : B;
    if (pivots.dtype() != DType::Int32 || pivots.device() != LU.device()) {
        TP_THROW(RuntimeError, api, ": pivots must be Int32 on the same device as LU");
    }
    if (pivots.dim() < 1 || pivots.size(-1) != LU.size(-1)) {
        TP_THROW(RuntimeError, api, ": pivots must contain one entry per matrix column");
    }
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
    if (!(left ? LU.size(-2) == B_2d.size(-2) : LU.size(-1) == B_2d.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of LU and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", LU.size(-2), "x", LU.size(-1), " and ",
                 B_2d.size(-2), "x", B_2d.size(-1), ")");
    }
    const auto batch = broadcast_batch(LU, B_2d);
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
        work_cm = clone_batched_column_major(
            expand_to_batch(B_2d, batch).conj().transpose(-2, -1));
    } else {
        work_cm = clone_batched_column_major(expand_to_batch(B_2d, batch));
    }
    const char trans = left
        ? (adjoint ? (isComplexType(LU.dtype()) ? 'C' : 'T') : 'N')
        : (adjoint ? 'N' : (isComplexType(LU.dtype()) ? 'C' : 'T'));
    run_linalg(LU.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* piv = piv_exp.data_ptr<int32_t>();
        const int64_t piv_stride = pivots.size(-1);
        const int64_t lu_ms = matrix_stride_of(LU_work);
        const int64_t rhs_ms = matrix_stride_of(work_cm);
        for (int64_t i = 0; i < bs; ++i) {
            getrs_inplace<T>(trans, LU_work, i * lu_ms, &piv[i * piv_stride],
                             pivots.size(-1), work_cm, i * rhs_ms);
        }
    });
    if (rhs_transposed) {
        Tensor result = work_cm.contiguous().conj().transpose(-2, -1).contiguous();
        if (vector_case) result = result.squeeze(-1);
        return result;
    }
    Tensor result = work_cm.contiguous();
    if (vector_case) result = result.squeeze(-1);
    return result;
}

// ------------------------------------------------------------------ syevd --

template <typename scalar_t>
void apply_syevd(const Tensor& vectors, const Tensor& values, const Tensor& infos,
                 bool upper, bool compute_eigenvectors) {
    using value_t = typename LinalgScalarTraits<scalar_t>::value_type;
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    const char uplo = upper ? 'U' : 'L';
    const char jobz = compute_eigenvectors ? 'V' : 'N';
    auto* vectors_data = vectors.data_ptr<scalar_t>();
    auto* values_data = values.data_ptr<value_t>();
    auto* infos_data = infos.data_ptr<int32_t>();
    const int64_t vectors_stride = matrix_stride_of(vectors);
    const int64_t values_stride = values.size(-1);
    const int64_t batch_size = batch_count_of(vectors);
    const int64_t n = vectors.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);

    int64_t lwork = -1;
    int64_t lrwork = -1;
    int64_t liwork = -1;
    std::vector<scalar_t> work(1);
    std::vector<value_t> rwork(1);
    std::vector<int64_t> iwork(1);
    if constexpr (is_float) {
        lapack_ssyevd(jobz, uplo, n, vectors_data, lda, values_data,
                      work.data(), lwork, iwork.data(), liwork);
    } else if constexpr (std::is_same_v<scalar_t, double>) {
        lapack_dsyevd(jobz, uplo, n, vectors_data, lda, values_data,
                      work.data(), lwork, iwork.data(), liwork);
    } else if constexpr (std::is_same_v<scalar_t, complex<float>>) {
        lapack_cheevd(jobz, uplo, n, vectors_data, lda, values_data, work.data(),
                      lwork, rwork.data(), lrwork, iwork.data(), liwork);
    } else {
        lapack_zheevd(jobz, uplo, n, vectors_data, lda, values_data, work.data(),
                      lwork, rwork.data(), lrwork, iwork.data(), liwork);
    }
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0].real()));
        lrwork = std::max<int64_t>(1, static_cast<int64_t>(rwork[0]));
    } else {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    }
    liwork = std::max<int64_t>(1, iwork[0]);
    work.resize(lwork);
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) rwork.resize(lrwork);
    iwork.resize(liwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* v = &vectors_data[i * vectors_stride];
        value_t* w = &values_data[i * values_stride];
        int64_t err;
        if constexpr (is_float) {
            err = lapack_ssyevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                iwork.data(), liwork);
        } else if constexpr (std::is_same_v<scalar_t, double>) {
            err = lapack_dsyevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                iwork.data(), liwork);
        } else if constexpr (std::is_same_v<scalar_t, complex<float>>) {
            err = lapack_cheevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                rwork.data(), lrwork, iwork.data(), liwork);
        } else {
            err = lapack_zheevd(jobz, uplo, n, v, lda, w, work.data(), lwork,
                                rwork.data(), lrwork, iwork.data(), liwork);
        }
        infos_data[i] = static_cast<int32_t>(err);
        if (err != 0) break;
    }
}

std::tuple<Tensor, Tensor> eigh_impl(const Tensor& A, bool upper, bool compute_eigenvectors) {
    require_lapack("linalg.eigh");
    square_check_inputs(A, "linalg.eigh");
    Tensor vectors = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    Tensor values = Tensor::empty(
        batch.empty() ? std::vector<int64_t>{A.size(-1)}
                      : cat_batch(batch, std::vector<int64_t>{A.size(-1)}),
        toRealValueType(A.dtype()), A.device());
    Tensor info = empty_info_like(A, batch);
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_syevd<T>(vectors, values, info, upper, compute_eigenvectors);
    });
    linalg_check_errors(info, "linalg.eigh", A.dim() == 2);
    return {values.contiguous(), vectors.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_eigh_internal_kernel(const Tensor& A,
                                                       std::string UPLO,
                                                       bool compute_v) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return eigh_impl(A, UPLO == "U", compute_v);
}

// Public entries: schema passes UPLO as a string.
std::tuple<Tensor, Tensor> linalg_eigh_kernel(const Tensor& A, std::string UPLO) {
    return linalg_eigh_internal_kernel(A, UPLO, true);
}

Tensor linalg_eigvalsh_kernel(const Tensor& A, std::string UPLO) {
    return std::get<0>(linalg_eigh_internal_kernel(A, UPLO, false));
}

std::tuple<Tensor, Tensor> linalg_eigh_internal_out_kernel(
        const Tensor& A, std::string UPLO, bool compute_v, Tensor& values,
        Tensor& vectors) {
    auto result = linalg_eigh_internal_kernel(A, UPLO, compute_v);
    write_linalg_output("linalg.eigh", std::get<0>(result), values);
    write_linalg_output("linalg.eigh", std::get<1>(result), vectors);
    return {values, vectors};
}

std::tuple<Tensor, Tensor> linalg_eigh_eigvals_out_kernel(
        const Tensor& A, std::string UPLO, Tensor& values, Tensor& vectors) {
    return linalg_eigh_internal_out_kernel(A, UPLO, true, values, vectors);
}

Tensor& linalg_eigvalsh_out_kernel(const Tensor& A, std::string UPLO,
                                   Tensor& out) {
    auto result = linalg_eigh_internal_kernel(A, UPLO, false);
    write_linalg_output("linalg.eigvalsh", std::get<0>(result), out);
    return out;
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

template <typename scalar_t>
void apply_geev_complex(const Tensor& input, const Tensor& values,
                        const Tensor& vectors, const Tensor& infos,
                        bool compute_eigenvectors) {
    using value_t = typename LinalgScalarTraits<scalar_t>::value_type;
    auto* a_data = input.data_ptr<scalar_t>();
    auto* w_data = values.data_ptr<scalar_t>();
    auto* v_data = compute_eigenvectors ? vectors.data_ptr<scalar_t>() : nullptr;
    auto* infos_data = infos.data_ptr<int32_t>();
    const char jobvl = 'N';
    const char jobvr = compute_eigenvectors ? 'V' : 'N';
    const int64_t n = input.size(-1);
    const int64_t lda = std::max<int64_t>(1, n);
    const int64_t ldvr = compute_eigenvectors ? lda : 1;
    const int64_t matrix_stride = matrix_stride_of(input);
    std::vector<scalar_t> work(1);
    std::vector<value_t> rwork(static_cast<size_t>(std::max<int64_t>(1, 2 * n)));
    int64_t lwork = -1;
    if constexpr (std::is_same_v<scalar_t, complex<float>>) {
        lapack_cgeev(jobvl, jobvr, n, a_data, lda, w_data, nullptr, 1, v_data,
                     ldvr, work.data(), lwork, rwork.data());
    } else {
        lapack_zgeev(jobvl, jobvr, n, a_data, lda, w_data, nullptr, 1, v_data,
                     ldvr, work.data(), lwork, rwork.data());
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0].real()));
    work.resize(static_cast<size_t>(lwork));

    const int64_t batch_size = batch_count_of(input);
    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &a_data[i * matrix_stride];
        scalar_t* w = &w_data[i * n];
        scalar_t* v = compute_eigenvectors ? &v_data[i * matrix_stride] : nullptr;
        int64_t err;
        if constexpr (std::is_same_v<scalar_t, complex<float>>) {
            err = lapack_cgeev(jobvl, jobvr, n, a, lda, w, nullptr, 1, v, ldvr,
                               work.data(), lwork, rwork.data());
        } else {
            err = lapack_zgeev(jobvl, jobvr, n, a, lda, w, nullptr, 1, v, ldvr,
                               work.data(), lwork, rwork.data());
        }
        infos_data[i] = static_cast<int32_t>(err);
    }
}

// Real-input GEEV stores conjugate pairs in adjacent real columns.
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
    Tensor input = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    if (isComplexType(A.dtype())) {
        Tensor values = Tensor::empty(repeat_batch(batch, n), A.dtype(), A.device());
        Tensor eigvecs;
        if (compute_eigenvectors) {
            eigvecs = empty_column_major(cat_batch(batch, {n, n}), A.dtype(), A.device());
        }
        Tensor info = empty_info_like(A, batch);
        run_linalg_complex(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_geev_complex<T>(input, values, eigvecs, info, compute_eigenvectors);
        });
        linalg_check_errors(info, "linalg.eig", A.dim() == 2);
        return {values.contiguous(), compute_eigenvectors ? eigvecs.contiguous() : eigvecs};
    }
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

std::tuple<Tensor, Tensor> linalg_eig_out_kernel(const Tensor& A,
                                                 Tensor& values,
                                                 Tensor& vectors) {
    auto result = linalg_eig_kernel(A);
    write_linalg_output("linalg.eig", std::get<0>(result), values);
    write_linalg_output("linalg.eig", std::get<1>(result), vectors);
    return {values, vectors};
}

Tensor& linalg_eigvals_out_kernel(const Tensor& A, Tensor& values) {
    write_linalg_output("linalg.eigvals", linalg_eigvals_kernel(A), values);
    return values;
}

// ------------------------------------------------------------------- gesdd --

template <typename scalar_t>
void apply_svd(const Tensor& A, bool full_matrices, bool compute_uv,
               const Tensor& U, const Tensor& S, const Tensor& Vh, const Tensor& info) {
    using value_t = typename LinalgScalarTraits<scalar_t>::value_type;
    constexpr bool is_float = std::is_same_v<scalar_t, float>;
    constexpr bool is_double = std::is_same_v<scalar_t, double>;
    auto* a_data = A.data_ptr<scalar_t>();
    auto* u_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
    auto* s_data = S.data_ptr<value_t>();
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
    std::vector<int64_t> iwork(static_cast<size_t>(std::max<int64_t>(1, 8 * k)));

    int64_t lwork = -1;
    std::vector<scalar_t> work(1);
    std::vector<value_t> rwork;
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        rwork.resize(static_cast<size_t>(std::max<int64_t>(1,
                                                           svd_real_workspace(jobz, m, n))));
    }
    value_t* rwork_data = rwork.empty() ? nullptr : rwork.data();
    if constexpr (is_float) {
        lapack_sgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, iwork.data());
    } else if constexpr (is_double) {
        lapack_dgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, iwork.data());
    } else if constexpr (std::is_same_v<scalar_t, complex<float>>) {
        lapack_cgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, rwork_data, iwork.data());
    } else {
        lapack_zgesdd(jobz, m, n, a_data, lda, s_data, u_data, ldu, vh_data, ldvh,
                      work.data(), lwork, rwork_data, iwork.data());
    }
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0].real()));
    } else {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    }
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t err;
        if constexpr (is_float) {
            err = lapack_sgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, iwork.data());
        } else if constexpr (is_double) {
            err = lapack_dgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, iwork.data());
        } else if constexpr (std::is_same_v<scalar_t, complex<float>>) {
            err = lapack_cgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, rwork_data, iwork.data());
        } else {
            err = lapack_zgesdd(jobz, m, n, &a_data[i * a_stride], lda,
                                &s_data[i * s_stride],
                                compute_uv ? &u_data[i * u_stride] : nullptr, ldu,
                                compute_uv ? &vh_data[i * vh_stride] : nullptr, ldvh,
                                work.data(), lwork, rwork_data, iwork.data());
        }
        info_data[i] = static_cast<int32_t>(err);
    }
}

std::tuple<Tensor, Tensor, Tensor> svd_impl(const Tensor& A, bool full_matrices,
                                            bool compute_uv) {
    require_lapack("linalg.svd");
    check_is_matrix(A, "linalg.svd");
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
    S = Tensor::empty(cat_batch(batch, std::vector<int64_t>{k}),
                      toRealValueType(A.dtype()), A.device());
    Tensor info = empty_info_like(A, batch);
    if (A.numel() == 0) {
        if (compute_uv && full_matrices) {
            if (U.numel() != 0) {
                U.zero_();
                U.diagonal(0, -2, -1).fill_(Scalar(1));
            }
            if (Vh.numel() != 0) {
                Vh.zero_();
                Vh.diagonal(0, -2, -1).fill_(Scalar(1));
            }
        }
    } else {
        Tensor a_copy = clone_batched_column_major(A);
        run_linalg(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_svd<T>(a_copy, full_matrices, compute_uv, U, S, Vh, info);
        });
    }
    linalg_check_errors(info, "linalg.svd", A.dim() == 2);
    if (!compute_uv) {
        U = Tensor::empty(cat_batch(batch, std::vector<int64_t>{m, 0}), A.dtype(), A.device());
        Vh = Tensor::empty(cat_batch(batch, std::vector<int64_t>{0, n}), A.dtype(), A.device());
    }
    return {U.contiguous(), S.contiguous(), Vh.contiguous()};
}

void check_cpu_svd_driver(const std::optional<std::string>& driver) {
    if (driver.has_value()) {
        TP_THROW(RuntimeError,
                 "linalg.svd: keyword argument `driver=` is only supported on CUDA inputs");
    }
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_internal_kernel(
        const Tensor& A, bool full_matrices, bool compute_uv,
        std::optional<std::string> driver) {
    check_cpu_svd_driver(driver);
    return svd_impl(A, full_matrices, compute_uv);
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_kernel(const Tensor& A, bool full_matrices,
                                                     std::optional<std::string> driver) {
    return linalg_svd_internal_kernel(A, full_matrices, true, driver);
}

Tensor linalg_svdvals_kernel(const Tensor& A, std::optional<std::string> driver) {
    return std::get<1>(linalg_svd_internal_kernel(A, false, false, driver));
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_internal_out_kernel(
        const Tensor& A, bool full_matrices, bool compute_uv,
        std::optional<std::string> driver, Tensor& U, Tensor& S, Tensor& Vh) {
    auto result = linalg_svd_internal_kernel(A, full_matrices, compute_uv,
                                             driver);
    write_linalg_output("linalg.svd", std::get<0>(result), U);
    write_linalg_output("linalg.svd", std::get<1>(result), S);
    write_linalg_output("linalg.svd", std::get<2>(result), Vh);
    return {U, S, Vh};
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_out_kernel(
        const Tensor& A, bool full_matrices, std::optional<std::string> driver,
        Tensor& U, Tensor& S, Tensor& Vh) {
    return linalg_svd_internal_out_kernel(A, full_matrices, true, driver, U,
                                          S, Vh);
}

Tensor& linalg_svdvals_out_kernel(const Tensor& A,
                                  std::optional<std::string> driver,
                                  Tensor& out) {
    auto result = linalg_svd_internal_kernel(A, false, false, driver);
    write_linalg_output("linalg.svdvals", std::get<1>(result), out);
    return out;
}

std::tuple<Tensor, Tensor> linalg_polar_kernel(const Tensor& A) {
    check_is_matrix(A, "linalg.polar");
    if (A.size(-2) < A.size(-1)) {
        TP_THROW(RuntimeError,
                 "linalg.polar: input must have at least as many rows as columns, but got ",
                 A.size(-2), " by ", A.size(-1), " matrices");
    }

    auto [Up, S, Vh] = ops::linalg_svd(A, false, std::nullopt);
    Tensor scaled_vh = ops::mul(ops::unsqueeze(S, -1), Vh);
    Tensor H = ops::matmul(ops::mH(Vh), scaled_vh);
    H = ops::mul(ops::add(H, ops::mH(H)), Scalar(0.5));
    Tensor U = ops::matmul(Up, Vh);
    return {U.contiguous(), H.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_polar_out_kernel(const Tensor& A, Tensor& U,
                                                   Tensor& H) {
    auto result = linalg_polar_kernel(A);
    write_linalg_output("linalg.polar", std::get<0>(result), U);
    write_linalg_output("linalg.polar", std::get<1>(result), H);
    return {U, H};
}

// ------------------------------------------------------------- geqrf/orgqr --

template <typename scalar_t>
void apply_geqrf(const Tensor& input, const Tensor& tau) {
    if (input.numel() == 0) return;
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
    lapack_geqrf(m, n, input_data, lda, tau_data, work.data(), lwork);
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        lwork = std::max<int64_t>(n, static_cast<int64_t>(work[0].real()));
    } else {
        lwork = std::max<int64_t>(n, static_cast<int64_t>(work[0]));
    }
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &input_data[i * input_matrix_stride];
        scalar_t* t = &tau_data[i * tau_stride];
        lapack_geqrf(m, n, a, lda, t, work.data(), lwork);
    }
}

template <typename scalar_t>
void apply_orgqr(Tensor& self, const Tensor& tau) {
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
    lapack_orgqr(m, n, k, self_data, lda, tau_data, work.data(), lwork);
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0].real()));
    } else {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    }
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* s = &self_data[i * self_matrix_stride];
        const scalar_t* t = &tau_data[i * tau_stride];
        lapack_orgqr(m, n, k, s, lda, t, work.data(), lwork);
    }
}

std::tuple<Tensor, Tensor> linalg_qr_kernel(const Tensor& A, std::string mode) {
    require_lapack("linalg.qr");
    check_is_matrix(A, "linalg.qr");
    if (mode != "reduced" && mode != "complete" && mode != "r") {
        TP_THROW(RuntimeError, "linalg.qr: mode '", mode,
                 "' not recognized. Mode must be one of 'reduced', 'complete' or 'r'");
    }
    const bool compute_q = mode != "r";
    const bool reduced = mode == "reduced" || !compute_q;
    Tensor QR = clone_batched_column_major(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    Tensor tau = Tensor::empty(cat_batch(batch, std::vector<int64_t>{k}), A.dtype(), A.device());
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geqrf<T>(QR, tau);
    });

    Tensor Q_in = Tensor::empty({0}, A.dtype(), A.device());
    if (compute_q) {
        const int64_t qcols = reduced ? k : m;
        // Pack the first qcols columns of the reflector buffer into an
        // (m x qcols) column-major buffer for orgqr.
        Q_in = empty_column_major(cat_batch(batch, std::vector<int64_t>{m, qcols}),
                                  A.dtype(), A.device());
        run_linalg(A.dtype(), [&](auto tag) {
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
    }

    // R: upper triangle of the geqrf buffer.
    const int64_t rrows = reduced || !compute_q ? k : m;
    Tensor R = empty_column_major(cat_batch(batch, std::vector<int64_t>{rrows, n}),
                                  A.dtype(), A.device());
    run_linalg(A.dtype(), [&](auto tag) {
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
    return {compute_q ? Q_in.contiguous() : Tensor::empty({0}, A.dtype(), A.device()),
            R.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_qr_out_kernel(const Tensor& A, std::string mode,
                                                Tensor& Q, Tensor& R) {
    auto result = linalg_qr_kernel(A, mode);
    write_linalg_output("linalg.qr", std::get<0>(result), Q);
    write_linalg_output("linalg.qr", std::get<1>(result), R);
    return {Q, R};
}

Tensor linalg_householder_product_kernel(const Tensor& input, const Tensor& tau) {
    require_lapack("linalg.householder_product");
    check_is_matrix(input, "linalg.householder_product");
    if (input.size(-2) < input.size(-1)) {
        TP_THROW(RuntimeError, "linalg.householder_product: If input has size (..., m, n), "
                 "n must be less than or equal to m, but got n = ",
                 input.size(-1), " and m = ", input.size(-2));
    }
    if (tau.dim() < 1 || input.size(-1) < tau.size(-1)) {
        TP_THROW(RuntimeError, "linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]");
    }
    if (input.dim() - tau.dim() != 1) {
        TP_THROW(RuntimeError, "linalg.householder_product: Expected tau to have one dimension less than input, but got tau.ndim equal to ",
                 tau.dim(), " and input.ndim is equal to ", input.dim());
    }
    for (int64_t i = 0; i < input.dim() - 2; ++i) {
        if (input.size(i) != tau.size(i)) {
            TP_THROW(RuntimeError, "linalg.householder_product: Expected batch dimensions of tau to be equal to input.shape[:-2]");
        }
    }
    if (tau.size(-1) != std::min(input.size(-2), input.size(-1))) {
        TP_THROW(RuntimeError, "linalg.householder_product: If tau has size (..., k), then "
                               "when input has size (..., m, n) we require k == min(m, n)");
    }
    if (tau.dtype() != input.dtype()) {
        TP_THROW(RuntimeError, "linalg.householder_product: input and tau must have the same dtype");
    }
    if (tau.device() != input.device()) {
        TP_THROW(DeviceMismatchError, "linalg.householder_product: input and tau must be on the same device");
    }
    Tensor tau_work = tau.is_contiguous() ? tau : tau.contiguous();
    Tensor result = clone_batched_column_major(input);
    run_linalg(input.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_orgqr<T>(result, tau_work);
    });
    return result.contiguous();
}

Tensor& linalg_householder_product_out_kernel(const Tensor& input,
                                              const Tensor& tau, Tensor& out) {
    write_linalg_output("linalg.householder_product",
                        linalg_householder_product_kernel(input, tau), out);
    return out;
}

template <typename T>
void apply_lstsq(const Tensor& A, Tensor& B, Tensor& rank,
                 Tensor& singular_values, double rcond, LstsqDriver driver) {
    using R = typename LinalgScalarTraits<T>::value_type;
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t nrhs = B.size(-1);
    const int64_t lda = std::max<int64_t>(1, m);
    const int64_t ldb = std::max<int64_t>({int64_t{1}, m, n});
    const int64_t bs = linear_batch_size(batch_shape_of(A));
    if (bs == 0) return;

    auto* a_data = A.data_ptr<T>();
    auto* b_data = B.data_ptr<T>();
    auto* rank_data = driver == LstsqDriver::Gels
        ? static_cast<int64_t*>(nullptr) : rank.data_ptr<int64_t>();
    auto* s_data = (driver == LstsqDriver::Gelsd || driver == LstsqDriver::Gelss)
        ? singular_values.data_ptr<R>() : static_cast<R*>(nullptr);

    std::vector<int64_t> jpvt;
    if (driver == LstsqDriver::Gelsy) {
        jpvt.resize(static_cast<size_t>(std::max<int64_t>(1, n)));
    }
    std::vector<R> rwork;
    std::vector<int64_t> iwork;
    T work_opt{};
    R rwork_opt{};
    int64_t iwork_opt = 0;
    int64_t rank_opt = 0;
    int64_t info = lapack_lstsq_call(
        driver, m, n, nrhs, a_data, lda, b_data, ldb, &work_opt, -1,
        jpvt.empty() ? nullptr : jpvt.data(), static_cast<R>(rcond), &rank_opt,
        &rwork_opt, s_data, &iwork_opt);
    if (info != 0) {
        TP_THROW(RuntimeError, "linalg.lstsq: workspace query failed with error code ", info);
    }
    int64_t lwork = 1;
    if constexpr (LinalgScalarTraits<T>::is_complex) {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work_opt.real()));
    } else {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work_opt));
    }
    std::vector<T> work(static_cast<size_t>(lwork));

    if constexpr (LinalgScalarTraits<T>::is_complex) {
        if (driver != LstsqDriver::Gels) {
            int64_t rwork_size = 1;
            if (driver == LstsqDriver::Gelsy) {
                rwork_size = std::max<int64_t>(1, 2 * n);
            } else if (driver == LstsqDriver::Gelss) {
                rwork_size = std::max<int64_t>(1, 5 * std::min(m, n));
            } else {
                rwork_size = std::max<int64_t>(1, rwork_opt);
            }
            rwork.resize(static_cast<size_t>(rwork_size));
        }
    }
    if (driver == LstsqDriver::Gelsd) {
        iwork.resize(static_cast<size_t>(std::max<int64_t>(1, iwork_opt)));
    }

    for (int64_t i = 0; i < bs; ++i) {
        if (!jpvt.empty()) std::fill(jpvt.begin(), jpvt.end(), int64_t{0});
        int64_t rank_value = 0;
        int64_t info_value = lapack_lstsq_call(
            driver, m, n, nrhs, a_data + i * m * n, lda,
            b_data + i * ldb * nrhs, ldb, work.data(), lwork,
            jpvt.empty() ? nullptr : jpvt.data(), static_cast<R>(rcond),
            &rank_value, rwork.empty() ? nullptr : rwork.data(),
            s_data ? s_data + i * std::min(m, n) : nullptr,
            iwork.empty() ? nullptr : iwork.data());
        if (info_value != 0) {
            TP_THROW(RuntimeError, "linalg.lstsq: (Batch element ", i,
                     ") The least squares solution could not be computed (error code: ",
                     info_value, ").");
        }
        if (rank_data) rank_data[i] = rank_value;
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq_kernel(
        const Tensor& A, const Tensor& B, std::optional<double> rcond,
        std::optional<std::string> driver_opt) {
    const char* api = "linalg.lstsq";
    require_lapack(api);
    check_is_matrix(A, api);
    if (B.dim() < 1) {
        TP_THROW(RuntimeError, api, ": B must have at least 1 dimension");
    }
    const int64_t dim_diff = A.dim() - B.dim();
    if (dim_diff < 0 || dim_diff > 1) {
        TP_THROW(RuntimeError, api,
                 ": A and B must have compatible numbers of dimensions");
    }
    if (A.device() != B.device()) {
        TP_THROW(DeviceMismatchError, api, ": A and B must be on the same device");
    }
    if (A.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, api, ": A and B must have the same dtype, but got ",
                 pretty_dtype_name(A.dtype()), " and ", pretty_dtype_name(B.dtype()));
    }

    bool vector_case = B.dim() == 1;
    if (!vector_case && A.dim() - 1 == B.dim()) {
        vector_case = true;
        for (int64_t i = 0; i < A.dim() - 1; ++i) {
            if (A.size(i) != B.size(i)) {
                vector_case = false;
                break;
            }
        }
    }
    Tensor B_2d = vector_case ? B.unsqueeze(-1) : B;
    if (A.size(-2) != B_2d.size(-2)) {
        TP_THROW(RuntimeError, api, ": A and B have incompatible row dimensions: ",
                 A.size(-2), " and ", B_2d.size(-2));
    }

    std::string driver = driver_opt.value_or("gelsy");
    std::transform(driver.begin(), driver.end(), driver.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    LstsqDriver driver_type;
    if (driver == "gels") driver_type = LstsqDriver::Gels;
    else if (driver == "gelsy") driver_type = LstsqDriver::Gelsy;
    else if (driver == "gelsd") driver_type = LstsqDriver::Gelsd;
    else if (driver == "gelss") driver_type = LstsqDriver::Gelss;
    else {
        TP_THROW(RuntimeError, api,
                 ": driver must be one of gels, gelsy, gelsd, or gelss");
    }

    const double epsilon = (A.dtype() == DType::Float32 ||
                            A.dtype() == DType::ComplexFloat)
        ? static_cast<double>(std::numeric_limits<float>::epsilon())
        : std::numeric_limits<double>::epsilon();
    const double rcond_value = rcond.value_or(
        epsilon * static_cast<double>(std::max(A.size(-2), A.size(-1))));
    const auto batch = broadcast_batch(A, B_2d);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t nrhs = B_2d.size(-1);
    const int64_t ldb = std::max<int64_t>({int64_t{1}, m, n});
    const int64_t bs = linear_batch_size(batch);

    Tensor A_work = clone_batched_column_major(expand_to_batch(A, batch));
    Tensor B_work = empty_column_major(
        cat_batch(batch, std::vector<int64_t>{ldb, nrhs}), B.dtype(), B.device());
    const Tensor B_source = expand_to_batch(B_2d, batch).contiguous();
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* dst = B_work.data_ptr<T>();
        const auto* src = B_source.data_ptr<T>();
        std::fill(dst, dst + B_work.numel(), T(0));
        for (int64_t i = 0; i < bs; ++i) {
            for (int64_t row = 0; row < m; ++row) {
                for (int64_t col = 0; col < nrhs; ++col) {
                    dst[i * ldb * nrhs + col * ldb + row] =
                        src[i * m * nrhs + row * nrhs + col];
                }
            }
        }
    });

    Tensor rank = driver_type == LstsqDriver::Gels
        ? Tensor::empty({0}, DType::Int64, B.device())
        : Tensor::empty(batch, DType::Int64, B.device());
    Tensor singular_values;
    if (driver_type == LstsqDriver::Gelsd || driver_type == LstsqDriver::Gelss) {
        singular_values = Tensor::empty(
            cat_batch(batch, std::vector<int64_t>{std::min(m, n)}),
            toRealValueType(A.dtype()), A.device());
    } else {
        singular_values = Tensor::empty(
            {0}, toRealValueType(A.dtype()), A.device());
    }
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_lstsq<T>(A_work, B_work, rank, singular_values, rcond_value,
                       driver_type);
    });

    const Tensor solved = B_work.contiguous();
    Tensor solution = solved.slice(-2, 0, n).contiguous();
    if (vector_case) solution = solution.squeeze(-1);

    bool compute_residuals = m > n && driver_type != LstsqDriver::Gelsy;
    if (compute_residuals &&
        (driver_type == LstsqDriver::Gelsd || driver_type == LstsqDriver::Gelss)) {
        const auto* rank_ptr = rank.data_ptr<int64_t>();
        for (int64_t i = 0; i < rank.numel(); ++i) {
            if (rank_ptr[i] != n) {
                compute_residuals = false;
                break;
            }
        }
    }
    Tensor residuals;
    if (!compute_residuals) {
        residuals = Tensor::empty({0}, toRealValueType(B.dtype()), B.device());
    } else {
        residuals = Tensor::empty(
            cat_batch(batch, std::vector<int64_t>{nrhs}),
            toRealValueType(B.dtype()), B.device());
        run_linalg(B.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            using R = typename LinalgScalarTraits<T>::value_type;
            const auto* src = solved.data_ptr<T>();
            auto* dst = residuals.data_ptr<R>();
            for (int64_t i = 0; i < bs; ++i) {
                for (int64_t col = 0; col < nrhs; ++col) {
                    R value = R(0);
                    for (int64_t row = n; row < m; ++row) {
                        const T x = src[i * ldb * nrhs + col * ldb + row];
                        if constexpr (LinalgScalarTraits<T>::is_complex) {
                            value += static_cast<R>(tensorplay::norm(x));
                        } else {
                            value += x * x;
                        }
                    }
                    dst[i * nrhs + col] = value;
                }
            }
        });
    }
    return {solution, residuals, rank, singular_values};
}

// --------------------------------------------------------------- sytrf LDL --

template <typename scalar_t>
void apply_ldl_factor(const Tensor& LD, const Tensor& pivots, const Tensor& info,
                      char uplo, bool hermitian) {
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
    lapack_ldl_factor(uplo, hermitian, n, a_data, lda, ipiv.data(), work.data(), lwork);
    if constexpr (LinalgScalarTraits<scalar_t>::is_complex) {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0].real()));
    } else {
        lwork = std::max<int64_t>(1, static_cast<int64_t>(work[0]));
    }
    work.resize(lwork);

    for (int64_t i = 0; i < batch_size; ++i) {
        scalar_t* a = &a_data[i * a_stride];
        int64_t err;
        err = lapack_ldl_factor(uplo, hermitian, n, a, lda, ipiv.data(), work.data(), lwork);
        for (int64_t j = 0; j < n; ++j)
            pivots_data[i * pivots_stride + j] = static_cast<int32_t>(ipiv[j]);
        info_data[i] = static_cast<int32_t>(err);
    }
}

std::tuple<Tensor, Tensor, Tensor> ldl_factor_impl(const Tensor& A, bool hermitian,
                                                   bool check_errors) {
    const char* api = check_errors ? "linalg.ldl_factor_ex" : "linalg.ldl_factor";
    require_lapack(api);
    square_check_inputs(A, api);
    Tensor LD = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    Tensor pivots = empty_pivots(A, batch, n);
    Tensor info = empty_info_like(A, batch);
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_ldl_factor<T>(LD, pivots, info, 'L', hermitian);
    });
    if (check_errors) linalg_check_errors(info, api, A.dim() == 2);
    return {LD.contiguous(), pivots, info};
}

Tensor ldl_solve_impl(const Tensor& LD, const Tensor& pivots, const Tensor& B,
                      bool hermitian) {
    const char* api = "linalg.ldl_solve";
    require_lapack(api);
    square_check_inputs(LD, api, "LD");
    check_is_matrix(B, api, "B");
    {
        Tensor pv64 = pivots.to(DType::Int64);
        const auto* pv = pv64.data_ptr<int64_t>();
        for (int64_t i = 0; i < pivots.numel(); ++i) {
            if (std::abs(pv[i]) < 1 || std::abs(pv[i]) > LD.size(-2)) {
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
    run_linalg(LD.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* a = LD_work.data_ptr<T>();
        auto* b = B_work.data_ptr<T>();
        const auto* piv = piv32.data_ptr<int32_t>();
        for (int64_t i = 0; i < bs; ++i) {
            std::vector<int64_t> ipiv(static_cast<size_t>(n));
            for (int64_t j = 0; j < n; ++j) ipiv[j] = piv[i * n + j];
            int64_t err;
            err = lapack_ldl_solve('L', hermitian, n, nrhs, &a[i * n * n], n,
                                    ipiv.data(), &b[i * n * nrhs], n);
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
    Tensor LU_tensor;
    Tensor pivots_tensor;
    std::tie(LU_tensor, pivots_tensor, std::ignore) =
        lu_factor_ex_impl(A, pivot, false, "linalg.lu_factor_ex");
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t kk = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const int64_t bs = linear_batch_size(batch);
    Tensor P = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{m, m}), A.dtype(), A.device());
    Tensor L = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{m, kk}), A.dtype(), A.device());
    Tensor U = Tensor::zeros(cat_batch(batch, std::vector<int64_t>{kk, n}), A.dtype(), A.device());
    run_linalg(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const auto* lu_all = LU_tensor.data_ptr<T>();  // column-major (*, m, n), lda = m
        const auto* piv = pivots_tensor.data_ptr<int32_t>();
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
    run_linalg(A.dtype(), [&](auto tag) {
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
        lu_factor_ex_impl(A, pivot, false, "linalg.lu_factor");
    (void)info;
    return {LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor> linalg_lu_factor_ex_kernel(const Tensor& A,
                                                              bool pivot,
                                                              bool check_errors) {
    return lu_factor_ex_impl(A, pivot, check_errors, "linalg.lu_factor_ex");
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
    m.impl("_linalg_check_errors", linalg_check_errors_kernel);
    m.impl("linalg_cholesky", linalg_cholesky_kernel);
    m.impl("linalg_cholesky_ex", linalg_cholesky_ex_kernel);
    m.impl("linalg_inv", linalg_inv_kernel);
    m.impl("linalg_inv_ex", linalg_inv_ex_kernel);
    m.impl("_linalg_det", linalg_det_internal_kernel);
    m.impl("_linalg_det.result", linalg_det_internal_out_kernel);
    m.impl("linalg_det", linalg_det_kernel);
    m.impl("_linalg_slogdet", linalg_slogdet_internal_kernel);
    m.impl("_linalg_slogdet.sign", linalg_slogdet_internal_out_kernel);
    m.impl("linalg_slogdet", linalg_slogdet_kernel);
    m.impl("linalg_solve", linalg_solve_kernel);
    m.impl("_linalg_solve_ex", linalg_solve_ex_internal_kernel);
    m.impl("_linalg_solve_ex.result", linalg_solve_ex_internal_out_kernel);
    m.impl("linalg_solve_ex", linalg_solve_ex_kernel);
    m.impl("linalg_lu_factor", linalg_lu_factor_kernel);
    m.impl("linalg_lu_factor_ex", linalg_lu_factor_ex_kernel);
    m.impl("linalg_lu", linalg_lu_kernel);
    m.impl("linalg_lu_solve", linalg_lu_solve_kernel);
    m.impl("linalg_solve_triangular", linalg_solve_triangular_kernel);
    m.impl("_linalg_eigh", linalg_eigh_internal_kernel);
    m.impl("_linalg_eigh.eigenvalues", linalg_eigh_internal_out_kernel);
    m.impl("linalg_eigh", linalg_eigh_kernel);
    m.impl("linalg_eigh.eigvals", linalg_eigh_eigvals_out_kernel);
    m.impl("linalg_eigvalsh", linalg_eigvalsh_kernel);
    m.impl("linalg_eigvalsh.out", linalg_eigvalsh_out_kernel);
    m.impl("linalg_eig", linalg_eig_kernel);
    m.impl("linalg_eig.out", linalg_eig_out_kernel);
    m.impl("linalg_eigvals", linalg_eigvals_kernel);
    m.impl("_linalg_eigvals", linalg_eigvals_kernel);
    m.impl("linalg_eigvals.out", linalg_eigvals_out_kernel);
    m.impl("_linalg_svd", linalg_svd_internal_kernel);
    m.impl("_linalg_svd.U", linalg_svd_internal_out_kernel);
    m.impl("linalg_svd", linalg_svd_kernel);
    m.impl("linalg_svd.U", linalg_svd_out_kernel);
    m.impl("linalg_svdvals", linalg_svdvals_kernel);
    m.impl("linalg_svdvals.out", linalg_svdvals_out_kernel);
    m.impl("linalg_lstsq", linalg_lstsq_kernel);
    m.impl("linalg_polar", linalg_polar_kernel);
    m.impl("linalg_polar.out", linalg_polar_out_kernel);
    m.impl("linalg_qr", linalg_qr_kernel);
    m.impl("linalg_qr.out", linalg_qr_out_kernel);
    m.impl("linalg_householder_product", linalg_householder_product_kernel);
    m.impl("linalg_householder_product.out", linalg_householder_product_out_kernel);
    m.impl("linalg_ldl_factor", linalg_ldl_factor_kernel);
    m.impl("linalg_ldl_factor_ex", linalg_ldl_factor_ex_kernel);
    m.impl("linalg_ldl_solve", ldl_solve_impl);
    m.impl("linalg_diagonal", linalg_diagonal_kernel);
}

}  // namespace cpu
}  // namespace tensorplay
