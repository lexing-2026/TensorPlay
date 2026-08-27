// CUDA linalg kernels — cusolverDn ports of the CPU paths, following
// third_party/pytorch/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp.
//
// Every routine loops over batch elements (as torch does for its non-batched
// cusolver paths), works on batched column-major buffers produced with
// clone_batched_column_major / empty_column_major, and reports LAPACK-style
// info codes through device int32 tensors.
//
// Not ported here: linalg_eig / linalg_eigvals raise torch's no-MAGMA error
// (torch's geev path needs MAGMA), and linalg_ldl_* stay CPU-only.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "LinearAlgebraNames.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#define CUSOLVER_CHECK(condition)                                              \
    do {                                                                       \
        const cusolverStatus_t _tp_cs = (condition);                           \
        if (_tp_cs != CUSOLVER_STATUS_SUCCESS) {                               \
            TP_THROW(RuntimeError, "cuSOLVER error ",                          \
                     std::to_string(static_cast<int>(_tp_cs)));                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(condition)                                                \
    do {                                                                       \
        const cublasStatus_t _tp_cb = (condition);                             \
        if (_tp_cb != CUBLAS_STATUS_SUCCESS) {                                 \
            TP_THROW(RuntimeError, "cuBLAS error ",                            \
                     std::to_string(static_cast<int>(_tp_cb)));                \
        }                                                                      \
    } while (0)

constexpr cublasFillMode_t fill_mode(char uplo) {
    return uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

// ------------------------------------------------------------- helpers -----

template <class Kernel>
decltype(auto) run_real(DType dt, Kernel&& k) {
    switch (dt) {
        case DType::Float32:
            return k(static_cast<float*>(nullptr));
        case DType::Float64:
            return k(static_cast<double*>(nullptr));
        default:
            TP_THROW(NotImplementedError,
                     "linalg: unsupported dtype ", pretty_dtype_name(dt),
                     " on CUDA (only float32/float64 are implemented)");
    }
}

void check_is_matrix(const Tensor& A, const char* fn, const char* arg = "A") {
    if (A.dim() < 2) {
        TP_THROW(RuntimeError, fn, ": The input tensor ", arg,
                 " must have at least 2 dimensions.");
    }
}

void square_check_inputs(const Tensor& A, const char* fn, const char* arg = "A") {
    check_is_matrix(A, fn, arg);
    if (A.size(-1) != A.size(-2)) {
        TP_THROW(RuntimeError, fn, ": ", arg,
                 " must be batches of square matrices, but they are ",
                 A.size(-2), " by ", A.size(-1), " matrices");
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

Tensor clone_batched_column_major(const Tensor& src) {
    // The clone must physically transpose into a contiguous buffer of the
    // transposed shape: clone() with Preserve would keep the transposed view's
    // strides (non-overlapping-and-dense), leaving the data row-major.
    auto result = src.transpose(-2, -1).clone(static_cast<int64_t>(MemoryFormat::Contiguous));
    return result.transpose(-2, -1);
}

Tensor empty_column_major(std::vector<int64_t> shape, DType dt, Device dev) {
    std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
    return Tensor::empty(shape, dt, dev).transpose(-2, -1);
}

std::vector<int64_t> cat_batch(const std::vector<int64_t>& batch,
                               std::vector<int64_t> tail) {
    std::vector<int64_t> out = batch;
    for (int64_t v : tail) out.push_back(v);
    return out;
}

int64_t linear_batch_size(const std::vector<int64_t>& batch) {
    return static_cast<int64_t>(std::accumulate(
        batch.begin(), batch.end(), int64_t{1}, std::multiplies<int64_t>()));
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
                     ") must match the size of tensor b (", db, ")");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

Tensor expand_to_batch(const Tensor& t, const std::vector<int64_t>& batch) {
    return t.expand(cat_batch(batch, {t.size(-2), t.size(-1)}));
}

int64_t first_nonzero_info(const Tensor& infos_host) {
    const auto* ptr = infos_host.data_ptr<int32_t>();
    for (int64_t i = 0; i < infos_host.numel(); ++i)
        if (ptr[i] != 0) return i;
    return -1;
}

// Port of at::native::_linalg_check_errors on a host copy of the info codes.
void check_infos(const Tensor& infos_dev, std::string_view api_name, bool is_matrix) {
    Tensor infos = infos_dev.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    const int64_t first = first_nonzero_info(infos);
    if (first < 0) return;
    const auto* ptr = infos.data_ptr<int32_t>();
    const int64_t info = ptr[first];
    const std::string b =
        is_matrix ? "" : ": (Batch element " + std::to_string(first) + ")";
    if (info < 0) {
        TP_THROW(RuntimeError, api_name, b, ": Argument ", -info,
                 " has illegal value.");
    }
    if (api_name.find("inv") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (api_name.find("lu_factor") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
                 "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
                 "linalg.lu_factor_ex(A, pivot)");
    } else if (api_name.find("cholesky") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (api_name.find("svd") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (api_name.find("eig") != std::string_view::npos ||
               api_name.find("syevd") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else {
        TP_THROW(RuntimeError, api_name, b,
                 ": The solver failed because the input matrix is singular.");
    }
}

// ------------------------------------------------- cusolver entry traits ---

template <typename scalar_t>
struct CusolverTraits;

template <>
struct CusolverTraits<float> {
    static constexpr bool is_float = true;
    using T = float;
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, char uplo, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSpotrf_bufferSize(h, fill_mode(uplo), n, a, lda, lw);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t h, char uplo, int n, float* a,
                                  int lda, float* work, int lwork, int* info) {
        return cusolverDnSpotrf(h, fill_mode(uplo), n, a, lda, work, lwork, info);
    }
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSgetrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, float* a,
                                  int lda, int* ipiv, float* work, int lwork,
                                  int* info) {
        return cusolverDnSgetrf(h, m, n, a, lda, work, ipiv, info); (void)lwork;
    }
    static cusolverStatus_t getrs(cusolverDnHandle_t h, cublasOperation_t trans,
                                  int n, int nrhs, const float* a, int lda,
                                  const int* ipiv, float* b, int ldb, int* info) {
        return cusolverDnSgetrs(h, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSgeqrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, float* a,
                                  int lda, float* tau, float* work, int lwork,
                                  int* info) {
        return cusolverDnSgeqrf(h, m, n, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             int k, const float* a, int lda,
                                             const float* tau, int* lw) {
        return cusolverDnSorgqr_bufferSize(h, m, n, k, a, lda, tau, lw);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k,
                                  float* a, int lda, const float* tau, float* work,
                                  int lwork, int* info) {
        return cusolverDnSorgqr(h, m, n, k, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                             cusolverEigMode_t jobz, char uplo,
                                             int n, const float* a, int lda,
                                             const float* w, int* lw) {
        return cusolverDnSsyevd_bufferSize(h, jobz, fill_mode(uplo), n, a, lda, w, lw);
    }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz,
                                  char uplo, int n, float* a, int lda, float* w,
                                  float* work, int lwork, int* info) {
        return cusolverDnSsyevd(h, jobz, fill_mode(uplo), n, a, lda, w, work,
                                lwork, info);
    }
    static cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char jobu,
                                  signed char jobvt, int m, int n, float* a,
                                  int lda, float* s, float* u, int ldu, float* vt,
                                  int ldvt, float* work, int lwork, float* rwork,
                                  int* info) {
        return cusolverDnSgesvd(h, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                work, lwork, rwork, info);
    }
};

template <>
struct CusolverTraits<double> {
    static constexpr bool is_float = false;
    using T = double;
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, char uplo, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDpotrf_bufferSize(h, fill_mode(uplo), n, a, lda, lw);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t h, char uplo, int n, double* a,
                                  int lda, double* work, int lwork, int* info) {
        return cusolverDnDpotrf(h, fill_mode(uplo), n, a, lda, work, lwork, info);
    }
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDgetrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, double* a,
                                  int lda, int* ipiv, double* work, int lwork,
                                  int* info) {
        return cusolverDnDgetrf(h, m, n, a, lda, work, ipiv, info); (void)lwork;
    }
    static cusolverStatus_t getrs(cusolverDnHandle_t h, cublasOperation_t trans,
                                  int n, int nrhs, const double* a, int lda,
                                  const int* ipiv, double* b, int ldb, int* info) {
        return cusolverDnDgetrs(h, trans, n, nrhs, a, lda, ipiv, b, ldb, info);
    }
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDgeqrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, double* a,
                                  int lda, double* tau, double* work, int lwork,
                                  int* info) {
        return cusolverDnDgeqrf(h, m, n, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             int k, const double* a, int lda,
                                             const double* tau, int* lw) {
        return cusolverDnDorgqr_bufferSize(h, m, n, k, a, lda, tau, lw);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k,
                                  double* a, int lda, const double* tau,
                                  double* work, int lwork, int* info) {
        return cusolverDnDorgqr(h, m, n, k, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                             cusolverEigMode_t jobz, char uplo,
                                             int n, const double* a, int lda,
                                             const double* w, int* lw) {
        return cusolverDnDsyevd_bufferSize(h, jobz, fill_mode(uplo), n, a, lda, w, lw);
    }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz,
                                  char uplo, int n, double* a, int lda, double* w,
                                  double* work, int lwork, int* info) {
        return cusolverDnDsyevd(h, jobz, fill_mode(uplo), n, a, lda, w, work,
                                lwork, info);
    }
    static cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char jobu,
                                  signed char jobvt, int m, int n, double* a,
                                  int lda, double* s, double* u, int ldu,
                                  double* vt, int ldvt, double* work, int lwork,
                                  double* rwork, int* info) {
        return cusolverDnDgesvd(h, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                work, lwork, rwork, info);
    }
};

// ------------------------------------------------------------- getrf -------

template <typename scalar_t>
void apply_getrf(const Tensor& input, const Tensor& pivots, const Tensor& info) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = input.data_ptr<scalar_t>();
    auto* ipiv = pivots.data_ptr<int32_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t ms = matrix_stride_of(input);
    const int64_t piv_stride = pivots.dim() > 1 ? pivots.size(-1) : pivots.numel();
    const int64_t batch = batch_count_of(input);
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();

    int lwork = 0;
    CUSOLVER_CHECK(Tr::getrf_bufferSize(handle, m, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::getrf(handle, m, n, &a[i * ms], lda, &ipiv[i * piv_stride],
                                 work_ptr, lwork, &info_data[i]));
    }
}

std::tuple<Tensor, Tensor, Tensor> lu_factor_ex_cuda_impl(const Tensor& A,
                                                          bool check_errors) {
    square_check_inputs(A, "linalg.lu_factor_ex");
    Tensor LU = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    Tensor pivots = Tensor::zeros(cat_batch(batch, {k}), DType::Int32, A.device());
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_getrf<T>(LU, pivots, info);
    });
    if (check_errors) check_infos(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
    return {LU.contiguous(), pivots.contiguous(), info.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_lu_factor_kernel_cuda(const Tensor& A, bool pivot) {
    (void)pivot;  // pivoting always on (torch CPU/CUDA parity).
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(A, false);
    check_infos(info, "torch.linalg.lu_factor", A.dim() == 2);
    return {LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor> linalg_lu_factor_ex_kernel_cuda(const Tensor& A,
                                                                   bool pivot,
                                                                   bool check_errors) {
    (void)pivot;
    return lu_factor_ex_cuda_impl(A, check_errors);
}

template <typename scalar_t>
void host_det_slogdet(const Tensor& LU_h, const Tensor& piv_h, Tensor& det_out,
                      Tensor* sign_out, Tensor* logabsdet_out) {
    const int64_t bs = batch_count_of(LU_h);
    const int64_t n = LU_h.size(-1);
    const scalar_t* lu = LU_h.data_ptr<scalar_t>();
    const int32_t* pv = piv_h.data_ptr<int32_t>();
    std::vector<scalar_t> dets(bs), signs(bs), logs(bs);
    const bool want_log = sign_out != nullptr;
    for (int64_t b = 0; b < bs; ++b) {
        scalar_t det = scalar_t(1), sgn = scalar_t(1), logdet = scalar_t(0);
        bool singular = false;
        for (int64_t i = 0; i < n; ++i) {
            const scalar_t v = lu[b * n * n + i * n + i];
            if (v == scalar_t(0)) { singular = true; break; }
            det *= v;
            logdet += std::log(std::abs(v));
            sgn *= v < scalar_t(0) ? scalar_t(-1) : scalar_t(1);
        }
        int64_t parity = 0;
        for (int64_t i = 0; i < n; ++i)
            if (pv[b * n + i] - 1 != static_cast<int32_t>(i)) ++parity;
        const scalar_t perm_sign = (parity % 2 == 0) ? scalar_t(1) : scalar_t(-1);
        if (want_log) {
            signs[b] = singular ? scalar_t(0) : sgn * perm_sign;
            logs[b] = singular ? -std::numeric_limits<scalar_t>::infinity() : logdet;
        } else {
            dets[b] = singular ? scalar_t(0) : det * perm_sign;
        }
    }
    cudaMemcpyAsync(det_out.data_ptr(), want_log ? signs.data() : dets.data(),
                    sizeof(scalar_t) * bs, cudaMemcpyHostToDevice,
                    getCurrentCUDAStream().stream());
    if (want_log) {
        cudaMemcpyAsync(logabsdet_out->data_ptr(), logs.data(),
                        sizeof(scalar_t) * bs, cudaMemcpyHostToDevice,
                        getCurrentCUDAStream().stream());
    }
}

Tensor linalg_det_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.det");
    // det(A^T) = det(A): reuse the contiguous layout like torch.
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(src, false);
    (void)info;
    Tensor LU_h = LU.to(Device(DeviceType::CPU), A.dtype()).contiguous();
    Tensor piv_h = pivots.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    Tensor out = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        host_det_slogdet<T>(LU_h, piv_h, out, nullptr, nullptr);
    });
    return out;
}

std::tuple<Tensor, Tensor> linalg_slogdet_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.slogdet");
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(src, false);
    (void)info;
    Tensor LU_h = LU.to(Device(DeviceType::CPU), A.dtype()).contiguous();
    Tensor piv_h = pivots.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    auto sign = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    auto logabsdet = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        Tensor dummy;  // unused in slogdet mode
        host_det_slogdet<T>(LU_h, piv_h, dummy, &sign, &logabsdet);
    });
    return {sign, logabsdet};
}

// ------------------------------------------------------- getrs-based solve --

// Solves op(A) X = B in place on the column-major buffer `B_cm`.
template <typename scalar_t>
void apply_getrs(const Tensor& LU_cm, const Tensor& pivots, Tensor& B_cm,
                 cublasOperation_t trans) {
    using Tr = CusolverTraits<scalar_t>;
    const int64_t n = LU_cm.size(-2);
    const int64_t nrhs = B_cm.size(-1);
    const int64_t bs = linear_batch_size(batch_shape_of(B_cm));
    auto* lu = LU_cm.data_ptr<scalar_t>();
    auto* b = B_cm.data_ptr<scalar_t>();
    const auto* piv = pivots.data_ptr<int32_t>();
    const int64_t lu_ms = matrix_stride_of(LU_cm);
    const int64_t b_ms = matrix_stride_of(B_cm);
    const int64_t piv_stride = n;
    Tensor dev_info = Tensor::zeros({bs}, DType::Int32, B_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    const auto handle = CUDAContext::getCusolverDnHandle();
    for (int64_t i = 0; i < bs; ++i) {
        CUSOLVER_CHECK(Tr::getrs(handle, trans, static_cast<int>(n),
                                 static_cast<int>(nrhs), &lu[i * lu_ms],
                                 static_cast<int>(n), &piv[i * piv_stride],
                                 &b[i * b_ms], static_cast<int>(B_cm.size(-2)),
                                 &info_data[i]));
    }
}

std::tuple<Tensor, Tensor> linalg_solve_ex_kernel_cuda(const Tensor& A,
                                                       const Tensor& B,
                                                       bool left, bool check_errors) {
    const char* api = "linalg.solve";
    check_is_matrix(A, api, "A");
    check_is_matrix(B, api, "B");
    if (!(left ? A.size(-2) == B.size(-2) : A.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", A.size(-2), "x", A.size(-1), " and ",
                 B.size(-2), "x", B.size(-1), ")");
    }
    const auto batch = broadcast_batch(A, B);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(A, batch));
    const int64_t n = A.size(-1);
    Tensor pivots = Tensor::zeros(cat_batch(batch_shape_of(LU_work), {n}),
                                  DType::Int32, A.device());
    Tensor info = Tensor::zeros(batch_shape_of(LU_work), DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_getrf<T>(LU_work, pivots, info);
    });
    check_infos(info, api, false);

    Tensor result;
    if (left) {
        // op(A) = A or conj-transposed A; real dtypes: adjoint handled via 'T'.
        // torch.linalg.solve has no adjoint flag; plain AX = B.
        Tensor B_cm = clone_batched_column_major(expand_to_batch(B, batch));
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_getrs<T>(LU_work, pivots, B_cm, CUBLAS_OP_N);
        });
        result = B_cm.contiguous();
    } else {
        // X A = B <=> A^T X^T = B^T: solve against the transposed RHS.
        Tensor BT_cm =
            clone_batched_column_major(expand_to_batch(B, batch).transpose(-2, -1));
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_getrs<T>(LU_work, pivots, BT_cm, CUBLAS_OP_T);
        });
        result = BT_cm.contiguous().transpose(-2, -1).contiguous();
    }
    if (check_errors) {
        // getrs reports only argument errors; singularity surfaced by getrf.
    }
    return {result, info};
}

Tensor linalg_solve_kernel_cuda(const Tensor& A, const Tensor& B, bool left) {
    return std::get<0>(linalg_solve_ex_kernel_cuda(A, B, left, false));
}

std::tuple<Tensor, Tensor> linalg_inv_ex_kernel_cuda(const Tensor& A,
                                                     bool check_errors) {
    square_check_inputs(A, "linalg.inv_ex");
    // torch composes inv through solve_ex against the identity.
    const int64_t n = A.size(-1);
    const int64_t ms = n * n;
    const int64_t bs = linear_batch_size(batch_shape_of(A));
    Tensor identity = Tensor::empty(A.shape(), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        std::vector<T> eye_host(static_cast<size_t>(ms), T(0));
        for (int64_t i = 0; i < n; ++i)
            eye_host[static_cast<size_t>(i * n + i)] = T(1);
        for (int64_t b = 1; b < bs; ++b)
            std::copy(eye_host.begin(), eye_host.end(),
                      eye_host.begin() + b * ms);
        cudaMemcpyAsync(identity.data_ptr(), eye_host.data(), sizeof(T) * ms * bs,
                        cudaMemcpyHostToDevice, getCurrentCUDAStream().stream());
    });
    auto [inv, info] = linalg_solve_ex_kernel_cuda(A, identity, true, false);
    if (check_errors) check_infos(info, "linalg.inv_ex", A.dim() == 2);
    return {inv, info};
}

Tensor linalg_inv_kernel_cuda(const Tensor& A) {
    auto [inv, info] = linalg_inv_ex_kernel_cuda(A, false);
    check_infos(info, "linalg.inv", A.dim() == 2);
    return inv;
}

// ------------------------------------------------------------- potrf -------

template <typename scalar_t>
void apply_potrf(const Tensor& L, const Tensor& info, char uplo) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = L.data_ptr<scalar_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t ms = matrix_stride_of(L);
    const int64_t batch = batch_count_of(L);
    const int n = static_cast<int>(L.size(-1));
    const int lda = std::max(1, n);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::potrf_bufferSize(handle, uplo, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, L.dtype(), L.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::potrf(handle, uplo, n, &a[i * ms], lda, work_ptr,
                                 lwork, &info_data[i]));
    }
}

std::tuple<Tensor, Tensor> linalg_cholesky_ex_kernel_cuda(const Tensor& A, bool upper,
                                                          bool check_errors) {
    square_check_inputs(A, "linalg.cholesky_ex");
    Tensor L = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_potrf<T>(L, info, upper ? 'U' : 'L');
    });
    if (check_errors) check_infos(info, "linalg.cholesky_ex", A.dim() == 2);
    return {L.contiguous(), info.contiguous()};
}

Tensor linalg_cholesky_kernel_cuda(const Tensor& A, bool upper) {
    auto [L, info] = linalg_cholesky_ex_kernel_cuda(A, upper, false);
    check_infos(info, "linalg.cholesky", A.dim() == 2);
    return L;
}

// -------------------------------------------------------------- lu_solve ---

Tensor linalg_lu_solve_kernel_cuda(const Tensor& LU, const Tensor& pivots,
                                   const Tensor& B, bool left, bool adjoint) {
    const char* api = "linalg.lu_solve";
    square_check_inputs(LU, api, "LU");
    check_is_matrix(B, api, "B");
    if (!(left ? LU.size(-2) == B.size(-2) : LU.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of LU and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", LU.size(-2), "x", LU.size(-1), " and ",
                 B.size(-2), "x", B.size(-1), ")");
    }
    const auto batch = broadcast_batch(LU, B);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(LU, batch));
    std::vector<int64_t> piv_shape = cat_batch(batch, {pivots.size(-1)});
    Tensor piv_exp = pivots.expand(piv_shape).contiguous();
    if (left) {
        Tensor B_cm = clone_batched_column_major(expand_to_batch(B, batch));
        run_real(LU.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_getrs<T>(LU_work, piv_exp, B_cm,
                           adjoint ? CUBLAS_OP_T : CUBLAS_OP_N);
        });
        return B_cm.contiguous();
    }
    Tensor BT_cm =
        clone_batched_column_major(expand_to_batch(B, batch).transpose(-2, -1));
    run_real(LU.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_getrs<T>(LU_work, piv_exp, BT_cm,
                       adjoint ? CUBLAS_OP_N : CUBLAS_OP_T);
    });
    return BT_cm.contiguous().transpose(-2, -1).contiguous();
}

// --------------------------------------------------------- geqrf / orgqr ---

template <typename scalar_t>
void apply_geqrf(const Tensor& QR_cm, const Tensor& tau) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = QR_cm.data_ptr<scalar_t>();
    auto* tau_ptr = tau.data_ptr<scalar_t>();
    const int64_t ms = matrix_stride_of(QR_cm);
    const int64_t tau_stride = tau.dim() > 1 ? tau.size(-1) : tau.numel();
    const int64_t batch = batch_count_of(QR_cm);
    const int m = static_cast<int>(QR_cm.size(-2));
    const int n = static_cast<int>(QR_cm.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::geqrf_bufferSize(handle, m, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, QR_cm.dtype(),
                                QR_cm.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    Tensor dev_info = Tensor::zeros({batch}, DType::Int32, QR_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::geqrf(handle, m, n, &a[i * ms], lda,
                                 &tau_ptr[i * tau_stride], work_ptr, lwork,
                                 &info_data[i]));
    }
}

template <typename scalar_t>
void apply_orgqr(Tensor& Q_cm, const Tensor& tau) {
    using Tr = CusolverTraits<scalar_t>;
    if (Q_cm.numel() == 0) return;
    auto* a = Q_cm.data_ptr<scalar_t>();
    const auto* tau_ptr = tau.data_ptr<scalar_t>();
    const int64_t ms = matrix_stride_of(Q_cm);
    const int64_t tau_stride = tau.dim() > 1 ? tau.size(-1) : tau.numel();
    const int64_t batch = batch_count_of(Q_cm);
    const int m = static_cast<int>(Q_cm.size(-2));
    const int n = static_cast<int>(Q_cm.size(-1));
    const int k = static_cast<int>(tau.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::orgqr_bufferSize(handle, m, n, k, a, lda, tau_ptr, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, Q_cm.dtype(),
                                Q_cm.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    Tensor dev_info = Tensor::zeros({batch}, DType::Int32, Q_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::orgqr(handle, m, n, k, &a[i * ms], lda,
                                 &tau_ptr[i * tau_stride], work_ptr, lwork,
                                 &info_data[i]));
    }
}

template <typename scalar_t>
__global__ void triu_extract_kernel(const scalar_t* src, scalar_t* dst,
                                    int64_t ld_src, int64_t rows, int64_t cols,
                                    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t col = idx % cols;
    const int64_t row = (idx / cols) % rows;
    const int64_t b = idx / (rows * cols);
    dst[idx] = col >= row ? src[b * ld_src * cols + col * ld_src + row]
                          : scalar_t(0);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> linalg_qr_kernel_cuda_impl(const Tensor& A,
                                                      const std::string& mode) {
    Tensor QR = clone_batched_column_major(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const bool reduced = mode != "complete";
    Tensor tau = Tensor::zeros(cat_batch(batch, {k}), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geqrf<T>(QR, tau);
    });

    const int64_t qcols = reduced ? k : m;
    const int64_t rrows = reduced ? k : m;
    const int64_t bs = linear_batch_size(batch);

    // Pack the first qcols reflector columns: column segments are contiguous
    // in both layouts, so one strided 2D copy suffices.
    Tensor Q_in = empty_column_major(cat_batch(batch, {m, qcols}),
                                     A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        cudaMemcpy2DAsync(
            Q_in.data_ptr<T>(), sizeof(T) * qcols, QR.data_ptr<T>(), sizeof(T) * n,
            sizeof(T) * m, qcols, cudaMemcpyDeviceToDevice,
            getCurrentCUDAStream().stream());
        apply_orgqr<T>(Q_in, tau);
    });

    // R: upper triangle of the geqrf buffer.
    Tensor R = empty_column_major(cat_batch(batch, {rrows, n}), A.dtype(),
                                  A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const int64_t total = bs * rrows * n;
        if (total > 0) {
            triu_extract_kernel<T><<<static_cast<unsigned>((total + 255) / 256),
                                     256, 0,
                                     getCurrentCUDAStream().stream()>>>(
                QR.data_ptr<T>(), R.data_ptr<T>(), m, rrows, n, total);
        }
    });
    return {Q_in.contiguous(), R.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_qr_kernel_cuda(const Tensor& A,
                                                 std::string mode) {
    check_is_matrix(A, "linalg.qr");
    if (mode != "reduced" && mode != "complete" && mode != "r" && mode != "R") {
        TP_THROW(RuntimeError, "linalg.qr: mode '", mode, "' not recognized.");
    }
    return run_real(A.dtype(), [&](auto tag) -> std::tuple<Tensor, Tensor> {
        using T = std::remove_pointer_t<decltype(tag)>;
        return linalg_qr_kernel_cuda_impl<T>(A, mode);
    });
}

Tensor linalg_householder_product_kernel_cuda(const Tensor& input, const Tensor& tau) {
    const char* api = "linalg.householder_product";
    check_is_matrix(input, api);
    if (input.size(-2) < input.size(-1)) {
        TP_THROW(RuntimeError, api, ": If input has size (..., m, n), "
                                    "n must be less than or equal to m, but got n = ",
                 input.size(-1), " and m = ", input.size(-2));
    }
    if (tau.dim() < 1 || tau.size(-1) != std::min(input.size(-2), input.size(-1))) {
        TP_THROW(RuntimeError, api,
                 ": If tau has size (..., k), then when input has size (..., m, n) "
                 "we require k == min(m, n)");
    }
    if (tau.dtype() != input.dtype()) {
        TP_THROW(RuntimeError, api, ": input and tau must have the same dtype");
    }
    Tensor result = clone_batched_column_major(input);
    run_real(input.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_orgqr<T>(result, tau);
    });
    return result.contiguous();
}

// --------------------------------------------------------------- syevd -----

std::tuple<Tensor, Tensor> eigh_impl_cuda(const Tensor& A, bool upper,
                                          bool compute_eigenvectors) {
    square_check_inputs(A, "linalg.eigh");
    Tensor vectors = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    Tensor values = Tensor::empty(cat_batch(batch, {n}), A.dtype(), A.device());
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using Tr = CusolverTraits<std::remove_pointer_t<decltype(tag)>>;
        using T = typename Tr::T;
        auto* v = vectors.data_ptr<T>();
        auto* w = values.data_ptr<T>();
        auto* info_data = info.data_ptr<int32_t>();
        const int64_t ms = matrix_stride_of(vectors);
        const int64_t batch_n = batch_count_of(vectors);
        const int nn = static_cast<int>(n);
        const int lda = std::max(1, nn);
        const char uplo = upper ? 'U' : 'L';
        const cusolverEigMode_t jobz =
            compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        const auto handle = CUDAContext::getCusolverDnHandle();
        int lwork = 0;
        CUSOLVER_CHECK(Tr::syevd_bufferSize(handle, jobz, uplo, nn, v, lda, w, &lwork));
        Tensor work = Tensor::empty({std::max(lwork, 1)}, A.dtype(), A.device());
        T* work_ptr =
            Tr::is_float
                ? reinterpret_cast<T*>(static_cast<void*>(work.data_ptr<float>()))
                : reinterpret_cast<T*>(static_cast<void*>(work.data_ptr<double>()));
        for (int64_t i = 0; i < batch_n; ++i) {
            CUSOLVER_CHECK(Tr::syevd(handle, jobz, uplo, nn, &v[i * ms], lda,
                                     &w[i * n], work_ptr, lwork, &info_data[i]));
            // torch returns early on the first failure.
            if (info_data[i] != 0) break;
        }
    });
    check_infos(info, "linalg.eigh", A.dim() == 2);
    return {values.contiguous(), vectors.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_eigh_kernel_cuda(const Tensor& A, std::string UPLO) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return eigh_impl_cuda(A, UPLO == "U", true);
}

Tensor linalg_eigvalsh_kernel_cuda(const Tensor& A, std::string UPLO) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigvalsh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return std::get<0>(eigh_impl_cuda(A, UPLO == "U", false));
}

// --------------------------------------------------------------- gesvd -----

std::tuple<Tensor, Tensor, Tensor> svd_impl_cuda(const Tensor& A, bool full_matrices,
                                                 bool compute_uv) {
    check_is_matrix(A, "linalg.svd");
    Tensor a_copy = clone_batched_column_major(A);  // gesvd destroys its input
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const int64_t bs = linear_batch_size(batch);
    Tensor U, S, Vh;
    if (compute_uv) {
        U = empty_column_major(cat_batch(batch, {m, full_matrices ? m : k}),
                               A.dtype(), A.device());
        Vh = empty_column_major(cat_batch(batch, {full_matrices ? n : k, n}),
                                A.dtype(), A.device());
    } else {
        U = Tensor::empty(cat_batch(batch, {m, 0}), A.dtype(), A.device());
        Vh = Tensor::empty(cat_batch(batch, {0, n}), A.dtype(), A.device());
    }
    S = Tensor::empty(cat_batch(batch, {k}), A.dtype(), A.device());
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using Tr = CusolverTraits<std::remove_pointer_t<decltype(tag)>>;
        using T = typename Tr::T;
        auto* a = a_copy.data_ptr<T>();
        auto* s = S.data_ptr<T>();
        auto* u = compute_uv ? U.data_ptr<T>() : nullptr;
        auto* vt = compute_uv ? Vh.data_ptr<T>() : nullptr;
        auto* info_data = info.data_ptr<int32_t>();
        const signed char jobz = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
        const int mm = static_cast<int>(m);
        const int nn = static_cast<int>(n);
        const int lda = std::max(1, mm);
        const int ldu = compute_uv ? static_cast<int>(U.stride(-1)) : 1;
        const int ldvt = compute_uv ? static_cast<int>(Vh.stride(-1)) : 1;
        const auto handle = CUDAContext::getCusolverDnHandle();
        int lwork = 0;
        CUSOLVER_CHECK(Tr::gesvd(handle, jobz, jobz, mm, nn, a, lda, nullptr,
                                 nullptr, ldu, nullptr, ldvt, nullptr, -1, nullptr,
                                 &lwork));
        Tensor work = Tensor::empty({std::max(lwork, 1)}, A.dtype(), A.device());
        Tensor rwork = Tensor::empty({std::max<int64_t>(5 * k, 1)},
                                     A.dtype(), A.device());
        T* work_ptr =
            Tr::is_float
                ? reinterpret_cast<T*>(static_cast<void*>(work.data_ptr<float>()))
                : reinterpret_cast<T*>(static_cast<void*>(work.data_ptr<double>()));
        T* rwork_ptr =
            Tr::is_float
                ? reinterpret_cast<T*>(static_cast<void*>(rwork.data_ptr<float>()))
                : reinterpret_cast<T*>(static_cast<void*>(rwork.data_ptr<double>()));
        for (int64_t i = 0; i < bs; ++i) {
            CUSOLVER_CHECK(Tr::gesvd(handle, jobz, jobz, mm, nn, &a[i * m * n],
                                     lda, &s[i * k],
                                     compute_uv ? &u[i * matrix_stride_of(U)] : nullptr,
                                     ldu,
                                     compute_uv ? &vt[i * matrix_stride_of(Vh)] : nullptr,
                                     ldvt, work_ptr, lwork, rwork_ptr,
                                     &info_data[i]));
        }
    });
    check_infos(info, "linalg.svd", A.dim() == 2);
    return {U.contiguous(), S.contiguous(), Vh.contiguous()};
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_kernel_cuda(
        const Tensor& A, bool full_matrices, std::optional<std::string> driver) {
    if (driver.has_value() && driver.value() != "gesvd" && driver.value() != "gesvdj") {
        TP_THROW(RuntimeError, "linalg.svd(): driver ", driver.value(),
                 " is not supported on CUDA");
    }
    return svd_impl_cuda(A, full_matrices, true);
}

Tensor linalg_svdvals_kernel_cuda(const Tensor& A, std::optional<std::string> driver) {
    (void)driver;
    return std::get<1>(svd_impl_cuda(A, false, false));
}

// ---------------------------------------------------- triangular solve -----

Tensor linalg_solve_triangular_kernel_cuda(const Tensor& A, const Tensor& B,
                                           bool upper, bool left, bool unitriangular) {
    const char* api = "linalg.solve_triangular";
    check_is_matrix(A, api, "A");
    check_is_matrix(B, api, "B");
    const auto batch = broadcast_batch(A, B);
    Tensor B_cm = clone_batched_column_major(expand_to_batch(B, batch));
    Tensor A_cm = clone_batched_column_major(expand_to_batch(A, batch));
    const int64_t bs = linear_batch_size(batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* a = A_cm.data_ptr<T>();
        auto* b = B_cm.data_ptr<T>();
        const cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
        const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        const cublasDiagType_t diag =
            unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
        // This allows rectangular A when left = True (torch parity).
        const int m = static_cast<int>(B.size(-2));
        const int n = static_cast<int>(B.size(-1));
        const int lda = std::max<int>(1, static_cast<int>(A.size(-2)));
        const int ldb = std::max(1, m);
        const T alpha = T(1);
        const int64_t a_ms = matrix_stride_of(A_cm);
        const int64_t b_ms = matrix_stride_of(B_cm);
        const auto handle = CUDAContext::getCublasHandle();
        for (int64_t i = 0; i < bs; ++i) {
            if constexpr (std::is_same_v<T, float>) {
                CUBLAS_CHECK(cublasStrsm(handle, side, uplo, CUBLAS_OP_N, diag,
                                         m, n, &alpha, &a[i * a_ms], lda,
                                         &b[i * b_ms], ldb));
            } else {
                CUBLAS_CHECK(cublasDtrsm(handle, side, uplo, CUBLAS_OP_N, diag,
                                         m, n, &alpha, &a[i * a_ms], lda,
                                         &b[i * b_ms], ldb));
            }
        }
    });
    return B_cm.contiguous();
}

// ------------------------------------------------------------------ lstsq --

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq_kernel_cuda(
        const Tensor& A, const Tensor& B, std::optional<double> rcond,
        std::optional<std::string> driver_opt) {
    const char* api = "linalg.lstsq";
    const std::string driver = driver_opt.value_or("gels");
    if (driver != "gels") {
        TP_THROW(NotImplementedError, api, ": driver '", driver,
                 "' is not supported on CUDA; only 'gels' is implemented");
    }
    check_is_matrix(A, api);
    check_is_matrix(B, api);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    if (m < n) {
        TP_THROW(RuntimeError, api,
                 ": The input tensor A should have at least as many rows as columns.");
    }
    (void)rcond;  // ignored by the gels driver (torch parity)

    const auto batch = broadcast_batch(A, B);
    const int64_t nrhs = B.size(-1);
    const int64_t ldb = std::max(m, n);
    const int64_t bs = linear_batch_size(batch);

    Tensor A_cm = clone_batched_column_major(expand_to_batch(A, batch));
    Tensor B_cm = empty_column_major(cat_batch(batch, {ldb, nrhs}),
                                     B.dtype(), B.device());
    run_real(B.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        // Zero-fill the padded buffer, then copy B into its top m rows.
        cudaMemsetAsync(B_cm.data_ptr(), 0, sizeof(T) * B_cm.numel(),
                        getCurrentCUDAStream().stream());
        Tensor b_exp = expand_to_batch(B, batch).contiguous();
        cudaMemcpy2DAsync(
            B_cm.data_ptr<T>(), sizeof(T) * ldb, b_exp.data_ptr<T>(),
            sizeof(T) * nrhs, sizeof(T) * m, nrhs, cudaMemcpyDeviceToDevice,
            getCurrentCUDAStream().stream());
        Tensor tau = Tensor::zeros(cat_batch(batch, {n}), A.dtype(), A.device());
        apply_geqrf<T>(A_cm, tau);
        // Apply Q^T to the padded RHS from the left.
        {
            using Tr = CusolverTraits<T>;
            const auto handle = CUDAContext::getCusolverDnHandle();
            int lwork = 0;
            CUSOLVER_CHECK(
                Tr::orgqr_bufferSize(handle, static_cast<int>(ldb),
                                     static_cast<int>(nrhs), static_cast<int>(n),
                                     A_cm.data_ptr<T>(), static_cast<int>(m),
                                     tau.data_ptr<T>(), &lwork));
            Tensor work = Tensor::empty({std::max(lwork, 1)}, A.dtype(), A.device());
            Tensor dev_info = Tensor::zeros({bs}, DType::Int32, A.device());
            for (int64_t i = 0; i < bs; ++i) {
                if constexpr (std::is_same_v<T, float>) {
                    CUSOLVER_CHECK(cusolverDnSormqr(
                        handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                        static_cast<int>(ldb), static_cast<int>(nrhs),
                        static_cast<int>(n), A_cm.data_ptr<float>() + i * m * n,
                        static_cast<int>(m), tau.data_ptr<float>() + i * n,
                        B_cm.data_ptr<float>() + i * ldb * nrhs,
                        static_cast<int>(ldb), work.data_ptr<float>(), lwork,
                        dev_info.data_ptr<int32_t>() + i));
                } else {
                    CUSOLVER_CHECK(cusolverDnDormqr(
                        handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                        static_cast<int>(ldb), static_cast<int>(nrhs),
                        static_cast<int>(n), A_cm.data_ptr<double>() + i * m * n,
                        static_cast<int>(m), tau.data_ptr<double>() + i * n,
                        B_cm.data_ptr<double>() + i * ldb * nrhs,
                        static_cast<int>(ldb), work.data_ptr<double>(), lwork,
                        dev_info.data_ptr<int32_t>() + i));
                }
            }
        }
        // Solve R X = Y with the leading n x n upper triangle.
        {
            const T alpha = T(1);
            const auto handle = CUDAContext::getCublasHandle();
            for (int64_t i = 0; i < bs; ++i) {
                if constexpr (std::is_same_v<T, float>) {
                    CUBLAS_CHECK(cublasStrsm(
                        handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, static_cast<int>(n),
                        static_cast<int>(nrhs), &alpha,
                        A_cm.data_ptr<float>() + i * m * n, static_cast<int>(m),
                        B_cm.data_ptr<float>() + i * ldb * nrhs,
                        static_cast<int>(ldb)));
                } else {
                    CUBLAS_CHECK(cublasDtrsm(
                        handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, static_cast<int>(n),
                        static_cast<int>(nrhs), &alpha,
                        A_cm.data_ptr<double>() + i * m * n, static_cast<int>(m),
                        B_cm.data_ptr<double>() + i * ldb * nrhs,
                        static_cast<int>(ldb)));
                }
            }
        }
    });
    Tensor solution = B_cm.contiguous().slice(-2, 0, n).contiguous();
    Tensor residuals;
    if (m > n) {
        residuals = Tensor::empty(cat_batch(batch, {nrhs}), B.dtype(), B.device());
        Tensor B_h = B_cm.contiguous().to(Device(DeviceType::CPU), B.dtype());
        run_real(B.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            const T* b = B_h.data_ptr<T>();
            std::vector<T> res(static_cast<size_t>(bs * nrhs), T(0));
            for (int64_t i = 0; i < bs; ++i)
                for (int64_t c = 0; c < nrhs; ++c)
                    for (int64_t r_ = n; r_ < ldb; ++r_) {
                        const T v = b[i * ldb * nrhs + c * ldb + r_];
                        res[i * nrhs + c] += v * v;
                    }
            Tensor staged = Tensor::tensor(res);
            cudaMemcpyAsync(residuals.data_ptr(), staged.data_ptr(),
                            sizeof(T) * bs * nrhs, cudaMemcpyHostToDevice,
                            getCurrentCUDAStream().stream());
        });
    } else {
        residuals = Tensor::empty(cat_batch(batch, {0}), B.dtype(), B.device());
    }
    Tensor rank = Tensor::full(batch, Scalar(static_cast<int64_t>(n)),
                               DType::Int64, B.device());
    return {solution, residuals, rank, solution};
}

// -------------------------------------------------------- lu with unpack ---

std::tuple<Tensor, Tensor, Tensor> linalg_lu_kernel_cuda(const Tensor& A, bool pivot) {
    (void)pivot;
    square_check_inputs(A, "linalg.lu");
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(A, false);
    (void)info;
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t kk = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const int64_t bs = linear_batch_size(batch);
    Tensor P = Tensor::zeros(cat_batch(batch, {m, m}), A.dtype(), A.device());
    Tensor L = Tensor::zeros(cat_batch(batch, {m, kk}), A.dtype(), A.device());
    Tensor U = Tensor::zeros(cat_batch(batch, {kk, n}), A.dtype(), A.device());
    Tensor LU_h = LU.to(Device(DeviceType::CPU), A.dtype()).contiguous();
    Tensor piv_h = pivots.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T* lu_all = LU_h.data_ptr<T>();
        const int32_t* piv = piv_h.data_ptr<int32_t>();
        std::vector<T> p_host(static_cast<size_t>(bs * m * m));
        std::vector<T> l_host(static_cast<size_t>(bs * m * kk));
        std::vector<T> u_host(static_cast<size_t>(bs * kk * n));
        for (int64_t b = 0; b < bs; ++b) {
            const T* lu = &lu_all[b * m * n];
            // LU is column-major (lda=m): element (i, j) at lu[j*m + i].
            for (int64_t col = 0; col < kk; ++col) {
                for (int64_t row = 0; row < m; ++row)
                    l_host[(b * m + row) * kk + col] =
                        row < col ? T(0)
                                  : (row == col ? T(1) : lu[col * m + row]);
            }
            for (int64_t col = 0; col < n; ++col)
                for (int64_t row = 0; row < kk; ++row)
                    u_host[(b * kk + row) * n + col] =
                        row <= col ? lu[col * m + row] : T(0);
            std::vector<int64_t> perm(static_cast<size_t>(m));
            for (int64_t i = 0; i < m; ++i) perm[i] = i;
            for (int64_t i = 0; i < kk; ++i) {
                const int64_t p_ = piv[b * kk + i] - 1;
                if (p_ != i) std::swap(perm[i], perm[p_]);
            }
            for (int64_t j = 0; j < m; ++j)
                p_host[(b * m + j) * m + perm[j]] = T(1);
        }
        Tensor p_stage = Tensor::tensor(p_host);
        Tensor l_stage = Tensor::tensor(l_host);
        Tensor u_stage = Tensor::tensor(u_host);
        cudaMemcpyAsync(P.data_ptr(), p_stage.data_ptr(),
                        sizeof(T) * bs * m * m, cudaMemcpyHostToDevice,
                        getCurrentCUDAStream().stream());
        cudaMemcpyAsync(L.data_ptr(), l_stage.data_ptr(),
                        sizeof(T) * bs * m * kk, cudaMemcpyHostToDevice,
                        getCurrentCUDAStream().stream());
        cudaMemcpyAsync(U.data_ptr(), u_stage.data_ptr(),
                        sizeof(T) * bs * kk * n, cudaMemcpyHostToDevice,
                        getCurrentCUDAStream().stream());
    });
    return {P, L, U};
}

// ------------------------------------------------------- eig (no MAGMA) ----

[[noreturn]] void throw_no_magma(const char* api) {
    TP_THROW(RuntimeError, "Calling ", api,
             " on a CUDA tensor requires compiling PyTorch with MAGMA. "
             "Please move the tensor to the CPU.");
}

std::tuple<Tensor, Tensor> linalg_eig_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.eig");
    throw_no_magma("torch.linalg.eig");
}

Tensor linalg_eigvals_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.eigvals");
    throw_no_magma("torch.linalg.eigvals");
}

// --------------------------------------------------------------- diagonal --

template <typename scalar_t>
__global__ void diagonal_gather_kernel(
    const scalar_t* src, scalar_t* dst, int64_t total, int64_t diag_len,
    int64_t outer, int64_t s_outer, int64_t s1, int64_t s2, int64_t base_off1,
    int64_t base_off2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t t = idx % diag_len;
    const int64_t o = idx / diag_len;
    dst[idx] = src[o * s_outer + base_off1 + t * s1 + base_off2 + t * s2];
}

Tensor linalg_diagonal_kernel_cuda(const Tensor& A, int64_t offset, int64_t dim1,
                                   int64_t dim2) {
    const char* api = "linalg.diagonal";
    check_is_matrix(A, api);
    const int64_t ndim = A.dim();
    const auto norm_dim = [&](int64_t d) {
        const int64_t r = d < 0 ? d + ndim : d;
        if (r < 0 || r >= ndim) {
            TP_THROW(RuntimeError, "Dimension out of range");
        }
        return r;
    };
    dim1 = norm_dim(dim1);
    dim2 = norm_dim(dim2);
    if (dim1 == dim2) {
        TP_THROW(RuntimeError, api, ": dimension 1 and dimension 2 cannot be equal");
    }
    // Move (dim1, dim2) to trailing positions via permute for a flat gather.
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < ndim; ++i)
        if (i != dim1 && i != dim2) perm.push_back(i);
    perm.push_back(dim1);
    perm.push_back(dim2);
    Tensor work = A.permute(perm).contiguous();
    const int64_t d1 = work.size(-2);
    const int64_t d2 = work.size(-1);
    const int64_t outer = work.numel() / (d1 * d2 == 0 ? 1 : d1 * d2);
    const int64_t diag_len =
        offset >= 0 ? std::max<int64_t>(std::min(d1, d2 - offset), 0)
                    : std::max<int64_t>(std::min(d1 + offset, d2), 0);
    Tensor out = Tensor::empty(cat_batch(std::vector<int64_t>{outer}, {diag_len}),
                               A.dtype(), A.device());
    if (out.numel() == 0) return out.reshape([&] {
        std::vector<int64_t> shape(perm.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i)
            shape[i] = i == shape.size() - 1 ? diag_len
                                             : A.size(static_cast<int64_t>(perm[i]));
        return shape;
    }());

    std::vector<int64_t> strides(ndim);
    {
        std::vector<int64_t> st(ndim, 1);
        for (int64_t i = ndim - 2; i >= 0; --i) st[i] = st[i + 1] * A.size(i + 1);
        strides = st;
    }
    const int64_t s_outer = [&] {
        int64_t acc = 1;
        bool first = true;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i == dim1 || i == dim2) continue;
            if (first) { acc = strides[i]; first = false; }
        }
        return first ? 0 : acc;
    }();
    // Base offsets fold the fixed index of every non-gathered axis.
    (void)s_outer;
    // Simpler correct path: iterate with explicit strides over the two dims.
    const int64_t total = outer * diag_len;
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const int64_t inner_block = d1 * d2;
        diagonal_gather_kernel<T><<<static_cast<unsigned>((total + 255) / 256),
                                    256, 0,
                                    getCurrentCUDAStream().stream()>>>(
            work.data_ptr<T>(), out.data_ptr<T>(), total, diag_len, outer,
            inner_block,
            /*s1=*/strides[ndim - 2], /*s2=*/strides[ndim - 1],
            offset >= 0 ? 0 : -offset * strides[ndim - 2],
            offset >= 0 ? offset * strides[ndim - 1] : 0);
    });
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) out_shape.push_back(A.size(i));
    }
    out_shape.push_back(diag_len);
    return out.reshape(out_shape);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, LinalgKernels) {
    m.impl("linalg_cholesky", linalg_cholesky_kernel_cuda);
    m.impl("linalg_cholesky_ex", linalg_cholesky_ex_kernel_cuda);
    m.impl("linalg_inv", linalg_inv_kernel_cuda);
    m.impl("linalg_inv_ex", linalg_inv_ex_kernel_cuda);
    m.impl("linalg_det", linalg_det_kernel_cuda);
    m.impl("linalg_slogdet", linalg_slogdet_kernel_cuda);
    m.impl("linalg_solve", linalg_solve_kernel_cuda);
    m.impl("linalg_solve_ex", linalg_solve_ex_kernel_cuda);
    m.impl("linalg_lu_factor", linalg_lu_factor_kernel_cuda);
    m.impl("linalg_lu_factor_ex", linalg_lu_factor_ex_kernel_cuda);
    m.impl("linalg_lu", linalg_lu_kernel_cuda);
    m.impl("linalg_lu_solve", linalg_lu_solve_kernel_cuda);
    m.impl("linalg_solve_triangular", linalg_solve_triangular_kernel_cuda);
    m.impl("linalg_eigh", linalg_eigh_kernel_cuda);
    m.impl("linalg_eigvalsh", linalg_eigvalsh_kernel_cuda);
    m.impl("linalg_eig", linalg_eig_kernel_cuda);
    m.impl("linalg_eigvals", linalg_eigvals_kernel_cuda);
    m.impl("linalg_svd", linalg_svd_kernel_cuda);
    m.impl("linalg_svdvals", linalg_svdvals_kernel_cuda);
    m.impl("linalg_lstsq", linalg_lstsq_kernel_cuda);
    m.impl("linalg_qr", linalg_qr_kernel_cuda);
    m.impl("linalg_householder_product", linalg_householder_product_kernel_cuda);
    m.impl("linalg_diagonal", linalg_diagonal_kernel_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
