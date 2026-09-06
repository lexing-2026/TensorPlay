// CUDA singular-value decomposition kernels.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "LinearAlgebraNames.h"
#include "Complex.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay::cuda {

namespace {

#define TP_SVD_CUSOLVER_CHECK(condition)                                      \
    do {                                                                       \
        const cusolverStatus_t status = (condition);                          \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            TP_THROW(RuntimeError, "cuSOLVER error ",                        \
                     std::to_string(static_cast<int>(status)));               \
        }                                                                       \
    } while (0)

constexpr int kGesvdjMaxSweeps = 400;

std::vector<int64_t> batch_shape_of(const Tensor& tensor) {
    const Size shape = tensor.shape();
    return std::vector<int64_t>(shape.begin(), shape.end() - 2);
}

std::vector<int64_t> append_shape(const std::vector<int64_t>& batch,
                                  int64_t rows, int64_t cols) {
    std::vector<int64_t> result = batch;
    result.push_back(rows);
    result.push_back(cols);
    return result;
}

int64_t batch_count_of(const Tensor& tensor) {
    int64_t result = 1;
    for (int64_t dim = 0; dim < tensor.dim() - 2; ++dim) {
        result *= tensor.size(dim);
    }
    return result;
}

int64_t matrix_stride_of(const Tensor& tensor) {
    return tensor.size(-2) * tensor.size(-1);
}

Tensor clone_batched_column_major(const Tensor& source) {
    return source.transpose(-2, -1)
        .clone(static_cast<int64_t>(MemoryFormat::Contiguous))
        .transpose(-2, -1);
}

Tensor empty_column_major(const std::vector<int64_t>& shape, DType dtype,
                          Device device) {
    std::vector<int64_t> transposed = shape;
    std::swap(transposed[transposed.size() - 2],
              transposed[transposed.size() - 1]);
    return Tensor::empty(transposed, dtype, device).transpose(-2, -1);
}

void check_is_matrix(const Tensor& input) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError,
                 "linalg.svd: The input tensor A must have at least 2 dimensions.");
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

template <typename Fn>
void dispatch_svd_dtype(DType dtype, Fn&& fn) {
    switch (dtype) {
        case DType::Float32:
            fn(static_cast<float*>(nullptr));
            return;
        case DType::Float64:
            fn(static_cast<double*>(nullptr));
            return;
        case DType::ComplexFloat:
            fn(static_cast<tensorplay::complex<float>*>(nullptr));
            return;
        case DType::ComplexDouble:
            fn(static_cast<tensorplay::complex<double>*>(nullptr));
            return;
        default:
            TP_THROW(NotImplementedError,
                     "linalg.svd: unsupported dtype ", pretty_dtype_name(dtype),
                     " on CUDA");
    }
}

void check_svd_infos(const Tensor& infos, bool is_matrix) {
    if (infos.numel() == 0) return;
    Tensor host = infos.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    const int32_t* data = host.data_ptr<int32_t>();
    for (int64_t index = 0; index < host.numel(); ++index) {
        const int32_t value = data[index];
        if (value == 0) continue;
        const std::string batch = is_matrix
            ? std::string()
            : ": (Batch element " + std::to_string(index) + ")";
        if (value < 0) {
            TP_THROW(RuntimeError, "linalg.svd", batch, ": Argument ", -value,
                     " has illegal value.");
        }
        TP_THROW(RuntimeError, "linalg.svd", batch,
                 ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ",
                 value, ").");
    }
}

template <typename scalar_t>
struct SvdCusolverTraits;

template <>
struct SvdCusolverTraits<float> {
    using value_t = float;

    static cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle,
                                              int m, int n, int* lwork) {
        return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
    }
    static cusolverStatus_t gesvd(
            cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
            int m, int n, float* a, int lda, float* s, float* u, int ldu,
            float* vt, int ldvt, float* work, int lwork, float* rwork,
            int* info) {
        return cusolverDnSgesvd(handle, jobu, jobvt, m, n, a, lda, s, u, ldu,
                                vt, ldvt, work, lwork, rwork, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, float* a, int lda, float* s, float* u, int ldu,
            float* v, int ldv, int* lwork, gesvdjInfo_t params) {
        return cusolverDnSgesvdj_bufferSize(
                handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, lwork,
                params);
    }
    static cusolverStatus_t gesvdj(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, float* a, int lda, float* s, float* u, int ldu,
            float* v, int ldv, float* work, int lwork, int* info,
            gesvdjInfo_t params) {
        return cusolverDnSgesvdj(handle, jobz, econ, m, n, a, lda, s, u, ldu,
                                 v, ldv, work, lwork, info, params);
    }
    static cusolverStatus_t gesvdj_batched_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            float* a, int lda, float* s, float* u, int ldu, float* v, int ldv,
            int* lwork, gesvdjInfo_t params, int batch) {
        return cusolverDnSgesvdjBatched_bufferSize(
                handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, lwork, params,
                batch);
    }
    static cusolverStatus_t gesvdj_batched(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            float* a, int lda, float* s, float* u, int ldu, float* v, int ldv,
            float* work, int lwork, int* info, gesvdjInfo_t params, int batch) {
        return cusolverDnSgesvdjBatched(
                handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, work, lwork,
                info, params, batch);
    }
    static cusolverStatus_t gesvda_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, float* a, int lda, long long stride_a, float* s,
            long long stride_s, float* u, int ldu, long long stride_u, float* v,
            int ldv, long long stride_v, int* lwork, int batch) {
        return cusolverDnSgesvdaStridedBatched_bufferSize(
                handle, jobz, rank, m, n, a, lda, stride_a, s, stride_s, u,
                ldu, stride_u, v, ldv, stride_v, lwork, batch);
    }
    static cusolverStatus_t gesvda(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, float* a, int lda, long long stride_a, float* s,
            long long stride_s, float* u, int ldu, long long stride_u, float* v,
            int ldv, long long stride_v, float* work, int lwork, int* info,
            double* residual, int batch) {
        return cusolverDnSgesvdaStridedBatched(
                handle, jobz, rank, m, n, a, lda, stride_a, s, stride_s, u,
                ldu, stride_u, v, ldv, stride_v, work, lwork, info, residual,
                batch);
    }
};

template <>
struct SvdCusolverTraits<double> {
    using value_t = double;

    static cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle,
                                              int m, int n, int* lwork) {
        return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
    }
    static cusolverStatus_t gesvd(
            cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
            int m, int n, double* a, int lda, double* s, double* u, int ldu,
            double* vt, int ldvt, double* work, int lwork, double* rwork,
            int* info) {
        return cusolverDnDgesvd(handle, jobu, jobvt, m, n, a, lda, s, u, ldu,
                                vt, ldvt, work, lwork, rwork, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, double* a, int lda, double* s, double* u, int ldu,
            double* v, int ldv, int* lwork, gesvdjInfo_t params) {
        return cusolverDnDgesvdj_bufferSize(
                handle, jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, lwork,
                params);
    }
    static cusolverStatus_t gesvdj(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, double* a, int lda, double* s, double* u, int ldu,
            double* v, int ldv, double* work, int lwork, int* info,
            gesvdjInfo_t params) {
        return cusolverDnDgesvdj(handle, jobz, econ, m, n, a, lda, s, u, ldu,
                                 v, ldv, work, lwork, info, params);
    }
    static cusolverStatus_t gesvdj_batched_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            double* a, int lda, double* s, double* u, int ldu, double* v,
            int ldv, int* lwork, gesvdjInfo_t params, int batch) {
        return cusolverDnDgesvdjBatched_bufferSize(
                handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, lwork, params,
                batch);
    }
    static cusolverStatus_t gesvdj_batched(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            double* a, int lda, double* s, double* u, int ldu, double* v,
            int ldv, double* work, int lwork, int* info, gesvdjInfo_t params,
            int batch) {
        return cusolverDnDgesvdjBatched(
                handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, work, lwork,
                info, params, batch);
    }
    static cusolverStatus_t gesvda_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, double* a, int lda, long long stride_a, double* s,
            long long stride_s, double* u, int ldu, long long stride_u,
            double* v, int ldv, long long stride_v, int* lwork, int batch) {
        return cusolverDnDgesvdaStridedBatched_bufferSize(
                handle, jobz, rank, m, n, a, lda, stride_a, s, stride_s, u,
                ldu, stride_u, v, ldv, stride_v, lwork, batch);
    }
    static cusolverStatus_t gesvda(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, double* a, int lda, long long stride_a, double* s,
            long long stride_s, double* u, int ldu, long long stride_u,
            double* v, int ldv, long long stride_v, double* work, int lwork,
            int* info, double* residual, int batch) {
        return cusolverDnDgesvdaStridedBatched(
                handle, jobz, rank, m, n, a, lda, stride_a, s, stride_s, u,
                ldu, stride_u, v, ldv, stride_v, work, lwork, info, residual,
                batch);
    }
};

template <>
struct SvdCusolverTraits<tensorplay::complex<float>> {
    using value_t = float;

    static cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle,
                                              int m, int n, int* lwork) {
        return cusolverDnCgesvd_bufferSize(handle, m, n, lwork);
    }
    static cusolverStatus_t gesvd(
            cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
            int m, int n, tensorplay::complex<float>* a, int lda, float* s,
            tensorplay::complex<float>* u, int ldu, tensorplay::complex<float>* vt, int ldvt,
            tensorplay::complex<float>* work, int lwork, float* rwork, int* info) {
        return cusolverDnCgesvd(
                handle, jobu, jobvt, m, n, reinterpret_cast<cuComplex*>(a), lda,
                s, reinterpret_cast<cuComplex*>(u), ldu,
                reinterpret_cast<cuComplex*>(vt), ldvt,
                reinterpret_cast<cuComplex*>(work), lwork, rwork, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, tensorplay::complex<float>* a, int lda, float* s,
            tensorplay::complex<float>* u, int ldu, tensorplay::complex<float>* v, int ldv,
            int* lwork, gesvdjInfo_t params) {
        return cusolverDnCgesvdj_bufferSize(
                handle, jobz, econ, m, n, reinterpret_cast<cuComplex*>(a), lda,
                s, reinterpret_cast<cuComplex*>(u), ldu,
                reinterpret_cast<cuComplex*>(v), ldv, lwork, params);
    }
    static cusolverStatus_t gesvdj(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, tensorplay::complex<float>* a, int lda, float* s,
            tensorplay::complex<float>* u, int ldu, tensorplay::complex<float>* v, int ldv,
            tensorplay::complex<float>* work, int lwork, int* info,
            gesvdjInfo_t params) {
        return cusolverDnCgesvdj(
                handle, jobz, econ, m, n, reinterpret_cast<cuComplex*>(a), lda,
                s, reinterpret_cast<cuComplex*>(u), ldu,
                reinterpret_cast<cuComplex*>(v), ldv,
                reinterpret_cast<cuComplex*>(work), lwork, info, params);
    }
    static cusolverStatus_t gesvdj_batched_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            tensorplay::complex<float>* a, int lda, float* s, tensorplay::complex<float>* u,
            int ldu, tensorplay::complex<float>* v, int ldv, int* lwork,
            gesvdjInfo_t params, int batch) {
        return cusolverDnCgesvdjBatched_bufferSize(
                handle, jobz, m, n, reinterpret_cast<cuComplex*>(a), lda, s,
                reinterpret_cast<cuComplex*>(u), ldu,
                reinterpret_cast<cuComplex*>(v), ldv, lwork, params, batch);
    }
    static cusolverStatus_t gesvdj_batched(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            tensorplay::complex<float>* a, int lda, float* s, tensorplay::complex<float>* u,
            int ldu, tensorplay::complex<float>* v, int ldv,
            tensorplay::complex<float>* work, int lwork, int* info,
            gesvdjInfo_t params, int batch) {
        return cusolverDnCgesvdjBatched(
                handle, jobz, m, n, reinterpret_cast<cuComplex*>(a), lda, s,
                reinterpret_cast<cuComplex*>(u), ldu,
                reinterpret_cast<cuComplex*>(v), ldv,
                reinterpret_cast<cuComplex*>(work), lwork, info, params, batch);
    }
    static cusolverStatus_t gesvda_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, tensorplay::complex<float>* a, int lda, long long stride_a,
            float* s, long long stride_s, tensorplay::complex<float>* u, int ldu,
            long long stride_u, tensorplay::complex<float>* v, int ldv,
            long long stride_v, int* lwork, int batch) {
        return cusolverDnCgesvdaStridedBatched_bufferSize(
                handle, jobz, rank, m, n, reinterpret_cast<cuComplex*>(a), lda,
                stride_a, s, stride_s, reinterpret_cast<cuComplex*>(u), ldu,
                stride_u, reinterpret_cast<cuComplex*>(v), ldv, stride_v,
                lwork, batch);
    }
    static cusolverStatus_t gesvda(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, tensorplay::complex<float>* a, int lda, long long stride_a,
            float* s, long long stride_s, tensorplay::complex<float>* u, int ldu,
            long long stride_u, tensorplay::complex<float>* v, int ldv,
            long long stride_v, tensorplay::complex<float>* work, int lwork,
            int* info, double* residual, int batch) {
        return cusolverDnCgesvdaStridedBatched(
                handle, jobz, rank, m, n, reinterpret_cast<cuComplex*>(a), lda,
                stride_a, s, stride_s, reinterpret_cast<cuComplex*>(u), ldu,
                stride_u, reinterpret_cast<cuComplex*>(v), ldv, stride_v,
                reinterpret_cast<cuComplex*>(work), lwork, info, residual,
                batch);
    }
};

template <>
struct SvdCusolverTraits<tensorplay::complex<double>> {
    using value_t = double;

    static cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle,
                                              int m, int n, int* lwork) {
        return cusolverDnZgesvd_bufferSize(handle, m, n, lwork);
    }
    static cusolverStatus_t gesvd(
            cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
            int m, int n, tensorplay::complex<double>* a, int lda, double* s,
            tensorplay::complex<double>* u, int ldu, tensorplay::complex<double>* vt,
            int ldvt, tensorplay::complex<double>* work, int lwork, double* rwork,
            int* info) {
        return cusolverDnZgesvd(
                handle, jobu, jobvt, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, s,
                reinterpret_cast<cuDoubleComplex*>(u), ldu,
                reinterpret_cast<cuDoubleComplex*>(vt), ldvt,
                reinterpret_cast<cuDoubleComplex*>(work), lwork, rwork, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, tensorplay::complex<double>* a, int lda, double* s,
            tensorplay::complex<double>* u, int ldu, tensorplay::complex<double>* v, int ldv,
            int* lwork, gesvdjInfo_t params) {
        return cusolverDnZgesvdj_bufferSize(
                handle, jobz, econ, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, s,
                reinterpret_cast<cuDoubleComplex*>(u), ldu,
                reinterpret_cast<cuDoubleComplex*>(v), ldv, lwork, params);
    }
    static cusolverStatus_t gesvdj(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ,
            int m, int n, tensorplay::complex<double>* a, int lda, double* s,
            tensorplay::complex<double>* u, int ldu, tensorplay::complex<double>* v, int ldv,
            tensorplay::complex<double>* work, int lwork, int* info,
            gesvdjInfo_t params) {
        return cusolverDnZgesvdj(
                handle, jobz, econ, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, s,
                reinterpret_cast<cuDoubleComplex*>(u), ldu,
                reinterpret_cast<cuDoubleComplex*>(v), ldv,
                reinterpret_cast<cuDoubleComplex*>(work), lwork, info, params);
    }
    static cusolverStatus_t gesvdj_batched_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            tensorplay::complex<double>* a, int lda, double* s,
            tensorplay::complex<double>* u, int ldu, tensorplay::complex<double>* v, int ldv,
            int* lwork, gesvdjInfo_t params, int batch) {
        return cusolverDnZgesvdjBatched_bufferSize(
                handle, jobz, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, s,
                reinterpret_cast<cuDoubleComplex*>(u), ldu,
                reinterpret_cast<cuDoubleComplex*>(v), ldv, lwork, params,
                batch);
    }
    static cusolverStatus_t gesvdj_batched(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,
            tensorplay::complex<double>* a, int lda, double* s,
            tensorplay::complex<double>* u, int ldu, tensorplay::complex<double>* v, int ldv,
            tensorplay::complex<double>* work, int lwork, int* info,
            gesvdjInfo_t params, int batch) {
        return cusolverDnZgesvdjBatched(
                handle, jobz, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, s,
                reinterpret_cast<cuDoubleComplex*>(u), ldu,
                reinterpret_cast<cuDoubleComplex*>(v), ldv,
                reinterpret_cast<cuDoubleComplex*>(work), lwork, info, params,
                batch);
    }
    static cusolverStatus_t gesvda_buffer_size(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, tensorplay::complex<double>* a, int lda, long long stride_a,
            double* s, long long stride_s, tensorplay::complex<double>* u, int ldu,
            long long stride_u, tensorplay::complex<double>* v, int ldv,
            long long stride_v, int* lwork, int batch) {
        return cusolverDnZgesvdaStridedBatched_bufferSize(
                handle, jobz, rank, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, stride_a, s,
                stride_s, reinterpret_cast<cuDoubleComplex*>(u), ldu, stride_u,
                reinterpret_cast<cuDoubleComplex*>(v), ldv, stride_v, lwork,
                batch);
    }
    static cusolverStatus_t gesvda(
            cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
            int n, tensorplay::complex<double>* a, int lda, long long stride_a,
            double* s, long long stride_s, tensorplay::complex<double>* u, int ldu,
            long long stride_u, tensorplay::complex<double>* v, int ldv,
            long long stride_v, tensorplay::complex<double>* work, int lwork,
            int* info, double* residual, int batch) {
        return cusolverDnZgesvdaStridedBatched(
                handle, jobz, rank, m, n,
                reinterpret_cast<cuDoubleComplex*>(a), lda, stride_a, s,
                stride_s, reinterpret_cast<cuDoubleComplex*>(u), ldu,
                stride_u, reinterpret_cast<cuDoubleComplex*>(v), ldv,
                stride_v, reinterpret_cast<cuDoubleComplex*>(work), lwork,
                info, residual, batch);
    }
};

template <typename scalar_t>
void configure_gesvdj(gesvdjInfo_t params) {
    using value_t = typename SvdCusolverTraits<scalar_t>::value_t;
    TP_SVD_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(
            params, std::numeric_limits<value_t>::epsilon()));
    TP_SVD_CUSOLVER_CHECK(
            cusolverDnXgesvdjSetMaxSweeps(params, kGesvdjMaxSweeps));
}

template <typename scalar_t>
void apply_svd_gesvdj(const Tensor& input, const Tensor& U, const Tensor& S,
                      const Tensor& V, const Tensor& infos,
                      bool full_matrices, bool compute_uv) {
    using Tr = SvdCusolverTraits<scalar_t>;
    using value_t = typename Tr::value_t;
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int k = std::min(m, n);
    const int batch = static_cast<int>(batch_count_of(input));

    Tensor U_workspace;
    Tensor V_workspace;
    if (!compute_uv) {
        U_workspace = Tensor::empty({m * k}, input.dtype(), input.device());
        V_workspace = Tensor::empty({n * k}, input.dtype(), input.device());
    }

    scalar_t* input_data = input.data_ptr<scalar_t>();
    value_t* singular_data = S.data_ptr<value_t>();
    scalar_t* u_data = compute_uv ? U.data_ptr<scalar_t>()
                                  : U_workspace.data_ptr<scalar_t>();
    scalar_t* v_data = compute_uv ? V.data_ptr<scalar_t>()
                                  : V_workspace.data_ptr<scalar_t>();
    const int64_t input_stride = matrix_stride_of(input);
    const int64_t singular_stride = S.size(-1);
    const int64_t u_stride = compute_uv ? matrix_stride_of(U) : 0;
    const int64_t v_stride = compute_uv ? matrix_stride_of(V) : 0;
    const int lda = static_cast<int>(input.stride(-1));
    const int ldu = compute_uv ? static_cast<int>(U.stride(-1)) : m;
    const int ldv = compute_uv ? static_cast<int>(V.stride(-1)) : n;
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cusolverEigMode_t jobz = compute_uv
        ? CUSOLVER_EIG_MODE_VECTOR
        : CUSOLVER_EIG_MODE_NOVECTOR;
    const int econ = full_matrices ? 0 : 1;

    gesvdjInfo_t params = nullptr;
    TP_SVD_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
    configure_gesvdj<scalar_t>(params);

    int lwork = 0;
    TP_SVD_CUSOLVER_CHECK(Tr::gesvdj_buffer_size(
            handle, jobz, econ, m, n, input_data, lda, singular_data, u_data,
            ldu, v_data, ldv, &lwork, params));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());

    for (int i = 0; i < batch; ++i) {
        TP_SVD_CUSOLVER_CHECK(Tr::gesvdj(
                handle, jobz, econ, m, n,
                input_data + i * input_stride, lda,
                singular_data + i * singular_stride,
                u_data + i * u_stride, ldu,
                v_data + i * v_stride, ldv,
                work.data_ptr<scalar_t>(), lwork,
                infos.data_ptr<int32_t>() + i, params));
    }

    TP_SVD_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
}

template <typename scalar_t>
void apply_svd_gesvdj_batched(const Tensor& input, const Tensor& U,
                              const Tensor& S, const Tensor& V,
                              const Tensor& infos, bool compute_uv) {
    using Tr = SvdCusolverTraits<scalar_t>;
    using value_t = typename Tr::value_t;
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int batch = static_cast<int>(batch_count_of(input));
    const int lda = std::max(1, m);
    const int ldu = std::max(1, m);
    const int ldv = std::max(1, n);

    Tensor U_workspace;
    Tensor V_workspace;
    if (!compute_uv) {
        U_workspace = Tensor::empty(
                {static_cast<int64_t>(batch) * m * ldu}, input.dtype(),
                input.device());
        V_workspace = Tensor::empty(
                {static_cast<int64_t>(batch) * n * ldv}, input.dtype(),
                input.device());
    }

    scalar_t* input_data = input.data_ptr<scalar_t>();
    value_t* singular_data = S.data_ptr<value_t>();
    scalar_t* u_data = compute_uv ? U.data_ptr<scalar_t>()
                                  : U_workspace.data_ptr<scalar_t>();
    scalar_t* v_data = compute_uv ? V.data_ptr<scalar_t>()
                                  : V_workspace.data_ptr<scalar_t>();
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cusolverEigMode_t jobz = compute_uv
        ? CUSOLVER_EIG_MODE_VECTOR
        : CUSOLVER_EIG_MODE_NOVECTOR;

    gesvdjInfo_t params = nullptr;
    TP_SVD_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
    configure_gesvdj<scalar_t>(params);
    TP_SVD_CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(params, 1));

    int lwork = 0;
    TP_SVD_CUSOLVER_CHECK(Tr::gesvdj_batched_buffer_size(
            handle, jobz, m, n, input_data, lda, singular_data, u_data, ldu,
            v_data, ldv, &lwork, params, batch));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());
    TP_SVD_CUSOLVER_CHECK(Tr::gesvdj_batched(
            handle, jobz, m, n, input_data, lda, singular_data, u_data, ldu,
            v_data, ldv, work.data_ptr<scalar_t>(), lwork,
            infos.data_ptr<int32_t>(), params, batch));
    TP_SVD_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
}

void svd_cusolver_gesvdj_batched(const Tensor& input, Tensor U,
                                 const Tensor& S, Tensor V,
                                 const Tensor& infos, bool full_matrices,
                                 bool compute_uv) {
    const int64_t m = input.size(-2);
    const int64_t n = input.size(-1);
    const int64_t k = std::min(m, n);
    Tensor U_work = U;
    Tensor V_work = V;
    bool copy_u = false;
    bool copy_v = false;
    if (compute_uv && !full_matrices) {
        const auto batch = batch_shape_of(input);
        if (m > n) {
            U_work = empty_column_major(append_shape(batch, m, m),
                                        input.dtype(), input.device());
            copy_u = true;
        } else if (m < n) {
            V_work = empty_column_major(append_shape(batch, n, n),
                                        input.dtype(), input.device());
            copy_v = true;
        }
    }

    dispatch_svd_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = std::remove_pointer_t<decltype(tag)>;
        apply_svd_gesvdj_batched<scalar_t>(input, U_work, S, V_work, infos,
                                           compute_uv);
    });

    if (copy_u) U.copy_(U_work.narrow(-1, 0, k));
    if (copy_v) V.copy_(V_work.narrow(-1, 0, k));
}

template <typename scalar_t>
void apply_svd_gesvd(const Tensor& input, const Tensor& U, const Tensor& S,
                     const Tensor& V, const Tensor& infos,
                     bool full_matrices, bool compute_uv) {
    using Tr = SvdCusolverTraits<scalar_t>;
    using value_t = typename Tr::value_t;
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int k = std::min(m, n);
    const int batch = static_cast<int>(batch_count_of(input));
    const int lda = std::max(1, m);
    const int ldv = std::max(1, n);
    const auto handle = CUDAContext::getCusolverDnHandle();
    const signed char job = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

    int lwork = 0;
    TP_SVD_CUSOLVER_CHECK(
            Tr::gesvd_buffer_size(handle, m, n, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());
    Tensor rwork = Tensor::empty(
            {std::max(k, 1)}, toRealValueType(input.dtype()), input.device());
    Tensor V_workspace = compute_uv
        ? Tensor::empty({n, full_matrices ? n : k}, input.dtype(),
                        input.device()).conj()
        : Tensor();
    Tensor V_view = compute_uv
        ? V.view({batch, n, V.size(-1)})
        : Tensor();

    scalar_t* input_data = input.data_ptr<scalar_t>();
    value_t* singular_data = S.data_ptr<value_t>();
    scalar_t* u_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
    scalar_t* vt_data = compute_uv ? V_workspace.data_ptr<scalar_t>() : nullptr;
    const int64_t input_stride = matrix_stride_of(input);
    const int64_t singular_stride = S.size(-1);
    const int64_t u_stride = compute_uv ? matrix_stride_of(U) : 0;

    for (int i = 0; i < batch; ++i) {
        TP_SVD_CUSOLVER_CHECK(Tr::gesvd(
                handle, job, job, m, n, input_data + i * input_stride, lda,
                singular_data + i * singular_stride,
                compute_uv ? u_data + i * u_stride : nullptr, lda,
                vt_data, ldv, work.data_ptr<scalar_t>(), lwork,
                rwork.data_ptr<value_t>(), infos.data_ptr<int32_t>() + i));
        if (compute_uv) {
            V_view.select(0, i).copy_(V_workspace);
        }
    }
}

void svd_cusolver_gesvd(const Tensor& input, const Tensor& U,
                        const Tensor& S, const Tensor& V, const Tensor& infos,
                        bool full_matrices, bool compute_uv) {
    const bool not_transposed = input.size(-2) >= input.size(-1);
    Tensor source = not_transposed
        ? input
        : input.transpose(-2, -1).conj();
    Tensor working = clone_batched_column_major(source);
    dispatch_svd_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = std::remove_pointer_t<decltype(tag)>;
        apply_svd_gesvd<scalar_t>(
                working, not_transposed ? U : V, S,
                not_transposed ? V : U, infos, full_matrices, compute_uv);
    });
}

template <typename scalar_t>
void apply_svd_gesvda(const Tensor& input, const Tensor& U, const Tensor& S,
                      const Tensor& V, const Tensor& infos,
                      bool compute_uv) {
    using Tr = SvdCusolverTraits<scalar_t>;
    using value_t = typename Tr::value_t;
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int batch = static_cast<int>(batch_count_of(input));
    const int64_t rank = S.size(-1);
    const int lda = static_cast<int>(input.stride(-1));
    const int ldu = compute_uv ? static_cast<int>(U.stride(-1)) : m;
    const int ldv = compute_uv ? static_cast<int>(V.stride(-1)) : n;
    const int64_t input_stride = matrix_stride_of(input);
    const int64_t singular_stride = S.size(-1);
    const int64_t u_stride = compute_uv ? matrix_stride_of(U)
                                       : static_cast<int64_t>(ldu) * rank;
    const int64_t v_stride = compute_uv ? matrix_stride_of(V)
                                       : static_cast<int64_t>(ldv) * rank;

    Tensor U_workspace;
    Tensor V_workspace;
    if (!compute_uv) {
        U_workspace = Tensor::empty(
                {static_cast<int64_t>(batch) * m * n}, input.dtype(),
                input.device());
        V_workspace = Tensor::empty(
                {static_cast<int64_t>(batch) * n * n}, input.dtype(),
                input.device());
    }
    scalar_t* input_data = input.data_ptr<scalar_t>();
    value_t* singular_data = S.data_ptr<value_t>();
    scalar_t* u_data = compute_uv ? U.data_ptr<scalar_t>()
                                  : U_workspace.data_ptr<scalar_t>();
    scalar_t* v_data = compute_uv ? V.data_ptr<scalar_t>()
                                  : V_workspace.data_ptr<scalar_t>();
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cusolverEigMode_t jobz = compute_uv
        ? CUSOLVER_EIG_MODE_VECTOR
        : CUSOLVER_EIG_MODE_NOVECTOR;

    int lwork = 0;
    TP_SVD_CUSOLVER_CHECK(Tr::gesvda_buffer_size(
            handle, jobz, static_cast<int>(rank), m, n, input_data, lda,
            input_stride, singular_data, singular_stride, u_data, ldu,
            u_stride, v_data, ldv, v_stride, &lwork, batch));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());
    TP_SVD_CUSOLVER_CHECK(Tr::gesvda(
            handle, jobz, static_cast<int>(rank), m, n, input_data, lda,
            input_stride, singular_data, singular_stride, u_data, ldu,
            u_stride, v_data, ldv, v_stride, work.data_ptr<scalar_t>(), lwork,
            infos.data_ptr<int32_t>(), nullptr, batch));
}

void svd_cusolver_gesvda(const Tensor& input, const Tensor& U,
                         const Tensor& S, const Tensor& V, const Tensor& infos,
                         bool compute_uv) {
    const bool not_transposed = input.size(-2) >= input.size(-1);
    Tensor source = not_transposed
        ? input
        : input.transpose(-2, -1).conj();
    Tensor working = clone_batched_column_major(source);
    dispatch_svd_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = std::remove_pointer_t<decltype(tag)>;
        apply_svd_gesvda<scalar_t>(
                working, not_transposed ? U : V, S,
                not_transposed ? V : U, infos, compute_uv);
    });
}

std::tuple<Tensor, Tensor, Tensor> svd_impl_cuda(
        const Tensor& input, bool full_matrices, bool compute_uv,
        const std::optional<std::string>& driver) {
    check_is_matrix(input);
    const auto batch = batch_shape_of(input);
    const int64_t m = input.size(-2);
    const int64_t n = input.size(-1);
    const int64_t k = std::min(m, n);
    Tensor U;
    Tensor V;
    if (compute_uv) {
        U = empty_column_major(
                append_shape(batch, m, full_matrices ? m : k), input.dtype(),
                input.device());
        V = empty_column_major(
                append_shape(batch, n, full_matrices ? n : k), input.dtype(),
                input.device());
    } else {
        U = Tensor::empty({0}, input.dtype(), input.device());
        V = Tensor::empty({0}, input.dtype(), input.device());
    }
    std::vector<int64_t> singular_shape = batch;
    singular_shape.push_back(k);
    Tensor S = Tensor::empty(singular_shape, toRealValueType(input.dtype()),
                             input.device());
    Tensor infos = Tensor::zeros(batch, DType::Int32, input.device());

    if (input.numel() == 0) {
        if (compute_uv && full_matrices) {
            if (U.numel() != 0) {
                U.zero_();
                U.diagonal(0, -2, -1).fill_(Scalar(1));
            }
            if (V.numel() != 0) {
                V.zero_();
                V.diagonal(0, -2, -1).fill_(Scalar(1));
            }
        }
    } else {
        const std::string selected_driver = driver.value_or("gesvdj");
        if (selected_driver == "gesvd") {
            svd_cusolver_gesvd(input, U, S, V, infos, full_matrices,
                               compute_uv);
        } else if (selected_driver == "gesvdj") {
            Tensor working = clone_batched_column_major(input);
            if (m <= 32 && n <= 32) {
                svd_cusolver_gesvdj_batched(
                        working, U, S, V, infos, full_matrices, compute_uv);
            } else {
                dispatch_svd_dtype(input.dtype(), [&](auto tag) {
                    using scalar_t = std::remove_pointer_t<decltype(tag)>;
                    apply_svd_gesvdj<scalar_t>(
                            working, U, S, V, infos, full_matrices, compute_uv);
                });
            }
        } else if (selected_driver == "gesvda") {
            svd_cusolver_gesvda(input, U, S, V, infos, compute_uv);
        } else {
            TP_THROW(RuntimeError, "linalg.svd: unknown svd driver ",
                     selected_driver);
        }
        check_svd_infos(infos, input.dim() == 2);
    }

    if (!compute_uv) return {U, S.contiguous(), V};
    Tensor Vh = V.transpose(-2, -1);
    if (isComplexType(input.dtype())) Vh = Vh.conj();
    return {U.contiguous(), S.contiguous(), Vh.contiguous()};
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> linalg_svd_internal_kernel_cuda(
        const Tensor& input, bool full_matrices, bool compute_uv,
        std::optional<std::string> driver) {
    return svd_impl_cuda(input, full_matrices, compute_uv, driver);
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_kernel_cuda(
        const Tensor& input, bool full_matrices,
        std::optional<std::string> driver) {
    return linalg_svd_internal_kernel_cuda(input, full_matrices, true, driver);
}

Tensor linalg_svdvals_kernel_cuda(const Tensor& input,
                                  std::optional<std::string> driver) {
    return std::get<1>(linalg_svd_internal_kernel_cuda(input, false, false,
                                                       driver));
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_internal_out_kernel_cuda(
        const Tensor& input, bool full_matrices, bool compute_uv,
        std::optional<std::string> driver, Tensor& U, Tensor& S, Tensor& Vh) {
    auto result = linalg_svd_internal_kernel_cuda(input, full_matrices,
                                                  compute_uv, driver);
    write_linalg_output("linalg.svd", std::get<0>(result), U);
    write_linalg_output("linalg.svd", std::get<1>(result), S);
    write_linalg_output("linalg.svd", std::get<2>(result), Vh);
    return {U, S, Vh};
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd_out_kernel_cuda(
        const Tensor& input, bool full_matrices,
        std::optional<std::string> driver, Tensor& U, Tensor& S, Tensor& Vh) {
    return linalg_svd_internal_out_kernel_cuda(input, full_matrices, true,
                                               driver, U, S, Vh);
}

Tensor& linalg_svdvals_out_kernel_cuda(const Tensor& input,
                                       std::optional<std::string> driver,
                                       Tensor& out) {
    auto result = linalg_svd_internal_kernel_cuda(input, false, false, driver);
    write_linalg_output("linalg.svdvals", std::get<1>(result), out);
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LinalgSVD) {
    m.impl("_linalg_svd", linalg_svd_internal_kernel_cuda);
    m.impl("_linalg_svd.U", linalg_svd_internal_out_kernel_cuda);
    m.impl("linalg_svd", linalg_svd_kernel_cuda);
    m.impl("linalg_svd.U", linalg_svd_out_kernel_cuda);
    m.impl("linalg_svdvals", linalg_svdvals_kernel_cuda);
    m.impl("linalg_svdvals.out", linalg_svdvals_out_kernel_cuda);
}

}  // namespace tensorplay::cuda
