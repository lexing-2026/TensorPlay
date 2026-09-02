// Native CUDA LDL factorization and solve implementation.
//
// LDL^T factorization uses cuSOLVER sytrf.  The solve path uses the same
// 64-bit pivot interface.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace {

#define CUSOLVER_CHECK(condition)                                             \
    do {                                                                      \
        const cusolverStatus_t status = (condition);                         \
        if (status != CUSOLVER_STATUS_SUCCESS) {                              \
            TP_THROW(RuntimeError, "cuSOLVER error ",                         \
                     std::to_string(static_cast<int>(status)));              \
        }                                                                       \
    } while (0)

std::vector<int64_t> batch_shape_of(const Tensor& tensor) {
    const Size shape = tensor.shape();
    return std::vector<int64_t>(shape.begin(), shape.end() - 2);
}

std::vector<int64_t> append_matrix_shape(const std::vector<int64_t>& batch,
                                         int64_t rows, int64_t cols) {
    std::vector<int64_t> result = batch;
    result.push_back(rows);
    result.push_back(cols);
    return result;
}

int64_t batch_count(const Tensor& tensor) {
    const int64_t n = tensor.size(-1);
    const int64_t matrix_elements = n * tensor.size(-2);
    return matrix_elements == 0 ? 0 : tensor.numel() / matrix_elements;
}

int64_t linear_batch_size(const std::vector<int64_t>& batch) {
    return std::accumulate(batch.begin(), batch.end(), int64_t{1},
                           std::multiplies<int64_t>());
}

std::vector<int64_t> broadcast_batch(const Tensor& a, const Tensor& b) {
    const auto a_batch = batch_shape_of(a);
    const auto b_batch = batch_shape_of(b);
    const size_t rank = std::max(a_batch.size(), b_batch.size());
    std::vector<int64_t> result(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t a_dim = i < rank - a_batch.size()
            ? 1 : a_batch[i - (rank - a_batch.size())];
        const int64_t b_dim = i < rank - b_batch.size()
            ? 1 : b_batch[i - (rank - b_batch.size())];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            TP_THROW(RuntimeError, "The size of tensor a (", a_dim,
                     ") must match the size of tensor b (", b_dim, ")");
        }
        result[i] = std::max(a_dim, b_dim);
    }
    return result;
}

Tensor expand_to_batch(const Tensor& tensor, const std::vector<int64_t>& batch) {
    return tensor.expand(append_matrix_shape(batch, tensor.size(-2), tensor.size(-1)));
}

Tensor clone_batched_column_major(const Tensor& source) {
    // cuSOLVER consumes column-major matrices.  Cloning the transposed view
    // into a contiguous row-major buffer and transposing it back gives a
    // column-major logical tensor without a host staging copy.
    return source.transpose(-2, -1)
        .clone(static_cast<int64_t>(MemoryFormat::Contiguous))
        .transpose(-2, -1);
}

void check_square(const Tensor& tensor, const char* api, const char* name) {
    if (tensor.dim() < 2) {
        TP_THROW(RuntimeError, api, ": expected ", name,
                 " to have at least two dimensions");
    }
    if (tensor.size(-1) != tensor.size(-2)) {
        TP_THROW(RuntimeError, api, ": ", name,
                 " must contain square matrices");
    }
}

void check_real_dtype(const Tensor& tensor, const char* api) {
    if (tensor.dtype() != DType::Float32 && tensor.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, api,
                 ": CUDA LDL supports Float32 and Float64 only");
    }
}

void check_factor_info(const Tensor& info, const char* api, bool is_matrix) {
    if (info.numel() == 0) return;
    Tensor host = info.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    const auto* data = host.data_ptr<int32_t>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        const int32_t value = data[i];
        if (value == 0) continue;
        const std::string batch = is_matrix
            ? std::string()
            : ": (Batch element " + std::to_string(i) + ")";
        if (value < 0) {
            TP_THROW(RuntimeError, api, batch, ": Argument ", -value,
                     " has an illegal value");
        }
        TP_THROW(RuntimeError, api, batch,
                 ": the factorization failed because the input matrix is singular");
    }
}

template <typename scalar_t>
struct LdlCusolver;

template <>
struct LdlCusolver<float> {
    static cusolverStatus_t buffer_size(cusolverDnHandle_t handle, int n,
                                         float* matrix, int lda, int* work) {
        return cusolverDnSsytrf_bufferSize(handle, n, matrix, lda, work);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, float* matrix, int lda, int* pivots,
                                   float* work, int lwork, int* info) {
        return cusolverDnSsytrf(handle, uplo, n, matrix, lda, pivots,
                                work, lwork, info);
    }
};

template <>
struct LdlCusolver<double> {
    static cusolverStatus_t buffer_size(cusolverDnHandle_t handle, int n,
                                         double* matrix, int lda, int* work) {
        return cusolverDnDsytrf_bufferSize(handle, n, matrix, lda, work);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, double* matrix, int lda, int* pivots,
                                   double* work, int lwork, int* info) {
        return cusolverDnDsytrf(handle, uplo, n, matrix, lda, pivots,
                                work, lwork, info);
    }
};

template <typename scalar_t>
void apply_ldl_factor(const Tensor& ld, const Tensor& pivots,
                      const Tensor& info) {
    const int n = static_cast<int>(ld.size(-2));
    const int lda = static_cast<int>(ld.stride(-1));
    const int64_t matrices = batch_count(ld);
    const int64_t matrix_stride = ld.dim() > 2 ? ld.stride(-3) : 0;
    const int64_t pivot_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

    scalar_t* matrix_data = ld.data_ptr<scalar_t>();
    int32_t* pivot_data = pivots.data_ptr<int32_t>();
    int32_t* info_data = info.data_ptr<int32_t>();
    cusolverDnHandle_t handle = CUDAContext::getCusolverDnHandle();

    int lwork = 0;
    CUSOLVER_CHECK(LdlCusolver<scalar_t>::buffer_size(
        handle, n, matrix_data, lda, &lwork));
    Tensor work = Tensor::empty(
        {std::max(lwork, 1)}, ld.dtype(), ld.device());
    for (int64_t i = 0; i < matrices; ++i) {
        CUSOLVER_CHECK(LdlCusolver<scalar_t>::factor(
            handle, CUBLAS_FILL_MODE_LOWER, n,
            matrix_data + i * matrix_stride, lda,
            pivot_data + i * pivot_stride, work.data_ptr<scalar_t>(),
            lwork, info_data + i));
    }
}

template <typename scalar_t>
cudaDataType ldl_cuda_dtype();

template <>
cudaDataType ldl_cuda_dtype<float>() { return CUDA_R_32F; }

template <>
cudaDataType ldl_cuda_dtype<double>() { return CUDA_R_64F; }

template <typename scalar_t>
void apply_ldl_solve(const Tensor& ld, const Tensor& pivots,
                     const Tensor& result) {
    const int64_t matrices = batch_count(result);
    const int64_t n = ld.size(-2);
    const int64_t nrhs = result.size(-1);
    const int64_t lda = ld.stride(-1);
    const int64_t ldb = result.stride(-1);
    const int64_t matrix_stride = ld.dim() > 2 ? ld.stride(-3) : 0;
    const int64_t result_stride = result.dim() > 2 ? result.stride(-3) : 0;
    const int64_t pivot_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

    const scalar_t* matrix_data = ld.data_ptr<scalar_t>();
    const int64_t* pivot_data = pivots.data_ptr<int64_t>();
    scalar_t* result_data = result.data_ptr<scalar_t>();
    cusolverDnHandle_t handle = CUDAContext::getCusolverDnHandle();

#if defined(USE_ROCM)
    // The batched-solve entry the solve path relies on is not part of the
    // AMD solver library (only the factorization routine exists there), so
    // report the limitation instead of failing to link.
    (void)handle;
    (void)matrix_data;
    (void)pivot_data;
    (void)result_data;
    (void)matrices;
    (void)n;
    (void)nrhs;
    (void)lda;
    (void)ldb;
    (void)matrix_stride;
    (void)result_stride;
    (void)pivot_stride;
    TP_THROW(NotImplementedError,
             "ldl_solve on this GPU backend: the solver library provides no "
             "symmetric solve routine; use the CPU path instead");
#else
    size_t device_workspace_bytes = 0;
    size_t host_workspace_bytes = 0;
    CUSOLVER_CHECK(cusolverDnXsytrs_bufferSize(
        handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, ldl_cuda_dtype<scalar_t>(),
        matrix_data, lda, pivot_data, ldl_cuda_dtype<scalar_t>(), result_data,
        ldb, &device_workspace_bytes, &host_workspace_bytes));

    Tensor device_workspace = Tensor::empty(
        {static_cast<int64_t>(std::max<size_t>(device_workspace_bytes, 1))},
        DType::UInt8, result.device());
    std::vector<uint8_t> host_workspace(
        std::max<size_t>(host_workspace_bytes, 1));
    Tensor info = Tensor::zeros({1}, DType::Int32, result.device());

    // guarantee when a preceding factor/copy was queued on this stream.
    const cudaError_t sync_error =
        cudaStreamSynchronize(getCurrentCUDAStream().stream());
    if (sync_error != cudaSuccess) {
        TP_THROW(RuntimeError, "ldl_solve CUDA stream synchronization: ",
                 cudaGetErrorString(sync_error));
    }
    for (int64_t i = 0; i < matrices; ++i) {
        CUSOLVER_CHECK(cusolverDnXsytrs(
            handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,
            ldl_cuda_dtype<scalar_t>(), matrix_data + i * matrix_stride, lda,
            pivot_data + i * pivot_stride, ldl_cuda_dtype<scalar_t>(),
            result_data + i * result_stride, ldb,
            device_workspace.data_ptr(), device_workspace_bytes,
            host_workspace.data(), host_workspace_bytes,
            info.data_ptr<int32_t>()));
    }
#endif
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> ldl_factor_ex_impl(
        const Tensor& input, bool check_errors) {
    const auto batch = batch_shape_of(input);
    const int64_t n = input.size(-1);
    Tensor lower = input.tril(0);
    Tensor ld = clone_batched_column_major(lower);
    Tensor pivots = Tensor::zeros(append_matrix_shape(batch, n, 1),
                                  DType::Int32, input.device())
                        .squeeze(-1);
    Tensor info = Tensor::zeros(batch, DType::Int32, input.device());
    if (input.numel() != 0) {
        apply_ldl_factor<scalar_t>(ld, pivots, info);
    }
    if (check_errors) {
        check_factor_info(info, "linalg.ldl_factor_ex", input.dim() == 2);
    }
    return {ld.contiguous(), pivots, info};
}

void validate_pivots(const Tensor& pivots, int64_t n) {
    if (pivots.dtype() == DType::Bool || !isIntegralType(pivots.dtype(), false)) {
        TP_THROW(TypeError, "linalg.ldl_solve: pivots must be integral");
    }
    Tensor host = pivots.to(Device(DeviceType::CPU), DType::Int64).contiguous();
    const auto* data = host.data_ptr<int64_t>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        const int64_t pivot = data[i];
        const int64_t magnitude = pivot < 0 ? -pivot : pivot;
        if (magnitude < 1 || magnitude > n) {
            TP_THROW(RuntimeError,
                     "linalg.ldl_solve: pivots must satisfy |pivot| >= 1 "
                     "and |pivot| <= matrix size");
        }
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> linalg_ldl_factor_ex_cuda(
        const Tensor& input, bool /*hermitian*/, bool check_errors) {
    const char* api = "linalg.ldl_factor_ex";
    check_square(input, api, "A");
    check_real_dtype(input, api);
    if (input.size(-1) == 0) {
        const auto batch = batch_shape_of(input);
        Tensor ld = input.clone();
        Tensor pivots = Tensor::empty(append_matrix_shape(batch, 0, 1),
                                      DType::Int32, input.device()).squeeze(-1);
        Tensor info = Tensor::zeros(batch, DType::Int32, input.device());
        return {ld, pivots, info};
    }
    if (input.dtype() == DType::Float32) {
        return ldl_factor_ex_impl<float>(input, check_errors);
    }
    return ldl_factor_ex_impl<double>(input, check_errors);
}

std::tuple<Tensor, Tensor> linalg_ldl_factor_cuda(
        const Tensor& input, bool hermitian) {
    auto result = linalg_ldl_factor_ex_cuda(input, hermitian, false);
    check_factor_info(std::get<2>(result), "linalg.ldl_factor", input.dim() == 2);
    return {std::get<0>(result), std::get<1>(result)};
}

Tensor linalg_ldl_solve_cuda(const Tensor& ld, const Tensor& pivots,
                             const Tensor& b, bool /*hermitian*/) {
    const char* api = "linalg.ldl_solve";
    check_square(ld, api, "LD");
    check_real_dtype(ld, api);
    if (b.dim() < 2) {
        TP_THROW(RuntimeError, api, ": B must have at least two dimensions");
    }
    if (b.size(-2) != ld.size(-2)) {
        TP_THROW(RuntimeError, api, ": B has an incompatible row dimension");
    }
    if (b.dtype() != ld.dtype()) {
        TP_THROW(TypeError, api, ": LD and B must have the same dtype");
    }
    const auto ld_batch = batch_shape_of(ld);
    const auto pivot_shape = static_cast<std::vector<int64_t>>(pivots.shape());
    std::vector<int64_t> expected_pivot_shape = ld_batch;
    expected_pivot_shape.push_back(ld.size(-2));
    if (pivot_shape != expected_pivot_shape) {
        TP_THROW(RuntimeError, api,
                 ": pivots shape must equal LD.shape[:-1]");
    }
    validate_pivots(pivots, ld.size(-2));

    const auto batch = broadcast_batch(ld, b);
    Tensor ld_work = clone_batched_column_major(expand_to_batch(ld, batch));
    Tensor result = clone_batched_column_major(expand_to_batch(b, batch));
    Tensor pivot_work = pivots;
    while (pivot_work.dim() < static_cast<int64_t>(batch.size()) + 1) {
        pivot_work = pivot_work.unsqueeze(0);
    }
    std::vector<int64_t> pivot_target_shape = batch;
    pivot_target_shape.push_back(ld.size(-2));
    pivot_work = pivot_work.expand(pivot_target_shape)
        .to(ld.device(), DType::Int64).contiguous();

    if (ld.dtype() == DType::Float32) {
        apply_ldl_solve<float>(ld_work, pivot_work, result);
    } else {
        apply_ldl_solve<double>(ld_work, pivot_work, result);
    }
    return result.contiguous();
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeLinalgLDL) {
    m.impl("linalg_ldl_factor", linalg_ldl_factor_cuda);
    m.impl("linalg_ldl_factor_ex", linalg_ldl_factor_ex_cuda);
    m.impl("linalg_ldl_solve", linalg_ldl_solve_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
