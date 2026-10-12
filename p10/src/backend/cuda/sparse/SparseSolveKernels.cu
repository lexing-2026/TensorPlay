// CUDA _spsolve: sparse CSR linear solve through cuSOLVER's sparse QR
// (cusolverSp*csrlsvqr).  Left solve computes A \ B; right solve computes
// B / A by solving A^T X^T = B^T on the transposed (CSC) matrix.

#include "SparseKernels.cuh"
#include "SparseKernels.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"

#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

#define TP_CUSOLVERSP_CHECK(condition)                                              \
    do {                                                                           \
        const cusolverStatus_t _tp_cs = (condition);                               \
        if (_tp_cs != CUSOLVER_STATUS_SUCCESS) {                                   \
            TP_THROW(RuntimeError, "cuSOLVER error ",                              \
                     std::to_string(static_cast<int>(_tp_cs)));                       \
        }                                                                          \
    } while (0)

#define TP_CUSPARSE_CHECK(condition)                                                \
    do {                                                                           \
        const cusparseStatus_t _tp_cs = (condition);                          \
        if (_tp_cs != CUSPARSE_STATUS_SUCCESS) { TP_THROW(RuntimeError, "cuSPARSE error ", static_cast<int>(_tp_cs)); }                                                               \
    } while (0)

namespace tensorplay {
namespace cuda {

namespace {

struct CsrSolverParts {
    cusparseMatDescr_t descr = nullptr;
    Tensor crow32;
    Tensor col32;
    Tensor values;
    int nnz = 0;
};

// Builds the int32 CSR parts and the cusparse descriptor for `A`.  When
// `transpose` is true, produces the CSR of A^T (i.e. the CSC of A) so the
// same solver routine can handle right-side solves.
CsrSolverParts prepare_csr(const Tensor& A, bool transpose) {
    Tensor crow;
    Tensor col;
    Tensor values = A._values().contiguous();
    if (!transpose) {
        crow = A._crow_indices().contiguous();
        col = A._col_indices().contiguous();
    } else {
        // CSR -> COO -> swap axes -> CSR of the transposed matrix.
        Tensor coo = Tensor::_convert_indices_from_csr_to_coo(
            A._crow_indices(), A._col_indices(), /*out_int32=*/false,
            /*transpose=*/false);
        Tensor row = coo.select(0, 0).contiguous();
        Tensor orig_col = coo.select(0, 1).contiguous();
        crow = Tensor::_convert_indices_from_coo_to_csr(
            orig_col, A.size(1), /*out_int32=*/false);
        col = row;
    }
    CsrSolverParts parts;
    parts.crow32 = crow.to(DType::Int32);
    parts.col32 = col.to(DType::Int32);
    parts.values = values;
    parts.nnz = static_cast<int>(A._nnz());
    TP_CUSPARSE_CHECK(cusparseCreateMatDescr(&parts.descr));
    cusparseSetMatType(parts.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(parts.descr, CUSPARSE_INDEX_BASE_ZERO);
    return parts;
}

template <typename scalar_t>
struct SpLsvqr {
    // Selects the right-precision solver.  Complex values map to cuComplex/
    // cuDoubleComplex, which share layout with std::complex.
    static cusolverStatus_t call(cusolverSpHandle_t h, int n, int nnz,
                                   cusparseMatDescr_t d, const scalar_t* val,
                                   const int* row, const int* col,
                                   const scalar_t* b, scalar_t* x, double tol,
                                   int reorder, int* singularity);
};

template <>
struct SpLsvqr<float> {
    static cusolverStatus_t call(cusolverSpHandle_t h, int n, int nnz,
                                 cusparseMatDescr_t d, const float* val,
                                 const int* row, const int* col, const float* b,
                                 float* x, double tol, int reorder,
                                 int* singularity) {
        return cusolverSpScsrlsvqr(h, n, nnz, d, val, row, col, b, tol,
                                   reorder, x, singularity);
    }
};

template <>
struct SpLsvqr<double> {
    static cusolverStatus_t call(cusolverSpHandle_t h, int n, int nnz,
                                 cusparseMatDescr_t d, const double* val,
                                 const int* row, const int* col,
                                 const double* b, double* x, double tol,
                                 int reorder, int* singularity) {
        return cusolverSpDcsrlsvqr(h, n, nnz, d, val, row, col, b, tol,
                                   reorder, x, singularity);
    }
};

template <>
struct SpLsvqr<complex<float>> {
    static cusolverStatus_t call(cusolverSpHandle_t h, int n, int nnz,
                                 cusparseMatDescr_t d,
                                 const complex<float>* val, const int* row,
                                 const int* col, const complex<float>* b,
                                 complex<float>* x, double tol, int reorder,
                                 int* singularity) {
        return cusolverSpCcsrlsvqr(
            h, n, nnz, d, reinterpret_cast<const cuComplex*>(val), row, col,
            reinterpret_cast<const cuComplex*>(b), tol, reorder,
            reinterpret_cast<cuComplex*>(x), singularity);
    }
};

template <>
struct SpLsvqr<complex<double>> {
    static cusolverStatus_t call(cusolverSpHandle_t h, int n, int nnz,
                                cusparseMatDescr_t d,
                                const complex<double>* val, const int* row,
                                const int* col, const complex<double>* b,
                                complex<double>* x, double tol, int reorder,
                                int* singularity) {
        return cusolverSpZcsrlsvqr(
            h, n, nnz, d, reinterpret_cast<const cuDoubleComplex*>(val), row,
            col, reinterpret_cast<const cuDoubleComplex*>(b), tol, reorder,
            reinterpret_cast<cuDoubleComplex*>(x), singularity);
    }
};

} // namespace

Tensor _spsolve_cuda(const Tensor& A, const Tensor& B, bool left) {
    if (!A.is_sparse_csr() || A.dim() != 2 || A.size(0) != A.size(1)) {
        TP_THROW(RuntimeError,
                 "_spsolve(): expected a square 2-D sparse CSR matrix A");
    }
    if (B.is_sparse()) {
        TP_THROW(RuntimeError,
                 "_spsolve(): expected B to be a strided tensor");
    }
    if (B.dtype() != A.dtype()) {
        TP_THROW(TypeError,
                 "_spsolve(): A and B must share one dtype");
    }
    if (B.device() != A.device()) {
        TP_THROW(DeviceMismatchError,
                 "_spsolve(): A and B must be on the same device");
    }
    if (B.dim() != 1 && B.dim() != 2) {
        TP_THROW(RuntimeError,
                 "_spsolve(): B must be 1-D or 2-D");
    }
    const int64_t n = A.size(0);
    if ((left && B.size(0) != n) || (!left && B.dim() == 2 && B.size(1) != n) ||
        (!left && B.dim() == 1 && B.size(0) != n)) {
        TP_THROW(RuntimeError,
                 "_spsolve(): B has an incompatible dimension");
    }

    // Build the (possibly transposed) CSR parts.
    CsrSolverParts parts = prepare_csr(A, /*transpose=*/!left);
    const int nnz = parts.nnz;

    // RHS workspace: contiguous (nrhs, n) row-major, so RHS column j starts
    // at offset j*n (column-major view of the logical RHS matrix).
    const int64_t nrhs = B.dim() == 1 ? 1 : B.size(left ? 1 : 0);
    Tensor B_work = B.dim() == 1 ? B.contiguous().unsqueeze(1)
                               : B.t().contiguous();
    Tensor Y = Tensor::empty({nrhs, n}, A.dtype(), A.device());

    const auto handle = CUDAContext::getCusolverSpHandle();
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    TP_CUSOLVERSP_CHECK(cusolverSpSetStream(handle, stream));

    const double tol = (A.dtype() == DType::Float32 ||
                        A.dtype() == DType::ComplexFloat)
                           ? 1e-6
                           : 1e-12;
    const int reorder = 1;

    auto solve = [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        scalar_t* x_data = Y.data_ptr<scalar_t>();
        const scalar_t* b_data = B_work.const_data_ptr<scalar_t>();
        for (int64_t j = 0; j < nrhs; ++j) {
            int singularity = 0;
            TP_CUSOLVERSP_CHECK(SpLsvqr<scalar_t>::call(
                handle, static_cast<int>(n), nnz, parts.descr,
                parts.values.const_data_ptr<scalar_t>(),
                parts.crow32.data_ptr<int>(), parts.col32.data_ptr<int>(),
                b_data + j * n, x_data + j * n, tol, reorder, &singularity));
            if (singularity != -1) {
                TP_THROW(RuntimeError,
                         "_spsolve(): matrix is singular to working "
                         "precision");
            }
        }
    };
    switch (A.dtype()) {
        case DType::Float32: solve(CoalesceTypeTag<float>{}); break;
        case DType::Float64: solve(CoalesceTypeTag<double>{}); break;
        case DType::ComplexFloat: solve(CoalesceTypeTag<complex<float>>{}); break;
        case DType::ComplexDouble: solve(CoalesceTypeTag<complex<double>>{}); break;
        default: TP_THROW(TypeError, "sparse solve requires floating or complex floating values");
    }

    TP_CUSPARSE_CHECK(cusparseDestroyMatDescr(parts.descr));

    if (B.dim() == 1) {
        // A single right-hand side stays 1-D in the output.
        return Y.squeeze(0);
    }
    if (left) {
        // Y is (nrhs, n) = (k, n); the left solution is its transpose.
        return Y.t();
    }
    // Right solve: Y already holds X (nrhs, n) = (k, n).
    return Y;
}

} // namespace cuda
} // namespace tensorplay
