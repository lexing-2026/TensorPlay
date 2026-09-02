// BLAS-level products outside the matmul family proper: addbmm, addmv,
// addr, vdot.  Native device implementations; the GEMV-shaped work runs
// through the shared cuBLAS GEMM wrapper (N = 1), addr is one elementwise
// pass, and vdot uses the conjugating/non-conjugating BLAS dot kernels.

#include "Tensor.h"
#include "TypePromotion.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "CudaGemm.h"
#include "Exception.h"
#include "Scalar.h"
#include "Utils.h"

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>

namespace tensorplay {
namespace cuda {

namespace {

#define CUBLAS_CHECK(condition)                                              \
    do {                                                                     \
        const cublasStatus_t _tp_cublas_status = (condition);                \
        if (_tp_cublas_status != CUBLAS_STATUS_SUCCESS) {                    \
            TP_THROW(RuntimeError,                                           \
                     "cuBLAS Error " +                                       \
                         std::to_string(static_cast<int>(_tp_cublas_status))); \
        }                                                                    \
    } while (0)

#define CUDA_CHECK(condition)                                                \
    do {                                                                     \
        const cudaError_t _tp_cuda_err = (condition);                        \
        if (_tp_cuda_err != cudaSuccess) {                                   \
            TP_THROW(RuntimeError,                                           \
                     "CUDA Error: " + std::string(cudaGetErrorString(        \
                         _tp_cuda_err)));                                    \
        }                                                                    \
    } while (0)

constexpr int kThreads = 256;

inline dim3 element_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
}

// out[i] = beta * self_b[i] + alpha * vec1[i/k] * vec2[i%k]; operands are
// pre-broadcast to the output shape in the accumulate dtype.
template <typename T>
__global__ void addr_kernel(int64_t total, int64_t k, double beta, double alpha,
                            const T* self_b, const T* v1, const T* v2, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < total; i += stride) {
        const int64_t r = i / k;
        const int64_t c = i % k;
        out[i] = static_cast<T>(beta * static_cast<double>(self_b[i]) +
                                alpha * static_cast<double>(v1[r]) *
                                    static_cast<double>(v2[c]));
    }
}

template <typename T>
void launch_addr(const Tensor& self_b, const Tensor& v1, const Tensor& v2,
                 Tensor& out, double beta, double alpha) {
    const int64_t m = out.size(0), k = out.size(1);
    const int64_t total = m * k;
    if (total == 0) return;
    addr_kernel<T><<<element_grid(total), kThreads, 0,
                    getCurrentCUDAStream().stream()>>>(
        total, k, beta, alpha,
        self_b.data_ptr<T>(), v1.data_ptr<T>(), v2.data_ptr<T>(),
        out.data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// ---------------------------------------------------------------------------
// addmv: beta * self + alpha * (mat @ vec)
// ---------------------------------------------------------------------------

Tensor addmv_cuda(const Tensor& self, const Tensor& mat, const Tensor& vec,
                  Scalar beta, Scalar alpha) {
    require_float(mat, "addmv");
    require_float(vec, "addmv");
    if (mat.dim() != 2) TP_THROW(RuntimeError, "addmv: mat must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "addmv: vec must be a vector");
    const int64_t m = mat.size(0), k = mat.size(1);
    if (vec.numel() != k)
        TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    const DType dt = promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    check_cublas_gemm_dtype(dt);
    const double alpha_v = alpha.toDouble();
    const double beta_v = beta.toDouble();

    Tensor mc = mat.dtype() == dt ? mat.contiguous() : mat.to(dt).contiguous();
    Tensor vc = vec.dtype() == dt ? vec.contiguous() : vec.to(dt).contiguous();
    // Seed y with the broadcast self; beta == 0 leaves y unread, so the seed
    // copy is skipped and the output stays uninitialized.
    Tensor result = beta_v != 0.0
        ? (self.dtype() == dt ? detail::contiguous_clone(self.expand({m}))
                              : detail::contiguous_clone(self.expand({m}).to(dt)))
        : Tensor::empty({m}, dt, mat.device());
    if (m == 0) return result;
    if (k == 0) {
        // Empty contraction: the product contributes nothing.
        if (beta_v == 0.0) result.zero_();
        return result;
    }
    Tensor v2d = vc.reshape({k, 1});
    Tensor y2d = result.reshape({m, 1});
    gemm_impl(mc, v2d, y2d, alpha_v, beta_v, nullptr);
    return result;
}

// ---------------------------------------------------------------------------
// addbmm: beta * self + alpha * sum_i batch1[i] @ batch2[i]
// ---------------------------------------------------------------------------

Tensor addbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                   Scalar beta, Scalar alpha) {
    require_float(batch1, "addbmm");
    require_float(batch2, "addbmm");
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Incompatible matrix sizes for bmm (",
                 batch1.size(1), "x", batch1.size(2), " and ",
                 batch2.size(1), "x", batch2.size(2), ")");
    }
    const int64_t b = batch1.size(0), n = batch1.size(1);
    const int64_t p = batch1.size(2), m = batch2.size(2);
    const DType dt = promoteTypes(batch1.dtype(), batch2.dtype());
    check_cublas_gemm_dtype(dt);
    const double beta_v = beta.toDouble();
    const double alpha_v = alpha.toDouble();

    Tensor result = dt == self.dtype()
        ? detail::contiguous_clone(self.expand({n, m}))
        : detail::contiguous_clone(self.expand({n, m}).to(dt));
    if (b == 0 || p == 0) {
        // No contraction happens; the seed (scaled by beta) is the answer.
        if (beta_v == 0.0) result.zero_();
        return result;
    }
    Tensor b1 = batch1.dtype() == dt ? batch1.contiguous() : batch1.to(dt).contiguous();
    Tensor b2 = batch2.dtype() == dt ? batch2.contiguous() : batch2.to(dt).contiguous();
    // Accumulating chain: the first GEMM carries beta, the rest fold into
    // the running sum with beta = 1 (one cuBLAS call per batch, no
    // intermediate (B, M, N) product).
    for (int64_t bi = 0; bi < b; ++bi) {
        gemm_impl(b1.select(0, bi), b2.select(0, bi), result,
                  alpha_v, bi == 0 ? beta_v : 1.0, nullptr);
    }
    return result;
}

// ---------------------------------------------------------------------------
// addr: beta * self + alpha * vec1 (outer) vec2
// ---------------------------------------------------------------------------

Tensor addr_cuda(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                 Scalar beta, Scalar alpha) {
    require_float(vec1, "addr");
    require_float(vec2, "addr");
    const int64_t m = vec1.numel(), k = vec2.numel();
    const DType dt = promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    const DType cdt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    const Tensor v1 = vec1.contiguous().to(cdt);
    const Tensor v2 = vec2.contiguous().to(cdt);
    const Tensor self_b = self.expand({m, k}).contiguous().to(cdt);
    Tensor out = Tensor::empty({m, k}, dt, self.device());
    const double beta_v = beta.toDouble(), alpha_v = alpha.toDouble();
    if (cdt == DType::Float64) {
        launch_addr<double>(self_b, v1, v2, out, beta_v, alpha_v);
    } else {
        launch_addr<float>(self_b, v1, v2, out, beta_v, alpha_v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// vdot: conj(a) . b over the flattened operands
// ---------------------------------------------------------------------------

Tensor vdot_cuda(const Tensor& a_in, const Tensor& b_in) {
    Tensor a = a_in.contiguous().reshape({a_in.numel()});
    Tensor b = b_in.contiguous().reshape({b_in.numel()});
    if (a.numel() != b.numel()) TP_THROW(RuntimeError, "vdot: sizes don't match");
    const DType dt = a_in.dtype();
    Tensor bmatch = b.dtype() == dt ? b : b.to(dt).contiguous();
    Tensor result = Tensor::empty({}, dt, a.device());
    const int64_t n = a.numel();
    if (n == 0) return result.zero_();

    cublasHandle_t handle = CUDAContext::getCublasHandle();
    switch (dt) {
        case DType::Float32:
            CUBLAS_CHECK(cublasSdot(handle, static_cast<int>(n),
                                    a.data_ptr<float>(), 1,
                                    bmatch.data_ptr<float>(), 1,
                                    result.data_ptr<float>()));
            return result;
        case DType::Float64:
            CUBLAS_CHECK(cublasDdot(handle, static_cast<int>(n),
                                    a.data_ptr<double>(), 1,
                                    bmatch.data_ptr<double>(), 1,
                                    result.data_ptr<double>()));
            return result;
        case DType::Float16:
        case DType::BFloat16: {
            // fp32 accumulation contract, native storage for the scalar.
            const cudaDataType_t cuda_type =
                dt == DType::Float16 ? CUDA_R_16F : CUDA_R_16BF;
            CUBLAS_CHECK(cublasDotEx(handle, static_cast<int>(n),
                                     a.data_ptr(), cuda_type, 1,
                                     bmatch.data_ptr(), cuda_type, 1,
                                     result.data_ptr(), cuda_type, CUDA_R_32F));
            return result;
        }
        case DType::ComplexFloat: {
            // Conjugating dot: sum conj(a[i]) * b[i].
#if defined(USE_ROCM)
            // The HIP entry takes the result through an out-parameter
            // instead of returning it by value.
            hipComplex out_c;
            CUBLAS_CHECK(hipblasCdotc(
                handle, static_cast<int>(n),
                static_cast<const hipComplex*>(static_cast<const void*>(a.data_ptr<std::complex<float>>())), 1,
                static_cast<const hipComplex*>(static_cast<const void*>(bmatch.data_ptr<std::complex<float>>())), 1,
                &out_c));
            result.data_ptr<std::complex<float>>()[0] =
                std::complex<float>(out_c.x, out_c.y);
#else
            cuComplex out = cublasCdotc(
                handle, static_cast<int>(n),
                static_cast<const cuComplex*>(static_cast<const void*>(a.data_ptr<std::complex<float>>())), 1,
                static_cast<const cuComplex*>(static_cast<const void*>(bmatch.data_ptr<std::complex<float>>())), 1);
            result.data_ptr<std::complex<float>>()[0] =
                std::complex<float>(out.x, out.y);
#endif
            return result;
        }
        case DType::ComplexDouble: {
#if defined(USE_ROCM)
            hipDoubleComplex out_z;
            CUBLAS_CHECK(hipblasZdotc(
                handle, static_cast<int>(n),
                static_cast<const hipDoubleComplex*>(static_cast<const void*>(a.data_ptr<std::complex<double>>())), 1,
                static_cast<const hipDoubleComplex*>(static_cast<const void*>(bmatch.data_ptr<std::complex<double>>())), 1,
                &out_z));
            result.data_ptr<std::complex<double>>()[0] =
                std::complex<double>(out_z.x, out_z.y);
#else
            cuDoubleComplex out = cublasZdotc(
                handle, static_cast<int>(n),
                static_cast<const cuDoubleComplex*>(static_cast<const void*>(a.data_ptr<std::complex<double>>())), 1,
                static_cast<const cuDoubleComplex*>(static_cast<const void*>(bmatch.data_ptr<std::complex<double>>())), 1);
            result.data_ptr<std::complex<double>>()[0] =
                std::complex<double>(out.x, out.y);
#endif
            return result;
        }
        default:
            TP_THROW(NotImplementedError, "vdot: unsupported dtype on CUDA");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Blas) {
    m.impl("addbmm", addbmm_cuda);
    m.impl("addmv", addmv_cuda);
    m.impl("addr", addr_cuda);
    m.impl("vdot", vdot_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
