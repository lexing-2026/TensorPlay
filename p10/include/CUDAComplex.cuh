// CUDAComplex.cuh -- shared complex-elementwise building blocks for the
// CUDA backend.  Storage is the standard interleaved (re, im) layout; device
// math uses the local complex scalar and its device overloads.  Reduced
// complexes (ComplexHalf/BComplex32) are intentionally not
// call site rejects them with NotImplementedError next to the real dtypes.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "CUDABroadcast.cuh"
#include "Complex.h"
#include "DType.h"
#include "Exception.h"
#include "Scalar.h"

namespace tensorplay {
namespace cuda {
namespace cplx {

using c64 = tensorplay::complex<float>;
using c128 = tensorplay::complex<double>;

struct RecipOp {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> z) const {
        return T(1) / z;
    }
};

inline constexpr float kInvSqrt2f = 0.70710678118654752f;
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

// keeps an already-complex tensor unchanged and widens a REAL tensor to its
// own complex width (float64 -> complex128, everything else -> complex64).
inline DType scalar_result_dtype(DType self_dt, const Scalar& other,
                                 const Scalar* alpha = nullptr) {
    const bool alpha_cplx = alpha && alpha->isComplex();
    if (isComplexType(self_dt)) return self_dt;
    if (other.isComplex() || alpha_cplx) {
        return isFloatingType(self_dt) ? toComplexType(self_dt)
                                       : DType::ComplexFloat;
    }
    return self_dt;
}

// --- elementwise unary over interleaved storage -----------------------------
template <typename T, typename F>
__global__ void unary_kernel(int64_t n,
                             const tensorplay::complex<T>* __restrict__ src,
                             tensorplay::complex<T>* __restrict__ dst, F f) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = f(src[i]);
}

template <typename T, typename F>
inline void launch_unary(int64_t n, const void* src, void* dst, F f,
                         cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    unary_kernel<T, F><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const tensorplay::complex<T>*>(src),
        static_cast<tensorplay::complex<T>*>(dst), f);
}

// --- abs / angle: C -> R -----------------------------------------------------
template <typename T>
__global__ void abs_kernel(int64_t n,
                           const tensorplay::complex<T>* __restrict__ src,
                           T* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = std::abs(src[i]);
}

template <typename T>
__global__ void angle_kernel(int64_t n,
                             const tensorplay::complex<T>* __restrict__ src,
                             T* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = std::arg(src[i]);
}

template <typename T>
inline void launch_abs(int64_t n, const void* src, void* dst,
                       cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    abs_kernel<T><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const tensorplay::complex<T>*>(src),
        static_cast<T*>(dst));
}

template <typename T>
inline void launch_angle(int64_t n, const void* src, void* dst,
                         cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    angle_kernel<T><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const tensorplay::complex<T>*>(src),
        static_cast<T*>(dst));
}

// --- binary same-shape (+,-,*,/ with optional alpha on rhs) ------------------
template <typename T, typename F>
__global__ void binary_kernel(int64_t n,
                              const tensorplay::complex<T>* __restrict__ a,
                              const tensorplay::complex<T>* __restrict__ b,
                              tensorplay::complex<T>* __restrict__ y, F f) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = f(a[i], b[i]);
}

struct DivOp {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> a,
                                                 tensorplay::complex<T> b) const {
        return a / b;
    }
};
struct MulOp {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> a,
                                                 tensorplay::complex<T> b) const {
        return a * b;
    }
};
// the opmath complex type (s2c<T>), so the member is complex, not bare T.
template <typename T>
struct AddAlphaOp {
    tensorplay::complex<T> alpha;
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> a,
                                                 tensorplay::complex<T> b) const {
        return a + alpha * b;
    }
};
template <typename T>
struct SubAlphaOp {
    tensorplay::complex<T> alpha;
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> a,
                                                 tensorplay::complex<T> b) const {
        return a - alpha * b;
    }
};
struct PowOp {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> a,
                                                 tensorplay::complex<T> b) const {
        return tensorplay::pow(a, b);
    }
};
struct EqOp {
    template <typename T>
    __device__ bool operator()(tensorplay::complex<T> a,
                               tensorplay::complex<T> b) const {
        return a == b;
    }
};
struct NeOp {
    template <typename T>
    __device__ bool operator()(tensorplay::complex<T> a,
                               tensorplay::complex<T> b) const {
        return a != b;
    }
};

template <typename T, typename F>
inline void launch_binary(int64_t n, const void* a, const void* b, void* y,
                          F f, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    binary_kernel<T, F><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const tensorplay::complex<T>*>(a),
        static_cast<const tensorplay::complex<T>*>(b),
        static_cast<tensorplay::complex<T>*>(y), f);
}

// --- binary broadcast (TensorDesc driven) -----------------------------------
#define TP_CPLX_GRIDSTRIDE(i)                                                \
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;              \
    int64_t tp_cplx_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;   \
    (void)tp_cplx_stride;

template <typename T, typename F>
__global__ void binary_broadcast_kernel(
        int64_t n,
        const tensorplay::complex<T>* __restrict__ a, TensorDesc a_desc,
        const tensorplay::complex<T>* __restrict__ b, TensorDesc b_desc,
        tensorplay::complex<T>* __restrict__ y, TensorDesc y_desc, F f) {
    TP_CPLX_GRIDSTRIDE(i) {
        const int64_t a_off = get_offset(i, a_desc, y_desc);
        const int64_t b_off = get_offset(i, b_desc, y_desc);
        y[i] = f(a[a_off], b[b_off]);
    }
}

template <typename T, typename F>
inline void launch_binary_broadcast(
        int64_t n, const void* a, const TensorDesc& a_desc, const void* b,
        const TensorDesc& b_desc, void* y, const TensorDesc& y_desc, F f,
        cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    binary_broadcast_kernel<T, F>
        <<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
            n, static_cast<const tensorplay::complex<T>*>(a), a_desc,
            static_cast<const tensorplay::complex<T>*>(b), b_desc,
            static_cast<tensorplay::complex<T>*>(y), y_desc, f);
}

// --- tensor-scalar kernels ----------------------------------------------------
// add: y = x + alpha * other   (alpha defaults to identity via MulOp path)
template <typename T>
__global__ void add_scalar_kernel_impl(
        int64_t n, const tensorplay::complex<T>* __restrict__ a,
        tensorplay::complex<T> other, tensorplay::complex<T> alpha,
        tensorplay::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + alpha * other;
}
template <typename T>
__global__ void sub_scalar_kernel_impl(
        int64_t n, const tensorplay::complex<T>* __restrict__ a,
        tensorplay::complex<T> other, tensorplay::complex<T> alpha,
        tensorplay::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] - alpha * other;
}
template <typename T>
__global__ void mul_scalar_kernel_impl(
        int64_t n, const tensorplay::complex<T>* __restrict__ a,
        tensorplay::complex<T> other, tensorplay::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * other;
}
template <typename T>
__global__ void div_scalar_kernel_impl(
        int64_t n, const tensorplay::complex<T>* __restrict__ a,
        tensorplay::complex<T> other, tensorplay::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] / other;
}

template <typename T>
__global__ void scale_complex_kernel(
        int64_t n, const tensorplay::complex<T>* __restrict__ src,
        tensorplay::complex<T> factor, tensorplay::complex<T>* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] * factor;
}

inline dim3 default_grid(int64_t n) {
    return dim3((unsigned)((n + 255) / 256));
}
inline dim3 default_block() { return dim3(256); }

// Scalar -> local complex at a given width (real scalars get zero imag).
inline c64 to_c64(const Scalar& s) {
    return s.to<c64>();
}
inline c128 to_c128(const Scalar& s) {
    return s.to<c128>();
}

}  // namespace cplx
}  // namespace cuda
}  // namespace tensorplay
