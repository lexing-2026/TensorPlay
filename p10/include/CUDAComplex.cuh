// CUDAComplex.cuh -- shared complex-elementwise building blocks for the
// CUDA backend.  Storage is the standard interleaved (re, im) layout; device
// math uses thrust::complex<T> (full __device__ overloads for exp/log/pow/
// trig), reinterpreting the raw buffers exactly like ATen's complex CUDA
// kernels.  Reduced complexes (ComplexHalf/BComplex32) are intentionally not
// wired here: torch's own CUDA coverage for chalf is minimal, and every
// call site rejects them with NotImplementedError next to the real dtypes.
#pragma once

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <cmath>

#include "CUDABroadcast.cuh"
#include "DType.h"
#include "Exception.h"
#include "Scalar.h"

namespace tensorplay {
namespace cuda {
namespace cplx {

using c64 = thrust::complex<float>;
using c128 = thrust::complex<double>;

struct RecipOp {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {
        return T(1) / z;
    }
};

inline constexpr float kInvSqrt2f = 0.70710678118654752f;
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

// ATen weak-scalar rules for tensor-scalar kernels: a wrapped complex scalar
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
                             const thrust::complex<T>* __restrict__ src,
                             thrust::complex<T>* __restrict__ dst, F f) {
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
        n, static_cast<const thrust::complex<T>*>(src),
        static_cast<thrust::complex<T>*>(dst), f);
}

// --- abs / angle: C -> R -----------------------------------------------------
template <typename T>
__global__ void abs_kernel(int64_t n,
                           const thrust::complex<T>* __restrict__ src,
                           T* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = abs(src[i]);
}

template <typename T>
__global__ void angle_kernel(int64_t n,
                             const thrust::complex<T>* __restrict__ src,
                             T* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = thrust::arg(src[i]);
}

template <typename T>
inline void launch_abs(int64_t n, const void* src, void* dst,
                       cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    abs_kernel<T><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const thrust::complex<T>*>(src),
        static_cast<T*>(dst));
}

template <typename T>
inline void launch_angle(int64_t n, const void* src, void* dst,
                         cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    angle_kernel<T><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const thrust::complex<T>*>(src),
        static_cast<T*>(dst));
}

// --- binary same-shape (+,-,*,/ with optional alpha on rhs) ------------------
template <typename T, typename F>
__global__ void binary_kernel(int64_t n,
                              const thrust::complex<T>* __restrict__ a,
                              const thrust::complex<T>* __restrict__ b,
                              thrust::complex<T>* __restrict__ y, F f) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = f(a[i], b[i]);
}

struct DivOp {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> a,
                                             thrust::complex<T> b) const {
        return a / b;
    }
};
struct MulOp {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> a,
                                             thrust::complex<T> b) const {
        return a * b;
    }
};
// add/sub carry the ATen alpha on the rhs operand.  Callers pass alpha in
// the opmath complex type (s2c<T>), so the member is complex, not bare T.
template <typename T>
struct AddAlphaOp {
    thrust::complex<T> alpha;
    __device__ thrust::complex<T> operator()(thrust::complex<T> a,
                                             thrust::complex<T> b) const {
        return a + alpha * b;
    }
};
template <typename T>
struct SubAlphaOp {
    thrust::complex<T> alpha;
    __device__ thrust::complex<T> operator()(thrust::complex<T> a,
                                             thrust::complex<T> b) const {
        return a - alpha * b;
    }
};
struct PowOp {
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> a,
                                             thrust::complex<T> b) const {
        return pow(a, b);
    }
};
struct EqOp {
    template <typename T>
    __device__ bool operator()(thrust::complex<T> a,
                               thrust::complex<T> b) const {
        return a == b;
    }
};
struct NeOp {
    template <typename T>
    __device__ bool operator()(thrust::complex<T> a,
                               thrust::complex<T> b) const {
        return a != b;
    }
};

template <typename T, typename F>
inline void launch_binary(int64_t n, const void* a, const void* b, void* y,
                          F f, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int64_t grid = (n + kBlock - 1) / kBlock;
    binary_kernel<T, F><<<dim3((unsigned)grid), dim3(kBlock), 0, stream>>>(
        n, static_cast<const thrust::complex<T>*>(a),
        static_cast<const thrust::complex<T>*>(b),
        static_cast<thrust::complex<T>*>(y), f);
}

// --- binary broadcast (TensorDesc driven, mirrors add_broadcast_kernel) ------
#define TP_CPLX_GRIDSTRIDE(i)                                                \
    int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;              \
    int64_t tp_cplx_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;   \
    (void)tp_cplx_stride;

template <typename T, typename F>
__global__ void binary_broadcast_kernel(
        int64_t n,
        const thrust::complex<T>* __restrict__ a, TensorDesc a_desc,
        const thrust::complex<T>* __restrict__ b, TensorDesc b_desc,
        thrust::complex<T>* __restrict__ y, TensorDesc y_desc, F f) {
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
            n, static_cast<const thrust::complex<T>*>(a), a_desc,
            static_cast<const thrust::complex<T>*>(b), b_desc,
            static_cast<thrust::complex<T>*>(y), y_desc, f);
}

// --- tensor-scalar kernels ----------------------------------------------------
// add: y = x + alpha * other   (alpha defaults to identity via MulOp path)
template <typename T>
__global__ void add_scalar_kernel_impl(
        int64_t n, const thrust::complex<T>* __restrict__ a,
        thrust::complex<T> other, thrust::complex<T> alpha,
        thrust::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + alpha * other;
}
template <typename T>
__global__ void sub_scalar_kernel_impl(
        int64_t n, const thrust::complex<T>* __restrict__ a,
        thrust::complex<T> other, thrust::complex<T> alpha,
        thrust::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] - alpha * other;
}
template <typename T>
__global__ void mul_scalar_kernel_impl(
        int64_t n, const thrust::complex<T>* __restrict__ a,
        thrust::complex<T> other, thrust::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * other;
}
template <typename T>
__global__ void div_scalar_kernel_impl(
        int64_t n, const thrust::complex<T>* __restrict__ a,
        thrust::complex<T> other, thrust::complex<T>* __restrict__ y) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] / other;
}

template <typename T>
__global__ void scale_complex_kernel(
        int64_t n, const thrust::complex<T>* __restrict__ src,
        thrust::complex<T> factor, thrust::complex<T>* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] * factor;
}

inline dim3 default_grid(int64_t n) {
    return dim3((unsigned)((n + 255) / 256));
}
inline dim3 default_block() { return dim3(256); }

// Scalar -> thrust complex at a given width (real scalars get zero imag).
inline c64 to_c64(const Scalar& s) {
    return s.isComplex() ? c64(s.to<std::complex<float>>())
                         : c64(s.to<float>());
}
inline c128 to_c128(const Scalar& s) {
    return s.isComplex() ? c128(s.to<std::complex<double>>())
                         : c128(s.to<double>());
}

}  // namespace cplx
}  // namespace cuda
}  // namespace tensorplay
