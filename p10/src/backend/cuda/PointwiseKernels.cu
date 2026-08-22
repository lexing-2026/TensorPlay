#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDABroadcast.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace tensorplay {
namespace cuda {

// --- Utils ---
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// ATen alignment: grid-stride loops (cuda::detail::elementwise_kernel) so any
// grid size is correct and huge tensors don't overflow gridDim.x.
template <typename T, typename Func>
__global__ void unary_kernel_cuda_impl(int64_t n, const T* input, T* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = func(input[i]);
    }
}

// ATen alignment: vectorized fast path — each thread processes 4 elements via
// float4/double4 loads when the pointers are 16B aligned (mirrors
// vectorize_elementwise_kernel<4>). Falls back to scalar loop otherwise.
// ATen alignment: vectorized fast path — each thread processes 4 elements via
// aligned wide loads (mirrors vectorize_elementwise_kernel<4>).
template <typename T, int VecSize>
struct alignas(VecSize * sizeof(T)) VecPack { T v[VecSize]; };

template <typename T, int VecSize, typename Func>
__global__ void unary_vectorized_kernel_cuda_impl(int64_t n, const T* input, T* output, Func func) {
    int64_t vec_n = n / VecSize;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < vec_n; i += stride) {
        VecPack<T, VecSize> in = *reinterpret_cast<const VecPack<T, VecSize>*>(input + i * VecSize);
        VecPack<T, VecSize> out;
        #pragma unroll
        for (int v = 0; v < VecSize; ++v) out.v[v] = func(in.v[v]);
        *reinterpret_cast<VecPack<T, VecSize>*>(output + i * VecSize) = out;
    }
    for (int64_t j = vec_n * VecSize + i; j < n; j += stride) {
        output[j] = func(input[j]);
    }
}

template <typename T, typename Func>
__global__ void binary_kernel_cuda_impl(int64_t n, const T* a, const T* b, T* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = func(a[i], b[i]);
    }
}

template <typename T, int VecSize, typename Func>
__global__ void binary_vectorized_kernel_cuda_impl(int64_t n, const T* a, const T* b, T* output, Func func) {
    int64_t vec_n = n / VecSize;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < vec_n; i += stride) {
        VecPack<T, VecSize> va = *reinterpret_cast<const VecPack<T, VecSize>*>(a + i * VecSize);
        VecPack<T, VecSize> vb = *reinterpret_cast<const VecPack<T, VecSize>*>(b + i * VecSize);
        VecPack<T, VecSize> vo;
        #pragma unroll
        for (int v = 0; v < VecSize; ++v) vo.v[v] = func(va.v[v], vb.v[v]);
        *reinterpret_cast<VecPack<T, VecSize>*>(output + i * VecSize) = vo;
    }
    for (int64_t j = vec_n * VecSize + i; j < n; j += stride) {
        output[j] = func(a[j], b[j]);
    }
}

// ATen alignment: reduced floating types (Half/BFloat16) compute in float32
// (opmath_t). Load -> convert -> op -> convert back, all in one kernel.
template <typename T, typename Func>
__global__ void unary_reduced_float_kernel_cuda_impl(int64_t n, const T* input, T* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = static_cast<T>(func(static_cast<float>(input[i])));
    }
}

// --- Dispatchers ---

// ATen alignment: elementwise launch config mirrors ATen's
// elementwise_kernel: 128 threads/block, 4 elems/thread when vectorized,
// grid capped at a few blocks per SM.
inline int device_sms() {
    static int sms = []() {
        int dev = 0, count = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev);
        return count > 0 ? count : 1;
    }();
    return sms;
}

inline void get_elementwise_config(int64_t n, bool vectorized, dim3& grid, dim3& block) {
    block.x = 128;
    int64_t per_thread = vectorized ? 4 : 8;
    int64_t want = (n + block.x * per_thread - 1) / (block.x * per_thread);
    int64_t cap = static_cast<int64_t>(device_sms()) * 4;
    grid.x = static_cast<unsigned>(want < 1 ? 1 : (want > cap ? cap : want));
}

inline bool ptr_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

template <typename T, typename Func>
void launch_unary(int64_t n, const T* in, T* out, Func func) {
    dim3 grid, block;
    if ((n % 4 == 0) && ptr_aligned16(in) && ptr_aligned16(out)) {
        get_elementwise_config(n, true, grid, block);
        unary_vectorized_kernel_cuda_impl<T, 4><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, in, out, func);
    } else {
        get_elementwise_config(n, false, grid, block);
        unary_kernel_cuda_impl<T><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, in, out, func);
    }
}

template <typename T, typename Func>
void launch_binary(int64_t n, const T* a, const T* b, T* out, Func func) {
    dim3 grid, block;
    if ((n % 4 == 0) && ptr_aligned16(a) && ptr_aligned16(b) && ptr_aligned16(out)) {
        get_elementwise_config(n, true, grid, block);
        binary_vectorized_kernel_cuda_impl<T, 4><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a, b, out, func);
    } else {
        get_elementwise_config(n, false, grid, block);
        binary_kernel_cuda_impl<T><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a, b, out, func);
    }
}

// Vectorized variant of unary_reduced_float_kernel_cuda_impl: same
// load -> convert to float32 -> op -> convert back flow, 4 elements per thread.
template <typename T, int VecSize, typename Func>
__global__ void unary_reduced_float_vectorized_kernel_cuda_impl(int64_t n, const T* input, T* output, Func func) {
    int64_t vec_n = n / VecSize;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < vec_n; i += stride) {
        VecPack<T, VecSize> in = *reinterpret_cast<const VecPack<T, VecSize>*>(input + i * VecSize);
        VecPack<T, VecSize> out;
#pragma unroll
        for (int v = 0; v < VecSize; ++v)
            out.v[v] = static_cast<T>(func(static_cast<float>(in.v[v])));
        *reinterpret_cast<VecPack<T, VecSize>*>(output + i * VecSize) = out;
    }
    for (int64_t j = vec_n * VecSize + i; j < n; j += stride) {
        output[j] = static_cast<T>(func(static_cast<float>(input[j])));
    }
}

template <typename T, typename Func>
void launch_unary_reduced_float(int64_t n, const T* in, T* out, Func func) {
    dim3 grid, block;
    if ((n % 4 == 0) && ptr_aligned16(in) && ptr_aligned16(out)) {
        get_elementwise_config(n, true, grid, block);
        unary_reduced_float_vectorized_kernel_cuda_impl<T, 4><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, in, out, func);
    } else {
        get_elementwise_config(n, false, grid, block);
        unary_reduced_float_kernel_cuda_impl<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, in, out, func);
    }
}

// Generic Unary Dispatcher
template<typename Func>
Tensor unary_op_kernel(const Tensor& self, Func func) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // For now, assume contiguous. TODO: Handle non-contiguous via collapse or strides
    Tensor self_contig = self.contiguous();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        unary_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), func); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "CUDA unary op: Unsupported dtype");
    }
    #undef OP_CASE
    
    CUDA_CHECK(cudaGetLastError());
    return result;
}

struct AbsFunctor { template<typename T> __device__ T operator()(T x) const { return x >= T(0) ? x : -x; } };
struct NegFunctor { template<typename T> __device__ T operator()(T x) const { return -x; } };
struct SquareFunctor { template<typename T> __device__ T operator()(T x) const { return x * x; } };
struct SignFunctor { 
    template<typename T> __device__ T operator()(T x) const { 
        if (x > T(0)) return static_cast<T>(1);
        if (x < T(0)) return static_cast<T>(-1);
        return static_cast<T>(0);
    } 
};

// Revised unary_op_kernel to use Functor with templated operator
template<typename Functor>
Tensor unary_op_kernel_v2(const Tensor& self, Functor functor) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        launch_unary<ctype>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), functor); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "CUDA unary op: Unsupported dtype");
    }
    #undef OP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor abs_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, AbsFunctor()); }
Tensor neg_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, NegFunctor()); }
Tensor square_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, SquareFunctor()); }
Tensor sign_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, SignFunctor()); }

// Float ops need simpler dispatch since we cast to float/double
template<typename Functor>
Tensor unary_float_op_kernel_v2(const Tensor& self, Functor functor) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();
    
    if (out_dtype == DType::Float16 || out_dtype == DType::BFloat16) {
        // ATen alignment: compute in float32 (opmath_t), single fused kernel.
        #define REDUCED_FLOAT_CASE(ctype, name) \
        case DType::name: { \
            launch_unary_reduced_float<ctype>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), functor); \
            break; \
        }
        switch (self.dtype()) {
            REDUCED_FLOAT_CASE(Half, Float16)
            REDUCED_FLOAT_CASE(BFloat16, BFloat16)
            default: TP_THROW(TypeError, "CUDA unary float op: Unsupported dtype");
        }
        #undef REDUCED_FLOAT_CASE
    } else if (out_dtype == DType::Float32) {
        Tensor in = (self.dtype() == DType::Float32) ? self_contig : self_contig.to(DType::Float32);
        launch_unary<float>(n, in.data_ptr<float>(), result.data_ptr<float>(), functor);
    } else if (out_dtype == DType::Float64) {
        Tensor in = (self.dtype() == DType::Float64) ? self_contig : self_contig.to(DType::Float64);
        launch_unary<double>(n, in.data_ptr<double>(), result.data_ptr<double>(), functor);
    } else {
        TP_THROW(TypeError, "CUDA unary float op: Unsupported output dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ATen alignment: all functors compute in scalar_t (T-typed literals) so
// float32 tensors never fall into slow double-precision device math.
struct ExpFunctor { template<typename T> __device__ T operator()(T x) const { return exp(x); } };
struct Expm1Functor { template<typename T> __device__ T operator()(T x) const { return expm1(x); } };
struct ErfFunctor { template<typename T> __device__ T operator()(T x) const { return erf(x); } };
struct ErfcFunctor { template<typename T> __device__ T operator()(T x) const { return erfc(x); } };
struct LogFunctor { template<typename T> __device__ T operator()(T x) const { return log(x); } };
struct Log10Functor { template<typename T> __device__ T operator()(T x) const { return log10(x); } };
struct Log1pFunctor { template<typename T> __device__ T operator()(T x) const { return log1p(x); } };
struct Log2Functor { template<typename T> __device__ T operator()(T x) const { return log2(x); } };
struct LgammaFunctor { template<typename T> __device__ T operator()(T x) const { return lgamma(x); } };
struct SqrtFunctor { template<typename T> __device__ T operator()(T x) const { return sqrt(x); } };
struct RsqrtFunctor { template<typename T> __device__ T operator()(T x) const { return rsqrt(x); } };
struct SinFunctor { template<typename T> __device__ T operator()(T x) const { return sin(x); } };
struct CosFunctor { template<typename T> __device__ T operator()(T x) const { return cos(x); } };
struct TanhFunctor { template<typename T> __device__ T operator()(T x) const { return tanh(x); } };
struct SigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
    }
};
// ATen alignment: relu == clamp_min(0), NaN propagates.
struct ReluFunctor { template<typename T> __device__ T operator()(T x) const { return x < T(0) ? T(0) : x; } };
struct GeluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        const T kAlpha = static_cast<T>(0.70710678118654752440);
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + erf(x * kAlpha));
    }
};
struct SiluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        return x / (static_cast<T>(1) + exp(-x));
    }
};

Tensor exp_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ExpFunctor()); }
Tensor expm1_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, Expm1Functor()); }
Tensor erf_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ErfFunctor()); }
Tensor erfc_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ErfcFunctor()); }
Tensor log_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, LogFunctor()); }
Tensor log10_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, Log10Functor()); }
Tensor log1p_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, Log1pFunctor()); }
Tensor log2_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, Log2Functor()); }
Tensor lgamma_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, LgammaFunctor()); }
Tensor sqrt_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SqrtFunctor()); }
Tensor rsqrt_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, RsqrtFunctor()); }
Tensor sin_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SinFunctor()); }
Tensor cos_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, CosFunctor()); }
Tensor tanh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, TanhFunctor()); }
Tensor sigmoid_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SigmoidFunctor()); }
Tensor relu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ReluFunctor()); }
Tensor gelu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, GeluFunctor()); }
Tensor silu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SiluFunctor()); }

// ---------------------------------------------------------------------------
// Activations (CUDA).  Formulas ported from ATen:
//   aten/src/ATen/native/cuda/ActivationGeluKernel.cu
//     (GeluCUDAKernelImpl / GeluBackwardCUDAKernelImpl)
//   aten/src/ATen/native/cuda/ActivationHardswishKernel.cu
//   aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu
//   aten/src/ATen/native/cuda/ActivationLeakyReluKernel.cu
//   aten/src/ATen/native/cuda/ActivationEluKernel.cu
//   aten/src/ATen/native/cuda/ActivationMishKernel.cu
//   aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu
//   aten/src/ATen/native/cpu/Activation.cpp (hardtanh / hardtanh_backward)
// Reduced-precision inputs compute in float32 opmath like ATen.
// ---------------------------------------------------------------------------

template<typename Functor>
__global__ void launch_activation_backward_reduced_float(int64_t n,
    const Half* dy, const Half* x, Half* out, Functor f) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<Half>(f(static_cast<float>(dy[i]), static_cast<float>(x[i])));
}
template<typename Functor>
__global__ void launch_activation_backward_reduced_float(int64_t n,
    const BFloat16* dy, const BFloat16* x, BFloat16* out, Functor f) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<BFloat16>(f(static_cast<float>(dy[i]), static_cast<float>(x[i])));
}

template<typename Functor>
Tensor activation_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Functor functor) {
    if (grad_output.shape() != self.shape()) TP_THROW(RuntimeError, "CUDA activation backward: shape mismatch");
    DType out_dtype = grad_output.dtype();
    if (!isFloatingType(out_dtype)) TP_THROW(TypeError, "CUDA activation backward: expected floating point dtype");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor grad_contig = grad_output.contiguous();
    Tensor self_contig = self.contiguous();

    #define ACT_BWD_REDUCED_CASE(ctype, name) \
    case DType::name: { \
        launch_activation_backward_reduced_float<<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
            n, grad_contig.data_ptr<ctype>(), self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), functor); \
        break; \
    }
    switch (out_dtype) {
        ACT_BWD_REDUCED_CASE(Half, Float16)
        ACT_BWD_REDUCED_CASE(BFloat16, BFloat16)
        case DType::Float32: {
            launch_binary<float>(n, grad_contig.data_ptr<float>(), self_contig.data_ptr<float>(),
                                 result.data_ptr<float>(), functor);
            break;
        }
        case DType::Float64: {
            launch_binary<double>(n, grad_contig.data_ptr<double>(), self_contig.data_ptr<double>(),
                                  result.data_ptr<double>(), functor);
            break;
        }
        default: TP_THROW(TypeError, "CUDA activation backward: Unsupported dtype");
    }
    #undef ACT_BWD_REDUCED_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// ATen ActivationGeluKernel.cu: kBeta = M_SQRT2 * M_2_SQRTPI * 0.5; kKappa = 0.044715
struct GeluTanhFunctor {
    template<typename T> __device__ T operator()(T x) const {
        const T kBeta = static_cast<T>(1.41421356237309504880) * static_cast<T>(1.12837916709551257390) * static_cast<T>(0.5);
        const T kKappa = static_cast<T>(0.044715);
        T x_cube = x * x * x;
        T inner = kBeta * (x + kKappa * x_cube);
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + tanh(inner));
    }
};
struct HardtanhFunctor {
    double min_val_, max_val_;
    HardtanhFunctor(double lo, double hi) : min_val_(lo), max_val_(hi) {}
    template<typename T> __device__ T operator()(T x) const {
        // ATen cpu/Activation.cpp hardtanh: std::min(std::max(a, min_val), max_val)
        T lo = static_cast<T>(min_val_), hi = static_cast<T>(max_val_);
        return x < lo ? lo : (x > hi ? hi : x);
    }
};
struct HardtanhBackwardFunctor {
    double min_val_, max_val_;
    HardtanhBackwardFunctor(double lo, double hi) : min_val_(lo), max_val_(hi) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen cpu/Activation.cpp: (self <= min || self >= max) ? 0 : grad
        return (x <= static_cast<T>(min_val_) || x >= static_cast<T>(max_val_)) ? static_cast<T>(0) : dy;
    }
};
struct HardswishFunctor {
    template<typename T> __device__ T operator()(T x) const {
        // ATen ActivationHardswishKernel.cu: x * clamp(x + 3, 0, 6) / 6
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return x * v / static_cast<T>(6);
    }
};
struct HardswishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen Activation.h hardswish_backward
        return x <= static_cast<T>(-3) ? static_cast<T>(0)
             : x >= static_cast<T>(3)  ? dy
             : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct HardsigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        // ATen ActivationHardsigmoidKernel.cu: clamp(x + 3, 0, 6) / 6
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return v / static_cast<T>(6);
    }
};
struct HardsigmoidBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen Activation.h hardsigmoid_backward
        return (x <= static_cast<T>(-3) || x >= static_cast<T>(3)) ? static_cast<T>(0)
                                                                   : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct LeakyReluFunctor {
    double negative_slope_;
    LeakyReluFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T x) const {
        // ATen ActivationLeakyReluKernel.cu: x > 0 ? x : negative_slope * x
        return x > static_cast<T>(0) ? x : static_cast<T>(negative_slope_) * x;
    }
};
struct LeakyReluBackwardFunctor {
    double negative_slope_;
    LeakyReluBackwardFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen Activation.h leaky_relu_backward
        return x > static_cast<T>(0) ? dy : static_cast<T>(negative_slope_) * dy;
    }
};
struct EluFunctor {
    double negcoef_, poscoef_, negiptcoef_;
    EluFunctor(double alpha, double scale, double input_scale)
        : negcoef_(alpha * scale), poscoef_(scale), negiptcoef_(input_scale) {}
    template<typename T> __device__ T operator()(T a) const {
        // ATen cpu/Elu.h get_scalar_elu_elementwise_func:
        //   a < 0 ? expm1(a*input_scale)*negcoef : a*poscoef
        return a < static_cast<T>(0)
            ? expm1(a * static_cast<T>(negiptcoef_)) * static_cast<T>(negcoef_)
            : a * static_cast<T>(poscoef_);
    }
};
struct EluBackwardFunctor {
    double negcoef_, poscoef_, negiptcoef_;
    bool is_result_;
    EluBackwardFunctor(double alpha, double scale, double input_scale, bool is_result)
        : negcoef_(alpha * scale), poscoef_(scale), negiptcoef_(input_scale), is_result_(is_result) {}
    template<typename T> __device__ T operator()(T dy, T b) const {
        // ATen cpu/Activation.cpp elu_backward_kernel:
        //   is_result: b <= 0 ? dy*negiptcoef*(b+negcoef) : dy*poscoef
        //   else:      b <= 0 ? dy*negiptcoef*negcoef*exp(b*negiptcoef) : dy*poscoef
        return b <= static_cast<T>(0)
            ? (is_result_
                  ? dy * static_cast<T>(negiptcoef_) * (b + static_cast<T>(negcoef_))
                  : dy * static_cast<T>(negiptcoef_) * static_cast<T>(negcoef_) * exp(b * static_cast<T>(negiptcoef_)))
            : dy * static_cast<T>(poscoef_);
    }
};
struct MishFunctor {
    template<typename T> __device__ T operator()(T x) const {
        // ATen ActivationMishKernel.cu: mish(x) = x * tanh(softplus(x))
        T sp = log1p(exp(x));
        return x * tanh(sp);
    }
};
struct MishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen ActivationMishKernel.cu MishBackwardCUDAKernelImpl
        T sp = log1p(exp(x));
        T tanh_sp = tanh(sp);
        T sech2 = static_cast<T>(1) - tanh_sp * tanh_sp;
        T gsp = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
        return dy * (tanh_sp + x * sech2 * gsp);
    }
};
struct SeluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        // ATen Activation.h selu: lambda_ = 1.0507009873554804; alpha_ = 1.6732632423543772
        constexpr double lambda_ = 1.0507009873554804934193349852946;
        constexpr double alpha_ = 1.6732632423543772848170429916717;
        return x > static_cast<T>(0) ? static_cast<T>(lambda_) * x
                                     : static_cast<T>(alpha_ * lambda_) * expm1(x);
    }
};
struct CeluFunctor {
    double alpha_;
    CeluFunctor(double a) : alpha_(a) {}
    template<typename T> __device__ T operator()(T x) const {
        // ATen Activation.h celu: max(0,x) + min(0, alpha * expm1(x / alpha))
        return x > static_cast<T>(0) ? x : static_cast<T>(alpha_) * expm1(x / static_cast<T>(alpha_));
    }
};
struct SoftplusFunctor {
    double beta_, threshold_;
    SoftplusFunctor(double beta, double threshold) : beta_(beta), threshold_(threshold) {}
    template<typename T> __device__ T operator()(T a) const {
        // ATen ActivationSoftplusKernel.cu:
        //   beta*a > threshold ? a : log1p(exp(beta*a)) / beta
        T beta_in = static_cast<T>(beta_);
        return a * beta_in > static_cast<T>(threshold_)
            ? a
            : log1p(exp(a * beta_in)) / beta_in;
    }
};
struct SoftplusBackwardFunctor {
    double beta_, threshold_;
    SoftplusBackwardFunctor(double beta, double threshold) : beta_(beta), threshold_(threshold) {}
    template<typename T> __device__ T operator()(T dy, T a) const {
        // ATen ActivationSoftplusKernel.cu:
        //   beta*a > threshold ? dy : dy * sigmoid(beta*a)
        T beta_in = static_cast<T>(beta_);
        return a * beta_in > static_cast<T>(threshold_)
            ? dy
            : dy / (static_cast<T>(1) + exp(-a * beta_in));
    }
};

struct GeluBackwardNoneFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen ActivationGeluKernel.cu ('none'):
        //   kAlpha = M_SQRT1_2; kBeta = M_2_SQRTPI*M_SQRT1_2*0.5
        //   cdf = 0.5*(1+erf(x*kAlpha)); pdf = kBeta*exp(-0.5*x*x)
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440);
        constexpr T kBeta = static_cast<T>(1.12837916709551257390) * static_cast<T>(0.70710678118654752440) * static_cast<T>(0.5);
        T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + erf(x * kAlpha));
        T pdf = kBeta * exp(x * x * static_cast<T>(-0.5));
        return dy * (cdf + x * pdf);
    }
};
struct GeluBackwardTanhFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // ATen ActivationGeluKernel.cu ('Tanh')
        constexpr T kBeta = static_cast<T>(1.41421356237309504880) * static_cast<T>(1.12837916709551257390) * static_cast<T>(0.5);
        constexpr T kKappa = static_cast<T>(0.044715);
        T x_sq = x * x;
        T x_cube = x_sq * x;
        T inner = kBeta * (x + kKappa * x_cube);
        T tanh_inner = tanh(inner);
        T left = static_cast<T>(0.5) * x;
        T right = static_cast<T>(1) + tanh_inner;
        T left_derivative = static_cast<T>(0.5) * right;
        T tanh_derivative = static_cast<T>(1) - tanh_inner * tanh_inner;
        T inner_derivative = kBeta * (static_cast<T>(1) + static_cast<T>(3) * kKappa * x_sq);
        T right_derivative = left * tanh_derivative * inner_derivative;
        return dy * (left_derivative + right_derivative);
    }
};

Tensor gelu_kernel_cuda_v2(const Tensor& self, const std::string& approximate) {
    // ATen ActivationGeluKernel.cu GeluCUDAKernelImpl
    if (approximate == "tanh") return unary_float_op_kernel_v2(self, GeluTanhFunctor());
    else if (approximate != "none") TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    return unary_float_op_kernel_v2(self, GeluFunctor());
}
Tensor gelu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    // ATen ActivationGeluKernel.cu GeluBackwardCUDAKernelImpl
    if (approximate == "tanh") return activation_backward_kernel_cuda(grad_output, self, GeluBackwardTanhFunctor());
    else if (approximate != "none") TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    return activation_backward_kernel_cuda(grad_output, self, GeluBackwardNoneFunctor());
}
Tensor hardtanh_kernel_cuda(const Tensor& self, Scalar min_val, Scalar max_val) {
    return unary_float_op_kernel_v2(self, HardtanhFunctor(min_val.toDouble(), max_val.toDouble()));
}
Tensor hardtanh_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
    return activation_backward_kernel_cuda(grad_output, self, HardtanhBackwardFunctor(min_val.toDouble(), max_val.toDouble()));
}
Tensor relu6_kernel_cuda(const Tensor& self) { return hardtanh_kernel_cuda(self, Scalar(0.0), Scalar(6.0)); }
Tensor hardswish_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, HardswishFunctor()); }
Tensor hardswish_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, HardswishBackwardFunctor());
}
Tensor hardsigmoid_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, HardsigmoidFunctor()); }
Tensor hardsigmoid_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, HardsigmoidBackwardFunctor());
}
Tensor leaky_relu_kernel_cuda(const Tensor& self, Scalar negative_slope) {
    return unary_float_op_kernel_v2(self, LeakyReluFunctor(negative_slope.toDouble()));
}
Tensor leaky_relu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar negative_slope, bool self_is_result) {
    (void)self_is_result;
    return activation_backward_kernel_cuda(grad_output, self, LeakyReluBackwardFunctor(negative_slope.toDouble()));
}
Tensor elu_kernel_cuda(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
    return unary_float_op_kernel_v2(self, EluFunctor(alpha.toDouble(), scale.toDouble(), input_scale.toDouble()));
}
Tensor elu_backward_kernel_cuda(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, const Tensor& self_or_result) {
    return activation_backward_kernel_cuda(grad_output, self_or_result,
        EluBackwardFunctor(alpha.toDouble(), scale.toDouble(), input_scale.toDouble(), is_result));
}
Tensor mish_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, MishFunctor()); }
Tensor mish_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, MishBackwardFunctor());
}
Tensor selu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SeluFunctor()); }
Tensor celu_kernel_cuda(const Tensor& self, Scalar alpha) { return unary_float_op_kernel_v2(self, CeluFunctor(alpha.toDouble())); }
Tensor softplus_kernel_cuda(const Tensor& self, Scalar beta, Scalar threshold) {
    return unary_float_op_kernel_v2(self, SoftplusFunctor(beta.toDouble(), threshold.toDouble()));
}
Tensor softplus_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar beta, Scalar threshold) {
    return activation_backward_kernel_cuda(grad_output, self, SoftplusBackwardFunctor(beta.toDouble(), threshold.toDouble()));
}

struct AcosFunctor { template<typename T> __device__ T operator()(T x) const { return acos(x); } };
struct AcoshFunctor { template<typename T> __device__ T operator()(T x) const { return acosh(x); } };
struct AsinFunctor { template<typename T> __device__ T operator()(T x) const { return asin(x); } };
struct AsinhFunctor { template<typename T> __device__ T operator()(T x) const { return asinh(x); } };
struct AtanFunctor { template<typename T> __device__ T operator()(T x) const { return atan(x); } };
struct AtanhFunctor { template<typename T> __device__ T operator()(T x) const { return atanh(x); } };
struct CeilFunctor { template<typename T> __device__ T operator()(T x) const { return ceil(x); } };
struct CoshFunctor { template<typename T> __device__ T operator()(T x) const { return cosh(x); } };
struct FloorFunctor { template<typename T> __device__ T operator()(T x) const { return floor(x); } };
struct RoundFunctor { template<typename T> __device__ T operator()(T x) const { return rint(x); } }; // rint matches round better in CUDA
struct SinhFunctor { template<typename T> __device__ T operator()(T x) const { return sinh(x); } };
struct TanFunctor { template<typename T> __device__ T operator()(T x) const { return tan(x); } };
struct TruncFunctor {
    template<typename T> __device__ T operator()(T x) const {
        // ::trunc/::truncf are the CUDA device functions; unqualified trunc
        // resolves to constexpr host std::trunc via ADL.
        if constexpr (std::is_same_v<T, float>) return ::truncf(x);
        else return ::trunc(static_cast<double>(x));
    }
};
struct FracFunctor {
    template<typename T> __device__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) return x - ::truncf(x);
        else return x - static_cast<T>(::trunc(static_cast<double>(x)));
    }
};

Tensor acos_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AcosFunctor()); }
Tensor acosh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AcoshFunctor()); }
Tensor asin_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AsinFunctor()); }
Tensor asinh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AsinhFunctor()); }
Tensor atan_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AtanFunctor()); }
Tensor atanh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, AtanhFunctor()); }
Tensor ceil_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, CeilFunctor()); }
Tensor cosh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, CoshFunctor()); }
Tensor floor_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, FloorFunctor()); }
Tensor round_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, RoundFunctor()); }
Tensor sinh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SinhFunctor()); }
Tensor tan_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, TanFunctor()); }
Tensor trunc_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, TruncFunctor()); }
Tensor frac_kernel_cuda(const Tensor& self) {
    if (isIntegralType(self.dtype())) {
        TP_THROW(NotImplementedError, "frac is not implemented for integral tensors");
    }
    return unary_float_op_kernel_v2(self, FracFunctor());
}

// --- Comparison ---
// ATen alignment: comparisons broadcast and promote to a common dtype like
// torch (TensorIterator). Fast path for same-shape contiguous inputs;
// generic strided path reuses the TensorDesc broadcast mapping.
template <typename T, typename Func>
__global__ void comparison_kernel_cuda_impl(int64_t n, const T* a, const T* b, bool* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = func(a[i], b[i]);
    }
}

template <typename T, typename Func>
__global__ void comparison_broadcast_kernel_cuda_impl(int64_t n,
                                                      const T* a, TensorDesc a_desc,
                                                      const T* b, TensorDesc b_desc,
                                                      bool* output, TensorDesc y_desc, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = func(a[get_offset(i, a_desc, y_desc)], b[get_offset(i, b_desc, y_desc)]);
    }
}

template <typename T, typename Func>
__global__ void comparison_scalar_kernel_cuda_impl(int64_t n, const T* a, T b, bool* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = func(a[i], b);
    }
}

// ATen alignment: wrapped Python scalars participate weakly in promotion.
static DType result_type_with_scalar_cuda(const Tensor& t, const Scalar& s) {
    DType td = t.dtype();
    if (s.dtype() == DType::Bool) return td;
    if (isFloatingType(s.dtype())) {
        if (isFloatingType(td)) return td;
        return DType::Float32;
    }
    return td;
}

template<typename Functor>
Tensor comparison_op_kernel(const Tensor& self, const Tensor& other, Functor functor) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    int64_t n = result.numel();
    if (n == 0) return result;

    dim3 block(256);
    dim3 grid((n + 255) / 256);

    Tensor a = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor b = (other.dtype() == common_dtype) ? other : other.to(common_dtype);
    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();

    #define COMP_CASE(ctype, name) \
    case DType::name: { \
        bool same_shape = (a_c.dim() == static_cast<int64_t>(out_shape.size())) && \
                          (b_c.dim() == static_cast<int64_t>(out_shape.size())); \
        if (same_shape) { \
            for (int64_t d = 0; d < static_cast<int64_t>(out_shape.size()); ++d) { \
                if (a_c.size(d) != out_shape[d] || b_c.size(d) != out_shape[d]) { same_shape = false; break; } \
            } \
        } \
        if (same_shape) { \
            comparison_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a_c.data_ptr<ctype>(), b_c.data_ptr<ctype>(), result.data_ptr<bool>(), functor); \
        } else { \
            TensorDesc a_desc = make_desc(a_c, out_shape.size()); \
            TensorDesc b_desc = make_desc(b_c, out_shape.size()); \
            TensorDesc y_desc = make_desc_from_shape(out_shape); \
            comparison_broadcast_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, a_c.data_ptr<ctype>(), a_desc, b_c.data_ptr<ctype>(), b_desc, result.data_ptr<bool>(), y_desc, functor); \
        } \
        break; \
    }
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_CASE)
        default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
    }
    #undef COMP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template<typename Functor>
Tensor comparison_scalar_op_kernel(const Tensor& self, Scalar other, Functor functor) {
    // ATen alignment: weak scalar promotion before comparing
    DType common = result_type_with_scalar_cuda(self, other);
    Tensor in = (self.dtype() == common) ? self : self.to(common);
    Scalar o = other;
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(in.shape()), DType::Bool, self.device());
    int64_t n = in.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    #define COMP_SCALAR_CASE(ctype, name) \
    case DType::name: { \
        comparison_scalar_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, in.data_ptr<ctype>(), o.to<ctype>(), result.data_ptr<bool>(), functor); \
        break; \
    }
    switch (common) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_SCALAR_CASE)
        default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
    }
    #undef COMP_SCALAR_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

struct EqFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a == b; } };
struct NeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a != b; } };
struct LtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a < b; } };
struct LeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a <= b; } };
struct GtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a > b; } };
struct GeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a >= b; } };

Tensor eq_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, EqFunctor()); }
Tensor ne_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, NeFunctor()); }
Tensor lt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, LtFunctor()); }
Tensor le_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, LeFunctor()); }
Tensor gt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, GtFunctor()); }
Tensor ge_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, GeFunctor()); }

Tensor eq_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, EqFunctor()); }
Tensor ne_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, NeFunctor()); }
Tensor lt_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, LtFunctor()); }
Tensor le_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, LeFunctor()); }
Tensor gt_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, GtFunctor()); }
Tensor ge_scalar_kernel_cuda(const Tensor& self, Scalar other) { return comparison_scalar_op_kernel(self, other, GeFunctor()); }

template <typename T>
__global__ void where_broadcast_kernel_cuda_impl(
    int64_t n, const bool* condition, TensorDesc condition_desc,
    const T* self, TensorDesc self_desc,
    const T* other, TensorDesc other_desc,
    T* output, TensorDesc output_desc) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int64_t condition_offset = get_offset(i, condition_desc, output_desc);
        const int64_t self_offset = get_offset(i, self_desc, output_desc);
        const int64_t other_offset = get_offset(i, other_desc, output_desc);
        output[i] = condition[condition_offset] ? self[self_offset] : other[other_offset];
    }
}

template <typename T, bool Maximum>
__global__ void maximum_minimum_broadcast_kernel_cuda_impl(
    int64_t n, const T* self, TensorDesc self_desc,
    const T* other, TensorDesc other_desc,
    T* output, TensorDesc output_desc) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const T a = self[get_offset(i, self_desc, output_desc)];
        const T b = other[get_offset(i, other_desc, output_desc)];
        if constexpr (Maximum) {
            output[i] = a < b ? b : a;
        } else {
            output[i] = a < b ? a : b;
        }
    }
}

Tensor where_cuda(const Tensor& condition, const Tensor& self, const Tensor& other) {
    if (condition.dtype() != DType::Bool) {
        TP_THROW(TypeError, "where condition must be a boolean tensor");
    }
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(condition.shape()),
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_casted = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor other_casted = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorDesc condition_desc = make_desc(condition, out_shape.size());
    TensorDesc self_desc = make_desc(self_casted, out_shape.size());
    TensorDesc other_desc = make_desc(other_casted, out_shape.size());
    TensorDesc output_desc = make_desc(result, out_shape.size());

    #define WHERE_CASE(ctype, name) \
        case DType::name: \
            where_broadcast_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
                n, condition.data_ptr<bool>(), condition_desc, \
                self_casted.data_ptr<ctype>(), self_desc, other_casted.data_ptr<ctype>(), \
                other_desc, result.data_ptr<ctype>(), output_desc); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(WHERE_CASE)
        default: TP_THROW(NotImplementedError, "CUDA where: unsupported dtype");
    }
    #undef WHERE_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor where_scalar_self_cuda(const Tensor& condition, Scalar self, const Tensor& other) {
    DType common_dtype = result_type(self, other.dtype());
    return where_cuda(condition, Tensor::full({}, self, common_dtype, other.device()), other);
}

Tensor where_scalar_other_cuda(const Tensor& condition, const Tensor& self, Scalar other) {
    DType common_dtype = result_type(other, self.dtype());
    return where_cuda(condition, self, Tensor::full({}, other, common_dtype, self.device()));
}

static DType where_scalar_dtype(const Scalar& self, const Scalar& other) {
    if (self.isComplex() || other.isComplex()) {
        return promoteTypes(self.dtype(), other.dtype());
    }
    if (self.isFloatingPoint() || other.isFloatingPoint()) {
        return self.dtype() == DType::Float64 || other.dtype() == DType::Float64
            ? DType::Float64 : DType::Float32;
    }
    return DType::Int64;
}

Tensor where_scalar_scalar_cuda(const Tensor& condition, Scalar self, Scalar other) {
    DType common_dtype = where_scalar_dtype(self, other);
    return where_cuda(
        condition,
        Tensor::full({}, self, common_dtype, condition.device()),
        Tensor::full({}, other, common_dtype, condition.device()));
}

template <bool Maximum>
Tensor maximum_minimum_cuda_impl(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "maximum/minimum is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc b_desc = make_desc(b, out_shape.size());
    TensorDesc result_desc = make_desc(result, out_shape.size());

    #define MAXMIN_CASE(ctype, name) \
        case DType::name: \
            maximum_minimum_broadcast_kernel_cuda_impl<ctype, Maximum><<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
                n, a.data_ptr<ctype>(), a_desc, b.data_ptr<ctype>(), b_desc, \
                result.data_ptr<ctype>(), result_desc); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MAXMIN_CASE)
        default: TP_THROW(NotImplementedError, "CUDA maximum/minimum: unsupported dtype");
    }
    #undef MAXMIN_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor maximum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<true>(self, other);
}

Tensor minimum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<false>(self, other);
}


// Clamp
template <typename T>
__global__ void clamp_kernel_cuda_impl(int64_t n, const T* input, T* output, T min_val, T max_val, bool has_min, bool has_max) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = input[i];
        if (has_min && val < min_val) val = min_val;
        if (has_max && val > max_val) val = max_val;
        output[i] = val;
    }
}

Tensor clamp_kernel_cuda(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();
    
    #define CLAMP_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = min.has_value() ? min->to<ctype>() : ctype(0); \
        ctype max_val = max.has_value() ? max->to<ctype>() : ctype(0); \
        clamp_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), min_val, max_val, min.has_value(), max.has_value()); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_CASE)
        default: TP_THROW(TypeError, "CUDA clamp: Unsupported dtype");
    }
    #undef CLAMP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// Clamp Backward
template <typename T>
__global__ void clamp_backward_kernel_cuda_impl(int64_t n, const T* grad, const T* input, T* output, T min_val, T max_val, bool has_min, bool has_max) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = input[i];
        if ((has_min && val < min_val) || (has_max && val > max_val)) {
            output[i] = 0;
        } else {
            output[i] = grad[i];
        }
    }
}

Tensor clamp_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();
    Tensor grad_contig = grad_output.contiguous();
    
    #define CLAMP_BW_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = min.has_value() ? min->to<ctype>() : ctype(0); \
        ctype max_val = max.has_value() ? max->to<ctype>() : ctype(0); \
        clamp_backward_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, grad_contig.data_ptr<ctype>(), self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), min_val, max_val, min.has_value(), max.has_value()); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_BW_CASE)
        default: TP_THROW(TypeError, "CUDA clamp_backward: Unsupported dtype");
    }
    #undef CLAMP_BW_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// --- Binary Ops ---

template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor) {
    if (self.shape() != other.shape()) TP_THROW(RuntimeError, "CUDA binary op: broadcasting not supported");
    
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();
    Tensor other_contig = other.contiguous();
    
    if (out_dtype == DType::Float32) {
        Tensor a = (self.dtype() == DType::Float32) ? self_contig : self_contig.to(DType::Float32);
        Tensor b = (other.dtype() == DType::Float32) ? other_contig : other_contig.to(DType::Float32);
        launch_binary<float>(n, a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), functor);
    } else if (out_dtype == DType::Float64) {
        Tensor a = (self.dtype() == DType::Float64) ? self_contig : self_contig.to(DType::Float64);
        Tensor b = (other.dtype() == DType::Float64) ? other_contig : other_contig.to(DType::Float64);
        launch_binary<double>(n, a.data_ptr<double>(), b.data_ptr<double>(), result.data_ptr<double>(), functor);
    }
    
    CUDA_CHECK(cudaGetLastError());
    return result;
}

struct PowFunctor { template<typename T> __device__ T operator()(T a, T b) const { return pow(a, b); } };
// ATen alignment: keep the exponent in double so Float64 tensors don't lose precision
struct PowScalarFunctor {
    double exponent;
    PowScalarFunctor(double e) : exponent(e) {}
    template<typename T> __device__ T operator()(T x) const { return pow(x, static_cast<T>(exponent)); }
};
struct Atan2Functor { template<typename T> __device__ T operator()(T a, T b) const { return atan2(a, b); } };

Tensor pow_kernel_cuda(const Tensor& self, const Tensor& other) { return binary_float_op_kernel_v2(self, other, PowFunctor()); }
Tensor pow_scalar_kernel_cuda(const Tensor& self, Scalar exponent) {
    // ATen alignment: integer base with negative integer exponent is rejected
    if (isIntegralType(self.dtype()) && !exponent.isFloatingPoint() && exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    return unary_float_op_kernel_v2(self, PowScalarFunctor(exponent.toDouble()));
}
Tensor atan2_kernel_cuda(const Tensor& self, const Tensor& other) { return binary_float_op_kernel_v2(self, other, Atan2Functor()); }

// --- Lerp ---
// ATen alignment: |w| < 0.5 uses s + w*(e-s), else e - (e-s)*(1-w)
// (numerically stable branch, see ATen native/cuda/Lerp.cu)
template <typename T>
__global__ void lerp_scalar_kernel_cuda_impl(int64_t n, const T* start, const T* end, T* output, T weight) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (std::abs(weight) < T(0.5))
            ? start[i] + weight * (end[i] - start[i])
            : end[i] - (end[i] - start[i]) * (static_cast<T>(1) - weight);
    }
}

template <typename T>
__global__ void lerp_tensor_kernel_cuda_impl(int64_t n, const T* start, const T* end, const T* weight, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T w = weight[i];
        output[i] = (std::abs(w) < T(0.5))
            ? start[i] + w * (end[i] - start[i])
            : end[i] - (end[i] - start[i]) * (static_cast<T>(1) - w);
    }
}

Tensor lerp_scalar_kernel_cuda(const Tensor& self, const Tensor& end, Scalar weight) {
    if (self.shape() != end.shape()) TP_THROW(RuntimeError, "CUDA lerp: broadcasting not supported");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    Tensor self_c = self.contiguous();
    Tensor end_c = end.contiguous();

    #define LERP_CASE(ctype, name) \
    case DType::name: { \
        lerp_scalar_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_c.data_ptr<ctype>(), end_c.data_ptr<ctype>(), result.data_ptr<ctype>(), weight.to<ctype>()); \
        break; \
    }
    switch (self.dtype()) {
        LERP_CASE(float, Float32)
        LERP_CASE(double, Float64)
        default: TP_THROW(NotImplementedError, "CUDA lerp: only float32/float64 supported");
    }
    #undef LERP_CASE
    return result;
}

Tensor lerp_tensor_kernel_cuda(const Tensor& self, const Tensor& end, const Tensor& weight) {
    if (self.shape() != end.shape() || self.shape() != weight.shape()) TP_THROW(RuntimeError, "CUDA lerp: broadcasting not supported");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    Tensor self_c = self.contiguous();
    Tensor end_c = end.contiguous();
    Tensor weight_c = weight.contiguous();

    #define LERPT_CASE(ctype, name) \
    case DType::name: { \
        lerp_tensor_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_c.data_ptr<ctype>(), end_c.data_ptr<ctype>(), weight_c.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break; \
    }
    switch (self.dtype()) {
        LERPT_CASE(float, Float32)
        LERPT_CASE(double, Float64)
        default: TP_THROW(NotImplementedError, "CUDA lerp: only float32/float64 supported");
    }
    #undef LERPT_CASE
    return result;
}

Tensor& lerp_scalar_inplace_kernel_cuda(Tensor& self, const Tensor& end, Scalar weight) {
    self.copy_(lerp_scalar_kernel_cuda(self, end, weight));
    return self;
}

Tensor& lerp_tensor_inplace_kernel_cuda(Tensor& self, const Tensor& end, const Tensor& weight) {
    self.copy_(lerp_tensor_kernel_cuda(self, end, weight));
    return self;
}

Tensor& abs_inplace_kernel_cuda(Tensor& self) {
    self.copy_(abs_kernel_cuda(self));
    return self;
}

Tensor& neg_inplace_kernel_cuda(Tensor& self) {
    self.copy_(neg_kernel_cuda(self));
    return self;
}

Tensor& sqrt_inplace_kernel_cuda(Tensor& self) {
    self.copy_(sqrt_kernel_cuda(self));
    return self;
}

Tensor& rsqrt_inplace_kernel_cuda(Tensor& self) {
    self.copy_(rsqrt_kernel_cuda(self));
    return self;
}

// --- Masked Select ---
template <typename T>
__global__ void count_mask_kernel(int64_t n, const bool* mask, int64_t* counter) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) {
        atomicAdd((unsigned long long*)counter, 1); // Use ULL for 64-bit atomic if supported, or cast to ULL. 
        // atomicAdd for int64 is supported on CC 6.0+. 
        // If not, use 32-bit counter or multiple passes. 
        // Assuming modern GPU.
    }
}

// Fallback for atomicAdd(int64_t*) on older devices or if ambiguous
__device__ void atomicAdd64(int64_t* address, int64_t val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + (unsigned long long)val);
    } while (assumed != old);
}

template <typename T>
__global__ void masked_select_kernel(int64_t n, const T* input, const bool* mask, T* output, int64_t* counter) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) {
        int64_t idx = atomicAdd((unsigned long long*)counter, 1);
        output[idx] = input[i];
    }
}

Tensor masked_select_kernel_cuda(const Tensor& self, const Tensor& mask) {
    if (self.shape() != mask.shape()) TP_THROW(RuntimeError, "CUDA masked_select: shapes must match");
    if (mask.dtype() != DType::Bool) TP_THROW(TypeError, "CUDA masked_select: mask must be bool");
    
    int64_t n = self.numel();
    if (n == 0) return Tensor::empty({0}, self.dtype(), self.device());
    
    Tensor self_c = self.contiguous();
    Tensor mask_c = mask.contiguous();
    
    // 1. Count elements
    Tensor counter({1}, DType::Int64, self.device());
    int64_t* d_counter = counter.data_ptr<int64_t>();
    auto stream = getCurrentCUDAStream();
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int64_t), stream.stream()));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // We can't use template for mask type, it's always bool.
    // But we need template for input type? No, count only needs mask.
    count_mask_kernel<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, mask_c.data_ptr<bool>(), d_counter); // Template arg unused but required if templated
    
    int64_t count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&count, d_counter, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream.stream()));
    stream.synchronize();
    
    // 2. Allocate output
    Tensor result = Tensor::empty({count}, self.dtype(), self.device());
    
    if (count > 0) {
        // Reset counter for indexing
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int64_t), stream.stream()));
        
        #define SEL_CASE(ctype, name) \
        case DType::name: { \
            masked_select_kernel<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_c.data_ptr<ctype>(), mask_c.data_ptr<bool>(), result.data_ptr<ctype>(), d_counter); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(SEL_CASE)
            default: TP_THROW(TypeError, "CUDA masked_select: Unsupported dtype");
        }
        #undef SEL_CASE
    }
    
    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PointwiseKernels) {
    m.impl("abs", abs_kernel_cuda);
    m.impl("neg", neg_kernel_cuda);
    m.impl("square", square_kernel_cuda);
    m.impl("sign", sign_kernel_cuda);
    
    m.impl("acos", acos_kernel_cuda);
    m.impl("acosh", acosh_kernel_cuda);
    m.impl("asin", asin_kernel_cuda);
    m.impl("asinh", asinh_kernel_cuda);
    m.impl("atan", atan_kernel_cuda);
    m.impl("atanh", atanh_kernel_cuda);
    m.impl("ceil", ceil_kernel_cuda);
    m.impl("cosh", cosh_kernel_cuda);
    m.impl("floor", floor_kernel_cuda);
    m.impl("round", round_kernel_cuda);
    m.impl("sinh", sinh_kernel_cuda);
    m.impl("tan", tan_kernel_cuda);
    
    m.impl("exp", exp_kernel_cuda);
    m.impl("expm1", expm1_kernel_cuda);
    m.impl("erf", erf_kernel_cuda);
    m.impl("erfc", erfc_kernel_cuda);
    m.impl("log", log_kernel_cuda);
    m.impl("log10", log10_kernel_cuda);
    m.impl("log1p", log1p_kernel_cuda);
    m.impl("log2", log2_kernel_cuda);
    m.impl("lgamma", lgamma_kernel_cuda);
    m.impl("sqrt", sqrt_kernel_cuda);
    m.impl("rsqrt", rsqrt_kernel_cuda);
    m.impl("sin", sin_kernel_cuda);
    m.impl("cos", cos_kernel_cuda);
    m.impl("tanh", tanh_kernel_cuda);
    m.impl("trunc", trunc_kernel_cuda);
    m.impl("frac", frac_kernel_cuda);
    
    m.impl("sigmoid", sigmoid_kernel_cuda);
    m.impl("relu", relu_kernel_cuda);
    m.impl("gelu", gelu_kernel_cuda_v2);
    m.impl("gelu_backward", gelu_backward_kernel_cuda);
    m.impl("silu", silu_kernel_cuda);
    // Activations — see the ATen citations above each functor.
    m.impl("hardtanh", hardtanh_kernel_cuda);
    m.impl("hardtanh_backward", hardtanh_backward_kernel_cuda);
    m.impl("relu6", relu6_kernel_cuda);
    m.impl("hardswish", hardswish_kernel_cuda);
    m.impl("hardswish_backward", hardswish_backward_kernel_cuda);
    m.impl("hardsigmoid", hardsigmoid_kernel_cuda);
    m.impl("hardsigmoid_backward", hardsigmoid_backward_kernel_cuda);
    m.impl("leaky_relu", leaky_relu_kernel_cuda);
    m.impl("leaky_relu_backward", leaky_relu_backward_kernel_cuda);
    m.impl("elu", elu_kernel_cuda);
    m.impl("elu_backward", elu_backward_kernel_cuda);
    m.impl("mish", mish_kernel_cuda);
    m.impl("mish_backward", mish_backward_kernel_cuda);
    m.impl("selu", selu_kernel_cuda);
    m.impl("celu", celu_kernel_cuda);
    m.impl("softplus", softplus_kernel_cuda);
    m.impl("softplus_backward", softplus_backward_kernel_cuda);
    
    m.impl("clamp", clamp_kernel_cuda);
    m.impl("clamp_backward", clamp_backward_kernel_cuda);
    
    m.impl("eq.Tensor", eq_kernel_cuda);
    m.impl("ne.Tensor", ne_kernel_cuda);
    m.impl("lt.Tensor", lt_kernel_cuda);
    m.impl("le.Tensor", le_kernel_cuda);
    m.impl("gt.Tensor", gt_kernel_cuda);
    m.impl("ge.Tensor", ge_kernel_cuda);
    
    m.impl("eq.Scalar", eq_scalar_kernel_cuda);
    m.impl("ne.Scalar", ne_scalar_kernel_cuda);
    m.impl("lt.Scalar", lt_scalar_kernel_cuda);
    m.impl("le.Scalar", le_scalar_kernel_cuda);
    m.impl("gt.Scalar", gt_scalar_kernel_cuda);
    m.impl("ge.Scalar", ge_scalar_kernel_cuda);

    m.impl("where.self", where_cuda);
    m.impl("where.ScalarSelf", where_scalar_self_cuda);
    m.impl("where.ScalarOther", where_scalar_other_cuda);
    m.impl("where.Scalar", where_scalar_scalar_cuda);
    m.impl("maximum", maximum_cuda);
    m.impl("minimum", minimum_cuda);
    
    m.impl("pow.Tensor_Tensor", pow_kernel_cuda);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel_cuda);
    m.impl("atan2", atan2_kernel_cuda);
    
    m.impl("lerp", lerp_scalar_kernel_cuda);
    m.impl("lerp.Tensor", lerp_tensor_kernel_cuda);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel_cuda);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel_cuda);
    m.impl("abs_", abs_inplace_kernel_cuda);
    m.impl("neg_", neg_inplace_kernel_cuda);
    m.impl("sqrt_", sqrt_inplace_kernel_cuda);
    m.impl("rsqrt_", rsqrt_inplace_kernel_cuda);
    m.impl("masked_select", masked_select_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
