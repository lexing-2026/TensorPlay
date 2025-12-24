#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
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

template <typename T, typename Func>
__global__ void unary_kernel_cuda_impl(int64_t n, const T* input, T* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = func(input[i]);
    }
}

template <typename T, typename Func>
__global__ void binary_kernel_cuda_impl(int64_t n, const T* a, const T* b, T* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = func(a[i], b[i]);
    }
}

// --- Dispatchers ---

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
        unary_kernel_cuda_impl<ctype><<<grid, block>>>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), func); \
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

// Float Unary Dispatcher (promotes integers to float)
template<typename Func>
Tensor unary_float_op_kernel(const Tensor& self, Func func) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) {
        out_dtype = DType::Float32;
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor self_contig = self.contiguous();

    // Helper kernel for cast + op
    // We can't pass lambda with template args easily to __global__ if it captures, 
    // but here func is stateless usually. 
    // However, device lambda support requires --expt-extended-lambda.
    // To be safe and portable, we should use functors or specialized kernels.
    // For this demo, we assume nvcc supports extended lambdas (standard in modern CUDA).
    
    // We need explicit instantiation for types.
    
    if (out_dtype == DType::Float32) {
        if (self.dtype() == DType::Float32) {
             unary_kernel_cuda_impl<float><<<grid, block>>>(n, self_contig.data_ptr<float>(), result.data_ptr<float>(), func);
        } else {
             // Cast needed. For simplicity, convert input to float first? Or use specialized kernel.
             // Let's convert input for now to save kernel variants.
             Tensor self_float = self_contig.to(DType::Float32);
             unary_kernel_cuda_impl<float><<<grid, block>>>(n, self_float.data_ptr<float>(), result.data_ptr<float>(), func);
        }
    } else if (out_dtype == DType::Float64) {
         if (self.dtype() == DType::Float64) {
             unary_kernel_cuda_impl<double><<<grid, block>>>(n, self_contig.data_ptr<double>(), result.data_ptr<double>(), func);
         } else {
             Tensor self_double = self_contig.to(DType::Float64);
             unary_kernel_cuda_impl<double><<<grid, block>>>(n, self_double.data_ptr<double>(), result.data_ptr<double>(), func);
         }
    } else {
        TP_THROW(TypeError, "CUDA unary float op: Unsupported output dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}

// --- Functors for Kernels ---
// NVCC extended lambda support allows passing lambdas to kernels, 
// but defining functors is more robust across versions.

template<typename T> struct AbsOp { __device__ T operator()(T x) const { return x >= 0 ? x : -x; } };
template<typename T> struct NegOp { __device__ T operator()(T x) const { return -x; } };
template<typename T> struct SquareOp { __device__ T operator()(T x) const { return x * x; } };
template<typename T> struct SignOp { 
    __device__ T operator()(T x) const { 
        if (x > 0) return static_cast<T>(1);
        if (x < 0) return static_cast<T>(-1);
        return static_cast<T>(0);
    } 
};
template<typename T> struct FloorOp { __device__ T operator()(T x) const { return floorf((float)x); } }; // Simplified
template<typename T> struct CeilOp { __device__ T operator()(T x) const { return ceilf((float)x); } };
template<typename T> struct RoundOp { __device__ T operator()(T x) const { return roundf((float)x); } };

template<typename T> struct ExpOp { __device__ T operator()(T x) const { return exp(x); } };
template<typename T> struct LogOp { __device__ T operator()(T x) const { return log(x); } };
template<typename T> struct SqrtOp { __device__ T operator()(T x) const { return sqrt(x); } };
template<typename T> struct RsqrtOp { __device__ T operator()(T x) const { return rsqrt(x); } };
template<typename T> struct SinOp { __device__ T operator()(T x) const { return sin(x); } };
template<typename T> struct CosOp { __device__ T operator()(T x) const { return cos(x); } };
template<typename T> struct TanOp { __device__ T operator()(T x) const { return tan(x); } };
template<typename T> struct TanhOp { __device__ T operator()(T x) const { return tanh(x); } };
template<typename T> struct SigmoidOp { __device__ T operator()(T x) const { return 1.0 / (1.0 + exp(-x)); } };
template<typename T> struct ReluOp { __device__ T operator()(T x) const { return x > 0 ? x : 0; } };
template<typename T> struct GeluOp { 
    __device__ T operator()(T x) const { 
        const T kAlpha = static_cast<T>(0.70710678118654752440);
        return 0.5 * x * (1.0 + erf(x * kAlpha));
    } 
};
template<typename T> struct SiluOp { 
    __device__ T operator()(T x) const { return x / (1.0 + exp(-x)); } 
};

// Implementations
Tensor abs_kernel(const Tensor& self) { return unary_op_kernel(self, AbsOp<float>()); } // Template hack: we need type-erased functor or typed dispatch inside
// Wait, unary_op_kernel does switch(dtype). So we need a functor that works for all types or template it.
// My unary_op_kernel takes Func. If Func is a template struct, we need to instantiate it inside the macro.

// Revised Dispatcher using generic lambda or struct with template operator()
// Since we can't pass template template to function easily, let's use a struct with templated operator()
struct AbsFunctor { template<typename T> __device__ T operator()(T x) const { return x >= 0 ? x : -x; } };
struct NegFunctor { template<typename T> __device__ T operator()(T x) const { return -x; } };
struct SquareFunctor { template<typename T> __device__ T operator()(T x) const { return x * x; } };
struct SignFunctor { 
    template<typename T> __device__ T operator()(T x) const { 
        if (x > 0) return static_cast<T>(1);
        if (x < 0) return static_cast<T>(-1);
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
        unary_kernel_cuda_impl<ctype><<<grid, block>>>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), functor); \
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
    
    if (out_dtype == DType::Float32) {
        Tensor in = (self.dtype() == DType::Float32) ? self_contig : self_contig.to(DType::Float32);
        unary_kernel_cuda_impl<float><<<grid, block>>>(n, in.data_ptr<float>(), result.data_ptr<float>(), functor);
    } else if (out_dtype == DType::Float64) {
        Tensor in = (self.dtype() == DType::Float64) ? self_contig : self_contig.to(DType::Float64);
        unary_kernel_cuda_impl<double><<<grid, block>>>(n, in.data_ptr<double>(), result.data_ptr<double>(), functor);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

struct ExpFunctor { template<typename T> __device__ T operator()(T x) const { return exp(x); } };
struct LogFunctor { template<typename T> __device__ T operator()(T x) const { return log(x); } };
struct SqrtFunctor { template<typename T> __device__ T operator()(T x) const { return sqrt(x); } };
struct RsqrtFunctor { template<typename T> __device__ T operator()(T x) const { return rsqrt(x); } };
struct SinFunctor { template<typename T> __device__ T operator()(T x) const { return sin(x); } };
struct CosFunctor { template<typename T> __device__ T operator()(T x) const { return cos(x); } };
struct TanhFunctor { template<typename T> __device__ T operator()(T x) const { return tanh(x); } };
struct SigmoidFunctor { template<typename T> __device__ T operator()(T x) const { return 1.0 / (1.0 + exp(-x)); } };
struct ReluFunctor { template<typename T> __device__ T operator()(T x) const { return x > 0 ? x : 0; } };
struct GeluFunctor { 
    template<typename T> __device__ T operator()(T x) const { 
        const T kAlpha = static_cast<T>(0.70710678118654752440);
        return 0.5 * x * (1.0 + erf(x * kAlpha));
    } 
};
struct SiluFunctor { 
    template<typename T> __device__ T operator()(T x) const { return x / (1.0 + exp(-x)); } 
};

Tensor exp_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ExpFunctor()); }
Tensor log_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, LogFunctor()); }
Tensor sqrt_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SqrtFunctor()); }
Tensor rsqrt_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, RsqrtFunctor()); }
Tensor sin_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SinFunctor()); }
Tensor cos_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, CosFunctor()); }
Tensor tanh_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, TanhFunctor()); }
Tensor sigmoid_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SigmoidFunctor()); }
Tensor relu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ReluFunctor()); }
Tensor gelu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, GeluFunctor()); }
Tensor silu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SiluFunctor()); }

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

// --- Comparison ---
template <typename T, typename Func>
__global__ void comparison_kernel_cuda_impl(int64_t n, const T* a, const T* b, bool* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = func(a[i], b[i]);
    }
}

template <typename T, typename Func>
__global__ void comparison_scalar_kernel_cuda_impl(int64_t n, const T* a, T b, bool* output, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = func(a[i], b);
    }
}

template<typename Functor>
Tensor comparison_op_kernel(const Tensor& self, const Tensor& other, Functor functor) {
    if (self.shape() != other.shape()) TP_THROW(RuntimeError, "CUDA comparison: broadcasting not supported");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // For now, assume same type
    if (self.dtype() != other.dtype()) TP_THROW(RuntimeError, "CUDA comparison: dtypes must match");
    
    #define COMP_CASE(ctype, name) \
    case DType::name: { \
        comparison_kernel_cuda_impl<ctype><<<grid, block>>>(n, self.data_ptr<ctype>(), other.data_ptr<ctype>(), result.data_ptr<bool>(), functor); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_CASE)
        default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
    }
    #undef COMP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template<typename Functor>
Tensor comparison_scalar_op_kernel(const Tensor& self, Scalar other, Functor functor) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    #define COMP_SCALAR_CASE(ctype, name) \
    case DType::name: { \
        comparison_scalar_kernel_cuda_impl<ctype><<<grid, block>>>(n, self.data_ptr<ctype>(), other.to<ctype>(), result.data_ptr<bool>(), functor); \
        break; \
    }
    switch (self.dtype()) {
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
        ctype min_val = min.has_value() ? min->to<ctype>() : 0; \
        ctype max_val = max.has_value() ? max->to<ctype>() : 0; \
        clamp_kernel_cuda_impl<ctype><<<grid, block>>>(n, self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), min_val, max_val, min.has_value(), max.has_value()); \
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
        ctype min_val = min.has_value() ? min->to<ctype>() : 0; \
        ctype max_val = max.has_value() ? max->to<ctype>() : 0; \
        clamp_backward_kernel_cuda_impl<ctype><<<grid, block>>>(n, grad_contig.data_ptr<ctype>(), self_contig.data_ptr<ctype>(), result.data_ptr<ctype>(), min_val, max_val, min.has_value(), max.has_value()); \
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
        binary_kernel_cuda_impl<float><<<grid, block>>>(n, a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), functor);
    } else if (out_dtype == DType::Float64) {
        Tensor a = (self.dtype() == DType::Float64) ? self_contig : self_contig.to(DType::Float64);
        Tensor b = (other.dtype() == DType::Float64) ? other_contig : other_contig.to(DType::Float64);
        binary_kernel_cuda_impl<double><<<grid, block>>>(n, a.data_ptr<double>(), b.data_ptr<double>(), result.data_ptr<double>(), functor);
    }
    
    CUDA_CHECK(cudaGetLastError());
    return result;
}

struct PowFunctor { template<typename T> __device__ T operator()(T a, T b) const { return pow(a, b); } };
struct PowScalarFunctor { 
    float exponent; 
    PowScalarFunctor(float e) : exponent(e) {}
    template<typename T> __device__ T operator()(T x) const { return pow(x, static_cast<T>(exponent)); } 
};
struct Atan2Functor { template<typename T> __device__ T operator()(T a, T b) const { return atan2(a, b); } };

Tensor pow_kernel_cuda(const Tensor& self, const Tensor& other) { return binary_float_op_kernel_v2(self, other, PowFunctor()); }
Tensor pow_scalar_kernel_cuda(const Tensor& self, Scalar exponent) { return unary_float_op_kernel_v2(self, PowScalarFunctor(exponent.to<float>())); }
Tensor atan2_kernel_cuda(const Tensor& self, const Tensor& other) { return binary_float_op_kernel_v2(self, other, Atan2Functor()); }

// --- Lerp ---
template <typename T>
__global__ void lerp_scalar_kernel_cuda_impl(int64_t n, const T* start, const T* end, T* output, T weight) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = start[i] + weight * (end[i] - start[i]);
    }
}

template <typename T>
__global__ void lerp_tensor_kernel_cuda_impl(int64_t n, const T* start, const T* end, const T* weight, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = start[i] + weight[i] * (end[i] - start[i]);
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
    
    if (self.dtype() == DType::Float32) {
        lerp_scalar_kernel_cuda_impl<float><<<grid, block>>>(n, self_c.data_ptr<float>(), end_c.data_ptr<float>(), result.data_ptr<float>(), weight.to<float>());
    } else {
        TP_THROW(NotImplementedError, "CUDA lerp: only float32 supported");
    }
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
    
    if (self.dtype() == DType::Float32) {
        lerp_tensor_kernel_cuda_impl<float><<<grid, block>>>(n, self_c.data_ptr<float>(), end_c.data_ptr<float>(), weight_c.data_ptr<float>(), result.data_ptr<float>());
    } else {
        TP_THROW(NotImplementedError, "CUDA lerp: only float32 supported");
    }
    return result;
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
    int64_t* d_counter;
    cudaMalloc(&d_counter, sizeof(int64_t));
    cudaMemset(d_counter, 0, sizeof(int64_t));
    
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    // We can't use template for mask type, it's always bool.
    // But we need template for input type? No, count only needs mask.
    count_mask_kernel<float><<<grid, block>>>(n, mask_c.data_ptr<bool>(), d_counter); // Template arg unused but required if templated
    
    int64_t count = 0;
    cudaMemcpy(&count, d_counter, sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    // 2. Allocate output
    Tensor result = Tensor::empty({count}, self.dtype(), self.device());
    
    if (count > 0) {
        // Reset counter for indexing
        cudaMemset(d_counter, 0, sizeof(int64_t));
        
        #define SEL_CASE(ctype, name) \
        case DType::name: { \
            masked_select_kernel<ctype><<<grid, block>>>(n, self_c.data_ptr<ctype>(), mask_c.data_ptr<bool>(), result.data_ptr<ctype>(), d_counter); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(SEL_CASE)
            default: TP_THROW(TypeError, "CUDA masked_select: Unsupported dtype");
        }
        #undef SEL_CASE
    }
    
    cudaFree(d_counter);
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
    m.impl("log", log_kernel_cuda);
    m.impl("sqrt", sqrt_kernel_cuda);
    m.impl("rsqrt", rsqrt_kernel_cuda);
    m.impl("sin", sin_kernel_cuda);
    m.impl("cos", cos_kernel_cuda);
    m.impl("tanh", tanh_kernel_cuda);
    
    m.impl("sigmoid", sigmoid_kernel_cuda);
    m.impl("relu", relu_kernel_cuda);
    m.impl("gelu", gelu_kernel_cuda);
    m.impl("silu", silu_kernel_cuda);
    
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
    
    m.impl("pow.Tensor_Tensor", pow_kernel_cuda);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel_cuda);
    m.impl("atan2", atan2_kernel_cuda);
    
    m.impl("lerp", lerp_scalar_kernel_cuda);
    m.impl("lerp.Tensor", lerp_tensor_kernel_cuda);
    m.impl("masked_select", masked_select_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
