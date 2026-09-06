#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "CUDABroadcast.cuh"
#include "CUDAComplex.cuh"
#include "ElementwiseStrided.cuh"
#include "CUDALoops.cuh"
#include "GradMode.h"
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cmath>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& self, int64_t dim);

// RAII over thread-local GradMode for mutation-free sections.
struct NoGradGuard {
    bool prev;
    NoGradGuard() : prev(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev); }
};

// --- Utils ---
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

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

// Unary dispatcher for same-dtype scalar operations.
template<typename Functor>
Tensor unary_op_kernel_v2(const Tensor& self, Functor functor) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();

    #define OP_CASE(ctype, name) \
    case DType::name: \
        gpu_kernel(iter, [functor] __device__(ctype x) -> ctype { \
            return functor(x); \
        }); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "CUDA unary op: Unsupported dtype");
    }
    #undef OP_CASE
    return result;
}

Tensor abs_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble) {
            TP_THROW(NotImplementedError,
                     "CUDA abs: half complexes are not supported yet");
        }
        DType out_dt = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), out_dt,
            self.device());
        const int64_t n = self.numel();
        auto stream = getCurrentCUDAStream().stream();
        Tensor sc = self.contiguous();
        if (self.dtype() == DType::ComplexFloat)
            cuda::cplx::launch_abs<float>(n, sc.data_ptr(), result.data_ptr(), stream);
        else
            cuda::cplx::launch_abs<double>(n, sc.data_ptr(), result.data_ptr(), stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return unary_op_kernel_v2(self, AbsFunctor());
}
// neg/square complex paths are defined after complex_math_kernel_cuda and the
// Cx* functors below (translation-unit ordering; see #undef CX_FUNCTOR).
// interleaved re/im storage directly.
__global__ void sign_cplx_f32_kernel(int64_t n, const float* __restrict__ src, float* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float re = src[2 * i], im = src[2 * i + 1];
    const float m = sqrtf(re * re + im * im);
    if (m == 0.f) { dst[2 * i] = 0.f; dst[2 * i + 1] = 0.f; }
    else { dst[2 * i] = re / m; dst[2 * i + 1] = im / m; }
}

__global__ void sign_cplx_f64_kernel(int64_t n, const double* __restrict__ src, double* __restrict__ dst) {
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const double re = src[2 * i], im = src[2 * i + 1];
    const double m = sqrt(re * re + im * im);
    if (m == 0.) { dst[2 * i] = 0.; dst[2 * i + 1] = 0.; }
    else { dst[2 * i] = re / m; dst[2 * i + 1] = im / m; }
}

Tensor sign_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        const int64_t n = self.numel();
        if (n > 0) {
            dim3 block(256);
            dim3 grid((n + 255) / 256);
            Tensor self_contig = self.contiguous();
            auto stream = getCurrentCUDAStream().stream();
            if (self.dtype() == DType::ComplexDouble) {
                sign_cplx_f64_kernel<<<grid, block, 0, stream>>>(
                    n, static_cast<const double*>(self_contig.data_ptr()),
                    static_cast<double*>(result.data_ptr()));
            } else {
                sign_cplx_f32_kernel<<<grid, block, 0, stream>>>(
                    n, static_cast<const float*>(self_contig.data_ptr()),
                    static_cast<float*>(result.data_ptr()));
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return result;
    }
    return unary_op_kernel_v2(self, SignFunctor());
}

// --- complex elementwise math (thrust::complex on interleaved storage) -----
template <typename F>
Tensor complex_math_kernel_cuda(const Tensor& self, F f) {
    const DType dt = self.dtype();
    TP_CHECK(isComplexType(dt),
             "complex_math_kernel_cuda: expected a complex dtype");
    if (dt != DType::ComplexFloat && dt != DType::ComplexDouble) {
        TP_THROW(NotImplementedError,
                 "CUDA complex math: half complexes are not supported yet");
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  dt, self.device());
    const int64_t n = self.numel();
    if (n == 0) return result;
    Tensor self_contig = self.contiguous();
    auto stream = getCurrentCUDAStream().stream();
    if (dt == DType::ComplexFloat)
        cuda::cplx::launch_unary<float>(n, self_contig.data_ptr(),
                                        result.data_ptr(), f, stream);
    else
        cuda::cplx::launch_unary<double>(n, self_contig.data_ptr(),
                                         result.data_ptr(), f, stream);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

#define CX_FUNCTOR(NAME, EXPR)                                                \
    struct NAME {                                                             \
        template <typename T>                                                 \
        __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {\
            return EXPR;                                                      \
        }                                                                     \
    };
CX_FUNCTOR(CxExp, exp(z))
CX_FUNCTOR(CxExpm1, exp(z) - static_cast<T>(1))
CX_FUNCTOR(CxLog, log(z))
CX_FUNCTOR(CxLog10, log10(z))
CX_FUNCTOR(CxLog1p, log(z + static_cast<T>(1)))
CX_FUNCTOR(CxLog2, log(z) / log(static_cast<T>(2)))
CX_FUNCTOR(CxSqrt, sqrt(z))
CX_FUNCTOR(CxRsqrt, static_cast<T>(1) / sqrt(z))
CX_FUNCTOR(CxSin, sin(z))
CX_FUNCTOR(CxCos, cos(z))
CX_FUNCTOR(CxTan, tan(z))
CX_FUNCTOR(CxAsin, asin(z))
CX_FUNCTOR(CxAcos, acos(z))
CX_FUNCTOR(CxAtan, atan(z))
CX_FUNCTOR(CxSinh, sinh(z))
CX_FUNCTOR(CxCosh, cosh(z))
CX_FUNCTOR(CxTanh, tanh(z))
CX_FUNCTOR(CxAsinh, asinh(z))
CX_FUNCTOR(CxAcosh, acosh(z))
CX_FUNCTOR(CxAtanh, atanh(z))
CX_FUNCTOR(CxSigmoid,
           static_cast<T>(1) / (static_cast<T>(1) + exp(-z)))
CX_FUNCTOR(CxRecip, static_cast<T>(1) / z)
CX_FUNCTOR(CxNeg, -z)
CX_FUNCTOR(CxSquare, z * z)
struct CxPowScalar {
    double re, im;
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {
        return pow(z, thrust::complex<T>(static_cast<T>(re), static_cast<T>(im)));
    }
};
struct CxPowScalarC {
    double re, im;
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> z) const {
        return pow(z, thrust::complex<T>(static_cast<T>(re), static_cast<T>(im)));
    }
};
#undef CX_FUNCTOR

Tensor neg_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxNeg{});
    return unary_op_kernel_v2(self, NegFunctor());
}
Tensor square_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSquare{});
    return unary_op_kernel_v2(self, SquareFunctor());
}

// Float ops need simpler dispatch since we cast to float/double
template<typename Functor>
Tensor unary_float_op_kernel_v2(const Tensor& self, Functor functor) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(self.dtype() == out_dtype)
        .add_output(result)
        .add_input(self)
        .build();

    switch (out_dtype) {
        case DType::Float16:
            gpu_kernel(iter, [functor] __device__(Half x) -> Half {
                return static_cast<Half>(functor(static_cast<float>(x)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __device__(BFloat16 x) -> BFloat16 {
                return static_cast<BFloat16>(functor(static_cast<float>(x)));
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __device__(float x) -> float {
                return functor(x);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __device__(double x) -> double {
                return functor(x);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA unary float op: Unsupported output dtype");
    }
    return result;
}

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
struct SiluBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // sigmoid = 1 / (1 + exp(-x)); dy * sigmoid * (1 + x * (1 - sigmoid))
        const T one = static_cast<T>(1);
        const T s = one / (one + exp(-x));
        return dy * s * (one + x * (one - s));
    }
};

Tensor exp_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxExp{});
    return unary_float_op_kernel_v2(self, ExpFunctor());
}
Tensor expm1_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxExpm1{});
    return unary_float_op_kernel_v2(self, Expm1Functor());
}
Tensor erf_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ErfFunctor()); }
Tensor erfc_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ErfcFunctor()); }
Tensor log_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxLog{});
    return unary_float_op_kernel_v2(self, LogFunctor());
}
Tensor log10_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxLog10{});
    return unary_float_op_kernel_v2(self, Log10Functor());
}
Tensor log1p_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxLog1p{});
    return unary_float_op_kernel_v2(self, Log1pFunctor());
}
Tensor log2_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxLog2{});
    return unary_float_op_kernel_v2(self, Log2Functor());
}
Tensor lgamma_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, LgammaFunctor()); }
Tensor sqrt_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSqrt{});
    return unary_float_op_kernel_v2(self, SqrtFunctor());
}
Tensor rsqrt_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxRsqrt{});
    return unary_float_op_kernel_v2(self, RsqrtFunctor());
}
Tensor sin_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSin{});
    return unary_float_op_kernel_v2(self, SinFunctor());
}
Tensor cos_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxCos{});
    return unary_float_op_kernel_v2(self, CosFunctor());
}
Tensor tanh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxTanh{});
    return unary_float_op_kernel_v2(self, TanhFunctor());
}
Tensor sigmoid_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSigmoid{});
    return unary_float_op_kernel_v2(self, SigmoidFunctor());
}
struct AngleFunctor {
    template<typename T> __device__ T operator()(T x) const {
        return x >= T(0) ? T(0) : static_cast<T>(3.14159265358979323846);
    }
};
Tensor angle_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble) {
            TP_THROW(NotImplementedError,
                     "CUDA angle: half complexes are not supported yet");
        }
        DType out_dt = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), out_dt,
            self.device());
        const int64_t n = self.numel();
        auto stream = getCurrentCUDAStream().stream();
        Tensor sc = self.contiguous();
        if (self.dtype() == DType::ComplexFloat)
            cuda::cplx::launch_angle<float>(n, sc.data_ptr(), result.data_ptr(), stream);
        else
            cuda::cplx::launch_angle<double>(n, sc.data_ptr(), result.data_ptr(), stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return unary_float_op_kernel_v2(self, AngleFunctor());
}
Tensor relu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ReluFunctor()); }
Tensor gelu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, GeluFunctor()); }
Tensor silu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SiluFunctor()); }
template<typename Functor>
Tensor activation_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Functor functor);
Tensor silu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, SiluBackwardFunctor());
}

// ---------------------------------------------------------------------------
//     (GeluCUDAKernelImpl / GeluBackwardCUDAKernelImpl)
// ---------------------------------------------------------------------------

template<typename Functor>
Tensor activation_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Functor functor) {
    if (grad_output.shape() != self.shape()) TP_THROW(RuntimeError, "CUDA activation backward: shape mismatch");
    DType out_dtype = grad_output.dtype();
    if (!isFloatingType(out_dtype)) TP_THROW(TypeError, "CUDA activation backward: expected floating point dtype");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    if (grad_output.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(grad_output)
        .add_input(self)
        .build();

    switch (out_dtype) {
        case DType::Float16:
            gpu_kernel(iter, [functor] __device__(Half dy, Half x) -> Half {
                return static_cast<Half>(functor(static_cast<float>(dy),
                                                 static_cast<float>(x)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __device__(BFloat16 dy,
                                                  BFloat16 x) -> BFloat16 {
                return static_cast<BFloat16>(functor(static_cast<float>(dy),
                                                     static_cast<float>(x)));
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __device__(float dy, float x) -> float {
                return functor(dy, x);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __device__(double dy, double x) -> double {
                return functor(dy, x);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA activation backward: Unsupported dtype");
    }
    return result;
}

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
        T lo = static_cast<T>(min_val_), hi = static_cast<T>(max_val_);
        return x < lo ? lo : (x > hi ? hi : x);
    }
};
struct HardtanhBackwardFunctor {
    double min_val_, max_val_;
    HardtanhBackwardFunctor(double lo, double hi) : min_val_(lo), max_val_(hi) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        return (x <= static_cast<T>(min_val_) || x >= static_cast<T>(max_val_)) ? static_cast<T>(0) : dy;
    }
};
struct HardswishFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return x * v / static_cast<T>(6);
    }
};
struct HardswishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        return x <= static_cast<T>(-3) ? static_cast<T>(0)
             : x >= static_cast<T>(3)  ? dy
             : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct HardsigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return v / static_cast<T>(6);
    }
};
struct HardsigmoidBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        return (x <= static_cast<T>(-3) || x >= static_cast<T>(3)) ? static_cast<T>(0)
                                                                   : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct LeakyReluFunctor {
    double negative_slope_;
    LeakyReluFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T x) const {
        return x > static_cast<T>(0) ? x : static_cast<T>(negative_slope_) * x;
    }
};
struct LeakyReluBackwardFunctor {
    double negative_slope_;
    LeakyReluBackwardFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        return x > static_cast<T>(0) ? dy : static_cast<T>(negative_slope_) * dy;
    }
};
struct EluFunctor {
    double negcoef_, poscoef_, negiptcoef_;
    EluFunctor(double alpha, double scale, double input_scale)
        : negcoef_(alpha * scale), poscoef_(scale), negiptcoef_(input_scale) {}
    template<typename T> __device__ T operator()(T a) const {
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
        T sp = log1p(exp(x));
        return x * tanh(sp);
    }
};
struct MishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        T sp = log1p(exp(x));
        T tanh_sp = tanh(sp);
        T sech2 = static_cast<T>(1) - tanh_sp * tanh_sp;
        T gsp = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
        return dy * (tanh_sp + x * sech2 * gsp);
    }
};
struct SeluFunctor {
    template<typename T> __device__ T operator()(T x) const {
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
        return x > static_cast<T>(0) ? x : static_cast<T>(alpha_) * expm1(x / static_cast<T>(alpha_));
    }
};
struct SoftplusFunctor {
    double beta_, threshold_;
    SoftplusFunctor(double beta, double threshold) : beta_(beta), threshold_(threshold) {}
    template<typename T> __device__ T operator()(T a) const {
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
        //   beta*a > threshold ? dy : dy * sigmoid(beta*a)
        T beta_in = static_cast<T>(beta_);
        return a * beta_in > static_cast<T>(threshold_)
            ? dy
            : dy / (static_cast<T>(1) + exp(-a * beta_in));
    }
};

struct GeluBackwardNoneFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
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
    if (approximate == "tanh") return unary_float_op_kernel_v2(self, GeluTanhFunctor());
    else if (approximate != "none") TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    return unary_float_op_kernel_v2(self, GeluFunctor());
}
Tensor gelu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
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

//   out = min(x, 0) - log1p(exp(-|x|))
struct LogSigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T z = x < static_cast<T>(0) ? x : static_cast<T>(0);
        T neg_abs = x < static_cast<T>(0) ? x : -x;
        return z - log1p(exp(neg_abs));
    }
};
// branch-split so exp() never overflows.
struct LogSigmoidBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        if (x >= static_cast<T>(0)) {
            T e = exp(-x);
            return dy * (e / (static_cast<T>(1) + e));
        }
        return dy / (static_cast<T>(1) + exp(x));
    }
};
Tensor log_sigmoid_kernel_cuda(const Tensor& self) {
    return unary_float_op_kernel_v2(self, LogSigmoidFunctor());
}
Tensor log_sigmoid_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, LogSigmoidBackwardFunctor());
}
// out-variant: the saved buffer only feeds the CPU loop's stable form; the
// CUDA elementwise formula recomputes the same expression from x directly.
Tensor& log_sigmoid_backward_out_cuda(const Tensor& grad_output,
                                      const Tensor& self, const Tensor& buffer,
                                      Tensor& grad_input) {
    (void)buffer;
    grad_input = activation_backward_kernel_cuda(grad_output, self,
                                                 LogSigmoidBackwardFunctor());
    return grad_input;
}

// the caller-provided noise; eval is leaky_relu with slope (lower+upper)/2.
template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor);

struct RreluWithNoiseFunctor {
    double lower_, upper_;
    bool training_;
    RreluWithNoiseFunctor(double l, double u, bool t) : lower_(l), upper_(u), training_(t) {}
    template<typename T> __device__ T operator()(T x, T r) const {
        if (training_) return x <= static_cast<T>(0) ? x * r : x;
        T slope = static_cast<T>((lower_ + upper_) / 2.0);
        return x >= static_cast<T>(0) ? x : x * slope;
    }
};
struct RreluWithNoiseTrainBackwardFunctor {
    // noise=1 for positive elements; this kernel masks with self instead).
    template<typename T> __device__ T operator()(T dy, T x, T r) const {
        return x <= static_cast<T>(0) ? dy * r : dy;
    }
};
struct RreluWithNoiseEvalBackwardFunctor {
    double slope_;
    explicit RreluWithNoiseEvalBackwardFunctor(double s) : slope_(s) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        return x >= static_cast<T>(0) ? dy : dy * static_cast<T>(slope_);
    }
};

template <typename T, typename Func>
__global__ void ternary_kernel_cuda_impl(int64_t n, const T* a, const T* b, const T* c, T* out, Func func) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = func(a[i], b[i], c[i]);
}

// Forward declaration: first call sites (rrelu_with_noise) precede the
// definition below, and nvcc's two-phase lookup needs the template declared.
template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor);

Tensor rrelu_with_noise_kernel_cuda(const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training) {
    return binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
}
// inplace out-variant: recompute and write back through the same functor.
Tensor& rrelu_with_noise__cuda(Tensor& self, Tensor& noise, Scalar lower,
                               Scalar upper, bool training) {
    Tensor result = binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
    self.copy_(result);
    return self;
}
// out-variant of the forward: the noise buffer is filled on the fly and
// returned alongside the result.
Tensor rrelu_with_noise_out_cuda(const Tensor& self, Tensor& noise, Scalar lower,
                                 Scalar upper, bool training) {
    noise = binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
    return noise;
}
// log_sigmoid forward with its saved buffer: log_sigmoid(x) = -softplus(-x);
// the buffer caches exp(-|x|), the stable remainder of the softplus
// evaluation the backward reuses elementwise.  Composed from the elementwise
// kernels in this translation unit and the dispatched add/sub wrappers.
std::tuple<Tensor, Tensor> log_sigmoid_forward_components_cuda(const Tensor& self) {
    const Scalar one(1.0);
    Tensor b = exp_kernel_cuda(neg_kernel_cuda(abs_kernel_cuda(self)));  // exp(-|x|)
    Tensor one_plus_b = b + one;
    Tensor log_b = log_kernel_cuda(b);
    Tensor log_one_plus_b = log_kernel_cuda(one_plus_b);
    Tensor pos_branch = log_b - log_one_plus_b;        // log(b) - log(1+b)
    Tensor neg_branch = self + log_b;                  // x + log(b), x < 0
    Tensor output = ops::where(self.lt(Scalar(0.0)), neg_branch, pos_branch);
    return {output, b};
}
// out-variants: run the value kernel, then transfer into the caller's buffer.
Tensor& gelu_out_cuda(const Tensor& self, const std::string& approximate,
                      Tensor& out) {
    out = gelu_kernel_cuda_v2(self, approximate);
    return out;
}
Tensor& gelu_backward_grad_input_cuda(const Tensor& grad_output,
                                      const Tensor& self,
                                      const std::string& approximate,
                                      Tensor& grad_input) {
    grad_input = gelu_backward_kernel_cuda(grad_output, self, approximate);
    return grad_input;
}
Tensor& glu_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& self,
                                     int64_t dim, Tensor& grad_input) {
    grad_input = glu_backward_cuda(grad_output, self, dim);
    return grad_input;
}
std::tuple<Tensor, Tensor> log_sigmoid_forward_out_cuda(const Tensor& self,
                                                        Tensor& output,
                                                        Tensor& buffer) {
    auto [o, b] = log_sigmoid_forward_components_cuda(self);
    output = std::move(o);
    buffer = std::move(b);
    return std::make_tuple(output, buffer);
}
Tensor rrelu_with_noise_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
    // (masked by self here, see functor note); eval -> leaky_relu_backward
    // with slope (lower + upper) / 2.
    if (training) {
        if (grad_output.shape() != self.shape() || grad_output.shape() != noise.shape())
            TP_THROW(RuntimeError, "rrelu_with_noise_backward: shape mismatch");
        DType dt = grad_output.dtype();
        if (dt != DType::Float32 && dt != DType::Float64)
            TP_THROW(TypeError, "rrelu_with_noise_backward CUDA supports Float32/Float64 only");
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), dt, grad_output.device());
        const int64_t n = grad_output.numel();
        if (n == 0) return result;
        const Tensor gc = grad_output.contiguous();
        const Tensor sc = self.contiguous();
        const Tensor nc = noise.contiguous();
        dim3 block(256);
        dim3 grid((unsigned)((n + 255) / 256));
        if (dt == DType::Float32) {
            ternary_kernel_cuda_impl<float><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                n, gc.data_ptr<float>(), sc.data_ptr<float>(), nc.data_ptr<float>(),
                result.data_ptr<float>(), RreluWithNoiseTrainBackwardFunctor());
        } else {
            ternary_kernel_cuda_impl<double><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                n, gc.data_ptr<double>(), sc.data_ptr<double>(), nc.data_ptr<double>(),
                result.data_ptr<double>(), RreluWithNoiseTrainBackwardFunctor());
        }
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    (void)self_is_result; // result >= 0 iff self >= 0 for a positive slope.
    const double slope = (lower.toDouble() + upper.toDouble()) / 2.0;
    return binary_float_op_kernel_v2(grad_output, self, RreluWithNoiseEvalBackwardFunctor(slope));
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

Tensor acos_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAcos{});
    return unary_float_op_kernel_v2(self, AcosFunctor());
}
Tensor acosh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAcosh{});
    return unary_float_op_kernel_v2(self, AcoshFunctor());
}
Tensor asin_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAsin{});
    return unary_float_op_kernel_v2(self, AsinFunctor());
}
Tensor asinh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAsinh{});
    return unary_float_op_kernel_v2(self, AsinhFunctor());
}
Tensor atan_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAtan{});
    return unary_float_op_kernel_v2(self, AtanFunctor());
}
Tensor atanh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxAtanh{});
    return unary_float_op_kernel_v2(self, AtanhFunctor());
}
Tensor ceil_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, CeilFunctor()); }
Tensor cosh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxCosh{});
    return unary_float_op_kernel_v2(self, CoshFunctor());
}
Tensor floor_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, FloorFunctor()); }
Tensor round_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, RoundFunctor()); }
Tensor sinh_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSinh{});
    return unary_float_op_kernel_v2(self, SinhFunctor());
}
Tensor tan_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxTan{});
    return unary_float_op_kernel_v2(self, TanFunctor());
}
Tensor trunc_kernel_cuda(const Tensor& self) { return unary_op_kernel_v2(self, TruncFunctor()); }
Tensor frac_kernel_cuda(const Tensor& self) {
    if (isIntegralType(self.dtype())) {
        TP_THROW(NotImplementedError, "frac is not implemented for integral tensors");
    }
    return unary_float_op_kernel_v2(self, FracFunctor());
}

// --- Comparison ---

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
    Tensor a = (self.dtype() == common_dtype) ? self : self.to(common_dtype);
    Tensor b = (other.dtype() == common_dtype) ? other : other.to(common_dtype);
    if (result.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define COMP_CASE(ctype, name) \
    case DType::name: \
        gpu_kernel(iter, [functor] __device__(ctype lhs, ctype rhs) -> bool { \
            return functor(lhs, rhs); \
        }); \
        break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_CASE)
        default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
    }
    #undef COMP_CASE
    return result;
}

template<typename Functor>
Tensor comparison_scalar_op_kernel(const Tensor& self, Scalar other, Functor functor) {
    DType common = result_type_with_scalar_cuda(self, other);
    Tensor in = (self.dtype() == common) ? self : self.to(common);
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(in.shape()), DType::Bool, self.device());
    if (in.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result)
        .add_input(in)
        .build();

    #define COMP_SCALAR_CASE(ctype, name) \
    case DType::name: { \
        const ctype rhs = other.to<ctype>(); \
        gpu_kernel(iter, [functor, rhs] __device__(ctype lhs) -> bool { \
            return functor(lhs, rhs); \
        }); \
        break; \
    }
    switch (common) {
        TENSORPLAY_FORALL_SCALAR_TYPES(COMP_SCALAR_CASE)
        default: TP_THROW(TypeError, "CUDA comparison: Unsupported dtype");
    }
    #undef COMP_SCALAR_CASE
    return result;
}

struct EqFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a == b; } };
struct NeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a != b; } };
struct LtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a < b; } };
struct LeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a <= b; } };
struct GtFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a > b; } };
struct GeFunctor { template<typename T> __device__ bool operator()(T a, T b) const { return a >= b; } };

// stay rejected by comparison_op_kernel's real-only dispatch.
template <typename CxOp>
Tensor complex_comparison_kernel(const Tensor& self, const Tensor& other,
                                 CxOp op) {
    DType rd = promoteTypes(self.dtype(), other.dtype());
    if (rd != DType::ComplexFloat && rd != DType::ComplexDouble)
        TP_THROW(NotImplementedError,
                 "CUDA complex comparison: half complexes not supported");
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    Tensor result = Tensor::empty(out_shape, DType::Bool, self.device());
    Tensor a = self.to(rd);
    Tensor b = other.to(rd);
    const int64_t n = result.numel();
    auto stream = getCurrentCUDAStream().stream();
    bool same = a.is_contiguous() && b.is_contiguous() &&
                a.dim() == static_cast<int64_t>(out_shape.size()) &&
                b.dim() == static_cast<int64_t>(out_shape.size());
    if (same)
        for (int64_t d = 0; d < static_cast<int64_t>(out_shape.size()); ++d)
            if (a.size(d) != out_shape[d] || b.size(d) != out_shape[d]) { same = false; break; }
    if (rd == DType::ComplexFloat) {
        if (same)
            cuda::cplx::launch_binary<float>(n, a.data_ptr(), b.data_ptr(),
                                             result.data_ptr(), op, stream);
        else
            cuda::cplx::launch_binary_broadcast<float>(
                n, a.data_ptr(), make_desc(a, out_shape.size()), b.data_ptr(),
                make_desc(b, out_shape.size()), result.data_ptr(),
                make_desc(result, out_shape.size()), op, stream);
    } else {
        if (same)
            cuda::cplx::launch_binary<double>(n, a.data_ptr(), b.data_ptr(),
                                              result.data_ptr(), op, stream);
        else
            cuda::cplx::launch_binary_broadcast<double>(
                n, a.data_ptr(), make_desc(a, out_shape.size()), b.data_ptr(),
                make_desc(b, out_shape.size()), result.data_ptr(),
                make_desc(result, out_shape.size()), op, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor eq_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype())))
        return complex_comparison_kernel(self, other, cuda::cplx::EqOp{});
    return comparison_op_kernel(self, other, EqFunctor());
}
Tensor ne_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype())))
        return complex_comparison_kernel(self, other, cuda::cplx::NeOp{});
    return comparison_op_kernel(self, other, NeFunctor());
}
Tensor lt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, LtFunctor()); }
Tensor le_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, LeFunctor()); }
Tensor gt_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, GtFunctor()); }
Tensor ge_kernel_cuda(const Tensor& self, const Tensor& other) { return comparison_op_kernel(self, other, GeFunctor()); }

Tensor eq_scalar_kernel_cuda(const Tensor& self, Scalar other) {
    if (other.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype())
                   ? promoteTypes(toComplexType(self.dtype()), other.dtype())
                   : promoteTypes(DType::ComplexFloat, other.dtype()));
        Tensor o = Tensor::full({}, other, rd, self.device());
        return eq_kernel_cuda(self.to(rd), o);
    }
    return comparison_scalar_op_kernel(self, other, EqFunctor());
}
Tensor ne_scalar_kernel_cuda(const Tensor& self, Scalar other) {
    if (other.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype())
                   ? promoteTypes(toComplexType(self.dtype()), other.dtype())
                   : promoteTypes(DType::ComplexFloat, other.dtype()));
        Tensor o = Tensor::full({}, other, rd, self.device());
        return ne_kernel_cuda(self.to(rd), o);
    }
    return comparison_scalar_op_kernel(self, other, NeFunctor());
}
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

template <bool Maximum, typename T>
__device__ inline T maximum_minimum_elem(T a, T b) {
    if (a != a) return a;
    if (b != b) return b;
    if constexpr (Maximum) {
        return a < b ? b : a;
    } else {
        return a < b ? a : b;
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
    if (result.numel() == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define MAXMIN_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __device__(ctype lhs, ctype rhs) -> ctype { \
                return maximum_minimum_elem<Maximum>(lhs, rhs); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MAXMIN_CASE)
        default: TP_THROW(NotImplementedError, "CUDA maximum/minimum: unsupported dtype");
    }
    #undef MAXMIN_CASE
    return result;
}

Tensor maximum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<true>(self, other);
}

Tensor minimum_cuda(const Tensor& self, const Tensor& other) {
    return maximum_minimum_cuda_impl<false>(self, other);
}

// fmax/fmin: a NaN operand yields the other input verbatim; two NaNs stay NaN.
// Integral types have no NaN and reduce to the plain comparison; Half/BFloat16
// are evaluated through the float overloads below.  double keeps its own
// overload so the comparison never narrows to float.
template <bool Maximum, typename T>
__device__ inline T fmaxfmin_elem(T a, T b) {
    if constexpr (Maximum) {
        return a < b ? b : a;
    } else {
        return a < b ? a : b;
    }
}

template <bool Maximum>
__device__ inline float fmaxfmin_elem(float a, float b) {
    if (a != a) return b;
    if (b != b) return a;
    if constexpr (Maximum) return a < b ? b : a;
    else return a < b ? a : b;
}

template <bool Maximum>
__device__ inline double fmaxfmin_elem(double a, double b) {
    if (a != a) return b;
    if (b != b) return a;
    if constexpr (Maximum) return a < b ? b : a;
    else return a < b ? a : b;
}

template <bool Maximum>
__device__ inline Half fmaxfmin_elem(Half a, Half b) {
    return Half(fmaxfmin_elem<Maximum>(static_cast<float>(a), static_cast<float>(b)));
}

template <bool Maximum>
__device__ inline BFloat16 fmaxfmin_elem(BFloat16 a, BFloat16 b) {
    return BFloat16(fmaxfmin_elem<Maximum>(static_cast<float>(a), static_cast<float>(b)));
}

template <bool Maximum>
Tensor fmaxfmin_cuda_impl(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "fmax/fmin is not implemented for complex tensors");
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    #define FMAXMIN_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __device__(ctype lhs, ctype rhs) -> ctype { \
                return fmaxfmin_elem<Maximum>(lhs, rhs); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(FMAXMIN_CASE)
        default: TP_THROW(NotImplementedError, "CUDA fmax/fmin: unsupported dtype");
    }
    #undef FMAXMIN_CASE
    return result;
}

Tensor fmax_cuda(const Tensor& self, const Tensor& other) {
    return fmaxfmin_cuda_impl<true>(self, other);
}

Tensor fmin_cuda(const Tensor& self, const Tensor& other) {
    return fmaxfmin_cuda_impl<false>(self, other);
}

template <typename T>
static inline __device__ std::enable_if_t<std::is_integral_v<T>, T>
ldexp_element(T x, T exponent) {
    return exponent >= static_cast<T>(8 * static_cast<int>(sizeof(T)))
        ? T(0) : static_cast<T>(x * (T(1) << exponent));
}

template <typename T>
static inline __device__ std::enable_if_t<!std::is_integral_v<T>, T>
ldexp_element(T x, T exponent) {
    return static_cast<T>(static_cast<double>(x) *
                          ::exp2(static_cast<double>(exponent)));
}

Tensor ldexp_cuda(const Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    DType common_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor b = other.dtype() == common_dtype ? other : other.to(common_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(a)
        .add_const_input(b)
        .build();

    #define LDEXP_CASE(ctype, name) \
        case DType::name: \
            gpu_kernel(iter, [] __device__(ctype x, ctype exponent) -> ctype { \
                return ldexp_element(x, exponent); \
            }); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(LDEXP_CASE)
        default: TP_THROW(NotImplementedError, "CUDA ldexp: unsupported dtype");
    }
    #undef LDEXP_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

// clamp.Tensor: same bound logic as clamp_kernel_cuda but each bound is a
// broadcastable tensor, evaluated per element.
template <typename T>
__global__ void clamp_tensor_broadcast_kernel_cuda_impl(
    int64_t n, const T* self, TensorDesc self_desc,
    const T* lo, TensorDesc lo_desc,
    const T* hi, TensorDesc hi_desc,
    T* output, TensorDesc output_desc,
    bool has_min, bool has_max) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T val = self[get_offset(i, self_desc, output_desc)];
        if (has_min) {
            const T m = lo[get_offset(i, lo_desc, output_desc)];
            if (val < m) val = m;
        }
        if (has_max) {
            const T m = hi[get_offset(i, hi_desc, output_desc)];
            if (val > m) val = m;
        }
        output[i] = val;
    }
}

Tensor clamp_tensor_cuda(const Tensor& self, const std::optional<Tensor>& min,
                         const std::optional<Tensor>& max) {
    if (!min.has_value() && !max.has_value()) {
        return self;
    }
    DType common_dtype = self.dtype();
    if (min.has_value()) common_dtype = promoteTypes(common_dtype, min->dtype());
    if (max.has_value()) common_dtype = promoteTypes(common_dtype, max->dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "clamp is not implemented for complex tensors");
    }
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(self.shape());
    if (min.has_value()) {
        out_shape = broadcast_shapes(out_shape,
            static_cast<std::vector<int64_t>>(min->shape()));
    }
    if (max.has_value()) {
        out_shape = broadcast_shapes(out_shape,
            static_cast<std::vector<int64_t>>(max->shape()));
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    const int64_t n = result.numel();
    if (n == 0) return result;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    Tensor a = self.dtype() == common_dtype ? self.contiguous() : self.to(common_dtype).contiguous();
    Tensor lo, hi;
    TensorDesc a_desc = make_desc(a, out_shape.size());
    TensorDesc lo_desc = a_desc, hi_desc = a_desc;
    if (min.has_value()) {
        lo = min->dtype() == common_dtype ? min->contiguous() : min->to(common_dtype).contiguous();
        lo_desc = make_desc(lo, out_shape.size());
    }
    if (max.has_value()) {
        hi = max->dtype() == common_dtype ? max->contiguous() : max->to(common_dtype).contiguous();
        hi_desc = make_desc(hi, out_shape.size());
    }
    TensorDesc result_desc = make_desc(result, out_shape.size());

    #define CLAMP_T_CASE(ctype, name) \
        case DType::name: \
            clamp_tensor_broadcast_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>( \
                n, a.data_ptr<ctype>(), a_desc, \
                min.has_value() ? lo.data_ptr<ctype>() : a.data_ptr<ctype>(), lo_desc, \
                max.has_value() ? hi.data_ptr<ctype>() : a.data_ptr<ctype>(), hi_desc, \
                result.data_ptr<ctype>(), result_desc, min.has_value(), max.has_value()); \
            break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_T_CASE)
        default: TP_THROW(NotImplementedError, "CUDA clamp.Tensor: unsupported dtype");
    }
    #undef CLAMP_T_CASE
    CUDA_CHECK(cudaGetLastError());
    return result;
}

Tensor& clamp_tensor__cuda(Tensor& self, const std::optional<Tensor>& min,
                           const std::optional<Tensor>& max) {
    NoGradGuard __tp_nograd;
    self.copy_(clamp_tensor_cuda(self, min, max));
    return self;
}

Tensor& clamp_tensor_out_cuda(const Tensor& self, const std::optional<Tensor>& min,
                              const std::optional<Tensor>& max, Tensor& out) {
    out.copy_(clamp_tensor_cuda(self, min, max));
    return out;
}


Tensor clamp_kernel_cuda(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    const bool has_min = min.has_value();
    const bool has_max = max.has_value();
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();

    #define CLAMP_CASE(ctype, name) \
    case DType::name: { \
        const ctype min_val = has_min ? min->to<ctype>() : ctype(0); \
        const ctype max_val = has_max ? max->to<ctype>() : ctype(0); \
        gpu_kernel(iter, [=] __device__(ctype value) -> ctype { \
            if (has_min && value < min_val) value = min_val; \
            if (has_max && value > max_val) value = max_val; \
            return value; \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_CASE)
        default: TP_THROW(TypeError, "CUDA clamp: Unsupported dtype");
    }
    #undef CLAMP_CASE
    return result;
}

Tensor clamp_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;

    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(true)
        .resize_outputs(false)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(grad_output)
        .build();
    const bool has_min = min.has_value();
    const bool has_max = max.has_value();

    #define CLAMP_BW_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = min.has_value() ? min->to<ctype>() : ctype(0); \
        ctype max_val = max.has_value() ? max->to<ctype>() : ctype(0); \
        gpu_kernel(iter, [=] __device__(ctype input_value, ctype grad_value) -> ctype { \
            if ((has_min && input_value < min_val) || \
                (has_max && input_value > max_val)) { \
                return ctype(0); \
            } \
            return grad_value; \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_BW_CASE)
        default: TP_THROW(TypeError, "CUDA clamp_backward: Unsupported dtype");
    }
    #undef CLAMP_BW_CASE
    return result;
}

// --- Binary Ops ---

template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor) {
    if (self.shape() != other.shape()) TP_THROW(RuntimeError, "CUDA binary op: broadcasting not supported");

    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    if (self.numel() == 0) return result;
    Tensor a = self.dtype() == out_dtype ? self : self.to(out_dtype);
    Tensor b = other.dtype() == out_dtype ? other : other.to(out_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    switch (out_dtype) {
        case DType::Float16:
            gpu_kernel(iter, [functor] __device__(Half lhs, Half rhs) -> Half {
                return static_cast<Half>(functor(static_cast<float>(lhs),
                                                 static_cast<float>(rhs)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __device__(BFloat16 lhs,
                                                  BFloat16 rhs) -> BFloat16 {
                return static_cast<BFloat16>(functor(static_cast<float>(lhs),
                                                     static_cast<float>(rhs)));
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __device__(float lhs, float rhs) -> float {
                return functor(lhs, rhs);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __device__(double lhs, double rhs) -> double {
                return functor(lhs, rhs);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA binary op: Unsupported output dtype");
    }
    return result;
}

struct PowFunctor { template<typename T> __device__ T operator()(T a, T b) const { return pow(a, b); } };
struct PowScalarFunctor {
    double exponent;
    PowScalarFunctor(double e) : exponent(e) {}
    template<typename T> __device__ T operator()(T x) const { return pow(x, static_cast<T>(exponent)); }
};
struct PowBaseFunctor {
    double base;
    template<typename T> __device__ T operator()(T exponent) const {
        return pow(static_cast<T>(base), exponent);
    }
};
struct CxPowBase {
    double re, im;
    template <typename T>
    __device__ thrust::complex<T> operator()(thrust::complex<T> exponent) const {
        return pow(thrust::complex<T>(static_cast<T>(re), static_cast<T>(im)), exponent);
    }
};
struct Atan2Functor { template<typename T> __device__ T operator()(T a, T b) const { return atan2(a, b); } };

Tensor pow_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype()))) {
        DType rd = promoteTypes(self.dtype(), other.dtype());
        if (rd != DType::ComplexFloat && rd != DType::ComplexDouble)
            TP_THROW(NotImplementedError,
                     "CUDA pow: half complexes are not supported yet");
        std::vector<int64_t> out_shape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(self.shape()),
            static_cast<std::vector<int64_t>>(other.shape()));
        Tensor result = Tensor::empty(out_shape, rd, self.device());
        Tensor a = self.to(rd);
        Tensor b = other.to(rd);
        const int64_t n = result.numel();
        auto stream = getCurrentCUDAStream().stream();
        bool same = a.is_contiguous() && b.is_contiguous() &&
                    a.dim() == static_cast<int64_t>(out_shape.size()) &&
                    b.dim() == static_cast<int64_t>(out_shape.size());
        if (same)
            for (int64_t d = 0; d < static_cast<int64_t>(out_shape.size()); ++d)
                if (a.size(d) != out_shape[d] || b.size(d) != out_shape[d]) { same = false; break; }
        if (rd == DType::ComplexFloat) {
            if (same)
                cuda::cplx::launch_binary<float>(n, a.data_ptr(), b.data_ptr(),
                                                 result.data_ptr(),
                                                 cuda::cplx::PowOp{}, stream);
            else
                cuda::cplx::launch_binary_broadcast<float>(
                    n, a.data_ptr(), make_desc(a, out_shape.size()),
                    b.data_ptr(), make_desc(b, out_shape.size()),
                    result.data_ptr(), make_desc(result, out_shape.size()),
                    cuda::cplx::PowOp{}, stream);
        } else {
            if (same)
                cuda::cplx::launch_binary<double>(n, a.data_ptr(), b.data_ptr(),
                                                  result.data_ptr(),
                                                  cuda::cplx::PowOp{}, stream);
            else
                cuda::cplx::launch_binary_broadcast<double>(
                    n, a.data_ptr(), make_desc(a, out_shape.size()),
                    b.data_ptr(), make_desc(b, out_shape.size()),
                    result.data_ptr(), make_desc(result, out_shape.size()),
                    cuda::cplx::PowOp{}, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return binary_float_op_kernel_v2(self, other, PowFunctor());
}
Tensor pow_scalar_kernel_cuda(const Tensor& self, Scalar exponent) {
    if (!isComplexType(self.dtype()) && !exponent.isComplex() &&
        isIntegralType(self.dtype()) && !exponent.isFloatingPoint() &&
        exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    if (isComplexType(self.dtype()) || exponent.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype()) ? toComplexType(self.dtype())
                                            : DType::ComplexFloat);
        Tensor base = self.to(rd);
        if (exponent.isFloatingPoint() && !exponent.isComplex()) {
            double ev = exponent.toDouble();
            if (ev == 0.5) return sqrt_kernel_cuda(base);
            if (ev == -0.5) return rsqrt_kernel_cuda(base);
            if (ev == 2.0) return square_kernel_cuda(base);
            return complex_math_kernel_cuda(base, CxPowScalar{ev});
        }
        if (rd == DType::ComplexDouble)
            return complex_math_kernel_cuda(
                base, CxPowScalarC{exponent.to<std::complex<double>>().real(),
                                   exponent.to<std::complex<double>>().imag()});
        return complex_math_kernel_cuda(
            base, CxPowScalarC{static_cast<float>(exponent.to<std::complex<double>>().real()),
                               static_cast<float>(exponent.to<std::complex<double>>().imag())});
    }
    return unary_float_op_kernel_v2(self, PowScalarFunctor(exponent.toDouble()));
}
Tensor pow_scalar_tensor_kernel_cuda(Scalar base, const Tensor& exponent) {
    const DType result_dtype = ops::result_type(base, exponent);
    if (!base.isComplex() && base.toDouble() == 1.0) {
        return Tensor::ones(static_cast<std::vector<int64_t>>(exponent.shape()),
                            result_dtype, exponent.device());
    }
    Tensor exponent_cast = exponent.dtype() == result_dtype
        ? exponent : exponent.to(result_dtype);
    if (isComplexType(result_dtype)) {
        const auto base_value = base.to<std::complex<double>>();
        return complex_math_kernel_cuda(
            exponent_cast, CxPowBase{base_value.real(), base_value.imag()});
    }
    return unary_float_op_kernel_v2(
        exponent_cast, PowBaseFunctor{base.toDouble()});
}
Tensor atan2_kernel_cuda(const Tensor& self, const Tensor& other) { return binary_float_op_kernel_v2(self, other, Atan2Functor()); }

// --- Lerp ---
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

// numerically-stable calculation and casts only once on store.  Keep that
// contract for TensorPlay's reduced floating types as well; doing the
// recurrence through separate mul/add TensorIterator launches rounds twice
// and is observable in optimizer moment buffers.
template <typename T>
__global__ void lerp_tensor_reduced_kernel_cuda_impl(
        int64_t n, const T* start, const T* end, const T* weight, T* output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        const float s = static_cast<float>(start[i]);
        const float e = static_cast<float>(end[i]);
        const float w = static_cast<float>(weight[i]);
        const float value = (fabsf(w) < 0.5f)
            ? s + w * (e - s)
            : e - (e - s) * (1.0f - w);
        output[i] = static_cast<T>(value);
    }
}

Tensor lerp_scalar_kernel_cuda(const Tensor& self, const Tensor& end, Scalar weight) {
    if (self.shape() != end.shape()) TP_THROW(RuntimeError, "CUDA lerp: broadcasting not supported");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(end)
        .build();

    switch (self.dtype()) {
        case DType::Float32: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __device__(float start, float finish) -> float {
                return (fabsf(weight_value) < 0.5f)
                    ? start + weight_value * (finish - start)
                    : finish - (finish - start) * (1.0f - weight_value);
            });
            break;
        }
        case DType::Float64: {
            const double weight_value = weight.to<double>();
            gpu_kernel(iter, [weight_value] __device__(double start, double finish) -> double {
                return (fabs(weight_value) < 0.5)
                    ? start + weight_value * (finish - start)
                    : finish - (finish - start) * (1.0 - weight_value);
            });
            break;
        }
        case DType::Float16: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __device__(Half start, Half finish) -> Half {
                const float s = static_cast<float>(start);
                const float e = static_cast<float>(finish);
                const float value = (fabsf(weight_value) < 0.5f)
                    ? s + weight_value * (e - s)
                    : e - (e - s) * (1.0f - weight_value);
                return static_cast<Half>(value);
            });
            break;
        }
        case DType::BFloat16: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __device__(BFloat16 start,
                                                        BFloat16 finish) -> BFloat16 {
                const float s = static_cast<float>(start);
                const float e = static_cast<float>(finish);
                const float value = (fabsf(weight_value) < 0.5f)
                    ? s + weight_value * (e - s)
                    : e - (e - s) * (1.0f - weight_value);
                return static_cast<BFloat16>(value);
            });
            break;
        }
        default: TP_THROW(NotImplementedError, "CUDA lerp: unsupported dtype");
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

    #define LERPT_CASE(ctype, name) \
    case DType::name: { \
        lerp_tensor_kernel_cuda_impl<ctype><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(n, self_c.data_ptr<ctype>(), end_c.data_ptr<ctype>(), weight_c.data_ptr<ctype>(), result.data_ptr<ctype>()); \
        break; \
    }
    switch (self.dtype()) {
        LERPT_CASE(float, Float32)
        LERPT_CASE(double, Float64)
        case DType::Float16:
            lerp_tensor_reduced_kernel_cuda_impl<Half><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                n, self_c.data_ptr<Half>(), end_c.data_ptr<Half>(),
                weight_c.data_ptr<Half>(), result.data_ptr<Half>());
            break;
        case DType::BFloat16:
            lerp_tensor_reduced_kernel_cuda_impl<BFloat16><<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
                n, self_c.data_ptr<BFloat16>(), end_c.data_ptr<BFloat16>(),
                weight_c.data_ptr<BFloat16>(), result.data_ptr<BFloat16>());
            break;
        default: TP_THROW(NotImplementedError, "CUDA lerp: unsupported dtype");
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
    m.impl("angle", angle_kernel_cuda);
    m.impl("relu", relu_kernel_cuda);
    m.impl("gelu", gelu_kernel_cuda_v2);
    m.impl("gelu_backward", gelu_backward_kernel_cuda);
    m.impl("silu", silu_kernel_cuda);
    m.impl("silu_backward", silu_backward_kernel_cuda);
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
    m.impl("log_sigmoid", log_sigmoid_kernel_cuda);
    m.impl("log_sigmoid_backward", log_sigmoid_backward_kernel_cuda);
    m.impl("log_sigmoid_backward.grad_input", log_sigmoid_backward_out_cuda);
    m.impl("log_sigmoid_forward", log_sigmoid_forward_components_cuda);
    m.impl("log_sigmoid_forward.output", log_sigmoid_forward_out_cuda);
    m.impl("rrelu_with_noise", rrelu_with_noise_kernel_cuda);
    m.impl("rrelu_with_noise.out", rrelu_with_noise_out_cuda);
    m.impl("rrelu_with_noise_", rrelu_with_noise__cuda);
    m.impl("rrelu_with_noise_backward", rrelu_with_noise_backward_kernel_cuda);

    m.impl("gelu.out", gelu_out_cuda);
    m.impl("gelu_backward.grad_input", gelu_backward_grad_input_cuda);
    m.impl("glu_backward.grad_input", glu_backward_grad_input_cuda);
    
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

    m.impl("fmax", fmax_cuda);
    m.impl("fmin", fmin_cuda);
    m.impl("ldexp", ldexp_cuda);
    m.impl("ldexp.Tensor", ldexp_cuda);
    m.impl("clamp.Tensor", clamp_tensor_cuda);
    m.impl("clamp_.Tensor", clamp_tensor__cuda);
    m.impl("clamp.Tensor_out", clamp_tensor_out_cuda);
    m.impl("clip.Tensor", clamp_tensor_cuda);
    m.impl("clip_.Tensor", clamp_tensor__cuda);
    m.impl("clip.Tensor_out", clamp_tensor_out_cuda);
    
    m.impl("pow.Tensor_Tensor", pow_kernel_cuda);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel_cuda);
    m.impl("pow.Scalar", pow_scalar_tensor_kernel_cuda);
    m.impl("atan2", atan2_kernel_cuda);
    m.impl("arctan2", atan2_kernel_cuda);
    
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
