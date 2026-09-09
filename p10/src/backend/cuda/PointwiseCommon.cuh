#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Allocator.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "Complex.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "CUDABroadcast.cuh"
#include "ElementwiseStrided.cuh"
#include "CUDALoops.cuh"
#include "GPUPrimitives.cuh"
#include "GradMode.h"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

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

template <typename complex_t, typename math_t, typename real_t>
void complex_abs_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __host__ __device__(complex_t value) -> real_t {
        const math_t z = static_cast<math_t>(value);
        const auto real = z.real();
        const auto imag = z.imag();
        return static_cast<real_t>(::sqrt(real * real + imag * imag));
    });
}

template <typename complex_t, typename math_t, typename real_t>
void complex_angle_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __host__ __device__(complex_t value) -> real_t {
        const math_t z = static_cast<math_t>(value);
        return static_cast<real_t>(::atan2(z.imag(), z.real()));
    });
}

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
        gpu_kernel(iter, [functor] __host__ __device__(ctype x) -> ctype { \
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

template <typename complex_t, typename math_t, typename Functor>
void complex_math_loop(const Tensor& input, const Tensor& output, Functor functor) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [functor] __host__ __device__(complex_t value) -> complex_t {
        const math_t z = static_cast<math_t>(value);
        return static_cast<complex_t>(functor(z));
    });
}

template <typename F>
Tensor complex_math_kernel_cuda(const Tensor& self, F f) {
    const DType dt = self.dtype();
    TP_CHECK(isComplexType(dt),
             "complex_math_kernel_cuda: expected a complex dtype");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  dt, self.device());
    if (result.numel() == 0) return result;
    switch (dt) {
        case DType::ComplexHalf:
            complex_math_loop<tensorplay::complex<Half>,
                              tensorplay::complex<float>>(self, result, f);
            break;
        case DType::ComplexFloat:
            complex_math_loop<tensorplay::complex<float>,
                              tensorplay::complex<float>>(self, result, f);
            break;
        case DType::ComplexDouble:
            complex_math_loop<tensorplay::complex<double>,
                              tensorplay::complex<double>>(self, result, f);
            break;
        case DType::BComplex32:
            complex_math_loop<tensorplay::complex<BFloat16>,
                              tensorplay::complex<float>>(self, result, f);
            break;
        default:
            TP_THROW(NotImplementedError, "CUDA complex math: unsupported dtype");
    }
    return result;
}

#define CX_FUNCTOR(NAME, EXPR)                                                \
    struct NAME {                                                             \
        template <typename T>                                                 \
        __device__ tensorplay::complex<T> operator()(                          \
            tensorplay::complex<T> z) const {                                 \
            return EXPR;                                                      \
        }                                                                     \
    };
CX_FUNCTOR(CxExp, tensorplay_complex_math::exp(z))
CX_FUNCTOR(CxExpm1, tensorplay_complex_math::expm1(z))
CX_FUNCTOR(CxLog, tensorplay_complex_math::log(z))
CX_FUNCTOR(CxLog10, tensorplay_complex_math::log10(z))
CX_FUNCTOR(CxLog1p, tensorplay_complex_math::log1p(z))
CX_FUNCTOR(CxLog2, tensorplay_complex_math::log2(z))
CX_FUNCTOR(CxSqrt, tensorplay_complex_math::sqrt(z))
CX_FUNCTOR(CxRsqrt, static_cast<T>(1) / tensorplay_complex_math::sqrt(z))
CX_FUNCTOR(CxSin, tensorplay_complex_math::sin(z))
CX_FUNCTOR(CxCos, tensorplay_complex_math::cos(z))
CX_FUNCTOR(CxTan, tensorplay_complex_math::tan(z))
CX_FUNCTOR(CxAsin, tensorplay_complex_math::asin(z))
CX_FUNCTOR(CxAcos, tensorplay_complex_math::acos(z))
CX_FUNCTOR(CxAtan, tensorplay_complex_math::atan(z))
CX_FUNCTOR(CxSinh, tensorplay_complex_math::sinh(z))
CX_FUNCTOR(CxCosh, tensorplay_complex_math::cosh(z))
CX_FUNCTOR(CxTanh, tensorplay_complex_math::tanh(z))
CX_FUNCTOR(CxAsinh, tensorplay_complex_math::asinh(z))
CX_FUNCTOR(CxAcosh, tensorplay_complex_math::acosh(z))
CX_FUNCTOR(CxAtanh, tensorplay_complex_math::atanh(z))
CX_FUNCTOR(CxSigmoid,
           static_cast<T>(1) / (static_cast<T>(1) +
                                 tensorplay_complex_math::exp(-z)))
CX_FUNCTOR(CxRecip, static_cast<T>(1) / z)
CX_FUNCTOR(CxNeg, -z)
CX_FUNCTOR(CxSquare, z * z)
struct CxPowScalar {
    tensorplay::complex<double> exponent;
    explicit CxPowScalar(double value) : exponent(value, 0.0) {}
    explicit CxPowScalar(tensorplay::complex<double> value) : exponent(value) {}

    template <typename T>
    __device__ tensorplay::complex<T> operator()(
        tensorplay::complex<T> z) const {
        const tensorplay::complex<T> exp_value(
            static_cast<T>(exponent.real()), static_cast<T>(exponent.imag()));
        return tensorplay_complex_math::pow(z, exp_value);
    }
};
#undef CX_FUNCTOR

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
            gpu_kernel(iter, [functor] __host__ __device__(Half x) -> Half {
                return static_cast<Half>(functor(static_cast<float>(x)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __host__ __device__(BFloat16 x) -> BFloat16 {
                return static_cast<BFloat16>(functor(static_cast<float>(x)));
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __host__ __device__(float x) -> float {
                return functor(x);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __host__ __device__(double x) -> double {
                return functor(x);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA unary float op: Unsupported output dtype");
    }
    return result;
}

// float32 tensors never fall into slow double-precision device math.
struct ExpFunctor { template<typename T> __device__ T operator()(T x) const { return ::exp(x); } };
struct Expm1Functor { template<typename T> __device__ T operator()(T x) const { return ::expm1(x); } };
struct ErfFunctor { template<typename T> __device__ T operator()(T x) const { return ::erf(x); } };
struct ErfcFunctor { template<typename T> __device__ T operator()(T x) const { return ::erfc(x); } };
struct LogFunctor { template<typename T> __device__ T operator()(T x) const { return ::log(x); } };
struct Log10Functor { template<typename T> __device__ T operator()(T x) const { return ::log10(x); } };
struct Log1pFunctor { template<typename T> __device__ T operator()(T x) const { return ::log1p(x); } };
struct Log2Functor { template<typename T> __device__ T operator()(T x) const { return ::log2(x); } };
struct LgammaFunctor { template<typename T> __device__ T operator()(T x) const { return ::lgamma(x); } };
struct SqrtFunctor { template<typename T> __device__ T operator()(T x) const { return ::sqrt(x); } };
struct RsqrtFunctor { template<typename T> __device__ T operator()(T x) const { return ::rsqrt(x); } };
struct SinFunctor { template<typename T> __device__ T operator()(T x) const { return ::sin(x); } };
struct CosFunctor { template<typename T> __device__ T operator()(T x) const { return ::cos(x); } };
struct TanhFunctor { template<typename T> __device__ T operator()(T x) const { return ::tanh(x); } };
struct SigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        return static_cast<T>(1) / (static_cast<T>(1) + ::exp(-x));
    }
};
struct ReluFunctor { template<typename T> __device__ T operator()(T x) const { return x < T(0) ? T(0) : x; } };
struct GeluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        const T kAlpha = static_cast<T>(0.70710678118654752440);
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + ::erf(x * kAlpha));
    }
};
struct SiluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        return x / (static_cast<T>(1) + ::exp(-x));
    }
};
struct SiluBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        // sigmoid = 1 / (1 + ::exp(-x)); dy * sigmoid * (1 + x * (1 - sigmoid))
        const T one = static_cast<T>(1);
        const T s = one / (one + ::exp(-x));
        return dy * s * (one + x * (one - s));
    }
};
struct AcosFunctor { template<typename T> __device__ T operator()(T x) const { return ::acos(x); } };
struct AcoshFunctor { template<typename T> __device__ T operator()(T x) const { return ::acosh(x); } };
struct AsinFunctor { template<typename T> __device__ T operator()(T x) const { return ::asin(x); } };
struct AsinhFunctor { template<typename T> __device__ T operator()(T x) const { return ::asinh(x); } };
struct AtanFunctor { template<typename T> __device__ T operator()(T x) const { return ::atan(x); } };
struct AtanhFunctor { template<typename T> __device__ T operator()(T x) const { return ::atanh(x); } };
struct CeilFunctor { template<typename T> __device__ T operator()(T x) const { return ::ceil(x); } };
struct CoshFunctor { template<typename T> __device__ T operator()(T x) const { return ::cosh(x); } };
struct FloorFunctor { template<typename T> __device__ T operator()(T x) const { return ::floor(x); } };
struct RoundFunctor { template<typename T> __device__ T operator()(T x) const { return rint(x); } }; // rint matches round better in CUDA
struct SinhFunctor { template<typename T> __device__ T operator()(T x) const { return ::sinh(x); } };
struct TanFunctor { template<typename T> __device__ T operator()(T x) const { return ::tan(x); } };
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


template <typename T>
static inline __host__ __device__ std::enable_if_t<std::is_integral_v<T>, T>
ldexp_element(T x, T exponent) {
    return exponent >= static_cast<T>(8 * static_cast<int>(sizeof(T)))
        ? T(0) : static_cast<T>(x * (T(1) << exponent));
}

template <typename T>
static inline __host__ __device__ std::enable_if_t<!std::is_integral_v<T>, T>
ldexp_element(T x, T exponent) {
    return static_cast<T>(static_cast<double>(x) *
                          ::exp2(static_cast<double>(exponent)));
}

static DType result_type_with_scalar_cuda(const Tensor& t, const Scalar& s) {
    DType td = t.dtype();
    if (s.dtype() == DType::Bool) return td;
    if (isFloatingType(s.dtype())) {
        if (isFloatingType(td)) return td;
        return DType::Float32;
    }
    return td;
}

struct RreluWithNoiseTrainBackwardFunctor {
    template<typename T> __host__ __device__ T operator()(T dy, T noise) const {
        return dy * noise;
    }
};
struct RreluWithNoiseEvalBackwardFunctor {
    double slope_;
    explicit RreluWithNoiseEvalBackwardFunctor(double s) : slope_(s) {}
    template<typename T> __host__ __device__ T operator()(T dy, T x) const {
        return x >= static_cast<T>(0) ? dy : dy * static_cast<T>(slope_);
    }
};

struct RreluWithNoiseFunctor {
    double lower_, upper_;
    bool training_;
    RreluWithNoiseFunctor(double l, double u, bool t) : lower_(l), upper_(u), training_(t) {}
    template<typename T> __host__ __device__ T operator()(T x, T r) const {
        if (training_) return x <= static_cast<T>(0) ? x * r : x;
        T slope = static_cast<T>((lower_ + upper_) / 2.0);
        return x >= static_cast<T>(0) ? x : x * slope;
    }
};

} // namespace cuda
} // namespace tensorplay
