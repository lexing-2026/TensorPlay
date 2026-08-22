#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "TypePromotion.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include "Parallel.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>

#ifdef USE_ONEDNN
#include "dnnl.hpp"
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

// --- Unary Kernels ---

#ifdef USE_ONEDNN
void onednn_eltwise(const Tensor& src, Tensor& dst, dnnl::algorithm algo, float alpha = 0.0f, float beta = 0.0f) {
    auto& engine = OneDNNContext::get_engine();
    auto& stream = OneDNNContext::get_stream();

    // Create memory descriptors
    dnnl::memory::dims dims;
    for(auto d : src.shape()) dims.push_back(d);
    
    dnnl::memory::dims strides;
    for(auto s : src.strides()) strides.push_back(s);
    
    auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, strides);

    // Create primitive descriptor directly
    auto pd = dnnl::eltwise_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_inference,
        algo,
        md,
        md,
        alpha,
        beta);
    
    auto src_mem = dnnl::memory(md, engine, src.data_ptr());
    // If inplace, dst is src
    auto dst_mem = (src.data_ptr() == dst.data_ptr()) ? src_mem : dnnl::memory(md, engine, dst.data_ptr());

    dnnl::eltwise_forward(pd).execute(stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    stream.wait();
}
#endif

// Helper for operations that preserve dtype (e.g. abs, neg, square)
template<typename Func>
Tensor unary_op_kernel(const Tensor& self, Func func) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]); \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Helper for operations that promote integer to float (e.g. sin, cos, exp)
template<typename Func>
Tensor unary_float_op_kernel(const Tensor& self, Func func) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) {
        out_dtype = DType::Float32;
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    // We need to handle the case where input is int, output is float
    // And input is float, output is float
    
    if (isIntegralType(self.dtype())) {
        // Input int, Output float
        #define INT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            float* dst = result.data_ptr<float>(); \
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<float>(func(static_cast<float>(src[i]))); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(INT_CASE) // This macro covers floats too, but we filtered with if
            default: TP_THROW(TypeError, "Unsupported dtype");
        }
        #undef INT_CASE
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        // ATen alignment: reduced floating types compute in float (opmath_t)
        int64_t n = self.numel();
        if (self.dtype() == DType::Float16) {
            const Half* src = self_contig.data_ptr<Half>();
            Half* dst = result.data_ptr<Half>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<Half>(func(static_cast<float>(src[i])));
            });
        } else {
            const BFloat16* src = self_contig.data_ptr<BFloat16>();
            BFloat16* dst = result.data_ptr<BFloat16>();
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<BFloat16>(func(static_cast<float>(src[i])));
            });
        }
    } else {
        // Input float, Output float
        #define FLOAT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            ctype* dst = result.data_ptr<ctype>(); \
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            case DType::Float32: {
                 const float* src = self_contig.data_ptr<float>();
                 float* dst = result.data_ptr<float>();
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
                 for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
                 });
                 break;
            }
            case DType::Float64: {
                 const double* src = self_contig.data_ptr<double>();
                 double* dst = result.data_ptr<double>();
                 parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
                 for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
                 });
                 break;
            }
            default: TP_THROW(TypeError, "Unsupported dtype (expected float)");
        }
        #undef FLOAT_CASE
    }
    
    return result;
}

// Implementations

Tensor abs_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::abs(x);
        }
    });
}

Tensor neg_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
             return x; // neg(bool) in same dtype is weird, just return x to avoid warning
        } else {
             return -x;
        }
    });
}

Tensor square_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) { return x * x; });
}

Tensor sign_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
            return x ? 1 : 0;
        } else {
            using ctype = decltype(x);
            if (x > ctype(0)) return static_cast<ctype>(1);
            if (x < ctype(0)) return static_cast<ctype>(-1);
            return static_cast<ctype>(0);
        }
    });
}

Tensor floor_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::floor(x); });
}

Tensor ceil_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::ceil(x); });
}

Tensor round_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    // ATen alignment: round uses nearbyint (round-half-to-even), not roundf
    return unary_op_kernel(self, [](auto x) { return std::nearbyint(x); });
}

// Float ops

Tensor acos_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::acos(x); }); }
Tensor acosh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::acosh(x); }); }
Tensor asin_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::asin(x); }); }
Tensor asinh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::asinh(x); }); }
Tensor atan_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::atan(x); }); }
Tensor atanh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::atanh(x); }); }
Tensor cos_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::cos(x); }); }
Tensor cosh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::cosh(x); }); }
Tensor sin_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sin(x); }); }
Tensor sinh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sinh(x); }); }
Tensor tan_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::tan(x); }); }
Tensor tanh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::tanh(x); }); }
Tensor exp_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::exp(x); }); }
Tensor expm1_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::expm1(x); }); }
Tensor erf_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erf(x); }); }
Tensor erfc_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erfc(x); }); }
Tensor log_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log(x); }); }
Tensor log10_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log10(x); }); }
Tensor log1p_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log1p(x); }); }
Tensor log2_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log2(x); }); }
Tensor lgamma_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::lgamma(x); }); }
Tensor sqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sqrt(x); }); }
Tensor rsqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / std::sqrt(x); }); }
Tensor sigmoid_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)); }); }

Tensor frac_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) {
        TP_THROW(NotImplementedError, "frac is not implemented for integral tensors");
    }
    return unary_op_kernel(self, [](auto x) { return x - std::trunc(x); });
}

Tensor trunc_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::trunc(x); });
}

Tensor relu_kernel(const Tensor& self) {
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled() && self.dtype() == DType::Float32) {
        try {
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            onednn_eltwise(self, result, dnnl::algorithm::eltwise_relu);
            return result;
        } catch (const std::exception& e) {
            // std::cerr << "OneDNN relu failed, falling back: " << e.what() << std::endl;
        }
    }
    #endif

    // Optimized AVX2/AVX512 implementation for Float32
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
         Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
         int64_t n = self.numel();
         const float* src = self.data_ptr<float>();
         float* dst = result.data_ptr<float>();

         #if defined(__AVX512F__)
         __m512 zero = _mm512_setzero_ps();
         parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
         for (int64_t i = begin; i < end; i += 16) {
             if (i + 16 <= end) {
                 __m512 x = _mm512_loadu_ps(src + i);
                 _mm512_storeu_ps(dst + i, _mm512_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < end; ++j) dst[j] = (src[j] < 0.0f ? 0.0f : src[j]);
             }
         }
         });
         return result;
         #elif defined(__AVX2__)
         __m256 zero = _mm256_setzero_ps();
         parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
         for (int64_t i = begin; i < end; i += 8) {
             if (i + 8 <= end) {
                 __m256 x = _mm256_loadu_ps(src + i);
                 _mm256_storeu_ps(dst + i, _mm256_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < end; ++j) dst[j] = (src[j] < 0.0f ? 0.0f : src[j]);
             }
         }
         });
         return result;
         #endif
    }

    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            // clamp_min semantics: NaN propagates (matches torch.relu)
            return x < static_cast<T>(0) ? static_cast<T>(0) : x;
        }
    });
}

Tensor& relu_inplace_kernel(Tensor& self) {
    // OneDNN's eltwise primitive does not accept this tensor/layout as both
    // source and destination.  Trying it first only produces a noisy
    // exception before reaching the already-optimized direct in-place path.
    // Keep OneDNN for out-of-place ReLU, but use the SIMD/scalar path here.
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
         // Optimized path
         int64_t n = self.numel();
         float* data = self.data_ptr<float>();

         #if defined(__AVX512F__)
         __m512 zero = _mm512_setzero_ps();
         parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
         for (int64_t i = begin; i < end; i += 16) {
             if (i + 16 <= end) {
                 __m512 x = _mm512_loadu_ps(data + i);
                 _mm512_storeu_ps(data + i, _mm512_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < end; ++j) data[j] = (data[j] < 0.0f ? 0.0f : data[j]);
             }
         }
         });
         return self;
         #elif defined(__AVX2__)
         __m256 zero = _mm256_setzero_ps();
         parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
         for (int64_t i = begin; i < end; i += 8) {
             if (i + 8 <= end) {
                 __m256 x = _mm256_loadu_ps(data + i);
                 _mm256_storeu_ps(data + i, _mm256_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < end; ++j) data[j] = (data[j] < 0.0f ? 0.0f : data[j]);
             }
         }
         });
         return self;
         #else
         // Scalar fallback for contiguous float32
         for (int64_t i = 0; i < n; ++i) data[i] = (data[i] < 0.0f ? 0.0f : data[i]);
         return self;
         #endif
    }

    // Generic fallback
    int64_t n = self.numel();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            if constexpr (!std::is_unsigned_v<ctype>) { \
                data[i] = data[i] < static_cast<ctype>(0) ? static_cast<ctype>(0) : data[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: {
             // Debug info
             std::cerr << "Unsupported dtype: " << (int)self.dtype() << " for relu_inplace" << std::endl;
             TP_THROW(TypeError, "Unsupported dtype");
        }
    }
    #undef OP_CASE
    
    return self;
}

// Defined below; used by the public gelu entry points above them.
Tensor gelu_tanh_impl(const Tensor& self);
Tensor gelu_backward_impl(const Tensor& grad_output, const Tensor& self, const std::string& approximate);

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))); tanh approximation from
    // ATen cpu/Gelu.h scalar_gelu_approximated_with_tanh.
    if (approximate == "tanh") {
        return gelu_tanh_impl(self);
    } else if (approximate != "none") {
        TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    }
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440); // M_SQRT1_2
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x * kAlpha));
    });
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    return gelu_backward_impl(grad_output, self, approximate);
}

Tensor silu_kernel(const Tensor& self) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return x / (static_cast<T>(1) + std::exp(-x));
    });
}

// ---------------------------------------------------------------------------
// Activations.  The element-wise formulas are ported from ATen:
//   third_party/pytorch/aten/src/ATen/native/cpu/Activation.cpp
//     (hardsigmoid_kernel, hardtanh_backward_kernel, hardswish_kernel,
//      leaky_relu_kernel)
//   third_party/pytorch/aten/src/ATen/native/cpu/Gelu.h
//     (scalar_gelu_approximated_with_tanh)
//   third_party/pytorch/aten/src/ATen/native/cpu/Elu.h
//     (get_scalar_elu_elementwise_func)
//   third_party/pytorch/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
//     (GeluBackwardCUDAKernelImpl — the reference backward formulas)
//   third_party/pytorch/aten/src/ATen/native/cuda/ActivationMishKernel.cu
//     (MishBackwardCUDAKernelImpl)
//   third_party/pytorch/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu
//     (SoftplusBackwardCUDAKernelImpl)
// Reduced-precision inputs compute in float (opmath), matching ATen.
// ---------------------------------------------------------------------------
template<typename Func>
Tensor activation_backward_kernel(const Tensor& grad_output, const Tensor& self, Func func) {
    DType out_dtype = grad_output.dtype();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;

    Tensor grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    Tensor self_contig = self.is_contiguous() ? self : self.clone();

    #define BACKWARD_CASE(ctype, name) \
    case DType::name: { \
        const ctype* dy = grad_contig.data_ptr<ctype>(); \
        const ctype* x = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                dst[i] = static_cast<ctype>(func(static_cast<float>(dy[i]), static_cast<float>(x[i]))); \
            } \
        }); \
        break; \
    }
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(BACKWARD_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype for activation backward");
    }
    #undef BACKWARD_CASE
    return result;
}

// ATen cpu/Gelu.h: scalar_gelu_approximated_with_tanh + GeluCUDAKernelImpl 'none'
static inline float gelu_none_scalar(float x) {
    constexpr float kAlpha = 0.70710678118654752440f; // M_SQRT1_2
    return x * 0.5f * (1.0f + std::erf(x * kAlpha));
}
static inline float gelu_tanh_scalar(float x) {
    constexpr float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f; // M_SQRT2 * M_2_SQRTPI * 0.5
    constexpr float kKappa = 0.044715f;
    float x_cube = x * x * x;
    float inner = kBeta * (x + kKappa * x_cube);
    return 0.5f * x * (1.0f + std::tanh(inner));
}
static inline float gelu_backward_none_scalar(float dy, float x) {
    // ATen ActivationGeluKernel.cu GeluBackwardCUDAKernelImpl ('none'):
    //   kAlpha = M_SQRT1_2; kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
    //   cdf = 0.5*(1+erf(x*kAlpha)); pdf = kBeta*exp(-x*x*0.5); return dy*(cdf + x*pdf);
    constexpr float kAlpha = 0.70710678118654752440f;
    constexpr float kBeta = 1.12837916709551257390f * 0.70710678118654752440f * 0.5f;
    float cdf = 0.5f * (1.0f + std::erf(x * kAlpha));
    float pdf = kBeta * std::exp(x * x * -0.5f);
    return dy * (cdf + x * pdf);
}
static inline float gelu_backward_tanh_scalar(float dy, float x) {
    // ATen ActivationGeluKernel.cu GeluBackwardCUDAKernelImpl ('Tanh')
    constexpr float kBeta = 1.41421356237309504880f * 1.12837916709551257390f * 0.5f;
    constexpr float kKappa = 0.044715f;
    float x_sq = x * x;
    float x_cube = x_sq * x;
    float inner = kBeta * (x + kKappa * x_cube);
    float tanh_inner = std::tanh(inner);
    float left = 0.5f * x;
    float right = 1.0f + tanh_inner;
    float left_derivative = 0.5f * right;
    float tanh_derivative = 1.0f - tanh_inner * tanh_inner;
    float inner_derivative = kBeta * (1.0f + 3.0f * kKappa * x_sq);
    float right_derivative = left * tanh_derivative * inner_derivative;
    return dy * (left_derivative + right_derivative);
}

Tensor gelu_tanh_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return static_cast<T>(gelu_tanh_scalar(static_cast<float>(x)));
    });
}

Tensor gelu_backward_impl(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    if (approximate == "none") {
        return activation_backward_kernel(grad_output, self, gelu_backward_none_scalar);
    } else if (approximate == "tanh") {
        return activation_backward_kernel(grad_output, self, gelu_backward_tanh_scalar);
    }
    TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
}

Tensor hardtanh_kernel_impl(const Tensor& self, Scalar min_val, Scalar max_val) {
    // ATen Activation.cpp hardtanh: std::clamp(x, min_val, max_val)
    return unary_float_op_kernel(self, [min_val, max_val](auto x) {
        using T = decltype(x);
        T lo = static_cast<T>(min_val.toDouble());
        T hi = static_cast<T>(max_val.toDouble());
        return x < lo ? lo : (x > hi ? hi : x);
    });
}

Tensor hardtanh_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
    // ATen Activation.cpp (~line 714): (self <= min || self >= max) ? 0 : grad
    double lo = min_val.toDouble();
    double hi = max_val.toDouble();
    return activation_backward_kernel(grad_output, self,
        [lo, hi](float dy, float x) -> float { return (x <= lo || x >= hi) ? 0.0f : dy; });
}

Tensor relu6_kernel_impl(const Tensor& self) {
    // relu6 == hardtanh(0, 6) (torch.nn.functional.relu6 documentation)
    return hardtanh_kernel_impl(self, Scalar(0.0), Scalar(6.0));
}

Tensor hardswish_kernel_impl(const Tensor& self) {
    // ATen Activation.cpp hardswish_kernel: x * clamp(x + 3, 0, 6) / 6
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T clamped = (xf + T(3) < T(0)) ? T(0) : (xf + T(3) > T(6)) ? T(6) : xf + T(3);
        return xf * clamped / T(6);
    });
}

Tensor hardswish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    // ATen Activation.h hardswish_backward:
    //   x <= -3 -> 0 ; x >= 3 -> dy ; else dy * (x/6 + 0.5)
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return dy;
            return dy * (x / 6.0f + 0.5f);
        });
}

Tensor hardsigmoid_kernel_impl(const Tensor& self) {
    // ATen Activation.cpp hardsigmoid_kernel: clamp(x + 3, 0, 6) / 6
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T v = xf + T(3);
        v = v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
        return v / T(6);
    });
}

Tensor hardsigmoid_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    // ATen Activation.h hardsigmoid_backward:
    //   x <= -3 -> 0 ; x >= 3 -> 0 ; else dy * (x/6 + 0.5)
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            if (x <= -3.0f || x >= 3.0f) return 0.0f;
            return dy * (x / 6.0f + 0.5f);
        });
}

Tensor leaky_relu_kernel_impl(const Tensor& self, Scalar negative_slope) {
    // ATen Activation.cpp leaky_relu_kernel: x >= 0 ? x : negative_slope * x
    double slope = negative_slope.toDouble();
    return unary_float_op_kernel(self, [slope](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        return xf < T(0) ? static_cast<T>(slope) * xf : xf;
    });
}

Tensor leaky_relu_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar negative_slope, bool self_is_result) {
    // ATen Activation.cpp leaky_relu_backward_kernel: x > 0 ? grad : grad*negative_slope
    (void)self_is_result; // out-of-place call always receives the input itself
    double slope = negative_slope.toDouble();
    return activation_backward_kernel(grad_output, self,
        [slope](float dy, float x) -> float { return x > 0.0f ? dy : dy * static_cast<float>(slope); });
}

Tensor elu_kernel_impl(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
    // ATen cpu/Elu.h get_scalar_elu_elementwise_func:
    //   a < 0 ? expm1(a * input_scale) * negcoef : a * poscoef
    double negcoef = alpha.toDouble() * scale.toDouble();
    double poscoef = scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    return unary_float_op_kernel(self, [negcoef, poscoef, negiptcoef](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        return a < T(0)
            ? static_cast<T>(std::expm1(static_cast<float>(a) * static_cast<float>(negiptcoef)) * static_cast<float>(negcoef))
            : a * static_cast<T>(poscoef);
    });
}

Tensor elu_backward_kernel_impl(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, const Tensor& self_or_result) {
    // ATen cpu/Activation.cpp elu_backward_kernel (lines 213-265):
    //   is_result: b <= 0 ? a*negiptcoef*(b + negcoef) : a*poscoef
    //   else:      b <= 0 ? a*negiptcoef*negcoef*exp(b*negiptcoef) : a*poscoef
    double negcoef = alpha.toDouble() * scale.toDouble();
    double poscoef = scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    return activation_backward_kernel(grad_output, self_or_result,
        [negcoef, poscoef, negiptcoef, is_result](float dy, float b) -> float {
            return b <= 0.0f
                ? (is_result
                      ? dy * static_cast<float>(negiptcoef) * (b + static_cast<float>(negcoef))
                      : dy * static_cast<float>(negiptcoef) * static_cast<float>(negcoef) * std::exp(b * static_cast<float>(negiptcoef)))
                : dy * static_cast<float>(poscoef);
        });
}

Tensor mish_kernel_impl(const Tensor& self) {
    // ATen ActivationMishKernel: mish(x) = x * tanh(softplus(x))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T sp = std::log(T(1) + std::exp(xf));
        return xf * std::tanh(sp);
    });
}

Tensor mish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    // ATen ActivationMishKernel.cu MishBackwardCUDAKernelImpl:
    //   sp = log1p(exp(x)); tanh_sp = tanh(sp); sech2 = 1 - tanh_sp^2
    //   return dy * (tanh_sp + x * sech2 * sigmoid(x))
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            float sp = std::log1p(std::exp(x));
            float tanh_sp = std::tanh(sp);
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            float gsp = 1.0f / (1.0f + std::exp(-x));
            return dy * (tanh_sp + x * sech2 * gsp);
        });
}

Tensor selu_kernel_impl(const Tensor& self) {
    // ATen Activation.h selu constants:
    //   lambda_ = 1.0507009873554804934193349852946
    //   alpha_  = 1.6732632423543772848170429916717
    constexpr double lambda_ = 1.0507009873554804934193349852946;
    constexpr double alpha_ = 1.6732632423543772848170429916717;
    return unary_float_op_kernel(self, [lambda_, alpha_](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        return a > T(0) ? a * static_cast<T>(lambda_)
                        : static_cast<T>(alpha_ * lambda_) * std::expm1(a);
    });
}

Tensor celu_kernel_impl(const Tensor& self, Scalar alpha) {
    // ATen Activation.h celu: max(0,x) + min(0, alpha * expm1(x / alpha))
    double a = alpha.toDouble();
    return unary_float_op_kernel(self, [a](auto x) {
        using T = decltype(x);
        T af = static_cast<T>(static_cast<float>(x));
        return af > T(0) ? af : static_cast<T>(a) * (std::expm1(af / static_cast<T>(a)));
    });
}

Tensor softplus_kernel_impl(const Tensor& self, Scalar beta, Scalar threshold) {
    // ATen ActivationSoftplusKernel.cu SoftplusCUDAKernelImpl:
    //   beta_in * a > threshold ? a : log1p(exp(beta_in * a)) / beta_in
    double beta_in = beta.toDouble();
    double threshold_in = threshold.toDouble();
    return unary_float_op_kernel(self, [beta_in, threshold_in](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        T beta_in_t = static_cast<T>(beta_in);
        return a * beta_in_t > static_cast<T>(threshold_in)
            ? a
            : static_cast<T>(std::log1p(std::exp(static_cast<float>(a * beta_in_t))) / beta_in);
    });
}

Tensor softplus_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar beta, Scalar threshold) {
    // ATen ActivationSoftplusKernel.cu SoftplusBackwardCUDAKernelImpl:
    //   beta_in * a > threshold ? dy : dy * sigmoid(beta_in * a)
    double beta_in = beta.toDouble();
    double threshold_in = threshold.toDouble();
    return activation_backward_kernel(grad_output, self,
        [beta_in, threshold_in](float dy, float a) -> float {
            return a * static_cast<float>(beta_in) > static_cast<float>(threshold_in)
                ? dy
                : dy * (1.0f / (1.0f + std::exp(-a * static_cast<float>(beta_in))));
        });
}

Tensor pow_scalar_kernel(const Tensor& self, Scalar exponent) {
    // ATen alignment: pow_tensor_scalar_optimized_kernel
    if (self.dtype() == DType::Bool) TP_THROW(TypeError, "pow is not supported for bool tensors");
    if (isIntegralType(self.dtype()) && exponent.isIntegral() && exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    if (exponent.isFloatingPoint()) {
        double exp_val = exponent.toDouble();
        // Fast paths mirroring ATen pow_tensor_scalar_optimized_kernel
        if (exp_val == 0.5 && self.dtype() != DType::Float64) return sqrt_kernel(self);
        if (exp_val == -0.5 && self.dtype() != DType::Float64) return rsqrt_kernel(self);
        if (exp_val == 1.0) return self.clone();
        if (exp_val == 2.0) return square_kernel(self);
        if (exp_val == 3.0) {
            return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return x * x * x; });
        }
        return unary_float_op_kernel(self, [exp_val](auto x) { using T = decltype(x); return std::pow(x, static_cast<T>(exp_val)); });
    } else {
        int64_t exp_val = exponent.to<int64_t>();
        if (exp_val < 0) {
             return unary_float_op_kernel(self, [exp_val](auto x) { using T = decltype(x); return std::pow(x, static_cast<T>(static_cast<double>(exp_val))); });
        }
        return unary_op_kernel(self, [exp_val](auto x) {
             using T = decltype(x);
             // repeated multiplication (ipow), matches ATen integral behavior
             T base = x;
             T acc = static_cast<T>(1);
             int64_t e = exp_val;
             while (e > 0) {
                 if (e & 1) acc = acc * base;
                 e >>= 1;
                 if (e) base = base * base;
             }
             return acc;
        });
    }
}



Tensor angle_kernel(const Tensor& self) {
    // For real numbers, angle is 0 if >=0, pi if <0
    return unary_float_op_kernel(self, [](auto x) { 
        if (x >= 0) return 0.0;
        return 3.14159265358979323846; 
    });
}

// --- Binary/Ternary Kernels ---

// Helper for clamp
Tensor clamp_kernel(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    Tensor self_contig = self.is_contiguous() ? self : self.clone();

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype min_val = min.has_value() ? min->to<ctype>() : std::numeric_limits<ctype>::lowest(); \
        ctype max_val = max.has_value() ? max->to<ctype>() : std::numeric_limits<ctype>::max(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i=begin; i<end; ++i) { \
            ctype val = src[i]; \
            if (min.has_value() && val < min_val) val = min_val; \
            if (max.has_value() && val > max_val) val = max_val; \
            dst[i] = val; \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    return result;
}

// Helper for clamp backward
Tensor clamp_backward_kernel(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    Tensor grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        const ctype* grad = grad_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype min_val = min.has_value() ? min->to<ctype>() : std::numeric_limits<ctype>::lowest(); \
        ctype max_val = max.has_value() ? max->to<ctype>() : std::numeric_limits<ctype>::max(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            ctype val = src[i]; \
            if ((min.has_value() && val < min_val) || (max.has_value() && val > max_val)) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& output, Scalar threshold) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    
    Tensor output_contig = output.is_contiguous() ? output : output.clone();
    Tensor grad_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = output_contig.data_ptr<ctype>(); \
        const ctype* grad = grad_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype thresh = threshold.to<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
        for(int64_t i = begin; i < end; ++i) { \
            if (src[i] <= thresh) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
        }); \
        break; \
    }

    switch (output.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Softmax — ATen alignment: single fused kernel over the reduction dim
// (max pass, exp+sum pass, write pass) instead of materializing 5 temporaries.
// Fast path: contiguous input, reduction over last dim. Fallback: composition.
template <bool LogMode>
static Tensor softmax_fused_kernel_impl(const Tensor& self, int64_t dim, DType out_dtype) {
    Tensor input = self.to(out_dtype);
    int64_t d = dim < 0 ? dim + input.dim() : dim;

    bool innermost = input.is_contiguous() && (d == input.dim() - 1);
    if (!innermost) {
        // generic fallback via transpose-to-end + fused row loop
        Tensor t = input.transpose(d, -1);
        if (!t.is_contiguous()) t = t.contiguous();
        Tensor result = softmax_fused_kernel_impl<LogMode>(t, t.dim() - 1, out_dtype);
        return result.transpose(d, -1).contiguous();
    }

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()), out_dtype, input.device());
    int64_t rows = input.numel() / input.size(-1);
    int64_t size = input.size(-1);

    #define SOFTMAX_CASE(ctype, name) \
    case DType::name: { \
        const ctype* in = input.data_ptr<ctype>(); \
        ctype* out = result.data_ptr<ctype>(); \
        parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) { \
            for (int64_t r = begin; r < end; ++r) { \
                const ctype* row = in + r * size; \
                ctype* orow = out + r * size; \
                ctype m = row[0]; \
                for (int64_t j = 1; j < size; ++j) m = std::max(m, row[j]); \
                ctype sum = ctype(0); \
                for (int64_t j = 0; j < size; ++j) { \
                    ctype e = std::exp(row[j] - m); \
                    orow[j] = e; \
                    sum += e; \
                } \
                if constexpr (LogMode) { \
                    ctype lse = std::log(sum); \
                    for (int64_t j = 0; j < size; ++j) orow[j] = (row[j] - m) - lse; \
                } else { \
                    ctype inv = ctype(1) / sum; \
                    for (int64_t j = 0; j < size; ++j) orow[j] *= inv; \
                } \
            } \
        }); \
        break; \
    }
    switch (out_dtype) {
        SOFTMAX_CASE(float, Float32)
        SOFTMAX_CASE(double, Float64)
        default: TP_THROW(TypeError, "softmax: unsupported dtype");
    }
    #undef SOFTMAX_CASE
    return result;
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    DType out_dtype = dtype.value_or(self.dtype());
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    // ATen alignment: reduced floats compute in float32
    if (isReducedFloatingType(out_dtype)) {
        return softmax_fused_kernel_impl<false>(self, dim, DType::Float32).to(out_dtype);
    }
    return softmax_fused_kernel_impl<false>(self, dim, out_dtype);
}

// Log Softmax — same fused structure as ATen (_log_softmax_vec)
Tensor log_softmax_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    DType out_dtype = dtype.value_or(self.dtype());
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    if (isReducedFloatingType(out_dtype)) {
        return softmax_fused_kernel_impl<true>(self, dim, DType::Float32).to(out_dtype);
    }
    return softmax_fused_kernel_impl<true>(self, dim, out_dtype);
}

// Helper for pow (Tensor, Tensor)
Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(exponent.shape()));
    DType result_dtype = promoteTypes(self.dtype(), exponent.dtype());

    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

    Tensor self_c = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    Tensor exp_c = (exponent.dtype() == result_dtype) ? exponent : exponent.to(result_dtype);

    ti_apply_binary(result, self_c, exp_c,
        [](auto b, auto e) { return static_cast<decltype(b)>(std::pow(static_cast<double>(b), static_cast<double>(e))); });
    return result;
    
    return result;
}

// Lerp implementations using composition
Tensor lerp_tensor_kernel(const Tensor& self, const Tensor& end, const Tensor& weight) {
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    common_dtype = promoteTypes(common_dtype, weight.dtype());
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;

    // result = self + weight * (end - self)
    // Ensure all operands are cast to common_dtype
    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);
    Tensor w = weight.to(common_dtype);

    return s + w * (e - s);
}

Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end, Scalar weight) {
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) common_dtype = promoteTypes(common_dtype, DType::Float32);
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;

    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);

    // ATen alignment: numerically stable branch chosen once for a scalar weight
    double w = weight.toDouble();
    if (std::abs(w) < 0.5) {
        return s + weight * (e - s);
    }
    return e - (e - s) * (1.0 - w);
}

Tensor& lerp_scalar_inplace_kernel(Tensor& self, const Tensor& end, Scalar weight) {
    self.copy_(lerp_scalar_kernel(self, end, weight));
    return self;
}

Tensor& lerp_tensor_inplace_kernel(Tensor& self, const Tensor& end, const Tensor& weight) {
    self.copy_(lerp_tensor_kernel(self, end, weight));
    return self;
}

Tensor& abs_inplace_kernel(Tensor& self) {
    self.copy_(abs_kernel(self));
    return self;
}

Tensor& neg_inplace_kernel(Tensor& self) {
    self.copy_(neg_kernel(self));
    return self;
}

Tensor& sqrt_inplace_kernel(Tensor& self) {
    self.copy_(sqrt_kernel(self));
    return self;
}

Tensor& rsqrt_inplace_kernel(Tensor& self) {
    self.copy_(rsqrt_kernel(self));
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, PointwiseKernels) {
    m.impl("abs", abs_kernel);
    m.impl("neg", neg_kernel);
    m.impl("square", square_kernel);
    m.impl("sign", sign_kernel);
    m.impl("floor", floor_kernel);
    m.impl("ceil", ceil_kernel);
    m.impl("round", round_kernel);
    m.impl("acos", acos_kernel);
    m.impl("acosh", acosh_kernel);
    m.impl("asin", asin_kernel);
    m.impl("asinh", asinh_kernel);
    m.impl("atan", atan_kernel);
    m.impl("atanh", atanh_kernel);
    m.impl("cos", cos_kernel);
    m.impl("cosh", cosh_kernel);
    m.impl("sin", sin_kernel);
    m.impl("sinh", sinh_kernel);
    m.impl("tan", tan_kernel);
    m.impl("tanh", tanh_kernel);
    m.impl("exp", exp_kernel);
    m.impl("expm1", expm1_kernel);
    m.impl("erf", erf_kernel);
    m.impl("erfc", erfc_kernel);
    m.impl("log", log_kernel);
    m.impl("log10", log10_kernel);
    m.impl("log1p", log1p_kernel);
    m.impl("log2", log2_kernel);
    m.impl("lgamma", lgamma_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("frac", frac_kernel);
    m.impl("trunc", trunc_kernel);
    m.impl("sigmoid", sigmoid_kernel);
    m.impl("relu", relu_kernel);
    m.impl("relu_", relu_inplace_kernel);
    m.impl("gelu", gelu_kernel);
    m.impl("gelu_backward", gelu_backward_kernel);
    m.impl("silu", silu_kernel);
    // Activations — see the ATen citations above each kernel.
    m.impl("hardtanh", hardtanh_kernel_impl);
    m.impl("hardtanh_backward", hardtanh_backward_kernel_impl);
    m.impl("relu6", relu6_kernel_impl);
    m.impl("hardswish", hardswish_kernel_impl);
    m.impl("hardswish_backward", hardswish_backward_kernel_impl);
    m.impl("hardsigmoid", hardsigmoid_kernel_impl);
    m.impl("hardsigmoid_backward", hardsigmoid_backward_kernel_impl);
    m.impl("leaky_relu", leaky_relu_kernel_impl);
    m.impl("leaky_relu_backward", leaky_relu_backward_kernel_impl);
    m.impl("elu", elu_kernel_impl);
    m.impl("elu_backward", elu_backward_kernel_impl);
    m.impl("mish", mish_kernel_impl);
    m.impl("mish_backward", mish_backward_kernel_impl);
    m.impl("softplus", softplus_kernel_impl);
    m.impl("softplus_backward", softplus_backward_kernel_impl);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel);
    m.impl("angle", angle_kernel);
    m.impl("clamp", clamp_kernel);
    m.impl("clamp_backward", clamp_backward_kernel);
    m.impl("threshold_backward", threshold_backward_kernel);
    m.impl("softmax", softmax_kernel);
    m.impl("log_softmax", log_softmax_kernel);
    m.impl("pow.Tensor_Tensor", pow_tensor_tensor_kernel);
    m.impl("lerp", lerp_scalar_kernel);
    m.impl("lerp.Tensor", lerp_tensor_kernel);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel);
    m.impl("abs_", abs_inplace_kernel);
    m.impl("neg_", neg_inplace_kernel);
    m.impl("sqrt_", sqrt_inplace_kernel);
    m.impl("rsqrt_", rsqrt_inplace_kernel);
}

} // namespace cpu
} // namespace tensorplay
