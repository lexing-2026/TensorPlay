#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OneDNNContext.h"
#include "Allocator.h"
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
        _Pragma("omp parallel for") \
        for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]); \
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
            _Pragma("omp parallel for") \
            for(int64_t i=0; i<n; ++i) dst[i] = static_cast<float>(func(static_cast<float>(src[i]))); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(INT_CASE) // This macro covers floats too, but we filtered with if
            default: TP_THROW(TypeError, "Unsupported dtype");
        }
        #undef INT_CASE
    } else {
        // Input float, Output float
        #define FLOAT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            ctype* dst = result.data_ptr<ctype>(); \
            _Pragma("omp parallel for") \
            for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]); \
            break; \
        }
        switch (self.dtype()) {
            case DType::Float32: {
                 const float* src = self_contig.data_ptr<float>();
                 float* dst = result.data_ptr<float>();
                 _Pragma("omp parallel for") \
                 for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]);
                 break;
            }
            case DType::Float64: {
                 const double* src = self_contig.data_ptr<double>();
                 double* dst = result.data_ptr<double>();
                 _Pragma("omp parallel for") \
                 for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]);
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
            if (x > 0) return 1;
            if (x < 0) return -1;
            return 0;
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
    return unary_op_kernel(self, [](auto x) { return std::round(x); });
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
Tensor log_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log(x); }); }
Tensor sqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sqrt(x); }); }
Tensor rsqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return 1.0 / std::sqrt(x); }); }
Tensor sigmoid_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return 1.0 / (1.0 + std::exp(-x)); }); }

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
         #ifdef _OPENMP
         #pragma omp parallel for
         #endif
         for (int64_t i = 0; i < n; i += 16) {
             if (i + 16 <= n) {
                 __m512 x = _mm512_loadu_ps(src + i);
                 _mm512_storeu_ps(dst + i, _mm512_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < n; ++j) dst[j] = std::max(0.0f, src[j]);
             }
         }
         return result;
         #elif defined(__AVX2__)
         __m256 zero = _mm256_setzero_ps();
         #ifdef _OPENMP
         #pragma omp parallel for
         #endif
         for (int64_t i = 0; i < n; i += 8) {
             if (i + 8 <= n) {
                 __m256 x = _mm256_loadu_ps(src + i);
                 _mm256_storeu_ps(dst + i, _mm256_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < n; ++j) dst[j] = std::max(0.0f, src[j]);
             }
         }
         return result;
         #endif
    }

    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::max(static_cast<T>(0), x);
        }
    });
}

Tensor& relu_inplace_kernel(Tensor& self) {
    // std::cout << "DEBUG: relu_inplace_kernel entry. dtype=" << (int)self.dtype() << " contiguous=" << self.is_contiguous() << std::endl;
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled() && self.dtype() == DType::Float32) {
        try {
            onednn_eltwise(self, self, dnnl::algorithm::eltwise_relu);
            return self;
        } catch (const std::exception& e) {
            std::cerr << "OneDNN relu_inplace failed, falling back: " << e.what() << std::endl;
        }
    }
    #endif

    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
         // Optimized path
         int64_t n = self.numel();
         float* data = self.data_ptr<float>();

         #if defined(__AVX512F__)
         __m512 zero = _mm512_setzero_ps();
         #ifdef _OPENMP
         #pragma omp parallel for
         #endif
         for (int64_t i = 0; i < n; i += 16) {
             if (i + 16 <= n) {
                 __m512 x = _mm512_loadu_ps(data + i);
                 _mm512_storeu_ps(data + i, _mm512_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < n; ++j) data[j] = std::max(0.0f, data[j]);
             }
         }
         return self;
         #elif defined(__AVX2__)
         __m256 zero = _mm256_setzero_ps();
         #ifdef _OPENMP
         #pragma omp parallel for
         #endif
         for (int64_t i = 0; i < n; i += 8) {
             if (i + 8 <= n) {
                 __m256 x = _mm256_loadu_ps(data + i);
                 _mm256_storeu_ps(data + i, _mm256_max_ps(zero, x));
             } else {
                 for (int64_t j = i; j < n; ++j) data[j] = std::max(0.0f, data[j]);
             }
         }
         return self;
         #else
         // Scalar fallback for contiguous float32
         for (int64_t i = 0; i < n; ++i) data[i] = std::max(0.0f, data[i]);
         return self;
         #endif
    }

    // Generic fallback
    int64_t n = self.numel();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype* data = self.data_ptr<ctype>(); \
        _Pragma("omp parallel for") \
        for(int64_t i=0; i<n; ++i) { \
            if constexpr (!std::is_unsigned_v<ctype>) { \
                data[i] = std::max(static_cast<ctype>(0), data[i]); \
            } \
        } \
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

Tensor gelu_kernel(const Tensor& self) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440); // 1/sqrt(2)
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x * kAlpha));
    });
}

Tensor silu_kernel(const Tensor& self) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return x / (static_cast<T>(1) + std::exp(-x));
    });
}

Tensor pow_scalar_kernel(const Tensor& self, Scalar exponent) {
    if (exponent.isFloatingPoint()) {
        double exp_val = exponent.toDouble();
        return unary_float_op_kernel(self, [exp_val](auto x) { return std::pow(x, exp_val); });
    } else {
        int64_t exp_val = exponent.to<int64_t>();
        if (exp_val < 0) {
             return unary_float_op_kernel(self, [exp_val](auto x) { return std::pow(x, static_cast<double>(exp_val)); });
        }
        return unary_op_kernel(self, [exp_val](auto x) {
             using T = decltype(x);
             return static_cast<T>(std::pow(x, exp_val));
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
        for(int64_t i=0; i<n; ++i) { \
            ctype val = src[i]; \
            if (min.has_value() && val < min_val) val = min_val; \
            if (max.has_value() && val > max_val) val = max_val; \
            dst[i] = val; \
        } \
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
        _Pragma("omp parallel for") \
        for(int64_t i=0; i<n; ++i) { \
            ctype val = src[i]; \
            if ((min.has_value() && val < min_val) || (max.has_value() && val > max_val)) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
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
        _Pragma("omp parallel for") \
        for(int64_t i=0; i<n; ++i) { \
            if (src[i] <= thresh) { \
                dst[i] = 0; \
            } else { \
                dst[i] = grad[i]; \
            } \
        } \
        break; \
    }

    switch (output.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Softmax
Tensor softmax_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    DType out_dtype = dtype.value_or(self.dtype());
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    
    Tensor input = self.to(out_dtype);
    
    Tensor max_val = input.max({dim}, true);
    Tensor shifted = input - max_val;
    Tensor exp_val = shifted.exp();
    Tensor sum_exp = exp_val.sum({dim}, true);
    return exp_val / sum_exp;
}

// Log Softmax
Tensor log_softmax_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    DType out_dtype = dtype.value_or(self.dtype());
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    
    Tensor input = self.to(out_dtype);
    
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    Tensor max_val = input.max({dim}, true);
    Tensor shifted = input - max_val;
    Tensor exp_val = shifted.exp();
    Tensor sum_exp = exp_val.sum({dim}, true);
    Tensor log_sum_exp = sum_exp.log();
    return shifted - log_sum_exp;
}

// Helper for pow (Tensor, Tensor)
Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(exponent.shape()));
    DType result_dtype = promoteTypes(self.dtype(), exponent.dtype());
    
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    
    Tensor self_expanded = self.expand(out_shape);
    Tensor exp_expanded = exponent.expand(out_shape);
    
    if (self_expanded.dtype() != result_dtype) self_expanded = self_expanded.to(result_dtype);
    if (exp_expanded.dtype() != result_dtype) exp_expanded = exp_expanded.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [](ctype b, ctype e) -> ctype { \
             return static_cast<ctype>(std::pow(static_cast<double>(b), static_cast<double>(e))); \
        }; \
        apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                 self_expanded, self_expanded.strides(), \
                                 exp_expanded, exp_expanded.strides(), \
                                 0, 0, 0, 0, out_shape, op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "pow: unsupported dtype");
    }
    #undef OP_CASE
    
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
    
    return s + weight * (e - s);
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
    m.impl("log", log_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("sigmoid", sigmoid_kernel);
    m.impl("relu", relu_kernel);
    m.impl("relu_", relu_inplace_kernel);
    m.impl("gelu", gelu_kernel);
    m.impl("silu", silu_kernel);
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
}

} // namespace cpu
} // namespace tensorplay
