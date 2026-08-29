#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "ErrorReporting.h"
#include "TensorIteratorOps.h"
#include "TypePromotion.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include "Parallel.h"
#include "cpu/VecUnary.h"
#include "cpu/ComplexUnary.h"
#include "cpu/VecComplex.h"
#include "cpu/ActivationUnaryKernels.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// DispatchStub instances for the tiered activation kernels; defined once here
// because the tier objects (TP_CPU_KERNEL_SRCS) each compile their own copy
// and would collide at link time.
namespace tensorplay { namespace cpu {
DEFINE_DISPATCH(sigmoid_f32_stub);
DEFINE_DISPATCH(silu_f32_stub);
DEFINE_DISPATCH(sigmoid_f64_stub);
DEFINE_DISPATCH(silu_f64_stub);
}} // namespace tensorplay::cpu

#ifdef USE_ONEDNN
#include "dnnl.hpp"
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
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

namespace {
// Only f32/f64 have vector kernels.  Templating the dispatch (instead of
// if-constexpr at the macro site) matters: the switch below expands with a
// concrete ctype inside this non-template function, and a discarded
// constexpr branch there would still be fully type-checked.
template <typename T>
inline void vec_run(vecunary::VOp op, const vecunary::VParams& prm,
                    const T* src, T* dst, int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<T, float>) {
        vecunary::run_f32(op, prm, src, dst, begin, end);
    } else if constexpr (std::is_same_v<T, double>) {
        vecunary::run_f64(op, prm, src, dst, begin, end);
    }
}
} // namespace

// Elementwise kernels use a finer grain than the global default: with the
// spinning intraop pool a chunk handoff costs ~1-2us, so splitting small-ish
// tensors across all workers wins far more than the handoff costs.
constexpr int64_t kUnaryGrain = 8192;

// Helper for operations that preserve dtype (e.g. abs, neg, square).
// vec_op selects the AVX2 fast path (see cpu/VecUnary.h) for float/double;
// the scalar lambda stays as the fallback for other dtypes and non-AVX2 hosts.
template<typename Func>
Tensor unary_op_kernel(const Tensor& self, Func func,
                       vecunary::VOp vec_op = vecunary::VOp::None,
                       vecunary::VParams vec_prm = {}) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();

    Tensor self_contig = self.contiguous();
    // Vector fast paths exist only for f32/f64; other dtypes take the
    // scalar-lambda fallback and must never instantiate the vec calls.
    const bool vec_ok = vecunary::vec_ready() && vec_op != vecunary::VOp::None
        && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64);

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        if (vec_ok) { \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
            vec_run(vec_op, vec_prm, src, dst, begin, end); \
            }); \
            break; \
        } \
        parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
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

// Helper for operations that promote integer to float (e.g. sin, cos, exp).
// vec_op selects the AVX2 fast path (see cpu/VecUnary.h) for float/double and
// the widen-compute-narrow paths for half/bfloat16; the scalar lambda remains
// as fallback.
template<typename Func>
Tensor unary_float_op_kernel(const Tensor& self, Func func,
                             vecunary::VOp vec_op = vecunary::VOp::None,
                             vecunary::VParams vec_prm = {}) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) {
        out_dtype = DType::Float32;
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();

    Tensor self_contig = self.contiguous();
    // Vector fast paths cover f32/f64 plus the widen-compute-narrow f16/bf16
    // kernels; integral inputs stay on the scalar-lambda fallback.
    const bool vec_ok = vecunary::vec_ready() && vec_op != vecunary::VOp::None
        && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64
            || self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16);

    if (isIntegralType(self.dtype())) {
        // Input int, Output float
        #define INT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            float* dst = result.data_ptr<float>(); \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
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
        int64_t n = self.numel();
        if (self.dtype() == DType::Float16) {
            const Half* src = self_contig.data_ptr<Half>();
            Half* dst = result.data_ptr<Half>();
            if (vec_ok && vecunary::f16c_available()) {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_f16(vec_op, vec_prm,
                                  reinterpret_cast<const uint16_t*>(src),
                                  reinterpret_cast<uint16_t*>(dst), begin, end);
                });
            } else {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<Half>(func(static_cast<float>(src[i])));
                });
            }
        } else {
            const BFloat16* src = self_contig.data_ptr<BFloat16>();
            BFloat16* dst = result.data_ptr<BFloat16>();
            if (vec_ok) {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                vecunary::run_bf16(vec_op, vec_prm,
                                   reinterpret_cast<const uint16_t*>(src),
                                   reinterpret_cast<uint16_t*>(dst), begin, end);
                });
            } else {
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for(int64_t i = begin; i < end; ++i) dst[i] = static_cast<BFloat16>(func(static_cast<float>(src[i])));
                });
            }
        }
    } else {
        // Input float, Output float
        #define FLOAT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            ctype* dst = result.data_ptr<ctype>(); \
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
            for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            case DType::Float32: {
                 const float* src = self_contig.data_ptr<float>();
                 float* dst = result.data_ptr<float>();
                 if (vec_ok) {
                     parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                     vecunary::run_f32(vec_op, vec_prm, src, dst, begin, end);
                     });
                     break;
                 }
                 parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                 for(int64_t i = begin; i < end; ++i) dst[i] = func(src[i]);
                 });
                 break;
            }
            case DType::Float64: {
                 const double* src = self_contig.data_ptr<double>();
                 double* dst = result.data_ptr<double>();
                 if (vec_ok) {
                     parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
                     vecunary::run_f64(vec_op, vec_prm, src, dst, begin, end);
                     });
                     break;
                 }
                 parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) { \
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

// corresponding real dtype (hypot(re, im)).
Tensor complex_abs_kernel(const Tensor& self) {
    DType out_dtype = toRealValueType(self.dtype());
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    const int64_t n = self.numel();
    Tensor self_contig = self.contiguous();

    switch (self.dtype()) {
        case DType::ComplexFloat: {
            using c_t = std::complex<float>;
            const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = std::hypot(src[i].real(), src[i].imag());
                }
            });
            break;
        }
        case DType::ComplexDouble: {
            using c_t = std::complex<double>;
            const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    dst[i] = std::hypot(src[i].real(), src[i].imag());
                }
            });
            break;
        }
        case DType::ComplexHalf:
        case DType::BComplex32: {
            // Reduced complexes compute the magnitude in float32.
            if (self.dtype() == DType::ComplexHalf) {
                const std::complex<Half>* src =
                    reinterpret_cast<const std::complex<Half>*>(self_contig.data_ptr());
                Half* dst = result.data_ptr<Half>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        float re = static_cast<float>(src[i].real());
                        float im = static_cast<float>(src[i].imag());
                        dst[i] = static_cast<Half>(std::hypot(re, im));
                    }
                });
            } else {
                const std::complex<BFloat16>* src =
                    reinterpret_cast<const std::complex<BFloat16>*>(self_contig.data_ptr());
                BFloat16* dst = result.data_ptr<BFloat16>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        float re = static_cast<float>(src[i].real());
                        float im = static_cast<float>(src[i].imag());
                        dst[i] = static_cast<BFloat16>(std::hypot(re, im));
                    }
                });
            }
            break;
        }
        default: TP_THROW(TypeError, "complex abs: unsupported dtype");
    }
    return result;
}

// Implementations

Tensor abs_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if ((self.dtype() == DType::ComplexFloat ||
             self.dtype() == DType::ComplexDouble) &&
            self.is_contiguous() && self.numel() > 0 &&
            veccomplex::avx2_available()) {
            Tensor out = Tensor::empty(
                static_cast<std::vector<int64_t>>(self.shape()),
                toRealValueType(self.dtype()), self.device());
            if (veccomplex::try_abs(self.data_ptr(), out.data_ptr(),
                                    self.numel(), self.dtype()))
                return out;
        }
        return complex_abs_kernel(self);
    }
    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::abs(x);
        }
    }, vecunary::VOp::Abs);
}

// Vectorized complex unary driver — defined below the float kernels; declared
// here because neg/square route through it.
template <typename F>
static Tensor cplx_unary_vec(const Tensor& self, veccomplex::Op op, F fb);

Tensor neg_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return cplx_unary_vec(self, veccomplex::Op::Neg,
                              [](auto x) { return -x; });
    }
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
             return x; // neg(bool) in same dtype is weird, just return x to avoid warning
        } else {
             return -x;
        }
    }, vecunary::VOp::Neg);
}

Tensor square_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return cplx_unary_vec(self, veccomplex::Op::Square,
                              [](auto x) { return x * x; });
    }
    return unary_op_kernel(self, [](auto x) { return x * x; }, vecunary::VOp::Square);
}

Tensor sign_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto x) {
            using C = decltype(x);
            using T = typename C::value_type;
            const T m = std::abs(x);
            if (m == T(0)) return C(T(0), T(0));
            return x / C(m, T(0));
        });
    }
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
            return x ? 1 : 0;
        } else {
            using ctype = decltype(x);
            if (x > ctype(0)) return static_cast<ctype>(1);
            if (x < ctype(0)) return static_cast<ctype>(-1);
            return static_cast<ctype>(0);
        }
    }, vecunary::VOp::Sign);
}

Tensor floor_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::floor(x); }, vecunary::VOp::Floor);
}

Tensor ceil_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::ceil(x); }, vecunary::VOp::Ceil);
}

Tensor round_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::nearbyint(x); }, vecunary::VOp::Round);
}

// Float ops
//
// the complex dtypes (see docs/source/complex_numbers.md).  Complex inputs
// route through complex_unary_op_kernel with std::complex math or the c10
// formulas above; real dtypes keep the vectorized paths.


// Vectorized complex unary fast path (cpu/VecComplex.h): contiguous
// Complex{Float,Double} inputs run the AVX2+libmvec cores; reduced complexes,
// non-contiguous input and pre-AVX2 hosts keep the scalar driver below.
template <typename F>
static Tensor cplx_unary_vec(const Tensor& self, veccomplex::Op op, F fb) {
    if ((self.dtype() == DType::ComplexFloat ||
         self.dtype() == DType::ComplexDouble) &&
        self.is_contiguous() && self.numel() > 0 &&
        veccomplex::unary_supported(op) && veccomplex::avx2_available()) {
        Tensor out = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        if (veccomplex::try_unary(self.data_ptr(), out.data_ptr(),
                                  self.numel(), self.dtype(), op))
            return out;
    }
    return complex_unary_op_kernel(self, fb);
}

Tensor acos_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Acos,
                              [](auto x) { return std::acos(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::acos(x); }, vecunary::VOp::Acos);
}
Tensor acosh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Acosh,
                              [](auto x) { return std::acosh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::acosh(x); }, vecunary::VOp::Acosh);
}
Tensor asin_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Asin,
                              [](auto x) { return std::asin(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::asin(x); }, vecunary::VOp::Asin);
}
Tensor asinh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Asinh,
                              [](auto x) { return std::asinh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::asinh(x); }, vecunary::VOp::Asinh);
}
Tensor atan_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Atan,
                              [](auto x) { return std::atan(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::atan(x); }, vecunary::VOp::Atan);
}
Tensor atanh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Atanh,
                              [](auto x) { return std::atanh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::atanh(x); }, vecunary::VOp::Atanh);
}
Tensor cos_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Cos,
                              [](auto x) { return std::cos(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::cos(x); }, vecunary::VOp::Cos);
}
Tensor cosh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Cosh,
                              [](auto x) { return std::cosh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::cosh(x); }, vecunary::VOp::Cosh);
}
Tensor sin_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sin,
                              [](auto x) { return std::sin(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sin(x); }, vecunary::VOp::Sin);
}
Tensor sinh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sinh,
                              [](auto x) { return std::sinh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sinh(x); }, vecunary::VOp::Sinh);
}
Tensor tan_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Tan,
                              [](auto x) { return std::tan(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::tan(x); }, vecunary::VOp::Tan);
}
Tensor tanh_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Tanh,
                              [](auto x) { return std::tanh(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::tanh(x); }, vecunary::VOp::Tanh);
}
Tensor exp_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Exp,
                              [](auto x) { return std::exp(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::exp(x); }, vecunary::VOp::Exp);
}
Tensor expm1_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Expm1,
                              [](auto x) { return cx_expm1(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::expm1(x); }, vecunary::VOp::Expm1);
}
Tensor erf_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erf(x); }, vecunary::VOp::Erf); }
Tensor erfc_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::erfc(x); }, vecunary::VOp::Erfc); }
Tensor log_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log,
                              [](auto x) { return std::log(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log(x); }, vecunary::VOp::Log);
}
Tensor log10_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log10,
                              [](auto x) { return std::log10(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log10(x); }, vecunary::VOp::Log10);
}
Tensor log1p_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log1p,
                              [](auto x) { return cx_log1p(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log1p(x); }, vecunary::VOp::Log1p);
}
Tensor log2_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Log2,
                              [](auto x) { return cx_log2(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::log2(x); }, vecunary::VOp::Log2);
}
Tensor lgamma_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::lgamma(x); }, vecunary::VOp::Lgamma); }
Tensor sqrt_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sqrt,
                              [](auto x) { return std::sqrt(x);  });
    return unary_float_op_kernel(self, [](auto x) { return std::sqrt(x); }, vecunary::VOp::Sqrt);
}
Tensor rsqrt_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Rsqrt,
                              [](auto x) { return cx_rsqrt(x);  });
    return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / std::sqrt(x); }, vecunary::VOp::Rsqrt);
}
Tensor sigmoid_kernel(const Tensor& self) {
    if (isComplexType(self.dtype()))
        return cplx_unary_vec(self, veccomplex::Op::Sigmoid,
                              [](auto x) { return cx_sigmoid(x);  });
    // Contiguous f32/f64 goes through the tier-compiled stub (three-tier
    // build, compile-time ISA selection); everything else -- non-contiguous
    // (normalized by clone below), f16/bf16 widen-compute-narrow, integral
    // promotion -- stays on the generic runtime-dispatched path.
    const DType dt = self.dtype();
    if (self.is_contiguous() &&
        (dt == DType::Float32 || dt == DType::Float64)) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dt, self.device());
        const int64_t n = self.numel();
        if (dt == DType::Float32) {
            const float* src = self.data_ptr<float>();
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                sigmoid_f32_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        } else {
            const double* src = self.data_ptr<double>();
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                sigmoid_f64_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        }
        return result;
    }
    return unary_float_op_kernel(self, [](auto x) { using T = decltype(x); return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x)); }, vecunary::VOp::Sigmoid);
}

Tensor frac_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) {
        TP_THROW(NotImplementedError, "frac is not implemented for integral tensors");
    }
    return unary_op_kernel(self, [](auto x) { return x - std::trunc(x); }, vecunary::VOp::Frac);
}

Tensor trunc_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::trunc(x); }, vecunary::VOp::Trunc);
}

Tensor relu_kernel(const Tensor& self) {
    // oneDNN eltwise rejected here (was: numel >= 4096): primitive
    // construction costs ~5us per call and measured *slower* than the native
    // kernels rather than oneDNN for ReLU.

    // Vectorized path for contiguous Float32 (see cpu/VecUnary.h).  The old
    // __AVX512F__/__AVX2__ blocks here were dead code: this TU compiles
    // without ISA flags, so dispatch goes through VecUnary's per-function
    // target attributes instead.  Non-AVX2 hosts fall through to the generic
    // kernel below.
    if (vecunary::vec_ready() && self.dtype() == DType::Float32 && self.is_contiguous()) {
         Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
         int64_t n = self.numel();
         const float* src = self.data_ptr<float>();
         float* dst = result.data_ptr<float>();
         parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
             vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, src, dst, begin, end);
         });
         return result;
    }

    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return x < static_cast<T>(0) ? static_cast<T>(0) : x;
        }
    });
}

Tensor& relu_inplace_kernel(Tensor& self) {
    // Keep OneDNN for out-of-place ReLU, but use the SIMD/scalar path here.
    // Vectorized path for contiguous Float32 (see cpu/VecUnary.h); the old
    // __AVX512F__/__AVX2__ blocks were dead code in this TU and the #else arm
    // was a serial full-tensor loop.  Non-AVX2 hosts fall through to the
    // parallel generic branch below.
    if (vecunary::vec_ready() && self.dtype() == DType::Float32 && self.is_contiguous()) {
         int64_t n = self.numel();
         float* data = self.data_ptr<float>();
         parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
             vecunary::run_f32(vecunary::VOp::Relu, vecunary::VParams{}, data, data, begin, end);
         });
         return self;
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
    if (approximate == "tanh") {
        return gelu_tanh_impl(self);
    } else if (approximate != "none") {
        TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    }
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440); // M_SQRT1_2
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x * kAlpha));
    }, vecunary::VOp::GeluNone);
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    return gelu_backward_impl(grad_output, self, approximate);
}

Tensor silu_kernel(const Tensor& self) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const DType dt = self.dtype();
    if (self.is_contiguous() &&
        (dt == DType::Float32 || dt == DType::Float64)) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dt, self.device());
        const int64_t n = self.numel();
        if (dt == DType::Float32) {
            const float* src = self.data_ptr<float>();
            float* dst = result.data_ptr<float>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                silu_f32_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        } else {
            const double* src = self.data_ptr<double>();
            double* dst = result.data_ptr<double>();
            parallel_for(0, n, kUnaryGrain, [&](int64_t b, int64_t e) {
                silu_f64_stub(DeviceType::CPU, src + b, dst + b, e - b);
            });
        }
        return result;
    }
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return x / (static_cast<T>(1) + std::exp(-x));
    }, vecunary::VOp::Silu);
}

// Fused gated activation primitives.  These belong with the existing SiLU
// native/GatedLinearUnit.cpp rather than introducing an LLM-specific kernel
// bucket.  The packed form follows the decoder convention [gate | up].
namespace {

inline void check_silu_mul_inputs(const Tensor& gate, const Tensor& up,
                                  const char* op) {
    if (gate.device() != up.device()) {
        TP_THROW(DeviceMismatchError, op,
                 ": gate and up must be on the same device");
    }
    if (gate.shape() != up.shape()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same shape");
    }
    if (gate.dtype() != up.dtype()) {
        TP_THROW(RuntimeError, op, ": gate and up must have the same dtype");
    }
    if (!isFloatingType(gate.dtype())) {
        TP_THROW(NotImplementedError, op,
                 ": only floating point dtypes are supported");
    }
}

template <typename T, typename Acc>
void silu_mul_loop(const T* gate, const T* up, T* output, int64_t n) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const Acc x = static_cast<Acc>(gate[i]);
            const Acc y = static_cast<Acc>(up[i]);
            const Acc sigmoid = Acc(1) / (Acc(1) + std::exp(-x));
            output[i] = static_cast<T>(x * sigmoid * y);
        }
    });
}

template <typename T, typename Acc>
void silu_and_mul_loop(const T* input, T* output, int64_t n,
                       int64_t half_width) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t row = i / half_width;
            const int64_t col = i - row * half_width;
            const int64_t base = row * (2 * half_width);
            const Acc gate = static_cast<Acc>(input[base + col]);
            const Acc up = static_cast<Acc>(input[base + half_width + col]);
            const Acc sigmoid = Acc(1) / (Acc(1) + std::exp(-gate));
            output[i] = static_cast<T>(gate * sigmoid * up);
        }
    });
}

template <typename T>
Tensor silu_mul_typed(const Tensor& gate, const Tensor& up) {
    Tensor gate_c = gate.is_contiguous() ? gate : gate.contiguous();
    Tensor up_c = up.is_contiguous() ? up : up.contiguous();
    Tensor output = Tensor::empty(
        static_cast<std::vector<int64_t>>(gate_c.shape()), gate_c.dtype(),
        gate_c.device());
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    silu_mul_loop<T, Acc>(gate_c.data_ptr<T>(), up_c.data_ptr<T>(),
                          output.data_ptr<T>(), gate_c.numel());
    return output;
}

template <typename T>
Tensor silu_and_mul_typed(const Tensor& input) {
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> output_shape =
        static_cast<std::vector<int64_t>>(input_c.shape());
    const int64_t packed_width = output_shape.back();
    output_shape.back() = packed_width / 2;
    Tensor output = Tensor::empty(output_shape, input_c.dtype(),
                                  input_c.device());
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    silu_and_mul_loop<T, Acc>(input_c.data_ptr<T>(), output.data_ptr<T>(),
                              input_c.numel() / 2, packed_width / 2);
    return output;
}

} // namespace

Tensor silu_mul_cpu(const Tensor& gate, const Tensor& up) {
    check_silu_mul_inputs(gate, up, "silu_mul");
    switch (gate.dtype()) {
        case DType::Float32:
            return silu_mul_typed<float>(gate, up);
        case DType::Float64:
            return silu_mul_typed<double>(gate, up);
        case DType::Float16:
            return silu_mul_typed<Half>(gate, up);
        case DType::BFloat16:
            return silu_mul_typed<BFloat16>(gate, up);
        default:
            TP_THROW(NotImplementedError, "silu_mul: unsupported dtype");
    }
}

Tensor fused_swiglu_cpu(const Tensor& gate, const Tensor& up) {
    return silu_mul_cpu(gate, up);
}

Tensor silu_and_mul_cpu(const Tensor& input) {
    if (input.dim() < 1) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: input must have at least one dimension");
    }
    const int64_t width = input.size(-1);
    if ((width & 1) != 0) {
        TP_THROW(RuntimeError,
                 "silu_and_mul: the packed last dimension must be even");
    }
    if (!isFloatingType(input.dtype())) {
        TP_THROW(NotImplementedError,
                 "silu_and_mul: only floating point dtypes are supported");
    }
    switch (input.dtype()) {
        case DType::Float32:
            return silu_and_mul_typed<float>(input);
        case DType::Float64:
            return silu_and_mul_typed<double>(input);
        case DType::Float16:
            return silu_and_mul_typed<Half>(input);
        case DType::BFloat16:
            return silu_and_mul_typed<BFloat16>(input);
        default:
            TP_THROW(NotImplementedError, "silu_and_mul: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
//     (hardsigmoid_kernel, hardtanh_backward_kernel, hardswish_kernel,
//      leaky_relu_kernel)
//     (scalar_gelu_approximated_with_tanh)
//     (get_scalar_elu_elementwise_func)
//     (GeluBackwardCUDAKernelImpl — the reference backward formulas)
//     (MishBackwardCUDAKernelImpl)
//     (SoftplusBackwardCUDAKernelImpl)
// ---------------------------------------------------------------------------
template<typename Func>
Tensor activation_backward_kernel(const Tensor& grad_output, const Tensor& self, Func func) {
    DType out_dtype = grad_output.dtype();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;

    Tensor grad_contig = grad_output.contiguous();
    Tensor self_contig = self.contiguous();

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
    //   kAlpha = M_SQRT1_2; kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
    //   cdf = 0.5*(1+erf(x*kAlpha)); pdf = kBeta*exp(-x*x*0.5); return dy*(cdf + x*pdf);
    constexpr float kAlpha = 0.70710678118654752440f;
    constexpr float kBeta = 1.12837916709551257390f * 0.70710678118654752440f * 0.5f;
    float cdf = 0.5f * (1.0f + std::erf(x * kAlpha));
    float pdf = kBeta * std::exp(x * x * -0.5f);
    return dy * (cdf + x * pdf);
}
static inline float gelu_backward_tanh_scalar(float dy, float x) {
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
    }, vecunary::VOp::GeluTanh);
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
    vecunary::VParams prm;
    prm.p0 = min_val.toDouble();
    prm.p1 = max_val.toDouble();
    return unary_float_op_kernel(self, [min_val, max_val](auto x) {
        using T = decltype(x);
        T lo = static_cast<T>(min_val.toDouble());
        T hi = static_cast<T>(max_val.toDouble());
        return x < lo ? lo : (x > hi ? hi : x);
    }, vecunary::VOp::Hardtanh, prm);
}

Tensor hardtanh_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
    double lo = min_val.toDouble();
    double hi = max_val.toDouble();
    return activation_backward_kernel(grad_output, self,
        [lo, hi](float dy, float x) -> float { return (x <= lo || x >= hi) ? 0.0f : dy; });
}

Tensor relu6_kernel_impl(const Tensor& self) {
    return hardtanh_kernel_impl(self, Scalar(0.0), Scalar(6.0));
}

Tensor hardswish_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T clamped = (xf + T(3) < T(0)) ? T(0) : (xf + T(3) > T(6)) ? T(6) : xf + T(3);
        return xf * clamped / T(6);
    }, vecunary::VOp::Hardswish);
}

Tensor hardswish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   x <= -3 -> 0 ; x >= 3 -> dy ; else dy * (x/6 + 0.5)
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return dy;
            return dy * (x / 6.0f + 0.5f);
        });
}

Tensor silu_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   sigmoid = 1 / (1 + exp(-x)); dy * sigmoid * (1 + x * (1 - sigmoid))
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            const float s = 1.0f / (1.0f + std::exp(-x));
            return dy * s * (1.0f + x * (1.0f - s));
        });
}

Tensor hardsigmoid_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T v = xf + T(3);
        v = v < T(0) ? T(0) : (v > T(6) ? T(6) : v);
        return v / T(6);
    }, vecunary::VOp::Hardsigmoid);
}

Tensor hardsigmoid_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   x <= -3 -> 0 ; x >= 3 -> 0 ; else dy * (x/6 + 0.5)
    return activation_backward_kernel(grad_output, self,
        [](float dy, float x) -> float {
            if (x <= -3.0f || x >= 3.0f) return 0.0f;
            return dy * (x / 6.0f + 0.5f);
        });
}

Tensor leaky_relu_kernel_impl(const Tensor& self, Scalar negative_slope) {
    double slope = negative_slope.toDouble();
    vecunary::VParams prm;
    prm.p0 = slope;
    return unary_float_op_kernel(self, [slope](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        return xf < T(0) ? static_cast<T>(slope) * xf : xf;
    }, vecunary::VOp::LeakyRelu, prm);
}

Tensor leaky_relu_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar negative_slope, bool self_is_result) {
    (void)self_is_result; // out-of-place call always receives the input itself
    double slope = negative_slope.toDouble();
    return activation_backward_kernel(grad_output, self,
        [slope](float dy, float x) -> float { return x > 0.0f ? dy : dy * static_cast<float>(slope); });
}

Tensor elu_kernel_impl(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
    //   a < 0 ? expm1(a * input_scale) * negcoef : a * poscoef
    double negcoef = alpha.toDouble() * scale.toDouble();
    double poscoef = scale.toDouble();
    double negiptcoef = input_scale.toDouble();
    vecunary::VParams prm; // p0=alpha*scale, p1=scale, p2=input_scale
    prm.p0 = negcoef;
    prm.p1 = poscoef;
    prm.p2 = negiptcoef;
    return unary_float_op_kernel(self, [negcoef, poscoef, negiptcoef](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        return a < T(0)
            ? static_cast<T>(std::expm1(static_cast<float>(a) * static_cast<float>(negiptcoef)) * static_cast<float>(negcoef))
            : a * static_cast<T>(poscoef);
    }, vecunary::VOp::Elu, prm);
}

Tensor elu_backward_kernel_impl(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, const Tensor& self_or_result) {
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
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T xf = static_cast<T>(static_cast<float>(x));
        T sp = std::log(T(1) + std::exp(xf));
        return xf * std::tanh(sp);
    }, vecunary::VOp::Mish);
}

Tensor mish_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
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
    //   lambda_ = 1.0507009873554804934193349852946
    //   alpha_  = 1.6732632423543772848170429916717
    constexpr double lambda_ = 1.0507009873554804934193349852946;
    constexpr double alpha_ = 1.6732632423543772848170429916717;
    return unary_float_op_kernel(self, [lambda_, alpha_](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        return a > T(0) ? a * static_cast<T>(lambda_)
                        : static_cast<T>(alpha_ * lambda_) * std::expm1(a);
    }, vecunary::VOp::Selu);
}

Tensor celu_kernel_impl(const Tensor& self, Scalar alpha) {
    double a = alpha.toDouble();
    vecunary::VParams prm;
    prm.p0 = a;
    return unary_float_op_kernel(self, [a](auto x) {
        using T = decltype(x);
        T af = static_cast<T>(static_cast<float>(x));
        return af > T(0) ? af : static_cast<T>(a) * (std::expm1(af / static_cast<T>(a)));
    }, vecunary::VOp::Celu, prm);
}

Tensor softplus_kernel_impl(const Tensor& self, Scalar beta, Scalar threshold) {
    //   beta_in * a > threshold ? a : log1p(exp(beta_in * a)) / beta_in
    double beta_in = beta.toDouble();
    double threshold_in = threshold.toDouble();
    vecunary::VParams prm; // p0=beta_in, p1=threshold_in
    prm.p0 = beta_in;
    prm.p1 = threshold_in;
    return unary_float_op_kernel(self, [beta_in, threshold_in](auto x) {
        using T = decltype(x);
        T a = static_cast<T>(static_cast<float>(x));
        T beta_in_t = static_cast<T>(beta_in);
        return a * beta_in_t > static_cast<T>(threshold_in)
            ? a
            : static_cast<T>(std::log1p(std::exp(static_cast<float>(a * beta_in_t))) / beta_in);
    }, vecunary::VOp::Softplus, prm);
}

Tensor softplus_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, Scalar beta, Scalar threshold) {
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

// ---------------------------------------------------------------------------
// log_sigmoid_cpu_kernel): out = min(x, 0) - log1p(exp(-|x|)).  The branch
// split keeps exp() bounded for both large-positive and large-negative inputs.
// ---------------------------------------------------------------------------
Tensor log_sigmoid_kernel_impl(const Tensor& self) {
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        T z = std::min(x, static_cast<T>(0));
        return static_cast<T>(z - std::log1p(std::exp(-std::abs(x))));
    });
}

Tensor log_sigmoid_backward_kernel_impl(const Tensor& grad_output, const Tensor& self) {
    //   grad * sigmoid(-x), branch-split so exp() never overflows:
    //     x >= 0: grad * exp(-x) / (1 + exp(-x))
    //     x <  0: grad / (1 + exp(x))
    // Computed in the storage dtype (f16/bf16 widen to float opmath) so that
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()),
                                  grad_output.dtype(), grad_output.device());
    const int64_t n = grad_output.numel();
    if (n == 0) return result;
    const Tensor gc = grad_output.contiguous();
    const Tensor sc = self.contiguous();
    #define LSIG_BWD_CASE(ctype, name) \
    case DType::name: { \
        const ctype* gp = gc.data_ptr<ctype>(); \
        const ctype* xp = sc.data_ptr<ctype>(); \
        ctype* yp = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                const ctype dy = gp[i]; \
                const ctype x = xp[i]; \
                yp[i] = x >= ctype(0) \
                    ? dy * (std::exp(-x) / (ctype(1) + std::exp(-x))) \
                    : dy / (ctype(1) + std::exp(x)); \
            } \
        }); \
        break; \
    }
    switch (grad_output.dtype()) {
        LSIG_BWD_CASE(float, Float32)
        LSIG_BWD_CASE(double, Float64)
        case DType::Float16:
        case DType::BFloat16: {
            if (grad_output.dtype() == DType::Float16) {
                const Half* gp = gc.data_ptr<Half>();
                const Half* xp = sc.data_ptr<Half>();
                Half* yp = result.data_ptr<Half>();
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        const float dy = static_cast<float>(gp[i]);
                        const float x = static_cast<float>(xp[i]);
                        const float v = x >= 0.0f
                            ? dy * (std::exp(-x) / (1.0f + std::exp(-x)))
                            : dy / (1.0f + std::exp(x));
                        yp[i] = Half(v);
                    }
                });
            } else {
                const BFloat16* gp = gc.data_ptr<BFloat16>();
                const BFloat16* xp = sc.data_ptr<BFloat16>();
                BFloat16* yp = result.data_ptr<BFloat16>();
                parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        const float dy = static_cast<float>(gp[i]);
                        const float x = static_cast<float>(xp[i]);
                        const float v = x >= 0.0f
                            ? dy * (std::exp(-x) / (1.0f + std::exp(-x)))
                            : dy / (1.0f + std::exp(x));
                        yp[i] = BFloat16(v);
                    }
                });
            }
            break;
        }
        default: TP_THROW(TypeError, "Unsupported dtype for log_sigmoid_backward");
    }
    #undef LSIG_BWD_CASE
    return result;
}

// ---------------------------------------------------------------------------
// negative elements by the (caller-provided) noise tensor, eval is leaky_relu
// TensorPlay kernels consume the noise the caller generated (nn.functional.rrelu
// draws it with rand), which keeps the kernel deterministic and RNG-free.
// ---------------------------------------------------------------------------
template <typename Func>
static Tensor binary_float_kernel(const Tensor& a, const Tensor& b, Func func) {
    if (a.shape() != b.shape())
        TP_THROW(RuntimeError, "rrelu_with_noise: expected noise to have the same shape as input");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(a.shape()), a.dtype(), a.device());
    int64_t n = a.numel();
    if (n == 0) return result;
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();
    #define RRELU_BIN_CASE(ctype, name) \
    case DType::name: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* yp = result.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                yp[i] = static_cast<ctype>(func(static_cast<float>(ap[i]), static_cast<float>(bp[i]))); \
            } \
        }); \
        break; \
    }
    switch (a.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(RRELU_BIN_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype for rrelu_with_noise");
    }
    #undef RRELU_BIN_CASE
    return result;
}

Tensor rrelu_with_noise_kernel_impl(const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training) {
    const float slope = static_cast<float>((lower.toDouble() + upper.toDouble()) / 2.0);
    if (training) {
        return binary_float_kernel(self, noise, [](float x, float r) -> float {
            return x <= 0.0f ? x * r : x;
        });
    }
    return binary_float_kernel(self, noise, [slope](float x, float) -> float {
        return x >= 0.0f ? x : x * slope;
    });
}

Tensor rrelu_with_noise_backward_kernel_impl(const Tensor& grad_output, const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
    // forward overwrites noise with 1 on positive elements, which lets its
    // backward be a plain noise*grad; this kernel leaves the caller's noise
    // untouched, so the training branch masks with self instead (same value).
    const float slope = static_cast<float>((lower.toDouble() + upper.toDouble()) / 2.0);
    if (training) {
        if (grad_output.shape() != self.shape() || grad_output.shape() != noise.shape())
            TP_THROW(RuntimeError, "rrelu_with_noise_backward: shape mismatch");
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()),
                                      grad_output.dtype(), grad_output.device());
        const int64_t n = grad_output.numel();
        if (n == 0) return result;
        const Tensor gc = grad_output.contiguous();
        const Tensor sc = self.contiguous();
        const Tensor nc = noise.contiguous();
        #define RRELU_TERN_CASE(ctype, name) \
        case DType::name: { \
            const ctype* gp = gc.data_ptr<ctype>(); \
            const ctype* sp = sc.data_ptr<ctype>(); \
            const ctype* np = nc.data_ptr<ctype>(); \
            ctype* yp = result.data_ptr<ctype>(); \
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
                for (int64_t i = begin; i < end; ++i) { \
                    const float x = static_cast<float>(sp[i]); \
                    yp[i] = static_cast<ctype>(x <= 0.0f \
                        ? static_cast<float>(gp[i]) * static_cast<float>(np[i]) \
                        : static_cast<float>(gp[i])); \
                } \
            }); \
            break; \
        }
        switch (grad_output.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(RRELU_TERN_CASE)
            default: TP_THROW(TypeError, "Unsupported dtype for rrelu_with_noise_backward");
        }
        #undef RRELU_TERN_CASE
        return result;
    }
    (void)self_is_result; // result >= 0 iff self >= 0 for a positive slope.
    return binary_float_kernel(grad_output, self, [slope](float dy, float x) -> float {
        return x >= 0.0f ? dy : dy * slope;
    });
}

Tensor pow_scalar_kernel(const Tensor& self, Scalar exponent) {
    if (self.dtype() == DType::Bool) TP_THROW(TypeError, "pow is not supported for bool tensors");
    if (isComplexType(self.dtype()) || exponent.isComplex()) {
        // scalar both produce complex results.  Negative integer exponents
        // are fine over complex.
        DType base_dt = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype()) ? toComplexType(self.dtype())
                                            : DType::ComplexFloat);
        DType result_dtype = promoteTypes(base_dt,
            isComplexType(exponent.dtype()) ? exponent.dtype() : toComplexType(base_dt));
        Tensor base = self.to(result_dtype);
        if (!isComplexType(exponent.dtype())) {
            double ev = exponent.toDouble();
            if (ev == 0.5) return sqrt_kernel(base);
            if (ev == -0.5) return rsqrt_kernel(base);
            if (ev == 1.0) return base.clone();
            if (ev == 2.0) return square_kernel(base);
            if (ev == 3.0) return complex_unary_op_kernel(base, [](auto x) { return x * x * x; });
            return complex_unary_op_kernel(base, [ev](auto x) {
                using V = typename decltype(x)::value_type;
                return std::pow(x, static_cast<V>(ev));
            });
        }
        // Complex exponent: promote the scalar into the result dtype once.
        if (result_dtype == DType::ComplexDouble) {
            auto e = exponent.to<std::complex<double>>();
            return complex_unary_op_kernel(base, [e](auto x) {
                using V = typename decltype(x)::value_type;
                return std::pow(x, static_cast<std::complex<V>>(e));
            });
        }
        auto e = exponent.to<std::complex<float>>();
        return complex_unary_op_kernel(base, [e](auto x) {
            using V = typename decltype(x)::value_type;
            return std::pow(x, static_cast<std::complex<V>>(e));
        });
    }
    if (isIntegralType(self.dtype()) && exponent.isIntegral() && exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    if (exponent.isFloatingPoint()) {
        double exp_val = exponent.toDouble();
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
    if (isComplexType(self.dtype())) {
        if ((self.dtype() == DType::ComplexFloat ||
             self.dtype() == DType::ComplexDouble) &&
            self.is_contiguous() && self.numel() > 0 &&
            veccomplex::avx2_available()) {
            Tensor out = Tensor::empty(
                static_cast<std::vector<int64_t>>(self.shape()),
                toRealValueType(self.dtype()), self.device());
            if (veccomplex::try_angle(self.data_ptr(), out.data_ptr(),
                                      self.numel(), self.dtype()))
                return out;
        }
        DType out_dtype = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
        const int64_t n = self.numel();
        Tensor self_contig = self.contiguous();
        switch (self.dtype()) {
            case DType::ComplexHalf:
            case DType::BComplex32: {
                // Reduced complexes compute in float32.
                if (self.dtype() == DType::ComplexHalf) {
                    const std::complex<Half>* src =
                        reinterpret_cast<const std::complex<Half>*>(self_contig.data_ptr());
                    Half* dst = result.data_ptr<Half>();
                    parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            dst[i] = static_cast<Half>(std::atan2(
                                static_cast<float>(src[i].imag()),
                                static_cast<float>(src[i].real())));
                        }
                    });
                } else {
                    const std::complex<BFloat16>* src =
                        reinterpret_cast<const std::complex<BFloat16>*>(self_contig.data_ptr());
                    BFloat16* dst = result.data_ptr<BFloat16>();
                    parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                        for (int64_t i = begin; i < end; ++i) {
                            dst[i] = static_cast<BFloat16>(std::atan2(
                                static_cast<float>(src[i].imag()),
                                static_cast<float>(src[i].real())));
                        }
                    });
                }
                break;
            }
            case DType::ComplexFloat: {
                using c_t = std::complex<float>;
                const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
                float* dst = result.data_ptr<float>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        dst[i] = std::atan2(src[i].imag(), src[i].real());
                    }
                });
                break;
            }
            case DType::ComplexDouble: {
                using c_t = std::complex<double>;
                const c_t* src = reinterpret_cast<const c_t*>(self_contig.data_ptr());
                double* dst = result.data_ptr<double>();
                parallel_for(0, n, kUnaryGrain, [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; ++i) {
                        dst[i] = std::atan2(src[i].imag(), src[i].real());
                    }
                });
                break;
            }
            default: TP_THROW(TypeError, "angle: unsupported dtype");
        }
        return result;
    }
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
    Tensor self_contig = self.contiguous();

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

// clamp(self, bound, nullopt); delegate to the same kernel here.
Tensor clamp_min_kernel(const Tensor& self, Scalar min) {
    return clamp_kernel(self, min, std::nullopt);
}
Tensor clamp_max_kernel(const Tensor& self, Scalar max) {
    return clamp_kernel(self, std::nullopt, max);
}
Tensor clamp_min__kernel(Tensor& self, Scalar min) {
    self.copy_(clamp_kernel(self, min, std::nullopt));
    return self;
}
Tensor clamp_max__kernel(Tensor& self, Scalar max) {
    self.copy_(clamp_kernel(self, std::nullopt, max));
    return self;
}

// Helper for clamp backward
Tensor clamp_backward_kernel(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    
    Tensor self_contig = self.contiguous();
    Tensor grad_contig = grad_output.contiguous();
    
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
    
    Tensor output_contig = output.contiguous();
    Tensor grad_contig = grad_output.contiguous();
    
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

// (max pass, exp+sum pass, write pass) instead of materializing 5 temporaries.
// Fast path: contiguous input, reduction over last dim. Fallback: composition.
template <bool LogMode>
static Tensor softmax_fused_kernel_impl(const Tensor& self, int64_t dim, DType out_dtype) {
    Tensor input = self.to(out_dtype);
    int64_t d = dim < 0 ? dim + input.dim() : dim;
    if (d < 0 || d >= input.dim()) {
        TP_THROW(IndexError, format_dim_range(input.dim(), dim));
    }

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

Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    DType out_dtype = (dtype != DType::Undefined) ? dtype : self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    if (isReducedFloatingType(out_dtype)) {
        return softmax_fused_kernel_impl<false>(self, dim, DType::Float32).to(out_dtype);
    }
    return softmax_fused_kernel_impl<false>(self, dim, out_dtype);
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
    DType out_dtype = (dtype != DType::Undefined) ? dtype : self.dtype();
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

    if (isComplexType(result_dtype)) {
        // and narrow back (std::<complex> internals assume a real float type).
        if (result_dtype == DType::ComplexHalf || result_dtype == DType::BComplex32) {
            return pow_tensor_tensor_kernel(self.to(DType::ComplexFloat),
                                            exponent.to(DType::ComplexFloat))
                .to(result_dtype);
        }
        ti_apply_arith(result, self_c, exp_c,
            [](auto b, auto e) {
                using B = decltype(b);
                if constexpr (is_complex_type_v<B>) {
                    using V = typename B::value_type;
                    if constexpr (std::is_same_v<V, float> || std::is_same_v<V, double>) {
                        return std::pow(b, e);
                    } else {
                        // Reduced complexes are unreachable at runtime
                        // (promoted above); instantiate in float so the
                        // generic lambda is valid for every iterator slot.
                        using F = std::complex<float>;
                        const auto r =
                            std::pow(F(static_cast<float>(b.real()), static_cast<float>(b.imag())),
                                     F(static_cast<float>(e.real()), static_cast<float>(e.imag())));
                        return B(static_cast<V>(r.real()), static_cast<V>(r.imag()));
                    }
                } else {
                    return static_cast<B>(std::pow(static_cast<double>(b), static_cast<double>(e)));
                }
            });
        return result;
    }

    ti_apply_binary(result, self_c, exp_c,
        [](auto b, auto e) { return static_cast<decltype(b)>(std::pow(static_cast<double>(b), static_cast<double>(e))); });
    return result;
}

// Lerp implementations using composition
template <typename T, typename W>
inline T lerp_scalar_value(T self, T end, W weight) {
    using compute_t = std::conditional_t<
        std::is_same_v<T, double>, double, float>;
    const compute_t s = static_cast<compute_t>(self);
    const compute_t e = static_cast<compute_t>(end);
    const compute_t w = static_cast<compute_t>(weight);
    const compute_t value = std::abs(w) < compute_t(0.5)
        ? s + w * (e - s)
        : e - (e - s) * (compute_t(1) - w);
    return static_cast<T>(value);
}

inline bool lerp_same_shape(const Tensor& self, const Tensor& end) {
    if (self.dim() != end.dim()) return false;
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (self.size(d) != end.size(d)) return false;
    }
    return true;
}

#if defined(__x86_64__)
namespace {

inline bool pointwise_cpu_has_avx512() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("fma") != 0;
    return ok;
}

inline bool pointwise_cpu_has_avx512_bf16() {
    static const bool ok = pointwise_cpu_has_avx512() &&
                           __builtin_cpu_supports("avx512bf16") != 0;
    return ok;
}

__attribute__((target("avx512f,fma")))
void lerp_f32_avx512(const float* self, const float* end, float* result,
                     int64_t n, float weight) {
    const __m512 w = _mm512_set1_ps(weight);
    const __m512 coeff = std::abs(weight) < 0.5f
        ? w : _mm512_sub_ps(w, _mm512_set1_ps(1.0f));
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 s = _mm512_loadu_ps(self + i);
        const __m512 e = _mm512_loadu_ps(end + i);
        const __m512 b = std::abs(weight) < 0.5f ? s : e;
        _mm512_storeu_ps(result + i,
                         _mm512_fmadd_ps(coeff, _mm512_sub_ps(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

__attribute__((target("avx512f,fma")))
void lerp_f64_avx512(const double* self, const double* end, double* result,
                     int64_t n, double weight) {
    const __m512d w = _mm512_set1_pd(weight);
    const __m512d coeff = std::abs(weight) < 0.5
        ? w : _mm512_sub_pd(w, _mm512_set1_pd(1.0));
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d s = _mm512_loadu_pd(self + i);
        const __m512d e = _mm512_loadu_pd(end + i);
        const __m512d b = std::abs(weight) < 0.5 ? s : e;
        _mm512_storeu_pd(result + i,
                         _mm512_fmadd_pd(coeff, _mm512_sub_pd(e, s), b));
    }
    for (; i < n; ++i) {
        result[i] = lerp_scalar_value(self[i], end[i], weight);
    }
}

__attribute__((target("avx512f,avx512bf16,fma")))
void lerp_bf16_avx512(const uint16_t* self, const uint16_t* end,
                      uint16_t* result, int64_t n, float weight) {
    const __m512 w = _mm512_set1_ps(weight);
    const bool small = std::abs(weight) < 0.5f;
    const __m512 coeff = small ? w : _mm512_sub_ps(w, _mm512_set1_ps(1.0f));
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i sr = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(self + i));
        const __m256i er = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(end + i));
        const __m512 s = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(sr), 16));
        const __m512 e = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(er), 16));
        const __m512 base = small ? s : e;
        const __m512 out = _mm512_fmadd_ps(coeff, _mm512_sub_ps(e, s), base);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i),
                            (__m256i)_mm512_cvtneps_pbh(out));
    }
    for (; i < n; ++i) {
        const float s = detail::bfloat16_to_float_bits(self[i]);
        const float e = detail::bfloat16_to_float_bits(end[i]);
        result[i] = detail::float_to_bfloat16_bits(
            lerp_scalar_value(s, e, weight));
    }
}

} // namespace
#endif

template <typename T, typename W>
void lerp_scalar_contiguous(const T* self, const T* end, T* result,
                            int64_t n, W weight) {
#if defined(__x86_64__)
    if constexpr (std::is_same_v<T, float>) {
        if (pointwise_cpu_has_avx512()) {
            const float w = static_cast<float>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f32_avx512(self + begin, end + begin, result + begin,
                                finish - begin, w);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (pointwise_cpu_has_avx512()) {
            const double w = static_cast<double>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_f64_avx512(self + begin, end + begin, result + begin,
                                finish - begin, w);
            });
            return;
        }
    } else if constexpr (std::is_same_v<T, BFloat16>) {
        if (pointwise_cpu_has_avx512_bf16()) {
            const auto* s = reinterpret_cast<const uint16_t*>(self);
            const auto* e = reinterpret_cast<const uint16_t*>(end);
            auto* r = reinterpret_cast<uint16_t*>(result);
            const float w = static_cast<float>(weight);
            parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
                lerp_bf16_avx512(s + begin, e + begin, r + begin,
                                 finish - begin, w);
            });
            return;
        }
    }
#endif
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t finish) {
        for (int64_t i = begin; i < finish; ++i) {
            result[i] = lerp_scalar_value(self[i], end[i], weight);
        }
    });
}

// TensorIterator's generic lerp composition is several full-tensor passes
// for reduced floating types.  Muon calls scalar-weight lerp twice per
// both operands are already dense and have the same dtype.  The general
// broadcasting/promotion path below remains the semantic fallback.
template <typename W>
bool lerp_scalar_fast(const Tensor& self, const Tensor& end, Tensor& result,
                      W weight) {
    if (!self.is_contiguous() || !end.is_contiguous() ||
        self.dtype() != end.dtype() || !lerp_same_shape(self, end)) {
        return false;
    }
    const int64_t n = self.numel();
    switch (self.dtype()) {
        case DType::Float16:
            lerp_scalar_contiguous(self.data_ptr<Half>(), end.data_ptr<Half>(),
                                   result.data_ptr<Half>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::BFloat16:
            lerp_scalar_contiguous(self.data_ptr<BFloat16>(),
                                   end.data_ptr<BFloat16>(),
                                   result.data_ptr<BFloat16>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::Float32:
            lerp_scalar_contiguous(self.data_ptr<float>(), end.data_ptr<float>(),
                                   result.data_ptr<float>(), n,
                                   static_cast<float>(weight));
            return true;
        case DType::Float64:
            lerp_scalar_contiguous(self.data_ptr<double>(),
                                   end.data_ptr<double>(),
                                   result.data_ptr<double>(), n,
                                   static_cast<double>(weight));
            return true;
        default:
            return false;
    }
}

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
    if (self.dtype() == end.dtype() && lerp_same_shape(self, end) &&
        self.is_contiguous() && end.is_contiguous() &&
        (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
         self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        // Native lerp keeps the reduced floating output dtype even though its
        // arithmetic is performed in float32.  This also avoids the three
        // temporary tensors used by the generic composition above.
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                      self.dtype(), self.device());
        if (lerp_scalar_fast(self, end, result, weight.to<double>())) {
            return result;
        }
    }

    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) common_dtype = promoteTypes(common_dtype, DType::Float32);
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;

    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);

    double w = weight.toDouble();
    if (std::abs(w) < 0.5) {
        return s + weight * (e - s);
    }
    return e - (e - s) * (1.0 - w);
}

Tensor& lerp_scalar_inplace_kernel(Tensor& self, const Tensor& end, Scalar weight) {
    if (self.dtype() == end.dtype() && lerp_same_shape(self, end) &&
        self.is_contiguous() && end.is_contiguous() &&
        (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16 ||
         self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        if (lerp_scalar_fast(self, end, self, weight.to<double>())) {
            return self;
        }
    }
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
    m.impl("silu_backward", silu_backward_kernel_impl);
    m.impl("silu_mul", silu_mul_cpu);
    m.impl("fused_swiglu", fused_swiglu_cpu);
    m.impl("silu_and_mul", silu_and_mul_cpu);
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
    m.impl("log_sigmoid", log_sigmoid_kernel_impl);
    m.impl("log_sigmoid_backward", log_sigmoid_backward_kernel_impl);
    m.impl("rrelu_with_noise", rrelu_with_noise_kernel_impl);
    m.impl("rrelu_with_noise_backward", rrelu_with_noise_backward_kernel_impl);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel);
    m.impl("angle", angle_kernel);
    m.impl("clamp", clamp_kernel);
    m.impl("clamp_min", clamp_min_kernel);
    m.impl("clamp_max", clamp_max_kernel);
    m.impl("clamp_min_", clamp_min__kernel);
    m.impl("clamp_max_", clamp_max__kernel);
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
