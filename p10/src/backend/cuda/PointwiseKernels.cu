#include "QuantTensorOperators.h"
// Pointwise CUDA kernels: core unary family.
#include "PointwiseCommon.cuh"

namespace tensorplay {
namespace cuda {

Tensor abs_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        DType out_dt = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), out_dt,
            self.device());
        if (result.numel() == 0) return result;
        switch (self.dtype()) {
            case DType::ComplexHalf:
                complex_abs_loop<tensorplay::complex<Half>,
                                 tensorplay::complex<float>, Half>(self, result);
                break;
            case DType::ComplexFloat:
                complex_abs_loop<tensorplay::complex<float>,
                                 tensorplay::complex<float>, float>(self, result);
                break;
            case DType::ComplexDouble:
                complex_abs_loop<tensorplay::complex<double>,
                                 tensorplay::complex<double>, double>(self, result);
                break;
            case DType::BComplex32:
                complex_abs_loop<tensorplay::complex<BFloat16>,
                                 tensorplay::complex<float>, BFloat16>(self, result);
                break;
            default:
                TP_THROW(NotImplementedError, "CUDA abs: unsupported complex dtype");
        }
        return result;
    }
    return unary_op_kernel_v2(self, AbsFunctor());
}
template <typename complex_t, typename math_t>
void complex_sign_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __host__ __device__(complex_t value) -> complex_t {
        const math_t z = static_cast<math_t>(value);
        const auto real = z.real();
        const auto imag = z.imag();
        const auto magnitude = ::sqrt(real * real + imag * imag);
        if (magnitude == 0) return complex_t(0, 0);
        return static_cast<complex_t>(z / math_t(magnitude, 0));
    });
}

Tensor sign_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        if (self.numel() == 0) return result;
        switch (self.dtype()) {
            case DType::ComplexHalf:
                complex_sign_loop<tensorplay::complex<Half>,
                                  tensorplay::complex<float>>(self, result);
                break;
            case DType::ComplexFloat:
                complex_sign_loop<tensorplay::complex<float>,
                                  tensorplay::complex<float>>(self, result);
                break;
            case DType::ComplexDouble:
                complex_sign_loop<tensorplay::complex<double>,
                                  tensorplay::complex<double>>(self, result);
                break;
            case DType::BComplex32:
                complex_sign_loop<tensorplay::complex<BFloat16>,
                                  tensorplay::complex<float>>(self, result);
                break;
            default:
                TP_THROW(NotImplementedError, "CUDA sign: unsupported complex dtype");
        }
        return result;
    }
    return unary_op_kernel_v2(self, SignFunctor());
}

Tensor neg_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxNeg{});
    return unary_op_kernel_v2(self, NegFunctor());
}
Tensor square_kernel_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxSquare{});
    return unary_op_kernel_v2(self, SquareFunctor());
}

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
    if (isQuantizedType(self.dtype())) return quantized::activation(self, 5);

    if (isComplexType(self.dtype())) return complex_math_kernel_cuda(self, CxTanh{});
    return unary_float_op_kernel_v2(self, TanhFunctor());
}
Tensor sigmoid_kernel_cuda(const Tensor& self) {
    if (isQuantizedType(self.dtype())) return quantized::activation(self, 4);

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
        DType out_dt = toRealValueType(self.dtype());
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), out_dt,
            self.device());
        if (result.numel() == 0) return result;
        switch (self.dtype()) {
            case DType::ComplexHalf:
                complex_angle_loop<tensorplay::complex<Half>,
                                   tensorplay::complex<float>, Half>(self, result);
                break;
            case DType::ComplexFloat:
                complex_angle_loop<tensorplay::complex<float>,
                                   tensorplay::complex<float>, float>(self, result);
                break;
            case DType::ComplexDouble:
                complex_angle_loop<tensorplay::complex<double>,
                                   tensorplay::complex<double>, double>(self, result);
                break;
            case DType::BComplex32:
                complex_angle_loop<tensorplay::complex<BFloat16>,
                                   tensorplay::complex<float>, BFloat16>(self, result);
                break;
            default:
                TP_THROW(NotImplementedError, "CUDA angle: unsupported complex dtype");
        }
        return result;
    }
    return unary_float_op_kernel_v2(self, AngleFunctor());
}

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

// --- out= entries ---
// The composite out= wrappers compute a fresh tensor and copy it into the
// caller's buffer; these run the same body straight into a destination that
// already matches what the allocating form produces.  A destination that
// needs a resize, a cast, or that overlaps an operand in an unsafe way keeps
// the compute-and-hand-over route inside the helpers.

#define TP_KEEP_UNARY_OUT(op, FunctorType)                                    \
    Tensor& op##_out_cuda(const Tensor& self, Tensor& out) {                  \
        if (isQuantizedType(self.dtype()) || isComplexType(self.dtype())) {   \
            return write_out(out, op##_kernel_cuda(self));                    \
        }                                                                     \
        return unary_op_kernel_out(self, out, FunctorType());                 \
    }

#define TP_FLOAT_UNARY_OUT(op, FunctorType)                                   \
    Tensor& op##_out_cuda(const Tensor& self, Tensor& out) {                  \
        if (isQuantizedType(self.dtype()) || isComplexType(self.dtype())) {   \
            return write_out(out, op##_kernel_cuda(self));                    \
        }                                                                     \
        return unary_float_op_kernel_out(self, out, FunctorType());           \
    }

TP_KEEP_UNARY_OUT(abs, AbsFunctor)
TP_KEEP_UNARY_OUT(neg, NegFunctor)
TP_KEEP_UNARY_OUT(square, SquareFunctor)
TP_KEEP_UNARY_OUT(sign, SignFunctor)
TP_KEEP_UNARY_OUT(trunc, TruncFunctor)

Tensor& frac_out_cuda(const Tensor& self, Tensor& out) {
    if (isIntegralType(self.dtype()) || isComplexType(self.dtype())) {
        return write_out(out, frac_kernel_cuda(self));
    }
    return unary_float_op_kernel_out(self, out, FracFunctor());
}

TP_FLOAT_UNARY_OUT(acos, AcosFunctor)
TP_FLOAT_UNARY_OUT(acosh, AcoshFunctor)
TP_FLOAT_UNARY_OUT(asin, AsinFunctor)
TP_FLOAT_UNARY_OUT(asinh, AsinhFunctor)
TP_FLOAT_UNARY_OUT(atan, AtanFunctor)
TP_FLOAT_UNARY_OUT(atanh, AtanhFunctor)
TP_FLOAT_UNARY_OUT(ceil, CeilFunctor)
TP_FLOAT_UNARY_OUT(cosh, CoshFunctor)
TP_FLOAT_UNARY_OUT(floor, FloorFunctor)
TP_FLOAT_UNARY_OUT(round, RoundFunctor)
TP_FLOAT_UNARY_OUT(sinh, SinhFunctor)
TP_FLOAT_UNARY_OUT(tan, TanFunctor)
TP_FLOAT_UNARY_OUT(exp, ExpFunctor)
TP_FLOAT_UNARY_OUT(expm1, Expm1Functor)
TP_FLOAT_UNARY_OUT(erf, ErfFunctor)
TP_FLOAT_UNARY_OUT(erfc, ErfcFunctor)
TP_FLOAT_UNARY_OUT(log, LogFunctor)
TP_FLOAT_UNARY_OUT(log10, Log10Functor)
TP_FLOAT_UNARY_OUT(log1p, Log1pFunctor)
TP_FLOAT_UNARY_OUT(log2, Log2Functor)
TP_FLOAT_UNARY_OUT(lgamma, LgammaFunctor)
TP_FLOAT_UNARY_OUT(sqrt, SqrtFunctor)
TP_FLOAT_UNARY_OUT(rsqrt, RsqrtFunctor)
TP_FLOAT_UNARY_OUT(sin, SinFunctor)
TP_FLOAT_UNARY_OUT(cos, CosFunctor)
TP_FLOAT_UNARY_OUT(angle, AngleFunctor)

// tanh and sigmoid carry a second registration that serves quantized and
// complex inputs, so the cases the direct path does not cover go back through
// the dispatcher and land on whichever implementation the operator resolves to.
Tensor& tanh_out_cuda(const Tensor& self, Tensor& out) {
    if (isQuantizedType(self.dtype()) || isComplexType(self.dtype())) {
        return write_out(out, ops::tanh(self));
    }
    return unary_float_op_kernel_out(self, out, TanhFunctor());
}

Tensor& sigmoid_out_cuda(const Tensor& self, Tensor& out) {
    if (isQuantizedType(self.dtype()) || isComplexType(self.dtype())) {
        return write_out(out, ops::sigmoid(self));
    }
    return unary_float_op_kernel_out(self, out, SigmoidFunctor());
}

#undef TP_KEEP_UNARY_OUT
#undef TP_FLOAT_UNARY_OUT

// --- Comparison ---

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

    // out= entries: write the same body straight into a matching destination.
    m.impl("abs.out", abs_out_cuda);
    m.impl("neg.out", neg_out_cuda);
    m.impl("square.out", square_out_cuda);
    m.impl("sign.out", sign_out_cuda);
    m.impl("trunc.out", trunc_out_cuda);
    m.impl("frac.out", frac_out_cuda);

    m.impl("acos.out", acos_out_cuda);
    m.impl("acosh.out", acosh_out_cuda);
    m.impl("asin.out", asin_out_cuda);
    m.impl("asinh.out", asinh_out_cuda);
    m.impl("atan.out", atan_out_cuda);
    m.impl("atanh.out", atanh_out_cuda);
    m.impl("ceil.out", ceil_out_cuda);
    m.impl("cosh.out", cosh_out_cuda);
    m.impl("floor.out", floor_out_cuda);
    m.impl("round.out", round_out_cuda);
    m.impl("sinh.out", sinh_out_cuda);
    m.impl("tan.out", tan_out_cuda);
    m.impl("exp.out", exp_out_cuda);
    m.impl("expm1.out", expm1_out_cuda);
    m.impl("erf.out", erf_out_cuda);
    m.impl("erfc.out", erfc_out_cuda);
    m.impl("log.out", log_out_cuda);
    m.impl("log10.out", log10_out_cuda);
    m.impl("log1p.out", log1p_out_cuda);
    m.impl("log2.out", log2_out_cuda);
    m.impl("lgamma.out", lgamma_out_cuda);
    m.impl("sqrt.out", sqrt_out_cuda);
    m.impl("rsqrt.out", rsqrt_out_cuda);
    m.impl("sin.out", sin_out_cuda);
    m.impl("cos.out", cos_out_cuda);
    m.impl("tanh.out", tanh_out_cuda);
    m.impl("sigmoid.out", sigmoid_out_cuda);
    m.impl("angle.out", angle_out_cuda);
}

} // namespace cuda
} // namespace tensorplay
