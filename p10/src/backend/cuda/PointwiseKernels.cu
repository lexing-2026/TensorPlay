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
}

} // namespace cuda
} // namespace tensorplay
