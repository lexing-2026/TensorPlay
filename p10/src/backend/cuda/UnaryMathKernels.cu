// Core operators - CUDA kernels: unary math operators: the exponential and special-function
// family, the sign and fractional helpers, and the geometric and log-sum
// families.
//
// Pointwise kernels over a grid-stride loop; the shared plumbing comes
// from the pointwise header and the per-operator bodies live here.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {
namespace {

Tensor negative_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            HFn9{},
                            "negative");
}

Tensor positive_cuda(const Tensor& self) { return self.clone(); }

// ===========================================================================
// Comparisons / logic
// ===========================================================================

Tensor reciprocal_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble)
            TP_THROW(NotImplementedError,
                     "CUDA reciprocal: half complexes not supported");
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        const int64_t n = self.numel();
        auto stream = getCurrentCUDAStream().stream();
        Tensor sc = self.contiguous();
        if (self.dtype() == DType::ComplexFloat)
            cuda::cplx::launch_unary<float>(
                n, sc.data_ptr(), result.data_ptr(),
                cuda::cplx::RecipOp{}, stream);
        else
            cuda::cplx::launch_unary<double>(
                n, sc.data_ptr(), result.data_ptr(),
                cuda::cplx::RecipOp{}, stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return float_math_cuda(self, HFn25{}, "reciprocal");
}

namespace {

// z/|z| for nonzero z, zero at the origin; NaN flows through the division.
struct ComplexSgnOp {
    template <typename T>
    __device__ tensorplay::complex<T> operator()(tensorplay::complex<T> z) const {
        if (z.real() == T(0) && z.imag() == T(0)) return tensorplay::complex<T>(T(0), T(0));
        const T r = ::hypot(z.real(), z.imag());
        return tensorplay::complex<T>(z.real() / r, z.imag() / r);
    }
};

}  // namespace

Tensor sgn_cuda(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        if (self.dtype() != DType::ComplexFloat &&
            self.dtype() != DType::ComplexDouble)
            TP_THROW(NotImplementedError, "CUDA sgn: half complexes not supported");
        Tensor result = Tensor::empty(
            static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
            self.device());
        const int64_t n = self.numel();
        if (n == 0) return result;
        auto stream = getCurrentCUDAStream().stream();
        Tensor sc = self.contiguous();
        if (self.dtype() == DType::ComplexFloat)
            cuda::cplx::launch_unary<float>(
                n, sc.data_ptr(), result.data_ptr(), ComplexSgnOp{}, stream);
        else
            cuda::cplx::launch_unary<double>(
                n, sc.data_ptr(), result.data_ptr(), ComplexSgnOp{}, stream);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }
    return dtype_unary_cuda(self,
                            HFn26{},
                            "sgn");
}

Tensor exp2_cuda(const Tensor& self) {
    return float_math_cuda(self, HFn27{}, "exp2");
}

Tensor sinc_cuda(const Tensor& self) {
    return float_math_cuda(self, HFn28{}, "sinc");
}

Tensor deg2rad_cuda(const Tensor& self) {
    return float_math_cuda(self, HFn29{}, "deg2rad");
}

Tensor rad2deg_cuda(const Tensor& self) {
    return float_math_cuda(self, HFn30{}, "rad2deg");
}

Tensor erfinv_cuda(const Tensor& self) {
    // CUDA has no native erfinv; use the Cephes calc_erfinv from SpecialMath.h
    // host-only ::erfinv — linking that from device code leaves an undefined
    // symbol in libp10.so.
    return float_math_cuda(self, HFn32{}, "erfinv");
}

// out= entries: run the same bodies straight into a matching destination;
// complex inputs and destinations needing a resize or cast go back through
// the allocating form.
Tensor& reciprocal_out_cuda(const Tensor& self, Tensor& out) {
    if (isComplexType(self.dtype())) {
        return write_out(out, reciprocal_cuda(self));
    }
    return float_math_out(self, out, HFn25{}, "reciprocal");
}

Tensor& sgn_out_cuda(const Tensor& self, Tensor& out) {
    if (isComplexType(self.dtype())) {
        return write_out(out, sgn_cuda(self));
    }
    return dtype_unary_out(self, out, HFn26{}, "sgn");
}

Tensor& exp2_out_cuda(const Tensor& self, Tensor& out) {
    return float_math_out(self, out, HFn27{}, "exp2");
}

Tensor& sinc_out_cuda(const Tensor& self, Tensor& out) {
    return float_math_out(self, out, HFn28{}, "sinc");
}

Tensor& deg2rad_out_cuda(const Tensor& self, Tensor& out) {
    return float_math_out(self, out, HFn29{}, "deg2rad");
}

Tensor& rad2deg_out_cuda(const Tensor& self, Tensor& out) {
    return float_math_out(self, out, HFn30{}, "rad2deg");
}

Tensor& erfinv_out_cuda(const Tensor& self, Tensor& out) {
    return float_math_out(self, out, HFn32{}, "erfinv");
}

Tensor logit_cuda(const Tensor& self, const std::optional<Scalar>& eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return float_math_cuda(self,
                           HFn33{e},
                           "logit");
}

Tensor& logit_out_cuda(const Tensor& self, std::optional<double> eps, Tensor& out) {
    double e = eps.value_or(-1.0);
    return float_math_out(self, out, HFn33{e}, "logit");
}

Tensor digamma_cuda(const Tensor& self) {
    return float_math_cuda(self, HFn34{}, "digamma");
}

Tensor i0_cuda(const Tensor& self) {
    // Chebyshev expansion, valid over the whole range; see i0_cpu.
    return float_math_cuda(self, HFn35{}, "i0");
}

void nan_to_num_into(const Tensor& self, Tensor& out, double nan_v,
                     const std::optional<double>& posinf, const std::optional<double>& neginf) {
    if (out.numel() == 0) return;
    if (isIntegralType(self.dtype(), true)) {
        out.copy_(self);
        return;
    }

    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_const_input(self)
        .build();
    if (isComplexType(self.dtype())) {
        switch (self.dtype()) {
            case DType::ComplexFloat: {
                using value_t = float;
                using complex_t = tensorplay::complex<value_t>;
                value_t nan_replacement = static_cast<value_t>(nan_v);
                value_t posinf_replacement = posinf.has_value()
                    ? static_cast<value_t>(*posinf)
                    : std::numeric_limits<value_t>::max();
                value_t neginf_replacement = neginf.has_value()
                    ? static_cast<value_t>(*neginf)
                    : std::numeric_limits<value_t>::lowest();
                gpu_kernel(iter, [nan_replacement, posinf_replacement, neginf_replacement] __host__ __device__(tensorplay::complex<float> value) -> tensorplay::complex<float> {
                    return tensorplay::complex<float>(
                        nan_to_num_replace_cuda(
                            value.real(), nan_replacement, posinf_replacement,
                            neginf_replacement),
                        nan_to_num_replace_cuda(
                            value.imag(), nan_replacement, posinf_replacement,
                            neginf_replacement));
                });
                break;
            }
            case DType::ComplexDouble: {
                using value_t = double;
                using complex_t = tensorplay::complex<value_t>;
                value_t nan_replacement = static_cast<value_t>(nan_v);
                value_t posinf_replacement = posinf.has_value()
                    ? static_cast<value_t>(*posinf)
                    : std::numeric_limits<value_t>::max();
                value_t neginf_replacement = neginf.has_value()
                    ? static_cast<value_t>(*neginf)
                    : std::numeric_limits<value_t>::lowest();
                gpu_kernel(iter, [nan_replacement, posinf_replacement, neginf_replacement] __host__ __device__(tensorplay::complex<double> value) -> tensorplay::complex<double> {
                    return tensorplay::complex<double>(
                        nan_to_num_replace_cuda(
                            value.real(), nan_replacement, posinf_replacement,
                            neginf_replacement),
                        nan_to_num_replace_cuda(
                            value.imag(), nan_replacement, posinf_replacement,
                            neginf_replacement));
                });
                break;
            }
            default:
                TP_THROW(NotImplementedError,
                         "nan_to_num: reduced complex types are not supported on CUDA");
        }
    } else {
#define TP_NTN_FLOAT(ctype, name_) \
        case DType::name_: { \
            ctype nan_replacement = static_cast<ctype>(nan_v); \
            ctype posinf_replacement = posinf.has_value() \
                ? static_cast<ctype>(*posinf) \
                : std::numeric_limits<ctype>::max(); \
            ctype neginf_replacement = neginf.has_value() \
                ? static_cast<ctype>(*neginf) \
                : std::numeric_limits<ctype>::lowest(); \
            gpu_kernel(iter, [nan_replacement, posinf_replacement, neginf_replacement] __host__ __device__(ctype value) -> ctype { \
                return nan_to_num_replace_cuda( \
                    value, nan_replacement, posinf_replacement, \
                    neginf_replacement); \
            }); \
            break; \
        }
        switch (self.dtype()) {
            TP_NTN_FLOAT(float, Float32)
            TP_NTN_FLOAT(double, Float64)
            TP_NTN_FLOAT(Half, Float16)
            TP_NTN_FLOAT(BFloat16, BFloat16)
            default: TP_THROW(TypeError, "nan_to_num: unsupported dtype");
        }
#undef TP_NTN_FLOAT
    }
    CUDA_CHECK(cudaGetLastError());
}

Tensor nan_to_num_cuda(const Tensor& self, const Scalar& nan,
                       const std::optional<Scalar>& posinf, const std::optional<Scalar>& neginf) {
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    nan_to_num_into(self, out, nan.toDouble(),
                    posinf.has_value() ? std::optional<double>(posinf->toDouble()) : std::nullopt,
                    neginf.has_value() ? std::optional<double>(neginf->toDouble()) : std::nullopt);
    return out;
}

Tensor& nan_to_num_out_cuda(const Tensor& self, std::optional<double> nan,
                            std::optional<double> posinf,
                            std::optional<double> neginf, Tensor& out) {
    if (destination_takes(out, self.device(), self.dtype(), shape_of(self)) &&
        read_while_write_is_safe(self, out)) {
        nan_to_num_into(self, out, nan.value_or(0.0), posinf, neginf);
        return out;
    }
    return write_out(out, nan_to_num_cuda(self, Scalar(nan.value_or(0.0)),
                                          posinf.has_value() ? std::optional<Scalar>(Scalar(*posinf)) : std::nullopt,
                                          neginf.has_value() ? std::optional<Scalar>(Scalar(*neginf)) : std::nullopt));
}

Tensor logaddexp_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, HFn40{}, "logaddexp");
}

Tensor logaddexp2_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, HFn41{}, "logaddexp2");
}

Tensor copysign_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, HFn42{}, "copysign");
}

Tensor copysign_scalar_cuda(const Tensor& self, const Scalar& other) {
    // The sign comes from the scalar alone, so the divisor width never
    // participates in promotion -- Float32 carries every sign bit exactly.
    return copysign_cuda(self, Tensor::full({}, other, DType::Float32, self.device()));
}

// The float-wide square root is several times cheaper than the double one,
// and the extra double digits cannot survive the float result anyway.
struct HypotFloatPair {
    __host__ __device__ double operator()(double x, double y) const {
        return static_cast<double>(
            ::hypotf(static_cast<float>(x), static_cast<float>(y)));
    }
};

Tensor hypot_cuda(const Tensor& a, const Tensor& b) {
    if (promoteTypes(a.dtype(), b.dtype()) == DType::Float64) {
        return binary_float_cuda(a, b, HFn43{}, "hypot");
    }
    return binary_float_cuda(a, b, HypotFloatPair{}, "hypot");
}

Tensor& hypot_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    if (promoteTypes(a.dtype(), b.dtype()) == DType::Float64) {
        return binary_float_cuda_out(a, b, out, HFn43{}, "hypot");
    }
    return binary_float_cuda_out(a, b, out, HypotFloatPair{}, "hypot");
}

Tensor nextafter_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, HFn44{}, "nextafter");
}

Tensor& nextafter_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return binary_float_cuda_out(a, b, out, HFn44{}, "nextafter");
}

Tensor gcd_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_cuda(a, b,
                            HFn45{},
                            "gcd");
}

Tensor& gcd_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_cuda_out(a, b, out,
                                HFn45{},
                                "gcd");
}

Tensor lcm_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_cuda(a, b,
                            HFn46{},
                            "lcm");
}

Tensor& lcm_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_cuda_out(a, b, out,
                                HFn46{},
                                "lcm");
}

Tensor heaviside_cuda(const Tensor& a, const Tensor& values) {
    return binary_same_cuda(a, values,
                            HFn47{},
                            "heaviside");
}

Tensor& heaviside_out_cuda(const Tensor& a, const Tensor& values, Tensor& out) {
    return binary_same_cuda_out(a, values, out,
                                HFn47{},
                                "heaviside");
}

// ===========================================================================
// Clamp family
// ===========================================================================

Tensor logit_backward_cuda(const Tensor& grad_output, const Tensor& self, const std::optional<Scalar>& eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return binary_same_cuda(grad_output, self,
                            HFn60{e},
                            "logit_backward");
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UnaryMathKernels) {
    m.impl("negative", negative_cuda);
    m.impl("positive", positive_cuda);
    m.impl("reciprocal", reciprocal_cuda);
    m.impl("sgn", sgn_cuda);
    m.impl("exp2", exp2_cuda);
    m.impl("sinc", sinc_cuda);
    m.impl("deg2rad", deg2rad_cuda);
    m.impl("rad2deg", rad2deg_cuda);
    m.impl("erfinv", erfinv_cuda);
    m.impl("reciprocal.out", reciprocal_out_cuda);
    m.impl("sgn.out", sgn_out_cuda);
    m.impl("exp2.out", exp2_out_cuda);
    m.impl("sinc.out", sinc_out_cuda);
    m.impl("deg2rad.out", deg2rad_out_cuda);
    m.impl("rad2deg.out", rad2deg_out_cuda);
    m.impl("erfinv.out", erfinv_out_cuda);
    m.impl("logit", logit_cuda);
    m.impl("logit.out", logit_out_cuda);
    m.impl("digamma", digamma_cuda);
    m.impl("i0", i0_cuda);
    m.impl("nan_to_num", nan_to_num_cuda);
    m.impl("nan_to_num.out", nan_to_num_out_cuda);
    m.impl("logaddexp", logaddexp_cuda);
    m.impl("logaddexp2", logaddexp2_cuda);
    m.impl("copysign.Tensor", copysign_cuda);
    m.impl("copysign.Scalar", copysign_scalar_cuda);
    m.impl("hypot", hypot_cuda);
    m.impl("hypot.out", hypot_out_cuda);
    m.impl("nextafter", nextafter_cuda);
    m.impl("nextafter.out", nextafter_out_cuda);
    m.impl("gcd", gcd_cuda);
    m.impl("gcd.out", gcd_out_cuda);
    m.impl("lcm", lcm_cuda);
    m.impl("lcm.out", lcm_out_cuda);
    m.impl("heaviside", heaviside_cuda);
    m.impl("heaviside.out", heaviside_out_cuda);
    m.impl("logit_backward", logit_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
