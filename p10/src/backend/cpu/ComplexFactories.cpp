// Complex factory views and constructors: real, imag, conj, adjoint,
// complex, polar.
//
// These live outside the vectorized complex arithmetic core
// (ComplexKernels.cpp) because they are small broadcast+copy kernels; the
// loop bodies stay scalar over contiguous buffers.

#include "Tensor.h"
#include "Complex.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Exception.h"

#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

bool is_cplx(DType d) {
    return isComplexType(d);
}

bool is_factory_real_dtype(DType d) {
    return d == DType::Float16 || d == DType::Float32 ||
           d == DType::Float64 || d == DType::BFloat16;
}

void check_factory_inputs(const Tensor& a, const Tensor& b, const char* name) {
    if (!is_factory_real_dtype(a.dtype()) || !is_factory_real_dtype(b.dtype())) {
        TP_THROW(NotImplementedError, name,
                 " expects floating-point inputs");
    }
    if (a.dtype() != b.dtype()) {
        TP_THROW(RuntimeError, name, " expects inputs with the same dtype");
    }
    if (a.device() != b.device()) {
        TP_THROW(DeviceMismatchError, name,
                 " expects inputs on the same device");
    }
}

std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

}  // namespace

Tensor real_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    Tensor out = Tensor::empty(shape_of(self), toRealValueType(self.dtype()), self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    switch (self.dtype()) {
        case DType::ComplexHalf: {
            const tensorplay::complex<Half>* sp = sc.data_ptr<tensorplay::complex<Half>>();
            Half* dp = out.data_ptr<Half>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
            break;
        }
        case DType::ComplexFloat: {
            const tensorplay::complex<float>* sp = sc.data_ptr<tensorplay::complex<float>>();
            float* dp = out.data_ptr<float>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double>* sp = sc.data_ptr<tensorplay::complex<double>>();
            double* dp = out.data_ptr<double>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
            break;
        }
        case DType::BComplex32: {
            const tensorplay::complex<BFloat16>* sp =
                sc.data_ptr<tensorplay::complex<BFloat16>>();
            BFloat16* dp = out.data_ptr<BFloat16>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
            break;
        }
        default:
            TP_THROW(NotImplementedError, "real does not support this dtype");
    }
    return out;
}

Tensor imag_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype()))
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    Tensor out = Tensor::empty(shape_of(self), toRealValueType(self.dtype()), self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    switch (self.dtype()) {
        case DType::ComplexHalf: {
            const tensorplay::complex<Half>* sp = sc.data_ptr<tensorplay::complex<Half>>();
            Half* dp = out.data_ptr<Half>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
            break;
        }
        case DType::ComplexFloat: {
            const tensorplay::complex<float>* sp = sc.data_ptr<tensorplay::complex<float>>();
            float* dp = out.data_ptr<float>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
            break;
        }
        case DType::ComplexDouble: {
            const tensorplay::complex<double>* sp = sc.data_ptr<tensorplay::complex<double>>();
            double* dp = out.data_ptr<double>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
            break;
        }
        case DType::BComplex32: {
            const tensorplay::complex<BFloat16>* sp =
                sc.data_ptr<tensorplay::complex<BFloat16>>();
            BFloat16* dp = out.data_ptr<BFloat16>();
            for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
            break;
        }
        default:
            TP_THROW(NotImplementedError, "imag does not support this dtype");
    }
    return out;
}

Tensor conj_cpu(const Tensor& self) {
    // Returning `self` itself would hand the same impl back to autograd
    // wrappers, which then re-tag the input's grad_fn and corrupt graphs.
    if (!is_cplx(self.dtype())) {
        return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                               static_cast<std::vector<int64_t>>(self.strides()));
    }
    Tensor out = detail::contiguous_clone(self);
    int64_t n = out.numel();
    switch (self.dtype()) {
        case DType::ComplexHalf: {
            tensorplay::complex<Half>* dp = out.data_ptr<tensorplay::complex<Half>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<Half>(
                    dp[i].real(), static_cast<Half>(-static_cast<float>(dp[i].imag())));
            break;
        }
        case DType::ComplexFloat: {
            tensorplay::complex<float>* dp = out.data_ptr<tensorplay::complex<float>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<float>(dp[i].real(), -dp[i].imag());
            break;
        }
        case DType::ComplexDouble: {
            tensorplay::complex<double>* dp = out.data_ptr<tensorplay::complex<double>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<double>(dp[i].real(), -dp[i].imag());
            break;
        }
        case DType::BComplex32: {
            tensorplay::complex<BFloat16>* dp =
                out.data_ptr<tensorplay::complex<BFloat16>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<BFloat16>(
                    dp[i].real(),
                    static_cast<BFloat16>(-static_cast<float>(dp[i].imag())));
            break;
        }
        default:
            TP_THROW(NotImplementedError, "conj does not support this dtype");
    }
    return out;
}

// transpose(-2, -1) composed with conj(); ndim <= 1 is plain conj.
Tensor adjoint_cpu(const Tensor& self) {
    if (self.dim() <= 1) return conj_cpu(self);
    return conj_cpu(self.transpose(-2, -1));
}

Tensor complex_cpu(const Tensor& real, const Tensor& imag) {
    check_factory_inputs(real, imag, "complex");
    const DType cdt = toComplexType(real.dtype());
    std::vector<int64_t> shape = broadcast_shapes(shape_of(real), shape_of(imag));
    Tensor rc = real.expand(shape).contiguous();
    Tensor ic = imag.expand(shape).contiguous().to(rc.dtype());
    Tensor out = Tensor::empty(shape, cdt, real.device());
    int64_t n = out.numel();
    switch (real.dtype()) {
        case DType::Float16: {
            const Half* rp = rc.data_ptr<Half>();
            const Half* ip = ic.data_ptr<Half>();
            tensorplay::complex<Half>* dp = out.data_ptr<tensorplay::complex<Half>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<Half>(rp[i], ip[i]);
            break;
        }
        case DType::Float32: {
            const float* rp = rc.data_ptr<float>();
            const float* ip = ic.data_ptr<float>();
            tensorplay::complex<float>* dp = out.data_ptr<tensorplay::complex<float>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<float>(rp[i], ip[i]);
            break;
        }
        case DType::Float64: {
            const double* rp = rc.data_ptr<double>();
            const double* ip = ic.data_ptr<double>();
            tensorplay::complex<double>* dp = out.data_ptr<tensorplay::complex<double>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<double>(rp[i], ip[i]);
            break;
        }
        case DType::BFloat16: {
            const BFloat16* rp = rc.data_ptr<BFloat16>();
            const BFloat16* ip = ic.data_ptr<BFloat16>();
            tensorplay::complex<BFloat16>* dp =
                out.data_ptr<tensorplay::complex<BFloat16>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::complex<BFloat16>(rp[i], ip[i]);
            break;
        }
        default:
            TP_THROW(NotImplementedError, "complex does not support this dtype");
    }
    return out;
}

Tensor polar_cpu(const Tensor& abs_, const Tensor& angle_) {
    check_factory_inputs(abs_, angle_, "polar");
    const DType cdt = toComplexType(abs_.dtype());
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor a = abs_.expand(shape).contiguous();
    Tensor th = angle_.expand(shape).contiguous();
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    int64_t n = out.numel();
    switch (abs_.dtype()) {
        case DType::Float16: {
            const Half* ap = a.data_ptr<Half>();
            const Half* tp = th.data_ptr<Half>();
            tensorplay::complex<Half>* dp = out.data_ptr<tensorplay::complex<Half>>();
            for (int64_t i = 0; i < n; ++i) {
                const tensorplay::complex<float> v = tensorplay::polar(
                    static_cast<float>(ap[i]), static_cast<float>(tp[i]));
                dp[i] = tensorplay::complex<Half>(static_cast<Half>(v.real()),
                                                  static_cast<Half>(v.imag()));
            }
            break;
        }
        case DType::Float32: {
            const float* ap = a.data_ptr<float>();
            const float* tp = th.data_ptr<float>();
            tensorplay::complex<float>* dp = out.data_ptr<tensorplay::complex<float>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::polar(ap[i], tp[i]);
            break;
        }
        case DType::Float64: {
            const double* ap = a.data_ptr<double>();
            const double* tp = th.data_ptr<double>();
            tensorplay::complex<double>* dp = out.data_ptr<tensorplay::complex<double>>();
            for (int64_t i = 0; i < n; ++i)
                dp[i] = tensorplay::polar(ap[i], tp[i]);
            break;
        }
        case DType::BFloat16: {
            const BFloat16* ap = a.data_ptr<BFloat16>();
            const BFloat16* tp = th.data_ptr<BFloat16>();
            tensorplay::complex<BFloat16>* dp =
                out.data_ptr<tensorplay::complex<BFloat16>>();
            for (int64_t i = 0; i < n; ++i) {
                const tensorplay::complex<float> v = tensorplay::polar(
                    static_cast<float>(ap[i]), static_cast<float>(tp[i]));
                dp[i] = tensorplay::complex<BFloat16>(static_cast<BFloat16>(v.real()),
                                                      static_cast<BFloat16>(v.imag()));
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError, "polar does not support this dtype");
    }
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CPU, ComplexFactories) {
    m.impl("real", real_cpu);
    m.impl("imag", imag_cpu);
    m.impl("conj", conj_cpu);
    m.impl("adjoint", adjoint_cpu);
    m.impl("complex", complex_cpu);
    m.impl("polar", polar_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
