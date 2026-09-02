// Complex factory views and constructors: real, imag, conj, adjoint,
// complex, polar.
//
// These live outside the vectorized complex arithmetic core
// (ComplexKernels.cpp) because they are small broadcast+copy kernels; the
// loop bodies stay scalar over contiguous buffers.

#include "Tensor.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Exception.h"

#include <vector>
#include <complex>

namespace tensorplay {
namespace cpu {

namespace {

bool is_cplx(DType d) {
    return d == DType::ComplexFloat || d == DType::ComplexDouble;
}

std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

}  // namespace

Tensor real_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    Tensor out = Tensor::empty(shape_of(self),
                               self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32,
                               self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    if (self.dtype() == DType::ComplexFloat) {
        const std::complex<float>* sp = sc.data_ptr<std::complex<float>>();
        float* dp = out.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
    } else {
        const std::complex<double>* sp = sc.data_ptr<std::complex<double>>();
        double* dp = out.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
    }
    return out;
}

Tensor imag_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype()))
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    Tensor out = Tensor::empty(shape_of(self),
                               self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32,
                               self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    if (self.dtype() == DType::ComplexFloat) {
        const std::complex<float>* sp = sc.data_ptr<std::complex<float>>();
        float* dp = out.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
    } else {
        const std::complex<double>* sp = sc.data_ptr<std::complex<double>>();
        double* dp = out.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
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
    if (self.dtype() == DType::ComplexFloat) {
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::conj(dp[i]);
    } else {
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::conj(dp[i]);
    }
    return out;
}

// transpose(-2, -1) composed with conj(); ndim <= 1 is plain conj.
Tensor adjoint_cpu(const Tensor& self) {
    if (self.dim() <= 1) return conj_cpu(self);
    return conj_cpu(self.transpose(-2, -1));
}

Tensor complex_cpu(const Tensor& real, const Tensor& imag) {
    DType fdt = promoteTypes(real.dtype(), imag.dtype());
    if (fdt == DType::Float64) fdt = DType::ComplexDouble;
    else fdt = DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(real), shape_of(imag));
    Tensor rc = real.expand(shape).contiguous().to(fdt == DType::ComplexDouble ? DType::Float64 : DType::Float32);
    Tensor ic = imag.expand(shape).contiguous().to(rc.dtype());
    Tensor out = Tensor::empty(shape, fdt, real.device());
    int64_t n = out.numel();
    if (fdt == DType::ComplexFloat) {
        const float* rp = rc.data_ptr<float>();
        const float* ip = ic.data_ptr<float>();
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::complex<float>(rp[i], ip[i]);
    } else {
        const double* rp = rc.data_ptr<double>();
        const double* ip = ic.data_ptr<double>();
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::complex<double>(rp[i], ip[i]);
    }
    return out;
}

Tensor polar_cpu(const Tensor& abs_, const Tensor& angle_) {
    DType fdt = promoteTypes(abs_.dtype(), angle_.dtype());
    if (fdt != DType::Float64) fdt = DType::Float32;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor a = abs_.expand(shape).contiguous().to(fdt);
    Tensor th = angle_.expand(shape).contiguous().to(fdt);
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    int64_t n = out.numel();
    if (fdt == DType::Float64) {
        const double* ap = a.data_ptr<double>();
        const double* tp = th.data_ptr<double>();
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = std::polar(ap[i], tp[i]);
    } else {
        const float* ap = a.data_ptr<float>();
        const float* tp = th.data_ptr<float>();
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = std::polar(ap[i], tp[i]);
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
