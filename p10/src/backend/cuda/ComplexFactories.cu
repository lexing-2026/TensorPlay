// Complex factory kernels on CUDA: real, imag, conj, adjoint, complex,
// polar.  Each elementwise operation uses the iterator launch machinery.

#include "Tensor.h"
#include "TypePromotion.h"
#include "CUDALoops.cuh"
#include "Complex.h"
#include "Exception.h"
#include "Utils.h"

#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

bool is_cplx(DType d) { return isComplexType(d); }

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

template <typename complex_t, typename real_t>
void real_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __device__(complex_t value) -> real_t {
        return value.real();
    });
}

template <typename complex_t, typename real_t>
void imag_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __device__(complex_t value) -> real_t {
        return value.imag();
    });
}

template <typename complex_t>
void conj_loop(const Tensor& input, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(output)
        .add_const_input(input)
        .build();
    gpu_kernel(iter, [] __device__(complex_t value) -> complex_t {
        return complex_t(value.real(), -value.imag());
    });
}

template <typename real_t, typename complex_t>
void complex_loop(const Tensor& real, const Tensor& imag, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(real)
        .add_const_input(imag)
        .build();
    gpu_kernel(iter, [] __device__(real_t real_value, real_t imag_value)
        -> complex_t {
        return complex_t(real_value, imag_value);
    });
}

template <typename real_t, typename math_t, typename complex_t>
void polar_loop(const Tensor& abs, const Tensor& angle, const Tensor& output) {
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(output)
        .add_const_input(abs)
        .add_const_input(angle)
        .build();
    gpu_kernel(iter, [] __device__(real_t radius, real_t angle_value)
        -> complex_t {
        const math_t r = static_cast<math_t>(radius);
        const math_t a = static_cast<math_t>(angle_value);
        const math_t cosine = ::cos(a);
        const math_t sine = ::sin(a);
        return complex_t(static_cast<real_t>(r * cosine),
                         static_cast<real_t>(r * sine));
    });
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

Tensor real_cuda(const Tensor& self) {
    // Real input is its own real part (zero-copy view, as the op contract
    // states); complex input materializes the real component at its paired
    // real precision.
    if (!is_cplx(self.dtype())) return self;
    DType rt = toRealValueType(self.dtype());
    Tensor out = Tensor::empty(shape_of(self), rt, self.device());
    if (out.numel() == 0) return out;
    switch (self.dtype()) {
        case DType::ComplexHalf:
            real_loop<tensorplay::complex<Half>, Half>(self, out);
            break;
        case DType::ComplexFloat:
            real_loop<tensorplay::complex<float>, float>(self, out);
            break;
        case DType::ComplexDouble:
            real_loop<tensorplay::complex<double>, double>(self, out);
            break;
        case DType::BComplex32:
            real_loop<tensorplay::complex<BFloat16>, BFloat16>(self, out);
            break;
        default:
            TP_THROW(NotImplementedError, "real does not support this dtype");
    }
    return out;
}

Tensor imag_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) {
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    }
    DType rt = toRealValueType(self.dtype());
    Tensor out = Tensor::empty(shape_of(self), rt, self.device());
    if (out.numel() == 0) return out;
    switch (self.dtype()) {
        case DType::ComplexHalf:
            imag_loop<tensorplay::complex<Half>, Half>(self, out);
            break;
        case DType::ComplexFloat:
            imag_loop<tensorplay::complex<float>, float>(self, out);
            break;
        case DType::ComplexDouble:
            imag_loop<tensorplay::complex<double>, double>(self, out);
            break;
        case DType::BComplex32:
            imag_loop<tensorplay::complex<BFloat16>, BFloat16>(self, out);
            break;
        default:
            TP_THROW(NotImplementedError, "imag does not support this dtype");
    }
    return out;
}

Tensor conj_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    switch (self.dtype()) {
        case DType::ComplexHalf:
            conj_loop<tensorplay::complex<Half>>(self, out);
            break;
        case DType::ComplexFloat:
            conj_loop<tensorplay::complex<float>>(self, out);
            break;
        case DType::ComplexDouble:
            conj_loop<tensorplay::complex<double>>(self, out);
            break;
        case DType::BComplex32:
            conj_loop<tensorplay::complex<BFloat16>>(self, out);
            break;
        default:
            TP_THROW(NotImplementedError, "conj does not support this dtype");
    }
    return out;
}

Tensor complex_cuda(const Tensor& re, const Tensor& im) {
    check_factory_inputs(re, im, "complex");
    const DType fdt = re.dtype();
    const DType cdt = toComplexType(fdt);
    std::vector<int64_t> shape = broadcast_shapes(shape_of(re), shape_of(im));
    Tensor out = Tensor::empty(shape, cdt, re.device());
    if (out.numel() == 0) return out;
    switch (fdt) {
        case DType::Float16:
            complex_loop<Half, tensorplay::complex<Half>>(re, im, out);
            break;
        case DType::Float32:
            complex_loop<float, tensorplay::complex<float>>(re, im, out);
            break;
        case DType::Float64:
            complex_loop<double, tensorplay::complex<double>>(re, im, out);
            break;
        case DType::BFloat16:
            complex_loop<BFloat16, tensorplay::complex<BFloat16>>(re, im, out);
            break;
        default:
            TP_THROW(NotImplementedError, "complex does not support this dtype");
    }
    return out;
}

Tensor polar_cuda(const Tensor& abs_, const Tensor& angle_) {
    check_factory_inputs(abs_, angle_, "polar");
    const DType fdt = abs_.dtype();
    const DType cdt = toComplexType(fdt);
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    if (out.numel() == 0) return out;
    switch (cdt) {
        case DType::ComplexHalf:
            polar_loop<Half, float, tensorplay::complex<Half>>(
                abs_, angle_, out);
            break;
        case DType::ComplexFloat:
            polar_loop<float, float, tensorplay::complex<float>>(
                abs_, angle_, out);
            break;
        case DType::ComplexDouble:
            polar_loop<double, double, tensorplay::complex<double>>(
                abs_, angle_, out);
            break;
        case DType::BComplex32:
            polar_loop<BFloat16, float, tensorplay::complex<BFloat16>>(
                abs_, angle_, out);
            break;
        default:
            TP_THROW(NotImplementedError, "polar does not support this dtype");
    }
    return out;
}


// composed with conj(); ndim <= 1 is plain conj.  conj_cuda materializes the
// conjugate for complex inputs and aliases real ones.
Tensor adjoint_cuda(const Tensor& self) {
    if (self.dim() <= 1) return conj_cuda(self);
    return conj_cuda(self.transpose(-2, -1));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ComplexFactories) {
    m.impl("real", real_cuda);
    m.impl("imag", imag_cuda);
    m.impl("conj", conj_cuda);
    m.impl("adjoint", adjoint_cuda);
    m.impl("complex", complex_cuda);
    m.impl("polar", polar_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
