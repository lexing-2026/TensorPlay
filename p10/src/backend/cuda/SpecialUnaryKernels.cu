#include "SpecialKernelUtils.cuh"
#include "CUDALoops.cuh"

#include <SpecialMath.h>

#include <cstdint>
#include <limits>
#include <tuple>

namespace tensorplay::cuda {
namespace {

using special_detail::typed_math_cuda;
using tensorplay::special_math::airy_ai_forward;
using tensorplay::special_math::bessel_j0_forward;
using tensorplay::special_math::bessel_j1_forward;
using tensorplay::special_math::bessel_y0_forward;
using tensorplay::special_math::bessel_y1_forward;
using tensorplay::special_math::calc_entr;
using tensorplay::special_math::calc_erfcx;
using tensorplay::special_math::calc_i0e;
using tensorplay::special_math::calc_i1e;
using tensorplay::special_math::calc_log_ndtr;
using tensorplay::special_math::calc_ndtr;
using tensorplay::special_math::calc_ndtri;
using tensorplay::special_math::calc_polygamma;
using tensorplay::special_math::modified_bessel_i0_forward;
using tensorplay::special_math::modified_bessel_i1_forward;
using tensorplay::special_math::modified_bessel_k0_forward;
using tensorplay::special_math::modified_bessel_k1_forward;
using tensorplay::special_math::scaled_modified_bessel_k0_forward;
using tensorplay::special_math::scaled_modified_bessel_k1_forward;
using tensorplay::special_math::spherical_bessel_j0_forward;

struct AiryAiFn {
    template <typename T>
    __device__ T operator()(T x) const { return airy_ai_forward(x); }
};
struct BesselJ0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return bessel_j0_forward(x); }
};
struct BesselJ1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return bessel_j1_forward(x); }
};
struct BesselY0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return bessel_y0_forward(x); }
};
struct BesselY1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return bessel_y1_forward(x); }
};
struct SphericalBesselJ0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return spherical_bessel_j0_forward(x); }
};
struct ModifiedBesselI0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return modified_bessel_i0_forward(x); }
};
struct ModifiedBesselI1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return modified_bessel_i1_forward(x); }
};
struct ModifiedBesselK0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return modified_bessel_k0_forward(x); }
};
struct ModifiedBesselK1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return modified_bessel_k1_forward(x); }
};
struct ScaledModifiedBesselK0Fn {
    template <typename T>
    __device__ T operator()(T x) const { return scaled_modified_bessel_k0_forward(x); }
};
struct ScaledModifiedBesselK1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return scaled_modified_bessel_k1_forward(x); }
};
struct I0eFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_i0e(x); }
};
struct I1Fn {
    template <typename T>
    __device__ T operator()(T x) const { return modified_bessel_i1_forward(x); }
};
struct I1eFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_i1e(x); }
};
struct ErfcxFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_erfcx(x); }
};
struct NdtrFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_ndtr(x); }
};
struct NdtriFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_ndtri(x); }
};
struct LogNdtrFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_log_ndtr(x); }
};
struct EntrFn {
    template <typename T>
    __device__ T operator()(T x) const { return calc_entr(x); }
};

__device__ inline double polygamma_digamma(double x) {
    if (x == 0.0) {
        return ::copysign(std::numeric_limits<double>::infinity(), -x);
    }
    if (x < 0.0) {
        if (x == ::trunc(x)) {
            return ::nan("");
        }
        double integer_part = 0.0;
        const double fraction = ::modf(x, &integer_part);
        return polygamma_digamma(1.0 - x) -
            (M_PI / ::tan(M_PI * fraction));
    }
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    if (x == 10.0) {
        return result + 2.25175258906672110764;
    }
    const double inverse = 1.0 / x;
    const double inverse_square = inverse * inverse;
    result += ::log(x) - 0.5 * inverse - inverse_square * (
        1.0 / 12.0 - inverse_square * (
            1.0 / 120.0 - inverse_square * (
                1.0 / 252.0 - inverse_square * (
                    1.0 / 240.0 - inverse_square / 132.0))));
    return result;
}

__device__ inline double polygamma_trigamma(double x) {
    double sign = 1.0;
    double result = 0.0;
    if (x < 0.5) {
        sign = -1.0;
        const double sine = ::sin(M_PI * x);
        result -= (M_PI * M_PI) / (sine * sine);
        x = 1.0 - x;
    }
    for (int i = 0; i < 6; ++i) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    const double inverse_square = 1.0 / (x * x);
    result += (1.0 + 0.5 / x + inverse_square * (
        1.0 / 6.0 - inverse_square * (
            1.0 / 30.0 - inverse_square / 42.0))) / x;
    return sign * result;
}

struct PolygammaFn {
    int n;

    template <typename T>
    __device__ T operator()(T x) const {
        if (n == 0) {
            return static_cast<T>(polygamma_digamma(static_cast<double>(x)));
        }
        if (n == 1) {
            return static_cast<T>(polygamma_trigamma(static_cast<double>(x)));
        }
        return calc_polygamma(x, n);
    }
};

Tensor airy_ai_cuda(const Tensor& self) {
    return typed_math_cuda(self, AiryAiFn{});
}
Tensor bessel_j0_cuda(const Tensor& self) {
    return typed_math_cuda(self, BesselJ0Fn{});
}
Tensor bessel_j1_cuda(const Tensor& self) {
    return typed_math_cuda(self, BesselJ1Fn{});
}
Tensor bessel_y0_cuda(const Tensor& self) {
    return typed_math_cuda(self, BesselY0Fn{});
}
Tensor bessel_y1_cuda(const Tensor& self) {
    return typed_math_cuda(self, BesselY1Fn{});
}
Tensor spherical_bessel_j0_cuda(const Tensor& self) {
    return typed_math_cuda(self, SphericalBesselJ0Fn{});
}
Tensor modified_bessel_i0_cuda(const Tensor& self) {
    return typed_math_cuda(self, ModifiedBesselI0Fn{});
}
Tensor modified_bessel_i1_cuda(const Tensor& self) {
    return typed_math_cuda(self, ModifiedBesselI1Fn{});
}
Tensor modified_bessel_k0_cuda(const Tensor& self) {
    return typed_math_cuda(self, ModifiedBesselK0Fn{});
}
Tensor modified_bessel_k1_cuda(const Tensor& self) {
    return typed_math_cuda(self, ModifiedBesselK1Fn{});
}
Tensor scaled_modified_bessel_k0_cuda(const Tensor& self) {
    return typed_math_cuda(self, ScaledModifiedBesselK0Fn{});
}
Tensor scaled_modified_bessel_k1_cuda(const Tensor& self) {
    return typed_math_cuda(self, ScaledModifiedBesselK1Fn{});
}
Tensor i0e_cuda(const Tensor& self) {
    return typed_math_cuda(self, I0eFn{});
}
Tensor i1_cuda(const Tensor& self) {
    return typed_math_cuda(self, I1Fn{});
}
Tensor i1e_cuda(const Tensor& self) {
    return typed_math_cuda(self, I1eFn{});
}
Tensor erfcx_cuda(const Tensor& self) {
    return typed_math_cuda(self, ErfcxFn{});
}
Tensor ndtr_cuda(const Tensor& self) {
    return typed_math_cuda(self, NdtrFn{});
}
Tensor ndtri_cuda(const Tensor& self) {
    return typed_math_cuda(self, NdtriFn{});
}
Tensor log_ndtr_cuda(const Tensor& self) {
    return typed_math_cuda(self, LogNdtrFn{});
}
Tensor entr_cuda(const Tensor& self) {
    return typed_math_cuda(self, EntrFn{});
}
Tensor polygamma_cuda(int64_t n, const Tensor& self) {
    if (n < 0) {
        TP_THROW(RuntimeError, "polygamma(n, x) does not support negative n");
    }
    if (n > std::numeric_limits<int>::max()) {
        TP_THROW(RuntimeError, "polygamma order is too large: ", n);
    }
    return typed_math_cuda(self, PolygammaFn{static_cast<int>(n)});
}

std::tuple<Tensor, Tensor> frexp_cuda(const Tensor& self) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "frexp(): only supports floating-point dtypes");
    }
    const DType compute_dtype = self.dtype() == DType::Float64
        ? DType::Float64 : DType::Float32;
    Tensor input = self.dtype() == compute_dtype
        ? self : self.to(compute_dtype);
    Tensor mantissa = Tensor::empty(
        special_detail::shape_of(input), compute_dtype, input.device());
    Tensor exponent = Tensor::empty(
        special_detail::shape_of(input), DType::Int32, input.device());
    if (input.numel() > 0) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(false)
            .add_output(mantissa)
            .add_output(exponent)
            .add_input(input)
            .build();
        if (compute_dtype == DType::Float64) {
            gpu_kernel_multiple_outputs(
                iter, [] __host__ __device__ (double value) -> std::tuple<double, int32_t> {
                    int exponent_value = 0;
                    const double mantissa_value = ::frexp(value, &exponent_value);
                    return {mantissa_value, static_cast<int32_t>(exponent_value)};
                });
        } else {
            gpu_kernel_multiple_outputs(
                iter, [] __host__ __device__ (float value) -> std::tuple<float, int32_t> {
                    int exponent_value = 0;
                    const float mantissa_value = ::frexp(value, &exponent_value);
                    return {mantissa_value, static_cast<int32_t>(exponent_value)};
                });
        }
    }
    if (compute_dtype != self.dtype()) {
        mantissa = mantissa.to(self.dtype());
    }
    return {mantissa, exponent};
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SpecialUnaryKernels) {
    m.impl("bessel_j0", bessel_j0_cuda);
    m.impl("bessel_j1", bessel_j1_cuda);
    m.impl("bessel_y0", bessel_y0_cuda);
    m.impl("bessel_y1", bessel_y1_cuda);
    m.impl("airy_ai", airy_ai_cuda);
    m.impl("spherical_bessel_j0", spherical_bessel_j0_cuda);
    m.impl("modified_bessel_i0", modified_bessel_i0_cuda);
    m.impl("modified_bessel_i1", modified_bessel_i1_cuda);
    m.impl("modified_bessel_k0", modified_bessel_k0_cuda);
    m.impl("modified_bessel_k1", modified_bessel_k1_cuda);
    m.impl("scaled_modified_bessel_k0", scaled_modified_bessel_k0_cuda);
    m.impl("scaled_modified_bessel_k1", scaled_modified_bessel_k1_cuda);
    m.impl("i0e", i0e_cuda);
    m.impl("i1", i1_cuda);
    m.impl("i1e", i1e_cuda);
    m.impl("erfcx", erfcx_cuda);
    m.impl("ndtr", ndtr_cuda);
    m.impl("ndtri", ndtri_cuda);
    m.impl("log_ndtr", log_ndtr_cuda);
    m.impl("entr", entr_cuda);
    m.impl("polygamma", polygamma_cuda);
    m.impl("frexp", frexp_cuda);
}

}  // namespace tensorplay::cuda
