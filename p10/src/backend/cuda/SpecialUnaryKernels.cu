#include "SpecialKernelUtils.cuh"

#include <SpecialMath.h>

#include <cstdint>

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
struct PolygammaFn {
    int n;

    template <typename T>
    __device__ T operator()(T x) const { return calc_polygamma(x, n); }
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
    return typed_math_cuda(self, PolygammaFn{static_cast<int>(n)});
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
}

}  // namespace tensorplay::cuda
