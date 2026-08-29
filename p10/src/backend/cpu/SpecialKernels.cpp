//
// (2.15.0a0) via p10/include/SpecialMath.h; the wrappers follow the house
// float_math_kernel / binary_float_kernel pattern from TierOpsKernels.cpp.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"

#include <SpecialMath.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace {

using namespace tensorplay::parallel;

using tensorplay::special_math::airy_ai_forward;
using tensorplay::special_math::bessel_j0_forward;
using tensorplay::special_math::bessel_j1_forward;
using tensorplay::special_math::bessel_y0_forward;
using tensorplay::special_math::bessel_y1_forward;
using tensorplay::special_math::calc_i0e;
using tensorplay::special_math::calc_i1e;
using tensorplay::special_math::calc_igamma;
using tensorplay::special_math::calc_igammac;
using tensorplay::special_math::calc_polygamma;
using tensorplay::special_math::chebyshev_polynomial_t_forward;
using tensorplay::special_math::chebyshev_polynomial_u_forward;
using tensorplay::special_math::chebyshev_polynomial_v_forward;
using tensorplay::special_math::chebyshev_polynomial_w_forward;
using tensorplay::special_math::hermite_polynomial_h_forward;
using tensorplay::special_math::hermite_polynomial_he_forward;
using tensorplay::special_math::laguerre_polynomial_l_forward;
using tensorplay::special_math::legendre_polynomial_p_forward;
using tensorplay::special_math::modified_bessel_i1_forward;
using tensorplay::special_math::modified_bessel_k0_forward;
using tensorplay::special_math::modified_bessel_k1_forward;
using tensorplay::special_math::scaled_modified_bessel_k0_forward;
using tensorplay::special_math::scaled_modified_bessel_k1_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_t_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_u_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_v_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_w_forward;
using tensorplay::special_math::spherical_bessel_j0_forward;
using tensorplay::special_math::zeta;

template <typename F>  // F: double -> double
Tensor float_math_kernel(const Tensor& self, F f, const char* name) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (self.dtype() == compute_dt) ? self.contiguous()
                                            : self.to(compute_dt).contiguous();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(w.shape()), compute_dt, w.device());
    int64_t n = w.numel();
    if (compute_dt == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = f(sp[i]);
        });
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<float>(f(sp[i]));
        });
    }
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

template <typename F>  // F: (double,double) -> double
Tensor binary_float_kernel(const Tensor& a_in, const Tensor& b_in, F f, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = f(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<float>(f(ap[i], bp[i]));
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// Unary specials
// ---------------------------------------------------------------------------

Tensor airy_ai_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return airy_ai_forward(x); }, "airy_ai");
}
Tensor bessel_j0_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return bessel_j0_forward(x); }, "bessel_j0");
}
Tensor bessel_j1_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return bessel_j1_forward(x); }, "bessel_j1");
}
Tensor bessel_y0_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return bessel_y0_forward(x); }, "bessel_y0");
}
Tensor bessel_y1_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return bessel_y1_forward(x); }, "bessel_y1");
}
Tensor spherical_bessel_j0_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return spherical_bessel_j0_forward(x); },
                             "spherical_bessel_j0");
}
Tensor modified_bessel_i1_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return modified_bessel_i1_forward(x); },
                             "modified_bessel_i1");
}
Tensor modified_bessel_k0_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return modified_bessel_k0_forward(x); },
                             "modified_bessel_k0");
}
Tensor modified_bessel_k1_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return modified_bessel_k1_forward(x); },
                             "modified_bessel_k1");
}
Tensor scaled_modified_bessel_k0_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return scaled_modified_bessel_k0_forward(x); },
                             "scaled_modified_bessel_k0");
}
Tensor scaled_modified_bessel_k1_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return scaled_modified_bessel_k1_forward(x); },
                             "scaled_modified_bessel_k1");
}
Tensor i0e_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_i0e(x); }, "i0e");
}
Tensor i1_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return modified_bessel_i1_forward(x); }, "i1");
}
Tensor i1e_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_i1e(x); }, "i1e");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

Tensor chebyshev_polynomial_t_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return chebyshev_polynomial_t_forward<double>(a, b);
    }, "chebyshev_polynomial_t");
}
Tensor chebyshev_polynomial_u_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return chebyshev_polynomial_u_forward<double>(a, b);
    }, "chebyshev_polynomial_u");
}
Tensor chebyshev_polynomial_v_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return chebyshev_polynomial_v_forward<double>(a, b);
    }, "chebyshev_polynomial_v");
}
Tensor chebyshev_polynomial_w_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return chebyshev_polynomial_w_forward<double>(a, b);
    }, "chebyshev_polynomial_w");
}
Tensor shifted_chebyshev_polynomial_t_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return shifted_chebyshev_polynomial_t_forward<double>(a, b);
    }, "shifted_chebyshev_polynomial_t");
}
Tensor shifted_chebyshev_polynomial_u_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return shifted_chebyshev_polynomial_u_forward<double>(a, b);
    }, "shifted_chebyshev_polynomial_u");
}
Tensor shifted_chebyshev_polynomial_v_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return shifted_chebyshev_polynomial_v_forward<double>(a, b);
    }, "shifted_chebyshev_polynomial_v");
}
Tensor shifted_chebyshev_polynomial_w_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return shifted_chebyshev_polynomial_w_forward<double>(a, b);
    }, "shifted_chebyshev_polynomial_w");
}
Tensor hermite_polynomial_h_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return hermite_polynomial_h_forward<double>(a, b);
    }, "hermite_polynomial_h");
}
Tensor hermite_polynomial_he_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return hermite_polynomial_he_forward<double>(a, b);
    }, "hermite_polynomial_he");
}
Tensor laguerre_polynomial_l_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return laguerre_polynomial_l_forward<double>(a, b);
    }, "laguerre_polynomial_l");
}
Tensor legendre_polynomial_p_cpu(const Tensor& x, const Tensor& n) {
    return binary_float_kernel(x, n, [](double a, double b) {
        return legendre_polynomial_p_forward<double>(a, b);
    }, "legendre_polynomial_p");
}

// ---------------------------------------------------------------------------
// Two-tensor specials
// ---------------------------------------------------------------------------

Tensor zeta_cpu(const Tensor& s, const Tensor& q) {
    return binary_float_kernel(s, q, [](double a, double b) {
        return zeta<double>(a, b);
    }, "zeta");
}
Tensor gammainc_cpu(const Tensor& a, const Tensor& x) {
    return binary_float_kernel(a, x, [](double p, double v) {
        return calc_igamma(p, v);
    }, "gammainc");
}
Tensor gammaincc_cpu(const Tensor& a, const Tensor& x) {
    return binary_float_kernel(a, x, [](double p, double v) {
        return calc_igammac(p, v);
    }, "gammaincc");
}
Tensor polygamma_cpu(int64_t n, const Tensor& x) {
    return float_math_kernel(x, [n](double v) {
        return calc_polygamma(v, static_cast<int>(n));
    }, "polygamma");
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, SpecialKernels) {
    // Unary specials
    m.impl("bessel_j0", bessel_j0_cpu);
    m.impl("bessel_j1", bessel_j1_cpu);
    m.impl("bessel_y0", bessel_y0_cpu);
    m.impl("bessel_y1", bessel_y1_cpu);
    m.impl("airy_ai", airy_ai_cpu);
    m.impl("spherical_bessel_j0", spherical_bessel_j0_cpu);
    m.impl("modified_bessel_i1", modified_bessel_i1_cpu);
    m.impl("modified_bessel_k0", modified_bessel_k0_cpu);
    m.impl("modified_bessel_k1", modified_bessel_k1_cpu);
    m.impl("scaled_modified_bessel_k0", scaled_modified_bessel_k0_cpu);
    m.impl("scaled_modified_bessel_k1", scaled_modified_bessel_k1_cpu);
    m.impl("i0e", i0e_cpu);
    m.impl("i1", i1_cpu);
    m.impl("i1e", i1e_cpu);
    // Polynomial family
    m.impl("chebyshev_polynomial_t", chebyshev_polynomial_t_cpu);
    m.impl("chebyshev_polynomial_u", chebyshev_polynomial_u_cpu);
    m.impl("chebyshev_polynomial_v", chebyshev_polynomial_v_cpu);
    m.impl("chebyshev_polynomial_w", chebyshev_polynomial_w_cpu);
    m.impl("shifted_chebyshev_polynomial_t", shifted_chebyshev_polynomial_t_cpu);
    m.impl("shifted_chebyshev_polynomial_u", shifted_chebyshev_polynomial_u_cpu);
    m.impl("shifted_chebyshev_polynomial_v", shifted_chebyshev_polynomial_v_cpu);
    m.impl("shifted_chebyshev_polynomial_w", shifted_chebyshev_polynomial_w_cpu);
    m.impl("hermite_polynomial_h", hermite_polynomial_h_cpu);
    m.impl("hermite_polynomial_he", hermite_polynomial_he_cpu);
    m.impl("laguerre_polynomial_l", laguerre_polynomial_l_cpu);
    m.impl("legendre_polynomial_p", legendre_polynomial_p_cpu);
    // Two-tensor / parametric
    m.impl("zeta", zeta_cpu);
    m.impl("gammainc", gammainc_cpu);
    m.impl("gammaincc", gammaincc_cpu);
    m.impl("polygamma", polygamma_cpu);
}

}  // namespace tensorplay
