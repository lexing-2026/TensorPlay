//
// (2.15.0a0) via p10/include/SpecialMath.h; the wrappers use the common
// float_math_kernel / binary_float_kernel helper pattern.
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
#include <tuple>
#include <vector>

namespace tensorplay {
namespace {

using namespace tensorplay::parallel;

using tensorplay::special_math::airy_ai_forward;
using tensorplay::special_math::bessel_j0_forward;
using tensorplay::special_math::bessel_j1_forward;
using tensorplay::special_math::bessel_y0_forward;
using tensorplay::special_math::bessel_y1_forward;
using tensorplay::special_math::calc_entr;
using tensorplay::special_math::calc_erfcx;
using tensorplay::special_math::calc_digamma;
using tensorplay::special_math::trigamma;
using tensorplay::special_math::calc_i0e;
using tensorplay::special_math::calc_log_ndtr;
using tensorplay::special_math::calc_ndtr;
using tensorplay::special_math::calc_ndtri;
using tensorplay::special_math::calc_xlog1py;
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
using tensorplay::special_math::modified_bessel_i0_forward;
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
    // Reduced-width inputs are evaluated in Float32 and narrowed once at the
    // end; the loops below only ever address float or double buffers.
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor ac = a_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, compute_dt, a_in.device());
    int64_t n = out.numel();
    if (compute_dt == DType::Float64) {
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
    return (dt == compute_dt) ? out : out.to(dt);
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
Tensor modified_bessel_i0_cpu(const Tensor& self) {
    return float_math_kernel(self,
                             [](double x) { return modified_bessel_i0_forward(x); },
                             "modified_bessel_i0");
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
// Error-function tail and normal-distribution family
// ---------------------------------------------------------------------------

Tensor erfcx_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_erfcx(x); }, "erfcx");
}
Tensor ndtr_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_ndtr(x); }, "ndtr");
}
Tensor ndtri_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_ndtri(x); }, "ndtri");
}
Tensor log_ndtr_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_log_ndtr(x); }, "log_ndtr");
}
Tensor entr_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return calc_entr(x); }, "entr");
}
Tensor xlog1py_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b,
                               [](double x, double y) { return calc_xlog1py(x, y); },
                               "xlog1py");
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
    if (n < 0) {
        TP_THROW(RuntimeError, "polygamma(n, x) does not support negative n");
    }
    if (n > std::numeric_limits<int>::max()) {
        TP_THROW(RuntimeError, "polygamma order is too large: ", n);
    }
    if (n == 0) {
        return float_math_kernel(x, [](double v) { return calc_digamma(v); }, "polygamma");
    }
    if (n == 1) {
        return float_math_kernel(x, [](double v) { return trigamma(v); }, "polygamma");
    }
    return float_math_kernel(x, [n](double v) {
        return calc_polygamma(v, static_cast<int>(n));
    }, "polygamma");
}

}  // namespace

// External linkage: the backend-neutral composite layer uses
// non-CPU inputs through these kernels (Pointwise.cpp declarations).
namespace cpu {

// Public names of the regularized incomplete gamma pair; the same
// calc_igamma / calc_igammac implementations as gammainc / gammaincc.
Tensor igamma_cpu(const Tensor& a, const Tensor& x) {
    return binary_float_kernel(a, x, [](double p, double v) {
        return calc_igamma(p, v);
    }, "igamma");
}
Tensor igammac_cpu(const Tensor& a, const Tensor& x) {
    return binary_float_kernel(a, x, [](double p, double v) {
        return calc_igammac(p, v);
    }, "igammac");
}

// mantissa in [0.5, 1) (or exactly 0), exponent as int32; computed via
// std::frexp in Float64 -- the decomposition of a Float32/16 input is exact
// on the double path and casts back losslessly.
std::tuple<Tensor, Tensor> frexp_cpu(const Tensor& self) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "frexp(): only supports floating-point dtypes");
    }
    const std::vector<int64_t> shape(self.shape());
    Tensor mantissa = Tensor::empty(shape, self.dtype(), self.device());
    Tensor exponent = Tensor::empty(shape, DType::Int32, self.device());
    const int64_t n = self.numel();
    if (n == 0) return {mantissa, exponent};
    Tensor s32 = (self.dtype() == DType::Float64 || self.dtype() == DType::Float32)
        ? self.contiguous() : self.to(DType::Float32).contiguous();
    Tensor mant32 = (self.dtype() == DType::Float32)
        ? mantissa : Tensor::empty(shape, DType::Float32, self.device());

    if (self.dtype() == DType::Float64) {
        const double* sp = s32.data_ptr<double>();
        double* mp = mantissa.data_ptr<double>();
        int32_t* ep = exponent.data_ptr<int32_t>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                int ex = 0;
                mp[i] = std::frexp(sp[i], &ex);
                ep[i] = ex;
            }
        });
    } else {
        const float* sp = s32.data_ptr<float>();
        float* mp = mant32.data_ptr<float>();
        int32_t* ep = exponent.data_ptr<int32_t>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                int ex = 0;
                mp[i] = std::frexp(static_cast<double>(sp[i]), &ex);
                ep[i] = ex;
            }
        });
        if (self.dtype() != DType::Float32) {
            mantissa = mant32.to(self.dtype());
        }
    }
    return {mantissa, exponent};
}

}  // namespace cpu

TENSORPLAY_LIBRARY_IMPL(CPU, SpecialKernels) {
    // Unary specials
    m.impl("bessel_j0", bessel_j0_cpu);
    m.impl("bessel_j1", bessel_j1_cpu);
    m.impl("bessel_y0", bessel_y0_cpu);
    m.impl("bessel_y1", bessel_y1_cpu);
    m.impl("airy_ai", airy_ai_cpu);
    m.impl("spherical_bessel_j0", spherical_bessel_j0_cpu);
    m.impl("modified_bessel_i0", modified_bessel_i0_cpu);
    m.impl("modified_bessel_i1", modified_bessel_i1_cpu);
    m.impl("modified_bessel_k0", modified_bessel_k0_cpu);
    m.impl("modified_bessel_k1", modified_bessel_k1_cpu);
    m.impl("scaled_modified_bessel_k0", scaled_modified_bessel_k0_cpu);
    m.impl("scaled_modified_bessel_k1", scaled_modified_bessel_k1_cpu);
    m.impl("i0e", i0e_cpu);
    m.impl("i1", i1_cpu);
    m.impl("i1e", i1e_cpu);
    // Error-function tail / normal distribution
    m.impl("erfcx", erfcx_cpu);
    m.impl("ndtr", ndtr_cpu);
    m.impl("ndtri", ndtri_cpu);
    m.impl("log_ndtr", log_ndtr_cpu);
    m.impl("entr", entr_cpu);
    m.impl("xlog1py", xlog1py_cpu);
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
    // Public names of the regularized incomplete gamma pair; the
    // kernels are the same calc_igamma / calc_igammac implementations.
    m.impl("igamma", cpu::igamma_cpu);
    m.impl("igammac", cpu::igammac_cpu);
    m.impl("frexp", cpu::frexp_cpu);
}

}  // namespace tensorplay
