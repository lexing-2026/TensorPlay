#include "SpecialKernelUtils.cuh"

#include <SpecialMath.h>

namespace tensorplay::cuda {
namespace {

using special_detail::binary_float_cuda;
using special_detail::typed_binary_cuda;
using tensorplay::special_math::calc_xlog1py;
using tensorplay::special_math::chebyshev_polynomial_t_forward;
using tensorplay::special_math::chebyshev_polynomial_u_forward;
using tensorplay::special_math::chebyshev_polynomial_v_forward;
using tensorplay::special_math::chebyshev_polynomial_w_forward;
using tensorplay::special_math::hermite_polynomial_h_forward;
using tensorplay::special_math::hermite_polynomial_he_forward;
using tensorplay::special_math::laguerre_polynomial_l_forward;
using tensorplay::special_math::legendre_polynomial_p_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_t_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_u_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_v_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_w_forward;
using tensorplay::special_math::zeta;

struct XLog1pyFn {
    template <typename T>
    __device__ T operator()(T x, T y) const { return calc_xlog1py(x, y); }
};

Tensor chebyshev_polynomial_t_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_t_forward(a, b);
    }, "chebyshev_polynomial_t");
}
Tensor chebyshev_polynomial_u_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_u_forward(a, b);
    }, "chebyshev_polynomial_u");
}
Tensor chebyshev_polynomial_v_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_v_forward(a, b);
    }, "chebyshev_polynomial_v");
}
Tensor chebyshev_polynomial_w_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_w_forward(a, b);
    }, "chebyshev_polynomial_w");
}
Tensor shifted_chebyshev_polynomial_t_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_t_forward(a, b);
    }, "shifted_chebyshev_polynomial_t");
}
Tensor shifted_chebyshev_polynomial_u_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_u_forward(a, b);
    }, "shifted_chebyshev_polynomial_u");
}
Tensor shifted_chebyshev_polynomial_v_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_v_forward(a, b);
    }, "shifted_chebyshev_polynomial_v");
}
Tensor shifted_chebyshev_polynomial_w_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_w_forward(a, b);
    }, "shifted_chebyshev_polynomial_w");
}
Tensor hermite_polynomial_h_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return hermite_polynomial_h_forward(a, b);
    }, "hermite_polynomial_h");
}
Tensor hermite_polynomial_he_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return hermite_polynomial_he_forward(a, b);
    }, "hermite_polynomial_he");
}
Tensor laguerre_polynomial_l_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return laguerre_polynomial_l_forward(a, b);
    }, "laguerre_polynomial_l");
}
Tensor legendre_polynomial_p_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return legendre_polynomial_p_forward(a, b);
    }, "legendre_polynomial_p");
}
Tensor zeta_cuda(const Tensor& s, const Tensor& q) {
    return binary_float_cuda(s, q, [] __device__ (double a, double b) {
        return zeta(a, b);
    }, "zeta");
}
Tensor xlog1py_cuda(const Tensor& x, const Tensor& y) {
    return typed_binary_cuda(x, y, XLog1pyFn{});
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SpecialBinaryKernels) {
    m.impl("chebyshev_polynomial_t", chebyshev_polynomial_t_cuda);
    m.impl("chebyshev_polynomial_u", chebyshev_polynomial_u_cuda);
    m.impl("chebyshev_polynomial_v", chebyshev_polynomial_v_cuda);
    m.impl("chebyshev_polynomial_w", chebyshev_polynomial_w_cuda);
    m.impl("shifted_chebyshev_polynomial_t", shifted_chebyshev_polynomial_t_cuda);
    m.impl("shifted_chebyshev_polynomial_u", shifted_chebyshev_polynomial_u_cuda);
    m.impl("shifted_chebyshev_polynomial_v", shifted_chebyshev_polynomial_v_cuda);
    m.impl("shifted_chebyshev_polynomial_w", shifted_chebyshev_polynomial_w_cuda);
    m.impl("hermite_polynomial_h", hermite_polynomial_h_cuda);
    m.impl("hermite_polynomial_he", hermite_polynomial_he_cuda);
    m.impl("laguerre_polynomial_l", laguerre_polynomial_l_cuda);
    m.impl("legendre_polynomial_p", legendre_polynomial_p_cuda);
    m.impl("zeta", zeta_cuda);
    m.impl("xlog1py", xlog1py_cuda);
}

}  // namespace tensorplay::cuda
