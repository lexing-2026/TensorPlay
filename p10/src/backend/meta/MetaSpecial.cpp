// Shape-only kernels for the special-function families on the meta device:
// the Bessel and Airy tables, the error-function tail and the normal
// distribution maps, the orthogonal polynomial pairs, the zeta and
// incomplete-gamma pairs, the xlog1py product form and the polygamma orders.
//
// Every sweep here computes at floating precision and reads no values, so a
// unary map repeats its operand's geometry and a two-operand map broadcasts
// the operands' shapes under the promoted type.  Operands with no floating
// width -- integers, booleans, complex planes -- are answered in single
// precision, which is the only width the elementwise pass knows how to
// compute in.

#include "MetaCommon.h"

#include "Dispatcher.h"
#include "Exception.h"
#include "OutWrite.h"

#include <limits>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// The unary sweeps: a floating operand keeps its width, everything else is
// answered in single precision.
Tensor float_math_answer(const Tensor& self) {
    return empty_meta(shape_of(self), float_math_dtype(self.dtype()),
                      self.device());
}

// The two-operand sweeps: broadcast geometry under the promoted type, lifted
// to single precision when no floating width survives the promotion.  The
// operands must share a device.
Tensor binary_float_answer(const Tensor& a, const Tensor& b) {
    check_all_meta(a, b);
    return empty_meta(broadcast2(a, b),
                      float_math_dtype(promoteTypes(a.dtype(), b.dtype())),
                      a.device());
}

#define META_SPECIAL_UNARY(NAME)                                              \
    Tensor NAME##_kernel(const Tensor& self) { return float_math_answer(self); }

#define META_SPECIAL_BINARY(NAME)                                             \
    Tensor NAME##_kernel(const Tensor& a, const Tensor& b) {                  \
        return binary_float_answer(a, b);                                     \
    }

// The Bessel and Airy tables.
META_SPECIAL_UNARY(airy_ai)
META_SPECIAL_UNARY(bessel_j0)
META_SPECIAL_UNARY(bessel_j1)
META_SPECIAL_UNARY(bessel_y0)
META_SPECIAL_UNARY(bessel_y1)
META_SPECIAL_UNARY(spherical_bessel_j0)
META_SPECIAL_UNARY(modified_bessel_i0)
META_SPECIAL_UNARY(modified_bessel_i1)
META_SPECIAL_UNARY(modified_bessel_k0)
META_SPECIAL_UNARY(modified_bessel_k1)
META_SPECIAL_UNARY(scaled_modified_bessel_k0)
META_SPECIAL_UNARY(scaled_modified_bessel_k1)

// The error-function tail and the normal distribution.
META_SPECIAL_UNARY(erfcx)
META_SPECIAL_UNARY(ndtr)
META_SPECIAL_UNARY(ndtri)
META_SPECIAL_UNARY(log_ndtr)

// The remaining unary members of the family.
META_SPECIAL_UNARY(i0e)
META_SPECIAL_UNARY(i1)
META_SPECIAL_UNARY(i1e)

// The two-operand members: zeta and the incomplete-gamma pair.
META_SPECIAL_BINARY(zeta)
META_SPECIAL_BINARY(gammainc)
META_SPECIAL_BINARY(gammaincc)
META_SPECIAL_BINARY(igamma)
META_SPECIAL_BINARY(igammac)

// The xlog1py product form.
META_SPECIAL_BINARY(xlog1py)

// The orthogonal polynomial pairs: chebyshev (four degrees), their shifted
// variants, hermite (physicists' and probabilists'), laguerre and legendre.
META_SPECIAL_BINARY(chebyshev_polynomial_t)
META_SPECIAL_BINARY(chebyshev_polynomial_u)
META_SPECIAL_BINARY(chebyshev_polynomial_v)
META_SPECIAL_BINARY(chebyshev_polynomial_w)
META_SPECIAL_BINARY(shifted_chebyshev_polynomial_t)
META_SPECIAL_BINARY(shifted_chebyshev_polynomial_u)
META_SPECIAL_BINARY(shifted_chebyshev_polynomial_v)
META_SPECIAL_BINARY(shifted_chebyshev_polynomial_w)
META_SPECIAL_BINARY(hermite_polynomial_h)
META_SPECIAL_BINARY(hermite_polynomial_he)
META_SPECIAL_BINARY(laguerre_polynomial_l)
META_SPECIAL_BINARY(legendre_polynomial_p)

#undef META_SPECIAL_UNARY
#undef META_SPECIAL_BINARY

// An order the recurrence cannot be run for is refused by every spelling of
// the n-th log-gamma derivative.
void require_polygamma_order(int64_t n) {
    if (n < 0) {
        TP_THROW(RuntimeError, "polygamma(n, x) does not support negative n");
    }
    if (n > std::numeric_limits<int>::max()) {
        TP_THROW(RuntimeError, "polygamma order is too large: ", n);
    }
}

Tensor polygamma_kernel(int64_t n, const Tensor& x) {
    require_polygamma_order(n);
    return float_math_answer(x);
}

// The out= forms take a destination that already names the answer's geometry;
// one that needs a resize or a safe cast is rewritten by the same write.
Tensor& digamma_out_kernel(const Tensor& self, Tensor& out) {
    return write_out(out, float_math_answer(self));
}

Tensor& i0_out_kernel(const Tensor& self, Tensor& out) {
    return write_out(out, float_math_answer(self));
}

Tensor& polygamma_out_kernel(int64_t n, const Tensor& x, Tensor& out) {
    require_polygamma_order(n);
    return write_out(out, float_math_answer(x));
}

Tensor& igamma_out_kernel(const Tensor& a, const Tensor& x, Tensor& out) {
    return write_out(out, binary_float_answer(a, x));
}

Tensor& igammac_out_kernel(const Tensor& a, const Tensor& x, Tensor& out) {
    return write_out(out, binary_float_answer(a, x));
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaSpecial) {
    m.impl("airy_ai", airy_ai_kernel);
    m.impl("bessel_j0", bessel_j0_kernel);
    m.impl("bessel_j1", bessel_j1_kernel);
    m.impl("bessel_y0", bessel_y0_kernel);
    m.impl("bessel_y1", bessel_y1_kernel);
    m.impl("spherical_bessel_j0", spherical_bessel_j0_kernel);
    m.impl("modified_bessel_i0", modified_bessel_i0_kernel);
    m.impl("modified_bessel_i1", modified_bessel_i1_kernel);
    m.impl("modified_bessel_k0", modified_bessel_k0_kernel);
    m.impl("modified_bessel_k1", modified_bessel_k1_kernel);
    m.impl("scaled_modified_bessel_k0", scaled_modified_bessel_k0_kernel);
    m.impl("scaled_modified_bessel_k1", scaled_modified_bessel_k1_kernel);
    m.impl("erfcx", erfcx_kernel);
    m.impl("ndtr", ndtr_kernel);
    m.impl("ndtri", ndtri_kernel);
    m.impl("log_ndtr", log_ndtr_kernel);
    m.impl("i0e", i0e_kernel);
    m.impl("i1", i1_kernel);
    m.impl("i1e", i1e_kernel);
    m.impl("zeta", zeta_kernel);
    m.impl("gammainc", gammainc_kernel);
    m.impl("gammaincc", gammaincc_kernel);
    m.impl("igamma", igamma_kernel);
    m.impl("igammac", igammac_kernel);
    m.impl("xlog1py", xlog1py_kernel);
    m.impl("chebyshev_polynomial_t", chebyshev_polynomial_t_kernel);
    m.impl("chebyshev_polynomial_u", chebyshev_polynomial_u_kernel);
    m.impl("chebyshev_polynomial_v", chebyshev_polynomial_v_kernel);
    m.impl("chebyshev_polynomial_w", chebyshev_polynomial_w_kernel);
    m.impl("shifted_chebyshev_polynomial_t", shifted_chebyshev_polynomial_t_kernel);
    m.impl("shifted_chebyshev_polynomial_u", shifted_chebyshev_polynomial_u_kernel);
    m.impl("shifted_chebyshev_polynomial_v", shifted_chebyshev_polynomial_v_kernel);
    m.impl("shifted_chebyshev_polynomial_w", shifted_chebyshev_polynomial_w_kernel);
    m.impl("hermite_polynomial_h", hermite_polynomial_h_kernel);
    m.impl("hermite_polynomial_he", hermite_polynomial_he_kernel);
    m.impl("laguerre_polynomial_l", laguerre_polynomial_l_kernel);
    m.impl("legendre_polynomial_p", legendre_polynomial_p_kernel);
    m.impl("polygamma", polygamma_kernel);
    m.impl("polygamma.out", polygamma_out_kernel);
    m.impl("igamma.out", igamma_out_kernel);
    m.impl("igammac.out", igammac_out_kernel);
    m.impl("digamma.out", digamma_out_kernel);
    m.impl("i0.out", i0_out_kernel);
}

} // namespace meta
} // namespace tensorplay
