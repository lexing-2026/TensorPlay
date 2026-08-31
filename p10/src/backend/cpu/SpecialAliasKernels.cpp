// CPU-side registration for the special_* operator family.
//
// Each special_<name> op shares its schema and numerics with the
// de-prefixed <name> operator. Two wiring flavors:
//   - direct reference: an existing CPU kernel with external linkage and
//     an identical signature is registered under the special name;
//   - adapter: a thin wrapper (namespace tensorplay::special_alias) that
//     converts argument types where they differ, lifts Scalars to
//     0-dim tensors, and routes through the dispatcher to the twin
//     operator; out-variants rebind the out tensor to the result.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "DType.h"
#include "Exception.h"
#include "TypePromotion.h"

#include "tensorplay/ops/TensorRedispatchGenerated.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <optional>
#include <vector>

namespace tensorplay {

// Existing CPU kernels referenced directly below.
namespace cpu {
Tensor digamma_cpu(const Tensor& self);
Tensor erf_kernel(const Tensor& self);
Tensor erfc_kernel(const Tensor& self);
Tensor erfinv_cpu(const Tensor& self);
Tensor exp2_cpu(const Tensor& self);
Tensor expm1_kernel(const Tensor& self);
Tensor igamma_cpu(const Tensor& self, const Tensor& other);
Tensor igammac_cpu(const Tensor& self, const Tensor& other);
Tensor lgamma_kernel(const Tensor& self);
Tensor i0_cpu(const Tensor& self);
Tensor log1p_kernel(const Tensor& self);
Tensor sinc_cpu(const Tensor& self);
Tensor xlogy_cpu(const Tensor& self, const Tensor& other);
}  // namespace cpu

namespace special_alias {

namespace ops = tensorplay::tpx::ops;

// 0-dim tensor carrying a Scalar with the dtype/device of `like`, so
// promotion inside the twin kernel matches tensor-scalar semantics.
Tensor scalar_like(const Scalar& value, const Tensor& like) {
    return ops::full(std::vector<int64_t>{}, value, like.dtype(),
                     like.device());
}

std::optional<Scalar> eps_to_scalar(const std::optional<double>& eps) {
    if (eps.has_value()) return Scalar(eps.value());
    return std::nullopt;
}

// Adapter prototypes (the CUDA registration unit declares these too).
Tensor alias_entr(const Tensor& self);
Tensor& alias_entr_out(const Tensor& self, Tensor& out);
Tensor alias_ndtri(const Tensor& self);
Tensor& alias_ndtri_out(const Tensor& self, Tensor& out);
Tensor alias_log_ndtr(const Tensor& self);
Tensor& alias_log_ndtr_out(const Tensor& self, Tensor& out);
Tensor alias_expm1(const Tensor& self);
Tensor& alias_expm1_out(const Tensor& self, Tensor& out);
Tensor alias_exp2(const Tensor& self);
Tensor& alias_exp2_out(const Tensor& self, Tensor& out);
Tensor alias_psi(const Tensor& self);
Tensor& alias_psi_out(const Tensor& self, Tensor& out);
Tensor alias_digamma(const Tensor& self);
Tensor& alias_digamma_out(const Tensor& self, Tensor& out);
Tensor alias_gammaln(const Tensor& self);
Tensor& alias_gammaln_out(const Tensor& self, Tensor& out);
Tensor alias_erf(const Tensor& self);
Tensor& alias_erf_out(const Tensor& self, Tensor& out);
Tensor alias_erfc(const Tensor& self);
Tensor& alias_erfc_out(const Tensor& self, Tensor& out);
Tensor alias_erfcx(const Tensor& self);
Tensor& alias_erfcx_out(const Tensor& self, Tensor& out);
Tensor alias_erfinv(const Tensor& self);
Tensor& alias_erfinv_out(const Tensor& self, Tensor& out);
Tensor alias_ndtr(const Tensor& self);
Tensor& alias_ndtr_out(const Tensor& self, Tensor& out);
Tensor alias_xlog1py(const Tensor& self, const Tensor& other);
Tensor alias_xlog1py_self_scalar(Scalar self, const Tensor& other);
Tensor alias_xlog1py_other_scalar(const Tensor& self, Scalar other);
Tensor& alias_xlog1py_out(const Tensor& self, const Tensor& other, Tensor& out);
Tensor& alias_xlog1py_self_scalar_out(Scalar self, const Tensor& other, Tensor& out);
Tensor& alias_xlog1py_other_scalar_out(const Tensor& self, Scalar other, Tensor& out);
Tensor alias_xlogy(const Tensor& self, const Tensor& other);
Tensor alias_xlogy_self_scalar(Scalar self, const Tensor& other);
Tensor alias_xlogy_other_scalar(const Tensor& self, Scalar other);
Tensor& alias_xlogy_out(const Tensor& self, const Tensor& other, Tensor& out);
Tensor& alias_xlogy_self_scalar_out(Scalar self, const Tensor& other, Tensor& out);
Tensor& alias_xlogy_other_scalar_out(const Tensor& self, Scalar other, Tensor& out);
Tensor alias_zeta(const Tensor& self, const Tensor& other);
Tensor alias_zeta_self_scalar(Scalar self, const Tensor& other);
Tensor alias_zeta_other_scalar(const Tensor& self, Scalar other);
Tensor& alias_zeta_out(const Tensor& self, const Tensor& other, Tensor& out);
Tensor& alias_zeta_self_scalar_out(Scalar self, const Tensor& other, Tensor& out);
Tensor& alias_zeta_other_scalar_out(const Tensor& self, Scalar other, Tensor& out);
Tensor alias_i0(const Tensor& self);
Tensor& alias_i0_out(const Tensor& self, Tensor& out);
Tensor alias_i0e(const Tensor& self);
Tensor& alias_i0e_out(const Tensor& self, Tensor& out);
Tensor alias_i1(const Tensor& self);
Tensor& alias_i1_out(const Tensor& self, Tensor& out);
Tensor alias_i1e(const Tensor& self);
Tensor& alias_i1e_out(const Tensor& self, Tensor& out);
Tensor alias_logit(const Tensor& self, std::optional<double> eps);
Tensor& alias_logit_out(const Tensor& self, std::optional<double> eps, Tensor& out);
Tensor alias_polygamma(int64_t n, const Tensor& self);
Tensor& alias_polygamma_out(int64_t n, const Tensor& self, Tensor& out);
Tensor alias_logsumexp(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim);
Tensor& alias_logsumexp_out(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, Tensor& out);
Tensor alias_expit(const Tensor& self);
Tensor& alias_expit_out(const Tensor& self, Tensor& out);
Tensor alias_sinc(const Tensor& self);
Tensor& alias_sinc_out(const Tensor& self, Tensor& out);
Tensor alias_round(const Tensor& self, int64_t decimals);
Tensor& alias_round_out(const Tensor& self, int64_t decimals, Tensor& out);
Tensor alias_log1p(const Tensor& self);
Tensor& alias_log1p_out(const Tensor& self, Tensor& out);
Tensor alias_log_softmax(const Tensor& self, int64_t dim, std::optional<DType> dtype);
Tensor& alias_gammainc_out(const Tensor& self, const Tensor& other, Tensor& out);
Tensor alias_gammainc(const Tensor& self, const Tensor& other);
Tensor& alias_gammaincc_out(const Tensor& self, const Tensor& other, Tensor& out);
Tensor alias_gammaincc(const Tensor& self, const Tensor& other);
Tensor alias_multigammaln(const Tensor& self, int64_t p);
Tensor& alias_multigammaln_out(const Tensor& self, int64_t p, Tensor& out);
Tensor alias_softmax(const Tensor& self, int64_t dim, std::optional<DType> dtype);
Tensor alias_airy_ai(const Tensor& x);
Tensor& alias_airy_ai_out(const Tensor& x, Tensor& out);
Tensor alias_bessel_j0(const Tensor& self);
Tensor& alias_bessel_j0_out(const Tensor& self, Tensor& out);
Tensor alias_bessel_j1(const Tensor& self);
Tensor& alias_bessel_j1_out(const Tensor& self, Tensor& out);
Tensor alias_bessel_y0(const Tensor& self);
Tensor& alias_bessel_y0_out(const Tensor& self, Tensor& out);
Tensor alias_bessel_y1(const Tensor& self);
Tensor& alias_bessel_y1_out(const Tensor& self, Tensor& out);
Tensor alias_chebyshev_polynomial_t(const Tensor& x, const Tensor& n);
Tensor alias_chebyshev_polynomial_t_x_scalar(Scalar x, const Tensor& n);
Tensor alias_chebyshev_polynomial_t_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_chebyshev_polynomial_t_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_t_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_t_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_chebyshev_polynomial_u(const Tensor& x, const Tensor& n);
Tensor alias_chebyshev_polynomial_u_x_scalar(Scalar x, const Tensor& n);
Tensor alias_chebyshev_polynomial_u_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_chebyshev_polynomial_u_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_u_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_u_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_chebyshev_polynomial_v(const Tensor& x, const Tensor& n);
Tensor alias_chebyshev_polynomial_v_x_scalar(Scalar x, const Tensor& n);
Tensor alias_chebyshev_polynomial_v_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_chebyshev_polynomial_v_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_v_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_v_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_chebyshev_polynomial_w(const Tensor& x, const Tensor& n);
Tensor alias_chebyshev_polynomial_w_x_scalar(Scalar x, const Tensor& n);
Tensor alias_chebyshev_polynomial_w_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_chebyshev_polynomial_w_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_w_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_chebyshev_polynomial_w_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_hermite_polynomial_h(const Tensor& x, const Tensor& n);
Tensor alias_hermite_polynomial_h_x_scalar(Scalar x, const Tensor& n);
Tensor alias_hermite_polynomial_h_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_hermite_polynomial_h_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_hermite_polynomial_h_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_hermite_polynomial_h_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_hermite_polynomial_he(const Tensor& x, const Tensor& n);
Tensor alias_hermite_polynomial_he_x_scalar(Scalar x, const Tensor& n);
Tensor alias_hermite_polynomial_he_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_hermite_polynomial_he_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_hermite_polynomial_he_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_hermite_polynomial_he_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_laguerre_polynomial_l(const Tensor& x, const Tensor& n);
Tensor alias_laguerre_polynomial_l_x_scalar(Scalar x, const Tensor& n);
Tensor alias_laguerre_polynomial_l_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_laguerre_polynomial_l_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_laguerre_polynomial_l_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_laguerre_polynomial_l_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_legendre_polynomial_p(const Tensor& x, const Tensor& n);
Tensor alias_legendre_polynomial_p_x_scalar(Scalar x, const Tensor& n);
Tensor alias_legendre_polynomial_p_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_legendre_polynomial_p_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_legendre_polynomial_p_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_legendre_polynomial_p_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_modified_bessel_i0(const Tensor& self);
Tensor& alias_modified_bessel_i0_out(const Tensor& self, Tensor& out);
Tensor alias_modified_bessel_i1(const Tensor& self);
Tensor& alias_modified_bessel_i1_out(const Tensor& self, Tensor& out);
Tensor alias_modified_bessel_k0(const Tensor& self);
Tensor& alias_modified_bessel_k0_out(const Tensor& self, Tensor& out);
Tensor alias_modified_bessel_k1(const Tensor& self);
Tensor& alias_modified_bessel_k1_out(const Tensor& self, Tensor& out);
Tensor alias_scaled_modified_bessel_k0(const Tensor& x);
Tensor& alias_scaled_modified_bessel_k0_out(const Tensor& x, Tensor& out);
Tensor alias_scaled_modified_bessel_k1(const Tensor& x);
Tensor& alias_scaled_modified_bessel_k1_out(const Tensor& x, Tensor& out);
Tensor alias_shifted_chebyshev_polynomial_t(const Tensor& x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_t_x_scalar(Scalar x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_t_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_shifted_chebyshev_polynomial_t_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_t_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_t_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_shifted_chebyshev_polynomial_u(const Tensor& x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_u_x_scalar(Scalar x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_u_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_shifted_chebyshev_polynomial_u_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_u_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_u_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_shifted_chebyshev_polynomial_v(const Tensor& x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_v_x_scalar(Scalar x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_v_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_shifted_chebyshev_polynomial_v_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_v_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_v_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_shifted_chebyshev_polynomial_w(const Tensor& x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_w_x_scalar(Scalar x, const Tensor& n);
Tensor alias_shifted_chebyshev_polynomial_w_n_scalar(const Tensor& x, Scalar n);
Tensor& alias_shifted_chebyshev_polynomial_w_out(const Tensor& x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_w_x_scalar_out(Scalar x, const Tensor& n, Tensor& out);
Tensor& alias_shifted_chebyshev_polynomial_w_n_scalar_out(const Tensor& x, Scalar n, Tensor& out);
Tensor alias_spherical_bessel_j0(const Tensor& x);
Tensor& alias_spherical_bessel_j0_out(const Tensor& x, Tensor& out);

// special_entr -> entr
Tensor alias_entr(const Tensor& self) {
    return detail::redispatch_entr_function(self);
}

Tensor& alias_entr_out(const Tensor& self, Tensor& out) {
    out = alias_entr(self);
    return out;
}

// special_ndtri -> ndtri
Tensor alias_ndtri(const Tensor& self) {
    return detail::redispatch_ndtri_function(self);
}

Tensor& alias_ndtri_out(const Tensor& self, Tensor& out) {
    out = alias_ndtri(self);
    return out;
}

// special_log_ndtr -> log_ndtr
Tensor alias_log_ndtr(const Tensor& self) {
    return detail::redispatch_log_ndtr_function(self);
}

Tensor& alias_log_ndtr_out(const Tensor& self, Tensor& out) {
    out = alias_log_ndtr(self);
    return out;
}

// special_expm1 -> expm1
Tensor alias_expm1(const Tensor& self) {
    return detail::redispatch_expm1_function(self);
}

Tensor& alias_expm1_out(const Tensor& self, Tensor& out) {
    out = alias_expm1(self);
    return out;
}

// special_exp2 -> exp2
Tensor alias_exp2(const Tensor& self) {
    return detail::redispatch_exp2_function(self);
}

Tensor& alias_exp2_out(const Tensor& self, Tensor& out) {
    out = alias_exp2(self);
    return out;
}

// special_psi -> digamma
Tensor alias_psi(const Tensor& self) {
    return detail::redispatch_digamma_function(self);
}

Tensor& alias_psi_out(const Tensor& self, Tensor& out) {
    out = alias_psi(self);
    return out;
}

// special_digamma -> digamma
Tensor alias_digamma(const Tensor& self) {
    return detail::redispatch_digamma_function(self);
}

Tensor& alias_digamma_out(const Tensor& self, Tensor& out) {
    out = alias_digamma(self);
    return out;
}

// special_gammaln -> lgamma
Tensor alias_gammaln(const Tensor& self) {
    return detail::redispatch_lgamma_function(self);
}

Tensor& alias_gammaln_out(const Tensor& self, Tensor& out) {
    out = alias_gammaln(self);
    return out;
}

// special_erf -> erf
Tensor alias_erf(const Tensor& self) {
    return detail::redispatch_erf_function(self);
}

Tensor& alias_erf_out(const Tensor& self, Tensor& out) {
    out = alias_erf(self);
    return out;
}

// special_erfc -> erfc
Tensor alias_erfc(const Tensor& self) {
    return detail::redispatch_erfc_function(self);
}

Tensor& alias_erfc_out(const Tensor& self, Tensor& out) {
    out = alias_erfc(self);
    return out;
}

// special_erfcx -> erfcx
Tensor alias_erfcx(const Tensor& self) {
    return detail::redispatch_erfcx_function(self);
}

Tensor& alias_erfcx_out(const Tensor& self, Tensor& out) {
    out = alias_erfcx(self);
    return out;
}

// special_erfinv -> erfinv
Tensor alias_erfinv(const Tensor& self) {
    return detail::redispatch_erfinv_function(self);
}

Tensor& alias_erfinv_out(const Tensor& self, Tensor& out) {
    out = alias_erfinv(self);
    return out;
}

// special_ndtr -> ndtr
Tensor alias_ndtr(const Tensor& self) {
    return detail::redispatch_ndtr_function(self);
}

Tensor& alias_ndtr_out(const Tensor& self, Tensor& out) {
    out = alias_ndtr(self);
    return out;
}

// special_xlog1py -> xlog1py
Tensor alias_xlog1py(const Tensor& self, const Tensor& other) {
    return detail::redispatch_xlog1py_function(self, other);
}

Tensor alias_xlog1py_self_scalar(Scalar self, const Tensor& other) {
    return detail::redispatch_xlog1py_function(scalar_like(self, other), other);
}

Tensor alias_xlog1py_other_scalar(const Tensor& self, Scalar other) {
    return detail::redispatch_xlog1py_function(self, scalar_like(other, self));
}

Tensor& alias_xlog1py_out(const Tensor& self, const Tensor& other, Tensor& out) {
    out = alias_xlog1py(self, other);
    return out;
}

Tensor& alias_xlog1py_self_scalar_out(Scalar self, const Tensor& other, Tensor& out) {
    out = alias_xlog1py_self_scalar(self, other);
    return out;
}

Tensor& alias_xlog1py_other_scalar_out(const Tensor& self, Scalar other, Tensor& out) {
    out = alias_xlog1py_other_scalar(self, other);
    return out;
}

// special_xlogy -> xlogy
Tensor alias_xlogy(const Tensor& self, const Tensor& other) {
    return detail::redispatch_xlogy_function(self, other);
}

Tensor alias_xlogy_self_scalar(Scalar self, const Tensor& other) {
    return detail::redispatch_xlogy_function(scalar_like(self, other), other);
}

Tensor alias_xlogy_other_scalar(const Tensor& self, Scalar other) {
    return detail::redispatch_xlogy_function(self, scalar_like(other, self));
}

Tensor& alias_xlogy_out(const Tensor& self, const Tensor& other, Tensor& out) {
    out = alias_xlogy(self, other);
    return out;
}

Tensor& alias_xlogy_self_scalar_out(Scalar self, const Tensor& other, Tensor& out) {
    out = alias_xlogy_self_scalar(self, other);
    return out;
}

Tensor& alias_xlogy_other_scalar_out(const Tensor& self, Scalar other, Tensor& out) {
    out = alias_xlogy_other_scalar(self, other);
    return out;
}

// special_zeta -> zeta
Tensor alias_zeta(const Tensor& self, const Tensor& other) {
    return detail::redispatch_zeta_function(self, other);
}

Tensor alias_zeta_self_scalar(Scalar self, const Tensor& other) {
    return detail::redispatch_zeta_function(scalar_like(self, other), other);
}

Tensor alias_zeta_other_scalar(const Tensor& self, Scalar other) {
    return detail::redispatch_zeta_function(self, scalar_like(other, self));
}

Tensor& alias_zeta_out(const Tensor& self, const Tensor& other, Tensor& out) {
    out = alias_zeta(self, other);
    return out;
}

Tensor& alias_zeta_self_scalar_out(Scalar self, const Tensor& other, Tensor& out) {
    out = alias_zeta_self_scalar(self, other);
    return out;
}

Tensor& alias_zeta_other_scalar_out(const Tensor& self, Scalar other, Tensor& out) {
    out = alias_zeta_other_scalar(self, other);
    return out;
}

// special_i0 -> i0
Tensor alias_i0(const Tensor& self) {
    return detail::redispatch_i0_function(self);
}

Tensor& alias_i0_out(const Tensor& self, Tensor& out) {
    out = alias_i0(self);
    return out;
}

// special_i0e -> i0e
Tensor alias_i0e(const Tensor& self) {
    return detail::redispatch_i0e_function(self);
}

Tensor& alias_i0e_out(const Tensor& self, Tensor& out) {
    out = alias_i0e(self);
    return out;
}

// special_i1 -> i1
Tensor alias_i1(const Tensor& self) {
    return detail::redispatch_i1_function(self);
}

Tensor& alias_i1_out(const Tensor& self, Tensor& out) {
    out = alias_i1(self);
    return out;
}

// special_i1e -> i1e
Tensor alias_i1e(const Tensor& self) {
    return detail::redispatch_i1e_function(self);
}

Tensor& alias_i1e_out(const Tensor& self, Tensor& out) {
    out = alias_i1e(self);
    return out;
}

// special_logit -> logit
Tensor alias_logit(const Tensor& self, std::optional<double> eps) {
    return detail::redispatch_logit_function(self, eps_to_scalar(eps));
}

Tensor& alias_logit_out(const Tensor& self, std::optional<double> eps, Tensor& out) {
    out = alias_logit(self, eps);
    return out;
}

// special_polygamma -> polygamma
Tensor alias_polygamma(int64_t n, const Tensor& self) {
    return detail::redispatch_polygamma_function(n, self);
}

Tensor& alias_polygamma_out(int64_t n, const Tensor& self, Tensor& out) {
    out = alias_polygamma(n, self);
    return out;
}

// special_logsumexp -> logsumexp
Tensor alias_logsumexp(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    TP_CHECK(dim.size() == 1,
             "special_logsumexp(): expected exactly one reduction dimension, got",
             dim.size());
    return detail::redispatch_logsumexp_function(self, dim[0], keepdim);
}

Tensor& alias_logsumexp_out(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, Tensor& out) {
    out = alias_logsumexp(self, dim, keepdim);
    return out;
}

// special_expit -> sigmoid
Tensor alias_expit(const Tensor& self) {
    return detail::redispatch_sigmoid_function(self);
}

Tensor& alias_expit_out(const Tensor& self, Tensor& out) {
    out = alias_expit(self);
    return out;
}

// special_sinc -> sinc
Tensor alias_sinc(const Tensor& self) {
    return detail::redispatch_sinc_function(self);
}

Tensor& alias_sinc_out(const Tensor& self, Tensor& out) {
    out = alias_sinc(self);
    return out;
}

// special_round -> round
Tensor alias_round(const Tensor& self, int64_t decimals) {
    if (decimals == 0) {
        return detail::redispatch_round_function(self);
    }
    TP_CHECK(isFloatingType(self.dtype()),
             "special_round(): decimals != 0 is only supported for floating-point tensors");
    const bool narrow = self.dtype() != DType::Float64;
    const Tensor work = narrow ? self.to(DType::Float32) : self;
    const bool inverse = decimals < 0;
    const Scalar factor(
        std::pow(10.0, static_cast<double>(inverse ? -decimals : decimals)));
    const Tensor scaled =
        inverse ? ops::div(work, factor) : ops::mul(work, factor);
    const Tensor rounded = ops::round(scaled);
    const Tensor result =
        inverse ? ops::mul(rounded, factor) : ops::div(rounded, factor);
    return narrow ? result.to(work.dtype()) : result;
}

Tensor& alias_round_out(const Tensor& self, int64_t decimals, Tensor& out) {
    out = alias_round(self, decimals);
    return out;
}

// special_log1p -> log1p
Tensor alias_log1p(const Tensor& self) {
    return detail::redispatch_log1p_function(self);
}

Tensor& alias_log1p_out(const Tensor& self, Tensor& out) {
    out = alias_log1p(self);
    return out;
}

// special_log_softmax -> log_softmax
Tensor alias_log_softmax(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    return detail::redispatch_log_softmax_function(self, dim, dtype.value_or(DType::Undefined));
}

// special_gammainc -> gammainc
Tensor& alias_gammainc_out(const Tensor& self, const Tensor& other, Tensor& out) {
    out = alias_gammainc(self, other);
    return out;
}

Tensor alias_gammainc(const Tensor& self, const Tensor& other) {
    return detail::redispatch_gammainc_function(self, other);
}

// special_gammaincc -> gammaincc
Tensor& alias_gammaincc_out(const Tensor& self, const Tensor& other, Tensor& out) {
    out = alias_gammaincc(self, other);
    return out;
}

Tensor alias_gammaincc(const Tensor& self, const Tensor& other) {
    return detail::redispatch_gammaincc_function(self, other);
}

// special_multigammaln -> mvlgamma
Tensor alias_multigammaln(const Tensor& self, int64_t p) {
    return detail::redispatch_mvlgamma_function(self, p);
}

Tensor& alias_multigammaln_out(const Tensor& self, int64_t p, Tensor& out) {
    out = alias_multigammaln(self, p);
    return out;
}

// special_softmax -> softmax
Tensor alias_softmax(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    return detail::redispatch_softmax_function(self, dim, dtype.value_or(DType::Undefined));
}

// special_airy_ai -> airy_ai
Tensor alias_airy_ai(const Tensor& x) {
    return detail::redispatch_airy_ai_function(x);
}

Tensor& alias_airy_ai_out(const Tensor& x, Tensor& out) {
    out = alias_airy_ai(x);
    return out;
}

// special_bessel_j0 -> bessel_j0
Tensor alias_bessel_j0(const Tensor& self) {
    return detail::redispatch_bessel_j0_function(self);
}

Tensor& alias_bessel_j0_out(const Tensor& self, Tensor& out) {
    out = alias_bessel_j0(self);
    return out;
}

// special_bessel_j1 -> bessel_j1
Tensor alias_bessel_j1(const Tensor& self) {
    return detail::redispatch_bessel_j1_function(self);
}

Tensor& alias_bessel_j1_out(const Tensor& self, Tensor& out) {
    out = alias_bessel_j1(self);
    return out;
}

// special_bessel_y0 -> bessel_y0
Tensor alias_bessel_y0(const Tensor& self) {
    return detail::redispatch_bessel_y0_function(self);
}

Tensor& alias_bessel_y0_out(const Tensor& self, Tensor& out) {
    out = alias_bessel_y0(self);
    return out;
}

// special_bessel_y1 -> bessel_y1
Tensor alias_bessel_y1(const Tensor& self) {
    return detail::redispatch_bessel_y1_function(self);
}

Tensor& alias_bessel_y1_out(const Tensor& self, Tensor& out) {
    out = alias_bessel_y1(self);
    return out;
}

// special_chebyshev_polynomial_t -> chebyshev_polynomial_t
Tensor alias_chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_t_function(x, n);
}

Tensor alias_chebyshev_polynomial_t_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_t_function(scalar_like(x, n), n);
}

Tensor alias_chebyshev_polynomial_t_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_chebyshev_polynomial_t_function(x, scalar_like(n, x));
}

Tensor& alias_chebyshev_polynomial_t_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_t(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_t_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_t_x_scalar(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_t_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_chebyshev_polynomial_t_n_scalar(x, n);
    return out;
}

// special_chebyshev_polynomial_u -> chebyshev_polynomial_u
Tensor alias_chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_u_function(x, n);
}

Tensor alias_chebyshev_polynomial_u_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_u_function(scalar_like(x, n), n);
}

Tensor alias_chebyshev_polynomial_u_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_chebyshev_polynomial_u_function(x, scalar_like(n, x));
}

Tensor& alias_chebyshev_polynomial_u_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_u(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_u_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_u_x_scalar(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_u_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_chebyshev_polynomial_u_n_scalar(x, n);
    return out;
}

// special_chebyshev_polynomial_v -> chebyshev_polynomial_v
Tensor alias_chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_v_function(x, n);
}

Tensor alias_chebyshev_polynomial_v_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_v_function(scalar_like(x, n), n);
}

Tensor alias_chebyshev_polynomial_v_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_chebyshev_polynomial_v_function(x, scalar_like(n, x));
}

Tensor& alias_chebyshev_polynomial_v_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_v(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_v_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_v_x_scalar(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_v_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_chebyshev_polynomial_v_n_scalar(x, n);
    return out;
}

// special_chebyshev_polynomial_w -> chebyshev_polynomial_w
Tensor alias_chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_w_function(x, n);
}

Tensor alias_chebyshev_polynomial_w_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_chebyshev_polynomial_w_function(scalar_like(x, n), n);
}

Tensor alias_chebyshev_polynomial_w_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_chebyshev_polynomial_w_function(x, scalar_like(n, x));
}

Tensor& alias_chebyshev_polynomial_w_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_w(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_w_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_chebyshev_polynomial_w_x_scalar(x, n);
    return out;
}

Tensor& alias_chebyshev_polynomial_w_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_chebyshev_polynomial_w_n_scalar(x, n);
    return out;
}

// special_hermite_polynomial_h -> hermite_polynomial_h
Tensor alias_hermite_polynomial_h(const Tensor& x, const Tensor& n) {
    return detail::redispatch_hermite_polynomial_h_function(x, n);
}

Tensor alias_hermite_polynomial_h_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_hermite_polynomial_h_function(scalar_like(x, n), n);
}

Tensor alias_hermite_polynomial_h_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_hermite_polynomial_h_function(x, scalar_like(n, x));
}

Tensor& alias_hermite_polynomial_h_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_hermite_polynomial_h(x, n);
    return out;
}

Tensor& alias_hermite_polynomial_h_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_hermite_polynomial_h_x_scalar(x, n);
    return out;
}

Tensor& alias_hermite_polynomial_h_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_hermite_polynomial_h_n_scalar(x, n);
    return out;
}

// special_hermite_polynomial_he -> hermite_polynomial_he
Tensor alias_hermite_polynomial_he(const Tensor& x, const Tensor& n) {
    return detail::redispatch_hermite_polynomial_he_function(x, n);
}

Tensor alias_hermite_polynomial_he_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_hermite_polynomial_he_function(scalar_like(x, n), n);
}

Tensor alias_hermite_polynomial_he_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_hermite_polynomial_he_function(x, scalar_like(n, x));
}

Tensor& alias_hermite_polynomial_he_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_hermite_polynomial_he(x, n);
    return out;
}

Tensor& alias_hermite_polynomial_he_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_hermite_polynomial_he_x_scalar(x, n);
    return out;
}

Tensor& alias_hermite_polynomial_he_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_hermite_polynomial_he_n_scalar(x, n);
    return out;
}

// special_laguerre_polynomial_l -> laguerre_polynomial_l
Tensor alias_laguerre_polynomial_l(const Tensor& x, const Tensor& n) {
    return detail::redispatch_laguerre_polynomial_l_function(x, n);
}

Tensor alias_laguerre_polynomial_l_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_laguerre_polynomial_l_function(scalar_like(x, n), n);
}

Tensor alias_laguerre_polynomial_l_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_laguerre_polynomial_l_function(x, scalar_like(n, x));
}

Tensor& alias_laguerre_polynomial_l_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_laguerre_polynomial_l(x, n);
    return out;
}

Tensor& alias_laguerre_polynomial_l_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_laguerre_polynomial_l_x_scalar(x, n);
    return out;
}

Tensor& alias_laguerre_polynomial_l_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_laguerre_polynomial_l_n_scalar(x, n);
    return out;
}

// special_legendre_polynomial_p -> legendre_polynomial_p
Tensor alias_legendre_polynomial_p(const Tensor& x, const Tensor& n) {
    return detail::redispatch_legendre_polynomial_p_function(x, n);
}

Tensor alias_legendre_polynomial_p_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_legendre_polynomial_p_function(scalar_like(x, n), n);
}

Tensor alias_legendre_polynomial_p_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_legendre_polynomial_p_function(x, scalar_like(n, x));
}

Tensor& alias_legendre_polynomial_p_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_legendre_polynomial_p(x, n);
    return out;
}

Tensor& alias_legendre_polynomial_p_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_legendre_polynomial_p_x_scalar(x, n);
    return out;
}

Tensor& alias_legendre_polynomial_p_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_legendre_polynomial_p_n_scalar(x, n);
    return out;
}

// special_modified_bessel_i0 -> modified_bessel_i0
Tensor alias_modified_bessel_i0(const Tensor& self) {
    return detail::redispatch_modified_bessel_i0_function(self);
}

Tensor& alias_modified_bessel_i0_out(const Tensor& self, Tensor& out) {
    out = alias_modified_bessel_i0(self);
    return out;
}

// special_modified_bessel_i1 -> modified_bessel_i1
Tensor alias_modified_bessel_i1(const Tensor& self) {
    return detail::redispatch_modified_bessel_i1_function(self);
}

Tensor& alias_modified_bessel_i1_out(const Tensor& self, Tensor& out) {
    out = alias_modified_bessel_i1(self);
    return out;
}

// special_modified_bessel_k0 -> modified_bessel_k0
Tensor alias_modified_bessel_k0(const Tensor& self) {
    return detail::redispatch_modified_bessel_k0_function(self);
}

Tensor& alias_modified_bessel_k0_out(const Tensor& self, Tensor& out) {
    out = alias_modified_bessel_k0(self);
    return out;
}

// special_modified_bessel_k1 -> modified_bessel_k1
Tensor alias_modified_bessel_k1(const Tensor& self) {
    return detail::redispatch_modified_bessel_k1_function(self);
}

Tensor& alias_modified_bessel_k1_out(const Tensor& self, Tensor& out) {
    out = alias_modified_bessel_k1(self);
    return out;
}

// special_scaled_modified_bessel_k0 -> scaled_modified_bessel_k0
Tensor alias_scaled_modified_bessel_k0(const Tensor& x) {
    return detail::redispatch_scaled_modified_bessel_k0_function(x);
}

Tensor& alias_scaled_modified_bessel_k0_out(const Tensor& x, Tensor& out) {
    out = alias_scaled_modified_bessel_k0(x);
    return out;
}

// special_scaled_modified_bessel_k1 -> scaled_modified_bessel_k1
Tensor alias_scaled_modified_bessel_k1(const Tensor& x) {
    return detail::redispatch_scaled_modified_bessel_k1_function(x);
}

Tensor& alias_scaled_modified_bessel_k1_out(const Tensor& x, Tensor& out) {
    out = alias_scaled_modified_bessel_k1(x);
    return out;
}

// special_shifted_chebyshev_polynomial_t -> shifted_chebyshev_polynomial_t
Tensor alias_shifted_chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_t_function(x, n);
}

Tensor alias_shifted_chebyshev_polynomial_t_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_t_function(scalar_like(x, n), n);
}

Tensor alias_shifted_chebyshev_polynomial_t_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_shifted_chebyshev_polynomial_t_function(x, scalar_like(n, x));
}

Tensor& alias_shifted_chebyshev_polynomial_t_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_t(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_t_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_t_x_scalar(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_t_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_t_n_scalar(x, n);
    return out;
}

// special_shifted_chebyshev_polynomial_u -> shifted_chebyshev_polynomial_u
Tensor alias_shifted_chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_u_function(x, n);
}

Tensor alias_shifted_chebyshev_polynomial_u_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_u_function(scalar_like(x, n), n);
}

Tensor alias_shifted_chebyshev_polynomial_u_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_shifted_chebyshev_polynomial_u_function(x, scalar_like(n, x));
}

Tensor& alias_shifted_chebyshev_polynomial_u_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_u(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_u_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_u_x_scalar(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_u_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_u_n_scalar(x, n);
    return out;
}

// special_shifted_chebyshev_polynomial_v -> shifted_chebyshev_polynomial_v
Tensor alias_shifted_chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_v_function(x, n);
}

Tensor alias_shifted_chebyshev_polynomial_v_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_v_function(scalar_like(x, n), n);
}

Tensor alias_shifted_chebyshev_polynomial_v_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_shifted_chebyshev_polynomial_v_function(x, scalar_like(n, x));
}

Tensor& alias_shifted_chebyshev_polynomial_v_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_v(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_v_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_v_x_scalar(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_v_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_v_n_scalar(x, n);
    return out;
}

// special_shifted_chebyshev_polynomial_w -> shifted_chebyshev_polynomial_w
Tensor alias_shifted_chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_w_function(x, n);
}

Tensor alias_shifted_chebyshev_polynomial_w_x_scalar(Scalar x, const Tensor& n) {
    return detail::redispatch_shifted_chebyshev_polynomial_w_function(scalar_like(x, n), n);
}

Tensor alias_shifted_chebyshev_polynomial_w_n_scalar(const Tensor& x, Scalar n) {
    return detail::redispatch_shifted_chebyshev_polynomial_w_function(x, scalar_like(n, x));
}

Tensor& alias_shifted_chebyshev_polynomial_w_out(const Tensor& x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_w(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_w_x_scalar_out(Scalar x, const Tensor& n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_w_x_scalar(x, n);
    return out;
}

Tensor& alias_shifted_chebyshev_polynomial_w_n_scalar_out(const Tensor& x, Scalar n, Tensor& out) {
    out = alias_shifted_chebyshev_polynomial_w_n_scalar(x, n);
    return out;
}

// special_spherical_bessel_j0 -> spherical_bessel_j0
Tensor alias_spherical_bessel_j0(const Tensor& x) {
    return detail::redispatch_spherical_bessel_j0_function(x);
}

Tensor& alias_spherical_bessel_j0_out(const Tensor& x, Tensor& out) {
    out = alias_spherical_bessel_j0(x);
    return out;
}

}  // namespace special_alias

TENSORPLAY_LIBRARY_IMPL(CPU, SpecialAliasOps) {
    using namespace special_alias;

    m.impl("special_entr", alias_entr);
    m.impl("special_entr.out", alias_entr_out);
    m.impl("special_ndtri", alias_ndtri);
    m.impl("special_ndtri.out", alias_ndtri_out);
    m.impl("special_log_ndtr", alias_log_ndtr);
    m.impl("special_log_ndtr.out", alias_log_ndtr_out);
    m.impl("special_expm1", cpu::expm1_kernel);  // direct: expm1_kernel
    m.impl("special_expm1.out", alias_expm1_out);
    m.impl("special_exp2", cpu::exp2_cpu);  // direct: exp2_cpu
    m.impl("special_exp2.out", alias_exp2_out);
    m.impl("special_psi", cpu::digamma_cpu);  // direct: digamma_cpu
    m.impl("special_psi.out", alias_psi_out);
    m.impl("special_digamma", cpu::digamma_cpu);  // direct: digamma_cpu
    m.impl("special_digamma.out", alias_digamma_out);
    m.impl("special_gammaln", cpu::lgamma_kernel);  // direct: lgamma_kernel
    m.impl("special_gammaln.out", alias_gammaln_out);
    m.impl("special_erf", cpu::erf_kernel);  // direct: erf_kernel
    m.impl("special_erf.out", alias_erf_out);
    m.impl("special_erfc", cpu::erfc_kernel);  // direct: erfc_kernel
    m.impl("special_erfc.out", alias_erfc_out);
    m.impl("special_erfcx", alias_erfcx);
    m.impl("special_erfcx.out", alias_erfcx_out);
    m.impl("special_erfinv", cpu::erfinv_cpu);  // direct: erfinv_cpu
    m.impl("special_erfinv.out", alias_erfinv_out);
    m.impl("special_ndtr", alias_ndtr);
    m.impl("special_ndtr.out", alias_ndtr_out);
    m.impl("special_xlog1py", alias_xlog1py);
    m.impl("special_xlog1py.self_scalar", alias_xlog1py_self_scalar);
    m.impl("special_xlog1py.other_scalar", alias_xlog1py_other_scalar);
    m.impl("special_xlog1py.out", alias_xlog1py_out);
    m.impl("special_xlog1py.self_scalar_out", alias_xlog1py_self_scalar_out);
    m.impl("special_xlog1py.other_scalar_out", alias_xlog1py_other_scalar_out);
    m.impl("special_xlogy", cpu::xlogy_cpu);  // direct: xlogy_cpu
    m.impl("special_xlogy.self_scalar", alias_xlogy_self_scalar);
    m.impl("special_xlogy.other_scalar", alias_xlogy_other_scalar);
    m.impl("special_xlogy.out", alias_xlogy_out);
    m.impl("special_xlogy.self_scalar_out", alias_xlogy_self_scalar_out);
    m.impl("special_xlogy.other_scalar_out", alias_xlogy_other_scalar_out);
    m.impl("special_zeta", alias_zeta);
    m.impl("special_zeta.self_scalar", alias_zeta_self_scalar);
    m.impl("special_zeta.other_scalar", alias_zeta_other_scalar);
    m.impl("special_zeta.out", alias_zeta_out);
    m.impl("special_zeta.self_scalar_out", alias_zeta_self_scalar_out);
    m.impl("special_zeta.other_scalar_out", alias_zeta_other_scalar_out);
    m.impl("special_i0", cpu::i0_cpu);  // direct: i0_cpu
    m.impl("special_i0.out", alias_i0_out);
    m.impl("special_i0e", alias_i0e);
    m.impl("special_i0e.out", alias_i0e_out);
    m.impl("special_i1", alias_i1);
    m.impl("special_i1.out", alias_i1_out);
    m.impl("special_i1e", alias_i1e);
    m.impl("special_i1e.out", alias_i1e_out);
    m.impl("special_logit", alias_logit);
    m.impl("special_logit.out", alias_logit_out);
    m.impl("special_polygamma", alias_polygamma);
    m.impl("special_polygamma.out", alias_polygamma_out);
    m.impl("special_logsumexp", alias_logsumexp);
    m.impl("special_logsumexp.out", alias_logsumexp_out);
    m.impl("special_expit", alias_expit);
    m.impl("special_expit.out", alias_expit_out);
    m.impl("special_sinc", cpu::sinc_cpu);  // direct: sinc_cpu
    m.impl("special_sinc.out", alias_sinc_out);
    m.impl("special_round", alias_round);
    m.impl("special_round.out", alias_round_out);
    m.impl("special_log1p", cpu::log1p_kernel);  // direct: log1p_kernel
    m.impl("special_log1p.out", alias_log1p_out);
    m.impl("special_log_softmax", alias_log_softmax);
    m.impl("special_gammainc.out", alias_gammainc_out);
    m.impl("special_gammainc", cpu::igamma_cpu);  // direct: igamma_cpu
    m.impl("special_gammaincc.out", alias_gammaincc_out);
    m.impl("special_gammaincc", cpu::igammac_cpu);  // direct: igammac_cpu
    m.impl("special_multigammaln", alias_multigammaln);
    m.impl("special_multigammaln.out", alias_multigammaln_out);
    m.impl("special_softmax", alias_softmax);
    m.impl("special_airy_ai", alias_airy_ai);
    m.impl("special_airy_ai.out", alias_airy_ai_out);
    m.impl("special_bessel_j0", alias_bessel_j0);
    m.impl("special_bessel_j0.out", alias_bessel_j0_out);
    m.impl("special_bessel_j1", alias_bessel_j1);
    m.impl("special_bessel_j1.out", alias_bessel_j1_out);
    m.impl("special_bessel_y0", alias_bessel_y0);
    m.impl("special_bessel_y0.out", alias_bessel_y0_out);
    m.impl("special_bessel_y1", alias_bessel_y1);
    m.impl("special_bessel_y1.out", alias_bessel_y1_out);
    m.impl("special_chebyshev_polynomial_t", alias_chebyshev_polynomial_t);
    m.impl("special_chebyshev_polynomial_t.x_scalar", alias_chebyshev_polynomial_t_x_scalar);
    m.impl("special_chebyshev_polynomial_t.n_scalar", alias_chebyshev_polynomial_t_n_scalar);
    m.impl("special_chebyshev_polynomial_t.out", alias_chebyshev_polynomial_t_out);
    m.impl("special_chebyshev_polynomial_t.x_scalar_out", alias_chebyshev_polynomial_t_x_scalar_out);
    m.impl("special_chebyshev_polynomial_t.n_scalar_out", alias_chebyshev_polynomial_t_n_scalar_out);
    m.impl("special_chebyshev_polynomial_u", alias_chebyshev_polynomial_u);
    m.impl("special_chebyshev_polynomial_u.x_scalar", alias_chebyshev_polynomial_u_x_scalar);
    m.impl("special_chebyshev_polynomial_u.n_scalar", alias_chebyshev_polynomial_u_n_scalar);
    m.impl("special_chebyshev_polynomial_u.out", alias_chebyshev_polynomial_u_out);
    m.impl("special_chebyshev_polynomial_u.x_scalar_out", alias_chebyshev_polynomial_u_x_scalar_out);
    m.impl("special_chebyshev_polynomial_u.n_scalar_out", alias_chebyshev_polynomial_u_n_scalar_out);
    m.impl("special_chebyshev_polynomial_v", alias_chebyshev_polynomial_v);
    m.impl("special_chebyshev_polynomial_v.x_scalar", alias_chebyshev_polynomial_v_x_scalar);
    m.impl("special_chebyshev_polynomial_v.n_scalar", alias_chebyshev_polynomial_v_n_scalar);
    m.impl("special_chebyshev_polynomial_v.out", alias_chebyshev_polynomial_v_out);
    m.impl("special_chebyshev_polynomial_v.x_scalar_out", alias_chebyshev_polynomial_v_x_scalar_out);
    m.impl("special_chebyshev_polynomial_v.n_scalar_out", alias_chebyshev_polynomial_v_n_scalar_out);
    m.impl("special_chebyshev_polynomial_w", alias_chebyshev_polynomial_w);
    m.impl("special_chebyshev_polynomial_w.x_scalar", alias_chebyshev_polynomial_w_x_scalar);
    m.impl("special_chebyshev_polynomial_w.n_scalar", alias_chebyshev_polynomial_w_n_scalar);
    m.impl("special_chebyshev_polynomial_w.out", alias_chebyshev_polynomial_w_out);
    m.impl("special_chebyshev_polynomial_w.x_scalar_out", alias_chebyshev_polynomial_w_x_scalar_out);
    m.impl("special_chebyshev_polynomial_w.n_scalar_out", alias_chebyshev_polynomial_w_n_scalar_out);
    m.impl("special_hermite_polynomial_h", alias_hermite_polynomial_h);
    m.impl("special_hermite_polynomial_h.x_scalar", alias_hermite_polynomial_h_x_scalar);
    m.impl("special_hermite_polynomial_h.n_scalar", alias_hermite_polynomial_h_n_scalar);
    m.impl("special_hermite_polynomial_h.out", alias_hermite_polynomial_h_out);
    m.impl("special_hermite_polynomial_h.x_scalar_out", alias_hermite_polynomial_h_x_scalar_out);
    m.impl("special_hermite_polynomial_h.n_scalar_out", alias_hermite_polynomial_h_n_scalar_out);
    m.impl("special_hermite_polynomial_he", alias_hermite_polynomial_he);
    m.impl("special_hermite_polynomial_he.x_scalar", alias_hermite_polynomial_he_x_scalar);
    m.impl("special_hermite_polynomial_he.n_scalar", alias_hermite_polynomial_he_n_scalar);
    m.impl("special_hermite_polynomial_he.out", alias_hermite_polynomial_he_out);
    m.impl("special_hermite_polynomial_he.x_scalar_out", alias_hermite_polynomial_he_x_scalar_out);
    m.impl("special_hermite_polynomial_he.n_scalar_out", alias_hermite_polynomial_he_n_scalar_out);
    m.impl("special_laguerre_polynomial_l", alias_laguerre_polynomial_l);
    m.impl("special_laguerre_polynomial_l.x_scalar", alias_laguerre_polynomial_l_x_scalar);
    m.impl("special_laguerre_polynomial_l.n_scalar", alias_laguerre_polynomial_l_n_scalar);
    m.impl("special_laguerre_polynomial_l.out", alias_laguerre_polynomial_l_out);
    m.impl("special_laguerre_polynomial_l.x_scalar_out", alias_laguerre_polynomial_l_x_scalar_out);
    m.impl("special_laguerre_polynomial_l.n_scalar_out", alias_laguerre_polynomial_l_n_scalar_out);
    m.impl("special_legendre_polynomial_p", alias_legendre_polynomial_p);
    m.impl("special_legendre_polynomial_p.x_scalar", alias_legendre_polynomial_p_x_scalar);
    m.impl("special_legendre_polynomial_p.n_scalar", alias_legendre_polynomial_p_n_scalar);
    m.impl("special_legendre_polynomial_p.out", alias_legendre_polynomial_p_out);
    m.impl("special_legendre_polynomial_p.x_scalar_out", alias_legendre_polynomial_p_x_scalar_out);
    m.impl("special_legendre_polynomial_p.n_scalar_out", alias_legendre_polynomial_p_n_scalar_out);
    m.impl("special_modified_bessel_i0", alias_modified_bessel_i0);
    m.impl("special_modified_bessel_i0.out", alias_modified_bessel_i0_out);
    m.impl("special_modified_bessel_i1", alias_modified_bessel_i1);
    m.impl("special_modified_bessel_i1.out", alias_modified_bessel_i1_out);
    m.impl("special_modified_bessel_k0", alias_modified_bessel_k0);
    m.impl("special_modified_bessel_k0.out", alias_modified_bessel_k0_out);
    m.impl("special_modified_bessel_k1", alias_modified_bessel_k1);
    m.impl("special_modified_bessel_k1.out", alias_modified_bessel_k1_out);
    m.impl("special_scaled_modified_bessel_k0", alias_scaled_modified_bessel_k0);
    m.impl("special_scaled_modified_bessel_k0.out", alias_scaled_modified_bessel_k0_out);
    m.impl("special_scaled_modified_bessel_k1", alias_scaled_modified_bessel_k1);
    m.impl("special_scaled_modified_bessel_k1.out", alias_scaled_modified_bessel_k1_out);
    m.impl("special_shifted_chebyshev_polynomial_t", alias_shifted_chebyshev_polynomial_t);
    m.impl("special_shifted_chebyshev_polynomial_t.x_scalar", alias_shifted_chebyshev_polynomial_t_x_scalar);
    m.impl("special_shifted_chebyshev_polynomial_t.n_scalar", alias_shifted_chebyshev_polynomial_t_n_scalar);
    m.impl("special_shifted_chebyshev_polynomial_t.out", alias_shifted_chebyshev_polynomial_t_out);
    m.impl("special_shifted_chebyshev_polynomial_t.x_scalar_out", alias_shifted_chebyshev_polynomial_t_x_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_t.n_scalar_out", alias_shifted_chebyshev_polynomial_t_n_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_u", alias_shifted_chebyshev_polynomial_u);
    m.impl("special_shifted_chebyshev_polynomial_u.x_scalar", alias_shifted_chebyshev_polynomial_u_x_scalar);
    m.impl("special_shifted_chebyshev_polynomial_u.n_scalar", alias_shifted_chebyshev_polynomial_u_n_scalar);
    m.impl("special_shifted_chebyshev_polynomial_u.out", alias_shifted_chebyshev_polynomial_u_out);
    m.impl("special_shifted_chebyshev_polynomial_u.x_scalar_out", alias_shifted_chebyshev_polynomial_u_x_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_u.n_scalar_out", alias_shifted_chebyshev_polynomial_u_n_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_v", alias_shifted_chebyshev_polynomial_v);
    m.impl("special_shifted_chebyshev_polynomial_v.x_scalar", alias_shifted_chebyshev_polynomial_v_x_scalar);
    m.impl("special_shifted_chebyshev_polynomial_v.n_scalar", alias_shifted_chebyshev_polynomial_v_n_scalar);
    m.impl("special_shifted_chebyshev_polynomial_v.out", alias_shifted_chebyshev_polynomial_v_out);
    m.impl("special_shifted_chebyshev_polynomial_v.x_scalar_out", alias_shifted_chebyshev_polynomial_v_x_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_v.n_scalar_out", alias_shifted_chebyshev_polynomial_v_n_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_w", alias_shifted_chebyshev_polynomial_w);
    m.impl("special_shifted_chebyshev_polynomial_w.x_scalar", alias_shifted_chebyshev_polynomial_w_x_scalar);
    m.impl("special_shifted_chebyshev_polynomial_w.n_scalar", alias_shifted_chebyshev_polynomial_w_n_scalar);
    m.impl("special_shifted_chebyshev_polynomial_w.out", alias_shifted_chebyshev_polynomial_w_out);
    m.impl("special_shifted_chebyshev_polynomial_w.x_scalar_out", alias_shifted_chebyshev_polynomial_w_x_scalar_out);
    m.impl("special_shifted_chebyshev_polynomial_w.n_scalar_out", alias_shifted_chebyshev_polynomial_w_n_scalar_out);
    m.impl("special_spherical_bessel_j0", alias_spherical_bessel_j0);
    m.impl("special_spherical_bessel_j0.out", alias_spherical_bessel_j0_out);
}

}  // namespace tensorplay
