// Generated batch completion of the in-place op family ("In-place family
// completion" section of the native schema).  Every wrapper
// re-dispatches the functional op on `self` and lands the result back via
// copy_ -- the established abs_/lerp_ pattern -- under a local grad-disabled
// guard so the composite does not record autograd nodes.
//
// The CPU and CUDA copies of this file are textually identical apart from
// namespace and library key; no device-specific kernel code lives here.
#include <optional>
#include <string>

#include "Dispatcher.h"
#include "GradMode.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace cuda {

namespace {

// RAII over thread-local GradMode for mutation-free sections.
struct NoGradGuard {
    bool prev;
    NoGradGuard() : prev(GradMode::is_enabled()) { GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev); }
};

Tensor& acos_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_acos_out_function(self, self);
    return self;
}

Tensor& acosh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_acosh_out_function(self, self);
    return self;
}

Tensor& addbmm_inplace_kernel(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addbmm(self, batch1, batch2, beta, alpha));
    return self;
}

Tensor& addmm_inplace_kernel(Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addmm(self, mat1, mat2, beta, alpha));
    return self;
}

Tensor& addmv_inplace_kernel(Tensor& self, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addmv(self, mat, vec, beta, alpha));
    return self;
}

Tensor& addr_inplace_kernel(Tensor& self, const Tensor& vec1, const Tensor& vec2, const Scalar& beta, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addr(self, vec1, vec2, beta, alpha));
    return self;
}

Tensor& asin_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_asin_out_function(self, self);
    return self;
}

Tensor& asinh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_asinh_out_function(self, self);
    return self;
}

Tensor& atan2_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_atan2_out_function(self, other, self);
    return self;
}

Tensor& atan_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_atan_out_function(self, self);
    return self;
}

Tensor& atanh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_atanh_out_function(self, self);
    return self;
}

Tensor& bitwise_not_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_bitwise_not_out_function(self, self);
    return self;
}

Tensor& baddbmm_inplace_kernel(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::baddbmm(self, batch1, batch2, beta, alpha));
    return self;
}

Tensor& ceil_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_ceil_out_function(self, self);
    return self;
}

Tensor& celu_inplace_kernel(Tensor& self, const Scalar& alpha) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_celu_out_function(self, alpha, self);
    return self;
}

Tensor& clip_inplace_kernel(Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_clamp_out_function(self, min, max, self);
    return self;
}

Tensor& cos_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_cos_out_function(self, self);
    return self;
}

Tensor& cosh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_cosh_out_function(self, self);
    return self;
}

Tensor& cumprod_inplace_kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_cumprod_out_function(self, dim, dtype, self);
    return self;
}

Tensor& cumsum_inplace_kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_cumsum_out_function(self, dim, dtype, self);
    return self;
}

Tensor& deg2rad_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_deg2rad_out_function(self, self);
    return self;
}

Tensor& elu_inplace_kernel(Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_elu_out_function(self, alpha, scale, input_scale, self);
    return self;
}

Tensor& erf_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_erf_out_function(self, self);
    return self;
}

Tensor& erfc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_erfc_out_function(self, self);
    return self;
}

Tensor& erfinv_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_erfinv_out_function(self, self);
    return self;
}

Tensor& exp2_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_exp2_out_function(self, self);
    return self;
}

Tensor& exp_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_exp_out_function(self, self);
    return self;
}

Tensor& expm1_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_expm1_out_function(self, self);
    return self;
}

Tensor& floor_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_floor_out_function(self, self);
    return self;
}

Tensor& frac_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_frac_out_function(self, self);
    return self;
}

Tensor& gcd_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_gcd_out_function(self, other, self);
    return self;
}

Tensor& gelu_inplace_kernel(Tensor& self, const std::string& approximate) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_gelu_out_function(self, approximate, self);
    return self;
}

Tensor& hardsigmoid_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_hardsigmoid_out_function(self, self);
    return self;
}

Tensor& hardswish_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_hardswish_out_function(self, self);
    return self;
}

Tensor& hardtanh_inplace_kernel(Tensor& self, const Scalar& min_val, const Scalar& max_val) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_hardtanh_out_function(self, min_val, max_val, self);
    return self;
}

Tensor& heaviside_inplace_kernel(Tensor& self, const Tensor& values) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_heaviside_out_function(self, values, self);
    return self;
}

Tensor& hypot_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_hypot_out_function(self, other, self);
    return self;
}

Tensor& xlogy_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_xlogy_OutTensor_function(self, other, self);
    return self;
}

Tensor& xlog1py_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_xlog1py_OutTensor_function(self, other, self);
    return self;
}

Tensor& index_reduce_inplace_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const std::string& reduce, bool include_self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_index_reduce_out_function(self, dim, index, source, reduce, include_self, self);
    return self;
}

Tensor& lcm_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_lcm_out_function(self, other, self);
    return self;
}

Tensor& leaky_relu_inplace_kernel(Tensor& self, const Scalar& negative_slope) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_leaky_relu_out_function(self, negative_slope, self);
    return self;
}

Tensor& lgamma_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_lgamma_out_function(self, self);
    return self;
}

Tensor& log10_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_log10_out_function(self, self);
    return self;
}

Tensor& log1p_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_log1p_out_function(self, self);
    return self;
}

Tensor& log2_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_log2_out_function(self, self);
    return self;
}

Tensor& log_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_log_out_function(self, self);
    return self;
}

Tensor& logical_and_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::logical_and(self, other));
    return self;
}

Tensor& logical_not_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_logical_not_out_function(self, self);
    return self;
}

Tensor& logical_or_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::logical_or(self, other));
    return self;
}

Tensor& logical_xor_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::logical_xor(self, other));
    return self;
}

Tensor& logit_inplace_kernel(Tensor& self, const std::optional<Scalar>& eps) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_logit_out_function(
        self, eps.has_value() ? std::optional<double>(eps->toDouble()) : std::nullopt, self);
    return self;
}

Tensor& masked_scatter_inplace_kernel(Tensor& self, const Tensor& mask, const Tensor& source) {
    NoGradGuard __tp_nograd;
    self.copy_(self.masked_scatter(mask, source));
    return self;
}

Tensor& mish_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_mish_out_function(self, self);
    return self;
}

Tensor& nan_to_num_inplace_kernel(Tensor& self, const Scalar& nan, const std::optional<Scalar>& posinf, const std::optional<Scalar>& neginf) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_nan_to_num_out_function(
        self, nan.toDouble(),
        posinf.has_value() ? std::optional<double>(posinf->toDouble()) : std::nullopt,
        neginf.has_value() ? std::optional<double>(neginf->toDouble()) : std::nullopt,
        self);
    return self;
}

Tensor& nextafter_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_nextafter_out_function(self, other, self);
    return self;
}

Tensor& rad2deg_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_rad2deg_out_function(self, self);
    return self;
}

Tensor& reciprocal_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_reciprocal_out_function(self, self);
    return self;
}

Tensor& relu6_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_relu6_out_function(self, self);
    return self;
}

Tensor& renorm_inplace_kernel(Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_renorm_out_function(self, p, dim, maxnorm, self);
    return self;
}

Tensor& round_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_round_out_function(self, self);
    return self;
}

Tensor& selu_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_selu_out_function(self, self);
    return self;
}

Tensor& sgn_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sgn_out_function(self, self);
    return self;
}

Tensor& sigmoid_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sigmoid_out_function(self, self);
    return self;
}

Tensor& sign_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sign_out_function(self, self);
    return self;
}

Tensor& silu_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_silu_out_function(self, self);
    return self;
}

Tensor& sin_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sin_out_function(self, self);
    return self;
}

Tensor& sinc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sinc_out_function(self, self);
    return self;
}

Tensor& sinh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_sinh_out_function(self, self);
    return self;
}

Tensor& square_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_square_out_function(self, self);
    return self;
}

Tensor& tan_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_tan_out_function(self, self);
    return self;
}

Tensor& tanh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_tanh_out_function(self, self);
    return self;
}

Tensor& tril_inplace_kernel(Tensor& self, int64_t diagonal) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_tril_out_function(self, diagonal, self);
    return self;
}

Tensor& triu_inplace_kernel(Tensor& self, int64_t diagonal) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_triu_out_function(self, diagonal, self);
    return self;
}

Tensor& trunc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_trunc_out_function(self, self);
    return self;
}

Tensor& threshold_inplace_kernel(Tensor& self, const Scalar& threshold, const Scalar& value) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::threshold(self, threshold, value));
    return self;
}
Tensor& i0_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::i0(self));
    return self;
}

Tensor& pow_scalar_inplace_kernel(Tensor& self, const Scalar& exponent) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_pow_Tensor_Scalar_out_function(self, exponent, self);
    return self;
}

Tensor& pow_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_pow_Tensor_Tensor_out_function(self, exponent, self);
    return self;
}

Tensor& float_power_scalar_inplace_kernel(Tensor& self, const Scalar& exponent) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_float_power_Tensor_Scalar_out_function(self, exponent, self);
    return self;
}

Tensor& float_power_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
    NoGradGuard __tp_nograd;
    tensorplay::detail::redispatch_float_power_Tensor_Tensor_out_function(self, exponent, self);
    return self;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, InplaceCompletionKernels) {
    m.impl("acos_", acos_inplace_kernel);
    m.impl("acosh_", acosh_inplace_kernel);
    m.impl("addbmm_", addbmm_inplace_kernel);
    m.impl("addmm_", addmm_inplace_kernel);
    m.impl("addmv_", addmv_inplace_kernel);
    m.impl("addr_", addr_inplace_kernel);
    m.impl("asin_", asin_inplace_kernel);
    m.impl("asinh_", asinh_inplace_kernel);
    m.impl("atan2_", atan2_inplace_kernel);
    m.impl("arctan2_", atan2_inplace_kernel);
    m.impl("atan_", atan_inplace_kernel);
    m.impl("atanh_", atanh_inplace_kernel);
    m.impl("bitwise_not_", bitwise_not_inplace_kernel);
    m.impl("baddbmm_", baddbmm_inplace_kernel);
    m.impl("ceil_", ceil_inplace_kernel);
    m.impl("celu_", celu_inplace_kernel);
    m.impl("clip_", clip_inplace_kernel);
    m.impl("cos_", cos_inplace_kernel);
    m.impl("cosh_", cosh_inplace_kernel);
    m.impl("cumprod_", cumprod_inplace_kernel);
    m.impl("cumsum_", cumsum_inplace_kernel);
    m.impl("deg2rad_", deg2rad_inplace_kernel);
    m.impl("elu_", elu_inplace_kernel);
    m.impl("erf_", erf_inplace_kernel);
    m.impl("erfc_", erfc_inplace_kernel);
    m.impl("erfinv_", erfinv_inplace_kernel);
    m.impl("exp2_", exp2_inplace_kernel);
    m.impl("exp_", exp_inplace_kernel);
    m.impl("expm1_", expm1_inplace_kernel);
    m.impl("floor_", floor_inplace_kernel);
    m.impl("frac_", frac_inplace_kernel);
    m.impl("gcd_", gcd_inplace_kernel);
    m.impl("gelu_", gelu_inplace_kernel);
    m.impl("hardsigmoid_", hardsigmoid_inplace_kernel);
    m.impl("hardswish_", hardswish_inplace_kernel);
    m.impl("hardtanh_", hardtanh_inplace_kernel);
    m.impl("heaviside_", heaviside_inplace_kernel);
    m.impl("hypot_", hypot_inplace_kernel);
    m.impl("index_reduce_", index_reduce_inplace_kernel);
    m.impl("lcm_", lcm_inplace_kernel);
    m.impl("leaky_relu_", leaky_relu_inplace_kernel);
    m.impl("lgamma_", lgamma_inplace_kernel);
    m.impl("log10_", log10_inplace_kernel);
    m.impl("log1p_", log1p_inplace_kernel);
    m.impl("log2_", log2_inplace_kernel);
    m.impl("log_", log_inplace_kernel);
    m.impl("logical_and_", logical_and_inplace_kernel);
    m.impl("logical_not_", logical_not_inplace_kernel);
    m.impl("logical_or_", logical_or_inplace_kernel);
    m.impl("logical_xor_", logical_xor_inplace_kernel);
    m.impl("logit_", logit_inplace_kernel);
    m.impl("masked_scatter_", masked_scatter_inplace_kernel);
    m.impl("mish_", mish_inplace_kernel);
    m.impl("nan_to_num_", nan_to_num_inplace_kernel);
    m.impl("nextafter_", nextafter_inplace_kernel);
    m.impl("rad2deg_", rad2deg_inplace_kernel);
    m.impl("reciprocal_", reciprocal_inplace_kernel);
    m.impl("relu6_", relu6_inplace_kernel);
    m.impl("renorm_", renorm_inplace_kernel);
    m.impl("round_", round_inplace_kernel);
    m.impl("selu_", selu_inplace_kernel);
    m.impl("sgn_", sgn_inplace_kernel);
    m.impl("sigmoid_", sigmoid_inplace_kernel);
    m.impl("sign_", sign_inplace_kernel);
    m.impl("silu_", silu_inplace_kernel);
    m.impl("sin_", sin_inplace_kernel);
    m.impl("sinc_", sinc_inplace_kernel);
    m.impl("sinh_", sinh_inplace_kernel);
    m.impl("square_", square_inplace_kernel);
    m.impl("tan_", tan_inplace_kernel);
    m.impl("tanh_", tanh_inplace_kernel);
    m.impl("tril_", tril_inplace_kernel);
    m.impl("triu_", triu_inplace_kernel);
    m.impl("trunc_", trunc_inplace_kernel);
    m.impl("threshold_", threshold_inplace_kernel);
    m.impl("i0_", i0_inplace_kernel);
    m.impl("pow_.Scalar", pow_scalar_inplace_kernel);
    m.impl("pow_.Tensor", pow_tensor_inplace_kernel);
    m.impl("float_power_.Scalar", float_power_scalar_inplace_kernel);
    m.impl("float_power_.Tensor", float_power_tensor_inplace_kernel);
    m.impl("xlogy_", xlogy_inplace_kernel);
    m.impl("xlogy_.Tensor", xlogy_inplace_kernel);
    m.impl("xlog1py_", xlog1py_inplace_kernel);
}

} // namespace cuda
} // namespace tensorplay
