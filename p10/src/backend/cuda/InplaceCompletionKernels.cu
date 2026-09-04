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
    self.copy_(self.acos());
    return self;
}

Tensor& acosh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.acosh());
    return self;
}

Tensor& addbmm_inplace_kernel(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addbmm(self, batch1, batch2, beta, alpha));
    return self;
}

Tensor& addmm_inplace_kernel(Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addmm(self, mat1, mat2, beta, alpha));
    return self;
}

Tensor& addmv_inplace_kernel(Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addmv(self, mat, vec, beta, alpha));
    return self;
}

Tensor& addr_inplace_kernel(Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::addr(self, vec1, vec2, beta, alpha));
    return self;
}

Tensor& asin_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.asin());
    return self;
}

Tensor& asinh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.asinh());
    return self;
}

Tensor& atan2_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(self.atan2(other));
    return self;
}

Tensor& atan_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.atan());
    return self;
}

Tensor& atanh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.atanh());
    return self;
}

Tensor& bitwise_not_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.bitwise_not());
    return self;
}

Tensor& baddbmm_inplace_kernel(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::baddbmm(self, batch1, batch2, beta, alpha));
    return self;
}

Tensor& ceil_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.ceil());
    return self;
}

Tensor& celu_inplace_kernel(Tensor& self, Scalar alpha) {
    NoGradGuard __tp_nograd;
    self.copy_(self.celu(alpha));
    return self;
}

Tensor& clip_inplace_kernel(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    NoGradGuard __tp_nograd;
    self.copy_(self.clamp(min, max));
    return self;
}

Tensor& cos_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.cos());
    return self;
}

Tensor& cosh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.cosh());
    return self;
}

Tensor& cumprod_inplace_kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    NoGradGuard __tp_nograd;
    self.copy_(self.cumprod(dim, dtype));
    return self;
}

Tensor& cumsum_inplace_kernel(Tensor& self, int64_t dim, std::optional<DType> dtype) {
    NoGradGuard __tp_nograd;
    self.copy_(self.cumsum(dim, dtype));
    return self;
}

Tensor& deg2rad_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.deg2rad());
    return self;
}

Tensor& elu_inplace_kernel(Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
    NoGradGuard __tp_nograd;
    self.copy_(self.elu(alpha, scale, input_scale));
    return self;
}

Tensor& erf_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.erf());
    return self;
}

Tensor& erfc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.erfc());
    return self;
}

Tensor& erfinv_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.erfinv());
    return self;
}

Tensor& exp2_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.exp2());
    return self;
}

Tensor& exp_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.exp());
    return self;
}

Tensor& expm1_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.expm1());
    return self;
}

Tensor& floor_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.floor());
    return self;
}

Tensor& frac_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.frac());
    return self;
}

Tensor& gcd_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(self.gcd(other));
    return self;
}

Tensor& gelu_inplace_kernel(Tensor& self, std::string approximate) {
    NoGradGuard __tp_nograd;
    self.copy_(self.gelu(approximate));
    return self;
}

Tensor& hardsigmoid_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.hardsigmoid());
    return self;
}

Tensor& hardswish_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.hardswish());
    return self;
}

Tensor& hardtanh_inplace_kernel(Tensor& self, Scalar min_val, Scalar max_val) {
    NoGradGuard __tp_nograd;
    self.copy_(self.hardtanh(min_val, max_val));
    return self;
}

Tensor& heaviside_inplace_kernel(Tensor& self, const Tensor& values) {
    NoGradGuard __tp_nograd;
    self.copy_(self.heaviside(values));
    return self;
}

Tensor& hypot_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(self.hypot(other));
    return self;
}

Tensor& index_add_inplace_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    NoGradGuard __tp_nograd;
    self.copy_(self.index_add(dim, index, source));
    return self;
}

Tensor& index_copy_inplace_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    NoGradGuard __tp_nograd;
    self.copy_(self.index_copy(dim, index, source));
    return self;
}

Tensor& index_reduce_inplace_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, std::string reduce, bool include_self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.index_reduce(dim, index, source, reduce, include_self));
    return self;
}

Tensor& lcm_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(self.lcm(other));
    return self;
}

Tensor& leaky_relu_inplace_kernel(Tensor& self, Scalar negative_slope) {
    NoGradGuard __tp_nograd;
    self.copy_(self.leaky_relu(negative_slope));
    return self;
}

Tensor& lgamma_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.lgamma());
    return self;
}

Tensor& log10_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.log10());
    return self;
}

Tensor& log1p_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.log1p());
    return self;
}

Tensor& log2_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.log2());
    return self;
}

Tensor& log_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.log());
    return self;
}

Tensor& logical_and_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::logical_and(self, other));
    return self;
}

Tensor& logical_not_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.logical_not());
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

Tensor& logit_inplace_kernel(Tensor& self, std::optional<Scalar> eps) {
    NoGradGuard __tp_nograd;
    self.copy_(self.logit(eps));
    return self;
}

Tensor& masked_scatter_inplace_kernel(Tensor& self, const Tensor& mask, const Tensor& source) {
    NoGradGuard __tp_nograd;
    self.copy_(self.masked_scatter(mask, source));
    return self;
}

Tensor& mish_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.mish());
    return self;
}

Tensor& nan_to_num_inplace_kernel(Tensor& self, Scalar nan, std::optional<Scalar> posinf, std::optional<Scalar> neginf) {
    NoGradGuard __tp_nograd;
    self.copy_(self.nan_to_num(nan, posinf, neginf));
    return self;
}

Tensor& nextafter_inplace_kernel(Tensor& self, const Tensor& other) {
    NoGradGuard __tp_nograd;
    self.copy_(self.nextafter(other));
    return self;
}

Tensor& rad2deg_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.rad2deg());
    return self;
}

Tensor& reciprocal_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.reciprocal());
    return self;
}

Tensor& relu6_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.relu6());
    return self;
}

Tensor& renorm_inplace_kernel(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
    NoGradGuard __tp_nograd;
    self.copy_(self.renorm(p, dim, maxnorm));
    return self;
}

Tensor& round_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.round());
    return self;
}

Tensor& selu_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.selu());
    return self;
}

Tensor& sgn_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sgn());
    return self;
}

Tensor& sigmoid_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sigmoid());
    return self;
}

Tensor& sign_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sign());
    return self;
}

Tensor& silu_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.silu());
    return self;
}

Tensor& sin_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sin());
    return self;
}

Tensor& sinc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sinc());
    return self;
}

Tensor& sinh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.sinh());
    return self;
}

Tensor& square_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.square());
    return self;
}

Tensor& tan_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.tan());
    return self;
}

Tensor& tanh_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.tanh());
    return self;
}

Tensor& tril_inplace_kernel(Tensor& self, int64_t diagonal) {
    NoGradGuard __tp_nograd;
    self.copy_(self.tril(diagonal));
    return self;
}

Tensor& triu_inplace_kernel(Tensor& self, int64_t diagonal) {
    NoGradGuard __tp_nograd;
    self.copy_(self.triu(diagonal));
    return self;
}

Tensor& trunc_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(self.trunc());
    return self;
}

Tensor& threshold_inplace_kernel(Tensor& self, Scalar threshold, Scalar value) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::threshold(self, threshold, value));
    return self;
}
Tensor& i0_inplace_kernel(Tensor& self) {
    NoGradGuard __tp_nograd;
    self.copy_(Tensor::i0(self));
    return self;
}

Tensor& pow_scalar_inplace_kernel(Tensor& self, Scalar exponent) {
    NoGradGuard __tp_nograd;
    self.copy_(self.pow(exponent));
    return self;
}

Tensor& pow_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
    NoGradGuard __tp_nograd;
    self.copy_(self.pow(exponent));
    return self;
}

Tensor& float_power_scalar_inplace_kernel(Tensor& self, Scalar exponent) {
    NoGradGuard __tp_nograd;
    self.copy_(self.float_power(exponent));
    return self;
}

Tensor& float_power_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
    NoGradGuard __tp_nograd;
    self.copy_(self.float_power(exponent));
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
    m.impl("index_add_", index_add_inplace_kernel);
    m.impl("index_copy_", index_copy_inplace_kernel);
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
}

} // namespace cuda
} // namespace tensorplay
