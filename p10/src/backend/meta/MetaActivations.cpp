// Shape-only kernels for pointwise activations and elementary math on the
// meta device.
//
// Every op here maps elements to elements (or, for `glu`, halves one axis), so
// the result repeats the operand's geometry and -- with the few exceptions
// noted per op -- its dtype.  Nothing reads or writes element data: the result
// is an unbacked tensor carrying only sizes, strides, dtype and device.
// In-place variants return the operand unchanged, because writes would only
// touch element data that a meta tensor does not own.

#include "MetaCommon.h"

#include "Scalar.h"
#include "TypePromotion.h"
#include "TypeProperties.h"

#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// A unary elementwise op keeps the input's shape and dtype.
Tensor unary_meta(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Activations.
// ---------------------------------------------------------------------------

Tensor relu_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& relu_inplace_kernel(Tensor& self) {
    return self;
}

Tensor gelu_kernel(const Tensor& self, const std::string& /*approximate*/) {
    return unary_meta(self);
}

Tensor& gelu_inplace_kernel(Tensor& self, const std::string& /*approximate*/) {
    return self;
}

Tensor tanh_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& tanh_inplace_kernel(Tensor& self) {
    return self;
}

Tensor sigmoid_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& sigmoid_inplace_kernel(Tensor& self) {
    return self;
}

Tensor silu_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& silu_inplace_kernel(Tensor& self) {
    return self;
}

Tensor mish_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& mish_inplace_kernel(Tensor& self) {
    return self;
}

Tensor softplus_kernel(const Tensor& self, const Scalar& /*beta*/,
                       const Scalar& /*threshold*/) {
    return unary_meta(self);
}

Tensor& softplus_inplace_kernel(Tensor& self, const Scalar& /*beta*/,
                                const Scalar& /*threshold*/) {
    return self;
}

Tensor elu_kernel(const Tensor& self, const Scalar& /*alpha*/,
                  const Scalar& /*scale*/, const Scalar& /*input_scale*/) {
    return unary_meta(self);
}

Tensor& elu_inplace_kernel(Tensor& self, const Scalar& /*alpha*/,
                           const Scalar& /*scale*/,
                           const Scalar& /*input_scale*/) {
    return self;
}

Tensor selu_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& selu_inplace_kernel(Tensor& self) {
    return self;
}

Tensor celu_kernel(const Tensor& self, const Scalar& /*alpha*/) {
    return unary_meta(self);
}

Tensor& celu_inplace_kernel(Tensor& self, const Scalar& /*alpha*/) {
    return self;
}

Tensor leaky_relu_kernel(const Tensor& self, const Scalar& /*negative_slope*/) {
    return unary_meta(self);
}

Tensor& leaky_relu_inplace_kernel(Tensor& self, const Scalar& /*negative_slope*/) {
    return self;
}

Tensor hardsigmoid_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& hardsigmoid_inplace_kernel(Tensor& self) {
    return self;
}

Tensor hardswish_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& hardswish_inplace_kernel(Tensor& self) {
    return self;
}

Tensor hardtanh_kernel(const Tensor& self, const Scalar& /*min_val*/,
                       const Scalar& /*max_val*/) {
    return unary_meta(self);
}

Tensor& hardtanh_inplace_kernel(Tensor& self, const Scalar& /*min_val*/,
                                const Scalar& /*max_val*/) {
    return self;
}

Tensor tanhshrink_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor softshrink_kernel(const Tensor& self, const Scalar& /*lambd*/) {
    return unary_meta(self);
}

Tensor hardshrink_kernel(const Tensor& self, const Scalar& /*lambd*/) {
    return unary_meta(self);
}

Tensor relu6_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& relu6_inplace_kernel(Tensor& self) {
    return self;
}

Tensor log_sigmoid_kernel(const Tensor& self) {
    return unary_meta(self);
}

// ---------------------------------------------------------------------------
// Rounding and sign.
// ---------------------------------------------------------------------------

Tensor floor_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "floor is not supported for complex inputs");
    }
    return unary_meta(self);
}

Tensor ceil_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "ceil is not supported for complex inputs");
    }
    return unary_meta(self);
}

Tensor round_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor trunc_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "trunc is not supported for complex inputs");
    }
    return unary_meta(self);
}

Tensor& trunc_inplace_kernel(Tensor& self) {
    return self;
}

Tensor frac_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& frac_inplace_kernel(Tensor& self) {
    return self;
}

Tensor sign_kernel(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        TP_THROW(NotImplementedError,
                 "sign is not supported for complex inputs");
    }
    return unary_meta(self);
}

Tensor& sign_inplace_kernel(Tensor& self) {
    return self;
}

// sgn is the complex-aware sign: it keeps the input dtype even for complex
// inputs, where sign is rejected.
Tensor sgn_kernel(const Tensor& self) {
    return unary_meta(self);
}

Tensor& sgn_inplace_kernel(Tensor& self) {
    return self;
}

// ---------------------------------------------------------------------------
// Clamps.  A scalar bound keeps the input's geometry and type; a tensor
// bound broadcasts like any binary pointwise op and promotes the pair.
// ---------------------------------------------------------------------------

Tensor clamp_min_scalar_kernel(const Tensor& self, const Scalar& /*min*/) {
    return unary_meta(self);
}

Tensor clamp_max_scalar_kernel(const Tensor& self, const Scalar& /*max*/) {
    return unary_meta(self);
}

Tensor clamp_min_tensor_kernel(const Tensor& self, const Tensor& min) {
    check_all_meta(self, min);
    return empty_meta(broadcast2(self, min), self.dtype(), self.device());
}

Tensor clamp_max_tensor_kernel(const Tensor& self, const Tensor& max) {
    check_all_meta(self, max);
    return empty_meta(broadcast2(self, max), self.dtype(), self.device());
}

// An in-place clamp writes element data a meta tensor does not own, so the
// result is the operand itself: its geometry and type are already the ones
// the write would leave behind.
Tensor& clamp_inplace_kernel(Tensor& self, const std::optional<Scalar>& /*min*/,
                             const std::optional<Scalar>& /*max*/) {
    return self;
}

Tensor& clamp_tensor_inplace_kernel(Tensor& self,
                                    const std::optional<Tensor>& /*min*/,
                                    const std::optional<Tensor>& /*max*/) {
    return self;
}

Tensor& clamp_min_scalar_inplace_kernel(Tensor& self, const Scalar& /*min*/) {
    return self;
}

Tensor& clamp_max_scalar_inplace_kernel(Tensor& self, const Scalar& /*max*/) {
    return self;
}

Tensor& clamp_min_tensor_inplace_kernel(Tensor& self, const Tensor& /*min*/) {
    return self;
}

Tensor& clamp_max_tensor_inplace_kernel(Tensor& self, const Tensor& /*max*/) {
    return self;
}

// ---------------------------------------------------------------------------
// Power.
// ---------------------------------------------------------------------------

DType pow_scalar_result_dtype(const Tensor& self, const Scalar& exponent) {
    if (self.dtype() == DType::Bool) {
        TP_THROW(TypeError, "pow is not supported for bool tensors");
    }
    if (isIntegralType(self.dtype()) && exponent.isIntegral() &&
        exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    return result_type(exponent, self.dtype());
}

Tensor pow_tensor_scalar_kernel(const Tensor& self, const Scalar& exponent) {
    return empty_meta(shape_of(self), pow_scalar_result_dtype(self, exponent),
                      self.device());
}

Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
    check_all_meta(self, exponent);
    if (self.dtype() == DType::Bool || exponent.dtype() == DType::Bool) {
        TP_THROW(TypeError, "pow is not supported for bool tensors");
    }
    return empty_meta(broadcast2(self, exponent),
                      promoteTypes(self.dtype(), exponent.dtype()),
                      self.device());
}

Tensor pow_scalar_tensor_kernel(const Scalar& base, const Tensor& exponent) {
    if (exponent.dtype() == DType::Bool) {
        TP_THROW(TypeError, "pow is not supported for bool tensors");
    }
    return empty_meta(shape_of(exponent), result_type(base, exponent.dtype()),
                      exponent.device());
}

// ---------------------------------------------------------------------------
// Gated linear unit: the activation axis is split in half, gated, and the
// two halves are multiplied pointwise, so the result drops half of that
// axis and keeps everything else.
// ---------------------------------------------------------------------------

Tensor glu_kernel(const Tensor& self, int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    }
    const int64_t nd = self.dim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        TP_THROW(IndexError, "glu: dimension out of range");
    }
    const int64_t size = self.size(dim);
    if (size % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension ",
                 dim, " is size ", size);
    }
    std::vector<int64_t> out = shape_of(self);
    out[static_cast<size_t>(dim)] /= 2;
    return empty_meta(out, self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// PReLU: one slope per channel, but the output keeps the input's geometry
// and element type (the weight is cast to the input's dtype when used).
// ---------------------------------------------------------------------------

Tensor prelu_kernel(const Tensor& self, const Tensor& /*weight*/) {
    return unary_meta(self);
}

// ---------------------------------------------------------------------------
// Scans: each slice along `dim` is folded left to right, so the answer keeps
// the input geometry.  An integral input (booleans included) answers in Int64
// unless the caller names a dtype; floating and complex inputs keep their own.
// ---------------------------------------------------------------------------

int64_t wrap_scan_dim(int64_t dim, int64_t ndim) {
    if (ndim == 0) {
        if (dim == -1 || dim == 0) return 0;
        TP_THROW(IndexError,
                 "Dimension out of range for a scalar tensor (expected -1 or "
                 "0, but got ", dim, ")");
    }
    return wrap_dim(dim, ndim);
}

DType scan_out_dtype(const Tensor& self, const std::optional<DType>& dtype) {
    return dtype.value_or(isIntegralType(self.dtype(), true) ? DType::Int64
                                                             : self.dtype());
}

void check_scan_dtype(DType out_dtype, const char* op) {
    if (!is_scalar_dtype(out_dtype) && !isComplexType(out_dtype)) {
        TP_THROW(TypeError, op, ": unsupported dtype");
    }
}

Tensor cumsum_kernel(const Tensor& self, int64_t dim,
                     std::optional<DType> dtype) {
    wrap_scan_dim(dim, self.dim());
    const DType out_dtype = scan_out_dtype(self, dtype);
    check_scan_dtype(out_dtype, "cumsum");
    return empty_meta(shape_of(self), out_dtype, self.device());
}

Tensor& cumsum_inplace_kernel(Tensor& self, int64_t dim,
                              std::optional<DType> dtype) {
    cumsum_kernel(self, dim, dtype);
    return self;
}

Tensor cumprod_kernel(const Tensor& self, int64_t dim,
                      std::optional<DType> dtype) {
    wrap_scan_dim(dim, self.dim());
    const DType out_dtype = scan_out_dtype(self, dtype);
    check_scan_dtype(out_dtype, "cumprod");
    return empty_meta(shape_of(self), out_dtype, self.device());
}

Tensor& cumprod_inplace_kernel(Tensor& self, int64_t dim,
                               std::optional<DType> dtype) {
    cumprod_kernel(self, dim, dtype);
    return self;
}

// ---------------------------------------------------------------------------
// Outer product: two vectors multiply into a matrix whose types promote.
// ---------------------------------------------------------------------------

Tensor outer_kernel(const Tensor& self, const Tensor& vec2) {
    if (self.dim() != 1 || vec2.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(),
                 "D and ", vec2.dim(), "D tensors");
    }
    return empty_meta({self.numel(), vec2.numel()},
                      promoteTypes(self.dtype(), vec2.dtype()),
                      self.device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaActivations) {
    // Activations.
    m.impl("relu", relu_kernel);
    m.impl("relu_", relu_inplace_kernel);
    m.impl("gelu", gelu_kernel);
    m.impl("gelu_", gelu_inplace_kernel);
    m.impl("tanh", tanh_kernel);
    m.impl("tanh_", tanh_inplace_kernel);
    m.impl("sigmoid", sigmoid_kernel);
    m.impl("sigmoid_", sigmoid_inplace_kernel);
    m.impl("silu_", silu_inplace_kernel);
    m.impl("mish", mish_kernel);
    m.impl("mish_", mish_inplace_kernel);
    m.impl("softplus", softplus_kernel);
    m.impl("softplus_", softplus_inplace_kernel);
    m.impl("elu", elu_kernel);
    m.impl("elu_", elu_inplace_kernel);
    m.impl("selu", selu_kernel);
    m.impl("selu_", selu_inplace_kernel);
    m.impl("celu", celu_kernel);
    m.impl("celu_", celu_inplace_kernel);
    m.impl("leaky_relu", leaky_relu_kernel);
    m.impl("leaky_relu_", leaky_relu_inplace_kernel);
    m.impl("hardsigmoid", hardsigmoid_kernel);
    m.impl("hardsigmoid_", hardsigmoid_inplace_kernel);
    m.impl("hardswish", hardswish_kernel);
    m.impl("hardswish_", hardswish_inplace_kernel);
    m.impl("hardtanh", hardtanh_kernel);
    m.impl("hardtanh_", hardtanh_inplace_kernel);
    m.impl("tanhshrink", tanhshrink_kernel);
    m.impl("softshrink", softshrink_kernel);
    m.impl("hardshrink", hardshrink_kernel);
    m.impl("relu6", relu6_kernel);
    m.impl("relu6_", relu6_inplace_kernel);
    m.impl("log_sigmoid", log_sigmoid_kernel);

    // Rounding and sign.
    m.impl("floor", floor_kernel);
    m.impl("ceil", ceil_kernel);
    m.impl("round", round_kernel);
    m.impl("trunc", trunc_kernel);
    m.impl("trunc_", trunc_inplace_kernel);
    m.impl("frac", frac_kernel);
    m.impl("frac_", frac_inplace_kernel);
    m.impl("sign", sign_kernel);
    m.impl("sign_", sign_inplace_kernel);
    m.impl("sgn", sgn_kernel);
    m.impl("sgn_", sgn_inplace_kernel);

    // Clamps.  The scalar forms are reachable both through the public
    // `clamp_min`/`clamp_max` functions (which dispatch to the .Scalar
    // keys) and through the C++ Tensor methods (which dispatch to the
    // unsuffixed keys); both spellings are registered.
    m.impl("clamp_min", clamp_min_scalar_kernel);
    m.impl("clamp_min.Scalar", clamp_min_scalar_kernel);
    m.impl("clamp_max", clamp_max_scalar_kernel);
    m.impl("clamp_max.Scalar", clamp_max_scalar_kernel);
    m.impl("clamp_min.Tensor", clamp_min_tensor_kernel);
    m.impl("clamp_max.Tensor", clamp_max_tensor_kernel);
    m.impl("clamp_", clamp_inplace_kernel);
    m.impl("clamp_.Scalar", clamp_inplace_kernel);
    m.impl("clamp_.Tensor", clamp_tensor_inplace_kernel);
    m.impl("clamp_min_", clamp_min_scalar_inplace_kernel);
    m.impl("clamp_min_.Scalar", clamp_min_scalar_inplace_kernel);
    m.impl("clamp_max_", clamp_max_scalar_inplace_kernel);
    m.impl("clamp_max_.Scalar", clamp_max_scalar_inplace_kernel);
    m.impl("clamp_min_.Tensor", clamp_min_tensor_inplace_kernel);
    m.impl("clamp_max_.Tensor", clamp_max_tensor_inplace_kernel);

    // Power.
    m.impl("pow.Tensor_Scalar", pow_tensor_scalar_kernel);
    m.impl("pow.Tensor_Tensor", pow_tensor_tensor_kernel);
    m.impl("pow.Scalar", pow_scalar_tensor_kernel);

    // Gated linear unit and PReLU.
    m.impl("glu", glu_kernel);
    m.impl("prelu", prelu_kernel);

    // Scans and the outer product.
    m.impl("cumsum", cumsum_kernel);
    m.impl("cumsum_", cumsum_inplace_kernel);
    m.impl("cumprod", cumprod_kernel);
    m.impl("cumprod_", cumprod_inplace_kernel);
    m.impl("outer", outer_kernel);
}

} // namespace meta
} // namespace tensorplay