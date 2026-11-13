// Shape-only kernels for the pointwise families on the meta device:
// arithmetic, the pair and scalar maps beyond the four arithmetic
// operators, the unary maps and their generated tables, the predicates,
// linear interpolation and dropout.  Every answer is the broadcast
// geometry under the op's own promotion rule; element values are never
// read.
#include "MetaCommon.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "Convolution.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include "Context.h"
#include "Generator.h"
#include "Scalar.h"
#include "DType.h"
#include "DTypeNames.h"
#include "DistributionDispatch.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "../BlasComposite.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace tensorplay {
namespace meta {
namespace {
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Pointwise arithmetic.
// ---------------------------------------------------------------------------
Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      add_like_dtype(self.dtype(), other.dtype(), alpha),
                      self.device());
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    return empty_meta(shape_of(self), add_like_scalar_dtype(self.dtype(), other, alpha),
                      self.device());
}

Tensor& add_out_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha,
                       Tensor& out) {
    Tensor expected = add_kernel(self, other, alpha);
    if (shape_of(out) != shape_of(expected) || out.dtype() != expected.dtype()) {
        TP_THROW(RuntimeError, "add.out: expected output with shape ",
                 expected.numel(), " and dtype ", toString(expected.dtype()),
                 " but got shape ", out.numel(), " and dtype ",
                 toString(out.dtype()));
    }
    return out;
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    (void)other;
    (void)alpha;
    return self;
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      add_like_dtype(self.dtype(), other.dtype(), alpha),
                      self.device());
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    return empty_meta(shape_of(self), add_like_scalar_dtype(self.dtype(), other, alpha),
                      self.device());
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    (void)alpha;
    check_inplace_fits(self, other);
    return self;
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    (void)other;
    (void)alpha;
    return self;
}

// Reversed subtraction keeps the tensor dtype unless the scalar itself is
// floating, in which case an integral tensor is answered at single precision.
DType weak_scalar_promote(DType dtype, const Scalar& scalar) {
    if (!isFloatingType(scalar.dtype())) return dtype;
    if (isFloatingType(dtype)) return dtype;
    return DType::Float32;
}

Tensor rsub_tensor_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    check_all_meta(self, other);
    (void)alpha;
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

Tensor rsub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    (void)alpha;
    const DType dtype = weak_scalar_promote(self.dtype(), other);
    // The scalar path has no complex arithmetic loop, so a complex tensor
    // times a whole-number scalar is refused while a floating scalar still
    // narrows it to single precision.
    if (isComplexType(dtype)) {
        TP_THROW(TypeError, "rsub: unsupported dtype");
    }
    return empty_meta(shape_of(self), dtype, self.device());
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), result_type(other, self.dtype()),
                      self.device());
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
    (void)other;
    return self;
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      div_like_dtype(self.dtype(), other.dtype()), self.device());
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), div_like_scalar_dtype(self.dtype(), other),
                      self.device());
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& div_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
    (void)other;
    return self;
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Pointwise maps beyond the four arithmetic ones.  The pair maps that compute
// at floating precision answer whole-number inputs at the default real type;
// the modulo and division families keep an integral pair integral, and true
// division lifts every integral pair to a real type.
// ---------------------------------------------------------------------------
DType float_pair_dtype(DType a, DType b) {
    const DType out = promoteTypes(a, b);
    return isFloatingType(out) ? out : DType::Float32;
}

Tensor float_pair_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      float_pair_dtype(self.dtype(), other.dtype()),
                      self.device());
}

Tensor float_pair_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self),
                      float_pair_dtype(self.dtype(),
                                       result_type(other, self.dtype())),
                      self.device());
}

Tensor same_pair_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      promoteTypes(self.dtype(), other.dtype()), self.device());
}

// A complex scalar carries no lane in the whole-number maps, so it answers
// at the operand's own type instead of widening the result.
Tensor same_pair_scalar_kernel(const Tensor& self, const Scalar& other) {
    const DType out = other.isComplex()
                          ? self.dtype()
                          : promoteTypes(self.dtype(),
                                         result_type(other, self.dtype()));
    return empty_meta(shape_of(self), out, self.device());
}

void reject_complex_pair(const Tensor& a, const Tensor& b, const char* op) {
    if (isComplexType(a.dtype()) || isComplexType(b.dtype())) {
        TP_THROW(TypeError, op, ": unsupported dtype for complex operands");
    }
}

void reject_complex_scalar(const Tensor& self, const Scalar& other,
                           const char* op) {
    if (isComplexType(self.dtype()) || other.isComplex()) {
        TP_THROW(TypeError, op, ": unsupported dtype for complex operands");
    }
}

Tensor fmod_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "fmod");
    return same_pair_kernel(self, other);
}

Tensor fmod_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "fmod");
    return same_pair_scalar_kernel(self, other);
}

Tensor remainder_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "remainder");
    return same_pair_kernel(self, other);
}

Tensor remainder_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "remainder");
    return same_pair_scalar_kernel(self, other);
}

Tensor remainder_scalar_tensor_kernel(const Scalar& self, const Tensor& other) {
    if (isComplexType(other.dtype()) || self.isComplex()) {
        TP_THROW(TypeError,
                 "remainder: unsupported dtype for complex operands");
    }
    const DType out = self.isComplex()
                          ? other.dtype()
                          : promoteTypes(result_type(self, other.dtype()),
                                         other.dtype());
    return empty_meta(shape_of(other), out, other.device());
}

Tensor floor_divide_pair_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "floor_divide");
    return same_pair_kernel(self, other);
}

Tensor floor_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
    reject_complex_scalar(self, other, "floor_divide");
    return same_pair_scalar_kernel(self, other);
}

Tensor heaviside_kernel(const Tensor& self, const Tensor& other) {
    reject_complex_pair(self, other, "heaviside");
    return same_pair_kernel(self, other);
}

Tensor copysign_scalar_kernel(const Tensor& self, const Scalar& other) {
    (void)other;
    // The sign is carried at single precision and never widens the answer.
    return empty_meta(shape_of(self),
                      float_pair_dtype(self.dtype(), DType::Float32),
                      self.device());
}

Tensor true_divide_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    return empty_meta(broadcast2(self, other),
                      div_like_dtype(self.dtype(), other.dtype()), self.device());
}

Tensor true_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
    return empty_meta(shape_of(self), div_like_scalar_dtype(self.dtype(), other),
                      self.device());
}

// An in-place pair map writes element data only, so the operand's geometry
// and value type are already the ones the write would leave behind.
Tensor& pair_inplace_kernel(Tensor& self, const Tensor& other) {
    check_inplace_fits(self, other);
    return self;
}

Tensor& pair_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
    (void)other;
    return self;
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Pointwise unary. abs of a complex input produces the matching real value
// type; every other unary keeps dtype and shape.
// ---------------------------------------------------------------------------
Tensor abs_kernel(const Tensor& self) {
    DType out = isComplexType(self.dtype()) ? toRealValueType(self.dtype())
                                            : self.dtype();
    return empty_meta(shape_of(self), out, self.device());
}

Tensor neg_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& neg_inplace_kernel(Tensor& self) {
    return self;
}

Tensor exp_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
}

Tensor log_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
}

Tensor sqrt_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),
                      self.device());
}

Tensor clamp_kernel(const Tensor& self, const std::optional<Scalar>& /*min*/,
                    const std::optional<Scalar>& /*max*/) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor threshold_kernel(const Tensor& self, const Scalar& /*threshold*/,
                        const Scalar& /*value*/) {
    // The elementwise scan has no complex form, every other dtype keeps its
    // own type and geometry.
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "threshold: unsupported dtype");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Activation and root pointwise operations. Every one of them maps an
// element to an element, so the result repeats the operand's geometry and
// type.
// ---------------------------------------------------------------------------
#define META_UNARY_KERNEL(NAME)                                               \
    Tensor NAME##_kernel(const Tensor& self) {                                \
        return empty_meta(shape_of(self), self.dtype(), self.device());       \
    }

// A float-only unary op answers an integral operand in single precision; see
// ``unary_float_dtype``.
#define META_UNARY_FLOAT_KERNEL(NAME)                                         \
    Tensor NAME##_kernel(const Tensor& self) {                                \
        return empty_meta(shape_of(self), unary_float_dtype(self.dtype()),     \
                          self.device());                                     \
    }

// A float-math sweep accepts operands of any width: the computation runs in
// floating precision, so a floating operand keeps its width and everything
// else -- integers, booleans, complex planes -- is answered in single
// precision; see ``float_math_dtype``.
#define META_UNARY_FLOAT_MATH_KERNEL(NAME)                                    \
    Tensor NAME##_kernel(const Tensor& self) {                                \
        return empty_meta(shape_of(self), float_math_dtype(self.dtype()),     \
                          self.device());                                     \
    }

// The error-function maps have a real-number form only: integers answer in
// single precision, every other operand -- complex planes, booleans, the
// 8-bit floats -- is refused.
Tensor erf_kernel(const Tensor& self) {
    const DType dt = self.dtype();
    if (!isIntegralType(dt) && !isFloatingType(dt)) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return empty_meta(shape_of(self), float_math_dtype(dt), self.device());
}

Tensor erfc_kernel(const Tensor& self) {
    const DType dt = self.dtype();
    if (!isIntegralType(dt) && !isFloatingType(dt)) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return empty_meta(shape_of(self), float_math_dtype(dt), self.device());
}

Tensor lgamma_kernel(const Tensor& self) {
    const DType dt = self.dtype();
    if (!isIntegralType(dt) && !isFloatingType(dt)) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return empty_meta(shape_of(self), float_math_dtype(dt), self.device());
}

META_UNARY_FLOAT_KERNEL(silu)
META_UNARY_FLOAT_KERNEL(rsqrt)
META_UNARY_FLOAT_KERNEL(cos)
META_UNARY_FLOAT_KERNEL(sin)
META_UNARY_FLOAT_MATH_KERNEL(exp2)
META_UNARY_FLOAT_KERNEL(expm1)
META_UNARY_FLOAT_KERNEL(log2)
META_UNARY_FLOAT_KERNEL(log10)
META_UNARY_FLOAT_KERNEL(log1p)
META_UNARY_FLOAT_KERNEL(tan)
META_UNARY_FLOAT_KERNEL(asin)
META_UNARY_FLOAT_KERNEL(acos)
META_UNARY_FLOAT_KERNEL(atan)
META_UNARY_FLOAT_KERNEL(sinh)
META_UNARY_FLOAT_KERNEL(cosh)
META_UNARY_FLOAT_KERNEL(asinh)
META_UNARY_FLOAT_KERNEL(acosh)
META_UNARY_FLOAT_KERNEL(atanh)
META_UNARY_FLOAT_KERNEL(reciprocal)
META_UNARY_KERNEL(square)
META_UNARY_FLOAT_MATH_KERNEL(sinc)
META_UNARY_FLOAT_MATH_KERNEL(erfinv)
META_UNARY_FLOAT_MATH_KERNEL(digamma)
META_UNARY_FLOAT_MATH_KERNEL(i0)
META_UNARY_FLOAT_KERNEL(deg2rad)
META_UNARY_FLOAT_KERNEL(rad2deg)
#undef META_UNARY_KERNEL
#undef META_UNARY_FLOAT_KERNEL
#undef META_UNARY_FLOAT_MATH_KERNEL

Tensor absolute_kernel(const Tensor& self) {
    return abs_kernel(self);
}

// An in-place pointwise operation overwrites the operand's own bytes, so the
// geometry and the value type both stay as they were.
#define META_UNARY_INPLACE_KERNEL(NAME)                                       \
    Tensor& NAME##_inplace_kernel(Tensor& self) { return self; }

META_UNARY_INPLACE_KERNEL(abs)
META_UNARY_INPLACE_KERNEL(absolute)
META_UNARY_INPLACE_KERNEL(sqrt)
META_UNARY_INPLACE_KERNEL(rsqrt)
META_UNARY_INPLACE_KERNEL(exp)
META_UNARY_INPLACE_KERNEL(exp2)
META_UNARY_INPLACE_KERNEL(expm1)
META_UNARY_INPLACE_KERNEL(log)
META_UNARY_INPLACE_KERNEL(log2)
META_UNARY_INPLACE_KERNEL(log10)
META_UNARY_INPLACE_KERNEL(log1p)
META_UNARY_INPLACE_KERNEL(sin)
META_UNARY_INPLACE_KERNEL(cos)
META_UNARY_INPLACE_KERNEL(tan)
META_UNARY_INPLACE_KERNEL(asin)
META_UNARY_INPLACE_KERNEL(acos)
META_UNARY_INPLACE_KERNEL(atan)
META_UNARY_INPLACE_KERNEL(sinh)
META_UNARY_INPLACE_KERNEL(cosh)
META_UNARY_INPLACE_KERNEL(asinh)
META_UNARY_INPLACE_KERNEL(acosh)
META_UNARY_INPLACE_KERNEL(atanh)
META_UNARY_INPLACE_KERNEL(ceil)
META_UNARY_INPLACE_KERNEL(floor)
META_UNARY_INPLACE_KERNEL(round)
META_UNARY_INPLACE_KERNEL(reciprocal)
META_UNARY_INPLACE_KERNEL(square)
META_UNARY_INPLACE_KERNEL(sinc)
META_UNARY_INPLACE_KERNEL(erf)
META_UNARY_INPLACE_KERNEL(erfc)
META_UNARY_INPLACE_KERNEL(erfinv)
META_UNARY_INPLACE_KERNEL(lgamma)
META_UNARY_INPLACE_KERNEL(digamma)
META_UNARY_INPLACE_KERNEL(i0)
META_UNARY_INPLACE_KERNEL(deg2rad)
META_UNARY_INPLACE_KERNEL(rad2deg)
#undef META_UNARY_INPLACE_KERNEL
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Predicates and the maps that answer beside the operand's own type.  A
// predicate reports element data, so its answer is Bool in the operand's
// geometry; the floating predicates have no complex lane, while isreal and
// is_complex inspect only the value type.  The remaining maps compute in real
// arithmetic: a complex operand answers at its real width and a whole-number
// operand at the default real type.
// ---------------------------------------------------------------------------
Tensor bool_answer(const Tensor& self, const char* op) {
    const DType dtype = self.dtype();
    const bool has_real_lane = dtype == DType::Bool || isIntegralType(dtype) ||
                               (isFloatingType(dtype) && !isFloat8Type(dtype));
    if (!has_real_lane) {
        TP_THROW(TypeError, op, ": unsupported dtype");
    }
    return empty_meta(shape_of(self), DType::Bool, self.device());
}

#define META_BOOL_PREDICATE(NAME)                                             \
    Tensor NAME##_kernel(const Tensor& self) { return bool_answer(self, #NAME); }

META_BOOL_PREDICATE(isnan)
META_BOOL_PREDICATE(isinf)
META_BOOL_PREDICATE(isfinite)
META_BOOL_PREDICATE(isneginf)
META_BOOL_PREDICATE(isposinf)
META_BOOL_PREDICATE(signbit)
#undef META_BOOL_PREDICATE

Tensor isreal_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), DType::Bool, self.device());
}

bool is_complex_kernel(const Tensor& self) {
    return isComplexType(self.dtype());
}

DType float_unary_dtype(DType in) {
    if (isComplexType(in)) return toRealValueType(in);
    return isFloatingType(in) ? in : DType::Float32;
}

Tensor angle_kernel(const Tensor& self) {
    if (self.dtype() == DType::Bool) {
        TP_THROW(TypeError, "Unsupported dtype (expected float)");
    }
    return empty_meta(shape_of(self), float_unary_dtype(self.dtype()),
                      self.device());
}

Tensor positive_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor entr_kernel(const Tensor& self) {
    return empty_meta(shape_of(self), float_math_dtype(self.dtype()),
                      self.device());
}

Tensor logit_kernel(const Tensor& self, const std::optional<Scalar>& /*eps*/) {
    return empty_meta(shape_of(self), float_math_dtype(self.dtype()),
                      self.device());
}

Tensor& logit_inplace_kernel(Tensor& self, const std::optional<Scalar>& /*eps*/) {
    return self;
}

Tensor nan_to_num_kernel(const Tensor& self, const Scalar& /*nan*/,
                         const std::optional<Scalar>& /*posinf*/,
                         const std::optional<Scalar>& /*neginf*/) {
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& nan_to_num_inplace_kernel(Tensor& self, const Scalar& /*nan*/,
                                  const std::optional<Scalar>& /*posinf*/,
                                  const std::optional<Scalar>& /*neginf*/) {
    if (isComplexType(self.dtype())) {
        TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
    return self;
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Linear interpolation: both endpoints take part in the broadcast, and the
// weight adds its own type when it is a tensor.  A whole-numbered promotion
// answers at single precision, and a floating scalar weight lifts it too.
// ---------------------------------------------------------------------------
Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end,
                          const Scalar& weight) {
    check_all_meta(self, end);
    DType out = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) out = promoteTypes(out, DType::Float32);
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(broadcast2(self, end), out, self.device());
}

Tensor lerp_tensor_kernel(const Tensor& self, const Tensor& end,
                          const Tensor& weight) {
    check_all_meta(self, end);
    check_all_meta(self, weight);
    DType out = promoteTypes(promoteTypes(self.dtype(), end.dtype()),
                             weight.dtype());
    if (isIntegralType(out)) out = DType::Float32;
    return empty_meta(
        broadcast_shapes(broadcast2(self, end), shape_of(weight)), out,
        self.device());
}

void check_lerp_destination(const Tensor& self,
                            const std::vector<int64_t>& out, const char* op) {
    if (out != shape_of(self)) {
        TP_THROW(RuntimeError, op,
                 ": the broadcast result does not take the destination's shape");
    }
}

Tensor& lerp_scalar_inplace_kernel(Tensor& self, const Tensor& end,
                                   const Scalar& weight) {
    check_lerp_destination(self, broadcast2(self, end), "lerp_");
    (void)weight;
    return self;
}

Tensor& lerp_tensor_inplace_kernel(Tensor& self, const Tensor& end,
                                   const Tensor& weight) {
    check_lerp_destination(self,
                           broadcast_shapes(broadcast2(self, end),
                                            shape_of(weight)),
                           "lerp_");
    return self;
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dropout. Dropping elements changes values, not the geometry, and the mask
// that says which were kept has the same geometry as a boolean tensor.
// ---------------------------------------------------------------------------
std::tuple<Tensor, Tensor> native_dropout_kernel(const Tensor& input, double p,
                                                 std::optional<bool> /*train*/) {
    if (p < 0 || p > 1) {
        TP_THROW(ValueError, "native_dropout: p must be in [0, 1]");
    }
    return {empty_meta(shape_of(input), input.dtype(), input.device()),
            empty_meta(shape_of(input), DType::Bool, input.device())};
}
} // anonymous namespace
TENSORPLAY_LIBRARY_IMPL(Meta, MetaPointwise) {
    m.impl("add.Tensor", add_kernel);
    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("add.out", add_out_kernel);
    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    m.impl("sub.Tensor", sub_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    m.impl("rsub.Tensor", rsub_tensor_kernel);
    m.impl("rsub.Scalar", rsub_scalar_kernel);
    m.impl("mul.Tensor", mul_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    m.impl("div.Tensor", div_kernel);
    m.impl("div.Scalar", div_scalar_kernel);
    m.impl("div_.Tensor", div_inplace_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);
    m.impl("fmod.Tensor", fmod_pair_kernel);
    m.impl("fmod.Scalar", fmod_scalar_kernel);
    m.impl("remainder.Tensor", remainder_pair_kernel);
    m.impl("remainder.Scalar", remainder_scalar_kernel);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_kernel);
    m.impl("floor_divide", floor_divide_pair_kernel);
    m.impl("floor_divide.Scalar", floor_divide_scalar_kernel);
    m.impl("true_divide.Tensor", true_divide_kernel);
    m.impl("true_divide.Scalar", true_divide_scalar_kernel);
    m.impl("copysign.Tensor", float_pair_kernel);
    m.impl("copysign.Scalar", copysign_scalar_kernel);
    m.impl("nextafter", float_pair_kernel);
    m.impl("hypot", float_pair_kernel);
    m.impl("atan2", float_pair_kernel);
    m.impl("heaviside", heaviside_kernel);
    m.impl("xlogy", float_pair_kernel);
    m.impl("atan2_", pair_inplace_kernel);
    m.impl("heaviside_", pair_inplace_kernel);
    m.impl("hypot_", pair_inplace_kernel);
    m.impl("nextafter_", pair_inplace_kernel);
    m.impl("neg", neg_kernel);
    m.impl("neg_", neg_inplace_kernel);
    m.impl("clamp", clamp_kernel);
    m.impl("threshold", threshold_kernel);
    m.impl("abs", abs_kernel);
    m.impl("absolute", absolute_kernel);
    m.impl("abs_", abs_inplace_kernel);
    m.impl("absolute_", absolute_inplace_kernel);
    m.impl("exp", exp_kernel);
    m.impl("exp2", exp2_kernel);
    m.impl("expm1", expm1_kernel);
    m.impl("exp_", exp_inplace_kernel);
    m.impl("exp2_", exp2_inplace_kernel);
    m.impl("expm1_", expm1_inplace_kernel);
    m.impl("log", log_kernel);
    m.impl("log2", log2_kernel);
    m.impl("log10", log10_kernel);
    m.impl("log1p", log1p_kernel);
    m.impl("log_", log_inplace_kernel);
    m.impl("log2_", log2_inplace_kernel);
    m.impl("log10_", log10_inplace_kernel);
    m.impl("log1p_", log1p_inplace_kernel);
    m.impl("sqrt", sqrt_kernel);
    m.impl("rsqrt", rsqrt_kernel);
    m.impl("sqrt_", sqrt_inplace_kernel);
    m.impl("rsqrt_", rsqrt_inplace_kernel);
    m.impl("tan", tan_kernel);
    m.impl("asin", asin_kernel);
    m.impl("acos", acos_kernel);
    m.impl("atan", atan_kernel);
    m.impl("sinh", sinh_kernel);
    m.impl("cosh", cosh_kernel);
    m.impl("asinh", asinh_kernel);
    m.impl("acosh", acosh_kernel);
    m.impl("atanh", atanh_kernel);
    // The arc-prefixed spellings and the word forms of the arithmetic
    // primitives answer the same geometry as their base names.
    m.impl("arcsin", asin_kernel);
    m.impl("arccos", acos_kernel);
    m.impl("arctan", atan_kernel);
    m.impl("arcsinh", asinh_kernel);
    m.impl("arccosh", acosh_kernel);
    m.impl("arctanh", atanh_kernel);
    m.impl("negative", neg_kernel);
    m.impl("multiply.Tensor", mul_kernel);
    m.impl("subtract.Tensor", sub_kernel);
    m.impl("divide.Tensor", div_kernel);
    m.impl("logaddexp", float_pair_kernel);
    m.impl("logaddexp2", float_pair_kernel);
    m.impl("tan_", tan_inplace_kernel);
    m.impl("asin_", asin_inplace_kernel);
    m.impl("acos_", acos_inplace_kernel);
    m.impl("atan_", atan_inplace_kernel);
    m.impl("sinh_", sinh_inplace_kernel);
    m.impl("cosh_", cosh_inplace_kernel);
    m.impl("asinh_", asinh_inplace_kernel);
    m.impl("acosh_", acosh_inplace_kernel);
    m.impl("atanh_", atanh_inplace_kernel);
    m.impl("reciprocal", reciprocal_kernel);
    m.impl("square", square_kernel);
    m.impl("sinc", sinc_kernel);
    m.impl("reciprocal_", reciprocal_inplace_kernel);
    m.impl("square_", square_inplace_kernel);
    m.impl("sinc_", sinc_inplace_kernel);
    m.impl("erf", erf_kernel);
    m.impl("erfc", erfc_kernel);
    m.impl("erfinv", erfinv_kernel);
    m.impl("lgamma", lgamma_kernel);
    m.impl("digamma", digamma_kernel);
    m.impl("i0", i0_kernel);
    m.impl("erf_", erf_inplace_kernel);
    m.impl("erfc_", erfc_inplace_kernel);
    m.impl("erfinv_", erfinv_inplace_kernel);
    m.impl("lgamma_", lgamma_inplace_kernel);
    m.impl("digamma_", digamma_inplace_kernel);
    m.impl("i0_", i0_inplace_kernel);
    m.impl("deg2rad", deg2rad_kernel);
    m.impl("rad2deg", rad2deg_kernel);
    m.impl("deg2rad_", deg2rad_inplace_kernel);
    m.impl("rad2deg_", rad2deg_inplace_kernel);
    m.impl("ceil_", ceil_inplace_kernel);
    m.impl("floor_", floor_inplace_kernel);
    m.impl("round_", round_inplace_kernel);
    m.impl("silu", silu_kernel);
    m.impl("cos", cos_kernel);
    m.impl("sin", sin_kernel);
    m.impl("cos_", cos_inplace_kernel);
    m.impl("sin_", sin_inplace_kernel);
    m.impl("angle", angle_kernel);
    m.impl("isnan", isnan_kernel);
    m.impl("isinf", isinf_kernel);
    m.impl("isfinite", isfinite_kernel);
    m.impl("isneginf", isneginf_kernel);
    m.impl("isposinf", isposinf_kernel);
    m.impl("signbit", signbit_kernel);
    m.impl("isreal", isreal_kernel);
    m.impl("is_complex", is_complex_kernel);
    m.impl("positive", positive_kernel);
    m.impl("entr", entr_kernel);
    m.impl("logit", logit_kernel);
    m.impl("logit_", logit_inplace_kernel);
    m.impl("nan_to_num", nan_to_num_kernel);
    m.impl("nan_to_num_", nan_to_num_inplace_kernel);
    m.impl("lerp", lerp_scalar_kernel);
    m.impl("lerp.Tensor", lerp_tensor_kernel);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel);
    m.impl("native_dropout", native_dropout_kernel);
}

} // namespace meta
} // namespace tensorplay
