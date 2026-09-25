// Core operators - CUDA kernels: binary pointwise arithmetic: the additive, multiplicative
// and remainder families plus the truncating and floor division modes.
//
// Pointwise kernels over a grid-stride loop; the shared plumbing comes
// from the pointwise header and the per-operator bodies live here.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {
namespace {

Tensor rsub_scalar_cuda(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    DType dt = isFloatingType(other.dtype())
                   ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32)
                   : self.dtype();
    Tensor sc = self.to(dt).contiguous();
    Tensor full = Tensor::full({}, other, dt, self.device())
                      .expand(shape_of(sc)).contiguous();
    // other - alpha * self, routed through the shared subtract kernel.
    return sub_kernel(full, sc, alpha);
}

Tensor rsub_tensor_cuda(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    return sub_kernel(other, self, alpha);
}

Tensor true_divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    // The float loop only addresses real buffers, so a complex operand takes
    // the canonical division path instead.
    if (isComplexType(self.dtype()) || isComplexType(other.dtype())) {
        return div_kernel(self, other);
    }
    return binary_float_cuda(self, other,
                             HFn1{}, "true_divide");
}

Tensor true_divide_scalar_cuda(const Tensor& self, const Scalar& other) {
    // A Float32 stand-in would widen Half/BFloat16 inputs.
    const DType dt = scalar_promote(self.dtype(), other);
    return true_divide_tensor_cuda(self, Tensor::full({}, other, dt, self.device()));
}

Tensor divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    return true_divide_tensor_cuda(self, other);
}

Tensor divide_scalar_cuda(const Tensor& self, const Scalar& other) {
    return true_divide_scalar_cuda(self, other);
}

Tensor remainder_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            HFn2{},
                            "remainder");
}

Tensor remainder_scalar_cuda(const Tensor& self, const Scalar& other) {
    // Forcing the scalar into self's dtype would truncate a float divisor
    // against an integral tensor; the pair promotes first.
    const DType dt = scalar_promote(self.dtype(), other);
    return remainder_tensor_cuda(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor remainder_scalar_tensor_cuda(const Scalar& self, const Tensor& other) {
    const DType dt = scalar_promote(other.dtype(), self);
    return remainder_tensor_cuda(Tensor::full({}, self, dt, other.device()), other.to(dt));
}

Tensor fmod_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            HFn3{},
                            "fmod");
}

Tensor fmod_scalar_cuda(const Tensor& self, const Scalar& other) {
    const DType dt = scalar_promote(self.dtype(), other);
    return fmod_tensor_cuda(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor subtract_tensor_cuda(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    // alpha == 1 is by far the common call and keeps the unscaled loop.
    if (!alpha.isComplex() && alpha.toDouble() == 1.0) {
        return binary_same_cuda(self, other,
                                HFn4{}, "subtract");
    }
    const double al = alpha.toDouble();
    return binary_same_cuda(self, other,
                            HFn5{al}, "subtract");
}

Tensor subtract_scalar_cuda(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    const DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cuda(self.to(dt),
                                Tensor::full({}, other, dt, self.device()), alpha);
}

Tensor multiply_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            HFn6{}, "multiply");
}

Tensor multiply_scalar_cuda(const Tensor& self, const Scalar& other) {
    double ov = other.toDouble();
    return dtype_unary_cuda(self,
                            HFn7{ov},
                            "multiply");
}

// ---------------------------------------------------------------------------
// Division with an explicit rounding mode
// ---------------------------------------------------------------------------

Tensor div_rounded_core(const Tensor& a, const Tensor& b, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_tensor_cuda(a, b);
    // Rounded division stays in the input dtype: an integral pair must come
    // back integral, which the float promotion of true division loses.
    const bool floor_mode = (rounding == DivRounding::kFloor);
    return binary_same_cuda(a, b, HFn8{floor_mode}, "div");
}

Tensor div_rounded_scalar(const Tensor& self, Scalar other, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_scalar_cuda(self, other);
    const DType dt = scalar_promote(self.dtype(), other);
    return div_rounded_core(self.to(dt), Tensor::full({}, other, dt, self.device()),
                            rounding);
}


Tensor div_mode_tensor_cuda(const Tensor& self, const Tensor& other,
                            const std::optional<std::string>& rounding_mode) {
    return div_rounded_core(self, other, parse_div_rounding(rounding_mode));
}

Tensor div_mode_scalar_cuda(const Tensor& self, const Scalar& other,
                            const std::optional<std::string>& rounding_mode) {
    return div_rounded_scalar(self, other, parse_div_rounding(rounding_mode));
}

Tensor floor_divide_cuda(const Tensor& self, const Tensor& other) {
    return div_rounded_core(self, other, DivRounding::kFloor);
}

Tensor floor_divide_scalar_cuda(const Tensor& self, const Scalar& other) {
    return div_rounded_scalar(self, other, DivRounding::kFloor);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, BinaryPointwiseKernels) {
    m.impl("rsub.Scalar", rsub_scalar_cuda);
    m.impl("rsub.Tensor", rsub_tensor_cuda);
    m.impl("true_divide.Tensor", true_divide_tensor_cuda);
    m.impl("true_divide.Scalar", true_divide_scalar_cuda);
    m.impl("divide.Tensor", divide_tensor_cuda);
    m.impl("divide.Scalar", divide_scalar_cuda);
    m.impl("remainder.Tensor", remainder_tensor_cuda);
    m.impl("remainder.Scalar", remainder_scalar_cuda);
    m.impl("fmod.Tensor", fmod_tensor_cuda);
    m.impl("fmod.Scalar", fmod_scalar_cuda);
    m.impl("subtract.Tensor", subtract_tensor_cuda);
    m.impl("subtract.Scalar", subtract_scalar_cuda);
    m.impl("multiply.Tensor", multiply_tensor_cuda);
    m.impl("multiply.Scalar", multiply_scalar_cuda);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_cuda);
    m.impl("div.Tensor_mode", div_mode_tensor_cuda);
    m.impl("div.Scalar_mode", div_mode_scalar_cuda);
    m.impl("divide.Tensor_mode", div_mode_tensor_cuda);
    m.impl("divide.Scalar_mode", div_mode_scalar_cuda);
    m.impl("floor_divide", floor_divide_cuda);
    m.impl("floor_divide.Scalar", floor_divide_scalar_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
