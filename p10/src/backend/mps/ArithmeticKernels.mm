#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

namespace {

// Materializes a Scalar as a 1-element MPS tensor of the promoted dtype so
// the scalar variants of binary ops reuse the tensor path.
Tensor scalarToMPS(const Scalar& s, DType dtype, const Device& device) {
  Tensor t = Tensor({1}, dtype, device);
  fill_kernel(t, s);
  return t;
}

MPSBinaryOpBlock addBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock subBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock mulBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock divBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock powBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                            MPSGraphTensor* b) {
  return [g powerWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock remainderBlock = ^MPSGraphTensor*(MPSGraph* g,
                                                  MPSGraphTensor* a,
                                                  MPSGraphTensor* b) {
  // Python-style remainder: a - floor(a/b) * b
  MPSGraphTensor* q = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
  MPSGraphTensor* qf = [g floorWithTensor:q name:nil];
  MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:qf
                                            secondaryTensor:b
                                                       name:nil];
  return [g subtractionWithPrimaryTensor:a secondaryTensor:prod name:nil];
};

MPSBinaryOpBlock fmodBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  // C-style fmod: a - trunc(a/b) * b
  MPSGraphTensor* q = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
  MPSGraphTensor* qt = [g truncateWithTensor:q name:nil];
  MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:qt
                                            secondaryTensor:b
                                                       name:nil];
  return [g subtractionWithPrimaryTensor:a secondaryTensor:prod name:nil];
};

MPSBinaryOpBlock maximumBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                MPSGraphTensor* b) {
  return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock minimumBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                MPSGraphTensor* b) {
  return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock atan2Block = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g atan2WithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock hypotBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                              MPSGraphTensor* b) {
  return [g hypotWithPrimaryTensor:a secondaryTensor:b name:nil];
};

} // namespace

// ---------------------------------------------------------------------------
// add / sub / mul / div
// ---------------------------------------------------------------------------

Tensor add_kernel(const Tensor& self, const Tensor& other,
                  const Scalar& alpha) {
  return binaryOp(self, other, addBlock, alpha);
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other,
                        const Scalar& alpha) {
  Tensor result = add_kernel(self, other, alpha);
  self.copy_(result);
  return self;
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other,
                         const Scalar& alpha) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return add_kernel(self, other_t, alpha);
}

Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other,
                                  const Scalar& alpha) {
  Tensor result = add_scalar_kernel(self, other, alpha);
  self.copy_(result);
  return self;
}

Tensor sub_kernel(const Tensor& self, const Tensor& other,
                  const Scalar& alpha) {
  if (self.dtype() == DType::Bool && other.dtype() == DType::Bool) {
    TP_THROW(RuntimeError,
             "Subtraction, the `-` operator, with two bool tensors is not "
             "supported on the MPS backend");
  }
  return binaryOp(self, other, subBlock, alpha);
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other,
                           const Scalar& alpha) {
  Tensor result = sub_kernel(self, other, alpha);
  self.copy_(result);
  return self;
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other,
                         const Scalar& alpha) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return sub_kernel(self, other_t, alpha);
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, const Scalar& other,
                                  const Scalar& alpha) {
  Tensor result = sub_scalar_kernel(self, other, alpha);
  self.copy_(result);
  return self;
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, mulBlock);
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
  Tensor result = mul_kernel(self, other);
  self.copy_(result);
  return self;
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return mul_kernel(self, other_t);
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  Tensor result = mul_scalar_kernel(self, other);
  self.copy_(result);
  return self;
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
  // Integral division promotes to the default floating dtype, matching the
  // framework-wide div semantics.
  std::optional<DType> common;
  if (isIntegralType(self.dtype(), true) &&
      isIntegralType(other.dtype(), true)) {
    common = DType::Float32;
  }
  return binaryOp(self, other, divBlock, std::nullopt, common);
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
  Tensor result = div_kernel(self, other);
  self.copy_(result);
  return self;
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return div_kernel(self, other_t);
}

Tensor& div_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  Tensor result = div_scalar_kernel(self, other);
  self.copy_(result);
  return self;
}

// ---------------------------------------------------------------------------
// pow
// ---------------------------------------------------------------------------

Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
  return binaryOp(self, exponent, powBlock);
}

Tensor pow_scalar_kernel(const Tensor& self, const Scalar& exponent) {
  const DType common = result_type(exponent, self.dtype());
  Tensor exponent_t = scalarToMPS(exponent, common, self.device());
  return pow_tensor_tensor_kernel(self, exponent_t);
}

Tensor pow_scalar_tensor_kernel(const Scalar& base, const Tensor& exponent) {
  const DType common = result_type(base, exponent.dtype());
  Tensor base_t = scalarToMPS(base, common, exponent.device());
  return pow_tensor_tensor_kernel(base_t, exponent);
}

Tensor& pow_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
  Tensor result = pow_tensor_tensor_kernel(self, exponent);
  self.copy_(result);
  return self;
}

Tensor& pow_scalar_inplace_kernel(Tensor& self, const Scalar& exponent) {
  Tensor result = pow_scalar_kernel(self, exponent);
  self.copy_(result);
  return self;
}

// ---------------------------------------------------------------------------
// remainder / fmod
// ---------------------------------------------------------------------------

Tensor remainder_tensor_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, remainderBlock);
}

Tensor remainder_scalar_kernel(const Tensor& self, const Scalar& other) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return remainder_tensor_kernel(self, other_t);
}

Tensor fmod_tensor_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, fmodBlock);
}

Tensor fmod_scalar_kernel(const Tensor& self, const Scalar& other) {
  const DType common = result_type(other, self.dtype());
  Tensor other_t = scalarToMPS(other, common, self.device());
  return fmod_tensor_kernel(self, other_t);
}

// ---------------------------------------------------------------------------
// maximum / minimum / fmax / fmin
// ---------------------------------------------------------------------------

Tensor maximum_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, maximumBlock);
}

Tensor minimum_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, minimumBlock);
}

Tensor fmax_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, maximumBlock);
}

Tensor fmin_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, minimumBlock);
}

// ---------------------------------------------------------------------------
// atan2 / hypot
// ---------------------------------------------------------------------------

Tensor atan2_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, atan2Block);
}

Tensor& atan2_inplace_kernel(Tensor& self, const Tensor& other) {
  Tensor result = atan2_kernel(self, other);
  self.copy_(result);
  return self;
}

Tensor hypot_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp(self, other, hypotBlock);
}

Tensor& hypot_inplace_kernel(Tensor& self, const Tensor& other) {
  Tensor result = hypot_kernel(self, other);
  self.copy_(result);
  return self;
}

// ---------------------------------------------------------------------------
// addcmul / addcdiv
// ---------------------------------------------------------------------------

Tensor addcmul_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& value) {
  // self + value * tensor1 * tensor2
  const DType common = promoteTypes(
      promoteTypes(self.dtype(), tensor1.dtype()), tensor2.dtype());
  Tensor v_t = scalarToMPS(value, common, self.device());
  Tensor t1 = binaryOp(tensor1, tensor2, mulBlock);
  Tensor t1v = binaryOp(t1, v_t, mulBlock);
  return binaryOp(self, t1v, addBlock);
}

Tensor& addcmul_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& value) {
  Tensor result = addcmul_kernel(self, tensor1, tensor2, value);
  self.copy_(result);
  return self;
}

Tensor addcdiv_kernel(const Tensor& self, const Tensor& tensor1,
                      const Tensor& tensor2, const Scalar& value) {
  if (isIntegralType(tensor1.dtype(), true) &&
      isIntegralType(tensor2.dtype(), true)) {
    TP_THROW(RuntimeError, "Integer division with addcdiv is not supported");
  }
  // self + value * tensor1 / tensor2
  const DType common = promoteTypes(
      promoteTypes(self.dtype(), tensor1.dtype()), tensor2.dtype());
  Tensor v_t = scalarToMPS(value, common, self.device());
  Tensor t1v = binaryOp(tensor1, v_t, mulBlock);
  Tensor t12 = binaryOp(t1v, tensor2, divBlock);
  return binaryOp(self, t12, addBlock);
}

Tensor& addcdiv_inplace_kernel(Tensor& self, const Tensor& tensor1,
                               const Tensor& tensor2, const Scalar& value) {
  Tensor result = addcdiv_kernel(self, tensor1, tensor2, value);
  self.copy_(result);
  return self;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ArithmeticKernels) {
  m.impl("add.Tensor", add_kernel);
  m.impl("add.Scalar", add_scalar_kernel);
  m.impl("add_.Tensor", add_inplace_kernel);
  m.impl("add_.Scalar", add_scalar_inplace_kernel);

  m.impl("sub.Tensor", sub_kernel);
  m.impl("sub.Scalar", sub_scalar_kernel);
  m.impl("sub_.Tensor", sub_inplace_kernel);
  m.impl("sub_.Scalar", sub_scalar_inplace_kernel);

  m.impl("mul.Tensor", mul_kernel);
  m.impl("mul.Scalar", mul_scalar_kernel);
  m.impl("mul_.Tensor", mul_inplace_kernel);
  m.impl("mul_.Scalar", mul_scalar_inplace_kernel);

  m.impl("div.Tensor", div_kernel);
  m.impl("div.Scalar", div_scalar_kernel);
  m.impl("div_.Tensor", div_inplace_kernel);
  m.impl("div_.Scalar", div_scalar_inplace_kernel);

  m.impl("pow.Tensor_Tensor", pow_tensor_tensor_kernel);
  m.impl("pow.Tensor_Scalar", pow_scalar_kernel);
  m.impl("pow.Scalar", pow_scalar_tensor_kernel);
  m.impl("pow_.Tensor", pow_tensor_inplace_kernel);
  m.impl("pow_.Scalar", pow_scalar_inplace_kernel);

  m.impl("remainder.Tensor", remainder_tensor_kernel);
  m.impl("remainder.Scalar", remainder_scalar_kernel);
  m.impl("fmod.Tensor", fmod_tensor_kernel);
  m.impl("fmod.Scalar", fmod_scalar_kernel);

  m.impl("maximum", maximum_kernel);
  m.impl("minimum", minimum_kernel);
  m.impl("fmax", fmax_kernel);
  m.impl("fmin", fmin_kernel);

  m.impl("atan2", atan2_kernel);
  m.impl("atan2_", atan2_inplace_kernel);
  m.impl("hypot", hypot_kernel);
  m.impl("hypot_", hypot_inplace_kernel);

  m.impl("addcmul", addcmul_kernel);
  m.impl("addcmul_", addcmul_inplace_kernel);
  m.impl("addcdiv", addcdiv_kernel);
  m.impl("addcdiv_", addcdiv_inplace_kernel);
}

} // namespace mps
} // namespace tensorplay