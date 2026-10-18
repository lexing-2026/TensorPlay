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

Tensor scalarToTensor(const Scalar& s, DType dtype, const Device& device) {
  Tensor t = Tensor({1}, dtype, device);
  fill_kernel(t, s);
  return t;
}

MPSUnaryOpBlock absBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g absoluteWithTensor:x name:nil];
};

MPSUnaryOpBlock negBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g negativeWithTensor:x name:nil];
};

MPSUnaryOpBlock squareBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g squareWithTensor:x name:nil];
};

MPSUnaryOpBlock signBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g signWithTensor:x name:nil];
};

MPSUnaryOpBlock expBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g exponentWithTensor:x name:nil];
};

MPSUnaryOpBlock logBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g logarithmWithTensor:x name:nil];
};

MPSUnaryOpBlock sqrtBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g squareRootWithTensor:x name:nil];
};

MPSUnaryOpBlock rsqrtBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
#ifdef __MAC_15_0
  if (is_macos_at_least(MacOSVersion::MACOS_15_0)) {
    return [g reciprocalSquareRootWithTensor:x name:nil];
  } else
#endif // __MAC_15_0
  {
    return [g reverseSquareRootWithTensor:x name:nil];
  }
};

MPSUnaryOpBlock reciprocalBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g reciprocalWithTensor:x name:nil];
};

MPSUnaryOpBlock sinBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g sinWithTensor:x name:nil];
};

MPSUnaryOpBlock cosBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g cosWithTensor:x name:nil];
};

MPSUnaryOpBlock tanBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g tanWithTensor:x name:nil];
};

MPSUnaryOpBlock asinBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g asinWithTensor:x name:nil];
};

MPSUnaryOpBlock acosBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g acosWithTensor:x name:nil];
};

MPSUnaryOpBlock atanBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g atanWithTensor:x name:nil];
};

MPSUnaryOpBlock sinhBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g sinhWithTensor:x name:nil];
};

MPSUnaryOpBlock coshBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g coshWithTensor:x name:nil];
};

MPSUnaryOpBlock tanhBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g tanhWithTensor:x name:nil];
};

MPSUnaryOpBlock asinhBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g asinhWithTensor:x name:nil];
};

MPSUnaryOpBlock acoshBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g acoshWithTensor:x name:nil];
};

MPSUnaryOpBlock atanhBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g atanhWithTensor:x name:nil];
};

MPSUnaryOpBlock erfBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g erfWithTensor:x name:nil];
};

MPSUnaryOpBlock erfcBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  // erfc(x) = 1 - erf(x)
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  return [g subtractionWithPrimaryTensor:one
                          secondaryTensor:[g erfWithTensor:x name:nil]
                                     name:nil];
};

MPSUnaryOpBlock floorBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g floorWithTensor:x name:nil];
};

MPSUnaryOpBlock ceilBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g ceilWithTensor:x name:nil];
};

MPSUnaryOpBlock roundBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g roundWithTensor:x name:nil];
};

MPSUnaryOpBlock truncBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g truncateWithTensor:x name:nil];
};

MPSUnaryOpBlock sigmoidBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  // sigmoid(x) = 1 / (1 + exp(-x))
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  MPSGraphTensor* negX = [g negativeWithTensor:x name:nil];
  MPSGraphTensor* expNeg = [g exponentWithTensor:negX name:nil];
  MPSGraphTensor* denom = [g additionWithPrimaryTensor:one
                                      secondaryTensor:expNeg
                                                 name:nil];
  return [g divisionWithPrimaryTensor:one secondaryTensor:denom name:nil];
};

MPSUnaryOpBlock reluBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g reLUWithTensor:x name:nil];
};

MPSUnaryOpBlock siluBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g multiplicationWithPrimaryTensor:x
                              secondaryTensor:sigmoidBlock(g, x)
                                         name:nil];
};

MPSUnaryOpBlock hardswishBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  // hardswish(x) = x * clamp(x/6 + 0.5, 0, 1)
  MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  MPSGraphTensor* sixth = [g constantWithScalar:1.0 / 6.0 dataType:x.dataType];
  MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:x.dataType];
  MPSGraphTensor* scaled = [g multiplicationWithPrimaryTensor:x
                                              secondaryTensor:sixth
                                                         name:nil];
  MPSGraphTensor* shifted =
      [g additionWithPrimaryTensor:scaled secondaryTensor:half name:nil];
  MPSGraphTensor* hs = [g clampWithTensor:shifted
                           minValueTensor:zero
                           maxValueTensor:one
                                     name:nil];
  return [g multiplicationWithPrimaryTensor:x secondaryTensor:hs name:nil];
};

MPSUnaryOpBlock mishBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  MPSGraphTensor* exp = [g exponentWithTensor:x name:nil];
  MPSGraphTensor* exp1 = [g additionWithPrimaryTensor:exp
                                     secondaryTensor:one
                                                name:nil];
  MPSGraphTensor* softplus = [g logarithmWithTensor:exp1 name:nil];
  MPSGraphTensor* tanh = [g tanhWithTensor:softplus name:nil];
  return [g multiplicationWithPrimaryTensor:x secondaryTensor:tanh name:nil];
};

MPSUnaryOpBlock tanhshrinkBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g subtractionWithPrimaryTensor:x
                           secondaryTensor:[g tanhWithTensor:x name:nil]
                                      name:nil];
};

MPSUnaryOpBlock expm1Block = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  return [g subtractionWithPrimaryTensor:[g exponentWithTensor:x name:nil]
                         secondaryTensor:one
                                    name:nil];
};

// log(x) / log(base)
MPSGraphTensor* logWithBaseBlock(MPSGraph* g, MPSGraphTensor* x, double base) {
  MPSGraphTensor* logx = [g logarithmWithTensor:x name:nil];
  MPSGraphTensor* logbase = [g constantWithScalar:base dataType:x.dataType];
  return [g divisionWithPrimaryTensor:logx secondaryTensor:logbase name:nil];
}

MPSUnaryOpBlock log1pBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
  MPSGraphTensor* added =
      [g additionWithPrimaryTensor:x secondaryTensor:one name:nil];
  return [g logarithmWithTensor:added name:nil];
};

MPSUnaryOpBlock fracBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  MPSGraphTensor* trunc = [g truncateWithTensor:x name:nil];
  return [g subtractionWithPrimaryTensor:x secondaryTensor:trunc name:nil];
};

MPSUnaryOpBlock bitwiseNotBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  // ~x == -1 - x for two's-complement integer types.
  MPSGraphTensor* minusOne = [g constantWithScalar:-1.0 dataType:x.dataType];
  return [g subtractionWithPrimaryTensor:minusOne secondaryTensor:x name:nil];
};

MPSBinaryOpBlock bitwiseAndBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                   MPSGraphTensor* b) {
  if (a.dataType == MPSDataTypeBool) {
    // Bool is logical rather than bitwise; MPSGraph's bitwiseAnd rejects it.
    return [g logicalANDWithPrimaryTensor:a secondaryTensor:b name:nil];
  }
  return [g bitwiseAndWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSUnaryOpBlock isfiniteBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
  return [g isFiniteWithTensor:x name:nil];
};

// --- comparison blocks ---

MPSBinaryOpBlock eqBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g equalWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock neBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g notEqualWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock ltBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g lessThanWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock leBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock gtBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g greaterThanWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock geBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                           MPSGraphTensor* b) {
  return [g greaterThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
};

// The bundled shader when the operand suits it, the graph when it does not. A
// call for an operation the metallib has no function for, or for an operand the
// shader cannot address, is answered by the graph without a second check here:
// `unaryShader` reports that it took nothing.
Tensor unaryShaderOrOp(const std::string& op, const Tensor& self,
                       MPSUnaryOpBlock block) {
  if (auto result = unaryShader(op, self)) {
    return *result;
  }
  return unaryOp(op, self, block);
}

} // namespace

// ---------------------------------------------------------------------------
// Unary pointwise ops
// ---------------------------------------------------------------------------

Tensor abs_kernel(const Tensor& self) {
  return unaryShaderOrOp("abs", self, absBlock);
}
Tensor neg_kernel(const Tensor& self) {
  return unaryShaderOrOp("neg", self, negBlock);
}
Tensor square_kernel(const Tensor& self) {
  return unaryShaderOrOp("square", self, squareBlock);
}
Tensor sign_kernel(const Tensor& self) {
  return unaryShaderOrOp("sign", self, signBlock);
}
Tensor exp_kernel(const Tensor& self) {
  return unaryShaderOrOp("exp", self, expBlock);
}
Tensor expm1_kernel(const Tensor& self) {
  return unaryOp("expm1", self, expm1Block);
}
Tensor log_kernel(const Tensor& self) {
  return unaryShaderOrOp("log", self, logBlock);
}
Tensor log10_kernel(const Tensor& self) {
  return unaryOp("log10", self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    return logWithBaseBlock(g, x, 10.0);
  });
}
Tensor log2_kernel(const Tensor& self) {
  return unaryOp("log2", self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    return logWithBaseBlock(g, x, 2.0);
  });
}
Tensor log1p_kernel(const Tensor& self) {
  return unaryOp("log1p", self, log1pBlock);
}
Tensor sqrt_kernel(const Tensor& self) {
  return unaryShaderOrOp("sqrt", self, sqrtBlock);
}
Tensor rsqrt_kernel(const Tensor& self) {
  return unaryShaderOrOp("rsqrt", self, rsqrtBlock);
}
Tensor reciprocal_kernel(const Tensor& self) {
  return unaryShaderOrOp("reciprocal", self, reciprocalBlock);
}
Tensor sin_kernel(const Tensor& self) {
  return unaryShaderOrOp("sin", self, sinBlock);
}
Tensor cos_kernel(const Tensor& self) {
  return unaryShaderOrOp("cos", self, cosBlock);
}
Tensor tan_kernel(const Tensor& self) { return unaryOp("tan", self, tanBlock); }
Tensor asin_kernel(const Tensor& self) {
  return unaryOp("asin", self, asinBlock);
}
Tensor acos_kernel(const Tensor& self) {
  return unaryOp("acos", self, acosBlock);
}
Tensor atan_kernel(const Tensor& self) {
  return unaryOp("atan", self, atanBlock);
}
Tensor sinh_kernel(const Tensor& self) {
  return unaryOp("sinh", self, sinhBlock);
}
Tensor cosh_kernel(const Tensor& self) {
  return unaryOp("cosh", self, coshBlock);
}
Tensor tanh_kernel(const Tensor& self) {
  return unaryShaderOrOp("tanh", self, tanhBlock);
}
Tensor asinh_kernel(const Tensor& self) {
  return unaryOp("asinh", self, asinhBlock);
}
Tensor acosh_kernel(const Tensor& self) {
  return unaryOp("acosh", self, acoshBlock);
}
Tensor atanh_kernel(const Tensor& self) {
  return unaryOp("atanh", self, atanhBlock);
}
Tensor erf_kernel(const Tensor& self) {
  return unaryShaderOrOp("erf", self, erfBlock);
}
Tensor erfc_kernel(const Tensor& self) {
  return unaryOp("erfc", self, erfcBlock);
}
Tensor floor_kernel(const Tensor& self) {
  return unaryShaderOrOp("floor", self, floorBlock);
}
Tensor ceil_kernel(const Tensor& self) {
  return unaryShaderOrOp("ceil", self, ceilBlock);
}
Tensor round_kernel(const Tensor& self) {
  return unaryShaderOrOp("round", self, roundBlock);
}
Tensor trunc_kernel(const Tensor& self) {
  return unaryOp("trunc", self, truncBlock);
}
Tensor frac_kernel(const Tensor& self) {
  return unaryOp("frac", self, fracBlock);
}
Tensor bitwise_not_kernel(const Tensor& self) {
  if (self.dtype() == DType::Bool) {
    TP_THROW(NotImplementedError,
             "bitwise_not on bool tensors is not implemented on MPS yet");
  }
  return unaryOp("bitwise_not", self, bitwiseNotBlock);
}

Tensor bitwise_and_kernel(const Tensor& self, const Tensor& other) {
  return binaryOp("bitwise_and", self, other, bitwiseAndBlock);
}

Tensor isfinite_kernel(const Tensor& self) {
  return unaryOp("isfinite", self, isfiniteBlock, DType::Bool);
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

Tensor sigmoid_kernel(const Tensor& self) {
  return unaryOp("sigmoid", self, sigmoidBlock);
}

Tensor relu_kernel(const Tensor& self) {
  return unaryShaderOrOp("relu", self, reluBlock);
}

Tensor& relu_inplace_kernel(Tensor& self) {
  Tensor result = relu_kernel(self);
  self.copy_(result);
  return self;
}

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
  return unaryOp(graphKey("gelu", {approximate}), self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    if (approximate == "tanh") {
      // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
      const double kSqrt2OverPi = 0.7978845608028654;
      MPSGraphTensor* x3 = [g multiplicationWithPrimaryTensor:[g squareWithTensor:x name:nil]
                                             secondaryTensor:x
                                                        name:nil];
      MPSGraphTensor* cubic = [g multiplicationWithPrimaryTensor:[g constantWithScalar:0.044715
                                                                             dataType:x.dataType]
                                                secondaryTensor:x3
                                                           name:nil];
      MPSGraphTensor* poly =
          [g additionWithPrimaryTensor:x secondaryTensor:cubic name:nil];
      poly = [g multiplicationWithPrimaryTensor:poly
                               secondaryTensor:[g constantWithScalar:kSqrt2OverPi
                                                            dataType:x.dataType]
                                          name:nil];
      MPSGraphTensor* tanh = [g tanhWithTensor:poly name:nil];
      MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
      MPSGraphTensor* inner =
          [g additionWithPrimaryTensor:one secondaryTensor:tanh name:nil];
      MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:x.dataType];
      return [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:half
                                                                    secondaryTensor:x
                                                                               name:nil]
                                secondaryTensor:inner
                                           name:nil];
    }
    // Exact: 0.5*x*(1 + erf(x/sqrt(2)))
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:x.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* sqrt2 = [g constantWithScalar:1.4142135623730951
                                         dataType:x.dataType];
    MPSGraphTensor* xOverSqrt2 = [g divisionWithPrimaryTensor:x
                                                  secondaryTensor:sqrt2
                                                             name:nil];
    MPSGraphTensor* erf = [g erfWithTensor:xOverSqrt2 name:nil];
    MPSGraphTensor* inner =
        [g additionWithPrimaryTensor:one secondaryTensor:erf name:nil];
    return [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:half
                                                                  secondaryTensor:x
                                                                             name:nil]
                              secondaryTensor:inner
                                         name:nil];
  });
}

Tensor silu_kernel(const Tensor& self) {
  return unaryOp("silu", self, siluBlock);
}

Tensor hardtanh_kernel_impl(const Tensor& self, const Scalar& min_val,
                            const Scalar& max_val) {
  return unaryOp(
      graphKey("hardtanh", {scalarKey(min_val), scalarKey(max_val)}),
      self, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* lo = [g constantWithScalar:min_val.to<double>()
                                     dataType:x.dataType];
    MPSGraphTensor* hi = [g constantWithScalar:max_val.to<double>()
                                     dataType:x.dataType];
    return [g clampWithTensor:x minValueTensor:lo maxValueTensor:hi name:nil];
  });
}

Tensor hardsigmoid_kernel_impl(const Tensor& self) {
  return unaryOp("hardsigmoid", self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* sixth = [g constantWithScalar:1.0 / 6.0 dataType:x.dataType];
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:x.dataType];
    MPSGraphTensor* scaled = [g multiplicationWithPrimaryTensor:x
                                                secondaryTensor:sixth
                                                           name:nil];
    MPSGraphTensor* shifted =
        [g additionWithPrimaryTensor:scaled secondaryTensor:half name:nil];
    return [g clampWithTensor:shifted minValueTensor:zero maxValueTensor:one name:nil];
  });
}

Tensor hardswish_kernel_impl(const Tensor& self) {
  return unaryOp("hardswish", self, hardswishBlock);
}

Tensor leaky_relu_kernel_impl(const Tensor& self,
                              const Scalar& negative_slope) {
  return unaryOp(
      graphKey("leaky_relu", {scalarKey(negative_slope)}), self,
      ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    // leaky_relu(x) = x > 0 ? x : alpha * x
    MPSGraphTensor* alpha = [g constantWithScalar:negative_slope.to<double>()
                                          dataType:x.dataType];
    MPSGraphTensor* scaled = [g multiplicationWithPrimaryTensor:x
                                                secondaryTensor:alpha
                                                           name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
    MPSGraphTensor* pred = [g greaterThanWithPrimaryTensor:x
                                          secondaryTensor:zero
                                                     name:nil];
    return [g selectWithPredicateTensor:pred
                    truePredicateTensor:x
                   falsePredicateTensor:scaled
                                   name:nil];
  });
}

Tensor elu_kernel_impl(const Tensor& self, const Scalar& alpha,
                       const Scalar& scale, const Scalar& input_scale) {
  return unaryOp(
      graphKey("elu", {scalarKey(alpha), scalarKey(scale),
                       scalarKey(input_scale)}), self,
      ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    // elu(x) = x>0 ? x : alpha*(exp(x)-1), with scale/input_scale folded in.
    MPSGraphTensor* alphaC = [g constantWithScalar:alpha.to<double>()
                                            dataType:x.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* exp = [g exponentWithTensor:x name:nil];
    MPSGraphTensor* expm1 =
        [g subtractionWithPrimaryTensor:exp secondaryTensor:one name:nil];
    MPSGraphTensor* negPart = [g multiplicationWithPrimaryTensor:alphaC
                                      secondaryTensor:expm1
                                                 name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:x.dataType];
    MPSGraphTensor* pred = [g greaterThanWithPrimaryTensor:x
                                          secondaryTensor:zero
                                                     name:nil];
    MPSGraphTensor* scaleC = [g constantWithScalar:scale.to<double>()
                                     dataType:x.dataType];
    MPSGraphTensor* inputScaleC = [g constantWithScalar:input_scale.to<double>()
                                            dataType:x.dataType];
    MPSGraphTensor* xScaled = [g multiplicationWithPrimaryTensor:x
                                                 secondaryTensor:inputScaleC
                                                            name:nil];
    MPSGraphTensor* result = [g selectWithPredicateTensor:pred
                                      truePredicateTensor:xScaled
                                     falsePredicateTensor:negPart
                                                     name:nil];
    return [g multiplicationWithPrimaryTensor:result
                              secondaryTensor:scaleC
                                         name:nil];
  });
}

Tensor mish_kernel_impl(const Tensor& self) {
  return unaryOp("mish", self, mishBlock);
}

Tensor tanhshrink_kernel_impl(const Tensor& self) {
  return unaryOp("tanhshrink", self, tanhshrinkBlock);
}

Tensor softplus_kernel_impl(const Tensor& self, const Scalar& beta,
                              const Scalar& threshold) {
  return unaryOp(
      graphKey("softplus", {scalarKey(beta), scalarKey(threshold)}), self,
      ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    // softplus(x) = (1/beta)*log(1+exp(beta*x)) when beta*x <= threshold,
    // else x.
    const double betaVal = beta.to<double>();
    MPSGraphTensor* betaC = [g constantWithScalar:betaVal dataType:x.dataType];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* bx = [g multiplicationWithPrimaryTensor:x
                                            secondaryTensor:betaC
                                                       name:nil];
    MPSGraphTensor* exp = [g exponentWithTensor:bx name:nil];
    MPSGraphTensor* exp1 =
        [g additionWithPrimaryTensor:exp secondaryTensor:one name:nil];
    MPSGraphTensor* log = [g logarithmWithTensor:exp1 name:nil];
    MPSGraphTensor* invBeta = [g reciprocalWithTensor:betaC name:nil];
    MPSGraphTensor* soft = [g multiplicationWithPrimaryTensor:log
                                             secondaryTensor:invBeta
                                                        name:nil];
    MPSGraphTensor* threshC = [g constantWithScalar:threshold.to<double>()
                                          dataType:x.dataType];
    MPSGraphTensor* pred = [g greaterThanWithPrimaryTensor:bx
                                          secondaryTensor:threshC
                                                     name:nil];
    return [g selectWithPredicateTensor:pred
                    truePredicateTensor:x
                   falsePredicateTensor:soft
                                   name:nil];
  });
}

// ---------------------------------------------------------------------------
// Comparison ops
// ---------------------------------------------------------------------------

// The shader answers a call over two operands that already share a shape; the
// graph is reached by every other call, including the ones that need an
// expansion or a promotion first.
#define DEFINE_MPS_CMP_TENSOR_KERNEL(NAME, BLOCK)                        \
  Tensor NAME##_tensor_kernel(const Tensor& self, const Tensor& other) { \
    if (auto result = compareShader(#NAME, self, other)) {               \
      return *result;                                                    \
    }                                                                    \
    return binaryOp(#NAME, self, other, BLOCK, std::nullopt,             \
                    std::nullopt, DType::Bool);                          \
  }

DEFINE_MPS_CMP_TENSOR_KERNEL(eq, eqBlock)
DEFINE_MPS_CMP_TENSOR_KERNEL(ne, neBlock)
DEFINE_MPS_CMP_TENSOR_KERNEL(lt, ltBlock)
DEFINE_MPS_CMP_TENSOR_KERNEL(le, leBlock)
DEFINE_MPS_CMP_TENSOR_KERNEL(gt, gtBlock)
DEFINE_MPS_CMP_TENSOR_KERNEL(ge, geBlock)

#undef DEFINE_MPS_CMP_TENSOR_KERNEL

#define DEFINE_MPS_CMP_SCALAR_KERNEL(NAME, TENSOR_FN)                \
  Tensor NAME##_scalar_kernel(const Tensor& self, const Scalar& other) { \
    const DType common = result_type(other, self.dtype());           \
    Tensor other_t = scalarToTensor(other, common, self.device());   \
    return TENSOR_FN(self, other_t);                                 \
  }

DEFINE_MPS_CMP_SCALAR_KERNEL(eq, eq_tensor_kernel)
DEFINE_MPS_CMP_SCALAR_KERNEL(ne, ne_tensor_kernel)
DEFINE_MPS_CMP_SCALAR_KERNEL(lt, lt_tensor_kernel)
DEFINE_MPS_CMP_SCALAR_KERNEL(le, le_tensor_kernel)
DEFINE_MPS_CMP_SCALAR_KERNEL(gt, gt_tensor_kernel)
DEFINE_MPS_CMP_SCALAR_KERNEL(ge, ge_tensor_kernel)

#undef DEFINE_MPS_CMP_SCALAR_KERNEL

// ---------------------------------------------------------------------------
// where / clamp
// ---------------------------------------------------------------------------

Tensor where_kernel(const Tensor& condition, const Tensor& self,
                    const Tensor& other) {
  return whereOp(condition, self, other);
}

Tensor clamp_kernel(const Tensor& self, const std::optional<Scalar>& min,
                    const std::optional<Scalar>& max) {
  if (min.has_value() || max.has_value()) {
    if (auto result = clampShader(self, min, max)) {
      return *result;
    }
  }
  return unaryOp(
      graphKey("clamp", {min ? scalarKey(*min) : std::string("none"),
                         max ? scalarKey(*max) : std::string("none")}), self,
      ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    if (min.has_value() && max.has_value()) {
      MPSGraphTensor* lo = [g constantWithScalar:min->to<double>()
                                         dataType:x.dataType];
      MPSGraphTensor* hi = [g constantWithScalar:max->to<double>()
                                       dataType:x.dataType];
      return [g clampWithTensor:x minValueTensor:lo maxValueTensor:hi name:nil];
    }
    if (min.has_value()) {
      MPSGraphTensor* lo = [g constantWithScalar:min->to<double>()
                                     dataType:x.dataType];
      return [g maximumWithPrimaryTensor:x secondaryTensor:lo name:nil];
    }
    if (max.has_value()) {
      MPSGraphTensor* hi = [g constantWithScalar:max->to<double>()
                                     dataType:x.dataType];
      return [g minimumWithPrimaryTensor:x secondaryTensor:hi name:nil];
    }
    return x;
  });
}

Tensor clamp_min_kernel(const Tensor& self, const Scalar& min) {
  if (auto result = clampShader(self, min, std::nullopt)) {
    return *result;
  }
  return unaryOp(graphKey("clamp_min", {scalarKey(min)}), self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* lo = [g constantWithScalar:min.to<double>()
                                 dataType:x.dataType];
    return [g maximumWithPrimaryTensor:x secondaryTensor:lo name:nil];
  });
}

Tensor clamp_max_kernel(const Tensor& self, const Scalar& max) {
  if (auto result = clampShader(self, std::nullopt, max)) {
    return *result;
  }
  return unaryOp(graphKey("clamp_max", {scalarKey(max)}), self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* hi = [g constantWithScalar:max.to<double>()
                                     dataType:x.dataType];
    return [g minimumWithPrimaryTensor:x secondaryTensor:hi name:nil];
  });
}

Tensor& clamp_min__kernel(Tensor& self, const Scalar& min) {
  Tensor result = clamp_min_kernel(self, min);
  self.copy_(result);
  return self;
}

Tensor& clamp_max__kernel(Tensor& self, const Scalar& max) {
  Tensor result = clamp_max_kernel(self, max);
  self.copy_(result);
  return self;
}

Tensor& clamp__kernel(Tensor& self, const std::optional<Scalar>& min,
                       const std::optional<Scalar>& max) {
  Tensor result = clamp_kernel(self, min, max);
  self.copy_(result);
  return self;
}

// ---------------------------------------------------------------------------
// softmax / log_softmax
// ---------------------------------------------------------------------------

DType resolve_softmax_dtype(DType dtype, DType self_dtype) {
  DType out_dtype = (dtype != DType::Undefined) ? dtype : self_dtype;
  if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
  return out_dtype;
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
  const DType out_dtype = resolve_softmax_dtype(dtype, self.dtype());
  if (dim < 0) dim += self.dim();
  return unaryOp(graphKey("softmax", {std::to_string(dim)}), self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    MPSDataType dt = getMPSDataType(out_dtype);
    MPSGraphTensor* xc = ([x dataType] == dt)
        ? x
        : [g castTensor:x toType:dt name:@"cast_in"];
    return [g softMaxWithTensor:xc axis:(NSInteger)dim name:nil];
  }, out_dtype);
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype) {
  const DType out_dtype = resolve_softmax_dtype(dtype, self.dtype());
  if (dim < 0) dim += self.dim();
  return unaryOp(graphKey("log_softmax", {std::to_string(dim)}), self,
                 ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* x) {
    // log_softmax(x) = x - log(sum(exp(x), axis))
    MPSDataType dt = getMPSDataType(out_dtype);
    MPSGraphTensor* xc = ([x dataType] == dt)
        ? x
        : [g castTensor:x toType:dt name:@"cast_in"];
    MPSGraphTensor* exp = [g exponentWithTensor:xc name:nil];
    MPSGraphTensor* sum = [g reductionSumWithTensor:exp axis:(NSInteger)dim name:nil];
    // Restore the reduced axis as a size-1 dimension so the subtraction
    // broadcasts along it.
    NSMutableArray<NSNumber*>* keepdimShape = [x.shape mutableCopy];
    keepdimShape[dim] = @1;
    MPSGraphTensor* sumKeep =
        [g reshapeTensor:sum withShape:keepdimShape name:nil];
    MPSGraphTensor* logsum = [g logarithmWithTensor:sumKeep name:nil];
    return [g subtractionWithPrimaryTensor:xc secondaryTensor:logsum name:nil];
  }, out_dtype);
}

TENSORPLAY_LIBRARY_IMPL(MPS, ElementwiseKernels) {
  m.impl("abs", abs_kernel);
  m.impl("neg", neg_kernel);
  m.impl("square", square_kernel);
  m.impl("sign", sign_kernel);
  m.impl("exp", exp_kernel);
  m.impl("expm1", expm1_kernel);
  m.impl("log", log_kernel);
  m.impl("log10", log10_kernel);
  m.impl("log2", log2_kernel);
  m.impl("log1p", log1p_kernel);
  m.impl("sqrt", sqrt_kernel);
  m.impl("rsqrt", rsqrt_kernel);
  m.impl("reciprocal", reciprocal_kernel);
  m.impl("sin", sin_kernel);
  m.impl("cos", cos_kernel);
  m.impl("tan", tan_kernel);
  m.impl("asin", asin_kernel);
  m.impl("acos", acos_kernel);
  m.impl("atan", atan_kernel);
  m.impl("sinh", sinh_kernel);
  m.impl("cosh", cosh_kernel);
  m.impl("tanh", tanh_kernel);
  m.impl("asinh", asinh_kernel);
  m.impl("acosh", acosh_kernel);
  m.impl("atanh", atanh_kernel);
  m.impl("erf", erf_kernel);
  m.impl("erfc", erfc_kernel);
  m.impl("floor", floor_kernel);
  m.impl("ceil", ceil_kernel);
  m.impl("round", round_kernel);
  m.impl("trunc", trunc_kernel);
  m.impl("frac", frac_kernel);
  m.impl("bitwise_not", bitwise_not_kernel);
  m.impl("bitwise_and.Tensor", bitwise_and_kernel);
  m.impl("isfinite", isfinite_kernel);

  m.impl("sigmoid", sigmoid_kernel);
  m.impl("relu", relu_kernel);
  m.impl("relu_", relu_inplace_kernel);
  m.impl("gelu", gelu_kernel);
  m.impl("silu", silu_kernel);
  m.impl("hardtanh", hardtanh_kernel_impl);
  m.impl("hardsigmoid", hardsigmoid_kernel_impl);
  m.impl("hardswish", hardswish_kernel_impl);
  m.impl("leaky_relu", leaky_relu_kernel_impl);
  m.impl("elu", elu_kernel_impl);
  m.impl("mish", mish_kernel_impl);
  m.impl("tanhshrink", tanhshrink_kernel_impl);
  m.impl("softplus", softplus_kernel_impl);

  m.impl("eq.Tensor", eq_tensor_kernel);
  m.impl("eq.Scalar", eq_scalar_kernel);
  m.impl("ne.Tensor", ne_tensor_kernel);
  m.impl("ne.Scalar", ne_scalar_kernel);
  m.impl("lt.Tensor", lt_tensor_kernel);
  m.impl("lt.Scalar", lt_scalar_kernel);
  m.impl("le.Tensor", le_tensor_kernel);
  m.impl("le.Scalar", le_scalar_kernel);
  m.impl("gt.Tensor", gt_tensor_kernel);
  m.impl("gt.Scalar", gt_scalar_kernel);
  m.impl("ge.Tensor", ge_tensor_kernel);
  m.impl("ge.Scalar", ge_scalar_kernel);

  m.impl("where.self", where_kernel);
  m.impl("clamp", clamp_kernel);
  m.impl("clamp_", clamp__kernel);
  m.impl("clamp_min", clamp_min_kernel);
  m.impl("clamp_max", clamp_max_kernel);
  m.impl("clamp_min_", clamp_min__kernel);
  m.impl("clamp_max_", clamp_max__kernel);

  m.impl("softmax", softmax_kernel);
  m.impl("log_softmax", log_softmax_kernel);
}

} // namespace mps
} // namespace tensorplay