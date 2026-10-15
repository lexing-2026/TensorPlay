#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>

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

MPSBinaryOpBlock mulBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                             MPSGraphTensor* b) {
  return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
};

MPSBinaryOpBlock addBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                               MPSGraphTensor* b) {
  return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
};

// Runs an MPSGraph matrix multiplication. `a_shape`/`b_shape` describe the
// graph inputs (1-D operands are lifted to 2-D), `out_data_shape` describes
// the output buffer (the result tensor may be 1-D, e.g. matvec).
void runMatMul(const Tensor& a, const Tensor& b,
               const std::vector<int64_t>& a_shape,
               const std::vector<int64_t>& b_shape,
               const std::vector<int64_t>& out_data_shape, Tensor& output) {
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* aShape = getMPSShape(a_shape);
    MPSShape* bShape = getMPSShape(b_shape);
    MPSShape* outShape = getMPSShape(out_data_shape);
    const MPSDataType a_dt = getMPSDataType(a.dtype());
    const MPSDataType b_dt = getMPSDataType(b.dtype());
    const MPSDataType out_dt = getMPSDataType(output.dtype());

    MPSGraphTensor* at = [graph placeholderWithShape:aShape
                                          dataType:a_dt
                                               name:@"a"];
    MPSGraphTensor* bt = [graph placeholderWithShape:bShape
                                          dataType:b_dt
                                               name:@"b"];
    MPSGraphTensor* product =
        [graph matrixMultiplicationWithPrimaryTensor:at
                                     secondaryTensor:bt
                                                name:nil];
    MPSGraphTensor* result = ([product dataType] == out_dt)
        ? product
        : [graph castTensor:product toType:out_dt name:@"cast_out"];

    id<MTLBuffer> aBuf = mtlBufferOf(a);
    id<MTLBuffer> bBuf = mtlBufferOf(b);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* aData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:aBuf
                                                shape:aShape
                                             dataType:a_dt] autorelease];
    MPSGraphTensorData* bData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:bBuf
                                                shape:bShape
                                             dataType:b_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:out_dt] autorelease];
    NSDictionary* feeds = @{at : aData, bt : bData};
    NSDictionary* results = @{result : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
}

void check_matmul_dtypes(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype()) {
    TP_THROW(RuntimeError,
             "matmul: expected both inputs to have the same dtype");
  }
}

} // namespace

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
  if (!self.device().is_mps() || !mat2.device().is_mps()) {
    TP_THROW(RuntimeError, "mm(): tensors must be on the MPS device");
  }
  if (self.dim() != 2 || mat2.dim() != 2) {
    TP_THROW(RuntimeError, "mm(): expected 2-D matrices");
  }
  if (self.size(1) != mat2.size(0)) {
    TP_THROW(RuntimeError,
             "mm(): mat1 and mat2 shapes cannot be multiplied (",
             self.size(0), "x", self.size(1), " and ", mat2.size(0), "x",
             mat2.size(1), ")");
  }
  check_matmul_dtypes(self, mat2);
  const std::vector<int64_t> out_shape = {self.size(0), mat2.size(1)};
  Tensor output = Tensor(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }
  Tensor a = materializeForGraph(self);
  Tensor b = materializeForGraph(mat2);
  runMatMul(a, b, static_cast<std::vector<int64_t>>(a.shape()),
            static_cast<std::vector<int64_t>>(b.shape()),
            static_cast<std::vector<int64_t>>(output.shape()), output);
  return output;
}

Tensor bmm_kernel(const Tensor& self, const Tensor& batch2) {
  if (!self.device().is_mps() || !batch2.device().is_mps()) {
    TP_THROW(RuntimeError, "bmm(): tensors must be on the MPS device");
  }
  if (self.dim() != 3 || batch2.dim() != 3) {
    TP_THROW(RuntimeError, "bmm(): expected 3-D tensors");
  }
  if (self.size(0) != batch2.size(0)) {
    TP_THROW(RuntimeError,
             "bmm(): batch dimensions must match");
  }
  if (self.size(2) != batch2.size(1)) {
    TP_THROW(RuntimeError, "bmm(): mat1 and mat2 shapes cannot be multiplied");
  }
  check_matmul_dtypes(self, batch2);
  const std::vector<int64_t> out_shape = {self.size(0), self.size(1),
                                           batch2.size(2)};
  Tensor output = Tensor(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }
  Tensor a = materializeForGraph(self);
  Tensor b = materializeForGraph(batch2);
  runMatMul(a, b, static_cast<std::vector<int64_t>>(a.shape()),
            static_cast<std::vector<int64_t>>(b.shape()),
            static_cast<std::vector<int64_t>>(output.shape()), output);
  return output;
}

Tensor dot_kernel(const Tensor& self_in, const Tensor& other_in) {
  if (!self_in.device().is_mps() || !other_in.device().is_mps()) {
    TP_THROW(RuntimeError, "dot(): tensors must be on the MPS device");
  }
  if (self_in.dim() != 1 || other_in.dim() != 1) {
    TP_THROW(RuntimeError, "dot(): 1D tensors expected");
  }
  if (self_in.numel() != other_in.numel()) {
    TP_THROW(RuntimeError, "dot(): inconsistent tensor size");
  }
  check_matmul_dtypes(self_in, other_in);
  Tensor output = Tensor({}, self_in.dtype(), self_in.device());
  if (self_in.numel() == 0) {
    fill_kernel(output, 0);
    return output;
  }
  Tensor a = materializeForGraph(self_in);
  Tensor b = materializeForGraph(other_in);
  const std::vector<int64_t> shape = {a.numel()};
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* mpsShape = getMPSShape(shape);
    const MPSDataType dt = getMPSDataType(a.dtype());
    MPSGraphTensor* at = [graph placeholderWithShape:mpsShape
                                           dataType:dt
                                                name:@"a"];
    MPSGraphTensor* bt = [graph placeholderWithShape:mpsShape
                                           dataType:dt
                                                name:@"b"];
    MPSGraphTensor* prod = [graph multiplicationWithPrimaryTensor:at
                                                 secondaryTensor:bt
                                                            name:nil];
    MPSGraphTensor* sum = [graph reductionSumWithTensor:prod axes:nil name:nil];
    MPSGraphTensor* reshaped =
        [graph reshapeTensor:sum withShape:@[ @1 ] name:nil];
    id<MTLBuffer> aBuf = mtlBufferOf(a);
    id<MTLBuffer> bBuf = mtlBufferOf(b);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* aData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:aBuf
                                                shape:mpsShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* bData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:bBuf
                                                shape:mpsShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:@[ @1 ]
                                             dataType:dt] autorelease];
    NSDictionary* feeds = @{at : aData, bt : bData};
    NSDictionary* results = @{reshaped : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
  if (!self.device().is_mps() || !vec.device().is_mps()) {
    TP_THROW(RuntimeError, "mv(): tensors must be on the MPS device");
  }
  if (self.dim() != 2 || vec.dim() != 1) {
    TP_THROW(RuntimeError, "mv(): expected matrix (2-D) * vector (1-D)");
  }
  if (self.size(1) != vec.size(0)) {
    TP_THROW(RuntimeError, "mv(): shapes cannot be multiplied");
  }
  check_matmul_dtypes(self, vec);
  const std::vector<int64_t> out_shape = {self.size(0)};
  Tensor output = Tensor(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }
  Tensor a = materializeForGraph(self);
  Tensor b = materializeForGraph(vec);
  const std::vector<int64_t> a_shape = {self.size(0), self.size(1)};
  const std::vector<int64_t> b_shape = {vec.size(0), 1};
  const std::vector<int64_t> out_data_shape = {self.size(0), 1};
  runMatMul(a, b, a_shape, b_shape, out_data_shape, output);
  return output;
}

Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    TP_THROW(RuntimeError, "matmul(): tensors must be on the MPS device");
  }
  check_matmul_dtypes(self, other);
  // Vector-vector -> dot.
  if (self.dim() == 1 && other.dim() == 1) {
    return dot_kernel(self, other);
  }
  // Matrix-matrix -> mm.
  if (self.dim() == 2 && other.dim() == 2) {
    return mm_kernel(self, other);
  }
  // Matrix-vector -> mv.
  if (self.dim() == 2 && other.dim() == 1) {
    return mv_kernel(self, other);
  }
  // Vector-matrix: lift the vector to a row matrix, then squeeze.
  if (self.dim() == 1 && other.dim() == 2) {
    if (self.size(0) != other.size(0)) {
      TP_THROW(RuntimeError, "matmul(): shapes cannot be multiplied");
    }
    const std::vector<int64_t> out_shape = {other.size(1)};
    Tensor output = Tensor(out_shape, self.dtype(), self.device());
    if (output.numel() == 0) {
      return output;
    }
    Tensor a = materializeForGraph(self);
    Tensor b = materializeForGraph(other);
    const std::vector<int64_t> a_shape = {1, self.size(0)};
    const std::vector<int64_t> out_data_shape = {1, other.size(1)};
    runMatMul(a, b, a_shape,
              static_cast<std::vector<int64_t>>(b.shape()),
              out_data_shape, output);
    return output;
  }
  // Batched matmul with equal batch dims (both 3-D).
  if (self.dim() == 3 && other.dim() == 3) {
    return bmm_kernel(self, other);
  }
  TP_THROW(NotImplementedError,
           "MPS matmul does not yet support this combination of input ranks");
}

Tensor addmm_kernel(const Tensor& input, const Tensor& mat1,
                    const Tensor& mat2, const Scalar& beta,
                    const Scalar& alpha) {
  Tensor prod = mm_kernel(mat1, mat2);
  // result = beta * input + alpha * prod
  Tensor beta_t = scalarToTensor(beta, input.dtype(), input.device());
  Tensor alpha_t = scalarToTensor(alpha, prod.dtype(), prod.device());
  Tensor scaled_input = binaryOp(input, beta_t, mulBlock);
  Tensor scaled_prod = binaryOp(prod, alpha_t, mulBlock);
  return binaryOp(scaled_input, scaled_prod, addBlock);
}

TENSORPLAY_LIBRARY_IMPL(MPS, BlasKernels) {
  m.impl("mm", mm_kernel);
  m.impl("matmul", matmul_kernel);
  m.impl("bmm", bmm_kernel);
  m.impl("addmm", addmm_kernel);
  m.impl("mv", mv_kernel);
  m.impl("dot", dot_kernel);
}

} // namespace mps
} // namespace tensorplay