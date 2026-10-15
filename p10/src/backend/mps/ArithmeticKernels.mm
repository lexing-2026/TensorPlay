#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

namespace {

// Maps a storage dtype to the Metal data type used by MPSGraph.  Double,
// complex and the 8-bit float formats are not representable in MPSGraph and
// are rejected explicitly rather than silently truncated.
MPSDataType getMPSDataType(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return MPSDataTypeFloat32;
    case DType::Float16:
      return MPSDataTypeFloat16;
    case DType::BFloat16:
      return MPSDataTypeBFloat16;
    case DType::Int32:
      return MPSDataTypeInt32;
    case DType::Int64:
      return MPSDataTypeInt64;
    case DType::Int16:
      return MPSDataTypeInt16;
    case DType::Int8:
      return MPSDataTypeInt8;
    case DType::UInt8:
      return MPSDataTypeUInt8;
    case DType::Bool:
      return MPSDataTypeBool;
    default:
      TP_THROW(NotImplementedError,
               "MPS add does not support dtype ", toString(dtype));
  }
}

MPSShape* getMPSShape(const std::vector<int64_t>& sizes) {
  NSMutableArray<NSNumber*>* shape =
      [NSMutableArray arrayWithCapacity:sizes.size()];
  for (int64_t s : sizes) {
    [shape addObject:@(s)];
  }
  return shape;
}

// Returns a contiguous, zero-storage-offset view of `t` so its contents can
// be wrapped directly in an MPSGraphTensorData (which has no offset concept).
Tensor materializeForGraph(const Tensor& t) {
  if (t.is_contiguous() && t.storage_offset() == 0) {
    return t;
  }
  Tensor c = t.contiguous();
  if (c.storage_offset() != 0) {
    Tensor copy = Tensor(c.sizes(), c.dtype(), c.device());
    copy.copy_(c);
    c = copy;
  }
  return c;
}

id<MTLBuffer> mtlBufferOf(const Tensor& t) {
  void* buf = getMTLBufferFromPointer(t.data_ptr());
  if (!buf) {
    TP_THROW(RuntimeError,
             "MPS tensor storage is not managed by the MPS allocator");
  }
  return (__bridge id<MTLBuffer>)buf;
}

// Runs an MPSGraph that computes `self + alpha * other` into `output`. All
// three tensors must be contiguous, offset-0, same-dtype and same-shape.
void runAddGraph(const Tensor& self, const Tensor& other, const Scalar& alpha,
                 Tensor& output) {
  @autoreleasepool {
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(self.shape());
    // MPSGraphTensorData needs a non-empty shape; a 0-d tensor (numel 1) is
    // represented as a single-element shape.
    const std::vector<int64_t> graph_sizes =
        sizes.empty() ? std::vector<int64_t>{1} : sizes;
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* shape = getMPSShape(graph_sizes);
    const MPSDataType dt = getMPSDataType(self.dtype());

    MPSGraphTensor* x = [graph placeholderWithShape:shape dataType:dt name:@"x"];
    MPSGraphTensor* y = [graph placeholderWithShape:shape dataType:dt name:@"y"];
    MPSGraphTensor* sum = nil;
    const double alphaVal = alpha.to<double>();
    if (alphaVal == 1.0) {
      sum = [graph additionWithPrimaryTensor:x secondaryTensor:y name:@"add"];
    } else {
      MPSGraphTensor* alphaConst = [graph constant:alphaVal dataType:dt];
      MPSGraphTensor* yScaled =
          [graph multiplicationWithPrimaryTensor:y
                                  secondaryTensor:alphaConst
                                             name:@"scale"];
      sum = [graph additionWithPrimaryTensor:x
                             secondaryTensor:yScaled
                                         name:@"add"];
    }

    id<MTLBuffer> xBuf = mtlBufferOf(self);
    id<MTLBuffer> yBuf = mtlBufferOf(other);
    id<MTLBuffer> outBuf = mtlBufferOf(output);

    MPSGraphTensorData* xData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:xBuf
                                                 shape:shape
                                              dataType:dt] autorelease];
    MPSGraphTensorData* yData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:yBuf
                                                 shape:shape
                                              dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:shape
                                             dataType:dt] autorelease];

    NSDictionary* feeds = @{x : xData, y : yData};
    NSDictionary* results = @{sum : outData};

    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
}

Tensor add_impl(const Tensor& self, const Tensor& other,
                const Scalar& alpha) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS add requires both inputs on the MPS device");
  }
  if (self.dtype() != other.dtype()) {
    TP_THROW(NotImplementedError,
             "MPS add does not yet support mixed-dtype inputs");
  }
  const std::vector<int64_t> out_shape = broadcast_shapes(
      static_cast<std::vector<int64_t>>(self.shape()),
      static_cast<std::vector<int64_t>>(other.shape()));
  Tensor self_b = self.shape() == out_shape
      ? self : self.expand(out_shape);
  Tensor other_b = other.shape() == out_shape
      ? other : other.expand(out_shape);
  Tensor self_c = materializeForGraph(self_b);
  Tensor other_c = materializeForGraph(other_b);

  Tensor output = Tensor(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }
  runAddGraph(self_c, other_c, alpha, output);
  return output;
}

} // namespace

Tensor add_kernel(const Tensor& self, const Tensor& other,
                  const Scalar& alpha) {
  return add_impl(self, other, alpha);
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other,
                         const Scalar& alpha) {
  // Materialize the scalar as a 1-element MPS tensor on the same device and
  // reuse the tensor path, keeping the graph construction uniform.
  Tensor other_t = Tensor({1}, self.dtype(), self.device());
  fill_kernel(other_t, other);
  return add_impl(self, other_t, alpha);
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other,
                           const Scalar& alpha) {
  Tensor result = add_impl(self, other, alpha);
  self.copy_(result);
  return self;
}

Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other,
                                  const Scalar& alpha) {
  Tensor result = add_scalar_kernel(self, other, alpha);
  self.copy_(result);
  return self;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ArithmeticKernels) {
  m.impl("add.Tensor", add_kernel);
  m.impl("add_.Tensor", add_inplace_kernel);
  m.impl("add.Scalar", add_scalar_kernel);
  m.impl("add_.Scalar", add_scalar_inplace_kernel);
}

} // namespace mps
} // namespace tensorplay