#include "OperationUtils.h"

#include "Utils.h"
#include "TypePromotion.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSStream.h"

#include <vector>

namespace tensorplay {
namespace mps {

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
               "MPS backend does not support dtype ", toString(dtype));
  }
}

MPSShape* getMPSShape(const std::vector<int64_t>& sizes) {
  NSMutableArray<NSNumber*>* shape =
      [NSMutableArray arrayWithCapacity:sizes.size()];
  for (int64_t s : sizes) {
    [shape addObject:@(s)];
  }
  if (shape.count == 0) {
    // MPSGraphTensorData cannot describe an empty shape; a 0-d tensor is
    // represented as a single-element shape.
    [shape addObject:@1];
  }
  return shape;
}

Tensor materializeForGraph(const Tensor& t) {
  if (t.is_contiguous() && t.storage_offset() == 0) {
    return t;
  }
  Tensor c = t.contiguous();
  if (c.storage_offset() != 0) {
    // A contiguous view into the middle of a buffer cannot be wrapped in an
    // MPSGraphTensorData; copy it into a fresh offset-0 tensor.
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
  return (id<MTLBuffer>)buf;
}

Tensor emptyMPS(const std::vector<int64_t>& sizes, DType dtype,
               const Device& device) {
  return Tensor(sizes, dtype, device);
}

namespace {

void runGraph(MPSGraph* graph, MPSShape* shape, MPSDataType inDataType,
              id<MTLBuffer> inputBuffer, MPSGraphTensor* inputTensor,
              MPSDataType outDataType, id<MTLBuffer> outputBuffer,
              MPSGraphTensor* outputTensor) {
  MPSGraphTensorData* inputData =
      [[[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                               shape:shape
                                            dataType:inDataType] autorelease];
  MPSGraphTensorData* outputData =
      [[[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                               shape:shape
                                            dataType:outDataType] autorelease];
  NSDictionary* feeds = @{inputTensor : inputData};
  NSDictionary* results = @{outputTensor : outputData};
  getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                        SyncType::COMMIT_AND_WAIT);
}

} // namespace

Tensor binaryOp(const Tensor& self, const Tensor& other, MPSBinaryOpBlock block,
                const std::optional<Scalar>& alpha,
                std::optional<DType> common_dtype,
                std::optional<DType> out_dtype) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS binary op requires both inputs on the MPS device");
  }
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const std::vector<int64_t> other_shape =
      static_cast<std::vector<int64_t>>(other.shape());
  const std::vector<int64_t> out_shape =
      broadcast_shapes(self_shape, other_shape);
  const DType common =
      common_dtype.value_or(promoteTypes(self.dtype(), other.dtype()));
  const DType output_dtype = out_dtype.value_or(common);

  Tensor self_c = materializeForGraph(self);
  Tensor other_c = materializeForGraph(other);
  if (static_cast<std::vector<int64_t>>(self_c.shape()) != out_shape) {
    self_c = materializeForGraph(self_c.expand(out_shape));
  }
  if (static_cast<std::vector<int64_t>>(other_c.shape()) != out_shape) {
    other_c = materializeForGraph(other_c.expand(out_shape));
  }

  Tensor output = emptyMPS(out_shape, output_dtype, self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* shape = getMPSShape(out_shape);
    const MPSDataType self_dt = getMPSDataType(self_c.dtype());
    const MPSDataType other_dt = getMPSDataType(other_c.dtype());
    const MPSDataType common_dt = getMPSDataType(common);
    const MPSDataType out_dt = getMPSDataType(output_dtype);

    MPSGraphTensor* x = [graph placeholderWithShape:shape
                                          dataType:self_dt
                                               name:@"x"];
    MPSGraphTensor* y = [graph placeholderWithShape:shape
                                          dataType:other_dt
                                               name:@"y"];
    MPSGraphTensor* xc = (self_c.dtype() == common)
        ? x
        : [graph castTensor:x toType:common_dt name:@"cast_x"];
    MPSGraphTensor* yc = (other_c.dtype() == common)
        ? y
        : [graph castTensor:y toType:common_dt name:@"cast_y"];

    MPSGraphTensor* result = nil;
    const bool has_alpha =
        alpha.has_value() && !alpha->isComplex();
    const bool alpha_is_one =
        has_alpha && alpha->toDouble() == 1.0;
    if (has_alpha && !alpha_is_one) {
      // Scale `other` by alpha (including 0) before running the op block.
      MPSGraphTensor* alphaConst = [graph constant:alpha->toDouble()
                                         dataType:common_dt];
      MPSGraphTensor* yScaled =
          [graph multiplicationWithPrimaryTensor:yc
                                      secondaryTensor:alphaConst
                                                 name:@"scale"];
      result = block(graph, xc, yScaled);
    } else {
      // No alpha (or alpha == 1): use `other` unchanged.
      result = block(graph, xc, yc);
    }
    if ([result dataType] != out_dt) {
      result = [graph castTensor:result toType:out_dt name:@"cast_out"];
    }

    id<MTLBuffer> xBuf = mtlBufferOf(self_c);
    id<MTLBuffer> yBuf = mtlBufferOf(other_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);

    MPSGraphTensorData* xData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:xBuf
                                                shape:shape
                                             dataType:self_dt] autorelease];
    MPSGraphTensorData* yData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:yBuf
                                                shape:shape
                                             dataType:other_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:shape
                                             dataType:out_dt] autorelease];

    NSDictionary* feeds = @{x : xData, y : yData};
    NSDictionary* results = @{result : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

Tensor unaryOp(const Tensor& self, MPSUnaryOpBlock block,
               std::optional<DType> out_dtype) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS unary op requires input on the MPS device");
  }
  const DType output_dtype = out_dtype.value_or(self.dtype());
  Tensor self_c = materializeForGraph(self);
  const std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(self_c.shape());
  Tensor output = emptyMPS(sizes, output_dtype, self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* shape = getMPSShape(sizes);
    const MPSDataType in_dt = getMPSDataType(self_c.dtype());
    const MPSDataType out_dt = getMPSDataType(output_dtype);

    MPSGraphTensor* x = [graph placeholderWithShape:shape
                                          dataType:in_dt
                                               name:@"x"];
    MPSGraphTensor* result = block(graph, x);
    if ([result dataType] != out_dt) {
      result = [graph castTensor:result toType:out_dt name:@"cast_out"];
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    runGraph(graph, shape, in_dt, inBuf, x, out_dt, outBuf, result);
  }
  return output;
}

Tensor whereOp(const Tensor& condition, const Tensor& self,
               const Tensor& other) {
  if (!condition.device().is_mps() || !self.device().is_mps() ||
      !other.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS where requires all inputs on the MPS device");
  }
  const std::vector<int64_t> cond_shape =
      static_cast<std::vector<int64_t>>(condition.shape());
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const std::vector<int64_t> other_shape =
      static_cast<std::vector<int64_t>>(other.shape());
  const std::vector<int64_t> out_shape =
      broadcast_shapes(broadcast_shapes(cond_shape, self_shape), other_shape);
  const DType common = promoteTypes(self.dtype(), other.dtype());

  Tensor cond_c = materializeForGraph(condition);
  Tensor self_c = materializeForGraph(self);
  Tensor other_c = materializeForGraph(other);
  if (static_cast<std::vector<int64_t>>(cond_c.shape()) != out_shape) {
    cond_c = materializeForGraph(cond_c.expand(out_shape));
  }
  if (static_cast<std::vector<int64_t>>(self_c.shape()) != out_shape) {
    self_c = materializeForGraph(self_c.expand(out_shape));
  }
  if (static_cast<std::vector<int64_t>>(other_c.shape()) != out_shape) {
    other_c = materializeForGraph(other_c.expand(out_shape));
  }

  Tensor output = emptyMPS(out_shape, common, self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* shape = getMPSShape(out_shape);
    const MPSDataType cond_dt = getMPSDataType(cond_c.dtype());
    const MPSDataType self_dt = getMPSDataType(self_c.dtype());
    const MPSDataType other_dt = getMPSDataType(other_c.dtype());
    const MPSDataType common_dt = getMPSDataType(common);

    MPSGraphTensor* cond = [graph placeholderWithShape:shape
                                             dataType:cond_dt
                                                  name:@"cond"];
    MPSGraphTensor* t = [graph placeholderWithShape:shape
                                           dataType:self_dt
                                                name:@"t"];
    MPSGraphTensor* f = [graph placeholderWithShape:shape
                                          dataType:other_dt
                                               name:@"f"];
    MPSGraphTensor* condBool =
        (cond_c.dtype() == DType::Bool)
        ? cond
        : [graph castTensor:cond toType:MPSDataTypeBool name:@"cast_cond"];
    MPSGraphTensor* tc = (self_c.dtype() == common)
        ? t
        : [graph castTensor:t toType:common_dt name:@"cast_t"];
    MPSGraphTensor* fc = (other_c.dtype() == common)
        ? f
        : [graph castTensor:f toType:common_dt name:@"cast_f"];
    MPSGraphTensor* result =
        [graph selectWithPredicateTensor:condBool
                     truePredicateTensor:tc
                    falsePredicateTensor:fc
                                    name:nil];

    id<MTLBuffer> condBuf = mtlBufferOf(cond_c);
    id<MTLBuffer> tBuf = mtlBufferOf(self_c);
    id<MTLBuffer> fBuf = mtlBufferOf(other_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);

    MPSGraphTensorData* condData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:condBuf
                                                shape:shape
                                             dataType:cond_dt] autorelease];
    MPSGraphTensorData* tData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:tBuf
                                                shape:shape
                                             dataType:self_dt] autorelease];
    MPSGraphTensorData* fData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:fBuf
                                                shape:shape
                                             dataType:other_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:shape
                                             dataType:common_dt] autorelease];

    NSDictionary* feeds = @{cond : condData, t : tData, f : fData};
    NSDictionary* results = @{result : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

} // namespace mps
} // namespace tensorplay