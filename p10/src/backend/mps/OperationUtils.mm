#include "OperationUtils.h"

#include "Utils.h"
#include "TypePromotion.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
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

namespace {

// One entry per distinct graph structure, for the process lifetime. Nothing is
// evicted: a workload with fixed shapes needs a handful, a workload that varies
// them still needs one per shape, and keeping every entry alive is what makes a
// pointer handed out by findCachedGraph safe to use while another thread
// populates the cache.
std::mutex& graphCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, CachedGraph*>& graphCache() {
  static std::unordered_map<std::string, CachedGraph*> cache;
  return cache;
}

void releaseCachedGraph(CachedGraph* entry) {
  [entry->graph release];
  [entry->input0 release];
  [entry->input1 release];
  [entry->input2 release];
  [entry->result release];
  [entry->result2 release];
  delete entry;
}

} // namespace

CachedGraph* findCachedGraph(const std::string& key) {
  std::lock_guard<std::mutex> guard(graphCacheMutex());
  auto it = graphCache().find(key);
  return it == graphCache().end() ? nullptr : it->second;
}

void storeCachedGraph(const std::string& key, CachedGraph* entry) {
  std::lock_guard<std::mutex> guard(graphCacheMutex());
  auto inserted = graphCache().emplace(key, entry);
  if (!inserted.second) {
    // Another thread built the same graph first; keep its entry and drop the
    // duplicate.
    releaseCachedGraph(entry);
  }
}

std::string shapeKey(const std::vector<int64_t>& shape) {
  std::string key;
  for (int64_t size : shape) {
    key += std::to_string(size);
    key += 'x';
  }
  return key;
}

std::string dtypeKey(DType dtype) {
  return std::to_string(static_cast<int>(dtype));
}

std::string scalarKey(const Scalar& value) {
  // The bit pattern, so two alphas that print the same stay distinct.
  double number = value.toDouble();
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(number));
  std::memcpy(&bits, &number, sizeof(bits));
  return std::to_string(bits);
}

std::string graphKey(const std::source_location& location,
                     std::initializer_list<std::string> parts) {
  // File, line and column: nested calls on one line are different call
  // sites and must not share a graph.
  std::string key = location.file_name();
  key += ':';
  key += std::to_string(location.line());
  key += ':';
  key += std::to_string(location.column());
  for (const std::string& part : parts) {
    key += '|';
    key += part;
  }
  return key;
}

Tensor binaryOp(const Tensor& self, const Tensor& other, MPSBinaryOpBlock block,
                const std::optional<Scalar>& alpha,
                std::optional<DType> common_dtype,
                std::optional<DType> out_dtype,
                std::source_location location) {
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

  const bool has_alpha =
      alpha.has_value() && !alpha->isComplex();
  const bool alpha_is_one =
      has_alpha && alpha->toDouble() == 1.0;
  // The call site names the operation; shapes, dtypes and alpha complete the
  // identity of the graph the block would build.
  const std::string key = graphKey(
      location, {shapeKey(out_shape), dtypeKey(self_c.dtype()),
                 dtypeKey(other_c.dtype()), dtypeKey(common),
                 dtypeKey(output_dtype),
                 has_alpha ? scalarKey(*alpha) : std::string("none")});

  @autoreleasepool {
    MPSShape* shape = getMPSShape(out_shape);
    const MPSDataType self_dt = getMPSDataType(self_c.dtype());
    const MPSDataType other_dt = getMPSDataType(other_c.dtype());
    const MPSDataType common_dt = getMPSDataType(common);
    const MPSDataType out_dt = getMPSDataType(output_dtype);

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
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
      entry = new CachedGraph{graph, [x retain], [y retain], nil,
                              [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
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

    NSDictionary* feeds = @{entry->input0 : xData, entry->input1 : yData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

namespace {

// The most dimensions the shader walks, and the element index width it uses.
// BroadcastElementwise.metal reads a 32-bit thread index and forms 32-bit
// offsets, so a call that needs more than this stays with the graph.
constexpr size_t kShaderMaxNdims = 16;

// The pipeline for `op` over `dtype`, or nil when the shader is not built for
// that combination. Half, bfloat and the narrow integers are absent from
// BroadcastElementwise.metal because their operation math type is not the
// storage type; a call with one of them belongs with the graph, which builds
// the cast and the operation together. The list below and the instantiations at
// the bottom of that file are checked in the same order: dtype first, so a
// missing name is only possible if the two get out of step. Whether a name
// resolves is answered once and kept, because finding a function in the
// metallib costs an object creation on every call.
MTLComputePipelineState_t broadcastShaderPipeline(const std::string& op,
                                                  DType dtype) {
  switch (dtype) {
    case DType::Float32:
    case DType::Int64:
    case DType::Int32:
      break;
    default:
      return nil;
  }
  static std::mutex mutex;
  static std::unordered_map<std::string, MTLComputePipelineState_t> resolved;
  const std::string name =
      "broadcast_" + op + "_" + scalarToMetalTypeString(dtype);
  std::lock_guard<std::mutex> guard(mutex);
  auto it = resolved.find(name);
  if (it != resolved.end()) {
    return it->second;
  }
  auto& lib = MetalShaderLibrary::getBundledLibrary();
  MTLComputePipelineState_t pso =
      lib.hasFunction(name) ? lib.getPipelineStateForFunc(name) : nil;
  resolved.emplace(name, pso);
  return pso;
}

// The stride array BroadcastElementwise.metal indexes `t` with, ordered
// fastest-varying dimension first, along with the largest element offset it can
// produce. A dimension of the output that `t` does not describe -- one it lacks
// because `t` is shorter, or one of size 1 that the output stretches -- gets
// stride 0, which re-reads the same element along that axis instead of stepping
// to a distinct one. Returns nil for a negative stride (a view that walks
// backwards through its storage would need an offset base the shader does not
// take) or for a stride array whose addressing leaves the 32-bit range.
std::optional<std::array<int32_t, kShaderMaxNdims>> broadcastShaderStrides(
    const Tensor& t, const std::vector<int64_t>& out_shape) {
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(t.shape());
  if (shape.size() > out_shape.size()) {
    return std::nullopt;
  }
  const std::vector<int64_t> strides =
      static_cast<std::vector<int64_t>>(t.strides());
  const int64_t lead = static_cast<int64_t>(out_shape.size()) -
                       static_cast<int64_t>(shape.size());
  std::array<int32_t, kShaderMaxNdims> offsets{};
  int64_t max_offset = 0;
  for (int64_t o = 0; o < static_cast<int64_t>(out_shape.size()); ++o) {
    const int64_t i = o - lead;
    int64_t stride = 0;
    if (i >= 0) {
      if (shape[i] == out_shape[o]) {
        if (strides[i] < 0) {
          return std::nullopt;
        }
        stride = strides[i];
      } else if (shape[i] != 1) {
        return std::nullopt;
      }
    }
    const int64_t extent = out_shape[o] > 1 ? out_shape[o] - 1 : 0;
    if (stride > std::numeric_limits<int32_t>::max() ||
        max_offset >
            std::numeric_limits<int32_t>::max() - stride * extent) {
      return std::nullopt;
    }
    max_offset += stride * extent;
    offsets[out_shape.size() - 1 - o] = static_cast<int32_t>(stride);
  }
  return offsets;
}

} // namespace

std::optional<Tensor> broadcastBinaryShader(
    const std::string& op, const Tensor& self, const Tensor& other,
    const std::optional<Scalar>& alpha) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    return std::nullopt;
  }
  // Scaling `other` is folded into the graph as a multiply; the shader adds the
  // two operands as they are.
  if (alpha.has_value() && (alpha->isComplex() || alpha->toDouble() != 1.0)) {
    return std::nullopt;
  }
  // Different storage dtypes are promoted to a common one, which the graph
  // builds as casts ahead of the operation.
  if (self.dtype() != other.dtype()) {
    return std::nullopt;
  }
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const std::vector<int64_t> other_shape =
      static_cast<std::vector<int64_t>>(other.shape());
  const std::vector<int64_t> out_shape =
      broadcast_shapes(self_shape, other_shape);
  // Where nothing expands, both paths are a single dispatch over the same
  // buffers, so there is nothing for the shader to save.
  if (self_shape == out_shape && other_shape == out_shape) {
    return std::nullopt;
  }
  if (out_shape.empty() || out_shape.size() > kShaderMaxNdims) {
    return std::nullopt;
  }
  int64_t numel = 1;
  for (int64_t size : out_shape) {
    numel *= size;
  }
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  // The shader addresses a buffer from its start; a view that begins inside one
  // has no base-offset slot to say so.
  if (self.storage_offset() != 0 || other.storage_offset() != 0) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = broadcastShaderPipeline(op, self.dtype());
  if (!pso) {
    return std::nullopt;
  }
  auto self_strides = broadcastShaderStrides(self, out_shape);
  auto other_strides = broadcastShaderStrides(other, out_shape);
  if (!self_strides.has_value() || !other_strides.has_value()) {
    return std::nullopt;
  }
  void* self_buf = getMTLBufferFromPointer(self.data_ptr());
  void* other_buf = getMTLBufferFromPointer(other.data_ptr());
  if (!self_buf || !other_buf) {
    return std::nullopt;
  }

  std::array<uint32_t, kShaderMaxNdims> sizes{};
  for (size_t o = 0; o < out_shape.size(); ++o) {
    sizes[out_shape.size() - 1 - o] = static_cast<uint32_t>(out_shape[o]);
  }
  const uint32_t ndim = static_cast<uint32_t>(out_shape.size());
  const uint32_t total = static_cast<uint32_t>(numel);
  const std::array<int32_t, kShaderMaxNdims> lhs_strides = *self_strides;
  const std::array<int32_t, kShaderMaxNdims> rhs_strides = *other_strides;

  Tensor output = emptyMPS(out_shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      stream->endKernelCoalescing();
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf offset:0 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)other_buf offset:0 atIndex:2];
      [encoder setBytes:&ndim length:sizeof(ndim) atIndex:3];
      [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:4];
      [encoder setBytes:lhs_strides.data() length:sizeof(lhs_strides)
                atIndex:5];
      [encoder setBytes:rhs_strides.data() length:sizeof(rhs_strides)
                atIndex:6];
      const uint32_t per_group =
          std::min(total,
                   static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
  return output;
}

Tensor unaryOp(const Tensor& self, MPSUnaryOpBlock block,
               std::optional<DType> out_dtype,
               std::source_location location) {
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

  const std::string key = graphKey(
      location, {shapeKey(sizes), dtypeKey(self_c.dtype()),
                 dtypeKey(output_dtype)});

  @autoreleasepool {
    MPSShape* shape = getMPSShape(sizes);
    const MPSDataType in_dt = getMPSDataType(self_c.dtype());
    const MPSDataType out_dt = getMPSDataType(output_dtype);

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* x = [graph placeholderWithShape:shape
                                             dataType:in_dt
                                                  name:@"x"];
      MPSGraphTensor* result = block(graph, x);
      if ([result dataType] != out_dt) {
        result = [graph castTensor:result toType:out_dt name:@"cast_out"];
      }
      entry = new CachedGraph{graph, [x retain], nil, nil,
                              [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    runGraph(entry->graph, shape, in_dt, inBuf, entry->input0, out_dt,
             outBuf, entry->result);
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