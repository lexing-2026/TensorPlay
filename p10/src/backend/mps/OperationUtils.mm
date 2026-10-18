#include "OperationUtils.h"

#include "Utils.h"
#include "TypePromotion.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/kernels/common.h"

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
                                        SyncType::COMMIT_ADAPTIVE);
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

std::string graphKey(const std::string& op,
                     std::initializer_list<std::string> parts) {
  // The name comes first: an entry is looked up by the computation it performs
  // and the structure it was built for, never by where it was called from.
  std::string key = op;
  for (const std::string& part : parts) {
    key += '|';
    key += part;
  }
  return key;
}

Tensor binaryOp(const std::string& op, const Tensor& self,
                const Tensor& other, MPSBinaryOpBlock block,
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

  const bool has_alpha =
      alpha.has_value() && !alpha->isComplex();
  const bool alpha_is_one =
      has_alpha && alpha->toDouble() == 1.0;
  // `op` names the operation (and the constants its block bakes in); shapes,
  // dtypes and alpha complete the identity of the graph the block would build.
  const std::string key = graphKey(
      op, {shapeKey(out_shape), dtypeKey(self_c.dtype()),
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
        MPSGraphTensor* alphaConst =
            [graph constantWithScalar:alpha->toDouble()
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
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

namespace {

// The most dimensions the shader walks, and the element index width it uses.
// BroadcastElementwise.metal reads a 32-bit thread index and forms 32-bit
// offsets, so a call that needs more than this stays with the graph.
constexpr size_t kShaderMaxNdims = 16;

// The pipeline for a kernel of the bundled metallib, or nil when no function
// carries that name. Whether a name resolves is answered once and kept: finding
// a function in the metallib costs an object creation on every call, and the
// callers ask on every operation.
MTLComputePipelineState_t shaderPipelineNamed(const std::string& name) {
  static std::mutex mutex;
  static std::unordered_map<std::string, MTLComputePipelineState_t> resolved;
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

// The pipeline for `op` over `dtype`, or nil when the shader is not built for
// that combination. Half, bfloat and the narrow integers are absent from
// BroadcastElementwise.metal because their operation math type is not the
// storage type; a call with one of them belongs with the graph, which builds
// the cast and the operation together. The list below and the instantiations at
// the bottom of that file are checked in the same order: dtype first, so a
// missing name is only possible if the two get out of step. `ilp` selects the
// form that takes a run of consecutive elements per thread; the same
// operations are instantiated in both forms.
MTLComputePipelineState_t broadcastShaderPipeline(const std::string& op,
                                                  DType dtype, bool ilp) {
  switch (dtype) {
    case DType::Float32:
    case DType::Int64:
    case DType::Int32:
      break;
    default:
      return nil;
  }
  return shaderPipelineNamed("broadcast_" + op + "_" +
                             scalarToMetalTypeString(dtype) +
                             (ilp ? "_ilp" : ""));
}

// The pipeline for a unit-stride form of `op` over `dtype`, or nil when the
// shader is not built for that combination. The dense forms live in
// BroadcastElementwise.metal next to the stride-addressed ones; the dtype list
// matches theirs, and an operation without a dense form (integral div) is
// answered by a missing kernel name, which resolves to nil.
MTLComputePipelineState_t denseBinaryShaderPipeline(const std::string& op,
                                                    DType dtype,
                                                    bool scalar_rhs) {
  switch (dtype) {
    case DType::Float32:
    case DType::Int64:
    case DType::Int32:
      break;
    default:
      return nil;
  }
  return shaderPipelineNamed("dense_" + op + (scalar_rhs ? "_scalar_" : "_") +
                             scalarToMetalTypeString(dtype));
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

// The pipeline for `op` over `dtype` in UnaryElementwise.metal, or nil when the
// shader is not built for that combination. Float is the only storage type that
// file instantiates for these operations, so a half, bfloat or integral input
// belongs with the graph, which builds the cast around the operation.
MTLComputePipelineState_t unaryShaderPipeline(const std::string& op,
                                              DType dtype) {
  if (dtype != DType::Float32) {
    return nil;
  }
  return shaderPipelineNamed("unary_" + op + "_" +
                             scalarToMetalTypeString(dtype));
}

// The pipeline for `op` over `dtype` in DenseCompare.metal, or nil when the
// shader is not built for that combination. The list below and the
// instantiations at the bottom of that file are checked in the same order, so a
// missing name is only possible if the two get out of step; every dtype outside
// it is answered by the graph.
MTLComputePipelineState_t compareShaderPipeline(const std::string& op,
                                                DType dtype, bool scalar_rhs) {
  switch (dtype) {
    case DType::Float32:
    case DType::Int64:
    case DType::Int32:
      break;
    default:
      return nil;
  }
  return shaderPipelineNamed("compare_" + op +
                             (scalar_rhs ? "_scalar_" : "_") +
                             scalarToMetalTypeString(dtype));
}

} // namespace

// A shader launch encodes into the command buffer the stream is already
// building and leaves it open, so consecutive launches share one commit and
// their host cost is paid once per run rather than once per launch. Whatever
// needs the result closes the encoder before it uses the buffer -- a copy to
// the host, a graph launch, an explicit synchronize -- and that commit carries
// everything the chain has encoded so far.
std::optional<Tensor> broadcastBinaryShader(
    const std::string& op, const Tensor& self, const Tensor& other,
    const std::optional<Scalar>& alpha, bool allow_unexpanded) {
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
  // buffers; callers that measure the graph as the cheaper of the two keep
  // that choice by leaving `allow_unexpanded` clear.
  if (!allow_unexpanded && self_shape == out_shape &&
      other_shape == out_shape) {
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
  // A thread can take a run of consecutive output elements when the innermost
  // dimension is a multiple of the run width, so no run crosses a row and the
  // coordinates of the outer dimensions are formed once per run.
  const bool ilp = out_shape.back() % p10::metal::ILP_PER_THREAD == 0;
  MTLComputePipelineState_t pso =
      broadcastShaderPipeline(op, self.dtype(), ilp);
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
      const uint32_t threads =
          ilp ? p10::metal::ceil_div<uint32_t>(total,
                                               p10::metal::ILP_PER_THREAD)
              : total;
      if (ilp) {
        [encoder setBytes:&total length:sizeof(total) atIndex:7];
      }
      const uint32_t per_group =
          std::min(threads,
                   static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
      [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

namespace {

// A dense call reads a thread's own index from each operand, so the operands
// have to share a shape and each run through its storage in the order the
// output is written, from the start of that storage.
bool denseOperandPair(const Tensor& self, const Tensor& other) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    return false;
  }
  if (self.dtype() != other.dtype()) {
    return false;
  }
  if (!self.is_contiguous() || !other.is_contiguous() ||
      self.storage_offset() != 0 || other.storage_offset() != 0) {
    return false;
  }
  return static_cast<std::vector<int64_t>>(self.shape()) ==
         static_cast<std::vector<int64_t>>(other.shape());
}

// The dense kernels take no scale factor; a call that scales its second
// operand is expressed by the stride-addressed form or the graph instead.
bool denseScaleIsOne(const std::optional<Scalar>& alpha) {
  return !alpha.has_value() ||
         (!alpha->isComplex() && alpha->toDouble() == 1.0);
}

// A grid no wider than one threadgroup keeps the whole call on one core, so a
// small span is dispatched one unit per thread with its groups capped well
// below the device maximum: the launch then spreads over several cores the way
// a larger span already does. Above the cutoff the ILP form's shorter grid
// already fills multiple groups.
constexpr uint32_t kSmallDenseElements = 1u << 14;
constexpr uint32_t kSmallDenseThreads = 256;

// `span` counts the units a thread reads: vectors for the four-wide form,
// elements for the scalar-indexed one. Only the four-wide kernels accept any
// grid width; the scalar-indexed ones always take ILP_PER_THREAD in one run.
uint32_t denseSpanThreads(uint32_t span, uint32_t elements, bool grid_stride) {
  if (elements <= kSmallDenseElements && grid_stride) {
    return span;
  }
  return static_cast<uint32_t>(
      p10::metal::ceil_div<uint32_t>(span, p10::metal::ILP_PER_THREAD));
}

uint32_t denseThreadsPerGroup(uint32_t total, uint32_t elements,
                              uint32_t max_threads) {
  const uint32_t cap =
      elements <= kSmallDenseElements ? kSmallDenseThreads : max_threads;
  return std::min(total, cap);
}

// A four-wide launch moves 16 bytes per access, which needs float storage,
// whole vectors over the span, and every base on a 16-byte boundary.
bool wholeFloatVectors(DType dtype, uint32_t elements,
                       std::initializer_list<const void*> bases) {
  if (dtype != DType::Float32 || elements % 4 != 0) {
    return false;
  }
  for (const void* base : bases) {
    if (reinterpret_cast<uintptr_t>(base) % 16 != 0) {
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<Tensor> denseBinaryShader(const std::string& op,
                                        const Tensor& self,
                                        const Tensor& other,
                                        const std::optional<Scalar>& alpha) {
  if (!denseScaleIsOne(alpha) || !denseOperandPair(self, other)) {
    return std::nullopt;
  }
  const int64_t numel = self.numel();
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso =
      denseBinaryShaderPipeline(op, self.dtype(), /*scalar_rhs=*/false);
  if (!pso) {
    return std::nullopt;
  }
  void* self_buf = getMTLBufferFromPointer(self.data_ptr());
  void* other_buf = getMTLBufferFromPointer(other.data_ptr());
  if (!self_buf || !other_buf) {
    return std::nullopt;
  }
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  Tensor output = emptyMPS(shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  const uint32_t elements = static_cast<uint32_t>(numel);
  MTLComputePipelineState_t vec_pso =
      wholeFloatVectors(self.dtype(), elements,
                        {self.data_ptr(), other.data_ptr(), output.data_ptr()})
          ? shaderPipelineNamed("dense_vec_" + op + "_float")
          : nil;
  MTLComputePipelineState_t active_pso = vec_pso ? vec_pso : pso;
  // The vector form counts 16-byte vectors, the scalar form elements.
  const uint32_t span = vec_pso ? elements / 4 : elements;
  const uint32_t total = denseSpanThreads(span, elements, vec_pso != nil);
  const uint32_t per_group = denseThreadsPerGroup(
      total, elements,
      static_cast<uint32_t>([active_pso maxTotalThreadsPerThreadgroup]));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:active_pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf offset:0 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)other_buf offset:0 atIndex:2];
      [encoder setBytes:&span length:sizeof(span) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

std::optional<Tensor> denseScalarBinaryShader(const std::string& op,
                                              const Tensor& self,
                                              const Scalar& other,
                                              const std::optional<Scalar>& alpha) {
  if (!denseScaleIsOne(alpha) || !self.device().is_mps()) {
    return std::nullopt;
  }
  // The scalar has to land in the operand's own storage type. A call whose
  // scalar promotes the result -- a real scalar against an integral tensor, or
  // a bool against anything but bool -- goes to the graph, which builds the
  // cast around the operation.
  if (result_type(other, self.dtype()) != self.dtype()) {
    return std::nullopt;
  }
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return std::nullopt;
  }
  const int64_t numel = self.numel();
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso =
      denseBinaryShaderPipeline(op, self.dtype(), /*scalar_rhs=*/true);
  if (!pso) {
    return std::nullopt;
  }
  void* self_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!self_buf) {
    return std::nullopt;
  }
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  Tensor output = emptyMPS(shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  const uint32_t elements = static_cast<uint32_t>(numel);
  MTLComputePipelineState_t vec_pso =
      wholeFloatVectors(self.dtype(), elements,
                        {self.data_ptr(), output.data_ptr()})
          ? shaderPipelineNamed("dense_vec_" + op + "_scalar_float")
          : nil;
  MTLComputePipelineState_t active_pso = vec_pso ? vec_pso : pso;
  // The vector form counts 16-byte vectors, the scalar form elements.
  const uint32_t span = vec_pso ? elements / 4 : elements;
  const uint32_t total = denseSpanThreads(span, elements, vec_pso != nil);
  const uint32_t per_group = denseThreadsPerGroup(
      total, elements,
      static_cast<uint32_t>([active_pso maxTotalThreadsPerThreadgroup]));
  // The scalar is converted once, in the storage type the kernel reads.
  float value_f = 0.0f;
  int64_t value_l = 0;
  int32_t value_i = 0;
  const void* value = nullptr;
  size_t value_size = 0;
  switch (self.dtype()) {
    case DType::Float32:
      value_f = other.to<float>();
      value = &value_f;
      value_size = sizeof(value_f);
      break;
    case DType::Int64:
      value_l = other.to<int64_t>();
      value = &value_l;
      value_size = sizeof(value_l);
      break;
    default:
      value_i = other.to<int32_t>();
      value = &value_i;
      value_size = sizeof(value_i);
      break;
  }

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:active_pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf offset:0 atIndex:1];
      [encoder setBytes:value length:value_size atIndex:2];
      [encoder setBytes:&span length:sizeof(span) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

std::optional<Tensor> broadcastTernaryShader(
    const std::string& op, const Tensor& self, const Tensor& lhs,
    const Tensor& rhs, const Scalar& value) {
  if (!self.device().is_mps() || !lhs.device().is_mps() ||
      !rhs.device().is_mps()) {
    return std::nullopt;
  }
  // Different storage dtypes are promoted to a common one, which the graph
  // builds as casts ahead of the operation.
  if (self.dtype() != lhs.dtype() || self.dtype() != rhs.dtype()) {
    return std::nullopt;
  }
  switch (self.dtype()) {
    case DType::Float32:
    case DType::Int64:
    case DType::Int32:
      break;
    default:
      return std::nullopt;
  }
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const std::vector<int64_t> lhs_shape =
      static_cast<std::vector<int64_t>>(lhs.shape());
  const std::vector<int64_t> rhs_shape =
      static_cast<std::vector<int64_t>>(rhs.shape());
  const std::vector<int64_t> out_shape = broadcast_shapes(
      broadcast_shapes(self_shape, lhs_shape), rhs_shape);
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
  if (self.storage_offset() != 0 || lhs.storage_offset() != 0 ||
      rhs.storage_offset() != 0) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed(
      "broadcast_" + op + "_" + scalarToMetalTypeString(self.dtype()));
  if (!pso) {
    return std::nullopt;
  }
  auto self_strides = broadcastShaderStrides(self, out_shape);
  auto lhs_strides = broadcastShaderStrides(lhs, out_shape);
  auto rhs_strides = broadcastShaderStrides(rhs, out_shape);
  if (!self_strides.has_value() || !lhs_strides.has_value() ||
      !rhs_strides.has_value()) {
    return std::nullopt;
  }
  void* self_buf = getMTLBufferFromPointer(self.data_ptr());
  void* lhs_buf = getMTLBufferFromPointer(lhs.data_ptr());
  void* rhs_buf = getMTLBufferFromPointer(rhs.data_ptr());
  if (!self_buf || !lhs_buf || !rhs_buf) {
    return std::nullopt;
  }

  std::array<uint32_t, kShaderMaxNdims> sizes{};
  for (size_t o = 0; o < out_shape.size(); ++o) {
    sizes[out_shape.size() - 1 - o] = static_cast<uint32_t>(out_shape[o]);
  }
  const uint32_t ndim = static_cast<uint32_t>(out_shape.size());
  const uint32_t total = static_cast<uint32_t>(numel);
  const std::array<int32_t, kShaderMaxNdims> self_stride_array = *self_strides;
  const std::array<int32_t, kShaderMaxNdims> lhs_stride_array = *lhs_strides;
  const std::array<int32_t, kShaderMaxNdims> rhs_stride_array = *rhs_strides;

  Tensor output = emptyMPS(out_shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf offset:0 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)lhs_buf offset:0 atIndex:2];
      [encoder setBuffer:(__bridge id<MTLBuffer>)rhs_buf offset:0 atIndex:3];
      if (self.dtype() == DType::Float32) {
        const float factor = value.to<float>();
        [encoder setBytes:&factor length:sizeof(factor) atIndex:4];
      } else if (self.dtype() == DType::Int64) {
        const int64_t factor = value.to<int64_t>();
        [encoder setBytes:&factor length:sizeof(factor) atIndex:4];
      } else {
        const int32_t factor = value.to<int32_t>();
        [encoder setBytes:&factor length:sizeof(factor) atIndex:4];
      }
      [encoder setBytes:&ndim length:sizeof(ndim) atIndex:5];
      [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:6];
      [encoder setBytes:self_stride_array.data()
                 length:sizeof(self_stride_array)
                atIndex:7];
      [encoder setBytes:lhs_stride_array.data()
                 length:sizeof(lhs_stride_array)
                atIndex:8];
      [encoder setBytes:rhs_stride_array.data()
                 length:sizeof(rhs_stride_array)
                atIndex:9];
      const uint32_t per_group =
          std::min(total,
                   static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

std::optional<Tensor> unaryShader(const std::string& op, const Tensor& self) {
  if (!self.device().is_mps()) {
    return std::nullopt;
  }
  // The thread index is the element index, so the input has to run through its
  // storage in the order the output is written in and from the start of it.
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return std::nullopt;
  }
  const int64_t numel = self.numel();
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = unaryShaderPipeline(op, self.dtype());
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  Tensor output = emptyMPS(shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  const uint32_t elements = static_cast<uint32_t>(numel);
  MTLComputePipelineState_t vec_pso =
      wholeFloatVectors(self.dtype(), elements,
                        {self.data_ptr(), output.data_ptr()})
          ? shaderPipelineNamed("unary_vec_" + op + "_float")
          : nil;
  MTLComputePipelineState_t active_pso = vec_pso ? vec_pso : pso;
  // The vector form counts 16-byte vectors, the scalar form elements.
  const uint32_t span = vec_pso ? elements / 4 : elements;
  const uint32_t total = denseSpanThreads(span, elements, vec_pso != nil);
  const uint32_t per_group = denseThreadsPerGroup(
      total, elements,
      static_cast<uint32_t>([active_pso maxTotalThreadsPerThreadgroup]));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:active_pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:1];
      [encoder setBytes:&span length:sizeof(span) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

std::optional<Tensor> clampShader(const Tensor& self,
                                  const std::optional<Scalar>& min,
                                  const std::optional<Scalar>& max) {
  if (!self.device().is_mps()) {
    return std::nullopt;
  }
  if (self.dtype() != DType::Float32) {
    return std::nullopt;
  }
  // A complex bound that the float storage cannot hold goes to the graph.
  if (min.has_value() && min->isComplex()) {
    return std::nullopt;
  }
  if (max.has_value() && max->isComplex()) {
    return std::nullopt;
  }
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return std::nullopt;
  }
  const int64_t numel = self.numel();
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed("clamp_elementwise_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  Tensor output = emptyMPS(shape, self.dtype(), self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  // An absent bound is neutral in the comparison the shader applies.
  const float lo = min.has_value()
                       ? min->to<float>()
                       : -std::numeric_limits<float>::infinity();
  const float hi = max.has_value()
                       ? max->to<float>()
                       : std::numeric_limits<float>::infinity();
  const uint32_t elements = static_cast<uint32_t>(numel);
  MTLComputePipelineState_t vec_pso =
      wholeFloatVectors(self.dtype(), elements,
                        {self.data_ptr(), output.data_ptr()})
          ? shaderPipelineNamed("clamp_elementwise_vec_float")
          : nil;
  MTLComputePipelineState_t active_pso = vec_pso ? vec_pso : pso;
  // The vector form counts 16-byte vectors, the scalar form elements.
  const uint32_t span = vec_pso ? elements / 4 : elements;
  const uint32_t total = denseSpanThreads(span, elements, vec_pso != nil);
  const uint32_t per_group = denseThreadsPerGroup(
      total, elements,
      static_cast<uint32_t>([active_pso maxTotalThreadsPerThreadgroup]));

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:active_pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:1];
      [encoder setBytes:&span length:sizeof(span) atIndex:2];
      [encoder setBytes:&lo length:sizeof(lo) atIndex:3];
      [encoder setBytes:&hi length:sizeof(hi) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

namespace {

// A row reduction serves one row per SIMD group, so a threadgroup of this many
// threads covers this many rows at once. The kernel reads the width it was
// dispatched with, so only the host has to keep it a whole number of groups.
constexpr uint32_t kRowReduceThreads = 256;
constexpr uint32_t kRowReduceSimdgroup = p10::metal::simdgroup_size;

// Encodes a reduction over `rows` rows of `length` consecutive elements.
// `divisor` rides only with the kernel that takes one.
void dispatchRowReduce(MTLComputePipelineState_t pso, void* out_buf,
                       void* in_buf, uint32_t rows, uint32_t length,
                       const std::optional<float>& divisor) {
  const uint32_t threads =
      std::min(kRowReduceThreads,
               static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]) /
                   kRowReduceSimdgroup * kRowReduceSimdgroup);
  const uint32_t rows_per_group = threads / kRowReduceSimdgroup;
  const uint32_t groups = (rows + rows_per_group - 1) / rows_per_group;
  // The shader reads the two as one pair, so they travel as one argument. A
  // block copies what it captures, which an object of class type survives and a
  // raw array does not.
  const std::array<uint32_t, 2> sizes = {rows, length};
  const float division = divisor.value_or(0.0f);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:1];
      [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:2];
      if (divisor.has_value()) {
        [encoder setBytes:&division length:sizeof(division) atIndex:3];
      }
      [encoder dispatchThreads:MTLSizeMake(groups * threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
  });
}

// Encodes a reduction that answers twice per row: with the extremum and with the
// position it came from, which the kernel writes to two buffers.
void dispatchRowValueIndex(MTLComputePipelineState_t pso, void* value_buf,
                           void* index_buf, void* in_buf, uint32_t rows,
                           uint32_t length) {
  const uint32_t threads =
      std::min(kRowReduceThreads,
               static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]) /
                   kRowReduceSimdgroup * kRowReduceSimdgroup);
  const uint32_t rows_per_group = threads / kRowReduceSimdgroup;
  const uint32_t groups = (rows + rows_per_group - 1) / rows_per_group;
  const std::array<uint32_t, 2> sizes = {rows, length};
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)value_buf offset:0
                atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)index_buf offset:0
                atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:2];
      [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(groups * threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
  });
}

// The rows of an operand reduced over its last axis and the length of one of
// them. A reduction whose result does not hold one value per row is not this
// kernel's to answer, and neither is a rank-1 one: its result has no shape for
// the write to be described with. `in_dtype` is the storage type the kernel the
// rows are headed for reads, so an operand of another type is declined rather
// than addressed as though it were that type.
bool rowReduceSpan(const Tensor& self, DType in_dtype,
                   const std::vector<int64_t>& out_shape,
                   uint32_t& rows, uint32_t& length) {
  if (!self.device().is_mps() || self.dtype() != in_dtype ||
      out_shape.empty()) {
    return false;
  }
  // The kernel numbers rows and elements as it addresses them, so the operand
  // has to run through its storage in that order and from the start of it.
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return false;
  }
  const std::vector<int64_t> in_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t numel = self.numel();
  int64_t out_numel = 1;
  for (int64_t size : out_shape) {
    out_numel *= size;
  }
  if (in_shape.empty() || numel <= 0 || in_shape.back() <= 0 ||
      in_shape.back() > std::numeric_limits<uint32_t>::max() ||
      out_numel > std::numeric_limits<uint32_t>::max() ||
      numel / in_shape.back() != out_numel) {
    return false;
  }
  rows = static_cast<uint32_t>(out_numel);
  length = static_cast<uint32_t>(in_shape.back());
  return true;
}

} // namespace

std::optional<Tensor> rowSumShader(const Tensor& self,
                                   const std::vector<int64_t>& out_shape,
                                   float divisor) {
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, DType::Float32, out_shape, rows, length)) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed("rowsum_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchRowReduce(pso, out_buf, in_buf, rows, length, divisor);
  return output;
}

std::optional<Tensor> softmaxRowShader(const Tensor& self, int64_t dim,
                                       DType out_dtype) {
  if (!self.device().is_mps() || self.dtype() != DType::Float32 ||
      out_dtype != DType::Float32) {
    return std::nullopt;
  }
  // The kernel walks one row after another through storage, so the operand has
  // to be that run from its start.
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return std::nullopt;
  }
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  if (shape.empty()) {
    return std::nullopt;
  }
  const int64_t nd = static_cast<int64_t>(shape.size());
  if (dim < 0) {
    dim += nd;
  }
  if (dim != nd - 1) {
    return std::nullopt;
  }
  const int64_t length64 = shape.back();
  const int64_t numel = self.numel();
  if (length64 <= 0 || numel <= 0 ||
      length64 > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  const int64_t rows64 = numel / length64;
  if (rows64 > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  const uint32_t rows = static_cast<uint32_t>(rows64);
  const uint32_t length = static_cast<uint32_t>(length64);
  MTLComputePipelineState_t pso = shaderPipelineNamed("softmax_row_float");
  if (!pso) {
    return std::nullopt;
  }
  // A threadgroup caches one row per SIMD group in it. As many groups as the
  // threadgroup memory budget seats run together; a row too long for one group
  // to cache is left to the graph.
  uint32_t groups =
      std::min(kRowReduceThreads,
               static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup])) /
      kRowReduceSimdgroup;
  const uint64_t cache_limit =
      static_cast<uint64_t>([[pso device] maxThreadgroupMemoryLength]);
  while (groups > 1 &&
         static_cast<uint64_t>(groups) * length * sizeof(float) >
             cache_limit) {
    groups >>= 1;
  }
  if (groups == 0 ||
      static_cast<uint64_t>(groups) * length * sizeof(float) >
          cache_limit) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  const uint32_t threads = groups * kRowReduceSimdgroup;
  const uint32_t threadgroups = (rows + groups - 1) / groups;
  const std::array<uint32_t, 2> sizes = {rows, length};
  const NSUInteger cache_bytes =
      static_cast<NSUInteger>(groups) * length * sizeof(float);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:1];
      [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:2];
      [encoder setThreadgroupMemoryLength:cache_bytes atIndex:0];
      [encoder dispatchThreads:MTLSizeMake(threadgroups * threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
  });
  return output;
}

std::optional<Tensor> logsumexpRowShader(const Tensor& self,
                                         const std::vector<int64_t>& out_shape) {
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, DType::Float32, out_shape, rows, length)) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed("logsumexp_row_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchRowReduce(pso, out_buf, in_buf, rows, length, std::nullopt);
  return output;
}

std::optional<Tensor> rowArgReduceShader(const Tensor& self,
                                         const std::vector<int64_t>& out_shape,
                                         bool is_max) {
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, DType::Float32, out_shape, rows, length)) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed(
      is_max ? "argmax_float" : "argmin_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Int64, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchRowReduce(pso, out_buf, in_buf, rows, length, std::nullopt);
  return output;
}

std::optional<Tensor> rowExtremumShader(const Tensor& self,
                                        const std::vector<int64_t>& out_shape,
                                        bool is_max) {
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, DType::Float32, out_shape, rows, length)) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed(
      is_max ? "rowmax_float" : "rowmin_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchRowReduce(pso, out_buf, in_buf, rows, length, std::nullopt);
  return output;
}

namespace {

// A reduce-everything call over one long row would occupy a single group in
// the per-row kernels. Past this many elements the operand is split into
// chunks, one group per chunk, and the partials are folded by a second
// dispatch; below it one group already reads the row fast enough that the
// extra launch would not pay.
constexpr uint32_t kChunkedRowMin = 32768;
constexpr uint32_t kChunkedRowChunk = 8192;

// The span of a reduce-everything call: one unit-stride run of floats and a
// result that holds one value.
bool wholeRowSpan(const Tensor& self, DType in_dtype,
                  const std::vector<int64_t>& out_shape, uint32_t& numel) {
  if (!self.device().is_mps() || self.dtype() != in_dtype) {
    return false;
  }
  if (!self.is_contiguous() || self.storage_offset() != 0) {
    return false;
  }
  int64_t out_numel = 1;
  for (int64_t size : out_shape) {
    out_numel *= size;
  }
  const int64_t elements = self.numel();
  if (out_numel != 1 || elements < kChunkedRowMin ||
      elements > std::numeric_limits<uint32_t>::max() - kChunkedRowChunk) {
    return false;
  }
  numel = static_cast<uint32_t>(elements);
  return true;
}

// Encodes one group per chunk over `numel` consecutive elements, writing one
// partial per chunk.
void dispatchChunkReduce(MTLComputePipelineState_t pso, void* out_buf,
                         void* in_buf, uint32_t numel, uint32_t chunk,
                         uint32_t groups) {
  const uint32_t threads =
      std::min(kRowReduceThreads,
               static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]) /
                   kRowReduceSimdgroup * kRowReduceSimdgroup);
  const uint32_t groups_per_threadgroup = threads / kRowReduceSimdgroup;
  const uint32_t threadgroups =
      (groups + groups_per_threadgroup - 1) / groups_per_threadgroup;
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:1];
      [encoder setBytes:&numel length:sizeof(numel) atIndex:2];
      [encoder setBytes:&chunk length:sizeof(chunk) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(threadgroups * threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
  });
}

} // namespace

std::optional<Tensor> chunkedRowSumShader(const Tensor& self,
                                          const std::vector<int64_t>& out_shape,
                                          float divisor) {
  uint32_t numel = 0;
  if (!wholeRowSpan(self, DType::Float32, out_shape, numel)) {
    return std::nullopt;
  }
  const uint32_t groups =
      p10::metal::ceil_div<uint32_t>(numel, kChunkedRowChunk);
  MTLComputePipelineState_t chunk_pso = shaderPipelineNamed("sumchunk_float");
  MTLComputePipelineState_t fold_pso = shaderPipelineNamed("rowsum_float");
  if (!chunk_pso || !fold_pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  // The partials live only until the two launches are encoded; the serial
  // queue keeps any later reuse of their block behind both of them.
  Tensor partials = emptyMPS({static_cast<int64_t>(groups)}, DType::Float32,
                             self.device());
  void* partial_buf = getMTLBufferFromPointer(partials.data_ptr());
  if (!partial_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchChunkReduce(chunk_pso, partial_buf, in_buf, numel, kChunkedRowChunk,
                      groups);
  dispatchRowReduce(fold_pso, out_buf, partial_buf, 1, groups, divisor);
  return output;
}

std::optional<Tensor> chunkedRowExtremumShader(
    const Tensor& self, const std::vector<int64_t>& out_shape, bool is_max) {
  uint32_t numel = 0;
  if (!wholeRowSpan(self, DType::Float32, out_shape, numel)) {
    return std::nullopt;
  }
  const uint32_t groups =
      p10::metal::ceil_div<uint32_t>(numel, kChunkedRowChunk);
  MTLComputePipelineState_t chunk_pso =
      shaderPipelineNamed(is_max ? "maxchunk_float" : "minchunk_float");
  MTLComputePipelineState_t fold_pso =
      shaderPipelineNamed(is_max ? "rowmax_float" : "rowmin_float");
  if (!chunk_pso || !fold_pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor partials = emptyMPS({static_cast<int64_t>(groups)}, DType::Float32,
                             self.device());
  void* partial_buf = getMTLBufferFromPointer(partials.data_ptr());
  if (!partial_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Float32, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchChunkReduce(chunk_pso, partial_buf, in_buf, numel, kChunkedRowChunk,
                      groups);
  dispatchRowReduce(fold_pso, out_buf, partial_buf, 1, groups, std::nullopt);
  return output;
}

std::optional<std::tuple<Tensor, Tensor>> rowValueIndexShader(
    const Tensor& self, const std::vector<int64_t>& out_shape, bool is_max) {
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, DType::Float32, out_shape, rows, length)) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso = shaderPipelineNamed(
      is_max ? "rowmaxidx_float" : "rowminidx_float");
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor values = emptyMPS(out_shape, DType::Float32, self.device());
  Tensor indices = emptyMPS(out_shape, DType::Int64, self.device());
  void* value_buf = getMTLBufferFromPointer(values.data_ptr());
  void* index_buf = getMTLBufferFromPointer(indices.data_ptr());
  if (!value_buf || !index_buf) {
    return std::nullopt;
  }
  dispatchRowValueIndex(pso, value_buf, index_buf, in_buf, rows, length);
  return std::make_tuple(values, indices);
}

std::optional<Tensor> rowPredicateShader(
    const Tensor& self, const std::vector<int64_t>& out_shape, bool is_all) {
  // A row of either type is read as so many nonzero elements, which is what a
  // test against zero of the operand's own type answers.
  const char* type_name = nullptr;
  if (self.dtype() == DType::Float32) {
    type_name = "float";
  } else if (self.dtype() == DType::Bool) {
    type_name = "bool";
  } else {
    return std::nullopt;
  }
  uint32_t rows = 0;
  uint32_t length = 0;
  if (!rowReduceSpan(self, self.dtype(), out_shape, rows, length)) {
    return std::nullopt;
  }
  const std::string name =
      std::string(is_all ? "rowall_" : "rowany_") + type_name;
  MTLComputePipelineState_t pso = shaderPipelineNamed(name);
  if (!pso) {
    return std::nullopt;
  }
  void* in_buf = getMTLBufferFromPointer(self.data_ptr());
  if (!in_buf) {
    return std::nullopt;
  }
  Tensor output = emptyMPS(out_shape, DType::Bool, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }
  dispatchRowReduce(pso, out_buf, in_buf, rows, length, std::nullopt);
  return output;
}

std::optional<Tensor> compareShader(const std::string& op, const Tensor& self,
                                    const Tensor& other) {
  if (!self.device().is_mps() || !other.device().is_mps()) {
    return std::nullopt;
  }
  // A thread reads both operands at its own index, which addresses the right
  // pair only when the two share a shape and a storage type and each runs
  // through its storage in the order the output is written in, from its start.
  // A single-element `other` applies to every element of `self`, so it stays on
  // the same path with the scalar read once per thread.
  const std::vector<int64_t> shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const bool same_shape =
      shape == static_cast<std::vector<int64_t>>(other.shape());
  const bool scalar_rhs = !same_shape && other.numel() == 1;
  if (self.dtype() != other.dtype() || !self.is_contiguous() ||
      !other.is_contiguous() || self.storage_offset() != 0 ||
      other.storage_offset() != 0 || (!same_shape && !scalar_rhs)) {
    return std::nullopt;
  }
  const int64_t numel = self.numel();
  if (numel <= 0 || numel > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  MTLComputePipelineState_t pso =
      compareShaderPipeline(op, self.dtype(), scalar_rhs);
  if (!pso) {
    return std::nullopt;
  }
  void* self_buf = getMTLBufferFromPointer(self.data_ptr());
  void* other_buf = getMTLBufferFromPointer(other.data_ptr());
  if (!self_buf || !other_buf) {
    return std::nullopt;
  }
  // A comparison answers one byte per element whatever the operands hold, which
  // is the width the boolean storage uses.
  Tensor output = emptyMPS(shape, DType::Bool, self.device());
  void* out_buf = getMTLBufferFromPointer(output.data_ptr());
  if (!out_buf) {
    return std::nullopt;
  }

  const uint32_t elements = static_cast<uint32_t>(numel);
  // The four-wide form moves one 16-byte read per operand and one 4-byte store
  // per four predicates. The scalar operand is read from its single element, so
  // only the operand and the result need to qualify for the vector form.
  bool vec_ok = false;
  if (self.dtype() == DType::Float32 && elements % 4 == 0) {
    vec_ok = scalar_rhs
                 ? wholeFloatVectors(self.dtype(), elements,
                                     {self.data_ptr(), output.data_ptr()})
                 : wholeFloatVectors(self.dtype(), elements,
                                     {self.data_ptr(), other.data_ptr(),
                                      output.data_ptr()});
  }
  MTLComputePipelineState_t vec_pso =
      vec_ok ? shaderPipelineNamed(std::string("compare_vec_") + op +
                                   (scalar_rhs ? "_scalar_float" : "_float"))
             : nil;
  MTLComputePipelineState_t active_pso = vec_pso ? vec_pso : pso;
  // The vector form counts 16-byte vectors, the scalar form elements.
  const uint32_t span = vec_pso ? elements / 4 : elements;
  const uint32_t total =
      vec_pso ? denseSpanThreads(span, elements, /*grid_stride=*/true)
              : elements;
  const uint32_t per_group = denseThreadsPerGroup(
      total, elements,
      static_cast<uint32_t>([active_pso maxTotalThreadsPerThreadgroup]));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:active_pso];
      [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:0];
      [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf offset:0 atIndex:1];
      [encoder setBuffer:(__bridge id<MTLBuffer>)other_buf offset:0 atIndex:2];
      if (vec_pso) {
        [encoder setBytes:&span length:sizeof(span) atIndex:3];
      }
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
    }
  });
  return output;
}

Tensor unaryOp(const std::string& op, const Tensor& self,
               MPSUnaryOpBlock block,
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

  const std::string key =
      graphKey(op, {shapeKey(sizes), dtypeKey(self_c.dtype()),
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

  // Three operands that already share the broadcast shape and run unit-stride
  // need no stride arithmetic: one thread selects one element directly.
  if (condition.dtype() == DType::Bool && self.dtype() == other.dtype() &&
      cond_shape == out_shape && self_shape == out_shape &&
      other_shape == out_shape && condition.is_contiguous() &&
      self.is_contiguous() && other.is_contiguous() &&
      condition.storage_offset() == 0 && self.storage_offset() == 0 &&
      other.storage_offset() == 0) {
    const int64_t numel = self.numel();
    if (numel > 0 && numel <= std::numeric_limits<uint32_t>::max()) {
      MTLComputePipelineState_t pso = shaderPipelineNamed(
          "where_dense_" + scalarToMetalTypeString(common));
      void* cond_buf = getMTLBufferFromPointer(condition.data_ptr());
      void* self_buf = getMTLBufferFromPointer(self.data_ptr());
      void* other_buf = getMTLBufferFromPointer(other.data_ptr());
      if (pso && cond_buf && self_buf && other_buf) {
        Tensor output = emptyMPS(out_shape, common, self.device());
        void* out_buf = getMTLBufferFromPointer(output.data_ptr());
        if (out_buf) {
          const uint32_t total = static_cast<uint32_t>(numel);
          MPSStream* stream = getCurrentMPSStream();
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              auto encoder = stream->commandEncoder();
              [encoder setComputePipelineState:pso];
              [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)cond_buf
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)other_buf
                          offset:0
                         atIndex:3];
              const uint32_t per_group = std::min(
                  total, static_cast<uint32_t>(
                             [pso maxTotalThreadsPerThreadgroup]));
              [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
            }
          });
          return output;
        }
      }
    }
  }

  // Selection over broadcast operands is one strided read per operand, which
  // the shader does directly; a missing kernel or an address outside the
  // shader's 32-bit range stays with the graph.
  if (condition.dtype() == DType::Bool && self.dtype() == other.dtype() &&
      out_shape.size() <= kShaderMaxNdims &&
      condition.storage_offset() == 0 && self.storage_offset() == 0 &&
      other.storage_offset() == 0) {
    int64_t numel = 1;
    for (int64_t size : out_shape) {
      numel *= size;
    }
    if (numel > 0 && numel <= std::numeric_limits<uint32_t>::max()) {
      MTLComputePipelineState_t pso = shaderPipelineNamed(
          "broadcast_where_" + scalarToMetalTypeString(common));
      auto cond_strides = broadcastShaderStrides(condition, out_shape);
      auto self_strides = broadcastShaderStrides(self, out_shape);
      auto other_strides = broadcastShaderStrides(other, out_shape);
      void* cond_buf = getMTLBufferFromPointer(condition.data_ptr());
      void* self_buf = getMTLBufferFromPointer(self.data_ptr());
      void* other_buf = getMTLBufferFromPointer(other.data_ptr());
      if (pso && cond_strides && self_strides && other_strides && cond_buf &&
          self_buf && other_buf) {
        std::array<uint32_t, kShaderMaxNdims> sizes{};
        for (size_t o = 0; o < out_shape.size(); ++o) {
          sizes[out_shape.size() - 1 - o] =
              static_cast<uint32_t>(out_shape[o]);
        }
        const uint32_t ndim = static_cast<uint32_t>(out_shape.size());
        const uint32_t total = static_cast<uint32_t>(numel);
        Tensor output = emptyMPS(out_shape, common, self.device());
        void* out_buf = getMTLBufferFromPointer(output.data_ptr());
        if (out_buf) {
          MPSStream* stream = getCurrentMPSStream();
          dispatch_sync_with_rethrow(stream->queue(), ^() {
            @autoreleasepool {
              auto encoder = stream->commandEncoder();
              [encoder setComputePipelineState:pso];
              [encoder setBuffer:(__bridge id<MTLBuffer>)out_buf
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)cond_buf
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)self_buf
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)other_buf
                          offset:0
                         atIndex:3];
              [encoder setBytes:&ndim length:sizeof(ndim) atIndex:4];
              [encoder setBytes:sizes.data() length:sizeof(sizes) atIndex:5];
              [encoder setBytes:cond_strides->data()
                         length:sizeof(*cond_strides)
                        atIndex:6];
              [encoder setBytes:self_strides->data()
                         length:sizeof(*self_strides)
                        atIndex:7];
              [encoder setBytes:other_strides->data()
                         length:sizeof(*other_strides)
                        atIndex:8];
              const uint32_t per_group = std::min(
                  total, static_cast<uint32_t>(
                             [pso maxTotalThreadsPerThreadgroup]));
              [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(per_group, 1, 1)];
            }
          });
          return output;
        }
      }
    }
  }

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
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

} // namespace mps
} // namespace tensorplay