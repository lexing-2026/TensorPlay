#pragma once

// Private helpers shared by the MPS backend kernels.  This header is only
// included from Objective-C++ translation units; the MPSGraph types used in
// the block signatures are available there.

#include "Tensor.h"
#include "Scalar.h"
#include "Exception.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <source_location>
#include <string>

namespace tensorplay {
namespace mps {

// Maps a storage dtype to the Metal data type used by MPSGraph. Double,
// complex and the 8-bit float formats are rejected explicitly.
MPSDataType getMPSDataType(DType dtype);

// Builds an MPSShape (NSArray<NSNumber*>) from tensor sizes. A 0-d tensor is
// represented as a single-element shape because MPSGraphTensorData cannot
// describe an empty shape.
MPSShape* getMPSShape(const std::vector<int64_t>& sizes);

// Returns a contiguous, storage-offset-0 view of `t` so its contents can be
// wrapped directly in an MPSGraphTensorData (which has no offset concept).
Tensor materializeForGraph(const Tensor& t);

// Returns the MTLBuffer backing `t`. `t` must own MPS-managed storage at
// offset 0 (materializeForGraph guarantees this).
id<MTLBuffer> mtlBufferOf(const Tensor& t);

// Allocates an empty MPS tensor on `device`.
Tensor emptyMPS(const std::vector<int64_t>& sizes, DType dtype,
               const Device& device);

// A built graph and the tensors a caller feeds it with. The placeholders are
// shape- and dtype-static, so every later call with the same key reuses them
// and only swaps the buffers behind them.
struct CachedGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* input0 = nil;
  MPSGraphTensor* input1 = nil;
  MPSGraphTensor* input2 = nil;
  MPSGraphTensor* result = nil;
};

// Cache of built graphs, keyed by the structure a call would build. Callers
// look the key up, build and store on a miss, then look the key up again so a
// racing builder's entry wins.
CachedGraph* findCachedGraph(const std::string& key);
void storeCachedGraph(const std::string& key, CachedGraph* entry);

// Key fragments: a shape, a dtype, a scalar's bit pattern, and a whole key
// assembled from a call site plus fragments.
std::string shapeKey(const std::vector<int64_t>& shape);
std::string dtypeKey(DType dtype);
std::string scalarKey(const Scalar& value);
std::string graphKey(const std::source_location& location,
                     std::initializer_list<std::string> parts);

// Block building the graph body of a binary operation. Receives the graph and
// both inputs already cast to the common dtype; returns the result tensor.
using MPSBinaryOpBlock = MPSGraphTensor* (^)(MPSGraph*, MPSGraphTensor*,
                                             MPSGraphTensor*);

// Block building the graph body of a unary operation.
using MPSUnaryOpBlock = MPSGraphTensor* (^)(MPSGraph*, MPSGraphTensor*);

// Executes a broadcastable binary operation on the MPS device. `alpha`, when
// present, is folded in as `self + alpha * other` before the block runs;
// `common_dtype` overrides the promoted input dtype (used by div to force
// float output for integral inputs); `out_dtype` overrides the output dtype
// (used by comparisons to produce bool).
//
// The graph built by the block is kept and reused by later calls with the same
// shapes, dtypes and alpha, so the per-call cost is feed and encode rather than
// graph construction and compilation. The call site identifies the operation,
// which is why the location parameter is defaulted rather than passed.
Tensor binaryOp(const Tensor& self, const Tensor& other, MPSBinaryOpBlock block,
                const std::optional<Scalar>& alpha = std::nullopt,
                std::optional<DType> common_dtype = std::nullopt,
                std::optional<DType> out_dtype = std::nullopt,
                std::source_location location =
                    std::source_location::current());

// Executes a unary operation on the MPS device, preserving the input dtype
// unless `out_dtype` is given. The graph is cached as in `binaryOp`.
Tensor unaryOp(const Tensor& self, MPSUnaryOpBlock block,
               std::optional<DType> out_dtype = std::nullopt,
               std::source_location location =
                   std::source_location::current());

// Executes `where(condition, self, other)` with broadcasting and dtype
// promotion of self/other.
Tensor whereOp(const Tensor& condition, const Tensor& self,
               const Tensor& other);

} // namespace mps
} // namespace tensorplay