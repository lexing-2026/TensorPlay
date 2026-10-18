#pragma once

// Private helpers shared by the MPS backend kernels.  This header is only
// included from Objective-C++ translation units; the MPSGraph types used in
// the block signatures are available there.

#include "Tensor.h"
#include "Scalar.h"
#include "Exception.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

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
// and only swaps the buffers behind them. `result` is the only output unless
// the operation writes two tensors from one dispatch, which `result2` carries.
struct CachedGraph {
  MPSGraph* graph = nil;
  MPSGraphTensor* input0 = nil;
  MPSGraphTensor* input1 = nil;
  MPSGraphTensor* input2 = nil;
  MPSGraphTensor* result = nil;
  MPSGraphTensor* result2 = nil;
};

// Cache of built graphs, keyed by the structure a call would build. Callers
// look the key up, build and store on a miss, then look the key up again so a
// racing builder's entry wins.
CachedGraph* findCachedGraph(const std::string& key);
void storeCachedGraph(const std::string& key, CachedGraph* entry);

// Key fragments: a shape, a dtype, a scalar's bit pattern, and a whole key
// assembled from an operation name plus fragments.
std::string shapeKey(const std::vector<int64_t>& shape);
std::string dtypeKey(DType dtype);
std::string scalarKey(const Scalar& value);
std::string graphKey(const std::string& op,
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
// graph construction and compilation. `op` names the computation the block
// performs and starts the cache key: two calls that share a name are read as
// building the same graph, so a block whose body depends on caller values must
// be given a name carrying them (graphKey appends them).
Tensor binaryOp(const std::string& op, const Tensor& self,
                const Tensor& other, MPSBinaryOpBlock block,
                const std::optional<Scalar>& alpha = std::nullopt,
                std::optional<DType> common_dtype = std::nullopt,
                std::optional<DType> out_dtype = std::nullopt);

// Executes a unary operation on the MPS device, preserving the input dtype
// unless `out_dtype` is given. The graph is cached as in `binaryOp`, so `op`
// names the computation and carries any value baked into the block body.
Tensor unaryOp(const std::string& op, const Tensor& self,
               MPSUnaryOpBlock block,
               std::optional<DType> out_dtype = std::nullopt);

// Executes `where(condition, self, other)` with broadcasting and dtype
// promotion of self/other.
Tensor whereOp(const Tensor& condition, const Tensor& self,
               const Tensor& other);

// Runs `op` -- "add", "sub" or "mul" -- over broadcast operands with the
// stride-addressed shader, which reads each operand out of the storage it
// already has instead of materializing an expanded copy. Returns std::nullopt
// when a call cannot be expressed that way: operands that need no expansion (a
// graph is then one dispatch either way), a dtype or an alpha the shader is not
// instantiated for, a non-zero storage offset, a negative stride, a rank or an
// element index beyond the 32-bit addressing the shader uses. The caller builds
// a graph in that case.
std::optional<Tensor> broadcastBinaryShader(
    const std::string& op, const Tensor& self, const Tensor& other,
    const std::optional<Scalar>& alpha = std::nullopt);

// Runs `op` -- "exp", "sqrt", "rsqrt", "abs" or "relu" -- over a contiguous
// float input with a single shader dispatch, which is what the operation costs
// in storage traffic and nothing more. Returns std::nullopt when a call cannot
// be expressed that way: a dtype the shader is not instantiated for, a view
// that is not contiguous or does not start at the beginning of its storage, an
// element count beyond the 32-bit addressing the shader uses. The caller builds
// a graph in that case.
std::optional<Tensor> unaryShader(const std::string& op, const Tensor& self);

// Reduces the innermost axis of a contiguous float operand with the row shader
// and returns the result in `out_shape`, which has to hold one value per row.
// One SIMD group takes one row, so the call is a single dispatch and needs no
// threadgroup memory: nothing crosses a row boundary. `divisor` above zero
// divides each row's accumulated value on the way out, which is how an average
// costs no more than a sum. Returns std::nullopt for an operand the shader
// cannot address, as above.
std::optional<Tensor> rowSumShader(const Tensor& self,
                                   const std::vector<int64_t>& out_shape,
                                   float divisor);

// The position along the innermost axis at which `self` holds its extremum --
// the first such position when several hold it, and a NaN counts as holding the
// extremum of its row -- written as Int64 in `out_shape`. Returns std::nullopt
// for an operand the shader cannot address, as above.
std::optional<Tensor> rowArgReduceShader(const Tensor& self,
                                         const std::vector<int64_t>& out_shape,
                                         bool is_max);

// The extremum along the innermost axis of a contiguous float operand, written
// in `out_shape`: the value alone where a reduction asks for one, and the value
// with the first position holding it where a reduction asks for both. One pass
// answers a paired reduction, so the position an extremum names and the
// extremum a position came from describe the same element. Returns
// std::nullopt for an operand the shader cannot address, as above.
std::optional<Tensor> rowExtremumShader(const Tensor& self,
                                        const std::vector<int64_t>& out_shape,
                                        bool is_max);
std::optional<std::tuple<Tensor, Tensor>> rowValueIndexShader(
    const Tensor& self, const std::vector<int64_t>& out_shape, bool is_max);

// Whether every element of the innermost axis of `self`, or at least one of
// them, is nonzero, written as one boolean byte per row. Float and bool
// operands are both read where they stand. Returns std::nullopt for an operand
// the shader cannot address, as above.
std::optional<Tensor> rowPredicateShader(const Tensor& self,
                                         const std::vector<int64_t>& out_shape,
                                         bool is_all);

// Compares `self` and `other` -- "eq", "ne", "lt", "le", "gt" or "ge" -- with a
// single shader dispatch that writes one boolean byte per element. The operands
// are read where they stand, so nothing is materialized for the comparison.
// Returns std::nullopt when a call cannot be expressed that way: operands of
// different shapes or storage types, a view that is not contiguous or does not
// start at the beginning of its storage, a dtype the shader is not instantiated
// for, an element count beyond the 32-bit addressing the shader uses. The caller
// builds a graph in that case.
std::optional<Tensor> compareShader(const std::string& op, const Tensor& self,
                                    const Tensor& other);

} // namespace mps
} // namespace tensorplay