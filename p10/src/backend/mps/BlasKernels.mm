#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

namespace {

// Attempts a cooperative-tensor GEMM (M5+, macOS 26.2+).  `a`/`b`/`output`
// must be contiguous row-major tensors; `m/k/n` are the 2-D GEMM dimensions
// (the batch dimension, if any, is carried by `batch`).  Returns false when
// the hardware or the bundled shader does not support the MPP path, leaving
// the caller to fall back to the MPSGraph implementation.
bool try_mpp_gemm(const Tensor& a, const Tensor& b, Tensor& output,
                  int64_t m, int64_t k, int64_t n, int64_t batch) {
  if (!has_mpp()) {
    return false;
  }
  if (a.dtype() != DType::Float32 || m <= 0 || k <= 0 || n <= 0 ||
      batch <= 0) {
    return false;
  }
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  if (m > kInt32Max || k > kInt32Max || n > kInt32Max ||
      batch > kInt32Max) {
    return false;
  }
  const std::string kernel = "gemm_mpp_float_64_64_4";
  auto& lib = MetalShaderLibrary::getBundledLibrary();
  if (!lib.hasFunction(kernel)) {
    return false;
  }
  auto pso = lib.getPipelineStateForFunc(kernel);
  constexpr uint32_t kTile = 64;
  const auto grid_x = (static_cast<uint64_t>(n) + kTile - 1) / kTile;
  const auto grid_y = (static_cast<uint64_t>(m) + kTile - 1) / kTile;
  const std::array<uint32_t, 3> sizes = {
      static_cast<uint32_t>(m), static_cast<uint32_t>(k),
      static_cast<uint32_t>(n)};
  // Row-major within-batch strides: A is M x K, B is K x N, C is M x N.
  const std::array<uint64_t, 6> strides = {
      static_cast<uint64_t>(k), 1,
      static_cast<uint64_t>(n), 1,
      static_cast<uint64_t>(n), 1};
  const uint32_t batch_u = static_cast<uint32_t>(batch);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      stream->endKernelCoalescing();
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, a, b, output, strides, sizes, batch_u);
      [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch_u)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      stream->synchronize(SyncType::COMMIT);
    }
  });
  return true;
}

// Runs an MPSGraph matrix multiplication. `a_shape`/`b_shape` describe the
// graph inputs (1-D operands are lifted to 2-D), `out_data_shape` describes
// the output buffer (the result tensor may be 1-D, e.g. matvec).
void runMatMul(const Tensor& a, const Tensor& b,
               const std::vector<int64_t>& a_shape,
               const std::vector<int64_t>& b_shape,
               const std::vector<int64_t>& out_data_shape, Tensor& output) {
  // The product graph depends only on the three shapes and the dtypes, so a
  // repeated multiply shape reuses it.
  const std::string key =
      "matmul|" + shapeKey(a_shape) + "|" + shapeKey(b_shape) + "|" +
      shapeKey(out_data_shape) + "|" + dtypeKey(a.dtype()) + "|" +
      dtypeKey(b.dtype()) + "|" + dtypeKey(output.dtype());

  @autoreleasepool {
    MPSShape* aShape = getMPSShape(a_shape);
    MPSShape* bShape = getMPSShape(b_shape);
    MPSShape* outShape = getMPSShape(out_data_shape);
    const MPSDataType a_dt = getMPSDataType(a.dtype());
    const MPSDataType b_dt = getMPSDataType(b.dtype());
    const MPSDataType out_dt = getMPSDataType(output.dtype());

    CachedGraph* entry = findCachedGraph(key);
    if (entry == nullptr) {
      MPSGraph* graph = [[MPSGraph alloc] init];
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
      entry = new CachedGraph{graph, [at retain], [bt retain], nil,
                              [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

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
    NSDictionary* feeds = @{entry->input0 : aData, entry->input1 : bData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT);
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
  if (try_mpp_gemm(a, b, output, self.size(0), self.size(1),
                   mat2.size(1), 1)) {
    return output;
  }
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
  if (try_mpp_gemm(a, b, output, self.size(1), self.size(2),
                   batch2.size(2), self.size(0))) {
    return output;
  }
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
    NSDictionary* results = @{sum : outData};
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
  if (!input.device().is_mps() || !mat1.device().is_mps() ||
      !mat2.device().is_mps()) {
    TP_THROW(RuntimeError, "addmm(): tensors must be on the MPS device");
  }
  check_matmul_dtypes(mat1, mat2);
  if (mat1.dim() != 2 || mat2.dim() != 2) {
    TP_THROW(RuntimeError, "addmm(): expected both matrices to be 2-D");
  }
  const int64_t m = mat1.size(0);
  const int64_t k = mat1.size(1);
  const int64_t n = mat2.size(1);
  if (mat2.size(0) != k) {
    TP_THROW(RuntimeError, "addmm(): shapes cannot be multiplied");
  }
  const std::vector<int64_t> out_shape = {m, n};
  const std::vector<int64_t> a_shape = {m, k};
  const std::vector<int64_t> b_shape = {k, n};
  const DType prod_dtype = mat1.dtype();
  const DType out_dtype = promoteTypes(input.dtype(), prod_dtype);

  Tensor mat1_c = materializeForGraph(mat1);
  Tensor mat2_c = materializeForGraph(mat2);
  Tensor input_c = materializeForGraph(input);
  const std::vector<int64_t> input_shape =
      static_cast<std::vector<int64_t>>(input_c.shape());
  if (input_shape != out_shape) {
    if (input_c.numel() == 1) {
      // A scalar addend has a single element behind an empty or unit shape;
      // describe it as a broadcastable one so the expansion is well-formed.
      input_c = input_c.reshape(std::vector<int64_t>{1, 1});
    }
    input_c = materializeForGraph(input_c.expand(out_shape));
  }
  Tensor output = Tensor(out_shape, out_dtype, input.device());
  if (output.numel() == 0) {
    return output;
  }

  // One graph does the product and both scalings, so the whole expression is
  // a single dispatch instead of a matrix multiply followed by three binary
  // operations. Alpha and beta are constants of the graph, hence part of the
  // key.
  const std::string key =
      "addmm|" + shapeKey(out_shape) + "|" + shapeKey(a_shape) + "|" +
      shapeKey(b_shape) + "|" + dtypeKey(input_c.dtype()) + "|" +
      dtypeKey(prod_dtype) + "|" + dtypeKey(out_dtype) + "|" +
      scalarKey(beta) + "|" + scalarKey(alpha);

  @autoreleasepool {
    MPSShape* outMPSShape = getMPSShape(out_shape);
    MPSShape* aMPSShape = getMPSShape(a_shape);
    MPSShape* bMPSShape = getMPSShape(b_shape);
    const MPSDataType in_dt = getMPSDataType(input_c.dtype());
    const MPSDataType a_dt = getMPSDataType(mat1_c.dtype());
    const MPSDataType b_dt = getMPSDataType(mat2_c.dtype());
    const MPSDataType out_dt = getMPSDataType(out_dtype);

    CachedGraph* entry = findCachedGraph(key);
    if (entry == nullptr) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* inp = [graph placeholderWithShape:outMPSShape
                                               dataType:in_dt
                                                    name:@"input"];
      MPSGraphTensor* at = [graph placeholderWithShape:aMPSShape
                                              dataType:a_dt
                                                   name:@"a"];
      MPSGraphTensor* bt = [graph placeholderWithShape:bMPSShape
                                              dataType:b_dt
                                                   name:@"b"];
      MPSGraphTensor* product =
          [graph matrixMultiplicationWithPrimaryTensor:at
                                       secondaryTensor:bt
                                                  name:@"product"];
      MPSGraphTensor* scaled_input =
          ([inp dataType] == out_dt) ? inp
                                     : [graph castTensor:inp
                                                  toType:out_dt
                                                    name:@"cast_input"];
      MPSGraphTensor* scaled_prod =
          ([product dataType] == out_dt)
          ? product
          : [graph castTensor:product toType:out_dt name:@"cast_product"];
      if (beta.toDouble() != 1.0) {
        MPSGraphTensor* betaConst = [graph constantWithScalar:beta.toDouble()
                                                     dataType:out_dt];
        scaled_input = [graph multiplicationWithPrimaryTensor:scaled_input
                                              secondaryTensor:betaConst
                                                         name:@"scale_beta"];
      }
      if (alpha.toDouble() != 1.0) {
        MPSGraphTensor* alphaConst =
            [graph constantWithScalar:alpha.toDouble()
                             dataType:out_dt];
        scaled_prod = [graph multiplicationWithPrimaryTensor:scaled_prod
                                             secondaryTensor:alphaConst
                                                        name:@"scale_alpha"];
      }
      MPSGraphTensor* result = [graph additionWithPrimaryTensor:scaled_input
                                                secondaryTensor:scaled_prod
                                                           name:@"sum"];
      entry = new CachedGraph{graph, [inp retain], [at retain], [bt retain],
                              [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
    id<MTLBuffer> aBuf = mtlBufferOf(mat1_c);
    id<MTLBuffer> bBuf = mtlBufferOf(mat2_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:outMPSShape
                                             dataType:in_dt] autorelease];
    MPSGraphTensorData* aData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:aBuf
                                                shape:aMPSShape
                                             dataType:a_dt] autorelease];
    MPSGraphTensorData* bData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:bBuf
                                                shape:bMPSShape
                                             dataType:b_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outMPSShape
                                             dataType:out_dt] autorelease];
    NSDictionary* feeds =
        @{entry->input0 : inData, entry->input1 : aData,
          entry->input2 : bData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT);
  }
  return output;
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