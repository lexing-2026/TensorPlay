#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

namespace {

std::vector<int64_t> tensor_shape(const Tensor& t) {
  return static_cast<std::vector<int64_t>>(t.shape());
}

} // namespace

// `expand` is a pure view: leading singleton dims get zero strides and any
// size-1 input dims broadcast with stride 0.  No GPU work is needed here; the
// resulting non-contiguous view is materialized by the consumer kernels.
Tensor expand_kernel(const Tensor& self, const std::vector<int64_t>& size,
                     bool /*implicit*/) {
  const int64_t ndim = self.dim();
  const int64_t new_ndim = static_cast<int64_t>(size.size());
  if (new_ndim < ndim) {
    TP_THROW(RuntimeError,
             "expand(): the number of sizes provided must be greater or "
             "equal to the number of dimensions in the tensor");
  }
  const std::vector<int64_t> self_sizes = tensor_shape(self);
  const std::vector<int64_t> self_strides = self.strides();

  std::vector<int64_t> out_sizes(size);
  std::vector<int64_t> out_strides(new_ndim, 0);
  for (int64_t i = new_ndim - 1; i >= 0; --i) {
    const int64_t offset = new_ndim - 1 - i;
    const int64_t dim = ndim - 1 - offset;
    const int64_t src_size = dim >= 0 ? self_sizes[static_cast<size_t>(dim)] : 1;
    int64_t target = out_sizes[static_cast<size_t>(i)];
    if (target == -1) {
      target = src_size;
      out_sizes[static_cast<size_t>(i)] = target;
    }
    if (src_size != target) {
      if (src_size != 1) {
        TP_THROW(RuntimeError,
                 "expand(): the expanded size (",
                 target,
                 ") must match the existing size (",
                 src_size,
                 ") at non-singleton dimension ",
                 i);
      }
      out_strides[static_cast<size_t>(i)] = 0;
    } else {
      out_strides[static_cast<size_t>(i)] =
          dim >= 0 ? self_strides[static_cast<size_t>(dim)] : 0;
    }
  }

  const int64_t storage_offset = self.storage_offset();
  return ops::as_strided(self, out_sizes, out_strides, storage_offset);
}

// `repeat` tiles the input along every dimension `repeats[i]` times.  It is
// implemented as an MPSGraph tile over the rank-extended input (leading
// singleton dims make the multiplier list length match the output rank).
Tensor repeat_kernel(const Tensor& self, const std::vector<int64_t>& repeats) {
  for (int64_t r : repeats) {
    if (r < 0) {
      TP_THROW(RuntimeError, "repeat(): repeats must be non-negative");
    }
  }
  const int64_t out_ndim = static_cast<int64_t>(repeats.size());
  const int64_t in_ndim = self.dim();
  if (out_ndim < in_ndim) {
    TP_THROW(RuntimeError,
             "repeat(): Number of dimensions of repeat dims can not be "
             "smaller than number of dimensions of tensor");
  }
  const int64_t add = out_ndim - in_ndim;
  const std::vector<int64_t> in_shape = tensor_shape(self);
  std::vector<int64_t> out_shape(out_ndim);
  for (int64_t i = 0; i < out_ndim; ++i) {
    const int64_t src = i < add ? 1 : in_shape[static_cast<size_t>(i - add)];
    out_shape[static_cast<size_t>(i)] = src * repeats[static_cast<size_t>(i)];
  }

  Tensor input = self.is_contiguous() ? self : self.contiguous();
  if (add > 0) {
    std::vector<int64_t> view_shape(static_cast<size_t>(add), 1);
    view_shape.insert(view_shape.end(), in_shape.begin(), in_shape.end());
    input = ops::reshape(input, view_shape);
    input = input.is_contiguous() ? input : input.contiguous();
  }

  Tensor output = Tensor::empty(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(tensor_shape(input));
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType dt = getMPSDataType(input.dtype());

    MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                            dataType:dt
                                                 name:@"input"];
    NSMutableArray<NSNumber*>* multipliers =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(out_ndim)];
    for (int64_t r : repeats) {
      [multipliers addObject:@(r)];
    }
    MPSGraphTensor* tiled = [graph tileTensor:xt
                                withMultiplier:multipliers
                                         name:@"tiled"];

    id<MTLBuffer> inBuf = mtlBufferOf(input);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:dt] autorelease];

    NSDictionary* feeds = @{xt : inData};
    NSDictionary* results = @{tiled : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ShapeExpandKernels) {
  m.impl("expand", expand_kernel);
  m.impl("repeat", repeat_kernel);
}

} // namespace mps
} // namespace tensorplay