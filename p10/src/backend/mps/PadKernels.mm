#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>

namespace tensorplay {
namespace mps {

// ---------------------------------------------------------------------------
// constant_pad_nd
//
// Negative paddings crop the input with slice views (mirroring the CPU
// behavior), then the remaining positive paddings are applied with the
// confirmed padTensor selector using MPSGraphPaddingModeConstant.
// ---------------------------------------------------------------------------

Tensor constant_pad_nd_kernel(const Tensor& self,
                               const std::vector<int64_t>& pad,
                               const Scalar& value) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "constant_pad_nd(): input must be on MPS");
  }
  const std::vector<int64_t> input_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t l_inp = self.dim();
  const int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;

  if (pad.size() % 2 != 0) {
    TP_THROW(ValueError, "Length of pad must be even but instead it equals ",
             pad.size());
  }
  if (l_inp < l_pad) {
    TP_THROW(ValueError,
             "Length of pad should be no more than twice the number of "
             "dimensions of the input. Pad length is ",
             pad.size(), " while the input has ", l_inp, " dimensions.");
  }

  bool all_pads_non_positive = true;
  Tensor c_input = self;
  for (int64_t i = l_inp - l_pad; i < l_inp; ++i) {
    const int64_t pad_idx = 2 * (l_inp - i - 1);
    if (pad[pad_idx] < 0) {
      c_input = c_input.slice(i, -pad[pad_idx], c_input.size(i));
    } else if (pad[pad_idx] != 0) {
      all_pads_non_positive = false;
    }
    if (pad[pad_idx + 1] < 0) {
      c_input = c_input.slice(i, 0, c_input.size(i) + pad[pad_idx + 1]);
    } else if (pad[pad_idx + 1] != 0) {
      all_pads_non_positive = false;
    }
  }

  if (all_pads_non_positive) {
    return c_input.clone();
  }

  // Positive padding counts per padded dim, in the order of the cropped
  // tensor's dimensions (last l_pad dims).
  std::vector<int64_t> left_pad(l_inp, 0);
  std::vector<int64_t> right_pad(l_inp, 0);
  std::vector<int64_t> new_shape;
  for (int64_t i = 0; i < l_inp - l_pad; ++i) {
    new_shape.push_back(c_input.size(i));
  }
  for (int64_t i = 0; i < l_pad; ++i) {
    const int64_t dim = l_inp - l_pad + i;
    const int64_t pad_idx = 2 * (l_pad - i - 1);
    left_pad[dim] = std::max<int64_t>(0, pad[pad_idx]);
    right_pad[dim] = std::max<int64_t>(0, pad[pad_idx + 1]);
    const int64_t new_dim =
        c_input.size(dim) + left_pad[dim] + right_pad[dim];
    if (new_dim < 0) {
      TP_THROW(ValueError, "The input size ", input_sizes[l_inp - l_pad + i],
               ", plus negative padding ", pad[pad_idx], " and ",
               pad[pad_idx + 1],
               " resulted in a negative output size, which is invalid. Check "
               "dimension ",
               l_inp - l_pad + i, " of your input.");
    }
    new_shape.push_back(new_dim);
  }

  Tensor output = Tensor(new_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }

  Tensor input_c = materializeForGraph(c_input);
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    const std::vector<int64_t> in_shape =
        static_cast<std::vector<int64_t>>(c_input.shape());
    MPSShape* inShape = getMPSShape(in_shape);
    MPSShape* outShape = getMPSShape(new_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                           dataType:dt
                                                name:@"x"];
    NSMutableArray<NSNumber*>* leftArr =
        [NSMutableArray arrayWithCapacity:left_pad.size()];
    NSMutableArray<NSNumber*>* rightArr =
        [NSMutableArray arrayWithCapacity:right_pad.size()];
    for (int64_t i = 0; i < l_inp; ++i) {
      [leftArr addObject:@(left_pad[i])];
      [rightArr addObject:@(right_pad[i])];
    }
    MPSGraphTensor* padded =
        [graph padTensor:xt
        withPaddingMode:MPSGraphPaddingModeConstant
           leftPadding:leftArr
          rightPadding:rightArr
         constantValue:value.toDouble()
                   name:nil];

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
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
    NSDictionary* results = @{padded : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, PadKernels) {
  m.impl("constant_pad_nd", constant_pad_nd_kernel);
}

} // namespace mps
} // namespace tensorplay