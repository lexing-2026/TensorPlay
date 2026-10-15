#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

int64_t conv_output_size(int64_t input, int64_t kernel, int64_t stride,
                         int64_t padding, int64_t dilation) {
  const int64_t d =
      input + 2 * padding - dilation * (kernel - 1) - 1;
  if (d < 0) return 0;
  return d / stride + 1;
}

} // namespace

// ---------------------------------------------------------------------------
// conv2d
//
// Runs the confirmed convolution2DWithSourceTensor selector with an explicit
// NCHW/OIHW layout.  The bias is broadcast over the output as (1, C, 1, 1).
// ---------------------------------------------------------------------------

Tensor conv2d_kernel(const Tensor& input, const Tensor& weight,
                     const Tensor& bias,
                     const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding,
                     const std::vector<int64_t>& dilation, int64_t groups) {
  if (!input.device().is_mps() || !weight.device().is_mps()) {
    TP_THROW(RuntimeError, "conv2d(): input and weight must be on MPS");
  }
  if (input.dim() != 4 || weight.dim() != 4) {
    TP_THROW(RuntimeError, "conv2d(): expected 4D input and 4D weight");
  }
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  const int64_t O = weight.size(0);
  const int64_t kH = weight.size(2);
  const int64_t kW = weight.size(3);
  if (weight.size(1) * groups != C) {
    TP_THROW(RuntimeError,
             "conv2d(): weight in_channels * groups must match input "
             "channels");
  }

  const int64_t sH = stride.empty() ? 1 : stride[0];
  const int64_t sW = stride.empty() ? 1 : (stride.size() == 2 ? stride[1] : stride[0]);
  const int64_t pH = padding.empty() ? 0 : padding[0];
  const int64_t pW = padding.empty() ? 0 : (padding.size() == 2 ? padding[1] : padding[0]);
  const int64_t dH = dilation.empty() ? 1 : dilation[0];
  const int64_t dW = dilation.empty() ? 1 : (dilation.size() == 2 ? dilation[1] : dilation[0]);

  const int64_t outH = conv_output_size(H, kH, sH, pH, dH);
  const int64_t outW = conv_output_size(W, kW, sW, pW, dW);
  const bool has_bias = bias.defined();

  std::vector<int64_t> out_shape = {N, O, outH, outW};
  Tensor output = Tensor(out_shape, input.dtype(), input.device());
  if (output.numel() == 0) {
    return output;
  }

  Tensor input_c = materializeForGraph(input);
  Tensor weight_c = materializeForGraph(weight);
  Tensor bias_c;
  if (has_bias) {
    if (bias.numel() != O) {
      TP_THROW(RuntimeError,
               "conv2d(): bias must have one element per output channel");
    }
    bias_c = materializeForGraph(bias);
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(static_cast<std::vector<int64_t>>(input.shape()));
    MPSShape* wShape = getMPSShape(static_cast<std::vector<int64_t>>(weight.shape()));
    MPSShape* outShape = getMPSShape(out_shape);
    std::vector<int64_t> bias_shape = {1, O, 1, 1};
    MPSShape* biasShape = getMPSShape(bias_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                           dataType:dt
                                                name:@"x"];
    MPSGraphTensor* wt = [graph placeholderWithShape:wShape
                                    dataType:dt
                                         name:@"w"];
    MPSGraphConvolution2DOpDescriptor* desc =
        [[MPSGraphConvolution2DOpDescriptor new] autorelease];
    desc.strideInX = sW;
    desc.strideInY = sH;
    desc.dilationRateInX = dW;
    desc.dilationRateInY = dH;
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.paddingLeft = pW;
    desc.paddingRight = pW;
    desc.paddingTop = pH;
    desc.paddingBottom = pH;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    desc.groups = groups;

    MPSGraphTensor* conv =
        [graph convolution2DWithSourceTensor:xt
                            weightsTensor:wt
                               descriptor:desc
                                     name:nil];
    MPSGraphTensor* result = conv;
    MPSGraphTensor* bt = nil;
    if (has_bias) {
      bt = [graph placeholderWithShape:biasShape dataType:dt name:@"b"];
      result = [graph additionWithPrimaryTensor:conv secondaryTensor:bt name:nil];
    }

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
    id<MTLBuffer> wBuf = mtlBufferOf(weight_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* wData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:wBuf
                                                shape:wShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:dt] autorelease];
    NSDictionary* feeds = @{xt : inData, wt : wData};
    NSDictionary* results = @{result : outData};
    if (has_bias) {
      MPSGraphTensorData* bData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(bias_c)
                                                  shape:biasShape
                                               dataType:dt] autorelease];
      [(NSMutableDictionary*)feeds setObject:bData forKey:bt];
    }

    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ConvolutionKernels) {
  m.impl("conv2d", conv2d_kernel);
}

} // namespace mps
} // namespace tensorplay