#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
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

// Same layout as the Conv2DParams struct in CooperativeOps.metal.
struct MpsConv2DParams {
  int32_t N;
  int32_t C_in;
  int32_t C_out;
  int32_t H;
  int32_t W;
  int32_t outH;
  int32_t outW;
  int32_t kH;
  int32_t kW;
  int32_t sH;
  int32_t sW;
  int32_t padH;
  int32_t padW;
  int32_t dH;
  int32_t dW;
  int32_t C_in_per_group;
  int32_t C_out_per_group;
  bool has_bias;
};

// Same layout as the ConvWeightPermuteParams struct in CooperativeOps.metal.
struct MpsConvWeightPermuteParams {
  uint32_t output_channels;
  uint32_t input_channels_per_group;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t output_channel_stride;
  uint32_t input_channel_stride;
  uint32_t height_stride;
  uint32_t width_stride;
};

// Attempts a cooperative-tensor convolution (M5+, macOS 26.2+).  Only the
// instantiated configurations (kernel 1x1/2x2/3x3, stride 1, dilation 1,
// groups 1, float/half) run here; anything else falls back to MPSGraph.
bool try_mpp_conv2d(const Tensor& input, const Tensor& weight,
                    const Tensor& bias, Tensor& output,
                    const MpsConv2DParams& params) {
  if (!has_mpp()) {
    return false;
  }
  const DType dt = input.dtype();
  if (dt != DType::Float32 && dt != DType::Float16) {
    return false;
  }
  const int32_t kH = params.kH;
  const int32_t kW = params.kW;
  if (params.sH != 1 || params.sW != 1 || params.dH != 1 ||
      params.dW != 1 || params.C_in_per_group != params.C_in) {
    return false;
  }
  if (kH != kW || (kH != 1 && kH != 2 && kH != 3)) {
    return false;
  }
  const char* cname = nullptr;
  int32_t src_channels = 0;
  switch (params.C_in) {
    case 1: cname = "c1"; src_channels = 1; break;
    case 3: cname = "c3"; src_channels = 3; break;
    case 16: cname = "c16"; src_channels = 16; break;
    case 32: cname = "c32"; src_channels = 32; break;
    case 64: cname = "c64"; src_channels = 64; break;
    default: return false;
  }
  const std::string dtype_str = scalarToMetalTypeString(dt);
  const std::string kernel =
      "conv2d_mpp_" + dtype_str + "_b32_w8_h8_s2_k" +
      std::to_string(kH) + std::to_string(kW) + "_s11_d11_c" + cname +
      (params.has_bias ? "_bias" : "_nobias");
  auto& lib = MetalShaderLibrary::getBundledLibrary();
  if (!lib.hasFunction(kernel)) {
    return false;
  }
  auto pso = lib.getPipelineStateForFunc(kernel);

  // NCHW -> NHWC activation copy.
  const int64_t spatial = static_cast<int64_t>(params.H) * params.W;
  Tensor nhwc = Tensor({params.N, params.H, params.W, params.C_in}, dt,
                       input.device());
  const std::string nchw_kernel = "nchw_to_nhwc_" + dtype_str;
  auto nchw_pso = lib.getPipelineStateForFunc(nchw_kernel);
  const uint32_t channels_u = static_cast<uint32_t>(params.C_in);
  const uint32_t spatial_u = static_cast<uint32_t>(spatial);

  // OIHW -> HWIO weight copy.
  Tensor hwio = Tensor({kH, kW, params.C_in, params.C_out}, dt,
                       weight.device());
  const std::string hwio_kernel = "conv_weight_to_hwio_" + dtype_str;
  auto hwio_pso = lib.getPipelineStateForFunc(hwio_kernel);
  MpsConvWeightPermuteParams wparams;
  wparams.output_channels = static_cast<uint32_t>(params.C_out);
  wparams.input_channels_per_group =
      static_cast<uint32_t>(params.C_in_per_group);
  wparams.kernel_height = static_cast<uint32_t>(kH);
  wparams.kernel_width = static_cast<uint32_t>(kW);
  wparams.output_channel_stride =
      static_cast<uint32_t>(weight.stride(0));
  wparams.input_channel_stride =
      static_cast<uint32_t>(weight.stride(1));
  wparams.height_stride = static_cast<uint32_t>(weight.stride(2));
  wparams.width_stride = static_cast<uint32_t>(weight.stride(3));

  MPSStream* stream = getCurrentMPSStream();
  run_serialized(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:nchw_pso];
      mtl_setArgs(encoder, input, nhwc, channels_u, spatial_u);
      [encoder dispatchThreads:MTLSizeMake(spatial_u, channels_u,
                                          static_cast<NSUInteger>(params.N))
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder setComputePipelineState:hwio_pso];
      mtl_setArgs(encoder, weight, hwio, wparams);
      [encoder dispatchThreads:MTLSizeMake(
                                   static_cast<NSUInteger>(params.C_out),
                                   static_cast<NSUInteger>(params.C_in),
                                   static_cast<NSUInteger>(kH))
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, nhwc, hwio, output, params,
                  params.has_bias ? bias : input);
      const uint32_t o_tiles =
          (static_cast<uint32_t>(params.C_out) + 31) / 32;
      const uint32_t w_tiles =
          (static_cast<uint32_t>(params.outW) + 7) / 8;
      const uint32_t h_tiles =
          (static_cast<uint32_t>(params.outH) + 7) / 8;
      [encoder dispatchThreadgroups:MTLSizeMake(o_tiles, w_tiles,
                                              h_tiles * params.N)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
  });
  return true;
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

  MpsConv2DParams params;
  params.N = static_cast<int32_t>(N);
  params.C_in = static_cast<int32_t>(C);
  params.C_out = static_cast<int32_t>(O);
  params.H = static_cast<int32_t>(H);
  params.W = static_cast<int32_t>(W);
  params.outH = static_cast<int32_t>(outH);
  params.outW = static_cast<int32_t>(outW);
  params.kH = static_cast<int32_t>(kH);
  params.kW = static_cast<int32_t>(kW);
  params.sH = static_cast<int32_t>(sH);
  params.sW = static_cast<int32_t>(sW);
  params.padH = static_cast<int32_t>(pH);
  params.padW = static_cast<int32_t>(pW);
  params.dH = static_cast<int32_t>(dH);
  params.dW = static_cast<int32_t>(dW);
  params.C_in_per_group = static_cast<int32_t>(C / groups);
  params.C_out_per_group = static_cast<int32_t>(O / groups);
  params.has_bias = has_bias;
  if (try_mpp_conv2d(input_c, weight_c, bias_c, output, params)) {
    return output;
  }

  const std::vector<int64_t> input_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  const std::vector<int64_t> weight_shape =
      static_cast<std::vector<int64_t>>(weight.shape());
  // One graph per shape and convolution configuration; the buffers behind its
  // placeholders are swapped on every call, so a repeated call only feeds and
  // encodes the graph.
  const std::string key = graphKey(
      "conv2d",
      {shapeKey(input_shape), shapeKey(weight_shape),
       dtypeKey(input_c.dtype()), dtypeKey(weight_c.dtype()),
       std::to_string(sH), std::to_string(sW), std::to_string(pH),
       std::to_string(pW), std::to_string(dH), std::to_string(dW),
       std::to_string(groups), std::string(has_bias ? "bias" : "nobias")});
  @autoreleasepool {
    @try {
    MPSShape* inShape = getMPSShape(input_shape);
    MPSShape* wShape = getMPSShape(weight_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    std::vector<int64_t> bias_shape = {1, O, 1, 1};
    MPSShape* biasShape = getMPSShape(bias_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
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
        result = [graph additionWithPrimaryTensor:conv
                                  secondaryTensor:bt
                                             name:nil];
      }
      entry = new CachedGraph{graph, [xt retain], [wt retain],
                              has_bias ? [bt retain] : nil,
                              [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
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
    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:
            inData, entry->input0, wData, entry->input1, nil];
    NSDictionary* results = @{entry->result : outData};
    if (has_bias) {
      MPSGraphTensorData* bData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(bias_c)
                                                  shape:biasShape
                                               dataType:dt] autorelease];
      feeds[entry->input2] = bData;
    }

    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
    } @catch (NSException* e) {
      std::string reason =
          [e reason] ? std::string([[e reason] UTF8String])
                     : std::string("unknown");
      throw tensorplay::RuntimeError(
          {__FILE__, __func__, __LINE__},
          tensorplay::detail::format_msg("MPS conv2d failed: ", reason));
    }
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ConvolutionKernels) {
  m.impl("conv2d", conv2d_kernel);
}

} // namespace mps
} // namespace tensorplay