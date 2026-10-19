#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) {
    --q;
  }
  return q;
}

int64_t ceil_div(int64_t a, int64_t b) {
  return -floor_div(-a, b);
}

int64_t pool_output_size(int64_t input, int64_t kernel, int64_t stride,
                         int64_t pad, bool ceil_mode) {
  const int64_t d = input + 2 * pad - kernel;
  int64_t out =
      ceil_mode ? ceil_div(d, stride) + 1 : floor_div(d, stride) + 1;
  if (ceil_mode && ((out - 1) * stride >= input + pad)) {
    --out;
  }
  return out;
}

void normalize_pool_params(const std::vector<int64_t>& kernel,
                           const std::vector<int64_t>& stride,
                           const std::vector<int64_t>& padding,
                           int64_t& kH, int64_t& kW, int64_t& dH, int64_t& dW,
                           int64_t& padH, int64_t& padW) {
  kH = kernel[0];
  kW = kernel.size() == 2 ? kernel[1] : kernel[0];
  dH = stride.empty() ? kH : stride[0];
  dW = stride.empty() ? kW : (stride.size() == 2 ? stride[1] : stride[0]);
  padH = padding.empty() ? 0 : padding[0];
  padW = padding.empty() ? 0 : (padding.size() == 2 ? padding[1] : padding[0]);
}

MPSGraphPooling2DOpDescriptor* poolingDescriptor(
    int64_t kH, int64_t kW, int64_t dH, int64_t dW, int64_t padH,
    int64_t padW, bool ceil_mode, bool count_include_pad) {
  MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:kW
                   kernelHeight:kH
                      strideInX:dW
                      strideInY:dH
                dilationRateInX:1
                dilationRateInY:1
                    paddingLeft:padW
                   paddingRight:padW
                     paddingTop:padH
                  paddingBottom:padH
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  desc.ceilMode = ceil_mode;
  desc.includeZeroPadToAverage = count_include_pad;
  return desc;
}

} // namespace

// ---------------------------------------------------------------------------
// avg_pool2d
// ---------------------------------------------------------------------------

Tensor avg_pool2d_kernel(const Tensor& input,
                         const std::vector<int64_t>& kernel_size,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding, bool ceil_mode,
                         bool count_include_pad,
                         std::optional<int64_t> divisor_override) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError, "avg_pool2d(): input must be on the MPS device");
  }
  if (input.dim() != 3 && input.dim() != 4) {
    TP_THROW(RuntimeError, "avg_pool2d(): Expected 3D or 4D input");
  }
  const bool batched = input.dim() == 4;
  Tensor x = batched ? input : input.unsqueeze(0);
  const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

  int64_t kH = 0, kW = 0, dH = 0, dW = 0, padH = 0, padW = 0;
  normalize_pool_params(kernel_size, stride, padding, kH, kW, dH, dW, padH,
                        padW);
  const int64_t outH = pool_output_size(H, kH, dH, padH, ceil_mode);
  const int64_t outW = pool_output_size(W, kW, dW, padW, ceil_mode);

  const bool use_divisor = divisor_override.has_value() &&
                           divisor_override.value() != 0;
  const double divisor_scale =
      use_divisor
          ? static_cast<double>(kH * kW) / static_cast<double>(*divisor_override)
          : 1.0;

  std::vector<int64_t> out_shape = {N, C, outH, outW};
  Tensor output = Tensor(out_shape, x.dtype(), x.device());
  if (output.numel() == 0) {
    return batched ? output : output.squeeze(0);
  }

  Tensor x_c = materializeForGraph(x);
  const std::vector<int64_t> x_shape =
      static_cast<std::vector<int64_t>>(x.shape());
  // The graph only depends on the shapes and pooling parameters; its buffers
  // are fed per call, so a repeated call pays feed and encode only.
  const std::string key = graphKey(
      "avg_pool2d",
      {shapeKey(x_shape), dtypeKey(x_c.dtype()),
       std::to_string(kH), std::to_string(kW), std::to_string(dH),
       std::to_string(dW), std::to_string(padH), std::to_string(padW),
       std::string(ceil_mode ? "ceil" : "floor"),
       std::string(count_include_pad ? "pad" : "nopad"),
       std::string(use_divisor ? "div" : "nodiv"),
       scalarKey(Scalar(divisor_scale))});
  @autoreleasepool {
    MPSShape* inShape = getMPSShape(x_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType dt = getMPSDataType(x_c.dtype());

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                             dataType:dt
                                                  name:@"x"];
      MPSGraphPooling2DOpDescriptor* desc = poolingDescriptor(
          kH, kW, dH, dW, padH, padW, ceil_mode,
          use_divisor ? true : count_include_pad);
      MPSGraphTensor* pooled =
          [graph avgPooling2DWithSourceTensor:xt descriptor:desc name:nil];
      MPSGraphTensor* result = pooled;
      if (use_divisor) {
        MPSGraphTensor* scaleTensor =
            [graph constantWithScalar:divisor_scale dataType:dt];
        result = [graph multiplicationWithPrimaryTensor:pooled
                                       secondaryTensor:scaleTensor
                                                  name:nil];
      }
      entry = new CachedGraph{graph, [xt retain], nil, nil, [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(x_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:dt] autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return batched ? output : output.squeeze(0);
}

// ---------------------------------------------------------------------------
// max_pool2d
//
// Values-only twin of max_pool2d_with_indices: the graph pools without the
// index outputs, so no index tensor is written and no integer cast is encoded.
// ---------------------------------------------------------------------------

Tensor max_pool2d_kernel(const Tensor& input,
                         const std::vector<int64_t>& kernel_size,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& dilation,
                         bool ceil_mode) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError, "max_pool2d(): input must be on the MPS device");
  }
  if (input.dim() != 3 && input.dim() != 4) {
    TP_THROW(RuntimeError, "max_pool2d(): Expected 3D or 4D input");
  }
  const bool batched = input.dim() == 4;
  Tensor x = batched ? input : input.unsqueeze(0);
  const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

  int64_t kH = 0, kW = 0, dH = 0, dW = 0, padH = 0, padW = 0;
  int64_t dilH = 1, dilW = 1;
  normalize_pool_params(kernel_size, stride, padding, kH, kW, dH, dW, padH,
                        padW);
  if (!dilation.empty()) {
    dilH = dilation[0];
    dilW = dilation.size() == 2 ? dilation[1] : dilation[0];
  }
  const int64_t outH = pool_output_size(H, kH, dH, padH, ceil_mode);
  const int64_t outW = pool_output_size(W, kW, dW, padW, ceil_mode);

  std::vector<int64_t> out_shape = {N, C, outH, outW};
  Tensor output = Tensor(out_shape, x.dtype(), x.device());
  if (output.numel() == 0) {
    return batched ? output : output.squeeze(0);
  }

  Tensor x_c = materializeForGraph(x);
  const std::vector<int64_t> x_shape =
      static_cast<std::vector<int64_t>>(x.shape());
  const std::string key = graphKey(
      "max_pool2d_values",
      {shapeKey(x_shape), dtypeKey(x_c.dtype()),
       std::to_string(kH), std::to_string(kW), std::to_string(dH),
       std::to_string(dW), std::to_string(dilH), std::to_string(dilW),
       std::to_string(padH), std::to_string(padW),
       std::string(ceil_mode ? "ceil" : "floor")});
  @autoreleasepool {
    MPSShape* inShape = getMPSShape(x_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType dt = getMPSDataType(x_c.dtype());

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                             dataType:dt
                                                  name:@"x"];
      MPSGraphPooling2DOpDescriptor* desc = poolingDescriptor(
          kH, kW, dH, dW, padH, padW, ceil_mode, false);
      MPSGraphTensor* pooled =
          [graph maxPooling2DWithSourceTensor:xt descriptor:desc name:nil];
      entry = new CachedGraph{graph, [xt retain], nil, nil, [pooled retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(x_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:outShape
                                             dataType:dt] autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return batched ? output : output.squeeze(0);
}

// ---------------------------------------------------------------------------
// max_pool2d_with_indices
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> max_pool2d_with_indices_kernel(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError,
             "max_pool2d_with_indices(): input must be on the MPS device");
  }
  if (input.dim() != 3 && input.dim() != 4) {
    TP_THROW(RuntimeError, "max_pool2d_with_indices(): Expected 3D or 4D input");
  }
  const bool batched = input.dim() == 4;
  Tensor x = batched ? input : input.unsqueeze(0);
  const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

  int64_t kH = 0, kW = 0, dH = 0, dW = 0, padH = 0, padW = 0;
  int64_t dilH = 1, dilW = 1;
  normalize_pool_params(kernel_size, stride, padding, kH, kW, dH, dW, padH,
                        padW);
  if (!dilation.empty()) {
    dilH = dilation[0];
    dilW = dilation.size() == 2 ? dilation[1] : dilation[0];
  }
  const int64_t outH = pool_output_size(H, kH, dH, padH, ceil_mode);
  const int64_t outW = pool_output_size(W, kW, dW, padW, ceil_mode);

  std::vector<int64_t> out_shape = {N, C, outH, outW};
  Tensor values = Tensor(out_shape, x.dtype(), x.device());
  Tensor indices = Tensor(out_shape, DType::Int64, x.device());
  if (values.numel() == 0) {
    return batched ? std::make_tuple(values, indices)
                   : std::make_tuple(values.squeeze(0), indices.squeeze(0));
  }

  Tensor x_c = materializeForGraph(x);
  const std::vector<int64_t> x_shape =
      static_cast<std::vector<int64_t>>(x.shape());
  const std::string key = graphKey(
      "max_pool2d",
      {shapeKey(x_shape), dtypeKey(x_c.dtype()),
       std::to_string(kH), std::to_string(kW), std::to_string(dH),
       std::to_string(dW), std::to_string(dilH), std::to_string(dilW),
       std::to_string(padH), std::to_string(padW),
       std::string(ceil_mode ? "ceil" : "floor")});
  @autoreleasepool {
    MPSShape* inShape = getMPSShape(x_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    const MPSDataType dt = getMPSDataType(x_c.dtype());
    const MPSDataType idx_dt = getMPSDataType(DType::Int64);

    CachedGraph* entry = findCachedGraph(key);
    if (!entry) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* xt = [graph placeholderWithShape:inShape
                                             dataType:dt
                                                  name:@"x"];
      MPSGraphPooling2DOpDescriptor* desc = poolingDescriptor(
          kH, kW, dH, dW, padH, padW, ceil_mode, false);
      desc.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
      desc.returnIndicesDataType = MPSDataTypeInt32;
      NSArray<MPSGraphTensor*>* poolOutputs =
          [graph maxPooling2DReturnIndicesWithSourceTensor:xt
                                                descriptor:desc
                                                      name:nil];
      MPSGraphTensor* idx32 = poolOutputs[1];
      MPSGraphTensor* idx64 = ([idx32 dataType] == idx_dt)
          ? idx32
          : [graph castTensor:idx32 toType:idx_dt name:@"cast_idx"];
      entry = new CachedGraph{graph, [xt retain], nil, nil,
                              [poolOutputs[0] retain], [idx64 retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(x_c);
    id<MTLBuffer> valBuf = mtlBufferOf(values);
    id<MTLBuffer> idxBuf = mtlBufferOf(indices);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* valData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:valBuf
                                                shape:outShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* idxData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:idxBuf
                                                shape:outShape
                                             dataType:idx_dt] autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : valData,
                              entry->result2 : idxData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return batched ? std::make_tuple(values, indices)
                 : std::make_tuple(values.squeeze(0), indices.squeeze(0));
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d
//
// When input sizes are divisible by the output sizes, adaptive average
// pooling is equivalent to a fixed-window avg_pool2d (kernel and stride from
// the standard decomposition).  Non-divisible sizes are not supported yet.
// ---------------------------------------------------------------------------

Tensor adaptive_avg_pool2d_kernel(const Tensor& input,
                                  const std::vector<int64_t>& output_size) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError,
             "adaptive_avg_pool2d(): input must be on the MPS device");
  }
  if (input.dim() != 3 && input.dim() != 4) {
    TP_THROW(RuntimeError, "adaptive_avg_pool2d(): Expected 3D or 4D input");
  }
  const int64_t isizeH = input.size(-2);
  const int64_t isizeW = input.size(-1);
  const int64_t osizeH = output_size[0];
  const int64_t osizeW = output_size[1];
  if (isizeH % osizeH != 0 || isizeW % osizeW != 0) {
    TP_THROW(NotImplementedError,
             "adaptive_avg_pool2d(): MPS requires input sizes to be "
             "divisible by output sizes");
  }
  const int64_t strideH = isizeH / osizeH;
  const int64_t strideW = isizeW / osizeW;
  const int64_t kernelH = isizeH - (osizeH - 1) * strideH;
  const int64_t kernelW = isizeW - (osizeW - 1) * strideW;
  return avg_pool2d_kernel(input, {kernelH, kernelW}, {strideH, strideW},
                           {0, 0}, false, true, std::nullopt);
}

TENSORPLAY_LIBRARY_IMPL(MPS, PoolingKernels) {
  m.impl("avg_pool2d", avg_pool2d_kernel);
  m.impl("max_pool2d", max_pool2d_kernel);
  m.impl("max_pool2d_with_indices", max_pool2d_with_indices_kernel);
  m.impl("adaptive_avg_pool2d", adaptive_avg_pool2d_kernel);
}

} // namespace mps
} // namespace tensorplay