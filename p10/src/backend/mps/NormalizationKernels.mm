#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

DType normalization_stats_dtype(DType input_dtype) {
  return (input_dtype == DType::Float16 || input_dtype == DType::BFloat16)
      ? DType::Float32
      : input_dtype;
}

} // namespace

// ---------------------------------------------------------------------------
// native_layer_norm / layer_norm
//
// mean and biased variance are computed over the trailing normalized_shape
// dims with the confirmed meanOfTensor/varianceOfTensor selectors, then
// broadcast back against the input and combined with weight/bias.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_kernel(
    const Tensor& input, const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_in,
    const std::optional<Tensor>& bias_in, double eps) {
  if (normalized_shape.empty()) {
    TP_THROW(RuntimeError, "native_layer_norm: normalized_shape must not be "
             "empty");
  }
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError,
             "native_layer_norm(): input must be on the MPS device");
  }
  int64_t inner_size = 1;
  for (int64_t s : normalized_shape) {
    inner_size *= s;
  }
  const int64_t outer_size = inner_size == 0 ? 0 : input.numel() / inner_size;
  const DType stats_dtype = normalization_stats_dtype(input.dtype());
  const DType in_dtype = input.dtype();

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  Tensor mean = Tensor({outer_size}, stats_dtype, input.device());
  Tensor rstd = Tensor({outer_size}, stats_dtype, input.device());
  Tensor out = Tensor(out_shape, in_dtype, input.device());

  if (input.numel() == 0 || outer_size == 0) {
    return std::make_tuple(out, mean, rstd);
  }

  // Shape used to broadcast the per-row statistics back to the input: the
  // leading dims stay, the normalized dims become 1.
  std::vector<int64_t> bcast_shape = out_shape;
  for (size_t i = 0; i < normalized_shape.size(); ++i) {
    bcast_shape[bcast_shape.size() - 1 - i] = 1;
  }

  // weight/bias: normalized_shape reshaped with leading 1s for broadcasting.
  std::vector<int64_t> affine_shape(bcast_shape.size(), 1);
  for (size_t i = 0; i < normalized_shape.size(); ++i) {
    affine_shape[affine_shape.size() - normalized_shape.size() + i] =
        normalized_shape[i];
  }

  Tensor input_c = materializeForGraph(input);
  Tensor weight_c;
  Tensor bias_c;
  if (weight_in.has_value() && weight_in->defined()) {
    if (!weight_in->device().is_mps()) {
      TP_THROW(RuntimeError, "native_layer_norm(): weight must be on MPS");
    }
    weight_c = materializeForGraph(weight_in->reshape(normalized_shape));
  }
  if (bias_in.has_value() && bias_in->defined()) {
    if (!bias_in->device().is_mps()) {
      TP_THROW(RuntimeError, "native_layer_norm(): bias must be on MPS");
    }
    bias_c = materializeForGraph(bias_in->reshape(normalized_shape));
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(out_shape);
    const MPSDataType dt = getMPSDataType(in_dtype);
    const MPSDataType stats_dt = getMPSDataType(stats_dtype);

    MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                          dataType:dt
                                               name:@"x"];
    NSMutableArray<NSNumber*>* axes =
        [NSMutableArray arrayWithCapacity:normalized_shape.size()];
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
      [axes addObject:@(input.dim() - normalized_shape.size() + i)];
    }

    MPSGraphTensor* mean0 = [graph meanOfTensor:x axes:axes name:nil];
    MPSGraphTensor* var0 = [graph varianceOfTensor:x axes:axes name:nil];
    MPSGraphTensor* epsT = [graph constantWithScalar:eps dataType:dt];
    MPSGraphTensor* varEps = [graph additionWithPrimaryTensor:var0
                                          secondaryTensor:epsT
                                                     name:nil];
    MPSGraphTensor* sqrtVar = [graph squareRootWithTensor:varEps name:nil];
    MPSGraphTensor* rstd0 = [graph reciprocalWithTensor:sqrtVar name:nil];

    MPSGraphTensor* mean1d =
        [graph reshapeTensor:mean0 withShape:@[ @(outer_size) ] name:nil];
    MPSGraphTensor* rstd1d =
        [graph reshapeTensor:rstd0 withShape:@[ @(outer_size) ] name:nil];
    MPSGraphTensor* meanBcast =
        [graph reshapeTensor:mean1d withShape:getMPSShape(bcast_shape) name:nil];
    MPSGraphTensor* rstdBcast =
        [graph reshapeTensor:rstd1d withShape:getMPSShape(bcast_shape) name:nil];

    MPSGraphTensor* centered = [graph subtractionWithPrimaryTensor:x
                                                  secondaryTensor:meanBcast
                                                              name:nil];
    MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:centered
                                                       secondaryTensor:rstdBcast
                                                                  name:nil];
    MPSGraphTensor* result = normalized;
    MPSGraphTensor* wt = nil;
    MPSGraphTensor* bt = nil;
    if (weight_c.defined()) {
      wt = [graph placeholderWithShape:getMPSShape(affine_shape)
                              dataType:dt
                                   name:@"w"];
      result = [graph multiplicationWithPrimaryTensor:result
                                     secondaryTensor:wt
                                                name:nil];
    }
    if (bias_c.defined()) {
      bt = [graph placeholderWithShape:getMPSShape(affine_shape)
                              dataType:dt
                                   name:@"b"];
      result = [graph additionWithPrimaryTensor:result
                               secondaryTensor:bt
                                          name:nil];
    }

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
    id<MTLBuffer> outBuf = mtlBufferOf(out);
    id<MTLBuffer> meanBuf = mtlBufferOf(mean);
    id<MTLBuffer> rstdBuf = mtlBufferOf(rstd);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* meanData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:meanBuf
                                                shape:@[ @(outer_size) ]
                                             dataType:stats_dt] autorelease];
    MPSGraphTensorData* rstdData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:rstdBuf
                                                shape:@[ @(outer_size) ]
                                             dataType:stats_dt] autorelease];

    MPSGraphTensor* meanOut = ([mean1d dataType] == stats_dt)
        ? mean1d
        : [graph castTensor:mean1d toType:stats_dt name:@"cast_mean"];
    MPSGraphTensor* rstdOut = ([rstd1d dataType] == stats_dt)
        ? rstd1d
        : [graph castTensor:rstd1d toType:stats_dt name:@"cast_rstd"];

    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:inData, x, nil];
    NSMutableDictionary* results =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:outData, result,
                                                          meanData, meanOut,
                                                          rstdData, rstdOut,
                                                          nil];
    if (weight_c.defined()) {
      MPSGraphTensorData* wData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(weight_c)
                                                  shape:getMPSShape(affine_shape)
                                               dataType:dt] autorelease];
      feeds[wt] = wData;
    }
    if (bias_c.defined()) {
      MPSGraphTensorData* bData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(bias_c)
                                                  shape:getMPSShape(affine_shape)
                                               dataType:dt] autorelease];
      feeds[bt] = bData;
    }

    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }
  return std::make_tuple(out, mean, rstd);
}

Tensor layer_norm_kernel(const Tensor& input,
                         const std::vector<int64_t>& normalized_shape,
                         const std::optional<Tensor>& weight_opt,
                         const std::optional<Tensor>& bias_opt, double eps) {
  return std::get<0>(native_layer_norm_kernel(input, normalized_shape,
                                              weight_opt, bias_opt, eps));
}

TENSORPLAY_LIBRARY_IMPL(MPS, NormalizationKernels) {
  m.impl("native_layer_norm", native_layer_norm_kernel);
  m.impl("layer_norm", layer_norm_kernel);
}

} // namespace mps
} // namespace tensorplay