#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "OperationUtils.h"
#include "DistributionDispatch.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/kernels/Normalization.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

DType normalization_stats_dtype(DType input_dtype) {
  return (input_dtype == DType::Float16 || input_dtype == DType::BFloat16)
      ? DType::Float32
      : input_dtype;
}

// Mean and biased variance of `x` over `axes`.  An array of two or more axes
// faults, so several axes are reduced one at a time; every step keeps the rank
// of the tensor before it and leaves a size-1 dimension where the reduced axis
// was, which is the shape the normalization selector broadcasts against.  With
// more than one axis the variance is the mean of the squared deviation from the
// mean of those same axes, the two-pass form of the same quantity.
std::pair<MPSGraphTensor*, MPSGraphTensor*> rankPreservingStats(
    MPSGraph* graph, MPSGraphTensor* x, const std::vector<int64_t>& axes) {
  MPSGraphTensor* mean = x;
  for (int64_t axis : axes) {
    mean = [graph meanOfTensor:mean axes:@[ @(axis) ] name:nil];
  }
  MPSGraphTensor* variance = nil;
  if (axes.size() == 1) {
    variance = [graph varianceOfTensor:x axes:@[ @(axes[0]) ] name:nil];
  } else {
    MPSGraphTensor* centered = [graph subtractionWithPrimaryTensor:x
                                                  secondaryTensor:mean
                                                             name:nil];
    MPSGraphTensor* squared = [graph multiplicationWithPrimaryTensor:centered
                                                     secondaryTensor:centered
                                                                name:nil];
    variance = squared;
    for (int64_t axis : axes) {
      variance = [graph meanOfTensor:variance axes:@[ @(axis) ] name:nil];
    }
  }
  return {mean, variance};
}

} // namespace

// ---------------------------------------------------------------------------
// native_layer_norm / layer_norm
//
// mean and biased variance are computed over the trailing normalized_shape
// dims, then broadcast back against the input and combined with weight/bias.
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
    std::vector<int64_t> axes;
    axes.reserve(normalized_shape.size());
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
      axes.push_back(input.dim() - static_cast<int64_t>(normalized_shape.size()) +
                     static_cast<int64_t>(i));
    }

    const auto stats = rankPreservingStats(graph, x, axes);
    MPSGraphTensor* mean0 = stats.first;
    MPSGraphTensor* var0 = stats.second;
    MPSGraphTensor* epsT = [graph constantWithScalar:eps
                                                shape:@[ @1 ]
                                             dataType:dt];
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
                                          SyncType::COMMIT_ADAPTIVE);
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

// ---------------------------------------------------------------------------
// batch_norm
//
// Training mode computes batch statistics over all non-channel axes,
// normalizes the input against them together with weight/bias, and updates
// running statistics on the host (only C elements).  Evaluation mode
// normalizes directly with the running statistics.
// ---------------------------------------------------------------------------

Tensor batch_norm_kernel(const Tensor& input,
                          const std::optional<Tensor>& weight_opt,
                          const std::optional<Tensor>& bias_opt,
                          const std::optional<Tensor>& running_mean_opt,
                          const std::optional<Tensor>& running_var_opt,
                          bool training, double momentum, double eps) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError, "batch_norm(): input must be on the MPS device");
  }
  if (input.dim() < 2 || input.dim() > 5) {
    TP_THROW(RuntimeError, "batch_norm: Input must be between 2D and 5D");
  }
  if (!isFloatingType(input.dtype())) {
    TP_THROW(NotImplementedError,
             "batch_norm: only floating point dtypes are supported");
  }
  const int64_t C = input.size(1);
  const bool has_running_mean =
      running_mean_opt.has_value() && running_mean_opt->defined();
  const bool has_running_var =
      running_var_opt.has_value() && running_var_opt->defined();
  if (has_running_mean != has_running_var) {
    TP_THROW(RuntimeError,
             "batch_norm: running_mean and running_var must either both be "
             "defined or both be absent");
  }
  if (!training && !(has_running_mean && has_running_var)) {
    TP_THROW(RuntimeError,
             "batch_norm: running_mean and running_var must be defined in "
             "evaluation mode");
  }

  std::vector<int64_t> mean_shape(static_cast<size_t>(input.dim()), 1);
  mean_shape[1] = C;

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  Tensor output = Tensor(out_shape, input.dtype(), input.device());
  if (output.numel() == 0) {
    return output;
  }

  Tensor input_c = materializeForGraph(input);
  Tensor weight_c;
  Tensor bias_c;
  if (weight_opt.has_value() && weight_opt->defined()) {
    weight_c = materializeForGraph(*weight_opt);
  }
  if (bias_opt.has_value() && bias_opt->defined()) {
    bias_c = materializeForGraph(*bias_opt);
  }

  // Evaluation mode is a per-channel affine transform of the input, which a
  // single elementwise pass applies after the host folds the running
  // statistics into scale and shift arrays. The graph below stays for the
  // training path and for mixed storage dtypes.
  const DType bn_dtype = input.dtype();
  if (!training && has_running_mean &&
      (bn_dtype == DType::Float32 || bn_dtype == DType::Float16 ||
       bn_dtype == DType::BFloat16)) {
    Tensor rm_c = materializeForGraph(*running_mean_opt);
    Tensor rv_c = materializeForGraph(*running_var_opt);
    const int64_t batch = input.size(0);
    const int64_t spatial = batch > 0 ? input.numel() / (batch * C) : 0;
    bool shader_ok =
        rm_c.dtype() == bn_dtype && rv_c.dtype() == bn_dtype &&
        rm_c.numel() == C && rv_c.numel() == C &&
        (!weight_c.defined() ||
         (weight_c.dtype() == bn_dtype && weight_c.numel() == C)) &&
        (!bias_c.defined() ||
         (bias_c.dtype() == bn_dtype && bias_c.numel() == C)) &&
        batch > 0 && spatial > 0 &&
        input.numel() <= std::numeric_limits<uint32_t>::max() &&
        spatial <= std::numeric_limits<uint32_t>::max() &&
        C <= std::numeric_limits<uint32_t>::max() &&
        batch <= std::numeric_limits<uint32_t>::max();
    if (shader_ok) {
      synchronizeHostAccess();
      Tensor scale_shift =
          Tensor::empty({2 * C}, DType::Float32, input.device());
      float* folded = scale_shift.data_ptr<float>();
      distribution::dispatch_dtype(bn_dtype, [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* rm = rm_c.data_ptr<scalar_t>();
        const scalar_t* rv = rv_c.data_ptr<scalar_t>();
        const scalar_t* w =
            weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;
        const scalar_t* b =
            bias_c.defined() ? bias_c.data_ptr<scalar_t>() : nullptr;
        for (int64_t c = 0; c < C; ++c) {
          const double scale =
              (w ? static_cast<double>(w[c]) : 1.0) /
              std::sqrt(static_cast<double>(rv[c]) + eps);
          const double shift =
              (b ? static_cast<double>(b[c]) : 0.0) -
              static_cast<double>(rm[c]) * scale;
          folded[c] = static_cast<float>(scale);
          folded[C + c] = static_cast<float>(shift);
        }
      });

      auto pso = MetalShaderLibrary::getBundledLibrary()
                     .getPipelineStateForFunc(
                         "batch_norm_eval_" +
                         scalarToMetalTypeString(bn_dtype));
      const uint32_t spatial_u = static_cast<uint32_t>(spatial);
      const uint32_t channels_u = static_cast<uint32_t>(C);
      const uint32_t batch_u = static_cast<uint32_t>(batch);
      MPSStream* mpsStream = getCurrentMPSStream();
      dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
        @autoreleasepool {
          id<MTLComputeCommandEncoder> encoder = mpsStream->commandEncoder();
          [encoder setComputePipelineState:pso];
          set_shader_arg(encoder, output, 0);
          set_shader_arg(encoder, input_c, 1);
          id<MTLBuffer> folded_buf = (__bridge id<MTLBuffer>)
              getMTLBufferFromPointer(scale_shift.data_ptr());
          [encoder setBuffer:folded_buf offset:0 atIndex:2];
          [encoder setBuffer:folded_buf
                      offset:static_cast<NSUInteger>(sizeof(float) * C)
                     atIndex:3];
          set_shader_arg(encoder, spatial_u, 4);
          set_shader_arg(encoder, channels_u, 5);
          mtl_dispatch3DJob(encoder, pso, static_cast<NSUInteger>(spatial_u),
                            static_cast<NSUInteger>(channels_u),
                            static_cast<NSUInteger>(batch_u));
          mpsStream->synchronize(SyncType::COMMIT);
        }
      });
      return output;
    }
  }

  // Retained instead of autoreleased: the graph must stay alive until after
  // the command buffer fully completes, and the pool drains before the
  // host-side running-stats update below.
  MPSGraph* graph = nil;

  @autoreleasepool {
    graph = [[MPSGraph new] retain];
    MPSShape* inShape = getMPSShape(out_shape);
    MPSShape* meanShape = getMPSShape(mean_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                          dataType:dt
                                               name:@"x"];

    MPSGraphTensor* mean0 = nil;
    MPSGraphTensor* var0 = nil;
    if (training) {
      // Batch statistics over every axis but the channel one.  Each step of
      // the reduction keeps the input's rank and leaves a size-1 dimension
      // behind, so the result already has the (1, C, 1, ...) shape that the
      // normalization selector broadcasts against.
      std::vector<int64_t> axes;
      axes.reserve(static_cast<size_t>(input.dim()) - 1);
      for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != 1) {
          axes.push_back(i);
        }
      }
      const auto stats = rankPreservingStats(graph, x, axes);
      mean0 = stats.first;
      var0 = stats.second;
    } else {
      mean0 = [graph placeholderWithShape:meanShape
                              dataType:dt
                                   name:@"rm"];
      var0 = [graph placeholderWithShape:meanShape
                              dataType:dt
                                   name:@"rv"];
    }
    MPSGraphTensor* meanB = mean0;
    MPSGraphTensor* varB = var0;

    MPSGraphTensor* gamma = nil;
    MPSGraphTensor* beta = nil;
    if (weight_c.defined()) {
      gamma = [graph placeholderWithShape:meanShape dataType:dt name:@"w"];
    }
    if (bias_c.defined()) {
      beta = [graph placeholderWithShape:meanShape dataType:dt name:@"b"];
    }
    MPSGraphTensor* outT = [graph normalizationWithTensor:x
                                               meanTensor:meanB
                                           varianceTensor:varB
                                              gammaTensor:gamma
                                               betaTensor:beta
                                                  epsilon:(float)eps
                                                     name:nil];

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:inData, x, nil];
    NSMutableDictionary* results =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:outData, outT, nil];

    if (!training) {
      Tensor rm_c = materializeForGraph(*running_mean_opt);
      Tensor rv_c = materializeForGraph(*running_var_opt);
      MPSGraphTensorData* rmData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(rm_c)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      MPSGraphTensorData* rvData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(rv_c)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      feeds[mean0] = rmData;
      feeds[var0] = rvData;
    }
    if (weight_c.defined()) {
      MPSGraphTensorData* wData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(weight_c)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      feeds[gamma] = wData;
    }
    if (bias_c.defined()) {
      MPSGraphTensorData* bData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(bias_c)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      feeds[beta] = bData;
    }

    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
  }

  if (training && has_running_mean) {
    const int64_t count = input.numel() / C;
    const double unbiased = count > 1.0 ? count / (count - 1.0) : 1.0;
    std::vector<double> mean(C), var(C);
    // The graph has completed (COMMIT_AND_WAIT), so the per-channel batch
    // statistics can be derived from the input through the unified-memory
    // mapping to refresh the running statistics.
    {
      const int64_t n_batch = input.size(0);
      const int64_t spatial =
          (n_batch > 0) ? count / n_batch : 0;
      std::vector<double> sum(static_cast<size_t>(C), 0.0);
      std::vector<double> sumsq(static_cast<size_t>(C), 0.0);
      distribution::dispatch_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* data = input_c.data_ptr<scalar_t>();
        const int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
          const int64_t c = (i / spatial) % C;
          const double v = static_cast<double>(data[i]);
          sum[static_cast<size_t>(c)] += v;
          sumsq[static_cast<size_t>(c)] += v * v;
        }
      });
      for (int64_t c = 0; c < C; ++c) {
        mean[c] = sum[static_cast<size_t>(c)] / static_cast<double>(count);
        var[c] = sumsq[static_cast<size_t>(c)] / static_cast<double>(count) -
                mean[c] * mean[c];
      }
    }
    Tensor rm = *running_mean_opt;
    Tensor rv = *running_var_opt;
    if (rm.dtype() == DType::Float32) {
      float* rmp = rm.data_ptr<float>();
      float* rvp = rv.data_ptr<float>();
      for (int64_t c = 0; c < C; ++c) {
        rmp[c] = static_cast<float>((1.0 - momentum) * rmp[c] +
                                    momentum * mean[c]);
        rvp[c] = static_cast<float>((1.0 - momentum) * rvp[c] +
                                      momentum * var[c] * unbiased);
      }
    } else {
      double* rmp = rm.data_ptr<double>();
      double* rvp = rv.data_ptr<double>();
      for (int64_t c = 0; c < C; ++c) {
        rmp[c] = (1.0 - momentum) * rmp[c] + momentum * mean[c];
        rvp[c] = (1.0 - momentum) * rvp[c] + momentum * var[c] * unbiased;
      }
    }
  }
  [graph release];
  return output;
}

// ---------------------------------------------------------------------------
// group_norm
//
// The input is read as the nested (N, G, C/G, spatial) layout of the same
// elements, the two inner dimensions are reduced into the group statistics, and
// the result is written back through that layout, which holds the elements in
// the order the caller's shape has them.
// ---------------------------------------------------------------------------

Tensor group_norm_kernel(const Tensor& input, int64_t num_groups,
                          const std::optional<Tensor>& weight_opt,
                          const std::optional<Tensor>& bias_opt, double eps) {
  if (!input.device().is_mps()) {
    TP_THROW(RuntimeError, "group_norm(): input must be on the MPS device");
  }
  if (input.dim() < 2) {
    TP_THROW(RuntimeError, "group_norm(): expected at least 2D input");
  }
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  if (C % num_groups != 0) {
    TP_THROW(RuntimeError, "group_norm(): num_groups must divide channels");
  }
  const int64_t G = num_groups;
  const int64_t spatial = input.numel() / (N * C);

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  Tensor output = Tensor(out_shape, input.dtype(), input.device());
  if (output.numel() == 0) {
    return output;
  }

  // One threadgroup reduces and normalizes each (batch, group) pair when the
  // storage is a floating-point type and the affine operands either both
  // exist or both are absent. Anything else keeps the graph path, which
  // builds the casts and broadcasts it needs.
  const DType gn_dtype = input.dtype();
  if (gn_dtype == DType::Float32 || gn_dtype == DType::Float16 ||
      gn_dtype == DType::BFloat16) {
    const bool has_weight =
        weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    if (has_weight == has_bias) {
      const int64_t channels_per_group = C / G;
      const int64_t elements_per_group = channels_per_group * spatial;
      const int64_t group_count = N * G;
      bool shader_ok =
          elements_per_group > 0 &&
          input.numel() <= std::numeric_limits<uint32_t>::max() &&
          elements_per_group <= std::numeric_limits<uint32_t>::max() &&
          group_count <= std::numeric_limits<uint32_t>::max() &&
          spatial <= std::numeric_limits<uint32_t>::max() &&
          channels_per_group <= std::numeric_limits<uint32_t>::max() &&
          G <= std::numeric_limits<uint32_t>::max();
      Tensor input_c = materializeForGraph(input);
      Tensor weight_c;
      Tensor bias_c;
      if (shader_ok && has_weight) {
        weight_c = materializeForGraph(*weight_opt);
        bias_c = materializeForGraph(*bias_opt);
        shader_ok = weight_c.dtype() == gn_dtype &&
                    bias_c.dtype() == gn_dtype &&
                    weight_c.numel() == C && bias_c.numel() == C;
      }
      if (shader_ok) {
        GroupNormParams params{
            static_cast<uint32_t>(spatial),
            static_cast<uint32_t>(channels_per_group),
            static_cast<uint32_t>(elements_per_group),
            static_cast<uint32_t>(G)};
        auto pso = MetalShaderLibrary::getBundledLibrary()
                       .getPipelineStateForFunc(
                           "group_norm_" + scalarToMetalTypeString(gn_dtype) +
                           (has_weight ? "_affine" : ""));
        const uint64_t blocks =
            static_cast<uint64_t>(elements_per_group + 3) / 4;
        uint32_t tptg = static_cast<uint32_t>(
            std::min<uint64_t>(32 * ((blocks + 31) / 32), 1024));
        tptg = std::min<uint32_t>(
            tptg, static_cast<uint32_t>(
                      [pso maxTotalThreadsPerThreadgroup]));
        tptg = std::max<uint32_t>(tptg, 32);
        const float eps_f = static_cast<float>(eps);
        MPSStream* mpsStream = getCurrentMPSStream();
        dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
          @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                mpsStream->commandEncoder();
            [encoder setComputePipelineState:pso];
            if (has_weight) {
              mtl_setArgs(encoder, output, input_c, weight_c, bias_c, params,
                          eps_f);
            } else {
              mtl_setArgs(encoder, output, input_c, params, eps_f);
            }
            [encoder
                dispatchThreadgroups:MTLSizeMake(
                                        static_cast<NSUInteger>(group_count),
                                        1, 1)
                threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
            mpsStream->synchronize(SyncType::COMMIT);
          }
        });
        return output;
      }
    }
  }

  // The four nested dimensions the graph works in.  Splitting a group into its
  // own channels and the elements those channels repeat over keeps the flat
  // order of the input intact and lets the per-channel weight and bias broadcast
  // against the normalized values where they are, so the graph never has to
  // change rank on the way to the write.
  const int64_t channels_per_group = C / G;
  std::vector<int64_t> grouped_shape = {N, G, channels_per_group, spatial};
  std::vector<int64_t> bcast_shape = {N, G, 1, 1};
  std::vector<int64_t> affine_shape = {1, G, channels_per_group, 1};

  Tensor input_c = materializeForGraph(input);
  Tensor weight_c;
  Tensor bias_c;
  if (weight_opt.has_value() && weight_opt->defined()) {
    weight_c = materializeForGraph(*weight_opt);
  }
  if (bias_opt.has_value() && bias_opt->defined()) {
    bias_c = materializeForGraph(*bias_opt);
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* groupedShape = getMPSShape(grouped_shape);
    MPSShape* bcastShape = getMPSShape(bcast_shape);
    MPSShape* affineShape = getMPSShape(affine_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* xr = [graph placeholderWithShape:groupedShape
                                           dataType:dt
                                                name:@"x"];
    // A group is the pair of inner dimensions, reduced one axis at a time.
    const auto stats = rankPreservingStats(graph, xr, {2, 3});
    MPSGraphTensor* var0 = stats.second;
    MPSGraphTensor* epsT = [graph constantWithScalar:eps
                                                shape:@[ @1 ]
                                             dataType:dt];
    MPSGraphTensor* varEps = [graph additionWithPrimaryTensor:var0
                                          secondaryTensor:epsT
                                                     name:nil];
    MPSGraphTensor* sqrtVar = [graph squareRootWithTensor:varEps name:nil];
    MPSGraphTensor* rstd0 = [graph reciprocalWithTensor:sqrtVar name:nil];
    // The reductions leave the size-1 dimensions in place; the reshapes name
    // them explicitly so the broadcast below does not depend on what a
    // reduction does to the rank of its input.
    MPSGraphTensor* meanB =
        [graph reshapeTensor:stats.first withShape:bcastShape name:nil];
    MPSGraphTensor* rstdB =
        [graph reshapeTensor:rstd0 withShape:bcastShape name:nil];
    MPSGraphTensor* centered = [graph subtractionWithPrimaryTensor:xr
                                                  secondaryTensor:meanB
                                                              name:nil];
    MPSGraphTensor* result = [graph multiplicationWithPrimaryTensor:centered
                                                  secondaryTensor:rstdB
                                                             name:nil];
    MPSGraphTensor* wt = nil;
    MPSGraphTensor* bt = nil;
    if (weight_c.defined()) {
      wt = [graph placeholderWithShape:affineShape dataType:dt name:@"w"];
      result = [graph multiplicationWithPrimaryTensor:result
                                     secondaryTensor:wt
                                                name:nil];
    }
    if (bias_c.defined()) {
      bt = [graph placeholderWithShape:affineShape dataType:dt name:@"b"];
      result = [graph additionWithPrimaryTensor:result
                               secondaryTensor:bt
                                          name:nil];
    }

    id<MTLBuffer> inBuf = mtlBufferOf(input_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                            shape:groupedShape
                                         dataType:dt] autorelease];
    // The nested shape holds the same elements in the same order as the input,
    // which is why writing it fills the output the caller asked for.
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:groupedShape
                                             dataType:dt] autorelease];
    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:inData, xr, nil];
    NSMutableDictionary* results =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:outData, result, nil];
    if (weight_c.defined()) {
      MPSGraphTensorData* wData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(weight_c)
                                                  shape:affineShape
                                               dataType:dt] autorelease];
      feeds[wt] = wData;
    }
    if (bias_c.defined()) {
      MPSGraphTensorData* bData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(bias_c)
                                                  shape:affineShape
                                               dataType:dt] autorelease];
      feeds[bt] = bData;
    }

    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, NormalizationKernels) {
  m.impl("native_layer_norm", native_layer_norm_kernel);
  m.impl("layer_norm", layer_norm_kernel);
  m.impl("batch_norm", batch_norm_kernel);
  m.impl("group_norm", group_norm_kernel);
}

} // namespace mps
} // namespace tensorplay