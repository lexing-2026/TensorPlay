#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "OperationUtils.h"
#include "DistributionDispatch.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

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
// Per-row mean and 1/sqrt(var+eps) over the trailing normalized_shape dims
// are computed on the host through the unified-memory mapping, fed into the
// graph as broadcast placeholders, then combined with weight/bias.
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

  // Per-row statistics: mean and 1/sqrt(var+eps) over the trailing
  // normalized dims, computed on the host (outer_size rows).
  getCurrentMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  distribution::dispatch_dtype(in_dtype, [&](auto tag) {
    using scalar_t = decltype(tag);
    const scalar_t* data = input_c.data_ptr<scalar_t>();
    distribution::dispatch_dtype(stats_dtype, [&](auto stag) {
      using stat_t = decltype(stag);
      stat_t* mp = mean.data_ptr<stat_t>();
      stat_t* rp = rstd.data_ptr<stat_t>();
      for (int64_t r = 0; r < outer_size; ++r) {
        double sum = 0;
        double sumsq = 0;
        for (int64_t i = 0; i < inner_size; ++i) {
          const double v =
              static_cast<double>(data[r * inner_size + i]);
          sum += v;
          sumsq += v * v;
        }
        const double mean_v = sum / static_cast<double>(inner_size);
        const double var_v =
            sumsq / static_cast<double>(inner_size) - mean_v * mean_v;
        mp[r] = static_cast<stat_t>(mean_v);
        rp[r] = static_cast<stat_t>(
            1.0 / std::sqrt(var_v + eps));
      }
    });
  });

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(out_shape);
    MPSShape* bcastMPSShape = getMPSShape(bcast_shape);
    const MPSDataType dt = getMPSDataType(in_dtype);

    MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                          dataType:dt
                                               name:@"x"];
    MPSGraphTensor* meanB = [graph placeholderWithShape:bcastMPSShape
                                             dataType:dt
                                                  name:@"mean"];
    MPSGraphTensor* rstdB = [graph placeholderWithShape:bcastMPSShape
                                              dataType:dt
                                                   name:@"rstd"];
    MPSGraphTensor* centered = [graph subtractionWithPrimaryTensor:x
                                                  secondaryTensor:meanB
                                                              name:nil];
    MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:centered
                                                       secondaryTensor:rstdB
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
                                                shape:bcastMPSShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* rstdData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:rstdBuf
                                                shape:bcastMPSShape
                                             dataType:dt] autorelease];

    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:inData, x, nil];
    NSMutableDictionary* results =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:outData, result, nil];
    feeds[meanB] = meanData;
    feeds[rstdB] = rstdData;
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

// ---------------------------------------------------------------------------
// batch_norm
//
// Training mode computes the batch statistics on the host through the
// unified-memory mapping (only C channels), feeds them into the graph as
// placeholders, normalizes with the confirmed normalizationWithTensor
// selector, and updates running statistics in place.  Evaluation mode
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

  // Batch statistics buffers (input dtype) computed on the host for the
  // training path; they are fed into the graph as placeholders and also
  // drive the running-statistics update.
  Tensor batch_mean;
  Tensor batch_var;
  if (training) {
    batch_mean = Tensor({C}, input.dtype(), input.device());
    batch_var = Tensor({C}, input.dtype(), input.device());
    getCurrentMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    const int64_t numel = input.numel();
    const int64_t n_batch = input.size(0);
    const int64_t spatial =
        (n_batch > 0) ? numel / (n_batch * C) : 0;
    std::vector<double> sum(static_cast<size_t>(C), 0.0);
    std::vector<double> sumsq(static_cast<size_t>(C), 0.0);
    distribution::dispatch_dtype(input.dtype(), [&](auto tag) {
      using scalar_t = decltype(tag);
      const scalar_t* data = input_c.data_ptr<scalar_t>();
      for (int64_t i = 0; i < numel; ++i) {
        const int64_t c = (i / spatial) % C;
        const double v = static_cast<double>(data[i]);
        sum[static_cast<size_t>(c)] += v;
        sumsq[static_cast<size_t>(c)] += v * v;
      }
    });
    const int64_t count = numel / C;
    distribution::dispatch_dtype(input.dtype(), [&](auto tag) {
      using scalar_t = decltype(tag);
      scalar_t* mp = batch_mean.data_ptr<scalar_t>();
      scalar_t* vp = batch_var.data_ptr<scalar_t>();
      for (int64_t c = 0; c < C; ++c) {
        const double mean =
            sum[static_cast<size_t>(c)] / static_cast<double>(count);
        const double var = sumsq[static_cast<size_t>(c)] /
                               static_cast<double>(count) -
                           mean * mean;
        mp[c] = static_cast<scalar_t>(mean);
        vp[c] = static_cast<scalar_t>(var);
      }
    });
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(out_shape);
    MPSShape* meanShape = getMPSShape(mean_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                          dataType:dt
                                               name:@"x"];

    // Both training and evaluation feed mean/variance as placeholders; the
    // training path supplies the host-computed batch statistics, while
    // evaluation supplies the running statistics.
    MPSGraphTensor* mean0 = [graph placeholderWithShape:meanShape
                                              dataType:dt
                                                   name:@"mean"];
    MPSGraphTensor* var0 = [graph placeholderWithShape:meanShape
                                              dataType:dt
                                                   name:@"var"];
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

    if (training) {
      MPSGraphTensorData* meanData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(batch_mean)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      MPSGraphTensorData* varData =
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(batch_var)
                                                  shape:meanShape
                                               dataType:dt] autorelease];
      feeds[mean0] = meanData;
      feeds[var0] = varData;
    } else {
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
    distribution::dispatch_dtype(input.dtype(), [&](auto tag) {
      using scalar_t = decltype(tag);
      const scalar_t* mp = batch_mean.data_ptr<scalar_t>();
      const scalar_t* vp = batch_var.data_ptr<scalar_t>();
      for (int64_t c = 0; c < C; ++c) {
        mean[c] = static_cast<double>(mp[c]);
        var[c] = static_cast<double>(vp[c]);
      }
    });
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
  return output;
}

// ---------------------------------------------------------------------------
// group_norm
//
// The input is reshaped to (N, G, -1).  Group mean and 1/sqrt(var+eps) are
// computed on the host through the unified-memory mapping (N*G values),
// fed into the graph as placeholders, then the normalized result is
// reshaped back and per-channel weight/bias are applied.
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
  const int64_t per_group = (C / G) * spatial;

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(input.shape());
  Tensor output = Tensor(out_shape, input.dtype(), input.device());
  if (output.numel() == 0) {
    return output;
  }

  // Reshaped view: (N, G, C/G * spatial).
  std::vector<int64_t> grouped_shape = {N, G, per_group};
  std::vector<int64_t> bcast_shape = {N, G, 1};
  std::vector<int64_t> affine_shape(static_cast<size_t>(input.dim()), 1);
  affine_shape[1] = C;

  Tensor input_c = materializeForGraph(input);
  Tensor weight_c;
  Tensor bias_c;
  if (weight_opt.has_value() && weight_opt->defined()) {
    weight_c = materializeForGraph(*weight_opt);
  }
  if (bias_opt.has_value() && bias_opt->defined()) {
    bias_c = materializeForGraph(*bias_opt);
  }

  // Group statistics: per-(n,g) mean and 1/sqrt(var+eps), computed on the
  // host (N*G values) and fed into the graph as placeholders.
  Tensor group_mean;
  Tensor group_rstd;
  {
    group_mean = Tensor({N * G}, input.dtype(), input.device());
    group_rstd = Tensor({N * G}, input.dtype(), input.device());
    getCurrentMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    distribution::dispatch_dtype(input.dtype(), [&](auto tag) {
      using scalar_t = decltype(tag);
      const scalar_t* data = input_c.data_ptr<scalar_t>();
      scalar_t* mp = group_mean.data_ptr<scalar_t>();
      scalar_t* rp = group_rstd.data_ptr<scalar_t>();
      for (int64_t ng = 0; ng < N * G; ++ng) {
        double sum = 0;
        double sumsq = 0;
        for (int64_t i = 0; i < per_group; ++i) {
          const double v =
              static_cast<double>(data[ng * per_group + i]);
          sum += v;
          sumsq += v * v;
        }
        const double mean = sum / static_cast<double>(per_group);
        const double var = sumsq / static_cast<double>(per_group) -
                           mean * mean;
        mp[ng] = static_cast<scalar_t>(mean);
        rp[ng] = static_cast<scalar_t>(
            1.0 / std::sqrt(var + eps));
      }
    });
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* groupedShape = getMPSShape(grouped_shape);
    MPSShape* bcastShape = getMPSShape(bcast_shape);
    MPSShape* outShape = getMPSShape(out_shape);
    MPSShape* affineShape = getMPSShape(affine_shape);
    const MPSDataType dt = getMPSDataType(input_c.dtype());

    MPSGraphTensor* xr = [graph placeholderWithShape:groupedShape
                                           dataType:dt
                                                name:@"x"];
    MPSGraphTensor* mean0 = [graph placeholderWithShape:bcastShape
                                      dataType:dt
                                           name:@"mean"];
    MPSGraphTensor* rstd0 = [graph placeholderWithShape:bcastShape
                                            dataType:dt
                                                 name:@"rstd"];
    MPSGraphTensor* centered = [graph subtractionWithPrimaryTensor:xr
                                                  secondaryTensor:mean0
                                                              name:nil];
    MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:centered
                                                       secondaryTensor:rstd0
                                                                  name:nil];
    MPSGraphTensor* result =
        [graph reshapeTensor:normalized withShape:outShape name:nil];
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
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                 shape:outShape
                                              dataType:dt] autorelease];
    NSMutableDictionary* feeds =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:inData, xr, nil];
    NSMutableDictionary* results =
        [NSMutableDictionary dictionaryWithObjectsAndKeys:outData, result, nil];
    MPSGraphTensorData* meanData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(group_mean)
                                                shape:bcastShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* rstdData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:mtlBufferOf(group_rstd)
                                                shape:bcastShape
                                             dataType:dt] autorelease];
    feeds[mean0] = meanData;
    feeds[rstd0] = rstdData;
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
                                          SyncType::COMMIT_AND_WAIT);
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