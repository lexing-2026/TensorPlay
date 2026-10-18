#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MPSStream.h"

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <limits>
#include <set>
#include <vector>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

namespace {

enum class ReductionKind { SUM, PROD, MEAN, MAX, MIN, ALL, ANY };

DType resolve_sum_dtype(DType dtype, DType self_dtype) {
  DType out_dtype = dtype;
  if (out_dtype == DType::Undefined) {
    out_dtype = self_dtype;
    if (isIntegralType(self_dtype, true)) {
      out_dtype = DType::Int64;
    }
  }
  return out_dtype;
}

DType resolve_mean_dtype(DType dtype, DType self_dtype) {
  if (dtype != DType::Undefined) {
    return dtype;
  }
  return isFloatingOrComplexType(self_dtype) ? self_dtype : DType::Float32;
}

// Narrow floating types accumulate in float32 to avoid overflow.
DType accumulation_dtype(DType out_dtype) {
  return isReducedFloatingType(out_dtype) ? DType::Float32 : out_dtype;
}

std::vector<int64_t> normalize_dims(const std::vector<int64_t>& dims,
                                    int64_t ndim) {
  std::vector<int64_t> result;
  for (int64_t d : dims) {
    int64_t nd = d < 0 ? d + ndim : d;
    if (nd < 0 || nd >= ndim) {
      TP_THROW(IndexError, "Dimension out of range (expected to be in range [",
               -ndim, ", ", ndim - 1, "], but got ", d, ")");
    }
    result.push_back(nd);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::vector<int64_t> compute_out_shape(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& axes, bool keepdim) {
  const std::set<int64_t> axes_set(axes.begin(), axes.end());
  std::vector<int64_t> out;
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (axes_set.count(static_cast<int64_t>(i)) != 0) {
      if (keepdim) out.push_back(1);
    } else {
      out.push_back(in_shape[i]);
    }
  }
  return out;
}

// Executes a reduction over `dims` (empty = all dims). `out_dtype` is the
// storage dtype of the result; `acc_dtype` is the dtype the graph reduces in.
Tensor reductionOp(const Tensor& self, const std::vector<int64_t>& dims,
                   bool keepdim, DType out_dtype, DType acc_dtype,
                   ReductionKind kind) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS reduction requires input on the MPS device");
  }
  const std::vector<int64_t> in_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  std::vector<int64_t> axes = normalize_dims(dims, self.dim());
  if (axes.empty()) {
    // Reducing over an empty dim list means "reduce all dimensions".
    for (int64_t i = 0; i < self.dim(); ++i) {
      axes.push_back(i);
    }
  }
  const std::vector<int64_t> out_shape =
      compute_out_shape(in_shape, axes, keepdim);

  // A 0-d input reduced over all dims is just a dtype conversion/copy.
  if (self.dim() == 0) {
    return self.dtype() == out_dtype ? self.clone() : self.to(out_dtype);
  }

  if (self.numel() == 0) {
    switch (kind) {
      case ReductionKind::SUM: {
        Tensor out = Tensor(out_shape, out_dtype, self.device());
        fill_kernel(out, 0);
        return out;
      }
      case ReductionKind::PROD: {
        Tensor out = Tensor(out_shape, out_dtype, self.device());
        fill_kernel(out, 1);
        return out;
      }
      case ReductionKind::MEAN: {
        Tensor out = Tensor(out_shape, out_dtype, self.device());
        fill_kernel(out, std::numeric_limits<float>::quiet_NaN());
        return out;
      }
      case ReductionKind::ALL: {
        Tensor out = Tensor(out_shape, DType::Bool, self.device());
        fill_kernel(out, true);
        return out;
      }
      case ReductionKind::ANY: {
        Tensor out = Tensor(out_shape, DType::Bool, self.device());
        fill_kernel(out, false);
        return out;
      }
      default:
        TP_THROW(RuntimeError,
                 "Expected reduction dim to be specified for input.numel() == 0");
    }
  }

  // A reduction over every axis of one long contiguous float operand fills the
  // device by splitting the run into chunks and folding the partials; the
  // per-row kernels below would serve it with a single group.
  if (static_cast<int64_t>(axes.size()) == self.dim() &&
      self.dtype() == DType::Float32 && out_dtype == DType::Float32 &&
      acc_dtype == DType::Float32) {
    if (kind == ReductionKind::SUM || kind == ReductionKind::MEAN) {
      const float divisor = kind == ReductionKind::MEAN
                                ? static_cast<float>(self.numel())
                                : 0.0f;
      if (auto result = chunkedRowSumShader(self, out_shape, divisor)) {
        return *result;
      }
    } else if (kind == ReductionKind::MAX || kind == ReductionKind::MIN) {
      if (auto result = chunkedRowExtremumShader(
              self, out_shape, kind == ReductionKind::MAX)) {
        return *result;
      }
    }
  }

  // A reduction over the last axis of a float operand is one row per SIMD
  // group, so the operand goes straight from the storage it is in and nothing
  // stands between the read and the write. A call the shader cannot address
  // falls through to the graph below.
  if (axes.size() == 1 && axes.front() == self.dim() - 1 &&
      out_dtype == DType::Float32 && acc_dtype == DType::Float32 &&
      (kind == ReductionKind::SUM || kind == ReductionKind::MEAN)) {
    const float divisor = kind == ReductionKind::MEAN
                              ? static_cast<float>(in_shape.back())
                              : 0.0f;
    if (auto result = rowSumShader(self, out_shape, divisor)) {
      return *result;
    }
  }

  // The extremum of a row and the answer to "is every one of these nonzero" are
  // read the same way, each by the group that takes its row: the first writes
  // the value the row holds, the second one byte per row. A reduction that
  // arrives with a type no kernel of these is instantiated for falls through to
  // the graph below, which reduces in a type of its own.
  if (axes.size() == 1 && axes.front() == self.dim() - 1) {
    if ((kind == ReductionKind::MAX || kind == ReductionKind::MIN) &&
        out_dtype == DType::Float32 && acc_dtype == DType::Float32) {
      if (auto result = rowExtremumShader(
              self, out_shape, kind == ReductionKind::MAX)) {
        return *result;
      }
    }
    if ((kind == ReductionKind::ALL || kind == ReductionKind::ANY) &&
        out_dtype == DType::Bool) {
      if (auto result = rowPredicateShader(
              self, out_shape, kind == ReductionKind::ALL)) {
        return *result;
      }
    }
  }

  Tensor self_c = materializeForGraph(self);
  Tensor output = Tensor(out_shape, out_dtype, self.device());
  if (output.numel() == 0) {
    return output;
  }

  // A partial reduction keeps the rank of its input and leaves a size-1
  // dimension where each reduced axis was, so its result is written straight
  // into a buffer described with that shape and nothing stands between the
  // reduction and the write.  The squeezed shape the caller asked for holds the
  // same elements in the same order, which is why one graph serves keepdim
  // either way.
  // A reduction over an array of two or more axes faults on the macOS 14
  // runtime, so those reduce one axis at a time, highest axis first; the rank is
  // kept, so the axes of the input still name the same dimensions one after
  // another.  A reduction over every axis leaves no rank of its own and is
  // reshaped up to the shape asked for.
  const bool reduce_all = (static_cast<int64_t>(axes.size()) == self.dim());
  const bool array_form =
      !reduce_all && axes.size() == 1 &&
      (kind == ReductionKind::SUM || kind == ReductionKind::PROD ||
       kind == ReductionKind::MEAN || kind == ReductionKind::ALL ||
       kind == ReductionKind::ANY);
  const std::vector<int64_t> apparent_shape =
      compute_out_shape(in_shape, axes, true);
  std::vector<int64_t> graph_shape =
      out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
  if (!reduce_all) {
    graph_shape = apparent_shape;
  }

  // The graph depends on the input shape, the axes actually reduced, the shape
  // its result is written with, the dtypes and the kind, so the same reduction
  // on later batches reuses it.
  const std::string key =
      "reduce|" + shapeKey(in_shape) + "|" + shapeKey(axes) + "|" +
      shapeKey(graph_shape) + "|" + dtypeKey(self_c.dtype()) + "|" +
      dtypeKey(acc_dtype) + "|" + dtypeKey(out_dtype) + "|" +
      std::to_string(static_cast<int>(kind));

  @autoreleasepool {
    MPSShape* inShape = getMPSShape(in_shape);
    const MPSDataType in_dt = getMPSDataType(self_c.dtype());
    const MPSDataType acc_dt = getMPSDataType(acc_dtype);
    const MPSDataType out_dt = getMPSDataType(out_dtype);

    CachedGraph* entry = findCachedGraph(key);
    if (entry == nullptr) {
      MPSGraph* graph = [[MPSGraph alloc] init];

      MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                            dataType:in_dt
                                                 name:@"x"];
      MPSGraphTensor* xc = (self_c.dtype() == acc_dtype)
          ? x
          : [graph castTensor:x toType:acc_dt name:@"cast_in"];

      MPSGraphTensor* reduced = nil;
      NSMutableArray<NSNumber*>* axesArray =
          [NSMutableArray arrayWithCapacity:axes.size()];
      for (int64_t a : axes) {
        [axesArray addObject:@(a)];
      }
      // A reduction over one axis takes the axes array; a reduction over every
      // axis takes the nil-axes form, because an explicit all-axes reduction is
      // not reliable on older SDKs; the rest take one axis at a time, highest
      // axis first, each keeping the rank of the tensor before it.
      switch (kind) {
        case ReductionKind::SUM:
        case ReductionKind::PROD:
        case ReductionKind::MEAN: {
          if (array_form) {
            if (kind == ReductionKind::SUM) {
              reduced =
                  [graph reductionSumWithTensor:xc axes:axesArray name:nil];
            } else if (kind == ReductionKind::PROD) {
              reduced =
                  [graph reductionProductWithTensor:xc axes:axesArray name:nil];
            } else {
              reduced = [graph meanOfTensor:xc axes:axesArray name:nil];
            }
          } else if (reduce_all) {
            if (kind == ReductionKind::SUM) {
              reduced = [graph reductionSumWithTensor:xc axes:nil name:nil];
            } else if (kind == ReductionKind::PROD) {
              reduced = [graph reductionProductWithTensor:xc axes:nil name:nil];
            } else {
              reduced = [graph meanOfTensor:xc axes:nil name:nil];
            }
          } else {
            reduced = xc;
            for (NSNumber* axis in [axesArray reverseObjectEnumerator]) {
              if (kind == ReductionKind::SUM) {
                reduced = [graph reductionSumWithTensor:reduced
                                             axis:[axis integerValue]
                                                 name:nil];
              } else if (kind == ReductionKind::PROD) {
                reduced = [graph reductionProductWithTensor:reduced
                                                      axis:[axis integerValue]
                                                      name:nil];
              } else {
                reduced = [graph meanOfTensor:reduced
                                         axes:@[ axis ]
                                         name:nil];
              }
            }
          }
          break;
        }
        case ReductionKind::MAX:
        case ReductionKind::MIN: {
          // MIN is computed as -MAX(-x) so only the maximum selector is
          // required.  The maximum selector is taken over one axis at a time:
          // an array of axes is not accepted by it.
          reduced = xc;
          if (kind == ReductionKind::MIN) {
            reduced = [graph negativeWithTensor:reduced name:nil];
          }
          for (NSNumber* axis in [axesArray reverseObjectEnumerator]) {
            reduced = [graph reductionMaximumWithTensor:reduced
                                                  axis:[axis integerValue]
                                                  name:nil];
          }
          if (kind == ReductionKind::MIN) {
            reduced = [graph negativeWithTensor:reduced name:nil];
          }
          break;
        }
        case ReductionKind::ALL:
        case ReductionKind::ANY: {
          // Cast to bool first, then reduce with the logical selectors: over the
          // axes array for one axis, one axis at a time for the rest.
          const MPSDataType bool_dt = MPSDataTypeBool;
          reduced = ([xc dataType] == bool_dt)
              ? xc
              : [graph castTensor:xc toType:bool_dt name:@"cast_bool"];
          if (array_form) {
            reduced = kind == ReductionKind::ALL
                ? [graph reductionAndWithTensor:reduced
                                           axes:axesArray
                                           name:nil]
                : [graph reductionOrWithTensor:reduced
                                          axes:axesArray
                                          name:nil];
          } else {
            for (NSNumber* axis in [axesArray reverseObjectEnumerator]) {
              if (kind == ReductionKind::ALL) {
                reduced = [graph reductionAndWithTensor:reduced
                                                   axis:[axis integerValue]
                                                   name:nil];
              } else {
                reduced = [graph reductionOrWithTensor:reduced
                                                  axis:[axis integerValue]
                                                  name:nil];
              }
            }
          }
          break;
        }
      }

      MPSGraphTensor* outTensor = reduced;
      if (reduce_all && !out_shape.empty()) {
        outTensor = [graph reshapeTensor:reduced
                                withShape:getMPSShape(graph_shape)
                                     name:nil];
      }
      outTensor = ([outTensor dataType] == out_dt)
          ? outTensor
          : [graph castTensor:outTensor toType:out_dt name:@"cast_out"];

      entry = new CachedGraph{graph, [x retain], nil, nil,
                              [outTensor retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:in_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:getMPSShape(graph_shape)
                                             dataType:out_dt] autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

} // namespace

// ---------------------------------------------------------------------------
// sum / mean
// ---------------------------------------------------------------------------

Tensor sum_kernel(const Tensor& self, DType dtype) {
  const DType out_dtype = resolve_sum_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, {}, false, out_dtype, acc_dtype, ReductionKind::SUM);
}

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                      bool keepdim, DType dtype) {
  const DType out_dtype = resolve_sum_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, dims, keepdim, out_dtype, acc_dtype,
                     ReductionKind::SUM);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
  const DType out_dtype = resolve_mean_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, {}, false, out_dtype, acc_dtype,
                     ReductionKind::MEAN);
}

Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
  const DType out_dtype = resolve_mean_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, dims, keepdim, out_dtype, acc_dtype,
                     ReductionKind::MEAN);
}

// ---------------------------------------------------------------------------
// max / min
// ---------------------------------------------------------------------------

Tensor max_kernel(const Tensor& self) {
  return reductionOp(self, {}, false, self.dtype(), self.dtype(),
                     ReductionKind::MAX);
}

Tensor min_kernel(const Tensor& self) {
  return reductionOp(self, {}, false, self.dtype(), self.dtype(),
                     ReductionKind::MIN);
}

// ---------------------------------------------------------------------------
// prod
// ---------------------------------------------------------------------------

Tensor prod_kernel(const Tensor& self, DType dtype) {
  const DType out_dtype = resolve_sum_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, {}, false, out_dtype, acc_dtype,
                     ReductionKind::PROD);
}

Tensor prod_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                       bool keepdim, DType dtype) {
  const DType out_dtype = resolve_sum_dtype(dtype, self.dtype());
  const DType acc_dtype = accumulation_dtype(out_dtype);
  return reductionOp(self, dims, keepdim, out_dtype, acc_dtype,
                     ReductionKind::PROD);
}

// ---------------------------------------------------------------------------
// amax / amin
// ---------------------------------------------------------------------------

Tensor amax_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                   bool keepdim) {
  return reductionOp(self, dims, keepdim, self.dtype(), self.dtype(),
                     ReductionKind::MAX);
}

Tensor amin_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                   bool keepdim) {
  return reductionOp(self, dims, keepdim, self.dtype(), self.dtype(),
                     ReductionKind::MIN);
}

// ---------------------------------------------------------------------------
// all / any
// ---------------------------------------------------------------------------

Tensor all_kernel(const Tensor& self) {
  return reductionOp(self, {}, false, DType::Bool, DType::Bool,
                     ReductionKind::ALL);
}

Tensor all_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                      bool keepdim) {
  return reductionOp(self, dims, keepdim, DType::Bool, DType::Bool,
                     ReductionKind::ALL);
}

Tensor all_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
  return all_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

Tensor any_kernel(const Tensor& self) {
  return reductionOp(self, {}, false, DType::Bool, DType::Bool,
                     ReductionKind::ANY);
}

Tensor any_dim_kernel(const Tensor& self, const std::vector<int64_t>& dims,
                      bool keepdim) {
  return reductionOp(self, dims, keepdim, DType::Bool, DType::Bool,
                     ReductionKind::ANY);
}

Tensor any_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
  return any_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

// ---------------------------------------------------------------------------
// var / std
// ---------------------------------------------------------------------------

// Computes variance (or std when `is_std`) over `dims` (empty = all dims).
// The biased variance is taken from the variance selector when one axis is
// reduced and from the E[x^2] - E[x]^2 sums when several are; Bessel's
// correction is applied afterwards as a multiplication by
// N / (N - correction) (N = number of reduced elements).
Tensor varStdOp(const Tensor& self, const std::vector<int64_t>& dims,
                bool keepdim, int64_t correction, bool is_std) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS var/std requires input on the MPS device");
  }
  if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
    TP_THROW(RuntimeError,
             "std and var only support floating point and complex dtypes");
  }
  const std::vector<int64_t> in_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  std::vector<int64_t> axes = normalize_dims(dims, self.dim());
  if (axes.empty()) {
    for (int64_t i = 0; i < self.dim(); ++i) {
      axes.push_back(i);
    }
  }
  const std::vector<int64_t> out_shape =
      compute_out_shape(in_shape, axes, keepdim);

  // A 0-d input reduced over all dims: var of a single value with
  // correction 0 is 0; with any correction it is NaN (0/0).
  if (self.dim() == 0) {
    Tensor out = Tensor(out_shape, self.dtype(), self.device());
    if (correction == 0) {
      fill_kernel(out, 0);
    } else {
      fill_kernel(out, std::numeric_limits<float>::quiet_NaN());
    }
    return out;
  }

  if (self.numel() == 0) {
    Tensor out = Tensor(out_shape, self.dtype(), self.device());
    if (out.numel() > 0) {
      fill_kernel(out, std::numeric_limits<float>::quiet_NaN());
    }
    return out;
  }

  int64_t correction_n = 1;
  for (int64_t a : axes) {
    correction_n *= in_shape[a];
  }
  const int64_t dof_i = std::max<int64_t>(0, correction_n - correction);
  const double bessel =
      static_cast<double>(correction_n) / static_cast<double>(dof_i);
  // (inf when dof == 0)

  Tensor self_c = materializeForGraph(self);
  Tensor output = Tensor(out_shape, self.dtype(), self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(in_shape);
    const MPSDataType dt = getMPSDataType(self_c.dtype());
    MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                          dataType:dt
                                               name:@"x"];
    NSMutableArray<NSNumber*>* axesArray =
        [NSMutableArray arrayWithCapacity:axes.size()];
    for (int64_t a : axes) {
      [axesArray addObject:@(a)];
    }
    // A reduction over one axis takes the axes array and keeps the rank of its
    // input, the reduced axis surviving with size 1; a reduction over every axis
    // takes the nil-axes form.  Several axes are reduced one at a time through
    // E[x^2]-E[x]^2 sums, because an array of two or more axes faults, and the
    // multi-axis variance selector is unreliable on older SDKs.
    const bool reduce_all = (static_cast<int64_t>(axes.size()) == self.dim());
    const bool one_axis = (!reduce_all && axes.size() == 1);
    const std::vector<int64_t> apparent_shape =
        compute_out_shape(in_shape, axes, true);
    MPSGraphTensor* variance = nil;
    if (reduce_all) {
      variance = [graph varianceOfTensor:x axes:nil name:nil];
    } else if (one_axis) {
      variance = [graph varianceOfTensor:x axes:axesArray name:nil];
    } else {
      const int64_t count = correction_n;
      MPSGraphTensor* sum = x;
      MPSGraphTensor* sumsq = [graph multiplicationWithPrimaryTensor:x
                                                     secondaryTensor:x
                                                                name:nil];
      for (NSNumber* axis in [axesArray reverseObjectEnumerator]) {
        sum = [graph reductionSumWithTensor:sum
                                       axis:[axis integerValue]
                                       name:nil];
        sumsq = [graph reductionSumWithTensor:sumsq
                                         axis:[axis integerValue]
                                         name:nil];
      }
      MPSGraphTensor* countT = [graph constantWithScalar:(double)count
                                                shape:@[ @1 ]
                                             dataType:dt];
      MPSGraphTensor* mean = [graph divisionWithPrimaryTensor:sum
                                              secondaryTensor:countT
                                                         name:nil];
      MPSGraphTensor* meanSq = [graph multiplicationWithPrimaryTensor:mean
                                                       secondaryTensor:mean
                                                                  name:nil];
      MPSGraphTensor* meanSqTerm =
          [graph divisionWithPrimaryTensor:sumsq
                           secondaryTensor:countT
                                      name:nil];
      variance = [graph subtractionWithPrimaryTensor:meanSqTerm
                                     secondaryTensor:meanSq
                                                name:nil];
    }
    MPSGraphTensor* corrected = variance;
    if (correction != 0) {
      MPSGraphTensor* besselTensor =
          [graph constantWithScalar:bessel
                              shape:@[ @1 ]
                           dataType:dt];
      corrected = [graph multiplicationWithPrimaryTensor:variance
                                         secondaryTensor:besselTensor
                                                    name:nil];
    }
    MPSGraphTensor* reduced =
        is_std ? [graph squareRootWithTensor:corrected name:nil] : corrected;

    // A partial reduction is already in the shape written to, keepdim included.
    // A full reduction leaves no rank, so its single element is described as [1]
    // and a kept-dim result reshapes the scalar up to the requested shape.
    const std::vector<int64_t> graph_shape =
        reduce_all ? (out_shape.empty() ? std::vector<int64_t>{1} : out_shape)
                   : apparent_shape;
    MPSGraphTensor* outTensor = reduced;
    if (reduce_all && !out_shape.empty()) {
      outTensor = [graph reshapeTensor:reduced
                             withShape:getMPSShape(graph_shape)
                                  name:nil];
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> outBuf = mtlBufferOf(output);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:getMPSShape(graph_shape)
                                             dataType:dt] autorelease];
    NSDictionary* feeds = @{x : inData};
    NSDictionary* results = @{outTensor : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return output;
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
  return varStdOp(self, {}, false, correction, false);
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                      int64_t correction, bool keepdim) {
  return varStdOp(self, dim, keepdim, correction, false);
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
  return varStdOp(self, {}, false, correction, true);
}

Tensor std_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                      int64_t correction, bool keepdim) {
  return varStdOp(self, dim, keepdim, correction, true);
}

// ---------------------------------------------------------------------------
// argmax / argmin / max.dim / min.dim
//
// argmax and argmin need only the graph's index reduction. max.dim and min.dim
// take their values and indices from one graph, written by a single dispatch.
// ---------------------------------------------------------------------------

namespace {

// Index of the extremum along `dim`, produced by the graph's index
// reduction and reshaped to the requested output shape.
Tensor arg_reduce_impl(const Tensor& self, int64_t dim, bool keepdim,
                       bool is_max) {
  Tensor self_c = self.contiguous();
  const std::vector<int64_t> in_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t ndim = self.dim();
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    TP_THROW(IndexError, "Dimension out of range");
  }
  std::vector<int64_t> out_shape;
  if (keepdim) {
    out_shape = in_shape;
    out_shape[dim] = 1;
  } else {
    for (int64_t i = 0; i < ndim; ++i) {
      if (i != dim) out_shape.push_back(in_shape[i]);
    }
  }
  Tensor indices = Tensor::empty(out_shape, DType::Int64, self.device());
  if (indices.numel() == 0) {
    return indices;
  }
  // Over the last axis, the whole of a row belongs to one SIMD group, so the
  // position of the extremum is a single dispatch and the index never leaves
  // the row it was found in.
  if (dim == ndim - 1) {
    if (auto result = rowArgReduceShader(self_c, out_shape, is_max)) {
      return *result;
    }
  }
  // The index reduction keeps the rank of its input and leaves a size-1
  // dimension where the axis was, so a partial one is written in that shape.  A
  // rank-1 input reduced over its only axis leaves no rank, so that result is
  // described with a single-element shape.
  const bool reduce_all = (ndim == 1);
  std::vector<int64_t> apparent_shape = in_shape;
  apparent_shape[dim] = 1;
  std::vector<int64_t> graph_shape =
      out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
  if (!reduce_all) {
    graph_shape = apparent_shape;
  }
  const std::string key =
      "argreduce|" + shapeKey(in_shape) + "|" + std::to_string(dim) + "|" +
      shapeKey(graph_shape) + "|" + dtypeKey(self_c.dtype()) + "|" +
      (is_max ? "max" : "min");

  @autoreleasepool {
    MPSShape* inShape = getMPSShape(in_shape);
    const MPSDataType in_dt = getMPSDataType(self_c.dtype());
    const MPSDataType out_dt = MPSDataTypeInt64;

    CachedGraph* entry = findCachedGraph(key);
    if (entry == nullptr) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                            dataType:in_dt
                                                 name:@"x"];
      // The index reduction has no boolean form; a bool input carries the
      // same 0/1 ordering once widened.
      MPSGraphTensor* xc =
          ([x dataType] == MPSDataTypeBool)
          ? [graph castTensor:x toType:MPSDataTypeInt32 name:@"cast_bool"]
          : x;
      MPSGraphTensor* reduced = nil;
      if (is_max) {
        reduced = [graph reductionArgMaximumWithTensor:xc axis:dim name:nil];
      } else {
        reduced = [graph reductionArgMinimumWithTensor:xc axis:dim name:nil];
      }
      MPSGraphTensor* result = reduced;
      if (reduce_all && !out_shape.empty()) {
        result = [graph reshapeTensor:reduced
                            withShape:getMPSShape(out_shape)
                                 name:nil];
      }
      if ([result dataType] != out_dt) {
        result = [graph castTensor:result toType:out_dt name:@"cast_out"];
      }
      entry = new CachedGraph{graph, [x retain], nil, nil, [result retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> outBuf = mtlBufferOf(indices);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:in_dt] autorelease];
    MPSGraphTensorData* outData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                shape:getMPSShape(graph_shape)
                                             dataType:out_dt] autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : outData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return indices;
}

// Values and indices of max.dim/min.dim out of one graph, so the pair costs a
// single dispatch instead of a value reduction followed by an index reduction.
// Both branches read the same placeholder, which also means an input only has
// to be brought into graph-ready storage once.
std::tuple<Tensor, Tensor> valueIndexReduce(const Tensor& self, int64_t dim,
                                            bool keepdim, bool is_max) {
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "MPS reduction requires input on the MPS device");
  }
  const std::vector<int64_t> in_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  // The same checks the value reduction applies to a requested axis, so an
  // out-of-range dim is rejected the same way here.
  const std::vector<int64_t> axes = normalize_dims({dim}, self.dim());
  const int64_t axis = axes[0];
  const std::vector<int64_t> out_shape =
      compute_out_shape(in_shape, axes, keepdim);

  if (self.numel() == 0) {
    // An input with no elements has no extremum to locate, and the value
    // reduction is the one that says so.
    Tensor values = is_max ? amax_kernel(self, axes, keepdim)
                           : amin_kernel(self, axes, keepdim);
    Tensor indices = arg_reduce_impl(self, axis, keepdim, is_max);
    return std::make_tuple(values, indices);
  }

  // Over the last axis of a float operand one pass answers both halves of the
  // pair, so the value is not reduced once and the position of that value a
  // second time, and the two cannot disagree about which element answered.
  if (axis == self.dim() - 1) {
    if (auto result = rowValueIndexShader(self, out_shape, is_max)) {
      return *result;
    }
  }

  Tensor self_c = materializeForGraph(self);
  const DType value_dtype = self_c.dtype();
  Tensor values = Tensor::empty(out_shape, value_dtype, self.device());
  Tensor indices = Tensor::empty(out_shape, DType::Int64, self.device());
  // The reductions keep the rank of their input and leave a size-1 dimension
  // where the axis was, so a partial one writes in that shape.  A rank-1 input
  // reduced over its only axis leaves no rank, so those results are described
  // with a single-element shape.
  const bool reduce_all = (self.dim() == 1);
  std::vector<int64_t> apparent_shape = in_shape;
  apparent_shape[axis] = 1;
  std::vector<int64_t> graph_shape =
      out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
  if (!reduce_all) {
    graph_shape = apparent_shape;
  }
  const std::string key =
      "valueindexreduce|" + shapeKey(in_shape) + "|" + std::to_string(axis) +
      "|" + shapeKey(graph_shape) + "|" + dtypeKey(value_dtype) + "|" +
      (is_max ? "max" : "min");

  @autoreleasepool {
    // The values come out in the input dtype, so one data type describes both
    // the placeholder and the value output.
    const MPSDataType value_dt = getMPSDataType(value_dtype);
    MPSShape* inShape = getMPSShape(in_shape);

    CachedGraph* entry = findCachedGraph(key);
    if (entry == nullptr) {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* x = [graph placeholderWithShape:inShape
                                             dataType:value_dt
                                                 name:@"x"];
      // The index reduction has no boolean form. Widening once feeds both
      // branches the same values and the extremum is narrowed again with the
      // output cast, so a bool input keeps a bool result.
      MPSGraphTensor* xc =
          ([x dataType] == MPSDataTypeBool)
          ? [graph castTensor:x toType:MPSDataTypeInt32 name:@"cast_bool"]
          : x;
      const NSInteger axisIndex = static_cast<NSInteger>(axis);
      MPSGraphTensor* value_reduced = nil;
      MPSGraphTensor* index_reduced = nil;
      if (is_max) {
        value_reduced =
            [graph reductionMaximumWithTensor:xc axis:axisIndex name:nil];
        index_reduced =
            [graph reductionArgMaximumWithTensor:xc axis:axisIndex name:nil];
      } else {
        // MIN stays -MAX(-x), the same shape the value reductions use; the
        // index reduction has its own minimum selector.
        MPSGraphTensor* negated = [graph negativeWithTensor:xc name:nil];
        value_reduced =
            [graph negativeWithTensor:[graph reductionMaximumWithTensor:negated
                                                                   axis:axisIndex
                                                                   name:nil]
                                 name:nil];
        index_reduced =
            [graph reductionArgMinimumWithTensor:xc axis:axisIndex name:nil];
      }
      MPSGraphTensor* value_out = value_reduced;
      MPSGraphTensor* index_out = index_reduced;
      if (reduce_all && !out_shape.empty()) {
        MPSShape* outMPSShape = getMPSShape(out_shape);
        value_out =
            [graph reshapeTensor:value_out withShape:outMPSShape name:nil];
        index_out =
            [graph reshapeTensor:index_out withShape:outMPSShape name:nil];
      }
      value_out = ([value_out dataType] == value_dt)
          ? value_out
          : [graph castTensor:value_out toType:value_dt name:@"cast_value"];
      if ([index_out dataType] != MPSDataTypeInt64) {
        index_out = [graph castTensor:index_out
                               toType:MPSDataTypeInt64
                                 name:@"cast_index"];
      }
      entry = new CachedGraph{graph, [x retain], nil, nil,
                              [value_out retain], [index_out retain]};
      storeCachedGraph(key, entry);
      entry = findCachedGraph(key);
    }

    id<MTLBuffer> inBuf = mtlBufferOf(self_c);
    id<MTLBuffer> valueBuf = mtlBufferOf(values);
    id<MTLBuffer> indexBuf = mtlBufferOf(indices);
    MPSShape* graphMPSShape = getMPSShape(graph_shape);
    MPSGraphTensorData* inData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                shape:inShape
                                             dataType:value_dt] autorelease];
    MPSGraphTensorData* valueData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:valueBuf
                                                shape:graphMPSShape
                                             dataType:value_dt] autorelease];
    MPSGraphTensorData* indexData =
        [[[MPSGraphTensorData alloc] initWithMTLBuffer:indexBuf
                                                shape:graphMPSShape
                                             dataType:MPSDataTypeInt64]
            autorelease];
    NSDictionary* feeds = @{entry->input0 : inData};
    NSDictionary* results = @{entry->result : valueData,
                              entry->result2 : indexData};
    getCurrentMPSStream()->executeMPSGraph(entry->graph, feeds, results,
                                          SyncType::COMMIT_ADAPTIVE);
  }
  return std::make_tuple(values, indices);
}

} // namespace

Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim,
                     bool keepdim) {
  if (!dim.has_value()) {
    // Flatten and scan the whole tensor.
    Tensor flat = self.contiguous().reshape({self.numel()});
    return arg_reduce_impl(flat, 0, false, true);
  }
  return arg_reduce_impl(self, *dim, keepdim, true);
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim,
                     bool keepdim) {
  if (!dim.has_value()) {
    Tensor flat = self.contiguous().reshape({self.numel()});
    return arg_reduce_impl(flat, 0, false, false);
  }
  return arg_reduce_impl(self, *dim, keepdim, false);
}

std::tuple<Tensor, Tensor> max_dim_kernel(const Tensor& self, int64_t dim,
                                            bool keepdim) {
  return valueIndexReduce(self, dim, keepdim, true);
}

std::tuple<Tensor, Tensor> min_dim_kernel(const Tensor& self, int64_t dim,
                                            bool keepdim) {
  return valueIndexReduce(self, dim, keepdim, false);
}

TENSORPLAY_LIBRARY_IMPL(MPS, ReductionKernels) {
  m.impl("sum", sum_kernel);
  m.impl("sum.dim_IntList", sum_dim_kernel);
  m.impl("mean", mean_kernel);
  m.impl("mean.dim", mean_dim_kernel);
  m.impl("max", max_kernel);
  m.impl("min", min_kernel);
  m.impl("prod", prod_kernel);
  m.impl("prod.dim_IntList", prod_dim_kernel);
  m.impl("amax", amax_kernel);
  m.impl("amin", amin_kernel);
  m.impl("all", all_kernel);
  m.impl("all.dim", all_dim_int_kernel);
  m.impl("any", any_kernel);
  m.impl("any.dim", any_dim_int_kernel);
  m.impl("var", var_kernel);
  m.impl("var.dim", var_dim_kernel);
  m.impl("std", std_kernel);
  m.impl("std.dim", std_dim_kernel);
  m.impl("argmax", argmax_kernel);
  m.impl("argmin", argmin_kernel);
  m.impl("max.dim", max_dim_kernel);
  m.impl("min.dim", min_dim_kernel);
}

} // namespace mps
} // namespace tensorplay
