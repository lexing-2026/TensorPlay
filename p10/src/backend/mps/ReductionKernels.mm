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

  Tensor self_c = materializeForGraph(self);
  Tensor output = Tensor(out_shape, out_dtype, self.device());
  if (output.numel() == 0) {
    return output;
  }

  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph new] autorelease];
    MPSShape* inShape = getMPSShape(in_shape);
    const MPSDataType in_dt = getMPSDataType(self_c.dtype());
    const MPSDataType acc_dt = getMPSDataType(acc_dtype);
    const MPSDataType out_dt = getMPSDataType(out_dtype);

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
    // Reducing every axis with an explicit axes array trips an MPSGraph bug
    // on older SDKs; use the nil-axes form (reduce all) for full reductions,
    // matching the scalar-output reduction pattern used elsewhere.
    const bool reduce_all = (static_cast<int64_t>(axes.size()) == self.dim());
    switch (kind) {
      case ReductionKind::SUM:
        reduced = reduce_all
            ? [graph reductionSumWithTensor:xc axes:nil name:nil]
            : [graph reductionSumWithTensor:xc axes:axesArray name:nil];
        break;
      case ReductionKind::PROD:
        reduced = reduce_all
            ? [graph reductionProductWithTensor:xc axes:nil name:nil]
            : [graph reductionProductWithTensor:xc axes:axesArray name:nil];
        break;
      case ReductionKind::MEAN:
        reduced = reduce_all
            ? [graph meanOfTensor:xc axes:nil name:nil]
            : [graph meanOfTensor:xc axes:axesArray name:nil];
        break;
      case ReductionKind::MAX:
      case ReductionKind::MIN: {
        // Reduce one axis at a time, highest axis first: removing a higher
        // axis does not shift the indices of the lower remaining axes.
        // MIN is computed as -MAX(-x) so only the confirmed
        // reductionMaximumWithTensor selector is required.
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
        // Cast to bool first, then reduce one axis at a time (highest
        // first) with the logical reduction selectors.
        const MPSDataType bool_dt = MPSDataTypeBool;
        reduced = ([xc dataType] == bool_dt)
            ? xc
            : [graph castTensor:xc toType:bool_dt name:@"cast_bool"];
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
        break;
      }
    }

    // The graph removes reduced axes. When keepdim is requested (or the
    // result has non-scalar shape) reshape the graph output so its shape
    // matches the output buffer description exactly. A fully-reduced scalar
    // is left unreshaped (reshaping an unranked tensor is unsupported on
    // older SDKs); its single-element buffer is described with shape [1].
    const std::vector<int64_t> graph_shape =
        out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    MPSGraphTensor* outTensor = reduced;
    if (!out_shape.empty()) {
      outTensor = [graph reshapeTensor:reduced
                              withShape:getMPSShape(graph_shape)
                                   name:nil];
    }
    outTensor = ([outTensor dataType] == out_dt)
        ? outTensor
        : [graph castTensor:outTensor toType:out_dt name:@"cast_out"];

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
    NSDictionary* feeds = @{x : inData};
    NSDictionary* results = @{outTensor : outData};
    getCurrentMPSStream()->executeMPSGraph(graph, feeds, results,
                                          SyncType::COMMIT_AND_WAIT);
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
// The biased variance comes from the confirmed varianceOfTensor selector;
// Bessel's correction is applied afterwards as a multiplication by
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
    // Full reductions use the nil-axes form to avoid the explicit all-axes
    // reduction bug on older SDKs (same rationale as reductionOp).
    const bool reduce_all = (static_cast<int64_t>(axes.size()) == self.dim());
    MPSGraphTensor* variance =
        reduce_all
            ? [graph varianceOfTensor:x axes:nil name:nil]
            : [graph varianceOfTensor:x axes:axesArray name:nil];
    MPSGraphTensor* corrected = variance;
    if (correction != 0) {
      MPSGraphTensor* besselTensor =
          [graph constantWithScalar:bessel dataType:dt];
      corrected = [graph multiplicationWithPrimaryTensor:variance
                                         secondaryTensor:besselTensor
                                                    name:nil];
    }
    MPSGraphTensor* reduced =
        is_std ? [graph squareRootWithTensor:corrected name:nil] : corrected;

    const std::vector<int64_t> graph_shape =
        out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    MPSGraphTensor* outTensor = reduced;
    if (!out_shape.empty()) {
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
                                          SyncType::COMMIT_AND_WAIT);
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
// Values for max.dim/min.dim come from the GPU reduction selectors; the
// index tensors are found with a host-side scan over the shared-memory
// mapping (correct, and cheap enough for the index-only pass).
// ---------------------------------------------------------------------------

namespace {

double read_element(const void* base, int64_t idx, DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return static_cast<const float*>(base)[idx];
    case DType::Float64:
      return static_cast<const double*>(base)[idx];
    case DType::Float16:
      return static_cast<float>(static_cast<const Half*>(base)[idx]);
    case DType::BFloat16:
      return static_cast<float>(static_cast<const BFloat16*>(base)[idx]);
    case DType::Int64:
      return static_cast<double>(static_cast<const int64_t*>(base)[idx]);
    case DType::Int32:
      return static_cast<double>(static_cast<const int32_t*>(base)[idx]);
    case DType::Int16:
      return static_cast<double>(static_cast<const int16_t*>(base)[idx]);
    case DType::Int8:
      return static_cast<double>(static_cast<const int8_t*>(base)[idx]);
    case DType::UInt8:
      return static_cast<double>(static_cast<const uint8_t*>(base)[idx]);
    default:
      TP_THROW(NotImplementedError,
               "argmax/argmin: unsupported dtype on MPS");
  }
}

// Host-side scan for the index of the first extremum along `dim`.
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
  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= in_shape[i];
  for (int64_t i = dim + 1; i < ndim; ++i) inner *= in_shape[i];
  const int64_t dim_size = in_shape[dim];
  const DType dt = self_c.dtype();
  const void* data = self_c.data_ptr();
  int64_t* ip = indices.data_ptr<int64_t>();
  const double init = is_max ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
  for (int64_t o = 0; o < outer; ++o) {
    const void* base = static_cast<const char*>(data) +
                       (o * dim_size) * inner * self_c.itemsize();
    int64_t* out_row = ip + o * inner;
    for (int64_t i = 0; i < inner; ++i) {
      double best = init;
      int64_t best_idx = 0;
      for (int64_t k = 0; k < dim_size; ++k) {
        const double v = read_element(
            static_cast<const char*>(base) + k * inner * self_c.itemsize(), i,
            dt);
        if (is_max ? (v > best) : (v < best)) {
          best = v;
          best_idx = k;
        }
      }
      out_row[i] = best_idx;
    }
  }
  return indices;
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
  Tensor values = amax_kernel(self, std::vector<int64_t>{dim}, keepdim);
  Tensor indices = arg_reduce_impl(self, dim, keepdim, true);
  return std::make_tuple(values, indices);
}

std::tuple<Tensor, Tensor> min_dim_kernel(const Tensor& self, int64_t dim,
                                            bool keepdim) {
  Tensor values = amin_kernel(self, std::vector<int64_t>{dim}, keepdim);
  Tensor indices = arg_reduce_impl(self, dim, keepdim, false);
  return std::make_tuple(values, indices);
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