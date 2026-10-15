#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

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

enum class ReductionKind { SUM, PROD, MEAN, MAX, MIN };

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
    switch (kind) {
      case ReductionKind::SUM:
        reduced = [graph reductionSumWithTensor:xc axes:axesArray name:nil];
        break;
      case ReductionKind::PROD:
        reduced = [graph reductionProductWithTensor:xc axes:axesArray name:nil];
        break;
      case ReductionKind::MEAN:
        reduced = [graph meanOfTensor:xc axes:axesArray name:nil];
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
    }

    // The graph removes reduced axes. When keepdim is requested (or the
    // result is a scalar) reshape the graph output so its shape matches the
    // output buffer description exactly.
    const std::vector<int64_t> graph_shape =
        out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    MPSGraphTensor* reshaped =
        [graph reshapeTensor:reduced withShape:getMPSShape(graph_shape) name:nil];
    MPSGraphTensor* outTensor = ([reshaped dataType] == out_dt)
        ? reshaped
        : [graph castTensor:reshaped toType:out_dt name:@"cast_out"];

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
}

} // namespace mps
} // namespace tensorplay