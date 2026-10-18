#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Context.h"
#include "TypePromotion.h"
#include "DTypeNames.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/OperationUtils.h"

#include <cmath>
#include <limits>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

// ---------------------------------------------------------------------------
// logsumexp
//
// Computed as max + log(sum(exp(x - max))) over the given dims.  A slice whose
// maximum is infinite has no finite shift to work from, so it answers with that
// maximum through a where.
// ---------------------------------------------------------------------------

Tensor logsumexp_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                         bool keepdim) {
  const bool integral = isIntegralType(self.dtype(), true);
  if (!isFloatingType(self.dtype()) && !integral) {
    TP_THROW(RuntimeError, "logsumexp(): Expected floating point type, got ",
             scalarTypeName(self.dtype()));
  }
  const DType out_dtype =
      integral ? globalContext().defaultDType() : self.dtype();
  // Integral inputs are widened to the default floating dtype because the
  // exponential step requires a floating-point type.
  Tensor x = integral ? self.to(out_dtype) : self;

  // Reducing the innermost axis alone is one pass down each row, which the row
  // shader answers in one launch: a maximum pass and a shifted-exponential sum
  // pass walk the same row without materializing the shifted intermediate.
  std::vector<int64_t> reduced_dims = dim;
  if (reduced_dims.empty()) {
    for (int64_t i = 0; i < self.dim(); ++i) {
      reduced_dims.push_back(i);
    }
  }
  const int64_t nd = self.dim();
  for (int64_t& d : reduced_dims) {
    if (d < 0) d += nd;
  }
  if (reduced_dims.size() == 1 && reduced_dims[0] == nd - 1) {
    std::vector<int64_t> out_shape =
        static_cast<std::vector<int64_t>>(x.shape());
    if (keepdim) {
      out_shape.back() = 1;
    } else {
      out_shape.pop_back();
    }
    if (auto fused = logsumexpRowShader(x, out_shape)) {
      return *fused;
    }
  }

  Tensor m = Tensor::amax(x, dim, true);
  Tensor shifted = x - m;
  Tensor s = shifted.exp().sum(dim, true);
  Tensor result = s.log() + m;
  if (out_dtype != result.dtype()) {
    result = result.to(out_dtype);
  }
  const double inf = std::numeric_limits<double>::infinity();
  Tensor inf_mask = m.abs().eq(Scalar(inf));
  result = Tensor::where(inf_mask, m, result);
  if (!keepdim) {
    // squeeze the reduced dims out of the result (m kept them as size 1).
    result = result.squeeze(reduced_dims);
  }
  return result;
}

// ---------------------------------------------------------------------------
// norm / norm.dim
//
// Frobenius/vector norm is (sum(|x|^p))^(1/p); the infinite norms are the
// max/min of |x|.  All building blocks are already registered MPS kernels.
// ---------------------------------------------------------------------------

Tensor norm_kernel(const Tensor& self, double p) {
  if (p == std::numeric_limits<double>::infinity()) {
    return self.abs().max();
  }
  if (p == -std::numeric_limits<double>::infinity()) {
    return self.abs().min();
  }
  return self.abs().pow(Scalar(p)).sum().pow(Scalar(1.0 / p));
}

Tensor norm_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       double p, bool keepdim) {
  if (p == std::numeric_limits<double>::infinity()) {
    return Tensor::amax(self.abs(), dim, keepdim);
  }
  if (p == -std::numeric_limits<double>::infinity()) {
    return Tensor::amin(self.abs(), dim, keepdim);
  }
  return self.abs().pow(Scalar(p)).sum(dim, keepdim).pow(Scalar(1.0 / p));
}

bool equal_kernel(const Tensor& self, const Tensor& src) {
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  return ops::item(ops::all(ops::eq(self, src))).to<bool>();
}

// ---------------------------------------------------------------------------
// item
//
// Copies the single element to a CPU tensor through the unified-memory host
// mapping (the CPU copy kernel handles MPS sources) and reads it back with
// the CPU item kernel.
// ---------------------------------------------------------------------------

Scalar item_mps(const Tensor& self) {
  if (self.numel() != 1) {
    TP_THROW(ValueError, "item() only supported for 1-element tensors");
  }
  const Tensor host = self.to(Device(DeviceType::CPU));
  return host.item();
}

TENSORPLAY_LIBRARY_IMPL(MPS, MiscKernels) {
  m.impl("logsumexp", logsumexp_kernel);
  m.impl("norm", norm_kernel);
  m.impl("norm.dim", norm_dim_kernel);
  m.impl("equal", equal_kernel);
  m.impl("item", item_mps);
}

} // namespace mps
} // namespace tensorplay