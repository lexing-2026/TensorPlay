// Pairwise p-norm distances: pairwise_distance, pdist.
//
// pairwise_distance reduces the broadcasted |x1 - x2| + eps over the last
// dimension through the shared norm entry point.  pdist produces the
// condensed upper-triangle distance vector of a single (N, D) batch.

#include "Tensor.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "cpu/DistanceKernels.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;
namespace ops = tensorplay::tpx::ops;

namespace {

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
}

// Norm selector for pdist_stub/cdist_stub; see cpu/DistanceKernels.h for the
// encoding.  Resolved once per operation, not per row pair.
inline int pdist_mode_code(double p) {
    if (p == 0.0) return 0;
    if (p == 1.0) return 1;
    if (p == 2.0) return 2;
    if (std::isinf(p)) return 3;
    return 4;
}

template <typename T>
Tensor pdist_impl(const Tensor& self, double p) {
    const int64_t n = self.size(0);
    const int64_t width = self.size(1);
    const int64_t outn = n * (n - 1) / 2;
    const DType work_dtype = std::is_same_v<T, double>
        ? DType::Float64
        : DType::Float32;

    if (outn == 0) {
        return Tensor::empty({0}, work_dtype, self.device());
    }
    if (width == 0) {
        return Tensor::zeros({outn}, work_dtype, self.device());
    }

    Tensor input = self.contiguous().to(work_dtype);
    const T* data = input.data_ptr<T>();
    Tensor out = Tensor::empty({outn}, work_dtype, self.device());

    // The parallel vectorized loop is tier-compiled; this hands it the
    // ready-to-run buffers together with the resolved norm selector.
    pdist_stub(DeviceType::CPU, data, out.data_ptr(), n, width, p,
               pdist_mode_code(p), static_cast<int>(work_dtype));

    return out;
}

inline int64_t product_all(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (const int64_t extent : shape) result *= extent;
    return result;
}

template <typename T>
Tensor cdist_impl(const Tensor& x1, const Tensor& x2, double p,
                  std::optional<int64_t> compute_mode,
                  const std::vector<int64_t>& batch_shape,
                  int64_t rows1, int64_t rows2, int64_t width) {
    const int64_t batches = product_all(batch_shape);
    std::vector<int64_t> output_shape = batch_shape;
    output_shape.push_back(rows1);
    output_shape.push_back(rows2);
    const DType work_dtype = std::is_same_v<T, double>
        ? DType::Float64
        : DType::Float32;
    Tensor output = Tensor::empty(output_shape, work_dtype, x1.device());
    if (batches == 0 || rows1 == 0 || rows2 == 0) return output;
    if (width == 0) return output.fill_(Scalar(0));

    std::vector<int64_t> lhs_shape = batch_shape;
    lhs_shape.push_back(rows1);
    lhs_shape.push_back(width);
    std::vector<int64_t> rhs_shape = batch_shape;
    rhs_shape.push_back(rows2);
    rhs_shape.push_back(width);
    Tensor lhs = x1.to(work_dtype).expand(lhs_shape).contiguous().reshape(
        {batches, rows1, width});
    Tensor rhs = x2.to(work_dtype).expand(rhs_shape).contiguous().reshape(
        {batches, rows2, width});

    const int64_t mode = compute_mode.value_or(0);
    if (p == 2.0 &&
        (mode == 1 || (mode == 0 && (rows1 > 25 || rows2 > 25)))) {
        Tensor lhs_norm = ops::sum(ops::mul(lhs, lhs), {-1}, true);
        Tensor rhs_norm = ops::sum(ops::mul(rhs, rhs), {-1}, true);
        Tensor lhs_augmented = ops::cat(
            {ops::mul(lhs, Scalar(-2)), lhs_norm, Tensor::ones_like(lhs_norm)},
            -1);
        Tensor rhs_augmented = ops::cat(
            {rhs, Tensor::ones_like(rhs_norm), rhs_norm}, -1);
        Tensor result = ops::matmul(
            lhs_augmented, ops::transpose(rhs_augmented, -2, -1));
        result.clamp_min_(Scalar(0));
        result.sqrt_();
        return result.reshape(output_shape);
    }

    const T* lhs_data = lhs.data_ptr<T>();
    const T* rhs_data = rhs.data_ptr<T>();

    // The parallel vectorized loop is tier-compiled; the matmul shortcut and
    // the batch plumbing above stay in this base-tier TU.
    cdist_stub(DeviceType::CPU, lhs_data, rhs_data, output.data_ptr(), batches,
               rows1, rows2, width, p, pdist_mode_code(p),
               static_cast<int>(work_dtype));
    return output;
}

}  // namespace

Tensor pairwise_distance_cpu(const Tensor& x1, const Tensor& x2, double p, double eps,
                             bool keepdim) {
    Tensor diff = x1 - x2 + eps;
    if (diff.dim() == 0) {
        TP_THROW(RuntimeError, "pairwise_distance: inputs must be at least 1-dimensional");
    }
    const int64_t dim = diff.dim() - 1;
    return detail::redispatch_norm_dim_function(
        diff, std::vector<int64_t>{dim}, p, keepdim);
}

Tensor pdist_cpu(const Tensor& self, double p) {
    require_float(self, "pdist");
    if (self.dim() != 2) {
        TP_THROW(RuntimeError, "pdist only supports 2D tensors, got: ",
                 self.dim(), "D");
    }
    if (p < 0.0) {
        TP_THROW(RuntimeError, "pdist only supports non-negative p values");
    }
    if (self.dtype() == DType::Float64) {
        return pdist_impl<double>(self, p);
    }
    return pdist_impl<float>(self, p);
}

Tensor cdist_cpu(const Tensor& x1, const Tensor& x2, double p,
                 std::optional<int64_t> compute_mode) {
    if (x1.dim() < 2) {
        TP_THROW(RuntimeError,
                 "cdist only supports at least 2D tensors, X1 got: ",
                 x1.dim(), "D");
    }
    if (x2.dim() < 2) {
        TP_THROW(RuntimeError,
                 "cdist only supports at least 2D tensors, X2 got: ",
                 x2.dim(), "D");
    }
    if (x1.size(-1) != x2.size(-1)) {
        TP_THROW(RuntimeError,
                 "X1 and X2 must have the same number of columns. X1: ",
                 x1.size(-1), " X2: ", x2.size(-1));
    }
    const DType common = promoteTypes(x1.dtype(), x2.dtype());
    if (!isFloatingType(common)) {
        TP_THROW(TypeError, "cdist only supports floating-point dtypes");
    }
    if (p < 0.0) {
        TP_THROW(RuntimeError, "cdist only supports non-negative p values");
    }
    const int64_t mode = compute_mode.value_or(0);
    if (mode < 0 || mode > 2) {
        TP_THROW(RuntimeError, "possible modes: 0, 1, 2, but was: ", mode);
    }

    const std::vector<int64_t> shape1 =
        static_cast<std::vector<int64_t>>(x1.shape());
    const std::vector<int64_t> shape2 =
        static_cast<std::vector<int64_t>>(x2.shape());
    const std::vector<int64_t> batch1(shape1.begin(), shape1.end() - 2);
    const std::vector<int64_t> batch2(shape2.begin(), shape2.end() - 2);
    const std::vector<int64_t> batch_shape = broadcast_shapes(batch1, batch2);
    const int64_t rows1 = x1.size(-2);
    const int64_t rows2 = x2.size(-2);
    const int64_t width = x1.size(-1);
    if (common == DType::Float64) {
        return cdist_impl<double>(
            x1, x2, p, compute_mode, batch_shape, rows1, rows2, width);
    }
    return cdist_impl<float>(
        x1, x2, p, compute_mode, batch_shape, rows1, rows2, width);
}

DEFINE_DISPATCH(pdist_stub);
DEFINE_DISPATCH(cdist_stub);

TENSORPLAY_LIBRARY_IMPL(CPU, Distance) {
    m.impl("cdist", cdist_cpu);
    m.impl("pairwise_distance", pairwise_distance_cpu);
    m.impl("pdist", pdist_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
