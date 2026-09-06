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
#include "cpu/vec/vec.h"
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

enum class PdistMode { One, Two, Infinity, General };

template <PdistMode mode, typename T>
T pdist_distance(const T* lhs, const T* rhs, int64_t width, double p) {
    using Vec = tensorplay::vec::Vectorized<T>;
    Vec aggregate(static_cast<T>(0));
    const int64_t vector_width = Vec::size();
    int64_t column = 0;
    for (; column + vector_width <= width; column += vector_width) {
        const Vec diff =
            (Vec::loadu(lhs + column) - Vec::loadu(rhs + column)).abs();
        if constexpr (mode == PdistMode::One) {
            aggregate = aggregate + diff;
        } else if constexpr (mode == PdistMode::Two) {
            aggregate = aggregate + diff * diff;
        } else if constexpr (mode == PdistMode::Infinity) {
            aggregate = tensorplay::vec::maximum(aggregate, diff);
        } else {
            aggregate = aggregate + diff.pow(Vec(static_cast<T>(p)));
        }
    }

    T result = mode == PdistMode::Infinity
        ? aggregate.reduce_max()
        : aggregate.reduce_add();
    for (; column < width; ++column) {
        const T diff = static_cast<T>(
            std::abs(lhs[column] - rhs[column]));
        if constexpr (mode == PdistMode::One) {
            result += diff;
        } else if constexpr (mode == PdistMode::Two) {
            result += diff * diff;
        } else if constexpr (mode == PdistMode::Infinity) {
            result = std::max(result, diff);
        } else {
            result += static_cast<T>(std::pow(diff, p));
        }
    }

    if constexpr (mode == PdistMode::Two) {
        return static_cast<T>(std::sqrt(result));
    } else if constexpr (mode == PdistMode::General) {
        return static_cast<T>(std::pow(result, 1.0 / p));
    } else {
        return result;
    }
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
    T* output = out.data_ptr<T>();

    parallel_for(0, outn, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        const double n2 = static_cast<double>(n) - 0.5;
        int64_t i = static_cast<int64_t>(
            n2 - std::sqrt(n2 * n2 - 2.0 * static_cast<double>(begin) - 1.0));
        int64_t j = begin - n * i + i * (i + 1) / 2 + i + 1;
        for (int64_t index = begin; index < end; ++index) {
            const T* lhs = data + i * width;
            const T* rhs = data + j * width;
            if (p == 0.0) {
                int64_t count = 0;
                for (int64_t column = 0; column < width; ++column) {
                    count += lhs[column] != rhs[column];
                }
                output[index] = static_cast<T>(count);
            } else if (p == 1.0) {
                output[index] = pdist_distance<PdistMode::One>(
                    lhs, rhs, width, p);
            } else if (p == 2.0) {
                output[index] = pdist_distance<PdistMode::Two>(
                    lhs, rhs, width, p);
            } else if (std::isinf(p)) {
                output[index] = pdist_distance<PdistMode::Infinity>(
                    lhs, rhs, width, p);
            } else {
                output[index] = pdist_distance<PdistMode::General>(
                    lhs, rhs, width, p);
            }
            ++j;
            if (j == n) {
                ++i;
                j = i + 1;
            }
        }
    });

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
    T* output_data = output.data_ptr<T>();
    const int64_t pair_count = batches * rows1 * rows2;
    const int64_t grain = std::max<int64_t>(1, GRAIN_SIZE / width);
    parallel_for(0, pair_count, grain, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t batch_pair = linear % (rows1 * rows2);
            const int64_t batch = linear / (rows1 * rows2);
            const int64_t row1 = batch_pair / rows2;
            const int64_t row2 = batch_pair % rows2;
            const T* lhs_row = lhs_data + (batch * rows1 + row1) * width;
            const T* rhs_row = rhs_data + (batch * rows2 + row2) * width;
            T value;
            if (p == 0.0) {
                int64_t count = 0;
                for (int64_t column = 0; column < width; ++column) {
                    count += lhs_row[column] != rhs_row[column];
                }
                value = static_cast<T>(count);
            } else if (p == 1.0) {
                value = pdist_distance<PdistMode::One>(
                    lhs_row, rhs_row, width, p);
            } else if (p == 2.0) {
                value = pdist_distance<PdistMode::Two>(
                    lhs_row, rhs_row, width, p);
            } else if (std::isinf(p)) {
                value = pdist_distance<PdistMode::Infinity>(
                    lhs_row, rhs_row, width, p);
            } else {
                value = pdist_distance<PdistMode::General>(
                    lhs_row, rhs_row, width, p);
            }
            output_data[linear] = value;
        }
    });
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

TENSORPLAY_LIBRARY_IMPL(CPU, Distance) {
    m.impl("cdist", cdist_cpu);
    m.impl("pairwise_distance", pairwise_distance_cpu);
    m.impl("pdist", pdist_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
