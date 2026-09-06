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
#include "cpu/vec/vec.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

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

TENSORPLAY_LIBRARY_IMPL(CPU, Distance) {
    m.impl("pairwise_distance", pairwise_distance_cpu);
    m.impl("pdist", pdist_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
