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
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <vector>
#include <cmath>
#include <limits>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
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
    int64_t n = self.size(0), D = self.size(1);
    int64_t outn = n * (n - 1) / 2;
    Tensor a = self.contiguous().to(DType::Float64);
    const double* ap = a.data_ptr<double>();
    Tensor out = Tensor::empty({std::max<int64_t>(outn, 1)}, DType::Float64, self.device());
    double* op = out.data_ptr<double>();
    parallel_for(0, std::max<int64_t>(outn, 1), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t li = begin; li < end; ++li) {
            // condensed index -> (i, j) pair with i < j
            int64_t i = static_cast<int64_t>(
                n - 2 - std::floor(std::sqrt(-8.0 * li + 4.0 * n * (n - 1) - 7) / 2.0 - 0.5));
            int64_t j = li + i + 1 - (n * (n - 1)) / 2 +
                        ((n - i) * (n - i - 1)) / 2;
            double d2 = 0;
            if (p == std::numeric_limits<double>::infinity()) {
                for (int64_t c = 0; c < D; ++c)
                    d2 = std::max(d2, std::fabs(ap[i * D + c] - ap[j * D + c]));
            } else if (p == 0.0) {
                int64_t cnt = 0;
                for (int64_t c = 0; c < D; ++c)
                    if (ap[i * D + c] != ap[j * D + c]) ++cnt;
                d2 = static_cast<double>(cnt);
            } else if (p == 2.0) {
                for (int64_t c = 0; c < D; ++c) {
                    double diff = ap[i * D + c] - ap[j * D + c];
                    d2 += diff * diff;
                }
                d2 = std::sqrt(d2);
            } else if (p == 1.0) {
                for (int64_t c = 0; c < D; ++c)
                    d2 += std::fabs(ap[i * D + c] - ap[j * D + c]);
            } else {
                for (int64_t c = 0; c < D; ++c)
                    d2 += std::pow(std::fabs(ap[i * D + c] - ap[j * D + c]), p);
                d2 = std::pow(d2, 1.0 / p);
            }
            op[li] = d2;
        }
    });
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor res = out.to(out_dt);
    return outn == 0 ? res.reshape({0}) : res;
}

TENSORPLAY_LIBRARY_IMPL(CPU, Distance) {
    m.impl("pairwise_distance", pairwise_distance_cpu);
    m.impl("pdist", pdist_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
