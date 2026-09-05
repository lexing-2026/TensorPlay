// Advanced index selection operators - CPU kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

Tensor take_along_dim_cpu(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
    if (indices.dtype() != DType::Int64) {
        TP_THROW(TypeError, "take_along_dim: expected indices to have dtype Int64");
    }
    if (self.device() != indices.device()) {
        TP_THROW(DeviceMismatchError,
                 "take_along_dim: self and indices must be on the same device");
    }
    if (!dim.has_value()) {
        Tensor flat = self.view({-1});
        Tensor idx = indices.view({-1});
        return flat.gather(0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
    // Broadcast both operands over every axis except d; along d only the
    // index extent matters (that is the gather length).
    std::vector<int64_t> target(nd);
    for (int64_t i = 0; i < nd; ++i) {
        if (i == d) { target[i] = indices.size(i); continue; }
        int64_t a = self.size(i), b = indices.size(i);
        if (a != b && a != 1 && b != 1) {
            TP_THROW(RuntimeError, "take_along_dim: input and indices must match on non-selected dimensions");
        }
        target[i] = std::max(a, b);
    }
    std::vector<int64_t> idx_target = target;
    std::vector<int64_t> self_target = target;
    self_target[d] = self.size(d);
    Tensor idx_b = indices.expand(idx_target).contiguous();
    Tensor self_b = self.expand(self_target).contiguous();
    idx_b = idx_b.remainder(Scalar(self_b.size(d)));
    return self_b.gather(d, idx_b);
}


}  // namespace

TENSORPLAY_LIBRARY_IMPL(CPU, TensorAdvancedIndexingKernels) {
    m.impl("take_along_dim", take_along_dim_cpu);
}

} // namespace cpu
} // namespace tensorplay
