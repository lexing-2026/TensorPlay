// Advanced index selection operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

Tensor gather_cuda(const Tensor& self, int64_t dim, const Tensor& index);

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

Tensor take_along_dim_cuda(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
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
        return gather_cuda(flat, 0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
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
    return gather_cuda(self_b, d, idx_b);
}


} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, TensorAdvancedIndexingKernels) {
    m.impl("take_along_dim", take_along_dim_cuda);
}

} // namespace cuda
} // namespace tensorplay
