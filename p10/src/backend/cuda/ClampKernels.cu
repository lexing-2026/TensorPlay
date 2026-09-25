// Core operators - CUDA kernels: clamp family: elementwise bounding with optional floor, ceiling and
// tensor-valued bounds.
//
// Pointwise kernels over a grid-stride loop; the shared plumbing comes
// from the pointwise header and the per-operator bodies live here.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {
namespace {

Tensor clamp_min_scalar_cuda(const Tensor& self, const Scalar& min) {
    double lo = min.toDouble();
    return dtype_unary_cuda(self,
                            HFn48{lo},
                            "clamp_min");
}

Tensor clamp_max_scalar_cuda(const Tensor& self, const Scalar& max) {
    double hi = max.toDouble();
    return dtype_unary_cuda(self,
                            HFn49{hi},
                            "clamp_max");
}

Tensor clamp_min_tensor_cuda(const Tensor& self, const Tensor& min) {
    return binary_same_cuda(self, min,
                            HFn50{},
                            "clamp_min");
}

Tensor clamp_max_tensor_cuda(const Tensor& self, const Tensor& max) {
    return binary_same_cuda(self, max,
                            HFn51{},
                            "clamp_max");
}

Tensor clip_cuda(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    if (min.has_value() && max.has_value()) {
        Tensor r = clamp_min_scalar_cuda(self, *min);
        return clamp_max_scalar_cuda(r, *max);
    }
    if (min.has_value()) return clamp_min_scalar_cuda(self, *min);
    if (max.has_value()) return clamp_max_scalar_cuda(self, *max);
    return self.clone();
}

Tensor& clamp__cuda(Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
    Tensor r = clip_cuda(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// Keep them separate from clamp_ (which accepts two optional bounds): these
// are the unsuffixed dispatcher names used by the generated native schemas.

Tensor& clamp_min__scalar_cuda(Tensor& self, const Scalar& min) {
    self.copy_(clamp_min_scalar_cuda(self, min));
    return self;
}

Tensor& clamp_max__scalar_cuda(Tensor& self, const Scalar& max) {
    self.copy_(clamp_max_scalar_cuda(self, max));
    return self;
}

// ===========================================================================
// Activations
// ===========================================================================

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, ClampKernels) {
    m.impl("clamp_", clamp__cuda);
    m.impl("clamp_min", clamp_min_scalar_cuda);
    m.impl("clamp_max", clamp_max_scalar_cuda);
    m.impl("clamp_min_", clamp_min__scalar_cuda);
    m.impl("clamp_max_", clamp_max__scalar_cuda);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cuda);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cuda);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cuda);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cuda);
    m.impl("clip", clip_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
