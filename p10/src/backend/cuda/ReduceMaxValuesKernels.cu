#include "ReduceEngine.cuh"

#include "QuantTensorOperators.h"

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

Tensor amax_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(max_same_dtype, self.dtype(), self, spec, keepdim);
}

// Max
std::tuple<Tensor, Tensor> max_dim_kernel(const Tensor& self, int64_t dim0, bool keepdim) {
    if (isQuantizedType(self.dtype())) return quantized::extrema_dim(self, dim0, keepdim, true);

    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "max(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "max(): Expected reduction dim ", dim, " to have non-zero size.");
    }
    const ReductionSpec spec = make_reduction_spec(self, {dim});
    // The dispatch macros expand to a returning switch; route them through an
    // immediately-invoked lambda to capture both outputs.
    return [&]() -> std::tuple<Tensor, Tensor> {
        TP_DISPATCH_REDUCTION(
            max_with_indices_same_dtype, self.dtype(), self, spec, keepdim);
        TP_THROW(NotImplementedError, "max(dim): unsupported dtype");
    }();
}

Tensor max_kernel(const Tensor& self) {
    if (isQuantizedType(self.dtype())) return quantized::extrema(self, true);

    if (self.numel() == 0) {
        TP_THROW(RuntimeError, "max(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    auto spec = make_reduction_spec(self, {});
    Tensor values = [&]() -> Tensor {
        TP_DISPATCH_REDUCTION(max_same_dtype, self.dtype(), self, spec, false);
        return Tensor();
    }();
    return values;
}

Tensor amax_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amax()");
    return amax_dim_kernel(self, dim, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMaxValuesKernels) {
    m.impl("amax", amax_cuda2);
    m.impl("max", max_kernel);
    m.impl("max.dim", max_dim_kernel);
}

} // namespace cuda
} // namespace tensorplay
