#include "ReduceEngine.cuh"

#include "QuantTensorOperators.h"

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

Tensor amin_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim,
                       bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(min_same_dtype, self.dtype(), self, spec, keepdim);
}

// Min
std::tuple<Tensor, Tensor> min_dim_kernel(const Tensor& self, int64_t dim0, bool keepdim) {
    if (isQuantizedType(self.dtype())) return quantized::extrema_dim(self, dim0, keepdim, false);

    const int64_t nd = self.dim();
    TP_CHECK(nd > 0, "min(): Expected input to have at least one dimension");
    const int64_t dim = dim0 < 0 ? dim0 + nd : dim0;
    TP_CHECK(dim >= 0 && dim < nd,
             "Dimension out of range (expected to be in range of [-", nd, ", ", nd - 1, "], but got ", dim0, ")");
    if (self.size(dim) == 0) {
        TP_THROW(IndexError, "min(): Expected reduction dim ", dim, " to have non-zero size.");
    }
    const ReductionSpec spec = make_reduction_spec(self, {dim});
    return [&]() -> std::tuple<Tensor, Tensor> {
        TP_DISPATCH_REDUCTION(
            min_with_indices_same_dtype, self.dtype(), self, spec, keepdim);
        TP_THROW(NotImplementedError, "min(dim): unsupported dtype");
    }();
}

Tensor min_kernel(const Tensor& self) {
    if (isQuantizedType(self.dtype())) return quantized::extrema(self, false);

    if (self.numel() == 0) {
        TP_THROW(RuntimeError, "min(): Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    auto spec = make_reduction_spec(self, {});
    Tensor values = [&]() -> Tensor {
        TP_DISPATCH_REDUCTION(min_same_dtype, self.dtype(), self, spec, false);
        return Tensor();
    }();
    return values;
}

Tensor amin_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amin()");
    return amin_dim_kernel(self, dim, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMinValuesKernels) {
    m.impl("amin", amin_cuda2);
    m.impl("min", min_kernel);
    m.impl("min.dim", min_dim_kernel);
}

} // namespace cuda
} // namespace tensorplay
