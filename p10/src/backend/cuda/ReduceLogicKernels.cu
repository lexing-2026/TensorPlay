#include "ReduceEngine.cuh"

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

namespace {

template <typename T>
Tensor all_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        return Tensor::ones(reduction_output_shape(input, spec, keepdim), DType::Bool, input.device());
    }
    return run_reduction_typed<T, bool, bool>(
        input, spec, keepdim, DType::Bool, AllOps<T, bool, bool>{}, true);
}

template <typename T>
Tensor any_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    if (input.numel() == 0) {
        return Tensor::zeros(reduction_output_shape(input, spec, keepdim), DType::Bool, input.device());
    }
    return run_reduction_typed<T, bool, bool>(
        input, spec, keepdim, DType::Bool, AnyOps<T, bool, bool>{}, false);
}

} // namespace

// All / Any
Tensor all_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(all_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor all_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return all_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

Tensor all_kernel(const Tensor& self) {
    return all_dim_kernel(self, {}, false);
}

Tensor any_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_REDUCTION(any_same_dtype, self.dtype(), self, spec, keepdim);
}

Tensor any_dim_int_kernel(const Tensor& self, int64_t dim, bool keepdim) {
    return any_dim_kernel(self, std::vector<int64_t>{dim}, keepdim);
}

Tensor any_kernel(const Tensor& self) {
    return any_dim_kernel(self, {}, false);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceLogicKernels) {
    m.impl("all", all_kernel);
    m.impl("all.dim", all_dim_int_kernel);
    m.impl("any", any_kernel);
    m.impl("any.dim", any_dim_int_kernel);
}

} // namespace cuda
} // namespace tensorplay
