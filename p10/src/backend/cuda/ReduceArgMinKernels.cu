#include "ReduceEngine.cuh"

#include <optional>

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

namespace {

template <typename T>
Tensor argmin_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim) {
    static_assert(!std::is_same_v<T, bool>, "argmin is not implemented for bool");
    using ValueT = same_dtype_acc_t<T>;
    // Same packed-u64 fast path as argmax, with the key inverted for the
    // min direction (identical winners — value asc / first occurrence).
    if constexpr (std::is_same_v<ValueT, float>) {
        if (spec.reduced_numel <= ((int64_t{1} << 31) - 1)) {
            return run_reduction_typed<T, unsigned long long, int64_t>(
                input, spec, keepdim, DType::Int64,
                reduction::PackedArgMinOps{}, ~0ull);
        }
    }
    using StateT = ArgPair<ValueT>;
    using Ops = ArgOps<ValueT, false>;
    return run_reduction_typed<T, StateT, int64_t>(
        input, spec, keepdim, DType::Int64, Ops{},
        StateT{reduction::reduction_upper_bound<ValueT>(), 0});
}

} // namespace

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        TP_CHECK_INDEX(self.numel() != 0,
                       "argmin(): Expected reduction dim to be specified for input.numel() == 0.");
    } else {
        const int64_t d = *dim < 0 ? *dim + self.dim() : *dim;
        TP_CHECK_INDEX(self.size(d) != 0,
                       "argmin(): Expected reduction dim ", d, " to have non-zero size.");
    }
    Tensor input = self;
    if (!dim.has_value() && !input.is_contiguous()) input = input.contiguous();
    const ReductionSpec spec = make_reduction_spec(
        input, dim.has_value() ? std::vector<int64_t>{*dim} : std::vector<int64_t>{});
    TP_DISPATCH_REDUCTION_NO_BOOL(argmin_same_dtype, input.dtype(), input, spec, keepdim);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceArgMinKernels) {
    m.impl("argmin", argmin_kernel);
}

} // namespace cuda
} // namespace tensorplay
