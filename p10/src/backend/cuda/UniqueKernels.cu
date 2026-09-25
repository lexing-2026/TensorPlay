#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// unique_consecutive: run-length encoding of adjacent equal values.  With a
// dimension given, the scan runs along that dimension; without one, the
// input is flattened first (so inverse indices index the flat input).
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor> interop_unique_consecutive_cuda(
        const Tensor& self, bool return_inverse, bool return_counts,
        std::optional<int64_t> dim) {
    Tensor flat = dim.has_value() ? self : self.reshape({-1});
    auto result = dispatch_cuda<std::tuple<Tensor, Tensor, Tensor>>(
        "unique_dim_consecutive", flat, dim.has_value() ? *dim : int64_t(0),
        return_inverse, return_counts);
    if (!dim.has_value() && return_inverse) {
        std::get<1>(result) = std::get<1>(result).reshape(
            static_cast<std::vector<int64_t>>(self.shape()));
    }
    return result;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, UniqueKernels) {
    // uniqueness / static nonzero
    m.impl("unique_consecutive", interop_unique_consecutive_cuda);
}

} // namespace cuda
} // namespace tensorplay
