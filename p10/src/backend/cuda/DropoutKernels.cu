#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// _fused_dropout: dropout given the probability of keeping an entry, with
// the mask returned as bytes.  tp's dropout RNG always draws from the
// default generator stream, so the generator argument is accepted but does
// not reroute the draws.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> interop__fused_dropout_cuda(
        const Tensor& self, double p, std::optional<Generator> generator) {
    (void)generator;
    auto [output, mask] = ops::native_dropout(self, 1.0 - p, true);
    return {output, mask.to(DType::UInt8)};
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, DropoutKernels) {
    m.impl("_fused_dropout", interop__fused_dropout_cuda);
}

} // namespace cuda
} // namespace tensorplay
