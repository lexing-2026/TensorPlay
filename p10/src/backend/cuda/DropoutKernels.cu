#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// _masked_scale: scale the masked-in entries by `scale`.
// ---------------------------------------------------------------------------

Tensor interop_masked_scale_cuda(const Tensor& self, const Tensor& mask,
                                 double scale) {
    return ops::where(mask, self * Scalar(scale), self);
}


// ---------------------------------------------------------------------------
// _masked_softmax: softmax over entries where mask is true; masked-out
// positions stay zero.  The reduction runs in float32 for reduced widths.
// ---------------------------------------------------------------------------

Tensor interop_masked_softmax_cuda(const Tensor& self, const Tensor& mask,
                                   std::optional<int64_t> dim,
                                   std::optional<int64_t> mask_type) {
    (void)mask_type;
    const int64_t d = dim.has_value() ? *dim : -1;
    Tensor neg_inf =
        ops::full_like(self, Scalar(-std::numeric_limits<double>::infinity()));
    Tensor masked = ops::where(mask, self, neg_inf);
    Tensor out = ops::softmax(masked, d, DType::Undefined);
    return ops::where(mask, out, ops::zeros_like(self));
}


Tensor interop_masked_softmax_backward_cuda(const Tensor& grad_output,
                                            const Tensor& output,
                                            const Tensor& mask,
                                            std::optional<int64_t> dim) {
    const int64_t d = dim.has_value() ? *dim : -1;
    Tensor g = ops::where(mask, grad_output, ops::zeros_like(grad_output));
    Tensor o = ops::where(mask, output, ops::zeros_like(output));
    Tensor dot = ops::sum(ops::mul(g, o), {d}, true);
    return ops::where(mask, ops::mul(o, ops::sub(g, dot)),
                      ops::zeros_like(grad_output));
}


// ---------------------------------------------------------------------------
// _fused_dropout: dropout with its boolean mask output.  tp's dropout RNG
// always draws from the default generator stream, so the generator argument
// is accepted but does not reroute the draws.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> interop__fused_dropout_cuda(
        const Tensor& self, double p, std::optional<Generator> generator) {
    (void)generator;
    return dispatch_cuda<std::tuple<Tensor, Tensor>>("native_dropout", self, p);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, DropoutKernels) {
    m.impl("_masked_scale", interop_masked_scale_cuda);
    m.impl("_masked_softmax", interop_masked_softmax_cuda);
    m.impl("_masked_softmax_backward", interop_masked_softmax_backward_cuda);
    m.impl("_fused_dropout", interop__fused_dropout_cuda);
}

} // namespace cuda
} // namespace tensorplay
