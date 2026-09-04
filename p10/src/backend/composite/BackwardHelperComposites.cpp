// Backend-neutral kernels for the backward-helper operators that the
// generated autograd nodes dispatch through.  Each kernel composes
// dispatched primitives, so a single Composite-key registration serves
// every backend: the inner ops resolve to the caller's device kernels.

#include "CompositeCommon.h"
#include "ManualNodes.h"

#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// d(tr(A))/dA = I on the two trailing axes.  A batched trace reduces only
// those axes, so the incoming gradient is reshaped to the batch shape with
// 1x1 trailing axes and the identity broadcasts over it.
Tensor trace_backward_impl(const Tensor& grad,
                           const std::vector<int64_t>& sizes) {
    TP_CHECK(sizes.size() >= 2,
             "trace_backward expects an input with at least 2 dimensions");
    const int64_t m = sizes[sizes.size() - 2];
    const int64_t n = sizes[sizes.size() - 1];
    std::vector<int64_t> grad_shape(sizes.begin(), sizes.end() - 2);
    grad_shape.push_back(1);
    grad_shape.push_back(1);
    const Tensor g = ops::reshape(grad, grad_shape);
    return ops::mul(ops::eye(m, n, grad.dtype(), grad.device()), g);
}

// masked_select keeps the true positions of the mask in order; scattering
// the incoming gradient into a zero buffer of the input's shape with the
// same mask inverts it exactly.
Tensor masked_select_backward_impl(const Tensor& grad, const Tensor& input,
                                   const Tensor& mask) {
    return ops::masked_scatter(ops::zeros_like(input), mask, grad);
}

// Route every gradient entry back to its winning position through
// scatter_add, so ties accumulate instead of overwrite.
Tensor cummaxmin_backward_impl(const Tensor& grad, const Tensor& input,
                               const Tensor& indices, int64_t dim) {
    return tensorplay::tpx::cummaxmin_backward(grad, input, indices, dim);
}

Tensor cumprod_backward_impl(const Tensor& grad, const Tensor& input,
                             int64_t dim, const Tensor& output) {
    return tensorplay::tpx::cumprod_backward(grad, input, dim, output);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(Composite, BackwardHelperComposites) {
    m.impl("trace_backward", trace_backward_impl);
    m.impl("masked_select_backward", masked_select_backward_impl);
    m.impl("cummaxmin_backward", cummaxmin_backward_impl);
    m.impl("cumprod_backward", cumprod_backward_impl);
}

} // namespace composite
} // namespace tensorplay
