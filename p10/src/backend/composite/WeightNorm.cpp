// Backend-neutral weight normalization.
//
// A weight-normalized parameter is stored as a direction v and a magnitude g,
// one magnitude per slice along `dim`:
//
//     norm = ||v||_2 reduced over every axis except `dim`
//     w    = v * g / norm
//
// Both the forward and the two backward spellings are fixed compositions of
// reductions and elementwise arithmetic, each of which resolves to the native
// kernel of whichever device holds the parameters, so one definition serves
// every backend at full speed.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

int64_t wrap_slice_dim(int64_t dim, int64_t rank) {
    TP_CHECK(rank > 0, "weight_norm: expected a tensor with at least one dimension");
    const int64_t wrapped = dim % rank + (dim % rank < 0 ? rank : 0);
    return wrapped;
}

// Every axis but the slice axis, i.e. the axes the slice norm reduces over.
std::vector<int64_t> axes_except(int64_t dim, int64_t rank) {
    std::vector<int64_t> axes;
    axes.reserve(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        if (d != dim) axes.push_back(d);
    }
    return axes;
}

// A per-slice value (one entry per slice along `dim`, whatever extra unit axes
// it carries) reshaped so it broadcasts against `like`.
Tensor as_slice_broadcast(const Tensor& value, const Tensor& like, int64_t dim) {
    const int64_t rank = like.dim();
    std::vector<int64_t> sizes(static_cast<size_t>(rank), 1);
    sizes[static_cast<size_t>(dim)] = value.numel();
    return ops::reshape(value, sizes);
}

}  // namespace

// The 2-norm of every slice along `dim`, kept in the slice's own position so
// the result broadcasts back over the parameter.  dim == -1 asks for the norm
// of the whole tensor rather than a per-slice norm.
Tensor norm_except_dim_native(const Tensor& v, int64_t pow, int64_t dim) {
    if (dim == -1) {
        return ops::norm(v, static_cast<double>(pow));
    }
    const int64_t rank = v.dim();
    const int64_t axis = wrap_slice_dim(dim, rank);
    return ops::norm(v, axes_except(axis, rank), static_cast<double>(pow), true);
}

std::tuple<Tensor, Tensor> _weight_norm_interface_native(const Tensor& v,
                                                         const Tensor& g,
                                                         int64_t dim) {
    TP_CHECK(v.device() == g.device(),
             "weight_norm: expected the direction and the magnitude to share a "
             "device, got ", v.device().toString(), " and ",
             g.device().toString());
    const int64_t axis = wrap_slice_dim(dim, v.dim());
    const Tensor norm =
        ops::norm(v, axes_except(axis, v.dim()), 2.0, true);
    const Tensor scale = ops::div(as_slice_broadcast(g, v, axis), norm);
    return {ops::mul(v, scale), norm};
}

// With vhat = v / norm, the differential of w = g * vhat is
//     dw/dv = (g / norm) (I - vhat vhat^T)   and   dw/dg = vhat,
// per slice; the adjoint therefore projects the incoming gradient onto vhat
// and reduces every axis except the slice axis.
std::tuple<Tensor, Tensor> _weight_norm_interface_backward_native(
        const Tensor& grad_w, const Tensor& saved_v, const Tensor& saved_g,
        const Tensor& saved_norms, int64_t dim) {
    const int64_t rank = saved_v.dim();
    const int64_t axis = wrap_slice_dim(dim, rank);
    const std::vector<int64_t> reduced = axes_except(axis, rank);

    const Tensor norm = as_slice_broadcast(saved_norms, saved_v, axis);
    const Tensor g = as_slice_broadcast(saved_g, saved_v, axis);
    const Tensor vhat = ops::div(saved_v, norm);
    const Tensor projection = ops::sum(ops::mul(grad_w, vhat), reduced, true);

    Tensor grad_v = ops::mul(ops::div(g, norm),
                             ops::sub(grad_w, ops::mul(vhat, projection)));
    Tensor grad_g = ops::reshape(
        projection, static_cast<std::vector<int64_t>>(saved_g.shape()));
    return {grad_v, grad_g};
}

Tensor _weight_norm_native(const Tensor& v_in, const Tensor& g_in, int64_t dim) {
    TP_CHECK(v_in.device() == g_in.device(),
             "weight_norm: expected the direction and the magnitude to share a "
             "device, got ", v_in.device().toString(), " and ",
             g_in.device().toString());
    const Tensor v = ops::contiguous(v_in, kContiguous);
    const Tensor g = ops::contiguous(g_in, kContiguous);
    return std::get<0>(_weight_norm_interface_native(v, g, dim));
}

// Same gradients as the fused backward, spelled entirely in differentiable
// primitives so a caller that is itself building a graph can differentiate
// through the backward pass.
std::tuple<Tensor, Tensor> _weight_norm_differentiable_backward_native(
        const Tensor& grad_w, const Tensor& saved_v, const Tensor& saved_g,
        const Tensor& saved_norms, int64_t dim) {
    const int64_t rank = saved_v.dim();
    const int64_t axis = wrap_slice_dim(dim, rank);
    const std::vector<int64_t> reduced = axes_except(axis, rank);

    const Tensor norms =
        as_slice_broadcast(ops::to(saved_norms, saved_g.dtype()), saved_v, axis);
    const Tensor g = as_slice_broadcast(saved_g, saved_v, axis);
    const Tensor per_slice_sums =
        ops::sum(ops::mul(grad_w, saved_v), reduced, true);

    Tensor grad_v = ops::mul(
        ops::div(g, norms),
        ops::sub(grad_w, ops::mul(saved_v,
                                  ops::div(per_slice_sums,
                                           ops::mul(norms, norms)))));
    Tensor grad_g = ops::reshape(
        ops::div(per_slice_sums, norms),
        static_cast<std::vector<int64_t>>(saved_g.shape()));
    return {grad_v, grad_g};
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, WeightNormComposite) {
    m.impl("norm_except_dim", composite::norm_except_dim_native);
    m.impl("_weight_norm", composite::_weight_norm_native);
    m.impl("_weight_norm_interface", composite::_weight_norm_interface_native);
    m.impl("_weight_norm_interface_backward",
           composite::_weight_norm_interface_backward_native);
    m.impl("_weight_norm_differentiable_backward",
           composite::_weight_norm_differentiable_backward_native);
}

}  // namespace tensorplay
