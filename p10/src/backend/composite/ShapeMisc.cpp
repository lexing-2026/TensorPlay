// Backend-neutral shape, view and type-bridging composites.
//
// Each entry here is a fixed rewrite onto primitives that already carry a
// native kernel per device, so the work runs entirely through those kernels
// on whichever device holds the input:
//
//   _stack / _stack.out   stacking with the same contract as stack;
//   nonzero_numpy         the nonzero coordinates split one tensor per axis,
//                         with a rank-0 input treated as a length-1 vector so
//                         a nonzero scalar reports index 0;
//   type_as               dtype (and device) adoption from another tensor;
//   _unsafe_view          a reshape whose result is not tracked as a view;
//   _reshape_copy         reshape that always owns its result;
//   empty_permuted        an allocation whose physical element order follows
//                         `physical_layout` while the logical shape stays
//                         `size` -- the strides are permuted by the inverse
//                         of the layout, which is why this is not an empty
//                         followed by a permute;
//   _safe_softmax         softmax that answers 0 instead of NaN on rows that
//                         are masked out entirely (every entry -inf);
//   _logcumsumexp(+out)   the internal spelling of the log-domain cumulative
//                         sum;
//   _pdist_forward,       the internal spellings of the pairwise and
//   _cdist_forward        cross distance forwards;
//   _euclidean_dist       the squared-norm expansion
//                         ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 folded into
//                         one matmul, clamped at zero before the root so
//                         cancellation cannot produce a negative radicand;
//   polygamma_, igamma_,  the in-place spellings of their functional forms;
//   igammac_
//   _add_relu             the fused add-then-rectify family; alpha == 1 rides
//                         the fused primitive, any other scale folds the
//                         weighted sum first;
//   _efficientzerotensor  a zero-filled allocation;
//   _pin_memory           the function spelling of the pinning method;
//   _resize_output_       resize a destination in place;
//   flatten_dense_tensors the concatenation of a parameter list flattened into
//   unflatten_dense_tensors  one vector, and its inverse split;
//   nll_loss_forward,     the (loss, total_weight) pair the backward consumes;
//   nll_loss2d_forward
//   _lu_with_info         the LU factorization with its per-matrix status.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor _stack_native(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::stack(tensors, dim);
}

// The out= form keeps the caller's buffer: the destination is only resized
// when the result does not already fit, and the values are written into the
// storage the caller handed over.
Tensor& _stack_out_native(const std::vector<Tensor>& tensors, int64_t dim,
                          Tensor& out) {
    const Tensor value = ops::stack(tensors, dim);
    if (!out.defined()) {
        out = value;
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, "_stack: output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 "_stack: output device must match input device");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

std::vector<Tensor> nonzero_numpy_native(const Tensor& self) {
    if (self.dim() == 0) {
        return ops::unbind(ops::nonzero(ops::unsqueeze(self, 0)), 1);
    }
    return ops::unbind(ops::nonzero(self), 1);
}

Tensor type_as_native(const Tensor& self, const Tensor& other) {
    return ops::to(self, other, false, false, std::nullopt);
}

Tensor _unsafe_view_native(const Tensor& self, const std::vector<int64_t>& size) {
    return ops::view(self, size);
}

Tensor _reshape_copy_native(const Tensor& self, const std::vector<int64_t>& size) {
    TP_CHECK(!self.unsafeGetTensorImpl()->is_sparse(),
             "_reshape_copy is not implemented for sparse tensors");
    return ops::clone(ops::reshape(self, size), kContiguous);
}

Tensor empty_permuted_native(const std::vector<int64_t>& size,
                             const std::vector<int64_t>& physical_layout,
                             std::optional<DType> dtype,
                             std::optional<int64_t> /*layout*/,
                             std::optional<Device> device,
                             std::optional<bool> pin_memory) {
    const int64_t rank = static_cast<int64_t>(size.size());
    TP_CHECK(static_cast<int64_t>(physical_layout.size()) == rank,
             "empty_permuted: size has ", rank,
             " dimensions but physical_layout has ", physical_layout.size());

    std::vector<bool> seen(static_cast<size_t>(rank), false);
    std::vector<int64_t> physical_size(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t logical = physical_layout[static_cast<size_t>(i)];
        TP_CHECK(logical >= 0 && logical < rank,
                 "empty_permuted: dimension out of range (expected to be "
                 "between 0 and ", rank - 1, ", but got ", logical,
                 " at index ", i, ")");
        TP_CHECK(!seen[static_cast<size_t>(logical)],
                 "empty_permuted: duplicate dimension ", logical,
                 " in physical_layout");
        seen[static_cast<size_t>(logical)] = true;
        physical_size[static_cast<size_t>(i)] = size[static_cast<size_t>(logical)];
    }

    Tensor physical = ops::empty(physical_size, dtype, device,
                                 pin_memory.value_or(false), false);
    // Inverse permutation: the stride of logical axis physical_layout[i] is
    // the stride the contiguous allocation gave to physical axis i.
    std::vector<int64_t> strides(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        strides[static_cast<size_t>(physical_layout[static_cast<size_t>(i)])] =
            physical.stride(i);
    }
    return ops::as_strided(physical, size, strides, std::nullopt);
}

// A row whose every logit is -inf carries no probability mass; the ordinary
// softmax would divide zero by zero there, so those rows are answered with
// zeros instead.
Tensor _safe_softmax_native(const Tensor& self, int64_t dim,
                            std::optional<DType> dtype) {
    Tensor out = ops::softmax(self, dim, dtype);
    const Tensor masked_rows =
        ops::all(ops::isneginf(self), dim, /*keepdim=*/true);
    return ops::where(masked_rows, Scalar(0.0), out);
}

Tensor _logcumsumexp_native(const Tensor& self, int64_t dim) {
    return ops::logcumsumexp(self, dim, std::nullopt);
}

Tensor& _logcumsumexp_out_native(const Tensor& self, int64_t dim, Tensor& out) {
    if (out.defined()) {
        if (out.dtype() != self.dtype()) {
            TP_THROW(RuntimeError, "logcumsumexp: output dtype must match input dtype");
        }
        if (out.device() != self.device()) {
            TP_THROW(RuntimeError, "logcumsumexp: output device must match input device");
        }
    }
    const Tensor value = ops::logcumsumexp(self, dim, std::nullopt);
    if (!out.defined()) {
        out = value;
        return out;
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

Tensor _pdist_forward_native(const Tensor& self, double p) {
    return ops::pdist(self, p);
}

Tensor _cdist_forward_native(const Tensor& x1, const Tensor& x2, double p,
                             std::optional<int64_t> compute_mode) {
    return ops::cdist(x1, x2, p, compute_mode);
}

// One matmul carries all three terms of ||a - b||^2: the rows of the left
// operand hold (-2a, ||a||^2, 1) and those of the right hold (b, 1, ||b||^2),
// so their inner product is exactly the expanded square.  Rounding can push a
// vanishing distance slightly below zero, hence the clamp before the root.
Tensor _euclidean_dist_native(const Tensor& x1, const Tensor& x2) {
    const Tensor x1_norm = ops::sum(ops::mul(x1, x1), {-1}, true);
    const Tensor x2_norm = ops::sum(ops::mul(x2, x2), {-1}, true);
    const Tensor x1_pad = ops::ones_like(x1_norm);
    const Tensor x2_pad = ops::ones_like(x2_norm);
    const Tensor left = ops::cat({ops::mul(x1, Scalar(-2.0)), x1_norm, x1_pad}, -1);
    const Tensor right = ops::cat({x2, x2_pad, x2_norm}, -1);
    const Tensor squared =
        ops::matmul(left, ops::transpose(right, -2, -1));
    return ops::sqrt(ops::clamp(squared, Scalar(0.0), std::nullopt));
}

Tensor& polygamma__native(Tensor& self, int64_t n) {
    self.copy_(ops::polygamma(n, self));
    return self;
}

Tensor& igamma__native(Tensor& self, const Tensor& other) {
    self.copy_(ops::igamma(self, other));
    return self;
}

Tensor& igammac__native(Tensor& self, const Tensor& other) {
    self.copy_(ops::igammac(self, other));
    return self;
}

// A fused add-then-rectify: relu(self + alpha * other).  The unit scale rides
// the fused primitive each backend already carries; any other scale folds the
// weighted sum first and rectifies it.
Tensor _add_relu_native(const Tensor& self, const Tensor& other, Scalar alpha) {
    if (!alpha.isComplex() && alpha.toDouble() == 1.0) {
        return ops::add_relu(self, other);
    }
    return ops::relu(ops::add(self, other, alpha));
}

Tensor& _add_relu__native(Tensor& self, const Tensor& other, Scalar alpha) {
    self.copy_(_add_relu_native(self, other, alpha));
    return self;
}

Tensor& _add_relu_out_native(const Tensor& self, const Tensor& other,
                             Scalar alpha, Tensor& out) {
    const Tensor value = _add_relu_native(self, other, alpha);
    if (!out.defined() || out.dtype() != value.dtype()) {
        out = value;
        return out;
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

Tensor _add_relu_scalar_native(const Tensor& self, Scalar other, Scalar alpha) {
    return ops::relu(ops::add(self, other, alpha));
}

Tensor& _add_relu__scalar_native(Tensor& self, Scalar other, Scalar alpha) {
    self.copy_(_add_relu_scalar_native(self, other, alpha));
    return self;
}

Tensor _efficientzerotensor_native(const std::vector<int64_t>& size,
                                   std::optional<DType> dtype,
                                   std::optional<int64_t> /*layout*/,
                                   std::optional<Device> device,
                                   std::optional<bool> pin_memory) {
    return ops::zeros(size, dtype, device, pin_memory.value_or(false), false);
}

Tensor _pin_memory_native(const Tensor& self, std::optional<Device> device) {
    return ops::pin_memory(self, device);
}

Tensor& _resize_output__native(Tensor& self, const std::vector<int64_t>& size,
                               Device device) {
    TP_CHECK(self.device() == device,
             "_resize_output_: the destination lives on ",
             self.device().toString(), " but ", device.toString(),
             " was requested");
    if (static_cast<std::vector<int64_t>>(self.shape()) != size) {
        self.resize_(size);
    }
    return self;
}

// One contiguous vector holding every element of the list, in order.
Tensor flatten_dense_tensors_native(const std::vector<Tensor>& tensors) {
    TP_CHECK(!tensors.empty(), "flatten_dense_tensors: expected a non-empty list");
    std::vector<Tensor> flattened;
    flattened.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        flattened.push_back(ops::reshape(ops::contiguous(t, kContiguous), {-1}));
    }
    if (flattened.size() == 1) return flattened.front();
    return ops::cat(flattened, 0);
}

// The inverse split: each output is a view of `flat` carrying the shape of the
// tensor at the same position.  An empty member gets its own allocation so it
// does not alias the vector.
std::vector<Tensor> unflatten_dense_tensors_native(
        const Tensor& flat, const std::vector<Tensor>& tensors) {
    std::vector<Tensor> outputs;
    outputs.reserve(tensors.size());
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        const auto shape = static_cast<std::vector<int64_t>>(t.shape());
        const int64_t count = t.numel();
        if (count == 0) {
            outputs.push_back(ops::empty(shape, flat.dtype(), flat.device(),
                                         false, false));
            continue;
        }
        outputs.push_back(ops::reshape(ops::narrow(flat, 0, offset, count), shape));
        offset += count;
    }
    return outputs;
}

std::tuple<Tensor, Tensor> nll_loss_forward_native(
        const Tensor& self, const Tensor& target,
        const std::optional<Tensor>& weight, int64_t reduction,
        int64_t ignore_index) {
    return ops::nll_loss(self, target, weight, reduction, ignore_index);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_native(
        const Tensor& self, const Tensor& target,
        const std::optional<Tensor>& weight, int64_t reduction,
        int64_t ignore_index) {
    return ops::nll_loss2d(self, target, weight, reduction, ignore_index);
}

std::tuple<Tensor, Tensor, Tensor> _lu_with_info_native(const Tensor& self,
                                                        bool pivot,
                                                        bool check_errors) {
    return ops::linalg_lu_factor_ex(self, pivot, check_errors);
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, ShapeMiscComposite) {
    m.impl("_stack", composite::_stack_native);
    m.impl("_stack.out", composite::_stack_out_native);
    m.impl("nonzero_numpy", composite::nonzero_numpy_native);
    m.impl("where", composite::nonzero_numpy_native);
    m.impl("type_as", composite::type_as_native);
    m.impl("_unsafe_view", composite::_unsafe_view_native);
    m.impl("_reshape_copy", composite::_reshape_copy_native);
    m.impl("empty_permuted", composite::empty_permuted_native);
    m.impl("_safe_softmax", composite::_safe_softmax_native);
    m.impl("_logcumsumexp", composite::_logcumsumexp_native);
    m.impl("_logcumsumexp.out", composite::_logcumsumexp_out_native);
    m.impl("_pdist_forward", composite::_pdist_forward_native);
    m.impl("_cdist_forward", composite::_cdist_forward_native);
    m.impl("_euclidean_dist", composite::_euclidean_dist_native);
    m.impl("polygamma_", composite::polygamma__native);
    m.impl("igamma_", composite::igamma__native);
    m.impl("igammac_", composite::igammac__native);
    m.impl("_add_relu.Tensor", composite::_add_relu_native);
    m.impl("_add_relu_.Tensor", composite::_add_relu__native);
    m.impl("_add_relu.out", composite::_add_relu_out_native);
    m.impl("_add_relu.Scalar", composite::_add_relu_scalar_native);
    m.impl("_add_relu_.Scalar", composite::_add_relu__scalar_native);
    m.impl("_efficientzerotensor", composite::_efficientzerotensor_native);
    m.impl("_pin_memory", composite::_pin_memory_native);
    m.impl("_resize_output_", composite::_resize_output__native);
    m.impl("flatten_dense_tensors", composite::flatten_dense_tensors_native);
    m.impl("unflatten_dense_tensors", composite::unflatten_dense_tensors_native);
    m.impl("nll_loss_forward", composite::nll_loss_forward_native);
    m.impl("nll_loss2d_forward", composite::nll_loss2d_forward_native);
    m.impl("_lu_with_info", composite::_lu_with_info_native);
}

}  // namespace tensorplay
