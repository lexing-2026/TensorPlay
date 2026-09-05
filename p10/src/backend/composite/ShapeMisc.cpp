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
//                         are masked out entirely (every entry -inf).

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

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, ShapeMiscComposite) {
    m.impl("_stack", composite::_stack_native);
    m.impl("_stack.out", composite::_stack_out_native);
    m.impl("nonzero_numpy", composite::nonzero_numpy_native);
    m.impl("type_as", composite::type_as_native);
    m.impl("_unsafe_view", composite::_unsafe_view_native);
    m.impl("_reshape_copy", composite::_reshape_copy_native);
    m.impl("empty_permuted", composite::empty_permuted_native);
    m.impl("_safe_softmax", composite::_safe_softmax_native);
}

}  // namespace tensorplay
