// Composite kernels: the *_copy view family -- alias_copy, t_copy,
// permute_copy, transpose_copy, squeeze_copy (all variants), unsqueeze_copy,
// select_copy, slice_copy, narrow_copy, diagonal_copy, unbind_copy,
// split_copy (both variants), view_copy (both variants), unfold_copy,
// expand_copy.
//
// Each kernel applies the corresponding view op to the input and materializes
// the result with clone(MemoryFormat::Contiguous): functional view semantics
// whose results never alias the input.  view_copy additionally falls back to
// reshape for inputs a plain view cannot express (non-viewable strides),
// reusing the reshape kernel's leniency instead of failing the copy.

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

Tensor alias_copy_native(const Tensor& self) {
    return ops::clone(self, kContiguous);
}

Tensor t_copy_native(const Tensor& self) {
    return ops::clone(ops::t(self), kContiguous);
}

Tensor permute_copy_native(const Tensor& self,
                           const std::vector<int64_t>& dims) {
    return ops::clone(ops::permute(self, dims), kContiguous);
}

Tensor transpose_copy_native(const Tensor& self, int64_t dim0, int64_t dim1) {
    return ops::clone(ops::transpose(self, dim0, dim1), kContiguous);
}

Tensor squeeze_copy_native(const Tensor& self) {
    return ops::clone(ops::squeeze(self), kContiguous);
}

Tensor squeeze_copy_dim_native(const Tensor& self, int64_t dim) {
    return ops::clone(ops::squeeze(self, dim), kContiguous);
}

Tensor squeeze_copy_dims_native(const Tensor& self,
                                const std::vector<int64_t>& dim) {
    return ops::clone(ops::squeeze(self, dim), kContiguous);
}

Tensor unsqueeze_copy_native(const Tensor& self, int64_t dim) {
    return ops::clone(ops::unsqueeze(self, dim), kContiguous);
}

Tensor select_copy_native(const Tensor& self, int64_t dim, int64_t index) {
    return ops::clone(ops::select(self, dim, index), kContiguous);
}

Tensor slice_copy_native(const Tensor& self, int64_t dim,
                         std::optional<int64_t> start,
                         std::optional<int64_t> end, int64_t step) {
    return ops::clone(ops::slice(self, dim, start, end, step), kContiguous);
}

Tensor narrow_copy_native(const Tensor& self, int64_t dim, int64_t start,
                          int64_t length) {
    return ops::clone(ops::narrow(self, dim, start, length), kContiguous);
}

Tensor diagonal_copy_native(const Tensor& self, int64_t offset, int64_t dim1,
                            int64_t dim2) {
    return ops::clone(ops::diagonal(self, offset, dim1, dim2), kContiguous);
}

std::vector<Tensor> unbind_copy_native(const Tensor& self, int64_t dim) {
    std::vector<Tensor> pieces = ops::unbind(self, dim);
    for (Tensor& piece : pieces) piece = ops::clone(piece, kContiguous);
    return pieces;
}

std::vector<Tensor> split_copy_native(const Tensor& self, int64_t split_size,
                                      int64_t dim) {
    std::vector<Tensor> pieces = ops::split(self, split_size, dim);
    for (Tensor& piece : pieces) piece = ops::clone(piece, kContiguous);
    return pieces;
}

std::vector<Tensor> split_copy_sizes_native(
        const Tensor& self, const std::vector<int64_t>& split_size,
        int64_t dim) {
    std::vector<Tensor> pieces = ops::split(self, split_size, dim);
    for (Tensor& piece : pieces) piece = ops::clone(piece, kContiguous);
    return pieces;
}

Tensor view_copy_native(const Tensor& self,
                        const std::vector<int64_t>& size) {
    try {
        return ops::clone(ops::view(self, size), kContiguous);
    } catch (const Exception&) {
        return ops::reshape(self, size);
    }
}

Tensor view_copy_dtype_native(const Tensor& self, DType dtype) {
    return ops::clone(self.view_dtype(dtype), kContiguous);
}

Tensor unfold_copy_native(const Tensor& self, int64_t dimension, int64_t size,
                          int64_t step) {
    return ops::clone(ops::unfold(self, dimension, size, step), kContiguous);
}

Tensor expand_copy_native(const Tensor& self,
                          const std::vector<int64_t>& size, bool implicit) {
    return ops::clone(ops::expand(self, size, implicit), kContiguous);
}

std::vector<Tensor> split_with_sizes_copy_native(
        const Tensor& self, const std::vector<int64_t>& split_sizes,
        int64_t dim) {
    std::vector<Tensor> pieces = ops::split(self, split_sizes, dim);
    for (Tensor& piece : pieces) piece = ops::clone(piece, kContiguous);
    return pieces;
}

std::vector<Tensor> unsafe_split_with_sizes_native(
        const Tensor& self, const std::vector<int64_t>& split_sizes,
        int64_t dim) {
    return ops::split(self, split_sizes, dim);
}

// detach is data-identity here (no lazy graph to cut), so the copy variant
// is a plain materialization and the in-place variant returns self.
Tensor detach_copy_native(const Tensor& self) {
    return ops::clone(self, kContiguous);
}

Tensor& detach__native(Tensor& self) {
    return self;
}

// -----------------------------------------------------------------------------
// Identity / transposed view family
// -----------------------------------------------------------------------------
//
// Inside an active transform layer a wrapper's payload lives in the transform
// value while its own storage stays empty, so plain backend kernels must never
// run on it.  The generated callers collapse vmap keys to their backend
// component before the composite fallthrough, which means a composite that
// re-enters through the ordinary ops:: wrappers would do exactly that.  The
// family therefore splits: with a transform active it decomposes over the
// registered batch rules (the composite-decomposition contract), without one
// it takes the plain path.

// alias shares storage, sizes, strides and the version counter but hands back
// a distinct tensor object, so metadata changes on either side stay
// independent.  Under a transform it is the identity: the wrapper must stay
// intact.
Tensor alias_native(const Tensor& self) {
    if (under_active_transform(self)) {
        return self;
    }
    const auto impl = self.unsafeGetTensorImpl();
    Tensor out(impl->storage(), impl->sizes().vec(), impl->strides().vec(),
               impl->dtype(), impl->storage_offset());
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    return out;
}

// mT swaps the last two dimensions; 0-d inputs are the identity and 1-d
// inputs are rejected the same way as the transpose itself.
Tensor mT_native(const Tensor& self) {
    const int64_t ndim = self.dim();
    TP_CHECK(ndim != 1,
             "tensor.mT is only supported on matrices or batches of matrices. "
             "Got 1-D tensor.");
    if (ndim == 0) {
        return self;
    }
    if (under_active_transform(self)) {
        return redispatch_below_transform<Tensor, const Tensor&, int64_t,
                                          int64_t>("transpose", self, -2, -1);
    }
    return ops::transpose(self, -2, -1);
}

// mH combines the conjugate with the mT swap.  The conjugate here is the
// materialized kernel, so the result carries real conjugated values.  No
// conjugate batch rule exists, so an active transform refuses the op.
Tensor mH_native(const Tensor& self) {
    const int64_t ndim = self.dim();
    TP_CHECK(ndim != 1,
             "tensor.mH is only supported on matrices or batches of matrices. "
             "Got 1-D tensor.");
    reject_active_transform(self, "mH");
    if (ndim == 0) {
        return ops::is_complex(self) ? ops::conj(self) : self;
    }
    return ops::transpose(ops::conj(self), -2, -1);
}

Tensor matrix_H_native(const Tensor& self) {
    const int64_t ndim = self.dim();
    TP_CHECK(ndim == 2 || ndim == 0,
             "matrix_H is only supported on matrices (2-D tensors) but got a ",
             ndim, "-D tensor; for batches of matrices use mH");
    if (ndim == 0) {
        return ops::is_complex(self) ? ops::conj(self) : self;
    }
    return mH_native(self);
}

// numpy_T reverses every dimension; 0-d and 1-d inputs are their own reverse.
Tensor numpy_T_native(const Tensor& self) {
    const int64_t ndim = self.dim();
    if (ndim <= 1) {
        return self;
    }
    std::vector<int64_t> dims(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        dims[static_cast<size_t>(i)] = ndim - 1 - i;
    }
    if (under_active_transform(self)) {
        return redispatch_below_transform<Tensor, const Tensor&,
                                          const std::vector<int64_t>&>(
            "permute", self, dims);
    }
    return ops::permute(self, dims);
}

Tensor view_as_native(const Tensor& self, const Tensor& other) {
    const auto sizes =
        static_cast<std::vector<int64_t>>(other.shape());
    if (under_active_transform(self)) {
        return redispatch_below_transform<Tensor, const Tensor&,
                                          const std::vector<int64_t>&>(
            "view", self, sizes);
    }
    return ops::view(self, sizes);
}

// Materialized variants of the _conj/_neg_view wrappers: the underlying
// kernels already produce real conjugated/negated values, so the "copy"
// naming carries no extra work.  neg has a batch rule, so the negated copy
// decomposes under a transform; conjugate has none and refuses.
Tensor _conj_copy_native(const Tensor& self) {
    reject_active_transform(self, "_conj_copy");
    return ops::conj(self);
}

Tensor _neg_view_copy_native(const Tensor& self) {
    if (under_active_transform(self)) {
        return redispatch_below_transform<Tensor, const Tensor&>("neg", self);
    }
    return ops::neg(self);
}

// -----------------------------------------------------------------------------
// Raw stride remapping
// -----------------------------------------------------------------------------

// _reshape_alias installs the requested size/stride pair on the existing
// storage without any compatibility checking: callers are expected to have
// validated the layout themselves.  It reads the wrapper's own (empty)
// storage, so an active transform refuses it.
Tensor _reshape_alias_native(const Tensor& self,
                             const std::vector<int64_t>& size,
                             const std::vector<int64_t>& stride) {
    reject_active_transform(self, "_reshape_alias");
    TP_CHECK(self.defined(), "_reshape_alias expected a defined tensor");
    TP_CHECK(size.size() == stride.size(),
             "_reshape_alias: size and stride must have the same length");
    const auto impl = self.unsafeGetTensorImpl();
    Tensor out(impl->storage(), size, stride, impl->dtype(),
               impl->storage_offset());
    out.unsafeGetTensorImpl()->share_version_counter(*impl);
    return out;
}

// Copy variant: materialize first, then remap strides on the fresh storage.
Tensor _reshape_alias_copy_native(const Tensor& self,
                                  const std::vector<int64_t>& size,
                                  const std::vector<int64_t>& stride) {
    reject_active_transform(self, "_reshape_alias_copy");
    TP_CHECK(size.size() == stride.size(),
             "_reshape_alias: size and stride must have the same length");
    Tensor out = ops::clone(self, kContiguous);
    const auto impl = out.unsafeGetTensorImpl();
    Tensor view(impl->storage(), size, stride, impl->dtype(),
                impl->storage_offset());
    view.unsafeGetTensorImpl()->share_version_counter(*impl);
    return view;
}

// as_strided_copy is as_strided under another name: the base op is already
// out-of-place, so no extra materialization is needed.  No as_strided batch
// rule exists, so an active transform refuses the op.
Tensor as_strided_copy_native(const Tensor& self,
                              const std::vector<int64_t>& size,
                              const std::vector<int64_t>& stride,
                              std::optional<int64_t> storage_offset) {
    reject_active_transform(self, "as_strided_copy");
    return ops::as_strided(self, size, stride, storage_offset);
}

// as_strided_scatter writes `src` through a size/stride window on a private
// clone of self; the window shares the clone's version counter so in-place
// bookkeeping still observes the write.  clone has no batch rule, so an
// active transform refuses the op.
Tensor as_strided_scatter_native(
        const Tensor& self, const Tensor& src,
        const std::vector<int64_t>& size, const std::vector<int64_t>& stride,
        std::optional<int64_t> storage_offset) {
    reject_active_transform(self, "as_strided_scatter");
    Tensor out = ops::clone(self, kContiguous);
    const auto impl = out.unsafeGetTensorImpl();
    Tensor view(impl->storage(), size, stride, impl->dtype(),
                static_cast<size_t>(storage_offset.value_or(0)));
    view.unsafeGetTensorImpl()->share_version_counter(*impl);
    ops::copy_(view, src);
    return out;
}

TENSORPLAY_LIBRARY_IMPL(Composite, ViewCopyComposite) {
    m.impl("alias_copy", alias_copy_native);
    m.impl("t_copy", t_copy_native);
    m.impl("permute_copy", permute_copy_native);
    m.impl("transpose_copy.int", transpose_copy_native);
    m.impl("squeeze_copy", squeeze_copy_native);
    m.impl("squeeze_copy.dim", squeeze_copy_dim_native);
    m.impl("squeeze_copy.dims", squeeze_copy_dims_native);
    m.impl("unsqueeze_copy", unsqueeze_copy_native);
    m.impl("select_copy.int", select_copy_native);
    m.impl("slice_copy.Tensor", slice_copy_native);
    m.impl("narrow_copy", narrow_copy_native);
    m.impl("diagonal_copy", diagonal_copy_native);
    m.impl("unbind_copy.int", unbind_copy_native);
    m.impl("split_copy.Tensor", split_copy_native);
    m.impl("split_copy.sizes", split_copy_sizes_native);
    m.impl("view_copy", view_copy_native);
    m.impl("view_copy.dtype", view_copy_dtype_native);
    m.impl("unfold_copy", unfold_copy_native);
    m.impl("expand_copy", expand_copy_native);
    m.impl("split_with_sizes_copy", split_with_sizes_copy_native);
    m.impl("unsafe_split_with_sizes", unsafe_split_with_sizes_native);
    m.impl("detach_copy", detach_copy_native);
    m.impl("detach_", detach__native);
    m.impl("alias", alias_native);
    m.impl("mT", mT_native);
    m.impl("mH", mH_native);
    m.impl("matrix_H", matrix_H_native);
    m.impl("numpy_T", numpy_T_native);
    m.impl("view_as", view_as_native);
    m.impl("_conj_copy", _conj_copy_native);
    m.impl("_neg_view_copy", _neg_view_copy_native);
    m.impl("_reshape_alias", _reshape_alias_native);
    m.impl("_reshape_alias_copy", _reshape_alias_copy_native);
    m.impl("as_strided_copy", as_strided_copy_native);
    m.impl("as_strided_scatter", as_strided_scatter_native);
}

} // namespace composite
} // namespace tensorplay
