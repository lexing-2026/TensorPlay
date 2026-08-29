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
}

} // namespace composite
} // namespace tensorplay
