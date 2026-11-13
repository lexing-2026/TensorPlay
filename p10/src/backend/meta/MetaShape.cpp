// Shape-only kernels for view and shape-manipulation ops on the meta device.
//
// Views (permute, diagonal, movedim, expand, narrow) reuse the storage-sharing
// as_strided path, which is metadata-only and therefore device-independent;
// splitting and stacking derive new geometries from existing ones; flip and
// roll would move elements around, so only the resulting extents are
// observable.  All validations follow the computing backends' rules so a call
// they would reject is rejected here too.

#include "MetaCommon.h"

#include "Exception.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

// ---------------------------------------------------------------------------
// Permute / movedim.
// ---------------------------------------------------------------------------

Tensor permute_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    if (static_cast<int64_t>(dims.size()) != ndim) {
        TP_THROW(RuntimeError,
                 "permute(sparse_coo): number of dimensions in the tensor input "
                 "does not match the length of the desired ordering of dimensions "
                 "i.e. input.dim() = ", ndim,
                 " is not equal to len(dims) = ", dims.size());
    }
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(static_cast<size_t>(ndim), false);
    for (int64_t i = 0; i < ndim; ++i) {
        const int64_t d = wrap_dim(dims[static_cast<size_t>(i)], ndim);
        if (seen[static_cast<size_t>(d)]) {
            TP_THROW(RuntimeError, "permute(): duplicate dims are not allowed.");
        }
        seen[static_cast<size_t>(d)] = true;
        new_sizes[static_cast<size_t>(i)] = self.size(d);
        new_strides[static_cast<size_t>(i)] = self.stride(d);
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor movedim_kernel(const Tensor& self,
                      const std::vector<int64_t>& source,
                      const std::vector<int64_t>& destination) {
    const int64_t ndim = self.dim();
    if (source.size() != destination.size()) {
        TP_THROW(RuntimeError,
                 "movedim: Invalid source or destination dims: source (",
                 source.size(), " dims) should contain the same number of dims "
                 "as destination (", destination.size(), " dims)");
    }
    std::vector<int64_t> src(source), dst(destination);
    for (auto& d : src) d = wrap_dim(d, ndim);
    for (auto& d : dst) d = wrap_dim(d, ndim);
    const auto all_unique = [](const std::vector<int64_t>& dims) {
        std::vector<int64_t> sorted = dims;
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    };
    if (!all_unique(src)) {
        TP_THROW(RuntimeError, "movedim: repeated dim in `source`");
    }
    if (!all_unique(dst)) {
        TP_THROW(RuntimeError, "movedim: repeated dim in `destination`");
    }
    if (ndim == 0) {
        return self.as_strided(shape_of(self), self.strides());
    }
    std::vector<bool> src_seen(static_cast<size_t>(ndim), false);
    std::vector<bool> dst_seen(static_cast<size_t>(ndim), false);
    for (const int64_t d : src) src_seen[static_cast<size_t>(d)] = true;
    for (const int64_t d : dst) dst_seen[static_cast<size_t>(d)] = true;

    std::vector<int64_t> permutation(ndim, -1);
    for (size_t k = 0; k < src.size(); ++k) {
        permutation[static_cast<size_t>(dst[k])] = src[k];
    }
    int64_t cursor = 0;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!dst_seen[static_cast<size_t>(i)]) {
            while (src_seen[static_cast<size_t>(cursor)]) ++cursor;
            permutation[static_cast<size_t>(i)] = cursor++;
        }
    }
    return permute_kernel(self, permutation);
}

// ---------------------------------------------------------------------------
// Diagonal.
// ---------------------------------------------------------------------------

Tensor diagonal_kernel(const Tensor& self, int64_t offset, int64_t dim1,
                       int64_t dim2) {
    const int64_t ndim = self.dim();
    const int64_t dim1_ = dim1, dim2_ = dim2;
    dim1 = wrap_dim(dim1, ndim);
    dim2 = wrap_dim(dim2, ndim);
    if (dim1 == dim2) {
        TP_THROW(RuntimeError, "diagonal dimensions cannot be identical ",
                 dim1_, ", ", dim2_);
    }
    const int64_t size1 = self.size(dim1);
    const int64_t size2 = self.size(dim2);
    const int64_t stride1 = self.stride(dim1);
    const int64_t stride2 = self.stride(dim2);

    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    sizes.reserve(static_cast<size_t>(ndim - 1));
    strides.reserve(static_cast<size_t>(ndim - 1));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) {
            sizes.push_back(self.size(i));
            strides.push_back(self.stride(i));
        }
    }

    int64_t diag_size;
    int64_t new_offset = static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset());
    if (offset >= 0) {
        diag_size = offset < size2 ? std::min(size1, size2 - offset) : 0;
    } else if (offset == std::numeric_limits<int64_t>::min()) {
        diag_size = 0;
    } else {
        const int64_t offset_abs = -offset;
        diag_size =
            offset_abs < size1 ? std::min(size1 - offset_abs, size2) : 0;
    }
    if (diag_size != 0) {
        if (offset >= 0) {
            new_offset += offset * stride2;
        } else {
            new_offset += (-offset) * stride1;
        }
    }
    sizes.push_back(diag_size);
    strides.push_back(stride1 + stride2);
    return self.as_strided(sizes, strides, new_offset);
}

// ---------------------------------------------------------------------------
// Expand / narrow.
// ---------------------------------------------------------------------------

Tensor expand_kernel(const Tensor& self, const std::vector<int64_t>& size,
                     bool /*implicit*/) {
    if (!self.defined()) TP_THROW(RuntimeError, "Tensor not defined");
    const int64_t ndim = self.dim();
    const int64_t new_ndim = static_cast<int64_t>(size.size());
    if (new_ndim < ndim) {
        TP_THROW(RuntimeError,
                 "expand: the number of sizes provided (", new_ndim,
                 ") must be greater or equal to the number of dimensions in the "
                 "tensor (", ndim, ")");
    }
    std::vector<int64_t> new_sizes(size);
    std::vector<int64_t> new_strides(static_cast<size_t>(new_ndim), 0);
    if (ndim == 0) {
        return self.as_strided(new_sizes, new_strides);
    }
    for (int64_t i = new_ndim - 1; i >= 0; --i) {
        const int64_t offset = new_ndim - 1 - i;
        const int64_t dim = ndim - 1 - offset;
        const int64_t sz = (dim >= 0) ? self.size(dim) : 1;
        const int64_t stride = (dim >= 0)
                                   ? self.stride(dim)
                                   : new_sizes[i + 1] * new_strides[i + 1];
        int64_t target = new_sizes[static_cast<size_t>(i)];
        if (target == -1) {
            if (dim < 0) {
                TP_THROW(RuntimeError,
                         "The expanded size of the tensor (", target,
                         ") isn't allowed in a leading, non-existing dimension ", i);
            }
            target = sz;
        }
        if (sz != target && sz != 1) {
            TP_THROW(RuntimeError,
                     "The expanded size of the tensor (", target,
                     ") must match the existing size (", sz,
                     ") at non-singleton dimension ", i);
        }
        new_sizes[static_cast<size_t>(i)] = target;
        new_strides[static_cast<size_t>(i)] = sz == target ? stride : 0;
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor narrow_kernel(const Tensor& self, int64_t dim, int64_t start,
                     int64_t length) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "narrow() cannot be applied to a 0-dim tensor.");
    }
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length must be non-negative.");
    }
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    const int64_t cur_size = self.size(dim);
    if (start < -cur_size || start > cur_size) {
        TP_THROW(IndexError, "start out of range (expected to be in range of [",
                 -cur_size, ", ", cur_size, "], but got ", start, ")");
    }
    if (start < 0) start += cur_size;
    if (start > cur_size - length) {
        TP_THROW(RuntimeError, "start (", start, ") + length (", length,
                 ") exceeds dimension size (", cur_size, ").");
    }
    return self.slice(dim, start, start + length, 1);
}

// ---------------------------------------------------------------------------
// Splitting.
// ---------------------------------------------------------------------------

std::vector<Tensor> split_kernel(const Tensor& self, int64_t split_size,
                                 int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    if (split_size < 0) {
        TP_THROW(RuntimeError,
                 "split expects split_size be non-negative, but got split_size=",
                 split_size);
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(dim);
    if (!(split_size > 0 || dim_size == 0)) {
        TP_THROW(RuntimeError,
                 "split_size can only be 0 if dimension size is 0, "
                 "but got dimension size of ", dim_size);
    }
    int64_t num_splits = 1;
    int64_t last_split_size = 0;
    if (split_size != 0) {
        const int64_t quotient = dim_size / split_size;
        const int64_t remainder = dim_size % split_size;
        num_splits = std::max<int64_t>(quotient + (remainder != 0 ? 1 : 0), 1);
        last_split_size =
            (dim_size != 0 && remainder == 0) ? split_size : remainder;
    }
    std::vector<Tensor> result;
    result.reserve(static_cast<size_t>(num_splits));
    for (int64_t i = 0; i < num_splits; ++i) {
        const int64_t length = i < num_splits - 1 ? split_size : last_split_size;
        result.push_back(self.slice(dim, i * split_size, i * split_size + length));
    }
    return result;
}

std::vector<Tensor> split_sizes_kernel(const Tensor& self,
                                       const std::vector<int64_t>& split_sizes,
                                       int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(dim);
    std::vector<Tensor> outs;
    outs.reserve(split_sizes.size());
    int64_t start = 0;
    for (const int64_t len : split_sizes) {
        if (len < 0) {
            TP_THROW(RuntimeError,
                     "split_with_sizes expects split_sizes have only "
                     "non-negative entries");
        }
        if (start + len > dim_size) {
            TP_THROW(RuntimeError,
                     "split_with_sizes: split_sizes add up to more than the "
                     "dimension size");
        }
        outs.push_back(self.slice(dim, start, start + len));
        start += len;
    }
    if (start != dim_size) {
        TP_THROW(RuntimeError,
                 "split_with_sizes: split_sizes add up to less than the "
                 "dimension size");
    }
    return outs;
}

// ---------------------------------------------------------------------------
// Stack: unsqueeze every entry along the new axis and concatenate.
// ---------------------------------------------------------------------------

Tensor stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "stack expects a non-empty TensorList");
    }
    const int64_t ndim = tensors[0].dim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        TP_THROW(IndexError, "stack: dimension out of range");
    }
    const std::vector<int64_t> entry_shape = shape_of(tensors[0]);
    DType out_dtype = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (shape_of(tensors[i]) != entry_shape) {
            TP_THROW(RuntimeError,
                     "stack expects each tensor to be equal size");
        }
        out_dtype = promoteTypes(out_dtype, tensors[i].dtype());
    }
    std::vector<int64_t> out_shape = entry_shape;
    out_shape.insert(out_shape.begin() + dim,
                     static_cast<int64_t>(tensors.size()));
    return empty_meta(out_shape, out_dtype, tensors[0].device());
}

// ---------------------------------------------------------------------------
// Flip / roll: element-moving ops whose only observable property is that the
// geometry is preserved.
// ---------------------------------------------------------------------------

Tensor flip_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t nd = self.dim();
    std::vector<bool> seen(static_cast<size_t>(nd > 0 ? nd : 1), false);
    for (const int64_t d : dims) {
        const int64_t w = wrap_dim(d, nd);
        if (seen[static_cast<size_t>(w)]) {
            TP_THROW(RuntimeError, "dim ", w,
                     " appears multiple times in the list of dims");
        }
        seen[static_cast<size_t>(w)] = true;
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor roll_kernel(const Tensor& self, const std::vector<int64_t>& /*shifts*/,
                   const std::vector<int64_t>& dims) {
    for (const int64_t d : dims) {
        (void)wrap_dim(d, self.dim());
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Repeat: the output shape is the input's shape aligned to the right under
// the repeat counts.
// ---------------------------------------------------------------------------

Tensor repeat_kernel(const Tensor& self, const std::vector<int64_t>& repeats) {
    const int64_t nd = self.dim();
    if (static_cast<int64_t>(repeats.size()) < nd) {
        TP_THROW(RuntimeError,
                 "Number of dimensions of repeat dims can not be smaller than "
                 "number of dimensions of tensor");
    }
    const int64_t out_nd = static_cast<int64_t>(repeats.size());
    std::vector<int64_t> padded(static_cast<size_t>(out_nd), 1);
    for (int64_t i = 0; i < nd; ++i) {
        padded[static_cast<size_t>(out_nd - nd + i)] = self.size(i);
    }
    std::vector<int64_t> target(static_cast<size_t>(out_nd));
    for (int64_t i = 0; i < out_nd; ++i) {
        const int64_t repeat = repeats[static_cast<size_t>(i)];
        if (repeat < 0) {
            TP_THROW(RuntimeError, "repeat: repeats must be non-negative");
        }
        const int64_t source = padded[static_cast<size_t>(i)];
        target[static_cast<size_t>(i)] =
            (source == 0 || repeat == 0) ? 0 : source * repeat;
    }
    return empty_meta(target, self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Transpose a matrix, and unfold a window over one axis.
// ---------------------------------------------------------------------------

Tensor t_kernel(const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "t() expects a tensor with <= 2 dimensions, but self is ",
                 self.dim(), "D");
    }
    if (self.dim() < 2) return self;
    return self.as_strided({self.size(1), self.size(0)},
                           {self.stride(1), self.stride(0)});
}

Tensor unfold_kernel(const Tensor& self, int64_t dimension, int64_t size,
                     int64_t step) {
    const int64_t nd = self.dim();
    dimension = wrap_dim(dimension, nd == 0 ? 1 : nd);
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    const int64_t max_size =
        nd == 0 ? 1 : sizes[static_cast<size_t>(dimension)];
    if (size < 0) {
        TP_THROW(RuntimeError, "size is ", size, " but must be >= 0");
    }
    if (size > max_size) {
        TP_THROW(RuntimeError, "maximum size for tensor at dimension ",
                 dimension, " is ", max_size, " but size is ", size);
    }
    if (step <= 0) {
        TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    }
    sizes.push_back(size);
    strides.push_back(nd == 0 ? 1 : strides[static_cast<size_t>(dimension)]);
    if (dimension < nd) {
        const size_t axis = static_cast<size_t>(dimension);
        sizes[axis] = (sizes[axis] - size) / step + 1;
        strides[axis] *= step;
    }
    return self.as_strided(sizes, strides);
}

// ---------------------------------------------------------------------------
// Triangular masks: the answer keeps the operand's geometry and type; only
// the entries below or above a diagonal survive.
// ---------------------------------------------------------------------------

Tensor triangular_kernel(const Tensor& self, int64_t /*diagonal*/) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError,
                 "tril/triu requires tensor with at least 2 dimensions");
    }
    if (!is_scalar_dtype(self.dtype())) {
        TP_THROW(TypeError, "tril/triu: unsupported dtype");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& triangular_inplace_kernel(Tensor& self, int64_t diagonal) {
    triangular_kernel(self, diagonal);
    return self;
}

// ---------------------------------------------------------------------------
// Diagonal embedding: each element of the last axis is laid on one diagonal
// of a new square plane, and the two new axes hold that diagonal's extent.
// ---------------------------------------------------------------------------

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sizes) {
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * sizes[static_cast<size_t>(i + 1)];
    }
    return strides;
}

int64_t diagonal_extent(int64_t base, int64_t offset, const char* op) {
    if (offset == std::numeric_limits<int64_t>::min()) {
        TP_THROW(ValueError, op, ": diagonal offset is too small");
    }
    const int64_t magnitude = offset < 0 ? -offset : offset;
    if (base > std::numeric_limits<int64_t>::max() - magnitude) {
        TP_THROW(ValueError, op, ": diagonal output dimension is too large");
    }
    return base + magnitude;
}

Tensor diag_embed_kernel(const Tensor& self, int64_t offset, int64_t dim1_,
                         int64_t dim2_) {
    const int64_t nDims = self.dim() + 1;
    const int64_t dim1 = wrap_dim(dim1_, nDims);
    const int64_t dim2 = wrap_dim(dim2_, nDims);
    if (dim1 == dim2) {
        TP_THROW(RuntimeError, "diagonal dimensions cannot be identical");
    }
    const int64_t extent = diagonal_extent(self.size(-1), offset, "diag_embed");
    std::vector<int64_t> sizes = shape_of(self);
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1, dim2), extent);
    sizes.insert(sizes.begin() + std::max(dim1, dim2), extent);
    return empty_meta(sizes, self.dtype(), self.device());
}

Tensor diagflat_kernel(const Tensor& self, int64_t offset) {
    const int64_t extent = diagonal_extent(self.numel(), offset, "diag_embed");
    return empty_meta({extent, extent}, self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Quarter turns: a rotation swaps the two plane axes for odd k and keeps the
// shape for even k.  The odd turns answer in the layout of a transposed
// contiguous plane, so their strides swap with the extents.
// ---------------------------------------------------------------------------

Tensor rot90_kernel(const Tensor& self, int64_t k,
                    const std::vector<int64_t>& dims) {
    if (dims.size() != 2) {
        TP_THROW(RuntimeError, "expected total rotation dims == 2, but got dims = ",
                 dims.size());
    }
    const int64_t nd = self.dim();
    if (nd < 2) {
        TP_THROW(RuntimeError, "expected total dims >= 2, but got total dims = ",
                 nd);
    }
    const int64_t dim0 = wrap_dim(dims[0], nd);
    const int64_t dim1 = wrap_dim(dims[1], nd);
    if (dim0 == dim1) {
        TP_THROW(RuntimeError,
                 "expected rotation dims to be different, but got dim0 = ",
                 dims[0], " and dim1 = ", dims[1]);
    }
    k = (4 + (k % 4)) % 4;
    if (k == 1 || k == 3) {
        std::vector<int64_t> sizes = shape_of(self);
        std::vector<int64_t> strides = contiguous_strides(sizes);
        std::swap(sizes[static_cast<size_t>(dim0)],
                  sizes[static_cast<size_t>(dim1)]);
        std::swap(strides[static_cast<size_t>(dim0)],
                  strides[static_cast<size_t>(dim1)]);
        return self.as_strided(sizes, strides);
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Cartesian product: every combination of one value from each 1-D operand,
// one combination per row, typed by the promoted operand type.
// ---------------------------------------------------------------------------

Tensor cartesian_prod_kernel(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError,
                 "cartesian_prod expects a non-empty TensorList");
    }
    DType dtype = tensors[0].dtype();
    int64_t rows = 1;
    for (const Tensor& tensor : tensors) {
        if (tensor.dim() != 1) {
            TP_THROW(RuntimeError, "Expect a 1D vector, but got shape ",
                     tensor.shape().toString());
        }
        rows *= tensor.numel();
        dtype = promoteTypes(dtype, tensor.dtype());
    }
    if (tensors.size() == 1) return tensors[0];
    return empty_meta({rows, static_cast<int64_t>(tensors.size())}, dtype,
                      tensors[0].device());
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaShape) {
    m.impl("t", t_kernel);
    m.impl("unfold", unfold_kernel);
    m.impl("tril", triangular_kernel);
    m.impl("tril_", triangular_inplace_kernel);
    m.impl("triu", triangular_kernel);
    m.impl("triu_", triangular_inplace_kernel);
    m.impl("diag_embed", diag_embed_kernel);
    m.impl("diagflat", diagflat_kernel);
    m.impl("rot90", rot90_kernel);
    m.impl("cartesian_prod", cartesian_prod_kernel);
    m.impl("permute", permute_kernel);
    m.impl("movedim", movedim_kernel);
    m.impl("diagonal", diagonal_kernel);
    m.impl("expand", expand_kernel);
    m.impl("narrow", narrow_kernel);
    m.impl("split", split_kernel);
    m.impl("split.sizes", split_sizes_kernel);
    m.impl("stack", stack_kernel);
    m.impl("flip", flip_kernel);
    m.impl("roll", roll_kernel);
    m.impl("repeat", repeat_kernel);
}

} // namespace meta
} // namespace tensorplay