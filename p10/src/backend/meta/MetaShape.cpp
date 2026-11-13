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
#include <string>
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

std::string format_sizes(const std::vector<int64_t>& sizes) {
    std::string text = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i != 0) text += ", ";
        text += std::to_string(sizes[i]);
    }
    text += "]";
    return text;
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
                     "non-negative entries, but got split_sizes=",
                     format_sizes(split_sizes));
        }
        if (len > dim_size - start) {
            TP_THROW(RuntimeError,
                     "split_with_sizes expects split_sizes to sum exactly to ",
                     dim_size, " (input tensor's size at dimension ", dim, ")");
        }
        outs.push_back(self.slice(dim, start, start + len));
        start += len;
    }
    if (start != dim_size) {
        TP_THROW(RuntimeError,
                 "split_with_sizes expects split_sizes to sum exactly to ",
                 dim_size, " (input tensor's size at dimension ", dim,
                 "), but got split_sizes=", format_sizes(split_sizes));
    }
    return outs;
}

// ---------------------------------------------------------------------------
// Unbind: one view per position along the axis, each dropping that axis.
// ---------------------------------------------------------------------------

std::vector<Tensor> unbind_kernel(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim < -1 || dim > 0) {
            TP_THROW(IndexError,
                     "Dimension out of range (expected to be in range of [-1, "
                     "0], but got ", dim, ")");
        }
        TP_THROW(IndexError,
                 "Dimension specified as 0 but tensor has no dimensions");
    }
    dim = wrap_dim(dim, nd);
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    const int64_t extent = self.size(dim);
    const int64_t step = self.stride(dim);
    const int64_t base = static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset());
    std::vector<Tensor> outs;
    outs.reserve(static_cast<size_t>(extent));
    for (int64_t i = 0; i < extent; ++i) {
        outs.push_back(self.as_strided(sizes, strides, base + i * step));
    }
    return outs;
}

// ---------------------------------------------------------------------------
// One-hot: append a class axis; the element type stays Int64.
// ---------------------------------------------------------------------------

Tensor one_hot_kernel(const Tensor& self, int64_t num_classes) {
    if (self.dtype() != DType::Int64) {
        TP_THROW(RuntimeError,
                 "one_hot is only applicable to index tensor of type "
                 "LongTensor.");
    }
    if (self.numel() == 0) {
        if (num_classes <= 0) {
            TP_THROW(RuntimeError,
                     "Can not infer total number of classes from empty tensor.");
        }
        std::vector<int64_t> out = shape_of(self);
        out.push_back(num_classes);
        return empty_meta(out, self.dtype(), self.device());
    }
    if (num_classes == -1) {
        TP_THROW(NotImplementedError,
                 "one_hot: num_classes = -1 needs the tensor's largest value, "
                 "which the meta device does not hold");
    }
    if (num_classes <= 0) {
        TP_THROW(RuntimeError, "Class values must be smaller than num_classes.");
    }
    std::vector<int64_t> out = shape_of(self);
    out.push_back(num_classes);
    return empty_meta(out, self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Kronecker product: pad both operands to the longer rank, then multiply the
// paired extents; the element type is the promoted operand type.
// ---------------------------------------------------------------------------

Tensor kron_kernel(const Tensor& self, const Tensor& other) {
    check_all_meta(self, other);
    const int64_t maxdim = std::max(self.dim(), other.dim());
    const int64_t pad_self = maxdim - self.dim();
    const int64_t pad_other = maxdim - other.dim();
    std::vector<int64_t> out(static_cast<size_t>(maxdim));
    for (int64_t i = 0; i < maxdim; ++i) {
        const int64_t a = i >= pad_self ? self.size(i - pad_self) : 1;
        const int64_t b = i >= pad_other ? other.size(i - pad_other) : 1;
        out[static_cast<size_t>(i)] = a * b;
    }
    return empty_meta(out, promoteTypes(self.dtype(), other.dtype()),
                      self.device());
}

// ---------------------------------------------------------------------------
// Trace: the sum along the main diagonal of a matrix, answered as a scalar.
// Integral (and boolean) operands widen to Int64.
// ---------------------------------------------------------------------------

Tensor trace_kernel(const Tensor& self) {
    if (self.dim() != 2) {
        TP_THROW(RuntimeError, "trace: expected a matrix, but got tensor with "
                 "dim ", self.dim());
    }
    const DType out_dtype =
        isIntegralType(self.dtype(), /*includeBool=*/true) ? DType::Int64
                                                           : self.dtype();
    return empty_meta({}, out_dtype, self.device());
}

// ---------------------------------------------------------------------------
// Bilinear upsampling: the two spatial axes are replaced by the requested
// extents, the leading axes are kept.
// ---------------------------------------------------------------------------

Tensor upsample_bilinear2d_kernel(const Tensor& self,
                                  const std::vector<int64_t>& output_size,
                                  bool /*align_corners*/,
                                  std::optional<double> /*scales_h*/,
                                  std::optional<double> /*scales_w*/) {
    std::vector<int64_t> out{self.size(0), self.size(1)};
    out.insert(out.end(), output_size.begin(), output_size.end());
    return empty_meta(out, self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Pixel shuffle: split the channel axis into the square of the upscale factor
// and move those sub-channels into two new trailing axes.
// ---------------------------------------------------------------------------

Tensor pixel_shuffle_kernel(const Tensor& self, int64_t upscale_factor) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError,
                 "pixel_shuffle expects input to have at least 3 dimensions, "
                 "but got input with ", self.dim(), " dimension(s)");
    }
    if (upscale_factor <= 0) {
        TP_THROW(RuntimeError, "pixel_shuffle expects a positive factor, but "
                 "got ", upscale_factor);
    }
    if (upscale_factor >
        std::numeric_limits<int64_t>::max() / upscale_factor) {
        TP_THROW(ValueError, "pixel_shuffle: factor is too large");
    }
    const int64_t factor_squared = upscale_factor * upscale_factor;
    const int64_t channels = self.size(-3);
    if (channels % factor_squared != 0) {
        TP_THROW(RuntimeError,
                 "pixel_shuffle: channel dim must be divisible by r^2");
    }
    const auto scale_extent = [&](int64_t extent) {
        if (extent > std::numeric_limits<int64_t>::max() / upscale_factor) {
            TP_THROW(ValueError, "pixel_shuffle: output dimension is too large");
        }
        return extent * upscale_factor;
    };
    std::vector<int64_t> out = shape_of(self);
    out.resize(out.size() - 3);
    out.push_back(channels / factor_squared);
    out.push_back(scale_extent(self.size(-2)));
    out.push_back(scale_extent(self.size(-1)));
    return empty_meta(out, self.dtype(), self.device());
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

// ---------------------------------------------------------------------------
// Tensor-list builders.
// ---------------------------------------------------------------------------

Tensor block_diag_kernel(const std::vector<Tensor>& tensors) {
    // Rectangular blocks laid along the diagonal; the empty call answers a
    // (1, 0) placeholder in the default floating type and device.  A rank-1
    // block spans one row and a rank-0 block is the 1x1 cell.
    if (tensors.empty()) {
        return empty_meta({1, 0}, DType::Float32, globalContext().defaultDevice());
    }
    const Device& device = tensors[0].device();
    DType out_dtype = tensors[0].dtype();
    int64_t rows = 0, cols = 0;
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        const Tensor& t = tensors[idx];
        if (!(t.device() == device)) {
            TP_THROW(RuntimeError,
                     "block_diag: input tensors must all be on the same device.");
        }
        out_dtype = promoteTypes(out_dtype, t.dtype());
        const int64_t nd = t.dim();
        if (nd > 2) {
            TP_THROW(RuntimeError,
                     "block_diag: Input tensors must have 2 or fewer dimensions. Input ",
                     static_cast<int64_t>(idx), " has ", nd, " dimensions");
        }
        rows += nd == 2 ? t.size(0) : 1;
        cols += nd == 2 ? t.size(1) : (nd == 1 ? t.size(0) : 1);
    }
    return empty_meta({rows, cols}, out_dtype, device);
}

std::vector<Tensor> broadcast_tensors_kernel(const std::vector<Tensor>& tensors) {
    // The common broadcast shape, then a stride-0 view for every operand that
    // needs one -- the expand view walks the source layout, repeating
    // dimensions grown from 1 and keeping every other source stride.  An
    // operand already at the common shape is returned unchanged.
    std::vector<int64_t> shape;
    for (const Tensor& t : tensors) {
        shape = broadcast_shapes(shape, shape_of(t));
    }
    std::vector<Tensor> outs;
    outs.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        if (shape_of(t) == shape) {
            outs.push_back(t);
            continue;
        }
        outs.push_back(t.expand(shape));
    }
    return outs;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Concatenation.
// ---------------------------------------------------------------------------
Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(ValueError, "cat(): expected a non-empty list of Tensors");
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].dim() == 0) {
            TP_THROW(RuntimeError, "zero-dimensional tensor (at position ", i,
                     ") cannot be concatenated");
        }
        if (tensors[i].device() != tensors[0].device()) {
            TP_THROW(DeviceMismatchError,
                     "cat(): all tensors must be on the same device");
        }
    }
    const int64_t nd = tensors[0].dim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd) {
        TP_THROW(IndexError, "cat(): dimension ", dim, " out of range");
    }

    DType out_dtype = tensors[0].dtype();
    int64_t size_at_dim = 0;
    for (const auto& t : tensors) {
        if (t.numel() == 0) continue;
        if (t.dim() != nd) {
            TP_THROW(RuntimeError, "cat(): tensors must have the same number of dims");
        }
        for (int64_t d = 0; d < nd; ++d) {
            if (d != dim && t.size(d) != tensors[0].size(d)) {
                TP_THROW(RuntimeError, "cat(): sizes of tensors must match except in ",
                         "dimension ", dim);
            }
        }
        size_at_dim += t.size(dim);
        out_dtype = promoteTypes(out_dtype, t.dtype());
    }

    std::vector<int64_t> out_shape = shape_of(tensors[0]);
    out_shape[static_cast<size_t>(dim)] = size_at_dim;
    return empty_meta(out_shape, out_dtype, tensors[0].device());
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Finite differences and the cross product. diff folds the optional boundary
// rows through the concatenation rules and then peels one slice per order,
// while cross broadcasts every axis but the length-three one.
// ---------------------------------------------------------------------------
int64_t wrap_diff_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                             "of [", min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

Tensor diff_kernel(const Tensor& self, int64_t n, int64_t dim,
                   const std::optional<Tensor>& prepend,
                   const std::optional<Tensor>& append) {
    const int64_t d = wrap_diff_dim(dim, self.dim());
    const bool has_prepend = prepend.has_value() && prepend->defined();
    const bool has_append = append.has_value() && append->defined();
    Tensor base = self;
    // A zero order answers the operand untouched, so the boundary rows are
    // neither joined nor validated.
    if ((has_prepend || has_append) && n != 0) {
        std::vector<Tensor> pieces;
        if (has_prepend) pieces.push_back(*prepend);
        pieces.push_back(self);
        if (has_append) pieces.push_back(*append);
        base = cat_kernel(pieces, d);
    }
    int64_t orders = n;
    if (orders > base.size(d)) orders = base.size(d);
    if (orders <= 0) {
        return empty_meta(shape_of(base), base.dtype(), base.device());
    }
    if (base.dtype() == DType::Bool) {
        TP_THROW(RuntimeError,
                 "Subtraction, the `-` operator, with two bool tensors is not "
                 "supported. Use the `^` or `logical_xor()` operator instead.");
    }
    std::vector<int64_t> out = shape_of(base);
    out[static_cast<size_t>(d)] -= orders;
    return empty_meta(out, base.dtype(), base.device());
}

int64_t wrap_cross_dim(int64_t dim, int64_t ndim) {
    if (dim < -ndim || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                             "of [", -ndim, ", ", ndim - 1, "], but got ", dim,
                 ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

std::vector<int64_t> cross_output_shape(const Tensor& self,
                                        const Tensor& other, int64_t& dim) {
    if (self.dim() != other.dim()) {
        TP_THROW(RuntimeError,
                 "linalg.cross: inputs must have the same number of "
                 "dimensions.");
    }
    if (self.dim() == 0) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                             "of [-1, 0], but got -1)");
    }
    dim = wrap_cross_dim(dim, self.dim());
    if (self.size(dim) != 3 || other.size(dim) != 3) {
        TP_THROW(RuntimeError, "linalg.cross: inputs dimension ", dim,
                 " must have length 3. Got ", self.size(dim), " and ",
                 other.size(dim));
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", toString(self.dtype()),
                 " but found ", toString(other.dtype()));
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "Expected all tensors to be on the same device, but got ",
                 self.device().toString(), " and ",
                 other.device().toString());
    }
    if (self.dtype() == DType::Bool) {
        TP_THROW(NotImplementedError, "\"cross\" not implemented for 'Bool'");
    }
    if (self.dtype() == DType::ComplexHalf ||
        self.dtype() == DType::BComplex32) {
        TP_THROW(NotImplementedError,
                 "\"cross\" not implemented for reduced complex dtype ",
                 toString(self.dtype()));
    }
    return broadcast_shapes(shape_of(self), shape_of(other));
}

Tensor linalg_cross_kernel(const Tensor& self, const Tensor& other,
                           int64_t dim) {
    return empty_meta(cross_output_shape(self, other, dim), self.dtype(),
                      self.device());
}

Tensor cross_kernel(const Tensor& self, const Tensor& other,
                    std::optional<int64_t> dim) {
    int64_t d = -1;
    if (dim.has_value()) {
        d = *dim;
    } else {
        for (int64_t axis = 0; axis < self.dim(); ++axis) {
            if (self.size(axis) == 3) {
                d = axis;
                break;
            }
        }
        if (d < 0) {
            TP_THROW(RuntimeError, "no dimension of size 3 in input");
        }
    }
    return empty_meta(cross_output_shape(self, other, d), self.dtype(),
                      self.device());
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// View kernels. These ops are backend-dispatched, so the meta device needs
// its own registrations; they reuse the storage-sharing as_strided path,
// which is metadata-only and therefore device-independent.
// ---------------------------------------------------------------------------
Tensor transpose_kernel(const Tensor& self, int64_t dim0, int64_t dim1) {
    const int64_t ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        TP_THROW(IndexError, "transpose: dimension out of range for a rank-",
                 ndim, " tensor");
    }
    std::vector<int64_t> sizes = shape_of(self);
    const std::vector<int64_t> strides = self.strides();
    std::vector<int64_t> new_strides(strides);
    std::swap(sizes[static_cast<size_t>(dim0)], sizes[static_cast<size_t>(dim1)]);
    std::swap(new_strides[static_cast<size_t>(dim0)], new_strides[static_cast<size_t>(dim1)]);
    return self.as_strided(sizes, new_strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor slice_kernel(const Tensor& self, int64_t dim,
                    std::optional<int64_t> start, std::optional<int64_t> end,
                    int64_t step) {
    return self.slice(dim, start.value_or(0),
                      end.value_or(std::numeric_limits<int64_t>::max()), step);
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Shapes that move extents around. These are views of their operand, so they
// go through the storage-sharing path and only the sizes and strides are
// computed here.
// ---------------------------------------------------------------------------
Tensor squeeze_kernel(const Tensor& self) {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (self.size(i) == 1) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor squeeze_dim_kernel(const Tensor& self, int64_t dim) {
    const int64_t ndim = self.dim();
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                                 "of [-1, 0], but got ", dim, ")");
        }
        return self.as_strided(shape_of(self), self.strides(),
                               static_cast<int64_t>(self.storage_offset()));
    }
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    }
    if (self.size(dim) != 1) {
        return self.as_strided(shape_of(self), self.strides(),
                               static_cast<int64_t>(self.storage_offset()));
    }
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor squeeze_dims_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t ndim = self.dim();
    std::vector<bool> drop(static_cast<size_t>(ndim), false);
    std::vector<bool> seen(static_cast<size_t>(ndim > 0 ? ndim : 1), false);
    for (int64_t d : dims) {
        int64_t wrapped = d < 0 ? d + ndim : d;
        if (ndim <= 0 || wrapped < 0 || wrapped >= ndim) {
            TP_THROW(IndexError, "Dimension out of range (expected to be in range "
                                 "of [", -ndim, ", ", ndim - 1, "], but got ", d, ")");
        }
        if (seen[static_cast<size_t>(wrapped)]) {
            TP_THROW(RuntimeError, "dim ", wrapped,
                     " appears multiple times in the list of dims");
        }
        seen[static_cast<size_t>(wrapped)] = true;
        drop[static_cast<size_t>(wrapped)] = true;
    }
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (drop[static_cast<size_t>(i)] && self.size(i) == 1) continue;
        sizes.push_back(self.size(i));
        strides.push_back(self.stride(i));
    }
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

Tensor unsqueeze_kernel(const Tensor& self, int64_t dim) {
    const int64_t ndim = self.dim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -(ndim + 1), ", ", ndim, "], but got ", dim, ")");
    }
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    // The inserted axis spans nothing, so its stride is whatever the axis
    // after it is worth in memory; appending one has nothing after it.
    const int64_t stride = dim == ndim ? 1 : self.size(dim) * self.stride(dim);
    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, stride);
    return self.as_strided(sizes, strides,
                           static_cast<int64_t>(self.storage_offset()));
}

std::vector<Tensor> chunk_kernel(const Tensor& self, int64_t chunks, int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
    }
    if (chunks <= 0) {
        TP_THROW(RuntimeError, "chunk expects `chunks` to be greater than 0, got: ",
                 chunks);
    }
    const int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "chunk(): dimension out of range");
    }
    const int64_t size = self.size(dim);
    const int64_t split = size / chunks + (size % chunks != 0 ? 1 : 0);
    std::vector<Tensor> out;
    if (split == 0) {
        // Splitting an empty dimension yields the requested number of empty
        // parts rather than none.
        out.reserve(static_cast<size_t>(chunks));
        for (int64_t i = 0; i < chunks; ++i) {
            out.push_back(self.slice(dim, 0, 0));
        }
        return out;
    }
    for (int64_t start = 0; start < size; start += split) {
        const int64_t end = start + split < size ? start + split : size;
        out.push_back(self.slice(dim, start, end));
    }
    return out;
}

// Padding extends the trailing dimensions; the pad vector lists the two sides
// of each of them, the last dimension first.  Negative sides narrow instead.
Tensor constant_pad_nd_kernel(const Tensor& self, const std::vector<int64_t>& pad,
                              const Scalar& /*value*/) {
    if (pad.size() % 2 != 0) {
        TP_THROW(ValueError, "Length of pad must be even but instead it equals ",
                 pad.size());
    }
    const int64_t rank = self.dim();
    const int64_t padded = static_cast<int64_t>(pad.size()) / 2;
    if (rank < padded) {
        TP_THROW(ValueError,
                 "Length of pad should be no more than twice the number of "
                 "dimensions of the input. Pad length is ", pad.size(),
                 " while the input has ", rank, " dimensions.");
    }
    std::vector<int64_t> out = shape_of(self);
    for (int64_t k = 0; k < padded; ++k) {
        const int64_t dim = rank - padded + k;
        const size_t pair = static_cast<size_t>(padded - 1 - k);
        const int64_t total = pad[2 * pair] + pad[2 * pair + 1];
        const int64_t extent = out[static_cast<size_t>(dim)] + total;
        if (extent < 0) {
            TP_THROW(RuntimeError,
                     "The input size ", out[static_cast<size_t>(dim)],
                     " plus negative padding ", total,
                     " resulted in a negative output size, which is invalid. "
                     "Check dimension ", dim, " of your input.");
        }
        out[static_cast<size_t>(dim)] = extent;
    }
    return empty_meta(out, self.dtype(), self.device());
}

// Resampling rearranges existing elements; only the requested output extent
// is observable, and the batch and channel extents pass through.
Tensor upsample_nearest2d_kernel(const Tensor& self,
                                 const std::vector<int64_t>& output_size,
                                 std::optional<double> /*scales_h*/,
                                 std::optional<double> /*scales_w*/) {
    if (self.dim() != 4) {
        TP_THROW(RuntimeError, "upsample_nearest2d: expected a 4-D input, got ",
                 self.dim(), "-D");
    }
    if (output_size.size() != 2) {
        TP_THROW(ValueError, "upsample_nearest2d: output_size must have 2 elements");
    }
    for (const int64_t size : output_size) {
        if (size <= 0) {
            TP_THROW(ValueError,
                     "upsample_nearest2d: output size must be positive");
        }
    }
    return empty_meta({self.size(0), self.size(1), output_size[0], output_size[1]},
                      self.dtype(), self.device());
}

// A tensor carries a quantizer or it does not; nothing about the values
// decides it, so the answer is available without reading any.
bool is_quantized_kernel(const Tensor& self) {
    return self.defined() && self.unsafeGetTensorImpl() != nullptr &&
           self.unsafeGetTensorImpl()->has_quantizer();
}

// Channel shuffling permutes the channel axis in blocks of `groups` and
// answers the operand's own extents; the channel extent must split evenly.
Tensor channel_shuffle_kernel(const Tensor& self, int64_t groups) {
    if (self.dim() <= 2) {
        TP_THROW(RuntimeError,
                 "channel_shuffle expects input with more than 2 dimensions");
    }
    if (groups <= 0) {
        TP_THROW(RuntimeError, "channel_shuffle: groups must be positive, but got ",
                 groups);
    }
    if (self.size(1) % groups != 0) {
        TP_THROW(RuntimeError,
                 "channel_shuffle: channel dim not divisible by groups");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// Unshuffling pixels folds each r x r tile back into the channel axis: the
// spatial extents must divide by r and the channel extent grows by r squared.
Tensor pixel_unshuffle_kernel(const Tensor& self, int64_t downscale_factor) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError,
                 "pixel_unshuffle expects input to have at least 3 dimensions, "
                 "but got input with ", self.dim(), " dimension(s)");
    }
    if (downscale_factor <= 0) {
        TP_THROW(RuntimeError, "pixel_unshuffle expects a positive factor, but got ",
                 downscale_factor);
    }
    const int64_t factor_squared =
        downscale_factor > std::numeric_limits<int64_t>::max() / downscale_factor
            ? -1
            : downscale_factor * downscale_factor;
    if (factor_squared < 0) {
        TP_THROW(ValueError, "pixel_unshuffle: factor is too large");
    }
    const int64_t channels = self.size(-3);
    const int64_t height = self.size(-2);
    const int64_t width = self.size(-1);
    if (height % downscale_factor != 0 || width % downscale_factor != 0) {
        TP_THROW(RuntimeError,
                 "pixel_unshuffle: spatial dims must be divisible by r");
    }
    if (channels > std::numeric_limits<int64_t>::max() / factor_squared) {
        TP_THROW(ValueError, "pixel_unshuffle: output channel dimension is too large");
    }
    std::vector<int64_t> out = shape_of(self);
    out.resize(out.size() - 3);
    out.push_back(channels * factor_squared);
    out.push_back(height / downscale_factor);
    out.push_back(width / downscale_factor);
    return empty_meta(out, self.dtype(), self.device());
}

// The scatter forms answer the destination's own extents after checking the
// source fits the slice it replaces: selecting drops a dimension, slicing
// keeps it with the span the step leaves, the diagonal keeps every dimension
// but the pair it walks.
Tensor select_scatter_kernel(const Tensor& self, const Tensor& src,
                             int64_t dim, int64_t index) {
    dim = wrap_dim(dim, self.dim());
    const int64_t length = self.size(dim);
    if (index < -length || index >= length) {
        TP_THROW(IndexError, "select(): index out of range");
    }
    if (index < 0) index += length;
    std::vector<int64_t> target = shape_of(self);
    target.erase(target.begin() + dim);
    if (target != shape_of(src)) {
        TP_THROW(RuntimeError,
                 "expected src to have a size equal to the target slice");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor slice_scatter_kernel(const Tensor& self, const Tensor& src, int64_t dim,
                            std::optional<int64_t> start,
                            std::optional<int64_t> end, int64_t step) {
    if (step <= 0) {
        TP_THROW(RuntimeError, "slice_scatter: step must be positive");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t length = self.size(dim);
    int64_t begin = start.value_or(0);
    int64_t finish = end.value_or(length);
    if (begin < 0) begin += length;
    if (finish < 0) finish += length;
    begin = std::max<int64_t>(0, std::min<int64_t>(begin, length));
    finish = std::max<int64_t>(0, std::min<int64_t>(finish, length));
    if (finish < begin) finish = begin;
    std::vector<int64_t> target = shape_of(self);
    target[static_cast<size_t>(dim)] =
        (finish - begin + step - 1) / step;
    if (target != shape_of(src)) {
        TP_THROW(RuntimeError,
                 "expected src to have a size equal to the target slice");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor diagonal_scatter_kernel(const Tensor& self, const Tensor& src,
                               int64_t offset, int64_t dim1, int64_t dim2) {
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
    int64_t diag_size;
    if (offset >= 0) {
        diag_size = offset < size2 ? std::min(size1, size2 - offset) : 0;
    } else if (offset == std::numeric_limits<int64_t>::min()) {
        diag_size = 0;
    } else {
        const int64_t offset_abs = -offset;
        diag_size =
            offset_abs < size1 ? std::min(size1 - offset_abs, size2) : 0;
    }
    std::vector<int64_t> target;
    target.reserve(static_cast<size_t>(ndim - 1));
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) {
            target.push_back(self.size(i));
        }
    }
    target.push_back(diag_size);
    if (target != shape_of(src)) {
        TP_THROW(RuntimeError,
                 "expected src to have a size equal to the target slice");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// The Vandermonde layout carries one row per element and `N` columns of
// successive powers; the element type gains the width the powers need.
Tensor vander_kernel(const Tensor& x, std::optional<int64_t> n,
                     bool /*increasing*/) {
    if (x.dim() != 1) {
        TP_THROW(RuntimeError, "x must be a one-dimensional tensor.");
    }
    const int64_t columns = n.value_or(x.size(0));
    if (columns < 0) {
        TP_THROW(RuntimeError, "N must be non-negative.");
    }
    return empty_meta({x.size(0), columns},
                      promoteTypes(x.dtype(), DType::Int64), x.device());
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
    m.impl("split_with_sizes", split_sizes_kernel);
    m.impl("stack", stack_kernel);
    m.impl("flip", flip_kernel);
    m.impl("roll", roll_kernel);
    m.impl("repeat", repeat_kernel);
    m.impl("unbind", unbind_kernel);
    m.impl("one_hot", one_hot_kernel);
    m.impl("kron", kron_kernel);
    m.impl("trace", trace_kernel);
    m.impl("upsample_bilinear2d", upsample_bilinear2d_kernel);
    m.impl("pixel_shuffle", pixel_shuffle_kernel);
    m.impl("pixel_unshuffle", pixel_unshuffle_kernel);
    m.impl("channel_shuffle", channel_shuffle_kernel);
    m.impl("select_scatter", select_scatter_kernel);
    m.impl("slice_scatter", slice_scatter_kernel);
    m.impl("diagonal_scatter", diagonal_scatter_kernel);
    m.impl("vander", vander_kernel);
    m.impl("block_diag", block_diag_kernel);
    m.impl("broadcast_tensors", broadcast_tensors_kernel);
    m.impl("cat", cat_kernel);
    m.impl("diff", diff_kernel);
    m.impl("cross", cross_kernel);
    m.impl("linalg_cross", linalg_cross_kernel);
    m.impl("transpose", transpose_kernel);
    m.impl("slice.Tensor", slice_kernel);
    m.impl("squeeze", squeeze_kernel);
    m.impl("squeeze.dim", squeeze_dim_kernel);
    m.impl("squeeze.dims", squeeze_dims_kernel);
    m.impl("unsqueeze", unsqueeze_kernel);
    m.impl("chunk", chunk_kernel);
    m.impl("constant_pad_nd", constant_pad_nd_kernel);
    m.impl("upsample_nearest2d", upsample_nearest2d_kernel);
    m.impl("is_quantized", is_quantized_kernel);
}

} // namespace meta
} // namespace tensorplay
