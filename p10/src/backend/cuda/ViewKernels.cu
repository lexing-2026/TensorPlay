#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "TypePromotion.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

namespace tensorplay {
namespace cuda {

namespace {

Tensor view_as_real_cuda(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "view_as_real: input must be defined");
    }
    if (!isComplexType(self.dtype())) {
        TP_THROW(RuntimeError,
                "view_as_real is only supported for complex tensors, but got " +
                std::string(toString(self.dtype())));
    }

    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides = self.strides();
    for (auto& stride : strides) {
        if (stride > std::numeric_limits<int64_t>::max() / 2) {
            TP_THROW(RuntimeError, "view_as_real: stride overflow");
        }
        stride *= 2;
    }
    sizes.push_back(2);
    strides.push_back(1);

    const size_t offset = self.unsafeGetTensorImpl()->storage_offset();
    if (offset > std::numeric_limits<size_t>::max() / 2) {
        TP_THROW(RuntimeError, "view_as_real: storage offset overflow");
    }
    Tensor result(self.unsafeGetTensorImpl()->storage(), sizes, strides,
                  toRealValueType(self.dtype()), offset * 2);
    result.unsafeGetTensorImpl()->share_version_counter(
        *self.unsafeGetTensorImpl());
    return result;
}

Tensor view_as_complex_cuda(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "view_as_complex: input must be defined");
    }
    if (self.dtype() != DType::Float16 && self.dtype() != DType::Float32 &&
        self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError,
                "view_as_complex is only supported for half, float and double "
                "tensors, but got " + std::string(toString(self.dtype())));
    }
    if (self.dim() == 0 || self.size(self.dim() - 1) != 2) {
        TP_THROW(RuntimeError,
                "view_as_complex: input tensor must have a last dimension of size 2");
    }
    if (self.stride(self.dim() - 1) != 1) {
        TP_THROW(RuntimeError,
                "view_as_complex: last dimension must have stride 1");
    }
    for (int64_t dim = 0; dim + 1 < self.dim(); ++dim) {
        if ((self.stride(dim) & 1) != 0) {
            TP_THROW(RuntimeError,
                    "view_as_complex: strides of all dimensions except the last "
                    "must be divisible by 2");
        }
    }
    const size_t offset = self.unsafeGetTensorImpl()->storage_offset();
    if ((offset & 1) != 0) {
        TP_THROW(RuntimeError,
                "view_as_complex: storage offset must be divisible by 2");
    }

    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides = self.strides();
    sizes.pop_back();
    strides.pop_back();
    for (auto& stride : strides) {
        stride /= 2;
    }

    Tensor result(self.unsafeGetTensorImpl()->storage(), sizes, strides,
                  toComplexType(self.dtype()), offset / 2);
    result.unsafeGetTensorImpl()->share_version_counter(
        *self.unsafeGetTensorImpl());
    return result;
}

bool is_complex_cuda(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "is_complex: input must be defined");
    }
    return isComplexType(self.dtype());
}

} // namespace

namespace join_detail {

std::string fmt_sizes(const std::vector<int64_t>& sizes) {
    std::string r = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i) r += ", ";
        r += std::to_string(sizes[i]);
    }
    r += "]";
    return r;
}

int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Scalar wrapping defaults to true: rank-0 tensors accept dims [-1, 0]
// (both wrap to 0).
int64_t wrap_dim_scalar(int64_t dim, int64_t ndim) {
    return wrap_dim(dim, ndim == 0 ? 1 : ndim);
}

// TensorShape.h cat_should_skip_tensor: size-[0] 1-d tensors are skipped for
// wrap-dim / shape-check purposes (backwards compatibility).
bool should_skip(const Tensor& t) {
    return t.numel() == 0 && t.dim() == 1;
}

} // namespace join_detail

Tensor reshape_kernel_cuda(const Tensor& self, const std::vector<int64_t>& shape) {
    // whenever the layout admits the view (computeStride), otherwise it is a
    // the ambiguous 0-element -1 case that used to divide by zero).
    if (self.is_sparse()) {
        TP_THROW(RuntimeError, "reshape is not implemented for sparse tensors");
    }
    std::vector<int64_t> inferred = SizesAndStrides::infer_size(shape, self.numel());
    auto stride = SizesAndStrides::compute_view_strides(
        static_cast<std::vector<int64_t>>(self.shape()), self.strides(), inferred);
    if (stride.has_value()) {
        return self.as_strided(inferred, *stride);
    }
    // The clone must be explicitly contiguous: clone() with Preserve keeps
    // non-overlapping-and-dense strides (e.g. transposed), which the
    // subsequent view would reject.
    return self.clone(static_cast<int64_t>(MemoryFormat::Contiguous)).view(inferred);
}

Tensor transpose_kernel_cuda(const Tensor& self, int64_t dim0, int64_t dim1) {
    // TensorShape.cpp transpose: maybe_wrap_dim both dims (wrap_scalar=true
    // makes transpose(0, 0) a no-op on 0-d tensors), then swap sizes/strides.
    const int64_t ndim = self.dim();
    dim0 = join_detail::wrap_dim_scalar(dim0, ndim);
    dim1 = join_detail::wrap_dim_scalar(dim1, ndim);
    if (dim0 == dim1) {
        return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                               self.strides());
    }
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> new_strides = self.strides();
    std::swap(new_sizes[dim0], new_sizes[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return self.as_strided(new_sizes, new_strides);
}

Tensor t_kernel_cuda(const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError, "t() expects a tensor with <= 2 dimensions, but self is " + std::to_string(self.dim()) + "D");
    }
    if (self.dim() < 2) return self;
    return transpose_kernel_cuda(self, 0, 1);
}

Tensor permute_kernel_cuda(const Tensor& self, const std::vector<int64_t>& dims) {
    // TensorShape.cpp _permute_size_stride_estimation.
    const int64_t ndim = self.dim();
    if (dims.size() != static_cast<size_t>(ndim)) {
        TP_THROW(RuntimeError,
                 "permute(sparse_coo): number of dimensions in the tensor input ",
                 "does not match the length of the desired ordering of dimensions ",
                 "i.e. input.dim() = ", ndim,
                 " is not equal to len(dims) = ", dims.size());
    }
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(ndim, false);
    for (int64_t i = 0; i < ndim; ++i) {
        const int64_t d = join_detail::wrap_dim_scalar(dims[i], ndim);
        if (seen[d]) TP_THROW(RuntimeError, "permute(): duplicate dims are not allowed.");
        seen[d] = true;
        new_sizes[i] = self.size(d);
        new_strides[i] = self.stride(d);
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor squeeze_kernel_cuda(const Tensor& self) {
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (self.size(i) != 1) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor squeeze_dim_kernel_cuda(const Tensor& self, int64_t dim) {
    // TensorShape.cpp squeeze(dim): maybe_wrap_dim (wrap_scalar=true makes
    // squeeze(0) a no-op on 0-d tensors); non-singleton dims return an
    // equivalent view.
    const int64_t ndim = self.dim();
    dim = join_detail::wrap_dim_scalar(dim, ndim);
    if (ndim == 0 || self.size(dim) != 1) {
        return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                               self.strides());
    }
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor squeeze_dims_kernel_cuda(const Tensor& self, const std::vector<int64_t>& dims) {
    // TensorShape.cpp squeeze(dims): dim_list_to_bitset (WrapDimUtilsMulti.h)
    // wraps with wrap_scalar=true and rejects duplicates, then squeezes every
    // listed size-1 dim.
    const int64_t ndim = self.dim();
    std::vector<bool> seen(ndim > 0 ? ndim : 1, false);
    std::vector<bool> mask(ndim, false);
    for (auto d : dims) {
        const int64_t w = join_detail::wrap_dim_scalar(d, ndim);
        if (ndim > 0) {
            if (seen[w]) {
                TP_THROW(RuntimeError, "dim ", w,
                         " appears multiple times in the list of dims");
            }
            seen[w] = true;
            mask[w] = true;
        }
    }
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!(mask[i] && self.size(i) == 1)) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }
    return self.as_strided(new_sizes, new_strides);
}

Tensor unsqueeze_kernel_cuda(const Tensor& self, int64_t dim) {
    // TensorShape.cpp unsqueeze + inferUnsqueezeGeometry: the inserted dim
    // stride is size(dim)*stride(dim) (1 when appended at the end).
    const int64_t ndim = self.dim();
    dim = join_detail::wrap_dim(dim, ndim + 1);

    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> new_strides = self.strides();

    const int64_t new_stride =
        (dim == ndim) ? 1 : self.size(dim) * self.stride(dim);
    new_sizes.insert(new_sizes.begin() + dim, 1);
    new_strides.insert(new_strides.begin() + dim, new_stride);

    return self.as_strided(new_sizes, new_strides);
}

// Tensor-list view operators need an explicit CUDA registration.  The actual
// copies are delegated to copy_ so they inherit the stream-aware CUDA allocator
// and non-blocking copy semantics; this keeps the implementation correct for
// non-contiguous inputs while avoiding a second bespoke concatenation kernel.
Tensor cat_kernel_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    // check_cat_no_zero_dim -> legacy_cat_wrap_dim -> non-empty check ->
    // first-valid-tensor selection -> shape checks -> result_type promotion.
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].dim() == 0) {
            TP_THROW(RuntimeError, "zero-dimensional tensor (at position ", i,
                     ") cannot be concatenated");
        }
    }
    for (const auto& t : tensors) {
        if (!join_detail::should_skip(t)) {
            dim = join_detail::wrap_dim(dim, t.dim());
            break;
        }
    }
    if (tensors.empty()) {
        TP_THROW(ValueError, "cat(): expected a non-empty list of Tensors");
    }

    DType out_dtype = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        out_dtype = promoteTypes(out_dtype, tensors[i].dtype());
    }

    int64_t valid = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!join_detail::should_skip(tensors[i])) { valid = static_cast<int64_t>(i); break; }
    }

    std::vector<int64_t> out_shape{0};
    if (valid >= 0) {
        const Tensor& first = tensors[valid];
        if (dim > first.dim()) {
            TP_THROW(IndexError, "cat(): dimension ", dim, " out of range");
        }
        int64_t size_at_dim = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const Tensor& t = tensors[i];
            if (t.device() != tensors[0].device()) {
                TP_THROW(DeviceMismatchError, "cat(): all tensors must be on the same device");
            }
            if (join_detail::should_skip(t)) continue;
            if (t.dim() != first.dim()) {
                TP_THROW(RuntimeError, "Tensors must have same number of dimensions: got ",
                         first.dim(), " and ", t.dim());
            }
            for (int64_t d = 0; d < first.dim(); ++d) {
                if (d == dim) continue;
                if (t.size(d) != first.size(d)) {
                    TP_THROW(RuntimeError, "Sizes of tensors must match except in dimension ",
                             dim, ". Expected size ", first.size(d), " but got size ",
                             t.size(d), " for tensor number ", i, " in the list.");
                }
            }
            size_at_dim += t.size(dim);
        }
        out_shape = static_cast<std::vector<int64_t>>(first.shape());
        out_shape[dim] = size_at_dim;
    }

    Tensor out = Tensor::empty(out_shape, out_dtype, tensors[0].device());

    int64_t offset = 0;
    for (const auto& t : tensors) {
        if (join_detail::should_skip(t)) continue;
        const int64_t size = t.size(dim);
        if (size > 0) {
            Tensor out_slice = out.slice(dim, offset, offset + size);
            out_slice.copy_(t, /*non_blocking=*/true);
            offset += size;
        }
    }
    return out;
}

std::vector<Tensor> split_kernel_cuda(const Tensor& self, int64_t split_size, int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    if (split_size < 0) {
        TP_THROW(RuntimeError, "split expects split_size be non-negative, but got split_size=",
                 split_size);
    }
    dim = join_detail::wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(dim);
    if (!(split_size > 0 || dim_size == 0)) {
        TP_THROW(RuntimeError, "split_size can only be 0 if dimension size is 0, "
                 "but got dimension size of ", dim_size);
    }
    int64_t num_splits = 1;
    if (split_size != 0) {
        num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
    }
    const int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
    std::vector<Tensor> result;
    result.reserve(num_splits);
    for (int64_t i = 0; i < num_splits; ++i) {
        const int64_t length = i < num_splits - 1 ? split_size : last_split_size;
        result.push_back(self.slice(dim, i * split_size, i * split_size + length));
    }
    return result;
}

std::vector<Tensor> split_sizes_kernel_cuda(const Tensor& self,
                                            const std::vector<int64_t>& split_sizes,
                                            int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    dim = join_detail::wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(dim);

    std::vector<Tensor> result;
    result.reserve(split_sizes.size());
    int64_t start_idx = 0;
    for (const auto length : split_sizes) {
        if (length < 0) {
            TP_THROW(RuntimeError, "split_with_sizes expects split_sizes have only non-negative "
                     "entries, but got split_sizes=", join_detail::fmt_sizes(split_sizes));
        }
        result.push_back(self.slice(dim, start_idx, start_idx + length));
        start_idx += length;
    }
    if (start_idx != dim_size) {
        TP_THROW(RuntimeError, "split_with_sizes expects split_sizes to sum exactly to ",
                 dim_size, " (input tensor's size at dimension ", dim,
                 "), but got split_sizes=", join_detail::fmt_sizes(split_sizes));
    }
    return result;
}

std::vector<Tensor> chunk_kernel_cuda(const Tensor& self, int64_t chunks, int64_t dim) {
    // still produce `chunks` empty chunks, so it routes through split_with_sizes.
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
    }
    if (chunks <= 0) {
        TP_THROW(RuntimeError, "chunk expects `chunks` to be greater than 0, got: ", chunks);
    }
    dim = join_detail::wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(dim);
    const int64_t split_size = (dim_size + chunks - 1) / chunks;
    if (split_size == 0 && dim_size == 0) {
        std::vector<int64_t> sizes(static_cast<size_t>(chunks), 0);
        sizes[static_cast<size_t>(chunks - 1)] = split_size - (split_size * chunks - dim_size);
        return split_sizes_kernel_cuda(self, sizes, dim);
    }
    return split_kernel_cuda(self, split_size, dim);
}

std::vector<Tensor> unbind_kernel_cuda(const Tensor& self, int64_t dim) {
    // size() raises the no-dimensions error for 0-d.
    const int64_t ndim = self.dim();
    int64_t d;
    if (ndim == 0) {
        if (dim < -1 || dim > 0) {
            TP_THROW(IndexError, "Dimension out of range (expected to be in range of [-1, 0], but got ", dim, ")");
        }
        TP_THROW(IndexError, "Dimension specified as ", 0, " but tensor has no dimensions");
    }
    d = join_detail::wrap_dim(dim, ndim);
    std::vector<Tensor> result;
    int64_t size_dim = self.size(d);
    result.reserve(size_dim);
    for (int64_t i = 0; i < size_dim; ++i) {
        result.push_back(self.select(d, i));
    }
    return result;
}

Tensor stack_kernel_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    // check_stack_inputs, then cat of unsqueezed inputs (dtype promotion
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "stack expects a non-empty TensorList");
    }
    const int64_t ndim = tensors[0].dim();
    dim = join_detail::wrap_dim(dim, ndim + 1);

    const std::vector<int64_t> entry_shape = static_cast<std::vector<int64_t>>(tensors[0].shape());
    for (size_t i = 1; i < tensors.size(); ++i) {
        const std::vector<int64_t> sh = static_cast<std::vector<int64_t>>(tensors[i].shape());
        if (sh != entry_shape) {
            TP_THROW(RuntimeError, "stack expects each tensor to be equal size, but got ",
                     join_detail::fmt_sizes(entry_shape), " at entry 0 and ",
                     join_detail::fmt_sizes(sh), " at entry ", i);
        }
    }

    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return cat_kernel_cuda(unsqueezed, dim);
}

Tensor permute_backward_kernel_cuda(const Tensor& grad, const Tensor& self, const std::vector<int64_t>& dims) {
    int64_t ndim = grad.dim();
    if (dims.size() != (size_t)ndim) {
        TP_THROW(RuntimeError, "permute_backward: dims size mismatch");
    }
    std::vector<int64_t> inv_dims(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        inv_dims[dims[i]] = i;
    }
    return grad.permute(inv_dims);
}

Tensor squeeze_backward_kernel_cuda(const Tensor& grad, const Tensor& self) {
    return grad.reshape(static_cast<std::vector<int64_t>>(self.shape()));
}

// Pure metadata op: identical to the CPU kernel, safe on any device.
Tensor diagonal_kernel_cuda(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2) {
    // TensorShape.cpp diagonal: wrap first (so 0/1-d inputs raise the
    // maybe_wrap_dim IndexError), then reject identical dims reporting the
    // original arguments.
    const int64_t ndim = self.dim();
    const int64_t dim1_ = dim1, dim2_ = dim2;
    dim1 = join_detail::wrap_dim_scalar(dim1, ndim);
    dim2 = join_detail::wrap_dim_scalar(dim2, ndim);
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
    sizes.reserve(ndim - 1);
    strides.reserve(ndim - 1);
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) {
            sizes.push_back(self.size(i));
            strides.push_back(self.stride(i));
        }
    }

    int64_t diag_size;
    int64_t new_offset = static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
    if (offset >= 0) {
        diag_size = std::max<int64_t>(std::min(size1, size2 - offset), 0);
    } else {
        diag_size = std::max<int64_t>(std::min(size1 + offset, size2), 0);
    }
    // NumPy allows offsets "off the end"; don't set a ridiculous storage
    // offset when the diagonal is empty.
    if (diag_size != 0) {
        if (offset >= 0) {
            new_offset += offset * stride2;
        } else {
            new_offset -= offset * stride1;
        }
    }
    sizes.push_back(diag_size);
    strides.push_back(stride1 + stride2);
    return self.as_strided(sizes, strides, new_offset);
}

Tensor diagonal_backward_kernel_cuda(const Tensor& grad, const std::vector<int64_t>& input_sizes,
                                     int64_t offset, int64_t dim1, int64_t dim2) {
    Tensor result = Tensor::zeros(input_sizes, grad.dtype(), grad.device());
    Tensor diag_view = diagonal_kernel_cuda(result, offset, dim1, dim2);
    if (diag_view.numel() != grad.numel()) {
        TP_THROW(RuntimeError, "diagonal_backward: gradient shape mismatch");
    }
    diag_view.copy_(grad.reshape(static_cast<std::vector<int64_t>>(diag_view.shape())));
    return result;
}

Tensor movedim_kernel_cuda(const Tensor& self, const std::vector<int64_t>& source,
                           const std::vector<int64_t>& destination) {
    // TensorShape.cpp movedim.
    const int64_t ndim = self.dim();
    if (source.size() != destination.size()) {
        TP_THROW(RuntimeError, "movedim: Invalid source or destination dims: source (",
                 join_detail::fmt_sizes(source),
                 " dims) should contain the same number of dims as destination (",
                 join_detail::fmt_sizes(destination), " dims)");
    }
    std::vector<int64_t> src(source), dst(destination);
    for (auto& d : src) d = join_detail::wrap_dim_scalar(d, ndim);
    for (auto& d : dst) d = join_detail::wrap_dim_scalar(d, ndim);
    auto all_unique = [](std::vector<int64_t> dims) {
        std::sort(dims.begin(), dims.end());
        return std::adjacent_find(dims.begin(), dims.end()) == dims.end();
    };
    if (!all_unique(src)) {
        TP_THROW(RuntimeError, "movedim: repeated dim in `source` (",
                 join_detail::fmt_sizes(source), ")");
    }
    if (!all_unique(dst)) {
        TP_THROW(RuntimeError, "movedim: repeated dim in `destination` (",
                 join_detail::fmt_sizes(destination), ")");
    }
    // handle the case of scalar tensor as a no-op
    if (ndim == 0) {
        return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                               self.strides());
    }

    std::vector<bool> src_seen(ndim, false), dst_seen(ndim, false);
    for (const int64_t d : src) src_seen[d] = true;
    for (const int64_t d : dst) dst_seen[d] = true;

    // Destination slots take the moved dimensions in order; every other slot
    // keeps its original dimension, preserving ascending input order.
    std::vector<int64_t> permutation(ndim, -1);
    for (size_t k = 0; k < src.size(); ++k) permutation[dst[k]] = src[k];
    int64_t cursor = 0;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!dst_seen[i]) {
            while (src_seen[cursor]) ++cursor;
            permutation[i] = cursor++;
        }
    }
    return permute_kernel_cuda(self, permutation);
}

// --- clone / slice / contiguous --------------------------------------------
// The detail implementations live in libp10 and route copies by device, so
// the CUDA registrations share the CPU bodies.  Sparse layouts are rejected
// before the dense-stride math, matching the CPU guards.  (select/item use
// skip_implementation -- their core Tensor methods are the implementation.)
Tensor clone_kernel_cuda(const Tensor& self, std::optional<int64_t> memory_format) {
    std::optional<MemoryFormat> format;
    if (memory_format.has_value()) {
        format = static_cast<MemoryFormat>(*memory_format);
    }
    return tensorplay::detail::clone_impl(self, format);
}

Tensor slice_kernel_cuda(const Tensor& self, int64_t dim,
                         std::optional<int64_t> start,
                         std::optional<int64_t> end, int64_t step) {
    if (self.is_sparse()) {
        TP_THROW(RuntimeError, "slice() is not supported for sparse COO tensors");
    }
    return self.slice(dim, start.value_or(0),
                      end.value_or(std::numeric_limits<int64_t>::max()), step);
}

Tensor contiguous_kernel_cuda(const Tensor& self, int64_t memory_format) {
    return tensorplay::detail::contiguous_impl(self, memory_format);
}

Tensor select_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self,
                                   int64_t dim, int64_t index) {
    if (self.is_sparse()) {
        TP_THROW(RuntimeError, "select(): gradient w.r.t. sparse COO tensors is not supported");
    }
    Tensor out(self.shape(), grad_output.dtype(), grad_output.device());
    out.zero_();
    out.select(dim, index).copy_(grad_output);
    return out;
}

Tensor slice_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self,
                                  int64_t dim, std::optional<int64_t> start,
                                  std::optional<int64_t> end, int64_t step) {
    if (self.is_sparse()) {
        TP_THROW(RuntimeError, "slice(): gradient w.r.t. sparse COO tensors is not supported");
    }
    Tensor out(self.shape(), grad_output.dtype(), grad_output.device());
    out.zero_();
    out.slice(dim, start.value_or(0),
              end.value_or(std::numeric_limits<int64_t>::max()), step)
        .copy_(grad_output);
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ViewKernels) {
    m.impl("view_as_real", view_as_real_cuda);
    m.impl("view_as_complex", view_as_complex_cuda);
    m.impl("is_complex", is_complex_cuda);
    m.impl("reshape", reshape_kernel_cuda);
    m.impl("transpose", transpose_kernel_cuda);
    m.impl("t", t_kernel_cuda);
    m.impl("permute", permute_kernel_cuda);
    m.impl("permute_backward", permute_backward_kernel_cuda);
    m.impl("squeeze", squeeze_kernel_cuda);
    m.impl("squeeze_backward", squeeze_backward_kernel_cuda);
    m.impl("squeeze.dim", squeeze_dim_kernel_cuda);
    m.impl("squeeze.dims", squeeze_dims_kernel_cuda);
    m.impl("unsqueeze", unsqueeze_kernel_cuda);
    m.impl("clone", clone_kernel_cuda);
    m.impl("slice", slice_kernel_cuda);
    m.impl("contiguous", contiguous_kernel_cuda);
    m.impl("select_backward", select_backward_kernel_cuda);
    m.impl("slice_backward", slice_backward_kernel_cuda);
    m.impl("diagonal", diagonal_kernel_cuda);
    m.impl("diagonal_backward", diagonal_backward_kernel_cuda);
    m.impl("movedim", movedim_kernel_cuda);
    m.impl("cat", cat_kernel_cuda);
    m.impl("stack", stack_kernel_cuda);
    m.impl("split", split_kernel_cuda);
    m.impl("split.sizes", split_sizes_kernel_cuda);
    m.impl("chunk", chunk_kernel_cuda);
    m.impl("unbind", unbind_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
