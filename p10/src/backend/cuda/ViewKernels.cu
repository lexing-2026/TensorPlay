#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
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

Tensor reshape_kernel_cuda(const Tensor& self, const std::vector<int64_t>& shape) {
    // Torch parity (TensorShape.cpp reshape): the result aliases `self`
    // whenever the layout admits the view (computeStride), otherwise it is a
    // contiguous copy.  infer_size throws torch's exact errors (including
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
    // torch reshape_symint fallback: _unsafe_view(clone(Contiguous), shape).
    // The clone must be explicitly contiguous: clone() with Preserve keeps
    // non-overlapping-and-dense strides (e.g. transposed), which the
    // subsequent view would reject.
    return self.clone(static_cast<int64_t>(MemoryFormat::Contiguous)).view(inferred);
}

Tensor transpose_kernel_cuda(const Tensor& self, int64_t dim0, int64_t dim1) {
    int64_t ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
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
    int64_t ndim = self.dim();
    if (dims.size() != (size_t)ndim) {
        TP_THROW(RuntimeError, "permute: number of dimensions mismatch");
    }
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(ndim, false);
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) TP_THROW(IndexError, "permute: dimension out of range");
        if (seen[d]) TP_THROW(RuntimeError, "permute: duplicate dimension");
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
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    if (self.size(dim) != 1) {
        return self;
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

Tensor unsqueeze_kernel_cuda(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < -(ndim + 1) || dim > ndim) {
         TP_THROW(IndexError, "Dimension out of range");
    }
    if (dim < 0) dim += (ndim + 1);
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> new_strides = self.strides();
    
    new_sizes.insert(new_sizes.begin() + dim, 1);
    int64_t stride = 1;
    if (dim < ndim) {
        stride = new_strides[dim]; 
    }
    new_strides.insert(new_strides.begin() + dim, stride);
    
    return self.as_strided(new_sizes, new_strides);
}

// Tensor-list view operators need an explicit CUDA registration.  The actual
// copies are delegated to copy_ so they inherit the stream-aware CUDA allocator
// and non-blocking copy semantics; this keeps the implementation correct for
// non-contiguous inputs while avoiding a second bespoke concatenation kernel.
Tensor cat_kernel_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "cat(): expected a non-empty list of tensors");
    }

    const Tensor& t0 = tensors[0];
    int64_t ndim = t0.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "cat(): dimension out of range");

    int64_t cat_dim_size = 0;
    for (const auto& t : tensors) {
        if (t.device() != t0.device()) {
            TP_THROW(DeviceMismatchError, "cat(): all tensors must be on the same device");
        }
        if (t.dim() != ndim) {
            TP_THROW(RuntimeError, "cat(): all tensors must have same number of dimensions");
        }
        if (t.dtype() != t0.dtype()) {
            TP_THROW(TypeError, "cat(): all tensors must have same dtype (type promotion not impl)");
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && t.size(i) != t0.size(i)) {
                TP_THROW(RuntimeError, "cat(): Sizes of tensors must match except in dimension " + std::to_string(dim));
            }
        }
        cat_dim_size += t.size(dim);
    }

    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(t0.shape());
    out_shape[dim] = cat_dim_size;
    Tensor out = Tensor::empty(out_shape, t0.dtype(), t0.device());

    int64_t offset = 0;
    for (const auto& t : tensors) {
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
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "split(): dimension out of range");
    if (split_size <= 0) TP_THROW(RuntimeError, "split(): split_size must be positive");

    int64_t dim_size = self.size(dim);
    std::vector<Tensor> result;
    for (int64_t i = 0; i < dim_size; i += split_size) {
        int64_t end = std::min(i + split_size, dim_size);
        result.push_back(self.slice(dim, i, end));
    }
    return result;
}

std::vector<Tensor> split_sizes_kernel_cuda(const Tensor& self,
                                            const std::vector<int64_t>& split_sizes,
                                            int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "split(): dimension out of range");

    int64_t dim_size = self.size(dim);
    int64_t sum_sizes = 0;
    for (auto s : split_sizes) sum_sizes += s;
    if (sum_sizes != dim_size) {
        TP_THROW(RuntimeError, "split(): sum of split_sizes must equal dimension size");
    }

    std::vector<Tensor> result;
    int64_t offset = 0;
    for (auto s : split_sizes) {
        result.push_back(self.slice(dim, offset, offset + s));
        offset += s;
    }
    return result;
}

std::vector<Tensor> chunk_kernel_cuda(const Tensor& self, int64_t chunks, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "chunk(): dimension out of range");
    if (chunks <= 0) TP_THROW(RuntimeError, "chunk(): chunks must be positive");

    int64_t dim_size = self.size(dim);
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    std::vector<Tensor> result;
    for (int64_t i = 0; i < dim_size; i += split_size) {
        int64_t end = std::min(i + split_size, dim_size);
        result.push_back(self.slice(dim, i, end));
    }
    return result;
}

std::vector<Tensor> unbind_kernel_cuda(const Tensor& self, int64_t dim) {
    int64_t d = dim < 0 ? dim + self.dim() : dim;
    if (d < 0 || d >= self.dim()) TP_THROW(IndexError, "Dimension out of range");
    std::vector<Tensor> result;
    int64_t size_dim = self.size(d);
    result.reserve(size_dim);
    for (int64_t i = 0; i < size_dim; ++i) {
        result.push_back(self.select(d, i));
    }
    return result;
}

Tensor stack_kernel_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "stack(): expected a non-empty list of tensors");
    }
    int64_t ndim = tensors[0].dim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) TP_THROW(IndexError, "stack(): dimension out of range");

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

// ATen semantics: remove dim1/dim2 and append the diagonal axis at the end.
// Pure metadata op: identical to the CPU kernel, safe on any device.
Tensor diagonal_kernel_cuda(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2) {
    const int64_t ndim = self.dim();
    if (ndim < 2) TP_THROW(RuntimeError, "diagonal(): input must be at least 2-dimensional");
    if (dim1 < 0) dim1 += ndim;
    if (dim2 < 0) dim2 += ndim;
    if (dim1 < 0 || dim1 >= ndim || dim2 < 0 || dim2 >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    if (dim1 == dim2) TP_THROW(RuntimeError, "diagonal(): dim1 and dim2 cannot be equal");

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
        new_offset += offset * stride2;
    } else {
        diag_size = std::max<int64_t>(std::min(size1 + offset, size2), 0);
        new_offset -= offset * stride1;
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
    const int64_t ndim = self.dim();
    if (source.size() != destination.size()) {
        TP_THROW(RuntimeError, "movedim: Source and destination dims must have same number of elements");
    }
    std::vector<int64_t> src(source), dst(destination);
    std::vector<bool> src_seen(ndim, false), dst_seen(ndim, false);
    for (auto& d : src) {
        const int64_t orig = d;
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            TP_THROW(IndexError, "movedim: Tried to move to index ", orig,
                     ", but the tensor has ", ndim, " dimensions");
        }
        if (src_seen[d]) TP_THROW(RuntimeError, "movedim: repeated source dimension");
        src_seen[d] = true;
    }
    for (auto& d : dst) {
        const int64_t orig = d;
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            TP_THROW(IndexError, "movedim: Tried to move to index ", orig,
                     ", but the tensor has ", ndim, " dimensions");
        }
        if (dst_seen[d]) TP_THROW(RuntimeError, "movedim: repeated destination dimension");
        dst_seen[d] = true;
    }

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
