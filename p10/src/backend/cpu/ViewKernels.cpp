#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace tensorplay {
namespace cpu {

Tensor transpose_kernel(const Tensor& self, int64_t dim0, int64_t dim1) {
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


Tensor t_kernel(const Tensor& self) {
    // std::cout << "DEBUG: t_kernel called with dim=" << self.dim() << std::endl;
    if (self.dim() > 2) {
        TP_THROW(RuntimeError, "t() expects a tensor with <= 2 dimensions, but self is " + std::to_string(self.dim()) + "D");
    }
    if (self.dim() < 2) return self;
    return transpose_kernel(self, 0, 1);
}


Tensor permute_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
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


Tensor squeeze_kernel(const Tensor& self) {
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


Tensor squeeze_dim_kernel(const Tensor& self, int64_t dim) {
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


Tensor unsqueeze_kernel(const Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < -(ndim + 1) || dim > ndim) {
         TP_THROW(IndexError, "Dimension out of range");
    }
    if (dim < 0) dim += (ndim + 1);
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> new_strides = self.strides();
    
    new_sizes.insert(new_sizes.begin() + dim, 1);
    // Use stride of next dimension or 1
    int64_t stride = 1;
    // For unsqueeze, the stride of the new dimension (size 1) doesn't strictly matter for addressing,
    // but using the stride of the next dimension (or 1 if last) is conventional.
    if (dim < ndim) {
        stride = new_strides[dim]; // Note: new_strides already has size ndim, but we inserted into new_sizes.
                                   // new_strides is still size ndim here. 'dim' index refers to old stride at that pos.
                                   // Wait, new_strides is not inserted yet.
                                   // So self.stride(dim) corresponds to the dimension AFTER the inserted one (in the new tensor).
    }
    // Actually simpler:
    new_strides.insert(new_strides.begin() + dim, stride);
    
    return self.as_strided(new_sizes, new_strides);
}


// --- Joining Ops ---

Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "cat(): expected a non-empty list of tensors");
    }
    
    const Tensor& t0 = tensors[0];
    int64_t ndim = t0.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "cat(): dimension out of range");
    
    // Validate shapes and dtype
    int64_t cat_dim_size = 0;
    for (const auto& t : tensors) {
        if (t.dim() != ndim) TP_THROW(RuntimeError, "cat(): all tensors must have same number of dimensions");
        if (t.dtype() != t0.dtype()) TP_THROW(TypeError, "cat(): all tensors must have same dtype (type promotion not impl)");
        
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
    
    // Copy data
    int64_t offset = 0;
    for (const auto& t : tensors) {
        int64_t size = t.size(dim);
        if (size > 0) {
            Tensor out_slice = out.slice(dim, offset, offset + size);
            out_slice.copy_(t);
            offset += size;
        }
    }
    
    return out;
}


Tensor stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        TP_THROW(RuntimeError, "stack(): expected a non-empty list of tensors");
    }
    int64_t ndim = tensors[0].dim();
    if (dim < 0) dim += (ndim + 1);
    if (dim < 0 || dim > ndim) TP_THROW(IndexError, "stack(): dimension out of range");
    
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for(const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return cat_kernel(unsqueezed, dim);
}


// --- Splitting Ops ---

std::vector<Tensor> split_kernel(const Tensor& self, int64_t split_size, int64_t dim) {
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


std::vector<Tensor> split_sizes_kernel(const Tensor& self, const std::vector<int64_t>& split_sizes, int64_t dim) {
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


std::vector<Tensor> chunk_kernel(const Tensor& self, int64_t chunks, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "chunk(): dimension out of range");
    
    if (chunks <= 0) TP_THROW(RuntimeError, "chunk(): chunks must be positive");
    
    int64_t dim_size = self.size(dim);
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    
    return split_kernel(self, split_size, dim);
}


Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
    // Check if new shape is compatible with number of elements
    int64_t new_numel = 1;
    int64_t infer_dim = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer_dim != -1) TP_THROW(RuntimeError, "only one dimension can be inferred");
            infer_dim = i;
        } else if (shape[i] < 0) {
            TP_THROW(RuntimeError, "invalid shape dimension " + std::to_string(shape[i]));
        } else {
            new_numel *= shape[i];
        }
    }
    
    if (self.numel() == new_numel && infer_dim == -1) {
        // Exact match
    } else if (infer_dim != -1) {
        int64_t missing = self.numel() / new_numel;
        if (self.numel() % new_numel != 0) {
             TP_THROW(RuntimeError, "shape '" + Size(shape).toString() + "' is invalid for input of size " + std::to_string(self.numel()));
        }
        // Modify shape const reference? No, need copy.
        std::vector<int64_t> mutable_shape = shape;
        mutable_shape[infer_dim] = missing;
        if (self.is_contiguous()) return self.view(mutable_shape);
        return self.clone().view(mutable_shape);
    } else {
         TP_THROW(RuntimeError, "shape '" + Size(shape).toString() + "' is invalid for input of size " + std::to_string(self.numel()));
    }

    if (self.is_contiguous()) {
         return self.view(shape);
    }
    return self.clone().view(shape);
}


std::vector<Tensor> unbind_kernel(const Tensor& self, int64_t dim) {
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

Tensor permute_backward_kernel(const Tensor& grad, const Tensor& self, const std::vector<int64_t>& dims) {
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

Tensor squeeze_backward_kernel(const Tensor& grad, const Tensor& self) {
    return grad.reshape(static_cast<std::vector<int64_t>>(self.shape()));
}


TENSORPLAY_LIBRARY_IMPL(CPU, ViewKernels) {
    m.impl("transpose", transpose_kernel);
    m.impl("t", t_kernel);
    m.impl("permute", permute_kernel);
    m.impl("permute_backward", permute_backward_kernel);
    m.impl("squeeze", squeeze_kernel);
    m.impl("squeeze_backward", squeeze_backward_kernel);
    m.impl("squeeze.dim", squeeze_dim_kernel);
    m.impl("unsqueeze", unsqueeze_kernel);
    m.impl("cat", cat_kernel);
    m.impl("stack", stack_kernel);
    m.impl("split", split_kernel);
    m.impl("split.sizes", split_sizes_kernel);
    m.impl("chunk", chunk_kernel);
    m.impl("reshape", reshape_kernel);
    m.impl("unbind", unbind_kernel);
}

} // namespace cpu
} // namespace tensorplay
