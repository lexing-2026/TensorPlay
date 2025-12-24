#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace tensorplay {
namespace cuda {

Tensor reshape_kernel_cuda(const Tensor& self, const std::vector<int64_t>& shape) {
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

TENSORPLAY_LIBRARY_IMPL(CUDA, ViewKernels) {
    m.impl("reshape", reshape_kernel_cuda);
    m.impl("transpose", transpose_kernel_cuda);
    m.impl("t", t_kernel_cuda);
    m.impl("permute", permute_kernel_cuda);
    m.impl("permute_backward", permute_backward_kernel_cuda);
    m.impl("squeeze", squeeze_kernel_cuda);
    m.impl("squeeze_backward", squeeze_backward_kernel_cuda);
    m.impl("squeeze.dim", squeeze_dim_kernel_cuda);
    m.impl("unsqueeze", unsqueeze_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
