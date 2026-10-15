#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "SizesAndStrides.h"

#include "backend/mps/MPSDevice.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

int64_t wrap_dim_scalar(int64_t dim, int64_t ndim) {
  if (ndim == 0) return 0;
  int64_t d = dim;
  if (d < 0) d += ndim;
  if (d < 0 || d >= ndim) {
    TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
             -ndim, ", ", ndim - 1, "], but got ", dim, ")");
  }
  return d;
}

bool diagonal_mul_overflow(int64_t lhs, int64_t rhs) {
  if (lhs == 0 || rhs == 0) return false;
  if ((lhs == std::numeric_limits<int64_t>::min() && rhs == -1) ||
      (rhs == std::numeric_limits<int64_t>::min() && lhs == -1)) {
    return true;
  }
  if (lhs > 0) {
    return rhs > 0
        ? lhs > std::numeric_limits<int64_t>::max() / rhs
        : rhs < std::numeric_limits<int64_t>::min() / lhs;
  }
  return rhs > 0
      ? lhs < std::numeric_limits<int64_t>::min() / rhs
      : lhs < std::numeric_limits<int64_t>::max() / rhs;
}

int64_t diagonal_add_checked(int64_t lhs, int64_t rhs) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    TP_THROW(ValueError, "diagonal storage offset overflow");
  }
  return lhs + rhs;
}

} // namespace

// ---------------------------------------------------------------------------
// View / metadata operations.  These only reshape the tensor metadata and
// share the underlying MPS storage; no GPU work is involved.
// ---------------------------------------------------------------------------

Tensor as_strided_kernel(const Tensor& self, const std::vector<int64_t>& size,
                         const std::vector<int64_t>& stride,
                         std::optional<int64_t> storage_offset) {
  return self.as_strided(size, stride, storage_offset);
}

Tensor transpose_kernel(const Tensor& self, int64_t dim0, int64_t dim1) {
  const int64_t ndim = self.dim();
  dim0 = wrap_dim_scalar(dim0, ndim);
  dim1 = wrap_dim_scalar(dim1, ndim);
  if (dim0 == dim1) {
    return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                           self.strides());
  }
  std::vector<int64_t> new_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  std::vector<int64_t> new_strides = self.strides();
  std::swap(new_sizes[dim0], new_sizes[dim1]);
  std::swap(new_strides[dim0], new_strides[dim1]);
  return self.as_strided(new_sizes, new_strides);
}

Tensor t_kernel(const Tensor& self) {
  if (self.dim() > 2) {
    TP_THROW(RuntimeError, "t() expects a tensor with <= 2 dimensions, but "
             "self is ", self.dim(), "D");
  }
  if (self.dim() < 2) return self;
  return transpose_kernel(self, 0, 1);
}

Tensor permute_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
  const int64_t ndim = self.dim();
  if (dims.size() != static_cast<size_t>(ndim)) {
    TP_THROW(RuntimeError,
             "permute(): number of dimensions in the tensor input does not "
             "match the length of the desired ordering of dimensions, i.e. "
             "input.dim() = ",
             ndim, " is not equal to len(dims) = ", dims.size());
  }
  std::vector<int64_t> new_sizes(ndim);
  std::vector<int64_t> new_strides(ndim);
  std::vector<bool> seen(ndim, false);
  for (int64_t i = 0; i < ndim; ++i) {
    const int64_t d = wrap_dim_scalar(dims[i], ndim);
    if (seen[d]) {
      TP_THROW(RuntimeError, "permute(): duplicate dims are not allowed.");
    }
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
  const int64_t ndim = self.dim();
  dim = wrap_dim_scalar(dim, ndim);
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

Tensor squeeze_dims_kernel(const Tensor& self,
                           const std::vector<int64_t>& dims) {
  const int64_t ndim = self.dim();
  std::vector<bool> seen(ndim > 0 ? ndim : 1, false);
  std::vector<bool> mask(ndim, false);
  for (int64_t d : dims) {
    const int64_t w = wrap_dim_scalar(d, ndim);
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

Tensor unsqueeze_kernel(const Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  dim = wrap_dim_scalar(dim, ndim + 1);
  std::vector<int64_t> new_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  std::vector<int64_t> new_strides = self.strides();
  int64_t new_stride = 1;
  if (dim != ndim) {
    new_stride = self.size(dim) * self.stride(dim);
  }
  new_sizes.insert(new_sizes.begin() + dim, 1);
  new_strides.insert(new_strides.begin() + dim, new_stride);
  return self.as_strided(new_sizes, new_strides);
}

Tensor clone_kernel(const Tensor& self, std::optional<int64_t> memory_format) {
  (void)memory_format;
  Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                             self.dtype(), self.device());
  out.copy_(self);
  return out;
}

Tensor contiguous_kernel(const Tensor& self, int64_t memory_format) {
  (void)memory_format;
  if (self.is_contiguous()) {
    return self;
  }
  return clone_kernel(self, std::nullopt);
}

Tensor slice_kernel(const Tensor& self, int64_t dim,
                    std::optional<int64_t> start, std::optional<int64_t> end,
                    int64_t step) {
  const int64_t ndim = self.dim();
  dim = wrap_dim_scalar(dim, ndim);
  return self.slice(dim, start.value_or(0),
                    end.value_or(std::numeric_limits<int64_t>::max()), step);
}

Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
  if (self.is_sparse()) {
    TP_THROW(RuntimeError, "reshape is not implemented for sparse tensors");
  }
  std::vector<int64_t> inferred =
      SizesAndStrides::infer_size(shape, self.numel());
  auto stride = SizesAndStrides::compute_view_strides(
      static_cast<std::vector<int64_t>>(self.shape()), self.strides(),
      inferred);
  if (stride.has_value()) {
    return self.as_strided(inferred, *stride);
  }
  return clone_kernel(self, std::nullopt).view(inferred);
}

std::vector<Tensor> split_kernel(const Tensor& self, int64_t split_size,
                                int64_t dim) {
  if (self.dim() == 0) {
    TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
  }
  if (split_size < 0) {
    TP_THROW(RuntimeError, "split expects split_size be non-negative, but got "
             "split_size=", split_size);
  }
  dim = wrap_dim_scalar(dim, self.dim());
  const int64_t dim_size = self.size(dim);
  if (!(split_size > 0 || dim_size == 0)) {
    TP_THROW(RuntimeError, "split_size can only be 0 if dimension size is 0, "
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
  result.reserve(num_splits);
  for (int64_t i = 0; i < num_splits; ++i) {
    const int64_t length =
        i < num_splits - 1 ? split_size : last_split_size;
    result.push_back(self.slice(dim, i * split_size, i * split_size + length));
  }
  return result;
}

std::vector<Tensor> split_sizes_kernel(
    const Tensor& self, const std::vector<int64_t>& split_sizes,
    int64_t dim) {
  if (self.dim() == 0) {
    TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
  }
  dim = wrap_dim_scalar(dim, self.dim());
  const int64_t dim_size = self.size(dim);
  std::vector<Tensor> result;
  result.reserve(split_sizes.size());
  int64_t start_idx = 0;
  for (const int64_t length : split_sizes) {
    if (length < 0) {
      TP_THROW(RuntimeError,
               "split_with_sizes expects split_sizes have only non-negative "
               "entries");
    }
    if (length > dim_size - start_idx) {
      TP_THROW(RuntimeError,
               "split_with_sizes expects split_sizes to sum exactly to ",
               dim_size, " (input tensor's size at dimension ", dim, ")");
    }
    result.push_back(self.slice(dim, start_idx, start_idx + length));
    start_idx += length;
  }
  if (start_idx != dim_size) {
    TP_THROW(RuntimeError,
             "split_with_sizes expects split_sizes to sum exactly to ",
             dim_size, " (input tensor's size at dimension ", dim, ")");
  }
  return result;
}

std::vector<Tensor> chunk_kernel(const Tensor& self, int64_t chunks,
                                 int64_t dim) {
  if (self.dim() == 0) {
    TP_THROW(RuntimeError, "chunk expects at least a 1-dimensional tensor");
  }
  if (chunks <= 0) {
    TP_THROW(RuntimeError, "chunk expects `chunks` to be greater than 0, got: ",
             chunks);
  }
  dim = wrap_dim_scalar(dim, self.dim());
  const int64_t dim_size = self.size(dim);
  const int64_t split_size =
      dim_size / chunks + (dim_size % chunks != 0 ? 1 : 0);
  if (split_size == 0 && dim_size == 0) {
    std::vector<int64_t> split_sizes(static_cast<size_t>(chunks), 0);
    split_sizes[static_cast<size_t>(chunks - 1)] =
        split_size - (split_size * chunks - dim_size);
    return split_sizes_kernel(self, split_sizes, dim);
  }
  return split_kernel(self, split_size, dim);
}

Tensor diagonal_kernel(const Tensor& self, int64_t offset, int64_t dim1,
                       int64_t dim2) {
  const int64_t ndim = self.dim();
  const int64_t dim1_ = dim1, dim2_ = dim2;
  dim1 = wrap_dim_scalar(dim1, ndim);
  dim2 = wrap_dim_scalar(dim2, ndim);
  if (dim1 == dim2) {
    TP_THROW(RuntimeError, "diagonal dimensions cannot be identical ", dim1_,
             ", ", dim2_);
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

  int64_t diag_size = 0;
  int64_t new_offset =
      static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
  if (offset >= 0) {
    diag_size = offset < size2 ? std::min(size1, size2 - offset) : 0;
  } else if (offset == std::numeric_limits<int64_t>::min()) {
    diag_size = 0;
  } else {
    const int64_t offset_abs = -offset;
    diag_size = offset_abs < size1 ? std::min(size1 - offset_abs, size2) : 0;
  }
  if (diag_size != 0) {
    if (offset >= 0) {
      if (diagonal_mul_overflow(offset, stride2)) {
        TP_THROW(ValueError, "diagonal storage offset overflow");
      }
      new_offset = diagonal_add_checked(new_offset, offset * stride2);
    } else {
      const int64_t offset_abs = -offset;
      if (diagonal_mul_overflow(offset_abs, stride1)) {
        TP_THROW(ValueError, "diagonal storage offset overflow");
      }
      new_offset = diagonal_add_checked(new_offset, offset_abs * stride1);
    }
  }
  sizes.push_back(diag_size);
  strides.push_back(stride1 + stride2);
  return self.as_strided(sizes, strides, new_offset);
}

Tensor movedim_kernel(const Tensor& self,
                      const std::vector<int64_t>& source,
                      const std::vector<int64_t>& destination) {
  const int64_t ndim = self.dim();
  if (source.size() != destination.size()) {
    TP_THROW(RuntimeError, "movedim: Invalid source or destination dims");
  }
  std::vector<int64_t> src(source), dst(destination);
  for (auto& d : src) d = wrap_dim_scalar(d, ndim);
  for (auto& d : dst) d = wrap_dim_scalar(d, ndim);
  auto all_unique = [](std::vector<int64_t> dims) {
    std::sort(dims.begin(), dims.end());
    return std::adjacent_find(dims.begin(), dims.end()) == dims.end();
  };
  if (!all_unique(src) || !all_unique(dst)) {
    TP_THROW(RuntimeError, "movedim: repeated dim in source or destination");
  }
  if (ndim == 0) {
    return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                           self.strides());
  }
  std::vector<bool> src_seen(ndim, false), dst_seen(ndim, false);
  for (const int64_t d : src) src_seen[d] = true;
  for (const int64_t d : dst) dst_seen[d] = true;
  std::vector<int64_t> permutation(ndim, -1);
  for (size_t k = 0; k < src.size(); ++k) {
    permutation[dst[k]] = src[k];
  }
  int64_t cursor = 0;
  for (int64_t i = 0; i < ndim; ++i) {
    if (!dst_seen[i]) {
      while (src_seen[cursor]) ++cursor;
      permutation[i] = cursor++;
    }
  }
  return permute_kernel(self, permutation);
}

std::vector<Tensor> unbind_kernel(const Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  if (ndim == 0) {
    TP_THROW(IndexError, "Dimension specified as ", dim,
             " but tensor has no dimensions");
  }
  dim = wrap_dim_scalar(dim, ndim);
  std::vector<Tensor> result;
  result.reserve(self.size(dim));
  for (int64_t i = 0; i < self.size(dim); ++i) {
    result.push_back(self.select(dim, i));
  }
  return result;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ViewKernels) {
  m.impl("as_strided", as_strided_kernel);
  m.impl("transpose", transpose_kernel);
  m.impl("t", t_kernel);
  m.impl("permute", permute_kernel);
  m.impl("squeeze", squeeze_kernel);
  m.impl("squeeze.dim", squeeze_dim_kernel);
  m.impl("squeeze.dims", squeeze_dims_kernel);
  m.impl("unsqueeze", unsqueeze_kernel);
  m.impl("clone", clone_kernel);
  m.impl("contiguous", contiguous_kernel);
  m.impl("slice.Tensor", slice_kernel);
  m.impl("reshape", reshape_kernel);
  m.impl("split", split_kernel);
  m.impl("split.sizes", split_sizes_kernel);
  m.impl("chunk", chunk_kernel);
  m.impl("diagonal", diagonal_kernel);
  m.impl("movedim", movedim_kernel);
  m.impl("unbind", unbind_kernel);
}

} // namespace mps
} // namespace tensorplay