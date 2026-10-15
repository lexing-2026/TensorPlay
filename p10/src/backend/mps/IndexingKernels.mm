#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "OperationUtils.h"

#include "backend/mps/MPSDevice.h"

#include <cstring>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

int64_t wrap_dim(int64_t dim, int64_t ndim) {
  if (ndim == 0) return 0;
  int64_t d = dim;
  if (d < 0) d += ndim;
  if (d < 0 || d >= ndim) {
    TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
             -ndim, ", ", ndim - 1, "], but got ", dim, ")");
  }
  return d;
}

// Returns a contiguous int64 copy of `index` with negative indices wrapped
// and every entry bounds-checked against `dim_size`.
Tensor normalize_index(const Tensor& index, int64_t dim_size) {
  Tensor idx = index;
  if (idx.dtype() != DType::Int64) {
    idx = idx.to(DType::Int64);
  }
  idx = idx.contiguous();
  const int64_t n = idx.numel();
  const int64_t* p = idx.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    int64_t v = p[i];
    if (v < 0) v += dim_size;
    if (v < 0 || v >= dim_size) {
      TP_THROW(IndexError, "index ", p[i], " is out of bounds for dimension ",
               dim_size);
    }
  }
  return idx;
}

// Outer/inner element counts around `dim` for a given shape.
void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                 int64_t& outer, int64_t& inner) {
  outer = 1;
  inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
  for (size_t i = dim + 1; i < shape.size(); ++i) inner *= shape[i];
}

double readScalar(const void* ptr, DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return *static_cast<const float*>(ptr);
    case DType::Float64:
      return *static_cast<const double*>(ptr);
    case DType::Float16:
      return static_cast<float>(*static_cast<const Half*>(ptr));
    case DType::BFloat16:
      return static_cast<float>(*static_cast<const BFloat16*>(ptr));
    case DType::Int64:
      return static_cast<double>(*static_cast<const int64_t*>(ptr));
    case DType::Int32:
      return static_cast<double>(*static_cast<const int32_t*>(ptr));
    case DType::Int16:
      return static_cast<double>(*static_cast<const int16_t*>(ptr));
    case DType::Int8:
      return static_cast<double>(*static_cast<const int8_t*>(ptr));
    case DType::UInt8:
      return static_cast<double>(*static_cast<const uint8_t*>(ptr));
    case DType::Bool:
      return *static_cast<const bool*>(ptr) ? 1.0 : 0.0;
    default:
      TP_THROW(TypeError, "scatter: unsupported dtype");
  }
}

void writeScalar(void* ptr, DType dtype, double v) {
  switch (dtype) {
    case DType::Float32:
      *static_cast<float*>(ptr) = static_cast<float>(v);
      break;
    case DType::Float64:
      *static_cast<double*>(ptr) = v;
      break;
    case DType::Float16:
      *static_cast<Half*>(ptr) = Half(static_cast<float>(v));
      break;
    case DType::BFloat16:
      *static_cast<BFloat16*>(ptr) = BFloat16(static_cast<float>(v));
      break;
    case DType::Int64:
      *static_cast<int64_t*>(ptr) = static_cast<int64_t>(v);
      break;
    case DType::Int32:
      *static_cast<int32_t*>(ptr) = static_cast<int32_t>(v);
      break;
    case DType::Int16:
      *static_cast<int16_t*>(ptr) = static_cast<int16_t>(v);
      break;
    case DType::Int8:
      *static_cast<int8_t*>(ptr) = static_cast<int8_t>(v);
      break;
    case DType::UInt8:
      *static_cast<uint8_t*>(ptr) = static_cast<uint8_t>(v);
      break;
    case DType::Bool:
      *static_cast<bool*>(ptr) = (v != 0.0);
      break;
    default:
      TP_THROW(TypeError, "scatter: unsupported dtype");
  }
}

} // namespace

// ---------------------------------------------------------------------------
// masked_fill / masked_fill_
// ---------------------------------------------------------------------------

Tensor masked_fill_kernel(const Tensor& self, const Tensor& mask,
                          const Scalar& value) {
  if (mask.dtype() != DType::Bool) {
    TP_THROW(TypeError, "masked_fill only supports boolean masks");
  }
  Tensor value_t = Tensor::full({}, value, self.dtype(), self.device());
  return whereOp(mask, value_t, self);
}

Tensor& masked_fill__kernel(Tensor& self, const Tensor& mask,
                            const Scalar& value) {
  Tensor r = masked_fill_kernel(self, mask, value);
  self.copy_(r);
  return self;
}

Tensor masked_fill_tensor_kernel(const Tensor& self, const Tensor& mask,
                                 const Tensor& value) {
  if (mask.dtype() != DType::Bool) {
    TP_THROW(TypeError, "masked_fill only supports boolean masks");
  }
  return whereOp(mask, value, self);
}

Tensor& masked_fill_tensor__kernel(Tensor& self, const Tensor& mask,
                                   const Tensor& value) {
  Tensor r = masked_fill_tensor_kernel(self, mask, value);
  self.copy_(r);
  return self;
}

// ---------------------------------------------------------------------------
// index_select
// ---------------------------------------------------------------------------

Tensor index_select_kernel(const Tensor& self, int64_t dim,
                           const Tensor& index) {
  const int64_t nd = self.dim();
  dim = wrap_dim(dim, nd);
  Tensor idx = index;
  if (idx.dim() == 0) {
    idx = idx.reshape({1});
  } else if (idx.dim() != 1) {
    TP_THROW(IndexError, "index_select(): index should be a vector");
  }
  Tensor idx_c = normalize_index(idx, self.size(dim));
  const int64_t n_idx = idx_c.numel();
  const int64_t row = self.size(dim);
  const int64_t* ip = idx_c.data_ptr<int64_t>();
  int64_t outer = 1;
  int64_t inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer,
              inner);

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  out_shape[dim] = n_idx;
  Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
  if (result.numel() == 0) {
    return result;
  }
  Tensor self_c = self.contiguous();
  const int64_t itemsize = result.itemsize();
  const char* s = static_cast<const char*>(self_c.data_ptr());
  char* d = static_cast<char*>(result.data_ptr());
  if (inner == 1) {
    for (int64_t o = 0; o < outer; ++o) {
      const char* src_row = s + o * row * itemsize;
      char* dst_row = d + o * n_idx * itemsize;
      for (int64_t k = 0; k < n_idx; ++k) {
        std::memcpy(dst_row + k * itemsize, src_row + ip[k] * itemsize,
                    itemsize);
      }
    }
  } else {
    const int64_t slice = inner * itemsize;
    for (int64_t t = 0; t < outer * n_idx; ++t) {
      const int64_t o = t / n_idx;
      const int64_t k = t % n_idx;
      std::memcpy(d + (o * n_idx + k) * inner * itemsize,
                  s + (o * row + ip[k]) * inner * itemsize, slice);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

Tensor gather_kernel(const Tensor& self, int64_t dim, const Tensor& index) {
  const int64_t nd = self.dim();
  dim = wrap_dim(dim, nd);
  if (index.dim() != nd) {
    TP_THROW(IndexError,
             "Index must have same number of dimensions as input tensor");
  }
  for (int64_t i = 0; i < nd; ++i) {
    if (i != dim && index.size(i) > self.size(i)) {
      TP_THROW(IndexError, "Size does not match at dimension ", i);
    }
  }
  Tensor idx_c = normalize_index(index, self.size(dim));
  Tensor self_c = self.contiguous();
  Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(idx_c.shape()),
                                self.dtype(), self.device());
  if (result.numel() == 0) {
    return result;
  }
  const int64_t* ip = idx_c.data_ptr<int64_t>();
  const char* s = static_cast<const char*>(self_c.data_ptr());
  char* d = static_cast<char*>(result.data_ptr());
  const int64_t itemsize = result.itemsize();

  int64_t idx_outer = 1;
  int64_t idx_inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(idx_c.shape()), dim, idx_outer,
              idx_inner);
  int64_t self_inner = 1;
  for (int64_t i = dim + 1; i < nd; ++i) {
    self_inner *= self.size(i);
  }
  const int64_t idx_dim = idx_c.size(dim);
  const int64_t self_dim = self.size(dim);

  for (int64_t g = 0; g < idx_outer; ++g) {
    const char* s_row = s + g * self_dim * self_inner * itemsize;
    char* d_row = d + g * idx_dim * idx_inner * itemsize;
    const int64_t* idx_row = ip + g * idx_dim * idx_inner;
    for (int64_t c = 0; c < idx_dim; ++c) {
      for (int64_t t = 0; t < idx_inner; ++t) {
        const int64_t ix = idx_row[c * idx_inner + t];
        std::memcpy(d_row + (c * idx_inner + t) * itemsize,
                    s_row + (ix * self_inner + t) * itemsize, itemsize);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// scatter / scatter_add
// ---------------------------------------------------------------------------

enum class ScatterMode { Assign, Add };

// Applies scatter with source values from `src` (or a constant `value` when
// `src` is undefined) into `result`, walking the index tensor in order.
void scatter_impl(Tensor& result, int64_t dim, const Tensor& index,
                  const Tensor* src, const Scalar* value, ScatterMode mode) {
  const int64_t nd = result.dim();
  dim = wrap_dim(dim, nd);
  Tensor idx_c = normalize_index(index, result.size(dim));
  const int64_t* ip = idx_c.data_ptr<int64_t>();
  const int64_t itemsize = result.itemsize();
  char* d = static_cast<char*>(result.data_ptr());

  int64_t idx_outer = 1;
  int64_t idx_inner = 1;
  outer_inner(static_cast<std::vector<int64_t>>(idx_c.shape()), dim, idx_outer,
              idx_inner);
  int64_t self_inner = 1;
  for (int64_t i = dim + 1; i < nd; ++i) {
    self_inner *= result.size(i);
  }
  const int64_t idx_dim = idx_c.size(dim);
  const int64_t self_dim = result.size(dim);

  const char* src_base = nullptr;
  if (src != nullptr) {
    Tensor src_c = src->contiguous();
    src_base = static_cast<const char*>(src_c.data_ptr());
  }
  double scalar_val = 0.0;
  if (value != nullptr) {
    scalar_val = value->to<double>();
  }

  for (int64_t g = 0; g < idx_outer; ++g) {
    char* d_row = d + g * self_dim * self_inner * itemsize;
    const int64_t* idx_row = ip + g * idx_dim * idx_inner;
    const char* src_row = src_base != nullptr
        ? src_base + g * idx_dim * idx_inner * itemsize
        : nullptr;
    for (int64_t c = 0; c < idx_dim; ++c) {
      for (int64_t t = 0; t < idx_inner; ++t) {
        const int64_t ix = idx_row[c * idx_inner + t];
        char* dst = d_row + (ix * self_inner + t) * itemsize;
        if (mode == ScatterMode::Assign) {
          if (src_base != nullptr) {
            std::memcpy(dst, src_row + (c * idx_inner + t) * itemsize,
                        itemsize);
          } else {
            writeScalar(dst, result.dtype(), scalar_val);
          }
        } else {
          double acc = readScalar(dst, result.dtype()) + readScalar(
              src_row + (c * idx_inner + t) * itemsize, result.dtype());
          writeScalar(dst, result.dtype(), acc);
        }
      }
    }
  }
}

Tensor scatter_src_kernel(const Tensor& self, int64_t dim,
                          const Tensor& index, const Tensor& src) {
  Tensor result = self.clone();
  scatter_impl(result, dim, index, &src, nullptr, ScatterMode::Assign);
  return result;
}

Tensor scatter_value_kernel(const Tensor& self, int64_t dim,
                            const Tensor& index, const Scalar& value) {
  Tensor result = self.clone();
  scatter_impl(result, dim, index, nullptr, &value, ScatterMode::Assign);
  return result;
}

Tensor& scatter_inplace_src_kernel(Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src) {
  if (self.is_contiguous()) {
    scatter_impl(self, dim, index, &src, nullptr, ScatterMode::Assign);
  } else {
    Tensor c = self.clone();
    scatter_impl(c, dim, index, &src, nullptr, ScatterMode::Assign);
    self.copy_(c);
  }
  return self;
}

Tensor& scatter_inplace_value_kernel(Tensor& self, int64_t dim,
                                     const Tensor& index,
                                     const Scalar& value) {
  if (self.is_contiguous()) {
    scatter_impl(self, dim, index, nullptr, &value, ScatterMode::Assign);
  } else {
    Tensor c = self.clone();
    scatter_impl(c, dim, index, nullptr, &value, ScatterMode::Assign);
    self.copy_(c);
  }
  return self;
}

Tensor scatter_add_kernel(const Tensor& self, int64_t dim,
                          const Tensor& index, const Tensor& src) {
  Tensor result = self.clone();
  scatter_impl(result, dim, index, &src, nullptr, ScatterMode::Add);
  return result;
}

TENSORPLAY_LIBRARY_IMPL(MPS, IndexingKernels) {
  m.impl("masked_fill", masked_fill_kernel);
  m.impl("masked_fill_", masked_fill__kernel);
  m.impl("masked_fill.Tensor", masked_fill_tensor_kernel);
  m.impl("masked_fill_.Tensor", masked_fill_tensor__kernel);
  m.impl("index_select", index_select_kernel);
  m.impl("gather", gather_kernel);
  m.impl("scatter.src", scatter_src_kernel);
  m.impl("scatter.value", scatter_value_kernel);
  m.impl("scatter_.src", scatter_inplace_src_kernel);
  m.impl("scatter_.value", scatter_inplace_value_kernel);
  m.impl("scatter_add", scatter_add_kernel);
}

} // namespace mps
} // namespace tensorplay