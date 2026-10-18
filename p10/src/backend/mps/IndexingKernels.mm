#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

static auto& lib = MetalShaderLibrary::getBundledLibrary();

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

// Returns a contiguous int64 copy of `index`. Short indices are checked on the
// host; longer ones are checked inside the access kernel itself, which reports
// a bad entry into the stream's error buffer while reading the index for the
// access. The access kernels wrap and clamp, so an entry that is still in
// flight while the report is pending stays inside the storage.
Tensor normalize_index(const Tensor& index, int64_t dim_size) {
  Tensor idx = index;
  if (idx.dtype() != DType::Int64) {
    idx = idx.to(DType::Int64);
  }
  idx = idx.contiguous();
  const int64_t n = idx.numel();
  if (n == 0) {
    return idx;
  }
  constexpr int64_t kHostScanLimit = 4096;
  // The scan reads shared storage from the host, so it is only safe while no
  // work that may write those bytes is in flight. A busy stream leaves the
  // entries to the access kernel, which reports the same error as it reads
  // them; draining the stream here instead would serialize every call in a
  // chain of back-to-back accesses.
  if (n <= kHostScanLimit && !getCurrentMPSStream()->hasPendingWork()) {
    const int64_t* p = idx.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t v = p[i];
      if (v < -dim_size || v >= dim_size) {
        TP_THROW(IndexError, "index ", v, " is out of bounds for dimension ",
                 dim_size);
      }
    }
    return idx;
  }
  return idx;
}

// Reports an out-of-range entry for a result with no elements for the access
// kernel to visit. The report goes onto the stream right before the wait that
// raises it.
void dispatch_bounds_check(const Tensor& idx, int64_t dim_size) {
  const std::string kernel =
      "index_bounds_check_" + scalarToMetalTypeString(idx.dtype());
  auto pso = lib.getPipelineStateForFunc(kernel);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, idx, dim_size, stream->errorBuffer());
      mtl_dispatch1DJob(encoder, pso,
                        static_cast<NSUInteger>(idx.numel()));
    }
  });
}

// Dispatches a dense indexing kernel that reads one element of `input` per
// output element. The grid is (inner, indexed dim, outer), the layout the
// kernels address directly. The kernels also report an out-of-range index
// entry into the stream's error buffer as they read it.
void dispatch_indexing(const std::string& kernel, const Tensor& output,
                       const Tensor& input, const Tensor& index,
                       int64_t inner, int64_t dim_size, int64_t idx_dim,
                       int64_t length) {
  auto pso = lib.getPipelineStateForFunc(kernel);
  const int64_t outer = length / (inner * idx_dim);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, output, input, index, inner, dim_size, idx_dim,
                  stream->errorBuffer());
      mtl_dispatch3DJob(encoder, pso, static_cast<NSUInteger>(inner),
                        static_cast<NSUInteger>(idx_dim),
                        static_cast<NSUInteger>(outer));
    }
  });
}

// Dispatches the byte-wise index_select when the row span and both base
// addresses admit a 16-, 8- or 4-byte unit; returns false when the row has to
// be copied one element at a time.
bool dispatch_index_select_wide(const Tensor& output, const Tensor& input,
                                const Tensor& index, int64_t inner,
                                int64_t dim_size, int64_t idx_dim) {
  const int64_t row_bytes = inner * output.itemsize();
  const uintptr_t src_bits = reinterpret_cast<uintptr_t>(input.data_ptr());
  const uintptr_t dst_bits = reinterpret_cast<uintptr_t>(output.data_ptr());
  const char* suffix = nullptr;
  int64_t unit = 0;
  if (row_bytes % 16 == 0 && ((src_bits | dst_bits) & 15) == 0) {
    unit = 16;
    suffix = "16";
  } else if (row_bytes % 8 == 0 && ((src_bits | dst_bits) & 7) == 0) {
    unit = 8;
    suffix = "8";
  } else if (row_bytes % 4 == 0 && ((src_bits | dst_bits) & 3) == 0) {
    unit = 4;
    suffix = "4";
  }
  if (unit == 0) {
    return false;
  }
  const int64_t units = row_bytes / unit;
  const int64_t outer = output.numel() / (inner * idx_dim);
  constexpr int64_t kMaxGrid =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  if (units > kMaxGrid || idx_dim > kMaxGrid || outer > kMaxGrid) {
    return false;
  }
  const std::string kernel =
      std::string("index_select_vec") + suffix + "_long";
  auto pso = lib.getPipelineStateForFunc(kernel);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, output, input, index, units, dim_size, idx_dim,
                  stream->errorBuffer());
      mtl_dispatch3DJob(encoder, pso, static_cast<NSUInteger>(units),
                        static_cast<NSUInteger>(idx_dim),
                        static_cast<NSUInteger>(outer));
    }
  });
  return true;
}

// Dispatches the scalar-value scatter kernel. `value` is converted to the
// output's dtype on the host so the shader reads the right byte width.
template <typename T>
void dispatch_scatter_value(const std::string& kernel, const Tensor& output,
                            T value, const Tensor& index, int64_t inner,
                            int64_t out_dim, int64_t idx_dim) {
  auto pso = lib.getPipelineStateForFunc(kernel);
  const int64_t outer = index.numel() / (inner * idx_dim);
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, output, value, index, inner, out_dim, idx_dim,
                  stream->errorBuffer());
      mtl_dispatch3DJob(encoder, pso, static_cast<NSUInteger>(inner),
                        static_cast<NSUInteger>(idx_dim),
                        static_cast<NSUInteger>(outer));
    }
  });
}

// Dispatches the elementwise masked fill over two matching-shape tensors.
template <typename T>
void dispatch_masked_fill(const Tensor& out, const Tensor& mask,
                          const Tensor& input, T value) {
  auto pso = lib.getPipelineStateForFunc(
      "masked_fill_scalar_" + scalarToMetalTypeString(out.dtype()));
  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, out, mask, input, value);
      mtl_dispatch1DJob(encoder, pso, static_cast<NSUInteger>(out.numel()));
    }
  });
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
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  // A mask that already has the input's shape needs no broadcast: one thread
  // selects one element.
  if (self.device().is_mps() && mask.device().is_mps() &&
      static_cast<std::vector<int64_t>>(mask.shape()) == self_shape) {
    Tensor self_c = materializeForGraph(self);
    Tensor mask_c = materializeForGraph(mask);
    Tensor result = Tensor(self_shape, self.dtype(), self.device());
    if (result.numel() == 0) {
      return result;
    }
    switch (self.dtype()) {
      case DType::Float32:
        dispatch_masked_fill(result, mask_c, self_c, value.to<float>());
        return result;
      case DType::Float16:
        dispatch_masked_fill(result, mask_c, self_c, value.to<Half>());
        return result;
      case DType::BFloat16:
        dispatch_masked_fill(result, mask_c, self_c, value.to<BFloat16>());
        return result;
      case DType::Int64:
        dispatch_masked_fill(result, mask_c, self_c, value.to<int64_t>());
        return result;
      case DType::Int32:
        dispatch_masked_fill(result, mask_c, self_c, value.to<int32_t>());
        return result;
      case DType::Int16:
        dispatch_masked_fill(result, mask_c, self_c, value.to<int16_t>());
        return result;
      case DType::Int8:
        dispatch_masked_fill(result, mask_c, self_c, value.to<int8_t>());
        return result;
      case DType::UInt8:
        dispatch_masked_fill(result, mask_c, self_c, value.to<uint8_t>());
        return result;
      case DType::Bool:
        dispatch_masked_fill(result, mask_c, self_c, value.to<bool>());
        return result;
      default:
        break;
    }
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
  if (value.dim() != 0) {
    TP_THROW(RuntimeError,
             "masked_fill only supports a 0-dimensional value tensor, but got tensor with ",
             value.dim(), " dimension(s).");
  }
  return masked_fill_kernel(self, mask, value.item());
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

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(self.shape());
  out_shape[dim] = n_idx;
  Tensor result = Tensor::empty(out_shape, self.dtype(), self.device());
  if (result.numel() == 0) {
    return result;
  }
  Tensor self_c = self.contiguous();
  // With the indexed dimension empty every entry is out of bounds; report the
  // first one and raise it before the access could touch the empty storage.
  if (self.numel() == 0) {
    dispatch_bounds_check(idx_c, row);
    synchronizeHostAccess();
    return result;
  }
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < nd; ++i) {
    inner *= self.size(i);
  }
  if (!dispatch_index_select_wide(result, self_c, idx_c, inner, row, n_idx)) {
    const std::string kernel =
        "index_select_dense_" + scalarToMetalTypeString(result.dtype()) +
        "_long";
    dispatch_indexing(kernel, result, self_c, idx_c, inner, row, n_idx,
                      result.numel());
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
  int64_t idx_inner = 1;
  for (int64_t i = dim + 1; i < nd; ++i) {
    idx_inner *= idx_c.size(i);
  }
  const int64_t idx_dim = idx_c.size(dim);
  const int64_t self_dim = self.size(dim);
  const std::string kernel =
      "gather_dense_" + scalarToMetalTypeString(result.dtype()) + "_long";
  dispatch_indexing(kernel, result, self_c, idx_c, idx_inner, self_dim,
                    idx_dim, result.numel());
  return result;
}

// ---------------------------------------------------------------------------
// scatter / scatter_add
// ---------------------------------------------------------------------------

enum class ScatterMode { Assign, Add };

// Applies scatter with source values from `src` (or a constant `value` when
// `src` is undefined) into `result`, walking the index tensor on the GPU.
void scatter_impl(Tensor& result, int64_t dim, const Tensor& index,
                  const Tensor* src, const Scalar* value, ScatterMode mode) {
  const int64_t nd = result.dim();
  dim = wrap_dim(dim, nd);
  Tensor idx_c = normalize_index(index, result.size(dim));
  if (idx_c.numel() == 0) {
    return;
  }
  const int64_t self_dim = result.size(dim);
  // Every entry addresses the empty indexed dimension; report one and raise it
  // before the access could write the empty storage.
  if (result.numel() == 0) {
    dispatch_bounds_check(idx_c, self_dim);
    synchronizeHostAccess();
    return;
  }

  int64_t idx_inner = 1;
  for (int64_t i = dim + 1; i < nd; ++i) {
    idx_inner *= idx_c.size(i);
  }
  const int64_t idx_dim = idx_c.size(dim);
  const std::string suffix = scalarToMetalTypeString(result.dtype());

  if (mode == ScatterMode::Assign && value != nullptr) {
    const std::string kernel = "scatter_set_dense_value_" + suffix + "_long";
    switch (result.dtype()) {
      case DType::Float32:
        dispatch_scatter_value(kernel, result, value->to<float>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Float16:
        dispatch_scatter_value(kernel, result, value->to<Half>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::BFloat16:
        dispatch_scatter_value(kernel, result, value->to<BFloat16>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Int64:
        dispatch_scatter_value(kernel, result, value->to<int64_t>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Int32:
        dispatch_scatter_value(kernel, result, value->to<int32_t>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Int16:
        dispatch_scatter_value(kernel, result, value->to<int16_t>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Int8:
        dispatch_scatter_value(kernel, result, value->to<int8_t>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::UInt8:
        dispatch_scatter_value(kernel, result, value->to<uint8_t>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      case DType::Bool:
        dispatch_scatter_value(kernel, result, value->to<bool>(), idx_c,
                               idx_inner, self_dim, idx_dim);
        return;
      default:
        break;
    }
  } else {
    const std::string kernel = (mode == ScatterMode::Add)
        ? "scatter_add_dense_" + suffix + "_long"
        : "scatter_set_dense_" + suffix + "_long";
    Tensor src_c = src->contiguous();
    dispatch_indexing(kernel, result, src_c, idx_c, idx_inner, self_dim,
                      idx_dim, idx_c.numel());
    return;
  }
  TP_THROW(RuntimeError,
           "scatter: unsupported dtype for the MPS shader: ",
           toString(result.dtype()));
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