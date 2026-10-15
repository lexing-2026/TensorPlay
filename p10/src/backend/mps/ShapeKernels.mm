#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <cstring>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

void check_cat_inputs(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    TP_THROW(RuntimeError, "cat(): expected a non-empty list of tensors");
  }
  const Tensor& first = tensors[0];
  const int64_t ndim = first.dim();
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) {
    TP_THROW(IndexError, "Dimension out of range in cat");
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& t = tensors[i];
    if (!t.device().is_mps()) {
      TP_THROW(RuntimeError, "cat(): all tensors must be on the MPS device");
    }
    if (t.dtype() != first.dtype()) {
      TP_THROW(RuntimeError, "cat(): all tensors must have the same dtype");
    }
    if (t.dim() != ndim) {
      TP_THROW(RuntimeError,
               "cat(): tensors must have the same number of dimensions");
    }
    for (int64_t d = 0; d < ndim; ++d) {
      if (d != dim && t.size(d) != first.size(d)) {
        TP_THROW(RuntimeError,
                 "cat(): sizes of tensors must match except in dimension ",
                 dim);
      }
    }
  }
}

// Element count of one slice along `dim` (the product of all other dims).
int64_t slice_numel(const std::vector<int64_t>& shape, int64_t dim) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int64_t>(i) != dim) n *= shape[i];
  }
  return n;
}

} // namespace

Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
  const Tensor& first = tensors[0];
  const int64_t ndim = first.dim();
  if (dim < 0) dim += ndim;
  check_cat_inputs(tensors, dim);

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(first.shape());
  int64_t cat_size = 0;
  for (const Tensor& t : tensors) {
    cat_size += t.size(dim);
  }
  out_shape[dim] = cat_size;

  Tensor output = Tensor(out_shape, first.dtype(), first.device());
  if (output.numel() == 0) {
    return output;
  }

  const int64_t itemsize = output.itemsize();
  const int64_t slice = slice_numel(out_shape, dim);
  int64_t offset = 0;
  for (const Tensor& t : tensors) {
    if (t.size(dim) == 0) continue;
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    const int64_t nbytes = tc.numel() * itemsize;
    std::memcpy(static_cast<char*>(output.data_ptr()) + offset * itemsize,
                tc.data_ptr(), nbytes);
    offset += tc.size(dim) * slice;
  }
  return output;
}

Tensor stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
  if (tensors.empty()) {
    TP_THROW(RuntimeError, "stack(): expected a non-empty list of tensors");
  }
  const Tensor& first = tensors[0];
  const int64_t ndim = first.dim();
  if (dim < 0) dim += ndim + 1;
  if (dim < 0 || dim > ndim) {
    TP_THROW(IndexError, "Dimension out of range in stack");
  }
  for (const Tensor& t : tensors) {
    if (!t.device().is_mps()) {
      TP_THROW(RuntimeError, "stack(): all tensors must be on the MPS device");
    }
    if (t.dtype() != first.dtype()) {
      TP_THROW(RuntimeError, "stack(): all tensors must have the same dtype");
    }
    if (t.dim() != ndim ||
        static_cast<std::vector<int64_t>>(t.shape()) !=
            static_cast<std::vector<int64_t>>(first.shape())) {
      TP_THROW(RuntimeError,
               "stack(): all tensors must have the same shape");
    }
  }

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(first.shape());
  out_shape.insert(out_shape.begin() + dim, static_cast<int64_t>(tensors.size()));

  Tensor output = Tensor(out_shape, first.dtype(), first.device());
  if (output.numel() == 0) {
    return output;
  }

  const int64_t itemsize = output.itemsize();
  int64_t offset = 0;
  for (const Tensor& t : tensors) {
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    const int64_t nbytes = tc.numel() * itemsize;
    std::memcpy(static_cast<char*>(output.data_ptr()) + offset * itemsize,
                tc.data_ptr(), nbytes);
    offset += tc.numel();
  }
  return output;
}

Tensor embedding_kernel(const Tensor& weight, const Tensor& indices,
                        int64_t padding_idx, bool scale_grad_by_freq,
                        bool sparse) {
  (void)padding_idx;
  (void)scale_grad_by_freq;
  (void)sparse;
  if (!weight.device().is_mps() || !indices.device().is_mps()) {
    TP_THROW(RuntimeError, "embedding(): tensors must be on the MPS device");
  }
  if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
    TP_THROW(RuntimeError,
             "embedding(): indices must be int32 or int64");
  }
  Tensor indices_c = indices.is_contiguous() ? indices : indices.contiguous();
  Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

  std::vector<int64_t> out_shape =
      static_cast<std::vector<int64_t>>(indices.shape());
  out_shape.push_back(weight.size(1));
  Tensor output = Tensor(out_shape, weight.dtype(), weight.device());
  if (output.numel() == 0) {
    return output;
  }

  const int64_t emb_dim = weight.size(1);
  const int64_t num_embeddings = weight.size(0);
  const int64_t n = indices_c.numel();

  // Range-check the indices on the host (unified memory makes the buffer
  // host-readable); the shader assumes valid indices afterwards.
  if (indices_c.dtype() == DType::Int64) {
    const int64_t* idx = indices_c.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t ix = idx[i] < 0 ? idx[i] + num_embeddings : idx[i];
      if (ix < 0 || ix >= num_embeddings) {
        TP_THROW(IndexError, "embedding(): index out of range in self");
      }
    }
  } else {
    const int32_t* idx = indices_c.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t ix = idx[i] < 0 ? static_cast<int64_t>(idx[i]) + num_embeddings
                                    : static_cast<int64_t>(idx[i]);
      if (ix < 0 || ix >= num_embeddings) {
        TP_THROW(IndexError, "embedding(): index out of range in self");
      }
    }
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      const std::string kernel = "embedding_forward_" +
          scalarToMetalTypeString(weight_c.dtype()) + "_" +
          scalarToMetalTypeString(indices_c.dtype());
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, weight_c, indices_c, output,
                  static_cast<uint32_t>(num_embeddings),
                  static_cast<uint32_t>(emb_dim),
                  static_cast<uint32_t>(n));
      mtl_dispatch1DJob(computeEncoder, PSO, static_cast<NSUInteger>(n));
      mpsStream->endKernelCoalescing();
    }
  });
  return output;
}

TENSORPLAY_LIBRARY_IMPL(MPS, ShapeKernels) {
  m.impl("cat", cat_kernel);
  m.impl("stack", stack_kernel);
  m.impl("embedding", embedding_kernel);
}

} // namespace mps
} // namespace tensorplay