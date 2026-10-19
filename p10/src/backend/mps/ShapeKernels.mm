#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/kernels/Copy.h"

#include <algorithm>
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

// Promoted storage dtype for a list of tensors being joined.
DType cat_promote_dtype(const std::vector<Tensor>& tensors) {
  DType out = tensors[0].dtype();
  for (size_t i = 1; i < tensors.size(); ++i) {
    out = promoteTypes(out, tensors[i].dtype());
  }
  return out;
}

// Element count of one slice along `dim` (the product of all other dims).
int64_t slice_numel(const std::vector<int64_t>& shape, int64_t dim) {
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int64_t>(i) != dim) n *= shape[i];
  }
  return n;
}

using BlockCopyPipeline = id<MTLComputePipelineState>;

// Encodes `input`'s bytes into the output blocks described by the geometry.
// The launch is chunked so the byte length fits its 32-bit field.
void encode_block_copy(id<MTLComputeCommandEncoder> encoder,
                       BlockCopyPipeline pso, const Tensor& input,
                       const Tensor& output, int64_t slice_bytes,
                       int64_t out_stride_bytes, int64_t off_bytes) {
  const int64_t nbytes = input.numel() * input.itemsize();
  if (nbytes == 0) {
    return;
  }
  constexpr int64_t kChunkBytes = int64_t{1} << 30;
  for (int64_t base = 0; base < nbytes; base += kChunkBytes) {
    StridedBlockCopyParams params{};
    params.base = static_cast<uint64_t>(base);
    params.slice_bytes = static_cast<uint64_t>(slice_bytes);
    params.out_stride_bytes = static_cast<uint64_t>(out_stride_bytes);
    params.off_bytes = static_cast<uint64_t>(off_bytes);
    params.nbytes = static_cast<uint32_t>(
        std::min<int64_t>(kChunkBytes, nbytes - base));
    params.flat = slice_bytes >= nbytes ? 1u : 0u;
    mtl_setArgs(encoder, input, output, params);
    mtl_dispatch1DJob(encoder, pso,
                      (static_cast<NSUInteger>(params.nbytes) + 15) / 16);
  }
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

  const DType out_dtype = cat_promote_dtype(tensors);
  Tensor output = Tensor(out_shape, out_dtype, first.device());
  if (output.numel() == 0) {
    return output;
  }

  // The inputs are copied on the GPU into their slots of the output: each is
  // one contiguous run of `size(dim)` slices, and consecutive slices sit
  // `output.size(dim)` apart.
  std::vector<Tensor> prepared;
  prepared.reserve(tensors.size());
  for (const Tensor& t : tensors) {
    if (t.size(dim) == 0) continue;
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    if (tc.dtype() != out_dtype) {
      tc = tc.to(out_dtype);
    }
    tc = materializeForGraph(tc);
    prepared.push_back(tc);
  }

  const int64_t itemsize = output.itemsize();
  const int64_t inner_bytes = slice_numel(out_shape, dim) * itemsize;
  const int64_t out_stride_bytes = output.size(dim) * inner_bytes;
  const BlockCopyPipeline pso =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
          "strided_block_copy");
  MPSStream* stream = getCurrentMPSStream();
  run_serialized(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      int64_t offset = 0;
      for (const Tensor& tc : prepared) {
        const int64_t slice_bytes = tc.size(dim) * inner_bytes;
        encode_block_copy(encoder, pso, tc, output, slice_bytes,
                          out_stride_bytes, offset * inner_bytes);
        offset += tc.size(dim);
      }
    }
  });
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

  const DType out_dtype = cat_promote_dtype(tensors);
  Tensor output = Tensor(out_shape, out_dtype, first.device());
  if (output.numel() == 0) {
    return output;
  }

  std::vector<Tensor> prepared;
  prepared.reserve(tensors.size());
  for (const Tensor& t : tensors) {
    Tensor tc = t.is_contiguous() ? t : t.contiguous();
    if (tc.dtype() != out_dtype) {
      tc = tc.to(out_dtype);
    }
    tc = materializeForGraph(tc);
    prepared.push_back(tc);
  }

  // Each input is [outer, inner] when the trailing dims from `dim` on are
  // flattened; the j-th input fills the j-th block of the new axis.
  const int64_t itemsize = output.itemsize();
  int64_t inner_bytes = itemsize;
  for (int64_t d = dim; d < ndim; ++d) {
    inner_bytes *= first.size(d);
  }
  const int64_t out_stride_bytes =
      static_cast<int64_t>(tensors.size()) * inner_bytes;
  const BlockCopyPipeline pso =
      MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
          "strided_block_copy");
  MPSStream* stream = getCurrentMPSStream();
  run_serialized(stream->queue(), ^() {
    @autoreleasepool {
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      int64_t off_bytes = 0;
      for (const Tensor& tc : prepared) {
        encode_block_copy(encoder, pso, tc, output, inner_bytes,
                          out_stride_bytes, off_bytes);
        off_bytes += inner_bytes;
      }
    }
  });
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
  indices_c = materializeForGraph(indices_c);
  weight_c = materializeForGraph(weight_c);

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

  if (num_embeddings == 0) {
    TP_THROW(IndexError, "embedding(): index out of range in self");
  }

  // Short index lists are range-checked on the host, which answers before the
  // call returns; longer ones are checked by the copy kernel itself as it
  // streams the indices, so the scan does not sit in the hot path. Either way
  // the copy clamps, so an entry that is still in flight while the report is
  // pending stays inside the weight. The stream only has to be drained when it
  // still holds work that may write the bytes the host scan reads.
  constexpr int64_t kHostIndexScanLimit = 4096;
  if (n <= kHostIndexScanLimit) {
    if (getCurrentMPSStream()->hasPendingWork()) {
      synchronizeHostAccess();
    }
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
  }

  const int64_t row_bytes = emb_dim * weight.itemsize();
  const NSUInteger chunks = static_cast<NSUInteger>(row_bytes + 15) / 16;
  const std::string kernel = "embedding_forward_" +
      scalarToMetalTypeString(indices_c.dtype());
  auto pso = MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
      kernel);
  MPSStream* mpsStream = getCurrentMPSStream();
  run_serialized(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, weight_c, indices_c, output,
                  static_cast<uint32_t>(num_embeddings),
                  static_cast<uint32_t>(row_bytes),
                  mpsStream->errorBuffer());
      mtl_dispatch3DJob(computeEncoder, pso, chunks,
                        static_cast<NSUInteger>(n), 1);
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