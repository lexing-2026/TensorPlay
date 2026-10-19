#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/kernels/Embedding.h"

#include <Metal/Metal.h>

#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

constexpr int64_t kMaxNdim = 16;

} // namespace

// Accumulates grad rows into an embedding weight by index.
//   grad: (*indices.shape, D)
//   indices: any shape
//   output: (num_weights, D)
Tensor embedding_dense_backward_kernel(const Tensor& grad_,
                                       const Tensor& indices_,
                                       int64_t num_weights,
                                       int64_t padding_idx,
                                       bool scale_grad_by_freq) {
  if (!grad_.device().is_mps() || !indices_.device().is_mps()) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: tensors must be on the MPS device");
  }
  const DType grad_dt = grad_.dtype();
  if (grad_dt != DType::Float32 && grad_dt != DType::Float16 &&
      grad_dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: grad must be float32, float16 or "
             "bfloat16");
  }
  const DType idx_dt = indices_.dtype();
  if (idx_dt != DType::Int64 && idx_dt != DType::Int32) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: indices must be int32 or int64");
  }

  const int64_t D = grad_.size(-1);
  const int64_t num_indices = indices_.numel();
  if (num_indices == 0 || D == 0 || num_weights == 0) {
    return Tensor::zeros({num_weights, D}, grad_dt, grad_.device());
  }

  Tensor grad = grad_.is_contiguous() ? grad_ : grad_.contiguous();
  Tensor indices = indices_.is_contiguous() ? indices_ : indices_.contiguous();

  const int64_t outer_ndim = indices.dim();
  if (grad.dim() != outer_ndim + 1) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: grad dim (",
             grad.dim(),
             ") must equal indices dim + 1 (",
             outer_ndim + 1,
             ")");
  }
  if (outer_ndim > kMaxNdim) {
    TP_THROW(RuntimeError,
             "embedding_dense_backward: indices ndim ",
             outer_ndim,
             " exceeds supported maximum ",
             kMaxNdim);
  }

  const bool low_prec =
      grad_dt == DType::Float16 || grad_dt == DType::BFloat16;
  const DType acc_dt = low_prec ? DType::Float32 : grad_dt;
  Tensor grad_weight_acc =
      Tensor::zeros({num_weights, D}, acc_dt, grad.device());

  EmbeddingDenseBackwardParams<uint32_t> params{};
  params.outer_ndim = static_cast<uint32_t>(outer_ndim);
  for (int64_t d = 0; d < outer_ndim; ++d) {
    params.outer_sizes[d] = static_cast<uint32_t>(indices.size(d));
    params.indices_strides[d] = static_cast<uint32_t>(indices.stride(d));
    params.grad_outer_strides[d] = static_cast<uint32_t>(grad.stride(d));
  }
  params.grad_feature_stride = static_cast<uint32_t>(grad.stride(-1));
  params.feature_size = static_cast<uint32_t>(D);
  params.padding_idx = padding_idx;
  params.scale_grad_by_freq = scale_grad_by_freq;

  const bool use_32bit =
      num_indices <= std::numeric_limits<int32_t>::max() &&
      num_weights <= std::numeric_limits<int32_t>::max() &&
      num_weights * D <= std::numeric_limits<int32_t>::max();
  const std::string offset_suffix = use_32bit ? "32" : "64";
  const std::string idx_type = scalarToMetalTypeString(idx_dt);
  const std::string grad_type = scalarToMetalTypeString(grad_dt);

  Tensor counts;
  if (scale_grad_by_freq) {
    counts = Tensor::zeros({num_weights}, DType::UInt32, grad.device());
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  run_serialized(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      if (scale_grad_by_freq) {
        const std::string count_kernel = "embedding_dense_backward_count_" +
            idx_type + "_" + offset_suffix;
        id<MTLComputePipelineState> countPSO =
            MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
                count_kernel);
        [computeEncoder setComputePipelineState:countPSO];
        mtl_setArgs(computeEncoder, indices, counts, params);
        mtl_dispatch1DJob(computeEncoder, countPSO,
                          static_cast<NSUInteger>(num_indices));
      }

      const std::string bwd_kernel = "embedding_dense_backward_" + grad_type +
          "_" + idx_type + "_" + offset_suffix;
      id<MTLComputePipelineState> bwdPSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              bwd_kernel);
      [computeEncoder setComputePipelineState:bwdPSO];
      const Tensor counts_buf = scale_grad_by_freq ? counts : grad_weight_acc;
      mtl_setArgs(computeEncoder, grad, indices, counts_buf, grad_weight_acc,
                  params);
      mtl_dispatch1DJob(computeEncoder, bwdPSO,
                        static_cast<NSUInteger>(num_indices * D));
      mpsStream->endKernelCoalescing();
    }
  });

  return low_prec ? grad_weight_acc.to(grad_dt) : grad_weight_acc;
}

TENSORPLAY_LIBRARY_IMPL(MPS, EmbeddingKernels) {
  m.impl("embedding_dense_backward", embedding_dense_backward_kernel);
}

} // namespace mps
} // namespace tensorplay