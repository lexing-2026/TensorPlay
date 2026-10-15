#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/kernels/EmbeddingBag.h"
#include "backend/mps/kernels/error.h"

#include <Metal/Metal.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

bool defined_and_mps(const std::optional<Tensor>& t, const char* op) {
  return t.has_value() && t->defined() && t->device().is_mps();
}

template <typename ParamsT>
void bind_bag_args(id<MTLComputeCommandEncoder> encoder,
                   const Tensor& t, unsigned idx) {
  Tensor c = materializeForGraph(t);
  [encoder setBuffer:mtlBufferOf(c) offset:0 atIndex:idx];
}

} // namespace

// _embedding_bag forward.  mode: 0=sum, 1=mean, 2=max.
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_kernel(
    const Tensor& weight, const Tensor& indices_, const Tensor& offsets_,
    bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset, int64_t padding_idx) {
  (void)scale_grad_by_freq;
  (void)sparse;
  if (!weight.device().is_mps() || !indices_.device().is_mps() ||
      !offsets_.device().is_mps()) {
    TP_THROW(RuntimeError, "_embedding_bag: tensors must be on the MPS device");
  }
  if (weight.dim() != 2) {
    TP_THROW(RuntimeError, "_embedding_bag: weight must be 2D");
  }
  if (indices_.dim() != 1 || offsets_.dim() != 1) {
    TP_THROW(RuntimeError,
             "_embedding_bag: indices and offsets must be 1D tensors");
  }

  Tensor indices = indices_;
  Tensor offsets = offsets_;
  if (indices.dtype() != offsets.dtype()) {
    offsets = offsets.to(indices.dtype());
  }
  if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
    TP_THROW(RuntimeError,
             "_embedding_bag: indices must be int32 or int64");
  }

  const int64_t num_indices = indices.numel();
  int64_t num_bags = offsets.numel();
  if (include_last_offset) {
    if (num_bags < 1) {
      TP_THROW(RuntimeError,
               "_embedding_bag: include_last_offset requires at least one "
               "offset");
    }
    num_bags -= 1;
  }
  const int64_t feature_size = weight.size(1);
  const DType dt = weight.dtype();
  if (dt != DType::Float32 && dt != DType::Float16 &&
      dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "_embedding_bag: weight must be float32/float16/bfloat16");
  }

  Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
  Tensor indices_c = indices.is_contiguous() ? indices : indices.contiguous();
  Tensor offsets_c = offsets.is_contiguous() ? offsets : offsets.contiguous();

  std::vector<int64_t> idx_shape = {num_bags};
  Tensor output = Tensor::empty({num_bags, feature_size}, dt, weight.device());
  Tensor offset2bag = Tensor::empty({num_indices}, indices.dtype(),
                                    weight.device());
  Tensor bag_size = Tensor::empty(idx_shape, indices.dtype(), weight.device());
  Tensor max_indices;
  if (mode == 2) {
    max_indices = Tensor::empty({num_bags, feature_size}, indices.dtype(),
                               weight.device());
  } else {
    max_indices = Tensor::empty({0}, indices.dtype(), weight.device());
  }
  if (output.numel() == 0) {
    return {output, offset2bag, bag_size, max_indices};
  }

  EmbeddingBagParams<uint32_t> params{};
  for (int64_t d = 0; d < 2; ++d) {
    params.weight_strides[d] = static_cast<uint32_t>(weight_c.stride(d));
    params.output_strides[d] = static_cast<uint32_t>(output.stride(d));
    if (mode == 2) {
      params.max_indices_strides[d] =
          static_cast<uint32_t>(max_indices.stride(d));
    }
  }
  const bool use_per_sample_weights =
      per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();
  Tensor per_sample_weights;
  if (use_per_sample_weights) {
    per_sample_weights = per_sample_weights_opt->is_contiguous()
        ? *per_sample_weights_opt
        : per_sample_weights_opt->contiguous();
  }
  params.use_per_sample_weights = use_per_sample_weights;
  params.per_sample_weights_stride =
      use_per_sample_weights
          ? static_cast<uint32_t>(per_sample_weights.stride(0))
          : 0;
  params.num_indices = static_cast<uint32_t>(num_indices);
  params.num_bags = static_cast<uint32_t>(num_bags);
  params.feature_size = static_cast<uint32_t>(feature_size);
  params.mode = static_cast<EmbeddingBagMode>(mode);
  params.padding_idx = padding_idx;
  params.num_weights = static_cast<uint32_t>(weight_c.size(0));

  const std::string kernel = "embedding_bag_" +
      scalarToMetalTypeString(dt) + "_" +
      scalarToMetalTypeString(indices_c.dtype());

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      id<MTLDevice> device = MPSDevice::getInstance()->device();
      id<MTLBuffer> errorBuf = [[device
          newBufferWithLength:sizeof(p10::metal::ErrorMessages)
                       options:0] autorelease];
      bind_bag_args(computeEncoder, weight_c, 0);
      bind_bag_args(computeEncoder, indices_c, 1);
      bind_bag_args(computeEncoder, offsets_c, 2);
      if (use_per_sample_weights) {
        bind_bag_args(computeEncoder, per_sample_weights, 3);
      } else {
        [computeEncoder setBuffer:nil offset:0 atIndex:3];
      }
      bind_bag_args(computeEncoder, output, 4);
      bind_bag_args(computeEncoder, offset2bag, 5);
      bind_bag_args(computeEncoder, bag_size, 6);
      bind_bag_args(computeEncoder, max_indices, 7);
      [computeEncoder setBytes:&params length:sizeof(params) atIndex:8];
      [computeEncoder setBuffer:errorBuf offset:0 atIndex:9];
      mtl_dispatch1DJob(computeEncoder, PSO,
                        static_cast<NSUInteger>(output.numel()));
      mpsStream->endKernelCoalescing();
    }
  });

  return {output, offset2bag, bag_size, max_indices};
}

// _embedding_bag dense backward.
Tensor embedding_bag_dense_backward_kernel(
    const Tensor& output_grad_, const Tensor& indices_,
    const Tensor& offset2bag_, const Tensor& bag_size_,
    const Tensor& maximum_indices_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode,
    const std::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  (void)scale_grad_by_freq;
  if (!output_grad_.device().is_mps()) {
    TP_THROW(RuntimeError,
             "_embedding_bag_dense_backward: tensors must be on the MPS "
             "device");
  }
  const DType dt = output_grad_.dtype();
  if (dt != DType::Float32 && dt != DType::Float16 &&
      dt != DType::BFloat16) {
    TP_THROW(RuntimeError,
             "_embedding_bag_dense_backward: grad must be "
             "float32/float16/bfloat16");
  }
  const int64_t feature_size = output_grad_.size(1);
  Tensor weight_grad =
      Tensor::zeros({num_weights, feature_size}, dt, output_grad_.device());

  Tensor output_grad =
      output_grad_.is_contiguous() ? output_grad_ : output_grad_.contiguous();
  Tensor indices = indices_.is_contiguous() ? indices_ : indices_.contiguous();
  Tensor offset2bag =
      offset2bag_.is_contiguous() ? offset2bag_ : offset2bag_.contiguous();
  Tensor bag_size = bag_size_.is_contiguous() ? bag_size_ : bag_size_.contiguous();
  Tensor max_indices =
      maximum_indices_.is_contiguous() ? maximum_indices_
                                       : maximum_indices_.contiguous();

  EmbeddingBagBackwardParams<uint32_t> params{};
  for (int64_t d = 0; d < 2; ++d) {
    params.output_grad_strides[d] =
        static_cast<uint32_t>(output_grad.stride(d));
    params.weight_grad_strides[d] =
        static_cast<uint32_t>(weight_grad.stride(d));
    if (mode == 2) {
      params.max_indices_strides[d] =
          static_cast<uint32_t>(max_indices.stride(d));
    }
  }
  const bool use_per_sample_weights =
      per_sample_weights_opt.has_value() && per_sample_weights_opt->defined();
  Tensor per_sample_weights;
  if (use_per_sample_weights) {
    per_sample_weights = per_sample_weights_opt->is_contiguous()
        ? *per_sample_weights_opt
        : per_sample_weights_opt->contiguous();
  }
  params.use_per_sample_weights = use_per_sample_weights;
  params.per_sample_weights_stride =
      use_per_sample_weights
          ? static_cast<uint32_t>(per_sample_weights.stride(0))
          : 0;
  params.feature_size = static_cast<uint32_t>(feature_size);
  params.mode = static_cast<EmbeddingBagMode>(mode);
  params.padding_idx = padding_idx;

  const int64_t num_indices = offset2bag.numel();
  const int64_t num_threads =
      (mode == 2) ? output_grad.numel() : num_indices * feature_size;
  if (num_threads == 0) {
    return weight_grad;
  }

  const std::string kernel = "embedding_bag_backward_" +
      scalarToMetalTypeString(dt) + "_" +
      scalarToMetalTypeString(indices.dtype());

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      bind_bag_args(computeEncoder, output_grad, 0);
      bind_bag_args(computeEncoder, indices, 1);
      bind_bag_args(computeEncoder, offset2bag, 2);
      bind_bag_args(computeEncoder, bag_size, 3);
      bind_bag_args(computeEncoder, max_indices, 4);
      if (use_per_sample_weights) {
        bind_bag_args(computeEncoder, per_sample_weights, 5);
      } else {
        [computeEncoder setBuffer:nil offset:0 atIndex:5];
      }
      bind_bag_args(computeEncoder, weight_grad, 6);
      [computeEncoder setBytes:&params length:sizeof(params) atIndex:7];
      mtl_dispatch1DJob(computeEncoder, PSO,
                        static_cast<NSUInteger>(num_threads));
      mpsStream->endKernelCoalescing();
    }
  });

  return weight_grad;
}

// per_sample_weights backward (SUM mode only).
Tensor embedding_bag_per_sample_weights_backward_kernel(
    const Tensor& output_grad_, const Tensor& weight_,
    const Tensor& indices_, const Tensor& offsets_,
    const Tensor& offset2bag_, int64_t mode, int64_t padding_idx) {
  (void)offsets_;
  if (mode != 0) {
    TP_THROW(RuntimeError,
             "_embedding_bag_per_sample_weights_backward: only SUM mode is "
             "supported");
  }
  const int64_t num_indices = indices_.numel();
  const int64_t feature_size = output_grad_.size(1);
  Tensor per_sample_weights_grad =
      Tensor::zeros({num_indices}, output_grad_.dtype(), output_grad_.device());

  Tensor output_grad =
      output_grad_.is_contiguous() ? output_grad_ : output_grad_.contiguous();
  Tensor weight = weight_.is_contiguous() ? weight_ : weight_.contiguous();
  Tensor indices = indices_.is_contiguous() ? indices_ : indices_.contiguous();
  Tensor offset2bag =
      offset2bag_.is_contiguous() ? offset2bag_ : offset2bag_.contiguous();

  EmbeddingBagPerSampleWeightsBackwardParams params{};
  for (int64_t d = 0; d < 2; ++d) {
    params.output_grad_strides[d] =
        static_cast<uint32_t>(output_grad.stride(d));
    params.weight_strides[d] = static_cast<uint32_t>(weight.stride(d));
  }
  params.per_sample_weights_grad_stride =
      static_cast<uint32_t>(per_sample_weights_grad.stride(0));
  params.feature_size = static_cast<uint32_t>(feature_size);
  params.padding_idx = padding_idx;

  if (num_indices * feature_size == 0) {
    return per_sample_weights_grad;
  }

  const std::string kernel = "embedding_bag_per_sample_weights_backward_" +
      scalarToMetalTypeString(output_grad.dtype()) + "_" +
      scalarToMetalTypeString(indices.dtype());

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      bind_bag_args(computeEncoder, output_grad, 0);
      bind_bag_args(computeEncoder, weight, 1);
      bind_bag_args(computeEncoder, indices, 2);
      bind_bag_args(computeEncoder, offset2bag, 3);
      bind_bag_args(computeEncoder, per_sample_weights_grad, 4);
      [computeEncoder setBytes:&params length:sizeof(params) atIndex:5];
      mtl_dispatch1DJob(computeEncoder, PSO,
                        static_cast<NSUInteger>(num_indices * feature_size));
      mpsStream->endKernelCoalescing();
    }
  });

  return per_sample_weights_grad;
}

TENSORPLAY_LIBRARY_IMPL(MPS, EmbeddingBagKernels) {
  m.impl("_embedding_bag", embedding_bag_kernel);
  m.impl("_embedding_bag_forward_only", embedding_bag_kernel);
  m.impl("_embedding_bag_dense_backward", embedding_bag_dense_backward_kernel);
  m.impl("_embedding_bag_per_sample_weights_backward",
         embedding_bag_per_sample_weights_backward_kernel);
}

} // namespace mps
} // namespace tensorplay