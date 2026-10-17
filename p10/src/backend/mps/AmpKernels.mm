#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSDevice.h"
#include "backend/mps/MetalShaderLibrary.h"
#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSStream.h"

#include <Metal/Metal.h>

#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace mps {

namespace {

bool is_mps_tensor(const Tensor& t, const char* op) {
  if (!t.defined() || !t.device().is_mps()) {
    TP_THROW(RuntimeError, op, ": all tensors must be on the MPS device");
  }
  return true;
}

// Returns a contiguous, offset-0 tensor safe for direct buffer binding.
Tensor shader_tensor(const Tensor& t) {
  return (t.is_contiguous() && t.storage_offset() == 0) ? t
                                                        : t.contiguous();
}

id<MTLBuffer> mtl_buffer_of(const Tensor& t) {
  void* buf = getMTLBufferFromPointer(t.data_ptr());
  if (!buf) {
    TP_THROW(RuntimeError,
             "MPS tensor storage is not managed by the MPS allocator");
  }
  return (__bridge id<MTLBuffer>)buf;
}

struct MPSMetadataArguments {
  uint64_t numels[32];
  uint64_t threadgroup_to_tensor[32];
  uint64_t threadgroup_to_chunk[32];
};

// Chunked multi-tensor dispatch for ampNonFiniteCheckAndUnscale (depth 1).
// Each tensor is processed in 64K-element chunks mapped to threadgroups; up
// to 32 tensors/chunks per kernel launch.
void dispatch_amp_multi(const std::string& kernel_name,
                        const std::vector<Tensor>& tensor_list,
                        Tensor& found_inf,
                        const Tensor& inv_scale) {
  const size_t num_tensors = tensor_list.size();
  if (num_tensors == 0) {
    return;
  }
  constexpr int64_t kChunkSize = 65536;
  constexpr int64_t kmaxThreadGroups = 32;

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel_name);
      [computeEncoder setComputePipelineState:PSO];
      MTLFunction_t func =
          MetalShaderLibrary::getBundledLibrary().getFunction(kernel_name);
      // The kernel reads one pointer array of tensors from its argument buffer.
      ForeachArguments arguments(computeEncoder, func,
                                 MPSDevice::getInstance()->device());
      [func release];
      id<MTLBuffer> tensorArgumentBuffer = arguments.buffer();

      int tensor_loc = 0;
      int threadgroup_loc = 0;
      MPSMetadataArguments metadata_arguments;
      memset(&metadata_arguments, 0, sizeof(metadata_arguments));

      for (size_t t = 0; t < num_tensors; ++t) {
        if (tensor_list[t].numel() == 0) {
          continue;
        }
        arguments.bind(static_cast<NSUInteger>(tensor_loc),
                       mtl_buffer_of(tensor_list[t]));
        metadata_arguments.numels[tensor_loc] =
            static_cast<uint64_t>(tensor_list[t].numel());
        const int64_t current_tensor_index = tensor_loc;
        tensor_loc++;

        const int64_t numel = tensor_list[t].numel();
        const int64_t chunks =
            numel / kChunkSize + ((numel % kChunkSize) ? 1 : 0);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
          metadata_arguments.threadgroup_to_tensor[threadgroup_loc] =
              static_cast<uint64_t>(current_tensor_index);
          metadata_arguments.threadgroup_to_chunk[threadgroup_loc] =
              static_cast<uint64_t>(chunk);
          threadgroup_loc++;

          const bool dispatch_now = (threadgroup_loc == kmaxThreadGroups) ||
              (chunk == chunks - 1);
          if (dispatch_now) {
            const bool partial = (chunk != chunks - 1);
            uint64_t carried_numels = 0;
            if (partial) {
              carried_numels =
                  metadata_arguments.numels[current_tensor_index];
            }

            [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
            [computeEncoder setBytes:&metadata_arguments
                              length:sizeof(MPSMetadataArguments)
                               atIndex:1];
            [computeEncoder setBuffer:mtl_buffer_of(found_inf) offset:0
                              atIndex:2];
            [computeEncoder setBuffer:mtl_buffer_of(inv_scale) offset:0
                              atIndex:3];
            MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
            const uint32_t maxThreads =
                [PSO maxTotalThreadsPerThreadgroup];
            MTLSize threadGroupSize = MTLSizeMake(
                std::min(maxThreads, static_cast<uint32_t>(64)), 1, 1);
            [computeEncoder dispatchThreadgroups:gridSize
                           threadsPerThreadgroup:threadGroupSize];

            threadgroup_loc = 0;
            tensorArgumentBuffer = arguments.buffer();

            if (partial) {
              arguments.bind(0, mtl_buffer_of(tensor_list[t]));
              metadata_arguments.numels[0] = carried_numels;
              tensor_loc = 1;
            } else {
              tensor_loc = 0;
            }
          }
        }
      }

      if (threadgroup_loc != 0) {
        [computeEncoder setBuffer:tensorArgumentBuffer offset:0 atIndex:0];
        [computeEncoder setBytes:&metadata_arguments
                          length:sizeof(MPSMetadataArguments)
                           atIndex:1];
        [computeEncoder setBuffer:mtl_buffer_of(found_inf) offset:0 atIndex:2];
        [computeEncoder setBuffer:mtl_buffer_of(inv_scale) offset:0 atIndex:3];
        MTLSize gridSize = MTLSizeMake(threadgroup_loc, 1, 1);
        const uint32_t maxThreads = [PSO maxTotalThreadsPerThreadgroup];
        MTLSize threadGroupSize =
            MTLSizeMake(std::min(maxThreads, static_cast<uint32_t>(64)), 1, 1);
        [computeEncoder dispatchThreadgroups:gridSize
                       threadsPerThreadgroup:threadGroupSize];
      }
      mpsStream->endKernelCoalescing();
      // These AMP ops mutate tensors in place; the caller may read the
      // results immediately, so flush and wait before returning.
      mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
}

// Single-tensor fallback for the non-finite check + unscale.  `scaled_grad`
// may be non-contiguous or an offset view; the shader writes through a
// contiguous copy which is copied back afterwards.
void amp_non_finite_check_and_unscale_single_impl(const Tensor& scaled_grad,
                                                  Tensor& found_inf,
                                                  const Tensor& inv_scale) {
  if (scaled_grad.numel() == 0) {
    return;
  }
  if (scaled_grad.numel() > std::numeric_limits<uint32_t>::max()) {
    TP_THROW(RuntimeError, "scaled_grad is too large");
  }
  Tensor data = shader_tensor(scaled_grad);
  const bool need_copy_back = !(scaled_grad.is_contiguous() &&
                                scaled_grad.storage_offset() == 0);
  Tensor found_c = shader_tensor(found_inf);
  Tensor inv_c = shader_tensor(inv_scale);

  const std::string kernel = "ampNonFiniteCheckAndUnscaleSingle_" +
      scalarToMetalTypeString(scaled_grad.dtype());
  const uint32_t threadsPerThreadgroup = 256;
  const uint32_t numel = static_cast<uint32_t>(scaled_grad.numel());
  MTLSize threadGroupSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
  MTLSize gridSize = MTLSizeMake(numel, 1, 1);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, data, found_c, inv_c);
      [computeEncoder dispatchThreads:gridSize
                 threadsPerThreadgroup:threadGroupSize];
      mpsStream->endKernelCoalescing();
      // Flush and wait so the copy-back below reads updated data.
      mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });

  if (need_copy_back) {
    const_cast<Tensor&>(scaled_grad).copy_(data);
  }
}

void amp_update_scale_impl(Tensor& self,
                           Tensor& growth_tracker,
                           const Tensor& found_inf,
                           float scale_growth_factor,
                           float scale_backoff_factor,
                           int32_t growth_interval) {
  const std::string kernel =
      "ampUpdateScale_" + scalarToMetalTypeString(self.dtype());
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> PSO =
          MetalShaderLibrary::getBundledLibrary().getPipelineStateForFunc(
              kernel);
      [computeEncoder setComputePipelineState:PSO];
      mtl_setArgs(computeEncoder, self, growth_tracker, found_inf,
                  scale_growth_factor, scale_backoff_factor, growth_interval);
      mtl_dispatch1DJob(computeEncoder, PSO, 1);
      mpsStream->endKernelCoalescing();
    }
  });
}

} // namespace

void amp_foreach_non_finite_check_and_unscale_mps(
    std::vector<Tensor> self, Tensor& found_inf, const Tensor& inv_scale) {
  if (self.empty()) {
    return;
  }
  is_mps_tensor(found_inf, "_amp_foreach_non_finite_check_and_unscale_");
  is_mps_tensor(inv_scale, "_amp_foreach_non_finite_check_and_unscale_");
  if (found_inf.numel() != 1 || inv_scale.numel() != 1) {
    TP_THROW(RuntimeError,
             "_amp_foreach_non_finite_check_and_unscale_: found_inf and "
             "inv_scale must be 1-element tensors");
  }
  if (found_inf.dtype() != DType::Float32 ||
      inv_scale.dtype() != DType::Float32) {
    TP_THROW(RuntimeError,
             "_amp_foreach_non_finite_check_and_unscale_: found_inf and "
             "inv_scale must be float32");
  }

  std::vector<Tensor> fast;
  fast.reserve(self.size());
  for (const Tensor& t : self) {
    is_mps_tensor(t, "_amp_foreach_non_finite_check_and_unscale_");
    if (t.is_contiguous() && t.storage_offset() == 0) {
      fast.push_back(t);
    } else {
      // Non-contiguous or offset views go through the single-tensor path.
      amp_non_finite_check_and_unscale_single_impl(t, found_inf, inv_scale);
    }
  }
  if (fast.empty()) {
    return;
  }

  const std::string kernel_name = "ampNonFiniteCheckAndUnscale_" +
      scalarToMetalTypeString(fast[0].dtype());
  MTLFunction_t func =
      MetalShaderLibrary::getBundledLibrary().getFunction(kernel_name);
  const bool vendsEncoders = argumentEncodersAvailable(func);
  [func release];
  if (!vendsEncoders) {
    // The foreach kernel reads its tensors from an argument buffer, which only
    // an encoder can fill.  Where the image vends none, every tensor gets its
    // own launch with direct bindings.
    for (const Tensor& t : fast) {
      amp_non_finite_check_and_unscale_single_impl(t, found_inf, inv_scale);
    }
    return;
  }
  dispatch_amp_multi(kernel_name, fast, found_inf, inv_scale);
}

Tensor& amp_update_scale_mps(Tensor& self,
                             Tensor& growth_tracker,
                             const Tensor& found_inf,
                             double growth_factor,
                             double backoff_factor,
                             int64_t growth_interval) {
  if (growth_interval > std::numeric_limits<int32_t>::max() ||
      growth_interval < std::numeric_limits<int32_t>::min()) {
    TP_THROW(RuntimeError, "_amp_update_scale_: growth_interval out of range");
  }
  amp_update_scale_impl(self, growth_tracker, found_inf,
                        static_cast<float>(growth_factor),
                        static_cast<float>(backoff_factor),
                        static_cast<int32_t>(growth_interval));
  return self;
}

TENSORPLAY_LIBRARY_IMPL(MPS, AmpKernels) {
  m.impl("_amp_foreach_non_finite_check_and_unscale_",
         amp_foreach_non_finite_check_and_unscale_mps);
  m.impl("_amp_update_scale_", amp_update_scale_mps);
}

} // namespace mps
} // namespace tensorplay