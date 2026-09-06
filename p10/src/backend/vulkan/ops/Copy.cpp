#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "Utils.h"
#include "../api/Context.h"
#include "../impl/Common.h"
#include "vulkan/Context.h"

#include <cstring>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

/*
 * Ensures the source tensor is dense and matches the destination's texture
 * dtype so a byte-level transfer into the staging buffer is valid.
 * Conversions and contiguity repairs run on the CPU.
 */
Tensor prepare_source(const Tensor& src, DType dst_dtype) {
  Tensor prepared = src;
  if (!prepared.is_contiguous()) {
    prepared = prepared.contiguous();
  }
  if (prepared.dtype() != dst_dtype) {
    prepared = prepared.to(dst_dtype);
  }
  return prepared;
}

//
// CPU -> Vulkan: bytes go into a host-visible staging buffer, then the
// pack shader (textures) or a copy command (buffers) moves them into the
// payload.
//
void transfer_cpu_to_vulkan(const Tensor& src, api::vTensor& v_dst) {
  api::Context* const context = api::context();

  // Float32 textures are packed through the compute pipeline: the staging
  // buffer holds contiguous NCHW values and the pack shader scatters them
  // into the texel layout.
  if (v_dst.storage_type() == api::StorageType::TEXTURE_3D &&
      v_dst.dtype() == DType::Float32) {
    Tensor src_contig = prepare_source(src, DType::Float32);

    api::StorageBuffer staging(context, DType::Float32, v_dst.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      mapping.invalidate();

      memcpy(
          mapping.data<void>(),
          src_contig.impl()->storage().data(),
          src_contig.numel() * src_contig.itemsize());
    }

    utils::pack_staging_to_vtensor(staging.buffer(), v_dst);
    return;
  }

  // Buffer payloads hold a plain linear layout.
  if (v_dst.storage_type() == api::StorageType::BUFFER) {
    Tensor src_lin = prepare_source(src, v_dst.texture_dtype());

    api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      mapping.invalidate();

      memcpy(
          mapping.data<void>(),
          src_lin.impl()->storage().data(),
          staging.nbytes());
    }

    api::PipelineBarrier pipeline_barrier{};
    context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
        pipeline_barrier,
        staging.buffer(),
        v_dst.buffer(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        {static_cast<uint32_t>(staging.buffer().mem_size()), 0u, 0u},
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
    return;
  }

  // Every other texture format moves through a byte-level transfer: the
  // staging buffer holds the texel-linear payload and the copy command is
  // format-agnostic, so no format-specific shader is needed.
  Tensor src_tex = prepare_source(src, v_dst.dtype());
  Tensor src_nc4hw = utils::nchw_to_nc4hw(src_tex);

  api::StorageBuffer staging(context, v_dst.dtype(), v_dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy(
        mapping.data<void>(),
        src_nc4hw.impl()->storage().data(),
        staging.nbytes());
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}

//
// Vulkan -> CPU: the payload streams into a staging buffer, the queue is
// synced through a fence, then the host reads the mapping.  Conversions are
// applied on the CPU side afterwards.
//
void transfer_vulkan_to_cpu(api::vTensor& v_src, Tensor& dst) {
  api::Context* const context = api::context();

  if (v_src.storage_type() == api::StorageType::BUFFER) {
    api::StorageBuffer staging(context, v_src.dtype(), v_src.numel());
    api::VulkanFence fence = context->fences().get_fence();
    {
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
      api::PipelineBarrier barrier{};
      context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
          barrier,
          v_src.buffer(barrier, api::PipelineStage::TRANSFER,
                       api::MemoryAccessType::READ),
          staging.buffer(), {static_cast<uint32_t>(v_src.nbytes()), 0u, 0u},
          {0u, 0u, 0u}, {0u, 0u, 0u}, fence.get_submit_handle());
      fence.wait();
      context->flush();
    }
    context->fences().return_fence(fence);
    Tensor host(v_src.sizes(), v_src.dtype(), Device(DeviceType::CPU));
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
      mapping.invalidate();
      std::memcpy(host.impl()->storage().data(), mapping.data<void>(), v_src.nbytes());
    }
    dst.copy_(host);
    return;
  }

  // Temporary tensor to receive copied NC4HW data
  Tensor dst_tmp = utils::create_staging_tensor(v_src);

  api::StorageBuffer staging(context, v_src.texture_dtype(), v_src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to the comment in submit_compute_job: when syncing with the GPU
    // the context must not allow other threads to record dispatches into it
    // between vkQueueSubmit and flushing, so the mutex is managed manually
    // here.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    api::PipelineBarrier pipeline_barrier{};
    utils::copy_vtensor_to_buffer(
        v_src, staging.buffer(), pipeline_barrier, fence.get_submit_handle());

    fence.wait();

    context->flush();
    // cmd_mutex_ will be released when exiting this scope.
  }

  context->fences().return_fence(fence);

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    memcpy(
        dst_tmp.impl()->storage().data(),
        mapping.data<void>(),
        staging.nbytes());
  }

  Tensor unpacked = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes());
  if (unpacked.dtype() != dst.dtype()) {
    unpacked = unpacked.to(dst.dtype());
  }
  if (!dst.is_contiguous()) {
    dst.copy_(unpacked.contiguous());
  } else {
    std::memcpy(
        dst.impl()->storage().data(),
        unpacked.impl()->storage().data(),
        dst.numel() * dst.itemsize());
  }
}

} // namespace

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
  if (self.numel() != src.numel()) {
    TP_THROW(RuntimeError, "Vulkan copy_: Tensor sizes are mismatched!");
  }
  (void)non_blocking;

  // Empty tensors have no bytes to move.
  if (self.numel() == 0) {
    return self;
  }

  // X -> Vulkan
  if (self.device().is_vulkan()) {
    api::vTensor v_self = convert(self);

    // Vulkan -> Vulkan
    if (src.device().is_vulkan()) {
      api::Context* const context = api::context();
      api::vTensor v_src = convert(src);

      if (v_src.dtype() != v_self.dtype()) {
        TP_CHECK(v_src.storage_type() == api::StorageType::TEXTURE_3D &&
                     v_self.storage_type() == api::StorageType::TEXTURE_3D,
                 "Vulkan dtype conversion requires texture storage");
        TP_CHECK(src.shape() == self.shape(),
                 "Vulkan dtype conversion requires equal shapes");
        const char* shader = nullptr;
        if (v_src.dtype() == DType::Int32 && v_self.dtype() == DType::Float32) {
          shader = "cast_i32_float";
        } else if (v_src.dtype() == DType::Float32 && v_self.dtype() == DType::Int32) {
          shader = "cast_float_i32";
        }
        TP_CHECK(shader != nullptr, "Unsupported Vulkan dtype conversion");
        api::PipelineBarrier barrier{};
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader), barrier, v_self.extents(),
            adaptive_work_group_size(v_self.extents()), VK_NULL_HANDLE,
            v_self.image(barrier, api::PipelineStage::COMPUTE,
                         api::MemoryAccessType::WRITE),
            v_src.image(barrier, api::PipelineStage::COMPUTE));
        return self;
      }

      if (v_self.storage_type() == api::StorageType::BUFFER &&
          v_src.storage_type() == api::StorageType::BUFFER) {
        api::PipelineBarrier pipeline_barrier{};
        context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
            pipeline_barrier,
            v_src.buffer(pipeline_barrier, api::PipelineStage::TRANSFER,
                         api::MemoryAccessType::READ),
            v_self.buffer(pipeline_barrier, api::PipelineStage::TRANSFER,
                          api::MemoryAccessType::WRITE),
            {static_cast<uint32_t>(v_self.nbytes()), 0u, 0u},
            {0u, 0u, 0u},
            {0u, 0u, 0u},
            VK_NULL_HANDLE);
        return self;
      }

      api::PipelineBarrier pipeline_barrier{};

      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // pipeline barrier
          pipeline_barrier,
          // images
          v_src.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::READ),
          v_self.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // copy details
          v_self.extents(),
          {0u, 0u, 0u},
          {0u, 0u, 0u},
          // fence handle
          VK_NULL_HANDLE);

      return self;
    }
    // CPU -> Vulkan
    transfer_cpu_to_vulkan(src, v_self);
    return self;
  }

  // Vulkan -> CPU
  if (src.device().is_vulkan()) {
    TP_CHECK(
        self.device().is_cpu(),
        "Vulkan copy_: only CPU destinations are supported");
    api::vTensor v_src = convert(src);
    transfer_vulkan_to_cpu(v_src, self);
    return self;
  }

  TP_THROW(RuntimeError, "Vulkan copy_: destination is not a Vulkan tensor");
}

void transfer_vulkan_to_cpu_impl(api::vTensor& v_src, Tensor& dst) {
  transfer_vulkan_to_cpu(v_src, dst);
}

//
// Backend registry hooks: availability probe and the copy entry point used
// by the generic copy path when the Vulkan impl registry is linked without
// the backend.
//
struct VulkanImpl final : public vulkan::ImplInterface {
  bool is_vulkan_available() const override {
    return api::available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return copy_kernel(self, src, /*non_blocking=*/false);
  }
};
static vulkan::ImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, CopyKernels) {
  m.impl("copy_", &tensorplay::vulkan::ops::copy_kernel);
}

#endif /* USE_VULKAN */
