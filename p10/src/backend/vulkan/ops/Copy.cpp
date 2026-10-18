#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "Factory.h"
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

  // Buffer payloads hold a plain linear layout.  The move is recorded as a
  // compute dispatch instead of a transfer command: payloads are consumed by
  // compute dispatches, and reaching the device through the same stage keeps
  // the producer/consumer dependency that a transfer-then-compute handoff
  // would have to spell out for every first read of a fresh upload.
  if (v_dst.storage_type() == api::StorageType::BUFFER) {
    Tensor src_lin = prepare_source(src, v_dst.dtype());

    api::StorageBuffer staging(context, v_dst.dtype(), v_dst.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      mapping.invalidate();

      memcpy(
          mapping.data<void>(),
          src_lin.impl()->storage().data(),
          staging.nbytes());
    }

    if (staging.nbytes() == 0u) {
      return;
    }

    utils::upload_staging_to_buffer(
        staging.buffer(), v_dst, staging.nbytes());
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

// Device-side dtype conversions.  Float32 is the hub: every payload type the
// backend stores has a direct route to and from it, and anything else has no
// kernel.  Byte-packed payloads agree on the read path, since a Bool lane is
// already 0 or 1 and an integer lane converts exactly, and split on the write
// path: Int8 and UInt8 destinations take the wrapped value, a Bool one only
// ever stores 0 or 1.
const char* texture_cast_shader(const DType from, const DType to) {
  if (to == DType::Float32) {
    switch (from) {
      case DType::Int32:
        return "cast_i32_float";
      case DType::Float16:
        return "cast_float16_float";
      case DType::Int8:
      case DType::Bool:
        return "cast_i8_float";
      case DType::UInt8:
        return "cast_u8_float";
      default:
        return nullptr;
    }
  }
  if (from == DType::Float32) {
    switch (to) {
      case DType::Int32:
        return "cast_float_i32";
      case DType::Float16:
        return "cast_float_float16";
      case DType::Int8:
        return "cast_float_i8";
      case DType::UInt8:
        return "cast_float_u8";
      case DType::Bool:
        return "cast_float_bool";
      default:
        return nullptr;
    }
  }
  return nullptr;
}

const char* linear_cast_shader(const DType from, const DType to) {
  if (to == DType::Float32) {
    switch (from) {
      case DType::Int32:
        return "cast_i32_float_buffer";
      case DType::Float16:
        return "cast_float16_float_buffer";
      case DType::Int8:
      case DType::Bool:
        return "cast_byte_float_buffer_i8";
      case DType::UInt8:
        return "cast_byte_float_buffer_u8";
      default:
        return nullptr;
    }
  }
  if (from == DType::Float32) {
    switch (to) {
      case DType::Int32:
        return "cast_float_i32_buffer";
      case DType::Float16:
        return "cast_float_float16_buffer";
      case DType::Int8:
      case DType::UInt8:
        return "cast_float_byte_buffer_wrap";
      case DType::Bool:
        return "cast_float_byte_buffer_bool";
      default:
        return nullptr;
    }
  }
  return nullptr;
}

// Elements one invocation owns in a linear payload: a half-precision payload
// packs two lanes per 32-bit word, every other element width fills a word with
// four.  The narrower side of a pair decides.
uint32_t linear_payload_lanes(const DType dtype) {
  return dtype == DType::Float16 ? 2u : 4u;
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
        TP_CHECK(src.shape() == self.shape(),
                 "Vulkan dtype conversion requires equal shapes");

        // A linear payload converts through its own kernels: the storage has
        // no texel format to name the element type, so a half payload keeps
        // two lanes per 32-bit word and a byte-packed one four.  Mixing an
        // image with a linear payload has no route here, since neither
        // binding can read the other.
        const bool linear_source =
            v_src.storage_type() == api::StorageType::BUFFER;
        const bool linear_destination =
            v_self.storage_type() == api::StorageType::BUFFER;
        if (linear_source || linear_destination) {
          TP_CHECK(linear_source && linear_destination,
                   "Vulkan dtype conversion does not mix storage types");
          const char* buffer_shader =
              linear_cast_shader(v_src.dtype(), v_self.dtype());
          TP_CHECK(
              buffer_shader != nullptr,
              "Unsupported Vulkan dtype conversion over linear storage");
          const uint32_t elements = static_cast<uint32_t>(v_self.numel());
          const uint32_t source_lanes = linear_payload_lanes(v_src.dtype());
          const uint32_t destination_lanes =
              linear_payload_lanes(v_self.dtype());
          const uint32_t lanes =
              source_lanes < destination_lanes ? source_lanes
                                               : destination_lanes;
          const struct CastBlock final {
            uint32_t buf_length;
            uint32_t fill0;
            uint32_t fill1;
            uint32_t fill2;
          } block{elements, 0u, 0u, 0u};
          api::UniformParamsBuffer params(context, block);
          api::PipelineBarrier barrier{};
          context->submit_compute_job(
              VK_KERNEL_FROM_STR(buffer_shader), barrier,
              {api::utils::div_up(elements, lanes), 1u, 1u}, {256u, 1u, 1u},
              VK_NULL_HANDLE,
              v_self.buffer(
                  barrier, api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_src.buffer(barrier, api::PipelineStage::COMPUTE),
              params.buffer());
          return self;
        }

        const char* shader = texture_cast_shader(v_src.dtype(), v_self.dtype());
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
// to.dtype(): a same-device element type conversion.  An unchanged type
// aliases the input unless a copy was requested; otherwise the payload is
// allocated in the target type and converted through the copy path, which
// owns the supported type pairs.  A kernel running on this backend has to go
// through the operator table for conversions, since the host-side one is a
// CPU entry point.
//
Tensor to_dtype_kernel(
    const Tensor& self,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format) {
  // The device payload is dense in whichever storage it lives in, so the
  // requested layout has nothing to choose.
  (void)memory_format;
  if (self.dtype() == dtype && !copy) {
    return self;
  }
  TP_CHECK(
      self.device().is_vulkan(),
      "Vulkan to.dtype expects a Vulkan-resident input");
  Tensor out = empty_like_kernel(self, dtype, std::nullopt);
  copy_kernel(out, self, non_blocking);
  return out;
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
  m.impl("to.dtype", &tensorplay::vulkan::ops::to_dtype_kernel);
}

#endif /* USE_VULKAN */
