#ifdef USE_VULKAN

#include "VulkanRuntime.h"

#include "backend/vulkan/api/Adapter.h"
#include "backend/vulkan/api/Context.h"
#include "backend/vulkan/api/Resource.h"
#include "backend/vulkan/api/Runtime.h"

#include <cstring>
#include <vulkan/vulkan.h>

namespace tensorplay {
namespace vulkan {

bool is_available() {
  return api::available();
}

void synchronize(int device) {
  (void)device;
  api::Context* context = api::context();
  TP_CHECK(context, "Vulkan context is not available");
  context->flush();
  // vkQueueWaitIdle inside flush guarantees completion.
}

int device_count() {
  api::Runtime* rt = api::try_runtime();
  if (!rt) {
    return 0;
  }
  // The runtime materializes adapters lazily; the physical device mappings
  // describe the enumeration.  One context exists for the default adapter,
  // which is the only device served for now.
  return rt->is_initialized() ? 1 : 0;
}

namespace {

const api::Adapter* default_adapter() {
  api::Runtime* rt = api::try_runtime();
  if (!rt || !rt->is_initialized()) {
    return nullptr;
  }
  return rt->get_adapter_p(rt->default_adapter_i());
}

} // namespace

std::string device_name(int device) {
  const api::Adapter* adapter = default_adapter();
  TP_CHECK(
      adapter && device == 0, "Vulkan device index out of range");
  return adapter->properties().deviceName;
}

uint32_t device_api_version(int device) {
  const api::Adapter* adapter = default_adapter();
  TP_CHECK(
      adapter && device == 0, "Vulkan device index out of range");
  return adapter->properties().apiVersion;
}

uint64_t device_total_memory(int device) {
  const api::Adapter* adapter = default_adapter();
  TP_CHECK(
      adapter && device == 0, "Vulkan device index out of range");
  const VkPhysicalDeviceMemoryProperties& mem = adapter->memory_properties();
  uint64_t total = 0;
  for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += mem.memoryHeaps[i].size;
    }
  }
  return total;
}

void copyHostVisibleBytes(void* destination, const Device& destination_device,
                          const void* source, const Device& source_device,
                          size_t nbytes) {
  if (!destination || !source || nbytes == 0) return;

  api::Context* context = api::context();
  TP_CHECK(context, "Vulkan context is not available");

  const bool dst_vk = destination_device.is_vulkan();
  const bool src_vk = source_device.is_vulkan();

  TP_CHECK(
      !dst_vk || !src_vk,
      "Storage resize between two Vulkan allocations is not supported");

  // Host <-> device transfers ride on host-visible staging buffers plus one
  // queue submission.  The opaque pointers are VulkanBuffer objects owned by
  // the storages' DataPtr contexts, which stay alive for the duration of the
  // call (storage resize keeps the old allocation until the copy completes).
  api::PipelineBarrier barrier{};
  const api::utils::uvec3 range{
      api::utils::safe_downcast_to_u32(static_cast<int64_t>(nbytes)),
      1u,
      1u};

  if (dst_vk) {
    const api::VulkanBuffer* dst =
        static_cast<const api::VulkanBuffer*>(destination);
    api::StorageBuffer staging(context, DType::UInt8, nbytes);
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      TP_CHECK(mapping.nbytes() >= nbytes, "Staging buffer too small");
      std::memcpy(mapping.data<uint8_t>(), source, nbytes);
    }
    context->submit_copy(
        barrier,
        staging.buffer(),
        *dst,
        range,
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
    context->flush();
    return;
  }

  if (src_vk) {
    const api::VulkanBuffer* src =
        static_cast<const api::VulkanBuffer*>(source);
    api::StorageBuffer staging(context, DType::UInt8, nbytes);
    context->submit_copy(
        barrier,
        *src,
        staging.buffer(),
        range,
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
    context->flush();
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();
    TP_CHECK(mapping.nbytes() >= nbytes, "Staging buffer too small");
    std::memcpy(destination, mapping.data<uint8_t>(), nbytes);
    return;
  }
}

} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
