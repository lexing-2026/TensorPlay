#ifdef USE_VULKAN

#define VMA_IMPLEMENTATION
#include "Allocator.h"
#include "Resource.h"
#include "Exception.h"

#include <utility>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// MemoryAllocator
//

MemoryAllocator::MemoryAllocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_(instance),
      physical_device_(physical_device),
      device_(device),
      allocator_(VK_NULL_HANDLE) {
  VK_CHECK_COND(instance, "Vulkan invalid instance handle!");
  VK_CHECK_COND(physical_device, "Vulkan invalid physical device handle!");
  VK_CHECK_COND(device, "Vulkan invalid device handle!");

  const VmaVulkanFunctions vulkan_functions{
      vkGetInstanceProcAddr,
      vkGetDeviceProcAddr,
  };

  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags
      physical_device_,
      device_,
      VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE, // preferredLargeHeapBlockSize
      nullptr, // pAllocationCallbacks
      nullptr, // pDeviceMemoryCallbacks
      nullptr, // pHeapSizeLimit
      &vulkan_functions, // pVulkanFunctions
      instance,
      VK_API_VERSION_1_0, // vulkanApiVersion
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_));
  VK_CHECK_COND(allocator_, "Invalid VMA allocator handle!");
}

MemoryAllocator::MemoryAllocator(MemoryAllocator&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      allocator_(other.allocator_) {
  other.instance_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.allocator_ = VK_NULL_HANDLE;
}

MemoryAllocator::~MemoryAllocator() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }

  vmaDestroyAllocator(allocator_);

  allocator_ = VK_NULL_HANDLE;
}

MemoryAllocation MemoryAllocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  return MemoryAllocation{
      allocator_,
      memory_requirements,
      create_info,
  };
}

VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat format,
    const VkImageType type,
    const VkImageViewType view_type,
    const VulkanImage::SamplerProperties& sampler_properties,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  const VmaAllocationCreateInfo allocation_create_info{
      VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT, // flags
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, // usage
      0u, // requiredFlags
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
  };

  return VulkanImage{
      allocator_,
      allocation_create_info,
      {
          type, // image_type
          format, // image_format
          extents, // image_extents
          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
              (allow_transfer
                   ? (VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                   : 0u), // image_usage
      },
      {
          view_type, // view_type
          format, // view_format
      },
      sampler_properties,
      // Supported layouts the shaders expect
      VK_IMAGE_LAYOUT_GENERAL,
      sampler,
      allocate_memory,
  };
}

VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only,
    const bool allocate_memory) {
  const VmaAllocationCreateInfo allocation_create_info{
      VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT, // flags
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, // usage
      gpu_only
          ? 0u
          : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
              VMA_ALLOCATION_CREATE_MAPPED_BIT, // requiredFlags
      0u, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
  };

  return VulkanBuffer{
      allocator_,
      size,
      allocation_create_info,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      allocate_memory,
  };
}

VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  const VmaAllocationCreateInfo allocation_create_info{
      VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT, // flags
      VMA_MEMORY_USAGE_AUTO, // usage
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, // requiredFlags
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
  };

  return VulkanBuffer{
      allocator_,
      size,
      allocation_create_info,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
  };
}

VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  const VmaAllocationCreateInfo allocation_create_info{
      VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT, // flags
      VMA_MEMORY_USAGE_AUTO, // usage
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, // requiredFlags
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
  };

  return VulkanBuffer{
      allocator_,
      size,
      allocation_create_info,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
  };
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
