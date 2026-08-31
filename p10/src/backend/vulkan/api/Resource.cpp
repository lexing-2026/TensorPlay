#ifdef USE_VULKAN

#include "Resource.h"
#include "Exception.h"

#include <cstring>
#include <utility>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// MemoryAllocation
//

MemoryAllocation::MemoryAllocation()
    : memory_requirements{},
      create_info{},
      allocator{VK_NULL_HANDLE},
      allocation{VK_NULL_HANDLE} {}

MemoryAllocation::MemoryAllocation(
    const VmaAllocator allocator_handle,
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(memory_requirements),
      create_info(create_info),
      allocator(allocator_handle),
      allocation{VK_NULL_HANDLE} {
  VK_CHECK(vmaAllocateMemory(
      allocator_handle, &memory_requirements, &create_info, &allocation, nullptr));
}

MemoryAllocation::MemoryAllocation(MemoryAllocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation) {
  other.memory_requirements = {};
  other.create_info = {};
  other.allocator = VK_NULL_HANDLE;
  other.allocation = VK_NULL_HANDLE;
}

MemoryAllocation& MemoryAllocation::operator=(MemoryAllocation&& other) noexcept {
  if (this != &other) {
    memory_requirements = other.memory_requirements;
    create_info = other.create_info;
    allocator = other.allocator;
    allocation = other.allocation;

    other.memory_requirements = {};
    other.create_info = {};
    other.allocator = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
  }

  return *this;
}

MemoryAllocation::~MemoryAllocation() {
  if (VK_NULL_HANDLE == allocation) {
    return;
  }

  vmaFreeMemory(allocator, allocation);

  allocation = VK_NULL_HANDLE;
}

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{
          0u,
          0u,
          0u,
          0u,
      },
      allocator_{},
      memory_{},
      owns_memory_(false),
      handle_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    const VmaAllocator allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_{
          size,
          0u,
          size,
          usage,
      },
      allocator_(allocator),
      memory_{},
      owns_memory_(allocate_memory),
      handle_(VK_NULL_HANDLE) {
  VK_CHECK_COND(allocator, "Invalid VMA allocator handle!");

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // size
      usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  if (allocate_memory) {
    VmaAllocationInfo allocation_info{};
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &memory_.allocation,
        &allocation_info));

    memory_.allocator = allocator_;
    memory_.create_info = VmaAllocationCreateInfo(allocation_create_info);
    vmaGetAllocationMemoryProperties(
        allocator_, memory_.allocation, &memory_.create_info.requiredFlags);
  } else {
    VK_CHECK(vkCreateBuffer(
        device(),
        &buffer_create_info,
        nullptr,
        &handle_));
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handle_(other.handle_) {
  other.buffer_properties_ = {
      0u,
      0u,
      0u,
      0u,
  };
  other.allocator_ = VK_NULL_HANDLE;
  other.owns_memory_ = false;
  other.handle_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  if (this != &other) {
    buffer_properties_ = other.buffer_properties_;
    allocator_ = other.allocator_;
    memory_ = std::move(other.memory_);
    owns_memory_ = other.owns_memory_;
    handle_ = other.handle_;

    other.buffer_properties_ = {
        0u,
        0u,
        0u,
        0u,
    };
    other.allocator_ = VK_NULL_HANDLE;
    other.owns_memory_ = false;
    other.handle_ = VK_NULL_HANDLE;
  }

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  if (owns_memory_ && memory_.allocation) {
    vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
  } else {
    vkDestroyBuffer(device(), handle_, nullptr);
  }

  // The allocation was released above by vmaDestroyBuffer, or is not owned by
  // this resource; either way it must not be freed again by MemoryAllocation.
  memory_.allocation = VK_NULL_HANDLE;

  handle_ = VK_NULL_HANDLE;
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VK_CHECK_COND(handle_, "Cannot query memory requirements of an empty buffer!");

  VkMemoryRequirements memory_requirements{};
  vkGetBufferMemoryRequirements(
      device(), handle_, &memory_requirements);

  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(
    const VulkanBuffer& buffer,
    const MemoryAccessFlags access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_(buffer.mem_size()) {
  VK_CHECK_COND(
      buffer.has_memory(),
      "Cannot map a Vulkan buffer that has no memory allocated to it!");

  VmaAllocationInfo allocation_info{};
  vmaGetAllocationInfo(allocator_, allocation_, &allocation_info);

  void* mapped_ptr = allocation_info.pMappedData;
  if (!mapped_ptr) {
    VK_CHECK(
        vmaMapMemory(allocator_, allocation_, &data_));
    mapped_ptr = data_;
  }

  data_ = mapped_ptr;
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_(other.data_len_) {
  other.access_ = MemoryAccessType::NONE;
  other.allocator_ = VK_NULL_HANDLE;
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
  other.data_len_ = 0u;
}

MemoryMap::~MemoryMap() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }

  if (access_ & MemoryAccessType::WRITE) {
    invalidate();
  }

  // Only unmap when we acquired our own mapping; persistent mappings are
  // managed by VMA.
  VmaAllocationInfo allocation_info{};
  vmaGetAllocationInfo(allocator_, allocation_, &allocation_info);
  if (data_ && data_ != allocation_info.pMappedData) {
    vmaUnmapMemory(allocator_, allocation_);
  }

  data_ = nullptr;
}

void MemoryMap::invalidate() {
  if (!allocator_ || !data_) {
    return;
  }

  // Buffers are allocated with HOST_VISIBLE memory; when the host has
  // written them the writes must be flushed before device access unless the
  // memory type is HOST_COHERENT.  Asking VMA to flush is a no-op for
  // coherent memory.
  vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle(), // buffer
          0u, // offset
          buffer.mem_range(), // size
      } {
  VK_CHECK_COND(
      buffer,
      "Vulkan BufferMemoryBarrier: VulkanBuffer is invalid!");
}

//
// ImageSampler
//

static bool operator==(
    const ImageSampler::Properties& lhs,
    const ImageSampler::Properties& rhs) {
  return lhs.filter == rhs.filter &&
      lhs.mipmap_mode == rhs.mipmap_mode &&
      lhs.address_mode == rhs.address_mode &&
      lhs.border_color == rhs.border_color;
}

ImageSampler::ImageSampler(VkDevice device, const Properties& properties)
    : device_(device),
      handle_{} {
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      properties.filter, // magFilter
      properties.filter, // minFilter
      properties.mipmap_mode, // mipmapMode
      properties.address_mode, // addressModeU
      properties.address_mode, // addressModeV
      properties.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      0.0f, // maxAnisotropy
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      0.0f, // maxLod
      properties.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  VK_CHECK(
      vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
  VK_CHECK_COND(handle_, "Invalid sampler handle!");
}

ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  std::swap(lhs.device_, rhs.device_);
  std::swap(lhs.handle_, rhs.handle_);
}

ImageSampler::~ImageSampler() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroySampler(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

size_t ImageSampler::Hasher::operator()(const Properties& properties) const {
  size_t seed = 0u;
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(properties.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(properties.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(properties.address_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkBorderColor>()(properties.border_color));

  return seed;
}

//
// VulkanImage
//

VulkanImage::VulkanImage()
    : image_properties_{
          VK_IMAGE_TYPE_2D,
          VK_FORMAT_UNDEFINED,
          {},
          0u,
      },
      view_properties_{
          VK_IMAGE_VIEW_TYPE_2D,
          VK_FORMAT_UNDEFINED,
      },
      sampler_properties_{
          VK_FILTER_NEAREST,
          VK_SAMPLER_MIPMAP_MODE_NEAREST,
          VK_SAMPLER_ADDRESS_MODE_REPEAT,
          VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      },
      allocator_{},
      memory_{},
      owns_memory_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_(VK_IMAGE_LAYOUT_UNDEFINED) {}

VulkanImage::VulkanImage(
    const VmaAllocator allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_properties,
    const ViewProperties& view_properties,
    const SamplerProperties& sampler_properties,
    const VkImageLayout layout,
    VkSampler sampler,
    const bool allocate_memory)
    : image_properties_(image_properties),
      view_properties_(view_properties),
      sampler_properties_(sampler_properties),
      allocator_(allocator),
      memory_{},
      owns_memory_(allocate_memory),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout) {
  VK_CHECK_COND(allocator, "Invalid VMA allocator handle!");

  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      image_properties.image_type, // imageType
      image_properties.image_format, // format
      image_properties.image_extents, // extent
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      VK_IMAGE_TILING_OPTIMAL, // tiling
      image_properties.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout, // initialLayout
  };

  if (allocate_memory) {
    VmaAllocationInfo allocation_info{};
    VK_CHECK(vmaCreateImage(
        allocator_,
        &image_create_info,
        &allocation_create_info,
        &handles_.image,
        &memory_.allocation,
        &allocation_info));

    memory_.allocator = allocator_;
    memory_.create_info = VmaAllocationCreateInfo(allocation_create_info);
    vmaGetAllocationMemoryProperties(
        allocator_, memory_.allocation, &memory_.create_info.requiredFlags);

    // Only create the image view if the image has been bound to memory
    create_image_view();
  } else {
    VK_CHECK(vkCreateImage(
        device(),
        &image_create_info,
        nullptr,
        &handles_.image));
  }
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handles_(other.handles_),
      layout_(other.layout_) {
  other.image_properties_ = {
      VK_IMAGE_TYPE_2D,
      VK_FORMAT_UNDEFINED,
      {},
      0u,
  };
  other.view_properties_ = {
      VK_IMAGE_VIEW_TYPE_2D,
      VK_FORMAT_UNDEFINED,
  };
  other.allocator_ = VK_NULL_HANDLE;
  other.owns_memory_ = false;
  other.handles_ = {
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
  };
  other.layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  if (this != &other) {
    image_properties_ = other.image_properties_;
    view_properties_ = other.view_properties_;
    sampler_properties_ = other.sampler_properties_;
    allocator_ = other.allocator_;
    memory_ = std::move(other.memory_);
    owns_memory_ = other.owns_memory_;
    handles_ = other.handles_;
    layout_ = other.layout_;

    other.image_properties_ = {
        VK_IMAGE_TYPE_2D,
        VK_FORMAT_UNDEFINED,
        {},
        0u,
    };
    other.view_properties_ = {
        VK_IMAGE_VIEW_TYPE_2D,
        VK_FORMAT_UNDEFINED,
    };
    other.allocator_ = VK_NULL_HANDLE;
    other.owns_memory_ = false;
    other.handles_ = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    other.layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  return *this;
}

void VulkanImage::create_image_view() {
  VK_CHECK_COND(handles_.image, "Image has not been bound to memory!");

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      handles_.image, // image
      view_properties_.view_type, // viewType
      view_properties_.view_format, // format
      {
          VK_COMPONENT_SWIZZLE_IDENTITY, // swizzleR
          VK_COMPONENT_SWIZZLE_IDENTITY, // swizzleG
          VK_COMPONENT_SWIZZLE_IDENTITY, // swizzleB
          VK_COMPONENT_SWIZZLE_IDENTITY, // swizzleA
      }, // components
      {
          VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
          0u, // baseMipLevel
          1u, // levelCount
          0u, // baseArrayLayer
          1u, // layerCount
      }, // subresourceRange
  };

  VK_CHECK(vkCreateImageView(
      device(),
      &image_view_create_info,
      nullptr,
      &handles_.image_view));
  VK_CHECK_COND(handles_.image_view, "Invalid image view handle!");
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE == handles_.image) {
    return;
  }

  if (handles_.image_view) {
    vkDestroyImageView(device(), handles_.image_view, nullptr);
  }

  if (owns_memory_ && memory_.allocation) {
    vmaDestroyImage(allocator_, handles_.image, memory_.allocation);
  } else {
    vkDestroyImage(device(), handles_.image, nullptr);
  }

  // The allocation was released above by vmaDestroyImage, or is not owned by
  // this resource; either way it must not be freed again by MemoryAllocation.
  memory_.allocation = VK_NULL_HANDLE;

  handles_ = {
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
      VK_NULL_HANDLE,
  };
}

VkMemoryRequirements VulkanImage::get_memory_requirements() const {
  VK_CHECK_COND(handles_.image, "Cannot query memory requirements of an empty image!");

  VkMemoryRequirements memory_requirements{};
  vkGetImageMemoryRequirements(
      device(), handles_.image, &memory_requirements);

  return memory_requirements;
}

//
// ImageMemoryBarrier
//

ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          src_layout_flags, // oldLayout
          dst_layout_flags, // newLayout
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          image.handle(), // image
          {
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
              0u, // baseMipLevel
              1u, // levelCount
              0u, // baseArrayLayer
              1u, // layerCount
          }, // subresourceRange
      } {
  VK_CHECK_COND(
      image,
      "Vulkan ImageMemoryBarrier: VulkanImage is invalid!");
}

//
// SamplerCache
//

SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      cache_(std::move(other.cache_)) {
  other.device_ = VK_NULL_HANDLE;
  other.cache_.clear();
}

SamplerCache::~SamplerCache() {
  try {
    purge();
  } catch (...) {
  }
}

VkSampler SamplerCache::retrieve(const Key& key) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  const auto it = cache_.find(key);

  if (cache_.cend() != it) {
    return it->second.handle();
  }

  Value sampler(device_, key);

  VkSampler handle = sampler.handle();

  cache_.emplace(key, std::move(sampler));

  return handle;
}

void SamplerCache::purge() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  cache_.clear();
}

//
// VulkanFence
//

VulkanFence::VulkanFence(VkDevice device)
    : device_(device),
      handle_{} {
  const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
  };

  VK_CHECK(vkCreateFence(device_, &fence_create_info, nullptr, &handle_));
}

VulkanFence::VulkanFence(VulkanFence&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept {
  if (this != &other) {
    device_ = other.device_;
    handle_ = other.handle_;

    other.device_ = VK_NULL_HANDLE;
    other.handle_ = VK_NULL_HANDLE;
  }

  return *this;
}

VulkanFence::~VulkanFence() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroyFence(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

void VulkanFence::wait() const {
  VK_CHECK_COND(handle_, "Invalid Vulkan fence!");

  VK_CHECK(vkWaitForFences(device_, 1u, &handle_, VK_TRUE, UINT64_MAX));
  VK_CHECK(vkResetFences(device_, 1u, &handle_));
}

bool VulkanFence::query() const {
  VK_CHECK_COND(handle_, "Invalid Vulkan fence!");

  return (vkGetFenceStatus(device_, handle_) == VK_SUCCESS);
}

VkFence VulkanFence::get_submit_handle() {
  return handle_;
}

VkFence VulkanFence::get_wait_handle() const {
  return handle_;
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
