#pragma once

#ifdef USE_VULKAN

#include "Resource.h"
#include "Shader.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// One allocated VkDescriptorSet plus the bindings recorded against it.
// Bindings are accumulated by the op code (bind_buffer / bind_vkbuffer),
// then written in one vkUpdateDescriptorSets call at bind time.
//
class DescriptorSet final {
 public:
  explicit DescriptorSet(VkDevice, VkDescriptorSet, ShaderLayout::Signature);

  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&&) noexcept;
  DescriptorSet& operator=(DescriptorSet&&) noexcept;

  ~DescriptorSet() = default;

  struct ResourceBinding final {
    uint32_t binding_idx;
    VkDescriptorType descriptor_type;
    bool is_image;

    union {
      VkDescriptorBufferInfo buffer_info;
      VkDescriptorImageInfo image_info;
    } resource_info;
  };

 private:
  VkDevice device_;
  VkDescriptorSet handle_;
  ShaderLayout::Signature shader_layout_signature_;
  std::vector<ResourceBinding> bindings_;

 public:
  DescriptorSet& bind_buffer(const uint32_t, const VulkanBuffer&);
  DescriptorSet& bind_image(const uint32_t, const VulkanImage&);

  VkDescriptorSet get_bind_handle() const;

 private:
  void add_binding(const ResourceBinding& resource);
};

//
// Pre-allocated batch of descriptor sets for one shader layout.  Sets are
// handed out linearly; when the batch is exhausted a new one is allocated
// from the pool.
//
class DescriptorSetPile final {
 public:
  DescriptorSetPile(
      const uint32_t,
      VkDescriptorSetLayout,
      VkDevice,
      VkDescriptorPool);

  DescriptorSetPile(const DescriptorSetPile&) = delete;
  DescriptorSetPile& operator=(const DescriptorSetPile&) = delete;

  DescriptorSetPile(DescriptorSetPile&&) = default;

  ~DescriptorSetPile() = default;

 private:
  uint32_t pile_size_;
  VkDescriptorSetLayout set_layout_;
  VkDevice device_;
  VkDescriptorPool pool_;
  std::vector<VkDescriptorSet> descriptors_;
  size_t in_use_;

 public:
  VkDescriptorSet get_descriptor_set();

 private:
  void allocate_new_batch();
};

struct DescriptorPoolConfig final {
  // Overall Pool capacity
  uint32_t descriptorPoolMaxSets;
  // DescriptorCounts by type
  uint32_t descriptorUniformBufferCount;
  uint32_t descriptorStorageBufferCount;
  uint32_t descriptorCombinedSamplerCount;
  uint32_t descriptorStorageImageCount;
  // Pile size for pre-allocating descriptor sets
  uint32_t descriptorPileSizes;
};

//
// Owns the VkDescriptorPool and one DescriptorSetPile per distinct shader
// layout.  Sets are recycled by resetting the pool at flush boundaries;
// that is safe because every recorded use of a set completes before the
// pool is reset (the reset happens after vkQueueWaitIdle).
//
class DescriptorPool final {
 public:
  explicit DescriptorPool(VkDevice, const DescriptorPoolConfig&);

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  DescriptorPool(DescriptorPool&&) = delete;
  DescriptorPool& operator=(DescriptorPool&&) = delete;

  ~DescriptorPool();

 private:
  VkDevice device_;
  VkDescriptorPool pool_;
  DescriptorPoolConfig config_;
  // New Descriptors
  std::mutex mutex_;
  std::unordered_map<VkDescriptorSetLayout, DescriptorSetPile> piles_;

 public:
  operator bool() const {
    return (pool_ != VK_NULL_HANDLE);
  }

  void init(const DescriptorPoolConfig& config);

  DescriptorSet get_descriptor_set(
      VkDescriptorSetLayout handle,
      const ShaderLayout::Signature& signature);

  void flush();
};

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
