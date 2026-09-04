#ifdef USE_VULKAN

#include "Descriptor.h"
#include "Utils.h"
#include "Exception.h"

#include <utility>

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

namespace tensorplay {
namespace vulkan {
namespace api {

//
// DescriptorSet
//

DescriptorSet::DescriptorSet(
    VkDevice device,
    VkDescriptorSet handle,
    ShaderLayout::Signature shader_layout_signature)
    : device_(device),
      handle_(handle),
      shader_layout_signature_(std::move(shader_layout_signature)),
      bindings_{} {}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      shader_layout_signature_(std::move(other.shader_layout_signature_)),
      bindings_(std::move(other.bindings_)) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.shader_layout_signature_.clear();
  other.bindings_.clear();
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  if (this != &other) {
    device_ = other.device_;
    handle_ = other.handle_;
    shader_layout_signature_ = std::move(other.shader_layout_signature_);
    bindings_ = std::move(other.bindings_);

    other.device_ = VK_NULL_HANDLE;
    other.handle_ = VK_NULL_HANDLE;
    other.shader_layout_signature_.clear();
    other.bindings_.clear();
  }

  return *this;
}

DescriptorSet& DescriptorSet::bind_buffer(
    const uint32_t binding_idx,
    const VulkanBuffer& buffer) {
  VK_CHECK_COND(handle_, "Cannot bind to an invalid descriptor set!");
  VK_CHECK_COND(buffer, "Cannot bind an invalid buffer!");
  VK_CHECK_COND(
      binding_idx < shader_layout_signature_.size(),
      "Binding index out of range!");

  ResourceBinding resource{
      binding_idx,
      shader_layout_signature_[binding_idx],
      false,
      {},
  };

  resource.resource_info.buffer_info = {
      buffer.handle(), // buffer
      buffer.mem_offset(), // offset
      buffer.mem_range(), // range
  };

  add_binding(resource);

  return *this;
}

DescriptorSet& DescriptorSet::bind_image(
    const uint32_t binding_idx,
    const VulkanImage& image) {
  VK_CHECK_COND(handle_, "Cannot bind to an invalid descriptor set!");
  VK_CHECK_COND(image, "Cannot bind an invalid image!");
  VK_CHECK_COND(
      binding_idx < shader_layout_signature_.size(),
      "Binding index out of range!");

  ResourceBinding resource{
      binding_idx,
      shader_layout_signature_[binding_idx],
      true,
      {},
  };

  const VulkanImage::Package image_package = image.package();
  resource.resource_info.image_info = {
      image_package.image_sampler, // sampler
      image_package.image_view, // imageView
      image_package.image_layout, // imageLayout
  };

  add_binding(resource);

  return *this;
}

void DescriptorSet::add_binding(const ResourceBinding& resource) {
  // Rebinding an occupied slot must overwrite: descriptor sets are recycled
  // across submissions, so a reused set arrives with the previous call's
  // bindings still recorded.  Keeping the first write would silently bind
  // stale resources (e.g. last call's uniform block) whenever a shader runs
  // twice before the pool is flushed.
  for (ResourceBinding& bound : bindings_) {
    if (resource.binding_idx == bound.binding_idx) {
      bound = resource;
      return;
    }
  }

  bindings_.push_back(resource);
}

VkDescriptorSet DescriptorSet::get_bind_handle() const {
  if (bindings_.empty()) {
    return VK_NULL_HANDLE;
  }

  std::vector<VkWriteDescriptorSet> writes;
  writes.reserve(bindings_.size());

  for (const ResourceBinding& binding : bindings_) {
    writes.push_back({
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
        nullptr, // pNext
        handle_, // dstSet
        binding.binding_idx, // dstBinding
        0u, // dstArrayElement
        1u, // descriptorCount
        binding.descriptor_type, // descriptorType
        binding.is_image ? &binding.resource_info.image_info : nullptr, // pImageInfo
        !binding.is_image ? &binding.resource_info.buffer_info : nullptr, // pBufferInfo
        nullptr, // pTexelBufferView
    });
  }

  vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0u, nullptr);

  return handle_;
}

//
// DescriptorSetPile
//

DescriptorSetPile::DescriptorSetPile(
    const uint32_t pile_size,
    VkDescriptorSetLayout set_layout,
    VkDevice device,
    VkDescriptorPool pool)
    : pile_size_(pile_size),
      set_layout_(set_layout),
      device_(device),
      pool_(pool),
      descriptors_{},
      in_use_(0u) {
  if (0u == pile_size) {
    return;
  }

  allocate_new_batch();
}

VkDescriptorSet DescriptorSetPile::get_descriptor_set() {
  if (in_use_ >= descriptors_.size()) {
    allocate_new_batch();
  }

  return descriptors_[in_use_++];
}

void DescriptorSetPile::allocate_new_batch() {
  std::vector<VkDescriptorSetLayout> layouts(pile_size_, set_layout_);

  const VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // descriptorPool
      static_cast<uint32_t>(layouts.size()), // descriptorSetCount
      layouts.data(), // pSetLayouts
  };

  const size_t offset = descriptors_.size();
  descriptors_.resize(offset + pile_size_);

  VK_CHECK(vkAllocateDescriptorSets(
      device_, &allocate_info, descriptors_.data() + offset));
}

//
// DescriptorPool
//

DescriptorPool::DescriptorPool(VkDevice device, const DescriptorPoolConfig& config)
    : device_(device),
      pool_{VK_NULL_HANDLE},
      config_(config),
      mutex_{},
      piles_{} {
  init(config_);
}

void DescriptorPool::init(const DescriptorPoolConfig& config) {
  const VkDescriptorPoolSize pool_sizes[] = {
      {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // type
          config.descriptorUniformBufferCount, // descriptorCount
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // type
          config.descriptorStorageBufferCount, // descriptorCount
      },
      {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, // type
          config.descriptorCombinedSamplerCount, // descriptorCount
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, // type
          config.descriptorStorageImageCount, // descriptorCount
      },
  };

  const VkDescriptorPoolCreateInfo pool_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      config.descriptorPoolMaxSets, // maxSets
      static_cast<uint32_t>(sizeof(pool_sizes) / sizeof(VkDescriptorPoolSize)), // poolSizeCount
      pool_sizes, // pPoolSizes
  };

  VK_CHECK(
      vkCreateDescriptorPool(device_, &pool_create_info, nullptr, &pool_));
  VK_CHECK_COND(pool_, "Invalid descriptor pool handle!");
}

DescriptorPool::~DescriptorPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }

  vkDestroyDescriptorPool(device_, pool_, nullptr);

  pool_ = VK_NULL_HANDLE;
}

DescriptorSet DescriptorPool::get_descriptor_set(
    VkDescriptorSetLayout set_layout,
    const ShaderLayout::Signature& signature) {
  std::lock_guard<std::mutex> mutex_lock(mutex_);

  const auto it = piles_.find(set_layout);

  if (piles_.cend() != it) {
    return DescriptorSet{
        device_,
        it->second.get_descriptor_set(),
        signature,
    };
  }

  piles_.emplace(
      set_layout,
      DescriptorSetPile{
          config_.descriptorPileSizes,
          set_layout,
          device_,
          pool_,
      });

  return DescriptorSet{
      device_,
      piles_.at(set_layout).get_descriptor_set(),
      signature,
  };
}

void DescriptorPool::flush() {
  std::lock_guard<std::mutex> mutex_lock(mutex_);

  // All in-flight submissions have completed by the time the pool is
  // flushed, so every set handed out so far can be recycled in bulk.
  if (VK_NULL_HANDLE != pool_) {
    VK_CHECK(vkResetDescriptorPool(device_, pool_, 0u));
  }

  piles_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
