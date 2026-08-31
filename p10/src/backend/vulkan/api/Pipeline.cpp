#ifdef USE_VULKAN

#include "Pipeline.h"
#include "Utils.h"
#include "Exception.h"

#include <utility>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// Stage helpers
//

namespace {

inline VkPipelineStageFlags to_vk_stage(const PipelineStageFlags stage) {
  VkPipelineStageFlags flags = 0u;

  if (stage & PipelineStage::COMPUTE) {
    flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }
  if (stage & PipelineStage::HOST) {
    flags |= VK_PIPELINE_STAGE_HOST_BIT;
  }
  if (stage & PipelineStage::TRANSFER) {
    flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  return flags;
}

} // namespace

VkAccessFlags vk_access(
    const PipelineStageFlags stage,
    const MemoryAccessFlags access) {
  VkAccessFlags flags = 0u;

  if (stage & PipelineStage::COMPUTE) {
    if (access & MemoryAccessType::READ) {
      flags |= VK_ACCESS_SHADER_READ_BIT;
    }
    if (access & MemoryAccessType::WRITE) {
      flags |= VK_ACCESS_SHADER_WRITE_BIT;
    }
  }

  if (stage & PipelineStage::HOST) {
    if (access & MemoryAccessType::READ) {
      flags |= VK_ACCESS_HOST_READ_BIT;
    }
    if (access & MemoryAccessType::WRITE) {
      flags |= VK_ACCESS_HOST_WRITE_BIT;
    }
  }

  if (stage & PipelineStage::TRANSFER) {
    if (access & MemoryAccessType::READ) {
      flags |= VK_ACCESS_TRANSFER_READ_BIT;
    }
    if (access & MemoryAccessType::WRITE) {
      flags |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
  }

  return flags;
}

VkPipelineStageFlags vk_stage(const PipelineStageFlags stage) {
  return to_vk_stage(stage);
}

//
// PipelineLayout
//

PipelineLayout::PipelineLayout(VkDevice device, VkDescriptorSetLayout descriptor_set_layout)
    : device_(device),
      handle_{} {
  VK_CHECK_COND(device_, "Invalid Vulkan device handle!");
  VK_CHECK_COND(descriptor_set_layout, "Invalid descriptor set layout!");

  const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      1u, // setLayoutCount
      &descriptor_set_layout, // pSetLayouts
      0u, // pushConstantRangeCount
      nullptr, // pPushConstantRanges
  };

  VK_CHECK(vkCreatePipelineLayout(
      device_, &pipeline_layout_create_info, nullptr, &handle_));
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept {
  std::swap(lhs.device_, rhs.device_);
  std::swap(lhs.handle_, rhs.handle_);
}

PipelineLayout::~PipelineLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroyPipelineLayout(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

//
// ComputePipeline
//

ComputePipeline::ComputePipeline(
    VkDevice device,
    const Descriptor& descriptor,
    VkPipelineCache pipeline_cache)
    : device_(device),
      handle_{} {
  VK_CHECK_COND(device_, "Invalid Vulkan device handle!");
  VK_CHECK_COND(descriptor.pipeline_layout, "Invalid pipeline layout!");
  VK_CHECK_COND(descriptor.shader_module, "Invalid shader module!");

  const VkPipelineShaderStageCreateInfo stage_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_SHADER_STAGE_COMPUTE_BIT, // stage
      descriptor.shader_module, // module
      "main", // pName
      nullptr, // pSpecializationInfo
  };

  const VkComputePipelineCreateInfo pipeline_create_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      stage_create_info, // stage
      descriptor.pipeline_layout, // layout
      VK_NULL_HANDLE, // basePipelineHandle
      -1, // basePipelineIndex
  };

  VK_CHECK(vkCreateComputePipelines(
      device_, pipeline_cache, 1u, &pipeline_create_info, nullptr, &handle_));
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept {
  std::swap(lhs.device_, rhs.device_);
  std::swap(lhs.handle_, rhs.handle_);
}

ComputePipeline::~ComputePipeline() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroyPipeline(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

static bool operator==(
    const ComputePipeline::Descriptor& lhs,
    const ComputePipeline::Descriptor& rhs) {
  return lhs.pipeline_layout == rhs.pipeline_layout &&
      lhs.shader_module == rhs.shader_module &&
      lhs.local_work_group == rhs.local_work_group;
}

//
// PipelineLayoutCache
//

PipelineLayoutCache::PipelineLayoutCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      cache_{} {}

PipelineLayoutCache::PipelineLayoutCache(PipelineLayoutCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      cache_(std::move(other.cache_)) {
  other.device_ = VK_NULL_HANDLE;
  other.cache_.clear();
}

PipelineLayoutCache::~PipelineLayoutCache() {
  try {
    purge();
  } catch (...) {
  }
}

VkPipelineLayout PipelineLayoutCache::retrieve(const Key& key) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  const auto it = cache_.find(key);

  if (cache_.cend() != it) {
    return it->second.handle();
  }

  Value pipeline_layout(device_, key);

  VkPipelineLayout handle = pipeline_layout.handle();

  cache_.emplace(key, std::move(pipeline_layout));

  return handle;
}

void PipelineLayoutCache::purge() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  cache_.clear();
}

//
// ComputePipelineCache
//

ComputePipelineCache::ComputePipelineCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      pipeline_cache_{VK_NULL_HANDLE},
      cache_{} {
  const VkPipelineCacheCreateInfo pipeline_cache_create_info{
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      0u, // initialDataSize
      nullptr, // pInitialData
  };

  VK_CHECK(vkCreatePipelineCache(
      device_, &pipeline_cache_create_info, nullptr, &pipeline_cache_));
}

ComputePipelineCache::ComputePipelineCache(ComputePipelineCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      pipeline_cache_(other.pipeline_cache_),
      cache_(std::move(other.cache_)) {
  other.device_ = VK_NULL_HANDLE;
  other.pipeline_cache_ = VK_NULL_HANDLE;
  other.cache_.clear();
}

ComputePipelineCache::~ComputePipelineCache() {
  try {
    purge();
  } catch (...) {
  }

  if (VK_NULL_HANDLE == pipeline_cache_) {
    return;
  }

  vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);

  pipeline_cache_ = VK_NULL_HANDLE;
}

VkPipeline ComputePipelineCache::retrieve(const Key& key) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  const auto it = cache_.find(key);

  if (cache_.cend() != it) {
    return it->second.handle();
  }

  Value pipeline(device_, key, pipeline_cache_);

  VkPipeline handle = pipeline.handle();

  cache_.emplace(key, std::move(pipeline));

  return handle;
}

void ComputePipelineCache::purge() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
