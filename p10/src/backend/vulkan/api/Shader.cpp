#ifdef USE_VULKAN

#include "Shader.h"
#include "Utils.h"
#include "Exception.h"

#include <utility>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// ShaderLayout
//

ShaderLayout::ShaderLayout(VkDevice device, const Signature& signature)
    : device_(device),
      handle_{} {
  VK_CHECK_COND(device_, "Invalid Vulkan device handle!");
  VK_CHECK_COND(!signature.empty(), "Shader layout signature cannot be empty!");

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(signature.size());

  for (uint32_t i = 0u; i < signature.size(); ++i) {
    bindings.push_back({
        i, // binding
        signature[i], // descriptorType
        1u, // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
        nullptr, // pImmutableSamplers
    });
  }

  const VkDescriptorSetLayoutCreateInfo set_layout_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(bindings.size()), // bindingCount
      bindings.data(), // pBindings
  };

  VK_CHECK(vkCreateDescriptorSetLayout(
      device_, &set_layout_create_info, nullptr, &handle_));
}

ShaderLayout::ShaderLayout(ShaderLayout&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept {
  std::swap(lhs.device_, rhs.device_);
  std::swap(lhs.handle_, rhs.handle_);
}

ShaderLayout::~ShaderLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroyDescriptorSetLayout(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

//
// ShaderInfo
//

ShaderInfo::ShaderInfo()
    : src_code{
          nullptr,
          0u,
      },
      kernel_name{},
      kernel_layout(),
      out_tile_size{1u, 1u, 1u} {}

ShaderInfo::ShaderInfo(
    std::string kernel,
    const uint32_t* const code,
    const uint32_t num_code_words,
    std::vector<VkDescriptorType> layout,
    const utils::uvec3 out_tile_size)
    : src_code{
          code,
          num_code_words,
      },
      kernel_name(std::move(kernel)),
      kernel_layout(std::move(layout)),
      out_tile_size(out_tile_size) {
  VK_CHECK_COND(code, "Shader binary cannot be null!");
  VK_CHECK_COND(num_code_words > 0u, "Shader binary cannot be empty!");
  VK_CHECK_COND(!kernel_layout.empty(), "Shader layout signature cannot be empty!");
}

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2) {
  return (_1.src_code.bin == _2.src_code.bin) &&
      (_1.src_code.size == _2.src_code.size);
}

//
// ShaderModule
//

ShaderModule::ShaderModule(VkDevice device, const ShaderInfo& source)
    : device_(device),
      handle_{} {
  VK_CHECK_COND(device_, "Invalid Vulkan device handle!");
  VK_CHECK_COND(
      source.src_code.size > 0u,
      "Invalid SPIRV binary size for ",
      source.kernel_name,
      "!");

  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      source.src_code.size * sizeof(uint32_t), // codeSize
      source.src_code.bin, // pCode
  };

  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_));
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_) {
  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  std::swap(lhs.device_, rhs.device_);
  std::swap(lhs.handle_, rhs.handle_);
}

ShaderModule::~ShaderModule() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }

  vkDestroyShaderModule(device_, handle_, nullptr);

  handle_ = VK_NULL_HANDLE;
}

//
// ShaderLayoutCache
//

ShaderLayoutCache::ShaderLayoutCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      cache_{} {}

ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      cache_(std::move(other.cache_)) {
  other.device_ = VK_NULL_HANDLE;
  other.cache_.clear();
}

ShaderLayoutCache::~ShaderLayoutCache() {
  try {
    purge();
  } catch (...) {
  }
}

VkDescriptorSetLayout ShaderLayoutCache::retrieve(const Key& key) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  const auto it = cache_.find(key);

  if (cache_.cend() != it) {
    return it->second.handle();
  }

  Value layout(device_, key);

  VkDescriptorSetLayout handle = layout.handle();

  cache_.emplace(key, std::move(layout));

  return handle;
}

void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  cache_.clear();
}

//
// ShaderCache
//

ShaderCache::ShaderCache(VkDevice device)
    : cache_mutex_{},
      device_(device),
      cache_{} {}

ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{},
      device_(other.device_),
      cache_(std::move(other.cache_)) {
  other.device_ = VK_NULL_HANDLE;
  other.cache_.clear();
}

ShaderCache::~ShaderCache() {
  try {
    purge();
  } catch (...) {
  }
}

VkShaderModule ShaderCache::retrieve(const Key& key) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  const auto it = cache_.find(key);

  if (cache_.cend() != it) {
    return it->second.handle();
  }

  Value shader_module(device_, key);

  VkShaderModule handle = shader_module.handle();

  cache_.emplace(key, std::move(shader_module));

  return handle;
}

void ShaderCache::purge() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);

  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
