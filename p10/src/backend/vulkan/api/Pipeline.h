#pragma once

#ifdef USE_VULKAN

#include "Resource.h"
#include "Shader.h"

#include <mutex>
#include <unordered_map>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// A recorded execution dependency between shader invocations.  Barriers are
// declared by the op code (via vTensor accesses) and replayed into the
// command buffer right before the dispatch that needs them.
//
struct PipelineBarrier final {
  struct Stages final {
    VkPipelineStageFlags src;
    VkPipelineStageFlags dst;
  } stage;

  std::vector<BufferMemoryBarrier> buffers;
  std::vector<ImageMemoryBarrier> images;
  std::vector<VkBufferMemoryBarrier> buffer_barrier_handles;
  std::vector<VkImageMemoryBarrier> image_barrier_handles;

  inline operator bool() const {
    return (0u != stage.src) || (0u != stage.dst) || !buffers.empty() ||
        !images.empty();
  }
};

using PipelineStageFlags = uint8_t;

enum PipelineStage : PipelineStageFlags {
  NO_STAGE = 0u << 0u,
  COMPUTE = 1u << 0u,
  HOST = 1u << 1u,
  TRANSFER = 1u << 2u,
};

VkAccessFlags vk_access(const PipelineStageFlags, const MemoryAccessFlags);
VkPipelineStageFlags vk_stage(const PipelineStageFlags);

class PipelineLayout final {
 public:
  explicit PipelineLayout(VkDevice, VkDescriptorSetLayout);

  PipelineLayout(const PipelineLayout&) = delete;
  PipelineLayout& operator=(const PipelineLayout&) = delete;

  PipelineLayout(PipelineLayout&&) noexcept;
  PipelineLayout& operator=(PipelineLayout&&) = delete;

  ~PipelineLayout();

 private:
  VkDevice device_;
  VkPipelineLayout handle_;

 public:
  VkPipelineLayout handle() const {
    return handle_;
  }

  friend void swap(PipelineLayout& lhs, PipelineLayout& rhs) noexcept;
};

class ComputePipeline final {
 public:
  struct Descriptor final {
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    utils::uvec3 local_work_group;
  };

  explicit ComputePipeline(
      VkDevice device,
      const Descriptor& descriptor,
      VkPipelineCache pipeline_cache);

  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  ComputePipeline(ComputePipeline&&) noexcept;
  ComputePipeline& operator=(ComputePipeline&&) = delete;

  ~ComputePipeline();

 private:
  VkDevice device_;
  VkPipeline handle_;

 public:
  inline VkPipeline handle() const {
    return handle_;
  }

  friend void swap(ComputePipeline& lhs, ComputePipeline& rhs) noexcept;
};

class PipelineLayoutCache final {
 public:
  explicit PipelineLayoutCache(VkDevice device);

  PipelineLayoutCache(const PipelineLayoutCache&) = delete;
  PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

  PipelineLayoutCache(PipelineLayoutCache&&) noexcept;
  PipelineLayoutCache& operator=(PipelineLayoutCache&&) = delete;

  ~PipelineLayoutCache();

  using Key = VkDescriptorSetLayout;
  using Value = PipelineLayout;

  struct Hasher {
    inline size_t operator()(VkDescriptorSetLayout descriptor_layout) const {
      return std::hash<VkDescriptorSetLayout>()(descriptor_layout);
    }
  };

 private:
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkPipelineLayout retrieve(const Key&);
  void purge();
};

class ComputePipelineCache final {
 public:
  explicit ComputePipelineCache(VkDevice device);

  ComputePipelineCache(const ComputePipelineCache&) = delete;
  ComputePipelineCache& operator=(const ComputePipelineCache&) = delete;

  ComputePipelineCache(ComputePipelineCache&&) noexcept;
  ComputePipelineCache& operator=(ComputePipelineCache&&) = delete;

  ~ComputePipelineCache();

  using Key = ComputePipeline::Descriptor;
  using Value = ComputePipeline;

  struct Hasher {
    inline size_t operator()(
        const ComputePipeline::Descriptor& descriptor) const {
      size_t seed = 0;
      seed = utils::hash_combine(
          seed, std::hash<VkPipelineLayout>()(descriptor.pipeline_layout));
      seed = utils::hash_combine(
          seed, std::hash<VkShaderModule>()(descriptor.shader_module));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group[0u]));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group[1u]));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(descriptor.local_work_group[2u]));

      return seed;
    }
  };

 private:
  std::mutex cache_mutex_;

  VkDevice device_;
  VkPipelineCache pipeline_cache_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkPipeline retrieve(const Key&);
  void purge();
};

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
