#pragma once

#ifdef USE_VULKAN

#include "Adapter.h"
#include "Command.h"
#include "Descriptor.h"
#include "Pipeline.h"
#include "Resource.h"
#include "Runtime.h"
#include "Shader.h"
#include "Types.h"
#include "Utils.h"

#include <memory>
#include <mutex>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace api {

struct ContextConfig final {
  uint32_t cmdSubmitFrequency;
  CommandPoolConfig cmdPoolConfig;
  DescriptorPoolConfig descriptorPoolConfig;
};

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan.  A Context is associated with one, and only one, Adapter as
// a precursor to multi-GPU support.  All Vulkan tensors are associated with a
// Context to make tensor <-> device affinity explicit. The context is
// currently a global object, but technically it does not need to be if we
// were to make it explicit to the user.
//

class Context final {
 public:
  explicit Context(size_t adapter_i, const ContextConfig&);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  ~Context();

 private:
  // Config
  ContextConfig config_;
  // Important handles
  Adapter* adapter_p_;
  VkDevice device_;
  Adapter::Queue queue_;
  // Resource Pools
  CommandPool command_pool_;
  DescriptorPool descriptor_pool_;
  FencePool fences_;
  // Command buffers submission
  std::mutex cmd_mutex_;
  CommandBuffer cmd_;
  uint32_t submit_count_;
  // Memory Management
  std::mutex buffer_clearlist_mutex_;
  std::vector<VulkanBuffer> buffers_to_clear_;
  std::mutex image_clearlist_mutex_;
  std::vector<VulkanImage> images_to_clear_;

 public:
  // Adapter access

  inline Adapter* adapter_ptr() {
    return adapter_p_;
  }

  inline VkDevice device() {
    return device_;
  }

  inline VkQueue queue() {
    return queue_.handle;
  }

  // Device Caches

  inline ShaderLayoutCache& shader_layout_cache() {
    return adapter_ptr()->shader_layout_cache();
  }

  inline ShaderCache& shader_cache() {
    return adapter_ptr()->shader_cache();
  }

  inline PipelineLayoutCache& pipeline_layout_cache() {
    return adapter_ptr()->pipeline_layout_cache();
  }

  inline ComputePipelineCache& pipeline_cache() {
    return adapter_ptr()->compute_pipeline_cache();
  }

  // Resource Pools

  inline DescriptorPool& descriptor_pool() {
    return descriptor_pool_;
  }

  inline FencePool& fences() {
    return fences_;
  }

  // Memory Management
  void register_buffer_cleanup(VulkanBuffer& buffer) {
    std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
    buffers_to_clear_.emplace_back(std::move(buffer));
  }

  void register_image_cleanup(VulkanImage& image) {
    std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
    images_to_clear_.emplace_back(std::move(image));
  }

  // GPU RPC

  inline std::unique_lock<std::mutex> dispatch_lock() {
    return std::unique_lock<std::mutex>(cmd_mutex_);
  }

  inline void set_cmd(bool reusable = false) {
    if (!cmd_) {
      cmd_ = command_pool_.get_new_cmd(reusable);
      cmd_.begin();
    }
  }

  DescriptorSet get_descriptor_set(const ShaderInfo&, const utils::uvec3&);

  void register_shader_dispatch(
      const DescriptorSet&,
      PipelineBarrier&,
      const ShaderInfo&,
      const utils::uvec3&);

  template <class S, class D>
  bool submit_copy(
      PipelineBarrier&,
      const S&,
      const D&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      VkFence fence_handle);

  template <typename... Arguments>
  bool submit_compute_job(
      const ShaderInfo&,
      PipelineBarrier&,
      const utils::uvec3&,
      const utils::uvec3&,
      VkFence fence_handle,
      Arguments&&...);

  void submit_cmd_to_gpu(
      VkFence fence_handle = VK_NULL_HANDLE,
      const bool final_use = false);

  void flush();
};

namespace detail {

inline void arg_is_empty(bool& any_is_empty, const VulkanBuffer& buffer) {
  // operator bool evaluates to false when no memory is associated
  any_is_empty = any_is_empty || !buffer;
}

inline void arg_is_empty(bool& any_is_empty, const VulkanImage& image) {
  // operator bool evaluates to false when no memory is associated
  any_is_empty = any_is_empty || !image;
}

// Reports whether any buffer or image argument in a variadic list is empty.
template <typename... Arguments>
inline bool any_arg_is_empty(Arguments&&... arguments) {
  bool any_is_empty = false;
  VK_UNUSED const int _[]{
      0,
      (arg_is_empty(any_is_empty, std::forward<Arguments>(arguments)), 0)...,
  };

  return any_is_empty;
}

inline void bind(
    DescriptorSet& descriptor_set,
    const size_t idx,
    const VulkanBuffer& buffer) {
  descriptor_set.bind_buffer(idx, buffer);
}

inline void bind(
    DescriptorSet& descriptor_set,
    const size_t idx,
    const VulkanImage& image) {
  descriptor_set.bind_image(idx, image);
}

template <size_t... Indices, typename... Arguments>
inline void bind(
    DescriptorSet& descriptor_set,
    const std::index_sequence<Indices...>&,
    Arguments&&... arguments) {
  VK_UNUSED const int _[]{
      0,
      (bind(descriptor_set, Indices, std::forward<Arguments>(arguments)), 0)...,
  };
}

} // namespace detail

template <class S, class D>
inline void record_copy(
    CommandBuffer&,
    const S&,
    const D&,
    const api::utils::uvec3&,
    const api::utils::uvec3&,
    const api::utils::uvec3&) = delete;

template <>
inline void record_copy<VulkanBuffer, VulkanBuffer>(
    CommandBuffer& cmd,
    const VulkanBuffer& source,
    const VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_buffer_to_buffer(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanImage, VulkanImage>(
    CommandBuffer& cmd,
    const VulkanImage& source,
    const VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_texture_to_texture(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanImage, VulkanBuffer>(
    CommandBuffer& cmd,
    const VulkanImage& source,
    const VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_texture_to_buffer(
      source, destination, copy_range, src_offset, dst_offset);
}

template <>
inline void record_copy<VulkanBuffer, VulkanImage>(
    CommandBuffer& cmd,
    const VulkanBuffer& source,
    const VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  cmd.copy_buffer_to_texture(
      source, destination, copy_range, src_offset, dst_offset);
}

//
// Records a GPU data copy into the current command buffer.  If the number of
// submit_*_job calls exceeds the configured frequency, or if a fence is
// provided, the command buffer is submitted to the GPU for execution.
// Returns true when the call resulted in a queue submission.
//
template <class S, class D>
inline bool Context::submit_copy(
    PipelineBarrier& pipeline_barrier,
    const S& source,
    const D& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset,
    VkFence fence_handle) {
  // Exit early when an argument has no memory attached.  With a fence
  // present, the pending command buffer must still be submitted so the
  // fence gets signaled.
  if (!source || !destination) {
    if (fence_handle != VK_NULL_HANDLE && submit_count_ > 0) {
      submit_cmd_to_gpu(fence_handle);
      return true;
    }
    return false;
  }

  // Serialize recording to the shared command buffer.  The lock is left
  // unlocked when a fence is passed, since the caller then manages the
  // lock externally around a flush.
  std::unique_lock<std::mutex> cmd_lock;
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();

  cmd_.insert_barrier(pipeline_barrier);

  record_copy(cmd_, source, destination, copy_range, src_offset, dst_offset);

  submit_count_++;
  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmdSubmitFrequency) {
    submit_cmd_to_gpu(fence_handle);
    return true;
  }
  return false;
}

//
// Records a compute shader dispatch into the current command buffer.  If the
// number of submit_*_job calls exceeds the configured frequency, or if a
// fence is provided, the command buffer is submitted to the GPU for
// execution.  Returns true when the call resulted in a queue submission.
//
template <typename... Arguments>
inline bool Context::submit_compute_job(
    const ShaderInfo& shader,
    PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_work_group,
    const utils::uvec3& local_work_group_size,
    VkFence fence_handle,
    Arguments&&... arguments) {
  // Exit early when an argument has no memory attached.  With a fence
  // present, the pending command buffer must still be submitted so the
  // fence gets signaled.
  if (detail::any_arg_is_empty(arguments...)) {
    if (fence_handle != VK_NULL_HANDLE && submit_count_ > 0) {
      submit_cmd_to_gpu(fence_handle);
      return true;
    }
    return false;
  }

  // Serialize recording to the shared command buffer.  The lock is left
  // unlocked when a fence is passed, since the caller then manages the
  // lock externally around a flush.
  std::unique_lock<std::mutex> cmd_lock;
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();

  // Factor out template-parameter-independent code to limit code bloat.
  DescriptorSet descriptor_set =
      get_descriptor_set(shader, local_work_group_size);

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);

  register_shader_dispatch(
      descriptor_set, pipeline_barrier, shader, global_work_group);

  submit_count_++;
  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmdSubmitFrequency) {
    submit_cmd_to_gpu(fence_handle);
    return true;
  }

  return false;
}

//
// Host-side staging helpers.  Both wrap buffers allocated host-visible so
// CPU code can memcpy into / out of them.
//

class StorageBuffer final {
 private:
  Context* context_p_;
  DType dtype_;
  size_t numel_;
  size_t nbytes_;
  VulkanBuffer vulkan_buffer_;

 public:
  StorageBuffer(
      Context* context_p,
      const DType dtype,
      const size_t numel)
      : context_p_(context_p),
        dtype_(dtype),
        numel_(numel),
        nbytes_(numel_ * tensorplay::elementSize(dtype_)),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_staging_buffer(
            nbytes_)) {}

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&&) = default;
  StorageBuffer& operator=(StorageBuffer&&) = default;

  ~StorageBuffer() {
    context_p_->register_buffer_cleanup(vulkan_buffer_);
  }

  inline DType dtype() {
    return dtype_;
  }

  inline VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  inline size_t numel() {
    return numel_;
  }

  inline size_t nbytes() {
    return nbytes_;
  }
};

class UniformParamsBuffer final {
 private:
  Context* context_p_;
  size_t nbytes_;
  VulkanBuffer vulkan_buffer_;

 public:
  UniformParamsBuffer() : context_p_{nullptr}, vulkan_buffer_{} {}

  template <class Block>
  UniformParamsBuffer(Context* context_p, const Block& block)
      : context_p_(context_p),
        nbytes_(sizeof(block)),
        vulkan_buffer_(context_p_->adapter_ptr()->vma().create_params_buffer(block)) {}

  UniformParamsBuffer(const UniformParamsBuffer&);
  UniformParamsBuffer& operator=(const UniformParamsBuffer&);

  UniformParamsBuffer(UniformParamsBuffer&&) = default;
  UniformParamsBuffer& operator=(UniformParamsBuffer&&) = default;

  ~UniformParamsBuffer() {
    if (vulkan_buffer_) {
      context_p_->register_buffer_cleanup(vulkan_buffer_);
    }
  }

  VulkanBuffer& buffer() {
    return vulkan_buffer_;
  }

  template <class Block>
  void update(const Block& block) {
    if (sizeof(block) != nbytes_) {
      VK_THROW(
          "Attempted to update UniformParamsBuffer with data of different size");
    }
    // Fill the uniform buffer with data in block
    {
      MemoryMap mapping(vulkan_buffer_, MemoryAccessType::WRITE);
      Block* data_ptr = mapping.template data<Block>();

      *data_ptr = block;
    }
  }
};

bool available();

// The global context is retrieved using this function, where it is declared as
// a static local variable.
Context* context();

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
