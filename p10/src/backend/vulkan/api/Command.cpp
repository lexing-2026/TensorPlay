#ifdef USE_VULKAN

#include "Command.h"
#include "Utils.h"
#include "Exception.h"

#include <utility>

namespace tensorplay {
namespace vulkan {
namespace api {

//
// CommandBuffer
//

CommandBuffer::CommandBuffer(
    VkCommandBuffer handle,
    const VkCommandBufferUsageFlags flags)
    : handle_(handle),
      flags_(flags),
      state_(CommandBuffer::State::NEW),
      bound_{} {}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : handle_(other.handle_),
      flags_(other.flags_),
      state_(CommandBuffer::State::INVALID),
      bound_(other.bound_) {
  other.handle_ = VK_NULL_HANDLE;
  other.bound_.reset();
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
  handle_ = other.handle_;
  flags_ = other.flags_;
  state_ = other.state_;
  bound_ = other.bound_;

  other.handle_ = VK_NULL_HANDLE;
  other.bound_.reset();
  other.state_ = CommandBuffer::State::INVALID;

  return *this;
}

void CommandBuffer::begin() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::NEW,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not NEW.");

  const VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      nullptr,
      flags_,
      nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(handle_, &begin_info));
  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::end() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING ||
          state_ == CommandBuffer::State::SUBMITTED,
      "Vulkan CommandBuffer: called end() on a command buffer whose state "
      "is not RECORDING or SUBMITTED.");

  if (state_ == CommandBuffer::State::RECORDING) {
    VK_CHECK(vkEndCommandBuffer(handle_));
  }
  state_ = CommandBuffer::State::READY;
}

void CommandBuffer::bind_pipeline(
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    const utils::uvec3 local_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called bind_pipeline() on a command buffer whose state "
      "is not RECORDING.");

  if (pipeline != bound_.pipeline) {
    vkCmdBindPipeline(handle_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    bound_.pipeline = pipeline;
  }

  bound_.pipeline_layout = pipeline_layout;
  bound_.local_workgroup_size = local_workgroup_size;

  state_ = CommandBuffer::State::PIPELINE_BOUND;
}

void CommandBuffer::bind_descriptors(VkDescriptorSet descriptors) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::PIPELINE_BOUND,
      "Vulkan CommandBuffer: called bind_descriptors() on a command buffer whose state "
      "is not PIPELINE_BOUND.");

  if (descriptors != bound_.descriptors) {
    vkCmdBindDescriptorSets(
        handle_, // commandBuffer
        VK_PIPELINE_BIND_POINT_COMPUTE, // pipelineBindPoint
        bound_.pipeline_layout, // layout
        0u, // firstSet
        1u, // descriptorSetCount
        &descriptors, // pDescriptorSets
        0u, // dynamicOffsetCount
        nullptr); // pDynamicOffsets
  }

  bound_.descriptors = descriptors;

  state_ = CommandBuffer::State::DESCRIPTORS_BOUND;
}

void CommandBuffer::insert_barrier(PipelineBarrier& pipeline_barrier) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::DESCRIPTORS_BOUND ||
          state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called insert_barrier() on a command buffer whose state "
      "is not DESCRIPTORS_BOUND or RECORDING.");

  if (pipeline_barrier) {
    if (!pipeline_barrier.buffer_barrier_handles.empty()) {
      pipeline_barrier.buffer_barrier_handles.clear();
    }
    for (const api::BufferMemoryBarrier& memory_barrier :
         pipeline_barrier.buffers) {
      pipeline_barrier.buffer_barrier_handles.push_back(memory_barrier.handle);
    }

    if (!pipeline_barrier.image_barrier_handles.empty()) {
      pipeline_barrier.image_barrier_handles.clear();
    }
    for (const api::ImageMemoryBarrier& memory_barrier :
         pipeline_barrier.images) {
      pipeline_barrier.image_barrier_handles.push_back(memory_barrier.handle);
    }
    vkCmdPipelineBarrier(
        handle_, // commandBuffer
        pipeline_barrier.stage.src, // srcStageMask
        pipeline_barrier.stage.dst, // dstStageMask
        0u, // dependencyFlags
        0u, // memoryBarrierCount
        nullptr, // pMemoryBarriers
        pipeline_barrier.buffers.size(), // bufferMemoryBarrierCount
        !pipeline_barrier.buffers.empty()
            ? pipeline_barrier.buffer_barrier_handles.data()
            : nullptr, // pMemoryBarriers
        pipeline_barrier.images.size(), // imageMemoryBarrierCount
        !pipeline_barrier.images.empty()
            ? pipeline_barrier.image_barrier_handles.data()
            : nullptr); // pImageMemoryBarriers
  }

  state_ = CommandBuffer::State::BARRIERS_INSERTED;
}

void CommandBuffer::dispatch(const utils::uvec3& global_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called dispatch() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  vkCmdDispatch(
      handle_,
      utils::div_up(
          global_workgroup_size[0u], bound_.local_workgroup_size[0u]),
      utils::div_up(
          global_workgroup_size[1u], bound_.local_workgroup_size[1u]),
      utils::div_up(
          global_workgroup_size[2u], bound_.local_workgroup_size[2u]));

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_buffer(
    const api::VulkanBuffer& source,
    const api::VulkanBuffer& destination,
    const utils::uvec3& copy_range,
    const utils::uvec3& src_offset,
    const utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkBufferCopy copy_details{
      src_offset[0u], // srcOffset
      dst_offset[0u], // dstOffset
      copy_range[0u], // size
  };

  vkCmdCopyBuffer(
      handle_, source.handle(), destination.handle(), 1u, &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_texture(
    const api::VulkanImage& source,
    const api::VulkanImage& destination,
    const utils::uvec3& copy_range,
    const utils::uvec3& src_offset,
    const utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageCopy copy_details{
      src_subresource_layers, // srcSubresource
      {
          static_cast<int32_t>(src_offset[0u]), // x
          static_cast<int32_t>(src_offset[1u]), // y
          static_cast<int32_t>(src_offset[2u]), // z
      }, // srcOffset
      dst_subresource_layers, // dstSubresource
      {
          static_cast<int32_t>(dst_offset[0u]), // x
          static_cast<int32_t>(dst_offset[1u]), // y
          static_cast<int32_t>(dst_offset[2u]), // z
      }, // dstOffset
      {
          copy_range[0u], // width
          copy_range[1u], // height
          copy_range[2u], // depth
      }, // extent
  };

  vkCmdCopyImage(
      handle_,
      source.handle(),
      source.layout(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_buffer(
    const api::VulkanImage& source,
    const api::VulkanBuffer& destination,
    const utils::uvec3& copy_range,
    const utils::uvec3& src_offset,
    const utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkBufferImageCopy copy_details{
      dst_offset[0u], // bufferOffset
      0u, // bufferRowLength
      0u, // bufferImageHeight
      {
          VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
          0u, // mipLevel
          0u, // baseArrayLayer
          1u, // layerCount
      }, // imageSubresource
      {
          static_cast<int32_t>(src_offset[0u]), // x
          static_cast<int32_t>(src_offset[1u]), // y
          static_cast<int32_t>(src_offset[2u]), // z
      }, // imageOffset
      {
          copy_range[0u], // width
          copy_range[1u], // height
          copy_range[2u], // depth
      }, // imageExtent
  };

  vkCmdCopyImageToBuffer(
      handle_, source.handle(), source.layout(), destination.handle(), 1u, &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_texture(
    const api::VulkanBuffer& source,
    const api::VulkanImage& destination,
    const utils::uvec3& copy_range,
    const utils::uvec3& src_offset,
    const utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkBufferImageCopy copy_details{
      src_offset[0u], // bufferOffset
      0u, // bufferRowLength
      0u, // bufferImageHeight
      {
          VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
          0u, // mipLevel
          0u, // baseArrayLayer
          1u, // layerCount
      }, // imageSubresource
      {
          static_cast<int32_t>(dst_offset[0u]), // x
          static_cast<int32_t>(dst_offset[1u]), // y
          static_cast<int32_t>(dst_offset[2u]), // z
      }, // imageOffset
      {
          copy_range[0u], // width
          copy_range[1u], // height
          copy_range[2u], // depth
      }, // imageExtent
  };

  vkCmdCopyBufferToImage(
      handle_, source.handle(), destination.handle(), destination.layout(), 1u, &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::write_timestamp(VkQueryPool query_pool, const uint32_t query_idx) const {
  vkCmdWriteTimestamp(
      handle_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      query_pool,
      query_idx);
}

void CommandBuffer::reset_querypool(
    VkQueryPool query_pool,
    const uint32_t first_query_idx,
    const uint32_t query_count) const {
  vkCmdResetQueryPool(handle_, query_pool, first_query_idx, query_count);
}

VkCommandBuffer CommandBuffer::get_submit_handle(const bool final_use) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::READY,
      "Vulkan CommandBuffer: requested submit handle with a command buffer "
      "in the wrong state!");

  const VkCommandBuffer handle = handle_;

  // Buffers handed out for one-time use (the default) are detached from the
  // Context after submission: the next recording must start on a fresh
  // buffer because recording into an ended VkCommandBuffer is invalid.  The
  // pool recycles the underlying resource at flush time.
  if (!is_reusable() || final_use) {
    invalidate();
  }
  state_ = CommandBuffer::State::SUBMITTED;

  return handle;
}

//
// CommandPool
//

CommandPool::CommandPool(
    VkDevice device,
    const uint32_t queue_family_idx,
    const CommandPoolConfig& config)
    : device_(device),
      queue_family_idx_(queue_family_idx),
      pool_{VK_NULL_HANDLE},
      config_(config),
      mutex_{},
      buffers_{},
      in_use_(0u) {
  const VkCommandPoolCreateInfo command_pool_create_info{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      queue_family_idx_, // queueFamilyIndex
  };

  VK_CHECK(
      vkCreateCommandPool(device_, &command_pool_create_info, nullptr, &pool_));
  VK_CHECK_COND(pool_, "Invalid command pool handle!");

  allocate_new_batch(config_.cmdPoolInitialSize);
}

CommandPool::~CommandPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }

  vkDestroyCommandPool(device_, pool_, nullptr);

  pool_ = VK_NULL_HANDLE;
}

CommandBuffer CommandPool::get_new_cmd(bool reusable) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (in_use_ >= buffers_.size()) {
    allocate_new_batch(config_.cmdPoolBatchSize);
  }

  return CommandBuffer{
      buffers_[in_use_++],
      reusable ? 0u : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
}

void CommandPool::flush() {
  std::lock_guard<std::mutex> lock(mutex_);

  // All in-flight submissions have completed by the time the pool is
  // flushed, so every command buffer handed out so far can be recycled in
  // bulk.
  if (in_use_ > 0u) {
    VK_CHECK(vkResetCommandPool(device_, pool_, 0u));
  }

  in_use_ = 0u;
}

void CommandPool::allocate_new_batch(const uint32_t batch_size) {
  const VkCommandBufferAllocateInfo cmd_alloc_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // commandPool
      VK_COMMAND_BUFFER_LEVEL_PRIMARY, // level
      batch_size, // commandBufferCount
  };

  const size_t offset = buffers_.size();
  buffers_.resize(offset + batch_size);

  VK_CHECK(vkAllocateCommandBuffers(
      device_, &cmd_alloc_info, buffers_.data() + offset));
}

} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
