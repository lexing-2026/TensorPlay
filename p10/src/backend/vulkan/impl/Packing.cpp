#ifdef USE_VULKAN

#include "Packing.h"
#include "Common.h"
#include "../ops/Common.h"
#include "../api/ShaderRegistry.h"

namespace tensorplay {
namespace vulkan {
namespace packing {

api::ShaderInfo get_nchw_to_image_shader(const api::vTensor& v_dst) {
  if (v_dst.dtype() == DType::Float32) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype for texture packing!");
  }
}

api::ShaderInfo get_image_to_nchw_shader(const api::vTensor& v_src) {
  if (v_src.dtype() == DType::Float32) {
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype for texture unpacking!");
  }
}

namespace {

struct ToFromTextureParams final {
  api::utils::ivec3 extents;
  int32_t planeSize;
  api::utils::ivec2 channelInfo;
};

} // namespace

void record_nchw_to_image_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle) {
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  int32_t height =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Height>(v_dst));
  int32_t width =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Width>(v_dst));
  int32_t channels =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Channel>(v_dst));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4u);

  ToFromTextureParams block{
      api::utils::ivec3(
          v_dst.extents()[0u], v_dst.extents()[1u], v_dst.extents()[2u]),
      plane_size,
      {c_depth, channels},
  };

  api::UniformParamsBuffer params(context, block);
  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      // params buffer
      params.buffer());
}

bool record_image_to_nchw_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    api::vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle) {
  api::utils::uvec3 global_size = v_src.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  int32_t height =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Height>(v_src));
  int32_t width =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Width>(v_src));
  int32_t channels =
      static_cast<int32_t>(ops::get_dim<ops::Dim4D::Channel>(v_src));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4u);

  ToFromTextureParams block{
      api::utils::ivec3(
          v_src.extents()[0u], v_src.extents()[1u], v_src.extents()[2u]),
      plane_size,
      {c_depth, channels},
  };

  api::UniformParamsBuffer params(context, block);

  return context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      dst_buffer,
      // params buffer
      params.buffer());
}

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle) {
  uint32_t gpu_buf_len = api::utils::safe_downcast_to_u32(
      static_cast<int64_t>(v_dst.gpu_numel()));

  api::utils::uvec3 global_size = {gpu_buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_dst.get_cpu_buffer_metadata());

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(buffer_to_buffer),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_dst.buffer_metadata(),
      src_buffer,
      cpu_buffer_metadata.buffer());
}

bool record_buffer_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle) {
  uint32_t buf_len = api::utils::safe_downcast_to_u32(
      static_cast<int64_t>(v_src.numel()));

  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());

  return context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(buffer_to_buffer),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

} // namespace packing
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
