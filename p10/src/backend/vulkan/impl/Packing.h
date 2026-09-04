#pragma once

#ifdef USE_VULKAN

#include "../api/Context.h"
#include "../api/Tensor.h"
#include "../api/Utils.h"

namespace tensorplay {
namespace vulkan {
namespace packing {

//
// Staging converters between host-linear buffers and the GPU
// representations.  Texture-backed tensors stream through the
// nchw_to_image / image_to_nchw shaders (NC4HW channel packing); buffer
// storage streams through buffer_to_buffer.
//

api::ShaderInfo get_nchw_to_image_shader(const api::vTensor& v_dst);
api::ShaderInfo get_image_to_nchw_shader(const api::vTensor& v_src);

void record_nchw_to_image_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle);

bool record_image_to_nchw_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    api::vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle);

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle);

bool record_buffer_to_nchw_op(
    api::Context* const context,
    api::vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle);

//
// GPU-side relayout: moves the tensor payload between the packed-layout
// encodings without leaving the device.  Each returns a new texture-backed
// vTensor with the same logical sizes but the requested packed layout.
// Kernels that reduce along the K axis consume these encodings directly
// (one texel lane per reduction step).
//

api::vTensor convert_image_channels_packed_to_width_packed(
    const api::vTensor& v_input);

api::vTensor convert_image_channels_packed_to_height_packed(
    const api::vTensor& v_input);

} // namespace packing
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
