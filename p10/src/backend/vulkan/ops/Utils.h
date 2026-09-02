#pragma once

#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"
#include "../api/Resource.h"
#include "../api/Tensor.h"

namespace tensorplay {
namespace vulkan {
namespace ops {
namespace utils {

Tensor nchw_to_nc4hw(const Tensor&);
Tensor create_staging_tensor(const api::vTensor&);
Tensor nc4hw_to_nchw(const Tensor&, IntArrayRef);

void copy_buffer_to_vtensor(
    api::VulkanBuffer&,
    api::vTensor&,
    api::PipelineBarrier&);

void copy_buffer_to_buffer(
    api::Context*,
    api::StorageBuffer&,
    api::StorageBuffer&,
    VkFence);

void copy_vtensor_to_buffer(
    api::vTensor&,
    api::VulkanBuffer&,
    api::PipelineBarrier&,
    VkFence);

void pack_staging_to_vtensor(api::VulkanBuffer&, api::vTensor&);
bool pack_vtensor_to_staging(api::vTensor&, api::VulkanBuffer&, VkFence);

// Raw byte transfer into the payload: staging -> texture (vkCopyBufferToImage)
// or staging -> buffer (vkCopyBuffer).  Valid for every supported VkFormat.
void copy_staging_to_vtensor(api::StorageBuffer&, api::vTensor&);

void upload_host_bytes(api::vTensor&, const void*, size_t);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
