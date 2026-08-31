#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor clone_kernel(const Tensor& self) {
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    api::PipelineBarrier pipeline_barrier{};
    context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
        pipeline_barrier,
        v_self.buffer(pipeline_barrier, api::PipelineStage::TRANSFER,
                      api::MemoryAccessType::READ),
        v_output.buffer(pipeline_barrier, api::PipelineStage::TRANSFER,
                        api::MemoryAccessType::WRITE),
        {static_cast<uint32_t>(v_self.nbytes()), 0u, 0u},
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
    return convert(v_output);
  }

  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanImage, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // images
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      v_self.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, CloneKernels) {
  m.impl("clone", &tensorplay::vulkan::ops::clone_kernel);
}

#endif /* USE_VULKAN */
