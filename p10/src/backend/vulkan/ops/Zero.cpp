#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "Utils.h"
#include "../impl/Common.h"
#include "../api/ShaderRegistry.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor& zero_kernel(Tensor& self) {
  TP_CHECK(self.dim() <= 4, "Vulkan zero_ supports up to 4d tensors");

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);

  if (v_self.storage_type() == api::StorageType::TEXTURE_3D &&
      self.dtype() == DType::Float32) {
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(zero),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        v_self.extents(),
        // local work group size
        adaptive_work_group_size(v_self.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_self.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE));

    return self;
  }

  if (v_self.storage_type() == api::StorageType::BUFFER &&
      tensorplay::elementSize(self.dtype()) == 4u) {
    // The zero shader writes single-precision words; the all-zero bit
    // pattern is also a zero element for 4-byte integer payloads.
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const struct BlockB final {
      uint32_t buf_length;
    } blockb{n};
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL(buffer_zero), pipeline_barrier, {n, 1u, 1u}, {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        params.buffer());
    return self;
  }

  // All-zero bytes are valid zeros for every dtype; stream the payload
  // through the staging pipeline without format-specific shaders.
  std::vector<uint8_t> host(v_self.gpu_nbytes(), 0);
  utils::upload_host_bytes(v_self, host.data(), host.size());
  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ZeroKernels) {
  m.impl("zero_", &tensorplay::vulkan::ops::zero_kernel);
}

#endif /* USE_VULKAN */
