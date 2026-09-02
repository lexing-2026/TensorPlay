#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../impl/Common.h"
#include "../api/ShaderRegistry.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

Tensor clamp(
    const Tensor& self_arg,
    const std::optional<Scalar>& min_arg,
    const std::optional<Scalar>& max_arg) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan clamp supports Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    ivec4 extents;
    // clamp range
    vec2 clamp;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      {min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
       max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity()},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(clamp),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& clamp_(
    Tensor& self_arg,
    const std::optional<Scalar>& min_arg,
    const std::optional<Scalar>& max_arg) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan clamp_ supports Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  const struct Block final {
    ivec4 extents;
    // clamp range
    vec2 clamp;
  } block{
      make_whcn_ivec4(v_self.sizes()),
      {min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
       max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity()},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(clampinplace),
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

} // namespace

Tensor clamp_kernel(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
  return clamp(self, min, max);
}

Tensor& clamp_inplace_kernel(
    Tensor& self,
    std::optional<Scalar> min,
    std::optional<Scalar> max) {
  return clamp_(self, min, max);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ClampKernels) {
  m.impl("clamp", &tensorplay::vulkan::ops::clamp_kernel);
  m.impl("clamp_", &tensorplay::vulkan::ops::clamp_inplace_kernel);
}

#endif /* USE_VULKAN */
