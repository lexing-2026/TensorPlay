#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../impl/Common.h"
#include "../api/Context.h"
#include "../api/Shader.h"
#include "../api/ShaderRegistry.h"

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const char* buffer_shader_name) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan unary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    const struct BlockB final {
      uint32_t buf_length;
    } blockb{
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel())),
    };
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name),
        pipeline_barrier,
        {safe_downcast_to_u32(static_cast<int64_t>(v_output.numel())), 1u, 1u},
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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

Tensor& unary_op_(
    Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const char* buffer_shader_name) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan unary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const struct BlockB final {
      uint32_t buf_length;
    } blockb{
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel())),
    };
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name),
        pipeline_barrier,
        {safe_downcast_to_u32(static_cast<int64_t>(v_self.numel())), 1u, 1u},
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        params.buffer());
    return self_arg;
  }

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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

Tensor exp_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(exp), "buffer_exp");
}

Tensor& exp_inplace_kernel(Tensor& self) {
  return unary_op_(self, VK_KERNEL(expinplace), "buffer_expinplace");
}

Tensor sqrt_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(sqrt), "buffer_sqrt");
}

Tensor& sqrt_inplace_kernel(Tensor& self) {
  return unary_op_(self, VK_KERNEL(sqrtinplace), "buffer_sqrtinplace");
}

Tensor log_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(log), "buffer_log");
}

Tensor& log_inplace_kernel(Tensor& self) {
  return unary_op_(self, VK_KERNEL(loginplace), "buffer_loginplace");
}

Tensor abs_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(abs), "buffer_abs");
}

Tensor neg_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(neg), "buffer_neg");
}

Tensor floor_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(floor), "buffer_floor");
}

Tensor sin_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(sin), "buffer_sin");
}

Tensor cos_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(cos), "buffer_cos");
}

Tensor tanh_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(tanh), "buffer_tanh");
}

Tensor sigmoid_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(sigmoid), "buffer_sigmoid");
}

Tensor relu_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(relu), "buffer_relu");
}

Tensor rsqrt_kernel(const Tensor& self) {
  return unary_op(self, VK_KERNEL(rsqrt), "buffer_rsqrt");
}

Tensor& relu_inplace_kernel(Tensor& self) {
  return unary_op_(self, VK_KERNEL(reluinplace), "buffer_reluinplace");
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, UnaryOpKernels) {
  m.impl("exp", &tensorplay::vulkan::ops::exp_kernel);
  m.impl("exp_", &tensorplay::vulkan::ops::exp_inplace_kernel);
  m.impl("sqrt", &tensorplay::vulkan::ops::sqrt_kernel);
  m.impl("sqrt_", &tensorplay::vulkan::ops::sqrt_inplace_kernel);
  m.impl("log", &tensorplay::vulkan::ops::log_kernel);
  m.impl("log_", &tensorplay::vulkan::ops::log_inplace_kernel);
  m.impl("abs", &tensorplay::vulkan::ops::abs_kernel);
  m.impl("neg", &tensorplay::vulkan::ops::neg_kernel);
  m.impl("floor", &tensorplay::vulkan::ops::floor_kernel);
  m.impl("sin", &tensorplay::vulkan::ops::sin_kernel);
  m.impl("cos", &tensorplay::vulkan::ops::cos_kernel);
  m.impl("tanh", &tensorplay::vulkan::ops::tanh_kernel);
  m.impl("sigmoid", &tensorplay::vulkan::ops::sigmoid_kernel);
  m.impl("relu", &tensorplay::vulkan::ops::relu_kernel);
  m.impl("relu_", &tensorplay::vulkan::ops::relu_inplace_kernel);
  m.impl("rsqrt", &tensorplay::vulkan::ops::rsqrt_kernel);
}

#endif /* USE_VULKAN */
