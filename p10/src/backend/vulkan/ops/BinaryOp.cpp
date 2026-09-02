#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../impl/Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

struct BlockB final {
  uint32_t buf_length;
  uint32_t fill0;
  float alpha;
};

/*
 * Shared implementation of the element-wise binary tensor op:
 * out = OP(self, other, alpha), with broadcasting over the other operand.
 */
Tensor binary_op_tensor(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan binary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);
  api::vTensor v_other = convert(other_arg);

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const struct BlockB final {
      uint32_t buf_length;
      uint32_t fill0;
      float alpha;
    } blockb{n, 0u, alpha.to<float>()};
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name), pipeline_barrier, {n, 1u, 1u},
        {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      alpha.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(shader_name),
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& binary_op_tensor_inplace(
    const char* shader_name,
    const char* buffer_shader_name,
    Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan binary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);
  api::vTensor v_other = convert(other_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const struct BlockB final {
      uint32_t buf_length;
      uint32_t fill0;
      float alpha;
    } blockb{n, 0u, alpha.to<float>()};
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name), pipeline_barrier, {n, 1u, 1u},
        {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return self_arg;
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      alpha.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(shader_name),
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

/*
 * Element-wise binary op against a broadcast scalar.
 */
Tensor binary_op_scalar(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan binary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const bool is_buffer = v_output.storage_type() == api::StorageType::BUFFER;

  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      other.to<float>() * alpha.to<float>(),
  };

  if (is_buffer) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    api::UniformParamsBuffer paramsb(
        context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name), pipeline_barrier, {n, 1u, 1u},
        {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        paramsb.buffer());
    return convert(v_output);
  }

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(shader_name),
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

Tensor& binary_op_scalar_inplace(
    const char* shader_name,
    const char* buffer_shader_name,
    Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  TP_CHECK(
      self_arg.dtype() == DType::Float32,
      "Vulkan binary ops support Float32 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    api::UniformParamsBuffer params(
        context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_name), pipeline_barrier, {n, 1u, 1u},
        {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        params.buffer());
    return self_arg;
  }

  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      make_whcn_ivec4(v_self.sizes()),
      other.to<float>() * alpha.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(shader_name),
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

Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor("add", "buffer_add", self, other, alpha);
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor("sub", "buffer_sub", self, other, alpha);
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("mul", "buffer_mul", self, other, Scalar(1.0));
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("div", "buffer_div", self, other, Scalar(1.0));
}

Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self, other, alpha);
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self,
                          Scalar(-other.toDouble()), alpha);
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self, other,
                          Scalar(1.0));
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self,
                          Scalar(1.0 / other.toDouble()), Scalar(1.0));
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor_inplace("addinplace", "buffer_addinplace", self,
                                  other, alpha);
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor_inplace("subinplace", "buffer_subinplace", self,
                                  other, alpha);
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("mulinplace", "buffer_mulinplace", self,
                                  other, Scalar(1.0));
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("divinplace", "buffer_divinplace", self,
                                  other, Scalar(1.0));
}

Tensor& add_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar_inplace("add_scalarinplace", "buffer_add_scalarinplace",
                                  self, other, alpha);
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, Scalar other) {
  return binary_op_scalar_inplace("mul_scalarinplace", "buffer_mul_scalarinplace",
                                  self, other, Scalar(1.0));
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, BinaryOpKernels) {
  m.impl("add.Tensor", &tensorplay::vulkan::ops::add_kernel);
  m.impl("add.Scalar", &tensorplay::vulkan::ops::add_scalar_kernel);
  m.impl("add_.Tensor", &tensorplay::vulkan::ops::add_inplace_kernel);
  m.impl("add_.Scalar", &tensorplay::vulkan::ops::add_scalar_inplace_kernel);
  m.impl("sub.Tensor", &tensorplay::vulkan::ops::sub_kernel);
  m.impl("sub.Scalar", &tensorplay::vulkan::ops::sub_scalar_kernel);
  m.impl("sub_.Tensor", &tensorplay::vulkan::ops::sub_inplace_kernel);
  m.impl("mul.Tensor", &tensorplay::vulkan::ops::mul_kernel);
  m.impl("mul.Scalar", &tensorplay::vulkan::ops::mul_scalar_kernel);
  m.impl("mul_.Tensor", &tensorplay::vulkan::ops::mul_inplace_kernel);
  m.impl("div.Tensor", &tensorplay::vulkan::ops::div_kernel);
  m.impl("div.Scalar", &tensorplay::vulkan::ops::div_scalar_kernel);
  m.impl("div_.Tensor", &tensorplay::vulkan::ops::div_inplace_kernel);
}

#endif /* USE_VULKAN */
