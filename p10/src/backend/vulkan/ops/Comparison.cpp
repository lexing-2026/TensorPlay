#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <Utils.h>

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Int32 ||
          t.dtype() == DType::Int8 || t.dtype() == DType::UInt8 ||
          t.dtype() == DType::Bool,
      std::string("Vulkan ") + name + " has an unsupported dtype");
  TP_CHECK(
      t.dim() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");
}

struct CompareBlock final {
  ivec4 out_sizes;
  int c_depth;
  int fill;
};

// Elementwise comparison driver for equal-shaped tensor operands.
Tensor compare_impl(
    const Tensor& self,
    const Tensor& other,
    const char* kernel_name,
    const char* name) {
  validate_operand(self, name);
  validate_operand(other, name);
  TP_CHECK(
      self.shape() == other.shape(),
      std::string("Vulkan ") + name +
          " requires equal-shaped operands (broadcast at the caller)");

  TP_CHECK(self.dtype() == other.dtype(),
           "Vulkan comparison requires matching operand dtypes");
  const char* suffix = self.dtype() == DType::Int32 ? "_i32"
      : self.dtype() == DType::UInt8 ? "_u8"
      : self.dtype() == DType::Bool || self.dtype() == DType::Int8 ? "_i8" : "";
  const std::string shader_name = std::string(kernel_name) + suffix;
  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_other = convert(other);
  api::vTensor v_output{context, v_input.sizes(), DType::Bool};

  const struct CompareBlock block{
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_output.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader_name.c_str()), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor compare_scalar(const Tensor& self, Scalar other, const char* name) {
  validate_operand(self, name);
  const char* suffix = self.dtype() == DType::Float32 ? "" :
      self.dtype() == DType::UInt8 ? "_u8" : "_i32";
  const std::string shader = std::string(name) + "_scalar" + suffix;
  api::Context* context = api::context();
  api::vTensor input = convert(self);
  api::vTensor output{context, input.sizes(), DType::Bool};
  const struct Block final {
    float value_float;
    int32_t value_int;
    uint32_t value_uint;
    int32_t fill;
  } block{self.dtype() == DType::Float32 ? other.to<float>() : 0.0f,
          self.dtype() != DType::Float32 && self.dtype() != DType::UInt8 ?
              other.to<int32_t>() : 0,
          self.dtype() == DType::UInt8 ? other.to<uint32_t>() : 0u, 0};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader.c_str()), barrier, output.extents(),
      adaptive_work_group_size(output.extents()), VK_NULL_HANDLE,
      output.image(barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
      input.image(barrier, api::PipelineStage::COMPUTE), params.buffer());
  return convert(output);
}

} // namespace

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "eq", "eq");
}
Tensor eq_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "eq");
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "ne", "ne");
}
Tensor ne_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "ne");
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "lt", "lt");
}
Tensor lt_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "lt");
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "le", "le");
}
Tensor le_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "le");
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "gt", "gt");
}
Tensor gt_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "gt");
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "ge", "ge");
}
Tensor ge_scalar_kernel(const Tensor& self, Scalar other) {
  return compare_scalar(self, other, "ge");
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ComparisonKernels) {
  m.impl("eq.Tensor", &tensorplay::vulkan::ops::eq_tensor_kernel);
  m.impl("ne.Tensor", &tensorplay::vulkan::ops::ne_tensor_kernel);
  m.impl("lt.Tensor", &tensorplay::vulkan::ops::lt_tensor_kernel);
  m.impl("le.Tensor", &tensorplay::vulkan::ops::le_tensor_kernel);
  m.impl("gt.Tensor", &tensorplay::vulkan::ops::gt_tensor_kernel);
  m.impl("ge.Tensor", &tensorplay::vulkan::ops::ge_tensor_kernel);
  m.impl("eq.Scalar", &tensorplay::vulkan::ops::eq_scalar_kernel);
  m.impl("ne.Scalar", &tensorplay::vulkan::ops::ne_scalar_kernel);
  m.impl("lt.Scalar", &tensorplay::vulkan::ops::lt_scalar_kernel);
  m.impl("le.Scalar", &tensorplay::vulkan::ops::le_scalar_kernel);
  m.impl("gt.Scalar", &tensorplay::vulkan::ops::gt_scalar_kernel);
  m.impl("ge.Scalar", &tensorplay::vulkan::ops::ge_scalar_kernel);
}

#endif /* USE_VULKAN */
