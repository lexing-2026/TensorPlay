#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"

#include <Utils.h>

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_float_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

struct WhereBlock final {
  ivec4 out_sizes;
  int c_depth;
  int fill;
};

} // namespace

Tensor where_kernel(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  validate_float_operand(self, "where");
  validate_float_operand(other, "where");
  TP_CHECK(
      condition.dtype() == DType::Bool,
      "Vulkan where expects a Bool condition");
  TP_CHECK(
      condition.shape() == self.shape() && self.shape() == other.shape(),
      "Vulkan where requires equal-shaped operands (broadcast at the "
      "caller)");

  api::Context* const context = api::context();

  api::vTensor v_cond = convert(condition);
  api::vTensor v_input = convert(self);
  api::vTensor v_other = convert(other);
  api::vTensor v_output{context, v_input.sizes(), DType::Float32};

  const struct WhereBlock block{
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_output.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(where), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_cond.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

// Scalar variants fold into full tensors and reuse the tensor form.
Tensor where_scalar_self_kernel(
    const Tensor& condition, Scalar self, const Tensor& other) {
  Tensor folded = full_kernel(
      static_cast<std::vector<int64_t>>(other.shape()),
      self,
      DType::Float32,
      other.device(),
      false);
  return where_kernel(condition, folded, other);
}

Tensor where_scalar_other_kernel(
    const Tensor& condition, const Tensor& self, Scalar other) {
  Tensor folded = full_kernel(
      static_cast<std::vector<int64_t>>(self.shape()),
      other,
      DType::Float32,
      self.device(),
      false);
  return where_kernel(condition, self, folded);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, WhereKernels) {
  m.impl("where.self", &tensorplay::vulkan::ops::where_kernel);
  m.impl(
      "where.ScalarSelf",
      &tensorplay::vulkan::ops::where_scalar_self_kernel);
  m.impl(
      "where.ScalarOther",
      &tensorplay::vulkan::ops::where_scalar_other_kernel);
}

#endif /* USE_VULKAN */
