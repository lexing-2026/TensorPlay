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

/*
 * Branch payload vocabulary: float planes sample through the float shader,
 * Int32 planes ride the `where_i32` twin with iimage loads.  Both operands
 * share one vocabulary, so mixed float/int branches stage the Int32 side to
 * Float32 on the host first.
 */
void validate_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          t.dtype() == DType::Int32,
      std::string("Vulkan ") + name +
          " supports Float32, Float16 and Int32 tensors only");
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
  validate_operand(self, "where");
  validate_operand(other, "where");
  TP_CHECK(
      condition.dtype() == DType::Bool,
      "Vulkan where expects a Bool condition");
  TP_CHECK(
      condition.shape() == self.shape() && self.shape() == other.shape(),
      "Vulkan where requires equal-shaped operands (broadcast at the "
      "caller)");

  // One shared element vocabulary for both branches; an Int32 branch pair
  // keeps word precision, anything mixed casts the int side to Float32.
  const bool int_branches =
      self.dtype() == DType::Int32 && other.dtype() == DType::Int32;

  api::Context* const context = api::context();

  api::vTensor v_cond = convert(condition);
  api::vTensor v_input = convert(self);
  api::vTensor v_other = convert(other);
  api::vTensor v_output{context, v_input.sizes(), self.dtype()};

  const struct WhereBlock block{
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_output.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      int_branches ? VK_KERNEL(where_i32) : VK_KERNEL(where),
      pipeline_barrier, v_output.extents(),
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

// Scalar variants fold into full tensors and reuse the tensor form.  The
// folded tensor takes the branch dtype so the pair stays in one vocabulary.
Tensor where_scalar_self_kernel(
    const Tensor& condition, Scalar self, const Tensor& other) {
  Tensor folded = full_kernel(
      static_cast<std::vector<int64_t>>(other.shape()),
      self,
      other.dtype(),
      other.device(),
      false);
  return where_kernel(condition, folded, other);
}

Tensor where_scalar_other_kernel(
    const Tensor& condition, const Tensor& self, Scalar other) {
  Tensor folded = full_kernel(
      static_cast<std::vector<int64_t>>(self.shape()),
      other,
      self.dtype(),
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
