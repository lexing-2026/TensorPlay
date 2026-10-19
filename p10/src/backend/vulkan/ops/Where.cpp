#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"

#include <TypeProperties.h>
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

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

Tensor where_kernel(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  TP_CHECK(
      condition.dtype() == DType::Bool,
      "Vulkan where expects a Bool condition");

  // Both branches move through one element vocabulary: the type that holds the
  // pair decides, and a side narrower than it is converted first.  Leaving the
  // pair mixed would read a texel in the wrong element width, which
  // reinterprets its bits instead of converting them.
  const DType vocabulary = tensorplay::native::result_type(self, other);
  TP_CHECK(
      vocabulary == DType::Float32 || vocabulary == DType::Float16 ||
          vocabulary == DType::Int32,
      "Vulkan where selects between Float32, Float16 and Int32 branches");
  const Tensor self_branch =
      self.dtype() == vocabulary ? self : self.to(vocabulary);
  const Tensor other_branch =
      other.dtype() == vocabulary ? other : other.to(vocabulary);
  validate_operand(self_branch, "where");
  validate_operand(other_branch, "where");
  TP_CHECK(
      condition.shape() == self_branch.shape() &&
          self_branch.shape() == other_branch.shape(),
      "Vulkan where requires equal-shaped operands (broadcast at the "
      "caller)");

  // An Int32 branch pair keeps word precision; anything else travels as a
  // float plane.
  const bool int_branches = vocabulary == DType::Int32;

  api::Context* const context = api::context();

  api::vTensor v_cond = convert(condition);
  api::vTensor v_input = convert(self_branch);
  api::vTensor v_other = convert(other_branch);

  // Linear payloads (shapes beyond the 3-D image axis limit) select through
  // buffer-to-buffer variants: the Bool condition is one byte per element and
  // rides a packed-word buffer, the operands are plain row-major memory.
  // Texel-stored operands are gathered into dense buffers first, because a
  // texel's lanes are not adjacent in the logical order a buffer reader walks.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    v_cond = to_buffer_storage_dispatch(context, v_cond);
    v_other = to_buffer_storage_dispatch(context, v_other);
    // A buffer-stored input needs a buffer-stored output: the default
    // allocation is a texture, whose image the linear variant cannot write.
    api::vTensor v_output{
        context,
        v_input.sizes(),
        vocabulary,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const char* buffer_shader = nullptr;
    uint32_t lanes_per_word = 4u;
    if (vocabulary == DType::Int32) {
      buffer_shader = "where_i32_buffer";
    } else if (vocabulary == DType::Float32) {
      buffer_shader = "where_buffer";
    } else if (vocabulary == DType::Float16) {
      // Half lanes pack two to a word, so a dispatch covers half as many
      // words as the four-lane vocabularies do.
      buffer_shader = "where_half_buffer";
      lanes_per_word = 2u;
    } else {
      TP_THROW(
          NotImplementedError,
          "Vulkan where over buffer supports Float32, Float16 and Int32 only");
    }
    const uint32_t n = static_cast<uint32_t>(v_input.numel());
    const struct WhereBufferBlock {
      uint32_t buf_length;
      uint32_t fill0;
      int32_t fill;
    } block{n, 0u, 0};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader), pipeline_barrier,
        {(n + lanes_per_word - 1u) / lanes_per_word, 1u, 1u}, {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_cond.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_input.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_other.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_output);
  }

  api::vTensor v_output{context, v_input.sizes(), vocabulary};

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
    const Tensor& condition, const Scalar& self, const Tensor& other) {
  Tensor folded = full_kernel(
      static_cast<std::vector<int64_t>>(other.shape()),
      self,
      other.dtype(),
      other.device(),
      false);
  return where_kernel(condition, folded, other);
}

Tensor where_scalar_other_kernel(
    const Tensor& condition, const Tensor& self, const Scalar& other) {
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
