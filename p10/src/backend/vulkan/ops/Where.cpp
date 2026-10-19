#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"

#include <TypeProperties.h>
#include <Utils.h>

#include <utility>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

/*
 * Branch payload vocabulary: float planes sample through the float shader,
 * Int32 planes ride the `where_i32` twin with iimage loads.  Both operands
 * share one vocabulary, so a mixed pair is converted to the type that holds it.
 */
void validate_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          t.dtype() == DType::Int32,
      std::string("Vulkan ") + name +
          " supports Float32, Float16 and Int32 tensors only");
  TP_CHECK(
      t.dim() <= 4, std::string("Vulkan ") + name + " supports up to 4d tensors");
}

// Texel metadata: the output extents plus each plane's own sizes, since a plane
// narrower than the output is addressed through them.
struct WhereBlock final {
  ivec4 out_sizes;
  ivec4 cond_sizes;
  ivec4 input_sizes;
  ivec4 other_sizes;
  int c_depth;
  int fill;
};

// Linear metadata: the output's stride vector plus sizes and strides per plane,
// and a mask naming the planes that are not already at the output shape.
struct WhereBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;
  int32_t broadcast;
  uint32_t fill1;
  ivec4 out_strides;
  ivec4 cond_sizes;
  ivec4 cond_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
};

// Row-major strides of an output shape: outer dimensions first, zero-filled
// past the rank, so the linear remap stops at the last real stride.
ivec4 where_out_strides(const std::vector<int64_t>& out_sizes) {
  ivec4 strides{0, 0, 0, 0};
  int64_t acc = 1;
  for (int d = static_cast<int>(out_sizes.size()) - 1; d >= 0; --d) {
    strides[d] = static_cast<int32_t>(acc);
    acc *= out_sizes[d];
  }
  return strides;
}

// Broadcast metadata of one plane: its shape right-aligned to the output rank,
// because broadcasting aligns trailing dimensions, with a zero stride on every
// singleton so the remap repeats that dimension.  A dimension the plane lacks
// keeps a size of one, which repeats the whole plane.
std::pair<ivec4, ivec4> where_broadcast_meta(
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& op_sizes) {
  const int ndim_out = static_cast<int>(out_sizes.size());
  const int ndim_op = static_cast<int>(op_sizes.size());
  TP_CHECK(
      ndim_op <= ndim_out,
      "Vulkan where: an operand rank exceeds the output rank");
  const int leading = ndim_out - ndim_op;
  ivec4 sizes{1, 1, 1, 1};
  ivec4 strides{0, 0, 0, 0};
  int64_t acc = 1;
  for (int d = ndim_op - 1; d >= 0; --d) {
    const int slot = leading + d;
    sizes[slot] = static_cast<int32_t>(op_sizes[d]);
    strides[slot] = static_cast<int32_t>(acc);
    acc *= op_sizes[d];
  }
  return {sizes, strides};
}

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
      condition.dim() <= 4, "Vulkan where supports up to 4d tensors");

  // The three planes are merged into one output shape; a plane that is not
  // already there is repeated inside the kernel rather than materialized.
  const std::vector<int64_t> cond_shape =
      static_cast<std::vector<int64_t>>(condition.shape());
  const std::vector<int64_t> self_shape =
      static_cast<std::vector<int64_t>>(self_branch.shape());
  const std::vector<int64_t> other_shape =
      static_cast<std::vector<int64_t>>(other_branch.shape());
  const std::vector<int64_t> out_sizes = tensorplay::broadcast_shapes(
      cond_shape, self_shape, other_shape);
  TP_CHECK(
      out_sizes.size() <= 4, "Vulkan where supports up to 4d tensors");

  const int32_t broadcast =
      (cond_shape != out_sizes ? 1 : 0) | (self_shape != out_sizes ? 2 : 0) |
      (other_shape != out_sizes ? 4 : 0);

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
  // Texel-stored planes are gathered into dense buffers first, because a
  // texel's lanes are not adjacent in the logical order a buffer reader walks.
  const bool is_buffer =
      v_cond.storage_type() == api::StorageType::BUFFER ||
      v_input.storage_type() == api::StorageType::BUFFER ||
      v_other.storage_type() == api::StorageType::BUFFER;
  if (is_buffer) {
    if (v_cond.storage_type() != api::StorageType::BUFFER) {
      v_cond = to_buffer_storage_dispatch(context, v_cond);
    }
    if (v_input.storage_type() != api::StorageType::BUFFER) {
      v_input = to_buffer_storage_dispatch(context, v_input);
    }
    if (v_other.storage_type() != api::StorageType::BUFFER) {
      v_other = to_buffer_storage_dispatch(context, v_other);
    }
    // A buffer-stored branch needs a buffer-stored output: the default
    // allocation is a texture, whose image the linear variant cannot write.
    api::vTensor v_output{
        context,
        out_sizes,
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
    const uint32_t n =
        static_cast<uint32_t>(multiply_integers(out_sizes));
    const auto cond_meta = where_broadcast_meta(out_sizes, cond_shape);
    const auto input_meta = where_broadcast_meta(out_sizes, self_shape);
    const auto other_meta = where_broadcast_meta(out_sizes, other_shape);
    const WhereBufferBlock block{
        n,
        0u,
        broadcast,
        0u,
        where_out_strides(out_sizes),
        cond_meta.first,
        cond_meta.second,
        input_meta.first,
        input_meta.second,
        other_meta.first,
        other_meta.second};
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

  api::vTensor v_output{context, out_sizes, vocabulary};

  const WhereBlock block{
      make_whcn_ivec4(out_sizes),
      make_whcn_ivec4(cond_shape),
      make_whcn_ivec4(self_shape),
      make_whcn_ivec4(other_shape),
      c_depth_of(out_sizes),
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

// A scalar branch is a plane of one element that the selection repeats, so
// folding it does not have to fill the output's shape; the folded plane takes
// the other branch's dtype, which is the type that holds the pair.
Tensor where_scalar_self_kernel(
    const Tensor& condition, const Scalar& self, const Tensor& other) {
  Tensor folded =
      full_kernel({1}, self, other.dtype(), other.device(), false);
  return where_kernel(condition, folded, other);
}

Tensor where_scalar_other_kernel(
    const Tensor& condition, const Tensor& self, const Scalar& other) {
  Tensor folded =
      full_kernel({1}, other, self.dtype(), self.device(), false);
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
