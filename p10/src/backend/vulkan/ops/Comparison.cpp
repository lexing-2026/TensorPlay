#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <Utils.h>

#include <utility>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Copies a texture-stored payload into a dense buffer vTensor; defined in
// BinaryOp.cpp and shared by every linear path.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

void validate_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          t.dtype() == DType::Int32 || t.dtype() == DType::Int8 ||
          t.dtype() == DType::UInt8 || t.dtype() == DType::Bool,
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

// Linear form: the Bool result packs one byte per element, four lanes to a
// word, so a dispatch covers four elements per invocation and one invocation
// owns one whole result word.  Single-precision operands move as 16-byte
// vectors; half-precision ones pack two lanes per word, so a result word
// draws on two operand words.
struct CompareBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;
  uint32_t fill1;
  uint32_t fill2;
};

struct CompareScalarBufferBlock final {
  uint32_t buf_length;
  float value_float;
  int32_t value_int;
  uint32_t fill0;
};

// A payload wider than the device's image axis lives in a storage buffer;
// the texel kernels have nothing to read or write there, so comparisons need
// a linear variant of their own.
void validate_linear_compare_dtype(DType dtype, const char* name) {
  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16 ||
          dtype == DType::Int32,
      std::string("Vulkan ") + name +
          " over buffer storage supports Float32, Float16 and Int32 only");
}

// Linear kernels specialize on the element vocabulary: single precision takes
// the plain name, integers the `_i32` twin, halves the `_f16` one.
const char* linear_dtype_suffix(DType dtype) {
  return dtype == DType::Int32
      ? "_i32"
      : (dtype == DType::Float16 ? "_f16" : "");
}

// A comparison against a scalar runs in the tensor's own element domain, so a
// half payload first takes the nearest representable half of the scalar.  Both
// sides then move through single precision, where half values map exactly and
// the decision is the same as in the narrow type.
float scalar_threshold(const Tensor& self, Scalar other) {
  const float value = other.to<float>();
  return self.dtype() == DType::Float16
      ? static_cast<float>(tensorplay::Half(value))
      : value;
}

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

  if (v_input.storage_type() == api::StorageType::BUFFER) {
    validate_linear_compare_dtype(self.dtype(), name);
    TP_CHECK(
        v_other.storage_type() == api::StorageType::BUFFER,
        std::string("Vulkan ") + name +
            " cannot mix a buffer operand with a texture operand");
    api::vTensor v_output{
        context,
        v_input.sizes(),
        DType::Bool,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

    const uint32_t n = safe_downcast_to_u32(static_cast<int64_t>(v_input.numel()));
    const CompareBufferBlock block{n, 0u, 0u, 0u};
    const std::string linear_shader =
        std::string(kernel_name) + "_buffer" +
        linear_dtype_suffix(self.dtype());

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()),
        pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    return convert(v_output);
  }

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
  // The single-precision texel shader samples an rgba16f view and the hardware
  // widens each half lane exactly, so a half payload reuses the float variant.
  const bool is_float =
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16;
  const char* suffix =
      is_float ? "" : (self.dtype() == DType::UInt8 ? "_u8" : "_i32");
  const std::string shader = std::string(name) + "_scalar" + suffix;
  const float threshold = is_float ? scalar_threshold(self, other) : 0.0f;
  const int32_t integer = is_float ? 0 : other.to<int32_t>();
  api::Context* context = api::context();
  api::vTensor input = convert(self);

  if (input.storage_type() == api::StorageType::BUFFER) {
    validate_linear_compare_dtype(self.dtype(), name);
    api::vTensor output{
        context,
        input.sizes(),
        DType::Bool,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const uint32_t n = safe_downcast_to_u32(static_cast<int64_t>(input.numel()));
    const CompareScalarBufferBlock block{n, threshold, integer, 0u};
    const std::string linear_shader =
        std::string(name) + "_scalar_buffer" + linear_dtype_suffix(self.dtype());
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()),
        barrier,
        {(n + 3u) / 4u, 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        output.buffer(
            barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        input.buffer(barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(output);
  }

  api::vTensor output{context, input.sizes(), DType::Bool};
  const struct Block final {
    float value_float;
    int32_t value_int;
    uint32_t value_uint;
    int32_t fill;
  } block{threshold,
          self.dtype() != DType::UInt8 ? integer : 0,
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

/*
 * Elementwise logic over Bool payloads.
 *
 * A Bool plane carries one byte per element and treats any nonzero byte as
 * true, so each operator decides lane by lane and writes 0 or 1 back.  The
 * connectives are the arithmetic forms of and, or, exclusive or and negation
 * over reduced lanes.
 *
 * Broadcasting is resolved here: a texel operand is addressed through its own
 * sizes, a linear one through row-major strides plus a bitmask naming the
 * operands that need the remap.  A linear payload keeps the whole op linear even
 * when the output also has an image form, because the texel kernels have no
 * buffer binding to read it from.
 */

// Row-major strides of an output shape: outer dimensions first, zero-filled
// past the rank, so the linear remap stops at the last real stride.
ivec4 logic_out_strides(const std::vector<int64_t>& out_sizes) {
  ivec4 strides{0, 0, 0, 0};
  int64_t acc = 1;
  for (int d = static_cast<int>(out_sizes.size()) - 1; d >= 0; --d) {
    strides[d] = static_cast<int32_t>(acc);
    acc *= out_sizes[d];
  }
  return strides;
}

// Broadcast metadata of one linear operand: its shape right-aligned to the
// output rank (broadcasting aligns trailing dimensions) with a stride of zero
// on every singleton, which the remap skips to repeat that dimension.
std::pair<ivec4, ivec4> logic_broadcast_meta(
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& op_sizes) {
  const int ndim_out = static_cast<int>(out_sizes.size());
  const int ndim_op = static_cast<int>(op_sizes.size());
  TP_CHECK(
      ndim_op <= ndim_out,
      "Vulkan logic: an operand rank exceeds the output rank");
  const int leading = ndim_out - ndim_op;
  ivec4 sizes{0, 0, 0, 0};
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

struct LogicBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;
  int32_t broadcast;
  uint32_t fill1;
  ivec4 out_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
};

struct LogicUnaryBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;
  int32_t broadcast;
  uint32_t fill1;
  ivec4 out_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
};

void validate_logic_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Bool,
      std::string("Vulkan ") + name +
          " connects Bool payloads; an integral or float plane has no "
          "bitwise kernel yet");
  TP_CHECK(
      t.dim() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");
}

Tensor logic_pair(
    const Tensor& self,
    const Tensor& other,
    const char* kernel_name,
    const char* name) {
  validate_logic_operand(self, name);
  validate_logic_operand(other, name);

  const std::vector<int64_t> out_sizes = tensorplay::broadcast_shapes(
      static_cast<std::vector<int64_t>>(self.shape()),
      static_cast<std::vector<int64_t>>(other.shape()));
  TP_CHECK(
      out_sizes.size() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_other = convert(other);

  const bool is_buffer =
      v_self.storage_type() == api::StorageType::BUFFER ||
      v_other.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      out_sizes,
      DType::Bool,
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  if (is_buffer) {
    api::vTensor v_self_buf = to_buffer_storage_dispatch(context, v_self);
    api::vTensor v_other_buf = to_buffer_storage_dispatch(context, v_other);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::vector<int64_t> self_sizes = v_self.sizes();
    const std::vector<int64_t> other_sizes = v_other.sizes();
    const int32_t broadcast =
        (self_sizes != out_sizes ? 1 : 0) |
        (other_sizes != out_sizes ? 2 : 0);
    const auto input_meta = logic_broadcast_meta(out_sizes, self_sizes);
    const auto other_meta = logic_broadcast_meta(out_sizes, other_sizes);
    const LogicBufferBlock block{
        n,
        0u,
        broadcast,
        0u,
        logic_out_strides(out_sizes),
        input_meta.first,
        input_meta.second,
        other_meta.first,
        other_meta.second};
    const std::string linear_shader = std::string(kernel_name) + "_buffer";
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    // Four packed lanes per word, so one invocation owns one whole word.
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()),
        pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_self_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes())};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

// A single operand keeps its own shape, so nothing broadcasts and the linear
// block needs no remap metadata beyond the element count.
Tensor logic_unary(
    const Tensor& self,
    const char* kernel_name,
    const char* name) {
  validate_logic_operand(self, name);

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  const bool is_buffer =
      v_self.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      v_self.sizes(),
      DType::Bool,
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  if (is_buffer) {
    api::vTensor v_self_buf = to_buffer_storage_dispatch(context, v_self);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::vector<int64_t> sizes = v_self.sizes();
    const auto input_meta = logic_broadcast_meta(sizes, sizes);
    const LogicUnaryBufferBlock block{
        n,
        0u,
        0,
        0u,
        logic_out_strides(sizes),
        input_meta.first,
        input_meta.second};
    const std::string linear_shader = std::string(kernel_name) + "_buffer";
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()),
        pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_self_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
  } block{make_whcn_ivec4(v_output.sizes()), make_whcn_ivec4(v_self.sizes())};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

// The logical connectives decide by "nonzero means true" in whatever
// vocabulary the operands carry.  A Bool plane already spells that out as 0 or
// 1, so it feeds the logic kernel untouched; anything else is first reduced to
// Bool by testing each element against zero, which is the same decision.
Tensor truth_of(const Tensor& t, const char* name) {
  if (t.dtype() == DType::Bool) {
    return t;
  }
  validate_operand(t, name);
  return compare_scalar(t, 0.0, "ne");
}

Tensor logical_pair(
    const Tensor& self,
    const Tensor& other,
    const char* kernel_name,
    const char* name) {
  return logic_pair(truth_of(self, name), truth_of(other, name), kernel_name,
                    name);
}

} // namespace

Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "eq", "eq");
}
Tensor eq_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "eq");
}

Tensor ne_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "ne", "ne");
}
Tensor ne_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "ne");
}

Tensor lt_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "lt", "lt");
}
Tensor lt_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "lt");
}

Tensor le_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "le", "le");
}
Tensor le_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "le");
}

Tensor gt_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "gt", "gt");
}
Tensor gt_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "gt");
}

Tensor ge_tensor_kernel(const Tensor& self, const Tensor& other) {
  return compare_impl(self, other, "ge", "ge");
}
Tensor ge_scalar_kernel(const Tensor& self, const Scalar& other) {
  return compare_scalar(self, other, "ge");
}

Tensor bitwise_not_kernel(const Tensor& self) {
  return logic_unary(self, "logic_not", "bitwise_not");
}
Tensor bitwise_and_tensor_kernel(const Tensor& self, const Tensor& other) {
  return logic_pair(self, other, "logic_and", "bitwise_and");
}
Tensor bitwise_or_tensor_kernel(const Tensor& self, const Tensor& other) {
  return logic_pair(self, other, "logic_or", "bitwise_or");
}
Tensor bitwise_xor_tensor_kernel(const Tensor& self, const Tensor& other) {
  return logic_pair(self, other, "logic_xor", "bitwise_xor");
}

Tensor logical_and_kernel(const Tensor& self, const Tensor& other) {
  return logical_pair(self, other, "logic_and", "logical_and");
}
Tensor logical_or_kernel(const Tensor& self, const Tensor& other) {
  return logical_pair(self, other, "logic_or", "logical_or");
}
Tensor logical_xor_kernel(const Tensor& self, const Tensor& other) {
  return logical_pair(self, other, "logic_xor", "logical_xor");
}

// Negation of a numeric plane is the test that fails every nonzero element,
// which is the equality against zero rather than the one against it.
Tensor logical_not_kernel(const Tensor& self) {
  if (self.dtype() == DType::Bool) {
    return logic_unary(self, "logic_not", "logical_not");
  }
  validate_operand(self, "logical_not");
  return compare_scalar(self, 0.0, "eq");
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
  m.impl("bitwise_not", &tensorplay::vulkan::ops::bitwise_not_kernel);
  m.impl(
      "bitwise_and.Tensor", &tensorplay::vulkan::ops::bitwise_and_tensor_kernel);
  m.impl("bitwise_or.Tensor", &tensorplay::vulkan::ops::bitwise_or_tensor_kernel);
  m.impl(
      "bitwise_xor.Tensor", &tensorplay::vulkan::ops::bitwise_xor_tensor_kernel);
  m.impl("logical_not", &tensorplay::vulkan::ops::logical_not_kernel);
  m.impl("logical_and", &tensorplay::vulkan::ops::logical_and_kernel);
  m.impl("logical_or", &tensorplay::vulkan::ops::logical_or_kernel);
  m.impl("logical_xor", &tensorplay::vulkan::ops::logical_xor_kernel);
}

#endif /* USE_VULKAN */
