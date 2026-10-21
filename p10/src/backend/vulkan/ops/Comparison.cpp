#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <Utils.h>

#include <cmath>
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

// The widths a comparison reads.  A count wider than a word has no texel to sit
// in, so it is walked as a pair of words over linear storage by both the
// two-payload form and the form that compares against a number; each of those
// refuses such a payload at its image path.
void validate_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          t.dtype() == DType::Int32 || t.dtype() == DType::Int64 ||
          t.dtype() == DType::Int8 || t.dtype() == DType::UInt8 ||
          t.dtype() == DType::Bool,
      std::string("Vulkan ") + name + " has an unsupported dtype");
  TP_CHECK(
      t.dim() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");
}

struct CompareBlock final {
  ivec4 out_sizes;
  // Each operand is fetched through its own texel geometry, so a dimension of
  // one repeats across the matching output range.
  ivec4 input_sizes;
  ivec4 other_sizes;
  int c_depth;
  int fill;
};

// Linear form: the Bool result packs one byte per element, four lanes to a
// word, so a dispatch covers four elements per invocation and one invocation
// owns one whole result word.  Single-precision operands move as 16-byte
// vectors; half-precision ones pack two lanes per word, so a result word
// draws on two operand words.  A broadcast operand is gathered lane by lane
// from its own row-major strides.
struct CompareBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;      // std140 padding
  int32_t broadcast;   // bit 0: the first operand, bit 1: the second
  uint32_t fill1;
  ivec4 out_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
};

struct CompareScalarBufferBlock final {
  uint32_t buf_length;
  float value_float;
  int32_t value_int;
  uint32_t fill0;
};

// The same walk over a code wider than a word, where the threshold travels as the
// pair of words such a count is carried in rather than as a number a lane holds.
struct CompareScalarWideBufferBlock final {
  uint32_t buf_length;
  uint32_t fill0;
  uint32_t value_low;
  uint32_t value_high;
};

// A payload wider than the device's image axis lives in a storage buffer;
// the texel kernels have nothing to read or write there, so comparisons need
// a linear variant of their own.
void validate_linear_compare_dtype(DType dtype, const char* name) {
  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16 ||
          dtype == DType::Int32 || dtype == DType::Int64,
      std::string("Vulkan ") + name +
          " over buffer storage supports Float32, Float16, Int32 and Int64 only");
}

// Linear kernels specialize on the element vocabulary: single precision takes
// the plain name, integers the `_i32` twin, halves the `_f16` one, and a count
// that needs a pair of words the `_wide` one.
const char* linear_dtype_suffix(DType dtype) {
  if (dtype == DType::Int64) {
    return "_wide";
  }
  return dtype == DType::Int32 ? "_i32"
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

// Row-major strides of an output shape: outer dimensions first, zero-filled
// past the rank, so the linear remap stops at the last real stride.
ivec4 linear_out_strides(const std::vector<int64_t>& out_sizes) {
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
std::pair<ivec4, ivec4> linear_broadcast_meta(
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& op_sizes) {
  const int ndim_out = static_cast<int>(out_sizes.size());
  const int ndim_op = static_cast<int>(op_sizes.size());
  TP_CHECK(
      ndim_op <= ndim_out,
      "Vulkan linear broadcast: an operand rank exceeds the output rank");
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

/*
 * Elementwise comparison driver.
 *
 * Broadcasting is resolved here: a texel operand is addressed through its own
 * sizes, a linear one through row-major strides plus a bitmask naming the
 * operands that need the remap.  A linear payload keeps the whole op linear even
 * when the output also has an image form, because the texel kernels have no
 * buffer binding to read it from.
 */
Tensor compare_impl(
    const Tensor& self,
    const Tensor& other,
    const char* kernel_name,
    const char* name) {
  validate_operand(self, name);
  validate_operand(other, name);
  TP_CHECK(self.dtype() == other.dtype(),
           "Vulkan comparison requires matching operand dtypes");

  const std::vector<int64_t> out_sizes = tensorplay::broadcast_shapes(
      static_cast<std::vector<int64_t>>(self.shape()),
      static_cast<std::vector<int64_t>>(other.shape()));
  TP_CHECK(
      out_sizes.size() <= 4,
      std::string("Vulkan ") + name + " supports up to 4d tensors");

  const char* suffix = self.dtype() == DType::Int32 ? "_i32"
      : self.dtype() == DType::UInt8 ? "_u8"
      : self.dtype() == DType::Bool || self.dtype() == DType::Int8 ? "_i8" : "";
  const std::string shader_name = std::string(kernel_name) + suffix;
  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_other = convert(other);

  const bool is_buffer =
      v_input.storage_type() == api::StorageType::BUFFER ||
      v_other.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      out_sizes,
      DType::Bool,
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  if (is_buffer) {
    validate_linear_compare_dtype(self.dtype(), name);
    // A texel operand has no buffer binding the linear kernel could read, so it
    // is copied into a dense payload of its own first.
    api::vTensor v_input_buf = to_buffer_storage_dispatch(context, v_input);
    api::vTensor v_other_buf = to_buffer_storage_dispatch(context, v_other);

    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::vector<int64_t> input_sizes = v_input.sizes();
    const std::vector<int64_t> other_sizes = v_other.sizes();
    const int32_t broadcast =
        (input_sizes != out_sizes ? 1 : 0) |
        (other_sizes != out_sizes ? 2 : 0);
    const auto input_meta = linear_broadcast_meta(out_sizes, input_sizes);
    const auto other_meta = linear_broadcast_meta(out_sizes, other_sizes);
    const CompareBufferBlock block{
        n,
        0u,
        broadcast,
        0u,
        linear_out_strides(out_sizes),
        input_meta.first,
        input_meta.second,
        other_meta.first,
        other_meta.second};
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
        v_input_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    return convert(v_output);
  }

  const struct CompareBlock block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_other.sizes()),
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
  const bool is_wide = self.dtype() == DType::Int64;
  // A count wider than a word is carried as a pair of words, so a threshold with
  // a fraction in it has no pair of its own to travel in: the walk would have to
  // answer for a number the width of the payload does not name.
  TP_CHECK(
      !is_wide || other.isIntegral() ||
          std::floor(other.to<double>()) == other.to<double>(),
      std::string("Vulkan ") + name +
          " compares a count wider than a word against a whole number");
  const char* suffix =
      is_float ? "" : (self.dtype() == DType::UInt8 ? "_u8" : "_i32");
  const std::string shader = std::string(name) + "_scalar" + suffix;
  const float threshold = is_float ? scalar_threshold(self, other) : 0.0f;
  const int32_t integer = (is_float || is_wide) ? 0 : other.to<int32_t>();
  const int64_t wide_threshold = is_wide ? other.to<int64_t>() : 0;
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
    // The threshold of a wide walk arrives as the two words such a count is
    // carried in, low one first.
    const uint64_t wide_bits = static_cast<uint64_t>(wide_threshold);
    api::UniformParamsBuffer params = is_wide
        ? api::UniformParamsBuffer(
              context,
              CompareScalarWideBufferBlock{n, 0u, static_cast<uint32_t>(wide_bits),
                                          static_cast<uint32_t>(wide_bits >> 32u)})
        : api::UniformParamsBuffer(
              context, CompareScalarBufferBlock{n, threshold, integer, 0u});
    const std::string linear_shader =
        std::string(name) + "_scalar_buffer" + linear_dtype_suffix(self.dtype());
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

  // An image names one payload per texel, so a code wider than a word has no
  // image form to be read through.
  TP_CHECK(
      !is_wide,
      std::string("Vulkan ") + name +
          " has no image form for a count wider than a word");

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
    const auto input_meta = linear_broadcast_meta(out_sizes, self_sizes);
    const auto other_meta = linear_broadcast_meta(out_sizes, other_sizes);
    const LogicBufferBlock block{
        n,
        0u,
        broadcast,
        0u,
        linear_out_strides(out_sizes),
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
    const auto input_meta = linear_broadcast_meta(sizes, sizes);
    const LogicUnaryBufferBlock block{
        n,
        0u,
        0,
        0u,
        linear_out_strides(sizes),
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
