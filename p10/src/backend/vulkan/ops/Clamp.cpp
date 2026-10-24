#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../impl/Common.h"
#include "../api/ShaderRegistry.h"

#include <limits>
#include <utility>

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

// One side of the closed range a count wider than a word is clipped to, carried
// as the pair of words such a count is held in, low one first.  A side the caller
// left unset is the extreme of the width, which clips nothing.
std::pair<uint32_t, uint32_t> wide_bound(
    const std::optional<Scalar>& value,
    const int64_t extreme) {
  const uint64_t bits = static_cast<uint64_t>(
      value ? value->to<int64_t>() : extreme);
  return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32u)};
}

Tensor clamp(
    const Tensor& self_arg,
    const std::optional<Scalar>& min_arg,
    const std::optional<Scalar>& max_arg) {
  const bool is_int = self_arg.dtype() == DType::Int32;
  TP_CHECK(
      is_int || self_arg.dtype() == DType::Float32 ||
          self_arg.dtype() == DType::Float16 ||
          self_arg.dtype() == DType::Int64,
      "Vulkan clamp supports Float32, Float16, Int32 and Int64 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  // Linear path.  A payload too wide for any image axis the device allows lives
  // in a storage buffer, where the texel-oriented clip below has nothing to
  // scan; this branch walks the same closed range over 16-byte vectors.  The
  // absent side of the range is the extreme of the element type, which the clip
  // leaves untouched.
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    validate_buffer_elementwise_dtype(v_self, "clamp");
    api::vTensor v_linear_output{
        context,
        v_self.sizes(),
        v_self.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

    const uint32_t elements =
        safe_downcast_to_u32(static_cast<int64_t>(v_linear_output.numel()));

    // A count wider than a word clips between two bounds of its own width, each
    // of which travels as a pair of words.
    if (self_arg.dtype() == DType::Int64) {
      const auto lo = wide_bound(min_arg, std::numeric_limits<int64_t>::min());
      const auto hi = wide_bound(max_arg, std::numeric_limits<int64_t>::max());
      const struct BlockW final {
        uint32_t buf_length;
        uint32_t lo_low;
        uint32_t lo_high;
        uint32_t hi_low;
        uint32_t hi_high;
      } blockw{
          elements,
          lo.first,
          lo.second,
          hi.first,
          hi.second,
      };
      api::UniformParamsBuffer wide_params(context, blockw);
      api::PipelineBarrier wide_barrier{};
      context->submit_compute_job(
          VK_KERNEL(buffer_clamp_wide),
          wide_barrier,
          {buffer_dispatch_count(elements, self_arg.dtype(), 4u), 1u, 1u},
          api::utils::uvec3{256u, 1u, 1u},
          VK_NULL_HANDLE,
          v_linear_output.buffer(
              wide_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_self.buffer(wide_barrier, api::PipelineStage::COMPUTE),
          wide_params.buffer());
      return convert(v_linear_output);
    }

    const struct BlockF final {
      uint32_t buf_length;
      uint32_t fill0;
      float lo;
      float hi;
    } blockf{
        elements,
        0u,
        min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
        max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity(),
    };

    const struct BlockI final {
      uint32_t buf_length;
      uint32_t fill0;
      int32_t lo;
      int32_t hi;
    } blocki{
        elements,
        0u,
        min_arg ? static_cast<int32_t>(min_arg->to<int64_t>())
                : std::numeric_limits<int32_t>::min(),
        max_arg ? static_cast<int32_t>(max_arg->to<int64_t>())
                : std::numeric_limits<int32_t>::max(),
    };

    // The half build takes its bounds in the two parameter slots of the
    // shared linear template, so it needs its own block layout.
    const struct BlockH final {
      uint32_t buf_length;
      float lo;
      float hi;
    } blockh{
        elements,
        min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
        max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity(),
    };

    const bool is_half = self_arg.dtype() == DType::Float16;

    api::UniformParamsBuffer linear_params =
        is_int ? api::UniformParamsBuffer(context, blocki)
        : is_half ? api::UniformParamsBuffer(context, blockh)
                  : api::UniformParamsBuffer(context, blockf);
    api::PipelineBarrier linear_barrier{};

    const std::string half_shader = buffer_shader_for_dtype(
        "buffer_clamp", self_arg.dtype());

    context->submit_compute_job(
        is_int ? VK_KERNEL(buffer_clamp_i32)
               : VK_KERNEL_FROM_STR(half_shader.c_str()),
        linear_barrier,
        {buffer_dispatch_count(elements, self_arg.dtype(), 4u), 1u, 1u},
        api::utils::uvec3{256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_linear_output.buffer(
            linear_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_self.buffer(linear_barrier, api::PipelineStage::COMPUTE),
        linear_params.buffer());

    return convert(v_linear_output);
  }

  // An image names one payload per texel, so a code wider than a word has no
  // image form to be clipped through.
  TP_CHECK(
      self_arg.dtype() != DType::Int64,
      "Vulkan clamp has no image form for a count wider than a word");

  // The clip below reads and writes texels, so the result is reserved as an
  // image even for a shape an unasked allocation would keep in words.
  api::vTensor v_output = allocate_output(
      context, api::StorageType::TEXTURE_3D, v_self.sizes(),
      v_self.dtype());

  const struct Block final {
    ivec4 extents;
    // clamp range
    vec2 clamp;
  } block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      {min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
       max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity()},
  };

  const struct BlockI final {
    ivec4 extents;
    // clamp range in int words
    ivec2 clamp;
  } blocki{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      {min_arg ? static_cast<int32_t>(min_arg->to<int64_t>())
               : std::numeric_limits<int32_t>::min(),
       max_arg ? static_cast<int32_t>(max_arg->to<int64_t>())
               : std::numeric_limits<int32_t>::max()},
  };

  api::UniformParamsBuffer params =
      is_int ? api::UniformParamsBuffer(context, blocki)
             : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      is_int ? kernel_for("clamp_i32", self_arg) : kernel_for("clamp", self_arg),
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
  const bool is_int = self_arg.dtype() == DType::Int32;
  TP_CHECK(
      is_int || self_arg.dtype() == DType::Float32 ||
          self_arg.dtype() == DType::Float16 ||
          self_arg.dtype() == DType::Int64,
      "Vulkan clamp_ supports Float32, Float16, Int32 and Int64 tensors only");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  // Linear path: the clip reads and writes the same storage buffer, so no second
  // copy of a payload that is already too wide for a texture has to be reserved.
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    validate_buffer_elementwise_dtype(v_self, "clamp");
    const uint32_t elements =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));

    // The destination of a written-back clip keeps the width it was allocated
    // in, so the bounds of a counting payload are pairs of words too.
    if (self_arg.dtype() == DType::Int64) {
      const auto lo = wide_bound(min_arg, std::numeric_limits<int64_t>::min());
      const auto hi = wide_bound(max_arg, std::numeric_limits<int64_t>::max());
      const struct BlockW final {
        uint32_t buf_length;
        uint32_t lo_low;
        uint32_t lo_high;
        uint32_t hi_low;
        uint32_t hi_high;
      } blockw{
          elements,
          lo.first,
          lo.second,
          hi.first,
          hi.second,
      };
      api::UniformParamsBuffer wide_params(context, blockw);
      api::PipelineBarrier wide_barrier{};
      context->submit_compute_job(
          VK_KERNEL(buffer_clamp_wideinplace),
          wide_barrier,
          {buffer_dispatch_count(elements, self_arg.dtype(), 4u), 1u, 1u},
          api::utils::uvec3{256u, 1u, 1u},
          VK_NULL_HANDLE,
          v_self.buffer(
              wide_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
          wide_params.buffer());
      return self_arg;
    }

    const struct BlockF final {
      uint32_t buf_length;
      uint32_t fill0;
      float lo;
      float hi;
    } blockf{
        elements,
        0u,
        min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
        max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity(),
    };

    const struct BlockI final {
      uint32_t buf_length;
      uint32_t fill0;
      int32_t lo;
      int32_t hi;
    } blocki{
        elements,
        0u,
        min_arg ? static_cast<int32_t>(min_arg->to<int64_t>())
                : std::numeric_limits<int32_t>::min(),
        max_arg ? static_cast<int32_t>(max_arg->to<int64_t>())
                : std::numeric_limits<int32_t>::max(),
    };

    // The half build takes its bounds in the two parameter slots of the
    // shared linear template, so it needs its own block layout.
    const struct BlockH final {
      uint32_t buf_length;
      float lo;
      float hi;
    } blockh{
        elements,
        min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
        max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity(),
    };

    const bool is_half = self_arg.dtype() == DType::Float16;

    api::UniformParamsBuffer linear_params =
        is_int ? api::UniformParamsBuffer(context, blocki)
        : is_half ? api::UniformParamsBuffer(context, blockh)
                  : api::UniformParamsBuffer(context, blockf);
    api::PipelineBarrier linear_barrier{};

    const std::string half_shader = buffer_shader_for_dtype(
        "buffer_clampinplace", self_arg.dtype());

    context->submit_compute_job(
        is_int ? VK_KERNEL(buffer_clamp_i32inplace)
               : VK_KERNEL_FROM_STR(half_shader.c_str()),
        linear_barrier,
        {buffer_dispatch_count(elements, self_arg.dtype(), 4u), 1u, 1u},
        api::utils::uvec3{256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_self.buffer(
            linear_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        linear_params.buffer());

    return self_arg;
  }

  TP_CHECK(
      self_arg.dtype() != DType::Int64,
      "Vulkan clamp_ has no image form for a count wider than a word");

  const struct Block final {
    ivec4 extents;
    // clamp range
    vec2 clamp;
  } block{
      ivec4(
          v_self.extents()[0u],
          v_self.extents()[1u],
          v_self.extents()[2u],
          0),
      {min_arg ? min_arg->to<float>() : -std::numeric_limits<float>::infinity(),
       max_arg ? max_arg->to<float>() : std::numeric_limits<float>::infinity()},
  };

  const struct BlockI final {
    ivec4 extents;
    // clamp range in int words
    ivec2 clamp;
  } blocki{
      ivec4(
          v_self.extents()[0u],
          v_self.extents()[1u],
          v_self.extents()[2u],
          0),
      {min_arg ? static_cast<int32_t>(min_arg->to<int64_t>())
               : std::numeric_limits<int32_t>::min(),
       max_arg ? static_cast<int32_t>(max_arg->to<int64_t>())
               : std::numeric_limits<int32_t>::max()},
  };

  api::UniformParamsBuffer params =
      is_int ? api::UniformParamsBuffer(context, blocki)
             : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      is_int ? kernel_for("clamp_i32inplace", self_arg)
             : kernel_for("clampinplace", self_arg),
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

Tensor clamp_kernel(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return clamp(self, min, max);
}

Tensor& clamp_inplace_kernel(
    Tensor& self,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  return clamp_(self, min, max);
}

// One-sided clamps delegate to the two-sided entry with the opposite bound
// left unset, and the in-place forms round-trip through the copy kernel.
Tensor clamp_min_kernel(const Tensor& self, const Scalar& min) {
  return clamp(self, min, std::nullopt);
}

Tensor clamp_max_kernel(const Tensor& self, const Scalar& max) {
  return clamp(self, std::nullopt, max);
}

Tensor& clamp_min_inplace_kernel(Tensor& self, const Scalar& min) {
  self.copy_(clamp(self, min, std::nullopt));
  return self;
}

Tensor& clamp_max_inplace_kernel(Tensor& self, const Scalar& max) {
  self.copy_(clamp(self, std::nullopt, max));
  return self;
}

/*
 * Hard tanh is the clamping range applied under the activation's name; the
 * entry points reuse the clamp dispatches directly.
 */
Tensor hardtanh_kernel(const Tensor& self, const Scalar& min, const Scalar& max) {
  return clamp(self, min, max);
}

Tensor& hardtanh_inplace_kernel(Tensor& self, const Scalar& min, const Scalar& max) {
  return clamp_(self, min, max);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ClampKernels) {
  m.impl("clamp", &tensorplay::vulkan::ops::clamp_kernel);
  m.impl("clamp_", &tensorplay::vulkan::ops::clamp_inplace_kernel);
  m.impl("clamp_min", &tensorplay::vulkan::ops::clamp_min_kernel);
  m.impl("clamp_max", &tensorplay::vulkan::ops::clamp_max_kernel);
  m.impl("clamp_min_", &tensorplay::vulkan::ops::clamp_min_inplace_kernel);
  m.impl("clamp_max_", &tensorplay::vulkan::ops::clamp_max_inplace_kernel);
  m.impl("hardtanh", &tensorplay::vulkan::ops::hardtanh_kernel);
  m.impl("hardtanh_", &tensorplay::vulkan::ops::hardtanh_inplace_kernel);
}

#endif /* USE_VULKAN */
