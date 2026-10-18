#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

void validate_float_1d_to_4d(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

struct MaskedFillBlock final {
  ivec4 out_sizes; // (W, H, C, N)
  int c_depth;
  float value;
  int fill;
};

// Runs the masked_fill shader with a scalar fill.  The value-tensor overload
// only accepts a 0-dimensional value on this backend, so both public forms
// funnel through one shader: the 0-dim tensor contributes its single element
// and the mask rides the device as a Bool payload sharing the input's
// texture geometry (a host-side mask is uploaded first).
Tensor masked_fill_impl(
    const Tensor& self,
    const Tensor& mask,
    Scalar value,
    const char* name) {
  validate_float_1d_to_4d(self, name);
  TP_CHECK(
      mask.dtype() == DType::Bool,
      std::string("Vulkan ") + name + " expects a Bool mask");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  const Tensor mask_on_device =
      mask.device().is_vulkan() ? mask : mask.to(self.device());
  api::vTensor v_mask = convert(mask_on_device);

  // Linear payloads (shapes beyond the 3-D image axis limit) fill through a
  // buffer-to-buffer variant: the Bool mask is one byte per element and rides
  // a packed-word buffer, the input is plain row-major memory.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    // The fill has to land in the storage the result is read from: an output
    // allocated in the default texture class would leave the linear write in
    // a buffer the image read-back never sees.
    api::vTensor v_lin{
        context,
        v_input.sizes(),
        DType::Float32,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    api::vTensor v_mask_lin = to_buffer_storage_dispatch(context, v_mask);
    const uint32_t n = static_cast<uint32_t>(v_input.numel());
    const struct MaskedFillBufferBlock {
      uint32_t buf_length;
      uint32_t fill0;
      float value;
      int32_t fill;
    } block{n, 0u, static_cast<float>(value.toDouble()), 0};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR("masked_fill_buffer"), pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_lin.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_mask_lin.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_lin);
  }

  api::vTensor v_output{context, v_input.sizes(), DType::Float32};

  const struct MaskedFillBlock block{
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_output.sizes()),
      static_cast<float>(value.toDouble()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR("masked_fill_scalar"),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_mask.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor masked_fill_scalar_kernel(
    const Tensor& self, const Tensor& mask, const Scalar& value) {
  return masked_fill_impl(self, mask, value, "masked_fill");
}

Tensor masked_fill_tensor_kernel(
    const Tensor& self, const Tensor& mask, const Tensor& value) {
  TP_CHECK(
      value.dim() == 0,
      "Vulkan masked_fill.Tensor: only a 0-dimensional value tensor is "
      "supported");
  return masked_fill_impl(self, mask, value.item(), "masked_fill.Tensor");
}

Tensor& masked_fill_scalar_inplace_kernel(
    Tensor& self, const Tensor& mask, const Scalar& value) {
  self.copy_(masked_fill_scalar_kernel(self, mask, value));
  return self;
}

Tensor& masked_fill_tensor_inplace_kernel(
    Tensor& self, const Tensor& mask, const Tensor& value) {
  self.copy_(masked_fill_tensor_kernel(self, mask, value));
  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MaskedFillKernels) {
  m.impl("masked_fill", &tensorplay::vulkan::ops::masked_fill_scalar_kernel);
  m.impl(
      "masked_fill.Tensor",
      &tensorplay::vulkan::ops::masked_fill_tensor_kernel);
  m.impl(
      "masked_fill_",
      &tensorplay::vulkan::ops::masked_fill_scalar_inplace_kernel);
  m.impl(
      "masked_fill_.Tensor",
      &tensorplay::vulkan::ops::masked_fill_tensor_inplace_kernel);
}

#endif /* USE_VULKAN */
