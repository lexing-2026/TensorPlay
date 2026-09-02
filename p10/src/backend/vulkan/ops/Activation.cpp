#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <string>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

//
// Pointwise activations.  Parameter-free formulas dispatch through the
// activation template; the parameterized family (leaky_relu, threshold,
// hardshrink) carries its scalars in the block.  Every activation exists in
// out-of-place and in-place form and in an image and a linear-buffer
// variant; the storage kind of the input selects between them.
//

void validate_activation(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

// Dispatches one activation formula by shader name, covering both storage
// kinds and both output modes.  The in-place shaders carry the "inplace"
// suffix appended by the code generator and bind a single payload.
Tensor& activation_dispatch(
    const std::string& shader,
    const std::string& buffer_shader,
    Tensor& output,
    const Tensor& input,
    float p0,
    float p1) {
  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);
  api::vTensor v_output = convert(output);
  const bool in_place = (&output == &input) || !input.defined();

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));

    if (in_place) {
      const struct Block final {
        uint32_t buf_length;
        float p0;
        float p1;
      } block{n, p0, p1};
      api::UniformParamsBuffer params(context, block);
      api::PipelineBarrier pipeline_barrier{};
      context->submit_compute_job(
          VK_KERNEL_FROM_STR((buffer_shader + "inplace").c_str()),
          pipeline_barrier, {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
          params.buffer());
      return output;
    }

    const struct Block final {
      uint32_t buf_length;
      float p0;
      float p1;
    } block{n, p0, p1};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader.c_str()), pipeline_barrier,
        {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return output;
  }

  const struct Block final {
    ivec4 extents;
    float p0;
    float p1;
  } block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      p0,
      p1};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  if (in_place) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR((shader + "inplace").c_str()), pipeline_barrier,
        v_output.extents(), adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        params.buffer());
    return output;
  }

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader.c_str()), pipeline_barrier,
      v_output.extents(), adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return output;
}

Tensor activation_out(
    const std::string& shader,
    const std::string& buffer_shader,
    const Tensor& self,
    float p0,
    float p1) {
  validate_activation(self, shader.c_str());
  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);

  api::vTensor v_output{context, v_self.sizes(), v_self.dtype()};
  Tensor out = convert(v_output);
  activation_dispatch(shader, buffer_shader, out, self, p0, p1);
  return out;
}

Tensor& activation_inplace(
    const std::string& shader,
    const std::string& buffer_shader,
    Tensor& self,
    float p0,
    float p1) {
  validate_activation(self, shader.c_str());
  return activation_dispatch(shader, buffer_shader, self, self, p0, p1);
}

} // namespace

#define TP_VK_ACTIVATION_KERNELS(FUNC, SHADER, BUFFER_SHADER, P0, P1)        \
  Tensor FUNC(const Tensor& self) {                                          \
    return activation_out(SHADER, BUFFER_SHADER, self, P0, P1);              \
  }                                                                          \
  Tensor& FUNC##_inplace(Tensor& self) {                                     \
    return activation_inplace(SHADER, BUFFER_SHADER, self, P0, P1);          \
  }

TP_VK_ACTIVATION_KERNELS(silu_kernel, "silu", "buffer_silu", 0.0f, 0.0f)
TP_VK_ACTIVATION_KERNELS(mish_kernel, "mish", "buffer_mish", 0.0f, 0.0f)
TP_VK_ACTIVATION_KERNELS(relu6_kernel, "relu6", "buffer_relu6", 0.0f, 0.0f)
TP_VK_ACTIVATION_KERNELS(
    hardsigmoid_kernel, "hardsigmoid", "buffer_hardsigmoid", 0.0f, 0.0f)
TP_VK_ACTIVATION_KERNELS(
    hardswish_kernel, "hardswish", "buffer_hardswish", 0.0f, 0.0f)

#undef TP_VK_ACTIVATION_KERNELS

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
  if (approximate == "tanh") {
    return activation_out("gelu_tanh", "buffer_gelu_tanh", self, 0.0f, 0.0f);
  }
  TP_CHECK(
      approximate == "none",
      "Vulkan gelu: approximate must be none or tanh");
  return activation_out("gelu", "buffer_gelu", self, 0.0f, 0.0f);
}

Tensor& gelu_inplace_kernel(Tensor& self, const std::string& approximate) {
  if (approximate == "tanh") {
    return activation_inplace("gelu_tanh", "buffer_gelu_tanh", self, 0.0f, 0.0f);
  }
  TP_CHECK(
      approximate == "none",
      "Vulkan gelu: approximate must be none or tanh");
  return activation_inplace("gelu", "buffer_gelu", self, 0.0f, 0.0f);
}

Tensor leaky_relu_kernel(const Tensor& self, Scalar negative_slope) {
  return activation_out(
      "leaky_relu", "buffer_leaky_relu", self,
      static_cast<float>(negative_slope.toDouble()), 0.0f);
}

Tensor& leaky_relu_inplace_kernel(Tensor& self, Scalar negative_slope) {
  return activation_inplace(
      "leaky_relu", "buffer_leaky_relu", self,
      static_cast<float>(negative_slope.toDouble()), 0.0f);
}

Tensor threshold_kernel(const Tensor& self, Scalar threshold, Scalar value) {
  return activation_out(
      "threshold", "buffer_threshold", self,
      static_cast<float>(threshold.toDouble()),
      static_cast<float>(value.toDouble()));
}

Tensor& threshold_inplace_kernel(
    Tensor& self,
    Scalar threshold,
    Scalar value) {
  return activation_inplace(
      "threshold", "buffer_threshold", self,
      static_cast<float>(threshold.toDouble()),
      static_cast<float>(value.toDouble()));
}

Tensor hardshrink_kernel(const Tensor& self, Scalar lambd) {
  return activation_out(
      "hardshrink", "buffer_hardshrink", self,
      static_cast<float>(lambd.toDouble()), 0.0f);
}

Tensor hardshrink_backward_kernel(
    const Tensor& grad_out,
    const Tensor& self,
    Scalar lambd) {
  // Gradient of hardshrink: identity where the input magnitude exceeded the
  // threshold, zero inside the dead band.  Composed from existing
  // elementwise kernels: mask = abs(x) <= lambd; grad * (1 - mask).
  TP_CHECK(
      grad_out.dtype() == DType::Float32 && self.dtype() == DType::Float32,
      "Vulkan hardshrink_backward supports Float32 tensors only");
  const float l = static_cast<float>(lambd.toDouble());

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_grad = convert(grad_out);

  api::vTensor v_mask{context, v_self.sizes(), self.dtype()};
  {
    const struct Block final {
      ivec4 extents;
      float p0;
      float p1;
    } block{
        ivec4(
            v_self.extents()[0u],
            v_self.extents()[1u],
            v_self.extents()[2u],
            0),
        l,
        0.0f};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        // dead-band mask: (|x| <= lambd) ? 1 : 0 through the threshold
        // formula with the roles of value/input inverted
        VK_KERNEL_FROM_STR("hardshrink_mask"), pipeline_barrier,
        v_self.extents(), adaptive_work_group_size(v_self.extents()),
        VK_NULL_HANDLE,
        v_mask.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  api::vTensor v_one_minus{context, v_self.sizes(), self.dtype()};
  {
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR("one_minus"), pipeline_barrier, v_self.extents(),
        adaptive_work_group_size(v_self.extents()), VK_NULL_HANDLE,
        v_one_minus.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_mask.image(pipeline_barrier, api::PipelineStage::COMPUTE));
  }

  api::vTensor v_out{context, v_self.sizes(), self.dtype()};
  {
    const struct Block final {
      ivec4 output_sizes;
      ivec4 input_sizes;
      ivec4 other_sizes;
      float alpha;
    } block{
        make_whcn_ivec4(v_out.sizes()),
        make_whcn_ivec4(v_grad.sizes()),
        make_whcn_ivec4(v_one_minus.sizes()),
        1.0f};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL(mul), pipeline_barrier, v_out.extents(),
        adaptive_work_group_size(v_out.extents()), VK_NULL_HANDLE,
        v_out.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_grad.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_one_minus.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_out);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ActivationKernels) {
  m.impl("silu", &tensorplay::vulkan::ops::silu_kernel);
  m.impl("silu_", &tensorplay::vulkan::ops::silu_kernel_inplace);
  m.impl("mish", &tensorplay::vulkan::ops::mish_kernel);
  m.impl("mish_", &tensorplay::vulkan::ops::mish_kernel_inplace);
  m.impl("relu6", &tensorplay::vulkan::ops::relu6_kernel);
  m.impl("relu6_", &tensorplay::vulkan::ops::relu6_kernel_inplace);
  m.impl("hardsigmoid", &tensorplay::vulkan::ops::hardsigmoid_kernel);
  m.impl("hardsigmoid_", &tensorplay::vulkan::ops::hardsigmoid_kernel_inplace);
  m.impl("hardswish", &tensorplay::vulkan::ops::hardswish_kernel);
  m.impl("hardswish_", &tensorplay::vulkan::ops::hardswish_kernel_inplace);
  m.impl("gelu", &tensorplay::vulkan::ops::gelu_kernel);
  m.impl("gelu_", &tensorplay::vulkan::ops::gelu_inplace_kernel);
  m.impl("leaky_relu", &tensorplay::vulkan::ops::leaky_relu_kernel);
  m.impl("leaky_relu_", &tensorplay::vulkan::ops::leaky_relu_inplace_kernel);
  m.impl("threshold", &tensorplay::vulkan::ops::threshold_kernel);
  m.impl("threshold_", &tensorplay::vulkan::ops::threshold_inplace_kernel);
  m.impl("hardshrink", &tensorplay::vulkan::ops::hardshrink_kernel);
  m.impl("hardshrink_backward",
         &tensorplay::vulkan::ops::hardshrink_backward_kernel);
}

#endif /* USE_VULKAN */
