#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_pool_input(const Tensor& input) {
  TP_CHECK(
      input.dtype() == DType::Float32,
      "Vulkan pooling supports Float32 tensors only");
  TP_CHECK(
      input.dim() == 4,
      "Vulkan pooling requires a 4d tensor (use a batch of 1 for 3d input)");
}

Tensor pool2d_impl(
    const Tensor& input_arg,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    bool ceil_mode) {
  validate_pool_input(input_arg);

  TP_CHECK(
      kernel_size.size() == 2 && stride.size() == 2 && padding.size() == 2,
      "Vulkan pooling expects 2d kernel, stride, and padding");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  const int64_t N = input_arg.size(0);
  const int64_t C = input_arg.size(1);
  const int64_t H = input_arg.size(2);
  const int64_t W = input_arg.size(3);

  const auto div_ceil = [](int64_t a, int64_t b) {
    return (a + b - 1) / b;
  };

  const int64_t OH = ceil_mode
      ? div_ceil(H + 2 * padding[1] - kernel_size[1], stride[1]) + 1
      : (H + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
  const int64_t OW = ceil_mode
      ? div_ceil(W + 2 * padding[0] - kernel_size[0], stride[0]) + 1
      : (W + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;

  TP_CHECK(OH > 0 && OW > 0, "Vulkan pooling: computed output size is empty");

  api::vTensor v_output{
      context,
      {N, C, OH, OW},
      input_arg.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan pooling requires texture storage");
  }

  const struct Pool2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    ivec2 kernel;
    ivec2 stride;
    ivec2 padding;
    int c_depth;
    int count_include_pad;
    float divisor_override;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec2(
          static_cast<int32_t>(kernel_size[0]),
          static_cast<int32_t>(kernel_size[1])),
      ivec2(
          static_cast<int32_t>(stride[0]),
          static_cast<int32_t>(stride[1])),
      ivec2(
          static_cast<int32_t>(padding[0]),
          static_cast<int32_t>(padding[1])),
      c_depth_of(v_input.sizes()),
      1,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(max_pool2d),
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
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor avg_pool2d_kernel(
    const Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  validate_pool_input(input);

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);

  const auto div_ceil = [](int64_t a, int64_t b) {
    return (a + b - 1) / b;
  };

  const int64_t OH = ceil_mode
      ? div_ceil(H + 2 * padding[1] - kernel_size[1], stride[1]) + 1
      : (H + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
  const int64_t OW = ceil_mode
      ? div_ceil(W + 2 * padding[0] - kernel_size[0], stride[0]) + 1
      : (W + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;

  TP_CHECK(OH > 0 && OW > 0, "Vulkan pooling: computed output size is empty");

  api::vTensor v_output{
      context,
      {N, C, OH, OW},
      input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan pooling requires texture storage");
  }

  const struct Pool2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    ivec2 kernel;
    ivec2 stride;
    ivec2 padding;
    int c_depth;
    int count_include_pad;
    float divisor_override;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec2(
          static_cast<int32_t>(kernel_size[0]),
          static_cast<int32_t>(kernel_size[1])),
      ivec2(
          static_cast<int32_t>(stride[0]),
          static_cast<int32_t>(stride[1])),
      ivec2(
          static_cast<int32_t>(padding[0]),
          static_cast<int32_t>(padding[1])),
      c_depth_of(v_input.sizes()),
      count_include_pad ? 1 : 0,
      static_cast<float>(
          divisor_override.value_or(0)),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(avg_pool2d), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor max_pool2d_kernel(
    const Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    bool ceil_mode) {
  TP_CHECK(
      dilation.size() == 2 &&
          dilation[0] == 1 && dilation[1] == 1,
      "Vulkan max_pool2d does not support dilation yet");
  return pool2d_impl(input, kernel_size, stride, padding, ceil_mode);
}

Tensor adaptive_avg_pool2d_kernel(
    const Tensor& input,
    const std::vector<int64_t>& output_size) {
  validate_pool_input(input);

  TP_CHECK(
      output_size.size() == 2,
      "Vulkan adaptive_avg_pool2d expects a 2d output size");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);

  api::vTensor v_output{
      context,
      {input.size(0), input.size(1), output_size[1], output_size[0]},
      input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan pooling requires texture storage");
  }

  const struct AdaptivePool2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int c_depth;
    int fill0;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(adaptive_avg_pool2d), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, PoolingKernels) {
  m.impl("avg_pool2d", &tensorplay::vulkan::ops::avg_pool2d_kernel);
  m.impl("max_pool2d", &tensorplay::vulkan::ops::max_pool2d_kernel);
  m.impl("adaptive_avg_pool2d", &tensorplay::vulkan::ops::adaptive_avg_pool2d_kernel);
}

#endif /* USE_VULKAN */
