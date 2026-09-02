#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>
#include <string>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

//
// Affine parameter addressing modes for layer_norm, resolved from the
// parameter ranks.  Mode 1 fetches the parameter at the invocation position
// (parameter rank equals input rank); mode 2 fetches a trailing (H, W)
// parameter per position and replicates it across lanes; mode 3 fetches a
// 1d parameter along the texel x axis with clamping; mode 0 substitutes the
// identity (ones / zeros).
//
int affine_mode(const std::optional<Tensor>& param, int64_t input_ndim) {
  if (!param.has_value()) {
    return 0;
  }
  const int64_t pdim = param->dim();
  if (pdim == input_ndim) {
    return 1;
  }
  if (pdim == input_ndim - 1) {
    return 2;
  }
  if (pdim == 1) {
    return 3;
  }
  TP_THROW(RuntimeError, "Vulkan layer_norm: unsupported parameter rank");
}

Tensor layer_norm_impl(
    const Tensor& input_arg,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  TP_CHECK(
      input_arg.dtype() == DType::Float32,
      "Vulkan layer_norm supports Float32 tensors only");
  TP_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4,
      "Vulkan layer_norm supports 1d to 4d tensors");
  TP_CHECK(
      !normalized_shape.empty() &&
          static_cast<int64_t>(normalized_shape.size()) <= input_arg.dim(),
      "Vulkan layer_norm: normalized_shape must match trailing dims");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(
        NotImplementedError, "Vulkan layer_norm requires texture storage");
  }

  const int64_t norm_ndim =
      static_cast<int64_t>(normalized_shape.size());
  const int64_t ndim = input_arg.dim();

  // Elements of the normalized span along the width and height slots: every
  // normalized axis except a possibly included channel axis maps onto the
  // texel x/y plane.
  const int64_t span_wh = input_arg.size(ndim - 1) *
      (norm_ndim >= 2 ? input_arg.size(ndim - 2) : 1);

  // The span covers the channel slot only when every input axis is
  // normalized; statistics then collapse across lanes.
  const bool norm_channels = norm_ndim == ndim;

  const int weight_mode = affine_mode(weight_opt, ndim);
  const int bias_mode = affine_mode(bias_opt, ndim);
  const int64_t param_len =
      (weight_mode == 3 && weight_opt.has_value())
      ? weight_opt->numel()
      : ((bias_mode == 3 && bias_opt.has_value()) ? bias_opt->numel() : 0);

  static const char* w_suffix[] = {"", "w", "w2", "w1"};
  static const char* b_suffix[] = {"", "b", "b2", "b1"};
  const std::string name =
      std::string("layer_norm") + w_suffix[weight_mode] + b_suffix[bias_mode];

  const struct LayerNormBlock final {
    ivec4 in_sizes;
    int c_depth;
    int channels;
    int span;
    int norm_channels;
    float eps;
    int weight_len;
    int fill0;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      c_depth_of(v_input.sizes()),
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input.sizes())),
      static_cast<int32_t>(span_wh),
      norm_channels ? 1 : 0,
      static_cast<float>(eps),
      static_cast<int32_t>(param_len),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // Modes 1..3 read the corresponding sampler; mode 0 never touches it but
  // the descriptor layout still requires a bound image, so the input is
  // bound again as a filler.
  const bool bind_weight = weight_mode != 0;
  const bool bind_bias = bias_mode != 0;

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(name.c_str()),
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
      bind_weight
          ? convert(weight_opt.value())
                .image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      bind_bias
          ? convert(bias_opt.value())
                .image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> bias_opt,
    double eps) {
  return layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
}

Tensor batch_norm_kernel(
    const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> bias_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    bool training,
    double momentum,
    double eps) {
  (void)momentum;
  TP_CHECK(
      !training,
      "Vulkan batch_norm only supports inference (training=false)");
  TP_CHECK(
      input.dtype() == DType::Float32,
      "Vulkan batch_norm supports Float32 tensors only");
  TP_CHECK(
      input.dim() >= 2 && input.dim() <= 4,
      "Vulkan batch_norm supports 2d to 4d tensors");
  TP_CHECK(
      running_mean_opt.has_value() && running_var_opt.has_value(),
      "Vulkan batch_norm requires running statistics");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);

  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(
        NotImplementedError, "Vulkan batch_norm requires texture storage");
  }

  const bool has_weight = weight_opt.has_value();
  const bool has_bias = bias_opt.has_value();

  const char* variant = has_weight
      ? (has_bias ? "batchnorm" : "batchnorm_w")
      : (has_bias ? "batchnorm_b" : "batchnorm_nowb");

  const struct BatchNormBlock final {
    ivec4 in_sizes;
    int c_depth;
    int channels;
    float eps;
    int fill0;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      c_depth_of(v_input.sizes()),
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input.sizes())),
      static_cast<float>(eps),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(variant),
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
      has_weight
          ? convert(weight_opt.value())
                .image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      has_bias
          ? convert(bias_opt.value())
                .image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      convert(running_mean_opt.value())
          .image(pipeline_barrier, api::PipelineStage::COMPUTE),
      convert(running_var_opt.value())
          .image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, NormalizationKernels) {
  m.impl("layer_norm", &tensorplay::vulkan::ops::layer_norm_kernel);
  m.impl("batch_norm", &tensorplay::vulkan::ops::batch_norm_kernel);
}

#endif /* USE_VULKAN */
