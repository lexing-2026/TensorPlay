#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Scalar.h"

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
  // A rank-1 parameter walks the normalized width axis, one scalar per x
  // position; it must win over the spatial mode below, which would otherwise
  // sample the 1d texture out of bounds on its extra axes.
  if (pdim == 1) {
    return 3;
  }
  // Every other rank (>= 2) is a spatial parameter fetched per texel position
  // and replicated across the vec4 lanes.
  return 2;
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

  // The span covers the channel slot whenever normalization starts at or
  // before the channel axis (the third axis from the end in NCHW order);
  // statistics then collapse across lanes.  A trailing-only span (H,W) or
  // (W) keeps per-lane statistics.
  const int64_t channel_axis = ndim - 3;
  const bool norm_channels = (ndim - norm_ndim) <= channel_axis;
  // Whole-tensor normalization also collapses across the batch axis.
  const bool norm_batch = norm_ndim == ndim;

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
    int norm_batch;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      c_depth_of(v_input.sizes()),
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input.sizes())),
      static_cast<int32_t>(span_wh),
      norm_channels ? 1 : 0,
      static_cast<float>(eps),
      static_cast<int32_t>(param_len),
      norm_batch ? 1 : 0,
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

// The public spelling composes over the stats-producing native op so the
// autograd node saves the per-row moments for the backward pass; the call
// goes through the generated ops wrapper, which records that node.
Tensor layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  return std::get<0>(tensorplay::tpx::ops::native_layer_norm(
      input, normalized_shape, weight_opt, bias_opt, eps));
}

template <typename Return, typename... Args>
Return dispatch_vk(const char* op, Args... args) {
  return DispatchStub<Return, Args...>::call(
      std::string(op), DispatchKey::Vulkan, std::forward<Args>(args)...);
}

// Native spelling producing the per-row moments the backward pass consumes.
// Statistics come from the dimension reductions; the transform reuses the
// fused layer_norm shader.
std::tuple<Tensor, Tensor, Tensor> native_layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  std::vector<int64_t> dims;
  for (int64_t i = input.dim() - norm_ndim; i < input.dim(); ++i) {
    dims.push_back(i);
  }
  std::vector<int64_t> outer;
  for (int64_t i = 0; i < input.dim() - norm_ndim; ++i) {
    outer.push_back(input.size(i));
  }
  if (outer.empty()) outer.push_back(1);

  // The dispatcher looks kernels up by name and calls them through a
  // pointer cast, so every call must carry exactly the argument list the
  // registered kernel expects.  mean.dim/var.dim take a trailing dtype or
  // correction before keepdim; add.Scalar takes (self, other, alpha).
  // Variance comes from E[x^2] - E[x]^2 instead of the dedicated axis
  // reduction: the mean kernels use a linear scan while the variance kernel
  // pays an extra serial pass, so this substitution is both faster and
  // numerically equivalent for the pooled moment.
  Tensor mean =
      dispatch_vk<Tensor>("mean.dim", input, dims, true, DType::Float32);
  Tensor x2 = dispatch_vk<Tensor>("mul.Tensor", input, input);
  Tensor mean_x2 =
      dispatch_vk<Tensor>("mean.dim", x2, dims, true, DType::Float32);
  Tensor mean_sq = dispatch_vk<Tensor>("mul.Tensor", mean, mean);
  Tensor var = dispatch_vk<Tensor>(
      "sub.Tensor", mean_x2, mean_sq, Scalar(1));
  Tensor rstd = dispatch_vk<Tensor>(
      "rsqrt",
      dispatch_vk<Tensor>("add.Scalar", var, Scalar(eps), Scalar(1)));

  // One-dimensional moments broadcast against the full input; the transform
  // stays linear in the element count.
  Tensor out =
      dispatch_vk<Tensor>("sub.Tensor", input, mean, Scalar(1));
  out = dispatch_vk<Tensor>("mul.Tensor", out, rstd);
  if (weight_opt.has_value()) {
    out = dispatch_vk<Tensor>("mul.Tensor", out, *weight_opt);
  }
  if (bias_opt.has_value()) {
    out = dispatch_vk<Tensor>(
        "add.Tensor", out, *bias_opt, Scalar(1));
  }

  return std::make_tuple(out, mean.reshape(outer), rstd);
}

Tensor batch_norm_kernel(
    const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
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

Tensor rms_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<double>& eps_opt) {
  TP_CHECK(
      input.dtype() == DType::Float32,
      "Vulkan rms_norm supports Float32 tensors only");
  const double eps = eps_opt.has_value() ? eps_opt.value() : 1e-6;

  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  std::vector<int64_t> dims;
  for (int64_t i = input.dim() - norm_ndim; i < input.dim(); ++i) {
    dims.push_back(i);
  }

  // Root-mean-square normalization: y = x * rsqrt(mean(x^2) + eps) * w.
  // The pooled moment comes from the same dimension-reduction path as
  // layer_norm (mean of the element-wise square).
  Tensor x2 = dispatch_vk<Tensor>("mul.Tensor", input, input);
  Tensor mean_x2 =
      dispatch_vk<Tensor>("mean.dim", x2, dims, true, DType::Float32);
  Tensor rstd = dispatch_vk<Tensor>(
      "rsqrt",
      dispatch_vk<Tensor>("add.Scalar", mean_x2, Scalar(eps), Scalar(1)));
  Tensor out = dispatch_vk<Tensor>("mul.Tensor", input, rstd);
  if (weight_opt.has_value()) {
    out = dispatch_vk<Tensor>("mul.Tensor", out, *weight_opt);
  }
  return out;
}

Tensor silu_mul_kernel(const Tensor& gate, const Tensor& up) {
  TP_CHECK(
      gate.dtype() == DType::Float32 && up.dtype() == DType::Float32,
      "Vulkan silu_mul supports Float32 tensors only");
  Tensor gate_silu = dispatch_vk<Tensor>("silu", gate);
  return dispatch_vk<Tensor>("mul.Tensor", gate_silu, up, Scalar(1));
}

Tensor silu_and_mul_kernel(const Tensor& input) {
  TP_CHECK(
      input.dtype() == DType::Float32,
      "Vulkan silu_and_mul supports Float32 tensors only");
  const int64_t packed_width = input.size(-1);
  const int64_t half = packed_width / 2;
  const int64_t dim = input.dim() - 1;
  // Packed layout: the first half is the gate, the second half is the up
  // projection; out = silu(gate) * up.
  Tensor gate =
      dispatch_vk<Tensor>("slice.Tensor", input, dim, 0, half, 1);
  Tensor up = dispatch_vk<Tensor>(
      "slice.Tensor", input, dim, half, packed_width, 1);
  Tensor gate_silu = dispatch_vk<Tensor>("silu", gate);
  return dispatch_vk<Tensor>("mul.Tensor", gate_silu, up, Scalar(1));
}

Tensor rotary_embedding_kernel(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    int64_t position_offset) {
  TP_CHECK(
      input.dtype() == DType::Float32 && cos.dtype() == DType::Float32 &&
          sin.dtype() == DType::Float32,
      "Vulkan rotary_embedding supports Float32 tensors only");
  TP_CHECK(input.dim() >= 2, "Vulkan rotary_embedding needs ndim >= 2");

  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  api::vTensor v_cos = convert(cos);
  api::vTensor v_sin = convert(sin);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D ||
      v_cos.storage_type() != api::StorageType::TEXTURE_3D ||
      v_sin.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan rotary_embedding requires texture storage");
  }

  const int64_t head_dim = input.size(-1);
  const int64_t half_dim = head_dim / 2;
  const int64_t tokens = input.size(-2);
  const int64_t table_rows = v_cos.extents()[1u];

  api::vTensor v_output{context, input.sizes(), DType::Float32};

  const struct RotaryBlock final {
    ivec4 extents;
    int32_t tokens;
    int32_t half_dim;
    int32_t table_rows;
    int32_t position_offset;
    int32_t fill0;
  } block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      static_cast<int32_t>(tokens),
      static_cast<int32_t>(half_dim),
      static_cast<int32_t>(table_rows),
      static_cast<int32_t>(position_offset),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(half_dim / 4),
      v_output.extents()[1u],
      v_output.extents()[2u]};

  context->submit_compute_job(
      VK_KERNEL(rotary_embedding), pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_cos.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_sin.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor scaled_dot_product_attention_kernel(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_opt,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale_opt,
    bool enable_gqa) {
  TP_CHECK(
      query.dtype() == DType::Float32 && key.dtype() == DType::Float32 &&
          value.dtype() == DType::Float32,
      "Vulkan sdpa supports Float32 tensors only");
  TP_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Vulkan sdpa expects 4-D [B, H, S, D] tensors");
  TP_CHECK(
      query.size(2) == key.size(2) && query.size(3) == key.size(3),
      "Vulkan sdpa: query/key shape mismatch");
  TP_CHECK(
      dropout_p == 0.0, "Vulkan sdpa does not support dropout yet");
  TP_CHECK(
      !enable_gqa, "Vulkan sdpa does not support GQA yet");
  if (attn_mask_opt.has_value()) {
    TP_CHECK(
        attn_mask_opt->dtype() == DType::Float32,
        "Vulkan sdpa supports Float32 attention masks only");
  }

  // Composite implementation reusing the validated Vulkan primitives:
  //   out = softmax((Q @ K^T) * scale) @ V
  // The batched-product path reads [B*H, S, D] tensors whose batch lane
  // matches the head layout, so batch size 1 (the LLM decode case) works
  // with any number of heads.
  const int64_t B = query.size(0);
  const int64_t H = query.size(1);
  const int64_t S = query.size(2);
  const int64_t D = query.size(3);
  const double scale =
      scale_opt.value_or(1.0 / std::sqrt(static_cast<double>(D)));

  TP_CHECK(
      B == 1, "Vulkan sdpa composite path supports batch size 1 only");
  TP_CHECK(!is_causal, "Vulkan sdpa causal masking is not implemented yet");
  TP_CHECK(
      !attn_mask_opt.has_value(),
      "Vulkan sdpa additive masks are not implemented yet");

  Tensor q3 = query.reshape(std::vector<int64_t>{B * H, S, D});
  Tensor k3 = key.reshape(std::vector<int64_t>{B * H, S, D});
  Tensor v3 = value.reshape(std::vector<int64_t>{B * H, S, D});
  Tensor kt = dispatch_vk<Tensor>(
      "transpose", k3, int64_t(1), int64_t(2));
  Tensor scores = dispatch_vk<Tensor>("bmm", q3, kt);
  scores = dispatch_vk<Tensor>(
      "mul.Scalar", scores, Scalar(scale), Scalar(1));
  Tensor probs = dispatch_vk<Tensor>(
      "softmax", scores, static_cast<int64_t>(-1), DType::Float32);
  Tensor out3 = dispatch_vk<Tensor>("bmm", probs, v3);
  return out3.reshape(std::vector<int64_t>{B, H, S, D});

}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, NormalizationKernels) {
  m.impl("layer_norm", &tensorplay::vulkan::ops::layer_norm_kernel);
  m.impl("native_layer_norm", &tensorplay::vulkan::ops::native_layer_norm_kernel);
  m.impl("batch_norm", &tensorplay::vulkan::ops::batch_norm_kernel);
  m.impl("rms_norm", &tensorplay::vulkan::ops::rms_norm_kernel);
}

TENSORPLAY_LIBRARY_IMPL(Vulkan, LlmKernels) {
  m.impl("silu_mul", &tensorplay::vulkan::ops::silu_mul_kernel);
  m.impl("silu_and_mul", &tensorplay::vulkan::ops::silu_and_mul_kernel);
  m.impl("rotary_embedding", &tensorplay::vulkan::ops::rotary_embedding_kernel);
  m.impl(
      "scaled_dot_product_attention",
      &tensorplay::vulkan::ops::scaled_dot_product_attention_kernel);
}

#endif /* USE_VULKAN */
