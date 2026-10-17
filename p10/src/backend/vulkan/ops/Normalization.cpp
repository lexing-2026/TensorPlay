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
      (input_arg.dtype() == DType::Float32 || input_arg.dtype() == DType::Float16),
      "Vulkan layer_norm supports Float32 and Float16 tensors only");
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

namespace {

// Single-dispatch root-mean-square normalization over the trailing width
// axis.  The fused shader computes per-lane row statistics (the mean of the
// element-wise squares) and applies the optional affine weight in one pass,
// replacing the multi-kernel composite for the common residual-block shape
// where normalized_shape covers exactly the last axis.
Tensor rms_norm_fused(
    const Tensor& input_arg,
    int64_t span,
    const std::optional<Tensor>& weight_opt,
    double eps) {
  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);
  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };
  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(
        NotImplementedError, "Vulkan rms_norm requires texture storage");
  }

  const bool has_weight = weight_opt.has_value();
  const std::string name = std::string("rms_norm") +
      (has_weight ? "w1" : "") +
      (v_input.dtype() == DType::Float16 ? "_f16" : "");

  // The span is exactly the texture width, so one invocation per texel
  // position scans its whole row; the block only carries the normalized
  // span, its reciprocal, and the epsilon.
  const struct RmsNormBlock final {
    ivec4 in_sizes;
    int32_t span;
    float inv_span;
    float eps;
    int32_t pad;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      static_cast<int32_t>(span),
      static_cast<float>(1.0 / static_cast<double>(span)),
      static_cast<float>(eps),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // One workgroup normalizes one texel row: the grid spans the (H x depth)
  // row space, with the 256 invocations of each workgroup striding over the
  // row width in parallel.
  const api::utils::uvec3 global = v_output.extents();
  const api::utils::uvec3 rows{1u, global[1u], global[2u]};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(name.c_str()),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      rows,
      // local work group size
      api::utils::uvec3{256u, 1u, 1u},
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
      // params buffer
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor rms_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const std::optional<Tensor>& weight_opt,
    std::optional<double> eps_opt) {
  TP_CHECK(
      input.dtype() == DType::Float32 || input.dtype() == DType::Float16,
      "Vulkan rms_norm supports Float32 and Float16 tensors only");
  const double eps = eps_opt.has_value() ? eps_opt.value() : 1e-6;

  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  const int64_t ndim = input.dim();

  // Fused single-dispatch path: normalization covers only the trailing axis
  // (the texture width) and any affine parameter is a rank-1 tensor along it.
  const bool fused_ok =
      norm_ndim == 1 && ndim >= 1 && ndim <= 4 &&
      normalized_shape[0] == input.size(ndim - 1) &&
      (!weight_opt.has_value() ||
       (weight_opt->dim() == 1 &&
        weight_opt->numel() == normalized_shape[0]));
  if (fused_ok) {
    return rms_norm_fused(input, normalized_shape[0], weight_opt, eps);
  }

  std::vector<int64_t> dims;
  for (int64_t i = ndim - norm_ndim; i < ndim; ++i) {
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
      (gate.dtype() == DType::Float32 || gate.dtype() == DType::Float16) &&
          (up.dtype() == DType::Float32 || up.dtype() == DType::Float16),
      "Vulkan silu_mul supports Float32 and Float16 tensors only");
  Tensor gate_silu = dispatch_vk<Tensor>("silu", gate);
  return dispatch_vk<Tensor>("mul.Tensor", gate_silu, up);
}

Tensor silu_and_mul_kernel(const Tensor& input) {
  TP_CHECK(
      (input.dtype() == DType::Float32 || input.dtype() == DType::Float16),
      "Vulkan silu_and_mul supports Float32 and Float16 tensors only");
  const int64_t packed_width = input.size(-1);
  const int64_t half = packed_width / 2;
  const int64_t dim = input.dim() - 1;
  // Packed layout: the first half is the gate, the second half is the up
  // projection; out = silu(gate) * up.
  Tensor gate = dispatch_vk<Tensor>(
      "slice.Tensor", input, dim, std::optional<int64_t>(0),
      std::optional<int64_t>(half), int64_t(1));
  Tensor up = dispatch_vk<Tensor>(
      "slice.Tensor", input, dim, std::optional<int64_t>(half),
      std::optional<int64_t>(packed_width), int64_t(1));
  Tensor gate_silu = dispatch_vk<Tensor>("silu", gate);
  return dispatch_vk<Tensor>("mul.Tensor", gate_silu, up);
}

Tensor rotary_embedding_kernel(
    const Tensor& input,
    const Tensor& cos,
    const Tensor& sin,
    int64_t position_offset) {
  TP_CHECK(
      (input.dtype() == DType::Float32 || input.dtype() == DType::Float16) &&
          (cos.dtype() == DType::Float32 || cos.dtype() == DType::Float16) &&
          (sin.dtype() == DType::Float32 || sin.dtype() == DType::Float16),
      "Vulkan rotary_embedding supports Float32 and Float16 tensors only");
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
          static_cast<int32_t>(half_dim),
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
      api::utils::safe_downcast_to_u32(half_dim),
      v_output.extents()[1u],
      v_output.extents()[2u]};

  context->submit_compute_job(
      kernel_for("rotary_embedding", input.dtype()), pipeline_barrier, grid,
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

//
// Fused SDPA: three chained compute passes over channel-packed textures:
//   stage 1  sdpa_qk     scores = Q K^T, scaled, causal-clamped, masked
//   stage 2  softmax     probabilities over the key axis
//   stage 3  sdpa_pv     output = P V
// Every intermediate stays in a texture, so a prefill/decode loop churns no
// host-visible intermediate tensor, and the composite's separate scalar
// multiply, causal mask materialization, and mask add passes all fold into
// stage 1.

// Plane group depth of a channel-packed [B, H, S, D] image: one plane per
// group of four heads.
int32_t sdpa_plane_div4(int64_t heads) {
  return static_cast<int32_t>(api::utils::div_up(
      api::utils::safe_downcast_to_u32(heads), 4u));
}

// Plane axis length of the same image: one head group per plane, once
// per batch.
int32_t sdpa_planes(int64_t heads, int64_t batches) {
  return static_cast<int32_t>(
      batches * static_cast<int64_t>(sdpa_plane_div4(heads)));
}


//
// Additive float mask from a bool mask.  True entries attend (0.0) and
// False ones reject (finitely large negative); a row whose every entry is
// rejected has no negative maximum to pivot on, so the exponent kernel
// normalizes it to a uniform distribution.
//
Tensor additive_float_mask(const Tensor& mask, const DType dtype) {
  const Tensor reject = full_kernel(
      static_cast<std::vector<int64_t>>(mask.shape()),
      Scalar(-std::numeric_limits<double>::infinity()),
      dtype,
      mask.device(),
      false);
  return dispatch_vk<Tensor>("where.ScalarSelf", mask, Scalar(0.0), reject);
}

//
// Mask plane addressing mode for the stage-1 shader, read off the mask's
// leading axes (the trailing axes must already be (Sq, Skv)):
//   mode 0  one shared plane (data in lane 0), broadcast across batches
//           and heads;
//   mode 1  one plane per head, riding the scores planes one to one;
//   mode 2  one plane per batch (data in lane 0), matching the batch field
//           of the score planes;
//   mode 3  one plane per head with a broadcast (size-1 or absent) batch
//           axis, so the head planes cycle per batch.
//
int32_t sdpa_mask_mode(
    const std::vector<int64_t>& mask_sizes,
    const int64_t batch,
    const int64_t heads) {
  const int ndim = static_cast<int>(mask_sizes.size());
  const int64_t lead_h = ndim >= 3 ? mask_sizes[ndim - 3] : 1;
  const int64_t lead_b = ndim == 4 ? mask_sizes[0] : 1;
  if (lead_h == heads) {
    if (lead_b == batch) {
      return 1;
    }
    if (lead_b == 1) {
      return 3;
    }
  } else if (lead_h == 1 && lead_b == 1) {
    return 0;
  } else if (lead_h == 1 && lead_b == batch) {
    return 2;
  }
  TP_THROW(
      ValueError,
      "Vulkan sdpa: the attention mask is not broadcastable to the query");
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
      (query.dtype() == DType::Float32 || query.dtype() == DType::Float16) &&
          (key.dtype() == DType::Float32 || key.dtype() == DType::Float16) &&
          (value.dtype() == DType::Float32 || value.dtype() == DType::Float16),
      "Vulkan sdpa supports Float32 and Float16 tensors only");
  TP_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Vulkan sdpa expects 4-D [B, H, S, D] tensors");
  TP_CHECK(
      query.size(3) == key.size(3) && key.size(3) == value.size(3),
      "Vulkan sdpa: Q/K/V must share the head size");
  TP_CHECK(
      query.size(2) >= 1 || key.size(2) == 0,
      "Vulkan sdpa: an empty key sequence leaves no row to reduce");
  TP_CHECK(
      dropout_p == 0.0, "Vulkan sdpa does not support dropout yet");

  const int64_t B = query.size(0);
  const int64_t Hq = query.size(1);
  const int64_t Sq = query.size(2);
  const int64_t Skv = key.size(2);
  const int64_t Hkv = key.size(1);
  const int64_t D = query.size(3);
  if (enable_gqa) {
    TP_CHECK(
        Hkv < Hq && Hq % Hkv == 0,
        "Vulkan sdpa: enable_gqa requires Hq to be a multiple of Hkv");
  }
  if (attn_mask_opt.has_value()) {
    TP_CHECK(
        attn_mask_opt->dim() >= 2 &&
            attn_mask_opt->size(attn_mask_opt->dim() - 1) == Skv &&
            attn_mask_opt->size(attn_mask_opt->dim() - 2) == Sq,
        "Vulkan sdpa: the attention mask must carry the (Sq, Skv) axes");
  }

  api::Context* const context = api::context();

  // Key/value shape: a single kv head broadcasts across the query heads
  // through the group mapping; other leading shapes carry through their
  // own planes.  Empty tensors need no work.
  TP_CHECK(
      key.size(2) == value.size(2) && key.size(1) == value.size(1),
      "Vulkan sdpa: key and value shapes must match");

  // Head grouping: one kv head broadcasts across the query heads through
  // the group mapping (also true for a single head without the flag);
  // equal counts map heads one to one.  Above one kv head the flag is the
  // only route, and it requires the counts divide.
  const int64_t group = Hq == Hkv ? 1 : (Hq / Hkv);
  // A kv count above one needs the flag with Hq a multiple of Hkv; a
  // single kv head broadcasts regardless of it.
  TP_CHECK(
      Hkv == Hq || enable_gqa || Hkv == 1,
      "Vulkan sdpa: key and value head counts must match or be "
      "broadcastable through enable_gqa");

  const double scale =
      scale_opt.value_or(1.0 / std::sqrt(static_cast<double>(D)));

  // Operands arrive channel-packed; a buffer-resident payload has no
  // sampler view, so the boundary check fails before any dispatch.
  api::vTensor v_query = convert(query);
  api::vTensor v_key = convert(key);
  api::vTensor v_value = convert(value);
  TP_CHECK(
      v_query.storage_type() == api::StorageType::TEXTURE_3D &&
          v_key.storage_type() == api::StorageType::TEXTURE_3D &&
          v_value.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan sdpa requires texture storage");

  // Attention mask: bool converts to an additive float mask; float masks
  // ride the texture planes per the addressing mode.
  Tensor mask_f;
  int32_t mask_mode = 0;
  if (attn_mask_opt.has_value()) {
    mask_f = attn_mask_opt->dtype() == DType::Bool
        ? additive_float_mask(*attn_mask_opt, query.dtype())
        : *attn_mask_opt;
    TP_CHECK(
        mask_f.dtype() == query.dtype(),
        "Vulkan sdpa: an attention mask must match the query dtype");
    if (!mask_f.device().is_vulkan()) {
      mask_f = mask_f.to(query.device());
    }
    mask_mode = sdpa_mask_mode(
        static_cast<std::vector<int64_t>>(mask_f.shape()), B, Hq);
  }

  // Stage 1: scores.  One texel lane per query head, four lanes per plane.
  api::vTensor v_scores{
      context,
      {B, Hq, Sq, Skv},
      query.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };
  TP_CHECK(
      v_scores.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan sdpa requires texture storage");

  const struct SdpaQkBlock final {
    ivec4 extents;      // (Skv, Sq, planes, unused)
    int dim;            // head size D
    float scale;
    int is_causal;
    int has_mask;
    int mask_mode;
    int q_c_depth;      // ceil(Hq / 4)
    int kv_c_depth;     // ceil(Hkv / 4)
    int gqa_group;
    int hq;
  } qk_block{
      ivec4(static_cast<int32_t>(Skv),
            static_cast<int32_t>(Sq),
            sdpa_planes(Hq, B),
            0),
      static_cast<int32_t>(D),
      static_cast<float>(scale),
      is_causal ? 1 : 0,
      attn_mask_opt.has_value() ? 1 : 0,
      mask_mode,
      static_cast<int32_t>(sdpa_plane_div4(Hq)),
      static_cast<int32_t>(sdpa_plane_div4(Hkv)),
      static_cast<int32_t>(group),
      static_cast<int32_t>(Hq),
  };

  api::UniformParamsBuffer qk_params(context, qk_block);
  api::PipelineBarrier barrier{};
  context->submit_compute_job(
      kernel_for("sdpa_qk", query.dtype()),
      barrier,
      v_scores.extents(),
      adaptive_work_group_size(v_scores.extents()),
      VK_NULL_HANDLE,
      v_scores.image(
          barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_query.image(barrier, api::PipelineStage::COMPUTE),
      v_key.image(barrier, api::PipelineStage::COMPUTE),
      // Without a mask, an operand texture (read-only here) occupies the
      // unused sampler slot.
      attn_mask_opt.has_value()
          ? convert(mask_f).image(barrier, api::PipelineStage::COMPUTE)
          : v_key.image(barrier, api::PipelineStage::COMPUTE),
      qk_params.buffer());

  // Stage 2: output.  For short queries over long key ranges the softmax
  // over the key axis splits across memory chunks (scan + combine), which
  // keeps every invocation's serial walk short and the chunk reduction
  // parallel; other shapes fuse the normalization into the single walk:
  // the shader derives the per-row statistics from the scores of the row
  // (the key walk is the texel x axis of a channel-packed [B, Hq, Sq, Skv]
  // image) and accumulates the normalized weights against the value.
  // The probability lanes ride the query heads exactly as stage 1 laid
  // them out, so the group mapping matches the value planes without an
  // epilogue relayout.
  api::vTensor v_output{
      context,
      {B, Hq, Sq, D},
      query.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  const int64_t scan_chunk_len = 128;
  const int64_t scan_chunks =
      Sq <= 8 && Skv > scan_chunk_len
          ? (Skv + scan_chunk_len - 1) / scan_chunk_len
          : 0;
  if (scan_chunks > 1) {
    // Split-memory scan: unnormalized per-chunk accumulators ride a
    // [B, Hq, Sq, chunks * D] image, chunk maxima and sums ride a
    // [B, Hq, Sq, chunks * 2] image, both in Float32.
    api::vTensor v_partial{
        context,
        {B, Hq, Sq, scan_chunks * D},
        DType::Float32,
        api::StorageType::TEXTURE_3D,
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    };
    api::vTensor v_stats{
        context,
        {B, Hq, Sq, scan_chunks * 2},
        DType::Float32,
        api::StorageType::TEXTURE_3D,
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    };

    const struct SdpaScanBlock final {
      ivec4 extents;      // (D, Sq, planes, chunks)
      int skv;            // Skv
      int chunk_len;
      int q_c_depth;      // ceil(Hq / 4)
      int kv_c_depth;     // ceil(Hkv / 4)
      int gqa_group;
      int hq;
    } scan_block{
        ivec4(static_cast<int32_t>(D),
              static_cast<int32_t>(Sq),
              sdpa_planes(Hq, B),
              static_cast<int32_t>(scan_chunks)),
        static_cast<int32_t>(Skv),
        static_cast<int32_t>(scan_chunk_len),
        static_cast<int32_t>(sdpa_plane_div4(Hq)),
        static_cast<int32_t>(sdpa_plane_div4(Hkv)),
        static_cast<int32_t>(group),
        static_cast<int32_t>(Hq),
    };

    api::UniformParamsBuffer scan_params(context, scan_block);
    api::PipelineBarrier scan_barrier{};
    // x packs (chunk, head-dim element): each invocation reduces one
    // chunk into one output texel for the head group.
    const api::utils::uvec3 scan_grid{
        static_cast<uint32_t>(scan_chunks * D),
        static_cast<uint32_t>(Sq),
        static_cast<uint32_t>(sdpa_planes(Hq, B)),
    };
    context->submit_compute_job(
        kernel_for("sdpa_scan", query.dtype()),
        scan_barrier,
        scan_grid,
        adaptive_work_group_size(scan_grid),
        VK_NULL_HANDLE,
        v_scores.image(scan_barrier, api::PipelineStage::COMPUTE),
        v_value.image(scan_barrier, api::PipelineStage::COMPUTE),
        v_partial.image(
            scan_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_stats.image(
            scan_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        scan_params.buffer());

    const struct SdpaCombineBlock final {
      ivec4 extents;      // (D, Sq, planes, unused)
      int chunks;
      int q_c_depth;      // ceil(Hq / 4)
      int kv_c_depth;     // ceil(Hkv / 4)
      int gqa_group;
      int hq;
      int fill1;
      int fill2;
      int fill3;
    } combine_block{
        ivec4(static_cast<int32_t>(D),
              static_cast<int32_t>(Sq),
              sdpa_planes(Hq, B),
              0),
        static_cast<int32_t>(scan_chunks),
        static_cast<int32_t>(sdpa_plane_div4(Hq)),
        static_cast<int32_t>(sdpa_plane_div4(Hkv)),
        static_cast<int32_t>(group),
        static_cast<int32_t>(Hq),
    };

    api::UniformParamsBuffer combine_params(context, combine_block);
    api::PipelineBarrier combine_barrier{};
    context->submit_compute_job(
        kernel_for("sdpa_combine", query.dtype()),
        combine_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            combine_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_partial.image(combine_barrier, api::PipelineStage::COMPUTE),
        v_stats.image(combine_barrier, api::PipelineStage::COMPUTE),
        combine_params.buffer());

    return convert(v_output);
  }

  const struct SdpaPvBlock final {
    ivec4 extents;      // (D, Sq, planes, unused)
    int seq_len;        // Skv
    int q_c_depth;      // ceil(Hq / 4)
    int kv_c_depth;     // ceil(Hkv / 4)
    int gqa_group;
    int hq;
    int fill1;
    int fill2;
    int fill3;
  } pv_block{
      ivec4(static_cast<int32_t>(D),
            static_cast<int32_t>(Sq),
            sdpa_planes(Hq, B),
            0),
      static_cast<int32_t>(Skv),
      static_cast<int32_t>(sdpa_plane_div4(Hq)),
      static_cast<int32_t>(sdpa_plane_div4(Hkv)),
      static_cast<int32_t>(group),
      static_cast<int32_t>(Hq),
  };

  api::UniformParamsBuffer pv_params(context, pv_block);
  api::PipelineBarrier pv_barrier{};
  context->submit_compute_job(
      kernel_for("sdpa_pv", query.dtype()),
      pv_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pv_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_scores.image(pv_barrier, api::PipelineStage::COMPUTE),
      v_value.image(pv_barrier, api::PipelineStage::COMPUTE),
      pv_params.buffer());

  return convert(v_output);
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
