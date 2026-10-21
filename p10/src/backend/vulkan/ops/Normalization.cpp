#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Dispatcher.h"
#include "../../composite/AttentionComposite.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Scalar.h"

#include <SizesAndStrides.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace vulkan {
namespace ops {

template <typename Return, typename... Args>
Return dispatch_vk(const char* op, Args... args);

// The smallest maxComputeWorkGroupCount any conformant device must report per
// dispatch axis.
constexpr int64_t kLinearDispatchAxisLimit = 65535;

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

// Defined in View.cpp; writes a strided view of a payload as a dense one,
// which is how a result computed in linear storage repacks into an image.
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset);

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

  const int64_t norm_ndim =
      static_cast<int64_t>(normalized_shape.size());
  const int64_t ndim = input_arg.dim();

  // Buffer-stored payloads (any axis beyond the 3-D image limit) normalize
  // through a linear composite: keepdim mean, broadcast subtract, mean of
  // squares, rsqrt, then the affine terms.  Every step routes through a
  // buffer-capable elementwise/reduction kernel.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(norm_ndim));
    for (int64_t i = ndim - norm_ndim; i < ndim; ++i) {
      dims.push_back(i);
    }
    Tensor mean = dispatch_vk<Tensor>(
        "mean.dim", input_arg, dims, true, DType::Float32);
    Tensor centered =
        dispatch_vk<Tensor>("sub.Tensor", input_arg, mean, Scalar(1.0));
    Tensor centered_sq =
        dispatch_vk<Tensor>("mul.Tensor", centered, centered);
    Tensor var = dispatch_vk<Tensor>(
        "mean.dim", centered_sq, dims, true, DType::Float32);
    Tensor rstd = dispatch_vk<Tensor>(
        "rsqrt",
        dispatch_vk<Tensor>("add.Scalar", var, Scalar(eps), Scalar(1)));
    Tensor out = dispatch_vk<Tensor>("mul.Tensor", centered, rstd);
    if (weight_opt.has_value()) {
      out = dispatch_vk<Tensor>("mul.Tensor", out, weight_opt.value());
    }
    if (bias_opt.has_value()) {
      out = dispatch_vk<Tensor>("add.Tensor", out, bias_opt.value());
    }
    return out;
  }

  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(
        NotImplementedError, "Vulkan layer_norm requires texture storage");
  }

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

  // The choice belongs to the operand, not to the result: this branch never
  // releases an allocation of its own, and an input already stored as words
  // cannot be sampled by the fused image kernel below.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    // Linear composite: (x - mean) / sqrt(var + eps) * weight + bias with
    // per-channel statistics reshaped to broadcast over the spatial axes.
    // Every step routes through a buffer-capable elementwise kernel.
    const int64_t channels = get_dim<Dim4D::Channel>(v_input.sizes());
    const std::vector<int64_t> stat_shape{1, channels, 1, 1};
    Tensor mean = dispatch_vk<Tensor>(
        "reshape", running_mean_opt.value(), stat_shape);
    Tensor var = dispatch_vk<Tensor>(
        "reshape", running_var_opt.value(), stat_shape);
    Tensor rstd = dispatch_vk<Tensor>(
        "rsqrt",
        dispatch_vk<Tensor>("add.Scalar", var, Scalar(eps), Scalar(1)));
    Tensor centered = dispatch_vk<Tensor>(
        "sub.Tensor", input, mean, Scalar(1.0));
    Tensor out = dispatch_vk<Tensor>("mul.Tensor", centered, rstd);
    if (weight_opt.has_value()) {
      out = dispatch_vk<Tensor>(
          "mul.Tensor", out,
          dispatch_vk<Tensor>("reshape", weight_opt.value(), stat_shape));
    }
    if (bias_opt.has_value()) {
      out = dispatch_vk<Tensor>(
          "add.Tensor", out,
          dispatch_vk<Tensor>("reshape", bias_opt.value(), stat_shape),
          Scalar(1));
    }
    return out;
  }

  api::vTensor v_output{context, v_input.sizes(), v_input.dtype()};

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

//
// The same normalization over linear storage.  A payload longer than the image
// axis limit has no texel row to scan, and the composite of square, mean, add,
// rsqrt and two multiplies moves the row through memory five times.  One
// workgroup of 256 invocations owns one row instead: it strides the row once to
// fold the squares, tree-reduces them through about 1 KB of shared memory, and
// scales the values it still holds in registers.  The row index sits on the
// workgroup axes and spills to the third one, because the row count is not
// bounded by a single dispatch axis.
Tensor rms_norm_buffer(
    const Tensor& input_arg,
    const api::vTensor& v_input,
    int64_t span,
    const std::optional<Tensor>& weight_opt,
    double eps) {
  const bool is_half = input_arg.dtype() == DType::Float16;
  TP_CHECK(
      input_arg.dtype() == DType::Float32 || is_half,
      "Vulkan rms_norm over linear storage supports Float32 and Float16 "
      "tensors only");
  TP_CHECK(
      span > 0 && input_arg.numel() % span == 0,
      "Vulkan rms_norm over linear storage needs a normalized axis that tiles "
      "the payload");
  // Two adjacent half lanes share one 32-bit word, and each invocation writes
  // the whole word it owns.  An odd axis would put the tail of one row and the
  // head of the next in the same word.
  TP_CHECK(
      !is_half || span % 2 == 0,
      "Vulkan rms_norm over linear storage needs an even normalized axis for a "
      "Float16 payload");

  const int64_t rows = input_arg.numel() / span;
  const int64_t rows_axis =
      rows < kLinearDispatchAxisLimit ? rows : kLinearDispatchAxisLimit;
  const uint32_t rows_pages = api::utils::div_up(
      api::utils::safe_downcast_to_u32(rows),
      static_cast<uint32_t>(rows_axis));
  TP_CHECK(
      static_cast<int64_t>(rows_pages) <= kLinearDispatchAxisLimit,
      "Vulkan rms_norm over linear storage needs the row count within the two "
      "dispatch axes");

  api::Context* const context = api::context();
  // The result is written as a linear payload, so it has to be allocated as
  // one: an output in the default texture class would keep the write in a
  // buffer the image read-back never sees.
  api::vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  std::optional<api::vTensor> v_weight;
  bool weight_linear = false;
  if (weight_opt.has_value()) {
    v_weight = convert(*weight_opt);
    weight_linear = v_weight->storage_type() == api::StorageType::BUFFER;
    // A flat weight is read as raw words of the payload's own width; an image
    // weight is fetched as a texel and its storage format already carries the
    // element type.
    TP_CHECK(
        !weight_linear || v_weight->dtype() == input_arg.dtype(),
        std::string("Vulkan rms_norm over linear storage needs a weight of the "
                    "same payload type as the ") +
            (is_half ? "Float16" : "Float32") + " input");
  }

  const struct RmsNormBufferBlock final {
    int32_t span; // elements of the normalized trailing axis
    int32_t rows; // rows to normalize
    int32_t rows_axis; // rows named by the second axis before the third takes over
    float inv_span; // 1 / span
    float eps;
    int32_t pad0;
    int32_t pad1;
    int32_t pad2;
  } block{
      static_cast<int32_t>(span),
      static_cast<int32_t>(rows),
      static_cast<int32_t>(rows_axis),
      static_cast<float>(1.0 / static_cast<double>(span)),
      static_cast<float>(eps),
      0,
      0,
      0};

  const api::utils::uvec3 grid{
      1u,
      api::utils::safe_downcast_to_u32(rows_axis),
      rows_pages,
  };
  const api::utils::uvec3 local{256u, 1u, 1u};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // The half forms share the block layout and the dispatch shape; only the lane
  // width of every buffer binding differs.
  const std::string kernel_root =
      is_half ? "rms_norm_buffer_half" : "rms_norm_buffer";

  if (!v_weight.has_value()) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_root.c_str()),
        pipeline_barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else if (weight_linear) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR((kernel_root + "_w").c_str()),
        pipeline_barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer(),
        v_weight->buffer(pipeline_barrier, api::PipelineStage::COMPUTE));
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR((kernel_root + "_w_tex").c_str()),
        pipeline_barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer(),
        v_weight->image(pipeline_barrier, api::PipelineStage::COMPUTE));
  }

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

  // The fused single-dispatch paths cover a normalized shape that is exactly
  // the trailing axis with an optional rank-1 weight.  Over texture storage the
  // shader scans a texel row, so it keeps the packed-layout rank bound; over
  // linear storage a row is simply a stride-spanning block of the flat payload,
  // so any rank works.  Everything else (several normalized axes, a
  // higher-rank weight) falls through to the composite, whose elementwise ops
  // (mul, mean, scalar add, rsqrt) all have linear variants.
  api::vTensor v_input = convert(input);
  const bool fused_shape =
      norm_ndim == 1 && ndim >= 1 &&
      normalized_shape[0] == input.size(ndim - 1) &&
      (!weight_opt.has_value() ||
       (weight_opt->dim() == 1 &&
        weight_opt->numel() == normalized_shape[0]));
  if (fused_shape && v_input.storage_type() == api::StorageType::BUFFER) {
    return rms_norm_buffer(
        input, v_input, normalized_shape[0], weight_opt, eps);
  }
  if (fused_shape && ndim <= 4) {
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

//
// Rotary embedding over linear storage.  A sequence longer than the device's
// image axis limit has no texture form at all, and the rotation is elementwise
// in the head dimension, so the payload is consumed as the row-major
// (..., tokens, head_dim) array it already is.  One invocation rotates one pair
// of head dimensions for one token of one batch row:
//   row = batch * tokens + token
//   out[row][2i]   = in[row][2i]     * cos[t][i] - in[row][2i+1] * sin[t][i]
//   out[row][2i+1] = in[row][2i]     * sin[t][i] + in[row][2i+1] * cos[t][i]
// The tables need no texel lanes either, but a cached one can outgrow the image
// limits along with the activation, so both of their storage forms are
// dispatchable.
Tensor rotary_embedding_buffer(
    const Tensor& input,
    const api::vTensor& v_input,
    const api::vTensor& v_cos,
    const api::vTensor& v_sin,
    int64_t half_dim,
    int64_t tokens,
    int64_t table_rows,
    int64_t position_offset) {
  const bool is_half = input.dtype() == DType::Float16;
  TP_CHECK(
      input.dtype() == DType::Float32 || is_half,
      "Vulkan rotary_embedding over linear storage supports Float32 and "
      "Float16 tensors only");

  const int64_t head_dim = 2 * half_dim;
  const int64_t batches = input.numel() / (tokens * head_dim);
  TP_CHECK(
      table_rows == 1 || position_offset + tokens <= table_rows,
      "Vulkan rotary_embedding: the cosine/sine table is shorter than the "
      "requested positions");
  TP_CHECK(
      half_dim <= kLinearDispatchAxisLimit &&
          tokens <= kLinearDispatchAxisLimit &&
          batches <= kLinearDispatchAxisLimit,
      "Vulkan rotary_embedding over linear storage needs every dispatch axis "
      "within the workgroup count limit");

  // A pair of rotated dimensions is two adjacent elements, which for a half
  // payload is exactly one 32-bit word.  A flat table is read as words of the
  // payload's own width, so it has to carry the same element type.
  if (is_half && v_cos.storage_type() == api::StorageType::BUFFER) {
    TP_CHECK(
        v_cos.dtype() == DType::Float16 && v_sin.dtype() == DType::Float16,
        "Vulkan rotary_embedding over linear storage needs Float16 cosine and "
        "sine tables for a Float16 payload");
  }

  api::Context* const context = api::context();
  // The rotation writes a linear payload, so the result has to be allocated as
  // one; a texture-class output would keep the write in a buffer the image
  // read-back never sees.
  api::vTensor v_output{
      context,
      input.sizes(),
      input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const struct RotaryLinearBlock final {
    ivec4 sizes; // (half_dim, tokens, batches, 0)
    int32_t head_dim; // elements of one token along the rotated axis
    int32_t table_rows; // cosine/sine row count (1 = broadcast)
    int32_t table_stride; // element step to the next table row
    int32_t position_offset;
    int32_t fill0;
    int32_t fill1;
  } block{
      ivec4(
          static_cast<int32_t>(half_dim),
          static_cast<int32_t>(tokens),
          static_cast<int32_t>(batches),
          0),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(table_rows),
      static_cast<int32_t>(half_dim),
      static_cast<int32_t>(position_offset),
      0,
      0};

  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(half_dim),
      api::utils::safe_downcast_to_u32(tokens),
      api::utils::safe_downcast_to_u32(batches)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const char* const linear_kernel =
      v_cos.storage_type() == api::StorageType::BUFFER
      ? (is_half ? "rotary_embedding_buffer_half" : "rotary_embedding_buffer")
      : (is_half ? "rotary_embedding_buffer_half_tex_table"
                 : "rotary_embedding_buffer_tex_table");

  if (v_cos.storage_type() == api::StorageType::BUFFER) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_kernel),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_cos.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_sin.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_kernel),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_cos.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_sin.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_output);
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

  const int64_t head_dim = input.size(-1);
  TP_CHECK(
      head_dim % 2 == 0,
      "Vulkan rotary_embedding needs an even head dimension");
  const int64_t half_dim = head_dim / 2;
  const int64_t tokens = input.size(-2);
  // A 1D table broadcasts over every token; a 2D table is [positions, half_dim]
  // and supplies one row per token.  Linear storage has no image extents, so
  // the geometry is read from the logical shapes.
  const int64_t table_rows = cos.dim() >= 2 ? cos.size(-2) : 1;

  TP_CHECK(
      v_cos.storage_type() == v_sin.storage_type(),
      "Vulkan rotary_embedding needs the cosine and sine tables in the same "
      "storage");
  TP_CHECK(
      cos.size(-1) == half_dim,
      "Vulkan rotary_embedding needs cosine/sine tables of head_dim/2 entries "
      "per position");
  const bool table_linear = v_cos.storage_type() == api::StorageType::BUFFER;
  // A linear table is read as raw words, so its element width has to match the
  // way the consumer indexes it: a texture-stored activation reads one float
  // texel-sized lane per entry, while a half linear payload unpacks two lanes
  // out of every word.
  if (table_linear) {
    if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
      TP_CHECK(
          cos.dtype() == input.dtype(),
          "Vulkan rotary_embedding needs cosine/sine tables of the payload "
          "dtype over linear storage");
    } else {
      TP_CHECK(
          cos.dtype() == DType::Float32,
          "Vulkan rotary_embedding needs Float32 cosine/sine tables over "
          "linear storage for a texture-stored payload");
    }
  }

  // An activation with no texture form rotates over its linear payload.
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    return rotary_embedding_buffer(
        input, v_input, v_cos, v_sin, half_dim, tokens, table_rows,
        position_offset);
  }

  // The rotation is elementwise in the head dimension, so the result keeps the
  // input's element type; widening a Float16 activation here would both change
  // the contract and double the bytes written per token.  The matching image
  // format comes from the dtype-selected shader variant.
  api::vTensor v_output{context, input.sizes(), input.dtype()};

  const struct RotaryBlock final {
    ivec4 extents;
    int32_t tokens;
    int32_t half_dim;
    int32_t table_rows;
    int32_t position_offset;
    int32_t table_stride;
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
      static_cast<int32_t>(half_dim)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(half_dim),
      v_output.extents()[1u],
      v_output.extents()[2u]};

  if (table_linear) {
    context->submit_compute_job(
        kernel_for("rotary_embedding_lin_table", input.dtype()),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_cos.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_sin.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        kernel_for("rotary_embedding", input.dtype()),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_cos.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_sin.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

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

// Whether the fused chain can address every payload it needs.  A
// channel-packed image carries the batch and head axes in its depth and the
// two sequence axes as its extent, so a sequence past the device's image
// limit leaves the score matrix with no texture at all.
bool sdpa_images_fit(
    api::Context* const context,
    int64_t batches,
    int64_t heads,
    int64_t query_len,
    int64_t key_len) {
  const uint32_t limit =
      context->adapter_ptr()->properties().limits.maxImageDimension3D;
  return api::utils::safe_downcast_to_u32(key_len) <= limit &&
      api::utils::safe_downcast_to_u32(query_len) <= limit &&
      api::utils::safe_downcast_to_u32(
          static_cast<int64_t>(sdpa_planes(heads, batches))) <= limit;
}

// Additive causal mask: query row t attends the keys up to t, and a rejected
// entry carries the value that contributes nothing once the row is
// normalized.  The two index axes are materialized as grids because a
// comparison kernel addresses both operands by the same linear position.
Tensor sdpa_causal_mask(
    const int64_t query_len,
    const int64_t key_len,
    const DType dtype,
    const Device& device) {
  Tensor index = tpx::ops::arange(
      Scalar(0),
      Scalar(std::max(query_len, key_len)),
      Scalar(1),
      DType::Int64,
      device);
  Tensor rows = tpx::ops::expand(
      tpx::ops::view(
          tpx::ops::narrow(index, 0, 0, query_len), {query_len, 1}),
      {query_len, key_len},
      false);
  Tensor cols = tpx::ops::expand(
      tpx::ops::view(
          tpx::ops::narrow(index, 0, 0, key_len), {1, key_len}),
      {query_len, key_len},
      false);
  Tensor reject = full_kernel(
      {query_len, key_len},
      Scalar(-std::numeric_limits<double>::infinity()),
      dtype,
      device,
      false);
  return dispatch_vk<Tensor>(
      "where.ScalarSelf", tpx::ops::ge(rows, cols), Scalar(0.0), reject);
}

//
// Attention for a payload without an image form.  The two products are the
// batched kernels, which read images and linear payloads alike and settle the
// leading axes between themselves, and the row reduction walks the score
// matrix in the storage it was written to.  The math is the fused chain's: the
// scale splits across both factors of the first product, a row's largest score
// pivots the exponentials, and a row with nothing left to attend comes back
// as zeros.
//
// Grouped query heads need no repetition here.  The group axis sits directly
// next to the query-row axis, so reading the head and the row axes as one
// turns (B, Hq, Sq, D) into (B, Hkv, group * Sq, D): every head of a group
// shares the key/value planes of its own kv head through the batch axes of a
// single product.  The result is read back as four axes the same way.  A mask
// goes through the same merge, after a broadcast onto the full grid when it
// carries fewer planes than the query.
//
Tensor sdpa_composed(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& mask,
    double scale,
    bool enable_gqa) {
  const DType origin = query.dtype();
  // A half payload accumulates in single precision, the same tier the rgba16f
  // builds reach through the hardware format conversion.
  const bool widened = origin == DType::Float16;
  const Tensor q = widened ? tpx::ops::to(query, DType::Float32) : query;
  const Tensor k = widened ? tpx::ops::to(key, DType::Float32) : key;
  const Tensor v = widened ? tpx::ops::to(value, DType::Float32) : value;

  const int64_t batches = q.size(0);
  const int64_t hq = q.size(1);
  const int64_t hkv = k.size(1);
  const int64_t sq = q.size(2);
  const int64_t skv = k.size(2);
  const int64_t group = enable_gqa ? hq / hkv : 1;

  auto merge_heads = [&](const Tensor& t, int64_t rows) {
    if (group == 1) {
      return t;
    }
    return tpx::ops::reshape(
        t, {batches, hkv, group * rows, t.size(-1)});
  };

  const Tensor q_m = merge_heads(q, sq);
  const double root = std::sqrt(std::abs(scale));
  Tensor scores = tpx::ops::matmul(
      tpx::ops::mul(q_m, Scalar(scale < 0.0 ? -root : root)),
      tpx::ops::mul(tpx::ops::transpose(k, -2, -1), Scalar(root)));
  if (mask.has_value()) {
    // The mask arrives in the accumulation tier: single precision for a half
    // payload, the query dtype otherwise.
    Tensor extra = *mask;
    if (group > 1) {
      while (extra.dim() < 4) {
        extra = tpx::ops::unsqueeze(extra, 0);
      }
      if (extra.size(0) != batches || extra.size(1) != hq) {
        extra = tpx::ops::expand(extra, {batches, hq, sq, skv}, false);
      }
      extra = merge_heads(extra, sq);
    }
    scores = tpx::ops::add(scores, extra);
  }

  Tensor pivot = tpx::ops::amax(scores, {-1}, true);
  // A row whose every entry was rejected has no pivot to subtract and no total
  // to divide by.  Taking zero for both keeps its weights at zero, which leaves
  // the row's output at zero instead of at not-a-number.
  pivot = tpx::ops::where(
      tpx::ops::eq(pivot, Scalar(-std::numeric_limits<double>::infinity())),
      Scalar(0.0),
      pivot);
  Tensor weights = tpx::ops::exp(tpx::ops::sub(scores, pivot));
  Tensor total = tpx::ops::sum(weights, {-1}, true);
  Tensor probs = tpx::ops::div(
      weights,
      tpx::ops::where(
          tpx::ops::eq(total, Scalar(0.0)), Scalar(1.0), total));

  Tensor out = tpx::ops::matmul(probs, v);
  if (group > 1) {
    out = tpx::ops::reshape(out, {batches, hq, sq, out.size(-1)});
  }
  return widened ? tpx::ops::to(out, origin) : out;
}

//
// Row tile, key chunk and widest head dimension of the single-walk attention
// kernel; they have to match the constants of its shader.
//
constexpr int32_t kFlashRowTile = 8;
constexpr int32_t kFlashKeyChunk = 16;
constexpr int32_t kFlashMaxHeadDim = 128;
constexpr int32_t kFlashThreads = 128;

// Shared bytes the walk reserves whatever the head dimension: the query tile,
// the key chunk and the accumulator all span the widest row in four-wide
// steps, each row padded by a whole quad so the lanes of a workgroup miss each
// other's bank columns, plus the score block and four running values per row.
// A device whose shared pool is below this cannot back the pipeline at all, so
// the walk stays off such hardware.
constexpr int64_t kFlashSharedBytes =
    (int64_t(2 * kFlashRowTile + kFlashKeyChunk) *
         (kFlashMaxHeadDim / 4 + 1) * 16 +
     (int64_t(kFlashRowTile) * (kFlashKeyChunk + 1) + 4 * kFlashRowTile) * 4);

// Element index range a 32-bit offset in a buffer kernel can reach.
constexpr int64_t kLinearIndexLimit = 2147483647;

// Texel lanes of a score matrix above which that one intermediate costs more
// than an on-device deployment can spare for it (64 MB in single precision).
constexpr int64_t kScoreTexelBudget = 4LL << 20;

// Row tiles the walk needs before the serial pass it takes over the keys, one
// per tile, still leaves the device with enough workgroups to fill.
constexpr int64_t kFlashMinWorkgroups = 1024;

namespace {

// A payload the linear reads can walk: single precision, and dense in logical
// order.  The cast and the stride fix are each one pass over one operand,
// which is what the walk saves by never writing the score matrix down.
Tensor to_float32_linear(const Tensor& t) {
  const Tensor widened =
      t.dtype() == DType::Float32 ? t : tpx::ops::to(t, DType::Float32);
  return widened.is_contiguous() ? widened : widened.contiguous();
}

// Element steps of a dense [..., a, b, c] payload read through four axes: the
// outermost of the four crosses whole matrices.
ivec4 dense_strides(int64_t a, int64_t b, int64_t c) {
  return ivec4(
      static_cast<int32_t>(a * b * c),
      static_cast<int32_t>(b * c),
      static_cast<int32_t>(c),
      1);
}

// The four axes a mask is addressed by, right-aligned and checked against the
// extents it has to cover.  An axis of size one broadcasts, and steps by
// nothing so every position reads the single slot it carries.
ivec4 mask_addressing(
    const std::vector<int64_t>& sizes,
    const int64_t batch,
    const int64_t heads,
    const int64_t query_len,
    const int64_t key_len) {
  const int rank = static_cast<int>(sizes.size());
  TP_CHECK(
      rank <= 4,
      "Vulkan sdpa over linear storage takes an attention mask of at most four axes");
  std::vector<int64_t> padded(4, 1);
  for (int axis = 0; axis < rank; ++axis) {
    padded[4 - rank + axis] = sizes[axis];
  }
  const int64_t expected[4] = {batch, heads, query_len, key_len};
  for (int axis = 0; axis < 4; ++axis) {
    TP_CHECK(
        padded[axis] == 1 || padded[axis] == expected[axis],
        "Vulkan sdpa over linear storage: the attention mask is not broadcastable "
        "to the query");
  }
  int64_t running = 1;
  ivec4 strides{0, 0, 0, 0};
  for (int axis = 3; axis >= 0; --axis) {
    strides[axis] = padded[axis] == 1 ? 0 : static_cast<int32_t>(running);
    running *= padded[axis];
  }
  return strides;
}

//
// Attention in one walk of the key sequence, over payloads read as raw words.
// A workgroup owns a tile of query rows of one (batch, head) pair and keeps
// only that tile's running maximum, exponential sum and weighted value
// accumulator, so the score matrix - one entry per (query row, key) pair,
// which is what a two-product chain has to write down before it can normalize
// - never exists.  Over a long prefill window that intermediate is the largest
// object in the model, and on a device with a few hundred megabytes to spare it
// is what keeps attention from running at all.
//
// The walk is the online form of the row softmax: every chunk rescales what the
// pass has kept so far by the exponential of the pivot it was taken against
// versus the new one, so one pass reaches the same result as normalizing once
// the whole row is known.  Rejected entries (past the causal window, outside
// either sequence, or carrying a rejection in the mask) fall out of both sums
// by themselves, and a row with nothing left to attend comes back as zeros.
//
// Grouped query heads need no repetition: a query head reads the key and value
// rows of head h / group.  The mask arrives as an additive payload read through
// its own strides, so any broadcast of it costs no extra pass.
//
Tensor sdpa_flash(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_opt,
    double scale,
    bool is_causal,
    int64_t group) {
  api::Context* const context = api::context();

  const DType origin = query.dtype();
  const int64_t B = query.size(0);
  const int64_t Hq = query.size(1);
  const int64_t Sq = query.size(2);
  const int64_t Hkv = key.size(1);
  const int64_t Skv = key.size(2);
  const int64_t D = query.size(3);

  // Every address the kernel forms is a 32-bit element offset into a payload.
  TP_CHECK(
      B * Hq * Sq * D <= kLinearIndexLimit &&
          B * Hkv * Skv * D <= kLinearIndexLimit,
      "Vulkan sdpa over linear storage exceeds the element index range a "
      "32-bit offset can reach");

  const Tensor q = to_float32_linear(query);
  const Tensor k = to_float32_linear(key);
  const Tensor v = to_float32_linear(value);

  std::optional<Tensor> mask;
  ivec4 mask_strides{0, 0, 0, 0};
  if (attn_mask_opt.has_value()) {
    mask = to_float32_linear(*attn_mask_opt);
    mask_strides = mask_addressing(
        static_cast<std::vector<int64_t>>(mask->shape()), B, Hq, Sq, Skv);
    TP_CHECK(
        mask->numel() <= kLinearIndexLimit,
        "Vulkan sdpa over linear storage exceeds the element index range a "
        "32-bit offset can reach");
  }

  const api::vTensor v_q = to_buffer_storage_dispatch(context, convert(q));
  const api::vTensor v_k = to_buffer_storage_dispatch(context, convert(k));
  const api::vTensor v_v = to_buffer_storage_dispatch(context, convert(v));
  // A call without a mask reads nothing from the fourth operand binding, but
  // the descriptor still needs memory behind it.
  const api::vTensor v_mask = mask.has_value()
      ? to_buffer_storage_dispatch(context, convert(*mask))
      : v_q;

  const int32_t chunks = static_cast<int32_t>(api::utils::div_up(
      api::utils::safe_downcast_to_u32(Skv),
      static_cast<uint32_t>(kFlashKeyChunk)));
  const int32_t d_quads = static_cast<int32_t>(
      api::utils::div_up(api::utils::safe_downcast_to_u32(D), 4u));

  // One workgroup per row tile of one (batch, head) pair.  The launch counts
  // invocations, so the lane axis carries exactly a workgroup's worth and the
  // two axes above it number the tiles and the planes.
  const uint32_t row_tiles = api::utils::safe_downcast_to_u32(
      api::utils::div_up(
          api::utils::safe_downcast_to_u32(Sq),
          static_cast<uint32_t>(kFlashRowTile)));
  const uint32_t planes = api::utils::safe_downcast_to_u32(B * Hq);
  TP_CHECK(
      static_cast<int64_t>(row_tiles) <= kLinearDispatchAxisLimit &&
          static_cast<int64_t>(planes) <= kLinearDispatchAxisLimit,
      "Vulkan sdpa over linear storage needs the row tiles and the head planes "
      "within the two dispatch axes");

  const struct SdpaFlashBlock final {
    ivec4 q_strides; // element steps of the query payload
    ivec4 kv_strides; // steps shared by the key and value payloads
    ivec4 out_strides; // element steps of the result
    ivec4 mask_strides; // steps of the mask, zero on a broadcast axis
    int32_t seq_q; // Sq
    int32_t seq_kv; // Skv
    int32_t dim; // head dimension D
    int32_t chunks; // key chunks the walk covers
    int32_t d_quads; // staging groups per row
    int32_t hq; // query head count
    int32_t group; // query heads per key/value head
    float scale; // multiplier of every score
    int32_t is_causal;
    int32_t has_mask;
    int32_t row_tiles; // tiles the y axis counts, so a tile can number itself
    int32_t fill2;
  } block{
      dense_strides(Hq, Sq, D),
      dense_strides(Hkv, Skv, D),
      dense_strides(Hq, Sq, D),
      mask_strides,
      static_cast<int32_t>(Sq),
      static_cast<int32_t>(Skv),
      static_cast<int32_t>(D),
      chunks,
      d_quads,
      static_cast<int32_t>(Hq),
      static_cast<int32_t>(group),
      static_cast<float>(scale),
      is_causal ? 1 : 0,
      mask.has_value() ? 1 : 0,
      static_cast<int32_t>(row_tiles),
      0};

  api::vTensor v_output{
      context,
      {B, Hq, Sq, D},
      DType::Float32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("sdpa_flash_buffer"),
      barrier,
      api::utils::uvec3{static_cast<uint32_t>(kFlashThreads), row_tiles, planes},
      api::utils::uvec3{static_cast<uint32_t>(kFlashThreads), 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_q.buffer(barrier, api::PipelineStage::COMPUTE),
      v_k.buffer(barrier, api::PipelineStage::COMPUTE),
      v_v.buffer(barrier, api::PipelineStage::COMPUTE),
      v_mask.buffer(barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  // Storage remains a function of the shape alone: a result whose extents still
  // have an image repacks into one on the way out, so a consumer that only
  // reads texels is handed what it expects whichever route produced it.
  const std::vector<int64_t> sizes = {B, Hq, Sq, D};
  Tensor out = convert(v_output);
  if (image_storage_supported(context, sizes)) {
    out = materialize_view(
        out, sizes, SizesAndStrides::compute_contiguous_strides(sizes), 0);
  }
  return origin == DType::Float32 ? out : tpx::ops::to(out, origin);
}

} // namespace

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
  // sampler view, so the fused chain cannot reach it.
  api::vTensor v_query = convert(query);
  api::vTensor v_key = convert(key);
  api::vTensor v_value = convert(value);

  // Both of the routes below write one score per (query row, key) pair before
  // it is normalized: the fused chain hands that matrix to a texture, the
  // composition to a buffer.  The walk keeps it off the device altogether, at
  // the cost of a serial pass over the keys per row tile.  So it runs when the
  // matrix has no image form at all, or when it is large enough that giving it
  // one is the point of the exercise, and the shapes below that keep the
  // per-texel parallelism of the chain.
  const int64_t score_lanes =
      static_cast<int64_t>(sdpa_planes(Hq, B)) * Sq * Skv;
  const int64_t row_tiles =
      static_cast<int64_t>(
          api::utils::div_up(
              api::utils::safe_downcast_to_u32(Sq),
              static_cast<uint32_t>(kFlashRowTile))) *
      B * Hq;
  const bool no_image_form =
      v_query.storage_type() != api::StorageType::TEXTURE_3D ||
      v_key.storage_type() != api::StorageType::TEXTURE_3D ||
      v_value.storage_type() != api::StorageType::TEXTURE_3D ||
      !sdpa_images_fit(context, B, Hq, Sq, Skv);
  const bool walk_reachable =
      D <= kFlashMaxHeadDim && Sq >= kFlashRowTile &&
      row_tiles >= kFlashMinWorkgroups &&
      kFlashSharedBytes <=
          static_cast<int64_t>(
              context->adapter_ptr()
                  ->properties()
                  .limits.maxComputeSharedMemorySize) &&
      B * Hq * Sq * D <= kLinearIndexLimit &&
      B * Hkv * Skv * D <= kLinearIndexLimit;
  if (walk_reachable && (no_image_form || score_lanes > kScoreTexelBudget)) {
    std::optional<Tensor> flash_mask;
    if (attn_mask_opt.has_value()) {
      TP_CHECK(
          attn_mask_opt->dtype() == DType::Bool ||
              attn_mask_opt->dtype() == DType::Float32 ||
              attn_mask_opt->dtype() == DType::Float16,
          "Vulkan sdpa over linear storage: an attention mask must be a bool "
          "or a floating payload");
      Tensor extra = *attn_mask_opt;
      if (!extra.device().is_vulkan()) {
        extra = extra.to(query.device());
      }
      // A rejection of a bool mask arrives as the additive payload the walk
      // reads; an additive one is already in that form.
      flash_mask = extra.dtype() == DType::Bool
          ? additive_float_mask(extra, DType::Float32)
          : extra;
    }
    return sdpa_flash(
        query, key, value, flash_mask, scale, is_causal, group);
  }

  // Without an image for every payload the attention composes instead: the
  // two products are the batched kernels, which stream from images and from
  // linear payloads alike, and the row reduction walks a score matrix that
  // lives in linear storage.  A causal mask rides the same route as an
  // explicit one, so the composite needs no triangle of its own.
  if (no_image_form) {
    TP_CHECK(
        !attn_mask_opt.has_value() ||
            attn_mask_opt->dtype() == DType::Bool ||
            attn_mask_opt->dtype() == query.dtype(),
        "Vulkan sdpa: an attention mask must match the query dtype");
    // The composed chain accumulates a half payload in single precision, so
    // its mask rides the same tier: a half rejection would otherwise have to
    // be selected over linear storage, where the byte-packed formats have no
    // selecting shader yet.
    const DType mask_dtype =
        query.dtype() == DType::Float16 ? DType::Float32 : query.dtype();
    std::optional<Tensor> composed_mask;
    if (is_causal) {
      composed_mask = sdpa_causal_mask(Sq, Skv, mask_dtype, query.device());
    }
    if (attn_mask_opt.has_value()) {
      Tensor extra = *attn_mask_opt;
      if (!extra.device().is_vulkan()) {
        extra = extra.to(query.device());
      }
      if (extra.dtype() == DType::Bool) {
        extra = additive_float_mask(extra, mask_dtype);
      } else if (extra.dtype() != mask_dtype) {
        extra = tpx::ops::to(extra, mask_dtype);
      }
      // Two rejections compose by addition: an entry already past the
      // window stays past it, and an allowed entry keeps its bias.
      composed_mask = composed_mask.has_value()
          ? tpx::ops::add(*composed_mask, extra)
          : extra;
    }
    return sdpa_composed(
        query, key, value, composed_mask, scale, enable_gqa);
  }

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

/*
 * The gradient of the attention above.
 *
 * Nothing is fused here: the scores are taken again from the query, key and
 * value, and the arithmetic that produced the output is the arithmetic that is
 * differentiated.  Every inner operation dispatches on the payload's own
 * backend, so a masked, scaled or longer-context call is differentiable for the
 * same reason a plain one is, and a fully masked query row keeps its row of
 * zeros rather than a number over nothing.
 *
 * A grouped call is the exception.  The key and value are handed back with one
 * head per query head by adding a group axis behind the head axis and reading
 * the two as one axis afterwards, and the expansion leaves a fifth axis that no
 * image can carry.  Until the grouping has a route that stays in rank four, an
 * unequal head count is refused.
 */
std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_attention_backward_kernel(
    const Tensor& grad_output,
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
          (value.dtype() == DType::Float32 || value.dtype() == DType::Float16) &&
          (grad_output.dtype() == DType::Float32 ||
           grad_output.dtype() == DType::Float16),
      "Vulkan sdpa backward supports Float32 and Float16 tensors only");
  TP_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
          grad_output.dim() == 4,
      "Vulkan sdpa backward expects 4-D [B, H, S, D] tensors");
  TP_CHECK(
      query.size(3) == key.size(3) && key.size(3) == value.size(3),
      "Vulkan sdpa backward: Q/K/V must share the head size");
  TP_CHECK(
      key.size(1) == value.size(1) && key.size(2) == value.size(2),
      "Vulkan sdpa backward: key and value shapes must match");
  TP_CHECK(
      static_cast<std::vector<int64_t>>(grad_output.shape()) ==
          static_cast<std::vector<int64_t>>(query.shape()),
      "Vulkan sdpa backward: the gradient must carry the shape the attention "
      "answered with");
  TP_CHECK(
      dropout_p == 0.0, "Vulkan sdpa backward does not support dropout yet");
  TP_CHECK(
      !enable_gqa || query.size(1) == key.size(1),
      "Vulkan sdpa backward: a grouped key/value count needs an expansion that "
      "leaves no fifth axis behind, which no image can carry");
  if (attn_mask_opt.has_value()) {
    TP_CHECK(
        attn_mask_opt->dim() >= 2 &&
            attn_mask_opt->size(attn_mask_opt->dim() - 1) == key.size(2) &&
            attn_mask_opt->size(attn_mask_opt->dim() - 2) == query.size(2),
        "Vulkan sdpa backward: the attention mask must carry the (Sq, Skv) "
        "axes");
  }

  // The scale is the one the scores were built with, so an absent scale means
  // the head width's own; the gradient folds the same factor back onto the
  // query side.
  return composite::sdpa_math_backward_composite(
      grad_output,
      query,
      key,
      value,
      attn_mask_opt,
      dropout_p,
      is_causal,
      scale_opt,
      enable_gqa);
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
  m.impl(
      "scaled_dot_product_attention_backward",
      &tensorplay::vulkan::ops::
          scaled_dot_product_attention_backward_kernel);
}

#endif /* USE_VULKAN */
