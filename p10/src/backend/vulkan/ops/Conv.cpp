#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <optional>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_conv_input(const Tensor& input) {
  TP_CHECK(
      input.dtype() == DType::Float32,
      "Vulkan convolution supports Float32 tensors only");
}

// Uploads a weight/bias CPU tensor into texture storage.  Convolution
// parameters stay on the host between calls; converting per invocation is
// acceptable because the parameter textures are tiny compared to the
// activations, but a persistent cache can be layered on later.
api::vTensor to_vtensor(
    const Tensor& param,
    const api::GPUMemoryLayout layout) {
  api::Context* const context = api::context();

  // Parameters usually arrive on the host; a Vulkan-device parameter (the
  // common case once the caller moved it) is used through its existing
  // payload without a re-upload.
  if (param.device().is_vulkan()) {
    return convert(param);
  }

  api::vTensor v{context, static_cast<std::vector<int64_t>>(param.shape()),
                 param.dtype(), api::StorageType::TEXTURE_3D, layout};

  // Host bytes live in logical order while the texture expects the
  // channel-packed layout, so the upload goes through the packing step
  // instead of a raw byte copy.
  Tensor packed = utils::nchw_to_nc4hw(param.contiguous());
  utils::upload_host_bytes(
      v, packed.impl()->storage().data(), packed.numel() * packed.itemsize());
  return v;
}

void validate_conv_args(
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation) {
  TP_CHECK(stride.size() == 2, "Vulkan conv2d expects a 2d stride");
  TP_CHECK(padding.size() == 2, "Vulkan conv2d expects 2d padding");
  TP_CHECK(dilation.size() == 2, "Vulkan conv2d expects 2d dilation");
  TP_CHECK(
      stride[0] > 0 && stride[1] > 0 && dilation[0] > 0 && dilation[1] > 0,
      "Vulkan conv2d: stride and dilation must be positive");
}

} // namespace

Tensor conv2d_kernel(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    std::optional<Tensor> bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  validate_conv_input(input_arg);
  validate_conv_args(stride, padding, dilation);

  TP_CHECK(input_arg.dim() == 4, "Vulkan conv2d requires a 4d input");
  TP_CHECK(weight_arg.dim() == 4, "Vulkan conv2d requires a 4d weight");
  TP_CHECK(
      groups == 1 || groups == input_arg.size(1),
      "Vulkan conv2d only supports groups == 1 (regular) or groups == C "
      "(depthwise)");
  const bool has_bias = bias.has_value() && bias->defined();


  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  const int64_t N = input_arg.size(0);
  const int64_t C = input_arg.size(1);
  const int64_t H = input_arg.size(2);
  const int64_t W = input_arg.size(3);

  const int64_t KH = weight_arg.size(2);
  const int64_t KW = weight_arg.size(3);

  const auto div_ceil = [](int64_t a, int64_t b) {
    return (a + b - 1) / b;
  };

  const int64_t OH =
      (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) / stride[1] + 1;
  const int64_t OW =
      (W + 2 * padding[0] - dilation[0] * (KW - 1) - 1) / stride[0] + 1;

  TP_CHECK(OH > 0 && OW > 0, "Vulkan conv2d: computed output size is empty");

  api::vTensor v_output{
      context,
      {N, groups == 1 ? weight_arg.size(0) : C, OH, OW},
      input_arg.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan conv2d requires texture storage");
  }

  api::PipelineBarrier pipeline_barrier{};

  if (groups == C && C != 1) {
    // Depthwise path: one filter plane per channel.
    api::vTensor v_weight = to_vtensor(
        weight_arg, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = to_vtensor(
          *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
    }

    const struct Conv2DBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      ivec4 weight_sizes;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilation;
      int in_c_depth;
      int out_c_depth;
      int weight_c_depth;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(C),
            1,
            static_cast<int32_t>(KH),
            static_cast<int32_t>(KW)),
        ivec2(
            static_cast<int32_t>(stride[0]),
            static_cast<int32_t>(stride[1])),
        ivec2(
            static_cast<int32_t>(padding[0]),
            static_cast<int32_t>(padding[1])),
        ivec2(
            static_cast<int32_t>(dilation[0]),
            static_cast<int32_t>(dilation[1])),
        c_depth_of(v_input.sizes()),
        c_depth_of(v_output.sizes()),
        1,
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL_FROM_STR(has_bias ? "conv2d_dw" : "conv2d_dw_nobias"),
        pipeline_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        has_bias
            ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
            : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else if (KH == 1 && KW == 1 && stride[0] == 1 && stride[1] == 1 &&
             padding[0] == 0 && padding[1] == 0 && dilation[0] == 1 &&
             dilation[1] == 1) {
    // Pointwise path: a per-position matrix product over the channel axis.
    TP_CHECK(
        groups == 1, "Vulkan conv2d: pointwise path requires groups == 1");

    const int64_t O = weight_arg.size(0);
    // The 1x1 weight keeps its {O, C, 1, 1} shape: reshaping a Vulkan
    // parameter would only produce a strided view over the same texture.
    api::vTensor v_weight = to_vtensor(
        weight_arg, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = to_vtensor(
          *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
    }

    const struct Conv1x1Block final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      ivec4 weight_sizes;
      int in_c_depth;
      int out_c_depth;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(O),
            static_cast<int32_t>(C),
            1,
            1),
        c_depth_of(v_input.sizes()),
        c_depth_of(v_output.sizes()),
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL_FROM_STR(has_bias ? "conv2d_pw" : "conv2d_pw_nobias"),
        pipeline_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        has_bias
            ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
            : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    // Regular grouped (single group) path.
    TP_CHECK(
        groups == 1, "Vulkan conv2d: only groups == 1 or depthwise supported");
    TP_CHECK(
        dilation[0] == 1 && dilation[1] == 1,
        "Vulkan conv2d: the regular path does not support dilation yet");

    const int64_t O = weight_arg.size(0);
    api::vTensor v_weight = to_vtensor(
        weight_arg, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = to_vtensor(
          *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
    }

    const struct Conv2DBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      ivec4 weight_sizes;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilation;
      int in_c_depth;
      int out_c_depth;
      int weight_c_depth;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(O),
            static_cast<int32_t>(C),
            static_cast<int32_t>(KH),
            static_cast<int32_t>(KW)),
        ivec2(
            static_cast<int32_t>(stride[0]),
            static_cast<int32_t>(stride[1])),
        ivec2(
            static_cast<int32_t>(padding[0]),
            static_cast<int32_t>(padding[1])),
        ivec2(
            static_cast<int32_t>(dilation[0]),
            static_cast<int32_t>(dilation[1])),
        c_depth_of(v_input.sizes()),
        c_depth_of(v_output.sizes()),
        c_depth_of(v_input.sizes()),
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL_FROM_STR(has_bias ? "conv2d" : "conv2d_nobias"),
        pipeline_barrier,
        v_output.extents(),
        adaptive_work_group_size(v_output.extents()),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        has_bias
            ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
            : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_output);
}

Tensor conv_transpose2d_kernel(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    std::optional<Tensor> bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    int64_t groups,
    const std::vector<int64_t>& dilation) {
  validate_conv_input(input_arg);
  const bool has_bias = bias.has_value() && bias->defined();


  TP_CHECK(input_arg.dim() == 4, "Vulkan conv_transpose2d needs a 4d input");
  TP_CHECK(weight_arg.dim() == 4, "Vulkan conv_transpose2d needs a 4d weight");
  TP_CHECK(
      stride.size() == 2 && padding.size() == 2 &&
          output_padding.size() == 2 && dilation.size() == 2,
      "Vulkan conv_transpose2d expects 2d stride/padding/output_padding/"
      "dilation");
  TP_CHECK(
      groups == 1,
      "Vulkan conv_transpose2d only supports groups == 1");
  TP_CHECK(
      dilation[0] == 1 && dilation[1] == 1,
      "Vulkan conv_transpose2d does not support dilation yet");
  TP_CHECK(
      stride[0] > 0 && stride[1] > 0,
      "Vulkan conv_transpose2d: stride must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  const int64_t N = input_arg.size(0);
  const int64_t C = input_arg.size(1);
  const int64_t H = input_arg.size(2);
  const int64_t W = input_arg.size(3);

  const int64_t O = weight_arg.size(1);
  const int64_t KH = weight_arg.size(2);
  const int64_t KW = weight_arg.size(3);

  const int64_t OH =
      (H - 1) * stride[1] - 2 * padding[1] + KH + output_padding[1];
  const int64_t OW =
      (W - 1) * stride[0] - 2 * padding[0] + KW + output_padding[0];

  TP_CHECK(
      OH > 0 && OW > 0,
      "Vulkan conv_transpose2d: computed output size is empty");

  api::vTensor v_output{
      context,
      {N, O, OH, OW},
      input_arg.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(
        NotImplementedError,
        "Vulkan conv_transpose2d requires texture storage");
  }

  api::vTensor v_weight = to_vtensor(
      weight_arg, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  std::optional<api::vTensor> v_bias;
  if (has_bias) {
    v_bias = to_vtensor(
        *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  }

  const struct ConvTranspose2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    ivec4 weight_sizes;
    ivec2 stride;
    ivec2 padding;
    ivec2 output_padding;
    int in_c_depth;
    int out_c_depth;
    int weight_c_depth;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec4(
          static_cast<int32_t>(C),
          static_cast<int32_t>(O),
          static_cast<int32_t>(KH),
          static_cast<int32_t>(KW)),
      ivec2(
          static_cast<int32_t>(stride[0]),
          static_cast<int32_t>(stride[1])),
      ivec2(
          static_cast<int32_t>(padding[0]),
          static_cast<int32_t>(padding[1])),
      ivec2(
          static_cast<int32_t>(output_padding[0]),
          static_cast<int32_t>(output_padding[1])),
      c_depth_of(v_input.sizes()),
      c_depth_of(v_output.sizes()),
      c_depth_of(v_output.sizes()),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(has_bias ? "conv_transpose2d" : "conv_transpose2d_nobias"),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      has_bias
          ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor conv1d_kernel(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    std::optional<Tensor> bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  validate_conv_input(input_arg);

  TP_CHECK(input_arg.dim() == 3, "Vulkan conv1d requires a 3d input");
  TP_CHECK(weight_arg.dim() == 3, "Vulkan conv1d requires a 3d weight");
  TP_CHECK(stride.size() >= 1, "Vulkan conv1d expects a stride");
  TP_CHECK(padding.size() >= 1, "Vulkan conv1d expects padding");
  TP_CHECK(dilation.size() >= 1, "Vulkan conv1d expects dilation");
  TP_CHECK(
      groups == 1, "Vulkan conv1d only supports groups == 1");
  const bool has_bias = bias.has_value() && bias->defined();


  api::Context* const context = api::context();

  api::vTensor v_input = convert(input_arg);

  const int64_t N = input_arg.size(0);
  const int64_t C = input_arg.size(1);
  const int64_t L = input_arg.size(2);

  const int64_t O = weight_arg.size(0);
  const int64_t K = weight_arg.size(2);

  const auto div_ceil_l = [](int64_t a, int64_t b) {
    return (a + b - 1) / b;
  };

  const int64_t OL =
      (L + 2 * padding[0] - dilation[0] * (K - 1) - 1) / stride[0] + 1;
  TP_CHECK(OL > 0, "Vulkan conv1d: computed output size is empty");
  (void)div_ceil_l;

  api::vTensor v_output{context, {N, O, OL}, input_arg.dtype()};

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan conv1d requires texture storage");
  }

  api::vTensor v_weight = to_vtensor(
      weight_arg, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  std::optional<api::vTensor> v_bias;
  if (has_bias) {
    v_bias = to_vtensor(
        *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  }

    const struct Conv2DBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      ivec4 weight_sizes;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilation;
      int in_c_depth;
      int out_c_depth;
      int weight_c_depth;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(O),
            static_cast<int32_t>(C),
            1,
            static_cast<int32_t>(K)),
        ivec2(
            static_cast<int32_t>(stride[0]),
            1),
        ivec2(
            static_cast<int32_t>(padding[0]),
            0),
        ivec2(
            static_cast<int32_t>(dilation[0]),
            1),
        c_depth_of(v_input.sizes()),
        c_depth_of(v_output.sizes()),
        c_depth_of(v_input.sizes()),
    };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(has_bias ? "conv1d" : "conv1d_nobias"),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      has_bias
          ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ConvKernels) {
  m.impl("conv1d", &tensorplay::vulkan::ops::conv1d_kernel);
  m.impl("conv2d", &tensorplay::vulkan::ops::conv2d_kernel);
  m.impl("conv_transpose2d", &tensorplay::vulkan::ops::conv_transpose2d_kernel);
}

#endif /* USE_VULKAN */
