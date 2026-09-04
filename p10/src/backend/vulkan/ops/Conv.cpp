#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "ParamCache.h"
#include "Utils.h"
#include "../impl/Packing.h"

#include <cstring>
#include <functional>
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

//
// Host-side weight packers.  Each produces the byte blob the consuming
// shader streams into its texture; the layouts are documented alongside the
// texel access patterns in glsl/conv2d*.glsl.
//

Tensor pack_identity(const Tensor& param) {
  return param.contiguous();
}

// Cache tags distinguishing the packed forms of one parameter tensor.
enum ParamTag : uint32_t {
  kTagIdentity = 0,
  kTagDepthwiseGroups = 1,
  kTagPointwiseTiles = 2,
  kTagWindowTaps = 3,
  kTagConv1dWidthPacked = 4,
  kTagTranspose2dPacked = 5,
};

Tensor pack_weight_pw(const Tensor& weight) {
  // Pointwise layout: texel (ic4, o4, lane) holds, per component, the weight
  // w[oc = 4*o4 + comp][ic = 4*ic4 + lane].  Stored as logical {4, 4, O4, C4}
  // so the packed-lane convention lands each component where the kernel
  // expects it.
  const int64_t O = weight.size(0);
  const int64_t C = weight.size(1);

  const int64_t O4 = api::utils::align_up(O, 4u) / 4;
  const int64_t C4 = api::utils::align_up(C, 4u) / 4;

  Tensor out({4, 4, O4, C4}, weight.dtype(), Device(DeviceType::CPU));
  float* dst = static_cast<float*>(out.impl()->storage().data());
  const float* src = static_cast<const float*>(weight.contiguous().impl()->storage().data());

  std::memset(dst, 0, out.numel() * sizeof(float));
  for (int64_t oc = 0; oc < O; ++oc) {
    const int64_t o4 = oc / 4;
    const int64_t m = oc % 4;
    for (int64_t ic = 0; ic < C; ++ic) {
      const int64_t ic4 = ic / 4;
      const int64_t lane = ic % 4;
      const float value = src[oc * C + ic];
      dst[((lane * 4 + m) * O4 + o4) * C4 + ic4] = value;
    }
  }
  return out;
}

Tensor pack_weight_sw(const Tensor& weight) {
  // Sliding-window layout: texel (kx, ky, (o4 * C4 + ic4) * 4 + lane) holds,
  // in its four components, w[oc = 4*o4 + comp][ic = 4*ic4 + lane][ky][kx].
  // Stored as logical {Z, 4, KH, KW} with Z = O4 * C4 * 4; the blob's own
  // {C=4} axis is what the NC4HW packing step folds into the texel lanes, so
  // component comp lands in lane comp.
  const int64_t O = weight.size(0);
  const int64_t C = weight.size(1);
  const int64_t KH = weight.size(2);
  const int64_t KW = weight.size(3);

  const int64_t O4 = api::utils::align_up(O, 4u) / 4;
  const int64_t C4 = api::utils::align_up(C, 4u) / 4;
  const int64_t Z = O4 * C4 * 4;

  Tensor out({Z, 4, KH, KW}, weight.dtype(), Device(DeviceType::CPU));
  float* dst = static_cast<float*>(out.impl()->storage().data());
  const float* src = static_cast<const float*>(weight.contiguous().impl()->storage().data());

  std::memset(dst, 0, out.numel() * sizeof(float));
  for (int64_t oc = 0; oc < O; ++oc) {
    const int64_t o4 = oc / 4;
    const int64_t comp = oc % 4;
    for (int64_t ic = 0; ic < C; ++ic) {
      const int64_t ic4 = ic / 4;
      const int64_t lane = ic % 4;
      const float* src_plane =
          src + ((oc * C + ic) * KH) * KW;
      float* dst_plane =
          dst + (((o4 * C4 + ic4) * 4 + lane) * 4 + comp) * KH * KW;
      std::memcpy(
          dst_plane, src_plane, static_cast<size_t>(KH * KW) * sizeof(float));
    }
  }
  return out;
}

std::vector<int64_t> shape_of(const Tensor& t) {
  return static_cast<std::vector<int64_t>>(t.shape());
}

Tensor pack_weight_dw(const Tensor& weight) {
  // Depthwise layout: texel (kx, ky, c4) lanes hold the four channels
  // w[4*c4 + lane][ky][kx].  The {C, 1, KH, KW} weight carries the same flat
  // sequence as {C, KH, KW}, which the generic packing step interleaves into
  // channel groups.
  const int64_t C = weight.size(0);
  const int64_t KH = weight.size(2);
  const int64_t KW = weight.size(3);
  return weight.reshape({C, KH, KW}).contiguous();
}

Tensor pack_weight_transpose2d(const Tensor& weight) {
  // The packed texture uses x = input_channel * KW + kx and
  // y = output_channel_group * KH + ky.  Its four lanes carry one complete
  // output-channel group, keeping kernel coordinates contiguous for gathers.
  const int64_t C = weight.size(0);
  const int64_t O = weight.size(1);
  const int64_t KH = weight.size(2);
  const int64_t KW = weight.size(3);
  const int64_t O4 = api::utils::align_up(O, 4u) / 4;
  const int64_t CAligned = api::utils::align_up(C, 4u);
  const int64_t packed_width = CAligned * KW;
  const int64_t packed_height = O4 * KH;

  Tensor out(
      {1, 4, packed_height, packed_width},
      weight.dtype(),
      Device(DeviceType::CPU));
  float* dst = static_cast<float*>(out.impl()->storage().data());
  const float* src =
      static_cast<const float*>(weight.contiguous().impl()->storage().data());

  std::memset(dst, 0, out.numel() * sizeof(float));
  for (int64_t ci = 0; ci < C; ++ci) {
    for (int64_t o = 0; o < O; ++o) {
      const int64_t o4 = o / 4;
      const int64_t lane = o % 4;
      for (int64_t ky = 0; ky < KH; ++ky) {
        for (int64_t kx = 0; kx < KW; ++kx) {
          const int64_t dst_x = ci * KW + kx;
          const int64_t dst_y = o4 * KH + ky;
          const int64_t dst_index =
              ((dst_y * packed_width + dst_x) * 4) + lane;
          const int64_t src_index =
              ((ci * O + o) * KH + ky) * KW + kx;
          dst[dst_index] = src[src_index];
        }
      }
    }
  }
  return out;
}


//
// Materializes a parameter in its packed texture form, cached per source
// identity.  Host tensors stream through `cpu_pack`; device tensors run
// `device_pack` (a gather dispatch producing the requested layout; empty
// when the existing payload already matches, in which case it is reused in
// place).
//
api::vTensor upload_parameter_cached(
    const Tensor& param,
    const std::vector<int64_t>& logical_sizes,
    uint32_t tag,
    Tensor (*cpu_pack)(const Tensor&),
    const std::function<void(api::vTensor&)>& device_pack = {}) {
  return ParamTextureCache::singleton().get_or_create(
      param,
      logical_sizes,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      tag,
      [&]() -> api::vTensor {
        if (param.device().is_vulkan()) {
          if (!device_pack) {
            // The payload already carries the consumer's layout; reuse it
            // in place.
            return convert(param);
          }
          api::Context* const context = api::context();
          api::vTensor v{
              context,
              logical_sizes,
              param.dtype(),
              api::StorageType::TEXTURE_3D,
              api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
          };
          device_pack(v);
          return v;
        }

        api::Context* const context = api::context();
        api::vTensor v{
            context,
            logical_sizes,
            param.dtype(),
            api::StorageType::TEXTURE_3D,
            api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
        };

        // upload_host_bytes streams bytes texel-linearly into the texture,
        // so the packed blob is interleaved into NC4HW order first: blob
        // channel c of group element n lands in texel lane c % 4 of the
        // matching z slot, which is exactly the lane convention the packed
        // weight shaders rely on.
        Tensor packed_nc4hw = utils::nchw_to_nc4hw(cpu_pack(param));
        utils::upload_host_bytes(
            v,
            packed_nc4hw.impl()->storage().data(),
            packed_nc4hw.numel() * packed_nc4hw.itemsize());
        return v;
      });
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
    // Depthwise path: one filter plane per channel.  The packed weight
    // texture carries four channels per texel.
    api::vTensor v_weight = upload_parameter_cached(
        weight_arg,
        {C, KH, KW},
        kTagDepthwiseGroups,
        &pack_weight_dw,
        [&weight_arg, C, KH, KW](api::vTensor& v_dst) {
          api::Context* const pack_context = api::context();
          api::vTensor v_src = convert(weight_arg);
          api::PipelineBarrier pack_barrier{};

          const struct WeightPackBlock final {
            ivec4 weight_sizes; // (C, KH, KW, -)
          } pack_block{
              ivec4(
                  static_cast<int32_t>(C),
                  static_cast<int32_t>(KH),
                  static_cast<int32_t>(KW),
                  1),
          };

          api::UniformParamsBuffer pack_params(pack_context, pack_block);
          pack_context->submit_compute_job(
              VK_KERNEL_FROM_STR("pack_weight_depthwise"),
              pack_barrier,
              v_dst.extents(),
              adaptive_work_group_size(v_dst.extents()),
              VK_NULL_HANDLE,
              v_dst.image(
                  pack_barrier,
                  api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
              pack_params.buffer());
        });

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = upload_parameter_cached(
          *bias, shape_of(*bias), kTagIdentity, &pack_identity);
    }

    const struct Conv2DBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      ivec4 weight_sizes;
      ivec2 stride;
      ivec2 padding;
      ivec2 dilation;
      int c_depth;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(KW),
            static_cast<int32_t>(KH),
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
    };

    api::UniformParamsBuffer params(context, block);

    // A 3x3 output tile amortizes the weight fetches further when adjacent
    // outputs step one input element at a time; larger strides fetch more
    // out-of-block area, so they drop back to a 2x2 tile.
    const bool tile3 = stride[0] == 1 && stride[1] == 1;
    const char* shader_name = has_bias
        ? (tile3 ? "conv2d_dw_ot3" : "conv2d_dw_ot2")
        : (tile3 ? "conv2d_dw_nobias_ot3" : "conv2d_dw_nobias_ot2");

    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader_name),
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
    // Pointwise path: a per-position matrix product over the channel axis,
    // tiled 2x2 in the spatial domain with channel-group-packed weights.
    TP_CHECK(
        groups == 1, "Vulkan conv2d: pointwise path requires groups == 1");

    const int64_t O = weight_arg.size(0);
    const int64_t O4 = api::utils::align_up(O, 4u) / 4;
    const int64_t C4 = c_depth_of(v_input.sizes());

    api::vTensor v_weight = upload_parameter_cached(
        weight_arg,
        {4, 4, O4, C4},
        kTagPointwiseTiles,
        &pack_weight_pw,
        [&weight_arg, O, C, O4, C4](api::vTensor& v_dst) {
          api::Context* const pack_context = api::context();
          api::vTensor v_src = convert(weight_arg);
          api::PipelineBarrier pack_barrier{};

          const struct WeightPackBlock final {
            ivec4 weight_sizes; // (O, C, O4, C4)
          } pack_block{
              ivec4(
                  static_cast<int32_t>(O),
                  static_cast<int32_t>(C),
                  static_cast<int32_t>(O4),
                  static_cast<int32_t>(C4)),
          };

          api::UniformParamsBuffer pack_params(pack_context, pack_block);
          pack_context->submit_compute_job(
              VK_KERNEL_FROM_STR("pack_weight_pointwise"),
              pack_barrier,
              v_dst.extents(),
              adaptive_work_group_size(v_dst.extents()),
              VK_NULL_HANDLE,
              v_dst.image(
                  pack_barrier,
                  api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
              pack_params.buffer());
        });

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = upload_parameter_cached(
          *bias, shape_of(*bias), kTagIdentity, &pack_identity);
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
        C4,
        O4,
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
    // Regular grouped (single group) path with tap-packed weights.
    TP_CHECK(
        groups == 1, "Vulkan conv2d: only groups == 1 or depthwise supported");
    TP_CHECK(
        dilation[0] == 1 && dilation[1] == 1,
        "Vulkan conv2d: the regular path does not support dilation yet");

    const int64_t O = weight_arg.size(0);
    const int64_t O4 = api::utils::align_up(O, 4u) / 4;
    const int64_t C4 = c_depth_of(v_input.sizes());

    api::vTensor v_weight = upload_parameter_cached(
        weight_arg,
        {O4 * C4 * 4, 4, KH, KW},
        kTagWindowTaps,
        &pack_weight_sw,
        [&weight_arg, O, C, KH, KW, O4, C4](api::vTensor& v_dst) {
          api::Context* const pack_context = api::context();
          api::vTensor v_src = convert(weight_arg);
          api::PipelineBarrier pack_barrier{};

          const struct WeightPackBlock final {
            ivec4 weight_sizes; // (O, C, KH, KW)
            int in_c_depth;
            int out_c_depth;
            int src_c_depth;
          } pack_block{
              ivec4(
                  static_cast<int32_t>(O),
                  static_cast<int32_t>(C),
                  static_cast<int32_t>(KH),
                  static_cast<int32_t>(KW)),
              C4,
              O4,
              C4,
          };

          api::UniformParamsBuffer pack_params(pack_context, pack_block);
          pack_context->submit_compute_job(
              VK_KERNEL_FROM_STR("pack_weight_sliding_window"),
              pack_barrier,
              v_dst.extents(),
              adaptive_work_group_size(v_dst.extents()),
              VK_NULL_HANDLE,
              v_dst.image(
                  pack_barrier,
                  api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
              pack_params.buffer());
        });

    std::optional<api::vTensor> v_bias;
    if (has_bias) {
      v_bias = upload_parameter_cached(
          *bias, shape_of(*bias), kTagIdentity, &pack_identity);
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
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(v_output.sizes()),
        ivec4(
            static_cast<int32_t>(KW),
            static_cast<int32_t>(KH),
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
        C4,
        O4,
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

  const int64_t O4 = api::utils::align_up(O, 4u) / 4;
  const int64_t CAligned = api::utils::align_up(C, 4u);
  api::vTensor v_weight = upload_parameter_cached(
      weight_arg,
      {1, 4, O4 * KH, CAligned * KW},
      kTagTranspose2dPacked,
      &pack_weight_transpose2d,
      [&weight_arg, C, O, KH, KW, O4](api::vTensor& v_dst) {
        api::Context* const pack_context = api::context();
        api::vTensor v_src = convert(weight_arg);
        api::PipelineBarrier pack_barrier{};

        const struct WeightPackBlock final {
          ivec4 weight_sizes; // (C, O, KH, KW)
          int out_c_depth;
        } pack_block{
            ivec4(
                static_cast<int32_t>(C),
                static_cast<int32_t>(O),
                static_cast<int32_t>(KH),
                static_cast<int32_t>(KW)),
            static_cast<int32_t>(O4),
        };

        api::UniformParamsBuffer pack_params(pack_context, pack_block);
        pack_context->submit_compute_job(
            VK_KERNEL_FROM_STR("pack_weight_transpose2d"),
            pack_barrier,
            v_dst.extents(),
            adaptive_work_group_size(v_dst.extents()),
            VK_NULL_HANDLE,
            v_dst.image(
                pack_barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
            pack_params.buffer());
      });
  std::optional<api::vTensor> v_bias;
  if (has_bias) {
    v_bias = upload_parameter_cached(
          *bias, shape_of(*bias), kTagIdentity, &pack_identity);
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
      static_cast<int32_t>(O4),
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

  const int64_t OL =
      (L + 2 * padding[0] - dilation[0] * (K - 1) - 1) / stride[0] + 1;
  TP_CHECK(OL > 0, "Vulkan conv1d: computed output size is empty");

  api::vTensor v_output{context, {N, O, OL}, input_arg.dtype()};

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan conv1d requires texture storage");
  }

  // Width-packed weights: texel (k / 4, c, o) carries four adjacent taps of
  // one (o, c) filter in its lanes.  Keep the public source tensor as the
  // cache identity even when it starts on the host; the identity upload and
  // the width relayout are then both amortized across inference calls.
  api::vTensor v_weight;
  if (weight_arg.device().is_vulkan()) {
    Tensor weight = weight_arg.contiguous();
    v_weight = ParamTextureCache::singleton().get_or_create(
        weight,
        shape_of(weight),
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      kTagConv1dWidthPacked,
      [&weight]() {
        api::vTensor v_src = convert(weight);
        if (v_src.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
          return v_src;
        }
        return packing::convert_image_channels_packed_to_width_packed(v_src);
      });
  } else {
    v_weight = ParamTextureCache::singleton().get_or_create(
        weight_arg,
        shape_of(weight_arg),
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        kTagConv1dWidthPacked,
        [&weight_arg]() {
          api::vTensor v_uploaded = upload_parameter_cached(
              weight_arg,
              shape_of(weight_arg),
              kTagIdentity,
              &pack_identity);
          if (v_uploaded.gpu_memory_layout() ==
              api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
            return v_uploaded;
          }
          return packing::convert_image_channels_packed_to_width_packed(
              v_uploaded);
        });
  }
  api::vTensor v_bias;
  if (has_bias) {
    v_bias = upload_parameter_cached(
        *bias, shape_of(*bias), kTagIdentity, &pack_identity);
  } else {
    // The shader signature keeps a valid bias image for both variants.
    api::vTensor v_zero_bias{
        context, {O}, input_arg.dtype()};
    Tensor zero_bias = convert(v_zero_bias);
    zero_bias.fill_(Scalar(0.0));
    v_bias = std::move(v_zero_bias);
  }

  // The all-scalar block has the same field order as the shader.
  const struct Conv1DBlock final {
    int in_length;
    int kernel_size;
    int strides;
    int padding;
    int dilation;
    int in_group_size;
    int out_group_size;
    int batch_size;
  } block{
      static_cast<int32_t>(L),
      static_cast<int32_t>(K),
      static_cast<int32_t>(stride[0]),
      static_cast<int32_t>(padding[0]),
      static_cast<int32_t>(dilation[0]),
      static_cast<int32_t>(C),
      static_cast<int32_t>(O),
      static_cast<int32_t>(N),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // One invocation per output channel; the kernel rolls the length and
  // batch sweeps internally.
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(has_bias ? "conv1d" : "conv1d_nobias"),
      pipeline_barrier,
      api::utils::uvec3{1u, static_cast<uint32_t>(O), 1u},
      api::utils::uvec3{1u, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
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
