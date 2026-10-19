#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "ParamCache.h"
#include "Quantizer.h"
#include "Utils.h"

#include <Utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Cache tag for the height-packed Int8 weight payload of quantized_linear.
// Distinct from the mm packing tags in Matmul.cpp (10..15).
constexpr uint32_t kTagQLinearWeightHeightPacked = 20u;
constexpr uint32_t kTagQLinearParams = 21u;
constexpr uint32_t kTagW4A16WeightHeightPacked = 22u;
constexpr uint32_t kTagW4A16Params = 23u;
constexpr uint32_t kTagW4A16WeightLinear = 24u;
constexpr uint32_t kTagW4A16ParamsLinear = 25u;

//
// Quantized elementwise arithmetic over Int8 textures with explicit affine
// qparams.  Each operand byte is dequantized as (q - zero_point) * scale,
// the float operation is applied lane-wise, and the result is requantized
// with round-to-nearest-even into [-128, 127] under the output qparams --
// the same transformation the CPU kernels perform.
//
// Quantized max pooling runs directly in the quantized domain: the window
// maximum preserves the ordering that scale/zero_point induce, so the
// output bytes inherit the input qparams without requantization.
//

namespace {

void validate_quantize_input(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float64,
      std::string("Vulkan ") + name + " expects a float tensor");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

void validate_quantized_operand(const Tensor& t) {
  TP_CHECK(
      t.dtype() == DType::QInt8,
      "Vulkan quantized op(): operands must be QInt8");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      "Vulkan quantized op(): supports 1d to 4d tensors");
}

struct QuantizeBlock final {
  ivec4 extents;
  float inv_scale;
  int zero_point;
  int quant_min;
  int quant_max;
  float scale;
  int fill;
};

struct DequantizeBlock final {
  ivec4 extents;
  float scale;
  int zero_point;
  int fill0;
  int fill1;
  int fill2;
};

struct QuantizedBinaryBlock final {
  ivec4 out_sizes; // (W, H, C, N) sizes of the output
  ivec4 a_sizes;   // (W, H, C, N) sizes of operand A
  ivec4 b_sizes;   // (W, H, C, N) sizes of operand B
  float a_scale;
  int a_zero_point;
  float b_scale;
  int b_zero_point;
  float inv_out_scale;
  int out_zero_point;
  int a_c_depth;
  int b_c_depth;
  int out_c_depth;
};

struct QuantizedClampBlock final {
  ivec4 extents;
  float in_scale;
  int in_zero_point;
  float inv_out_scale;
  int out_zero_point;
  int has_min;
  int has_max;
  float min_value;
  float max_value;
};

struct QuantizedPool2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int c_depth;
};

struct QuantizedLinearBlock final {
  int out_m;
  int out_n;
  int k;
  float input_scale;
  int input_zero_point;
  float out_inv_scale;
  int out_zero_point;
  int quant_min;
  int quant_max;
};

struct QuantizedConv2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec4 weight_sizes;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth;
  int out_c_depth;
  int weight_c_depth;
  float in_scale;
  int in_zero_point;
  float weight_scale;
  int weight_zero_point;
  float inv_out_scale;
  int out_zero_point;
};

// Shader-side parameter block for the quantized convolution family; field
// order matches the uBlock declarations of the quantized_conv2d_* shaders.
struct QConvParams final {
  vec4 scales;
  ivec4 zero_points;
  ivec3 out_extents;
  int32_t fill0;
  ivec3 in_extents;
  int32_t fill1;
  ivec4 overlay_region;
  ivec2 kernel_size;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilate;
  vec2 clamp_thresh;
};

int64_t pool_output_len(int64_t in_len, int64_t k, int64_t s, int64_t p,
                        int64_t d, bool ceil_mode) {
  const int64_t num = in_len + 2 * p - d * (k - 1) - 1;
  if (ceil_mode) {
    return static_cast<int64_t>(std::ceil(
               static_cast<double>(num) / static_cast<double>(s))) +
        1;
  }
  return num / s + 1;
}

void check_quantized_binary_pair(
    const Tensor& a, const Tensor& b, std::vector<int64_t>& out_sizes) {
  validate_quantized_operand(a);
  validate_quantized_operand(b);
  out_sizes = broadcast_shapes(
      static_cast<std::vector<int64_t>>(a.shape()),
      static_cast<std::vector<int64_t>>(b.shape()));
}

} // namespace

Tensor quantize_per_tensor_qint8_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  validate_quantize_input(self, "quantize_per_tensor");
  TP_CHECK(scale > 0.0, "Vulkan quantize(): scale must be positive");
  TP_CHECK(
      quant_min < quant_max,
      "Vulkan quantize(): quant_min must be < quant_max");
  TP_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "Vulkan quantize(): zero_point out of the quantized range");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError, "Vulkan quantize requires texture storage");
  }

  // Downcast double inputs once on the host so the shader reads floats.
  const Tensor input_f32 =
      self.dtype() == DType::Float64 ? self.to(DType::Float32) : self;
  api::vTensor v_src =
      self.dtype() == DType::Float64 ? convert(input_f32) : v_input;

  // The output texture is allocated over the underlying integer code type;
  // the quantized dtype view with its quantizer is wrapped on below.
  api::vTensor v_output{
      context, v_src.sizes(), DType::Int8};

  const struct QuantizeBlock block{
      ivec4(
          v_src.extents()[0u],
          v_src.extents()[1u],
          v_src.extents()[2u],
          0),
      static_cast<float>(1.0 / scale),
      static_cast<int32_t>(zero_point),
      static_cast<int32_t>(quant_min),
      static_cast<int32_t>(quant_max),
      static_cast<float>(scale),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantize_per_tensor), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt8),
      DType::QInt8);
}

Tensor dequantize_per_tensor_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  TP_CHECK(
      self.dtype() == DType::QInt8,
      "Vulkan dequantize(): expected a QInt8 tensor");
  TP_CHECK(scale > 0.0, "Vulkan dequantize(): scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  // A code payload in linear storage keeps four lanes to a 32-bit word, so the
  // affine undo reads and writes whole words instead of sampling texels.  The
  // result stays in the storage its payload drew: a weight whose row count has
  // no image form is converted where it already sits.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    api::vTensor v_output{
        context,
        v_input.sizes(),
        DType::Float32,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

    const struct DequantizeLinearBlock final {
      float scale;
      int32_t zero_point;
      uint32_t buf_length; // element count, shared by both payloads
      int32_t fill;
    } block{
        static_cast<float>(scale),
        static_cast<int32_t>(zero_point),
        static_cast<uint32_t>(v_output.numel()),
        0};

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    // One invocation owns one word of four lanes.
    const api::utils::uvec3 grid{
        api::utils::div_up(
            api::utils::safe_downcast_to_u32(
                static_cast<int64_t>(v_output.numel())),
            4u),
        1u,
        1u};

    context->submit_compute_job(
        VK_KERNEL_FROM_STR("dequantize_per_tensor_buffer"),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    return convert(v_output);
  }

  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError, "Vulkan dequantize requires texture storage");
  }

  api::vTensor v_output{
      context, v_input.sizes(), DType::Float32};

  const struct DequantizeBlock block{
      ivec4(
          v_input.extents()[0u],
          v_input.extents()[1u],
          v_input.extents()[2u],
          0),
      static_cast<float>(scale),
      static_cast<int32_t>(zero_point),
      0,
      0,
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(dequantize_per_tensor), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

/*
 * The block grid a block_size names over a logical shape: one parameter per
 * block, numbered in the same row-major order as the elements it covers.  An
 * entry that reaches the end of its axis collapses that axis to a single block,
 * so a block_size of the whole shape is the per-tensor case and an entry of one
 * makes every element its own block.
 *
 * Axes are held outermost first and padded to four with ones, which is how the
 * shaders walk them: a padded axis always names coordinate zero.
 */
struct AffineGrid final {
  api::utils::uvec4 sizes;
  api::utils::uvec4 size_strides;
  api::utils::uvec4 blocks;
  api::utils::uvec4 block_strides;
  uint32_t num_blocks;
};

AffineGrid affine_grid(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& block_size) {
  const int rank = static_cast<int>(sizes.size());
  const int offset = 4 - rank;
  uint32_t extents[4] = {1u, 1u, 1u, 1u};
  uint32_t spans[4] = {1u, 1u, 1u, 1u};
  for (int axis = 0; axis < rank; ++axis) {
    const int padded = offset + axis;
    extents[padded] = api::utils::safe_downcast_to_u32(sizes[axis]);
    spans[padded] = api::utils::safe_downcast_to_u32(
        std::min(block_size[axis], sizes[axis]));
  }

  uint32_t strides[4] = {1u, 1u, 1u, 1u};
  for (int axis = 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * extents[axis + 1];
  }
  uint32_t block_strides[4] = {1u, 1u, 1u, 1u};
  uint32_t blocks_after = 1u;
  for (int axis = 3; axis >= 0; --axis) {
    block_strides[axis] = blocks_after;
    blocks_after *= api::utils::div_up(extents[axis], spans[axis]);
  }

  AffineGrid grid;
  grid.sizes = api::utils::uvec4(extents[0], extents[1], extents[2], extents[3]);
  grid.size_strides =
      api::utils::uvec4(strides[0], strides[1], strides[2], strides[3]);
  grid.blocks = api::utils::uvec4(spans[0], spans[1], spans[2], spans[3]);
  grid.block_strides = api::utils::uvec4(
      block_strides[0], block_strides[1], block_strides[2], block_strides[3]);
  grid.num_blocks = blocks_after;
  return grid;
}

} // namespace

//
// Block-wise (group-wise) affine dequantization: the payload keeps one affine
// parameter pair per block of the grid a block_size names, so a quantized
// weight recovers its values without the parameters crossing back over the host.
// Both the parameters and the codes are read where they already sit, and the
// result is written in the storage the code payload drew -- a weight too long
// for an image is converted in linear storage, one that fits stays in texels.
//
Tensor dequantize_affine_kernel(
    const Tensor& self,
    const std::vector<int64_t>& block_size,
    const Tensor& scales,
    const std::optional<Tensor>& zero_points) {
  TP_CHECK(
      self.dtype() == DType::QInt8 || self.dtype() == DType::QUInt8,
      "Vulkan dequantize_affine(): expected a QInt8 or QUInt8 payload");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan dequantize_affine(): supports 1d to 4d payloads");
  TP_CHECK(
      static_cast<int64_t>(block_size.size()) == self.dim(),
      "Vulkan dequantize_affine(): block_size rank must match the input rank");
  for (const auto entry : block_size) {
    TP_CHECK(
        entry > 0,
        "Vulkan dequantize_affine(): block_size entries must be positive");
  }

  // An owning copy: the block grid reads the shape twice, and shape() hands
  // back a fresh value object each call.
  const std::vector<int64_t> sizes = self.shape();
  const AffineGrid affine = affine_grid(sizes, block_size);
  TP_CHECK(
      scales.numel() == static_cast<int64_t>(affine.num_blocks) ||
          scales.numel() == 1,
      "Vulkan dequantize_affine(): scales must have one entry per block");
  TP_CHECK(
      scales.device() == self.device(),
      "Vulkan dequantize_affine(): scales and zero_points must be on the input "
      "device");
  const bool has_zeros =
      zero_points.has_value() && zero_points->numel() > 0;
  if (has_zeros) {
    TP_CHECK(
        zero_points->numel() == static_cast<int64_t>(affine.num_blocks) ||
            zero_points->numel() == 1,
        "Vulkan dequantize_affine(): zero_points must have one entry per block, "
        "or be a scalar");
    TP_CHECK(
        zero_points->device() == self.device(),
        "Vulkan dequantize_affine(): scales and zero_points must be on the "
        "input device");
  }

  api::Context* const context = api::context();
  const Tensor codes = self.is_contiguous() ? self : self.contiguous();
  if (codes.numel() == 0) {
    return empty_like_kernel(codes, DType::Float32, std::nullopt);
  }

  // Both parameter payloads are named: the descriptor binds the storage they
  // own, so a cast that lived only inside the call expression would release its
  // buffer back to the pool while the kernel still reads it.
  const Tensor scale_values = scales.to(DType::Float32).contiguous();
  const Tensor zero_values = has_zeros
      ? zero_points->to(DType::Float32).contiguous()
      : scale_values;
  api::vTensor v_scales =
      to_buffer_storage_dispatch(context, convert(scale_values));
  // A call without addends reads nothing from the second parameter binding, but
  // the descriptor still needs memory behind it.
  const api::vTensor v_zeros =
      to_buffer_storage_dispatch(context, convert(zero_values));

  const uint32_t element_count =
      api::utils::safe_downcast_to_u32(codes.numel());
  const uint32_t scale_count =
      api::utils::safe_downcast_to_u32(scale_values.numel());
  const uint32_t zero_count =
      has_zeros
      ? api::utils::safe_downcast_to_u32(zero_points->numel())
      : 0u;
  // One byte of the payload names a signed code for QInt8 and an unsigned one
  // for QUInt8; both share the eight bits the shaders fetch.
  const uint32_t unsigned_payload = self.dtype() == DType::QUInt8 ? 1u : 0u;

  api::vTensor v_codes = convert(codes);

  if (v_codes.storage_type() == api::StorageType::BUFFER) {
    api::vTensor v_output{
        context,
        v_codes.sizes(),
        DType::Float32,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

    const struct DequantizeAffineLinearBlock final {
      api::utils::uvec4 sizes;
      api::utils::uvec4 size_strides;
      api::utils::uvec4 blocks;
      api::utils::uvec4 block_strides;
      api::utils::uvec4 grid; // unused; a linear run has no texel extents
      api::utils::uvec4 flags;
    } block{
        affine.sizes,
        affine.size_strides,
        affine.blocks,
        affine.block_strides,
        api::utils::uvec4(0u, 0u, 0u, 1u),
        api::utils::uvec4(
            element_count,
            unsigned_payload,
            scale_count,
            zero_count)};

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 grid{api::utils::div_up(element_count, 4u), 1u, 1u};

    context->submit_compute_job(
        VK_KERNEL_FROM_STR("dequantize_affine_buffer"),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_codes.buffer(
            pipeline_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_scales.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_zeros.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    return convert(v_output);
  }

  if (v_codes.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan dequantize_affine(): requires texture or buffer storage");
  }

  api::vTensor v_output{context, v_codes.sizes(), DType::Float32};
  // A texel's four lanes are four channels, so the depth of the channel groups
  // splits the image's z axis back into a batch and a group within it.
  const uint32_t channels = api::utils::safe_downcast_to_u32(
      self.dim() >= 3 ? self.size(self.dim() - 3) : 1);
  const api::utils::uvec3 extents = v_output.extents();

  const struct DequantizeAffineBlock final {
    api::utils::uvec4 sizes;
    api::utils::uvec4 size_strides;
    api::utils::uvec4 blocks;
    api::utils::uvec4 block_strides;
    api::utils::uvec4 grid; // (texel x, texel y, texel z, channel groups)
    api::utils::uvec4 flags;
  } block{
      affine.sizes,
      affine.size_strides,
      affine.blocks,
      affine.block_strides,
      api::utils::uvec4(
          extents[0u],
          extents[1u],
          extents[2u],
          api::utils::div_up(channels, 4u)),
      api::utils::uvec4(
          element_count,
          unsigned_payload,
          scale_count,
          zero_count)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR("dequantize_affine"),
      pipeline_barrier,
      extents,
      adaptive_work_group_size(extents),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_codes.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_scales.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_zeros.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

namespace {

// Uploads a convolution weight/bias CPU tensor into channel-packed texture
// storage; device tensors reuse their existing payload.  Host bytes live in
// logical order while the texture expects the channel-packed layout, so the
// upload goes through the packing step instead of a raw byte copy.
api::vTensor upload_conv_param(
    const Tensor& param,
    const api::GPUMemoryLayout layout) {
  api::Context* const context = api::context();

  if (param.device().is_vulkan()) {
    return convert(param);
  }

  api::vTensor v{
      context,
      static_cast<std::vector<int64_t>>(param.shape()),
      param.dtype(),
      api::StorageType::TEXTURE_3D,
      layout};

  Tensor packed = utils::nchw_to_nc4hw(param.contiguous());
  utils::upload_host_bytes(
      v, packed.impl()->storage().data(), packed.numel() * packed.itemsize());
  return v;
}

Tensor quantized_binary_kernel_impl(
    const Tensor& a,
    const Tensor& b,
    double a_scale,
    int64_t a_zero_point,
    double b_scale,
    int64_t b_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const char* kernel_name) {
  std::vector<int64_t> result_sizes;
  check_quantized_binary_pair(a, b, result_sizes);
  TP_CHECK(out_scale > 0.0, "Vulkan quantized op(): out_scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_a = convert(a);
  api::vTensor v_b = convert(b);
  if (v_a.storage_type() != api::StorageType::TEXTURE_3D ||
      v_b.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized ops require texture storage");
  }

  api::vTensor v_output{context, result_sizes, DType::Int8};

  const std::vector<int64_t> a_whcn{
      get_dim<Dim4D::Width>(v_a.sizes()),
      get_dim<Dim4D::Height>(v_a.sizes()),
      get_dim<Dim4D::Channel>(v_a.sizes()),
      get_dim<Dim4D::Batch>(v_a.sizes())};
  const std::vector<int64_t> b_whcn{
      get_dim<Dim4D::Width>(v_b.sizes()),
      get_dim<Dim4D::Height>(v_b.sizes()),
      get_dim<Dim4D::Channel>(v_b.sizes()),
      get_dim<Dim4D::Batch>(v_b.sizes())};
  const std::vector<int64_t> out_whcn{
      get_dim<Dim4D::Width>(v_output.sizes()),
      get_dim<Dim4D::Height>(v_output.sizes()),
      get_dim<Dim4D::Channel>(v_output.sizes()),
      get_dim<Dim4D::Batch>(v_output.sizes())};

  const struct QuantizedBinaryBlock block{
      ivec4(
          static_cast<int32_t>(out_whcn[0]),
          static_cast<int32_t>(out_whcn[1]),
          static_cast<int32_t>(out_whcn[2]),
          static_cast<int32_t>(out_whcn[3])),
      ivec4(
          static_cast<int32_t>(a_whcn[0]),
          static_cast<int32_t>(a_whcn[1]),
          static_cast<int32_t>(a_whcn[2]),
          static_cast<int32_t>(a_whcn[3])),
      ivec4(
          static_cast<int32_t>(b_whcn[0]),
          static_cast<int32_t>(b_whcn[1]),
          static_cast<int32_t>(b_whcn[2]),
          static_cast<int32_t>(b_whcn[3])),
      static_cast<float>(a_scale),
      static_cast<int32_t>(a_zero_point),
      static_cast<float>(b_scale),
      static_cast<int32_t>(b_zero_point),
      static_cast<float>(1.0 / out_scale),
      static_cast<int32_t>(out_zero_point),
      c_depth_of(v_a.sizes()),
      c_depth_of(v_b.sizes()),
      c_depth_of(v_output.sizes())};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_a.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_b.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                       DType::QInt8),
      DType::QInt8);
}

} // namespace

Tensor quantized_add_kernel(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
  return quantized_binary_kernel_impl(
      a, b, a_scale, a_zero_point, b_scale, b_zero_point, out_scale,
      out_zero_point, "quantized_add");
}

Tensor quantized_sub_kernel(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
  return quantized_binary_kernel_impl(
      a, b, a_scale, a_zero_point, b_scale, b_zero_point, out_scale,
      out_zero_point, "quantized_sub");
}

Tensor quantized_mul_kernel(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
  return quantized_binary_kernel_impl(
      a, b, a_scale, a_zero_point, b_scale, b_zero_point, out_scale,
      out_zero_point, "quantized_mul");
}

Tensor quantized_div_kernel(
    const Tensor& a, const Tensor& b,
    double a_scale, int64_t a_zero_point,
    double b_scale, int64_t b_zero_point,
    double out_scale, int64_t out_zero_point) {
  return quantized_binary_kernel_impl(
      a, b, a_scale, a_zero_point, b_scale, b_zero_point, out_scale,
      out_zero_point, "quantized_div");
}

Tensor quantized_clamp_kernel(
    const Tensor& self,
    double self_scale,
    int64_t self_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  validate_quantized_operand(self);
  TP_CHECK(
      out_scale > 0.0, "Vulkan quantized_clamp(): out_scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized clamp requires texture storage");
  }

  api::vTensor v_output{context, v_input.sizes(), DType::Int8};

  const struct QuantizedClampBlock block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      static_cast<float>(self_scale),
      static_cast<int32_t>(self_zero_point),
      static_cast<float>(1.0 / out_scale),
      static_cast<int32_t>(out_zero_point),
      min.has_value() ? 1 : 0,
      max.has_value() ? 1 : 0,
      min.has_value() ? static_cast<float>(min->toDouble()) : 0.0f,
      max.has_value() ? static_cast<float>(max->toDouble()) : 0.0f};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantized_clamp), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                       DType::QInt8),
      DType::QInt8);
}

Tensor quantized_max_pool2d_kernel(
    const Tensor& self,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    bool ceil_mode) {
  validate_quantized_operand(self);
  TP_CHECK(
      self.dim() == 4 || self.dim() == 3,
      "Vulkan quantized_max_pool2d: expects a 3d or 4d tensor");
  const bool squeezed = self.dim() == 3;
  const Tensor input = squeezed ? self.unsqueeze(0) : self;

  TP_CHECK(
      kernel_size.size() == 2,
      "Vulkan quantized_max_pool2d: expects a 2-element kernel size");
  const auto expand2 = [](const std::vector<int64_t>& v) {
    return v.size() == 1 ? std::vector<int64_t>{v[0], v[0]}
                         : std::vector<int64_t>{v[0], v[1]};
  };
  const std::vector<int64_t> stride2 =
      expand2(stride.empty() ? kernel_size : stride);
  const std::vector<int64_t> padding2 = expand2(padding);
  const std::vector<int64_t> dilation2 = expand2(dilation);

  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size[1];
  const int64_t sH = stride2[0];
  const int64_t sW = stride2[1];
  const int64_t pH = padding2[0];
  const int64_t pW = padding2[1];
  const int64_t dH = dilation2[0];
  const int64_t dW = dilation2[1];
  TP_CHECK(sH > 0 && sW > 0, "Vulkan quantized_max_pool2d: stride must be positive");

  const int64_t H_in = input.size(2);
  const int64_t W_in = input.size(3);
  const int64_t H_out = pool_output_len(H_in, kH, sH, pH, dH, ceil_mode);
  const int64_t W_out = pool_output_len(W_in, kW, sW, pW, dW, ceil_mode);
  TP_CHECK(
      H_out > 0 && W_out > 0,
      "Vulkan quantized_max_pool2d: output size must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized max pooling requires texture storage");
  }

  // Window maximum preserves the code ordering, so the output inherits the
  // input's quantizer.
  quantized::require_quantized(input, "quantized_max_pool2d");
  api::vTensor v_output{
      context,
      {input.size(0), input.size(1), H_out, W_out},
      DType::Int8};

  const struct QuantizedPool2DBlock block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec2(static_cast<int32_t>(kW), static_cast<int32_t>(kH)),
      ivec2(static_cast<int32_t>(sW), static_cast<int32_t>(sH)),
      ivec2(static_cast<int32_t>(pW), static_cast<int32_t>(pH)),
      ivec2(static_cast<int32_t>(dW), static_cast<int32_t>(dH)),
      c_depth_of(v_input.sizes())};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantized_max_pool2d), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  Tensor out = quantized::make_qtensor(
      out_codes, quantized::quantizer_of(input), DType::QInt8);
  return squeezed ? out.squeeze(0) : out;
}

Tensor quantized_linear_kernel(
    const Tensor& input,
    const Tensor& weight,
    double input_scale,
    int64_t input_zero_point,
    const Tensor& weight_scales,
    const Tensor& weight_zero_points,
    const std::optional<Tensor>& bias,
    double out_scale,
    int64_t out_zero_point) {
  TP_CHECK(
      input.dtype() == DType::QInt8 && weight.dtype() == DType::QInt8,
      "Vulkan quantized_linear: activations and weights must be QInt8");
  TP_CHECK(
      out_scale > 0.0, "Vulkan quantized_linear: out_scale must be positive");
  TP_CHECK(
      input.dim() == 2 && weight.dim() == 2,
      "Vulkan quantized_linear: expected 2-D [M,K] activations and [N,K] "
      "weights");
  TP_CHECK(
      input.size(1) == weight.size(1),
      "Vulkan quantized_linear: incompatible K dimensions");
  TP_CHECK(
      input_scale > 0.0, "Vulkan quantized_linear: scale must be positive");
  const int64_t out_features = weight.size(0);
  TP_CHECK(
      weight_scales.dim() == 1 && weight_scales.size(0) == out_features &&
          weight_zero_points.shape() == weight_scales.shape(),
      "Vulkan quantized_linear: weight scales/zero_points must be 1-D of "
      "length out_features");
  if (bias.has_value()) {
    TP_CHECK(
        bias->dtype() == DType::Float32 && bias->dim() == 1 &&
            bias->size(0) == out_features,
        "Vulkan quantized_linear: bias must be a 1-D Float32 tensor of "
        "length out_features");
  }

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);
  api::vTensor v_weight = convert(weight);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D ||
      v_weight.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized_linear requires texture storage");
  }

  // Pack the per-channel parameters into a 3-row float texture indexed by
  // the output channel: scale (row 0), zero point (row 1), bias (row 2).
  // The payload is cached on the weight identity so steady-state inference
  // never rounds-trips the per-channel qparams through the host.  Rows are
  // padded to a multiple of four so the width-packed params texture never
  // straddles a row boundary.
  const int64_t params_width = api::utils::align_up(out_features, 4);
  api::vTensor v_params = ParamTextureCache::singleton().get_or_create(
      weight, {3, params_width}, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      kTagQLinearParams,
      [&]() {
        Tensor params_cpu = Tensor::zeros(
            {3, params_width}, DType::Float32, Device(DeviceType::CPU));
        Tensor sc = weight_scales.to(Device(DeviceType::CPU))
                        .to(DType::Float32)
                        .contiguous();
        Tensor zp = weight_zero_points.to(Device(DeviceType::CPU))
                        .to(DType::Float32)
                        .contiguous();
        std::memcpy(
            params_cpu.impl()->storage().data() + sizeof(float) * params_width,
            zp.data_ptr<float>(),
            sizeof(float) * out_features);
        std::memcpy(
            params_cpu.impl()->storage().data(),
            sc.data_ptr<float>(),
            sizeof(float) * out_features);
        if (bias.has_value()) {
          Tensor bias_f = bias->to(Device(DeviceType::CPU))
                              .to(DType::Float32)
                              .contiguous();
          std::memcpy(
              params_cpu.impl()->storage().data() +
                  sizeof(float) * 2 * params_width,
              bias_f.data_ptr<float>(),
              sizeof(float) * out_features);
        }
        api::vTensor v_p{
            context,
            {3, params_width},
            DType::Float32,
            api::StorageType::TEXTURE_3D,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
        // The params texture is width-packed: texel (n/4, row) carries four
        // consecutive output channels of row `row`.  A row-major upload of
        // the padded [3, params_width] host buffer lands exactly there.
        utils::upload_host_bytes(
            v_p,
            params_cpu.contiguous().impl()->storage().data(),
            params_cpu.numel() * params_cpu.itemsize());
        return v_p;
      });

  const int64_t M = input.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = input.size(1);
  const int64_t K_tex = (K + 3) / 4;
  api::vTensor v_output{context, {M, N}, DType::Int8};

  // Pack activations and weights into the packed Int8 layouts the tiled
  // GEMM streams: width-packed activations (texel (k/4, m) holds four K of
  // row m) and height-packed weights (texel (n, k/4) holds four K of
  // column n).
  api::vTensor v_input_packed{
      context,
      {M, K},
      DType::QInt8,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const struct QuantizedLinearBlock block{
      static_cast<int32_t>(M),
      static_cast<int32_t>(N),
      static_cast<int32_t>(K),
      static_cast<float>(input_scale),
      static_cast<int32_t>(input_zero_point),
      static_cast<float>(1.0 / out_scale),
      static_cast<int32_t>(out_zero_point),
      -128,
      127};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const struct PackBlockWidth final {
    int k;
    int m;
  } pack_in_block{
      static_cast<int32_t>(K),
      static_cast<int32_t>(M)};
  const struct PackBlockHeight final {
    int k;
    int n;
  } pack_w_block{
      static_cast<int32_t>(K),
      static_cast<int32_t>(N)};
  api::UniformParamsBuffer pack_in_params(context, pack_in_block);
  api::UniformParamsBuffer pack_w_params(context, pack_w_block);

  const api::utils::uvec3 pack_in_grid{
      api::utils::safe_downcast_to_u32(K_tex),
      api::utils::safe_downcast_to_u32(M),
      1u};
  const api::utils::uvec3 pack_w_grid{
      api::utils::safe_downcast_to_u32(N),
      api::utils::safe_downcast_to_u32(K_tex),
      1u};

  context->submit_compute_job(
      VK_KERNEL(pack_int8_width), pipeline_barrier, pack_in_grid,
      adaptive_work_group_size(pack_in_grid), VK_NULL_HANDLE,
      v_input_packed.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      pack_in_params.buffer());
  // The packed weight payload is materialized once per source storage
  // identity and reused across calls (weight prepack for inference).
  api::vTensor v_weight_packed = ParamTextureCache::singleton().get_or_create(
      weight, {K, N}, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      kTagQLinearWeightHeightPacked,
      [&]() {
        api::vTensor v_wp{
            context,
            {K, N},
            DType::QInt8,
            api::StorageType::TEXTURE_3D,
            api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED};
        context->submit_compute_job(
            VK_KERNEL(pack_int8_height), pipeline_barrier, pack_w_grid,
            adaptive_work_group_size(pack_w_grid), VK_NULL_HANDLE,
            v_wp.image(
                pipeline_barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
            pack_w_params.buffer());
        return v_wp;
      });

  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32((N + 3) / 4),
      api::utils::safe_downcast_to_u32((M + 3) / 4),
      1u};

  context->submit_compute_job(
      VK_KERNEL(quantized_linear_tiled), pipeline_barrier, grid,
      {8u, 8u, 1u}, VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_params.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  // The int32 accumulation is requantized onto the output grid inside the
  // GEMM shader, so the returned tensor is already QInt8.
  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(out_scale, out_zero_point, DType::QInt8),
      DType::QInt8);
}

Tensor quantized_conv2d_kernel(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    double input_scale,
    int64_t input_zero_point,
    double weight_scale,
    int64_t weight_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  TP_CHECK(
      input.dtype() == DType::QInt8 && weight.dtype() == DType::QInt8,
      "Vulkan quantized_conv2d: activations and weights must be QInt8");
  TP_CHECK(input.dim() == 4, "Vulkan quantized_conv2d requires a 4d input");
  TP_CHECK(weight.dim() == 4, "Vulkan quantized_conv2d requires a 4d weight");
  const auto expand2 = [](const std::vector<int64_t>& v, int64_t def) {
    return v.size() == 1 ? std::vector<int64_t>{v[0], v[0]}
                         : std::vector<int64_t>{v[0], v[1]};
  };
  const std::vector<int64_t> stride2 = expand2(stride, 1);
  const std::vector<int64_t> padding2 = expand2(padding, 0);
  const std::vector<int64_t> dilation2 = expand2(dilation, 1);
  TP_CHECK(
      stride2[0] > 0 && stride2[1] > 0 && dilation2[0] > 0 && dilation2[1] > 0,
      "Vulkan quantized_conv2d: stride and dilation must be positive");
  TP_CHECK(
      groups == 1, "Vulkan quantized_conv2d only supports groups == 1");
  TP_CHECK(
      out_scale > 0.0, "Vulkan quantized_conv2d: out_scale must be positive");
  const bool has_bias = bias.has_value() && bias->defined();
  if (has_bias) {
    TP_CHECK(
        bias->dtype() == DType::Float32,
        "Vulkan quantized_conv2d: bias must be Float32");
  }

  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized_conv2d requires texture storage");
  }

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  const int64_t O = weight.size(0);
  const int64_t KH = weight.size(2);
  const int64_t KW = weight.size(3);
  TP_CHECK(
      weight.size(1) == C,
      "Vulkan quantized_conv2d: weight channel count must match the input");

  const int64_t OH =
      (H + 2 * padding2[1] - dilation2[1] * (KH - 1) - 1) / stride2[1] + 1;
  const int64_t OW =
      (W + 2 * padding2[0] - dilation2[0] * (KW - 1) - 1) / stride2[0] + 1;
  TP_CHECK(
      OH > 0 && OW > 0,
      "Vulkan quantized_conv2d: computed output size is empty");

  api::vTensor v_output{context, {N, O, OH, OW}, DType::Int8};

  api::vTensor v_weight = upload_conv_param(
      weight, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  std::optional<api::vTensor> v_bias;
  if (has_bias) {
    v_bias = upload_conv_param(
        *bias, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  }

  const struct QuantizedConv2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    ivec4 weight_sizes;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;
    int in_c_depth;
    int out_c_depth;
    int weight_c_depth;
    float in_scale;
    int in_zero_point;
    float weight_scale;
    int weight_zero_point;
    float inv_out_scale;
    int out_zero_point;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec4(
          static_cast<int32_t>(O),
          static_cast<int32_t>(C),
          static_cast<int32_t>(KH),
          static_cast<int32_t>(KW)),
      ivec2(
          static_cast<int32_t>(stride2[0]),
          static_cast<int32_t>(stride2[1])),
      ivec2(
          static_cast<int32_t>(padding2[0]),
          static_cast<int32_t>(padding2[1])),
      ivec2(
          static_cast<int32_t>(dilation2[0]),
          static_cast<int32_t>(dilation2[1])),
      c_depth_of(v_input.sizes()),
      c_depth_of(v_output.sizes()),
      c_depth_of(v_input.sizes()),
      static_cast<float>(input_scale),
      static_cast<int32_t>(input_zero_point),
      static_cast<float>(weight_scale),
      static_cast<int32_t>(weight_zero_point),
      static_cast<float>(1.0 / out_scale),
      static_cast<int32_t>(out_zero_point),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(has_bias ? "quantized_conv2d" : "quantized_conv2d_nobias"),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      has_bias
          ? v_bias->image(pipeline_barrier, api::PipelineStage::COMPUTE)
          : v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                       DType::QInt8),
      DType::QInt8);
}

Tensor quantize_per_tensor_quint8_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  validate_quantize_input(self, "quantize_per_tensor_quint8");
  TP_CHECK(scale > 0.0, "Vulkan quantize(): scale must be positive");
  TP_CHECK(
      quant_min < quant_max,
      "Vulkan quantize(): quant_min must be < quant_max");
  TP_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "Vulkan quantize(): zero_point out of the quantized range");

  api::Context* const context = api::context();

  api::vTensor v_src = convert(self);

  // Codes land in a UInt8 texture; the QUInt8 view wraps on below.
  api::vTensor v_output{
      context, v_src.sizes(), DType::UInt8};

  const struct QuantizeBlock block{
      ivec4(
          v_src.extents()[0u],
          v_src.extents()[1u],
          v_src.extents()[2u],
          0),
      static_cast<float>(1.0 / scale),
      static_cast<int32_t>(zero_point),
      static_cast<int32_t>(quant_min),
      static_cast<int32_t>(quant_max),
      static_cast<float>(scale),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantize_per_tensor_quint8), pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(scale, zero_point, DType::QUInt8),
      DType::QUInt8);
}

Tensor dequantize_per_tensor_quint8_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  TP_CHECK(
      self.dtype() == DType::QUInt8,
      "Vulkan dequantize(): expected a QUInt8 tensor");
  TP_CHECK(scale > 0.0, "Vulkan dequantize(): scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  api::vTensor v_output{
      context, v_input.sizes(), DType::Float32};

  const struct DequantizeBlock block{
      ivec4(
          v_input.extents()[0u],
          v_input.extents()[1u],
          v_input.extents()[2u],
          0),
      static_cast<float>(scale),
      static_cast<int32_t>(zero_point),
      0,
      0,
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(dequantize_per_tensor_quint8), pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor quantize_per_tensor_qint32_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  validate_quantize_input(self, "quantize_per_tensor_qint32");
  TP_CHECK(scale > 0.0, "Vulkan quantize(): scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_src = convert(self);

  // Codes land in an Int32 texture; the QInt32 view wraps on below.
  api::vTensor v_output{
      context, v_src.sizes(), DType::Int32};

  const struct QuantizeBlock block{
      ivec4(
          v_src.extents()[0u],
          v_src.extents()[1u],
          v_src.extents()[2u],
          0),
      static_cast<float>(1.0 / scale),
      static_cast<int32_t>(zero_point),
      -2147483647 - 1,
      2147483647,
      static_cast<float>(scale),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantize_per_tensor_qint32), pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(scale, zero_point, DType::QInt32),
      DType::QInt32);
}

Tensor dequantize_per_tensor_qint32_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  TP_CHECK(
      self.dtype() == DType::QInt32,
      "Vulkan dequantize(): expected a QInt32 tensor");
  TP_CHECK(scale > 0.0, "Vulkan dequantize(): scale must be positive");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  api::vTensor v_output{
      context, v_input.sizes(), DType::Float32};

  const struct DequantizeBlock block{
      ivec4(
          v_input.extents()[0u],
          v_input.extents()[1u],
          v_input.extents()[2u],
          0),
      static_cast<float>(scale),
      static_cast<int32_t>(zero_point),
      0,
      0,
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(dequantize_per_tensor_qint32), pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor quantize_per_tensor_kernel(const Tensor& self, double scale,
                                  int64_t zero_point, DType dtype) {
  switch (dtype) {
    case DType::QInt8:
      return quantize_per_tensor_qint8_kernel(
          self, scale, zero_point, -128, 127);
    case DType::QUInt8:
      return quantize_per_tensor_quint8_kernel(
          self, scale, zero_point, 0, 255);
    case DType::QInt32:
      return quantize_per_tensor_qint32_kernel(self, scale, zero_point);
    default:
      TP_THROW(TypeError, "quantize_per_tensor(): unsupported quantized dtype");
  }
}

Tensor dequantize_per_tensor_kernel(const Tensor& self, double scale,
                                    int64_t zero_point, DType dtype) {
  switch (dtype) {
    case DType::QInt8:
      return dequantize_per_tensor_kernel(self, scale, zero_point);
    case DType::QUInt8:
      return dequantize_per_tensor_quint8_kernel(self, scale, zero_point);
    case DType::QInt32:
      return dequantize_per_tensor_qint32_kernel(self, scale, zero_point);
    default:
      TP_THROW(TypeError, "dequantize(): unsupported quantized dtype");
  }
}

namespace {

inline int64_t conv_align_up_4(int64_t v) {
  return (v + 3) / 4 * 4;
}

// Picks the quantized convolution compute path following the family rules:
// transposed kernels use the gather transposed shader, a depthwise weight
// (groups == out-channels, one input channel) uses the depthwise shader, a
// 1x1 kernel uses the 2x2-tiled pointwise shader, and everything else runs
// through the sliding-window shader.
enum class QConvMethod { kSlidingWindow, kDepthwise, kPointwise };

QConvMethod determine_qconv_method(
    const std::vector<int64_t>& weight_sizes,
    int64_t groups,
    bool transposed) {
  if (transposed) {
    return QConvMethod::kSlidingWindow;
  }
  if (weight_sizes[0] == groups && weight_sizes[1] == 1) {
    return QConvMethod::kDepthwise;
  }
  if (weight_sizes[2] == 1 && weight_sizes[3] == 1) {
    return QConvMethod::kPointwise;
  }
  return QConvMethod::kSlidingWindow;
}

// Size of the overlay region of the kernel: the spatial reach after the
// dilation, plus the 4-aligned channel roles.  For the transposed kernel the
// channel roles of the weight size list stay in [in, out] order.
std::vector<int64_t> compute_qconv_overlay_region(
    const std::vector<int64_t>& weight_sizes,
    const std::vector<int64_t>& dilation,
    bool transposed) {
  const auto overlay_length = [](int64_t k, int64_t d) {
    return k + (k - 1) * (d - 1);
  };
  return {
      conv_align_up_4(transposed ? weight_sizes[1] : weight_sizes[0]),
      conv_align_up_4(transposed ? weight_sizes[0] : weight_sizes[1]),
      overlay_length(weight_sizes[2], dilation[1]),
      overlay_length(weight_sizes[3], dilation[0]),
  };
}

// Uploads a pre-packed float payload (weight or bias) into channel-packed
// texture storage: the leading size-4 axis lands in the texel lanes, so a
// sampler fetch returns the four folded channels exactly like the packed
// 2D textures of the quantized convolution family.
api::vTensor upload_qconv_packed(const Tensor& packed_cpu) {
  api::Context* const context = api::context();
  Tensor packed = packed_cpu.to(Device(DeviceType::CPU)).contiguous();
  api::vTensor v{
      context,
      static_cast<std::vector<int64_t>>(packed.shape()),
      DType::Float32,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  Tensor nc = utils::nchw_to_nc4hw(packed);
  utils::upload_host_bytes(
      v, nc.impl()->storage().data(), nc.numel() * nc.itemsize());
  return v;
}

// One dispatch into the shader selected by the conv method, with the packed
// weight/bias textures and the shared qparam block.
Tensor record_qconv_op(
    api::Context* const context,
    const char* kernel_name,
    api::vTensor& v_output,
    api::vTensor& v_input,
    api::vTensor& v_weight,
    api::vTensor& v_bias,
    const std::vector<int64_t>& overlay_region,
    const std::vector<int64_t>& weight_sizes,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    double out_scale,
    int64_t out_zero_point,
    double input_scale,
    int64_t input_zero_point,
    float output_min,
    float output_max) {
  const struct QConvParams block{
      vec4(
          static_cast<float>(out_scale),
          static_cast<float>(input_scale),
          1.0f, // the packed kernel already carries dequantized weights
          1.0f), // the packed bias already carries float values
      ivec4(
          static_cast<int32_t>(out_zero_point),
          static_cast<int32_t>(input_zero_point),
          0, // kernel zero point: folded into the packed payload
          0), // bias zero point: float domain
      ivec3(
          static_cast<int32_t>(v_output.extents()[0u]),
          static_cast<int32_t>(v_output.extents()[1u]),
          static_cast<int32_t>(v_output.extents()[2u])),
      0,
      ivec3(
          static_cast<int32_t>(v_input.extents()[0u]),
          static_cast<int32_t>(v_input.extents()[1u]),
          static_cast<int32_t>(v_input.extents()[2u])),
      0,
      ivec4(
          static_cast<int32_t>(overlay_region[3]),
          static_cast<int32_t>(overlay_region[2]),
          static_cast<int32_t>(overlay_region[1]),
          static_cast<int32_t>(overlay_region[0])),
      ivec2(
          static_cast<int32_t>(weight_sizes[3]),
          static_cast<int32_t>(weight_sizes[2])),
      // The shader indexes the x axis with the width direction, so each
      // parameter pair lands as (width, height).
      ivec2(
          static_cast<int32_t>(stride[0]),
          static_cast<int32_t>(stride[1])),
      ivec2(
          static_cast<int32_t>(padding[0]),
          static_cast<int32_t>(padding[1])),
      ivec2(
          static_cast<int32_t>(dilation[0]),
          static_cast<int32_t>(dilation[1])),
      vec2(output_min, output_max)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(kernel_name),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out_codes = convert(v_output);
  return quantized::make_qtensor(
      out_codes,
      make_per_tensor_affine_quantizer(out_scale, out_zero_point,
                                       DType::QUInt8),
      DType::QUInt8);
}

} // namespace

Tensor quantized_conv2d_run_kernel(
    const Tensor& input,
    const Tensor& weight_packed,
    const Tensor& bias_packed,
    const std::vector<int64_t>& weight_sizes,
    double input_scale,
    int64_t input_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& output_padding,
    int64_t groups,
    bool transposed,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  TP_CHECK(
      input.dtype() == DType::QInt8,
      "Vulkan quantized_conv2d_run: activations must be QInt8");
  TP_CHECK(
      input.dim() == 4, "Vulkan quantized_conv2d_run requires a 4d input");
  TP_CHECK(
      weight_sizes.size() == 4,
      "Vulkan quantized_conv2d_run: expected a 4-D weight size list");
  TP_CHECK(
      out_scale > 0.0 && input_scale > 0.0,
      "Vulkan quantized_conv2d_run: scales must be positive");

  const auto expand2 = [](const std::vector<int64_t>& v, int64_t def) {
    return v.size() == 1 ? std::vector<int64_t>{v[0], v[0]}
                         : std::vector<int64_t>{v[0], v[1]};
  };
  const std::vector<int64_t> stride2 = expand2(stride, 1);
  const std::vector<int64_t> padding2 = expand2(padding, 0);
  const std::vector<int64_t> dilation2 = expand2(dilation, 1);
  const std::vector<int64_t> output_padding2 = expand2(output_padding, 0);

  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(
        NotImplementedError,
        "Vulkan quantized_conv2d_run requires texture storage");
  }

  const QConvMethod method =
      determine_qconv_method(weight_sizes, groups, transposed);
  const std::vector<int64_t> overlay_region =
      compute_qconv_overlay_region(weight_sizes, dilation2, transposed);

  // Output geometry: the regular convolution window formula, or the
  // transposed scatter formula over stride/padding/output_padding.
  const int64_t N = input.size(0);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  int64_t OH, OW, OC;
  if (transposed) {
    OC = weight_sizes[1]; // [in, out, KH, KW]
    OH = stride2[1] * (H - 1) + weight_sizes[2] - 2 * padding2[1] +
        output_padding2[1];
    OW = stride2[0] * (W - 1) + weight_sizes[3] - 2 * padding2[0] +
        output_padding2[0];
  } else {
    OC = weight_sizes[0];
    OH = (H + 2 * padding2[1] - dilation2[1] * (weight_sizes[2] - 1) - 1) /
            stride2[1] +
        1;
    OW = (W + 2 * padding2[0] - dilation2[0] * (weight_sizes[3] - 1) - 1) /
            stride2[0] +
        1;
  }
  TP_CHECK(
      OH > 0 && OW > 0,
      "Vulkan quantized_conv2d_run: computed output size is empty");

  api::vTensor v_output{context, {N, OC, OH, OW}, DType::UInt8};

  api::vTensor v_weight = upload_qconv_packed(weight_packed);
  api::vTensor v_bias = upload_qconv_packed(bias_packed);

  const float out_min = output_min.has_value()
      ? static_cast<float>(output_min->toDouble())
      : -std::numeric_limits<float>::infinity();
  const float out_max = output_max.has_value()
      ? static_cast<float>(output_max->toDouble())
      : std::numeric_limits<float>::infinity();

  const char* kernel_name;
  switch (method) {
    case QConvMethod::kDepthwise:
      kernel_name = "quantized_conv2d_dw";
      break;
    case QConvMethod::kPointwise:
      kernel_name = "quantized_conv2d_pw_2x2";
      break;
    default:
      kernel_name = transposed ? "quantized_conv_transpose2d"
                               : "quantized_conv2d_sw";
      break;
  }

  return record_qconv_op(
      context,
      kernel_name,
      v_output,
      v_input,
      v_weight,
      v_bias,
      overlay_region,
      weight_sizes,
      stride2,
      padding2,
      dilation2,
      out_scale,
      out_zero_point,
      input_scale,
      input_zero_point,
      out_min,
      out_max);
}

// Quantized-tensor metadata probes and the pass-through dequantize entry.
// The scale/zero-point pair lives on the host-side quantizer, so both
// probes answer without touching the payload; dequantize rides the
// per-dtype kernels that the quantizer's scheme selects.
double q_scale_kernel(const Tensor& self) {
  return quantized::q_scale(self);
}

int64_t q_zero_point_kernel(const Tensor& self) {
  return quantized::q_zero_point(self);
}

int64_t qscheme_kernel(const Tensor& self) {
  return static_cast<int64_t>(quantized::quantizer_of(self)->qscheme());
}

Tensor int_repr_kernel(const Tensor& self) {
  quantized::require_quantized(self, "int_repr");
  return quantized::strip_quantizer(self).clone();
}

//
// A code payload that reached the device as plain integers names a quantized
// tensor once the affine parameters that read it are attached.  The naming
// happens where the codes already sit: the payload is copied on the device, so
// a weight uploaded once never crosses back over the host, and the copy keeps
// the quantized tensor independent of the integer tensor it was made from.
//
Tensor make_per_tensor_quantized_tensor_kernel(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  TP_CHECK(
      scale > 0.0,
      "Vulkan _make_per_tensor_quantized_tensor: scale must be positive");
  const DType quantized_dtype = toQIntType(self.dtype());
  TP_CHECK(
      isQuantizedType(quantized_dtype),
      "Vulkan _make_per_tensor_quantized_tensor: expected an Int8, UInt8 or "
      "Int32 code payload to name");
  return quantized::make_qtensor(
      clone_kernel(self),
      make_per_tensor_affine_quantizer(scale, zero_point, quantized_dtype),
      quantized_dtype);
}

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

//
// The per-output-channel scales and addends of a W4A16 weight, staged as one
// linear single-precision payload: `params_width` slots of scales followed by
// `params_width` slots of addends, both zero-filled past their own length.
// The reader takes the pair from one binding, so a call without an addend
// still lands on the same shape and adds zero.
//
api::vTensor w4a16_linear_params(
    api::Context* const context,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& bias,
    const int64_t out_features,
    const int64_t params_width) {
  return ParamTextureCache::singleton().get_or_create(
      weight, {2, params_width}, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      kTagW4A16ParamsLinear,
      [&]() {
        std::vector<float> host(
            static_cast<size_t>(2 * params_width), 0.0f);
        const Tensor scales = weight_scales.to(Device(DeviceType::CPU))
                                  .to(DType::Float32)
                                  .contiguous();
        std::memcpy(
            host.data(),
            scales.data_ptr<float>(),
            sizeof(float) * static_cast<size_t>(out_features));
        if (bias.has_value()) {
          const Tensor bias_f = bias->to(Device(DeviceType::CPU))
                                    .to(DType::Float32)
                                    .contiguous();
          std::memcpy(
              host.data() + params_width,
              bias_f.data_ptr<float>(),
              sizeof(float) * static_cast<size_t>(out_features));
        }
        api::vTensor v{
            context,
            {2, params_width},
            DType::Float32,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
        utils::upload_host_bytes(
            v, host.data(), host.size() * sizeof(float));
        return v;
      });
}

//
// The 4-bit weight codes of an [N, K] weight packed into linear storage, eight
// to a word with one row per run of ceil(K / 8) words.  The pack runs device
// side over the weight's own payload, whichever storage it drew, so a weight
// uploaded once stays packed for every later call.
//
api::vTensor w4a16_linear_packed_weight(
    api::Context* const context,
    const Tensor& weight,
    const api::vTensor& v_weight,
    const int64_t K,
    const int64_t N,
    const int64_t k_words) {
  const int64_t words = N * k_words;
  return ParamTextureCache::singleton().get_or_create(
      weight, {words}, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      kTagW4A16WeightLinear,
      [&]() {
        api::vTensor v_packed{
            context,
            {words},
            DType::Int32,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
        const struct PackBlock final {
          int k;
          int n;
          int k_words;
          int fill;
        } pack_block{
            static_cast<int32_t>(K),
            static_cast<int32_t>(N),
            static_cast<int32_t>(k_words),
            0};
        api::UniformParamsBuffer pack_params(context, pack_block);
        api::PipelineBarrier pack_barrier{};
        const api::utils::uvec3 pack_grid{
            api::utils::safe_downcast_to_u32(words), 1u, 1u};
        const bool weight_linear =
            v_weight.storage_type() == api::StorageType::BUFFER;
        // The two builds differ only in where the codes come from, so the
        // payload the reader sees never depends on which form the weight drew.
        if (weight_linear) {
          context->submit_compute_job(
              VK_KERNEL(w4a16_pack_buffer),
              pack_barrier,
              pack_grid,
              adaptive_work_group_size(pack_grid),
              VK_NULL_HANDLE,
              v_packed.buffer(
                  pack_barrier,
                  api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_weight.buffer(pack_barrier, api::PipelineStage::COMPUTE),
              pack_params.buffer());
        } else {
          context->submit_compute_job(
              VK_KERNEL(w4a16_pack_image_input),
              pack_barrier,
              pack_grid,
              adaptive_work_group_size(pack_grid),
              VK_NULL_HANDLE,
              v_packed.buffer(
                  pack_barrier,
                  api::PipelineStage::COMPUTE,
                  api::MemoryAccessType::WRITE),
              v_weight.image(pack_barrier, api::PipelineStage::COMPUTE),
              pack_params.buffer());
        }
        return v_packed;
      });
}

//
// The W4A16 product over operands that have no texel form.  A weight whose row
// count passes the device's image axis limit -- the shape a large output
// projection has -- arrives in linear storage and never gains an image form, so
// the reduction must read codes and halves out of buffer words instead of
// sampling texels.  The result follows whichever storage its own shape fits,
// and a linear one keeps the packed pair of halves that its payload stores.
//
Tensor w4a16_linear_over_buffers(
    api::Context* const context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& bias,
    const api::vTensor& v_weight,
    const int64_t M,
    const int64_t N,
    const int64_t K) {
  const int64_t k_words = (K + 7) / 8;
  const int64_t params_width = api::utils::align_up(N, 4);

  // Every offset the kernels below compute is a 32-bit integer: the reader
  // names an element by row * N + col and the activation by row * K, so a
  // product that passes that range has no addressable form here.
  TP_CHECK(
      M * N <= std::numeric_limits<int32_t>::max(),
      "Vulkan w4a16_linear: the product of the row and channel counts passes "
      "what one linear payload can address; split the batch");
  TP_CHECK(
      M * K <= std::numeric_limits<int32_t>::max(),
      "Vulkan w4a16_linear: the activation rows pass what one linear payload "
      "can address; split the batch");

  api::PipelineBarrier pipeline_barrier{};

  api::vTensor v_act = to_buffer_storage_dispatch(context, convert(input));
  api::vTensor v_packed = w4a16_linear_packed_weight(
      context, weight, v_weight, K, N, k_words);
  api::vTensor v_params = w4a16_linear_params(
      context, weight, weight_scales, bias, N, params_width);

  api::vTensor v_output{context, {M, N}, DType::Float16};
  const bool out_buffer =
      v_output.storage_type() == api::StorageType::BUFFER;

  const struct LinearBlock final {
    ivec4 out_sizes;
    int k;
    int k_words;
    int params_words;
    int fill;
  } block{
      api::utils::make_whcn_ivec4({M, N}),
      static_cast<int32_t>(K),
      static_cast<int32_t>(k_words),
      static_cast<int32_t>(params_width / 4),
      0};
  api::UniformParamsBuffer params(context, block);

  // A linear result folds two elements into every word, so the launch counts
  // words; a texel one names one element per invocation.
  const api::utils::uvec3 grid = out_buffer
      ? api::utils::uvec3{
            api::utils::div_up(
                api::utils::safe_downcast_to_u32(M * N), 2u),
            1u,
            1u}
      : api::utils::uvec3{
            api::utils::safe_downcast_to_u32(N),
            api::utils::safe_downcast_to_u32(M),
            1u};
  const char* const kernel =
      out_buffer ? "w4a16_linear_buffer" : "w4a16_linear_buffer_image";

  if (out_buffer) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_act.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_packed.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_params.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel),
        pipeline_barrier,
        grid,
        adaptive_work_group_size(grid),
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_act.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_packed.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_params.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_output);
}

Tensor w4a16_linear_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scales,
    const std::optional<Tensor>& bias) {
  TP_CHECK(
      input.dtype() == DType::Float16,
      "Vulkan w4a16_linear: activations must be Float16");
  TP_CHECK(
      weight.dtype() == DType::QInt8,
      "Vulkan w4a16_linear: weights must be QInt8 with 4-bit codes in "
      "[-8, 7]");
  TP_CHECK(
      input.dim() == 2 && weight.dim() == 2,
      "Vulkan w4a16_linear: expected 2-D [M,K] activations and [N,K] "
      "weights");
  TP_CHECK(
      input.size(1) == weight.size(1),
      "Vulkan w4a16_linear: incompatible K dimensions");
  const int64_t out_features = weight.size(0);
  TP_CHECK(
      weight_scales.dim() == 1 && weight_scales.size(0) == out_features,
      "Vulkan w4a16_linear: weight_scales must be 1-D of length "
      "out_features");
  if (bias.has_value()) {
    TP_CHECK(
        bias->dtype() == DType::Float32 && bias->dim() == 1 &&
            bias->size(0) == out_features,
        "Vulkan w4a16_linear: bias must be a 1-D Float32 tensor of "
        "length out_features");
  }

  api::Context* const context = api::context();

  const int64_t M = input.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = input.size(1);
  if (M == 0 || N == 0) {
    // A product with no rows or no channels has no payload for either route to
    // stage, so the result is released empty instead of dispatched.
    return empty_kernel({M, N}, DType::Float16, input.device(), false);
  }
  const int64_t K_tex8 = (K + 7) / 8;
  const int64_t params_width = api::utils::align_up(out_features, 4);

  api::vTensor v_input = convert(input);
  api::vTensor v_weight = convert(weight);

  // The texel route stages three payloads as images: the packed weight row per
  // output channel, the parameter pair, and the product itself.  A shape with
  // no image form for any of them runs on linear storage instead, because
  // which storage a weight draws is a property of the device's image limit,
  // not of the caller's choosing.
  const bool texture_route =
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_weight.storage_type() == api::StorageType::TEXTURE_3D &&
      image_storage_supported(context, {K, N}) &&
      image_storage_supported(context, {M, N}) &&
      image_storage_supported(context, {3, params_width});
  if (!texture_route) {
    return w4a16_linear_over_buffers(
        context, input, weight, weight_scales, bias, v_weight, M, N, K);
  }

  // Per-channel scale (row 0) and bias (row 2) float texture, cached on
  // the weight identity.  Row 1 (zero points) is unused by the symmetric
  // W4A16 scheme and stays zero.
  api::vTensor v_params = ParamTextureCache::singleton().get_or_create(
      weight, {3, params_width}, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      kTagW4A16Params,
      [&]() {
        Tensor params_cpu = Tensor::zeros(
            {3, params_width}, DType::Float32, Device(DeviceType::CPU));
        Tensor sc = weight_scales.to(Device(DeviceType::CPU))
                        .to(DType::Float32)
                        .contiguous();
        std::memcpy(
            params_cpu.impl()->storage().data(),
            sc.data_ptr<float>(),
            sizeof(float) * out_features);
        if (bias.has_value()) {
          Tensor bias_f = bias->to(Device(DeviceType::CPU))
                              .to(DType::Float32)
                              .contiguous();
          std::memcpy(
              params_cpu.impl()->storage().data() +
                  sizeof(float) * 2 * params_width,
              bias_f.data_ptr<float>(),
              sizeof(float) * out_features);
        }
        api::vTensor v_p{
            context,
            {3, params_width},
            DType::Float32,
            api::StorageType::TEXTURE_3D,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
        utils::upload_host_bytes(
            v_p,
            params_cpu.contiguous().impl()->storage().data(),
            params_cpu.numel() * params_cpu.itemsize());
        return v_p;
      });

  // The packed 4-bit weight payload (eight nibbles per texel) is
  // materialized once per weight identity (weight prepack).
  api::PipelineBarrier pipeline_barrier{};
  api::vTensor v_weight_packed = ParamTextureCache::singleton().get_or_create(
      weight, {K, N}, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      kTagW4A16WeightHeightPacked,
      [&]() {
        const struct PackBlock final {
          int k;
          int n;
        } pack_w_block{static_cast<int32_t>(K), static_cast<int32_t>(N)};
        api::UniformParamsBuffer pack_w_params(context, pack_w_block);
        // Logical [K, N] under TENSOR_HEIGHT_PACKED yields a texture of
        // {N, ceil(K/4), 1}; pack_int4_height writes only the first
        // ceil(K/8) rows (eight nibbles per texel) and the GEMM reads no
        // further, so the extra rows are simply never touched.
        api::vTensor v_wp{
            context,
            {K, N},
            DType::Int8,
            api::StorageType::TEXTURE_3D,
            api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED};
        const api::utils::uvec3 pack_grid{
            api::utils::safe_downcast_to_u32(N),
            api::utils::safe_downcast_to_u32(K_tex8),
            1u};
        context->submit_compute_job(
            VK_KERNEL(pack_int4_height), pipeline_barrier, pack_grid,
            adaptive_work_group_size(pack_grid), VK_NULL_HANDLE,
            v_wp.image(
                pipeline_barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
            pack_w_params.buffer());
        return v_wp;
      });

  // Width-pack the FP16 activations (one half per texel in the dense
  // channel-packed layout -> four halves per texel) using the same
  // relayout kernel the FP16 matmul path relies on.
  api::vTensor v_input_packed{
      context,
      {M, K},
      DType::Float16,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const struct PackBlockAct final {
    ivec4 sizes;
  } pack_in_block{api::utils::make_ivec4_prepadded1({M, K})};
  api::UniformParamsBuffer pack_in_params(context, pack_in_block);
  context->submit_compute_job(
      VK_KERNEL(convert_channels_to_width_packed_f16), pipeline_barrier,
      v_input_packed.extents(),
      adaptive_work_group_size(v_input_packed.extents()), VK_NULL_HANDLE,
      v_input_packed.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      pack_in_params.buffer());

  api::vTensor v_output{context, {M, N}, DType::Float16};

  if (M == 1) {
    // Decode shape: one thread per output channel, no shared-memory
    // staging (see w4a16_linear_m1.glsl).
    const struct W4A16M1Block final {
      int k;
      int out_n;
    } m1_block{static_cast<int32_t>(K), static_cast<int32_t>(N)};
    api::UniformParamsBuffer params(context, m1_block);
    const api::utils::uvec3 grid{
        api::utils::safe_downcast_to_u32(N), 1u, 1u};
    context->submit_compute_job(
        VK_KERNEL(w4a16_linear_m1), pipeline_barrier, grid,
        adaptive_work_group_size(grid), VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_params.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct W4A16Block final {
    int out_m;
    int out_n;
    int k; // logical reduction width
  } gemm_block{
      static_cast<int32_t>(M),
      static_cast<int32_t>(N),
      static_cast<int32_t>(K)};
  api::UniformParamsBuffer params(context, gemm_block);

  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32((N + 3) / 4),
      api::utils::safe_downcast_to_u32((M + 3) / 4),
      1u};

  context->submit_compute_job(
      VK_KERNEL(w4a16_linear_tiled), pipeline_barrier, grid,
      {8u, 8u, 1u}, VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight_packed.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_params.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor dequantize_self_kernel(const Tensor& self) {
  if (!quantized::is_quantized(self)) {
    return self.to(DType::Float32);
  }
  return quantized::quantizer_of(self)->dequantize(self);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, QuantKernels) {
  m.impl("quantize_per_tensor",
         &tensorplay::vulkan::ops::quantize_per_tensor_kernel);
  m.impl("quantized_add", &tensorplay::vulkan::ops::quantized_add_kernel);
  m.impl("quantized_sub", &tensorplay::vulkan::ops::quantized_sub_kernel);
  m.impl("quantized_mul", &tensorplay::vulkan::ops::quantized_mul_kernel);
  m.impl("quantized_div", &tensorplay::vulkan::ops::quantized_div_kernel);
  m.impl("quantized_clamp", &tensorplay::vulkan::ops::quantized_clamp_kernel);
  m.impl("quantized_max_pool2d",
         &tensorplay::vulkan::ops::quantized_max_pool2d_kernel);
  m.impl("quantized_linear",
         &tensorplay::vulkan::ops::quantized_linear_kernel);
  m.impl("w4a16_linear",
         &tensorplay::vulkan::ops::w4a16_linear_kernel);
  m.impl("quantized_conv2d",
         &tensorplay::vulkan::ops::quantized_conv2d_kernel);
  m.impl("quantized_conv2d_run",
         &tensorplay::vulkan::ops::quantized_conv2d_run_kernel);
  m.impl("q_scale", &tensorplay::vulkan::ops::q_scale_kernel);
  m.impl("q_zero_point", &tensorplay::vulkan::ops::q_zero_point_kernel);
  m.impl("qscheme", &tensorplay::vulkan::ops::qscheme_kernel);
  m.impl("int_repr", &tensorplay::vulkan::ops::int_repr_kernel);
  m.impl(
      "_make_per_tensor_quantized_tensor",
      &tensorplay::vulkan::ops::make_per_tensor_quantized_tensor_kernel);
  m.impl("dequantize.self", &tensorplay::vulkan::ops::dequantize_self_kernel);
  m.impl("dequantize_affine",
         &tensorplay::vulkan::ops::dequantize_affine_kernel);
}

#endif /* USE_VULKAN */
