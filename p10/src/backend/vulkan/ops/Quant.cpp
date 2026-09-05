#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Quantizer.h"
#include "Utils.h"

#include <Utils.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

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
    std::optional<Scalar> min,
    std::optional<Scalar> max) {
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
  // input's quantizer; the CUDA path follows the same shape.
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
    std::optional<Tensor> bias) {
  TP_CHECK(
      input.dtype() == DType::QInt8 && weight.dtype() == DType::QInt8,
      "Vulkan quantized_linear: activations and weights must be QInt8");
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
  // The callers may hand the qparams over as device tensors; the packing
  // happens on the host, so everything is gathered to CPU first.
  Tensor params_cpu = Tensor::zeros(
      {3, out_features}, DType::Float32, Device(DeviceType::CPU));
  Tensor sc =
      weight_scales.to(Device(DeviceType::CPU)).to(DType::Float32).contiguous();
  Tensor zp = weight_zero_points.to(Device(DeviceType::CPU))
                  .to(DType::Float32)
                  .contiguous();
  std::memcpy(
      params_cpu.impl()->storage().data() + sizeof(float) * out_features,
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
            sizeof(float) * 2 * out_features,
        bias_f.data_ptr<float>(),
        sizeof(float) * out_features);
  }

  api::vTensor v_params{context, {3, out_features}, DType::Float32};
  {
    Tensor params_nc4hw = utils::nchw_to_nc4hw(params_cpu.contiguous());
    utils::upload_host_bytes(
        v_params,
        params_nc4hw.impl()->storage().data(),
        params_nc4hw.numel() * params_nc4hw.itemsize());
  }

  const int64_t M = input.size(0);
  const int64_t N = weight.size(0);
  api::vTensor v_output{context, {M, N}, DType::Float32};

  const struct QuantizedLinearBlock block{
      static_cast<int32_t>(M),
      static_cast<int32_t>(N),
      static_cast<int32_t>(input.size(1)),
      static_cast<float>(input_scale),
      static_cast<int32_t>(input_zero_point)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(quantized_linear), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_params.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor quantized_conv2d_kernel(
    const Tensor& input,
    const Tensor& weight,
    std::optional<Tensor> bias,
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
    std::optional<Scalar> output_min,
    std::optional<Scalar> output_max) {
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
  m.impl("quantized_conv2d",
         &tensorplay::vulkan::ops::quantized_conv2d_kernel);
  m.impl("quantized_conv2d_run",
         &tensorplay::vulkan::ops::quantized_conv2d_run_kernel);
  m.impl("q_scale", &tensorplay::vulkan::ops::q_scale_kernel);
  m.impl("q_zero_point", &tensorplay::vulkan::ops::q_zero_point_kernel);
  m.impl("qscheme", &tensorplay::vulkan::ops::qscheme_kernel);
  m.impl("int_repr", &tensorplay::vulkan::ops::int_repr_kernel);
  m.impl("dequantize.self", &tensorplay::vulkan::ops::dequantize_self_kernel);
}

#endif /* USE_VULKAN */
