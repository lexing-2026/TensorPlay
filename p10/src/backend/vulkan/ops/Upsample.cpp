#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_float_4d(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() == 4, std::string("Vulkan ") + name + " requires a 4d tensor");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

struct Nearest2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  float scale_w;
  float scale_h;
  int c_depth;
  int fill0;
};

struct BilinearBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  float rwidth;
  float rheight;
  int align_corners;
  int out_c_depth;
};

} // namespace

Tensor upsample_nearest2d_kernel(
    const Tensor& self,
    std::vector<int64_t> output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  validate_float_4d(self, "upsample_nearest2d");
  TP_CHECK(
      output_size.size() == 2,
      "Vulkan upsample_nearest2d expects a 2d output size");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  // The output size list is (H, W), matching the CPU convention.
  const int64_t OH = output_size[0];
  const int64_t OW = output_size[1];

  api::vTensor v_output{context, {self.size(0), self.size(1), OH, OW},
                        self.dtype()};

  const struct Nearest2DBlock block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      static_cast<float>(
          scales_w.value_or(static_cast<double>(self.size(3)) /
                            static_cast<double>(OW))),
      static_cast<float>(
          scales_h.value_or(static_cast<double>(self.size(2)) /
                            static_cast<double>(OH))),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(upsample_nearest2d), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor upsample_bilinear2d_kernel(
    const Tensor& self,
    std::vector<int64_t> output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  validate_float_4d(self, "upsample_bilinear2d");
  TP_CHECK(
      output_size.size() == 2,
      "Vulkan upsample_bilinear2d expects a 2d output size");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  // The output size list is (H, W), matching the CPU convention.
  const int64_t OH = output_size[0];
  const int64_t OW = output_size[1];
  const int64_t IH = self.size(2);
  const int64_t IW = self.size(3);
  TP_CHECK(OH > 0 && OW > 0, "Vulkan upsample_bilinear2d: empty output");

  // Area-pixel scale: corner-aligned steps span (in-1)/(out-1); the
  // half-pixel form falls back to in/out unless an explicit scale rides in.
  const auto scale_of = [&](int64_t in_len, int64_t out_len,
                            std::optional<double> scale) {
    if (align_corners) {
      return out_len > 1
          ? static_cast<double>(in_len - 1) / static_cast<double>(out_len - 1)
          : 0.0;
    }
    return (scale.has_value() && scale.value() > 0.)
        ? scale.value()
        : static_cast<double>(in_len) / static_cast<double>(out_len);
  };
  const double rwidth = scale_of(IW, OW, scales_w);
  const double rheight = scale_of(IH, OH, scales_h);

  api::vTensor v_output{context, {self.size(0), self.size(1), OH, OW},
                        self.dtype()};

  const struct BilinearBlock block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      static_cast<float>(rwidth),
      static_cast<float>(rheight),
      align_corners ? 1 : 0,
      c_depth_of(v_output.sizes()),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL_FROM_STR(align_corners ? "upsample_bilinear2d"
                                       : "upsample_bilinear2d_half"),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
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

TENSORPLAY_LIBRARY_IMPL(Vulkan, UpsampleKernels) {
  m.impl(
      "upsample_nearest2d",
      &tensorplay::vulkan::ops::upsample_nearest2d_kernel);
  m.impl(
      "upsample_bilinear2d",
      &tensorplay::vulkan::ops::upsample_bilinear2d_kernel);
}

#endif /* USE_VULKAN */
