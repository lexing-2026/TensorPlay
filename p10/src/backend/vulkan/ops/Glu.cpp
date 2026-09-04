#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_float_1d_to_4d(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

// Maps a logical dimension index onto the shader's axis numbering, which
// counts from the innermost (width) position of the WHCN order.
int innermost_axis(const Tensor& t, int64_t dim) {
  const int64_t wrapped = dim < 0 ? dim + t.dim() : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < t.dim(),
      "Vulkan glu: dimension out of range");
  return static_cast<int>(t.dim() - 1 - wrapped);
}

struct GluBlock final {
  ivec4 in_sizes; // (W, H, C, N)
  ivec4 out_sizes;
  int axis;
  int in_c_depth;
  int out_c_depth;
  int fill;
};

} // namespace

Tensor glu_kernel(const Tensor& self, int64_t dim) {
  validate_float_1d_to_4d(self, "glu");
  const int64_t wrapped = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(wrapped >= 0 && wrapped < self.dim(), "Vulkan glu: bad dim");
  const int64_t in_len = self.size(wrapped);
  TP_CHECK(
      in_len % 2 == 0,
      "Vulkan glu: halving dimension must be even");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(NotImplementedError, "Vulkan glu requires texture storage");
  }

  std::vector<int64_t> out_sizes(
      v_input.sizes().begin(), v_input.sizes().end());
  out_sizes[static_cast<size_t>(wrapped)] = in_len / 2;

  api::vTensor v_output{context, out_sizes, DType::Float32};

  const struct GluBlock block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      innermost_axis(self, dim),
      c_depth_of(v_input.sizes()),
      c_depth_of(v_output.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(glu), pipeline_barrier, v_output.extents(),
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

TENSORPLAY_LIBRARY_IMPL(Vulkan, GluKernels) {
  m.impl("glu", &tensorplay::vulkan::ops::glu_kernel);
}

#endif /* USE_VULKAN */
