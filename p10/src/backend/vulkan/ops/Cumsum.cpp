#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>

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
      "Vulkan cumsum: dimension out of range");
  return static_cast<int>(t.dim() - 1 - wrapped);
}

struct ScanBlock final {
  ivec4 in_sizes; // (W, H, C, N)
  int axis;
  int c_depth;
  int fill;
};

} // namespace

Tensor cumsum_kernel(
    const Tensor& self, int64_t dim, std::optional<DType> dtype) {
  validate_float_1d_to_4d(self, "cumsum");
  TP_CHECK(
      !dtype.has_value() || dtype.value() == DType::Float32,
      "Vulkan cumsum supports Float32 output only");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(NotImplementedError, "Vulkan cumsum requires texture storage");
  }

  api::vTensor v_output{context, v_input.sizes(), DType::Float32};

  const struct ScanBlock block{
      make_whcn_ivec4(v_input.sizes()),
      innermost_axis(self, dim),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(cumsum), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor& cumsum_inplace_kernel(
    Tensor& self, int64_t dim, std::optional<DType> dtype) {
  self.copy_(cumsum_kernel(self, dim, dtype));
  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, CumsumKernels) {
  m.impl("cumsum", &tensorplay::vulkan::ops::cumsum_kernel);
  m.impl("cumsum_", &tensorplay::vulkan::ops::cumsum_inplace_kernel);
}

#endif /* USE_VULKAN */
