#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include <optional>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Forward declaration of the element type conversion entry point.
Tensor to_dtype_kernel(
    const Tensor& self,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format);

namespace {

void validate_float_1d_to_4d(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16,
      std::string("Vulkan ") + name +
          " supports Float32 and Float16 tensors only");
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
  const DType out_dtype = dtype.value_or(self.dtype());
  TP_CHECK(
      out_dtype == DType::Float32 || out_dtype == DType::Float16,
      "Vulkan cumsum supports Float32 and Float16 outputs only");
  if (out_dtype != self.dtype()) {
    // A requested result width scans in the payload's own type first: the
    // conversion path owns both directions over each storage, so the scan
    // kernels keep one element type end to end.
    return to_dtype_kernel(
        cumsum_kernel(self, dim, std::nullopt),
        out_dtype,
        /*non_blocking=*/false,
        /*copy=*/true,
        std::nullopt);
  }

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  // Buffer-stored payloads (any axis beyond the 3-D image limit) scan
  // through a linear row-walking variant: one invocation accumulates the
  // running total along the axis and writes every prefix.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    const int64_t wrapped = dim < 0 ? dim + self.dim() : dim;
    const int64_t axis_size = self.size(wrapped);
    int64_t inner = 1;
    for (int64_t d = wrapped + 1; d < self.dim(); ++d) {
      inner *= self.size(d);
    }
    const int64_t rows = self.numel() / axis_size;
    api::vTensor v_output{
        context,
        v_input.sizes(),
        out_dtype,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    // A half payload shares one 32-bit word between two neighbouring inner
    // positions, which belong to different rows, so the scan runs twice and
    // each pass stores only the flat positions of one word lane.
    const bool is_half = out_dtype == DType::Float16;
    const int32_t passes = is_half ? 2 : 1;
    api::PipelineBarrier pipeline_barrier{};
    for (int32_t pass = 0; pass < passes; ++pass) {
      const struct CumsumBufferBlock final {
        int32_t axis_size;
        int32_t inner;
        int32_t rows;
        int32_t word_lane;
      } block{
          static_cast<int32_t>(axis_size),
          static_cast<int32_t>(inner),
          static_cast<int32_t>(rows),
          is_half ? pass : 0,
      };
      api::UniformParamsBuffer params(context, block);
      context->submit_compute_job(
          is_half ? VK_KERNEL_FROM_STR("cumsum_buffer_half")
                  : VK_KERNEL(cumsum_buffer),
          pipeline_barrier,
          {static_cast<uint32_t>(rows), 1u, 1u},
          {256u, 1u, 1u},
          VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              is_half ? (api::MemoryAccessType::READ |
                         api::MemoryAccessType::WRITE)
                      : api::MemoryAccessType::WRITE),
          v_input.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ),
          params.buffer());
    }
    return convert(v_output);
  }

  if (v_input.storage_type() != api::StorageType::TEXTURE_3D) {
    TP_THROW(NotImplementedError, "Vulkan cumsum requires texture storage");
  }

  api::vTensor v_output{context, v_input.sizes(), out_dtype};

  const struct ScanBlock block{
      make_whcn_ivec4(v_input.sizes()),
      innermost_axis(self, dim),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // The running total is kept in single precision inside the shader, so a
  // half result only needs the store to name the narrower format.
  context->submit_compute_job(
      kernel_for("cumsum", out_dtype), pipeline_barrier, v_output.extents(),
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
