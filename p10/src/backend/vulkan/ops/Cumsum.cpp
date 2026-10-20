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

// A payload that counts rather than measures.  Its running total can leave the
// range one element fills, so a count never scans in its own width.
bool is_count_type(const DType dtype) {
  return dtype == DType::Int8 || dtype == DType::UInt8 ||
      dtype == DType::Int32 || dtype == DType::Int64 || dtype == DType::Bool;
}

void validate_scan_target(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16 ||
          is_count_type(t.dtype()),
      std::string("Vulkan ") + name +
          " supports Float32, Float16, Int8, UInt8, Int32, Int64 and Bool "
          "tensors");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

// The scan kernels read a flat position per word, so a view has to be laid out
// before it is walked.
Tensor dense_payload(const Tensor& t) {
  if (t.is_contiguous() && t.unsafeGetTensorImpl()->storage_offset() == 0) {
    return t;
  }
  return t.clone();
}

Tensor as_type(const Tensor& t, const DType dtype) {
  if (t.dtype() == dtype) {
    return t;
  }
  return to_dtype_kernel(
      t, dtype, /*non_blocking=*/false, /*copy=*/true, std::nullopt);
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

/*
 * Row-walking scan over linear storage.  One invocation owns one scan row: it
 * accumulates the running total along the axis and writes it at every position
 * of that row, so the walk is serial only within a row and the launch covers
 * the rows in parallel.  A half payload shares one 32-bit word between two
 * neighbouring inner positions, which belong to different rows, so that build
 * runs one pass per word lane; a single-precision word and a 64-bit pair of
 * words are each owned by one position, so they need a single pass.
 */
Tensor cumsum_linear(const Tensor& payload, const int64_t axis) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(payload);

  const int64_t axis_size = payload.size(axis);
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < payload.dim(); ++d) {
    inner *= payload.size(d);
  }
  api::vTensor v_output{
      context,
      v_input.sizes(),
      payload.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  // Nothing to walk, and a row count that an empty axis would leave undefined.
  if (payload.numel() == 0) {
    return convert(v_output);
  }
  const int64_t rows = payload.numel() / axis_size;

  const bool is_half = payload.dtype() == DType::Float16;
  const bool is_wide = payload.dtype() == DType::Int64;
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
        is_wide ? VK_KERNEL_FROM_STR("cumsum_buffer_i64")
                : (is_half ? VK_KERNEL_FROM_STR("cumsum_buffer_half")
                           : VK_KERNEL(cumsum_buffer)),
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

} // namespace

Tensor cumsum_kernel(
    const Tensor& self, int64_t dim, std::optional<DType> dtype) {
  validate_scan_target(self, "cumsum");
  const int64_t axis = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(
      axis >= 0 && axis < self.dim(), "Vulkan cumsum: dimension out of range");

  // A count answers with the wide code unless the caller names a width: the
  // total of a prefix needs more range than one count.
  const DType out_dtype = dtype.value_or(
      is_count_type(self.dtype()) ? DType::Int64 : self.dtype());
  TP_CHECK(
      out_dtype == DType::Float32 || out_dtype == DType::Float16 ||
          out_dtype == DType::Int32 || out_dtype == DType::Int64,
      "Vulkan cumsum supports Float32, Float16, Int32 and Int64 results");

  const Tensor payload = dense_payload(self);

  if (out_dtype == DType::Int32 || out_dtype == DType::Int64) {
    // Every counted result is scanned in the wide code and narrowed after it.
    // Truncating a prefix equals wrapping the total inside one word at every
    // step, so a 32-bit answer does not depend on the wider carry.  A
    // byte-packed payload and a half one have no direct route to a 64-bit code,
    // and single precision holds each of them exactly, so those widen through
    // the single-precision hub.
    Tensor wide = payload;
    if (wide.dtype() == DType::Bool || wide.dtype() == DType::Int8 ||
        wide.dtype() == DType::UInt8 || wide.dtype() == DType::Float16) {
      wide = as_type(wide, DType::Float32);
    }
    wide = as_type(wide, DType::Int64);
    const Tensor scanned = cumsum_linear(wide, axis);
    return out_dtype == DType::Int64
        ? scanned
        : as_type(scanned, DType::Int32);
  }

  if (payload.dtype() != out_dtype) {
    // The walk runs in the width the result is asked for, so the running total
    // rounds the way that width does rather than the payload's own.
    return cumsum_kernel(as_type(payload, out_dtype), axis, std::nullopt);
  }

  // Buffer-stored payloads (any axis beyond the 3-D image limit) keep the
  // linear walk; an image scans texel by texel instead.
  if (convert(payload).storage_type() == api::StorageType::BUFFER) {
    return cumsum_linear(payload, axis);
  }

  api::Context* const context = api::context();
  api::vTensor v_input = convert(payload);
  api::vTensor v_output{context, v_input.sizes(), out_dtype};

  const struct ScanBlock block{
      make_whcn_ivec4(v_input.sizes()),
      innermost_axis(payload, axis),
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
