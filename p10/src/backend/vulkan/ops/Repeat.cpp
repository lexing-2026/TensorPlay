#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "Utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in Cumsum.cpp; the running totals of the counts bound every span.
Tensor cumsum_kernel(
    const Tensor& self, int64_t dim, std::optional<DType> dtype);

// Defined in Copy.cpp; the element type conversion entry point.
Tensor to_dtype_kernel(
    const Tensor& self,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format);

namespace {

// The fill walk reads flat positions, so counts that reach it as a view - one
// number spread over a block, say - are laid out first.
Tensor dense_counts(const Tensor& t) {
  if (t.is_contiguous() && t.unsafeGetTensorImpl()->storage_offset() == 0) {
    return t;
  }
  return t.clone();
}

// Lanes asked to serve one segment and the workgroup they are launched in.  The
// workgroup is a whole multiple of the lanes, so every segment's lanes stay in
// one workgroup and an idle lane still costs nothing but a compare.
constexpr int32_t kLanesPerSegment = 32;
constexpr int32_t kFillWorkgroup = 256;

// A launch this wide already keeps a mobile GPU busy walking spans, and the
// segment count is bounded only by the size of the count payload, so the extent
// stays inside a fixed grid - well under the workgroups per axis any conformant
// device must offer - and the shader strides the rest of the segments through
// it.
constexpr int32_t kMaxFillWorkgroups = 2048;

// The grid the fill runs in: the extent to launch and the segment-and-lane
// groups it covers, which is the stride a run of more segments than that steps
// by.
struct FillGrid final {
  api::utils::uvec3 extent;
  int32_t groups;
};

FillGrid fill_grid(int64_t segments) {
  const int64_t needed =
      (segments * kLanesPerSegment + kFillWorkgroup - 1) / kFillWorkgroup;
  const int64_t workgroups = std::min<int64_t>(needed, kMaxFillWorkgroups);
  return FillGrid{
      {static_cast<uint32_t>(workgroups * kFillWorkgroup), 1u, 1u},
      static_cast<int32_t>(workgroups * (kFillWorkgroup / kLanesPerSegment))};
}

} // namespace

/*
 * The source position of every slot in an expansion: segment i answers with the
 * number i in the slots its count spans, so an expansion of counts is one write
 * per slot and none of them twice.
 *
 * The length of the answer is the total of the counts, which only the whole
 * payload can tell, so the running totals come to the host once: they give the
 * length, the bound that keeps every span inside one word, and the check that no
 * count is negative - a decreasing prefix is exactly what a negative count
 * leaves behind.  The spans go back as pairs of 32-bit codes rather than have
 * every segment re-read the totals of the segments above it.
 */
Tensor repeat_interleave_indices_kernel(
    const Tensor& repeats,
    std::optional<int64_t> output_size) {
  TP_CHECK(
      repeats.dim() == 1, "Vulkan repeat_interleave takes a 1d count payload");
  TP_CHECK(
      repeats.dtype() == DType::Int32 || repeats.dtype() == DType::Int64,
      "Vulkan repeat_interleave counts in Int32 or Int64");

  const int64_t segments = repeats.numel();
  const DType code_dtype = repeats.dtype();
  if (segments == 0) {
    return empty_kernel({0}, code_dtype, repeats.device(), false);
  }

  const Tensor ends =
      cumsum_kernel(dense_counts(repeats), 0, DType::Int64);
  const Tensor drawn = ends.to(Device(DeviceType::CPU));
  const int64_t* prefixes = drawn.data_ptr<int64_t>();

  const int64_t required = prefixes[segments - 1];
  const int64_t total = output_size.value_or(required);
  TP_CHECK(
      total == required,
      "Vulkan repeat_interleave: the answer length is ", total,
      " but the counts total ", required);
  TP_CHECK(
      total >= 0 && total <= std::numeric_limits<int32_t>::max(),
      "Vulkan repeat_interleave answers with at most a word of slots, got ",
      total);

  std::vector<int32_t> spans(static_cast<size_t>(2 * segments));
  int64_t previous = 0;
  for (int64_t i = 0; i < segments; ++i) {
    const int64_t end = prefixes[static_cast<size_t>(i)];
    TP_CHECK(
        end >= previous, "repeats can not be negative");
    spans[static_cast<size_t>(2 * i)] = static_cast<int32_t>(previous);
    spans[static_cast<size_t>(2 * i) + 1] = static_cast<int32_t>(end);
    previous = end;
  }

  api::Context* const context = api::context();
  api::vTensor v_spans{
      context,
      {2 * segments},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  utils::upload_host_bytes(
      v_spans, spans.data(), spans.size() * sizeof(int32_t));
  api::vTensor v_output{
      context,
      {total},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const FillGrid grid = fill_grid(segments);
  const struct FillBlock final {
    int32_t segments;
    int32_t total;
    int32_t groups;
    int32_t fill1;
  } block{
      static_cast<int32_t>(segments),
      static_cast<int32_t>(total),
      grid.groups,
      0,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("repeat_interleave_fill"),
      pipeline_barrier,
      grid.extent,
      {static_cast<uint32_t>(kFillWorkgroup), 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_spans.buffer(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());

  const Tensor codes = convert(v_output);
  if (code_dtype == DType::Int64) {
    // The walk answers in one word per slot; a payload counted in the wide code
    // asks for its positions in that width too.
    return to_dtype_kernel(
        codes, DType::Int64, /*non_blocking=*/false, /*copy=*/true,
        std::nullopt);
  }
  return codes;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, RepeatKernels) {
  m.impl(
      "repeat_interleave.Tensor",
      &tensorplay::vulkan::ops::repeat_interleave_indices_kernel);
}

#endif /* USE_VULKAN */
