#ifdef USE_VULKAN

#include "Blocks.h"
#include "Convert.h"
#include "Shape.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {
namespace {

std::tuple<Tensor, Tensor> sort_impl(
    const Tensor& self,
    int64_t dim,
    bool descending,
    int64_t count,
    int64_t start = 0) {
  TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
           "Vulkan sort supports Float32 and Float16 tensors only");
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4,
           "Vulkan sort supports 1d to 4d tensors");
  dim = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(dim >= 0 && dim < self.dim(), "Vulkan sort: dim out of range");
  const int64_t length = self.size(dim);
  if (count < 0) count = length;
  TP_CHECK(count >= 0 && start >= 0 && start + count <= length,
           "Vulkan selection range is out of bounds");
  TP_CHECK(self.numel() <= std::numeric_limits<int32_t>::max(),
           "Vulkan sort exceeds the index range");
  int64_t inner = 1;
  for (int64_t d = dim + 1; d < self.dim(); ++d) inner *= self.size(d);
  const bool is_half = self.dtype() == DType::Float16;
  const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
  auto out_sizes = sizes;
  out_sizes[dim] = count;
  api::Context* context = api::context();
  api::vTensor input = convert(self);
  api::vTensor values{context, out_sizes, self.dtype()};
  api::vTensor indices{context, out_sizes, DType::Int32};
  if (values.numel() == 0) return {convert(values), convert(indices)};

  // Sort 2048-element tiles in shared memory (one workgroup per tile, all
  // 256 lanes active), then merge sorted runs.  Larger tiles cut the number of
  // global merge passes; the final gather also truncates top-k.
  // The shader drives every strided loop off 256 lanes, so the local work
  // group must stay exactly 256 wide.  Dispatches convert their invocation
  // domain to workgroup counts by dividing by the local size, hence tiles
  // workgroups need tiles * 256 invocations on the x axis.
  // Scratch keeps every (value, index) pair as one uint32 pair and must live
  // in device-local memory: the merge passes binary-search it densely, and
  // host-visible backing would turn every gather into a PCIe round trip.
  constexpr uint32_t kSortLanes = 256u;
  const int32_t tiles = static_cast<int32_t>((length + 2047) / 2048);
  const int32_t rows = static_cast<int32_t>(self.numel() / length);
  struct SortBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int32_t length;
    int32_t inner;
    int32_t rows;
    int32_t tiles;
    int32_t descending;
    int32_t run;
    int32_t count;
    int32_t start;
  } block{
      make_whcn_ivec4(sizes), make_whcn_ivec4(out_sizes),
      static_cast<int32_t>(length), static_cast<int32_t>(inner),
      rows, tiles, descending ? 1 : 0, 2048,
      static_cast<int32_t>(count), static_cast<int32_t>(start)};
  const std::vector<int64_t> scratch_sizes{self.numel() * 2};
  api::vTensor scratch(
      context, scratch_sizes, DType::Int32, api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  // Shapes beyond the 3-D image axis limit fall back to buffer storage at
  // allocation time; their tiles read the contiguous input buffer and their
  // final gather writes contiguous output buffers through dedicated
  // variants.  The merge passes already work on raw storage buffers.
  // A narrow payload needs its own reading of the input: an image carries a
  // half in its texel, which a plain sampler widens on the way in, while a
  // linear payload packs two of them into the word a tile sort must unpack.
  const bool buffer_input =
      input.storage_type() == api::StorageType::BUFFER;
  if (buffer_input) {
    context->submit_compute_job(
        is_half ? VK_KERNEL(sort_buffer_half) : VK_KERNEL(sort_buffer),
        barrier,
        {static_cast<uint32_t>(tiles) * kSortLanes,
         static_cast<uint32_t>(rows), 1u},
        {kSortLanes, 1u, 1u}, VK_NULL_HANDLE,
        scratch.buffer(barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::WRITE),
        input.buffer(barrier, api::PipelineStage::COMPUTE,
                     api::MemoryAccessType::READ),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL(sort), barrier,
        {static_cast<uint32_t>(tiles) * kSortLanes,
         static_cast<uint32_t>(rows), 1u},
        {kSortLanes, 1u, 1u}, VK_NULL_HANDLE,
        scratch.buffer(barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::WRITE),
        input.image(barrier, api::PipelineStage::COMPUTE), params.buffer());
  }

  std::unique_ptr<api::vTensor> alternate;
  if (length > 2048) {
    alternate = std::make_unique<api::vTensor>(
        context, scratch_sizes, DType::Int32, api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  }
  api::vTensor* current = &scratch;
  api::vTensor* next = alternate.get();
  for (int64_t run = 2048; run < length; run *= 2) {
    block.run = static_cast<int32_t>(run);
    api::UniformParamsBuffer merge_params(context, block);
    api::PipelineBarrier merge_barrier{};
    context->submit_compute_job(
        VK_KERNEL(sort_merge), merge_barrier,
        {static_cast<uint32_t>(length), static_cast<uint32_t>(rows), 1u},
        {kSortLanes, 1u, 1u}, VK_NULL_HANDLE,
        next->buffer(merge_barrier, api::PipelineStage::COMPUTE,
                     api::MemoryAccessType::WRITE),
        current->buffer(merge_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::READ),
        merge_params.buffer());
    std::swap(current, next);
  }
  api::PipelineBarrier output_barrier{};
  if (values.storage_type() == api::StorageType::BUFFER) {
    // A narrow result is written a word at a time, so the gather walks half as
    // many destinations as there are elements.
    const uint32_t destinations = is_half
        ? static_cast<uint32_t>((values.numel() + 1) / 2)
        : static_cast<uint32_t>(values.numel());
    context->submit_compute_job(
        is_half ? VK_KERNEL(sort_output_buffer_half)
                : VK_KERNEL(sort_output_buffer),
        output_barrier,
        {destinations, 1u, 1u},
        {kSortLanes, 1u, 1u}, VK_NULL_HANDLE,
        values.buffer(output_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        indices.buffer(output_barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::WRITE),
        current->buffer(output_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::READ),
        params.buffer());
    return {convert(values), convert(indices)};
  }
  context->submit_compute_job(
      kernel_for("sort_output", self.dtype()), output_barrier, values.extents(),
      adaptive_work_group_size(values.extents()), VK_NULL_HANDLE,
      values.image(output_barrier, api::PipelineStage::COMPUTE,
                   api::MemoryAccessType::WRITE),
      indices.image(output_barrier, api::PipelineStage::COMPUTE,
                    api::MemoryAccessType::WRITE),
      current->buffer(output_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::READ),
      params.buffer());
  return {convert(values), convert(indices)};
}

} // namespace

// The ranks a walk keeps live in the signed word the walk writes them in.  A
// consumer that binds those positions a word at a time — the sampler, which
// reads an ordered row straight out of the walk — takes this pair, while every
// answer handed to a caller names its lane on the wide count.
std::tuple<Tensor, Tensor> sorted_positions(
    const Tensor& self, int64_t dim, bool descending, int64_t count) {
  return sort_impl(self, dim, descending, count);
}

Tensor median_kernel(const Tensor& self) {
  TP_CHECK(self.numel() > 0, "Vulkan median requires a non-empty tensor");
  const Tensor flat = reshape_kernel(self, {self.numel()});
  const auto selected = sort_impl(flat, 0, false, 1, (self.numel() - 1) / 2);
  return reshape_kernel(std::get<0>(selected), {});
}

std::tuple<Tensor, Tensor> sort_kernel(
    const Tensor& self, int64_t dim, bool descending) {
  const auto ordered = sort_impl(self, dim, descending, -1);
  return {std::get<0>(ordered), std::get<1>(ordered).to(DType::Int64)};
}

// The permutation alone: the same selection, with the ordered values dropped
// rather than written out a second time.
Tensor argsort_kernel(
    const Tensor& self, int64_t dim, bool descending) {
  return std::get<1>(sort_impl(self, dim, descending, -1)).to(DType::Int64);
}

// The k-th smallest entry of a span and the lane that holds it.  The ordered
// walk keeps the earlier lane of a tie and parks a lane that is not a number
// past every ordered one, so the answer is the single entry of a one-wide
// window of that walk.  The order counts from one; the kept lane is numbered
// from zero and reported on the wide count.
std::tuple<Tensor, Tensor> kthvalue_kernel(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan kthvalue supports Float32 and Float16 payloads only");
  const int64_t rank = self.dim();
  if (rank == 0) {
    // A lone entry has one axis to name and one rank to hold.
    if (dim != 0 && dim != -1) {
      TP_THROW(
          IndexError, "Dimension out of range for scalar kthvalue input: ",
          dim);
    }
    if (k != 1) {
      TP_THROW(
          RuntimeError,
          "kthvalue(): selected number k out of range for dim 0");
    }
    Tensor indices =
        full_kernel({}, Scalar(0), DType::Int64, self.device(), false);
    return {clone_kernel(self), std::move(indices)};
  }
  const int64_t axis = dim < 0 ? dim + rank : dim;
  if (axis < 0 || axis >= rank) {
    TP_THROW(
        IndexError, "Dimension out of range (expected to be in range of [",
        -rank, ", ", rank - 1, "], but got ", dim, ")");
  }
  const int64_t length = self.size(axis);
  if (k < 1 || k > length) {
    TP_THROW(
        RuntimeError,
        "kthvalue(): selected number k out of range for dim ", axis);
  }
  const auto window = sort_impl(self, axis, false, 1, k - 1);
  Tensor values = std::get<0>(window);
  Tensor indices = std::get<1>(window).to(DType::Int64);
  if (!keepdim) {
    std::vector<int64_t> kept;
    kept.reserve(static_cast<size_t>(rank - 1));
    for (int64_t d = 0; d < rank; ++d) {
      if (d != axis) kept.push_back(self.size(d));
    }
    values = reshape_kernel(values, kept);
    indices = reshape_kernel(indices, kept);
  }
  return {std::move(values), std::move(indices)};
}

std::tuple<Tensor, Tensor> topk_kernel(
    const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted,
    int64_t impl) {
  (void)sorted; // Sorted output also satisfies the unordered top-k contract.
  (void)impl;   // Algorithm selection hint; one implementation here.
  TP_CHECK(k >= 0, "Vulkan topk: k must be non-negative");
  const auto ordered = sort_impl(self, dim, largest, k);
  return {std::get<0>(ordered), std::get<1>(ordered).to(DType::Int64)};
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, SelectionKernels) {
  m.impl("median", &tensorplay::vulkan::ops::median_kernel);
  m.impl("sort", &tensorplay::vulkan::ops::sort_kernel);
  m.impl("argsort", &tensorplay::vulkan::ops::argsort_kernel);
  m.impl("topk", &tensorplay::vulkan::ops::topk_kernel);
  m.impl("kthvalue", &tensorplay::vulkan::ops::kthvalue_kernel);
}

#endif /* USE_VULKAN */
