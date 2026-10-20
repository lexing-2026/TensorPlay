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
  TP_CHECK(self.dtype() == DType::Float32,
           "Vulkan sort supports Float32 tensors only");
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
  const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
  auto out_sizes = sizes;
  out_sizes[dim] = count;
  api::Context* context = api::context();
  api::vTensor input = convert(self);
  api::vTensor values{context, out_sizes, DType::Float32};
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
  const bool buffer_input =
      input.storage_type() == api::StorageType::BUFFER;
  if (buffer_input) {
    context->submit_compute_job(
        VK_KERNEL(sort_buffer), barrier,
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
    context->submit_compute_job(
        VK_KERNEL(sort_output_buffer), output_barrier,
        {static_cast<uint32_t>(values.numel()), 1u, 1u},
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
      VK_KERNEL(sort_output), output_barrier, values.extents(),
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

Tensor median_kernel(const Tensor& self) {
  TP_CHECK(self.numel() > 0, "Vulkan median requires a non-empty tensor");
  const Tensor flat = reshape_kernel(self, {self.numel()});
  const auto selected = sort_impl(flat, 0, false, 1, (self.numel() - 1) / 2);
  return reshape_kernel(std::get<0>(selected), {});
}

std::tuple<Tensor, Tensor> sort_kernel(
    const Tensor& self, int64_t dim, bool descending) {
  return sort_impl(self, dim, descending, -1);
}

// The permutation alone: the same selection, with the ordered values dropped
// rather than written out a second time.
Tensor argsort_kernel(
    const Tensor& self, int64_t dim, bool descending) {
  return std::get<1>(sort_impl(self, dim, descending, -1));
}

std::tuple<Tensor, Tensor> topk_kernel(
    const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted,
    int64_t impl) {
  (void)sorted; // Sorted output also satisfies the unordered top-k contract.
  (void)impl;   // Algorithm selection hint; one implementation here.
  TP_CHECK(k >= 0, "Vulkan topk: k must be non-negative");
  return sort_impl(self, dim, largest, k);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, SelectionKernels) {
  m.impl("median", &tensorplay::vulkan::ops::median_kernel);
  m.impl("sort", &tensorplay::vulkan::ops::sort_kernel);
  m.impl("argsort", &tensorplay::vulkan::ops::argsort_kernel);
  m.impl("topk", &tensorplay::vulkan::ops::topk_kernel);
}

#endif /* USE_VULKAN */
