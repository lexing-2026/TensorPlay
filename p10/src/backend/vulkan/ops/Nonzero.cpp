#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <Utils.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer, and
// shares a payload that is already dense as it stands.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

// Mask words one invocation owns.  A long walk spends several of them per
// invocation to keep the number of blocks - and with it the host transfer of
// the per-block totals - small; a short one gives every word its own invocation
// so the device has enough work in flight.
constexpr int32_t kLongMaskWordsPerLane = 4;
constexpr int32_t kShortMaskWordsPerLane = 1;
constexpr int32_t kLongMaskWords = 16384;
constexpr int32_t kLaneCount = 256;

// A launch extent counts invocations and the driver divides it by the width of a
// workgroup, so a walk of `blocks` groups asks for a whole group each.
api::utils::uvec3 walk_extent(int32_t blocks) {
  return {
      static_cast<uint32_t>(blocks) * static_cast<uint32_t>(kLaneCount),
      1u,
      1u};
}

struct MaskMarkBlock final {
  int32_t elems;
  int32_t words;
  int32_t lane_words;
  int32_t fill;
};

struct MaskCompactBlock final {
  int32_t elems;
  int32_t words;
  int32_t lane_words;
  int32_t ndim;
  ivec4 sizes;
};

// Everything both walks of one mask need: the payload they read, how its
// elements are spread over blocks, where each block starts in the result, and
// how many elements the mask selects.
struct MaskWalk final {
  // Held because the memory the walks read belongs to this payload: a mask
  // made on the device (a comparison, say) hands its buffer back to the pool
  // as soon as the last handle to it goes, which would let a later allocation
  // overwrite the mask between the two walks.
  Tensor flags;
  api::vTensor mask;
  std::vector<int32_t> bases;
  int32_t elems = 0;
  int32_t words = 0;
  int32_t lane_words = kShortMaskWordsPerLane;
  int32_t blocks = 1;
  int64_t count = 0;
};

void validate_flag_source(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Bool || t.dtype() == DType::Float32 ||
          t.dtype() == DType::Float16 || t.dtype() == DType::Int32 ||
          t.dtype() == DType::Int8 || t.dtype() == DType::UInt8,
      std::string("Vulkan ") + name +
          " tests a mask of Bool, Float32, Float16, Int32, Int8 or UInt8 "
          "values; a wide count is not tested on the device");
}

// The one-vote-per-element form the walks read.  A Bool payload already holds
// it, so it is taken as it stands; any other payload is asked whether it holds a
// value, which is the test the compaction makes on it.
Tensor flag_payload(const Tensor& self) {
  if (self.dtype() == DType::Bool) {
    return self.contiguous();
  }
  return tpx::ops::ne(self, Scalar(0));
}

// Reads the mask once and derives the offsets each block starts at.  The totals
// come to the host because the length of the result is only known after the
// whole mask has been read - the caller has to be told it too, to size the
// answer - and the offsets go back rather than have every block re-read the
// totals of the blocks above it.
MaskWalk mark_mask(
    api::Context* const context,
    const Tensor& flags,
    int64_t numel) {
  MaskWalk walk;
  walk.flags = flags;
  walk.mask = to_buffer_storage_dispatch(context, convert(flags.contiguous()));
  TP_CHECK(
      numel <= std::numeric_limits<int32_t>::max(),
      "Vulkan mask walks support payloads up to a word long");
  walk.elems = static_cast<int32_t>(numel);
  walk.words = static_cast<int32_t>((numel + 3) / 4);
  walk.lane_words = walk.words >= kLongMaskWords
      ? kLongMaskWordsPerLane
      : kShortMaskWordsPerLane;
  walk.blocks = static_cast<int32_t>(api::utils::div_up(
      static_cast<uint32_t>(walk.words),
      static_cast<uint32_t>(kLaneCount * walk.lane_words)));
  // The walk is one workgroup axis deep, so the block count has to stay inside
  // the per-axis limit the smallest tilers give a compute dispatch.
  TP_CHECK(
      walk.blocks <= 65535,
      "Vulkan mask walks stop above 268 million elements: the block count "
      "would leave the workgroup limit a mobile tiler sets");

  api::vTensor v_agg{
      context,
      {static_cast<int64_t>(walk.blocks)},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const struct MaskMarkBlock block{
      walk.elems,
      walk.words,
      walk.lane_words,
      0,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("mask_mark"),
      pipeline_barrier,
      walk_extent(walk.blocks),
      {static_cast<uint32_t>(kLaneCount), 1u, 1u},
      VK_NULL_HANDLE,
      v_agg.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      walk.mask.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());

  const Tensor totals_host = convert(v_agg).to(Device(DeviceType::CPU));
  const int32_t* totals = totals_host.data_ptr<int32_t>();
  walk.bases.resize(static_cast<size_t>(walk.blocks));
  int64_t seen = 0;
  for (int32_t block_index = 0; block_index < walk.blocks; ++block_index) {
    walk.bases[static_cast<size_t>(block_index)] = static_cast<int32_t>(seen);
    seen += totals[static_cast<size_t>(block_index)];
  }
  walk.count = seen;
  return walk;
}

// Writes the selected coordinates, one row of `sizes.size()` words per
// selection.  The rank is deliberately part of the walk rather than a reshape
// of a flat result: a payload of several axes and a flat one of the same length
// compact to the same rows, only numbered differently.
api::vTensor compact_positions(
    api::Context* const context,
    const MaskWalk& walk,
    const std::vector<int64_t>& sizes) {
  const int32_t ndim = static_cast<int32_t>(sizes.size());
  TP_CHECK(
      walk.count * std::max<int64_t>(1, ndim) <=
          std::numeric_limits<int32_t>::max(),
      "Vulkan mask walks hold fewer than a word of coordinates");

  api::vTensor v_bases{
      context,
      {static_cast<int64_t>(walk.blocks)},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  utils::upload_host_bytes(
      v_bases,
      walk.bases.data(),
      walk.bases.size() * sizeof(int32_t));

  api::vTensor v_out{
      context,
      {walk.count, static_cast<int64_t>(ndim)},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  ivec4 padded{1, 1, 1, 1};
  for (int32_t axis = 0; axis < ndim; ++axis) {
    padded[axis] = static_cast<int32_t>(sizes[static_cast<size_t>(axis)]);
  }
  const struct MaskCompactBlock block{
      walk.elems,
      walk.words,
      walk.lane_words,
      ndim,
      padded,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  api::vTensor v_mask = walk.mask;
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("mask_compact"),
      pipeline_barrier,
      walk_extent(walk.blocks),
      {static_cast<uint32_t>(kLaneCount), 1u, 1u},
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_mask.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      v_bases.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());
  return v_out;
}

} // namespace

Tensor nonzero_kernel(const Tensor& self) {
  const int64_t ndim = self.dim();
  TP_CHECK(
      ndim <= 4,
      "Vulkan nonzero names the axes of up to 4d tensors");
  validate_flag_source(self, "nonzero");

  const std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const int64_t numel = self.numel();
  const Device device = self.device();

  if (numel == 0) {
    return tpx::ops::empty(
        std::vector<int64_t>{0, ndim}, DType::Int64, device);
  }

  const MaskWalk walk = mark_mask(
      api::context(), flag_payload(self), numel);
  // A payload without axes has no coordinate to name, and one that selects
  // nothing has no row: either way the count is the whole answer.
  if (ndim == 0 || walk.count == 0) {
    return tpx::ops::empty(
        std::vector<int64_t>{walk.count, ndim}, DType::Int64, device);
  }

  api::vTensor v_positions = compact_positions(
      api::context(), walk, sizes);
  return convert(v_positions).to(DType::Int64);
}

Tensor masked_select_kernel(const Tensor& self, const Tensor& mask) {
  validate_flag_source(mask, "masked_select");
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan masked_select takes Float32 and Float16 payloads");

  const std::vector<int64_t> self_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const std::vector<int64_t> mask_sizes =
      static_cast<std::vector<int64_t>>(mask.shape());
  // The answer is flat, so the pair only has to line up element by element;
  // broadcasting settles how often each of them is read.
  const std::vector<int64_t> out_sizes =
      tensorplay::broadcast_shapes(self_sizes, mask_sizes);
  TP_CHECK(
      out_sizes.size() <= 4,
      "Vulkan masked_select supports up to 4d tensors");

  Tensor picked = self;
  Tensor flags = flag_payload(mask);
  int64_t numel = 1;
  for (const int64_t size : out_sizes) {
    numel *= size;
  }
  if (picked.sizes() != out_sizes) {
    picked = tpx::ops::expand(picked, out_sizes, false).contiguous();
  }
  if (flags.sizes() != out_sizes) {
    flags = tpx::ops::expand(flags, out_sizes, false).contiguous();
  }

  const Device device = self.device();
  if (numel == 0) {
    return tpx::ops::empty(
        std::vector<int64_t>{0}, self.dtype(), device);
  }

  const MaskWalk walk = mark_mask(api::context(), flags, numel);
  if (walk.count == 0) {
    return tpx::ops::empty(
        std::vector<int64_t>{0}, self.dtype(), device);
  }

  // The selected elements sit on one flat axis, so the coordinate the walk
  // writes is the element number itself, and the rows read it as a list of
  // codes.
  api::vTensor v_positions = compact_positions(
      api::context(), walk, std::vector<int64_t>{numel});
  const Tensor codes =
      convert(v_positions).reshape({static_cast<int64_t>(walk.count)});
  return tpx::ops::index_select(
      picked.reshape({numel}), 0, codes);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MaskCompactionKernels) {
  m.impl("nonzero", &tensorplay::vulkan::ops::nonzero_kernel);
  m.impl("masked_select", &tensorplay::vulkan::ops::masked_select_kernel);
}

#endif /* USE_VULKAN */
