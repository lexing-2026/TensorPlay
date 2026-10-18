#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <Utils.h>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Copies a texture-stored payload into a dense buffer vTensor; sibling
// kernels use it to bring every operand of a linear path into buffer
// storage (see BinaryOp.cpp).
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

void validate_float_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

struct EmbeddingBlock final {
  int rows;
  int features;
  int weight_rows;
  int fill;
};

struct IndexSelectBlock final {
  int inner;
  int row_stride;
  int count;
  int fill;
};

// Uploads the index payload: the Int64 codes narrow to Int32 on the host
// (the backend's texture vocabulary has no 8-byte format) and ride a flat
// 1-D Int32 texture with one index per texel.
api::vTensor upload_index_payload(const Tensor& indices) {
  api::Context* const context = api::context();
  Tensor flat_cpu =
      indices.to(Device(DeviceType::CPU))
          .reshape({indices.numel()})
          .to(DType::Int32)
          .contiguous();

  api::vTensor v{context, {flat_cpu.numel()}, DType::Int32};
  Tensor flat_packed = utils::nchw_to_nc4hw(flat_cpu);
  utils::upload_host_bytes(
      v, flat_packed.impl()->storage().data(),
      flat_packed.numel() * flat_packed.itemsize());
  return v;
}

} // namespace

namespace {

// Leaves autograd off for the duration of an in-place row mutation; a local
// copy of the completion-kernel guard, which the vulkan side has not needed
// until the scatter family.
struct NoGradGuard final {
  const bool previous;
  NoGradGuard() : previous(GradMode::is_enabled()) {
    GradMode::set_enabled(false);
  }
  ~NoGradGuard() { GradMode::set_enabled(previous); }
};

} // namespace

Tensor embedding_kernel(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  (void)padding_idx;
  (void)scale_grad_by_freq;
  (void)sparse;
  TP_CHECK(
      weight.dtype() == DType::Float32,
      "Vulkan embedding supports a Float32 weight only");
  TP_CHECK(
      indices.dtype() == DType::Int64 || indices.dtype() == DType::Int32,
      "Vulkan embedding: indices must be Int64 or Int32");
  TP_CHECK(
      weight.dim() == 2,
      "Vulkan embedding expects a 2d [rows, features] weight");

  api::Context* const context = api::context();

  api::vTensor v_weight = convert(weight);
  api::vTensor v_indices = upload_index_payload(indices);

  const int64_t index_count = indices.numel();
  const int64_t features = weight.size(1);

  // A buffer-stored weight (large vocabulary table) is looked up through a
  // linear path: the index payload is copied to a dense Int32 buffer and the
  // output is forced to buffer storage, so the same row-major shader serves
  // small and large lookups without a texture fallback.
  if (v_weight.storage_type() == api::StorageType::BUFFER) {
    api::vTensor v_indices_buf =
        to_buffer_storage_dispatch(context, v_indices);
    api::vTensor v_output{
        context, {index_count, features}, DType::Float32,
        api::StorageType::BUFFER, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const uint32_t nf4 =
        static_cast<uint32_t>((features + 3) / 4);
    const uint32_t n = nf4 * static_cast<uint32_t>(index_count);
    const struct EmbeddingBufferBlock {
      int32_t rows;
      int32_t features;
      int32_t weight_rows;
      int32_t fill;
    } block{
        static_cast<int32_t>(index_count),
        static_cast<int32_t>(features),
        static_cast<int32_t>(weight.size(0)),
        0};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR("embedding_buffer"), pipeline_barrier,
        {n, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_weight.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_indices_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    Tensor out = convert(v_output);
    if (indices.dim() != 1) {
      std::vector<int64_t> out_sizes =
          static_cast<std::vector<int64_t>>(indices.shape());
      out_sizes.push_back(features);
      out = out.reshape(out_sizes);
    }
    return out;
  }

  // Output {index_count, features}: the 2-D texture geometry puts feature
  // columns on x and the lookup row on y.  Non-1d index tensors are flattened
  // for the lookup and reshaped back to `index_shape + (features,)`.
  api::vTensor v_output{context, {index_count, features}, DType::Float32};

  const struct EmbeddingBlock block{
      static_cast<int32_t>(index_count),
      static_cast<int32_t>(features),
      static_cast<int32_t>(weight.size(0)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(embedding), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_indices.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor out = convert(v_output);
  if (indices.dim() != 1) {
    std::vector<int64_t> out_sizes = static_cast<std::vector<int64_t>>(indices.shape());
    out_sizes.push_back(features);
    out = out.reshape(out_sizes);
  }
  return out;
}

//
// In-place index_copy_: role as the vulkan write path for row updates along
// the sequence axis, e.g. appending decode steps to a preallocated kv cache.
// The destination rows arrive through an Int32 payload that the shader walks;
// the host folds an Int64 payload, validates the range, and uploads it.
//
// Geometry note: a 4-D channel-packed [N, C, S, W] tensor keeps its rows on
// the texture's y extent, so a dim == 2 scatter is texel aligned from row to
// row whatever the channel lanes carry.  Other ranks and axes need a
// channel-aware tile map and stay composite-side.
Tensor& index_copy_rows_inplace(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  NoGradGuard no_grad;
  TP_CHECK(
      self.device().is_vulkan() && source.device().is_vulkan(),
      "Vulkan index_copy_: both tensors must be on the device");
  TP_CHECK(
      self.dim() == 4 && source.dim() == 4,
      "Vulkan index_copy_: 4-D tensors only");
  TP_CHECK(
      dim == 2 || dim == -2,
      "Vulkan index_copy_: scatter runs along the sequence axis (dim 2)");
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan index_copy_: Float32 and Float16 only");
  TP_CHECK(
      source.dtype() == self.dtype(),
      "Vulkan index_copy_: source dtype must match the destination");
  TP_CHECK(
      index.dtype() == DType::Int64 || index.dtype() == DType::Int32,
      "Vulkan index_copy_: index must be Int64 or Int32");

  if (self.numel() == 0 || index.numel() == 0) {
    return self;
  }

  const int64_t rows = self.size(2);
  const int64_t n_idx = index.numel();
  TP_CHECK(
      source.size(2) == n_idx,
      "Vulkan index_copy_: source rows must match the index length");
  TP_CHECK(
      self.size(0) == source.size(0) && self.size(1) == source.size(1) &&
          self.size(3) == source.size(3),
      "Vulkan index_copy_: source must match off-axis sizes");

  Tensor idx_host = index.to(Device(DeviceType::CPU)).to(DType::Int64).contiguous();
  const int64_t* ivals = idx_host.data_ptr<int64_t>();
  for (int64_t i = 0; i < n_idx; ++i) {
    TP_CHECK(
        ivals[i] >= 0 && ivals[i] < rows,
        "Vulkan index_copy_: index out of range");
  }

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_source = convert(source);
  api::vTensor v_indices = upload_index_payload(idx_host);

  const struct IndexCopyBlock final {
    ivec4 extents; // (texel width, destination rows, texel planes, unused)
    int n_idx;
    int fill;
    int fill2;
    int fill3;
  } block{
      ivec4(
          static_cast<int32_t>(v_self.extents()[0u]),
          static_cast<int32_t>(v_self.extents()[1u]),
          static_cast<int32_t>(v_self.extents()[2u]),
          0),
      static_cast<int32_t>(n_idx),
      0,
      0,
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 grid{
      v_self.extents()[0u],
      safe_downcast_to_u32(n_idx),
      v_self.extents()[2u],
  };
  context->submit_compute_job(
      kernel_for("index_copy", self.dtype()),
      pipeline_barrier,
      grid,
      adaptive_work_group_size(grid),
      VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_source.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_indices.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return self;
}

Tensor& index_copy__kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  return index_copy_rows_inplace(self, dim, index, source);
}

Tensor index_copy_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  Tensor out = self.clone();
  index_copy_rows_inplace(out, dim, index, source);
  return out;
}

Tensor index_select_kernel(
    const Tensor& self, int64_t dim, const Tensor& index) {
  validate_float_operand(self, "index_select");
  TP_CHECK(
      index.dim() == 1,
      "Vulkan index_select: index should be a vector");
  const int64_t wrapped = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < self.dim(),
      "Vulkan index_select: dimension out of range");
  // The gather kernel walks a flattened payload with one outer block, so
  // the selected axis must be the only axis left of the inner span.
  const int64_t inner =
      wrapped < self.dim() - 1 ? self.size(wrapped + 1) : 1;
  const int64_t outer = self.numel() / (self.size(wrapped) * inner);
  TP_CHECK(
      outer == 1,
      "Vulkan index_select supports a single outer block (1-d tensors and "
      "row selections of 2-d tensors)");

  api::Context* const context = api::context();

  // Flatten the dense payload into a 1-D texture.
  Tensor flat = self.reshape({self.numel()});
  api::vTensor v_input = convert(flat);
  api::vTensor v_indices = upload_index_payload(index);

  const int64_t index_count = index.numel();
  const int64_t out_count = index_count * inner;

  api::vTensor v_output{context, {out_count}, DType::Float32};

  const struct IndexSelectBlock block{
      static_cast<int32_t>(inner),
      static_cast<int32_t>(inner),
      static_cast<int32_t>(index_count),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(index_select), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_indices.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  // Rebuild the logical shape through the materializing reshape.
  std::vector<int64_t> out_sizes(
      static_cast<std::vector<int64_t>>(self.shape()));
  out_sizes[static_cast<size_t>(wrapped)] = index_count;
  Tensor flat_out = convert(v_output);
  return flat_out.reshape(out_sizes);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, IndexingKernels) {
  m.impl("embedding", &tensorplay::vulkan::ops::embedding_kernel);
  m.impl("index_select", &tensorplay::vulkan::ops::index_select_kernel);
  m.impl("index_copy_", &tensorplay::vulkan::ops::index_copy__kernel);
  m.impl("index_copy", &tensorplay::vulkan::ops::index_copy_kernel);
}

#endif /* USE_VULKAN */
