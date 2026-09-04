#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <Utils.h>

namespace tensorplay {
namespace vulkan {
namespace ops {

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
      indices.dim() == 1,
      "Vulkan embedding supports a 1d index payload");
  TP_CHECK(
      weight.dim() == 2,
      "Vulkan embedding expects a 2d [rows, features] weight");

  api::Context* const context = api::context();

  api::vTensor v_weight = convert(weight);
  api::vTensor v_indices = upload_index_payload(indices);

  const int64_t index_count = indices.numel();
  const int64_t features = weight.size(1);

  // Output {index_count, features}: the 2-D texture geometry puts feature
  // columns on x and the lookup row on y.
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

  return convert(v_output);
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
}

#endif /* USE_VULKAN */
