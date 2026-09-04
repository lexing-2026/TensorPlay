#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <optional>
#include <set>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

void validate_elementwise_4d(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

} // namespace

Tensor pow_scalar_kernel(const Tensor& self, Scalar exponent) {
  validate_elementwise_4d(self, "pow");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan pow requires texture storage");
  }

  api::vTensor v_output{context, v_self.sizes(), v_self.dtype()};

  const struct Block final {
    ivec4 extents;
    float other;
  } block{
      ivec4(
          v_self.extents()[0u],
          v_self.extents()[1u],
          v_self.extents()[2u],
          0),
      exponent.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(pow_tensor_scalar), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor pow_tensor_kernel(const Tensor& self, const Tensor& exponent) {
  validate_elementwise_4d(self, "pow");
  validate_elementwise_4d(exponent, "pow");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_other = convert(exponent);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan pow requires texture storage");
  }

  api::vTensor v_output{context, v_self.sizes(), v_self.dtype()};

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      1.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(pow), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end, Scalar weight) {
  validate_elementwise_4d(self, "lerp");
  TP_CHECK(
      end.numel() == 1 || end.shape() == self.shape(),
      "Vulkan lerp expects a scalar or same-shape end tensor");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_end = convert(end);

  api::vTensor v_output{context, v_self.sizes(), v_self.dtype()};

  const struct LerpBlock final {
    ivec4 extents;
    int scalar_end;
    int scalar_weight;
    float weight;
  } block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      end.numel() == 1 ? 1 : 0,
      0,
      weight.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(lerp), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor lerp_tensor_kernel(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight) {
  validate_elementwise_4d(self, "lerp");
  TP_CHECK(
      (end.numel() == 1 || end.shape() == self.shape()) &&
          (weight.numel() == 1 || weight.shape() == self.shape()),
      "Vulkan lerp.Tensor expects scalar or same-shape end and weight");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_end = convert(end);
  api::vTensor v_weight = convert(weight);

  api::vTensor v_output{context, v_self.sizes(), v_self.dtype()};

  const struct LerpBlock final {
    ivec4 extents;
    int scalar_end;
    int scalar_weight;
    float weight;
  } block{
      ivec4(
          v_output.extents()[0u],
          v_output.extents()[1u],
          v_output.extents()[2u],
          0),
      end.numel() == 1 ? 1 : 0,
      weight.numel() == 1 ? 1 : 0,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(lerp_tensor), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor& lerp_scalar_inplace_kernel(
    Tensor& self,
    const Tensor& end,
    Scalar weight) {
  validate_elementwise_4d(self, "lerp");
  TP_CHECK(
      end.numel() == 1 || end.shape() == self.shape(),
      "Vulkan lerp expects a scalar or same-shape end tensor");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_end = convert(end);

  const struct LerpBlock final {
    ivec4 extents;
    int scalar_end;
    int scalar_weight;
    float weight;
  } block{
      ivec4(
          v_self.extents()[0u],
          v_self.extents()[1u],
          v_self.extents()[2u],
          0),
      end.numel() == 1 ? 1 : 0,
      0,
      weight.to<float>(),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(lerpinplace), pipeline_barrier, v_self.extents(),
      adaptive_work_group_size(v_self.extents()), VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return self;
}

Tensor& lerp_tensor_inplace_kernel(
    Tensor& self,
    const Tensor& end,
    const Tensor& weight) {
  validate_elementwise_4d(self, "lerp");
  TP_CHECK(
      (end.numel() == 1 || end.shape() == self.shape()) &&
          (weight.numel() == 1 || weight.shape() == self.shape()),
      "Vulkan lerp.Tensor expects scalar or same-shape end and weight");
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_end = convert(end);
  api::vTensor v_weight = convert(weight);

  const struct LerpBlock final {
    ivec4 extents;
    int scalar_end;
    int scalar_weight;
    float weight;
  } block{
      ivec4(
          v_self.extents()[0u],
          v_self.extents()[1u],
          v_self.extents()[2u],
          0),
      end.numel() == 1 ? 1 : 0,
      weight.numel() == 1 ? 1 : 0,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(lerp_tensorinplace), pipeline_barrier, v_self.extents(),
      adaptive_work_group_size(v_self.extents()), VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return self;
}


namespace {

// Materializes a tensor's shape into a plain sizes vector.  The shape
// accessor returns a value object, so the copy must be taken once instead
// of reaching through repeated temporaries.
std::vector<int64_t> shape_vector(const Tensor& t) {
  const Size shape = t.shape();
  return std::vector<int64_t>(shape.begin(), shape.end());
}

} // namespace

Tensor flip_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
  validate_elementwise_4d(self, "flip");

  const int64_t ndim = self.dim();
  std::set<int64_t> unique;
  ivec4 flip_axes(0, 0, 0, 0);
  for (const int64_t d : dims) {
    int64_t axis = d < 0 ? d + ndim : d;
    TP_CHECK(
        axis >= 0 && axis < ndim, "Vulkan flip: dim out of range");
    TP_CHECK(unique.insert(axis).second, "Vulkan flip: repeated dim");
    // Innermost-first slot for the axis.
    const int slot = static_cast<int>(ndim - 1 - axis);
    flip_axes[slot] = 1;
  }

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_output{context, v_input.sizes(), v_input.dtype()};

  const struct FlipBlock final {
    ivec4 in_sizes;
    ivec4 flip_axes;
    int c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      flip_axes,
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(flip), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

namespace {

Tensor pad2d_impl(
    const Tensor& self,
    const std::vector<int64_t>& pad,
    bool replicate) {
  validate_elementwise_4d(self, "pad2d");
  TP_CHECK(
      self.dim() == 4 || self.dim() == 3,
      "Vulkan pad2d supports 3d and 4d tensors");
  TP_CHECK(
      pad.size() == 4,
      "Vulkan pad2d expects four paddings (left, right, top, bottom)");

  const int64_t W = self.size(self.dim() - 1);
  const int64_t H = self.size(self.dim() - 2);
  TP_CHECK(
      pad[0] >= 0 && pad[1] >= 0 && pad[0] < W && pad[1] < W && pad[2] >= 0 &&
          pad[3] >= 0 && pad[2] < H && pad[3] < H,
      "Vulkan pad2d: padding must be smaller than the corresponding extent");

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  std::vector<int64_t> new_sizes = shape_vector(self);
  new_sizes[new_sizes.size() - 1] += pad[0] + pad[1];
  new_sizes[new_sizes.size() - 2] += pad[2] + pad[3];

  api::vTensor v_output{context, new_sizes, self.dtype()};

  const struct Pad2DBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    ivec2 padding;
    int c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(v_output.sizes()),
      ivec2(
          static_cast<int32_t>(pad[0]),
          static_cast<int32_t>(pad[2])),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      replicate ? VK_KERNEL(replication_pad2d) : VK_KERNEL(reflection_pad2d),
      pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

} // namespace

Tensor reflection_pad_nd_kernel(
    const Tensor& self,
    const std::vector<int64_t>& pad) {
  return pad2d_impl(self, pad, /*replicate=*/false);
}

Tensor replication_pad_nd_kernel(
    const Tensor& self,
    const std::vector<int64_t>& pad) {
  return pad2d_impl(self, pad, /*replicate=*/true);
}

/*
 * Scalar extraction from a single-element tensor.  The read rides the
 * device-to-host copy path, which stages the payload out in logical order;
 * no unpack kernel of its own is needed.
 */
Scalar item_kernel(const Tensor& self) {
  TP_CHECK(
      self.numel() == 1, "item() only supported for 1-element tensors");
  const Tensor host = self.to(Device(DeviceType::CPU));
  return host.item();
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MiscKernels) {
  m.impl("pow.Tensor_Scalar", &tensorplay::vulkan::ops::pow_scalar_kernel);
  m.impl("pow.Tensor_Tensor", &tensorplay::vulkan::ops::pow_tensor_kernel);
  m.impl("item", &tensorplay::vulkan::ops::item_kernel);
  m.impl("lerp", &tensorplay::vulkan::ops::lerp_scalar_kernel);
  m.impl("lerp.Tensor", &tensorplay::vulkan::ops::lerp_tensor_kernel);
  m.impl("lerp_.Scalar", &tensorplay::vulkan::ops::lerp_scalar_inplace_kernel);
  m.impl("lerp_.Tensor", &tensorplay::vulkan::ops::lerp_tensor_inplace_kernel);
  m.impl("flip", &tensorplay::vulkan::ops::flip_kernel);
  m.impl("reflection_pad_nd", &tensorplay::vulkan::ops::reflection_pad_nd_kernel);
  m.impl("replication_pad_nd", &tensorplay::vulkan::ops::replication_pad_nd_kernel);
}

#endif /* USE_VULKAN */
