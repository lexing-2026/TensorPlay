#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"

#include "Tensor.h"
#include "TensorImpl.h"
#include "SizesAndStrides.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

//
// View semantics for Vulkan tensors.
//
// The backend owns GPU resources only; sizes/strides metadata stays on the
// ordinary TensorImpl.  A strided view (select / slice / as_strided / expand /
// reshape-fallback) of a Vulkan tensor would produce a TensorImpl whose
// strides no longer describe the texture payload: reading such a tensor back
// through the staging pipeline walks storage order, silently returning
// garbage.  Instead, every view op on a Vulkan tensor materializes its output
// on the GPU: a dense payload is allocated and one gather dispatch copies the
// viewed elements.  The result is an owning dense tensor that stays correct
// under copy-out, further views, autograd, and batching.
//
// The host-side pieces of this file only reshape metadata; all element
// movement happens in the view_gather shader.
//

namespace {

// Prepads sizes/strides to four slots in {N, C, H, W} order (leading 1s).
std::vector<int64_t> prepadded(const std::vector<int64_t>& v) {
  std::vector<int64_t> out(4, 1);
  const int64_t offset = 4 - static_cast<int64_t>(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[static_cast<size_t>(offset + static_cast<int64_t>(i))] = v[i];
  }
  return out;
}

// Launches one view_gather dispatch: the input is read densely, the output
// is produced with the view's element mapping expressed as strides over the
// logical (row-major) element order.
Tensor gather_view(
    const Tensor& input,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides_rowmajor,
    int64_t offset) {
  TP_CHECK(
      input.dtype() == DType::Float32 || input.dtype() == DType::Int8,
      "Vulkan view materialization supports Float32 and Int8 tensors only");
  const bool is_int8 = input.dtype() == DType::Int8;

  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  api::vTensor v_output{context, out_sizes, input.dtype()};

  const std::vector<int64_t> in_sizes = prepadded(
      std::vector<int64_t>(v_input.sizes().begin(), v_input.sizes().end()));

  const auto in_dense = SizesAndStrides::compute_contiguous_strides(in_sizes);

  const std::vector<int64_t> out_padded = prepadded(out_sizes);
  const std::vector<int64_t> out_strides = prepadded(out_strides_rowmajor);

  const struct ViewGatherBlock final {
    ivec4 in_sizes;
    ivec4 in_strides;
    ivec4 out_sizes;
    ivec4 out_strides;
    int in_c_depth;
    int out_c_depth;
    int offset;
  } block{
      ivec4(
          static_cast<int32_t>(in_sizes[0]),
          static_cast<int32_t>(in_sizes[1]),
          static_cast<int32_t>(in_sizes[2]),
          static_cast<int32_t>(in_sizes[3])),
      ivec4(
          static_cast<int32_t>(in_dense[0]),
          static_cast<int32_t>(in_dense[1]),
          static_cast<int32_t>(in_dense[2]),
          static_cast<int32_t>(in_dense[3])),
      ivec4(
          static_cast<int32_t>(out_padded[0]),
          static_cast<int32_t>(out_padded[1]),
          static_cast<int32_t>(out_padded[2]),
          static_cast<int32_t>(out_padded[3])),
      ivec4(
          static_cast<int32_t>(out_strides[0]),
          static_cast<int32_t>(out_strides[1]),
          static_cast<int32_t>(out_strides[2]),
          static_cast<int32_t>(out_strides[3])),
      c_depth_of(in_sizes),
      c_depth_of(out_padded),
      static_cast<int32_t>(offset),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const api::ShaderInfo shader = is_int8
      ? VK_KERNEL(view_gather_int8)
      : VK_KERNEL(view_gather);

  context->submit_compute_job(
      shader, pipeline_barrier, v_output.extents(),
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

//
// Materializes a strided view of a Vulkan tensor into a dense payload.
// The view is described by its row-major strides over the logical element
// order plus a storage offset; any mapping expressible through as_strided
// is supported, including broadcast (zero-stride) expansions.
//
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset) {
  return gather_view(self, out_sizes, out_strides, storage_offset);
}

//
// view(): same stride inference rules as the CPU kernel; the inferred view
// strides feed the gather directly.
//
Tensor view_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
  const std::vector<int64_t> inferred =
      SizesAndStrides::infer_size(shape, self.numel());
  auto stride = SizesAndStrides::compute_view_strides(
      static_cast<std::vector<int64_t>>(self.shape()), self.strides(),
      inferred);
  TP_CHECK(
      stride.has_value(),
      "view size is not compatible with input tensor's size and stride");

  const int64_t offset =
      static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
  return materialize_view(self, inferred, *stride, offset);
}

//
// as_strided(): the general form; the caller's size/stride pair drives the
// gather.
//
Tensor as_strided_kernel(
    const Tensor& self,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    std::optional<int64_t> storage_offset) {
  TP_CHECK(
      size.size() == stride.size(),
      "as_strided(): sizes and strides must have the same length");
  for (const int64_t value : size) {
    TP_CHECK(value >= 0, "as_strided(): sizes must be non-negative");
  }
  const int64_t offset = storage_offset.value_or(
      static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset()));
  TP_CHECK(offset >= 0, "as_strided(): storage_offset must be non-negative");

  return materialize_view(self, size, stride, offset);
}

//
// expand(): zero strides on broadcast axes.
//
Tensor expand_kernel(
    const Tensor& self,
    const std::vector<int64_t>& size,
    bool /*implicit*/) {
  const int64_t ndim = self.dim();
  const int64_t new_ndim = static_cast<int64_t>(size.size());
  TP_CHECK(
      new_ndim >= ndim,
      "expand(): the number of sizes provided must be greater or equal to "
      "the number of dimensions in the tensor");

  const auto self_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const auto self_strides = self.strides();

  std::vector<int64_t> out_sizes(size);
  std::vector<int64_t> out_strides(new_ndim, 0);
  for (int64_t i = new_ndim - 1; i >= 0; --i) {
    const int64_t offset = new_ndim - 1 - i;
    const int64_t dim = ndim - 1 - offset;
    const int64_t src_size = dim >= 0 ? self_sizes[static_cast<size_t>(dim)] : 1;
    int64_t target = out_sizes[static_cast<size_t>(i)];
    if (target == -1) {
      TP_CHECK(
          dim >= 0,
          "expand(): the expanded size -1 is invalid in a leading, "
          "non-existing dimension");
      target = src_size;
      out_sizes[static_cast<size_t>(i)] = target;
    }
    if (src_size != target) {
      TP_CHECK(
          src_size == 1,
          "expand(): the expanded size must match the existing size at "
          "non-singleton dimensions");
    }
    out_strides[static_cast<size_t>(i)] =
        (dim >= 0 && src_size == target) ? self_strides[static_cast<size_t>(dim)] : 0;
  }

  const int64_t offset =
      static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
  return materialize_view(self, out_sizes, out_strides, offset);
}

//
// reshape(): view-compatible layouts re-gather directly; otherwise the input
// is first gathered into a dense copy and the final view applies.
//
Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape) {
  const std::vector<int64_t> inferred =
      SizesAndStrides::infer_size(shape, self.numel());
  auto stride = SizesAndStrides::compute_view_strides(
      static_cast<std::vector<int64_t>>(self.shape()), self.strides(),
      inferred);
  if (stride.has_value()) {
    const int64_t offset =
        static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
    return materialize_view(self, inferred, *stride, offset);
  }

  // Not viewable: materialize a dense copy in logical order first.
  const auto self_sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  const auto self_strides = self.strides();
  const int64_t offset =
      static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
  Tensor dense = materialize_view(self, self_sizes, self_strides, offset);
  return view_kernel(dense, inferred);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ViewKernels) {
  m.impl("as_strided", &tensorplay::vulkan::ops::as_strided_kernel);
  m.impl("view", &tensorplay::vulkan::ops::view_kernel);
  m.impl("expand", &tensorplay::vulkan::ops::expand_kernel);
  m.impl("reshape", &tensorplay::vulkan::ops::reshape_kernel);
}

#endif /* USE_VULKAN */
