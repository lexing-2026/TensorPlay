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
// Swapping the two axes of a dense operand is the exception, because its
// elements are a relabeling of a payload that is already dense: the result
// carries the strides and aliases the storage, and readers below reach the
// bytes through the twin or a gather.
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
    int64_t offset,
    std::optional<api::StorageType> destination = std::nullopt) {
  TP_CHECK(
      input.dtype() == DType::Float32 || input.dtype() == DType::Float16 ||
          input.dtype() == DType::Int8 || input.dtype() == DType::QInt8 ||
          input.dtype() == DType::Int32 || input.dtype() == DType::Int64 ||
          input.dtype() == DType::Bool,
      "Vulkan view materialization supports Float32, Float16, Int8, QInt8, "
      "Int32, Int64 and Bool tensors only");
  // A byte names one code, whether it arrives as a plain count or as the
  // signed payload a quantized weight keeps.
  const bool is_int8 = input.dtype() == DType::Int8 ||
      input.dtype() == DType::QInt8 || input.dtype() == DType::Bool;
  const bool is_int32 = input.dtype() == DType::Int32;
  const bool is_half = input.dtype() == DType::Float16;
  // A code wider than a word has no texel format to name it, so both sides of
  // its gather are linear and one element is a pair of words.
  const bool is_wide = input.dtype() == DType::Int64;

  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  // The destination class: whatever a caller named, and otherwise the shape
  // alone.  Every pairing of a texel or word source with a texel or word
  // destination has its own gather below, so naming the class asks for one of
  // them rather than leaving the choice to the layout.
  api::vTensor v_output = destination.has_value()
      ? allocate_output(
            context, *destination, out_sizes, input.dtype())
      : api::vTensor{context, out_sizes, input.dtype()};

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

  // An output whose extents exceed the 3-D image axis limit falls back to
  // buffer storage; the gather then writes plain row-major elements through
  // a buffer-binding variant instead of an image store.
  if (v_output.storage_type() == api::StorageType::BUFFER) {
    // How many elements of the destination share one 32-bit word: a half pair,
    // a byte of four codes.  A walk whose lanes share a word runs over words,
    // so the invocation that covers a word is the only one that writes it.
    const uint32_t elements_per_word =
        is_int8 ? 4u : (is_half ? 2u : 1u);
    const bool word_walk = elements_per_word > 1u;
    const api::utils::uvec3 word_grid{
        static_cast<uint32_t>(
            (multiply_integers(out_sizes) + elements_per_word - 1u) /
            elements_per_word),
        1u,
        1u};
    const api::utils::uvec3 grid{
        static_cast<uint32_t>(out_padded[3]),
        static_cast<uint32_t>(out_padded[2]),
        static_cast<uint32_t>(out_padded[0]) *
            static_cast<uint32_t>(c_depth_of(out_padded)),
    };
    // A buffer-stored input is already plain row-major memory, so the
    // gather reads through the linear binding and writes the dense output
    // in one pass.
    if (v_input.storage_type() == api::StorageType::BUFFER) {
      const api::ShaderInfo shader = is_wide
          ? VK_KERNEL(view_gather_i64_buffer_to_buffer)
          : is_int32
          ? VK_KERNEL(view_gather_int32_buffer_to_buffer)
          : is_int8
          ? VK_KERNEL(view_gather_int8_buffer_to_buffer)
          : is_half
          ? VK_KERNEL_FROM_STR("view_gather_buffer_to_buffer_half")
          : kernel_for("view_gather_buffer_to_buffer", input.dtype());
      const api::utils::uvec3 launch = word_walk ? word_grid : grid;
      context->submit_compute_job(
          shader, pipeline_barrier, launch,
          word_walk ? api::utils::uvec3{256u, 1u, 1u}
                    : adaptive_work_group_size(launch),
          VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_input.buffer(
              pipeline_barrier, api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ),
          params.buffer());
      return convert(v_output);
    }
    const api::ShaderInfo shader = is_int32
        ? VK_KERNEL(view_gather_int32_buffer)
        : is_int8
        ? VK_KERNEL(view_gather_int8_buffer)
        : is_half
        ? VK_KERNEL_FROM_STR("view_gather_buffer_half")
        : kernel_for("view_gather_buffer", input.dtype());
    const api::utils::uvec3 launch = word_walk ? word_grid : grid;
    context->submit_compute_job(
        shader, pipeline_barrier, launch,
        word_walk ? api::utils::uvec3{256u, 1u, 1u}
                  : adaptive_work_group_size(launch),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  // A buffer-stored input is already plain row-major memory; the gather
  // reads through the linear binding and writes the dense output in one
  // pass, for both buffer and image destinations.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    TP_CHECK(
        input.dtype() == DType::Float32 || input.dtype() == DType::Int32 ||
            input.dtype() == DType::Float16 || is_int8,
        "Vulkan view materialization from buffer storage supports "
        "Float32, Int32, Float16 and one-byte inputs only");
    const api::ShaderInfo shader = is_int32
        ? VK_KERNEL(view_gather_int32_buffer_to_texture)
        : is_int8
        ? VK_KERNEL(view_gather_int8_buffer_to_texture)
        : is_half
        ? VK_KERNEL_FROM_STR("view_gather_buffer_to_texture_half")
        : kernel_for("view_gather_buffer_to_texture", input.dtype());
    context->submit_compute_job(
        shader, pipeline_barrier, v_output.extents(),
        adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(
            pipeline_barrier, api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_output);
  }

  const api::ShaderInfo shader = is_int8
      ? VK_KERNEL(view_gather_int8)
      : is_int32 ? VK_KERNEL(view_gather_int32)
                 : kernel_for("view_gather", input.dtype());

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
    int64_t storage_offset,
    std::optional<api::StorageType> destination = std::nullopt) {
  // The gather names lanes by their position in the storage, so an axis-swapped
  // operand is read through the twin that describes that storage: the strides a
  // caller computed from the view index the same bytes either way, and nothing
  // has to be laid out again before the walk.
  Tensor twin;
  if (transpose_view_of_dense(self, &twin)) {
    return gather_view(
        twin, out_sizes, out_strides, storage_offset, destination);
  }
  return gather_view(
      self, out_sizes, out_strides, storage_offset, destination);
}

Tensor to_image_storage(api::Context* const context, const Tensor& self) {
  api::vTensor v_self = convert(self);
  if (v_self.storage_type() == api::StorageType::TEXTURE_3D) {
    return self;
  }
  const std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(self.shape());
  TP_CHECK(
      image_storage_supported(context, sizes),
      "Vulkan: this shape has no image form, so a kernel that samples texels "
      "cannot be handed this payload");
  return materialize_view(
      self,
      sizes,
      SizesAndStrides::compute_contiguous_strides(sizes),
      0,
      api::StorageType::TEXTURE_3D);
}

bool transpose_view_of_dense(const Tensor& t, Tensor* dense_out) {
  if (t.dim() != 2 || t.is_contiguous() ||
      t.unsafeGetTensorImpl()->storage_offset() != 0) {
    return false;
  }
  const int64_t r = t.size(0);
  const int64_t c = t.size(1);
  if (t.stride(0) == 1 && t.stride(1) == r) {
    Tensor twin(
        t.unsafeGetTensorImpl()->storage(),
        std::vector<int64_t>{c, r},
        std::vector<int64_t>{r, 1},
        t.dtype(),
        /*storage_offset=*/0);
    twin.unsafeGetTensorImpl()->share_version_counter(
        *t.unsafeGetTensorImpl());
    *dense_out = std::move(twin);
    return true;
  }
  return false;
}

Tensor dense_payload_of_view(const Tensor& t) {
  if (t.is_contiguous() &&
      t.unsafeGetTensorImpl()->storage_offset() == 0) {
    return t;
  }
  // shape() and strides() each return a fresh container, so the walk is
  // described by named copies rather than an iterator pair taken from two
  // temporaries.
  const std::vector<int64_t> sizes =
      static_cast<std::vector<int64_t>>(t.shape());
  const std::vector<int64_t> strides = t.strides();
  return materialize_view(
      t,
      sizes,
      strides,
      static_cast<int64_t>(t.unsafeGetTensorImpl()->storage_offset()));
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
    // Buffer payloads are plain row-major memory, so a stride-compatible
    // reshape of a contiguous buffer tensor is a pure metadata view: the
    // gather pipeline maps only texture-backed shapes.
    api::vTensor v_self = convert(self);
    if (v_self.storage_type() == api::StorageType::BUFFER) {
      const auto self_sizes = static_cast<std::vector<int64_t>>(self.shape());
      const auto dense_strides =
          SizesAndStrides::compute_contiguous_strides(self_sizes);
      if (self.is_contiguous() &&
          self.strides() ==
              std::vector<int64_t>(dense_strides.begin(), dense_strides.end())) {
        Tensor result(
            self.unsafeGetTensorImpl()->storage(), inferred, *stride,
            self.dtype(), offset);
        result.unsafeGetTensorImpl()->share_version_counter(
            *self.unsafeGetTensorImpl());
        return result;
      }
    }
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
