#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "SizesAndStrides.h"
#include "Utils.h"

#include <Utils.h>

#include <limits>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Copies a texture-stored payload into a dense buffer vTensor; sibling
// kernels use it to bring every operand of a linear path into buffer
// storage (see BinaryOp.cpp).
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

// Defined in View.cpp; materializes a strided view of a payload.  An identity
// mapping over a word-stored source releases an image copy of the same
// elements, which is how a lookup that had to stream its table still hands its
// rows to the kernels that read texels.
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset);

namespace {

void validate_gather_operand(const Tensor& t, const char* name) {
  TP_CHECK(
      t.dtype() == DType::Float32 || t.dtype() == DType::Float16,
      std::string("Vulkan ") + name +
          " supports Float32 and Float16 tensors only");
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

// A shape left-padded to four axes with ones, outermost first - the form the
// row lookups address a 1-d to 4-d payload with.
ivec4 padded_sizes(const std::vector<int64_t>& sizes) {
  ivec4 padded{1, 1, 1, 1};
  const int rank = static_cast<int>(sizes.size());
  const int offset = 4 - rank;
  for (int axis = 0; axis < rank; ++axis) {
    padded[offset + axis] = static_cast<int32_t>(sizes[static_cast<size_t>(axis)]);
  }
  return padded;
}

// Contiguous strides of that padded shape.  A padded axis has size one and
// inherits the stride of the axis below it, and every element of the payload
// sits below that stride, so its coordinate comes out zero.
ivec4 padded_strides(const std::vector<int64_t>& sizes) {
  const ivec4 padded = padded_sizes(sizes);
  ivec4 strides{1, 1, 1, 1};
  strides[3] = 1;
  for (int axis = 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * padded[axis + 1];
  }
  return strides;
}

// Builds the Int32 payload a lookup walks: one code per entry of a flat 1-D
// list, in the storage class the looked-up rows live in.  Following the model
// keeps a lookup on the route it already chose: a dense list beside a texel
// table would otherwise drag the whole table into linear storage to bind both
// operands alike.  A list already on the device stays there, because a queue
// sync to narrow its codes on the host costs a full wait per call and a decode
// step makes one per token.  Codes in any other element type still come through
// the host, which converts whatever it is handed.
api::vTensor upload_index_payload(
    const Tensor& indices,
    api::StorageType model_storage) {
  api::Context* const context = api::context();

  const api::GPUMemoryLayout model_layout =
      model_storage == api::StorageType::BUFFER
      ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
      : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  if (indices.device().is_vulkan() &&
      (indices.dtype() == DType::Int32 || indices.dtype() == DType::Int64)) {
    Tensor flat =
        indices.dim() == 1 ? indices : indices.reshape({indices.numel()});
    api::vTensor v{
        context, {flat.numel()}, DType::Int32, model_storage, model_layout};
    // The copy is the whole move: it narrows each 64-bit code to its low word,
    // crosses the storage class if the list arrived in the other one, and is a
    // plain transfer when both already agree.
    Tensor codes = convert(v);
    codes.copy_(flat);
    return v;
  }

  Tensor flat_cpu =
      indices.to(Device(DeviceType::CPU))
          .reshape({indices.numel()})
          .to(DType::Int32)
          .contiguous();

  api::vTensor v{
      context, {flat_cpu.numel()}, DType::Int32, model_storage, model_layout};
  // The payload layout follows the storage: a texel holds one index in its .x
  // lane, so the packed host bytes are the texel-linear image, while a linear
  // payload is one int per element in row order.  Uploading the packed bytes
  // over a linear allocation would also write past it, since the host repack
  // is four ints per index.
  if (v.storage_type() == api::StorageType::BUFFER) {
    utils::upload_host_bytes(
        v,
        flat_cpu.impl()->storage().data(),
        flat_cpu.numel() * flat_cpu.itemsize());
  } else {
    Tensor flat_packed = utils::nchw_to_nc4hw(flat_cpu);
    utils::upload_host_bytes(
        v,
        flat_packed.impl()->storage().data(),
        flat_packed.numel() * flat_packed.itemsize());
  }
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
      weight.dtype() == DType::Float32 || weight.dtype() == DType::Float16,
      "Vulkan embedding supports a Float32 or Float16 weight only");
  TP_CHECK(
      indices.dtype() == DType::Int64 || indices.dtype() == DType::Int32,
      "Vulkan embedding: indices must be Int64 or Int32");
  TP_CHECK(
      weight.dim() == 2,
      "Vulkan embedding expects a 2d [rows, features] weight");

  api::Context* const context = api::context();

  api::vTensor v_weight = convert(weight);

  const int64_t index_count = indices.numel();
  const int64_t features = weight.size(1);

  // The list is built in the class the lookup is going to use: a table with no
  // image form, or a result too long to have one, takes the flat route and wants
  // a dense list, while a lookup that can stay in texels keeps one code per
  // texel.
  const api::StorageType index_storage =
      v_weight.storage_type() == api::StorageType::BUFFER ||
          !image_storage_supported(context, {index_count, features})
      ? api::StorageType::BUFFER
      : api::StorageType::TEXTURE_3D;
  api::vTensor v_indices = upload_index_payload(indices, index_storage);

  // A lookup runs on the flat path as soon as one of the three payloads it
  // touches has no image form: the table (a wide vocabulary), the index list
  // (one texel per entry) or the output rows it implies.  The texture-stored
  // operands are copied into dense linear storage so the same row-major
  // shader serves small and large lookups, and the output is released as a
  // linear payload beside them.
  const bool linear_lookup =
      v_weight.storage_type() == api::StorageType::BUFFER ||
      v_indices.storage_type() == api::StorageType::BUFFER ||
      !image_storage_supported(context, {index_count, features});
  if (linear_lookup) {
    api::vTensor v_weight_buf = to_buffer_storage_dispatch(context, v_weight);
    api::vTensor v_indices_buf =
        to_buffer_storage_dispatch(context, v_indices);
    api::vTensor v_output{
        context, {index_count, features}, weight.dtype(),
        api::StorageType::BUFFER, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    // Words of the output payload: single precision fills one with four lanes,
    // half packs two.
    const uint32_t lanes = weight.dtype() == DType::Float16 ? 2u : 4u;
    const uint32_t n =
        static_cast<uint32_t>(index_count) * static_cast<uint32_t>(features);
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
        VK_KERNEL_FROM_STR(weight.dtype() == DType::Float16
                               ? "embedding_buffer_half"
                               : "embedding_buffer"),
        pipeline_barrier,
        {api::utils::div_up(n, lanes), 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_weight_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_indices_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    Tensor out = convert(v_output);
    // A lookup that had to stream its table still hands its rows out in the
    // form their own shape fits: the kernels downstream read a texture, and a
    // small payload left in words has no linear variant to run on.
    if (image_storage_supported(context, {index_count, features})) {
      out = materialize_view(
          out, {index_count, features}, {features, 1}, 0);
    }
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
  api::vTensor v_output{context, {index_count, features}, weight.dtype()};

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
// any axis, e.g. appending decode steps to a preallocated kv cache or writing
// a page of a slot table.  The destination rows arrive through an Int32 payload
// that the shader walks, so a list already on the device stays there: folding it
// onto the host would cost a queue sync per call, and a decode step makes one
// per token.  Range is settled in the shader, where a code naming no row writes
// nothing.
//
// Geometry: a row is the block of elements the scattered axis names, so the
// payload flattens to planes x rows x width with planes the axes above it and
// width the axes below it.  A linear payload is that block in row-major order
// and any axis works.  An image payload keeps the axis just above its last one
// on the texture's y extent and packs the rest of the row into texels and lanes,
// so only that axis names whole texels; a row straddling the packed axis would
// have to be copied lane by lane and stays composite-side.
//
// Two codes naming the same row race for it texel by texel, so which source row
// survives is unspecified; nothing reads a row back mid-scatter, so the race
// costs no correctness beyond that freedom.
Tensor& index_copy_rows_inplace(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
  NoGradGuard no_grad;
  TP_CHECK(
      self.device().is_vulkan() && source.device().is_vulkan(),
      "Vulkan index_copy_: both tensors must be on the device");
  const int64_t rank = self.dim();
  const int64_t wrapped = dim < 0 ? dim + rank : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < rank,
      "Vulkan index_copy_: dimension out of range");
  TP_CHECK(
      index.dim() < 2,
      "Vulkan index_copy_: index must be a vector or a single code");
  TP_CHECK(
      source.dim() == rank,
      "Vulkan index_copy_: source and destination must have the same rank");
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

  const int64_t rows = self.size(wrapped);
  const int64_t n_idx = index.numel();
  int64_t planes = 1;
  int64_t width = 1;
  for (int64_t axis = 0; axis < wrapped; ++axis) {
    planes *= self.size(axis);
  }
  for (int64_t axis = wrapped + 1; axis < rank; ++axis) {
    width *= self.size(axis);
  }
  TP_CHECK(
      source.size(wrapped) == n_idx,
      "Vulkan index_copy_: source rows must match the index length");
  for (int64_t axis = 0; axis < rank; ++axis) {
    TP_CHECK(
        axis == wrapped || source.size(axis) == self.size(axis),
        "Vulkan index_copy_: source must match off-axis sizes");
  }
  // The row walk addresses elements as words of a four-lane group, so the whole
  // payload has to stay inside the signed range the block arithmetic uses.
  TP_CHECK(
      self.numel() <= std::numeric_limits<int32_t>::max(),
      "Vulkan index_copy_: destination holds more than two billion elements");

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_source = convert(source);
  api::vTensor v_indices = upload_index_payload(index, v_self.storage_type());

  const bool image_destination =
      v_self.storage_type() != api::StorageType::BUFFER;
  // A source that arrives without an image form is given one below; the payload
  // that owns those elements has to outlive the dispatch that reads them.
  Tensor source_image;
  if (image_destination) {
    TP_CHECK(
        rank >= 2 && wrapped == rank - 2,
        "Vulkan index_copy_: an image payload scatters the axis above its last "
        "one only");
    // The texel route binds both payloads as images, so a source that arrived
    // in words crosses over first; the identity gather releases the same
    // elements as a texture of the source's own shape.
    if (v_source.storage_type() == api::StorageType::BUFFER) {
      const std::vector<int64_t> source_sizes(
          static_cast<std::vector<int64_t>>(source.shape()));
      source_image = materialize_view(
          source,
          source_sizes,
          SizesAndStrides::compute_contiguous_strides(source_sizes),
          0);
      v_source = convert(source_image);
    }
  } else {
    // Linear payloads scatter through a buffer-to-buffer variant: the index
    // payload is copied to a dense Int32 buffer and each invocation copies four
    // feature lanes from a source row into the indexed destination row, walking
    // every plane above the scattered axis.
    v_indices = to_buffer_storage_dispatch(context, v_indices);
    // The source payload may still fit texture storage when the scatter is
    // small; bring it to a dense buffer so every linear-path argument has
    // memory attached (an empty buffer binding would silently skip the job).
    v_source = to_buffer_storage_dispatch(context, v_source);
    const uint32_t nf4 = static_cast<uint32_t>((width + 3) / 4);
    const int64_t invocations = nf4 * planes * n_idx;
    TP_CHECK(
        invocations <= std::numeric_limits<int32_t>::max(),
        "Vulkan index_copy_: the row walk needs more than two billion "
        "invocations");
    const uint32_t n = static_cast<uint32_t>(invocations);
    const struct IndexCopyBufferBlock {
      int32_t planes;
      int32_t n_idx;
      int32_t width;
      int32_t rows;
    } block{
        static_cast<int32_t>(planes),
        static_cast<int32_t>(n_idx),
        static_cast<int32_t>(width),
        static_cast<int32_t>(rows),
    };
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR("index_copy_buffer"), pipeline_barrier,
        {n, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_source.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_indices.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    return self;
  }

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
  validate_gather_operand(self, "index_select");
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

  api::vTensor v_self = convert(self);

  const int64_t index_count = index.numel();
  const int64_t out_count = index_count * inner;

  // The list is built in the class the gather is going to use, so a source with
  // an image form is not dragged into dense storage just because the codes
  // arrived wide.
  const api::StorageType index_storage =
      v_self.storage_type() == api::StorageType::BUFFER ||
          !image_storage_supported(context, {self.numel()}) ||
          !image_storage_supported(context, {out_count})
      ? api::StorageType::BUFFER
      : api::StorageType::TEXTURE_3D;
  api::vTensor v_indices = upload_index_payload(index, index_storage);

  // The texel route walks the payload as a flat 1-D texture, so the source seen
  // whole and the gathered rows must both have an image form, and the index
  // list must fit one texel per entry.  Otherwise the gather runs flat: a
  // texture-stored source is copied into dense linear storage, whose row-major
  // order is already the flattened one, so no 1-D view is materialized.
  const bool texture_gather =
      v_self.storage_type() != api::StorageType::BUFFER &&
      v_indices.storage_type() != api::StorageType::BUFFER &&
      image_storage_supported(context, {self.numel()}) &&
      image_storage_supported(context, {out_count});
  if (!texture_gather) {
    api::vTensor v_input_buf =
        v_self.storage_type() == api::StorageType::BUFFER
        ? v_self
        : to_buffer_storage_dispatch(context, v_self);
    api::vTensor v_indices_buf =
        to_buffer_storage_dispatch(context, v_indices);
    api::vTensor v_out_buf{
        context, {out_count}, self.dtype(), api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const uint32_t lanes = self.dtype() == DType::Float16 ? 2u : 4u;
    const uint32_t n = static_cast<uint32_t>(out_count);
    const struct IndexSelectBufferBlock {
      int32_t inner;
      int32_t row_stride;
      int32_t count;
      int32_t rows;
    } block{
        static_cast<int32_t>(inner),
        static_cast<int32_t>(inner),
        static_cast<int32_t>(index_count),
        static_cast<int32_t>(self.size(wrapped)),
    };
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(self.dtype() == DType::Float16
                               ? "index_select_buffer_half"
                               : "index_select_buffer"),
        pipeline_barrier,
        {api::utils::div_up(n, lanes), 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_out_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_indices_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    std::vector<int64_t> out_sizes(
        static_cast<std::vector<int64_t>>(self.shape()));
    out_sizes[static_cast<size_t>(wrapped)] = index_count;
    Tensor flat_out = convert(v_out_buf);
    return flat_out.reshape(out_sizes);
  }

  // Both operands have an image form, so the gather walks the flattened texture
  // with one texel per selected element.
  Tensor flat = self.reshape({self.numel()});
  api::vTensor v_input = convert(flat);
  api::vTensor v_output{context, {out_count}, self.dtype()};

  const struct IndexSelectBlock block{
      static_cast<int32_t>(inner),
      static_cast<int32_t>(inner),
      static_cast<int32_t>(index_count),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // The shader reads invocation (slot inside the gathered row, index), so the
  // grid spans inner by the index count even though each payload is a flat
  // 1-D texture: the output's own extents only run over the flat element and
  // would leave every row past the first slot unwritten.
  const api::utils::uvec3 grid{
      static_cast<uint32_t>(inner),
      static_cast<uint32_t>(index_count),
      1u};

  context->submit_compute_job(
      VK_KERNEL(index_select), pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
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

//
// Per-position row selection: output element e stands for the source element at
// e's own coordinates with the gathered coordinate replaced by the e-th code.
// The index tensor has the source's rank and is no wider than it along any other
// axis, so the output has exactly the index shape.  Which route runs depends on
// where the gathered axis sits in the texel layout: the two innermost axes are
// the width and the height, so a code moves the fetch along a texel axis and one
// invocation can write a whole output texel; the channel and batch axes live in
// the lanes and the depth planes, where a code would have to move a value
// between lanes of different texels, so those gathers run over linear storage.
// A code naming no row of the source contributes a zero, the answer the other row
// lookups of this backend already give.
Tensor gather_kernel(const Tensor& self, int64_t dim, const Tensor& index) {
  validate_gather_operand(self, "gather");
  TP_CHECK(
      index.dtype() == DType::Int32 || index.dtype() == DType::Int64,
      "Vulkan gather: index must be Int32 or Int64");
  const int64_t wrapped = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < self.dim(),
      "Vulkan gather: dimension out of range");
  TP_CHECK(
      index.dim() == self.dim(),
      "Vulkan gather: index must have the same rank as the source");
  for (int64_t axis = 0; axis < self.dim(); ++axis) {
    TP_CHECK(
        axis == wrapped || index.size(axis) <= self.size(axis),
        "Vulkan gather: index is wider than the source at dimension ",
        axis);
  }

  const std::vector<int64_t> out_sizes(
      static_cast<std::vector<int64_t>>(index.shape()));
  const int64_t out_count = index.numel();
  if (out_count == 0) {
    return empty_kernel(out_sizes, self.dtype(), self.device(), false);
  }

  api::Context* const context = api::context();

  // Both routes read a row-major payload, so a strided source is laid out flat
  // first; the index list is flattened by its own upload.
  Tensor src = self.is_contiguous() ? self : self.contiguous();
  api::vTensor v_self = convert(src);

  // The gathered axis counted in the padded four-axis shape the mapping walks.
  const int32_t axis = static_cast<int32_t>(wrapped + 4 - self.dim());
  const bool texel_axis =
      wrapped == self.dim() - 1 || wrapped == self.dim() - 2;
  const bool texture_gather =
      texel_axis && v_self.storage_type() != api::StorageType::BUFFER &&
      image_storage_supported(context, out_sizes) &&
      image_storage_supported(context, {out_count});

  api::vTensor v_indices = upload_index_payload(
      index,
      texture_gather ? api::StorageType::TEXTURE_3D
                     : api::StorageType::BUFFER);

  if (!texture_gather) {
    api::vTensor v_input_buf =
        v_self.storage_type() == api::StorageType::BUFFER
        ? v_self
        : to_buffer_storage_dispatch(context, v_self);
    api::vTensor v_indices_buf =
        to_buffer_storage_dispatch(context, v_indices);
    api::vTensor v_out_buf{
        context, {out_count}, self.dtype(), api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const uint32_t lanes = self.dtype() == DType::Float16 ? 2u : 4u;
    const ivec4 out_strides = padded_strides(out_sizes);
    const ivec4 self_strides = padded_strides(
        static_cast<std::vector<int64_t>>(self.shape()));
    const struct GatherBufferBlock {
      ivec4 out_strides;
      ivec4 self_strides;
      int32_t axis;
      int32_t self_rows;
      int32_t count;
      int32_t fill;
    } block{
        out_strides,
        self_strides,
        axis,
        static_cast<int32_t>(self.size(wrapped)),
        static_cast<int32_t>(out_count),
        0};
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(self.dtype() == DType::Float16 ? "gather_buffer_half"
                                                          : "gather_buffer"),
        pipeline_barrier,
        {api::utils::div_up(static_cast<uint32_t>(out_count), lanes), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_out_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        v_indices_buf.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        params.buffer());
    Tensor flat_out = convert(v_out_buf);
    return flat_out.reshape(out_sizes);
  }

  // The texel route: one invocation per output texel, four channel lanes at a
  // time.  The depth axis stacks each payload's own channel groups, so the plane
  // a code is read from is counted with the source's groups per batch - the index
  // tensor may hold fewer channels than the source.
  api::vTensor v_output{context, out_sizes, self.dtype()};

  const ivec4 out_padded = padded_sizes(out_sizes);
  const ivec4 self_padded =
      padded_sizes(static_cast<std::vector<int64_t>>(self.shape()));
  const struct GatherBlock {
    ivec4 out_sizes; // (width, height, channel, batch) of the output
    int32_t self_planes_per_batch;
    int32_t gather_width;
    int32_t self_rows;
    int32_t fill;
  } block{
      ivec4{
          out_padded[3],
          out_padded[2],
          out_padded[1],
          out_padded[0]},
      (self_padded[1] + 3) / 4,
      wrapped == self.dim() - 1 ? 1 : 0,
      static_cast<int32_t>(self.size(wrapped)),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 grid = v_output.extents();

  context->submit_compute_job(
      VK_KERNEL(gather), pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_indices.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

//
// Scatter of one repeated number: the codes of the index list name elements of
// the destination, and each of those takes the number in place of what it held.
//
// The write cannot be a single pass, because a destination element is a lane of
// a texel or of a packed word, and an image or word store always replaces the
// whole group.  Two invocations that each own one lane of the same group would
// hand each other's lane back as stale, so the pass is split the way the row
// lookups of this backend split a request that outgrows one store:
//
//   1. a mark pass over the index list, which records in a word of its own
//      which entry names each element of the destination;
//   2. an apply pass over the destination, where one invocation owns one whole
//      texel or word and rewrites the lanes its marks name.
//
// The mark payload is one 32-bit word per element, which a whole-word store
// reaches without disturbing a neighbour, and it holds the position of the
// entry plus one so that the clearing pass can leave "nothing names me" as
// zero.  Recording a position rather than a bare hit is what lets one mark pass
// serve both rewrites: a constant needs only the mark, a source payload reads
// the entry it points at.
//
// Both passes address the destination in its row-major logical order: the mark
// pass runs the gather mapping backwards from an entry of the list, the apply
// pass walks the texel or word layout of the payload it rewrites.  A code that
// names no row marks nothing, which is the answer a row lookup already gives to
// a request outside its rows.
//

// Words of the mark payload, one per element of the destination, cleared on the
// device: an upload of the same size would cross the host twice over.
api::vTensor zeroed_marks(api::Context* const context, const int64_t elements) {
  api::vTensor v{
      context,
      {elements},
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const uint32_t count = safe_downcast_to_u32(elements);
  const struct MarksZero final {
    uint32_t buf_length;
  } block{count};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL(buffer_zero), pipeline_barrier, {count, 1u, 1u},
      {256u, 1u, 1u}, VK_NULL_HANDLE,
      v.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      params.buffer());
  return v;
}

// Addressing shared by both passes: the rank and axis checks a gather already
// makes, since the index list of a scatter has the geometry of the destination
// it writes.
void check_scatter_geometry(
    const Tensor& self,
    int64_t wrapped,
    const Tensor& index) {
  TP_CHECK(
      index.dim() == self.dim(),
      "Vulkan scatter: index must have the same rank as the destination");
  for (int64_t axis = 0; axis < self.dim(); ++axis) {
    TP_CHECK(
        axis == wrapped || index.size(axis) <= self.size(axis),
        "Vulkan scatter: index is wider than the destination at dimension ",
        axis);
  }
}

// The mark pass of a scatter: one word per element of the destination, holding
// the position of the entry that names it plus one.  Both rewrites share it,
// because a position is what a payload needs to find its value and a constant
// simply ignores it.
void mark_scatter_destinations(
    api::Context* const context,
    api::PipelineBarrier& pipeline_barrier,
    const Tensor& self,
    const Tensor& index,
    int64_t wrapped,
    api::vTensor& v_marks,
    api::vTensor& v_indices) {
  const int64_t rank = self.dim();
  const ivec4 index_strides =
      padded_strides(static_cast<std::vector<int64_t>>(index.shape()));
  const ivec4 self_strides =
      padded_strides(static_cast<std::vector<int64_t>>(self.shape()));
  const struct MarkBlock final {
    ivec4 index_strides;
    ivec4 self_strides;
    int32_t axis; // the scattered axis of the padded four-axis shape
    int32_t self_rows; // entries available along that axis
    int32_t count; // entries of the index list
    int32_t fill;
  } mark_block{
      index_strides,
      self_strides,
      static_cast<int32_t>(wrapped + 4 - rank),
      static_cast<int32_t>(self.size(wrapped)),
      static_cast<int32_t>(index.numel()),
      0};

  api::UniformParamsBuffer mark_params(context, mark_block);
  context->submit_compute_job(
      VK_KERNEL(scatter_mark), pipeline_barrier,
      {safe_downcast_to_u32(index.numel()), 1u, 1u}, {256u, 1u, 1u},
      VK_NULL_HANDLE,
      v_marks.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_indices.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      mark_params.buffer());
}

Tensor& scatter_value_inplace_kernel(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  NoGradGuard no_grad;
  validate_gather_operand(self, "scatter");
  TP_CHECK(
      index.dtype() == DType::Int32 || index.dtype() == DType::Int64,
      "Vulkan scatter: index must be Int32 or Int64");
  const int64_t rank = self.dim();
  int64_t wrapped = dim < 0 ? dim + rank : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < rank,
      "Vulkan scatter: dimension out of range");
  check_scatter_geometry(self, wrapped, index);

  if (self.numel() == 0 || index.numel() == 0) {
    return self;
  }

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_marks = zeroed_marks(context, self.numel());
  // The list is walked by its own flat order in both passes, so it always rides
  // linear storage; the upload narrows a wider code to the word the pass reads.
  api::vTensor v_indices =
      upload_index_payload(index, api::StorageType::BUFFER);

  api::PipelineBarrier pipeline_barrier{};
  mark_scatter_destinations(
      context, pipeline_barrier, self, index, wrapped, v_marks, v_indices);

  const float written = static_cast<float>(value.toDouble());

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const struct ValueBufferBlock final {
      int32_t count; // elements of the destination
      float value;
      int32_t fill;
      int32_t fill2;
    } block{static_cast<int32_t>(self.numel()), written, 0, 0};
    api::UniformParamsBuffer params(context, block);
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_for_dtype(
            "buffer_scatter_value", self.dtype())),
        pipeline_barrier,
        {buffer_dispatch_count(
             safe_downcast_to_u32(self.numel()), self.dtype(), 4u),
         1u,
         1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        // The apply pass reads the lanes it leaves alone, so the destination is
        // bound in both directions.
        v_self.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_marks.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return self;
  }

  const ivec4 self_sizes = make_whcn_ivec4(
      static_cast<std::vector<int64_t>>(self.shape()));
  const struct ValueBlock final {
    ivec4 self_sizes; // (width, height, channels, batch)
    int32_t planes_per_batch; // channel groups of four stacked per batch
    float value;
    int32_t fill;
    int32_t fill2;
  } block{
      self_sizes,
      static_cast<int32_t>((self_sizes[2] + 3) / 4),
      written,
      0,
      0};
  api::UniformParamsBuffer params(context, block);
  const api::utils::uvec3 grid = v_self.extents();

  context->submit_compute_job(
      kernel_for("scatter_value", self.dtype()), pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_marks.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return self;
}

Tensor scatter_value_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  Tensor out = self.clone();
  scatter_value_inplace_kernel(out, dim, index, value);
  return out;
}

// The values a scatter moves are read at the flat position of the entry that
// names a destination, so the payload has to be walkable by position: whatever
// storage it arrived in, it is brought to dense linear order here, which is
// already the row-major order its entries are counted in.
//
// A payload that lives in the very storage being written would have to be bound
// in both directions at once, and the apply pass reads a lane it was not asked
// to change as well as the one it stores.  Copying such a source out first is
// what keeps the two bindings apart; the usual separate payload copies nothing.
api::vTensor scatter_source_payload(
    api::Context* const context,
    const Tensor& self,
    const Tensor& src) {
  const bool shares_storage = self.impl()->storage().unsafeGetStorageImpl() ==
      src.impl()->storage().unsafeGetStorageImpl();
  api::vTensor v_src = convert(shares_storage ? src.clone() : src);
  if (v_src.storage_type() == api::StorageType::BUFFER) {
    return v_src;
  }
  return to_buffer_storage_dispatch(context, v_src);
}

Tensor& scatter_src_inplace_kernel(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  NoGradGuard no_grad;
  validate_gather_operand(self, "scatter");
  validate_gather_operand(src, "scatter");
  TP_CHECK(
      index.dtype() == DType::Int32 || index.dtype() == DType::Int64,
      "Vulkan scatter: index must be Int32 or Int64");
  const int64_t rank = self.dim();
  int64_t wrapped = dim < 0 ? dim + rank : dim;
  TP_CHECK(
      wrapped >= 0 && wrapped < rank,
      "Vulkan scatter: dimension out of range");
  check_scatter_geometry(self, wrapped, index);

  // The values are read at the position of the entry that names a destination,
  // so the payload has to cover the list: fewer axes than the list has take the
  // outer ones as they stand, one value repeats over every entry, a payload
  // narrower along an axis is read again for the entries below it, and a
  // narrower element type widens to the one the destination carries.
  const std::vector<int64_t> list_shape(
      static_cast<std::vector<int64_t>>(index.shape()));
  const int64_t depth = static_cast<int64_t>(list_shape.size());
  TP_CHECK(
      src.dim() <= depth,
      "Vulkan scatter: the source must not have more axes than the list");
  // The axes the list carries that the payload does not spell out stand at the
  // front of it, each with one value to reach the entries below.
  const int64_t lead = depth - src.dim();
  std::vector<int64_t> held_shape(static_cast<size_t>(lead), 1);
  for (int64_t axis = 0; axis < src.dim(); ++axis) {
    TP_CHECK(
        src.size(axis) == 1 || src.size(axis) == list_shape[lead + axis],
        "Vulkan scatter: the source must reach the list at dimension ", axis);
    held_shape.push_back(src.size(axis));
  }
  Tensor payload =
      src.dtype() == self.dtype() ? src : src.to(self.dtype());
  if (held_shape != list_shape) {
    payload = payload.reshape(held_shape).expand(list_shape).contiguous();
  }

  if (self.numel() == 0 || index.numel() == 0) {
    return self;
  }

  api::Context* const context = api::context();
  api::vTensor v_self = convert(self);
  api::vTensor v_marks = zeroed_marks(context, self.numel());
  api::vTensor v_indices =
      upload_index_payload(index, api::StorageType::BUFFER);
  api::vTensor v_src = scatter_source_payload(context, self, payload);

  api::PipelineBarrier pipeline_barrier{};
  mark_scatter_destinations(
      context, pipeline_barrier, self, index, wrapped, v_marks, v_indices);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const struct SrcBufferBlock final {
      int32_t count; // elements of the destination
      int32_t fill;
      int32_t fill2;
      int32_t fill3;
    } block{static_cast<int32_t>(self.numel()), 0, 0, 0};
    api::UniformParamsBuffer params(context, block);
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader_for_dtype(
            "buffer_scatter_src", self.dtype())),
        pipeline_barrier,
        {buffer_dispatch_count(
             safe_downcast_to_u32(self.numel()), self.dtype(), 4u),
         1u,
         1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        // The apply pass reads the lanes it leaves alone, so the destination is
        // bound in both directions.
        v_self.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_marks.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_src.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return self;
  }

  const ivec4 self_sizes = make_whcn_ivec4(
      static_cast<std::vector<int64_t>>(self.shape()));
  const struct SrcBlock final {
    ivec4 self_sizes; // (width, height, channels, batch)
    int32_t planes_per_batch; // channel groups of four stacked per batch
    int32_t fill;
    int32_t fill2;
  } block{
      self_sizes,
      static_cast<int32_t>((self_sizes[2] + 3) / 4),
      0,
      0};
  api::UniformParamsBuffer params(context, block);
  const api::utils::uvec3 grid = v_self.extents();

  context->submit_compute_job(
      kernel_for("scatter_src", self.dtype()), pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_marks.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_src.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return self;
}

Tensor scatter_src_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  Tensor out = self.clone();
  scatter_src_inplace_kernel(out, dim, index, src);
  return out;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, IndexingKernels) {
  m.impl("embedding", &tensorplay::vulkan::ops::embedding_kernel);
  m.impl("index_select", &tensorplay::vulkan::ops::index_select_kernel);
  m.impl("gather", &tensorplay::vulkan::ops::gather_kernel);
  m.impl("index_copy_", &tensorplay::vulkan::ops::index_copy__kernel);
  m.impl("index_copy", &tensorplay::vulkan::ops::index_copy_kernel);
  m.impl("scatter.value", &tensorplay::vulkan::ops::scatter_value_kernel);
  m.impl(
      "scatter_.value",
      &tensorplay::vulkan::ops::scatter_value_inplace_kernel);
  m.impl("scatter.src", &tensorplay::vulkan::ops::scatter_src_kernel);
  m.impl(
      "scatter_.src",
      &tensorplay::vulkan::ops::scatter_src_inplace_kernel);
}

#endif /* USE_VULKAN */
