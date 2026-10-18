#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in View.cpp; materializes a strided view into a dense payload.
// Buffer-stored inputs use the linear gather path.
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset);

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

//
// Shape ops on Vulkan tensors.  The backend owns GPU resources but not
// stride bookkeeping (the ordinary TensorImpl does), so metadata-style ops
// (transpose/permute/unsqueeze/squeeze) materialize a payload with the new
// shape by streaming through the linear host-order representation, and
// data-movement ops (slice/select/cat) gather elements in a shader.
//

Tensor repack_with_sizes(
    const Tensor& self,
    const std::vector<int64_t>& new_sizes) {
  return reshape_kernel(self, new_sizes);
}


namespace {

// Materializes a tensor's shape into a plain sizes vector.  The shape
// accessor returns a value object, so the copy must be taken once instead
// of reaching through repeated temporaries.
std::vector<int64_t> shape_vector(const Tensor& t) {
  const Size shape = t.shape();
  return std::vector<int64_t>(shape.begin(), shape.end());
}

// Sizes of `self` after the prepadded-axis permutation `perm` (out axis k
// reads in axis perm[k]).
std::vector<int64_t> permuted_sizes(
    const Tensor& self,
    const ivec4& perm) {
  const std::vector<int64_t> in_sizes = shape_vector(self);
  std::vector<int64_t> padded(4, 1);
  const int64_t offset = 4 - static_cast<int64_t>(in_sizes.size());
  for (size_t i = 0; i < in_sizes.size(); ++i) {
    padded[static_cast<size_t>(offset + static_cast<int64_t>(i))] = in_sizes[i];
  }

  std::vector<int64_t> out_padded(4, 1);
  for (int64_t k = 0; k < 4; ++k) {
    out_padded[static_cast<size_t>(k)] =
        padded[static_cast<size_t>(perm[static_cast<size_t>(k)])];
  }

  // Collapse the prepadded leading axes that the input rank did not have.
  const int64_t extra = 4 - static_cast<int64_t>(in_sizes.size());
  return std::vector<int64_t>(
      out_padded.begin() + extra, out_padded.end());
}

} // namespace

} // namespace

// Permutes a texture-backed tensor through the dedicated gather shader:
// the output invocation reconstructs its logical coordinates, permutes them
// back to input coordinates, and fetches one element per channel lane, so
// payload reordering happens on the GPU without a host round-trip.
Tensor permute_with_dims(const Tensor& self, const ivec4& perm) {
  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);

  // Permute over a linear payload is a stride-permuted view: output axis
  // k reads input slot perm[base + k], whose logical dim is that slot
  // minus the prepadded base.  Materializing through the strided gather
  // reorders the elements into a dense buffer in one pass.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    const int64_t ndim = self.dim();
    const int64_t base = 4 - ndim;
    std::vector<int64_t> out_strides;
    out_strides.reserve(static_cast<size_t>(ndim));
    for (int64_t k = 0; k < ndim; ++k) {
      const int64_t in_slot =
          perm[static_cast<size_t>(base + static_cast<int64_t>(k))];
      out_strides.push_back(
          self.stride(static_cast<size_t>(in_slot - base)));
    }
    const int64_t offset = self.unsafeGetTensorImpl()->storage_offset();
    return materialize_view(
        self, permuted_sizes(self, perm), out_strides, offset);
  }

  api::vTensor v_output{
      context, permuted_sizes(self, perm), self.dtype()};

  // Prepadded {d0, d1, d2, d3} logical sizes, d0 = batch, d1 = channel.
  auto prepadded = [](const std::vector<int64_t>& sizes) {
    std::vector<int64_t> padded(4, 1);
    const int64_t offset = 4 - static_cast<int64_t>(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      padded[static_cast<size_t>(offset + static_cast<int64_t>(i))] = sizes[i];
    }
    return padded;
  };

  const std::vector<int64_t> in_logical = prepadded(shape_vector(self));
  const std::vector<int64_t> out_logical =
      prepadded(permuted_sizes(self, perm));

  const struct PermuteBlock final {
    ivec4 in_logical;
    ivec4 out_logical;
    ivec4 perm;
    int in_c_depth;
    int out_c_depth;
  } block{
      // The shader reads the prepadded axes in natural {d0, d1, d2, d3}
      // order, so the block is built directly rather than through the
      // right-aligned WHCN helper (which would reverse the vector).
      ivec4(
          static_cast<int32_t>(in_logical[0]),
          static_cast<int32_t>(in_logical[1]),
          static_cast<int32_t>(in_logical[2]),
          static_cast<int32_t>(in_logical[3])),
      ivec4(
          static_cast<int32_t>(out_logical[0]),
          static_cast<int32_t>(out_logical[1]),
          static_cast<int32_t>(out_logical[2]),
          static_cast<int32_t>(out_logical[3])),
      perm,
      c_depth_of(in_logical),
      c_depth_of(out_logical),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(permute), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor transpose_kernel(const Tensor& self, int64_t dim0, int64_t dim1) {
  const int64_t ndim = self.dim();
  dim0 = dim0 < 0 ? dim0 + ndim : dim0;
  dim1 = dim1 < 0 ? dim1 + ndim : dim1;
  TP_CHECK(
      dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim,
      "Vulkan transpose: dim out of range");

  // A swap of innermost-first axes k0 = ndim-1-dim0, k1 = ndim-1-dim1
  // becomes a swap of prepadded axes p0 = 4-(k0+1), p1 = 4-(k1+1).
  std::vector<int64_t> perm{0, 1, 2, 3};
  std::swap(
      perm[static_cast<size_t>(3 - (ndim - 1 - dim0))],
      perm[static_cast<size_t>(3 - (ndim - 1 - dim1))]);

  return permute_with_dims(
      self,
      ivec4(
          static_cast<int32_t>(perm[0]),
          static_cast<int32_t>(perm[1]),
          static_cast<int32_t>(perm[2]),
          static_cast<int32_t>(perm[3])));
}

Tensor permute_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
  const int64_t ndim = self.dim();
  TP_CHECK(
      static_cast<int64_t>(dims.size()) == ndim,
      "Vulkan permute: number of dims does not match tensor rank");

  std::set<int64_t> unique(dims.begin(), dims.end());
  TP_CHECK(
      static_cast<int64_t>(unique.size()) == ndim,
      "Vulkan permute: repeated dim in the permutation");

  std::vector<int64_t> new_sizes;
  new_sizes.reserve(dims.size());
  for (const int64_t d : dims) {
    const int64_t wrapped = d < 0 ? d + ndim : d;
    TP_CHECK(
        wrapped >= 0 && wrapped < ndim,
        "Vulkan permute: dim out of range");
    new_sizes.push_back(self.size(wrapped));
  }

  // out axis k reads in axis perm[k].  Both sides are expressed in
  // prepadded d-slots: leftmost axis k lives in slot 4-ndim+k.
  std::vector<int64_t> perm(4, 0);
  for (int64_t k = 0; k < ndim; ++k) {
    int64_t wrapped = dims[static_cast<size_t>(k)];
    wrapped = wrapped < 0 ? wrapped + ndim : wrapped;
    perm[static_cast<size_t>(4 - ndim + k)] =
        static_cast<int64_t>(4 - ndim + wrapped);
  }

  return permute_with_dims(
      self,
      ivec4(
          static_cast<int32_t>(perm[0]),
          static_cast<int32_t>(perm[1]),
          static_cast<int32_t>(perm[2]),
          static_cast<int32_t>(perm[3])));
}

Tensor t_kernel(const Tensor& self) {
  TP_CHECK(
      self.dim() <= 2,
      "Vulkan t() expects a tensor with <= 2 dimensions");
  if (self.dim() < 2) {
    return self;
  }
  return transpose_kernel(self, 0, 1);
}

Tensor squeeze_dim_kernel(const Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  dim = dim < 0 ? dim + ndim : dim;
  TP_CHECK(dim >= 0 && dim < ndim, "Vulkan squeeze: dim out of range");
  if (self.size(dim) != 1) {
    // Non-singleton dims return an equivalent tensor.
    return self;
  }

  std::vector<int64_t> new_sizes = shape_vector(self);
  new_sizes.erase(new_sizes.begin() + dim);

  return repack_with_sizes(self, new_sizes);
}

Tensor unsqueeze_kernel(const Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  dim = dim < 0 ? dim + ndim + 1 : dim;
  TP_CHECK(
      dim >= 0 && dim <= ndim, "Vulkan unsqueeze: dim out of range");

  std::vector<int64_t> new_sizes = shape_vector(self);
  new_sizes.insert(new_sizes.begin() + dim, 1);

  return repack_with_sizes(self, new_sizes);
}

Tensor slice_kernel(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16 ||
          self.dtype() == DType::Int32,
      "Vulkan slice supports Float32, Float16 and Int32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan slice supports 1d to 4d tensors");
  TP_CHECK(step > 0, "Vulkan slice: step must be positive");

  const int64_t ndim = self.dim();
  dim = dim < 0 ? dim + ndim : dim;
  TP_CHECK(dim >= 0 && dim < ndim, "Vulkan slice: dim out of range");

  const int64_t length = self.size(dim);
  const int64_t s = start.value_or(0) < 0
      ? start.value_or(0) + length
      : start.value_or(0);
  const int64_t e = end.value_or(length) < 0
      ? end.value_or(length) + length
      : end.value_or(length);
  const int64_t clamped_start = std::clamp<int64_t>(s, 0, length);
  const int64_t clamped_end = std::clamp<int64_t>(e, 0, length);

  const int64_t out_len =
      (clamped_end - clamped_start + step - 1) / step;

  std::vector<int64_t> new_sizes = shape_vector(self);
  new_sizes[static_cast<size_t>(dim)] = out_len;

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_output{context, new_sizes, self.dtype()};

  if (v_input.storage_type() == api::StorageType::BUFFER) {
    // A linear payload must be sliced through the strided-view gather,
    // whether the dense result lands in buffer or image storage: the
    // sliced axis keeps out_len elements and its stride is scaled by the
    // step, while the storage offset advances by clamped_start elements along
    // that axis.
    const std::vector<int64_t> self_strides = self.strides();
    std::vector<int64_t> out_strides(
        self_strides.begin(), self_strides.end());
    out_strides[static_cast<size_t>(dim)] *= step;
    const int64_t offset =
        self.unsafeGetTensorImpl()->storage_offset() +
        clamped_start * self_strides[static_cast<size_t>(dim)];
    return materialize_view(self, new_sizes, out_strides, offset);
  }

  const struct SliceBlock final {
    ivec4 out_sizes;
    int axis;
    int start;
    int step;
    int removed;
    int in_c_depth;
    int out_c_depth;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      static_cast<int32_t>(ndim - 1 - dim),
      static_cast<int32_t>(clamped_start),
      static_cast<int32_t>(step),
      0,
      c_depth_of(v_input.sizes()),
      c_depth_of(v_output.sizes()),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const api::ShaderInfo shader = self.dtype() == DType::Int32
      ? VK_KERNEL(slice_int32)
      : kernel_for("slice", self.dtype());
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

Tensor select_kernel(const Tensor& self, int64_t dim, int64_t index) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan select supports Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan select supports 1d to 4d tensors");

  const int64_t ndim = self.dim();
  dim = dim < 0 ? dim + ndim : dim;
  TP_CHECK(dim >= 0 && dim < ndim, "Vulkan select: dim out of range");

  const int64_t length = self.size(dim);
  TP_CHECK(
      index >= -length && index < length,
      "Vulkan select: index out of range");
  if (index < 0) {
    index += length;
  }

  std::vector<int64_t> new_sizes = shape_vector(self);
  new_sizes.erase(new_sizes.begin() + dim);

  api::Context* const context = api::context();

  api::vTensor v_input = convert(self);
  api::vTensor v_output{context, new_sizes, self.dtype()};

  if (v_input.storage_type() == api::StorageType::BUFFER) {
    // Select over a linear payload is a strided view: the selected axis is
    // dropped from the output strides and the storage offset advances by
    // index elements along it.
    const std::vector<int64_t> self_strides = self.strides();
    std::vector<int64_t> out_strides(
        self_strides.begin(), self_strides.end());
    out_strides.erase(out_strides.begin() + dim);
    const int64_t offset =
        self.unsafeGetTensorImpl()->storage_offset() +
        index * self_strides[static_cast<size_t>(dim)];
    return materialize_view(self, new_sizes, out_strides, offset);
  }

  const struct SliceBlock final {
    ivec4 out_sizes;
    int axis;
    int start;
    int step;
    int removed;
    int in_c_depth;
    int out_c_depth;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      static_cast<int32_t>(ndim - 1 - dim),
      static_cast<int32_t>(index),
      1,
      1,
      c_depth_of(v_input.sizes()),
      c_depth_of(v_output.sizes()),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(slice), pipeline_barrier, v_output.extents(),
      adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
  TP_CHECK(!tensors.empty(), "Vulkan cat expects a non-empty tensor list");
  if (tensors.size() == 1) {
    return tensors[0].clone();
  }

  const int64_t ndim = tensors[0].dim();
  dim = dim < 0 ? dim + ndim : dim;
  TP_CHECK(
      dim >= 0 && dim < ndim, "Vulkan cat: dim out of range");

  for (const Tensor& t : tensors) {
    TP_CHECK(
        t.dim() == ndim, "Vulkan cat: tensors must have the same rank");
    TP_CHECK(
        t.dtype() == tensors[0].dtype(),
        "Vulkan cat: tensors must have the same dtype");
  }

  std::vector<int64_t> new_sizes = shape_vector(tensors[0]);
  new_sizes[static_cast<size_t>(dim)] = 0;
  for (const Tensor& t : tensors) {
    new_sizes[static_cast<size_t>(dim)] += t.size(dim);
    if (&t != &tensors[0]) {
      for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) {
          TP_CHECK(
              t.size(d) == tensors[0].size(d),
              "Vulkan cat: tensor sizes must match outside the concat dim");
        }
      }
    }
  }

  api::Context* const context = api::context();

  api::vTensor v_output{context, new_sizes, tensors[0].dtype()};

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_CHECK(
        tensors[0].dtype() == DType::Float32 ||
            tensors[0].dtype() == DType::Int32,
        "Vulkan cat over buffer storage supports Float32 and Int32 only");
    TP_CHECK(
        ndim >= 1 && ndim <= 4,
        "Vulkan cat supports 1d to 4d tensors");
    const bool is_int32 = tensors[0].dtype() == DType::Int32;
    const api::ShaderInfo shader = is_int32
        ? VK_KERNEL(cat_int32_buffer_to_buffer)
        : kernel_for("cat_buffer_to_buffer", tensors[0].dtype());

    const auto prepad = [](const std::vector<int64_t>& v) {
      std::vector<int64_t> out(4, 1);
      const int64_t offset_slots = 4 - static_cast<int64_t>(v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        out[static_cast<size_t>(offset_slots + static_cast<int64_t>(i))] =
            v[i];
      }
      return out;
    };

    const std::vector<int64_t> out_pre = prepad(
        std::vector<int64_t>(v_output.sizes().begin(), v_output.sizes().end()));
    const int32_t axis = static_cast<int32_t>(ndim - 1 - dim);
    const int32_t out_c_depth =
        static_cast<int32_t>((out_pre[1] + 3) / 4);

    // One linear dispatch per input: the shader scatters each input's
    // elements into the row-major output buffer, shifting only the
    // concatenated axis by the running offset.  Passes reuse the barrier
    // object and declare read/write on the output to serialize through the
    // barrier system, mirroring the image path.
    api::PipelineBarrier pipeline_barrier{};
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
      api::vTensor v_src = convert(t);
      // A small texture input has no linear buffer backing; copy it into a
      // dense buffer first so the linear cat shader can read it.
      if (v_src.storage_type() == api::StorageType::TEXTURE_3D) {
        v_src = to_buffer_storage_dispatch(context, v_src);
      }
      const std::vector<int64_t> in_pre = prepad(
          std::vector<int64_t>(v_src.sizes().begin(), v_src.sizes().end()));
      const std::vector<int64_t> in_strides =
          SizesAndStrides::compute_contiguous_strides(in_pre);

      const struct CatBufferBlock final {
        ivec4 out_sizes;
        ivec4 in_sizes;
        ivec4 in_strides;
        int32_t axis;
        int32_t offset;
        int32_t out_c_depth;
        int32_t pad; // std140 rounds the uniform block up to 64 bytes
      } block{
          ivec4(out_pre[0], out_pre[1], out_pre[2], out_pre[3]),
          ivec4(in_pre[0], in_pre[1], in_pre[2], in_pre[3]),
          ivec4(in_strides[0], in_strides[1], in_strides[2], in_strides[3]),
          axis,
          static_cast<int32_t>(offset),
          out_c_depth,
          0,
      };

      api::UniformParamsBuffer params(context, block);

      const api::utils::uvec3 grid{
          static_cast<uint32_t>(out_pre[3]),
          static_cast<uint32_t>(out_pre[2]),
          static_cast<uint32_t>(out_pre[0]) *
              static_cast<uint32_t>(out_c_depth),
      };

      context->submit_compute_job(
          shader, pipeline_barrier, grid,
          adaptive_work_group_size(grid), VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
          v_src.buffer(
              pipeline_barrier, api::PipelineStage::COMPUTE,
              api::MemoryAccessType::READ),
          params.buffer());

      offset += t.size(dim);
    }
    return convert(v_output);
  }

  // One gather dispatch per input: the shader maps each output texel back
  // to its input coordinates in innermost-first axis order, so inputs whose
  // batch or channel slots land inside another input's texel (3d inputs
  // concatenated along the batch axis) still reach the right planes.  When
  // the channel-axis offset falls mid-texel, the four lane writes stay
  // pairwise disjoint, so the per-pass imageStore never overlaps a previous
  // pass; every pass declares READ|WRITE on the output to serialize the
  // passes through the barrier system.
  api::PipelineBarrier pipeline_barrier{};

  int64_t offset = 0;
  for (const Tensor& t : tensors) {
    api::vTensor v_src = convert(t);

    const struct CatBlock final {
      ivec4 out_sizes;
      ivec4 in_sizes;
      int axis;
      int offset;
      int in_c_depth;
      int out_c_depth;
    } block{
        make_whcn_ivec4(v_output.sizes()),
        make_whcn_ivec4(v_src.sizes()),
        static_cast<int32_t>(ndim - 1 - dim),
        static_cast<int32_t>(offset),
        c_depth_of(v_src.sizes()),
        c_depth_of(v_output.sizes()),
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL(cat), pipeline_barrier, v_output.extents(),
        adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    offset += t.size(dim);
  }

  return convert(v_output);
}

Tensor stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) {
  TP_CHECK(!tensors.empty(), "Vulkan stack expects a non-empty tensor list");

  const int64_t ndim = tensors[0].dim();
  dim = dim < 0 ? dim + ndim + 1 : dim;
  TP_CHECK(
      dim >= 0 && dim <= ndim, "Vulkan stack: dim out of range");
  TP_CHECK(
      ndim + 1 <= 4,
      "Vulkan stack: result would exceed the 4d backend limit");

  std::vector<Tensor> unsqueezed;
  unsqueezed.reserve(tensors.size());
  for (const Tensor& t : tensors) {
    unsqueezed.push_back(unsqueeze_kernel(t, dim));
  }
  return cat_kernel(unsqueezed, dim);
}

/*
 * Tiling via repeated concatenation, one output axis at a time.  The
 * source is padded to the repeat rank with leading singleton axes first;
 * short repeat lists are promoted with leading ones by the tile() entry,
 * matching the public tiling contract.
 */
Tensor repeat_kernel(const Tensor& self, const std::vector<int64_t>& repeats) {
  TP_CHECK(
      self.dim() <= 4,
      "Vulkan repeat only supports tensors <= 4 dimensions");
  const int64_t in_ndims = self.dim();
  const int64_t out_ndims = static_cast<int64_t>(repeats.size());
  TP_CHECK(
      out_ndims >= in_ndims,
      "Number of dimensions of repeat dims can not be smaller than "
      "number of dimensions of tensor");
  const int64_t add_ndims = out_ndims - in_ndims;

  Tensor tensor_to_repeat = clone_kernel(self);
  for (int64_t i = 0; i < add_ndims; ++i) {
    tensor_to_repeat = unsqueeze_kernel(tensor_to_repeat, 0);
  }

  for (int64_t i = 0; i < out_ndims; ++i) {
    if (repeats[i] == 1) {
      continue;
    }
    TP_CHECK(repeats[i] >= 0, "Vulkan repeat: repeats must be non-negative");
    std::vector<Tensor> tensor_seq_to_concat;
    tensor_seq_to_concat.reserve(static_cast<size_t>(repeats[i]));
    for (int64_t k = 0; k < repeats[i]; ++k) {
      tensor_seq_to_concat.push_back(clone_kernel(tensor_to_repeat));
    }
    tensor_to_repeat = cat_kernel(tensor_seq_to_concat, i);
  }
  return tensor_to_repeat;
}

Tensor tile_kernel(const Tensor& self, const std::vector<int64_t>& dims) {
  const int64_t size_diff = self.dim() - static_cast<int64_t>(dims.size());
  if (size_diff > 0) {
    std::vector<int64_t> new_repeats(static_cast<size_t>(size_diff), 1);
    new_repeats.insert(new_repeats.end(), dims.begin(), dims.end());
    return repeat_kernel(self, new_repeats);
  }
  return repeat_kernel(self, dims);
}

namespace {

int64_t wrap_dim_checked(int64_t dim, int64_t ndim) {
  const int64_t min = -ndim;
  const int64_t max = ndim - 1;
  TP_CHECK(
      dim >= min && dim <= max,
      "Dimension out of range (expected to be in range of [", min, ", ",
      max, "], but got ", dim, ")");
  return dim < 0 ? dim + ndim : dim;
}

} // namespace

/*
 * Chunking is repeated slicing; unbinding is repeated selection.  Both ride
 * the strided-view materialization the slice/select entries own.
 */
std::vector<Tensor> split_sizes_kernel(
    const Tensor& self,
    const std::vector<int64_t>& split_sizes,
    int64_t dim) {
  TP_CHECK(
      self.dim() > 0, "split expects at least a 1-dimensional tensor");
  dim = wrap_dim_checked(dim, self.dim());
  const int64_t dim_size = self.size(dim);

  std::vector<Tensor> result;
  result.reserve(split_sizes.size());
  int64_t start_idx = 0;
  for (const int64_t length : split_sizes) {
    TP_CHECK(
        length >= 0,
        "split_with_sizes expects split_sizes have only non-negative entries");
    result.push_back(slice_kernel(self, dim, start_idx, start_idx + length, 1));
    start_idx += length;
  }
  TP_CHECK(
      start_idx == dim_size,
      "split_with_sizes expects split_sizes to sum exactly to the input "
      "tensor's size at dimension ", dim);
  return result;
}

std::vector<Tensor> split_kernel(
    const Tensor& self,
    int64_t split_size,
    int64_t dim) {
  TP_CHECK(
      self.dim() > 0, "split expects at least a 1-dimensional tensor");
  TP_CHECK(
      split_size >= 0,
      "split expects split_size be non-negative, but got split_size=",
      split_size);
  dim = wrap_dim_checked(dim, self.dim());
  const int64_t dim_size = self.size(dim);
  TP_CHECK(
      split_size > 0 || dim_size == 0,
      "split_size can only be 0 if dimension size is 0, but got dimension "
      "size of ", dim_size);

  // split_size == 0 && dim_size == 0 yields exactly one empty split.
  int64_t num_splits = 1;
  int64_t last_split_size = 0;
  if (split_size != 0) {
    const int64_t quotient = dim_size / split_size;
    const int64_t remainder = dim_size % split_size;
    num_splits = std::max<int64_t>(quotient + (remainder != 0 ? 1 : 0), 1);
    last_split_size =
        (dim_size != 0 && remainder == 0) ? split_size : remainder;
  }
  std::vector<Tensor> result;
  result.reserve(static_cast<size_t>(num_splits));
  for (int64_t i = 0; i < num_splits; ++i) {
    const int64_t length =
        i < num_splits - 1 ? split_size : last_split_size;
    result.push_back(
        slice_kernel(self, dim, i * split_size, i * split_size + length, 1));
  }
  return result;
}

std::vector<Tensor> chunk_kernel(
    const Tensor& self,
    int64_t chunks,
    int64_t dim) {
  TP_CHECK(
      self.dim() > 0, "chunk expects at least a 1-dimensional tensor");
  TP_CHECK(chunks > 0, "chunk expects `chunks` to be greater than 0");
  dim = wrap_dim_checked(dim, self.dim());
  const int64_t dim_size = self.size(dim);
  const int64_t split_size = (dim_size + chunks - 1) / chunks;
  std::vector<int64_t> split_sizes(
      static_cast<size_t>(chunks), split_size);
  // The tail chunk absorbs the remainder so the sizes sum to the axis.
  if (split_size > 0 && dim_size % split_size != 0) {
    split_sizes.back() = dim_size - split_size * (chunks - 1);
  } else if (split_size == 0) {
    split_sizes.back() = 0;
  }
  return split_sizes_kernel(self, split_sizes, dim);
}

std::vector<Tensor> unbind_kernel(const Tensor& self, int64_t dim) {
  TP_CHECK(
      self.dim() > 0,
      "Dimension specified as ", dim, " but tensor has no dimensions");
  dim = wrap_dim_checked(dim, self.dim());
  std::vector<Tensor> result;
  result.reserve(static_cast<size_t>(self.size(dim)));
  for (int64_t i = 0; i < self.size(dim); ++i) {
    result.push_back(select_kernel(self, dim, i));
  }
  return result;
}

/*
 * Axis moves are permutations: destination slots take the moved dims in
 * order, remaining slots keep their original dims in ascending order.
 */
Tensor movedim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& source,
    const std::vector<int64_t>& destination) {
  const int64_t ndim = self.dim();
  TP_CHECK(
      source.size() == destination.size(),
      "movedim: source and destination must contain the same number of dims");
  std::vector<int64_t> src(source), dst(destination);
  for (auto& d : src) d = wrap_dim_checked(d, ndim);
  for (auto& d : dst) d = wrap_dim_checked(d, ndim);
  auto all_unique = [](std::vector<int64_t> dims) {
    std::sort(dims.begin(), dims.end());
    return std::adjacent_find(dims.begin(), dims.end()) == dims.end();
  };
  TP_CHECK(all_unique(src), "movedim: repeated dim in `source`");
  TP_CHECK(all_unique(dst), "movedim: repeated dim in `destination`");
  if (ndim == 0) {
    return self;
  }

  std::vector<bool> src_seen(static_cast<size_t>(ndim), false);
  std::vector<bool> dst_seen(static_cast<size_t>(ndim), false);
  for (const int64_t d : src) src_seen[static_cast<size_t>(d)] = true;
  for (const int64_t d : dst) dst_seen[static_cast<size_t>(d)] = true;

  std::vector<int64_t> permutation(static_cast<size_t>(ndim), -1);
  for (size_t k = 0; k < src.size(); ++k) {
    permutation[static_cast<size_t>(dst[k])] = src[k];
  }
  int64_t cursor = 0;
  for (int64_t i = 0; i < ndim; ++i) {
    if (!dst_seen[static_cast<size_t>(i)]) {
      while (src_seen[static_cast<size_t>(cursor)]) ++cursor;
      permutation[static_cast<size_t>(i)] = cursor++;
    }
  }
  return permute_kernel(self, permutation);
}

// narrow is slice over a fixed-length window starting at `start`.
Tensor narrow_kernel(const Tensor& self, int64_t dim, int64_t start,
                     int64_t length) {
  return slice_kernel(self, dim, start, start + length, 1);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ShapeKernels) {
  m.impl("transpose", &tensorplay::vulkan::ops::transpose_kernel);
  m.impl("t", &tensorplay::vulkan::ops::t_kernel);
  m.impl("permute", &tensorplay::vulkan::ops::permute_kernel);
  m.impl("squeeze.dim", &tensorplay::vulkan::ops::squeeze_dim_kernel);
  m.impl("unsqueeze", &tensorplay::vulkan::ops::unsqueeze_kernel);
  m.impl("slice.Tensor", &tensorplay::vulkan::ops::slice_kernel);
  m.impl("narrow", &tensorplay::vulkan::ops::narrow_kernel);
  m.impl("select.int", &tensorplay::vulkan::ops::select_kernel);
  m.impl("cat", &tensorplay::vulkan::ops::cat_kernel);
  m.impl("stack", &tensorplay::vulkan::ops::stack_kernel);
  m.impl("repeat", &tensorplay::vulkan::ops::repeat_kernel);
  m.impl("tile", &tensorplay::vulkan::ops::tile_kernel);
  m.impl("split.sizes", &tensorplay::vulkan::ops::split_sizes_kernel);
  m.impl("split", &tensorplay::vulkan::ops::split_kernel);
  m.impl("chunk", &tensorplay::vulkan::ops::chunk_kernel);
  m.impl("unbind", &tensorplay::vulkan::ops::unbind_kernel);
  m.impl("movedim", &tensorplay::vulkan::ops::movedim_kernel);
}

#endif /* USE_VULKAN */
