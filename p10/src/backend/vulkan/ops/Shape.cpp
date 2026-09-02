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
  api::Context* const context = api::context();
  api::vTensor v_src = convert(self);

  api::vTensor v_dst{context, new_sizes, self.dtype()};

  // The staging buffer is addressed in the packed texture layout, so its
  // size follows the GPU element count (channels padded to lanes) rather
  // than the logical count.
  api::StorageBuffer staging(
      context, v_src.texture_dtype(),
      std::max(v_src.gpu_numel(), v_dst.gpu_numel()));

  utils::pack_vtensor_to_staging(v_src, staging.buffer(), VK_NULL_HANDLE);
  utils::pack_staging_to_vtensor(staging.buffer(), v_dst);

  return convert(v_dst);
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
  api::vTensor v_output{
      context, permuted_sizes(self, perm), self.dtype()};

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan permute requires texture storage");
  }

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
      self.dtype() == DType::Float32,
      "Vulkan slice supports Float32 tensors only");
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

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan slice requires texture storage");
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

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan select requires texture storage");
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
    TP_THROW(NotImplementedError, "Vulkan cat requires texture storage");
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

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ShapeKernels) {
  m.impl("transpose", &tensorplay::vulkan::ops::transpose_kernel);
  m.impl("t", &tensorplay::vulkan::ops::t_kernel);
  m.impl("permute", &tensorplay::vulkan::ops::permute_kernel);
  m.impl("squeeze.dim", &tensorplay::vulkan::ops::squeeze_dim_kernel);
  m.impl("unsqueeze", &tensorplay::vulkan::ops::unsqueeze_kernel);
  m.impl("slice", &tensorplay::vulkan::ops::slice_kernel);
  m.impl("select.int", &tensorplay::vulkan::ops::select_kernel);
  m.impl("cat", &tensorplay::vulkan::ops::cat_kernel);
  m.impl("stack", &tensorplay::vulkan::ops::stack_kernel);
}

#endif /* USE_VULKAN */
