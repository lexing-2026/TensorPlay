#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

// Materializes a tensor's shape into a plain sizes vector.  The shape
// accessor returns a value object, so the copy must be taken once instead
// of reaching through repeated temporaries.
std::vector<int64_t> shape_vector(const Tensor& t) {
  const Size shape = t.shape();
  return std::vector<int64_t>(shape.begin(), shape.end());
}

//
// Reduction geometry for one dispatch: a single reduced axis expressed in
// innermost-first order, the element count along it, and the output sizes
// the dispatch writes into.
//
struct ReduceStep final {
  int axis; // innermost-first axis index
  int64_t count; // elements along the reduced axis
  std::vector<int64_t> out_sizes;
};

// One single-axis reduce pass.  The output keeps every axis, with the
// reduced axis collapsed to length one; squeezing (when the public call
// asked for it) happens as a layout repack after all axes are reduced.
Tensor reduce_one_axis(
    const Tensor& input,
    int64_t axis_from_left,
    bool mean_mode,
    int64_t correction_for_var,
    int var_mode) {
  api::Context* const context = api::context();

  api::vTensor v_input = convert(input);

  const int64_t ndim = input.dim();
  const int axis_innermost = static_cast<int>(ndim - 1 - axis_from_left);

  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;

  api::vTensor v_output{
      context,
      out_sizes,
      input.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    TP_THROW(NotImplementedError, "Vulkan reductions require texture storage");
  }

  if (var_mode) {
    const struct VarAxisBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      int axis;
      int in_c_depth;
      int out_c_depth;
      int count;
      int correction;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(out_sizes),
        axis_innermost,
        c_depth_of(v_input.sizes()),
        c_depth_of(out_sizes),
        static_cast<int32_t>(input.size(axis_from_left)),
        static_cast<int32_t>(correction_for_var),
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        VK_KERNEL(var_axis), pipeline_barrier, v_output.extents(),
        adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    const struct SumAxisBlock final {
      ivec4 in_sizes;
      ivec4 out_sizes;
      int axis;
      int in_c_depth;
      int out_c_depth;
      float scale;
      int fill1;
    } block{
        make_whcn_ivec4(v_input.sizes()),
        make_whcn_ivec4(out_sizes),
        axis_innermost,
        c_depth_of(v_input.sizes()),
        c_depth_of(out_sizes),
        mean_mode ? 1.0f / static_cast<float>(input.size(axis_from_left))
                  : 1.0f,
        0,
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        VK_KERNEL(sum_axis), pipeline_barrier, v_output.extents(),
        adaptive_work_group_size(v_output.extents()), VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_output);
}


// Streams a texture-backed tensor through its linear host-order
// representation into a fresh tensor with new sizes.  Used to collapse the
// reduced axes: the keepdim-style intermediate and the squeezed result
// differ in how their extents map onto the texture, so a real data move is
// required.
Tensor repack_with_sizes(const Tensor& src, std::vector<int64_t> final_sizes) {
  api::Context* const context = api::context();

  api::vTensor v_src = convert(src);
  api::vTensor v_dst{context, final_sizes, src.dtype()};

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

std::vector<int64_t> normalized_dims(
    const Tensor& input,
    const std::vector<int64_t>& dims) {
  std::set<int64_t> unique;
  for (const int64_t d : dims) {
    int64_t axis = d < 0 ? d + input.dim() : d;
    TP_CHECK(
        axis >= 0 && axis < input.dim(),
        "Vulkan reductions: dim out of range");
    unique.insert(axis);
  }
  return std::vector<int64_t>(unique.begin(), unique.end());
}

Tensor reduce_impl(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    bool mean_mode,
    bool var_mode,
    int64_t correction) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan reductions support Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan reductions support 1d to 4d tensors");
  TP_CHECK(!dims_in.empty(), "Vulkan reductions require at least one dim");
  TP_CHECK(self.numel() > 0, "Vulkan reductions do not support empty tensors");

  const std::vector<int64_t> dims = normalized_dims(self, dims_in);

  Tensor current = self;
  for (const int64_t axis : dims) {
    current = reduce_one_axis(
        current, axis, mean_mode, correction, var_mode);
  }

  if (keepdim || self.dim() == 1) {
    return current;
  }

  // Squeeze the reduced axes: fold consecutive length-one axes out of the
  // shape and move the payload into the squeezed layout.
  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t i = 0; i < self.dim(); ++i) {
    const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (!reduced || current.size(i) != 1) {
      final_sizes.push_back(current.size(i));
    }
  }

  return repack_with_sizes(current, std::move(final_sizes));
}

} // namespace

Tensor sum_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    DType dtype) {
  (void)dtype;
  return reduce_impl(self, dims, keepdim, false, false, 0);
}

Tensor mean_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    DType dtype) {
  (void)dtype;
  return reduce_impl(self, dims, keepdim, true, false, 0);
}

namespace {

//
// Broadcast elementwise helpers used by the variance composition.  The
// patterns mirror the binary op shader conventions: out = a - b with b
// broadcast, and out = a * a elementwise.
//
Tensor broadcast_sub(const Tensor& a, const Tensor& b) {
  api::Context* const context = api::context();

  api::vTensor v_a = convert(a);
  api::vTensor v_b = convert(b);
  api::vTensor v_out{context, v_a.sizes(), a.dtype()};

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_out.sizes()),
      make_whcn_ivec4(v_a.sizes()),
      make_whcn_ivec4(v_b.sizes()),
      1.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(sub), pipeline_barrier, v_out.extents(),
      adaptive_work_group_size(v_out.extents()), VK_NULL_HANDLE,
      v_out.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_a.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_b.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_out);
}

Tensor square(const Tensor& a) {
  api::Context* const context = api::context();

  api::vTensor v_a = convert(a);
  api::vTensor v_out{context, v_a.sizes(), a.dtype()};

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_out.sizes()),
      make_whcn_ivec4(v_a.sizes()),
      make_whcn_ivec4(v_a.sizes()),
      1.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(mul), pipeline_barrier, v_out.extents(),
      adaptive_work_group_size(v_out.extents()), VK_NULL_HANDLE,
      v_out.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_a.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_a.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_out);
}

} // namespace

Tensor var_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dim,
    int64_t correction,
    bool keepdim) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan var supports Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan var supports 1d to 4d tensors");
  TP_CHECK(!dim.empty(), "Vulkan var requires at least one dim");
  TP_CHECK(self.numel() > 0, "Vulkan var does not support empty tensors");

  const std::vector<int64_t> dims = normalized_dims(self, dim);

  // Element count along the reduced span: the product of the reduced axes'
  // lengths; the divisor of the two-pass variance.
  int64_t count = 1;
  for (const int64_t axis : dims) {
    count *= self.size(axis);
  }

  // Two-pass variance over the whole reduced span: a keepdim mean
  // broadcasts against the input exactly, so the elementwise kernels apply
  // to any axis set and the sum aggregates the squared deviations.
  Tensor mean = reduce_impl(self, dims, /*keepdim=*/true, true, false, 0);
  Tensor centered_sq = square(broadcast_sub(self, mean));
  Tensor summed =
      reduce_impl(centered_sq, dims, keepdim, false, false, 0);

  const double denom = static_cast<double>(
      std::max<int64_t>(count - correction, 1));
  return summed.mul(Scalar(1.0 / denom));
}

Tensor std_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dim,
    int64_t correction,
    bool keepdim) {
  return var_dim_kernel(self, dim, correction, keepdim).sqrt();
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, ReductionKernels) {
  m.impl("sum.dim_IntList", &tensorplay::vulkan::ops::sum_dim_kernel);
  m.impl("mean.dim", &tensorplay::vulkan::ops::mean_dim_kernel);
  m.impl("var.dim", &tensorplay::vulkan::ops::var_dim_kernel);
  m.impl("std.dim", &tensorplay::vulkan::ops::std_dim_kernel);
}

#endif /* USE_VULKAN */
