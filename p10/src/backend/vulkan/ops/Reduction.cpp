#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "Utils.h"
#include "../../../../include/Utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha);

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

// One linear reduce pass: `rows` sort rows of `tile` elements each, where
// consecutive elements of a row stride by `inner`.  Sums and means write the
// scaled row sum; variances write the corrected variance built from the row's
// raw moments.
void reduce_axis_buffer_pass(
    api::Context* const context,
    api::vTensor& output,
    api::vTensor& input,
    int64_t tile,
    int64_t inner,
    int64_t rows,
    int32_t correction,
    float scale,
    bool var_mode) {
  const struct ReduceAxisBufferBlock final {
    int32_t axis_size;
    int32_t inner;
    int32_t rows;
    int32_t total;
    int32_t correction;
    float scale;
  } block{
      static_cast<int32_t>(tile),
      static_cast<int32_t>(inner),
      static_cast<int32_t>(rows),
      static_cast<int32_t>(input.numel()),
      correction,
      scale,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      var_mode ? VK_KERNEL(var_axis_buffer) : VK_KERNEL(sum_axis_buffer),
      pipeline_barrier,
      {static_cast<uint32_t>(rows), 1u, 1u},
      {256u, 1u, 1u},
      VK_NULL_HANDLE,
      output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      input.buffer(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());
}

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

// Buffer-stored inputs (any axis beyond the 3-D image limit) reduce
  // through a linear row-walking variant; the reduced output is kept in
  // buffer storage.  Float16 buffer payloads are byte-packed with no
  // buffer-writing reduction yet.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    TP_CHECK(
        input.dtype() == DType::Float32,
        "Vulkan reductions over Float16 buffer storage are not supported "
        "yet");
    api::vTensor v_output{
        context, out_sizes, input.dtype(), api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const int64_t axis_size = input.size(axis_from_left);
    int64_t inner = 1;
    for (int64_t d = axis_from_left + 1; d < ndim; ++d) {
      inner *= input.size(d);
    }
    const int64_t rows = input.numel() / axis_size;
    const int32_t correction =
        var_mode ? static_cast<int32_t>(correction_for_var) : 0;
    const float scale =
        mean_mode ? 1.0f / static_cast<float>(axis_size) : 1.0f;

    // A reduce that collapses the whole payload into one row would hand every
    // element to a single thread, so split it into fixed tiles, sum each tile,
    // and add the tile results in a second pass.  Variance keeps its raw
    // moments inside the row walk and stays on the single pass.
    constexpr int64_t kTileElements = 256;
    if (!var_mode && rows == 1 && axis_size > kTileElements) {
      const int64_t tiles =
          (axis_size + kTileElements - 1) / kTileElements;
      api::vTensor v_tiles{
          context,
          std::vector<int64_t>{tiles},
          input.dtype(),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
      reduce_axis_buffer_pass(
          context, v_tiles, v_input, kTileElements, 1, tiles, 0, 1.0f, false);
      reduce_axis_buffer_pass(
          context, v_output, v_tiles, tiles, 1, 1, 0, scale, false);
      return convert(v_output);
    }

    reduce_axis_buffer_pass(
        context, v_output, v_input, axis_size, inner, rows, correction, scale,
        var_mode);
    return convert(v_output);
  }

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
        kernel_for("var_axis", input.dtype()), pipeline_barrier, v_output.extents(),
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
        kernel_for("sum_axis", input.dtype()), pipeline_barrier, v_output.extents(),
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
  return reshape_kernel(src, final_sizes);
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
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan reductions support Float32 and Float16 tensors only");
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

  if (keepdim) {
    return current;
  }

  // Squeeze the reduced axes: fold consecutive length-one axes out of the
  // shape and move the payload into the squeezed layout.  Reducing the only
  // axis of a 1d input squeezes everything, producing a scalar (0d result).
  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t i = 0; i < self.dim(); ++i) {
    const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (!reduced || current.size(i) != 1) {
      final_sizes.push_back(current.size(i));
    }
  }

  if (final_sizes.empty()) {
    return repack_with_sizes(current, final_sizes);
  }

  return repack_with_sizes(current, std::move(final_sizes));
}

enum class ExtremumKind { kMax, kMin, kProd };

const char* extremum_shader_name(ExtremumKind kind) {
  switch (kind) {
    case ExtremumKind::kMax:
      return "reduce_max";
    case ExtremumKind::kMin:
      return "reduce_min";
    case ExtremumKind::kProd:
      return "reduce_prod";
  }
  return "reduce_max";
}

// One extremum pass over a linear payload: `rows` sort rows of `tile`
// elements each, where consecutive elements of a row stride by `inner`.
void extremum_buffer_pass(
    api::Context* const context,
    api::vTensor& output,
    api::vTensor& input,
    int64_t tile,
    int64_t inner,
    int64_t rows,
    const char* shader) {
  const struct ExtremumBufferBlock final {
    int32_t axis_size;
    int32_t inner;
    int32_t rows;
    int32_t total;
    int32_t fill;
    int32_t fill1;
  } block{
      static_cast<int32_t>(tile),
      static_cast<int32_t>(inner),
      static_cast<int32_t>(rows),
      static_cast<int32_t>(input.numel()),
      0,
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader),
      pipeline_barrier,
      {static_cast<uint32_t>(rows), 1u, 1u},
      {256u, 1u, 1u},
      VK_NULL_HANDLE,
      output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      input.buffer(
          pipeline_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());
}

Tensor extremum_one_axis(
    const Tensor& input,
    int64_t axis_from_left,
    ExtremumKind kind) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);

  const int axis_innermost =
      static_cast<int>(input.dim() - 1 - axis_from_left);
  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;

  // Buffer payloads reduce through the linear variant; only the extremum
  // kinds have one, so a product over buffer storage still reports the
  // texture requirement below.
  const char* buffer_shader = kind == ExtremumKind::kMax
      ? "reduce_max_buffer"
      : kind == ExtremumKind::kMin ? "reduce_min_buffer" : nullptr;
  if (v_input.storage_type() == api::StorageType::BUFFER &&
      buffer_shader != nullptr) {
    api::vTensor v_output{
        context,
        out_sizes,
        DType::Float32,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

    const int64_t axis_size = input.size(axis_from_left);
    int64_t inner = 1;
    for (int64_t d = axis_from_left + 1; d < input.dim(); ++d) {
      inner *= input.size(d);
    }
    const int64_t rows = input.numel() / axis_size;

    // A row per invocation is already parallel when the reduce leaves many
    // rows behind.  One that collapses the whole payload into a single row
    // would hand every element to one thread, so split it into fixed tiles,
    // reduce each tile in place, and reduce the tile results afterwards.
    constexpr int64_t kTileElements = 256;
    if (rows == 1 && axis_size > kTileElements) {
      const int64_t tiles =
          (axis_size + kTileElements - 1) / kTileElements;
      api::vTensor v_tiles{
          context,
          std::vector<int64_t>{tiles},
          DType::Float32,
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
      extremum_buffer_pass(
          context, v_tiles, v_input, kTileElements, 1, tiles, buffer_shader);
      extremum_buffer_pass(
          context, v_output, v_tiles, tiles, 1, 1, buffer_shader);
      return convert(v_output);
    }

    extremum_buffer_pass(
        context, v_output, v_input, axis_size, inner, rows, buffer_shader);
    return convert(v_output);
  }

  api::vTensor v_output{context, out_sizes, DType::Float32};
  TP_CHECK(
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
          v_output.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan reductions require texture storage");

  const struct ExtremumBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int axis;
    int in_c_depth;
    int out_c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(out_sizes),
      axis_innermost,
      c_depth_of(v_input.sizes()),
      c_depth_of(out_sizes),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(extremum_shader_name(kind)),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor extremum_impl(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    ExtremumKind kind) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan reductions support Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan reductions support 1d to 4d tensors");
  TP_CHECK(self.numel() > 0, "Vulkan reductions do not support empty tensors");

  std::vector<int64_t> dims = dims_in;
  if (dims.empty()) {
    dims.reserve(static_cast<size_t>(self.dim()));
    for (int64_t d = 0; d < self.dim(); ++d) {
      dims.push_back(d);
    }
  }
  dims = normalized_dims(self, dims);
  TP_CHECK(!dims.empty(), "Vulkan reductions require at least one dim");

  Tensor current = self;
  for (const int64_t axis : dims) {
    current = extremum_one_axis(current, axis, kind);
  }
  if (keepdim) {
    return current;
  }

  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t axis = 0; axis < self.dim(); ++axis) {
    if (std::find(dims.begin(), dims.end(), axis) == dims.end()) {
      final_sizes.push_back(current.size(axis));
    }
  }
  return repack_with_sizes(current, std::move(final_sizes));
}

Tensor integer_prod_one_axis(
    const Tensor& input,
    int64_t axis_from_left) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);

  const int axis_innermost =
      static_cast<int>(input.dim() - 1 - axis_from_left);
  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;

  api::vTensor v_output{context, out_sizes, DType::Int32};
  TP_CHECK(
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
          v_output.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan integer products require texture storage");

  const struct IntegerProdBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int axis;
    int in_c_depth;
    int out_c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(out_sizes),
      axis_innermost,
      c_depth_of(v_input.sizes()),
      c_depth_of(out_sizes),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL(reduce_prod_i32),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor integer_prod_impl(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim) {
  TP_CHECK(
      self.dtype() == DType::Int32,
      "Vulkan integer products require Int32 tensors");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan integer products support 1d to 4d tensors");
  TP_CHECK(self.numel() > 0, "Vulkan integer products do not support empty tensors");

  std::vector<int64_t> dims = dims_in;
  if (dims.empty()) {
    dims.reserve(static_cast<size_t>(self.dim()));
    for (int64_t d = 0; d < self.dim(); ++d) {
      dims.push_back(d);
    }
  }
  dims = normalized_dims(self, dims);
  TP_CHECK(!dims.empty(), "Vulkan integer products require at least one dim");

  Tensor current = self;
  for (const int64_t axis : dims) {
    current = integer_prod_one_axis(current, axis);
  }
  if (keepdim) {
    return current;
  }

  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t axis = 0; axis < self.dim(); ++axis) {
    if (std::find(dims.begin(), dims.end(), axis) == dims.end()) {
      final_sizes.push_back(current.size(axis));
    }
  }
  return repack_with_sizes(current, std::move(final_sizes));
}

Tensor arg_one_axis(
    const Tensor& input,
    int64_t axis_from_left,
    bool greater) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;

  // Buffer-stored inputs (any axis beyond the 3-D image limit) reduce
  // through a linear variant; the reduced output is kept in buffer storage.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    const int64_t axis_size = input.size(axis_from_left);
    int64_t inner = 1;
    for (int64_t d = axis_from_left + 1; d < input.dim(); ++d) {
      inner *= input.size(d);
    }
    const int64_t rows = input.numel() / axis_size;
    api::vTensor v_output{
        context, out_sizes, DType::Int32, api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    const struct ArgBufferBlock final {
      int32_t axis_size;
      int32_t inner;
      int32_t rows;
      int32_t greater;
    } block{
        static_cast<int32_t>(axis_size), static_cast<int32_t>(inner),
        static_cast<int32_t>(rows), greater ? 1 : 0,
    };
    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        greater ? VK_KERNEL(reduce_argmax_buffer)
                : VK_KERNEL(reduce_argmin_buffer),
        pipeline_barrier,
        {static_cast<uint32_t>(rows), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_output);
  }

  api::vTensor v_output{context, out_sizes, DType::Int32};
  TP_CHECK(
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
          v_output.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan index reductions require texture storage");

  const struct ArgBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int axis;
    int in_c_depth;
    int out_c_depth;
    int greater;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(out_sizes),
      static_cast<int>(input.dim() - 1 - axis_from_left),
      c_depth_of(v_input.sizes()),
      c_depth_of(out_sizes),
      greater ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      greater ? VK_KERNEL(reduce_argmax) : VK_KERNEL(reduce_argmin),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor arg_impl(
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim,
    bool greater) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan index reductions support Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan index reductions support 1d to 4d tensors");
  TP_CHECK(self.numel() > 0, "Vulkan index reductions do not support empty tensors");

  if (!dim.has_value()) {
    Tensor flat = reshape_kernel(self, {static_cast<int64_t>(self.numel())});
    Tensor reduced = arg_one_axis(flat, 0, greater);
    return repack_with_sizes(reduced, {});
  }

  int64_t axis = dim.value();
  axis = axis < 0 ? axis + self.dim() : axis;
  TP_CHECK(axis >= 0 && axis < self.dim(), "Vulkan index reduction: dim out of range");
  TP_CHECK(self.size(axis) > 0, "Vulkan index reduction requires a non-empty dim");

  Tensor reduced = arg_one_axis(self, axis, greater);
  if (keepdim) {
    return reduced;
  }
  std::vector<int64_t> final_sizes = shape_vector(self);
  final_sizes.erase(final_sizes.begin() + axis);
  return repack_with_sizes(reduced, std::move(final_sizes));
}

enum class BoolReduceKind { kAll, kAny };

const char* bool_reduce_shader_name(BoolReduceKind kind) {
  return kind == BoolReduceKind::kAll ? "reduce_all" : "reduce_any";
}

Tensor bool_reduce_one_axis(
    const Tensor& input,
    int64_t axis_from_left,
    BoolReduceKind kind) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);
  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;

  api::vTensor v_output{context, out_sizes, DType::Bool};
  const struct BoolBlock final {
    ivec4 in_sizes;
    ivec4 out_sizes;
    int axis;
    int in_c_depth;
    int out_c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      make_whcn_ivec4(out_sizes),
      static_cast<int>(input.dim() - 1 - axis_from_left),
      c_depth_of(v_input.sizes()),
      c_depth_of(out_sizes),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(bool_reduce_shader_name(kind)),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

Tensor bool_reduce_impl(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    BoolReduceKind kind) {
  TP_CHECK(self.dtype() == DType::Bool, "Vulkan boolean reductions require Bool input");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan boolean reductions support 1d to 4d tensors");
  TP_CHECK(self.numel() > 0, "Vulkan boolean reductions do not support empty tensors");

  std::vector<int64_t> dims = dims_in;
  if (dims.empty()) {
    dims.reserve(static_cast<size_t>(self.dim()));
    for (int64_t d = 0; d < self.dim(); ++d) {
      dims.push_back(d);
    }
  }
  dims = normalized_dims(self, dims);
  TP_CHECK(!dims.empty(), "Vulkan boolean reductions require at least one dim");

  Tensor current = self;
  for (const int64_t axis : dims) {
    current = bool_reduce_one_axis(current, axis, kind);
  }
  if (keepdim) {
    return current;
  }

  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t axis = 0; axis < self.dim(); ++axis) {
    if (std::find(dims.begin(), dims.end(), axis) == dims.end()) {
      final_sizes.push_back(current.size(axis));
    }
  }
  return repack_with_sizes(current, std::move(final_sizes));
}

Tensor reduce_int64(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    const char* shader) {
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4,
           "Vulkan integer reductions support 1d to 4d tensors");
  TP_CHECK(self.numel() > 0, "Vulkan integer reductions require a non-empty tensor");
  TP_CHECK(self.numel() <= std::numeric_limits<int32_t>::max(),
           "Vulkan integer reduction exceeds the index range");
  std::vector<int64_t> dims = dims_in;
  if (dims.empty()) {
    for (int64_t d = 0; d < self.dim(); ++d) dims.push_back(d);
  }
  dims = normalized_dims(self, dims);
  auto kept_sizes = shape_vector(self);
  ivec4 reduced{0, 0, 0, 0};
  int64_t span = 1;
  for (const int64_t d : dims) {
    span *= self.size(d);
    kept_sizes[d] = 1;
    reduced[static_cast<uint32_t>(self.dim() - 1 - d)] = 1;
  }
  std::vector<int64_t> output_sizes;
  for (int64_t d = 0; d < self.dim(); ++d) {
    if (keepdim || std::find(dims.begin(), dims.end(), d) == dims.end()) {
      output_sizes.push_back(kept_sizes[d]);
    }
  }
  api::Context* context = api::context();
  api::vTensor input = convert(self);
  api::vTensor output{
      context, output_sizes, DType::Int64, api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const struct Block final {
    ivec4 sizes;
    ivec4 output_sizes;
    ivec4 reduced;
    ivec4 counts;
  } block{
      make_whcn_ivec4(input.sizes()), make_whcn_ivec4(kept_sizes), reduced,
      {static_cast<int32_t>(output.numel()), static_cast<int32_t>(span), 0, 0}};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  const uint32_t groups = std::min<uint32_t>(output.numel(), 65535u);
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader), barrier,
      {groups * 64u, api::utils::div_up(static_cast<uint32_t>(output.numel()), groups), 1u},
      {256u, 1u, 1u}, VK_NULL_HANDLE,
      output.buffer(barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
      input.image(barrier, api::PipelineStage::COMPUTE), params.buffer());
  return convert(output);
}

Tensor count_impl(const Tensor& self, const std::vector<int64_t>& dims) {
  TP_CHECK(self.dtype() == DType::Float32 || self.dtype() == DType::Int32,
           "Vulkan count_nonzero supports Float32 and Int32 tensors only");
  return reduce_int64(self, dims, false,
      self.dtype() == DType::Int32 ? "count_int64_i32" : "count_int64_float");
}

Tensor bool_mask_for_reduction(const Tensor& self) {
  if (self.dtype() == DType::Bool) {
    return self;
  }
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan boolean reductions support Float32 and Bool tensors only");
  return self.ne(Scalar(0.0));
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

//
// Whole-tensor reductions fold into the axis-wise path by listing every
// axis in one call, exactly like the dim-listed entry points do.
//
Tensor sum_kernel(const Tensor& self, DType dtype) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan reductions support Float32 tensors only");
  if (self.numel() == 0) {
    return full_kernel(
        {}, Scalar(0.0), DType::Float32, Device(DeviceType::Vulkan), false);
  }
  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(self.dim()));
  for (int64_t d = 0; d < self.dim(); ++d) {
    dims.push_back(d);
  }
  return reduce_impl(self, dims, false, false, 0, 0);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan reductions support Float32 tensors only");
  Tensor summed = sum_kernel(self, dtype);
  return summed.mul(Scalar(1.0 / static_cast<double>(self.numel())));
}

Tensor max_kernel(const Tensor& self) {
  return extremum_impl(self, {}, false, ExtremumKind::kMax);
}

std::tuple<Tensor, Tensor> max_dim_kernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return {
      extremum_impl(self, {dim}, keepdim, ExtremumKind::kMax),
      arg_impl(self, dim, keepdim, true),
  };
}

Tensor min_kernel(const Tensor& self) {
  return extremum_impl(self, {}, false, ExtremumKind::kMin);
}

std::tuple<Tensor, Tensor> min_dim_kernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return {
      extremum_impl(self, {dim}, keepdim, ExtremumKind::kMin),
      arg_impl(self, dim, keepdim, false),
  };
}

Tensor amax_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim) {
  return extremum_impl(self, dims, keepdim, ExtremumKind::kMax);
}

Tensor amin_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim) {
  return extremum_impl(self, dims, keepdim, ExtremumKind::kMin);
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
  if (self.dtype() == DType::Int32) {
    if (dtype == DType::Undefined || dtype == DType::Int64) {
      return reduce_int64(self, {}, false, "prod_int64");
    }
    if (dtype == DType::Int32) return integer_prod_impl(self, {}, false);
    TP_CHECK(dtype == DType::Float32, "Unsupported Vulkan product dtype");
    return extremum_impl(self.to(dtype), {}, false, ExtremumKind::kProd);
  }
  TP_CHECK(dtype == DType::Undefined || dtype == DType::Float32,
           "Unsupported Vulkan product dtype");
  return extremum_impl(self, {}, false, ExtremumKind::kProd);
}

Tensor prod_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    DType dtype) {
  if (self.dtype() == DType::Int32) {
    if (dtype == DType::Undefined || dtype == DType::Int64) {
      return reduce_int64(self, dims, keepdim, "prod_int64");
    }
    if (dtype == DType::Int32) return integer_prod_impl(self, dims, keepdim);
    TP_CHECK(dtype == DType::Float32, "Unsupported Vulkan product dtype");
    return extremum_impl(self.to(dtype), dims, keepdim, ExtremumKind::kProd);
  }
  TP_CHECK(dtype == DType::Undefined || dtype == DType::Float32,
           "Unsupported Vulkan product dtype");
  return extremum_impl(self, dims, keepdim, ExtremumKind::kProd);
}

Tensor all_kernel(const Tensor& self) {
  return bool_reduce_impl(
      bool_mask_for_reduction(self), {}, false, BoolReduceKind::kAll);
}

Tensor all_dim_kernel(const Tensor& self, int64_t dim, bool keepdim) {
  return bool_reduce_impl(
      bool_mask_for_reduction(self), {dim}, keepdim, BoolReduceKind::kAll);
}

Tensor any_kernel(const Tensor& self) {
  return bool_reduce_impl(
      bool_mask_for_reduction(self), {}, false, BoolReduceKind::kAny);
}

Tensor any_dim_kernel(const Tensor& self, int64_t dim, bool keepdim) {
  return bool_reduce_impl(
      bool_mask_for_reduction(self), {dim}, keepdim, BoolReduceKind::kAny);
}

Tensor argmax_kernel(
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim) {
  return arg_impl(self, dim, keepdim, true);
}

Tensor argmin_kernel(
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim) {
  return arg_impl(self, dim, keepdim, false);
}

Tensor count_nonzero_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims) {
  return count_impl(self, dims);
}

Tensor cumprod_kernel(
    const Tensor& self,
    int64_t dim,
    std::optional<DType> dtype) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan cumprod supports Float32 tensors only");
  TP_CHECK(
      !dtype.has_value() || dtype.value() == DType::Float32,
      "Vulkan cumprod supports Float32 output only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan cumprod supports 1d to 4d tensors");

  int64_t axis = dim < 0 ? dim + self.dim() : dim;
  TP_CHECK(axis >= 0 && axis < self.dim(), "Vulkan cumprod: dim out of range");
  api::Context* const context = api::context();
  api::vTensor v_input = convert(self);
  api::vTensor v_output{context, v_input.sizes(), DType::Float32};

  const struct CumprodBlock final {
    ivec4 in_sizes;
    int axis;
    int c_depth;
    int fill;
  } block{
      make_whcn_ivec4(v_input.sizes()),
      static_cast<int>(self.dim() - 1 - axis),
      c_depth_of(v_input.sizes()),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  uvec3 global = v_output.extents();
  if (block.axis == 0) global[0u] = 1u;
  else if (block.axis == 1) global[1u] = 1u;
  else if (block.axis == 2) global[2u] = get_dim<Dim4D::Batch>(v_input);
  else global[2u] = static_cast<uint32_t>(block.c_depth);
  context->submit_compute_job(
      VK_KERNEL(cumprod),
      pipeline_barrier,
      global,
      adaptive_work_group_size(global),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

// Every listed dimension is reduced; an empty list reduces them all.
Tensor logsumexp_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan logsumexp supports Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan logsumexp supports 1d to 4d tensors");
  std::vector<int64_t> axes;
  for (int64_t d : dim) {
    const int64_t axis = d < 0 ? d + self.dim() : d;
    TP_CHECK(axis >= 0 && axis < self.dim(), "Vulkan logsumexp: dim out of range");
    TP_CHECK(std::find(axes.begin(), axes.end(), axis) == axes.end(),
             "logsumexp(): dim ", axis, " appears multiple times in the list of dims");
    axes.push_back(axis);
  }
  if (axes.empty()) {
    for (int64_t i = 0; i < self.dim(); ++i) axes.push_back(i);
  }

  const Tensor max_keep = extremum_impl(self, axes, true, ExtremumKind::kMax);
  const Tensor shifted = self - max_keep;
  const Tensor summed = shifted.exp().sum(axes, keepdim);
  Tensor result = summed.log();
  if (keepdim) {
    return result + max_keep;
  }
  return result + max_keep.reshape(result.shape());
}

Tensor isnan_kernel(const Tensor& self) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan isnan supports Float32 tensors only");
  TP_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan isnan supports 1d to 4d tensors");

  api::Context* const context = api::context();
  api::vTensor v_input = convert(self);
  api::vTensor v_output{context, v_input.sizes(), DType::Bool};
  const struct IsnanBlock final {
    ivec4 extents;
    int fill0;
    int fill1;
    int fill2;
  } block{
      ivec4(
          static_cast<int32_t>(v_output.extents()[0u]),
          static_cast<int32_t>(v_output.extents()[1u]),
          static_cast<int32_t>(v_output.extents()[2u]),
          0),
      0,
      0,
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL(isnan),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

//
// p-norms: abs then pow(1/p) fold around the whole-tensor sum, matching
// the norm formula the dim-listed CPU entry evaluates.  Infinite p needs
// a max reduction the backend does not carry, so it is rejected.
//
Tensor norm_impl(const Tensor& self, double p) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan norm supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4, "Vulkan norm: 1d to 4d only");
  TP_CHECK(!std::isinf(p), "Vulkan norm: infinite p is not supported");

  Tensor reduced;
  if (p == 2.0) {
    Tensor sq = self * self;
    reduced = sum_kernel(sq, DType::Undefined);
  } else if (p == 1.0) {
    reduced = sum_kernel(self.abs(), DType::Undefined);
  } else {
    Tensor powered = self.abs().pow(Scalar(p));
    reduced = sum_kernel(powered, DType::Undefined);
    return reduced.pow(Scalar(1.0 / p));
  }
  return p == 1.0 ? reduced : reduced.sqrt();
}

Tensor norm_kernel(const Tensor& self, double p) {
  return norm_impl(self, p);
}

Tensor norm_dim_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    double p,
    bool keepdim) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan norm supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4, "Vulkan norm: 1d to 4d only");
  TP_CHECK(!std::isinf(p), "Vulkan norm: infinite p is not supported");
  TP_CHECK(!dims.empty(), "Vulkan norm requires at least one dim");

  Tensor reduced;
  if (p == 2.0) {
    Tensor sq = self * self;
    reduced = reduce_impl(sq, dims, keepdim, false, false, 0);
  } else if (p == 1.0) {
    reduced = reduce_impl(self.abs(), dims, keepdim, false, false, 0);
  } else {
    Tensor powered = self.abs().pow(Scalar(p));
    reduced = reduce_impl(powered, dims, keepdim, false, false, 0);
    return reduced.pow(Scalar(1.0 / p));
  }
  return p == 1.0 ? reduced : reduced.sqrt();
}

namespace {

//
// Broadcast elementwise helpers used by the variance composition.  The
// patterns mirror the binary op shader conventions: out = a - b with b
// broadcast, and out = a * a elementwise.
//
Tensor broadcast_sub(const Tensor& a, const Tensor& b) {
  // Delegate to the binary op path: it infers the broadcasted output shape
  // and maps both operands through the texel access utilities, which is
  // required for a keepdim mean (e.g. a [2,1,1,1] tensor) to be expanded
  // to the full reduced span before the elementwise subtraction.
  return sub_kernel(a, b, Scalar(1.0));
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

//
// Whole-tensor variance: the same two-pass composition the dim-listed
// entry uses (keepdim mean, broadcast subtract, square, sum), with the
// reduced span covering every axis.
//
// Buffer-stored inputs use a one-pass composition instead: linear binary
// ops reject broadcasts, so the keepdim mean cannot be expanded.  The
// algebraically equivalent (E[x^2] - E[x]^2) form only needs same-shape
// reductions and scalar arithmetic.
Tensor var_buffer_one_pass(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    int64_t correction) {
  int64_t count = 1;
  for (const int64_t axis : dims) {
    count *= self.size(axis);
  }
  Tensor sum = reduce_impl(self, dims, /*keepdim=*/true, false, false, 0);
  // self.mul(self) routes through the buffer-capable square kernel; the
  // local image-only square helper would read no buffer payload.
  Tensor sum_sq =
      reduce_impl(self.mul(self), dims, /*keepdim=*/true, false, false, 0);
  const double denom =
      static_cast<double>(std::max<int64_t>(count - correction, 1));
  Tensor centered =
      sum_sq.sub(sum.mul(sum).mul(Scalar(1.0 / count)));
  Tensor result = centered.mul(Scalar(1.0 / denom));
  if (keepdim) {
    return result;
  }
  std::vector<int64_t> final_sizes;
  final_sizes.reserve(static_cast<size_t>(self.dim()));
  for (int64_t i = 0; i < self.dim(); ++i) {
    const bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (!reduced || result.size(i) != 1) {
      final_sizes.push_back(result.size(i));
    }
  }
  if (final_sizes.empty()) {
    return repack_with_sizes(result, final_sizes);
  }
  return repack_with_sizes(result, std::move(final_sizes));
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan var supports Float32 tensors only");
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4, "Vulkan var: 1d to 4d only");
  TP_CHECK(self.numel() > 0, "Vulkan var does not support empty tensors");

  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(self.dim()));
  for (int64_t d = 0; d < self.dim(); ++d) {
    dims.push_back(d);
  }

  api::vTensor v_self = convert(self);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    return var_buffer_one_pass(self, dims, /*keepdim=*/false, correction);
  }

  const int64_t count = self.numel();
  Tensor mean = reduce_impl(self, dims, /*keepdim=*/true, true, false, 0);
  Tensor centered_sq = square(broadcast_sub(self, mean));
  Tensor summed = reduce_impl(centered_sq, dims, false, false, false, 0);
  const double denom =
      static_cast<double>(std::max<int64_t>(count - correction, 1));
  return summed.mul(Scalar(1.0 / denom));
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
  Tensor v = var_kernel(self, correction);
  return v.sqrt();
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

  api::vTensor v_self = convert(self);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    return var_buffer_one_pass(self, dims, keepdim, correction);
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
  m.impl("sum", &tensorplay::vulkan::ops::sum_kernel);
  m.impl("sum.dim_IntList", &tensorplay::vulkan::ops::sum_dim_kernel);
  m.impl("mean", &tensorplay::vulkan::ops::mean_kernel);
  m.impl("mean.dim", &tensorplay::vulkan::ops::mean_dim_kernel);
  m.impl("max", &tensorplay::vulkan::ops::max_kernel);
  m.impl("max.dim", &tensorplay::vulkan::ops::max_dim_kernel);
  m.impl("min", &tensorplay::vulkan::ops::min_kernel);
  m.impl("min.dim", &tensorplay::vulkan::ops::min_dim_kernel);
  m.impl("amax", &tensorplay::vulkan::ops::amax_kernel);
  m.impl("amin", &tensorplay::vulkan::ops::amin_kernel);
  m.impl("prod", &tensorplay::vulkan::ops::prod_kernel);
  m.impl("prod.dim_IntList", &tensorplay::vulkan::ops::prod_dim_kernel);
  m.impl("all", &tensorplay::vulkan::ops::all_kernel);
  m.impl("all.dim", &tensorplay::vulkan::ops::all_dim_kernel);
  m.impl("any", &tensorplay::vulkan::ops::any_kernel);
  m.impl("any.dim", &tensorplay::vulkan::ops::any_dim_kernel);
  m.impl("argmax", &tensorplay::vulkan::ops::argmax_kernel);
  m.impl("argmin", &tensorplay::vulkan::ops::argmin_kernel);
  m.impl("count_nonzero", &tensorplay::vulkan::ops::count_nonzero_kernel);
  m.impl(
      "count_nonzero.dim_IntList",
      &tensorplay::vulkan::ops::count_nonzero_kernel);
  m.impl("cumprod", &tensorplay::vulkan::ops::cumprod_kernel);
  m.impl("logsumexp", &tensorplay::vulkan::ops::logsumexp_kernel);
  m.impl("isnan", &tensorplay::vulkan::ops::isnan_kernel);
  m.impl("var.dim", &tensorplay::vulkan::ops::var_dim_kernel);
  m.impl("std.dim", &tensorplay::vulkan::ops::std_dim_kernel);
  m.impl("var", &tensorplay::vulkan::ops::var_kernel);
  m.impl("std", &tensorplay::vulkan::ops::std_kernel);
  m.impl("norm", &tensorplay::vulkan::ops::norm_kernel);
  m.impl("norm.dim", &tensorplay::vulkan::ops::norm_dim_kernel);
}

#endif /* USE_VULKAN */
