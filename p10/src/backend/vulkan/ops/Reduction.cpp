#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "SizesAndStrides.h"
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

// Defined in View.cpp; materializes a strided view into a dense payload,
// reading a linear payload through the gather's buffer variant.
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset);

namespace {

// Materializes a tensor's shape into a plain sizes vector.  The shape
// accessor returns a value object, so the copy must be taken once instead
// of reaching through repeated temporaries.
std::vector<int64_t> shape_vector(const Tensor& t) {
  const Size shape = t.shape();
  return std::vector<int64_t>(shape.begin(), shape.end());
}

//
// A reduced payload inherits linear storage from its source, but storage is
// otherwise a function of the shape alone: consumers pick their route by
// looking at the operand, and a texture-only kernel handed a 32-element
// buffer payload has no linear variant to run.  A result that still fits an
// image therefore repacks into one on the way out.  The gather covers the
// 4-byte element types; every other type stays where the kernel wrote it.
//
Tensor release_reduced_output(
    api::Context* const context,
    api::vTensor& v_reduced) {
  Tensor flat = convert(v_reduced);
  const std::vector<int64_t> sizes(
      v_reduced.sizes().begin(), v_reduced.sizes().end());
  if (!image_storage_supported(context, sizes) ||
      (v_reduced.dtype() != DType::Float32 &&
       v_reduced.dtype() != DType::Int32)) {
    return flat;
  }
  return materialize_view(flat, sizes,
                          SizesAndStrides::compute_contiguous_strides(sizes), 0);
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
  const bool is_half = input.dtype() == DType::Float16;
  const std::string shader =
      std::string(var_mode ? "var_axis_buffer" : "sum_axis_buffer") +
      (is_half ? "_half" : "");
  // A half payload stores two reduced lanes per word, so the row grid becomes a
  // word grid and every invocation owns the word it writes.
  const int64_t grid_rows = is_half ? (rows + 1) / 2 : rows;
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
      VK_KERNEL_FROM_STR(shader.c_str()),
      pipeline_barrier,
      {static_cast<uint32_t>(grid_rows), 1u, 1u},
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
  // through a linear row-walking variant, which writes linear storage; a
  // result whose own shape fits an image repacks into one on the way out.
  // A half payload accumulates in single precision and packs two reduced
  // lanes per output word.
  if (v_input.storage_type() == api::StorageType::BUFFER) {
    TP_CHECK(
        input.dtype() == DType::Float32 || input.dtype() == DType::Float16,
        "Vulkan reductions over linear storage support Float32 and Float16 "
        "tensors only");
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
      return release_reduced_output(context, v_output);
    }

    reduce_axis_buffer_pass(
        context, v_output, v_input, axis_size, inner, rows, correction, scale,
        var_mode);
    return release_reduced_output(context, v_output);
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

// One pass over a linear payload: `rows` sort rows of `tile` elements each,
// where consecutive elements of a row stride by `inner`.  Extrema and the
// wide totals share the walk and so the metadata block.
void linear_reduce_pass(
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
  // A half payload stores two reduced lanes per word, so the row grid becomes a
  // word grid and every invocation owns the word it writes.
  const int64_t grid_rows = output.dtype() == DType::Float16
      ? (rows + 1) / 2
      : rows;
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader),
      pipeline_barrier,
      {static_cast<uint32_t>(grid_rows), 1u, 1u},
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
  // texture requirement below.  The half builds pack two winners per word.
  // A counted payload folds in the width it names: the signed word for a
  // narrow count, a pair of words for a wide one.
  const bool is_half = input.dtype() == DType::Float16;
  const bool is_wide = input.dtype() == DType::Int64;
  const bool integral =
      kind != ExtremumKind::kProd &&
      (input.dtype() == DType::Int32 || is_wide);
  const bool greater = kind == ExtremumKind::kMax;
  const char* buffer_shader = nullptr;
  if (integral) {
    buffer_shader = greater
        ? (is_wide ? "reduce_max_buffer_i64" : "reduce_max_buffer_i32")
        : (is_wide ? "reduce_min_buffer_i64" : "reduce_min_buffer_i32");
  } else if (greater) {
    buffer_shader = is_half ? "reduce_max_buffer_half" : "reduce_max_buffer";
  } else if (kind == ExtremumKind::kMin) {
    buffer_shader = is_half ? "reduce_min_buffer_half" : "reduce_min_buffer";
  }
  if (v_input.storage_type() == api::StorageType::BUFFER &&
      buffer_shader != nullptr) {
    api::vTensor v_output{
        context,
        out_sizes,
        input.dtype(),
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
          input.dtype(),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
      linear_reduce_pass(
          context, v_tiles, v_input, kTileElements, 1, tiles, buffer_shader);
      linear_reduce_pass(
          context, v_output, v_tiles, tiles, 1, 1, buffer_shader);
      return release_reduced_output(context, v_output);
    }

    linear_reduce_pass(
        context, v_output, v_input, axis_size, inner, rows, buffer_shader);
    return release_reduced_output(context, v_output);
  }

  api::vTensor v_output{context, out_sizes, input.dtype()};
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
  // A counted payload answers with the signed-word builds; a measurement keeps
  // the dtype-selected texture family.  A wide count has no texel format to
  // name, so it never reaches this walk.
  const api::ShaderInfo& texture_kernel =
      input.dtype() == DType::Int32
      ? VK_KERNEL_FROM_STR(greater ? "reduce_max_i32" : "reduce_min_i32")
      : kernel_for(extremum_shader_name(kind), input.dtype());
  context->submit_compute_job(
      texture_kernel, pipeline_barrier,
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
  // An extremum folds in the width the payload names, so a counted payload has
  // its own walk; a product is a measurement and keeps the floating family.
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16 ||
          (kind != ExtremumKind::kProd &&
           (self.dtype() == DType::Int32 || self.dtype() == DType::Int64)),
      "Vulkan extrema support Float32, Float16, Int32 and Int64 tensors; "
      "products fold a measured payload");
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

enum class TotalKind { kSum, kProd };

//
// A payload whose elements are wider than a word has no texel to read and no
// single-precision register to fold into, so its total walks the pairs of words
// it is carried in.  One row per invocation, the same stride along the axis and
// the same tile split an extremum of such a payload uses; the fold starts at the
// identity it names, so a short tile contributes fewer terms and an empty fold
// answers with that identity.  Wrapping happens at the width of the pair, which
// is where a count wrapped at its own width lands.
//
Tensor wide_total_one_axis(
    const Tensor& input,
    int64_t axis_from_left,
    TotalKind kind) {
  api::Context* const context = api::context();
  api::vTensor v_input = convert(input);

  std::vector<int64_t> out_sizes = shape_vector(input);
  out_sizes[static_cast<size_t>(axis_from_left)] = 1;
  api::vTensor v_output{
      context,
      out_sizes,
      input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const int64_t axis_size = input.size(axis_from_left);
  int64_t inner = 1;
  for (int64_t d = axis_from_left + 1; d < input.dim(); ++d) {
    inner *= input.size(d);
  }
  const int64_t rows = input.numel() / axis_size;
  const char* shader =
      kind == TotalKind::kSum ? "reduce_sum_buffer_i64"
                              : "reduce_prod_buffer_i64";

  // A row per invocation is already parallel while many rows are left behind.
  // One that folds the whole payload into a single row would hand every
  // element to one thread, so split the axis into fixed tiles, fold each tile,
  // and fold the tile results afterwards.
  constexpr int64_t kTileElements = 256;
  if (rows == 1 && axis_size > kTileElements) {
    const int64_t tiles = (axis_size + kTileElements - 1) / kTileElements;
    api::vTensor v_tiles{
        context,
        std::vector<int64_t>{tiles},
        input.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
    linear_reduce_pass(
        context, v_tiles, v_input, kTileElements, 1, tiles, shader);
    linear_reduce_pass(context, v_output, v_tiles, tiles, 1, 1, shader);
    return release_reduced_output(context, v_output);
  }

  linear_reduce_pass(
      context, v_output, v_input, axis_size, inner, rows, shader);
  return release_reduced_output(context, v_output);
}

Tensor wide_total_impl(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    TotalKind kind) {
  TP_CHECK(
      self.dtype() == DType::Int64,
      "Vulkan wide totals fold a payload of counts only");
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
    current = wide_total_one_axis(current, axis, kind);
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
  // through a linear variant, which writes linear storage; a result whose
  // own shape fits an image repacks into one on the way out.
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
    // The winning position is an integer, so the output keeps a 32-bit element
    // per row and only the payload is read as half lanes.
    const bool is_half = input.dtype() == DType::Float16;
    const std::string buffer_shader =
        std::string(greater ? "reduce_argmax_buffer" : "reduce_argmin_buffer") +
        (is_half ? "_half" : "");
    // A half payload keeps one invocation per row along the x axis; the float
    // builds are workgroup-cooperative, so the row rides on the y axis and x
    // carries the 256 lanes that stride along it.
    api::utils::uvec3 dispatch_grid{static_cast<uint32_t>(rows), 1u, 1u};
    if (!is_half) {
      dispatch_grid = api::utils::uvec3{1u, static_cast<uint32_t>(rows), 1u};
    }
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_shader.c_str()),
        pipeline_barrier,
        dispatch_grid,
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::READ),
        params.buffer());
    return release_reduced_output(context, v_output);
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
      kernel_for(greater ? "reduce_argmax" : "reduce_argmin", input.dtype()),
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
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan index reductions support Float32 and Float16 tensors only");
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
  // One workgroup per answer, and a group is exactly as wide as the tree that
  // folds the sixty-four strides of the walk back into one number.  The launch
  // extent counts invocations, so an answer costs a whole group of sixty-four
  // of them.
  const uint32_t groups = std::min<uint32_t>(output.numel(), 65535u);
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader), barrier,
      {groups * 64u, api::utils::div_up(static_cast<uint32_t>(output.numel()), groups), 1u},
      {64u, 1u, 1u}, VK_NULL_HANDLE,
      output.buffer(barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
      input.image(barrier, api::PipelineStage::COMPUTE), params.buffer());
  return convert(output);
}

//
// A payload that counts rather than measures has no reduced type to carry its
// own total: the answer to a sum of truth lanes, or of lanes holding whole
// numbers, is named by the caller's request or defaults to the wide word.
//
// Only a lane that fits single precision exactly can be added up here: a truth
// lane, a byte, or a signed word is, while a lane already wide enough to hold a
// number past that reach has no walk that carries it without rounding.
//
bool totals_in_singles(DType dtype) {
  return dtype == DType::Bool || dtype == DType::Int8 ||
      dtype == DType::UInt8 || dtype == DType::Int32;
}

// The codes a total folds in its own width: the narrow ones reach that width
// through single precision or a signed word, and a count wider than a word
// folds in the pairs it is carried in.
bool totals_as_a_count(DType dtype) {
  return totals_in_singles(dtype) || dtype == DType::Int64;
}

//
// The walk that adds lanes into a wide word reads a texture, so a plane whose
// shape has one is reduced in a single pass.  A signed word is read as itself,
// since a lane past the exact reach of single precision must not be rounded on
// the way in; every narrower code (a truth lane, a byte) is exact in single
// precision, so it steps through that width to reach the walk.  A payload with
// no texture form has no such walk: it sums through the single-precision row
// walk, exact while the running total stays inside the reach of a float, and
// the total then crosses to the wide word by way of the signed one.
//
Tensor integral_total(
    const Tensor& self,
    const std::vector<int64_t>& dims_in,
    bool keepdim,
    DType wanted) {
  std::vector<int64_t> dims = dims_in;
  if (dims.empty()) {
    dims.reserve(static_cast<size_t>(self.dim()));
    for (int64_t d = 0; d < self.dim(); ++d) {
      dims.push_back(d);
    }
  }
  dims = normalized_dims(self, dims);
  TP_CHECK(!dims.empty(), "Vulkan reductions require at least one dim");
  TP_CHECK(
      self.numel() > 0, "Vulkan reductions do not support empty tensors");

  const bool wide = wanted == DType::Undefined || wanted == DType::Int64;
  TP_CHECK(
      wide || wanted == DType::Float32 || wanted == DType::Float16 ||
          wanted == DType::Int32,
      "Vulkan sums of a counted payload answer in Int64, Float32, Float16 "
      "or Int32");

  // A payload with no texel to read and no single-precision reach folds in the
  // pairs it is carried in, and the answer then moves to the width asked for.
  if (self.dtype() == DType::Int64) {
    Tensor total = wide_total_impl(self, dims, keepdim, TotalKind::kSum);
    return wanted == DType::Undefined || wanted == DType::Int64
        ? total
        : total.to(wanted);
  }

  const Tensor singles = self.to(DType::Float32);

  if (!wide) {
    Tensor total = reduce_impl(singles, dims, keepdim, false, false, 0);
    return wanted == DType::Float32 ? total : total.to(wanted);
  }

  if (convert(self).storage_type() == api::StorageType::TEXTURE_3D) {
    return self.dtype() == DType::Int32
        ? reduce_int64(self, dims, keepdim, "sum_int64")
        : reduce_int64(singles, dims, keepdim, "sum_int64_float");
  }
  return reduce_impl(singles, dims, keepdim, false, false, 0)
      .to(DType::Int32)
      .to(DType::Int64);
}

Tensor count_impl(const Tensor& self, const std::vector<int64_t>& dims) {
  if (self.dtype() == DType::Bool) {
    // A truth lane holds 0 or 1, so the lanes a count asks for are exactly the
    // lanes a total adds up.
    return integral_total(self, dims, false, DType::Int64);
  }
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
  if (totals_as_a_count(self.dtype())) {
    return integral_total(self, dims, keepdim, dtype);
  }
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
  if (totals_as_a_count(self.dtype())) {
    if (self.numel() == 0) {
      // Nothing adds to nothing: the width the total was asked to carry still
      // names the answer.
      return full_kernel(
          {}, Scalar(0),
          dtype == DType::Undefined ? DType::Int64 : dtype,
          self.device(), false);
    }
    return integral_total(self, {}, false, dtype);
  }
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan reductions support Float32 and Float16 tensors only");
  if (self.numel() == 0) {
    return full_kernel(
        {}, Scalar(0.0), self.dtype(), Device(DeviceType::Vulkan), false);
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
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan reductions support Float32 and Float16 tensors only");
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

//
// The range a payload covers, which is the first thing a quantization observer
// asks of a weight plane before it names a scale.  Both ends answer in the width
// the payload is carried in, so the two folds take the walks a single extremum
// already uses and cover every element type on both storages.  A payload with no
// lanes has no identity to answer a range with, and the folds say so.
//
std::tuple<Tensor, Tensor> aminmax_kernel(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim) {
  return {amin_kernel(self, dims, keepdim), amax_kernel(self, dims, keepdim)};
}

Tensor prod_kernel(const Tensor& self, DType dtype) {
  if (self.dtype() == DType::Int64) {
    // A count wider than a word multiplies in the pairs it is carried in; the
    // answer then moves to the width the caller asked the product to carry.
    if (dtype == DType::Undefined || dtype == DType::Int64) {
      return wide_total_impl(self, {}, false, TotalKind::kProd);
    }
    if (dtype == DType::Int32) {
      return wide_total_impl(self, {}, false, TotalKind::kProd).to(
          DType::Int32);
    }
    TP_CHECK(dtype == DType::Float32, "Unsupported Vulkan product dtype");
    return extremum_impl(
        self.to(DType::Float32), {}, false, ExtremumKind::kProd);
  }
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
  if (self.dtype() == DType::Int64) {
    if (dtype == DType::Undefined || dtype == DType::Int64) {
      return wide_total_impl(self, dims, keepdim, TotalKind::kProd);
    }
    if (dtype == DType::Int32) {
      return wide_total_impl(self, dims, keepdim, TotalKind::kProd).to(
          DType::Int32);
    }
    TP_CHECK(dtype == DType::Float32, "Unsupported Vulkan product dtype");
    return extremum_impl(
        self.to(DType::Float32), dims, keepdim, ExtremumKind::kProd);
  }
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
// A half variance over one axis has a walk that keeps both raw moments in
// single precision, so only the variance itself narrows.  The two-pass
// composition would round every squared deviation to the payload format, and
// the sum of those squares leaves the finite range of the format on a long
// axis.  Both storages have that walk.
std::optional<Tensor> var_half_single_axis(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    int64_t correction) {
  if (self.dtype() != DType::Float16 || dims.size() != 1) {
    return std::nullopt;
  }
  return reduce_impl(self, dims, keepdim, /*mean_mode=*/false,
                     /*var_mode=*/true, correction);
}

Tensor var_buffer_one_pass(
    const Tensor& self,
    const std::vector<int64_t>& dims,
    bool keepdim,
    int64_t correction) {
  int64_t count = 1;
  for (const int64_t axis : dims) {
    count *= self.size(axis);
  }
  const double denom =
      static_cast<double>(std::max<int64_t>(count - correction, 1));

  // A half payload cannot carry the raw sums: the square sum of a long axis
  // leaves the finite range of the format well before the axis ends.  Several
  // axes have no fused walk, so the moments are taken as means instead, each
  // bounded by the largest squared element rather than by the span length.
  const bool is_half = self.dtype() == DType::Float16;

  Tensor result = [&]() {
    if (is_half) {
      const Tensor mean =
          reduce_impl(self, dims, /*keepdim=*/true, true, false, 0);
      const Tensor mean_sq =
          reduce_impl(self.mul(self), dims, /*keepdim=*/true, true, false, 0);
      return mean_sq.sub(mean.mul(mean))
          .mul(Scalar(static_cast<double>(count) / denom));
    }
    const Tensor sum =
        reduce_impl(self, dims, /*keepdim=*/true, false, false, 0);
    // self.mul(self) routes through the buffer-capable square kernel; the
    // local image-only square helper would read no buffer payload.
    const Tensor sum_sq =
        reduce_impl(self.mul(self), dims, /*keepdim=*/true, false, false, 0);
    return sum_sq.sub(sum.mul(sum).mul(Scalar(1.0 / count)))
        .mul(Scalar(1.0 / denom));
  }();

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
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan var supports Float32 and Float16 tensors only");
  TP_CHECK(self.dim() >= 1 && self.dim() <= 4, "Vulkan var: 1d to 4d only");
  TP_CHECK(self.numel() > 0, "Vulkan var does not support empty tensors");

  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(self.dim()));
  for (int64_t d = 0; d < self.dim(); ++d) {
    dims.push_back(d);
  }

  if (auto fused = var_half_single_axis(self, dims, false, correction)) {
    return *fused;
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
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan var supports Float32 and Float16 tensors only");
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

  if (auto fused = var_half_single_axis(self, dims, keepdim, correction)) {
    return *fused;
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
  m.impl("aminmax", &tensorplay::vulkan::ops::aminmax_kernel);
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
  m.impl("var.dim", &tensorplay::vulkan::ops::var_dim_kernel);
  m.impl("std.dim", &tensorplay::vulkan::ops::std_dim_kernel);
  m.impl("var", &tensorplay::vulkan::ops::var_kernel);
  m.impl("std", &tensorplay::vulkan::ops::std_kernel);
  m.impl("norm", &tensorplay::vulkan::ops::norm_kernel);
  m.impl("norm.dim", &tensorplay::vulkan::ops::norm_dim_kernel);
}

#endif /* USE_VULKAN */
