#ifdef USE_VULKAN

#include "Blocks.h"
#include "Bucketization.h"
#include "Common.h"
#include "Convert.h"
#include "DType.h"
#include "Factory.h"
#include "Tensor.h"
#include "Utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in Copy.cpp; the element type conversion entry point.
Tensor to_dtype_kernel(
    const Tensor& self,
    DType dtype,
    bool non_blocking,
    bool copy,
    std::optional<int64_t> memory_format);

// Defined in BinaryOp.cpp; copies a texture payload into a dense buffer.
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src);

namespace {

// One query per invocation, and a walk that reads a table entry per step, so a
// launch this wide already keeps a mobile GPU searching.  A payload with more
// queries than the grid holds steps through the rest.
constexpr uint32_t kSearchWorkgroup = 256;
constexpr int32_t kMaxSearchWorkgroups = 2048;

struct SearchGrid final {
  api::utils::uvec3 extent;
  int32_t stride;
};

SearchGrid search_grid(int64_t queries) {
  const int64_t needed = (queries + kSearchWorkgroup - 1) / kSearchWorkgroup;
  const int64_t workgroups = std::min<int64_t>(needed, kMaxSearchWorkgroups);
  return SearchGrid{
      {static_cast<uint32_t>(workgroups * kSearchWorkgroup), 1u, 1u},
      static_cast<int32_t>(workgroups * kSearchWorkgroup)};
}

/*
 * The width a search compares in.
 *
 * A half payload reaches single precision without loss and a byte-packed one
 * reaches the signed word the same way, so both operands widen into whichever
 * of the two their common type asks for.
 */
DType search_dtype(const DType common) {
  switch (common) {
    case DType::Float32:
    case DType::Float16:
      return DType::Float32;
    case DType::Int32:
    case DType::Int8:
    case DType::UInt8:
    case DType::Bool:
      return DType::Int32;
    default:
      // A wide code or a double would have to be compared as one value spanning
      // two words, and no lane of this walk reads a pair.  Rounding either into
      // single precision would answer for a different number, so the pair is
      // refused rather than quietly narrowed.
      TP_THROW(
          TypeError,
          "Vulkan searchsorted compares in single precision or the signed "
          "word, not ",
          toString(common));
  }
}

// The walk reads flat words in the dense order, so a payload that arrives as an
// image, or as a view of one, is laid out first.  The tensor is what keeps the
// storage alive, so it travels with the view of it.
struct Dense final {
  Tensor payload;
  api::vTensor view;
};

Dense lay_out(api::Context* const context, const Tensor& t) {
  Tensor local = (t.is_contiguous() &&
                  t.unsafeGetTensorImpl()->storage_offset() == 0)
      ? t
      : t.contiguous();
  api::vTensor v_local = convert(local);
  if (v_local.storage_type() != api::StorageType::BUFFER) {
    api::vTensor v_buffer = to_buffer_storage_dispatch(context, v_local);
    local = convert(v_buffer);
    v_local = convert(local);
  }
  return Dense{std::move(local), std::move(v_local)};
}

// A scalar query takes the width the sequence carries.  The walk compares two
// operands of one width, and a host number arrives in a width this backend has
// no lane for (a float literal is double, an integral one the wide code), so
// the sequence, which is already resident, names the width instead.
Tensor query_from_scalar(const Scalar& scalar, const Tensor& sequence) {
  return Tensor::full({}, scalar, sequence.dtype(), sequence.device());
}

/*
 * The insertion position of every query: for each one the walk halves the table
 * that answers it and counts the entries that come before the position the
 * query takes.  Either one sequence answers every query, or every row of the
 * input has a table of its own, and the two differ only in where a table
 * starts.
 *
 * Positions are one word wide, which is what a table of entries can be counted
 * in; the caller widens them when the answer is asked for in the wide code.
 */
Tensor searchsorted_codes(
    const Tensor& sorted_sequence,
    const Tensor& self,
    const bool is_right) {
  const int64_t queries = self.numel();
  const int64_t span = sorted_sequence.size(-1);
  TP_CHECK(
      span <= std::numeric_limits<int32_t>::max(),
      "Vulkan searchsorted counts a table in the signed word, got ",
      span, " entries");

  const DType common = promoteTypes(sorted_sequence.dtype(), self.dtype());
  TP_CHECK(
      common != DType::Undefined,
      "Vulkan searchsorted cannot compare a sequence of ",
      toString(sorted_sequence.dtype()),
      " against queries of ",
      toString(self.dtype()));
  const DType width = search_dtype(common);

  const auto widen = [&](const Tensor& t) {
    return t.dtype() == width
        ? t
        : to_dtype_kernel(
              t, width, /*non_blocking=*/false, /*copy=*/true, std::nullopt);
  };

  api::Context* const context = api::context();
  Dense seq = lay_out(context, widen(sorted_sequence));
  Dense vals = lay_out(context, widen(self));

  api::vTensor v_output{
      context,
      std::vector<int64_t>(self.sizes().begin(), self.sizes().end()),
      DType::Int32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const bool one_table = sorted_sequence.dim() == 1;
  const int64_t row = self.dim() == 0 ? 1 : self.size(-1);
  const SearchGrid grid = search_grid(queries);
  const struct SearchBlock final {
    int32_t queries;
    int32_t row;
    int32_t span;
    int32_t upper;
    int32_t one_d;
    int32_t stride;
    int32_t fill0;
    int32_t fill1;
  } block{
      static_cast<int32_t>(queries),
      static_cast<int32_t>(row),
      static_cast<int32_t>(span),
      is_right ? 1 : 0,
      one_table ? 1 : 0,
      grid.stride,
      0,
      0};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(width == DType::Float32 ? "searchsorted_float"
                                                : "searchsorted_int"),
      pipeline_barrier,
      grid.extent,
      {kSearchWorkgroup, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      seq.view.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      vals.view.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      params.buffer());

  return convert(v_output);
}

Tensor searchsorted_answer(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<std::string>& side_opt,
    const Tensor& sorter) {
  TP_CHECK(
      !sorter.defined(),
      "Vulkan searchsorted does not read a sorter; order the sequence first");
  const DType code_dtype = out_int32 ? DType::Int32 : DType::Int64;
  // The width the answer is asked for is settled before the walk runs, and the
  // shared check reads it from the tensor handed to it.
  Tensor answer =
      empty_kernel({0}, code_dtype, self.device(), /*pin_memory=*/false);
  bucketization::pre_check(
      sorted_sequence, self, answer, out_int32, right, side_opt, sorter);

  const std::vector<int64_t> answer_shape(
      self.sizes().begin(), self.sizes().end());
  if (self.numel() == 0) {
    // No query has a position to answer.
    return empty_kernel(
        answer_shape,
        code_dtype,
        self.device(),
        /*pin_memory=*/false);
  }
  if (sorted_sequence.numel() == 0) {
    // Every query takes the position at the front of a table with no entries,
    // which is zero; the walk is not launched for it.
    return zeros_kernel(
        answer_shape,
        code_dtype,
        self.device(),
        /*pin_memory=*/false);
  }
  TP_CHECK(
      self.numel() <= std::numeric_limits<int32_t>::max(),
      "Vulkan searchsorted answers at most a word of queries, got ",
      self.numel());

  const bool is_right = side_opt.has_value() ? *side_opt == "right" : right;
  Tensor codes = searchsorted_codes(sorted_sequence, self, is_right);
  if (out_int32) {
    return codes;
  }
  return to_dtype_kernel(
      codes, DType::Int64, /*non_blocking=*/false, /*copy=*/true, std::nullopt);
}

} // namespace

/*
 * The position each query takes in a sorted sequence, and the bucket a value
 * falls into once the sequence is read as a set of edges.
 *
 * Both operands reach the walk as dense linear payloads of one width, so a
 * half, a mask or a byte code is widened first - which is where a sequence of
 * edges kept in a model's own precision becomes searchable - and the answer is
 * one position per query.  The sequence is not checked for order: a caller
 * that hands an unordered one gets the position the halving walk stops at.
 */

Tensor searchsorted_tensor_kernel(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<std::string>& side_opt,
    const std::optional<Tensor>& sorter_opt) {
  return searchsorted_answer(
      sorted_sequence,
      self,
      out_int32,
      right,
      side_opt,
      sorter_opt.value_or(Tensor()));
}

Tensor searchsorted_scalar_kernel(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<std::string>& side_opt,
    const std::optional<Tensor>& sorter_opt) {
  const Tensor query = query_from_scalar(self, sorted_sequence);
  return searchsorted_answer(
      sorted_sequence,
      query,
      out_int32,
      right,
      side_opt,
      sorter_opt.value_or(Tensor()));
}

Tensor bucketize_tensor_kernel(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  TP_CHECK(
      boundaries.dim() == 1,
      "bucketize(): boundaries tensor must be 1 dimension, but got dim(",
      boundaries.dim(), ")");
  return searchsorted_answer(
      boundaries, self, out_int32, right, std::nullopt, Tensor());
}

Tensor bucketize_scalar_kernel(
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  const Tensor query = query_from_scalar(self, boundaries);
  return bucketize_tensor_kernel(query, boundaries, out_int32, right);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, BucketizationKernels) {
  m.impl(
      "searchsorted.Tensor",
      &tensorplay::vulkan::ops::searchsorted_tensor_kernel);
  m.impl(
      "searchsorted.Scalar",
      &tensorplay::vulkan::ops::searchsorted_scalar_kernel);
  m.impl(
      "bucketize.Tensor",
      &tensorplay::vulkan::ops::bucketize_tensor_kernel);
  m.impl(
      "bucketize.Scalar",
      &tensorplay::vulkan::ops::bucketize_scalar_kernel);
}

#endif /* USE_VULKAN */
