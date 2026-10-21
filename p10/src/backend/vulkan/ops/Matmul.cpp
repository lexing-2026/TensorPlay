#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "ParamCache.h"
#include "Tensor.h"
#include "../impl/Packing.h"

#include <SizesAndStrides.h>
#include <Utils.h>

#include <algorithm>
#include <functional>

namespace tensorplay {
namespace vulkan {
namespace ops {

// Defined in View.cpp; materializes a strided view into a dense payload,
// reading a linear payload through the gather's buffer variant.
Tensor materialize_view(
    const Tensor& self,
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& out_strides,
    int64_t storage_offset);

namespace {

// Cache tags distinguishing the packed forms of one operand tensor.  These
// extend the convolution tags in Conv.cpp; the numeric identity is local to
// this translation unit's cache entries.  Batched packings fold the batch
// count into the low bits so distinct plane counts cache separately.
constexpr uint32_t kTagMmWidthPacked = 10u;
constexpr uint32_t kTagMmHeightPacked = 11u;
constexpr uint32_t kTagMmWidthPackedTransposed = 12u;
constexpr uint32_t kTagMmHeightPackedTransposed = 13u;
constexpr uint32_t kTagMmWidthPackedBatched = 14u;
constexpr uint32_t kTagMmHeightPackedBatched = 15u;

constexpr uint32_t kBatchTagShift = 16u;

uint32_t batched_tag(uint32_t base, int64_t batches) {
  return base |
      (static_cast<uint32_t>(batches) << kBatchTagShift);
}

std::vector<int64_t> shape_of(const Tensor& t) {
  return static_cast<std::vector<int64_t>>(t.shape());
}

//
// Operand packing.  Every packer gathers directly from the operand's
// channel-packed texture into the layout a product kernel streams, so one
// dispatch replaces both a materialized relayout and any transpose copy.
// Results go through the persistent pack cache keyed on the source storage
// identity (pointer + version + tag); steady-state loops pay the relayout
// once per weight, and the cache evicts so fresh-tensor call patterns cannot
// grow it without bound.
//

api::vTensor packed_cached(
    const Tensor& src,
    const std::vector<int64_t>& cache_sizes,
    api::GPUMemoryLayout layout,
    uint32_t tag,
    const std::function<api::vTensor()>& build) {
  return ParamTextureCache::singleton().get_or_create(
      src, cache_sizes, layout, tag, build);
}

// Dense (M x K) -> width-packed: texel (k / 4, m) lanes carry M[m][4j..4j+3].
api::vTensor pack_lhs_width(const Tensor& src) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      src_sizes,
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  context->submit_compute_job(
      kernel_for("convert_channels_to_width_packed", src.dtype()),
      pack_barrier,
      v_dst.extents(),
      adaptive_work_group_size(v_dst.extents()),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

// Dense (K x N) -> height-packed: texel (n, k / 4) lanes carry
// M[4j..4j+3][n].
api::vTensor pack_rhs_height(const Tensor& src) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      src_sizes,
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  context->submit_compute_job(
      kernel_for("convert_channels_to_height_packed", src.dtype()),
      pack_barrier,
      v_dst.extents(),
      adaptive_work_group_size(v_dst.extents()),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

// Dense (K x M) holding A^T -> width-packed (M x K) planes for A: texel
// (j, m) lane l gathers A[m][4j + l] from the source column m, row 4j + l.
api::vTensor pack_lhs_width_transposed(const Tensor& src) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  // The packed operand carries the transposed logical shape.
  std::vector<int64_t> dst_sizes{src_sizes.back(), src_sizes[src_sizes.size() - 2]};
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      dst_sizes,
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  const api::utils::uvec3 global_size{
      api::utils::div_up(
          api::utils::safe_downcast_to_u32(dst_sizes[1]), 4u),
      api::utils::safe_downcast_to_u32(dst_sizes[0]),
      1u,
  };

  context->submit_compute_job(
      kernel_for("pack_lhs_transposed", src.dtype()),
      pack_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

// Dense (N x K) holding B^T -> height-packed (K x N) planes for B: texel
// (n, j) lane l gathers B[4j + l][n] from the source row n, column 4j + l.
api::vTensor pack_rhs_height_transposed(const Tensor& src) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  std::vector<int64_t> dst_sizes{src_sizes.back(), src_sizes[src_sizes.size() - 2]};
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      dst_sizes,
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  const api::utils::uvec3 global_size{
      api::utils::safe_downcast_to_u32(dst_sizes[1]),
      api::utils::div_up(
          api::utils::safe_downcast_to_u32(dst_sizes[0]), 4u),
      1u,
  };

  context->submit_compute_job(
      kernel_for("pack_rhs_transposed", src.dtype()),
      pack_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

// Batched variants: one width/height-packed plane per batch, texel z equal
// to the batch index.  A single-batch source broadcasts across all dst
// planes inside the gather shader, so no expand materialization is needed.
api::vTensor bmm_pack_lhs_width(const Tensor& src, int64_t batches) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  const int64_t M = src_sizes[src_sizes.size() - 2];
  const int64_t K = src_sizes.back();
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      {batches, M, K},
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  const api::utils::uvec3 global_size{
      api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u),
      api::utils::safe_downcast_to_u32(M),
      api::utils::safe_downcast_to_u32(batches),
  };

  context->submit_compute_job(
      kernel_for("bmm_pack_lhs", src.dtype()),
      pack_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

api::vTensor bmm_pack_rhs_height(const Tensor& src, int64_t batches) {
  api::Context* const context = api::context();

  const std::vector<int64_t> src_sizes = shape_of(src);
  const int64_t K = src_sizes[src_sizes.size() - 2];
  const int64_t N = src_sizes.back();
  api::vTensor v_src = convert(src);

  api::vTensor v_dst{
      context,
      {batches, K, N},
      src.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  const struct PackBlock final {
    ivec4 sizes;
  } block{make_ivec4_prepadded1(src_sizes)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pack_barrier{};

  const api::utils::uvec3 global_size{
      api::utils::safe_downcast_to_u32(N),
      api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u),
      api::utils::safe_downcast_to_u32(batches),
  };

  context->submit_compute_job(
      kernel_for("bmm_pack_rhs", src.dtype()),
      pack_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_dst.image(
          pack_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.image(pack_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return v_dst;
}

//
// A 2d operand that is a pure transpose view of its own storage is packed
// from the dense twin with a transposed gather instead of being copied into
// a contiguous tensor first.  Returns false when the operand is not such a
// view, in which case the caller materializes and packs normally.
//
bool transpose_view_of_dense(const Tensor& t, Tensor* dense_out) {
  if (t.dim() != 2 || t.is_contiguous() ||
      t.unsafeGetTensorImpl()->storage_offset() != 0) {
    return false;
  }
  const int64_t r = t.size(0);
  const int64_t c = t.size(1);
  if (t.stride(0) == 1 && t.stride(1) == r) {
    *dense_out = t.as_strided({c, r}, {r, 1});
    return true;
  }
  return false;
}

//
// Product dispatch over pre-packed operands.  Both packed textures use one
// texel z slot per (batch of four output lanes); the 2d kernels keep a
// single slot.
//

/*
 * Decode-shape products run as a two-pass split reduction instead of the
 * tiled product.  The tiled kernel keeps a full 32x32 block per workgroup:
 * with a few output rows most staging and most threads go unused while the
 * reduction loop still walks every K texel serially per thread, which caps
 * the memory stream far below device bandwidth.  Splitting K across
 * invocations restores occupancy for the latency-bound fetch chain, and the
 * fold pass replaces the epilogue.
 *
 * The split count bounds the per-invocation serial span and targets a fixed
 * thread budget across the grid, whichever needs more splits; the fold read
 * cost stays trivial since partials carry one float per texel lane.
 */
constexpr int64_t kGemvMaxRows = 4;
constexpr int64_t kGemvSpanTexels = 32;
constexpr int64_t kGemvSplitMax = 64;
constexpr int64_t kGemvMinReduction = 512;
constexpr int64_t kGemvThreadBudget = 8192;

Tensor gemv_decode(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t M,
    int64_t N,
    int64_t K,
    const std::optional<Tensor>& bias,
    Scalar beta,
    Scalar alpha,
    DType dtype) {
  api::Context* const context = api::context();

  const uint32_t k_texels =
      api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u);
  const uint32_t col_jobs = api::utils::div_up(
      api::utils::safe_downcast_to_u32(N), 4u);
  uint32_t splits = api::utils::div_up(k_texels, static_cast<uint32_t>(kGemvSpanTexels));
  const uint32_t col_row_jobs = col_jobs * api::utils::safe_downcast_to_u32(M);
  if (col_row_jobs > 0u) {
    splits = std::max(splits, api::utils::div_up(
        static_cast<uint32_t>(kGemvThreadBudget), col_row_jobs));
  }
  splits = std::max(splits, 1u);
  splits = std::min(splits, static_cast<uint32_t>(kGemvSplitMax));
  splits = std::min(splits, k_texels);
  const uint32_t chunk = api::utils::div_up(k_texels, splits);

  api::vTensor v_output{context, {M, N}, dtype};

  api::vTensor v_partials{
      context,
      {splits * static_cast<int64_t>(M), N},
      DType::Float32,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct GemvSplitBlock final {
    ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical sizes
    int k_texels;
    int chunk;
    int splits;
    int fill;
  } split_block{
      api::utils::make_whcn_ivec4({M, N}),
      static_cast<int32_t>(k_texels),
      static_cast<int32_t>(chunk),
      static_cast<int32_t>(splits),
      0,
  };

  api::UniformParamsBuffer split_params(context, split_block);
  api::PipelineBarrier split_barrier{};

  const api::utils::uvec3 split_grid{
      col_jobs,
      splits,
      api::utils::safe_downcast_to_u32(M),
  };

  context->submit_compute_job(
      VK_KERNEL(gemv_split),
      split_barrier,
      split_grid,
      adaptive_work_group_size(split_grid),
      VK_NULL_HANDLE,
      v_partials.image(
          split_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_lhs.image(split_barrier, api::PipelineStage::COMPUTE),
      v_rhs.image(split_barrier, api::PipelineStage::COMPUTE),
      split_params.buffer());

  // The addend broadcasts over singleton rows/columns; a bias-free product
  // carries no addend term, expressed here as a zero beta alongside a
  // stand-in binding the epilogue never fetches.
  api::vTensor v_bias = bias.has_value() ? convert(*bias) : v_partials;
  const float alpha_f = alpha.to<float>();
  const float beta_f = bias.has_value() ? beta.to<float>() : 0.0f;
  const std::vector<int64_t> bias_sizes = bias.has_value()
      ? shape_of(*bias)
      : std::vector<int64_t>({1, 1});

  const struct GemvCombineBlock final {
    ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical sizes
    ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
    int splits;
    float alpha;
    float beta;
  } combine_block{
      api::utils::make_whcn_ivec4(v_output.sizes()),
      api::utils::make_whcn_ivec4(bias_sizes),
      static_cast<int32_t>(splits),
      alpha_f,
      beta_f,
  };

  api::UniformParamsBuffer combine_params(context, combine_block);
  api::PipelineBarrier combine_barrier{};

  const api::utils::uvec3 combine_grid{
      col_jobs,
      api::utils::safe_downcast_to_u32(M),
      1u,
  };

  context->submit_compute_job(
      kernel_for("gemv_combine", dtype),
      combine_barrier,
      combine_grid,
      adaptive_work_group_size(combine_grid),
      VK_NULL_HANDLE,
      v_output.image(
          combine_barrier, api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_partials.image(combine_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(combine_barrier, api::PipelineStage::COMPUTE),
      combine_params.buffer());

  return convert(v_output);
}

/*
 * Decode products over a buffer-stored weight.  A weight whose extents pass
 * the device's 3-D image axis limit has no texture form at all, so the split
 * pass reads it as a row-major linear payload: one invocation per output
 * column and per chunk of the reduction axis.  The caller describes where the
 * weight keeps a column step (`stride_n`) and a reduction step (`stride_k`),
 * which lets a transposed (K x N) operand and the row-major (N x K) operand a
 * linear layer stores share one kernel family.  The activation operand still
 * rides a width-packed texture for as long as its own row fits an image; past
 * that point it too streams from a buffer.
 *
 * The chunk target is looser than the texture path's because every column now
 * gets its own invocation, so the same output width already buys four times
 * the parallelism.
 */
constexpr int64_t kGemvLinearSpanElements = 256;

// The decode kernels read activations either from a width-packed texture or
// from their linear payload, whichever the activation row actually fits.
api::vTensor gemv_operand_lhs(const Tensor& src) {
  api::vTensor v_src = convert(src);
  if (v_src.storage_type() == api::StorageType::BUFFER) {
    return v_src;
  }
  return (v_src.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED)
      ? v_src
      : packed_cached(
            src, shape_of(src),
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, kTagMmWidthPacked,
            [&] { return pack_lhs_width(src); });
}

Tensor gemv_decode_linear(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t stride_n,
    int64_t stride_k,
    const std::optional<Tensor>& bias,
    Scalar beta,
    Scalar alpha,
    DType dtype) {
  api::Context* const context = api::context();

  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16,
      "Vulkan products over buffer-stored weights support Float32 and "
      "Float16 only");
  // The payload families differ in how a word is read, so every operand has to
  // carry the element type the result is computed in.
  TP_CHECK(
      v_lhs.dtype() == dtype && v_rhs.dtype() == dtype,
      "Vulkan products over buffer-stored weights need operands of the "
      "result dtype");
  TP_CHECK(
      M <= kGemvMaxRows,
      "Vulkan products over buffer-stored weights support decode shapes only");

  const bool lhs_buffer =
      v_lhs.storage_type() == api::StorageType::BUFFER;
  // A unit reduction step means the weight rows run along the reduction axis,
  // so each output column streams one contiguous stretch of it.
  const bool rhs_nk = (stride_k == 1);
  // Walking a row four words per fetch needs every row to start on a vector
  // boundary; a reduction length that is not a multiple of four leaves the
  // tail ragged and stays on the element-at-a-time path.
  const bool rhs_vec4 = rhs_nk && (K % 4 == 0) && (stride_n == K);

  const uint32_t k_texels =
      api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u);
  const uint32_t column_row_jobs =
      api::utils::safe_downcast_to_u32(N * M);
  uint32_t splits = std::max(
      api::utils::div_up(
          api::utils::safe_downcast_to_u32(K),
          static_cast<uint32_t>(kGemvLinearSpanElements)),
      column_row_jobs > 0u
          ? api::utils::div_up(
                static_cast<uint32_t>(kGemvThreadBudget), column_row_jobs)
          : 1u);
  splits = std::max(splits, 1u);
  splits = std::min(splits, static_cast<uint32_t>(kGemvSplitMax));
  splits = std::min(splits, k_texels);
  // Split boundaries open on texel edges so the packed operand never starts a
  // chunk in the middle of a group; a chunk therefore covers at least four
  // elements.
  const uint32_t chunk = 4u * api::utils::div_up(k_texels, splits);

  api::vTensor v_output{context, {M, N}, dtype};
  const bool out_buffer =
      v_output.storage_type() == api::StorageType::BUFFER;

  api::vTensor v_partials{
      context,
      {static_cast<int64_t>(splits) * M, N},
      DType::Float32,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct GemvLinearSplitBlock final {
    ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
    int k_elements;  // length of the reduction axis
    int chunk;       // reduction elements covered by one split
    int splits;
    int stride_n;    // element step to the next weight column
    int stride_k;    // element step to the next weight row
    int fill;
  } split_block{
      api::utils::make_whcn_ivec4({M, N}),
      static_cast<int32_t>(K),
      static_cast<int32_t>(chunk),
      static_cast<int32_t>(splits),
      static_cast<int32_t>(stride_n),
      static_cast<int32_t>(stride_k),
      0,
  };

  api::UniformParamsBuffer split_params(context, split_block);
  api::PipelineBarrier split_barrier{};

  std::string split_shader;
  if (rhs_vec4) {
    split_shader = lhs_buffer ? "gemv_linear_split_buffer_nk_vec4"
                              : "gemv_linear_split_nk_vec4";
  } else if (rhs_nk) {
    split_shader =
        lhs_buffer ? "gemv_linear_split_buffer_nk" : "gemv_linear_split_nk";
  } else {
    split_shader = lhs_buffer ? "gemv_linear_split_buffer" : "gemv_linear_split";
  }
  split_shader = buffer_shader_for_dtype(split_shader, dtype);

  const api::utils::uvec3 split_grid{
      api::utils::safe_downcast_to_u32(N),
      splits,
      api::utils::safe_downcast_to_u32(M),
  };

  if (lhs_buffer) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(split_shader),
        split_barrier,
        split_grid,
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_partials.buffer(
            split_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.buffer(split_barrier, api::PipelineStage::COMPUTE),
        v_rhs.buffer(split_barrier, api::PipelineStage::COMPUTE),
        split_params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(split_shader),
        split_barrier,
        split_grid,
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_partials.buffer(
            split_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.image(split_barrier, api::PipelineStage::COMPUTE),
        v_rhs.buffer(split_barrier, api::PipelineStage::COMPUTE),
        split_params.buffer());
  }

  const float alpha_f = alpha.to<float>();
  const float beta_f = bias.has_value() ? beta.to<float>() : 0.0f;

  std::optional<api::vTensor> v_bias;
  if (bias.has_value()) {
    v_bias = convert(*bias);
    // The addend binding follows the result's storage, because a linear output
    // reads it as a flat array while an image output fetches it as a texel.
    TP_CHECK(
        v_bias->storage_type() == v_output.storage_type(),
        "Vulkan products over buffer-stored weights require an addend in the "
        "same storage as the result");
    // A linear addend is read as words of the result's element type, while an
    // image one hands back floats whatever its format, so only the former
    // needs to match.
    TP_CHECK(
        v_output.storage_type() != api::StorageType::BUFFER ||
            v_bias->dtype() == dtype,
        "Vulkan products over buffer-stored weights need a linear addend of "
        "the result dtype");
  }
  const std::vector<int64_t> bias_sizes = bias.has_value()
      ? shape_of(*bias)
      : std::vector<int64_t>({1, 1});

  const struct GemvLinearCombineBlock final {
    ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical product sizes
    ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
    int splits;       // partial rows folded per output element
    int fill;
    float alpha;
    float beta;
  } combine_block{
      api::utils::make_whcn_ivec4(v_output.sizes()),
      api::utils::make_whcn_ivec4(bias_sizes),
      static_cast<int32_t>(splits),
      0,
      alpha_f,
      beta_f,
  };

  std::string combine_shader = out_buffer
      ? std::string(v_bias.has_value() ? "gemv_linear_combine_buffer"
                                       : "gemv_linear_combine_buffer_nobias")
      : std::string(v_bias.has_value() ? "gemv_linear_combine"
                                       : "gemv_linear_combine_nobias");
  combine_shader = buffer_shader_for_dtype(combine_shader, dtype);

  api::UniformParamsBuffer combine_params(context, combine_block);
  api::PipelineBarrier combine_barrier{};

  // A half result in linear storage folds two elements per word, so the grid
  // counts words; an image result keeps one element per texel channel and
  // spreads over the product's own axes.
  const api::utils::uvec3 combine_grid = (out_buffer && dtype == DType::Float16)
      ? api::utils::uvec3{
            api::utils::div_up(
                api::utils::safe_downcast_to_u32(M * N), 2u),
            1u,
            1u}
      : api::utils::uvec3{
            api::utils::safe_downcast_to_u32(N),
            api::utils::safe_downcast_to_u32(M),
            1u};

  if (out_buffer) {
    if (v_bias.has_value()) {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(combine_shader),
          combine_barrier,
          combine_grid,
          {64u, 1u, 1u},
          VK_NULL_HANDLE,
          v_output.buffer(
              combine_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_partials.buffer(combine_barrier, api::PipelineStage::COMPUTE),
          combine_params.buffer(),
          v_bias->buffer(combine_barrier, api::PipelineStage::COMPUTE));
    } else {
      context->submit_compute_job(
          VK_KERNEL_FROM_STR(combine_shader),
          combine_barrier,
          combine_grid,
          {64u, 1u, 1u},
          VK_NULL_HANDLE,
          v_output.buffer(
              combine_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_partials.buffer(combine_barrier, api::PipelineStage::COMPUTE),
          combine_params.buffer());
    }
  } else if (v_bias.has_value()) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(combine_shader),
        combine_barrier,
        combine_grid,
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.image(
            combine_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_partials.buffer(combine_barrier, api::PipelineStage::COMPUTE),
        combine_params.buffer(),
        v_bias->image(combine_barrier, api::PipelineStage::COMPUTE));
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(combine_shader),
        combine_barrier,
        combine_grid,
        {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.image(
            combine_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_partials.buffer(combine_barrier, api::PipelineStage::COMPUTE),
        combine_params.buffer());
  }

  return convert(v_output);
}

/*
 * Products too wide for a texture, at any row count.  The decode kernels above
 * give every output column its own invocation and never restages an operand,
 * which is optimal while a handful of rows makes each weight word a
 * contributing one; past that point the same words get re-read once per row and
 * the product becomes bound by how much it moves rather than by how many
 * threads it has.
 *
 * Staging a 32x32 block of each operand through shared memory bounds that
 * re-reading at 32 uses per fetch, which is also the only way a row-major
 * weight can be read without striding every warp across the whole matrix: the
 * staging pass hands consecutive lanes to whichever operand axis runs along
 * memory.  One workgroup of 8x8 threads covers a 32x32 output block and each
 * thread keeps a 4x4 accumulator.
 */
constexpr int64_t kMmBufferTile = 32;
constexpr int64_t kMmBufferTileSide = 8;

Tensor mm_buffer_tiled(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t a_stride_m,
    int64_t a_stride_k,
    int64_t b_stride_n,
    int64_t b_stride_k,
    const std::optional<Tensor>& bias,
    Scalar beta,
    Scalar alpha,
    DType dtype) {
  api::Context* const context = api::context();

  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16,
      "Vulkan products over buffer-stored operands support Float32 and "
      "Float16 only");
  TP_CHECK(
      v_lhs.dtype() == dtype && v_rhs.dtype() == dtype,
      "Vulkan products over buffer-stored operands need operands of the "
      "result dtype");

  const bool a_buffer = v_lhs.storage_type() == api::StorageType::BUFFER;

  api::vTensor v_output{context, {M, N}, dtype};
  const bool out_buffer =
      v_output.storage_type() == api::StorageType::BUFFER;
  // A thread of the tiled kernel owns four consecutive columns, so a half
  // result pairs them into words only when a row starts on an even element.
  TP_CHECK(
      !(out_buffer && dtype == DType::Float16 && N % 2 != 0),
      "Vulkan products over buffer-stored operands need an even result width "
      "for a half linear output");

  std::optional<api::vTensor> v_bias;
  if (bias.has_value()) {
    v_bias = convert(*bias);
    // The addend binding follows the result's storage, because a linear output
    // reads it as a flat array while an image output fetches it as a texel.
    TP_CHECK(
        v_bias->storage_type() == v_output.storage_type(),
        "Vulkan products over buffer-stored operands require an addend in the "
        "same storage as the result");
    TP_CHECK(
        v_output.storage_type() != api::StorageType::BUFFER ||
            v_bias->dtype() == dtype,
        "Vulkan products over buffer-stored operands need a linear addend of "
        "the result dtype");
  }
  const std::vector<int64_t> bias_sizes = bias.has_value()
      ? shape_of(*bias)
      : std::vector<int64_t>({1, 1});

  const struct MmBufferTiledBlock final {
    ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
    ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
    int k_elements; // length of the reduction axis
    int k_blocks; // reduction tiles to walk
    int a_stride_m; // element step to the next left row
    int a_stride_k; // element step to the next left column
    int b_stride_n; // element step to the next right column
    int b_stride_k; // element step to the next right row
    float alpha;
    float beta;
  } block{
      api::utils::make_whcn_ivec4({M, N}),
      api::utils::make_whcn_ivec4(bias_sizes),
      static_cast<int32_t>(K),
      static_cast<int32_t>(
          api::utils::div_up(api::utils::safe_downcast_to_u32(K),
                             static_cast<uint32_t>(kMmBufferTile))),
      static_cast<int32_t>(a_stride_m),
      static_cast<int32_t>(a_stride_k),
      static_cast<int32_t>(b_stride_n),
      static_cast<int32_t>(b_stride_k),
      alpha.to<float>(),
      bias.has_value() ? beta.to<float>() : 0.0f,
  };

  std::string shader = "mm_buffer_tiled";
  if (a_buffer) {
    shader += "_lhs";
  }
  if (out_buffer) {
    shader += "_buffer";
  }
  if (v_bias.has_value()) {
    shader += "_bias";
  }
  const std::string shader_name = buffer_shader_for_dtype(shader, dtype);

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};

  // Every workgroup item owns four output entries on each axis.
  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(
          api::utils::div_up(api::utils::safe_downcast_to_u32(N),
                            static_cast<uint32_t>(kMmBufferTile)) *
          kMmBufferTileSide),
      api::utils::safe_downcast_to_u32(
          api::utils::div_up(api::utils::safe_downcast_to_u32(M),
                            static_cast<uint32_t>(kMmBufferTile)) *
          kMmBufferTileSide),
      1u,
  };

  const api::utils::uvec3 local{8u, 8u, 1u};
  const bool has_bias = v_bias.has_value();

  if (out_buffer) {
    if (a_buffer) {
      if (has_bias) {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.buffer(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer(),
            v_bias->buffer(barrier, api::PipelineStage::COMPUTE));
      } else {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.buffer(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer());
      }
    } else {
      if (has_bias) {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.buffer(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.image(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer(),
            v_bias->buffer(barrier, api::PipelineStage::COMPUTE));
      } else {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.buffer(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.image(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer());
      }
    }
  } else {
    if (a_buffer) {
      if (has_bias) {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.image(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer(),
            v_bias->image(barrier, api::PipelineStage::COMPUTE));
      } else {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.image(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer());
      }
    } else {
      if (has_bias) {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.image(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.image(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer(),
            v_bias->image(barrier, api::PipelineStage::COMPUTE));
      } else {
        context->submit_compute_job(
            VK_KERNEL_FROM_STR(shader_name),
            barrier,
            grid,
            local,
            VK_NULL_HANDLE,
            v_output.image(
                barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_lhs.image(barrier, api::PipelineStage::COMPUTE),
            v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
            params.buffer());
      }
    }
  }

  return convert(v_output);
}

/*
 * Products over a right operand with no texture form.  A decode-shaped one
 * never restages an operand, so the split reduction wins there; once more than
 * a handful of rows share the same weight, the staged tiled product is the one
 * that keeps the reads minimal.
 */
Tensor buffer_product(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t b_stride_n,
    int64_t b_stride_k,
    const std::optional<Tensor>& bias,
    Scalar beta,
    Scalar alpha,
    DType dtype) {
  if (M <= kGemvMaxRows) {
    return gemv_decode_linear(
        v_lhs,
        v_rhs,
        M,
        N,
        K,
        b_stride_n,
        b_stride_k,
        bias,
        beta,
        alpha,
        dtype);
  }
  // Every linear activation reaching here is a dense row-major (M x K)
  // payload, so its own extents are its strides.
  return mm_buffer_tiled(
      v_lhs,
      v_rhs,
      M,
      N,
      K,
      K,
      1,
      b_stride_n,
      b_stride_k,
      bias,
      beta,
      alpha,
      dtype);
}

Tensor product_dispatch(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t M,
    int64_t N,
    int64_t K,
    std::optional<Tensor> bias,
    Scalar beta,
    Scalar alpha,
    DType dtype) {
  api::Context* const context = api::context();

  // Decode-shape products take the split-reduction path; see gemv_decode.
  if (M <= kGemvMaxRows && K >= kGemvMinReduction) {
    return gemv_decode(
        v_lhs, v_rhs, M, N, K, bias, beta, alpha, dtype);
  }

  api::vTensor v_output{context, {M, N}, dtype};

  TP_CHECK(
      v_output.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan mm requires texture storage");

  const int32_t step_size = static_cast<int32_t>(
      api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u));
  const float alpha_f = alpha.to<float>();
  const float beta_f = beta.to<float>();

  if (bias.has_value()) {
    TP_CHECK(
        bias->device().is_vulkan(),
        "Vulkan addmm: the addend must live on the Vulkan device");
    api::vTensor v_bias = convert(*bias);

    const struct AddMMBlock final {
      ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical sizes
      ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
      int step_size;
      float alpha;
      float beta;
    } block{
        make_whcn_ivec4(v_output.sizes()),
        make_whcn_ivec4(v_bias.sizes()),
        step_size,
        alpha_f,
        beta_f,
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    // Each invocation produces a 4x4 output tile.
    const api::utils::uvec3 global_size{
        api::utils::div_up(api::utils::safe_downcast_to_u32(N), 4u),
        api::utils::div_up(api::utils::safe_downcast_to_u32(M), 4u),
        1u,
    };

    context->submit_compute_job(
        kernel_for("addmm", dtype),
        pipeline_barrier,
        global_size,
        {8u, 8u, 1u},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_rhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    const struct MMBlock final {
      ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical sizes
      int step_size;   // number of K texels: ceil(K / 4)
    } block{
        make_whcn_ivec4(v_output.sizes()),
        step_size,
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    const api::utils::uvec3 global_size{
        api::utils::div_up(api::utils::safe_downcast_to_u32(N), 4u),
        api::utils::div_up(api::utils::safe_downcast_to_u32(M), 4u),
        1u,
    };

    context->submit_compute_job(
        kernel_for("mm", dtype),
        pipeline_barrier,
        global_size,
        {8u, 8u, 1u},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_rhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return convert(v_output);
}

Tensor mm_impl(
    const Tensor& self,
    const Tensor& mat2,
    std::optional<Tensor> bias,
    Scalar beta,
    Scalar alpha) {
  const int64_t M = self.size(0);
  const int64_t N = mat2.size(1);
  const int64_t K = self.size(1);

  if (M == 0 || N == 0 || K == 0) {
    // Zero-sized operands: with K == 0 the product term is zero by
    // definition and only the epilogue remains; empty results need no work.
    api::Context* const context = api::context();
    api::vTensor v_output{context, {M, N}, self.dtype()};
    Tensor out = convert(v_output);
    if (K == 0 && M != 0 && N != 0 && bias.has_value()) {
      out.fill_(beta.to<float>() * bias->item().to<float>());
    } else if (K == 0 && M != 0 && N != 0) {
      out.fill_(Scalar(0.0));
    }
    return out;
  }

  api::vTensor v_self = convert(self);
  api::vTensor v_other = convert(mat2);

  // A weight whose extents pass the image axis limit has no texture form at
  // all; decode products stream it straight out of its linear payload rather
  // than relaying it.
  if (v_other.storage_type() == api::StorageType::BUFFER) {
    api::vTensor lhs = gemv_operand_lhs(self);
    return buffer_product(
        lhs, v_other, M, N, K, 1, N, bias, beta, alpha, self.dtype());
  }

  TP_CHECK(
      v_self.storage_type() == api::StorageType::TEXTURE_3D &&
          v_other.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan mm requires texture storage");

  // The tiled kernel reduces along four-wide texel lanes, so both operands
  // stream through relayout passes that align the K axis with the lanes.
  // The relayout shaders zero-fill lanes past the K edge, which keeps the
  // tail step of non-multiple-of-4 K harmless.  Each packing is cached per
  // source storage identity.
  api::vTensor lhs =
      (v_self.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED)
      ? v_self
      : packed_cached(
            self, shape_of(self),
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, kTagMmWidthPacked,
            [&] { return pack_lhs_width(self); });

  api::vTensor rhs =
      (v_other.gpu_memory_layout() ==
       api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED)
      ? v_other
      : packed_cached(
            mat2, shape_of(mat2),
            api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, kTagMmHeightPacked,
            [&] { return pack_rhs_height(mat2); });

  return product_dispatch(
      lhs, rhs, M, N, K, std::move(bias), beta, alpha, self.dtype());
}

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
  TP_CHECK(
      (self.dtype() == DType::Float32 || self.dtype() == DType::Float16) &&
          (mat2.dtype() == DType::Float32 || mat2.dtype() == DType::Float16),
      "Vulkan mm supports Float32 and Float16 tensors only");
  TP_CHECK(self.dim() == 2, "Vulkan mm: self must be a matrix");
  TP_CHECK(mat2.dim() == 2, "Vulkan mm: mat2 must be a matrix");
  TP_CHECK(
      self.size(1) == mat2.size(0),
      "Vulkan mm: matrix dimensions do not match");

  return mm_impl(self, mat2, std::nullopt, Scalar(1.0), Scalar(1.0));
}

Tensor addmm_kernel(
    const Tensor& bias,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  TP_CHECK(
      (mat1.dtype() == DType::Float32 || mat1.dtype() == DType::Float16) &&
          (mat2.dtype() == DType::Float32 || mat2.dtype() == DType::Float16),
      "Vulkan addmm supports Float32 and Float16 tensors only");
  TP_CHECK(mat1.dim() == 2, "Vulkan addmm: mat1 must be a matrix");
  TP_CHECK(mat2.dim() == 2, "Vulkan addmm: mat2 must be a matrix");
  TP_CHECK(
      mat1.size(1) == mat2.size(0),
      "Vulkan addmm: matrix dimensions do not match");

  // The addend broadcasts over singleton rows/columns; a 1d addend is a row.
  Tensor bias_vulkan = bias;
  std::optional<Tensor> bias_owned;
  if (bias.dim() == 1) {
    bias_owned = bias.unsqueeze(0);
    bias_vulkan = *bias_owned;
  }
  if (!bias_vulkan.device().is_vulkan()) {
    bias_owned = bias_vulkan.to(Device(DeviceType::Vulkan));
    bias_vulkan = *bias_owned;
  }
  TP_CHECK(
      bias_vulkan.dim() == 2 &&
          (bias_vulkan.size(0) == mat1.size(0) ||
           bias_vulkan.size(0) == 1) &&
          (bias_vulkan.size(1) == mat2.size(1) ||
           bias_vulkan.size(1) == 1),
      "Vulkan addmm: addend is not broadcastable to the result shape");

  return mm_impl(mat1, mat2, bias_vulkan, beta, alpha);
}

/*
 * Whether a pair of operands can fold to the plain 2d product.  That route
 * relays both operands through images, so an activation whose row axis passes
 * the device's image limit has no texture to relay into.  The batched kernels
 * read such a payload straight out of linear memory while the weight keeps its
 * image, so the product stays batched with its size-1 batch.
 */
bool folds_to_mm(const Tensor& batch1, const Tensor& batch2) {
  const api::vTensor v_lhs = convert(batch1);
  const api::vTensor v_rhs = convert(batch2);
  return !(
      v_lhs.storage_type() == api::StorageType::BUFFER &&
      v_rhs.storage_type() == api::StorageType::TEXTURE_3D);
}

/*
 * Batched products with no image form.  A batched product whose result rows,
 * result columns or batch planes run past the device's image limit has no
 * texture to write, and a long context grows all three axes at once.  One
 * dispatch still covers every matrix: the staged tiled kernel takes the batch
 * count on its z axis and reaches the next matrix through an element stride,
 * so nothing is restaged between batches.
 *
 * An operand keeps whichever form it already has.  One that stayed an image is
 * packed into per-batch planes by the same cached pass the batched texture
 * kernel uses; one that lost its image form is streamed out of its own linear
 * payload, since relaying it through an image would cost as much again as the
 * product.  Both cases meet in the kernel's two addressing modes.
 */
constexpr int64_t kLinearIndexLimit = 2147483647;

/*
 * A linear product inherits its payload from the route that had to stream its
 * operands, but storage is otherwise a function of the shape alone: consumers
 * pick their route by looking at what they were handed, and a texture-only
 * kernel given a small buffer payload has no linear variant to run.  A result
 * whose shape still has an image therefore repacks into one on the way out.
 * The gather knows the linear layout of the types that fill or share a word --
 * single precision, integer codes, and halves two to a word -- so a product of
 * any of those crosses over; every other type stays where the kernel wrote it.
 */
Tensor release_product_output(
    api::Context* const context,
    api::vTensor& v_product) {
  Tensor flat = convert(v_product);
  const std::vector<int64_t> sizes(
      v_product.sizes().begin(), v_product.sizes().end());
  if (v_product.storage_type() != api::StorageType::BUFFER ||
      !image_storage_supported(context, sizes)) {
    return flat;
  }
  const DType dtype = v_product.dtype();
  if (dtype != DType::Float32 && dtype != DType::Int32 &&
      dtype != DType::Float16) {
    return flat;
  }
  return materialize_view(
      flat, sizes, SizesAndStrides::compute_contiguous_strides(sizes), 0);
}

Tensor bmm_buffer_tiled(
    const api::vTensor& v_lhs,
    const api::vTensor& v_rhs,
    int64_t batches,
    int64_t M,
    int64_t N,
    int64_t K,
    DType dtype) {
  api::Context* const context = api::context();

  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16,
      "Vulkan batched products over linear storage support Float32 and "
      "Float16 only");
  TP_CHECK(
      v_lhs.dtype() == dtype && v_rhs.dtype() == dtype,
      "Vulkan batched products over linear storage need operands of the "
      "result dtype");

  const bool a_packed = v_lhs.storage_type() != api::StorageType::BUFFER;
  const bool b_packed = v_rhs.storage_type() != api::StorageType::BUFFER;
  // Two image operands are the batched texture kernel's own case; reaching
  // here means at least one payload has no image form.
  TP_CHECK(
      !(a_packed && b_packed),
      "Vulkan batched products over linear storage need a linear operand");

  // Every address in the kernel is a 32-bit element index, and the batch
  // strides multiply what a single matrix already spans.
  TP_CHECK(
      batches * M * K <= kLinearIndexLimit &&
          batches * K * N <= kLinearIndexLimit &&
          batches * M * N <= kLinearIndexLimit,
      "Vulkan batched products over linear storage exceed the element index "
      "range a 32-bit offset can reach");

  api::vTensor v_output{
      context,
      {batches, M, N},
      dtype,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  // A thread owns four consecutive columns of each result row, and a word of a
  // half payload holds one such pair, so a matrix must start on an even
  // element: the host refuses an odd width rather than letting two matrices
  // claim one word.
  TP_CHECK(
      !(dtype == DType::Float16 && N % 2 != 0),
      "Vulkan batched products over linear storage need an even result width "
      "for a half linear output");

  // A linear payload is dense in the operand's own logical order, so its
  // extents are its strides.  A single matrix serving every batch steps by
  // nothing, so each z lane reads the same planes.
  const int64_t a_matrices = v_lhs.sizes().front();
  const int64_t b_matrices = v_rhs.sizes().front();
  const int32_t a_stride_b =
      (a_matrices == 1 && batches > 1)
          ? 0
          : static_cast<int32_t>(M * K);
  const int32_t b_stride_b =
      (b_matrices == 1 && batches > 1)
          ? 0
          : static_cast<int32_t>(K * N);

  const struct BmmBufferTiledBlock final {
    ivec4 out_sizes; // (W=N, H=M, C=B, N=1) logical product sizes
    int k_elements; // length of the reduction axis
    int k_blocks; // reduction tiles to walk
    int a_stride_m; // element step to the next left row
    int a_stride_k; // element step to the next left column
    int a_stride_b; // element step to the next left matrix
    int b_stride_n; // element step to the next right column
    int b_stride_k; // element step to the next right row
    int b_stride_b; // element step to the next right matrix
    int out_stride_b; // element step to the next result matrix
  } block{
      api::utils::make_whcn_ivec4({batches, M, N}),
      static_cast<int32_t>(K),
      static_cast<int32_t>(
          api::utils::div_up(
              api::utils::safe_downcast_to_u32(K),
              static_cast<uint32_t>(kMmBufferTile))),
      static_cast<int32_t>(K),
      1,
      a_stride_b,
      1,
      static_cast<int32_t>(N),
      b_stride_b,
      static_cast<int32_t>(M * N),
  };

  std::string shader = "bmm_buffer_tiled";
  if (a_packed) {
    shader += "_a";
  } else if (b_packed) {
    shader += "_b";
  }
  const std::string shader_name = buffer_shader_for_dtype(shader, dtype);

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};

  // A workgroup of 8x8 items owns one 32x32 output block of one matrix, and
  // the z axis walks the matrices.
  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(
          api::utils::div_up(
              api::utils::safe_downcast_to_u32(N),
              static_cast<uint32_t>(kMmBufferTile)) *
          kMmBufferTileSide),
      api::utils::safe_downcast_to_u32(
          api::utils::div_up(
              api::utils::safe_downcast_to_u32(M),
              static_cast<uint32_t>(kMmBufferTile)) *
          kMmBufferTileSide),
      api::utils::safe_downcast_to_u32(batches),
  };

  const api::utils::uvec3 local{8u, 8u, 1u};

  if (a_packed) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader_name),
        barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.image(barrier, api::PipelineStage::COMPUTE),
        v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else if (b_packed) {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader_name),
        barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
        v_rhs.image(barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader_name),
        barrier,
        grid,
        local,
        VK_NULL_HANDLE,
        v_output.buffer(
            barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_lhs.buffer(barrier, api::PipelineStage::COMPUTE),
        v_rhs.buffer(barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  return release_product_output(context, v_output);
}

//
// Batched product: one dispatch covers the whole batch.  The result keeps
// the standard channel-packed batched layout (four batches per texel z
// slot, one per lane); operands carry one packed plane per batch, with
// size-1 sources broadcasting across the planes inside the pack gather.
//
Tensor bmm_impl(
    const Tensor& batch1,
    const Tensor& batch2,
    int64_t batches) {
  const int64_t M = batch1.size(batch1.dim() - 2);
  const int64_t K = batch1.size(batch1.dim() - 1);
  const int64_t N = batch2.size(batch2.dim() - 1);

  api::Context* const context = api::context();

  if (batches == 0 || M == 0 || N == 0 || K == 0) {
    api::vTensor v_result{
        context,
        {batches, M, N},
        batch1.dtype(),
        api::StorageType::TEXTURE_3D,
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    };
    Tensor out = convert(v_result);
    if (M != 0 && N != 0) {
      out.fill_(Scalar(0.0));
    }
    return out;
  }

  api::vTensor v_raw_lhs = convert(batch1);
  api::vTensor v_raw_rhs = convert(batch2);

  // A result with no image form, or an operand that lost one, runs the batched
  // staged product over linear storage instead.  Whichever operand kept an
  // image still goes through the per-batch pack.
  if (!image_storage_supported(context, {batches, M, N}) ||
      v_raw_lhs.storage_type() == api::StorageType::BUFFER ||
      v_raw_rhs.storage_type() == api::StorageType::BUFFER) {
    api::vTensor lhs =
        v_raw_lhs.storage_type() == api::StorageType::BUFFER
        ? v_raw_lhs
        : packed_cached(
              batch1, shape_of(batch1),
              api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
              batched_tag(kTagMmWidthPackedBatched, batches),
              [&] { return bmm_pack_lhs_width(batch1, batches); });
    api::vTensor rhs =
        v_raw_rhs.storage_type() == api::StorageType::BUFFER
        ? v_raw_rhs
        : packed_cached(
              batch2, shape_of(batch2),
              api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
              batched_tag(kTagMmHeightPackedBatched, batches),
              [&] { return bmm_pack_rhs_height(batch2, batches); });
    return bmm_buffer_tiled(lhs, rhs, batches, M, N, K, batch1.dtype());
  }

  api::vTensor v_result{
      context,
      {batches, M, N},
      batch1.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  TP_CHECK(
      v_result.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan bmm requires texture storage");

  api::vTensor lhs = packed_cached(
      batch1, shape_of(batch1),
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      batched_tag(kTagMmWidthPackedBatched, batches),
      [&] { return bmm_pack_lhs_width(batch1, batches); });

  api::vTensor rhs = packed_cached(
      batch2, shape_of(batch2),
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      batched_tag(kTagMmHeightPackedBatched, batches),
      [&] { return bmm_pack_rhs_height(batch2, batches); });

  const struct BMMBlock final {
    ivec4 out_sizes; // (W=N, H=M, C=B, N=1) logical sizes
    int step_size;   // number of K texels: ceil(K / 4)
  } block{
      make_whcn_ivec4(v_result.sizes()),
      static_cast<int32_t>(
          api::utils::div_up(api::utils::safe_downcast_to_u32(K), 4u)),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  // Each invocation produces a 4x4 output tile across up to four batches.
  const api::utils::uvec3 global_size{
      api::utils::div_up(api::utils::safe_downcast_to_u32(N), 4u),
      api::utils::div_up(api::utils::safe_downcast_to_u32(M), 4u),
      api::utils::div_up(api::utils::safe_downcast_to_u32(batches), 4u),
  };

  context->submit_compute_job(
      kernel_for("bmm", batch1.dtype()),
      pipeline_barrier,
      global_size,
      {4u, 4u, 1u},
      VK_NULL_HANDLE,
      v_result.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      lhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      rhs.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_result);
}

Tensor bmm_kernel(const Tensor& batch1, const Tensor& batch2) {
  TP_CHECK(
      (batch1.dtype() == DType::Float32 || batch1.dtype() == DType::Float16) &&
          (batch2.dtype() == DType::Float32 || batch2.dtype() == DType::Float16),
      "Vulkan bmm supports Float32 and Float16 tensors only");
  TP_CHECK(batch1.dim() == 3, "Vulkan bmm: batch1 must be 3d");
  TP_CHECK(batch2.dim() == 3, "Vulkan bmm: batch2 must be 3d");
  const int64_t B = batch1.size(0);
  const int64_t M = batch1.size(1);
  const int64_t K = batch1.size(2);
  const int64_t N = batch2.size(2);
  TP_CHECK(
      B == batch2.size(0),
      "Vulkan bmm: batch dimensions do not match");
  TP_CHECK(
      K == batch2.size(1),
      "Vulkan bmm: matrix dimensions do not match");

  // A single batch is a plain matrix product; fold to the 2d kernel unless an
  // operand left the 2d route without a texture to relay.
  if (B == 1 && folds_to_mm(batch1, batch2)) {
    Tensor flat1 = batch1.reshape({M, K});
    Tensor flat2 = batch2.reshape({K, N});
    return mm_kernel(flat1, flat2).reshape({1, M, N});
  }

  return bmm_impl(batch1, batch2, B);
}

/*
 * Batched product with a scaled addend: out = beta * self + alpha * bmm.
 * The product reuses the batched dispatches; the addend applies through
 * the broadcast add in the epilogue, accepting (B, M, N), (M, N) or
 * broadcastable row/column shapes like the 2d addmm.
 */
Tensor baddbmm_kernel(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  TP_CHECK(
      (batch1.dtype() == DType::Float32 || batch1.dtype() == DType::Float16) &&
          (batch2.dtype() == DType::Float32 || batch2.dtype() == DType::Float16),
      "Vulkan baddbmm supports Float32 and Float16 tensors only");
  TP_CHECK(batch1.dim() == 3, "Vulkan baddbmm: batch1 must be 3d");
  TP_CHECK(batch2.dim() == 3, "Vulkan baddbmm: batch2 must be 3d");
  const int64_t B = batch1.size(0);
  const int64_t M = batch1.size(1);
  const int64_t K = batch1.size(2);
  const int64_t N = batch2.size(2);
  TP_CHECK(
      B == batch2.size(0),
      "Vulkan baddbmm: batch dimensions do not match");
  TP_CHECK(
      K == batch2.size(1),
      "Vulkan baddbmm: matrix dimensions do not match");
  TP_CHECK(
      self.dim() <= 3,
      "Vulkan baddbmm: addend must be at most 3d");

  const std::vector<int64_t> target{B, M, N};

  Tensor addend = self;
  std::optional<Tensor> addend_owned;
  if (!addend.device().is_vulkan()) {
    addend_owned = addend.to(Device(DeviceType::Vulkan));
    addend = *addend_owned;
  }
  TP_CHECK(
      addend.numel() == 1 ||
          (addend.dim() == 3 && addend.shape() == Size(target)) ||
          (addend.dim() == 2 && addend.size(0) == M &&
           addend.size(1) == N) ||
          (addend.dim() == 2 && addend.size(0) == 1 &&
           addend.size(1) == N) ||
          (addend.dim() == 2 && addend.size(0) == M &&
           addend.size(1) == 1) ||
          (addend.dim() == 1 && addend.numel() == N),
      "Vulkan baddbmm: addend is not broadcastable to the result shape");

  Tensor product = bmm_kernel(batch1, batch2);
  Tensor scaled = product.mul(alpha);

  if (beta.to<double>() == 0.0) {
    return scaled;
  }

  // Broadcast the addend to the batched result shape; singletons expand
  // through zero-stride gathers before the scaled add.
  const auto addend_sizes =
      static_cast<std::vector<int64_t>>(addend.shape());
  Tensor broadcast_addend = addend;
  if (addend_sizes != target) {
    broadcast_addend = expand_kernel(addend, target, false).contiguous();
  }
  Tensor scaled_addend = broadcast_addend.mul(beta);
  return scaled.add(scaled_addend);
}

// Defined further down; a matrix against a vector folds to the decode
// product, so the (2,1) case below can reach it before its definition.
Tensor mv_kernel(const Tensor& self, const Tensor& vec);

Tensor matmul_kernel(const Tensor& self_in, const Tensor& other_in) {
  TP_CHECK(
      (self_in.dtype() == DType::Float32 || self_in.dtype() == DType::Float16) &&
          (other_in.dtype() == DType::Float32 || other_in.dtype() == DType::Float16),
      "Vulkan matmul supports Float32 and Float16 tensors only");

  const int64_t self_ndim = self_in.dim();
  const int64_t other_ndim = other_in.dim();

  // Vector · vector: elementwise product reduced over the single axis.
  if (self_ndim == 1 && other_ndim == 1) {
    TP_CHECK(
        self_in.size(0) == other_in.size(0),
        "Vulkan matmul: vector dimensions do not match");
    Tensor product = self_in * other_in;
    return product.sum(std::vector<int64_t>{0}, false);
  }

  // Matrix · vector: the same reduction over one axis, taken through the
  // decode product rather than a tile whose row axis holds a single lane.
  if (self_ndim == 2 && other_ndim == 1) {
    return mv_kernel(self_in, other_in);
  }

  // Vector · matrix: the vector reads as one row of the product, so the
  // result keeps the matrix's far axis.
  if (self_ndim == 1 && other_ndim == 2) {
    const int64_t K = self_in.size(0);
    TP_CHECK(
        K == other_in.size(0),
        "Vulkan matmul: matrix dimensions do not match");
    TP_CHECK(
        self_in.dtype() == other_in.dtype(),
        "Vulkan matmul: operands must have the same element type");
    Tensor row = mm_impl(
        self_in.reshape({1, K}),
        other_in,
        std::nullopt,
        Scalar(1.0),
        Scalar(1.0));
    return row.reshape({other_in.size(1)});
  }

  // Matrix · matrix: transpose views fold into the pack gathers, so no
  // operand is copied just to undo a layout swap.
  if (self_ndim == 2 && other_ndim == 2) {
    TP_CHECK(
        self_in.size(1) == other_in.size(0),
        "Vulkan matmul: matrix dimensions do not match");

    Tensor lhs_dense;
    Tensor rhs_dense;
    const bool lhs_transposed = transpose_view_of_dense(self_in, &lhs_dense);
    const bool rhs_transposed = transpose_view_of_dense(other_in, &rhs_dense);
    const Tensor& lhs_src = lhs_transposed ? lhs_dense : self_in;
    const Tensor& rhs_src = rhs_transposed ? rhs_dense : other_in;

    if (lhs_transposed && rhs_transposed) {
      // A^T B^T = (B A)^T: one plain product, transposed on the way out.
      Tensor t = mm_kernel(rhs_dense, lhs_dense);
      return t.transpose(0, 1).contiguous();
    }

    const int64_t M = self_in.size(0);
    const int64_t N = other_in.size(1);
    const int64_t K = self_in.size(1);

    api::vTensor v_rhs_dense = convert(rhs_src);
    if (v_rhs_dense.storage_type() == api::StorageType::BUFFER) {
      // A weight this size has no texture form to relay into.  The decode
      // kernels read it in whichever orientation it actually sits in: the
      // view's own [K, N] order, or the [N, K] order of a dense operand seen
      // through a transpose.
      api::vTensor v_lhs = lhs_transposed
          ? packed_cached(
                lhs_src, shape_of(lhs_src),
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
                kTagMmWidthPackedTransposed,
                [&] { return pack_lhs_width_transposed(lhs_src); })
          : gemv_operand_lhs(lhs_src);
      return buffer_product(
          v_lhs,
          v_rhs_dense,
          M,
          N,
          K,
          rhs_transposed ? K : 1,
          rhs_transposed ? 1 : N,
          std::nullopt,
          Scalar(1.0),
          Scalar(1.0),
          self_in.dtype());
    }

    api::vTensor v_lhs;
    api::vTensor v_rhs;
    if (lhs_transposed) {
      v_lhs = packed_cached(
          lhs_src, shape_of(lhs_src),
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
          kTagMmWidthPackedTransposed,
          [&] { return pack_lhs_width_transposed(lhs_src); });
    } else {
      v_lhs = packed_cached(
          lhs_src, shape_of(lhs_src),
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, kTagMmWidthPacked,
          [&] { return pack_lhs_width(lhs_src); });
    }
    if (rhs_transposed) {
      v_rhs = packed_cached(
          rhs_src, shape_of(rhs_src),
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
          kTagMmHeightPackedTransposed,
          [&] { return pack_rhs_height_transposed(rhs_src); });
    } else {
      v_rhs = packed_cached(
          rhs_src, shape_of(rhs_src),
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, kTagMmHeightPacked,
          [&] { return pack_rhs_height(rhs_src); });
    }

    return product_dispatch(
        v_lhs,
        v_rhs,
        self_in.size(0),
        other_in.size(1),
        self_in.size(1),
        std::nullopt,
        Scalar(1.0),
        Scalar(1.0),
        self_in.dtype());
  }

  // Batch of matrices against a shared matrix: the product folds into one
  // wide matrix product, so flatten the batch into the row axis, run the
  // tiled product once, and restore the batch dims.
  if (self_ndim >= 3 && other_ndim == 2) {
    const int64_t K = self_in.size(self_ndim - 1);
    TP_CHECK(
        K == other_in.size(0),
        "Vulkan matmul: matrix dimensions do not match");
    Tensor flat = self_in.reshape({self_in.numel() / K, K});
    Tensor out = mm_kernel(flat, other_in);
    const std::vector<int64_t> self_sizes = shape_of(self_in);
    std::vector<int64_t> out_sizes(self_sizes.begin(), self_sizes.end() - 1);
    out_sizes.push_back(other_in.size(1));
    return out.reshape(out_sizes);
  }

  // Batch of matrices against a shared vector: fold the batch into the row
  // axis, take the one reduction, and restore the batch dims.
  if (self_ndim >= 3 && other_ndim == 1) {
    const int64_t K = self_in.size(self_ndim - 1);
    TP_CHECK(
        K == other_in.size(0),
        "Vulkan matmul: matrix dimensions do not match");
    Tensor flat = self_in.reshape({self_in.numel() / K, K});
    Tensor out = mv_kernel(flat, other_in);
    const std::vector<int64_t> self_sizes = shape_of(self_in);
    std::vector<int64_t> out_sizes(self_sizes.begin(), self_sizes.end() - 1);
    return out.reshape(out_sizes);
  }

  // A shared vector against a batch of matrices: the vector reads as one row
  // of every plane, so the batched product runs with a singleton row axis that
  // drops out of the result.
  if (self_ndim == 1 && other_ndim >= 3) {
    const int64_t K = self_in.size(0);
    TP_CHECK(
        K == other_in.size(other_ndim - 2),
        "Vulkan matmul: matrix dimensions do not match");
    TP_CHECK(
        self_in.dtype() == other_in.dtype(),
        "Vulkan matmul: operands must have the same element type");
    Tensor rows = matmul_kernel(self_in.unsqueeze(0), other_in);
    std::vector<int64_t> out_sizes = shape_of(rows);
    out_sizes.erase(out_sizes.end() - 2);
    return rows.reshape(out_sizes);
  }

  // Batched on at least one side: the leading axes broadcast to one result
  // batch shape, fold into a single batch axis, and the product runs as one
  // batched dispatch.  A lone matrix broadcasts across the other operand's
  // batches.
  if (other_ndim >= 3) {
    const int64_t M = self_in.size(self_ndim - 2);
    const int64_t K = self_in.size(self_ndim - 1);
    const int64_t N = other_in.size(other_ndim - 1);
    TP_CHECK(
        K == other_in.size(other_ndim - 2),
        "Vulkan matmul: matrix dimensions do not match");

    const std::vector<int64_t> self_sizes = shape_of(self_in);
    const std::vector<int64_t> other_sizes = shape_of(other_in);
    const std::vector<int64_t> self_batch(
        self_sizes.begin(), self_sizes.end() - 2);
    const std::vector<int64_t> other_batch(
        other_sizes.begin(), other_sizes.end() - 2);
    const std::vector<int64_t> batch_shape =
        broadcast_shapes(self_batch, other_batch);

    int64_t batches = 1;
    for (const int64_t size : batch_shape) {
      batches *= size;
    }

    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    // An operand already shaped to the result batch axis needs only a view of
    // the same memory.  A lone matrix keeps its size-1 batch and broadcasts:
    // through the pack gather for an image, through a zero batch stride for a
    // linear payload.  Anything else materializes the broadcast.
    auto fold = [&](const Tensor& tensor,
                    const std::vector<int64_t>& tensor_batch,
                    int64_t rows,
                    int64_t cols) {
      if (tensor_batch == batch_shape) {
        return tensor.reshape({batches, rows, cols});
      }
      if (tensor.dim() == 3 && tensor.size(0) == 1) {
        return tensor;
      }
      std::vector<int64_t> expanded = batch_shape;
      expanded.push_back(rows);
      expanded.push_back(cols);
      return expand_kernel(tensor, expanded, false)
          .contiguous()
          .reshape({batches, rows, cols});
    };

    Tensor a3 = fold(self_in, self_batch, M, K);
    Tensor b3 = fold(other_in, other_batch, K, N);

    Tensor product =
        batches == 1 && folds_to_mm(a3, b3)
        ? mm_kernel(a3.reshape({M, K}), b3.reshape({K, N}))
        : bmm_impl(a3, b3, batches);

    return product.reshape(out_shape);
  }

  TP_THROW(
      NotImplementedError,
      "Vulkan matmul: unsupported operand combination");
}

//
// Linear: y = x · W^T + b computed in one seeded product.  The weight
// (N x K) is packed straight into the height-packed (K x N) operand planes
// with a transposed gather — no transpose copy, no intermediate tensor —
// and the packed planes persist in the pack cache across calls, so a
// steady-state inference loop re-streams the weights without re-packing.
//
Tensor linear_kernel(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  TP_CHECK(
      (input.dtype() == DType::Float32 || input.dtype() == DType::Float16) &&
          input.dtype() == weight.dtype(),
      "Vulkan linear supports Float32 and Float16 operands of one dtype only");
  TP_CHECK(
      weight.dim() == 2,
      "Vulkan linear: weight must be 2d (out_features, in_features)");
  TP_CHECK(input.dim() >= 1, "Vulkan linear: input must be at least 1d");

  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TP_CHECK(
      input.size(input.dim() - 1) == K,
      "Vulkan linear: input feature dimension does not match the weight");

  TP_CHECK(
      input.device().is_vulkan() && weight.device().is_vulkan(),
      "Vulkan linear: operands must live on the Vulkan device");

  // Fold the input to a dense (M x K) matrix; a leading view of a dense
  // tensor shares the storage, anything else materializes once here.
  Tensor in2 = input;
  if (input.dim() == 1) {
    in2 = input.reshape({1, K});
  } else if (input.dim() > 2) {
    in2 = input.reshape({input.numel() / K, K});
  }

  Tensor bias_vulkan;
  std::optional<Tensor> bias_owned;
  if (bias_opt.has_value() && bias_opt->defined()) {
    bias_vulkan = *bias_opt;
    if (bias_vulkan.dim() == 1) {
      bias_owned = bias_vulkan.unsqueeze(0);
      bias_vulkan = *bias_owned;
    }
    if (!bias_vulkan.device().is_vulkan()) {
      bias_owned = bias_vulkan.to(Device(DeviceType::Vulkan));
      bias_vulkan = *bias_owned;
    }
  }
  const std::optional<Tensor> bias_arg =
      (bias_opt.has_value() && bias_opt->defined())
      ? std::optional<Tensor>(bias_vulkan)
      : std::nullopt;

  const int64_t M = in2.size(0);
  api::vTensor v_lhs = gemv_operand_lhs(in2);
  api::vTensor v_weight = convert(weight);

  Tensor out2;
  if (v_weight.storage_type() == api::StorageType::BUFFER) {
    // A weight this size has no texture form, and relaying it into the
    // transposed shape a matrix product asks for would cost as much again as
    // the layer itself.  The kernel reads the layer's own [out, in] order
    // instead: a column step skips a whole input row, a reduction step walks
    // one.
    out2 = buffer_product(
        v_lhs,
        v_weight,
        M,
        N,
        K,
        K,
        1,
        bias_arg,
        Scalar(1.0),
        Scalar(1.0),
        input.dtype());
  } else {
    api::vTensor v_rhs = packed_cached(
        weight, shape_of(weight),
        api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
        kTagMmHeightPackedTransposed,
        [&] { return pack_rhs_height_transposed(weight); });

    out2 = product_dispatch(
        v_lhs, v_rhs, M, N, K, bias_arg, Scalar(1.0), Scalar(1.0),
        input.dtype());
  }

  if (input.dim() == 1) {
    return out2.reshape({N});
  }
  if (input.dim() == 2) {
    return out2;
  }
  const std::vector<int64_t> in_sizes = shape_of(input);
  std::vector<int64_t> out_sizes(in_sizes.begin(), in_sizes.end() - 1);
  out_sizes.push_back(N);
  return out2.reshape(out_sizes);
}

//
// Matrix · vector: y[i] = sum_k A[i][k] * x[k].  One row of the activation
// against a layer, so the product takes the decode route: the reduction axis
// splits across invocations and combines once, and a weight too wide for a
// texture streams straight out of its linear payload.  The pack cache keys on
// the weight's own storage identity, so a steady loop over a fixed matrix
// re-streams it without re-packing.
//
Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
  TP_CHECK(
      self.dim() == 2,
      "Vulkan mv: expected a matrix and a vector, got a ", self.dim(),
      "d matrix");
  TP_CHECK(
      vec.dim() == 1,
      "Vulkan mv: expected a matrix and a vector, got a ", vec.dim(),
      "d vector");
  TP_CHECK(
      self.dtype() == vec.dtype(),
      "Vulkan mv: operands must have the same element type");
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan mv supports Float32 and Float16 operands only");
  TP_CHECK(
      self.size(1) == vec.size(0),
      "Vulkan mv: size mismatch, mat (", self.size(0), "x", self.size(1),
      ") against vec (", vec.size(0), ")");
  TP_CHECK(
      self.device().is_vulkan() && vec.device().is_vulkan(),
      "Vulkan mv: operands must live on the Vulkan device");

  const int64_t M = self.size(0);
  const int64_t K = self.size(1);

  if (M == 0 || K == 0) {
    // A reduction over no entries answers zero on every row it reaches; a
    // matrix with no rows has nothing to answer for.
    api::Context* const context = api::context();
    api::vTensor v_output{context, {M}, self.dtype()};
    Tensor out = convert(v_output);
    if (M != 0) {
      out.zero_();
    }
    return out;
  }

  // A matrix that sits in memory as its own transpose is read through its
  // dense twin: the product's packing gather folds the swap, so no transpose
  // copy is needed.
  Tensor dense;
  if (transpose_view_of_dense(self, &dense)) {
    Tensor row = mm_impl(
        vec.reshape({1, K}),
        dense,
        std::nullopt,
        Scalar(1.0),
        Scalar(1.0));
    return row.reshape({M});
  }
  if (!self.is_contiguous()) {
    // Any other stride pattern has no gather that reads it as a weight, so it
    // materializes once here.
    return mv_kernel(self.contiguous(), vec);
  }

  // The vector is the activation row and the matrix the layer seen through
  // [out, in]: y = x · A^T, which is exactly A · x.
  return linear_kernel(vec, self, std::nullopt);
}

//
// Vector · vector: the shared axis reduced to a single number, which is what
// the (1 x 1) case of a wider product already answers.
//
Tensor dot_kernel(const Tensor& self, const Tensor& other) {
  TP_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "Vulkan dot: expected two vectors, got a ", self.dim(), "d and a ",
      other.dim(), "d operand");
  TP_CHECK(
      self.dtype() == other.dtype(),
      "Vulkan dot: expected both vectors to have the same element type");
  TP_CHECK(
      self.numel() == other.numel(),
      "Vulkan dot: expected both vectors to hold ", self.numel(),
      " entries, got ", other.numel());
  if (self.numel() == 0) {
    // Nothing to combine, so the answer is zero; the reduction walk the wider
    // product would take refuses a payload with no entries at all.
    return Tensor::zeros({}, self.dtype(), self.device());
  }
  return matmul_kernel(self, other);
}

//
// Outer: a column against a row, which is the rank-one payload every entry
// of one vector scaled by every entry of the other.
//
Tensor outer_kernel(const Tensor& self, const Tensor& vec2) {
  TP_CHECK(
      self.dim() == 1 && vec2.dim() == 1,
      "Vulkan outer: expected two vectors, got a ", self.dim(), "d and a ",
      vec2.dim(), "d operand");
  // Every entry is one product of two readings of the operands, so the pair
  // needs no shared element type: the broadcast product below settles the
  // type its entries are answered in.  A gradient of this product is a
  // matrix-vector walk, which does read both operands as one type.
  return self.unsqueeze(1).mul(vec2.unsqueeze(0));
}

//
// Inner product over the last axis of either rank: the axes before it are
// rows of one wide product against the partner seen through its transpose.
//
Tensor inner_kernel(const Tensor& self, const Tensor& other) {
  TP_CHECK(
      self.dtype() == other.dtype(),
      "Vulkan inner: expected both operands to have the same element type");
  if (self.dim() == 0 || other.dim() == 0) {
    // No axis is left to contract; the two numbers simply multiply.
    return self.mul(other);
  }
  const int64_t n = self.size(-1);
  TP_CHECK(
      other.size(-1) == n,
      "Vulkan inner: the last axis of both operands must match, got ", n,
      " and ", other.size(-1));
  if (self.dim() == 1 && other.dim() == 1) {
    return dot_kernel(self, other);
  }

  // Entries ahead of the contracted axis fold into the row axis of one
  // product; a lone vector folds to a single row.
  auto leading_entries = [](const Tensor& t) {
    int64_t rows = 1;
    const std::vector<int64_t> sizes = shape_of(t);
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      rows *= sizes[i];
    }
    return rows;
  };

  Tensor a = self.reshape({leading_entries(self), n});
  Tensor b = other.reshape({leading_entries(other), n});

  std::vector<int64_t> out_shape;
  const std::vector<int64_t> self_sizes = shape_of(self);
  const std::vector<int64_t> other_sizes = shape_of(other);
  for (size_t i = 0; i + 1 < self_sizes.size(); ++i) {
    out_shape.push_back(self_sizes[i]);
  }
  for (size_t i = 0; i + 1 < other_sizes.size(); ++i) {
    out_shape.push_back(other_sizes[i]);
  }
  return matmul_kernel(a, b.transpose(0, 1)).reshape(out_shape);
}

//
// A vector plus a matrix against a vector: the reduction the matrix-vector
// product answers, scaled and seeded with the addend.
//
Tensor addmv_kernel(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha) {
  TP_CHECK(
      mat.dim() == 2,
      "Vulkan addmv: the matrix must carry two axes, got ", mat.dim());
  TP_CHECK(
      vec.dim() == 1,
      "Vulkan addmv: the vector must carry one axis, got ", vec.dim());
  TP_CHECK(
      self.dim() <= 1,
      "Vulkan addmv: the addend must be a vector or a number, got ",
      self.dim(), " axes");
  const int64_t m = mat.size(0);
  TP_CHECK(
      mat.size(1) == vec.size(0),
      "Vulkan addmv: the reduction axis does not match, mat (", mat.size(0),
      "x", mat.size(1), ") against vec (", vec.size(0), ")");
  TP_CHECK(
      self.numel() == m || self.numel() == 1,
      "Vulkan addmv: the addend holds ", self.numel(),
      " entries and cannot reach the ", m, " rows of the matrix");
  TP_CHECK(
      self.dtype() == mat.dtype() && mat.dtype() == vec.dtype(),
      "Vulkan addmv: operands must have the same element type");

  Tensor product = mv_kernel(mat, vec);
  if (alpha.to<double>() != 1.0) {
    product = product.mul(alpha);
  }
  if (beta.to<double>() == 0.0) {
    // A zero weight on the addend carries no number of its across, not even
    // an indeterminate one, so the seed is dropped rather than multiplied.
    return product;
  }
  return self.mul(beta).add(product);
}

Tensor& addmv_inplace_kernel(
    Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha) {
  self.copy_(addmv_kernel(self, mat, vec, beta, alpha));
  return self;
}

} // namespace

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, MatmulKernels) {
  m.impl("mm", &tensorplay::vulkan::ops::mm_kernel);
  m.impl("addmm", &tensorplay::vulkan::ops::addmm_kernel);
  m.impl("matmul", &tensorplay::vulkan::ops::matmul_kernel);
  m.impl("bmm", &tensorplay::vulkan::ops::bmm_kernel);
  m.impl("baddbmm", &tensorplay::vulkan::ops::baddbmm_kernel);
  m.impl("linear", &tensorplay::vulkan::ops::linear_kernel);
  m.impl("mv", &tensorplay::vulkan::ops::mv_kernel);
  m.impl("dot", &tensorplay::vulkan::ops::dot_kernel);
  m.impl("outer", &tensorplay::vulkan::ops::outer_kernel);
  m.impl("inner", &tensorplay::vulkan::ops::inner_kernel);
  m.impl("addmv", &tensorplay::vulkan::ops::addmv_kernel);
  m.impl("addmv_", &tensorplay::vulkan::ops::addmv_inplace_kernel);
}

#endif /* USE_VULKAN */
