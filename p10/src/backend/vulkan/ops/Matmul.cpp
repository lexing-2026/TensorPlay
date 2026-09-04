#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "ParamCache.h"
#include "Tensor.h"
#include "../impl/Packing.h"

#include <algorithm>
#include <functional>

namespace tensorplay {
namespace vulkan {
namespace ops {

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
      VK_KERNEL(convert_channels_to_width_packed),
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
      VK_KERNEL(convert_channels_to_height_packed),
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
      VK_KERNEL(pack_lhs_transposed),
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
      VK_KERNEL(pack_rhs_transposed),
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
      VK_KERNEL(bmm_pack_lhs),
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
      VK_KERNEL(bmm_pack_rhs),
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
        VK_KERNEL(addmm),
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
        VK_KERNEL(mm),
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
      self.dtype() == DType::Float32 && mat2.dtype() == DType::Float32,
      "Vulkan mm supports Float32 tensors only");
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
    Scalar beta,
    Scalar alpha) {
  TP_CHECK(
      mat1.dtype() == DType::Float32 && mat2.dtype() == DType::Float32,
      "Vulkan addmm supports Float32 tensors only");
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

  api::vTensor v_result{
      context,
      {batches, M, N},
      batch1.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  if (M == 0 || N == 0 || K == 0) {
    Tensor out = convert(v_result);
    if (M != 0 && N != 0) {
      out.fill_(Scalar(0.0));
    }
    return out;
  }

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
      VK_KERNEL(bmm),
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
      batch1.dtype() == DType::Float32 && batch2.dtype() == DType::Float32,
      "Vulkan bmm supports Float32 tensors only");
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

  // A single batch is a plain matrix product; fold to the 2d kernel.
  if (B == 1) {
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
    Scalar beta,
    Scalar alpha) {
  TP_CHECK(
      batch1.dtype() == DType::Float32 && batch2.dtype() == DType::Float32,
      "Vulkan baddbmm supports Float32 tensors only");
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

Tensor matmul_kernel(const Tensor& self_in, const Tensor& other_in) {
  TP_CHECK(
      self_in.dtype() == DType::Float32 && other_in.dtype() == DType::Float32,
      "Vulkan matmul supports Float32 tensors only");

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

  // Batched on both sides: one tiled dispatch over the batch axis; a
  // size-1 batch broadcasts inside the pack gather.
  if (self_ndim >= 3 && other_ndim >= 3) {
    const int64_t self_b = self_in.size(0);
    const int64_t other_b = other_in.size(0);
    TP_CHECK(
        self_b == other_b || self_b == 1 || other_b == 1,
        "Vulkan matmul: batch dimensions are not broadcastable");
    const int64_t batches = std::max(self_b, other_b);

    const int64_t M = self_in.size(self_ndim - 2);
    const int64_t K = self_in.size(self_ndim - 1);
    const int64_t N = other_in.size(other_ndim - 1);
    TP_CHECK(
        K == other_in.size(other_ndim - 2),
        "Vulkan matmul: matrix dimensions do not match");

    // Higher-rank operands fold their leading axes into the batch axis.
    Tensor a3 = self_ndim == 3
        ? self_in
        : self_in.reshape({self_in.numel() / (M * K), M, K});
    Tensor b3 = other_ndim == 3
        ? other_in
        : other_in.reshape({other_in.numel() / (K * N), K, N});

    if (batches == 1) {
      // A single batch is a plain matrix product.
      Tensor flat1 = a3.reshape({M, K});
      Tensor flat2 = b3.reshape({K, N});
      return mm_kernel(flat1, flat2).reshape(
          {1, M, N});
    }

    return bmm_impl(a3, b3, batches);
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
      input.dtype() == DType::Float32 && weight.dtype() == DType::Float32,
      "Vulkan linear supports Float32 tensors only");
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

  api::vTensor v_lhs = packed_cached(
      in2, shape_of(in2),
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, kTagMmWidthPacked,
      [&] { return pack_lhs_width(in2); });

  api::vTensor v_rhs = packed_cached(
      weight, shape_of(weight),
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      kTagMmHeightPackedTransposed,
      [&] { return pack_rhs_height_transposed(weight); });

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

  const int64_t M = in2.size(0);
  Tensor out2 = product_dispatch(
      v_lhs,
      v_rhs,
      M,
      N,
      K,
      (bias_opt.has_value() && bias_opt->defined())
          ? std::optional<Tensor>(bias_vulkan)
          : std::nullopt,
      Scalar(1.0),
      Scalar(1.0),
      input.dtype());

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
}

#endif /* USE_VULKAN */
