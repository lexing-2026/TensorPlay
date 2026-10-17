#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "SizesAndStrides.h"
#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../impl/Common.h"
#include "../../../../include/Utils.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

struct BlockB final {
  uint32_t buf_length;
  uint32_t fill0;
  float alpha;
  int32_t broadcast;
  ivec4 out_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
};

struct BlockBi final {
  uint32_t buf_length;
  uint32_t fill0;
  int32_t alpha;
  int32_t broadcast;
  ivec4 out_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
};

// Contiguous row-major strides of a shape, outer dims first, zero-padded
// to four entries.  The linear broadcast remap in the buffer shaders walks
// the output index through these strides.
ivec4 buffer_out_strides(const std::vector<int64_t>& out_sizes) {
  ivec4 strides{0, 0, 0, 0};
  int64_t acc = 1;
  for (int d = static_cast<int>(out_sizes.size()) - 1; d >= 0; --d) {
    strides[d] = static_cast<int32_t>(acc);
    acc *= out_sizes[d];
  }
  return strides;
}

// Broadcast metadata for one operand of a linear binary op.  The operand
// shape is right-aligned to the output rank (broadcasting aligns trailing
// dimensions); singleton dims get stride 0 and are skipped by the remap.
// Returns {sizes4, strides4}.
std::pair<ivec4, ivec4> buffer_broadcast_meta(
    const std::vector<int64_t>& out_sizes,
    const std::vector<int64_t>& op_sizes) {
  const int ndim_out = static_cast<int>(out_sizes.size());
  const int ndim_op = static_cast<int>(op_sizes.size());
  TP_CHECK(
      ndim_out <= 4,
      "Vulkan buffer-storage binary ops support up to 4 dims");
  TP_CHECK(
      ndim_op <= ndim_out,
      "Vulkan buffer-storage binary ops: operand rank exceeds output rank");
  const int leading = ndim_out - ndim_op;
  ivec4 sizes{0, 0, 0, 0};
  ivec4 strides{0, 0, 0, 0};
  int64_t acc = 1;
  for (int d = ndim_op - 1; d >= 0; --d) {
    const int slot = leading + d;
    sizes[slot] = static_cast<int32_t>(op_sizes[d]);
    strides[slot] = static_cast<int32_t>(acc);
    acc *= op_sizes[d];
  }
  return {sizes, strides};
}

// Prepads sizes to four slots in {N, C, H, W} order (leading 1s), matching
// the texture gather convention the buffer-copy kernel reuses.
std::vector<int64_t> prepadded(const std::vector<int64_t>& v) {
  std::vector<int64_t> out(4, 1);
  const int64_t offset = 4 - static_cast<int64_t>(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[static_cast<size_t>(offset + static_cast<int64_t>(i))] = v[i];
  }
  return out;
}

// Returns a buffer-stored view of a tensor for the linear binary path.
// Buffer-stored sources are shared as-is; texture-stored sources are copied
// into a fresh dense buffer through the gather kernel (identity strides), which
// is how small broadcast operands (e.g. a bias loaded from host memory) can
// participate in a buffer binary op.  Byte-packed dtypes have no buffer-copy
// shader yet and stay loudly rejected.
api::vTensor to_buffer_storage(
    api::Context* context,
    const api::vTensor& src) {
  if (src.storage_type() == api::StorageType::BUFFER) {
    return src;
  }
  TP_CHECK(
      src.dtype() == DType::Float32 || src.dtype() == DType::Int32,
      "Vulkan buffer-storage binary ops cannot convert this dtype from "
      "texture storage yet");
  api::vTensor dst{
      context, src.sizes(), src.dtype(), api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const std::vector<int64_t> in_sizes = prepadded(
      std::vector<int64_t>(src.sizes().begin(), src.sizes().end()));
  const auto in_dense = SizesAndStrides::compute_contiguous_strides(in_sizes);
  const std::vector<int64_t> out_padded = prepadded(
      std::vector<int64_t>(src.sizes().begin(), src.sizes().end()));
  const struct CopyBlock final {
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
          static_cast<int32_t>(in_dense[0]),
          static_cast<int32_t>(in_dense[1]),
          static_cast<int32_t>(in_dense[2]),
          static_cast<int32_t>(in_dense[3])),
      c_depth_of(in_sizes),
      c_depth_of(out_padded),
      0,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 grid{
      static_cast<uint32_t>(out_padded[3]),
      static_cast<uint32_t>(out_padded[2]),
      static_cast<uint32_t>(out_padded[0]) *
          static_cast<uint32_t>(c_depth_of(out_padded)),
  };
  const api::ShaderInfo shader = src.dtype() == DType::Int32
      ? VK_KERNEL(view_gather_int32_buffer)
      : VK_KERNEL(view_gather_buffer);
  context->submit_compute_job(
      shader, pipeline_barrier, grid,
      adaptive_work_group_size(grid), VK_NULL_HANDLE,
      dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return dst;
}

// Element-width classes used by binary dispatch.
enum class BinaryVocab { kFloat, kInt };

BinaryVocab binary_vocab(const Tensor& t, const char* name) {
  if (t.dtype() == DType::Float32 || t.dtype() == DType::Float16) {
    return BinaryVocab::kFloat;
  }
  if (t.dtype() == DType::Int32) {
    return BinaryVocab::kInt;
  }
  TP_THROW(
      NotImplementedError,
      std::string("Vulkan ") + name +
          " supports Float32, Float16 and Int32 tensors only");
  return BinaryVocab::kFloat;
}

// Resolves the variant name against the registry: an Int32 payload needs the
// `_i32` twin to exist, otherwise the op has no integer kernel and the call
// fails with an explicit message.  The marker is inserted before the
// generated "inplace" suffix, matching the variant naming of the generator
// (`add_i32inplace`, not `addinplace_i32`).
std::string binary_variant(
    const char* base,
    BinaryVocab vocab,
    const char* name) {
  std::string b(base);
  if (vocab == BinaryVocab::kInt) {
    static const std::string kInplace = "inplace";
    const size_t pos = b.rfind(kInplace);
    if (pos != std::string::npos && pos + kInplace.size() == b.size()) {
      b.insert(pos, "_i32");
    } else {
      b += "_i32";
    }
    TP_CHECK(
        api::shader_registry().has_shader(b),
        std::string("Vulkan ") + name +
            " has no Int32 kernel; the op is float-only");
  }
  return b;
}

/*
 * Shared implementation of the element-wise binary tensor op:
 * out = OP(self, other, alpha), with broadcasting over the other operand.
 */
Tensor binary_op_tensor(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha,
    const char* name) {
  const BinaryVocab vocab = binary_vocab(self_arg, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);
  api::vTensor v_other = convert(other_arg);

  // Infer the broadcasted output shape. Texture storage supports arbitrary
  // leading-singleton broadcasts through the texel mapping; buffer storage
  // has no per-element index remapping, so it requires matching shapes.
  const std::vector<int64_t> out_sizes = tensorplay::broadcast_shapes(
      static_cast<std::vector<int64_t>>(self_arg.shape()),
      static_cast<std::vector<int64_t>>(other_arg.shape()));

  // A buffer-stored operand keeps the whole op on the linear path even when
  // the output shape is small enough for image storage; the image path has
  // no buffer input binding.
  const bool is_buffer =
      v_self.storage_type() == api::StorageType::BUFFER ||
      v_other.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      out_sizes,
      v_self.dtype(),
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  if (is_buffer) {
    // The linear path reads both operands through buffer bindings; a
    // texture-stored operand (small broadcast bias, keepdim mean, etc.) is
    // copied into a dense buffer first.
    api::vTensor v_self_buf = to_buffer_storage(context, v_self);
    api::vTensor v_other_buf = to_buffer_storage(context, v_other);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::vector<int64_t> self_sizes = v_self.sizes();
    const std::vector<int64_t> other_sizes = v_other.sizes();
    const int32_t broadcast =
        (self_sizes != out_sizes ? 1 : 0) |
        (other_sizes != out_sizes ? 2 : 0);
    const ivec4 out_strides = buffer_out_strides(out_sizes);
    const auto input_meta = buffer_broadcast_meta(out_sizes, self_sizes);
    const auto other_meta = buffer_broadcast_meta(out_sizes, other_sizes);
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(alpha.to<int64_t>()),
                      broadcast, out_strides, input_meta.first,
                      input_meta.second, other_meta.first, other_meta.second})
        : api::UniformParamsBuffer(
              context,
              BlockB{n, 0u, alpha.to<float>(), broadcast, out_strides,
                     input_meta.first, input_meta.second, other_meta.first,
                     other_meta.second});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 output_sizes;
    ivec4 input_sizes;
    ivec4 other_sizes;
    int32_t alpha;
  } blocki{
      make_whcn_ivec4(v_output.sizes()),
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      static_cast<int32_t>(alpha.to<int64_t>()),
  };

  api::UniformParamsBuffer params =
      (vocab == BinaryVocab::kInt)
      ? api::UniformParamsBuffer(context, blocki)
      : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      api::get_shader_info_for_dtype(variant.c_str(), v_output.dtype()),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& binary_op_tensor_inplace(
    const char* shader_name,
    const char* buffer_shader_name,
    Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha,
    const char* name) {
  const BinaryVocab vocab = binary_vocab(self_arg, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);
  api::vTensor v_other = convert(other_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    api::vTensor v_other_buf = to_buffer_storage(context, v_other);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const std::vector<int64_t> self_sizes = v_self.sizes();
    const std::vector<int64_t> other_sizes = v_other.sizes();
    const int32_t broadcast = (other_sizes != self_sizes) ? 2 : 0;
    const ivec4 out_strides = buffer_out_strides(self_sizes);
    const ivec4 zero{0, 0, 0, 0};
    const auto other_meta = buffer_broadcast_meta(self_sizes, other_sizes);
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(alpha.to<int64_t>()),
                      broadcast, out_strides, zero, zero, other_meta.first,
                      other_meta.second})
        : api::UniformParamsBuffer(
              context,
              BlockB{n, 0u, alpha.to<float>(), broadcast, out_strides,
                     zero, zero, other_meta.first, other_meta.second});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {(n + 3u) / 4u, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        v_other_buf.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return self_arg;
  }

  const struct Block final {
    ivec4 output_sizes;
    ivec4 other_sizes;
    float alpha;
  } block{
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 output_sizes;
    ivec4 other_sizes;
    int32_t alpha;
  } blocki{
      make_whcn_ivec4(v_self.sizes()),
      make_whcn_ivec4(v_other.sizes()),
      static_cast<int32_t>(alpha.to<int64_t>()),
  };

  api::UniformParamsBuffer params =
      (vocab == BinaryVocab::kInt)
      ? api::UniformParamsBuffer(context, blocki)
      : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      api::get_shader_info_for_dtype(variant.c_str(), v_self.dtype()),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

/*
 * Element-wise binary op against a broadcast scalar.
 */
Tensor binary_op_scalar(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha,
    const char* name) {
  const BinaryVocab vocab = binary_vocab(self_arg, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  // A buffer-stored input keeps the whole op on the linear path even when
  // the output shape is small enough for image storage; the image path has
  // no buffer input binding.
  const bool is_buffer =
      v_self.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  // The shader bounds-checks against extents.xyz, so the block must carry
  // the actual three-dimensional image extents (width, height,
  // batch * ceil(channels / 4)); the logical WHCN shape is not a valid
  // depth for images whose batch exceeds the channel planes.
  const uvec3 out_extents = v_output.extents();
  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      ivec4(
          static_cast<int32_t>(out_extents[0u]),
          static_cast<int32_t>(out_extents[1u]),
          static_cast<int32_t>(out_extents[2u]),
          0),
      other.to<float>() * alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 extents;
    // scalar argument
    int32_t other;
  } blocki{
      ivec4(
          static_cast<int32_t>(out_extents[0u]),
          static_cast<int32_t>(out_extents[1u]),
          static_cast<int32_t>(out_extents[2u]),
          0),
      static_cast<int32_t>(other.to<int64_t>() * alpha.to<int64_t>()),
  };

  if (is_buffer) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    api::UniformParamsBuffer paramsb = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(
                  other.to<int64_t>() * alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {n, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        paramsb.buffer());
    return convert(v_output);
  }

  api::UniformParamsBuffer params =
      (vocab == BinaryVocab::kInt)
      ? api::UniformParamsBuffer(context, blocki)
      : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      api::get_shader_info_for_dtype(variant.c_str(), v_output.dtype()),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& binary_op_scalar_inplace(
    const char* shader_name,
    const char* buffer_shader_name,
    Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha,
    const char* name) {
  const BinaryVocab vocab = binary_vocab(self_arg, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(
                  other.to<int64_t>() * alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {n, 1u, 1u}, {256u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        params.buffer());
    return self_arg;
  }

  const uvec3 self_extents = v_self.extents();
  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      ivec4(
          static_cast<int32_t>(self_extents[0u]),
          static_cast<int32_t>(self_extents[1u]),
          static_cast<int32_t>(self_extents[2u]),
          0),
      other.to<float>() * alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 extents;
    // scalar argument
    int32_t other;
  } blocki{
      ivec4(
          static_cast<int32_t>(self_extents[0u]),
          static_cast<int32_t>(self_extents[1u]),
          static_cast<int32_t>(self_extents[2u]),
          0),
      static_cast<int32_t>(other.to<int64_t>() * alpha.to<int64_t>()),
  };

  api::UniformParamsBuffer params =
      (vocab == BinaryVocab::kInt)
      ? api::UniformParamsBuffer(context, blocki)
      : api::UniformParamsBuffer(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      api::get_shader_info_for_dtype(variant.c_str(), v_self.dtype()),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

} // namespace

// External entry points to the anonymous-namespace binary op engines, for
// sibling translation units that want the shared broadcast/buffer routing
// (the pow kernels delegate here).
Tensor binary_op_tensor_dispatch(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha,
    const char* name) {
  return binary_op_tensor(
      shader_name, buffer_shader_name, self_arg, other_arg, alpha, name);
}

Tensor binary_op_scalar_dispatch(
    const char* shader_name,
    const char* buffer_shader_name,
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha,
    const char* name) {
  return binary_op_scalar(
      shader_name, buffer_shader_name, self_arg, other, alpha, name);
}

Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return binary_op_tensor("add", "buffer_add", self, other, alpha, "add");
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return binary_op_tensor("sub", "buffer_sub", self, other, alpha, "sub");
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("mul", "buffer_mul", self, other, Scalar(1.0), "mul");
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
  if (self.dtype() == DType::Int32) {
    return binary_op_tensor(
        "div", "buffer_div", self.to(DType::Float32),
        other.to(DType::Float32), Scalar(1.0), "div");
  }
  return binary_op_tensor("div", "buffer_div", self, other, Scalar(1.0), "div");
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self, other,
                          alpha, "add");
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self,
                          Scalar(-other.toDouble()), alpha, "add");
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self, other,
                          Scalar(1.0), "mul");
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
  if (self.dtype() == DType::Int32) {
    return binary_op_scalar(
        "mul_scalar", "buffer_mul_scalar", self.to(DType::Float32),
        Scalar(1.0 / other.toDouble()), Scalar(1.0), "mul");
  }
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self,
                          Scalar(1.0 / other.toDouble()), Scalar(1.0), "mul");
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return binary_op_tensor_inplace("addinplace", "buffer_addinplace", self,
                                  other, alpha, "add");
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return binary_op_tensor_inplace("subinplace", "buffer_subinplace", self,
                                  other, alpha, "sub");
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("mulinplace", "buffer_mulinplace", self,
                                  other, Scalar(1.0), "mul");
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("divinplace", "buffer_divinplace", self,
                                  other, Scalar(1.0), "div");
}

Tensor& add_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return binary_op_scalar_inplace("add_scalarinplace",
                                  "buffer_add_scalarinplace",
                                  self, other, alpha, "add");
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  return binary_op_scalar_inplace("mul_scalarinplace",
                                  "buffer_mul_scalarinplace",
                                  self, other, Scalar(1.0), "mul");
}

// The subtraction in-place scalar flavor uses the add writer with the
// negated product: x -= y * alpha runs as x += -(y * alpha).
Tensor& sub_scalar_inplace_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return binary_op_scalar_inplace("add_scalarinplace",
                                  "buffer_add_scalarinplace",
                                  self, other, Scalar(-alpha.toDouble()), "add");
}

Tensor& div_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "div_.Scalar: can't divide by zero");
  return binary_op_scalar_inplace("mul_scalarinplace",
                                  "buffer_mul_scalarinplace",
                                  self, Scalar(1.0 / other.toDouble()),
                                  Scalar(1.0), "mul");
}

// Rounded quotient: the tensor form runs the IS_DIV variant, and the scalar
// form multiplies by the reciprocal through the floor-multiply shader.
Tensor floor_divide_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("floor_divide", "buffer_floor_divide", self, other,
                          Scalar(1.0), "floor_divide");
}

Tensor floor_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "floor_divide.Scalar: can't divide by zero");
  return binary_op_scalar("floor_mul_scalar", "buffer_floor_mul_scalar", self,
                          Scalar(1.0 / other.toDouble()), Scalar(1.0),
                          "floor_divide");
}

Tensor& floor_divide_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("floor_divideinplace",
                                  "floor_divideinplace",
                                  self, other, Scalar(1.0), "floor_divide");
}

Tensor& floor_divide_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "floor_divide_.Scalar: can't divide by zero");
  return binary_op_scalar_inplace("floor_mul_scalarinplace",
                                  "buffer_floor_mul_scalarinplace", self,
                                  Scalar(1.0 / other.toDouble()), Scalar(1.0),
                                  "floor_divide");
}

/*
 * Scalar-base power: the base broadcast comes in as the shader's scalar
 * argument while the tensor operand supplies the exponents.
 */
Tensor pow_scalar_base_kernel(const Scalar& base, const Tensor& self) {
  return binary_op_scalar(
      "pow_scalar_tensor", "pow_scalar_tensor", self, base,
      Scalar(1.0), "pow");
}

Tensor& pow_tensor_scalar_inplace_kernel(Tensor& self, const Scalar& exponent) {
  return binary_op_scalar_inplace(
      "pow_tensor_scalarinplace", "pow_tensor_scalarinplace", self, exponent,
      Scalar(1.0), "pow");
}

Tensor& pow_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
  return binary_op_tensor_inplace(
      "powinplace", "powinplace", self, exponent, Scalar(1.0), "pow");
}

//
// New binary entries.  The max/min/remainder/fmod/atan2/logaddexp families
// ride the binary_op_tensor shader with their dedicated variants; the
// `_i32` twins cover Int32 where the formula has an integer meaning.
//
Tensor maximum_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("maximum", "buffer_maximum", self, other,
                          Scalar(1.0), "maximum");
}

Tensor minimum_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("minimum", "buffer_minimum", self, other,
                          Scalar(1.0), "minimum");
}

Tensor remainder_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("remainder", "buffer_remainder", self, other,
                          Scalar(1.0), "remainder");
}

Tensor fmod_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("fmod", "buffer_fmod", self, other,
                          Scalar(1.0), "fmod");
}

Tensor atan2_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("atan2", "buffer_atan2", self, other,
                          Scalar(1.0), "atan2");
}

Tensor& atan2__kernel(Tensor& self, const Tensor& other) {
  self.copy_(atan2_kernel(self, other));
  return self;
}

Tensor logaddexp_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("logaddexp", "buffer_logaddexp", self, other,
                          Scalar(1.0), "logaddexp");
}

Tensor rsub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return binary_op_scalar("rsub_scalar", "buffer_rsub_scalar", self, other,
                          alpha, "rsub");
}

Tensor true_divide_kernel(const Tensor& self, const Tensor& other) {
  return div_kernel(self, other);
}

Tensor true_divide_scalar_kernel(const Tensor& self, const Scalar& other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "true_divide.Scalar: can't divide by zero");
  return div_scalar_kernel(self, other);
}

Tensor divide_tensor_kernel(const Tensor& self, const Tensor& other) {
  return div_kernel(self, other);
}

Tensor divide_scalar_kernel(const Tensor& self, const Scalar& other) {
  return true_divide_scalar_kernel(self, other);
}

Tensor subtract_tensor_kernel(const Tensor& self, const Tensor& other,
                              const Scalar& alpha) {
  return binary_op_tensor("sub", "buffer_sub", self, other, alpha, "subtract");
}

Tensor multiply_tensor_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("mul", "buffer_mul", self, other, Scalar(1.0),
                          "multiply");
}

Tensor& sub_scalar_kernel_2(Tensor& self, Scalar other, Scalar alpha) {
  return sub_scalar_inplace_kernel(self, other, alpha);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, BinaryOpKernels) {
  m.impl("add.Tensor", &tensorplay::vulkan::ops::add_kernel);
  m.impl("add.Scalar", &tensorplay::vulkan::ops::add_scalar_kernel);
  m.impl("add_.Tensor", &tensorplay::vulkan::ops::add_inplace_kernel);
  m.impl("add_.Scalar", &tensorplay::vulkan::ops::add_scalar_inplace_kernel);
  m.impl("sub.Tensor", &tensorplay::vulkan::ops::sub_kernel);
  m.impl("sub.Scalar", &tensorplay::vulkan::ops::sub_scalar_kernel);
  m.impl("sub_.Tensor", &tensorplay::vulkan::ops::sub_inplace_kernel);
  m.impl("sub_.Scalar", &tensorplay::vulkan::ops::sub_scalar_inplace_kernel);
  m.impl("mul.Tensor", &tensorplay::vulkan::ops::mul_kernel);
  m.impl("mul.Scalar", &tensorplay::vulkan::ops::mul_scalar_kernel);
  m.impl("mul_.Tensor", &tensorplay::vulkan::ops::mul_inplace_kernel);
  m.impl("mul_.Scalar", &tensorplay::vulkan::ops::mul_scalar_inplace_kernel);
  m.impl("div.Tensor", &tensorplay::vulkan::ops::div_kernel);
  m.impl("div.Scalar", &tensorplay::vulkan::ops::div_scalar_kernel);
  m.impl("div_.Tensor", &tensorplay::vulkan::ops::div_inplace_kernel);
  m.impl("div_.Scalar", &tensorplay::vulkan::ops::div_scalar_inplace_kernel);
  m.impl("floor_divide", &tensorplay::vulkan::ops::floor_divide_kernel);
  m.impl("floor_divide.Scalar",
         &tensorplay::vulkan::ops::floor_divide_scalar_kernel);
  m.impl("floor_divide_.Tensor",
         &tensorplay::vulkan::ops::floor_divide_inplace_kernel);
  m.impl("floor_divide_.Scalar",
         &tensorplay::vulkan::ops::floor_divide_scalar_inplace_kernel);
  m.impl("pow.Scalar", &tensorplay::vulkan::ops::pow_scalar_base_kernel);
  m.impl("pow_.Scalar", &tensorplay::vulkan::ops::pow_tensor_scalar_inplace_kernel);
  m.impl("pow_.Tensor", &tensorplay::vulkan::ops::pow_tensor_inplace_kernel);
  m.impl("maximum", &tensorplay::vulkan::ops::maximum_kernel);
  m.impl("minimum", &tensorplay::vulkan::ops::minimum_kernel);
  m.impl("remainder.Tensor", &tensorplay::vulkan::ops::remainder_kernel);
  m.impl("fmod.Tensor", &tensorplay::vulkan::ops::fmod_kernel);
  m.impl("atan2", &tensorplay::vulkan::ops::atan2_kernel);
  m.impl("atan2_", &tensorplay::vulkan::ops::atan2__kernel);
  m.impl("logaddexp", &tensorplay::vulkan::ops::logaddexp_kernel);
  m.impl("rsub.Scalar", &tensorplay::vulkan::ops::rsub_scalar_kernel);
  m.impl("rsub.Tensor", &tensorplay::vulkan::ops::subtract_tensor_kernel);
  m.impl("true_divide.Tensor", &tensorplay::vulkan::ops::true_divide_kernel);
  m.impl("true_divide.Scalar", &tensorplay::vulkan::ops::true_divide_scalar_kernel);
  m.impl("divide.Tensor", &tensorplay::vulkan::ops::divide_tensor_kernel);
  m.impl("divide.Scalar", &tensorplay::vulkan::ops::divide_scalar_kernel);
  m.impl("divide_.Scalar", &tensorplay::vulkan::ops::div_scalar_inplace_kernel);
  m.impl("divide_.Tensor", &tensorplay::vulkan::ops::div_inplace_kernel);
  m.impl("subtract.Tensor", &tensorplay::vulkan::ops::subtract_tensor_kernel);
  m.impl("subtract_.Tensor", &tensorplay::vulkan::ops::sub_inplace_kernel);
  m.impl("subtract.Scalar", &tensorplay::vulkan::ops::sub_scalar_kernel);
  m.impl("subtract_.Scalar", &tensorplay::vulkan::ops::sub_scalar_inplace_kernel);
  m.impl("multiply.Tensor", &tensorplay::vulkan::ops::multiply_tensor_kernel);
  m.impl("multiply_.Tensor", &tensorplay::vulkan::ops::mul_inplace_kernel);
  m.impl("multiply.Scalar", &tensorplay::vulkan::ops::mul_scalar_kernel);
  m.impl("multiply_.Scalar", &tensorplay::vulkan::ops::mul_scalar_inplace_kernel);
}

#endif /* USE_VULKAN */
