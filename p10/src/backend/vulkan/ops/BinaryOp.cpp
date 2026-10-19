#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "SizesAndStrides.h"
#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../impl/Common.h"
#include "../../../../include/Utils.h"

#include <TypeProperties.h>
#include <TypePromotion.h>

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

// Dense single precision narrowed into a half-precision linear payload: a half
// word holds two lanes, so one invocation owns the destination word it writes.
api::vTensor narrow_linear_to_half(
    api::Context* context,
    const api::vTensor& src) {
  api::vTensor dst{
      context,
      src.sizes(),
      DType::Float16,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const uint32_t elements = static_cast<uint32_t>(src.numel());
  const struct NarrowBlock final {
    uint32_t buf_length;
    uint32_t fill0;
    uint32_t fill1;
    uint32_t fill2;
  } block{elements, 0u, 0u, 0u};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("cast_float_float16_buffer"),
      pipeline_barrier,
      {api::utils::div_up(elements, 2u), 1u, 1u},
      {256u, 1u, 1u},
      VK_NULL_HANDLE,
      dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return dst;
}

// A Bool plane is an image of signed bytes with one logical channel per texel
// lane; a buffer payload keeps one byte per element in row-major order.  The
// two disagree about which four elements share a 32-bit unit, so the repack
// assembles whole destination words, one per invocation.
api::vTensor bool_texture_to_buffer(
    api::Context* context,
    const api::vTensor& src) {
  api::vTensor dst{
      context,
      src.sizes(),
      DType::Bool,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  const uint32_t elements = static_cast<uint32_t>(src.numel());
  const struct BoolToBufferBlock final {
    ivec4 sizes; // (W, H, C, N)
    int32_t c_depth;
    int32_t numel;
    int32_t fill0;
    int32_t fill1;
  } block{
      make_whcn_ivec4(src.sizes()),
      static_cast<int32_t>(c_depth_of(src.sizes())),
      static_cast<int32_t>(elements),
      0,
      0};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR("bool_to_buffer"),
      pipeline_barrier,
      {api::utils::div_up(elements, 4u), 1u, 1u},
      {256u, 1u, 1u},
      VK_NULL_HANDLE,
      dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return dst;
}

// Returns a buffer-stored view of a tensor for the linear binary path.
// Buffer-stored sources are shared as-is; texture-stored sources are copied
// into a fresh dense buffer through the gather kernel (identity strides), which
// is how small broadcast operands (e.g. a bias loaded from host memory) can
// participate in a buffer binary op.  Int8 and UInt8 planes have no buffer-copy
// shader yet and stay loudly rejected.
api::vTensor to_buffer_storage(
    api::Context* context,
    const api::vTensor& src) {
  if (src.storage_type() == api::StorageType::BUFFER) {
    return src;
  }
  if (src.dtype() == DType::Bool) {
    return bool_texture_to_buffer(context, src);
  }
  TP_CHECK(
      src.dtype() == DType::Float32 || src.dtype() == DType::Int32 ||
          src.dtype() == DType::Float16,
      "Vulkan buffer-storage binary ops cannot convert this dtype from "
      "texture storage yet");
  // A texel's four lanes are not adjacent in the dense order, so a half
  // payload cannot be gathered straight into two-lanes-per-word packing: two
  // invocations would have to write the same destination word.  Copying
  // through single precision and narrowing the linear payload afterwards keeps
  // one writer per word.
  const DType staging_dtype =
      src.dtype() == DType::Float16 ? DType::Float32 : src.dtype();
  api::vTensor dst{
      context, src.sizes(), staging_dtype, api::StorageType::BUFFER,
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
  if (staging_dtype == src.dtype()) {
    return dst;
  }
  return narrow_linear_to_half(context, dst);
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

// The width a pair of operands is read in is the width that holds both of
// them.  Sampling a mixed pair instead would address a texel in one element
// width while its bits were written in another, which reinterprets the payload
// rather than converting it, and the linear path has no index remapping to
// hide the mistake either.  A pair that promotes to a width no kernel here
// stores is named in the refusal rather than guessed at.
DType pair_vocabulary(
    const Tensor& self,
    const Tensor& other,
    const char* name) {
  const DType vocabulary = tensorplay::native::result_type(self, other);
  TP_CHECK(
      vocabulary == DType::Float32 || vocabulary == DType::Float16 ||
          vocabulary == DType::Int32,
      std::string("Vulkan ") + name +
          " has no kernel for the operand widths it was given");
  return vocabulary;
}

Tensor to_vocabulary(const Tensor& t, DType vocabulary) {
  if (t.dtype() == vocabulary) {
    return t;
  }
  // Single precision is where the widths meet: every other code has an edge
  // into it and out of it, but a truth read as a word or a narrow float has
  // none of its own.  Such a payload is stepped through the shared width
  // rather than asked for a conversion no shader declares.
  if (t.dtype() == DType::Bool && vocabulary != DType::Float32) {
    return t.to(DType::Float32).to(vocabulary);
  }
  return t.to(vocabulary);
}

// The width a scalar op computes in is the width of its result, not the width
// of the payload: a count scaled by a fractional number ends up with fractional
// lanes, and so does a truth.  Only the step into single precision is taken
// here -- a result that needs a wider code than a lane holds stays on the
// payload's own dtype, where the op's dtype check reports it.  The scale factor
// plays no part: it multiplies the number that is already in the operand slot,
// so it never decides what the lanes have to hold.
Tensor scalar_op_operand(const Tensor& self, const Scalar& other) {
  if (self.dtype() == DType::Float32 || self.dtype() == DType::Float16) {
    return self;
  }
  if (result_type(other, self.dtype()) == DType::Float32) {
    return self.to(DType::Float32);
  }
  if (self.dtype() == DType::Bool) {
    // A truth counted with a whole number stops being a truth: the pair names
    // an integral code wider than any lane here, so the count is read in single
    // precision, which holds each of its values exactly and keeps a quotient of
    // one representable too.
    return self.to(DType::Float32);
  }
  return self;
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
  const DType vocabulary = pair_vocabulary(self_arg, other_arg, name);
  const Tensor self = to_vocabulary(self_arg, vocabulary);
  const Tensor other = to_vocabulary(other_arg, vocabulary);

  const BinaryVocab vocab = binary_vocab(self, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);
  api::vTensor v_other = convert(other);

  // Infer the broadcasted output shape. Texture storage supports arbitrary
  // leading-singleton broadcasts through the texel mapping; buffer storage
  // has no per-element index remapping, so it requires matching shapes.
  const std::vector<int64_t> out_sizes = tensorplay::broadcast_shapes(
      static_cast<std::vector<int64_t>>(self.shape()),
      static_cast<std::vector<int64_t>>(other.shape()));

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
    validate_buffer_elementwise_dtype(v_self, name);
    validate_buffer_elementwise_dtype(v_other, name);
    api::vTensor v_self_buf = to_buffer_storage(context, v_self);
    api::vTensor v_other_buf = to_buffer_storage(context, v_other);
    TP_CHECK(
        v_self_buf.dtype() == v_other_buf.dtype(),
        std::string("Vulkan ") + name +
            " over buffer storage needs operands of one payload dtype");
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::string linear_shader =
        buffer_shader_for_dtype(buffer_variant, v_self_buf.dtype());
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
        VK_KERNEL_FROM_STR(linear_shader.c_str()), pipeline_barrier,
        {buffer_dispatch_count(n, v_self_buf.dtype(), 4u), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
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
  if (vocab == BinaryVocab::kInt) {
    // As in the scalar writer: a count is scaled by a whole factor or not at
    // all, since the packing of the factor into the operand slot truncates.
    TP_CHECK(
        alpha.toDouble() == static_cast<double>(alpha.to<int64_t>()),
        std::string("Vulkan ") + name +
            " in place over a count refuses a fractional scale factor");
  }

  // The destination keeps the width it was allocated in, so only the incoming
  // side moves: it steps to the destination's width when the pair still fits
  // there.  A pair whose vocabulary is wider than the destination would have
  // to round the result down on the way in, so it is refused instead.
  const DType destination = self_arg.dtype();
  if (other_arg.dtype() != destination) {
    TP_CHECK(
        tensorplay::native::result_type(self_arg, other_arg) == destination,
        std::string("Vulkan ") + name +
            " in place cannot fold an operand wider than its destination");
  }
  const Tensor other = to_vocabulary(other_arg, destination);

  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);
  api::vTensor v_other = convert(other);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    validate_buffer_elementwise_dtype(v_self, name);
    validate_buffer_elementwise_dtype(v_other, name);
    api::vTensor v_other_buf = to_buffer_storage(context, v_other);
    TP_CHECK(
        v_self.dtype() == v_other_buf.dtype(),
        std::string("Vulkan ") + name +
            " over buffer storage needs operands of one payload dtype");
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const std::string linear_shader =
        buffer_shader_for_dtype(buffer_variant, v_self.dtype());
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
        VK_KERNEL_FROM_STR(linear_shader.c_str()), pipeline_barrier,
        {buffer_dispatch_count(n, v_self.dtype(), 4u), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        // The in-place shaders read the payload they also write, so the
        // barrier has to cover both directions.
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::READ |
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
  const Tensor operand = scalar_op_operand(self_arg, other);
  const BinaryVocab vocab = binary_vocab(operand, name);
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(operand);

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
    validate_buffer_elementwise_dtype(v_output, name);
    validate_buffer_elementwise_dtype(v_self, name);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::string linear_shader =
        buffer_shader_for_dtype(buffer_variant, v_output.dtype());
    api::UniformParamsBuffer paramsb = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(
                  other.to<int64_t>() * alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()), pipeline_barrier,
        {buffer_dispatch_count(n, v_output.dtype(), 1u), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
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
  if (vocab == BinaryVocab::kInt) {
    // The operand slot of a count holds a whole number, and the destination
    // keeps its own width, so a fraction arriving here would be rounded on the
    // way in rather than written.  Each factor is tested where it is packed,
    // because a product can look whole while one of them is not.
    TP_CHECK(
        other.toDouble() == static_cast<double>(other.to<int64_t>()) &&
            alpha.toDouble() == static_cast<double>(alpha.to<int64_t>()),
        std::string("Vulkan ") + name +
            " in place over a count refuses a fractional operand");
  }
  const std::string variant = binary_variant(shader_name, vocab, name);
  const std::string buffer_variant =
      binary_variant(buffer_shader_name, vocab, name);

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    validate_buffer_elementwise_dtype(v_self, name);
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const std::string linear_shader =
        buffer_shader_for_dtype(buffer_variant, v_self.dtype());
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context,
              BlockBi{n, 0u, static_cast<int32_t>(
                  other.to<int64_t>() * alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, other.to<float>() * alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(linear_shader.c_str()), pipeline_barrier,
        {buffer_dispatch_count(n, v_self.dtype(), 1u), 1u, 1u},
        {256u, 1u, 1u}, VK_NULL_HANDLE,
        // The in-place shaders read the payload they also write, so the
        // barrier has to cover both directions.
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::READ |
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

// Copies a texture-stored payload into a dense buffer vTensor, for sibling
// translation units that need every operand of a buffer path in linear
// storage (concatenation mixes large buffer inputs with small texture ones).
api::vTensor to_buffer_storage_dispatch(
    api::Context* context,
    const api::vTensor& src) {
  return to_buffer_storage(context, src);
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
  // Subtracting runs as adding a negation, and the negation stays in the family
  // of the number it came from: a whole number subtracted from a count leaves a
  // count, and that family is what decides the width the lanes need.
  const Scalar negated = other.isIntegral(true)
      ? Scalar(-other.to<int64_t>())
      : Scalar(-other.toDouble());
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self, negated,
                          alpha, "add");
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
  // Rounding a quotient down takes the divisor as an operand instead of as a
  // rounded reciprocal folded into a multiply: a quotient that is a whole
  // number has to land on it exactly, and the error of the fold sits right at
  // the boundary the floor is deciding.  Which width the lanes divide in
  // follows from the pair -- a count beside a whole number keeps word lanes,
  // and a fraction moves both operands to a width that has fractions.
  return binary_op_scalar(
      "floor_div_scalar", "buffer_floor_div_scalar", self, other,
      Scalar(1.0), "floor_divide");
}

Tensor& floor_divide_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("floor_divideinplace",
                                  "buffer_floor_divideinplace",
                                  self, other, Scalar(1.0), "floor_divide");
}

Tensor& floor_divide_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "floor_divide_.Scalar: can't divide by zero");
  // The writer keeps the destination's width, so a payload with no fractions
  // floors its own quotient; a divisor with fractions cannot be written back
  // into a count and is refused by the dispatch.
  return binary_op_scalar_inplace(
      "floor_div_scalarinplace", "buffer_floor_div_scalarinplace", self, other,
      Scalar(1.0), "floor_divide");
}

/*
 * Scalar-base power: the base broadcast comes in as the shader's scalar
 * argument while the tensor operand supplies the exponents.
 */
Tensor pow_scalar_base_kernel(const Scalar& base, const Tensor& self) {
  return binary_op_scalar(
      "pow_scalar_tensor", "buffer_pow_scalar_tensor", self, base,
      Scalar(1.0), "pow");
}

// A count has no lane for the fraction a negative whole power names: the
// quotient sits below one, and reading it back into a word answers with a
// number no division would give.  The float widths take it, so only a count is
// refused -- and only when the exponent itself is whole, since a fractional
// exponent already moves the pair into single precision.
void check_whole_power_destination(
    const Tensor& base,
    const Scalar& exponent,
    const char* name) {
  if (base.dtype() != DType::Int32 || !exponent.isIntegral(true)) {
    return;
  }
  TP_CHECK(
      exponent.to<int64_t>() >= 0,
      std::string("Vulkan ") + name +
          " refuses a negative exponent written back into a count");
}

Tensor& pow_tensor_scalar_inplace_kernel(Tensor& self, const Scalar& exponent) {
  check_whole_power_destination(self, exponent, "pow");
  return binary_op_scalar_inplace(
      "pow_tensor_scalarinplace", "buffer_pow_scalarinplace", self, exponent,
      Scalar(1.0), "pow");
}

Tensor& pow_tensor_inplace_kernel(Tensor& self, const Tensor& exponent) {
  return binary_op_tensor_inplace(
      "powinplace", "buffer_powinplace", self, exponent, Scalar(1.0), "pow");
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

Tensor& remainder_inplace_kernel(Tensor& self, const Tensor& other) {
  return binary_op_tensor_inplace("remainderinplace", "buffer_remainderinplace",
                                  self, other, Scalar(1.0), "remainder");
}

// The remainder carries the sign of the divisor, so the difference is taken
// against the floored quotient: the truncated one answers with the other
// family's sign whenever the operands disagreed, which is the other op.
Tensor remainder_scalar_kernel(const Tensor& self, const Scalar& other) {
  return binary_op_scalar("mod_scalar", "buffer_mod_scalar", self, other,
                          Scalar(1.0), "remainder");
}

Tensor& remainder_scalar_inplace_kernel(Tensor& self, const Scalar& other) {
  return binary_op_scalar_inplace("mod_scalarinplace",
                                  "buffer_mod_scalarinplace", self, other,
                                  Scalar(1.0), "remainder");
}

// A number on the left of the operator keeps the dividend's role: the payload
// samples the divisor's slot, so the subtraction runs against the tensor.
Tensor remainder_scalar_tensor_kernel(const Scalar& self, const Tensor& other) {
  return binary_op_scalar("mod_scalar_tensor", "buffer_mod_scalar_tensor",
                          other, self, Scalar(1.0), "remainder");
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
  m.impl("remainder_.Tensor",
         &tensorplay::vulkan::ops::remainder_inplace_kernel);
  m.impl("remainder.Scalar",
         &tensorplay::vulkan::ops::remainder_scalar_kernel);
  m.impl("remainder_.Scalar",
         &tensorplay::vulkan::ops::remainder_scalar_inplace_kernel);
  m.impl("remainder.Scalar_Tensor",
         &tensorplay::vulkan::ops::remainder_scalar_tensor_kernel);
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
