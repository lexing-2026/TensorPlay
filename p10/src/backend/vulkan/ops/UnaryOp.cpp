#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../impl/Common.h"
#include "../api/Context.h"
#include "../api/Shader.h"
#include "../api/ShaderRegistry.h"

#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

/*
 * Element-width classes the unary dispatch covers.  kFloat covers Float32 /
 * Float16 (sampler planes), kInt covers Int32 (signed-word iimage planes), and
 * kWide covers a code wider than a word, which has no texel to sit in and so is
 * only ever walked over linear storage.
 */
enum class UnaryVocab { kFloat, kInt, kWide };

UnaryVocab unary_vocab(const Tensor& t, const char* name) {
  if (t.dtype() == DType::Float32 || t.dtype() == DType::Float16) {
    return UnaryVocab::kFloat;
  }
  if (t.dtype() == DType::Int32) {
    return UnaryVocab::kInt;
  }
  if (t.dtype() == DType::Int64) {
    return UnaryVocab::kWide;
  }
  TP_THROW(
      NotImplementedError,
      std::string("Vulkan ") + name +
          " supports Float32, Float16, Int32 and Int64 tensors only");
  return UnaryVocab::kFloat;
}

// Variant name for the payload vocabulary: the marker of the width is inserted
// before the generated "inplace" suffix (variants compile as
// `<base>_i32inplace`), and the registry check turns an op with no kernel in
// that width into an explicit error instead of a missing-kernel crash.  A code
// wider than a word has no image form, so the image side of such a payload keeps
// its own name and is refused at the image path.
std::string unary_variant(
    const char* base,
    UnaryVocab vocab,
    const char* name,
    bool linear) {
  std::string b(base);
  std::string marker;
  if (vocab == UnaryVocab::kInt) {
    marker = "_i32";
  } else if (vocab == UnaryVocab::kWide) {
    if (!linear) {
      return b;
    }
    marker = "_wide";
  } else {
    return b;
  }
  static const std::string kInplace = "inplace";
  const size_t pos = b.rfind(kInplace);
  if (pos != std::string::npos && pos + kInplace.size() == b.size() &&
      (pos < marker.size() ||
       b.compare(pos - marker.size(), marker.size(), marker) != 0)) {
    b.insert(pos, marker);
  } else if (b.size() < marker.size() ||
             b.compare(b.size() - marker.size(), marker.size(), marker) != 0) {
    b += marker;
  }
  TP_CHECK(
      api::shader_registry().has_shader(b),
      std::string("Vulkan ") + name + " has no kernel" +
          (vocab == UnaryVocab::kWide
               ? " for a count wider than a word"
               : " for a signed word; the op is float-only"));
  return b;
}

// An image names one payload per texel, so a code wider than a word has no image
// form to be read through.  Such a payload never gets image storage either; the
// refusal is named rather than left to a kernel that would read the word above
// an element as a second lane.
void require_image_vocab(const Tensor& self_arg, const char* name) {
  TP_CHECK(
      self_arg.dtype() != DType::Int64,
      std::string("Vulkan ") + name +
          " has no image form for a count wider than a word");
}

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const char* buffer_shader_name) {
  api::Context* const context = api::context();

  // An out-of-place walk reads a payload, so an operand carried through
  // swapped strides is gathered before the lane names are computed.  The
  // binding borrows that payload, so the gathered operand stays named here.
  const Tensor dense_self = dense_payload_of_view(self_arg);
  api::vTensor v_self = convert(dense_self);

  // A buffer-stored input keeps the op on the linear path even when the
  // output shape is small enough for image storage; the image path has no
  // buffer input binding.
  const bool is_buffer =
      v_self.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      is_buffer ? api::StorageType::BUFFER : api::StorageType::TEXTURE_3D,
      is_buffer ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};

  if (is_buffer) {
    validate_buffer_elementwise_dtype(v_self, buffer_shader_name);
    const uint32_t elements =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    const std::string shader =
        buffer_shader_for_dtype(buffer_shader_name, v_self.dtype());
    const struct BlockB final {
      uint32_t buf_length;
    } blockb{
        elements,
    };
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader.c_str()),
        pipeline_barrier,
        {buffer_dispatch_count(elements, v_self.dtype(), 4u), 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
    return convert(v_output);
  }
  require_image_vocab(self_arg, buffer_shader_name);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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

Tensor& unary_op_(
    Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const char* buffer_shader_name) {
  api::Context* const context = api::context();

  api::vTensor v_self = convert(self_arg);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    validate_buffer_elementwise_dtype(v_self, buffer_shader_name);
    const uint32_t elements =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    const std::string shader =
        buffer_shader_for_dtype(buffer_shader_name, v_self.dtype());
    const struct BlockB final {
      uint32_t buf_length;
    } blockb{
        elements,
    };
    api::UniformParamsBuffer params(context, blockb);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(shader.c_str()),
        pipeline_barrier,
        {buffer_dispatch_count(elements, v_self.dtype(), 4u), 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        params.buffer());
    return self_arg;
  }
  require_image_vocab(self_arg, buffer_shader_name);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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

// Out-of-place unary with dtype-keyed kernel resolution.
//
// The flag says whether the formula keeps a count integral.  A series or a
// ratio has fractions in it, so its answer is not a count and the payload is
// read in the width the result names; rounding, sign and magnitude work keep
// the count and ride its own kernel.  In place the destination keeps the width
// it was allocated in, so such a formula is refused there by the missing twin.
Tensor unary_op_typed(
    const Tensor& self_arg,
    const char* shader_name,
    const char* buffer_shader_name,
    bool int_capable,
    const char* name) {
  const bool counts = self_arg.dtype() == DType::Int32 ||
      self_arg.dtype() == DType::Int64;
  const Tensor operand =
      (int_capable || !counts) ? self_arg : self_arg.to(DType::Float32);
  const UnaryVocab vocab = unary_vocab(operand, name);
  return unary_op(
      operand,
      kernel_for(
          unary_variant(shader_name, vocab, name, false).c_str(), operand),
      unary_variant(buffer_shader_name, vocab, name, true).c_str());
}

Tensor& unary_op_typed_inplace(
    Tensor& self_arg,
    const char* shader_name,
    const char* buffer_shader_name,
    const char* name) {
  const UnaryVocab vocab = unary_vocab(self_arg, name);
  return unary_op_(
      self_arg,
      kernel_for(
          unary_variant(shader_name, vocab, name, false).c_str(), self_arg),
      unary_variant(buffer_shader_name, vocab, name, true).c_str());
}

} // namespace

Tensor exp_kernel(const Tensor& self) {
  return unary_op_typed(self, "exp", "buffer_exp", false, "unary exp");
}

Tensor& exp_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "expinplace", "buffer_expinplace", "unary exp");
}

Tensor sqrt_kernel(const Tensor& self) {
  return unary_op_typed(self, "sqrt", "buffer_sqrt", false, "unary sqrt");
}

Tensor& sqrt_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "sqrtinplace", "buffer_sqrtinplace", "unary sqrt");
}

Tensor log_kernel(const Tensor& self) {
  return unary_op_typed(self, "log", "buffer_log", false, "unary log");
}

Tensor& log_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "loginplace", "buffer_loginplace", "unary log");
}

Tensor abs_kernel(const Tensor& self) {
  return unary_op_typed(self, "abs", "buffer_abs", true, "unary abs");
}

Tensor& abs_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "absinplace", "buffer_absinplace", "unary abs");
}

Tensor neg_kernel(const Tensor& self) {
  return unary_op_typed(self, "neg", "buffer_neg", true, "unary neg");
}

Tensor& neg_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "neginplace", "buffer_neginplace", "unary neg");
}

Tensor floor_kernel(const Tensor& self) {
  return unary_op_typed(self, "floor", "buffer_floor", true, "unary floor");
}

Tensor& floor_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "floorinplace", "buffer_floorinplace", "unary floor");
}

Tensor sin_kernel(const Tensor& self) {
  return unary_op_typed(self, "sin", "buffer_sin", false, "unary sin");
}

Tensor& sin_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "sininplace", "buffer_sininplace", "unary sin");
}

Tensor cos_kernel(const Tensor& self) {
  return unary_op_typed(self, "cos", "buffer_cos", false, "unary cos");
}

Tensor& cos_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "cosinplace", "buffer_cosinplace", "unary cos");
}

Tensor tanh_kernel(const Tensor& self) {
  return unary_op_typed(self, "tanh", "buffer_tanh", false, "unary tanh");
}

Tensor& tanh_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "tanhinplace", "buffer_tanhinplace", "unary tanh");
}

Tensor sigmoid_kernel(const Tensor& self) {
  return unary_op_typed(
      self, "sigmoid", "buffer_sigmoid", false, "unary sigmoid");
}

Tensor& sigmoid_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "sigmoidinplace", "buffer_sigmoidinplace", "unary sigmoid");
}

// log_sigmoid: min(x, 0) - log(1 + exp(-|x|)), evaluated by one kernel so the
// exponential and the logarithm share a single pass over the payload.
Tensor log_sigmoid_kernel(const Tensor& self) {
  return unary_op_typed(
      self, "log_sigmoid", "buffer_log_sigmoid", false, "log_sigmoid");
}

Tensor relu_kernel(const Tensor& self) {
  return unary_op_typed(self, "relu", "buffer_relu", true, "unary relu");
}

Tensor rsqrt_kernel(const Tensor& self) {
  return unary_op_typed(self, "rsqrt", "buffer_rsqrt", false, "unary rsqrt");
}

Tensor sign_kernel(const Tensor& self) {
  return unary_op_typed(self, "sign", "buffer_sign", true, "unary sign");
}

Tensor& relu_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "reluinplace", "buffer_reluinplace", "unary relu");
}

//
// New pointwise entries: each maps onto one generated unary variant.  The
// Int32 column lists the ops with a dedicated `_i32` build; the rest are
// float-vocabulary ops (math functions without an integer meaning).
//
Tensor ceil_kernel(const Tensor& self) {
  return unary_op_typed(self, "ceil", "buffer_ceil", true, "unary ceil");
}

Tensor& ceil_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "ceilinplace", "buffer_ceilinplace", "unary ceil");
}

Tensor round_kernel(const Tensor& self) {
  return unary_op_typed(self, "round", "buffer_round", true, "unary round");
}

Tensor& round_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "roundinplace", "buffer_roundinplace", "unary round");
}

Tensor trunc_kernel(const Tensor& self) {
  return unary_op_typed(self, "trunc", "buffer_trunc", true, "unary trunc");
}

Tensor& trunc_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "truncinplace", "buffer_truncinplace", "unary trunc");
}

// The fraction a lane keeps after the whole toward zero is taken away.  A
// count holds none of itself to take, so a counting payload is turned away
// rather than read wide and answered with a zero.
Tensor frac_kernel(const Tensor& self) {
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan frac takes the fraction off a measured payload; a count has "
      "none");
  return unary_op_typed(self, "frac", "buffer_frac", false, "unary frac");
}

Tensor& frac_inplace_kernel(Tensor& self) {
  TP_CHECK(
      self.dtype() == DType::Float32 || self.dtype() == DType::Float16,
      "Vulkan frac_ takes the fraction off a measured payload; a count has "
      "none");
  return unary_op_typed_inplace(
      self, "fracinplace", "buffer_fracinplace", "unary frac");
}

Tensor tan_kernel(const Tensor& self) {
  return unary_op_typed(self, "tan", "buffer_tan", false, "unary tan");
}

Tensor& tan_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "taninplace", "buffer_taninplace", "unary tan");
}

Tensor sinh_kernel(const Tensor& self) {
  return unary_op_typed(self, "sinh", "buffer_sinh", false, "unary sinh");
}

Tensor& sinh_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "sinhinplace", "buffer_sinhinplace", "unary sinh");
}

Tensor cosh_kernel(const Tensor& self) {
  return unary_op_typed(self, "cosh", "buffer_cosh", false, "unary cosh");
}

Tensor& cosh_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "coshinplace", "buffer_coshinplace", "unary cosh");
}

Tensor asin_kernel(const Tensor& self) {
  return unary_op_typed(self, "asin", "buffer_asin", false, "unary asin");
}

Tensor& asin_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "asininplace", "buffer_asininplace", "unary asin");
}

Tensor acos_kernel(const Tensor& self) {
  return unary_op_typed(self, "acos", "buffer_acos", false, "unary acos");
}

Tensor& acos_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "acosinplace", "buffer_acosinplace", "unary acos");
}

Tensor atan_kernel(const Tensor& self) {
  return unary_op_typed(self, "atan", "buffer_atan", false, "unary atan");
}

Tensor& atan_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "ataninplace", "buffer_ataninplace", "unary atan");
}

Tensor exp2_kernel(const Tensor& self) {
  return unary_op_typed(self, "exp2", "buffer_exp2", false, "unary exp2");
}

Tensor& exp2_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "exp2inplace", "buffer_exp2inplace", "unary exp2");
}

Tensor expm1_kernel(const Tensor& self) {
  return unary_op_typed(self, "expm1", "buffer_expm1", false, "unary expm1");
}

Tensor& expm1_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "expm1inplace", "buffer_expm1inplace", "unary expm1");
}

Tensor log1p_kernel(const Tensor& self) {
  return unary_op_typed(self, "log1p", "buffer_log1p", false, "unary log1p");
}

Tensor& log1p_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "log1pinplace", "buffer_log1pinplace", "unary log1p");
}

Tensor log2_kernel(const Tensor& self) {
  return unary_op_typed(self, "log2", "buffer_log2", false, "unary log2");
}

Tensor& log2_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "log2inplace", "buffer_log2inplace", "unary log2");
}

Tensor log10_kernel(const Tensor& self) {
  return unary_op_typed(self, "log10", "buffer_log10", false, "unary log10");
}

Tensor& log10_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(self, "log10inplace", "buffer_log10inplace", "unary log10");
}

Tensor square_kernel(const Tensor& self) {
  return unary_op_typed(self, "square", "buffer_square", true, "unary square");
}

Tensor& square_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "squareinplace", "buffer_squareinplace", "unary square");
}

Tensor reciprocal_kernel(const Tensor& self) {
  return unary_op_typed(
      self, "reciprocal", "buffer_reciprocal", false, "unary reciprocal");
}

Tensor& reciprocal_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "reciprocalinplace", "buffer_reciprocalinplace",
      "unary reciprocal");
}

Tensor erf_kernel(const Tensor& self) {
  return unary_op_typed(self, "erf", "buffer_erf", false, "unary erf");
}

Tensor& erf_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "erfinplace", "buffer_erfinplace", "unary erf");
}

Tensor erfc_kernel(const Tensor& self) {
  return unary_op_typed(self, "erfc", "buffer_erfc", false, "unary erfc");
}

Tensor& erfc_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "erfcinplace", "buffer_erfcinplace", "unary erfc");
}

Tensor lgamma_kernel(const Tensor& self) {
  return unary_op_typed(
      self, "lgamma", "buffer_lgamma", false, "unary lgamma");
}

Tensor& lgamma_inplace_kernel(Tensor& self) {
  return unary_op_typed_inplace(
      self, "lgammainplace", "buffer_lgammainplace", "unary lgamma");
}

// Conjugation is the identity for the real and integer dtypes this backend
// supports; complex tensors are rejected at allocation time.
Tensor conj_kernel(const Tensor& self) {
  return self;
}

Tensor resolve_conj_identity(const Tensor& self) { return self; }
Tensor resolve_neg_identity(const Tensor& self) { return self; }

TENSORPLAY_LIBRARY_IMPL(Vulkan, UnaryOpKernels) {
  m.impl("resolve_conj", &tensorplay::vulkan::ops::resolve_conj_identity);
  m.impl("resolve_neg", &tensorplay::vulkan::ops::resolve_neg_identity);

  m.impl("exp", &tensorplay::vulkan::ops::exp_kernel);
  m.impl("exp_", &tensorplay::vulkan::ops::exp_inplace_kernel);
  m.impl("sqrt", &tensorplay::vulkan::ops::sqrt_kernel);
  m.impl("sqrt_", &tensorplay::vulkan::ops::sqrt_inplace_kernel);
  m.impl("log", &tensorplay::vulkan::ops::log_kernel);
  m.impl("log_", &tensorplay::vulkan::ops::log_inplace_kernel);
  m.impl("abs", &tensorplay::vulkan::ops::abs_kernel);
  m.impl("abs_", &tensorplay::vulkan::ops::abs_inplace_kernel);
  m.impl("neg", &tensorplay::vulkan::ops::neg_kernel);
  m.impl("neg_", &tensorplay::vulkan::ops::neg_inplace_kernel);
  m.impl("floor", &tensorplay::vulkan::ops::floor_kernel);
  m.impl("floor_", &tensorplay::vulkan::ops::floor_inplace_kernel);
  m.impl("sin", &tensorplay::vulkan::ops::sin_kernel);
  m.impl("sin_", &tensorplay::vulkan::ops::sin_inplace_kernel);
  m.impl("cos", &tensorplay::vulkan::ops::cos_kernel);
  m.impl("cos_", &tensorplay::vulkan::ops::cos_inplace_kernel);
  m.impl("tanh", &tensorplay::vulkan::ops::tanh_kernel);
  m.impl("tanh_", &tensorplay::vulkan::ops::tanh_inplace_kernel);
  m.impl("sigmoid", &tensorplay::vulkan::ops::sigmoid_kernel);
  m.impl("sigmoid_", &tensorplay::vulkan::ops::sigmoid_inplace_kernel);
  m.impl("log_sigmoid", &tensorplay::vulkan::ops::log_sigmoid_kernel);
  m.impl("relu", &tensorplay::vulkan::ops::relu_kernel);
  m.impl("relu_", &tensorplay::vulkan::ops::relu_inplace_kernel);
  m.impl("rsqrt", &tensorplay::vulkan::ops::rsqrt_kernel);
  m.impl("sign", &tensorplay::vulkan::ops::sign_kernel);
  m.impl("ceil", &tensorplay::vulkan::ops::ceil_kernel);
  m.impl("ceil_", &tensorplay::vulkan::ops::ceil_inplace_kernel);
  m.impl("round", &tensorplay::vulkan::ops::round_kernel);
  m.impl("round_", &tensorplay::vulkan::ops::round_inplace_kernel);
  m.impl("trunc", &tensorplay::vulkan::ops::trunc_kernel);
  m.impl("trunc_", &tensorplay::vulkan::ops::trunc_inplace_kernel);
  m.impl("frac", &tensorplay::vulkan::ops::frac_kernel);
  m.impl("frac_", &tensorplay::vulkan::ops::frac_inplace_kernel);
  m.impl("tan", &tensorplay::vulkan::ops::tan_kernel);
  m.impl("tan_", &tensorplay::vulkan::ops::tan_inplace_kernel);
  m.impl("sinh", &tensorplay::vulkan::ops::sinh_kernel);
  m.impl("sinh_", &tensorplay::vulkan::ops::sinh_inplace_kernel);
  m.impl("cosh", &tensorplay::vulkan::ops::cosh_kernel);
  m.impl("cosh_", &tensorplay::vulkan::ops::cosh_inplace_kernel);
  m.impl("asin", &tensorplay::vulkan::ops::asin_kernel);
  m.impl("asin_", &tensorplay::vulkan::ops::asin_inplace_kernel);
  m.impl("acos", &tensorplay::vulkan::ops::acos_kernel);
  m.impl("acos_", &tensorplay::vulkan::ops::acos_inplace_kernel);
  m.impl("atan", &tensorplay::vulkan::ops::atan_kernel);
  m.impl("atan_", &tensorplay::vulkan::ops::atan_inplace_kernel);
  m.impl("exp2", &tensorplay::vulkan::ops::exp2_kernel);
  m.impl("exp2_", &tensorplay::vulkan::ops::exp2_inplace_kernel);
  m.impl("expm1", &tensorplay::vulkan::ops::expm1_kernel);
  m.impl("expm1_", &tensorplay::vulkan::ops::expm1_inplace_kernel);
  m.impl("log1p", &tensorplay::vulkan::ops::log1p_kernel);
  m.impl("log1p_", &tensorplay::vulkan::ops::log1p_inplace_kernel);
  m.impl("log2", &tensorplay::vulkan::ops::log2_kernel);
  m.impl("log2_", &tensorplay::vulkan::ops::log2_inplace_kernel);
  m.impl("log10", &tensorplay::vulkan::ops::log10_kernel);
  m.impl("log10_", &tensorplay::vulkan::ops::log10_inplace_kernel);
  m.impl("square", &tensorplay::vulkan::ops::square_kernel);
  m.impl("square_", &tensorplay::vulkan::ops::square_inplace_kernel);
  m.impl("reciprocal", &tensorplay::vulkan::ops::reciprocal_kernel);
  m.impl("reciprocal_", &tensorplay::vulkan::ops::reciprocal_inplace_kernel);
  m.impl("erf", &tensorplay::vulkan::ops::erf_kernel);
  m.impl("erf_", &tensorplay::vulkan::ops::erf_inplace_kernel);
  m.impl("erfc", &tensorplay::vulkan::ops::erfc_kernel);
  m.impl("erfc_", &tensorplay::vulkan::ops::erfc_inplace_kernel);
  m.impl("lgamma", &tensorplay::vulkan::ops::lgamma_kernel);
  m.impl("lgamma_", &tensorplay::vulkan::ops::lgamma_inplace_kernel);
  m.impl("conj", &tensorplay::vulkan::ops::conj_kernel);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
