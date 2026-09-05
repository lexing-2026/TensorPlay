#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../impl/Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

struct BlockB final {
  uint32_t buf_length;
  uint32_t fill0;
  float alpha;
};

struct BlockBi final {
  uint32_t buf_length;
  uint32_t fill0;
  int32_t alpha;
};

// Element-width classes for the binary dispatch, mirroring the unary one.
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

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  if (v_output.storage_type() == api::StorageType::BUFFER) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_output.numel()));
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context, BlockBi{n, 0u, static_cast<int32_t>(alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
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

  api::UniformParamsBuffer params(
      context,
      (vocab == BinaryVocab::kInt)
          ? api::UniformParamsBuffer(context, blocki)
          : api::UniformParamsBuffer(context, block));
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(variant.c_str()),
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
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    api::UniformParamsBuffer params = (vocab == BinaryVocab::kInt)
        ? api::UniformParamsBuffer(
              context, BlockBi{n, 0u, static_cast<int32_t>(alpha.to<int64_t>())})
        : api::UniformParamsBuffer(
              context, BlockB{n, 0u, alpha.to<float>()});
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL_FROM_STR(buffer_variant.c_str()), pipeline_barrier,
        {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
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

  api::UniformParamsBuffer params(
      context,
      (vocab == BinaryVocab::kInt)
          ? api::UniformParamsBuffer(context, blocki)
          : api::UniformParamsBuffer(context, block));
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(variant.c_str()),
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

  api::vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const bool is_buffer = v_output.storage_type() == api::StorageType::BUFFER;

  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      make_whcn_ivec4(v_output.sizes()),
      other.to<float>() * alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 extents;
    // scalar argument
    int32_t other;
  } blocki{
      make_whcn_ivec4(v_output.sizes()),
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
        {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_output.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::WRITE),
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        paramsb.buffer());
    return convert(v_output);
  }

  api::UniformParamsBuffer params(
      context,
      (vocab == BinaryVocab::kInt)
          ? api::UniformParamsBuffer(context, blocki)
          : api::UniformParamsBuffer(context, block));
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(variant.c_str()),
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
        {n, 1u, 1u}, {64u, 1u, 1u}, VK_NULL_HANDLE,
        v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                      api::MemoryAccessType::WRITE),
        params.buffer());
    return self_arg;
  }

  const struct Block final {
    ivec4 extents;
    // scalar argument
    float other;
  } block{
      make_whcn_ivec4(v_self.sizes()),
      other.to<float>() * alpha.to<float>(),
  };

  const struct BlockI final {
    ivec4 extents;
    // scalar argument
    int32_t other;
  } blocki{
      make_whcn_ivec4(v_self.sizes()),
      static_cast<int32_t>(other.to<int64_t>() * alpha.to<int64_t>()),
  };

  api::UniformParamsBuffer params(
      context,
      (vocab == BinaryVocab::kInt)
          ? api::UniformParamsBuffer(context, blocki)
          : api::UniformParamsBuffer(context, block));
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL_FROM_STR(variant.c_str()),
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

Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor("add", "buffer_add", self, other, alpha, "add");
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor("sub", "buffer_sub", self, other, alpha, "sub");
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("mul", "buffer_mul", self, other, Scalar(1.0), "mul");
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("div", "buffer_div", self, other, Scalar(1.0), "div");
}

Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self, other,
                          alpha, "add");
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar("add_scalar", "buffer_add_scalar", self,
                          Scalar(-other.toDouble()), alpha, "add");
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self, other,
                          Scalar(1.0), "mul");
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self,
                          Scalar(1.0 / other.toDouble()), Scalar(1.0), "mul");
}

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
  return binary_op_tensor_inplace("addinplace", "buffer_addinplace", self,
                                  other, alpha, "add");
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
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

Tensor& add_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar_inplace("add_scalarinplace",
                                  "buffer_add_scalarinplace",
                                  self, other, alpha, "add");
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, Scalar other) {
  return binary_op_scalar_inplace("mul_scalarinplace",
                                  "buffer_mul_scalarinplace",
                                  self, other, Scalar(1.0), "mul");
}

// The subtraction in-place scalar flavor rides the add writer with the
// negated product: x -= y * alpha runs as x += -(y * alpha), matching the
// reference decomposition where one shader family covers both directions.
Tensor& sub_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar_inplace("add_scalarinplace",
                                  "buffer_add_scalarinplace",
                                  self, other, Scalar(-alpha.toDouble()), "add");
}

Tensor& div_scalar_inplace_kernel(Tensor& self, Scalar other) {
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

Tensor floor_divide_scalar_kernel(const Tensor& self, Scalar other) {
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

Tensor& floor_divide_scalar_inplace_kernel(Tensor& self, Scalar other) {
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
Tensor pow_scalar_base_kernel(Scalar base, const Tensor& self) {
  return binary_op_scalar(
      "pow_scalar_tensor", "pow_scalar_tensor", self, base,
      Scalar(1.0), "pow");
}

Tensor& pow_tensor_scalar_inplace_kernel(Tensor& self, Scalar exponent) {
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

Tensor logaddexp_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("logaddexp", "buffer_logaddexp", self, other,
                          Scalar(1.0), "logaddexp");
}

Tensor rsub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
  return binary_op_scalar("rsub_scalar", "buffer_rsub_scalar", self, other,
                          alpha, "rsub");
}

Tensor true_divide_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("div", "buffer_div", self, other, Scalar(1.0),
                          "true_divide");
}

Tensor true_divide_scalar_kernel(const Tensor& self, Scalar other) {
  TP_CHECK(
      other.toDouble() != 0.0,
      "true_divide.Scalar: can't divide by zero");
  return binary_op_scalar("mul_scalar", "buffer_mul_scalar", self,
                          Scalar(1.0 / other.toDouble()), Scalar(1.0), "mul");
}

Tensor divide_tensor_kernel(const Tensor& self, const Tensor& other) {
  return binary_op_tensor("div", "buffer_div", self, other, Scalar(1.0),
                          "divide");
}

Tensor divide_scalar_kernel(const Tensor& self, Scalar other) {
  return true_divide_scalar_kernel(self, other);
}

Tensor subtract_tensor_kernel(const Tensor& self, const Tensor& other,
                              Scalar alpha) {
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
  m.impl("atan2_", &tensorplay::vulkan::ops::atan2_kernel);
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
