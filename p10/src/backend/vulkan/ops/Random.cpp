#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Factory.h"
#include "Utils.h"

#include "../api/Context.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

//
// Random fill for Vulkan tensors.
//
// The draws happen on the device: a counter-based Philox stream derives every
// element's words from (key, offset, invocation id), so nothing is generated
// on the host and nothing is uploaded.  One texel carries four elements and
// one Philox call yields four words, so each channel lane consumes its own
// word.
//
// The host generator stays the single source of entropy: each call draws the
// Philox key and offset from it, which advances its state exactly as a host
// fill would.  A seeded run therefore reproduces the same tensors, and two
// calls in a row never repeat a stream.
//

struct RandomFillBlock final {
  uvec3 extents;
  uint32_t extents_fill;
  float from;  // uniform lower bound / bernoulli probability
  float to;    // uniform upper bound / normal mean
  float std;   // normal standard deviation
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t offset;
  uint32_t fill;
};

struct BernoulliTensorBlock final {
  uvec3 extents;
  uint32_t extents_fill;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t offset;
  uint32_t fill;
};

struct StreamKey final {
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t offset;
};

// One key per call, drawn from the host generator so its state advances and a
// seeded run stays reproducible.
StreamKey next_stream_key(std::optional<Generator>& generator) {
  Generator& gen = generator.has_value() ? *generator : default_generator();
  const uint64_t key = gen.random64();
  return StreamKey{
      static_cast<uint32_t>(key & 0xffffffffULL),
      static_cast<uint32_t>(key >> 32),
      gen.random(),
  };
}

void validate_random_target(const Tensor& t, const char* name) {
  TP_CHECK(
      t.device().is_vulkan(), std::string("Vulkan ") + name +
          " expects a Vulkan tensor");
  TP_CHECK(
      t.dtype() == DType::Float32,
      std::string("Vulkan ") + name + " supports Float32 tensors only");
  TP_CHECK(
      t.dim() >= 1 && t.dim() <= 4,
      std::string("Vulkan ") + name + " supports 1d to 4d tensors");
}

// Shared dispatch for the three nullary distributions; `from`, `to` and `std`
// carry whichever parameters the selected variant reads.
void dispatch_random_fill(
    Tensor& self,
    const api::ShaderInfo& shader,
    float from,
    float to,
    float std,
    std::optional<Generator>& generator) {
  api::Context* const context = api::context();
  api::vTensor v = convert(self);
  const StreamKey key = next_stream_key(generator);

  const RandomFillBlock block{
      v.extents(), 0u, from, to, std,
      key.seed_lo, key.seed_hi, key.offset, 0u,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      shader,
      pipeline_barrier,
      v.extents(),
      adaptive_work_group_size(v.extents()),
      VK_NULL_HANDLE,
      v.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      params.buffer());
}

} // namespace

Tensor& uniform_kernel(
    Tensor& self,
    double from,
    double to,
    std::optional<Generator> generator) {
  validate_random_target(self, "uniform_");
  TP_CHECK(
      from <= to,
      "uniform_ expects to return a [from, to) range, but found from=",
      from, " > to=", to);

  if (self.numel() == 0) {
    return self;
  }
  dispatch_random_fill(
      self, VK_KERNEL(uniform_fill), static_cast<float>(from),
      static_cast<float>(to), 0.0f, generator);
  return self;
}

Tensor& normal_kernel(
    Tensor& self,
    double mean,
    double std,
    std::optional<Generator> generator) {
  validate_random_target(self, "normal_");
  TP_CHECK(
      std >= 0.0,
      "normal expects std >= 0.0, but found std ", std);

  if (self.numel() == 0) {
    return self;
  }
  // The normal variant reads the mean from `to` and the spread from `std`.
  dispatch_random_fill(
      self, VK_KERNEL(normal_fill), 0.0f, static_cast<float>(mean),
      static_cast<float>(std), generator);
  return self;
}

Tensor& bernoulli_scalar_kernel(
    Tensor& self,
    double p,
    std::optional<Generator> generator) {
  validate_random_target(self, "bernoulli_");
  TP_CHECK(
      p >= 0.0 && p <= 1.0,
      "bernoulli expects p to be in [0, 1], but got p=", p);

  if (self.numel() == 0) {
    return self;
  }
  // The bernoulli variant reads its probability from `from`.
  dispatch_random_fill(
      self, VK_KERNEL(bernoulli_fill), static_cast<float>(p), 0.0f, 0.0f,
      generator);
  return self;
}

Tensor& bernoulli_tensor_kernel(
    Tensor& self,
    const Tensor& p,
    std::optional<Generator> generator) {
  validate_random_target(self, "bernoulli_");
  TP_CHECK(
      p.numel() == 1 || p.shape() == self.shape(),
      "Vulkan bernoulli_ expects p to be a scalar or match self's shape");
  TP_CHECK(
      p.device().is_vulkan(),
      "Vulkan bernoulli_ expects p on the same device");

  if (self.numel() == 0) {
    return self;
  }
  if (p.numel() == 1 && p.shape() != self.shape()) {
    // A single probability governs every element; the scalar variant needs no
    // probability plane.
    return bernoulli_scalar_kernel(self, p.item().toDouble(), generator);
  }

  api::Context* const context = api::context();
  api::vTensor v = convert(self);
  api::vTensor v_prob = convert(p);
  const StreamKey key = next_stream_key(generator);

  const BernoulliTensorBlock block{
      v.extents(), 0u, key.seed_lo, key.seed_hi, key.offset, 0u,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(bernoulli_tensor_fill),
      pipeline_barrier,
      v.extents(),
      adaptive_work_group_size(v.extents()),
      VK_NULL_HANDLE,
      v.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_prob.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return self;
}

/*
 * *_like factories: clone the source (same sizes, dtype and device), then
 * refill the clone in place from the random stream.  dtype/device overrides
 * fall back to the generic factory path, matching the public *_like contract
 * for non-defaulted arguments.
 */
Tensor rand_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device) {
  TP_CHECK(
      dtype == DType::Undefined || dtype == DType::Float32,
      "Vulkan rand_like supports Float32 tensors only");
  TP_CHECK(
      !device.has_value() || device->is_vulkan(),
      "Vulkan rand_like expects a Vulkan device");
  if (dtype != DType::Undefined || (device.has_value() && !device->is_vulkan())) {
    return full_like_kernel(self, Scalar(0.0), dtype, device);
  }
  Tensor out = self.clone();
  return uniform_kernel(out, 0.0, 1.0, std::nullopt);
}

Tensor randn_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device) {
  TP_CHECK(
      dtype == DType::Undefined || dtype == DType::Float32,
      "Vulkan randn_like supports Float32 tensors only");
  TP_CHECK(
      !device.has_value() || device->is_vulkan(),
      "Vulkan randn_like expects a Vulkan device");
  if (dtype != DType::Undefined || (device.has_value() && !device->is_vulkan())) {
    return full_like_kernel(self, Scalar(0.0), dtype, device);
  }
  Tensor out = self.clone();
  return normal_kernel(out, 0.0, 1.0, std::nullopt);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, RandomKernels) {
  m.impl("uniform_", &tensorplay::vulkan::ops::uniform_kernel);
  m.impl("normal_", &tensorplay::vulkan::ops::normal_kernel);
  m.impl("bernoulli_.float", &tensorplay::vulkan::ops::bernoulli_scalar_kernel);
  m.impl("bernoulli_.Tensor", &tensorplay::vulkan::ops::bernoulli_tensor_kernel);
  m.impl("rand_like", &tensorplay::vulkan::ops::rand_like_kernel);
  m.impl("randn_like", &tensorplay::vulkan::ops::randn_like_kernel);
}

#endif /* USE_VULKAN */
