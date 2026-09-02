#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include "../api/Context.h"

#include <cmath>
#include <optional>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

//
// Random fill for Vulkan tensors.  The random stream comes from the host
// generator: values are drawn with exactly the same formulas the CPU
// kernels use (24-bit-mantissa uniforms, Box-Muller normals), so a given
// seed yields the same stream regardless of which backend receives the
// call.  The payload is then streamed into the GPU texture through the
// staging pipeline, which handles every supported VkFormat.
//
// A shader-side Philox path also exists for element-wise draws, but the
// host stream keeps one canonical generator story for the backend and
// avoids splitting the seed state between host and device.
//

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

// Reuploads the whole payload after the host wrote fresh values into a
// staging tensor with the texture-linear layout.
void upload_payload(api::vTensor& v, Tensor& host_packed) {
  utils::upload_host_bytes(
      v,
      host_packed.impl()->storage().data(),
      host_packed.numel() * host_packed.itemsize());
}

//
// Scatters logical-order values into the texture-linear staging buffer:
// texture plane (z = batch * ceil(C/4) + channel/4) holds the four channel
// lanes of one spatial position, mirroring the nchw packing used by the
// staging copy pipeline.  Padding lanes stay zero and are never read back.
//
void pack_logical_into_staging(
    const api::vTensor& v,
    const float* logical,
    Tensor& host) {
  const int64_t N = get_dim<Dim4D::Batch>(v.sizes());
  const int64_t C = get_dim<Dim4D::Channel>(v.sizes());
  const int64_t H = get_dim<Dim4D::Height>(v.sizes());
  const int64_t W = get_dim<Dim4D::Width>(v.sizes());
  const int64_t c_depth = (C + 3) / 4;

  float* data = host.data_ptr<float>();
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t z = n * c_depth + c / 4;
      const int64_t lane = c % 4;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          data[((z * H + h) * W + w) * 4 + lane] =
              logical[((n * C + c) * H + h) * W + w];
        }
      }
    }
  }
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

  api::Context* const context = api::context();
  api::vTensor v = convert(self);

  // Draw the stream on the host in logical order, matching the CPU kernel's
  // sampling formula element for element, then scatter into the
  // texture-linear staging layout.
  Tensor host = utils::create_staging_tensor(v);
  const int64_t n = self.numel();
  std::vector<float> logical(static_cast<size_t>(n));
  Generator& gen = generator.has_value() ? *generator : default_generator();
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t r = gen.random();
    const double x = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
    logical[static_cast<size_t>(i)] =
        static_cast<float>(x * (to - from) + from);
  }

  pack_logical_into_staging(v, logical.data(), host);
  upload_payload(v, host);
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

  api::Context* const context = api::context();
  api::vTensor v = convert(self);

  Tensor host = utils::create_staging_tensor(v);
  const int64_t n = self.numel();
  std::vector<float> logical(static_cast<size_t>(n));
  float* data = logical.data();
  Generator& gen = generator.has_value() ? *generator : default_generator();

  // Box-Muller over 24-bit-mantissa uniforms, matching the CPU path's
  // distribution.  Pairs are consumed sequentially; an odd element count
  // discards the second draw of the final pair.
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  for (int64_t i = 0; i < n; i += 2) {
    const double u1 = 1.0 - ((gen.random() & ((1u << 24) - 1)) *
        std::ldexp(1.0, -24)); // (0, 1]
    const double u2 = (gen.random() & ((1u << 24) - 1)) *
        std::ldexp(1.0, -24); // [0, 1)
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = kTwoPi * u2;
    data[i] = static_cast<float>(mean + std * r * std::cos(theta));
    if (i + 1 < n) {
      data[i + 1] = static_cast<float>(mean + std * r * std::sin(theta));
    }
  }

  pack_logical_into_staging(v, data, host);
  upload_payload(v, host);
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

  api::Context* const context = api::context();
  api::vTensor v = convert(self);

  Tensor host = utils::create_staging_tensor(v);
  const int64_t n = self.numel();
  std::vector<float> logical(static_cast<size_t>(n));
  float* data = logical.data();
  Generator& gen = generator.has_value() ? *generator : default_generator();
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t r = gen.random();
    const double u = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
    data[i] = (u < p) ? 1.0f : 0.0f;
  }

  pack_logical_into_staging(v, data, host);
  upload_payload(v, host);
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

  // Materialize p on the host once, then sample against it element-wise.
  const Tensor p_host = p.to(Device(DeviceType::CPU)).contiguous();
  const float* p_data = p_host.data_ptr<float>();

  api::Context* const context = api::context();
  api::vTensor v = convert(self);

  Tensor host = utils::create_staging_tensor(v);
  const int64_t n = self.numel();
  std::vector<float> logical(static_cast<size_t>(n));
  float* data = logical.data();
  Generator& gen = generator.has_value() ? *generator : default_generator();
  const int64_t p_n = p_host.numel();

  for (int64_t i = 0; i < n; ++i) {
    const uint32_t r = gen.random();
    const double u = (r & ((1u << 24) - 1)) * std::ldexp(1.0, -24);
    data[i] = (u < p_data[p_n == 1 ? 0 : i]) ? 1.0f : 0.0f;
  }

  pack_logical_into_staging(v, data, host);
  upload_payload(v, host);
  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, RandomKernels) {
  m.impl("uniform_", &tensorplay::vulkan::ops::uniform_kernel);
  m.impl("normal_", &tensorplay::vulkan::ops::normal_kernel);
  m.impl("bernoulli_.float", &tensorplay::vulkan::ops::bernoulli_scalar_kernel);
  m.impl("bernoulli_.Tensor", &tensorplay::vulkan::ops::bernoulli_tensor_kernel);
}

#endif /* USE_VULKAN */
