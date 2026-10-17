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

// Sibling Vulkan selection kernels used by the token sampler.
std::tuple<Tensor, Tensor> sort_kernel(
    const Tensor& self, int64_t dim, bool descending);
std::tuple<Tensor, Tensor> topk_kernel(
    const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted,
    int64_t impl);

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

// Buffer-stored fills carry the element count instead of image extents.
struct RandomBufferFillBlock final {
  uint32_t buf_length;
  float from; // uniform lower bound / bernoulli probability
  float to;   // uniform upper bound / normal mean
  float std;  // normal standard deviation
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
// carry whichever parameters the selected variant reads.  `shader` fills
// texture-stored payloads and `buffer_shader` fills shapes whose extents fall
// back to buffer storage.
void dispatch_random_fill(
    Tensor& self,
    const api::ShaderInfo& shader,
    const api::ShaderInfo& buffer_shader,
    float from,
    float to,
    float std,
    std::optional<Generator>& generator) {
  api::Context* const context = api::context();
  api::vTensor v = convert(self);
  const StreamKey key = next_stream_key(generator);
  api::PipelineBarrier pipeline_barrier{};

  if (v.storage_type() == api::StorageType::BUFFER) {
    const uint32_t groups =
        (static_cast<uint32_t>(self.numel()) + 3u) / 4u;
    api::UniformParamsBuffer params(
        context, RandomBufferFillBlock{
            static_cast<uint32_t>(self.numel()), from, to, std,
            key.seed_lo, key.seed_hi, key.offset, 0u,
        });
    context->submit_compute_job(
        buffer_shader,
        pipeline_barrier,
        {groups, 1u, 1u},
        {256u, 1u, 1u},
        VK_NULL_HANDLE,
        v.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        params.buffer());
    return;
  }

  const RandomFillBlock block{
      v.extents(), 0u, from, to, std,
      key.seed_lo, key.seed_hi, key.offset, 0u,
  };
  api::UniformParamsBuffer params(context, block);

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
      self, VK_KERNEL(uniform_fill), VK_KERNEL(buffer_uniform_fill),
      static_cast<float>(from), static_cast<float>(to), 0.0f, generator);
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
      self, VK_KERNEL(normal_fill), VK_KERNEL(buffer_normal_fill), 0.0f,
      static_cast<float>(mean), static_cast<float>(std), generator);
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
      self, VK_KERNEL(bernoulli_fill), VK_KERNEL(buffer_bernoulli_fill),
      static_cast<float>(p), 0.0f, 0.0f, generator);
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
  // The tensor-probability sampler reads both payloads through image
  // bindings; buffer-stored shapes (beyond the 3-D image axis limit) have no
  // images to read.
  TP_CHECK(
      v.storage_type() != api::StorageType::BUFFER &&
          v_prob.storage_type() != api::StorageType::BUFFER,
      "Vulkan bernoulli_ with a probability tensor requires texture-stored "
      "payloads; shapes beyond the 3-D image axis limit are not supported "
      "yet");
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

// Nullary normal factory: allocate a Float32 tensor on the device and fill
// it in place from the same counter-based stream as normal_.  The host
// generator advances per call, so seeded runs reproduce their tensors.
Tensor randn_kernel(
    const std::vector<int64_t>& size,
    std::optional<DType> dtype,
    std::optional<Device> device) {
  const DType out_dtype = dtype.has_value() ? *dtype : DType::Float32;
  const Device out_device =
      device.has_value() ? *device : Device(DeviceType::Vulkan);
  TP_CHECK(
      out_device.is_vulkan(), "Vulkan randn expects a Vulkan device");
  Tensor out = empty_kernel(size, out_dtype, out_device, false);
  if (out.numel() > 0) {
    normal_kernel(out, 0.0, 1.0, std::nullopt);
  }
  return out;
}

Tensor rand_kernel(
    const std::vector<int64_t>& size,
    std::optional<DType> dtype,
    std::optional<Device> device) {
  const DType out_dtype = dtype.has_value() ? *dtype : DType::Float32;
  const Device out_device =
      device.has_value() ? *device : Device(DeviceType::Vulkan);
  TP_CHECK(
      out_device.is_vulkan(), "Vulkan rand expects a Vulkan device");
  Tensor out = empty_kernel(size, out_dtype, out_device, false);
  if (out.numel() > 0) {
    uniform_kernel(out, 0.0, 1.0, std::nullopt);
  }
  return out;
}

Tensor multinomial_kernel(
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    std::optional<Generator> generator) {
  TP_CHECK(
      num_samples >= 0, "multinomial: num_samples must be >= 0");
  TP_CHECK(
      self.dim() <= 2, "Vulkan multinomial: input must be 1D or 2D");
  const bool is_1d = self.dim() == 1;
  const int64_t rows = is_1d ? 1 : self.size(0);
  const int64_t cols = is_1d ? self.numel() : self.size(1);
  if (cols == 0 && num_samples > 0) {
    TP_THROW(
        RuntimeError, "multinomial: input must have at least one category");
  }
  TP_CHECK(
      replacement,
      "Vulkan multinomial currently supports replacement=True only");

  const std::vector<int64_t> out_shape =
      is_1d ? std::vector<int64_t>{num_samples}
            : std::vector<int64_t>{rows, num_samples};
  if (num_samples == 0) {
    return empty_kernel(out_shape, DType::Int64, self.device(), false);
  }

  Tensor prob = self.dtype() == DType::Float32 ? self : self.to(DType::Float32);

  // Draw uniform noise on the device through the same counter-based stream
  // as uniform_, then run one inverse-CDF thread per row.
  Tensor noise = empty_kernel(
      {rows * num_samples}, DType::Float32, self.device(), false);
  dispatch_random_fill(
      noise, VK_KERNEL(uniform_fill), VK_KERNEL(buffer_uniform_fill), 0.0f,
      1.0f, 0.0f, generator);

  api::Context* const context = api::context();
  api::vTensor v_prob = convert(prob);
  api::vTensor v_noise = convert(noise);
  // The inverse-CDF scan reads the whole probability row through an image
  // binding; buffer-stored rows (beyond the 3-D image axis limit) have no
  // image to read.
  TP_CHECK(
      v_prob.storage_type() != api::StorageType::BUFFER,
      "Vulkan multinomial requires texture-stored probabilities; shapes "
      "beyond the 3-D image axis limit are not supported yet");
  api::vTensor v_result{
      context, out_shape, DType::Int64, api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};

  const struct MultinomialBlock final {
    int cols;
    int num_samples;
    int total;
  } block{
      static_cast<int32_t>(cols),
      static_cast<int32_t>(num_samples),
      static_cast<int32_t>(rows * num_samples)};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(rows), 1u, 1u};

  context->submit_compute_job(
      VK_KERNEL(multinomial), barrier, grid, {1u, 1u, 1u}, VK_NULL_HANDLE,
      v_result.buffer(
          barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
      v_prob.image(barrier, api::PipelineStage::COMPUTE),
      v_noise.image(barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_result);
}

Tensor sample_kernel(
    const Tensor& logits,
    double temperature,
    int64_t top_k,
    double top_p,
    int64_t impl) {
  (void)impl;
  TP_CHECK(
      logits.device().is_vulkan(),
      "Vulkan sample expects a Vulkan tensor");
  TP_CHECK(temperature > 0.0, "sample: temperature must be > 0");
  TP_CHECK(
      top_p > 0.0 && top_p <= 1.0, "sample: top_p must be in (0, 1]");
  TP_CHECK(
      logits.dim() == 2, "sample: logits must be 2D [batch, vocab]");

  Tensor input =
      logits.dtype() == DType::Float32 ? logits : logits.to(DType::Float32);
  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  if (cols == 0) {
    TP_THROW(RuntimeError, "sample: vocab size must be > 0");
  }

  // Sort the raw logits directly: ordering is monotonic under softmax,
  // so the sampler shader can apply temperature scaling, softmax, and the
  // top-k / top-p cutoffs on the sorted row without a full-vocabulary
  // softmax pass.  When top-k is requested the selection kernel sorts only
  // the k best entries (much cheaper than a full vocabulary sort).
  int64_t sorted_len = cols;
  std::tuple<Tensor, Tensor> sorted;
  if (top_k > 0 && top_k < cols) {
    sorted = topk_kernel(input, top_k, -1, true, true, 0);
    sorted_len = top_k;
  } else {
    sorted = sort_kernel(input, -1, true);
  }
  const Tensor& values = std::get<0>(sorted);
  const Tensor& indices = std::get<1>(sorted);

  api::Context* const context = api::context();
  api::vTensor v_result{
      context, {rows}, DType::Int64, api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  if (rows == 0) {
    return convert(v_result);
  }

  api::vTensor v_values = convert(values);
  api::vTensor v_indices = convert(indices);

  // The sampler reads the sorted row through images for the top-k prefix;
  // a full-vocabulary sort of a buffer-stored row reads linear buffers
  // instead.  The noise row follows the same storage choice so the draw
  // always lands in the selected payload layout.
  const bool buffer_sorted =
      v_values.storage_type() == api::StorageType::BUFFER;
  api::vTensor v_noise =
      buffer_sorted
          ? api::vTensor{
                context, {rows}, DType::Float32, api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED}
          : api::vTensor{context, {rows}, DType::Float32};
  Tensor noise = convert(v_noise);
  std::optional<Generator> no_generator;
  dispatch_random_fill(
      noise, VK_KERNEL(uniform_fill), VK_KERNEL(buffer_uniform_fill), 0.0f,
      1.0f, 0.0f, no_generator);
  v_noise = convert(noise);

  // With the top-k selection kernel the row is already truncated, so the
  // shader receives top_k=0 and works on every sorted entry; with a full
  // sort it applies the cutoff itself.
  const int64_t shader_top_k = (top_k > 0 && top_k < cols) ? 0 : top_k;
  const struct SampleBlock final {
    int k;
    int rows;
    int top_k;
    float top_p;
    float temperature;
  } block{
      static_cast<int32_t>(sorted_len),
      static_cast<int32_t>(rows),
      static_cast<int32_t>(shader_top_k),
      static_cast<float>(top_p),
      static_cast<float>(temperature)};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier barrier{};
  const api::utils::uvec3 grid{
      api::utils::safe_downcast_to_u32(rows), 1u, 1u};

  if (buffer_sorted) {
    context->submit_compute_job(
        VK_KERNEL(sample_from_sorted_buffer), barrier, grid, {1u, 1u, 1u},
        VK_NULL_HANDLE,
        v_result.buffer(
            barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
        v_values.buffer(barrier, api::PipelineStage::COMPUTE,
                        api::MemoryAccessType::READ),
        v_indices.buffer(barrier, api::PipelineStage::COMPUTE,
                         api::MemoryAccessType::READ),
        v_noise.buffer(barrier, api::PipelineStage::COMPUTE,
                       api::MemoryAccessType::READ),
        params.buffer());
    return convert(v_result);
  }

  context->submit_compute_job(
      VK_KERNEL(sample_from_sorted), barrier, grid, {1u, 1u, 1u},
      VK_NULL_HANDLE,
      v_result.buffer(
          barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE),
      v_values.image(barrier, api::PipelineStage::COMPUTE),
      v_indices.image(barrier, api::PipelineStage::COMPUTE),
      v_noise.image(barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_result);
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, RandomKernels) {
  m.impl("uniform_", &tensorplay::vulkan::ops::uniform_kernel);
  m.impl("normal_", &tensorplay::vulkan::ops::normal_kernel);
  m.impl("bernoulli_.float", &tensorplay::vulkan::ops::bernoulli_scalar_kernel);
  m.impl("bernoulli_.Tensor", &tensorplay::vulkan::ops::bernoulli_tensor_kernel);
  m.impl("randn", &tensorplay::vulkan::ops::randn_kernel);
  m.impl("rand", &tensorplay::vulkan::ops::rand_kernel);
  m.impl("rand_like", &tensorplay::vulkan::ops::rand_like_kernel);
  m.impl("randn_like", &tensorplay::vulkan::ops::randn_like_kernel);
  m.impl("multinomial", &tensorplay::vulkan::ops::multinomial_kernel);
  m.impl("sample", &tensorplay::vulkan::ops::sample_kernel);
}

#endif /* USE_VULKAN */
