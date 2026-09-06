#ifdef USE_VULKAN

#include "Factory.h"
#include "Convert.h"
#include "Utils.h"
#include "../impl/Common.h"
#include "../api/ShaderRegistry.h"

#include <functional>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

/*
 * Fills every element with the requested value.  Float32 payloads run the
 * zero-plus-add shader pair; every other dtype is materialized on the CPU in
 * the payload's element type and streamed through the staging pipeline, which
 * is valid for every supported VkFormat.
 */
Tensor& fill_impl(Tensor& self, Scalar value) {
  TP_CHECK(
      self.dim() <= 4, "Vulkan fill_ supports up to 4d tensors");

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);

  if (v_self.storage_type() == api::StorageType::BUFFER &&
      self.dtype() == DType::Float32) {
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v_self.numel()));
    api::vTensor v = v_self;
    // Zero then add the scalar, reusing the buffer kernels.
    {
      const struct BlockZ final {
        uint32_t buf_length;
      } blockz{n};
      api::UniformParamsBuffer params(context, blockz);
      api::PipelineBarrier pipeline_barrier{};
      context->submit_compute_job(
          VK_KERNEL(buffer_zero), pipeline_barrier, {n, 1u, 1u}, {64u, 1u, 1u},
          VK_NULL_HANDLE,
          v.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                   api::MemoryAccessType::WRITE),
          params.buffer());
    }
    const struct BlockF final {
      uint32_t buf_length;
      uint32_t fill0;
      float other;
    } blockf{n, 0u, value.to<float>()};
    api::UniformParamsBuffer params(context, blockf);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL(buffer_add_scalarinplace), pipeline_barrier, {n, 1u, 1u},
        {64u, 1u, 1u}, VK_NULL_HANDLE,
        v.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                 api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        params.buffer());
    return self;
  }

  if (v_self.storage_type() == api::StorageType::TEXTURE_3D &&
      self.dtype() == DType::Float32) {
    const struct Block final {
      ivec4 extents;
      float other;
    } block{
        make_whcn_ivec4(v_self.gpu_sizes()),
        value.to<float>(),
    };

    // Newly allocated payloads hold arbitrary bits: clear every texel to a
    // known zero before the in-place add reads it back.
    {
      api::PipelineBarrier pipeline_barrier{};

      context->submit_compute_job(
          // shader descriptor
          VK_KERNEL(zero),
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
              api::MemoryAccessType::WRITE));
    }

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(add_scalarinplace),
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

    return self;
  }

  // Non-float payloads: materialize the value on the CPU in the texture's
  // element type, repack to the texel-linear layout, and transfer.
  const DType texture_dtype =
      (v_self.storage_type() == api::StorageType::TEXTURE_3D)
      ? v_self.texture_dtype()
      : self.dtype();
  Tensor host(
      static_cast<std::vector<int64_t>>(self.shape()),
      texture_dtype,
      Device(DeviceType::CPU));
  host.fill_(value);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    utils::upload_host_bytes(
        v_self,
        host.impl()->storage().data(),
        host.numel() * host.itemsize());
    return self;
  }

  Tensor host_nc4hw = utils::nchw_to_nc4hw(host.contiguous());
  utils::upload_host_bytes(
      v_self,
      host_nc4hw.impl()->storage().data(),
      host_nc4hw.numel() * host_nc4hw.itemsize());
  return self;
}

} // namespace

Tensor empty_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  TP_CHECK(
      !pin_memory, "pin_memory is only valid for CPU tensors");
  const DType resolved =
      (dtype == DType::Undefined) ? DType::Float32 : dtype;
  api::vTensor v{
      api::context(),
      size,
      convert_dtype(resolved),
  };
  return convert(v);
}

Tensor zeros_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  if (t.numel() == 0) {
    return t;
  }
  api::Context* const context = api::context();
  api::vTensor v = convert(t);

  if (v.storage_type() == api::StorageType::TEXTURE_3D &&
      t.dtype() == DType::Float32) {
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(zero),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        v.extents(),
        // local work group size
        adaptive_work_group_size(v.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE));
    return t;
  }

  if (v.storage_type() == api::StorageType::BUFFER &&
      tensorplay::elementSize(t.dtype()) == 4u) {
    // The zero shader writes single-precision words; the all-zero bit
    // pattern is also a zero element for 4-byte integer payloads.
    const uint32_t n =
        safe_downcast_to_u32(static_cast<int64_t>(v.numel()));
    const struct BlockZ final {
      uint32_t buf_length;
    } blockz{n};
    api::UniformParamsBuffer params(context, blockz);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        VK_KERNEL(buffer_zero), pipeline_barrier, {n, 1u, 1u}, {64u, 1u, 1u},
        VK_NULL_HANDLE,
        v.buffer(pipeline_barrier, api::PipelineStage::COMPUTE,
                 api::MemoryAccessType::WRITE),
        params.buffer());
    return t;
  }

  // All-zero bytes are valid zeros for every dtype; stream the payload
  // through the staging pipeline without format-specific shaders.
  std::vector<uint8_t> host(v.gpu_nbytes(), 0);
  utils::upload_host_bytes(v, host.data(), host.size());
  return t;
}

Tensor ones_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  fill_impl(t, Scalar(1.0));
  return t;
}

Tensor full_kernel(
    const std::vector<int64_t>& size,
    Scalar fill_value,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  fill_impl(t, fill_value);
  return t;
}

/*
 * Host-computed factories: the values are materialized on the CPU with the
 * same formulas the CPU factory kernels apply, then streamed into the
 * payload through the staging pipeline, which covers every VkFormat.  The
 * staging buffer carries the texture's element type, so the writer is
 * instantiated per element width the backend supports.
 */
template <typename T>
void scatter_staging_1d(
    api::vTensor& v,
    int64_t steps,
    const std::function<double(int64_t)>& fill) {
  Tensor host = utils::create_staging_tensor(v);
  T* data = static_cast<T*>(host.impl()->storage().data());
  for (int64_t i = 0; i < steps; ++i) {
    data[i * 4] = static_cast<T>(fill(i));
  }
  utils::upload_host_bytes(
      v, host.impl()->storage().data(), host.numel() * host.itemsize());
}

void scatter_staging_values_1d(
    api::vTensor& v,
    int64_t steps,
    const std::function<double(int64_t)>& fill) {
  switch (v.texture_dtype()) {
    case DType::Float32:
      scatter_staging_1d<float>(v, steps, fill);
      return;
    case DType::Float16:
      scatter_staging_1d<tensorplay::Half>(v, steps, fill);
      return;
    case DType::Int32:
      scatter_staging_1d<int32_t>(v, steps, fill);
      return;
    case DType::Int8:
    case DType::Bool:
      scatter_staging_1d<int8_t>(v, steps, fill);
      return;
    case DType::UInt8:
      scatter_staging_1d<uint8_t>(v, steps, fill);
      return;
    default:
      TP_THROW(
          NotImplementedError,
          "Vulkan host-filled factory: unsupported texture dtype");
  }
}

Device resolve_device(std::optional<Device> device) {
  return device.value_or(Device(DeviceType::Vulkan));
}

template <typename Filler>
Tensor host_filled_1d(int64_t steps, DType dtype, Device device, Filler fill) {
  Tensor t = zeros_kernel({steps}, dtype, device, false);
  if (steps == 0) {
    return t;
  }
  api::vTensor v = convert(t);
  TP_CHECK(
      v.storage_type() == api::StorageType::TEXTURE_3D,
      "Vulkan host-filled factories require texture storage");
  scatter_staging_values_1d(
      v, steps, [fill](int64_t i) { return static_cast<double>(fill(i)); });
  return t;
}

Tensor eye_kernel(
    int64_t n,
    int64_t m,
    DType dtype,
    std::optional<Device> device) {
  if (m < 0) m = n;
  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16 ||
          dtype == DType::Int32,
      "Vulkan eye supports Float32, Float16 and Int32 only");
  Tensor t = empty_kernel({n, m}, dtype, resolve_device(device), false);
  if (t.numel() == 0) return t;
  api::vTensor v = convert(t);
  api::Context* context = api::context();
  const char* shader = dtype == DType::Int32 ? "eye_i32" :
      dtype == DType::Float16 ? "eye_f16" : "eye";
  api::PipelineBarrier barrier{};
  context->submit_compute_job(
      VK_KERNEL_FROM_STR(shader), barrier, v.extents(),
      adaptive_work_group_size(v.extents()), VK_NULL_HANDLE,
      v.image(barrier, api::PipelineStage::COMPUTE, api::MemoryAccessType::WRITE));
  return t;
}

Tensor linspace_kernel(
    Scalar start,
    Scalar end,
    int64_t steps,
    DType dtype,
    std::optional<Device> device) {
  TP_CHECK(steps >= 0, "number of steps must be non-negative");
  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16 ||
          dtype == DType::Int32,
      "Vulkan linspace supports Float32, Float16 and Int32 only");
  const double s = start.toDouble();
  const double e = end.toDouble();
  const double step = steps > 1 ? (e - s) / (steps - 1) : 0.0;
  return host_filled_1d(
      steps, dtype, resolve_device(device),
      [&](int64_t i) { return s + i * step; });
}

Tensor logspace_kernel(
    Scalar start,
    Scalar end,
    int64_t steps,
    double base,
    DType dtype,
    std::optional<Device> device) {
  TP_CHECK(steps >= 0, "number of steps must be non-negative");
  TP_CHECK(
      dtype == DType::Float32 || dtype == DType::Float16 ||
          dtype == DType::Int32,
      "Vulkan logspace supports Float32, Float16 and Int32 only");
  const double s = start.toDouble();
  const double e = end.toDouble();
  const double step = steps > 1 ? (e - s) / (steps - 1) : 0.0;
  return host_filled_1d(
      steps, dtype, resolve_device(device),
      [&](int64_t i) { return std::pow(base, s + i * step); });
}

Tensor empty_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device) {
  Device dev = device.value_or(self.device());
  return empty_kernel(
      static_cast<std::vector<int64_t>>(self.shape()),
      dtype == DType::Undefined ? self.dtype() : dtype,
      dev,
      false);
}

Tensor zeros_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device) {
  DType dt = dtype == DType::Undefined ? self.dtype() : dtype;
  return zeros_kernel(
      static_cast<std::vector<int64_t>>(self.shape()), dt, self.device(), false);
}

Tensor ones_like_kernel(
    const Tensor& self,
    DType dtype,
    std::optional<Device> device) {
  DType dt = dtype == DType::Undefined ? self.dtype() : dtype;
  return ones_kernel(
      static_cast<std::vector<int64_t>>(self.shape()), dt, self.device(), false);
}

Tensor full_like_kernel(
    const Tensor& self,
    Scalar fill_value,
    DType dtype,
    std::optional<Device> device) {
  DType dt = dtype == DType::Undefined ? self.dtype() : dtype;
  return full_kernel(
      static_cast<std::vector<int64_t>>(self.shape()),
      fill_value,
      dt,
      self.device(),
      false);
}

Tensor& fill_kernel(Tensor& self, Scalar value) {
  return fill_impl(self, value);
}

namespace {

DType resolve_dtype(std::optional<DType> dtype) {
  if (!dtype.has_value() || *dtype == DType::Undefined) {
    return DType::Float32;
  }
  return *dtype;
}

// Schema-level adapters: the dispatcher invokes kernels with the optional
// argument types of the schema, while the kernels above keep concrete
// parameters for internal reuse.
Tensor empty_stub(
    const std::vector<int64_t>& size,
    std::optional<DType> dtype,
    std::optional<Device> device,
    bool pin_memory) {
  return empty_kernel(size, resolve_dtype(dtype), resolve_device(device), pin_memory);
}

Tensor zeros_stub(
    const std::vector<int64_t>& size,
    std::optional<DType> dtype,
    std::optional<Device> device,
    bool pin_memory) {
  return zeros_kernel(size, resolve_dtype(dtype), resolve_device(device), pin_memory);
}

Tensor ones_stub(
    const std::vector<int64_t>& size,
    std::optional<DType> dtype,
    std::optional<Device> device,
    bool pin_memory) {
  return ones_kernel(size, resolve_dtype(dtype), resolve_device(device), pin_memory);
}

Tensor full_stub(
    const std::vector<int64_t>& size,
    Scalar fill_value,
    DType dtype,
    std::optional<Device> device,
    bool pin_memory) {
  return full_kernel(size, fill_value, dtype, resolve_device(device), pin_memory);
}

} // namespace

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, FactoryKernels) {
  m.impl("empty", &tensorplay::vulkan::ops::empty_stub);
  m.impl("zeros", &tensorplay::vulkan::ops::zeros_stub);
  m.impl("ones", &tensorplay::vulkan::ops::ones_stub);
  m.impl("full", &tensorplay::vulkan::ops::full_stub);
  m.impl("empty_like", &tensorplay::vulkan::ops::empty_like_kernel);
  m.impl("zeros_like", &tensorplay::vulkan::ops::zeros_like_kernel);
  m.impl("ones_like", &tensorplay::vulkan::ops::ones_like_kernel);
  m.impl("full_like", &tensorplay::vulkan::ops::full_like_kernel);
  m.impl("fill_.Scalar", &tensorplay::vulkan::ops::fill_kernel);
  m.impl("eye", &tensorplay::vulkan::ops::eye_kernel);
  m.impl("linspace", &tensorplay::vulkan::ops::linspace_kernel);
  m.impl("logspace", &tensorplay::vulkan::ops::logspace_kernel);
}

#endif /* USE_VULKAN */
