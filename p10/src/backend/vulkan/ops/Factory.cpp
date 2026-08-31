#ifdef USE_VULKAN

#include "Factory.h"
#include "Convert.h"
#include "Utils.h"
#include "../impl/Common.h"
#include "../api/ShaderRegistry.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

/*
 * Uploads host bytes into the texture (or buffer) payload through the
 * staging pipeline: staging buffer -> pack shader / copy command.
 */
void upload_host_bytes(api::vTensor& v_dst, const void* bytes, size_t nbytes) {
  api::Context* const context = api::context();

  api::StorageBuffer staging(
      context, v_dst.texture_dtype(), v_dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    memcpy(mapping.data<void>(), bytes, nbytes);
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_dst);
}

/*
 * Fills every element with the requested value.  The zero shader writes
 * vec4(0) for every texel, an in-place scalar add then applies the value.
 */
Tensor& fill_impl(Tensor& self, Scalar value) {
  TP_CHECK(
      self.dtype() == DType::Float32,
      "Vulkan fill_ supports Float32 tensors only");

  api::Context* const context = api::context();

  api::vTensor v_self = convert(self);

  if (v_self.storage_type() == api::StorageType::BUFFER) {
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

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
    float other;
  } block{
      v_self.extents(),
      0u,
      value.to<float>(),
  };

  // Zero every texel.
  {
    const struct Block0 final {
      uvec3 extents;
      uint32_t fill0;
    } block0{
        v_self.extents(),
        0u,
    };

    api::UniformParamsBuffer params(context, block0);
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
            api::MemoryAccessType::WRITE),
        // params buffer
        params.buffer());
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

} // namespace

Tensor empty_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  if (pin_memory) {
    TP_THROW(RuntimeError, "pin_memory is only valid for CPU tensors");
  }
  api::vTensor v{
      api::context(),
      size,
      dtype,
  };
  return convert(v);
}

Tensor zeros_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  if (dtype == DType::Float32) {
    api::vTensor v = convert(t);
    api::Context* const context = api::context();

    if (v.storage_type() == api::StorageType::BUFFER) {
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

    const struct Block final {
      uvec3 extents;
      uint32_t fill0;
    } block{
        v.extents(),
        0u,
    };

    api::UniformParamsBuffer params(context, block);
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
            api::MemoryAccessType::WRITE),
        // params buffer
        params.buffer());
    return t;
  }
  // Non-float payloads are staged from the CPU: all-zero bytes are valid
  // for every dtype.
  const size_t nbytes = t.numel() * tensorplay::elementSize(dtype);
  std::vector<uint8_t> host(nbytes, 0);
  api::vTensor v = convert(t);
  upload_host_bytes(v, host.data(), nbytes);
  return t;
}

Tensor ones_kernel(
    const std::vector<int64_t>& size,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  if (dtype == DType::Float32) {
    return fill_impl(t, Scalar(1.0)), t;
  }
  TP_THROW(
      NotImplementedError,
      "Vulkan ones supports Float32 tensors only");
}

Tensor full_kernel(
    const std::vector<int64_t>& size,
    Scalar fill_value,
    DType dtype,
    Device device,
    bool pin_memory) {
  Tensor t = empty_kernel(size, dtype, device, pin_memory);
  if (dtype == DType::Float32) {
    return fill_impl(t, fill_value), t;
  }
  TP_THROW(
      NotImplementedError,
      "Vulkan full supports Float32 tensors only");
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

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, FactoryKernels) {
  m.impl("empty", &tensorplay::vulkan::ops::empty_kernel);
  m.impl("zeros", &tensorplay::vulkan::ops::zeros_kernel);
  m.impl("ones", &tensorplay::vulkan::ops::ones_kernel);
  m.impl("full", &tensorplay::vulkan::ops::full_kernel);
  m.impl("empty_like", &tensorplay::vulkan::ops::empty_like_kernel);
  m.impl("zeros_like", &tensorplay::vulkan::ops::zeros_like_kernel);
  m.impl("ones_like", &tensorplay::vulkan::ops::ones_like_kernel);
  m.impl("full_like", &tensorplay::vulkan::ops::full_like_kernel);
  m.impl("fill_.Scalar", &tensorplay::vulkan::ops::fill_kernel);
}

#endif /* USE_VULKAN */
