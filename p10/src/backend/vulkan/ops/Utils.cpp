#ifdef USE_VULKAN

#include "Utils.h"
#include "../impl/Common.h"
#include "../impl/Packing.h"

namespace tensorplay {
namespace vulkan {
namespace ops {
namespace utils {

using namespace api::utils;

/*
 * This function formats an input tensor in NCHW layout to NC4HW layout such
 * that the buffer of the formatted tensor can be directly copied into a GPU
 * texture. Conceptually, the formatting can be achieved via the following
 * steps:
 *
 * 1. Given that the src tensor has size {N,C,H,W}
 *
 * 2. Combine the batch and channel dims by reshaping to {N*C, H, W}
 *
 * 3. Determine the amount of padding to add: determine how many channels to
 *    add in order to align N*C to the next multiple of 4
 *
 * 4. Add padding to the tensor so that the batch-channel dimension is a
 *    multiple of four; the shape of the tensor is now {NC_aligned, H, W}
 *
 * 5. Split the batch-channel dimension into groups of 4 by reshaping the
 *    tensor to size {NC_aligned/4, 4, H, W}
 *
 * 6. The groups of 4 channels (dim 1) should be contiguous. Therefore,
 *    permute the dims of the tensor in the order {0, 2, 3, 1}
 *
 * 7. Finally, return a contiguous version of the tensor. The final shape of
 *    the tensor would be {NC_aligned/4, H, W, 4}
 */
Tensor nchw_to_nc4hw(const Tensor& src) {
  const size_t itemsize = src.itemsize();

  const int64_t N = get_dim<Dim4D::Batch>(src.shape());
  const int64_t C = get_dim<Dim4D::Channel>(src.shape());
  const int64_t H = get_dim<Dim4D::Height>(src.shape());
  const int64_t W = get_dim<Dim4D::Width>(src.shape());

  const int64_t C_aligned = api::utils::align_up(C, 4u);
  const int64_t NC4 = (N * C_aligned) / 4;

  Tensor out(
      {NC4, H, W, 4}, src.dtype(), Device(DeviceType::CPU));
  uint8_t* out_ptr = static_cast<uint8_t*>(out.impl()->storage().data());
  std::memset(out_ptr, 0, out.numel() * itemsize);

  const uint8_t* in_ptr =
      static_cast<const uint8_t*>(src.impl()->storage().data());
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t z = n * (C_aligned / 4) + c / 4;
      const int64_t c_idx = c % 4;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          std::memcpy(
              out_ptr + (((z * H + h) * W + w) * 4 + c_idx) * itemsize,
              in_ptr + (((n * C + c) * H + h) * W + w) * itemsize,
              itemsize);
        }
      }
    }
  }

  return out;
}

/*
 * Creates a staging tensor into which texture data, which will be in NC4HW
 * format, can be copied directly. The shape of the staging tensor will be
 * the same as the tensor produced by a call to nchw_to_nc4hw().
 */
Tensor create_staging_tensor(const api::vTensor& v_in) {
  uint32_t N = get_dim<Dim4D::Batch>(v_in.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(v_in.sizes());
  uint32_t H = get_dim<Dim4D::Height>(v_in.sizes());
  uint32_t W = get_dim<Dim4D::Width>(v_in.sizes());

  uint32_t NC4 = N * api::utils::div_up(C, 4u);

  // Note that the dtype corresponding with the texture format of the
  // vTensor is used instead of options().dtype(). This is to ensure the
  // number of bytes in the staging tensor matches the number of bytes in
  // the image texture.
  return Tensor(
      {static_cast<int64_t>(NC4), static_cast<int64_t>(H),
       static_cast<int64_t>(W), 4},
      v_in.texture_dtype(),
      Device(DeviceType::CPU));
}

/*
 * After copying texture data, which will be in NC4HW format, to a staging
 * tensor created in create_staging_tensor(), this function reformats the
 * tensor to NCHW format. It essentially reverses the transformations made
 * by nchw_to_nc4hw().
 *
 * Note that the sizes of the original tensor must be passed in to fully
 * restore the properties of the original tensor.
 */
Tensor nc4hw_to_nchw(const Tensor& t_in, IntArrayRef sizes) {
  const size_t itemsize = t_in.itemsize();

  const int64_t N = get_dim<Dim4D::Batch>(sizes);
  const int64_t C = get_dim<Dim4D::Channel>(sizes);
  const int64_t H = get_dim<Dim4D::Height>(sizes);
  const int64_t W = get_dim<Dim4D::Width>(sizes);

  const int64_t C_aligned = api::utils::align_up(C, 4u);

  Tensor out({N, C, H, W}, t_in.dtype(), Device(DeviceType::CPU));
  uint8_t* out_ptr = static_cast<uint8_t*>(out.impl()->storage().data());
  const uint8_t* in_ptr =
      static_cast<const uint8_t*>(t_in.impl()->storage().data());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t z = n * (C_aligned / 4) + c / 4;
      const int64_t c_idx = c % 4;
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          std::memcpy(
              out_ptr + (((n * C + c) * H + h) * W + w) * itemsize,
              in_ptr + (((z * H + h) * W + w) * 4 + c_idx) * itemsize,
              itemsize);
        }
      }
    }
  }

  return out;
}

void copy_buffer_to_vtensor(
    api::VulkanBuffer& src_buffer,
    api::vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  TP_CHECK(
      src_buffer.mem_size() == v_dst.gpu_nbytes(),
      "Vulkan copy_buffer_to_vtensor: source buffer and destination texture "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanBuffer, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src_buffer,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      v_dst.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

void copy_buffer_to_buffer(
    api::Context* context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle) {
  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src.buffer(),
      dst.buffer(),
      // copy details
      {static_cast<uint32_t>(src.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void copy_vtensor_to_buffer(
    api::vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    VkFence fence_handle) {
  api::Context* const context = api::context();

  TP_CHECK(
      v_src.gpu_nbytes() == dst_buffer.mem_size(),
      "Vulkan copy_vtensor_to_buffer: source texture and destination buffer "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanImage, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      dst_buffer,
      // copy details
      v_src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer& buffer,
    api::vTensor& v_self,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    packing::record_nchw_to_buffer_op(
        context, buffer, v_self, pipeline_barrier, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_nchw_to_image_shader(v_self);
    packing::record_nchw_to_image_op(
        context,
        compute_shader,
        buffer,
        v_self,
        pipeline_barrier,
        VK_NULL_HANDLE);
  }
}

void pack_staging_to_vtensor(api::VulkanBuffer& staging, api::vTensor& v_self) {
  api::PipelineBarrier pipeline_barrier{};
  pack_buffer_to_vtensor(staging, v_self, pipeline_barrier);
}

bool pack_vtensor_to_staging(
    api::vTensor& v_self,
    api::VulkanBuffer& staging,
    VkFence fence_handle) {
  api::Context* const context = api::context();
  api::PipelineBarrier pipeline_barrier{};

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    return packing::record_buffer_to_nchw_op(
        context, v_self, staging, pipeline_barrier, fence_handle);
  } else {
    api::ShaderInfo compute_shader =
        packing::get_image_to_nchw_shader(v_self);
    return packing::record_image_to_nchw_op(
        context,
        compute_shader,
        v_self,
        staging,
        pipeline_barrier,
        fence_handle);
  }
}

void copy_staging_to_vtensor(
    api::StorageBuffer& staging,
    api::vTensor& v_dst) {
  api::Context* const context = api::context();

  if (v_dst.storage_type() == api::StorageType::BUFFER) {
    api::PipelineBarrier pipeline_barrier{};
    context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
        pipeline_barrier,
        staging.buffer(),
        v_dst.buffer(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        {static_cast<uint32_t>(staging.buffer().mem_size()), 0u, 0u},
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
  } else {
    api::PipelineBarrier pipeline_barrier{};
    copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
  }
}

void upload_host_bytes(
    api::vTensor& v_dst,
    const void* bytes,
    const size_t nbytes) {
  api::Context* const context = api::context();

  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy(mapping.data<void>(), bytes, nbytes);
  }

  copy_staging_to_vtensor(staging, v_dst);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
