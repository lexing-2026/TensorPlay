#pragma once

#ifdef USE_VULKAN

#include "vk_api.h"
#include "Exception.h"

#include "DType.h"

#ifndef VK_FORMAT_FLOAT4
#define VK_FORMAT_FLOAT4 VK_FORMAT_R32G32B32A32_SFLOAT
#endif

namespace tensorplay {
namespace vulkan {
namespace api {

//
// Maps a core DType onto the VkFormat used when that dtype is viewed through
// an image view.  The covered set is the backend's dtype vocabulary: element
// widths of 1, 2, and 4 bytes with matching signed/unsigned formats.  Buffer
// storage does not consume these formats, but keeping the mapping here
// centralizes the dtype policy for the backend.
//
inline VkFormat to_vkformat(const DType dtype) {
  // Quantized dtypes share the VkFormat of their underlying integer code.
  switch (toUnderlyingStorageType(dtype)) {
    case DType::UInt8:
      return VK_FORMAT_R8G8B8A8_UINT;
    case DType::Int8:
    case DType::Bool:
      return VK_FORMAT_R8G8B8A8_SINT;
    case DType::Int32:
      return VK_FORMAT_R32G32B32A32_SINT;
    case DType::Float16:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DType::Float32:
      return VK_FORMAT_FLOAT4;
    default:
      VK_CHECK_COND(false, "DType not supported by the Vulkan backend!");
  }
  return VK_FORMAT_UNDEFINED;
}

/*
 * Given a VkFormat, return the DType of the individual elements in an image
 * texture of that format.  Note this mapping differs from to_vkformat():
 * several dtypes share one VkFormat (texture traffic is always vec4-packed).
 */
inline DType element_scalartype(const VkFormat vkformat) {
  switch (vkformat) {
    case VK_FORMAT_R8G8B8A8_SINT:
      return DType::Int8;
    case VK_FORMAT_R8G8B8A8_UINT:
      return DType::UInt8;
    case VK_FORMAT_R32G32B32A32_SINT:
      return DType::Int32;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return DType::Float32;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return DType::Float16;
    default:
      VK_CHECK_COND(
          false, "No corresponding scalar type for unknown VkFormat!");
  }
  return DType::Undefined;
}

//
// Where a tensor's payload physically lives.  TEXTURE_3D is the default
// (channel-packed rgba images, matching the image shaders); BUFFER keeps a
// linear SSBO for ops that prefer addressable memory.
//
enum class StorageType : int8_t {
  UNKNOWN = 0,
  BUFFER,
  TEXTURE_3D,
};

//
// Which logical dimension is packed into vec4 texels / buffer words.
//
enum class GPUMemoryLayout : int8_t {
  TENSOR_WIDTH_PACKED = 0,
  TENSOR_HEIGHT_PACKED,
  TENSOR_CHANNELS_PACKED,
};

//
// Image format selectors consumed by the GLSL ${FORMAT} substitution.  The
// FLOAT_* values are fixed at build time by gen_vulkan_spv.py via
// FLOAT_IMAGE_FORMAT (rgba32f unless USE_VULKAN_FP16_INFERENCE).
//
} // namespace api
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
