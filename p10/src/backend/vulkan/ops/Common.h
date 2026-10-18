#pragma once

#ifdef USE_VULKAN

#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../api/Tensor.h"
#include "../api/Types.h"
#include "../api/Utils.h"

#include "Tensor.h"


namespace tensorplay {
namespace vulkan {
namespace ops {

using namespace api::utils;

/*
 * dtype-aware shader resolution: float payloads dispatch to the storage
 * format variant matching their texture (rgba16f twins for Float16), and
 * every other dtype keeps the base build.  Callers pass the payload
 * tensor instead of hardcoding the shader name.
 */
inline const api::ShaderInfo& kernel_for(const char* base_name, DType dtype) {
  return api::get_shader_info_for_dtype(base_name, dtype);
}

inline const api::ShaderInfo& kernel_for(const char* base_name, const Tensor& t) {
  return api::get_shader_info_for_dtype(base_name, t.dtype());
}

/*
 * The functions below safely return the size of the dimension at the N-th
 * innermost index. If the dimensionality of the size array is not sufficient
 * then 1 will be returned.
 */
template <uint32_t N>
uint32_t get_dim(const IntArrayRef sizes) {
  const uint32_t dims = static_cast<uint32_t>(sizes.size());
  return dims < N ? 1u : safe_downcast_to_u32(sizes[dims - N]);
}

template <uint32_t N>
uint32_t get_dim(const Size& sizes) {
  const uint32_t dims = static_cast<uint32_t>(sizes.size());
  return dims < N ? 1u : safe_downcast_to_u32(sizes[dims - N]);
}

template <uint32_t N>
uint32_t get_dim(const std::vector<int64_t>& sizes) {
  const uint32_t dims = static_cast<uint32_t>(sizes.size());
  return dims < N ? 1u : safe_downcast_to_u32(sizes[dims - N]);
}

template <uint32_t N>
uint32_t get_dim(const Tensor& t_in) {
  return get_dim<N>(t_in.shape());
}

template <uint32_t N>
uint32_t get_dim(const api::vTensor& v_in) {
  return get_dim<N>(v_in.sizes());
}

// Dimension positions for a 4D {N, C, H, W} tensor.
struct Dim4D final {
  static constexpr uint32_t Batch = 4u;
  static constexpr uint32_t Channel = 3u;
  static constexpr uint32_t Height = 2u;
  static constexpr uint32_t Width = 1u;
};

// Elementwise linear-buffer kernels read and write whole 4-byte words per
// lane.  Half-precision, integer-byte and Bool payloads have no such shader
// yet; rejecting them here keeps a missing buffer variant visible instead of
// silently reinterpreting 2-byte (or 1-byte) elements as Float32 words.
inline void validate_buffer_elementwise_dtype(
    const api::vTensor& v,
    const char* name) {
  if (v.storage_type() == api::StorageType::BUFFER &&
      v.dtype() != DType::Float32 && v.dtype() != DType::Int32) {
    TP_THROW(
        NotImplementedError,
        std::string("Vulkan ") + name +
            " over buffer storage supports Float32 and Int32 tensors only");
  }
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
