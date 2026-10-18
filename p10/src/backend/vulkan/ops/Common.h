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
 * Linear payloads carry their element type in the kernel name: a storage
 * buffer has no texel format to select, so the fp16 builds are a separate
 * family that reads two half lanes per 32-bit word.  The tag goes before the
 * in-place suffix, which the op registrations already spell into the name.
 */
inline std::string buffer_shader_for_dtype(
    const std::string& name,
    DType dtype) {
  if (dtype != DType::Float16) {
    return name;
  }
  constexpr std::string_view in_place_suffix = "inplace";
  if (name.size() >= in_place_suffix.size() &&
      name.compare(
          name.size() - in_place_suffix.size(),
          in_place_suffix.size(),
          in_place_suffix) == 0) {
    return name.substr(0, name.size() - in_place_suffix.size()) +
        "_halfinplace";
  }
  return name + "_half";
}

/*
 * Global invocation count for an elementwise linear dispatch.  The Float32
 * builds were authored to move a 16-byte vector (four elements) per
 * invocation, the Float16 ones a single word holding two lanes.  Dispatch
 * rounds up to whole workgroups, which is what covers a tail element whose
 * owning index falls past the exact quotient.
 */
inline uint32_t buffer_dispatch_count(
    uint32_t elements,
    DType dtype,
    uint32_t float_elements_per_invocation) {
  const uint32_t per =
      dtype == DType::Float16 ? 2u : float_elements_per_invocation;
  return (elements + per - 1u) / per;
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

// Elementwise linear-buffer kernels address whole 4-byte words per lane: a
// Float32 element fills one word, a Float16 pair shares one.  Byte-packed and
// Bool payloads have no such shader yet; rejecting them here keeps a missing
// buffer variant visible instead of silently reinterpreting 1-byte (or 2-byte)
// elements as Float32 words.
inline void validate_buffer_elementwise_dtype(
    const api::vTensor& v,
    const char* name) {
  if (v.storage_type() == api::StorageType::BUFFER &&
      v.dtype() != DType::Float32 && v.dtype() != DType::Float16 &&
      v.dtype() != DType::Int32) {
    TP_THROW(
        NotImplementedError,
        std::string("Vulkan ") + name +
            " over buffer storage supports Float32, Float16 and Int32 tensors "
            "only");
  }
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
