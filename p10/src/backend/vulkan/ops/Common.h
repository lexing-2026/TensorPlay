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
 * invocation, the Float16 ones a single word holding two lanes.  Rounding up
 * to a whole element gives the last invocation an index that the kernels use
 * to own the leftover lanes, so coverage never depends on how far the launch
 * is padded to whole workgroups.
 */
inline uint32_t buffer_dispatch_count(
    uint32_t elements,
    DType dtype,
    uint32_t float_elements_per_invocation) {
  // A code wider than a word owns a whole pair of them, so one invocation
  // carries exactly one element of it.
  const uint32_t per = dtype == DType::Int64
      ? 1u
      : (dtype == DType::Float16 ? 2u : float_elements_per_invocation);
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

//
// Whether a channel-packed 3-D image exists for these sizes on the current
// device.  The texel layout carries the two innermost axes as width and
// height and folds the channel and batch axes into the depth, four channels
// per texel lane; every axis obeys the device's image limit, so a shape with
// one long axis has no image at all.  A kernel that wrote its result into a
// linear payload consults this before releasing the payload, because which
// storage a shape gets must not depend on which route produced it.
//
inline bool image_storage_supported(
    api::Context* const context,
    const std::vector<int64_t>& sizes) {
  if (sizes.empty() || sizes.size() > 4 ||
      api::utils::multiply_integers(sizes) == 0u) {
    return false;
  }
  const uint32_t limit =
      context->adapter_ptr()->properties().limits.maxImageDimension3D;
  const uint32_t width = get_dim<1>(sizes);
  const uint32_t height = get_dim<2>(sizes);
  const uint64_t depth = static_cast<uint64_t>(div_up(get_dim<3>(sizes), 4u)) *
      static_cast<uint64_t>(get_dim<4>(sizes));
  return width <= limit && height <= limit && depth <= limit;
}

//
// The storage a result has to live in.  Kernel variants are written per
// storage class: an image payload is read by sampling texels, a dense one by
// indexing 32-bit words, and no variant crosses that line.  Releasing an
// allocation picks an image whenever the shape fits one, so pairing it with a
// dense operand of the same shape selects the texel variant, which then fails
// on the operand's missing image -- or, when the kernel skips that check,
// leaves the image holding whatever the allocator recycled into it while the
// real result sat in a buffer no read-back samples.  Naming the class the
// operands use keeps the write in the storage the read uses.
//
inline api::vTensor allocate_output(
    api::Context* const context,
    api::StorageType storage_type,
    const std::vector<int64_t>& sizes,
    DType dtype) {
  if (storage_type == api::StorageType::BUFFER) {
    return api::vTensor{
        context,
        sizes,
        dtype,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  }
  return api::vTensor{context, sizes, dtype};
}

inline api::vTensor allocate_output_like(
    api::Context* const context,
    const api::vTensor& operand,
    const std::vector<int64_t>& sizes,
    DType dtype) {
  return allocate_output(
      context, operand.storage_type(), sizes, dtype);
}

// Elementwise linear-buffer kernels address payload in the slots of a 16-byte
// word: a Float32 element fills one, a Float16 pair shares one, and a code
// wider than a word owns a pair of them.  Byte-packed and Bool payloads have no
// such shader yet; rejecting them here keeps a missing buffer variant visible
// instead of silently reinterpreting 1-byte (or 2-byte) elements as Float32
// words.
inline void validate_buffer_elementwise_dtype(
    const api::vTensor& v,
    const char* name) {
  if (v.storage_type() == api::StorageType::BUFFER &&
      v.dtype() != DType::Float32 && v.dtype() != DType::Float16 &&
      v.dtype() != DType::Int32 && v.dtype() != DType::Int64) {
    TP_THROW(
        NotImplementedError,
        std::string("Vulkan ") + name +
            " over buffer storage supports Float32, Float16, Int32 and Int64 "
            "tensors only");
  }
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
