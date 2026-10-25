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
// Whether a payload of this shape and element type actually lands in an image.
// An allocation that is not told otherwise keeps a matrix of the types that fill
// a texel's lanes in words, so a shape with an image form is not yet a payload
// with one; a kernel that is about to hand such a payload to a texel walk, or
// to relay one into an image, asks this instead and skips the crossing when the
// answer is no.
//
inline bool image_form_of(
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    DType dtype) {
  return image_storage_supported(context, sizes) &&
      api::StorageType::TEXTURE_3D ==
          api::storage_type_of_payload(sizes, dtype);
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
// A caller that names TEXTURE_3D is asking for a texel walk on the result, so
// the answer stays an image even for a shape that an unasked allocation would
// keep in words: the request is about the kernel, not about the shape.
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
  return api::vTensor{
      context,
      sizes,
      dtype,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
}

inline api::vTensor allocate_output_like(
    api::Context* const context,
    const api::vTensor& operand,
    const std::vector<int64_t>& sizes,
    DType dtype) {
  return allocate_output(
      context, operand.storage_type(), sizes, dtype);
}

//
// A two-dimensional operand that is a pure axis swap of its own dense storage
// keeps its bytes in the other orientation, so no payload describes the order
// the view names.  The twin names the same storage in the order it really sits
// in, which is what a binding expects, and it aliases rather than copies: a
// consumer can therefore read the swap straight out of the storage instead of
// relaying every element.  Anything else (a rank that is not two, a nonzero
// storage offset, strides that are not the plain swap) returns false.
//
bool transpose_view_of_dense(const Tensor& t, Tensor* dense_out);

//
// The payload an element-by-element reader can bind: the operand itself when
// its strides already agree with the storage, otherwise a dense tensor holding
// the viewed elements.  Products prefer the twin above; a read-back, a clone,
// or an epilogue that cannot reorder the axes needs the elements laid out the
// way the view names them.
//
Tensor dense_payload_of_view(const Tensor& t);

//
// The payload in image form, for a kernel that samples texels and was handed
// one that carries its elements in words.  The crossing is one pass of the
// strided gather and is worth it when the walk it feeds reads the payload many
// times; a shape with no image form at all reports instead of quietly keeping
// the caller in words.  An operand that already has an image comes back as it
// was, without a copy.
//
Tensor to_image_storage(api::Context* const context, const Tensor& self);

// The texel form of a parameter a sampling walk was handed in words, kept under
// the parameter's storage identity: a walk that reads the same weight, shift or
// statistic every call crosses it into an image once rather than paying the
// gather on each call.  The tags name which parameter was drawn, so two payloads
// sharing a storage never read one another's form.  A payload that already
// arrived as an image comes back as it was.
constexpr uint32_t kTagTexelAffineWeight = 30u;
constexpr uint32_t kTagTexelAffineBias = 31u;
constexpr uint32_t kTagTexelRunningMean = 32u;
constexpr uint32_t kTagTexelRunningVar = 33u;

Tensor texel_form_of_param(
    api::Context* const context,
    const Tensor& param,
    uint32_t tag);

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
