#pragma once

#ifdef USE_VULKAN

#include "Common.h"
#include "Convert.h"
#include "../api/Context.h"
#include "../api/ShaderRegistry.h"
#include "../impl/Common.h"

namespace tensorplay {
namespace vulkan {
namespace ops {

//
// Block layouts shared by the image-shader ops.  Field order must match the
// uBlock declarations inside the corresponding GLSL templates exactly.
//

struct Sizes2Block final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int c_depth;
  int fill;
};

struct SoftmaxBlock final {
  ivec4 sizes;
  int c_depth;
  int axis;
};

struct SumAxisBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis;
  int in_c_depth;
  int out_c_depth;
  float scale;
  int fill1;
};

struct VarAxisBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis;
  int in_c_depth;
  int out_c_depth;
  int count;
  int correction;
};

struct LayerNormBlock final {
  ivec4 in_sizes;
  int c_depth;
  int channels;
  int span;
  int norm_channels;
  float eps;
  int fill0;
  int fill1;
};

struct BatchNormBlock final {
  ivec4 in_sizes;
  int c_depth;
  int channels;
  float eps;
  int fill0;
};

struct Pool2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec2 kernel;
  ivec2 stride;
  ivec2 padding;
  int c_depth;
  int count_include_pad;
  float divisor_override;
};

struct AdaptivePool2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int c_depth;
  int fill0;
};

struct UpsampleNearest2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  float scale_w;
  float scale_h;
  int c_depth;
  int fill0;
};

struct Conv2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec4 weight_sizes;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth;
  int out_c_depth;
  int weight_c_depth;
};

// Pointwise (1x1) and 1D convolution parameter block.
struct Conv1x1Block final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec4 weight_sizes;
  int in_c_depth;
  int out_c_depth;
};

struct ConvTranspose2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec4 weight_sizes;
  ivec2 stride;
  ivec2 padding;
  ivec2 output_padding;
  int in_c_depth;
  int out_c_depth;
  int weight_c_depth;
};

struct FlipBlock final {
  ivec4 in_sizes;
  ivec4 flip_axes;
  int c_depth;
  int fill;
};

struct CatChannelBlock final {
  ivec4 out_sizes;
  ivec4 in_sizes;
  int axis;
  int offset;
  int in_c_depth;
  int out_c_depth;
};

struct Pad2DBlock final {
  ivec4 in_sizes;
  ivec4 out_sizes;
  ivec2 padding;
  int c_depth;
  int fill;
};

struct PermuteBlock final {
  ivec4 in_logical;
  ivec4 out_logical;
  ivec4 perm;
  int in_c_depth;
  int out_c_depth;
};

struct SliceBlock final {
  ivec4 out_sizes;
  int axis;
  int start;
  int step;
  int removed;
  int in_c_depth;
  int out_c_depth;
};

// Strided-view materialization: sizes/strides are prepadded to four axes in
// {N, C, H, W} order, matching the view_gather shader's uBlock.
struct ViewGatherBlock final {
  ivec4 in_sizes;
  ivec4 in_strides;
  ivec4 out_sizes;
  ivec4 out_strides;
  int in_c_depth;
  int out_c_depth;
  int offset;
};

struct LerpBlock final {
  ivec4 extents;
  int scalar_end;
  int scalar_weight;
  float weight;
};

// ceil(C / 4) as int32 for uniform blocks.
inline int32_t c_depth_of(const std::vector<int64_t>& sizes) {
  return static_cast<int32_t>(
      api::utils::div_up(get_dim<Dim4D::Channel>(sizes), 4u));
}

// Convenience: block with an (W,H,C,N) sizes vector and its c_depth.
template <uint32_t N>
int32_t get_dim_i(const std::vector<int64_t>& sizes) {
  return static_cast<int32_t>(get_dim<N>(sizes));
}

// Defined in Clone.cpp; shared by ops that build on exact copies.
Tensor clone_kernel(const Tensor& self);

// Defined in View.cpp; broadcast-expands through zero-stride gathers.
Tensor expand_kernel(const Tensor& self, const std::vector<int64_t>& size, bool implicit);

// Defined in Shape.cpp; shared by ops that build on slicing and joins.
Tensor slice_kernel(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step);
Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim);
Tensor reshape_kernel(const Tensor& self, const std::vector<int64_t>& shape);
Tensor permute_kernel(const Tensor& self, const std::vector<int64_t>& dims);

// Defined in Factory.cpp; host-visible fill entry used by selection ops.
Tensor full_kernel(
    const std::vector<int64_t>& size,
    Scalar fill_value,
    DType dtype,
    Device device,
    bool pin_memory);

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

#endif /* USE_VULKAN */
