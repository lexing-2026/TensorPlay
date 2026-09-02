#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) logical sizes
  ivec4 out_sizes; // (OW, OH, O, N) logical sizes
  ivec4 weight_sizes; // (KW, KH, ..., ...) with .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth; // ceil(C / 4)
  int out_c_depth; // ceil(O / 4)
  int weight_c_depth; // ceil(C / 4), channel depth of the weight texture
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Grouped (single group) 2D convolution in gather form: one invocation
 * computes the four output channels covered by one texel at one spatial
 * position, looping the kernel window and the input channels.  Output
 * channels beyond the channel count accumulate into padding lanes that no
 * consumer reads.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x ||
      pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  // Logical output coordinates: pos.x = output width (ow), pos.y = output
  // height (oh), pos.z = batch * out_c_depth + output channel group.
  const int ow = pos.x;
  const int oh = pos.y;
  const int n = pos.z / uBlock.out_c_depth;
  const int o4 = pos.z % uBlock.out_c_depth;

  vec4 acc = vec4(0.0f);

  for (int ky = 0; ky < uBlock.weight_sizes.z; ++ky) {
    const int ih =
        oh * uBlock.stride.y - uBlock.padding.y + ky * uBlock.dilation.y;
    if (ih < 0 || ih >= uBlock.in_sizes.y) {
      continue;
    }
    for (int kx = 0; kx < uBlock.weight_sizes.w; ++kx) {
      const int iw =
          ow * uBlock.stride.x - uBlock.padding.x + kx * uBlock.dilation.x;
      if (iw < 0 || iw >= uBlock.in_sizes.x) {
        continue;
      }
      for (int ci = 0; ci < uBlock.in_sizes.z; ++ci) {
        const float v = texelFetch(
            uInput,
            ivec3(iw, ih, n * uBlock.in_c_depth + ci / 4),
            0)[ci % 4];
        // Weight texture: {O, C, KH, KW} logical sizes map to
        // (W=KW, H=KH, C=C, N=O); lane selects the input channel, texel z
        // the output channel and input channel group.
        const int o = o4 * 4;
        const vec4 w = vec4(
            texelFetch(
                uWeight,
                ivec3(kx, ky, (o + 0) * uBlock.weight_c_depth + ci / 4),
                0)[ci % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, (o + 1) * uBlock.weight_c_depth + ci / 4),
                0)[ci % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, (o + 2) * uBlock.weight_c_depth + ci / 4),
                0)[ci % 4],
            texelFetch(
                uWeight,
                ivec3(kx, ky, (o + 3) * uBlock.weight_c_depth + ci / 4),
                0)[ci % 4]);
        acc += v * w;
      }
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, o4, uBlock.out_sizes.z);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
