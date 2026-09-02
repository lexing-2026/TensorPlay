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
  ivec4 out_sizes; // (OW, OH, C, N) logical sizes
  ivec4 weight_sizes; // (KW, KH, ..., ...) with .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int c_depth; // ceil(C / 4)
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Depthwise 2D convolution (groups == channels, one plane per channel):
 * every output channel filters its own input channel with a 2D kernel.
 * One invocation computes one texel, i.e. four channels of one output
 * position; the input fetch is shared across those channels while the
 * kernel weights are fetched per channel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x ||
      pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const int ow = pos.x;
  const int oh = pos.y;
  const int n = pos.z / uBlock.c_depth;
  const int c4 = pos.z % uBlock.c_depth;

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
      const vec4 v = texelFetch(
          uInput, ivec3(iw, ih, n * uBlock.c_depth + c4), 0);
      // Weight texture: {C, 1, KH, KW} logical sizes map to
      // (W=KW, H=KH, C=1, N=C); texel z carries the channel.
      const ivec4 c_idx = min(
          ivec4(c4 * 4) + ivec4(0, 1, 2, 3),
          ivec4(uBlock.in_sizes.z - 1));
      const vec4 w = vec4(
          texelFetch(uWeight, ivec3(kx, ky, c_idx.x), 0).x,
          texelFetch(uWeight, ivec3(kx, ky, c_idx.y), 0).x,
          texelFetch(uWeight, ivec3(kx, ky, c_idx.z), 0).x,
          texelFetch(uWeight, ivec3(kx, ky, c_idx.w), 0).x);
      acc += v * w;
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, c4, uBlock.in_sizes.z);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
