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
  ivec4 in_sizes; // (W=L, H=C, C=N, 1) of a {N, C, L} input
  ivec4 out_sizes; // (W=OL, H=O, C=N, 1) of a {N, O, OL} output
  ivec4 weight_sizes; // {O, C, K} logical sizes
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth; // ceil(N / 4) for a 3d layout
  int out_c_depth; // ceil(N / 4) for a 3d layout
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Grouped (single group) 1D convolution.  A {N, C, L} tensor is stored with
 * the length on the texel x axis and channels on y; the texel z axis holds
 * groups of four batches.  One invocation computes one output length
 * position for one output channel, across four batches at once (the lanes
 * of the input fetch).
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x ||
      pos.y >= uBlock.out_sizes.y ||
      pos.z >= (uBlock.in_sizes.z + 3) / 4) {
    return;
  }

  const int ol = pos.x;
  const int o = pos.y;
  const int b4 = pos.z;

  vec4 acc = vec4(0.0f);

  for (int c = 0; c < uBlock.in_sizes.y; ++c) {
    for (int k = 0; k < uBlock.weight_sizes.z; ++k) {
      const int il =
          ol * uBlock.stride.x - uBlock.padding.x + k * uBlock.dilation.x;
      if (il < 0 || il >= uBlock.in_sizes.x) {
        continue;
      }
      // Input texel lanes carry the four batches b = b4*4 + lane.
      const vec4 v = texelFetch(uInput, ivec3(il, c, b4), 0);
      // Weight texel: {O, C, K} logical sizes map to (W=K, H=C, C=O, N=1);
      // lane selects the output channel.
      const float w = texelFetch(
          uWeight, ivec3(k, c, o / 4), 0)[o % 4];
      acc += v * w;
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += vec4(texelFetch(uBias, ivec3(o, 0, 0), 0).x);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
