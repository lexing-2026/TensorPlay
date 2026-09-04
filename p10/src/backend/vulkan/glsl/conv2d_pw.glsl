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
  ivec4 weight_sizes; // (unused, unused, O4, C4) of the packed weight texture
  int in_c_depth; // ceil(C / 4)
  int out_c_depth; // ceil(O / 4)
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Pointwise (1x1) convolution: a matrix product between the channel axis of
 * the input and the weight rows, evaluated at every spatial position.
 *
 * Packed weight layout: texel (ic4, o4, lane) holds, in its four components,
 * w[oc = 4*o4 + comp][ic = 4*ic4 + lane].  One invocation computes a 2x2
 * block of output positions; the four weight fetches per input-channel group
 * are shared across the block (the per-position form re-fetched them four
 * times), and input fetches past the output edges return zeros whose stores
 * are discarded.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x * 2 >= uBlock.out_sizes.x ||
      pos.y * 2 >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n = pos.z / uBlock.out_c_depth;
  const int o4 = pos.z % uBlock.out_c_depth;
  const int x0 = pos.x * 2;
  const int y0 = pos.y * 2;

  vec4 acc00 = vec4(0.0f);
  vec4 acc01 = vec4(0.0f);
  vec4 acc10 = vec4(0.0f);
  vec4 acc11 = vec4(0.0f);

  for (int ic4 = 0; ic4 < uBlock.in_c_depth; ++ic4) {
    const int in_z = n * uBlock.in_c_depth + ic4;
    const vec4 v00 = texelFetch(uInput, ivec3(x0, y0, in_z), 0);
    const vec4 v01 = texelFetch(uInput, ivec3(x0 + 1, y0, in_z), 0);
    const vec4 v10 = texelFetch(uInput, ivec3(x0, y0 + 1, in_z), 0);
    const vec4 v11 = texelFetch(uInput, ivec3(x0 + 1, y0 + 1, in_z), 0);
    for (int lane = 0; lane < 4; ++lane) {
      const vec4 w = texelFetch(uWeight, ivec3(ic4, o4, lane), 0);
      acc00 += v00[lane] * w;
      acc01 += v01[lane] * w;
      acc10 += v10[lane] * w;
      acc11 += v11[lane] * w;
    }
  }

  // clang-format off
  $if HAS_BIAS:
    const vec4 b = param_vec(uBias, o4, uBlock.out_sizes.z);
    acc00 += b;
    acc01 += b;
    acc10 += b;
    acc11 += b;
  // clang-format on

  imageStore(uOutput, ivec3(x0, y0, pos.z), acc00);
  imageStore(uOutput, ivec3(x0 + 1, y0, pos.z), acc01);
  imageStore(uOutput, ivec3(x0, y0 + 1, pos.z), acc10);
  imageStore(uOutput, ivec3(x0 + 1, y0 + 1, pos.z), acc11);
}
