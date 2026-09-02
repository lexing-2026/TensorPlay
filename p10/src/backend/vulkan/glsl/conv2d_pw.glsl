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
  ivec4 weight_sizes; // (1, 1, ..., ...) with .y = C (row length)
  int in_c_depth; // ceil(C / 4)
  int out_c_depth; // ceil(O / 4)
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Pointwise (1x1) convolution: a matrix product between the channel axis of
 * the input and the weight rows, evaluated at every spatial position.  One
 * invocation computes the four output channels of one texel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x ||
      pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n = pos.z / uBlock.out_c_depth;
  const int o4 = pos.z % uBlock.out_c_depth;
  const int o = o4 * 4;

  vec4 acc = vec4(0.0f);

  for (int ci = 0; ci < uBlock.in_sizes.z; ++ci) {
    const vec4 v = texelFetch(
        uInput,
        ivec3(pos.x, pos.y, n * uBlock.in_c_depth + ci / 4),
        0);
    const float lane = v[ci % 4];
    // Weight texture: {O, C, 1, 1} logical sizes map to (W=1, H=1, C=C,
    // N=O); the row of output channel o lives at texel z = o, lane ci % 4.
    acc += lane *
        vec4(
            texelFetch(uWeight, ivec3(0, 0, o + 0), 0)[ci % 4],
            texelFetch(uWeight, ivec3(0, 0, o + 1), 0)[ci % 4],
            texelFetch(uWeight, ivec3(0, 0, o + 2), 0)[ci % 4],
            texelFetch(uWeight, ivec3(0, 0, o + 3), 0)[ci % 4]);
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, o4, uBlock.out_sizes.z);
  // clang-format on

  imageStore(uOutput, pos, acc);
}
