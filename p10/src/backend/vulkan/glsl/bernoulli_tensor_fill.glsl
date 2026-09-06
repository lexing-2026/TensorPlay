#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

#include "philox.h"

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uProb;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  uint seed_lo;
  uint seed_hi;
  uint offset;
  uint fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Bernoulli draws against a per-element probability plane: the probability
 * shares the output's texel geometry, so each of the four channel lanes
 * compares its own uniform draw against its own probability.  One Philox
 * call yields exactly the four words a texel needs.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const uint tid =
      uint(pos.x) + uint(uBlock.extents.x) *
          (uint(pos.y) + uint(uBlock.extents.y) * uint(pos.z));
  const uvec4 ctr = uvec4(tid, uBlock.offset, 0u, 0u);
  const uvec4 words =
      philox4x32_10(ctr, uvec2(uBlock.seed_lo, uBlock.seed_hi));

  const vec4 u = vec4(
      word_to_uniform(words.x), word_to_uniform(words.y),
      word_to_uniform(words.z), word_to_uniform(words.w));
  const vec4 p = texelFetch(uProb, pos, 0);
  imageStore(uOutput, pos, mix(vec4(0.0f), vec4(1.0f), lessThan(u, p)));
}
