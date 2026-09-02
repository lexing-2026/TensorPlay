#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define DIST ${DIST}
// clang-format on

layout(std430) buffer;

#include "philox.h"

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  ivec4 extents;
  float from; // uniform lower bound (uniform) / bernoulli probability
  float to; // uniform upper bound (uniform) / normal mean
  float std; // normal standard deviation
  uint seed_lo;
  uint seed_hi;
  uint offset;
  uint fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Nullary random fill for three distributions, selected by DIST:
 *   0: uniform_[from, to)         value = u * (to - from) + from, with a
 *                                 draw equal to `to` snapped back to `from`
 *   1: normal_(mean, std)         two Box-Muller draws per uniform pair
 *   2: bernoulli_(p)              (u < p)
 * Every invocation derives its random words from (seed, offset, id), so the
 * fill is reproducible for a given seed and independent of scheduling.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const uint tid =
      uint(pos.x) + uint(uBlock.extents.x) *
          (uint(pos.y) + uint(uBlock.extents.y) * uint(pos.z));

  // Counter words carry the element position and generation offset; the
  // seed pair forms the Philox key.
  const uvec4 ctr = uvec4(tid, uBlock.offset, 0u, 0u);
  const uvec4 words =
      philox4x32_10(ctr, uvec2(uBlock.seed_lo, uBlock.seed_hi));

  // clang-format off
  $if DIST == 0:
    const float u = word_to_uniform(words.x);
    float value = u * (uBlock.to - uBlock.from) + uBlock.from;
    // The draw is in [0, 1); after scaling the value can only hit `to`
    // through rounding, which snaps back to keep the [from, to) contract.
    if (value == uBlock.to) {
      value = uBlock.from;
    }
    imageStore(uOutput, pos, vec4(value));
  $elif DIST == 1:
    const vec2 pair = box_muller(
        word_to_uniform(words.x), word_to_uniform(words.y));
    imageStore(uOutput, pos, vec4(uBlock.to + uBlock.std * pair.x));
  $else:
    const float u = word_to_uniform(words.x);
    imageStore(uOutput, pos, vec4(u < uBlock.from ? 1.0f : 0.0f));
  // clang-format on
}
