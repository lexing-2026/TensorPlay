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
 *
 * One texel carries four independent elements, one per channel lane, and a
 * single Philox call yields exactly four words -- so each lane consumes its
 * own word and no two elements of a texel ever share a draw.
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
    vec4 value = vec4(
        word_to_uniform(words.x), word_to_uniform(words.y),
        word_to_uniform(words.z), word_to_uniform(words.w));
    value = value * (uBlock.to - uBlock.from) + uBlock.from;
    // The draws are in [0, 1); after scaling a value can only reach `to`
    // through rounding, which snaps back to keep the [from, to) contract.
    value = mix(value, vec4(uBlock.from), equal(value, vec4(uBlock.to)));
    imageStore(uOutput, pos, value);
  $elif DIST == 1:
    // Two uniform pairs give four standard normals; the affine map carries
    // the requested mean and spread.
    const vec2 first = box_muller(
        word_to_uniform(words.x), word_to_uniform(words.y));
    const vec2 second = box_muller(
        word_to_uniform(words.z), word_to_uniform(words.w));
    imageStore(
        uOutput, pos,
        uBlock.to + uBlock.std * vec4(first.x, first.y, second.x, second.y));
  $else:
    const vec4 u = vec4(
        word_to_uniform(words.x), word_to_uniform(words.y),
        word_to_uniform(words.z), word_to_uniform(words.w));
    imageStore(
        uOutput, pos,
        mix(vec4(0.0f), vec4(1.0f), lessThan(u, vec4(uBlock.from))));
  // clang-format on
}
