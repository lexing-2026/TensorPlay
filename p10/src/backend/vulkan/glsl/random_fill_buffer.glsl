#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define DIST ${DIST}
// clang-format on

layout(std430) buffer;

#include "philox.h"

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  uint buf_length;
  float from; // uniform lower bound (uniform) / bernoulli probability / rate (exponential)
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
 * Nullary random fill for buffer-stored tensors, selected by DIST:
 *   0: uniform_[from, to)         value = u * (to - from) + from, with a
 *                                 draw equal to `to` snapped back to `from`
 *   1: normal_(mean, std)         two Box-Muller draws per uniform pair
 *   2: bernoulli_(p)              (u < p)
 *   3: exponential_(lambda)       value = -log(u) / lambda
 * Four consecutive linear elements consume one Philox call, one word per
 * element, so no two elements ever share a draw and the sequence is
 * reproducible for a given seed regardless of scheduling.
 */
void main() {
  const uint tid = gl_GlobalInvocationID.x; // four consecutive elements
  const uint first = tid * 4u;
  if (first >= uBlock.buf_length) {
    return;
  }

  // Counter words carry the element group and generation offset; the seed
  // pair forms the Philox key.
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
  $elif DIST == 1:
    // Two uniform pairs give four standard normals; the affine map carries
    // the requested mean and spread.
    const vec2 first_pair = box_muller(
        word_to_uniform(words.x), word_to_uniform(words.y));
    const vec2 second_pair = box_muller(
        word_to_uniform(words.z), word_to_uniform(words.w));
    vec4 value =
        uBlock.to + uBlock.std * vec4(first_pair.x, first_pair.y, second_pair.x, second_pair.y);
  $elif DIST == 2:
    vec4 value = vec4(
        word_to_uniform(words.x), word_to_uniform(words.y),
        word_to_uniform(words.z), word_to_uniform(words.w));
  $else:
    // Inverse-CDF draw: -log(u) / rate.  The draws lie in [0, 1), so the edge
    // to guard is the one at 0 -- its logarithm is -infinity, and a value of
    // infinity leaves its category unrankable -- so a draw of no bits is
    // lifted to the smallest value the lane holds.  The upper edge is open, so
    // the result is never 0 and a division by it stays defined.
    vec4 value = -log(max(
        vec4(word_to_uniform(words.x), word_to_uniform(words.y),
             word_to_uniform(words.z), word_to_uniform(words.w)),
        vec4(1.17549435e-38))) / uBlock.from;
  // clang-format on

  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint index = first + lane;
    if (index < uBlock.buf_length) {
      // clang-format off
      $if DIST == 2:
        uOutput.data[index] = mix(0.0f, 1.0f, value[lane] < uBlock.from);
      $else:
        uOutput.data[index] = value[lane];
      // clang-format on
    }
  }
}
