#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

#include "philox.h"

// Bernoulli draws against a per-element probability plane held in buffer
// storage, for shapes beyond the 3-D image axis limit.  One Philox call
// yields exactly the four words a group of four consecutive elements
// needs; each lane compares its own uniform draw against its own
// probability at the same linear position.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Prob {
  float data[];
}
uProb;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uint buf_length;
  uint seed_lo;
  uint seed_hi;
  uint offset;
  uint fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

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

  const vec4 u = vec4(
      word_to_uniform(words.x), word_to_uniform(words.y),
      word_to_uniform(words.z), word_to_uniform(words.w));

  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint index = first + lane;
    if (index < uBlock.buf_length) {
      uOutput.data[index] =
          mix(0.0f, 1.0f, u[lane] < uProb.data[index]);
    }
  }
}