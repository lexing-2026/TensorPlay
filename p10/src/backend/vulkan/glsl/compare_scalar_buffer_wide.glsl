#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "wide_ext.h"

/*
 * Elementwise comparison of a linear payload of codes wider than a word against
 * a number.  The threshold arrives as the pair of words such a count is held in,
 * so the order is the same walk the two-payload form takes, with one side fixed.
 *
 * The Bool result packs one byte per element, four lanes to a word.  An
 * invocation owns one destination word and answers for its four elements, so no
 * two invocations write the same word.  A word past the logical end of a result
 * is still inside its allocation, which rounds up to sixteen bytes, and nothing
 * reads those lanes back.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uint buf_length; // element count
  uint fill0;      // std140 padding
  uint value_low;  // the threshold, low word first
  uint value_high;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }
  const uvec2 threshold = uvec2(uBlock.value_low, uBlock.value_high);

  uint word = 0u;
  for (uint k = 0u; k < 4u; ++k) {
    const uint i = idx * 4u + k;
    if (i >= uBlock.buf_length) {
      break;
    }
    if (OP(uInput.data[i], threshold)) {
      word |= (1u << (k * 8u));
    }
  }

  uOutput.data[idx] = word;
}
