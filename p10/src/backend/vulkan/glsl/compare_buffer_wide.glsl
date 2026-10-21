#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "broadcast_index.h"

/*
 * Elementwise comparison of two payloads whose elements are wider than a word.
 *
 * A count that needs more than 32 bits has no texel form, so it always lives in
 * linear storage as a pair of words per element, the low word first.  One such
 * element is therefore read as a pair, and the decision is made word against
 * word: the high word carries the sign and settles the order on its own, and
 * the low word only breaks a tie between them - where it counts as unsigned,
 * because the sign has already been settled above it.
 *
 * The Bool result packs one byte per element, four lanes to a word.  An
 * invocation owns one destination word and answers for its four elements, so no
 * two invocations write the same word.  A word past the logical end of a result
 * is still inside its allocation, which rounds up to sixteen bytes, and nothing
 * reads those lanes back.
 *
 * With `broadcast` set (bit 0: the first operand, bit 1: the second), an
 * element's flat index is recomputed from row-major strides, so a dimension of
 * one repeats across the output range without the operand being materialized at
 * the output shape.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Other {
  uvec2 data[];
}
uOther;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  uint buf_length; // element count
  uint fill0;      // std140 padding
  int broadcast;   // bit 0: the first operand, bit 1: the second
  uint fill1;
  ivec4 out_strides; // row-major strides of the output shape
  ivec4 input_sizes; // each operand's own extents, with singletons
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Signed order decided word against word, as described above.
bool wide_less(const uvec2 a, const uvec2 b) {
  return int(a.y) < int(b.y) || (a.y == b.y && a.x < b.x);
}
bool wide_less_equal(const uvec2 a, const uvec2 b) {
  return !wide_less(b, a);
}
bool wide_greater(const uvec2 a, const uvec2 b) {
  return wide_less(b, a);
}
bool wide_greater_equal(const uvec2 a, const uvec2 b) {
  return !wide_less(a, b);
}
bool wide_equal(const uvec2 a, const uvec2 b) {
  return a.x == b.x && a.y == b.y;
}
bool wide_not_equal(const uvec2 a, const uvec2 b) {
  return !wide_equal(a, b);
}

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }

  uint word = 0u;
  for (uint k = 0u; k < 4u; ++k) {
    const uint i = idx * 4u + k;
    if (i >= uBlock.buf_length) {
      break;
    }
    const uint ai = (uBlock.broadcast & 1) != 0
        ? remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                      uBlock.input_strides)
        : i;
    const uint bi = (uBlock.broadcast & 2) != 0
        ? remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                      uBlock.other_strides)
        : i;
    if (OP(uInput.data[ai], uOther.data[bi])) {
      word |= (1u << (k * 8u));
    }
  }

  uOutput.data[idx] = word;
}
