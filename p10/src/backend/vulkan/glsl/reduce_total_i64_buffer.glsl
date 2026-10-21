#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

#include "wide_ext.h"

/*
 * One axis of a wide payload folded into a total or a product over linear
 * storage.  An element owns a pair of words, low one first, and one invocation
 * owns one row: it walks the axis of that row, striding by `inner`, and
 * accumulates from the identity the fold names.  `total` bounds the walk, so a
 * tile grid that does not divide the payload evenly cannot read past the last
 * element and a short tile contributes fewer terms.
 *
 * The accumulator stays a pair rather than reaching for a wider register: a
 * running total past every word lands where the counts it added would land,
 * which is what a count wrapped at its own width means.  Nothing here asks a
 * shader for 64-bit arithmetic.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int axis_size;
  int inner;
  int rows;
  int total;
  int fill;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);
  const int count = min(uBlock.axis_size, uBlock.total - base);

  uvec2 folded = ${IDENTITY};
  for (int index = 0; index < count; ++index) {
    folded = ${OPERATOR}(
        folded, uInput.data[base + index * uBlock.inner + inner_pos]);
  }
  uOutput.data[row] = folded;
}
