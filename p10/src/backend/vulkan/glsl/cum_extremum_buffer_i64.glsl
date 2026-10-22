#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;
#include "wide_ext.h"

/*
 * Running extremum of a wide payload along one axis over linear storage.  One
 * element owns a pair of words, low one first, and one invocation owns one
 * scan row: it carries the winning pair with the position it came from and
 * writes both at every step.  The pair is ordered the way a signed number of
 * twice its width is, so a count past two to the thirty-second still folds in
 * its own type.  An equal pair takes over, so the index names the latest
 * position attaining the extremum.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly ValueBuffer {
  uvec2 data[];
}
uValues;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly IndexBuffer {
  int data[];
}
uIndices;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InputBuffer {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // Length of the scanned axis
  int axis_size;
  // Element count of the trailing dimensions
  int inner;
  // Scan rows, one per (leading block, inner position) pair
  int rows;
  // Unused: a wide element owns a whole pair of words
  int word_lane;
  // Whether the running value is the maximum (1) or the minimum (0)
  int is_max;
  int fill;
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

  const int first = base + inner_pos;
  uvec2 best = uInput.data[first];
  int best_index = 0;
  uValues.data[first] = best;
  uIndices.data[first] = 0;
  for (int i = 1; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    const uvec2 value = uInput.data[index];
    const bool take = uBlock.is_max != 0 ? wide_greater_equal(value, best)
                                         : wide_less_equal(value, best);
    if (take) {
      best = value;
      best_index = i;
    }
    uValues.data[index] = best;
    uIndices.data[index] = best_index;
  }
}
