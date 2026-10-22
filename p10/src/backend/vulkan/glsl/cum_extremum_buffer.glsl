#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Running extremum of a single-precision payload along one axis over linear
 * storage.  One invocation owns one scan row: it carries the winning value and
 * the position that value came from, and writes both at every step.
 *
 * A later element that equals the running winner takes over, so the index names
 * the latest position attaining the extremum.  A non-number always takes over
 * and then keeps the row, because a non-number accumulator wins every later
 * comparison; that is the fold the host kernels implement.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly ValueBuffer {
  float data[];
}
uValues;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly IndexBuffer {
  int data[];
}
uIndices;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InputBuffer {
  float data[];
}
uInput;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // Length of the scanned axis
  int axis_size;
  // Element count of the trailing dimensions
  int inner;
  // Scan rows, one per (leading block, inner position) pair
  int rows;
  // Unused: a single-precision element owns a whole word
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
  float best = uInput.data[first];
  int best_index = 0;
  uValues.data[first] = best;
  uIndices.data[first] = 0;
  for (int i = 1; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    const float value = uInput.data[index];
    bool take = isnan(value);
    if (!take && !isnan(best)) {
      take = uBlock.is_max != 0 ? value >= best : value <= best;
    }
    if (take) {
      best = value;
      best_index = i;
    }
    uValues.data[index] = best;
    uIndices.data[index] = best_index;
  }
}
