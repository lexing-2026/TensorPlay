#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define GREATER ${GREATER}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int axis_size;
  int inner;
  int rows;
  int greater;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Single-axis extremum-index reduction over buffer storage, for shapes
 * whose extents exceed the 3-D image axis limit.  Each sort row decomposes
 * into an outer block and an inner run, and one invocation walks the axis
 * of that row; strict comparisons implement the first-occurrence tie-break.
 * The winning axis position is written as a linear element of the reduced
 * output row.
 */
void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);
  float best = uInput.data[base + inner_pos];
  int best_index = 0;
  for (int index = 1; index < uBlock.axis_size; ++index) {
    const float value = uInput.data[base + index * uBlock.inner + inner_pos];
    const bool take = uBlock.greater != 0 ? value > best : value < best;
    if (take) {
      best = value;
      best_index = index;
    }
  }
  uOutput.data[row] = best_index;
}