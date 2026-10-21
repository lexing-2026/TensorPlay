#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum of a signed-word payload over linear storage, for
 * shapes whose extents exceed the 3-D image axis limit.  One invocation owns
 * one row: it walks the axis of that row, striding by `inner`, and writes the
 * winning element.  The fold starts at the row's first element, so it needs no
 * identity that a count could fall outside of, and `total` bounds the walk so a
 * tile grid that does not divide the payload evenly cannot read past the last
 * element.  Comparisons are strict: the first of tied elements stays the
 * winner.  A count is ordered as the signed word it names, never as a
 * measurement, so nothing rounds on the way in.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  int data[];
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

  int best = uInput.data[base + inner_pos];
  for (int index = 1; index < count; ++index) {
    const int value = uInput.data[base + index * uBlock.inner + inner_pos];
    // clang-format off
    $if GREATER == 1:
      if (value > best) {
        best = value;
      }
    $else:
      if (value < best) {
        best = value;
      }
    // clang-format on
  }
  uOutput.data[row] = best;
}
