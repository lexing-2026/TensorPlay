#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum of a wide payload over linear storage.  One element
 * owns a pair of words, low one first, and one invocation owns one row: it
 * walks the axis of that row, striding by `inner`, and writes the winning
 * element.  The fold starts at the row's first element, so no identity value
 * has to reach the ends of the range a wide count can name, and `total` bounds
 * the walk so a tile grid that does not divide the payload evenly cannot read
 * past the last element.  Comparisons are strict: the first of tied elements
 * stays the winner.
 *
 * The pair is ordered the way a signed number of twice its width is, without
 * carrying the fold in a wider register: bias the sign out of the high word and
 * compare the words from the top down.  A high word that leads names the
 * larger number, and two elements sharing a biased high word are ordered by
 * their low words.
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

  uvec2 best = uInput.data[base + inner_pos];
  for (int index = 1; index < count; ++index) {
    const uvec2 value = uInput.data[base + index * uBlock.inner + inner_pos];
    const uint held = best.y ^ 0x80000000u;
    const uint other = value.y ^ 0x80000000u;
    // clang-format off
    $if GREATER == 1:
      if (other > held || (other == held && value.x > best.x)) {
        best = value;
      }
    $else:
      if (other < held || (other == held && value.x < best.x)) {
        best = value;
      }
    // clang-format on
  }
  uOutput.data[row] = best;
}
