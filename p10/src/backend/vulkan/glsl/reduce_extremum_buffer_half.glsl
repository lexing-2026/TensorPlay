#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define GREATER ${GREATER}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum reduction over a half-precision linear payload.  Two
 * adjacent elements share one 32-bit word, so an invocation owns one whole
 * output word and reduces the two sort rows that land in it together; the
 * store then needs no read-modify-write and no two invocations can race on a
 * lane.
 *
 * Comparisons are strict, keeping the first occurrence when several elements
 * tie.  A lane that is not a number wins the row and nothing displaces it
 * afterwards, so both ends of a range name it.  The winner is folded in single
 * precision and narrowed once at the store.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
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

float row_extremum(const in int row) {
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);
  // A row grid that tiles the payload in fixed chunks can end past its last
  // element, so the walk stops at the element count instead of at the tile
  // size.
  const int count = min(uBlock.axis_size, uBlock.total - base);

  vec2 pair = unpackHalf2x16(uInput.data[(base + inner_pos) >> 1]);
  float best = ((base + inner_pos) & 1) == 0 ? pair.x : pair.y;
  for (int index = 1; index < count; ++index) {
    const int e = base + index * uBlock.inner + inner_pos;
    pair = unpackHalf2x16(uInput.data[e >> 1]);
    const float value = (e & 1) == 0 ? pair.x : pair.y;
    // clang-format off
    $if GREATER == 1:
      if (!isnan(best) && (isnan(value) || value > best)) {
        best = value;
      }
    $else:
      if (!isnan(best) && (isnan(value) || value < best)) {
        best = value;
      }
    // clang-format on
  }
  return best;
}

void main() {
  const int word = int(gl_GlobalInvocationID.x);
  const int first = 2 * word;
  if (first >= uBlock.rows) {
    return;
  }
  // An odd row count repeats the final row in the tail lane; the reduced
  // extent stops before it, so nothing reads it.
  const int second = min(first + 1, uBlock.rows - 1);
  uOutput.data[word] =
      packHalf2x16(vec2(row_extremum(first), row_extremum(second)));
}
