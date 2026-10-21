#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define GREATER ${GREATER}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum-index reduction over a half-precision linear payload.
 * The winning axis position is an integer, so the output keeps its own 32-bit
 * element and one invocation owns one row outright; only the input is read as
 * pairs of half lanes.  Strict comparisons implement the first-occurrence
 * tie-break, which is what makes greedy decoding pick the lowest index of a
 * tied set.  A lane that is not a number owns the row, and the earliest such
 * lane is the position named.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
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
  int greater;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float element_at(const in int index) {
  const vec2 pair = unpackHalf2x16(uInput.data[index >> 1]);
  return (index & 1) == 0 ? pair.x : pair.y;
}

// Whether a candidate (value, position) takes the row from the pair held: a
// lane that is not a number wins, a held lane that is not a number keeps its
// position, and a numeric pair falls back to the strict comparison.
bool takes_row(const in float value, const in int index, const in float best,
               const in int best_index) {
  const bool is_nan = isnan(value);
  if (isnan(best)) {
    return is_nan && index < best_index;
  }
  if (is_nan) {
    return true;
  }
  const bool lower_index = index < best_index;
  return uBlock.greater != 0
      ? (value > best || (value == best && lower_index))
      : (value < best || (value == best && lower_index));
}

void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);

  float best = element_at(base + inner_pos);
  int best_index = 0;
  for (int index = 1; index < uBlock.axis_size; ++index) {
    const float value = element_at(base + index * uBlock.inner + inner_pos);
    if (takes_row(value, index, best, best_index)) {
      best = value;
      best_index = index;
    }
  }
  uOutput.data[row] = best_index;
}
