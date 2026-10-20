#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Cumulative sum of a 64-bit payload over linear storage.  One element owns a
 * pair of words, low one first, and one invocation walks one scan row and
 * writes the running total at every position.  The total is carried across the
 * pair instead of living in one wider register: the payload has no texel format
 * to name and a sum of counts must keep every bit a count can reach.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Length of the scanned axis
  int axis_size;
  // Element count of the trailing dimensions
  int inner;
  // Scan rows, one per (leading block, inner position) pair
  int rows;
  // Unused: a wide element owns a whole pair of words, so one pass covers a
  // row completely
  int word_lane;
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

  uint low = 0u;
  uint high = 0u;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    const uvec2 code = uInput.data[index];
    const uint next = low + code.x;
    // A low word that wraps backwards has handed one to the high word.  Word
    // arithmetic is modular, so a total past 2^64 lands where the codes it
    // added would land.
    high += code.y + (next < low ? 1u : 0u);
    low = next;
    uOutput.data[index] = uvec2(low, high);
  }
}
