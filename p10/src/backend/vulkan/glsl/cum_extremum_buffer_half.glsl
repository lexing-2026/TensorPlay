#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Half variant of the linear running extremum.  The payload packs two 16-bit
 * lanes per word, and the two lanes of a word sit on neighbouring inner
 * positions, which belong to different scan rows.  The walk therefore runs
 * twice: a pass carries every row as usual but stores only the fields whose
 * flat index falls in that pass's word lane, so each destination word is
 * touched by exactly one invocation per pass.  The comparison widens a lane to
 * single precision, where every half value is held exactly, and the winner
 * narrows back without rounding.  The index is one word per element, so the
 * two passes write disjoint words.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict ValueBuffer {
  uint data[];
}
uValues;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly IndexBuffer {
  int data[];
}
uIndices;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InputBuffer {
  uint data[];
}
uInput;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // Length of the scanned axis
  int axis_size;
  // Element count of the trailing dimensions
  int inner;
  // Scan rows, one per (leading block, inner position) pair
  int rows;
  // Word lane this pass owns: 0 fills the low field, 1 the high one
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

  const uint first = uint(base + inner_pos);
  float best = unpackHalf2x16(uInput.data[first >> 1u])[first & 1u];
  int best_index = 0;
  if (uBlock.word_lane == int(first & 1u)) {
    const uint shift = (first & 1u) * 16u;
    const uint word = uValues.data[first >> 1u];
    uValues.data[first >> 1u] = (word & ~(0xffffu << shift)) |
        ((packHalf2x16(vec2(best, 0.0)) & 0xffffu) << shift);
    uIndices.data[first] = 0;
  }

  for (int i = 1; i < uBlock.axis_size; ++i) {
    const uint index = uint(base + i * uBlock.inner + inner_pos);
    const float value = unpackHalf2x16(uInput.data[index >> 1u])[index & 1u];
    bool take = isnan(value);
    if (!take && !isnan(best)) {
      take = uBlock.is_max != 0 ? value >= best : value <= best;
    }
    if (take) {
      best = value;
      best_index = i;
    }
    if (int(index & 1u) != uBlock.word_lane) {
      continue;
    }
    // Read-modify-write keeps the neighbouring lane: the other pass owns it.
    const uint shift = (index & 1u) * 16u;
    const uint word = uValues.data[index >> 1u];
    uValues.data[index >> 1u] = (word & ~(0xffffu << shift)) |
        ((packHalf2x16(vec2(best, 0.0)) & 0xffffu) << shift);
    uIndices.data[index] = best_index;
  }
}
