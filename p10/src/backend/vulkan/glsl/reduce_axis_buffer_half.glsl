#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define MODE ${MODE}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis sum/mean/variance reduction over a half-precision linear payload.
 * Two adjacent elements share one 32-bit word, so an invocation owns one whole
 * output word: it reduces the two sort rows that land in it and stores both
 * halves together.  That keeps every output write free of read-modify-write,
 * which the reduced layout cannot get any other way because neighbouring rows
 * of the output are neighbours in memory.
 *
 * Rows are addressed by element, exactly as the float builds do; `inner`,
 * `axis_size` and `total` all count elements, not words.  The last word of an
 * odd row count repeats the final row in its high lane, which no reader ever
 * reaches because the reduced extent stops before it.
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
  int correction;
  float scale;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float element_at(const in int index) {
  const vec2 pair = unpackHalf2x16(uInput.data[index >> 1]);
  return (index & 1) == 0 ? pair.x : pair.y;
}

// Walks one sort row: an outer block of `axis_size` * `inner` elements,
// sampled every `inner`-th from `inner_pos` inside it.
void walk(const in int row, out float acc, out float sq) {
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);
  // A row grid that tiles the payload in fixed chunks can end past its last
  // element, so the walk stops at the element count instead of at the tile
  // size.
  const int count = min(uBlock.axis_size, uBlock.total - base);

  acc = 0.0;
  sq = 0.0;
  for (int i = 0; i < count; ++i) {
    const float v = element_at(base + i * uBlock.inner + inner_pos);
    acc += v;
    // clang-format off
    $if MODE == 1:
      sq += v * v;
    // clang-format on
  }
}

float reduced(const in float acc, const in float sq) {
  // clang-format off
  $if MODE == 1:
    const float mean = acc / float(uBlock.axis_size);
    return (sq / float(uBlock.axis_size) - mean * mean) *
        float(uBlock.axis_size) /
        float(max(uBlock.axis_size - uBlock.correction, 1));
  $else:
    return acc * uBlock.scale;
  // clang-format on
}

void main() {
  const int word = int(gl_GlobalInvocationID.x);
  const int first = 2 * word;
  if (first >= uBlock.rows) {
    return;
  }
  const int second = min(first + 1, uBlock.rows - 1);

  float acc0 = 0.0;
  float sq0 = 0.0;
  float acc1 = 0.0;
  float sq1 = 0.0;
  walk(first, acc0, sq0);
  walk(second, acc1, sq1);

  uOutput.data[word] =
      packHalf2x16(vec2(reduced(acc0, sq0), reduced(acc1, sq1)));
}
