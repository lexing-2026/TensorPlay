#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Half variant of the linear log-sum-exp scan: the payload packs two 16-bit
// lanes per word, and the two lanes of a word sit on neighbouring inner
// positions, which belong to different scan rows.  The scan therefore runs
// twice: a pass accumulates every row as usual but stores only the fields
// whose flat index falls in that pass's word lane, so each destination word is
// touched by exactly one invocation per pass.  The accumulator stays in single
// precision, which is the width the closed form is evaluated in.
layout(set = 0, binding = 0) buffer PRECISION restrict Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Length of the scanned axis
  int axis_size;
  // Element count of the trailing dimensions
  int inner;
  // Scan rows, one per (leading block, inner position) pair
  int rows;
  // Word lane this pass owns: 0 fills the low field, 1 the high one
  int word_lane;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// ln(1 + t) for t in (0, 1].  The library offers no direct form, and adding
// one before the logarithm discards every bit of a small t, so the series
// carries the small range and the direct form handles the rest.
float log1p_of(float t) {
  if (t < 0.0625) {
    return t * (1.0 - t * (0.5 - t * (0.33333334 - t * (0.25 -
        t * (0.2 - t * 0.16666667)))));
  }
  return log(1.0 + t);
}

void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);

  const float neg_inf = uintBitsToFloat(0xff800000u);
  float acc = neg_inf;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    const uint index = uint(base + i * uBlock.inner + inner_pos);
    const float x = unpackHalf2x16(uInput.data[index >> 1u])[index & 1u];
    if (isnan(acc) || isnan(x)) {
      // A non-number on either side makes the sum one, so the running value
      // becomes one too and every later position of the row carries it.
      acc = acc + x;
    } else {
      const float lo = min(acc, x);
      const float hi = max(acc, x);
      acc = (lo != hi || !isinf(lo)) ? log1p_of(exp(lo - hi)) + hi : x;
    }

    if (int(index & 1u) != uBlock.word_lane) {
      continue;
    }
    // Read-modify-write keeps the neighbouring lane: the other pass owns it.
    const uint shift = (index & 1u) * 16u;
    const uint word = uOutput.data[index >> 1u];
    uOutput.data[index >> 1u] = (word & ~(0xffffu << shift)) |
        ((packHalf2x16(vec2(acc, 0.0)) & 0xffffu) << shift);
  }
}
