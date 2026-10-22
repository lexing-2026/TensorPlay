#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Inclusive log-sum-exp scan along one axis over buffer storage.  Each row
// decomposes into an outer block and an inner run; one invocation walks the
// axis once and writes the running value at every position.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  float data[];
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

  // The accumulator starts below every finite value, so the first element
  // stands alone whatever it is; two equal infinities answer with the
  // incoming element, and an accumulator that is not a number stays one.
  const float neg_inf = uintBitsToFloat(0xff800000u);
  float acc = neg_inf;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    const float x = uInput.data[index];
    if (isnan(acc) || isnan(x)) {
      // A non-number on either side makes the sum one, so the running value
      // becomes one too and every later position of the row carries it.
      acc = acc + x;
    } else {
      const float lo = min(acc, x);
      const float hi = max(acc, x);
      acc = (lo != hi || !isinf(lo)) ? log1p_of(exp(lo - hi)) + hi : x;
    }
    uOutput.data[index] = acc;
  }
}
