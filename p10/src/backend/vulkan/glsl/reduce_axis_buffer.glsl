#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define MODE ${MODE}
// clang-format on

layout(std430) buffer;

// Single-axis reduction over buffer storage, for shapes whose extents
// exceed the 3-D image axis limit.  Each row decomposes into an outer
// block and an inner run; one invocation accumulates the line sum and
// mean square, then emits either the scaled sum (MODE 0) or the variance
// with the caller's correction (MODE 1).
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
  int correction;
  float scale;
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

  float acc = 0.0;
  float sq = 0.0;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    const float v = uInput.data[base + i * uBlock.inner + inner_pos];
    acc += v;
    sq += v * v;
  }

  // clang-format off
  $if MODE == 1:
    const float mean = acc / float(uBlock.axis_size);
    uOutput.data[row] =
        (sq / float(uBlock.axis_size) - mean * mean) *
        float(uBlock.axis_size) /
        float(max(uBlock.axis_size - uBlock.correction, 1));
  $else:
    uOutput.data[row] = acc * uBlock.scale;
  // clang-format on
}