#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Cumulative sum along one axis over buffer storage, for shapes whose
// extents exceed the 3-D image axis limit.  Each row decomposes into an
// outer block and an inner run; one invocation walks the axis once and
// writes the running total at every position.
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
  for (int i = 0; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    acc += uInput.data[index];
    uOutput.data[index] = acc;
  }
}