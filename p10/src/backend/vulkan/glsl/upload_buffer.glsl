#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Linear payload upload through the compute pipeline.  The host-written
// staging buffer is read as 16-byte groups and written straight into the
// tensor payload, so the payload reaches the device with the same pipeline
// stage as any other tensor write and later consumers order against it like
// against any other dispatch.  Linear allocations are 16-byte rounded, so a
// whole group always stays inside both the payload and the staging block.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec4 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uint buf_length; // number of 16-byte groups to move
  uint fill0;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  uOutput.data[idx] = uInput.data[idx];
}
