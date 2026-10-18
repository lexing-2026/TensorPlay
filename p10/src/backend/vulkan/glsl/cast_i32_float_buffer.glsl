#version 450 core

layout(std430) buffer;

/*
 * Int32 to single precision over linear storage.  Both payloads have 4-byte
 * elements, so one invocation converts one 16-byte word of four elements.
 * The guard is the logical element count: the last word of an odd-length
 * payload lands inside the allocation, which is rounded up to 16 bytes, and
 * nothing reads those lanes back.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  ivec4 data[];
}
uInput;
layout(set = 0, binding = 2) uniform highp restrict Block {
  uint buf_length;  // element count, shared by both payloads
  uint fill0;
  uint fill1;
  uint fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  if ((word << 2u) >= uBlock.buf_length) {
    return;
  }
  uOutput.data[word] = vec4(uInput.data[word]);
}
