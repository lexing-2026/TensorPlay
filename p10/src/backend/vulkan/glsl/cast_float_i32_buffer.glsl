#version 450 core

layout(std430) buffer;

/*
 * Single precision to Int32 over linear storage: one invocation converts one
 * 16-byte word of four elements, truncating toward zero, which is the rule a
 * host cast follows.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  ivec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  vec4 data[];
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
  uOutput.data[word] = ivec4(trunc(uInput.data[word]));
}
