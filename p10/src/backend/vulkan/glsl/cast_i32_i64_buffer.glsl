#version 450 core

layout(std430) buffer;

/*
 * Int32 to 64-bit codes over linear storage.  One invocation owns four
 * elements: 16 bytes of input, 32 of output, and each element is guarded on its
 * own because an allocation only rounds up to a multiple of 16 bytes while a
 * wide element at the end of a payload needs all eight of its bytes.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  uvec4 data[];
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
  const uint base = word << 2u;
  if (base >= uBlock.buf_length) {
    return;
  }
  for (uint i = 0u; i < 4u; ++i) {
    const uint e = base + i;
    if (e >= uBlock.buf_length) {
      return;
    }
    const uint code = uInput.data[word][i];
    // The high word repeats the sign bit, read off the pattern itself: no
    // numeric conversion of a possibly wrapped value has to be defined.
    uOutput.data[e] =
        uvec2(code, (code & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0u);
  }
}
