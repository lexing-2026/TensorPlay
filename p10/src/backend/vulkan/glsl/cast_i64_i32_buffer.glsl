#version 450 core

layout(std430) buffer;

/*
 * 64-bit codes to Int32 over linear storage.  A wide element occupies a pair of
 * words, so one invocation owns four elements: 32 bytes of input, 16 of output.
 * Each element is guarded on its own, because an allocation only rounds up to a
 * multiple of 16 bytes and a wide payload can end exactly on that boundary.
 * The high word is dropped and the low word kept, which is how a narrowing cast
 * wraps.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uvec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  uvec2 data[];
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
    uOutput.data[word][i] = uInput.data[e].x;
  }
}
