#version 450 core

layout(std430) buffer;

/*
 * Single-precision linear payload to a byte-packed one (Int8, UInt8 or Bool).
 * The destination keeps one byte per element, four lanes to a 32-bit word, so
 * one invocation reads one float word and writes one packed word.
 *
 * An integer destination wraps: the byte is trunc(value) taken modulo 256,
 * which is the two's-complement byte of the truncated value for either sign,
 * and the destination never has to represent a signed integer.  A Bool
 * destination instead stores 0 or 1.  The lane rule is a template parameter.
 */
#define LANE(V) ${LANE}

layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uint data[];
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
  const uint first = word << 2u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const vec4 value = uInput.data[word];
  uOutput.data[word] = LANE(value.x) | (LANE(value.y) << 8u) |
      (LANE(value.z) << 16u) | (LANE(value.w) << 24u);
}
