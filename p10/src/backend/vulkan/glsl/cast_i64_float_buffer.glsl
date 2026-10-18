#version 450 core

layout(std430) buffer;

/*
 * 64-bit codes to single precision over linear storage.  One invocation owns
 * four elements: 32 bytes of input, 16 of output, and each element is guarded
 * on its own because an allocation only rounds up to a multiple of 16 bytes.
 *
 * The pair of words is read as a sign and a magnitude rather than as two
 * unsigned halves of a value: a negative code is the complement of its
 * magnitude plus one, so the two words can be combined in single precision
 * without anything ever holding a 64-bit integer, which no shader on this
 * backend can name.  A magnitude whose high word needs more than 24 bits
 * rounds to the nearest float, as any conversion into the destination
 * format must.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  vec4 data[];
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
    const uvec2 code = uInput.data[e];
    const bool negative = (code.y & 0x80000000u) != 0u;
    uint mag_lo = code.x;
    uint mag_hi = code.y;
    if (negative) {
      // The carry out of the low word only happens when the complemented low
      // word was all ones, that is when the code's own low word was zero.
      mag_lo = ~mag_lo + 1u;
      mag_hi = ~mag_hi + ((code.x == 0u) ? 1u : 0u);
    }
    const float magnitude = float(mag_lo) + float(mag_hi) * 4294967296.0;
    uOutput.data[word][i] = negative ? -magnitude : magnitude;
  }
}
