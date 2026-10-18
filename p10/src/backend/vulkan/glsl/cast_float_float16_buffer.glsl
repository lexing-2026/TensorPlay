#version 450 core

layout(std430) buffer;

/*
 * Single precision to a half-precision linear payload.  The destination keeps
 * two lanes per 32-bit word, so one invocation owns one destination word and
 * reads the two source elements it covers; the rounding happens in the pack,
 * the same single rounding the rgba16f texture builds get from the hardware
 * format conversion.
 *
 * An odd element count leaves the high lane of the last word without a source.
 * Buffer allocations round up to 16 bytes, so writing it stays inside the
 * allocation and the logical length stops before anything reads it.
 */
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
  const uint first = word << 1u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const float lo = uInput.data[first >> 2u][first & 3u];
  const uint second = first + 1u;
  const float hi =
      second < uBlock.buf_length ? uInput.data[second >> 2u][second & 3u] : 0.0;

  uOutput.data[word] = packHalf2x16(vec2(lo, hi));
}
