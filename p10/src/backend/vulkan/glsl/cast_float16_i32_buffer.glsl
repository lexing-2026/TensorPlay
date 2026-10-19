#version 450 core

layout(std430) buffer;

/*
 * Half to signed word over linear storage.  The source keeps two lanes to a
 * 32-bit word and the destination four to a 16-byte one, so one invocation owns
 * one source word and writes the two destination lanes it covers.  A fraction
 * dropped toward zero is the rule a host cast follows; widening to single
 * precision first keeps the magnitude exact, since a half fits it whole.
 *
 * Buffer allocations round up to 16 bytes, so an odd-length tail lands inside
 * the allocation; the logical length stops before it and nothing reads it back.
 */
layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  ivec4 data[];
}
uOutput;

layout(set = 0, binding = 1) buffer highp restrict readonly InBuffer {
  uint data[];
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

  const vec2 pair = unpackHalf2x16(uInput.data[word]);
  uOutput.data[first >> 2u][first & 3u] = int(trunc(pair.x));

  const uint second = first + 1u;
  if (second < uBlock.buf_length) {
    uOutput.data[second >> 2u][second & 3u] = int(trunc(pair.y));
  }
}
