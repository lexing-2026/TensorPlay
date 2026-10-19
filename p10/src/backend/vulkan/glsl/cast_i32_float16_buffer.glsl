#version 450 core

layout(std430) buffer;

/*
 * Signed word to half over linear storage.  The source keeps four elements to
 * a 16-byte word, the destination two to a 32-bit word, so one invocation owns
 * one destination word and gathers the two source elements it covers.  Each
 * widens to single precision, which holds a small code exactly, and quantizes
 * to the nearest half there, because the word pack rounds to nearest even while
 * leaving the value to the hardware would round however the device prefers.
 *
 * An odd element count leaves the high lane of the last word without a source.
 * Buffer allocations round up to 16 bytes, so writing it stays inside the
 * allocation and the logical length stops before anything reads it.
 */
#include "half_round.h"

layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uint data[];
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
  const uint first = word << 1u;
  if (first >= uBlock.buf_length) {
    return;
  }

  const float lo = round_to_half(float(uInput.data[first >> 2u][first & 3u]));
  const uint second = first + 1u;
  const float hi = second < uBlock.buf_length
      ? round_to_half(float(uInput.data[second >> 2u][second & 3u]))
      : 0.0;

  uOutput.data[word] = packHalf2x16(vec2(lo, hi));
}
