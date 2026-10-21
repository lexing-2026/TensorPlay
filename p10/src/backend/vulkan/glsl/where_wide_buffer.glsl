#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

#include "broadcast_index.h"

/*
 * Buffer-to-buffer selection between payloads whose elements are wider than a
 * word.  A count that needs more than 32 bits has no texel form, so both
 * branches and the result are linear memory holding one pair of words per
 * element, low word first, and one invocation owns one element.
 *
 * The condition is the usual Bool payload: one byte per element, four lanes to
 * a word.  Its word is shared by four elements, so invocations read it
 * concurrently and never write it.
 *
 * A plane that is not already at the output shape is addressed element by
 * element through its own strides, so a singleton dimension repeats without the
 * branch being materialized at the output shape.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Condition {
  uint data[];
}
uCond;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Input {
  uvec2 data[];
}
uInput;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Other {
  uvec2 data[];
}
uOther;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  uint buf_length; // element count of the output
  uint fill0; // std140 padding
  int broadcast; // bit 0: the condition, bit 1: the first branch, bit 2: the second
  uint fill1; // std140 padding
  ivec4 out_strides; // row-major strides of the output shape
  ivec4 cond_sizes; // each plane's extents right-aligned to that rank
  ivec4 cond_strides;
  ivec4 input_sizes;
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  const uint ci = (uBlock.broadcast & 1) != 0
      ? remap_index(idx, uBlock.out_strides, uBlock.cond_sizes,
                    uBlock.cond_strides)
      : idx;
  const uint si = (uBlock.broadcast & 2) != 0
      ? remap_index(idx, uBlock.out_strides, uBlock.input_sizes,
                    uBlock.input_strides)
      : idx;
  const uint oi = (uBlock.broadcast & 4) != 0
      ? remap_index(idx, uBlock.out_strides, uBlock.other_sizes,
                    uBlock.other_strides)
      : idx;

  const uint word = uCond.data[ci >> 2u];
  const bool held = ((word >> ((ci & 3u) * 8u)) & 0xffu) != 0u;
  uOutput.data[idx] = held ? uInput.data[si] : uOther.data[oi];
}
