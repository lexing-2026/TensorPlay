#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

#include "broadcast_index.h"

// Buffer-to-buffer where: condition, both operands and the output are plain
// row-major buffer payloads (the shapes that exceed the 3-D image axis
// limit).  The Bool condition is stored as one byte per element, read as
// packed words and tested lane by lane.  Selection is exact per lane so the
// rejected operand's infinities or not-a-numbers cannot poison the result
// through arithmetic recombination.
//
// The three planes either carry the output's shape, in which case one word of
// each feeds one word of the result, or they are addressed lane by lane
// through their own strides, so a singleton dimension repeats without the
// operand being materialized at the output shape.  Either way one invocation
// owns a whole destination word, which is what keeps two invocations from
// writing different lanes of the same word.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Condition {
  uint data[];
}
uCond;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Other {
  vec4 data[];
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

// The condition byte of one element, whichever word and lane it sits in.
bool cond_at(const uint i) {
  const uint word = uCond.data[i >> 2u];
  return ((word >> ((i & 3u) * 8u)) & 0xffu) != 0u;
}

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (uBlock.broadcast == 0) {
    if (idx < nvec) {
      const uint j = idx;
      const uint m = uCond.data[j];
      const vec4 a = uInput.data[j];
      const vec4 b = uOther.data[j];
      vec4 res;
      res.x = (m & 0xffu) != 0u ? a.x : b.x;
      res.y = ((m >> 8u) & 0xffu) != 0u ? a.y : b.y;
      res.z = ((m >> 16u) & 0xffu) != 0u ? a.z : b.z;
      res.w = ((m >> 24u) & 0xffu) != 0u ? a.w : b.w;
      uOutput.data[j] = res;
    } else if (idx == nvec) {
      // One invocation owns the whole partial word, so the dispatch stays at
      // div_up(n, 4) groups and tail coverage does not depend on how far the
      // workgroup count is rounded up.
      const uint j = nvec;
      for (uint i = 0u; i < rem; ++i) {
        const uint byte = (uCond.data[j] >> (8u * i)) & 0xffu;
        uOutput.data[j][i] = byte != 0u ? uInput.data[j][i] : uOther.data[j][i];
      }
    }
    return;
  }

  // Remapped lanes: the last invocation covers a partial word, as above, so
  // the grid stays at div_up(n, 4) whatever the operand shapes.
  const uint words = nvec + (rem != 0u ? 1u : 0u);
  if (idx >= words) {
    return;
  }
  const uint lanes = idx < nvec ? 4u : rem;
  for (uint k = 0u; k < lanes; ++k) {
    const uint i = idx * 4u + k;
    const uint ci = (uBlock.broadcast & 1) != 0
        ? remap_index(i, uBlock.out_strides, uBlock.cond_sizes,
                      uBlock.cond_strides)
        : i;
    const uint si = (uBlock.broadcast & 2) != 0
        ? remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                      uBlock.input_strides)
        : i;
    const uint oi = (uBlock.broadcast & 4) != 0
        ? remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                      uBlock.other_strides)
        : i;
    uOutput.data[idx][k] =
        cond_at(ci) ? uInput.data[si >> 2u][si & 3u]
                    : uOther.data[oi >> 2u][oi & 3u];
  }
}
