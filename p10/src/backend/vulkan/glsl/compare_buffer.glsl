#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define DTYPE ${DTYPE}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "broadcast_index.h"

// Elementwise comparison of two linear payloads.  The Bool result is one byte
// per element, so four lanes travel packed in each 32-bit word: lane k of a
// word occupies bits [8k, 8k + 8).  A single-precision operand is read as
// 16-byte vectors, which lines one word of the result up with one word of each
// operand; a half-precision operand packs two lanes per word, so one result
// word draws on two operand words.
//
// A dispatch covers ceil(numel / 4) words.  The last word may hold lanes past
// the logical end of the payload, but buffer allocations are rounded up to 16
// bytes, so those lanes are always inside the allocation; whatever they
// contain is never read back, because the logical length stops before them.
//
// With `broadcast` set (bit 0: the first operand, bit 1: the second), an
// element's flat index is recomputed from row-major strides, so a dimension of
// one repeats across the output range without the operand being materialized at
// it.  The remap is per element, which is also what lets an operand whose lanes
// sit at another word offset feed this word.
$if DTYPE == "int":
  #define VECTOR ivec4
$elif DTYPE == "half":
  #define VECTOR uint
$else:
  #define VECTOR vec4

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  VECTOR data[];
}
uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Other {
  VECTOR data[];
}
uOther;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  uint buf_length; // element count
  uint fill0;      // std140 padding
  int broadcast;   // bit 0: the first operand, bit 1: the second
  uint fill1;
  ivec4 out_strides; // row-major strides of the output shape
  ivec4 input_sizes; // each operand's own extents, with singletons
  ivec4 input_strides;
  ivec4 other_sizes;
  ivec4 other_strides;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if DTYPE == "half":
  // One element of a half payload is half a word: the word holds it and its
  // neighbour, and the element's low bit says which of the two.
  uint word_of(const uint element) {
    return element >> 1u;
  }
  float lane_of(const uint word, const uint element) {
    const vec2 pair = unpackHalf2x16(word);
    return (element & 1u) == 0u ? pair.x : pair.y;
  }
$elif DTYPE == "int":
  int lane_of(const ivec4 lanes, const uint k) {
    return k == 0u ? lanes.x : (k == 1u ? lanes.y : (k == 2u ? lanes.z : lanes.w));
  }
$else:
  float lane_of(const vec4 lanes, const uint k) {
    return k == 0u ? lanes.x : (k == 1u ? lanes.y : (k == 2u ? lanes.z : lanes.w));
  }
// End of the lane readers.

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }

  bvec4 hit;

$if DTYPE == "half":
  if (uBlock.broadcast == 0) {
    // Half lanes widen to single precision before comparing: the mapping is
    // exact and order preserving, so a float comparison decides exactly what a
    // comparison in the narrow type would.  One invocation owns a whole result
    // word, so no two invocations can race on the same destination.
    const vec2 a01 = unpackHalf2x16(uInput.data[idx * 2u]);
    const vec2 a23 = unpackHalf2x16(uInput.data[idx * 2u + 1u]);
    const vec2 b01 = unpackHalf2x16(uOther.data[idx * 2u]);
    const vec2 b23 = unpackHalf2x16(uOther.data[idx * 2u + 1u]);
    hit = OP(vec4(a01.x, a01.y, a23.x, a23.y), vec4(b01.x, b01.y, b23.x, b23.y));
  } else {
    vec4 a = vec4(0.0);
    vec4 b = vec4(0.0);
    for (uint k = 0u; k < 4u; ++k) {
      const uint i = idx * 4u + k;
      if (i >= uBlock.buf_length) {
        break;
      }
      const uint ai = (uBlock.broadcast & 1) != 0
          ? remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                        uBlock.input_strides)
          : i;
      const uint bi = (uBlock.broadcast & 2) != 0
          ? remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                        uBlock.other_strides)
          : i;
      a[k] = lane_of(uInput.data[word_of(ai)], ai);
      b[k] = lane_of(uOther.data[word_of(bi)], bi);
    }
    hit = OP(a, b);
  }
$else:
  if (uBlock.broadcast == 0) {
    hit = OP(uInput.data[idx], uOther.data[idx]);
  } else {
    VECTOR a = VECTOR(0);
    VECTOR b = VECTOR(0);
    for (uint k = 0u; k < 4u; ++k) {
      const uint i = idx * 4u + k;
      if (i >= uBlock.buf_length) {
        break;
      }
      const uint ai = (uBlock.broadcast & 1) != 0
          ? remap_index(i, uBlock.out_strides, uBlock.input_sizes,
                        uBlock.input_strides)
          : i;
      const uint bi = (uBlock.broadcast & 2) != 0
          ? remap_index(i, uBlock.out_strides, uBlock.other_sizes,
                        uBlock.other_strides)
          : i;
      a[k] = lane_of(uInput.data[ai >> 2u], ai & 3u);
      b[k] = lane_of(uOther.data[bi >> 2u], bi & 3u);
    }
    hit = OP(a, b);
  }
// End of the operand loads.

  uOutput.data[idx] = (hit.x ? 1u : 0u) | (hit.y ? 256u : 0u) |
      (hit.z ? 65536u : 0u) | (hit.w ? 16777216u : 0u);
}
