#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

#include "broadcast_index.h"

// Buffer-to-buffer where over half-precision payloads.  A half payload packs
// two lanes per 32-bit word and the Bool condition one byte per element, four
// lanes per word, so one condition word feeds two result words and which pair
// of bytes a result word draws on follows the half of the word it sits in.
// Selection moves the 16-bit fields themselves: nothing is widened and
// re-narrowed, so a rejected lane's infinity or not-a-number cannot reach the
// result.
//
// One invocation owns one whole result word, which is what keeps two
// invocations from writing different lanes of the same word.  An odd length
// leaves the second lane of the final word past the logical end; the read
// stays inside the allocation, because buffer sizes round up to 16 bytes, and
// no read-back reaches that lane.  When a plane carries the output's shape its
// word is read whole; otherwise each lane is remapped through its own strides,
// so a singleton dimension repeats without the plane being materialized at the
// output shape.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Condition {
  uint data[];
}
uCond;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Other {
  uint data[];
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
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  if (uBlock.broadcast == 0) {
    const uint flags = uCond.data[word >> 1u];
    // A condition byte occupies eight bits of its word, so a result word's two
    // bytes start at bit 0 or bit 16, whichever half of the word they sit in.
    const uint base = (word & 1u) * 16u;
    const bool first = ((flags >> base) & 0xffu) != 0u;
    const bool second = ((flags >> (base + 8u)) & 0xffu) != 0u;

    const uint a = uInput.data[word];
    const uint b = uOther.data[word];
    const uint low = first ? (a & 0xffffu) : (b & 0xffffu);
    const uint high = second ? (a >> 16u) : (b >> 16u);
    uOutput.data[word] = low | (high << 16u);
    return;
  }

  uint result = 0u;
  for (uint k = 0u; k < 2u; ++k) {
    const uint i = word * 2u + k;
    if (i >= n) {
      break;
    }
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
    // The 16-bit field sits in the word of the branch that wins, so its
    // position follows that branch's own element index.
    const bool take_self = cond_at(ci);
    const uint taken = take_self ? si : oi;
    const uint source =
        take_self ? uInput.data[taken >> 1u] : uOther.data[taken >> 1u];
    const uint field = (source >> ((taken & 1u) * 16u)) & 0xffffu;
    result |= k == 0u ? field : (field << 16u);
  }
  uOutput.data[word] = result;
}
