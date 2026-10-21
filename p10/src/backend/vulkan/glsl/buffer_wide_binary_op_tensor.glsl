#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;
#include "wide_ext.h"

$if not INPLACE:
  // A wide element owns a pair of words, so the planes are read as pairs and
  // one element sits in every slot.
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    uvec2 data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
    uvec2 data[];
  }
  uInput;
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
    uvec2 data[];
  }
  uOther;
  layout(set = 0, binding = 3) uniform PRECISION restrict Block {
    uint buf_length;
    uint fill0;
    // The scale factor of a count is a count too, so it arrives as its two
    // words rather than a register no shader is promised.
    uint alpha_low;
    uint alpha_high;
    int broadcast;
    int fill1;
    int fill2;
    int fill3;
    ivec4 out_strides;
    ivec4 input_sizes;
    ivec4 input_strides;
    ivec4 other_sizes;
    ivec4 other_strides;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uvec2 data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer2 {
    uvec2 data[];
  }
  uOther;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    uint buf_length;
    uint fill0;
    uint alpha_low;
    uint alpha_high;
    int broadcast;
    int fill1;
    int fill2;
    int fill3;
    ivec4 out_strides;
    ivec4 input_sizes;
    ivec4 input_strides;
    ivec4 other_sizes;
    ivec4 other_strides;
  }
  uBlock;
// end-of-inplace-branches

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise binary operation over two linear payloads of codes wider than a
 * word: out = OP(in, other, alpha).  One invocation owns one element, because
 * an element already fills the pair of words a 16-byte slot carries four of.
 *
 * `broadcast` is a bitmask of the operands whose flat index has to be
 * recomputed from the output (bit 0: input, bit 1: other); the remap walks the
 * row-major strides of the output and skips a dimension of one, so a singleton
 * operand repeats without being materialized.  Strides count elements and an
 * element is one whole pair here, so no lane masking appears.
 */
uint remap_index(
    uint i, ivec4 out_strides, ivec4 sizes, ivec4 strides) {
  uint rem = i;
  uint acc = 0u;
  for (int d = 0; d < 4; ++d) {
    const uint s = uint(out_strides[d]);
    if (s == 0u) {
      break;
    }
    const uint coord = rem / s;
    rem = rem % s;
    if (uint(sizes[d]) > 1u) {
      acc += coord * uint(strides[d]);
    }
  }
  return acc;
}

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  const uvec2 alpha = uvec2(uBlock.alpha_low, uBlock.alpha_high);

  uint a = idx;
  uint b = idx;
  if ((uBlock.broadcast & 1) != 0) {
    a = remap_index(idx, uBlock.out_strides, uBlock.input_sizes,
                    uBlock.input_strides);
  }
  if ((uBlock.broadcast & 2) != 0) {
    b = remap_index(idx, uBlock.out_strides, uBlock.other_sizes,
                    uBlock.other_strides);
  }

  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = OP(uInput.data[a], uOther.data[b], alpha);
  $else:
    // The destination keeps its own slot; a broadcast partner reads through
    // the remapped index.
    uOutput.data[idx] = OP(uOutput.data[a], uOther.data[b], alpha);
  // clang-format on
}
