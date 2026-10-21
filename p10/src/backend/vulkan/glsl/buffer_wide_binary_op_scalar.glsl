#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;
#include "wide_ext.h"

$if not INPLACE:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    uvec2 data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
    uvec2 data[];
  }
  uInput;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    uint buf_length;
    uint fill0;
    // The scalar is a count, so it arrives as the two words a shader is
    // promised to hold.  The op's own scale factor is already folded into it.
    uint other_low;
    uint other_high;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uvec2 data[];
  }
  uOutput;
  layout(set = 0, binding = 1) uniform PRECISION restrict Block {
    uint buf_length;
    uint fill0;
    uint other_low;
    uint other_high;
  }
  uBlock;
// end-of-inplace-branches

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise binary op of a wide payload against one count: every element
 * meets the same pair of words.  The scalar already carries the op's scale
 * factor, so the operator's third argument is a count of one.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  const uvec2 other = uvec2(uBlock.other_low, uBlock.other_high);

  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = OP(uInput.data[idx], other, uvec2(1u, 0u));
  $else:
    uOutput.data[idx] = OP(uOutput.data[idx], other, uvec2(1u, 0u));
  // clang-format on
}
