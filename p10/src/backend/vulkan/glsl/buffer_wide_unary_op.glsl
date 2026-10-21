#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define OP(X) ${OPERATOR}
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
  layout(set = 0, binding = 2) uniform PRECISION Block {
    uint buf_length;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uvec2 data[];
  }
  uOutput;
  // Every variant's bindings number contiguously from zero: the descriptor set
  // layout is rebuilt from the ordered binding list, so a reserved slot for the
  // absent input buffer would shift the uniform block out of the position the
  // arguments are bound to.
  layout(set = 0, binding = 1) uniform PRECISION Block {
    uint buf_length;
  }
  uBlock;
// end-of-inplace-branches

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise unary operation over a linear payload of codes wider than a
 * word: out = OP(in).  One invocation owns one element, since an element fills
 * the pair of words a 16-byte slot carries four words of.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = OP(uInput.data[idx]);
  $else:
    uOutput.data[idx] = OP(uOutput.data[idx]);
  // clang-format on
}
