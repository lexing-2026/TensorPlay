#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
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
    // The closed range, each side held as the pair of words a count is carried
    // in.  A side the caller left unset arrives as the extreme of the width,
    // which clips nothing.
    uint lo_low;
    uint lo_high;
    uint hi_low;
    uint hi_high;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    uvec2 data[];
  }
  uOutput;
  layout(set = 0, binding = 1) uniform PRECISION restrict Block {
    uint buf_length;
    uint lo_low;
    uint lo_high;
    uint hi_low;
    uint hi_high;
  }
  uBlock;
// end-of-inplace-branches

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Elementwise clip of a linear payload of codes wider than a word to a closed
 * range: one invocation owns one element, since an element fills the pair of
 * words a 16-byte slot carries four words of.  The greater of the element and
 * the low bound is settled word against word, and the lesser of that and the
 * high bound likewise, so a count outside the range lands on its edge.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= uBlock.buf_length) {
    return;
  }
  const uvec2 lo = uvec2(uBlock.lo_low, uBlock.lo_high);
  const uvec2 hi = uvec2(uBlock.hi_low, uBlock.hi_high);

  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = wide_min(wide_max(uInput.data[idx], lo), hi);
  $else:
    uOutput.data[idx] = wide_min(wide_max(uOutput.data[idx], lo), hi);
  // clang-format on
}
