#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define DTYPE ${DTYPE}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

// Elementwise comparison of two linear payloads.  The Bool result is one byte
// per element, so four lanes travel packed in each 32-bit word: lane k of a
// word occupies bits [8k, 8k + 8).  Operands are 4-byte elements and are read
// as 16-byte vectors, which lines one word of the result up with one word of
// each operand.
//
// A dispatch covers ceil(numel / 4) words.  The last word may hold lanes past
// the logical end of the payload, but buffer allocations are rounded up to 16
// bytes, so those lanes are always inside the allocation; whatever they
// contain is never read back, because the logical length stops before them.
$if DTYPE == "int":
  #define VECTOR ivec4
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
  uint fill1;
  uint fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }

  const bvec4 hit = OP(uInput.data[idx], uOther.data[idx]);
  uOutput.data[idx] = (hit.x ? 1u : 0u) | (hit.y ? 256u : 0u) |
      (hit.z ? 65536u : 0u) | (hit.w ? 16777216u : 0u);
}
