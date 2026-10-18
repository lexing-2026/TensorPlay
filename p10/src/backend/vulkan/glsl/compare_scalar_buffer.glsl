#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define DTYPE ${DTYPE}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

// Comparison of a linear payload against a scalar.  The Bool result packs four
// lanes per 32-bit word, lane k occupying bits [8k, 8k + 8), so one word of
// result lines up with one 16-byte vector of the operand.
//
// A dispatch covers ceil(numel / 4) words; the trailing word may reach past the
// logical end of the payload, which stays inside the allocation because buffer
// allocations are rounded up to 16 bytes, and those lanes are never read back.
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
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uint buf_length; // element count
  // The threshold arrives in the field matching the element type; the other
  // two are inert.
  float value_float;
  int value_int;
  uint fill0;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  if (idx >= ((uBlock.buf_length + 3u) >> 2u)) {
    return;
  }

  $if DTYPE == "int":
    const VECTOR threshold = ivec4(uBlock.value_int);
  $else:
    const VECTOR threshold = vec4(uBlock.value_float);

  const bvec4 hit = OP(uInput.data[idx], threshold);
  uOutput.data[idx] = (hit.x ? 1u : 0u) | (hit.y ? 256u : 0u) |
      (hit.z ? 65536u : 0u) | (hit.w ? 16777216u : 0u);
}
