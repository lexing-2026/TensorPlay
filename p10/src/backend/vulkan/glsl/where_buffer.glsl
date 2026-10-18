#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer where: condition, both operands and the output are plain
// row-major buffer payloads (the shapes that exceed the 3-D image axis
// limit).  The Bool condition is stored as one byte per element, read as
// packed words and tested lane by lane.  Selection is exact per lane so the
// rejected operand's infinities or not-a-numbers cannot poison the result
// through arithmetic recombination.
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
  uint buf_length; // element count
  uint fill0;      // std140 padding
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

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
  } else if (idx < nvec + rem) {
    const uint i = (nvec << 2u) + (idx - nvec);
    const uint j = i >> 2u;
    const uint k = i & 3u;
    const uint byte = (uCond.data[j] >> (8u * k)) & 0xffu;
    uOutput.data[j][k] = byte != 0u ? uInput.data[j][k] : uOther.data[j][k];
  }
}