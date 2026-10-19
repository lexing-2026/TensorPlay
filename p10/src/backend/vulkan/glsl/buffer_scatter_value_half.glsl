#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Apply pass of a scatter over linear half storage: one invocation owns one
// whole 32-bit word, which carries two lanes, and rewrites it in place.  The
// number arrives as a single-precision uniform and is rounded to a half field
// once, then moved into the lane a mark names as 16 bits, so the destination
// keeps the one rounding its own storage asks for instead of widening and
// narrowing the neighbours.
layout(set = 0, binding = 0) buffer PRECISION restrict InOut {
  uint data[];
}
uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int count;    // elements of the destination
  float value;  // the number written into every marked lane
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint n = uint(uBlock.count);
  if (word >= ((n + 1u) >> 1u)) {
    return;
  }

  const uint field = packHalf2x16(vec2(uBlock.value, 0.0)) & 0xffffu;
  uint packed = uSelf.data[word];
  bool touched = false;
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
    if (element >= n) {
      break;
    }
    if (uMarks.data[element] > 0) {
      const uint shift = 16u * lane;
      packed = (packed & ~(0xffffu << shift)) | (field << shift);
      touched = true;
    }
  }
  if (touched) {
    uSelf.data[word] = packed;
  }
}
